"""Typer CLI for the latency benchmark.

Commands:

* ``ir3-bench run``     — run the cold/warm benchmark and write a report.
* ``ir3-bench render``  — re-render console + markdown from a saved JSON.

Most knobs mirror :class:`retrieval.config.Settings`; only the
benchmark-only ones (``--warmup``, ``--warm-iters``, ``--n-queries``)
are genuinely new. When a retrieval knob isn't passed, we fall back to
whatever ``.env`` says.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from retrieval.config import get_settings
from retrieval.qdrant_store import build_client

from .latency import BenchConfig, BenchReport, StageStats, run_benchmark
from .report import print_console, write_json, write_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
for _noisy in ("httpx", "httpcore", "httpcore.http11", "httpcore.connection"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = typer.Typer(add_completion=False, help="IR3 — Retrieval latency benchmark.")
console = Console()

REPORTS_DIR = Path("bench/reports")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command("run")
def run_cmd(
    n_queries: int = typer.Option(
        50, "--n-queries", "-n",
        help="Number of distinct queries to sample (from NFCorpus test split unless --queries-file is given).",
    ),
    queries_file: Path | None = typer.Option(
        None, "--queries-file",
        help="Optional newline-delimited queries file. Overrides NFCorpus sampling.",
    ),
    warmup: int = typer.Option(5, "--warmup", help="Warmup queries run before the cold phase (not measured)."),
    warm_iters: int = typer.Option(
        10, "--warm-iters",
        help="Additional iterations per query after the cold hit — feeds the warm distribution.",
    ),
    collection: str | None = typer.Option(None, "--collection", help="Qdrant collection to benchmark. Default: from .env."),
    dense_model: str | None = typer.Option(None, "--dense-model"),
    sparse_model: str | None = typer.Option(None, "--sparse-model"),
    reranker_model: str | None = typer.Option(None, "--reranker-model"),
    use_reranker: bool = typer.Option(
        True, "--use-reranker/--no-reranker",
        help="Include (or skip) the cross-encoder rerank stage.",
    ),
    weight_dense: float | None = typer.Option(None, "--weight-dense"),
    weight_sparse: float | None = typer.Option(None, "--weight-sparse"),
    top_k: int | None = typer.Option(None, "--top-k"),
    rerank_top_n: int | None = typer.Option(None, "--rerank-top-n"),
    prefetch_limit: int | None = typer.Option(None, "--prefetch-limit"),
    seed: int = typer.Option(0, "--seed", help="RNG seed for query shuffling."),
    output: Path = typer.Option(
        REPORTS_DIR / "latency.json", "--output", "-o",
        help="Where to write the JSON report. A sibling .md is written alongside.",
    ),
) -> None:
    """Run the cold/warm latency benchmark end-to-end."""
    base = get_settings()

    cfg = BenchConfig(
        collection=collection or base.qdrant_collection,
        dense_model=dense_model or base.dense_model,
        sparse_model=sparse_model or base.sparse_model,
        reranker_model=reranker_model or base.reranker_model,
        use_reranker=use_reranker,
        weight_dense=base.weight_dense if weight_dense is None else weight_dense,
        weight_sparse=base.weight_sparse if weight_sparse is None else weight_sparse,
        rrf_k=base.rrf_k,
        top_k=top_k or base.top_k,
        rerank_top_n=rerank_top_n or base.rerank_top_n,
        prefetch_limit=prefetch_limit or base.prefetch_limit,
        warmup=warmup,
        warm_iters=warm_iters,
        seed=seed,
    )

    queries = _load_queries(queries_file, n_queries=n_queries, seed=seed)
    if len(queries) <= cfg.warmup:
        raise typer.BadParameter(
            f"Only {len(queries)} queries available; need more than warmup={cfg.warmup}."
        )

    console.print(
        f"[bold]Latency bench[/bold] collection=[cyan]{cfg.collection}[/cyan] "
        f"queries={len(queries)} warmup={cfg.warmup} warm_iters={cfg.warm_iters} "
        f"rerank={'[green]on[/green]' if cfg.use_reranker else '[yellow]off[/yellow]'}"
    )

    client = build_client(base)
    try:
        # Total work is warmup + cold + warm_iters * cold_count. We don't
        # know warmup exactly (clamped in run_benchmark) so we over-estimate;
        # Progress doesn't complain if advance() doesn't reach total.
        cold_count = len(queries) - min(cfg.warmup, len(queries) - 1)
        total = cfg.warmup + cold_count * (1 + cfg.warm_iters)

        with Progress(
            TextColumn("[bold blue]{task.fields[phase]}[/bold blue]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("bench", total=total, phase="warmup")

            def _cb(phase: str, done: int, _total: int) -> None:
                progress.update(task, completed=done, phase=phase)

            report = run_benchmark(
                client=client,
                settings=base,
                queries=queries,
                cfg=cfg,
                progress_cb=_cb,
            )
    finally:
        client.close()

    json_path = write_json(report, output)
    md_path = write_markdown(report, output.with_suffix(".md"))

    print_console(report, console=console)
    console.print(f"[green]wrote[/green] {json_path}")
    console.print(f"[green]wrote[/green] {md_path}")


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


@app.command("render")
def render_cmd(
    input: Path = typer.Argument(..., help="Path to a bench JSON produced by `ir3-bench run`."),
) -> None:
    """Re-print the console table + refresh the markdown from a saved JSON.

    Cheap to iterate on formatting without re-running the benchmark.
    """
    if not input.exists():
        raise typer.BadParameter(f"No such file: {input}")
    raw = json.loads(input.read_text())
    report = _report_from_dict(raw)
    print_console(report, console=console)
    md_path = write_markdown(report, input.with_suffix(".md"))
    console.print(f"[green]wrote[/green] {md_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_queries(path: Path | None, *, n_queries: int, seed: int) -> list[str]:
    if path is not None:
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if not lines:
            raise typer.BadParameter(f"Queries file {path} is empty.")
        return lines[:n_queries] if n_queries > 0 else lines

    # Default: NFCorpus test queries. This matches the corpus the ingestion
    # pipeline loads, so the bench hits realistic payloads. Import lazy to
    # keep `--help` fast.
    from eval.qrels import load_nfcorpus_eval

    data = load_nfcorpus_eval(split="test").filter_to_qrels()
    pool = list(data.queries.values())
    if not pool:
        raise typer.BadParameter("NFCorpus test split returned zero queries.")
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n_queries] if n_queries > 0 else pool


def _report_from_dict(raw: dict) -> BenchReport:
    """Reverse of :meth:`BenchReport.to_dict` — just enough to re-render."""
    from .latency import BenchConfig, QueryTiming

    cfg_raw = raw["config"]
    cfg = BenchConfig(
        collection=cfg_raw["collection"],
        dense_model=cfg_raw["dense_model"],
        sparse_model=cfg_raw["sparse_model"],
        reranker_model=cfg_raw["reranker_model"] or "",
        use_reranker=cfg_raw["use_reranker"],
        weight_dense=cfg_raw["weight_dense"],
        weight_sparse=cfg_raw["weight_sparse"],
        rrf_k=cfg_raw["rrf_k"],
        top_k=cfg_raw["top_k"],
        rerank_top_n=cfg_raw["rerank_top_n"],
        prefetch_limit=cfg_raw["prefetch_limit"],
        warmup=cfg_raw["warmup"],
        warm_iters=cfg_raw["warm_iters"],
        seed=cfg_raw["seed"],
    )
    timings = [
        QueryTiming(
            query=t["query"],
            iteration=t["iteration"],
            regime=t["regime"],
            stages=t["stages_ms"],
        )
        for t in raw.get("timings", [])
    ]

    def _stats(d: dict) -> dict[str, StageStats]:
        return {
            k: StageStats(
                n=v["n"],
                mean_ms=v["mean_ms"],
                stdev_ms=v["stdev_ms"],
                min_ms=v["min_ms"],
                p50_ms=v["p50_ms"],
                p95_ms=v["p95_ms"],
                p99_ms=v["p99_ms"],
                max_ms=v["max_ms"],
            )
            for k, v in d.items()
        }

    return BenchReport(
        config=cfg,
        num_queries=raw["num_queries"],
        timings=timings,
        cold_stats=_stats(raw["cold"]),
        warm_stats=_stats(raw["warm"]),
    )


if __name__ == "__main__":
    app()
