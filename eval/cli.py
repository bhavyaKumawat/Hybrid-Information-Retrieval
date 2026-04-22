"""Typer CLI for running ablations and producing reports.

Commands:

* ``ir3-eval list``      — print every known ablation + config.
* ``ir3-eval run``       — run one ablation (or one config, via ``--tag``).
* ``ir3-eval run-all``   — run every config across all six ablations.
* ``ir3-eval score``     — re-score existing run JSONs, no retrieval.
* ``ir3-eval report``    — build ``summary.csv`` + ``summary.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from retrieval.config import get_settings
from retrieval.qdrant_store import build_client

from .configs import (
    ABLATION_REGISTRY,
    RetrievalConfig,
    all_ablations,
)
from .qrels import load_nfcorpus_eval
from .report import collect_run_metrics, write_csv, write_markdown
from .runner import run_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
# Quiet the per-request HTTP transport chatter from qdrant_client; otherwise
# every batched upsert prints an INFO line and drowns the rich progress bar.
for _noisy in ("httpx", "httpcore", "httpcore.http11", "httpcore.connection"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = typer.Typer(add_completion=False, help="IR3 — Offline evaluation harness.")
console = Console()

RUNS_DIR = Path("eval/runs")
REPORTS_DIR = Path("eval/reports")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command("list")
def list_configs(
    ablation: str | None = typer.Option(
        None, "--ablation", "-a", help="One of 1..6 or the ablation name."
    ),
) -> None:
    """Enumerate every config (or just one ablation's) that would run."""
    configs = _resolve_configs(ablation)
    table = Table(title=f"{len(configs)} configs")
    table.add_column("tag")
    table.add_column("ablation")
    table.add_column("dense")
    table.add_column("chunker")
    table.add_column("weights (d/s)")
    table.add_column("rerank")
    table.add_column("collection")
    for c in configs:
        table.add_row(
            c.tag,
            c.ablation,
            c.dense_tag,
            c.chunker_tag,
            f"{c.weight_dense}/{c.weight_sparse}",
            (c.reranker_model or "none") if c.use_reranker else "none",
            c.collection_name,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# run / run-all
# ---------------------------------------------------------------------------


@app.command("run")
def run_cmd(
    ablation: str = typer.Argument(..., help="Ablation id (1..6) or name."),
    tag: str | None = typer.Option(
        None, "--tag", "-t", help="Only run the config with this tag."
    ),
    best_embedder: str = typer.Option(
        "BAAI/bge-small-en-v1.5",
        "--best-embedder",
        help="Used by ablations that depend on 'the best embedder from ablation 2'.",
    ),
    skip_ingest: bool = typer.Option(
        False, "--skip-ingest", help="Don't touch the collection; assume it's built."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-run even if the run JSON already exists."
    ),
) -> None:
    """Execute all configs in one ablation (or one tagged config)."""
    configs = _configs_for_ablation(ablation, best_embedder=best_embedder)
    if tag:
        configs = [c for c in configs if c.tag == tag]
        if not configs:
            raise typer.BadParameter(f"No config with tag={tag!r} in ablation {ablation}")
    _execute(configs, skip_ingest=skip_ingest, overwrite=overwrite)


@app.command("run-all")
def run_all_cmd(
    skip_ingest: bool = typer.Option(
        False, "--skip-ingest", help="Don't touch collections; assume they're built."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-run even if the run JSON already exists."
    ),
) -> None:
    """Run every configured ablation, in order 1..6."""
    _execute(all_ablations(), skip_ingest=skip_ingest, overwrite=overwrite)


# ---------------------------------------------------------------------------
# score / report
# ---------------------------------------------------------------------------


@app.command("score")
def score_cmd(
    split: str = typer.Option("test", "--split", help="NFCorpus qrels split."),
) -> None:
    """Re-score every cached run JSON against the BEIR qrels. Prints a quick table."""
    data = load_nfcorpus_eval(split=split).filter_to_qrels()
    rows = collect_run_metrics(RUNS_DIR, qrels=data.qrels)
    if not rows:
        console.print(f"[yellow]No runs found in {RUNS_DIR}[/yellow]")
        return
    table = Table(title=f"Run metrics (n={len(rows)})")
    table.add_column("tag")
    for mk in ("precision@5", "recall@5", "ndcg@5", "precision@10", "recall@10", "ndcg@10"):
        table.add_column(mk)
    rows_sorted = sorted(rows, key=lambda r: r.get("ndcg@10", 0.0), reverse=True)
    for r in rows_sorted:
        table.add_row(
            r["tag"],
            f"{r.get('precision@5', 0):.4f}",
            f"{r.get('recall@5', 0):.4f}",
            f"{r.get('ndcg@5', 0):.4f}",
            f"{r.get('precision@10', 0):.4f}",
            f"{r.get('recall@10', 0):.4f}",
            f"{r.get('ndcg@10', 0):.4f}",
        )
    console.print(table)


@app.command("report")
def report_cmd(
    split: str = typer.Option("test", "--split", help="NFCorpus qrels split."),
) -> None:
    """Write ``eval/reports/summary.{csv,md}`` from the cached run JSONs."""
    data = load_nfcorpus_eval(split=split).filter_to_qrels()
    rows = collect_run_metrics(RUNS_DIR, qrels=data.qrels)
    if not rows:
        raise typer.Exit(f"No runs found in {RUNS_DIR}; run `ir3-eval run` first.")
    csv_path = write_csv(rows, REPORTS_DIR / "summary.csv")
    md_path = write_markdown(rows, REPORTS_DIR / "summary.md")
    console.print(f"[green]wrote[/green] {csv_path}")
    console.print(f"[green]wrote[/green] {md_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_configs(ablation: str | None) -> list[RetrievalConfig]:
    if ablation is None:
        return all_ablations()
    return _configs_for_ablation(ablation)


def _configs_for_ablation(ablation: str, best_embedder: str | None = None) -> list[RetrievalConfig]:
    key = ablation.lower()
    if key not in ABLATION_REGISTRY:
        raise typer.BadParameter(
            f"Unknown ablation {ablation!r}. Known: {sorted(set(ABLATION_REGISTRY.keys()))}"
        )
    fn = ABLATION_REGISTRY[key]
    # Only ablation_3 takes the best_embedder hint, but passing it to
    # others would be a TypeError — sniff the signature by name.
    if fn.__name__ == "ablation_3_chunker" and best_embedder:
        return fn(best_embedder=best_embedder)
    return fn()


def _execute(
    configs: list[RetrievalConfig],
    *,
    skip_ingest: bool,
    overwrite: bool,
) -> None:
    data = load_nfcorpus_eval(split="test").filter_to_qrels()
    base = get_settings()

    # Group configs by collection so we only build/open the client once per
    # collection. Within a group, ingestion runs at most once.
    by_collection: dict[str, list[RetrievalConfig]] = {}
    for c in configs:
        by_collection.setdefault(c.collection_name, []).append(c)

    console.print(
        f"[bold]Running[/bold] {len(configs)} config(s) across "
        f"{len(by_collection)} collection(s) over {len(data.queries)} queries."
    )

    for collection, cfgs in by_collection.items():
        console.rule(f"[bold cyan]{collection}[/bold cyan] — {len(cfgs)} config(s)")
        client = build_client(base)
        try:
            for i, cfg in enumerate(cfgs):
                run_config(
                    config=cfg,
                    base=base,
                    eval_data=data,
                    runs_dir=RUNS_DIR,
                    client=client,
                    console=console,
                    # Only ingest on the first config for each collection
                    # (subsequent configs in the same group reuse the same
                    # Qdrant collection — ingest would be a no-op anyway).
                    skip_ingest=skip_ingest or i > 0,
                    overwrite=overwrite,
                )
        finally:
            client.close()


if __name__ == "__main__":
    app()
