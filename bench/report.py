"""Pretty-print + persist a :class:`bench.latency.BenchReport`.

Three outputs are produced for every run:

* ``latency.json``        — full, machine-readable report (raw samples + stats).
* ``latency_summary.md``  — human-readable breakdown table (cold vs warm).
* a Rich console rendering — printed inline, same content as the markdown.

Keeping rendering separate from the measurement loop lets callers re-run
``ir3-bench render`` over a saved JSON without re-executing any queries.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .latency import STAGES, BenchReport, StageStats


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_json(report: BenchReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=False))
    return path


def write_markdown(report: BenchReport, path: Path) -> Path:
    """Dump a summary table + breakdown chart as markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg = report.config
    lines: list[str] = []
    lines.append("# Retrieval latency report")
    lines.append("")
    lines.append(f"- collection: `{cfg.collection}`")
    lines.append(f"- dense model: `{cfg.dense_model}`")
    lines.append(f"- sparse model: `{cfg.sparse_model}`")
    lines.append(
        f"- reranker: `{cfg.reranker_model}`" if cfg.use_reranker else "- reranker: _disabled_"
    )
    lines.append(
        f"- weights: dense={cfg.weight_dense}, sparse={cfg.weight_sparse}, rrf_k={cfg.rrf_k}"
    )
    lines.append(
        f"- top_k={cfg.top_k}, rerank_top_n={cfg.rerank_top_n}, prefetch_limit={cfg.prefetch_limit}"
    )
    lines.append(
        f"- queries={report.num_queries}, warmup={cfg.warmup}, warm_iters={cfg.warm_iters}"
    )
    lines.append("")

    lines.append("## Per-stage latency (ms)")
    lines.append("")
    lines.append(_markdown_table(report))
    lines.append("")

    lines.append("## End-to-end breakdown (warm, mean ms)")
    lines.append("")
    lines.append(_markdown_breakdown(report))
    lines.append("")

    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


def print_console(report: BenchReport, console: Console | None = None) -> None:
    console = console or Console()
    cfg = report.config

    console.rule("[bold cyan]Retrieval latency report[/bold cyan]")
    console.print(
        f"collection=[bold]{cfg.collection}[/bold] "
        f"dense=[bold]{cfg.dense_model}[/bold] "
        f"rerank={'[green]on[/green]' if cfg.use_reranker else '[yellow]off[/yellow]'}"
    )
    console.print(
        f"queries={report.num_queries} warmup={cfg.warmup} warm_iters={cfg.warm_iters}"
    )

    for regime, stats in (("cold", report.cold_stats), ("warm", report.warm_stats)):
        table = Table(title=f"{regime} cache — per-stage latency (ms)", show_lines=False)
        table.add_column("stage", style="bold")
        table.add_column("n", justify="right")
        table.add_column("mean", justify="right")
        table.add_column("p50", justify="right")
        table.add_column("p95", justify="right")
        table.add_column("p99", justify="right")
        table.add_column("max", justify="right")
        for stage in STAGES:
            s = stats[stage]
            if s.n == 0:
                continue
            style = "bold green" if stage == "total" else None
            table.add_row(
                stage,
                str(s.n),
                _fmt(s.mean_ms),
                _fmt(s.p50_ms),
                _fmt(s.p95_ms),
                _fmt(s.p99_ms),
                _fmt(s.max_ms),
                style=style,
            )
        console.print(table)

    # Breakdown: where did warm mean time go?
    breakdown = Table(title="Warm breakdown (share of total, mean ms)")
    breakdown.add_column("stage", style="bold")
    breakdown.add_column("mean ms", justify="right")
    breakdown.add_column("% of total", justify="right")
    breakdown.add_column("bar")
    total_mean = report.warm_stats["total"].mean_ms or 1.0
    for stage in STAGES:
        if stage == "total":
            continue
        s = report.warm_stats[stage]
        if s.n == 0 or s.mean_ms == 0:
            continue
        pct = 100.0 * s.mean_ms / total_mean
        bar = "█" * max(1, int(round(pct / 2)))  # 2% per block, max ~50 blocks
        breakdown.add_row(stage, _fmt(s.mean_ms), f"{pct:5.1f}%", bar)
    console.print(breakdown)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    if x >= 100:
        return f"{x:,.1f}"
    if x >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _markdown_table(report: BenchReport) -> str:
    header = "| stage | regime | n | mean | p50 | p95 | p99 | max |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|"
    rows: list[str] = [header, sep]
    for regime, stats in (("cold", report.cold_stats), ("warm", report.warm_stats)):
        for stage in STAGES:
            s: StageStats = stats[stage]
            if s.n == 0:
                continue
            rows.append(
                f"| {stage} | {regime} | {s.n} | "
                f"{_fmt(s.mean_ms)} | {_fmt(s.p50_ms)} | "
                f"{_fmt(s.p95_ms)} | {_fmt(s.p99_ms)} | {_fmt(s.max_ms)} |"
            )
    return "\n".join(rows)


def _markdown_breakdown(report: BenchReport) -> str:
    total = report.warm_stats["total"].mean_ms or 1.0
    rows: list[str] = ["| stage | mean ms | % of total |", "|---|---:|---:|"]
    for stage in STAGES:
        if stage == "total":
            continue
        s = report.warm_stats[stage]
        if s.n == 0 or s.mean_ms == 0:
            continue
        pct = 100.0 * s.mean_ms / total
        rows.append(f"| {stage} | {_fmt(s.mean_ms)} | {pct:.1f}% |")
    rows.append(f"| **total** | **{_fmt(total)}** | 100.0% |")
    return "\n".join(rows)
