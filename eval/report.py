"""Summarise the cached runs into a CSV + a human-readable Markdown table.

Decoupled from retrieval: read ``eval/runs/*.json``, re-score each with
ranx against the NFCorpus test qrels, write:

* ``eval/reports/summary.csv`` — machine-readable, one row per run.
* ``eval/reports/summary.md``  — one Markdown table per ablation,
  sorted by NDCG@10 desc.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .metrics import DEFAULT_KS, DEFAULT_METRICS, evaluate_run

log = logging.getLogger(__name__)

# Stable column order for the CSV and MD tables.
METRIC_COLUMNS: list[str] = [
    f"{m}@{k}" for m in DEFAULT_METRICS for k in DEFAULT_KS
]
CONFIG_COLUMNS: list[str] = [
    "tag",
    "ablation",
    "dense_model",
    "chunk_strategy",
    "chunk_size",
    "chunk_overlap",
    "weight_dense",
    "weight_sparse",
    "use_reranker",
    "reranker_model",
    "rerank_top_n",
    "prefetch_limit",
    "collection_name",
]


def collect_run_metrics(
    runs_dir: Path | str,
    qrels: dict[str, dict[str, int]],
) -> list[dict]:
    """Score every ``runs/*.json`` file. Returns a list of row dicts."""
    runs_dir = Path(runs_dir)
    rows: list[dict] = []
    for run_path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(run_path.read_text())
        cfg = payload.get("config", {})
        try:
            metrics = evaluate_run(run_path, qrels=qrels)
        except Exception as exc:
            log.warning("report.evaluate_failed run=%s err=%r", run_path.name, exc)
            continue
        row = {col: cfg.get(col) for col in CONFIG_COLUMNS}
        row["run_file"] = run_path.name
        row["num_queries"] = payload.get("num_queries")
        row["elapsed_s"] = payload.get("elapsed_s")
        for mk in METRIC_COLUMNS:
            row[mk] = round(float(metrics.get(mk, 0.0)), 4)
        rows.append(row)
    return rows


def write_csv(rows: Iterable[dict], out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = CONFIG_COLUMNS + ["run_file", "num_queries", "elapsed_s"] + METRIC_COLUMNS
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def write_markdown(rows: list[dict], out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by ablation, sort each group by NDCG@10 desc.
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("ablation") or "unknown"].append(r)

    ablation_order = [
        "retrieval_mode",
        "embedder_size",
        "chunker",
        "reranker",
        "rerank_top_n",
        "fusion_weights",
    ]
    ordered_keys = [k for k in ablation_order if k in groups] + [
        k for k in groups if k not in ablation_order
    ]

    lines: list[str] = [
        "# NFCorpus evaluation — summary",
        "",
        "Metrics: precision@5, precision@10, recall@5, recall@10, ndcg@5, ndcg@10.",
        "Split: BEIR NFCorpus `test`. Aggregation: max-pool chunks → docs.",
        "",
    ]

    # One-shot overall table (every run, sorted by ndcg@10) for a quick look.
    lines.append("## All runs (sorted by NDCG@10)")
    lines.append("")
    lines.extend(
        _md_table(
            sorted(rows, key=lambda r: r.get("ndcg@10", 0.0), reverse=True),
            columns=["tag", "ablation", *METRIC_COLUMNS],
            headers=["tag", "ablation", *METRIC_COLUMNS],
        )
    )
    lines.append("")

    for ab in ordered_keys:
        group_rows = sorted(
            groups[ab], key=lambda r: r.get("ndcg@10", 0.0), reverse=True
        )
        lines.append(f"## Ablation — {ab}")
        lines.append("")
        lines.extend(
            _md_table(
                group_rows,
                columns=_per_ablation_columns(ab),
                headers=_per_ablation_headers(ab),
            )
        )
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path


def _per_ablation_columns(ablation: str) -> list[str]:
    # Surface the knob that changed in each ablation.
    knob_cols: dict[str, list[str]] = {
        "retrieval_mode": ["weight_dense", "weight_sparse", "use_reranker"],
        "embedder_size": ["dense_model"],
        "chunker": ["chunk_strategy", "chunk_size", "chunk_overlap"],
        "reranker": ["reranker_model", "use_reranker"],
        "rerank_top_n": ["rerank_top_n", "prefetch_limit"],
        "fusion_weights": ["weight_dense", "weight_sparse"],
    }
    return ["tag", *knob_cols.get(ablation, []), *METRIC_COLUMNS]


def _per_ablation_headers(ablation: str) -> list[str]:
    return _per_ablation_columns(ablation)


def _md_table(
    rows: list[dict],
    columns: list[str],
    headers: list[str],
) -> list[str]:
    if not rows:
        return ["_(no runs)_"]
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for r in rows:
        vals = []
        for col in columns:
            v = r.get(col)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return [header_row, sep_row, *body]
