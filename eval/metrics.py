"""Metric computation via ``ranx``.

``ranx`` reads BEIR-style qrels/runs as ``{qid: {did: score}}`` dicts
and computes ``precision@k``, ``recall@k``, ``ndcg@k`` (and friends) in
one call.

The input to :func:`evaluate_run` is a run JSON written by
:mod:`eval.runner` — chunk-level hits with per-component scores. This
module:

1. Reads the chunk hits.
2. Aggregates chunks → docs via ``max-pool`` (see :mod:`eval.aggregate`).
3. Hands the resulting document-level run to ranx along with the qrels.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

# See ``eval/__init__.py`` — ranx emits ``SyntaxWarning`` for LaTeX string
# literals on import. Re-install the filter here in case this module is
# imported without going through the ``eval`` package initializer.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"ranx\..*")

from ranx import Qrels, Run, evaluate  # noqa: E402

from .aggregate import max_pool_chunks_to_docs

log = logging.getLogger(__name__)

DEFAULT_KS: tuple[int, ...] = (5, 10)
DEFAULT_METRICS: tuple[str, ...] = ("precision", "recall", "ndcg")


def _metric_strings(
    metrics: Iterable[str] = DEFAULT_METRICS,
    ks: Iterable[int] = DEFAULT_KS,
) -> list[str]:
    return [f"{m}@{k}" for m in metrics for k in ks]


def doc_run_from_chunk_hits(
    per_query_chunks: dict[str, list[dict]],
    score_field: str = "score",
) -> dict[str, dict[str, float]]:
    """Aggregate chunk hits to a document run (max-pool)."""
    return {
        qid: max_pool_chunks_to_docs(hits, score_field=score_field)
        for qid, hits in per_query_chunks.items()
    }


def evaluate_run(
    run_path: Path | str,
    qrels: dict[str, dict[str, int]],
    ks: Iterable[int] = DEFAULT_KS,
    metrics: Iterable[str] = DEFAULT_METRICS,
    score_field: str = "score",
) -> dict[str, float]:
    """Load a run JSON, aggregate, evaluate with ranx. Returns metric dict."""
    run_path = Path(run_path)
    with run_path.open() as fh:
        payload = json.load(fh)

    per_query_chunks = payload.get("queries", {})
    # Keep only queries that have qrels — anything else would be noise.
    per_query_chunks = {
        qid: hits for qid, hits in per_query_chunks.items() if qid in qrels
    }

    doc_run = doc_run_from_chunk_hits(per_query_chunks, score_field=score_field)

    # ranx expects every qrels qid to appear in the run (even if empty),
    # or it'll raise. Ensure that invariant.
    for qid in qrels:
        doc_run.setdefault(qid, {})

    r_qrels = Qrels(qrels)
    r_run = Run(doc_run)

    metric_list = _metric_strings(metrics=metrics, ks=ks)
    raw = evaluate(r_qrels, r_run, metric_list)

    # ranx returns a dict when multiple metrics are requested, else a scalar.
    if not isinstance(raw, dict):
        raw = {metric_list[0]: float(raw)}

    # Cast values to float for clean JSON serialisation.
    return {k: float(v) for k, v in raw.items()}
