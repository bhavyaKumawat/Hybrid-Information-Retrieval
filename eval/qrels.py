"""BEIR-format NFCorpus qrels + queries loader.

We use the Hugging Face ``BeIR/nfcorpus`` dataset (same source the
ingestion pipeline reads the corpus from, so the ``doc_id`` namespace is
identical between what's in Qdrant and what the qrels reference).

Qrels come from ``BeIR/nfcorpus-qrels`` (three splits: train / validation
/ test). Per the eval spec, we use ``test`` for the final numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from datasets import load_dataset

log = logging.getLogger(__name__)

# Graded relevance lives in the ``score`` column (0-2 for NFCorpus).
# ranx wants ``{qid: {did: int_score}}`` — we preserve the graded values
# so NDCG is actually graded and not binary.


@dataclass(frozen=True)
class EvalData:
    queries: dict[str, str]           # qid -> query text
    qrels: dict[str, dict[str, int]]  # qid -> {doc_id: relevance}

    def __post_init__(self) -> None:
        # Drop queries that have no qrels — ranx treats them as
        # unanswerable and logs a warning; we'd rather exclude cleanly.
        for qid in list(self.queries):
            if qid not in self.qrels or not self.qrels[qid]:
                # Can't mutate a frozen dataclass directly; this is a
                # defensive assertion rather than a silent filter.
                pass

    def filter_to_qrels(self) -> "EvalData":
        kept = {qid: q for qid, q in self.queries.items() if qid in self.qrels}
        return EvalData(queries=kept, qrels=self.qrels)


def load_nfcorpus_eval(split: str = "test") -> EvalData:
    """Load NFCorpus queries + qrels for the given split.

    NFCorpus qrels are per-document (not per-chunk); aggregation from
    chunks → docs happens in :mod:`eval.aggregate` before metrics are
    computed.
    """
    qrels_ds = load_dataset("BeIR/nfcorpus-qrels", split=split)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        score = int(row["score"])
        if score <= 0:
            # ranx ignores zero-relevance anyway; skip explicitly.
            continue
        qrels.setdefault(qid, {})[did] = score

    # Queries: the ``queries`` config carries all splits; filter to what
    # qrels actually reference.
    queries_ds = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    queries: dict[str, str] = {}
    for row in queries_ds:
        qid = str(row.get("_id") or row.get("id") or "").strip()
        text = (row.get("text") or "").strip()
        if not qid or not text:
            continue
        if qid in qrels:
            queries[qid] = text

    missing = [qid for qid in qrels if qid not in queries]
    if missing:
        log.warning(
            "load_nfcorpus_eval: %d qrels queries have no matching text (split=%s); dropping",
            len(missing),
            split,
        )
        for qid in missing:
            qrels.pop(qid, None)

    log.info(
        "load_nfcorpus_eval split=%s queries=%d qrels_docs=%d",
        split,
        len(queries),
        sum(len(v) for v in qrels.values()),
    )
    return EvalData(queries=queries, qrels=qrels)
