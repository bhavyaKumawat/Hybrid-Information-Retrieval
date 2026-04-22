"""Chunk-level hits → document-level ranking.

NFCorpus qrels are per-document, our retrieval is per-chunk. This
module is the bridge. We use **max-pool**: the document's score is the
maximum score across its retrieved chunks. It's the simplest defensible
choice for single-relevant-passage corpora like NFCorpus, and it's what
BEIR reports for chunked setups.

Alternative aggregations (sum, mean, log-sum-exp) are easy to add later;
the run JSON stores per-chunk scores, so re-aggregation doesn't require
re-running retrieval.
"""

from __future__ import annotations

from collections.abc import Iterable


def max_pool_chunks_to_docs(
    chunk_hits: Iterable[dict],
    score_field: str = "score",
    doc_field: str = "doc_id",
) -> dict[str, float]:
    """Take a list of chunk hits (dicts with ``doc_id`` + score) and
    collapse to ``{doc_id: max_score}``.

    Ordering in the input doesn't matter: we scan once and keep the max
    per doc. The caller is responsible for sorting the result by score
    descending if they need a ranked list (ranx does this internally).
    """
    out: dict[str, float] = {}
    for hit in chunk_hits:
        did = hit.get(doc_field)
        if not did:
            continue
        s = hit.get(score_field)
        if s is None:
            continue
        s = float(s)
        if did not in out or s > out[did]:
            out[did] = s
    return out
