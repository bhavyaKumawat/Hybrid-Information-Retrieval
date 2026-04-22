"""Composable hybrid retrieval engine.

The pipeline is three swappable stages:

1.  **Retrieve (parallel, component-preserving)** — two single-vector
    ``query_points`` calls to Qdrant, one via the ``dense`` named vector and
    one via the ``bm25`` sparse vector. Keeping the calls separate is a
    deliberate choice: Qdrant's built-in RRF fuses server-side and throws
    away the component scores, which the assignment requires we surface.

2.  **Fuse — weighted Reciprocal Rank Fusion.** For each point that appeared
    in either result set, compute ``score = w_dense / (k + r_dense) +
    w_sparse / (k + r_sparse)``. Weights and ``k`` are tunable per-request.

3.  **Rerank (optional) — cross-encoder.** Take the top ``rerank_top_n`` fused
    candidates, run the reranker, then keep the best ``top_k``. Can be
    disabled per-request; when disabled, top_k is taken straight from the
    fused list.

All three stages compose: skip (2) by setting one weight to zero, skip (3)
by passing ``use_reranker=False``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import Settings
from .embeddings import DenseEmbedder, SparseEmbedder
from .models import (
    DateRange,
    ScoreBreakdown,
    SearchDebug,
    SearchFilters,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from .qdrant_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from .reranker import CrossEncoderReranker

# ---------------------------------------------------------------------------
# Internal bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    point_id: str
    payload: dict[str, Any]
    bm25_score: float | None = None
    bm25_rank: int | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None
    fused_rrf: float = 0.0
    reranker_score: float | None = None

    def final_score(self) -> float:
        return self.reranker_score if self.reranker_score is not None else self.fused_rrf


@dataclass
class HybridSearchEngine:
    settings: Settings
    client: QdrantClient
    dense: DenseEmbedder
    sparse: SparseEmbedder
    reranker: CrossEncoderReranker | None = None
    _cache: dict[str, Any] = field(default_factory=dict)

    # ---- Public API ---------------------------------------------------

    def search(self, request: SearchRequest) -> SearchResponse:
        s = self.settings
        top_k = request.top_k or s.top_k
        rerank_top_n = request.rerank_top_n or s.rerank_top_n
        prefetch_limit = request.prefetch_limit or s.prefetch_limit
        w_dense = s.weight_dense if request.weight_dense is None else request.weight_dense
        w_sparse = s.weight_sparse if request.weight_sparse is None else request.weight_sparse
        use_reranker = s.use_reranker if request.use_reranker is None else request.use_reranker
        use_reranker = use_reranker and self.reranker is not None

        qfilter = _build_qdrant_filter(request.filters)

        timings: dict[str, float] = {}

        # 1. Retrieve (parallel component lookups)
        t0 = time.perf_counter()
        dense_hits, sparse_hits = self._retrieve_components(
            query=request.query,
            limit=prefetch_limit,
            qfilter=qfilter,
            w_dense=w_dense,
            w_sparse=w_sparse,
        )
        timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 2. Fuse via weighted RRF
        t0 = time.perf_counter()
        candidates = _weighted_rrf_fuse(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            w_dense=w_dense,
            w_sparse=w_sparse,
            k=s.rrf_k,
        )
        timings["fuse_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 3. Optional rerank.
        # Failure to load the reranker model degrades gracefully: log and
        # fall back to the fused order. This is what we want when, e.g., the
        # FastEmbed cache has a partial download — `/search` still returns
        # results instead of 500-ing. The `debug.use_reranker` flag in the
        # response will then reflect reality (False) so callers can tell.
        if use_reranker and candidates and self.reranker is not None:
            t0 = time.perf_counter()
            top = candidates[:rerank_top_n]
            texts = [c.payload.get("chunk_text", "") for c in top]
            try:
                ranked = self.reranker.rerank(request.query, top, texts)
                for r in ranked:
                    r.item.reranker_score = r.score
                top_sorted = [r.item for r in ranked]
                # Keep the tail (non-reranked) after, in original fused order.
                candidates = top_sorted + candidates[rerank_top_n:]
                timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            except Exception as exc:
                log.warning(
                    "reranker_failed model=%s err=%r — falling back to fused order",
                    self.reranker.model_name,
                    exc,
                )
                use_reranker = False
                timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        candidates.sort(key=lambda c: c.final_score(), reverse=True)
        top_hits = candidates[:top_k]

        return SearchResponse(
            query=request.query,
            hits=[_candidate_to_hit(c) for c in top_hits],
            debug=SearchDebug(
                dense_model=self.dense.model_name,
                sparse_model=self.sparse.model_name,
                reranker_model=self.reranker.model_name if self.reranker else None,
                use_reranker=use_reranker,
                fusion="weighted_rrf",
                rrf_k=s.rrf_k,
                weights={"dense": w_dense, "sparse": w_sparse},
                prefetch_limit=prefetch_limit,
                rerank_top_n=rerank_top_n,
                timings_ms=timings,
            ),
        )

    # ---- Retrieval ----------------------------------------------------

    def _retrieve_components(
        self,
        query: str,
        limit: int,
        qfilter: qm.Filter | None,
        w_dense: float,
        w_sparse: float,
    ) -> tuple[list[qm.ScoredPoint], list[qm.ScoredPoint]]:
        collection = self.settings.qdrant_collection

        dense_hits: list[qm.ScoredPoint] = []
        sparse_hits: list[qm.ScoredPoint] = []

        # Skip the corresponding retrieval entirely when weight==0; saves a
        # round-trip and an embedding.
        if w_dense > 0:
            dense_vec = self.dense.embed_query(query)
            dense_res = self.client.query_points(
                collection_name=collection,
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                query_filter=qfilter,
                limit=limit,
                with_payload=True,
            )
            dense_hits = list(dense_res.points)

        if w_sparse > 0:
            sparse_vec = self.sparse.embed_query(query)
            sparse_res = self.client.query_points(
                collection_name=collection,
                query=sparse_vec,
                using=SPARSE_VECTOR_NAME,
                query_filter=qfilter,
                limit=limit,
                with_payload=True,
            )
            sparse_hits = list(sparse_res.points)

        return dense_hits, sparse_hits


# ---------------------------------------------------------------------------
# Fusion + filter helpers
# ---------------------------------------------------------------------------


def _weighted_rrf_fuse(
    dense_hits: list[qm.ScoredPoint],
    sparse_hits: list[qm.ScoredPoint],
    w_dense: float,
    w_sparse: float,
    k: int,
) -> list[_Candidate]:
    pool: dict[str, _Candidate] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        pid = str(hit.id)
        cand = pool.setdefault(
            pid, _Candidate(point_id=pid, payload=dict(hit.payload or {}))
        )
        cand.semantic_score = float(hit.score)
        cand.semantic_rank = rank
        cand.fused_rrf += w_dense / (k + rank)

    for rank, hit in enumerate(sparse_hits, start=1):
        pid = str(hit.id)
        cand = pool.setdefault(
            pid, _Candidate(point_id=pid, payload=dict(hit.payload or {}))
        )
        cand.bm25_score = float(hit.score)
        cand.bm25_rank = rank
        cand.fused_rrf += w_sparse / (k + rank)

    return sorted(pool.values(), key=lambda c: c.fused_rrf, reverse=True)


def _build_qdrant_filter(filters: SearchFilters | None) -> qm.Filter | None:
    if filters is None:
        return None

    must: list[qm.FieldCondition] = []

    if filters.source is not None:
        if isinstance(filters.source, list):
            must.append(qm.FieldCondition(key="source", match=qm.MatchAny(any=filters.source)))
        else:
            must.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=filters.source)))

    if filters.doc_ids:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=filters.doc_ids)))

    if filters.date is not None:
        rng = _date_range_from_filter(filters.date)
        if rng is not None:
            must.append(qm.FieldCondition(key="date", range=rng))

    if not must:
        return None
    return qm.Filter(must=must)


def _date_range_from_filter(date: DateRange) -> qm.DatetimeRange | None:
    if date.gte is None and date.lte is None:
        return None
    return qm.DatetimeRange(
        gte=_to_utc(date.gte) if date.gte else None,
        lte=_to_utc(date.lte) if date.lte else None,
    )


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _candidate_to_hit(c: _Candidate) -> SearchHit:
    p = c.payload
    date_str = p.get("date")
    date_val: datetime | None = None
    if isinstance(date_str, str):
        try:
            date_val = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            date_val = None
    return SearchHit(
        doc_id=p.get("doc_id", ""),
        chunk_index=int(p.get("chunk_index", 0)),
        chunk_total=int(p.get("chunk_total", 0)),
        chunk_text=p.get("chunk_text", ""),
        doc_title=p.get("doc_title"),
        source=p.get("source"),
        source_url=p.get("source_url"),
        date=date_val,
        scores=ScoreBreakdown(
            bm25=c.bm25_score,
            semantic=c.semantic_score,
            fused_rrf=c.fused_rrf,
            reranker=c.reranker_score,
            final=c.final_score(),
            bm25_rank=c.bm25_rank,
            semantic_rank=c.semantic_rank,
        ),
        payload={
            k: v
            for k, v in p.items()
            if k
            not in {
                "chunk_text",
                "chunk_index",
                "chunk_total",
                "doc_id",
                "doc_title",
                "source_url",
                "source",
                "date",
            }
        },
    )
