"""Domain + API Pydantic models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Domain models (used internally by ingestion/search)
# ---------------------------------------------------------------------------


class RawDocument(BaseModel):
    """A document as pulled from the source dataset, before chunking."""

    doc_id: str
    title: str
    text: str
    source: str
    date: datetime | None = None


class Chunk(BaseModel):
    """A chunk of a document, ready to embed + upsert."""

    doc_id: str
    chunk_index: int
    chunk_total: int
    chunk_text: str
    chunk_hash: str


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class DateRange(BaseModel):
    gte: datetime | None = None
    lte: datetime | None = None


class SearchFilters(BaseModel):
    """Payload filters applied at the Prefetch level.

    `source` takes a single value or a list (OR semantics).
    `date` is inclusive on both ends.
    `doc_ids` lets callers constrain to specific documents.
    """

    source: str | list[str] | None = None
    date: DateRange | None = None
    doc_ids: list[str] | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    rerank_top_n: int | None = Field(default=None, ge=1, le=200)
    prefetch_limit: int | None = Field(default=None, ge=1, le=500)
    weight_dense: float | None = Field(default=None, ge=0.0)
    weight_sparse: float | None = Field(default=None, ge=0.0)
    use_reranker: bool | None = None
    filters: SearchFilters | None = None


class ScoreBreakdown(BaseModel):
    """Per-component score transparency.

    Any given component may be ``None`` if it did not retrieve the document
    (e.g. dense retrieved it, BM25 didn't) or was disabled for the query.
    """

    bm25: float | None = None
    semantic: float | None = None
    fused_rrf: float | None = None
    reranker: float | None = None
    final: float
    bm25_rank: int | None = None
    semantic_rank: int | None = None


class SearchHit(BaseModel):
    doc_id: str
    chunk_index: int
    chunk_total: int
    chunk_text: str
    doc_title: str | None = None
    source: str | None = None
    date: datetime | None = None
    scores: ScoreBreakdown
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchDebug(BaseModel):
    dense_model: str
    sparse_model: str
    reranker_model: str | None
    use_reranker: bool
    fusion: Literal["weighted_rrf"]
    rrf_k: int
    weights: dict[str, float]
    prefetch_limit: int
    rerank_top_n: int
    timings_ms: dict[str, float]


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    debug: SearchDebug


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant_reachable: bool
    collection_exists: bool
    collection: str
    num_points: int | None = None
