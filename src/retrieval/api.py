"""FastAPI application exposing the retrieval platform."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .config import Settings, get_settings
from .embeddings import DenseEmbedder, SparseEmbedder
from .models import HealthResponse, SearchRequest, SearchResponse
from .qdrant_store import build_client, collection_point_count
from .reranker import CrossEncoderReranker
from .search import HybridSearchEngine

log = logging.getLogger("retrieval.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory.

    Models and the Qdrant client are created once at startup and shared across
    requests. Swapping any component at runtime is a matter of building a new
    ``HybridSearchEngine`` and assigning it to ``app.state.engine``.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Construction is cheap (model loading is deferred). Heavy ONNX
        # downloads happen on first call to embed/rerank.
        client = build_client(settings)
        dense = DenseEmbedder(settings.dense_model)
        sparse = SparseEmbedder(settings.sparse_model)
        reranker = (
            CrossEncoderReranker(settings.reranker_model)
            if settings.use_reranker
            else None
        )
        app.state.settings = settings
        app.state.client = client
        app.state.engine = HybridSearchEngine(
            settings=settings,
            client=client,
            dense=dense,
            sparse=sparse,
            reranker=reranker,
        )
        try:
            yield
        finally:
            client.close()

    app = FastAPI(
        title="IR3 — Hybrid Retrieval Platform",
        version="0.1.0",
        description=(
            "POST /search returns top-k chunks with per-component score "
            "breakdown: BM25 (sparse), semantic (dense), weighted RRF fused, "
            "and optional cross-encoder reranker."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        s: Settings = app.state.settings
        client = app.state.client
        reachable = False
        exists = False
        count: int | None = None
        try:
            exists = client.collection_exists(s.qdrant_collection)
            reachable = True
            if exists:
                count = collection_point_count(client, s)
        except Exception:
            reachable = False
        status = "ok" if (reachable and exists) else "degraded"
        return HealthResponse(
            status=status,
            qdrant_reachable=reachable,
            collection_exists=exists,
            collection=s.qdrant_collection,
            num_points=count,
        )

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        engine: HybridSearchEngine = app.state.engine
        try:
            return engine.search(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"search failed: {exc!r}") from exc

    return app


app = create_app()
