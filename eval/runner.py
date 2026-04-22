"""Execute a :class:`RetrievalConfig` end-to-end and cache the raw hits.

Responsibilities:

1. Figure out which Qdrant collection this config needs
   (one per unique ``embedder × chunker`` pair).
2. **Idempotently ingest** NFCorpus into that collection — the shared
   SQLite manifest makes re-runs a no-op when nothing's changed.
3. Build a :class:`HybridSearchEngine` wired to that collection and the
   right dense/sparse/reranker models.
4. Run every test query through it, capturing top-``top_k_retrieve``
   chunk hits with their per-component scores.
5. Serialise the result to ``eval/runs/<tag>.json`` so metrics can be
   re-computed without re-running retrieval.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from qdrant_client import QdrantClient
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from retrieval.chunkers import build_chunker
from retrieval.config import Settings
from retrieval.datasets import load_nfcorpus
from retrieval.embeddings import DenseEmbedder, SparseEmbedder
from retrieval.ingestion import IncrementalIngestor
from retrieval.manifest import Manifest
from retrieval.models import SearchRequest
from retrieval.qdrant_store import build_client, ensure_collection
from retrieval.reranker import CrossEncoderReranker
from retrieval.search import HybridSearchEngine

from .configs import RetrievalConfig
from .qrels import EvalData

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings shaping
# ---------------------------------------------------------------------------


def settings_for(config: RetrievalConfig, base: Settings) -> Settings:
    """Produce a ``Settings`` matching ``config`` (same env base, collection swapped)."""
    overrides = base.model_dump()
    overrides.update(
        qdrant_collection=config.collection_name,
        dense_model=config.dense_model,
        sparse_model=config.sparse_model,
        reranker_model=config.reranker_model or base.reranker_model,
        chunk_strategy=config.chunk_strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        semantic_breakpoint_type=config.semantic_breakpoint_type,
        semantic_breakpoint_amount=config.semantic_breakpoint_amount,
        weight_dense=config.weight_dense,
        weight_sparse=config.weight_sparse,
        use_reranker=config.use_reranker,
        rerank_top_n=config.rerank_top_n,
        prefetch_limit=config.prefetch_limit,
    )
    return Settings(**overrides)


# ---------------------------------------------------------------------------
# Ingestion (per-collection, idempotent)
# ---------------------------------------------------------------------------


def ensure_ingested(
    config: RetrievalConfig,
    base: Settings,
    client: QdrantClient,
    console: Console | None = None,
) -> None:
    """Ingest NFCorpus into the collection for ``config`` if needed.

    The manifest short-circuits the document loop when the doc's
    content + chunker + embedder fingerprint hasn't changed, so
    calling this on an already-ingested collection is cheap.
    """
    settings = settings_for(config, base)
    ensure_collection(client, settings)

    manifest = Manifest(settings.manifest_path, settings.qdrant_collection)
    chunker = build_chunker(settings)
    dense = DenseEmbedder(settings.dense_model)
    sparse = SparseEmbedder(settings.sparse_model)

    # Warmup once so failures show up before the first doc.
    dense.warmup()
    sparse.warmup()

    limit = settings.max_docs if settings.max_docs >= 0 else None
    documents = list(load_nfcorpus(limit=limit))

    ingestor = IncrementalIngestor(
        settings=settings,
        client=client,
        manifest=manifest,
        chunker=chunker,
        dense=dense,
        sparse=sparse,
        console=console or Console(),
    )
    stats = ingestor.ingest(documents, total=len(documents))
    log.info(
        "eval.ingest collection=%s summary=%s",
        settings.qdrant_collection,
        stats.summary(),
    )


# ---------------------------------------------------------------------------
# Query loop
# ---------------------------------------------------------------------------


def run_config(
    config: RetrievalConfig,
    base: Settings,
    eval_data: EvalData,
    runs_dir: Path,
    client: QdrantClient | None = None,
    console: Console | None = None,
    skip_ingest: bool = False,
    overwrite: bool = False,
) -> Path:
    """Run one config over all queries, save the raw hits to disk.

    Returns the path of the run JSON.
    """
    console = console or Console()
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / config.run_filename

    if out_path.exists() and not overwrite:
        log.info("eval.run skipped (cached): %s", out_path)
        return out_path

    settings = settings_for(config, base)
    owns_client = client is None
    if owns_client:
        client = build_client(settings)

    try:
        if not skip_ingest:
            ensure_ingested(config, base, client, console=console)

        engine = _build_engine(settings, client, config)

        per_query: dict[str, list[dict]] = {}
        total = len(eval_data.queries)
        t0 = time.perf_counter()

        with Progress(
            TextColumn(f"[bold blue]{config.tag}[/bold blue]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("search", total=total)
            for qid, query in eval_data.queries.items():
                try:
                    hits = _run_single(engine, query, config)
                except Exception as exc:
                    log.warning("eval.query_failed qid=%s err=%r", qid, exc)
                    hits = []
                per_query[qid] = hits
                progress.advance(task)

        elapsed = time.perf_counter() - t0

        payload = {
            "config": config.to_dict(),
            "num_queries": len(per_query),
            "elapsed_s": round(elapsed, 3),
            "queries": per_query,
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
        log.info(
            "eval.run saved tag=%s queries=%d elapsed_s=%.1f -> %s",
            config.tag,
            len(per_query),
            elapsed,
            out_path,
        )
        return out_path
    finally:
        if owns_client:
            client.close()


def _build_engine(
    settings: Settings,
    client: QdrantClient,
    config: RetrievalConfig,
) -> HybridSearchEngine:
    dense = DenseEmbedder(settings.dense_model)
    sparse = SparseEmbedder(settings.sparse_model)
    reranker: CrossEncoderReranker | None = None
    if config.use_reranker and config.reranker_model:
        reranker = CrossEncoderReranker(config.reranker_model)
    return HybridSearchEngine(
        settings=settings,
        client=client,
        dense=dense,
        sparse=sparse,
        reranker=reranker,
    )


def _run_single(
    engine: HybridSearchEngine,
    query: str,
    config: RetrievalConfig,
) -> list[dict]:
    req = SearchRequest(
        query=query,
        top_k=config.top_k_retrieve,
        rerank_top_n=config.rerank_top_n,
        prefetch_limit=config.prefetch_limit,
        weight_dense=config.weight_dense,
        weight_sparse=config.weight_sparse,
        use_reranker=config.use_reranker,
    )
    resp = engine.search(req)
    out: list[dict] = []
    for h in resp.hits:
        sc = h.scores
        out.append(
            {
                "doc_id": h.doc_id,
                "chunk_index": h.chunk_index,
                "score": sc.final,
                "bm25": sc.bm25,
                "semantic": sc.semantic,
                "fused_rrf": sc.fused_rrf,
                "reranker": sc.reranker,
                "bm25_rank": sc.bm25_rank,
                "semantic_rank": sc.semantic_rank,
            }
        )
    return out


