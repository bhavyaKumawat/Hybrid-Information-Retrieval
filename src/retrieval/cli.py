"""Typer CLI for ingestion + operations.

Main commands:

* ``ir3 ingest``   — run incremental ingestion of NFCorpus.
* ``ir3 stats``    — manifest + collection stats.
* ``ir3 reset``    — drop Qdrant collection and manifest.
* ``ir3 serve``    — run the FastAPI server.
* ``ir3 search``   — fire a one-off search against the running engine (in-process).
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from .chunkers import build_chunker
from .config import Settings, get_settings
from .datasets import load_nfcorpus
from .embeddings import DenseEmbedder, SparseEmbedder
from .ingestion import IncrementalIngestor
from .manifest import Manifest
from .models import SearchFilters, SearchRequest
from .qdrant_store import build_client, collection_point_count, drop_collection, ensure_collection
from .reranker import CrossEncoderReranker
from .search import HybridSearchEngine

app = typer.Typer(add_completion=False, help="IR3 — Production-grade hybrid retrieval platform.")
console = Console()


@app.command()
def ingest(
    max_docs: int | None = typer.Option(
        None,
        "--max-docs",
        "-n",
        help="Limit ingestion to N NFCorpus documents. Overrides MAX_DOCS from .env.",
    ),
    dense_model: str | None = typer.Option(
        None, "--dense-model", help="Override DENSE_MODEL for this run."
    ),
    chunk_strategy: str | None = typer.Option(
        None, "--chunk-strategy", help="recursive | fixed | semantic"
    ),
    chunk_size: int | None = typer.Option(None, "--chunk-size"),
    chunk_overlap: int | None = typer.Option(None, "--chunk-overlap"),
) -> None:
    """Incrementally ingest NFCorpus into Qdrant."""
    settings = _settings_with_overrides(
        dense_model=dense_model,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_docs=max_docs,
    )
    _print_run_config(settings)

    client = build_client(settings)
    manifest = Manifest(settings.manifest_path)
    chunker = build_chunker(settings)
    dense = DenseEmbedder(settings.dense_model)
    sparse = SparseEmbedder(settings.sparse_model)

    # Eagerly pull both ONNX models now, so any download error blows up
    # *before* we start streaming docs through the pipeline.
    console.print("[dim]Warming up dense + sparse models...[/dim]")
    dense.warmup()
    sparse.warmup()

    ensure_collection(client, settings)

    limit = settings.max_docs if settings.max_docs >= 0 else None
    documents = list(load_nfcorpus(limit=limit))

    ingestor = IncrementalIngestor(
        settings=settings,
        client=client,
        manifest=manifest,
        chunker=chunker,
        dense=dense,
        sparse=sparse,
        console=console,
    )
    stats = ingestor.ingest(documents, total=len(documents))
    console.print(Panel(Pretty(stats.summary()), title="Ingestion summary"))


@app.command()
def stats() -> None:
    """Show manifest + collection stats."""
    settings = get_settings()
    manifest = Manifest(settings.manifest_path)
    client = build_client(settings)

    m = manifest.stats()
    points = collection_point_count(client, settings)

    table = Table(title="IR3 state")
    table.add_column("key")
    table.add_column("value")
    table.add_row("collection", settings.qdrant_collection)
    table.add_row("manifest.documents", str(m["documents"]))
    table.add_row("manifest.chunks", str(m["chunks"]))
    table.add_row("qdrant.points", str(points))
    console.print(table)


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Drop the Qdrant collection and reset the manifest."""
    settings = get_settings()
    if not yes:
        confirm = typer.confirm(f"Drop collection '{settings.qdrant_collection}' + manifest?")
        if not confirm:
            raise typer.Exit(code=1)
    client = build_client(settings)
    drop_collection(client, settings)
    Manifest(settings.manifest_path).reset()
    console.print("[green]reset complete[/green]")


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "retrieval.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Query string."),
    top_k: int = typer.Option(5, "--top-k"),
    no_rerank: bool = typer.Option(False, "--no-rerank"),
    source: str | None = typer.Option(None, "--source"),
    date_gte: str | None = typer.Option(None, "--date-gte", help="ISO 8601"),
    date_lte: str | None = typer.Option(None, "--date-lte", help="ISO 8601"),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON response."),
) -> None:
    """Search from the CLI (spins up the engine in-process)."""
    from datetime import datetime

    settings = get_settings()
    client = build_client(settings)
    engine = HybridSearchEngine(
        settings=settings,
        client=client,
        dense=DenseEmbedder(settings.dense_model),
        sparse=SparseEmbedder(settings.sparse_model),
        reranker=(
            None if no_rerank or not settings.use_reranker
            else CrossEncoderReranker(settings.reranker_model)
        ),
    )

    date_range = None
    if date_gte or date_lte:
        date_range = {
            "gte": datetime.fromisoformat(date_gte) if date_gte else None,
            "lte": datetime.fromisoformat(date_lte) if date_lte else None,
        }

    filters = None
    if source or date_range:
        filters = SearchFilters(source=source, date=date_range)

    req = SearchRequest(
        query=query,
        top_k=top_k,
        use_reranker=(not no_rerank),
        filters=filters,
    )
    resp = engine.search(req)
    if json_out:
        console.print_json(resp.model_dump_json())
        return
    _render_hits(resp)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _settings_with_overrides(**overrides) -> Settings:
    base = get_settings().model_dump()
    for key, value in overrides.items():
        if value is not None:
            base[key] = value
    # ``get_settings()`` is cached; build a fresh Settings for this invocation
    # so CLI overrides actually take effect.
    return Settings(**base)


def _print_run_config(settings: Settings) -> None:
    cfg = {
        "qdrant_url": settings.qdrant_url,
        "collection": settings.qdrant_collection,
        "dense_model": settings.dense_model,
        "sparse_model": settings.sparse_model,
        "chunk_strategy": settings.chunk_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "manifest": str(settings.manifest_path),
        "max_docs": settings.max_docs,
    }
    console.print(Panel(Pretty(cfg), title="Run config"))


def _render_hits(resp) -> None:
    console.print(Panel(f"[bold]Query:[/bold] {resp.query}"))
    for i, hit in enumerate(resp.hits, start=1):
        sc = hit.scores
        header = (
            f"[bold]#{i}[/bold]  {hit.doc_id}  "
            f"(chunk {hit.chunk_index + 1}/{hit.chunk_total})"
        )
        scores_str = json.dumps(
            {
                "final": round(sc.final, 4),
                "bm25": None if sc.bm25 is None else round(sc.bm25, 4),
                "semantic": None if sc.semantic is None else round(sc.semantic, 4),
                "fused_rrf": round(sc.fused_rrf, 6) if sc.fused_rrf is not None else None,
                "reranker": None if sc.reranker is None else round(sc.reranker, 4),
            },
            indent=2,
        )
        body = (
            f"[dim]{hit.doc_title or ''}[/dim]\n"
            f"{hit.chunk_text[:400]}{'...' if len(hit.chunk_text) > 400 else ''}\n\n"
            f"[bold]scores[/bold] {scores_str}"
        )
        console.print(Panel(body, title=header))
    console.print(Panel(Pretty(resp.debug.model_dump()), title="debug"))


if __name__ == "__main__":
    app()
