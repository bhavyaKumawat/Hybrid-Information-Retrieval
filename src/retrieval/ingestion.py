"""Incremental ingestion pipeline.

Decision tree when ingesting a document:

1.  Compute ``content_hash`` from (title, text).
2.  Look up the doc in the manifest.
3.  If the manifest record matches on (content_hash, chunker_config_hash,
    dense_model, sparse_model) → skip. Nothing to do.
4.  Otherwise re-chunk, then for each chunk:

    * Hash the chunk (bound to the chunker config).
    * If the manifest already has a chunk at the same index with the same
      hash *and* the same dense_model → re-use the existing point id, skip
      embedding + upsert.
    * Else embed (dense + sparse), upsert the point, update the manifest.

5.  If the new chunk count is smaller than before, delete the orphaned
    points from Qdrant and the manifest.

The pipeline is idempotent: re-running on identical inputs performs zero
embeddings and zero Qdrant writes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from .chunkers import Chunker
from .config import Settings
from .embeddings import DenseEmbedder, SparseEmbedder
from .hashing import hash_chunk, hash_document
from .manifest import ChunkRecord, DocumentRecord, Manifest
from .models import RawDocument
from .qdrant_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    point_id_for,
)


@dataclass
class IngestStats:
    documents_seen: int = 0
    documents_skipped: int = 0
    documents_updated: int = 0
    documents_new: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    chunks_deleted: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int | list[str]]:
        return {
            "documents_seen": self.documents_seen,
            "documents_new": self.documents_new,
            "documents_updated": self.documents_updated,
            "documents_skipped": self.documents_skipped,
            "chunks_embedded": self.chunks_embedded,
            "chunks_reused": self.chunks_reused,
            "chunks_deleted": self.chunks_deleted,
            "errors": self.errors,
        }


class IncrementalIngestor:
    def __init__(
        self,
        settings: Settings,
        client: QdrantClient,
        manifest: Manifest,
        chunker: Chunker,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        console: Console | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.manifest = manifest
        self.chunker = chunker
        self.dense = dense
        self.sparse = sparse
        self.console = console or Console()
        self._chunker_config_hash = chunker.config_hash()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ingest(self, documents: Iterable[RawDocument], total: int | None = None) -> IngestStats:
        ensure_collection(self.client, self.settings)
        stats = IngestStats()

        # Batch Qdrant upserts so we don't round-trip per chunk.
        pending_points: list[qm.PointStruct] = []
        pending_chunk_records: list[ChunkRecord] = []

        with Progress(
            TextColumn("[bold blue]Ingesting"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("ingest", total=total)
            for doc in documents:
                stats.documents_seen += 1
                try:
                    self._ingest_document(doc, stats, pending_points, pending_chunk_records)
                except Exception as exc:
                    stats.errors.append(f"{doc.doc_id}: {exc!r}")
                    self.console.print(f"[red]error[/red] {doc.doc_id}: {exc!r}")

                if len(pending_points) >= self.settings.batch_size:
                    self._flush(pending_points, pending_chunk_records)

                progress.advance(task)

            self._flush(pending_points, pending_chunk_records)

        return stats

    # ------------------------------------------------------------------
    # Per-document logic
    # ------------------------------------------------------------------

    def _ingest_document(
        self,
        doc: RawDocument,
        stats: IngestStats,
        pending_points: list[qm.PointStruct],
        pending_chunk_records: list[ChunkRecord],
    ) -> None:
        content_hash = hash_document(doc.title, doc.text)
        existing_doc = self.manifest.get_document(doc.doc_id)
        dense_model = self.dense.model_name
        sparse_model = self.sparse.model_name

        doc_unchanged = (
            existing_doc is not None
            and existing_doc.content_hash == content_hash
            and existing_doc.chunker_config_hash == self._chunker_config_hash
            and existing_doc.dense_model == dense_model
            and existing_doc.sparse_model == sparse_model
        )
        if doc_unchanged:
            stats.documents_skipped += 1
            return

        chunk_texts = self._full_chunks(doc)
        if not chunk_texts:
            # Empty after splitting — treat as a deletion if we had this doc before.
            if existing_doc is not None:
                self._purge_document(doc.doc_id, stats)
            return

        existing_chunks = {
            c.chunk_index: c for c in self.manifest.get_chunks(doc.doc_id)
        }

        # Decide per chunk: reuse vs re-embed.
        to_embed_indices: list[int] = []
        to_embed_texts: list[str] = []
        chunk_hashes: list[str] = []

        for i, text in enumerate(chunk_texts):
            c_hash = hash_chunk(self._chunker_config_hash, text)
            chunk_hashes.append(c_hash)
            existing = existing_chunks.get(i)
            if (
                existing is not None
                and existing.chunk_hash == c_hash
                and existing.dense_model == dense_model
            ):
                stats.chunks_reused += 1
                continue
            to_embed_indices.append(i)
            to_embed_texts.append(text)

        # Embed the ones that changed.
        if to_embed_texts:
            dense_vecs = self.dense.embed(to_embed_texts)
            sparse_vecs = self.sparse.embed(to_embed_texts)
            now_iso = _utc_now_iso()
            total_chunks = len(chunk_texts)

            for embed_pos, chunk_idx in enumerate(to_embed_indices):
                text = to_embed_texts[embed_pos]
                pid = point_id_for(doc.doc_id, chunk_idx)
                payload = _build_payload(
                    doc=doc,
                    chunk_text=text,
                    chunk_index=chunk_idx,
                    chunk_total=total_chunks,
                    content_hash=content_hash,
                    chunk_hash=chunk_hashes[chunk_idx],
                    chunker_config_hash=self._chunker_config_hash,
                    dense_model=dense_model,
                    sparse_model=sparse_model,
                    ingested_at=now_iso,
                )
                pending_points.append(
                    qm.PointStruct(
                        id=pid,
                        vector={
                            DENSE_VECTOR_NAME: dense_vecs[embed_pos],
                            SPARSE_VECTOR_NAME: sparse_vecs[embed_pos],
                        },
                        payload=payload,
                    )
                )
                pending_chunk_records.append(
                    ChunkRecord(
                        doc_id=doc.doc_id,
                        chunk_index=chunk_idx,
                        chunk_hash=chunk_hashes[chunk_idx],
                        point_id=pid,
                        dense_model=dense_model,
                        ingested_at=now_iso,
                    )
                )
                stats.chunks_embedded += 1

        # Reconcile orphaned chunks (old doc had more chunks than new).
        new_max_index = len(chunk_texts) - 1
        orphans = [
            rec for idx, rec in existing_chunks.items() if idx > new_max_index
        ]
        if orphans:
            self._delete_points(orphans)
            self.manifest.delete_chunks_above(doc.doc_id, new_max_index)
            stats.chunks_deleted += len(orphans)

        # Manifest: doc row always refreshed.
        self.manifest.upsert_document(
            DocumentRecord(
                doc_id=doc.doc_id,
                content_hash=content_hash,
                chunker_config_hash=self._chunker_config_hash,
                dense_model=dense_model,
                sparse_model=sparse_model,
                chunk_count=len(chunk_texts),
                ingested_at=_utc_now_iso(),
            )
        )
        if existing_doc is None:
            stats.documents_new += 1
        else:
            stats.documents_updated += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _full_chunks(self, doc: RawDocument) -> list[str]:
        # Prepend title so BM25 sees it too; it dominates relevance on
        # short docs and is how most BEIR baselines evaluate.
        payload = doc.title + "\n\n" + doc.text if doc.title else doc.text
        return self.chunker.split(payload)

    def _flush(
        self,
        pending_points: list[qm.PointStruct],
        pending_chunk_records: list[ChunkRecord],
    ) -> None:
        if not pending_points:
            return
        self.client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=pending_points,
            wait=True,
        )
        self.manifest.upsert_chunks(pending_chunk_records)
        pending_points.clear()
        pending_chunk_records.clear()

    def _delete_points(self, chunks: list[ChunkRecord]) -> None:
        ids = [c.point_id for c in chunks]
        if not ids:
            return
        self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=qm.PointIdsList(points=ids),
            wait=True,
        )

    def _purge_document(self, doc_id: str, stats: IngestStats) -> None:
        chunks = self.manifest.delete_chunks_for_doc(doc_id)
        self._delete_points(chunks)
        self.manifest.delete_document(doc_id)
        stats.chunks_deleted += len(chunks)


# ----------------------------------------------------------------------
# Free helpers
# ----------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_payload(
    *,
    doc: RawDocument,
    chunk_text: str,
    chunk_index: int,
    chunk_total: int,
    content_hash: str,
    chunk_hash: str,
    chunker_config_hash: str,
    dense_model: str,
    sparse_model: str,
    ingested_at: str,
) -> dict:
    return {
        # Retrieval essentials
        "chunk_text": chunk_text,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        # Source metadata
        "doc_id": doc.doc_id,
        "doc_title": doc.title,
        # Filterable fields
        "source": doc.source,
        "date": doc.date.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if doc.date
        else None,
        # Ingestion provenance
        "content_hash": content_hash,
        "chunk_hash": chunk_hash,
        "chunker_config_hash": chunker_config_hash,
        "dense_model": dense_model,
        "sparse_model": sparse_model,
        "ingested_at": ingested_at,
    }
