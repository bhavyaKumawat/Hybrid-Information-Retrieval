"""SQLite-backed manifest for incremental ingestion.

Two tables, both keyed on deterministic IDs so the manifest can be the
single source of truth about what lives in Qdrant:

* ``documents`` — one row per document. Tracks the document-level
  content hash plus the chunker/model fingerprint that produced its
  current set of chunks.
* ``chunks``    — one row per chunk. Holds the chunk-level hash and
  the Qdrant point id, so we can reconcile upserts and deletions.

No ORM — this is a tiny, hot-path table.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id               TEXT PRIMARY KEY,
    content_hash         TEXT NOT NULL,
    chunker_config_hash  TEXT NOT NULL,
    dense_model          TEXT NOT NULL,
    sparse_model         TEXT NOT NULL,
    chunk_count          INTEGER NOT NULL,
    ingested_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    doc_id       TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    chunk_hash   TEXT NOT NULL,
    point_id     TEXT NOT NULL,
    dense_model  TEXT NOT NULL,
    ingested_at  TEXT NOT NULL,
    PRIMARY KEY (doc_id, chunk_index),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(doc_id, chunk_hash);
"""


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    content_hash: str
    chunker_config_hash: str
    dense_model: str
    sparse_model: str
    chunk_count: int
    ingested_at: str


@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    chunk_index: int
    chunk_hash: str
    point_id: str
    dense_model: str
    ingested_at: str


class Manifest:
    """Thread-safe wrapper around a single SQLite file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A fresh connection per call; sqlite3 connections aren't safe to share
        # across threads, and we're not in a hot-enough path to need pooling.
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return _doc_from_row(row) if row else None

    def upsert_document(self, record: DocumentRecord) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (doc_id, content_hash, chunker_config_hash, dense_model,
                     sparse_model, chunk_count, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    content_hash        = excluded.content_hash,
                    chunker_config_hash = excluded.chunker_config_hash,
                    dense_model         = excluded.dense_model,
                    sparse_model        = excluded.sparse_model,
                    chunk_count         = excluded.chunk_count,
                    ingested_at         = excluded.ingested_at
                """,
                (
                    record.doc_id,
                    record.content_hash,
                    record.chunker_config_hash,
                    record.dense_model,
                    record.sparse_model,
                    record.chunk_count,
                    record.ingested_at,
                ),
            )

    def delete_document(self, doc_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def get_chunks(self, doc_id: str) -> list[ChunkRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()
        return [_chunk_from_row(r) for r in rows]

    def upsert_chunks(self, records: Iterable[ChunkRecord]) -> None:
        rows = [
            (
                r.doc_id,
                r.chunk_index,
                r.chunk_hash,
                r.point_id,
                r.dense_model,
                r.ingested_at,
            )
            for r in records
        ]
        if not rows:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunks
                    (doc_id, chunk_index, chunk_hash, point_id, dense_model, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id, chunk_index) DO UPDATE SET
                    chunk_hash  = excluded.chunk_hash,
                    point_id    = excluded.point_id,
                    dense_model = excluded.dense_model,
                    ingested_at = excluded.ingested_at
                """,
                rows,
            )

    def delete_chunks_above(self, doc_id: str, max_index: int) -> list[ChunkRecord]:
        """Remove chunk rows with ``chunk_index > max_index`` (returns what was removed)."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND chunk_index > ?",
                (doc_id, max_index),
            ).fetchall()
            stale = [_chunk_from_row(r) for r in rows]
            conn.execute(
                "DELETE FROM chunks WHERE doc_id = ? AND chunk_index > ?",
                (doc_id, max_index),
            )
        return stale

    def delete_chunks_for_doc(self, doc_id: str) -> list[ChunkRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchall()
            stale = [_chunk_from_row(r) for r in rows]
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        return stale

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": docs, "chunks": chunks}

    def reset(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")


# ----------------------------------------------------------------------
# Row helpers
# ----------------------------------------------------------------------


def _doc_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        doc_id=row["doc_id"],
        content_hash=row["content_hash"],
        chunker_config_hash=row["chunker_config_hash"],
        dense_model=row["dense_model"],
        sparse_model=row["sparse_model"],
        chunk_count=row["chunk_count"],
        ingested_at=row["ingested_at"],
    )


def _chunk_from_row(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        doc_id=row["doc_id"],
        chunk_index=row["chunk_index"],
        chunk_hash=row["chunk_hash"],
        point_id=row["point_id"],
        dense_model=row["dense_model"],
        ingested_at=row["ingested_at"],
    )
