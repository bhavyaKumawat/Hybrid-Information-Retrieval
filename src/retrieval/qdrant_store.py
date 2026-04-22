"""Qdrant collection setup and helpers.

The collection is built with two **named vectors**:

* ``dense``  — ``VectorParams(size=dense_dim, distance=COSINE)``
* ``bm25``   — ``SparseVectorParams(modifier=IDF)`` (BM25 via FastEmbed)

Payload indexes are created explicitly on every filterable field the API
exposes (``source``, ``date``, ``doc_id``) so ``Prefetch.filter`` stays fast.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import Settings

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# Namespace for deterministic point IDs: f"{doc_id}:{chunk_index}" -> UUID5.
# Stable across runs so re-ingests overwrite instead of duplicating.
_POINT_NAMESPACE = uuid.UUID("6f1c0b4a-0f3a-4a9b-8d0c-1e3a7b9c5d21")


def build_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        prefer_grpc=False,
    )


def point_id_for(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{chunk_index}"))


def ensure_collection(client: QdrantClient, settings: Settings) -> None:
    """Create the collection + payload indexes if they don't already exist."""
    name = settings.qdrant_collection
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: qm.VectorParams(
                    size=settings.dense_dim,
                    distance=qm.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                    modifier=qm.Modifier.IDF,
                )
            },
        )

    # Filterable fields — idempotent; Qdrant no-ops when the index already exists.
    _create_payload_index(client, name, "source", qm.PayloadSchemaType.KEYWORD)
    _create_payload_index(client, name, "doc_id", qm.PayloadSchemaType.KEYWORD)
    _create_payload_index(client, name, "date", qm.PayloadSchemaType.DATETIME)


def _create_payload_index(
    client: QdrantClient,
    collection: str,
    field: str,
    schema: qm.PayloadSchemaType,
) -> None:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
        )
    except Exception:
        # Re-running is expected; Qdrant returns an error if the index already
        # exists in some versions. Silence that specific case.
        pass


def drop_collection(client: QdrantClient, settings: Settings) -> None:
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)


def collection_point_count(client: QdrantClient, settings: Settings) -> int | None:
    try:
        info = client.get_collection(settings.qdrant_collection)
        return info.points_count
    except Exception:
        return None
