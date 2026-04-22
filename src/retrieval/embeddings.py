"""FastEmbed-backed dense + sparse (BM25) embedders.

Thin wrappers that normalise FastEmbed's generator-returning APIs into the
shapes the rest of the pipeline wants (flat lists, plain dicts for sparse),
and memoise the underlying model per (class, model_name).

The underlying ONNX model is loaded **lazily** on first call, so that:

* the API can start even if the model isn't fully downloaded yet (FastEmbed
  fetches on first instantiation; a slow/interrupted fetch shouldn't take
  the whole server down at startup), and
* `warmup()` is provided when callers want to pay that cost up front
  (the ingestion CLI does).
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.http import models as qm


@lru_cache(maxsize=4)
def _text_embedding(model_name: str) -> TextEmbedding:
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=4)
def _sparse_embedding(model_name: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=model_name)


class DenseEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: TextEmbedding | None = None

    def _ensure(self) -> TextEmbedding:
        if self._model is None:
            self._model = _text_embedding(self.model_name)
        return self._model

    def warmup(self) -> None:
        self._ensure()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self._ensure().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._ensure().query_embed(text))).tolist()


class SparseEmbedder:
    """BM25 sparse vectors via FastEmbed -> Qdrant ``SparseVector``."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self.model_name = model_name
        self._model: SparseTextEmbedding | None = None

    def _ensure(self) -> SparseTextEmbedding:
        if self._model is None:
            self._model = _sparse_embedding(self.model_name)
        return self._model

    def warmup(self) -> None:
        self._ensure()

    def embed(self, texts: list[str]) -> list[qm.SparseVector]:
        if not texts:
            return []
        out: list[qm.SparseVector] = []
        for sv in self._ensure().embed(texts):
            out.append(
                qm.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
            )
        return out

    def embed_query(self, text: str) -> qm.SparseVector:
        sv = next(iter(self._ensure().query_embed(text)))
        return qm.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
