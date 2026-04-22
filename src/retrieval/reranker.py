"""Cross-encoder reranker via FastEmbed's TextCrossEncoder.

A thin, composable wrapper: instantiate with a model name, call
:meth:`rerank` with a query + candidates, get back the candidates
re-ordered by cross-encoder score. Safe to disable (see
:class:`retrieval.search.HybridSearchEngine`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, TypeVar

from fastembed.rerank.cross_encoder import TextCrossEncoder

T = TypeVar("T")


@lru_cache(maxsize=4)
def _cross_encoder(model_name: str) -> TextCrossEncoder:
    return TextCrossEncoder(model_name=model_name)


@dataclass
class RerankResult(Generic[T]):
    item: T
    score: float


class CrossEncoderReranker:
    """Lazy cross-encoder.

    The underlying ONNX model is downloaded + loaded on first ``rerank`` call,
    not at construction time. That keeps the API resilient when FastEmbed's
    snapshot cache is partial — `/health` and `/search` (with reranker
    disabled) still work even if the reranker download is mid-flight or has
    failed.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model: TextCrossEncoder | None = None

    def _ensure(self) -> TextCrossEncoder:
        if self._model is None:
            self._model = _cross_encoder(self.model_name)
        return self._model

    def warmup(self) -> None:
        self._ensure()

    def rerank(
        self,
        query: str,
        items: list[T],
        texts: list[str],
    ) -> list[RerankResult[T]]:
        """Score each (query, text) pair, return items sorted by score desc."""
        if not items:
            return []
        if len(items) != len(texts):
            raise ValueError("items and texts must be the same length")
        scores = list(self._ensure().rerank(query, texts))
        ranked = sorted(
            (RerankResult(item=it, score=float(s)) for it, s in zip(items, scores, strict=True)),
            key=lambda r: r.score,
            reverse=True,
        )
        return ranked
