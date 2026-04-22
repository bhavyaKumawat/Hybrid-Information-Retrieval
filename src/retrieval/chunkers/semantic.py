"""Semantic chunker (langchain-experimental).

Wraps :class:`langchain_experimental.text_splitter.SemanticChunker`, driven by
a FastEmbed-backed embeddings adapter. Chunk boundaries are decided by
embedding-similarity drop, not a fixed window, so there is no ``chunk_size``
parameter here — the knobs are the breakpoint threshold type + amount.
"""

from __future__ import annotations

from typing import Any

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker

from .base import Chunker


class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(
        self,
        embedding_model: str,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
    ) -> None:
        self.embedding_model = embedding_model
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        embeddings = FastEmbedEmbeddings(model_name=embedding_model)
        self._splitter = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return [c for c in self._splitter.split_text(text) if c.strip()]

    def config(self) -> dict[str, Any]:
        # Semantic chunking depends on the embedding model, so it has to
        # be part of the fingerprint — a model change means different boundaries.
        return {
            "strategy": self.name,
            "embedding_model": self.embedding_model,
            "breakpoint_threshold_type": self.breakpoint_threshold_type,
            "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
        }
