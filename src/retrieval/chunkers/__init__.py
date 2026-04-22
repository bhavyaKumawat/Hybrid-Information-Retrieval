"""Chunking strategies behind a common interface.

Use :func:`build_chunker` to construct a chunker from a :class:`Settings`
instance. All chunkers implement :class:`Chunker`.
"""

from __future__ import annotations

from ..config import Settings
from .base import Chunker
from .fixed import FixedSizeChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker

__all__ = [
    "Chunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "build_chunker",
]


def build_chunker(settings: Settings) -> Chunker:
    """Factory: pick a chunker implementation from settings."""
    strategy = settings.chunk_strategy
    if strategy == "recursive":
        return RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    if strategy == "fixed":
        return FixedSizeChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    if strategy == "semantic":
        return SemanticChunker(
            embedding_model=settings.dense_model,
            breakpoint_threshold_type=settings.semantic_breakpoint_type,
            breakpoint_threshold_amount=settings.semantic_breakpoint_amount,
        )
    raise ValueError(f"Unknown chunk strategy: {strategy}")
