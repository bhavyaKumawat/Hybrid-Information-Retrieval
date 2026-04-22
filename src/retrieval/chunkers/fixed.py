"""Fixed-size character chunker with overlap (via langchain-text-splitters)."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import CharacterTextSplitter

from .base import Chunker


class FixedSizeChunker(Chunker):
    name = "fixed"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # ``separator=""`` -> pure character window; length-by-char keeps behaviour
        # identical across tokenizers, which matters when the only goal is a
        # reproducible fixed-size window.
        self._splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str) -> list[str]:
        return [c for c in self._splitter.split_text(text) if c.strip()]

    def config(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "unit": "characters",
        }
