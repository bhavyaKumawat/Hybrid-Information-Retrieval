"""Chunker interface.

Every implementation returns a deterministic list of chunk strings from a
document and exposes a JSON-serialisable configuration dict. That dict is
fingerprinted (see :mod:`retrieval.hashing`) and bound into every chunk
hash, so a config change invalidates the right cache entries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..hashing import hash_chunker_config


class Chunker(ABC):
    """Common chunker interface."""

    name: str

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split a single document into ordered chunks. Empty strings are filtered."""

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of this chunker.

        Must include the strategy name and every parameter that affects output,
        so the config hash is a complete fingerprint.
        """

    def config_hash(self) -> str:
        return hash_chunker_config(self.config())
