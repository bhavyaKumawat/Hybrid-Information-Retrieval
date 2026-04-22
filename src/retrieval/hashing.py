"""Two-level hashing for incremental ingestion.

Incremental correctness hinges on three hashes:

* ``chunker_config_hash`` — fingerprints the chunker (strategy + size + overlap + params).
  Baked into every chunk hash so a chunker-config change forces re-chunking.
* ``content_hash`` — document-level. ``sha256(title + "\\n" + text)``.
  If unchanged *and* chunker/model config unchanged, we skip the document entirely.
* ``chunk_hash`` — chunk-level. ``sha256(chunker_config_hash + chunk_text)``.
  Drives per-chunk re-embedding decisions.

Hashes are prefixed with ``"sha256:"`` so the payload is self-describing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PREFIX = "sha256:"


def _sha256(data: bytes) -> str:
    return PREFIX + hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    return _sha256(text.encode("utf-8"))


def hash_document(title: str, text: str) -> str:
    """Document-level hash. Title + text is enough for NFCorpus-style inputs."""
    payload = f"{title}\n{text}"
    return _sha256(payload.encode("utf-8"))


def hash_chunker_config(config: dict[str, Any]) -> str:
    """Stable hash over a chunker's configuration dict."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return _sha256(canonical.encode("utf-8"))


def hash_chunk(chunker_config_hash: str, chunk_text: str) -> str:
    """Chunk-level hash binds the chunk text to the chunker config that produced it."""
    payload = f"{chunker_config_hash}\0{chunk_text}"
    return _sha256(payload.encode("utf-8"))
