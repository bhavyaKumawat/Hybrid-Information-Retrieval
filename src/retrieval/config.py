"""Configuration management via pydantic-settings.

All tunables are surfaced as environment variables (or entries in `.env`).
Any callable/service in the codebase that needs config should accept a
`Settings` instance explicitly — we never reach for a global.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known FastEmbed dense-model dimensions. Kept explicit so a
# misconfigured model fails fast at collection-creation time instead
# of silently shipping wrong-sized vectors to Qdrant.
DENSE_MODEL_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


ChunkStrategy = Literal["recursive", "fixed", "semantic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "nfcorpus"

    # Models
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Chunking
    chunk_strategy: ChunkStrategy = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    semantic_breakpoint_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = "percentile"
    semantic_breakpoint_amount: float = 95.0

    # Retrieval defaults
    top_k: int = 5
    rerank_top_n: int = 20
    prefetch_limit: int = 50
    weight_dense: float = 1.0
    weight_sparse: float = 1.0
    rrf_k: int = 60
    use_reranker: bool = True

    # Ingestion
    manifest_path: Path = Path("./data/manifest.db")
    batch_size: int = 64
    max_docs: int = -1

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @field_validator("dense_model")
    @classmethod
    def _validate_dense_model(cls, v: str) -> str:
        if v not in DENSE_MODEL_DIMS:
            raise ValueError(
                f"Unknown dense model '{v}'. Known: {sorted(DENSE_MODEL_DIMS)}"
            )
        return v

    @property
    def dense_dim(self) -> int:
        return DENSE_MODEL_DIMS[self.dense_model]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
