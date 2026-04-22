"""Ablation plan: the full Cartesian matrix distilled into six one-axis sweeps.

Every element of the design space is a separate ablation function that
returns a list of :class:`RetrievalConfig`. Each :class:`RetrievalConfig`
maps one-to-one to a ``runs/<tag>.json`` file on disk, so runs are cheap
to reproduce, cache, and re-score.

Defaults below are the "hold everything else fixed" baseline, chosen to
match ``.env.example``:

* dense:     BAAI/bge-small-en-v1.5
* sparse:    Qdrant/bm25
* chunker:   recursive, 512 / 50
* retrieval: hybrid (RRF equal weights) + rerank
* reranker:  BAAI/bge-reranker-base, rerank_top_n=20, prefetch_limit=50
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

ChunkStrategy = Literal["recursive", "fixed", "semantic"]
SemanticBreakpointType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]

# ---------------------------------------------------------------------------
# Defaults (match .env.example)
# ---------------------------------------------------------------------------

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_RERANKER = "BAAI/bge-reranker-base"

DEFAULT_CHUNK_STRATEGY: ChunkStrategy = "recursive"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

DEFAULT_RERANK_TOP_N = 20
DEFAULT_PREFETCH_LIMIT = 50

# Retrieve deep enough for after-aggregation @10. We cap at 100 because
# SearchRequest.top_k is validated to ``<= 100``; 100 chunks is more than
# enough to surface 10 unique docs after max-pool for NFCorpus.
DEFAULT_TOP_K_RETRIEVE = 100

# ---------------------------------------------------------------------------
# Core config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalConfig:
    """Everything needed to (a) build a collection and (b) issue one query.

    The split between "collection-dependent" fields (``dense_model`` +
    ``chunk_*``) and "query-time" fields (``weight_*``, ``rerank_*``,
    etc.) is deliberate: two configs that differ only in query-time
    fields can re-use the same Qdrant collection.
    """

    tag: str
    ablation: str

    # Collection-defining fields
    dense_model: str = DEFAULT_DENSE_MODEL
    sparse_model: str = DEFAULT_SPARSE_MODEL
    chunk_strategy: ChunkStrategy = DEFAULT_CHUNK_STRATEGY
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    semantic_breakpoint_type: SemanticBreakpointType = "percentile"
    semantic_breakpoint_amount: float = 95.0

    # Query-time fields
    weight_dense: float = 1.0
    weight_sparse: float = 1.0
    use_reranker: bool = True
    reranker_model: str | None = DEFAULT_RERANKER
    rerank_top_n: int = DEFAULT_RERANK_TOP_N
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT
    top_k_retrieve: int = DEFAULT_TOP_K_RETRIEVE

    # Optional free-form metadata (e.g. "baseline", notes)
    notes: dict[str, str] = field(default_factory=dict)

    # ---- Derived identifiers -----------------------------------------

    @property
    def dense_tag(self) -> str:
        # "BAAI/bge-small-en-v1.5" -> "bge-small"
        name = self.dense_model.split("/")[-1]
        return re.sub(r"-en-v[0-9.]+$", "", name)

    @property
    def chunker_tag(self) -> str:
        if self.chunk_strategy == "semantic":
            return (
                f"semantic-{self.semantic_breakpoint_type}"
                f"-{int(self.semantic_breakpoint_amount)}"
            )
        return f"{self.chunk_strategy}-{self.chunk_size}-{self.chunk_overlap}"

    @property
    def collection_name(self) -> str:
        """One collection per unique (embedder × chunker) pair."""
        return f"nfcorpus__{self.dense_tag}__{self.chunker_tag}"

    @property
    def run_filename(self) -> str:
        return f"{self.tag}.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dense_tag"] = self.dense_tag
        d["chunker_tag"] = self.chunker_tag
        d["collection_name"] = self.collection_name
        return d


# ---------------------------------------------------------------------------
# Ablation 1 — Retrieval mode
# ---------------------------------------------------------------------------


def ablation_1_retrieval_mode() -> list[RetrievalConfig]:
    """BM25-only vs Dense-only vs Hybrid(RRF, equal) vs Hybrid+rerank.

    Everything else held at defaults; a single collection covers all four.
    """
    base: dict = dict(
        ablation="retrieval_mode",
        dense_model=DEFAULT_DENSE_MODEL,
        chunk_strategy=DEFAULT_CHUNK_STRATEGY,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    return [
        RetrievalConfig(
            **base,
            tag="abl1_retrieval_mode__bm25_only",
            weight_dense=0.0,
            weight_sparse=1.0,
            use_reranker=False,
            reranker_model=None,
        ),
        RetrievalConfig(
            **base,
            tag="abl1_retrieval_mode__dense_only",
            weight_dense=1.0,
            weight_sparse=0.0,
            use_reranker=False,
            reranker_model=None,
        ),
        RetrievalConfig(
            **base,
            tag="abl1_retrieval_mode__hybrid_rrf_equal",
            weight_dense=1.0,
            weight_sparse=1.0,
            use_reranker=False,
            reranker_model=None,
        ),
        RetrievalConfig(
            **base,
            tag="abl1_retrieval_mode__hybrid_rrf_rerank",
            weight_dense=1.0,
            weight_sparse=1.0,
            use_reranker=True,
            reranker_model=DEFAULT_RERANKER,
        ),
    ]


# ---------------------------------------------------------------------------
# Ablation 2 — Embedder size (hybrid + rerank, chunker fixed)
# ---------------------------------------------------------------------------


def ablation_2_embedder_size() -> list[RetrievalConfig]:
    """Small vs base vs large, hybrid+rerank, recursive-512/50.

    Each creates its own collection (dense dim differs).
    """
    models = [
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    ]
    out: list[RetrievalConfig] = []
    for model in models:
        short = model.split("/")[-1].replace("-en-v1.5", "")
        out.append(
            RetrievalConfig(
                tag=f"abl2_embedder_size__{short}",
                ablation="embedder_size",
                dense_model=model,
                chunk_strategy=DEFAULT_CHUNK_STRATEGY,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                weight_dense=1.0,
                weight_sparse=1.0,
                use_reranker=True,
                reranker_model=DEFAULT_RERANKER,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Ablation 3 — Chunker (best embedder from Ablation 2, hybrid + rerank)
# ---------------------------------------------------------------------------


def ablation_3_chunker(best_embedder: str = DEFAULT_DENSE_MODEL) -> list[RetrievalConfig]:
    """Fixed/recursive at several sizes + semantic at two percentiles.

    ``best_embedder`` should be whatever Ablation 2 picked as the winner;
    defaults to the repo-wide default so this can be run stand-alone.
    Each config gets its own collection (the chunker differs).
    """
    short = best_embedder.split("/")[-1].replace("-en-v1.5", "")
    # (strategy, size, overlap) for "sized" chunkers
    sized = [
        ("fixed", 256, 25),
        ("fixed", 512, 50),
        ("fixed", 1024, 100),
        ("recursive", 256, 25),
        ("recursive", 512, 50),
    ]
    semantic_variants = [
        ("percentile", 90.0),
        ("percentile", 95.0),
    ]

    out: list[RetrievalConfig] = []
    for strat, size, overlap in sized:
        out.append(
            RetrievalConfig(
                tag=f"abl3_chunker__{short}__{strat}-{size}-{overlap}",
                ablation="chunker",
                dense_model=best_embedder,
                chunk_strategy=strat,
                chunk_size=size,
                chunk_overlap=overlap,
                weight_dense=1.0,
                weight_sparse=1.0,
                use_reranker=True,
                reranker_model=DEFAULT_RERANKER,
            )
        )
    for btype, bamount in semantic_variants:
        out.append(
            RetrievalConfig(
                tag=f"abl3_chunker__{short}__semantic-{btype}-{int(bamount)}",
                ablation="chunker",
                dense_model=best_embedder,
                chunk_strategy="semantic",
                semantic_breakpoint_type=btype,
                semantic_breakpoint_amount=bamount,
                weight_dense=1.0,
                weight_sparse=1.0,
                use_reranker=True,
                reranker_model=DEFAULT_RERANKER,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Ablation 4 — Reranker
# ---------------------------------------------------------------------------


def ablation_4_reranker(
    dense_model: str = DEFAULT_DENSE_MODEL,
    chunk_strategy: ChunkStrategy = DEFAULT_CHUNK_STRATEGY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RetrievalConfig]:
    """No rerank vs bge-reranker-base vs bge-reranker-large.

    Uses the "best retrieval config" (hybrid RRF equal weights); defaults
    for the (embedder × chunker) pair can be overridden via args, same
    collection is reused across the three variants.
    """
    short = dense_model.split("/")[-1].replace("-en-v1.5", "")
    base_kwargs = dict(
        ablation="reranker",
        dense_model=dense_model,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        weight_dense=1.0,
        weight_sparse=1.0,
    )
    return [
        RetrievalConfig(
            **base_kwargs,
            tag=f"abl4_reranker__{short}__none",
            use_reranker=False,
            reranker_model=None,
        ),
        RetrievalConfig(
            **base_kwargs,
            tag=f"abl4_reranker__{short}__bge-base",
            use_reranker=True,
            reranker_model="BAAI/bge-reranker-base",
        ),
        RetrievalConfig(
            **base_kwargs,
            tag=f"abl4_reranker__{short}__bge-large",
            use_reranker=True,
            reranker_model="BAAI/bge-reranker-large",
        ),
    ]


# ---------------------------------------------------------------------------
# Ablation 5 — Rerank depth (RERANK_TOP_N)
# ---------------------------------------------------------------------------


def ablation_5_rerank_top_n(
    dense_model: str = DEFAULT_DENSE_MODEL,
    chunk_strategy: ChunkStrategy = DEFAULT_CHUNK_STRATEGY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RetrievalConfig]:
    """How many candidates does the cross-encoder see?

    ``prefetch_limit = 2 × rerank_top_n`` per spec, so the reranker
    always has a safety margin of non-reranked candidates to draw from.
    """
    values = [10, 20, 50]
    out: list[RetrievalConfig] = []
    for n in values:
        out.append(
            RetrievalConfig(
                tag=f"abl5_rerank_top_n__n{n}",
                ablation="rerank_top_n",
                dense_model=dense_model,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                weight_dense=1.0,
                weight_sparse=1.0,
                use_reranker=True,
                reranker_model=DEFAULT_RERANKER,
                rerank_top_n=n,
                prefetch_limit=max(DEFAULT_PREFETCH_LIMIT, 2 * n),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Ablation 6 — Fusion weights (w_dense, w_sparse)
# ---------------------------------------------------------------------------


def ablation_6_fusion_weights(
    dense_model: str = DEFAULT_DENSE_MODEL,
    chunk_strategy: ChunkStrategy = DEFAULT_CHUNK_STRATEGY,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RetrievalConfig]:
    """Equal vs dense-biased vs sparse-biased weighted RRF."""
    weights = [
        (1.0, 1.0, "equal"),
        (0.75, 0.25, "dense_heavy"),
        (0.25, 0.75, "sparse_heavy"),
    ]
    out: list[RetrievalConfig] = []
    for w_d, w_s, label in weights:
        out.append(
            RetrievalConfig(
                tag=f"abl6_weights__{label}",
                ablation="fusion_weights",
                dense_model=dense_model,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                weight_dense=w_d,
                weight_sparse=w_s,
                use_reranker=True,
                reranker_model=DEFAULT_RERANKER,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Registry + lookup
# ---------------------------------------------------------------------------


ABLATION_REGISTRY: dict[str, Callable[..., list["RetrievalConfig"]]] = {
    "1": ablation_1_retrieval_mode,
    "2": ablation_2_embedder_size,
    "3": ablation_3_chunker,
    "4": ablation_4_reranker,
    "5": ablation_5_rerank_top_n,
    "6": ablation_6_fusion_weights,
    "retrieval_mode": ablation_1_retrieval_mode,
    "embedder_size": ablation_2_embedder_size,
    "chunker": ablation_3_chunker,
    "reranker": ablation_4_reranker,
    "rerank_top_n": ablation_5_rerank_top_n,
    "fusion_weights": ablation_6_fusion_weights,
}


def all_ablations() -> list[RetrievalConfig]:
    """Every config across all six ablations, de-duplicated by tag."""
    seen: set[str] = set()
    out: list[RetrievalConfig] = []
    for ab_id in ("1", "2", "3", "4", "5", "6"):
        for cfg in ABLATION_REGISTRY[ab_id]():
            if cfg.tag in seen:
                continue
            seen.add(cfg.tag)
            out.append(cfg)
    return out
