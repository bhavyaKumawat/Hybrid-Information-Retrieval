"""End-to-end latency benchmark for the hybrid retrieval pipeline.

Why a separate harness instead of re-using :mod:`eval.runner`?
The eval runner cares about *relevance* (nDCG / recall); it calls
``engine.search()`` as a black box and the per-request ``timings_ms`` it
records lump all of retrieval into a single ``retrieve_ms`` bucket. Here we
want the opposite: a white-box view of each stage so we can answer
"where is the time going?".

Concretely the benchmark instruments these stages independently, around
the exact code paths :class:`HybridSearchEngine` uses:

* ``dense_embed``    — FastEmbed dense query embedding
* ``dense_query``    — Qdrant ``query_points`` round-trip via the dense vector
* ``sparse_embed``   — FastEmbed BM25 query embedding
* ``sparse_query``   — Qdrant ``query_points`` round-trip via the sparse vector
* ``fuse``           — weighted RRF in Python
* ``rerank``         — cross-encoder scoring (optional)
* ``total``          — sum of the above (end-to-end)

Cold vs warm methodology
------------------------
Each query is measured in two regimes:

* **Cold** — the first (and only first) time this query is run, after a
  dedicated warmup phase has loaded every model and exercised unrelated
  warmup queries against Qdrant. This captures the first-hit latency a
  real user would see: embedder is ready, but Qdrant's in-memory caches
  (HNSW node pages, BM25 term postings, result cache) haven't seen *this*
  query yet.
* **Warm** — the same queries run ``warm_iters`` additional times back-to-back.
  All distributions skip the first iteration (which is the cold one) so
  the warm histogram is uncontaminated.

We don't touch OS page caches from Python — an end-user benchmark
shouldn't need root. ``--warmup`` lets you exercise caches with throwaway
queries before the cold phase; that's the closest portable analog.
"""

from __future__ import annotations

import logging
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from retrieval.config import Settings
from retrieval.embeddings import DenseEmbedder, SparseEmbedder
from retrieval.qdrant_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from retrieval.reranker import CrossEncoderReranker
from retrieval.search import _weighted_rrf_fuse

log = logging.getLogger(__name__)


# Canonical stage order. Anything iterating stages should use this so
# JSON output and console tables are consistent.
STAGES: tuple[str, ...] = (
    "dense_embed",
    "dense_query",
    "sparse_embed",
    "sparse_query",
    "fuse",
    "rerank",
    "total",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchConfig:
    """Everything that parameterises a latency run.

    Kept separate from :class:`retrieval.config.Settings` because most of
    those fields (ingestion paths, chunker, API host) are irrelevant here
    and pre-populating them from env would just be noise in the report.
    """

    collection: str
    dense_model: str
    sparse_model: str
    reranker_model: str
    use_reranker: bool = True
    weight_dense: float = 1.0
    weight_sparse: float = 1.0
    rrf_k: int = 60
    top_k: int = 5
    rerank_top_n: int = 20
    prefetch_limit: int = 50
    warmup: int = 5
    warm_iters: int = 10
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "dense_model": self.dense_model,
            "sparse_model": self.sparse_model,
            "reranker_model": self.reranker_model if self.use_reranker else None,
            "use_reranker": self.use_reranker,
            "weight_dense": self.weight_dense,
            "weight_sparse": self.weight_sparse,
            "rrf_k": self.rrf_k,
            "top_k": self.top_k,
            "rerank_top_n": self.rerank_top_n,
            "prefetch_limit": self.prefetch_limit,
            "warmup": self.warmup,
            "warm_iters": self.warm_iters,
            "seed": self.seed,
        }


@dataclass
class QueryTiming:
    """Per-query, per-iteration latencies (milliseconds)."""

    query: str
    iteration: int
    regime: str  # "cold" | "warm"
    stages: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "iteration": self.iteration,
            "regime": self.regime,
            "stages_ms": self.stages,
        }


@dataclass
class StageStats:
    """Aggregated latency stats for one stage, one regime."""

    n: int
    mean_ms: float
    stdev_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> "StageStats":
        if not samples:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            n=len(samples),
            mean_ms=round(statistics.fmean(samples), 3),
            stdev_ms=round(statistics.pstdev(samples), 3) if len(samples) > 1 else 0.0,
            min_ms=round(min(samples), 3),
            p50_ms=round(_percentile(samples, 50), 3),
            p95_ms=round(_percentile(samples, 95), 3),
            p99_ms=round(_percentile(samples, 99), 3),
            max_ms=round(max(samples), 3),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_ms": self.mean_ms,
            "stdev_ms": self.stdev_ms,
            "min_ms": self.min_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


@dataclass
class BenchReport:
    """Full benchmark output: raw samples + aggregated cold/warm stats."""

    config: BenchConfig
    num_queries: int
    timings: list[QueryTiming]
    cold_stats: dict[str, StageStats]
    warm_stats: dict[str, StageStats]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "num_queries": self.num_queries,
            "cold": {k: v.to_dict() for k, v in self.cold_stats.items()},
            "warm": {k: v.to_dict() for k, v in self.warm_stats.items()},
            "timings": [t.to_dict() for t in self.timings],
        }


# ---------------------------------------------------------------------------
# Core timing loop
# ---------------------------------------------------------------------------


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def time_single_query(
    *,
    client: QdrantClient,
    dense: DenseEmbedder,
    sparse: SparseEmbedder,
    reranker: CrossEncoderReranker | None,
    cfg: BenchConfig,
    query: str,
) -> dict[str, float]:
    """Run one query through the full pipeline and return per-stage ms.

    Mirrors :meth:`retrieval.search.HybridSearchEngine.search` stage-for-stage
    so numbers are comparable to production, but times each component
    individually. The retrieval call itself is not parallelised here
    (matching the current production engine), so per-stage ms do sum
    cleanly into ``total``.
    """
    stages: dict[str, float] = dict.fromkeys(STAGES, 0.0)

    # ---- Dense branch (skip cleanly when weight==0) -----------------------
    dense_hits: list[qm.ScoredPoint] = []
    if cfg.weight_dense > 0:
        t = _now_ms()
        dvec = dense.embed_query(query)
        stages["dense_embed"] = _now_ms() - t

        t = _now_ms()
        dres = client.query_points(
            collection_name=cfg.collection,
            query=dvec,
            using=DENSE_VECTOR_NAME,
            limit=cfg.prefetch_limit,
            with_payload=True,
        )
        stages["dense_query"] = _now_ms() - t
        dense_hits = list(dres.points)

    # ---- Sparse branch ---------------------------------------------------
    sparse_hits: list[qm.ScoredPoint] = []
    if cfg.weight_sparse > 0:
        t = _now_ms()
        svec = sparse.embed_query(query)
        stages["sparse_embed"] = _now_ms() - t

        t = _now_ms()
        sres = client.query_points(
            collection_name=cfg.collection,
            query=svec,
            using=SPARSE_VECTOR_NAME,
            limit=cfg.prefetch_limit,
            with_payload=True,
        )
        stages["sparse_query"] = _now_ms() - t
        sparse_hits = list(sres.points)

    # ---- Fuse ------------------------------------------------------------
    t = _now_ms()
    candidates = _weighted_rrf_fuse(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        w_dense=cfg.weight_dense,
        w_sparse=cfg.weight_sparse,
        k=cfg.rrf_k,
    )
    stages["fuse"] = _now_ms() - t

    # ---- Rerank (optional) ----------------------------------------------
    if cfg.use_reranker and reranker is not None and candidates:
        t = _now_ms()
        top = candidates[: cfg.rerank_top_n]
        texts = [c.payload.get("chunk_text", "") for c in top]
        try:
            reranker.rerank(query, top, texts)
        except Exception as exc:  # same graceful degradation as the engine
            log.warning("bench.rerank_failed err=%r", exc)
        stages["rerank"] = _now_ms() - t

    # ``total`` is the sum of the measured stages rather than a separate
    # perf_counter around everything; otherwise small scheduler gaps between
    # stages would make ``total`` slightly larger than the sum of parts and
    # confuse readers of the breakdown chart.
    stages["total"] = sum(
        stages[s] for s in STAGES if s != "total"
    )
    return stages


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    client: QdrantClient,
    settings: Settings,
    queries: list[str],
    cfg: BenchConfig,
    progress_cb: Any = None,
) -> BenchReport:
    """Execute the cold/warm benchmark and return a :class:`BenchReport`.

    ``progress_cb`` is an optional ``(phase: str, done: int, total: int)``
    callback so a Rich progress bar can hook in without this module
    depending on a UI framework.
    """
    rng = random.Random(cfg.seed)
    queries = list(queries)
    rng.shuffle(queries)

    # Build the models. We hold one instance for the whole run so FastEmbed's
    # ONNX session is reused (which is exactly what happens in the API
    # process). Warming them first keeps model-load cost out of the cold
    # distribution — we're benchmarking retrieval, not first-import latency.
    dense = DenseEmbedder(cfg.dense_model)
    sparse = SparseEmbedder(cfg.sparse_model)
    reranker: CrossEncoderReranker | None = None
    if cfg.use_reranker:
        reranker = CrossEncoderReranker(cfg.reranker_model)

    dense.warmup()
    sparse.warmup()
    if reranker is not None:
        reranker.warmup()

    timings: list[QueryTiming] = []

    # ---- Warmup phase (not measured) -------------------------------------
    # Use a disjoint slice of queries so the cold phase isn't accidentally
    # warmed by identical inputs. If we don't have enough distinct queries
    # we skip the dedicated warmup and accept a slightly hotter cold
    # distribution.
    warmup_n = min(cfg.warmup, max(len(queries) - 1, 0))
    warmup_queries = queries[:warmup_n]
    cold_queries = queries[warmup_n:]
    if not cold_queries:
        raise ValueError(
            f"Need at least warmup+1 queries; got {len(queries)} with warmup={cfg.warmup}"
        )

    total_work = warmup_n + len(cold_queries) * (1 + cfg.warm_iters)
    done = 0

    for q in warmup_queries:
        time_single_query(
            client=client, dense=dense, sparse=sparse,
            reranker=reranker, cfg=cfg, query=q,
        )
        done += 1
        if progress_cb:
            progress_cb("warmup", done, total_work)

    # ---- Cold phase: each query measured exactly once --------------------
    for q in cold_queries:
        stages = time_single_query(
            client=client, dense=dense, sparse=sparse,
            reranker=reranker, cfg=cfg, query=q,
        )
        timings.append(QueryTiming(query=q, iteration=0, regime="cold", stages=stages))
        done += 1
        if progress_cb:
            progress_cb("cold", done, total_work)

    # ---- Warm phase: repeat each query warm_iters more times -------------
    # Iteration 0 is the cold one (already recorded); iterations 1..N are
    # warm. Outer loop is by-iteration rather than by-query so each query
    # gets roughly the same spacing between repeats — closer to steady-state
    # throughput than a hot inner loop that would keep one query entirely
    # resident.
    for i in range(1, cfg.warm_iters + 1):
        for q in cold_queries:
            stages = time_single_query(
                client=client, dense=dense, sparse=sparse,
                reranker=reranker, cfg=cfg, query=q,
            )
            timings.append(QueryTiming(query=q, iteration=i, regime="warm", stages=stages))
            done += 1
            if progress_cb:
                progress_cb("warm", done, total_work)

    cold_stats = _aggregate(timings, "cold")
    warm_stats = _aggregate(timings, "warm")

    _ = settings  # settings threaded through for callers that want to persist it next to the report
    return BenchReport(
        config=cfg,
        num_queries=len(cold_queries),
        timings=timings,
        cold_stats=cold_stats,
        warm_stats=warm_stats,
    )


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _aggregate(timings: list[QueryTiming], regime: str) -> dict[str, StageStats]:
    out: dict[str, StageStats] = {}
    for stage in STAGES:
        samples = [
            t.stages.get(stage, 0.0)
            for t in timings
            if t.regime == regime and stage in t.stages
        ]
        out[stage] = StageStats.from_samples(samples)
    return out


def _percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolated percentile, matching numpy's default.

    We roll our own to avoid pulling numpy into the benchmark's hot path
    (the benchmark already depends on it transitively via fastembed, but
    keeping this module import-light means `ir3-bench --help` stays fast).
    """
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    ordered = sorted(samples)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac
