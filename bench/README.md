# `bench/` — Retrieval latency benchmark

White-box latency harness for the hybrid retrieval pipeline. Where
`eval/` cares about relevance, this cares about **where the time goes**.

## What it measures

Every query is instrumented stage-by-stage, mirroring
`retrieval.search.HybridSearchEngine.search`:

| stage          | what it times                                             |
| -------------- | --------------------------------------------------------- |
| `dense_embed`  | FastEmbed dense query embedding                           |
| `dense_query`  | Qdrant `query_points` round-trip (dense named vector)     |
| `sparse_embed` | FastEmbed BM25 query embedding                            |
| `sparse_query` | Qdrant `query_points` round-trip (sparse named vector)    |
| `fuse`         | Weighted RRF in Python                                    |
| `rerank`       | Cross-encoder scoring of top `rerank_top_n` (if enabled)  |
| `total`        | Sum of the above                                          |

For each stage we report `mean`, `p50`, `p95`, `p99`, `max` — separately
for **cold** and **warm** cache regimes (see below).

## Cold vs warm methodology

Model loading is always done once, up front, so the benchmark measures
**retrieval latency**, not first-import cost.

1. **Warmup** (`--warmup`, default 5 queries) — run throwaway queries
   to exercise Qdrant's process-level caches with unrelated inputs.
   Not measured.
2. **Cold** — each test query is run **exactly once**. This is the
   first-hit latency a real user would see: embedder ready, but Qdrant's
   HNSW node cache / BM25 postings / result cache haven't seen this
   particular query yet.
3. **Warm** (`--warm-iters`, default 10) — each test query is run an
   additional *N* times, back-to-back, feeding the warm distribution.

Why this split? Cold-p95 tells you what latency a new user at peak hour
sees. Warm-p50 tells you what your steady-state backend produces under
load. Both matter and they're rarely the same number.

We don't drop OS page caches from Python — that'd require root and make
the harness non-portable. `--warmup` is the closest portable analog.

## Usage

```bash
# Default: 50 NFCorpus test queries, 5 warmup, 10 warm iterations/query.
uv run ir3-bench run

# Smaller smoke test.
uv run ir3-bench run --n-queries 20 --warm-iters 5

# Compare with / without the reranker — two files, same queries.
uv run ir3-bench run --no-reranker -o bench/reports/latency_no_rerank.json
uv run ir3-bench run            -o bench/reports/latency_rerank.json

# BM25-only or dense-only profiles.
uv run ir3-bench run --weight-dense 0 --weight-sparse 1 -o bench/reports/latency_bm25.json
uv run ir3-bench run --weight-dense 1 --weight-sparse 0 -o bench/reports/latency_dense.json

# Re-render an existing JSON report (no queries re-run).
uv run ir3-bench render bench/reports/latency.json
```

### Custom queries

```bash
uv run ir3-bench run --queries-file my_queries.txt --n-queries 100
```

One query per line. Queries are shuffled with `--seed` (default 0).

## Outputs

Every `ir3-bench run` writes:

- `bench/reports/latency.json` — raw per-query samples + aggregated
  cold/warm stats. Everything downstream (plotting, regression tests) can
  read this without re-running retrieval.
- `bench/reports/latency.md` — human-readable summary: a cold/warm
  per-stage table and a "warm breakdown" table showing each stage's
  share of total latency.
- Rich console table, printed inline.

## Caveats

- Qdrant is benchmarked over HTTP by default (`prefer_grpc=False`, same as
  production). If you switch to gRPC the numbers will shift — mostly
  `{dense,sparse}_query`.
- `total` is the **sum of measured stages**, not an outer `perf_counter`
  around the whole thing. This is intentional: it keeps the breakdown
  chart's percentages honest.
- The dense and sparse branches run *sequentially* (matching the current
  production engine), so `total ≈ dense_embed + dense_query + sparse_embed
  + sparse_query + fuse + rerank`. If the engine ever moves to parallel
  component retrieval, this module should measure the gathered wall-clock
  instead.
