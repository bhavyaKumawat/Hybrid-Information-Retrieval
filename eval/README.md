# IR3 — Evaluation Harness

Offline IR evaluation over **NFCorpus** (BEIR) with real `qrels/test.tsv`.
Runs six ablations — one per design decision — and produces a single
CSV + Markdown summary.

---

## What gets measured

| Cutoff | Metrics |
|---|---|
| `k=5`  | precision@5, recall@5, NDCG@5 |
| `k=10` | precision@10, recall@10, NDCG@10 |

- Graded relevance is preserved from NFCorpus qrels (score ∈ {1, 2}), so
  **NDCG is graded, not binary**.
- Library: [`ranx`](https://amenra.github.io/ranx/). Reads BEIR-format
  qrels directly (`{qid: {did: score}}`).
- Split: **`BeIR/nfcorpus-qrels`, split=`test`** for the headline numbers.
- Number of queries per ablation: **all** test queries (~323).

### Chunk → document aggregation

NFCorpus qrels are **per-document**. The retriever returns **per-chunk**
hits. We bridge with **max-pool**: `score(doc) = max(score(chunks of doc))`.
This matches what BEIR reports for chunked setups; the choice is in one
place (`eval/aggregate.py`) so it's swappable.

---

## The six ablations

| # | Ablation | Axis | Fixed at |
|---|---|---|---|
| 1 | Retrieval mode | BM25-only / Dense-only / Hybrid(RRF, equal) / Hybrid+rerank | bge-small, recursive-512/50 |
| 2 | Embedder size | bge-{small, base, large} | recursive-512/50, hybrid+rerank |
| 3 | Chunker | fixed/recursive at 256/512/1024, semantic@{90,95} | best embedder from (2), hybrid+rerank |
| 4 | Reranker | none / bge-reranker-base / bge-reranker-large | best retrieval config |
| 5 | `rerank_top_n` | 10, 20, 50 (prefetch_limit = 2×n) | default collection |
| 6 | Fusion weights | (1,1) / (0.75,0.25) / (0.25,0.75) | hybrid + rerank |

**Query-time vs collection-dependent.** Ablations 1, 4, 5, 6 are
query-time only: they reuse a single collection. Ablations 2 and 3
build a **new Qdrant collection per unique `(embedder × chunker)`
pair** — their dimensions, point IDs, and chunk boundaries differ.

### One collection per (embedder × chunker) pair

Collections are named

```
nfcorpus__<embedder-short>__<chunker-tag>
```

e.g. `nfcorpus__bge-base__recursive-512-50`,
`nfcorpus__bge-small__semantic-percentile-90`.

A **single shared SQLite manifest** (`data/manifest.db`) tracks per-
collection ingestion state. Its primary keys include `collection`, so
re-running ingestion on any collection is idempotent — no embedding,
no Qdrant writes, unless something actually changed.

---

## Run-result caching

Each run produces one JSON file in `eval/runs/`:

```
eval/runs/<tag>.json
```

The JSON stores the full `config` + the top-100 chunk hits per query,
with every per-component score (`bm25`, `semantic`, `fused_rrf`,
`reranker`, `final`). That means:

- Changing a metric (e.g. switching to recall@20) does **not** require
  re-running retrieval — just re-score.
- Changing the chunk→doc aggregation (e.g. max-pool → mean-pool) does
  **not** require re-running retrieval either.

The runner treats an existing run JSON as a cache hit by default; pass
`--overwrite` to force a re-run.

---

## Commands

```bash
# Enumerate every config that would run (no side effects)
uv run ir3-eval list
uv run ir3-eval list --ablation 1

# Run one ablation (builds collections + runs every query in the test split)
uv run ir3-eval run 1
uv run ir3-eval run retrieval_mode

# Run one specific config inside an ablation
uv run ir3-eval run 1 --tag abl1_retrieval_mode__bm25_only

# Run everything (all 6 ablations). Idempotent: cached runs are skipped.
uv run ir3-eval run-all

# Re-score cached runs + dump a CSV + Markdown summary
uv run ir3-eval score      # quick table in the terminal
uv run ir3-eval report     # writes eval/reports/summary.{csv,md}
```

### Dependencies

Qdrant must be running (`docker compose up -d`), and `ranx` is added as
a project dependency — `uv sync` installs it with the rest.

---

## Ordering recommendation

Run in this order, because later ablations optionally depend on
earlier ones' "best" result:

1. `run 1` → pick the best retrieval mode (usually hybrid+rerank).
2. `run 2` → pick the best embedder.
3. `run 3 --best-embedder <winner-from-2>` → pick the best chunker.
4. `run 4` → pick the best reranker.
5. `run 5` → pick the best `rerank_top_n`.
6. `run 6` → pick the best fusion weights.
7. `report` → produce `eval/reports/summary.{csv,md}`.

Or just `run-all` and accept the defaults; every config is
independent, the defaults are the same "sensible baseline" for all of
them.

---

## Directory layout

```
eval/
├── README.md              # this file
├── configs.py             # RetrievalConfig + 6 ablation builders
├── qrels.py               # BeIR/nfcorpus-qrels loader (test split)
├── aggregate.py           # max-pool chunks → docs
├── metrics.py             # ranx wrapper, P/R/NDCG @ {5, 10}
├── runner.py              # ingest-on-demand + query loop + JSON cache
├── report.py              # summary.csv + summary.md
├── cli.py                 # Typer CLI (`ir3-eval`)
├── runs/                  # <tag>.json per config (retrieval cache)
└── reports/
    ├── summary.csv
    └── summary.md
```
