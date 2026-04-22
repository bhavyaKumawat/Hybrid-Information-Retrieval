# IR3 — Production-Grade Hybrid Retrieval Platform

A hybrid retrieval system over **NFCorpus** (BEIR) that composes:

- **BM25** via Qdrant native sparse vectors (FastEmbed `Qdrant/bm25`)
- **Dense semantic search** via FastEmbed `BAAI/bge-{small,base,large}-en-v1.5`
- **Weighted Reciprocal Rank Fusion** (client-side, so component scores survive)
- **Cross-encoder reranking** via FastEmbed `BAAI/bge-reranker-{base,large}`
- **Incremental ingestion** with a two-level hash design and a SQLite manifest
- **Metadata filtering** on `source`, `date`, `doc_id`, with Qdrant payload indexes

Every stage is composable and disableable. The `POST /search` endpoint returns the top-k hits with a full score breakdown so you can see exactly what each component contributed.

---

## Stack

| Concern | Choice |
|---|---|
| Vector DB | Qdrant (Docker, local) |
| Dense embeddings | `fastembed.TextEmbedding` (BAAI bge-{small,base,large}-en-v1.5) |
| Sparse embeddings | `fastembed.SparseTextEmbedding` (`Qdrant/bm25`) |
| Reranker | `fastembed.rerank.cross_encoder.TextCrossEncoder` (bge-reranker-{base,large}) |
| Chunking (fixed/recursive) | `langchain-text-splitters` |
| Chunking (semantic) | `langchain-experimental.SemanticChunker` |
| State store | SQLite (`data/manifest.db`) |
| Config | `pydantic-settings` + `.env` |
| API | FastAPI |
| CLI | Typer |
| Packaging | `uv` + `hatchling` |

---

## Quick start

### 1. Start Qdrant

```bash
docker compose up -d
```

### 2. Install dependencies

```bash
cp .env.example .env
uv sync
```

### 3. Ingest NFCorpus

```bash
uv run ir3 ingest
```

Re-running `ir3 ingest` performs zero embeddings and zero Qdrant writes unless the source docs, chunker config, or model changed (see **Incremental ingestion** below).

You can cap the dataset for quick smoke tests:

```bash
uv run ir3 ingest --max-docs 100
```

### 4. Run the API

```bash
uv run ir3 serve
# -> http://localhost:8000/docs
```

### 5. Search

```bash
curl -s -X POST http://localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{
        "query": "statins and muscle pain",
        "top_k": 5,
        "filters": {
          "source": "pubmed",
          "date": {"gte": "2015-01-01T00:00:00Z"}
        }
      }' | jq .
```

Or from the CLI:

```bash
uv run ir3 search "statins and muscle pain" --top-k 5
```

---

## Architecture

```
             NFCorpus (BEIR)
                  │
                  ▼
          ┌───────────────┐
          │  Chunker      │  RecursiveChunker | FixedSizeChunker | SemanticChunker
          └───────┬───────┘  (langchain-text-splitters / langchain-experimental)
                  │
                  ▼
     ┌──────────────────────────┐
     │  Two-level hashing       │  content_hash, chunk_hash, chunker_config_hash
     │  + SQLite manifest       │
     └──────────┬───────────────┘
                │
                ▼
 ┌─────────────────────────────────────────────┐
 │  Qdrant collection (named vectors)          │
 │    • "dense"  : VectorParams(cosine, dim)   │
 │    • "bm25"   : SparseVectorParams(IDF)     │
 │  Payload indexes: source (keyword),         │
 │                   doc_id (keyword),         │
 │                   date   (datetime)         │
 └──────────┬──────────────────────────────────┘
            │
   ┌────────┴─────────┐
   │ 1. Retrieve (||) │  query_points(dense) + query_points(sparse)
   │                  │  filters applied server-side on each call
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ 2. Weighted RRF  │  score = Σ w_i / (k + rank_i)
   │   (client-side)  │  preserves per-component scores
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ 3. Cross-encoder │  reranks top N, keeps top K
   │   (optional)     │
   └────────┬─────────┘
            │
            ▼
   POST /search response with full score breakdown
```

### Why client-side RRF?

Qdrant's built-in RRF (via `FusionQuery`) fuses prefetches server-side and returns one fused score per hit — the component scores are lost. The assignment specifically asks for BM25 vs semantic vs reranker contributions **per result**. To surface those, we run the two prefetches as separate `query_points` calls, keep the component scores, then fuse with weighted RRF client-side. This also gives us tunable per-component weights, which Qdrant's native RRF does not expose.

Filters are still applied at the Prefetch level (via `query_filter` on each `query_points` call), which is what the Qdrant docs recommend.

---

## Incremental ingestion

Two hashes drive the short-circuit logic:

- **`content_hash` (document-level)** — `sha256(title + "\n" + text)`. If this is unchanged AND the chunker config and models are unchanged, the whole document is skipped: no splitting, no embedding, no writes.
- **`chunk_hash` (chunk-level)** — `sha256(chunker_config_hash + chunk_text)`. Used per-chunk: if a chunk at a given index already exists with the same chunk_hash and the same dense model, we re-use the existing Qdrant point and skip re-embedding.

Both are written into every point's payload as `content_hash` / `chunk_hash` / `chunker_config_hash`, alongside model names and `ingested_at`.

The manifest (`data/manifest.db`, SQLite) keeps two tables:

- `documents(doc_id PK, content_hash, chunker_config_hash, dense_model, sparse_model, chunk_count, ingested_at)`
- `chunks(doc_id, chunk_index PK, chunk_hash, point_id, dense_model, ingested_at)`

**Reconciliation.** When a changed document produces fewer chunks than before, the manifest is used to identify orphan chunks, which are deleted from Qdrant *and* the manifest. Re-running on identical inputs is a strict no-op.

**Idempotency proof.** After running `ir3 ingest` twice in a row, the second run reports `chunks_embedded=0`, `chunks_reused=<all>`, `documents_skipped=<all>`.

---

## Payload schema

Exactly what the assignment specifies:

```json
{
  "chunk_text": "...",
  "chunk_index": 3,
  "chunk_total": 47,

  "doc_id": "MED-1234",
  "doc_title": "Effects of ...",
  "source_url": "http://www.ncbi.nlm.nih.gov/pubmed/25329299",

  "source": "pubmed",
  "date": "2018-06-15T00:00:00Z",

  "content_hash": "sha256:...",
  "chunk_hash": "sha256:...",
  "chunker_config_hash": "sha256:...",
  "dense_model": "BAAI/bge-small-en-v1.5",
  "sparse_model": "Qdrant/bm25",
  "ingested_at": "2026-04-21T12:00:00Z"
}
```

### Note on `date`

NFCorpus doesn't ship a publication date. So the filter can be exercised end-to-end, we synthesise a deterministic date per `doc_id` (seeded from its SHA-256, spread 2010–2022). This is explicitly a synthetic demo value, not a real publication date. The synthesis lives in `src/retrieval/datasets/nfcorpus.py` — swap it out for a real source if you have one.

---

## REST API

### `POST /search`

Request body (all fields except `query` are optional — defaults from `.env`):

```json
{
  "query": "statins and muscle pain",
  "top_k": 5,
  "rerank_top_n": 20,
  "prefetch_limit": 50,
  "weight_dense": 1.0,
  "weight_sparse": 1.0,
  "use_reranker": true,
  "filters": {
    "source": "pubmed",
    "date": {"gte": "2015-01-01T00:00:00Z", "lte": "2022-12-31T23:59:59Z"},
    "doc_ids": ["MED-123", "MED-456"]
  }
}
```

Response (truncated):

```json
{
  "query": "statins and muscle pain",
  "hits": [
    {
      "doc_id": "MED-...",
      "chunk_index": 0,
      "chunk_total": 3,
      "chunk_text": "...",
      "doc_title": "...",
      "source": "pubmed",
      "source_url": "http://...",
      "date": "2018-06-15T00:00:00Z",
      "scores": {
        "bm25": 12.34,
        "semantic": 0.812,
        "fused_rrf": 0.0328,
        "reranker": 4.21,
        "final": 4.21,
        "bm25_rank": 2,
        "semantic_rank": 1
      }
    }
  ],
  "debug": {
    "dense_model": "BAAI/bge-small-en-v1.5",
    "sparse_model": "Qdrant/bm25",
    "reranker_model": "BAAI/bge-reranker-base",
    "use_reranker": true,
    "fusion": "weighted_rrf",
    "rrf_k": 60,
    "weights": {"dense": 1.0, "sparse": 1.0},
    "prefetch_limit": 50,
    "rerank_top_n": 20,
    "timings_ms": {"retrieve_ms": 42.1, "fuse_ms": 0.3, "rerank_ms": 128.4}
  }
}
```

- `scores.bm25` / `scores.semantic`: raw component scores from each retriever (Qdrant's BM25 for sparse, cosine for dense). `None` if the retriever didn't return this doc at all, or was disabled (`weight=0`).
- `scores.fused_rrf`: the weighted-RRF fused score used to rank *before* reranking.
- `scores.reranker`: cross-encoder score, or `None` if the reranker was skipped (either globally via `.env` or per-request via `use_reranker=false`).
- `scores.final`: the score the hit was sorted by — reranker if present, otherwise `fused_rrf`.

### `GET /health`

Liveness + wiring check:

```json
{"status":"ok","qdrant_reachable":true,"collection_exists":true,"collection":"nfcorpus","num_points":12345}
```

---

## CLI reference

```
ir3 ingest   [--max-docs N] [--dense-model ...] [--chunk-strategy recursive|fixed|semantic]
             [--chunk-size N] [--chunk-overlap N]
ir3 stats
ir3 reset    [-y]
ir3 serve    [--host H] [--port P] [--reload]
ir3 search   "your query" [--top-k N] [--no-rerank] [--source pubmed]
             [--date-gte ISO] [--date-lte ISO] [--json]
```

---

## Configuration (`.env`)

All defaults live in `.env.example`. Highlights:

| Key | Default | Notes |
|---|---|---|
| `DENSE_MODEL` | `BAAI/bge-small-en-v1.5` | or `-base-en-v1.5` (768) / `-large-en-v1.5` (1024) |
| `SPARSE_MODEL` | `Qdrant/bm25` | |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | or `-large` |
| `CHUNK_STRATEGY` | `recursive` | `recursive` \| `fixed` \| `semantic` |
| `CHUNK_SIZE` | `512` | characters. For fixed, try 256 / 512 / 1024. |
| `CHUNK_OVERLAP` | `50` | ~10–20 % of `CHUNK_SIZE`. |
| `TOP_K` | `5` | Final hits returned. |
| `RERANK_TOP_N` | `20` | Candidates sent to the cross-encoder. |
| `PREFETCH_LIMIT` | `50` | Per-component prefetch depth. |
| `WEIGHT_DENSE` / `WEIGHT_SPARSE` | `1.0` / `1.0` | Weighted RRF weights. |
| `RRF_K` | `60` | Standard RRF constant. |
| `USE_RERANKER` | `true` | Global toggle; per-request override supported. |

---

## Tuning notes

- **Skip dense or BM25 entirely** by setting `weight_dense=0` or `weight_sparse=0` on a per-request basis — the engine short-circuits the corresponding Qdrant call.
- **Model upgrades** invalidate only the affected chunks: switching from `bge-small` to `bge-base` will re-embed everything (dim change), but switching from `recursive/512/50` to `recursive/1024/100` re-chunks + re-embeds only because chunk text changed.
- **Sparse model swap** will mark every document as changed (it's baked into the manifest's document row), triggering full re-embedding. This is the correct behaviour — Qdrant's sparse vector dimensions aren't fixed but the tokenisation differs between sparse models.

---

## Project layout

```
src/retrieval/
├── api.py               # FastAPI app (POST /search, GET /health)
├── cli.py               # Typer CLI (ingest / stats / reset / serve / search)
├── config.py            # pydantic-settings
├── models.py            # Pydantic domain + API models
├── hashing.py           # Two-level hash helpers
├── manifest.py          # SQLite manifest store
├── qdrant_store.py      # Qdrant collection setup + payload indexes
├── embeddings.py        # FastEmbed dense + sparse wrappers
├── reranker.py          # FastEmbed TextCrossEncoder wrapper
├── ingestion.py         # Incremental ingestion pipeline
├── search.py            # HybridSearchEngine (composable: retrieve → fuse → rerank)
├── chunkers/
│   ├── base.py          # Chunker interface
│   ├── fixed.py         # CharacterTextSplitter (langchain-text-splitters)
│   ├── recursive.py     # RecursiveCharacterTextSplitter
│   └── semantic.py      # SemanticChunker (langchain-experimental)
└── datasets/
    └── nfcorpus.py      # BEIR NFCorpus loader (BeIR/nfcorpus)
```
