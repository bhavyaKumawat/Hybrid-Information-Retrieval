# Retrieval latency report

- collection: `nfcorpus`
- dense model: `BAAI/bge-small-en-v1.5`
- sparse model: `Qdrant/bm25`
- reranker: _disabled_
- weights: dense=1.0, sparse=1.0, rrf_k=60
- top_k=5, rerank_top_n=20, prefetch_limit=50
- queries=45, warmup=5, warm_iters=10

## Per-stage latency (ms)

| stage | regime | n | mean | p50 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|---:|---:|
| dense_embed | cold | 45 | 7.710 | 7.339 | 10.14 | 10.78 | 10.82 |
| dense_query | cold | 45 | 9.942 | 9.888 | 12.02 | 13.24 | 13.92 |
| sparse_embed | cold | 45 | 0.120 | 0.100 | 0.225 | 0.311 | 0.372 |
| sparse_query | cold | 45 | 6.077 | 6.235 | 10.77 | 12.26 | 13.10 |
| fuse | cold | 45 | 0.217 | 0.142 | 0.641 | 1.488 | 2.004 |
| rerank | cold | 45 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| total | cold | 45 | 24.07 | 23.64 | 30.18 | 31.64 | 32.14 |
| dense_embed | warm | 450 | 7.693 | 7.363 | 10.08 | 11.78 | 28.14 |
| dense_query | warm | 450 | 6.992 | 6.632 | 9.715 | 12.36 | 38.36 |
| sparse_embed | warm | 450 | 0.132 | 0.110 | 0.277 | 0.388 | 0.582 |
| sparse_query | warm | 450 | 3.758 | 3.553 | 6.122 | 7.534 | 15.86 |
| fuse | warm | 450 | 0.200 | 0.162 | 0.502 | 0.990 | 2.537 |
| rerank | warm | 450 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| total | warm | 450 | 18.78 | 18.15 | 22.98 | 30.42 | 60.21 |

## End-to-end breakdown (warm, mean ms)

| stage | mean ms | % of total |
|---|---:|---:|
| dense_embed | 7.693 | 41.0% |
| dense_query | 6.992 | 37.2% |
| sparse_embed | 0.132 | 0.7% |
| sparse_query | 3.758 | 20.0% |
| fuse | 0.200 | 1.1% |
| **total** | **18.78** | 100.0% |
