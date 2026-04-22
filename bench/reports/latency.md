# Retrieval latency report

- collection: `nfcorpus`
- dense model: `BAAI/bge-small-en-v1.5`
- sparse model: `Qdrant/bm25`
- reranker: `BAAI/bge-reranker-base`
- weights: dense=1.0, sparse=1.0, rrf_k=60
- top_k=5, rerank_top_n=20, prefetch_limit=50
- queries=45, warmup=5, warm_iters=10

## Per-stage latency (ms)

| stage | regime | n | mean | p50 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|---:|---:|
| dense_embed | cold | 45 | 2.439 | 2.203 | 4.215 | 6.559 | 8.037 |
| dense_query | cold | 45 | 3.781 | 3.446 | 6.157 | 6.437 | 6.443 |
| sparse_embed | cold | 45 | 0.051 | 0.050 | 0.058 | 0.062 | 0.063 |
| sparse_query | cold | 45 | 1.917 | 1.747 | 4.216 | 4.943 | 4.960 |
| fuse | cold | 45 | 0.060 | 0.053 | 0.108 | 0.112 | 0.114 |
| rerank | cold | 45 | 698.5 | 688.0 | 773.3 | 823.2 | 856.4 |
| total | cold | 45 | 706.7 | 696.4 | 780.9 | 831.3 | 864.1 |
| dense_embed | warm | 450 | 2.506 | 2.224 | 4.153 | 6.334 | 10.15 |
| dense_query | warm | 450 | 4.148 | 3.475 | 6.828 | 9.449 | 13.66 |
| sparse_embed | warm | 450 | 0.052 | 0.050 | 0.069 | 0.075 | 0.081 |
| sparse_query | warm | 450 | 2.104 | 1.847 | 4.352 | 5.189 | 7.700 |
| fuse | warm | 450 | 0.068 | 0.053 | 0.115 | 0.166 | 2.132 |
| rerank | warm | 450 | 717.0 | 703.6 | 811.4 | 861.7 | 928.6 |
| total | warm | 450 | 725.8 | 711.8 | 821.8 | 871.3 | 947.8 |

## End-to-end breakdown (warm, mean ms)

| stage | mean ms | % of total |
|---|---:|---:|
| dense_embed | 2.506 | 0.3% |
| dense_query | 4.148 | 0.6% |
| sparse_embed | 0.052 | 0.0% |
| sparse_query | 2.104 | 0.3% |
| fuse | 0.068 | 0.0% |
| rerank | 717.0 | 98.8% |
| **total** | **725.8** | 100.0% |
