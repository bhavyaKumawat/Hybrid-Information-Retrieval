# NFCorpus evaluation — summary

Metrics: precision@5, precision@10, recall@5, recall@10, ndcg@5, ndcg@10.
Split: BEIR NFCorpus `test`. Aggregation: max-pool chunks → docs.

## All runs (sorted by NDCG@10)

| tag | ablation | precision@5 | precision@10 | recall@5 | recall@10 | ndcg@5 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abl1_retrieval_mode__hybrid_rrf_equal | retrieval_mode | 0.3368 | 0.2601 | 0.1397 | 0.1763 | 0.3959 | 0.3622 |
| abl4_reranker__bge-small__bge-large | reranker | 0.3368 | 0.2601 | 0.1397 | 0.1763 | 0.3959 | 0.3622 |
| abl4_reranker__bge-small__none | reranker | 0.3368 | 0.2601 | 0.1397 | 0.1763 | 0.3959 | 0.3622 |
| abl1_retrieval_mode__dense_only | retrieval_mode | 0.3368 | 0.2591 | 0.1379 | 0.1710 | 0.3904 | 0.3546 |
| abl5_rerank_top_n__n10 | rerank_top_n | 0.3065 | 0.2402 | 0.1155 | 0.1454 | 0.3497 | 0.3177 |
| abl1_retrieval_mode__bm25_only | retrieval_mode | 0.2941 | 0.2220 | 0.1232 | 0.1501 | 0.3420 | 0.3092 |
| abl1_retrieval_mode__hybrid_rrf_rerank | retrieval_mode | 0.2941 | 0.2282 | 0.1149 | 0.1421 | 0.3402 | 0.3072 |
| abl2_embedder_size__bge-small | embedder_size | 0.2941 | 0.2282 | 0.1149 | 0.1421 | 0.3402 | 0.3072 |
| abl3_chunker__bge-small__recursive-512-50 | chunker | 0.2941 | 0.2282 | 0.1149 | 0.1421 | 0.3402 | 0.3072 |
| abl4_reranker__bge-small__bge-base | reranker | 0.2941 | 0.2282 | 0.1149 | 0.1421 | 0.3402 | 0.3072 |

## Ablation — retrieval_mode

| tag | weight_dense | weight_sparse | use_reranker | precision@5 | precision@10 | recall@5 | recall@10 | ndcg@5 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abl1_retrieval_mode__hybrid_rrf_equal | 1.0000 | 1.0000 | False | 0.3368 | 0.2601 | 0.1397 | 0.1763 | 0.3959 | 0.3622 |
| abl1_retrieval_mode__dense_only | 1.0000 | 0.0000 | False | 0.3368 | 0.2591 | 0.1379 | 0.1710 | 0.3904 | 0.3546 |
| abl1_retrieval_mode__bm25_only | 0.0000 | 1.0000 | False | 0.2941 | 0.2220 | 0.1232 | 0.1501 | 0.3420 | 0.3092 |
| abl1_retrieval_mode__hybrid_rrf_rerank | 1.0000 | 1.0000 | True | 0.2941 | 0.2282 | 0.1149 | 0.1421 | 0.3402 | 0.3072 |

## Ablation — embedder_size

| tag | dense_model | precision@5 | precision@10 | recall@5 | recall@10 | ndcg@5 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abl2_embedder_size__bge-small | BAAI/bge-small-en-v1.5 | 0.2941 | 0.2282 | 0.1149 | 0.1421 | 0.3402 | 0.3072 |

## Ablation — chunker

| tag | chunk_strategy | chunk_size | chunk_overlap | precision@5 | precision@10 | recall@5 | recall@10 | ndcg@5 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abl3_chunker__bge-small__recursive-512-50 | recursive | 512 | 50 | 0.2941 | 0.2282 | 0.1149 | 0.1421 | 0.3402 | 0.3072 |

## Ablation — reranker

| tag | reranker_model | use_reranker | precision@5 | precision@10 | recall@5 | recall@10 | ndcg@5 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abl4_reranker__bge-small__bge-large | BAAI/bge-reranker-large | True | 0.3368 | 0.2601 | 0.1397 | 0.1763 | 0.3959 | 0.3622 |
| abl4_reranker__bge-small__none |  | False | 0.3368 | 0.2601 | 0.1397 | 0.1763 | 0.3959 | 0.3622 |
| abl4_reranker__bge-small__bge-base | BAAI/bge-reranker-base | True | 0.2941 | 0.2282 | 0.1149 | 0.1421 | 0.3402 | 0.3072 |

## Ablation — rerank_top_n

| tag | rerank_top_n | prefetch_limit | precision@5 | precision@10 | recall@5 | recall@10 | ndcg@5 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abl5_rerank_top_n__n10 | 10 | 50 | 0.3065 | 0.2402 | 0.1155 | 0.1454 | 0.3497 | 0.3177 |
