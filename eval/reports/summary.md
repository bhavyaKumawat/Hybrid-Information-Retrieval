# NFCorpus evaluation — summary

Metrics: precision@5, precision@10, recall@5, recall@10, ndcg@5, ndcg@10.
Split: BEIR NFCorpus `test`. Aggregation: max-pool chunks → docs.

## All runs (sorted by NDCG@10)

| tag | ablation | precision@5 | precision@10 | recall@5 | recall@10 | ndcg@5 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abl1_retrieval_mode__hybrid_rrf_equal | retrieval_mode | 0.3362 | 0.2601 | 0.1396 | 0.1762 | 0.3961 | 0.3626 |
| abl1_retrieval_mode__dense_only | retrieval_mode | 0.3368 | 0.2588 | 0.1379 | 0.1709 | 0.3904 | 0.3546 |
| abl1_retrieval_mode__bm25_only | retrieval_mode | 0.2941 | 0.2223 | 0.1232 | 0.1502 | 0.3420 | 0.3094 |
| abl1_retrieval_mode__hybrid_rrf_rerank | retrieval_mode | 0.2929 | 0.2282 | 0.1147 | 0.1431 | 0.3397 | 0.3078 |

## Ablation — retrieval_mode

| tag | weight_dense | weight_sparse | use_reranker | precision@5 | precision@10 | recall@5 | recall@10 | ndcg@5 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abl1_retrieval_mode__hybrid_rrf_equal | 1.0000 | 1.0000 | False | 0.3362 | 0.2601 | 0.1396 | 0.1762 | 0.3961 | 0.3626 |
| abl1_retrieval_mode__dense_only | 1.0000 | 0.0000 | False | 0.3368 | 0.2588 | 0.1379 | 0.1709 | 0.3904 | 0.3546 |
| abl1_retrieval_mode__bm25_only | 0.0000 | 1.0000 | False | 0.2941 | 0.2223 | 0.1232 | 0.1502 | 0.3420 | 0.3094 |
| abl1_retrieval_mode__hybrid_rrf_rerank | 1.0000 | 1.0000 | True | 0.2929 | 0.2282 | 0.1147 | 0.1431 | 0.3397 | 0.3078 |
