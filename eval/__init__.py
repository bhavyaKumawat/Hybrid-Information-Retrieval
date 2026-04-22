"""Offline evaluation harness for the IR3 retrieval platform.

One ablation per design decision, holding everything else fixed at
sensible defaults. Each run:

1. Ensures the right Qdrant collection exists (one per unique
   ``embedder × chunker`` pair, tracked in the shared SQLite manifest).
2. Runs every NFCorpus test query through the retrieval engine with the
   ablation's settings.
3. Saves the raw chunk-level hits to ``eval/runs/<tag>.json`` so
   metrics can be recomputed offline without re-running retrieval.

Metrics are computed via ``ranx`` after max-pooling chunk scores up to
document scores (NFCorpus qrels are per-document).

See :mod:`eval.cli` for the command-line entrypoint.
"""

import warnings

warnings.filterwarnings("ignore")

__all__: list[str] = []
