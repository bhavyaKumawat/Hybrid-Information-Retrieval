"""NFCorpus loader.

Pulls the NFCorpus corpus from the BEIR mirror on Hugging Face
(``BeIR/nfcorpus``, ``corpus`` split). Yields :class:`RawDocument` with:

* ``source`` fixed to ``"pubmed"`` — NFCorpus is PubMed-derived.
* ``date`` — NFCorpus doesn't ship a publication date, so we synthesise a
  deterministic one (seeded from the doc id, spread 2010-01-01 .. 2022-12-31).
  That lets the metadata date-filter be exercised end-to-end; it's *not* a
  real publication date. The synthesised value is documented as such in the
  README.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta

from datasets import load_dataset

from ..models import RawDocument

SOURCE = "pubmed"
_SYNTH_START = datetime(2010, 1, 1, tzinfo=UTC)
_SYNTH_SPAN_DAYS = 365 * 13  # through 2022-12-31


def _synthetic_date(doc_id: str) -> datetime:
    """Deterministic pseudo-date derived from doc_id.

    NFCorpus has no publication date; the assignment requires a filterable
    ``date`` field. Synthesising one keeps ingest honest and filters demo-able
    without pretending we have real metadata.
    """
    h = hashlib.sha256(doc_id.encode("utf-8")).digest()
    offset_days = int.from_bytes(h[:4], "big") % _SYNTH_SPAN_DAYS
    offset_seconds = int.from_bytes(h[4:8], "big") % 86400
    return _SYNTH_START + timedelta(days=offset_days, seconds=offset_seconds)


def load_nfcorpus(limit: int | None = None) -> Iterator[RawDocument]:
    """Iterate NFCorpus documents (title + text).

    ``limit=None`` or ``limit<0`` loads everything (~3.6k rows).
    """
    ds = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    seen = 0
    for row in _iter_rows(ds):
        if limit is not None and limit >= 0 and seen >= limit:
            break
        doc_id = str(row.get("_id") or row.get("id") or "").strip()
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        if not doc_id or not text:
            continue
        yield RawDocument(
            doc_id=doc_id,
            title=title,
            text=text,
            source=SOURCE,
            date=_synthetic_date(doc_id),
        )
        seen += 1


def _iter_rows(ds) -> Iterable[dict]:
    # ``datasets`` is iterable and yields dict-like rows.
    for row in ds:
        yield dict(row)
