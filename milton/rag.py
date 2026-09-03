"""Source-reliability weights for retrieval ranking, as tunable parameters.

shrub scores a retrieved chunk as `rrf * recency * reliability`, where
reliability comes from a hand-ordered table in `ingest/base.py` and recency
decays on a per-source half-life. Both were guesses. This module makes them
parameters so they can be measured.

WHAT IS ACTUALLY BEING MEASURED, AND WHAT ISN'T

The screener had a clean label: a pick, a date, a forward return. Retrieval has
none — nowhere does anything record whether a retrieved chunk was the right one.
So the question has to be reframed into one the data can answer:

    does the presence of evidence from source S, near date D, carry any
    information about ticker T's subsequent return?

That is a proxy, and its limits should be stated rather than discovered later.
It measures PREDICTIVENESS, not TRUTHFULNESS. A source can be scrupulously
accurate and score nothing here because what it reports is already in the
price; a source can be junk and score well because it moves crowds. For a
ranking whose job is to feed a return-seeking process, predictiveness is the
defensible criterion — but it is not the same claim as "this source tells the
truth", and a weight fitted here should never be described as one.

The features are recency-weighted evidence counts per source, using shrub's own
decay shape, so a fitted half-life means the same thing it means in production.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from datetime import date, datetime

# Source types with enough live corpus to fit anything. The other five entries
# in shrub's RELIABILITY_RANK rank nothing: earnings_transcript, press_release
# and macro have never produced a document, prediction_market carries no ticker
# so ticker-filtered retrieval cannot reach it, and analyst has 20 tagged
# chunks. Fitting weights for them would be fitting noise to an empty set.
FITTABLE_SOURCES = ("sec_filing", "news", "newsletter", "social")

# shrub's incumbent values, for the sources that exist.
INCUMBENT_RELIABILITY = {
    "sec_filing": 1.000, "news": 0.500, "newsletter": 0.375, "social": 0.250,
}
INCUMBENT_HALF_LIFE = {
    "sec_filing": 180.0, "news": 14.0, "newsletter": 10.0, "social": 5.0,
}

# shrub's recency shape: floor + span * 0.5 ** (age / half_life).
RECENCY_FLOOR = 0.4
RECENCY_SPAN = 0.6

# Chunks per ticker the economist's evidence block actually receives
# (`_corpus_evidence_block(..., per_ticker=3)`). The weights decide which few
# chunks fill these slots, so the fit has to be scored over the same few.
TOP_K = 3


@dataclass(frozen=True)
class SourceWeights:
    """Reliability weight and decay half-life per source, plus the shared
    recency floor. Defaults are shrub's live values."""
    sec_filing: float = 1.000
    news: float = 0.500
    newsletter: float = 0.375
    social: float = 0.250

    hl_sec_filing: float = 180.0
    hl_news: float = 14.0
    hl_newsletter: float = 10.0
    hl_social: float = 5.0

    recency_floor: float = RECENCY_FLOOR

    def reliability(self, source: str) -> float:
        return float(getattr(self, source, 0.0))

    def half_life(self, source: str) -> float:
        return float(getattr(self, f"hl_{source}", 30.0))

    def replace(self, **kw) -> "SourceWeights":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


INCUMBENT = SourceWeights()

RELIABILITY_PARAMS = FITTABLE_SOURCES
HALF_LIFE_PARAMS = tuple(f"hl_{s}" for s in FITTABLE_SOURCES)
TUNABLE = RELIABILITY_PARAMS + HALF_LIFE_PARAMS + ("recency_floor",)

BOUNDS = {
    **{s: (0.0, 1.0) for s in RELIABILITY_PARAMS},
    **{f"hl_{s}": (1.0, 365.0) for s in FITTABLE_SOURCES},
    "recency_floor": (0.0, 0.9),
}


def recency_factor(age_days: float, half_life: float, floor: float) -> float:
    """shrub's decay, reproduced exactly so a fitted half-life transfers.

    The floor matters as much as the half-life: at 0.4 an ancient filing keeps
    40% of its weight no matter how old, which is what stops long-lived primary
    sources from decaying to nothing behind a day-old tweet."""
    if half_life <= 0:
        return floor
    return floor + (1.0 - floor) * (0.5 ** (max(0.0, age_days) / half_life))


@dataclass(frozen=True)
class Evidence:
    """One chunk that was retrievable for a ticker on a given day."""
    source_type: str
    age_days: float


@dataclass(frozen=True)
class EvidenceDay:
    """All evidence available for one ticker on one day, plus the outcome."""
    symbol: str
    day: date
    horizon_days: int
    alpha: float
    evidence: tuple[Evidence, ...]

    def score(self, w: SourceWeights, k: int = TOP_K) -> float:
        """Mean weight of the top-k chunks, as retrieval would select them.

        Summing every chunk was the obvious first move and it was wrong. Social
        is 496,663 of the corpus's 544,611 ticker-tagged chunks, so a sum is
        dominated by how much a name is talked about rather than by how good
        its evidence is — the score became a mention count wearing a
        reliability weight. shrub does not sum: `_corpus_evidence_block` asks
        for the top 3 chunks per ticker, so what a weight actually decides is
        WHICH FEW chunks are read, not how much total weight a ticker
        accumulates.

        Taking the top k reproduces that. Raising social's weight now competes
        for a scarce slot against a filing instead of adding to a pile, which
        is the decision the weight really governs.

        The RRF term is dropped: it depends on a query, and these two factors
        are the ones that are properties of the chunk itself."""
        scored = []
        for e in self.evidence:
            rel = w.reliability(e.source_type)
            if rel == 0.0:
                continue
            scored.append(rel * recency_factor(
                e.age_days, w.half_life(e.source_type), w.recency_floor))
        if not scored:
            return 0.0
        scored.sort(reverse=True)
        top = scored[:k]
        return sum(top) / len(top)


def _to_date(v) -> date:
    return v.date() if isinstance(v, datetime) else v


def build_evidence_days(rows, *, sources=FITTABLE_SOURCES) -> list[EvidenceDay]:
    """Group joined (pick, chunk) rows into one EvidenceDay per symbol-day.

    Rows arrive as one per chunk; a symbol-day with no evidence at all simply
    has an empty tuple and scores zero, which is meaningful — it says the
    corpus had nothing to say about that name that day."""
    wanted = set(sources)
    by_key: dict[tuple, dict] = {}
    for r in rows:
        day = _to_date(r["pick_date"])
        key = (r["symbol"], day, int(r["horizon_days"]))
        entry = by_key.setdefault(key, {"alpha": float(r["alpha"]), "ev": []})
        st = r.get("source_type")
        if st in wanted and r.get("age_days") is not None:
            entry["ev"].append(Evidence(st, float(r["age_days"])))
    out = [
        EvidenceDay(symbol=k[0], day=k[1], horizon_days=k[2],
                    alpha=v["alpha"], evidence=tuple(v["ev"]))
        for k, v in by_key.items()
    ]
    return sorted(out, key=lambda d: (d.day, d.symbol))
