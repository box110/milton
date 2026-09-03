"""Retrieval source weights: the scoring model and its gate.

The property under test throughout is that the score reflects what shrub's
retrieval actually does — pick a few chunks — rather than how much evidence a
ticker happens to have.
"""
from datetime import date

import pytest

from milton.rag import (
    INCUMBENT, TOP_K, Evidence, EvidenceDay, SourceWeights,
    build_evidence_days, recency_factor,
)
from milton.ragfit import evaluate_sources, optimise_sources, split_days


def _day(d, evidence, alpha=1.0, symbol="AAPL"):
    return EvidenceDay(symbol=symbol, day=d, horizon_days=20, alpha=alpha,
                       evidence=tuple(evidence))


# --- recency --------------------------------------------------------------

def test_recency_decays_to_the_floor_not_to_zero():
    """The floor is what stops a decade-old filing decaying below a day-old
    tweet — it matters as much as the half-life."""
    fresh = recency_factor(0, 30, 0.4)
    ancient = recency_factor(100_000, 30, 0.4)
    assert fresh == pytest.approx(1.0)
    assert ancient == pytest.approx(0.4)


def test_one_half_life_halves_the_decaying_part():
    assert recency_factor(30, 30, 0.4) == pytest.approx(0.4 + 0.6 * 0.5)


# --- scoring --------------------------------------------------------------

def test_score_is_top_k_not_a_sum():
    """Summing made the score a mention count: social is 91% of the corpus, so
    a ticker with hundreds of posts would outrank one with a fresh filing."""
    filing = _day(date(2026, 8, 3), [Evidence("sec_filing", 1.0)])
    chatter = _day(date(2026, 8, 3), [Evidence("social", 1.0)] * 200)
    assert filing.score(INCUMBENT) > chatter.score(INCUMBENT)


def test_score_uses_at_most_k_chunks():
    one = _day(date(2026, 8, 3), [Evidence("news", 1.0)])
    many = _day(date(2026, 8, 3), [Evidence("news", 1.0)] * (TOP_K * 5))
    assert one.score(INCUMBENT) == pytest.approx(many.score(INCUMBENT))


def test_a_weight_of_zero_removes_a_source_entirely():
    d = _day(date(2026, 8, 3), [Evidence("social", 1.0)])
    assert d.score(INCUMBENT) > 0
    assert d.score(INCUMBENT.replace(social=0.0)) == 0.0


def test_no_evidence_scores_zero_rather_than_erroring():
    """An empty corpus for a name is meaningful, not missing: it says nothing
    was available to read about it that day."""
    assert _day(date(2026, 8, 3), []).score(INCUMBENT) == 0.0


def test_fresher_evidence_of_the_same_source_scores_higher():
    old = _day(date(2026, 8, 3), [Evidence("news", 60.0)])
    new = _day(date(2026, 8, 3), [Evidence("news", 0.5)])
    assert new.score(INCUMBENT) > old.score(INCUMBENT)


def test_half_life_controls_how_fast_that_happens():
    old = _day(date(2026, 8, 3), [Evidence("news", 30.0)])
    slow = old.score(INCUMBENT.replace(hl_news=365.0))
    fast = old.score(INCUMBENT.replace(hl_news=1.0))
    assert slow > fast


# --- dataset construction -------------------------------------------------

def test_rows_group_into_symbol_days_and_drop_unfittable_sources():
    rows = [
        {"symbol": "AAPL", "pick_date": date(2026, 8, 3), "horizon_days": 20,
         "alpha": 1.5, "source_type": "news", "age_days": 2.0},
        {"symbol": "AAPL", "pick_date": date(2026, 8, 3), "horizon_days": 20,
         "alpha": 1.5, "source_type": "prediction_market", "age_days": 1.0},
        {"symbol": "MSFT", "pick_date": date(2026, 8, 3), "horizon_days": 20,
         "alpha": -0.5, "source_type": None, "age_days": None},
    ]
    days = build_evidence_days(rows)
    assert len(days) == 2
    by_sym = {d.symbol: d for d in days}
    # prediction_market carries no ticker in the real corpus and is not fittable
    assert [e.source_type for e in by_sym["AAPL"].evidence] == ["news"]
    assert by_sym["MSFT"].evidence == ()      # LEFT JOIN miss, still a row


# --- split and gate -------------------------------------------------------

def test_split_is_chronological_on_whole_days():
    days = [_day(date(2026, 8, d), [Evidence("news", 1.0)], symbol=f"S{i}")
            for d in range(1, 11) for i in range(4)]
    train, test = split_days(days, test_fraction=0.4)
    assert max(d.day for d in train) < min(d.day for d in test)
    assert len({d.day for d in test}) == 4


def test_gate_rejects_an_identical_candidate():
    days = [_day(date(2026, 8, d), [Evidence("news", 1.0)], alpha=float(i),
                 symbol=f"S{i}") for d in range(1, 20) for i in range(4)]
    res = evaluate_sources(days, INCUMBENT, INCUMBENT)
    assert not res.promoted
    assert any("identical" in r for r in res.reasons)


def test_gate_rejects_when_the_candidate_is_worse_out_of_sample():
    """The real result on shrub's corpus: the search wanted to downweight
    sec_filing, and that made the held-out IC substantially worse."""
    days = []
    for i, d in enumerate(range(1, 20)):
        day = date(2026, 8, d)
        # Filings mark the winners; anything that downweights them loses signal.
        days += [
            _day(day, [Evidence("sec_filing", 1.0)], alpha=3.0, symbol="A"),
            _day(day, [Evidence("social", 1.0)], alpha=-1.0, symbol="B"),
            _day(day, [Evidence("social", 20.0)], alpha=-2.0, symbol="C"),
        ]
    worse = INCUMBENT.replace(sec_filing=0.1)
    res = evaluate_sources(days, worse, INCUMBENT)
    assert not res.promoted
    assert res.paired_mean < 0


def test_optimiser_starts_from_and_can_return_the_incumbent():
    days = [_day(date(2026, 8, d), [Evidence("news", 1.0)], alpha=1.0,
                 symbol=f"S{i}") for d in range(1, 15) for i in range(4)]
    cand = optimise_sources(days, max_sweeps=1)
    assert cand.weights.as_dict().keys() == INCUMBENT.as_dict().keys()
    assert isinstance(cand.moves, list)
