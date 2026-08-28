"""Optimiser and gate behaviour.

The properties under test are mostly about restraint: the search must not drift
without cause, and the gate must refuse anything it cannot actually justify.
"""
from datetime import date, datetime

import pytest

from milton.features import Pick
from milton.gate import (
    MIN_TEST_DAYS, evaluate, paired_daily_ic, split_by_time,
)
from milton.optimize import SearchSpec, optimise, penalised_objective
from milton.screener import INCUMBENT, MACD_CROSS, MACD_NONE, ScreenerWeights


def _pick(day, rsi, alpha, *, vol=1.0, sma=10.0, macd=MACD_NONE, h=20, pid=0):
    return Pick(pick_id=pid, symbol=f"S{pid}", pick_date=day, horizon_days=h,
                rsi=rsi, volume_ratio=vol, price_vs_sma20=sma, macd_state=macd,
                stored_score=0.0, alpha=alpha)


def _days(n, start=1):
    base = date(2026, 7, 1).toordinal() + start - 1
    return [date.fromordinal(base + i) for i in range(n)]


def _signal_picks(days, *, flip=False):
    """A clean signal the optimiser should be able to find: oversold RSI names
    outperform. `flip` inverts it so the incumbent's sign is wrong."""
    out, pid = [], 0
    for d in days:
        for rsi, base in ((30.0, 3.0), (40.0, 1.0), (60.0, -1.0), (70.0, -3.0)):
            pid += 1
            out.append(_pick(d, rsi, -base if flip else base, pid=pid))
    return out


def _rank_only_picks(days):
    """Signal the optimiser can fix WITHOUT changing the score scale.

    Every name carries a MACD cross (+35), so all of them clear MIN_SCORE under
    any sane weight vector and the selection-drift check has nothing to fire
    on. The mistake to find is an ordering one: "recovering" RSI names actually
    outperform "oversold" ones, and the incumbent has that backwards (30 vs
    15). The remedy is a swap, not a rescale."""
    out, pid = [], 0
    for d in days:
        for rsi, base in ((40.0, 3.0), (30.0, 1.0), (60.0, -1.0), (70.0, -3.0)):
            pid += 1
            out.append(_pick(d, rsi, base, macd=MACD_CROSS, pid=pid))
    return out


# --- split ----------------------------------------------------------------

def test_split_is_chronological_and_never_divides_a_day():
    picks = _signal_picks(_days(10))
    train, test = split_by_time(picks, test_fraction=0.4)
    train_days = {p.pick_date for p in train}
    test_days = {p.pick_date for p in test}
    assert not (train_days & test_days)          # no day on both sides
    assert max(train_days) < min(test_days)      # train strictly earlier
    assert len(test_days) == 4


def test_split_handles_empty_input():
    assert split_by_time([]) == ([], [])


# --- paired comparison ----------------------------------------------------

def test_paired_ic_uses_identical_name_sets_per_day():
    picks = _signal_picks(_days(5))
    a, b, days = paired_daily_ic(picks, INCUMBENT, INCUMBENT.replace(rsi_oversold=0))
    assert len(a) == len(b) == len(days) == 5


def test_paired_ic_skips_days_where_either_side_is_undefined():
    """A vector that scores every name identically has no defined IC, so the
    day cannot contribute to a paired comparison."""
    picks = _signal_picks(_days(4))
    flat = ScreenerWeights(rsi_oversold=0, rsi_recovering=0, macd_cross=0,
                           macd_bullish=0, sma_support=0, sma_above=0,
                           volume_high=0)
    a, b, days = paired_daily_ic(picks, flat, INCUMBENT)
    assert days == []


# --- optimiser ------------------------------------------------------------

def _noise_picks(days, seed=7):
    import random
    rng = random.Random(seed)
    out, pid = [], 0
    for d in days:
        for rsi in (30.0, 40.0, 60.0, 70.0):
            pid += 1
            out.append(_pick(d, rsi, rng.gauss(0, 5), macd=MACD_CROSS, pid=pid))
    return out


def test_noise_does_not_survive_the_gate():
    """The end-to-end property that actually matters.

    On a sample this small the optimiser WILL find something in pure noise —
    daily IC over four names swings far more than any defensible shrinkage
    penalty, so no penalty value makes the search safe by itself. That is the
    gate's whole reason to exist: the search is allowed to be credulous
    because nothing it produces ships without surviving days it never saw."""
    picks = _noise_picks(_days(40))
    train, test = split_by_time(picks, test_fraction=0.5)
    cand = optimise(train, spec=SearchSpec(max_sweeps=4)).weights
    res = evaluate(train, test, cand, INCUMBENT)
    assert not res.promoted, res.reasons


def test_shrinkage_keeps_the_search_still_on_a_weak_signal():
    """With the penalty on, a search over noise should stay at or very near the
    incumbent — it may not be immovable, but it must not roam."""
    picks = _noise_picks(_days(40), seed=3)
    restrained = optimise(picks, spec=SearchSpec(shrinkage=0.5, max_sweeps=4))
    loose = optimise(picks, spec=SearchSpec(shrinkage=0.0, max_sweeps=4))
    assert len(restrained.moves) < len(loose.moves) or restrained.moves == []


def test_optimiser_finds_a_real_inverted_signal():
    """When the incumbent's sign is genuinely wrong, the search should move —
    negative point values are allowed precisely so it can."""
    picks = _signal_picks(_days(12), flip=True)
    best = optimise(picks, spec=SearchSpec(shrinkage=0.0, max_sweeps=4))
    assert best.moves
    assert best.ic.mean_ic > 0


def test_shrinkage_penalises_distance_from_the_incumbent():
    picks = _signal_picks(_days(8))
    far = INCUMBENT.replace(rsi_oversold=60.0, volume_high=-30.0)
    plain, ic, dist = penalised_objective(picks, far, INCUMBENT,
                                          SearchSpec(shrinkage=0.0))
    penalised, _, _ = penalised_objective(picks, far, INCUMBENT,
                                          SearchSpec(shrinkage=0.5))
    assert dist > 0
    assert penalised < plain


def test_bands_are_frozen_unless_asked_for():
    assert all(p in SearchSpec().params() for p in ("rsi_oversold", "macd_cross"))
    assert "rsi_oversold_lo" not in SearchSpec().params()
    assert "rsi_oversold_lo" in SearchSpec(tune_bands=True).params()


# --- gate -----------------------------------------------------------------

def test_gate_rejects_when_there_are_too_few_test_days():
    picks = _signal_picks(_days(6), flip=True)
    train, test = split_by_time(picks, test_fraction=0.5)
    res = evaluate(train, test, INCUMBENT.replace(rsi_oversold=-30.0), INCUMBENT)
    assert not res.promoted
    assert any("too few" in r for r in res.reasons)


def test_gate_rejects_a_candidate_identical_to_the_incumbent():
    picks = _signal_picks(_days(40))
    train, test = split_by_time(picks)
    res = evaluate(train, test, INCUMBENT, INCUMBENT)
    assert not res.promoted
    assert any("identical" in r for r in res.reasons)


def test_gate_rejects_when_the_edge_does_not_survive_out_of_sample():
    """Signal in the training half, pure noise in the test half — the classic
    overfit the gate exists to catch."""
    import random
    rng = random.Random(11)
    picks = _signal_picks(_days(20, start=1), flip=True)
    pid = 1000
    for d in _days(20, start=21):
        for rsi in (30.0, 40.0, 60.0, 70.0):
            pid += 1
            picks.append(_pick(d, rsi, rng.gauss(0, 5), pid=pid))
    train, test = split_by_time(picks, test_fraction=0.5)
    cand = optimise(train, spec=SearchSpec(shrinkage=0.0, max_sweeps=4)).weights
    res = evaluate(train, test, cand, INCUMBENT)
    assert not res.promoted


def test_gate_promotes_a_genuine_persistent_improvement():
    picks = _rank_only_picks(_days(30))
    train, test = split_by_time(picks, test_fraction=0.5)
    cand = optimise(train, spec=SearchSpec(shrinkage=0.0, max_sweeps=4)).weights
    res = evaluate(train, test, cand, INCUMBENT)
    assert res.promoted, res.reasons
    assert res.paired_mean > 0
    assert res.test_days >= MIN_TEST_DAYS
    assert res.selection_candidate == res.selection_incumbent


def test_gate_blocks_a_change_that_would_resize_the_watchlist():
    """Ranking can improve while the number of names clearing MIN_SCORE moves
    sharply — that silently changes how many stocks reach the traders, which is
    a different decision from how they are ordered."""
    # These names score 10 under the incumbent (price just above SMA20 only),
    # so none of them clear MIN_SCORE and none reach a trader today.
    picks, pid = [], 0
    for d in _days(30):
        for sma, alpha in ((1.0, 3.0), (2.0, 1.0), (3.0, -1.0), (4.0, -3.0)):
            pid += 1
            picks.append(_pick(d, 60.0, alpha, sma=sma, pid=pid))
    train, test = split_by_time(picks, test_fraction=0.5)
    # Raising sma_above to 70 pushes every one of them over the threshold.
    inflated = INCUMBENT.replace(sma_above=70.0)
    res = evaluate(train, test, inflated, INCUMBENT)
    assert res.selection_incumbent == 0 and res.selection_candidate > 0
    assert not res.promoted
    assert any("selects none" in r for r in res.reasons)


def test_gate_blocks_a_proportional_watchlist_resize():
    """The other selection branch: the incumbent already selects names, and the
    candidate changes how many by more than the tolerance."""
    picks, pid = [], 0
    for d in _days(30):
        # Half clear MIN_SCORE on the MACD cross alone; half score 10 and don't.
        for sma, alpha, macd in ((1.0, 3.0, MACD_CROSS), (2.0, 1.0, MACD_CROSS),
                                 (3.0, -1.0, MACD_NONE), (4.0, -3.0, MACD_NONE)):
            pid += 1
            picks.append(_pick(d, 60.0, alpha, sma=sma, macd=macd, pid=pid))
    train, test = split_by_time(picks, test_fraction=0.5)
    inflated = INCUMBENT.replace(sma_above=70.0)   # now everything clears
    res = evaluate(train, test, inflated, INCUMBENT)
    assert res.selection_incumbent > 0
    assert res.selection_candidate > res.selection_incumbent
    assert not res.promoted
    assert any("resize the watchlist" in r for r in res.reasons)
