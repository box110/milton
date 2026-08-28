"""Screener re-scoring and MACD residual recovery.

The replay must reproduce shrub's stored score exactly; if it doesn't, every IC
measured on top of it is measuring a different function than the one in
production.
"""
import pytest

from milton.features import (
    DecompositionError, build_dataset, macd_state_from_residual,
)
from milton.screener import (
    INCUMBENT, MACD_BULLISH, MACD_CROSS, MACD_NONE, score,
)


def test_incumbent_matches_shrubs_constants():
    # RSI oversold (30) + MACD cross (35) + SMA support (20) + volume (20)
    assert score(30.0, MACD_CROSS, -2.0, 2.0, INCUMBENT) == 105.0
    # RSI recovering (15) + MACD bullish (15) + price above SMA (10), quiet volume
    assert score(40.0, MACD_BULLISH, 2.0, 1.0, INCUMBENT) == 40.0
    # Nothing fires: overbought, no MACD, far above SMA, normal volume
    assert score(70.0, MACD_NONE, 12.0, 1.0, INCUMBENT) == 0.0


def test_missing_indicators_score_zero_not_neutral():
    """shrub skips a term on short history rather than substituting a neutral
    value; a fake-neutral RSI must never earn points."""
    assert score(None, MACD_NONE, None, None, INCUMBENT) == 0.0


def test_band_edges_are_exclusive_as_in_shrub():
    assert score(25.0, MACD_NONE, None, None, INCUMBENT) == 0.0    # not > 25
    assert score(35.0, MACD_NONE, None, None, INCUMBENT) == 15.0   # recovering band
    assert score(45.0, MACD_NONE, None, None, INCUMBENT) == 0.0    # not < 45
    assert score(None, MACD_NONE, 0.0, None, INCUMBENT) == 10.0    # above, not support
    assert score(None, MACD_NONE, None, 1.5, INCUMBENT) == 0.0     # not > 1.5


def test_weights_actually_move_the_score():
    w = INCUMBENT.replace(rsi_oversold=50.0, volume_high=0.0)
    assert score(30.0, MACD_NONE, None, 2.0, w) == 50.0


def test_bands_are_tunable_too():
    w = INCUMBENT.replace(rsi_oversold_hi=40.0)
    assert score(37.0, MACD_NONE, None, None, INCUMBENT) == 15.0   # recovering
    assert score(37.0, MACD_NONE, None, None, w) == 30.0           # now oversold


@pytest.mark.parametrize("state,expected", [
    (MACD_NONE, 0.0), (MACD_BULLISH, 15.0), (MACD_CROSS, 35.0)])
def test_macd_residual_roundtrips(state, expected):
    total = score(30.0, state, -2.0, 2.0, INCUMBENT)
    assert total == 70.0 + expected
    assert macd_state_from_residual(total, 30.0, -2.0, 2.0) == state


def test_unexplained_residual_raises_rather_than_guessing():
    """A score our model can't account for means shrub's scoring has drifted.
    Absorbing it silently would corrupt every fit built on top."""
    with pytest.raises(DecompositionError, match="no longer matches"):
        macd_state_from_residual(77.0, 30.0, -2.0, 2.0)


def _row(pick_id, score_val, rsi, vol, sma, alpha, when, horizon=20):
    return {
        "id": pick_id, "symbol": "AAPL", "created_at": when,
        "horizon_days": horizon, "alpha": alpha,
        "source_signals": {"score": score_val, "rsi": rsi,
                           "volume_ratio": vol, "price_vs_sma20": sma},
    }


def test_build_dataset_reports_drops_instead_of_hiding_them():
    from datetime import datetime
    when = datetime(2026, 8, 3, 13, 30)
    rows = [
        _row(1, 105.0, 30.0, 2.0, -2.0, 1.5, when),   # decomposes
        _row(2, 77.0, 30.0, 2.0, -2.0, 0.5, when),    # bad residual
        _row(3, None, 30.0, 2.0, -2.0, 0.5, when),    # no stored score
    ]
    picks, problems = build_dataset(rows)
    assert len(picks) == 1 and picks[0].pick_id == 1
    assert len(problems) == 2
    assert picks[0].macd_state == MACD_CROSS
    assert picks[0].rescore(INCUMBENT) == picks[0].stored_score


def test_duplicate_symbol_days_are_collapsed():
    """A research failure loop re-runs the screener many times a day, logging
    the same names repeatedly. Each repeat carries the same forward return, so
    keeping them inflates the sample without adding information."""
    from datetime import datetime
    rows = []
    for i in range(5):                      # same symbol, same day, 5 times
        rows.append(_row(i + 1, 105.0, 30.0, 2.0, -2.0, 1.5,
                         datetime(2026, 7, 30, 13 + i, 0)))
    rows.append(_row(99, 105.0, 30.0, 2.0, -2.0, 2.0,
                     datetime(2026, 7, 31, 13, 0)))   # next day survives
    picks, problems = build_dataset(rows)
    assert len(picks) == 2
    assert picks[0].pick_id == 1            # earliest row for the symbol-day
    assert any("collapsed 4 duplicate" in p for p in problems)


def test_dedupe_keeps_distinct_symbols_and_horizons():
    from datetime import datetime
    when = datetime(2026, 7, 30, 13, 0)
    rows = [
        _row(1, 105.0, 30.0, 2.0, -2.0, 1.5, when, horizon=20),
        _row(2, 105.0, 30.0, 2.0, -2.0, 0.5, when, horizon=5),   # other horizon
    ]
    rows[1]["symbol"] = "AAPL"
    picks, _ = build_dataset(rows)
    assert len(picks) == 2


def test_dedupe_can_be_turned_off():
    from datetime import datetime
    when = datetime(2026, 7, 30, 13, 0)
    rows = [_row(i, 105.0, 30.0, 2.0, -2.0, 1.5, when) for i in (1, 2, 3)]
    picks, _ = build_dataset(rows, dedupe=False)
    assert len(picks) == 3
