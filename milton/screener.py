"""The screener's scoring function, as tunable parameters rather than constants.

shrub's `screener.py` scores a candidate by adding fixed points for four
technical conditions. Those point values were guesses, and this module exists
to find better ones. It is a faithful re-implementation, parameterised: with
INCUMBENT weights it reproduces shrub's stored scores exactly (verified against
all 14,680 logged picks, zero unexplained residuals).

The replay does NOT recompute indicators from bars. Every logged pick already
carries the raw RSI, volume ratio and price-vs-SMA20 that produced it, so
re-scoring under different weights is arithmetic over stored values. MACD is
the exception — shrub logs the score but not `macd_histogram` — and its
contribution is recovered as a residual instead (see features.py). That works
because the MACD term can only ever be one of three values, but it means MACD
*thresholds* are not tunable until shrub logs the histogram. Everything else
is, thresholds included.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, fields

# MACD contributes one of exactly three point values, which is what makes it
# recoverable from a residual. The states are ordered by strength.
MACD_NONE, MACD_BULLISH, MACD_CROSS = "none", "bullish", "cross"


@dataclass(frozen=True)
class ScreenerWeights:
    """Point values and band edges for the screener's score.

    Defaults are shrub's current constants — the incumbent any candidate has to
    beat. Band edges are parameters too: the question isn't only "how many
    points for oversold RSI" but "what counts as oversold".
    """
    # Points per condition.
    rsi_oversold: float = 30.0
    rsi_recovering: float = 15.0
    macd_cross: float = 35.0
    macd_bullish: float = 15.0
    sma_support: float = 20.0
    sma_above: float = 10.0
    volume_high: float = 20.0

    # Band edges.
    rsi_oversold_lo: float = 25.0
    rsi_oversold_hi: float = 35.0
    rsi_recovering_hi: float = 45.0
    sma_support_lo: float = -5.0
    sma_above_hi: float = 5.0
    volume_ratio_min: float = 1.5

    def replace(self, **kw) -> "ScreenerWeights":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


INCUMBENT = ScreenerWeights()

# The parameters an optimiser may move, split by what the stored data supports.
# MACD points are tunable (the state is recoverable); MACD *bands* are not,
# because the histogram was never logged.
POINT_PARAMS = ("rsi_oversold", "rsi_recovering", "macd_cross", "macd_bullish",
                "sma_support", "sma_above", "volume_high")
BAND_PARAMS = ("rsi_oversold_lo", "rsi_oversold_hi", "rsi_recovering_hi",
               "sma_support_lo", "sma_above_hi", "volume_ratio_min")
TUNABLE = POINT_PARAMS + BAND_PARAMS


def rsi_points(rsi: float | None, w: ScreenerWeights) -> float:
    """Oversold-bounce points. A missing RSI scores nothing rather than
    defaulting to neutral — shrub skips the term entirely on short history,
    and a fake-neutral RSI must not earn points."""
    if rsi is None:
        return 0.0
    if w.rsi_oversold_lo < rsi < w.rsi_oversold_hi:
        return w.rsi_oversold
    if w.rsi_oversold_hi <= rsi < w.rsi_recovering_hi:
        return w.rsi_recovering
    return 0.0


def macd_points(state: str | None, w: ScreenerWeights) -> float:
    if state == MACD_CROSS:
        return w.macd_cross
    if state == MACD_BULLISH:
        return w.macd_bullish
    return 0.0


def sma_points(price_vs_sma20: float | None, w: ScreenerWeights) -> float:
    """Points for sitting near the 20-day average. Note the incumbent rewards
    being just *below* it (support) more than just above — the mean-reversion
    thesis the whole screener rests on."""
    if price_vs_sma20 is None:
        return 0.0
    if w.sma_support_lo < price_vs_sma20 < 0:
        return w.sma_support
    if 0 <= price_vs_sma20 < w.sma_above_hi:
        return w.sma_above
    return 0.0


def volume_points(volume_ratio: float | None, w: ScreenerWeights) -> float:
    if volume_ratio is None:
        return 0.0
    return w.volume_high if volume_ratio > w.volume_ratio_min else 0.0


def score(rsi, macd_state, price_vs_sma20, volume_ratio,
          w: ScreenerWeights = INCUMBENT) -> float:
    """Total screener score under `w`. With INCUMBENT this equals the score
    shrub actually recorded."""
    return (rsi_points(rsi, w) + macd_points(macd_state, w)
            + sma_points(price_vs_sma20, w) + volume_points(volume_ratio, w))
