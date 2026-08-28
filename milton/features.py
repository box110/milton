"""Turn shrub's logged picks into a labelled dataset milton can re-score.

One row per (pick, horizon): the component states that produced the pick, plus
the realised forward alpha versus SPY. This is the training set — it exists
already, as a side effect of `_log_screener_candidates` writing every screened
candidate to `discovery_picks` with its raw signals.

The one gap is MACD. shrub logs the total score but not `macd_histogram`, so
the MACD term is recovered by subtracting the three terms we CAN recompute and
reading what's left. That residual is only ever 0, 15 or 35 — the three values
the incumbent MACD term can take — which both identifies the state and acts as
a checksum on the decomposition: an unexpected residual means the stored score
disagrees with our model of it, and the row is dropped rather than guessed at.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .screener import (
    INCUMBENT, MACD_BULLISH, MACD_CROSS, MACD_NONE,
    ScreenerWeights, rsi_points, sma_points, volume_points, score,
)


@dataclass(frozen=True)
class Pick:
    """One screened candidate and what happened to it."""
    pick_id: int
    symbol: str
    pick_date: date
    horizon_days: int
    # Raw indicator values, as recorded at pick time.
    rsi: float | None
    volume_ratio: float | None
    price_vs_sma20: float | None
    macd_state: str
    stored_score: float
    # Outcome. `alpha` is the pick's return minus SPY's over the same sessions.
    alpha: float

    def rescore(self, w: ScreenerWeights) -> float:
        return score(self.rsi, self.macd_state, self.price_vs_sma20,
                     self.volume_ratio, w)


class DecompositionError(ValueError):
    """The stored score can't be explained by our model of the screener."""


def macd_state_from_residual(stored_score: float, rsi, price_vs_sma20,
                             volume_ratio, w: ScreenerWeights = INCUMBENT) -> str:
    """Recover the MACD term shrub applied but didn't log.

    Raises DecompositionError when the leftover isn't one of the three values
    the MACD term can produce — which would mean shrub's scoring has drifted
    from this model, and silently absorbing it would corrupt every fit built on
    top. Loud is correct here.
    """
    explained = (rsi_points(rsi, w) + sma_points(price_vs_sma20, w)
                 + volume_points(volume_ratio, w))
    residual = round(stored_score - explained, 6)
    if residual == w.macd_cross:
        return MACD_CROSS
    if residual == w.macd_bullish:
        return MACD_BULLISH
    if residual == 0:
        return MACD_NONE
    raise DecompositionError(
        f"score {stored_score} leaves residual {residual}, which is not a MACD "
        f"term ({w.macd_bullish} / {w.macd_cross} / 0) — shrub's scoring no "
        f"longer matches milton's model of it")


def build_pick(row: dict, w: ScreenerWeights = INCUMBENT) -> Pick:
    """Build one labelled row from a joined discovery_picks/returns record."""
    signals = row.get("source_signals") or {}
    rsi = _f(signals.get("rsi"))
    volume_ratio = _f(signals.get("volume_ratio"))
    price_vs_sma20 = _f(signals.get("price_vs_sma20"))
    stored = _f(signals.get("score"))
    if stored is None:
        raise DecompositionError(f"pick {row.get('id')} has no stored score")
    return Pick(
        pick_id=int(row["id"]),
        symbol=str(row["symbol"]),
        pick_date=row["created_at"].date(),
        horizon_days=int(row["horizon_days"]),
        rsi=rsi,
        volume_ratio=volume_ratio,
        price_vs_sma20=price_vs_sma20,
        macd_state=macd_state_from_residual(stored, rsi, price_vs_sma20, volume_ratio, w),
        stored_score=stored,
        alpha=float(row["alpha"]),
    )


def build_dataset(rows, w: ScreenerWeights = INCUMBENT) -> tuple[list[Pick], list[str]]:
    """Build the dataset, collecting rather than raising on bad rows. Returns
    (picks, problems) so a caller can see how much was dropped and why — a
    silent drop rate is how a fit ends up trained on an unrepresentative
    subset."""
    picks: list[Pick] = []
    problems: list[str] = []
    for row in rows:
        try:
            picks.append(build_pick(row, w))
        except (DecompositionError, KeyError, TypeError, ValueError) as e:
            problems.append(f"pick {row.get('id')}: {e}")
    return picks, problems


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f
