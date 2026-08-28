"""Rank information coefficient — the thing being optimised.

IC is the correlation between what the screener predicted and what actually
happened. Two choices are baked in and both matter:

RANK, not raw. Forward single-name returns have fat tails; one buyout would
dominate a Pearson correlation and drag the fit toward whatever ranked that one
name highly. Spearman bounds every observation's influence, so the fit answers
"did it order the names correctly", which is the job.

PER-DAY cross-sectional, then averaged — never pooled. Pooling every
(pick, day) pair mixes ordering skill with market beta: days when the whole
market rose lift every name at once. On shrub's own data the two disagree in
SIGN (5d: pooled +0.196 vs mean daily -0.039), and the pooled number is the
wrong one, because you only ever choose among a single day's candidates.

Averaging per-day ICs also hands you the significance test for free: IR is
mean/stdev and t is IR*sqrt(days). The promotion gate needs exactly that — a
higher IC that isn't distinguishable from noise must not ship.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# Below this many names a day's ordering carries almost no information and its
# IC is mostly noise, so it's excluded rather than averaged in.
MIN_NAMES_PER_DAY = 3


@dataclass(frozen=True)
class ICResult:
    mean_ic: float
    stdev_ic: float
    information_ratio: float
    t_stat: float
    n_days: int
    n_obs: int
    daily: list[float]

    def __str__(self) -> str:
        return (f"IC={self.mean_ic:+.4f} IR={self.information_ratio:+.2f} "
                f"t={self.t_stat:+.2f} over {self.n_days} days "
                f"({self.n_obs} obs)")


def spearman(x, y) -> float | None:
    """Spearman correlation. None when it isn't defined — fewer than three
    points, or one side constant (every name scoring the same tells you
    nothing about ordering, and would otherwise be a divide-by-zero)."""
    if len(x) < 3 or len(x) != len(y):
        return None
    rx = _ranks(np.asarray(x, dtype=float))
    ry = _ranks(np.asarray(y, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks, so ties don't get an arbitrary order. Ties are common
    here: the score is a sum of a few discrete point values, so many names in a
    day share one."""
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def rank_ic(predictions, outcomes, days, *,
            min_names: int = MIN_NAMES_PER_DAY) -> ICResult:
    """Mean daily cross-sectional rank IC, with its dispersion and t-stat.

    `days` groups the observations; one IC is computed per group and the
    results averaged. Groups too small to rank are skipped.
    """
    by_day: dict[object, list[tuple[float, float]]] = defaultdict(list)
    for p, o, d in zip(predictions, outcomes, days):
        by_day[d].append((p, o))

    daily: list[float] = []
    n_obs = 0
    for group in by_day.values():
        if len(group) < min_names:
            continue
        ic = spearman([g[0] for g in group], [g[1] for g in group])
        if ic is None:
            continue
        daily.append(ic)
        n_obs += len(group)

    if not daily:
        return ICResult(float("nan"), float("nan"), float("nan"), float("nan"),
                        0, 0, [])
    mean = float(np.mean(daily))
    stdev = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    ir = mean / stdev if stdev else float("nan")
    t = ir * math.sqrt(len(daily)) if stdev else float("nan")
    return ICResult(mean, stdev, ir, t, len(daily), n_obs, daily)


def score_weights(picks, weights, *, horizon: int) -> ICResult:
    """Re-score `picks` under `weights` and measure the result at one horizon."""
    subset = [p for p in picks if p.horizon_days == horizon]
    return rank_ic([p.rescore(weights) for p in subset],
                   [p.alpha for p in subset],
                   [p.pick_date for p in subset])
