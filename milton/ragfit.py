"""Search over retrieval source weights, reusing the screener's machinery.

Same shape as `optimize.py` — coordinate descent from the incumbent, coarse
grid, shrinkage penalty, strict improvement only — for the same reason: on this
much data the risk is an optimiser strong enough to fit noise, not one too weak
to find signal.

One difference matters. The screener's score is scale-invariant under rank IC,
so only ratios between weights mattered. That is true here too, which means the
absolute level of a reliability weight is unidentifiable from ranking alone:
doubling every weight changes nothing. The search is therefore over the
weights' RELATIVE sizes, and any candidate should be read that way — as "news
should count more than social, by this much", never as "news is worth 0.62".
"""
from __future__ import annotations

from dataclasses import dataclass

from collections import defaultdict
from statistics import mean, stdev

from .objective import MIN_NAMES_PER_DAY, ICResult, rank_ic, spearman
from .rag import (
    BOUNDS, HALF_LIFE_PARAMS, RELIABILITY_PARAMS, SourceWeights,
)

# Steps per parameter across its range. Coarse deliberately: the data cannot
# resolve a 0.01 difference in a reliability weight.
GRID_STEPS = 10
MAX_SWEEPS = 6
SHRINKAGE = 0.05


@dataclass(frozen=True)
class SourceCandidate:
    weights: SourceWeights
    ic: ICResult
    penalised: float
    distance: float
    evaluations: int
    moves: list[str]


def split_days(days, test_fraction: float = 0.4):
    """Chronological split on whole days, same rule as the screener gate: names
    sharing a day share that day's market move."""
    all_days = sorted({d.day for d in days})
    if not all_days:
        return [], []
    n_test = max(1, int(round(len(all_days) * test_fraction)))
    test_days = set(all_days[len(all_days) - n_test:])
    return ([d for d in days if d.day not in test_days],
            [d for d in days if d.day in test_days])


def _score_ic(days, w: SourceWeights) -> ICResult:
    return rank_ic([d.score(w) for d in days], [d.alpha for d in days],
                   [d.day for d in days])


def _distance(w: SourceWeights, incumbent: SourceWeights,
              params: tuple[str, ...]) -> float:
    total = 0.0
    for name in params:
        lo, hi = BOUNDS[name]
        span = (hi - lo) or 1.0
        total += abs(getattr(w, name) - getattr(incumbent, name)) / span
    return total


def _grid(name: str) -> list[float]:
    lo, hi = BOUNDS[name]
    step = (hi - lo) / GRID_STEPS
    return [lo + i * step for i in range(GRID_STEPS + 1)]


def optimise_sources(days, incumbent: SourceWeights | None = None, *,
                     tune_half_lives: bool = True,
                     shrinkage: float = SHRINKAGE,
                     max_sweeps: int = MAX_SWEEPS) -> SourceCandidate:
    incumbent = incumbent or SourceWeights()
    params = RELIABILITY_PARAMS + (HALF_LIFE_PARAMS if tune_half_lives else ())

    best = incumbent
    best_ic = _score_ic(days, best)
    best_obj = (best_ic.mean_ic if best_ic.n_days else float("-inf"))
    evaluations = 1

    for _ in range(max_sweeps):
        improved = False
        for name in params:
            for value in _grid(name):
                if value == getattr(best, name):
                    continue
                trial = best.replace(**{name: value})
                ic = _score_ic(days, trial)
                evaluations += 1
                if not ic.n_days:
                    continue
                obj = ic.mean_ic - shrinkage * _distance(trial, incumbent, params)
                if obj > best_obj:
                    best, best_ic, best_obj = trial, ic, obj
                    improved = True
        if not improved:
            break

    moves = [f"{n}: {getattr(incumbent, n):g} -> {getattr(best, n):g}"
             for n in params if getattr(incumbent, n) != getattr(best, n)]
    return SourceCandidate(
        weights=best, ic=best_ic, penalised=best_obj,
        distance=_distance(best, incumbent, params),
        evaluations=evaluations, moves=moves)


# --- Gate ------------------------------------------------------------------
#
# Same bar as the screener's, minus the MIN_SCORE check, which has no analogue
# here: retrieval weights change which chunks are read, not how many names
# reach a trader.

MIN_TEST_DAYS = 12
MIN_PAIRED_T = 2.0


@dataclass(frozen=True)
class SourceGateResult:
    verdict: str
    reasons: list[str]
    test_days: int
    test_incumbent: ICResult
    test_candidate: ICResult
    paired_mean: float
    paired_t: float

    @property
    def promoted(self) -> bool:
        return self.verdict == "promote"

    def __str__(self) -> str:
        return (f"{self.verdict.upper()}: test IC "
                f"{self.test_incumbent.mean_ic:+.4f} -> "
                f"{self.test_candidate.mean_ic:+.4f} "
                f"(paired {self.paired_mean:+.4f}, t={self.paired_t:+.2f}, "
                f"{self.test_days} test days)")


def evaluate_sources(test, candidate: SourceWeights, incumbent: SourceWeights,
                     *, min_test_days: int = MIN_TEST_DAYS,
                     min_paired_t: float = MIN_PAIRED_T) -> SourceGateResult:
    """Judge candidate source weights on held-out days, paired day by day."""
    by_day = defaultdict(list)
    for d in test:
        by_day[d.day].append(d)

    diffs, days_used = [], 0
    for day in sorted(by_day):
        group = by_day[day]
        if len(group) < MIN_NAMES_PER_DAY:
            continue
        outcomes = [d.alpha for d in group]
        ic_c = spearman([d.score(candidate) for d in group], outcomes)
        ic_i = spearman([d.score(incumbent) for d in group], outcomes)
        if ic_c is None or ic_i is None:
            continue
        diffs.append(ic_c - ic_i)
        days_used += 1

    paired_mean = mean(diffs) if diffs else float("nan")
    if len(diffs) > 1:
        sd = stdev(diffs)
        paired_t = (paired_mean / (sd / len(diffs) ** 0.5)) if sd else float("inf")
    else:
        paired_t = float("nan")

    test_c = _score_ic(test, candidate)
    test_i = _score_ic(test, incumbent)

    reasons: list[str] = []
    if candidate.as_dict() == incumbent.as_dict():
        reasons.append("candidate is identical to the incumbent")
    if days_used < min_test_days:
        reasons.append(
            f"only {days_used} usable test day(s), need {min_test_days}")
    if diffs and paired_mean <= 0:
        reasons.append(
            f"no improvement out of sample (paired mean {paired_mean:+.4f})")
    elif diffs and not (paired_t >= min_paired_t):
        reasons.append(
            f"improvement not distinguishable from noise (paired "
            f"t={paired_t:+.2f}, need >= {min_paired_t})")

    return SourceGateResult(
        verdict="promote" if not reasons else "reject", reasons=reasons,
        test_days=days_used, test_incumbent=test_i, test_candidate=test_c,
        paired_mean=paired_mean, paired_t=paired_t)
