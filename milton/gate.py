"""The out-of-sample promotion gate.

Its job is to say no. An optimiser will always return something that beat the
incumbent on the data it was fitted to — that is what optimisers do — so the
only question that matters is whether the improvement survives on days the
search never saw. Everything here is built around making that hard to fake.

THE SPLIT IS BY TIME. Earlier days train, later days test, never shuffled. A
random split leaks: two picks from the same day share that day's market move,
so a shuffled test set is partly the training set wearing a hat.

THE COMPARISON IS PAIRED. Candidate and incumbent rank the SAME names on the
SAME day, so the two ICs can be differenced day by day. That removes the
day-to-day variation both share — which is most of it — and turns "is A better
than B" into a one-sample test on the differences. Comparing two independent
means would need far more days to see the same effect.

THE MARGIN COMES FROM THE DATA. There is no hand-picked "must beat by 0.02".
The bar is a t-statistic on the paired differences, so a candidate is judged
against how noisy its own improvement is. A big improvement that swings wildly
does not pass; a small consistent one can.

The gate also refuses on grounds that have nothing to do with the objective: too
few test days to conclude anything, and a change that would alter how many names
clear MIN_SCORE, which silently resizes the watchlist even when the ranking
improves.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from statistics import mean, stdev

from .objective import MIN_NAMES_PER_DAY, ICResult, rank_ic, spearman
from .screener import ScreenerWeights

# Days of held-out data below which no promotion is possible, whatever the
# numbers say. With fewer than this the paired t-statistic is not measuring
# anything you would want to act on.
MIN_TEST_DAYS = 12

# Paired t the improvement must clear on the test window.
MIN_PAIRED_T = 2.0

# shrub's own MIN_SCORE. Changing point weights moves names across it, which
# changes watchlist size without touching the ranking the gate is judging.
SHRUB_MIN_SCORE = 30.0

# How much the count of names clearing MIN_SCORE may move, as a fraction.
MAX_SELECTION_DRIFT = 0.25


@dataclass(frozen=True)
class GateResult:
    verdict: str                      # "promote" | "reject"
    reasons: list[str]
    train_days: int
    test_days: int
    train_candidate: ICResult
    test_incumbent: ICResult
    test_candidate: ICResult
    paired_mean: float
    paired_t: float
    selection_incumbent: int
    selection_candidate: int
    daily_diffs: list[float] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return self.verdict == "promote"

    def __str__(self) -> str:
        return (f"{self.verdict.upper()}: test IC "
                f"{self.test_incumbent.mean_ic:+.4f} -> "
                f"{self.test_candidate.mean_ic:+.4f} "
                f"(paired {self.paired_mean:+.4f}, t={self.paired_t:+.2f}, "
                f"{self.test_days} test days)")


def split_by_time(picks, test_fraction: float = 0.4) -> tuple[list, list]:
    """Split into (train, test) on the calendar, oldest first.

    Whole days go to one side or the other. Splitting within a day would put
    names that shared a market move on both sides of the wall."""
    days = sorted({p.pick_date for p in picks})
    if not days:
        return [], []
    n_test = max(1, int(round(len(days) * test_fraction)))
    cutoff_index = max(0, len(days) - n_test)
    test_days = set(days[cutoff_index:])
    train = [p for p in picks if p.pick_date not in test_days]
    test = [p for p in picks if p.pick_date in test_days]
    return train, test


def paired_daily_ic(picks, a: ScreenerWeights, b: ScreenerWeights,
                    *, min_names: int = MIN_NAMES_PER_DAY
                    ) -> tuple[list[float], list[float], list[date]]:
    """Per-day IC under each weight vector, over identical name sets.

    A day is only used when BOTH sides produce a defined IC — if one ranks every
    name the same, its IC is undefined and the pair would not be comparable."""
    by_day = defaultdict(list)
    for p in picks:
        by_day[p.pick_date].append(p)

    ics_a: list[float] = []
    ics_b: list[float] = []
    days: list[date] = []
    for day in sorted(by_day):
        group = by_day[day]
        if len(group) < min_names:
            continue
        outcomes = [p.alpha for p in group]
        ic_a = spearman([p.rescore(a) for p in group], outcomes)
        ic_b = spearman([p.rescore(b) for p in group], outcomes)
        if ic_a is None or ic_b is None:
            continue
        ics_a.append(ic_a)
        ics_b.append(ic_b)
        days.append(day)
    return ics_a, ics_b, days


def _selection_count(picks, w: ScreenerWeights) -> int:
    return sum(1 for p in picks if p.rescore(w) >= SHRUB_MIN_SCORE)


def evaluate(train, test, candidate: ScreenerWeights,
             incumbent: ScreenerWeights, *, horizon: int = 20,
             min_test_days: int = MIN_TEST_DAYS,
             min_paired_t: float = MIN_PAIRED_T) -> GateResult:
    """Judge a candidate on held-out days. Rejects unless every check passes."""
    train = [p for p in train if p.horizon_days == horizon]
    test = [p for p in test if p.horizon_days == horizon]

    train_cand = rank_ic([p.rescore(candidate) for p in train],
                         [p.alpha for p in train], [p.pick_date for p in train])

    cand_ics, inc_ics, days = paired_daily_ic(test, candidate, incumbent)
    test_cand = rank_ic([p.rescore(candidate) for p in test],
                        [p.alpha for p in test], [p.pick_date for p in test])
    test_inc = rank_ic([p.rescore(incumbent) for p in test],
                       [p.alpha for p in test], [p.pick_date for p in test])

    diffs = [c - i for c, i in zip(cand_ics, inc_ics)]
    paired_mean = mean(diffs) if diffs else float("nan")
    if len(diffs) > 1:
        sd = stdev(diffs)
        paired_t = (paired_mean / (sd / len(diffs) ** 0.5)) if sd else float("inf")
    else:
        paired_t = float("nan")

    sel_inc = _selection_count(test, incumbent)
    sel_cand = _selection_count(test, candidate)

    reasons: list[str] = []

    if candidate.as_dict() == incumbent.as_dict():
        reasons.append("candidate is identical to the incumbent — nothing to promote")

    if len(days) < min_test_days:
        reasons.append(
            f"only {len(days)} usable test day(s), need {min_test_days} — too few "
            f"to tell an improvement from noise")

    if diffs and paired_mean <= 0:
        reasons.append(
            f"no improvement out of sample (paired mean {paired_mean:+.4f})")
    elif diffs and not (paired_t >= min_paired_t):
        reasons.append(
            f"improvement not distinguishable from noise "
            f"(paired t={paired_t:+.2f}, need >= {min_paired_t})")

    if test_cand.n_days and test_cand.mean_ic <= 0:
        reasons.append(
            f"candidate has no edge out of sample (test IC {test_cand.mean_ic:+.4f})")

    # A zero incumbent count is the most extreme drift there is, not a case to
    # skip: guarding the division by falling through let "selects nothing" ->
    # "selects everything" past the check unexamined.
    if sel_inc == 0 and sel_cand > 0:
        reasons.append(
            f"would start selecting {sel_cand} name(s) where the incumbent "
            f"selects none — re-fit MIN_SCORE before promoting these weights")
    elif sel_inc > 0:
        drift = abs(sel_cand - sel_inc) / sel_inc
        if drift > MAX_SELECTION_DRIFT:
            reasons.append(
                f"would resize the watchlist by {drift:.0%} "
                f"({sel_inc} -> {sel_cand} names clearing MIN_SCORE) — re-fit "
                f"MIN_SCORE before promoting these weights")

    verdict = "promote" if not reasons else "reject"
    return GateResult(
        verdict=verdict, reasons=reasons,
        train_days=rank_ic([p.rescore(incumbent) for p in train],
                           [p.alpha for p in train],
                           [p.pick_date for p in train]).n_days,
        test_days=len(days), train_candidate=train_cand,
        test_incumbent=test_inc, test_candidate=test_cand,
        paired_mean=paired_mean, paired_t=paired_t,
        selection_incumbent=sel_inc, selection_candidate=sel_cand,
        daily_diffs=diffs,
    )
