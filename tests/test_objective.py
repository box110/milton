"""Rank IC: ordering, tie handling, and the per-day vs pooled distinction."""
from datetime import date

from milton.objective import rank_ic, spearman


def test_perfect_and_inverted_ordering():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_rank_not_magnitude():
    """One extreme outcome must not dominate — the reason Spearman is used
    rather than Pearson on fat-tailed forward returns."""
    assert spearman([1, 2, 3, 4], [1, 2, 3, 1000]) == 1.0


def test_undefined_cases_return_none_not_zero():
    assert spearman([1, 2], [1, 2]) is None            # too few points
    assert spearman([5, 5, 5], [1, 2, 3]) is None      # constant predictions
    assert spearman([1, 2, 3], [7, 7, 7]) is None      # constant outcomes


def test_ties_get_average_ranks():
    """Scores are sums of a few discrete point values, so ties are everywhere.
    Tied names must get the same rank rather than an arbitrary order, which
    means the result cannot depend on how tied rows happen to be sorted."""
    a = spearman([10, 10, 20, 20], [1, 2, 3, 4])
    b = spearman([10, 10, 20, 20], [2, 1, 4, 3])   # swap within each tie group
    assert a == b
    # Tied predictions genuinely cannot order distinct outcomes perfectly.
    assert 0.0 < a < 1.0
    # Naive positional ranking would leak the input order into the answer.
    assert spearman([10, 10, 20, 20], [4, 3, 2, 1]) == -a


def test_ic_is_averaged_per_day_not_pooled():
    """Two days, each perfectly ordered internally, but with opposite level
    shifts. Per-day IC is +1; a pooled correlation would be dragged down by
    the across-day move, which is market beta the screener never chose."""
    d1, d2 = date(2026, 8, 3), date(2026, 8, 4)
    preds = [10, 20, 30, 10, 20, 30]
    outs = [1.0, 2.0, 3.0, -3.0, -2.0, -1.0]
    days = [d1, d1, d1, d2, d2, d2]
    res = rank_ic(preds, outs, days)
    assert res.mean_ic == 1.0
    assert res.n_days == 2 and res.n_obs == 6
    assert spearman(preds, outs) < 1.0        # pooled disagrees


def test_thin_days_are_excluded_not_averaged_in():
    d1, d2 = date(2026, 8, 3), date(2026, 8, 4)
    res = rank_ic([10, 20, 30, 10, 20], [1.0, 2.0, 3.0, 1.0, 2.0],
                  [d1, d1, d1, d2, d2])
    assert res.n_days == 1 and res.n_obs == 3


def test_significance_stats_reported():
    days = [date(2026, 8, d) for d in (3, 4, 5) for _ in range(3)]
    res = rank_ic([10, 20, 30] * 3, [1.0, 2.0, 3.0] * 3, days)
    assert res.mean_ic == 1.0
    assert res.stdev_ic == 0.0
    assert res.n_days == 3


def test_empty_input_is_not_a_crash():
    res = rank_ic([], [], [])
    assert res.n_days == 0 and res.daily == []
