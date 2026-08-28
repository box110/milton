"""Search the screener's weight space for a higher rank IC.

Deliberately a plain coordinate descent rather than anything clever. The search
space is small, the objective is cheap, and the real risk here is not a weak
optimiser — it is a strong one. With a handful of screening days on record, any
method flexible enough to chase the objective hard will find a beautiful
in-sample fit that means nothing. Transparency is worth more than power: this
walks one parameter at a time over a coarse grid, so every move it makes can be
read off and argued with.

Two guards on that risk live here rather than in the gate, because they shape
what the search will even consider:

SHRINKAGE. The objective is IC minus a penalty on distance from the incumbent,
so a departure has to pay for itself. The incumbent is not sacred, but it is
the prior — a candidate that moves every parameter to win +0.001 IC is
overwhelmingly likely to be reading noise.

A RESTRICTED PARAMETER SET. By default only the point values move and the band
edges stay put. Fitting thresholds as well as weights roughly doubles the
degrees of freedom for a sample that cannot support the ones it already has.
"""
from __future__ import annotations

from dataclasses import dataclass

from .objective import ICResult, score_weights
from .screener import BAND_PARAMS, POINT_PARAMS, ScreenerWeights

# Points may go negative. The per-component ICs suggest at least one term (the
# volume bonus) may be actively anti-predictive at the 20-day horizon, and a
# floor of zero would only let the search neutralise it rather than reverse it.
POINT_BOUNDS = (-30.0, 60.0)

BAND_BOUNDS = {
    "rsi_oversold_lo": (10.0, 35.0),
    "rsi_oversold_hi": (25.0, 50.0),
    "rsi_recovering_hi": (35.0, 65.0),
    "sma_support_lo": (-15.0, -1.0),
    "sma_above_hi": (1.0, 15.0),
    "volume_ratio_min": (1.0, 3.0),
}


@dataclass(frozen=True)
class SearchSpec:
    """What the search is allowed to do."""
    horizon: int = 20
    # Grid step for point values, in points. Coarse on purpose: the data cannot
    # resolve a 1-point difference, and pretending otherwise invites the search
    # to fit noise to three decimal places.
    point_step: float = 5.0
    band_steps: int = 6
    max_sweeps: int = 8
    # Penalty per unit of normalised distance from the incumbent. At 0.05 a
    # single weight crossing its full range costs about 0.045 IC — of the same
    # order as the entire edge the incumbent has, which is the point: on this
    # much data, a move that size should need overwhelming evidence.
    shrinkage: float = 0.05
    tune_bands: bool = False

    def params(self) -> tuple[str, ...]:
        return POINT_PARAMS + BAND_PARAMS if self.tune_bands else POINT_PARAMS


def _distance(w: ScreenerWeights, incumbent: ScreenerWeights,
              params: tuple[str, ...]) -> float:
    """Normalised L1 distance from the incumbent — summed, not averaged.

    Averaging was wrong and quietly defeated the whole guard: dividing by the
    parameter count made a single parameter's move look seven times smaller
    than it is, so one weight could wander the full width of its range for
    almost no penalty. On pure noise the search duly moved rsi_oversold from
    30 to -5. Summing means moving one parameter across its range costs the
    same whatever else is in the vector, which is what "this departure has to
    pay for itself" was supposed to mean."""
    total = 0.0
    for name in params:
        lo, hi = BAND_BOUNDS.get(name, POINT_BOUNDS)
        span = (hi - lo) or 1.0
        total += abs(getattr(w, name) - getattr(incumbent, name)) / span
    return total


@dataclass(frozen=True)
class Candidate:
    weights: ScreenerWeights
    ic: ICResult
    penalised: float
    distance: float
    evaluations: int
    sweeps: int
    moves: list[str]


def penalised_objective(picks, w: ScreenerWeights, incumbent: ScreenerWeights,
                        spec: SearchSpec) -> tuple[float, ICResult, float]:
    ic = score_weights(picks, w, horizon=spec.horizon)
    if ic.n_days == 0:
        return float("-inf"), ic, 0.0
    dist = _distance(w, incumbent, spec.params())
    return ic.mean_ic - spec.shrinkage * dist, ic, dist


def _grid(name: str, spec: SearchSpec) -> list[float]:
    if name in BAND_BOUNDS:
        lo, hi = BAND_BOUNDS[name]
        step = (hi - lo) / spec.band_steps
        return [lo + i * step for i in range(spec.band_steps + 1)]
    lo, hi = POINT_BOUNDS
    n = int(round((hi - lo) / spec.point_step))
    return [lo + i * spec.point_step for i in range(n + 1)]


def optimise(picks, incumbent: ScreenerWeights | None = None,
             spec: SearchSpec | None = None) -> Candidate:
    """Coordinate descent from the incumbent. Returns the best candidate found.

    Starting at the incumbent rather than a random point is deliberate: it makes
    "no change" the natural outcome of a search that finds nothing, and it means
    every reported move is a move away from what is already running.
    """
    spec = spec or SearchSpec()
    incumbent = incumbent or ScreenerWeights()
    params = spec.params()

    best = incumbent
    best_obj, best_ic, best_dist = penalised_objective(picks, best, incumbent, spec)
    evaluations = 1
    moves: list[str] = []
    sweeps = 0

    for sweep in range(spec.max_sweeps):
        sweeps = sweep + 1
        improved = False
        for name in params:
            current = getattr(best, name)
            for value in _grid(name, spec):
                if value == current:
                    continue
                trial = best.replace(**{name: value})
                obj, ic, dist = penalised_objective(picks, trial, incumbent, spec)
                evaluations += 1
                # Strict improvement only. Accepting ties lets the search drift
                # away from the incumbent for free, which is exactly what the
                # shrinkage term exists to prevent.
                if obj > best_obj:
                    best, best_obj, best_ic, best_dist = trial, obj, ic, dist
                    current = value
                    improved = True
        if not improved:
            break

    for name in params:
        was, now = getattr(incumbent, name), getattr(best, name)
        if was != now:
            moves.append(f"{name}: {was:g} -> {now:g}")

    return Candidate(weights=best, ic=best_ic, penalised=best_obj,
                     distance=best_dist, evaluations=evaluations,
                     sweeps=sweeps, moves=moves)
