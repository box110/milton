"""Milton's web UI — a read-only window on the weight-fitting loop.

The point of this UI is to make the loop legible: what data exists, what the
incumbent weights actually score, and which stages are built versus still
sketched. Stages that don't exist yet report `built: false` and render as such
rather than showing plausible-looking placeholder numbers — a scaffold that
fakes its own output is worse than no scaffold, because you start reasoning
about results that were never computed.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .features import build_dataset
from .objective import score_weights
from .screener import BAND_PARAMS, INCUMBENT, POINT_PARAMS, TUNABLE

STATIC = Path(__file__).parent / "static"

# The loop, in order. `built` is the honest state of the code, not a roadmap
# aspiration — the UI colours stages by it.
PIPELINE = [
    {"id": "source", "name": "shrub picks", "kind": "source", "built": True,
     "detail": "discovery_picks joined to discovery_pick_returns — every screened "
               "candidate with a scored +1/+5/+20d outcome vs SPY."},
    {"id": "labels", "name": "Decompose", "kind": "milton", "built": True,
     "detail": "Recover each pick's component states from its logged signals. "
               "MACD was never logged, so its term is recovered as a residual."},
    {"id": "replay", "name": "Replay", "kind": "milton", "built": True,
     "detail": "Re-score every pick under a candidate weight vector. Pure "
               "arithmetic over stored values — no market data refetch."},
    {"id": "objective", "name": "Rank IC", "kind": "milton", "built": True,
     "detail": "Mean daily cross-sectional Spearman IC vs forward alpha, with "
               "IR and a t-stat."},
    {"id": "optimize", "name": "Optimise", "kind": "milton", "built": False,
     "detail": "Search the weight space for a higher out-of-sample IC."},
    {"id": "gate", "name": "OOS gate", "kind": "milton", "built": False,
     "detail": "Time-based holdout plus a margin over the incumbent. Its job is "
               "to say no most of the time."},
    {"id": "approve", "name": "Your approval", "kind": "human", "built": False,
     "detail": "Emails a summary tagged [milton #N]; your reply approves or "
               "rejects. shrub's inbound router hands the reply back instead of "
               "feeding it to the CEO agent."},
    {"id": "writeback", "name": "shrub settings", "kind": "sink", "built": False,
     "detail": "Approved weights land in tenant_settings and take effect on the "
               "next screening cycle."},
]


def _weight_rows() -> list[dict]:
    """The parameter vector, annotated with what the logged data can support.

    MACD points are tunable because the state is recoverable from the residual;
    MACD *bands* are not, because shrub logs the score but not the histogram.
    Saying so in the UI is the point — it's the difference between a parameter
    we can fit and one we're guessing at forever.
    """
    values = INCUMBENT.as_dict()
    rows = []
    for name in values:
        rows.append({
            "name": name,
            "value": values[name],
            "group": "points" if name in POINT_PARAMS else "band",
            "tunable": name in TUNABLE,
        })
    rows.append({
        "name": "macd_histogram_threshold", "value": None, "group": "band",
        "tunable": False,
        "note": "not tunable — shrub logs the score but not macd_histogram",
    })
    return rows


app = FastAPI(title="Milton")


@app.get("/api/state")
async def state():
    rows = await db.screener_picks()
    picks, problems = build_dataset(rows)

    mismatched = sum(1 for p in picks
                     if abs(p.rescore(INCUMBENT) - p.stored_score) > 1e-9)

    horizons = {}
    for h in (1, config.SECONDARY_HORIZON, config.PRIMARY_HORIZON):
        r = score_weights(picks, INCUMBENT, horizon=h)
        horizons[h] = {
            "horizon": h, "mean_ic": r.mean_ic, "stdev_ic": r.stdev_ic,
            "information_ratio": r.information_ratio, "t_stat": r.t_stat,
            "n_days": r.n_days, "n_obs": r.n_obs,
            "primary": h == config.PRIMARY_HORIZON,
        }

    primary = [p for p in picks if p.horizon_days == config.PRIMARY_HORIZON]
    daily = _daily_series(primary)

    return {
        "pipeline": PIPELINE,
        "weights": _weight_rows(),
        "dataset": {
            "rows": len(rows), "picks": len(picks), "dropped": len(problems),
            "problems": problems[:10],
            "replay_exact": len(picks) - mismatched,
            "replay_total": len(picks),
        },
        "horizons": list(horizons.values()),
        "primary_horizon": config.PRIMARY_HORIZON,
        "daily_ic": daily,
    }


def _daily_series(picks) -> list[dict]:
    """Per-day IC at the primary horizon — the dispersion that IR and t
    summarise. Seeing it matters: a healthy mean IC built from a few wild days
    is a different animal from a consistent one, and only the series shows
    which you have."""
    from collections import defaultdict

    from .objective import MIN_NAMES_PER_DAY, spearman

    by_day = defaultdict(list)
    for p in picks:
        by_day[p.pick_date].append(p)

    out = []
    for day in sorted(by_day):
        group = by_day[day]
        if len(group) < MIN_NAMES_PER_DAY:
            continue
        ic = spearman([p.rescore(INCUMBENT) for p in group],
                      [p.alpha for p in group])
        if ic is None:
            continue
        out.append({"date": day.isoformat(), "ic": ic, "n": len(group)})
    return out


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
