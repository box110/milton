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

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .features import build_dataset
from .gate import MIN_PAIRED_T, MIN_TEST_DAYS, evaluate, split_by_time
from .objective import score_weights
from .optimize import SearchSpec, optimise
from .approval import parse_decision
from .proposals import APPROVED, ProposalStore
from .screener import BAND_PARAMS, INCUMBENT, POINT_PARAMS, TUNABLE

STATIC = Path(__file__).parent / "static"

store = ProposalStore(config.STATE_DB)

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
    {"id": "optimize", "name": "Optimise", "kind": "milton", "built": True,
     "detail": "Coordinate descent from the incumbent, one parameter at a time, "
               "paying a shrinkage penalty on every departure."},
    {"id": "gate", "name": "OOS gate", "kind": "milton", "built": True,
     "detail": "Chronological holdout, paired day-by-day against the incumbent. "
               "Its job is to say no, and it usually does."},
    {"id": "approve", "name": "Your approval", "kind": "human", "built": True,
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
        "fit": _fit(primary),
        "proposals": {
            "open": (lambda o: _proposal_json(o) if o else None)(store.open_proposal()),
            "recent": [_proposal_json(p) for p in store.recent(limit=8)],
        },
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


def _fit(picks) -> dict:
    """Run the search and the gate, and report both. The candidate shown here
    is never live — it is what the search proposes and what the gate made of
    it, which is usually a rejection."""
    train, test = split_by_time(picks, test_fraction=0.4)
    if not train or not test:
        return {"available": False,
                "note": "not enough scored days to split into train and test yet"}

    spec = SearchSpec(horizon=config.PRIMARY_HORIZON)
    cand = optimise(train, spec=spec)
    res = evaluate(train, test, cand.weights, INCUMBENT,
                   horizon=config.PRIMARY_HORIZON)

    incumbent_values = INCUMBENT.as_dict()
    candidate_values = cand.weights.as_dict()
    return {
        "available": True,
        "verdict": res.verdict,
        "reasons": res.reasons,
        "train_days": len({p.pick_date for p in train}),
        "test_days": res.test_days,
        "min_test_days": MIN_TEST_DAYS,
        "min_paired_t": MIN_PAIRED_T,
        "evaluations": cand.evaluations,
        "train_incumbent_ic": score_weights(train, INCUMBENT,
                                            horizon=config.PRIMARY_HORIZON).mean_ic,
        "train_candidate_ic": cand.ic.mean_ic,
        "test_incumbent_ic": res.test_incumbent.mean_ic,
        "test_candidate_ic": res.test_candidate.mean_ic,
        "paired_mean": res.paired_mean,
        "paired_t": res.paired_t,
        "selection_incumbent": res.selection_incumbent,
        "selection_candidate": res.selection_candidate,
        "moves": [
            {"name": k, "from": incumbent_values[k], "to": candidate_values[k]}
            for k in candidate_values if incumbent_values[k] != candidate_values[k]
        ],
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


def _proposal_json(p) -> dict:
    return {
        "id": p.id, "status": p.status,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "sent_at": p.sent_at.isoformat() if p.sent_at else None,
        "decided_at": p.decided_at.isoformat() if p.decided_at else None,
        "decided_by": p.decided_by, "note": p.note,
        "horizon": p.horizon, "weights": p.weights, "gate": p.gate,
    }


@app.get("/api/proposals")
async def proposals():
    store.expire_stale()
    return {"open": (lambda o: _proposal_json(o) if o else None)(store.open_proposal()),
            "recent": [_proposal_json(p) for p in store.recent()]}


@app.post("/api/approval")
async def approval(request: Request):
    """Receive an emailed reply, relayed by shrub's inbound router.

    shrub intercepts `[milton #N]` subjects before its own operator->CEO branch
    and posts the reply here. The decision is parsed strictly and applied at
    most once; a repeat delivery reports what the proposal already was rather
    than deciding it again."""
    if config.INTERNAL_TOKEN and \
            request.headers.get("X-Internal-Token") != config.INTERNAL_TOKEN:
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    data = await request.json()
    try:
        proposal_id = int(data.get("proposal_id"))
    except (TypeError, ValueError):
        return JSONResponse({"detail": "proposal_id required"}, status_code=400)

    body = data.get("body") or ""
    decision, why = parse_decision(body)
    if decision is None:
        return {"status": "no_decision", "detail": why, "proposal_id": proposal_id}

    proposal, outcome = store.decide(
        proposal_id, decision, by=data.get("sender"), reply_text=body[:2000])
    if outcome == "not_found":
        return JSONResponse({"detail": f"no proposal #{proposal_id}"},
                            status_code=404)
    return {
        "status": outcome, "decision": proposal.status,
        "proposal_id": proposal_id,
        # The caller emails this back to the sender, so it has to read as an
        # answer to a person rather than an API response.
        "detail": {
            "applied": f"Recorded: proposal #{proposal_id} {proposal.status}.",
            "already_decided": (f"Proposal #{proposal_id} was already "
                                f"{proposal.status} — no change made."),
            "expired": (f"Proposal #{proposal_id} expired before this reply "
                        f"arrived, so it was not applied."),
        }.get(outcome, outcome),
        "writeback": (
            "Recorded only — writing weights into shrub is not built yet."
            if proposal.status == APPROVED else None),
    }


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
