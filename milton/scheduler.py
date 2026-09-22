"""Run the fit daily, and raise a proposal only when the gate allows one.

Until now nothing ran the loop: `scripts/propose.py` existed and was only ever
invoked by hand, so a candidate could clear the gate and nobody would hear
about it. This closes that — the whole point of a promotion gate is that it
watches continuously and speaks up rarely.

Three restraints, all of which exist so the mail stays worth reading:

ONE RUN A DAY. The objective only moves when the backtester scores another
day, which happens once or twice daily. Fitting more often would burn cycles
re-deriving the same answer.

MAIL ONLY ON PROMOTE. A rejection notice every day teaches you to ignore the
sender, and the UI already shows what the gate has been turning down.

NEVER TWO OPEN PROPOSALS. If one is already awaiting a decision, the run is
recorded and no second mail goes out. Two live offers and one reply is an
ambiguity with no safe resolution.
"""
from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from . import config, db, notify
from .features import build_dataset
from .gate import evaluate, split_by_time
from .objective import score_weights
from .optimize import SearchSpec, optimise
from .proposals import ProposalStore
from .screener import INCUMBENT

# Checked this often; the daily guard decides whether anything actually runs.
_TICK_SECONDS = 600


@dataclass
class SchedulerState:
    enabled: bool = True
    last_run: datetime | None = None
    last_verdict: str | None = None
    last_reasons: list = field(default_factory=list)
    last_error: str | None = None
    last_proposal_id: int | None = None
    runs: int = 0

    @property
    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "runs": self.runs,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_verdict": self.last_verdict,
            "last_reasons": self.last_reasons,
            "last_proposal_id": self.last_proposal_id,
            "last_error": self.last_error,
            "run_hour_utc": config.PROPOSE_HOUR_UTC,
        }


state = SchedulerState()


async def run_once(store: ProposalStore, *, send: bool = True) -> dict:
    """Fit, gate, and propose if the gate allows. Returns what happened."""
    rows = await db.screener_picks()
    picks, _ = build_dataset(rows)
    horizon = config.PRIMARY_HORIZON
    picks = [p for p in picks if p.horizon_days == horizon]

    train, test = split_by_time(picks, test_fraction=0.4)
    if not train or not test:
        return {"verdict": "skipped", "reasons": ["not enough scored days yet"]}

    cand = optimise(train, spec=SearchSpec(horizon=horizon))
    res = evaluate(train, test, cand.weights, INCUMBENT, horizon=horizon)

    inc, got = INCUMBENT.as_dict(), cand.weights.as_dict()
    gate = {
        "verdict": res.verdict, "reasons": res.reasons, "horizon": horizon,
        "train_days": len({p.pick_date for p in train}), "test_days": res.test_days,
        "train_incumbent_ic": score_weights(train, INCUMBENT, horizon=horizon).mean_ic,
        "train_candidate_ic": cand.ic.mean_ic,
        "test_incumbent_ic": res.test_incumbent.mean_ic,
        "test_candidate_ic": res.test_candidate.mean_ic,
        "paired_mean": res.paired_mean, "paired_t": res.paired_t,
        "selection_incumbent": res.selection_incumbent,
        "selection_candidate": res.selection_candidate,
        "moves": [{"name": k, "from": inc[k], "to": got[k]}
                  for k in got if inc[k] != got[k]],
    }

    out = {"verdict": res.verdict, "reasons": res.reasons, "proposal_id": None}
    if not res.promoted:
        return out

    open_already = store.open_proposal()
    if open_already is not None:
        out["reasons"] = [f"proposal #{open_already.id} is still awaiting a "
                          f"decision; not raising a second"]
        out["verdict"] = "held"
        return out

    proposal = store.create(horizon=horizon, weights=got, gate=gate)
    out["proposal_id"] = proposal.id
    if send:
        ok, detail = await notify.send(proposal)
        if ok:
            store.mark_sent(proposal.id)
        out["sent"] = ok
        out["detail"] = detail
    return out


async def loop(store: ProposalStore) -> None:
    """Wake periodically; act once per day at the configured hour."""
    last_day: date | None = None
    while state.enabled:
        try:
            now = datetime.now(timezone.utc)
            due = (now.hour >= config.PROPOSE_HOUR_UTC and now.date() != last_day)
            if due:
                last_day = now.date()
                store.expire_stale()
                result = await run_once(store)
                state.runs += 1
                state.last_run = now
                state.last_verdict = result.get("verdict")
                state.last_reasons = result.get("reasons") or []
                state.last_proposal_id = result.get("proposal_id")
                state.last_error = None
                print(f"[milton/scheduler] {result.get('verdict')} — "
                      f"{'; '.join(state.last_reasons) or 'all checks passed'}"
                      + (f" (proposal #{result['proposal_id']})"
                         if result.get("proposal_id") else ""), flush=True)
        except Exception as e:
            # A scheduler that dies on one bad day stops watching entirely,
            # which is the failure mode this exists to prevent.
            state.last_error = f"{type(e).__name__}: {e}"
            print(f"[milton/scheduler] run failed: {state.last_error}\n"
                  f"{traceback.format_exc()}", flush=True)
        await asyncio.sleep(_TICK_SECONDS)
