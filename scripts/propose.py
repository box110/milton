"""Fit, gate, and — only on a PROMOTE — raise a proposal and email it.

Nothing is sent when the gate rejects. A rejection notice on every run trains
you to ignore the mail, and the UI already shows what the gate has been turning
down. Mail is reserved for the case that needs a human.
"""
import argparse
import asyncio
import sys

from milton import config, db, notify
from milton.features import build_dataset
from milton.gate import evaluate, split_by_time
from milton.objective import score_weights
from milton.optimize import SearchSpec, optimise
from milton.proposals import ProposalStore
from milton.screener import INCUMBENT


async def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true",
                    help="actually email an approved-by-the-gate proposal")
    ap.add_argument("--force", action="store_true",
                    help="raise a proposal even if the gate rejected it "
                         "(for exercising the loop; the mail says so)")
    args = ap.parse_args(argv)

    rows = await db.screener_picks()
    picks, _ = build_dataset(rows)
    await db.close_pool()
    h = config.PRIMARY_HORIZON
    picks = [p for p in picks if p.horizon_days == h]

    train, test = split_by_time(picks, test_fraction=0.4)
    cand = optimise(train, spec=SearchSpec(horizon=h))
    res = evaluate(train, test, cand.weights, INCUMBENT, horizon=h)

    inc = INCUMBENT.as_dict()
    got = cand.weights.as_dict()
    gate = {
        "verdict": res.verdict, "reasons": res.reasons, "horizon": h,
        "train_days": len({p.pick_date for p in train}), "test_days": res.test_days,
        "train_incumbent_ic": score_weights(train, INCUMBENT, horizon=h).mean_ic,
        "train_candidate_ic": cand.ic.mean_ic,
        "test_incumbent_ic": res.test_incumbent.mean_ic,
        "test_candidate_ic": res.test_candidate.mean_ic,
        "paired_mean": res.paired_mean, "paired_t": res.paired_t,
        "selection_incumbent": res.selection_incumbent,
        "selection_candidate": res.selection_candidate,
        "moves": [{"name": k, "from": inc[k], "to": got[k]}
                  for k in got if inc[k] != got[k]],
    }

    print(f"gate: {res}")
    for r in res.reasons:
        print(f"  - {r}")

    if not res.promoted and not args.force:
        print("\nno proposal raised — the gate rejected it, and a rejection "
              "is not something to email about")
        return 0

    store = ProposalStore(config.STATE_DB)
    note = "raised with --force despite a gate rejection" if not res.promoted else None
    proposal = store.create(horizon=h, weights=got, gate=gate, note=note)
    print(f"\nraised proposal #{proposal.id} ({proposal.status})")

    if not args.send:
        subject, body = notify.render(proposal)
        print(f"\n--- not sent (pass --send) ---\nSubject: {subject}\n\n{body}")
        return 0

    ok, detail = await notify.send(proposal)
    if ok:
        store.mark_sent(proposal.id)
    print(f"send: {'ok' if ok else 'FAILED'} — {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
