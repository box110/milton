"""Fit candidate screener weights and put them through the gate.

Reports whatever it finds, promotes nothing on its own — the gate's verdict is
an input to a decision, not the decision.
"""
import asyncio
import sys

from milton import config, db
from milton.features import build_dataset
from milton.gate import evaluate
from milton.gate import split_by_time
from milton.objective import score_weights
from milton.optimize import SearchSpec, optimise
from milton.screener import INCUMBENT


async def main() -> int:
    rows = await db.screener_picks()
    picks, problems = build_dataset(rows)
    await db.close_pool()
    h = config.PRIMARY_HORIZON
    picks = [p for p in picks if p.horizon_days == h]

    train, test = split_by_time(picks, test_fraction=0.4)
    tr_days = len({p.pick_date for p in train})
    te_days = len({p.pick_date for p in test})
    print(f"{len(picks)} picks at {h}d over "
          f"{tr_days + te_days} days  (train {tr_days} / test {te_days})")
    if problems:
        print(f"  {len(problems)} row(s) dropped")

    print(f"\nincumbent  train  {score_weights(train, INCUMBENT, horizon=h)}")
    print(f"incumbent  test   {score_weights(test, INCUMBENT, horizon=h)}")

    spec = SearchSpec(horizon=h)
    cand = optimise(train, spec=spec)
    print(f"\nsearch: {cand.evaluations} evaluations over {cand.sweeps} sweep(s), "
          f"distance {cand.distance:.3f}")
    if cand.moves:
        for m in cand.moves:
            print(f"  {m}")
    else:
        print("  no move improved on the incumbent")
    print(f"candidate  train  {cand.ic}")

    res = evaluate(train, test, cand.weights, INCUMBENT, horizon=h)
    print(f"\ncandidate  test   {res.test_candidate}")
    print(f"\nGATE: {res}")
    for r in res.reasons:
        print(f"  - {r}")
    print(f"\nnames clearing MIN_SCORE on test: "
          f"{res.selection_incumbent} -> {res.selection_candidate}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
