"""Fit the retrieval source weights, and put them through the same gate.

Reports; promotes nothing. The objective is deliberately the same rank IC used
for the screener, so the two fits are directly comparable and the gate needs no
special case.
"""
import argparse
import asyncio
import sys
from collections import Counter

from milton import config, db
from milton.objective import rank_ic
from milton.rag import (
    FITTABLE_SOURCES, INCUMBENT, WORLD_SCOPED_SOURCES, SourceWeights,
    build_evidence_days,
)
from milton.ragfit import evaluate_sources, optimise_sources, split_days


def _ic(days, w):
    return rank_ic([d.score(w) for d in days], [d.alpha for d in days],
                   [d.day for d in days])


async def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=90)
    args = ap.parse_args(argv)

    rows = await db.evidence_rows(horizon=config.PRIMARY_HORIZON,
                                  lookback_days=args.lookback)
    await db.close_pool()
    days = build_evidence_days(rows)

    counts = Counter(e.source_type for d in days for e in d.evidence)
    empty = sum(1 for d in days if not d.evidence)
    print(f"{len(days)} symbol-days at {config.PRIMARY_HORIZON}d, "
          f"{len({d.day for d in days})} distinct days "
          f"({empty} with no evidence in a {args.lookback}-day window)")
    for s in FITTABLE_SOURCES:
        n = counts.get(s, 0)
        note = ""
        if n == 0:
            note = "  <- no data in the labelled window; its weight does nothing"
        print(f"  {s:<12} {n:>8} chunks{note}")
    print(f"  (not fittable: {', '.join(WORLD_SCOPED_SOURCES)} — world-scoped, "
          f"identical for every name on a day, so they cannot move a "
          f"within-day ranking)")

    train, test = split_days(days, test_fraction=0.4)
    print(f"\ntrain {len({d.day for d in train})} days / "
          f"test {len({d.day for d in test})} days")
    print(f"incumbent  train  {_ic(train, INCUMBENT)}")
    print(f"incumbent  test   {_ic(test, INCUMBENT)}")

    cand = optimise_sources(train)
    print(f"\nsearch: {cand.evaluations} evaluations")
    for m in cand.moves:
        print(f"  {m}")
    if not cand.moves:
        print("  no move improved on the incumbent")
    print(f"candidate  train  {cand.ic}")
    print(f"candidate  test   {_ic(test, cand.weights)}")

    res = evaluate_sources(test, cand.weights, INCUMBENT)
    print(f"\nGATE: {res}")
    for r in res.reasons:
        print(f"  - {r}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
