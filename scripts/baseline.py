"""Measure the incumbent weights against real shrub data.

Two things are checked, and the first gates the second: milton's re-scoring
must reproduce shrub's stored score EXACTLY for every pick. If the replay is
not faithful, every IC computed on top of it is measuring the wrong function.
"""
import asyncio
import sys

from milton import config, db
from milton.features import build_dataset
from milton.objective import score_weights
from milton.screener import INCUMBENT


async def main() -> int:
    rows = await db.screener_picks()
    picks, problems = build_dataset(rows)
    await db.close_pool()

    print(f"rows fetched      : {len(rows)}")
    print(f"picks decomposed  : {len(picks)}")
    print(f"dropped           : {len(problems)}")
    for p in problems[:5]:
        print(f"    {p}")

    mismatched = [p for p in picks if abs(p.rescore(INCUMBENT) - p.stored_score) > 1e-9]
    print(f"\nreplay fidelity   : {len(picks) - len(mismatched)}/{len(picks)} exact")
    if mismatched:
        print("  REPLAY IS NOT FAITHFUL — IC below would be measuring the wrong function")
        for p in mismatched[:5]:
            print(f"    {p.symbol} {p.pick_date}: stored {p.stored_score} "
                  f"!= replay {p.rescore(INCUMBENT)}")
        return 1

    print("\nincumbent weights, mean daily rank IC vs forward alpha:")
    for h in (1, config.SECONDARY_HORIZON, config.PRIMARY_HORIZON):
        print(f"  {h:>3}d  {score_weights(picks, INCUMBENT, horizon=h)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
