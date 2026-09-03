"""Read-only access to shrub's database.

Every query here is a SELECT and the connection is expected to hold a
SELECT-only grant (scripts/create_ro_user.sql). Milton's only write is an
approved weight promotion, and that goes through shrub rather than this pool —
an evolver that can write to the live fund's tables is one bug away from
being the thing that breaks it.
"""
from __future__ import annotations

import json

import aiomysql

from . import config

_pool = None


async def get_pool():
    global _pool
    if _pool is None:
        _pool = await aiomysql.create_pool(
            host=config.DB_HOST, port=config.DB_PORT,
            user=config.DB_USER, password=config.DB_PASSWORD,
            db=config.DB_NAME, autocommit=True, minsize=1, maxsize=4,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None


async def fetchall(sql: str, args=()) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, args)
            return list(await cur.fetchall())


# The labelled set: every screened candidate that has a scored outcome. Picks
# are joined to returns rather than the reverse so an unscored pick (horizon
# not yet elapsed) simply doesn't appear.
_SCREENER_PICKS_SQL = """
SELECT p.id, p.symbol, p.created_at, p.source_signals,
       r.horizon_days, r.alpha, r.pct_return, r.spy_pct_return
FROM discovery_picks p
JOIN discovery_pick_returns r ON r.pick_id = p.id
WHERE p.scout_model = 'screener'
  AND p.source_signals IS NOT NULL
  AND r.alpha IS NOT NULL
ORDER BY p.created_at, p.id
"""


async def screener_picks() -> list[dict]:
    rows = await fetchall(_SCREENER_PICKS_SQL)
    for r in rows:
        if isinstance(r.get("source_signals"), str):
            try:
                r["source_signals"] = json.loads(r["source_signals"])
            except ValueError:
                r["source_signals"] = {}
    return rows


# Evidence retrievable for each scored symbol-day: every chunk about that
# ticker published in the lookback window BEFORE the pick. The window bound is
# the point of the query — using anything published after the pick date would
# score the weights against information that did not exist yet, which is the
# easiest way to build a backtest that works beautifully and means nothing.
_EVIDENCE_SQL = """
SELECT p.symbol, DATE(p.created_at) AS pick_date, r.horizon_days, r.alpha,
       c.source_type,
       TIMESTAMPDIFF(HOUR, c.published_at, p.created_at) / 24.0 AS age_days
FROM discovery_picks p
JOIN discovery_pick_returns r ON r.pick_id = p.id
LEFT JOIN research_chunks c
       ON c.ticker = p.symbol
      AND c.published_at <= p.created_at
      AND c.published_at >= DATE_SUB(p.created_at, INTERVAL %s DAY)
WHERE p.scout_model = 'screener'
  AND r.alpha IS NOT NULL
  AND r.horizon_days = %s
ORDER BY p.created_at, p.symbol
"""


async def evidence_rows(horizon: int = 20, lookback_days: int = 90) -> list[dict]:
    return await fetchall(_EVIDENCE_SQL, (int(lookback_days), int(horizon)))
