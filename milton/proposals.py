"""Proposal store — milton's own state, in its own SQLite file.

Milton holds SELECT-only credentials on shrub's database by design, so it has
nowhere in there to record what it has proposed. That is the right constraint,
not an obstacle to work around: an evolver that can write to the live fund's
tables is one bug away from being the thing that breaks it.

A proposal is a weight vector, the gate's verdict on it, and a decision. The
invariants exist because this is the one part of milton that acts on the world:

ONE PENDING PROPOSAL AT A TIME. Sending a second while the first is unanswered
would leave two live offers and no way to know which reply meant what. A new
proposal supersedes the old one explicitly.

DECISIONS ARE FINAL AND IDEMPOTENT. A reply can arrive twice — mail gets
retried, people reply twice, a poll re-reads a thread. Deciding an already
decided proposal is a no-op that reports what it already was, never a
resurrection or a second application.

PROPOSALS EXPIRE. An approval reply to a three-week-old proposal is answering a
question about data that has since changed, so past the window it is refused
rather than honoured.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PENDING, APPROVED, REJECTED, EXPIRED, SUPERSEDED = (
    "pending", "approved", "rejected", "expired", "superseded")
OPEN_STATES = (PENDING,)

# How long an unanswered proposal stays answerable.
EXPIRY_DAYS = 7

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    status       TEXT NOT NULL,
    horizon      INTEGER NOT NULL,
    weights      TEXT NOT NULL,
    gate         TEXT NOT NULL,
    sent_at      TEXT,
    decided_at   TEXT,
    decided_by   TEXT,
    reply_text   TEXT,
    note         TEXT
);
CREATE INDEX IF NOT EXISTS proposals_status ON proposals(status);
"""


@dataclass(frozen=True)
class Proposal:
    id: int
    created_at: datetime
    status: str
    horizon: int
    weights: dict
    gate: dict
    sent_at: datetime | None = None
    decided_at: datetime | None = None
    decided_by: str | None = None
    reply_text: str | None = None
    note: str | None = None

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATES

    def expired_at(self, expiry_days: int = EXPIRY_DAYS) -> datetime:
        return self.created_at + timedelta(days=expiry_days)


class ProposalStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # --- writes -----------------------------------------------------------

    def create(self, *, horizon: int, weights: dict, gate: dict,
               note: str | None = None, now: datetime | None = None) -> Proposal:
        """Record a new proposal, superseding any still open."""
        now = now or datetime.now(timezone.utc)
        with self._conn() as c:
            c.execute(
                "UPDATE proposals SET status=?, decided_at=?, "
                "note=COALESCE(note,'') || ' superseded by a newer proposal' "
                "WHERE status IN (?)", (SUPERSEDED, now.isoformat(), PENDING))
            cur = c.execute(
                "INSERT INTO proposals (created_at, status, horizon, weights, "
                "gate, note) VALUES (?,?,?,?,?,?)",
                (now.isoformat(), PENDING, int(horizon), json.dumps(weights),
                 json.dumps(gate), note))
            return self.get(int(cur.lastrowid))

    def mark_sent(self, proposal_id: int, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        with self._conn() as c:
            c.execute("UPDATE proposals SET sent_at=? WHERE id=?",
                      (now.isoformat(), proposal_id))

    def decide(self, proposal_id: int, decision: str, *, by: str | None = None,
               reply_text: str | None = None,
               now: datetime | None = None,
               expiry_days: int = EXPIRY_DAYS) -> tuple[Proposal, str]:
        """Apply a decision. Returns (proposal, outcome).

        `outcome` is one of "applied", "already_decided", "expired" or
        "not_found" — the caller needs to tell a fresh approval from a repeat,
        because only the first one may ever act."""
        if decision not in (APPROVED, REJECTED):
            raise ValueError(f"decision must be {APPROVED} or {REJECTED}")
        now = now or datetime.now(timezone.utc)
        existing = self.get(proposal_id)
        if existing is None:
            return None, "not_found"
        if not existing.is_open:
            return existing, "already_decided"
        if now > existing.expired_at(expiry_days):
            with self._conn() as c:
                c.execute(
                    "UPDATE proposals SET status=?, decided_at=?, note=? "
                    "WHERE id=?",
                    (EXPIRED, now.isoformat(),
                     f"reply arrived after the {expiry_days}-day window",
                     proposal_id))
            return self.get(proposal_id), "expired"
        with self._conn() as c:
            c.execute(
                "UPDATE proposals SET status=?, decided_at=?, decided_by=?, "
                "reply_text=? WHERE id=?",
                (decision, now.isoformat(), by, reply_text, proposal_id))
        return self.get(proposal_id), "applied"

    def expire_stale(self, *, now: datetime | None = None,
                     expiry_days: int = EXPIRY_DAYS) -> int:
        now = now or datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=expiry_days)).isoformat()
        with self._conn() as c:
            cur = c.execute(
                "UPDATE proposals SET status=?, decided_at=?, "
                "note='expired unanswered' WHERE status=? AND created_at < ?",
                (EXPIRED, now.isoformat(), PENDING, cutoff))
            return cur.rowcount or 0

    # --- reads ------------------------------------------------------------

    def get(self, proposal_id: int) -> Proposal | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM proposals WHERE id=?",
                            (proposal_id,)).fetchone()
        return _row_to_proposal(row) if row else None

    def open_proposal(self) -> Proposal | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM proposals WHERE status=? "
                "ORDER BY id DESC LIMIT 1", (PENDING,)).fetchone()
        return _row_to_proposal(row) if row else None

    def recent(self, limit: int = 20) -> list[Proposal]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM proposals ORDER BY id DESC LIMIT ?",
                             (int(limit),)).fetchall()
        return [_row_to_proposal(r) for r in rows]


def _row_to_proposal(row: sqlite3.Row) -> Proposal:
    def _dt(v):
        return datetime.fromisoformat(v) if v else None
    return Proposal(
        id=int(row["id"]), created_at=_dt(row["created_at"]),
        status=row["status"], horizon=int(row["horizon"]),
        weights=json.loads(row["weights"]), gate=json.loads(row["gate"]),
        sent_at=_dt(row["sent_at"]), decided_at=_dt(row["decided_at"]),
        decided_by=row["decided_by"], reply_text=row["reply_text"],
        note=row["note"],
    )
