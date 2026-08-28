"""Reading a decision out of an emailed reply, and rendering the ask.

Parsing free text into an irreversible action deserves suspicion, so the rule
here is narrow on purpose: an explicit approval word must appear, and anything
ambiguous is treated as no decision at all. Silence, a question, "maybe", or a
reply containing both words all leave the proposal open. The cost of failing to
read an approval is that you send it again; the cost of misreading one is that
weights ship because a sentence happened to contain the word "yes".
"""
from __future__ import annotations

import re

from .proposals import APPROVED, REJECTED

# Anchored at the start of a line so a word buried mid-sentence doesn't count —
# "I'm not sure I'd approve that" must not read as an approval.
_APPROVE = re.compile(
    r"^\s*(approve[d]?|approved|yes|ok|okay|lgtm|ship it|ship|go|promote)\b",
    re.IGNORECASE)
_REJECT = re.compile(
    r"^\s*(reject|rejected|no|nope|deny|denied|hold|stop|don'?t|do not)\b",
    re.IGNORECASE)

SUBJECT_RE = re.compile(r"\[\s*milton\s*#\s*(\d+)\s*\]", re.IGNORECASE)


def proposal_id_from_subject(subject: str) -> int | None:
    """Pull the proposal number out of a `[milton #12]` subject tag."""
    m = SUBJECT_RE.search(subject or "")
    return int(m.group(1)) if m else None


def subject_tag(proposal_id: int, tag: str = "milton") -> str:
    return f"[{tag} #{proposal_id}]"


def parse_decision(body: str) -> tuple[str | None, str]:
    """Read a decision from a reply body. Returns (decision, why).

    `decision` is None whenever the text does not clearly say one thing, which
    includes saying both."""
    text = (body or "").strip()
    if not text:
        return None, "empty reply"

    approve = reject = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue          # quoted history is not the reply
        if _APPROVE.match(line):
            approve = True
        elif _REJECT.match(line):
            reject = True

    if approve and reject:
        return None, "reply contains both an approval and a rejection"
    if approve:
        return APPROVED, "approved"
    if reject:
        return REJECTED, "rejected"
    return None, ("no decision word at the start of any line — reply with "
                  "'approve' or 'reject'")
