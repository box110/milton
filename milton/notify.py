"""Render a proposal as an email and hand it to shrub to send.

Milton has no mail credentials and should not acquire any. shrub already owns a
working mailbox, an SMTP config and an IMAP poller, so milton renders the
message and posts it to shrub's internal API to go out from there. The reply
comes back the same way, intercepted by subject tag before shrub's inbound
router hands operator mail to the CEO agent.
"""
from __future__ import annotations

import httpx

from . import config
from .approval import subject_tag
from .proposals import Proposal


def _fmt(v, d=4):
    if v is None:
        return "—"
    try:
        return f"{v:+.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def render(proposal: Proposal) -> tuple[str, str]:
    """(subject, plain-text body). Plain text only — this is a message asking
    for a decision, and the numbers that matter should survive any client."""
    g = proposal.gate
    tag = subject_tag(proposal.id, config.SUBJECT_TAG)
    subject = (f"{tag} screener weights: candidate beats incumbent by "
               f"{_fmt(g.get('paired_mean'))} IC")

    moves = "\n".join(
        f"  {m['name']:<22} {m['from']:>6g} -> {m['to']:<6g}"
        for m in g.get("moves", [])) or "  (no moves)"

    body = f"""Milton proposes new screener weights.

Reply with APPROVE or REJECT on the first line. Anything else leaves this
open — the reply is read strictly, so an ambiguous answer changes nothing.

PROPOSED MOVES
{moves}

EVIDENCE ({g.get('horizon', proposal.horizon)}-day horizon)
  in sample      incumbent {_fmt(g.get('train_incumbent_ic'))}   candidate {_fmt(g.get('train_candidate_ic'))}
  out of sample  incumbent {_fmt(g.get('test_incumbent_ic'))}   candidate {_fmt(g.get('test_candidate_ic'))}
  paired         {_fmt(g.get('paired_mean'))} per day, t = {_fmt(g.get('paired_t'), 2)}
  days           {g.get('train_days')} train / {g.get('test_days')} test
  watchlist      {g.get('selection_incumbent')} -> {g.get('selection_candidate')} names clearing MIN_SCORE

The in-sample figure is what the search fitted and should not be trusted on its
own; the out-of-sample column is the one that means anything.

GATE: {str(g.get('verdict', '?')).upper()}
{chr(10).join('  - ' + r for r in g.get('reasons', [])) or '  all checks passed'}

Proposal #{proposal.id}, raised {proposal.created_at:%Y-%m-%d %H:%M UTC}.
Expires 7 days from then; a reply after that is refused rather than applied.
Approving records the decision. It does not yet write the weights into shrub —
that stage is not built.
"""
    return subject, body


async def send(proposal: Proposal, *, timeout: float = 15.0) -> tuple[bool, str]:
    """Post the rendered mail to shrub for delivery. Never raises."""
    if not config.APPROVER:
        return False, "MILTON_APPROVER is not set — nobody to send to"
    subject, body = render(proposal)
    headers = ({"X-Internal-Token": config.INTERNAL_TOKEN}
               if config.INTERNAL_TOKEN else {})
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{config.SHRUB_URL.rstrip('/')}/api/internal/milton/send",
                json={"to": config.APPROVER, "subject": subject, "body": body},
                headers=headers)
            r.raise_for_status()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, f"sent to {config.APPROVER}"
