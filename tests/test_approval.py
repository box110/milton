"""Approval parsing and the proposal lifecycle.

These guard the one part of milton that acts on the world, so the bias
throughout is toward doing nothing: ambiguous text decides nothing, a repeat
reply changes nothing, a late reply applies nothing.
"""
from datetime import datetime, timedelta, timezone

import pytest

from milton.approval import parse_decision, proposal_id_from_subject, subject_tag
from milton.proposals import (
    APPROVED, EXPIRED, PENDING, REJECTED, SUPERSEDED, ProposalStore,
)


# --- parsing --------------------------------------------------------------

@pytest.mark.parametrize("body", ["approve", "APPROVE", "yes", "ok", "lgtm",
                                  "ship it", "approve\n\nlooks right to me"])
def test_clear_approvals(body):
    assert parse_decision(body)[0] == APPROVED


@pytest.mark.parametrize("body", ["reject", "no", "nope", "hold",
                                  "don't", "stop\nnot yet"])
def test_clear_rejections(body):
    assert parse_decision(body)[0] == REJECTED


@pytest.mark.parametrize("body", [
    "", "   ",
    "maybe later",
    "I'm not sure I'd approve that",          # word buried mid-sentence
    "what does the paired t mean?",
    "approve\nreject",                         # says both
])
def test_ambiguous_text_decides_nothing(body):
    decision, why = parse_decision(body)
    assert decision is None
    assert why


def test_quoted_history_is_not_the_reply():
    """The original mail says 'Reply with APPROVE or REJECT'. Quoted back by
    the mail client, that must not read as an approval."""
    quoted = "> Reply with APPROVE or REJECT on the first line."
    assert parse_decision(quoted)[0] is None
    assert parse_decision(f"reject\n\n{quoted}")[0] == REJECTED


def test_a_deferral_is_not_a_rejection():
    """'not yet' declines to answer rather than answering no. Both leave the
    proposal open, but only an explicit rejection records one."""
    assert parse_decision("not yet")[0] is None
    assert parse_decision("let me think about it")[0] is None


def test_subject_tag_roundtrip():
    assert proposal_id_from_subject(f"Re: {subject_tag(42)} weights") == 42
    assert proposal_id_from_subject("no tag here") is None


# --- lifecycle ------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    return ProposalStore(tmp_path / "s.sqlite3")


def _make(store, **kw):
    return store.create(horizon=20, weights={"rsi_oversold": 45},
                        gate={"verdict": "promote"}, **kw)


def test_create_and_read_back(store):
    p = _make(store)
    assert p.status == PENDING
    assert store.open_proposal().id == p.id
    assert store.get(p.id).weights == {"rsi_oversold": 45}


def test_a_new_proposal_supersedes_the_open_one(store):
    """Two live offers and one reply is an ambiguity that must not exist."""
    first = _make(store)
    second = _make(store)
    assert store.get(first.id).status == SUPERSEDED
    assert store.open_proposal().id == second.id


def test_decision_is_applied_once_and_then_is_a_no_op(store):
    p = _make(store)
    decided, outcome = store.decide(p.id, APPROVED, by="op@example.com",
                                    reply_text="approve")
    assert outcome == "applied" and decided.status == APPROVED

    again, outcome2 = store.decide(p.id, REJECTED, by="op@example.com")
    assert outcome2 == "already_decided"
    assert again.status == APPROVED           # not flipped by the second reply


def test_a_late_reply_expires_rather_than_applying(store):
    old = datetime.now(timezone.utc) - timedelta(days=30)
    p = _make(store, now=old)
    decided, outcome = store.decide(p.id, APPROVED)
    assert outcome == "expired"
    assert decided.status == EXPIRED


def test_unanswered_proposals_expire_on_sweep(store):
    old = datetime.now(timezone.utc) - timedelta(days=30)
    p = _make(store, now=old)
    assert store.expire_stale() == 1
    assert store.get(p.id).status == EXPIRED
    assert store.open_proposal() is None


def test_deciding_an_unknown_proposal_is_not_found(store):
    proposal, outcome = store.decide(999, APPROVED)
    assert proposal is None and outcome == "not_found"


def test_only_real_decisions_are_accepted(store):
    p = _make(store)
    with pytest.raises(ValueError):
        store.decide(p.id, "maybe")


def test_mark_sent_records_the_time(store):
    p = _make(store)
    store.mark_sent(p.id)
    assert store.get(p.id).sent_at is not None


def test_render_includes_the_evidence_and_the_tag():
    from milton.notify import render
    store_gate = {
        "verdict": "promote", "reasons": [], "horizon": 20,
        "train_incumbent_ic": 0.08, "train_candidate_ic": 0.37,
        "test_incumbent_ic": 0.23, "test_candidate_ic": 0.47,
        "paired_mean": 0.236, "paired_t": 3.55, "train_days": 11,
        "test_days": 14, "selection_incumbent": 134,
        "selection_candidate": 110,
        "moves": [{"name": "volume_high", "from": 20, "to": 0}],
    }
    from milton.proposals import Proposal
    p = Proposal(id=7, created_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
                 status=PENDING, horizon=20, weights={}, gate=store_gate)
    subject, body = render(p)
    assert "[milton #7]" in subject
    assert "volume_high" in body
    assert "APPROVE or REJECT" in body
    # Out-of-sample must be present and labelled as the one that counts.
    assert "out of sample" in body
    assert "should not be trusted" in body
