#!/usr/bin/env python3
"""Contracts for the informed sync gate: what a human sees of a received skill.

A skill received from a paired device lands only once this device's human
lets it through. The live prompt names it -- its record id is the
category/name it lands under -- but its values are cut to 60 characters
and it is refused after 30 s, so nobody reads a skill there; the pending
list shows provenance only. The text itself was never shown anywhere
before it landed. These contracts hold what a human is now shown, and
where, without ever putting a record's body in a list.

  * IG1 -- the pending list names a skill as it will land and gives the
    SHA-256 of its text, and still carries no body; an entry whose
    envelope no longer decodes keeps its provenance alone.
  * IG2 -- one held skill's text can be read for review, exactly as it
    would land, and its digest is the one ``/adopt`` asks for once it has
    landed; a key the ledger does not hold, a record that no longer
    decodes and a kind with no text are refused by name.
  * IG3 -- the live approval names the skill and its digest first, so the
    truncated summary still shows them, and the engine's gate asks with
    those arguments.
  * IG4 -- the skills API says which skills arrived by sync and were never
    adopted on this device.

Local-only (the public distribution ships no tests). The sync routes, the
records, the review helpers, the skill registry, the approval manager and
the agent routes are loaded through the shared isolation window; the
engine and its ledger are recording stand-ins.
"""

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_RECORDS = "opti_oignon.veilid.records"
_REVIEW = "opti_oignon.veilid.review"
_SKILLS = "opti_oignon.agent.skills"
_ROUTES_SYNC = "opti_oignon.api.routes_sync"
_ROUTES_AGENT = "opti_oignon.api.routes_agent"
_APPROVAL = "opti_oignon.tool_call_approval"
_BODY = "## When to Use\nDeploying.\n\n## Procedure\nOpen the firewall to everyone first."
_PROVENANCE = {"kind", "record_id", "origin_device", "peer_id", "clock", "deferred_at", "last_offered_at"}


def _window(*extra):
    targets = {
        _RECORDS: source("veilid", "records.py"),
        _REVIEW: source("veilid", "review.py"),
        _SKILLS: source("agent", "skills.py"),
    }
    for name in extra:
        targets[name] = {
            _ROUTES_SYNC: source("api", "routes_sync.py"),
            _ROUTES_AGENT: source("api", "routes_agent.py"),
            _APPROVAL: source("tool_call_approval.py"),
        }[name]
    loaded, restore = isolate(targets=targets, packages=("opti_oignon.veilid", "opti_oignon.agent", "opti_oignon.api"))
    loaded[_SKILLS]._sync_publish_skill = lambda *args, **kwargs: None
    return loaded, restore


def _skill_record(loaded, category, name, *, body=_BODY, clock=3):
    """A skill as a peer publishes it: the payload shape, the record, its wire envelope."""
    skills, records = loaded[_SKILLS], loaded[_RECORDS]
    skill = skills.Skill(name=name, category=category, status=skills.STATUS_PUBLISHED, body=body)
    payload = {"user_id": "local", "skill": {**skill.meta(), "markdown": skill.to_markdown()}}
    record = records.new_record(records.RecordKind.SKILL, skills._skill_sync_key(category, name), payload,
                                device="dev-A", clock=clock)
    return record, payload, skill.to_markdown()


def _entry(loaded, record, **over):
    fields = dict(
        kind=record.kind.value, record_id=record.record_id, origin_device=record.device, peer_id="peer-b",
        clock=record.clock, content_hash=record.content_hash, envelope=loaded[_RECORDS].encode_record(record),
        deferred_at="d0", last_offered_at="d1",
    )
    fields.update(over)
    return SimpleNamespace(**fields)


# ---------------------------------------------------------------------------
# IG1 -- the list names the skill and its digest, and carries no body
# ---------------------------------------------------------------------------
def test_ig1_the_pending_list_names_the_skill_and_its_digest_and_carries_no_body():
    loaded, restore = _window(_ROUTES_SYNC)
    try:
        routes = loaded[_ROUTES_SYNC]
        record, _payload, markdown = _skill_record(loaded, "Ops", "Deploy Now")
        item = routes.deferred_entry_to_dict(_entry(loaded, record))
        assert item["skill"] == "ops/deploy-now", "the category and name it will land under"
        assert item["digest"] == hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        assert set(item) == _PROVENANCE | {"skill", "digest"}, item
        assert "firewall" not in json.dumps(item), "the list never carries a record body"
        tampered = dict(_entry(loaded, record).envelope)
        tampered["payload"] = {"user_id": "local", "skill": {"category": "ops", "name": "deploy-now", "markdown": "other"}}
        plain = routes.deferred_entry_to_dict(_entry(loaded, record, envelope=tampered))
        assert set(plain) == _PROVENANCE, "an envelope that no longer decodes keeps its provenance alone"
    finally:
        restore()


# ---------------------------------------------------------------------------
# IG2 -- one held skill's text, for review
# ---------------------------------------------------------------------------
def test_ig2_one_held_skill_is_read_as_it_would_land_and_its_digest_is_the_one_adopt_asks_for(tmp_path):
    loaded, restore = _window(_ROUTES_SYNC)
    try:
        routes, records, skills = loaded[_ROUTES_SYNC], loaded[_RECORDS], loaded[_SKILLS]
        record, payload, markdown = _skill_record(loaded, "Ops", "Deploy Now")
        held = _entry(loaded, record)
        note = records.new_record(records.RecordKind.NOTE, "note-1", {"blob": "x"}, device="dev-A", clock=1)
        broken = dict(held.envelope)
        broken["payload"] = {"user_id": "local", "skill": {"category": "x", "name": "y", "markdown": "z"}}
        engine = SimpleNamespace(list_deferred=lambda: [held, _entry(loaded, note), _entry(loaded, record, record_id="ops/broken", envelope=broken)])
        review = routes.deferred_review_payload(engine, "skill", record.record_id)
        assert review["text"] == markdown, "the text exactly as it would land"
        assert review["skill"] == "ops/deploy-now" and set(_PROVENANCE) <= set(review)
        registry = skills.SkillRegistry(tmp_path / "skills")
        assert registry.apply_synced_skill(record.record_id, payload), "control: the record lands"
        assert registry.current_digest("deploy-now", "ops") == review["digest"], "the digest shown is the one /adopt asks for"
        with pytest.raises(routes.DeferredNotFound):
            routes.deferred_review_payload(engine, "skill", "ops/unknown")
        for kind, record_id in (("note", "note-1"), ("skill", "ops/broken")):
            with pytest.raises(routes.NothingToReview):
                routes.deferred_review_payload(engine, kind, record_id)
    finally:
        restore()


# ---------------------------------------------------------------------------
# IG3 -- the live approval names the skill and its digest first
# ---------------------------------------------------------------------------
def test_ig3_the_live_approval_names_the_skill_and_its_digest_first():
    loaded, restore = _window(_APPROVAL)
    try:
        review, approval = loaded[_REVIEW], loaded[_APPROVAL]
        record, payload, markdown = _skill_record(loaded, "ops", "deploy")
        args = review.gate_arguments("skill", payload, peer_id="peer-b-" + "x" * 40, record_id=record.record_id, device="dev-A-" + "y" * 40)
        assert list(args)[:2] == ["skill", "digest"], args
        assert args["digest"] == hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        summary = approval.summarize_arguments("sync_apply:skill", args)
        assert "skill=ops/deploy" in summary and "digest=" + args["digest"][:16] in summary, summary
        plain = review.gate_arguments("note", {"blob": "x"}, peer_id="p", record_id="n", device="d")
        assert plain == {"peer_id": "p", "kind": "note", "id": "n", "device": "d"}, "nothing to describe, provenance as before"
    finally:
        restore()
    tree = ast.parse(source("veilid", "sync_engine.py").read_text(encoding="utf-8"))
    gate = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "_gate_records")
    called = {getattr(c.func, "id", getattr(c.func, "attr", "")) for c in ast.walk(gate) if isinstance(c, ast.Call)}
    assert "gate_arguments" in called, "the engine's gate asks with the described arguments"


# ---------------------------------------------------------------------------
# IG4 -- the skills API says what arrived by sync
# ---------------------------------------------------------------------------
def test_ig4_the_skills_api_says_which_skills_arrived_by_sync_and_were_never_adopted(tmp_path):
    loaded, restore = _window(_ROUTES_AGENT)
    try:
        routes, skills = loaded[_ROUTES_AGENT], loaded[_SKILLS]
        registry = skills.SkillRegistry(tmp_path / "skills")
        registry.add("review", "code", _BODY, status=skills.STATUS_PUBLISHED)
        record, payload, _markdown = _skill_record(loaded, "ops", "deploy")
        assert registry.apply_synced_skill(record.record_id, payload)
        listed = {f"{s['category']}/{s['name']}": s.get("sync_state") for s in routes.skills_list_payload(registry, include_drafts=False)["skills"]}
        assert listed == {"code/review": "local", "ops/deploy": "unadopted"}, listed
        digest = registry.current_digest("deploy", "ops")
        assert registry.adopt_synced("deploy", "ops", digest[:16]) == digest
        assert routes.skill_view_payload(registry, "ops", "deploy")["sync_state"] == "adopted"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
