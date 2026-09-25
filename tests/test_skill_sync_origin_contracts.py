#!/usr/bin/env python3
"""Contracts for the provenance of skills received by sync.

A skill received from a paired device lands in the published tree once the
receiving device's human gate lets the record through, and that gate shows
provenance only -- peer, device, the category/name it lands under -- never
the skill's text. So the registry remembers, on this device alone, that a skill
arrived by sync, and which of its bytes were adopted here after being
shown: only those may later ride a system prompt.

  * SO1 -- a skill applied from sync is marked as received on this device,
    and its bytes are not adopted; a skill written here carries no mark.
  * SO2 -- adoption names the exact bytes: a digest prefix of twelve hex
    characters or more of the bytes on disk now, nothing shorter and
    nothing else; a new version received afterwards is unadopted again.
  * SO3 -- a local write adopts only what it writes whole: a new skill or a
    published draft is adopted, an edit of unadopted bytes stays unadopted.
  * SO4 -- a mark that cannot be read adopts nothing: the bytes are treated
    as received and unadopted, never as written here.

Local-only (the public distribution ships no tests). ``skills.py`` is
loaded through the shared isolation window; the sync publish hook is a
seam, stubbed so a local write journals nothing.
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_SKILLS = "opti_oignon.agent.skills"
_BODY = "## When to Use\nShipping a release.\n\n## Procedure\nTag, build, then announce."


def _open():
    loaded, restore = isolate(targets={_SKILLS: source("agent", "skills.py")}, packages=("opti_oignon.agent",))
    skills = loaded[_SKILLS]
    skills._sync_publish_skill = lambda *args, **kwargs: None
    return skills, restore


def _receive(skills, registry, category, name, body):
    markdown = skills.Skill(name=name, category=category, status=skills.STATUS_PUBLISHED, body=body).to_markdown()
    applied = registry.apply_synced_skill(
        skills._skill_sync_key(category, name),
        {"skill": {"category": category, "name": name, "markdown": markdown}},
    )
    assert applied, "control: the apply sink materialised the record"
    return markdown


def _digest(root, category, name):
    return hashlib.sha256((root / category / name / "SKILL.md").read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# SO1 -- received is marked
# ---------------------------------------------------------------------------
def test_so1_a_skill_received_by_sync_is_marked_and_its_bytes_are_not_adopted(tmp_path):
    skills, restore = _open()
    try:
        root = tmp_path / "skills"
        registry = skills.SkillRegistry(root)
        registry.add("review", "code", _BODY, status=skills.STATUS_PUBLISHED)
        assert registry.sync_state("review", "code") == skills.SYNC_LOCAL, "a skill written here carries no mark"
        markdown = _receive(skills, registry, "ops", "deploy", _BODY)
        assert registry.get("deploy", "ops") is not None, "it is published: the sync gate let the record through"
        assert registry.sync_state("deploy", "ops") == skills.SYNC_UNADOPTED, "its bytes were never shown here"
        assert (root / "ops" / "deploy" / skills.ORIGIN_FILENAME).is_file(), "the mark lives beside the skill"
        assert registry.raw_text("deploy", "ops") == markdown, "the bytes on disk are the bytes received"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SO2 -- adoption names the exact bytes
# ---------------------------------------------------------------------------
def test_so2_adoption_names_the_exact_bytes_and_a_new_version_asks_again(tmp_path):
    skills, restore = _open()
    try:
        root = tmp_path / "skills"
        registry = skills.SkillRegistry(root)
        _receive(skills, registry, "ops", "deploy", _BODY)
        digest = _digest(root, "ops", "deploy")
        assert registry.adopt_synced("deploy", "ops", digest[:11]) is None, "fewer than twelve characters name nothing"
        assert registry.adopt_synced("deploy", "ops", "0" * 16) is None, "the digest of other bytes adopts nothing"
        assert registry.sync_state("deploy", "ops") == skills.SYNC_UNADOPTED
        assert registry.adopt_synced("deploy", "ops", digest[:16].upper()) == digest, "the full digest is recorded"
        assert registry.sync_state("deploy", "ops") == skills.SYNC_ADOPTED
        assert registry.adopt_synced("deploy", "ops", digest[:16]) is None, "nothing is left to adopt"
        _receive(skills, registry, "ops", "deploy", _BODY + "\n\n## Pitfalls\nNever skip the build.")
        assert _digest(root, "ops", "deploy") != digest
        assert registry.sync_state("deploy", "ops") == skills.SYNC_UNADOPTED, "a new version is shown again before it runs"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SO3 -- a local write adopts only what it writes whole
# ---------------------------------------------------------------------------
def test_so3_a_local_write_adopts_only_what_it_writes_whole(tmp_path):
    skills, restore = _open()
    try:
        root = tmp_path / "skills"
        registry = skills.SkillRegistry(root)
        _receive(skills, registry, "ops", "deploy", _BODY)
        registry.update("deploy", "ops", body=_BODY + "\nOne more line.")
        assert registry.sync_state("deploy", "ops") == skills.SYNC_UNADOPTED, "an edit of unadopted bytes stays unadopted"
        registry.patch("deploy", "ops", "One more line.", "Another line.")
        assert registry.sync_state("deploy", "ops") == skills.SYNC_UNADOPTED
        registry.add("deploy", "ops", _BODY, status=skills.STATUS_PUBLISHED)
        assert registry.sync_state("deploy", "ops") == skills.SYNC_ADOPTED, "a whole text written here is adopted"
        registry.update("deploy", "ops", body=_BODY + "\nLocal note.")
        assert registry.sync_state("deploy", "ops") == skills.SYNC_ADOPTED, "an edit of adopted bytes stays adopted"
        _receive(skills, registry, "ops", "ship", _BODY)
        registry.add("ship", "ops", _BODY + "\nDrafted here.")
        assert registry.publish("ship", "ops") is not None
        assert registry.sync_state("ship", "ops") == skills.SYNC_ADOPTED, "a published draft is written whole here"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SO4 -- an unreadable mark adopts nothing
# ---------------------------------------------------------------------------
def test_so4_a_mark_that_cannot_be_read_adopts_nothing(tmp_path):
    skills, restore = _open()
    try:
        root = tmp_path / "skills"
        registry = skills.SkillRegistry(root)
        _receive(skills, registry, "ops", "deploy", _BODY)
        digest = _digest(root, "ops", "deploy")
        assert registry.adopt_synced("deploy", "ops", digest[:16]) == digest
        mark = root / "ops" / "deploy" / skills.ORIGIN_FILENAME
        for broken in ("{not json", "[]", '{"origin": "sync", "adopted": "' + digest + '"}', ""):
            mark.write_text(broken, encoding="utf-8")
            assert registry.sync_state("deploy", "ops") == skills.SYNC_UNADOPTED, f"{broken!r} is read as nothing adopted"
        assert registry.adopt_synced("deploy", "ops", digest[:16]) == digest, "adopting again rewrites a readable mark"
        assert registry.sync_state("deploy", "ops") == skills.SYNC_ADOPTED
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
