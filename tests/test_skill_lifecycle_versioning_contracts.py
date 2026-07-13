#!/usr/bin/env python3
"""Lifecycle and versioning contracts for the SKILL.md registry.

The registry keeps three separations that make the skill store auditable: a
draft is a proposal that never shadows the published skill; every published
revision archives its predecessor under ``.versions/``; and consuming a skill
touches only the ``_usage.json`` sidecar, never the SKILL.md text. This suite
pins that lifecycle:

  * LC1 -- a draft written over an existing published skill carries the
    version it would become and leaves the published text untouched;
  * LC2 -- publishing a draft promotes its body to the published tree and
    removes the draft file;
  * LC3 -- publishing over an existing published skill archives it and bumps
    the version by one;
  * LC4 -- an edit archives the previous version before writing the new one;
  * LC5 -- a find-and-replace patch requires the search string to occur
    exactly once (absent or duplicated targets change nothing);
  * LC6 -- deleting a published skill archives its final version first and
    keeps the ``.versions`` history;
  * LC7 -- bumping the usage counter never rewrites SKILL.md (byte-identical
    before and after);
  * LC8 -- a draft is a draft because of WHERE it lives: listings force the
    draft status from location, and the category filter compares slugs.

The module is loaded in isolation; the sync-journal and audit hooks are
stubbed. Local-only. Runs under pytest or the __main__ runner.
"""

import hashlib
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = _REPO / "opti_oignon" / "agent"

_KEYS = ("opti_oignon", "opti_oignon.agent", "opti_oignon.agent.skills")


def _load():
    saved = {k: sys.modules.get(k) for k in _KEYS}

    for n in ("opti_oignon", "opti_oignon.agent"):
        pkg = types.ModuleType(n)
        pkg.__path__ = []
        sys.modules[n] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.agent.skills", _AGENT / "skills.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.agent.skills"] = mod
    spec.loader.exec_module(mod)

    mod._sync_publish_skill = lambda *a, **k: None
    mod._audit = lambda *a, **k: None

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


_BODY_A = "## When to Use\nWhen greeting.\n\n## Procedure\nSay hello.\n"
_BODY_B = "## When to Use\nWhen greeting warmly.\n\n## Procedure\nSay welcome.\n"


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_lc1_draft_over_published_carries_next_version_and_never_shadows():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            pub = reg.add("greet", "general", _BODY_A, status=mod.STATUS_PUBLISHED)
            assert pub.version == 1
            draft = reg.add("greet", "general", _BODY_B)  # default: a draft
            assert draft.status == mod.STATUS_DRAFT
            assert draft.version == 2  # the version it would become
            still = reg.get("greet", "general", draft=False)
            assert still is not None
            assert still.version == 1 and still.body == pub.body.strip()
            # A first draft with no published counterpart starts at 1.
            fresh = reg.add("other", "general", _BODY_A)
            assert fresh.version == 1
        finally:
            restore()


def test_lc2_publish_promotes_the_draft_and_removes_it():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            reg.add("greet", "general", _BODY_B)  # draft only
            published = reg.publish("greet", "general")
            assert published is not None
            assert published.status == mod.STATUS_PUBLISHED
            assert published.version == 1
            got = reg.get("greet", "general", draft=False)
            assert got is not None and got.body == published.body
            assert reg.get("greet", "general", draft=True) is None
            draft_path = reg._skill_path("general", "greet", draft=True)
            assert draft_path is not None and not draft_path.exists()
        finally:
            restore()


def test_lc3_publish_over_existing_archives_and_bumps():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            reg.add("greet", "general", _BODY_A, status=mod.STATUS_PUBLISHED)
            reg.add("greet", "general", _BODY_B)  # the draft to promote
            published = reg.publish("greet", "general")
            assert published is not None and published.version == 2
            versions = reg._skill_dir("general", "greet", draft=False) / mod.VERSIONS_DIR
            archived = versions / "v1.md"
            assert archived.is_file()
            assert "Say hello." in archived.read_text(encoding="utf-8")
        finally:
            restore()


def test_lc4_update_archives_the_previous_version():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            reg.add("greet", "general", _BODY_A, status=mod.STATUS_PUBLISHED)
            updated = reg.update("greet", "general", body=_BODY_B)
            assert updated is not None and updated.version == 2
            assert "Say welcome." in updated.body
            versions = reg._skill_dir("general", "greet", draft=False) / mod.VERSIONS_DIR
            archived = versions / "v1.md"
            assert archived.is_file()
            assert "Say hello." in archived.read_text(encoding="utf-8")
            # Editing a skill that does not exist returns None, writes nothing.
            assert reg.update("missing", "general", body=_BODY_B) is None
        finally:
            restore()


def test_lc5_patch_requires_a_unique_target():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            body = "## When to Use\nhello and hello again.\n"
            reg.add("greet", "general", body, status=mod.STATUS_PUBLISHED)
            # Absent target: nothing changes.
            assert reg.patch("greet", "general", "zzz-absent", "x") is None
            # Duplicated target: nothing changes.
            assert reg.patch("greet", "general", "hello", "hi") is None
            unchanged = reg.get("greet", "general")
            assert unchanged.version == 1 and unchanged.body == body.strip()
            # Unique target: replaced once, version bumped.
            patched = reg.patch("greet", "general", "hello again", "hi again")
            assert patched is not None and patched.version == 2
            assert "hi again" in patched.body and "hello and" in patched.body
        finally:
            restore()


def test_lc6_delete_archives_the_final_version_and_keeps_history():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            reg.add("greet", "general", _BODY_A, status=mod.STATUS_PUBLISHED)
            reg.update("greet", "general", body=_BODY_B)  # now v2, v1 archived
            skill_dir = reg._skill_dir("general", "greet", draft=False)
            assert reg.delete("greet", "general") is True
            assert not (skill_dir / mod.SKILL_FILENAME).exists()
            versions = skill_dir / mod.VERSIONS_DIR
            assert (versions / "v2.md").is_file()  # final version archived
            assert versions.is_dir()  # the history directory is retained
            assert reg.delete("greet", "general") is False
        finally:
            restore()


def test_lc7_usage_bump_never_rewrites_the_skill_text():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            reg.add("greet", "general", _BODY_A, status=mod.STATUS_PUBLISHED)
            path = reg._skill_path("general", "greet", draft=False)
            before = _md5(path)
            first = reg.increment_usage("greet", "general")
            second = reg.increment_usage("greet", "general")
            assert first.uses == 1 and second.uses == 2
            assert second.last_used != ""
            assert _md5(path) == before  # SKILL.md byte-identical
            stored = reg.get_usage("greet", "general")
            assert stored.uses == 2
            # A missing skill reads as the default counter and bumps nothing.
            assert reg.get_usage("missing", "general").uses == 0
            assert reg.increment_usage("missing", "general").uses == 0
            # Deleting the skill removes its sidecar along with it.
            sidecar = reg._usage_path("greet", "general")
            assert sidecar is not None and sidecar.is_file()
            assert reg.delete("greet", "general") is True
            assert not sidecar.exists()
        finally:
            restore()


def test_lc8_location_defines_a_draft_and_categories_compare_as_slugs():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            reg.add("greet", "general-cat", _BODY_A, status=mod.STATUS_PUBLISHED)
            reg.add("other", "misc", _BODY_A, status=mod.STATUS_PUBLISHED)
            # A raw draft file whose frontmatter CLAIMS to be published must
            # still be reported as a draft: its location decides.
            liar_dir = Path(td) / mod.DRAFTS_DIR / "general-cat" / "liar"
            liar_dir.mkdir(parents=True)
            (liar_dir / mod.SKILL_FILENAME).write_text(
                "---\nname: liar\ncategory: general-cat\nstatus: published\n"
                "version: 1\n---\n\n## When to Use\nNever trust me.\n",
                encoding="utf-8",
            )
            everything = reg.list(include_drafts=True)
            liar = next(s for s in everything if s.name == "liar")
            assert liar.status == mod.STATUS_DRAFT
            published_only = [s.name for s in reg.list(include_drafts=False)]
            assert "liar" not in published_only
            # The category filter folds messy input to the same slug.
            filtered = reg.list(category="  general-cat!!")
            assert [s.name for s in filtered] == ["greet"]
        finally:
            restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
