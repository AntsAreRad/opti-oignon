#!/usr/bin/env python3
"""Tests for the skill materialization apply method (SYN-01 receive half).

``SkillRegistry.apply_synced_skill`` is the receiving half of a sync round for
the GATED ``SKILL`` kind. By the time a record reaches it the engine's human
gate has already approved adoption (an unapproved skill is deferred to the
ledger and never lands), so this is purely the materialisation of an approved
skill onto the file store. This suite loads ``skills.py`` in isolation (it
imports only stdlib; the publish hook is stubbed) and proves:

  * a record rebuilds the published SKILL.md BYTE-FAITHFULLY (the wire markdown
    is written verbatim) and round-trips through the registry's own reader;
  * an update overwrites the file in place;
  * a tombstone unlinks the published skill and its usage sidecar;
  * the device-local ``_usage.json`` is PRESERVED across an update (only
    SKILL.md is touched);
  * a hostile ``category``/``name`` cannot escape the registry root (path
    confinement via ``_skill_path``);
  * a malformed payload / a key mismatch / a missing markdown fails secure
    (returns False, raises nothing, writes nothing);
  * the apply is HOOK-FREE (no apply -> write -> publish echo), proven by a
    publish-hook spy that stays at zero across an apply.

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load():
    """Load skills.py in isolation; the publish hook is neutralised so setup
    never reaches the (absent) veilid framework."""
    keys = ("opti_oignon", "opti_oignon.agent", "opti_oignon.agent.skills")
    saved = {k: sys.modules.get(k) for k in keys}

    for n in ("opti_oignon", "opti_oignon.agent"):
        pkg = types.ModuleType(n)
        pkg.__path__ = []
        sys.modules[n] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.agent.skills", _OO / "agent" / "skills.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.agent.skills"] = mod
    spec.loader.exec_module(mod)

    # apply is hook-free; _write (used by setup helpers) publishes -- stub it.
    mod._sync_publish_skill = lambda *a, **k: None

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


def _wire(mod, category="general", name="greet", body="## When to Use\nGreet politely.\n", version=1):
    """Build the full-state wire payload exactly as a producer would: the nested
    skill is the frontmatter meta plus ``markdown`` (the exact to_markdown text)."""
    s = mod.Skill(name=name, category=category, body=body, version=version)
    nested = s.meta()
    nested["markdown"] = s.to_markdown()
    key = mod._skill_sync_key(category, name)
    return key, {"user_id": "local", "skill": nested}


def test_apply_creates_skill_byte_faithful():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            key, payload = _wire(mod)
            assert reg.apply_synced_skill(key, payload) is True

            path = reg._skill_path("general", "greet", draft=False)
            assert path is not None and path.is_file()
            # byte-faithful: the file is exactly the wire markdown
            assert path.read_text(encoding="utf-8") == payload["skill"]["markdown"]
            # and it round-trips through the registry's own reader
            got = reg.get("greet", "general")
            assert got is not None
            assert got.name == "greet" and got.category == "general"
            assert "Greet politely." in got.body
        finally:
            restore()


def test_apply_overwrites_existing():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            k1, p1 = _wire(mod, body="## When to Use\nv1 body\n", version=1)
            assert reg.apply_synced_skill(k1, p1) is True
            k2, p2 = _wire(mod, body="## When to Use\nv2 body\n", version=2)
            assert k2 == k1  # same identity
            assert reg.apply_synced_skill(k2, p2) is True

            path = reg._skill_path("general", "greet", draft=False)
            assert path.read_text(encoding="utf-8") == p2["skill"]["markdown"]
            assert "v2 body" in path.read_text(encoding="utf-8")
        finally:
            restore()


def test_apply_tombstone_unlinks():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            key, payload = _wire(mod)
            assert reg.apply_synced_skill(key, payload) is True
            path = reg._skill_path("general", "greet", draft=False)
            usage = path.parent / mod.USAGE_FILENAME
            usage.write_text('{"uses": 3, "last_used": "t0"}', encoding="utf-8")
            assert path.is_file() and usage.is_file()

            assert reg.apply_synced_skill(key, payload, deleted=True) is True
            assert not path.exists()
            assert not usage.exists()
        finally:
            restore()


def test_apply_preserves_usage_on_update():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            key, p1 = _wire(mod, body="## When to Use\nv1\n", version=1)
            assert reg.apply_synced_skill(key, p1) is True
            path = reg._skill_path("general", "greet", draft=False)
            usage = path.parent / mod.USAGE_FILENAME
            usage.write_text('{"uses": 9, "last_used": "t5"}', encoding="utf-8")

            _, p2 = _wire(mod, body="## When to Use\nv2\n", version=2)
            assert reg.apply_synced_skill(key, p2) is True
            # SKILL.md updated, usage sidecar untouched (device-local)
            assert "v2" in path.read_text(encoding="utf-8")
            assert usage.read_text(encoding="utf-8") == '{"uses": 9, "last_used": "t5"}'
        finally:
            restore()


def test_apply_path_traversal_contained():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            # A hostile pair laced with traversal tokens. _skill_sync_key
            # sanitises both segments, so the record key matches what apply
            # recomputes, and _skill_path confines the write to the root.
            key, payload = _wire(mod, category="../../../etc", name="passwd")
            assert reg.apply_synced_skill(key, payload) is True

            written = reg._skill_path("../../../etc", "passwd", draft=False)
            assert written is not None
            # the resolved file is strictly inside the registry root
            written.resolve().relative_to(Path(td).resolve())
            # nothing landed at the real absolute target
            assert not Path("/etc/passwd/SKILL.md").exists()
        finally:
            restore()


def test_apply_malformed_returns_false():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            key, good = _wire(mod)
            # not a dict
            assert reg.apply_synced_skill(key, "nope") is False
            # skill not a dict
            assert reg.apply_synced_skill(key, {"user_id": "local"}) is False
            # missing name
            bad = {"user_id": "local", "skill": {"category": "general", "markdown": "x"}}
            assert reg.apply_synced_skill(key, bad) is False
            # key mismatch (record_id does not match the nested identity)
            assert reg.apply_synced_skill("general/other", good) is False
            # markdown missing on a non-tombstone
            nomd = {"user_id": "local", "skill": {"category": "general", "name": "greet"}}
            assert reg.apply_synced_skill(mod._skill_sync_key("general", "greet"), nomd) is False

            # nothing was written by any malformed apply
            assert reg.get("greet", "general") is None
            assert reg.get("other", "general") is None
        finally:
            restore()


def test_apply_is_hook_free():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            calls = []
            mod._sync_publish_skill = lambda *a, **k: calls.append((a, k))
            key, payload = _wire(mod)
            assert reg.apply_synced_skill(key, payload) is True
            assert reg.apply_synced_skill(key, payload, deleted=True) is True
            assert calls == []  # apply never re-publishes
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
