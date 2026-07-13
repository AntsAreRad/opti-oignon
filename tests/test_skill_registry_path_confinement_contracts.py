#!/usr/bin/env python3
"""Path-confinement contracts for the on-disk SKILL.md registry.

The registry stores each skill at ``<root>/<category>/<name>/SKILL.md`` and
derives both segments from caller input, so its filesystem hygiene is the
boundary that keeps a hostile name or category from writing outside the
registry root. This suite pins that boundary:

  * CF1 -- a traversal payload (``..``, ``/``, ``\\``, absolute paths) never
    survives ``_safe_segment``: only ``[a-z0-9_-]`` remains, the fallback
    covers an empty result, and the ``category/name`` join contains exactly
    one separator;
  * CF2 -- messy but honest input (case, spaces, punctuation) folds to a
    stable slug, lands under the root, and round-trips through the reader;
  * CF3 -- the resolved-path check is a second, independent layer: a skill
    directory that escapes the root yields no path and the write is REFUSED
    with ``ValueError`` (nothing is created outside);
  * CF4 -- the usage sidecar path is confined by the same check: an
    out-of-root resolution reads as a default counter and writes nothing;
  * CF5 -- reserved directory names (versions, drafts, bytecode caches) are
    never surfaced as skills by the scanner.

The module is loaded in isolation (stdlib-only imports); the sync-journal and
audit hooks are stubbed. Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = _REPO / "opti_oignon" / "agent"

_KEYS = ("opti_oignon", "opti_oignon.agent", "opti_oignon.agent.skills")


def _load():
    """Load skills.py in isolation with the journal and audit hooks stubbed."""
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


_BODY = "## When to Use\nWhen greeting.\n\n## Procedure\nSay hello.\n"


def test_cf1_traversal_payload_never_survives_the_slug():
    mod, restore = _load()
    try:
        assert mod._safe_segment("../../etc/passwd", "fallback") == "etc-passwd"
        assert mod._safe_segment("..\\..\\windows", "fallback") == "windows"
        assert mod._safe_segment("/absolute/path", "fallback") == "absolute-path"
        for hostile in ("../../etc/passwd", "/absolute", "a/../b", "..", "."):
            slug = mod._safe_segment(hostile, "fallback")
            assert "/" not in slug and "\\" not in slug and ".." not in slug
        # An input reduced to nothing falls back, so a segment is never empty.
        assert mod._safe_segment("////", "fallback") == "fallback"
        assert mod._safe_segment(None, "fallback") == "fallback"
        # The category/name join therefore contains exactly one separator.
        key = mod._skill_sync_key("../../etc", "pass/wd")
        assert key.count("/") == 1
        assert key == "etc/pass-wd"
    finally:
        restore()


def test_cf2_messy_input_folds_to_a_slug_under_the_root():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            skill = reg.add(
                "My Skill!!", "General Category", _BODY,
                status=mod.STATUS_PUBLISHED,
            )
            assert skill.name == "my-skill"
            assert skill.category == "general-category"
            path = reg._skill_path("General Category", "My Skill!!", draft=False)
            assert path is not None and path.is_file()
            root = Path(td).resolve()
            assert path.resolve().relative_to(root)  # raises if outside
            # The messy original identity round-trips through the reader.
            got = reg.get("My Skill!!", "General Category")
            assert got is not None and got.name == "my-skill"
        finally:
            restore()


def test_cf3_out_of_root_resolution_refuses_the_write():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            outside = Path(td) / "outside-target"
            root = Path(td) / "registry-root"
            root.mkdir()
            reg = mod.SkillRegistry(root=root)
            # Force the first layer to hand back an escaping directory: the
            # resolved-path check must refuse it on its own.
            reg._skill_dir = lambda category, name, *, draft: outside
            assert reg._skill_path("general", "greet", draft=False) is None
            skill = mod.Skill(name="greet", category="general", body=_BODY)
            raised = False
            try:
                reg._write(skill, draft=False)
            except ValueError:
                raised = True
            assert raised, "an out-of-root write must raise, not proceed"
            assert not (outside / mod.SKILL_FILENAME).exists()
            assert not outside.exists() or not any(outside.iterdir())
        finally:
            restore()


def test_cf4_usage_sidecar_is_confined_by_the_same_check():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            outside = Path(td) / "outside-target"
            root = Path(td) / "registry-root"
            root.mkdir()
            reg = mod.SkillRegistry(root=root)
            reg._skill_dir = lambda category, name, *, draft: outside
            assert reg._usage_path("greet", "general") is None
            # Reads degrade to the default counter; writes never land outside.
            usage = reg.get_usage("greet", "general")
            assert usage.uses == 0 and usage.last_used == ""
            bumped = reg.increment_usage("greet", "general")
            assert bumped.uses == 0
            assert not (outside / mod.USAGE_FILENAME).exists()
        finally:
            restore()


def test_cf5_reserved_directory_names_are_never_skills():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            reg.add("greet", "general", _BODY, status=mod.STATUS_PUBLISHED)
            # A SKILL.md planted under a reserved name inside a category must
            # stay invisible to list / index / search.
            planted = Path(td) / "general" / mod.VERSIONS_DIR
            planted.mkdir(parents=True, exist_ok=True)
            (planted / mod.SKILL_FILENAME).write_text(
                "---\nname: intruder\ncategory: general\n---\n\n"
                "## When to Use\nNever.\n",
                encoding="utf-8",
            )
            names = [s.name for s in reg.list(include_drafts=True)]
            assert names == ["greet"]
            idx = reg.index()
            assert [s["name"] for s in idx["published"]] == ["greet"]
            assert all(
                r.skill.name != "intruder" for r in reg.search("never", limit=10)
            )
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
