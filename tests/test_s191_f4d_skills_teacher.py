"""S191 F4d -- evolving skills + teacher.

No code change this lot (record + verify). These tests pin two concrete claims:

- A behavioural check of the registry's traversal safety: a `..` / `/` name is
  slugified and the resolved skill path always stays under the registry root
  (the security-relevant invariant the write side relies on). `agent/skills.py`
  is importlib-isolatable (stdlib-only top-level imports; the backend imports
  are lazy/guarded), so it is loaded with the parent package stubbed.
- Source assertions pinning the wiring gaps recorded as TCH-01 / SKL-01: the
  generic agent loop does not invoke teacher escalation, and
  `publish_teacher_draft` has no live caller (only the package re-export).
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

OO_DIR = Path(__file__).resolve().parent.parent / "opti_oignon"
SKILLS = OO_DIR / "agent" / "skills.py"
LOOP = OO_DIR / "agent" / "loop.py"


def _stub_parents():
    if "opti_oignon" not in sys.modules:
        sys.modules["opti_oignon"] = types.ModuleType("opti_oignon")
    if "opti_oignon.agent" not in sys.modules:
        sys.modules["opti_oignon.agent"] = types.ModuleType("opti_oignon.agent")


def _load_skills():
    _stub_parents()
    spec = importlib.util.spec_from_file_location("opti_oignon.agent.skills", SKILLS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Behavioural: traversal safety
# --------------------------------------------------------------------------

def test_f4d_safe_segment_strips_traversal():
    skills = _load_skills()
    for hostile in ("../../etc/passwd", "/abs/path", "..", "a/b/c", "name\\with\\sep"):
        seg = skills._safe_segment(hostile, "fallback")
        assert "/" not in seg and "\\" not in seg
        assert ".." not in seg
        assert seg  # never empty (falls back)


def test_f4d_registry_write_stays_within_root():
    skills = _load_skills()
    with tempfile.TemporaryDirectory() as tmp:
        reg = skills.SkillRegistry(root=tmp)
        root = Path(tmp).resolve()
        # A hostile category/name must still resolve under the root.
        skill = reg.add(
            name="../../escape",
            category="../../../etc",
            body="When to Use\nx\n\nProcedure\ny\n",
            status=skills.STATUS_DRAFT,
        )
        # The draft file exists and is contained under the registry root.
        path = reg._skill_path(skill.category, skill.name, draft=True)
        assert path is not None
        assert path.resolve().is_relative_to(root)
        assert path.is_file()


def test_f4d_published_and_draft_paths_separated():
    skills = _load_skills()
    with tempfile.TemporaryDirectory() as tmp:
        reg = skills.SkillRegistry(root=tmp)
        reg.add("alpha", "general", "When to Use\nuse it\n", status=skills.STATUS_DRAFT)
        assert reg.get("alpha", "general", draft=True) is not None
        assert reg.get("alpha", "general", draft=False) is None  # draft does not shadow
        published = reg.publish("alpha", "general")
        assert published is not None and published.status == skills.STATUS_PUBLISHED
        assert reg.get("alpha", "general", draft=True) is None  # draft removed on publish


# --------------------------------------------------------------------------
# Source: the wiring gaps (TCH-01, SKL-01)
# --------------------------------------------------------------------------

def test_f4d_loop_does_not_wire_teacher_escalation():
    src = LOOP.read_text(encoding="utf-8")
    assert "teacher" not in src.lower()
    assert "escalate" not in src.lower()


def test_f4d_publish_teacher_draft_has_no_live_caller():
    # Defined in skills.py, re-exported in __init__, but called by no live
    # loop/route module (pins SKL-01). Scan opti_oignon, excluding the
    # definition, the package re-export, and tests.
    callers = []
    for py in OO_DIR.rglob("*.py"):
        rel = py.relative_to(OO_DIR).as_posix()
        if rel in ("agent/skills.py", "agent/__init__.py"):
            continue
        if "test" in rel:
            continue
        text = py.read_text(encoding="utf-8")
        if "publish_teacher_draft(" in text:
            callers.append(rel)
    assert callers == [], f"unexpected live caller(s): {callers}"
