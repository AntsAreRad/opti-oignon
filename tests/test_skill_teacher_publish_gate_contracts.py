#!/usr/bin/env python3
"""Publication contracts for the teacher-draft path (a pinned chokepoint).

A teacher-produced SKILL.md draft is guidance, not authority: it reaches the
published tree only through ``publish_teacher_draft``, which asks the human
gate FIRST, sandbox-tests any executable verification steps, and never raises
into the loop. The path has no production caller yet; these contracts pin the
chokepoint so that wiring it later inherits a proven posture:

  * TP1 -- without an explicit approval (denial or a raising callback) the
    draft is not published and its verification steps never run: the gate
    precedes the sandbox;
  * TP2 -- an approved draft whose body carries verification steps is still
    refused when no usable sandbox exists (nothing runs on the host);
  * TP3 -- an approved, step-free draft is published directly to the
    published tree with the teacher provenance;
  * TP4 -- any internal error becomes a result, never an exception.

Loads the gate, untrusted-context, teacher, and skills modules in isolation;
the sync-journal and audit hooks are stubbed. Local-only. Runs under pytest
or the __main__ runner.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = _REPO / "opti_oignon" / "agent"

_MODULES = ("allowlists", "untrusted_context", "teacher", "skills")
_KEYS = ("opti_oignon", "opti_oignon.agent") + tuple(
    f"opti_oignon.agent.{m}" for m in _MODULES
)


def _load():
    saved = {k: sys.modules.get(k) for k in _KEYS}

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    agent = types.ModuleType("opti_oignon.agent")
    agent.__path__ = []
    sys.modules["opti_oignon"] = root
    sys.modules["opti_oignon.agent"] = agent

    loaded = {}
    for m in _MODULES:
        full = f"opti_oignon.agent.{m}"
        spec = importlib.util.spec_from_file_location(full, _AGENT / f"{m}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        setattr(agent, m, mod)
        spec.loader.exec_module(mod)
        loaded[m] = mod

    loaded["skills"]._sync_publish_skill = lambda *a, **k: None
    loaded["skills"]._audit = lambda *a, **k: None

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return loaded["skills"], loaded["teacher"], restore


class _SandboxManager:
    def __init__(self, ok):
        self.bwrap_available = ok


class _Sandbox:
    def __init__(self, ok=True, active=True):
        self.sandbox_manager = _SandboxManager(ok)
        self.active = active
        self.commands = []

    def bash(self, cmd):
        self.commands.append(cmd)
        return "ran: " + cmd


_CONTENT_PLAIN = "## When to Use\nWhen deploying.\n\n## Procedure\nCheck twice.\n"
_CONTENT_CMDS = (
    "## When to Use\nWhen deploying.\n\n## Procedure\nCheck twice.\n\n"
    "## Verification\n```\necho verify\n```\n"
)


def test_tp1_no_approval_means_no_publication_and_no_execution():
    with tempfile.TemporaryDirectory() as td:
        mod, teacher, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            sandbox = _Sandbox(ok=True)
            draft = teacher.TeacherSkillDraft(
                name="deploy-check", category="ops", content=_CONTENT_CMDS,
            )
            res = mod.publish_teacher_draft(
                draft, registry=reg,
                approval_fn=lambda cid, label, args: False,
                sandbox=sandbox,
            )
            assert res.published is False and res.reason == "not_approved"
            assert sandbox.commands == []  # the gate came before the sandbox
            assert reg.get("deploy-check", "ops", draft=False) is None
            # A raising approval callback is the same denial, fail-secure.
            def _boom(cid, label, args):
                raise RuntimeError("gate exploded")

            res2 = mod.publish_teacher_draft(
                draft, registry=reg, approval_fn=_boom, sandbox=sandbox,
            )
            assert res2.published is False and res2.reason == "not_approved"
            assert sandbox.commands == []
            assert reg.index()["published"] == []
        finally:
            restore()


def test_tp2_approved_steps_without_a_sandbox_are_refused():
    with tempfile.TemporaryDirectory() as td:
        mod, teacher, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            draft = teacher.TeacherSkillDraft(
                name="deploy-check", category="ops", content=_CONTENT_CMDS,
            )
            res = mod.publish_teacher_draft(
                draft, registry=reg,
                approval_fn=lambda cid, label, args: True,
                sandbox=None,
            )
            assert res.published is False
            assert res.reason == "verification_failed"
            assert "sandbox is unavailable" in res.detail
            assert reg.get("deploy-check", "ops", draft=False) is None
        finally:
            restore()


def test_tp3_approved_clean_draft_publishes_with_teacher_provenance():
    with tempfile.TemporaryDirectory() as td:
        mod, teacher, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            draft = teacher.TeacherSkillDraft(
                name="deploy-check", category="ops", content=_CONTENT_PLAIN,
            )
            res = mod.publish_teacher_draft(
                draft, registry=reg,
                approval_fn=lambda cid, label, args: True,
                sandbox=None,  # no steps, so no sandbox is needed
            )
            assert res.published is True and res.reason == "published"
            got = reg.get("deploy-check", "ops", draft=False)
            assert got is not None
            assert got.status == mod.STATUS_PUBLISHED
            assert got.source == mod.SOURCE_TEACHER
            assert got.version == 1
            assert reg.get("deploy-check", "ops", draft=True) is None
        finally:
            restore()


def test_tp4_internal_errors_become_a_result_not_an_exception():
    mod, teacher, restore = _load()
    try:
        class _BoomRegistry:
            def add(self, *a, **k):
                raise RuntimeError("registry exploded")

        draft = teacher.TeacherSkillDraft(
            name="deploy-check", category="ops", content=_CONTENT_PLAIN,
        )
        res = mod.publish_teacher_draft(
            draft, registry=_BoomRegistry(),
            approval_fn=lambda cid, label, args: True,
            sandbox=None,
        )
        assert res.published is False and res.reason == "error"
        assert "exploded" in res.detail
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
