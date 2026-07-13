#!/usr/bin/env python3
"""Write-gate contracts for the ``manage_skills`` tool handler.

Every write action of the skills tool (add / edit / patch / publish / delete)
must pass an explicit human gate, and any skill body that carries executable
verification steps runs them ONLY inside the disposable sandbox. This suite
pins the gate and the sandbox seam exactly as they behave today:

  * WG1 -- a write without a positive approval is refused and writes nothing;
  * WG2 -- an approval callback that raises counts as a denial (fail-secure);
  * WG3 -- with no approval infrastructure importable at all, writes are
    denied (the absence of a gate never opens the gate);
  * WG4 -- read actions (list / index / view / view_ref / search) never
    consult the gate;
  * WG5 -- on ``add``, a body with verification steps and no usable sandbox
    is refused BEFORE the gate is consulted (the sandbox check precedes the
    human question; nothing runs on the host);
  * WG6 -- with a usable sandbox, the fenced verification commands run in it,
    in order, and only there;
  * WG7 -- an agent ``add`` is always a draft: the published tree is never
    written directly and an existing published skill is never shadowed;
  * WG8 -- the handler never raises: an internal error becomes an
    observation string;
  * WG9 -- on ``publish``, the gate is consulted BEFORE the verification
    steps run (the opposite order from ``add``, pinned as-is).

The module is loaded in isolation; the sync-journal and audit hooks are
stubbed. Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = _REPO / "opti_oignon" / "agent"

# The gate module is snapshotted and BLOCKED (None entry) too: the
# missing-infrastructure clause needs its lateral import to fail inside the
# isolation window. A bare eviction is not enough -- a residual real module in
# a shared process, or an editable-install meta-path finder that resolves the
# name without consulting the stand-in package path, would both let the import
# succeed. A None entry in sys.modules raises ImportError before any finder
# runs, deterministically.
_KEYS = (
    "opti_oignon", "opti_oignon.agent", "opti_oignon.agent.skills",
    "opti_oignon.agent.allowlists",
)


def _load():
    saved = {k: sys.modules.get(k) for k in _KEYS}

    for n in ("opti_oignon", "opti_oignon.agent"):
        pkg = types.ModuleType(n)
        pkg.__path__ = []
        sys.modules[n] = pkg

    sys.modules["opti_oignon.agent.allowlists"] = None  # block the lateral import

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


class _Gate:
    """An approval spy: records every consultation, answers as configured."""

    def __init__(self, allow=True, boom=False):
        self.allow = allow
        self.boom = boom
        self.calls = []

    def __call__(self, conversation_id, label, args):
        self.calls.append((label, dict(args)))
        if self.boom:
            raise RuntimeError("gate exploded")
        return self.allow


class _SandboxManager:
    def __init__(self, ok):
        self.bwrap_available = ok


class _Sandbox:
    """A deterministic stand-in for the disposable sandbox session."""

    def __init__(self, ok=True, active=True):
        self.sandbox_manager = _SandboxManager(ok)
        self.active = active
        self.commands = []

    def bash(self, cmd):
        self.commands.append(cmd)
        return "ran: " + cmd


_BODY_PLAIN = "## When to Use\nWhen greeting.\n\n## Procedure\nSay hello.\n"
_BODY_CMDS = (
    "## When to Use\nWhen checking.\n\n## Procedure\nRun checks.\n\n"
    "## Verification\n```\necho one\n```\n\n```\necho two\n```\n"
)


def _empty(reg):
    idx = reg.index()
    return not idx["published"] and not idx["drafts"]


def test_wg1_write_without_approval_is_refused():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            gate = _Gate(allow=False)
            handler = mod.make_manage_skills_handler(registry=reg, approval_fn=gate)
            out = handler({"action": "add", "name": "greet", "body": _BODY_PLAIN})
            assert "not approved" in out
            assert _empty(reg)
            assert gate.calls and gate.calls[0][0] == "manage_skills:add"
        finally:
            restore()


def test_wg2_approval_callback_raising_counts_as_denial():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            gate = _Gate(boom=True)
            handler = mod.make_manage_skills_handler(registry=reg, approval_fn=gate)
            out = handler({"action": "add", "name": "greet", "body": _BODY_PLAIN})
            assert "not approved" in out
            assert _empty(reg)
        finally:
            restore()


def test_wg3_missing_gate_infrastructure_denies_writes():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            # No approval_fn injected; under isolation the default gate module
            # cannot be imported. The absence of a gate must read as denial.
            handler = mod.make_manage_skills_handler(registry=reg)
            out = handler({"action": "add", "name": "greet", "body": _BODY_PLAIN})
            assert "not approved" in out
            assert _empty(reg)
            out = handler({"action": "delete", "name": "greet"})
            assert "not approved" in out
        finally:
            restore()


def test_wg4_read_actions_never_consult_the_gate():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            reg.add("greet", "general", _BODY_PLAIN, status=mod.STATUS_PUBLISHED)
            gate = _Gate(allow=False)  # would deny anything it were asked
            handler = mod.make_manage_skills_handler(registry=reg, approval_fn=gate)
            assert handler({"action": "list"}).startswith("Published skills")
            assert handler({"action": "index"}).startswith("Published skills")
            assert "greet" in handler({"action": "view", "name": "greet"})
            assert "greet" in handler({"action": "view_ref", "name": "greet"})
            assert handler({"action": "search", "query": "zzz-no-match"}) == (
                "No matching skills."
            )
            assert gate.calls == []
        finally:
            restore()


def test_wg5_sandbox_refusal_precedes_the_gate_on_add():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            gate = _Gate(allow=True)
            handler = mod.make_manage_skills_handler(
                registry=reg, approval_fn=gate, sandbox=None,
            )
            out = handler({"action": "add", "name": "check", "body": _BODY_CMDS})
            assert "not drafted" in out and "sandbox is unavailable" in out
            assert gate.calls == []  # refused before the human was asked
            assert _empty(reg)
            # An unavailable manager is the same refusal as a missing session.
            handler2 = mod.make_manage_skills_handler(
                registry=reg, approval_fn=gate, sandbox=_Sandbox(ok=False),
            )
            out2 = handler2({"action": "add", "name": "check", "body": _BODY_CMDS})
            assert "sandbox is unavailable" in out2
            assert gate.calls == [] and _empty(reg)
        finally:
            restore()


def test_wg6_verification_commands_run_only_in_the_sandbox():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            sandbox = _Sandbox(ok=True)
            handler = mod.make_manage_skills_handler(
                registry=reg, approval_fn=_Gate(allow=True), sandbox=sandbox,
            )
            out = handler({"action": "add", "name": "check", "body": _BODY_CMDS})
            assert "Draft skill 'check'" in out
            assert "sandbox-tested" in out
            assert sandbox.commands == ["echo one", "echo two"]
        finally:
            restore()


def test_wg7_agent_add_is_always_a_draft_and_never_shadows():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            seeded = reg.add(
                "greet", "general", _BODY_PLAIN, status=mod.STATUS_PUBLISHED,
            )
            handler = mod.make_manage_skills_handler(
                registry=reg, approval_fn=_Gate(allow=True),
            )
            out = handler({
                "action": "add", "name": "greet",
                "body": "## When to Use\nReplacement attempt.\n",
            })
            assert "Draft skill" in out and "awaiting approval" in out
            published = reg.get("greet", "general", draft=False)
            assert published is not None
            assert published.body == seeded.body.strip()  # never shadowed
            assert published.version == seeded.version
            draft = reg.get("greet", "general", draft=True)
            assert draft is not None and draft.status == mod.STATUS_DRAFT
        finally:
            restore()


def test_wg8_handler_never_raises():
    mod, restore = _load()
    try:
        class _BoomRegistry:
            def index(self):
                raise RuntimeError("registry exploded")

        handler = mod.make_manage_skills_handler(
            registry=_BoomRegistry(), approval_fn=_Gate(allow=True),
        )
        out = handler({"action": "list"})
        assert out.startswith("manage_skills failed:")
    finally:
        restore()


def test_wg9_publish_consults_the_gate_before_verification():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            reg = mod.SkillRegistry(root=td)
            reg.add("check", "general", _BODY_CMDS)  # draft with steps
            timeline = []

            def _allowing_gate(cid, label, args):
                timeline.append(("gate", label))
                return True

            class _TimelineSandbox(_Sandbox):
                def bash(self, cmd):
                    timeline.append(("bash", cmd))
                    return super().bash(cmd)

            handler = mod.make_manage_skills_handler(
                registry=reg, approval_fn=_allowing_gate,
                sandbox=_TimelineSandbox(ok=True),
            )
            out = handler({"action": "publish", "name": "check"})
            assert "published" in out
            # The human question came before anything executed.
            assert timeline and timeline[0] == ("gate", "manage_skills:publish")
            # Once approved, steps still refuse without a usable sandbox.
            reg.add("check2", "general", _BODY_CMDS)
            handler2 = mod.make_manage_skills_handler(
                registry=reg, approval_fn=_Gate(allow=True), sandbox=None,
            )
            out2 = handler2({"action": "publish", "name": "check2"})
            assert "not published" in out2 and "sandbox is unavailable" in out2
            assert reg.get("check2", "general", draft=False) is None
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
