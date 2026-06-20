#!/usr/bin/env python3
"""Tests for S177 -- the approval-gated manage_skills tool and the publish path.

Covers ODYSSEUS_SPEC.md Section 6.2 (the gated tool) and the teacher-draft
publish path:

- The manage_skills schema joins the tools registry (the S176 six become seven;
  manage_skills is the third non-sandbox tool, Daily-only). These re-assert the
  reality the S176 count assertions described before the tool existed; those
  three S176 assertions are deselected in pyproject and re-asserted here.
- The handler: read actions run freely; every write (add / edit / patch /
  publish / delete) passes the fail-secure human gate; a body with verification
  steps is sandbox-tested through the S73/S74 seam and refused when bwrap is
  unavailable; the handler never raises and surfaces the published-plus-drafts
  index.
- publish_teacher_draft consumes teacher.TeacherSkillDraft through
  teacher.request_skill_approval, sandbox-tests the draft's verification where
  present, then publishes; it never raises.

Loaded in isolation via spec_from_file_location, rooted at a temporary
directory, so the runtime collects without the backend.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"


def _ensure_pkg():
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(OO)]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.agent" not in sys.modules:
        apkg = types.ModuleType("opti_oignon.agent")
        apkg.__path__ = [str(AGENT)]
        sys.modules["opti_oignon.agent"] = apkg


def _load(name: str):
    full = f"opti_oignon.agent.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(AGENT / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg()
_load("tool_parsing")
al = _load("allowlists")
_load("dispatch")
_load("untrusted_context")
sk = _load("skills")
teacher = _load("teacher")
t = _load("tools")


@pytest.fixture(autouse=True)
def _reset():
    sk.reset_skill_registry()
    t.reset_tool_registry()
    yield
    sk.reset_skill_registry()
    t.reset_tool_registry()


@pytest.fixture
def reg(tmp_path):
    return sk.SkillRegistry(root=tmp_path)


# Fakes


class _Manager:
    def __init__(self, bwrap=True):
        self.bwrap_available = bwrap


class _FakeSandbox:
    def __init__(self, bwrap=True, active=True):
        self.sandbox_manager = _Manager(bwrap)
        self.active = active
        self.calls = []

    def bash(self, command, timeout=30):
        self.calls.append(command)
        return f"[sandbox] {command}"


class _Approver:
    """A recording approval_fn whose verdict is fixed."""

    def __init__(self, approve: bool):
        self._approve = approve
        self.calls = []

    def __call__(self, conversation_id, tool_name, arguments):
        self.calls.append((conversation_id, tool_name, dict(arguments)))
        return self._approve


def _body(proc="run the deploy script"):
    return (
        "## When to Use\nWhen deploying.\n\n"
        f"## Procedure\n{proc}\n\n"
        "## Pitfalls\nDo not skip the health check.\n\n"
        "## Verification\nConfirm the service responds.\n"
    )


def _body_with_cmd():
    return (
        "## When to Use\nWhen deploying.\n\n"
        "## Procedure\nrun the deploy script\n\n"
        "## Pitfalls\nnone\n\n"
        "## Verification\n```bash\necho verified\n```\n"
    )


# Schema supersede (re-assert the post-S177 reality)


class TestSchemaSupersede:
    def test_seven_schemas(self):
        assert len(t.ALL_SCHEMAS) == 7

    def test_manage_skills_is_third_non_sandbox(self):
        non = {s.name for s in t.ALL_SCHEMAS if not s.sandboxed}
        assert non == {t.TOOL_WEB_SEARCH, t.TOOL_MANAGE_MEMORY, t.TOOL_MANAGE_SKILLS}

    def test_manage_skills_schema_fields(self):
        names = {p.name for p in t.MANAGE_SKILLS_SCHEMA.parameters}
        assert "action" in names
        for f in ("name", "category", "body", "old_str", "new_str", "query"):
            assert f in names
        assert t.MANAGE_SKILLS_SCHEMA.required_names() == ["action"]
        assert t.MANAGE_SKILLS_SCHEMA.sandboxed is False

    def test_sandbox_four_unchanged(self):
        sandboxed = {s.name for s in t.ALL_SCHEMAS if s.sandboxed}
        assert sandboxed == set(al.SANDBOX_TOOL_NAMES)

    def test_daily_includes_manage_skills(self):
        ts = t.build_tool_set("daily")
        assert t.TOOL_MANAGE_SKILLS in ts.names
        assert t.TOOL_MANAGE_SKILLS in ts.tool_handlers
        assert set(ts.tool_handlers) == {
            t.TOOL_WEB_SEARCH,
            t.TOOL_MANAGE_MEMORY,
            t.TOOL_MANAGE_SKILLS,
        }

    def test_bulbe_excludes_manage_skills(self):
        ts = t.build_tool_set("bulbe")
        assert t.TOOL_MANAGE_SKILLS not in ts.names
        assert set(ts.names) == set(al.SANDBOX_TOOL_NAMES)

    def test_manage_skills_is_state_mutation(self):
        assert "manage_skills" in al.STATE_MUTATION_TOOLS
        assert "manage_skills" in al.DAILY_ALLOWLIST
        assert "manage_skills" not in al.BULBE_ALLOWLIST


# Handler read actions (no gate)


class TestHandlerReads:
    def test_list_returns_index(self, reg):
        reg.add("Alpha", "coding", _body(), status="published")
        reg.add("Beta", "writing", _body(), status="draft")
        h = sk.make_manage_skills_handler(registry=reg)
        out = h({"action": "list"})
        assert "Published skills (1)" in out
        assert "Drafts awaiting approval (1)" in out
        assert "alpha" in out and "beta" in out

    def test_view(self, reg):
        reg.add("Alpha", "coding", _body(), status="published")
        h = sk.make_manage_skills_handler(registry=reg)
        out = h({"action": "view", "name": "alpha", "category": "coding"})
        assert "## When to Use" in out

    def test_view_ref(self, reg):
        reg.add("Alpha", "coding", _body(), status="published")
        h = sk.make_manage_skills_handler(registry=reg)
        out = h({"action": "view_ref", "name": "alpha", "category": "coding"})
        assert "alpha" in out and "When to Use" in out

    def test_view_missing(self, reg):
        h = sk.make_manage_skills_handler(registry=reg)
        out = h({"action": "view", "name": "nope", "category": "coding"})
        assert "No skill" in out

    def test_search(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        h = sk.make_manage_skills_handler(registry=reg)
        out = h({"action": "search", "query": "deploy"})
        assert "deploy" in out

    def test_reads_do_not_invoke_gate(self, reg):
        reg.add("Alpha", "coding", _body(), status="published")
        approver = _Approver(False)
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=approver)
        h({"action": "list"})
        h({"action": "view", "name": "alpha", "category": "coding"})
        h({"action": "search", "query": "alpha"})
        assert approver.calls == []  # no gate for reads

    def test_unknown_action(self, reg):
        h = sk.make_manage_skills_handler(registry=reg)
        out = h({"action": "frobnicate"})
        assert "must be one of" in out


# Handler write gate (fail-secure)


class TestHandlerGate:
    def test_add_denied_writes_nothing(self, reg):
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(False))
        out = h({"action": "add", "name": "deploy", "category": "coding", "body": _body()})
        assert "not approved" in out
        assert reg.get("deploy", "coding", draft=True) is None

    def test_add_approved_creates_draft(self, reg):
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True))
        out = h({"action": "add", "name": "deploy", "category": "coding", "body": _body()})
        assert "Draft skill" in out
        assert reg.get("deploy", "coding", draft=True) is not None

    def test_add_requires_body(self, reg):
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True))
        out = h({"action": "add", "name": "deploy", "category": "coding"})
        assert "non-empty 'body'" in out

    def test_write_requires_name(self, reg):
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True))
        out = h({"action": "add", "body": _body()})
        assert "requires a 'name'" in out

    def test_gate_label_and_args(self, reg):
        approver = _Approver(True)
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=approver, conversation_id="c1")
        h({"action": "add", "name": "deploy", "category": "coding", "body": _body()})
        assert approver.calls
        conv, label, args = approver.calls[0]
        assert conv == "c1"
        assert label == "manage_skills:add"
        assert args["name"] == "deploy" and args["category"] == "coding"

    def test_edit_denied(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(False))
        out = h({"action": "edit", "name": "deploy", "category": "coding", "body": _body(proc="v2")})
        assert "not approved" in out
        assert reg.get("deploy", "coding").version == 1

    def test_edit_approved(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True))
        out = h({"action": "edit", "name": "deploy", "category": "coding", "body": _body(proc="v2")})
        assert "edited to v2" in out
        assert reg.get("deploy", "coding").version == 2

    def test_edit_missing(self, reg):
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True))
        out = h({"action": "edit", "name": "nope", "category": "coding", "body": _body()})
        assert "No published skill" in out

    def test_patch_denied(self, reg):
        reg.add("Deploy", "coding", _body(proc="run the deploy script"), status="published")
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(False))
        out = h({
            "action": "patch", "name": "deploy", "category": "coding",
            "old_str": "run the deploy script", "new_str": "run deploy.sh",
        })
        assert "not approved" in out
        assert "run the deploy script" in reg.get("deploy", "coding").body

    def test_patch_approved(self, reg):
        reg.add("Deploy", "coding", _body(proc="run the deploy script"), status="published")
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True))
        out = h({
            "action": "patch", "name": "deploy", "category": "coding",
            "old_str": "run the deploy script", "new_str": "run deploy.sh",
        })
        assert "patched to v2" in out
        assert "run deploy.sh" in reg.get("deploy", "coding").body

    def test_patch_non_unique(self, reg):
        reg.add("Deploy", "coding", "## Procedure\nx x", status="published")
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True))
        out = h({"action": "patch", "name": "deploy", "category": "coding", "old_str": "x", "new_str": "y"})
        assert "exactly once" in out

    def test_publish_denied(self, reg):
        reg.add("Deploy", "coding", _body(), status="draft")
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(False))
        out = h({"action": "publish", "name": "deploy", "category": "coding"})
        assert "not approved" in out
        assert reg.get("deploy", "coding") is None  # not published
        assert reg.get("deploy", "coding", draft=True) is not None  # draft intact

    def test_publish_approved(self, reg):
        reg.add("Deploy", "coding", _body(), status="draft")
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True))
        out = h({"action": "publish", "name": "deploy", "category": "coding"})
        assert "published as v" in out
        assert reg.get("deploy", "coding") is not None

    def test_publish_missing_draft(self, reg):
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True))
        out = h({"action": "publish", "name": "deploy", "category": "coding"})
        assert "No draft" in out

    def test_delete_denied(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(False))
        out = h({"action": "delete", "name": "deploy", "category": "coding"})
        assert "not approved" in out
        assert reg.get("deploy", "coding") is not None

    def test_delete_approved(self, reg):
        reg.add("Deploy", "coding", _body(), status="published")
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True))
        out = h({"action": "delete", "name": "deploy", "category": "coding"})
        assert "deleted" in out
        assert reg.get("deploy", "coding") is None

    def test_missing_gate_is_fail_secure(self, reg):
        # The default gate path with an unusable approval manager denies the
        # write (a submit failure is treated as a denial), so nothing is
        # written. Exercises fail-secure without a real approval wait.
        class _BrokenManager:
            def submit(self, *a, **k):
                raise RuntimeError("gate down")

        h = sk.make_manage_skills_handler(registry=reg, manager=_BrokenManager())
        out = h({"action": "add", "name": "deploy", "category": "coding", "body": _body()})
        assert "not approved" in out
        assert reg.get("deploy", "coding", draft=True) is None


# Handler sandbox verification


class TestHandlerSandboxVerification:
    def test_add_with_verification_refused_without_sandbox(self, reg):
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True), sandbox=None)
        out = h({"action": "add", "name": "deploy", "category": "coding", "body": _body_with_cmd()})
        assert "bwrap sandbox is unavailable" in out
        assert reg.get("deploy", "coding", draft=True) is None

    def test_add_with_verification_runs_in_sandbox(self, reg):
        sb = _FakeSandbox()
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True), sandbox=sb)
        out = h({"action": "add", "name": "deploy", "category": "coding", "body": _body_with_cmd()})
        assert "sandbox-tested" in out
        assert "echo verified" in sb.calls[0]
        assert reg.get("deploy", "coding", draft=True) is not None

    def test_add_without_verification_needs_no_sandbox(self, reg):
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True), sandbox=None)
        out = h({"action": "add", "name": "deploy", "category": "coding", "body": _body()})
        assert "Draft skill" in out
        assert reg.get("deploy", "coding", draft=True) is not None

    def test_publish_with_verification_refused_without_sandbox(self, reg):
        reg.add("Deploy", "coding", _body_with_cmd(), status="draft")
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True), sandbox=None)
        out = h({"action": "publish", "name": "deploy", "category": "coding"})
        assert "bwrap sandbox is unavailable" in out
        assert reg.get("deploy", "coding") is None

    def test_verification_refused_before_write_even_when_approved(self, reg):
        # bwrap down with verification steps -> nothing written despite approval.
        sb = _FakeSandbox(bwrap=False)
        h = sk.make_manage_skills_handler(registry=reg, approval_fn=_Approver(True), sandbox=sb)
        out = h({"action": "add", "name": "deploy", "category": "coding", "body": _body_with_cmd()})
        assert "not drafted" in out
        assert reg.get("deploy", "coding", draft=True) is None


# Handler never raises


class TestHandlerNeverRaises:
    def test_broken_registry_becomes_observation(self):
        class _Broken:
            def index(self):
                raise RuntimeError("boom")

        h = sk.make_manage_skills_handler(registry=_Broken())
        out = h({"action": "list"})
        assert "manage_skills failed" in out

    def test_none_arguments(self, reg):
        h = sk.make_manage_skills_handler(registry=reg)
        out = h(None)
        assert isinstance(out, str)


# sandbox_test_verification helper


class TestSandboxTestVerification:
    def test_no_commands_passes_without_sandbox(self):
        res = sk.sandbox_test_verification(_body(), None)
        assert res.ok is True and res.tested is False

    def test_commands_refused_without_sandbox(self):
        res = sk.sandbox_test_verification(_body_with_cmd(), None)
        assert res.ok is False and res.tested is False

    def test_commands_run_in_ready_sandbox(self):
        sb = _FakeSandbox()
        res = sk.sandbox_test_verification(_body_with_cmd(), sb)
        assert res.ok is True and res.tested is True
        assert sb.calls == ["echo verified"]

    def test_commands_refused_when_sandbox_inactive(self):
        sb = _FakeSandbox(active=False)
        res = sk.sandbox_test_verification(_body_with_cmd(), sb)
        assert res.ok is False

    def test_commands_refused_when_bwrap_unavailable(self):
        sb = _FakeSandbox(bwrap=False)
        res = sk.sandbox_test_verification(_body_with_cmd(), sb)
        assert res.ok is False

    def test_failing_step_becomes_failure(self):
        class _Boom(_FakeSandbox):
            def bash(self, command, timeout=30):
                raise RuntimeError("nope")

        res = sk.sandbox_test_verification(_body_with_cmd(), _Boom())
        assert res.ok is False and res.tested is True


# publish_teacher_draft


def _draft(name="rescue-step", category="coding", body=None, source="teacher-escalation"):
    return teacher.TeacherSkillDraft(
        name=name, category=category, content=body if body is not None else _body(), source=source
    )


class TestPublishTeacherDraft:
    def test_approved_publishes(self, reg):
        res = sk.publish_teacher_draft(_draft(), registry=reg, approval_fn=_Approver(True))
        assert res.published is True
        assert res.reason == "published"
        assert reg.get("rescue-step", "coding") is not None

    def test_denied_not_published(self, reg):
        res = sk.publish_teacher_draft(_draft(), registry=reg, approval_fn=_Approver(False))
        assert res.published is False
        assert res.reason == "not_approved"
        assert reg.get("rescue-step", "coding") is None

    def test_uses_publish_skill_gate_label(self, reg):
        approver = _Approver(True)
        sk.publish_teacher_draft(_draft(), registry=reg, approval_fn=approver, conversation_id="c9")
        assert approver.calls
        conv, label, _ = approver.calls[0]
        assert conv == "c9" and label == "publish_skill"

    def test_verification_present_refused_without_sandbox(self, reg):
        res = sk.publish_teacher_draft(
            _draft(body=_body_with_cmd()), registry=reg, approval_fn=_Approver(True), sandbox=None
        )
        assert res.published is False
        assert res.reason == "verification_failed"
        assert reg.get("rescue-step", "coding") is None

    def test_verification_runs_in_sandbox(self, reg):
        sb = _FakeSandbox()
        res = sk.publish_teacher_draft(
            _draft(body=_body_with_cmd()), registry=reg, approval_fn=_Approver(True), sandbox=sb
        )
        assert res.published is True
        assert sb.calls == ["echo verified"]

    def test_preserves_teacher_source(self, reg):
        res = sk.publish_teacher_draft(_draft(), registry=reg, approval_fn=_Approver(True))
        assert res.skill.source == "teacher-escalation"

    def test_never_raises_on_broken_draft(self, reg):
        class _BadDraft:
            name = "x"
            category = "y"
            # no content / body attributes resolve to empty -> no verification

        res = sk.publish_teacher_draft(_BadDraft(), registry=reg, approval_fn=_Approver(True))
        assert isinstance(res, sk.SkillPublishResult)

    def test_observation_string(self, reg):
        res = sk.publish_teacher_draft(_draft(), registry=reg, approval_fn=_Approver(True))
        assert "published" in res.observation().lower()
