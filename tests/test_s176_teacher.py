#!/usr/bin/env python3
"""Tests for S176 -- teacher escalation (Theme 3 / Odysseus Core).

Covers ODYSSEUS_SPEC.md Section 5.6:

- The configurable escalation policy (laptop-lite reference) and the decision
  logic over an ``AgentRunResult``-shaped outcome.
- The teacher call: guidance plus an optional authoritative SKILL.md draft,
  tagged with a teacher-escalation source; the call never raises (missing
  client, error, timeout, empty are reasons, not exceptions).
- Guidance, not authority: the draft is unapproved and is not published in
  S176; the human-approval gate hook is fail-secure and reused from
  ``allowlists`` by default.
- Cartography: ``teacher.py`` is registered in ODYSSEUS_SPEC.md.

Loaded in isolation via ``spec_from_file_location`` with ``opti_oignon``
stubbed, so the runtime collects without the backend.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"
SPEC = ROOT / "ODYSSEUS_SPEC.md"


def _ensure_pkg():
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(OO)]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.agent" not in sys.modules:
        apkg = types.ModuleType("opti_oignon.agent")
        apkg.__path__ = [str(AGENT)]
        sys.modules["opti_oignon.agent"] = apkg


def _ensure_agent(name: str):
    full = f"opti_oignon.agent.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(AGENT / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg()
_ensure_agent("tool_parsing")
_ensure_agent("allowlists")
_ensure_agent("untrusted_context")
te = _ensure_agent("teacher")


@pytest.fixture(autouse=True)
def _reset_policy():
    te.reset_default_policy()
    yield
    te.reset_default_policy()


# AgentRunResult-shaped doubles


class _ToolRes:
    def __init__(self, executed: bool):
        self.executed = executed


class _Verifier:
    def __init__(self, verdict: str):
        self.verdict = verdict


class _RunResult:
    def __init__(self, stop_reason="done", tool_results=None, verifier=None):
        self.stop_reason = stop_reason
        self.tool_results = tool_results or []
        self.verifier = verifier


# Teacher client doubles


def _stream_with_draft(messages, tools=None):
    yield {
        "message": {
            "content": (
                "Step 1: read the file.\n"
                "```skill\n"
                "name: Parse FASTA\n"
                "category: Bioinformatics\n"
                "## When to Use\nWhen reading sequence files.\n"
                "## Procedure\nOpen, iterate.\n"
                "## Pitfalls\nWatch encodings.\n"
                "## Verification\nCount records.\n"
                "```\n"
            )
        }
    }


def _stream_guidance_only(messages, tools=None):
    yield {"message": {"content": "Just use a context manager.\n```python\nopen(p)\n```"}}


# Module conventions


class TestModuleConventions:
    def test_sentinels(self):
        assert te.checkpoint_before_apply is True
        assert te.FEATURE_AVAILABLE is True

    def test_source_tag(self):
        assert te.SOURCE_TEACHER == "teacher-escalation"

    def test_default_policy_singleton(self):
        a = te.get_default_policy()
        b = te.get_default_policy()
        assert a is b

    def test_reset_default_policy(self):
        a = te.get_default_policy()
        te.reset_default_policy()
        assert te.get_default_policy() is not a

    def test_set_default_policy(self):
        p = te.EscalationPolicy(failure_threshold=5)
        te.set_default_policy(p)
        assert te.get_default_policy() is p


# Policy


class TestPolicy:
    def test_defaults(self):
        p = te.EscalationPolicy()
        assert p.enabled is True
        assert p.failure_threshold == te.DEFAULT_FAILURE_THRESHOLD
        assert p.teacher_model == te.DEFAULT_TEACHER_MODEL
        assert p.student_model == te.DEFAULT_STUDENT_MODEL

    def test_from_dict_overrides(self):
        p = te.EscalationPolicy.from_dict(
            {"failure_threshold": 3, "teacher_model": "qwen3:32b", "on_max_rounds": False}
        )
        assert p.failure_threshold == 3
        assert p.teacher_model == "qwen3:32b"
        assert p.on_max_rounds is False

    def test_from_dict_clamps_threshold(self):
        p = te.EscalationPolicy.from_dict({"failure_threshold": 0})
        assert p.failure_threshold >= 1

    def test_from_dict_ignores_unknown_and_bad_types(self):
        p = te.EscalationPolicy.from_dict({"unknown": 1, "failure_threshold": "x", "timeout": "y"})
        assert p.failure_threshold == te.DEFAULT_FAILURE_THRESHOLD
        assert p.timeout == te.DEFAULT_TEACHER_TIMEOUT

    def test_from_dict_none(self):
        p = te.EscalationPolicy.from_dict(None)
        assert isinstance(p, te.EscalationPolicy)


# should_escalate


class TestShouldEscalate:
    def test_success_no_escalation(self):
        d = te.should_escalate(_RunResult("done", [_ToolRes(True)]))
        assert d.escalate is False
        assert d.reason == te.REASON_NONE

    def test_model_error(self):
        d = te.should_escalate(_RunResult("error"))
        assert d.escalate is True
        assert d.reason == te.REASON_MODEL_ERROR

    def test_max_rounds(self):
        d = te.should_escalate(_RunResult("max_rounds"))
        assert d.escalate is True
        assert d.reason == te.REASON_MAX_ROUNDS

    def test_verifier_fail(self):
        d = te.should_escalate(_RunResult("done", [_ToolRes(True)], _Verifier("fail")))
        assert d.escalate is True
        assert d.reason == te.REASON_VERIFIER_FAIL

    def test_verifier_pass_no_escalation(self):
        d = te.should_escalate(_RunResult("done", [_ToolRes(True)], _Verifier("pass")))
        assert d.escalate is False

    def test_repeated_failure_at_threshold(self):
        d = te.should_escalate(_RunResult("done", [_ToolRes(True), _ToolRes(False), _ToolRes(False)]))
        assert d.escalate is True
        assert d.reason == te.REASON_REPEATED_FAILURE

    def test_single_failure_below_threshold(self):
        d = te.should_escalate(_RunResult("done", [_ToolRes(False)]))
        assert d.escalate is False

    def test_disabled_policy(self):
        p = te.EscalationPolicy(enabled=False)
        d = te.should_escalate(_RunResult("error"), p)
        assert d.escalate is False
        assert d.reason == te.REASON_DISABLED

    def test_on_max_rounds_flag_off(self):
        p = te.EscalationPolicy(on_max_rounds=False)
        assert te.should_escalate(_RunResult("max_rounds"), p).escalate is False

    def test_on_verifier_fail_flag_off(self):
        p = te.EscalationPolicy(on_verifier_fail=False)
        r = _RunResult("done", [_ToolRes(True)], _Verifier("fail"))
        assert te.should_escalate(r, p).escalate is False

    def test_on_model_error_flag_off(self):
        p = te.EscalationPolicy(on_model_error=False)
        assert te.should_escalate(_RunResult("error"), p).escalate is False

    def test_higher_threshold_respected(self):
        p = te.EscalationPolicy(failure_threshold=3)
        r = _RunResult("done", [_ToolRes(False), _ToolRes(False)])
        assert te.should_escalate(r, p).escalate is False


# escalate


class TestEscalate:
    def test_returns_guidance(self):
        res = te.escalate("task", teacher_client=_stream_guidance_only)
        assert res.escalated is True
        assert res.reason == te.REASON_ESCALATED
        assert "context manager" in res.guidance

    def test_extracts_draft(self):
        res = te.escalate("task", teacher_client=_stream_with_draft)
        assert res.draft is not None
        assert res.draft.name == "parse-fasta"
        assert res.draft.category == "bioinformatics"

    def test_draft_tagged_and_unapproved(self):
        res = te.escalate("task", teacher_client=_stream_with_draft)
        assert res.draft.source == te.SOURCE_TEACHER
        assert res.draft.approved is False

    def test_guidance_only_no_draft(self):
        res = te.escalate("task", teacher_client=_stream_guidance_only)
        assert res.draft is None

    def test_no_teacher(self):
        res = te.escalate("task", teacher_client=None)
        assert res.escalated is False
        assert res.reason == te.REASON_NO_TEACHER

    def test_teacher_error_never_raises(self):
        def boom(messages, tools=None):
            raise RuntimeError("down")

        res = te.escalate("task", teacher_client=boom)
        assert res.escalated is False
        assert res.reason == te.REASON_TEACHER_ERROR
        assert "down" in res.error

    def test_teacher_timeout(self):
        def slow(messages, tools=None):
            raise TimeoutError("slow")

        res = te.escalate("task", teacher_client=slow)
        assert res.reason == te.REASON_TEACHER_TIMEOUT

    def test_teacher_empty(self):
        res = te.escalate("task", teacher_client=lambda m, tools=None: "")
        assert res.reason == te.REASON_TEACHER_EMPTY

    def test_accepts_single_dict_response(self):
        res = te.escalate("task", teacher_client=lambda m, tools=None: {"message": {"content": "ok"}})
        assert res.escalated is True
        assert res.guidance == "ok"

    def test_accepts_plain_string_response(self):
        res = te.escalate("task", teacher_client=lambda m, tools=None: "plain")
        assert res.escalated is True

    def test_accepts_object_with_stream(self):
        class Client:
            def stream(self, messages, tools=None):
                yield {"message": {"content": "streamed"}}

        res = te.escalate("task", teacher_client=Client())
        assert "streamed" in res.guidance

    def test_callable_without_tools_kwarg(self):
        def no_tools(messages):
            return "no-tools-arg"

        res = te.escalate("task", teacher_client=no_tools)
        assert res.escalated is True

    def test_on_event_called(self):
        seen = []
        te.escalate(
            "task",
            teacher_client=_stream_with_draft,
            on_event=lambda kind, data: seen.append((kind, data)),
        )
        assert seen and seen[0][0] == "teacher_guidance"

    def test_observation_for_escalated(self):
        res = te.escalate("task", teacher_client=_stream_with_draft)
        obs = res.observation()
        assert "Teacher guidance" in obs
        assert "awaits approval" in obs

    def test_observation_for_not_escalated(self):
        res = te.escalate("task", teacher_client=None)
        assert te.REASON_NO_TEACHER in res.observation()


# SKILL.md draft extraction


class TestSkillDraftExtraction:
    def test_extract_from_skill_fence(self):
        text = "intro\n```skill\nname: Foo Bar\ncategory: General\nbody\n```\nrest"
        d = te.extract_skill_draft(text)
        assert d is not None
        assert d.name == "foo-bar"
        assert d.category == "general"

    def test_ignores_ordinary_code_fence(self):
        text = "```python\nprint('x')\n```"
        assert te.extract_skill_draft(text) is None

    def test_no_block_returns_none(self):
        assert te.extract_skill_draft("no skill here") is None

    def test_empty_text_returns_none(self):
        assert te.extract_skill_draft("") is None

    def test_missing_fields_use_fallbacks(self):
        text = "```skill\njust a body with no frontmatter\n```"
        d = te.extract_skill_draft(text)
        assert d.name == "untitled-skill"
        assert d.category == "general"

    def test_empty_block_returns_none(self):
        text = "```skill\n\n```"
        assert te.extract_skill_draft(text) is None

    def test_content_preserved(self):
        text = "```skill\nname: X\ncategory: Y\nProcedure body\n```"
        d = te.extract_skill_draft(text)
        assert "Procedure body" in d.content


# The approval gate hook (publish is S177)


class _Event:
    def wait(self, timeout):
        return True


class _Mgr:
    def __init__(self, status):
        self._status = status
        self.submitted = []

    def submit(self, conversation_id, tool_name, arguments):
        self.submitted.append((conversation_id, tool_name, arguments))
        return ("approval-1", _Event())

    def get_status(self, approval_id):
        return self._status


class TestApprovalGateHook:
    def _draft(self):
        return te.TeacherSkillDraft(name="x", category="y", content="body")

    def test_injected_approval_grant(self):
        assert te.request_skill_approval(self._draft(), approval_fn=lambda c, t, a: True) is True

    def test_injected_approval_deny(self):
        assert te.request_skill_approval(self._draft(), approval_fn=lambda c, t, a: False) is False

    def test_injected_approval_error_is_fail_secure(self):
        def boom(c, t, a):
            raise ValueError("x")

        assert te.request_skill_approval(self._draft(), approval_fn=boom) is False

    def test_default_delegates_to_manager_approved(self):
        ok = te.request_skill_approval(self._draft(), manager=_Mgr("approved"))
        assert ok is True

    def test_default_delegates_to_manager_denied(self):
        ok = te.request_skill_approval(self._draft(), manager=_Mgr("denied"))
        assert ok is False

    def test_submits_publish_skill_pseudo_tool(self):
        mgr = _Mgr("approved")
        te.request_skill_approval(self._draft(), manager=mgr)
        assert mgr.submitted and mgr.submitted[0][1] == "publish_skill"

    def test_escalate_does_not_auto_approve(self):
        # The draft is guidance only; escalate never approves or publishes it.
        res = te.escalate("task", teacher_client=_stream_with_draft)
        assert res.draft.approved is False


# The escalator class


class TestEscalator:
    def test_should_escalate_uses_bound_policy(self):
        esc = te.TeacherEscalator(policy=te.EscalationPolicy(on_max_rounds=False))
        assert esc.should_escalate(_RunResult("max_rounds")).escalate is False

    def test_escalate_uses_bound_client(self):
        esc = te.TeacherEscalator(teacher_client=_stream_with_draft)
        res = esc.escalate("task")
        assert res.escalated is True
        assert res.draft is not None

    def test_per_call_client_override(self):
        esc = te.TeacherEscalator(teacher_client=None)
        res = esc.escalate("task", teacher_client=_stream_guidance_only)
        assert res.escalated is True

    def test_maybe_escalate_none_when_not_needed(self):
        esc = te.TeacherEscalator(teacher_client=_stream_with_draft)
        out = esc.maybe_escalate(_RunResult("done", [_ToolRes(True)]), "task")
        assert out is None

    def test_maybe_escalate_runs_when_needed(self):
        esc = te.TeacherEscalator(teacher_client=_stream_with_draft)
        out = esc.maybe_escalate(_RunResult("error"), "task")
        assert out is not None
        assert out.escalated is True


# Cartography


class TestCartography:
    def test_teacher_registered_in_spec(self):
        text = SPEC.read_text(encoding="utf-8")
        assert "opti_oignon/agent/teacher.py" in text

    def test_teacher_file_on_disk(self):
        assert (AGENT / "teacher.py").exists()
