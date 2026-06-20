#!/usr/bin/env python3
"""Tests for S228 -- AGT Lot 1: tools and feedback (AGT_SPEC Section 5).

Container-provable coverage (AGT_SPEC Section 11, Lot 1):

- Schema and registry growth per mode: twelve schemas, the sandboxed seven,
  Daily exposing everything, Bulbe gaining todo and task plus its first tool
  handler; allowlist derivations and the Bulbe approval exemption for the
  session-state and subagent tools (the 5.6 mode-posture table).
- Dispatch seams: the three new read-only lambdas with schema-exact argument
  names; the lambda-coverage pin re-asserted over all SEVEN sandboxed schemas.
- grep / glob / ls behaviour on tempdir workspace fixtures: path confinement
  via the real validate_sandbox_path, caps and truncated flags, deterministic
  sorted output under fixed mtimes, the null-byte binary sniff and the 1 MiB
  skip, symlinks never followed.
- todo handler semantics (replacement list, validation, never raises) and the
  loop's gated binding emitting AgentEvent ``todo_updated``.
- task bounding with a scripted fake client: depth 1 through the child's
  empty handler map, the min(requested, TASK_CHILD_CAP, remaining - 1)
  arithmetic, the parent-budget debit, the bound report, Bulbe approval
  inheritance, and the task marker on re-emitted child events.
- Diagnostics gating logic: the suffix map, probe caching (one probe per
  session), bwrap-only execution asserted by backend injection, clean-path
  byte-identity on both the degraded backend and a clean bwrap run, the
  finding caps, and the pure Svelte tag-balance checker on synthetic content.

Host-assured, NAMED here and never simulated in the container (AGT_SPEC
Section 11): real in-bwrap linter execution and real ruff output text (the
container has no bwrap; the ladder is exercised through an injected
execute_command); the tag-balance path on real Svelte trees (the pure checker
is exercised on synthetic components only).

Supersessions (S228; deselect-plus-reassert, originals never edited). The
fourteen deselected originals and their reassertions in this file:

 1. test_s177_manage_skills::TestSchemaSupersede::test_seven_schemas
    -> TestSupersessionReassertions::test_twelve_schemas
 2. ...::test_manage_skills_is_third_non_sandbox
    -> ...::test_non_sandbox_set_gains_todo_and_task
 3. ...::test_daily_includes_manage_skills
    -> ...::test_daily_handler_set_gains_todo
 4. ...::test_bulbe_excludes_manage_skills
    -> ...::test_bulbe_excludes_manage_skills_and_gains_session_tools
 5. test_s176_tools::TestRegistryPerMode::test_bulbe_has_no_handlers
    -> ...::test_bulbe_first_handler_is_todo
 6. ...::test_bulbe_exposes_sandbox_only
    -> ...::test_bulbe_exposes_sandbox_plus_session_tools
 7. ...::test_unknown_mode_is_fail_secure_bulbe
    -> ...::test_unknown_mode_gets_bulbe_set
 8. test_s176_tools::TestSchemas::test_sandbox_argument_names_cover_dispatch_lambdas
    -> ...::test_sandbox_argument_names_cover_all_seven_lambdas
 9. test_s222_agt_spec::TestSeamToolsSchemas::test_all_schemas_is_seven_today
    -> ...::test_all_schemas_is_twelve_today
10. ...::test_handler_names_are_the_three
    -> ...::test_handler_names_are_the_four
11. test_s222_agt_spec::TestSeamAllowlists::test_frozensets_exact
    -> ...::test_frozensets_exact_post_s228
12. test_s175_allowlists::TestAllowlistContents::test_sandbox_tools
    -> ...::test_sandbox_tools_are_the_seven
13. ...::test_bulbe_equals_sandbox_tools
    -> ...::test_bulbe_equals_sandbox_plus_session_tools
14. test_s176_config::TestPerModeTools::test_bulbe_equals_sandbox_set
    -> ...::test_config_bulbe_list_equals_sandbox_plus_session_tools

Loaded in isolation via ``spec_from_file_location`` with the package stubs and
the ollama setdefault, so the runtime collects without the backend.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

sys.modules.setdefault("ollama", types.ModuleType("ollama"))

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"


def _ensure_pkg(name: str, path: Path) -> None:
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg


_ensure_pkg("opti_oignon", OO)
_ensure_pkg("opti_oignon.agent", AGENT)


def _load(register: str, path: Path):
    if register in sys.modules:
        return sys.modules[register]
    spec = importlib.util.spec_from_file_location(register, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[register] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load("opti_oignon.sandbox_manager", OO / "sandbox_manager.py")
st = _load("opti_oignon.sandbox_tools", OO / "sandbox_tools.py")
_load("opti_oignon.agent.tool_parsing", AGENT / "tool_parsing.py")
al = _load("opti_oignon.agent.allowlists", AGENT / "allowlists.py")
d = _load("opti_oignon.agent.dispatch", AGENT / "dispatch.py")
t = _load("opti_oignon.agent.tools", AGENT / "tools.py")
uc = _load("opti_oignon.agent.untrusted_context", AGENT / "untrusted_context.py")
L = _load("opti_oignon.agent.loop", AGENT / "loop.py")
cfg = _load("opti_oignon.agent.config_loader", AGENT / "config_loader.py")


@pytest.fixture(autouse=True)
def _reset_registry():
    t.reset_tool_registry()
    yield
    t.reset_tool_registry()


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class ScriptedClient:
    """A fake model client replaying scripted Ollama-shaped rounds."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def stream(self, messages, tools=None):
        self.calls += 1
        step = self.script.pop(0) if self.script else {"content": "done", "tool_calls": None}
        return {"message": step}


def _native(name, args):
    return [{"function": {"name": name, "arguments": args}}]


class FakeLoopSession:
    """A recording sandbox session for loop/dispatch tests (bwrap-shaped)."""

    def __init__(self, bwrap: bool = True, active: bool = True):
        self.sandbox_manager = types.SimpleNamespace(bwrap_available=bwrap)
        self.active = active
        self.calls: list[tuple] = []

    def bash(self, command, timeout=30):
        self.calls.append(("bash", command, timeout))
        return f"[sandbox] {command}"

    def view(self, path, start_line=0, end_line=0):
        self.calls.append(("view", path, start_line, end_line))
        return "[sandbox] view"

    def create_file(self, path, content):
        self.calls.append(("create_file", path, content))
        return "[sandbox] create"

    def str_replace(self, path, old_str, new_str=""):
        self.calls.append(("str_replace", path, old_str, new_str))
        return "[sandbox] replace"

    def grep(self, pattern, path=".", *, glob="", is_regex=False,
             case_sensitive=False, context_lines=0, max_results=100):
        self.calls.append(
            ("grep", pattern, path, glob, is_regex, case_sensitive, context_lines, max_results)
        )
        return "[sandbox] grep"

    def glob(self, pattern, path=".", *, max_results=200):
        self.calls.append(("glob", pattern, path, max_results))
        return "[sandbox] glob"

    def ls(self, path=".", *, max_entries=200):
        self.calls.append(("ls", path, max_entries))
        return "[sandbox] ls"


class WorkspaceMgr:
    """The minimal manager surface SandboxToolSession's read tools need."""

    def __init__(self, workspace: str):
        self.workspace = workspace

    def get_workspace_path(self, session_id):
        return self.workspace


class DiagMgr(WorkspaceMgr):
    """A workspace manager with a scripted execute_command for diagnostics."""

    def __init__(self, workspace: str, responder):
        super().__init__(workspace)
        self.responder = responder
        self.commands: list[str] = []

    def execute_command(self, session_id, command, timeout=None):
        self.commands.append(command)
        return self.responder(command)


class CmdResult:
    def __init__(self, rc, stdout="", stderr="", blocked=False, timed_out=False):
        self.return_code = rc
        self.stdout = stdout
        self.stderr = stderr
        self.blocked = blocked
        self.timed_out = timed_out


def _session(workspace: str, backend: str = "tempdir", mgr=None) -> "st.SandboxToolSession":
    """A SandboxToolSession bound to a real directory with an injected backend."""
    s = st.SandboxToolSession(sandbox_mgr=mgr or WorkspaceMgr(workspace), tool_registry=None)
    s._session = types.SimpleNamespace(active=True, isolation_backend=backend)
    s._session_id = "s228-test"
    return s


@pytest.fixture()
def workspace(tmp_path):
    """A deterministic workspace tree with fixed mtimes.

    src/a.py (mtime 1000), b.txt (mtime 2000), bin.dat (binary, mtime 3000),
    big.txt (>1 MiB, mtime 4000), and a symlink escape attempt.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("alpha\nBeta needle here\ngamma\n")
    (tmp_path / "b.txt").write_text("needle\nNEEDLE again\n")
    (tmp_path / "bin.dat").write_bytes(b"x\x00y needle")
    (tmp_path / "big.txt").write_text("needle\n" + "x" * (1024 * 1024))
    os.utime(tmp_path / "src" / "a.py", (1000, 1000))
    os.utime(tmp_path / "b.txt", (2000, 2000))
    os.utime(tmp_path / "bin.dat", (3000, 3000))
    os.utime(tmp_path / "big.txt", (4000, 4000))
    try:
        os.symlink("/etc/passwd", tmp_path / "leak.txt")
    except OSError:
        pass
    return tmp_path


# ---------------------------------------------------------------------------
# Registry growth and allowlists
# ---------------------------------------------------------------------------


class TestRegistryGrowth:
    def test_schema_order_is_stable(self):
        names = [s.name for s in t.ALL_SCHEMAS]
        assert names == [
            "bash", "view", "create_file", "str_replace",
            "grep", "glob", "ls",
            "web_search", "manage_memory", "manage_skills",
            "todo", "task",
        ]

    def test_sandboxed_schemas_match_allowlist(self):
        sandboxed = {s.name for s in t.ALL_SCHEMAS if s.sandboxed}
        assert sandboxed == set(al.SANDBOX_TOOL_NAMES)
        assert len(sandboxed) == 7

    def test_daily_exposes_all_twelve(self):
        ts = t.build_tool_set("daily")
        assert set(ts.names) == {s.name for s in t.ALL_SCHEMAS}
        assert len(ts.names) == 12

    def test_daily_handlers_include_todo(self):
        ts = t.build_tool_set("daily")
        assert set(ts.tool_handlers) == {
            t.TOOL_WEB_SEARCH, t.TOOL_MANAGE_MEMORY, t.TOOL_MANAGE_SKILLS, t.TOOL_TODO,
        }

    def test_task_has_schema_but_no_handler(self):
        ts = t.build_tool_set("daily")
        assert t.TOOL_TASK in ts.names
        assert t.TOOL_TASK not in ts.tool_handlers

    def test_todo_closure_is_fresh_per_build(self):
        h1 = t.build_tool_set("daily").tool_handlers[t.TOOL_TODO]
        h2 = t.build_tool_set("daily").tool_handlers[t.TOOL_TODO]
        assert h1 is not h2
        h1({"todos": [{"content": "only in h1"}]})
        assert h1.state and not h2.state

    def test_include_handlers_false_yields_no_handlers(self):
        ts = t.build_tool_set("daily", include_handlers=False)
        assert ts.tool_handlers == {}

    def test_bulbe_names_are_strict_subset_of_daily(self):
        daily = set(t.build_tool_set("daily").names)
        bulbe = set(t.build_tool_set("bulbe").names)
        assert bulbe < daily

    def test_registry_subset_of_allowlists(self):
        for mode in ("daily", "bulbe"):
            for name in t.build_tool_set(mode).names:
                assert al.is_tool_allowed(name, mode)

    def test_new_schema_required_names(self):
        assert t.GREP_SCHEMA.required_names() == ["pattern"]
        assert t.GLOB_SCHEMA.required_names() == ["pattern"]
        assert t.LS_SCHEMA.required_names() == []
        assert set(t.TASK_SCHEMA.required_names()) == {"description", "prompt"}
        assert t.TODO_SCHEMA.required_names() == ["todos"]

    def test_new_sandbox_schemas_marked_sandboxed(self):
        for schema in (t.GREP_SCHEMA, t.GLOB_SCHEMA, t.LS_SCHEMA):
            assert schema.sandboxed is True
        for schema in (t.TODO_SCHEMA, t.TASK_SCHEMA):
            assert schema.sandboxed is False


class TestAllowlistGate:
    def test_session_and_subagent_classes(self):
        assert al.SESSION_STATE_TOOLS == frozenset({"todo"})
        assert al.SUBAGENT_TOOLS == frozenset({"task"})

    def test_daily_is_the_union_of_five(self):
        assert al.DAILY_ALLOWLIST == frozenset(
            al.SANDBOX_TOOL_NAMES
            | al.NETWORK_TOOLS
            | al.STATE_MUTATION_TOOLS
            | al.SESSION_STATE_TOOLS
            | al.SUBAGENT_TOOLS
        )

    def test_bulbe_derivation_is_structural(self):
        assert al.BULBE_ALLOWLIST == frozenset(
            al.DAILY_ALLOWLIST - al.NETWORK_TOOLS - al.STATE_MUTATION_TOOLS
        )
        assert al.BULBE_ALLOWLIST < al.DAILY_ALLOWLIST

    def test_network_and_state_tools_still_out_of_bulbe(self):
        assert not (al.NETWORK_TOOLS & al.BULBE_ALLOWLIST)
        assert not (al.STATE_MUTATION_TOOLS & al.BULBE_ALLOWLIST)

    def test_bulbe_todo_and_task_skip_the_approval_gate(self):
        # The 5.6 mode-posture table: the ceremony gates actions with
        # consequences; a deny-everything approval_fn must not block these.
        deny = lambda c, tool, a: False  # noqa: E731
        for name in ("todo", "task"):
            decision = al.evaluate(name, mode="bulbe", approval_fn=deny)
            assert decision.allowed and decision.reason == al.REASON_ALLOWED

    def test_bulbe_sandbox_tools_still_ride_the_approval(self):
        deny = lambda c, tool, a: False  # noqa: E731
        for name in sorted(al.SANDBOX_TOOL_NAMES):
            decision = al.evaluate(name, mode="bulbe", approval_fn=deny)
            assert not decision.allowed and decision.reason == al.REASON_DENIED

    def test_bulbe_new_sandbox_tools_execute_only_when_approved(self):
        approvals = []

        def approve(conv, tool, args):
            approvals.append(tool)
            return True

        sess = FakeLoopSession()
        call = d.ToolCall(name="grep", arguments={"pattern": "x"})
        r = d.dispatch_tool_call(call, mode="bulbe", sandbox=sess, approval_fn=approve)
        assert r.executed and approvals == ["grep"]


# ---------------------------------------------------------------------------
# Dispatch seam coverage
# ---------------------------------------------------------------------------


class TestDispatchSeam:
    def test_three_new_lambdas_route_through_session(self):
        sess = FakeLoopSession()
        for call, expect in [
            (d.ToolCall(name="grep", arguments={"pattern": "x"}), "grep"),
            (d.ToolCall(name="glob", arguments={"pattern": "*.py"}), "glob"),
            (d.ToolCall(name="ls", arguments={}), "ls"),
        ]:
            r = d.dispatch_tool_call(call, mode="daily", sandbox=sess)
            assert r.executed is True and r.reason == d.REASON_EXECUTED
        kinds = [c[0] for c in sess.calls]
        assert kinds == ["grep", "glob", "ls"]
        # Defaults: path '.', caps from the schemas, booleans off.
        assert sess.calls[0] == ("grep", "x", ".", "", False, False, 0, 100)
        assert sess.calls[1] == ("glob", "*.py", ".", 200)
        assert sess.calls[2] == ("ls", ".", 200)

    def test_grep_arguments_pass_through(self):
        sess = FakeLoopSession()
        call = d.ToolCall(
            name="grep",
            arguments={
                "pattern": "p", "path": "src", "glob": "*.py",
                "is_regex": "true", "case_sensitive": True,
                "context_lines": 2, "max_results": 9,
            },
        )
        d.dispatch_tool_call(call, mode="daily", sandbox=sess)
        assert sess.calls[0] == ("grep", "p", "src", "*.py", True, True, 2, 9)

    def test_new_tools_refused_without_bwrap(self):
        sess = FakeLoopSession(bwrap=False)
        r = d.dispatch_tool_call(
            d.ToolCall(name="grep", arguments={"pattern": "x"}), mode="daily", sandbox=sess
        )
        assert r.executed is False and r.reason == d.REASON_SANDBOX_UNAVAILABLE

    def test_new_tools_refused_without_active_session(self):
        sess = FakeLoopSession(active=False)
        r = d.dispatch_tool_call(
            d.ToolCall(name="ls", arguments={}), mode="daily", sandbox=sess
        )
        assert r.executed is False and r.reason == d.REASON_SANDBOX_UNAVAILABLE

    def test_as_bool_coercion(self):
        assert d._as_bool(None, True) is True
        assert d._as_bool("false", True) is False
        assert d._as_bool("1", False) is True
        assert d._as_bool("weird", False) is False
        assert d._as_bool(0, True) is False


# ---------------------------------------------------------------------------
# grep / glob / ls behaviour on a real tempdir workspace
# ---------------------------------------------------------------------------


class TestGrepBehaviour:
    def test_literal_case_insensitive_default(self, workspace):
        s = _session(str(workspace))
        out = s.grep("needle")
        lines = out.splitlines()
        assert lines[0] == "3 match(es) in 2 file(s) [2 file(s) skipped: binary or >1 MiB]"
        assert lines[1:] == [
            "b.txt:1: needle",
            "b.txt:2: NEEDLE again",
            "src/a.py:2: Beta needle here",
        ]

    def test_case_sensitive_literal(self, workspace):
        s = _session(str(workspace))
        out = s.grep("NEEDLE", case_sensitive=True)
        assert out.splitlines()[0].startswith("1 match(es) in 1 file(s)")
        assert "b.txt:2: NEEDLE again" in out

    def test_regex_with_glob_filter(self, workspace):
        s = _session(str(workspace))
        out = s.grep("ne+dle", glob="*.txt", is_regex=True)
        lines = out.splitlines()
        # big.txt matches the filter but exceeds 1 MiB, so it is counted.
        assert lines[0] == "2 match(es) in 1 file(s) [1 file(s) skipped: binary or >1 MiB]"
        assert all(line.startswith("b.txt:") for line in lines[1:])

    def test_invalid_regex_is_structured_error(self, workspace):
        s = _session(str(workspace))
        out = s.grep("(unclosed", is_regex=True)
        assert out.startswith("Error: invalid regex pattern:")

    def test_context_lines_indented_beneath(self, workspace):
        s = _session(str(workspace))
        out = s.grep("Beta", path="src/a.py", context_lines=1)
        assert out.splitlines() == [
            "1 match(es) in 1 file(s)",
            "src/a.py:2: Beta needle here",
            "  1| alpha",
            "  3| gamma",
        ]

    def test_max_results_cap_sets_truncated_flag(self, workspace):
        s = _session(str(workspace))
        out = s.grep("needle", max_results=1)
        header = out.splitlines()[0]
        assert header.startswith("1 match(es) in 1 file(s) [truncated]")

    def test_caps_are_clamped(self, workspace):
        s = _session(str(workspace))
        out = s.grep("needle", max_results=10_000, context_lines=99)
        assert out.splitlines()[0].startswith("3 match(es)")

    def test_escape_refused_with_established_shape(self, workspace):
        s = _session(str(workspace))
        out = s.grep("x", path="../outside")
        assert out.startswith("Error: Path rejected:")

    def test_missing_path(self, workspace):
        s = _session(str(workspace))
        assert s.grep("x", path="nope/").startswith("Error: Path not found:")

    def test_single_file_target(self, workspace):
        s = _session(str(workspace))
        out = s.grep("needle", path="b.txt")
        assert out.splitlines()[0] == "2 match(es) in 1 file(s)"

    def test_symlink_never_read(self, workspace):
        if not (workspace / "leak.txt").is_symlink():
            pytest.skip("symlinks unavailable on this filesystem")
        s = _session(str(workspace))
        out = s.grep("root", glob="leak.txt")
        assert out.splitlines()[0].startswith("0 match(es) in 0 file(s)")

    def test_no_active_session_raises_established_idiom(self, workspace):
        s = st.SandboxToolSession(sandbox_mgr=WorkspaceMgr(str(workspace)), tool_registry=None)
        with pytest.raises(RuntimeError, match="No active sandbox session"):
            s.grep("x")


class TestGlobBehaviour:
    def test_sorted_by_mtime_desc_then_name(self, workspace):
        s = _session(str(workspace))
        out = s.glob("**/*")
        assert out.splitlines() == [
            "4 file(s)",
            "big.txt",
            "bin.dat",
            "b.txt",
            "src/a.py",
        ]

    def test_equal_mtimes_fall_back_to_name_order(self, workspace):
        for name in ("b.txt", "bin.dat", "big.txt"):
            os.utime(workspace / name, (5000, 5000))
        s = _session(str(workspace))
        out = s.glob("*.*")
        assert out.splitlines() == ["3 file(s)", "b.txt", "big.txt", "bin.dat"]

    def test_cap_sets_truncated_flag(self, workspace):
        s = _session(str(workspace))
        out = s.glob("**/*", max_results=2)
        lines = out.splitlines()
        assert lines[0] == "2 file(s) [truncated]"
        assert len(lines) == 3

    def test_pattern_scoped_to_subdir(self, workspace):
        s = _session(str(workspace))
        out = s.glob("*.py", path="src")
        # Relpaths are workspace-relative everywhere, matching grep's lines.
        assert out.splitlines() == ["1 file(s)", "src/a.py"]

    def test_symlinks_excluded(self, workspace):
        if not (workspace / "leak.txt").is_symlink():
            pytest.skip("symlinks unavailable on this filesystem")
        s = _session(str(workspace))
        assert "leak.txt" not in s.glob("**/*")

    def test_escape_refused(self, workspace):
        s = _session(str(workspace))
        assert s.glob("*", path="../").startswith("Error: Path rejected:")

    def test_empty_pattern_error(self, workspace):
        s = _session(str(workspace))
        assert s.glob("").startswith("Error: glob requires")


class TestLsBehaviour:
    def test_dirs_first_then_files_name_sorted(self, workspace):
        s = _session(str(workspace))
        out = s.ls(".")
        lines = out.splitlines()
        assert lines[0] == "dir 0 src"
        assert lines[1] == f"file {os.path.getsize(workspace / 'b.txt')} b.txt"
        names = [line.split(" ", 2)[2] for line in lines]
        assert names == ["src", "b.txt", "big.txt", "bin.dat"]

    def test_truncation_line(self, workspace):
        s = _session(str(workspace))
        out = s.ls(".", max_entries=2)
        lines = out.splitlines()
        assert lines[-1] == "[truncated at 2 entries]"
        assert len(lines) == 3

    def test_empty_directory(self, workspace):
        (workspace / "void").mkdir()
        s = _session(str(workspace))
        assert s.ls("void") == "0 entries"

    def test_file_target_is_error(self, workspace):
        s = _session(str(workspace))
        assert s.ls("b.txt").startswith("Error: Path not found:")

    def test_symlink_skipped(self, workspace):
        if not (workspace / "leak.txt").is_symlink():
            pytest.skip("symlinks unavailable on this filesystem")
        s = _session(str(workspace))
        assert "leak.txt" not in s.ls(".")

    def test_escape_refused(self, workspace):
        s = _session(str(workspace))
        assert s.ls("/etc").startswith("Error: Path rejected:")


# ---------------------------------------------------------------------------
# todo: handler semantics and the loop binding
# ---------------------------------------------------------------------------


class TestTodoHandler:
    def test_output_shape_and_counts(self):
        h = t.make_todo_handler()
        out = h({"todos": [
            {"content": "write tests", "status": "in_progress", "priority": "high"},
            {"content": "ship"},
            {"content": "old idea", "status": "cancelled", "priority": "low"},
            {"content": "done part", "status": "completed"},
        ]})
        lines = out.splitlines()
        assert lines[0] == "Todo list updated (4 items, 1 completed)"
        assert lines[1] == "1. [in_progress] write tests (high)"
        assert lines[2] == "2. [pending] ship (medium)"

    def test_each_call_replaces_the_list(self):
        h = t.make_todo_handler()
        h({"todos": [{"content": "a"}, {"content": "b"}]})
        h({"todos": [{"content": "only"}]})
        assert [it["content"] for it in h.state] == ["only"]

    def test_validation_never_raises(self):
        h = t.make_todo_handler()
        assert h({"todos": "nope"}).startswith("todo requires 'todos'")
        assert "must be an object" in h({"todos": ["raw string"]})
        assert "non-empty 'content'" in h({"todos": [{"content": "  "}]})
        assert "'status' must be one of" in h({"todos": [{"content": "x", "status": "zzz"}]})
        assert "'priority' must be one of" in h({"todos": [{"content": "x", "priority": "zzz"}]})
        assert h.state == []  # invalid calls leave the state untouched

    def test_on_update_payload(self):
        seen = []
        h = t.make_todo_handler(on_update=lambda p: seen.append(p))
        h({"todos": [{"content": "x", "status": "completed"}, {"content": "y"}]})
        assert seen and seen[0]["total"] == 2 and seen[0]["completed"] == 1
        assert seen[0]["todos"][0]["content"] == "x"

    def test_on_update_exception_is_swallowed(self):
        def boom(payload):
            raise RuntimeError("observer down")

        h = t.make_todo_handler(on_update=boom)
        out = h({"todos": [{"content": "x"}]})
        assert out.startswith("Todo list updated")


class TestTodoInLoop:
    def test_todo_advertised_injects_handler_and_emits_event(self):
        events = []
        res = L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "x", "tool_calls": _native("todo", {"todos": [
                    {"content": "step1"}, {"content": "step2", "status": "completed"},
                ]})},
                {"content": "done", "tool_calls": None},
            ]),
            sandbox=None,
            mode="daily",
            tools=[{"function": {"name": "todo"}}],
            on_event=lambda e: events.append(e),
        )
        todo_events = [e for e in events if e.kind == "todo_updated"]
        assert len(todo_events) == 1
        assert todo_events[0].round == 1
        assert todo_events[0].data == {
            "todos": [
                {"content": "step1", "status": "pending", "priority": "medium"},
                {"content": "step2", "status": "completed", "priority": "medium"},
            ],
            "total": 2,
            "completed": 1,
        }
        result = [r for r in res.tool_results if r.tool_name == "todo"][0]
        assert result.executed and result.observation.startswith(
            "Todo list updated (2 items, 1 completed)"
        )

    def test_run_without_todo_is_unchanged(self):
        events = []
        injected = {}
        res = L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "x", "tool_calls": _native("bash", {"command": "ls"})},
                {"content": "done", "tool_calls": None},
            ]),
            sandbox=FakeLoopSession(),
            mode="daily",
            tool_handlers=injected,
            on_event=lambda e: events.append(e.kind),
        )
        assert res.stop_reason == "done"
        assert "todo_updated" not in events
        assert injected == {}  # the caller's mapping is never mutated

    def test_caller_injected_todo_handler_is_kept_and_bound(self):
        events = []
        mine = t.make_todo_handler()
        L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "x", "tool_calls": _native("todo", {"todos": [{"content": "z"}]})},
                {"content": "done", "tool_calls": None},
            ]),
            mode="daily",
            tool_handlers={"todo": mine},
            on_event=lambda e: events.append(e.kind),
        )
        assert [it["content"] for it in mine.state] == ["z"]
        assert "todo_updated" in events

    def test_todo_works_in_bulbe_without_approval(self):
        approvals = []

        def approve(conv, tool, args):
            approvals.append(tool)
            return True

        res = L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "x", "tool_calls": _native("todo", {"todos": [{"content": "p"}]})},
                {"content": "done", "tool_calls": None},
            ]),
            mode="bulbe",
            tools=[{"function": {"name": "todo"}}],
            approval_fn=approve,
        )
        result = [r for r in res.tool_results if r.tool_name == "todo"][0]
        assert result.executed and approvals == []


# ---------------------------------------------------------------------------
# task: the bounded subagent
# ---------------------------------------------------------------------------


class TestTaskBounding:
    def test_child_runs_and_reports_bounds(self):
        sess = FakeLoopSession()
        events = []
        res = L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "p", "tool_calls": _native(
                    "task", {"description": "probe", "prompt": "do it", "max_rounds": 3})},
                {"content": "c1", "tool_calls": _native("bash", {"command": "echo hi"})},
                {"content": "child done", "tool_calls": None},
                {"content": "parent done", "tool_calls": None},
            ]),
            sandbox=sess,
            mode="daily",
            max_rounds=10,
            on_event=lambda e: events.append(e),
        )
        task_result = [r for r in res.tool_results if r.tool_name == "task"][0]
        assert task_result.executed and task_result.reason == d.REASON_EXECUTED
        assert "child done" in task_result.observation
        assert task_result.observation.endswith("task used 2 rounds of 3")
        assert ("bash", "echo hi", 30) in sess.calls
        marked = [e for e in events if e.data.get("task") == "probe"]
        assert marked, "child events must carry the task marker"
        assert res.final_text == "parent done"

    def test_child_rounds_are_debited_from_parent_budget(self):
        # Parent cap 5. Round 1 launches a child that burns its 2-round cap;
        # the debit leaves the parent 3 total rounds, so two more tool-call
        # rounds end the run at max_rounds with 3 parent + 2 child = 5.
        script = [
            {"content": "p1", "tool_calls": _native(
                "task", {"description": "d", "prompt": "go", "max_rounds": 2})},
            {"content": "c1", "tool_calls": _native("bash", {"command": "a"})},
            {"content": "c2", "tool_calls": _native("bash", {"command": "b"})},
            {"content": "p2", "tool_calls": _native("bash", {"command": "c"})},
            {"content": "p3", "tool_calls": _native("bash", {"command": "e"})},
        ]
        client = ScriptedClient(script)
        res = L.run("q", model_client=client, sandbox=FakeLoopSession(),
                    mode="daily", max_rounds=5)
        task_result = [r for r in res.tool_results if r.tool_name == "task"][0]
        assert task_result.observation.endswith("task used 2 rounds of 2")
        assert res.stop_reason == "max_rounds"
        assert res.rounds == 3
        assert client.calls == 5

    def test_requested_rounds_capped_by_task_child_cap(self):
        assert L.TASK_CHILD_CAP == 6
        script = [{"content": "p", "tool_calls": _native(
            "task", {"description": "d", "prompt": "go", "max_rounds": 50})}]
        script += [
            {"content": f"c{i}", "tool_calls": _native("bash", {"command": "x"})}
            for i in range(1, 7)
        ]
        script += [{"content": "parent done", "tool_calls": None}]
        res = L.run("q", model_client=ScriptedClient(script),
                    sandbox=FakeLoopSession(), mode="daily", max_rounds=20)
        task_result = [r for r in res.tool_results if r.tool_name == "task"][0]
        assert task_result.observation.endswith("task used 6 rounds of 6")

    def test_budget_exhaustion_refuses_with_reason(self):
        res = L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "p", "tool_calls": _native(
                    "task", {"description": "d", "prompt": "go"})},
                {"content": "done", "tool_calls": None},
            ]),
            sandbox=FakeLoopSession(),
            mode="daily",
            max_rounds=2,  # remaining - 1 == 0: no room for a child
        )
        task_result = [r for r in res.tool_results if r.tool_name == "task"][0]
        assert task_result.executed is False
        assert task_result.reason == "task_budget_exhausted"
        assert "insufficient round budget" in task_result.observation

    def test_empty_prompt_is_structured_error(self):
        res = L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "p", "tool_calls": _native("task", {"description": "d", "prompt": " "})},
                {"content": "done", "tool_calls": None},
            ]),
            sandbox=FakeLoopSession(),
            mode="daily",
            max_rounds=10,
        )
        task_result = [r for r in res.tool_results if r.tool_name == "task"][0]
        assert task_result.executed is False and task_result.reason == d.REASON_ERROR

    def test_depth_one_child_cannot_start_tasks(self):
        events = []
        res = L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "p", "tool_calls": _native(
                    "task", {"description": "outer", "prompt": "go", "max_rounds": 3})},
                {"content": "c1", "tool_calls": _native(
                    "task", {"description": "inner", "prompt": "nested"})},
                {"content": "child done", "tool_calls": None},
                {"content": "parent done", "tool_calls": None},
            ]),
            sandbox=FakeLoopSession(),
            mode="daily",
            max_rounds=10,
            on_event=lambda e: events.append(e),
        )
        nested = [
            e for e in events
            if e.kind == "tool_result"
            and e.data.get("task") == "outer"
            and e.data.get("tool_name") == "task"
        ]
        assert nested and nested[0].data["reason"] == d.REASON_NO_EXECUTOR
        outer = [r for r in res.tool_results if r.tool_name == "task"]
        assert len(outer) == 1  # one parent-level task only; no recursion

    def test_child_registry_excludes_parent_handlers(self):
        called = []
        res = L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "p", "tool_calls": _native(
                    "task", {"description": "d", "prompt": "go", "max_rounds": 2})},
                {"content": "c1", "tool_calls": _native("web_search", {"query": "x"})},
                {"content": "child done", "tool_calls": None},
                {"content": "parent done", "tool_calls": None},
            ]),
            sandbox=FakeLoopSession(),
            mode="daily",
            max_rounds=10,
            tool_handlers={"web_search": lambda a: called.append(a) or "hit"},
        )
        assert called == []  # the child never sees the parent's handlers
        assert [r for r in res.tool_results if r.tool_name == "task"][0].executed

    def test_bulbe_child_inherits_per_call_approval(self):
        approvals = []

        def approve(conv, tool, args):
            approvals.append(tool)
            return True

        sess = FakeLoopSession()
        res = L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "p", "tool_calls": _native(
                    "task", {"description": "d", "prompt": "go", "max_rounds": 2})},
                {"content": "c1", "tool_calls": _native("bash", {"command": "safe"})},
                {"content": "child done", "tool_calls": None},
                {"content": "parent done", "tool_calls": None},
            ]),
            sandbox=sess,
            mode="bulbe",
            max_rounds=10,
            approval_fn=approve,
        )
        assert approvals == ["bash"]  # the child's sandbox call rode the gate
        assert ("bash", "safe", 30) in sess.calls
        assert [r for r in res.tool_results if r.tool_name == "task"][0].executed

    def test_child_surface_is_exactly_the_sandbox_seven(self):
        native, section = L._child_task_surface()
        names = {entry["function"]["name"] for entry in native}
        assert names == set(al.SANDBOX_TOOL_NAMES)
        assert "task" not in section.split("\n")[0]

    def test_no_task_round_uses_dispatch_round_path(self, monkeypatch):
        seen = {"rounds": 0}
        original = d.dispatch_round

        def spy(*args, **kwargs):
            seen["rounds"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(d, "dispatch_round", spy)
        L.run(
            "q",
            model_client=ScriptedClient([
                {"content": "x", "tool_calls": _native("bash", {"command": "ls"})},
                {"content": "done", "tool_calls": None},
            ]),
            sandbox=FakeLoopSession(),
            mode="daily",
        )
        assert seen["rounds"] == 2

    def test_verifier_caps_untouched(self):
        assert L._VERIFIER_MAX_ROUNDS == 2
        assert L.MAX_AGENT_ROUNDS == 20


# ---------------------------------------------------------------------------
# Diagnostics-after-write gating
# ---------------------------------------------------------------------------


def _diag_responder_with_findings(command):
    if command.startswith("command -v ruff"):
        return CmdResult(0, "/usr/bin/ruff")
    if command.startswith("ruff check"):
        return CmdResult(1, "f.py:1:5: E999 SyntaxError\nf.py:2:1: F821 undefined name")
    return CmdResult(1)


class TestDiagnosticsGating:
    def test_bwrap_backend_appends_findings_block(self, tmp_path):
        mgr = DiagMgr(str(tmp_path), _diag_responder_with_findings)
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        out = s.create_file("f.py", "x = (\n")
        assert out.startswith("File created: f.py (6 bytes)")
        assert "\n\n[diagnostics] 2 finding(s):\n" in out
        assert "E999" in out and "F821" in out

    def test_probe_runs_once_per_session(self, tmp_path):
        mgr = DiagMgr(str(tmp_path), _diag_responder_with_findings)
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        s.create_file("f.py", "x = (\n")
        s.str_replace("f.py", "x = (", "y = (")
        probes = [c for c in mgr.commands if c.startswith("command -v")]
        assert probes == ["command -v ruff"]
        runs = [c for c in mgr.commands if c.startswith("ruff check --quiet -- ")]
        assert len(runs) == 2

    def test_suffix_map_excludes_other_extensions(self, tmp_path):
        mgr = DiagMgr(str(tmp_path), _diag_responder_with_findings)
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        out = s.create_file("notes.txt", "plain")
        assert out == "File created: notes.txt (5 bytes)"
        assert mgr.commands == []

    def test_tempdir_backend_is_byte_identical(self, tmp_path):
        def explode(command):
            raise AssertionError("diagnostics must never execute off-bwrap")

        mgr = DiagMgr(str(tmp_path), explode)
        s = _session(str(tmp_path), backend="tempdir", mgr=mgr)
        out = s.create_file("f.py", "x = (\n")
        assert out == "File created: f.py (6 bytes)"
        assert mgr.commands == []

    def test_clean_bwrap_write_is_byte_identical(self, tmp_path):
        mgr = DiagMgr(str(tmp_path), lambda c: CmdResult(0))
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        out = s.create_file("ok.py", "x = 1\n")
        assert out == "File created: ok.py (6 bytes)"

    def test_failed_write_never_triggers_diagnostics(self, tmp_path):
        mgr = DiagMgr(str(tmp_path), _diag_responder_with_findings)
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        out = s.str_replace("absent.py", "a", "b")
        assert out.startswith("Error")
        assert mgr.commands == []

    def test_disabled_config_means_no_probe_and_no_run(self, tmp_path, monkeypatch):
        disabled = dict(st.DIAGNOSTICS_DEFAULTS)
        disabled["enabled"] = False
        monkeypatch.setattr(st, "load_diagnostics_config", lambda: disabled)
        mgr = DiagMgr(str(tmp_path), _diag_responder_with_findings)
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        out = s.create_file("f.py", "x = (\n")
        assert out == "File created: f.py (6 bytes)"
        assert mgr.commands == []

    def test_ladder_falls_through_to_pyflakes(self, tmp_path):
        def responder(command):
            if command.startswith("command -v ruff"):
                return CmdResult(1)
            if command.startswith("command -v pyflakes"):
                return CmdResult(0, "/usr/bin/pyflakes")
            if command.startswith("pyflakes "):
                return CmdResult(1, "f.py:1: invalid syntax")
            return CmdResult(1)

        mgr = DiagMgr(str(tmp_path), responder)
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        out = s.create_file("f.py", "x = (\n")
        assert "[diagnostics] 1 finding(s):" in out and "invalid syntax" in out

    def test_findings_cap_and_truncation_marker(self, tmp_path):
        many = "\n".join(f"f.py:{i}:1: E{i} problem" for i in range(1, 41))

        def responder(command):
            if command.startswith("command -v ruff"):
                return CmdResult(0)
            return CmdResult(1, many)

        mgr = DiagMgr(str(tmp_path), responder)
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        out = s.create_file("f.py", "x = (\n")
        assert "[diagnostics] 40 finding(s):" in out
        body = out.split("finding(s):\n", 1)[1].splitlines()
        assert len(body) == 26 and body[-1] == "[diagnostics truncated]"

    def test_execution_failure_is_silent_skip(self, tmp_path):
        def responder(command):
            if command.startswith("command -v ruff"):
                return CmdResult(0)
            raise ValueError("session gone")

        mgr = DiagMgr(str(tmp_path), responder)
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        out = s.create_file("f.py", "x = (\n")
        assert out == "File created: f.py (6 bytes)"

    def test_svelte_uses_host_side_tag_balance(self, tmp_path):
        def no_exec(command):
            if command.startswith("command -v"):
                return CmdResult(0)
            raise AssertionError("the Svelte path must not execute commands")

        mgr = DiagMgr(str(tmp_path), no_exec)
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        out = s.create_file("App.svelte", "<div><span>x</span>{#if a}<p>y</p>")
        assert "[diagnostics] 2 finding(s):" in out
        assert "unbalanced tag <div>" in out
        assert "unbalanced Svelte block {#if}" in out
        assert all(not c.startswith(("ruff", "pyflakes", "python3")) for c in mgr.commands)

    def test_diag_cache_reset_on_lifecycle(self, tmp_path):
        mgr = DiagMgr(str(tmp_path), _diag_responder_with_findings)
        s = _session(str(tmp_path), backend="bwrap", mgr=mgr)
        s.create_file("f.py", "x = (\n")
        assert s._diag_probed is True
        s._session = None
        s._session_id = None
        s._reset_diag_cache()
        assert s._diag_probed is False and s._diag_tool is None


class TestSvelteTagFindings:
    def test_balanced_component_is_clean(self):
        text = (
            "<script>let a = 1 < 2;</script>\n"
            "<div class=\"x\"><br><img src=\"y\"/>\n"
            "{#if a}<p>{ a < b }</p>{:else}<p>no</p>{/if}\n"
            "{#each items as it}<li>{it}</li>{/each}\n"
            "</div>\n<style>.x { color: var(--oo-fg); }</style>"
        )
        assert st.svelte_tag_findings(text) == []

    def test_unbalanced_tag_and_block(self):
        findings = st.svelte_tag_findings("<div><span>x</span>{#if a}")
        assert "unbalanced tag <div>: 1 opened, 0 closed" in findings
        assert "unbalanced Svelte block {#if}: 1 opened, 0 closed" in findings

    def test_comments_and_script_content_ignored(self):
        text = "<!-- <div> --><script>if (a < b) { run('</div>') }</script><main></main>"
        assert st.svelte_tag_findings(text) == []

    def test_load_diagnostics_config_defaults(self):
        cfg_block = st.load_diagnostics_config()
        assert cfg_block["enabled"] is True
        assert cfg_block["tools"] == ["ruff", "pyflakes", "py_compile"]
        assert cfg_block["timeout_s"] == 10
        assert cfg_block["max_block_bytes"] == 4096
        assert cfg_block["max_findings"] == 25


# ---------------------------------------------------------------------------
# Supersession reassertions (deselect-plus-reassert; originals never edited)
# ---------------------------------------------------------------------------


class TestSupersessionReassertions:
    # 1. s177 test_seven_schemas
    def test_twelve_schemas(self):
        assert len(t.ALL_SCHEMAS) == 12

    # 2. s177 test_manage_skills_is_third_non_sandbox
    def test_non_sandbox_set_gains_todo_and_task(self):
        non = {s.name for s in t.ALL_SCHEMAS if not s.sandboxed}
        assert non == {
            t.TOOL_WEB_SEARCH, t.TOOL_MANAGE_MEMORY, t.TOOL_MANAGE_SKILLS,
            t.TOOL_TODO, t.TOOL_TASK,
        }

    # 3. s177 test_daily_includes_manage_skills
    def test_daily_handler_set_gains_todo(self):
        ts = t.build_tool_set("daily")
        assert t.TOOL_MANAGE_SKILLS in ts.names
        assert t.TOOL_MANAGE_SKILLS in ts.tool_handlers
        assert set(ts.tool_handlers) == {
            t.TOOL_WEB_SEARCH, t.TOOL_MANAGE_MEMORY, t.TOOL_MANAGE_SKILLS, t.TOOL_TODO,
        }

    # 4. s177 test_bulbe_excludes_manage_skills
    def test_bulbe_excludes_manage_skills_and_gains_session_tools(self):
        ts = t.build_tool_set("bulbe")
        assert t.TOOL_MANAGE_SKILLS not in ts.names
        assert set(ts.names) == set(al.SANDBOX_TOOL_NAMES) | {t.TOOL_TODO, t.TOOL_TASK}

    # 5. s176 test_bulbe_has_no_handlers
    def test_bulbe_first_handler_is_todo(self):
        ts = t.build_tool_set("bulbe")
        assert set(ts.tool_handlers) == {t.TOOL_TODO}

    # 6. s176 test_bulbe_exposes_sandbox_only
    def test_bulbe_exposes_sandbox_plus_session_tools(self):
        ts = t.build_tool_set("bulbe")
        assert set(ts.names) == set(al.SANDBOX_TOOL_NAMES) | {"todo", "task"}

    # 7. s176 test_unknown_mode_is_fail_secure_bulbe
    def test_unknown_mode_gets_bulbe_set(self):
        ts = t.build_tool_set("nonsense")
        assert set(ts.names) == set(al.SANDBOX_TOOL_NAMES) | {"todo", "task"}
        assert set(ts.names) == set(al.BULBE_ALLOWLIST)

    # 8. s176 test_sandbox_argument_names_cover_dispatch_lambdas
    def test_sandbox_argument_names_cover_all_seven_lambdas(self):
        expected = {
            "bash": {"command", "timeout"},
            "view": {"path", "start_line", "end_line"},
            "create_file": {"path", "content"},
            "str_replace": {"path", "old_str", "new_str"},
            "grep": {
                "pattern", "path", "glob", "is_regex",
                "case_sensitive", "context_lines", "max_results",
            },
            "glob": {"pattern", "path", "max_results"},
            "ls": {"path", "max_entries"},
        }
        covered = set()
        for schema in t.ALL_SCHEMAS:
            if schema.sandboxed:
                got = {p.name for p in schema.parameters}
                assert got == expected[schema.name]
                covered.add(schema.name)
        assert covered == set(expected)

    # 9. s222 test_all_schemas_is_seven_today
    def test_all_schemas_is_twelve_today(self):
        assert len(t.ALL_SCHEMAS) == 12

    # 10. s222 test_handler_names_are_the_three
    def test_handler_names_are_the_four(self):
        assert t.HANDLER_TOOL_NAMES == frozenset(
            {"web_search", "manage_memory", "manage_skills", "todo"}
        )

    # 11. s222 test_frozensets_exact
    def test_frozensets_exact_post_s228(self):
        assert al.SANDBOX_TOOL_NAMES == frozenset(
            {"bash", "view", "create_file", "str_replace", "grep", "glob", "ls"}
        )
        assert al.NETWORK_TOOLS == frozenset({"web_search"})
        assert al.STATE_MUTATION_TOOLS == frozenset({"manage_memory", "manage_skills"})

    # 12. s175 test_sandbox_tools
    def test_sandbox_tools_are_the_seven(self):
        assert al.SANDBOX_TOOL_NAMES == frozenset(
            {"bash", "view", "create_file", "str_replace", "grep", "glob", "ls"}
        )

    # 13. s175 test_bulbe_equals_sandbox_tools
    def test_bulbe_equals_sandbox_plus_session_tools(self):
        assert al.BULBE_ALLOWLIST == frozenset(
            al.SANDBOX_TOOL_NAMES | al.SESSION_STATE_TOOLS | al.SUBAGENT_TOOLS
        )

    # 14. s176_config test_bulbe_equals_sandbox_set
    def test_config_bulbe_list_equals_sandbox_plus_session_tools(self):
        c = cfg.load_config()
        assert set(c.bulbe_tools) == set(
            al.SANDBOX_TOOL_NAMES | al.SESSION_STATE_TOOLS | al.SUBAGENT_TOOLS
        )
        assert set(c.daily_tools) == set(t.build_tool_set("daily").names)


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class TestFacade:
    def test_init_source_exports_s228_names(self):
        src = (AGENT / "__init__.py").read_text(encoding="utf-8")
        for needle in (
            "TASK_CHILD_CAP",
            "SESSION_STATE_TOOLS",
            "SUBAGENT_TOOLS",
            "HANDLER_TOOL_NAMES",
            "make_todo_handler",
        ):
            assert needle in src, needle

    def test_config_yaml_lists_grown(self):
        c = cfg.load_config()
        assert len(c.daily_tools) == 12
        assert len(c.bulbe_tools) == 9
