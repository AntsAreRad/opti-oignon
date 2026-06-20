#!/usr/bin/env python3
"""
S230 -- AGT Lot 3: the micro-task eval harness (AGT_SPEC Section 7).

Container-provable surface covered here:
- Module conventions and the facade (checkpoint_before_apply hardcoded,
  FEATURE_AVAILABLE sentinels, exports).
- TaskSpec defaults and the suite loader's structural validation; the
  micro suite's 12 tasks; the defensive pin that every fixture and every
  check stays clean against the REAL sandbox CommandValidator (fixtures
  against the write-then-execute content patterns, checks executed
  unblocked on the real tempdir backend).
- The results store: schema, lifecycle, task rows, summary math, history,
  the separate-database rule, and the no-f-string-SQL pin.
- The pure classification helpers: the seven failure classes
  deterministic, the transcript spill-ref scan, the diagnostics scan, the
  host fingerprint shape.
- End-to-end on the REAL tempdir backend through the dispatch facade (see
  _BwrapFacadeManager below): fixture materialization through the
  path-confined create_file seam, REAL tool dispatch, REAL check
  execution, fresh-session-per-task destruction, the busy guard, cancel
  semantics (the in-flight row dropped), the stubbed state/egress
  handlers, the blocked-check-is-error rule, max_rounds containment, and
  the spill-ref capture from a REAL oversized observation.
- The three admission provenance paths with a sys.modules-stubbed
  governor (absent / admitted / refused, including the estop-shaped
  refusal), the 6.6 ticket composition (the REAL thread-local scope seen
  from inside a running tool handler), invalidate_on_load, and the
  evict-between-models seam.
- The API router: the five endpoints, 409-busy, 404s, 422s, the auth
  parity dependency.
- The CLI: smoke through the injectable runner seam, argument errors,
  exit codes.
- Deliverable pins: the ATREST row, the two scripts, the guarded app
  mount, AST validity of every new file, version holds.

The dispatch gate (agent/dispatch.py sandbox_ready) is the S73/S74
posture: sandbox tools act only when the session's manager reports bwrap
available -- deliberately no tempdir path. The container has no bwrap, so
these tests run the REAL downstream path (creation, execution, checks,
destroy on the real tempdir backend) behind a test-only facade that
raises ONLY the one bit the gate reads. This is the honest evolution of
the s175/s228 FakeLoopSession(bwrap=True): the fake session replaced by
the real one plus one bit. Nothing ships: on a host without bwrap the
eval harness inherits the product posture and refuses the tools.

Host-assured (named, never simulated in the container): the
model-in-the-loop eval runs themselves (a live Ollama endpoint and the
local fleet), eviction effectiveness between models on real VRAM, real
bwrap spill and diagnostics behaviour ([diagnostics] capture stays a
pure-scan test here), and the opencode side-by-side baseline script
execution (pinned for shape only).
"""

import ast
import json
import re
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opti_oignon import sandbox_manager as sm  # noqa: E402
from opti_oignon.agent_eval import (  # noqa: E402
    DEFAULT_MAX_ROUNDS,
    DEFAULT_REQUESTED_CTX,
    DEFAULT_TIMEOUT_S,
    FAILURE_CLASSES,
    EvalResultsStore,
    EvalRunner,
    TaskSpec,
    get_eval_runner,
    load_suite,
    max_requested_ctx,
    reset_eval_runner,
    resolve_suite_path,
)
from opti_oignon.agent_eval import __main__ as eval_cli  # noqa: E402
from opti_oignon.agent_eval import runner as runner_mod  # noqa: E402
from opti_oignon.agent_eval import store as store_mod  # noqa: E402
from opti_oignon.agent_eval import tasks as tasks_mod  # noqa: E402

import opti_oignon.resource_governor as real_governor  # noqa: E402

NEW_PY_FILES = [
    ROOT / "opti_oignon" / "agent_eval" / "__init__.py",
    ROOT / "opti_oignon" / "agent_eval" / "__main__.py",
    ROOT / "opti_oignon" / "agent_eval" / "tasks.py",
    ROOT / "opti_oignon" / "agent_eval" / "store.py",
    ROOT / "opti_oignon" / "agent_eval" / "runner.py",
    ROOT / "opti_oignon" / "api" / "routes_agent_eval.py",
]


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------


def _make_manager(tmp_path):
    """The s229 tempdir-manager idiom: real manager, degraded backend."""
    return sm.SandboxManager(
        config=sm.SandboxConfig(
            workspace_base=str(tmp_path / "sbx"),
            audit_db_path=str(tmp_path / "audit.db"),
            isolation_backend="tempdir",
            require_degraded_confirmation=False,
            strict_mode=False,
            idle_ttl_seconds=0,
        )
    )


class _BwrapFacadeManager:
    """Test-only facade over the REAL tempdir manager.

    The dispatch gate (sandbox_ready) reads exactly one bit,
    ``manager.bwrap_available``; this facade raises that bit and delegates
    EVERYTHING else to the real manager, so creation, execution, checks
    and destruction all run the real tempdir code. Diagnostics stay
    honestly off: their gate reads the session's isolation backend, which
    remains "tempdir".
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    @property
    def bwrap_available(self):
        return True


class ScriptedClient:
    """A fake model client replaying scripted Ollama-shaped rounds."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def stream(self, messages, tools=None):
        self.calls += 1
        step = (
            self.script.pop(0)
            if self.script
            else {"content": "done", "tool_calls": None}
        )
        if isinstance(step, Exception):
            raise step
        sleep_s = step.get("_sleep")
        if sleep_s:
            event = step.get("_started")
            if event is not None:
                event.set()
            time.sleep(sleep_s)
        return {"message": {k: v for k, v in step.items() if not k.startswith("_")}}


def _native(name, args):
    return [{"function": {"name": name, "arguments": args}}]


def _final(text="done"):
    return {"content": text, "tool_calls": None}


def _tool(name, args):
    return {"content": "", "tool_calls": _native(name, args)}


def _task(task_id="t-test", prompt="do the task", checks=None, **over):
    return TaskSpec(
        id=task_id,
        title=task_id,
        prompt=prompt,
        checks=checks or ["test -d ."],
        **over,
    )


def _runner_for(tmp_path, mgr, scripts_by_call=None, **over):
    """An EvalRunner wired for the container: facade manager, scripted
    client factory (one fresh script copy per factory call)."""
    scripts = scripts_by_call or [[_final()]]
    state = {"calls": 0, "clients": []}

    def factory(model):
        index = min(state["calls"], len(scripts) - 1)
        state["calls"] += 1
        client = ScriptedClient(list(scripts[index]))
        state["clients"].append(client)
        return client

    runner = EvalRunner(
        db_path=str(tmp_path / "eval.db"),
        sandbox_manager=mgr,
        client_factory=over.pop("client_factory", factory),
        **over,
    )
    return runner, state


@pytest.fixture
def tempdir_world(tmp_path):
    real = _make_manager(tmp_path)
    facade = _BwrapFacadeManager(real)
    yield tmp_path, real, facade


@pytest.fixture
def absent_governor(monkeypatch):
    """Deterministic 'absent' provenance: the resolver sees a feature-off
    module regardless of the container's real governor state."""
    stub = types.SimpleNamespace(FEATURE_AVAILABLE=False)
    monkeypatch.setitem(sys.modules, "opti_oignon.resource_governor", stub)
    yield stub


def _decision(**over):
    base = dict(
        admitted=True,
        reason="test",
        num_ctx=4096,
        requested_ctx=4096,
        action="admit",
        load_expected=False,
        is_estop=False,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


class _StubGovernor:
    def __init__(self, decision):
        self.decision = decision
        self.admit_calls = []
        self.load_invalidations = []
        self.evict_invalidations = []
        self.config = types.SimpleNamespace(enabled=True)

    def admit(self, model, requested_ctx=None, caller="chat", **kwargs):
        self.admit_calls.append(
            {"model": model, "requested_ctx": requested_ctx, "caller": caller}
        )
        return self.decision

    def invalidate_on_load(self, model, num_ctx=None):
        self.load_invalidations.append((model, num_ctx))

    def invalidate_on_evict(self, model):
        self.evict_invalidations.append(model)


@pytest.fixture
def stub_governor_factory(monkeypatch):
    def _install(decision):
        stub = _StubGovernor(decision)
        module = types.SimpleNamespace(
            FEATURE_AVAILABLE=True,
            get_resource_governor=lambda: stub,
        )
        monkeypatch.setitem(
            sys.modules, "opti_oignon.resource_governor", module
        )
        return stub

    return _install


# ---------------------------------------------------------------------------
# Module conventions and facade
# ---------------------------------------------------------------------------


class TestModuleConventions:
    def test_checkpoint_hardcoded_everywhere(self):
        import opti_oignon.agent_eval as facade
        import opti_oignon.api.routes_agent_eval as rae

        for module in (facade, tasks_mod, store_mod, runner_mod, eval_cli, rae):
            assert getattr(module, "checkpoint_before_apply", None) is True

    def test_feature_sentinels(self):
        assert tasks_mod.FEATURE_AVAILABLE is True
        assert store_mod.FEATURE_AVAILABLE is True
        assert isinstance(runner_mod.FEATURE_AVAILABLE, bool)
        assert runner_mod.FEATURE_AVAILABLE is True

    def test_facade_exports(self):
        import opti_oignon.agent_eval as facade

        for name in (
            "EvalRunner",
            "EvalResultsStore",
            "TaskSpec",
            "load_suite",
            "FAILURE_CLASSES",
            "get_eval_runner",
            "reset_eval_runner",
        ):
            assert hasattr(facade, name)

    def test_failure_classes_taxonomy(self):
        assert FAILURE_CLASSES == (
            "none",
            "test_fail",
            "timeout",
            "doom_loop",
            "refusal",
            "not_admitted",
            "error",
        )


# ---------------------------------------------------------------------------
# TaskSpec and the loader
# ---------------------------------------------------------------------------


class TestTaskSpecAndLoader:
    def test_defaults(self):
        spec = TaskSpec(id="a", title="a", prompt="p", checks=["true"])
        assert spec.timeout_s == DEFAULT_TIMEOUT_S == 180.0
        assert spec.max_rounds == DEFAULT_MAX_ROUNDS == 10
        assert spec.requested_ctx == DEFAULT_REQUESTED_CTX == 8192

    def test_micro_suite_twelve_tasks(self):
        tasks = load_suite("micro")
        assert [t.id for t in tasks] == [
            "t01-create-file",
            "t02-create-nested",
            "t03-edit-config",
            "t04-bump-version",
            "t05-grep-constant",
            "t06-glob-draft",
            "t07-grep-distractor",
            "t08-rename-function",
            "t09-add-function",
            "t10-fix-multiply",
            "t11-fix-count-words",
            "t12-long-output",
        ]
        assert len({t.id for t in tasks}) == 12
        for task in tasks:
            assert task.checks
            assert task.prompt.strip()
            assert task.requested_ctx == 8192

    def test_resolve_name_and_path(self, tmp_path):
        named = resolve_suite_path("micro")
        assert named.name == "micro.yaml"
        suite = tmp_path / "mini.yaml"
        suite.write_text(
            "suite: mini\ntasks:\n  - id: a\n    prompt: p\n"
            "    checks: ['true']\n",
            encoding="utf-8",
        )
        assert resolve_suite_path(str(suite)) == suite

    def test_unknown_suite_raises(self):
        with pytest.raises(FileNotFoundError):
            load_suite("no-such-suite")

    @pytest.mark.parametrize(
        "body",
        [
            "tasks:\n  - prompt: p\n    checks: ['true']\n",  # missing id
            (
                "tasks:\n  - id: a\n    prompt: p\n    checks: ['true']\n"
                "  - id: a\n    prompt: p\n    checks: ['true']\n"
            ),  # duplicate id
            "tasks:\n  - id: a\n    prompt: p\n    checks: []\n",  # empty checks
            "tasks:\n  - id: a\n    checks: ['true']\n",  # missing prompt
            (
                "tasks:\n  - id: a\n    prompt: p\n    checks: ['true']\n"
                "    fixture:\n      /abs/path: x\n"
            ),  # absolute fixture path
            (
                "tasks:\n  - id: a\n    prompt: p\n    checks: ['true']\n"
                "    fixture:\n      ../up.txt: x\n"
            ),  # traversal
            (
                "tasks:\n  - id: a\n    prompt: p\n    checks: ['true']\n"
                "    timeout_s: 0\n"
            ),  # bad timeout
            (
                "tasks:\n  - id: a\n    prompt: p\n    checks: ['true']\n"
                "    max_rounds: 0\n"
            ),  # bad rounds
        ],
    )
    def test_structural_validation(self, tmp_path, body):
        suite = tmp_path / "bad.yaml"
        suite.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError):
            load_suite(str(suite))

    def test_max_requested_ctx(self):
        tasks = [
            _task("a", requested_ctx=2048),
            _task("b", requested_ctx=8192),
            _task("c", requested_ctx=4096),
        ]
        assert max_requested_ctx(tasks) == 8192
        assert max_requested_ctx([]) == DEFAULT_REQUESTED_CTX


class TestMicroSuiteBlocklistClean:
    """Defensive pins: the suite stays clean against the REAL validator."""

    def test_fixture_contents_clean_against_content_patterns(self):
        patterns = sm.CommandValidator._DANGEROUS_FILE_CONTENT
        compiled = []
        for entry in patterns:
            pattern = entry[0] if isinstance(entry, (tuple, list)) else entry
            if isinstance(pattern, re.Pattern):
                compiled.append(pattern)
            else:
                compiled.append(re.compile(pattern, re.IGNORECASE))
        for task in load_suite("micro"):
            for relpath, content in task.fixture.items():
                for needle in compiled:
                    assert not needle.search(content), (
                        f"{task.id}:{relpath} matches dangerous content"
                        f" pattern {needle.pattern!r}"
                    )

    def test_every_check_executes_unblocked_on_real_backend(self, tmp_path):
        """Every check string runs (rc free, blocked forbidden) on the real
        tempdir backend with the task fixture registered -- the exact
        runtime shape the harness uses."""
        mgr = _make_manager(tmp_path)
        for task in load_suite("micro"):
            sid = f"clean-{task.id}"
            mgr.create_sandbox(sid)
            try:
                from opti_oignon.file_tools import _handle_sandbox_create_file

                for relpath, content in task.fixture.items():
                    result = _handle_sandbox_create_file(
                        sid, relpath, content, _sandbox_manager=mgr
                    )
                    assert not str(result).startswith("Error"), result
                for command in task.checks:
                    outcome = mgr.execute_command(sid, command, timeout=60)
                    assert not outcome.blocked, (
                        f"{task.id}: check blocked: {command!r}"
                        f" ({outcome.block_reason})"
                    )
            finally:
                mgr.destroy_sandbox(sid)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class TestStore:
    def _store(self, tmp_path):
        return EvalResultsStore(tmp_path / "eval.db")

    def test_schema_columns(self, tmp_path):
        store = self._store(tmp_path)
        import sqlite3

        conn = sqlite3.connect(store.db_path)
        try:
            runs_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(eval_runs)")
            }
            task_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(eval_task_results)")
            }
        finally:
            conn.close()
        assert runs_cols == {
            "run_id",
            "started_at",
            "finished_at",
            "suite",
            "models",
            "repeats",
            "status",
            "governor_present",
            "host_fingerprint",
            "error",
        }
        assert task_cols == {
            "id",
            "run_id",
            "model",
            "task_id",
            "repeat",
            "passed",
            "rounds",
            "tool_calls",
            "wall_s",
            "failure_class",
            "admitted",
            "admitted_ctx",
            "spill_ref",
            "diagnostics_seen",
        }

    def test_run_lifecycle_and_rows(self, tmp_path):
        store = self._store(tmp_path)
        store.create_run("r1", "micro", ["m1"], 1, True, "fp")
        run = store.get_run("r1")
        assert run["status"] == "running"
        assert run["governor_present"] is True
        assert run["models"] == ["m1"]
        store.record_task(
            "r1", "m1", "t1", 0, True, 3, 2, 1.5, "none", "yes", 4096,
            ".agent/spill/obs_1_0.txt", True,
        )
        store.record_task(
            "r1", "m1", "t2", 0, False, 1, 0, 0.5, "test_fail", "yes", 4096
        )
        store.finish_run("r1", "completed")
        details = store.get_run_details("r1")
        assert details["run"]["status"] == "completed"
        assert details["run"]["finished_at"] > 0
        rows = details["tasks"]
        assert [r["task_id"] for r in rows] == ["t1", "t2"]
        assert rows[0]["passed"] is True
        assert rows[0]["diagnostics_seen"] is True
        assert rows[0]["spill_ref"] == ".agent/spill/obs_1_0.txt"
        assert rows[1]["admitted_ctx"] == 4096

    def test_summary_math(self, tmp_path):
        store = self._store(tmp_path)
        store.create_run("r2", "micro", ["m"], 1, False, "")
        store.record_task("r2", "m", "a", 0, True, 2, 1, 1.0, "none", "absent")
        store.record_task(
            "r2", "m", "b", 0, False, 4, 3, 3.0, "test_fail", "absent"
        )
        store.record_task(
            "r2", "m", "c", 0, False, 6, 5, 2.0, "timeout", "absent"
        )
        summary = store.get_run_details("r2")["summary"]["m"]
        assert summary["total"] == 3
        assert summary["passed"] == 1
        assert summary["failures"] == {"test_fail": 1, "timeout": 1}
        assert summary["rounds_avg"] == 4.0
        assert summary["wall_avg_s"] == 2.0

    def test_invalid_values_raise(self, tmp_path):
        store = self._store(tmp_path)
        store.create_run("r3", "micro", ["m"], 1, False, "")
        with pytest.raises(ValueError):
            store.record_task(
                "r3", "m", "a", 0, False, 0, 0, 0.0, "bogus", "absent"
            )
        with pytest.raises(ValueError):
            store.record_task(
                "r3", "m", "a", 0, False, 0, 0, 0.0, "none", "maybe"
            )
        with pytest.raises(ValueError):
            store.finish_run("r3", "bogus-status")

    def test_history_order_limit_and_suite_filter(self, tmp_path):
        store = self._store(tmp_path)
        for index in range(4):
            store.create_run(f"h{index}", "micro", ["m"], 1, False, "")
            time.sleep(0.01)
        store.create_run("other", "alt", ["m"], 1, False, "")
        history = store.get_history(limit=3)
        assert len(history) == 3
        assert history[0]["run_id"] == "other"
        micro_only = store.get_history(limit=10, suite="micro")
        assert {r["suite"] for r in micro_only} == {"micro"}
        assert micro_only[0]["run_id"] == "h3"

    def test_default_db_is_separate(self):
        assert str(store_mod._DEFAULT_DB_PATH).endswith("agent_eval_results.db")
        assert "benchmark" not in str(store_mod._DEFAULT_DB_PATH)

    def test_no_fstring_sql(self):
        source = (ROOT / "opti_oignon" / "agent_eval" / "store.py").read_text(
            encoding="utf-8"
        )
        for keyword in ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE"):
            assert not re.search(r'f"[^"]*' + keyword, source)
            assert not re.search(r"f'[^']*" + keyword, source)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    @pytest.mark.parametrize(
        "args,expected",
        [
            # stop_reason, deadline, codes, blocked, attempted, executed
            (("error", False, None, False, 0, 0), (False, "error")),
            (("doom_loop", False, None, False, 4, 4), (False, "doom_loop")),
            (("cancelled", True, None, False, 1, 1), (False, "timeout")),
            (("done", False, [0, 0], False, 2, 2), (True, "none")),
            (("done", False, [0, 1], False, 2, 2), (False, "test_fail")),
            (("done", False, [1], False, 3, 0), (False, "refusal")),
            (("done", False, [1], False, 0, 0), (False, "test_fail")),
            (("done", False, None, True, 1, 1), (False, "error")),
            (("max_rounds", False, [0], False, 5, 5), (True, "none")),
        ],
    )
    def test_classification_matrix(self, args, expected):
        assert runner_mod._classify_outcome(*args) == expected

    def test_classification_only_emits_known_classes(self):
        result = runner_mod._classify_outcome("done", False, [1], False, 1, 1)
        assert result[1] in FAILURE_CLASSES

    def test_extract_spill_refs(self):
        messages = [
            {"role": "user", "content": "task"},
            {
                "role": "tool",
                "content": "[output truncated; full output: .agent/spill/obs_2_0.txt]",
            },
            {
                "role": "tool",
                "content": "[pruned round 1; spill: .agent/spill/obs_1_0.txt]",
            },
            {
                "role": "tool",
                "content": "again .agent/spill/obs_2_0.txt mentioned",
            },
            {"role": "assistant", "content": None},
        ]
        refs = runner_mod._extract_spill_refs(messages)
        assert refs == ".agent/spill/obs_2_0.txt,.agent/spill/obs_1_0.txt"
        assert runner_mod._extract_spill_refs([]) is None
        assert (
            runner_mod._extract_spill_refs([{"content": "no spill here"}])
            is None
        )

    def test_diagnostics_seen_scan(self):
        hit = types.SimpleNamespace(observation="ok\n[diagnostics]\nE501 ...")
        miss = types.SimpleNamespace(observation="clean output")
        assert runner_mod._diagnostics_seen([miss, hit]) is True
        assert runner_mod._diagnostics_seen([miss]) is False
        assert runner_mod._diagnostics_seen([]) is False

    def test_host_fingerprint_lite_shape(self):
        payload = json.loads(runner_mod._host_fingerprint_lite())
        assert set(payload) == {"system", "machine", "cpus", "python"}


# ---------------------------------------------------------------------------
# End to end on the real tempdir backend (through the facade)
# ---------------------------------------------------------------------------


class TestEndToEndTempdir:
    def test_pass_path_real_dispatch_real_checks(
        self, tempdir_world, absent_governor
    ):
        tmp_path, real, facade = tempdir_world
        script = [
            _tool(
                "create_file",
                {
                    "path": "greeting.txt",
                    "content": "Hello from the eval harness",
                },
            ),
            _final("created"),
        ]
        runner, state = _runner_for(tmp_path, facade, [script])
        task = _task(
            "pass",
            checks=["grep -qx 'Hello from the eval harness' greeting.txt"],
            timeout_s=30,
        )
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        details = runner.store.get_run_details(run_id)
        row = details["tasks"][0]
        assert details["run"]["status"] == "completed"
        assert details["run"]["governor_present"] is False
        assert row["passed"] is True
        assert row["failure_class"] == "none"
        assert row["rounds"] == 2
        assert row["tool_calls"] == 1
        assert row["admitted"] == "absent"
        assert row["admitted_ctx"] is None
        assert row["wall_s"] > 0

    def test_fixture_materialized_and_visible_to_real_dispatch(
        self, tempdir_world, absent_governor
    ):
        tmp_path, real, facade = tempdir_world
        marker = "FIXTURE-MARKER-7Q"
        script = [_tool("view", {"path": "src/depth/module.py"}), _final()]
        runner, state = _runner_for(tmp_path, facade, [script])
        task = _task(
            "fixture",
            checks=[f"grep -q '{marker}' src/depth/module.py"],
            fixture={"src/depth/module.py": f"VALUE = '{marker}'\n"},
            timeout_s=30,
        )
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        row = runner.store.get_run_details(run_id)["tasks"][0]
        assert row["passed"] is True
        assert row["failure_class"] == "none"

    def test_check_failure_is_test_fail(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        runner, _ = _runner_for(tmp_path, facade, [[_final()]])
        task = _task("red", checks=["grep -q missing-needle ."], timeout_s=30)
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        row = runner.store.get_run_details(run_id)["tasks"][0]
        assert row["passed"] is False
        assert row["failure_class"] == "test_fail"

    def test_refusal_when_gate_refuses_every_call(
        self, tempdir_world, absent_governor
    ):
        tmp_path, real, facade = tempdir_world
        script = [_tool("frobnicate", {"x": 1}), _final()]
        runner, _ = _runner_for(tmp_path, facade, [script])
        task = _task("refused", checks=["test -f nothing.txt"], timeout_s=30)
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        row = runner.store.get_run_details(run_id)["tasks"][0]
        assert row["passed"] is False
        assert row["failure_class"] == "refusal"
        assert row["tool_calls"] == 1

    def test_model_error_is_error_class(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        script = [RuntimeError("model exploded")]
        runner, _ = _runner_for(tmp_path, facade, [script])
        task = _task("boom", checks=["true"], timeout_s=30)
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        row = runner.store.get_run_details(run_id)["tasks"][0]
        assert row["failure_class"] == "error"
        assert row["passed"] is False

    def test_doom_loop_class(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        same = _tool("bash", {"command": "echo same"})
        script = [dict(same) for _ in range(6)]
        runner, _ = _runner_for(tmp_path, facade, [script])
        task = _task("doom", checks=["true"], timeout_s=60, max_rounds=10)
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        row = runner.store.get_run_details(run_id)["tasks"][0]
        assert row["failure_class"] == "doom_loop"
        assert row["passed"] is False
        assert row["rounds"] <= 5

    def test_timeout_class(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        script = [
            _tool("bash", {"command": f"sleep 0.2 # round {i}"})
            for i in range(10)
        ]
        runner, _ = _runner_for(tmp_path, facade, [script])
        task = _task("slow", checks=["true"], timeout_s=0.3, max_rounds=10)
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        row = runner.store.get_run_details(run_id)["tasks"][0]
        assert row["failure_class"] == "timeout"
        assert row["passed"] is False

    def test_blocked_check_is_error(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        runner, _ = _runner_for(tmp_path, facade, [[_final()]])
        # "rm -rf /" is hard-blocked by the validator on every backend,
        # strict mode or not -- the environment refusal the error class
        # covers (a curl-style command merely fails in non-strict tempdir).
        task = _task(
            "blocked",
            checks=["rm -rf /"],
            timeout_s=30,
        )
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        row = runner.store.get_run_details(run_id)["tasks"][0]
        assert row["failure_class"] == "error"
        assert row["passed"] is False

    def test_max_rounds_contained(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        script = [
            _tool("bash", {"command": f"echo round {i}"}) for i in range(8)
        ]
        runner, _ = _runner_for(tmp_path, facade, [script])
        task = _task("capped", checks=["true"], timeout_s=30, max_rounds=2)
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        row = runner.store.get_run_details(run_id)["tasks"][0]
        assert row["rounds"] == 2
        assert row["passed"] is True  # checks still run after max_rounds

    def test_sessions_destroyed_per_task(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        runner, _ = _runner_for(tmp_path, facade, [[_final()]])
        tasks = [
            _task("one", checks=["true"], timeout_s=30),
            _task("two", checks=["true"], timeout_s=30),
        ]
        runner.run_sync(["fake"], suite="inline", tasks=tasks, evict_between=False)
        active = [s for s in real.list_sessions() if s.get("active")]
        assert active == []

    @pytest.mark.parametrize(
        "tool_name", ["manage_skills", "manage_memory", "web_search"]
    )
    def test_state_and_egress_handlers_stubbed(
        self, tempdir_world, absent_governor, tool_name
    ):
        tmp_path, real, facade = tempdir_world
        captured = {}

        def surface_factory():
            native, handlers, prompt = runner_mod._build_eval_surface()
            captured["handlers"] = handlers
            return native, handlers, prompt

        script = [_tool(tool_name, {"action": "list", "query": "x"}), _final()]
        runner, _ = _runner_for(
            tmp_path, facade, [script], surface_factory=surface_factory
        )
        task = _task("stub", checks=["true"], timeout_s=30)
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        details = runner.store.get_run_details(run_id)
        assert details["run"]["status"] == "completed"
        handler = captured["handlers"][tool_name]
        assert handler({"action": "list"}) == (
            f"Error: {tool_name} is disabled in the eval harness"
        )

    def test_spill_ref_captured_from_real_oversized_observation(
        self, tempdir_world, absent_governor
    ):
        tmp_path, real, facade = tempdir_world
        gen = (
            "def main():\n"
            "    for i in range(1, 1501):\n"
            "        print('line', i, 'padding-padding-padding-padding')\n"
            "\n"
            "main()\n"
        )
        script = [_tool("bash", {"command": "python3 gen.py"}), _final()]
        runner, _ = _runner_for(tmp_path, facade, [script])
        task = _task(
            "spill",
            checks=["test -f gen.py"],
            fixture={"gen.py": gen},
            timeout_s=60,
        )
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        row = runner.store.get_run_details(run_id)["tasks"][0]
        assert row["passed"] is True
        assert row["spill_ref"] is not None
        assert ".agent/spill/" in row["spill_ref"]
        # Honest container reality: diagnostics are bwrap-gated and the
        # session backend stays tempdir, so the marker can never appear.
        assert row["diagnostics_seen"] is False

    def test_busy_guard_and_status(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        started = threading.Event()
        script = [
            {"content": "", "_sleep": 0.6, "_started": started,
             "tool_calls": None},
        ]
        runner, _ = _runner_for(tmp_path, facade, [script])
        task = _task("long", checks=["true"], timeout_s=30)
        run_id = runner.start_run(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        assert started.wait(5)
        assert runner.is_busy is True
        snapshot = runner.status()
        assert snapshot["busy"] is True
        assert snapshot["run_id"] == run_id
        assert snapshot["total"] == 1
        with pytest.raises(RuntimeError):
            runner.start_run(["fake"], suite="inline", tasks=[task])
        deadline = time.time() + 10
        while runner.is_busy and time.time() < deadline:
            time.sleep(0.05)
        assert runner.is_busy is False
        assert runner.store.get_run(run_id)["status"] == "completed"

    def test_cancel_drops_in_flight_row(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        started = threading.Event()
        scripts = [
            [
                _final("first done"),
                {"content": "", "_sleep": 1.2, "_started": started,
                 "tool_calls": None},
            ]
        ]
        runner, _ = _runner_for(tmp_path, facade, scripts)
        tasks = [
            _task("quick", checks=["true"], timeout_s=30),
            _task("victim", checks=["true"], timeout_s=30),
        ]
        run_id = runner.start_run(
            ["fake"], suite="inline", tasks=tasks, evict_between=False
        )
        assert started.wait(5)
        assert runner.cancel() is True
        deadline = time.time() + 10
        while runner.is_busy and time.time() < deadline:
            time.sleep(0.05)
        details = runner.store.get_run_details(run_id)
        assert details["run"]["status"] == "cancelled"
        assert [r["task_id"] for r in details["tasks"]] == ["quick"]
        active = [s for s in real.list_sessions() if s.get("active")]
        assert active == []

    def test_repeats_produce_rows(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        runner, _ = _runner_for(tmp_path, facade, [[_final()]])
        task = _task("rep", checks=["true"], timeout_s=30)
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], repeats=2,
            evict_between=False,
        )
        rows = runner.store.get_run_details(run_id)["tasks"]
        assert [(r["task_id"], r["repeat"]) for r in rows] == [
            ("rep", 0),
            ("rep", 1),
        ]


# ---------------------------------------------------------------------------
# Admission provenance paths
# ---------------------------------------------------------------------------


class TestAdmissionPaths:
    def test_absent_path(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        runner, _ = _runner_for(tmp_path, facade, [[_final()]])
        task = _task("abs", checks=["true"], timeout_s=30)
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        details = runner.store.get_run_details(run_id)
        assert details["run"]["governor_present"] is False
        row = details["tasks"][0]
        assert row["admitted"] == "absent"
        assert row["admitted_ctx"] is None

    def test_admitted_path_and_ticket_composition(
        self, tempdir_world, stub_governor_factory
    ):
        tmp_path, real, facade = tempdir_world
        decision = _decision(admitted=True, num_ctx=4096)
        stub = stub_governor_factory(decision)
        seen_tickets = []

        def surface_factory():
            native, handlers, prompt = runner_mod._build_eval_surface()

            def capture(arguments):
                seen_tickets.append(real_governor.get_active_ticket())
                return "captured"

            handlers["todo"] = capture
            return native, handlers, prompt

        script = [_tool("todo", {"items": []}), _final()]
        runner, _ = _runner_for(
            tmp_path, facade, [script], surface_factory=surface_factory
        )
        tasks = [
            _task("low", checks=["true"], timeout_s=30, requested_ctx=4096),
            _task("high", checks=["true"], timeout_s=30, requested_ctx=8192),
        ]
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=tasks, evict_between=False
        )
        details = runner.store.get_run_details(run_id)
        assert details["run"]["governor_present"] is True
        for row in details["tasks"]:
            assert row["admitted"] == "yes"
            assert row["admitted_ctx"] == 4096
        # One admission per model, at the suite's max requested_ctx, with
        # the eval caller string.
        assert len(stub.admit_calls) == 1
        assert stub.admit_calls[0]["requested_ctx"] == 8192
        assert stub.admit_calls[0]["caller"] == "agent_eval"
        # The 6.6 composition: the REAL thread-local ticket the loop reads
        # is the held decision, seen from inside a running tool handler.
        assert seen_tickets and all(t is decision for t in seen_tickets)
        # Outside the scope the ticket is cleared.
        assert real_governor.get_active_ticket() is None

    def test_refused_rows_and_model_skipped(
        self, tempdir_world, stub_governor_factory
    ):
        tmp_path, real, facade = tempdir_world
        stub = stub_governor_factory(
            _decision(admitted=False, reason="insufficient vram")
        )
        factory_calls = []

        def factory(model):
            factory_calls.append(model)
            return ScriptedClient([_final()])

        runner = EvalRunner(
            db_path=str(tmp_path / "eval.db"),
            sandbox_manager=facade,
            client_factory=factory,
        )
        tasks = [
            _task("a", checks=["true"], timeout_s=30),
            _task("b", checks=["true"], timeout_s=30),
        ]
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=tasks, repeats=2,
            evict_between=False,
        )
        details = runner.store.get_run_details(run_id)
        assert details["run"]["status"] == "completed"
        rows = details["tasks"]
        assert len(rows) == 4
        for row in rows:
            assert row["failure_class"] == "not_admitted"
            assert row["admitted"] == "refused"
            assert row["admitted_ctx"] is None
            assert row["passed"] is False
        assert factory_calls == []  # the model was never even built
        assert [s for s in real.list_sessions() if s.get("active")] == []

    def test_estop_shaped_refusal_is_not_admitted(
        self, tempdir_world, stub_governor_factory
    ):
        tmp_path, real, facade = tempdir_world
        stub_governor_factory(
            _decision(admitted=False, reason="estop", is_estop=True)
        )
        runner, _ = _runner_for(tmp_path, facade, [[_final()]])
        task = _task("estop", checks=["true"], timeout_s=30)
        run_id = runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        row = runner.store.get_run_details(run_id)["tasks"][0]
        assert row["failure_class"] == "not_admitted"
        assert row["admitted"] == "refused"

    def test_invalidate_on_load_called_when_load_expected(
        self, tempdir_world, stub_governor_factory
    ):
        tmp_path, real, facade = tempdir_world
        stub = stub_governor_factory(
            _decision(admitted=True, num_ctx=2048, load_expected=True)
        )
        runner, _ = _runner_for(tmp_path, facade, [[_final()]])
        task = _task("load", checks=["true"], timeout_s=30)
        runner.run_sync(
            ["fake"], suite="inline", tasks=[task], evict_between=False
        )
        assert stub.load_invalidations == [("fake", 2048)]

    def test_evict_between_models_seam(
        self, tempdir_world, stub_governor_factory, monkeypatch
    ):
        tmp_path, real, facade = tempdir_world
        stub = stub_governor_factory(_decision(admitted=True, num_ctx=4096))
        unloads = []

        class _FakeBackend:
            def unload_all(self):
                unloads.append(1)
                return 1

        fake_ib = types.SimpleNamespace(
            get_backend_registry=lambda: types.SimpleNamespace(
                backends=lambda: [_FakeBackend()]
            )
        )
        monkeypatch.setitem(
            sys.modules, "opti_oignon.inference_backend", fake_ib
        )
        runner, _ = _runner_for(tmp_path, facade, [[_final()]])
        task = _task("ev", checks=["true"], timeout_s=30)
        runner.run_sync(
            ["m1", "m2"], suite="inline", tasks=[task], evict_between=True
        )
        # Called once BETWEEN the two models, not after the last.
        assert unloads == [1]
        assert stub.evict_invalidations == [None]
        unloads.clear()
        stub.evict_invalidations.clear()
        runner.run_sync(
            ["m1", "m2"], suite="inline", tasks=[task], evict_between=False
        )
        assert unloads == []


# ---------------------------------------------------------------------------
# Runner misc
# ---------------------------------------------------------------------------


class TestRunnerValidationAndSingleton:
    def test_prepare_validations(self, tempdir_world, absent_governor):
        tmp_path, real, facade = tempdir_world
        runner, _ = _runner_for(tmp_path, facade)
        with pytest.raises(ValueError):
            runner.run_sync([], suite="inline", tasks=[_task()])
        with pytest.raises(ValueError):
            runner.run_sync(["m"], suite="inline", tasks=[_task()], repeats=0)
        with pytest.raises(FileNotFoundError):
            runner.run_sync(["m"], suite="no-such-suite")
        with pytest.raises(ValueError):
            runner.run_sync(["m"], suite="inline", tasks=[])

    def test_singleton_get_reset(self):
        reset_eval_runner()
        first = get_eval_runner()
        assert get_eval_runner() is first
        reset_eval_runner()
        assert get_eval_runner() is not first
        reset_eval_runner()


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self):
        self.details = None
        self.history = []

    def get_run_details(self, run_id):
        return self.details

    def get_history(self, limit, suite=None):
        return self.history


class _FakeRunner:
    def __init__(self):
        self.busy = False
        self.started = []
        self.store = _FakeStore()
        self.cancelled = False

    @property
    def is_busy(self):
        return self.busy

    def start_run(self, **kwargs):
        self.started.append(kwargs)
        return "eval-fake12345678"

    def status(self):
        return {"busy": self.busy, "run_id": "eval-fake12345678"}

    def cancel(self):
        self.cancelled = True
        return True


@pytest.fixture
def api_client(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import opti_oignon.api.routes_agent_eval as rae

    assert rae.router is not None
    fake = _FakeRunner()
    monkeypatch.setattr(rae, "get_eval_runner", lambda: fake)
    app = fastapi.FastAPI()
    app.include_router(rae.router)
    if getattr(rae, "_get_current_user", None) is not None:
        app.dependency_overrides[rae._get_current_user] = lambda: {
            "username": "tester"
        }
    return TestClient(app), fake, rae


class TestAPIRoutes:
    def test_route_paths(self):
        import opti_oignon.api.routes_agent_eval as rae

        paths = {route.path for route in rae.router.routes}
        assert paths == {
            "/api/agent-eval/run",
            "/api/agent-eval/status",
            "/api/agent-eval/results/{run_id}",
            "/api/agent-eval/history",
            "/api/agent-eval/cancel",
        }

    def test_auth_parity_dependency(self):
        import opti_oignon.api.routes_agent_eval as rae

        assert getattr(rae, "_get_current_user", None) is not None
        deps = [d.dependency for d in rae.router.dependencies]
        assert rae._get_current_user in deps

    def test_run_happy_path(self, api_client):
        client, fake, _ = api_client
        response = client.post(
            "/api/agent-eval/run",
            json={"models": ["qwen3:4b"], "suite": "micro", "repeats": 1},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["started"] is True
        assert body["run_id"].startswith("eval-")
        assert fake.started[0]["models"] == ["qwen3:4b"]

    def test_run_accepts_csv_models(self, api_client):
        client, fake, _ = api_client
        response = client.post(
            "/api/agent-eval/run",
            json={"models": "a, b ,", "suite": "micro"},
        )
        assert response.status_code == 200
        assert fake.started[0]["models"] == ["a", "b"]

    def test_run_409_when_busy(self, api_client):
        client, fake, _ = api_client
        fake.busy = True
        response = client.post(
            "/api/agent-eval/run", json={"models": ["m"], "suite": "micro"}
        )
        assert response.status_code == 409

    def test_run_422_empty_models_and_bad_repeats(self, api_client):
        client, _, _ = api_client
        assert (
            client.post(
                "/api/agent-eval/run", json={"models": [], "suite": "micro"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/agent-eval/run",
                json={"models": ["m"], "suite": "micro", "repeats": 0},
            ).status_code
            == 422
        )

    def test_run_404_unknown_suite(self, api_client):
        client, _, _ = api_client
        response = client.post(
            "/api/agent-eval/run",
            json={"models": ["m"], "suite": "no-such-suite"},
        )
        assert response.status_code == 404

    def test_status_endpoint(self, api_client):
        client, _, _ = api_client
        response = client.get("/api/agent-eval/status")
        assert response.status_code == 200
        assert response.json()["run_id"] == "eval-fake12345678"

    def test_results_404_and_ok(self, api_client):
        client, fake, _ = api_client
        assert client.get("/api/agent-eval/results/nope").status_code == 404
        fake.store.details = {"run": {"run_id": "x"}, "tasks": [], "summary": {}}
        response = client.get("/api/agent-eval/results/x")
        assert response.status_code == 200
        assert response.json()["run"]["run_id"] == "x"

    def test_history_shape(self, api_client):
        client, fake, _ = api_client
        fake.store.history = [{"run_id": "a"}]
        response = client.get("/api/agent-eval/history?limit=5")
        assert response.status_code == 200
        assert response.json() == {"runs": [{"run_id": "a"}]}

    def test_cancel(self, api_client):
        client, fake, _ = api_client
        response = client.post("/api/agent-eval/cancel")
        assert response.status_code == 200
        assert response.json() == {"cancelled": True}
        assert fake.cancelled is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def _suite_file(self, tmp_path):
        suite = tmp_path / "cli.yaml"
        suite.write_text(
            "suite: cli\n"
            "tasks:\n"
            "  - id: t-cli\n"
            "    prompt: create greeting.txt with the marker line\n"
            "    checks:\n"
            "      - \"grep -qx 'cli marker' greeting.txt\"\n"
            "    timeout_s: 30\n",
            encoding="utf-8",
        )
        return suite

    def test_main_smoke_exit_zero(
        self, tempdir_world, absent_governor, tmp_path, capsys
    ):
        _, real, facade = tempdir_world
        suite = self._suite_file(tmp_path)
        script = [
            _tool(
                "create_file",
                {"path": "greeting.txt", "content": "cli marker"},
            ),
            _final(),
        ]
        runner, _ = _runner_for(tmp_path, facade, [script])
        code = eval_cli.main(
            ["--models", "fake", "--suite", str(suite)], runner=runner
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "1/1 passed" in out
        assert "governor=absent" in out

    def test_main_argument_errors(self, tmp_path, capsys):
        assert eval_cli.main(["--models", " , "]) == 2
        assert (
            eval_cli.main(["--models", "m", "--suite", "no-such-suite"]) == 2
        )
        assert eval_cli.main(["--models", "m", "--repeats", "0"]) == 2

    def test_main_exit_one_on_failed_run(self, tmp_path, capsys):
        suite = self._suite_file(tmp_path)
        fake = _FakeRunner()
        fake.run_sync = lambda **kw: "eval-x"
        fake.store.details = {
            "run": {
                "run_id": "eval-x",
                "suite": "cli",
                "status": "failed",
                "governor_present": False,
                "models": ["m"],
                "error": "boom",
            },
            "tasks": [],
            "summary": {},
        }
        code = eval_cli.main(
            ["--models", "m", "--suite", str(suite)], runner=fake
        )
        assert code == 1
        assert "status=failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Deliverable pins
# ---------------------------------------------------------------------------


class TestDeliverablePins:
    def test_atrest_row_present_and_clean(self):
        text = (ROOT / "ATREST_INVENTORY.md").read_text(encoding="utf-8")
        row = next(
            (
                line
                for line in text.splitlines()
                if "agent_eval_results.db" in line
            ),
            None,
        )
        assert row is not None
        assert "pending-scoping" in row
        assert "added S230" in row
        assert "post-BK-06 additive rule" in row
        assert "bk06-candidate" not in row

    def test_scripts_exist_and_shaped(self):
        run_script = ROOT / "scripts" / "run_agent_eval.sh"
        baseline = ROOT / "scripts" / "agent_eval_opencode_baseline.sh"
        assert run_script.exists()
        assert baseline.exists()
        run_text = run_script.read_text(encoding="utf-8")
        assert "python3 -m opti_oignon.agent_eval" in run_text
        base_text = baseline.read_text(encoding="utf-8")
        assert "4519a1da329c1a4fc384054e7203ba7d06928205" in base_text
        assert "sst/opencode" in base_text
        assert "MIT" in base_text
        assert "command -v opencode" in base_text
        assert "exit 2" in base_text
        assert "opencode-baseline" in base_text

    def test_app_mounts_guarded(self):
        source = (ROOT / "opti_oignon" / "api" / "app.py").read_text(
            encoding="utf-8"
        )
        assert (
            "from .routes_agent_eval import router as agent_eval_router"
            in source
        )
        assert "if agent_eval_router is not None:" in source
        assert "app.include_router(agent_eval_router)" in source

    def test_micro_suite_file_shipped(self):
        assert (
            ROOT / "opti_oignon" / "agent_eval" / "suites" / "micro.yaml"
        ).exists()

    def test_ast_validity_of_new_files(self):
        for path in NEW_PY_FILES:
            ast.parse(path.read_text(encoding="utf-8"))

    def test_version_holds(self):
        # Read at the SOURCE: other suites in a shared sweep replace
        # sys.modules["opti_oignon"] with isolation stubs lacking
        # __version__, so the deliverable pin must not depend on the
        # import system's state.
        source = (ROOT / "opti_oignon" / "__version__.py").read_text(
            encoding="utf-8"
        )
        match = re.search(r'__version__\s*=\s*"([^"]+)"', source)
        assert match is not None
        assert match.group(1) == "3.9.0"
