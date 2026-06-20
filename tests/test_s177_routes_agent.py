#!/usr/bin/env python3
"""End-to-end tests for S177 -- the live agent route wiring (Theme 3).

Exercises opti_oignon/api/routes_agent.py: the AgentRunManager that wires the
loop, the per-mode tool set, the context-bound manage_skills handler, the
skill-consultation seam, and the working-memory provider into a running agent,
plus the status / cancel / event-stream contract the panel consumes and the
register() hook used by app.py.

The run engine has no web dependency, so it runs here with an injected model
client and sandbox; the FastAPI surface is absent in this environment (router is
None), which the tests treat as the expected isolated state. Loaded via
spec_from_file_location with opti_oignon stubbed.
"""

import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"
API = OO / "api"


def _ensure_pkgs():
    for name, sub in (
        ("opti_oignon", OO),
        ("opti_oignon.agent", AGENT),
        ("opti_oignon.api", API),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod


def _load_agent(name: str):
    full = f"opti_oignon.agent.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(AGENT / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkgs()
for _m in ("tool_parsing", "allowlists", "dispatch", "untrusted_context", "loop", "tools", "teacher"):
    _load_agent(_m)
sk = _load_agent("skills")

_ra_spec = importlib.util.spec_from_file_location(
    "opti_oignon.api.routes_agent", str(API / "routes_agent.py")
)
ra = importlib.util.module_from_spec(_ra_spec)
sys.modules["opti_oignon.api.routes_agent"] = ra
_ra_spec.loader.exec_module(ra)


# Fakes (model client uses the loop's stream(messages, tools) contract)


def _native(name, args):
    return [{"function": {"name": name, "arguments": args}}]


class ScriptedClient:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def stream(self, messages, tools=None):
        step = self.script[self.i] if self.i < len(self.script) else {"content": "done.", "tool_calls": None}
        self.i += 1
        yield {"message": {"content": step.get("content", ""), "tool_calls": step.get("tool_calls")}}


class BoomClient:
    def stream(self, messages, tools=None):
        raise RuntimeError("model crashed")


class GatedClient:
    """Blocks inside the first stream call until released (for cancel timing)."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def stream(self, messages, tools=None):
        self.calls += 1
        self.entered.set()
        self.release.wait(timeout=5)
        yield {"message": {"content": "again", "tool_calls": _native("view", {"path": "/x"})}}


class _FakeMgr:
    def __init__(self, bwrap=True):
        self.bwrap_available = bwrap


class FakeSession:
    def __init__(self, bwrap=True, active=True):
        self.sandbox_manager = _FakeMgr(bwrap)
        self.active = active
        self.calls = []

    def view(self, path, start_line=0, end_line=0):
        self.calls.append(("view", path))
        return f"view {path}"

    def bash(self, command, timeout=30):
        self.calls.append(("bash", command))
        return f"out {command}"


class _Approver:
    def __init__(self, approve):
        self._approve = approve
        self.calls = []

    def __call__(self, conversation_id, tool_name, arguments):
        self.calls.append((conversation_id, tool_name, dict(arguments)))
        return self._approve


def _skill_body():
    return (
        "## When to Use\nWhen deploying a service to the cluster.\n\n"
        "## Procedure\nrun the deploy script\n\n"
        "## Pitfalls\nnone\n\n"
        "## Verification\nConfirm the service responds.\n"
    )


def _finish(mgr, timeout=10.0):
    """Wait deterministically for a run to finish.

    join() alone can return before the thread is scheduled under load, so we
    join then poll the engine's own running flag. Because the loop broadcasts
    every event synchronously before it returns and the flag is cleared after,
    a finished run guarantees all events have already been delivered.
    """
    mgr.join(timeout)
    deadline = time.time() + timeout
    while mgr.is_running() and time.time() < deadline:
        time.sleep(0.01)
    assert not mgr.is_running(), "agent run did not finish in time"


@pytest.fixture(autouse=True)
def _reset():
    ra.reset_run_manager()
    sk.reset_skill_registry()
    yield
    ra.reset_run_manager()
    sk.reset_skill_registry()


@pytest.fixture
def mgr():
    return ra.AgentRunManager()


@pytest.fixture
def reg(tmp_path):
    return sk.SkillRegistry(root=tmp_path)


# Module structure


class TestModuleStructure:
    def test_sentinels(self):
        assert ra.FEATURE_AVAILABLE is True
        assert ra.checkpoint_before_apply is True

    def test_agent_imports_available(self):
        assert ra._AGENT_OK is True

    def test_router_is_none_without_fastapi(self):
        # FastAPI is absent in this environment; the module still loads.
        assert ra.router is None

    def test_singleton_and_reset(self):
        a = ra.get_run_manager()
        ra.reset_run_manager()
        b = ra.get_run_manager()
        assert a is not b

    def test_register_no_router_returns_false(self):
        class _App:
            included = []

            def include_router(self, r):
                self.included.append(r)

        app = _App()
        assert ra.register(app) is False  # router is None here
        assert app.included == []

    def test_register_with_router_includes(self, monkeypatch):
        class _App:
            def __init__(self):
                self.included = []

            def include_router(self, r):
                self.included.append(r)

        sentinel = object()
        monkeypatch.setattr(ra, "router", sentinel)
        app = _App()
        assert ra.register(app) is True
        assert app.included == [sentinel]


# Run lifecycle and status


class TestRunLifecycle:
    def test_single_final_answer(self, mgr, reg):
        r = mgr.start(
            "do it",
            model_client=ScriptedClient([{"content": "All set.", "tool_calls": None}]),
            mode="daily",
            sandbox=FakeSession(),
            registry=reg,
            consult=False,
        )
        assert r == {"started": True}
        _finish(mgr)
        st = mgr.status()
        assert st == {"running": False, "rounds": 1, "stop_reason": "done"}

    def test_multi_round_tool_then_final(self, mgr, reg):
        sess = FakeSession()
        script = [
            {"content": "looking", "tool_calls": _native("view", {"path": "/p"})},
            {"content": "Done.", "tool_calls": None},
        ]
        mgr.start("look", model_client=ScriptedClient(script), mode="daily", sandbox=sess, registry=reg, consult=False)
        _finish(mgr)
        assert mgr.status()["rounds"] == 2
        assert mgr.status()["stop_reason"] == "done"
        assert ("view", "/p") in sess.calls

    def test_status_shape(self, mgr):
        st = mgr.status()
        assert set(st) == {"running", "rounds", "stop_reason"}
        assert st["running"] is False

    def test_already_running_rejected(self, mgr):
        mgr._running = True
        r = mgr.start("x", model_client=ScriptedClient([]))
        assert r["started"] is False
        assert r["reason"] == "already_running"

    def test_boom_client_does_not_crash(self, mgr, reg):
        mgr.start("x", model_client=BoomClient(), mode="daily", sandbox=FakeSession(), registry=reg, consult=False)
        _finish(mgr)
        st = mgr.status()
        assert st["running"] is False
        # The loop turns a model error into a terminal error stop reason.
        assert st["stop_reason"] in {"error", "done"}

    def test_agent_unavailable_guard(self, monkeypatch, mgr):
        monkeypatch.setattr(ra, "_AGENT_OK", False)
        r = mgr.start("x", model_client=ScriptedClient([]))
        assert r == {"started": False, "reason": "agent_unavailable"}


# Cancellation


class TestCancellation:
    def test_cancel_idle_is_false(self, mgr):
        assert mgr.cancel() == {"cancelled": False}

    def test_cancel_active_run(self, mgr, reg):
        client = GatedClient()
        mgr.start("loop", model_client=client, mode="daily", sandbox=FakeSession(), registry=reg, consult=False)
        assert client.entered.wait(3)  # round 1 in progress
        assert mgr.is_running() is True
        assert mgr.cancel() == {"cancelled": True}
        client.release.set()  # let round 1 finish; round 2 sees the cancel
        _finish(mgr)
        st = mgr.status()
        assert st["running"] is False
        assert st["stop_reason"] == "cancelled"
        assert client.calls == 1  # the loop did not start a second round


# Event stream fan-out


class TestEventStream:
    def test_subscriber_receives_events(self, mgr, reg):
        received = []
        mgr.subscribe(lambda p: received.append(p))
        mgr.start(
            "x",
            model_client=ScriptedClient([{"content": "Hi.", "tool_calls": None}]),
            mode="daily",
            sandbox=FakeSession(),
            registry=reg,
            consult=False,
        )
        _finish(mgr)
        kinds = [json.loads(p)["kind"] for p in received]
        assert "round_start" in kinds
        assert "model_output" in kinds
        assert "done" in kinds

    def test_event_payload_shape(self, mgr, reg):
        received = []
        mgr.subscribe(lambda p: received.append(p))
        mgr.start(
            "x",
            model_client=ScriptedClient([{"content": "Hi.", "tool_calls": None}]),
            mode="daily",
            sandbox=FakeSession(),
            registry=reg,
            consult=False,
        )
        _finish(mgr)
        obj = json.loads(received[0])
        assert set(obj) == {"kind", "round", "data"}
        assert isinstance(obj["round"], int)

    def test_unsubscribe_stops_delivery(self, mgr, reg):
        received = []
        cb = mgr.subscribe(lambda p: received.append(p))
        mgr.unsubscribe(cb)
        mgr.start(
            "x",
            model_client=ScriptedClient([{"content": "Hi.", "tool_calls": None}]),
            mode="daily",
            sandbox=FakeSession(),
            registry=reg,
            consult=False,
        )
        _finish(mgr)
        assert received == []


# manage_skills is wired with the run's registry, gate, and sandbox


class TestManageSkillsWiring:
    def test_agent_can_draft_skill_through_loop(self, mgr, reg):
        approver = _Approver(True)
        script = [
            {
                "content": "I will save this procedure.",
                "tool_calls": _native(
                    "manage_skills",
                    {"action": "add", "name": "deploy", "category": "coding", "body": _skill_body()},
                ),
            },
            {"content": "Saved as a draft.", "tool_calls": None},
        ]
        mgr.start(
            "save deploy procedure",
            model_client=ScriptedClient(script),
            mode="daily",
            conversation_id="c1",
            sandbox=FakeSession(),
            approval_fn=approver,
            registry=reg,
            consult=False,
        )
        _finish(mgr)
        # The context-bound handler wrote the draft into this run's registry,
        # and its gate (the injected approval_fn) was consulted.
        assert reg.get("deploy", "coding", draft=True) is not None
        assert any(label == "manage_skills:add" for _, label, _ in approver.calls)

    def test_gate_denial_blocks_skill_write_through_loop(self, mgr, reg):
        approver = _Approver(False)
        script = [
            {
                "content": "saving",
                "tool_calls": _native(
                    "manage_skills",
                    {"action": "add", "name": "deploy", "category": "coding", "body": _skill_body()},
                ),
            },
            {"content": "could not save", "tool_calls": None},
        ]
        mgr.start(
            "save",
            model_client=ScriptedClient(script),
            mode="daily",
            sandbox=FakeSession(),
            approval_fn=approver,
            registry=reg,
            consult=False,
        )
        _finish(mgr)
        assert reg.get("deploy", "coding", draft=True) is None


# Skill consultation seam


class TestConsultation:
    def test_consult_increments_usage(self, mgr, reg):
        reg.add("Deploy Service", "coding", _skill_body(), status="published")
        mgr.start(
            "deploy a service to the cluster",
            model_client=ScriptedClient([{"content": "ok", "tool_calls": None}]),
            mode="daily",
            sandbox=FakeSession(),
            registry=reg,
            consult=True,
        )
        _finish(mgr)
        # consult_skills ran during start and bumped the matched skill's sidecar.
        assert reg.get_usage("deploy-service", "coding").uses == 1

    def test_no_consult_when_disabled(self, mgr, reg):
        reg.add("Deploy Service", "coding", _skill_body(), status="published")
        mgr.start(
            "deploy a service",
            model_client=ScriptedClient([{"content": "ok", "tool_calls": None}]),
            mode="daily",
            sandbox=FakeSession(),
            registry=reg,
            consult=False,
        )
        _finish(mgr)
        assert reg.get_usage("deploy-service", "coding").uses == 0
