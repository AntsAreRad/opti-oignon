#!/usr/bin/env python3
"""S215 -- Emergency-stop lot: panic control + resume.

Per-fix suite for the ROADMAP_POST_AUDIT standalone entry, as arbitrated at
the S215 gate: the global "stopped" flag set FIRST (admission closes before
the drain; fail-secure on the flag by construction), the ordered
fail-tolerant stop sequence over faked primitives (cancel generations,
cancel agent runs, stop the coding background run, unload models on every
registered backend, destroy sandbox sessions, stop the Veilid node -- never
gated, F9e -- and the optional drop-to-Bulbe), the no-ceremony resume
(client reconnect reported honestly; the Veilid restart Daily-only via the
existing binding-layer guard and only when the node was running at stop
time), the audit rows both ways, the admission guards across the compute
entries (refused, not hung: 503 on REST, the error token on the chat
websockets), the routes_security auth/status ladders, the new inference
unload primitives (OllamaBackend.unload_all via ps + keep_alive=0, the
CC-01 dict/object tolerance; the IB-04 pop-based LlamaCpp unload pick-up),
and the UI/docs registrations by source. The flag is in-process only: a
crash-restart comes back UNSTOPPED by design.

Harness: the s210..s214 ``_load_fresh`` shape -- this file ALWAYS execs its
own module copies and never reuses a pre-loaded canonical module.
"""

import ast
import importlib.util
import os
import re
import sys
import types

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_OO = os.path.join(_ROOT, "opti_oignon")
_API = os.path.join(_OO, "api")
_FRONT = os.path.join(_ROOT, "frontend", "src", "lib")


def _ensure_pkg(name: str, path: str) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod


_ensure_pkg("opti_oignon", _OO)
_ensure_pkg("opti_oignon.api", _API)


def _load_fresh(relpath: str, register: str, bind: dict | None = None):
    """ALWAYS exec this file's own copy; never reuse a pre-loaded module.

    Temporarily register ``bind`` plus the module's own name, exec the
    fresh copy, restore every touched sys.modules entry afterwards. A
    ``None`` bind value makes the corresponding import raise (the
    guarded-import refusal path).
    """
    bind = dict(bind or {})
    touched = list(bind.keys()) + [register]
    saved = {name: sys.modules.get(name) for name in touched}
    try:
        for name, mod in bind.items():
            sys.modules[name] = mod
        path = os.path.join(_ROOT, relpath)
        spec = importlib.util.spec_from_file_location(register, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[register] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# The seam module under test (stdlib-only at import time).
_es = _load_fresh(
    os.path.join("opti_oignon", "emergency_stop.py"),
    register="opti_oignon.emergency_stop",
)


# ---------------------------------------------------------------------------
# Fakes (the proof seams: every primitive recorded on a shared call log)
# ---------------------------------------------------------------------------

class FakeExecutor:
    def __init__(self, calls, name, fail=False, flag_probe=None):
        self._calls = calls
        self._name = name
        self._fail = fail
        self._flag_probe = flag_probe

    def cancel(self):
        if self._flag_probe is not None:
            self._flag_probe.append(_es.is_stopped())
        self._calls.append(f"cancel:{self._name}")
        if self._fail:
            raise RuntimeError(f"{self._name} cancel boom")


class FakeRunManager:
    def __init__(self, calls, fail=False):
        self._calls = calls
        self._fail = fail

    def cancel(self):
        self._calls.append("agent.cancel")
        if self._fail:
            raise RuntimeError("agent cancel boom")
        return {"cancelled": True}


class FakeRunState:
    def __init__(self, calls, fail=False, running=True):
        self._calls = calls
        self._fail = fail
        self._running = running

    def stop(self):
        self._calls.append("coding.stop")
        if self._fail:
            raise RuntimeError("coding stop boom")
        return self._running


class FakeBackend:
    def __init__(self, calls, name, fail=False, count=2, healthy=True,
                 supports_unload=True):
        self._calls = calls
        self.name = name
        self._fail = fail
        self._count = count
        self._healthy = healthy
        if not supports_unload:
            self.unload_all = None  # not callable: the "unsupported" path

    def unload_all(self):
        self._calls.append(f"unload:{self.name}")
        if self._fail:
            raise RuntimeError(f"{self.name} unload boom")
        return self._count

    def health_check(self):
        self._calls.append(f"health:{self.name}")
        if isinstance(self._healthy, Exception):
            raise self._healthy
        return self._healthy


class FakeRegistry:
    def __init__(self, backends, active=None):
        self._list = backends
        self.active = active

    def backends(self):
        return list(self._list)


class FakeSandboxManager:
    def __init__(self, calls, ids=("ws-a", "ws-b"), fail_ids=(), list_fail=False):
        self._calls = calls
        self._ids = list(ids)
        self._fail_ids = set(fail_ids)
        self._list_fail = list_fail

    def list_sessions(self):
        self._calls.append("sandbox.list")
        if self._list_fail:
            raise RuntimeError("list boom")
        return [{"session_id": sid} for sid in self._ids]

    def destroy_sandbox(self, session_id):
        self._calls.append(f"sandbox.destroy:{session_id}")
        if session_id in self._fail_ids:
            raise RuntimeError(f"destroy boom {session_id}")
        return True


class FakeNode:
    def __init__(self, calls, running=True, stop_fail=False, start_exc=None):
        self._calls = calls
        self.is_running = running
        self._stop_fail = stop_fail
        self._start_exc = start_exc

    def stop(self):
        self._calls.append("node.stop")
        if self._stop_fail:
            raise RuntimeError("node stop boom")
        return {"state": "stopped"}

    def start(self):
        self._calls.append("node.start")
        if self._start_exc is not None:
            raise self._start_exc
        return {"state": "started"}


class FakeModeManager:
    def __init__(self, calls, success=True):
        self._calls = calls
        self._success = success
        self.last_user = None

    def escalate_to_bulbe(self, user_id):
        self._calls.append("mode.escalate")
        self.last_user = user_id
        if self._success:
            return {"success": True, "message": "ok"}
        return {"success": False, "message": "refused"}


class FakeWarmer:
    def __init__(self, calls, success=True):
        self._calls = calls
        self._success = success
        self.last_model = None

    def warmup(self, model):
        self._calls.append(f"warmup:{model}")
        self.last_model = model
        return types.SimpleNamespace(success=self._success)


class _Seam:
    """One wired fake world per test."""

    def __init__(self, monkeypatch, **kw):
        self.calls: list = []
        self.chain_rows: list = []
        self.flag_probe: list = []
        # Distinct class names: the seam identifies executors by type name.
        exec_a_cls = type("ExecA", (FakeExecutor,), {})
        exec_b_cls = type("ExecB", (FakeExecutor,), {})
        self.exec_a = exec_a_cls(
            self.calls, "ExecA",
            fail=kw.get("exec_a_fail", False),
            flag_probe=self.flag_probe,
        )
        self.exec_b = exec_b_cls(self.calls, "ExecB")
        self.run_manager = FakeRunManager(
            self.calls, fail=kw.get("agent_fail", False))
        self.run_state = FakeRunState(
            self.calls, fail=kw.get("coding_fail", False))
        self.ollama = FakeBackend(
            self.calls, "ollama",
            fail=kw.get("ollama_fail", False),
            healthy=kw.get("ollama_healthy", True),
        )
        self.llamacpp = FakeBackend(
            self.calls, "llama_cpp",
            supports_unload=kw.get("llamacpp_supports", True),
        )
        self.registry = FakeRegistry(
            [self.ollama, self.llamacpp], active=self.ollama)
        self.sandbox = FakeSandboxManager(
            self.calls,
            fail_ids=kw.get("sandbox_fail_ids", ()),
            list_fail=kw.get("sandbox_list_fail", False),
        )
        self.node = FakeNode(
            self.calls,
            running=kw.get("node_running", True),
            stop_fail=kw.get("node_stop_fail", False),
            start_exc=kw.get("node_start_exc", None),
        )
        self.mode = FakeModeManager(
            self.calls, success=kw.get("bulbe_success", True))
        self.warmer = FakeWarmer(self.calls)

        monkeypatch.setattr(
            _es, "_resolve_executors", lambda: [self.exec_a, self.exec_b])
        monkeypatch.setattr(
            _es, "_resolve_run_manager", lambda: self.run_manager)
        monkeypatch.setattr(
            _es, "_resolve_coding_run_state", lambda: self.run_state)
        monkeypatch.setattr(
            _es, "_resolve_backend_registry", lambda: self.registry)
        monkeypatch.setattr(
            _es, "_resolve_sandbox_manager", lambda: self.sandbox)
        monkeypatch.setattr(_es, "_resolve_node", lambda: self.node)
        monkeypatch.setattr(_es, "_resolve_mode_manager", lambda: self.mode)
        monkeypatch.setattr(_es, "_resolve_warmup", lambda: self.warmer)
        monkeypatch.setattr(
            _es, "_chain",
            lambda action, severity, **d: self.chain_rows.append(
                {"action": action, "severity": severity, **d}),
        )


@pytest.fixture()
def seam(monkeypatch):
    _es.reset_for_tests()
    yield lambda **kw: _Seam(monkeypatch, **kw)
    _es.reset_for_tests()


_STOP_ORDER = [
    "cancel_generations",
    "cancel_agent_runs",
    "stop_coding_run",
    "unload_models",
    "destroy_sandboxes",
    "stop_veilid_node",
]


# ---------------------------------------------------------------------------
# 1. The flag and the stop sequence
# ---------------------------------------------------------------------------

class TestFlag:
    def test_initially_unstopped(self, seam):
        assert _es.is_stopped() is False
        st = _es.status()
        assert st["stopped"] is False
        assert st["since"] is None
        assert st["last_stop"] is None

    def test_stop_sets_flag_and_status(self, seam):
        seam()
        result = _es.stop(user_id="leon")
        assert result["stopped"] is True
        assert _es.is_stopped() is True
        st = _es.status()
        assert st["stopped"] is True
        assert st["by"] == "leon"
        assert st["since"] is not None
        assert st["last_stop"]["failed_steps"] == []

    def test_flag_is_set_before_the_first_step_runs(self, seam):
        s = seam()
        _es.stop()
        # The first primitive observed the flag already set: admission
        # closed BEFORE the drain began (fail-secure by construction).
        assert s.flag_probe and s.flag_probe[0] is True

    def test_refusal_payload_shape(self, seam):
        seam()
        _es.stop()
        payload = _es.refusal_payload()
        assert payload["error"] == "emergency_stopped"
        assert "refused" in payload["message"]
        assert payload["since"] is not None

    def test_guard_http_noop_when_unstopped(self, seam):
        seam()
        _es.guard_http()  # must not raise

    def test_guard_http_503_when_stopped(self, seam):
        seam()
        _es.stop()
        with pytest.raises(fastapi.HTTPException) as exc:
            _es.guard_http()
        assert exc.value.status_code == 503
        assert exc.value.detail["error"] == "emergency_stopped"

    def test_reset_for_tests_restores_pristine(self, seam):
        seam()
        _es.stop()
        _es.reset_for_tests()
        assert _es.is_stopped() is False
        assert _es.status()["last_stop"] is None


class TestStopSequence:
    def test_step_order(self, seam):
        seam()
        result = _es.stop()
        assert [s["step"] for s in result["steps"]] == _STOP_ORDER

    def test_primitive_call_order(self, seam):
        s = seam()
        _es.stop()
        assert s.calls == [
            "cancel:ExecA",
            "cancel:ExecB",
            "agent.cancel",
            "coding.stop",
            "unload:ollama",
            "unload:llama_cpp",
            "sandbox.list",
            "sandbox.destroy:ws-a",
            "sandbox.destroy:ws-b",
            "node.stop",
        ]

    def test_all_steps_ok_on_the_happy_path(self, seam):
        seam()
        result = _es.stop()
        assert result["failed_steps"] == []
        assert all(s["ok"] for s in result["steps"])

    @pytest.mark.parametrize(
        "kw,failed_step,later_call",
        [
            ({"agent_fail": True}, "cancel_agent_runs", "coding.stop"),
            ({"coding_fail": True}, "stop_coding_run", "unload:ollama"),
            ({"sandbox_list_fail": True}, "destroy_sandboxes", "node.stop"),
            ({"node_stop_fail": True}, "stop_veilid_node", None),
        ],
    )
    def test_per_step_fail_tolerance(self, seam, kw, failed_step, later_call):
        s = seam(**kw)
        result = _es.stop()
        assert result["failed_steps"] == [failed_step]
        if later_call is not None:
            assert later_call in s.calls  # the sequence continued
        assert _es.is_stopped() is True  # fail-secure on the flag

    def test_inner_tolerance_one_executor_failing(self, seam):
        s = seam(exec_a_fail=True)
        result = _es.stop()
        # The second executor was still cancelled; the step records the
        # error without failing the whole step.
        assert "cancel:ExecB" in s.calls
        step = result["steps"][0]
        assert step["ok"] is True
        assert any("ExecA" in e for e in step["detail"]["errors"])
        assert step["detail"]["cancelled"] == ["ExecB"]

    def test_inner_tolerance_one_backend_failing(self, seam):
        s = seam(ollama_fail=True)
        result = _es.stop()
        assert "unload:llama_cpp" in s.calls
        unloaded = result["steps"][3]["detail"]["unloaded"]
        assert unloaded["ollama"].startswith("error:")
        assert unloaded["llama_cpp"] == 2

    def test_backend_without_unload_is_unsupported(self, seam):
        seam(llamacpp_supports=False)
        result = _es.stop()
        unloaded = result["steps"][3]["detail"]["unloaded"]
        assert unloaded["llama_cpp"] == "unsupported"

    def test_inner_tolerance_one_sandbox_failing(self, seam):
        s = seam(sandbox_fail_ids=("ws-a",))
        result = _es.stop()
        detail = result["steps"][4]["detail"]
        assert detail["destroyed"] == ["ws-b"]
        assert detail["failed"] == ["ws-a"]
        assert "sandbox.destroy:ws-b" in s.calls

    def test_node_not_running_is_not_stopped(self, seam):
        s = seam(node_running=False)
        result = _es.stop()
        assert "node.stop" not in s.calls
        assert result["steps"][5]["detail"]["was_running"] is False

    def test_node_stop_failure_keeps_was_running_for_resume(self, seam):
        seam(node_stop_fail=True)
        result = _es.stop()
        assert result["failed_steps"] == ["stop_veilid_node"]
        # The node was running; the resume must still attempt the restart.
        resumed = _es.resume()
        assert resumed["steps"][1]["detail"] == {"restarted": True}

    def test_idempotent_re_stop_reruns_the_drain(self, seam):
        s = seam()
        first = _es.stop()
        assert first["already_stopped"] is False
        n_calls = len(s.calls)
        second = _es.stop()
        assert second["already_stopped"] is True
        assert len(s.calls) == 2 * n_calls  # the drain ran again


class TestDropToBulbe:
    def test_default_does_not_escalate(self, seam):
        s = seam()
        result = _es.stop()
        assert "mode.escalate" not in s.calls
        assert [x["step"] for x in result["steps"]] == _STOP_ORDER

    def test_variant_escalates_after_the_drain(self, seam):
        s = seam()
        result = _es.stop(user_id="leon", drop_to_bulbe=True)
        assert s.calls[-1] == "mode.escalate"
        assert s.mode.last_user == "leon"
        assert [x["step"] for x in result["steps"]][-1] == "drop_to_bulbe"
        assert result["failed_steps"] == []

    def test_refused_escalation_is_a_failed_step_flag_still_set(self, seam):
        seam(bulbe_success=False)
        result = _es.stop(drop_to_bulbe=True)
        assert result["failed_steps"] == ["drop_to_bulbe"]
        assert _es.is_stopped() is True


class TestAuditRows:
    def test_stop_row_warning_with_per_step_outcomes(self, seam):
        s = seam(coding_fail=True)
        _es.stop(user_id="leon", drop_to_bulbe=True)
        assert len(s.chain_rows) == 1
        row = s.chain_rows[0]
        assert row["action"] == "stop"
        assert row["severity"] == "WARNING"
        assert row["user_id"] == "leon"
        assert row["drop_to_bulbe"] is True
        assert row["failed_steps"] == ["stop_coding_run"]
        assert [x["step"] for x in row["steps"]][:6] == _STOP_ORDER

    def test_resume_row_info(self, seam):
        s = seam()
        _es.stop()
        _es.resume(user_id="leon", warmup_model="m1")
        row = s.chain_rows[-1]
        assert row["action"] == "resume"
        assert row["severity"] == "INFO"
        assert row["warmup_model"] == "m1"
        assert row["failed_steps"] == []


# ---------------------------------------------------------------------------
# 2. Resume
# ---------------------------------------------------------------------------

class TestResume:
    def test_clears_the_flag(self, seam):
        seam()
        _es.stop()
        result = _es.resume()
        assert result["stopped"] is False
        assert result["was_stopped"] is True
        assert _es.is_stopped() is False

    def test_resume_when_not_stopped_is_honest(self, seam):
        seam()
        result = _es.resume()
        assert result["was_stopped"] is False

    def test_reconnect_reports_healthy(self, seam):
        seam()
        _es.stop()
        result = _es.resume()
        detail = result["steps"][0]["detail"]
        assert detail["backend"] == "ollama"
        assert detail["healthy"] is True

    def test_reconnect_reports_unreachable_honestly(self, seam):
        seam(ollama_healthy=RuntimeError("down"))
        _es.stop()
        result = _es.resume()
        detail = result["steps"][0]["detail"]
        assert detail["healthy"] is False
        assert "health check failed" in detail["detail"]
        assert result["steps"][0]["ok"] is True  # a finding, not a crash

    def test_warmup_called_when_model_given_and_healthy(self, seam):
        s = seam()
        _es.stop()
        result = _es.resume(warmup_model="qwen3:8b")
        assert s.warmer.last_model == "qwen3:8b"
        assert result["steps"][0]["detail"]["warmup"]["success"] is True

    def test_warmup_skipped_when_unhealthy(self, seam):
        s = seam(ollama_healthy=False)
        _es.stop()
        result = _es.resume(warmup_model="qwen3:8b")
        assert s.warmer.last_model is None
        assert "skipped" in result["steps"][0]["detail"]["warmup"]

    def test_no_warmup_without_model(self, seam):
        s = seam()
        _es.stop()
        result = _es.resume()
        assert s.warmer.last_model is None
        assert "warmup" not in result["steps"][0]["detail"]

    def test_veilid_restarted_only_if_was_running(self, seam):
        s = seam(node_running=False)
        _es.stop()
        result = _es.resume()
        assert "node.start" not in s.calls
        detail = result["steps"][1]["detail"]
        assert detail["restarted"] is False
        assert "not running at stop" in detail["reason"]

    def test_veilid_restart_happy_path(self, seam):
        s = seam()
        _es.stop()
        result = _es.resume()
        assert "node.start" in s.calls
        assert result["steps"][1]["detail"] == {"restarted": True}

    def test_veilid_bulbe_refusal_reported_not_raised(self, seam, monkeypatch):
        boom = RuntimeError("bulbe gate")
        seam(node_start_exc=boom)
        monkeypatch.setattr(_es, "_is_bulbe_refusal", lambda exc: exc is boom)
        _es.stop()
        result = _es.resume()
        step = result["steps"][1]
        assert step["ok"] is True
        assert step["detail"]["restarted"] is False
        assert "bulbe" in step["detail"]["reason"]
        assert result["failed_steps"] == []

    def test_veilid_other_start_error_is_a_failed_step(self, seam, monkeypatch):
        seam(node_start_exc=RuntimeError("wire down"))
        monkeypatch.setattr(_es, "_is_bulbe_refusal", lambda exc: False)
        _es.stop()
        result = _es.resume()
        assert result["failed_steps"] == ["restart_veilid_node"]
        assert _es.is_stopped() is False  # the flag stays cleared

    def test_second_resume_does_not_restart_again(self, seam):
        s = seam()
        _es.stop()
        _es.resume()
        n = s.calls.count("node.start")
        result = _es.resume()
        assert s.calls.count("node.start") == n  # was_running consumed
        assert result["steps"][1]["detail"]["restarted"] is False


# ---------------------------------------------------------------------------
# 3. The inference unload primitives (S215 + the IB-04 pick-up)
# ---------------------------------------------------------------------------

_ib = _load_fresh(
    os.path.join("opti_oignon", "inference_backend.py"),
    register="opti_oignon.inference_backend",
)


class _FakeOllamaLib:
    def __init__(self, ps_response, fail_models=()):
        self._ps = ps_response
        self._fail = set(fail_models)
        self.generated: list = []

    def ps(self):
        if isinstance(self._ps, Exception):
            raise self._ps
        return self._ps

    def generate(self, model, keep_alive):
        if model in self._fail:
            raise RuntimeError(f"stuck {model}")
        self.generated.append((model, keep_alive))


class TestOllamaUnloadAll:
    def _backend(self, monkeypatch, lib):
        monkeypatch.setattr(_ib, "OLLAMA_AVAILABLE", True)
        monkeypatch.setattr(_ib, "_ollama_module", lib)
        return _ib.OllamaBackend()

    def test_dict_form_evicts_each_with_keep_alive_zero(self, monkeypatch):
        lib = _FakeOllamaLib({"models": [{"name": "a"}, {"model": "b"}]})
        backend = self._backend(monkeypatch, lib)
        assert backend.unload_all() == 2
        assert lib.generated == [("a", 0), ("b", 0)]

    def test_object_form_handled_cc01(self, monkeypatch):
        ps = types.SimpleNamespace(models=[
            types.SimpleNamespace(name=None, model="m1"),
            types.SimpleNamespace(name="m2", model=None),
        ])
        lib = _FakeOllamaLib(ps)
        backend = self._backend(monkeypatch, lib)
        assert backend.unload_all() == 2
        assert [m for m, _ in lib.generated] == ["m1", "m2"]

    def test_ps_failure_returns_zero(self, monkeypatch):
        lib = _FakeOllamaLib(RuntimeError("no daemon"))
        backend = self._backend(monkeypatch, lib)
        assert backend.unload_all() == 0

    def test_one_stuck_model_does_not_block_the_rest(self, monkeypatch):
        lib = _FakeOllamaLib(
            {"models": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
            fail_models=("b",),
        )
        backend = self._backend(monkeypatch, lib)
        assert backend.unload_all() == 2
        assert [m for m, _ in lib.generated] == ["a", "c"]

    def test_unavailable_lib_returns_zero(self, monkeypatch):
        monkeypatch.setattr(_ib, "OLLAMA_AVAILABLE", False)
        backend = _ib.OllamaBackend()
        assert backend.unload_all() == 0


class TestLlamaCppUnloadIB04:
    def test_unload_model_pop_based_absent_is_false(self):
        backend = _ib.LlamaCppBackend()
        assert backend.unload_model("ghost") is False  # no KeyError

    def test_unload_model_present_is_true_and_removed(self):
        backend = _ib.LlamaCppBackend()
        backend._loaded_models["m"] = object()
        assert backend.unload_model("m") is True
        assert "m" not in backend._loaded_models

    def test_unload_all_counts_and_empties(self):
        backend = _ib.LlamaCppBackend()
        backend._loaded_models.update({"a": 1, "b": 2})
        assert backend.unload_all() == 2
        assert backend._loaded_models == {}

    def test_source_has_no_in_then_del(self):
        src = _read(os.path.join(_OO, "inference_backend.py"))
        body = src.split("def unload_model", 1)[1].split("def unload_all", 1)[0]
        assert "del self._loaded_models" not in body
        assert ".pop(" in body


class TestRegistryBackends:
    def test_backends_snapshot(self):
        registry = _ib.BackendRegistry()
        backend = _ib.OllamaBackend()
        registry.register(backend)
        snapshot = registry.backends()
        assert snapshot == [backend]
        snapshot.append("junk")  # mutating the snapshot must not leak
        assert registry.backends() == [backend]


# ---------------------------------------------------------------------------
# 4. routes_security: the estop endpoints and their auth/status ladders
# ---------------------------------------------------------------------------

def _fake_estop_module():
    mod = types.ModuleType("opti_oignon.emergency_stop")
    mod.calls = []
    mod._stopped = False

    def status():
        return {
            "stopped": mod._stopped,
            "since": None,
            "by": "",
            "last_stop": None,
            "last_resume": None,
        }

    def stop(user_id="", drop_to_bulbe=False):
        mod.calls.append(("stop", user_id, drop_to_bulbe))
        mod._stopped = True
        return {
            "stopped": True,
            "already_stopped": False,
            "since": 1.0,
            "drop_to_bulbe": drop_to_bulbe,
            "steps": [],
            "failed_steps": [],
        }

    def resume(user_id="", warmup_model=None):
        mod.calls.append(("resume", user_id, warmup_model))
        mod._stopped = False
        return {
            "stopped": False,
            "was_stopped": True,
            "steps": [],
            "failed_steps": [],
        }

    def is_stopped():
        return mod._stopped

    def refusal_payload():
        return {
            "error": "emergency_stopped",
            "message": (
                "Emergency stop is engaged: new work is refused until resume."
            ),
            "since": 1.0,
        }

    def guard_http():
        if mod._stopped:
            raise fastapi.HTTPException(
                status_code=503, detail=refusal_payload())

    mod.status = status
    mod.stop = stop
    mod.resume = resume
    mod.is_stopped = is_stopped
    mod.refusal_payload = refusal_payload
    mod.guard_http = guard_http
    return mod


def _fake_routes_auth():
    mod = types.ModuleType("opti_oignon.api.routes_auth")

    def _get_current_user(request: fastapi.Request):
        if request.headers.get("x-test-auth") != "ok":
            raise fastapi.HTTPException(
                status_code=401, detail="Authentication required")
        return {"user_id": "tester", "username": "tester"}

    mod._get_current_user = _get_current_user
    return mod


_RS_NONE_BINDS = (
    "opti_oignon.security_mode",
    "opti_oignon.plugin_allowlist",
    "opti_oignon.db_encryption",
    "opti_oignon.search_killswitch",
    "opti_oignon.auth_2fa",
    "opti_oignon.tool_call_approval",
    "opti_oignon.pqc_signatures",
    "opti_oignon.signed_audit_log",
)

_RS_CACHE: dict = {}


def _load_routes_security():
    if "mod" in _RS_CACHE:
        return _RS_CACHE["mod"], _RS_CACHE["estop"]
    estop_fake = _fake_estop_module()
    binds: dict = {
        "opti_oignon.api.routes_auth": _fake_routes_auth(),
        "opti_oignon.emergency_stop": estop_fake,
    }
    for name in _RS_NONE_BINDS:
        binds[name] = None
    mod = _load_fresh(
        os.path.join("opti_oignon", "api", "routes_security.py"),
        register="opti_oignon.api.routes_security",
        bind=binds,
    )
    _RS_CACHE["mod"] = mod
    _RS_CACHE["estop"] = estop_fake
    return mod, estop_fake


_AUTH = {"x-test-auth": "ok"}


@pytest.fixture()
def security_api():
    mod, estop_fake = _load_routes_security()
    estop_fake._stopped = False
    estop_fake.calls.clear()
    app = fastapi.FastAPI()
    app.include_router(mod.router)
    return mod, TestClient(app), estop_fake


class TestEstopRoutes:
    def test_status_requires_auth(self, security_api):
        _, client, _ = security_api
        assert client.get("/api/security/estop").status_code == 401

    def test_engage_requires_auth(self, security_api):
        _, client, _ = security_api
        assert client.post(
            "/api/security/estop", json={}).status_code == 401

    def test_resume_requires_auth(self, security_api):
        _, client, _ = security_api
        assert client.post(
            "/api/security/estop/resume", json={}).status_code == 401

    def test_status_shape(self, security_api):
        _, client, _ = security_api
        resp = client.get("/api/security/estop", headers=_AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is True
        assert body["stopped"] is False

    def test_engage_forwards_default(self, security_api):
        _, client, fake = security_api
        resp = client.post("/api/security/estop", json={}, headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json()["stopped"] is True
        assert fake.calls == [("stop", "admin", False)]

    def test_engage_forwards_drop_to_bulbe(self, security_api):
        _, client, fake = security_api
        resp = client.post(
            "/api/security/estop",
            json={"drop_to_bulbe": True},
            headers=_AUTH,
        )
        assert resp.status_code == 200
        assert fake.calls == [("stop", "admin", True)]

    def test_resume_forwards_warmup_model(self, security_api):
        _, client, fake = security_api
        resp = client.post(
            "/api/security/estop/resume",
            json={"warmup_model": "qwen3:8b"},
            headers=_AUTH,
        )
        assert resp.status_code == 200
        assert resp.json()["stopped"] is False
        assert fake.calls == [("resume", "admin", "qwen3:8b")]

    def test_resume_default_has_no_model(self, security_api):
        _, client, fake = security_api
        resp = client.post(
            "/api/security/estop/resume", json={}, headers=_AUTH)
        assert resp.status_code == 200
        assert fake.calls == [("resume", "admin", None)]

    def test_unavailable_status_is_honest(self, security_api, monkeypatch):
        mod, client, _ = security_api
        monkeypatch.setattr(mod, "EMERGENCY_STOP_AVAILABLE", False)
        resp = client.get("/api/security/estop", headers=_AUTH)
        assert resp.status_code == 200
        assert resp.json() == {"available": False, "stopped": False}

    def test_unavailable_posts_answer_503(self, security_api, monkeypatch):
        mod, client, _ = security_api
        monkeypatch.setattr(mod, "EMERGENCY_STOP_AVAILABLE", False)
        assert client.post(
            "/api/security/estop", json={}, headers=_AUTH
        ).status_code == 503
        assert client.post(
            "/api/security/estop/resume", json={}, headers=_AUTH
        ).status_code == 503


# ---------------------------------------------------------------------------
# 5. The admission guard, behavioural (routes_code as the representative)
# ---------------------------------------------------------------------------

_SCHEMAS = _load_fresh(
    os.path.join("opti_oignon", "api", "schemas.py"),
    register="opti_oignon.api.schemas",
)


def _load_routes_code(estop_bind):
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.CODE_EXECUTOR_AVAILABLE = False
    deps.code_executor = None
    return _load_fresh(
        os.path.join("opti_oignon", "api", "routes_code.py"),
        register="opti_oignon.api.routes_code",
        bind={
            "opti_oignon.api.deps": deps,
            "opti_oignon.api.schemas": _SCHEMAS,
            "opti_oignon.emergency_stop": estop_bind,
        },
    )


class TestAdmissionBehaviour:
    def test_stopped_system_answers_503_refused_not_hung(self):
        fake = _fake_estop_module()
        fake._stopped = True
        mod = _load_routes_code(fake)
        app = fastapi.FastAPI()
        app.include_router(mod.router)
        resp = TestClient(app).post(
            "/api/code/execute", json={"code": "print(1)"})
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "emergency_stopped"

    def test_unstopped_request_passes_the_guard(self):
        fake = _fake_estop_module()
        mod = _load_routes_code(fake)
        app = fastapi.FastAPI()
        app.include_router(mod.router)
        resp = TestClient(app).post(
            "/api/code/execute", json={"code": "print(1)"})
        # The guard passed; the request reached the availability check.
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Code executor module not available"

    def test_guard_module_absent_fails_open(self):
        # An availability control, not a security boundary: with the seam
        # module absent the entries keep working (documented posture).
        mod = _load_routes_code(None)
        assert mod._emergency_stop is None
        app = fastapi.FastAPI()
        app.include_router(mod.router)
        resp = TestClient(app).post(
            "/api/code/execute", json={"code": "print(1)"})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Code executor module not available"


# ---------------------------------------------------------------------------
# 6. The admission guard, by source, across every arbitrated site
# ---------------------------------------------------------------------------

_GUARD_CALL = "_emergency_stop.guard_http()"
_GUARD_IMPORT = "from opti_oignon import emergency_stop as _emergency_stop"

_REST_GUARD_SITES = [
    ("routes_agent.py", "agent_run"),
    ("routes_coding.py", "start_coding_task"),
    ("routes_coding.py", "execute_next_step"),
    ("routes_coding.py", "execute_all_steps_background"),
    ("routes_coding.py", "resume_task"),
    ("routes_sandbox.py", "create_sandbox"),
    ("routes_sandbox.py", "execute_sandbox_tool"),
    ("routes_sandbox.py", "provision_workspace"),
    ("routes_benchmark.py", "start_llm_benchmark"),
    ("routes_code.py", "execute_code"),
    ("routes_sync.py", "sync_run"),
]

_GUARDED_FILES = sorted({name for name, _ in _REST_GUARD_SITES}) + [
    "routes_chat.py",
]


def _handler_body(src: str, name: str) -> str:
    match = re.search(rf"\n(\s*)(?:async )?def {name}\(", src)
    assert match, f"handler {name} not found"
    indent = match.group(1)
    start = match.end()
    nxt = re.search(
        rf"\n{indent}(?:@|def |async def |class )", src[start:])
    return src[start:start + (nxt.start() if nxt else len(src) - start)]


class TestAdmissionBySource:
    @pytest.mark.parametrize("fname,handler", _REST_GUARD_SITES)
    def test_rest_handler_carries_the_guard(self, fname, handler):
        src = _read(os.path.join(_API, fname))
        body = _handler_body(src, handler)
        assert _GUARD_CALL in body, f"{fname}:{handler} misses the guard"

    @pytest.mark.parametrize("fname", _GUARDED_FILES)
    def test_guarded_import_present(self, fname):
        src = _read(os.path.join(_API, fname))
        assert _GUARD_IMPORT in src

    @pytest.mark.parametrize("handler", ["chat_stream", "chat_retry"])
    def test_chat_websockets_refuse_with_the_error_token(self, handler):
        src = _read(os.path.join(_API, "routes_chat.py"))
        body = _handler_body(src, handler)
        assert "_emergency_stop.is_stopped()" in body
        assert "refusal_payload()" in body
        # The refusal is an error token followed by an honest close.
        idx = body.index("refusal_payload()")
        tail = body[idx:idx + 200]
        assert "websocket.close()" in tail

    def test_estop_routes_registered_with_the_distinction_stated(self):
        src = _read(os.path.join(_API, "routes_security.py"))
        assert '@router.get("/estop")' in src
        assert '@router.post("/estop")' in src
        assert '@router.post("/estop/resume")' in src
        assert "distinct" in src and "kill switch" in src.lower()
        assert "no ceremony" in src.lower()


# ---------------------------------------------------------------------------
# 7. Read-only seams, module hygiene, AST validity
# ---------------------------------------------------------------------------

_READ_ONLY_SEAMS = [
    "sandbox_manager.py",
    "sandbox_egress.py",
    "auth.py",
    "auth_2fa.py",
    "security_mode.py",
]

_TOUCHED_PY = [
    os.path.join(_OO, "emergency_stop.py"),
    os.path.join(_OO, "inference_backend.py"),
    os.path.join(_API, "routes_security.py"),
    os.path.join(_API, "routes_chat.py"),
    os.path.join(_API, "routes_agent.py"),
    os.path.join(_API, "routes_coding.py"),
    os.path.join(_API, "routes_sandbox.py"),
    os.path.join(_API, "routes_benchmark.py"),
    os.path.join(_API, "routes_code.py"),
    os.path.join(_API, "routes_sync.py"),
]


class TestSeamDiscipline:
    @pytest.mark.parametrize("fname", _READ_ONLY_SEAMS)
    def test_read_only_modules_carry_no_estop_reference(self, fname):
        src = _read(os.path.join(_OO, fname))
        assert "emergency_stop" not in src

    def test_module_states_the_arbitrated_posture(self):
        doc = _es.__doc__ or ""
        assert "NOT a security boundary" in doc
        assert "kill switch" in doc
        assert "no ceremony" in doc.lower()
        assert "UNSTOPPED" in doc

    @pytest.mark.parametrize(
        "path", _TOUCHED_PY, ids=[os.path.basename(p) for p in _TOUCHED_PY])
    def test_ast_valid(self, path):
        ast.parse(_read(path))


# ---------------------------------------------------------------------------
# 8. UI and documentation registrations, by source
# ---------------------------------------------------------------------------

_COMPONENT = os.path.join(
    _FRONT, "components", "ui", "EmergencyStopControl.svelte")
_HEADER = os.path.join(_FRONT, "components", "layout", "Header.svelte")
_ESTOP_TS = os.path.join(_FRONT, "api", "estop.ts")


def _order_ok(src: str, needles: list[str]) -> bool:
    positions = [src.index(n) for n in needles]
    return positions == sorted(positions)


class TestFrontendRegistration:
    def test_component_exists_with_the_two_actions(self):
        src = _read(_COMPONENT)
        assert "Stop compute</" in src or ">\n\t\t\t\t\t\tStop compute" in src
        assert "Stop compute + Bulbe" in src
        assert "Resume" in src

    def test_component_is_two_step_and_announced(self):
        src = _read(_COMPONENT)
        assert "aria-expanded" in src  # step one: the control opens
        assert "confirming" in src  # step two: the explicit choice
        assert "aria-live" in src
        assert "Escape" in src

    def test_component_uses_the_estop_api(self):
        src = _read(_COMPONENT)
        assert "$lib/api/estop" in src
        for fn in (
            "getEmergencyStopStatus",
            "engageEmergencyStop",
            "resumeFromEmergencyStop",
        ):
            assert fn in src

    def test_component_token_discipline(self):
        src = _read(_COMPONENT)
        assert "var(--oo-" in src
        for match in re.finditer(r"#[0-9a-fA-F]{3,8}\b", src):
            window = src[max(0, match.start() - 60):match.start()]
            assert "var(--oo-" in window and window.rstrip().endswith(","), (
                f"raw hex outside a var(--oo-*, #fallback) form: "
                f"{match.group(0)!r}"
            )

    def test_component_block_balance(self):
        src = _read(_COMPONENT)
        assert len(re.findall(r"\{#if[\s}]", src)) == src.count("{/if}")

    def test_header_mounts_the_control_first(self):
        src = _read(_HEADER)
        assert "import EmergencyStopControl" in src
        assert "<EmergencyStopControl" in src
        assert _order_ok(src, ["<EmergencyStopControl", "<BackendStatus"])

    def test_header_legacy_cluster_order_intact(self):
        src = _read(_HEADER)
        assert _order_ok(src, [
            "<BackendStatus",
            "<ThemeSwitcher",
            "<NotificationCenter",
            "<UserMenu",
        ])

    def test_api_module_covers_the_three_endpoints(self):
        src = _read(_ESTOP_TS)
        assert src.count("'/api/security/estop'") == 2
        assert "'/api/security/estop/resume'" in src
        for fn in (
            "getEmergencyStopStatus",
            "engageEmergencyStop",
            "resumeFromEmergencyStop",
        ):
            assert f"export function {fn}" in src


class TestDocRegistration:
    def test_frontend_spec_row(self):
        src = _read(os.path.join(_ROOT, "FRONTEND_REDESIGN_SPEC.md"))
        row = next(
            (line for line in src.splitlines()
             if "EmergencyStopControl.svelte" in line),
            "",
        )
        assert row, "spec 11.3 row missing"
        assert "NEW" in row and "S215" in row

    def test_handoff_es01_row(self):
        src = _read(os.path.join(_ROOT, "SHAKEDOWN_S198_HANDOFF.md"))
        assert "ES-01 (S215)" in src
        idx = src.index("ES-01 (S215)")
        block = src[idx:idx + 1200]
        assert "ollama ps" in block
        assert "503" in block
        assert "Bulbe" in block
        assert "audit" in block
