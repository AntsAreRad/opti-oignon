#!/usr/bin/env python3
"""S225 -- Resource Governor Bloc 2: runtime backpressure (R-02, Section 5).

Container-provable coverage, the spec Section 11 Bloc 2 list verbatim:

- threshold crossings with faked snapshots (the in_use/effective ratio
  against soft/hard, the learned ceiling as the capacity when lower, the
  bounded refusal-rate window raising to soft, window pruning on the
  injected clock, estop refusals excluded from the window);
- the per-decision keep_alive fill (soft -> pressure_keep_alive; off
  pressure -> None, preserving the s224 pin scenario; the capacity-known
  guard) and override-then-restore on the warmup's existing settable
  property (sustained write once, restore at the first clear, fail-open
  on a raising setter or an absent warmup);
- the targeted evict calls the per-model idiom, asserted against a fake
  backend (oldest-idle first, only as many as the shortfall needs, the
  about-to-load model never self-evicted, audit append observed through
  a seeded signed_audit_log, every failure path open);
- queue bounds and re-admission on wake (depth bound immediate refusal,
  wait bound resolving to the caller's refusal, a freed-capacity wake
  admitting, the enqueue visible in the ring as the 4.4 "queue" action);
- estop-never-bypassed (a drain notification releases a waiter straight
  to the estop refusal while the clock is frozen, proving the exit is
  the estop path and not the deadline).

Host-assured (named, never simulated): real eviction latency; whether a
shortened keep_alive actually relieves pressure on the real card.

Isolation: the established spec_from_file_location idiom with
sys.modules pre-seeding (an ollama stub and an opti_oignon package stub
carrying a real __path__), order-independent; collaborator seams
(emergency_stop, context_manager) are seeded per-test through
monkeypatch.setitem so standalone and in-sweep runs resolve identically.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_BASE = Path(__file__).resolve().parent.parent
_MODULE_PATH = _BASE / "opti_oignon" / "resource_governor.py"
_IB_PATH = _BASE / "opti_oignon" / "inference_backend.py"
_EXEC_PATH = _BASE / "opti_oignon" / "executor.py"
_BENCH_PATH = _BASE / "opti_oignon" / "benchmark_runner.py"
_SPEC_PATH = _BASE / "RESOURCE_GOVERNOR_SPEC.md"
_YAML_PATH = _BASE / "opti_oignon" / "config" / "resource_governor.yaml"

SRC = _MODULE_PATH.read_text(encoding="utf-8")
IB_SRC = _IB_PATH.read_text(encoding="utf-8")
EXEC_SRC = _EXEC_PATH.read_text(encoding="utf-8")
BENCH_SRC = _BENCH_PATH.read_text(encoding="utf-8")

GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Isolated module loading (the established idiom, mirrored from s223/s224)
# ---------------------------------------------------------------------------

sys.modules.setdefault("ollama", types.ModuleType("ollama"))

if "opti_oignon" not in sys.modules:
    _pkg = types.ModuleType("opti_oignon")
    _pkg.__path__ = [str(_BASE / "opti_oignon")]
    sys.modules["opti_oignon"] = _pkg


def _load_module(dotted: str, relpath: str):
    existing = sys.modules.get(dotted)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(dotted, str(_BASE / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


for _dotted, _rel in (
    ("opti_oignon.db_utils", "opti_oignon/db_utils.py"),
    ("opti_oignon.model_warmup", "opti_oignon/model_warmup.py"),
    ("opti_oignon.inference_backend", "opti_oignon/inference_backend.py"),
    ("opti_oignon.speculative_decoding", "opti_oignon/speculative_decoding.py"),
):
    _load_module(_dotted, _rel)

rg = _load_module(
    "opti_oignon.resource_governor", "opti_oignon/resource_governor.py"
)
ib = sys.modules["opti_oignon.inference_backend"]


# ---------------------------------------------------------------------------
# Fakes (the s224 vocabulary, extended for Bloc 2)
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class KeepAliveWarmup:
    """Injectable S1 fake; keep_alive is a plain instance attribute the
    governor reads (idle derivation) AND writes (the Bloc 2 sustained
    override through the existing settable surface)."""

    def __init__(self, models=None, keep_alive: str = "30m"):
        self.calls = 0
        self.models = list(models or [])
        self.keep_alive = keep_alive

    def get_loaded_models(self):
        self.calls += 1
        return list(self.models)


class RaisingSetterWarmup:
    """keep_alive readable but the setter raises: the fail-open proof."""

    def __init__(self, models=None):
        self.models = list(models or [])

    def get_loaded_models(self):
        return list(self.models)

    @property
    def keep_alive(self) -> str:
        return "30m"

    @keep_alive.setter
    def keep_alive(self, value: str):
        raise RuntimeError("setter forbidden in this fake")


class FakeEstop:
    """is_stopped() answers from a scripted sequence (last value sticky);
    refusal_payload() is the fixed S215-shaped body. Tests may mutate
    ``_seq`` mid-flight to flip the flag."""

    def __init__(self, sequence=(False,)):
        self._seq = list(sequence)
        self.payload = {
            "error": "emergency_stopped",
            "message": "Emergency stop engaged. Resume from Security.",
            "since": 1234.5,
        }

    def is_stopped(self):
        if len(self._seq) > 1:
            return self._seq.pop(0)
        return bool(self._seq[0])

    def refusal_payload(self):
        return dict(self.payload)


class FakeBackend:
    """Records per-model unload calls; scripted result or raise."""

    def __init__(self, name="fake", result=True, raise_on=None):
        self.name = name
        self.result = result
        self.raise_on = set(raise_on or [])
        self.calls: list[str] = []

    def unload_model(self, model_name: str) -> bool:
        self.calls.append(model_name)
        if model_name in self.raise_on:
            raise RuntimeError("unload exploded")
        return self.result


class NoUnloadBackend:
    name = "no-unload"


class FakeRegistry:
    def __init__(self, backend_list):
        self._list = list(backend_list)

    def backends(self):
        return list(self._list)


def _fake_cm(window=32768, max_output=4096):
    return SimpleNamespace(
        get_model_limits=lambda model: SimpleNamespace(
            context_window=window, max_output=max_output
        )
    )


def _loaded(name, size_gb, expires_in_s=None, digest=None):
    expires = None if expires_in_s is None else time.time() + expires_in_s
    return SimpleNamespace(
        name=name,
        size_vram=int(size_gb * GB),
        expires_at=expires,
        context_length=None,
        digest=digest,
    )


def _meminfo_file(
    tmp_path: Path,
    available_kb: int = 8 * 1024 * 1024,
    name: str = "meminfo",
) -> Path:
    p = tmp_path / name
    p.write_text(
        f"MemTotal: {available_kb * 2} kB\nMemAvailable: {available_kb} kB\n",
        encoding="utf-8",
    )
    return p


def _write_yaml(tmp_path: Path, content: str, name: str = "gov.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


_BASE_YAML = (
    "total_vram_gb: 10\n"
    "safety_margin_gb: 1.0\n"
    "kv_coefficient: 0.5\n"
    "idle_evict_threshold_s: 600\n"
)

_QUEUE_YAML = _BASE_YAML + (
    "queue:\n"
    "  enabled_per_caller:\n"
    "    benchmark: true\n"
    "  depth: 2\n"
    "  wait_s: 30\n"
)


# ---------------------------------------------------------------------------
# Fixtures (mirroring the s224 factory contract)
# ---------------------------------------------------------------------------


@pytest.fixture()
def seams(monkeypatch):
    estop = FakeEstop((False,))
    monkeypatch.setitem(sys.modules, "opti_oignon.emergency_stop", estop)
    monkeypatch.setitem(sys.modules, "opti_oignon.context_manager", _fake_cm())
    return estop


@pytest.fixture()
def gov_factory(tmp_path, seams):
    """ResourceGovernor factory: capacity 10 GB, margin 1, kv 0.5 GiB per
    1024 tokens, idle threshold 600 s, tmp DB, deterministic meminfo."""

    counter = {"n": 0}

    def make(yaml_text=None, **kw):
        counter["n"] += 1
        if yaml_text is None:
            yaml_text = _BASE_YAML
        kw.setdefault(
            "config_path",
            _write_yaml(tmp_path, yaml_text, f"gov_{counter['n']}.yaml"),
        )
        kw.setdefault("db_path", tmp_path / f"gov_{counter['n']}.db")
        kw.setdefault("warmup", None)
        kw.setdefault("registry", None)
        kw.setdefault("clock", FakeClock())
        kw.setdefault("meminfo_path", _meminfo_file(tmp_path))
        return rg.ResourceGovernor(**kw)

    return make


@pytest.fixture()
def singleton_guard():
    rg.reset_resource_governor()
    yield
    rg.reset_resource_governor()


def _release_waiter(gov, thread):
    """Always-release helper: jump the deadline, wake, join."""
    try:
        gov._clock.advance(10_000.0)
    except Exception:
        pass
    gov.invalidate_on_evict()
    thread.join(timeout=5.0)
    assert not thread.is_alive()


def _start_waiter(gov, model, ctx, caller="benchmark"):
    box: dict = {}

    def run():
        box["decision"] = gov.admit_or_wait(model, ctx, caller=caller)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def _wait_for_depth(gov, n, timeout=5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if gov.queue_depth == n:
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# A. Config keys (Section 10, the two S225 additions)
# ---------------------------------------------------------------------------


class TestConfigKeys:
    def test_defaults_carry_the_bloc2_windows(self):
        cfg = rg.GovernorConfig()
        assert cfg.pressure_sustain_s == 60.0
        assert cfg.pressure_refusal_window_s == 60.0

    def test_loader_parses_the_pressure_subkeys(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            "pressure:\n  sustain_s: 12\n  refusal_window_s: 7\n",
        )
        cfg = rg.load_config(p)
        assert cfg.pressure_sustain_s == 12.0
        assert cfg.pressure_refusal_window_s == 7.0

    def test_shipped_yaml_still_mirrors_defaults(self):
        # The s223 mirror contract survives the two additive keys.
        assert rg.load_config(_YAML_PATH) == rg.GovernorConfig()


# ---------------------------------------------------------------------------
# B. The pressure signal: threshold crossings on faked snapshots
# ---------------------------------------------------------------------------


class TestPressureSignal:
    def test_below_soft_is_none(self, gov_factory):
        gov = gov_factory(
            warmup=KeepAliveWarmup(models=[_loaded("m", 4.0, 1800)])
        )
        state = gov.pressure_state()
        assert state["level"] == "none"
        assert state["ratio"] == pytest.approx(0.4)

    def test_soft_threshold_crossing(self, gov_factory):
        gov = gov_factory(
            warmup=KeepAliveWarmup(models=[_loaded("m", 9.0, 1800)])
        )
        state = gov.pressure_state()
        assert state["level"] == "soft"
        assert state["ratio"] == pytest.approx(0.9)

    def test_hard_threshold_crossing(self, gov_factory):
        gov = gov_factory(
            warmup=KeepAliveWarmup(models=[_loaded("m", 9.6, 1800)])
        )
        state = gov.pressure_state()
        assert state["level"] == "hard"
        assert state["ratio"] == pytest.approx(0.96)

    def test_learned_ceiling_is_the_capacity_when_lower(self, gov_factory):
        gov = gov_factory(
            warmup=KeepAliveWarmup(models=[_loaded("m", 4.5, 1800)])
        )
        # A load failure at 6.0 GB learns ceiling 5.0 (6.0 - margin 1.0).
        assert gov.record_load_failure(6.0) == pytest.approx(5.0)
        gov.refresh(force=True)
        state = gov.pressure_state()
        assert state["effective_capacity_gb"] == pytest.approx(5.0)
        assert state["ratio"] == pytest.approx(0.9)
        assert state["level"] == "soft"

    def test_refusal_rate_raises_to_soft(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        for _ in range(3):
            d = gov.admit("huge", 32768, caller="benchmark")
            assert d.admitted is False
        state = gov.pressure_state()
        assert state["decisions_in_window"] == 3
        assert state["refusals_in_window"] == 3
        assert state["refusal_rate"] == pytest.approx(1.0)
        assert state["level"] == "soft"

    def test_refusal_rate_never_reaches_hard_alone(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        for _ in range(5):
            gov.admit("huge", 32768, caller="benchmark")
        assert gov.pressure_state()["level"] == "soft"

    def test_two_refusals_are_below_the_minimum(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        for _ in range(2):
            gov.admit("huge", 32768, caller="benchmark")
        state = gov.pressure_state()
        assert state["decisions_in_window"] == 2
        assert state["level"] == "none"

    def test_window_prunes_on_the_injected_clock(self, gov_factory):
        clock = FakeClock()
        gov = gov_factory(warmup=KeepAliveWarmup(), clock=clock)
        for _ in range(3):
            gov.admit("huge", 32768, caller="benchmark")
        assert gov.pressure_state()["level"] == "soft"
        clock.advance(61.0)
        state = gov.pressure_state()
        assert state["decisions_in_window"] == 0
        assert state["level"] == "none"

    def test_estop_refusals_never_enter_the_window(self, gov_factory, seams):
        gov = gov_factory(warmup=KeepAliveWarmup())
        seams._seq = [True]
        d = gov.admit("m", 4096, caller="chat")
        assert d.is_estop is True
        seams._seq = [False]
        state = gov.pressure_state()
        assert state["decisions_in_window"] == 0

    def test_capacity_unknown_ratio_is_none(self, gov_factory):
        gov = gov_factory(
            yaml_text="kv_coefficient: 0.5\n", warmup=KeepAliveWarmup()
        )
        state = gov.pressure_state()
        assert state["ratio"] is None
        assert state["level"] == "none"


# ---------------------------------------------------------------------------
# C. The per-decision keep_alive fill (Section 5, escalation step 1)
# ---------------------------------------------------------------------------


class TestKeepAliveDecisionFill:
    def test_soft_pressure_fills_the_override(self, gov_factory):
        gov = gov_factory(
            warmup=KeepAliveWarmup(models=[_loaded("m", 8.6, 1800)])
        )
        d = gov.admit("m", None, caller="chat")
        assert d.admitted is True
        assert d.keep_alive == "5m"

    def test_no_pressure_keeps_none(self, gov_factory):
        # The s224 pin scenario, re-stated: off pressure the field is
        # untouched, so the Bloc 1 pin survives by construction.
        gov = gov_factory(warmup=KeepAliveWarmup())
        d = gov.admit("m", 4096, caller="chat")
        assert d.admitted is True
        assert d.keep_alive is None

    def test_refusals_carry_no_override(self, gov_factory):
        gov = gov_factory(
            warmup=KeepAliveWarmup(models=[_loaded("m", 9.0, 1800)])
        )
        d = gov.admit("huge", 32768, caller="benchmark")
        assert d.admitted is False
        assert d.keep_alive is None

    def test_capacity_unknown_guard_keeps_none(self, gov_factory, tmp_path):
        # Refusal-rate soft with capacity unknown: the fill stays off
        # (the DI-S225-4 capacity-known guard).
        gov = gov_factory(
            yaml_text="kv_coefficient: 0.5\nsafety_margin_gb: 1.0\n",
            warmup=KeepAliveWarmup(),
            meminfo_path=_meminfo_file(tmp_path, available_kb=1024 * 1024),
        )
        gov.store.record_model_cost("heavy", None, 16 * GB, None)
        for _ in range(3):
            d = gov.admit("heavy", None, caller="benchmark")
            assert d.admitted is False and d.reason == "ram_insufficient"
        assert gov.pressure_state()["level"] == "soft"
        d = gov.admit("light", 1024, caller="chat")
        assert d.admitted is True
        assert d.reason == "capacity_unknown_fail_open"
        assert d.keep_alive is None


# ---------------------------------------------------------------------------
# D. Override-then-restore on the warmup property (the sustained half)
# ---------------------------------------------------------------------------


class TestSustainedOverrideRestore:
    def test_write_after_sustain_then_restore_on_clear(self, gov_factory):
        clock = FakeClock()
        warmup = KeepAliveWarmup(models=[_loaded("m", 9.0, 1800)])
        gov = gov_factory(warmup=warmup, clock=clock)
        # First soft observation seeds the timer; no write yet.
        assert gov.pressure_state()["level"] == "soft"
        assert warmup.keep_alive == "30m"
        # Sustained past the window: written once, original remembered.
        clock.advance(61.0)
        state = gov.pressure_state()
        assert state["level"] == "soft"
        assert warmup.keep_alive == "5m"
        assert state["keep_alive_overridden"] is True
        # Still soft: no rewrite, the original stays remembered.
        clock.advance(1.0)
        gov.pressure_state()
        assert warmup.keep_alive == "5m"
        assert gov._keep_alive_original == "30m"
        # Pressure clears: restored at the first none observation.
        warmup.models = []
        gov.invalidate_on_evict()
        state = gov.pressure_state()
        assert state["level"] == "none"
        assert warmup.keep_alive == "30m"
        assert state["keep_alive_overridden"] is False
        assert gov._pressure_soft_since is None

    def test_original_equal_to_override_is_not_written(self, gov_factory):
        clock = FakeClock()
        warmup = KeepAliveWarmup(
            models=[_loaded("m", 9.0, 1800)], keep_alive="5m"
        )
        gov = gov_factory(warmup=warmup, clock=clock)
        gov.pressure_state()
        clock.advance(61.0)
        state = gov.pressure_state()
        assert warmup.keep_alive == "5m"
        assert state["keep_alive_overridden"] is False

    def test_raising_setter_fails_open(self, gov_factory):
        clock = FakeClock()
        warmup = RaisingSetterWarmup(models=[_loaded("m", 9.0, 1800)])
        gov = gov_factory(warmup=warmup, clock=clock)
        gov.pressure_state()
        clock.advance(61.0)
        state = gov.pressure_state()  # must not raise
        assert state["keep_alive_overridden"] is False
        assert warmup.keep_alive == "30m"

    def test_absent_warmup_is_a_no_op(self, gov_factory):
        clock = FakeClock()
        gov = gov_factory(warmup=None, clock=clock)
        for _ in range(3):
            gov.admit("huge", 32768, caller="benchmark")
        assert gov.pressure_state()["level"] == "soft"
        clock.advance(61.0)
        state = gov.pressure_state()  # must not raise
        assert state["keep_alive_overridden"] is False


# ---------------------------------------------------------------------------
# E. Targeted eviction: the per-model idiom against fake backends
# ---------------------------------------------------------------------------


class TestTargetedEviction:
    def test_evict_model_calls_the_idiom_and_invalidates(self, gov_factory):
        backend = FakeBackend()
        gov = gov_factory(
            warmup=KeepAliveWarmup(models=[_loaded("victim", 5.0, 60)]),
            registry=FakeRegistry([backend]),
        )
        gov.refresh(force=True)
        assert gov.evict_model("victim") is True
        assert backend.calls == ["victim"]
        assert gov._snapshot is None  # invalidate_on_evict fired

    def test_evict_audit_appends_off_the_hot_path(
        self, gov_factory, monkeypatch
    ):
        captured: dict = {}
        done = threading.Event()
        fake_log = types.ModuleType("opti_oignon.signed_audit_log")

        def chain_log(**kwargs):
            captured.update(kwargs)
            done.set()
            return 1

        fake_log.chain_log = chain_log
        monkeypatch.setitem(
            sys.modules, "opti_oignon.signed_audit_log", fake_log
        )
        gov = gov_factory(registry=FakeRegistry([FakeBackend()]))
        assert gov.evict_model("victim", trigger="manual") is True
        assert done.wait(timeout=2.0), "audit append never ran"
        assert captured["event_type"] == "resource_governor"
        assert captured["source"] == "resource_governor"
        assert captured["action"] == "evict_model"
        assert captured["model"] == "victim"
        assert captured["trigger"] == "manual"

    def test_unload_failure_fails_open(self, gov_factory):
        gov = gov_factory(
            registry=FakeRegistry([FakeBackend(raise_on={"victim"})])
        )
        assert gov.evict_model("victim") is False

    def test_backend_without_the_method_is_skipped(self, gov_factory):
        second = FakeBackend(name="second")
        gov = gov_factory(
            registry=FakeRegistry([NoUnloadBackend(), second])
        )
        assert gov.evict_model("victim") is True
        assert second.calls == ["victim"]

    def test_first_false_then_second_true(self, gov_factory):
        first = FakeBackend(name="first", result=False)
        second = FakeBackend(name="second", result=True)
        gov = gov_factory(registry=FakeRegistry([first, second]))
        assert gov.evict_model("victim") is True
        assert first.calls == ["victim"] and second.calls == ["victim"]

    def test_absent_registry_fails_open(self, gov_factory):
        gov = gov_factory(registry=None)
        assert gov.evict_model("victim") is False


class TestConditionalHonour:
    def _gov(self, gov_factory, backend, models):
        return gov_factory(
            warmup=KeepAliveWarmup(models=models),
            registry=FakeRegistry([backend]),
        )

    def test_oldest_idle_first_only_as_needed(self, gov_factory):
        backend = FakeBackend()
        # old1 idle 1740 s (5 GB), old2 idle 840 s (3 GB): both evictable,
        # in_use 8 -> budget 1; new at 4096 costs kv 2 -> needed 1 ->
        # the oldest alone covers it.
        gov = self._gov(
            gov_factory,
            backend,
            [_loaded("old1", 5.0, 60), _loaded("old2", 3.0, 960)],
        )
        d = gov.admit("new", 4096, caller="chat")
        assert d.admitted and d.conditional_on_eviction
        gov._honour_conditional_eviction(d)
        assert backend.calls == ["old1"]

    def test_walks_until_the_shortfall_is_covered(self, gov_factory):
        backend = FakeBackend()
        gov = self._gov(
            gov_factory,
            backend,
            [_loaded("old1", 5.0, 60), _loaded("old2", 3.0, 960)],
        )
        # 16384 costs kv 8 -> needed 7: old1 (5) is not enough, old2 joins.
        d = gov.admit("new", 16384, caller="chat")
        assert d.admitted and d.conditional_on_eviction
        gov._honour_conditional_eviction(d)
        assert backend.calls == ["old1", "old2"]

    def test_the_admitted_model_is_never_self_evicted(self, gov_factory):
        backend = FakeBackend()
        gov = self._gov(
            gov_factory,
            backend,
            [_loaded("old1", 5.0, 60), _loaded("old2", 3.0, 960)],
        )
        d = gov.admit("old1", 16384, caller="chat")
        assert d.admitted and d.conditional_on_eviction
        gov._honour_conditional_eviction(d)
        assert "old1" not in backend.calls
        assert backend.calls == ["old2"]

    def test_shortfall_gone_means_no_eviction(self, gov_factory):
        backend = FakeBackend()
        warmup = KeepAliveWarmup(models=[_loaded("old1", 5.0, 60)])
        gov = gov_factory(warmup=warmup, registry=FakeRegistry([backend]))
        d = gov.admit("new", 16384, caller="chat")
        assert d.conditional_on_eviction
        # The hog left between the grant and the load: nothing needed.
        warmup.models = []
        gov.invalidate_on_evict()
        gov._honour_conditional_eviction(d)
        assert backend.calls == []

    def test_capacity_unknown_is_a_no_op(self, gov_factory):
        backend = FakeBackend()
        gov = gov_factory(
            yaml_text="kv_coefficient: 0.5\n",
            warmup=KeepAliveWarmup(models=[_loaded("old1", 5.0, 60)]),
            registry=FakeRegistry([backend]),
        )
        d = rg.AdmissionDecision(
            admitted=True,
            model="new",
            num_ctx=4096,
            conditional_on_eviction=True,
            load_expected=True,
        )
        gov._honour_conditional_eviction(d)  # must not raise
        assert backend.calls == []


# ---------------------------------------------------------------------------
# F. The gate honours conditional grants (both branches, account-once kept)
# ---------------------------------------------------------------------------


class TestGateHonour:
    def _singleton(self, tmp_path, monkeypatch, backend, models):
        monkeypatch.setitem(
            sys.modules, "opti_oignon.emergency_stop", FakeEstop((False,))
        )
        monkeypatch.setitem(
            sys.modules, "opti_oignon.context_manager", _fake_cm()
        )
        cfg = _write_yaml(tmp_path, _BASE_YAML, "singleton.yaml")
        gov = rg.get_resource_governor(
            config_path=cfg, db_path=tmp_path / "singleton.db"
        )
        gov._warmup = KeepAliveWarmup(models=models)
        gov._registry_override = FakeRegistry([backend])
        return gov

    def test_matched_conditional_ticket_evicts_before_account(
        self, tmp_path, monkeypatch, singleton_guard
    ):
        backend = FakeBackend()
        gov = self._singleton(
            tmp_path, monkeypatch, backend, [_loaded("victim", 5.0, 60)]
        )
        d = gov.admit("new", 16384, caller="chat")
        assert d.admitted and d.conditional_on_eviction and d.load_expected
        loads = []
        monkeypatch.setattr(
            gov, "invalidate_on_load", lambda m, c: loads.append((m, c))
        )
        with rg.ticket_scope(d):
            rg.backend_admission_gate("new", {"temperature": 0.2})
            rg.backend_admission_gate("new", {"temperature": 0.2})
        # Evicted exactly once, accounted exactly once.
        assert backend.calls == ["victim"]
        assert loads == [("new", 16384)]
        assert d.load_expected is False

    def test_ticketless_backstop_honours_too(
        self, tmp_path, monkeypatch, singleton_guard
    ):
        backend = FakeBackend()
        gov = self._singleton(
            tmp_path, monkeypatch, backend, [_loaded("victim", 5.0, 60)]
        )
        rg.clear_active_ticket()
        rg.backend_admission_gate("new", {"num_ctx": 16384})
        assert backend.calls == ["victim"]
        ring = gov.store.recent_decisions(limit=5)
        assert any(r["caller"] == "direct" for r in ring)


# ---------------------------------------------------------------------------
# G. The bounded opt-in queue (Section 5; estop never bypassed)
# ---------------------------------------------------------------------------


class TestBoundedQueue:
    def test_not_enrolled_degrades_to_plain_admit(self, gov_factory):
        gov = gov_factory(
            warmup=KeepAliveWarmup(models=[_loaded("hog", 9.5, 1800)])
        )
        d = gov.admit_or_wait("new", 4096, caller="benchmark")
        assert d.admitted is False
        assert gov.queue_depth == 0
        ring = gov.store.recent_decisions(limit=10)
        assert not any(r["decision"] == "queue" for r in ring)

    def test_enqueue_is_ring_visible_and_wait_bound_resolves_to_refusal(
        self, gov_factory
    ):
        clock = FakeClock()
        gov = gov_factory(
            yaml_text=_QUEUE_YAML,
            warmup=KeepAliveWarmup(models=[_loaded("hog", 9.5, 1800)]),
            clock=clock,
        )
        thread, box = _start_waiter(gov, "new", 4096)
        try:
            assert _wait_for_depth(gov, 1)
            ring = gov.store.recent_decisions(limit=10)
            queued = [r for r in ring if r["decision"] == "queue"]
            assert queued and queued[0]["reason"] == "enqueued"
            assert queued[0]["caller"] == "benchmark"
            # Jump past the wait bound: the entry resolves to the
            # caller's refusal semantics.
            clock.advance(31.0)
            gov.invalidate_on_evict()
        finally:
            _release_waiter(gov, thread)
        d = box["decision"]
        assert d.admitted is False
        assert d.reason == "vram_insufficient"
        assert gov.queue_depth == 0

    def test_depth_bound_refuses_immediately(self, gov_factory):
        clock = FakeClock()
        gov = gov_factory(
            yaml_text=_BASE_YAML
            + (
                "queue:\n"
                "  enabled_per_caller:\n"
                "    benchmark: true\n"
                "  depth: 1\n"
                "  wait_s: 30\n"
            ),
            warmup=KeepAliveWarmup(models=[_loaded("hog", 9.5, 1800)]),
            clock=clock,
        )
        thread, _box = _start_waiter(gov, "new", 4096)
        try:
            assert _wait_for_depth(gov, 1)
            d = gov.admit_or_wait("other", 4096, caller="benchmark")
            assert d.admitted is False
            assert gov.queue_depth == 1
        finally:
            _release_waiter(gov, thread)

    def test_re_admission_on_wake_admits_when_capacity_freed(
        self, gov_factory
    ):
        clock = FakeClock()
        warmup = KeepAliveWarmup(models=[_loaded("hog", 9.5, 1800)])
        gov = gov_factory(yaml_text=_QUEUE_YAML, warmup=warmup, clock=clock)
        thread, box = _start_waiter(gov, "new", 4096)
        try:
            assert _wait_for_depth(gov, 1)
            warmup.models = []
            gov.invalidate_on_evict()
            thread.join(timeout=5.0)
            assert not thread.is_alive()
        finally:
            _release_waiter(gov, thread)
        assert box["decision"].admitted is True
        assert gov.queue_depth == 0

    def test_estop_never_bypassed_drain_releases_to_refusal(
        self, gov_factory, seams
    ):
        # The clock stays frozen, so the only way out before the
        # deadline is the estop refusal: the drain notification wakes
        # the waiter and its re-admission honours the flag FIRST.
        clock = FakeClock()
        gov = gov_factory(
            yaml_text=_QUEUE_YAML,
            warmup=KeepAliveWarmup(models=[_loaded("hog", 9.5, 1800)]),
            clock=clock,
        )
        thread, box = _start_waiter(gov, "new", 4096)
        try:
            assert _wait_for_depth(gov, 1)
            seams._seq = [True]
            gov.invalidate_on_estop_drain()
            thread.join(timeout=5.0)
            assert not thread.is_alive()
        finally:
            seams._seq = [False]
            _release_waiter(gov, thread)
        d = box["decision"]
        assert d.admitted is False
        assert d.is_estop is True
        assert d.payload.get("error") == "emergency_stopped"
        assert gov.queue_depth == 0

    def test_disabled_governor_passthrough_returns_immediately(
        self, gov_factory
    ):
        gov = gov_factory(yaml_text="enabled: false\n" + _QUEUE_YAML)
        d = gov.admit_or_wait("m", 4096, caller="benchmark")
        assert d.admitted is True
        assert d.reason == "governor_disabled"
        assert gov.queue_depth == 0


# ---------------------------------------------------------------------------
# H. Benchmark routing: the queue entry with the defensive fallback
# ---------------------------------------------------------------------------


class _SpyGov:
    def __init__(self, with_queue: bool):
        self.config = SimpleNamespace(enabled=True)
        self.calls: list = []
        if with_queue:
            self.admit_or_wait = self._admit_or_wait

    def _admit_or_wait(self, model, ctx, caller):
        self.calls.append(("admit_or_wait", model, caller))
        return SimpleNamespace(admitted=True, load_expected=False, num_ctx=None)

    def admit(self, model, ctx, caller):
        self.calls.append(("admit", model, caller))
        return SimpleNamespace(admitted=True, load_expected=False, num_ctx=None)

    def invalidate_on_load(self, model, ctx):
        self.calls.append(("invalidate_on_load", model, ctx))


class TestBenchmarkRouting:
    def _bench(self):
        return _load_module(
            "opti_oignon.benchmark_runner", "opti_oignon/benchmark_runner.py"
        )

    def test_routes_through_admit_or_wait(self, monkeypatch):
        bench = self._bench()
        spy = _SpyGov(with_queue=True)
        monkeypatch.setattr(
            bench,
            "_resolve_resource_governor",
            lambda: SimpleNamespace(get_resource_governor=lambda: spy),
        )
        d = bench._admit_benchmark_model("some-model")
        assert d is not None and d.admitted is True
        assert spy.calls == [("admit_or_wait", "some-model", "benchmark")]

    def test_falls_back_to_admit_without_the_entry(self, monkeypatch):
        bench = self._bench()
        spy = _SpyGov(with_queue=False)
        monkeypatch.setattr(
            bench,
            "_resolve_resource_governor",
            lambda: SimpleNamespace(get_resource_governor=lambda: spy),
        )
        d = bench._admit_benchmark_model("some-model")
        assert d is not None and d.admitted is True
        assert spy.calls == [("admit", "some-model", "benchmark")]


# ---------------------------------------------------------------------------
# I. Helper semantics: the candidates list behind both consumers
# ---------------------------------------------------------------------------


class TestEvictableCandidates:
    def test_order_filtering_and_delegation(self, gov_factory):
        gov = gov_factory(
            warmup=KeepAliveWarmup(
                models=[
                    _loaded("old", 5.0, 60),
                    _loaded("mid", 3.0, 960),
                    _loaded("fresh", 2.0, 1740),
                    _loaded("zero", 0.0, 60),
                    SimpleNamespace(
                        name="bad",
                        size_vram=1 * GB,
                        expires_at="garbage",
                        context_length=None,
                        digest=None,
                    ),
                ]
            )
        )
        snap = gov.refresh(force=True)
        cands = gov._evictable_candidates(snap)
        assert [c[0] for c in cands] == ["old", "mid"]
        assert cands[0][2] == pytest.approx(5.0)
        # _evictable_now_gb is the delegated sum of the same definition.
        assert gov._evictable_now_gb(snap) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# J. Source pins on the touched funnels and the new primitive
# ---------------------------------------------------------------------------


class TestSourcePins:
    def test_executor_override_wired_at_the_three_funnels(self):
        ast.parse(EXEC_SRC)
        assert (
            EXEC_SRC.count("S225: per-decision keep_alive override") == 3
        )
        assert EXEC_SRC.count("ka = _admission.keep_alive") == 2
        assert EXEC_SRC.count("ka = _gov_decision.keep_alive") == 1

    def test_benchmark_routes_with_the_defensive_getattr(self):
        ast.parse(BENCH_SRC)
        assert (
            'getattr(governor, "admit_or_wait", governor.admit)' in BENCH_SRC
        )
        assert 'caller="benchmark"' in BENCH_SRC

    def test_ollama_gains_the_narrowed_primitive(self):
        ast.parse(IB_SRC)
        assert IB_SRC.count("def unload_model(self, model_name: str)") == 2
        assert IB_SRC.count("keep_alive=0") >= 2
        # The s215 split-based span stays valid with the chosen
        # placement (after unload_all): the span still carries the
        # pop-based llama.cpp body and no in-then-del.
        body = IB_SRC.split("def unload_model", 1)[1].split(
            "def unload_all", 1
        )[0]
        assert "del self._loaded_models" not in body
        assert ".pop(" in body

    def test_governor_bloc2_surface_defined_once(self):
        for name in (
            "def pressure_state",
            "def admit_or_wait",
            "def evict_model",
            "def _honour_conditional_eviction",
            "def _evictable_candidates",
            "def _notify_queue",
        ):
            assert SRC.count(name) == 1, name
        # def + the two gate-branch calls.
        assert SRC.count("_honour_conditional_eviction(") == 3
        # Module conventions hold through the bloc.
        assert rg.checkpoint_before_apply is True
        assert rg.FEATURE_AVAILABLE is True


# ---------------------------------------------------------------------------
# K. Doc and config pins (red-before provable on the pristine tree)
# ---------------------------------------------------------------------------


class TestDocAndConfigPins:
    def test_spec_section_10_names_the_bloc2_keys(self):
        spec = _SPEC_PATH.read_text(encoding="utf-8")
        section_10 = spec.split("## 10. Configuration", 1)[1].split(
            "## 11.", 1
        )[0]
        assert "pressure.sustain_s (60.0" in section_10
        assert "pressure.refusal_window_s (60.0" in section_10
        # The S224 substrings survive the additive touch.
        assert "ceiling_floor_gb (4.0" in section_10
        assert "decisions_ring_size (200" in section_10

    def test_shipped_yaml_carries_the_bloc2_keys(self):
        text = _YAML_PATH.read_text(encoding="utf-8")
        assert "sustain_s: 60.0" in text
        assert "refusal_window_s: 60.0" in text
