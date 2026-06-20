#!/usr/bin/env python3
"""S224 -- Resource Governor Bloc 1: the admission gate (R-01) and R-04.

Per-fix suite for the S224 lot against RESOURCE_GOVERNOR_SPEC.md Sections
4 / 4.1-4.5 / 7 / 8 and the S224 read-gate decisions (DI-S224 1..13),
covering the spec's Bloc 1 container-provable list verbatim:

- every decision path with a faked snapshot: admit / downsize through the
  ladder to the floor / refuse below it / conditional-on-eviction;
- the requested-ctx clamp (ModelLimits authority), resident-model
  zero-weight admission, the unknown-model never-too-large rule, the
  extra_models pair folding (the speculative draft+verify decision);
- capacity-unknown VRAM fail-open (3.1) with the RAM half still applying;
- R-04: the estop flag honoured FIRST (before any fit math), the refusal
  payload mirrored from the existing seam, and the drain/resume
  invalidations fired on flag transitions (emergency_stop NOT edited);
- the disabled-governor unrecorded passthrough; the decisions ring
  written with the caller, the requested and the admitted ctx;
- the thread-local ticket (scope semantics) and the mechanical backend
  gate: ticket stand-down with the account-once load attribution, the
  ticketless backstop (caller "direct", admit-or-refuse, GovernorRefusal
  typed and carrying the decision);
- the inference_backend hook: module-absent/unavailable fail-open, other
  errors fail open, only the typed refusal propagates, the four
  generate/stream signatures stable;
- the pipeline per-step gate (abort message semantics beside the S216
  estop check) and the benchmark refuse-or-skip with the not-admitted
  recording persisted in results (never a silent downsize), plus the
  evict-between default and its kwarg;
- the executor funnel wiring by source pins (six funnels, the ticket
  hold/release pairs, the project's first options["num_ctx"]);
- the doc pin on the spec Section 10 additive line and the pyproject
  deselect line (both red-before provable on the pristine tree) and the
  superseded S223 boundary pin reasserted (admit present, 4.4 shape).

Host-assured (named, never simulated here): the real meaning of
size_vram and expirations on the real driver; a real refused load on a
real GPU. Nothing else (spec Section 11, Bloc 1).

Isolation: the established spec_from_file_location idiom with
sys.modules pre-seeding (an ollama stub and an opti_oignon package stub
carrying a real __path__), order-independent: collaborator seams
(emergency_stop, context_manager, resource_governor for the consumers)
are seeded per-test through monkeypatch.setitem so standalone and
in-sweep runs resolve identically.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import sqlite3
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_BASE = Path(__file__).resolve().parent.parent
_MODULE_PATH = _BASE / "opti_oignon" / "resource_governor.py"
_IB_PATH = _BASE / "opti_oignon" / "inference_backend.py"
_EXEC_PATH = _BASE / "opti_oignon" / "executor.py"
_PIPELINES_PATH = _BASE / "opti_oignon" / "pipelines.py"
_BENCH_PATH = _BASE / "opti_oignon" / "benchmark_runner.py"
_SPEC_PATH = _BASE / "RESOURCE_GOVERNOR_SPEC.md"
_PYPROJECT_PATH = _BASE / "pyproject.toml"

SRC = _MODULE_PATH.read_text(encoding="utf-8")
IB_SRC = _IB_PATH.read_text(encoding="utf-8")
EXEC_SRC = _EXEC_PATH.read_text(encoding="utf-8")
PIPELINES_SRC = _PIPELINES_PATH.read_text(encoding="utf-8")
BENCH_SRC = _BENCH_PATH.read_text(encoding="utf-8")

GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Isolated module loading (the established idiom, mirrored from s223)
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


# Pre-load the governor's conditional dependencies by path, REUSING
# whatever a prior suite already put in sys.modules (never replacing),
# so this suite is order-independent (the documented S220 pollution
# class): standalone and in-sweep resolve the same flags.
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
# Fakes
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class KeepAliveWarmup:
    """Injectable S1 fake with the keep_alive attribute the evictable
    computation derives idle time from."""

    keep_alive = "30m"

    def __init__(self, models=None):
        self.calls = 0
        self.models = list(models or [])

    def get_loaded_models(self):
        self.calls += 1
        return list(self.models)


class RaisingWarmup:
    """Any consultation raises: the estop-precedence proof."""

    def get_loaded_models(self):
        raise AssertionError("snapshot must not be built before the flag")


def _loaded(name, size_gb, expires_in_s=None, digest=None):
    expires = None if expires_in_s is None else time.time() + expires_in_s
    return SimpleNamespace(
        name=name,
        size_vram=int(size_gb * GB),
        expires_at=expires,
        context_length=None,
        digest=digest,
    )


class FakeEstop:
    """is_stopped() answers from a scripted sequence (last value sticky);
    refusal_payload() is the fixed S215-shaped body."""

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


def _fake_cm(window=32768, max_output=4096):
    return SimpleNamespace(
        get_model_limits=lambda model: SimpleNamespace(
            context_window=window, max_output=max_output
        )
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def seams(monkeypatch):
    """Deterministic collaborator seams for fit-math tests: a not-stopped
    estop and a 32768-window ModelLimits, seeded in sys.modules so the
    governor's sys.modules-first resolvers find them identically
    standalone and in-sweep."""
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
            yaml_text = (
                "total_vram_gb: 10\n"
                "safety_margin_gb: 1.0\n"
                "kv_coefficient: 0.5\n"
                "idle_evict_threshold_s: 600\n"
            )
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
    """Reset the module-level governor before and after gate tests."""
    rg.reset_resource_governor()
    rg.clear_active_ticket()
    yield
    rg.clear_active_ticket()
    rg.reset_resource_governor()


# ---------------------------------------------------------------------------
# A. admit(): the decision paths on faked snapshots (4.2 / 4.3)
# ---------------------------------------------------------------------------


class TestAdmitDecisionPaths:
    def test_fits_unconditionally(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        d = gov.admit("some-model", requested_ctx=16384, caller="chat")
        # kv(16384) = 8 <= budget 9 (10 - 0 - 1); unknown weights count 0.
        assert d.admitted is True
        assert d.action == "admit"
        assert d.num_ctx == 16384
        assert d.reason == "fits"
        assert d.load_expected is True
        assert d.conditional_on_eviction is False
        assert d.ticket_id

    def test_downsize_through_the_ladder(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        d = gov.admit("some-model", requested_ctx=200000, caller="chat")
        # clamp 32768 -> kv 16 > 9; ladder 16384 -> kv 8 <= 9.
        assert d.admitted is True
        assert d.action == "downsize"
        assert d.num_ctx == 16384
        assert "ctx_laddered_to_fit" in d.reason

    def test_refuse_below_the_floor_with_shortfall(self, gov_factory):
        gov = gov_factory(
            yaml_text=(
                "total_vram_gb: 4\n"
                "safety_margin_gb: 1.0\n"
                "kv_coefficient: 1.0\n"
            ),
            warmup=KeepAliveWarmup(),
        )
        d = gov.admit("some-model", requested_ctx=4096, caller="chat")
        # kv(4096) = 4 > budget 3; the floor IS 4096: no smaller step.
        assert d.admitted is False
        assert d.action == "refuse"
        assert d.reason == "vram_insufficient"
        assert d.shortfall_gb == pytest.approx(1.0)
        payload = d.refusal_payload()
        assert payload["error"] == "resource_admission_refused"
        assert payload["model"] == "some-model"
        assert payload["options"] == [
            "evict idle models",
            "pick a smaller model",
            "lower context",
        ]

    def test_conditional_on_eviction(self, gov_factory):
        warmup = KeepAliveWarmup([_loaded("idle-m", 5.0, expires_in_s=60)])
        gov = gov_factory(warmup=warmup)
        # idle = 1800 - 60 = 1740 >= 600 -> evictable 5; in_use 5 ->
        # budget_uncond 4, budget_with_eviction 9; kv(16384) = 8.
        d = gov.admit("new-model", requested_ctx=16384, caller="chat")
        assert d.admitted is True
        assert d.action == "admit"
        assert d.conditional_on_eviction is True
        assert "conditional_on_eviction" in d.reason

    def test_resident_model_is_a_ctx_check_only(self, gov_factory):
        warmup = KeepAliveWarmup([_loaded("resident", 4.0, expires_in_s=1790)])
        gov = gov_factory(warmup=warmup)
        # Fresh (idle 10 s < 600): nothing evictable; in_use 4 ->
        # budget 5; resident weights count 0; kv(8192) = 4 <= 5.
        d = gov.admit("resident", requested_ctx=8192, caller="chat")
        assert d.admitted is True
        assert d.num_ctx == 8192
        assert d.load_expected is False

    def test_unknown_model_is_never_too_large(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        d = gov.admit("mystery-model", requested_ctx=None, caller="chat")
        assert d.admitted is True
        assert d.num_ctx is None
        assert d.reason == "fits"

    def test_extra_models_fold_the_pair(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        gov.store.record_model_cost("verify-m", None, int(6 * GB))
        gov.store.record_model_cost("draft-m", None, int(2 * GB))
        alone = gov.admit("verify-m", None, caller="chat")
        assert alone.admitted is True  # 6 <= 9
        pair = gov.admit(
            "verify-m", None, caller="chat", extra_models=["draft-m"]
        )
        # 6 + 2 = 8 <= 9 still fits; tighten with a requested ctx:
        assert pair.admitted is True
        tight = gov.admit(
            "verify-m", 4096, caller="chat", extra_models=["draft-m"]
        )
        # 8 + kv(4096)=2 = 10 > 9 and no step below the floor: refuse.
        assert tight.admitted is False
        assert tight.shortfall_gb == pytest.approx(1.0)

    def test_clamp_to_the_model_window(self, gov_factory, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "opti_oignon.context_manager", _fake_cm(window=8192)
        )
        gov = gov_factory(warmup=KeepAliveWarmup())
        d = gov.admit("some-model", requested_ctx=200000, caller="chat")
        # clamp 8192 -> kv 4 <= 9: admitted AT the clamp, not laddered.
        assert d.admitted is True
        assert d.action == "admit"
        assert d.num_ctx == 8192
        assert "clamped_to_model_limit" in d.reason

    def test_benchmark_caller_never_downsizes(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        d = gov.admit("some-model", requested_ctx=32768, caller="benchmark")
        # kv 16 > 9 and benchmark has no floor: straight refusal.
        assert d.admitted is False
        assert d.action == "refuse"
        assert d.shortfall_gb == pytest.approx(7.0)

    def test_num_gpu_and_keep_alive_stay_none_in_bloc1(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        d = gov.admit("some-model", requested_ctx=4096, caller="chat")
        assert d.num_gpu is None
        assert d.keep_alive is None

    def test_provenance_carries_snapshot_sources(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        d = gov.admit("some-model", requested_ctx=4096, caller="chat")
        assert isinstance(d.provenance, list)
        assert "S1" in d.provenance


# ---------------------------------------------------------------------------
# B. Capacity unknown: VRAM half fail-open, RAM half still applying (3.1)
# ---------------------------------------------------------------------------


class TestCapacityUnknownFailOpen:
    def test_unknown_capacity_admits_fail_open(self, gov_factory, tmp_path):
        gov = gov_factory(
            yaml_text="safety_margin_gb: 1.0\n", warmup=KeepAliveWarmup()
        )
        d = gov.admit("mystery-model", requested_ctx=32768, caller="chat")
        assert d.admitted is True
        assert d.reason == "capacity_unknown_fail_open"
        assert d.num_ctx == 32768

    def test_ram_half_still_refuses_known_weights(self, gov_factory, tmp_path):
        gov = gov_factory(
            yaml_text="safety_margin_gb: 1.0\n",
            warmup=KeepAliveWarmup(),
            meminfo_path=_meminfo_file(
                tmp_path, available_kb=1024 * 1024, name="meminfo_small"
            ),
        )
        gov.store.record_model_cost("ram-hog", None, int(8 * GB))
        d = gov.admit("ram-hog", requested_ctx=4096, caller="chat")
        # 8 GB known weights > 1 GiB MemAvailable.
        assert d.admitted is False
        assert d.reason == "ram_insufficient"
        assert d.shortfall_gb and d.shortfall_gb > 0


# ---------------------------------------------------------------------------
# C. R-04: estop precedence and the transition invalidations (4.5)
# ---------------------------------------------------------------------------


class TestEstopPrecedence:
    def test_flag_refuses_before_any_fit_math(
        self, gov_factory, monkeypatch
    ):
        estop = FakeEstop((True,))
        monkeypatch.setitem(
            sys.modules, "opti_oignon.emergency_stop", estop
        )
        gov = gov_factory(warmup=RaisingWarmup())
        d = gov.admit("any-model", requested_ctx=32768, caller="chat")
        # RaisingWarmup proves the snapshot was never built.
        assert d.admitted is False
        assert d.action == "refuse"
        assert d.reason == "emergency_stopped"
        assert d.is_estop is True

    def test_refusal_payload_mirrors_the_seam(self, gov_factory, monkeypatch):
        estop = FakeEstop((True,))
        monkeypatch.setitem(
            sys.modules, "opti_oignon.emergency_stop", estop
        )
        gov = gov_factory(warmup=RaisingWarmup())
        d = gov.admit("any-model", caller="pipeline")
        assert d.refusal_payload() == estop.payload

    def test_estop_refusal_is_recorded_in_the_ring(
        self, gov_factory, monkeypatch
    ):
        estop = FakeEstop((True,))
        monkeypatch.setitem(
            sys.modules, "opti_oignon.emergency_stop", estop
        )
        gov = gov_factory(warmup=RaisingWarmup())
        gov.admit("any-model", requested_ctx=4096, caller="benchmark")
        row = gov.store.recent_decisions(1)[0]
        assert row["caller"] == "benchmark"
        assert row["decision"] == "refuse"
        assert row["reason"] == "emergency_stopped"

    def test_transitions_fire_drain_and_resume_once(
        self, gov_factory, monkeypatch
    ):
        estop = FakeEstop((False, True, True, False))
        monkeypatch.setitem(
            sys.modules, "opti_oignon.emergency_stop", estop
        )
        gov = gov_factory(warmup=KeepAliveWarmup())
        counts = {"drain": 0, "resume": 0}
        real_drain = gov.invalidate_on_estop_drain
        real_resume = gov.invalidate_on_resume

        def spy_drain():
            counts["drain"] += 1
            real_drain()

        def spy_resume():
            counts["resume"] += 1
            real_resume()

        monkeypatch.setattr(gov, "invalidate_on_estop_drain", spy_drain)
        monkeypatch.setattr(gov, "invalidate_on_resume", spy_resume)
        gov.admit("m", caller="chat")  # False: seeds state
        gov.admit("m", caller="chat")  # False -> True: drain
        gov.admit("m", caller="chat")  # True (repeat): nothing
        gov.admit("m", caller="chat")  # True -> False: resume
        assert counts == {"drain": 1, "resume": 1}

    def test_emergency_stop_module_is_not_edited(self):
        # R-04 wiring lives in the governor; the seam itself is intact:
        # no governor reference appears in emergency_stop.py.
        estop_src = (_BASE / "opti_oignon" / "emergency_stop.py").read_text(
            encoding="utf-8"
        )
        assert "resource_governor" not in estop_src
        assert "GovernorRefusal" not in estop_src


# ---------------------------------------------------------------------------
# D. Disabled-by-config passthrough (unrecorded)
# ---------------------------------------------------------------------------


class TestDisabledPassthrough:
    def test_disabled_admits_and_records_nothing(self, gov_factory):
        gov = gov_factory(
            yaml_text="enabled: false\ntotal_vram_gb: 10\n",
            warmup=RaisingWarmup(),
        )
        d = gov.admit("any-model", requested_ctx=32768, caller="chat")
        assert d.admitted is True
        assert d.reason == "governor_disabled"
        assert gov.store.decision_count() == 0


# ---------------------------------------------------------------------------
# E. The decisions ring (4.4)
# ---------------------------------------------------------------------------


class TestDecisionRing:
    def test_ring_row_carries_caller_and_contexts(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        gov.admit("ring-model", requested_ctx=16384, caller="chat")
        row = gov.store.recent_decisions(1)[0]
        assert row["caller"] == "chat"
        assert row["model"] == "ring-model"
        assert row["requested_ctx"] == 16384
        assert row["admitted_ctx"] == 16384
        assert row["decision"] == "admit"
        assert row["reason"] == "fits"

    def test_every_decision_path_records(self, gov_factory):
        gov = gov_factory(warmup=KeepAliveWarmup())
        gov.admit("m1", 16384, caller="chat")  # admit
        gov.admit("m2", 200000, caller="chat")  # downsize
        gov.admit("m3", 32768, caller="benchmark")  # refuse
        kinds = [r["decision"] for r in gov.store.recent_decisions(3)]
        assert sorted(kinds) == ["admit", "downsize", "refuse"]


# ---------------------------------------------------------------------------
# F. The ticket pass-through and the mechanical gate (4.4 / 4.1)
# ---------------------------------------------------------------------------


class TestTicketScope:
    def test_scope_sets_and_restores(self):
        d = rg.AdmissionDecision(admitted=True, model="m")
        assert rg.get_active_ticket() is None
        with rg.ticket_scope(d):
            assert rg.get_active_ticket() is d
        assert rg.get_active_ticket() is None

    def test_scope_restores_the_previous_ticket(self):
        outer = rg.AdmissionDecision(admitted=True, model="outer")
        inner = rg.AdmissionDecision(admitted=True, model="inner")
        with rg.ticket_scope(outer):
            with rg.ticket_scope(inner):
                assert rg.get_active_ticket() is inner
            assert rg.get_active_ticket() is outer
        assert rg.get_active_ticket() is None

    def test_none_decision_is_a_noop_scope(self):
        with rg.ticket_scope(None):
            assert rg.get_active_ticket() is None

    def test_set_and_clear_helpers(self):
        d = rg.AdmissionDecision(admitted=True, model="m")
        rg.set_active_ticket(d)
        assert rg.get_active_ticket() is d
        rg.clear_active_ticket()
        assert rg.get_active_ticket() is None


class TestBackendGate:
    def _singleton(self, tmp_path, monkeypatch, yaml_text=None):
        monkeypatch.setitem(
            sys.modules, "opti_oignon.emergency_stop", FakeEstop((False,))
        )
        monkeypatch.setitem(
            sys.modules, "opti_oignon.context_manager", _fake_cm()
        )
        if yaml_text is None:
            yaml_text = (
                "total_vram_gb: 10\n"
                "safety_margin_gb: 1.0\n"
                "kv_coefficient: 0.5\n"
            )
        cfg = _write_yaml(tmp_path, yaml_text, "singleton.yaml")
        gov = rg.get_resource_governor(
            config_path=cfg, db_path=tmp_path / "singleton.db"
        )
        gov._warmup = KeepAliveWarmup()
        return gov

    def test_matching_ticket_stands_down_and_accounts_once(
        self, tmp_path, monkeypatch, singleton_guard
    ):
        gov = self._singleton(tmp_path, monkeypatch)
        decision = gov.admit("ticketed-m", 4096, caller="chat")
        assert decision.admitted and decision.load_expected
        loads = []
        monkeypatch.setattr(
            gov, "invalidate_on_load", lambda m, c: loads.append((m, c))
        )
        monkeypatch.setattr(
            gov,
            "admit",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("backstop must not run with a ticket")
            ),
        )
        with rg.ticket_scope(decision):
            rg.backend_admission_gate("ticketed-m", {"temperature": 0.2})
            rg.backend_admission_gate("ticketed-m", {"temperature": 0.2})
        assert loads == [("ticketed-m", 4096)]
        assert decision.load_expected is False

    def test_ticketless_backstop_refuses_with_typed_error(
        self, tmp_path, monkeypatch, singleton_guard
    ):
        self._singleton(tmp_path, monkeypatch)
        with pytest.raises(rg.GovernorRefusal) as exc_info:
            rg.backend_admission_gate("huge-m", {"num_ctx": 32768})
        refusal = exc_info.value
        assert isinstance(refusal, RuntimeError)
        assert refusal.decision.action == "refuse"
        assert refusal.decision.caller == "direct"
        assert "Not enough resources" in str(refusal)

    def test_ticketless_backstop_admits_and_accounts(
        self, tmp_path, monkeypatch, singleton_guard
    ):
        gov = self._singleton(tmp_path, monkeypatch)
        loads = []
        monkeypatch.setattr(
            gov, "invalidate_on_load", lambda m, c: loads.append((m, c))
        )
        rg.backend_admission_gate("small-m", {"num_ctx": 4096})
        assert loads == [("small-m", 4096)]
        row = gov.store.recent_decisions(1)[0]
        assert row["caller"] == "direct"

    def test_disabled_governor_stands_the_gate_down(
        self, tmp_path, monkeypatch, singleton_guard
    ):
        gov = self._singleton(
            tmp_path, monkeypatch, yaml_text="enabled: false\n"
        )
        monkeypatch.setattr(
            gov,
            "admit",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("disabled gate must not admit")
            ),
        )
        rg.backend_admission_gate("any-m", {})  # no raise

    def test_refusal_carries_the_decision(self):
        d = rg.AdmissionDecision(
            admitted=False, model="m", action="refuse",
            reason="vram_insufficient", shortfall_gb=2.5,
        )
        err = rg.GovernorRefusal(d)
        assert err.decision is d
        assert issubclass(rg.GovernorRefusal, RuntimeError)


# ---------------------------------------------------------------------------
# G. The inference_backend hook (4.1): fail-open trio, signatures
# ---------------------------------------------------------------------------


class _SentinelLib:
    """Fake ollama module: touching the transport raises the sentinel."""

    def chat(self, **kwargs):
        raise RuntimeError("reached-transport")


class TestBackendHook:
    @pytest.fixture(autouse=True)
    def _ollama_on(self, monkeypatch):
        monkeypatch.setattr(ib, "OLLAMA_AVAILABLE", True)
        monkeypatch.setattr(ib, "_ollama_module", _SentinelLib())

    def test_unavailable_governor_means_unguarded(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules,
            "opti_oignon.resource_governor",
            SimpleNamespace(FEATURE_AVAILABLE=False),
        )
        backend = ib.OllamaBackend()
        with pytest.raises(RuntimeError, match="reached-transport"):
            backend.generate("m", [{"role": "user", "content": "hi"}])

    def test_gate_errors_fail_open(self, monkeypatch):
        def exploding_gate(model, options):
            raise ValueError("governor exploded")

        monkeypatch.setitem(
            sys.modules,
            "opti_oignon.resource_governor",
            SimpleNamespace(
                FEATURE_AVAILABLE=True,
                backend_admission_gate=exploding_gate,
                GovernorRefusal=rg.GovernorRefusal,
            ),
        )
        backend = ib.OllamaBackend()
        with pytest.raises(RuntimeError, match="reached-transport"):
            backend.generate("m", [{"role": "user", "content": "hi"}])

    def test_typed_refusal_propagates_from_generate(self, monkeypatch):
        decision = rg.AdmissionDecision(
            admitted=False, model="m", action="refuse", reason="x"
        )

        def refusing_gate(model, options):
            raise rg.GovernorRefusal(decision)

        monkeypatch.setitem(
            sys.modules,
            "opti_oignon.resource_governor",
            SimpleNamespace(
                FEATURE_AVAILABLE=True,
                backend_admission_gate=refusing_gate,
                GovernorRefusal=rg.GovernorRefusal,
            ),
        )
        backend = ib.OllamaBackend()
        with pytest.raises(rg.GovernorRefusal):
            backend.generate("m", [{"role": "user", "content": "hi"}])

    def test_typed_refusal_propagates_from_stream_at_first_next(
        self, monkeypatch
    ):
        decision = rg.AdmissionDecision(
            admitted=False, model="m", action="refuse", reason="x"
        )

        def refusing_gate(model, options):
            raise rg.GovernorRefusal(decision)

        monkeypatch.setitem(
            sys.modules,
            "opti_oignon.resource_governor",
            SimpleNamespace(
                FEATURE_AVAILABLE=True,
                backend_admission_gate=refusing_gate,
                GovernorRefusal=rg.GovernorRefusal,
            ),
        )
        backend = ib.OllamaBackend()
        gen = backend.stream("m", [{"role": "user", "content": "hi"}])
        with pytest.raises(rg.GovernorRefusal):
            next(gen)

    def test_four_head_signatures_are_stable(self):
        import inspect

        expected = [
            "self", "model", "messages", "options",
            "keep_alive", "think", "images",
        ]
        for cls in (ib.OllamaBackend, ib.LlamaCppBackend):
            for meth in ("generate", "stream"):
                params = list(
                    inspect.signature(getattr(cls, meth)).parameters
                )
                assert params == expected, (cls.__name__, meth, params)

    def test_hook_sits_after_the_availability_guard(self):
        # The Ollama heads keep the s105 "not installed" error semantics:
        # the guard precedes the hook in both method bodies.
        for head in ("Non-streaming chat via ollama.chat()",
                     "Streaming chat via ollama.chat(stream=True)"):
            body = IB_SRC.split(head, 1)[1][:800]
            guard_pos = body.index("if not OLLAMA_AVAILABLE:")
            hook_pos = body.index("_governor_admission(model, options)")
            assert guard_pos < hook_pos

    def test_hook_wired_at_all_four_heads(self):
        assert IB_SRC.count("_governor_admission(model, options)") == 4
        assert IB_SRC.count("def _governor_admission") == 1
        ast.parse(IB_SRC)


# ---------------------------------------------------------------------------
# H. The pipeline per-step gate (functional, the s216 runner idiom)
# ---------------------------------------------------------------------------


class FakeAgenticExecutor:
    """Records every execute() call and yields configured chunks."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        chunks = self.outputs[idx] if idx < len(self.outputs) else []
        yield from chunks


class FakeSmartRouter:
    enabled = False


class _RefusingGov:
    def __init__(self, refuse_models=()):
        self.config = SimpleNamespace(enabled=True)
        self.refuse_models = set(refuse_models)
        self.admit_calls = []

    def admit(self, model, requested_ctx=None, caller="chat",
              extra_models=None, digest=None):
        self.admit_calls.append((model, requested_ctx, caller))
        if model in self.refuse_models:
            return rg.AdmissionDecision(
                admitted=False, model=model, action="refuse",
                reason="vram_insufficient", caller=caller,
            )
        return rg.AdmissionDecision(
            admitted=True, model=model, action="admit", reason="fits",
            caller=caller, load_expected=True,
        )

    def invalidate_on_load(self, model, num_ctx):
        pass

    def invalidate_on_evict(self, model):
        pass


def _gov_module(gov):
    return SimpleNamespace(
        FEATURE_AVAILABLE=True, get_resource_governor=lambda: gov
    )


@pytest.fixture()
def pipelines_mod():
    return _load_module("oo_pipelines_s224", "opti_oignon/pipelines.py")


def _two_step_pipeline(mod):
    now = "2026-01-01T00:00:00"
    steps = [
        mod.ExecutionStep(step_type="direct", label="first"),
        mod.ExecutionStep(step_type="direct", label="second"),
    ]
    return mod.ExecutionPipeline(
        id="p1",
        name="P1",
        description="two-step",
        steps=steps,
        created_at=now,
        updated_at=now,
        is_builtin=False,
    )


def _run_pipeline(mod, executor, monkeypatch, gov):
    monkeypatch.setattr(
        mod, "_resolve_emergency_stop", lambda: FakeEstop((False,))
    )
    monkeypatch.setitem(
        sys.modules, "opti_oignon.resource_governor", _gov_module(gov)
    )
    runner = mod.PipelineRunner(
        agentic_executor=executor, smart_router=FakeSmartRouter()
    )
    pipeline = _two_step_pipeline(mod)
    routing = SimpleNamespace(model="step-model")
    return list(
        runner.execute(pipeline=pipeline, message="hello", routing=routing)
    )


class TestPipelinePerStepGate:
    def test_refusal_aborts_with_the_established_prefix(
        self, pipelines_mod, monkeypatch
    ):
        executor = FakeAgenticExecutor([["a"], ["b"]])
        gov = _RefusingGov(refuse_models={"step-model"})
        chunks = _run_pipeline(pipelines_mod, executor, monkeypatch, gov)
        strings = [c for c in chunks if isinstance(c, str)]
        assert strings, "the abort chunk must be yielded"
        assert strings[-1].startswith(
            "\n[ERR] Pipeline aborted: resource admission refused for "
            "step-model"
        )
        assert "vram_insufficient" in strings[-1]
        assert executor.calls == []

    def test_admission_lets_the_run_proceed(
        self, pipelines_mod, monkeypatch
    ):
        executor = FakeAgenticExecutor([["a"], ["b"]])
        gov = _RefusingGov()
        chunks = _run_pipeline(pipelines_mod, executor, monkeypatch, gov)
        assert len(executor.calls) == 2
        assert [m for m, _, _ in gov.admit_calls] == [
            "step-model", "step-model"
        ]
        assert all(caller == "pipeline" for _, _, caller in gov.admit_calls)
        assert all(req is None for _, req, _ in gov.admit_calls)
        strings = [c for c in chunks if isinstance(c, str)]
        assert "a" in strings and "b" in strings

    def test_absent_governor_runs_unguarded(self, pipelines_mod, monkeypatch):
        executor = FakeAgenticExecutor([["a"], ["b"]])
        monkeypatch.setattr(
            pipelines_mod, "_resolve_emergency_stop",
            lambda: FakeEstop((False,)),
        )
        monkeypatch.setitem(
            sys.modules,
            "opti_oignon.resource_governor",
            SimpleNamespace(FEATURE_AVAILABLE=False),
        )
        runner = pipelines_mod.PipelineRunner(
            agentic_executor=executor, smart_router=FakeSmartRouter()
        )
        pipeline = _two_step_pipeline(pipelines_mod)
        routing = SimpleNamespace(model="step-model")
        list(runner.execute(pipeline=pipeline, message="x", routing=routing))
        assert len(executor.calls) == 2

    def test_s216_estop_check_is_intact_and_first(self):
        estop_pos = PIPELINES_SRC.index(
            "[ERR] Pipeline aborted: emergency stop engaged"
        )
        gate_pos = PIPELINES_SRC.index(
            "[ERR] Pipeline aborted: resource admission"
        )
        assert estop_pos < gate_pos
        assert "def _resolve_resource_governor" in PIPELINES_SRC
        ast.parse(PIPELINES_SRC)


# ---------------------------------------------------------------------------
# I. Benchmark semantics: refuse-or-skip recorded, evict-between (4.3)
# ---------------------------------------------------------------------------


class _CountingIB:
    def __init__(self):
        self.unloads = 0

    def get_backend_registry(self):
        outer = self

        class _B:
            def unload_all(self):
                outer.unloads += 1
                return 1

        return SimpleNamespace(backends=lambda: [_B()])


@pytest.fixture()
def bench_mod():
    return _load_module(
        "opti_oignon.benchmark_runner", "opti_oignon/benchmark_runner.py"
    )


def _bench_qfn_factory(seen):
    def qfn(model, prompt, timeout=45, max_tokens=800):
        seen.append(model)
        return f"resp from {model}", 5.0, 50.0, 10

    return qfn


class TestBenchmarkSemantics:
    def _run(self, bench_mod, tmp_path, monkeypatch, models,
             refuse=(), **run_kw):
        gov = _RefusingGov(refuse_models=set(refuse))
        counting_ib = _CountingIB()
        monkeypatch.setitem(
            sys.modules, "opti_oignon.resource_governor", _gov_module(gov)
        )
        monkeypatch.setitem(
            sys.modules,
            "opti_oignon.inference_backend",
            SimpleNamespace(
                get_backend_registry=counting_ib.get_backend_registry
            ),
        )
        runner = bench_mod.BenchmarkRunner(db_path=tmp_path / "bench.db")
        seen = []
        result = runner.run_sync(
            "fast_answer", list(models),
            query_fn=_bench_qfn_factory(seen), **run_kw,
        )
        return runner, result, gov, counting_ib, seen

    def test_refused_model_is_skipped_and_recorded(
        self, bench_mod, tmp_path, monkeypatch
    ):
        runner, result, gov, _, seen = self._run(
            bench_mod, tmp_path, monkeypatch,
            ["too-big", "ok-model"], refuse=("too-big",),
        )
        assert "too-big" not in seen  # never queried
        assert "ok-model" in seen
        conn = sqlite3.connect(tmp_path / "bench.db")
        conn.row_factory = sqlite3.Row
        rows = {
            r["model"]: dict(r)
            for r in conn.execute(
                "SELECT * FROM benchmark_model_scores"
            ).fetchall()
        }
        conn.close()
        assert rows["too-big"]["not_admitted"] == 1
        assert rows["too-big"]["admission_reason"] == "vram_insufficient"
        assert rows["too-big"]["questions_evaluated"] == 0
        assert rows["ok-model"]["not_admitted"] == 0
        assert rows["ok-model"]["questions_evaluated"] > 0

    def test_benchmark_admissions_use_the_benchmark_caller(
        self, bench_mod, tmp_path, monkeypatch
    ):
        _, _, gov, _, _ = self._run(
            bench_mod, tmp_path, monkeypatch, ["ok-model"]
        )
        assert gov.admit_calls == [("ok-model", None, "benchmark")]

    def test_evict_between_is_the_default(
        self, bench_mod, tmp_path, monkeypatch
    ):
        _, _, _, counting_ib, _ = self._run(
            bench_mod, tmp_path, monkeypatch, ["m-a", "m-b"]
        )
        assert counting_ib.unloads == 2

    def test_evict_between_false_skips_eviction(
        self, bench_mod, tmp_path, monkeypatch
    ):
        _, _, _, counting_ib, _ = self._run(
            bench_mod, tmp_path, monkeypatch, ["m-a", "m-b"],
            evict_between=False,
        )
        assert counting_ib.unloads == 0

    def test_refused_model_is_not_evicted_after(
        self, bench_mod, tmp_path, monkeypatch
    ):
        _, _, _, counting_ib, _ = self._run(
            bench_mod, tmp_path, monkeypatch,
            ["too-big", "ok-model"], refuse=("too-big",),
        )
        # only the completed (admitted) model triggers evict-between.
        assert counting_ib.unloads == 1

    def test_old_schema_is_migrated_by_guarded_alters(
        self, bench_mod, tmp_path
    ):
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.execute(
            """CREATE TABLE benchmark_model_scores (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   run_id TEXT NOT NULL,
                   model TEXT NOT NULL,
                   accuracy_avg REAL DEFAULT 0,
                   code_avg REAL DEFAULT 0,
                   structure_avg REAL DEFAULT 0,
                   speed_avg REAL DEFAULT 0,
                   composite REAL DEFAULT 0,
                   questions_evaluated INTEGER DEFAULT 0
               )"""
        )
        conn.commit()
        conn.close()
        bench_mod.ResultsStore(db)
        conn = sqlite3.connect(db)
        cols = {
            r[1]
            for r in conn.execute(
                "PRAGMA table_info(benchmark_model_scores)"
            ).fetchall()
        }
        conn.close()
        assert {"not_admitted", "admission_reason"} <= cols

    def test_model_score_defaults_stay_backward_compatible(self, bench_mod):
        ms = bench_mod.ModelScore(model="m")
        assert ms.not_admitted is False
        assert ms.admission_reason == ""


# ---------------------------------------------------------------------------
# J. Executor funnel wiring (source pins: the funnels are exercised
#    functionally at the governor level; the executor's import web makes
#    source pins the established proof here, the s215 precedent)
# ---------------------------------------------------------------------------


class TestExecutorWiring:
    def test_ast_valid(self):
        ast.parse(EXEC_SRC)

    def test_helpers_defined_once(self):
        assert EXEC_SRC.count("def _governor_admit(") == 1
        assert EXEC_SRC.count("def _governor_hold_ticket(") == 1
        assert EXEC_SRC.count("def _governor_release_ticket(") == 1
        assert EXEC_SRC.count("def _governor_account_load(") == 1

    def test_six_funnels_admit_with_chat_semantics(self):
        # def + refine + main + simple + speculative + cascade + vision.
        assert EXEC_SRC.count("_governor_admit(") == 7
        assert EXEC_SRC.count('caller="chat"') >= 6

    def test_ticket_held_on_the_three_backend_paths(self):
        # def + refine + simple + stream_thread.
        assert EXEC_SRC.count("_governor_hold_ticket(") == 4
        assert EXEC_SRC.count("_governor_release_ticket()") == 4

    def test_main_call_sends_the_admitted_num_ctx(self):
        assert (
            'options["num_ctx"] = int(_gov_decision.num_ctx)' in EXEC_SRC
        )
        assert "_gov_requested_ctx" in EXEC_SRC

    def test_refine_degrades_on_refusal(self):
        assert "Refinement admission refused for" in EXEC_SRC

    def test_simple_uses_the_error_contract(self):
        assert "Simple execution admission refused for" in EXEC_SRC

    def test_speculative_folds_the_pair(self):
        assert "speculative admission refused for" in EXEC_SRC
        assert 'extra_models=[_draft_model] if _draft_model else None' \
            in EXEC_SRC

    def test_cascade_admits_the_first_tier(self):
        assert "cascade admission refused for first tier" in EXEC_SRC

    def test_vision_refuses_the_request_typed(self):
        assert "Resource admission refused for the vision model" in EXEC_SRC
        assert "detect_needs_delegation" in EXEC_SRC

    def test_delegated_funnels_account_loads(self):
        # def + speculative (verify + draft) + cascade + vision.
        assert EXEC_SRC.count("_governor_account_load(") == 5

    def test_no_estop_string_enters_the_executor(self):
        assert "emergency_stop" not in EXEC_SRC

    def test_benchmark_and_pipeline_callers_named(self):
        assert 'caller="pipeline"' in PIPELINES_SRC
        assert 'caller="benchmark"' in BENCH_SRC
        ast.parse(BENCH_SRC)


# ---------------------------------------------------------------------------
# K. Boundary reassert (supersedes the deselected s223 boundary pin)
# ---------------------------------------------------------------------------


class TestBoundaryReassert:
    def test_admit_is_now_the_bloc1_surface(self):
        # Supersedes test_s223 TestModuleConventions::
        # test_no_admission_surface_this_bloc (deselected in pyproject):
        # the Bloc 0 boundary moved by design at S224.
        assert hasattr(rg.ResourceGovernor, "admit")

    def test_ticket_carries_the_44_shape(self):
        names = {f.name for f in dataclasses.fields(rg.AdmissionDecision)}
        assert {
            "admitted", "model", "num_ctx", "num_gpu", "keep_alive",
            "action", "reason", "provenance", "ticket_id",
        } <= names

    def test_ticket_to_dict_round_trips_the_shape(self):
        d = rg.AdmissionDecision(
            admitted=True, model="m", num_ctx=4096, action="admit",
            reason="fits", ticket_id="abc123",
        )
        payload = d.to_dict()
        for key in ("admitted", "model", "num_ctx", "num_gpu",
                    "keep_alive", "action", "reason", "provenance",
                    "ticket_id"):
            assert key in payload

    def test_module_conventions_hold(self):
        assert rg.checkpoint_before_apply is True
        assert rg.FEATURE_AVAILABLE is True


# ---------------------------------------------------------------------------
# L. Doc and config pins (red-before provable on the pristine tree)
# ---------------------------------------------------------------------------


class TestDocAndConfigPins:
    def test_spec_section_10_names_the_landed_keys(self):
        spec = _SPEC_PATH.read_text(encoding="utf-8")
        section_10 = spec.split("## 10. Configuration", 1)[1].split(
            "## 11.", 1
        )[0]
        assert "ceiling_floor_gb (4.0" in section_10
        assert "decisions_ring_size (200" in section_10

    def test_pyproject_deselects_the_superseded_boundary_pin(self):
        pyproject = _PYPROJECT_PATH.read_text(encoding="utf-8")
        assert (
            "--deselect=tests/test_s223_governor_bloc0.py::"
            "TestModuleConventions::test_no_admission_surface_this_bloc"
        ) in pyproject


# ---------------------------------------------------------------------------
# M. Helper semantics (duration parsing, expiry coercion, evictable_now)
# ---------------------------------------------------------------------------


class TestHelpers:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("30m", 1800.0),
            ("1h", 3600.0),
            ("90s", 90.0),
            ("300", 300.0),
            (300, 300.0),
            (0, None),
            ("", None),
            ("garbage", None),
            (None, None),
        ],
    )
    def test_parse_duration(self, value, expected):
        assert rg._parse_duration_s(value) == expected

    def test_coerce_epoch_forms(self):
        assert rg._coerce_epoch_s(123.5) == 123.5
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert rg._coerce_epoch_s(dt) == dt.timestamp()
        iso = rg._coerce_epoch_s("2026-01-01T00:00:00Z")
        assert iso == dt.timestamp()
        assert rg._coerce_epoch_s(True) is None
        assert rg._coerce_epoch_s("junk") is None
        assert rg._coerce_epoch_s(None) is None

    def test_evictable_counts_only_idle_coercible_entries(
        self, gov_factory
    ):
        warmup = KeepAliveWarmup([
            _loaded("idle-m", 3.0, expires_in_s=60),      # idle 1740
            _loaded("fresh-m", 2.0, expires_in_s=1790),   # idle 10
            SimpleNamespace(                              # non-coercible
                name="weird-m", size_vram=4 * GB,
                expires_at=object(), context_length=None, digest=None,
            ),
        ])
        gov = gov_factory(warmup=warmup)
        snapshot = gov.refresh(force=True)
        assert gov._evictable_now_gb(snapshot) == pytest.approx(3.0)

    def test_no_keep_alive_means_nothing_evictable(self, gov_factory):
        warmup = KeepAliveWarmup([_loaded("idle-m", 3.0, expires_in_s=60)])
        warmup.keep_alive = None
        gov = gov_factory(warmup=warmup)
        snapshot = gov.refresh(force=True)
        assert gov._evictable_now_gb(snapshot) == 0.0
