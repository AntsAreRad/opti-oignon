#!/usr/bin/env python3
"""S223 -- Resource Governor Bloc 0: the measurement layer.

Per-fix suite for opti_oignon/resource_governor.py against the
RESOURCE_GOVERNOR_SPEC.md Section 3 / 3.1 / 3.2 contract and the S223
read-gate decisions (DI-1..DI-10), covering the spec's Bloc 0
container-provable list verbatim:

- fake ps responses in BOTH client forms (the CC-01 idiom), exercised
  through a REAL ModelWarmup so the dual-form handling is consumed, not
  reimplemented in the fake;
- fake registry state (S2) and the estimation chain
  (learned > static table > file size > unknown, never "too large");
- fake clock for TTL expiry, the fast-path primitive and the four eager
  invalidation hooks;
- fake load outcomes driving the learning rules exactly as specced
  (fast-down with the floor, slow-up toward the configured capacity,
  failure-resets-the-counter);
- /proc absent and present; provenance honesty; fail-open on unknown
  capacity with the logged warning;
- the adapt store schema, the name+digest keying, the prune-by-count
  ring, parameterized SQL;
- the config loader (defaults, overrides, null capacity, invalid YAML)
  and the shipped resource_governor.yaml;
- the doc pins (the ATREST_INVENTORY.md row and the roadmap roll),
  red-before proven on the pristine tree.

Host-assured (named, never simulated here): the real meaning of
size_vram on the real driver. Nothing else (spec Section 11, Bloc 0).

Isolation: the established spec_from_file_location idiom with
sys.modules pre-seeding (an ollama stub and an opti_oignon package stub
carrying a real __path__), so the module chain resolves by path without
executing opti_oignon/__init__.py.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_BASE = Path(__file__).resolve().parent.parent
_MODULE_PATH = _BASE / "opti_oignon" / "resource_governor.py"
_YAML_PATH = _BASE / "opti_oignon" / "config" / "resource_governor.yaml"
_ATREST_PATH = _BASE / "ATREST_INVENTORY.md"
_ROADMAP_PATH = _BASE / "ROADMAP_POST_AUDIT.md"

SRC = _MODULE_PATH.read_text(encoding="utf-8")

GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Isolated module loading (the established idiom)
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


# The sweep-order pollution class (documented at S220): earlier suites in
# the regression selection seed an "opti_oignon" package stub WITHOUT
# __path__ (and, in test_s140_coverage, a synthetic db_utils), so package
# machinery cannot resolve submodules that no earlier suite happened to
# cache. Pre-load the governor's four conditional dependencies by path,
# REUSING whatever a prior suite already put in sys.modules (never
# replacing it), so this suite is order-independent: standalone and
# in-sweep resolve the same flags.
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
mw = sys.modules.get("opti_oignon.model_warmup")


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class CountingWarmup:
    """Injectable S1 fake counting its reads (TTL/invalidation proofs)."""

    def __init__(self, models=None):
        self.calls = 0
        self.models = list(models or [])

    def get_loaded_models(self):
        self.calls += 1
        return list(self.models)


class RaisingWarmup:
    def get_loaded_models(self):
        raise RuntimeError("boom")


def _ps_entry(name, size_vram=0, digest=None, context_length=None):
    return SimpleNamespace(
        name=name,
        size_vram=size_vram,
        expires_at=None,
        context_length=context_length,
        digest=digest,
    )


class FakeBackend:
    """An in-process backend exposing the _loaded_models dict idiom."""

    name = "llama_cpp"

    def __init__(self, loaded=None, infos=None):
        self._loaded_models = dict(loaded or {})
        self._infos = dict(infos or {})

    def model_info(self, model_name):
        return self._infos.get(model_name)


class NoResidentBackend:
    name = "other"


class FakeRegistry:
    def __init__(self, *backends):
        self._backends = list(backends)

    def backends(self):
        return list(self._backends)


class RaisingRegistry:
    def backends(self):
        raise RuntimeError("registry down")


class _ByteSize:
    """Mimics the typed client's ByteSize: int-coercible, not an int."""

    def __init__(self, value: int):
        self._value = int(value)

    def __int__(self) -> int:
        return self._value


class _FakeExpires:
    def __init__(self, ts: float):
        self._ts = ts

    def timestamp(self) -> float:
        return self._ts


def _meminfo_file(tmp_path: Path, available_kb: int = 8 * 1024 * 1024) -> Path:
    p = tmp_path / "meminfo"
    p.write_text(
        "MemTotal:       16384000 kB\n"
        f"MemAvailable:   {available_kb} kB\n"
        "SwapTotal:       2097152 kB\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def no_psutil(monkeypatch):
    """Force the psutil fallback to fail (None in sys.modules raises
    ImportError on import), making the 0.0 fail-open branch reachable."""
    monkeypatch.setitem(sys.modules, "psutil", None)


@pytest.fixture
def gov_factory(tmp_path):
    """ResourceGovernor factory with deterministic injectable defaults:
    no warmup, no registry, defaults config, a tmp DB, a real tmp meminfo
    (so "S4-ram" is deterministic), a fake clock."""

    counter = {"n": 0}

    def make(**kw):
        counter["n"] += 1
        kw.setdefault("config_path", tmp_path / "absent.yaml")
        kw.setdefault("db_path", tmp_path / f"gov_{counter['n']}.db")
        kw.setdefault("warmup", None)
        kw.setdefault("registry", None)
        kw.setdefault("clock", FakeClock())
        kw.setdefault("meminfo_path", _meminfo_file(tmp_path))
        return rg.ResourceGovernor(**kw)

    return make


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "resource_governor.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Module conventions and the Bloc 0 boundary
# ---------------------------------------------------------------------------


class TestModuleConventions:
    def test_checkpoint_before_apply_hardcoded_true(self):
        assert rg.checkpoint_before_apply is True

    def test_feature_available_sentinel(self):
        assert rg.FEATURE_AVAILABLE is True

    def test_ast_valid(self):
        ast.parse(SRC)

    def test_pure_ascii_no_emoji(self):
        offending = [
            (i + 1, line)
            for i, line in enumerate(SRC.splitlines())
            if any(ord(ch) > 127 for ch in line)
        ]
        assert offending == []

    def test_conditional_imports_resolved_in_container(self):
        # The pinned container dep set resolves the whole source chain.
        assert rg.DB_UTILS_AVAILABLE is True
        assert rg.MODEL_WARMUP_AVAILABLE is True
        assert rg.INFERENCE_BACKEND_AVAILABLE is True
        assert rg.SPECULATIVE_AVAILABLE is True

    def test_no_admission_surface_this_bloc(self):
        # Bloc 0 boundary: measurement only. admit() is Bloc 1 territory.
        assert not hasattr(rg.ResourceGovernor, "admit")

    def test_static_table_not_duplicated(self):
        # S3 is reused BY IMPORT: the table literal must not be copied in.
        assert "_VRAM_PER_BILLION_PARAMS: dict" not in SRC
        assert '"Q4_K_M": 0.58' not in SRC

    def test_relax_knobs_are_module_constants(self):
        assert rg._CEILING_RELAX_AFTER_SUCCESSES == 5
        assert rg._CEILING_RELAX_STEP_GB == 1.0


# ---------------------------------------------------------------------------
# Config loader (Section 10)
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_defaults_when_file_missing(self, tmp_path):
        cfg = rg.load_config(tmp_path / "absent.yaml")
        assert cfg == rg.GovernorConfig()
        assert cfg.total_vram_gb is None
        assert cfg.snapshot_ttl_s == 2.0
        assert cfg.safety_margin_gb == 1.5
        assert cfg.kv_coefficient == 0.5
        assert cfg.ceiling_floor_gb == 4.0
        assert cfg.decisions_ring_size == 200

    def test_shipped_file_mirrors_defaults(self):
        assert _YAML_PATH.is_file()
        cfg = rg.load_config(_YAML_PATH)
        assert cfg == rg.GovernorConfig()

    def test_total_vram_override_and_null(self, tmp_path):
        p = _write_yaml(tmp_path, "total_vram_gb: 24\n")
        assert rg.load_config(p).total_vram_gb == 24.0
        p2 = tmp_path / "null.yaml"
        p2.write_text("total_vram_gb: null\n", encoding="utf-8")
        assert rg.load_config(p2).total_vram_gb is None

    def test_measurement_subset_overrides(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            "enabled: false\n"
            "snapshot_ttl_s: 0.5\n"
            "safety_margin_gb: 2.5\n"
            "kv_coefficient: 0.25\n"
            "ceiling_floor_gb: 6\n"
            "decisions_ring_size: 3\n",
        )
        cfg = rg.load_config(p)
        assert cfg.enabled is False
        assert cfg.snapshot_ttl_s == 0.5
        assert cfg.safety_margin_gb == 2.5
        assert cfg.kv_coefficient == 0.25
        assert cfg.ceiling_floor_gb == 6.0
        assert cfg.decisions_ring_size == 3

    def test_invalid_yaml_warns_and_defaults(self, tmp_path, caplog):
        p = _write_yaml(tmp_path, "enabled: [unclosed\n")
        with caplog.at_level(logging.WARNING, logger=rg.logger.name):
            cfg = rg.load_config(p)
        assert cfg == rg.GovernorConfig()
        assert any("Failed to parse" in r.message for r in caplog.records)

    def test_non_mapping_root_defaults(self, tmp_path, caplog):
        p = _write_yaml(tmp_path, "- just\n- a\n- list\n")
        with caplog.at_level(logging.WARNING, logger=rg.logger.name):
            cfg = rg.load_config(p)
        assert cfg == rg.GovernorConfig()
        assert any("not a mapping" in r.message for r in caplog.records)

    def test_bad_types_fall_back_to_defaults(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            "snapshot_ttl_s: not-a-number\n"
            "decisions_ring_size: 0\n"
            "enabled: definitely\n",
        )
        cfg = rg.load_config(p)
        assert cfg.snapshot_ttl_s == 2.0
        assert cfg.decisions_ring_size >= 1
        assert cfg.enabled is True

    def test_nested_later_bloc_blocks_parsed(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            "ctx_ladder: [8192, 4096]\n"
            "ctx_floor:\n  chat: 2048\n  benchmark: 1024\n"
            "pressure:\n  soft_threshold: 0.7\n  hard_threshold: 0.9\n"
            "queue:\n  enabled_per_caller:\n    chat: true\n"
            "  depth: 4\n  wait_s: 12\n"
            "rlimits:\n  enabled: true\n  as_gb: 30\n"
            "ollama_limits:\n  max_loaded_models: 2\n  spawn_applies: false\n",
        )
        cfg = rg.load_config(p)
        assert cfg.ctx_ladder == [8192, 4096]
        assert cfg.ctx_floor["chat"] == 2048
        assert cfg.ctx_floor["benchmark"] == 1024
        assert cfg.ctx_floor["pipeline"] == 4096  # default preserved
        assert cfg.pressure_soft_threshold == 0.7
        assert cfg.pressure_hard_threshold == 0.9
        assert cfg.queue_enabled_per_caller == {"chat": True}
        assert cfg.queue_depth == 4
        assert cfg.queue_wait_s == 12.0
        assert cfg.rlimits_enabled is True
        assert cfg.rlimits_as_gb == 30.0
        assert cfg.ollama_max_loaded_models == 2
        assert cfg.ollama_spawn_applies is False


# ---------------------------------------------------------------------------
# S4 RAM read (/proc absent and present; DI-1 local equivalent)
# ---------------------------------------------------------------------------


class TestRamRead:
    def test_meminfo_present(self, tmp_path):
        p = _meminfo_file(tmp_path, available_kb=8 * 1024 * 1024)
        assert rg._read_available_ram_mb(p) == pytest.approx(8192.0)

    def test_meminfo_absent_and_no_psutil_is_zero(self, tmp_path, no_psutil):
        assert rg._read_available_ram_mb(tmp_path / "absent") == 0.0

    def test_meminfo_malformed_and_no_psutil_is_zero(self, tmp_path, no_psutil):
        p = tmp_path / "meminfo"
        p.write_text("garbage without the key\n", encoding="utf-8")
        assert rg._read_available_ram_mb(p) == 0.0

    def test_meminfo_absent_psutil_fallback(self, tmp_path):
        # psutil answers a positive value when /proc is unreadable; the
        # importorskip documents the dependency without simulating it.
        pytest.importorskip("psutil")
        assert rg._read_available_ram_mb(tmp_path / "absent") > 0.0

    def test_helper_is_local_by_decision(self):
        # DI-1: the S171 idiom is replicated locally, not imported from
        # smart_router (whose pre-flight stays where it is).
        assert "def _read_available_ram_mb(" in SRC
        assert "from opti_oignon.smart_router" not in SRC
        assert "MemAvailable" in SRC


# ---------------------------------------------------------------------------
# KV-cache increment (DI-4)
# ---------------------------------------------------------------------------


class TestKvCoefficient:
    def test_zero_or_none_costs_nothing(self, gov_factory):
        gov = gov_factory()
        assert gov.estimate_kv_cache_gb(None) == 0.0
        assert gov.estimate_kv_cache_gb(0) == 0.0
        assert gov.estimate_kv_cache_gb(-5) == 0.0

    def test_default_coefficient(self, gov_factory):
        gov = gov_factory()
        assert gov.estimate_kv_cache_gb(1024) == pytest.approx(0.5)
        assert gov.estimate_kv_cache_gb(8192) == pytest.approx(4.0)

    def test_config_tunable(self, tmp_path, gov_factory):
        p = _write_yaml(tmp_path, "kv_coefficient: 0.25\n")
        gov = gov_factory(config_path=p)
        assert gov.estimate_kv_cache_gb(8192) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# S1: the ps view in BOTH client forms (CC-01), through a REAL ModelWarmup
# ---------------------------------------------------------------------------


class TestS1DualForm:
    def _real_warmup(self, monkeypatch, ps_callable):
        assert mw is not None
        fake_client = SimpleNamespace(ps=ps_callable)
        monkeypatch.setattr(mw, "_ollama", fake_client)
        monkeypatch.setattr(mw, "OLLAMA_AVAILABLE", True)
        return mw.ModelWarmup()

    def test_dict_form(self, monkeypatch, gov_factory):
        warmup = self._real_warmup(
            monkeypatch,
            lambda: {
                "models": [
                    {
                        "name": "qwen3:32b",
                        "size_vram": 20 * GB,
                        "digest": "sha256:abc",
                        "context_length": 32768,
                    }
                ]
            },
        )
        gov = gov_factory(warmup=warmup)
        snap = gov.refresh(force=True)
        assert "S1" in snap.sources
        assert len(snap.loaded) == 1
        view = snap.loaded[0]
        assert view.name == "qwen3:32b"
        assert view.size_vram_bytes == 20 * GB
        assert view.digest == "sha256:abc"
        assert view.context_length == 32768
        assert snap.vram_in_use_gb == pytest.approx(20.0)

    def test_object_form_with_bytesize_and_model_field(
        self, monkeypatch, gov_factory
    ):
        entry = SimpleNamespace(
            name=None,
            model="phi4:14b",
            size_vram=_ByteSize(9 * GB),
            expires_at=_FakeExpires(1234.5),
            context_length=16384,
            digest="sha256:def",
        )
        warmup = self._real_warmup(
            monkeypatch, lambda: SimpleNamespace(models=[entry])
        )
        gov = gov_factory(warmup=warmup)
        snap = gov.refresh(force=True)
        assert "S1" in snap.sources
        view = snap.loaded[0]
        assert view.name == "phi4:14b"
        assert view.size_vram_bytes == 9 * GB
        assert view.expires_at == pytest.approx(1234.5)
        assert view.digest == "sha256:def"

    def test_warmup_absent_means_no_s1(self, gov_factory):
        snap = gov_factory(warmup=None).refresh(force=True)
        assert "S1" not in snap.sources
        assert snap.loaded == []

    def test_ollama_package_absent_means_no_s1(self, monkeypatch, gov_factory):
        warmup = self._real_warmup(monkeypatch, lambda: {"models": []})
        monkeypatch.setattr(mw, "OLLAMA_AVAILABLE", False)
        snap = gov_factory(warmup=warmup).refresh(force=True)
        assert "S1" not in snap.sources

    def test_raising_warmup_degrades_without_crash(self, gov_factory):
        snap = gov_factory(warmup=RaisingWarmup()).refresh(force=True)
        assert "S1" not in snap.sources
        assert snap.loaded == []


# ---------------------------------------------------------------------------
# S2: backend-resident set and the estimation chain (DI-7)
# ---------------------------------------------------------------------------


class TestS2BackendResident:
    def test_static_table_estimate(self, gov_factory):
        info = SimpleNamespace(
            parameter_size="7B",
            quantization_level="Q4_K_M",
            path=None,
            size=None,
        )
        registry = FakeRegistry(
            FakeBackend({"local-7b": object()}, {"local-7b": info})
        )
        gov = gov_factory(registry=registry)
        snap = gov.refresh(force=True)
        assert "S2" in snap.sources
        assert "S3" in snap.sources
        assert len(snap.backend_resident) == 1
        view = snap.backend_resident[0]
        assert view.basis == "static_table"
        assert view.estimated_gb == pytest.approx(7.0 * 0.58)
        assert snap.vram_in_use_gb == pytest.approx(7.0 * 0.58)

    def test_s1_models_never_double_counted(self, gov_factory):
        warmup = CountingWarmup([_ps_entry("shared", size_vram=2 * GB)])
        registry = FakeRegistry(FakeBackend({"shared": object()}))
        snap = gov_factory(warmup=warmup, registry=registry).refresh(force=True)
        assert [v.name for v in snap.loaded] == ["shared"]
        assert snap.backend_resident == []
        assert snap.vram_in_use_gb == pytest.approx(2.0)

    def test_learned_cost_preferred_over_static(self, gov_factory):
        info = SimpleNamespace(
            parameter_size="7B", quantization_level="Q4_K_M", path=None, size=None
        )
        registry = FakeRegistry(FakeBackend({"m": object()}, {"m": info}))
        gov = gov_factory(registry=registry)
        gov.store.record_model_cost("m", "sha256:x", 3 * GB, num_ctx=8192)
        snap = gov.refresh(force=True)
        view = snap.backend_resident[0]
        assert view.basis == "learned"
        assert view.estimated_gb == pytest.approx(3.0)

    def test_file_size_from_size_string(self, gov_factory):
        info = SimpleNamespace(
            parameter_size=None, quantization_level=None, path=None, size="2.0GB"
        )
        registry = FakeRegistry(FakeBackend({"gguf": object()}, {"gguf": info}))
        snap = gov_factory(registry=registry).refresh(force=True)
        view = snap.backend_resident[0]
        assert view.basis == "file_size"
        assert view.estimated_gb == pytest.approx(2.0)

    def test_file_size_from_real_path(self, tmp_path, gov_factory):
        blob = tmp_path / "weights.gguf"
        blob.write_bytes(b"\0" * (3 * 1024 * 1024))
        info = SimpleNamespace(
            parameter_size=None,
            quantization_level=None,
            path=str(blob),
            size=None,
        )
        registry = FakeRegistry(FakeBackend({"tiny": object()}, {"tiny": info}))
        snap = gov_factory(registry=registry).refresh(force=True)
        view = snap.backend_resident[0]
        assert view.basis == "file_size"
        assert view.estimated_gb == pytest.approx(3.0 / 1024.0)

    def test_unknown_model_never_too_large(self, gov_factory):
        registry = FakeRegistry(FakeBackend({"mystery": object()}))
        snap = gov_factory(registry=registry).refresh(force=True)
        view = snap.backend_resident[0]
        assert view.basis == "unknown"
        assert view.estimated_gb is None
        # An unknown estimate contributes nothing to in-use.
        assert snap.vram_in_use_gb == pytest.approx(0.0)

    def test_registry_absent_means_no_s2(self, gov_factory):
        snap = gov_factory(registry=None).refresh(force=True)
        assert "S2" not in snap.sources
        assert snap.backend_resident == []

    def test_raising_registry_degrades_without_crash(self, gov_factory):
        snap = gov_factory(registry=RaisingRegistry()).refresh(force=True)
        assert "S2" not in snap.sources

    def test_backend_without_resident_dict_is_ignored(self, gov_factory):
        registry = FakeRegistry(NoResidentBackend())
        snap = gov_factory(registry=registry).refresh(force=True)
        assert "S2" in snap.sources  # the registry read path answered
        assert snap.backend_resident == []


class TestEstimateModelVram:
    def test_observed_beats_everything(self, gov_factory):
        warmup = CountingWarmup([_ps_entry("m", size_vram=5 * GB, digest="d")])
        gov = gov_factory(warmup=warmup)
        gov.store.record_model_cost("m", "d", 1 * GB)
        gov.refresh(force=True)
        est, basis = gov.estimate_model_vram_gb("m")
        assert basis == "observed"
        assert est == pytest.approx(5.0)

    def test_learned_when_not_loaded(self, gov_factory):
        gov = gov_factory()
        gov.store.record_model_cost("m", "d", 6 * GB)
        est, basis = gov.estimate_model_vram_gb("m", "d")
        assert basis == "learned"
        assert est == pytest.approx(6.0)

    def test_static_via_registry_metadata(self, gov_factory):
        info = SimpleNamespace(
            parameter_size="3.2B", quantization_level="Q8_0", path=None, size=None
        )
        registry = FakeRegistry(FakeBackend({}, {"never-loaded": info}))
        gov = gov_factory(registry=registry)
        est, basis = gov.estimate_model_vram_gb("never-loaded")
        assert basis == "static_table"
        assert est == pytest.approx(3.2 * 1.00)

    def test_unknown_yields_none_not_too_large(self, gov_factory):
        est, basis = gov_factory().estimate_model_vram_gb("ghost")
        assert est is None
        assert basis == "unknown"


# ---------------------------------------------------------------------------
# TTL cache, the fast-path primitive and eager invalidation (DI-5, DI-9)
# ---------------------------------------------------------------------------


class TestTtlCache:
    def test_fresh_snapshot_served_from_cache(self, gov_factory):
        warmup = CountingWarmup()
        clock = FakeClock()
        gov = gov_factory(warmup=warmup, clock=clock)
        first = gov.get_snapshot()
        again = gov.get_snapshot()
        assert again is first
        assert warmup.calls == 1

    def test_ttl_expiry_rebuilds(self, gov_factory):
        warmup = CountingWarmup()
        clock = FakeClock()
        gov = gov_factory(warmup=warmup, clock=clock)
        first = gov.get_snapshot()
        clock.advance(2.5)  # past the 2.0 s default TTL
        second = gov.get_snapshot()
        assert second is not first
        assert warmup.calls == 2

    def test_ttl_is_config_tunable(self, tmp_path, gov_factory):
        p = _write_yaml(tmp_path, "snapshot_ttl_s: 10.0\n")
        warmup = CountingWarmup()
        clock = FakeClock()
        gov = gov_factory(config_path=p, warmup=warmup, clock=clock)
        gov.get_snapshot()
        clock.advance(5.0)  # within the tuned TTL
        gov.get_snapshot()
        assert warmup.calls == 1

    def test_force_refresh_always_rebuilds(self, gov_factory):
        warmup = CountingWarmup()
        gov = gov_factory(warmup=warmup)
        gov.refresh(force=True)
        gov.refresh(force=True)
        assert warmup.calls == 2

    def test_fast_path_serves_stale_and_refreshes_in_background(
        self, gov_factory
    ):
        warmup = CountingWarmup()
        clock = FakeClock()
        gov = gov_factory(warmup=warmup, clock=clock)
        first = gov.get_snapshot()
        clock.advance(5.0)
        served = gov.get_snapshot_fast()
        # The stale snapshot is served immediately (conservative use).
        assert served is first
        deadline = time.time() + 2.0
        while warmup.calls < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert warmup.calls >= 2
        deadline = time.time() + 2.0
        while gov.get_snapshot_fast() is first and time.time() < deadline:
            time.sleep(0.01)
        assert gov.get_snapshot_fast() is not first

    def test_fast_path_first_call_builds_synchronously(self, gov_factory):
        warmup = CountingWarmup()
        gov = gov_factory(warmup=warmup)
        snap = gov.get_snapshot_fast()
        assert warmup.calls == 1
        assert snap.loaded == []


class TestEagerInvalidation:
    @pytest.mark.parametrize(
        "hook,args",
        [
            ("invalidate_on_load", ("m",)),
            ("invalidate_on_evict", ("m",)),
            ("invalidate_on_estop_drain", ()),
            ("invalidate_on_resume", ()),
        ],
    )
    def test_each_hook_drops_the_cache(self, gov_factory, hook, args):
        warmup = CountingWarmup()
        clock = FakeClock()
        gov = gov_factory(warmup=warmup, clock=clock)
        gov.get_snapshot()
        assert warmup.calls == 1
        getattr(gov, hook)(*args)
        gov.get_snapshot()  # clock unchanged: only invalidation explains it
        assert warmup.calls == 2

    def test_hooks_exist_unwired(self):
        # Bloc 1 wires the callers; this bloc only lands the entry points.
        for hook in (
            "invalidate_on_load",
            "invalidate_on_evict",
            "invalidate_on_estop_drain",
            "invalidate_on_resume",
        ):
            assert SRC.count(f"def {hook}") == 1


class TestPostLoadAttribution:
    def test_attribution_records_cost_with_ctx_and_digest(self, gov_factory):
        warmup = CountingWarmup()
        gov = gov_factory(warmup=warmup)
        gov.invalidate_on_load("qwen3:32b", requested_num_ctx=8192)
        warmup.models = [
            _ps_entry("qwen3:32b", size_vram=21 * GB, digest="sha256:abc")
        ]
        gov.get_snapshot()
        learned = gov.store.get_model_cost("qwen3:32b", "sha256:abc")
        assert learned is not None
        assert learned["size_vram_bytes"] == 21 * GB
        assert learned["num_ctx"] == 8192
        assert learned["digest"] == "sha256:abc"

    def test_pending_cleared_after_attribution(self, gov_factory):
        warmup = CountingWarmup([_ps_entry("m", size_vram=4 * GB, digest="d")])
        gov = gov_factory(warmup=warmup)
        gov.invalidate_on_load("m", requested_num_ctx=4096)
        gov.refresh(force=True)
        first = gov.store.get_model_cost("m")
        # A later, different observation must NOT be re-attributed.
        warmup.models = [_ps_entry("m", size_vram=9 * GB, digest="d")]
        gov.refresh(force=True)
        assert (
            gov.store.get_model_cost("m")["size_vram_bytes"]
            == first["size_vram_bytes"]
        )

    def test_zero_size_view_does_not_attribute(self, gov_factory):
        warmup = CountingWarmup([_ps_entry("m", size_vram=0)])
        gov = gov_factory(warmup=warmup)
        gov.invalidate_on_load("m")
        gov.refresh(force=True)
        assert gov.store.get_model_cost("m") is None
        # The pending entry survives until a positive view arrives.
        warmup.models = [_ps_entry("m", size_vram=2 * GB)]
        gov.invalidate_on_evict()
        gov.refresh(force=True)
        assert gov.store.get_model_cost("m")["size_vram_bytes"] == 2 * GB


# ---------------------------------------------------------------------------
# Ceiling learning: fast down, slow up, the config floor (Section 3.2, DI-8)
# ---------------------------------------------------------------------------


class TestCeilingLearning:
    def _store(self, tmp_path):
        return rg.AdaptStore(tmp_path / "adapt.db")

    def test_fresh_store_has_no_ceiling(self, tmp_path):
        assert self._store(tmp_path).get_learned_ceiling() is None

    def test_fast_down_basic(self, tmp_path):
        store = self._store(tmp_path)
        new = store.record_load_failure(20.0, safety_margin_gb=1.5, floor_gb=4.0)
        assert new == pytest.approx(18.5)
        assert store.get_learned_ceiling() == pytest.approx(18.5)

    def test_fast_down_respects_the_floor(self, tmp_path):
        store = self._store(tmp_path)
        new = store.record_load_failure(5.0, safety_margin_gb=1.5, floor_gb=4.0)
        assert new == pytest.approx(4.0)

    def test_fast_down_is_monotonic_min(self, tmp_path):
        store = self._store(tmp_path)
        store.record_load_failure(20.0, 1.5, 4.0)
        new = store.record_load_failure(30.0, 1.5, 4.0)
        # A later failure at a higher observed in-use cannot raise it.
        assert new == pytest.approx(18.5)

    def test_success_below_ceiling_changes_nothing(self, tmp_path):
        store = self._store(tmp_path)
        store.record_load_failure(20.0, 1.5, 4.0)
        assert store.record_load_success(10.0, None) == pytest.approx(18.5)
        assert store.get_learned_ceiling() == pytest.approx(18.5)

    def test_slow_up_after_five_consecutive_successes_above(self, tmp_path):
        store = self._store(tmp_path)
        store.record_load_failure(20.0, 1.5, 4.0)  # ceiling 18.5
        for _ in range(4):
            assert store.record_load_success(19.0, None) == pytest.approx(18.5)
        assert store.record_load_success(19.0, None) == pytest.approx(19.5)

    def test_slow_up_counter_resets_after_a_relax(self, tmp_path):
        store = self._store(tmp_path)
        store.record_load_failure(20.0, 1.5, 4.0)
        for _ in range(5):
            store.record_load_success(19.0, None)  # -> 19.5
        for _ in range(4):
            assert store.record_load_success(20.0, None) == pytest.approx(19.5)
        assert store.record_load_success(20.0, None) == pytest.approx(20.5)

    def test_slow_up_capped_at_configured_capacity(self, tmp_path):
        store = self._store(tmp_path)
        store.record_load_failure(20.0, 1.5, 4.0)  # 18.5
        for _ in range(5):
            store.record_load_success(19.0, configured_capacity_gb=19.2)
        assert store.get_learned_ceiling() == pytest.approx(19.2)

    def test_failure_resets_the_success_counter(self, tmp_path):
        store = self._store(tmp_path)
        store.record_load_failure(20.0, 1.5, 4.0)  # 18.5
        for _ in range(3):
            store.record_load_success(19.0, None)
        store.record_load_failure(20.0, 1.5, 4.0)  # reset; ceiling still 18.5
        for _ in range(4):
            assert store.record_load_success(19.0, None) == pytest.approx(18.5)
        assert store.record_load_success(19.0, None) == pytest.approx(19.5)

    def test_success_without_learned_ceiling_is_none(self, tmp_path):
        assert self._store(tmp_path).record_load_success(10.0, 24.0) is None

    def test_governor_passthrough_uses_config_and_invalidates(
        self, tmp_path, gov_factory
    ):
        p = _write_yaml(
            tmp_path, "safety_margin_gb: 2.0\nceiling_floor_gb: 5.0\n"
        )
        warmup = CountingWarmup()
        gov = gov_factory(config_path=p, warmup=warmup)
        gov.get_snapshot()
        assert warmup.calls == 1
        new = gov.record_load_failure(observed_in_use_gb=6.0)
        assert new == pytest.approx(5.0)  # floored at the config floor
        gov.get_snapshot()
        assert warmup.calls == 2  # the cache was eagerly invalidated


class TestCapacityComposition:
    def test_configured_only(self, tmp_path, gov_factory):
        p = _write_yaml(tmp_path, "total_vram_gb: 24\n")
        snap = gov_factory(config_path=p).refresh(force=True)
        assert snap.capacity_gb == pytest.approx(24.0)
        assert snap.capacity_source == "config"
        assert "S4-capacity-config" in snap.sources
        assert "S4-capacity-learned" not in snap.sources

    def test_learned_only(self, gov_factory):
        gov = gov_factory()
        gov.store.record_load_failure(20.0, 1.5, 4.0)
        snap = gov.refresh(force=True)
        assert snap.capacity_gb == pytest.approx(18.5)
        assert snap.capacity_source == "learned"
        assert "S4-capacity-learned" in snap.sources

    def test_min_of_configured_and_learned(self, tmp_path, gov_factory):
        p = _write_yaml(tmp_path, "total_vram_gb: 24\n")
        gov = gov_factory(config_path=p)
        gov.store.record_load_failure(20.0, 1.5, 4.0)  # learned 18.5
        snap = gov.refresh(force=True)
        assert snap.capacity_gb == pytest.approx(18.5)
        assert snap.capacity_source == "config+learned"

    def test_available_is_capacity_minus_in_use_no_margin(
        self, tmp_path, gov_factory
    ):
        # The safety margin belongs to the Bloc 1 fit math (Section 4.2),
        # not to the raw snapshot.
        p = _write_yaml(tmp_path, "total_vram_gb: 24\n")
        warmup = CountingWarmup([_ps_entry("m", size_vram=10 * GB)])
        snap = gov_factory(config_path=p, warmup=warmup).refresh(force=True)
        assert snap.vram_available_gb == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# Fail-open on unknown capacity (Section 3.1) and provenance honesty
# ---------------------------------------------------------------------------


class TestFailOpenCapacityUnknown:
    def test_vram_half_disabled_with_honest_status(self, gov_factory):
        snap = gov_factory().refresh(force=True)
        assert snap.capacity_gb is None
        assert snap.vram_status == "disabled_capacity_unknown"
        assert snap.vram_available_gb is None

    def test_warning_logged_once(self, gov_factory, caplog):
        gov = gov_factory()
        with caplog.at_level(logging.WARNING, logger=rg.logger.name):
            gov.refresh(force=True)
            gov.refresh(force=True)
        hits = [
            r for r in caplog.records if "capacity unknown" in r.message
        ]
        assert len(hits) == 1
        assert "fail-open" in hits[0].message

    def test_ram_half_still_applies(self, gov_factory):
        snap = gov_factory().refresh(force=True)
        assert snap.ram_available_mb == pytest.approx(8192.0)
        assert "S4-ram" in snap.sources

    def test_status_ok_when_capacity_known(self, tmp_path, gov_factory):
        p = _write_yaml(tmp_path, "total_vram_gb: 24\n")
        snap = gov_factory(config_path=p).refresh(force=True)
        assert snap.vram_status == "ok"


class TestProvenanceHonesty:
    def test_everything_absent_is_an_empty_list(
        self, tmp_path, gov_factory, no_psutil
    ):
        gov = gov_factory(meminfo_path=tmp_path / "absent_meminfo")
        snap = gov.refresh(force=True)
        assert snap.sources == []
        assert snap.ram_available_mb == 0.0

    def test_full_chain_exact_set(self, tmp_path, gov_factory):
        p = _write_yaml(tmp_path, "total_vram_gb: 24\n")
        warmup = CountingWarmup([_ps_entry("loaded", size_vram=1 * GB)])
        info = SimpleNamespace(
            parameter_size="7B", quantization_level="Q4_K_M", path=None, size=None
        )
        registry = FakeRegistry(FakeBackend({"resident": object()}, {"resident": info}))
        gov = gov_factory(config_path=p, warmup=warmup, registry=registry)
        snap = gov.refresh(force=True)
        assert set(snap.sources) == {
            "S1",
            "S2",
            "S3",
            "S4-capacity-config",
            "S4-ram",
        }

    def test_learned_ceiling_appears_in_provenance(self, gov_factory):
        gov = gov_factory()
        gov.store.record_load_failure(20.0, 1.5, 4.0)
        snap = gov.refresh(force=True)
        assert "S4-capacity-learned" in snap.sources

    def test_to_dict_carries_provenance_and_status(self, gov_factory):
        snap = gov_factory().refresh(force=True)
        d = snap.to_dict()
        assert d["vram_status"] == "disabled_capacity_unknown"
        assert d["sources"] == snap.sources
        assert "capacity_gb" in d and "ram_available_mb" in d


# ---------------------------------------------------------------------------
# Adapt store: schema, keying, the bounded ring, parameterized SQL
# ---------------------------------------------------------------------------


class TestAdaptStore:
    def test_schema_tables_exist(self, tmp_path):
        store = rg.AdaptStore(tmp_path / "adapt.db")
        conn = rg._safe_connect(str(tmp_path / "adapt.db"))
        try:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        finally:
            conn.close()
        assert {"model_costs", "ceiling", "decisions"} <= names
        assert store.decision_count() == 0

    def test_name_digest_keying(self, tmp_path):
        store = rg.AdaptStore(tmp_path / "adapt.db")
        store.record_model_cost("m", "d1", 1 * GB, observed_at=100.0)
        store.record_model_cost("m", "d2", 2 * GB, observed_at=200.0)
        assert store.get_model_cost("m", "d1")["size_vram_bytes"] == 1 * GB
        assert store.get_model_cost("m", "d2")["size_vram_bytes"] == 2 * GB
        # No digest: the latest observation wins.
        assert store.get_model_cost("m")["size_vram_bytes"] == 2 * GB
        # Unknown digest falls back to the latest by name.
        assert store.get_model_cost("m", "ghost")["size_vram_bytes"] == 2 * GB

    def test_missing_digest_stored_as_empty(self, tmp_path):
        store = rg.AdaptStore(tmp_path / "adapt.db")
        store.record_model_cost("m", None, 1 * GB)
        learned = store.get_model_cost("m")
        assert learned["digest"] == ""

    def test_replace_supersedes_same_key(self, tmp_path):
        store = rg.AdaptStore(tmp_path / "adapt.db")
        store.record_model_cost("m", "d", 1 * GB, observed_at=100.0)
        store.record_model_cost("m", "d", 3 * GB, observed_at=200.0)
        assert store.get_model_cost("m", "d")["size_vram_bytes"] == 3 * GB

    def test_decisions_ring_pruned_by_count(self, tmp_path):
        store = rg.AdaptStore(tmp_path / "adapt.db")
        for i in range(8):
            store.record_decision(
                "chat", f"model-{i}", 8192, 8192, "admitted", ring_size=5
            )
        assert store.decision_count() == 5
        recent = store.recent_decisions(10)
        assert [r["model"] for r in recent] == [
            "model-7",
            "model-6",
            "model-5",
            "model-4",
            "model-3",
        ]

    def test_governor_record_decision_uses_config_ring(
        self, tmp_path, gov_factory
    ):
        p = _write_yaml(tmp_path, "decisions_ring_size: 3\n")
        gov = gov_factory(config_path=p)
        for i in range(6):
            gov.record_decision("chat", f"m{i}", None, None, "refused", "test")
        assert gov.store.decision_count() == 3

    def test_parameterized_sql_only(self):
        # The house rule: placeholders everywhere, no formatted SQL.
        assert "VALUES (?, ?, ?, ?, ?)" in SRC
        for forbidden in ('f"INSERT', "f'INSERT", 'f"SELECT', "f'SELECT",
                          'f"DELETE', "f'DELETE", '" % (', "' % ("):
            assert forbidden not in SRC


# ---------------------------------------------------------------------------
# Singleton and reset hook (DI-10)
# ---------------------------------------------------------------------------


class TestSingleton:
    @pytest.fixture(autouse=True)
    def _clean_singleton(self):
        rg.reset_resource_governor()
        yield
        rg.reset_resource_governor()

    def test_same_instance_returned(self, tmp_path):
        a = rg.get_resource_governor(
            config_path=tmp_path / "absent.yaml", db_path=tmp_path / "s.db"
        )
        b = rg.get_resource_governor()
        assert a is b

    def test_reset_drops_the_instance(self, tmp_path):
        a = rg.get_resource_governor(
            config_path=tmp_path / "absent.yaml", db_path=tmp_path / "s.db"
        )
        rg.reset_resource_governor()
        b = rg.get_resource_governor(
            config_path=tmp_path / "absent.yaml", db_path=tmp_path / "s2.db"
        )
        assert a is not b


# ---------------------------------------------------------------------------
# Shipped YAML (Section 10 contract surface)
# ---------------------------------------------------------------------------


class TestShippedYaml:
    def test_file_exists_and_parses(self):
        assert _YAML_PATH.is_file()
        raw = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)

    def test_section10_keys_present(self):
        raw = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
        for key in (
            "enabled",
            "total_vram_gb",
            "safety_margin_gb",
            "snapshot_ttl_s",
            "ctx_ladder",
            "ctx_floor",
            "idle_evict_threshold_s",
            "pressure",
            "pressure_keep_alive",
            "queue",
            "kv_coefficient",
            "rlimits",
            "ollama_limits",
            "ceiling_floor_gb",
            "decisions_ring_size",
        ):
            assert key in raw

    def test_total_vram_ships_null(self):
        raw = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
        assert raw["total_vram_gb"] is None

    def test_benchmark_and_agt_have_no_ctx_floor(self):
        # D3: benchmark/AGT never downsize, so they carry no floor entry.
        raw = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
        assert set(raw["ctx_floor"].keys()) == {"chat", "pipeline"}


# ---------------------------------------------------------------------------
# Doc pins (red-before on the pristine tree)
# ---------------------------------------------------------------------------


class TestAtrestRow:
    def test_row_present_in_matrix(self):
        content = _ATREST_PATH.read_text(encoding="utf-8")
        assert "| resource governor adapt store |" in content
        assert "data/resource_governor.db" in content
        assert "pending-scoping (added S223" in content
        assert "excluded (derived measurement state" in content

    def test_no_live_bk06_candidate_rows_reasserted(self):
        # Reassert the s220 invariant over the matrix INCLUDING the new row.
        content = _ATREST_PATH.read_text(encoding="utf-8")
        matrix_rows = [
            line
            for line in content.splitlines()
            if line.startswith("|") and "bk06-candidate" in line
        ]
        assert matrix_rows == []


class TestRoadmapRolled:
    def test_bloc0_landed_clause_present(self):
        content = _ROADMAP_PATH.read_text(encoding="utf-8")
        assert "Bloc 0 (measurement layer) landed at S223" in content

    def test_surviving_s221_pins_intact(self):
        content = _ROADMAP_PATH.read_text(encoding="utf-8")
        assert "spec WRITTEN at S221" in content
        assert "RESOURCE_GOVERNOR_SPEC.md is the design contract" in content
