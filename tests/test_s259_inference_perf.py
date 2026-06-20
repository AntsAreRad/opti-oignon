#!/usr/bin/env python3
"""S259 -- Inference performance lot: per-model KV admission, the
llama-server seam, and native think plumbing.

Container-provable surfaces pinned here, red-before proven on the
pristine S258 tree (every pristine red an AssertionError through the
assert-before-call guard idiom; the declared design-green families pass
on pristine):

A. Governor ``kv_overrides`` (resolution model > family > global; the
   global default 0.5 preserved fail-secure for unknown models; the
   loader parses the nested mapping conservatively; ``estimate_kv_cache_gb``
   gains an optional ``model`` kwarg, byte-compatible when absent; the
   admit path consumes the per-model coefficient -- proven functionally
   over the s224 fake-snapshot harness with a 262144-token request).
B. ``build_llama_server_command`` in speculative_decoding: the pure flag
   materialisation of SpeculativeConfig (the S110 module finally wired
   to something real), draft and self-draft (MTP) postures, KV-quant and
   flash-attention flags, validation rejections.
C. ``LlamaServerBackend`` in inference_backend: the external-process
   seam (name, degraded health/list/info/generate against an unreachable
   host, direct registry registration, the config-section source pin).
D. ``LlamaCppBackend`` knobs: flash_attn / type_k / type_v accepted and
   threaded from the llama_cpp config section.
E. Executor native think: the pure ``_native_think_kwargs`` helper and
   its call-site merge (source pins plus an ast-extracted behavioural
   check); the existing agentic think threading reasserted.
F. Reassertions (design-green): the s223 default coefficient, the
   loader override case, the default ladder, the estimate math, the
   SpeculativeConfig defaults, the LlamaCppBackend n_ctx default, the
   shipped yaml's stable global coefficient.
G. Docs: the spec roll and the INFERENCE_PERF_S259 runbook.
H. Structure: AST and ASCII over the touched sources (design-green on
   pristine by construction).

Host-assured (named, never simulated here): launching a real
llama-server with the built command, real draft acceptance rates, real
MTP self-drafting, real KV-quant memory savings, the real think switch
against a live reasoning model.

Isolation: the established spec_from_file_location idiom with
sys.modules pre-seeding (an ollama stub and an opti_oignon package stub
carrying a real __path__), so the module chain resolves by path without
executing opti_oignon/__init__.py.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import yaml

_BASE = Path(__file__).resolve().parent.parent
_RG_PATH = _BASE / "opti_oignon" / "resource_governor.py"
_IB_PATH = _BASE / "opti_oignon" / "inference_backend.py"
_SD_PATH = _BASE / "opti_oignon" / "speculative_decoding.py"
_EX_PATH = _BASE / "opti_oignon" / "executor.py"
_AX_PATH = _BASE / "opti_oignon" / "agentic_executor.py"
_YAML_PATH = _BASE / "opti_oignon" / "config" / "resource_governor.yaml"
_SPEC_PATH = _BASE / "RESOURCE_GOVERNOR_SPEC.md"
_RUNBOOK_PATH = _BASE / "INFERENCE_PERF_S259.md"

RG_SRC = _RG_PATH.read_text(encoding="utf-8")
IB_SRC = _IB_PATH.read_text(encoding="utf-8")
SD_SRC = _SD_PATH.read_text(encoding="utf-8")
EX_SRC = _EX_PATH.read_text(encoding="utf-8")
AX_SRC = _AX_PATH.read_text(encoding="utf-8")

GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Guard idiom (assert-before-call): a missing surface is an AssertionError
# on the pristine tree, never an AttributeError or a collection error.
# ---------------------------------------------------------------------------


def _guard_attr(mod, name: str):
    assert hasattr(mod, name), f"missing attribute: {name}"
    return getattr(mod, name)


def _guard_param(func, name: str):
    params = inspect.signature(func).parameters
    assert name in params, f"missing parameter: {name}"
    return params[name]


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
sd = sys.modules["opti_oignon.speculative_decoding"]


# ---------------------------------------------------------------------------
# Fakes and fixtures (the s224 harness shapes, reused verbatim in spirit)
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class CountingWarmup:
    """Injectable S1 fake with the keep_alive attribute the evictable
    computation derives idle time from."""

    keep_alive = "30m"

    def __init__(self, models=None):
        self.calls = 0
        self.models = list(models or [])

    def get_loaded_models(self):
        self.calls += 1
        return list(self.models)


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


def _fake_cm(window=262144, max_output=4096):
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


@pytest.fixture()
def seams(monkeypatch):
    """Deterministic collaborator seams: a not-stopped estop and a
    262144-window ModelLimits so a long-context request is NOT clamped
    before the walk (the 4.2 clamp is out of scope here)."""
    estop = FakeEstop((False,))
    monkeypatch.setitem(sys.modules, "opti_oignon.emergency_stop", estop)
    monkeypatch.setitem(sys.modules, "opti_oignon.context_manager", _fake_cm())
    return estop


@pytest.fixture()
def gov_factory(tmp_path, seams):
    """ResourceGovernor factory: capacity 10 GB, margin 1, tmp DB,
    deterministic meminfo, empty warmup (no resident weights, no
    evictable), so the admission cost reduces to the KV term for an
    unknown model -- the cleanest lens on the coefficient resolution."""

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
        kw.setdefault("warmup", CountingWarmup([]))
        kw.setdefault("registry", None)
        kw.setdefault("clock", FakeClock())
        kw.setdefault("meminfo_path", _meminfo_file(tmp_path))
        return rg.ResourceGovernor(**kw)

    return make


_OVERRIDE_YAML = (
    "total_vram_gb: 10\n"
    "safety_margin_gb: 1.0\n"
    "kv_coefficient: 0.5\n"
    "idle_evict_threshold_s: 600\n"
    "kv_overrides:\n"
    "  models:\n"
    "    longctx-exact: 0.002\n"
    "  families:\n"
    "    longctx: 0.004\n"
    "    longctx-wide: 0.008\n"
)


# ---------------------------------------------------------------------------
# A. Governor kv_overrides
# ---------------------------------------------------------------------------


class TestKvOverridesConfig:
    def test_config_has_override_fields(self):
        cfg_cls = _guard_attr(rg, "GovernorConfig")
        cfg = cfg_cls()
        assert hasattr(cfg, "kv_override_models")
        assert hasattr(cfg, "kv_override_families")
        assert cfg.kv_override_models == {}
        assert cfg.kv_override_families == {}

    def test_loader_parses_nested_mapping(self, tmp_path):
        load = _guard_attr(rg, "load_config")
        cfg = load(_write_yaml(tmp_path, _OVERRIDE_YAML))
        assert hasattr(cfg, "kv_override_models")
        assert cfg.kv_override_models.get("longctx-exact") == 0.002
        assert cfg.kv_override_families.get("longctx") == 0.004
        assert cfg.kv_override_families.get("longctx-wide") == 0.008
        assert cfg.kv_coefficient == 0.5

    def test_loader_coerces_values_to_float(self, tmp_path):
        load = _guard_attr(rg, "load_config")
        cfg = load(
            _write_yaml(
                tmp_path,
                "kv_overrides:\n  models:\n    m1: 1\n  families:\n    f1: '0.25'\n",
            )
        )
        assert hasattr(cfg, "kv_override_models")
        assert cfg.kv_override_models.get("m1") == 1.0
        assert cfg.kv_override_families.get("f1") == 0.25

    def test_loader_ignores_non_mapping_conservatively(self, tmp_path):
        load = _guard_attr(rg, "load_config")
        cfg = load(_write_yaml(tmp_path, "kv_overrides: 7\n"))
        assert hasattr(cfg, "kv_override_models")
        assert cfg.kv_override_models == {}
        assert cfg.kv_override_families == {}
        cfg2 = load(
            _write_yaml(
                tmp_path,
                "kv_overrides:\n  models: 3\n  families: [a, b]\n",
                name="gov2.yaml",
            )
        )
        assert cfg2.kv_override_models == {}
        assert cfg2.kv_override_families == {}

    def test_loader_drops_unparseable_entries(self, tmp_path):
        load = _guard_attr(rg, "load_config")
        cfg = load(
            _write_yaml(
                tmp_path,
                "kv_overrides:\n  models:\n    good: 0.1\n    bad: not-a-number\n",
            )
        )
        assert hasattr(cfg, "kv_override_models")
        assert cfg.kv_override_models.get("good") == 0.1
        assert "bad" not in cfg.kv_override_models


class TestKvCoefficientResolution:
    def test_resolver_exists_on_governor(self, gov_factory):
        gov = gov_factory()
        _guard_attr(gov, "resolve_kv_coefficient")

    def test_unknown_and_none_fall_back_to_global(self, gov_factory):
        gov = gov_factory(yaml_text=_OVERRIDE_YAML)
        resolve = _guard_attr(gov, "resolve_kv_coefficient")
        assert resolve(None) == 0.5
        assert resolve("totally-unrelated") == 0.5

    def test_exact_model_beats_family(self, gov_factory):
        gov = gov_factory(yaml_text=_OVERRIDE_YAML)
        resolve = _guard_attr(gov, "resolve_kv_coefficient")
        assert resolve("longctx-exact") == 0.002

    def test_family_substring_matches(self, gov_factory):
        gov = gov_factory(yaml_text=_OVERRIDE_YAML)
        resolve = _guard_attr(gov, "resolve_kv_coefficient")
        assert resolve("longctx-test:latest") == 0.004

    def test_family_match_is_case_insensitive(self, gov_factory):
        gov = gov_factory(yaml_text=_OVERRIDE_YAML)
        resolve = _guard_attr(gov, "resolve_kv_coefficient")
        assert resolve("LongCTX-Test") == 0.004

    def test_longest_family_key_wins(self, gov_factory):
        gov = gov_factory(yaml_text=_OVERRIDE_YAML)
        resolve = _guard_attr(gov, "resolve_kv_coefficient")
        assert resolve("longctx-wide-9b") == 0.008


class TestKvEstimateModelKwarg:
    def test_estimate_accepts_optional_model(self, gov_factory):
        gov = gov_factory(yaml_text=_OVERRIDE_YAML)
        _guard_param(gov.estimate_kv_cache_gb, "model")
        assert gov.estimate_kv_cache_gb(262144, model="longctx-test") == (
            pytest.approx(256.0 * 0.004)
        )

    def test_estimate_without_model_is_byte_compatible(self, gov_factory):
        gov = gov_factory(yaml_text=_OVERRIDE_YAML)
        _guard_param(gov.estimate_kv_cache_gb, "model")
        assert gov.estimate_kv_cache_gb(4096) == pytest.approx(2.0)
        assert gov.estimate_kv_cache_gb(0) == 0.0
        assert gov.estimate_kv_cache_gb(None) == 0.0


class TestAdmitUsesOverrides:
    def test_admit_source_consumes_resolver(self):
        assert RG_SRC.count("resolve_kv_coefficient") >= 2, (
            "the admit cost path must consume the resolver"
        )

    def test_long_context_admitted_under_override(self, gov_factory):
        gov = gov_factory(yaml_text=_OVERRIDE_YAML)
        _guard_attr(gov, "resolve_kv_coefficient")
        decision = gov.admit("longctx-test", 262144, caller="chat")
        assert decision.admitted is True
        assert decision.num_ctx == 262144
        assert decision.action == "admit"

    def test_long_context_ladders_without_override(self, gov_factory):
        # Design-green control: the pristine behaviour, preserved.
        gov = gov_factory()
        decision = gov.admit("longctx-test", 262144, caller="chat")
        assert decision.admitted is True
        assert decision.num_ctx == 16384
        assert "ctx_laddered_to_fit" in decision.reason

    def test_override_does_not_leak_to_other_models(self, gov_factory):
        gov = gov_factory(yaml_text=_OVERRIDE_YAML)
        _guard_attr(gov, "resolve_kv_coefficient")
        decision = gov.admit("unrelated-model", 262144, caller="chat")
        assert decision.num_ctx == 16384
        assert "ctx_laddered_to_fit" in decision.reason


# ---------------------------------------------------------------------------
# B. build_llama_server_command (speculative_decoding finally wired)
# ---------------------------------------------------------------------------


def _spec_cfg(**kw):
    cfg_cls = sd.SpeculativeConfig
    defaults = dict(
        enabled=True,
        draft_model="/models/draft.gguf",
        draft_max=24,
        draft_min=8,
        draft_gpu_layers=99,
        auto_select_draft=False,
    )
    defaults.update(kw)
    return cfg_cls(**defaults)


def _pair(cmd: list, flag: str) -> Optional[str]:
    assert flag in cmd, f"missing flag: {flag}"
    i = cmd.index(flag)
    assert i + 1 < len(cmd), f"flag {flag} has no value"
    return cmd[i + 1]


class TestLlamaServerCommandBuilder:
    def test_builder_exists(self):
        _guard_attr(sd, "build_llama_server_command")

    def test_happy_path_draft_flags(self):
        build = _guard_attr(sd, "build_llama_server_command")
        cmd = build(
            "/models/main.gguf",
            _spec_cfg(),
            host="127.0.0.1",
            port=8089,
            n_ctx=262144,
            flash_attn=True,
            type_k="q8_0",
            type_v="q8_0",
        )
        assert isinstance(cmd, list)
        assert cmd[0] == "llama-server"
        assert all(isinstance(part, str) for part in cmd)
        assert _pair(cmd, "-m") == "/models/main.gguf"
        assert _pair(cmd, "-md") == "/models/draft.gguf"
        assert _pair(cmd, "--draft-max") == "24"
        assert _pair(cmd, "--draft-min") == "8"
        assert _pair(cmd, "-ngld") == "99"
        assert _pair(cmd, "--host") == "127.0.0.1"
        assert _pair(cmd, "--port") == "8089"
        assert _pair(cmd, "-c") == "262144"
        assert _pair(cmd, "--cache-type-k") == "q8_0"
        assert _pair(cmd, "--cache-type-v") == "q8_0"
        assert "--flash-attn" in cmd

    def test_disabled_config_emits_no_draft_flags(self):
        build = _guard_attr(sd, "build_llama_server_command")
        cmd = build("/models/main.gguf", _spec_cfg(enabled=False))
        assert "-md" not in cmd
        assert "--draft-max" not in cmd
        assert "--draft-min" not in cmd
        assert "-ngld" not in cmd
        assert _pair(cmd, "-m") == "/models/main.gguf"

    def test_self_draft_posture_no_external_draft(self):
        # MTP models self-draft: an enabled config with no draft model
        # emits the base server command without -md (the draft lives in
        # the model file; host-assured verification per the runbook).
        build = _guard_attr(sd, "build_llama_server_command")
        cmd = build("/models/mtp.gguf", _spec_cfg(draft_model=""))
        assert "-md" not in cmd
        assert _pair(cmd, "-m") == "/models/mtp.gguf"

    def test_optional_flags_absent_by_default(self):
        build = _guard_attr(sd, "build_llama_server_command")
        cmd = build("/models/main.gguf", _spec_cfg())
        assert "-c" not in cmd
        assert "--flash-attn" not in cmd
        assert "--cache-type-k" not in cmd
        assert "--cache-type-v" not in cmd

    def test_invalid_config_rejected(self):
        build = _guard_attr(sd, "build_llama_server_command")
        with pytest.raises(ValueError):
            build("/models/main.gguf", _spec_cfg(draft_min=50, draft_max=10))

    def test_empty_model_path_rejected(self):
        build = _guard_attr(sd, "build_llama_server_command")
        with pytest.raises(ValueError):
            build("", _spec_cfg())

    def test_builder_is_pure_no_side_effects(self, tmp_path):
        build = _guard_attr(sd, "build_llama_server_command")
        before = sorted(p.name for p in tmp_path.iterdir())
        cmd1 = build("/models/main.gguf", _spec_cfg())
        cmd2 = build("/models/main.gguf", _spec_cfg())
        assert cmd1 == cmd2
        assert sorted(p.name for p in tmp_path.iterdir()) == before


# ---------------------------------------------------------------------------
# C. LlamaServerBackend (the external-process seam)
# ---------------------------------------------------------------------------


class TestLlamaServerBackend:
    def _backend(self):
        cls = _guard_attr(ib, "LlamaServerBackend")
        return cls(host="http://127.0.0.1:9", timeout_s=0.2)

    def test_class_exists_and_names(self):
        backend = self._backend()
        assert backend.name == "llama_server"
        assert isinstance(backend.display_name, str)
        assert backend.display_name

    def test_health_check_false_fast_when_unreachable(self):
        backend = self._backend()
        start = time.monotonic()
        assert backend.health_check() is False
        assert time.monotonic() - start < 2.0

    def test_list_models_degrades_to_empty(self):
        backend = self._backend()
        assert backend.list_models() == []

    def test_model_info_degrades_to_none(self):
        backend = self._backend()
        assert backend.model_info("anything") is None

    def test_generate_raises_when_unreachable(self):
        backend = self._backend()
        with pytest.raises(RuntimeError):
            backend.generate(model="m", prompt="hello")

    def test_registers_in_a_fresh_registry(self):
        backend = self._backend()
        registry = ib.BackendRegistry()
        registry.register(backend)
        assert registry.get("llama_server") is backend

    def test_config_loader_reads_llama_server_section(self):
        assert 'cfg.get("llama_server"' in IB_SRC, (
            "init_backends_from_config must read the llama_server section"
        )


# ---------------------------------------------------------------------------
# D. LlamaCppBackend knobs
# ---------------------------------------------------------------------------


class TestLlamaCppKnobs:
    def test_init_accepts_flash_attn(self):
        cls = _guard_attr(ib, "LlamaCppBackend")
        _guard_param(cls.__init__, "flash_attn")

    def test_init_accepts_kv_cache_types(self):
        cls = _guard_attr(ib, "LlamaCppBackend")
        _guard_param(cls.__init__, "type_k")
        _guard_param(cls.__init__, "type_v")

    def test_knob_defaults_change_nothing(self):
        cls = _guard_attr(ib, "LlamaCppBackend")
        params = inspect.signature(cls.__init__).parameters
        assert "flash_attn" in params and params["flash_attn"].default in (
            None,
            False,
        )
        assert "type_k" in params and params["type_k"].default is None
        assert "type_v" in params and params["type_v"].default is None

    def test_config_loader_threads_the_knobs(self):
        assert 'llama_cfg.get("flash_attn"' in IB_SRC
        assert '"type_k"' in IB_SRC
        assert '"type_v"' in IB_SRC


# ---------------------------------------------------------------------------
# E. Executor native think plumbing
# ---------------------------------------------------------------------------


def _extract_function(src: str, name: str):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(mod, "<extracted>", "exec"), ns)  # noqa: S102
            return ns[name]
    return None


class TestNativeThinkHelper:
    def test_helper_defined_once(self):
        assert EX_SRC.count("def _native_think_kwargs(") == 1

    def test_helper_consumed_at_a_call_site(self):
        assert EX_SRC.count("_native_think_kwargs") >= 2, (
            "the helper must be merged into at least one client call"
        )

    def test_helper_behaviour(self):
        fn = _extract_function(EX_SRC, "_native_think_kwargs")
        assert fn is not None, "missing helper: _native_think_kwargs"
        assert fn(None) == {}
        assert fn(True) == {"think": True}
        assert fn(False) == {"think": False}

    def test_agentic_think_threading_reasserted(self):
        # Design-green: the S70-era threading the helper rides on.
        assert "think=think" in AX_SRC


# ---------------------------------------------------------------------------
# F. Reassertions (design-green: the pristine contracts preserved)
# ---------------------------------------------------------------------------


class TestReassertions:
    def test_default_global_coefficient_is_half(self):
        assert rg.GovernorConfig().kv_coefficient == 0.5

    def test_loader_override_case_still_parses(self, tmp_path):
        cfg = rg.load_config(
            _write_yaml(tmp_path, "kv_coefficient: 0.25\n")
        )
        assert cfg.kv_coefficient == 0.25

    def test_default_ladder_unchanged(self):
        assert rg.GovernorConfig().ctx_ladder == [32768, 16384, 8192, 4096]

    def test_estimate_math_unchanged_under_default(self, gov_factory):
        gov = gov_factory()
        assert gov.estimate_kv_cache_gb(4096) == pytest.approx(2.0)

    def test_speculative_config_defaults(self):
        cfg = sd.SpeculativeConfig()
        assert cfg.draft_max == 16
        assert cfg.draft_min == 5
        assert cfg.enabled is False

    def test_llamacpp_default_ctx_param(self):
        params = inspect.signature(ib.LlamaCppBackend.__init__).parameters
        assert params["n_ctx"].default == 4096

    def test_shipped_yaml_global_coefficient_stable(self):
        data = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
        assert data.get("kv_coefficient") == 0.5


# ---------------------------------------------------------------------------
# G. Docs
# ---------------------------------------------------------------------------


class TestDocs:
    def test_spec_documents_kv_overrides(self):
        spec = _SPEC_PATH.read_text(encoding="utf-8")
        assert "kv_overrides" in spec

    def test_shipped_yaml_carries_kv_overrides_key(self):
        data = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
        assert "kv_overrides" in data
        assert isinstance(data["kv_overrides"], dict)
        assert "models" in data["kv_overrides"]
        assert "families" in data["kv_overrides"]

    def test_runbook_exists(self):
        assert _RUNBOOK_PATH.exists(), "INFERENCE_PERF_S259.md must exist"

    def test_runbook_required_sections(self):
        assert _RUNBOOK_PATH.exists(), "INFERENCE_PERF_S259.md must exist"
        text = _RUNBOOK_PATH.read_text(encoding="utf-8")
        assert "llama-server" in text
        assert "--draft-max" in text
        assert "MTP" in text
        assert "never simulated in-container" in text


# ---------------------------------------------------------------------------
# H. Structure (AST and ASCII over the touched sources)
# ---------------------------------------------------------------------------

_TOUCHED = [
    _RG_PATH,
    _IB_PATH,
    _SD_PATH,
    _EX_PATH,
    _AX_PATH,
]


class TestStructure:
    @pytest.mark.parametrize("path", _TOUCHED, ids=lambda p: p.name)
    def test_sources_parse(self, path):
        ast.parse(path.read_text(encoding="utf-8"))

    @pytest.mark.parametrize("path", _TOUCHED, ids=lambda p: p.name)
    def test_sources_ascii(self, path):
        raw = path.read_text(encoding="utf-8")
        assert all(ord(ch) < 128 for ch in raw), f"non-ASCII in {path.name}"

    def test_this_suite_avoids_the_selection_literal(self):
        here = Path(__file__).read_text(encoding="utf-8")
        token = "sandbox" + "_manager"
        assert token not in here
