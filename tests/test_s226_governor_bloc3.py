#!/usr/bin/env python3
"""S226 -- Resource Governor Bloc 3: limit management (R-03, Section 6).

Container-provable coverage, the spec Section 11 Bloc 3 list verbatim:

- env construction on the spawn path (build_ollama_spawn_env: null keys
  omitted, values stringified, spawn_applies honoured as the config
  switch it is; the helper IS the posture-(a) deliverable, the S226
  read having confirmed no in-app Ollama spawner exists);
- the advisory shapes (compute_ollama_limits_advisory: not_configured /
  match / mismatch / unknown, mixed resolution mismatch > unknown,
  non-coercible env values counted as mismatch, the honest "unknown"
  wording for the documented systemd case, env injectable and
  defaulting to os.environ) and the checklist line present and
  NON-BLOCKING (the S145 pattern pins: the new check never returns
  severity "critical", a forced mismatch never sets blocked, the
  external_advisory switch honoured, fail-open on a raising advisory);
- setrlimit applied / skipped per config and per availability asserted
  IN A CHILD PROCESS (the honest way to observe a process-wide limit),
  plus the once-per-process latch;
- the cgroup reference script: existence and the warning text only
  (host-bound, never executed by the application, never simulated).

Named supersession (the read gate's grep, the spec forecast's single
watch item): test_s145_code_signing_guards.py::TestStartupChecklist::
test_all_passed pinned the checklist at EXACTLY five items; deselected
in pyproject addopts and REASSERTED here at six with the original
semantics (TestChecklistReassert). The original is never edited.

Host-assured (named, never simulated): the cgroup helper end to end;
real OLLAMA_* limit behaviour of an external server.

Isolation: the established spec_from_file_location idiom with
sys.modules pre-seeding (ollama stub, opti_oignon package stub with a
real __path__), order-independent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

_BASE = Path(__file__).resolve().parent.parent
_OO = _BASE / "opti_oignon"
_MODULE_PATH = _OO / "resource_governor.py"
_SC_PATH = _OO / "startup_checks.py"
_IB_PATH = _OO / "inference_backend.py"
_ROUTES_SEC_PATH = _OO / "api" / "routes_security.py"
_YAML_PATH = _OO / "config" / "resource_governor.yaml"
_SCRIPT_PATH = _BASE / "scripts" / "ollama_cgroup_limits.sh"
_PYPROJECT_PATH = _BASE / "pyproject.toml"

SRC = _MODULE_PATH.read_text(encoding="utf-8")
SC_SRC = _SC_PATH.read_text(encoding="utf-8")
IB_SRC = _IB_PATH.read_text(encoding="utf-8")

GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Isolated module loading (the established idiom, mirrored from s223..s225)
# ---------------------------------------------------------------------------

sys.modules.setdefault("ollama", types.ModuleType("ollama"))

if "opti_oignon" not in sys.modules:
    _pkg = types.ModuleType("opti_oignon")
    _pkg.__path__ = [str(_OO)]
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


rg = _load_module(
    "opti_oignon.resource_governor", "opti_oignon/resource_governor.py"
)
sc = _load_module("opti_oignon.startup_checks", "opti_oignon/startup_checks.py")


def _cfg(**kw) -> "rg.GovernorConfig":
    return rg.GovernorConfig(**kw)


# ---------------------------------------------------------------------------
# A. Posture (a): spawn-path env construction
# ---------------------------------------------------------------------------


class TestSpawnEnv:
    def test_defaults_yield_empty_dict(self):
        assert rg.build_ollama_spawn_env(_cfg()) == {}

    def test_only_non_null_keys_emitted(self):
        cfg = _cfg(ollama_max_loaded_models=2, ollama_num_parallel=1)
        env = rg.build_ollama_spawn_env(cfg)
        assert env == {
            "OLLAMA_MAX_LOADED_MODELS": "2",
            "OLLAMA_NUM_PARALLEL": "1",
        }
        assert "OLLAMA_MAX_QUEUE" not in env

    def test_all_three_keys_stringified(self):
        cfg = _cfg(
            ollama_max_loaded_models=3,
            ollama_num_parallel=2,
            ollama_max_queue=128,
        )
        env = rg.build_ollama_spawn_env(cfg)
        assert env == {
            "OLLAMA_MAX_LOADED_MODELS": "3",
            "OLLAMA_NUM_PARALLEL": "2",
            "OLLAMA_MAX_QUEUE": "128",
        }
        assert all(isinstance(v, str) for v in env.values())

    def test_spawn_applies_false_yields_empty_dict(self):
        cfg = _cfg(ollama_max_loaded_models=2, ollama_spawn_applies=False)
        assert rg.build_ollama_spawn_env(cfg) == {}

    def test_non_coercible_value_skipped_never_raises(self):
        cfg = _cfg(ollama_num_parallel=1)
        cfg.ollama_max_loaded_models = "garbage"  # type: ignore[assignment]
        env = rg.build_ollama_spawn_env(cfg)
        assert env == {"OLLAMA_NUM_PARALLEL": "1"}

    def test_helper_documents_the_spawnless_finding(self):
        seg = SRC.split("def build_ollama_spawn_env", 1)[1].split("\ndef ", 1)[0]
        assert "no in-app Ollama spawner exists" in seg
        assert "standing list" in seg


# ---------------------------------------------------------------------------
# B. Posture (b): the advisory computation
# ---------------------------------------------------------------------------


class TestAdvisoryShapes:
    def test_not_configured(self):
        adv = rg.compute_ollama_limits_advisory(_cfg(), env={})
        assert adv["status"] == "not_configured"
        assert adv["mismatches"] == []
        assert adv["unknown_keys"] == []

    def test_unknown_when_configured_but_invisible(self):
        cfg = _cfg(ollama_max_loaded_models=2, ollama_num_parallel=1)
        adv = rg.compute_ollama_limits_advisory(cfg, env={})
        assert adv["status"] == "unknown"
        assert sorted(adv["unknown_keys"]) == [
            "max_loaded_models",
            "num_parallel",
        ]
        assert "not visible from this process" in adv["detail"]
        assert "config not enforced externally" in adv["detail"]

    def test_match(self):
        cfg = _cfg(ollama_max_loaded_models=2, ollama_max_queue=128)
        adv = rg.compute_ollama_limits_advisory(
            cfg,
            env={"OLLAMA_MAX_LOADED_MODELS": "2", "OLLAMA_MAX_QUEUE": "128"},
        )
        assert adv["status"] == "match"
        assert adv["mismatches"] == []
        assert adv["unknown_keys"] == []

    def test_mismatch_payload_shape(self):
        cfg = _cfg(ollama_max_loaded_models=2)
        adv = rg.compute_ollama_limits_advisory(
            cfg, env={"OLLAMA_MAX_LOADED_MODELS": "3"}
        )
        assert adv["status"] == "mismatch"
        assert adv["mismatches"] == [
            {
                "key": "max_loaded_models",
                "env_var": "OLLAMA_MAX_LOADED_MODELS",
                "configured": 2,
                "visible": "3",
            }
        ]
        assert "OLLAMA_MAX_LOADED_MODELS" in adv["detail"]

    def test_mixed_mismatch_beats_unknown(self):
        cfg = _cfg(ollama_max_loaded_models=2, ollama_num_parallel=1)
        adv = rg.compute_ollama_limits_advisory(
            cfg, env={"OLLAMA_MAX_LOADED_MODELS": "3"}
        )
        assert adv["status"] == "mismatch"
        assert adv["unknown_keys"] == ["num_parallel"]

    def test_mixed_unknown_beats_match(self):
        cfg = _cfg(ollama_max_loaded_models=2, ollama_num_parallel=1)
        adv = rg.compute_ollama_limits_advisory(
            cfg, env={"OLLAMA_MAX_LOADED_MODELS": "2"}
        )
        assert adv["status"] == "unknown"
        assert adv["unknown_keys"] == ["num_parallel"]

    def test_garbage_env_value_counts_as_mismatch(self):
        cfg = _cfg(ollama_num_parallel=1)
        adv = rg.compute_ollama_limits_advisory(
            cfg, env={"OLLAMA_NUM_PARALLEL": "many"}
        )
        assert adv["status"] == "mismatch"
        assert adv["mismatches"][0]["visible"] == "many"

    def test_whitespace_tolerated_in_env_value(self):
        cfg = _cfg(ollama_num_parallel=1)
        adv = rg.compute_ollama_limits_advisory(
            cfg, env={"OLLAMA_NUM_PARALLEL": " 1 "}
        )
        assert adv["status"] == "match"

    def test_config_switches_carried_in_payload(self):
        cfg = _cfg(ollama_spawn_applies=False, ollama_external_advisory=False)
        adv = rg.compute_ollama_limits_advisory(cfg, env={})
        assert adv["spawn_applies"] is False
        assert adv["external_advisory"] is False

    def test_env_defaults_to_os_environ(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MAX_QUEUE", "64")
        cfg = _cfg(ollama_max_queue=64)
        adv = rg.compute_ollama_limits_advisory(cfg)
        assert adv["status"] == "match"
        assert adv["visible"]["OLLAMA_MAX_QUEUE"] == "64"

    def test_visible_map_always_carries_the_three_vars(self):
        adv = rg.compute_ollama_limits_advisory(_cfg(), env={})
        assert set(adv["visible"]) == {
            "OLLAMA_MAX_LOADED_MODELS",
            "OLLAMA_NUM_PARALLEL",
            "OLLAMA_MAX_QUEUE",
        }

    def test_governor_method_delegates_with_its_config(self, tmp_path):
        yaml_text = (
            "ollama_limits:\n"
            "  max_loaded_models: 2\n"
        )
        p = tmp_path / "gov.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        gov = rg.ResourceGovernor(
            config_path=p,
            db_path=tmp_path / "gov.db",
            warmup=None,
            registry=None,
        )
        with patch.dict(os.environ, {"OLLAMA_MAX_LOADED_MODELS": "2"}):
            adv = gov.ollama_limits_advisory()
        assert adv["status"] == "match"
        assert adv["configured"]["max_loaded_models"] == 2


# ---------------------------------------------------------------------------
# C. The checklist integration (the S145 pattern pins)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_checklist():
    sc.clear_cache()
    yield
    sc.clear_cache()


def _force_advisory(monkeypatch, cfg, env_updates):
    """Route the checklist's advisory read through a known config + env."""
    monkeypatch.setattr(rg, "load_config", lambda *a, **kw: cfg)
    for var in (
        "OLLAMA_MAX_LOADED_MODELS",
        "OLLAMA_NUM_PARALLEL",
        "OLLAMA_MAX_QUEUE",
    ):
        monkeypatch.delenv(var, raising=False)
    for var, value in env_updates.items():
        monkeypatch.setenv(var, value)


class TestChecklistAdvisory:
    def test_check_function_exists_and_appended_sixth(self, fresh_checklist):
        result = sc.run_startup_checks(force=True)
        names = [c.name for c in result.checks]
        assert names[-1] == "governor_ollama_limits"
        assert len(result.checks) == 6

    def test_container_default_passes_info_zero_impact(self, monkeypatch):
        _force_advisory(monkeypatch, _cfg(), {})
        item = sc._check_governor_ollama_limits()
        assert item.passed is True
        assert item.severity == "info"
        assert item.score_impact == 0

    def test_mismatch_is_warning_with_actionable_tips(self, monkeypatch):
        _force_advisory(
            monkeypatch,
            _cfg(ollama_max_loaded_models=2),
            {"OLLAMA_MAX_LOADED_MODELS": "9"},
        )
        item = sc._check_governor_ollama_limits()
        assert item.passed is False
        assert item.severity == "warning"
        assert item.severity != "critical"
        assert item.score_impact == -3
        assert item.tips
        assert any("resource_governor.yaml" in t for t in item.tips)

    def test_unknown_is_passing_info_with_tips(self, monkeypatch):
        _force_advisory(monkeypatch, _cfg(ollama_num_parallel=1), {})
        item = sc._check_governor_ollama_limits()
        assert item.passed is True
        assert item.severity == "info"
        assert item.score_impact == 0
        assert item.tips
        assert "values unknown" in item.detail

    def test_external_advisory_switch_honoured(self, monkeypatch):
        _force_advisory(
            monkeypatch,
            _cfg(ollama_max_loaded_models=2, ollama_external_advisory=False),
            {"OLLAMA_MAX_LOADED_MODELS": "9"},
        )
        item = sc._check_governor_ollama_limits()
        assert item.passed is True
        assert item.severity == "info"
        assert "disabled by config" in item.detail

    def test_raising_advisory_fails_open(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("no config today")

        monkeypatch.setattr(rg, "load_config", boom)
        item = sc._check_governor_ollama_limits()
        assert item.passed is True
        assert item.severity == "info"
        assert "unavailable" in item.detail

    def test_forced_mismatch_never_blocks_startup(
        self, fresh_checklist, monkeypatch
    ):
        _force_advisory(
            monkeypatch,
            _cfg(ollama_max_loaded_models=2),
            {"OLLAMA_MAX_LOADED_MODELS": "9"},
        )
        ok = sc.CheckItem(
            name="ok", passed=True, severity="info", detail="ok"
        )
        with patch.object(sc, "_check_code_signing_scripts", return_value=ok), \
                patch.object(sc, "_check_ollama_bind", return_value=ok), \
                patch.object(sc, "_check_luks", return_value=ok), \
                patch.object(sc, "_check_security_mode", return_value=ok), \
                patch.object(sc, "_check_encrypted_swap", return_value=ok):
            result = sc.run_startup_checks(force=True)
        assert result.blocked is False
        assert result.all_passed is False
        gov_item = result.checks[-1]
        assert gov_item.name == "governor_ollama_limits"
        assert gov_item.severity == "warning"

    def test_check_source_never_emits_critical(self):
        seg = SC_SRC.split("def _check_governor_ollama_limits", 1)[1]
        seg = seg.split("\ndef ", 1)[0] if "\ndef " in seg else seg
        assert 'severity="critical"' not in seg
        assert "S145" in seg

    def test_six_appends_in_run_startup_checks(self):
        assert SC_SRC.count("result.checks.append(") == 6
        assert SC_SRC.count("def _check_governor_ollama_limits") == 1


class TestChecklistReassert:
    """Deselect-plus-reassert of the named S226 supersession.

    The original (test_s145_code_signing_guards.py::TestStartupChecklist::
    test_all_passed) pinned the checklist at EXACTLY five items with all
    five checks mocked passing; the S226 advisory check is the sixth.
    The original semantics are restated here at the new count; the
    original test is deselected in pyproject addopts and never edited.
    """

    def test_all_passed_at_six(self, fresh_checklist):
        ok = sc.CheckItem(
            name="ok", passed=True, severity="info", detail="ok"
        )
        with patch.object(sc, "_check_code_signing_scripts", return_value=ok), \
                patch.object(sc, "_check_ollama_bind", return_value=ok), \
                patch.object(sc, "_check_luks", return_value=ok), \
                patch.object(sc, "_check_security_mode", return_value=ok), \
                patch.object(sc, "_check_encrypted_swap", return_value=ok), \
                patch.object(
                    sc, "_check_governor_ollama_limits", return_value=ok
                ):
            result = sc.run_startup_checks(force=True)
        assert result.all_passed is True
        assert result.blocked is False
        assert len(result.checks) == 6

    def test_named_deselect_recorded_in_pyproject(self):
        text = _PYPROJECT_PATH.read_text(encoding="utf-8")
        assert (
            "--deselect=tests/test_s145_code_signing_guards.py::"
            "TestStartupChecklist::test_all_passed"
        ) in text


# ---------------------------------------------------------------------------
# D. The optional rlimits (child-process assertions)
# ---------------------------------------------------------------------------

_CHILD_PRELUDE = r"""
import json, sys, types
sys.modules.setdefault("ollama", types.ModuleType("ollama"))
import importlib.util
from pathlib import Path
base = Path({base!r})
pkg = types.ModuleType("opti_oignon")
pkg.__path__ = [str(base / "opti_oignon")]
sys.modules.setdefault("opti_oignon", pkg)
spec = importlib.util.spec_from_file_location(
    "opti_oignon.resource_governor",
    str(base / "opti_oignon" / "resource_governor.py"),
)
rg = importlib.util.module_from_spec(spec)
sys.modules["opti_oignon.resource_governor"] = rg
spec.loader.exec_module(rg)
"""


def _run_child(body: str) -> dict:
    code = _CHILD_PRELUDE.format(base=str(_BASE)) + body
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_BASE),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])
    raise AssertionError(f"no RESULT line in child stdout: {proc.stdout!r}")


class TestRlimitsChildProcess:
    def test_applied_observable_via_getrlimit(self):
        out = _run_child(
            """
import resource
cfg = rg.GovernorConfig(
    rlimits_enabled=True, rlimits_as_gb=1.0, rlimits_data_gb=2.0
)
outcome = rg.apply_llamacpp_rlimits(cfg)
a_soft, a_hard = resource.getrlimit(resource.RLIMIT_AS)
d_soft, d_hard = resource.getrlimit(resource.RLIMIT_DATA)
print("RESULT:" + json.dumps({
    "outcome": outcome, "as_soft": a_soft, "data_soft": d_soft,
}))
"""
        )
        assert out["outcome"]["applied"] is True
        assert out["as_soft"] == GB
        assert out["data_soft"] == 2 * GB
        assert out["outcome"]["as_bytes"] == GB
        assert out["outcome"]["data_bytes"] == 2 * GB

    def test_disabled_skips_and_leaves_limits_untouched(self):
        out = _run_child(
            """
import resource
before = resource.getrlimit(resource.RLIMIT_AS)
outcome = rg.apply_llamacpp_rlimits(rg.GovernorConfig())
after = resource.getrlimit(resource.RLIMIT_AS)
print("RESULT:" + json.dumps({
    "outcome": outcome, "unchanged": before == after,
}))
"""
        )
        assert out["outcome"]["applied"] is False
        assert out["outcome"]["reason"] == "disabled"
        assert out["unchanged"] is True

    def test_resource_module_unavailable_fails_open(self):
        out = _run_child(
            """
sys.modules["resource"] = None
cfg = rg.GovernorConfig(rlimits_enabled=True, rlimits_as_gb=1.0)
outcome = rg.apply_llamacpp_rlimits(cfg)
print("RESULT:" + json.dumps({"outcome": outcome}))
"""
        )
        assert out["outcome"]["applied"] is False
        assert "resource module unavailable" in out["outcome"]["reason"]

    def test_raising_setrlimit_fails_open(self):
        out = _run_child(
            """
import resource as real_resource
stub = types.ModuleType("resource")
stub.RLIMIT_AS = real_resource.RLIMIT_AS
stub.RLIMIT_DATA = real_resource.RLIMIT_DATA
stub.RLIM_INFINITY = real_resource.RLIM_INFINITY
stub.getrlimit = real_resource.getrlimit
def _refuse(*a, **kw):
    raise PermissionError("denied by stub")
stub.setrlimit = _refuse
sys.modules["resource"] = stub
cfg = rg.GovernorConfig(
    rlimits_enabled=True, rlimits_as_gb=1.0, rlimits_data_gb=1.0
)
outcome = rg.apply_llamacpp_rlimits(cfg)
print("RESULT:" + json.dumps({"outcome": outcome}))
"""
        )
        assert out["outcome"]["applied"] is False
        assert "RLIMIT_AS" in out["outcome"]["reason"]
        assert "RLIMIT_DATA" in out["outcome"]["reason"]

    def test_once_per_process_latch(self):
        out = _run_child(
            """
import resource
cfg1 = rg.GovernorConfig(rlimits_enabled=True, rlimits_as_gb=2.0)
first = rg.apply_llamacpp_rlimits(cfg1)
cfg2 = rg.GovernorConfig(rlimits_enabled=True, rlimits_as_gb=1.0)
second = rg.apply_llamacpp_rlimits(cfg2)
a_soft, _ = resource.getrlimit(resource.RLIMIT_AS)
print("RESULT:" + json.dumps({
    "first": first, "second": second,
    "same": first is second, "as_soft": a_soft,
}))
"""
        )
        assert out["same"] is True
        assert out["as_soft"] == 2 * GB
        assert out["second"]["as_bytes"] == 2 * GB

    def test_null_values_with_enabled_true_apply_nothing(self):
        out = _run_child(
            """
cfg = rg.GovernorConfig(rlimits_enabled=True)
outcome = rg.apply_llamacpp_rlimits(cfg)
print("RESULT:" + json.dumps({"outcome": outcome}))
"""
        )
        assert out["outcome"]["applied"] is False
        assert out["outcome"]["reason"] == "no limit values configured"


class TestRlimitsInParent:
    def test_outcome_shape_and_latch_without_touching_limits(
        self, monkeypatch
    ):
        monkeypatch.setattr(rg, "_RLIMITS_OUTCOME", None)
        outcome = rg.apply_llamacpp_rlimits(rg.GovernorConfig())
        assert set(outcome) == {"applied", "reason", "as_bytes", "data_bytes"}
        again = rg.apply_llamacpp_rlimits(
            rg.GovernorConfig(rlimits_enabled=True, rlimits_as_gb=1.0)
        )
        assert again is outcome

    def test_docstring_states_the_process_wide_caveat(self):
        doc = rg.apply_llamacpp_rlimits.__doc__ or ""
        assert "PROCESS-WIDE" in doc
        assert "ENTIRE Opti-Oignon" in doc
        assert "optional and off by default" in doc


# ---------------------------------------------------------------------------
# E. The llama.cpp load-seam hook (source pins, red-before provable)
# ---------------------------------------------------------------------------


def _method_segment(src: str, cls_name: str, method: str) -> str:
    cls_m = re.search(
        rf"^class {cls_name}\b.*?(?=^class |\Z)", src, re.S | re.M
    )
    assert cls_m is not None, cls_name
    m = re.search(
        rf"^    def {method}\b.*?(?=^    def |\Z)", cls_m.group(0), re.S | re.M
    )
    assert m is not None, method
    return m.group(0)


class TestLoadSeamHook:
    def test_hook_sits_in_get_or_load_before_the_constructor(self):
        seg = _method_segment(IB_SRC, "LlamaCppBackend", "_get_or_load")
        assert "apply_llamacpp_rlimits" in seg
        assert seg.index("apply_llamacpp_rlimits") < seg.index(
            "_LlamaCpp(**kwargs)"
        )

    def test_hook_is_fail_open_and_marked(self):
        seg = _method_segment(IB_SRC, "LlamaCppBackend", "_get_or_load")
        assert "S226 (R-03)" in seg
        assert "except Exception:" in seg
        assert IB_SRC.count("apply_llamacpp_rlimits") == 2

    def test_s188_segment_invariants_survive_the_hook(self):
        seg = _method_segment(IB_SRC, "LlamaCppBackend", "_get_or_load")
        assert "with self._lock_for(self._load_locks, model_name):" in seg
        assert seg.count("self._loaded_models.get(model_name)") >= 2
        assert "_locks_guard" not in seg


# ---------------------------------------------------------------------------
# F. The cgroup reference script (existence + warnings only)
# ---------------------------------------------------------------------------


class TestCgroupScript:
    def test_exists_and_executable(self):
        assert _SCRIPT_PATH.exists()
        assert _SCRIPT_PATH.stat().st_mode & 0o111

    def test_bash_header_conventions(self):
        text = _SCRIPT_PATH.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        assert "set -euo pipefail" in text

    def test_host_bound_warnings_present(self):
        text = _SCRIPT_PATH.read_text(encoding="utf-8")
        assert "HOST-BOUND, reference only" in text
        assert "never executed by the application" in text
        assert "never simulated in tests" in text

    def test_recipes_named_print_only(self):
        text = _SCRIPT_PATH.read_text(encoding="utf-8")
        assert "systemd-run --scope" in text
        assert "MemoryMax" in text
        assert "Environment=OLLAMA_MAX_LOADED_MODELS" in text
        assert "Nothing was applied by this script" in text


# ---------------------------------------------------------------------------
# G. Doc, config and module-convention pins
# ---------------------------------------------------------------------------


class TestDocAndConfigPins:
    def test_yaml_carries_the_process_wide_caveat(self):
        text = _YAML_PATH.read_text(encoding="utf-8")
        assert "setrlimit is process-wide" in text
        assert "ENTIRE Opti-Oignon process" in text

    def test_shipped_yaml_still_mirrors_defaults(self):
        assert rg.load_config(_YAML_PATH) == rg.GovernorConfig()

    def test_routes_security_docstring_names_the_sixth_check(self):
        text = _ROUTES_SEC_PATH.read_text(encoding="utf-8")
        assert "Resource governor Ollama limits advisory (R-03, S226)" in text

    def test_startup_checks_docstring_names_the_advisory(self):
        assert "Resource governor Ollama limits advisory (R-03, S226)" in SC_SRC

    def test_module_docstring_owns_bloc3(self):
        assert "Blocs 0-3" in SRC
        assert "Bloc 3 (S226)" in SRC


class TestModuleConventions:
    def test_bloc3_surface_defined_once(self):
        for name in (
            "def build_ollama_spawn_env",
            "def compute_ollama_limits_advisory",
            "def apply_llamacpp_rlimits",
            "def ollama_limits_advisory",
            "def _check_governor_ollama_limits",
        ):
            src = SC_SRC if "check_governor" in name else SRC
            assert src.count(name) == 1, name

    def test_bloc2_surface_still_defined_once(self):
        for name in (
            "def pressure_state",
            "def admit_or_wait",
            "def evict_model",
            "def _honour_conditional_eviction",
            "def _evictable_candidates",
            "def _notify_queue",
        ):
            assert SRC.count(name) == 1, name
        assert SRC.count("_honour_conditional_eviction(") == 3

    def test_pure_ascii_and_sentinels_hold(self):
        SRC.encode("ascii")
        assert rg.checkpoint_before_apply is True
        assert rg.FEATURE_AVAILABLE is True
        assert sc.STARTUP_CHECKS_AVAILABLE is True

    def test_no_new_invalidate_hook_names(self):
        for hook in (
            "def invalidate_on_load",
            "def invalidate_on_evict",
            "def invalidate_on_estop_drain",
            "def invalidate_on_resume",
        ):
            assert SRC.count(hook) == 1, hook
