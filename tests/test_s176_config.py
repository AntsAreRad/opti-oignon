#!/usr/bin/env python3
"""Tests for S176 -- the agent configuration loader (Theme 3 / Odysseus Core).

Covers ODYSSEUS_SPEC.md Section 5 configuration surface:

- The defaults match the S175 reference values (round cap == loop.MAX_AGENT_
  ROUNDS, verifier cap == loop._VERIFIER_MAX_ROUNDS) and the teacher reference
  thresholds; the per-mode tool exposure matches the registry.
- The laptop-lite preset is represented and resolves to a compact profile.
- The loader is guarded: a missing file, absent PyYAML, a malformed file, or a
  non-dict document all degrade to the defaults and never raise.
- Bounds: the round and verifier caps are clamped to the loop's own limits.
- Cartography: ``config.yaml`` and ``config_loader.py`` are registered in
  ODYSSEUS_SPEC.md.

Loaded in isolation via ``spec_from_file_location`` with ``opti_oignon``
stubbed, so the runtime collects without the backend.
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"
SPEC = ROOT / "ODYSSEUS_SPEC.md"
CONFIG_YAML = AGENT / "config.yaml"


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
al = _ensure_agent("allowlists")
_ensure_agent("untrusted_context")
_ensure_agent("dispatch")
loop = _ensure_agent("loop")
te = _ensure_agent("teacher")
tools = _ensure_agent("tools")
cfg = _ensure_agent("config_loader")


@pytest.fixture(autouse=True)
def _reset():
    cfg.reset_agent_config()
    tools.reset_tool_registry()
    yield
    cfg.reset_agent_config()
    tools.reset_tool_registry()


def _write_yaml(text: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    return f.name


# Module conventions


class TestModuleConventions:
    def test_sentinels(self):
        assert cfg.checkpoint_before_apply is True
        assert cfg.FEATURE_AVAILABLE is True

    def test_default_path_points_at_config_yaml(self):
        assert cfg.DEFAULT_CONFIG_PATH.name == "config.yaml"
        assert cfg.DEFAULT_CONFIG_PATH == CONFIG_YAML

    def test_singleton(self):
        a = cfg.get_agent_config()
        b = cfg.get_agent_config()
        assert a is b

    def test_reset(self):
        a = cfg.get_agent_config()
        cfg.reset_agent_config()
        assert cfg.get_agent_config() is not a

    def test_set_config(self):
        c = cfg.AgentConfig(round_cap=7)
        cfg.set_agent_config(c)
        assert cfg.get_agent_config() is c

    def test_config_defaults_constant(self):
        assert "loop" in cfg.CONFIG_DEFAULTS
        assert "teacher" in cfg.CONFIG_DEFAULTS
        assert "tools" in cfg.CONFIG_DEFAULTS


# config.yaml file


class TestConfigFile:
    def test_file_exists(self):
        assert CONFIG_YAML.exists()

    def test_file_is_valid_yaml_with_sections(self):
        import yaml

        data = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        for key in ("loop", "teacher", "tools", "presets"):
            assert key in data

    def test_no_emojis_in_config(self):
        text = CONFIG_YAML.read_text(encoding="utf-8")
        for ch in text:
            assert ord(ch) < 0x1F000, f"emoji-range char present: {ch!r}"


# Defaults match the S175 reference


class TestDefaultsMatchReference:
    def test_round_cap_matches_loop(self):
        assert cfg.load_config().round_cap == loop.MAX_AGENT_ROUNDS

    def test_verifier_cap_matches_loop(self):
        assert cfg.load_config().verifier_cap == loop._VERIFIER_MAX_ROUNDS

    def test_verify_default_off(self):
        assert cfg.load_config().verify is False

    def test_teacher_thresholds_match_reference(self):
        c = cfg.load_config()
        assert c.teacher.get("failure_threshold") == te.DEFAULT_FAILURE_THRESHOLD
        assert c.teacher.get("teacher_model") == te.DEFAULT_TEACHER_MODEL
        assert c.teacher.get("student_model") == te.DEFAULT_STUDENT_MODEL

    def test_constant_defaults_match_loop(self):
        # The hardcoded defaults themselves track the loop reference.
        assert cfg.CONFIG_DEFAULTS["loop"]["round_cap"] == loop.MAX_AGENT_ROUNDS
        assert cfg.CONFIG_DEFAULTS["loop"]["verifier_cap"] == loop._VERIFIER_MAX_ROUNDS


# Per-mode tool exposure matches the registry / allowlists


class TestPerModeTools:
    def test_daily_matches_registry(self):
        c = cfg.load_config()
        assert set(c.daily_tools) == set(tools.build_tool_set("daily").names)

    def test_bulbe_matches_registry(self):
        c = cfg.load_config()
        assert set(c.bulbe_tools) == set(tools.build_tool_set("bulbe").names)

    def test_daily_subset_of_allowlist(self):
        c = cfg.load_config()
        for name in c.daily_tools:
            assert name in al.DAILY_ALLOWLIST

    def test_bulbe_equals_sandbox_set(self):
        c = cfg.load_config()
        assert set(c.bulbe_tools) == set(al.SANDBOX_TOOL_NAMES)

    def test_network_and_state_tools_daily_only(self):
        c = cfg.load_config()
        assert "web_search" in c.daily_tools
        assert "manage_memory" in c.daily_tools
        assert "web_search" not in c.bulbe_tools
        assert "manage_memory" not in c.bulbe_tools


# Presets


class TestPresets:
    def test_laptop_lite_available(self):
        assert cfg.LAPTOP_LITE in cfg.available_presets()

    def test_laptop_lite_is_compact(self):
        lite = cfg.load_config(preset=cfg.LAPTOP_LITE)
        base = cfg.load_config()
        assert lite.round_cap < base.round_cap
        assert lite.verifier_cap <= base.verifier_cap
        assert lite.preset == cfg.LAPTOP_LITE

    def test_laptop_lite_lighter_teacher(self):
        lite = cfg.load_config(preset=cfg.LAPTOP_LITE)
        assert lite.teacher.get("teacher_model") != te.DEFAULT_TEACHER_MODEL

    def test_unknown_preset_falls_back(self):
        c = cfg.load_config(preset="does_not_exist")
        # Falls back to base values without raising.
        assert c.round_cap == loop.MAX_AGENT_ROUNDS

    def test_no_preset_is_base(self):
        c = cfg.load_config()
        assert c.preset is None


# Guards


class TestGuards:
    def test_missing_file_uses_defaults(self):
        c = cfg.load_config(path="/nonexistent/agent-config.yaml")
        assert c.round_cap == loop.MAX_AGENT_ROUNDS
        assert c.verifier_cap == loop._VERIFIER_MAX_ROUNDS

    def test_malformed_yaml_uses_defaults(self):
        path = _write_yaml("loop: [unclosed\n  : :::")
        try:
            c = cfg.load_config(path=path)
            assert c.round_cap == loop.MAX_AGENT_ROUNDS
        finally:
            os.unlink(path)

    def test_non_dict_document_uses_defaults(self):
        path = _write_yaml("- just\n- a\n- list\n")
        try:
            c = cfg.load_config(path=path)
            assert c.round_cap == loop.MAX_AGENT_ROUNDS
        finally:
            os.unlink(path)

    def test_load_config_data_never_raises(self):
        # Even a directory path returns the defaults rather than raising.
        data = cfg.load_config_data(path=str(AGENT))
        assert isinstance(data, dict)
        assert "loop" in data

    def test_yaml_unavailable_uses_defaults(self, monkeypatch):
        monkeypatch.setattr(cfg, "_load_yaml", lambda: None)
        c = cfg.load_config()
        assert c.round_cap == loop.MAX_AGENT_ROUNDS


# Clamping


class TestClamping:
    def test_round_cap_clamped_high(self):
        path = _write_yaml("loop:\n  round_cap: 999999\n")
        try:
            assert cfg.load_config(path=path).round_cap == 1000
        finally:
            os.unlink(path)

    def test_round_cap_clamped_low(self):
        path = _write_yaml("loop:\n  round_cap: 0\n")
        try:
            assert cfg.load_config(path=path).round_cap == 1
        finally:
            os.unlink(path)

    def test_verifier_cap_clamped(self):
        path = _write_yaml("loop:\n  verifier_cap: 9\n")
        try:
            assert cfg.load_config(path=path).verifier_cap == 2
        finally:
            os.unlink(path)

    def test_bad_types_fall_back(self):
        path = _write_yaml("loop:\n  round_cap: not-a-number\n")
        try:
            assert cfg.load_config(path=path).round_cap == loop.MAX_AGENT_ROUNDS
        finally:
            os.unlink(path)


# Teacher policy bridge


class TestTeacherPolicy:
    def test_teacher_policy_builds_policy(self):
        c = cfg.load_config()
        pol = c.teacher_policy()
        assert isinstance(pol, te.EscalationPolicy)
        assert pol.failure_threshold == te.DEFAULT_FAILURE_THRESHOLD
        assert pol.teacher_model == te.DEFAULT_TEACHER_MODEL

    def test_laptop_lite_policy_lighter(self):
        c = cfg.load_config(preset=cfg.LAPTOP_LITE)
        pol = c.teacher_policy()
        assert pol.teacher_model != te.DEFAULT_TEACHER_MODEL


# Cartography


class TestCartography:
    def test_config_yaml_registered(self):
        text = SPEC.read_text(encoding="utf-8")
        assert "opti_oignon/agent/config.yaml" in text

    def test_config_loader_registered(self):
        text = SPEC.read_text(encoding="utf-8")
        assert "opti_oignon/agent/config_loader.py" in text

    def test_files_on_disk(self):
        assert CONFIG_YAML.exists()
        assert (AGENT / "config_loader.py").exists()
