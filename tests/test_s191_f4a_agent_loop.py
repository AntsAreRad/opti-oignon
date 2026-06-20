"""S191 F4a -- Odysseus agent loop.

Two behavioural fixes, each pinned against a pre-fix copy of its module (the
test fails on the pre-fix code and passes on the fix):

- AGL-01: ``agent/loop._extract_verdict`` misread "incorrect" / "wrong" as a
  pass, because ``"correct" in low`` matched the substring inside "incorrect".
  The fix matches failure tokens first and uses word boundaries.
- ACF-01: ``agent/config_loader.load_config_data`` claimed a deep copy but
  shared nested dicts with the module-global ``CONFIG_DEFAULTS``; a caller
  mutating a nested value corrupted the process-wide defaults.

Both modules sit under ``opti_oignon.agent``; importing them normally would
trigger the heavy ``opti_oignon`` import chain (ollama et al.). They are loaded
in isolation via ``spec_from_file_location`` with the parent packages stubbed in
``sys.modules`` (the register-before-exec idiom), which is valid here because
the functions under test are pure and reference their guarded imports only
lazily (and annotations are strings under ``from __future__ import annotations``).
"""

import importlib.util
import sys
import types
from pathlib import Path

OO_DIR = Path(__file__).resolve().parent.parent / "opti_oignon"
LOOP = OO_DIR / "agent" / "loop.py"
CONFIG_LOADER = OO_DIR / "agent" / "config_loader.py"


def _stub_parents():
    """Register empty stubs for the agent package and the loop's deps."""
    if "opti_oignon" not in sys.modules:
        sys.modules["opti_oignon"] = types.ModuleType("opti_oignon")
    if "opti_oignon.agent" not in sys.modules:
        sys.modules["opti_oignon.agent"] = types.ModuleType("opti_oignon.agent")
    agent_pkg = sys.modules["opti_oignon.agent"]
    for sub in ("dispatch", "untrusted_context"):
        full = f"opti_oignon.agent.{sub}"
        if full not in sys.modules:
            sys.modules[full] = types.ModuleType(full)
        setattr(agent_pkg, sub, sys.modules[full])


def _load(module_name, source_path):
    """Load a module from explicit source, registered before exec."""
    _stub_parents()
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_from_text(module_name, text):
    """Load a module from an in-memory source string (for the pre-fix copy)."""
    _stub_parents()
    mod = types.ModuleType(module_name)
    mod.__file__ = str(CONFIG_LOADER)  # so DEFAULT_CONFIG_PATH resolves sanely
    sys.modules[module_name] = mod
    exec(compile(text, module_name, "exec"), mod.__dict__)
    return mod


# --------------------------------------------------------------------------
# AGL-01 -- verdict extraction
# --------------------------------------------------------------------------

def test_agl01_incorrect_reads_as_fail():
    loop = _load("opti_oignon.agent.loop", LOOP)
    assert loop._extract_verdict("The result is incorrect.") == "fail"
    assert loop._extract_verdict("This is plainly wrong") == "fail"


def test_agl01_pass_tokens_still_pass():
    loop = _load("opti_oignon.agent.loop", LOOP)
    assert loop._extract_verdict("PASS") == "pass"
    assert loop._extract_verdict("This is correct.") == "pass"
    assert loop._extract_verdict("looks good to me") == "pass"


def test_agl01_fail_keyword_and_unknown():
    loop = _load("opti_oignon.agent.loop", LOOP)
    assert loop._extract_verdict("FAIL: a step is missing") == "fail"
    assert loop._extract_verdict("no clear verdict here") == "unknown"
    assert loop._extract_verdict("") == "unknown"


def test_agl01_prefix_pattern_is_used_not_substring():
    # The fix replaced the substring scan with compiled word-boundary patterns.
    loop = _load("opti_oignon.agent.loop", LOOP)
    assert hasattr(loop, "_VERDICT_FAIL_RE")
    assert hasattr(loop, "_VERDICT_PASS_RE")


def test_agl01_prefix_code_misclassifies_incorrect():
    # Pin the bug: the original substring logic returns "pass" for "incorrect".
    def _prefix_extract(content):
        low = (content or "").lower()
        if "fail" in low:
            return "fail"
        if "pass" in low or "correct" in low or "looks good" in low:
            return "pass"
        return "unknown"

    assert _prefix_extract("The result is incorrect.") == "pass"  # the bug
    loop = _load("opti_oignon.agent.loop", LOOP)
    assert loop._extract_verdict("The result is incorrect.") == "fail"  # fixed


# --------------------------------------------------------------------------
# ACF-01 -- config deep copy
# --------------------------------------------------------------------------

def test_acf01_mutating_returned_config_does_not_touch_defaults():
    cl = _load("opti_oignon.agent.config_loader", CONFIG_LOADER)
    baseline = cl.CONFIG_DEFAULTS["loop"]["round_cap"]
    data = cl.load_config_data("/nonexistent/agent/config/xyz.yaml")  # -> defaults
    data["loop"]["round_cap"] = 99999
    data["teacher"]["teacher_model"] = "mutated"
    assert cl.CONFIG_DEFAULTS["loop"]["round_cap"] == baseline
    assert cl.CONFIG_DEFAULTS["teacher"]["teacher_model"] != "mutated"


def test_acf01_prefix_code_corrupts_global_defaults():
    # Pin the bug: with the pre-fix shallow merge, mutating a nested value in
    # the returned dict corrupts the module-global CONFIG_DEFAULTS.
    fixed_src = CONFIG_LOADER.read_text(encoding="utf-8")
    prefix_src = fixed_src.replace(
        "defaults = copy.deepcopy(CONFIG_DEFAULTS)  # a private deep copy, not shared",
        "defaults = _deep_merge(CONFIG_DEFAULTS, {})  # deep copy of defaults",
    )
    assert prefix_src != fixed_src, "the pre-fix line must be present to revert"

    prefix = _load_from_text("opti_oignon.agent._config_loader_prefix", prefix_src)
    data = prefix.load_config_data("/nonexistent/agent/config/xyz.yaml")
    data["loop"]["round_cap"] = 12345
    assert prefix.CONFIG_DEFAULTS["loop"]["round_cap"] == 12345  # the bug

    fixed = _load("opti_oignon.agent.config_loader", CONFIG_LOADER)
    data2 = fixed.load_config_data("/nonexistent/agent/config/xyz.yaml")
    data2["loop"]["round_cap"] = 12345
    assert fixed.CONFIG_DEFAULTS["loop"]["round_cap"] != 12345  # fixed
