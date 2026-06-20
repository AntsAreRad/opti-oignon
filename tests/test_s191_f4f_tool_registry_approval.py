"""S191 F4f -- tool registry & approval: TCA-02 (the approval-UI risk
classifier's tool-name sets were stale and mislabelled almost every real tool
as "low"). The fix aligns the high/medium sets with the actual tool names.

`tool_call_approval.py` is stdlib-only at module scope (the signed-audit import
is lazy), so it loads in isolation with the parent package stubbed; `assess_risk`
is a pure module function.
"""

import importlib.util
import sys
import types
from pathlib import Path

OO_DIR = Path(__file__).resolve().parent.parent / "opti_oignon"
APPROVAL = OO_DIR / "tool_call_approval.py"


def _load_approval():
    if "opti_oignon" not in sys.modules:
        sys.modules["opti_oignon"] = types.ModuleType("opti_oignon")
    spec = importlib.util.spec_from_file_location("opti_oignon.tool_call_approval", APPROVAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tca02_real_tool_names_are_not_mislabelled_low():
    m = _load_approval()
    # Execution / write / state-mutation tools -> high.
    for tool in ("bash", "execute_code", "create_file", "write_file",
                 "str_replace", "manage_memory", "manage_skills", "web_search"):
        assert m.assess_risk(tool) == "high", tool
    # Read / inspect tools -> medium.
    for tool in ("view", "read_file", "list_files"):
        assert m.assess_risk(tool) == "medium", tool
    # Unknown tools still default to low.
    assert m.assess_risk("totally_unknown_tool") == "low"


def test_tca02_case_insensitive():
    m = _load_approval()
    assert m.assess_risk("BASH") == "high"
    assert m.assess_risk("View") == "medium"


def test_tca02_prefix_sets_mislabelled_bash_low():
    # Pin the bug: the pre-fix high/medium sets did not contain the real tool
    # names, so the classifier returned "low" for bash / create_file / etc.
    prefix_high = {
        "web_search", "web_fetch", "http_request",
        "file_write", "file_delete", "shell_exec",
        "code_execute", "sandbox_exec",
    }
    prefix_medium = {
        "file_read", "file_list", "database_query",
        "rag_search", "memory_write",
    }
    for tool in ("bash", "create_file", "str_replace", "manage_memory", "execute_code"):
        assert tool not in prefix_high and tool not in prefix_medium  # -> "low" pre-fix
    m = _load_approval()
    assert m.assess_risk("bash") == "high"  # fixed
