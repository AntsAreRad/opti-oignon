"""S190 F3g -- web search: SR-03 (kill-switch gate on the live chat egress) and
CH-01 (stale chat_ui.py docstrings corrected).

executor.py and search_integration.py both pull heavy/optional imports (ollama,
ddgs) that are absent from the verification container, so these are source /
AST-level assertions per the audit's stated idiom for heavy import chains.
"""

import ast
from pathlib import Path

OO_DIR = Path(__file__).resolve().parent.parent / "opti_oignon"
EXECUTOR = OO_DIR / "executor.py"
SEARCH_INTEGRATION = OO_DIR / "search_integration.py"


def _read(p):
    return p.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# SR-03
# --------------------------------------------------------------------------

def test_sr03_executor_file_parses():
    ast.parse(_read(EXECUTOR))  # must remain valid Python after the edit


def test_sr03_web_search_block_consults_killswitch():
    src = _read(EXECUTOR)
    idx = src.find("if web_search:")
    assert idx != -1, "the web_search block must exist"
    # The kill-switch gate must live inside the web_search block, before the
    # web_search_engine.search() egress call.
    region = src[idx:idx + 2000]
    assert "search_killswitch" in region, "executor web_search block must import the kill switch"
    assert "is_killed()" in region, "executor web_search block must consult is_killed()"
    assert "kill switch engaged" in region, "a skip status must be emitted when killed"
    # The kill check must precede the actual egress call within the block.
    assert region.index("is_killed()") < region.index("web_search_engine.search(")


def test_sr03_egress_call_preserved():
    src = _read(EXECUTOR)
    # The surgical change must not have removed the actual search call.
    assert "web_search_engine.search(question, max_results=5)" in src


# --------------------------------------------------------------------------
# CH-01
# --------------------------------------------------------------------------

def test_ch01_search_integration_file_parses():
    ast.parse(_read(SEARCH_INTEGRATION))


def test_ch01_no_stale_chat_ui_references():
    src = _read(SEARCH_INTEGRATION)
    assert "chat_ui" not in src, "stale chat_ui.py references must be gone"
    assert "handle_chat_submit" not in src, "stale handle_chat_submit reference must be gone"


def test_ch01_no_french_residue():
    src = _read(SEARCH_INTEGRATION)
    for token in ("non disponible", "fonction", "utilisee", "Gardee", "dans la boucle"):
        assert token not in src, f"French residue {token!r} must be removed (HY-01)"


def test_ch01_points_at_real_integration_site():
    src = _read(SEARCH_INTEGRATION)
    # The corrected docs name the real path (executor / web_search_engine).
    assert "web_search_engine" in src
    assert "executor.py" in src
