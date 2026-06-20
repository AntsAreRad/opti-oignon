"""S185 audit fix SR-02 -- remove dead search_and_augment; fix the gate.

routes_chat's coding-agent callback imported and called
search_integration.search_and_augment, a symbol defined nowhere (and it imported
SEARCH_AVAILABLE, which is not exported), so the import always raised ImportError
and the whole web-search-augmentation block was dead. It is removed.

executor.py's live web_search gate imported SearchInterceptor/wrap_system_prompt
only to set SEARCH_AVAILABLE but never used them (the path injects results
directly via web_search_engine); the gate now imports the real dependency. The
<search>-tag SearchInterceptor state machine is implemented and unit-tested
(test_live_v130) but not wired into a streaming loop; it is retained with a
docstring note rather than deleted.

Supersedes tests/test_chat_coding_agent_s118.py::TestRoutesSourceCode::
test_web_search_in_rich_callback (which asserted the now-removed symbol is
present). That test is deselected per the project discipline (never edited); the
re-assertion below records the corrected reality. These are source-content
assertions; there is no runtime behaviour change (the removed block never ran).
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The dead symbol is gone from the codebase
# ---------------------------------------------------------------------------

def test_search_and_augment_removed_from_routes_chat():
    # Re-assertion superseding test_web_search_in_rich_callback: the dead
    # web-search-augmentation symbol no longer appears in routes_chat at all.
    assert "search_and_augment" not in _src("opti_oignon/api/routes_chat.py")


def test_search_and_augment_absent_repo_wide():
    # The symbol is defined nowhere and called nowhere across the package.
    for rel in (
        "opti_oignon/api/routes_chat.py",
        "opti_oignon/search_integration.py",
        "opti_oignon/executor.py",
        "opti_oignon/chat_coding_agent.py",
    ):
        assert "search_and_augment" not in _src(rel)


def test_coding_agent_callback_keeps_live_parts():
    # The removal was surgical: the live vision and plugin-hook stages remain.
    src = _src("opti_oignon/api/routes_chat.py")
    assert "vision_pipeline" in src
    assert "pre_inference" in src
    assert "post_inference" in src


# ---------------------------------------------------------------------------
# executor web_search gate is on the real dependency, not the unused import
# ---------------------------------------------------------------------------

def test_executor_gate_uses_web_search_engine():
    src = _src("opti_oignon/executor.py")
    # The misleading interceptor import in the gate is gone.
    assert "from opti_oignon.search_integration import SearchInterceptor" not in src
    # The web_search block gates on and uses the real dependency.
    block = src.split("Step 2d: Web search injection", 1)[1].split("Step 3:", 1)[0]
    assert "from opti_oignon.web_search import web_search_engine" in block
    assert "SEARCH_AVAILABLE = True" in block
    assert "web_search_engine.search(" in block


# ---------------------------------------------------------------------------
# The inert interceptor is documented as unwired (recorded decision)
# ---------------------------------------------------------------------------

def test_search_integration_documents_unwired_interceptor():
    src = _src("opti_oignon/search_integration.py").lower()
    assert "sr-02" in src
    assert "not currently wired" in src or "not wired" in src
    assert "wire or remove" in src
