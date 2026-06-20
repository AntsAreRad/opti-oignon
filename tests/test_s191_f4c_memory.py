"""S191 F4c -- two-tier memory: MEM-06 (legacy ollama chat response parsed
dict-only -> object-form ChatResponse raised a swallowed TypeError, yielding an
empty extraction). The fix adds a dict-or-object `_reply_text` helper and uses
it at both legacy extraction sites.

`memory/legacy.py` imports config / db_utils / ollama at module scope, so it is
not importable in isolation; these are source / AST assertions per the audit's
heavy-import-chain idiom. The pure `_reply_text` helper is extracted from the
source and exercised directly (it has no dependencies), and the pre-fix failure
mode is pinned (subscripting an object-form response raises TypeError).
"""

import ast
import typing
from pathlib import Path

OO_DIR = Path(__file__).resolve().parent.parent / "opti_oignon"
LEGACY = OO_DIR / "memory" / "legacy.py"


def _read(p):
    return p.read_text(encoding="utf-8")


def _extract_function_source(src, name):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            assert seg is not None
            return seg
    raise AssertionError(f"function {name} not found")


def _load_reply_text():
    seg = _extract_function_source(_read(LEGACY), "_reply_text")
    ns = {"Any": typing.Any}
    exec(compile(seg, "legacy._reply_text", "exec"), ns)
    return ns["_reply_text"]


# --------------------------------------------------------------------------
# Source / AST: the file is valid and the dict-only parse is gone
# --------------------------------------------------------------------------

def test_mem06_legacy_file_parses():
    ast.parse(_read(LEGACY))


def test_mem06_no_dict_only_response_parse():
    src = _read(LEGACY)
    assert 'response["message"]["content"]' not in src, (
        "the dict-only ollama response parse must be replaced"
    )


def test_mem06_reply_text_used_at_both_sites():
    src = _read(LEGACY)
    assert src.count("_reply_text(response)") >= 2, (
        "both extraction sites must use the dict-or-object accessor"
    )
    assert "def _reply_text(" in src


# --------------------------------------------------------------------------
# Behavioural: the helper handles both forms; the pre-fix code did not
# --------------------------------------------------------------------------

class _Msg:
    def __init__(self, content):
        self.content = content


class _ChatResponse:
    """Mimics newer ollama-python: an object, NOT subscriptable."""

    def __init__(self, content):
        self.message = _Msg(content)


def test_mem06_prefix_parse_raises_on_object_form():
    # Pin the bug: the pre-fix expression raises TypeError on an object form,
    # which the surrounding try/except swallowed into an empty extraction.
    resp = _ChatResponse("name is Leon")
    assert not isinstance(resp, dict)
    raised = False
    try:
        _ = resp["message"]["content"]  # the pre-fix access
    except TypeError:
        raised = True
    assert raised, "object-form response must not be subscriptable"


def test_mem06_reply_text_handles_all_forms():
    reply_text = _load_reply_text()
    assert reply_text(_ChatResponse("name is Leon")) == "name is Leon"  # object
    assert reply_text({"message": {"content": "a fact"}}) == "a fact"  # dict
    assert reply_text("already text") == "already text"  # str
    assert reply_text(None) == ""  # None
    assert reply_text({"message": None}) == ""  # malformed dict
    assert reply_text(_ChatResponse(None)) == ""  # object, empty content
