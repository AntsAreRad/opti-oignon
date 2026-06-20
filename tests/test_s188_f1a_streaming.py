#!/usr/bin/env python3
"""S188 phase F1-A: chat streaming fixes (SSE-04, SSE-05).

Source-level assertions. ``routes_chat.py`` pulls a heavy import chain
(ollama, executor, agentic executor, plugin hooks, ...), so the streaming
loop is verified against the module source rather than by importing it.
The function bodies are isolated with ``ast`` so the sibling
``_stream_chat_coding`` loop cannot satisfy the ``_stream_response``
assertions by accident.

SSE-05: the steady-state poll in ``_stream_response`` must not block the
asyncio event loop on a ``threading.Event``; it must ``await asyncio.sleep``
like the sibling coding-agent loop already does.

SSE-04: a successful keepalive ping proves the client socket is still
draining, so the idle-timeout consumer timer must be refreshed on ping;
otherwise a slow producer (long tool/search/think phase with no emitted
events) is mistaken for a slow client and a legitimate long generation is
cancelled.
"""

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
ROUTES_CHAT = REPO / "opti_oignon" / "api" / "routes_chat.py"
SRC = ROUTES_CHAT.read_text(encoding="utf-8")
# Parsing here also asserts the edited file is syntactically valid.
TREE = ast.parse(SRC)


def _segment(name: str) -> str:
    for node in ast.walk(TREE):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            seg = ast.get_source_segment(SRC, node)
            assert seg is not None, f"could not extract source for {name}"
            return seg
    raise AssertionError(f"function {name} not found in routes_chat.py")


def test_asyncio_imported_at_module_level():
    # SSE-05 prerequisite: await asyncio.sleep in _stream_response needs asyncio
    # in scope at module level, not only via the local import that
    # _stream_chat_coding carries.
    top_imports = {
        alias.name
        for node in TREE.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "asyncio" in top_imports


def test_sse05_stream_response_poll_does_not_block_event_loop():
    seg = _segment("_stream_response")
    # the steady-state poll must yield to the loop, not block it on a threading.Event
    assert "generation_done.wait(timeout=0.05)" not in seg
    assert "await asyncio.sleep(0.05)" in seg


def test_sse05_terminal_waits_left_unchanged():
    # scope guard: only the steady-state poll was changed; the terminal-path
    # joins (client gone / idle / error) remain threading.Event waits and are
    # recorded as a lower-priority note, not changed in this fix.
    seg = _segment("_stream_response")
    assert "generation_done.wait(timeout=5.0)" in seg


def test_sse04_keepalive_refreshes_consumer_timer():
    seg = _segment("_stream_response")
    # init + on-send + keepalive == 3 refreshes; before the fix it was 2.
    assert seg.count("_bp_last_consumer_time = time.time()") >= 3
    # and the refresh sits in the keepalive success path, right after the ping send.
    ping_at = seg.index('send_json({"type": "ping"')
    after_ping = seg[ping_at:ping_at + 600]
    assert "_bp_last_consumer_time = time.time()" in after_ping


def test_sibling_coding_loop_unchanged():
    seg = _segment("_stream_chat_coding")
    # the sibling already polled with a non-blocking sleep before S188
    assert "await asyncio.sleep(0.05)" in seg
    # and it has no backpressure consumer timer to refresh
    assert "_bp_last_consumer_time" not in seg
