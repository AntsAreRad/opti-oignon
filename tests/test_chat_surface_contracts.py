#!/usr/bin/env python3
"""Contracts for the chat surface: what the store and the client promise each other.

A reply streams through one client (``lib/api/chat.ts``, which hands the
browser's socket to ``lib/chat/stream.ts``) and one store
(``lib/stores/chat.ts``). The client declares its callbacks in
``ChatStreamCallbacks`` (``lib/types.ts``). A callback declared and never
called, or called and never provided, is a frame the interface silently
drops.

The names the contracts read:

  * ``ChatStreamCallbacks`` -- the declared callbacks, one per key of the
    interface.
  * The client: ``lib/api/chat.ts`` and the ``lib/chat`` modules it imports.
    Its ``streamChat`` and ``retryChat`` both open the stream through
    ``openStream``.
  * The store: ``sendMessage`` and ``retryLastMessage``, each passing a
    literal ``callbacks: ChatStreamCallbacks`` object to the client;
    ``cancelCurrentGeneration``, the Stop, which keeps the partial reply.
  * The chat route (``routes/(app)/(use)/chat/[id]/+page.svelte``): its
    thread, the element named "Chat messages", and the components it mounts
    there; its composer, the component it mounts with ``on:send``.

They read sources as text; they prove a form, never a browser.

Local-only (the public distribution ships no tests).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_navigation_contracts as _nav  # noqa: E402
from _frontend import REPO  # noqa: E402

# Seconds each contract may take on this machine, read back by the ladder
# from the junit file.
BUDGET_S = {
    "test_cm6_every_declared_callback_has_a_caller_and_a_provider[caller]": 1.0,
    "test_cm6_every_declared_callback_has_a_caller_and_a_provider[provider]": 1.0,
    "test_cm5_the_chat_textarea_stays_usable_while_a_reply_streams": 1.0,
    "test_cm7_a_stopped_reply_keeps_its_text_and_says_it_stopped": 1.0,
    "test_cm14_one_status_region_speaks_for_the_stream[content]": 1.0,
    "test_cm14_one_status_region_speaks_for_the_stream[status]": 1.0,
}

_SRC = "frontend/src"
_LIB = f"{_SRC}/lib"
_TYPES = f"{_LIB}/types.ts"
_API = f"{_LIB}/api/chat.ts"
_STORE = f"{_LIB}/stores/chat.ts"
_CHAT = f"{_LIB}/components/chat"
_ROUTE = f"{_SRC}/routes/(app)/(use)/chat/[id]/+page.svelte"

# The paths the store sends through, and the client call each one makes.
_PATHS = {"sendMessage": "streamChat", "retryLastMessage": "retryChat"}


def _balanced(text, start):
    """The index just past the bracket that closes the one at ``start``."""
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack, at = [], start
    quote = None
    while at < len(text):
        char = text[at]
        if quote:
            if char == "\\":
                at += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"`":
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return at + 1
        at += 1
    raise AssertionError(f"unbalanced bracket at {start}")


def _interface_keys(script, name):
    """The member names of ``interface name { ... }``."""
    match = re.search(rf"\binterface\s+{re.escape(name)}\s*\{{", script)
    assert match, f"interface {name} is declared"
    body = script[match.end():_balanced(script, match.end() - 1) - 1]
    keys, depth = [], 0
    for line in body.splitlines():
        if depth == 0:
            found = re.match(r"\s*(\w+)\??\s*[:(]", line)
            if found:
                keys.append(found.group(1))
        depth += line.count("{") + line.count("(") - line.count("}") - line.count(")")
    return keys


def _object_keys(literal):
    """The top-level keys of an object literal (``{`` ... ``}``)."""
    assert literal.startswith("{"), literal[:40]
    keys, at, depth = [], 1, 0
    expect = True
    quote = None
    while at < len(literal) - 1:
        char = literal[at]
        if quote:
            if char == "\\":
                at += 2
                continue
            if char == quote:
                quote = None
            at += 1
            continue
        if char in "'\"`":
            quote = char
        elif char in "{([":
            depth += 1
        elif char in "})]":
            depth -= 1
        elif depth == 0 and char == ",":
            expect = True
        elif depth == 0 and expect and not char.isspace():
            found = re.match(r"(?:async\s+)?(\w+)\s*(?=[:(,}]|$)", literal[at:])
            if found:
                keys.append(found.group(1))
                at += found.end()
                expect = False
                continue
            expect = False
        at += 1
    return keys


def _function_body(script, name):
    match = re.search(rf"\bfunction\s+{re.escape(name)}\s*\(", script)
    assert match, f"{name} is defined"
    close = _balanced(script, match.end() - 1)
    start = script.index("{", close)
    return script[start:_balanced(script, start)]


def _declared():
    keys = _interface_keys(_nav._script(_TYPES), "ChatStreamCallbacks")
    assert {"onToken", "onDone", "onError"} <= set(keys), f"the declared callbacks are read: {keys}"
    return keys


def _client_files():
    """The client: the API module and every lib/chat module it imports."""
    script = _nav._script(_API)
    found = [_API]
    for match in _nav._IMPORT.finditer(script):
        path = _nav._resolve(_API, match.group(2))
        if path and path.startswith(f"{_LIB}/chat/") and path not in found:
            found.append(path)
    return found


@pytest.mark.parametrize("half", ("caller", "provider"))
def test_cm6_every_declared_callback_has_a_caller_and_a_provider(half):
    declared = _declared()

    if half == "caller":
        sample = "callbacks.onToken(data.content); callbacks.onThinking?.(data.content);"
        assert all(re.search(rf"\bcallbacks\.{n}\s*(?:\?\.)?\s*\(", sample) for n in ("onToken", "onThinking")), (
            "the probe reads a plain and an optional call"
        )
        client = _client_files()
        code = "\n".join(_nav._code(path) for path in client)
        uncalled = [name for name in declared
                    if not re.search(rf"\bcallbacks\.{name}\s*(?:\?\.)?\s*\(", code)]
        assert not uncalled, f"every declared callback is called by the client ({client}): {uncalled}"
        api = _nav._script(_API)
        for path in _PATHS.values():
            assert "openStream" in _function_body(api, path), (
                f"{path} reads its stream through openStream, where every callback is called"
            )
        return

    sample = "{ onToken: (c) => { a({ x: 1 }); }, onDone, async onError(e) { f(e); }, onFrame: feed }"
    assert _object_keys(sample) == ["onToken", "onDone", "onError", "onFrame"], (
        f"the probe reads top-level keys only: {_object_keys(sample)}"
    )
    store = _nav._script(_STORE)
    missing = {}
    for name, call in _PATHS.items():
        body = _function_body(store, name)
        match = re.search(r"\bconst\s+callbacks\s*:\s*ChatStreamCallbacks\s*=\s*\{", body)
        assert match, f"{name} builds its callbacks as one ChatStreamCallbacks literal"
        literal = body[match.end() - 1:_balanced(body, match.end() - 1)]
        provided = _object_keys(literal)
        absent = [key for key in declared if key not in provided]
        if absent:
            missing[name] = absent
        assert re.search(rf"\b{call}\s*\([^)]*\bcallbacks\b", body), f"{name} hands its callbacks to {call}"
    assert not missing, f"every declared callback is provided by the store on both paths: {missing}"


# ---------------------------------------------------------------------------
# The chat route: its composer and its thread
# ---------------------------------------------------------------------------
def _closure(root):
    """Every repository file ``root`` reaches by import, itself first."""
    seen, todo = [], [root]
    while todo:
        path = todo.pop(0)
        if path in seen or not (REPO / path).is_file():
            continue
        seen.append(path)
        todo.extend(_nav._imports(path))
    return seen


def _mounted(path, name):
    """The repository path of the component ``path`` imports as ``name``."""
    match = re.search(rf"\bimport\s+{re.escape(name)}\s+from\s+(['\"])([^'\"]+)\1", _nav._script(path))
    assert match, f"{path} imports {name}"
    return _nav._resolve(path, match.group(2))


def _composer():
    """The component the chat route mounts with ``on:send``: its composer."""
    names = re.findall(r"<([A-Z]\w*)\b[^>]*\bon:send\s*=", _nav._markup(_ROUTE))
    assert len(names) == 1, f"the chat route mounts one composer: {names}"
    return _mounted(_ROUTE, names[0])


def _element(markup, opening):
    """The inner markup of the element ``opening`` (a match of its open tag) starts."""
    tag = re.match(r"<(\w+)", opening.group(0)).group(1)
    depth, at = 1, opening.end()
    pattern = re.compile(rf"<{tag}\b[^>]*?(/?)>|</{tag}\s*>")
    while depth:
        found = pattern.search(markup, at)
        assert found, f"the <{tag}> is closed"
        if found.group(0).startswith("</"):
            depth -= 1
        elif not found.group(1):
            depth += 1
        at = found.end()
    return markup[opening.end():found.start()]


def _thread(route=_ROUTE):
    """The route and every component it mounts in its thread, with theirs:
    what a reader hears while a reply is written into it."""
    markup = _nav._markup(route)
    opening = re.search(r"<\w+\b[^>]*\baria-label\s*=\s*[\"']Chat messages[\"'][^>]*>", markup)
    assert opening, "the chat route names its thread"
    names = sorted(set(re.findall(r"<([A-Z]\w*)\b", _element(markup, opening))))
    found = [route]
    for name in names:
        for path in _closure(_mounted(route, name)):
            if path.endswith(".svelte") and path not in found:
                found.append(path)
    return found, names


# ---------------------------------------------------------------------------
# CM5: the chat textarea stays usable while a reply streams
# ---------------------------------------------------------------------------
_TEXTAREA = re.compile(r"<textarea\b(?:[^>{]|\{[^}]*\})*>")
_DISABLED = re.compile(r"\bdisabled\s*=\s*\{([^}]*)\}")


def test_cm5_the_chat_textarea_stays_usable_while_a_reply_streams():
    sample = '<textarea bind:value={text} on:keydown={(e) => send(e)} disabled={disabled || isStreaming} />'
    tags = _TEXTAREA.findall(sample)
    assert len(tags) == 1 and _DISABLED.search(tags[0]).group(1) == "disabled || isStreaming", (
        f"the probe reads a textarea and what disables it: {tags}"
    )
    composer = _composer()
    markup, script = _nav._markup(composer), _nav._script(composer)
    tags = _TEXTAREA.findall(markup)
    assert len(tags) == 1, f"the composer the chat route mounts ({composer}) holds the chat's textarea: {len(tags)}"
    streaming = re.findall(r"\bexport\s+let\s+(\w*[Ss]treaming\w*)\b", script)
    assert streaming, f"the composer is told a reply streams: {composer}"
    disabled = _DISABLED.search(tags[0])
    read = [name for name in streaming if disabled and re.search(rf"\b{name}\b", disabled.group(1))]
    assert not read, f"the textarea is never disabled by streaming: disabled={{{disabled.group(1)}}}"
    can_send = re.search(r"\$:\s*canSend\s*=([^;]*);", script)
    assert can_send and any(re.search(rf"!\s*{name}\b", can_send.group(1)) for name in streaming), (
        f"and sending still refuses while a reply streams: {can_send.group(1) if can_send else None}"
    )


# ---------------------------------------------------------------------------
# CM7: a stopped reply keeps its text, marked stopped
# ---------------------------------------------------------------------------
def _replies(body):
    """The assistant messages a function body builds: its object literals
    that carry ``role: 'assistant'``."""
    found = []
    for match in re.finditer(r"\{", body):
        literal = body[match.start():_balanced(body, match.start())]
        if re.search(r"^\{\s*[^{}]*?\brole\s*:\s*['\"]assistant['\"]", literal, re.S) and literal not in found:
            found.append(literal)
    return found


def _value(literal, key):
    """The top-level value of ``key`` in an object literal, as written."""
    match = re.search(rf"(?:^\{{|,)\s*{key}\s*:\s*", literal)
    if not match:
        return None
    at, depth = match.end(), 0
    for end in range(at, len(literal) - 1):
        char = literal[end]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            return literal[at:end].strip()
    return literal[at:len(literal) - 1].strip()


def test_cm7_a_stopped_reply_keeps_its_text_and_says_it_stopped():
    sample = (
        "{ const kept = { id: null, role: 'assistant', content: partial + '\\n\\n[Generation cancelled]', model }; }"
    )
    replies = _replies(sample)
    assert len(replies) == 1 and _value(replies[0], "content") == "partial + '\\n\\n[Generation cancelled]'", (
        f"the probe reads a kept reply and its content: {replies}"
    )
    body = _function_body(_nav._script(_STORE), "cancelCurrentGeneration")
    kept = _replies(body)
    assert kept, "a Stop without an answer keeps the partial reply as a message"
    for literal in kept:
        content = _value(literal, "content")
        assert content and re.fullmatch(r"[A-Za-z_$][\w$]*", content), (
            f"the kept reply is the text written, nothing added to it: content: {content}"
        )
        assert _value(literal, "stopped") == "true", f"and it says it stopped: {literal}"


# ---------------------------------------------------------------------------
# CM14: one status region speaks for the stream
# ---------------------------------------------------------------------------
_LIVE_CONTENT = re.compile(r"""\baria-live\s*=|\brole\s*=\s*\{?\s*["'`](?:log|marquee)["'`]""")
_STATUS_ROLE = re.compile(r"""\brole\s*=\s*\{?\s*["'`]status["'`]""")
_STREAM_STATUS = f"{_CHAT}/StreamingStatus.svelte"

# The live regions of the thread that hold no reply, each with its reason.
_NOT_CONTENT = {
    "frontend/src/lib/components/ui/ErrorBoundary.svelte": (
        "the alert that stands in for a thread that failed to render, never a reply"
    ),
    "frontend/src/lib/ds/Toast.svelte": "a notification of the reader's own action, never a reply",
}

# The regions of the thread that do not speak for the stream, each with its reason.
_NOT_THE_STREAM = {
    f"{_CHAT}/markdown/CodeBlock.svelte": (
        "a code block's Copy outcome, said on the reader's own gesture (mk16)"
    ),
    f"{_CHAT}/MessageSkeleton.svelte": "shown while a thread loads, never while a reply streams",
}


@pytest.mark.parametrize("half", ("content", "status"))
def test_cm14_one_status_region_speaks_for_the_stream(half):
    thread, names = _thread()
    assert "ChatMessage" in names and len(thread) >= 5, (
        f"the census reads the thread and the components it mounts: {names}"
    )

    if half == "content":
        sample = '<div role="log"></div><p aria-live={on ? "polite" : "off"}></p><i role={"marquee"}></i>'
        assert len(_LIVE_CONTENT.findall(sample)) == 3, "the probe reads a live attribute and the live roles"
        assert not _LIVE_CONTENT.search('<p role="status"></p>'), "the one region is not content"
        assert all(_NOT_CONTENT.values()), "every exception carries its reason"
        live = {path: hits for path in thread if path not in _NOT_CONTENT
                for hits in [_LIVE_CONTENT.findall(_nav._markup(path))] if hits}
        assert not live, f"nothing the reply is written into is live: {live}"
        return

    assert len(_STATUS_ROLE.findall("<p role=\"status\"></p><span role={'status'}></span>")) == 2, (
        "the probe reads the role in each spelling"
    )
    assert all(_NOT_THE_STREAM.values()), "every exception carries its reason"
    counts = {path: n for path in thread for n in [len(_STATUS_ROLE.findall(_nav._markup(path)))] if n}
    speaking = {path: n for path, n in counts.items() if path not in _NOT_THE_STREAM}
    assert speaking == {_STREAM_STATUS: 1}, (
        f"while a reply streams, one region speaks for it, the loader's: {speaking}"
    )
