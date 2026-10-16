#!/usr/bin/env python3
"""Contracts for what the llama-server head says of an answer: the server's counts, on the stream's last chunk.

The Ollama head's last chunk names what the engine counted -- the whole
prompt, its cached prefix included, and the tokens it generated -- and how
long each took. The llama-server head does the same from what the server
reports at the end of its stream (its timings, its usage), adds why it
stopped, and uses the Ollama head's names, so the executor reads one
shape on either engine: a journal answer that used all its room, or that
the engine stopped at its own limit, is said so, and a cold window's
prompt is counted. A cut answer is never told as a whole one, and the
request the head sends does not change.

  * JW31 -- the llama-server stream's last chunk carries the server's
    counts, its times and why it stopped, the last it reported of each,
    and every earlier chunk none; its prompt count is the whole prompt,
    the timings' cached and evaluated tokens together, else the usage's
    prompt tokens, and a prompt the timings count only in part is not
    counted; a count that is no whole number, a time that is no finite
    number and a reason that is no plain word are left out, and a server
    that reports nothing leaves the last chunk empty. An error the server
    reports mid-stream, on an error line or in a chunk, raises after what
    was streamed before it, and so does a stream that ends with neither
    its end marker nor a reason it stopped; one whose last chunk says why
    it stopped needs no marker. The request is the one the options always
    made, whatever the server reports. Through the executor, a journal
    turn the head streamed ends "length", with its status line, when the
    server generated all the room the window kept and when it says it
    stopped at its own limit short of it, keeps the server's count of a
    cold prompt, and is cached once the server says it stopped on its own
    short of the room.

Local-only (the public distribution ships no tests). The head is loaded
through the shared isolation window over a scripted transport, its urlopen
restored however each case ends; the executor runs in the onion window
world (tests/_onion_window_world.py). No server, no model and no network
is reached.
"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_window_world as w  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_jw31_the_llama_server_streams_last_chunk_carries_the_servers_counts_and_the_request_is_unchanged": 4.0,
}

_BACKEND = "opti_oignon.inference_backend"


class _Response:
    """One scripted HTTP response, read whole or line by line."""

    def __init__(self, body=b"", lines=()):
        self._body, self._lines = body, list(lines)

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Server:
    """A llama-server as its HTTP surface answers: its slot listing, and the lines of its stream."""

    def __init__(self, lines, slots=()):
        self.lines, self.slots, self.requests = list(lines), [dict(slot) for slot in slots], []

    def urlopen(self, request, timeout=None):
        payload = json.loads(request.data.decode("utf-8")) if request.data else None
        self.requests.append({"url": request.full_url, "payload": payload})
        if request.full_url.endswith("/slots"):
            return _Response(json.dumps(self.slots).encode("utf-8"))
        return _Response(b"{}", self.lines)


def _sse(*bodies, done=True):
    """A stream as the server writes it: one ``data:`` line per chunk, then its end marker."""
    lines = [body if isinstance(body, bytes) else b"data: " + json.dumps(body).encode("utf-8") + b"\n"
             for body in bodies]
    return lines + ([b"data: [DONE]\n"] if done else [])


def _chunk(content=None, finish=None, **reported):
    """A chunk as llama-server's chat endpoint streams it, with what it reports beside it."""
    body = {"choices": [{"finish_reason": finish, "index": 0,
                         "delta": {} if content is None else {"content": content}}],
            "created": 1, "id": "chatcmpl-1", "model": "m", "object": "chat.completion.chunk"}
    body.update(reported)
    return body


_TIMINGS = {"cache_n": 1536, "prompt_n": 512, "prompt_ms": 120.5, "prompt_per_token_ms": 0.235,
            "prompt_per_second": 4250.0, "predicted_n": 300, "predicted_ms": 8000.25,
            "predicted_per_token_ms": 26.7, "predicted_per_second": 37.5}
_USAGE = {"completion_tokens": 300, "prompt_tokens": 2048, "total_tokens": 2348}
# The whole prompt, as Ollama counts it: the timings' cached tokens and the ones evaluated past them.
_REPORTED = {"prompt_eval_count": 1536 + 512, "eval_count": 300, "prompt_eval_duration": 120_500_000,
             "eval_duration": 8_000_250_000, "done_reason": "length"}
_MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
_OPTIONS = {"temperature": 0.2, "num_ctx": 8192, "num_predict": 300, "max_tokens": 300, "cache_prompt": True,
            "id_slot": 1}


@contextmanager
def _head(server):
    """The llama-server head from its file, its transport ``server``; urlopen restored however the block ends."""
    loaded, close = isolate(targets={_BACKEND: source("inference_backend.py")}, seeded={}, packages=("opti_oignon",))
    module = loaded[_BACKEND]
    real = module.urllib.request.urlopen
    module.urllib.request.urlopen = server.urlopen
    try:
        # A host off the machine is refused in Bulbe mode; the head is read in Daily.
        module._live_mode = lambda: "daily"
        yield module.LlamaServerBackend(host="http://fake:8080")
    finally:
        module.urllib.request.urlopen = real
        close()


def _streamed(lines, options=_OPTIONS):
    """What the head yields for a server streaming ``lines``, and the request it sent."""
    server = _Server(lines)
    with _head(server) as head:
        chunks = list(head.stream(model="m", messages=list(_MESSAGES), options=dict(options)))
    (request,) = server.requests
    return chunks, request["payload"]


def _broken(lines):
    """What the head yields for a server streaming ``lines`` before it raises, and what it raised."""
    chunks = []
    with _head(_Server(lines)) as head, pytest.raises(RuntimeError) as raised:
        for chunk in head.stream(model="m", messages=list(_MESSAGES), options=dict(_OPTIONS)):
            chunks.append(chunk)
    return chunks, str(raised.value)


def test_jw31_the_llama_server_streams_last_chunk_carries_the_servers_counts_and_the_request_is_unchanged():
    # The server's report, on the last chunk alone, the last it said of each figure.
    chunks, payload = _streamed(_sse(
        _chunk(), _chunk("Hello"), _chunk(" world", timings=dict(_TIMINGS, prompt_n=511, predicted_n=2)),
        _chunk(finish="length", usage=_USAGE, timings=_TIMINGS)))
    assert [(chunk.content, chunk.done) for chunk in chunks] == [("Hello", False), (" world", False), ("", True)]
    assert [chunk.extra for chunk in chunks[:-1]] == [{}, {}], "an earlier chunk reports nothing"
    assert chunks[-1].extra == _REPORTED, chunks[-1].extra
    assert all(type(chunks[-1].extra[key]) is int for key in _REPORTED if key != "done_reason")
    # The request is the one the options always made: the reply changes nothing of it.
    assert payload == {"model": "m", "messages": _MESSAGES, "stream": True, "temperature": 0.2, "max_tokens": 300,
                       "cache_prompt": True, "id_slot": 1}, payload
    silent, unreported = _streamed(_sse(_chunk("Hello")))
    assert unreported == payload, "a server that reports nothing is sent the same request"
    assert silent[-1].done and silent[-1].extra == {}, "a count never reported stays missing"
    # The prompt is counted whole or not at all: its evaluated part alone is no count of it.
    partial, _payload = _streamed(_sse(_chunk("Hi"), _chunk(finish="stop", timings={"prompt_n": 512})))
    assert partial[-1].extra == {"done_reason": "stop"}, partial[-1].extra
    # Without its end marker, a stream whose last chunk says why it stopped still carries the report.
    cut, _payload = _streamed(_sse(_chunk("Hi"), _chunk(finish="stop", timings=_TIMINGS), done=False))
    assert cut[-1].done and cut[-1].extra == dict(_REPORTED, done_reason="stop"), cut[-1].extra
    # A stream that ends with neither raises, after what it streamed: a cut answer is no whole one.
    unmarked, why = _broken(_sse(_chunk("Hello"), done=False))
    assert [chunk.content for chunk in unmarked] == ["Hello"] and not any(chunk.done for chunk in unmarked), why
    assert "ended before its last chunk" in why, why
    # An error the server reports mid-stream raises, on an error line or in a chunk.
    for lines in (_sse(_chunk("Hello"), done=False) + [b'error: {"code": 500, "message": "decode failed"}\n'],
                  _sse(_chunk("Hello"), {"error": {"code": 500, "message": "decode failed"}})):
        streamed, why = _broken(lines)
        assert [chunk.content for chunk in streamed] == ["Hello"], streamed
        assert why == "llama-server reported an error mid-stream", why
    # What is no whole number, no finite time or no plain word is left out; the usage counts the whole prompt
    # and what was generated when the timings do not.
    hostile, _payload = _streamed(_sse(
        _chunk("Hi"),
        _chunk(finish="stop now!", usage={"completion_tokens": 7, "prompt_tokens": 99},
               timings={"prompt_n": True, "predicted_n": -4, "prompt_ms": "12", "predicted_ms": -1.0}),
        b'data: {"choices": [], "timings": {"prompt_ms": NaN, "predicted_ms": Infinity}}\n'))
    assert hostile[-1].extra == {"eval_count": 7, "prompt_eval_count": 99}, hostile[-1].extra

    # Through the executor: the journal's window streamed by this head, the server's counts read on its last chunk.
    cases = ((1024, "length", "length", False, "[!] The answer used all its room (1024 tokens)"),
             (700, "length", "length", False, "[!] The engine stopped the answer at its limit (700 of 1024 tokens)"),
             (1023, "stop", "completed", True, None))
    for generated, finish, outcome, stored, said in cases:
        server = _Server(_sse(_chunk("Hello"), _chunk(" world"),
                              _chunk(finish=finish, timings=dict(_TIMINGS, prompt_n=900, predicted_n=generated))),
                         slots=[{"id": 0, "n_ctx": 16384}])
        with _head(server) as head:
            world, close = w.open_world(gov={"num_ctx": 16384}, backend="llama_server", journal={"num_predict": 1024})
            try:
                world.executor_module.get_backend_registry().register(head)
                (record,) = w.run_turns(world, (("conv-1", w.FIRST, {}),))
            finally:
                close()
        (window,) = world.journal.composed
        assert window.num_predict == 1024 and window.cold, "control: a cold window keeping 1024 tokens"
        (row,) = world.rows
        assert row["outcome"] == outcome, (generated, finish, row["outcome"])
        assert row["engine_prompt_tokens"] == 1536 + 900, ("the server's count of a cold prompt is kept whole", row)
        assert row["cache_stored"] is stored and ("cache.put" in w.kinds(record)) is stored, (generated, row)
        statuses = [line for line in record["statuses"] if "room" in line or "its limit" in line]
        assert statuses == ([said] if said else []), statuses
        assert record["yields"] == ["Hello", " world"], record["yields"]
        streamed = [request["payload"] for request in server.requests if request["payload"] is not None]
        assert len(streamed) == 1 and streamed[0]["max_tokens"] == 1024, streamed
        assert row["win_slot_unknown"] is False and row["win_num_ctx"] == 16384, "control: the head's slot was read"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
