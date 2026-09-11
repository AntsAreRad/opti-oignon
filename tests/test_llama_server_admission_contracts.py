#!/usr/bin/env python3
"""Contracts that the external server path asks the governor like the others.

The governor's admission hook stood at four heads: generate and stream on the
Ollama backend, generate and stream on the in-process llama.cpp backend. The
llama-server backend had neither. So the one backend that talks to a process
the governor cannot see -- launched host-side, holding whatever it holds -- was
also the only one that never asked the governor anything before sending a
request. A refusal the governor would have issued for the same model on
another backend was simply never solicited on this one.

  * GA1 -- generate consults the hook with the model and options it was
    given, before the request leaves.
  * GA2 -- a refusal from the hook stops the request: nothing reaches the
    transport, because a refusal that arrives after the request has left is
    a log line, not a refusal.
  * GA3 -- the streaming head is gated the same way.

Nothing here decides what the governor should answer; the hook is a recorder
and its verdict is the test's. These contracts pin only that it is asked.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; the transport is injected and no server is reached.
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"

_SSE_LINES = [
    b'data: {"choices": [{"delta": {"content": "ok"}}]}\n',
    b"data: [DONE]\n",
]


class _FakeResponse:
    def __init__(self, body, lines=None):
        self._body = body
        self._lines = lines or []

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transport:
    def __init__(self):
        self.requests = []

    def urlopen(self, req, timeout=None):
        self.requests.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(
            json.dumps(
                {"choices": [{"message": {"content": "ok"}}], "model": "m"}
            ).encode("utf-8"),
            lines=_SSE_LINES,
        )


class _Gate:
    """A stand-in admission hook: records every consultation, may refuse."""

    def __init__(self, refuse=False):
        self.calls = []
        self.refuse = refuse

    def __call__(self, model, options):
        self.calls.append({"model": model, "options": options})
        if self.refuse:
            raise RuntimeError("refused by the stand-in gate")


def _open(refuse=False):
    loaded, restore = isolate(
        targets={_BACKEND: source("inference_backend.py")},
        packages=("opti_oignon",),
    )
    mod = loaded[_BACKEND]
    transport = _Transport()
    mod.urllib.request.urlopen = transport.urlopen
    gate = _Gate(refuse=refuse)
    mod._governor_admission = gate
    backend = mod.LlamaServerBackend(host="http://fake:8080")
    return backend, gate, transport, restore


_MESSAGES = [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# GA1 -- generate consults the hook before the request leaves
# ---------------------------------------------------------------------------
def test_ga1_generate_consults_the_governor_first():
    backend, gate, transport, restore = _open()
    try:
        backend.generate("m", _MESSAGES, options={"num_ctx": 4096})
        assert len(gate.calls) == 1, (
            "the governor was consulted exactly once for one request"
        )
        assert gate.calls[0]["model"] == "m", "with the model being served"
        assert gate.calls[0]["options"] == {"num_ctx": 4096}, (
            "and the options it will be served with, which is what an "
            "admission decision is made from"
        )
        assert len(transport.requests) == 1, (
            "and the request did leave afterwards, so the gate is not a wall"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# GA2 -- a refusal stops the request
# ---------------------------------------------------------------------------
def test_ga2_a_refusal_stops_the_request():
    backend, gate, transport, restore = _open(refuse=True)
    try:
        raised = ""
        try:
            backend.generate("m", _MESSAGES)
        except RuntimeError as exc:
            raised = str(exc)
        assert "refused" in raised, "the refusal propagates to the caller"
        assert gate.calls, "the governor was asked, so this is not vacuous"
        assert transport.requests == [], (
            "nothing reached the transport: a refusal issued after the "
            "request has left is a log line, not a refusal"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# GA3 -- the streaming head is gated the same way
# ---------------------------------------------------------------------------
def test_ga3_stream_is_gated_the_same_way():
    backend, gate, transport, restore = _open()
    try:
        list(backend.stream("m", _MESSAGES, options={"num_ctx": 2048}))
        assert len(gate.calls) == 1, "the streaming head asks once"
        assert gate.calls[0]["options"] == {"num_ctx": 2048}
        assert len(transport.requests) == 1, "and the stream then leaves"
    finally:
        restore()

    backend, gate, transport, restore = _open(refuse=True)
    try:
        raised = False
        try:
            list(backend.stream("m", _MESSAGES))
        except RuntimeError:
            raised = True
        assert raised, "a refusal on the streaming head propagates too"
        assert transport.requests == [], (
            "and nothing left, exactly as on the whole-answer head"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("GA1 generate consults the governor first", test_ga1_generate_consults_the_governor_first),
        ("GA2 a refusal stops the request", test_ga2_a_refusal_stops_the_request),
        ("GA3 stream is gated the same way", test_ga3_stream_is_gated_the_same_way),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
