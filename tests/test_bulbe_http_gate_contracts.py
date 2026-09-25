#!/usr/bin/env python3
"""Contracts that Bulbe mode keeps llama-server's requests on the machine.

Ollama's requests already ask the live mode first: unless it is exactly
Daily, one whose endpoint is off the machine, or unknown, is refused by
name. The external llama-server backend had no such gate -- its host comes
from ``backends.yaml`` and every request went to it whatever the mode, and
the registry's docstring said the gate existed. It now asks the same
question, before the governor is even asked for room.

  * BG1 -- in Bulbe, and whenever the mode cannot be read, a llama-server
    whose host is off the machine answers nothing and sends nothing: health
    is false, the listings are unknown, the slot listing is empty, and a
    generation or a stream is refused by name, naming the host, before any
    request is built.
  * BG2 -- the loopback, localhost and the unspecified address are this
    machine and are served in Bulbe; Daily lifts the gate; a mode that is
    not exactly Daily keeps it shut.
  * BG3 -- a refused generation never asks the governor for admission.

Local-only (the public distribution ships no tests). The backend module is
loaded through the shared isolation window with its HTTP transport faked.
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"
_REMOTE = "http://10.9.8.7:8080"
_MSGS = [{"role": "user", "content": "hi"}]


class _Response:
    def __init__(self, body, lines=None):
        self._body, self._lines = body, lines or []

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transport:
    """Stands in for urlopen: records every request and answers like a llama-server."""

    def __init__(self):
        self.requests = []

    def urlopen(self, req, timeout=None):
        self.requests.append(req.full_url)
        path = req.full_url.split("://", 1)[1].split("/", 1)[1]
        if path == "health":
            return _Response(b'{"status": "ok"}')
        if path == "v1/models":
            return _Response(b'{"data": [{"id": "served.gguf"}]}')
        if path == "slots":
            return _Response(b'[{"id": 0}]')
        payload = json.loads(req.data.decode("utf-8"))
        if payload.get("stream"):
            return _Response(b"", [b'data: {"choices": [{"delta": {"content": "ok"}}]}\n', b"data: [DONE]\n"])
        return _Response(b'{"choices": [{"message": {"content": "ok"}}], "model": "served.gguf"}')


def _governor():
    module = types.ModuleType("opti_oignon.resource_governor")

    class GovernorRefusal(RuntimeError):
        pass

    def backend_admission_gate(model, options):
        module.asked.append(model)

    module.FEATURE_AVAILABLE = True
    module.GovernorRefusal = GovernorRefusal
    module.backend_admission_gate = backend_admission_gate
    module.asked = []
    return module


def _open(mode=None, governor=None):
    seeded = {"opti_oignon.resource_governor": governor} if governor is not None else None
    loaded, restore = isolate(targets={_BACKEND: source("inference_backend.py")}, seeded=seeded, packages=("opti_oignon",))
    mod = loaded[_BACKEND]
    transport = _Transport()
    original = mod.urllib.request.urlopen
    mod.urllib.request.urlopen = transport.urlopen
    if mode is not None:
        mod._live_mode = mode

    def restore_all():
        # The transport is the process-wide urllib module: put back exactly
        # what was there, so no later suite inherits this stand-in.
        mod.urllib.request.urlopen = original
        restore()

    return mod, transport, restore_all


# ---------------------------------------------------------------------------
# BG1 -- off the machine in Bulbe: nothing answered, nothing sent
# ---------------------------------------------------------------------------
def test_bg1_in_bulbe_a_llama_server_off_the_machine_answers_nothing_and_sends_nothing():
    for mode in (lambda: "bulbe", None):
        mod, transport, restore = _open(mode)
        try:
            backend = mod.LlamaServerBackend(host=_REMOTE)
            assert backend.health_check() is False, "never resolved"
            assert backend.list_models() is None and backend.model_info("served.gguf") is None
            assert backend.slots() == []
            with pytest.raises(RuntimeError, match="Bulbe") as refused:
                backend.generate("served.gguf", _MSGS)
            assert _REMOTE in str(refused.value), "the refusal names where it would have gone"
            with pytest.raises(RuntimeError, match="Bulbe"):
                next(backend.stream("served.gguf", _MSGS))
            assert transport.requests == [], "no request was built"
        finally:
            restore()


# ---------------------------------------------------------------------------
# BG2 -- this machine is served; only Daily lifts the gate
# ---------------------------------------------------------------------------
def test_bg2_this_machine_is_served_in_bulbe_and_only_daily_lifts_the_gate():
    mod, transport, restore = _open(lambda: "bulbe")
    try:
        for local in ("http://127.0.0.1:8080", "http://localhost:8080", "http://[::1]:8080", "http://0.0.0.0:8080"):
            backend = mod.LlamaServerBackend(host=local)
            assert backend.health_check() is True, local
            assert backend.generate("served.gguf", _MSGS).content == "ok", local
        assert len(transport.requests) == 8, "control: the local hosts were really asked"
        for unexpected in ("", "unknown", "Daily "):
            mod._live_mode = lambda value=unexpected: value
            with pytest.raises(RuntimeError, match="Bulbe"):
                mod.LlamaServerBackend(host=_REMOTE).generate("served.gguf", _MSGS)
        assert len(transport.requests) == 8, "a mode that is not exactly Daily keeps the gate shut"
        mod._live_mode = lambda: "daily"
        remote = mod.LlamaServerBackend(host=_REMOTE)
        assert remote.generate("served.gguf", _MSGS).content == "ok", "Daily lifts the gate"
        assert "".join(c.content for c in remote.stream("served.gguf", _MSGS)) == "ok"
    finally:
        restore()


# ---------------------------------------------------------------------------
# BG3 -- a refused request never asks the governor
# ---------------------------------------------------------------------------
def test_bg3_a_refused_generation_never_asks_the_governor():
    governor = _governor()
    mod, transport, restore = _open(lambda: "bulbe", governor=governor)
    try:
        backend = mod.LlamaServerBackend(host=_REMOTE)
        with pytest.raises(RuntimeError, match="Bulbe"):
            backend.generate("served.gguf", _MSGS)
        with pytest.raises(RuntimeError, match="Bulbe"):
            next(backend.stream("served.gguf", _MSGS))
        assert governor.asked == [], "nothing is admitted that will not be sent"
        mod._live_mode = lambda: "daily"
        backend.generate("served.gguf", _MSGS)
        assert governor.asked == ["served.gguf"], "control: an allowed generation is admitted"
    finally:
        restore()
