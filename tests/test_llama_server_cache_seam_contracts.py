#!/usr/bin/env python3
"""What the llama-server seam promises about prompt-cache piloting.

The external llama-server keeps a per-slot KV cache and honours a
``cache_prompt`` switch on its completion endpoints; it also reports slot
state on a read-only endpoint. The seam here is deliberately thin: forward
the switch when the caller sets it, never invent it, and read the slot
state honestly. These contracts pin both sides of that thinness.

Forwarded, never invented. When the caller's options carry
``cache_prompt``, the completion payload carries it too -- non-streaming
and streaming alike. When the options do not, the payload does not: the
wire bytes of every existing call are unchanged, and the option whitelist
otherwise stays exactly what it was.

Read-only slot state. The slots listing is a plain GET parsed as JSON.
An unreachable server answers an empty list -- observability degrades to
silence, it never degrades to an exception -- while the generation paths
keep their historical contract of raising when the server is gone.

Wired under the flag, and only for this backend. The execution hub writes
``cache_prompt`` into the options only when the stable-prefix flag is on
AND the resolved backend is the llama-server one; the flag off, or any
other backend resolved, and the option never appears.

Loaded through the shared isolation window with a scripted transport; no
server, no model, no network is ever reached.
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"
_OPTIMIZER = "opti_oignon.context_optimizer"
_EXECUTOR = "opti_oignon.executor"


# ---------------------------------------------------------------------------
# Scripted transport for the backend-level contracts
# ---------------------------------------------------------------------------

class _FakeResponse:
    """One scripted HTTP response usable by both transport shapes."""

    def __init__(self, body: bytes, lines: list[bytes] | None = None):
        self._body = body
        self._lines = list(lines or [])

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transport:
    """Records every request; plays scripted responses per path."""

    def __init__(self):
        self.requests = []
        self.responses = {}
        self.unreachable = False

    def urlopen(self, req, timeout=None):
        if self.unreachable:
            raise OSError("connection refused")
        url = req.full_url
        payload = None
        if req.data:
            payload = json.loads(req.data.decode("utf-8"))
        self.requests.append({"url": url, "payload": payload})
        for path, resp in self.responses.items():
            if url.endswith(path):
                return resp
        return _FakeResponse(b"{}")


_CHAT_BODY = json.dumps(
    {"choices": [{"message": {"content": "hi"}}], "model": "m"}
).encode("utf-8")

_SSE_LINES = [
    b'data: {"choices": [{"delta": {"content": "hi"}}]}\n',
    b"data: [DONE]\n",
]


def _load_backend():
    loaded, win_restore = isolate(
        targets={_BACKEND: source("inference_backend.py")},
        seeded={},
        packages=("opti_oignon",),
    )
    mod = loaded[_BACKEND]
    transport = _Transport()
    transport.responses["/v1/chat/completions"] = _FakeResponse(
        _CHAT_BODY, lines=_SSE_LINES
    )
    real_urlopen = mod.urllib.request.urlopen
    mod.urllib.request.urlopen = transport.urlopen
    backend = mod.LlamaServerBackend(host="http://fake:8080")

    def restore():
        mod.urllib.request.urlopen = real_urlopen
        win_restore()

    return mod, backend, transport, restore


# ---------------------------------------------------------------------------
# ls1 -- control: the whitelist forwards known keys and drops strangers
# ---------------------------------------------------------------------------

def test_ls1_generate_forwards_the_whitelist_and_drops_unknown_options():
    mod, backend, transport, restore = _load_backend()
    try:
        backend.generate(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            options={"temperature": 0.3, "num_ctx": 4096, "foo": 1},
        )
        payload = transport.requests[-1]["payload"]
        assert payload["temperature"] == 0.3
        assert "foo" not in payload
        assert "num_ctx" not in payload
    finally:
        restore()


# ---------------------------------------------------------------------------
# ls2 -- generate forwards cache_prompt when the caller sets it
# ---------------------------------------------------------------------------

def test_ls2_generate_forwards_cache_prompt_when_present():
    mod, backend, transport, restore = _load_backend()
    try:
        backend.generate(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            options={"cache_prompt": True},
        )
        payload = transport.requests[-1]["payload"]
        assert payload.get("cache_prompt") is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# ls3 -- control: absent from the options, absent from the wire
# ---------------------------------------------------------------------------

def test_ls3_generate_payload_omits_cache_prompt_when_not_asked():
    mod, backend, transport, restore = _load_backend()
    try:
        backend.generate(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            options={"temperature": 0.3},
        )
        payload = transport.requests[-1]["payload"]
        assert "cache_prompt" not in payload
    finally:
        restore()


# ---------------------------------------------------------------------------
# ls4 -- the streaming path forwards it too
# ---------------------------------------------------------------------------

def test_ls4_stream_forwards_cache_prompt_when_present():
    mod, backend, transport, restore = _load_backend()
    try:
        chunks = list(
            backend.stream(
                model="m",
                messages=[{"role": "user", "content": "q"}],
                options={"cache_prompt": True, "temperature": 0.2},
            )
        )
        payload = transport.requests[-1]["payload"]
        assert payload.get("cache_prompt") is True
        assert payload["temperature"] == 0.2
        assert any(c.content == "hi" for c in chunks)
    finally:
        restore()


# ---------------------------------------------------------------------------
# ls5 -- slots reads the server's slot state as data
# ---------------------------------------------------------------------------

def test_ls5_slots_returns_the_parsed_slot_listing():
    mod, backend, transport, restore = _load_backend()
    try:
        listing = [{"id": 0, "state": 1}, {"id": 1, "state": 0}]
        transport.responses["/slots"] = _FakeResponse(
            json.dumps(listing).encode("utf-8")
        )
        got = backend.slots()
        assert got == listing
        assert transport.requests[-1]["url"].endswith("/slots")
        assert transport.requests[-1]["payload"] is None, "a read, never a write"
    finally:
        restore()


# ---------------------------------------------------------------------------
# ls6 -- unreachable server: slot observability degrades to an empty list
# ---------------------------------------------------------------------------

def test_ls6_slots_answers_an_empty_list_when_the_server_is_gone():
    mod, backend, transport, restore = _load_backend()
    try:
        transport.unreachable = True
        assert backend.slots() == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# ls7 -- control: generation still raises when the server is gone
# ---------------------------------------------------------------------------

def test_ls7_generate_still_raises_runtime_error_when_unreachable():
    mod, backend, transport, restore = _load_backend()
    try:
        transport.unreachable = True
        try:
            backend.generate(model="m", prompt="q")
        except RuntimeError:
            pass
        else:
            raise AssertionError("an unreachable server must raise")
    finally:
        restore()


# ---------------------------------------------------------------------------
# Executor-level wiring (the real hub, scripted registry)
# ---------------------------------------------------------------------------

class _RecorderBackend:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield SimpleNamespace(content="ok", thinking=None)


def _load_executor(*, flag_on, backend_name):
    recorder = _RecorderBackend(backend_name)
    registry = SimpleNamespace(
        active=recorder,
        resolve_backend=lambda model: recorder,
    )

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = lambda **k: iter([{"message": {"content": "ok"}}])

    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_model=lambda *a, **k: "test-model:1b",
        get_temperature=lambda *a, **k: 0.2,
    )
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    backends = types.ModuleType("opti_oignon.inference_backend")
    backends.get_backend_registry = lambda: registry

    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
        "opti_oignon.inference_backend": backends,
    }
    targets = {
        _OPTIMIZER: source("context_optimizer.py"),
        _EXECUTOR: source("executor.py"),
    }
    had = "ollama" in sys.modules
    prev = sys.modules.get("ollama")
    sys.modules["ollama"] = ollama_stub
    loaded, win_restore = isolate(
        targets=targets, seeded=seeded, packages=("opti_oignon",)
    )
    loaded[_OPTIMIZER].init_optimizer(
        config={"enabled": False, "stable_prefix": {"enabled": bool(flag_on)}}
    )

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return loaded[_EXECUTOR], recorder, restore


def _drive(mod, recorder):
    ex = mod.Executor()
    routing = SimpleNamespace(
        model="test-model:1b",
        task_type="general",
        temperature=0.2,
        prompt_variant="standard",
        timeout=30,
    )
    gen = ex.execute("q", routing, refine=False)
    try:
        while True:
            next(gen)
    except StopIteration:
        pass
    assert recorder.calls, "the request must reach the scripted backend"
    return recorder.calls[-1]["options"]


# ---------------------------------------------------------------------------
# ls8 -- flag on + llama-server backend: the hub asks for the prompt cache
# ---------------------------------------------------------------------------

def test_ls8_hub_sets_cache_prompt_for_llama_server_under_the_flag():
    mod, recorder, restore = _load_executor(flag_on=True, backend_name="llama_server")
    try:
        options = _drive(mod, recorder)
        assert options.get("cache_prompt") is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# ls9 -- control: flag off, the option never appears
# ---------------------------------------------------------------------------

def test_ls9_hub_leaves_options_untouched_with_the_flag_off():
    mod, recorder, restore = _load_executor(flag_on=False, backend_name="llama_server")
    try:
        options = _drive(mod, recorder)
        assert "cache_prompt" not in options
    finally:
        restore()


# ---------------------------------------------------------------------------
# ls10 -- control: another backend resolved, the option never appears
# ---------------------------------------------------------------------------

def test_ls10_hub_never_sets_cache_prompt_for_other_backends():
    mod, recorder, restore = _load_executor(flag_on=True, backend_name="ollama")
    try:
        options = _drive(mod, recorder)
        assert "cache_prompt" not in options
    finally:
        restore()
