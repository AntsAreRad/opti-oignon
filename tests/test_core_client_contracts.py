#!/usr/bin/env python3
"""Contracts for the remote core backend: the registry member that reaches
the core daemon over the loopback.

Registered in an application's registry when ``core.yaml`` says so, it makes
the daemon serve that application's inference: the funnel crosses the
process boundary and the application is the first pack. It is a backend
like any other to the registry, answers with the daemon's answers, and
reports itself unhealthy -- never raising -- when the daemon is absent.

  * RD1 -- generate and stream through the daemon give the daemon's
    backend's answers, options included.
  * RD2 -- an absent daemon is unhealthy, and a request to it is a
    refusal by name, not a hang.
  * RD3 -- the registry initialiser registers the remote backend only when
    the configuration enables it.

Local-only (the public distribution ships no tests). The client and the
daemon are loaded through the shared isolation window; the daemon runs in
a thread on an ephemeral loopback port.
"""

import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import ScriptedBackend, StubRegistry  # noqa: E402

_MSGS = [{"role": "user", "content": "hello"}]


class _Scripted:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([{"message": {"content": "a"}}, {"message": {"content": "b"}, "done": True}])
        return {"message": {"content": "answer", "tool_calls": []}}


class _Base:
    """The registry's abstract backend, as a stand-in: the names the client subclasses."""


class _Response:
    def __init__(self, content, thinking=None, model="", done=True, total_duration=None, extra=None, tool_calls=None):
        self.content, self.thinking, self.model, self.done = content, thinking, model, done
        self.total_duration, self.extra, self.tool_calls = total_duration, extra or {}, tool_calls or []


class _Chunk:
    def __init__(self, content="", thinking="", done=False, model=""):
        self.content, self.thinking, self.done, self.model = content, thinking, done, model


class _ModelInfo:
    def __init__(self, name, backend, **extra):
        self.name, self.backend, self.extra = name, backend, extra


def _backend_module(registry):
    module = types.ModuleType("opti_oignon.inference_backend")
    module.get_backend_registry = lambda: registry
    module._governor_admission = lambda model, options: None
    module.InferenceBackend = _Base
    module.ChatResponse = _Response
    module.StreamChunk = _Chunk
    module.BackendModelInfo = _ModelInfo
    return module


def _open():
    scripted = _Scripted()
    registry = StubRegistry()
    backend = ScriptedBackend(scripted)
    registry.register(backend)
    registry.activate(backend.name)
    loaded, restore = isolate(
        targets={
            "opti_oignon.core_daemon": source("core_daemon.py"),
            "opti_oignon.core_client": source("core_client.py"),
        },
        seeded={"opti_oignon.inference_backend": _backend_module(registry)},
        packages=("opti_oignon",),
    )
    return loaded["opti_oignon.core_daemon"], loaded["opti_oignon.core_client"], scripted, registry, restore


def _serve(daemon, token=""):
    config = daemon.CoreConfig(enabled=True, host="127.0.0.1", port=0, token=token, timeout_s=5.0)
    server = daemon.make_server(config, daemon.CoreService(token=token))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


# ---------------------------------------------------------------------------
# RD1 -- through the daemon
# ---------------------------------------------------------------------------
def test_rd1_generate_and_stream_through_the_daemon_give_the_backends_answers():
    daemon, client, scripted, registry, restore = _open()
    try:
        server, base = _serve(daemon, token="t0k")
        try:
            remote = client.RemoteCoreBackend(base, token="t0k", timeout_s=5.0)
            assert remote.name == "core" and remote.health_check() is True
            response = remote.generate("m", _MSGS, options={"temperature": 0.2, "schema": {"type": "object"}}, keep_alive="0")
            assert response.content == "answer" and response.model == "m" and response.tool_calls == []
            chunks = list(remote.stream("m", _MSGS, options={"temperature": 0.3}))
            assert [c.content for c in chunks] == ["a", "b"] and chunks[-1].done is True
        finally:
            server.shutdown()
        gen, st = scripted.calls
        assert gen["options"] == {"temperature": 0.2} and gen["format"] == {"type": "object"} and gen["keep_alive"] == "0"
        assert st["stream"] is True and st["options"] == {"temperature": 0.3}
        assert remote.model_info("m").backend == "core"
    finally:
        restore()


# ---------------------------------------------------------------------------
# RD2 -- absent daemon
# ---------------------------------------------------------------------------
def test_rd2_an_absent_daemon_is_unhealthy_and_a_request_is_a_refusal_by_name():
    daemon, client, scripted, registry, restore = _open()
    try:
        server, base = _serve(daemon)
        server.shutdown()
        server.server_close()
        remote = client.RemoteCoreBackend(base, timeout_s=1.0)
        assert remote.health_check() is False
        with pytest.raises(RuntimeError, match="core daemon"):
            remote.generate("m", _MSGS)
        with pytest.raises(RuntimeError, match="core daemon"):
            list(remote.stream("m", _MSGS))
        assert remote.list_models() == []
        assert scripted.calls == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# RD3 -- registered from the configuration only when enabled
# ---------------------------------------------------------------------------
def test_rd3_the_registry_registers_the_remote_backend_only_when_enabled(tmp_path):
    daemon, client, scripted, registry, restore = _open()
    try:
        off = tmp_path / "off.yaml"
        off.write_text("core:\n  enabled: false\n  host: 127.0.0.1\n  port: 7411\n  token: ''\n  timeout_s: 5\n", encoding="utf-8")
        assert client.register_from_config(registry, path=off) is None
        assert registry.get("core") is None
        on = tmp_path / "on.yaml"
        on.write_text("core:\n  enabled: true\n  host: 127.0.0.1\n  port: 7411\n  token: 'abc'\n  timeout_s: 5\n", encoding="utf-8")
        remote = client.register_from_config(registry, path=on)
        assert remote is not None and registry.get("core") is remote
        assert remote.base_url == "http://127.0.0.1:7411" and remote.token == "abc"
        assert registry.active_name == "core", "enabled means the daemon serves this process"
        bad = tmp_path / "bad.yaml"
        bad.write_text("core:\n  enabled: true\n  host: 0.0.0.0\n  port: 7411\n", encoding="utf-8")
        with pytest.raises(daemon.CoreError):
            client.register_from_config(registry, path=bad)
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
