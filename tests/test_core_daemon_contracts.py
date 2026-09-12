#!/usr/bin/env python3
"""Contracts for the core daemon: the resident process that serves inference
to everything else, through its own registry.

The daemon is a loopback HTTP server over the inference registry. A
request from another process reaches a backend exactly as a chat turn
does: through the registry, so admission, provenance, schema and tool
options apply. The transport is the standard library's; the daemon imports
only the core. It fails closed: an unknown route, a malformed body, a
missing backend and a missing token are each a refusal by name.

  * DM1 -- health names the active backend and answers on the loopback.
  * DM2 -- generate round-trips content, thinking, tool calls and extras
    through the registry, with the options the caller sent.
  * DM3 -- stream yields the backend's chunks as JSON lines, in order,
    the last one marked done.
  * DM4 -- admission asks the governor's gate and reports admitted or
    refused with the refusal's own words.
  * DM5 -- refusals by name: 404 for an unknown route, 400 for a body that
    is not JSON, 503 when no backend serves the model.
  * DM6 -- a configured token is required on every route but health, and
    the configuration refuses a non-loopback host.

Local-only (the public distribution ships no tests). The daemon module is
loaded through the shared isolation window over a scripted registry; each
contract binds an ephemeral loopback port in a thread.
"""

import json
import sys
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402
from _registry_bridge import ScriptedBackend, StubRegistry  # noqa: E402

# The loopback needs no proxy and this suite's own opener, not the global
# urlopen another suite may have replaced.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_YAML = REPO / "opti_oignon" / "config" / "core.yaml"
_MSGS = [{"role": "user", "content": "hello"}]


class _Scripted:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([{"message": {"content": "hel"}}, {"message": {"content": "lo"}, "done": True}])
        return {"message": {"content": "hello back", "tool_calls": [
            SimpleNamespace(function=SimpleNamespace(name="search", arguments={"q": "x"}))]}}


class _Refusal(Exception):
    pass


def _backend_module(registry, refuse_model=None):
    module = types.ModuleType("opti_oignon.inference_backend")
    module.get_backend_registry = lambda: registry

    def _governor_admission(model, options):
        if model == refuse_model:
            raise _Refusal(f"governor refused {model}: capacity exhausted")
    module._governor_admission = _governor_admission
    return module


def _open(*, registry=None, refuse_model=None):
    scripted = _Scripted()
    if registry is None:
        registry = StubRegistry()
        backend = ScriptedBackend(scripted)
        registry.register(backend)
        registry.activate(backend.name)
    seeded = {"opti_oignon.inference_backend": _backend_module(registry, refuse_model)}
    loaded, restore = isolate(
        targets={"opti_oignon.core_daemon": source("core_daemon.py")},
        seeded=seeded,
        packages=("opti_oignon",),
    )
    return loaded["opti_oignon.core_daemon"], scripted, registry, restore


def _serve(mod, *, token="", registry=None):
    config = mod.CoreConfig(enabled=True, host="127.0.0.1", port=0, token=token, timeout_s=5.0)
    service = mod.CoreService(token=token)
    server = mod.make_server(config, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post(url, payload, token=None, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with _OPENER.open(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8")


def _get(url, token=None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with _OPENER.open(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8")


# ---------------------------------------------------------------------------
# DM1 -- health
# ---------------------------------------------------------------------------
def test_dm1_health_names_the_active_backend_on_the_loopback():
    mod, scripted, registry, restore = _open()
    try:
        server, base = _serve(mod)
        try:
            status, body = _get(base + "/health")
            assert status == 200
            seen = json.loads(body)
            assert seen["ok"] is True and seen["backend"] == "ollama"
            assert seen["source"] == "core-daemon"
        finally:
            server.shutdown()
        assert scripted.calls == [], "health asks the registry, not a model"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DM2 -- generate round-trips through the registry
# ---------------------------------------------------------------------------
def test_dm2_generate_round_trips_through_the_registry_with_the_callers_options():
    mod, scripted, registry, restore = _open()
    try:
        server, base = _serve(mod)
        try:
            status, body = _post(base + "/inference/generate", {
                "model": "m", "messages": _MSGS, "options": {"temperature": 0.1, "tools": [{"type": "function"}]},
                "keep_alive": "0", "think": True,
            })
        finally:
            server.shutdown()
        assert status == 200, body
        seen = json.loads(body)
        assert seen["content"] == "hello back" and seen["model"] == "m"
        assert seen["tool_calls"] == [{"name": "search", "arguments": {"q": "x"}}]
        call = scripted.calls[0]
        assert call["model"] == "m" and call["messages"] == _MSGS
        assert call["options"] == {"temperature": 0.1} and call["tools"] == [{"type": "function"}]
        assert call["keep_alive"] == "0" and call["think"] is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# DM3 -- stream as JSON lines
# ---------------------------------------------------------------------------
def test_dm3_stream_yields_the_backends_chunks_as_json_lines_in_order():
    mod, scripted, registry, restore = _open()
    try:
        server, base = _serve(mod)
        try:
            req = urllib.request.Request(base + "/inference/stream", data=json.dumps({"model": "m", "messages": _MSGS}).encode(),
                                         method="POST", headers={"Content-Type": "application/json"})
            with _OPENER.open(req, timeout=5) as resp:
                assert resp.status == 200
                lines = [json.loads(line) for line in resp.read().decode("utf-8").splitlines() if line.strip()]
        finally:
            server.shutdown()
        assert [c["content"] for c in lines] == ["hel", "lo"]
        assert [c["done"] for c in lines] == [False, True]
        assert all(c["model"] == "m" for c in lines)
        assert scripted.calls[0]["stream"] is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# DM4 -- admission
# ---------------------------------------------------------------------------
def test_dm4_admission_asks_the_gate_and_reports_the_refusal_in_its_own_words():
    mod, scripted, registry, restore = _open(refuse_model="huge:70b")
    try:
        server, base = _serve(mod)
        try:
            status, body = _post(base + "/admission", {"model": "small:1b", "requested_ctx": 4096})
            assert status == 200 and json.loads(body) == {"admitted": True, "model": "small:1b", "reason": ""}
            status, body = _post(base + "/admission", {"model": "huge:70b", "requested_ctx": 4096})
            assert status == 200
            seen = json.loads(body)
            assert seen["admitted"] is False and "capacity exhausted" in seen["reason"]
        finally:
            server.shutdown()
        assert scripted.calls == [], "admission sends nothing to a model"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DM5 -- refusals by name
# ---------------------------------------------------------------------------
def test_dm5_unknown_route_malformed_body_and_missing_backend_are_refused_by_name():
    mod, scripted, registry, restore = _open()
    try:
        server, base = _serve(mod)
        try:
            status, body = _get(base + "/nowhere")
            assert status == 404 and "route" in body
            status, body = _post(base + "/inference/generate", None, raw=b"{not json")
            assert status == 400 and "JSON" in body
            status, body = _post(base + "/inference/generate", {"messages": _MSGS})
            assert status == 400 and "model" in body, "a request without a model is malformed"
            registry.unregister("ollama")
            status, body = _post(base + "/inference/generate", {"model": "m", "messages": _MSGS})
            assert status == 503 and "no inference backend" in body
        finally:
            server.shutdown()
        assert scripted.calls == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# DM6 -- token, and loopback only
# ---------------------------------------------------------------------------
def test_dm6_a_configured_token_is_required_and_a_non_loopback_host_is_refused():
    import yaml

    mod, scripted, registry, restore = _open()
    try:
        raw = yaml.safe_load(_YAML.read_text(encoding="utf-8"))["core"]
        cfg = mod.load_config()
        assert raw["enabled"] is False and cfg.enabled is False, "off until the maintainer turns it on"
        assert cfg.host == "127.0.0.1" and cfg.port == int(raw["port"]) and cfg.validate() == []
        with pytest.raises(mod.CoreError):
            mod.CoreConfig(enabled=True, host="0.0.0.0", port=1, token="", timeout_s=5.0).validate_or_raise()
        server, base = _serve(mod, token="s3cret")
        try:
            status, _ = _get(base + "/health")
            assert status == 200, "health needs no token: it says nothing a peer could not learn by connecting"
            status, body = _post(base + "/inference/generate", {"model": "m", "messages": _MSGS})
            assert status == 401 and "token" in body
            status, body = _post(base + "/inference/generate", {"model": "m", "messages": _MSGS}, token="wrong")
            assert status == 401
            status, body = _post(base + "/inference/generate", {"model": "m", "messages": _MSGS}, token="s3cret")
            assert status == 200
        finally:
            server.shutdown()
        assert len(scripted.calls) == 1, "only the authorised request reached a model"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
