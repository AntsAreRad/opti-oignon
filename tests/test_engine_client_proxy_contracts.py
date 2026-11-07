#!/usr/bin/env python3
"""Contracts that a request of the Ollama engine to this machine goes there, and only there.
The HTTP client under Ollama's library (httpx) follows a proxy of the environment even to a loopback address, and
the library follows an engine's redirects: a request every "on this machine" check had judged local -- Bulbe mode's
gate, the Mycelium's encoder, bridge and reranker, the title refiner -- could leave through a proxy or a redirect.
  * PX1 -- with HTTP_PROXY, http_proxy, a name in mixed case or ALL_PROXY pointing at a proxy, generate, the stream
    and the embedding head reach the engine at a loopback address directly: the proxy receives nothing.
  * PX2 -- an engine at a loopback address that answers with a redirect is not followed: the request fails, said,
    and the address it pointed to receives nothing.
  * PX3 -- the client the engine builds for an address of this machine reads no proxy of the environment and
    follows no redirect; one for an address of another machine is built as before (a proxy may be needed to reach
    it); so is one whose address cannot be read.
  * PX4 -- a client built for this machine keeps the certificates the environment names (SSL_CERT_FILE, else
    SSL_CERT_DIR), which a client that reads the environment would have taken: a local engine behind TLS the
    environment trusts is still reached.
The engine and the proxy are servers of the contract on 127.0.0.1, on free ports (never 11434, which the session's
engine firewall refuses); the real client library speaks to them. Local-only.
"""
import http.server
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_px1_a_request_to_this_machine_never_goes_through_a_proxy_of_the_environment": 5.0,
    "test_px2_a_redirect_of_an_engine_on_this_machine_is_never_followed": 5.0,
    "test_px3_only_a_client_for_this_machine_drops_the_environments_proxies_and_redirects": 3.0,
    "test_px4_a_client_for_this_machine_keeps_the_certificates_the_environment_names": 3.0,
}

_MSGS = [{"role": "user", "content": "the owner's question"}]
_BACKEND = "opti_oignon.inference_backend"


class _Server:
    """An HTTP server of the contract on 127.0.0.1: every request line it receives, answered by ``answer``."""

    def __init__(self, answer):
        seen = self.seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - the server's spelling
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                seen.append((self.requestline, body))
                answer(self, body)

            do_GET = do_POST  # noqa: N815

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _reply(handler, status, payload, content_type="application/json", headers=()):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    for name, value in headers:
        handler.send_header(name, value)
    handler.end_headers()
    handler.wfile.write(data)


def _engine(handler, body):
    """An Ollama engine: a chat, a streamed chat, a batch of embeddings."""
    asked = json.loads(body or b"{}")
    if handler.path.endswith("/api/embed"):
        _reply(handler, 200, {"embeddings": [[0.5]] * len(asked.get("input") or [])})
    elif asked.get("stream"):
        lines = [{"message": {"role": "assistant", "content": "x"}, "done": False},
                 {"message": {"role": "assistant", "content": ""}, "done": True}]
        _reply(handler, 200, b"".join(json.dumps(line).encode() + b"\n" for line in lines), "application/x-ndjson")
    else:
        _reply(handler, 200, {"message": {"role": "assistant", "content": "ok"}, "done": True})


def _refuse(handler, _body):
    _reply(handler, 502, {"error": "a proxy of the contract"})


@pytest.fixture
def live(monkeypatch):
    """The real engine module and client library, the mode as Bulbe reads it, no governor, no proxy of the shell."""
    import os

    from opti_oignon import inference_backend

    if not inference_backend.OLLAMA_AVAILABLE:
        pytest.skip("the ollama client library is not installed")
    for name in list(os.environ):
        if name.lower().endswith("_proxy") or name == "OLLAMA_HOST":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(inference_backend, "_live_mode", lambda: "bulbe")
    # The models' provenance is OV's (test_ollama_provenance_contracts): passed here, never read from the data.
    from opti_oignon import model_provenance

    monkeypatch.setattr(model_provenance, "guard_ollama", lambda name, digest, **kwargs: None)
    monkeypatch.setattr(inference_backend, "_governor_admission", lambda model, options, unsplittable=None: None)
    monkeypatch.setattr(inference_backend, "_governor_note_threads", lambda model, options: None)
    servers = []
    yield inference_backend, monkeypatch, servers
    for server in servers:
        server.close()


def test_px1_a_request_to_this_machine_never_goes_through_a_proxy_of_the_environment(live):
    """PX1 -- the engine at 127.0.0.1 and a proxy named by HTTP_PROXY, http_proxy, Http_Proxy or ALL_PROXY: each
    head reaches the engine, and the proxy receives none of the owner's words. A client that follows the
    environment's proxy to a loopback address -> RED."""
    backends, monkeypatch, servers = live
    engine, proxy = _Server(_engine), _Server(_refuse)
    servers += [engine, proxy]
    for variable in ("HTTP_PROXY", "http_proxy", "Http_Proxy", "ALL_PROXY"):
        monkeypatch.setenv(variable, f"http://127.0.0.1:{proxy.port}")
        backend = backends.OllamaBackend(host=f"http://127.0.0.1:{engine.port}", connect_timeout=5)
        assert backend.generate("m", _MSGS).content == "ok", variable
        assert "".join(c.content for c in backend.stream("m", _MSGS)) == "x", variable
        assert backend.embed_many("m", ["a", "b"], timeout=5) == [[0.5], [0.5]], variable
        monkeypatch.delenv(variable)
    assert proxy.seen == [], ("the proxy received the owner's words", proxy.seen)
    assert len(engine.seen) == 12


def test_px2_a_redirect_of_an_engine_on_this_machine_is_never_followed(live):
    """PX2 -- the engine at 127.0.0.1 answers a chat and an embedding with a 307 to another address: neither is
    sent there, each fails and says so. A client that carries the owner's words wherever the engine points ->
    RED."""
    backends, _monkeypatch, servers = live
    elsewhere = _Server(_engine)
    redirecting = _Server(lambda handler, _body: _reply(handler, 307, b"", headers=(
        ("Location", f"http://127.0.0.1:{elsewhere.port}{handler.path}"),)))
    servers += [elsewhere, redirecting]
    backend = backends.OllamaBackend(host=f"http://127.0.0.1:{redirecting.port}", connect_timeout=5)
    with pytest.raises(Exception):
        backend.generate("m", _MSGS)
    with pytest.raises(Exception):
        backend.embed_many("m", ["a"], timeout=5)
    assert len(redirecting.seen) == 2
    assert elsewhere.seen == [], ("the redirect was followed", elsewhere.seen)


class _Library:
    """The client library, its clients' arguments kept."""

    def __init__(self):
        self.clients = []

    def Client(self, **kwargs):  # noqa: N802 - the client library's spelling
        self.clients.append(dict(kwargs))
        return self

    def chat(self, **kwargs):
        return {"message": {"content": "ok"}}


def test_px3_only_a_client_for_this_machine_drops_the_environments_proxies_and_redirects(monkeypatch):
    """PX3 -- a client built for 127.0.0.1 or localhost reads no proxy of the environment and follows no redirect;
    one for another machine's address (Daily mode, which lets it be asked) is built as before, with its host and
    timeout alone; so is one whose address cannot be read; ``OLLAMA_HOST`` moved from another machine to this one
    builds a client of its own, never the cached one. A local client built as the library's default, a remote one
    stripped of the proxy it may need, or a remote client reused for this machine -> RED."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    loaded, restore = isolate(targets={_BACKEND: source("inference_backend.py")}, packages=("opti_oignon",))
    mod = loaded[_BACKEND]
    try:
        library = _Library()
        mod.OLLAMA_AVAILABLE = True
        mod._ollama_module = library
        mod._live_mode = lambda: "daily"
        flags = {"trust_env": False, "follow_redirects": False}
        for host in ("http://127.0.0.1:11434", "http://localhost:11434"):
            mod.OllamaBackend(host=host).generate("m", _MSGS)
            assert library.clients[-1] == {"host": host, **flags}, library.clients[-1]
        mod.OllamaBackend(host="http://10.9.8.7:11434").generate("m", _MSGS, options={"timeout": 5})
        assert library.clients[-1] == {"host": "http://10.9.8.7:11434", "timeout": 5}, library.clients[-1]
        unreadable = mod.OllamaBackend(host="http://127.0.0.1:11434")
        unreadable.endpoint = lambda: None
        unreadable.generate("m", _MSGS, options={"timeout": 6})
        assert library.clients[-1] == {"host": "http://127.0.0.1:11434", "timeout": 6}, library.clients[-1]
        moving = mod.OllamaBackend()
        monkeypatch.setenv("OLLAMA_HOST", "10.9.8.7:11434")
        moving.generate("m", _MSGS, options={"timeout": 8})
        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        moving.generate("m", _MSGS, options={"timeout": 8})
        assert library.clients[-2:] == [{"timeout": 8}, {"timeout": 8, **flags}], library.clients[-2:]
    finally:
        restore()


def test_px4_a_client_for_this_machine_keeps_the_certificates_the_environment_names(monkeypatch, tmp_path):
    """PX4 -- SSL_CERT_FILE names a bundle of one certificate, SSL_CERT_DIR an empty directory: the client built
    for 127.0.0.1 is given the TLS context the installed httpx builds from them, holding exactly that certificate
    (the file first); SSL_CERT_DIR alone, or with an empty SSL_CERT_FILE, a context of that directory, which holds
    none yet; neither set, no context is given (the library's own); a client for another machine reads the
    environment itself and is given none. A local engine behind TLS the environment trusts refused, a context of
    every default certificate, or a context forced on every client -> RED."""
    certifi = pytest.importorskip("certifi")
    import ssl

    for name in ("OLLAMA_HOST", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        monkeypatch.delenv(name, raising=False)
    loaded, restore = isolate(targets={_BACKEND: source("inference_backend.py")}, packages=("opti_oignon",))
    mod = loaded[_BACKEND]
    try:
        library = _Library()
        mod.OLLAMA_AVAILABLE = True
        mod._ollama_module = library
        mod._live_mode = lambda: "daily"
        mod.OllamaBackend(host="http://127.0.0.1:11434").generate("m", _MSGS)
        assert "verify" not in library.clients[-1], "no certificate named, the library's own"
        bundle = Path(certifi.where()).read_text(encoding="ascii")
        one = bundle[bundle.index("-----BEGIN CERTIFICATE-----"):bundle.index("-----END CERTIFICATE-----") + 25]
        (tmp_path / "one.pem").write_text(one + "\n", encoding="ascii")
        (tmp_path / "empty").mkdir()
        monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "one.pem"))
        monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "empty"))
        mod.OllamaBackend(host="http://127.0.0.1:11434").generate("m", _MSGS)
        context = library.clients[-1].get("verify")
        assert isinstance(context, ssl.SSLContext) and len(context.get_ca_certs()) == 1, "the file's one certificate"
        mod.OllamaBackend(host="http://10.9.8.7:11434").generate("m", _MSGS, options={"timeout": 5})
        assert "verify" not in library.clients[-1], "a remote client reads the environment itself"
        for cafile in (None, ""):
            if cafile is None:
                monkeypatch.delenv("SSL_CERT_FILE")
            else:
                monkeypatch.setenv("SSL_CERT_FILE", cafile)
            mod.OllamaBackend(host="http://localhost:11434").generate("m", _MSGS)
            context = library.clients[-1].get("verify")
            assert isinstance(context, ssl.SSLContext) and context.get_ca_certs() == [], (
                "the empty directory's context, not every default certificate", cafile)
    finally:
        restore()
