"""Contracts of llama-server under the provenance seal (MP-01, what remained): the llama-server backend speaks to an
EXTERNAL server, which loads the GGUF it was started with; no request checked which. The gate now asks the server which
file it serves (``/props`` -> ``model_path``) and classifies that file against the GGUF pins as the in-process load
seam does, before each request: refused under Bulbe unless verified, observed in Daily.

  * SG1 -- THE SERVED FILE: a real local server serving a pinned GGUF: the request goes; another file (unpinned), or
    the pinned one rewritten, is refused in Bulbe before the request leaves -- streamed or whole, and before the
    governor is asked -- and observed in Daily, said once over several requests.
  * SG2 -- NOTHING TO VERIFY: a server that names no file, names it by a relative path, or a file this process cannot
    read: not verified -- refused in Bulbe, observed in Daily, said; a server that answers oddly never raises; a
    server off this machine has no file opened for it; a manifest missing is decided without asking the server.
  * SG3 -- THE KEPT DIGEST: a GGUF is hashed once per identity (device, inode, size, mtime and ctime in ns): a file
    rewritten to the same size with its mtime set back is read again (its ctime moved); two first readers at once hash
    it once; something that is not a regular file is unreadable, never waited on.
  * SG4 -- THE SERVER ASKED: ``/props`` kept SERVED_TTL_S seconds; the host rule first, before the server is asked.
  * SG5 -- THE GUARDED BACKENDS (supersedes OV6 of tests/test_ollama_provenance_contracts.py, which held llama-server
    unguarded): llama_cpp, Ollama and llama-server guarded, no name not; the provenance module unimportable refuses a
    llama-server request in Bulbe; the startup checklist's coverage line says what llama-server verifies.
  * SG6 -- THE ESCALATION (supersedes OV7 there, which let llama-server escalate whatever its manifest): a
    llama-server host whose manifest pins no GGUF file is held, one that pins one escalates; the Ollama and llama_cpp
    rules of OV7 unchanged.
  * SG7 -- THE TEXTS: SECURITY.md says MP-01 covers the three engines and the gate's limits; the CHANGELOG says it.
  * SG8 -- EVERY FILE THE SERVER LOADS: each shard of a split model must be pinned and hold; a server that applies
    adapters (LoRA) is refused in Bulbe, observed in Daily.
The suite runs the real package: the key a test key, the manifest and the GGUF in a test folder, the mode named per
contract; the server a local HTTP server of the test's own.
"""
import hashlib
import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ollama_provenance_contracts import _KEY  # noqa: E402

BUDGET_S = {
    "test_sg1_the_served_file": 4.0,
    "test_sg2_nothing_to_verify": 4.0,
    "test_sg3_the_kept_digest": 3.0,
    "test_sg4_the_server_asked": 4.0,
    "test_sg5_the_guarded_backends": 4.0,
    "test_sg6_the_escalation": 4.0,
    "test_sg7_the_texts": 2.0,
    "test_sg8_every_file_the_server_loads": 4.0,
}

_ROOT = Path(__file__).resolve().parents[1]
_MESSAGES = [{"role": "user", "content": "hello"}]


class _Server:
    """A local llama-server stand-in: ``/props`` names ``self.path``; ``/v1/chat/completions`` answers; every path
    asked is recorded."""

    def __init__(self):
        self.path = None
        self.adapters = []
        self.asked = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, body):
                data = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                owner.asked.append(self.path)
                if self.path.startswith("/lora-adapters"):
                    self._answer(owner.adapters)
                else:
                    self._answer({"model_path": owner.path} if owner.path is not None else {})

            def do_POST(self):
                owner.asked.append(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if body.get("stream"):
                    data = (b'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}\n\n'
                            b"data: [DONE]\n\n")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self._answer({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "model": "m"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.host = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The real provenance module over a test manifest and key, a GGUF enrolled, a local server serving it; the
    backend's admission hook a pass; the mode Bulbe unless a contract names another."""
    from opti_oignon import inference_backend as ib
    from opti_oignon import model_provenance as mp

    keys = mp.SealKeys(scheme=mp.SCHEME_HMAC, sign_key=_KEY, verify_key=_KEY)
    monkeypatch.setattr(mp, "resolve_seal_keys", lambda: keys)
    monkeypatch.setattr(mp, "DEFAULT_MANIFEST_PATH", tmp_path / "model_provenance.json")
    monkeypatch.setattr(mp, "_load_config", lambda: {})
    monkeypatch.setattr(mp, "_said_served", set())
    monkeypatch.setattr(mp, "_kept", {})
    mode = {"now": "bulbe"}
    monkeypatch.setattr(ib, "_provenance_mode", lambda: mode["now"])
    monkeypatch.setattr(mp, "current_mode", lambda: mode["now"])
    admitted = []
    monkeypatch.setattr(ib, "_governor_admission", lambda model, options: admitted.append(model))
    gguf = tmp_path / "models" / "tiny.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF the weights of tiny")
    mp.enroll_models([gguf])
    server = _Server()
    server.path = str(gguf)
    backend = ib.LlamaServerBackend(host=server.host)
    yield type("World", (), {"mp": mp, "ib": ib, "gguf": gguf, "server": server, "backend": backend, "mode": mode,
                             "tmp": tmp_path, "admitted": admitted})
    server.close()


def _refused(world):
    """The reason a request is refused for under the provenance gate, or None when it goes."""
    try:
        world.backend.generate("m", _MESSAGES)
    except world.mp.ProvenanceRefusal as refusal:
        return refusal.decision.reason
    return None


def _refused_streaming(world):
    """The reason a streamed request is refused for, or None when it goes (its first chunk read)."""
    try:
        next(iter(world.backend.stream("m", _MESSAGES)))
    except world.mp.ProvenanceRefusal as refusal:
        return refusal.decision.reason
    return None


def _chats(world):
    return [p for p in world.server.asked if p.startswith("/v1/chat/completions")]


def test_sg1_the_served_file(world, caplog):
    mp = world.mp
    assert _refused(world) is None and len(_chats(world)) == 1, world.server.asked
    # Another file, not pinned: refused before the request leaves.
    other = world.gguf.parent / "other.gguf"
    other.write_bytes(b"GGUF other weights")
    world.server.path = str(other)
    world.backend._served = None
    assert _refused(world) == mp.REASON_UNPINNED and len(_chats(world)) == 1, world.server.asked
    assert _refused_streaming(world) == mp.REASON_UNPINNED and len(_chats(world)) == 1, "a refused stream left"
    assert world.admitted == ["m"], "the governor was asked for a request the gate refused"
    # The pinned file rewritten (its identity changes): refused.
    world.server.path = str(world.gguf)
    world.backend._served = None
    world.gguf.write_bytes(b"GGUF rewritten weights, same name")
    assert _refused(world) == mp.REASON_DIGEST_MISMATCH and len(_chats(world)) == 1
    # Daily observes it, said once over three requests.
    world.mode["now"] = "daily"
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="opti_oignon.model_provenance"):
        for _ in range(3):
            assert _refused(world) is None
    said = [r for r in caplog.records if "llama-server serves" in r.getMessage()]
    assert len(said) == 1 and mp.REASON_DIGEST_MISMATCH in said[0].getMessage(), [r.getMessage() for r in said]
    assert len(_chats(world)) == 4
    assert _refused_streaming(world) is None and len(_chats(world)) == 5, "a stream observed in Daily did not go"


def test_sg2_nothing_to_verify(world):
    mp = world.mp
    for path, reason in ((None, mp.REASON_NOT_SERVED), ("models/tiny.gguf", mp.REASON_NOT_SERVED)):
        world.server.path = path
        world.backend._served = None
        assert _refused(world) == reason, path
    world.server.path = str(world.gguf)
    world.backend._served = None
    world.gguf.chmod(0o000)
    try:
        if os.access(world.gguf, os.R_OK):
            pytest.skip("this process reads a file whatever its rights")
        mp._kept.clear()
        assert _refused(world) == mp.REASON_FILE_UNREADABLE
        world.mode["now"] = "daily"
        assert _refused(world) is None
    finally:
        world.gguf.chmod(0o644)
    assert _chats(world) == ["/v1/chat/completions"], world.server.asked
    # A server that answers oddly never raises: a name this machine cannot open, an answer cut short.
    world.mode["now"] = "daily"
    world.server.path = "/tmp/a\x00b/tiny.gguf"
    world.backend._served = None
    assert _refused(world) is None
    assert mp.verify_served("/tmp/a\x00b/tiny.gguf", mode="bulbe").reason == mp.REASON_NOT_SERVED
    import http.client

    def cut(path, *args, **kwargs):
        raise http.client.IncompleteRead(b"")

    original = world.backend._request
    world.backend._request = cut
    try:
        assert world.backend.served_path() is None and world.backend.served_adapters() is None
    finally:
        world.backend._request = original
    # A server off this machine: no file opened here for it, the server never asked for one.
    world.server.path = str(world.gguf)
    world.backend._served = None
    world.server.asked.clear()
    original_rule = world.ib._is_on_this_machine
    world.ib._is_on_this_machine = lambda url: False
    try:
        assert _refused(world) is None and "/props" not in world.server.asked, world.server.asked
    finally:
        world.ib._is_on_this_machine = original_rule
    world.mode["now"] = "bulbe"
    # A manifest missing: decided without asking the server.
    world.mode["now"] = "bulbe"
    (world.tmp / "model_provenance.json").unlink()
    world.backend._served = None
    world.server.asked.clear()
    assert _refused(world) == mp.REASON_MANIFEST_MISSING and world.server.asked == [], world.server.asked


def test_sg3_the_kept_digest(world, monkeypatch):
    mp = world.mp
    gguf = world.gguf
    hashed = []

    def sha256():
        hashed.append(1)
        return hashlib.sha256()

    monkeypatch.setattr(mp, "hashlib", type("Counting", (), {"sha256": staticmethod(sha256)}))
    first = mp.kept_digest(gguf)
    assert mp.kept_digest(gguf) == first and len(hashed) == 1, "a file whose identity holds was read again"
    # Its bytes rewritten to the same size, its mtime set back: its ctime moved, so it is read again.
    st = gguf.stat()
    data = b"X" * st.st_size
    gguf.write_bytes(data)
    os.utime(gguf, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert gguf.stat().st_size == st.st_size and gguf.stat().st_mtime_ns == st.st_mtime_ns
    assert mp.kept_digest(gguf) == hashlib.sha256(data).hexdigest() and len(hashed) == 2
    # Two first readers at once: one reads, the other finds it kept. The first reader's hash waits for a second one
    # to start hashing (a rendezvous, half a second at most): one per path at a time, so none comes.
    other = world.tmp / "models" / "other.gguf"
    other.write_bytes(b"GGUF " * 100000)
    hashed.clear()
    second = threading.Event()

    def meeting():
        hashed.append(1)
        if len(hashed) == 1:
            second.wait(0.5)
        else:
            second.set()
        return hashlib.sha256()

    monkeypatch.setattr(mp, "hashlib", type("Meeting", (), {"sha256": staticmethod(meeting)}))
    seen = []
    readers = [threading.Thread(target=lambda: seen.append(mp.kept_digest(other))) for _ in range(2)]
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join(10)
    assert len(seen) == 2 and seen[0] == seen[1] and len(hashed) == 1, (seen, hashed)
    # Not a regular file: never waited on.
    fifo = world.tmp / "pipe.gguf"
    os.mkfifo(fifo)
    out = []
    reader = threading.Thread(target=lambda: out.append(_raised(mp.kept_digest, fifo)), daemon=True)
    reader.start()
    reader.join(5)
    assert not reader.is_alive() and out and isinstance(out[0], OSError), out


def _raised(function, *args):
    try:
        return function(*args)
    except OSError as exc:
        return exc


def test_sg4_the_server_asked(world, monkeypatch):
    for _ in range(3):
        assert _refused(world) is None
    assert [p for p in world.server.asked if p == "/props"] == ["/props"], world.server.asked
    monkeypatch.setattr(world.backend, "SERVED_TTL_S", 0.0)
    assert _refused(world) is None
    assert [p for p in world.server.asked if p == "/props"] == ["/props", "/props"], world.server.asked
    # The host rule first: a host off the machine in Bulbe is refused before the server is asked.
    world.server.asked.clear()
    far = world.ib.LlamaServerBackend(host="http://192.0.2.1:8080")
    with pytest.raises(RuntimeError, match="(?i)bulbe"):
        far.generate("m", _MESSAGES)
    assert world.server.asked == []


def test_sg5_the_guarded_backends(world, monkeypatch):
    mp, ib = world.mp, world.ib
    for backend in ("llama_cpp", "ollama", "llama_server"):
        assert mp.backend_enforces_provenance(backend), backend
    assert not mp.backend_enforces_provenance(None)
    import opti_oignon

    monkeypatch.delattr(opti_oignon, "model_provenance", raising=False)
    monkeypatch.setitem(sys.modules, "opti_oignon.model_provenance", None)
    world.server.asked.clear()
    with pytest.raises(RuntimeError, match="refusing to send a request"):
        world.backend.generate("m", _MESSAGES)
    assert world.server.asked == []
    monkeypatch.undo()
    from opti_oignon import security_mode as sm
    from opti_oignon import startup_checks as sc

    monkeypatch.setattr(sm, "_default_backend", lambda: "llama_server")
    item = sc._check_backend_provenance_coverage()
    assert item.passed is True and "the file the server says it serves" in item.detail, item.detail


def test_sg6_the_escalation(tmp_path):
    import test_escalation_preflight_verifies_contracts as ev

    world = ev._world(tmp_path / "llama_server_ollama_only", backend="llama_server")
    try:
        payload = ev._old_payload({"ollama:llama3:latest": {"sha256": "a" * 64, "kind": "ollama", "layers": [],
                                                             "recorded_at": 1.0}})
        ev._old_write(world.manifest, payload, ev._old_pqc(payload))
        result = ev._escalated(world)
        assert result["success"] is False and "pins no GGUF model" in str(result), result
        assert world.written == []
    finally:
        world.restore()
    world = ev._world(tmp_path / "llama_server_signed", backend="llama_server")
    try:
        ev._state(world, "signed")
        result = ev._escalated(world)
        assert result["success"] is True and world.written == [world.mod.MODE_BULBE], result
    finally:
        world.restore()
    for backend, name, said in (("ollama", "signed", "pins no Ollama model"), ("ollama", "another_keypair", "refused"),
                                 (None, "signed", "pins no Ollama model")):
        world = ev._world(tmp_path / f"{backend}_{name}", backend=backend)
        try:
            ev._state(world, name)
            result = ev._escalated(world)
            assert result["success"] is False and world.written == [], (backend, name, result)
            assert said in str(result), (backend, name, result)
        finally:
            world.restore()


def test_sg7_the_texts():
    security = " ".join((_ROOT / "SECURITY.md").read_text(encoding="utf-8").split())
    for said in ("the llama-server backend before each request", "the file the server says it serves",
                 "a hostile server may name another", "the file's identity"):
        assert said in security, said
    assert "the llama-server backend loads weights without an integrity check" not in security
    changelog = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    added = changelog.split("### Added\n", 1)[1].split("\n### ", 1)[0]
    entry = " ".join(next((entry for entry in added.split("\n- ") if "SG1-SG8" in entry), "").split())
    assert "/props" in entry and "llama-server" in entry, entry[:300]



def test_sg8_every_file_the_server_loads(world):
    mp = world.mp
    # A split model: every shard pinned and holding verifies; a shard rewritten, or not pinned, refuses.
    first = world.gguf.parent / "big-00001-of-00002.gguf"
    second = world.gguf.parent / "big-00002-of-00002.gguf"
    first.write_bytes(b"GGUF shard one")
    second.write_bytes(b"GGUF shard two")
    mp.enroll_models([first, second])
    world.server.path = str(first)
    world.backend._served = None
    assert _refused(world) is None
    second.write_bytes(b"GGUF shard two, rewritten")
    world.backend._served = None
    decision = mp.verify_served(str(first), mode="bulbe")
    assert decision.reason == mp.REASON_DIGEST_MISMATCH and decision.model == second.name, decision
    assert _refused(world) == mp.REASON_DIGEST_MISMATCH
    alone = world.gguf.parent / "lone-00001-of-00002.gguf"
    alone.write_bytes(b"GGUF a lone shard one")
    (world.gguf.parent / "lone-00002-of-00002.gguf").write_bytes(b"GGUF a shard never pinned")
    mp.enroll_models([alone])
    world.server.path = str(alone)
    world.backend._served = None
    assert _refused(world) == mp.REASON_UNPINNED
    # Adapters applied by the server: no pin covers them, refused in Bulbe, observed in Daily.
    world.server.path = str(world.gguf)
    world.server.adapters = [{"id": 0, "path": "/models/lora.gguf", "scale": 1.0}]
    world.backend._served = None
    world.backend._adapters = None
    assert _refused(world) == mp.REASON_ADAPTERS
    world.mode["now"] = "daily"
    assert _refused(world) is None
