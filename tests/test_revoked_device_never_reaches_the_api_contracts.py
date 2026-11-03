#!/usr/bin/env python3
"""Contracts for remote access that never lets a revoked device's certificate reach the API (block 226).

The check of each request read the client certificate from ``scope["transport"]``, which uvicorn never puts in a
request's scope: it ran only on a header a client chose to send, and the TLS layer loaded no CRL, so a revoked
device kept its access, before and after a restart. A revocation kept no earlier one when the CRL did not load. And
with remote access on and a TLS file missing, the launcher served the network address in plain HTTP.

  * TG1 -- a real server, started by ``cmd_api`` with the kwargs it gives uvicorn: a device's certificate is answered;
    revoked, its open connection is ended, a new one is refused before the app sees a request, and so after a
    restart; a certificate made again for its device ends the old one's connection and refuses it, the new one
    answered; another device keeps its connection throughout.
  * TG2 -- ``client_cert_refusal`` on each state: a device's certificate None; revoked by its record, revoked by the
    CRL alone, made again for its device (the CRL lists the old one); no device's record naming it, a folder with no
    record beside (``not_issued_here``); a CRL that does not load, a record that does not load and is not another's,
    no certificate or one that does not parse (``cannot_check``); a CRL that is not there holds no serial.
  * TG3 -- the CRL lists every certificate revoked or replaced: over one that loads, its entries and their dates;
    over one that does not load, every serial the devices' records name as revoked, or held before a certificate made
    again, with their dates; revoking a device already revoked writes one that does not load again; the record is
    marked first and names the serial, so a certificate file gone or replaced, or a CA key gone, leaves no device in;
    a malformed record blocks nothing; two revocations at once, the second read while the first writes, both listed;
    no staged file left behind.
  * TG4 -- the launcher: remote access on and a network host, with the server key, its certificate or the CA
    missing, a TLS configuration that raises or asks no client certificate, exits 1 before uvicorn runs, audited
    ``remote_bind_refused``; the whole mTLS is served through the gate; a loopback host with remote access off is
    served plain, and in Bulbe the network host is loopback, as before.
  * TG5 -- ``is_cert_serial_revoked``: a listed serial True, another False, a CRL that does not load True, none
    False; on cryptography 41's CRL entries (no ``revocation_date_utc``) the serials and dates read the same, and a
    device is not refused for them.
  * TG6 -- the texts: SECURITY.md says how a revoked or replaced certificate is refused, again while its connection is
    open, by another process's revocation too, and that a network address needs the whole mTLS, and no longer that a
    Bulbe account with no factor is not refused; ``tls_manager`` no longer says the CRL is checked on every request,
    and the middleware's check says it sees no certificate served over mTLS directly.
  * TG7 -- a revocation made by another process (the loopback server the settings page talks to), which serves none
    of the mTLS server's connections: the revoked device's open keep-alive connection and its websocket are ended
    within a few seconds, and so is the old certificate's connection when another process makes a device's again;
    another device's connection is kept; a connection the client closed, a websocket among them, leaves the open ones
    and is asked no more.

The modules run in an isolation window (``_isolation``): the real ``tls_manager``, ``network_bind_guard`` and
``main``, the mode and the audit chain seeded, the application a small ASGI app that records what it answers, the
TLS folder in the test's own directory. ``remote_session_guard`` is out of the window: the revocation does not lean
on it. Nothing reads or writes the platform's data.

Local-only (the public distribution ships no tests).
"""

import contextlib
import http.client
import json
import socket
import ssl
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from cryptography import x509
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PASSPHRASE = "a passphrase long enough"

BUDGET_S = {
    "test_tg1_a_revoked_or_replaced_certificate_is_refused_at_its_connection_and_after_a_restart": 4.0,
    "test_tg2_each_certificate_state_has_its_answer": 2.0,
    "test_tg3_the_crl_lists_every_certificate_revoked_or_replaced": 10.0,
    "test_tg4_a_network_address_is_served_with_the_whole_mtls_or_not_at_all": 2.0,
    "test_tg5_a_serial_in_a_crl_that_does_not_load_reads_revoked": 2.0,
    "test_tg6_the_texts_say_what_is_checked": 1.0,
    "test_tg7_another_process_revocation_ends_the_open_connections": 16.0,
}


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _ask(conn, path):
    """GET ``path`` on ``conn``: ``(status, body)``, or the name of the error that came instead of an answer."""
    try:
        conn.request("GET", path)
        answer = conn.getresponse()
        return answer.status, answer.read()
    except (OSError, http.client.HTTPException) as exc:
        return type(exc).__name__


class _World:
    """The window: the real TLS manager, bind guard and launcher, the TLS folder under ``root``, remote access on in a
    security.yaml of its own, the mode Daily unless told."""

    def __init__(self, root, monkeypatch):
        self.root = root
        self.monkeypatch = monkeypatch
        self.mode = "daily"
        self.audit = []
        self.seen = []
        self.servers = []
        self.clients = []
        self.runs = []
        mode = types.ModuleType("opti_oignon.security_mode")
        mode.get_current_mode = lambda: self.mode
        mode.is_bulbe = lambda: self.mode == "bulbe"
        chain = types.ModuleType("opti_oignon.signed_audit_log")
        chain.chain_log = lambda **entry: self.audit.append(entry)
        api_app = types.ModuleType("opti_oignon.api.app")
        api_app.app = self._app
        loaded, self._restore = isolate(
            targets={
                "opti_oignon.__version__": source("__version__.py"),
                "opti_oignon.tls_manager": source("tls_manager.py"),
                "opti_oignon.network_bind_guard": source("network_bind_guard.py"),
                "opti_oignon.main": source("main.py"),
            },
            blocked=("opti_oignon.remote_session_guard",),
            seeded={"opti_oignon.security_mode": mode, "opti_oignon.signed_audit_log": chain,
                    "opti_oignon.api.app": api_app},
            packages=("opti_oignon.api",),
        )
        self.tls = loaded["opti_oignon.tls_manager"]
        self.guard = loaded["opti_oignon.network_bind_guard"]
        self.main = loaded["opti_oignon.main"]
        folder = root / "tls"
        tls = self.tls
        tls._TLS_DIR = folder
        tls._CRL_PATH = folder / "crl.pem"
        tls._CA_KEY_PATH = folder / "ca.key"
        tls._CA_CERT_PATH = folder / "ca.crt"
        tls._SERVER_KEY_PATH = folder / "server.key"
        tls._SERVER_CERT_PATH = folder / "server.crt"
        tls._CA_KEY_ENC_PATH = folder / "ca.key.enc"
        tls._CLIENT_DIR = folder / "clients"
        self.security_yaml = root / "security.yaml"
        self.remote(True)
        self.guard._SECURITY_YAML = self.security_yaml
        # The guard's own check reads the platform's folder: here, the window's.
        self.guard._tls_files_exist = lambda: all(
            path.exists() and path.stat().st_size > 0
            for path in (tls._SERVER_KEY_PATH, tls._SERVER_CERT_PATH, tls._CA_CERT_PATH)
        )

    async def _app(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] == "websocket":
            await receive()
            await send({"type": "websocket.accept"})
            self.seen.append(scope["path"])
            while True:
                message = await receive()
                if message["type"] == "websocket.disconnect":
                    return
                await send({"type": "websocket.send", "text": message.get("text") or ""})
        self.seen.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
        await send({"type": "http.response.body", "body": b"ok"})

    def remote(self, enabled):
        self.security_yaml.write_text(f"remote_access:\n  enabled: {'true' if enabled else 'false'}\n",
                                      encoding="utf-8")

    def setup(self):
        assert self.tls.setup_tls(PASSPHRASE)["success"]

    def device(self, name):
        made = self.tls.generate_client_cert(name, PASSPHRASE)
        assert made["success"], made
        return made["fingerprint"]

    def der(self, name, folder=None):
        pem = ((folder or self.tls._CLIENT_DIR / name) / "client.crt").read_text(encoding="ascii")
        return ssl.PEM_cert_to_DER_cert(pem)

    def keep(self, name):
        """A copy of a device's certificate and key as they are now (a device made again overwrites them)."""
        kept = self.root / f"kept-{name}"
        kept.mkdir()
        for leaf in ("client.crt", "client.key"):
            (kept / leaf).write_bytes((self.tls._CLIENT_DIR / name / leaf).read_bytes())
        return kept

    def launch(self, host="127.0.0.1"):
        """``cmd_api`` as the launcher runs it, ``uvicorn.run`` recording what it is given: ``(port, kwargs)``, or
        the exit code when the launcher exits."""
        import uvicorn

        port = _free_port()
        self.monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: self.runs.append(kwargs))
        try:
            self.main.cmd_api(types.SimpleNamespace(host=host, port=port))
        except SystemExit as exc:
            return exc.code
        return port, self.runs[-1]

    def serve(self):
        """A real server on the kwargs the launcher gives uvicorn (loopback, an ephemeral port): its port."""
        import uvicorn

        port, given = self.launch()
        # An idle connection is kept a minute (uvicorn's five seconds would end it under the wait for a revocation).
        server = uvicorn.Server(uvicorn.Config(self._app, log_config=None, interface="asgi3", timeout_keep_alive=60,
                                               **given))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        self.servers.append((server, thread))
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "the server did not start"
            time.sleep(0.02)
        return port

    def stop(self):
        """Close the test's own connections (a server waits for an open one's TLS close), then the servers."""
        while self.clients:
            self.clients.pop().close()
        while self.servers:
            server, thread = self.servers.pop()
            server.should_exit = True
            thread.join(10)
            assert not thread.is_alive(), "the server did not stop"

    def context(self, name, folder=None):
        held = folder or self.tls._CLIENT_DIR / name
        context = ssl.create_default_context(cafile=str(self.tls._CA_CERT_PATH))
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT  # the CA's certificates carry no key identifiers
        context.load_cert_chain(str(held / "client.crt"), str(held / "client.key"))
        return context

    def connect(self, port, name, folder=None):
        self.clients.append(http.client.HTTPSConnection("127.0.0.1", port, context=self.context(name, folder),
                                                        timeout=5))
        return self.clients[-1]

    def websocket(self, port, name):
        self.clients.append(ws_connect(f"wss://127.0.0.1:{port}/ws", ssl=self.context(name), open_timeout=5,
                                       close_timeout=1))
        return self.clients[-1]

    @contextlib.contextmanager
    def elsewhere(self):
        """As another process: one that serves none of this server's connections."""
        served = self.tls._live_connections
        self.tls._live_connections = {}
        try:
            yield
        finally:
            self.tls._live_connections = served

    def open_connections(self, fingerprint):
        """The connections the gate let through with ``fingerprint`` and keeps."""
        return [t for t, (fp, _loop) in self.tls._live_connections.items() if fp == fingerprint]

    def close(self):
        self.stop()
        self._restore()


@pytest.fixture
def world(tmp_path, monkeypatch):
    made = _World(tmp_path, monkeypatch)
    yield made
    made.close()


def _closing(transports):
    deadline = time.monotonic() + 3
    while not all(t.is_closing() for t in transports):
        assert time.monotonic() < deadline, "a connection of the certificate was not ended"
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# TG1 -- a real server: revoked or replaced, refused at the connection, open connections ended, after a restart too
# ---------------------------------------------------------------------------
def test_tg1_a_revoked_or_replaced_certificate_is_refused_at_its_connection_and_after_a_restart(world):
    # The periodic check put off: here a revocation in the serving process ends the connections by itself, at once.
    world.monkeypatch.setattr(world.tls, "_RECHECK_S", 60.0)
    world.setup()
    phone = world.device("phone")
    laptop = world.device("laptop")
    tablet = world.device("tablet")
    port = world.serve()
    assert getattr(world.runs[-1].get("http"), "__name__", None) == "ClientCertGate", (
        "mTLS served without the gate", world.runs[-1])

    phone_conn = world.connect(port, "phone")
    laptop_conn = world.connect(port, "laptop")
    tablet_conn = world.connect(port, "tablet")
    assert _ask(phone_conn, "/phone-1") == (200, b"ok")
    assert _ask(laptop_conn, "/laptop-1") == (200, b"ok")
    assert _ask(tablet_conn, "/tablet-1") == (200, b"ok")

    open_phone = world.open_connections(phone)
    assert open_phone, "the gate keeps the connections it lets through"
    assert world.tls.revoke_client_cert("phone")["success"]
    _closing(open_phone)
    assert not isinstance(_ask(phone_conn, "/phone-2"), tuple), "a revoked device's open connection still answers"
    assert not isinstance(_ask(world.connect(port, "phone"), "/phone-3"), tuple), "a revoked device connects again"

    old_tablet = world.keep("tablet")
    open_tablet = world.open_connections(tablet)
    assert world.device("tablet") != tablet
    _closing(open_tablet)
    assert not isinstance(_ask(tablet_conn, "/tablet-2"), tuple), "a replaced certificate's open connection answers"
    assert not isinstance(_ask(world.connect(port, "tablet", old_tablet), "/tablet-3"), tuple), (
        "a certificate made again for its device leaves the old one in")
    assert _ask(world.connect(port, "tablet"), "/tablet-4") == (200, b"ok")
    assert _ask(laptop_conn, "/laptop-2") == (200, b"ok"), "another device's connection was ended"

    world.stop()
    port = world.serve()
    assert not isinstance(_ask(world.connect(port, "phone"), "/phone-4"), tuple), "a restart lets a revoked device in"
    assert not isinstance(_ask(world.connect(port, "tablet", old_tablet), "/tablet-5"), tuple), (
        "a restart lets a replaced certificate in")
    assert _ask(world.connect(port, "laptop"), "/laptop-3") == (200, b"ok")
    assert world.seen == ["/phone-1", "/laptop-1", "/tablet-1", "/tablet-4", "/laptop-2", "/laptop-3"], world.seen


# ---------------------------------------------------------------------------
# TG2 -- client_cert_refusal on each state
# ---------------------------------------------------------------------------
def test_tg2_each_certificate_state_has_its_answer(world):
    refusal = getattr(world.tls, "client_cert_refusal", None)
    assert callable(refusal), "no client_cert_refusal"
    world.setup()
    world.device("phone")
    world.device("laptop")
    world.device("tablet")
    phone, laptop, tablet = world.der("phone"), world.der("laptop"), world.der("tablet")
    assert (refusal(phone), refusal(laptop), refusal(tablet)) == (None, None, None)

    record = world.tls._CLIENT_DIR / "laptop" / "metadata.json"
    meta = json.loads(record.read_text(encoding="utf-8"))
    record.write_text(json.dumps({**meta, "revoked": True}), encoding="utf-8")
    assert refusal(laptop) == "revoked", "a device recorded as revoked"
    record.write_text(json.dumps({**meta, "revoked": "no"}), encoding="utf-8")
    assert refusal(laptop) == "revoked", "a record whose revoked flag is not false"
    record.write_text(json.dumps(meta), encoding="utf-8")

    assert world.tls.revoke_client_cert("phone")["success"]
    record = world.tls._CLIENT_DIR / "phone" / "metadata.json"
    record.write_text(json.dumps({**json.loads(record.read_text(encoding="utf-8")), "revoked": False}),
                      encoding="utf-8")
    assert refusal(phone) == "revoked", "a serial in the CRL, its record saying otherwise"

    old_tablet = tablet
    world.device("tablet")
    assert refusal(old_tablet) == "revoked", "a certificate made again for its device: the CRL lists the old one"
    assert refusal(world.der("tablet")) is None

    crl = world.tls._CRL_PATH.read_bytes()
    world.tls._CRL_PATH.write_bytes(b"-----BEGIN X509 CRL-----\nnot one\n-----END X509 CRL-----\n")
    assert refusal(laptop) == "cannot_check", "a CRL that does not load"
    world.tls._CRL_PATH.unlink()
    assert refusal(laptop) is None, "a CRL that is not there holds no serial"
    assert refusal(old_tablet) == "not_issued_here", "a certificate no device's record names"
    (world.tls._CLIENT_DIR / "half-made").mkdir()
    assert refusal(old_tablet) == "not_issued_here", "a folder with no record is no device's"
    world.tls._CRL_PATH.write_bytes(crl)

    other = world.tls._CLIENT_DIR / "tablet" / "metadata.json"
    kept = other.read_text(encoding="utf-8")
    other.write_text("{ not json", encoding="utf-8")
    assert refusal(laptop) is None, "a record that does not load is another device's"
    assert refusal(world.der("tablet")) == "cannot_check", "its own record does not load"
    other.write_text("[]", encoding="utf-8")
    assert refusal(world.der("tablet")) == "cannot_check", "its own record is no mapping"
    other.write_text(kept, encoding="utf-8")

    assert refusal(None) == "cannot_check"
    assert refusal(b"") == "cannot_check"
    assert refusal(b"not a certificate") == "cannot_check"


# ---------------------------------------------------------------------------
# TG3 -- the CRL lists every certificate revoked or replaced
# ---------------------------------------------------------------------------
def test_tg3_the_crl_lists_every_certificate_revoked_or_replaced(world):
    names = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
             "thirteen", "fourteen")

    def listed():
        crl = x509.load_pem_x509_crl(world.tls._CRL_PATH.read_bytes())
        return {entry.serial_number: entry.revocation_date_utc for entry in crl}

    def serial_of(name, folder=None):
        return x509.load_der_x509_certificate(world.der(name, folder)).serial_number

    world.setup()
    for name in names:
        world.device(name)
    serial = {name: serial_of(name) for name in names}

    assert world.tls.revoke_client_cert("one")["success"]
    first = listed()
    assert set(first) == {serial["one"]}
    time.sleep(1.1)  # a CRL's dates are to the second
    assert world.tls.revoke_client_cert("two")["success"]
    second = listed()
    assert set(second) == {serial["one"], serial["two"]}
    assert second[serial["one"]] == first[serial["one"]], "an earlier revocation's date was rewritten"

    world.tls._CRL_PATH.write_bytes(b"garbage")
    assert world.tls.revoke_client_cert("three")["success"]
    third = listed()
    assert set(third) == {serial["one"], serial["two"], serial["three"]}, "an unreadable CRL dropped a revocation"
    assert abs((third[serial["one"]] - first[serial["one"]]).total_seconds()) <= 1, (
        "a revocation written again from its record lost its date")
    assert third[serial["one"]] < second[serial["two"]], "a revocation written again from its record is dated now"

    world.tls._CRL_PATH.write_bytes(b"garbage")
    again = world.tls.revoke_client_cert("one")
    assert again["success"] and again.get("already_revoked"), again
    assert set(listed()) == set(third), "revoking a device already revoked leaves a CRL that does not load"
    assert listed()[serial["one"]] == third[serial["one"]]
    assert world.tls.revoke_client_cert("one").get("already_revoked")

    world.device("four")
    assert serial["four"] in listed(), "a certificate made again leaves the old one out of the CRL"
    world.tls.revoke_client_cert("two")  # already revoked and listed: nothing written
    world.tls._CRL_PATH.write_bytes(b"garbage")
    assert world.tls.revoke_client_cert("five")["success"]
    assert serial["four"] in listed(), "a CRL written again from the records dropped a replaced certificate"
    assert listed()[serial["one"]] == third[serial["one"]], "revoking a device again moved its revocation's date"
    assert world.tls.is_cert_serial_revoked(serial["four"])
    assert world.tls.client_cert_refusal(world.der("four")) is None

    world.tls._CRL_PATH.unlink()
    assert world.tls.revoke_client_cert("six")["success"]
    assert {serial[name] for name in ("one", "two", "three", "four", "five", "six")} <= set(listed()), (
        "a missing CRL dropped a revocation")

    # The record first, its serial from the record: a certificate file gone, or one a remake replaced, or no CA key.
    nine = world.der("nine")
    (world.tls._CLIENT_DIR / "nine" / "client.crt").unlink()
    assert world.tls.revoke_client_cert("nine")["success"], "a device whose certificate file is gone is not revoked"
    assert world.tls.client_cert_refusal(nine) == "revoked" and serial["nine"] in listed()
    (world.tls._CLIENT_DIR / "ten" / "client.crt").write_bytes(
        (world.tls._CLIENT_DIR / "eleven" / "client.crt").read_bytes())
    assert world.tls.revoke_client_cert("ten")["success"]
    assert serial["ten"] in listed() and serial["eleven"] not in listed(), (
        "the serial was read from a certificate file a remake replaced")
    eleven = world.der("eleven")
    key = world.tls._CA_KEY_PATH
    key.rename(key.with_name("ca.key.aside"))
    try:
        answer = world.tls.revoke_client_cert("eleven")
        assert answer["success"] and answer.get("crl_written") is False, answer
        assert world.tls.client_cert_refusal(eleven) == "revoked", "no CA key: the device is left in"
    finally:
        key.with_name("ca.key.aside").rename(key)
    assert serial["eleven"] not in listed()
    assert world.tls.revoke_client_cert("eleven").get("already_revoked")
    assert serial["eleven"] in listed(), "revoking it again leaves it out of the CRL"

    # A malformed record blocks no revocation and no remake.
    for name, change in (("twelve", {"previous_serials": None}),
                         ("thirteen", {"previous_serials": [{"serial_number": "ff", "withdrawn_at": 1e30}]})):
        record = world.tls._CLIENT_DIR / name / "metadata.json"
        record.write_text(json.dumps({**json.loads(record.read_text(encoding="utf-8")), **change}), encoding="utf-8")
    assert world.tls.revoke_client_cert("fourteen")["success"]
    assert serial["fourteen"] in listed() and 0xFF in listed()
    world.device("twelve")
    world.device("thirteen")

    # Two revocations at once: the second is asked while the first, its CRL and records read, has yet to write.
    real = world.tls._recorded_withdrawn_serials
    other = []

    def records_then_another_revocation():
        read = real()
        if not other:
            other.append(threading.Thread(target=world.tls.revoke_client_cert, args=("eight",)))
            other[0].start()
            other[0].join(2.0)  # held by the lock it waits the two seconds out; unheld it ends at once
        return read

    world.monkeypatch.setattr(world.tls, "_recorded_withdrawn_serials", records_then_another_revocation)
    assert world.tls.revoke_client_cert("seven")["success"]
    other[0].join(10)
    world.monkeypatch.setattr(world.tls, "_recorded_withdrawn_serials", real)
    assert {serial["seven"], serial["eight"]} <= set(listed()), "two revocations at once: one dropped from the CRL"

    staged = sorted(path.name for path in world.tls._TLS_DIR.rglob("*.tmp"))
    assert staged == [], ("a staged file left behind", staged)


# ---------------------------------------------------------------------------
# TG4 -- the launcher: the whole mTLS on a network address, or not at all
# ---------------------------------------------------------------------------
def test_tg4_a_network_address_is_served_with_the_whole_mtls_or_not_at_all(world, capsys):
    world.setup()
    world.device("phone")
    tls = world.tls

    for missing in (tls._SERVER_KEY_PATH, tls._SERVER_CERT_PATH, tls._CA_CERT_PATH):
        aside = missing.with_name(missing.name + ".aside")
        missing.rename(aside)
        try:
            assert world.launch("0.0.0.0") == 1, f"{missing.name} missing: served anyway"
        finally:
            aside.rename(missing)

    real = tls.get_tls_config

    def raising():
        raise OSError("cannot read")

    world.monkeypatch.setattr(tls, "get_tls_config", raising)
    assert world.launch("192.168.1.20") == 1, "a TLS configuration that raises: served anyway"
    world.monkeypatch.setattr(tls, "get_tls_config",
                              lambda: {key: value for key, value in real().items() if key in ("ssl_keyfile",
                                                                                            "ssl_certfile")})
    assert world.launch("::") == 1, "no client certificate asked: served anyway"
    world.monkeypatch.setattr(tls, "get_tls_config", real)

    assert world.runs == [], "uvicorn ran for a refused address"
    refused = [entry for entry in world.audit if entry.get("event_type") == "remote_bind_refused"]
    assert [entry.get("bound_address") for entry in refused] == ["0.0.0.0", "0.0.0.0", "0.0.0.0", "192.168.1.20",
                                                                 "::"], world.audit
    assert "without mTLS" in capsys.readouterr().out

    port, given = world.launch("0.0.0.0")
    assert given["host"] == "0.0.0.0" and given["ssl_cert_reqs"] == ssl.CERT_REQUIRED, given
    assert getattr(given.get("http"), "__name__", None) == "ClientCertGate", given
    assert given["ssl_ca_certs"] == str(tls._CA_CERT_PATH), given

    world.remote(False)
    port, given = world.launch("127.0.0.1")
    assert set(given) == {"host", "port"}, given

    world.remote(True)
    world.mode = "bulbe"
    port, given = world.launch("0.0.0.0")
    assert set(given) == {"host", "port"} and given["host"] == "127.0.0.1", given


# ---------------------------------------------------------------------------
# TG5 -- is_cert_serial_revoked fails closed
# ---------------------------------------------------------------------------
def test_tg5_a_serial_in_a_crl_that_does_not_load_reads_revoked(world):
    world.setup()
    world.device("phone")
    world.device("laptop")
    phone = x509.load_der_x509_certificate(world.der("phone")).serial_number
    laptop = x509.load_der_x509_certificate(world.der("laptop")).serial_number
    assert world.tls.revoke_client_cert("phone")["success"]
    assert (world.tls.is_cert_serial_revoked(phone), world.tls.is_cert_serial_revoked(laptop)) == (True, False)
    world.tls._CRL_PATH.write_bytes(b"garbage")
    assert world.tls.is_cert_serial_revoked(laptop) is True, "a CRL that does not load reads not revoked"
    world.tls._CRL_PATH.unlink()
    assert world.tls.is_cert_serial_revoked(laptop) is False

    # cryptography 41, the floor: CRL entries with a naive revocation_date and no revocation_date_utc.
    assert world.tls.revoke_client_cert("laptop")["success"]
    entries = list(x509.load_pem_x509_crl(world.tls._CRL_PATH.read_bytes()))
    old_style = [types.SimpleNamespace(serial_number=entry.serial_number,
                                       revocation_date=entry.revocation_date_utc.replace(tzinfo=None))
                 for entry in entries]
    world.monkeypatch.setattr(x509, "load_pem_x509_crl", lambda data: old_style)
    world.device("tablet")
    assert world.tls.is_cert_serial_revoked(phone) and world.tls.is_cert_serial_revoked(laptop)
    assert world.tls.client_cert_refusal(world.der("tablet")) is None, "cryptography 41's CRL refuses every device"
    assert world.tls._crl_entries() == {entry.serial_number: entry.revocation_date_utc for entry in entries}


# ---------------------------------------------------------------------------
# TG6 -- the texts
# ---------------------------------------------------------------------------
def test_tg6_the_texts_say_what_is_checked():
    security = " ".join((REPO / "SECURITY.md").read_text(encoding="utf-8").split())
    for said in ("Every connection is checked as soon as its handshake is done",
                 "and again every second while it is open",
                 "a revoked certificate, one made again for its device since",
                 "Revoking a device ends its open connections at once in the server that revokes it, and within a "
                 "second in another one",
                 "only with the whole of it",
                 "The CRL lists every certificate revoked, or replaced since this version",
                 "A revocation marks the device's record first, which refuses it",
                 "Revocations are one at a time across processes where `fcntl` is (not on Windows)"):
        assert said in security, said
    assert "with neither is not refused yet" not in security
    manager = (REPO / "opti_oignon" / "tls_manager.py").read_text(encoding="utf-8")
    assert "checked on every request" not in manager
    assert "Takes effect on the NEXT request" not in manager
    middleware = " ".join((REPO / "opti_oignon" / "api" / "security_mode_middleware.py").read_text(
        encoding="utf-8").split())
    assert "a request served over mTLS directly names no certificate here" in middleware
    assert "For a request that does present one (remote mTLS)" not in middleware


# ---------------------------------------------------------------------------
# TG7 -- a revocation made by another process ends the open connections
# ---------------------------------------------------------------------------
def test_tg7_another_process_revocation_ends_the_open_connections(world):
    world.setup()
    phone = world.device("phone")
    laptop = world.device("laptop")
    tablet = world.device("tablet")
    port = world.serve()
    phone_conn = world.connect(port, "phone")
    phone_ws = world.websocket(port, "phone")
    laptop_conn = world.connect(port, "laptop")
    tablet_conn = world.connect(port, "tablet")
    assert _ask(phone_conn, "/phone-1") == (200, b"ok")
    phone_ws.send("a")
    assert phone_ws.recv(timeout=5) == "a"
    assert _ask(laptop_conn, "/laptop-1") == (200, b"ok")
    assert _ask(tablet_conn, "/tablet-1") == (200, b"ok")
    time.sleep(1.5)  # the connections outlive their first check again
    assert _ask(phone_conn, "/phone-2") == (200, b"ok")

    open_phone = world.open_connections(phone)
    assert len(open_phone) == 2, open_phone
    with world.elsewhere():
        assert world.tls.revoke_client_cert("phone")["success"]
    _closing(open_phone)
    with pytest.raises(ConnectionClosed):
        phone_ws.recv(timeout=3)
    assert not isinstance(_ask(phone_conn, "/phone-3"), tuple), "a revocation elsewhere leaves the connection open"

    open_tablet = world.open_connections(tablet)
    with world.elsewhere():
        assert world.device("tablet") != tablet
    _closing(open_tablet)
    assert not isinstance(_ask(tablet_conn, "/tablet-2"), tuple), "a certificate made again elsewhere: old one open"
    assert _ask(laptop_conn, "/laptop-2") == (200, b"ok"), "another device's connection was ended"
    assert world.open_connections(laptop)
    assert world.seen == ["/ws", "/phone-1", "/laptop-1", "/tablet-1", "/phone-2", "/laptop-2"], world.seen

    # A connection the client closed, a websocket among them, leaves the open ones and is asked no more.
    laptop_ws = world.websocket(port, "laptop")
    laptop_ws.send("x")
    assert laptop_ws.recv(timeout=5) == "x"
    asked = []
    real = world.tls.client_cert_refusal

    def counted(cert_der):
        asked.append(cert_der)
        return real(cert_der)

    world.monkeypatch.setattr(world.tls, "client_cert_refusal", counted)
    laptop_ws.close()
    laptop_conn.close()
    deadline = time.monotonic() + 3
    while world.open_connections(laptop):
        assert time.monotonic() < deadline, "a closed connection stays among the open ones"
        time.sleep(0.05)
    asked.clear()
    time.sleep(2.5)
    assert world.der("laptop") not in asked, "a closed connection is still asked again"
