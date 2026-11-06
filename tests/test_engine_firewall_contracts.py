"""Contracts of the engine firewall: no contract reaches the machine's inference engine (``tests/_engine_firewall.py``).

  * EF1 -- INSTALLED FOR THE SESSION, OLLAMA'S PORT REFUSED: the conftest's
    firewall is in place before the contracts are collected; a connection
    to port 11434 is refused as an engine that is down refuses it, by name,
    and counted with the contract that asked.
  * EF2 -- THE ENGINE OLLAMA_HOST NAMES, REFUSED WHEREVER THE CLIENT RESOLVES
    IT: urllib, httpx and asyncio's loop meet the same refusal, with the
    address written as an IP or as a host name, and no connection reaches
    it; once the firewall is taken off, the address is reached again.
  * EF3 -- NOTHING ELSE REFUSED: a server of the contract's own on another
    port is reached with the firewall in place; ``connect_ex`` to the engine
    answers ECONNREFUSED and is counted.
  * EF4 -- THE SUMMARY NAMES WHO ASKED: the session's line counts the
    refusals and names the contracts; with none, it says so.
  * EF5 -- THE ENGINES READ AS THEIR CLIENTS READ THEM: OLLAMA_HOST as the
    engine's client parses it (a scheme's own port, a bare port, a bare
    host), and the hosts ``config/backends.yaml`` gives Ollama and the
    llama-server.
  * EF6 -- THE OLLAMA COMMAND REFUSED TOO: ``ollama`` on the PATH is the
    firewall's; it refuses and is counted with the contract, and the
    package's own call (a model the config does not know) never reaches
    the engine.
  * EF7 -- WHO ASKED, WHEREVER: a refusal outside any contract is counted
    apart, with the line of code that asked.
  * EF8 -- NEVER OFF UNSEEN: a conftest copied without the firewall says so
    in its summary; a firewall that fails to import fails the session; a
    command of the firewall's that does not answer as its own stops the
    install.
  * EF9 -- CHARGED WHERE ASKED: a command asked at collection is counted
    outside any contract, one asked by a contract whose teardown fails is
    that contract's, and the next contract is charged with neither.

The contracts that send a request aim at a server of their own, named by
``OLLAMA_HOST`` to a firewall of their own: a firewall that let it through
reaches the contract's server, never the maintainer's engine.
"""
from __future__ import annotations

import asyncio
import errno
import os
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _engine_firewall as ef  # noqa: E402

_TESTS = Path(__file__).resolve().parent
# The firewall in place when this file was collected (EF1).
_AT_COLLECTION = ef.ACTIVE

BUDGET_S = {
    "test_ef1_installed_for_the_session_ollamas_port_refused": 2.0,
    "test_ef2_the_engine_ollama_host_names_refused_through_any_client": 3.0,
    "test_ef3_nothing_else_refused": 2.0,
    "test_ef4_the_summary_names_who_asked": 1.0,
    "test_ef5_the_engines_read_as_their_clients_read_them": 1.0,
    "test_ef6_the_ollama_command_refused_too": 3.0,
    "test_ef7_who_asked_wherever": 1.0,
    "test_ef8_never_off_unseen": 4.0,
    "test_ef9_charged_where_asked": 3.0,
}


class Server:
    """A listening socket of the contract's own, counting the connections it accepts."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.accepted = 0
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                conn, _addr = self.sock.accept()
            except OSError:
                return
            self.accepted += 1
            conn.close()

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)    # wakes the accept a close alone leaves waiting
        except OSError:
            pass
        self.sock.close()
        self.thread.join(2)


def test_ef1_installed_for_the_session_ollamas_port_refused(request):
    """EF1: the conftest's firewall, in place at collection, refuses port 11434 by name, counted with this contract."""
    firewall = ef.ACTIVE
    owners = [plugin for plugin in request.config.pluginmanager.get_plugins()
              if isinstance(getattr(plugin, "_ENGINE", None), ef.EngineFirewall)]
    assert owners and firewall is owners[0]._ENGINE, "the conftest's engine firewall is the one installed"
    assert _AT_COLLECTION is firewall, "installed before the contracts are collected"
    before = len(firewall.refused)
    with pytest.raises(ef.EngineRefused, match="11434") as caught:
        socket.create_connection(("127.0.0.1", ef.ENGINE_PORT), timeout=1)
    assert isinstance(caught.value, ConnectionRefusedError) and caught.value.errno == errno.ECONNREFUSED
    assert len(firewall.refused) == before + 1 and firewall.refused[-1][0] == request.node.nodeid
    assert firewall.refuses(("192.0.2.7", ef.ENGINE_PORT)) and firewall.refuses(("::1", ef.ENGINE_PORT)), "any host"
    firewall.forget(request.node.nodeid)    # asked on purpose: the summary names the contracts that leaned on it


def test_ef2_the_engine_ollama_host_names_refused_through_any_client():
    """EF2: urllib and httpx refused before any connection reaches the named address; reached once taken off."""
    server = Server()
    firewall = ef.EngineFirewall({"OLLAMA_HOST": f"127.0.0.1:{server.port}"}, backends=None)
    try:
        firewall.install()
        url = f"http://127.0.0.1:{server.port}/api/tags"
        with pytest.raises(urllib.error.URLError) as caught:
            urllib.request.urlopen(url, timeout=2)
        assert isinstance(caught.value.reason, ef.EngineRefused), caught.value
        httpx = pytest.importorskip("httpx")
        with pytest.raises(httpx.ConnectError):
            httpx.get(url, timeout=2)

        async def opened():
            _reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            writer.close()

        with pytest.raises(ef.EngineRefused):
            asyncio.run(opened())
        assert server.accepted == 0 and len(firewall.refused) == 3
        firewall.uninstall()
        by_name = ef.EngineFirewall({"OLLAMA_HOST": f"localhost:{server.port}"}, backends=None)
        by_name.install()
        try:
            with pytest.raises(urllib.error.URLError) as caught:
                urllib.request.urlopen(f"http://localhost:{server.port}/api/tags", timeout=2)
            assert isinstance(caught.value.reason, ef.EngineRefused), caught.value
        finally:
            by_name.uninstall()
        assert server.accepted == 0
        by_ip = ef.EngineFirewall({"OLLAMA_HOST": f"127.0.0.1:{server.port}"}, backends=None)
        by_ip.install()
        raw = socket.socket()
        try:
            with pytest.raises(ef.EngineRefused):
                raw.connect(("localhost", server.port))     # resolved below Python, spelled otherwise
        finally:
            raw.close()
            by_ip.uninstall()
        assert server.accepted == 0
        socket.create_connection(("127.0.0.1", server.port), timeout=2).close()
        deadline = threading.Event()
        for _ in range(100):
            if server.accepted:
                break
            deadline.wait(0.02)
        assert server.accepted == 1, "once taken off, the address is reached"
    finally:
        firewall.uninstall()
        server.close()


def test_ef3_nothing_else_refused(request):
    """EF3: a server on another port is reached; connect_ex to the engine answers ECONNREFUSED, counted."""
    firewall = ef.ACTIVE
    assert firewall is not None
    server = Server()
    try:
        socket.create_connection(("127.0.0.1", server.port), timeout=2).close()
        before = len(firewall.refused)
        probe = socket.socket()
        try:
            assert probe.connect_ex(("127.0.0.1", ef.ENGINE_PORT)) == errno.ECONNREFUSED
        finally:
            probe.close()
        assert len(firewall.refused) == before + 1
        firewall.forget(request.node.nodeid)    # asked on purpose
    finally:
        server.close()


def test_ef4_the_summary_names_who_asked():
    """EF4: the line counts the refusals and names each contract once; with none, it says none was asked."""
    firewall = ef.EngineFirewall({}, backends=None)
    assert "no connection" in firewall.summary()
    firewall.refused = [("tests/a.py::t1", "127.0.0.1", 11434, "opti_oignon/x.py:1"),
                        ("tests/a.py::t1", "::1", 11434, "opti_oignon/x.py:1"),
                        ("tests/b.py::t2", "127.0.0.1", 11434, "opti_oignon/y.py:2")]
    line = firewall.summary()
    assert "3 connection(s)" in line and "2 contract(s)" in line
    assert "tests/a.py::t1" in line and "tests/b.py::t2" in line


def test_ef5_the_engines_read_as_their_clients_read_them(tmp_path):
    """EF5: OLLAMA_HOST as the engine's client parses it; the hosts backends.yaml gives Ollama and the llama-server."""
    for value, endpoint in (("127.0.0.1:18000", ("127.0.0.1", 18000)), ("http://GPU-box:11500/", ("gpu-box", 11500)),
                            ("gpu-box", ("gpu-box", 11434)), ("http://127.0.0.1", ("127.0.0.1", 80)),
                            ("https://box", ("box", 443)), (":56789", ("127.0.0.1", 56789))):
        assert ef.named_engines({"OLLAMA_HOST": value}, None) == [endpoint], value
    assert ef.named_engines({"OLLAMA_HOST": "http://box:port"}, None) == []
    assert ef.named_engines({}, None) == []
    backends = tmp_path / "backends.yaml"
    backends.write_text('ollama:\n  host: "http://127.0.0.1:18080"\nllama_server:\n  host: "http://127.0.0.1:18081"\n',
                        encoding="utf-8")
    assert ef.named_engines({}, backends) == [("127.0.0.1", 18080), ("127.0.0.1", 18081)]
    firewall = ef.EngineFirewall({"OLLAMA_HOST": "gpu-box:11500"}, backends=backends)
    assert firewall.refuses(("127.0.0.1", 18081)) and firewall.refuses(("10.0.0.9", 11434))
    assert not firewall.refuses(("127.0.0.1", 18082)) and not firewall.refuses("/tmp/socket")
    shipped = ef.named_engines({}, ef.BACKENDS)
    assert ("127.0.0.1", 11434) in shipped and ("127.0.0.1", 8080) in shipped


def test_ef6_the_ollama_command_refused_too(request):
    """EF6: the ollama command on the PATH is the firewall's: refused, counted with this contract; the package's
    own call for a model the config does not know reaches no engine."""
    from opti_oignon.context_manager import ContextManager

    firewall = ef.ACTIVE
    assert firewall is not None and firewall.command_dir is not None
    assert shutil.which("ollama") == str(firewall.command_dir / "ollama")
    before = len(firewall.commands_asked)
    done = subprocess.run(["ollama", "show", "no-such-model-238"], capture_output=True, text=True, timeout=20)
    assert done.returncode != 0 and "contracts never reach the inference engine" in done.stderr
    limits = ContextManager().get_model_limits("no-such-model-238:1b")
    firewall.settle()
    asked = firewall.commands_asked[before:]
    assert [args for _contract, args in asked][:1] == ["show no-such-model-238"], asked
    assert any("no-such-model-238:1b" in args for _contract, args in asked), asked
    assert {contract for contract, _args in asked} == {request.node.nodeid} and limits.source != "verified"
    firewall.forget(request.node.nodeid)    # asked on purpose


def test_ef7_who_asked_wherever(tmp_path):
    """EF7: a refusal outside a contract is counted apart, with the line that asked; contracts are named; a line
    of the tree reached through a link is named in the tree."""
    firewall = ef.EngineFirewall({}, backends=None)
    link = tmp_path / "linked"
    link.symlink_to(ef.TREE, target_is_directory=True)
    exec(compile("firewall._refusal(('127.0.0.1', 11434))", str(link / "opti_oignon" / "linked_238.py"), "exec"),
         {"firewall": firewall})
    assert firewall.refused[-1][3] == "opti_oignon/linked_238.py:1", firewall.refused[-1]
    firewall.refused.clear()
    firewall.install()
    try:
        firewall.current = None
        with pytest.raises(ef.EngineRefused):
            socket.create_connection(("127.0.0.1", ef.ENGINE_PORT), timeout=1)   # through the standard library
        firewall.current = "tests/a.py::t1"
        firewall._refusal(("127.0.0.1", ef.ENGINE_PORT))
    finally:
        firewall.uninstall()
    line = firewall.summary()
    assert "1 outside any contract" in line and "(tests/test_engine_firewall_contracts.py:" in line, line
    assert "1 contract(s)" in line and "tests/a.py::t1" in line, line



def _child_session(tmp_path, firewall_text, canary="def test_canary():\n    assert True\n"):
    """A session on one canary beside a copy of the conftest and the data firewall; the engine firewall's text."""
    shutil.copy2(_TESTS / "conftest.py", tmp_path / "conftest.py")
    shutil.copy2(_TESTS / "_data_firewall.py", tmp_path / "_data_firewall.py")
    if firewall_text is not None:
        (tmp_path / "_engine_firewall.py").write_text(firewall_text, encoding="utf-8")
    (tmp_path / "test_canary.py").write_text(canary, encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
                           str(tmp_path / "test_canary.py")],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


def test_ef8_never_off_unseen(tmp_path):
    """EF8: without the firewall beside it, the conftest says so; a firewall that fails to import fails the session;
    with it, a command a contract asks is charged to that contract."""
    (tmp_path / "present").mkdir()
    canary = ("import subprocess\n\n\ndef test_canary():\n"
              "    assert subprocess.run(['ollama', 'list'], capture_output=True).returncode != 0\n")
    present = _child_session(tmp_path / "present", (_TESTS / "_engine_firewall.py").read_text(encoding="utf-8"),
                             canary)
    assert present.returncode == 0 and "1 ollama command(s)" in present.stdout, present.stdout[-800:]
    assert "test_canary.py::test_canary  (ollama list)" in present.stdout, present.stdout[-800:]
    silent = ef.EngineFirewall({}, backends=None)
    good = ef._COMMAND
    ef._COMMAND = "#!/bin/sh\nexit 0\n"
    try:
        with pytest.raises(RuntimeError, match="does not answer"):
            silent.install()
    finally:
        ef._COMMAND = good
        silent.uninstall()
    assert ef.ACTIVE is not silent and os.environ["PATH"].split(os.pathsep)[0] == str(ef.ACTIVE.command_dir)
    (tmp_path / "absent").mkdir()
    absent = _child_session(tmp_path / "absent", None)
    assert absent.returncode == 0 and "engine firewall: not installed" in absent.stdout, absent.stdout[-800:]
    for case, text in (("name", "from os import no_such_name_238\n"), ("module", "import no_such_module_238\n")):
        (tmp_path / case).mkdir()
        broken = _child_session(tmp_path / case, text)
        assert broken.returncode != 0 and "1 passed" not in broken.stdout, (case, broken.stdout[-800:])
        assert f"no_such_{case}_238" in broken.stdout + broken.stderr, case



def test_ef9_charged_where_asked(tmp_path):
    """EF9: a command at collection is outside any contract; one asked by a contract whose teardown fails is that
    contract's; the next contract is charged with neither."""
    canary = ("import subprocess\n\nimport pytest\n\n"
              "subprocess.run(['ollama', 'ps'], capture_output=True)\n\n\n"
              "@pytest.fixture\ndef breaks():\n    yield\n    raise RuntimeError('a teardown that fails')\n\n\n"
              "def test_a(breaks):\n    subprocess.run(['ollama', 'show', 'm'], capture_output=True)\n\n\n"
              "def test_b():\n    assert True\n")
    done = _child_session(tmp_path, (_TESTS / "_engine_firewall.py").read_text(encoding="utf-8"), canary)
    out = done.stdout
    assert "2 ollama command(s)" in out and "1 outside any contract" in out, out[-1200:]
    assert "test_canary.py::test_a  (ollama show m)" in out and "outside any contract  (ollama ps)" in out, out[-1200:]
    assert "test_canary.py::test_b" not in out.split("engine firewall:")[-1], out[-1200:]
