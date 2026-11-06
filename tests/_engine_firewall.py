"""No contract reaches the machine's inference engine.

A contract that forgets its stand-in reaches the local engine through the
registry -- seen on 4 November, when a draft of the Mycelium's bridge asked
Ollama for a model that does not exist and got its 404 -- and its verdict
then depends on what the maintainer's engine serves and how loaded it is.
For the whole session, ``EngineFirewall`` refuses the engines the platform
reaches, by both ways it reaches them:

  * every socket connection to Ollama's port, 11434, on any host, and to
    each engine named elsewhere: ``OLLAMA_HOST``, read as the engine's
    client reads it (a scheme's own port, a bare port on 127.0.0.1, a bare
    host on 11434), and the hosts ``config/backends.yaml`` gives Ollama and
    the llama-server. A named host is compared with the address the client
    resolved it to, so ``localhost`` is refused as ``127.0.0.1`` is. The
    refusal is ``EngineRefused``, a ``ConnectionRefusedError`` -- what a
    contract that points at an engine that is down already expects --
    naming the address and why; ``connect_ex`` answers ``ECONNREFUSED`` as
    for a closed port;
  * the ``ollama`` command, which the platform runs to read a model's
    limits and to list the models: the firewall puts a command of its own
    first on the PATH, which answers as a failing command does and writes
    down what it was asked; it is asked once at install, and a command that
    does not answer so (a temporary directory mounted noexec would hand the
    PATH's next one, the real one) stops the session.

Each connection refused is counted with the contract running when it came
and the line of code that asked (the first frame in the tree, a tree
reached through a link included); each command, with its arguments and the
contract during which it was asked. The session's summary names those
contracts, and counts apart what came at collection, between contracts or
after the last. A contract named there leaned on the engine and needs its
stand-in. A thread that outlives its contract is charged to the contract
running when it connects; its line says where it came from. A session-wide
fixture's teardown is charged to the last contract. ``OLLAMA_HOST`` and
``backends.yaml`` are read once, when the conftest is imported: a contract
that names a stand-in engine of its own through ``OLLAMA_HOST`` reaches it
(Ollama's port stays refused on any host). Once the session ends, the PATH
is the shell's again.

It wraps ``connect`` and ``connect_ex`` of ``socket.socket`` itself, so a
suite that swaps ``socket.create_connection`` for a refusal of its own (some
do) keeps it, and a client built on Python's sockets -- urllib, httpx under
the engine's client, asyncio's own loop -- meets the same refusal (EF2).
Not covered: a connection made below Python's sockets (uvloop's loop, a
library's C code), a child process's sockets, a child started with a PATH
of its own (the sandbox's, a plugin's), which finds the real command, and
what reads the engine without connecting to it (``/proc/net/tcp``, the
environment of a running ``ollama serve``).

Local-only (the public distribution ships no tests).
"""

import errno
import ipaddress
import os
import shlex
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

ENGINE_PORT = 11434
TREE = Path(__file__).resolve().parents[1]
BACKENDS = TREE / "opti_oignon" / "config" / "backends.yaml"
# The firewall installed for the session (``EngineFirewall.install``), or None.
ACTIVE = None
_MESSAGE = "contracts never reach the inference engine"
# The firewall's ``ollama`` command: it writes down what it was asked and fails, naming why.
_COMMAND = ("#!/bin/sh\n"
            "printf '%s\\n' \"$*\" >> {log}\n"
            f"echo \"ollama: {_MESSAGE} (ollama $*): give the contract its stand-in\" >&2\n"
            "exit 1\n")


class EngineRefused(ConnectionRefusedError):
    """A connection to the machine's inference engine, refused during the contracts."""


def endpoint(raw):
    """``(host, port)`` a host setting names, read as the engine's client reads ``OLLAMA_HOST``; None if none."""
    raw = (raw or "").strip()
    if not raw:
        return None
    port = ENGINE_PORT
    scheme, _sep, hostport = raw.partition("://")
    if not hostport:
        scheme, hostport = "http", raw
    elif scheme == "http":
        port = 80
    elif scheme == "https":
        port = 443
    parts = urlsplit(f"{scheme}://{hostport}")
    try:
        named_port = parts.port
    except ValueError:
        return None
    return (parts.hostname or "127.0.0.1").lower(), named_port or port


def named_engines(environ=None, backends=BACKENDS):
    """The engines named outside the port rule: ``OLLAMA_HOST``, then the hosts ``backends`` gives; in order."""
    environ = os.environ if environ is None else environ
    raws = [environ.get("OLLAMA_HOST", "")]
    if backends is not None and Path(backends).exists():    # the tree's file is held to it by EF5
        import yaml

        settings = yaml.safe_load(Path(backends).read_text(encoding="utf-8")) or {}
        for name in ("ollama", "llama_server"):
            section = settings.get(name)
            if isinstance(section, dict) and isinstance(section.get("host"), str):
                raws.append(section["host"])
    named = []
    for raw in raws:
        found = endpoint(raw)
        if found is not None and found not in named:
            named.append(found)
    return named


def _asker():
    """``path:line`` of the code that asked: the first frame in the tree, outside this module."""
    here = os.path.normcase(os.path.realpath(__file__))
    root = str(TREE) + os.sep
    frame, fallback = sys._getframe(1), None
    while frame is not None:
        name = os.path.realpath(frame.f_code.co_filename)     # a tree reached through a link is the tree
        if os.path.normcase(name) != here:
            spot = f"{name}:{frame.f_lineno}"
            if name.startswith(root):
                return name[len(root):] + f":{frame.f_lineno}"
            fallback = fallback or spot
        frame = frame.f_back
    return fallback or "(unknown)"


class EngineFirewall:
    def __init__(self, environ=None, backends=BACKENDS):
        self.named = named_engines(environ, backends)
        self.refused = []           # (contract, host, port, line that asked), in order
        self.commands_asked = []    # (contract, the command's arguments), in order
        self.current = None         # the contract running, set by the conftest
        self.command_dir = None     # where the firewall's ``ollama`` command is, once installed
        self._addresses = {}        # named host -> the addresses it resolves to
        self._read = 0              # lines of the command's log already charged
        self._saved = None

    def _literal(self, host):
        try:
            ipaddress.ip_address(host.split("%", 1)[0])
            return True
        except ValueError:
            return False

    def _resolved(self, host):
        if host not in self._addresses:
            found = {host}
            try:
                for *_rest, sockaddr in socket.getaddrinfo(host, None):
                    found.add(str(sockaddr[0]).lower())
            except OSError:
                pass                # a name that does not resolve is reached by no client either
            self._addresses[host] = found
        return self._addresses[host]

    def refuses(self, address):
        """Whether ``address`` (a socket address) is an engine's."""
        if not isinstance(address, tuple) or len(address) < 2:
            return False        # a Unix socket's path, or anything else that is no network address
        host, port = str(address[0]).lower(), address[1]
        if port == ENGINE_PORT:
            return True
        hosts = [named_host for named_host, named_port in self.named if port == named_port]
        if not hosts:
            return False
        # A name handed to ``connect`` itself is resolved below Python: compare what both names resolve to.
        asked = {host} if self._literal(host) else self._resolved(host)
        return any(asked & self._resolved(named_host) for named_host in hosts)

    def _refusal(self, address):
        self.refused.append((self.current, str(address[0]), address[1], _asker()))
        return EngineRefused(errno.ECONNREFUSED, f"{_MESSAGE} ({address[0]}:{address[1]}): give the contract its "
                                                 "stand-in")

    def install(self):
        """Wrap ``socket.socket.connect`` and ``connect_ex``, put the ``ollama`` command first on the PATH; once."""
        if self._saved is not None:
            return
        cls = socket.socket
        connect, connect_ex = cls.connect, cls.connect_ex
        firewall = self

        def guarded_connect(sock, address):
            if firewall.refuses(address):
                raise firewall._refusal(address)
            return connect(sock, address)

        def guarded_connect_ex(sock, address):
            if firewall.refuses(address):
                firewall._refusal(address)
                return errno.ECONNREFUSED
            return connect_ex(sock, address)

        self.command_dir = Path(tempfile.mkdtemp(prefix="engine-firewall-"))
        log = self.command_dir / "asked.log"
        log.touch()
        command = self.command_dir / "ollama"
        command.write_text(_COMMAND.format(log=shlex.quote(str(log))), encoding="utf-8")
        command.chmod(0o755)
        # A command that cannot run (a temporary directory mounted noexec) would hand the PATH's next ``ollama``,
        # the real one: it is asked once, and must answer as the firewall's.
        import subprocess

        try:
            check = subprocess.run([str(command), "check"], capture_output=True, text=True, timeout=10)
            answered = check.returncode == 1 and _MESSAGE in check.stderr
        except OSError as exc:
            answered, check = False, exc
        log.write_text("", encoding="utf-8")
        if not answered:
            shutil.rmtree(self.command_dir, ignore_errors=True)
            self.command_dir = None
            raise RuntimeError(f"engine firewall: its ollama command does not answer ({check!r}); the session "
                               "would reach the real one")
        path = os.environ.get("PATH")
        self._saved = (cls, connect, connect_ex, path, ACTIVE)
        cls.connect, cls.connect_ex = guarded_connect, guarded_connect_ex
        os.environ["PATH"] = str(self.command_dir) + (os.pathsep + path if path else "")
        _set_active(self)

    def uninstall(self):
        if self._saved is None:
            return
        self.settle()
        cls, connect, connect_ex, path, before = self._saved
        cls.connect, cls.connect_ex = connect, connect_ex
        if path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = path
        shutil.rmtree(self.command_dir, ignore_errors=True)
        self._saved = None
        _set_active(before)

    def settle(self):
        """Charge what the ``ollama`` command was asked since the last call to the contract running now (None:
        outside any contract); the conftest calls it before each contract starts and once it ends."""
        if self.command_dir is None or not (self.command_dir / "asked.log").exists():
            return
        lines = (self.command_dir / "asked.log").read_text(encoding="utf-8", errors="replace").splitlines()
        self.commands_asked += [(self.current, line) for line in lines[self._read:]]
        self._read = len(lines)

    def forget(self, contract):
        """Drop what ``contract`` asked on purpose (the firewall's own contracts)."""
        self.refused = [entry for entry in self.refused if entry[0] != contract]
        self.commands_asked = [entry for entry in self.commands_asked if entry[0] != contract]

    def summary(self):
        if not self.refused and not self.commands_asked:
            return "engine firewall: no connection to the inference engine and no ollama command was asked"
        lines = {}
        for contract, _host, _port, asker in self.refused:
            lines.setdefault(contract, set()).add(asker)
        for contract, args in self.commands_asked:
            lines.setdefault(contract, set()).add(f"ollama {args}")
        contracts = sorted(name for name in lines if name is not None)
        head = (f"engine firewall: {len(self.refused)} connection(s) to the inference engine and "
                f"{len(self.commands_asked)} ollama command(s) refused, from {len(contracts)} contract(s), each to "
                "be given its stand-in")
        outside = [entry for entry in self.refused if entry[0] is None] + \
                  [entry for entry in self.commands_asked if entry[0] is None]
        if outside:
            head += f", and {len(outside)} outside any contract"
        out = [head + ":"]
        out += [f"  {name}  ({'; '.join(sorted(lines[name])[:3])})" for name in contracts[:60]]
        out += ["  ..."] if len(contracts) > 60 else []
        if outside:
            out.append(f"  outside any contract  ({'; '.join(sorted(lines[None])[:5])})")
        return "\n".join(out)


def _set_active(firewall):
    global ACTIVE
    ACTIVE = firewall
