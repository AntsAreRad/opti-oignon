"""Contracts for the outbound gate: every request that leaves for the web asks the gate first.

The platform reaches the web through a small number of paths, and each one
must ask the same web gate the page fetch asks: exactly Daily mode, the search
kill switch released. The front door, ``opti_oignon/egress.py``, has no rule of
its own; it asks the web gate and the local rule and names what they answer.

  * OG1 -- the front door delegates and fails closed: it returns what each
    gate returns, labelled, and an absent or broken gate is a refusal named
    ``unreadable``; its own text holds no sink, no mode literal and no
    import beyond the standard library.
  * OG2 -- the marketplace install is gated, destination-checked and
    configured: refused outside Daily and while the switch is engaged or
    unreadable, in its own words, before any request; every destination the
    page fetch refuses is refused; its configuration is read.
  * OG3 -- the index refresh is gated and destination-checked, and the
    listing route refreshes only when asked or allowed to.
  * OG4 -- the model downloader asks the gate before it starts, before every
    hop and after every block, raises the refusal to its route, and refuses
    every address the page fetch refuses.
  * OG13 -- the downloader writes only a ``.gguf`` inside a model directory.
  * OG14 -- importing the signature library never fetches: its package is
    imported only when its shared library loads, and nothing tells the user
    to import it.
  * OG15 -- every gate reads the mode on disk: a mode written by another
    process is the mode read next.

No request is ever made. Attempts are counted at two layers that the code as
it stands and the code behind the front door both reach: the global urllib
opener, and the socket module faked by the web-gates suite. The front door and
the web gate are loaded from their files when they exist; before they do, the
code under contract never imports them, and what it does is what the clauses
read.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import ast
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import types
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urljoin, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402
from test_search_gates_contracts import (  # noqa: E402
    _LINK_TABLE_NAMES,
    _TLS,
    _entries,
    _entry,
    _fake_ddgs,
    _http,
    _is_ip,
    _mode_module,
    _moved,
    _Net,
    _raise_runtime,
    _route_line,
)

BUDGET_S = {
    "test_og1_the_front_door_delegates_and_fails_closed": 2.0,
    "test_og2_the_marketplace_install_is_gated_destination_checked_and_configured": 2.0,
    "test_og3_the_index_refresh_is_gated_and_destination_checked": 2.0,
    "test_og4_the_downloader_asks_the_gate_at_every_step_and_refuses_what_the_page_fetch_refuses": 2.0,
    "test_og13_the_downloader_writes_only_a_gguf_inside_a_model_directory": 2.0,
    "test_og14_importing_the_signature_library_never_fetches": 2.0,
    "test_og15_every_gate_reads_the_mode_on_disk": 2.0,
}

_EG = "opti_oignon.egress"
_WG = "opti_oignon.web_gate"
_WS = "opti_oignon.web_search"
_RAG = "opti_oignon.rag_sanitizer"
_KS = "opti_oignon.search_killswitch"
_SM = "opti_oignon.security_mode"
_IB = "opti_oignon.inference_backend"
_PI = "opti_oignon.plugin_installer"
_PX = "opti_oignon.plugin_index"
_PM = "opti_oignon.plugin_manifest"
_MKT = "opti_oignon.api.routes_plugin_marketplace"
_MM = "opti_oignon.model_manager"
_SCH = "opti_oignon.api.schemas"
_RB = "opti_oignon.api.routes_backends"
_PQ = "opti_oignon.pqc_signatures"

_GUARD = REPO / ".github" / "scripts" / "egress_census_guard.py"
_ABSENT = object()

# Public addresses a scripted host is given, in order; none of them is this
# machine's or on one of its links.
_PUBLIC_POOL = (
    "93.184.215.14", "151.101.1.69", "104.16.132.229", "151.101.193.69", "104.18.32.7", "104.18.33.7",
    "93.184.216.34", "104.16.133.229",
)
_HELD = "151.101.65.69"
_ON_LINK_NET = "151.101.128.0/24"
_ON_LINK = "151.101.128.7"

# The mode sweep: every value that is not exactly Daily, a reader that raises
# included. It is fed through the recording security-mode stand-in, so the
# gate's own normalisation is what is exercised.
_OUTSIDE_DAILY = ("bulbe", "", "unknown", "Daily ", "Daily", "DAILY", None, RuntimeError("the mode cannot be read"))
_REFUSAL_STATES = tuple(("mode", value) for value in _OUTSIDE_DAILY) + (("kill_switch", None), ("unreadable", None))


def _texts(label):
    """The refusal texts the front door builds from a label, by refusal name."""
    return {
        "mode": f"{label} is refused outside Daily mode.",
        "kill_switch": f"{label} is refused while the kill switch is engaged.",
        "unreadable": f"{label} is refused: the kill switch cannot be read.",
    }


def _outcome(call):
    """(value, None) when the call returns, (None, exception) when it raises."""
    try:
        return call(), None
    except Exception as exc:  # the clause reads what was raised
        return None, exc


def _tree(path):
    """Every path under a directory, relative, sorted; empty when it does not exist."""
    if not path.exists():
        return []
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*"))


def _guard():
    loaded, restore = isolate(targets={"egress_census_guard": _GUARD})
    return loaded["egress_census_guard"], restore


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _called_names(node):
    """Every name a node calls, bare or as an attribute."""
    names = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            if isinstance(call.func, ast.Name):
                names.add(call.func.id)
            elif isinstance(call.func, ast.Attribute):
                names.add(call.func.attr)
    return names


# ---------------------------------------------------------------------------
# The world: the real web gate, page fetch and switch, a recording mode, the
# global opener and the socket module faked.
# ---------------------------------------------------------------------------
class _Response:
    """What the faked global opener hands back: status, headers, a body read in pieces."""

    def __init__(self, status, headers, body):
        self.status = status
        self.headers = dict(headers)
        self._body = io.BytesIO(body)

    def read(self, amt=-1):
        return self._body.read(-1 if amt is None else amt)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    """The global ``urllib.request.urlopen``, faked: every URL asked recorded, a scripted site served.

    Redirects are followed as urllib follows them, each hop recorded. A URL
    nothing serves raises as an unreachable host does.
    """

    def __init__(self):
        self.site = {}
        self.calls = []

    def __call__(self, request, *args, **kwargs):
        url = getattr(request, "full_url", request)
        for _hop in range(6):
            self.calls.append(url)
            answer = self.site.get(url)
            if answer is None:
                raise OSError(f"nothing answers at {url}")
            status, headers, body = answer
            if status in (301, 302, 303, 307, 308):
                url = urljoin(url, headers["Location"])
                continue
            return _Response(status, headers, body)
        raise OSError("too many redirects")


def _serve(world, url, body=b"", *, status=200, location=None, address=None):
    """Script one answer at both layers: the faked opener and the faked network."""
    headers = {"Content-Length": str(len(body))}
    if location is not None:
        headers["Location"] = location
    world.opener.site[url] = (status, headers, body)
    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if not _is_ip(host):
        address = address or world.addresses.setdefault(host, _PUBLIC_POOL[len(world.addresses)])
        world.net.dns[host.lower()] = [[address]]
    else:
        address = host
    answer = _moved(location, status) if location is not None else _http(status, tuple(headers.items()), body)
    world.net.answers.setdefault((address, port), []).append(answer)


def _reset(world):
    """Forget what the layers recorded, keep what they serve."""
    world.opener.calls.clear()
    world.net.lookups.clear()
    world.net.connections.clear()
    world.net.refused.clear()
    world.fetches.clear()
    world.gate.clear()


@contextmanager
def _world(tmp_path, under_test, *, seeded=None, local=(), links=None):
    """The code under contract behind the real gate, over the faked network.

    The web gate and the front door are loaded when their files exist. The
    security mode is a recording stand-in answering Daily; the switch is the
    real one, released, its record under ``tmp_path``. The web gate's
    answers to the front door are counted in ``gate``, and every call of the
    page fetch made through its module is recorded in ``fetches`` with what
    it returned. Temporary directories are made under ``scratch``.
    """
    sm = _mode_module("daily")
    targets = {_RAG: source("rag_sanitizer.py"), _KS: source("search_killswitch.py")}
    for name, path in ((_WG, source("web_gate.py")), (_WS, source("web_search.py")), (_EG, source("egress.py"))):
        if path.exists():
            targets[name] = path
    targets.update(under_test)
    tables = {}
    for table in _LINK_TABLE_NAMES:
        path = tmp_path / f"links.{table}"
        path.write_text((links or {}).get(table, ""), encoding="ascii")
        tables[table] = str(path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    world = SimpleNamespace(
        sm=sm, net=_Net(local=local), opener=_Opener(), gate=[], fetches=[], addresses={}, engaged=[],
        scratch=scratch, tmp=tmp_path,
    )
    saved = (urllib.request.urlopen, tempfile.tempdir)
    with _entries(ddgs=_fake_ddgs(())):
        loaded, restore = isolate(
            targets=targets, seeded={_SM: sm, **(seeded or {})}, packages=("opti_oignon.api",),
        )
        try:
            world.mods, world.ks, world.ws = loaded, loaded[_KS], loaded[_WS]
            world.ks._STATE_PATH = tmp_path / "switch.json"
            world.ws._LINK_TABLES = dict(tables)
            world.tls = _TLS()
            world.ws._tls_context = lambda: world.tls
            wg = loaded.get(_WG)
            if wg is not None:
                asked = wg.search_refusal

                def counted():
                    answer = asked()
                    world.gate.append(answer)
                    return answer

                wg.search_refusal = counted
            fetch = world.ws.fetch_page

            def recorded(url, **kwargs):
                entry = SimpleNamespace(url=url, kwargs=kwargs, page=None)
                world.fetches.append(entry)
                entry.page = fetch(url, **kwargs)
                return entry.page

            world.ws.fetch_page = recorded
            urllib.request.urlopen = world.opener
            tempfile.tempdir = str(scratch)
            with world.net.installed():
                yield world
        finally:
            urllib.request.urlopen, tempfile.tempdir = saved
            restore()


@contextmanager
def _refused(world, state):
    """Put the gate in one refusing state for the length of a block; yield the refusal's name."""
    kind, value = state
    if kind == "mode":
        saved = world.sm.mode
        world.sm.mode = value
        try:
            yield "mode"
        finally:
            world.sm.mode = saved
    elif kind == "kill_switch":
        released = world.ks.search_killswitch
        engaged = type(released)(state_path=world.tmp / f"engaged-{len(world.engaged)}.json")
        world.engaged.append(engaged)
        engaged.kill(reason="manual")
        world.ks.search_killswitch = engaged
        try:
            yield "kill_switch"
        finally:
            world.ks.search_killswitch = released
    else:
        raising = types.ModuleType(_KS)
        raising.search_killswitch = SimpleNamespace(is_killed=_raise_runtime, is_enabled=_raise_runtime)
        with _entry(_KS, raising):
            yield "unreadable"


# ---------------------------------------------------------------------------
# OG1 -- the front door delegates and fails closed
# ---------------------------------------------------------------------------
def _gate_stand_ins(calls):
    """A web gate and a local rule that record every question and answer as told."""
    wg = types.ModuleType(_WG)
    wg.answer = None

    def search_refusal():
        calls.append("web")
        return wg.answer

    wg.search_refusal = search_refusal
    ib = types.ModuleType(_IB)
    ib.answer = None

    def local_refusal(label, endpoint=None, where=""):
        calls.append(("local", label, endpoint))
        return ib.answer

    ib.local_refusal = local_refusal
    return wg, ib


class _Raising:
    """A finder first on the meta path: importing one module name raises the given error."""

    def __init__(self, name, exc):
        self.name, self.exc = name, exc

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            raise self.exc
        return None


def _mode_literals(tree):
    """Every string constant that is a mode name: a comparison the front door must not hold."""
    return [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value.strip().lower() in ("daily", "bulbe")
    ]


def _foreign_imports(tree):
    """Every module-scope import that is not the standard library, relative imports included."""
    found = []
    pending = list(tree.body)
    while pending:
        node = pending.pop(0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] not in sys.stdlib_module_names]
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if node.level or top not in sys.stdlib_module_names:
                found.append("." * node.level + (node.module or ""))
        pending += list(ast.iter_child_nodes(node))
    return found


def test_og1_the_front_door_delegates_and_fails_closed(tmp_path):
    calls = []
    wg, ib = _gate_stand_ins(calls)
    loaded, restore = isolate(targets={_EG: source("egress.py")}, seeded={_WG: wg, _IB: ib})
    try:
        eg = loaded[_EG]

        # c1 -- the web gate's answer, whatever it is, labelled; require_web
        # raises it by name and returns when the gate is open.
        for answer in (None, "mode", "kill_switch", "unreadable"):
            wg.answer = answer
            del calls[:]
            got = eg.web_refusal()
            labelled = eg.web_refusal("Plugin install")
            assert calls == ["web", "web"], (answer, calls)
            if answer is None:
                assert got is None and labelled is None, (got, labelled)
            else:
                assert got == (answer, _texts("Web request")[answer]), got
                assert labelled == (answer, _texts("Plugin install")[answer]), labelled
            _value, exc = _outcome(lambda: eg.require_web("Plugin install"))
            if answer is None:
                assert exc is None, exc
            else:
                assert isinstance(exc, eg.EgressRefused) and isinstance(exc, RuntimeError), repr(exc)
                assert (exc.gate, exc.refusal, str(exc)) == ("web", answer, _texts("Plugin install")[answer])

        # c4 -- require_delegated asks both, the local rule first; either
        # refusal raises, named by its gate.
        wg.answer, ib.answer = None, None
        del calls[:]
        assert eg.require_delegated("Model pull", "http://127.0.0.1:11434") is None
        assert calls == [("local", "Model pull", "http://127.0.0.1:11434"), "web"], calls
        ib.answer = "Model pull is refused: the endpoint is not on this machine."
        _value, exc = _outcome(lambda: eg.require_delegated("Model pull", "https://ollama.example"))
        assert isinstance(exc, eg.EgressRefused) and exc.gate == "local", repr(exc)
        assert str(exc) == ib.answer, str(exc)
        ib.answer, wg.answer = None, "kill_switch"
        _value, exc = _outcome(lambda: eg.require_delegated("Model pull", "http://127.0.0.1:11434"))
        assert isinstance(exc, eg.EgressRefused) and (exc.gate, exc.refusal) == ("web", "kill_switch"), repr(exc)
        assert str(exc) == _texts("Model pull")["kill_switch"], str(exc)

        # c3 (delegation) -- the local rule's answer, as it gave it.
        wg.answer = None
        for answer in (None, "Model pull is refused: a proxy would carry the request."):
            ib.answer = answer
            del calls[:]
            assert eg.local_refusal("Model pull", "http://127.0.0.1:11434") == answer
            assert calls == [("local", "Model pull", "http://127.0.0.1:11434")], calls
            _value, exc = _outcome(lambda: eg.require_local("Model pull", "http://127.0.0.1:11434"))
            if answer is None:
                assert exc is None, exc
            else:
                assert isinstance(exc, eg.EgressRefused) and exc.gate == "local" and str(exc) == answer, repr(exc)
    finally:
        restore()

    # c2 and c3 -- a gate module that is absent, or whose import fails, is a
    # refusal named unreadable.
    for gate, blocked_name in (("web", _WG), ("local", _IB)):
        for how in ("absent", "import fails"):
            others = {_WG: wg, _IB: ib}
            del others[blocked_name]
            loaded, restore = isolate(
                targets={_EG: source("egress.py")},
                seeded=others,
                blocked=(blocked_name,) if how == "absent" else (),
            )
            try:
                if how == "import fails":
                    sys.modules.pop(blocked_name, None)
                    sys.meta_path.insert(0, _Raising(blocked_name, ModuleNotFoundError("No module named 'yaml'", name="yaml")))
                eg = loaded[_EG]
                wg.answer, ib.answer = None, None
                if gate == "web":
                    assert eg.web_refusal("Plugin install") == ("unreadable", _texts("Plugin install")["unreadable"]), how
                    _value, exc = _outcome(lambda: eg.require_web("Plugin install"))
                    assert isinstance(exc, eg.EgressRefused), (how, repr(exc))
                    assert (exc.gate, exc.refusal) == ("web", "unreadable"), (how, exc.gate, exc.refusal)
                else:
                    text = eg.local_refusal("Model pull", "http://127.0.0.1:11434")
                    assert isinstance(text, str) and text, (how, text)
                    _value, exc = _outcome(lambda: eg.require_local("Model pull", "http://127.0.0.1:11434"))
                    assert isinstance(exc, eg.EgressRefused), (how, repr(exc))
                    assert (exc.gate, exc.refusal, str(exc)) == ("local", "unreadable", text), (how, repr(exc))
                    _value, exc = _outcome(lambda: eg.require_delegated("Model pull", "http://127.0.0.1:11434"))
                    assert isinstance(exc, eg.EgressRefused) and exc.gate == "local", (how, repr(exc))
            finally:
                restore()

    # c5 -- the front door's own text: no sink, no mode literal, the standard
    # library only at module scope. Each probe is shown able to find one.
    text = source("egress.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    guard, restore = _guard()
    try:
        assert guard.count_sinks("import urllib.request\n\ndef f(u):\n    return urllib.request.urlopen(u)\n") >= 1
        assert guard.count_sinks(text) == 0, guard.sink_sites(text)
    finally:
        restore()
    assert _mode_literals(ast.parse('def f(mode):\n    return mode == "daily"\n')) == ["daily"]
    assert _mode_literals(tree) == [], _mode_literals(tree)
    assert _foreign_imports(ast.parse("import os\nimport yaml\nfrom .web_gate import search_refusal\n")) == [
        "yaml", ".web_gate",
    ]
    assert _foreign_imports(tree) == [], _foreign_imports(tree)


# ---------------------------------------------------------------------------
# OG2 -- the marketplace install is gated, destination-checked and configured
# ---------------------------------------------------------------------------
_PLUGIN = "onion-tool"
_PAYLOAD = bytes(range(256)) * 16
_MIB = 1024 * 1024
_INSTALL_DEFAULTS = (
    "install:\n  allow_remote_install: true\n  max_download_size_mb: 50\n"
    "  require_hash: false\n  timeout_s: 120\n"
)


def _manifest():
    return (
        f"name: {_PLUGIN}\nversion: 1.0.0\nauthor: Tester\n"
        "description: A plugin served by the test.\nentry_point: plugin.py\n"
    )


def _zip_archive(padding=0):
    """A plugin archive whose payload is bytes no text decoding keeps intact."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr(f"{_PLUGIN}/manifest.yaml", _manifest())
        archive.writestr(f"{_PLUGIN}/plugin.py", "def setup():\n    return None\n")
        archive.writestr(f"{_PLUGIN}/payload.bin", _PAYLOAD)
        if padding:
            archive.writestr(f"{_PLUGIN}/padding.bin", bytes(padding))
    return buffer.getvalue()


def _tar_archive():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in (
            (f"{_PLUGIN}/manifest.yaml", _manifest().encode("utf-8")),
            (f"{_PLUGIN}/plugin.py", b"def setup():\n    return None\n"),
            (f"{_PLUGIN}/payload.bin", _PAYLOAD),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _manifest_module():
    """The manifest model, standing in: any mapping with the required fields validates."""
    mod = types.ModuleType(_PM)

    class PluginManifest:
        @classmethod
        def from_dict(cls, data):
            return SimpleNamespace(**data)

    mod.PluginManifest = PluginManifest
    return mod


def _installer(world, config=_INSTALL_DEFAULTS):
    """A fresh installer, its configuration read from a file the clause wrote."""
    pi = world.mods[_PI]
    path = world.tmp / f"marketplace-{len(world.configs)}.yaml"
    path.write_text(config, encoding="utf-8")
    world.configs.append(path)
    pi._CONFIG_PATH = path
    return pi.RemotePluginInstaller(plugins_dir=world.plugins)


@contextmanager
def _plugin_world(tmp_path):
    with _world(tmp_path, {_PI: source("plugin_installer.py")}, seeded={_PM: _manifest_module()}) as world:
        world.plugins = tmp_path / "plugins"
        world.configs = []
        yield world


# (URL, what the refusal names, the address no connection may reach)
_REFUSED_DESTINATIONS = (
    ("file:///etc/hostname", "only http and https are fetched", None),
    ("ftp://example.org/p.zip", "only http and https are fetched", None),
    ("data:application/zip;base64,UEs=", "only http and https are fetched", None),
    ("http://127.0.0.1/p.zip", "(loopback)", "127.0.0.1"),
    ("http://[::1]/p.zip", "(loopback)", "::1"),
    ("http://10.0.0.1/p.zip", "(private)", "10.0.0.1"),
    ("http://100.64.0.1/p.zip", "(shared address space)", "100.64.0.1"),
    ("http://[::ffff:127.0.0.1]/p.zip", "(loopback, an IPv4 address inside IPv6)", "::ffff:127.0.0.1"),
    ("https://lan.example.org/p.zip", "(private)", "192.168.1.2"),
    ("https://moved.example.org/p.zip", "Refused at redirect 1", "127.0.0.1"),
)


def test_og2_the_marketplace_install_is_gated_destination_checked_and_configured(tmp_path):
    archive = _zip_archive()
    with _plugin_world(tmp_path) as w:
        installer = _installer(w)

        # c1 -- outside Daily, switch engaged, switch unreadable: refused in
        # the front door's words, before any request, nothing written.
        url = "https://example.org/p.zip"
        for state in _REFUSAL_STATES:
            with _refused(w, state) as refusal:
                _reset(w)
                before = _tree(w.plugins)
                result = installer.install_from_url(url)
                assert w.opener.calls == [] and w.net.quiet(), (state, w.opener.calls, w.net.lookups, w.net.connections)
                assert result["success"] is False, (state, result)
                assert result["error"] == _texts("Plugin install")[refusal], (state, result["error"])
                assert _tree(w.plugins) == before and _tree(w.scratch) == [], (state, _tree(w.scratch))
        _reset(w)
        installer.install_from_url(url)
        assert len(w.opener.calls) + len(w.net.lookups) >= 1, "in Daily the install reaches the network layer"

        # c2 -- in Daily, every destination the page fetch refuses is refused,
        # named, with no connection to the refused address and nothing written.
        for literal in ("http://127.0.0.1/p.zip", "http://[::1]/p.zip", "http://10.0.0.1/p.zip",
                        "http://100.64.0.1/p.zip", "http://[::ffff:127.0.0.1]/p.zip"):
            _serve(w, literal, archive)
        _serve(w, "https://lan.example.org/p.zip", archive, address="192.168.1.2")
        _serve(w, "https://moved.example.org/p.zip", status=302, location="http://127.0.0.1/")
        _serve(w, "http://127.0.0.1/", archive)
        for target, why, address in _REFUSED_DESTINATIONS:
            _reset(w)
            before = _tree(w.plugins)
            result = installer.install_from_url(target)
            reached = {host for host, _port in w.net.connections}
            assert w.opener.calls == [] and address not in reached, (target, w.opener.calls, reached)
            if address is None:
                assert w.net.quiet(), (target, w.net.lookups, w.net.connections)
            assert result["success"] is False, (target, result)
            assert why in (result["error"] or ""), (target, result["error"])
            assert _tree(w.plugins) == before and _tree(w.scratch) == [], target

        # c4 -- the GitHub rewrite reads the host, not a substring of the URL.
        assert installer._normalize_url("https://github.com/u/r") == "https://github.com/u/r/archive/refs/heads/main.zip"
        for look_alike in ("https://evil.example/github.com/x", "https://github.com.evil.example/u/r"):
            assert installer._normalize_url(look_alike) == look_alike, look_alike

        # c5 -- witness, Daily: a served archive installs, it reached the
        # installer as bytes, and the gate was asked.
        _reset(w)
        _serve(w, "https://plugins.example.org/p.zip", archive)
        result = installer.install_from_url("https://plugins.example.org/p.zip")
        assert result["success"] is True and result["name"] == _PLUGIN, result
        assert (w.plugins / _PLUGIN / "payload.bin").read_bytes() == _PAYLOAD
        assert len(w.gate) >= 1, "the install asked the web gate"
        assert len(w.fetches) == 1, [f.url for f in w.fetches]
        assert w.fetches[0].page.text == "" and w.fetches[0].page.body == archive, "the archive was decoded as text"

        # c7 -- a release asset redirected to a signed URL keeps its name.
        _reset(w)
        signed = "https://cdn.example.org/x?X-Sig=" + "a" * 400
        _serve(w, "https://example.org/r/v1/p.tar.gz", status=302, location=signed)
        _serve(w, signed, _tar_archive())
        result = installer.install_from_url("https://example.org/r/v1/p.tar.gz")
        assert result["success"] is True, result
        assert (w.plugins / _PLUGIN / "payload.bin").read_bytes() == _PAYLOAD

        # c3 -- the configuration is read, key by key.
        _serve(w, "https://big.example.org/p.zip", _zip_archive(padding=2 * _MIB))
        _reset(w)
        before = _tree(w.plugins)
        result = _installer(w, "install:\n  allow_remote_install: false\n").install_from_url(
            "https://plugins.example.org/p.zip")
        assert result["success"] is False and w.opener.calls == [] and w.net.quiet(), result
        assert "allow_remote_install" in (result["error"] or ""), result["error"]
        assert w.gate == [], "remote install off is refused before the gate is asked"
        result = _installer(w, "install:\n  max_download_size_mb: 1\n").install_from_url(
            "https://big.example.org/p.zip")
        assert result["success"] is False and "too large" in (result["error"] or "").lower(), result
        assert _tree(w.plugins) == before
        _reset(w)
        result = _installer(w, "install:\n  require_hash: true\n").install_from_url(
            "https://plugins.example.org/p.zip")
        assert result["success"] is False and w.opener.calls == [] and w.net.quiet(), result
        low = (result["error"] or "").lower()
        assert "hash" in low or "sha256" in low or "digest" in low, result["error"]
        _reset(w)
        _installer(w, "install:\n  timeout_s: 7\n").install_from_url("https://plugins.example.org/p.zip")
        assert [f.kwargs.get("timeout") for f in w.fetches] == [7], [f.kwargs for f in w.fetches]

    # c6 -- the installer holds no sink and fetches through the page fetch.
    text = source("plugin_installer.py").read_text(encoding="utf-8")
    guard, restore = _guard()
    try:
        assert guard.count_sinks("import urllib.request\n\ndef f(u):\n    return urllib.request.urlopen(u)\n") >= 1
        assert guard.count_sinks(text) == 0, guard.sink_sites(text)
    finally:
        restore()
    assert "fetch_page" in _called_names(ast.parse(text))


# ---------------------------------------------------------------------------
# OG3 -- the index refresh is gated and destination-checked
# ---------------------------------------------------------------------------
_INDEX_JSON = json.dumps(
    {"plugins": [{"name": "remote-onion", "version": "1.0.0", "description": "Served by the test."}]}
).encode("utf-8")


@contextmanager
def _index_world(tmp_path):
    under = {_PX: source("plugin_index.py"), _MKT: source("api", "routes_plugin_marketplace.py")}
    with _world(tmp_path, under) as world:
        world.indexes = []
        yield world


def _index(world, url, config="index:\n  auto_refresh: true\n"):
    """A fresh index under the marketplace configuration a clause wrote; the route serves it."""
    px, mkt = world.mods[_PX], world.mods[_MKT]
    n = len(world.indexes)
    path = world.tmp / f"index-config-{n}.yaml"
    path.write_text(config, encoding="utf-8")
    px._CONFIG_PATH = path
    mkt._CONFIG_PATH = path
    index = px.PluginIndex(db_path=world.tmp / f"index-{n}.db", index_url=url)
    world.indexes.append(index)
    mkt._get_index = lambda: index
    return index


def _listing(world, refresh):
    response = world.mods[_MKT].browse_marketplace(sort_by="name", limit=50, offset=0, refresh=refresh)
    return [plugin.name for plugin in response.plugins]


def test_og3_the_index_refresh_is_gated_and_destination_checked(tmp_path):
    with _index_world(tmp_path) as w:
        px = w.mods[_PX]

        # c1 -- each refusal: nothing fetched, the refusal kept in the front
        # door's words, the cached listing served.
        index = _index(w, "https://index.example.org/index.json")
        index.upsert(px.IndexEntry.from_dict({"name": "cached-onion", "version": "1.0.0"}))
        for state in _REFUSAL_STATES:
            with _refused(w, state) as refusal:
                _reset(w)
                assert index.refresh_from_remote(force=True) == 0, state
                assert w.opener.calls == [] and w.net.quiet(), (state, w.opener.calls, w.net.lookups)
                assert getattr(index, "last_refusal", None) == _texts("Plugin index refresh")[refusal], state
                assert _listing(w, refresh=True) == ["cached-onion"], state
                assert w.opener.calls == [] and w.net.quiet(), state

        # c2 -- in Daily, an index on this machine is refused by the
        # destination rule, with no connection.
        _reset(w)
        _serve(w, "http://127.0.0.1/index.json", _INDEX_JSON)
        loopback = _index(w, "http://127.0.0.1/index.json")
        assert loopback.refresh_from_remote(force=True) == 0
        assert w.opener.calls == [] and w.net.connections == [], (w.opener.calls, w.net.connections)
        assert loopback.get("remote-onion") is None

        # c3 -- witness: a public index is fetched once and its entries kept.
        _reset(w)
        _serve(w, "https://plugins.example.org/index.json", _INDEX_JSON)
        public = _index(w, "https://plugins.example.org/index.json")
        assert public.refresh_from_remote(force=True) == 1
        assert public.get("remote-onion") is not None
        assert len(w.opener.calls) + len(w.net.connections) == 1, (w.opener.calls, w.net.connections)

        # c4 -- the route: a stale index is refreshed on its own only when
        # the configuration allows it, and always when asked.
        quiet = _index(w, "https://plugins.example.org/index.json", "index:\n  auto_refresh: false\n")
        asked = []
        quiet.refresh_from_remote = lambda force=False: asked.append(force) or 0
        assert quiet.is_stale
        _listing(w, refresh=False)
        assert asked == [], asked
        _listing(w, refresh=True)
        assert len(asked) == 1, asked
        eager = _index(w, "https://plugins.example.org/index.json", "index:\n  auto_refresh: true\n")
        eager_asked = []
        eager.refresh_from_remote = lambda force=False: eager_asked.append(force) or 0
        _listing(w, refresh=False)
        assert len(eager_asked) == 1, "a stale index refreshes on its own when allowed"


# ---------------------------------------------------------------------------
# OG4 and OG13 -- the model downloader
# ---------------------------------------------------------------------------
_MODEL_HOST = "models.example.org"
_MODEL_URL = f"https://{_MODEL_HOST}/m.gguf"
_GGUF = b"GGUF" + bytes(28)


class _Answer:
    """One hop's answer to the downloader's pinned opener; ``after_read`` runs after each read."""

    def __init__(self, status, headers=None, blocks=(), after_read=None):
        self.status = status
        self.headers = dict(headers or {})
        self._blocks = list(blocks)
        self._after = after_read
        self.reads = 0

    def read(self, amt=-1):
        if not self._blocks:
            return b""
        block = self._blocks.pop(0)
        self.reads += 1
        if self._after is not None:
            self._after(self.reads)
        return block

    def close(self):
        pass


def _small():
    return _Answer(200, {"Content-Length": str(len(_GGUF))}, [_GGUF])


class _Hops:
    """The downloader's pinned opener, faked: each hop's URL and address recorded, a script replayed."""

    def __init__(self, *script, default=None):
        self.script = list(script)
        self.default = default
        self.calls = []

    def __call__(self, url, pinned_ip, headers, timeout):
        self.calls.append((url, pinned_ip))
        if self.script:
            step = self.script.pop(0)
        elif self.default is not None:
            step = self.default
        else:
            raise OSError(f"nothing scripted answers {url}")
        return step() if callable(step) else step


def _script(world, *steps, default=None):
    world.hops = _Hops(*steps, default=default)
    world.mods[_MM]._default_pinned_opener = world.hops
    return world.hops


@contextmanager
def _model_world(tmp_path):
    under = {_MM: source("model_manager.py"), _SCH: source("api", "schemas.py"), _RB: source("api", "routes_backends.py")}
    with _world(tmp_path, under, local=(_HELD,), links={"routes4": _route_line(_ON_LINK_NET)}) as world:
        world.models = tmp_path / "models"
        world.models.mkdir()
        world.manager = world.mods[_MM].ModelManager(
            model_dirs=[str(world.models)], default_dir=str(world.models),
        )
        world.net.dns[_MODEL_HOST] = [[_PUBLIC_POOL[0]]]
        _script(world, default=_small)
        yield world


def _route(world, manager, **fields):
    """POST /api/backends/gguf/download through its handler: (status, detail)."""
    rb, sch = world.mods[_RB], world.mods[_SCH]
    rb.MODEL_MANAGER_AVAILABLE = True
    rb.get_model_manager = lambda: manager
    try:
        rb.download_gguf_model(sch.GGUFDownloadRequest(**fields))
    except Exception as exc:  # an HTTPException carries the answer
        return getattr(exc, "status_code", None), getattr(exc, "detail", repr(exc))
    return 200, None


def _leftovers(world):
    return [name for name in _tree(world.models) if name.endswith((".part", ".gguf"))]


class _Refusing:
    """A manager whose download raises the refusal it was given."""

    def __init__(self, exc):
        self.exc = exc

    def download_model(self, **kwargs):
        raise self.exc


# (host, address, the class the refusal names)
_REFUSED_BY_CLASS = (
    ("shared.example.net", "100.64.0.1", "shared address space"),
    ("sitelocal.example.net", "fec0::1", "site-local"),
    ("here.example.net", _HELD, "this machine"),
    ("onlink.example.net", _ON_LINK, "local network, on the link"),
)
# Refused as they stand too; their refusal is not asked to name the class.
_REFUSED_AS_IT_STANDS = (("mapped.example.net", "::ffff:127.0.0.1"), ("nat64.example.net", "64:ff9b::a00:1"))


def test_og4_the_downloader_asks_the_gate_at_every_step_and_refuses_what_the_page_fetch_refuses(tmp_path):
    with _model_world(tmp_path) as w:
        mm = w.mods[_MM]

        # c1 -- each refusal: raised before any resolution, nothing written;
        # the route answers 403 with the refusal's text.
        for state in _REFUSAL_STATES:
            with _refused(w, state) as refusal:
                _reset(w)
                hops = _script(w, default=_small)
                value, exc = _outcome(lambda: w.manager.download_model(_MODEL_URL))
                assert exc is not None, (state, f"download_model returned {value!r}; nothing refused it")
                assert w.net.lookups == [] and hops.calls == [], (state, w.net.lookups, hops.calls)
                assert _leftovers(w) == [], (state, _leftovers(w))
                eg = w.mods.get(_EG)
                assert eg is not None, "absence: opti_oignon/egress.py is not written"
                text = _texts("Model download")[refusal]
                assert isinstance(exc, eg.EgressRefused) and (exc.gate, exc.refusal, str(exc)) == ("web", refusal, text), (
                    state, repr(exc))
                refused = eg.EgressRefused("web", refusal, text)
                assert _route(w, _Refusing(refused), url=_MODEL_URL) == (403, text), state

        # c2 -- the mode turns Bulbe while hop 0 answers a redirect: refused
        # before the next hop's host is resolved.
        _reset(w)
        w.net.dns["cdn.example.net"] = [[_PUBLIC_POOL[1]]]

        def redirect():
            w.sm.mode = "bulbe"
            return _Answer(302, {"Location": "https://cdn.example.net/m.gguf"})

        hops = _script(w, redirect, default=_small)
        value, exc = _outcome(lambda: w.manager.download_model(_MODEL_URL, filename="c2.gguf"))
        w.sm.mode = "daily"
        assert _MODEL_HOST in w.net.lookups, "the first hop was resolved"
        assert "cdn.example.net" not in w.net.lookups and len(hops.calls) == 1, (w.net.lookups, hops.calls)
        assert exc is not None and type(exc).__name__ == "EgressRefused", (value, exc)
        assert _leftovers(w) == [], _leftovers(w)

        # c3 -- the mode turns Bulbe after the first block: the refusal is
        # raised, not returned; the partial file is gone; the route says 403.
        def turning(reads):
            if reads == 1:
                w.sm.mode = "bulbe"

        def three_blocks():
            return _Answer(200, {"Content-Length": str(3 * _MIB)}, [bytes(_MIB)] * 3, after_read=turning)

        _reset(w)
        _script(w, three_blocks)
        value, exc = _outcome(lambda: w.manager.download_model(_MODEL_URL, filename="c3.gguf"))
        w.sm.mode = "daily"
        assert exc is not None, f"download_model returned {value!r} after the mode turned"
        assert type(exc).__name__ == "EgressRefused", repr(exc)
        assert _leftovers(w) == [], _leftovers(w)
        _script(w, three_blocks)
        status = _route(w, w.manager, url=_MODEL_URL, filename="c3-route.gguf")
        w.sm.mode = "daily"
        assert status == (403, _texts("Model download")["mode"]), status
        assert _leftovers(w) == [], _leftovers(w)

        # c4 -- the addresses the page fetch refuses, refused by name.
        for host, address, name in _REFUSED_BY_CLASS:
            w.net.dns[host] = [[address]]
            value, exc = _outcome(lambda: mm._resolve_validated_ips(host, 443, resolver=w.net.getaddrinfo))
            assert isinstance(exc, ValueError), (address, value, exc)
            assert name in str(exc), (address, str(exc))
        for host, address in _REFUSED_AS_IT_STANDS:
            w.net.dns[host] = [[address]]
            value, exc = _outcome(lambda: mm._resolve_validated_ips(host, 443, resolver=w.net.getaddrinfo))
            assert isinstance(exc, ValueError), (address, value, exc)
        w.net.dns["one.example.net"] = [["1.1.1.1"]]
        assert mm._resolve_validated_ips("one.example.net", 443, resolver=w.net.getaddrinfo) == ["1.1.1.1"]

    # c5 -- one implementation: the page fetch's address check asks the
    # shared refusal, and never the two classes directly.
    assert _called_names(ast.parse("def _checked_addresses():\n    return _address_class(x)\n")) == {"_address_class"}
    checked = _function(ast.parse(source("web_search.py").read_text(encoding="utf-8")), "_checked_addresses")
    assert checked is not None
    called = _called_names(checked)
    assert "address_refusal" in called, sorted(called)
    assert not called & {"_address_class", "_local_class"}, sorted(called)


def test_og13_the_downloader_writes_only_a_gguf_inside_a_model_directory(tmp_path):
    with _model_world(tmp_path) as w:
        models = w.models
        outside = tmp_path / "outside"
        outside.mkdir()

        # c1 -- the name is reduced to its last segment and ends in .gguf.
        for given, saved in (
            ("../x1.gguf", "x1.gguf"),
            (str(outside / "x2.gguf"), "x2.gguf"),
            ("a/b.gguf", "b.gguf"),
            ("x.pth", "x.pth.gguf"),
        ):
            _script(w, default=_small)
            value, exc = _outcome(lambda: w.manager.download_model(_MODEL_URL, filename=given))
            assert exc is None and value["status"] == "completed", (given, value, exc)
            assert Path(value["path"]).resolve() == (models / saved).resolve(), (given, value["path"])
            assert (models / saved).read_bytes() == _GGUF, given
        assert _tree(outside) == [] and not (tmp_path / "x1.gguf").exists() and not (models / "a").exists()
        before = _tree(models)
        value, exc = _outcome(lambda: w.manager.download_model(_MODEL_URL, filename=".."))
        assert isinstance(exc, ValueError), (value, exc)
        assert _tree(models) == before

        # c2 -- a target outside every model directory, or a link out of one,
        # is refused by name before any resolution.
        elsewhere = tmp_path / "elsewhere"
        escape = models / "escape"
        escape.symlink_to(outside, target_is_directory=True)
        for target in (elsewhere, escape):
            _reset(w)
            hops = _script(w, default=_small)
            value, exc = _outcome(
                lambda: w.manager.download_model(_MODEL_URL, filename="y.gguf", target_dir=str(target)))
            assert isinstance(exc, ValueError), (str(target), value, exc)
            assert str(models) in str(exc) or str(models.resolve()) in str(exc), str(exc)
            assert w.net.lookups == [] and hops.calls == [], (str(target), w.net.lookups, hops.calls)
        assert _tree(outside) == [] and _tree(elsewhere) == []

        # c3 -- witness: a directory inside a model directory is accepted.
        _script(w, default=_small)
        value, exc = _outcome(
            lambda: w.manager.download_model(_MODEL_URL, filename="z.gguf", target_dir=str(models / "sub")))
        assert exc is None and value["status"] == "completed", (value, exc)
        assert (models / "sub" / "z.gguf").read_bytes() == _GGUF


# ---------------------------------------------------------------------------
# OG14 -- importing the signature library never fetches
# ---------------------------------------------------------------------------
class _OqsFinder:
    """First on the meta path: records every import of ``oqs`` and answers it itself.

    With no body, the import fails as a package that is not there; with a
    body, the module is made and the body runs as its code.
    """

    def __init__(self, body=None):
        self.attempts = []
        self.body = body

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] != "oqs":
            return None
        self.attempts.append(fullname)
        if self.body is None:
            raise ModuleNotFoundError("oqs is answered by the window", name="oqs")
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.body(module)


def _exits(module):
    raise SystemExit(1)


@contextmanager
def _signature_window(tmp_path, finder, *, found=None, seeded=_ABSENT):
    """The signature module loaded with its library probe scripted.

    ``find_library`` answers ``found`` for the library's two names, and
    ``CDLL`` loads only that; the install paths the probe reads point at empty
    directories. A module the import raises ``SystemExit`` from is caught and
    handed back as ``escaped``.
    """
    import ctypes
    import ctypes.util

    probe = SimpleNamespace(found=[], loaded=[])

    def find_library(name):
        probe.found.append(name)
        return found if name in ("oqs", "liboqs") else None

    def cdll(name, *args, **kwargs):
        probe.loaded.append(str(name))
        if found is not None and str(name) == found:
            return SimpleNamespace(name=str(name))
        raise OSError(f"{name}: cannot open shared object file")

    saved = (ctypes.util.find_library, ctypes.CDLL)
    saved_env = {name: os.environ.get(name) for name in ("OQS_INSTALL_PATH", "HOME")}
    saved_oqs = sys.modules.get("oqs", _ABSENT)
    for name in ("oqs-install", "home"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    sys.meta_path.insert(0, finder)
    restore = None
    try:
        ctypes.util.find_library, ctypes.CDLL = find_library, cdll
        os.environ["OQS_INSTALL_PATH"] = str(tmp_path / "oqs-install")
        os.environ["HOME"] = str(tmp_path / "home")
        if seeded is _ABSENT:
            sys.modules.pop("oqs", None)
        else:
            sys.modules["oqs"] = seeded
        loaded, escaped = None, None
        try:
            loaded, restore = isolate(targets={_PQ: source("pqc_signatures.py")})
        except SystemExit as exc:
            escaped = exc
        yield SimpleNamespace(pqc=loaded[_PQ] if loaded else None, escaped=escaped, probe=probe)
    finally:
        if restore is not None:
            restore()
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        ctypes.util.find_library, ctypes.CDLL = saved
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if saved_oqs is _ABSENT:
            sys.modules.pop("oqs", None)
        else:
            sys.modules["oqs"] = saved_oqs


def _import_oqs_texts(tree):
    return [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "import oqs" in node.value
    ]


def test_og14_importing_the_signature_library_never_fetches(tmp_path):
    # c1 -- the shared library loads nowhere: the package is never imported,
    # and the reason names the library.
    finder = _OqsFinder()
    with _signature_window(tmp_path / "c1", finder) as w:
        assert w.escaped is None, repr(w.escaped)
        assert finder.attempts == [], f"the package was imported: {finder.attempts}"
        assert w.pqc.PQC_AVAILABLE is False
        assert "shared library" in (w.pqc.PQC_UNAVAILABLE_REASON or ""), w.pqc.PQC_UNAVAILABLE_REASON
        assert w.probe.found, "the probe asked for the library"

    # c2 -- the library loads, and the package exits while loading: the
    # module still loads, and its reason names the exit.
    finder = _OqsFinder(_exits)
    with _signature_window(tmp_path / "c2", finder, found="liboqs.so.9") as w:
        assert w.escaped is None, "SystemExit escaped the module's import"
        assert finder.attempts == ["oqs"], finder.attempts
        assert w.pqc.PQC_AVAILABLE is False
        assert "exited" in (w.pqc.PQC_UNAVAILABLE_REASON or ""), w.pqc.PQC_UNAVAILABLE_REASON

    # c3 -- witness: a package already in the module cache is used as it is,
    # and the probe is not asked.
    oqs = types.ModuleType("oqs")
    oqs.get_enabled_sig_mechanisms = lambda: []
    oqs.Signature = lambda name, *args: SimpleNamespace(name=name)
    finder = _OqsFinder()
    with _signature_window(tmp_path / "c3", finder, seeded=oqs) as w:
        assert w.escaped is None and w.pqc.oqs is oqs
        assert w.pqc.PQC_AVAILABLE is True
        assert w.probe.found == [] and w.probe.loaded == [] and finder.attempts == []

    # c4 -- no remedy tells the user to import the package.
    planted = ast.parse('tips = ["Check: python -c " \'"import oqs; print(1)"\']\n')
    assert len(_import_oqs_texts(planted)) == 1
    tree = ast.parse(source("startup_checks.py").read_text(encoding="utf-8"))
    assert _import_oqs_texts(tree) == [], _import_oqs_texts(tree)


# ---------------------------------------------------------------------------
# OG15 -- every gate reads the mode on disk
# ---------------------------------------------------------------------------
def _lock(mode):
    return f"MODE:{mode}\nTIMESTAMP:1700000000.0\nUSER_ID:tester\nHMAC:unsigned\n"


def _rewrite(path, text):
    """Rewrite a file in place, as another process would, one second after its last change."""
    before = os.stat(path)
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))


@contextmanager
def _mode_world(tmp_path):
    """The real security mode over two files of the test's, the real web gate and front door.

    No signing key, so the lockfile's HMAC is not asked; audit records are
    kept in ``audit`` and every read of the YAML mode is counted in ``reads``.
    Both files say Daily when the window opens.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    targets = {_KS: source("search_killswitch.py")}
    if source("web_gate.py").exists():
        targets[_WG] = source("web_gate.py")
    targets[_SM] = source("security_mode.py")
    if source("egress.py").exists():
        targets[_EG] = source("egress.py")
    loaded, restore = isolate(targets=targets)
    try:
        sm = loaded[_SM]
        loaded[_KS]._STATE_PATH = tmp_path / "switch.json"
        world = SimpleNamespace(
            sm=sm, eg=loaded.get(_EG), audit=[], reads=[],
            yaml=tmp_path / "security.yaml", lock=tmp_path / "mode.lock",
        )
        world.yaml.write_text("security_mode: daily\n", encoding="utf-8")
        world.lock.write_text(_lock("daily"), encoding="utf-8")
        sm._SECURITY_YAML, sm._LOCKFILE_PATH = world.yaml, world.lock
        sm._load_signing_key = lambda: None
        sm._audit_log = lambda event, severity="INFO", **details: world.audit.append(event)
        read = sm._read_yaml_mode

        def counted():
            world.reads.append(1)
            return read()

        sm._read_yaml_mode = counted
        yield world
    finally:
        restore()


def test_og15_every_gate_reads_the_mode_on_disk(tmp_path):
    # c1 -- Daily is read and kept; both files rewritten to Bulbe by another
    # process: the next read is Bulbe, and the front door refuses on it.
    with _mode_world(tmp_path / "c1") as w:
        assert w.sm.get_current_mode() == "daily"
        _rewrite(w.yaml, "security_mode: bulbe\n")
        _rewrite(w.lock, _lock("bulbe"))
        assert w.sm.get_current_mode() == "bulbe", "a mode written on disk was not read"
        assert w.eg is not None, "absence: opti_oignon/egress.py is not written"
        refusal = w.eg.web_refusal()
        assert refusal is not None and refusal[0] == "mode", refusal

    # c2 -- only the lockfile rewritten: the two disagree, the mode fails
    # secure to Bulbe, and the disagreement is recorded once.
    with _mode_world(tmp_path / "c2") as w:
        assert w.sm.get_current_mode() == "daily"
        _rewrite(w.lock, _lock("bulbe"))
        assert [w.sm.get_current_mode() for _ in range(3)] == ["bulbe"] * 3
        assert w.audit.count("security_mode_mismatch") == 1, w.audit

    # c3 -- witness: files unchanged, ten reads of the mode read the YAML once.
    with _mode_world(tmp_path / "c3") as w:
        assert [w.sm.get_current_mode() for _ in range(10)] == ["daily"] * 10
        assert len(w.reads) == 1, len(w.reads)
