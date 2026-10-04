#!/usr/bin/env python3
"""Egress census guard: every network sink in the package is gated, owed, or exempt by name.

A request that leaves the machine is the one thing a local-first platform
promises not to make behind its user's back. The platform keeps that promise
with gates: the web gate (exactly Daily mode, the kill switch released), the
local rule (this machine, and the operator's own services), and the peer rule
(Veilid). A gate protects only the calls that ask it. This guard proves, on
every commit, that every network sink in the package is one of:

  (a) in a gate home -- a module where the platform holds a connection
      behind a gate -- and, inside it, behind a gate in its own function;
  (b) exempt by name, with the reason it is not an outbound request of the
      platform;
  (c) owed, in a ledger of counts by module and by kind that MAY ONLY SHRINK.

A bundled plugin with a sink is accepted only when its manifest, parsed,
lists ``network_outbound`` among its permissions: the host gates what such a
plugin runs, and a comment in the manifest is not a permission. Every
third-party library the package imports is classified in ``LIBRARIES``, and a
library that fetches on first use names what bounds it.

WHAT COUNTS AS A SINK. A call, resolved through the module's bindings, to:

  * urllib and http.client: ``urlopen``, ``urlretrieve``, ``open`` on an
    opener built by ``build_opener`` or ``OpenerDirector``, an
    ``HTTPConnection`` or ``HTTPSConnection`` built (a local subclass
    included), or their ``request``, ``putrequest`` or ``connect``;
  * socket: ``create_connection``, ``getaddrinfo``, ``gethostbyname``,
    ``gethostbyname_ex``, ``gethostbyaddr``, ``getnameinfo``, ``getfqdn``;
    and ``connect``, ``connect_ex``, ``sendto`` or ``sendmsg`` on an internet
    socket. A name counts as a Unix socket only when every binding of it in
    the module builds ``socket.socket(AF_UNIX, ...)``; a default family, a
    variable family, ``fromfd``, ``socketpair`` and the result of
    ``wrap_socket`` are internet sockets;
  * asyncio: ``open_connection``, and, in a module that imports asyncio, any
    attribute call named ``create_connection``,
    ``create_datagram_endpoint``, ``sock_connect``, ``getaddrinfo`` or
    ``getnameinfo`` (unscoped, so it can only over-count);
  * multiprocessing: ``multiprocessing.connection.Client`` whose address is
    a tuple or not a string literal;
  * logging: ``SocketHandler``, ``DatagramHandler``, ``HTTPHandler`` and
    ``SMTPHandler``, and ``SysLogHandler`` whose address is a tuple, absent
    (its default is a UDP address), or not a string literal;
  * a client library: any call on ``requests``, ``httpx``, ``aiohttp``,
    ``urllib3``, ``primp``, ``pycurl``, ``curl_cffi``, ``websockets``,
    ``websocket``, ``ddgs``, ``duckduckgo_search``, ``huggingface_hub``,
    ``qdrant_client``, ``weaviate``, ``pinecone``, ``smtplib``, ``ftplib``,
    ``imaplib``, ``poplib``, ``nntplib``, ``telnetlib`` or ``xmlrpc.client``,
    or on what their classes build; ``chromadb.HttpClient`` and
    ``CloudClient``; ``veilid.api_connector``; ``webbrowser.open*``. An
    exception class is not a client: a name ending in ``Error``,
    ``Exception`` or ``Warning`` is never charged, and neither is a value
    object that holds settings and sends nothing, named in ``_INERT``;
  * the Ollama client: every request method, the model management methods
    (``pull``, ``push``, ``create``, ``copy``, ``delete``), ``web_search``,
    ``web_fetch``, ``Client`` and ``AsyncClient``; and, in a module that
    imports an HTTP transport, a string literal ending in ``/api/pull``,
    ``/api/push``, ``/api/create``, ``/api/copy`` or ``/api/delete``;
  * chromadb embedding by itself, in a module that imports chromadb: an
    attribute call named ``add``, ``upsert``, ``update`` or ``query`` given
    ``query_texts``, given ``documents`` without ``embeddings``, given
    ``embeddings`` that is ``None`` or a conditional with a ``None`` branch,
    or given a ``**`` splat; a string constant ``"query_texts"`` outside a
    docstring; and a ``PersistentClient``, ``Client`` or ``EphemeralClient``
    built without settings that name ``anonymized_telemetry=False``;
  * a process: ``subprocess.run``, ``Popen``, ``call``, ``check_call``,
    ``check_output``, ``getoutput``, ``getstatusoutput``; ``os.system``,
    ``os.popen``, ``os.exec*``, ``os.spawn*``, ``os.posix_spawn``,
    ``os.posix_spawnp``; ``asyncio.create_subprocess_*``; ``pty.spawn`` --
    when the program it runs is network-capable, a shell or an interpreter,
    or cannot be read. The program is the first argv token; behind a wrapper
    (``env``, ``nice``, ``timeout``, ``sudo``, ``unshare`` and the others in
    ``_WRAPPERS``) it is the next token that is neither an option, an
    option's argument, nor a ``NAME=value`` assignment; ``bwrap`` and
    ``systemd-run`` take options this reader does not know, so their program
    cannot be read. A string a shell runs is split on ``;``, ``&``, ``|``
    and newlines, and each command's program is read.

The Rust crates are read too: a ``Cargo.lock`` package in
``_NETWORK_CRATES``, ``tokio`` with its ``net`` or ``full`` feature in a
``Cargo.toml``, or a ``std::net`` path in a source file, outside comments and
literals (read with the Rust lexer of the comment-only guard).

BINDINGS FOLLOWED: imports and aliases, star imports, ``import_module`` and
``__import__`` with a constant, a ``sys.modules`` lookup, ``getattr`` with a
constant, assignments to names and to attributes, unpacking, the walrus, a
``with ... as`` or ``for`` target, a parameter default, ``or`` and
conditional expressions, a class attribute, a local subclass of a sink class,
and a function, a method or a lambda that returns a sink object. What a
request returns is not a client: only a class, or one of the factories in
``_FACTORIES``, builds an object the census follows. Names are not scoped: a
name bound to a sink object anywhere in a module is one everywhere in it,
which can only over-count.

GATE PRESENCE IS PER FUNCTION. In a home, each sink sits in a function F that:

  1. calls one of the home's ``gates`` on a line before the sink; or
  2. makes the request on a receiver built by one of the home's
     ``factories``, a function that itself calls one of the gates, or
     another such factory; or
  3. is private (its name starts with an underscore) and is only ever called,
     by name, from functions that satisfy one of these rules, searched in the
     home and in the modules its entry names in ``callers_in``. A reference
     that is not a call, or having no caller at all, leaves F ungated: an
     unresolved call site never counts as gated; or
  4. is named in the home's ``ungated`` map, with its reason.

"Before" is line order, not control flow: a gate in a branch not taken
counts. A sink at module level sits in no function and is never gated.

THE QUESTIONS, with disjoint domains so no one can cover for another:

  * ``find_violations``           -- a module with a sink that is not a home,
                                     owed, exempt, or a bundled plugin.
  * ``find_ungated_sinks``        -- a sink in a home that no rule gates.
  * ``find_count_drift``          -- a home whose sink count differs from its
                                     entry, in either direction.
  * ``find_ledger_growth``        -- an owed module with more sinks of a kind
                                     than its entry, or a kind it did not owe.
  * ``find_stale_ledger_entries`` -- an owed count above what the census finds
                                     (lower it), or an owed module with no sink.
  * ``find_stale_exemptions``     -- an exempt module with no sink, or gone.
  * ``find_stale_homes``          -- a home with no sink, or gone.
  * ``find_home_proofs_missing``  -- a home's contract id that names no test
                                     function, or whose suite does not name
                                     the home.
  * ``find_unpermitted_plugins``  -- a bundled plugin with a sink and no parsed
                                     ``network_outbound`` permission.
  * ``find_unclassified_imports`` -- a third-party import absent from
                                     ``LIBRARIES``, or a library that fetches on
                                     first use and names nothing that bounds it.
  * ``find_network_crates``       -- the Rust rule above.

The green line carries its denominator: the modules scanned, the sink sites
in the gate homes, the sinks owed and the modules that owe them, the
exemptions, the bundled plugins gated at the host, the third-party imports
classified and the crate manifests read.

WHAT THE CENSUS DOES NOT SEE, said here rather than claimed covered:

  * a name assembled at run time, or read through ``vars``, ``__dict__``,
    ``globals``, ``operator.attrgetter``, ``methodcaller``, ``exec`` or
    ``eval``;
  * a network object received from another module;
  * the program of an argv built elsewhere: the call is counted, its program
    is not read;
  * a library's own egress beyond its class in ``LIBRARIES``;
  * what another program does once it runs: the Ollama server, pip, npm,
    git, the veilid-server;
  * control flow: "before" is line order;
  * the destination of any sink;
  * C extensions;
  * code that is not Python, other than the Rust rule: the frontend, the
    ``android/`` tree;
  * user-installed plugins under ``opti_oignon/data/``, which it never opens:
    every directory named ``data`` is pruned before listing;
  * the maintainer scripts outside the package (``scripts/``,
    ``.github/scripts/``, ``tests/``).

The helpers are pure and import-safe; ``main`` scans the repository and exits
non-zero on any finding. Usage: ``egress_census_guard.py [REPO_ROOT]``.
"""

import ast
import importlib.util
import os
import re
import shlex
import sys
from collections import namedtuple
from pathlib import Path

import tomllib

_PACKAGE_DIR = "opti_oignon"
_PRUNED = frozenset({"data", "__pycache__"})

Home = namedtuple("Home", "classes gates factories callers_in ungated sinks contracts reason")
Site = namedtuple("Site", "kind line function")
Estate = namedtuple("Estate", "root modules manifests rust cache unread", defaults=((),))
Result = namedtuple("Result", "code lines estate census")


# ---------------------------------------------------------------------------
# The tables.
# ---------------------------------------------------------------------------
# Gate homes: where the platform holds a connection behind a gate. Each entry
# names its gates, the functions that build its transports, the modules that
# call into it, the functions it leaves ungated with their reason, its sink
# count, and the contracts that prove its gate.
HOMES = {
    "opti_oignon/web_search.py": Home(
        classes=("web",),
        gates=("search_refusal", "_gate", "_require_open"),
        factories=(),
        callers_in=(),
        ungated={
            "_connect_checked": (
                "the pinned connections' connect() calls it, and http.client calls connect() inside "
                "fetch_page, after its gate; it connects only to the addresses _checked_addresses "
                "returned for that hop"
            ),
        },
        sinks=10,
        contracts=("ud1", "ks6"),
        reason=(
            "the web search and the page fetch: a search asks the web gate before it runs, a fetch "
            "asks it before every hop and after every read, and every address it connects to is checked"
        ),
    ),
    "opti_oignon/model_manager.py": Home(
        classes=("web",),
        gates=("require_web",),
        factories=(),
        callers_in=(),
        ungated={
            "connect": (
                "the pinned connections' connect(), which http.client calls inside the pinned opener; "
                "it connects only to the address _validate_and_resolve checked for that hop, after "
                "urlopen_ssrf_safe asked the web gate"
            ),
            "_default_pinned_opener": (
                "urlopen_ssrf_safe's default opener, handed over as a value and called only inside its "
                "loop, after the web gate is asked for that hop and the hop's address is checked"
            ),
        },
        sinks=6,
        contracts=("og4",),
        reason=(
            "the model downloader: it asks the web gate before any request, before every hop and "
            "after every block, and connects to a public address only"
        ),
    ),
    "opti_oignon/inference_backend.py": Home(
        classes=("local", "operator", "web"),
        gates=("_bulbe_refusal_for", "_bulbe_refusal", "_bulbe_gate", "local_refusal", "require_web"),
        factories=("_transport", "_embed_client"),
        callers_in=(),
        ungated={},
        sinks=14,
        contracts=("bg1", "bg2"),
        reason=(
            "the inference transports: every request to Ollama or to a llama-server asks the local "
            "rule for its endpoint first"
        ),
    ),
}

# Debt found when this guard was written: module -> {sink kind: count}.
# MAY ONLY SHRINK: a new sink, or a sink of a new kind, in an owed module is a
# finding, and so is a count the census no longer finds.
LEDGER = {
    "opti_oignon/context_manager.py": {"process": 2},
    "opti_oignon/core_client.py": {"urllib": 1},
    "opti_oignon/dep_monitor.py": {"process": 1},
    "opti_oignon/memory/vector_store.py": {"chromadb": 1},
    "opti_oignon/model_lifecycle.py": {"ollama": 5, "requests": 2},
    "opti_oignon/network_hardening.py": {"socket": 1},
    "opti_oignon/plugin_subprocess.py": {"process": 1},
    "opti_oignon/rag_external.py": {"pinecone": 12, "qdrant_client": 1, "weaviate": 2},
    "opti_oignon/rag_store.py": {"chromadb": 4},
    "opti_oignon/sandbox_manager.py": {"process": 3},
    "opti_oignon/token_counter.py": {"urllib": 1},
    "opti_oignon/ui.py": {"process": 3, "urllib": 3, "webbrowser": 1},
    # The peer rule is asked in node.py before the connector runs, but the
    # connector is a nested function handed back as a value, which no rule
    # of gate presence can follow, and no contract names this module yet.
    "opti_oignon/veilid/client.py": {"veilid": 2},
}

# Not an outbound request of the platform, by name, with the reason.
EXEMPT = {
    "opti_oignon/cli/client.py": (
        "the command-line client talks only to the platform's own API, at the address its user "
        "configures, to drive the platform; it is the user's request, not one the platform makes"
    ),
    "opti_oignon/network_bind_guard.py": (
        "the bind guard's probe connects to 127.0.0.1 on the API's own port to learn whether the "
        "API is already bound there; it never leaves this machine"
    ),
}

# Every third-party top-level import: (class, reason, what bounds it). The
# class is "network" (a client library, whose calls the census counts),
# "offline", or "fetches on first use", which must name the contract or the
# owed module that bounds it.
_CLIENT = "a client library: every call on it is a sink this census counts"
LIBRARIES = {
    "Crypto": ("offline", "symmetric ciphers and hashes, computed in process", ()),
    "argon2": ("offline", "password hashing, computed in process", ()),
    "bcrypt": ("offline", "password hashing, computed in process", ()),
    "chromadb": (
        "fetches on first use",
        "its default embedding function downloads a model, and its client sends telemetry unless "
        "told not to; both are sinks this census counts, owed in the two vector stores",
        ("opti_oignon/rag_store.py", "opti_oignon/memory/vector_store.py"),
    ),
    "click": ("offline", "the command line's argument parser", ()),
    "cryptography": ("offline", "keys, signatures and ciphers, computed in process", ()),
    "ddgs": ("network", _CLIENT, ()),
    "docx": ("offline", "reads and writes Word documents on disk", ()),
    "duckduckgo_search": ("network", _CLIENT, ()),
    "fastapi": ("offline", "serves the API to this machine's clients; it never connects out", ()),
    "fido2": ("offline", "talks to a security key on this machine's USB bus", ()),
    "httpx": ("network", _CLIENT, ()),
    "joblib": ("offline", "persists and parallelises computations on this machine", ()),
    "llama_cpp": (
        "offline",
        "loads a GGUF file from a local path; from_pretrained, which downloads, is never called", (),
    ),
    "numpy": ("offline", "array arithmetic, computed in process", ()),
    "ollama": ("network", _CLIENT, ()),
    "openpyxl": ("offline", "reads and writes spreadsheets on disk", ()),
    "oqs": (
        "fetches on first use",
        "liboqs-python clones and builds liboqs from GitHub at import when its shared library is "
        "missing; the signature library imports it only when that library loads",
        ("og14",),
    ),
    "pinecone": ("network", _CLIENT, ()),
    "psutil": ("offline", "reads this machine's processes and resources", ()),
    "pydantic": ("offline", "validates data in process", ()),
    "pyotp": ("offline", "computes one-time passwords in process", ()),
    "pypdf": ("offline", "reads PDF files on disk", ()),
    "pysqlcipher3": ("offline", "an encrypted SQLite database on disk", ()),
    "pytesseract": ("offline", "runs the local tesseract program on an image", ()),
    "pywhispercpp": (
        "offline",
        "imported only to learn whether the binding is installed; nothing is loaded through it", (),
    ),
    "qdrant_client": ("network", _CLIENT, ()),
    "qrcode": ("offline", "draws a QR code in process", ()),
    "requests": ("network", _CLIENT, ()),
    "sklearn": ("offline", "fits and applies models in process; its dataset fetchers are never called", ()),
    "sqlcipher3": ("offline", "an encrypted SQLite database on disk", ()),
    "starlette": ("offline", "serves the API to this machine's clients; it never connects out", ()),
    "tqdm": ("offline", "draws progress bars", ()),
    "uvicorn": ("offline", "serves the API on the address the bind guard allows; it never connects out", ()),
    "veilid": ("network", "the peer network's client; its connector is a sink this census counts", ()),
    "weaviate": ("network", _CLIENT, ()),
    "websockets": ("network", _CLIENT, ()),
    "yaml": ("offline", "parses YAML text in process", ()),
}

_CLASSES = frozenset({"network", "offline", "fetches on first use"})


# ---------------------------------------------------------------------------
# What counts as a sink.
# ---------------------------------------------------------------------------
_URLLIB = frozenset({"urllib.request.urlopen", "urllib.request.urlretrieve"})
_OPENERS = frozenset({"urllib.request.build_opener()", "urllib.request.OpenerDirector()"})
_HTTP_CLASSES = frozenset({"http.client.HTTPConnection", "http.client.HTTPSConnection"})
_HTTP_INSTANCES = frozenset(name + "()" for name in _HTTP_CLASSES)
_HTTP_METHODS = frozenset({"request", "putrequest", "connect"})
_SOCKET_FUNCTIONS = frozenset("socket." + name for name in (
    "create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex",
    "gethostbyaddr", "getnameinfo", "getfqdn",
))
_SOCKET_METHODS = frozenset({"connect", "connect_ex", "sendto", "sendmsg"})
_UNIX_SOCKET = "socket.socket(AF_UNIX)"
_INET_SOCKETS = frozenset({"socket.socket()", "socket.fromfd()", "socket.socketpair()"})
_ASYNCIO_METHODS = frozenset({
    "create_connection", "create_datagram_endpoint", "sock_connect", "getaddrinfo", "getnameinfo",
})
_MP_CLIENT = "multiprocessing.connection.Client"
_LOG_HANDLERS = frozenset("logging.handlers." + name for name in (
    "SocketHandler", "DatagramHandler", "HTTPHandler", "SMTPHandler",
))
_SYSLOG = "logging.handlers.SysLogHandler"
_LIBRARY_ROOTS = (
    "requests", "httpx", "aiohttp", "urllib3", "primp", "pycurl", "curl_cffi", "websockets",
    "websocket", "ddgs", "duckduckgo_search", "huggingface_hub", "qdrant_client", "weaviate",
    "pinecone", "smtplib", "ftplib", "imaplib", "poplib", "nntplib", "telnetlib", "xmlrpc.client",
)
_LIBRARY_CALLABLES = {
    "chromadb.HttpClient": "chromadb",
    "chromadb.CloudClient": "chromadb",
    "veilid.api_connector": "veilid",
    "webbrowser.open": "webbrowser",
    "webbrowser.open_new": "webbrowser",
    "webbrowser.open_new_tab": "webbrowser",
}
_OLLAMA = "ollama"
_OLLAMA_NAMES = frozenset({
    "chat", "generate", "embeddings", "embed", "ps", "list", "show", "pull", "push", "create",
    "copy", "delete", "web_search", "web_fetch", "Client", "AsyncClient",
})
_OLLAMA_PATHS = ("/api/pull", "/api/push", "/api/create", "/api/copy", "/api/delete")
_TRANSPORTS = frozenset({"requests", "httpx", "urllib", "http", "aiohttp", "urllib3"})
_CHROMA_METHODS = frozenset({"add", "upsert", "update", "query"})
_CHROMA_LOCAL = frozenset({"chromadb.PersistentClient", "chromadb.Client", "chromadb.EphemeralClient"})
# Value objects of a client library: they hold settings and send nothing.
_INERT = frozenset({
    "httpx.Timeout", "httpx.Limits", "httpx.Headers", "httpx.URL", "httpx.Cookies",
    "aiohttp.ClientTimeout",
})

# Calls whose result the census follows besides a class: they build the
# object a request is later made on.
_FACTORIES = frozenset({"build_opener", "fromfd", "socketpair", "session"})
_IMPORT_FUNCTIONS = frozenset({"importlib.import_module", "importlib.__import__", "builtins.__import__"})

_PROCESS_CALLS = frozenset({
    "subprocess.run", "subprocess.Popen", "subprocess.call", "subprocess.check_call",
    "subprocess.check_output", "subprocess.getoutput", "subprocess.getstatusoutput",
    "os.system", "os.popen", "os.posix_spawn", "os.posix_spawnp", "pty.spawn",
})
_PROCESS_PREFIXES = ("os.exec", "os.spawn", "asyncio.create_subprocess_")
_SHELL_STRINGS = frozenset({
    "os.system", "os.popen", "subprocess.getoutput", "subprocess.getstatusoutput",
    "asyncio.create_subprocess_shell",
})
_NETWORK_PROGRAMS = frozenset({
    "curl", "wget", "git", "pip", "pip3", "pip-audit", "npm", "npx", "pnpm", "yarn", "node", "deno",
    "bun", "ollama", "cargo", "rustup", "go", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "socat",
    "telnet", "ftp", "dig", "nslookup", "host", "ping", "apt", "apt-get", "snap", "flatpak", "docker",
    "podman", "uv", "uvx", "pipx", "conda", "mamba", "gh", "hf", "huggingface-cli",
})
_INTERPRETERS = frozenset({"bash", "sh", "dash", "zsh", "python", "python3", "perl", "ruby", "node"})
_VERSIONED_PYTHON = re.compile(r"python3\.[0-9]+$")
# A wrapper runs another program: name -> the options that take a separate
# argument. ``timeout`` also takes one operand, its duration, before the
# program.
_WRAPPERS = {
    "env": ("-u", "--unset", "-C", "--chdir"),
    "nice": ("-n", "--adjustment"),
    "ionice": ("-c", "--class", "-n", "--classdata", "-p", "--pid", "-P", "--pgid", "-u", "--uid"),
    "timeout": ("-s", "--signal", "-k", "--kill-after"),
    "stdbuf": ("-i", "--input", "-o", "--output", "-e", "--error"),
    "nohup": (),
    "setsid": (),
    "sudo": ("-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt", "-C", "--close-from",
             "-D", "--chdir", "-U", "--other-user", "-r", "--role", "-t", "--type", "-T",
             "--command-timeout"),
    "doas": ("-u", "-C"),
    "unshare": ("-S", "--setuid", "-G", "--setgid", "-R", "--root", "-w", "--wd"),
    "firejail": (),
    "flatpak-spawn": (),
    "xdg-open": (),
    "gio": (),
}
_WRAPPER_OPERANDS = {"timeout": 1}
_OPAQUE_WRAPPERS = frozenset({"bwrap", "systemd-run"})
_ENV_SPLIT = ("-S", "--split-string")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")

_NETWORK_CRATES = frozenset({
    "reqwest", "hyper", "ureq", "isahc", "surf", "curl", "mio", "socket2", "async-std",
    "trust-dns-resolver", "hickory-resolver",
})
_STD_NET = re.compile(r"\bstd\s*::\s*(?:net\b|\{[^}]*\bnet\b)")

_BINDERS = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Assign, ast.AnnAssign,
    ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor, ast.comprehension, ast.withitem,
)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


# The roots a sink can be reached from. The census binds a name only to what
# starts at one of them, and never deeper than ``_DEPTH`` segments: an
# attribute chain that rebinds itself (``node = node.parent``) would grow
# without end, and a name bound to anything else cannot reach a sink.
_WATCHED = frozenset({
    "urllib", "http", "socket", "asyncio", "multiprocessing", "logging", "subprocess", "os", "pty",
    "importlib", "builtins", "ollama", "chromadb", "veilid", "webbrowser", "xmlrpc",
}) | frozenset(root.split(".")[0] for root in _LIBRARY_ROOTS)
_DEPTH = 8


def _watched(q):
    return q.split(".", 1)[0].split("(", 1)[0] in _WATCHED and q.count(".") < _DEPTH


def _leaf(q):
    return q.rsplit(".", 1)[-1]


def _rooted(q, root):
    return q == root or q.startswith(root + ".") or q.startswith(root + "(")


def _is_exception(q):
    return _leaf(q).split("(", 1)[0].endswith(("Error", "Exception", "Warning"))


def _is_none(node):
    return isinstance(node, ast.Constant) and node.value is None


# Every sink needs, somewhere in its module, an import of the module it comes
# from, or, for the process calls of ``os`` and the logging handlers, their
# names, or a dynamic import. A module with none of them holds no sink, and
# only its import statements are read. This is a filter for speed, never a
# rule: it lets through every spelling above.
_SINK_ROOTS = frozenset({
    "urllib", "http", "socket", "asyncio", "multiprocessing", "subprocess", "pty", "webbrowser",
    "ollama", "chromadb", "veilid", "xmlrpc",
}) | frozenset(root.split(".")[0] for root in _LIBRARY_ROOTS)
_OS_SINK = re.compile(r"\b(?:system|popen|exec[lv]p?e?|spawn[lv]p?e?|posix_spawnp?)\b")
_DYNAMIC = ("__import__", "import_module", "modules[", "modules.get", "wrap_socket")


def _may_hold_sink(text, imported):
    tops = {name.split(".")[0] for name in imported}
    if tops & _SINK_ROOTS:
        return True
    if "os" in tops and _OS_SINK.search(text):
        return True
    if "logging" in tops and "Handler" in text:
        return True
    return any(word in text for word in _DYNAMIC)


def _statement_imports(tree):
    """The import statements of a module, found without walking its expressions."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        for field in ("body", "orelse", "finalbody", "handlers", "cases"):
            stack.extend(getattr(node, field, None) or ())


def _target_names(target):
    """The names and the attributes an assignment target binds."""
    names, attributes = set(), set()
    stack = [target]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Subscript):
            node = node.value
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
        elif isinstance(node, (ast.Tuple, ast.List)):
            stack.extend(node.elts)
        elif isinstance(node, ast.Starred):
            stack.append(node.value)
    return names, attributes


def _func_name(call):
    """The name a call is made by: the bare name or the attribute."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


# ---------------------------------------------------------------------------
# Programs a process call runs.
# ---------------------------------------------------------------------------
def _program_of(tokens):
    """The program an argv runs: its name, "" when nothing runs, or None when unreadable.

    ``tokens`` holds each argv entry as a string, or None where the entry is
    not a literal.
    """
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token is None:
            return None
        if _ASSIGNMENT.match(token):
            i += 1
            continue
        name = token.rsplit("/", 1)[-1]
        if name in _OPAQUE_WRAPPERS:
            return None
        if name not in _WRAPPERS:
            return name
        takes_argument = _WRAPPERS[name]
        i += 1
        while i < len(tokens):
            option = tokens[i]
            if option is None:
                return None
            if not option.startswith("-") or option == "-":
                break
            if option == "--":
                i += 1
                break
            if name == "env" and (option in _ENV_SPLIT or option.startswith("--split-string=")):
                return None
            i += 2 if option in takes_argument else 1
        i += _WRAPPER_OPERANDS.get(name, 0)
    return ""


def _shell_programs(text):
    programs = []
    for command in re.split(r"[;&|\n]", text):
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if tokens:
            programs.append(_program_of(tokens))
    return programs


def _literal_tokens(elts):
    tokens = []
    for elt in elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            tokens.append(elt.value)
        else:
            tokens.append(None)
            break
    return tokens


def _argv_programs(node, shell):
    """The programs an argv expression runs, or None when it cannot be read."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if shell:
            return _shell_programs(node.value)
        return [_program_of(node.value.split()[:1])]
    if isinstance(node, (ast.List, ast.Tuple)):
        tokens = _literal_tokens(node.elts)
        if shell:
            return _shell_programs(tokens[0]) if tokens and tokens[0] is not None else None
        return [_program_of(tokens)]
    return None


def _reaches_network(program):
    if program is None:
        return True
    return program in _NETWORK_PROGRAMS or program in _INTERPRETERS or bool(_VERSIONED_PYTHON.match(program))


# ---------------------------------------------------------------------------
# The census of one module.
# ---------------------------------------------------------------------------
class _Census:
    """The sink sites of one module, with the syntax the gate rules read."""

    def __init__(self, text, keep=False):
        """``keep`` holds on to the syntax, which the gate rules read.

        Otherwise the tree is let go as soon as the sites are taken: a census
        of hundreds of modules that held every tree alive at once would spend
        more time in the garbage collector than in the census.
        """
        self.sites, self.nodes, self.imports, self.assigned = [], [], set(), {}
        self.tree = None
        self.unparsed = None
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError) as exc:
            # Recorded, and the census fails by name: a module that does not
            # parse counts no sink, which is not the same as having none.
            self.unparsed = f"{type(exc).__name__}: {exc}"
            return
        self.import_nodes = list(_statement_imports(tree))
        imported = []
        for node in self.import_nodes:
            self._note_import(node)
            if isinstance(node, ast.Import):
                imported += [alias.name for alias in node.names]
            elif not node.level and node.module:
                imported.append(node.module)
        if not _may_hold_sink(text, imported):
            # Nothing a sink needs: only the imports are read, from the
            # statements, without walking the expressions.
            self.tree = tree if keep else None
            return
        self.tree = tree
        self.names, self.attrs, self.returns = {}, {}, {}
        self.stars = set()
        self.calls, self.strings, self.binders = [], [], []
        self.function_returns, self.class_body = {}, set()
        self._collect()
        self._bind_imports()
        self._grown = True
        while self._grown:
            self._grown = False
            self._memo = {}
            for node in self.binders:
                self._bind_node(node)
        self._memo = {}
        self._take_sites()
        if not keep:
            kept = {"sites": self.sites, "imports": self.imports}
            self.__dict__.clear()
            self.__dict__.update(kept, nodes=[], assigned={}, tree=None, unparsed=None)

    # -- collection -------------------------------------------------------
    def _collect(self):
        # The hot loop of the census: the children are read from each node's
        # fields in place, in the order ast.iter_child_nodes gives them,
        # without its two generators per node.
        docstrings = set()
        stack = [(self.tree, None)]
        scopes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        node_type = ast.AST
        functions, binders = frozenset(_FUNCTIONS), frozenset(_BINDERS)
        assigns = frozenset((ast.Assign, ast.AnnAssign, ast.NamedExpr))
        push, calls, strings, bind = stack.append, self.calls.append, self.strings.append, self.binders.append
        while stack:
            node, owner = stack.pop()
            if isinstance(node, scopes):
                first = node.body[0] if node.body else None
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                        and isinstance(first.value.value, str):
                    docstrings.add(id(first.value))
            if isinstance(node, ast.ClassDef):
                self.class_body.update(id(stmt) for stmt in node.body)
            children = []
            for field in node._fields:
                value = getattr(node, field, None)
                if isinstance(value, node_type):
                    children.append(value)
                elif isinstance(value, list):
                    children.extend(item for item in value if isinstance(item, node_type))
            for child in children:
                # ast.parse makes exactly these classes, never a subclass of
                # one, so the exact type answers what isinstance would.
                kind = type(child)
                inner = child if kind in functions else owner
                push((child, inner))
                if kind is ast.Call:
                    calls((child, owner))
                elif kind is ast.Constant:
                    if isinstance(child.value, str) and id(child) not in docstrings:
                        strings((child, owner))
                elif kind is ast.Return and owner is not None and child.value is not None:
                    self.function_returns.setdefault(id(owner), []).append(child.value)
                if kind in binders:
                    bind(child)
                if kind in assigns and child.value is not None:
                    targets = child.targets if kind is ast.Assign else [child.target]
                    for target in targets:
                        names, attributes = _target_names(target)
                        for name in names | attributes:
                            self.assigned.setdefault(name, []).append(child.value)

    def _note_import(self, node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.imports.add(alias.name.split(".")[0])
        elif not node.level and node.module:
            self.imports.add(node.module.split(".")[0])

    def _bind_imports(self):
        for node in self.import_nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self._add(self.names, alias.asname, {alias.name})
                    else:
                        top = alias.name.split(".")[0]
                        self._add(self.names, top, {top})
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                for alias in node.names:
                    if alias.name == "*":
                        if _watched(node.module):
                            self.stars.add(node.module)
                    else:
                        self._add(self.names, alias.asname or alias.name, {node.module + "." + alias.name})

    # -- binding ----------------------------------------------------------
    def _add(self, table, key, values):
        values = {q for q in values if _watched(q)}
        if not values:
            return
        known = table.setdefault(key, set())
        if not values <= known:
            known |= values
            self._grown = True

    def _bind_node(self, node):
        if isinstance(node, _FUNCTIONS):
            values = set()
            for value in self.function_returns.get(id(node), ()):
                values |= self.resolve(value)
            self._add(self.returns, node.name, values)
            self._bind_defaults(node.args)
        elif isinstance(node, ast.Lambda):
            self._bind_defaults(node.args)
        elif isinstance(node, ast.ClassDef):
            values = set()
            for base in node.bases:
                values |= self.resolve(base)
            self._add(self.names, node.name, values)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                self._bind_target(target, node.value, id(node) in self.class_body)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if node.value is not None:
                self._bind_target(node.target, node.value, id(node) in self.class_body)
        elif isinstance(node, ast.NamedExpr):
            self._bind_target(node.target, node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            self._bind_target(node.target, node.iter)
        elif isinstance(node, ast.withitem):
            if node.optional_vars is not None:
                self._bind_target(node.optional_vars, node.context_expr)

    def _bind_defaults(self, args):
        positional = list(args.posonlyargs) + list(args.args)
        for param, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
            self._add(self.names, param.arg, set(self.resolve(default)))
        for param, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                self._add(self.names, param.arg, set(self.resolve(default)))

    def _bind_target(self, target, value, in_class=False):
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) \
                and len(target.elts) == len(value.elts) \
                and not any(isinstance(e, ast.Starred) for e in target.elts + value.elts):
            for sub_target, sub_value in zip(target.elts, value.elts):
                self._bind_target(sub_target, sub_value, in_class)
            return
        names, attributes = _target_names(target)
        if isinstance(value, ast.Lambda):
            values = set(self.resolve(value.body))
            for name in names | attributes:
                self._add(self.returns, name, values)
            return
        values = set(self.resolve(value))
        if not values:
            return
        for name in names:
            self._add(self.names, name, values)
        for name in attributes | (names if in_class else set()):
            self._add(self.attrs, name, values)

    # -- resolution -------------------------------------------------------
    def resolve(self, node):
        """Every dotted name an expression may stand for; ``()`` marks an object built."""
        if node is None:
            return frozenset()
        key = id(node)
        memo = self._memo
        if key in memo:
            return memo[key]
        memo[key] = frozenset()
        out = frozenset(self._resolve(node))
        memo[key] = out
        return out

    def _resolve(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.names:
                return self.names[node.id]
            return {module + "." + node.id for module in self.stars}
        if isinstance(node, ast.Attribute):
            out = {q + "." + node.attr for q in self.resolve(node.value)}
            return out | self.attrs.get(node.attr, set())
        if isinstance(node, ast.Call):
            return self._call_result(node)
        if isinstance(node, (ast.NamedExpr, ast.Await, ast.Starred)):
            return self.resolve(node.value)
        if isinstance(node, ast.BoolOp):
            return set().union(*(self.resolve(v) for v in node.values))
        if isinstance(node, ast.IfExp):
            return self.resolve(node.body) | self.resolve(node.orelse)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return set().union(*(self.resolve(e) for e in node.elts)) if node.elts else set()
        if isinstance(node, ast.Subscript):
            module = self._modules_lookup(node.value, node.slice)
            return {module} if module else self.resolve(node.value)
        return set()

    @staticmethod
    def _modules_lookup(container, key):
        if isinstance(container, ast.Attribute) and container.attr == "modules" \
                and isinstance(key, ast.Constant) and isinstance(key.value, str):
            return key.value
        return None

    def _imported(self, call):
        """The module a dynamic import names by a constant, or None."""
        func = call.func
        arg = call.args[0] if call.args else next((kw.value for kw in call.keywords if kw.arg == "name"), None)
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            if isinstance(func, ast.Attribute) and func.attr == "get" and call.args:
                return self._modules_lookup(func.value, call.args[0])
            return None
        dunder = (isinstance(func, ast.Name) and func.id == "__import__" and func.id not in self.names) or (
            isinstance(func, ast.Attribute) and func.attr == "__import__")
        named = isinstance(func, ast.Attribute) and func.attr == "import_module"
        resolved = self.resolve(func) & _IMPORT_FUNCTIONS
        if dunder or any(q.endswith("__import__") for q in resolved):
            return arg.value.split(".")[0]
        if named or resolved:
            return arg.value
        if isinstance(func, ast.Attribute) and func.attr == "get":
            return self._modules_lookup(func.value, arg)
        return None

    def _call_result(self, call):
        module = self._imported(call)
        if module:
            return {module}
        func = call.func
        if isinstance(func, ast.Name) and func.id == "getattr" and len(call.args) >= 2:
            name = call.args[1]
            leaf = name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else "?"
            return {q + "." + leaf for q in self.resolve(call.args[0])}
        out = set()
        if isinstance(func, ast.Attribute) and func.attr == "wrap_socket":
            out.add("socket.socket()")
        name = _func_name(call)
        if name in self.returns and isinstance(func, (ast.Name, ast.Attribute)):
            out |= self.returns[name]
        if isinstance(func, ast.Lambda):
            out |= self.resolve(func.body)
        for q in self.resolve(func):
            leaf = _leaf(q)
            if "(" in leaf:
                continue
            if q == "socket.socket":
                out.add(self._socket_object(call))
            elif leaf in _FACTORIES or leaf[:1].isupper():
                out.add(q + "()")
        return out

    def _socket_object(self, call):
        family = call.args[0] if call.args else next(
            (kw.value for kw in call.keywords if kw.arg == "family"), None)
        if family is None:
            return "socket.socket()"
        resolved = self.resolve(family)
        if resolved and all(q == "socket.AF_UNIX" for q in resolved):
            return _UNIX_SOCKET
        return "socket.socket()"

    # -- the sites --------------------------------------------------------
    def _take_sites(self):
        uses_asyncio = "asyncio" in self.imports
        uses_chroma = "chromadb" in self.imports
        uses_transport = bool(self.imports & _TRANSPORTS)
        found = []
        for call, owner in self.calls:
            kind = self._sink_kind(call, uses_asyncio, uses_chroma)
            if kind:
                found.append((call.lineno, kind, owner, call))
        for constant, owner in self.strings:
            value = constant.value
            if uses_transport and value.strip().split("?", 1)[0].rstrip("/").endswith(_OLLAMA_PATHS):
                found.append((constant.lineno, _OLLAMA, owner, constant))
            elif uses_chroma and value == "query_texts":
                found.append((constant.lineno, "chromadb", owner, constant))
        found.sort(key=lambda item: (item[0], item[1]))
        for line, kind, owner, node in found:
            site = Site(kind, line, owner.name if owner is not None else None)
            self.sites.append(site)
            self.nodes.append((site, owner, node))

    def _sink_kind(self, call, uses_asyncio, uses_chroma):
        func = call.func
        # Sorted, so a call that resolves to two roots is charged to the same
        # kind on every run, whatever the hash seed.
        quals = sorted(q for q in self.resolve(func) if q not in _INERT)
        named = set(quals)
        for q in quals:
            if q in _PROCESS_CALLS or q.startswith(_PROCESS_PREFIXES):
                return "process" if self._runs_network(call, q) else None
        if named & _URLLIB:
            return "urllib"
        method = func.attr if isinstance(func, ast.Attribute) else None
        receiver = self.resolve(func.value) if method else frozenset()
        if method == "open" and receiver & _OPENERS:
            return "urllib"
        if named & _HTTP_CLASSES or (method in _HTTP_METHODS and receiver & _HTTP_INSTANCES):
            return "http.client"
        if named & _SOCKET_FUNCTIONS or (method in _SOCKET_METHODS and receiver & _INET_SOCKETS):
            return "socket"
        if "asyncio.open_connection" in quals or (uses_asyncio and method in _ASYNCIO_METHODS):
            return "asyncio"
        if _MP_CLIENT in quals and self._address_counts(call):
            return "multiprocessing"
        if named & _LOG_HANDLERS or (_SYSLOG in named and self._address_counts(call)):
            return "logging"
        for q in quals:
            if _rooted(q, _OLLAMA) and _leaf(q) in _OLLAMA_NAMES:
                return _OLLAMA
        if uses_chroma:
            if method in _CHROMA_METHODS and self._embeds_itself(call):
                return "chromadb"
            if named & _CHROMA_LOCAL and not self._telemetry_off(call):
                return "chromadb"
        for q in quals:
            if _is_exception(q):
                continue
            for name, kind in _LIBRARY_CALLABLES.items():
                if _rooted(q, name):
                    return kind
            for root in _LIBRARY_ROOTS:
                if _rooted(q, root):
                    return root
        return None

    @staticmethod
    def _address_counts(call):
        address = call.args[0] if call.args else next(
            (kw.value for kw in call.keywords if kw.arg == "address"), None)
        if address is None:
            return True
        return not (isinstance(address, ast.Constant) and isinstance(address.value, str))

    @staticmethod
    def _embeds_itself(call):
        keywords = {}
        for kw in call.keywords:
            if kw.arg is None:
                return True
            keywords[kw.arg] = kw.value
        if "query_texts" in keywords:
            return True
        if "documents" in keywords and "embeddings" not in keywords:
            return True
        embeddings = keywords.get("embeddings")
        if embeddings is None:
            return False
        if _is_none(embeddings):
            return True
        return isinstance(embeddings, ast.IfExp) and (_is_none(embeddings.body) or _is_none(embeddings.orelse))

    def _telemetry_off(self, call):
        settings = next((kw.value for kw in call.keywords if kw.arg == "settings"), None)
        if settings is None:
            return False
        candidates = [settings]
        if isinstance(settings, ast.Name):
            candidates += self.assigned.get(settings.id, [])
        for candidate in candidates:
            for node in ast.walk(candidate):
                if isinstance(node, ast.keyword) and node.arg == "anonymized_telemetry" \
                        and isinstance(node.value, ast.Constant) and node.value.value is False:
                    return True
        return False

    def _runs_network(self, call, q):
        args = call.args
        keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        if q in _SHELL_STRINGS:
            command = args[0] if args else keywords.get("cmd", keywords.get("command"))
            programs = _argv_programs(command, shell=True) if command is not None else None
        elif q.startswith("os.exec") or q in ("os.posix_spawn", "os.posix_spawnp"):
            programs = _argv_programs(args[0], shell=False) if args else None
        elif q.startswith("os.spawn"):
            programs = _argv_programs(args[1], shell=False) if len(args) > 1 else None
        elif q.startswith("asyncio.create_subprocess_"):
            programs = [_program_of(_literal_tokens(args))] if args else None
        elif q == "pty.spawn":
            argv = args[0] if args else keywords.get("argv")
            programs = _argv_programs(argv, shell=False) if argv is not None else None
        else:
            argv = args[0] if args else keywords.get("args")
            shell = keywords.get("shell")
            if shell is not None and not (isinstance(shell, ast.Constant) and not shell.value):
                if not isinstance(shell, ast.Constant):
                    return True
                programs = _argv_programs(argv, shell=True) if argv is not None else None
            else:
                programs = _argv_programs(argv, shell=False) if argv is not None else None
        if programs is None:
            return True
        return any(_reaches_network(program) for program in programs)


def sink_sites(text):
    """Every sink site of a module's text: ``[Site(kind, line, function), ...]``."""
    return list(_Census(text).sites)


def count_sinks(text):
    """How many sink sites a module's text holds. Prose never counts."""
    return len(_Census(text).sites)


# ---------------------------------------------------------------------------
# The estate.
# ---------------------------------------------------------------------------
def _read_text(path):
    """Every file this guard reads goes through here, strictly.

    A file that is not UTF-8 text raises instead of coming back with its
    bytes dropped: read that way it would not be the file on disk.
    """
    return Path(path).read_text(encoding="utf-8")


def _walk(top, prune, unlisted=None):
    """Files under ``top``, sorted, with every pruned directory left unlisted.

    A directory the walk cannot list is appended to ``unlisted`` as
    ``(path, reason)``; without a list, it raises.
    """
    def failed(error):
        if unlisted is None:
            raise error
        unlisted.append((error.filename, error.strerror))

    out = []
    for current, dirs, files in os.walk(top, onerror=failed):
        dirs[:] = sorted(d for d in dirs if d not in prune and not d.startswith("."))
        out.extend(Path(current) / name for name in sorted(files))
    return out


def read_estate(root):
    """The package's modules, the bundled plugins' manifests and the Rust crates under ``root``.

    What could not be read -- a file that is not UTF-8 text, a directory the
    walk cannot list -- is named in ``unread``, never read as empty.
    """
    root = Path(root)
    package = root / _PACKAGE_DIR
    modules, manifests, rust, unread, unlisted = {}, {}, {}, [], []

    def read(path):
        try:
            return _read_text(path)
        except (OSError, UnicodeDecodeError) as exc:
            rel = path.relative_to(root).as_posix()
            unread.append(f"{rel}: cannot be read as UTF-8 text ({type(exc).__name__})")
            return None

    if package.is_dir():
        for path in _walk(package, _PRUNED, unlisted):
            rel = path.relative_to(root).as_posix()
            if path.suffix == ".py":
                text = read(path)
                if text is not None:
                    modules[rel] = text
        plugins = package / "plugins"
        if plugins.is_dir():
            try:
                entries = sorted(plugins.iterdir())
            except OSError:
                entries = []  # the walk above has named the directory
            for entry in entries:
                if entry.is_dir() and entry.name not in _PRUNED:
                    manifest = entry / "manifest.yaml"
                    rel = entry.relative_to(root).as_posix()
                    manifests[rel] = read(manifest) if manifest.is_file() else None
    crates = root / "rust"
    if crates.is_dir():
        for path in _walk(crates, _PRUNED | {"target"}, unlisted):
            rel = path.relative_to(root).as_posix()
            if path.name in ("Cargo.toml", "Cargo.lock") or (
                    path.suffix == ".rs" and "src" in path.relative_to(crates).parts):
                text = read(path)
                if text is not None:
                    rust[rel] = text
    for path, reason in unlisted:
        rel = Path(os.path.relpath(path, root)).as_posix()
        unread.append(f"{rel}: cannot be listed ({reason})")
    return Estate(root, modules, manifests, rust, {}, tuple(unread))


def _censuses(estate):
    cached = estate.cache.get("censuses")
    if cached is None:
        keep = set(HOMES) | {other for home in HOMES.values() for other in home.callers_in}
        cached = {rel: _Census(text, keep=rel in keep) for rel, text in estate.modules.items()}
        estate.cache["censuses"] = cached
    return cached


def take_census(estate):
    """module -> its sink sites, for every module of the estate."""
    return {rel: list(census.sites) for rel, census in _censuses(estate).items()}


def third_party_imports(estate):
    """Every third-party top-level name the estate imports, sorted."""
    names = set()
    for census in _censuses(estate).values():
        names |= census.imports
    stdlib = set(sys.stdlib_module_names) | {"__future__", _PACKAGE_DIR}
    return sorted(name for name in names if name not in stdlib)


def _plugin_of(rel):
    parts = rel.split("/")
    if len(parts) >= 4 and parts[0] == _PACKAGE_DIR and parts[1] == "plugins":
        return "/".join(parts[:3])
    return None


class ManifestParserMissing(Exception):
    """The YAML parser the plugin manifests are read with is not installed."""


def _manifest_refusal(manifest):
    """Why ``manifest`` does not permit network egress; ``None`` when it does.

    Each cause is named apart: no manifest, a manifest that does not parse,
    one that is not a mapping, one parsed without the permission. A parser
    that is not installed says nothing about the plugin: it raises
    ``ManifestParserMissing``, and the census fails by name.
    """
    if manifest is None:
        return "no manifest.yaml"
    try:
        import yaml
    except ImportError as exc:
        raise ManifestParserMissing("the YAML parser (PyYAML) is not installed") from exc
    try:
        data = yaml.safe_load(manifest)
    except Exception as exc:
        return f"a manifest.yaml that does not parse ({type(exc).__name__})"
    if not isinstance(data, dict):
        return "a manifest.yaml that is not a mapping"
    permissions = data.get("permissions")
    if isinstance(permissions, list) and "network_outbound" in permissions:
        return None
    return "no network_outbound permission in a parsed manifest"


def _permits_network(manifest):
    return _manifest_refusal(manifest) is None


def _kinds(sites):
    counts = {}
    for site in sites:
        counts[site.kind] = counts.get(site.kind, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# The questions.
# ---------------------------------------------------------------------------
def find_violations(estate, census=None):
    """A module with a sink that is not a home, owed, exempt, or a bundled plugin."""
    census = take_census(estate) if census is None else census
    out = []
    for rel, sites in sorted(census.items()):
        if sites and rel not in HOMES and rel not in LEDGER and rel not in EXEMPT and _plugin_of(rel) is None:
            kinds = ", ".join(f"{kind} {n}" for kind, n in sorted(_kinds(sites).items()))
            out.append(f"{rel}: {len(sites)} sink site(s) nobody gates, owes or exempts ({kinds}); "
                       f"first at line {sites[0].line}")
    return out


def _gate_lines(tree, gates):
    """function node id -> the lines where it calls one of ``gates``."""
    lines = {}
    for node in ast.walk(tree):
        if isinstance(node, _FUNCTIONS):
            found = [c.lineno for c in ast.walk(node) if isinstance(c, ast.Call) and _func_name(c) in gates]
            if found:
                lines[id(node)] = found
    return lines


def _references(tree, names):
    """name -> [(function node or None, line, is_call)] for every reference to it."""
    callees = set()
    refs = {}
    stack = [(tree, None)]
    while stack:
        node, owner = stack.pop()
        for child in ast.iter_child_nodes(node):
            stack.append((child, child if isinstance(child, _FUNCTIONS) else owner))
            if isinstance(child, ast.Call):
                callees.add(id(child.func))
            if isinstance(child, ast.Name) and child.id in names and isinstance(child.ctx, ast.Load):
                refs.setdefault(child.id, []).append((owner, child.lineno, id(child)))
            elif isinstance(child, ast.Attribute) and child.attr in names and isinstance(child.ctx, ast.Load):
                refs.setdefault(child.attr, []).append((owner, child.lineno, id(child)))
    return {name: [(owner, line, key in callees) for owner, line, key in items] for name, items in refs.items()}


def _home_verdicts(rel, home, estate):
    """[(site, gated)] for every sink of a home."""
    census = _censuses(estate).get(rel)
    if census is None or census.tree is None:
        return []
    gates = set(home.gates)
    trees = [census.tree] + [
        _censuses(estate)[other].tree for other in home.callers_in
        if other in estate.modules and _censuses(estate)[other].tree is not None
    ]
    gate_lines = {}
    for tree in trees:
        gate_lines.update(_gate_lines(tree, gates))
    functions = [n for n in ast.walk(census.tree) if isinstance(n, _FUNCTIONS)]
    private = {n.name for n in functions if n.name.startswith("_")}
    references = {}
    for tree in trees:
        for name, items in _references(tree, private).items():
            references.setdefault(name, []).extend(items)
    # A factory is gated when it calls a gate, or a factory already gated.
    factories = set()
    grown = True
    while grown:
        grown = False
        for n in functions:
            if n.name in home.factories and n.name not in factories and (
                    id(n) in gate_lines
                    or any(isinstance(c, ast.Call) and _func_name(c) in factories for c in ast.walk(n))):
                factories.add(n.name)
                grown = True

    def gate_before(owner, line):
        return owner is not None and any(found < line for found in gate_lines.get(id(owner), ()))

    via_callers = set()
    grown = True
    while grown:
        grown = False
        for name in sorted(private - via_callers):
            items = references.get(name, [])
            if items and all(is_call and owner is not None and (gate_before(owner, line) or owner.name in via_callers)
                             for owner, line, is_call in items):
                via_callers.add(name)
                grown = True

    def from_factory(node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            return False
        receivers = [node.func.value]
        base = node.func.value
        if isinstance(base, ast.Name):
            receivers += census.assigned.get(base.id, [])
        elif isinstance(base, ast.Attribute):
            receivers += census.assigned.get(base.attr, [])
        return any(isinstance(c, ast.Call) and _func_name(c) in factories
                   for receiver in receivers for c in ast.walk(receiver))

    verdicts = []
    for site, owner, node in census.nodes:
        gated = owner is not None and (
            owner.name in home.ungated
            or gate_before(owner, site.line)
            or owner.name in via_callers
            or from_factory(node)
        )
        verdicts.append((site, gated))
    return verdicts


def find_ungated_sinks(estate, census=None):
    """A sink in a home that sits in no gated function."""
    out = []
    for rel, home in sorted(HOMES.items()):
        if rel not in estate.modules:
            continue
        for site, gated in _home_verdicts(rel, home, estate):
            if not gated:
                where = f"in {site.function}()" if site.function else "at module level"
                out.append(f"{rel}:{site.line}: a {site.kind} sink {where} with no gate before it")
    return out


def find_count_drift(estate, census=None):
    """A home whose sink count differs from its entry, each direction named."""
    census = take_census(estate) if census is None else census
    out = []
    for rel, home in sorted(HOMES.items()):
        found = len(census.get(rel, ()))
        if found and found != home.sinks:
            way = "raise" if found > home.sinks else "lower"
            out.append(f"{rel}: {found} sink site(s), its entry says {home.sinks} ({way} the entry "
                       f"once the new count is proven gated)")
    return out


def find_ledger_growth(estate, census=None):
    """An owed module with more sinks of a kind than it owes, or a kind it did not owe."""
    census = take_census(estate) if census is None else census
    out = []
    for rel, owed in sorted(LEDGER.items()):
        for kind, n in sorted(_kinds(census.get(rel, ())).items()):
            if kind not in owed:
                out.append(f"{rel}: {n} {kind} sink(s), a kind it does not owe")
            elif n > owed[kind]:
                out.append(f"{rel}: {n} {kind} sink(s), above the {owed[kind]} it owes")
    return out


def find_stale_ledger_entries(estate, census=None):
    """An owed count above what the census finds, or an owed module with no sink."""
    census = take_census(estate) if census is None else census
    out = []
    for rel, owed in sorted(LEDGER.items()):
        sites = census.get(rel)
        if not sites:
            out.append(f"{rel}: owed, but {'it is gone' if sites is None else 'it has no sink'}; "
                       f"take it off the ledger")
            continue
        found = _kinds(sites)
        for kind, n in sorted(owed.items()):
            if found.get(kind, 0) < n:
                out.append(f"{rel}: owes {n} {kind} sink(s), the census finds {found.get(kind, 0)}; lower it")
    return out


def find_stale_exemptions(estate, census=None):
    """An exempt module with no sink, or gone."""
    census = take_census(estate) if census is None else census
    return [f"{rel}: exempt, but {'it is gone' if rel not in census else 'it has no sink'}"
            for rel in sorted(EXEMPT) if not census.get(rel)]


def find_stale_homes(estate, census=None):
    """A home with no sink, or gone."""
    census = take_census(estate) if census is None else census
    return [f"{rel}: a gate home, but {'it is gone' if rel not in census else 'it has no sink'}"
            for rel in sorted(HOMES) if not census.get(rel)]


class SuitesUnreadable(Exception):
    """A test suite the census reads for its home proofs could not be read."""


def _suites(root):
    tests = Path(root) / "tests"
    if not tests.is_dir():
        return {}
    suites, unread = {}, []
    for path in sorted(tests.glob("test_*.py")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            suites[rel] = _read_text(path)
        except (OSError, UnicodeDecodeError) as exc:
            unread.append(f"{rel}: cannot be read as UTF-8 text ({type(exc).__name__})")
    if unread:
        raise SuitesUnreadable("; ".join(unread))
    return suites


_TEST_FUNCTION = re.compile(r"^\s*(?:async\s+)?def\s+test_(\w+)", re.M)


def _suite_ids(estate):
    """suite -> every contract id its test functions answer to, read once per estate.

    A function ``test_<id>_...`` answers to each prefix of its name that a
    ``_`` follows: ``test_a_b_c`` to ``a`` and ``a_b``, as a search for
    ``test_<id>_`` at the start of a line would find it.
    """
    cached = estate.cache.get("suite_ids")
    if cached is None:
        cached = {}
        for suite, text in _suites(estate.root).items():
            ids = set()
            for name in _TEST_FUNCTION.findall(text):
                parts = name.split("_")
                ids.update("_".join(parts[:i]) for i in range(1, len(parts)))
            cached[suite] = (text, ids)
        estate.cache["suite_ids"] = cached
    return cached


def find_home_proofs_missing(estate, census=None):
    """A home contract id that names no test function, or whose suite does not name the home."""
    out = []
    suites = None
    for rel, home in sorted(HOMES.items()):
        if not home.contracts:
            out.append(f"{rel}: a gate home that names no contract")
            continue
        if suites is None:
            suites = _suite_ids(estate)
        dotted = rel[:-3].replace("/", ".")
        for contract in home.contracts:
            owners = [suite for suite, (_text, ids) in suites.items() if contract in ids]
            if not owners:
                out.append(f"{rel}: contract {contract} names no test function")
            elif not any(dotted in suites[suite][0] or rel in suites[suite][0] for suite in owners):
                out.append(f"{rel}: the suite of contract {contract} ({', '.join(owners)}) never names {dotted}")
    return out


def find_unpermitted_plugins(estate, census=None):
    """A bundled plugin with a sink and no parsed network_outbound permission, by cause."""
    census = take_census(estate) if census is None else census
    counted = {}
    for rel, sites in census.items():
        plugin = _plugin_of(rel)
        if plugin is not None and sites:
            counted[plugin] = counted.get(plugin, 0) + len(sites)
    out = []
    for plugin, n in sorted(counted.items()):
        refusal = _manifest_refusal(estate.manifests.get(plugin))
        if refusal is not None:
            out.append(f"{plugin}: {n} sink site(s) and {refusal}")
    return out


def find_unclassified_imports(estate, census=None):
    """A third-party import absent from LIBRARIES, or a first-use fetch that names no bound."""
    out = [f"{name}: a third-party import that LIBRARIES does not classify"
           for name in third_party_imports(estate) if name not in LIBRARIES]
    for name, entry in sorted(LIBRARIES.items()):
        kind, reason, bounds = entry
        if kind not in _CLASSES or not str(reason).strip():
            out.append(f"{name}: class {kind!r} with reason {reason!r} is not a classification")
        elif kind == "fetches on first use" and not bounds:
            out.append(f"{name}: fetches on first use and names nothing that bounds it")
    return out


_RUST_LEXER = None


def _rust_lexer():
    global _RUST_LEXER
    if _RUST_LEXER is None:
        path = Path(__file__).resolve().with_name("comment_only_guard.py")
        spec = importlib.util.spec_from_file_location("_egress_census_rust_lexer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RUST_LEXER = module
    return _RUST_LEXER


def _rust_code(text):
    """Rust source with every comment and literal blanked, lines kept."""
    pieces, last = [], 0
    for _kind, start, end in _rust_lexer().rust_tokens(text):
        pieces.append(text[last:start])
        pieces.append(re.sub(r"[^\n]", " ", text[start:end]))
        last = end
    pieces.append(text[last:])
    return "".join(pieces)


def _dependency_tables(manifest):
    tables = [manifest.get(key) for key in ("dependencies", "dev-dependencies", "build-dependencies")]
    tables.append(manifest.get("workspace", {}).get("dependencies"))
    for target in manifest.get("target", {}).values():
        if isinstance(target, dict):
            tables += [target.get(key) for key in ("dependencies", "dev-dependencies", "build-dependencies")]
    return [table for table in tables if isinstance(table, dict)]


def find_network_crates(estate, census=None):
    """A network crate in a Cargo.lock, tokio with net in a Cargo.toml, or std::net in code."""
    out = []
    for rel, text in sorted(estate.rust.items()):
        name = rel.rsplit("/", 1)[-1]
        if name in ("Cargo.lock", "Cargo.toml"):
            try:
                data = tomllib.loads(text)
            except tomllib.TOMLDecodeError as exc:
                out.append(f"{rel}: cannot be read ({exc})")
                continue
            if name == "Cargo.lock":
                for package in data.get("package", []):
                    if package.get("name") in _NETWORK_CRATES:
                        out.append(f"{rel}: the crate {package['name']} can reach the network")
            else:
                for table in _dependency_tables(data):
                    tokio = table.get("tokio")
                    features = set(tokio.get("features", [])) if isinstance(tokio, dict) else set()
                    if features & {"net", "full"}:
                        out.append(f"{rel}: tokio with its {sorted(features & {'net', 'full'})} feature")
        elif "net" in text:
            # A source that never spells "net" holds no std::net path; the
            # lexer is asked only about the others.
            try:
                code = _rust_code(text)
            except Exception as exc:
                out.append(f"{rel}: cannot be lexed ({exc})")
                continue
            for number, line in enumerate(code.splitlines(), 1):
                if _STD_NET.search(line):
                    out.append(f"{rel}:{number}: a std::net path")
    return out


_QUESTIONS = (
    ("violations", find_violations),
    ("ungated_sinks", find_ungated_sinks),
    ("count_drift", find_count_drift),
    ("ledger_growth", find_ledger_growth),
    ("stale_ledger_entries", find_stale_ledger_entries),
    ("stale_exemptions", find_stale_exemptions),
    ("stale_homes", find_stale_homes),
    ("home_proofs_missing", find_home_proofs_missing),
    ("unpermitted_plugins", find_unpermitted_plugins),
    ("unclassified_imports", find_unclassified_imports),
    ("network_crates", find_network_crates),
)


def find_all(estate, census=None):
    """Every question asked, in order: name -> its findings."""
    census = take_census(estate) if census is None else census
    return {name: question(estate, census) for name, question in _QUESTIONS}


def green_line(estate, census):
    home_sites = sum(len(census.get(rel, ())) for rel in HOMES)
    homes = sum(1 for rel in HOMES if rel in census)
    owed = sum(len(census.get(rel, ())) for rel in LEDGER)
    plugins = {_plugin_of(rel) for rel, sites in census.items() if sites and _plugin_of(rel)}
    gated = sum(1 for plugin in plugins if _permits_network(estate.manifests.get(plugin)))
    imports = sum(1 for name in third_party_imports(estate) if name in LIBRARIES)
    crates = sum(1 for rel in estate.rust if rel.endswith(("Cargo.toml", "Cargo.lock")))
    return (
        f"Egress census OK: {len(estate.modules)} module(s) scanned, {home_sites} sink site(s) in "
        f"{homes} gate home(s), {owed} sink(s) owed in {len(LEDGER)} module(s), {len(EXEMPT)} exempt "
        f"by name, {gated} bundled plugin(s) gated at the host, {imports} third-party import(s) "
        f"classified, {crates} crate manifest(s) read; the ledger may only shrink."
    )


def run(root):
    """The census of the repository at ``root``: its exit code, its lines, and what it read."""
    estate = read_estate(root)
    if estate.unread:
        return Result(1, [f"Egress census: FAILED -- {len(estate.unread)} part(s) of the estate could "
                          f"not be read, and what was not read was not counted:"]
                      + [f"  {reason}" for reason in estate.unread], estate, {})
    if not estate.modules:
        return Result(1, [f"Egress census: nothing was scanned under {root}: no Python module "
                          f"in {_PACKAGE_DIR}/."], estate, {})
    census = take_census(estate)
    unparsed = sorted((rel, c.unparsed) for rel, c in _censuses(estate).items() if c.unparsed)
    if unparsed:
        return Result(1, ["Egress census: FAILED -- these modules do not parse, so their sinks "
                          "cannot be counted:"]
                      + [f"  {rel}: {reason}" for rel, reason in unparsed], estate, census)
    try:
        found = find_all(estate, census)
    except SuitesUnreadable as exc:
        return Result(1, [f"Egress census: FAILED -- a suite that proves a gate home could not be "
                          f"read: {exc}"], estate, census)
    except ManifestParserMissing as exc:
        manifests = sum(1 for text in estate.manifests.values() if text is not None)
        return Result(1, [f"Egress census: FAILED -- {exc}, so the {manifests} bundled plugin "
                          f"manifest(s) were left unread: an unread manifest is neither a "
                          f"permission missing nor one granted."], estate, census)
    lines = []
    for name, findings in found.items():
        if findings:
            lines.append(f"Egress census: {name.replace('_', ' ')} ({len(findings)}):")
            lines.extend(f"  {finding}" for finding in findings)
    if lines:
        return Result(1, lines, estate, census)
    return Result(0, [green_line(estate, census)], estate, census)


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[2]
    result = run(root)
    for line in result.lines:
        print(line)
    return result.code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
