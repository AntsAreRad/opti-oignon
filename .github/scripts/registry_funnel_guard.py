#!/usr/bin/env python3
"""Registry-funnel guard: every inference request goes through the registry.

BackendRegistry is where the guarantees live. A request that passes through it
is admitted by the resource governor, may carry a schema or a tool list as an
engine option, is served against a probed VRAM capacity, and comes back with a
figure that knows where it came from. A request that reaches the client
library directly -- ``ollama.chat``, an alias of it, a name imported from it,
or a client constructed from it -- gets none of that, and there is no log line
to say so. When this guard was written, twenty-nine modules did exactly that,
at fifty-one sites; four were paid in the same block -- the funnel itself, the
two summarisers the memory block depends on, and structured output -- nine
more in the next, and the last eleven in the third convergence block. The
fourth widened what "reaching the client" means -- a receiver the client
module was bound to, a request method handed on uncalled, and the catalogue
reads ``list`` and ``show`` -- and found twenty-two more sites in sixteen
modules, all paid in that block. The ledger below is empty, and it stays
empty: a direct site anywhere outside the funnel is a violation by name.

A request that never touches the client library was still invisible: a
module that posts to the inference server's endpoint with an HTTP transport
of its own. The raw census counts those. It found six modules at nine
sites; the project trigger detector was paid in the block that widened it
and the RAG embedder in the next, once the batch had a head on the backend
contract; the red team followed, its loopback check moved onto the
backend's real endpoint. RAW_LEDGER, a ledger of its own with the same
seals and the same ratchet, is empty. The launcher's liveness probe is
exempt by name in RAW_EXEMPT, with its reason.

The client also carries Ollama's cloud search and fetch: ``web_search`` and
``web_fetch`` post a query or a URL to ollama.com under an account key, and
the paths ``/api/web_search`` and ``/api/web_fetch`` do the same through a
server that relays them. None is called; none has a head on the backend
contract. The methods are counted like a request, and the paths and a URL
on the cloud host are raw sites in any module, with or without a transport
of its own: the client posts there with its own. An exemption or a raw
ledger entry excuses local endpoints only; a cloud site is refused by name
in every module outside the funnel.

The census follows the client through a submodule import, a star import,
``import_module`` or ``__import__`` with a constant (however the function
is reached or renamed), a ``sys.modules`` lookup, and ``getattr``; through
an assignment, an unpacking, a walrus, a ``for`` or ``with`` target, a
parameter default, an argument passed to a function, a method or a class
of the same module, a call given it, a container or a comprehension, a
class attribute, a lambda or a function that hands it back, and any
attribute chain rooted at one of those; what a request returns is not the
client. Names are not scoped: a
name bound to the client anywhere in a module is the client everywhere in
it, which can only over-count. A path is read as a server would route it:
bytes decoded, percent encoding, surrounding spaces, the query and the
fragment, dot segments and repeated slashes resolved; a host is read with
its Unicode full stops folded.

What the census does not see, said here rather than claimed covered: a name
assembled at run time, or read through ``vars``, ``__dict__``,
``globals``, ``operator.attrgetter``, ``methodcaller``, ``exec`` or
``eval``; a client object received from another module, or passed to a
function of another module; a class or other capitalised name imported
from the package that is in fact a submodule; a local endpoint, or a cloud
URL, assembled from pieces none of which names an endpoint or the cloud
host; a local endpoint posted with a transport the list below does not
name, such as ``urllib3``, ``primp`` or ``subprocess`` (the context manager
runs the ``ollama`` CLI, which reaches the server's show endpoint; the egress
census guard counts it).

RATCHET, in the shape of the isolation-seal guard and for the same reason: a
ratchet that only counts is a ratchet on the count. Every owed module carries
the digest of its text as the debt was enumerated. An owed module that changes
while still calling the client directly no longer matches its seal and becomes
a violation: touch it, and you migrate it. The debt is frozen as it was found,
it can be paid, and it cannot grow -- not in modules, and not in lines.

The census is taken on the syntax tree, never on the text. A docstring that
mentions the client is not a request, and a guard that charged for prose would
be green or red for reasons unrelated to what leaves the process.

One module is exempt by name: ``opti_oignon/inference_backend.py`` is where the
registry's own Ollama backend talks to the client. That is the funnel; it is
not a bypass of the funnel.

Three questions, three answers, with disjoint domains so no one can cover for
another:

  * ``find_violations``           -- a direct caller nobody owes for.
  * ``find_broken_seals``         -- an owed module whose bytes moved.
  * ``find_stale_ledger_entries`` -- an owed module that migrated, or vanished.

The raw census answers the same three for its own ledger, and a fourth,
``find_cloud_violations``, for a module an exemption or the raw ledger
excuses that spells the cloud all the same.

The helpers are pure and import-safe; ``main`` scans the package and exits
non-zero on any finding. Usage: ``registry_funnel_guard.py [REPO_ROOT]``.
"""

import ast
import hashlib
import posixpath
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

_PACKAGE_DIR = "opti_oignon"
_FUNNEL = "opti_oignon/inference_backend.py"

# What counts as reaching the client: a request method, a read of the
# engine's loaded set, a read of the model catalogue, or a client object
# from which requests are made. ``ps`` joined the set when the loaded set
# became a head on the backend contract; ``list`` and ``show`` joined it in
# the fourth convergence block, when the catalogue went through
# ``list_models`` and ``model_info``: a module that reads any of them from
# the client bypasses the head that answers it. Model management --
# ``pull``, ``delete``, ``copy``, ``create``, ``push`` -- has no head on the
# contract and is not counted: whether it belongs in the funnel is a
# decision the guard does not take on its own.
#
# Ollama's cloud search and fetch post a query to ollama.com under an
# account key. They have no head on the backend contract and are counted
# like a request: a module that reaches them outside the funnel is refused
# by name. Model management is still not counted.
_CLIENT_CALLS = frozenset({
    "chat", "generate", "embeddings", "embed", "ps", "list", "show",
    "web_search", "web_fetch",
})
_CLIENT_CLASSES = frozenset({"Client", "AsyncClient"})

# The client package by name, as ``import_module`` and ``__import__`` take it,
# and the modules those two functions are imported from by name.
_CLIENT_PACKAGE = "ollama"
_IMPORT_FUNCTIONS = frozenset({"import_module", "__import__"})
_IMPORT_PROVIDERS = frozenset({"importlib", "builtins"})

# Debt that predates the funnel: repo-relative module -> sha256 of its text as
# the debt was enumerated. MAY ONLY SHRINK, and no entry may move. It is
# empty since the third convergence block paid the last eleven modules: a
# direct site anywhere outside the funnel is now a violation by name, and
# nothing may be added here to make one tolerable.
LEDGER = {
}


# What counts as posting to the inference server without the client: a
# string literal ending with one of its request, catalogue or loaded-set
# endpoints, in a module that imports an HTTP transport. Model management
# endpoints are left out by the same decision as the client's methods. A
# module without a transport cannot send what it spells, and the
# application's own routes share these paths: they are not requests.
#
# The cloud search and fetch paths are the exception: the application has
# no route that spells them, and the client posts there with a transport of
# its own, so they count in any module, and so does a URL on the cloud host
# whose path is empty or under ``/api`` or ``/v1`` -- a base to post from,
# not a link to a page. No exemption and no raw ledger entry excuses them.
_CLOUD_ENDPOINTS = ("/api/web_search", "/api/web_fetch")
_CLOUD_HOST = "ollama.com"
# Full stops a host name may be written with, read as the ASCII one (IDNA).
_FULL_STOPS = {0x3002: ".", 0xFF0E: ".", 0xFF61: "."}
_CLOUD_API_PREFIXES = ("/api", "/v1")
_URL_IN_TEXT = re.compile(r"[a-z][a-z0-9+.\-]*://[^\s'\"<>]+")
_RAW_ENDPOINTS = (
    "/api/chat", "/api/generate", "/api/embed", "/api/embeddings",
    "/api/tags", "/api/ps", "/api/show",
) + _CLOUD_ENDPOINTS
_HTTP_TRANSPORTS = frozenset({"requests", "httpx", "urllib", "http", "aiohttp"})

# Raw debt found when the census was widened: repo-relative module -> sha256
# of its text. MAY ONLY SHRINK, and no entry may move. Empty since the red
# team went through the registry with its loopback check on the backend's
# real endpoint.
RAW_LEDGER = {
}

# Modules that spell an endpoint and are not requests to the model, each
# with its reason. An exemption is a decision taken by name, never a place
# to put a module that should migrate; one whose module no longer posts is
# stale and must come off.
RAW_EXEMPT = {
    "opti_oignon/ui.py": (
        "the launcher's liveness probe asks whether the server process "
        "answers before it starts the application; that is not an inference "
        "request, and routing it would make the launcher build the registry"
    ),
}


def digest(text):
    """The seal of a module, taken on the text this guard reads."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _names_in(node):
    """Every bare name read anywhere inside an expression."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _is_client_module(name):
    """The client package or one of its submodules, by dotted name."""
    return isinstance(name, str) and (name == _CLIENT_PACKAGE or name.startswith(_CLIENT_PACKAGE + "."))


def _may_be_submodule(name):
    """A name imported from the package that may be one of its modules.

    A private name may (every submodule of the installed client is
    private), and so may a lowercase one that is not counted, as a public
    submodule would be; a capitalised name -- a class, an exception -- is
    not followed.
    """
    return name.startswith("_") or name.islower()


def _target_names(target):
    """The names and the attributes an assignment target binds.

    ``self._c`` binds the attribute ``_c``, never ``self``; ``d[0]`` and
    ``self.d[0]`` bind the container ``d``; an unpacking binds each element.
    """
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


def _parameters(args):
    """The positional parameters, then the keyword-only ones, of a signature."""
    return list(getattr(args, "posonlyargs", [])) + list(args.args), list(args.kwonlyargs)


def count_sites(text):
    """How many times the text reaches the client library directly.

    Counted on the syntax tree: a request method or a client class reached
    on the module, on any alias of it, on a name or an attribute the module
    was bound to, or imported by name from it. A reference counts whether or
    not it is called: a request method handed on as a callable is a route to
    the client. Prose never counts. A text that does not parse is counted as
    zero here -- the syntax tier owns that failure and reports it by name.

    Binding is followed until nothing new binds: ``self._c = injected or
    _ollama`` makes ``_c`` a client attribute, and ``c = ollama.Client(host)``
    makes ``c`` a client name. The receiver is what the fourth convergence
    block found the census blind to, with a ``chat`` request behind it. So
    is an unpacking, a walrus, a ``for`` or ``with`` target, a parameter
    whose default is the client or to which a function, a method or a class
    of the same module is passed it, a container or a comprehension holding
    it, a call given it (a wrapper may hand it back), and a name assigned in
    a class body, which is read as an attribute as well.

    A function, a method or a lambda that returns the client -- or what
    another such function returns -- is followed the same way: a request
    made on its result is a site, and so is a name bound to its result. A
    helper is not a disguise. An attribute chain rooted at anything the
    census follows is the client too: ``ollama._client``, ``o._client`` or
    ``import_module("ollama")._client``.

    The client's submodules are the client too: ``import ollama._client``
    binds the package, ``import ollama._client as c`` binds ``c``, ``from
    ollama import _client`` binds ``_client``, and a counted name imported
    from a submodule is a bare name. A star import from the package or a
    submodule binds every counted name. A submodule import that reaches
    nothing counted is not a site. ``import_module`` and ``__import__`` with
    a constant naming the package return the client, however they are
    reached: ``importlib.__import__``, ``builtins.__import__``, a renamed
    import, a keyword argument. So does a ``sys.modules`` lookup of it.
    ``getattr`` on the client is a site when the name it reads is counted,
    or when it cannot be read at all; ``getattr(ollama, "pull")`` is not,
    and what ``getattr`` returns from the client is followed like any
    attribute.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    counted = _CLIENT_CALLS | _CLIENT_CLASSES
    aliases = set()
    bare = set()
    importers = set(_IMPORT_FUNCTIONS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_client_module(alias.name):
                    # ``import ollama.x`` binds the package; ``as y`` binds the submodule.
                    aliases.add(alias.asname or _CLIENT_PACKAGE)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            module = node.module or ""
            if _is_client_module(module):
                for alias in node.names:
                    if alias.name == "*":
                        bare |= counted
                    elif alias.name in counted:
                        bare.add(alias.asname or alias.name)
                    elif module == _CLIENT_PACKAGE and _may_be_submodule(alias.name):
                        aliases.add(alias.asname or alias.name)
            elif module in _IMPORT_PROVIDERS:
                for alias in node.names:
                    if alias.name in _IMPORT_FUNCTIONS:
                        importers.add(alias.asname or alias.name)

    def _imports(node):
        """A call importing the client by name, or a ``sys.modules`` lookup of it."""
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id in importers) or (
                isinstance(func, ast.Attribute) and func.attr in _IMPORT_FUNCTIONS
            ):
                arg = node.args[0] if node.args else next(
                    (kw.value for kw in node.keywords if kw.arg == "name"), None
                )
                return isinstance(arg, ast.Constant) and _is_client_module(arg.value)
            if (
                isinstance(func, ast.Attribute) and func.attr == "get" and node.args
                and isinstance(func.value, ast.Attribute) and func.value.attr == "modules"
            ):
                return isinstance(node.args[0], ast.Constant) and _is_client_module(node.args[0].value)
            return False
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "modules":
            return isinstance(node.slice, ast.Constant) and _is_client_module(node.slice.value)
        return False

    dynamic = any(_imports(n) for n in ast.walk(tree))
    if not aliases and not bare and not dynamic:
        return 0
    attrs = set()
    returners = set()
    class_body = {
        id(stmt) for node in ast.walk(tree) if isinstance(node, ast.ClassDef) for stmt in node.body
    }
    # Callable name -> (signature, parameters a call does not pass): a
    # function or a method by its name, a class by its name and its
    # ``__init__``, which receives ``self`` besides.
    signatures = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures.setdefault(node.name, []).append((node.args, None))
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
                    signatures.setdefault(node.name, []).append((stmt.args, 1))

    def _returns_client(node):
        """A call to a function, a method or a lambda that hands back the client."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Name) and func.id in returners) or (
            isinstance(func, ast.Attribute) and func.attr in returners
        )

    def _receiver(value):
        """What a counted name read on it reaches the client through."""
        if isinstance(value, ast.Name):
            return value.id in aliases
        if isinstance(value, ast.Attribute):
            return value.attr in attrs or _receiver(value.value)
        if isinstance(value, ast.Subscript):
            return _imports(value) or _receiver(value.value)
        if isinstance(value, ast.NamedExpr):
            return _receiver(value.value)
        if isinstance(value, ast.Call):
            if _imports(value) or _returns_client(value):
                return True
            func = value.func
            return isinstance(func, ast.Name) and func.id == "getattr" and bool(value.args) and _receiver(value.args[0])
        return False

    def _is_client(value):
        """True when an expression evaluates to the client, a client object, or a container holding one.

        What a request returns is not the client: ``c = ollama.Client(h)``
        binds ``c``, ``r = ollama.chat(...)`` does not bind ``r``. A call
        given the client -- a wrapper, a context manager, a cast -- may hand
        it back, and is read as doing so.
        """
        if value is None:
            return False
        if isinstance(value, ast.Name):
            return value.id in aliases or value.id in bare
        if isinstance(value, ast.Call):
            if _receiver(value):
                return True
            if any(_is_client(a) for a in list(value.args) + [kw.value for kw in value.keywords]):
                return True
            func = value.func
            if isinstance(func, ast.Name):
                return func.id in bare and func.id in _CLIENT_CLASSES
            return isinstance(func, ast.Attribute) and func.attr in _CLIENT_CLASSES and _is_client(func.value)
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return _is_client(value.elt)
        if isinstance(value, ast.DictComp):
            return _is_client(value.value)
        if isinstance(value, ast.BoolOp):
            return any(_is_client(v) for v in value.values)
        if isinstance(value, ast.IfExp):
            return _is_client(value.body) or _is_client(value.orelse)
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return any(_is_client(e) for e in value.elts)
        if isinstance(value, ast.Dict):
            return any(_is_client(v) for v in value.values)
        if isinstance(value, (ast.Starred, ast.Await)):
            return _is_client(value.value)
        return _receiver(value)

    grown = True

    def _bind(names=(), attributes=(), returning=()):
        nonlocal grown
        for name in names:
            if name not in aliases:
                aliases.add(name)
                grown = True
        for name in attributes:
            if name not in attrs:
                attrs.add(name)
                grown = True
        for name in returning:
            if name not in returners:
                returners.add(name)
                grown = True

    def _bind_target(target, value, in_class=False):
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) \
                and len(target.elts) == len(value.elts) \
                and not any(isinstance(e, ast.Starred) for e in target.elts + value.elts):
            for sub_target, sub_value in zip(target.elts, value.elts):
                _bind_target(sub_target, sub_value, in_class)
            return
        names, attributes = _target_names(target)
        if isinstance(value, ast.Lambda):
            if _is_client(value.body):
                _bind(returning=names | attributes)
            return
        if not _is_client(value):
            return
        _bind(names, attributes | (names if in_class else set()))

    def _bind_parameters(args, values, keywords, offset=0):
        positional, keyword_only = _parameters(args)
        for index, value in enumerate(values):
            if index + offset < len(positional) and _is_client(value):
                _bind({positional[index + offset].arg})
        for name, value in keywords:
            for param in positional + keyword_only:
                if param.arg == name and _is_client(value):
                    _bind({param.arg})

    while grown:
        grown = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in returners:
                if any(isinstance(r, ast.Return) and _is_client(r.value) for r in ast.walk(node)):
                    _bind(returning={node.name})
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                positional, keyword_only = _parameters(node.args)
                defaults = list(node.args.defaults)
                for param, default in zip(positional[len(positional) - len(defaults):], defaults):
                    if _is_client(default):
                        _bind({param.arg})
                for param, default in zip(keyword_only, node.args.kw_defaults):
                    if _is_client(default):
                        _bind({param.arg})
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    _bind_target(target, node.value, id(node) in class_body)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                _bind_target(node.target, node.value, id(node) in class_body)
            elif isinstance(node, ast.NamedExpr):
                _bind_target(node.target, node.value)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                _bind_target(node.target, node.iter)
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                _bind_target(node.optional_vars, node.context_expr)
            elif isinstance(node, ast.Call):
                keywords = [(kw.arg, kw.value) for kw in node.keywords if kw.arg]
                func = node.func
                if isinstance(func, ast.Lambda):
                    _bind_parameters(func.args, node.args, keywords)
                elif isinstance(func, (ast.Name, ast.Attribute)):
                    name = func.id if isinstance(func, ast.Name) else func.attr
                    for args, skipped in signatures.get(name, ()):
                        positional = _parameters(args)[0]
                        offset = skipped if skipped is not None else (
                            1 if isinstance(func, ast.Attribute) and positional
                            and positional[0].arg in ("self", "cls") else 0
                        )
                        _bind_parameters(args, node.args, keywords, offset)

    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in counted:
            if _receiver(node.value):
                n += 1
        elif isinstance(node, ast.Name) and node.id in bare and isinstance(node.ctx, ast.Load):
            n += 1
        elif (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
            and len(node.args) >= 2 and _receiver(node.args[0])
        ):
            name = node.args[1]
            if not isinstance(name, ast.Constant) or name.value in counted:
                # A name that cannot be read is charged: the census cannot
                # tell which method it reaches.
                n += 1
    return n


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def _constant_text(node):
    """The text a constant carries, bytes decoded as a transport would, or None."""
    if not isinstance(node, ast.Constant):
        return None
    if isinstance(node.value, bytes):
        return node.value.decode("latin-1")
    return node.value if isinstance(node.value, str) else None


def _folded(value):
    """Percent encoding decoded, compatibility forms and full stops folded."""
    return unicodedata.normalize("NFKC", unquote(value)).translate(_FULL_STOPS)


def _routed(path):
    """A path with its dot segments and repeated slashes resolved, no trailing slash."""
    if not path:
        return ""
    routed = posixpath.normpath(path)
    return "" if routed == "." else routed.rstrip("/")


def _endpoint_path(value):
    """The path a string constant names, read as a server would route it.

    Percent encoding is decoded and surrounding spaces dropped, then the
    query and the fragment are cut at the first ``?`` or ``#``; after the
    scheme and the host, dot segments and repeated slashes are resolved and
    a trailing slash dropped: ``/api/x/../web%5Fsearch#x `` is
    ``/api/web_search``.
    """
    text = _folded(value).strip()
    for mark in ("?", "#"):
        text = text.split(mark, 1)[0]
    text = text.strip()
    if "://" in text:
        origin_end = text.index("://") + 3
        slash = text.find("/", origin_end)
        if slash < 0:
            return text
        return text[:slash] + _routed(text[slash:])
    return _routed(text)


def _is_cloud_host(host):
    host = (host or "").lower().rstrip(".")
    return host == _CLOUD_HOST or host.endswith("." + _CLOUD_HOST)


def _names_cloud_base(value):
    """True when a constant is a URL on the cloud host to post from.

    A URL on the host whose path is empty or under ``/api`` or ``/v1``, or
    the host alone; a link to a page on it -- ``https://ollama.com/download``
    in a message -- is not a request. In a text with spaces, each URL in it
    is read on its own.
    """
    text = _folded(value).strip().lower()
    candidates = _URL_IN_TEXT.findall(text) if any(ch.isspace() for ch in text) else [text]
    for candidate in candidates:
        candidate = candidate.split("?", 1)[0].split("#", 1)[0]
        if "://" in candidate:
            try:
                parts = urlsplit(candidate)
                host, path = parts.hostname, parts.path
            except ValueError:
                if _CLOUD_HOST in candidate:
                    return True
                continue
        else:
            host_part, _slash, rest = candidate.partition("/")
            host = host_part.rsplit("@", 1)[-1].split(":", 1)[0]
            path = "/" + rest if rest else ""
        path = _routed(path)
        if _is_cloud_host(host) and (
            path == "" or any(path == p or path.startswith(p + "/") for p in _CLOUD_API_PREFIXES)
        ):
            return True
    return False


def _raw_census(text):
    """(cloud sites, local sites) a module spells; see ``count_raw_sites``."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0, 0
    transports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            transports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            transports.add(node.module.split(".")[0])
    has_transport = bool(transports & _HTTP_TRANSPORTS)
    prose = _docstring_nodes(tree)
    cloud = local = 0
    for node in ast.walk(tree):
        value = _constant_text(node)
        if value is None or id(node) in prose:
            continue
        path = _endpoint_path(value)
        if path.endswith(_CLOUD_ENDPOINTS) or _names_cloud_base(value):
            cloud += 1
        elif has_transport and path.endswith(_RAW_ENDPOINTS):
            local += 1
    return cloud, local


def count_raw_sites(text):
    """How many endpoint literals of the inference server a module spells.

    Counted on the syntax tree: every string or bytes constant, the pieces
    of an f-string included, whose path ends with one of the endpoints, in a
    module that imports an HTTP transport. A cloud search or fetch path, and
    a URL on the cloud host to post from, counts in any module, with or
    without a transport. A constant counts once. A docstring is prose and
    never counts. A text that does not parse counts zero; the syntax tier
    owns that failure.
    """
    cloud, local = _raw_census(text)
    return cloud + local


def count_cloud_sites(text):
    """How many of those name the cloud search or fetch, or the cloud host."""
    return _raw_census(text)[0]


def posts_raw(name, text):
    """True when the module posts to the inference server itself, is not the funnel, and is not exempt."""
    return name != _FUNNEL and name not in RAW_EXEMPT and count_raw_sites(text) > 0


def reaches_cloud(name, text):
    """True when a module outside the funnel spells the cloud search or fetch, whatever excuses it."""
    return name != _FUNNEL and count_cloud_sites(text) > 0


def find_stale_raw_exemptions(files):
    """Exempt names that no longer spell a local endpoint, or that vanished.

    Read on the local endpoints alone: a cloud site is never what an
    exemption was granted for.
    """
    seen = dict(files)
    return sorted(name for name in RAW_EXEMPT if name not in seen or _raw_census(seen[name])[1] == 0)


def find_cloud_violations(files):
    """Modules an exemption or the raw ledger excuses that spell the cloud all the same.

    Disjoint from ``find_raw_violations``, which answers for every module
    nothing excuses: together they refuse a cloud site in every module
    outside the funnel.
    """
    return sorted(
        name for name, text in files
        if (name in RAW_EXEMPT or name in RAW_LEDGER) and reaches_cloud(name, text)
    )


def find_raw_violations(files):
    """Modules that post raw and that the raw ledger does not owe for."""
    return sorted(name for name, text in files if posts_raw(name, text) and name not in RAW_LEDGER)


def find_raw_broken_seals(files):
    """Raw-owed modules whose bytes moved while they still post."""
    seen = dict(files)
    return sorted(
        name for name, sealed in RAW_LEDGER.items()
        if name in seen and posts_raw(name, seen[name]) and digest(seen[name]) != sealed
    )


def find_stale_raw_entries(files):
    """Raw ledger names that no longer post, or that vanished."""
    seen = dict(files)
    return sorted(name for name in RAW_LEDGER if name not in seen or not posts_raw(name, seen[name]))


def calls_directly(name, text):
    """True when the module reaches the client and is not the funnel itself."""
    return name != _FUNNEL and count_sites(text) > 0


def find_violations(files):
    """Modules that call the client directly and that the ledger does not owe for.

    ``files`` is an iterable of (repo-relative name, text) pairs. An owed name
    is passed over HERE and answered for by the seal below; the domains are
    disjoint by construction.
    """
    return sorted(
        name for name, text in files
        if calls_directly(name, text) and name not in LEDGER
    )


def find_broken_seals(files):
    """Owed modules whose bytes no longer match their seal, still calling directly.

    The ratchet's tooth. A module that migrated is NOT broken -- migrating is
    what the seal asks for; it becomes stale instead and comes off the ledger.
    """
    seen = dict(files)
    return sorted(
        name for name, sealed in LEDGER.items()
        if name in seen
        and calls_directly(name, seen[name])
        and digest(seen[name]) != sealed
    )


def find_stale_ledger_entries(files):
    """Ledger names that no longer call the client directly, or that vanished.

    An entry that has been paid must come OFF the list, or the debt count stops
    meaning anything and a later regression could hide behind it.
    """
    seen = dict(files)
    return sorted(
        name for name in LEDGER
        if name not in seen or not calls_directly(name, seen[name])
    )


def _estate(root):
    package = Path(root) / _PACKAGE_DIR
    return [
        (p.relative_to(root).as_posix(), p.read_text(encoding="utf-8", errors="ignore"))
        for p in sorted(package.rglob("*.py"))
    ]


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[2]
    files = _estate(root)
    if not files:
        # A guard that scanned nothing has proven nothing. The zero it would
        # otherwise print is the silent kind this repository treats as a
        # defect, so an empty estate is a refusal, not a pass.
        print(f"Registry funnel: no Python module found under {root / _PACKAGE_DIR}; nothing was scanned.")
        return 1
    violations = find_violations(files)
    broken = find_broken_seals(files)
    stale = find_stale_ledger_entries(files)
    raw_violations = find_raw_violations(files)
    raw_broken = find_raw_broken_seals(files)
    raw_stale = find_stale_raw_entries(files)
    raw_exempt_stale = find_stale_raw_exemptions(files)
    cloud_violations = find_cloud_violations(files)

    if violations:
        print("Registry-funnel violations -- these modules reach the client")
        print("library directly instead of going through BackendRegistry:")
        for name in violations:
            print(f"  {name}")
        print()
        print("A request that bypasses the registry is not admitted by the")
        print("governor, cannot carry a schema or a tool list, and comes back")
        print("with no provenance. Route it through the registry.")
    if broken:
        print("Broken seals -- the ledger owes for these modules and their bytes")
        print("have moved while they still call the client directly:")
        for name in broken:
            print(f"  {name}")
        print()
        print("Touch an owed module and you migrate it. The debt may be")
        print("carried; it may not be added to.")
    if stale:
        print("Stale ledger entries -- these have been paid or have vanished and")
        print("must be removed from LEDGER so the debt count stays honest:")
        for name in stale:
            print(f"  {name}")

    if raw_violations:
        print("Raw HTTP violations -- these modules post to the inference")
        print("server's endpoint with a transport of their own:")
        for name in raw_violations:
            print(f"  {name}")
        print()
        print("A request posted around the registry gets none of its")
        print("guarantees. Route it through the registry.")
    if raw_broken:
        print("Broken raw seals -- the raw ledger owes for these modules and")
        print("their bytes have moved while they still post:")
        for name in raw_broken:
            print(f"  {name}")
    if raw_stale:
        print("Stale raw ledger entries -- paid or vanished; remove them from")
        print("RAW_LEDGER:")
        for name in raw_stale:
            print(f"  {name}")

    if cloud_violations:
        print("Cloud search and fetch violations -- an exemption or a raw ledger")
        print("entry excuses local endpoints only, and these modules spell")
        print("Ollama's cloud web_search or web_fetch, or the cloud host:")
        for name in cloud_violations:
            print(f"  {name}")
        print()
        print("A query posted to the cloud leaves the machine under an account")
        print("key. No exemption covers it.")

    if raw_exempt_stale:
        print("Stale raw exemptions -- these no longer spell an endpoint, or")
        print("vanished; remove them from RAW_EXEMPT:")
        for name in raw_exempt_stale:
            print(f"  {name}")

    if (
        violations or broken or stale or raw_violations or raw_broken or raw_stale
        or raw_exempt_stale or cloud_violations
    ):
        return 1

    seen = dict(files)
    raw_sites = sum(count_raw_sites(seen[name]) for name in RAW_LEDGER if name in seen)
    exempt = ", ".join(sorted(RAW_EXEMPT)) or "none"
    print(
        f"Raw HTTP: {len(RAW_LEDGER)} module(s) owed, {raw_sites} raw site(s) "
        f"between them, none outside the raw ledger; {len(RAW_EXEMPT)} exempt "
        f"by name ({exempt}). It is sealed: it may only shrink, and an owed "
        f"module that changes must migrate."
    )
    if not LEDGER:
        # The green with its denominator: how many modules were read to find
        # no direct site, so that a scan of the wrong tree cannot pass as a
        # paid debt.
        print(
            f"Registry funnel OK: 0 module(s) owed, {len(files)} module(s) "
            f"scanned and none reaches the client outside the funnel. The "
            f"ledger is empty and sealed: it may only shrink, so a new direct "
            f"site is a violation by name."
        )
        return 0
    sites = sum(count_sites(seen[name]) for name in LEDGER if name in seen)
    print(
        f"Registry funnel OK: {len(LEDGER)} module(s) owed, {sites} direct "
        f"site(s) between them, none outside the ledger. The ledger is sealed: "
        f"it may only shrink, and an owed module that changes must migrate."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
