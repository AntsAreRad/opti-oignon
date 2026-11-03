"""The Host gate: the API answers only the names this machine is served as, and takes no state-changing request
from a page of another site.

A page of another site can rebind its own name to this machine's loopback
address (DNS rebinding): the browser then sends the page's requests to the
local API under the page's own origin -- same-origin for the browser, so
CORS never applies -- with the page's name in the Host header. In
single-user Daily mode those requests are the local user's. The gate
refuses every request, and every websocket handshake, whose Host header is
not exactly one name this machine is served as: a loopback name, or a name
listed under ``api.hosts`` of ``security.yaml``. Registered outside every
other middleware, a refused request never reaches the security headers,
CORS, the authentication, the CSRF check or a route.

A page of any site can also send a request to the loopback address itself,
a loopback Host: a GET, a HEAD, or a POST with a form, multipart or plain
text body, which the browser sends without asking (no preflight) and whose
answer only the page cannot read; and it can open a websocket, which no
browser asks CORS for. The gate refuses a state-changing request (any method
but GET, HEAD and OPTIONS), and a websocket handshake, that a browser sends
for a page of another site. A page is of this site when its ``Origin`` has
the request's scheme (the scope's: ``https`` for ``https`` and ``wss``)
and the name and the port of its Host header (the scheme's default port
when none is written; ``https`` for a name other than a loopback one: a
page served under a listed name over plain ``http`` changes nothing,
remote access is to be served over TLS), or when it is exactly one of the
origins ``cors.origins`` of ``security.yaml`` lists (``listed_origins``).
A proxy that ends TLS before the API passes the scheme on
(``X-Forwarded-Proto``, from an address uvicorn trusts), or its pages'
origin is listed.
A page of another port, of another name or of another scheme of this very
machine -- a dev server, a local app -- is another site, and so is
``Origin: null``; with no Origin, a ``Sec-Fetch-Site`` other than
``same-origin`` or ``none`` is another site's. A client that sends neither
header is no current browser (the CLI, curl) and passes. A GET with an
effect is not covered here (a census, BACKLOG 218).

The same list is the only one CORS answers (``cors_settings``,
``ListedOriginsCORS``): a page of this site needs no CORS, and a page of an
origin not listed gets no CORS header at all, so no other page of this
machine is let read (the frontend's dev server answers no CORS of its own).

The grammar and the loopback names are the garden's routes'
(``routes_allium.host_origin``): exactly one raw Host header, never
``X-Forwarded-Host``; a bracketed IPv6 literal or dot-separated labels with
no trailing dot, matched lowercase; a port from 1 to 65535, compared here
with the Origin's (the garden compares names only).
"""
from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
from pathlib import Path

import yaml
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse

logger = logging.getLogger(__name__)

# The names the API always answers.
LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
# A name as a request may carry it: a bracketed IPv6 literal, or dot-separated labels with no trailing dot.
_NAME = r"(?P<name>\[[0-9a-f:]+\]|[a-z0-9-]+(?:\.[a-z0-9-]+)*)"
_PORT = r"(?::(?P<port>[0-9]{1,5}))?"
_HOST = re.compile(_NAME + _PORT)
_ORIGIN = re.compile(r"(?P<scheme>https?)://" + _NAME + _PORT)
# The methods that change nothing, compared exactly as the router compares them: every other spelling is asked where
# it comes from.
_SAFE = frozenset({"GET", "HEAD", "OPTIONS"})
# The Sec-Fetch-Site values of a request a page of this site, or the user, made.
_SAME_SITE = ("same-origin", "none")
# A name ``api.hosts`` may list: an exact lowercase name, or a bracketed IPv6 literal.
_LISTED = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)*|\[[0-9a-f:]+\]")
SECURITY_YAML = Path(__file__).resolve().parent.parent / "config" / "security.yaml"
# The port a browser leaves out of an Origin, and a Host header, for each scheme.
_DEFAULT_PORT = {"http": "80", "https": "443"}
# The scheme of the page a request of each scheme comes from, when it is of this site.
_PAGE_SCHEME = {"http": "http", "ws": "http", "https": "https", "wss": "https"}
# The scope key the gate sets on a websocket handshake whose Origin it let through, for the route's own check.
ORIGIN_CHECKED = "opti_oignon.origin_checked"
# The variable that lists the origins when security.yaml lists none.
ORIGINS_ENV = "OPTI_CORS_ORIGINS"
# The refusals' bodies: fixed words, never the name or the origin refused.
REFUSED = b'{"detail":"This server does not answer that host name."}'
REFUSED_SITE = b'{"detail":"This server takes no change from that page."}'
# How many refused names are logged by name (each once); past them, one line says the count every _LOG_EVERY.
_LOG_NAMES = 32
_LOG_EVERY = 100


def _is_name(entry) -> bool:
    """Whether ``entry`` is a name ``api.hosts`` may list, by the garden's rule (``routes_allium._hosts_of``): an
    exact lowercase name or a bracketed IPv6 literal, with no port, scheme or wildcard, neither an unspecified address
    nor an IPv4 address spelled otherwise than as its dotted quad; and, stricter than the garden, an IPv6 literal in
    its canonical spelling, the only one a browser sends."""
    if not isinstance(entry, str) or not _LISTED.fullmatch(entry):
        return False
    bracketed = entry.startswith("[")
    try:
        address = ipaddress.ip_address(entry[1:-1] if bracketed else entry)
    except ValueError:
        address = None
    if address is None and bracketed:
        return False
    # A bracketed literal in its canonical spelling, the one a browser sends: another spelling would never match
    if bracketed and entry != f"[{address.compressed}]":
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    if address is not None and (address.is_unspecified or (mapped is not None and mapped.is_unspecified)):
        return False
    if not bracketed:
        try:
            packed = socket.inet_aton(entry)
        except (OSError, ValueError):
            packed = None
        if packed is not None and socket.inet_ntoa(packed) != entry:
            return False
    return True


def listed_hosts(path: Path = SECURITY_YAML) -> tuple[str, ...]:
    """The names ``api.hosts`` of ``security.yaml`` lists, read once when the app starts.

    ``()`` when the file or the key is absent; ``()`` and a warning when the
    file or the list cannot be read whole (one entry that is not a name by
    the garden's rule, ``_is_name``): the loopback names alone are then
    answered, never more.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        return ()
    except Exception:  # noqa: BLE001 -- any file the parser cannot read whole lists no name
        logger.warning("security.yaml cannot be read: the API answers the loopback names only")
        return ()
    section = data.get("api") if isinstance(data, dict) else None
    if not isinstance(section, dict) or "hosts" not in section:
        return ()
    raw = section["hosts"]
    if isinstance(raw, list) and all(_is_name(name) for name in raw):
        return tuple(raw)
    logger.warning("api.hosts of security.yaml is not a list of names: the API answers the loopback names only")
    return ()


def _origin(text: str):
    """``(scheme, name, port)`` of an origin by the grammar, lowercase, the port written out (the scheme's default
    when none is); None when ``text`` is not one, or its port is out of range."""
    match = _ORIGIN.fullmatch(text.lower())
    if match is None:
        return None
    scheme, port = match.group("scheme"), match.group("port")
    if port is not None and not 1 <= int(port) <= 65535:
        return None
    return scheme, match.group("name"), str(int(port)) if port is not None else _DEFAULT_PORT[scheme]


def _spelled(origin) -> str:
    """An origin as a browser sends it: the scheme's default port left out."""
    scheme, name, port = origin
    return f"{scheme}://{name}" if port == _DEFAULT_PORT[scheme] else f"{scheme}://{name}:{port}"


def _is_origin(entry) -> bool:
    """Whether ``entry`` is an origin ``cors.origins`` may list: exactly ``scheme://name[:port]``, lowercase, with
    no path, no trailing slash and no wildcard; ``http`` for a loopback name only, ``https`` for any name
    ``api.hosts`` may list or a loopback one."""
    if not isinstance(entry, str) or entry != entry.lower():
        return False
    origin = _origin(entry)
    if origin is None:
        return False
    scheme, name, _port = origin
    if name in LOOPBACK:
        return True
    return scheme == "https" and _is_name(name)


def _entries(raw):
    """The entries of a ``cors.origins`` value or of the variable: a list of strings, or one string of origins
    separated by commas; each stripped, the empty ones dropped; none for no value. None for a value of another
    type."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.split(",")
    elif not isinstance(raw, list) or not all(isinstance(entry, str) for entry in raw):
        return None
    return [entry.strip() for entry in raw if entry.strip()]


def listed_origins(path: Path = SECURITY_YAML, environ=None) -> tuple[str, ...]:
    """The origins ``cors.origins`` of ``security.yaml`` lists, or, when it lists none, the variable
    ``OPTI_CORS_ORIGINS``: read once when the app starts, each as a browser sends it.

    ``("*",)`` when the list holds ``*``: any origin for CORS, without
    credentials, and no origin for the gate. ``()`` when neither lists one;
    ``()`` and a warning when the file cannot be read whole, or when one
    entry is not an origin by ``_is_origin``: no origin is then listed,
    never one more than was meant.
    """
    environ = os.environ if environ is None else environ
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        data = None
    except Exception:  # noqa: BLE001 -- any file the parser cannot read whole lists no origin
        logger.warning("security.yaml cannot be read: no origin is listed for the API, a page of this site alone "
                       "is answered")
        return ()
    section = data.get("cors") if isinstance(data, dict) else None
    where = "cors.origins of security.yaml"
    if section is not None and not isinstance(section, dict):
        entries = None
    elif section is not None and "origins" in section:
        entries = _entries(section["origins"])
    else:
        entries = []
    if entries == []:
        where = ORIGINS_ENV
        entries = _entries(environ.get(ORIGINS_ENV, ""))
    if entries is None or not all(entry == "*" or _is_origin(entry) for entry in entries):
        logger.warning("%s is not a list of origins: no origin is listed for the API, a page of this site alone "
                       "is answered", where)
        return ()
    if "*" in entries:
        return ("*",)
    return tuple(dict.fromkeys(_spelled(_origin(entry)) for entry in entries))


def cors_settings(origins) -> dict:
    """The keyword arguments of the app's ``ListedOriginsCORS`` for the ``origins`` ``listed_origins`` gives:
    those origins exactly, with credentials; ``*`` without. No loopback origin is added and no pattern is answered:
    a page of this site needs no CORS."""
    if "*" in origins:
        logger.warning("CORS origins set to wildcard (*). Credentials are DISABLED to prevent token theft. List "
                       "the origins under cors.origins for credential support.")
        return {"allow_origins": ["*"], "allow_credentials": False, "allow_methods": ["*"], "allow_headers": ["*"]}
    return {"allow_origins": list(origins), "allow_credentials": True, "allow_methods": ["*"],
            "allow_headers": ["*"]}


class ListedOriginsCORS(CORSMiddleware):
    """Starlette's CORS middleware, which answers a page of an origin it does not list nothing at all.

    Starlette puts its simple headers -- ``Access-Control-Allow-Credentials``
    among them -- on every answer to a request that carries an Origin,
    listed or not, and its preflight headers on a refused preflight: a
    server in front that adds an ``Access-Control-Allow-Origin`` of its own
    would complete the pair. Here a request of an unlisted origin reaches
    the app untouched, and its preflight is a bare 400.
    """

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            origin = headers.get("origin")
            if origin is not None and not self.is_allowed_origin(origin=origin):
                if scope["method"] == "OPTIONS" and "access-control-request-method" in headers:
                    await PlainTextResponse("Disallowed CORS origin", status_code=400)(scope, receive, send)
                    return
                await self.app(scope, receive, send)
                return
        await super().__call__(scope, receive, send)


def _host(headers):
    """The match of the one Host header in raw ASGI ``headers``, by the grammar; None when it is not exactly one
    well-formed name with a port in range."""
    hosts = [value for key, value in headers if key.lower() == b"host"]
    if len(hosts) != 1:
        return None
    match = _HOST.fullmatch(hosts[0].decode("latin-1").lower())
    if match is None:
        return None
    port = match.group("port")
    if port is not None and not 1 <= int(port) <= 65535:
        return None
    return match


def host_name(headers) -> str | None:
    """The name of the one Host header in raw ASGI ``headers``, by the grammar; None when it is not exactly one
    well-formed name with a port in range."""
    match = _host(headers)
    return None if match is None else match.group("name")


def site_allowed(headers, allowed, origins=(), scheme="http") -> bool:
    """Whether a request with raw ASGI ``headers``, made over ``scheme`` (the scope's), comes from a page of this
    site, or from no browser.

    An ``Origin``, when sent, is exactly one, ``http`` or ``https``, a name
    and a port in range, and either exactly one of the listed ``origins``,
    or the request's own: the request's scheme (``https`` for ``https`` and
    ``wss``, ``http`` for ``http`` and ``ws``), the name and the port of the
    Host header (each the scheme's default when none is written), an
    ``allowed`` name, ``https`` for a name other than a loopback one. With
    no Origin, a ``Sec-Fetch-Site`` (exactly one) is ``same-origin`` or
    ``none``; neither header: a client that is no browser.
    """
    sent = [value for key, value in headers if key.lower() == b"origin"]
    if len(sent) > 1:
        return False
    if sent:
        origin = _origin(sent[0].decode("latin-1"))
        if origin is None:
            return False
        if _spelled(origin) in origins:
            return True
        page_scheme, name, port = origin
        if _PAGE_SCHEME.get(scheme) != page_scheme:
            return False
        host = _host(headers)
        if host is None or host.group("name") != name:
            return False
        host_port = host.group("port")
        if (str(int(host_port)) if host_port is not None else _DEFAULT_PORT[page_scheme]) != port:
            return False
        return name in allowed and (name in LOOPBACK or page_scheme == "https")
    sites = [value for key, value in headers if key.lower() == b"sec-fetch-site"]
    if len(sites) > 1:
        return False
    return not sites or sites[0].decode("latin-1").lower() in _SAME_SITE


class HostGateMiddleware:
    """Refuse an HTTP request (403) or a websocket handshake (closed before it is accepted) addressed to a name
    this machine is not served as, and a state-changing request (403) or a websocket handshake (closed) a page of
    another site sent; everything else, the lifespan included, passes untouched. A handshake it lets through
    reaches the app with ``ORIGIN_CHECKED`` set in its scope."""

    def __init__(self, app, hosts: tuple[str, ...] | None = None, origins: tuple[str, ...] | None = None) -> None:
        self.app = app
        self.allowed = frozenset(LOOPBACK + (listed_hosts() if hosts is None else tuple(hosts)))
        self.origins = frozenset(origin for origin in (listed_origins() if origins is None else tuple(origins))
                                 if origin != "*")
        self._logged: set[str] = set()
        self._site_logged = False
        self._socket_logged = False
        self._refused = 0

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers") or ()
        name = host_name(headers)
        if name not in self.allowed:
            self._log_refusal(name)
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            await _refuse(send, REFUSED)
            return
        # The ASGI scheme, when a server leaves it out: http for a request, ws for a websocket.
        scheme = scope.get("scheme") or ("ws" if scope["type"] == "websocket" else "http")
        if scope["type"] == "websocket":
            if not site_allowed(headers, self.allowed, self.origins, scheme):
                self._log_socket_refusal()
                await send({"type": "websocket.close", "code": 1008})
                return
            scope = {**scope, ORIGIN_CHECKED: True}
        if (scope["type"] == "http" and scope.get("method") not in _SAFE
                and not site_allowed(headers, self.allowed, self.origins, scheme)):
            self._log_site_refusal()
            await _refuse(send, REFUSED_SITE)
            return
        await self.app(scope, receive, send)

    def _log_socket_refusal(self) -> None:
        """One warning the first time a page of another site opens a socket, then counted with the refusals."""
        self._refused += 1
        if self._refused % _LOG_EVERY == 0:
            logger.warning("Host gate: %d requests refused since the start, the latest a socket opened by a page "
                           "of another site", self._refused)
        if self._socket_logged:
            return
        self._socket_logged = True
        logger.warning("Host gate: refused a socket opened by a page of another site (its Origin: another port, "
                       "name or scheme than the Host's, and not listed under cors.origins)")

    def _log_site_refusal(self) -> None:
        """One warning the first time a page asks a change it may not, then counted with the refusals; the names
        logged are not spent on it."""
        self._refused += 1
        if self._refused % _LOG_EVERY == 0:
            logger.warning("Host gate: %d requests refused since the start, the latest a change asked by a page "
                           "of another site or a listed name over http", self._refused)
        if self._site_logged:
            return
        self._site_logged = True
        logger.warning("Host gate: refused a change asked by a page of another site, or by a listed name over "
                       "http (its Origin or Sec-Fetch-Site)")

    def _log_refusal(self, name: str | None) -> None:
        """One warning per refused name, the first ``_LOG_NAMES`` of them, a malformed Host never repeated; and,
        whatever the names, one warning with the count and the latest name every ``_LOG_EVERY`` refusals, so a
        flood never hides the rest (the operator's own unlisted name included)."""
        self._refused += 1
        shown = name if name is not None else "(no well-formed Host)"
        if self._refused % _LOG_EVERY == 0:
            logger.warning("Host gate: %d requests refused since the start, the latest addressed to %s",
                           self._refused, shown)
        if shown in self._logged or len(self._logged) >= _LOG_NAMES:
            return
        self._logged.add(shown)
        logger.warning(
            "Host gate: refused a request addressed to %s (list the name under api.hosts of security.yaml "
            "if this machine is served as it)", shown)


async def _refuse(send, body: bytes) -> None:
    """Answer 403 with one of the fixed bodies."""
    await send({
        "type": "http.response.start",
        "status": 403,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
    })
    await send({"type": "http.response.body", "body": body})
