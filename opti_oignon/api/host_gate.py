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
answer only the page cannot read. The gate refuses a state-changing request
(any method but GET, HEAD and OPTIONS) a browser sends for a page of another
site: an ``Origin`` that does not name an allowed host (``http`` or
``https`` for a loopback name, on any port -- a page of this machine;
``https`` only for a listed one: a page served under a listed name over
plain ``http`` changes nothing, remote access is to be served over TLS),
``Origin: null``, or, with no Origin, a ``Sec-Fetch-Site`` other than
``same-origin`` or ``none``. A client that sends neither header is no
current browser (the CLI, curl) and passes. A GET with an effect is not
covered here (a census, BACKLOG 218).

The grammar and the loopback names are the garden's routes'
(``routes_allium.host_origin``): exactly one raw Host header, never
``X-Forwarded-Host``; a bracketed IPv6 literal or dot-separated labels with
no trailing dot, matched lowercase; a port from 1 to 65535, never compared.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import socket
from pathlib import Path

import yaml

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


def host_name(headers) -> str | None:
    """The name of the one Host header in raw ASGI ``headers``, by the grammar; None when it is not exactly one
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
    return match.group("name")


def site_allowed(headers, allowed) -> bool:
    """Whether a request with raw ASGI ``headers`` comes from a page of this machine's own sites, or from no browser.

    An ``Origin``, when sent, is exactly one, ``http`` or ``https`` with an
    allowed name (``https`` for a name other than a loopback one) and a port
    in range; with no Origin, a ``Sec-Fetch-Site`` (exactly one) is
    ``same-origin`` or ``none``; neither header: a client that is no browser.
    """
    origins = [value for key, value in headers if key.lower() == b"origin"]
    if len(origins) > 1:
        return False
    if origins:
        match = _ORIGIN.fullmatch(origins[0].decode("latin-1").lower())
        if match is None:
            return False
        port = match.group("port")
        if port is not None and not 1 <= int(port) <= 65535:
            return False
        name = match.group("name")
        return name in allowed and (name in LOOPBACK or match.group("scheme") == "https")
    sites = [value for key, value in headers if key.lower() == b"sec-fetch-site"]
    if len(sites) > 1:
        return False
    return not sites or sites[0].decode("latin-1").lower() in _SAME_SITE


class HostGateMiddleware:
    """Refuse an HTTP request (403) or a websocket handshake (closed before it is accepted) addressed to a name
    this machine is not served as, and a state-changing request a page of another site sent (403); everything
    else, the lifespan included, passes untouched."""

    def __init__(self, app, hosts: tuple[str, ...] | None = None) -> None:
        self.app = app
        self.allowed = frozenset(LOOPBACK + (listed_hosts() if hosts is None else tuple(hosts)))
        self._logged: set[str] = set()
        self._site_logged = False
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
        if (scope["type"] == "http" and scope.get("method") not in _SAFE
                and not site_allowed(headers, self.allowed)):
            self._log_site_refusal()
            await _refuse(send, REFUSED_SITE)
            return
        await self.app(scope, receive, send)

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
