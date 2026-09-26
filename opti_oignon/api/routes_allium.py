#!/usr/bin/env python3
"""The componion's garden in the API: one read route, and the checks every garden route carries.

    GET /api/allium/status  -> the componion as it is served now (a simulation, read only)

The garden is the terminal's (``oo garden``); the API only reads it, through
the same service, the same ``Look`` and the same served projection
(``describe.web_fields``), with a caller built from the platform's explicit
user dependency and nothing else. Every route of this router carries, at
router level and in this order:

* ``host_origin``: the request must be addressed to this machine by a
  loopback name or a name listed in ``api.hosts`` of ``allium.yaml`` (the
  raw Host header, exactly one, never ``X-Forwarded-Host``); an Origin, when
  one is sent, must name the same set, over https for a listed name; and a
  ``Sec-Fetch-Site`` other than ``same-origin`` or ``none`` is answered only
  beside an accepted Origin, so a page of another site cannot make the
  garden look (an image pointed at the loopback API carries a good Host and
  no Origin). Ports are never compared for a read: a rebound name is the
  attacker's, the port is not.
* the platform's user dependency (``routes_auth._get_current_user``), whose
  principal is the only input the caller is built from.

``GardenRoute`` turns every refusal into a closed body,
``{"detail": <the garden's line>, "refusal": <code>}``: nothing of the
request or of an exception is repeated, a 401 keeps its status so the
frontend's sign-in redirect still works, and a fault is a 500 with the
garden's own line. A 404 or 405 under the prefix is the router's, with the
platform's body, and runs no garden work.

The switch and the names of ``api.hosts`` are read here, from the one
settings file, without importing the being (``switched_on``,
``listed_hosts``, each held equal to the garden's own reader by a
contract): with ``enabled`` anything but ``true``, the status route answers
``disabled`` and imports nothing of the being, whether the request named a
loopback name or a listed one, and so does the health key
(``health_flag``). Only a refused request imports something of the being:
the catalogue and its nets, to say its line (standard library only, no
store, no file). The API's garden is built once per process, at the first
status request with the switch on, and its store is closed at shutdown
(``close_garden``). It serves no attended verb, refuses a missing caller,
caps every view with the API's own bound, reads the emergency stop of this
server, and takes its single-user rule from the auth manager the process
already runs.
"""

import importlib.util
import logging
import re
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from .routes_auth import _get_auth_manager, _get_current_user
from .schemas_allium import AlliumRefusal, AlliumStatus

logger = logging.getLogger(__name__)

# The names a garden route always answers, for the Host and for the Origin.
LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
# A refusal's code and its HTTP status; the body is {"detail": the line web.<code>, "refusal": code}.
REFUSAL_STATUS = {"host": 403, "origin": 403, "site": 403, "sign_in": 401, "auth_unavailable": 503,
                  "request": 422, "fault": 500}
# The garden's settings file, found from this package's location, as the settings reader finds it.
ALLIUM_YAML = Path(__file__).resolve().parent.parent / "config" / "allium.yaml"
# The switched-off form: every field empty and no line, so no path of this machine is ever served.
DISABLED = {"as_of": None, "being": None, "habitat": None, "labels": [], "law": None, "lines": [],
            "source": "simulation", "status": "disabled"}
# A name as a request may carry it: a bracketed IPv6 literal, or dot-separated labels with no trailing dot.
_NAME = r"(?P<name>\[[0-9a-f:]+\]|[a-z0-9-]+(?:\.[a-z0-9-]+)*)"
_PORT = r"(?::(?P<port>[0-9]{1,5}))?"
_HOST = re.compile(_NAME + _PORT)
_ORIGIN = re.compile(r"(?P<scheme>https?)://" + _NAME + _PORT)
_SAME_SITE = ("same-origin", "none")
# A catalogue key as a fault record may name it; anything else is logged as unnamed.
_KEY = re.compile(r"[a-z0-9_.]{1,64}")
# A name ``api.hosts`` may list, by the garden's settings rule: an exact lowercase name, or a bracketed IPv6 literal.
_LISTED = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)*|\[[0-9a-f:]+\]")
# What a refusal says when the garden's catalogue itself cannot be read: the catalogue's ``WEB_FALLBACK``.
FALLBACK = "The garden could not answer."

_switch_seen = {"entry": None}
_hosts_seen = {"entry": None}
_held = {"garden": None}
_held_lock = threading.Lock()


class GardenRefused(Exception):
    """A garden route refused the request before anything ran; ``code`` is a key of ``REFUSAL_STATUS``."""

    def __init__(self, code):
        if code not in REFUSAL_STATUS:
            raise ValueError("not a refusal code of the garden's routes")
        super().__init__(code)
        self.code = code


# ---------------------------------------------------------------------------
# The switch and the health key, read without importing the being
# ---------------------------------------------------------------------------
def switched_on(path=None):
    """Whether allium.yaml says ``enabled: true``, read without importing the garden; ``False`` when unsure.

    The garden's own rule (``settings.switch(path) == "on"``): a top-level
    mapping whose ``enabled`` is the YAML boolean ``true``. A missing file or
    key, any other value, a file that does not parse and a top level that is
    not a mapping are all off. One ``stat`` per call; the file is parsed
    again only when its modification time, size or inode changes, and the
    cache is one pair stored in one assignment.
    """
    try:
        file = Path(path) if path is not None else ALLIUM_YAML
        st = file.stat()
        key = (str(file), st.st_mtime_ns, st.st_size, st.st_ino)
        entry = _switch_seen["entry"]
        if entry is None or entry[0] != key:
            import yaml

            data = yaml.safe_load(file.read_text(encoding="utf-8"))
            entry = (key, isinstance(data, dict) and data.get("enabled") is True)
            _switch_seen["entry"] = entry
        return entry[1]
    except Exception:  # noqa: BLE001 - a switch that cannot be read is off
        return False


def health_flag(path=None):
    """The health map's ``allium`` key: the switch is on and the garden's package is found, never imported.

    ``find_spec`` finds the package without executing it, so the key costs
    one ``stat`` of the settings file and builds no garden and no store.
    Any failure is ``False``.
    """
    try:
        if not switched_on(path):
            return False
        return "opti_oignon.allium" in sys.modules or importlib.util.find_spec("opti_oignon.allium") is not None
    except Exception:  # noqa: BLE001 - a key that cannot be computed says the garden is not there
        return False


# ---------------------------------------------------------------------------
# The Host, the Origin and the page's site
# ---------------------------------------------------------------------------
def _hosts_of(data):
    """The names ``api.hosts`` lists in a parsed settings file, by the garden's own rule; ``()`` for anything else.

    The rule of ``settings.api``: a list of exact lowercase names or
    bracketed IPv6 literals, with no port, scheme or wildcard; a list that
    is not one, holds one entry that is not such a name, names an
    unspecified address or spells an IPv4 address otherwise than as its
    dotted quad allows the loopback names only.
    """
    import ipaddress
    import socket

    section = data.get("api") if isinstance(data, dict) else None
    if not isinstance(section, dict) or "hosts" not in section:
        return ()
    raw = section["hosts"]
    if not isinstance(raw, list):
        return ()
    for entry in raw:
        if not isinstance(entry, str) or not _LISTED.fullmatch(entry):
            return ()
        bracketed = entry.startswith("[")
        try:
            address = ipaddress.ip_address(entry[1:-1] if bracketed else entry)
        except ValueError:
            address = None
        if address is None and bracketed:
            return ()
        mapped = getattr(address, "ipv4_mapped", None)
        if address is not None and (address.is_unspecified or (mapped is not None and mapped.is_unspecified)):
            return ()
        if not bracketed:
            try:
                packed = socket.inet_aton(entry)
            except (OSError, ValueError):
                packed = None
            if packed is not None and socket.inet_ntoa(packed) != entry:
                return ()
    return tuple(raw)


def listed_hosts(path=None):
    """The names ``api.hosts`` of allium.yaml lists, read without importing the garden; ``()`` when unsure.

    The file is the one the switch is read from (``ALLIUM_YAML``), and the
    rule is the garden's settings rule (``settings.api(path).hosts``, held
    equal by a contract). One ``stat`` per call; the file is parsed again
    only when its modification time, size or inode changes, and the cache is
    one pair stored in one assignment. A list that cannot be read is logged
    once per change of the file, and allows the loopback names only.
    """
    try:
        file = Path(path) if path is not None else ALLIUM_YAML
        st = file.stat()
        key = (str(file), st.st_mtime_ns, st.st_size, st.st_ino)
    except Exception:  # noqa: BLE001 - a file that cannot be found lists nothing
        return ()
    entry = _hosts_seen["entry"]
    if entry is None or entry[0] != key:
        try:
            import yaml

            data = yaml.safe_load(file.read_text(encoding="utf-8"))
            names = _hosts_of(data)
        except Exception:  # noqa: BLE001 - a file that cannot be read lists nothing
            data, names = None, ()
        section = data.get("api") if isinstance(data, dict) else None
        if not names and isinstance(section, dict) and "hosts" in section and section["hosts"] != []:
            logger.warning("api.hosts cannot be read as a list of names: the garden's routes answer the loopback "
                           "names only")
        entry = (key, names)
        _hosts_seen["entry"] = entry
    return entry[1]


def host_origin(request: Request) -> None:
    """Refuse a request not addressed to this machine by an allowed name, or made by a page of another site.

    Runs first on every garden route, before the user dependency and the
    garden. The allowed names are ``LOOPBACK`` and the names of ``api.hosts``;
    the listed names are read (``listed_hosts``) only when a request carries
    a name outside the loopback names. The Host header is the only name read
    for the request: the authority of an absolute-form request target, which
    a browser never sends to an origin server, is not. Raises
    ``GardenRefused("host")``, ``("origin")`` or ``("site")``; a header's
    value is never repeated.
    """
    raw = request.scope.get("headers") or ()
    listed = []

    def allowed(name):
        if name in LOOPBACK:
            return True
        if not listed:
            listed.append(listed_hosts())
        return name in listed[0]

    def values(header):
        return [value.decode("latin-1") for key, value in raw if key.lower() == header]

    def port_ok(port):
        return port is None or 1 <= int(port) <= 65535

    hosts = values(b"host")
    if len(hosts) != 1:
        raise GardenRefused("host")
    host = _HOST.fullmatch(hosts[0].lower())
    if host is None or not port_ok(host.group("port")) or not allowed(host.group("name")):
        raise GardenRefused("host")

    origins = values(b"origin")
    if len(origins) > 1:
        raise GardenRefused("origin")
    if origins:
        origin = _ORIGIN.fullmatch(origins[0].lower())
        if origin is None or not port_ok(origin.group("port")) or not allowed(origin.group("name")):
            raise GardenRefused("origin")
        if origin.group("name") not in LOOPBACK and origin.group("scheme") != "https":
            raise GardenRefused("origin")

    sites = values(b"sec-fetch-site")
    if len(sites) > 1:
        raise GardenRefused("site")
    if sites and sites[0] not in _SAME_SITE and not origins:
        raise GardenRefused("site")


# ---------------------------------------------------------------------------
# The closed refusal bodies
# ---------------------------------------------------------------------------
def _refusal(code):
    """The closed body of a refusal: the garden's line for ``code``, or its fixed fallback when that line fails.

    The catalogue is imported here, to say the line; a catalogue that cannot
    even be imported gives ``FALLBACK``, the same fixed words.
    """
    try:
        from opti_oignon.allium import wording

        detail = wording.web("web." + code).text
    except Exception:  # noqa: BLE001 - a line that fails its check is never served
        detail = FALLBACK
    return JSONResponse({"detail": detail, "refusal": code}, status_code=REFUSAL_STATUS[code])


def _refused(code):
    logger.debug("garden route refused: %s", code)
    return _refusal(code)


def _copy_key(exc):
    """The catalogue key a ``CopyRefused`` names, or ``None`` for any other exception."""
    try:
        from opti_oignon.allium import wording
    except Exception:  # noqa: BLE001 - no catalogue, no key to name
        return None
    if not isinstance(exc, wording.CopyRefused):
        return None
    key = exc.key
    return key if isinstance(key, str) and _KEY.fullmatch(key) else "unnamed"


def _fault(exc):
    key = _copy_key(exc)
    if key is None:
        logger.warning("garden route fault: %s", type(exc).__name__)
    else:
        logger.warning("garden route fault: %s (%s)", type(exc).__name__, key)
    return _refusal("fault")


class GardenRoute(APIRoute):
    """A garden route: whatever is raised while its dependencies, its request or its handler run is a closed body.

    ``GardenRefused`` gives its own code; a request that does not validate
    gives ``request`` (422); the user dependency's 401 and 503 give
    ``sign_in`` and ``auth_unavailable`` with their status; anything else
    -- another HTTP exception, a response that does not validate, a line
    that fails its check, an exception the garden does not map -- is a
    ``fault`` (500). Refusals are logged at debug with their code, faults at
    warning with the exception's class name (and a failed line's catalogue
    key), never its text.
    """

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def garden_handler(request: Request):
            try:
                return await handler(request)
            except GardenRefused as refused:
                return _refused(refused.code)
            except RequestValidationError:
                return _refused("request")
            except StarletteHTTPException as exc:
                if exc.status_code == 401:
                    return _refused("sign_in")
                if exc.status_code == 503:
                    return _refused("auth_unavailable")
                return _fault(exc)
            except Exception as exc:  # noqa: BLE001 - every other exception is a fault, said without its text
                return _fault(exc)

        return garden_handler


router = APIRouter(prefix="/api/allium", tags=["allium"], route_class=GardenRoute,
                   dependencies=[Depends(host_origin), Depends(_get_current_user)],
                   responses={status: {"model": AlliumRefusal} for status in (401, 403, 422, 500, 503)})


# ---------------------------------------------------------------------------
# The API's garden: one per process, built at the first status request that needs it
# ---------------------------------------------------------------------------
# A zero-argument callable returning a garden; ``None`` means the production garden.
garden_factory = None


def _single_user():
    """The rule the API process already runs: the auth manager's own single-user mode.

    Raises the platform's 503 when the auth module is absent, which the
    garden's store reads as an account it cannot establish.
    """
    return _get_auth_manager().single_user_mode is True


def _stopped():
    """Whether the emergency stop of this server is on."""
    from opti_oignon import emergency_stop

    return emergency_stop.is_stopped()


def _production_garden():
    """The API's garden: no attended verb, a caller required, the API's cap, this server's stop and auth rule."""
    from opti_oignon.allium import service

    return service.Garden.production(stopped=_stopped, terminal=False, view_cap=service.api_view_cap,
                                     single_user=_single_user)


def _garden():
    """This process's garden, built once, under a lock, at the first call."""
    with _held_lock:
        if _held["garden"] is None:
            factory = garden_factory
            _held["garden"] = factory() if factory is not None else _production_garden()
        return _held["garden"]


def close_garden():
    """Close the store of this process's garden if one was built; builds nothing."""
    with _held_lock:
        garden = _held["garden"]
    if garden is not None:
        garden.close()


@router.get("/status", response_model=AlliumStatus)
def garden_status(principal: dict = Depends(_get_current_user)) -> dict:
    """The componion as it is served now: its status, its labels and the lines that say it.

    A simulation, read only: nothing is written in the being's store, and a
    view the engine cannot finish within the API's cap is shown as of its last
    kept state, labelled. Every served status is a 200; a refusal names its reason.
    """
    if not switched_on():
        return dict(DISABLED, labels=[], lines=[])
    from opti_oignon.allium import describe, service

    caller = service.transport("web", principal=principal)
    return describe.web_fields(_garden().look(caller=caller))
