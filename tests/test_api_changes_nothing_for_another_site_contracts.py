"""Contracts for the gate's second question: a page of another site changes nothing through the API.

A page of any site can send a request to the loopback address itself, a loopback Host that the Host gate answers:
a form posted, a ``fetch`` in ``no-cors`` mode -- requests the browser sends without asking (no preflight) and whose
answer only the page cannot read. In single-user Daily mode the synthetic admin served them and the CSRF check was
skipped ("no cross-site request to forge"): a POST whose parameters are in its path or query, a form or multipart
route, ran for the page. Now ``api/host_gate.HostGateMiddleware`` refuses, in every mode, a state-changing request
(POST, PUT, PATCH, DELETE) whose ``Origin`` does not name an allowed host (``https`` for a listed name), or, with no
Origin, whose ``Sec-Fetch-Site`` is not ``same-origin`` or ``none``; a client that sends neither is no browser.

  * XS1 -- ANOTHER SITE'S CHANGE IS REFUSED: a POST, a PUT, a PATCH, a DELETE, a method in another case or any other
    unsafe method with the Origin of another site, ``null``, two Origins, a malformed one, a listed name over ``http``
    (remote access serves it over TLS), a port zero or out of range, or with no Origin and a ``Sec-Fetch-Site`` of
    another site: 403 before the app, the fixed words, never the origin repeated (round 1: the methods, the port).
  * XS2 -- THE MACHINE'S OWN PAGES AND THE CLI ARE ANSWERED: a POST from the frontend (``localhost:5173``), from the
    API's own address, from ``[::1]``, from a listed name over ``https``, with ``Sec-Fetch-Site`` ``same-origin`` or
    ``none``, and with neither header (the CLI); a page of this machine on another port, an Origin or a
    ``Sec-Fetch-Site`` in capitals (round 1); a GET, a HEAD and an OPTIONS (a preflight, which CORS answers) from
    another site (round 2); a websocket
    from another site reaches its route, whose own Origin check refuses it (``authenticate_websocket``).
  * XS3 -- SAID ONCE, AND SAID TRUE: the first refusal of a change from another site is logged once, and spends none
    of the names the Host gate logs (round 1); the CSRF middleware no longer says that single-user mode has no
    cross-site request to forge.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_api_answers_only_its_host_names_contracts as hg_t  # noqa: E402
from _isolation import source  # noqa: E402

BUDGET_S = {
    "test_xs1_another_sites_change_is_refused": 5.0,
    "test_xs2_the_machines_own_pages_and_the_cli_are_answered": 5.0,
    "test_xs3_said_once_and_said_true": 5.0,
}

_LISTED = ("oo.lan",)
_SITE_REFUSED = b'{"detail":"This server takes no change from that page."}'


def _change(gate_mod, extra, method="POST", host="localhost:8001"):
    """A state-changing request to ``host`` with the ``extra`` headers: (status, whether the app ran, body)."""
    sent, ran = hg_t._ask(gate_mod, [("host", host)] + list(extra), method=method, hosts=_LISTED)
    return hg_t._status(sent), ran, b"".join(m.get("body", b"") for m in sent)


def test_xs1_another_sites_change_is_refused():
    gate_mod, restore = hg_t._gate_module()
    seen = {}
    try:
        for method in ("POST", "PUT", "PATCH", "DELETE", "post", "Patch", "PROPFIND"):
            seen[f"{method} another site"] = _change(gate_mod, [("origin", "https://evil.example")], method)
        for label, extra, host in (
            ("null origin", [("origin", "null")], "localhost:8001"),
            ("two origins", [("origin", "http://localhost:5173"), ("origin", "http://localhost:5173")], "localhost"),
            ("malformed", [("origin", "localhost:5173")], "localhost"),
            ("port zero", [("origin", "http://localhost:0")], "localhost"),
            ("loopback as a label", [("origin", "http://localhost.evil.com")], "localhost"),
            ("a listed name over http", [("origin", "http://oo.lan")], "oo.lan"),
            ("a port out of range", [("origin", "http://localhost:99999")], "localhost"),
            ("cross-site, no origin", [("sec-fetch-site", "cross-site")], "localhost"),
            ("same-site, no origin", [("sec-fetch-site", "same-site")], "localhost"),
            ("two fetch sites", [("sec-fetch-site", "none"), ("sec-fetch-site", "none")], "localhost"),
        ):
            seen[label] = _change(gate_mod, extra, host=host)
    finally:
        restore()
    for label, (status, ran, body) in seen.items():
        assert status == 403 and not ran, f"{label}: a change from another site reached the app: {status} {ran}"
        assert body == _SITE_REFUSED, f"{label}: not the fixed words: {body!r}"
        assert b"evil" not in body, f"{label}: the refusal repeated the origin"


def test_xs2_the_machines_own_pages_and_the_cli_are_answered():
    gate_mod, restore = hg_t._gate_module()
    seen = {}
    try:
        for label, extra, host in (
            ("the frontend", [("origin", "http://localhost:5173"), ("sec-fetch-site", "same-site")], "localhost:8001"),
            ("the API's own address", [("origin", "http://127.0.0.1:8001")], "127.0.0.1:8001"),
            ("ipv6 loopback", [("origin", "http://[::1]:8001")], "[::1]:8001"),
            ("a listed name over https", [("origin", "https://oo.lan")], "oo.lan"),
            ("the same origin", [("sec-fetch-site", "same-origin")], "localhost"),
            ("typed by the user", [("sec-fetch-site", "none")], "localhost"),
            ("the CLI", [], "localhost:8001"),
            ("a page of this machine on another port", [("origin", "http://localhost:3000")], "localhost:8001"),
            ("an origin in capitals", [("origin", "HTTP://LOCALHOST:5173")], "localhost:8001"),
            ("a fetch site in capitals", [("sec-fetch-site", "Same-Origin")], "localhost"),
        ):
            seen[label] = _change(gate_mod, extra, host=host)
        get_status, get_ran, _body = _change(gate_mod, [("origin", "https://evil.example")], method="GET")
        safe = {method: _change(gate_mod, [("origin", "https://evil.example")], method=method)
                for method in ("HEAD", "OPTIONS")}
        ws_sent, ws_ran = hg_t._ask(gate_mod, [("host", "localhost:8001"), ("origin", "https://evil.example")],
                                    kind="websocket")
    finally:
        restore()
    for label, (status, ran, _body) in seen.items():
        assert ran and status == 200, f"{label}: a change from this machine's own page or the CLI was refused: {status}"
    assert get_ran and get_status == 200, "a GET from another site was refused: its answer is unreadable to it"
    for method, (status, ran, _body) in safe.items():
        assert ran and status == 200, f"a {method} from another site (a preflight, a head) was refused: {status}"
    assert ws_ran and ws_sent == [], (
        "a websocket from another site did not reach its route, whose own Origin check answers it")


def test_xs3_said_once_and_said_true(caplog):
    gate_mod, restore = hg_t._gate_module()
    try:
        gate = gate_mod.HostGateMiddleware(None, hosts=())
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            for _n in range(5):
                hg_t._ask(gate_mod, [("host", "localhost"), ("origin", "https://evil.example")], method="POST",
                          gate=gate)
            for n in range(32):
                hg_t._ask(gate_mod, [("host", f"name{n}.example")], gate=gate)
        lines = [record.getMessage() for record in caplog.records]
        csrf = source("api", "csrf_middleware.py").read_text(encoding="utf-8")
    finally:
        restore()
    said = [line for line in lines if line.startswith("Host gate: refused a change asked by a page of another site")]
    assert len(said) == 1 and "evil" not in said[0], f"the refusal of a change from another site was not said once: {lines}"
    named = [line for line in lines if "refused a request addressed to" in line]
    assert len(named) == 32, f"a refused change spent one of the names the Host gate logs: {len(named)}"
    assert "no\n    cross-site request to forge" not in csrf and "no cross-site request to forge" not in csrf, (
        "the CSRF middleware still says single-user mode has no cross-site request to forge")
    assert "host_gate.site_allowed" in csrf, "the CSRF middleware does not name the gate that refuses another site"
    assert "nothing for CSRF to protect" not in csrf, "the CSRF middleware still says single-user mode needs nothing"
