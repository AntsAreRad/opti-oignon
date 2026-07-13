#!/usr/bin/env python3
"""Contracts for the model-download SSRF defense.

The model manager downloads GGUF weights from the public internet. To do
that safely it carries a complete SSRF defense -- private-range rejection,
validated resolution, a scheme gate, and per-hop IP pinning that follows
redirects manually so a public host cannot 302 to an internal address and a
name cannot flip its A record between validation and the connect (DNS
rebinding / TOCTOU). None of that was pinned by a test. A silent regression
-- someone swapping the pinned opener for an ordinary client, or a redirect
follower that re-resolves -- would pass unnoticed and the defense would fall
without a single red line. These contracts make that impossible.

The clauses pin distinct mechanisms so that one probe reddens exactly one
clause:

  * Contract P1 -- the address primitive blocks every internal category
    (private, loopback, link-local, multicast, reserved, unspecified) and
    passes routable addresses. This is the leaf the whole defense rests on.
  * Contract P2 -- a non-HTTP(S) scheme is refused. file://, ftp://, gopher://
    never reach resolution.
  * Contract P3 -- plaintext HTTP is refused for any host that is not
    localhost. A public name over http:// is rejected even though its scheme
    passed P2's gate -- a distinct mechanism, so P2 and P3 do not co-redden.
  * Contract P4 -- a URL with no hostname is refused before resolution.
  * Contract P5 -- resolution is strict across the whole answer: if ANY of
    several resolved addresses is internal, the request is refused. This
    removes the rebinding window where one A record of several points inside.
  * Contract P6 -- every redirect hop is re-validated and re-pinned against
    the redirect TARGET, not the original URL. A public host that 302s to an
    internal address is refused at the second hop.
  * Contract P7 -- the pin is real at the socket: the pinned connection
    connects to the validated IP, never to the hostname. This is what makes
    a flipped DNS record after validation harmless.
  * Contract P8 -- the redirect chain is bounded; exceeding the hop limit is
    refused rather than followed forever.

Injectable seams (resolver=, opener=) carry the tests; no test performs real
network access. Local-only (the public distribution ships no tests). Runs
under pytest or directly via the __main__ runner. Loading follows the house
idiom: canonical dotted names, an empty-path package stand-in, and a
meta-path guard sealing the isolation window.
"""

import importlib.util
import socket
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the test's
    back -- silently importing live code. This guard sits ahead of every
    finder and refuses the names that were not seeded, so a load behaves
    identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


# ---------------------------------------------------------------------------
# Isolated loading of the model manager (stdlib-only imports at module top,
# so no sibling project module needs seeding).
# ---------------------------------------------------------------------------
def _load():
    """Load model_manager in isolation; returns (module, restore)."""
    keys = ("opti_oignon", "opti_oignon.model_manager")
    saved = {k: sys.modules.get(k) for k in keys}

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.model_manager", _OO / "model_manager.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.model_manager"] = mod
    spec.loader.exec_module(mod)
    pkg.model_manager = mod

    def restore():
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


# ---------------------------------------------------------------------------
# Deterministic stand-ins for the network seams (recorders)
# ---------------------------------------------------------------------------
def _resolver_for(mapping: dict[str, list[str]]):
    """A getaddrinfo-shaped resolver returning fixed IPs per hostname.

    Each mapped host answers a getaddrinfo-style list whose sockaddr slot
    carries the IP the code reads via ``info[4][0]``. An unmapped host
    raises gaierror, exactly like a real NXDOMAIN.
    """

    def _resolve(hostname, port):
        ips = mapping.get(hostname)
        if ips is None:
            raise socket.gaierror(f"unmapped host in stub: {hostname}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]

    return _resolve


class _Headers:
    """A minimal case-insensitive header holder exposing .get()."""

    def __init__(self, pairs: dict[str, str] | None = None):
        self._d = dict(pairs or {})

    def get(self, key, default=None):
        for k, v in self._d.items():
            if k.lower() == key.lower():
                return v
        return default


class _FakeRaw:
    """A stand-in for one raw HTTP response hop.

    Exposes the subset urlopen_ssrf_safe touches: ``.status``, ``.headers``,
    ``.read()`` and ``.close()``. ``_conn_to_close`` mirrors the real
    downloader's attribute so the module's closer path runs unchanged.
    """

    def __init__(self, status: int, headers: dict[str, str] | None = None, body: bytes = b""):
        self.status = status
        self.headers = _Headers(headers)
        self._body = body
        self.closed = False
        self._conn_to_close = None

    def read(self, amt: int = -1):
        return self._body

    def close(self):
        self.closed = True


class _OpenerRecorder:
    """Records every (url, pinned_ip) an opener is handed and replays a script.

    ``script`` is a list of _FakeRaw returned in order; the recorder captures
    the call arguments so a test can assert what the loop actually opened.
    """

    def __init__(self, script: list[_FakeRaw]):
        self._script = list(script)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url, pinned_ip, headers, timeout):
        self.calls.append((url, pinned_ip))
        if not self._script:
            raise AssertionError("opener called more times than scripted")
        return self._script.pop(0)


# ---------------------------------------------------------------------------
# Contract P1 -- the address primitive blocks every internal category
# ---------------------------------------------------------------------------
def test_p1_ip_primitive_blocks_every_internal_category():
    mod, restore = _load()
    try:
        blocked = {
            "private 10.x": "10.0.0.1",
            "private 172.16.x": "172.16.5.4",
            "private 192.168.x": "192.168.1.1",
            "loopback": "127.0.0.1",
            "loopback v6": "::1",
            "link-local": "169.254.169.254",
            "link-local v6": "fe80::1",
            "multicast": "224.0.0.1",
            "reserved": "240.0.0.1",
            "unspecified": "0.0.0.0",
        }
        for label, ip in blocked.items():
            assert mod._ip_is_blocked(ip) is True, (
                f"{label} ({ip}) must be blocked as internal"
            )

        routable = {
            "cloudflare": "1.1.1.1",
            "google dns": "8.8.8.8",
            "public host": "93.184.216.34",
            "public v6": "2606:4700:4700::1111",
        }
        for label, ip in routable.items():
            assert mod._ip_is_blocked(ip) is False, (
                f"{label} ({ip}) is routable and must NOT be blocked"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P2 -- a non-HTTP(S) scheme is refused before resolution
# ---------------------------------------------------------------------------
def test_p2_non_http_scheme_is_refused():
    mod, restore = _load()
    try:
        # A resolver that would happily answer public, proving the refusal is
        # the scheme gate and not a resolution failure.
        resolver = _resolver_for({"example.com": ["93.184.216.34"]})
        for url in (
            "file:///etc/passwd",
            "ftp://example.com/x.gguf",
            "gopher://example.com/1",
            "data:text/plain,hello",
        ):
            try:
                mod._validate_and_resolve(url, resolver=resolver)
            except ValueError:
                continue
            raise AssertionError(f"scheme in {url!r} must be refused")
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P3 -- plaintext HTTP is refused for any non-localhost host
# ---------------------------------------------------------------------------
def test_p3_plaintext_http_is_refused_for_non_localhost():
    mod, restore = _load()
    try:
        resolver = _resolver_for({
            "example.com": ["93.184.216.34"],
            "localhost": ["127.0.0.1"],
            "127.0.0.1": ["127.0.0.1"],
        })
        # http:// to a public host: scheme passes P2's gate, but the
        # http-localhost mechanism refuses it.
        try:
            mod._validate_and_resolve("http://example.com/x.gguf", resolver=resolver)
        except ValueError:
            pass
        else:
            raise AssertionError("http:// to a public host must be refused")

        # https:// to the same public host resolves fine: this isolates the
        # refusal to the plaintext gate, not the host.
        scheme, host, port, pinned = mod._validate_and_resolve(
            "https://example.com/x.gguf", resolver=resolver,
        )
        assert (scheme, host, port, pinned) == (
            "https", "example.com", 443, "93.184.216.34",
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P4 -- a URL with no hostname is refused before resolution
# ---------------------------------------------------------------------------
def test_p4_url_without_hostname_is_refused():
    mod, restore = _load()
    try:
        # A permissive resolver that would answer the empty host, proving the
        # refusal is the hostname check and not a resolution failure.
        resolver = _resolver_for({"": ["93.184.216.34"]})
        for url in ("https:///models/x.gguf", "https://"):
            try:
                mod._validate_and_resolve(url, resolver=resolver)
            except ValueError:
                continue
            raise AssertionError(f"URL {url!r} without a hostname must be refused")
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P5 -- resolution is strict across the whole answer
# ---------------------------------------------------------------------------
def test_p5_any_internal_address_among_several_is_refused():
    mod, restore = _load()
    try:
        # Public first, internal second: a lenient "check only the first"
        # implementation would pass. Strictness means the internal one is
        # still caught.
        resolver = _resolver_for({"rebind.example": ["93.184.216.34", "10.0.0.5"]})
        try:
            mod._resolve_validated_ips("rebind.example", 443, resolver=resolver)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "an internal address anywhere in the answer must be refused"
            )

        # All-public answer is accepted and returns every address.
        resolver_ok = _resolver_for({"cdn.example": ["93.184.216.34", "151.101.1.1"]})
        ips = mod._resolve_validated_ips("cdn.example", 443, resolver=resolver_ok)
        assert ips == ["93.184.216.34", "151.101.1.1"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P6 -- every redirect hop is re-validated against the target
# ---------------------------------------------------------------------------
def test_p6_redirect_to_internal_is_refused_at_the_second_hop():
    mod, restore = _load()
    try:
        # Hop 1: a public host answers 302 pointing at an internal host.
        # A loop that only validates the original URL would follow it.
        resolver = _resolver_for({
            "public.example": ["93.184.216.34"],
            "internal.example": ["169.254.169.254"],
        })
        opener = _OpenerRecorder([
            _FakeRaw(302, {"Location": "https://internal.example/creds"}),
            # If the loop wrongly proceeds, this second hop would be opened.
            _FakeRaw(200, body=b"SHOULD-NOT-REACH"),
        ])
        try:
            mod.urlopen_ssrf_safe(
                "https://public.example/model.gguf",
                resolver=resolver,
                opener=opener,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("a redirect to an internal host must be refused")

        # The internal target must have been rejected BEFORE it was opened:
        # exactly one opener call, for the public first hop.
        assert opener.calls == [("https://public.example/model.gguf", "93.184.216.34")], (
            f"the internal hop must never be opened; opener calls were {opener.calls}"
        )

        # A redirect to another PUBLIC host is followed and re-pinned.
        resolver2 = _resolver_for({
            "public.example": ["93.184.216.34"],
            "cdn.example": ["151.101.1.1"],
        })
        opener2 = _OpenerRecorder([
            _FakeRaw(302, {"Location": "https://cdn.example/model.gguf"}),
            _FakeRaw(200, body=b"OK"),
        ])
        resp = mod.urlopen_ssrf_safe(
            "https://public.example/model.gguf",
            resolver=resolver2,
            opener=opener2,
        )
        assert resp.status == 200 and resp.read() == b"OK"
        assert opener2.calls == [
            ("https://public.example/model.gguf", "93.184.216.34"),
            ("https://cdn.example/model.gguf", "151.101.1.1"),
        ], (
            "the second public hop must be re-pinned to its own validated IP; "
            f"opener calls were {opener2.calls}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P7 -- the pin is real at the socket
# ---------------------------------------------------------------------------
def test_p7_pinned_connection_connects_to_the_validated_ip():
    mod, restore = _load()
    try:
        class _StopConnect(Exception):
            """Halts connect() right after the target is captured."""

        captured: dict[str, object] = {}

        def _fake_create_connection(address, timeout=None):
            captured["address"] = address
            captured["timeout"] = timeout
            raise _StopConnect()

        original = mod.socket.create_connection
        mod.socket.create_connection = _fake_create_connection
        try:
            conn = mod._PinnedHTTPConnection(
                "example.com", pinned_ip="93.184.216.34", port=80, timeout=7,
            )
            try:
                conn.connect()
            except _StopConnect:
                pass
        finally:
            mod.socket.create_connection = original

        assert captured.get("address") == ("93.184.216.34", 80), (
            "the socket must connect to the validated IP, not the hostname; "
            f"got {captured.get('address')!r}"
        )
        assert captured.get("timeout") == 7
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P8 -- the redirect chain is bounded
# ---------------------------------------------------------------------------
def test_p8_redirect_chain_is_bounded():
    mod, restore = _load()
    try:
        # An opener that answers 302 forever, always to a public host: only
        # the hop bound stops it.
        resolver = _resolver_for({"loop.example": ["93.184.216.34"]})

        class _Looping:
            # A safety cap far above the bound under test: the code's own hop
            # limit must stop the loop first. If it does not (a broken bound),
            # the cap raises rather than letting the test hang forever.
            _SAFETY_CAP = 50

            def __init__(self):
                self.count = 0

            def __call__(self, url, pinned_ip, headers, timeout):
                self.count += 1
                if self.count > self._SAFETY_CAP:
                    raise RuntimeError(
                        "redirect loop not bounded by the code (safety cap hit)"
                    )
                return _FakeRaw(302, {"Location": "https://loop.example/next"})

        looping = _Looping()
        try:
            mod.urlopen_ssrf_safe(
                "https://loop.example/start",
                resolver=resolver,
                opener=looping,
                max_redirects=3,
            )
        except ValueError as exc:
            assert "redirect" in str(exc).lower(), (
                f"the bound must be reported as a redirect limit, got: {exc}"
            )
        else:
            raise AssertionError("an unbounded redirect chain must be refused")

        # Bounded at max_redirects: the initial hop plus max_redirects follows.
        assert looping.count == 4, (
            f"expected 1 initial + 3 redirect hops before the bound, "
            f"got {looping.count}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
