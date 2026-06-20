"""S185 audit fix MM-01 -- model-download SSRF (redirect following + rebinding).

``_validate_download_url`` (S136) only validated the original URL. The download
then used ``urllib.request.urlopen``, whose default opener followed HTTP
redirects whose Location was never re-validated, and which re-resolved DNS
independently of validation. Two bypasses defeated the control:

  - redirect following: a public host could 302 to ``http://127.0.0.1/...`` or
    the cloud-metadata address, followed past the check;
  - DNS rebinding / TOCTOU: a name whose A record flips between the validation
    resolve and the connect resolve reached a private IP after passing.

The fix (``urlopen_ssrf_safe``) follows redirects manually -- legitimate CDN
redirects (HuggingFace -> S3/CloudFront) still work -- re-validating and
re-pinning every hop, and pins the connection to the validated IP. TLS SNI and
certificate verification still run against the original hostname.

These tests drive the validation + redirect loop with an injected resolver and
opener, so no real socket, DNS, or TLS is touched. The socket-level pinning of
``_PinnedHTTPSConnection`` is exercised only on the live machine (the shakedown).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "model_manager.py"


def _load():
    spec = importlib.util.spec_from_file_location("model_manager_mm01", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12 dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


mm = _load()


# ---------------------------------------------------------------------------
# Fake resolver helpers (getaddrinfo-shaped tuples) and a fake opener
# ---------------------------------------------------------------------------

def _info(ip, port):
    # (family, type, proto, canonname, sockaddr)
    return (2, 1, 6, "", (ip, port))


def _resolver_const(ip):
    def _r(host, port):
        return [_info(ip, port)]
    return _r


_PUBLIC = "93.184.216.34"
_PRIVATE = "127.0.0.1"


class _FakeResp:
    def __init__(self, status, headers, body=b""):
        self.status = status
        self.headers = headers
        self._body = body
        self._off = 0
        self._conn_to_close = None
        self.closed = False

    def read(self, n=-1):
        if n is None or n < 0:
            chunk = self._body[self._off:]
            self._off = len(self._body)
            return chunk
        chunk = self._body[self._off:self._off + n]
        self._off += n
        return chunk

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# IP classification
# ---------------------------------------------------------------------------

def test_ip_is_blocked_covers_internal_ranges():
    for blocked in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1",
                    "169.254.1.1", "::1", "0.0.0.0"):
        assert mm._ip_is_blocked(blocked) is True
    for ok in ("93.184.216.34", "8.8.8.8", "1.1.1.1"):
        assert mm._ip_is_blocked(ok) is False


# ---------------------------------------------------------------------------
# The public early-reject gate keeps its contract (S136)
# ---------------------------------------------------------------------------

def test_validate_download_url_rejects_non_https():
    with pytest.raises(ValueError):
        mm.ModelManager._validate_download_url("ftp://example.com/x.gguf")


def test_validate_and_resolve_rejects_initial_private():
    with pytest.raises(ValueError):
        mm._validate_and_resolve(
            "https://evil.test/x.gguf", resolver=_resolver_const(_PRIVATE)
        )


def test_validate_and_resolve_returns_pinned_ip_for_public():
    scheme, host, port, ip = mm._validate_and_resolve(
        "https://cdn.test/model.gguf", resolver=_resolver_const(_PUBLIC)
    )
    assert scheme == "https"
    assert host == "cdn.test"
    assert port == 443
    assert ip == _PUBLIC


# ---------------------------------------------------------------------------
# Redirect to a private IP is rejected on the next hop's validation
# ---------------------------------------------------------------------------

def test_redirect_to_private_ip_is_rejected():
    def resolver_map(host, port):
        if host == "internal.test":
            return [_info(_PRIVATE, port)]
        return [_info(_PUBLIC, port)]

    def opener(url, pinned_ip, headers, timeout):
        return _FakeResp(302, {"Location": "https://internal.test/secret"})

    with pytest.raises(ValueError):
        mm.urlopen_ssrf_safe(
            "https://cdn.test/model.gguf",
            resolver=resolver_map,
            opener=opener,
        )


# ---------------------------------------------------------------------------
# DNS rebinding: public on the first resolve, private on the second
# ---------------------------------------------------------------------------

def test_dns_rebinding_is_rejected_on_second_resolve():
    calls = {"n": 0}

    def resolver_rebind(host, port):
        calls["n"] += 1
        return [_info(_PUBLIC if calls["n"] == 1 else _PRIVATE, port)]

    # A redirect back to the same host forces a second validation/resolve.
    def opener(url, pinned_ip, headers, timeout):
        return _FakeResp(302, {"Location": "https://same.test/again"})

    with pytest.raises(ValueError):
        mm.urlopen_ssrf_safe(
            "https://same.test/model.gguf",
            resolver=resolver_rebind,
            opener=opener,
        )
    assert calls["n"] >= 2  # the second resolve is what catches the flip


# ---------------------------------------------------------------------------
# Legitimate public -> public redirect chain is followed to completion
# ---------------------------------------------------------------------------

def test_public_redirect_chain_is_followed():
    def opener(url, pinned_ip, headers, timeout):
        if url.endswith("/model.gguf"):
            return _FakeResp(302, {"Location": "https://cdn2.test/blob"})
        return _FakeResp(200, {"Content-Length": "4"}, b"DATA")

    resp = mm.urlopen_ssrf_safe(
        "https://cdn.test/model.gguf",
        resolver=_resolver_const(_PUBLIC),
        opener=opener,
    )
    assert resp.status == 200
    assert resp.read() == b"DATA"


# ---------------------------------------------------------------------------
# A redirect loop is bounded
# ---------------------------------------------------------------------------

def test_redirect_loop_is_bounded():
    def opener(url, pinned_ip, headers, timeout):
        # Always redirect to a fresh public host -> never terminates on its own.
        return _FakeResp(302, {"Location": "https://cdn.test/next"})

    with pytest.raises(ValueError):
        mm.urlopen_ssrf_safe(
            "https://cdn.test/model.gguf",
            resolver=_resolver_const(_PUBLIC),
            opener=opener,
            max_redirects=3,
        )


def test_redirect_without_location_is_rejected():
    def opener(url, pinned_ip, headers, timeout):
        return _FakeResp(302, {})  # no Location header

    with pytest.raises(ValueError):
        mm.urlopen_ssrf_safe(
            "https://cdn.test/model.gguf",
            resolver=_resolver_const(_PUBLIC),
            opener=opener,
        )
