#!/usr/bin/env python3
"""Contracts for the external audit-chain anchor (export and verification).

An anchor externalizes the audit chain tip so an operator can later prove
the chain was not rewritten behind their back. These clauses pin what an
anchor must actually guarantee:

  * A keyed anchor is signed with a key DERIVED FROM A SECRET (the chain's
    own anchor secret), never with material readable in the public source
    tree. Knowledge of the repository alone must never let anyone produce
    an anchor that verification reports as authenticated.
  * Verifying an anchor walks the chain: an entry altered after anchoring
    is detected AND localized to its entry id, even when the stored tip
    hash still matches the anchor (the historical silent-pass case).
  * A signature that fails to verify under the same key id is a net
    rejection.
  * One export operation renders the SAME anchor in all three formats
    (JSON file, QR content, clipboard text): same canonical digest.
  * The whole export + verification path performs no network access.
  * QR rendering is optional. When it is absent the refusal NAMES the extra
    that would satisfy it, so a stock install is never mistaken for a broken
    anchor path; when it is installed, the bytes come from it and not from
    the stand-in this suite renders with elsewhere.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
idiom: canonical dotted names under a package stub, the real modules under
test loaded from source.
"""

import hashlib
import contextlib
import hmac as hmac_mod
import json
import socket
import sqlite3
import sys
import tempfile
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_OO = _ROOT / "opti_oignon"


class _Skipped(BaseException):
    """A clause that cannot run here, carrying what would make it run."""


def _skip(reason):
    """Refuse to run the calling clause, naming what would make it run.

    A clause that cannot execute must be COUNTED as not having executed. The
    one thing it must never do is return quietly, which a reader cannot tell
    apart from a pass. Routed through pytest when pytest is the one driving --
    consulted through the cache rather than imported, so the direct runner
    does not acquire a dependency it never had.
    """
    driver = sys.modules.get("pytest")
    if driver is not None:
        driver.skip(reason)
    raise _Skipped(reason)


# ---------------------------------------------------------------------------
# The optional renderer
#
# QR rendering is optional and is declared as the ``anchor`` extra. Two of the
# clauses below need a bundle to exist but assert nothing at all about the
# rendered bytes -- the offline guarantee in particular has nothing to do with
# drawing an image, and a guarantee that stops being checked on every stock
# install is not a guarantee. They stand the renderer in.
#
# The stand-in deliberately emits something that is NOT a PNG. Any clause that
# means to speak about real rendered output therefore fails loudly against it
# instead of passing on the stand-in's own bytes.
# ---------------------------------------------------------------------------

_STAND_IN_PREFIX = b"not-a-png:stand-in-renderer:"


def _renderer_stand_in():
    """Minimal stand-in for the qrcode package: enough to build, never to draw."""
    module = types.ModuleType("qrcode")

    class _Image:
        def __init__(self, payload):
            self._payload = payload

        def save(self, buf, format=None):  # noqa: A002 - mirrors the real API
            buf.write(_STAND_IN_PREFIX + self._payload)

    class _QRCode:
        def __init__(self, **_kwargs):
            self._data = b""

        def add_data(self, data):
            self._data = data.encode("utf-8") if isinstance(data, str) else data

        def make(self, fit=True):
            return None

        def make_image(self, **_kwargs):
            return _Image(self._data)

    module.QRCode = _QRCode
    module.constants = types.SimpleNamespace(ERROR_CORRECT_M=0)
    return module


def _renderer_installed():
    """True when the real renderer is importable in this environment."""
    try:
        import qrcode  # noqa: F401
    except ImportError:
        return False
    return True


def _load_modules(*, renderer="stand-in"):
    """Load the chain modules through the shared isolation window.

    The window is what makes this suite safe to run beside any other. The
    hand-rolled predecessor DELETED every ``opti_oignon.*`` key from the cache
    and never put one back, then stood in a package whose ``__path__`` was the
    REAL source directory -- so the next real import of ``plugin_hooks`` (or of
    anything else) rebuilt a SECOND module object, with a SECOND module-level
    singleton. No ImportError, no honest red: production writes into one object
    and the test interrogates the other. The window neutralises rather than
    deletes, names every lateral it needs, and restores ``sys.modules`` and
    ``sys.meta_path`` exactly on the way out.

    ``renderer`` selects the posture of the optional QR renderer:
    ``"stand-in"`` seeds the stand-in above, ``"absent"`` declares the name
    unreachable and has the window PROVE it before any target runs, and
    ``"real"`` leaves resolution alone so the installed package answers. The
    window's guard fences project names only, so the real package resolves
    through it untouched.

    Returns ``(signed_audit_log, audit_anchor_export, restore)``.
    """
    # Importing the chain module instantiates its module-level singleton, which
    # touches the default chain db under the repository root. Leave a real
    # repository's chain strictly alone; only remove artifacts this very load
    # created in a tree that had none.
    default_db = _ROOT / "data" / "audit_chain.db"
    default_anchor = _ROOT / "data" / ".audit_chain_anchor"
    had_db = default_db.exists()
    had_anchor = default_anchor.exists()

    # Ordered: a target may import one that precedes it. signed_audit_log binds
    # db_utils.safe_connect at its top; audit_anchor_export binds
    # signed_audit_log._anchor_mac at its top. The three laterals below are
    # reached LAZILY, inside functions, and are loaded from their real sources
    # so this suite exercises exactly the code paths it exercised before.
    window = {}
    if renderer == "stand-in":
        window["seeded"] = {"qrcode": _renderer_stand_in()}
    elif renderer == "absent":
        window["blocked"] = ("qrcode",)

    loaded, restore = isolate(
        targets={
            "opti_oignon.db_utils": source("db_utils.py"),
            "opti_oignon.encryption": source("encryption.py"),
            "opti_oignon.db_encryption": source("db_encryption.py"),
            "opti_oignon.security_mode": source("security_mode.py"),
            "opti_oignon.signed_audit_log": source("signed_audit_log.py"),
            "opti_oignon.audit_anchor_export": source("audit_anchor_export.py"),
        },
        **window,
    )

    if not had_db and default_db.exists():
        default_db.unlink()
    if not had_anchor and default_anchor.exists():
        default_anchor.unlink()

    return (
        loaded["opti_oignon.signed_audit_log"],
        loaded["opti_oignon.audit_anchor_export"],
        restore,
    )


@contextlib.contextmanager
def _loaded_modules(**kwargs):
    """Context-manager form; closes the window on every exit path."""
    sal, aae, restore = _load_modules(**kwargs)
    try:
        yield sal, aae
    finally:
        restore()


def _fresh_chain(sal, tmpdir, n_entries=5):
    """Real SignedAuditLog on a throwaway db with n appended entries."""
    db_path = Path(tmpdir) / "audit_chain.db"
    chain = sal.SignedAuditLog(db_path=db_path)
    for i in range(n_entries):
        chain.append_event(
            event_type="test_event",
            source="contracts",
            action=f"action {i}",
            severity="INFO",
            details={"i": i},
        )
    return chain, db_path


_TEST_KEY = hashlib.sha256(b"contract-only-secret-not-in-tree").digest()
_TEST_KEY_ID = "cafe0123deadbeef"


def _keyed(chain):
    """Force a known keyed anchor secret on this chain instance."""
    chain._anchor_secret = lambda: (_TEST_KEY, _TEST_KEY_ID)
    return chain


def _canonical_digest(anchor_dict):
    """Digest of an anchor dict over canonical JSON (order-insensitive)."""
    return hashlib.sha256(
        json.dumps(anchor_dict, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Clause 1 -- keyed export round-trips as authenticated on an intact chain
# ---------------------------------------------------------------------------


def test_c1_keyed_anchor_round_trips_authenticated_on_intact_chain():
    with _loaded_modules() as (sal, aae), \
            tempfile.TemporaryDirectory() as tmp:
        chain, _ = _fresh_chain(sal, tmp)
        _keyed(chain)

        anchor = aae.generate_anchor_json(chain, "test-version")
        payload = anchor.get("payload", anchor)
        assert "key_id" in payload, (
            "anchor payload carries no key_id: verification cannot bind the "
            "signature to the secret that produced it"
        )
        assert payload["key_id"] == _TEST_KEY_ID

        result = aae.verify_anchor(chain, _flatten(anchor), "test-version")
        assert result.match is True, result.details
        assert result.hmac_valid is True, (
            "keyed anchor on an intact chain must verify as authenticated"
        )
        scheme = getattr(result, "signature_scheme", None)
        assert scheme == "keyed", (
            f"verification must report the keyed scheme, got {scheme!r}"
        )


def _flatten(anchor):
    """Anchor dict as the wire shape: payload fields + hmac at top level."""
    if "payload" in anchor:
        flat = dict(anchor["payload"])
        flat["hmac_sha256"] = anchor.get("hmac_sha256")
        return flat
    return dict(anchor)


# ---------------------------------------------------------------------------
# Clause 2 -- an entry altered after anchoring is detected AND localized
# ---------------------------------------------------------------------------


def test_c2_post_anchor_alteration_is_detected_and_localized():
    with _loaded_modules() as (sal, aae), \
            tempfile.TemporaryDirectory() as tmp:
        chain, db_path = _fresh_chain(sal, tmp, n_entries=6)
        _keyed(chain)

        anchor = aae.generate_anchor_json(chain, "test-version")

        # Tamper a mid-chain row IN PLACE: content changes, stored hashes
        # do not, so the tip still equals the anchored tip. Only a chain
        # walk can see it.
        tampered_id = 3
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE audit_chain SET action = ? WHERE id = ?",
            ("forged action", tampered_id),
        )
        conn.commit()
        conn.close()

        result = aae.verify_anchor(chain, _flatten(anchor), "test-version")
        assert result.match is False, (
            "an anchor must not verify as OK over a chain whose entry "
            f"{tampered_id} was rewritten (tip-only comparison is blind here)"
        )
        located = getattr(result, "first_divergent_entry", None)
        assert located == tampered_id, (
            f"divergence must be localized to entry {tampered_id}, "
            f"got {located!r}"
        )


# ---------------------------------------------------------------------------
# Clause 3 -- repo-public knowledge must never yield an authenticated anchor
# ---------------------------------------------------------------------------


def test_c3_anchor_forged_from_public_source_is_never_authenticated():
    with _loaded_modules() as (sal, aae), \
            tempfile.TemporaryDirectory() as tmp:
        chain, _ = _fresh_chain(sal, tmp)
        _keyed(chain)

        # Everything below is computable by anyone holding the public tree:
        # a fabricated payload plus an HMAC keyed on a byte-string constant
        # that appears verbatim in the source.
        public_constant = b"opti-oignon-audit-anchor-v1"
        forged_payload = {
            "anchor_version": 1,
            "chain_tip_hash": "f" * 128,
            "entry_count": 999,
            "timestamp": 1.0,
            "version": "test-version",
        }
        forged_json = json.dumps(
            forged_payload, separators=(",", ":"), sort_keys=True
        )
        forged_mac = hmac_mod.new(
            public_constant, forged_json.encode(), hashlib.sha256
        ).hexdigest()
        wire = dict(forged_payload)
        wire["hmac_sha256"] = forged_mac

        result = aae.verify_anchor(chain, wire, "test-version")
        assert result.hmac_valid is not True, (
            "a signature computed purely from public source material was "
            "reported as authenticated: the anchor key is not a secret"
        )


# ---------------------------------------------------------------------------
# Clause 4 -- one export, three formats, one canonical anchor digest
# ---------------------------------------------------------------------------


def test_c4_bundle_renders_the_same_anchor_in_all_three_formats():
    with _loaded_modules() as (sal, aae), \
            tempfile.TemporaryDirectory() as tmp:
        chain, _ = _fresh_chain(sal, tmp)
        _keyed(chain)

        bundle_fn = getattr(aae, "export_anchor_bundle", None)
        assert bundle_fn is not None, (
            "no single-build bundle export exists: per-format exports each "
            "rebuild their own payload, so formats cannot carry one anchor"
        )
        bundle = bundle_fn(chain, "test-version")

        ref = _canonical_digest(bundle["anchor"])

        from_json = json.loads(bundle["json_bytes"].decode("utf-8"))
        assert _canonical_digest(from_json) == ref, "JSON file diverges"

        from_qr = json.loads(bundle["qr_content"])
        assert _canonical_digest(from_qr) == ref, "QR content diverges"

        from_text = aae.parse_anchor_text(bundle["text"])
        assert _canonical_digest(from_text) == ref, "clipboard text diverges"


# ---------------------------------------------------------------------------
# Clause 5 -- the whole path is offline (born-green pin; a directed break proves its teeth)
# ---------------------------------------------------------------------------


def test_c5_export_and_verification_touch_no_network():
    sal, aae, restore = _load_modules()

    def _deny(*_a, **_k):
        raise AssertionError("network access attempted on the anchor path")

    saved = (socket.socket, socket.create_connection, socket.getaddrinfo)
    socket.socket = _deny
    socket.create_connection = _deny
    socket.getaddrinfo = _deny
    try:
        with tempfile.TemporaryDirectory() as tmp:
            chain, db_path = _fresh_chain(sal, tmp)
            _keyed(chain)
            bundle_fn = getattr(aae, "export_anchor_bundle", None)
            if bundle_fn is not None:
                bundle = bundle_fn(chain, "test-version")
                anchor = bundle["anchor"]
            else:
                anchor = aae.generate_anchor_json(chain, "test-version")
                aae.generate_anchor_text(chain, "test-version")
                aae.generate_anchor_qr_png(chain, "test-version")
            result = aae.verify_anchor(chain, _flatten(anchor), "test-version")
            assert result.match is True
    finally:
        socket.socket, socket.create_connection, socket.getaddrinfo = saved
        restore()


# ---------------------------------------------------------------------------
# Clause 6 -- invalid signature under the same key id is a net rejection
#             (born-green pin; a directed break proves its teeth)
# ---------------------------------------------------------------------------


def test_c6_bit_flipped_signature_is_rejected_outright():
    with _loaded_modules() as (sal, aae), \
            tempfile.TemporaryDirectory() as tmp:
        chain, _ = _fresh_chain(sal, tmp)
        _keyed(chain)

        anchor = aae.generate_anchor_json(chain, "test-version")
        wire = _flatten(anchor)
        mac = wire["hmac_sha256"]
        flipped = ("0" if mac[0] != "0" else "1") + mac[1:]
        wire["hmac_sha256"] = flipped

        result = aae.verify_anchor(chain, wire, "test-version")
        assert result.match is False, "tampered signature must reject"
        assert result.hmac_valid is False, (
            "hmac_valid must be False on a signature that fails to verify"
        )


# ---------------------------------------------------------------------------
# Clause 7 -- an absent optional capability says which capability it is
# ---------------------------------------------------------------------------


def test_c7_absent_renderer_refuses_under_the_name_that_would_satisfy_it():
    with _loaded_modules(renderer="absent") as (sal, aae), \
            tempfile.TemporaryDirectory() as tmp:
        chain, _ = _fresh_chain(sal, tmp)
        _keyed(chain)

        try:
            aae.export_anchor_bundle(chain, "test-version")
        except ImportError as exc:
            message = str(exc)
        else:
            raise AssertionError(
                "the bundle rendered with the renderer proven unreachable"
            )

        assert "opti-oignon[anchor]" in message, (
            "the refusal does not name what would satisfy it, so an install "
            "that simply lacks the extra is indistinguishable from a broken "
            "anchor path: " + message
        )


# ---------------------------------------------------------------------------
# Clause 8 -- installed, the renderer emits real image bytes
# ---------------------------------------------------------------------------


def test_c8_installed_renderer_emits_image_bytes_not_a_stand_in():
    if not _renderer_installed():
        _skip(
            "the QR renderer is not installed here; "
            "pip install 'opti-oignon[anchor]' makes this clause run"
        )

    with _loaded_modules(renderer="real") as (sal, aae), \
            tempfile.TemporaryDirectory() as tmp:
        chain, _ = _fresh_chain(sal, tmp)
        _keyed(chain)

        png = aae.export_anchor_bundle(chain, "test-version")["qr_png"]
        assert not png.startswith(_STAND_IN_PREFIX), (
            "the stand-in answered where the installed renderer was required"
        )
        assert png[:4] == b"\x89PNG", "QR PNG missing/invalid"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# Derived from the module, not listed beside it. A roster written out by hand
# is stale the moment a clause is added and nobody notices, which is the same
# silence a clause that never runs produces.
_CLAUSES = sorted(name for name in dict(globals()) if name.startswith("test_c"))


def _main() -> int:
    passed = 0
    skipped = 0
    for name in _CLAUSES:
        try:
            globals()[name]()
        except _Skipped as reason:
            print(f"SKIP {name}: {reason}")
            skipped += 1
        except Exception:
            print(f"FAIL {name}:")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
            passed += 1
    total = len(_CLAUSES)
    print(f"{passed}/{total} passed, {skipped} skipped")
    return 0 if passed + skipped == total else 1


if __name__ == "__main__":
    raise SystemExit(_main())
