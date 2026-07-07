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

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
idiom: canonical dotted names under a package stub, the real modules under
test loaded from source.
"""

import hashlib
import hmac as hmac_mod
import importlib.util
import json
import socket
import sqlite3
import sys
import tempfile
import traceback
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_OO = _ROOT / "opti_oignon"


def _load_modules():
    """Load db_utils, signed_audit_log and audit_anchor_export for real.

    Registered under their canonical dotted names beneath a package stub so
    intra-package imports resolve without triggering the full package
    __init__ (which requires the inference client).
    """
    for name in list(sys.modules):
        if name == "opti_oignon" or name.startswith("opti_oignon."):
            del sys.modules[name]

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = [str(_OO)]
    sys.modules["opti_oignon"] = pkg

    def _load(dotted, path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("opti_oignon.db_utils", _OO / "db_utils.py")

    # Importing the chain module instantiates its module-level singleton,
    # which touches the default chain db under the repository root. Leave
    # a real repository's chain strictly alone; only remove artifacts this
    # very load created in a tree that had none.
    default_db = _ROOT / "data" / "audit_chain.db"
    default_anchor = _ROOT / "data" / ".audit_chain_anchor"
    had_db = default_db.exists()
    had_anchor = default_anchor.exists()

    sal = _load("opti_oignon.signed_audit_log", _OO / "signed_audit_log.py")
    aae = _load("opti_oignon.audit_anchor_export", _OO / "audit_anchor_export.py")

    if not had_db and default_db.exists():
        default_db.unlink()
    if not had_anchor and default_anchor.exists():
        default_anchor.unlink()

    return sal, aae


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
    sal, aae = _load_modules()
    with tempfile.TemporaryDirectory() as tmp:
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
    sal, aae = _load_modules()
    with tempfile.TemporaryDirectory() as tmp:
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
    sal, aae = _load_modules()
    with tempfile.TemporaryDirectory() as tmp:
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
    sal, aae = _load_modules()
    with tempfile.TemporaryDirectory() as tmp:
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
        assert bundle["qr_png"][:4] == b"\x89PNG", "QR PNG missing/invalid"

        from_text = aae.parse_anchor_text(bundle["text"])
        assert _canonical_digest(from_text) == ref, "clipboard text diverges"


# ---------------------------------------------------------------------------
# Clause 5 -- the whole path is offline (born-green pin; a directed break proves its teeth)
# ---------------------------------------------------------------------------


def test_c5_export_and_verification_touch_no_network():
    sal, aae = _load_modules()

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


# ---------------------------------------------------------------------------
# Clause 6 -- invalid signature under the same key id is a net rejection
#             (born-green pin; a directed break proves its teeth)
# ---------------------------------------------------------------------------


def test_c6_bit_flipped_signature_is_rejected_outright():
    sal, aae = _load_modules()
    with tempfile.TemporaryDirectory() as tmp:
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
# Runner
# ---------------------------------------------------------------------------

_CLAUSES = [
    "test_c1_keyed_anchor_round_trips_authenticated_on_intact_chain",
    "test_c2_post_anchor_alteration_is_detected_and_localized",
    "test_c3_anchor_forged_from_public_source_is_never_authenticated",
    "test_c4_bundle_renders_the_same_anchor_in_all_three_formats",
    "test_c5_export_and_verification_touch_no_network",
    "test_c6_bit_flipped_signature_is_rejected_outright",
]


def _main() -> int:
    passed = 0
    for name in _CLAUSES:
        try:
            globals()[name]()
        except Exception:
            print(f"FAIL {name}:")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
            passed += 1
    print(f"{passed}/{len(_CLAUSES)} passed")
    return 0 if passed == len(_CLAUSES) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
