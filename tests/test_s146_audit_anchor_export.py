#!/usr/bin/env python3
"""
Tests for S146 — Audit Chain External Anchor Export & Verification.

Covers:
- Part 1: AnchorPayload dataclass and serialization
- Part 2: SignedAnchor dataclass and HMAC computation
- Part 3: HMAC helpers (_compute_anchor_hmac, _verify_anchor_hmac)
- Part 4: QR code generation (generate_anchor_qr_png, generate_anchor_qr_base64)
- Part 5: JSON / USB export (generate_anchor_json, generate_anchor_json_bytes)
- Part 6: Clipboard text (generate_anchor_text)
- Part 7: Anchor verification — match cases
- Part 8: Anchor verification — mismatch cases
- Part 9: Anchor verification — HMAC tamper detection
- Part 10: _get_hash_at_entry helper
- Part 11: API endpoint schemas (4 endpoints)
- Part 12: VerifyAnchorRequest Pydantic model
- Part 13: Version bump (3.2.0-rc2)
"""

import base64
import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Isolated module loading (avoids __init__ import chain)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _load_module(name: str, rel_path: str) -> types.ModuleType:
    """Load a module by file path without triggering __init__."""
    full = _PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full))
    assert spec and spec.loader, f"Cannot load {full}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load db_utils stub so signed_audit_log can import it
def _ensure_db_utils():
    """Ensure db_utils is loadable or provide a stub."""
    if "opti_oignon.db_utils" not in sys.modules:
        try:
            _load_module("opti_oignon.db_utils", "opti_oignon/db_utils.py")
        except Exception:
            stub = types.ModuleType("opti_oignon.db_utils")
            stub.safe_connect = lambda *a, **kw: __import__("sqlite3").connect(":memory:")
            sys.modules["opti_oignon.db_utils"] = stub

_ensure_db_utils()

# Load the module under test
_mod = _load_module(
    "opti_oignon.audit_anchor_export",
    "opti_oignon/audit_anchor_export.py",
)

AnchorPayload = _mod.AnchorPayload
SignedAnchor = _mod.SignedAnchor
VerificationResult = _mod.VerificationResult
_compute_anchor_hmac = _mod._compute_anchor_hmac
_verify_anchor_hmac = _mod._verify_anchor_hmac
_build_anchor_payload = _mod._build_anchor_payload
_build_signed_anchor = _mod._build_signed_anchor
generate_anchor_qr_png = _mod.generate_anchor_qr_png
generate_anchor_qr_base64 = _mod.generate_anchor_qr_base64
generate_anchor_json = _mod.generate_anchor_json
generate_anchor_json_bytes = _mod.generate_anchor_json_bytes
generate_anchor_text = _mod.generate_anchor_text
verify_anchor = _mod.verify_anchor
_get_hash_at_entry = _mod._get_hash_at_entry
_ANCHOR_VERSION = _mod._ANCHOR_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_chain(entry_count: int = 5, tip_hash: str = "abc123def456"):
    """Create a mock chain object matching SignedAuditLog interface."""
    chain = MagicMock()
    chain.entry_count.return_value = entry_count
    chain._get_tip_hash.return_value = tip_hash
    chain._db_path = ":memory:"
    return chain


# =========================================================================
# Part 1: AnchorPayload dataclass
# =========================================================================

class TestAnchorPayload(unittest.TestCase):
    """Part 1 — AnchorPayload dataclass and serialization."""

    def test_creation(self):
        p = AnchorPayload(
            chain_tip_hash="aaa", entry_count=10,
            timestamp=1000.0, version="3.2.0-rc2",
        )
        self.assertEqual(p.chain_tip_hash, "aaa")
        self.assertEqual(p.entry_count, 10)
        self.assertEqual(p.timestamp, 1000.0)
        self.assertEqual(p.version, "3.2.0-rc2")
        self.assertEqual(p.anchor_version, _ANCHOR_VERSION)

    def test_to_dict(self):
        p = AnchorPayload("h", 5, 100.0, "v1")
        d = p.to_dict()
        self.assertIn("chain_tip_hash", d)
        self.assertIn("entry_count", d)
        self.assertIn("timestamp", d)
        self.assertIn("version", d)
        self.assertIn("anchor_version", d)
        self.assertEqual(d["chain_tip_hash"], "h")

    def test_to_json_deterministic(self):
        p = AnchorPayload("h", 5, 100.0, "v1")
        j1 = p.to_json()
        j2 = p.to_json()
        self.assertEqual(j1, j2)
        # Must be valid JSON
        parsed = json.loads(j1)
        self.assertEqual(parsed["entry_count"], 5)

    def test_to_json_sorted_keys(self):
        p = AnchorPayload("h", 5, 100.0, "v1")
        j = p.to_json()
        parsed = json.loads(j)
        keys = list(parsed.keys())
        self.assertEqual(keys, sorted(keys))

    def test_default_anchor_version(self):
        p = AnchorPayload("x", 1, 0.0, "v")
        self.assertEqual(p.anchor_version, 1)


# =========================================================================
# Part 2: SignedAnchor dataclass
# =========================================================================

class TestSignedAnchor(unittest.TestCase):
    """Part 2 — SignedAnchor dataclass and HMAC."""

    def test_creation(self):
        p = AnchorPayload("h", 5, 100.0, "v1")
        s = SignedAnchor(payload=p, hmac_sha256="deadbeef")
        self.assertEqual(s.hmac_sha256, "deadbeef")
        self.assertEqual(s.payload.entry_count, 5)

    def test_to_dict_includes_hmac(self):
        p = AnchorPayload("h", 5, 100.0, "v1")
        s = SignedAnchor(payload=p, hmac_sha256="abc")
        d = s.to_dict()
        self.assertIn("hmac_sha256", d)
        self.assertEqual(d["hmac_sha256"], "abc")
        # Also includes payload fields
        self.assertIn("chain_tip_hash", d)

    def test_to_json(self):
        p = AnchorPayload("h", 5, 100.0, "v1")
        s = SignedAnchor(payload=p, hmac_sha256="abc")
        j = s.to_json()
        parsed = json.loads(j)
        self.assertIn("hmac_sha256", parsed)
        self.assertIn("chain_tip_hash", parsed)


# =========================================================================
# Part 3: HMAC helpers
# =========================================================================

class TestHMACHelpers(unittest.TestCase):
    """Part 3 — _compute_anchor_hmac and _verify_anchor_hmac."""

    def test_compute_returns_hex_string(self):
        mac = _compute_anchor_hmac('{"test":1}')
        self.assertIsInstance(mac, str)
        self.assertEqual(len(mac), 64)  # SHA-256 hex

    def test_compute_deterministic(self):
        mac1 = _compute_anchor_hmac('{"a":1}')
        mac2 = _compute_anchor_hmac('{"a":1}')
        self.assertEqual(mac1, mac2)

    def test_compute_different_inputs(self):
        mac1 = _compute_anchor_hmac('{"a":1}')
        mac2 = _compute_anchor_hmac('{"a":2}')
        self.assertNotEqual(mac1, mac2)

    def test_verify_valid(self):
        payload = '{"test":"data"}'
        mac = _compute_anchor_hmac(payload)
        self.assertTrue(_verify_anchor_hmac(payload, mac))

    def test_verify_invalid(self):
        self.assertFalse(_verify_anchor_hmac('{"test":"data"}', "bad_hmac"))

    def test_verify_tampered_payload(self):
        mac = _compute_anchor_hmac('{"original":true}')
        self.assertFalse(_verify_anchor_hmac('{"tampered":true}', mac))


# =========================================================================
# Part 4: QR code generation
# =========================================================================

class TestQRCodeGeneration(unittest.TestCase):
    """Part 4 — generate_anchor_qr_png and generate_anchor_qr_base64."""

    def test_qr_png_returns_bytes(self):
        chain = _make_mock_chain(entry_count=3, tip_hash="abc" * 40)
        png = generate_anchor_qr_png(chain, "3.2.0-rc2")
        self.assertIsInstance(png, bytes)
        # PNG magic bytes
        self.assertTrue(png[:4] == b"\x89PNG")

    def test_qr_png_empty_chain_raises(self):
        chain = _make_mock_chain(entry_count=0)
        with self.assertRaises(RuntimeError):
            generate_anchor_qr_png(chain, "3.2.0-rc2")

    def test_qr_base64_structure(self):
        chain = _make_mock_chain(entry_count=5, tip_hash="def" * 40)
        result = generate_anchor_qr_base64(chain, "3.2.0-rc2")
        self.assertIn("qr_base64", result)
        self.assertIn("payload", result)
        self.assertIn("content_type", result)
        self.assertEqual(result["content_type"], "image/png")

    def test_qr_base64_decodable(self):
        chain = _make_mock_chain(entry_count=2, tip_hash="aaa" * 40)
        result = generate_anchor_qr_base64(chain, "3.2.0-rc2")
        raw = base64.b64decode(result["qr_base64"])
        self.assertTrue(raw[:4] == b"\x89PNG")

    def test_qr_payload_contains_required_fields(self):
        chain = _make_mock_chain(entry_count=7, tip_hash="bbb" * 40)
        result = generate_anchor_qr_base64(chain, "3.2.0-rc2")
        payload = result["payload"]
        self.assertIn("chain_tip_hash", payload)
        self.assertIn("entry_count", payload)
        self.assertIn("timestamp", payload)
        self.assertIn("version", payload)
        self.assertEqual(payload["entry_count"], 7)


# =========================================================================
# Part 5: JSON / USB export
# =========================================================================

class TestJSONExport(unittest.TestCase):
    """Part 5 — generate_anchor_json and generate_anchor_json_bytes."""

    def test_json_structure(self):
        chain = _make_mock_chain(entry_count=10, tip_hash="fff" * 40)
        data = generate_anchor_json(chain, "3.2.0-rc2")
        self.assertIn("chain_tip_hash", data)
        self.assertIn("entry_count", data)
        self.assertIn("timestamp", data)
        self.assertIn("version", data)
        self.assertIn("hmac_sha256", data)
        self.assertEqual(data["entry_count"], 10)

    def test_json_hmac_valid(self):
        chain = _make_mock_chain(entry_count=10, tip_hash="fff" * 40)
        data = generate_anchor_json(chain, "3.2.0-rc2")
        # Reconstruct payload and verify HMAC
        payload_data = {k: v for k, v in data.items() if k != "hmac_sha256"}
        p = AnchorPayload(**payload_data)
        self.assertTrue(_verify_anchor_hmac(p.to_json(), data["hmac_sha256"]))

    def test_json_empty_chain_raises(self):
        chain = _make_mock_chain(entry_count=0)
        with self.assertRaises(RuntimeError):
            generate_anchor_json(chain, "3.2.0-rc2")

    def test_json_bytes_is_bytes(self):
        chain = _make_mock_chain(entry_count=3, tip_hash="ccc" * 40)
        raw = generate_anchor_json_bytes(chain, "3.2.0-rc2")
        self.assertIsInstance(raw, bytes)
        parsed = json.loads(raw)
        self.assertIn("hmac_sha256", parsed)

    def test_json_bytes_pretty_printed(self):
        chain = _make_mock_chain(entry_count=3, tip_hash="ccc" * 40)
        raw = generate_anchor_json_bytes(chain, "3.2.0-rc2")
        text = raw.decode("utf-8")
        # Pretty printed means it has newlines
        self.assertIn("\n", text)


# =========================================================================
# Part 6: Clipboard text
# =========================================================================

class TestClipboardText(unittest.TestCase):
    """Part 6 — generate_anchor_text."""

    def test_text_contains_header(self):
        chain = _make_mock_chain(entry_count=5, tip_hash="abc" * 40)
        text = generate_anchor_text(chain, "3.2.0-rc2")
        self.assertIn("OPTI-OIGNON AUDIT ANCHOR", text)
        self.assertIn("========================", text)

    def test_text_contains_fields(self):
        chain = _make_mock_chain(entry_count=42, tip_hash="xyz" * 40)
        text = generate_anchor_text(chain, "3.2.0-rc2")
        self.assertIn("Chain Tip Hash", text)
        self.assertIn("Entry Count", text)
        self.assertIn("42", text)
        self.assertIn("Timestamp", text)
        self.assertIn("Version", text)
        self.assertIn("3.2.0-rc2", text)
        self.assertIn("HMAC-SHA256", text)

    def test_text_contains_tip_hash(self):
        tip = "a1b2c3" * 20
        chain = _make_mock_chain(entry_count=1, tip_hash=tip)
        text = generate_anchor_text(chain, "v1")
        self.assertIn(tip, text)

    def test_text_timestamp_iso_format(self):
        chain = _make_mock_chain(entry_count=1, tip_hash="aaa")
        text = generate_anchor_text(chain, "v1")
        # Should contain a UTC timestamp like 2026-03-17T...Z
        self.assertIn("T", text)
        self.assertIn("Z", text)


# =========================================================================
# Part 7: Verification — match cases
# =========================================================================

class TestVerificationMatch(unittest.TestCase):
    """Part 7 — verify_anchor match scenarios."""

    def test_exact_match(self):
        """Same count and same tip hash."""
        chain = _make_mock_chain(entry_count=5, tip_hash="abc123")
        anchor = {
            "chain_tip_hash": "abc123",
            "entry_count": 5,
            "timestamp": 1000.0,
            "version": "3.2.0-rc2",
        }
        result = verify_anchor(chain, anchor, "3.2.0-rc2")
        self.assertTrue(result.match)
        self.assertIn("verified", result.details.lower())

    def test_chain_grown_match(self):
        """Chain grew since anchor, but historical tip still matches."""
        chain = _make_mock_chain(entry_count=10, tip_hash="current_tip")
        anchor = {
            "chain_tip_hash": "old_tip",
            "entry_count": 5,
        }
        # Mock _get_hash_at_entry to return the anchor's tip
        with patch.object(_mod, "_get_hash_at_entry", return_value="old_tip"):
            result = verify_anchor(chain, anchor, "v1")
        self.assertTrue(result.match)
        self.assertIn("grown", result.details.lower())

    def test_match_with_valid_hmac(self):
        """Match with valid HMAC present."""
        chain = _make_mock_chain(entry_count=5, tip_hash="abc123")
        # Build a proper signed anchor
        p = AnchorPayload("abc123", 5, 1000.0, "v1")
        mac = _compute_anchor_hmac(p.to_json())
        anchor = {**p.to_dict(), "hmac_sha256": mac}
        result = verify_anchor(chain, anchor, "v1")
        self.assertTrue(result.match)
        self.assertTrue(result.hmac_valid)


# =========================================================================
# Part 8: Verification — mismatch cases
# =========================================================================

class TestVerificationMismatch(unittest.TestCase):
    """Part 8 — verify_anchor mismatch scenarios."""

    def test_truncation_detected(self):
        """Chain shorter than anchor — truncation."""
        chain = _make_mock_chain(entry_count=3, tip_hash="short")
        anchor = {"chain_tip_hash": "long_tip", "entry_count": 10}
        result = verify_anchor(chain, anchor, "v1")
        self.assertFalse(result.match)
        self.assertIn("truncat", result.details.lower())

    def test_tip_hash_mismatch_same_count(self):
        """Same count but different tip — chain rewritten."""
        chain = _make_mock_chain(entry_count=5, tip_hash="new_hash")
        anchor = {"chain_tip_hash": "old_hash", "entry_count": 5}
        result = verify_anchor(chain, anchor, "v1")
        self.assertFalse(result.match)
        self.assertIn("mismatch", result.details.lower())

    def test_chain_grown_but_historical_mismatch(self):
        """Chain grew but hash at old position doesn't match."""
        chain = _make_mock_chain(entry_count=10, tip_hash="current")
        anchor = {"chain_tip_hash": "expected_old", "entry_count": 5}
        with patch.object(_mod, "_get_hash_at_entry", return_value="different"):
            result = verify_anchor(chain, anchor, "v1")
        self.assertFalse(result.match)

    def test_result_contains_counts(self):
        chain = _make_mock_chain(entry_count=3, tip_hash="x")
        anchor = {"chain_tip_hash": "y", "entry_count": 3}
        result = verify_anchor(chain, anchor, "v1")
        self.assertEqual(result.current_entry_count, 3)
        self.assertEqual(result.anchor_entry_count, 3)
        self.assertEqual(result.current_tip_hash, "x")
        self.assertEqual(result.anchor_tip_hash, "y")


# =========================================================================
# Part 9: Verification — HMAC tamper detection
# =========================================================================

class TestVerificationHMAC(unittest.TestCase):
    """Part 9 — HMAC tamper detection in verify_anchor."""

    def test_invalid_hmac_rejected(self):
        chain = _make_mock_chain(entry_count=5, tip_hash="abc")
        anchor = {
            "chain_tip_hash": "abc",
            "entry_count": 5,
            "timestamp": 1000.0,
            "version": "v1",
            "hmac_sha256": "definitely_wrong",
        }
        result = verify_anchor(chain, anchor, "v1")
        self.assertFalse(result.match)
        self.assertFalse(result.hmac_valid)
        self.assertIn("tamper", result.details.lower())

    def test_valid_hmac_accepted(self):
        chain = _make_mock_chain(entry_count=5, tip_hash="abc")
        p = AnchorPayload("abc", 5, 1000.0, "v1")
        mac = _compute_anchor_hmac(p.to_json())
        anchor = {**p.to_dict(), "hmac_sha256": mac}
        result = verify_anchor(chain, anchor, "v1")
        self.assertTrue(result.hmac_valid)

    def test_no_hmac_field_skips_check(self):
        chain = _make_mock_chain(entry_count=5, tip_hash="abc")
        anchor = {"chain_tip_hash": "abc", "entry_count": 5}
        result = verify_anchor(chain, anchor, "v1")
        self.assertIsNone(result.hmac_valid)
        self.assertTrue(result.match)

    def test_hmac_tampered_payload(self):
        """HMAC computed on original, but payload fields changed."""
        p = AnchorPayload("original_hash", 5, 1000.0, "v1")
        mac = _compute_anchor_hmac(p.to_json())
        # Tamper: change entry count
        anchor = {**p.to_dict(), "entry_count": 999, "hmac_sha256": mac}
        chain = _make_mock_chain(entry_count=999, tip_hash="original_hash")
        result = verify_anchor(chain, anchor, "v1")
        self.assertFalse(result.match)
        self.assertFalse(result.hmac_valid)


# =========================================================================
# Part 10: _get_hash_at_entry helper
# =========================================================================

class TestGetHashAtEntry(unittest.TestCase):
    """Part 10 — _get_hash_at_entry with real SQLite."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "test_chain.db")
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE audit_chain (
                id INTEGER PRIMARY KEY,
                entry_hash TEXT NOT NULL
            )
        """)
        conn.execute("INSERT INTO audit_chain (id, entry_hash) VALUES (1, 'hash_one')")
        conn.execute("INSERT INTO audit_chain (id, entry_hash) VALUES (2, 'hash_two')")
        conn.execute("INSERT INTO audit_chain (id, entry_hash) VALUES (3, 'hash_three')")
        conn.commit()
        conn.close()

    def test_get_existing_entry(self):
        chain = MagicMock()
        chain._db_path = self._db_path
        result = _get_hash_at_entry(chain, 2)
        self.assertEqual(result, "hash_two")

    def test_get_first_entry(self):
        chain = MagicMock()
        chain._db_path = self._db_path
        result = _get_hash_at_entry(chain, 1)
        self.assertEqual(result, "hash_one")

    def test_get_nonexistent_entry(self):
        chain = MagicMock()
        chain._db_path = self._db_path
        result = _get_hash_at_entry(chain, 999)
        self.assertEqual(result, "")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# =========================================================================
# Part 11: API endpoint schemas
# =========================================================================

class TestAPIEndpointSchemas(unittest.TestCase):
    """Part 11 — Verify API endpoints exist and have correct paths."""

    def _get_routes(self):
        """Extract route info from routes_security module."""
        mod_path = _PROJECT_ROOT / "opti_oignon" / "api" / "routes_security.py"
        with open(mod_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    def test_export_qr_endpoint_exists(self):
        content = self._get_routes()
        self.assertIn('"/audit/export-qr"', content)
        self.assertIn("async def audit_export_qr", content)

    def test_export_anchor_endpoint_exists(self):
        content = self._get_routes()
        self.assertIn('"/audit/export-anchor"', content)
        self.assertIn("async def audit_export_anchor", content)

    def test_anchor_text_endpoint_exists(self):
        content = self._get_routes()
        self.assertIn('"/audit/anchor-text"', content)
        self.assertIn("async def audit_anchor_text", content)

    def test_verify_anchor_endpoint_exists(self):
        content = self._get_routes()
        self.assertIn('"/audit/verify-anchor"', content)
        self.assertIn("async def audit_verify_anchor", content)

    def test_verify_anchor_uses_pydantic_model(self):
        content = self._get_routes()
        self.assertIn("VerifyAnchorRequest", content)

    def test_qr_endpoint_is_post(self):
        content = self._get_routes()
        # Find the decorator before the function
        idx = content.find("async def audit_export_qr")
        before = content[max(0, idx - 200):idx]
        self.assertIn('@router.post("/audit/export-qr")', before)

    def test_anchor_text_endpoint_is_get(self):
        content = self._get_routes()
        idx = content.find("async def audit_anchor_text")
        before = content[max(0, idx - 200):idx]
        self.assertIn('@router.get("/audit/anchor-text")', before)


# =========================================================================
# Part 12: VerifyAnchorRequest Pydantic model
# =========================================================================

class TestVerifyAnchorRequest(unittest.TestCase):
    """Part 12 — VerifyAnchorRequest Pydantic model validation."""

    @classmethod
    def setUpClass(cls):
        """Load the Pydantic model from routes_security."""
        try:
            routes_path = _PROJECT_ROOT / "opti_oignon" / "api" / "routes_security.py"
            content = routes_path.read_text(encoding="utf-8")
            # Check it's defined
            assert "class VerifyAnchorRequest" in content
            cls._model_defined = True
        except Exception:
            cls._model_defined = False

    def test_model_exists(self):
        self.assertTrue(self._model_defined)

    def test_model_has_required_fields(self):
        content = (_PROJECT_ROOT / "opti_oignon" / "api" / "routes_security.py").read_text()
        self.assertIn("chain_tip_hash", content)
        self.assertIn("entry_count", content)
        self.assertIn("hmac_sha256", content)

    def test_hmac_field_is_optional(self):
        content = (_PROJECT_ROOT / "opti_oignon" / "api" / "routes_security.py").read_text()
        # Find the model definition area
        idx = content.find("class VerifyAnchorRequest")
        model_area = content[idx:idx + 800]
        self.assertIn("Optional[str]", model_area)
        self.assertIn("default=None", model_area)


# =========================================================================
# Part 13: Version bump check
# =========================================================================

class TestVersionBump(unittest.TestCase):
    """Part 13 — Version must be 3.2.0-rc2."""

    def test_version_file(self):
        version_path = _PROJECT_ROOT / "opti_oignon" / "__version__.py"
        content = version_path.read_text(encoding="utf-8")
        self.assertIn('3.2.0-rc2', content)

    def test_version_value(self):
        mod = _load_module(
            "opti_oignon.__version__check",
            "opti_oignon/__version__.py",
        )
        self.assertEqual(mod.__version__, "3.2.0-rc2")


# =========================================================================
# Part 14: VerificationResult dataclass
# =========================================================================

class TestVerificationResult(unittest.TestCase):
    """Part 14 — VerificationResult dataclass."""

    def test_creation(self):
        r = VerificationResult(
            match=True, details="OK",
            current_entry_count=5, anchor_entry_count=5,
            current_tip_hash="a", anchor_tip_hash="a",
        )
        self.assertTrue(r.match)
        self.assertIsNone(r.hmac_valid)

    def test_to_dict(self):
        r = VerificationResult(
            match=False, details="bad",
            current_entry_count=3, anchor_entry_count=10,
            current_tip_hash="x", anchor_tip_hash="y",
            hmac_valid=False,
        )
        d = r.to_dict()
        self.assertIn("match", d)
        self.assertIn("details", d)
        self.assertIn("hmac_valid", d)
        self.assertFalse(d["match"])
        self.assertFalse(d["hmac_valid"])

    def test_to_dict_keys(self):
        r = VerificationResult(
            match=True, details="ok",
            current_entry_count=1, anchor_entry_count=1,
            current_tip_hash="a", anchor_tip_hash="a",
        )
        d = r.to_dict()
        expected_keys = {
            "match", "details", "current_entry_count",
            "anchor_entry_count", "current_tip_hash",
            "anchor_tip_hash", "hmac_valid",
        }
        self.assertEqual(set(d.keys()), expected_keys)


# =========================================================================
# Part 15: _build helpers
# =========================================================================

class TestBuildHelpers(unittest.TestCase):
    """Part 15 — _build_anchor_payload and _build_signed_anchor."""

    def test_build_payload(self):
        chain = _make_mock_chain(entry_count=7, tip_hash="tip777")
        p = _build_anchor_payload(chain, "3.2.0-rc2")
        self.assertEqual(p.entry_count, 7)
        self.assertEqual(p.chain_tip_hash, "tip777")
        self.assertEqual(p.version, "3.2.0-rc2")
        self.assertGreater(p.timestamp, 0)

    def test_build_payload_empty_chain(self):
        chain = _make_mock_chain(entry_count=0, tip_hash="")
        p = _build_anchor_payload(chain, "v1")
        self.assertEqual(p.entry_count, 0)
        self.assertEqual(p.chain_tip_hash, "")

    def test_build_signed_anchor(self):
        chain = _make_mock_chain(entry_count=5, tip_hash="signed_tip")
        sa = _build_signed_anchor(chain, "v1")
        self.assertIsInstance(sa, SignedAnchor)
        self.assertEqual(sa.payload.entry_count, 5)
        self.assertEqual(len(sa.hmac_sha256), 64)

    def test_build_signed_anchor_hmac_valid(self):
        chain = _make_mock_chain(entry_count=5, tip_hash="signed_tip")
        sa = _build_signed_anchor(chain, "v1")
        self.assertTrue(
            _verify_anchor_hmac(sa.payload.to_json(), sa.hmac_sha256)
        )


if __name__ == "__main__":
    unittest.main()
