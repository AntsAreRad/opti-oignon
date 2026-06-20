"""
S194 F7b -- export & artifacts fix lot tests.

Covers:
- EXP-01: JSON conversation export carries the real app version
  (was hardcoded "1.6.3"). Supersedes the live-suite assertion
  `data["opti_oignon_version"] == "1.4.0"` in test_live_v130.py
  (live-only, 0 items collected under container pytest); the
  re-assertion lives here per the deselect-plus-reassert protocol.
- EXP-02: HTML export escapes role label / timestamp and whitelists
  the role CSS class.
- EXP-03: HTML content joins with plain newlines (pre-wrap container).
- EXP-04: QR base64 metadata is the exact payload encoded in the QR.
- ART-01: artifact version numbers come from the chain maximum
  (no duplicate versions when titles drift within a chain).

conversation.py does relative imports plus an absolute
`opti_oignon.db_utils` import, so it is loaded with the S193
package-stub idiom extended with a pre-seeded db_utils stub.
"""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
_PKG = "s194b_pkg"


def _read(rel):
    return (_PROJECT / rel).read_text(encoding="utf-8")


def _real_version():
    for line in _read("opti_oignon/__version__.py").splitlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    raise RuntimeError("version not found")


def _seed_stub_package():
    """Pre-seed stub package modules for conversation.py's imports."""
    if _PKG in sys.modules:
        return
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(_PROJECT / "opti_oignon")]
    sys.modules[_PKG] = pkg

    cfg = types.ModuleType(f"{_PKG}.config")
    cfg.DATA_DIR = Path(tempfile.mkdtemp(prefix="s194b_data_"))
    sys.modules[f"{_PKG}.config"] = cfg

    ver = types.ModuleType(f"{_PKG}.__version__")
    ver.__version__ = _real_version()
    sys.modules[f"{_PKG}.__version__"] = ver

    # Absolute import target: opti_oignon.db_utils.safe_connect
    if "opti_oignon" not in sys.modules:
        oo = types.ModuleType("opti_oignon")
        oo.__path__ = []
        sys.modules["opti_oignon"] = oo
    if "opti_oignon.db_utils" not in sys.modules:
        import sqlite3 as _sq3
        dbu = types.ModuleType("opti_oignon.db_utils")
        dbu.safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)
        sys.modules["opti_oignon.db_utils"] = dbu
        sys.modules["opti_oignon"].db_utils = dbu


def _load_pkg_module(sub, rel_path):
    _seed_stub_package()
    name = f"{_PKG}.{sub}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(_PROJECT / rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        # The loaded module keeps direct references to the stubs; drop
        # the real-name entries so later tests in the same process can
        # import the real package without colliding with the stubs.
        for stub in ("opti_oignon.db_utils", "opti_oignon"):
            entry = sys.modules.get(stub)
            if entry is not None and not getattr(entry, "__file__", None):
                del sys.modules[stub]
    return mod


def _load_module(name, rel_path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(_PROJECT / rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


conv_mod = _load_pkg_module("conversation", "opti_oignon/conversation.py")
art_mod = _load_module("s194b_artifacts", "opti_oignon/artifacts.py")
anchor_mod = _load_module(
    "s194b_anchor", "opti_oignon/audit_anchor_export.py"
)


def _make_manager(tmpdir):
    return conv_mod.ConversationManager(
        db_path=Path(tmpdir) / "conv.db"
    )


class _ExportFixtureMixin:
    def _conv_with_messages(self, mgr, messages):
        conv = mgr.create_conversation(title="Export Test")
        for role, content in messages:
            mgr.add_message(conv.id, role=role, content=content)
        return conv


class TestEXP01JsonVersion(_ExportFixtureMixin, unittest.TestCase):
    """EXP-01: JSON export carries the real package version."""

    def test_version_matches_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _make_manager(tmp)
            conv = self._conv_with_messages(
                mgr, [("user", "hi"), ("assistant", "hello")]
            )
            data = json.loads(mgr.export_conversation_json(conv.id))
            self.assertEqual(
                data["opti_oignon_version"], _real_version()
            )
            self.assertNotEqual(data["opti_oignon_version"], "1.6.3")
            self.assertNotEqual(data["opti_oignon_version"], "1.4.0")

    def test_no_hardcoded_version_in_source(self):
        src = _read("opti_oignon/conversation.py")
        self.assertNotIn('"opti_oignon_version": "1.6.3"', src)
        self.assertIn('"opti_oignon_version": _APP_VERSION', src)


class TestEXP02HtmlEscaping(_ExportFixtureMixin, unittest.TestCase):
    """EXP-02: role class whitelisted, role label and timestamp escaped."""

    def test_known_role_classes_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _make_manager(tmp)
            conv = self._conv_with_messages(
                mgr, [("user", "q"), ("assistant", "a")]
            )
            html = mgr.export_conversation_html(conv.id)
            self.assertIn('class="message user"', html)
            self.assertIn('class="message assistant"', html)

    def test_unknown_role_whitelisted_and_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _make_manager(tmp)
            conv = mgr.create_conversation(title="Roles")
            mgr.add_message(
                conv.id, role='x"><script>alert(1)</script>', content="body"
            )
            html = mgr.export_conversation_html(conv.id)
            self.assertIn('class="message other"', html)
            self.assertNotIn("<script>alert(1)</script>", html)


class TestEXP03NewlineJoin(_ExportFixtureMixin, unittest.TestCase):
    """EXP-03: no <br> joins; pre-wrap handles line breaks."""

    def test_prose_message_has_no_br(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _make_manager(tmp)
            conv = self._conv_with_messages(
                mgr, [("user", "line one\nline two\nline three")]
            )
            html = mgr.export_conversation_html(conv.id)
            self.assertNotIn("<br>", html)
            self.assertIn("line one\nline two", html)

    def test_code_block_still_wrapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _make_manager(tmp)
            content = "intro\n```python\nprint(1)\nprint(2)\n```\noutro"
            conv = self._conv_with_messages(mgr, [("user", content)])
            html = mgr.export_conversation_html(conv.id)
            self.assertIn('<pre class="code-block">', html)
            self.assertIn("</code></pre>", html)


class _FakeChain:
    """Minimal SignedAuditLog interface for anchor tests."""

    def __init__(self, count=3, tip="abc123"):
        self._count = count
        self._tip = tip
        self._db_path = ":memory:"

    def entry_count(self):
        return self._count

    def _get_tip_hash(self):
        return self._tip


class TestEXP04QrPayloadConsistency(unittest.TestCase):
    """EXP-04: returned payload metadata equals the QR-encoded payload."""

    def test_single_payload_build(self):
        src = _read("opti_oignon/audit_anchor_export.py")
        self.assertIn("_qr_png_from_payload", src)
        # base64 path builds the payload exactly once
        b64_fn = src.split("def generate_anchor_qr_base64")[1]
        b64_fn = b64_fn.split("\ndef ")[0]
        self.assertEqual(b64_fn.count("_build_anchor_payload"), 1)
        self.assertIn("_qr_png_from_payload(payload)", b64_fn)

    def test_empty_chain_still_raises(self):
        chain = _FakeChain(count=0, tip="")
        with self.assertRaises(RuntimeError):
            anchor_mod._qr_png_from_payload(
                anchor_mod._build_anchor_payload(chain, "3.6.0")
            )

    def test_signed_json_unchanged(self):
        chain = _FakeChain()
        data = anchor_mod.generate_anchor_json(chain, "3.6.0")
        self.assertEqual(data["entry_count"], 3)
        self.assertIn("hmac_sha256", data)


class TestART01VersionChainNumbering(unittest.TestCase):
    """ART-01: versions come from the chain max, never duplicated."""

    def _artifact(self, title, art_type="html"):
        return art_mod.Artifact(
            id="",
            artifact_type=art_type,
            title=title,
            content="<html></html>",
            language="html",
            created_at="2026-06-04T00:00:00",
        )

    def test_no_duplicate_versions_on_title_drift(self):
        mgr = art_mod.ArtifactManager()
        conv_id = "conv-art01"
        mgr._cache[conv_id] = []

        # Chain: root v1 "alpha beta gamma delta", child v2 with a
        # drifted title sharing fewer words.
        root = self._artifact("alpha beta gamma delta")
        root.id = "root0001"
        root.version = 1
        child = self._artifact("alpha beta gamma epsilon zeta")
        child.id = "child001"
        child.version = 2
        child.parent_id = "root0001"
        mgr._cache[conv_id] = [root, child]

        # New artifact whose title is closest to the ROOT (v1): the
        # pre-fix logic took parent.version + 1 = 2 -> duplicate v2.
        new = self._artifact("alpha beta gamma delta")
        parent = mgr._find_version_parent(conv_id, new)
        self.assertIsNotNone(parent)

        # Drive the version-linking path through the public flow.
        detector_backup = mgr._detector.detect
        mgr._detector.detect = lambda text, cid: [new]
        try:
            stored = mgr.detect_and_store("ignored", conv_id)
        finally:
            mgr._detector.detect = detector_backup

        self.assertEqual(len(stored), 1)
        versions = [a.version for a in mgr._cache[conv_id]]
        self.assertEqual(
            len(versions), len(set(versions)),
            f"duplicate versions: {sorted(versions)}",
        )
        self.assertEqual(stored[0].version, 3)
        self.assertEqual(stored[0].parent_id, "root0001")

    def test_same_title_chain_increments_normally(self):
        mgr = art_mod.ArtifactManager()
        conv_id = "conv-art01b"
        mgr._cache[conv_id] = []

        for expected_version in (1, 2, 3):
            art = self._artifact("Dashboard Page")
            detector_backup = mgr._detector.detect
            mgr._detector.detect = lambda text, cid, a=art: [a]
            try:
                stored = mgr.detect_and_store("ignored", conv_id)
            finally:
                mgr._detector.detect = detector_backup
            self.assertEqual(stored[0].version, expected_version)


if __name__ == "__main__":
    unittest.main()
