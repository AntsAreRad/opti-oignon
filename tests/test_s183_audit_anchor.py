#!/usr/bin/env python3
"""
S183 / A-01: the truncation anchor must be keyed on a secret, not the DB path,
and must not raise a false CRITICAL on a relocated or fresh install.

Verified behaviors:
- the anchor MAC is independent of the DB path (two logs at different paths but
  the same secret produce the same MAC), so a path-only attacker cannot forge it;
- a genuine tamper (same key id, altered content) logs CRITICAL;
- a truncation under an authentic keyed anchor logs CRITICAL;
- a relocated install / key change (different key id) is informational, not
  CRITICAL, and re-anchors;
- a legacy pre-v2 anchor is informational, not CRITICAL, and is rewritten in v2;
- with no master key, a foreign/altered anchor never raises CRITICAL (advisory).
"""

import logging
import os
import sys
import types

import pytest

# Guarded stub so the isolated module load resolves opti_oignon.db_utils.
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

import importlib.util

_mod_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "signed_audit_log.py",
)
_spec = importlib.util.spec_from_file_location("signed_audit_log_s183", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve cls.__module__ in sys.modules
# (the module uses `from __future__ import annotations`; on Python 3.12+ the
# dataclass decorator looks the module up by name during class processing).
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

AuditLog = _mod.SignedAuditLog
_anchor_mac = _mod._anchor_mac
_ANCHOR_FORMAT_VERSION = _mod._ANCHOR_FORMAT_VERSION
_ANCHOR_NOKEY_ID = _mod._ANCHOR_NOKEY_ID

KEY_A = b"A" * 32
KEY_B = b"B" * 32


@pytest.fixture
def make_log(tmp_path, monkeypatch):
    """Build an AuditLog with no master key (plaintext DB, unkeyed anchor)."""
    monkeypatch.delenv("OPTI_ENCRYPTION_KEY", raising=False)
    created = []

    def _factory(name="audit_chain.db"):
        log = AuditLog(db_path=str(tmp_path / name))
        created.append(log)
        return log

    return _factory


def _read_anchor(log):
    return log._get_anchor_path().read_text(encoding="utf-8").strip()


def _has_level(caplog, level):
    return any(r.levelno == level for r in caplog.records)


class TestAnchorNotKeyedOnPath:
    def test_mac_is_path_independent(self, make_log):
        log1 = make_log("a.db")
        log2 = make_log("b.db")
        for log in (log1, log2):
            log._anchor_secret = lambda: (KEY_A, "kidA")
            log._save_anchor(5, "TIP")
        mac1 = _read_anchor(log1).split("|")[-1]
        mac2 = _read_anchor(log2).split("|")[-1]
        # Same key + same content => same MAC regardless of the DB path.
        assert mac1 == mac2

    def test_saved_format_is_v2_with_key_id(self, make_log):
        log = make_log()
        log._anchor_secret = lambda: (KEY_A, "kidA")
        log._save_anchor(3, "TIP")
        parts = _read_anchor(log).split("|")
        assert len(parts) == 5
        assert parts[0] == _ANCHOR_FORMAT_VERSION
        assert parts[1] == "kidA"


class TestGenuineTamperIsCritical:
    def test_altered_content_same_key_is_critical(self, make_log, caplog):
        log = make_log()
        log._anchor_secret = lambda: (KEY_A, "kidA")
        log.append_event("e1", source="t")
        log.append_event("e2", source="t")  # count == 2, anchor under kidA
        # Corrupt the saved tip but keep the key id, so the MAC no longer
        # verifies under the same secret -> genuine tamper.
        parts = _read_anchor(log).split("|")
        parts[3] = "deadbeef"
        log._get_anchor_path().write_text("|".join(parts), encoding="utf-8")
        with caplog.at_level(logging.CRITICAL):
            log._check_anchor(2)
        assert _has_level(caplog, logging.CRITICAL)
        assert any("TAMPERED" in r.getMessage() for r in caplog.records)

    def test_truncation_under_authentic_anchor_is_critical(self, make_log, caplog):
        log = make_log()
        log._anchor_secret = lambda: (KEY_A, "kidA")
        log.append_event("e1", source="t")
        log.append_event("e2", source="t")
        log.append_event("e3", source="t")  # anchor saved for count == 3
        with caplog.at_level(logging.CRITICAL):
            log._check_anchor(1)  # DB now reports fewer entries
        assert any("TRUNCATED" in r.getMessage() for r in caplog.records)


class TestRelocatedOrLegacyIsInformational:
    def test_different_key_id_is_not_critical(self, make_log, caplog):
        log = make_log()
        log._anchor_secret = lambda: (KEY_A, "kidA")
        log.append_event("e1", source="t")  # anchor under kidA
        # Now the environment presents a different secret (relocated/rotated).
        log._anchor_secret = lambda: (KEY_B, "kidB")
        with caplog.at_level(logging.INFO):
            log._check_anchor(1)
        assert not _has_level(caplog, logging.CRITICAL)
        # Re-anchored under the new key id.
        assert _read_anchor(log).split("|")[1] == "kidB"

    def test_legacy_three_part_anchor_is_not_critical(self, make_log, caplog):
        log = make_log()
        log._anchor_secret = lambda: (KEY_A, "kidA")
        log.append_event("e1", source="t")
        # Overwrite with the old pre-v2 format (count|tip|mac).
        log._get_anchor_path().write_text("1|sometip|abcdef", encoding="utf-8")
        with caplog.at_level(logging.INFO):
            log._check_anchor(1)
        assert not _has_level(caplog, logging.CRITICAL)
        # Rewritten in v2 form.
        assert len(_read_anchor(log).split("|")) == 5


class TestUnkeyedModeNeverCritical:
    def test_foreign_anchor_no_critical_without_master_key(self, make_log, caplog):
        log = make_log()  # unkeyed: _anchor_secret returns (None, "nokey")
        log.append_event("e1", source="t")  # anchor written as nokey checksum
        # Tamper with the content under the same (nokey) id.
        parts = _read_anchor(log).split("|")
        assert parts[1] == _ANCHOR_NOKEY_ID
        parts[3] = "deadbeef"
        log._get_anchor_path().write_text("|".join(parts), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            log._check_anchor(1)
        assert not _has_level(caplog, logging.CRITICAL)
