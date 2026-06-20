"""S185 audit fix BK-01 -- backup restore requires a signature when keyed.

import_backup validated structure then called _verify_backup_pqc, which returns
True (verified), False (present but invalid), or None (no signature OR PQC
unavailable). The importer rejected only on False and proceeded on None, so an
unsigned backup -- or a signed one with the signature field stripped -- passed
the integrity gate and was applied (with strategy=replace it overwrites
security-relevant sections: routing, plugins, sandbox, presets).

The fix keeps the True/False/None contract of _verify_backup_pqc but adds the
policy in import_backup: a present-but-invalid signature is always rejected; a
None result is rejected when this install has a PQC keypair (distinguishing
"no signature" from "PQC unavailable" in the message), unless the caller passes
an explicit allow_unsigned=True override. The override never relaxes a failed
verification.

The module loads in isolation (opti_oignon stubbed; liboqs absent, so the PQC
import falls back). Backups use empty sections so the apply loop is a no-op and
the test isolates the signature policy. _verify_backup_pqc is monkeypatched for
the verified/invalid cases; pqc_keypair_exists and _PQC_LIB_AVAILABLE drive the
None policy.
"""

import importlib.util
import sys
import types
from pathlib import Path

sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "backup_manager.py"


def _load():
    spec = importlib.util.spec_from_file_location("backup_manager_bk01", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12 dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


bm_mod = _load()


def _valid_backup(signed: bool = False) -> dict:
    data = {
        "schema_version": "1.0",
        "metadata": {},
        "sections": {},  # empty -> apply loop is a no-op
    }
    if signed:
        data[bm_mod._PQC_SIGNATURE_KEY] = "c2ln"  # base64; content irrelevant here
        data[bm_mod._PQC_PUBLIC_KEY_KEY] = "cHVi"
    return data


def _mgr():
    return bm_mod.BackupManager()


# ---------------------------------------------------------------------------
# Unsigned backups
# ---------------------------------------------------------------------------

def test_unsigned_rejected_when_keypair_exists(monkeypatch):
    monkeypatch.setattr(bm_mod, "pqc_keypair_exists", lambda *a, **k: True)
    result = _mgr().import_backup(_valid_backup(signed=False))
    assert result.success is False
    assert any("unsigned backup refused" in e.lower() for e in result.errors)


def test_unsigned_accepted_with_override(monkeypatch):
    monkeypatch.setattr(bm_mod, "pqc_keypair_exists", lambda *a, **k: True)
    result = _mgr().import_backup(_valid_backup(signed=False), allow_unsigned=True)
    assert result.success is True


def test_unsigned_accepted_when_no_keypair(monkeypatch):
    # PQC not configured here -> backward-compatible allow.
    monkeypatch.setattr(bm_mod, "pqc_keypair_exists", lambda *a, **k: False)
    result = _mgr().import_backup(_valid_backup(signed=False))
    assert result.success is True


# ---------------------------------------------------------------------------
# Signed but unverifiable (PQC library unavailable)
# ---------------------------------------------------------------------------

def test_signed_but_pqc_unavailable_rejected(monkeypatch):
    monkeypatch.setattr(bm_mod, "_PQC_LIB_AVAILABLE", False)
    monkeypatch.setattr(bm_mod, "pqc_keypair_exists", lambda *a, **k: False)
    result = _mgr().import_backup(_valid_backup(signed=True))
    assert result.success is False
    assert any("cannot be verified" in e.lower() for e in result.errors)


def test_signed_but_pqc_unavailable_accepted_with_override(monkeypatch):
    monkeypatch.setattr(bm_mod, "_PQC_LIB_AVAILABLE", False)
    result = _mgr().import_backup(_valid_backup(signed=True), allow_unsigned=True)
    assert result.success is True


# ---------------------------------------------------------------------------
# Verified / tampered (drive _verify_backup_pqc directly)
# ---------------------------------------------------------------------------

def test_valid_signature_accepted():
    mgr = _mgr()
    mgr._verify_backup_pqc = lambda data: True
    result = mgr.import_backup(_valid_backup(signed=True))
    assert result.success is True


def test_invalid_signature_rejected_even_with_override():
    mgr = _mgr()
    mgr._verify_backup_pqc = lambda data: False
    result = mgr.import_backup(_valid_backup(signed=True), allow_unsigned=True)
    assert result.success is False
    assert any("tampered" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# The override parameter exists with a safe default
# ---------------------------------------------------------------------------

def test_allow_unsigned_defaults_false():
    import inspect

    sig = inspect.signature(bm_mod.BackupManager.import_backup)
    assert "allow_unsigned" in sig.parameters
    assert sig.parameters["allow_unsigned"].default is False
