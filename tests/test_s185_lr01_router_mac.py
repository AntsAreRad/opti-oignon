"""S185 audit fix LR-01 -- learned-router pickle load is authenticated.

``LearnedRouter._try_load_model`` called ``joblib.load(learned_router.pkl)`` at
init whenever the file existed. joblib is pickle under the hood: loading a
tampered or swapped file is arbitrary code execution, and there was no integrity
check. The file is plaintext at rest (not a SQLCipher store) and writable by
anything with FS access (backup-restore, a future Veilid sync of the data dir,
an accidental commit).

The fix verifies a keyed MAC over the file before loading and refuses on
mismatch, missing MAC, or no key (fail-safe skip -- the router falls back to its
heuristic). The MAC subkey is derived from the master key with domain
separation (HMAC-SHA256), the same construction as the SQLCipher subkey;
Kerckhoffs-clean (only the master key is secret).

The MAC helpers (``write_model_mac`` / ``verify_model_mac``) are pure file+HMAC
and tested directly with an explicit key, so no master key, sklearn, or DB is
required. A guarded integration test exercises ``_try_load_model`` end to end
when joblib is installed. A source assertion locks the verify-before-load order.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "learned_router.py"


def _load():
    spec = importlib.util.spec_from_file_location("learned_router_lr01", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12 dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


lr = _load()

_KEY = b"\x11" * 32
_OTHER_KEY = b"\x22" * 32


# ---------------------------------------------------------------------------
# Keyed-MAC helpers
# ---------------------------------------------------------------------------

def test_write_requires_a_key(tmp_path):
    model = tmp_path / "learned_router.pkl"
    model.write_bytes(b"fake-pickle-bytes")
    mac = lr._model_mac_path(model)
    assert lr.write_model_mac(model, mac, None) is False
    assert not mac.exists()
    assert lr.write_model_mac(model, mac, _KEY) is True
    assert mac.exists()


def test_verify_round_trip(tmp_path):
    model = tmp_path / "learned_router.pkl"
    model.write_bytes(b"fake-pickle-bytes")
    mac = lr._model_mac_path(model)
    lr.write_model_mac(model, mac, _KEY)
    assert lr.verify_model_mac(model, mac, _KEY) is True


def test_verify_rejects_tamper(tmp_path):
    model = tmp_path / "learned_router.pkl"
    model.write_bytes(b"fake-pickle-bytes")
    mac = lr._model_mac_path(model)
    lr.write_model_mac(model, mac, _KEY)
    model.write_bytes(b"fake-pickle-bytes-TAMPERED")
    assert lr.verify_model_mac(model, mac, _KEY) is False


def test_verify_rejects_wrong_key(tmp_path):
    model = tmp_path / "learned_router.pkl"
    model.write_bytes(b"fake-pickle-bytes")
    mac = lr._model_mac_path(model)
    lr.write_model_mac(model, mac, _KEY)
    assert lr.verify_model_mac(model, mac, _OTHER_KEY) is False


def test_verify_rejects_no_key(tmp_path):
    model = tmp_path / "learned_router.pkl"
    model.write_bytes(b"fake-pickle-bytes")
    mac = lr._model_mac_path(model)
    lr.write_model_mac(model, mac, _KEY)
    assert lr.verify_model_mac(model, mac, None) is False


def test_verify_rejects_missing_mac(tmp_path):
    model = tmp_path / "learned_router.pkl"
    model.write_bytes(b"fake-pickle-bytes")
    mac = lr._model_mac_path(model)  # never written
    assert lr.verify_model_mac(model, mac, _KEY) is False


def test_subkey_is_domain_separated():
    # The router MAC subkey must not equal the raw master key nor a plain
    # HMAC over a different info string (domain separation).
    import hashlib
    import hmac

    sub = lr._derive_model_mac_subkey(_KEY)
    assert sub is not None
    assert sub != _KEY
    sqlcipher_like = hmac.new(_KEY, b"opti-oignon-sqlcipher-v1", hashlib.sha256).digest()
    assert sub != sqlcipher_like
    assert lr._derive_model_mac_subkey(None) is None


# ---------------------------------------------------------------------------
# Source assertion: verify must run before joblib.load in _try_load_model
# ---------------------------------------------------------------------------

def test_try_load_verifies_before_loading():
    src = _PATH.read_text(encoding="utf-8")
    body = src.split("def _try_load_model", 1)[1].split("\n    def ", 1)[0]
    assert "verify_model_mac(" in body
    assert "joblib.load(" in body
    assert body.index("verify_model_mac(") < body.index("joblib.load("), (
        "_try_load_model must verify the MAC before deserializing the model"
    )


# ---------------------------------------------------------------------------
# Integration: _try_load_model end to end (requires joblib)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not lr.SKLEARN_AVAILABLE, reason="sklearn/joblib not installed")
def test_try_load_model_round_trip_and_refusals(tmp_path, monkeypatch):
    import joblib

    monkeypatch.setattr(lr, "_router_master_key", lambda: _KEY)
    model = tmp_path / "learned_router.pkl"
    joblib.dump({"hello": "world"}, str(model))
    assert lr.write_model_mac(model, lr._model_mac_path(model), _KEY) is True

    # Bypass __init__ (heavy: config/yaml + DB); _try_load_model uses only
    # self._model_path and self._pipeline plus module-level helpers.
    obj = object.__new__(lr.LearnedRouter)
    obj._model_path = model
    obj._pipeline = None

    assert lr.LearnedRouter._try_load_model(obj) is True
    assert obj._pipeline == {"hello": "world"}

    # Tampering the pickle invalidates the MAC -> refuse, pipeline stays None.
    model.write_bytes(model.read_bytes() + b"x")
    obj._pipeline = None
    assert lr.LearnedRouter._try_load_model(obj) is False
    assert obj._pipeline is None

    # Restore a valid file+MAC, then drop the key -> still refused (fail-safe).
    joblib.dump({"hello": "world"}, str(model))
    lr.write_model_mac(model, lr._model_mac_path(model), _KEY)
    monkeypatch.setattr(lr, "_router_master_key", lambda: None)
    obj._pipeline = None
    assert lr.LearnedRouter._try_load_model(obj) is False
    assert obj._pipeline is None
