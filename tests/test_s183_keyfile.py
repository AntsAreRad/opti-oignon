#!/usr/bin/env python3
"""
S183 / K-01: the keyfile must wrap the key under a passphrase-derived KEK
(envelope), not store it raw, and must enforce/verify 600 permissions.

Verified behaviors:
- save+load round-trip with a passphrase recovers the key;
- the enveloped file is not plaintext-equivalent (the raw key is not present);
- loading without (or with a wrong) passphrase fails;
- saving without a passphrase is refused unless allow_unprotected is set;
- a legacy unprotected keyfile is still readable (backward compatibility);
- the passphrase is read from OPTI_KEYFILE_PASSPHRASE when not passed;
- get_encryption_key returns None for an enveloped keyfile with no passphrase,
  and the key when the passphrase is available.
"""

import base64
import importlib.util
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_encryption():
    """Load encryption.py under its real name, stubbing the package parent."""
    name = "opti_oignon.encryption"
    fpath = _PROJECT_ROOT / "opti_oignon" / "encryption.py"
    spec = importlib.util.spec_from_file_location(name, str(fpath))
    mod = importlib.util.module_from_spec(spec)
    if "opti_oignon" not in sys.modules:
        sys.modules["opti_oignon"] = types.ModuleType("opti_oignon")
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


enc = _load_encryption()

PW = "correct horse battery staple"


def _raw(secure):
    return secure.as_bytes() if hasattr(secure, "as_bytes") else bytes(secure)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE", raising=False)
    monkeypatch.delenv("OPTI_ENCRYPTION_KEY", raising=False)


class TestEnvelope:
    def test_roundtrip_with_passphrase(self, tmp_path):
        kf = tmp_path / ".keyfile"
        key = enc.generate_key()
        enc.save_keyfile(key, path=kf, passphrase=PW)
        loaded, _salt, _kdf = enc.load_keyfile(kf, passphrase=PW)
        assert _raw(loaded) == key

    def test_file_is_not_plaintext_equivalent(self, tmp_path):
        kf = tmp_path / ".keyfile"
        key = enc.generate_key()
        enc.save_keyfile(key, path=kf, passphrase=PW)
        text = kf.read_text(encoding="ascii")
        payload = json.loads(text)
        assert payload["version"] == "envelope-v1"
        # The raw key (in any base64 form) must not appear in the file.
        assert base64.urlsafe_b64encode(key).decode("ascii") not in text
        assert base64.b64encode(key).decode("ascii") not in text

    def test_load_without_passphrase_raises(self, tmp_path):
        kf = tmp_path / ".keyfile"
        enc.save_keyfile(enc.generate_key(), path=kf, passphrase=PW)
        with pytest.raises(ValueError):
            enc.load_keyfile(kf)

    def test_wrong_passphrase_fails(self, tmp_path):
        kf = tmp_path / ".keyfile"
        enc.save_keyfile(enc.generate_key(), path=kf, passphrase=PW)
        with pytest.raises(Exception):
            enc.load_keyfile(kf, passphrase="wrong passphrase")

    def test_perms_are_600(self, tmp_path):
        kf = tmp_path / ".keyfile"
        enc.save_keyfile(enc.generate_key(), path=kf, passphrase=PW)
        mode = stat.S_IMODE(os.stat(kf).st_mode)
        assert mode == 0o600


class TestNoPassphraseRefused:
    def test_save_without_passphrase_raises(self, tmp_path):
        kf = tmp_path / ".keyfile"
        with pytest.raises(ValueError):
            enc.save_keyfile(enc.generate_key(), path=kf)

    def test_allow_unprotected_writes_legacy_and_reads_back(self, tmp_path):
        kf = tmp_path / ".keyfile"
        key = enc.generate_key()
        enc.save_keyfile(key, kdf_name="random", path=kf, allow_unprotected=True)
        # Legacy raw format: base64 key on line 1.
        first_line = kf.read_text(encoding="ascii").splitlines()[0]
        assert base64.urlsafe_b64decode(first_line) == key
        loaded, _salt, _kdf = enc.load_keyfile(kf)
        assert _raw(loaded) == key
        assert stat.S_IMODE(os.stat(kf).st_mode) == 0o600


class TestEnvPassphrase:
    def test_env_passphrase_used_for_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", PW)
        kf = tmp_path / ".keyfile"
        key = enc.generate_key()
        enc.save_keyfile(key, path=kf)  # no explicit passphrase -> uses env
        assert json.loads(kf.read_text())["version"] == "envelope-v1"
        loaded, _s, _k = enc.load_keyfile(kf)  # uses env
        assert _raw(loaded) == key


class TestGetEncryptionKey:
    def test_envelope_without_passphrase_returns_none(self, tmp_path, monkeypatch):
        kf = tmp_path / ".keyfile"
        enc.save_keyfile(enc.generate_key(), path=kf, passphrase=PW)
        monkeypatch.setattr(enc, "_DEFAULT_KEYFILE", kf)
        # No env passphrase: encryption stays off rather than crashing.
        assert enc.get_encryption_key() is None

    def test_envelope_with_passphrase_returns_key(self, tmp_path, monkeypatch):
        kf = tmp_path / ".keyfile"
        key = enc.generate_key()
        enc.save_keyfile(key, path=kf, passphrase=PW)
        monkeypatch.setattr(enc, "_DEFAULT_KEYFILE", kf)
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", PW)
        got = enc.get_encryption_key()
        assert got is not None
        assert _raw(got) == key
