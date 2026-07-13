#!/usr/bin/env python3
"""Encrypted backup format contracts: authenticated, honest, fail-secure.

The encrypted backup container is the one artifact meant to leave the
machine, so its format discipline is load-bearing: a document encrypts to a
magic-tagged container carrying its key-derivation identifier and salt, and
decrypts back byte-for-byte; a wrong password or a single flipped bit is an
authenticated-encryption refusal, never garbage output; a password below
the floor is refused before any derivation; a container without the magic
or below the minimum size is refused by the honest early check that names
the actual problem; the key-derivation dispatch honors the stored
identifier so a PBKDF2-derived container decrypts through PBKDF2; and a
container whose decrypted payload is not backup JSON is a named refusal.
This suite pins that behavior:

  * BE1 -- round trip: magic tag, known KDF identifier, exact document
    back, and the format detector answers honestly both ways;
  * BE2 -- a wrong password and a flipped ciphertext bit both refuse with
    the decryption failure, never silent output;
  * BE3 -- a password below eight characters is refused before anything
    derives;
  * BE4 -- a container without the magic tag is refused by the check that
    names the magic, not by a late decryption error;
  * BE5 -- a truncated container is refused by the early size check that
    names the truncation, not by a late decryption error;
  * BE6 -- the stored KDF identifier drives the derivation: a container
    built with the PBKDF2 fallback decrypts through the PBKDF2 path;
  * BE7 -- a decrypted payload that is not valid JSON is a named refusal.

Loads the backup manager module in isolation under a stand-in package with
the REAL encryption module seeded beside it (the round trip must exercise
the true key derivation and authenticated cipher); every ``opti_oignon.*``
entry plus the model-client entry is snapshotted and evicted first. A
meta-path guard refuses any project submodule that was not seeded, so the
load behaves identically whether or not the project is installed.
Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_PASSWORD = "correct-horse-battery"


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


def _load():
    """Load the backup manager with the real encryption module beside it."""
    keys = ["ollama"] + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # no client import exists; drift fails loud

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    sys.modules["opti_oignon"] = root

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
        sys.modules.pop("ollama", None)
        for k, v in saved.items():
            sys.modules[k] = v

    try:
        enc_full = "opti_oignon.encryption"
        enc_spec = importlib.util.spec_from_file_location(
            enc_full, _OO / "encryption.py"
        )
        enc = importlib.util.module_from_spec(enc_spec)
        sys.modules[enc_full] = enc
        root.encryption = enc
        enc_spec.loader.exec_module(enc)

        full = "opti_oignon.backup_manager"
        spec = importlib.util.spec_from_file_location(
            full, _OO / "backup_manager.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        root.backup_manager = mod
        spec.loader.exec_module(mod)
    except BaseException:
        restore()
        raise

    return SimpleNamespace(mod=mod, enc=enc, restore=restore)


def _document():
    return {
        "schema_version": "1.0",
        "metadata": {"note": "isolated"},
        "sections": {"theme": {"accent": "teal", "depth": 3}},
    }


def test_round_trip_magic_kdf_and_exact_document():
    """BE1 -- container round-trips exactly and detects honestly."""
    ctx = _load()
    try:
        blob = ctx.mod.encrypt_backup(_document(), _PASSWORD)
        assert blob[:6] == b"OOENC1", blob[:8]
        kdf_id = blob[6:8]
        assert kdf_id in (ctx.enc._KDF_ARGON2ID, ctx.enc._KDF_PBKDF2), kdf_id
        assert ctx.mod.is_encrypted_backup(blob) is True
        assert ctx.mod.is_encrypted_backup(b'{"schema_version": "1.0"}') is False
        restored = ctx.mod.decrypt_backup(blob, _PASSWORD)
        assert restored == _document(), (
            "the decrypted document must be exactly the encrypted one"
        )
    finally:
        ctx.restore()


def test_wrong_password_and_flipped_bit_both_refuse():
    """BE2 -- wrong password or tampered byte is a refusal, never output."""
    ctx = _load()
    try:
        blob = ctx.mod.encrypt_backup(_document(), _PASSWORD)

        refused = None
        try:
            ctx.mod.decrypt_backup(blob, "not-the-password")
        except ValueError as exc:
            refused = exc
        assert refused is not None and "Decryption failed" in str(refused), (
            refused
        )

        first_ct = 6 + 2 + 16 + 1 + 12  # magic, kdf, salt, version, nonce
        tampered = (
            blob[:first_ct]
            + bytes([blob[first_ct] ^ 0x01])
            + blob[first_ct + 1 :]
        )
        refused = None
        try:
            ctx.mod.decrypt_backup(tampered, _PASSWORD)
        except ValueError as exc:
            refused = exc
        assert refused is not None and "Decryption failed" in str(refused), (
            "a single flipped ciphertext bit must refuse, never emit output"
        )
    finally:
        ctx.restore()


def test_password_below_floor_refused_before_derivation():
    """BE3 -- a short password never reaches the key derivation."""
    ctx = _load()
    try:
        refused = None
        try:
            ctx.mod.encrypt_backup(_document(), "short")
        except ValueError as exc:
            refused = exc
        assert refused is not None and "at least 8" in str(refused), refused
    finally:
        ctx.restore()


def test_missing_magic_refused_by_the_named_check():
    """BE4 -- no magic tag: the refusal names the magic, and comes early."""
    ctx = _load()
    try:
        bogus = b"BADMAGIC" + b"\x00" * 52
        refused = None
        try:
            ctx.mod.decrypt_backup(bogus, _PASSWORD)
        except ValueError as exc:
            refused = exc
        assert refused is not None and "invalid magic bytes" in str(refused), (
            "the magic check must refuse by name before any decryption runs, "
            f"got {refused!r}"
        )
    finally:
        ctx.restore()


def test_truncated_container_refused_by_the_size_check():
    """BE5 -- truncation is refused early by the honest size check."""
    ctx = _load()
    try:
        blob = ctx.mod.encrypt_backup(_document(), _PASSWORD)
        truncated = blob[:30]
        refused = None
        try:
            ctx.mod.decrypt_backup(truncated, _PASSWORD)
        except ValueError as exc:
            refused = exc
        assert refused is not None and str(refused).startswith(
            "Encrypted backup data is too short"
        ), (
            "the size check must refuse truncation by name before any "
            f"decryption runs, got {refused!r}"
        )
    finally:
        ctx.restore()


def test_kdf_identifier_drives_the_derivation_path():
    """BE6 -- a PBKDF2-tagged container decrypts through PBKDF2."""
    ctx = _load()
    try:
        key, salt, kdf_name = ctx.enc.derive_key_from_passphrase(
            _PASSWORD, force_pbkdf2=True
        )
        assert kdf_name == "pbkdf2"
        payload = json.dumps(_document()).encode("utf-8")
        blob = (
            b"OOENC1"
            + ctx.enc._KDF_PBKDF2
            + salt
            + ctx.enc.encrypt_bytes(key, payload)
        )
        restored = ctx.mod.decrypt_backup(blob, _PASSWORD)
        assert restored == _document(), (
            "the stored KDF identifier must route derivation to PBKDF2; a "
            "mismatch would derive the wrong key and refuse"
        )
    finally:
        ctx.restore()


def test_non_json_payload_is_a_named_refusal():
    """BE7 -- a decrypted payload that is not JSON refuses by name."""
    ctx = _load()
    try:
        key, salt, kdf_name = ctx.enc.derive_key_from_passphrase(_PASSWORD)
        kdf_id = (
            ctx.enc._KDF_ARGON2ID
            if kdf_name == "argon2id"
            else ctx.enc._KDF_PBKDF2
        )
        blob = (
            b"OOENC1"
            + kdf_id
            + salt
            + ctx.enc.encrypt_bytes(key, b"this is not backup json {")
        )
        refused = None
        try:
            ctx.mod.decrypt_backup(blob, _PASSWORD)
        except ValueError as exc:
            refused = exc
        assert refused is not None and "not valid backup JSON" in str(refused), (
            refused
        )
    finally:
        ctx.restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
