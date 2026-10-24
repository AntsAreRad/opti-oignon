#!/usr/bin/env python3
"""Contracts for the key the platform signs with: the key file's key when it loads, and never its bytes.

The plugin allowlist signs its entries and the model provenance seal its
manifest with an HMAC under the key file's key. Both used to fall back, when
the key file did not load, on its first 32 bytes -- for a wrapped key file,
the same text in every one of them, a key anyone holds.

  * KY1 -- THE ALLOWLIST'S KEY IS THE MODE'S: the key file's key when it
    loads, wrapped or not; none without a key file; a key file that is there
    and does not load is refused by the mode's ``SigningKeyUnavailable``,
    naming it and the passphrase variable -- its bytes are never a key --,
    and so is an empty or cut one, which loads a key of the wrong size.
  * KY2 -- THE VERIFIER'S FOUR POINTS UNDER THE KEY THAT LOADS: a plugin
    absent from the allowlist is refused; a signed entry with a matching
    hash and unchanged permissions verifies; a modified plugin tree is
    refused on hash; a tampered entry is refused on signature; a permission
    escalation is refused while equal or narrower requests pass; with the key
    file gone the hash refusal still holds; revocation removes the entry at
    once, from memory and from the file. (C2 of the allowlist's own suite,
    under the real key.)
  * KY3 -- THE BATCH CEREMONY UNDER THE KEY THAT LOADS: prepare computes the
    composite hash; approval refuses an unknown batch, refuses without a key
    file, refuses a plugin changed since preparation -- each persisting
    nothing --, and a clean approval persists signed entries a fresh manager
    verifies. (C3 of the allowlist's own suite, under the real key.)
  * KY4 -- AN ENTRY VERIFIES ONLY UNDER THE KEY THAT LOADS: one signed under
    the bytes every wrapped key file starts with is refused; with a key file
    that does not load, with none, or with a security mode that does not
    import -- it reads the key --, a listed plugin whose files match is
    refused, saying why.
  * KY5 -- THE CEREMONY SIGNS ONLY UNDER THE KEY THAT LOADS: with a key file
    that does not load, or none, approval refuses ``no_signing_key``, saying
    why; nothing is written and the batch stays pending; once the key loads
    the same batch is approved, its entries signed under that key.
  * KY6 -- THE PROVENANCE SEAL UNDER THE KEY THAT LOADS: the MAC's key is the
    key file's; with none, or one that does not load -- said, by name --, no
    key: a model cannot be recorded, a manifest sealed under the bytes every
    wrapped key file starts with is ``key_unavailable``, refused where
    provenance is enforced; once the key loads that manifest's seal is
    invalid and one sealed under the key verifies.
  * KY7 -- WHO NAMES THE KEY FILE: outside docstrings, the key file's name is
    written only by the loader (``encryption``), the security mode (its path,
    to see whether one is there) and ``auth_2fa``, whose scheme from before
    AU-03 still takes a key from the file's text -- for recovery codes and
    app passwords hashed while no master key loads, and to check older ones:
    a debt, named. A census of names, not of reads: it sees a module that
    writes the file's name, the way a key file's bytes became a key twice;
    a reader through the loader's own path constant is not seen.
  * KY8 -- A KEY FILE THAT HOLDS NO 32-BYTE KEY IS NO KEY: an empty key file,
    one cut short and one holding more load a key of the wrong size; the
    mode then reads Bulbe under a lockfile signed with it, saying why,
    the allowlist verifies no entry signed with it and approves nothing,
    and no provenance MAC is made or checked under it, which is said.
  * KY9 -- AN ENTRY'S SIGNATURE IS THE ALLOWLIST'S OWN: it is made under a
    key derived from the master key for the allowlist alone, over all the
    entry's fields as one canonical record; an approved plugin id laid out
    as a lockfile -- the forgery the scheme before let through -- reads
    Bulbe; an approver, a time or a batch rewritten in the file is
    tampering, and so is an entry that does not hold what the allowlist
    writes; an entry signed before -- its hash of the scheme before too --
    is refused as outdated, saying to approve it again, never compared
    with today's hash, and one with no scheme that the scheme before does
    not verify is tampering.
  * KY10 -- A PLUGIN RUNS FROM ITS SOURCE: bytecode planted beside its
    entry point in ``__pycache__``, stamped with the source's time and
    size, never runs in its place; the worker sets every import to read no
    bytecode cached beside a source and to write none, before anything else
    it does.
  * KY11 -- A PLUGIN'S HASH COVERS EVERYTHING ITS FOLDER HOLDS: every file,
    whatever its suffix -- a note, data, bytecode with or without a source,
    native code -- and every folder enter it, a content or a name changed
    changes it; a folder that is missing or holds no file, or holds a
    symbolic link, a pipe or something unreadable, cannot be hashed. The
    batch hash is order-independent and element-sensitive. (C1 of the
    allowlist's own suite, in place of its four suffixes.)
  * KY12 -- A PLUGIN WHOSE FOLDER CHANGED, OR CANNOT BE HASHED, IS NEITHER
    APPROVED NOR VERIFIED: bytecode planted in a package folder beside an
    approved module is refused on hash; a folder that cannot be hashed is
    refused by name at approval, which writes nothing, and at load, where
    an entry naming an empty hash used to verify.

The modules are loaded from source in an isolation window, every path they
reach -- the key file, the lockfile, security.yaml, the allowlist, the
manifest -- moved into a temporary folder.
"""

import ast
import base64
import hashlib
import hmac
import importlib.util
import json
import logging
import os
import py_compile
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_census import modules as package_modules  # noqa: E402
from _isolation import isolate, source  # noqa: E402

K1 = bytes(range(1, 33))
PHRASE = "a phrase only these contracts know"
ALLOW = "opti_oignon.plugin_allowlist"
PROV = "opti_oignon.model_provenance"
_KEYFILE_NAME = re.compile(r"(?:^|/)\.keyfile(?:$|\W)")


def _world(tmp_path, monkeypatch, *, mode_imports=True):
    """encryption, the security mode and the allowlist in a window, every path they reach under ``tmp_path``.

    ``mode_imports=False`` declares the security mode unreachable: the
    allowlist then has nothing to read its key with.
    """
    pytest.importorskip("cryptography")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
        monkeypatch.delenv(name, raising=False)
    names = (("process_secrets", "encryption", "security_mode", "plugin_allowlist") if mode_imports
             else ("process_secrets", "encryption", "plugin_allowlist"))
    blocked = ("opti_oignon.secure_bytes",) + (() if mode_imports else ("opti_oignon.security_mode",))
    loaded, restore = isolate(targets={f"opti_oignon.{name}": source(f"{name}.py") for name in names},
                              blocked=blocked)
    try:
        enc, pa = loaded["opti_oignon.encryption"], loaded[ALLOW]
        sm = loaded.get("opti_oignon.security_mode")
        data = tmp_path / "data"
        data.mkdir(parents=True)
        keyfile = data / ".keyfile"
        enc._DEFAULT_KEYFILE = keyfile
        if sm is not None:
            sm._DEFAULT_KEYFILE = keyfile
            sm._DATA_DIR, sm._LOCKFILE_PATH, sm._SECURITY_YAML = (data, data / ".security_mode_lock",
                                                                  tmp_path / "security.yaml")
            sm._audit_log = lambda event, severity="INFO", **details: None
        pa._DATA_DIR, pa._ALLOWLIST_PATH = data, data / "plugin_allowlist.json"
        world = SimpleNamespace(enc=enc, sm=sm, pa=pa, data=data, keyfile=keyfile, allowlist=pa._ALLOWLIST_PATH,
                                lock=getattr(sm, "_LOCKFILE_PATH", None), yaml=getattr(sm, "_SECURITY_YAML", None),
                                restore=restore)
    except BaseException:
        restore()
        raise
    return world


def _key_file(world, *, wrapped):
    """The key file holding K1, made as setup makes one: wrapped under the phrase, or in the unprotected format.

    Returns its bytes; for a wrapped one, its first 32 are the text every wrapped key file starts with.
    """
    if wrapped:
        world.enc.save_keyfile(K1, passphrase=PHRASE)
    else:
        world.enc.save_keyfile(K1, allow_unprotected=True)
    return world.keyfile.read_bytes()


def _hash_before(folder):
    """A plugin's hash as the allowlist made it before: the paths and bytes of its .py, .yaml, .yml and .json
    files, one after the other."""
    root = Path(folder).resolve()
    hasher = hashlib.sha512()
    for file in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json")):
        hasher.update(str(file.relative_to(root)).encode("utf-8"))
        hasher.update(file.read_bytes())
    return f"sha512:{hasher.hexdigest()}"


def _entry_before(pa, plugin_id, code_hash, **fields):
    """An entry as the scheme before wrote it: no scheme, the master key K1 itself over plugin_id, code_hash,
    the sorted permissions and batch_id joined by "||"."""
    data = dict({"plugin_id": plugin_id, "code_hash": code_hash, "approved_by": "tester", "approved_at": 1000.0,
                 "batch_id": "batch-reference", "permissions": []}, **fields)
    record = "||".join((data["plugin_id"], data["code_hash"], ",".join(sorted(data["permissions"])),
                        data["batch_id"]))
    data["signature"] = hmac.new(K1, record.encode(), hashlib.sha512).hexdigest()
    return pa.AllowlistEntry.from_dict(data)


def _cut_key_file():
    """An unprotected key file cut to half its key: it loads 16 bytes."""
    return base64.urlsafe_b64encode(K1[:16]) + b"\n\n\n"


def _long_key_file():
    """An unprotected key file holding more than a key: it loads 48 bytes."""
    return base64.urlsafe_b64encode(bytes(range(48))) + b"\n\n\n"


def _put_back(world, held):
    """The key file as it was, owner-only, after a step that removed it."""
    world.keyfile.write_bytes(held)
    os.chmod(world.keyfile, 0o600)


def _mk_plugin(root: Path, name: str, code: str = "VALUE = 1\n") -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(code, encoding="utf-8")
    (plugin_dir / "manifest.yaml").write_text(f"id: {name}\n", encoding="utf-8")
    return plugin_dir


def _signed_entry(pa, plugin_id, plugin_dir, permissions, key):
    """An allowlist entry for the plugin as it is, signed under ``key`` (raw bytes)."""
    entry = pa.AllowlistEntry(
        plugin_id=plugin_id,
        code_hash=pa.compute_plugin_hash(plugin_dir),
        approved_by="tester",
        approved_at=1000.0,
        batch_id="batch-reference",
        permissions=list(permissions),
    )
    entry.signature = pa._sign_entry(entry, key)
    return entry


def _raw(key):
    return key.as_bytes() if hasattr(key, "as_bytes") else key


def _says_why_it_does_not_load(world, text):
    assert str(world.keyfile) in text and "does not load here" in text and "OPTI_KEYFILE_PASSPHRASE" in text, text


# ---------------------------------------------------------------------------
# KY1 -- the allowlist's key is the mode's
# ---------------------------------------------------------------------------
def test_ky1_the_allowlist_s_key_is_the_mode_s_the_key_file_s_when_it_loads_and_never_its_bytes(tmp_path,
                                                                                                monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        pa, sm = world.pa, world.sm
        assert pa._load_signing_key() is None, "no key file: no key"
        assert pa._key_or_why() == (None, "there is no key file")

        _key_file(world, wrapped=False)
        assert _raw(pa._load_signing_key()) == K1, "an unprotected key file: its key"
        world.keyfile.unlink()

        opening = _key_file(world, wrapped=True)[:32]
        assert opening != K1
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", PHRASE)
        assert _raw(pa._load_signing_key()) == K1 == _raw(sm._load_signing_key()), "wrapped, with its phrase"

        monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE")
        with pytest.raises(sm.SigningKeyUnavailable) as refused:
            pa._load_signing_key()
        _says_why_it_does_not_load(world, str(refused.value))
        key, why = pa._key_or_why()
        assert key is None, "a key file that does not load gives no key -- never its first 32 bytes"
        _says_why_it_does_not_load(world, why)

        # An empty key file, and one cut short, load as a key of the wrong size: no key either.
        for held, size in ((b"", 0), (_cut_key_file(), 16)):
            world.keyfile.unlink()
            _put_back(world, held)
            with pytest.raises(sm.SigningKeyUnavailable) as refused:
                pa._load_signing_key()
            assert str(refused.value) == (f"the key file {world.keyfile} does not load here as a key: it holds "
                                          f"{size} bytes, not 32"), refused.value
            assert pa._key_or_why() == (None, str(refused.value))
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# KY2 -- the verifier's four points under the key that loads (C2, superseded)
# ---------------------------------------------------------------------------
def test_ky2_the_verifier_s_four_points_hold_under_the_key_that_loads(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        pa = world.pa
        held = _key_file(world, wrapped=False)
        plugin = _mk_plugin(tmp_path, "sample")
        good_code = (plugin / "plugin.py").read_bytes()

        # Absent from the allowlist: refused.
        verdict = pa.PluginAllowlistManager().verify_plugin("sample", plugin)
        assert verdict["allowed"] is False
        assert "not in the allowlist" in verdict["reason"]

        # Signed entry, matching hash, same permissions: verified.
        entry = _signed_entry(pa, "sample", plugin, ["read"], K1)
        pa._save_allowlist([entry])
        verdict = pa.PluginAllowlistManager().verify_plugin("sample", plugin, permissions=["read"])
        assert verdict == {"allowed": True, "reason": "Verified"}

        # Modified plugin tree: refused on hash.
        (plugin / "plugin.py").write_text("VALUE = 99\n", encoding="utf-8")
        verdict = pa.PluginAllowlistManager().verify_plugin("sample", plugin, permissions=["read"])
        assert verdict["allowed"] is False
        assert "changed since approval" in verdict["reason"]
        (plugin / "plugin.py").write_bytes(good_code)

        # Tampered entry (permissions widened on disk, not signed again): refused on signature.
        stored = json.loads(world.allowlist.read_text())
        stored[0]["permissions"].append("network")
        world.allowlist.write_text(json.dumps(stored))
        verdict = pa.PluginAllowlistManager().verify_plugin("sample", plugin, permissions=["read"])
        assert verdict["allowed"] is False
        assert "tampered" in verdict["reason"]
        pa._save_allowlist([entry])

        # Permission escalation: refused; equal or narrower passes.
        fresh = pa.PluginAllowlistManager()
        verdict = fresh.verify_plugin("sample", plugin, permissions=["read", "network"])
        assert verdict["allowed"] is False
        assert "Permission escalation" in verdict["reason"]
        assert fresh.verify_plugin("sample", plugin, permissions=[])["allowed"] is True
        assert fresh.verify_plugin("sample", plugin, permissions=None)["allowed"] is True

        # The key file gone: the hash refusal still holds.
        world.keyfile.unlink()
        (plugin / "plugin.py").write_text("VALUE = 99\n", encoding="utf-8")
        verdict = pa.PluginAllowlistManager().verify_plugin("sample", plugin, permissions=["read"])
        assert verdict["allowed"] is False
        assert "changed since approval" in verdict["reason"]
        (plugin / "plugin.py").write_bytes(good_code)
        _put_back(world, held)

        # Revocation removes the entry at once, from memory and from the file.
        manager = pa.PluginAllowlistManager()
        assert manager.is_allowed("sample") is True
        assert manager.revoke_plugin("sample") is True
        assert manager.is_allowed("sample") is False
        assert manager.revoke_plugin("sample") is False
        verdict = pa.PluginAllowlistManager().verify_plugin("sample", plugin, permissions=["read"])
        assert verdict["allowed"] is False
        assert "not in the allowlist" in verdict["reason"]
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# KY3 -- the batch ceremony under the key that loads (C3, superseded)
# ---------------------------------------------------------------------------
def test_ky3_the_batch_ceremony_holds_under_the_key_that_loads(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        pa = world.pa
        held = _key_file(world, wrapped=False)
        first = _mk_plugin(tmp_path, "first")
        second = _mk_plugin(tmp_path, "second", code="VALUE = 2\n")
        first_code = (first / "plugin.py").read_bytes()

        manager = pa.PluginAllowlistManager()
        manifest = manager.prepare_batch([
            {"plugin_id": "first", "plugin_dir": str(first), "permissions": ["read"]},
            {"plugin_id": "second", "plugin_dir": str(second), "permissions": []},
        ])
        hashes = [p["code_hash"] for p in manifest.plugins]
        assert manifest.batch_hash == pa.compute_batch_hash(hashes)
        assert manager.get_pending_batch() is manifest

        # Unknown batch id: refused, nothing persisted.
        result = manager.approve_batch("not-a-batch", "tester")
        assert result["success"] is False
        assert result["error"] == "no_matching_batch"
        assert not world.allowlist.exists()

        # No key file: refused (fail-closed), nothing persisted.
        world.keyfile.unlink()
        result = manager.approve_batch(manifest.batch_id, "tester")
        assert result["success"] is False
        assert result["error"] == "no_signing_key"
        assert not world.allowlist.exists()
        _put_back(world, held)

        # A plugin changed between prepare and approve: refused.
        (first / "plugin.py").write_text("VALUE = 99\n", encoding="utf-8")
        result = manager.approve_batch(manifest.batch_id, "tester")
        assert result["success"] is False
        assert result["error"] == "hash_changed"
        assert not world.allowlist.exists()
        (first / "plugin.py").write_bytes(first_code)

        # Clean approval: signed entries persisted, pending cleared, a fresh manager verifies both.
        result = manager.approve_batch(manifest.batch_id, "tester")
        assert result == {"success": True, "entries_added": 2, "batch_id": manifest.batch_id}
        assert manager.get_pending_batch() is None
        assert world.allowlist.exists()

        fresh = pa.PluginAllowlistManager()
        assert fresh.verify_plugin("first", first, permissions=["read"]) == {"allowed": True, "reason": "Verified"}
        assert fresh.verify_plugin("second", second, permissions=[]) == {"allowed": True, "reason": "Verified"}
        for stored in fresh.list_entries():
            assert stored.batch_id == manifest.batch_id
            assert stored.approved_by == "tester"
            assert stored.signature
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# KY4 -- an entry verifies only under the key that loads
# ---------------------------------------------------------------------------
def test_ky4_an_entry_verifies_only_under_the_key_that_loads_or_is_refused_saying_why(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        pa = world.pa
        opening = _key_file(world, wrapped=True)[:32]
        plugin, other = _mk_plugin(tmp_path, "sample"), _mk_plugin(tmp_path, "other", code="VALUE = 2\n")
        pa._save_allowlist([_signed_entry(pa, "sample", plugin, ["read"], K1),
                            _signed_entry(pa, "other", other, ["read"], opening)])

        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", PHRASE)
        manager = pa.PluginAllowlistManager()
        assert manager.verify_plugin("sample", plugin, permissions=["read"]) == {"allowed": True,
                                                                                  "reason": "Verified"}
        verdict = manager.verify_plugin("other", other, permissions=["read"])
        assert verdict["allowed"] is False and "tampered" in verdict["reason"], verdict

        # The key file does not load: neither entry verifies, and the refusal says why.
        monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE")
        for plugin_id, folder in (("sample", plugin), ("other", other)):
            verdict = manager.verify_plugin(plugin_id, folder, permissions=["read"])
            assert verdict["allowed"] is False, (plugin_id, verdict)
            assert verdict["reason"].startswith(f"Unverifiable signature for '{plugin_id}': "), verdict
            _says_why_it_does_not_load(world, verdict["reason"])

        # No key file: refused, saying so.
        world.keyfile.unlink()
        verdict = manager.verify_plugin("sample", plugin, permissions=["read"])
        assert verdict == {"allowed": False, "reason": "Unverifiable signature for 'sample': there is no key file"}
    finally:
        world.restore()

    # A security mode that does not import leaves the allowlist nothing to read its key with: refused.
    blind = _world(tmp_path / "blind", monkeypatch, mode_imports=False)
    try:
        pa = blind.pa
        opening = _key_file(blind, wrapped=True)[:32]
        plugin, other = _mk_plugin(tmp_path / "blind", "sample"), _mk_plugin(tmp_path / "blind", "other")
        pa._save_allowlist([_signed_entry(pa, "sample", plugin, [], K1),
                            _signed_entry(pa, "other", other, [], opening)])
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", PHRASE)
        for plugin_id, folder in (("sample", plugin), ("other", other)):
            verdict = pa.PluginAllowlistManager().verify_plugin(plugin_id, folder, permissions=[])
            assert verdict["allowed"] is False, (plugin_id, verdict)
            assert verdict["reason"].startswith(
                f"Unverifiable signature for '{plugin_id}': the security mode, which reads the key, does not import"
            ), verdict
    finally:
        blind.restore()


# ---------------------------------------------------------------------------
# KY5 -- the ceremony signs only under the key that loads
# ---------------------------------------------------------------------------
def test_ky5_the_ceremony_signs_only_under_the_key_that_loads_or_refuses_saying_why(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        pa = world.pa
        opening = _key_file(world, wrapped=True)[:32]
        plugin = _mk_plugin(tmp_path, "sample")
        before = [_signed_entry(pa, "kept", _mk_plugin(tmp_path, "kept"), [], K1)]
        pa._save_allowlist(before)
        on_disk = world.allowlist.read_bytes()

        manager = pa.PluginAllowlistManager()
        manifest = manager.prepare_batch([{"plugin_id": "sample", "plugin_dir": str(plugin), "permissions": ["read"]}])

        # The key file does not load: refused, saying why; nothing written, the batch still pending.
        result = manager.approve_batch(manifest.batch_id, "tester")
        assert result["success"] is False and result["error"] == "no_signing_key", result
        assert result["message"].startswith("Cannot approve without the signing key: "), result
        _says_why_it_does_not_load(world, result["message"])
        assert world.allowlist.read_bytes() == on_disk
        assert manager.get_pending_batch() is manifest

        # No key file: refused, saying so.
        held = world.keyfile.read_bytes()
        world.keyfile.unlink()
        result = manager.approve_batch(manifest.batch_id, "tester")
        assert result == {"success": False, "error": "no_signing_key",
                          "message": "Cannot approve without the signing key: there is no key file"}
        assert world.allowlist.read_bytes() == on_disk and manager.get_pending_batch() is manifest
        _put_back(world, held)

        # The key loads: the same batch is approved, its entry signed under the key, not the opening bytes.
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", PHRASE)
        result = manager.approve_batch(manifest.batch_id, "tester")
        assert result == {"success": True, "entries_added": 1, "batch_id": manifest.batch_id}
        stored = {entry.plugin_id: entry for entry in pa.PluginAllowlistManager().list_entries()}
        assert set(stored) == {"kept", "sample"}
        assert stored["sample"].signature == pa._sign_entry(stored["sample"], K1)
        assert stored["sample"].signature != pa._sign_entry(stored["sample"], opening)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# KY6 -- the provenance seal under the key that loads
# ---------------------------------------------------------------------------
def _provenance(tmp_path, monkeypatch):
    """encryption and model provenance in a window; no post-quantum seal asked for or available."""
    pytest.importorskip("cryptography")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE"):
        monkeypatch.delenv(name, raising=False)
    pqc = ModuleType("opti_oignon.pqc_signatures")
    pqc.PQC_AVAILABLE = False
    pqc.is_pqc_enabled = lambda: False
    pqc.pqc_required = lambda: False

    def no_keypair():
        raise RuntimeError("no post-quantum keypair in these contracts")

    pqc.load_pqc_keypair = no_keypair
    loaded, restore = isolate(targets={"opti_oignon.process_secrets": source("process_secrets.py"), "opti_oignon.encryption": source("encryption.py"),
                                       PROV: source("model_provenance.py")},
                              blocked=("opti_oignon.secure_bytes", "opti_oignon.security_mode"),
                              seeded={"opti_oignon.pqc_signatures": pqc})
    try:
        enc, prov = loaded["opti_oignon.encryption"], loaded[PROV]
        data = tmp_path / "data"
        data.mkdir(parents=True)
        enc._DEFAULT_KEYFILE = data / ".keyfile"
        prov._DATA_DIR, prov.DEFAULT_MANIFEST_PATH = data, data / "model_provenance.json"
        world = SimpleNamespace(enc=enc, prov=prov, data=data, keyfile=enc._DEFAULT_KEYFILE,
                                manifest=prov.DEFAULT_MANIFEST_PATH, restore=restore)
    except BaseException:
        restore()
        raise
    return world


def _sealed(prov, model, key):
    """A manifest pinning ``model`` as it is, sealed with an HMAC under ``key``."""
    payload = {"version": prov.MANIFEST_VERSION,
               "entries": {model.name: {"sha256": prov.compute_digest(model), "size": model.stat().st_size,
                                        "recorded_at": 1.0}}}
    keys = prov.SealKeys(scheme=prov.SCHEME_HMAC, sign_key=key, verify_key=key)
    return dict(payload, seal=prov.compute_seal(payload, keys))


def test_ky6_the_provenance_seal_is_made_and_checked_only_under_the_key_that_loads(tmp_path, monkeypatch, caplog):
    world = _provenance(tmp_path, monkeypatch)
    try:
        prov = world.prov
        model = tmp_path / "pinned-model-Q4_K_M.gguf"
        model.write_bytes(b"GGUF" + bytes(range(256)) * 4)

        def unsealed_and_refused(manifest):
            assert prov.resolve_seal_keys() is None
            reason, _ = prov.classify_model(model, manifest)
            assert reason == prov.REASON_KEY_UNAVAILABLE, reason
            assert prov.decide(reason, prov.ENFORCE) is False
            with pytest.raises(prov.ProvenanceError):
                prov.record_model(model, manifest_path=world.manifest)
            assert not world.manifest.exists(), "a model that cannot be sealed writes no manifest"

        # No key file: no key.
        assert prov._hmac_key() is None
        unsealed_and_refused(_sealed(prov, model, K1))

        # A wrapped key file that does not load: no key, said by name; a manifest sealed under its opening bytes
        # is not verified.
        world.enc.save_keyfile(K1, passphrase=PHRASE)
        opening = world.keyfile.read_bytes()[:32]
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger=PROV):
            assert prov._hmac_key() is None
        assert any("does not load here" in r.getMessage() and "OPTI_KEYFILE_PASSPHRASE" in r.getMessage()
                   for r in caplog.records), [r.getMessage() for r in caplog.records]
        unsealed_and_refused(_sealed(prov, model, opening))

        # The key loads: the seal under the opening bytes is invalid, one under the key verifies, and a model
        # recorded is sealed under the key.
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", PHRASE)
        keys = prov.resolve_seal_keys()
        assert keys is not None and keys.scheme == prov.SCHEME_HMAC and bytes(keys.sign_key) == K1
        assert prov.classify_model(model, _sealed(prov, model, opening))[0] == prov.REASON_SEAL_INVALID
        assert prov.classify_model(model, _sealed(prov, model, K1))[0] == prov.REASON_VERIFIED
        prov.record_model(model, manifest_path=world.manifest)
        written = json.loads(world.manifest.read_text(encoding="utf-8"))
        assert prov.classify_model(model, written)[0] == prov.REASON_VERIFIED
        assert written["seal"] == prov.compute_seal(prov._payload_of(written),
                                                    prov.SealKeys(scheme=prov.SCHEME_HMAC, sign_key=K1,
                                                                  verify_key=K1))
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# KY7 -- who names the key file
# ---------------------------------------------------------------------------
# Each module that may write the key file's name, and why.
_NAMERS = {
    "opti_oignon.encryption": "the loader: the key file is read there for its key",
    "opti_oignon.security_mode": "its path, to see whether a key file is there; the key comes from the loader",
    "opti_oignon.auth_2fa": "its scheme from before AU-03 takes a key from the file's text, for recovery codes and "
                            "app passwords hashed while no master key loads and to check older ones -- a debt, "
                            "named",
}


def _names_the_key_file(text):
    """The lines of ``text`` where a string that is no docstring names the key file."""
    tree = ast.parse(text)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return sorted(node.lineno for node in ast.walk(tree)
                  if isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and id(node) not in docstrings and _KEYFILE_NAME.search(node.value))


def test_ky7_only_the_loader_the_mode_and_the_old_recovery_codes_name_the_key_file():
    # The witness: a planted module that reads the key file is seen, its docstring and a look-alike are not.
    planted = ('"""Reads data/.keyfile, says its docstring."""\n'
               'KEY = DATA / ".keyfile"\n'
               'TLS = {"ssl_keyfile": "server.key"}\n'
               'def load():\n'
               '    """The key file at data/.keyfile."""\n'
               '    return open(f"{DATA}/.keyfile", "rb").read()[:32]\n')
    assert _names_the_key_file(planted) == [2, 6]

    namers = {}
    for name, path in package_modules().items():
        text = path.read_text(encoding="utf-8")
        if ".keyfile" in text:
            lines = _names_the_key_file(text)
            if lines:
                namers[name] = lines
    assert set(namers) == set(_NAMERS), {name: lines for name, lines in namers.items() if name not in _NAMERS}


# ---------------------------------------------------------------------------
# KY8 -- a key file that holds no 32-byte key is no key
# ---------------------------------------------------------------------------
def test_ky8_a_key_file_that_holds_no_32_byte_key_is_no_key_for_the_mode_the_allowlist_or_the_seal(
        tmp_path, monkeypatch, caplog):
    world = _world(tmp_path, monkeypatch)
    try:
        pa, sm = world.pa, world.sm
        plugin = _mk_plugin(tmp_path, "sample")
        manager = pa.PluginAllowlistManager()
        manifest = manager.prepare_batch([{"plugin_id": "sample", "plugin_dir": str(plugin), "permissions": []}])
        for held, size in ((b"", 0), (_cut_key_file(), 16), (_long_key_file(), 48)):
            if world.keyfile.exists():
                world.keyfile.unlink()
            _put_back(world, held)
            loaded = _raw(world.enc.load_keyfile()[0])
            assert len(loaded) == size, "the loader gives what the damaged file holds"
            why = f"the key file {world.keyfile} does not load here as a key: it holds {size} bytes, not 32"

            # The mode: a lockfile saying Daily, signed under what the file loads, reads Bulbe -- never Daily.
            world.yaml.write_text("security_mode: daily\n", encoding="utf-8")
            stamp = 1781753580.0
            signature = sm._compute_lockfile_hmac("daily", stamp, "leon", loaded)
            world.lock.write_text(f"MODE:daily\nTIMESTAMP:{stamp}\nUSER_ID:leon\nHMAC:{signature}\n",
                                  encoding="utf-8")
            mode, failure = sm._read_mode()
            assert mode == "bulbe" and failure[0] == "lockfile_unverifiable", (mode, failure)
            assert str(failure[3][0]) == why

            # The allowlist: an entry signed under it is refused, saying why; the approval signs nothing.
            pa._save_allowlist([_signed_entry(pa, "sample", plugin, [], loaded)])
            on_disk = world.allowlist.read_bytes()
            verdict = pa.PluginAllowlistManager().verify_plugin("sample", plugin, permissions=[])
            assert verdict == {"allowed": False, "reason": f"Unverifiable signature for 'sample': {why}"}
            result = manager.approve_batch(manifest.batch_id, "tester")
            assert result == {"success": False, "error": "no_signing_key",
                              "message": f"Cannot approve without the signing key: {why}"}
            assert world.allowlist.read_bytes() == on_disk and manager.get_pending_batch() is manifest
    finally:
        world.restore()

    seal = _provenance(tmp_path / "seal", monkeypatch)
    try:
        prov = seal.prov
        model = tmp_path / "pinned-model-Q4_K_M.gguf"
        model.write_bytes(b"GGUF" + bytes(range(256)) * 4)
        for held, size in ((b"", 0), (_cut_key_file(), 16), (_long_key_file(), 48)):
            if seal.keyfile.exists():
                seal.keyfile.unlink()
            seal.keyfile.write_bytes(held)
            os.chmod(seal.keyfile, 0o600)
            loaded = _raw(seal.enc.load_keyfile()[0])
            caplog.clear()
            with caplog.at_level(logging.ERROR, logger=PROV):
                assert prov._hmac_key() is None
            assert [r.getMessage() for r in caplog.records] == [
                f"the key file does not load here as a key: it holds {size} bytes, not 32; no model seal is made "
                "or checked under it"]
            reason, _ = prov.classify_model(model, _sealed(prov, model, loaded))
            assert reason == prov.REASON_KEY_UNAVAILABLE and prov.decide(reason, prov.ENFORCE) is False
    finally:
        seal.restore()


# ---------------------------------------------------------------------------
# KY9 -- an entry's signature is the allowlist's own
# ---------------------------------------------------------------------------
def test_ky9_an_entry_s_signature_is_the_allowlist_s_own_never_a_lockfile_s(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        pa, sm = world.pa, world.sm
        _key_file(world, wrapped=False)
        crafted = _mk_plugin(tmp_path, "crafted")
        manager = pa.PluginAllowlistManager()
        manifest = manager.prepare_batch([{"plugin_id": "daily||1.0||x", "plugin_dir": str(crafted),
                                           "permissions": []}])
        assert manager.approve_batch(manifest.batch_id, "tester")["success"] is True
        entry = pa.PluginAllowlistManager().get_entry("daily||1.0||x")

        # The forgery the scheme before let through: the entry's signature as a lockfile's, its fields laid out
        # so the lockfile's message is the entry's.
        world.yaml.write_text("security_mode: daily\n", encoding="utf-8")
        world.lock.write_text(f"MODE:daily\nTIMESTAMP:1.0\nUSER_ID:x||{entry.code_hash}||||{entry.batch_id}\n"
                              f"HMAC:{entry.signature}\n", encoding="utf-8")
        mode, failure = sm._read_mode()
        assert mode == "bulbe" and failure[0] == "lockfile_hmac_failure", (mode, failure)

        # Under the allowlist's own key, never the master key itself, over one record no other entry shares.
        record = pa._entry_message(entry)
        assert entry.signature == hmac.new(pa._entry_key(K1), record, hashlib.sha512).hexdigest()
        assert entry.signature != hmac.new(K1, record, hashlib.sha512).hexdigest()
        one = pa.AllowlistEntry(plugin_id="a||b", code_hash="c", approved_by="t", approved_at=1.0, batch_id="z")
        other = pa.AllowlistEntry(plugin_id="a", code_hash="b||c", approved_by="t", approved_at=1.0, batch_id="z")
        assert pa._sign_entry(one, K1) != pa._sign_entry(other, K1)

        # An entry signed before -- no scheme, the hash of its four suffixes, the master key over fields joined by
        # "||" -- is refused as outdated, never compared with today's hash; one whose files changed since, as changed;
        # one the scheme before does not verify, as tampered; and, with no key, as unverifiable.
        sample = _mk_plugin(tmp_path, "sample")
        code_hash = _hash_before(sample)
        assert code_hash != pa.compute_plugin_hash(sample)
        old = _entry_before(pa, "sample", code_hash)
        pa._save_allowlist([old])
        verdict = pa.PluginAllowlistManager().verify_plugin("sample", sample, permissions=[])
        outdated = {"allowed": False, "reason": "Outdated signature for 'sample': signed before the allowlist had a "
                                                "key of its own; approve it again"}
        assert verdict == outdated
        (sample / "plugin.py").write_text("VALUE = 2\n", encoding="utf-8")
        assert pa.PluginAllowlistManager().verify_plugin("sample", sample, permissions=[]) == {
            "allowed": False, "reason": "Code hash mismatch for 'sample': plugin files changed since approval"}
        (sample / "plugin.py").write_text("VALUE = 1\n", encoding="utf-8")
        tampered = {"allowed": False, "reason": "Invalid signature for 'sample': allowlist entry may be tampered"}
        forged = pa.AllowlistEntry.from_dict(dict(old.to_dict(), signature="0" * 128))
        pa._save_allowlist([forged])
        assert pa.PluginAllowlistManager().verify_plugin("sample", sample, permissions=[]) == tampered
        pa._save_allowlist([old])
        held = world.keyfile.read_bytes()
        world.keyfile.unlink()
        assert pa.PluginAllowlistManager().verify_plugin("sample", sample, permissions=[]) == {
            "allowed": False, "reason": "Unverifiable signature for 'sample': there is no key file"}
        _put_back(world, held)

        # An entry that does not hold what the allowlist writes is tampering, whatever its scheme -- each one below
        # carries a signature that verifies over its own fields, so only its form refuses it.
        today = _signed_entry(pa, "sample", sample, [], K1)
        assert pa._well_formed(today) and pa._well_formed(old)
        for field, value in (("approved_at", True), ("approved_at", "1000"), ("approved_by", 7), ("scheme", None)):
            malformed = pa.AllowlistEntry.from_dict(dict(old.to_dict(), **{field: value}))
            world.allowlist.write_text(json.dumps([malformed.to_dict()]), encoding="utf-8")
            assert pa.PluginAllowlistManager().verify_plugin("sample", sample, permissions=[]) == tampered, (field, value)
        for field, value in (("approved_at", True), ("approved_by", 7), ("batch_id", 7), ("permissions", [1]),
                             ("batch_id", "lone \udcff"), ("permissions", ["lone \udcff"])):
            malformed = pa.AllowlistEntry.from_dict(dict(today.to_dict(), **{field: value}))
            malformed.signature = pa._sign_entry(malformed, K1)
            world.allowlist.write_text(json.dumps([malformed.to_dict()]), encoding="utf-8")
            assert pa.PluginAllowlistManager().verify_plugin("sample", sample, permissions=[]) == tampered, (field, value)
        for signature in ("\u00e9" * 128, "A" * 128, "0" * 127):
            for entry in (old, today):
                malformed = pa.AllowlistEntry.from_dict(dict(entry.to_dict(), signature=signature))
                world.allowlist.write_text(json.dumps([malformed.to_dict()]), encoding="utf-8")
                assert pa.PluginAllowlistManager().verify_plugin("sample", sample, permissions=[]) == tampered, signature

        # Every field is signed: an approver, a time or a batch rewritten in the file is tampering.
        pa._save_allowlist([_signed_entry(pa, "sample", sample, [], K1)])
        signed = json.loads(world.allowlist.read_text(encoding="utf-8"))
        assert pa.PluginAllowlistManager().verify_plugin("sample", sample, permissions=[])["allowed"] is True
        for field, value in (("approved_by", "someone else"), ("approved_at", 2000.0), ("batch_id", "another")):
            world.allowlist.write_text(json.dumps([dict(signed[0], **{field: value})]), encoding="utf-8")
            assert pa.PluginAllowlistManager().verify_plugin("sample", sample, permissions=[]) == tampered, field
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# KY10 -- a plugin runs from its source
# ---------------------------------------------------------------------------
def test_ky10_a_plugin_runs_from_its_source_never_from_bytecode_cached_beside_it(tmp_path):
    saved = (sys.dont_write_bytecode, sys.pycache_prefix)
    loaded, restore = isolate(targets={"opti_oignon.plugin_worker": source("plugin_worker.py")})
    try:
        worker = loaded["opti_oignon.plugin_worker"]
        plugin = tmp_path / "sample"
        plugin.mkdir()
        entry = plugin / "plugin.py"
        # Bytecode beside the entry point, stamped with the source's time and size, holding other code.
        stamp = 1781753580
        entry.write_text("VALUE = 6\n", encoding="utf-8")
        os.utime(entry, (stamp, stamp))
        cache = Path(importlib.util.cache_from_source(str(entry)))
        py_compile.compile(str(entry), cfile=str(cache), doraise=True,
                           invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
        entry.write_text("VALUE = 1\n", encoding="utf-8")
        os.utime(entry, (stamp, stamp))
        planted = cache.read_bytes()

        module = worker.load_plugin_module("sample", str(plugin), "plugin.py")
        assert module.VALUE == 1, "the bytecode planted beside the entry point ran in its place"
        assert cache.read_bytes() == planted

        # Every other import of the worker compiles from its source, and none writes bytecode -- set before
        # anything else the worker does.
        worker._no_cached_bytecode()
        assert sys.dont_write_bytecode is True
        assert sys.pycache_prefix.startswith(os.path.join(os.sep, "oo-plugin-no-bytecode-"))
        assert not os.path.lexists(sys.pycache_prefix)
        main = next(node for node in ast.parse(source("plugin_worker.py").read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        first = main.body[1]
        assert (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
                and getattr(first.value.func, "id", None) == "_no_cached_bytecode"), ast.dump(first)
    finally:
        sys.dont_write_bytecode, sys.pycache_prefix = saved
        sys.modules.pop("_oo_worker_plugin_sample", None)
        restore()


# ---------------------------------------------------------------------------
# KY11 -- a plugin's hash covers everything its folder holds (C1, superseded)
# ---------------------------------------------------------------------------
def test_ky11_a_plugin_s_hash_covers_everything_its_folder_holds(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        pa = world.pa
        plugin = _mk_plugin(tmp_path, "sample")
        base = pa.compute_plugin_hash(plugin)
        assert base.startswith("sha512:") and pa.compute_plugin_hash(plugin) == base

        # Every file enters the digest, whatever its suffix, and every folder.
        for rel in ("notes.txt", "data.json", "extra.yml", "__pycache__/plugin.cpython-313.pyc", "helper/__init__.pyc",
                    "helper.abi3.so"):
            target = plugin / rel
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(b"planted")
            assert pa.compute_plugin_hash(plugin) not in (base, ""), rel
            target.unlink()
            if target.parent != plugin:
                target.parent.rmdir()
            assert pa.compute_plugin_hash(plugin) == base, rel
        (plugin / "empty_folder").mkdir()
        assert pa.compute_plugin_hash(plugin) not in (base, "")
        (plugin / "empty_folder").rmdir()

        # A content or a name changed changes it.
        (plugin / "plugin.py").write_text("VALUE = 2\n", encoding="utf-8")
        changed = pa.compute_plugin_hash(plugin)
        assert changed not in (base, "")
        (plugin / "plugin.py").rename(plugin / "plugin_renamed.py")
        renamed = pa.compute_plugin_hash(plugin)
        assert renamed not in (changed, "")
        (plugin / "plugin_renamed.py").rename(plugin / "plugin.py")
        assert pa.compute_plugin_hash(plugin) == changed

        # Missing, holding no file, holding a symbolic link or a pipe: nothing to hash.
        assert pa.compute_plugin_hash(tmp_path / "ghost") == ""
        (tmp_path / "empty" / "inner").mkdir(parents=True)
        assert pa.compute_plugin_hash(tmp_path / "empty") == ""
        outside = tmp_path / "outside.py"
        outside.write_text("VALUE = 7\n", encoding="utf-8")
        for link, target in ((plugin / "linked.py", outside), (plugin / "linked_folder", tmp_path / "empty")):
            link.symlink_to(target)
            assert pa.compute_plugin_hash(plugin) == "", link.name
            link.unlink()
        os.mkfifo(plugin / "pipe")
        assert pa.compute_plugin_hash(plugin) == ""
        (plugin / "pipe").unlink()

        # Something it cannot read -- a folder it may search but not list, a file -- whose content a plugin could
        # still reach by name.
        if os.geteuid() != 0:
            hidden = plugin / "hidden"
            hidden.mkdir()
            (hidden / "code.py").write_text("VALUE = 8\n", encoding="utf-8")
            hidden.chmod(0o111)
            try:
                assert pa.compute_plugin_hash(plugin) == ""
            finally:
                hidden.chmod(0o755)
            (hidden / "code.py").unlink()
            hidden.rmdir()
            secret = plugin / "secret.py"
            secret.write_text("VALUE = 9\n", encoding="utf-8")
            secret.chmod(0)
            try:
                assert pa.compute_plugin_hash(plugin) == ""
            finally:
                secret.chmod(0o644)
            secret.unlink()
        assert pa.compute_plugin_hash(plugin) == changed

        # Batch hash: order-independent, element-sensitive.
        pair = pa.compute_batch_hash([base, changed])
        assert pair == pa.compute_batch_hash([changed, base]) and pair.startswith("sha512:")
        assert pair != pa.compute_batch_hash([base, base]) and pair != pa.compute_batch_hash([base])
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# KY12 -- a plugin whose folder changed, or cannot be hashed, is neither approved nor verified
# ---------------------------------------------------------------------------
def test_ky12_a_plugin_whose_folder_changed_or_cannot_be_hashed_is_neither_approved_nor_verified(tmp_path,
                                                                                                  monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        pa = world.pa
        _key_file(world, wrapped=False)
        plugin = _mk_plugin(tmp_path, "sample")
        (plugin / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        manager = pa.PluginAllowlistManager()
        manifest = manager.prepare_batch([{"plugin_id": "sample", "plugin_dir": str(plugin), "permissions": []}])
        assert manager.approve_batch(manifest.batch_id, "tester")["success"] is True
        assert pa.PluginAllowlistManager().verify_plugin("sample", plugin, permissions=[]) == {"allowed": True,
                                                                                            "reason": "Verified"}

        # Bytecode planted in a package folder beside the approved module, which the import system finds first.
        (plugin / "helper").mkdir()
        (plugin / "helper" / "__init__.pyc").write_bytes(b"planted")
        verdict = pa.PluginAllowlistManager().verify_plugin("sample", plugin, permissions=[])
        assert verdict == {"allowed": False,
                           "reason": "Code hash mismatch for 'sample': plugin files changed since approval"}
        (plugin / "helper" / "__init__.pyc").unlink()
        (plugin / "helper").rmdir()

        # A folder that cannot be hashed is refused by name, at load and at approval.
        unhashable = ("Unhashable plugin 'sample': its folder is missing, holds no file, or holds a symbolic link, "
                      "a special file or something unreadable")
        (plugin / "linked.py").symlink_to(plugin / "helper.py")
        assert pa.PluginAllowlistManager().verify_plugin("sample", plugin, permissions=[]) == {
            "allowed": False, "reason": unhashable}
        on_disk = world.allowlist.read_bytes()
        batch = manager.prepare_batch([{"plugin_id": "sample", "plugin_dir": str(plugin), "permissions": []}])
        result = manager.approve_batch(batch.batch_id, "tester")
        assert result == {"success": False, "error": "unhashable",
                          "message": "Plugin 'sample' cannot be hashed: its folder is missing, holds no file, or "
                                     "holds a symbolic link, a special file or something unreadable."}
        assert world.allowlist.read_bytes() == on_disk and manager.get_pending_batch() is batch
        (plugin / "linked.py").unlink()

        # An entry naming the empty hash -- signed under the key, for a folder that is not there -- verified, the
        # hash of nothing matching the hash of nothing.
        pa._save_allowlist([_signed_entry(pa, "ghost", tmp_path / "ghost", [], K1)])
        verdict = pa.PluginAllowlistManager().verify_plugin("ghost", tmp_path / "ghost", permissions=[])
        assert verdict == {"allowed": False, "reason": unhashable.replace("'sample'", "'ghost'")}
    finally:
        world.restore()
