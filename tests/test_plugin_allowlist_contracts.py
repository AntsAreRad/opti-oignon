#!/usr/bin/env python3
"""Contracts for the Bulbe plugin allowlist (hashing, verification, ceremony).

  * Contract 1 -- hash discipline: the plugin hash is deterministic,
    covers exactly the source suffixes (.py/.yaml/.yml/.json), changes
    on content change AND on rename (the relative path is part of the
    digest), answers the empty string for missing or sourceless
    directories, and the batch hash is order-independent but sensitive
    to every element.
  * Contract 2 -- the four-point verifier: a plugin absent from the
    allowlist is refused; a signed entry with a matching on-disk hash
    and unchanged permissions verifies; a modified plugin tree is
    refused on hash; a tampered allowlist entry is refused on
    signature; a permission escalation is refused while equal or
    narrower requests pass; with the signing key absent the hash
    refusal still holds; revocation removes trust immediately and
    persistently.
  * Contract 3 -- the batch ceremony: prepare computes the composite
    hash over the per-plugin hashes; approval refuses an unknown batch
    id, refuses without a signing key (fail-closed), refuses when any
    plugin changed between preparation and approval, and a clean
    approval persists signed entries that a fresh manager verifies.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. The module is loaded in isolation with
a stub encryption seam so the signing key always comes from the
redirected on-disk keyfile -- deterministic in any environment.
"""

import importlib.util
import json
import sys
import tempfile
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

KEY_BYTES = b"K" * 48  # the loader keeps the first 32 bytes


# ---------------------------------------------------------------------------
# Isolated loading
# ---------------------------------------------------------------------------
def _load():
    keys = (
        "opti_oignon",
        "opti_oignon.encryption",
        "opti_oignon.security_mode",
        "opti_oignon.plugin_allowlist",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    enc = types.ModuleType("opti_oignon.encryption")

    def _no_keyfile_module():
        raise RuntimeError("scripted: keyfile module unavailable")

    enc.load_keyfile = _no_keyfile_module
    sys.modules["opti_oignon.encryption"] = enc
    pkg.encryption = enc

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.plugin_allowlist", _OO / "plugin_allowlist.py",
    )
    pa = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.plugin_allowlist"] = pa
    spec.loader.exec_module(pa)

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return pa, restore


def _env(pa, tmp: Path, with_key: bool = True) -> None:
    """Point the module at a scratch data dir; optionally seed the keyfile."""
    pa._DATA_DIR = tmp
    pa._ALLOWLIST_PATH = tmp / "plugin_allowlist.json"
    keyfile = tmp / ".keyfile"
    if with_key:
        keyfile.write_bytes(KEY_BYTES)
    elif keyfile.exists():
        keyfile.unlink()


def _mk_plugin(root: Path, name: str, code: str = "VALUE = 1\n") -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(code, encoding="utf-8")
    (plugin_dir / "manifest.yaml").write_text(
        f"id: {name}\n", encoding="utf-8",
    )
    return plugin_dir


def _signed_entry(pa, plugin_id: str, plugin_dir: Path,
                  permissions: list[str]):
    key = pa._load_signing_key()
    assert key is not None, "keyfile must be seeded before signing"
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


# ---------------------------------------------------------------------------
# Contract 1 -- hash discipline
# ---------------------------------------------------------------------------
def test_c1_hash_discipline():
    pa, restore = _load()
    try:
        with tempfile.TemporaryDirectory(prefix="allowlist-c1-") as raw:
            tmp = Path(raw)
            plugin = _mk_plugin(tmp, "sample")

            base = pa.compute_plugin_hash(plugin)
            assert base.startswith("sha512:"), base
            assert pa.compute_plugin_hash(plugin) == base

            # Files outside the source suffixes do not enter the digest.
            (plugin / "notes.txt").write_text("scratch", encoding="utf-8")
            assert pa.compute_plugin_hash(plugin) == base

            # .json and .yml are covered.
            (plugin / "data.json").write_text("{}", encoding="utf-8")
            with_json = pa.compute_plugin_hash(plugin)
            assert with_json != base
            (plugin / "extra.yml").write_text("a: 1\n", encoding="utf-8")
            with_yml = pa.compute_plugin_hash(plugin)
            assert with_yml != with_json
            (plugin / "data.json").unlink()
            (plugin / "extra.yml").unlink()
            assert pa.compute_plugin_hash(plugin) == base

            # Content change changes the digest.
            (plugin / "plugin.py").write_text(
                "VALUE = 2\n", encoding="utf-8",
            )
            changed = pa.compute_plugin_hash(plugin)
            assert changed != base

            # Rename changes the digest: the relative path is hashed.
            (plugin / "plugin.py").rename(plugin / "plugin_renamed.py")
            renamed = pa.compute_plugin_hash(plugin)
            assert renamed != changed
            (plugin / "plugin_renamed.py").rename(plugin / "plugin.py")
            assert pa.compute_plugin_hash(plugin) == changed

            # Missing or sourceless directories answer the empty string.
            assert pa.compute_plugin_hash(tmp / "ghost") == ""
            empty = tmp / "empty"
            empty.mkdir()
            assert pa.compute_plugin_hash(empty) == ""

            # Batch hash: order-independent, element-sensitive.
            pair = pa.compute_batch_hash([base, changed])
            assert pair == pa.compute_batch_hash([changed, base])
            assert pair.startswith("sha512:")
            assert pair != pa.compute_batch_hash([base, base])
            assert pair != pa.compute_batch_hash([base])
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- the four-point verifier and revocation
# ---------------------------------------------------------------------------
def test_c2_verifier_four_points():
    pa, restore = _load()
    try:
        with tempfile.TemporaryDirectory(prefix="allowlist-c2-") as raw:
            tmp = Path(raw)
            _env(pa, tmp, with_key=True)
            plugin = _mk_plugin(tmp, "sample")
            good_code = (plugin / "plugin.py").read_bytes()

            # Absent from the allowlist: refused.
            manager = pa.PluginAllowlistManager()
            verdict = manager.verify_plugin("sample", plugin)
            assert verdict["allowed"] is False
            assert "not in the allowlist" in verdict["reason"]

            # Signed entry, matching hash, same permissions: verified.
            entry = _signed_entry(pa, "sample", plugin, ["read"])
            pa._save_allowlist([entry])
            manager = pa.PluginAllowlistManager()
            verdict = manager.verify_plugin(
                "sample", plugin, permissions=["read"],
            )
            assert verdict == {"allowed": True, "reason": "Verified"}

            # Modified plugin tree: refused on hash.
            (plugin / "plugin.py").write_text(
                "VALUE = 99\n", encoding="utf-8",
            )
            verdict = pa.PluginAllowlistManager().verify_plugin(
                "sample", plugin, permissions=["read"],
            )
            assert verdict["allowed"] is False
            assert "changed since approval" in verdict["reason"]
            (plugin / "plugin.py").write_bytes(good_code)

            # Tampered allowlist entry (permissions widened on disk
            # without re-signing): refused on signature.
            data = json.loads(pa._ALLOWLIST_PATH.read_text())
            data[0]["permissions"].append("network")
            pa._ALLOWLIST_PATH.write_text(json.dumps(data))
            verdict = pa.PluginAllowlistManager().verify_plugin(
                "sample", plugin, permissions=["read"],
            )
            assert verdict["allowed"] is False
            assert "tampered" in verdict["reason"]
            pa._save_allowlist([entry])

            # Permission escalation: refused; equal or narrower passes.
            fresh = pa.PluginAllowlistManager()
            verdict = fresh.verify_plugin(
                "sample", plugin, permissions=["read", "network"],
            )
            assert verdict["allowed"] is False
            assert "Permission escalation" in verdict["reason"]
            assert fresh.verify_plugin(
                "sample", plugin, permissions=[],
            )["allowed"] is True
            assert fresh.verify_plugin(
                "sample", plugin, permissions=None,
            )["allowed"] is True

            # Signing key absent: the hash refusal still holds.
            (tmp / ".keyfile").unlink()
            (plugin / "plugin.py").write_text(
                "VALUE = 99\n", encoding="utf-8",
            )
            verdict = pa.PluginAllowlistManager().verify_plugin(
                "sample", plugin, permissions=["read"],
            )
            assert verdict["allowed"] is False
            assert "changed since approval" in verdict["reason"]
            (plugin / "plugin.py").write_bytes(good_code)
            (tmp / ".keyfile").write_bytes(KEY_BYTES)

            # Revocation removes trust immediately and persistently.
            manager = pa.PluginAllowlistManager()
            assert manager.is_allowed("sample") is True
            assert manager.revoke_plugin("sample") is True
            assert manager.is_allowed("sample") is False
            assert manager.revoke_plugin("sample") is False
            verdict = pa.PluginAllowlistManager().verify_plugin(
                "sample", plugin, permissions=["read"],
            )
            assert verdict["allowed"] is False
            assert "not in the allowlist" in verdict["reason"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- the batch approval ceremony
# ---------------------------------------------------------------------------
def test_c3_batch_ceremony():
    pa, restore = _load()
    try:
        with tempfile.TemporaryDirectory(prefix="allowlist-c3-") as raw:
            tmp = Path(raw)
            _env(pa, tmp, with_key=True)
            first = _mk_plugin(tmp, "first")
            second = _mk_plugin(tmp, "second", code="VALUE = 2\n")
            first_code = (first / "plugin.py").read_bytes()

            manager = pa.PluginAllowlistManager()
            manifest = manager.prepare_batch([
                {
                    "plugin_id": "first",
                    "plugin_dir": str(first),
                    "permissions": ["read"],
                },
                {
                    "plugin_id": "second",
                    "plugin_dir": str(second),
                    "permissions": [],
                },
            ])
            hashes = [p["code_hash"] for p in manifest.plugins]
            assert manifest.batch_hash == pa.compute_batch_hash(hashes)
            assert manager.get_pending_batch() is manifest

            # Unknown batch id: refused, nothing persisted.
            result = manager.approve_batch("not-a-batch", "tester")
            assert result["success"] is False
            assert result["error"] == "no_matching_batch"
            assert not pa._ALLOWLIST_PATH.exists()

            # No signing key: refused (fail-closed), nothing persisted.
            (tmp / ".keyfile").unlink()
            result = manager.approve_batch(manifest.batch_id, "tester")
            assert result["success"] is False
            assert result["error"] == "no_signing_key"
            assert not pa._ALLOWLIST_PATH.exists()
            (tmp / ".keyfile").write_bytes(KEY_BYTES)

            # A plugin changed between prepare and approve: refused.
            (first / "plugin.py").write_text(
                "VALUE = 99\n", encoding="utf-8",
            )
            result = manager.approve_batch(manifest.batch_id, "tester")
            assert result["success"] is False
            assert result["error"] == "hash_changed"
            assert not pa._ALLOWLIST_PATH.exists()
            (first / "plugin.py").write_bytes(first_code)

            # Clean approval: signed entries persisted, pending cleared,
            # a fresh manager verifies both plugins.
            result = manager.approve_batch(manifest.batch_id, "tester")
            assert result == {
                "success": True,
                "entries_added": 2,
                "batch_id": manifest.batch_id,
            }
            assert manager.get_pending_batch() is None
            assert pa._ALLOWLIST_PATH.exists()

            fresh = pa.PluginAllowlistManager()
            verdict = fresh.verify_plugin(
                "first", first, permissions=["read"],
            )
            assert verdict == {"allowed": True, "reason": "Verified"}
            verdict = fresh.verify_plugin("second", second, permissions=[])
            assert verdict == {"allowed": True, "reason": "Verified"}
            for stored in fresh.list_entries():
                assert stored.batch_id == manifest.batch_id
                assert stored.approved_by == "tester"
                assert stored.signature
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
