#!/usr/bin/env python3
"""
Plugin Allowlist for Opti-Oignon Bulbe mode (S126).

In Bulbe mode, only explicitly approved plugins may be loaded.  Each plugin
is identified by a SHA-512 hash of its source files.  Approval uses a
batch ceremony (same security as mode degradation) and produces HMAC-SHA512
signed allowlist entries.

The allowlist is stored in ``data/plugin_allowlist.json``.  Every plugin
load in Bulbe mode verifies:

  1. The plugin is in the allowlist
  2. Its current on-disk hash matches the approved hash
  3. The HMAC signature on the allowlist entry is valid
  4. Its permissions have not escalated since approval

Security derives from the signing key and the human approval ceremony,
not from code obscurity (Kerckhoffs principle).

In Daily mode, the allowlist is ignored and all plugins load normally.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import logging
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_ALLOWLIST_PATH = _DATA_DIR / "plugin_allowlist.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AllowlistEntry:
    """A single approved plugin in the allowlist."""
    plugin_id: str
    code_hash: str  # "sha512:<hex>"
    approved_by: str
    approved_at: float
    batch_id: str
    permissions: list[str] = field(default_factory=list)
    signature: str = ""  # HMAC-SHA512

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AllowlistEntry:
        return cls(
            plugin_id=data.get("plugin_id", ""),
            code_hash=data.get("code_hash", ""),
            approved_by=data.get("approved_by", ""),
            approved_at=data.get("approved_at", 0.0),
            batch_id=data.get("batch_id", ""),
            permissions=data.get("permissions", []),
            signature=data.get("signature", ""),
        )


@dataclass
class BatchManifest:
    """A batch of plugins pending approval."""
    batch_id: str
    plugins: list[dict[str, Any]] = field(default_factory=list)
    batch_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_plugin_hash(plugin_dir: Path) -> str:
    """Compute SHA-512 of all plugin source files.

    Files are sorted by relative path for deterministic ordering.
    The hash covers the file content of every ``.py`` and ``.yaml``
    file in the plugin directory.
    """
    plugin_path = Path(plugin_dir).resolve()
    if not plugin_path.is_dir():
        return ""

    hasher = hashlib.sha512()
    source_files = sorted(
        p for p in plugin_path.rglob("*")
        if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json")
    )

    if not source_files:
        return ""

    for fpath in source_files:
        rel = fpath.relative_to(plugin_path)
        # Include the relative path in the hash so renaming is detected
        hasher.update(str(rel).encode("utf-8"))
        hasher.update(fpath.read_bytes())

    return f"sha512:{hasher.hexdigest()}"


def compute_batch_hash(plugin_hashes: list[str]) -> str:
    """Compute a composite hash over all individual plugin hashes.

    The batch hash covers sorted concatenation of all individual
    hashes so that any plugin change invalidates the batch.
    """
    combined = "||".join(sorted(plugin_hashes))
    return f"sha512:{hashlib.sha512(combined.encode('utf-8')).hexdigest()}"


# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------

def _load_signing_key():
    """Load signing key from the keyfile.

    S129: Returns SecureBytes when available (from load_keyfile()),
    falls back to raw bytes from file if keyfile module is unavailable.
    Use .as_bytes() to extract raw bytes for HMAC operations.
    """
    try:
        from opti_oignon.encryption import load_keyfile
        key, _salt, _kdf = load_keyfile()
        return key  # SecureBytes (S129)
    except Exception:
        pass
    try:
        keyfile = _DATA_DIR / ".keyfile"
        if keyfile.exists():
            raw = keyfile.read_bytes()
            if len(raw) >= 32:
                return raw[:32]
    except Exception:
        pass
    return None


def _extract_key_bytes(key) -> bytes:
    """Extract raw bytes from a key (SecureBytes or plain bytes).

    S129: Helper for HMAC operations that require raw bytes.
    """
    if hasattr(key, "as_bytes"):
        return key.as_bytes()
    return key


def _sign_entry(entry: AllowlistEntry, key) -> str:
    """Compute HMAC-SHA512 signature for an allowlist entry.

    Covers plugin_id || code_hash || permissions || batch_id.
    Security derives from the key, not from format secrecy.

    S129: Accepts SecureBytes or raw bytes via _extract_key_bytes().
    """
    raw_key = _extract_key_bytes(key)
    perms_str = ",".join(sorted(entry.permissions))
    message = (
        f"{entry.plugin_id}||{entry.code_hash}||{perms_str}||{entry.batch_id}"
    ).encode()
    return _hmac.new(raw_key, message, hashlib.sha512).hexdigest()


def _verify_entry_signature(entry: AllowlistEntry, key) -> bool:
    """Verify the HMAC signature on an allowlist entry."""
    expected = _sign_entry(entry, key)
    return _hmac.compare_digest(expected, entry.signature)


# ---------------------------------------------------------------------------
# Allowlist persistence
# ---------------------------------------------------------------------------

def _load_allowlist() -> list[AllowlistEntry]:
    """Load the allowlist from disk."""
    try:
        if _ALLOWLIST_PATH.exists():
            with open(_ALLOWLIST_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return [AllowlistEntry.from_dict(d) for d in data]
    except Exception as exc:
        logger.warning("Failed to load plugin allowlist: %s", exc)
    return []


def _save_allowlist(entries: list[AllowlistEntry]) -> None:
    """Save the allowlist to disk."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ALLOWLIST_PATH, "w", encoding="utf-8") as fh:
        json.dump([e.to_dict() for e in entries], fh, indent=2)


# ---------------------------------------------------------------------------
# PluginAllowlistManager
# ---------------------------------------------------------------------------

class PluginAllowlistManager:
    """Manages the plugin allowlist for Bulbe mode.

    In Bulbe mode, every plugin load is checked against the allowlist.
    In Daily mode, the allowlist is ignored.
    """

    def __init__(self) -> None:
        self._entries: list[AllowlistEntry] = []
        self._loaded = False
        self._pending_batch: BatchManifest | None = None

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._entries = _load_allowlist()
            self._loaded = True

    def reload(self) -> None:
        """Force reload from disk."""
        self._entries = _load_allowlist()
        self._loaded = True

    # -- Query ---------------------------------------------------------------

    def get_entry(self, plugin_id: str) -> AllowlistEntry | None:
        """Get the allowlist entry for a plugin, or None."""
        self._ensure_loaded()
        for entry in self._entries:
            if entry.plugin_id == plugin_id:
                return entry
        return None

    def list_entries(self) -> list[AllowlistEntry]:
        """Return all allowlist entries."""
        self._ensure_loaded()
        return list(self._entries)

    def is_allowed(self, plugin_id: str) -> bool:
        """Check if a plugin is in the allowlist (without hash verification)."""
        return self.get_entry(plugin_id) is not None

    # -- Verification (called at load time) ----------------------------------

    def verify_plugin(
        self, plugin_id: str, plugin_dir: Path, permissions: list[str] | None = None
    ) -> dict[str, Any]:
        """Verify a plugin against the allowlist.

        Returns a dict with 'allowed' bool and 'reason' string.
        Called by PluginLoader before loading in Bulbe mode.
        """
        self._ensure_loaded()
        entry = self.get_entry(plugin_id)

        if entry is None:
            return {
                "allowed": False,
                "reason": f"Plugin '{plugin_id}' is not in the allowlist",
            }

        # Verify code hash
        current_hash = compute_plugin_hash(plugin_dir)
        if current_hash != entry.code_hash:
            logger.critical(
                "Plugin '%s' code hash MISMATCH. Expected %s, got %s. "
                "Plugin may have been modified since approval.",
                plugin_id, entry.code_hash[:30], current_hash[:30],
            )
            return {
                "allowed": False,
                "reason": (
                    f"Code hash mismatch for '{plugin_id}': "
                    "plugin files changed since approval"
                ),
            }

        # Verify HMAC signature
        key = _load_signing_key()
        if key:
            if not _verify_entry_signature(entry, key):
                logger.critical(
                    "Plugin '%s' allowlist signature INVALID. "
                    "Possible allowlist tampering.",
                    plugin_id,
                )
                return {
                    "allowed": False,
                    "reason": (
                        f"Invalid signature for '{plugin_id}': "
                        "allowlist entry may be tampered"
                    ),
                }

        # Check permission escalation
        if permissions is not None:
            new_perms = set(permissions) - set(entry.permissions)
            if new_perms:
                logger.warning(
                    "Plugin '%s' requests new permissions not in approval: %s",
                    plugin_id, new_perms,
                )
                return {
                    "allowed": False,
                    "reason": (
                        f"Permission escalation for '{plugin_id}': "
                        f"new permissions {new_perms} require re-approval"
                    ),
                }

        return {"allowed": True, "reason": "Verified"}

    # -- Batch approval ceremony ---------------------------------------------

    def prepare_batch(
        self, plugins: list[dict[str, Any]]
    ) -> BatchManifest:
        """Prepare a batch manifest for approval.

        Each plugin dict should have:
          - plugin_id: str
          - plugin_dir: str (path)
          - permissions: list[str]

        Returns a BatchManifest ready for the ceremony.
        """
        batch_id = secrets.token_urlsafe(16)
        manifest_plugins = []
        hashes = []

        for p in plugins:
            plugin_dir = Path(p["plugin_dir"]).resolve()
            code_hash = compute_plugin_hash(plugin_dir)
            hashes.append(code_hash)
            manifest_plugins.append({
                "plugin_id": p["plugin_id"],
                "code_hash": code_hash,
                "permissions": p.get("permissions", []),
                "plugin_dir": str(plugin_dir),
            })

        batch_hash = compute_batch_hash(hashes)
        manifest = BatchManifest(
            batch_id=batch_id,
            plugins=manifest_plugins,
            batch_hash=batch_hash,
        )
        self._pending_batch = manifest
        return manifest

    def get_pending_batch(self) -> BatchManifest | None:
        """Return the current pending batch, if any."""
        return self._pending_batch

    def approve_batch(
        self, batch_id: str, user_id: str
    ) -> dict[str, Any]:
        """Approve a pending batch after ceremony verification.

        The ceremony (visual code + password + 2FA) is verified by
        the API route before calling this method.

        Returns {success, entries_added, batch_id}.
        """
        batch = self._pending_batch
        if not batch or batch.batch_id != batch_id:
            return {
                "success": False,
                "error": "no_matching_batch",
                "message": "No pending batch with that ID",
            }

        key = _load_signing_key()
        if not key:
            return {
                "success": False,
                "error": "no_signing_key",
                "message": "Cannot approve without signing key",
            }

        # Verify batch hash still matches (no changes since prepare)
        current_hashes = []
        for p in batch.plugins:
            current_hash = compute_plugin_hash(Path(p["plugin_dir"]))
            if current_hash != p["code_hash"]:
                return {
                    "success": False,
                    "error": "hash_changed",
                    "message": (
                        f"Plugin '{p['plugin_id']}' changed since review. "
                        "Re-prepare the batch."
                    ),
                }
            current_hashes.append(current_hash)

        current_batch_hash = compute_batch_hash(current_hashes)
        if current_batch_hash != batch.batch_hash:
            return {
                "success": False,
                "error": "batch_hash_mismatch",
                "message": "Batch content changed since preparation",
            }

        # Create signed entries
        self._ensure_loaded()
        now = time.time()
        new_entries = []

        for p in batch.plugins:
            entry = AllowlistEntry(
                plugin_id=p["plugin_id"],
                code_hash=p["code_hash"],
                approved_by=user_id,
                approved_at=now,
                batch_id=batch.batch_id,
                permissions=p.get("permissions", []),
            )
            entry.signature = _sign_entry(entry, key)
            new_entries.append(entry)

        # Replace existing entries for these plugins
        existing_ids = {e.plugin_id for e in new_entries}
        self._entries = [
            e for e in self._entries if e.plugin_id not in existing_ids
        ]
        self._entries.extend(new_entries)
        _save_allowlist(self._entries)
        self._pending_batch = None

        # Audit
        try:
            from opti_oignon.security_mode import _audit_log
            _audit_log(
                "plugins_batch_approved",
                severity="WARNING",
                user_id=user_id,
                batch_id=batch.batch_id,
                plugin_count=len(new_entries),
                plugin_ids=[e.plugin_id for e in new_entries],
            )
        except Exception:
            pass

        return {
            "success": True,
            "entries_added": len(new_entries),
            "batch_id": batch.batch_id,
        }

    # -- Revocation ----------------------------------------------------------

    def revoke_plugin(self, plugin_id: str) -> bool:
        """Revoke a single plugin.  No ceremony needed (removing trust is safe)."""
        self._ensure_loaded()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.plugin_id != plugin_id]
        if len(self._entries) < before:
            _save_allowlist(self._entries)
            try:
                from opti_oignon.security_mode import _audit_log
                _audit_log(
                    "plugin_revoked",
                    severity="INFO",
                    plugin_id=plugin_id,
                )
            except Exception:
                pass
            return True
        return False

    def revoke_batch(self, batch_id: str) -> int:
        """Revoke all plugins from a given batch.  No ceremony needed."""
        self._ensure_loaded()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.batch_id != batch_id]
        removed = before - len(self._entries)
        if removed > 0:
            _save_allowlist(self._entries)
            try:
                from opti_oignon.security_mode import _audit_log
                _audit_log(
                    "plugin_batch_revoked",
                    severity="INFO",
                    batch_id=batch_id,
                    plugins_removed=removed,
                )
            except Exception:
                pass
        return removed

    def revoke_all(self) -> int:
        """Revoke all plugins."""
        self._ensure_loaded()
        count = len(self._entries)
        self._entries = []
        _save_allowlist(self._entries)
        return count

    # -- Status ---------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return allowlist status for the API."""
        self._ensure_loaded()
        batches: dict[str, int] = {}
        for e in self._entries:
            batches[e.batch_id] = batches.get(e.batch_id, 0) + 1
        return {
            "total_entries": len(self._entries),
            "batches": batches,
            "pending_batch": self._pending_batch.to_dict() if self._pending_batch else None,
            "entries": [e.to_dict() for e in self._entries],
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

plugin_allowlist_manager = PluginAllowlistManager()
