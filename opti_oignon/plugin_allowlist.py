#!/usr/bin/env python3
"""
Plugin Allowlist for Opti-Oignon Bulbe mode.

In Bulbe mode, only explicitly approved plugins may be loaded.  Each plugin
is identified by a SHA-512 hash of everything its folder holds.  Approval uses a
batch ceremony -- the owner's password, through the owner's own session
(``ceremony_factors``); the visual code and the second factor it carries are
not read yet -- and produces HMAC-SHA512 signed allowlist entries.

The allowlist is stored in ``data/plugin_allowlist.json``.  Every plugin
load in Bulbe mode verifies:

  1. The plugin is in the allowlist
  2. Its current on-disk hash matches the approved hash
  3. The HMAC signature on the allowlist entry is valid under the
     allowlist's own key, derived from the key that loads -- the security
     mode's, the key file's key -- over all the entry's fields
  4. Its permissions have not escalated since approval

Security derives from the signing key and the human approval ceremony,
not from code obscurity (Kerckhoffs principle). With no key file, or one
that does not load here, no entry is signed and none verifies: the key
file's bytes are never a key. The allowlist's key is its own, so an
entry's signature is never a signature of anything else the master key
signs -- the mode's lockfile is signed under the master key itself.

In Daily mode, the allowlist is ignored and all plugins load normally.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import logging
import os
import secrets
import stat
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

# The signing scheme of an entry: what its signature covers and under which
# key. An entry that carries another scheme was signed before and is refused.
_ENTRY_SCHEME = "allowlist-v2"
# The domain the allowlist's key is derived on from the master key.
_ENTRY_KEY_LABEL = b"opti-oignon-plugin-allowlist-v2"


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
    scheme: str = _ENTRY_SCHEME

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
            scheme=data.get("scheme", ""),
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
    """SHA-512 of everything the plugin's folder holds; "" when it cannot be hashed.

    Every file and folder under it enters the digest, sorted by relative
    path: a file by its path and the SHA-512 of its bytes, a folder by its
    path, all of it as one canonical record. Whatever the import system
    could run from there -- a source, bytecode with or without one, native
    code -- is covered with every other file: the plugin changes when
    anything in its folder does. A folder that is missing or holds no file
    gives "", and so does one holding a symbolic link -- hashed as one thing,
    it could lead to another when the plugin runs --, a special file (a
    pipe, a socket, a device), which is never read, or a folder or file it
    cannot read, whose content the plugin might still reach. An unhashable
    plugin is never approved and never verifies.
    """
    plugin_path = Path(plugin_dir).resolve()
    if not plugin_path.is_dir():
        return ""
    listing, unread = [], []
    for folder, dirs, files in os.walk(plugin_path, onerror=unread.append, followlinks=False):
        for name in sorted(dirs) + sorted(files):
            entry = Path(folder) / name
            rel = entry.relative_to(plugin_path).as_posix()
            try:
                mode = os.lstat(entry).st_mode
            except OSError:
                return ""
            if stat.S_ISDIR(mode):
                listing.append(["folder", rel])
                continue
            digest = _file_digest(entry) if stat.S_ISREG(mode) else None
            if digest is None:
                return ""
            listing.append(["file", rel, digest])
    if unread or not any(kind == "file" for kind, *_rest in listing):
        return ""
    record = json.dumps(sorted(listing, key=lambda item: (item[1], item[0])), separators=(",", ":"),
                        ensure_ascii=True)
    return f"sha512:{hashlib.sha512(record.encode('ascii')).hexdigest()}"


def _file_digest(path: Path) -> str | None:
    """The SHA-512 of a regular file's bytes, or None: opened without following a link and without
    waiting on a pipe, so a file swapped for either after it was listed is not read."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    digest = hashlib.sha512()
    try:
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _hash_before(plugin_dir: Path) -> str:
    """A plugin's hash as the scheme before made it: the paths and bytes of its .py, .yaml, .yml and .json files,
    one after the other. Kept only to tell an entry of that scheme whose files changed since its approval from one
    whose files did not; never compared with an entry of today's scheme."""
    plugin_path = Path(plugin_dir).resolve()
    if not plugin_path.is_dir():
        return ""
    hasher = hashlib.sha512()
    sources = sorted(
        p for p in plugin_path.rglob("*")
        if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json")
    )
    if not sources:
        return ""
    try:
        for source in sources:
            hasher.update(str(source.relative_to(plugin_path)).encode("utf-8"))
            with open(source, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    hasher.update(block)
    except OSError:
        return ""
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
    """The key the allowlist signs and verifies with: the security mode's.

    The key file's key when it loads (SecureBytes); None when there is no key
    file. A key file that is there and does not load -- wrapped under a
    passphrase this process lacks, damaged, unreadable, or holding no
    32-byte key, as an empty or cut one loads -- raises the mode's
    ``SigningKeyUnavailable``. Its bytes are never a key: the first 32 bytes
    of every wrapped key file are the same text, and an entry signed with
    them is signed with a key anyone holds. Entries are signed under the
    allowlist's own key, derived from this one (``_entry_key``).
    """
    from opti_oignon.security_mode import _load_signing_key as mode_signing_key

    return mode_signing_key()


def _key_or_why() -> tuple[Any, str | None]:
    """``(key, None)`` when the key loads; ``(None, why)`` when none does.

    No key file, a key file that does not load here, and a security mode
    that does not import -- it reads the key -- each say why.
    """
    try:
        key = _load_signing_key()
    except ImportError as exc:
        return None, f"the security mode, which reads the key, does not import ({exc})"
    except Exception as exc:  # noqa: BLE001 - whatever stops the load, no key loads
        return None, str(exc) or type(exc).__name__
    if key is None:
        return None, "there is no key file"
    return key, None


def _extract_key_bytes(key) -> bytes:
    """Extract raw bytes from a key (SecureBytes or plain bytes).

    Helper for HMAC operations that require raw bytes.
    """
    if hasattr(key, "as_bytes"):
        return key.as_bytes()
    return key


def _entry_key(key) -> bytes:
    """The allowlist's own key: HMAC-SHA256 of its domain under the master key.

    The mode's lockfile is signed under the master key itself, over fields
    joined by ``||``; an entry signed under it too, over fields a caller
    chooses, could be laid out as a lockfile that verifies. Under a key of
    its own, an entry's signature is never anything else's.
    """
    return _hmac.new(_extract_key_bytes(key), _ENTRY_KEY_LABEL, hashlib.sha256).digest()


def _entry_message(entry: AllowlistEntry) -> bytes:
    """The bytes an entry's signature covers: every field but the signature, as one record.

    Canonical JSON -- sorted keys, no whitespace, ASCII -- so two entries
    never share it whatever their fields hold: fields joined by ``||``
    could be cut elsewhere and read as other fields. The approver and the
    time are covered too.
    """
    return json.dumps(
        {
            "scheme": _ENTRY_SCHEME,
            "plugin_id": entry.plugin_id,
            "code_hash": entry.code_hash,
            "permissions": sorted(entry.permissions),
            "batch_id": entry.batch_id,
            "approved_by": entry.approved_by,
            "approved_at": entry.approved_at,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _signed_before(entry: AllowlistEntry, key) -> bool:
    """Whether the entry carries a signature of the scheme before ``allowlist-v2``: the master key itself over
    plugin_id, code_hash, the sorted permissions and batch_id joined by ``||``. Checked, never made: a signature
    of that scheme could stand in for the mode's lockfile's."""
    record = "||".join((entry.plugin_id, entry.code_hash, ",".join(sorted(entry.permissions)), entry.batch_id))
    expected = _hmac.new(_extract_key_bytes(key), record.encode(), hashlib.sha512).hexdigest()
    return _hmac.compare_digest(expected, entry.signature)


def _sign_entry(entry: AllowlistEntry, key) -> str:
    """HMAC-SHA512 of the entry's record under the allowlist's own key.

    ``key`` is the master key (SecureBytes or raw bytes); the allowlist's
    key is derived from it (``_entry_key``). Security derives from the key,
    not from format secrecy.
    """
    return _hmac.new(_entry_key(key), _entry_message(entry), hashlib.sha512).hexdigest()


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


def _well_formed(entry: AllowlistEntry) -> bool:
    """Whether every field holds what the allowlist writes there: texts that encode as UTF-8, a list of them, a
    time in seconds, and a signature of 128 lowercase hexadecimal digits -- so checking it refuses, and never
    raises."""
    texts = (entry.plugin_id, entry.code_hash, entry.approved_by, entry.batch_id, entry.signature, entry.scheme)
    if not (
        all(isinstance(value, str) for value in texts)
        and isinstance(entry.permissions, list)
        and all(isinstance(permission, str) for permission in entry.permissions)
        and isinstance(entry.approved_at, (int, float))
        and not isinstance(entry.approved_at, bool)
    ):
        return False
    if len(entry.signature) != 128 or not all(digit in "0123456789abcdef" for digit in entry.signature):
        return False
    try:
        for value in (*texts, *entry.permissions):
            value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _tampered(plugin_id: str) -> dict[str, Any]:
    return {
        "allowed": False,
        "reason": f"Invalid signature for '{plugin_id}': allowlist entry may be tampered",
    }


def _unverifiable(plugin_id: str, why: str | None) -> dict[str, Any]:
    logger.error("Plugin '%s' allowlist signature cannot be checked: %s.", plugin_id, why)
    return {
        "allowed": False,
        "reason": f"Unverifiable signature for '{plugin_id}': {why}",
    }


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

        # An entry that does not hold what the allowlist writes -- text, a list
        # of texts, a time -- was not written by it.
        if not _well_formed(entry):
            logger.critical(
                "Plugin '%s' allowlist entry is malformed. Possible allowlist tampering.",
                plugin_id,
            )
            return _tampered(plugin_id)

        # An entry of another scheme is settled first, under the key: its hash
        # is of the scheme before too, and is not compared with today's. It is
        # outdated only when the scheme before -- the master key itself over
        # fields joined by "||", checked, never made -- verifies it; any other
        # is tampered.
        if entry.scheme != _ENTRY_SCHEME:
            key, why = _key_or_why()
            if key is None:
                return _unverifiable(plugin_id, why)
            if _signed_before(entry, key):
                # Its files changed since that approval when the hash the
                # scheme before made of them no longer matches: say so, rather
                # than ask for an approval again.
                if _hash_before(plugin_dir) != entry.code_hash:
                    logger.critical(
                        "Plugin '%s' code hash MISMATCH under the scheme before. "
                        "Plugin may have been modified since approval.",
                        plugin_id,
                    )
                    return {
                        "allowed": False,
                        "reason": (
                            f"Code hash mismatch for '{plugin_id}': "
                            "plugin files changed since approval"
                        ),
                    }
                logger.error(
                    "Plugin '%s' allowlist entry was signed before the scheme %s.",
                    plugin_id, _ENTRY_SCHEME,
                )
                return {
                    "allowed": False,
                    "reason": (
                        f"Outdated signature for '{plugin_id}': signed before the "
                        "allowlist had a key of its own; approve it again"
                    ),
                }
            logger.critical(
                "Plugin '%s' allowlist signature INVALID. "
                "Possible allowlist tampering.",
                plugin_id,
            )
            return _tampered(plugin_id)

        # Verify code hash: a folder that cannot be hashed matches nothing
        current_hash = compute_plugin_hash(plugin_dir)
        if not current_hash:
            logger.critical(
                "Plugin '%s' cannot be hashed: its folder is missing, holds no file, "
                "or holds a symbolic link, a special file or something it cannot read.",
                plugin_id,
            )
            return {
                "allowed": False,
                "reason": (
                    f"Unhashable plugin '{plugin_id}': its folder is missing, holds no "
                    "file, or holds a symbolic link, a special file or something "
                    "unreadable"
                ),
            }
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

        # Verify the HMAC signature under the key that loads, and no other:
        # an entry that cannot be checked is not trusted.
        key, why = _key_or_why()
        if key is None:
            return _unverifiable(plugin_id, why)
        if not _verify_entry_signature(entry, key):
            logger.critical(
                "Plugin '%s' allowlist signature INVALID. "
                "Possible allowlist tampering.",
                plugin_id,
            )
            return _tampered(plugin_id)

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

        The ceremony's password -- the owner's, through the owner's own
        session -- is checked by the API route before this method is called;
        the visual code and the second factor the request carries are not
        read yet.

        Returns {success, entries_added, batch_id}.
        """
        batch = self._pending_batch
        if not batch or batch.batch_id != batch_id:
            return {
                "success": False,
                "error": "no_matching_batch",
                "message": "No pending batch with that ID",
            }

        key, why = _key_or_why()
        if key is None:
            return {
                "success": False,
                "error": "no_signing_key",
                "message": f"Cannot approve without the signing key: {why}",
            }

        # Verify batch hash still matches (no changes since prepare)
        current_hashes = []
        for p in batch.plugins:
            current_hash = compute_plugin_hash(Path(p["plugin_dir"]))
            if not current_hash:
                return {
                    "success": False,
                    "error": "unhashable",
                    "message": (
                        f"Plugin '{p['plugin_id']}' cannot be hashed: its folder is "
                        "missing, holds no file, or holds a symbolic link, a special "
                        "file or something unreadable."
                    ),
                }
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

        # Replace existing entries for these plugins: the file first, the memory after, so that a write that
        # fails approves nothing, in this process either.
        existing_ids = {e.plugin_id for e in new_entries}
        entries = [e for e in self._entries if e.plugin_id not in existing_ids] + new_entries
        try:
            _save_allowlist(entries)
        except OSError as exc:
            logger.error("Plugin allowlist not written: %s", exc)
            return {
                "success": False,
                "error": "not_recorded",
                "message": f"The allowlist could not be written ({type(exc).__name__}): nothing was approved.",
            }
        self._entries = entries
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
