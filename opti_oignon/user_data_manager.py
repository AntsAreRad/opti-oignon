#!/usr/bin/env python3
"""
User data management for Opti-Oignon.

Provides GDPR-ready user data export and cascade deletion:
- Export all user data (conversations, memories, RAG docs, plugin configs,
  preferences) as JSON
- Cascade delete all user artifacts with audit trail
- Per-user RAG collection namespacing helpers
- Per-user plugin configuration scoping

This module orchestrates cross-module data operations without
tightly coupling to individual storage implementations.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for cross-module access
# ---------------------------------------------------------------------------


def _get_conversation_manager() -> Any:
    try:
        from opti_oignon.conversation import conversation_manager
        return conversation_manager
    except ImportError:
        return None


def _get_memory_manager() -> Any:
    try:
        from opti_oignon.memory import memory_manager
        return memory_manager
    except ImportError:
        return None


def _get_canonical_memory_store() -> Any:
    """Two-tier canonical memory store (user-scoped list/clear). UD-03."""
    try:
        from opti_oignon.memory import get_canonical_store
        return get_canonical_store()
    except Exception:
        return None


def _get_vector_memory_store() -> Any:
    """Two-tier vector memory store (user-scoped clear). UD-03."""
    try:
        from opti_oignon.memory import get_vector_store
        return get_vector_store()
    except Exception:
        return None


def _get_rag_store() -> Any:
    try:
        from opti_oignon.rag_store import get_rag_store
        return get_rag_store()
    except ImportError:
        return None


def _get_pending_store() -> Any:
    """The review queue of pending writes: the agent's memory and notes writes waiting for the user."""
    try:
        from opti_oignon.pending_writes import get_pending_store
        return get_pending_store()
    except Exception:
        return None


def _get_user_settings_store() -> Any:
    try:
        from opti_oignon.user_isolation import user_settings_store
        return user_settings_store
    except ImportError:
        return None


def _get_plugin_config_store() -> Any:
    try:
        from opti_oignon.plugin_user_config import get_plugin_user_config_store
        return get_plugin_user_config_store()
    except ImportError:
        return None


def _get_plugin_review_store() -> Any:
    """Plugin review store."""
    try:
        from opti_oignon.plugin_reviews import plugin_review_store
        return plugin_review_store
    except ImportError:
        return None


def _get_user_key_manager() -> Any:
    try:
        from opti_oignon.user_key_manager import get_user_key_manager
        return get_user_key_manager()
    except ImportError:
        return None


def _get_admin_audit() -> Any:
    try:
        from opti_oignon.admin_audit import log_admin_event
        return log_admin_event
    except ImportError:
        return None


def _get_response_cache() -> Any:
    """The response cache: answers kept for a repeated request."""
    try:
        from opti_oignon.response_cache import response_cache
        return response_cache
    except ImportError:
        return None


def _get_semantic_cache() -> Any:
    """The semantic cache: answers kept for a similar request."""
    try:
        from opti_oignon.semantic_cache import semantic_cache
        return semantic_cache
    except ImportError:
        return None


def _get_offline_queue() -> Any:
    """The offline queue: what an older version queued while the engine was unreachable."""
    try:
        from opti_oignon.sync_queue import sync_queue
        return sync_queue
    except ImportError:
        return None


def _single_user() -> bool:
    """Whether the install runs for one user, by the auth manager's own rule -- multi-user when it cannot be read.
    Then every install-wide store is that one user's, and the wipe removes it."""
    try:
        from opti_oignon.auth import auth_manager
    except ImportError:
        return False
    try:
        return auth_manager is not None and auth_manager.single_user_mode is True
    except Exception:  # noqa: BLE001 - unsure is multi-user, which removes nothing install-wide
        return False


def _the_one_user(user_id: str) -> bool:
    """Whether ``user_id`` names a single-user install's one user: the local user, or its one account. A wipe asked
    for any other name removes nothing install-wide: the install's stores are not that name's."""
    if user_id == "local":
        return True
    try:
        from opti_oignon.auth import auth_manager
    except ImportError:
        return False
    try:
        return (auth_manager is not None and auth_manager.count_users() == 1
                and auth_manager.get_user(user_id) is not None)
    except Exception:  # noqa: BLE001 - unsure is not the one user, which removes nothing install-wide
        return False


def _get_run_journal() -> Any:
    """The run journal: the opt-in record of the user's agent runs."""
    try:
        from opti_oignon import run_journal
        return run_journal
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# RAG collection namespacing
# ---------------------------------------------------------------------------

_RAG_USER_PREFIX = "user_"

# Stores that the per-user export/wipe CANNOT cover today,
# either because the store is single-user (CA-04 class, no user_id
# column) or because its data is not user-scoped. Surfaced in results so
# a wipe never silently implies completeness. Shrinks as the scoping
# cycle (FBK-01 and family) lands.
# Completed against the full at-rest inventory:
# projects, conversation branches, learned
# routing, and the plugin-owned data stores were missing from this
# surface; plugin reviews left it the same session (REV-2:
# identity-bound and covered by the cascade below).
WIPE_NOT_COVERED = (
    "conversations (store not user-scoped)",
    "artifacts (stored in conversation metadata)",
    "feedback store (unscoped, FBK-01)",
    "performance metrics store",
    "benchmark stores (history, results)",
    "response/semantic caches (conversation-scoped)",
    "coding history store",
    "humanizer feedback store",
    "fine-tune variants registry",
    "telemetry/analytics stores",
    "projects store (single-user)",
    "conversation branches store (single-user)",
    "learned routing store (usage-derived)",
    "plugin-owned data stores (plugin-local DBs)",
    "the onion's store (keyed by conversation)",
    "offline queue (single-user)",
    "old memory store (single-user, emptied at boot once migrated)",
    "sandbox audit's words (single-user; the record of what ran is retained)",
    "notes and their updates",
    "sync: paired devices, skills a peer offered, and every record's latest version in the change feed",
    "the garden (allium: its own wipe)",
    "knowledge-base ingest jobs",
    "context ledger (what each window held, counted)",
    "agent evaluation stores",
    "the account itself and its second factor (deleting it is its own act)",
)

# The stores above the wipe removes in single-user mode, where every install-wide store is the one user's: every
# conversation deleted through its deletion -- which forgets its branches, artifacts, the onion's copy, its cached
# answers, its sandbox audit's words, its versions in the change feed -- and the stores of the user's requests
# emptied. Named as not covered otherwise.
WIPE_SINGLE_USER = (
    "conversations (store not user-scoped)",
    "artifacts (stored in conversation metadata)",
    "response/semantic caches (conversation-scoped)",
    "conversation branches store (single-user)",
    "the onion's store (keyed by conversation)",
    "offline queue (single-user)",
    "old memory store (single-user, emptied at boot once migrated)",
    "sandbox audit's words (single-user; the record of what ran is retained)",
)

# Stores the export leaves out though the wipe covers them. The run journal
# keeps the user's typed words sealed under the retention they consented to:
# an export would be a copy outside both. The wipe removes it whole.
EXPORT_NOT_COVERED = (
    "run journal (opt-in record of agent runs; wiped with the user's data, never exported)",
)

# Stores deliberately RETAINED on a per-user wipe. Audit
# trails are tamper-evident accountability records; erasing them on
# request would defeat their purpose. Surfaced alongside not_covered so
# the wipe result is honest about what survives by design (a known GDPR
# tension).
WIPE_RETAINED_BY_DESIGN = (
    "admin audit log (accountability trail)",
    "audit chain (tamper-evident security trail)",
    "signed audit logs (RAG injection)",
    "sandbox audit: the record of what ran, without its words",
)


def user_collection_name(user_id: str, collection_name: str) -> str:
    """Generate a user-scoped RAG collection name.

    Format: user_{user_id}_{collection_name}

    Args:
        user_id: User identifier.
        collection_name: Base collection name.

    Returns:
        Namespaced collection name.
    """
    return f"{_RAG_USER_PREFIX}{user_id}_{collection_name}"


def is_user_collection(user_id: str, collection_name: str) -> bool:
    """Check if a collection belongs to a user.

    The prefix includes the trailing separator (``user_{id}_``) to match the
    form produced by ``user_collection_name`` and used by
    ``get_user_collections``/``strip_user_prefix``. Without it, user_id "1"
    would match "10"/"11"/"1abc" collections, leaking across users in
    export and cascade-delete. (A user_id containing "_" can still collide;
    that deeper case is tracked as a hardening item.)
    """
    prefix = f"{_RAG_USER_PREFIX}{user_id}_"
    return collection_name.startswith(prefix)


def get_user_collections(user_id: str, all_collections: list[str]) -> list[str]:
    """Filter collections belonging to a specific user."""
    prefix = f"{_RAG_USER_PREFIX}{user_id}_"
    return [c for c in all_collections if c.startswith(prefix)]


def strip_user_prefix(user_id: str, collection_name: str) -> str:
    """Remove the user prefix from a collection name for display."""
    prefix = f"{_RAG_USER_PREFIX}{user_id}_"
    if collection_name.startswith(prefix):
        return collection_name[len(prefix):]
    return collection_name


# ---------------------------------------------------------------------------
# User data export (GDPR)
# ---------------------------------------------------------------------------


class UserDataExporter:
    """Exports all data for a specific user as a JSON-serializable dict.

    Collects data from all user-scoped storage modules:
    - Conversations (with messages)
    - Memories
    - RAG collections and documents
    - Plugin configurations
    - User settings/preferences
    """

    def export(self, user_id: str) -> dict[str, Any]:
        """Export all data for a user.

        Args:
            user_id: The user whose data to export.

        Returns:
            Dict containing all user data, suitable for JSON serialization.
        """
        export_time = time.time()
        data: dict[str, Any] = {
            "export_metadata": {
                "user_id": user_id,
                "exported_at": export_time,
                # 1.1: adds plugin_reviews and retained_by_design.
                # 1.2: adds pending_writes.
                "format_version": "1.2",
                "not_covered": [*WIPE_NOT_COVERED, *EXPORT_NOT_COVERED],
                # Audit trails survive a wipe by design.
                "retained_by_design": list(WIPE_RETAINED_BY_DESIGN),
            },
            "conversations": self._export_conversations(user_id),
            "memories": self._export_memories(user_id),
            "rag_collections": self._export_rag(user_id),
            "plugin_configs": self._export_plugin_configs(user_id),
            "plugin_reviews": self._export_plugin_reviews(user_id),
            "settings": self._export_settings(user_id),
            "pending_writes": self._export_pending_writes(user_id),
        }
        logger.info(
            "Exported data for user %s: %d conversations, %d memories",
            user_id,
            len(data["conversations"]),
            len(data["memories"]),
        )
        return data

    def _export_pending_writes(self, user_id: str) -> list[dict[str, Any]]:
        """Export the user's pending writes, whatever their decision: their words are the user's data."""
        store = _get_pending_store()
        if store is None:
            return []
        try:
            return [record.to_dict() for record in store.list(status=None, user_id=user_id, limit=None)]
        except Exception as e:
            logger.warning("Pending writes export failed for %s: %s", user_id, e)
            return []

    def _export_conversations(self, user_id: str) -> list[dict[str, Any]]:
        """Export user conversations.

        The conversation store is single-user today (no
        user_id column); probing it with a user_id kwarg raised a
        swallowed TypeError that masqueraded as "no data". The kwarg is
        now checked explicitly and the unscoped case is an explicit,
        logged skip (forward-compatible: works the day the store grows
        a user_id parameter).
        """
        mgr = _get_conversation_manager()
        if mgr is None:
            return []
        try:
            if hasattr(mgr, "list_conversations"):
                import inspect
                params = inspect.signature(mgr.list_conversations).parameters
                if "user_id" not in params:
                    logger.debug(
                        "Conversation store is not user-scoped; "
                        "conversations excluded from export (see not_covered)"
                    )
                    return []
                convs = mgr.list_conversations(user_id=user_id)
                if isinstance(convs, list):
                    return [
                        c.to_dict() if hasattr(c, "to_dict") else c
                        for c in convs
                    ]
            return []
        except Exception as e:
            logger.warning("Failed to export conversations for %s: %s", user_id, e)
            return []

    def _export_memories(self, user_id: str) -> list[dict[str, Any]]:
        """Export user memories.

        The two-tier canonical store carries the
        user-scoped data and a `list(user_id=...)` API; the legacy
        facade (whose API never matched the probes below) is kept as a
        fallback only.
        """
        canonical = _get_canonical_memory_store()
        if canonical is not None:
            try:
                records = canonical.list(user_id=user_id, active_only=False)
                return [
                    r.to_dict() if hasattr(r, "to_dict") else r
                    for r in records
                ]
            except Exception as e:
                logger.warning(
                    "Canonical memory export failed for %s: %s", user_id, e
                )

        mgr = _get_memory_manager()
        if mgr is None:
            return []
        try:
            if hasattr(mgr, "list_memories"):
                mems = mgr.list_memories(user_id=user_id)
                if isinstance(mems, list):
                    return [
                        m.to_dict() if hasattr(m, "to_dict") else m
                        for m in mems
                    ]
            if hasattr(mgr, "get_all"):
                mems = mgr.get_all(user_id=user_id)
                if isinstance(mems, list):
                    return mems
            return []
        except Exception as e:
            logger.warning("Failed to export memories for %s: %s", user_id, e)
            return []

    def _export_rag(self, user_id: str) -> list[dict[str, Any]]:
        """Export user RAG collections and document metadata."""
        store = _get_rag_store()
        if store is None:
            return []
        try:
            collections = []
            if hasattr(store, "list_collections"):
                all_colls = store.list_collections()
                for coll in all_colls:
                    name = coll.get("name", "") if isinstance(coll, dict) else str(coll)
                    if is_user_collection(user_id, name):
                        collections.append({
                            "name": strip_user_prefix(user_id, name),
                            "internal_name": name,
                        })
            return collections
        except Exception as e:
            logger.warning("Failed to export RAG for %s: %s", user_id, e)
            return []

    def _export_plugin_configs(self, user_id: str) -> list[dict[str, Any]]:
        """Export user plugin configurations."""
        store = _get_plugin_config_store()
        if store is None:
            return []
        try:
            if hasattr(store, "get_all_configs"):
                return store.get_all_configs(user_id=user_id)
            return []
        except Exception as e:
            logger.warning("Failed to export plugin configs for %s: %s", user_id, e)
            return []

    def _export_plugin_reviews(self, user_id: str) -> list[dict[str, Any]]:
        """Export the user's plugin reviews."""
        store = _get_plugin_review_store()
        if store is None:
            return []
        try:
            if hasattr(store, "get_reviews_for_user"):
                return [
                    r.to_dict() for r in store.get_reviews_for_user(user_id)
                ]
            return []
        except Exception as e:
            logger.warning(
                "Failed to export plugin reviews for %s: %s", user_id, e
            )
            return []

    def _export_settings(self, user_id: str) -> dict[str, Any]:
        """Export user settings/preferences."""
        store = _get_user_settings_store()
        if store is None:
            return {}
        try:
            settings = store.get_settings(user_id)
            return settings.to_dict() if hasattr(settings, "to_dict") else {}
        except Exception as e:
            logger.warning("Failed to export settings for %s: %s", user_id, e)
            return {}


# ---------------------------------------------------------------------------
# User data deletion (cascade)
# ---------------------------------------------------------------------------


class UserDataDeleter:
    """Cascade-deletes all data for a specific user.

    Removes data from all user-scoped storage modules and logs
    the deletion in the admin audit trail.
    """

    def delete_all(
        self,
        user_id: str,
        admin_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete all data for a user.

        Args:
            user_id: The user whose data to delete.
            admin_id: Optional admin who initiated the deletion.

        Returns:
            Dict summarizing what was deleted.
        """
        single = _single_user() and _the_one_user(user_id)
        journal = self._delete_run_journal()
        conversations, left = self._delete_conversations(user_id, single=single)
        results: dict[str, Any] = {
            "user_id": user_id,
            "deleted_at": time.time(),
            "conversations": conversations,
            "memories": self._delete_memories(user_id),
            "rag_collections": self._delete_rag(user_id),
            "plugin_configs": self._delete_plugin_configs(user_id),
            "plugin_reviews": self._delete_plugin_reviews(user_id),
            "settings": self._delete_settings(user_id),
            "encryption_keys": self._delete_encryption_keys(user_id),
            "pending_writes": self._delete_pending_writes(user_id),
            "run_journal": journal,
            # Stores the per-user wipe cannot cover today.
            "not_covered": list(WIPE_NOT_COVERED),
            # Audit trails survive the wipe by design.
            "retained_by_design": list(WIPE_RETAINED_BY_DESIGN),
        }
        # A journal the wipe could not remove is said where the answer is
        # read: the route's response keeps not_covered, not this key.
        if not journal.get("wiped"):
            results["not_covered"].append(f"run journal (not wiped: {journal.get('reason')})")
        if single:
            # Single-user mode: the install-wide stores are the one user's, and those the wipe removed leave the
            # list; each that could not be is named with why.
            failed = [*left, *self._empty_requests(), *self._forget_sandbox_words()]
            old = _empty_old_store()
            if old is not None:
                failed.append(old)
            results["not_covered"] = [label for label in results["not_covered"]
                                      if label not in WIPE_SINGLE_USER] + failed

        # Log to admin audit
        audit_fn = _get_admin_audit()
        if audit_fn and admin_id:
            audit_fn(
                admin_id=admin_id,
                action="delete_user_data",
                target_type="user",
                target_id=user_id,
                details=json.dumps(results),
            )

        logger.info("Deleted all data for user %s", user_id)
        return results

    def _delete_pending_writes(self, user_id: str) -> int:
        """Delete the user's pending writes, so a proposal left waiting cannot bring wiped words back."""
        store = _get_pending_store()
        if store is None:
            return 0
        try:
            return store.delete_user(user_id)
        except Exception as e:
            logger.warning("Pending writes delete failed for %s: %s", user_id, e)
            return 0

    def _delete_run_journal(self) -> dict[str, Any]:
        """Wipe the run journal whole: it is single-user, so every user's wipe removes it."""
        journal = _get_run_journal()
        if journal is None:
            return {"wiped": False, "reason": "unavailable"}
        try:
            return journal.wipe()
        except Exception as e:
            logger.warning("Run journal wipe failed: %s", e)
            return {"wiped": False, "reason": type(e).__name__}

    def _delete_conversations(self, user_id: str, *, single: bool = False) -> tuple[int, list[str]]:
        """Delete user conversations: how many, and what stayed, said.

        The conversation store is single-user today. In single-user mode its
        conversations are the one user's: every one is deleted through its
        deletion, which forgets what every store drew from it; a refusal stops
        and says how many are left. Otherwise the unscoped case is an
        explicit, logged skip instead of a swallowed TypeError
        (forward-compatible with a future user_id parameter).
        """
        mgr = _get_conversation_manager()
        if mgr is None:
            return 0, (["conversations (the conversation store is not available)"] if single else [])
        if single:
            try:
                from opti_oignon.conversation_wipe import purge_every_conversation

                purged, left, refused = purge_every_conversation(mgr)
            except Exception as e:  # noqa: BLE001 - what stays is said
                logger.warning("Failed to delete conversations for %s: %s", user_id, e)
                return 0, [f"conversations (not deleted: {type(e).__name__})"]
            return len(purged), ([f"conversations ({left} left: a deletion was refused, {refused})"]
                                 if refused is not None else [])
        return self._delete_scoped_conversations(user_id), []

    def _delete_scoped_conversations(self, user_id: str) -> int:
        """Delete the conversations of a user-scoped store; 0 when the store is not user-scoped."""
        mgr = _get_conversation_manager()
        if mgr is None:
            return 0
        try:
            if hasattr(mgr, "delete_user_conversations"):
                return mgr.delete_user_conversations(user_id=user_id)
            # Fallback: list and delete one by one
            if hasattr(mgr, "list_conversations") and hasattr(mgr, "delete_conversation"):
                import inspect
                params = inspect.signature(mgr.list_conversations).parameters
                if "user_id" not in params:
                    logger.debug(
                        "Conversation store is not user-scoped; "
                        "conversations excluded from wipe (see not_covered)"
                    )
                    return 0
                convs = mgr.list_conversations(user_id=user_id)
                count = 0
                for c in (convs or []):
                    cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
                    if cid and mgr.delete_conversation(cid):
                        count += 1
                return count
            return 0
        except Exception as e:
            logger.warning("Failed to delete conversations for %s: %s", user_id, e)
            return 0

    def _delete_memories(self, user_id: str) -> int:
        """Delete user memories.

        Clears the two-tier stores (canonical SQLite +
        vector layer), both user-scoped, the facts' log emptied after. The
        legacy facade probes (whose API never matched) remain as a fallback
        only.
        """
        canonical = _get_canonical_memory_store()
        if canonical is not None:
            try:
                count = canonical.clear(user_id=user_id)
                empty_log = getattr(canonical, "empty_log", None)
                if callable(empty_log):
                    empty_log("the user's facts wiped")
                vec = _get_vector_memory_store()
                if vec is not None:
                    try:
                        vec.clear(user_id=user_id)
                    except Exception as e:
                        logger.warning(
                            "Vector memory clear failed for %s: %s", user_id, e
                        )
                return count
            except Exception as e:
                logger.warning(
                    "Canonical memory clear failed for %s: %s", user_id, e
                )

        mgr = _get_memory_manager()
        if mgr is None:
            return 0
        try:
            if hasattr(mgr, "delete_user_memories"):
                return mgr.delete_user_memories(user_id=user_id)
            if hasattr(mgr, "clear"):
                mgr.clear(user_id=user_id)
                return 1  # Can't know exact count
            return 0
        except Exception as e:
            logger.warning("Failed to delete memories for %s: %s", user_id, e)
            return 0

    def _delete_rag(self, user_id: str) -> int:
        """Delete user RAG collections."""
        store = _get_rag_store()
        if store is None:
            return 0
        try:
            count = 0
            if hasattr(store, "list_collections") and hasattr(store, "delete_collection"):
                all_colls = store.list_collections()
                for coll in all_colls:
                    name = coll.get("name", "") if isinstance(coll, dict) else str(coll)
                    if is_user_collection(user_id, name):
                        store.delete_collection(name)
                        count += 1
            return count
        except Exception as e:
            logger.warning("Failed to delete RAG for %s: %s", user_id, e)
            return 0

    @staticmethod
    def _empty_requests() -> list[str]:
        """Single-user mode: the stores of the user's requests emptied whole, overwritten -- the response cache, the
        semantic cache, the offline queue; each that could not be, named. A store whose file is not there holds none,
        and is not created."""
        failed = []
        for label, held in (("response cache", _get_response_cache), ("semantic cache", _get_semantic_cache),
                            ("offline queue", _get_offline_queue)):
            try:
                cache = held()
                if cache is not None:
                    # Each raises when its file is there and cannot be emptied: never taken for an empty one.
                    cache.empty()
            except Exception as e:  # noqa: BLE001 - what stays is said
                logger.warning("The %s could not be emptied: %s", label, e)
                failed.append(f"{label} (not emptied: {type(e).__name__})")
        return failed

    @staticmethod
    def _forget_sandbox_words() -> list[str]:
        """Single-user mode: the words of every row of the sandbox audit forgotten, the record of what ran kept, by
        a process that loaded the sandbox manager (its log, or the one its settings name with the sandbox off). A
        process that did not is never made to load it: its construction reaps every workspace it does not track."""
        held = sys.modules.get("opti_oignon.sandbox_manager")
        if held is None:
            return ["sandbox audit's words (this process did not load the sandbox)"]
        try:
            held.forget_every_word()
        except Exception as e:  # noqa: BLE001 - what stays is said
            logger.warning("The sandbox audit's words could not be forgotten: %s", e)
            return [f"sandbox audit's words (not forgotten: {type(e).__name__})"]
        return []

    def _delete_plugin_configs(self, user_id: str) -> int:
        """Delete user plugin configurations."""
        store = _get_plugin_config_store()
        if store is None:
            return 0
        try:
            if hasattr(store, "delete_all_configs"):
                return store.delete_all_configs(user_id=user_id)
            return 0
        except Exception as e:
            logger.warning("Failed to delete plugin configs for %s: %s", user_id, e)
            return 0

    def _delete_plugin_reviews(self, user_id: str) -> int:
        """Delete the user's plugin reviews.

        Legacy rows with a NULL user_id are unattributable and stay
        untouched by construction (the store's equality predicate).
        """
        store = _get_plugin_review_store()
        if store is None:
            return 0
        try:
            if hasattr(store, "delete_reviews_for_user"):
                return store.delete_reviews_for_user(user_id)
            return 0
        except Exception as e:
            logger.warning(
                "Failed to delete plugin reviews for %s: %s", user_id, e
            )
            return 0

    def _delete_settings(self, user_id: str) -> bool:
        """Delete user settings."""
        store = _get_user_settings_store()
        if store is None:
            return False
        try:
            return store.delete_settings(user_id)
        except Exception as e:
            logger.warning("Failed to delete settings for %s: %s", user_id, e)
            return False

    def _delete_encryption_keys(self, user_id: str) -> bool:
        """Delete user encryption keys and salts."""
        mgr = _get_user_key_manager()
        if mgr is None:
            return False
        try:
            return mgr.delete_user_keys(user_id)
        except Exception as e:
            logger.warning("Failed to delete encryption keys for %s: %s", user_id, e)
            return False


def _empty_old_store() -> str | None:
    """Single-user mode: the old memory store emptied whole, overwritten, its log emptied -- its memories were moved
    into the facts, and what an older version left there is the one user's. None when it is empty, else why not. A
    tree without it holds none."""
    try:
        from opti_oignon.memory.legacy import memory_manager
    except ImportError:
        return None
    path = getattr(memory_manager, "db_path", None)
    if memory_manager is None or path is None or not Path(path).exists():
        return None
    try:
        if memory_manager.held():
            memory_manager.clear_all()
            left = len(memory_manager.held())
            if left:
                return f"old memory store ({left} row(s) not deleted)"
        memory_manager.empty_log()
    except Exception as e:  # noqa: BLE001 - what stays is said
        logger.warning("The old memory store could not be emptied: %s", e)
        return f"old memory store (not emptied: {type(e).__name__})"
    return None


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_exporter: UserDataExporter | None = None
_deleter: UserDataDeleter | None = None


def get_user_data_exporter() -> UserDataExporter:
    """Get or create the singleton UserDataExporter."""
    global _exporter
    if _exporter is None:
        _exporter = UserDataExporter()
    return _exporter


def get_user_data_deleter() -> UserDataDeleter:
    """Get or create the singleton UserDataDeleter."""
    global _deleter
    if _deleter is None:
        _deleter = UserDataDeleter()
    return _deleter


USER_DATA_MANAGER_AVAILABLE = True
