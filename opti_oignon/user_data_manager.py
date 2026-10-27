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
    """Two-tier canonical memory store (user-scoped list/clear). UD-03. A store that cannot be opened raises: its
    caller names it."""
    try:
        from opti_oignon.memory import get_canonical_store
    except ImportError:
        return None
    return get_canonical_store()


def _get_vector_memory_store() -> Any:
    """Two-tier vector memory store (user-scoped clear). UD-03. A store that cannot be opened raises: its caller
    names it."""
    try:
        from opti_oignon.memory import get_vector_store
    except ImportError:
        return None
    return get_vector_store()


def _get_rag_store() -> Any:
    """The knowledge base's store when there is one -- built by this process, or its file on disk -- never built to be
    asked: building it makes its folder, its files and its vector client."""
    try:
        from opti_oignon.rag_store import existing_rag_store
        return existing_rag_store()
    except ImportError:
        return None


def _get_pending_store() -> Any:
    """The review queue of pending writes: the agent's memory and notes writes waiting for the user. A queue that
    cannot be opened raises: its caller names it."""
    try:
        from opti_oignon.pending_writes import get_pending_store
    except ImportError:
        return None
    return get_pending_store()


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


def _get_decision_preferences() -> Any:
    """The coding agent's decision preferences: each approve, modify or abort, with its context."""
    try:
        from opti_oignon.session_fingerprint import empty_preferences
        return empty_preferences
    except ImportError:
        return None


def _get_humanizer_feedback() -> Any:
    """The humanizer's feedback: the texts it compared, and how the user rated them."""
    try:
        from opti_oignon.humanizer import empty_feedback
        return empty_feedback
    except ImportError:
        return None


def _get_learned_router() -> Any:
    """The learned router: the queries it kept to learn from, and the model learned from them."""
    try:
        from opti_oignon.learned_router import empty
        return empty
    except ImportError:
        return None


def _get_command_line_history() -> Any:
    """The command line's history: each question asked there, and its answer."""
    try:
        from opti_oignon.history import forget_every_entry
        return forget_every_entry
    except ImportError:
        return None


def _get_feedback() -> Any:
    """The feedback store: each rating and its words, a conversation's or none."""
    try:
        from opti_oignon.feedback import empty
        return empty
    except ImportError:
        return None


def _get_knowledge_base_searches() -> Any:
    """The knowledge base's search log: each query, word for word, and what it cited."""
    try:
        from opti_oignon.rag_store import forget_every_query
        return forget_every_query
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


def _failed(failed: list[str] | None, label: str, error: BaseException, done: str = "deleted") -> None:
    """Name a store the wipe removes for the user that could not be removed (or its log, emptied): added to
    ``failed``, which the answer's ``not_covered`` carries."""
    if failed is not None:
        failed.append(f"{label} (not {done}: {type(error).__name__})")


def _guarded(failed: list[str], label: str, step: Any, default: Any) -> Any:
    """One step of the wipe: what it answers, or, when it raises -- a store that cannot be opened, say -- its default,
    the step named in ``failed``: no step stops the ones after it."""
    try:
        return step()
    except Exception as e:  # noqa: BLE001 - what stays is said
        logger.warning("The wipe's step for the %s failed: %s", label, e)
        _failed(failed, label, e)
        return default


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

# Every store the package keeps is classified in ``store_inventory`` (what it holds, what the wipe does with it),
# which a census keeps complete against the code: the three lists below are its derivation, in its order, and a
# contract holds them equal to it. They are written out here so that loading the wipe loads nothing else.
#
# Stores a wipe does not remove for the user it is asked for: named in the answer, so it never implies more than it
# did. In single-user mode, for the install's one user, those of the next list leave it.
WIPE_NOT_COVERED = (
    "conversations (store not user-scoped)",
    "artifacts (stored in conversation metadata)",
    "conversation branches store (single-user)",
    "the onion's store (keyed by conversation)",
    "response/semantic caches (conversation-scoped)",
    "offline queue (single-user)",
    "sandbox audit's words (single-user; the record of what ran is retained)",
    "feedback store (unscoped, FBK-01)",
    "old memory store (single-user, emptied at boot once migrated)",
    "the coding agent's decision preferences (not user-scoped)",
    "humanizer feedback store",
    "learned routing store (the queries, word for word, and the model learned from them)",
    "the command line's history (questions and answers, not user-scoped)",
    "the knowledge base's search log (the queries, not user-scoped)",
    "notes and their updates",
    "note attachments (voice, images)",
    "projects store (single-user)",
    "project files and saved outputs",
    "the knowledge base: documents, their chunks and vectors (not user-scoped)",
    "the knowledge base's index cache (the documents' paths and hashes)",
    "project indexes (chunks of the project files)",
    "the knowledge base's injection audit (the flagged chunks' text)",
    "knowledge-base ingest jobs",
    "legacy vector folders (the documents' chunks in plain text until `db_encryption migrate`)",
    "coding history store",
    "code execution outputs (the files a conversation's code wrote, kept by no conversation)",
    "sandbox exports (the files copied out of a sandbox)",
    "skills (written by the user or accepted from the agent)",
    "the garden (allium; no wipe of its own yet)",
    "plugin-owned data stores (plugin-local DBs)",
    "plugin logs (not checked for words)",
    "settings files the user writes (presets, custom pipelines, benchmark profiles, their backups)",
    "sync: paired devices, skills a peer offered, and every record's latest version in the change feed",
    "the onion's counters (counts only)",
    "context ledger (what each window held, counted)",
    "performance metrics store",
    "telemetry/analytics stores",
    "benchmark stores (history, results)",
    "fine-tune variants registry",
    "agent evaluation stores",
    "the account itself and its second factor (deleting it is its own act)",
)

# The stores above the wipe removes in single-user mode, for the install's one user, where every install-wide store
# is theirs: every conversation deleted through its deletion -- which forgets its branches, artifacts, the onion's
# copy, its cached answers, its sandbox audit's words, its versions in the change feed -- and the stores that only
# keep copies of the user's words emptied. Named as not covered otherwise.
WIPE_SINGLE_USER = (
    "conversations (store not user-scoped)",
    "artifacts (stored in conversation metadata)",
    "conversation branches store (single-user)",
    "the onion's store (keyed by conversation)",
    "response/semantic caches (conversation-scoped)",
    "offline queue (single-user)",
    "sandbox audit's words (single-user; the record of what ran is retained)",
    "feedback store (unscoped, FBK-01)",
    "old memory store (single-user, emptied at boot once migrated)",
    "the coding agent's decision preferences (not user-scoped)",
    "humanizer feedback store",
    "learned routing store (the queries, word for word, and the model learned from them)",
    "the command line's history (questions and answers, not user-scoped)",
    "the knowledge base's search log (the queries, not user-scoped)",
)

# Stores the export leaves out though the wipe covers them. The run journal
# keeps the user's typed words sealed under the retention they consented to:
# an export would be a copy outside both. The wipe removes it whole.
EXPORT_NOT_COVERED = (
    "run journal (opt-in record of agent runs; wiped with the user's data, never exported)",
)

# Stores deliberately RETAINED on a per-user wipe. Audit trails are accountability records; erasing them on request
# would defeat their purpose. Surfaced alongside not_covered so the wipe result is honest about what survives by
# design (a known GDPR tension).
WIPE_RETAINED_BY_DESIGN = (
    "sandbox audit: the record of what ran, without its words",
    "admin audit log (accountability trail)",
    "audit chain (tamper-evident security trail)",
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
        try:
            store = _get_pending_store()
            if store is None:
                return []
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
        try:
            canonical = _get_canonical_memory_store()
        except Exception as e:
            logger.warning("Canonical memory store could not be opened for %s: %s", user_id, e)
            canonical = None
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
        try:
            store = _get_rag_store()
            if store is None:
                return []
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
        try:
            store = _get_plugin_config_store()
            if store is None:
                return []
            if hasattr(store, "get_all_configs"):
                return store.get_all_configs(user_id=user_id)
            return []
        except Exception as e:
            logger.warning("Failed to export plugin configs for %s: %s", user_id, e)
            return []

    def _export_plugin_reviews(self, user_id: str) -> list[dict[str, Any]]:
        """Export the user's plugin reviews."""
        try:
            store = _get_plugin_review_store()
            if store is None:
                return []
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
        try:
            store = _get_user_settings_store()
            if store is None:
                return {}
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
        # The stores removed for the user that could not be, each named in the answer with why; each step guarded,
        # so a store that cannot be opened stops none of the others.
        failed: list[str] = []
        journal = self._delete_run_journal()
        conversations, left = _guarded(failed, "conversations",
                                       lambda: self._delete_conversations(user_id, single=single), (0, []))
        results: dict[str, Any] = {
            "user_id": user_id,
            "deleted_at": time.time(),
            "conversations": conversations,
            "memories": _guarded(failed, "facts", lambda: self._delete_memories(user_id, failed=failed), 0),
            "rag_collections": _guarded(failed, "knowledge base collections named for the user",
                                        lambda: self._delete_rag(user_id, failed=failed), 0),
            "plugin_configs": _guarded(failed, "plugin settings",
                                       lambda: self._delete_plugin_configs(user_id, failed=failed), 0),
            "plugin_reviews": _guarded(failed, "plugin reviews",
                                       lambda: self._delete_plugin_reviews(user_id, failed=failed), 0),
            "settings": _guarded(failed, "settings", lambda: self._delete_settings(user_id, failed=failed), False),
            "encryption_keys": _guarded(failed, "encryption key salts",
                                        lambda: self._delete_encryption_keys(user_id, failed=failed), False),
            "pending_writes": _guarded(failed, "pending writes",
                                       lambda: self._delete_pending_writes(user_id, failed=failed), 0),
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
            failed_here = [*left, *self._empty_requests(), *self._empty_copies(), *self._forget_sandbox_words()]
            old = _empty_old_store()
            if old is not None:
                failed_here.append(old)
            results["not_covered"] = [label for label in results["not_covered"]
                                      if label not in WIPE_SINGLE_USER] + failed_here
        results["not_covered"] += failed

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

    def _delete_pending_writes(self, user_id: str, *, failed: list[str] | None = None) -> int:
        """Delete the user's pending writes, so a proposal left waiting cannot bring wiped words back; a failure is
        added to ``failed``."""
        try:
            store = _get_pending_store()
            if store is None:
                return 0
            return store.delete_user(user_id)
        except Exception as e:
            logger.warning("Pending writes delete failed for %s: %s", user_id, e)
            _failed(failed, "pending writes", e)
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

    def _delete_memories(self, user_id: str, *, failed: list[str] | None = None) -> int:
        """Delete user memories.

        Clears the two-tier stores (canonical SQLite +
        vector layer), both user-scoped, the facts' log emptied after. The
        legacy facade probes (whose API never matched) remain as a fallback
        only. A failure is added to ``failed``.
        """
        try:
            canonical = _get_canonical_memory_store()
        except Exception as e:
            # A store there that cannot be opened is named, its vectors cleared all the same; the legacy facade is
            # for a tree without one.
            logger.warning("Canonical memory store could not be opened for %s: %s", user_id, e)
            _failed(failed, "facts", e)
            self._clear_vectors(user_id, failed)
            return 0
        if canonical is not None:
            # Three steps, each named when it fails: the facts, their log, their vectors.
            count = 0
            try:
                count = canonical.clear(user_id=user_id)
            except Exception as e:
                logger.warning("Canonical memory clear failed for %s: %s", user_id, e)
                _failed(failed, "facts", e)
            else:
                empty_log = getattr(canonical, "empty_log", None)
                if callable(empty_log):
                    try:
                        empty_log("the user's facts wiped")
                    except Exception as e:
                        logger.warning("The facts' log could not be emptied for %s: %s", user_id, e)
                        _failed(failed, "facts' log", e, "emptied")
            self._clear_vectors(user_id, failed)
            return count

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
            _failed(failed, "facts", e)
            return 0

    @staticmethod
    def _clear_vectors(user_id: str, failed: list[str] | None) -> None:
        """The facts' vectors cleared for the user; a failure is added to ``failed``."""
        try:
            vec = _get_vector_memory_store()
            if vec is not None:
                vec.clear(user_id=user_id)
        except Exception as e:
            logger.warning("Vector memory clear failed for %s: %s", user_id, e)
            _failed(failed, "facts' vectors", e)

    def _delete_rag(self, user_id: str, *, failed: list[str] | None = None) -> int:
        """Delete user RAG collections -- those named for the user, which nothing creates; the knowledge base itself is
        named in ``not_covered`` -- from a store there is, never built to be asked; a failure is added to ``failed``."""
        try:
            store = _get_rag_store()
        except Exception as e:
            logger.warning("Failed to open the knowledge base for %s: %s", user_id, e)
            _failed(failed, "knowledge base collections named for the user", e)
            return 0
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
            _failed(failed, "knowledge base collections named for the user", e)
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
    def _empty_copies() -> list[str]:
        """Single-user mode: the stores that only keep copies of the user's words emptied whole, overwritten -- the
        coding agent's decision preferences, the humanizer's feedback, the learned router's queries and its model,
        the command line's history, the feedback store, the knowledge base's search log; each that could not be,
        named. A store whose file is not there holds none, and is not created; one whose module does not load is
        not asked."""
        failed = []
        for label, held in (("decision preferences", _get_decision_preferences),
                            ("humanizer feedback", _get_humanizer_feedback),
                            ("learned routing store", _get_learned_router),
                            ("command line history", _get_command_line_history),
                            ("feedback store", _get_feedback),
                            ("knowledge base search log", _get_knowledge_base_searches)):
            try:
                empty = held()
                if empty is not None:
                    # Each raises when its file is there and cannot be emptied: never taken for an empty one.
                    empty()
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

    def _delete_plugin_configs(self, user_id: str, *, failed: list[str] | None = None) -> int:
        """Delete user plugin configurations; a failure is added to ``failed``."""
        try:
            store = _get_plugin_config_store()
            if store is None:
                return 0
            if hasattr(store, "delete_all_configs"):
                return store.delete_all_configs(user_id=user_id)
            return 0
        except Exception as e:
            logger.warning("Failed to delete plugin configs for %s: %s", user_id, e)
            _failed(failed, "plugin settings", e)
            return 0

    def _delete_plugin_reviews(self, user_id: str, *, failed: list[str] | None = None) -> int:
        """Delete the user's plugin reviews; a failure is added to ``failed``.

        Legacy rows with a NULL user_id are unattributable and stay
        untouched by construction (the store's equality predicate).
        """
        try:
            store = _get_plugin_review_store()
            if store is None:
                return 0
            if hasattr(store, "delete_reviews_for_user"):
                return store.delete_reviews_for_user(user_id)
            return 0
        except Exception as e:
            logger.warning(
                "Failed to delete plugin reviews for %s: %s", user_id, e
            )
            _failed(failed, "plugin reviews", e)
            return 0

    def _delete_settings(self, user_id: str, *, failed: list[str] | None = None) -> bool:
        """Delete user settings; a failure is added to ``failed``."""
        try:
            store = _get_user_settings_store()
            if store is None:
                return False
            return store.delete_settings(user_id)
        except Exception as e:
            logger.warning("Failed to delete settings for %s: %s", user_id, e)
            _failed(failed, "settings", e)
            return False

    def _delete_encryption_keys(self, user_id: str, *, failed: list[str] | None = None) -> bool:
        """Delete user encryption keys and salts; a failure is added to ``failed``."""
        try:
            mgr = _get_user_key_manager()
            if mgr is None:
                return False
            return mgr.delete_user_keys(user_id)
        except Exception as e:
            logger.warning("Failed to delete encryption keys for %s: %s", user_id, e)
            _failed(failed, "encryption key salts", e)
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
