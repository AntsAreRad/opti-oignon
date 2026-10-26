"""Deduplication and coordinated CRUD for Opti-Oignon memory.

Two responsibilities:

1. Double deduplication. On every add, two stages run in order: a Jaccard text
   similarity check at threshold 0.6, then a vector near-duplicate check via the
   vector layer's ``find_similar`` at cosine 0.92. A candidate that trips either
   check is merged into the existing fact rather than duplicated. The text stage
   runs first because it is cheap and needs no embedding.

2. Coordinated CRUD. ``MemoryStore`` wraps the canonical store and the vector
   layer and keeps them consistent: an insert writes to both, a merge reinforces
   the existing fact, an update propagates text and metadata to both, and a
   delete is mirrored (soft delete also drops the vector entry so it stops being
   surfaced, while the canonical row is retained for restore).

The module imports no sibling at module scope (the stores are injected), so it
loads and tests in isolation without ollama or fastapi.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Module sentinels (project convention).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

JACCARD_THRESHOLD = 0.6
COSINE_THRESHOLD = 0.92

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# The sources a fact the model drew from a conversation is written under: the extraction the user asks of one, and
# the capture that runs as one grows. Each names its conversation after the colon, whole, and the conversation's
# deletion erases the fact. Any other source -- written by hand, asked of the agent, a proposal accepted, the old
# store's, whose origin is not known -- is the user's, and stays.
DRAWN_SOURCES = ("extract:", "auto-capture:")


def drawn_from(source: str) -> str | None:
    """The conversation a fact's source says the model drew it from; None for a fact that is the user's."""
    for prefix in DRAWN_SOURCES:
        if source.startswith(prefix) and len(source) > len(prefix):
            return source[len(prefix):]
    return None


# The sources of the user's own acts: a fact written by hand, asked of the agent, or moved from the old store (whose
# origin is not known), and a proposal accepted (``accepted:<id>``). Such an act landing on a fact the model drew
# makes the fact the user's: it takes the act's source, and stays when its conversation goes.
_USERS_OWN = ("manual", "agent", "legacy-import")


def users_own(source: str) -> bool:
    """Whether ``source`` is one of the user's own acts."""
    return source in _USERS_OWN or (source.startswith("accepted:") and len(source) > len("accepted:"))


def forgotten(conversation_id: str) -> bool:
    """Whether this process forgot the conversation: a fact drawn from it is refused, a peer's included."""
    return str(conversation_id) in _FORGOTTEN


class ConversationForgotten(ValueError):
    """A fact drawn from a conversation this process forgot: refused, as its deletion erased the others."""


# The conversations this process forgot: a fact drawn from one is refused, so a capture in flight lands nowhere,
# until a refused deletion hands it back (``unforget_conversation``).
_FORGOTTEN: set[str] = set()
# Held from a write's mark check through its canonical and vector writes, and while a forgetting marks, reads and
# erases: a fact that took it first is erased with the others, one that takes it after is refused.
_WRITE_LOCK = threading.RLock()


def write_lock() -> Any:
    """The facts' write lock: a peer's fact lands under it too (``apply_synced_memory_canonical``), so a forgetting
    that comes erases it, or a check made after the forgetting refuses it."""
    return _WRITE_LOCK


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity in [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union if union else 0.0


@dataclass
class DedupDecision:
    """The outcome of a duplicate check.

    action is "insert" or "merge"; on a merge, target_id names the existing fact,
    reason is "jaccard" or "cosine", and score is the similarity that tripped it.
    """

    action: str
    target_id: str | None = None
    reason: str = ""
    score: float = 0.0


class MemoryDeduplicator:
    """The two-stage duplicate finder over the canonical and vector layers."""

    def __init__(
        self,
        canonical: Any,
        vector: Any,
        *,
        jaccard_threshold: float = JACCARD_THRESHOLD,
        cosine_threshold: float = COSINE_THRESHOLD,
        top_k: int = 5,
    ) -> None:
        self._canonical = canonical
        self._vector = vector
        self._jaccard_threshold = jaccard_threshold
        self._cosine_threshold = cosine_threshold
        self._top_k = top_k

    def find_duplicate(
        self,
        text: str,
        *,
        embedding: list[float] | None = None,
        user_id: str | None = None,
    ) -> DedupDecision:
        # Stage 1: Jaccard over the user's active facts (cheap, no embedding).
        best_id: str | None = None
        best_score = 0.0
        for record in self._canonical.list(active_only=True, user_id=user_id):
            score = jaccard_similarity(text, record.text)
            if score >= self._jaccard_threshold and score > best_score:
                best_id, best_score = record.id, score
        if best_id is not None:
            return DedupDecision("merge", best_id, "jaccard", best_score)

        # Stage 2: cosine near-duplicate via the vector layer.
        if embedding is not None:
            neighbours = self._vector.find_similar(
                embedding,
                user_id=user_id,
                top_k=self._top_k,
                threshold=self._cosine_threshold,
            )
            if neighbours:
                top = neighbours[0]
                return DedupDecision("merge", top.id, "cosine", top.similarity)

        return DedupDecision("insert")


class MemoryStore:
    """Coordinated facade keeping the canonical store and vector layer in sync."""

    def __init__(
        self,
        canonical: Any,
        vector: Any,
        *,
        deduplicator: MemoryDeduplicator | None = None,
    ) -> None:
        self._canonical = canonical
        self._vector = vector
        self._dedup = (
            deduplicator
            if deduplicator is not None
            else MemoryDeduplicator(canonical, vector)
        )

    def _uid(self, user_id: str | None) -> str:
        return self._canonical.resolve_user(user_id)

    def resolve_user(self, user_id: str | None = None) -> str:
        """Resolve the effective user id through the canonical store.

        Exposed so a coordinating layer (e.g. curation) can key its own state by
        the same resolved id the store uses, without touching the canonical
        store directly.
        """
        return self._uid(user_id)

    def _embed(self, text: str, embedding: list[float] | None) -> list[float] | None:
        if embedding is not None:
            return list(embedding)
        return self._vector.embed(text)

    # Create (with dedup)

    def add(
        self,
        text: str,
        category: str = "fact",
        *,
        source: str = "",
        user_id: str | None = None,
        embedding: list[float] | None = None,
        claim: bool = True,
    ) -> tuple[Any, DedupDecision]:
        """Add a fact unless it duplicates an existing one, in which case merge.

        Returns the resulting record and the dedup decision. A merge keeps the
        existing fact's text, and so its endorsement or the lack of one: a
        writer that adopts what it wrote adopts the bytes it wrote, never the
        existing fact's. A fact drawn from a conversation this process forgot
        raises ``ConversationForgotten``, nothing written. One of the user's
        own acts merged into a fact the model drew makes it theirs (``claim``),
        unless ``claim`` is False; a fact erased or put in the trash meanwhile
        takes no merge, and the fact is written anew.
        """
        uid = self._uid(user_id)
        emb = self._embed(text, embedding)
        decision = self._dedup.find_duplicate(text, embedding=emb, user_id=uid)

        with _WRITE_LOCK:
            drawn = drawn_from(source)
            if drawn is not None and drawn in _FORGOTTEN:
                raise ConversationForgotten(f"conversation {drawn} was deleted; a fact drawn from it is not kept")
            if decision.action == "merge" and decision.target_id is not None:
                target = self._canonical.get(decision.target_id, user_id=uid)
                if target is not None and getattr(target, "active", True):
                    # Conservative merge: reinforce the existing fact, keep its text.
                    self._canonical.touch(target.id, user_id=uid)
                    if claim:
                        self.claim(target.id, source, user_id=uid)
                    return self._canonical.get(target.id, user_id=uid), decision
                decision = DedupDecision("insert")
            record = self._canonical.add(text, category, source=source, user_id=uid)
            if emb is not None:
                self._vector.add(
                    record.id,
                    text,
                    embedding=emb,
                    user_id=record.user_id,
                    category=record.category,
                    source=source,
                )
        return record, decision

    # Read (canonical is the source of truth)

    def get(self, fact_id: str, *, user_id: str | None = None) -> Any | None:
        return self._canonical.get(fact_id, user_id=self._uid(user_id))

    def list(
        self,
        *,
        category: str | None = None,
        active_only: bool = True,
        user_id: str | None = None,
        order_by: str = "created_at",
        descending: bool = True,
        limit: int | None = None,
    ) -> list[Any]:
        return self._canonical.list(
            category=category,
            active_only=active_only,
            user_id=self._uid(user_id),
            order_by=order_by,
            descending=descending,
            limit=limit,
        )

    def count(self, *, active_only: bool = True, user_id: str | None = None) -> int:
        return self._canonical.count(active_only=active_only, user_id=self._uid(user_id))

    # Update (mirrored)

    def update(
        self,
        fact_id: str,
        *,
        text: str | None = None,
        category: str | None = None,
        source: str | None = None,
        user_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> Any | None:
        uid = self._uid(user_id)
        fields: dict[str, Any] = {}
        if text is not None:
            fields["text"] = text
        if category is not None:
            fields["category"] = category
        if source is not None:
            fields["source"] = source

        vector_embedding = embedding
        if text is not None and vector_embedding is None:
            vector_embedding = self._vector.embed(text)
        with _WRITE_LOCK:
            # A fact in the trash is not edited: an edit would restart the days it waits there. (A fact not there
            # is the canonical update's to say.)
            reader = getattr(self._canonical, "get", None)
            current = reader(fact_id, user_id=uid) if callable(reader) else None
            if current is not None and not getattr(current, "active", True):
                return None
            record = self._canonical.update(fact_id, user_id=uid, **fields)
            if record is None:
                return None
        # After the lock: an update re-creates no vector a forgetting erased meanwhile, and the embedder, asked
        # again when it gave nothing before, is never called under the lock.
        self._vector.update(
            fact_id,
            text=text,
            embedding=vector_embedding,
            category=category,
            source=source,
            user_id=uid,
        )
        return record

    def claim(self, fact_id: str, source: str, *, user_id: str | None = None) -> bool:
        """One of the user's own acts (``source``) lands on a fact: one the model drew becomes theirs, written under
        the act's source, and stays when its conversation goes; whether it was claimed."""
        if not users_own(source):
            return False
        uid = self._uid(user_id)
        with _WRITE_LOCK:
            reader = getattr(self._canonical, "get", None)
            record = reader(fact_id, user_id=uid) if callable(reader) else None
            # A fact in the trash is left as it is: restored, it can be made theirs.
            if record is None or not getattr(record, "active", True) or drawn_from(record.source) is None:
                return False
            return self.update(fact_id, source=source, user_id=uid) is not None

    def settle_vectors(self) -> int:
        """Settle what the vector layer owes the facts a peer changed (``owed_vectors``): drop the vector of a fact
        out of service or erased, give back the vector of a fact back in service; how many settled. A layer that
        fails keeps them owed, for the next pass."""
        settled = 0
        for fact_id, act in self._canonical.owed_vectors():
            try:
                fact = self._canonical.by_id(fact_id)
                embedding = None
                if act == "add" and fact is not None and fact.active:
                    embedding = self._vector.embed(fact.text)
                    if embedding is None:
                        continue
                with _WRITE_LOCK:
                    current = self._canonical.by_id(fact_id)
                    if act == "drop" and (current is None or not current.active):
                        self._vector.delete(fact_id)
                    elif (act == "add" and embedding is not None and current is not None and current.active
                          and current.text == fact.text):
                        self._vector.add(current.id, current.text, embedding=embedding, user_id=current.user_id,
                                         category=current.category, source=current.source)
                    self._canonical.settle_vector(fact_id)
                settled += 1
            except Exception as exc:  # noqa: BLE001 - owed still, for the next pass
                logger.warning("fact %s: the vector it is owed could not be settled (%s: %s); the next pass tries "
                               "again", fact_id, type(exc).__name__, exc)
        return settled

    def touch(self, fact_id: str, *, user_id: str | None = None) -> bool:
        return self._canonical.touch(fact_id, user_id=self._uid(user_id))

    def adopt(self, fact_id: str, digest: str, *, user_id: str | None = None) -> bool:
        """The user's endorsement of a fact's text, named by its digest (see the canonical store)."""
        return self._canonical.adopt(fact_id, digest, user_id=self._uid(user_id))

    def unendorsed(self, *, user_id: str | None = None) -> list:
        """The active facts the user has not endorsed as they read now."""
        return self._canonical.unendorsed(user_id=self._uid(user_id))

    # Delete (mirrored); soft delete drops the vector entry, keeps the row

    def soft_delete(self, fact_id: str, *, user_id: str | None = None) -> bool:
        uid = self._uid(user_id)
        ok = self._canonical.soft_delete(fact_id, user_id=uid)
        if ok:
            self._vector.delete(fact_id, user_id=uid)
        return ok

    def restore(self, fact_id: str, *, user_id: str | None = None) -> bool:
        uid = self._uid(user_id)
        held = self._canonical.get(fact_id, user_id=uid)
        embedding = self._vector.embed(held.text) if held is not None else None
        # Under the writes' lock: a forgetting that comes erases the row and the vector this gives back together.
        with _WRITE_LOCK:
            ok = self._canonical.restore(fact_id, user_id=uid)
            if ok and embedding is not None:
                record = self._canonical.get(fact_id, user_id=uid)
                # A vector only for the text it was made from.
                if record is not None and record.text == held.text:
                    self._vector.add(
                        record.id,
                        record.text,
                        embedding=embedding,
                        user_id=record.user_id,
                        category=record.category,
                        source=record.source,
                    )
        return ok

    def hard_delete(self, fact_id: str, *, user_id: str | None = None) -> bool:
        uid = self._uid(user_id)
        ok = self._canonical.hard_delete(fact_id, user_id=uid)
        self._vector.delete(fact_id, user_id=uid)
        return ok

    def erase(self, fact_id: str) -> bool:
        """Erase a fact still in the trash, by its id whoever owns it -- its row, then its vector, the peers told;
        whether its row went. The row goes only while the fact is in the trash: one restored meanwhile stays, with
        the vector its restore gave it."""
        gone = self._canonical.erase(fact_id, trashed_only=True)
        if gone:
            self._vector.delete(fact_id)
        return gone

    def trashed_before(self, cutoff: str) -> list[tuple[str, str]]:
        """The id and the owner of every fact in the trash since ``cutoff`` or before (the canonical store's)."""
        return self._canonical.trashed_before(cutoff)

    def empty_log(self, what: str) -> None:
        """Empty the facts' write-ahead log after an erasure, never waiting (the canonical store's)."""
        self._canonical.empty_log(what)

    # Forget a deleted conversation

    def forget_conversation(self, conversation_id: str) -> int:
        """Erase every fact the model drew from a deleted conversation -- in service or in the trash, every user's
        -- with its vector, the peers told; how many. What is erased is overwritten and the log emptied; a failure
        raises. From the start, no fact drawn from the conversation is taken in this process (``_FORGOTTEN``),
        unless the forgetting fails."""
        cid = str(conversation_id)
        try:
            with _WRITE_LOCK:
                # Marked under the writes' lock: a fact that took it first is erased below, one that takes it after
                # is refused.
                _FORGOTTEN.add(cid)
                return self._erase_drawn(cid)
        except BaseException:
            _FORGOTTEN.discard(cid)
            raise

    def _erase_drawn(self, conversation_id: str) -> int:
        return _erase_drawn(self._canonical, self._vector, conversation_id)


# Module-level singleton with a reset for test isolation. The sibling stores are
# imported lazily here so this module stays importable in isolation.
_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        from .canonical_store import get_canonical_store
        from .vector_store import get_vector_store

        _memory_store = MemoryStore(get_canonical_store(), get_vector_store())
    return _memory_store


def reset_memory_store() -> None:
    global _memory_store
    _memory_store = None


def _erase_drawn(canonical: Any, vector: Any, conversation_id: str) -> int:
    """Erase every fact the model drew from the conversation, its vector first; with no vector layer open (``vector``
    None), its vector is owed before its row goes, so an end between the two leaves it owed. Then the log emptied."""
    held = canonical.held_under(prefix + conversation_id for prefix in DRAWN_SOURCES)
    for fact_id, _owner in held:
        if vector is not None:
            vector.delete(fact_id)
        else:
            canonical.owe_vector(fact_id, "drop")
        canonical.erase(fact_id)
    if held:
        canonical.empty_log(f"the facts drawn from conversation {conversation_id} erased")
    return len(held)


def forget_conversation(conversation_id: str) -> int:
    """Erase the facts the model drew from a deleted conversation; how many (``MemoryStore.forget_conversation``).

    A process that never opened the facts whole erases their rows through the canonical store alone, their vectors
    dropped if the layer is open here and owed to the trash's next pass if not; one that finds no file of them has
    nothing to erase, and creates none. Either way it refuses, from now on, a fact drawn from the conversation.
    """
    cid = str(conversation_id)
    with _WRITE_LOCK:
        if _memory_store is None:
            from .canonical_store import _default_db_path, get_canonical_store, open_vector_layer

            _FORGOTTEN.add(cid)
            if not _default_db_path().exists():
                return 0
            try:
                return _erase_drawn(get_canonical_store(), open_vector_layer(), cid)
            except BaseException:
                _FORGOTTEN.discard(cid)
                raise
        return get_memory_store().forget_conversation(cid)


def unforget_conversation(conversation_id: str) -> None:
    """Take facts drawn from the conversation again: its deletion was refused, and it stays."""
    _FORGOTTEN.discard(str(conversation_id))
