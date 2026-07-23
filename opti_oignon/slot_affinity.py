#!/usr/bin/env python3
"""Per-conversation slot affinity for the external llama-server.

The server keeps one prompt-KV cache PER SLOT. Left to itself it picks the
slot for every request, so two conversations routinely land on the same one:
the second request either evicts the first's cached prefix, or -- when the two
share leading bytes, which conversations under one system prompt always do --
is decoded on top of attention state computed over ANOTHER conversation's
content. Deciding the slot here is what makes reuse safe rather than merely
fast.

WHAT KEYS A SLOT. A conversation identity, the fingerprint of the invariant
head, and the trust-envelope state, hashed together. The identity is what
keeps two conversations apart. The head fingerprint is what makes the key
survive a turn: the current turn's bytes are deliberately NOT part of it, or
the key would move every turn and the cache it protects would never be reused.
The envelope state is what stops attention state computed with no untrusted
content from being extended by a turn that carries some, and the reverse.

THE INVARIANT. Two distinct keys never hold the same slot at the same time.
It is enforced by construction: a slot already held is not offered to another
key, and an eviction removes the previous holder before the slot is handed
over. A caller with no conversation identity -- question refinement, one-shot
execution -- gets no slot at all rather than a shared one, because a slot it
were given is a slot some conversation loses.

EVICTION IS EXPLICIT. When every listed slot is held and none is free, the
least recently used assignment is dropped and its slot handed over. Silence
would mean the server picking instead, which is the behaviour this module
exists to replace.

DEGRADING IS THE DEFAULT PATH, NOT THE EDGE CASE. The server does not serve
its slot listing unless it was started with the flag that enables it, so an
empty listing is what an ordinary host reports. No listing, no identity, no
eligible slot, a malformed entry, a listing that is not a list: every one of
them answers None, meaning no slot is requested and the server decides as it
did before. This module never raises, and it never invents a slot it did not
read.

Import-safe and stdlib-only: it reads a listing it is handed and returns an
integer or None. It opens no socket, imports no backend, and knows nothing
about how the listing was obtained.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module conventions.
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Key namespace. Bumping it invalidates every affinity at once, which is the
# intended migration path if the key's composition ever changes.
_KEY_NAMESPACE = "opti-oignon:slot-affinity:1"

# Envelope label used when a turn carries no untrusted content at all.
ENVELOPE_NONE = "none"


def envelope_state(sources: Any) -> str:
    """A canonical label for the trust envelope of one assembled context.

    ``sources`` is any iterable of untrusted-source labels present in the
    context. Order and duplicates are not part of the envelope; the set is.
    Anything unusable -- None, a bare string, a non-iterable -- reads as no
    untrusted content, which is the conservative direction: it can only split
    keys that a caller meant to keep together, never merge keys that a caller
    meant to keep apart.
    """
    if not sources or isinstance(sources, (str, bytes)):
        return ENVELOPE_NONE
    try:
        labels = sorted({str(item) for item in sources if item})
    except TypeError:
        return ENVELOPE_NONE
    return "+".join(labels) if labels else ENVELOPE_NONE


def routing_key(
    *,
    conversation_id: Any,
    prefix_fingerprint: Any,
    envelope: str = ENVELOPE_NONE,
) -> str | None:
    """The affinity key, or None when there is no conversation to key on.

    The three components are joined under a namespace and hashed. The current
    turn is not among them, by design: a key that moved every turn would hand
    out a fresh slot every turn and defeat the cache it is meant to protect.
    """
    # Any empty identity -- None, the empty string, a zero left over from a
    # caller that had no conversation to name -- reads as no identity at all.
    # Refusing is the safe direction: a slot named on a doubtful identity is
    # a slot some real conversation loses, while refusing costs only reuse.
    if not conversation_id:
        return None
    identity = str(conversation_id)
    material = "\x1f".join(
        (
            _KEY_NAMESPACE,
            identity,
            str(prefix_fingerprint or ""),
            str(envelope or ENVELOPE_NONE),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _slot_id(entry: Any) -> int | None:
    """The slot number of one listing entry, or None when unreadable."""
    if not isinstance(entry, dict):
        return None
    raw = entry.get("id")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw if raw >= 0 else None


def _is_busy(entry: Any) -> bool:
    """Whether the listing reports this slot as currently decoding.

    Two markers are read because the server has used both: an explicit
    processing flag, and a task number that is non-negative while a task is
    attached. Neither present means idle -- the listing shapes that carry no
    marker at all are the ones where every slot is free.
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("is_processing"):
        return True
    task = entry.get("id_task")
    return isinstance(task, int) and not isinstance(task, bool) and task >= 0


class SlotAffinity:
    """Holds which key owns which slot, and hands out slots accordingly.

    One instance per process. It is not thread-safe by itself; the execution
    hub calls it from the thread that owns the turn.
    """

    def __init__(self) -> None:
        # key -> slot id
        self._held: dict[str, int] = {}
        # keys oldest-use first; the eviction order
        self._recency: list[str] = []
        # conversation identity -> the key it currently holds. A conversation
        # whose head or envelope moves produces a NEW key, and without this
        # map its previous key would keep a slot nobody will ever ask for
        # again -- an estate of live slots held by nothing, which is the
        # shape eviction pressure comes from when it should not.
        self._by_identity: dict[str, str] = {}

    # -- state -------------------------------------------------------------

    @property
    def assignments(self) -> dict[str, int]:
        """A copy of the current key-to-slot map."""
        return dict(self._held)

    def forget(self, key: str) -> bool:
        """Drop one assignment. True when there was one to drop."""
        if key not in self._held:
            return False
        del self._held[key]
        if key in self._recency:
            self._recency.remove(key)
        for identity, held_key in list(self._by_identity.items()):
            if held_key == key:
                del self._by_identity[identity]
        return True

    def clear(self) -> None:
        """Drop every assignment."""
        self._held.clear()
        self._recency.clear()
        self._by_identity.clear()

    def _touch(self, key: str) -> None:
        if key in self._recency:
            self._recency.remove(key)
        self._recency.append(key)

    # -- the decision ------------------------------------------------------

    def choose(
        self,
        *,
        conversation_id: Any,
        prefix_fingerprint: Any,
        envelope: str = ENVELOPE_NONE,
        slots: Any,
    ) -> int | None:
        """The slot this conversation should decode on, or None.

        None means no slot is requested and the server keeps deciding, which
        is what every degraded shape answers: no identity, no listing, a
        listing that is not a list, no eligible slot. It never raises.
        """
        key = routing_key(
            conversation_id=conversation_id,
            prefix_fingerprint=prefix_fingerprint,
            envelope=envelope,
        )
        if key is None:
            return None
        if not isinstance(slots, list) or not slots:
            return None

        listed: dict[int, bool] = {}
        for entry in slots:
            sid = _slot_id(entry)
            if sid is not None and sid not in listed:
                listed[sid] = _is_busy(entry)
        if not listed:
            return None

        # An assignment survives only while its slot is still listed. A slot
        # that vanished takes its assignment with it rather than pointing at
        # a number the server no longer has.
        for held_key, held_slot in list(self._held.items()):
            if held_slot not in listed:
                self.forget(held_key)

        # One conversation, one slot. A new key for an identity that already
        # holds one releases the old key first, so a conversation whose head
        # or envelope moved moves WITH its slot instead of stranding it.
        identity = str(conversation_id)
        previous = self._by_identity.get(identity)
        if previous is not None and previous != key:
            self.forget(previous)
        self._by_identity[identity] = key

        held = self._held.get(key)
        if held is not None:
            self._touch(key)
            return held

        taken = set(self._held.values())
        free = [sid for sid in sorted(listed) if sid not in taken]

        # A fresh assignment only ever lands on an idle slot: taking a slot
        # mid-decode would queue this turn behind work that is not ours.
        idle = [sid for sid in free if not listed[sid]]
        if idle:
            chosen = idle[0]
            self._held[key] = chosen
            self._touch(key)
            return chosen

        # Every listed slot is held. Eviction is explicit and least-recently
        # used, and only ever takes an idle slot.
        for victim in list(self._recency):
            victim_slot = self._held.get(victim)
            if victim_slot is None or listed.get(victim_slot, True):
                continue
            self.forget(victim)
            self._held[key] = victim_slot
            self._touch(key)
            logger.debug(
                "slot affinity: evicted an assignment to free slot %d",
                victim_slot,
            )
            return victim_slot

        # Nothing idle anywhere: ask for no slot rather than fight for one.
        return None


_INSTANCE: SlotAffinity | None = None


def get_slot_affinity() -> SlotAffinity:
    """The process-wide instance."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SlotAffinity()
    return _INSTANCE


def reset_slot_affinity() -> None:
    """Drop the process-wide instance. For tests and for a backend swap."""
    global _INSTANCE
    _INSTANCE = None
