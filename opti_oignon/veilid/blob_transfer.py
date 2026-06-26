"""Vault blob on-demand transfer (SYN-01, lot 3d-2).

The eager manifest (lot 3c) puts a peer's vault item on screen with a thumbnail
but no bytes. This module moves the bytes ON DEMAND, when the user opens an
item, without ever holding a multi-GB file whole in memory on either side.

Shape (the cas-7 ``remote_streaming`` chunk-buffer, adapted):

  * a fetch is a SESSION keyed by the ``(route-authenticated peer_id,
    request_id)`` pair -- the literal match is the binding, so one peer can
    never pull another's session;
  * the PRODUCER streams the plaintext straight from
    :meth:`NotesBlobStore.open_stream` -- one chunk per pull, off disk, NOT a
    buffered chunk list (the cas-7 list would pin a whole blob in RAM and
    defeat the point). The cursor is forward-only with a one-chunk retry
    window, so a lost ack re-serves the same chunk rather than rewinding;
  * the bytes cross the wire as PLAINTEXT (decision D-B1): the Veilid channel
    is already E2E-encrypted, and the RECEIVER re-seals under its OWN
    per-attachment subkey (a thin client need not hold the producer's master
    key), via :meth:`NotesBlobStore.seal_stream` -- again one chunk at a time;
  * the receiver verifies the manifest's ``content_hash`` (the plaintext
    SHA-256) over the reassembled stream and DISCARDS a blob that does not
    match (decision D-B4), so a corrupted or substituted transfer never lands.

Security posture: the producer gates each open through an injectable
``serve_ok`` predicate (the note's ``mobile_allowed`` filter for a phone-class
peer -- only opted-in items are fetchable), refuses once ``MAX_FETCH_SESSIONS``
are live, and caps a single transfer at ``MAX_CHUNKS_PER_FETCH`` (a DoS bound).
The real Veilid transport that carries the request and the chunks is wired
host-side; here the seam is a direct call, so the whole fetch logic --
streaming, peer binding, cursor, reassembly, hash verification, bounds -- is
exercisable in isolation. ``checkpoint_before_apply`` is hardcoded True.

All collaborators are injectable; nothing in this module imports the Veilid or
Ollama stack at import time.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import threading
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

# Hardcoded and never overridable: a checkpoint is taken before any mutation is
# applied. Project-wide non-negotiable for every new module.
checkpoint_before_apply = True

# At most this many concurrent producer sessions; a new fetch past the bound is
# refused rather than evicting a live one (a peer cannot exhaust memory/handles
# by opening unlimited streams).
MAX_FETCH_SESSIONS = 64
# A single transfer may not exceed this many chunks. At the 1 MiB framed chunk
# size this caps one fetch at 64 GiB -- generous for video, finite by design.
MAX_CHUNKS_PER_FETCH = 65536


def _key(peer_id: Any, request_id: Any) -> tuple | None:
    if not isinstance(peer_id, str) or not peer_id:
        return None
    if not isinstance(request_id, str) or not request_id:
        return None
    return (peer_id, request_id)


class _FetchSession:
    """One in-flight producer stream: a forward-only cursor over open_stream.

    Holds the open generator (and thus the blob's file handle) for the life of
    the fetch. ``serve(cursor)`` returns the next plaintext chunk and advances,
    re-serves the just-served chunk if the same cursor is retried, and signals
    ``done`` once the stream is exhausted. Any other cursor is rejected.
    """

    __slots__ = ("attachment_id", "_gen", "_next", "_last", "_count", "_done")

    def __init__(self, attachment_id: str, gen: Iterator[bytes]) -> None:
        self.attachment_id = attachment_id
        self._gen = gen
        self._next = 0
        self._last: tuple[int, bytes] | None = None
        self._count = 0
        self._done = False

    def serve(self, cursor: int) -> tuple[bytes, bool]:
        if self._last is not None and cursor == self._last[0]:
            # Retry of the just-served chunk (a lost ack): re-serve, no advance.
            return self._last[1], False
        if cursor != self._next:
            raise ValueError("out-of-order blob cursor")
        if self._done:
            return b"", True
        self._count += 1
        if self._count > MAX_CHUNKS_PER_FETCH:
            raise ValueError("blob exceeds the per-fetch chunk bound")
        try:
            data = next(self._gen)
        except StopIteration:
            self._done = True
            return b"", True
        self._last = (self._next, data)
        self._next += 1
        return data, False

    def close(self) -> None:
        try:
            self._gen.close()
        except Exception:
            pass


_sessions: dict[tuple, _FetchSession] = {}
_lock = threading.RLock()


def open_fetch(
    peer_id: str,
    request_id: str,
    attachment_id: str,
    *,
    blob_store: Any,
    serve_ok: Callable[[str], bool] | None = None,
) -> bool:
    """Register a producer fetch session for a confirmed peer. Returns success.

    Refuses (returns ``False``, opens nothing) when the key is malformed, the
    ``serve_ok`` gate denies the attachment (e.g. the note is not phone-allowed
    for this peer-class), the blob is absent, or the session bound is reached.
    Idempotent on an identical key: a repeated open replaces the prior session.
    """
    key = _key(peer_id, request_id)
    if key is None or not isinstance(attachment_id, str) or not attachment_id:
        return False
    if serve_ok is not None:
        try:
            if not serve_ok(attachment_id):
                return False
        except Exception:
            logger.debug("blob serve_ok gate raised", exc_info=True)
            return False
    exists = getattr(blob_store, "exists", None)
    if callable(exists):
        try:
            if not exists(attachment_id):
                return False
        except Exception:
            return False
    with _lock:
        if key not in _sessions and len(_sessions) >= MAX_FETCH_SESSIONS:
            logger.warning("blob fetch refused: session bound reached")
            return False
        prior = _sessions.pop(key, None)
        if prior is not None:
            prior.close()
        try:
            gen = blob_store.open_stream(attachment_id)
        except Exception:
            logger.debug("blob open_stream failed at fetch open", exc_info=True)
            return False
        _sessions[key] = _FetchSession(attachment_id, gen)
    return True


def pull_chunk(peer_id: str, request_id: str, cursor: Any) -> dict | None:
    """Serve one chunk of a fetch by ``(peer_id, request_id)`` and cursor.

    The lookup is a literal key match -- the peer binding. Returns
    ``{"request_id", "content_b64", "cursor", "done"}`` with ``cursor``
    advanced, or ``None`` when no such session exists for this peer. The session
    is dropped (and its file handle released) on ``done`` or on any serve error.
    """
    key = _key(peer_id, request_id)
    if key is None:
        return None
    if not isinstance(cursor, int) or cursor < 0:
        return None
    with _lock:
        session = _sessions.get(key)
        if session is None:
            return None
        try:
            data, done = session.serve(cursor)
        except Exception:
            logger.debug("blob serve error; dropping session", exc_info=True)
            _sessions.pop(key, None)
            session.close()
            return None
        if done:
            _sessions.pop(key, None)
            session.close()
            return {
                "request_id": request_id,
                "content_b64": "",
                "cursor": cursor,
                "done": True,
            }
        return {
            "request_id": request_id,
            "content_b64": base64.b64encode(data).decode("ascii"),
            "cursor": cursor + 1,
            "done": False,
        }


def close_fetch(peer_id: str, request_id: str) -> None:
    """Drop a producer session early (the receiver gave up or finished)."""
    key = _key(peer_id, request_id)
    if key is None:
        return
    with _lock:
        session = _sessions.pop(key, None)
    if session is not None:
        session.close()


def active_fetches() -> int:
    with _lock:
        return len(_sessions)


def reset_blob_transfer() -> None:
    """Drop all sessions (test isolation / shutdown)."""
    with _lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for s in sessions:
        s.close()


def pull_iter(
    peer_id: str, request_id: str, *, start: int = 0
) -> Iterator[bytes]:
    """Yield a fetch's plaintext chunks by driving :func:`pull_chunk` to done.

    The receiver's pull loop. In isolation this drives the producer directly;
    host-side the same loop runs over the Veilid channel. Stops on ``done`` or
    on a dropped session (``None``).
    """
    cursor = start
    while True:
        resp = pull_chunk(peer_id, request_id, cursor)
        if resp is None or resp.get("done"):
            return
        content = resp.get("content_b64") or ""
        try:
            yield base64.b64decode(content)
        except Exception:
            return
        nxt = resp.get("cursor")
        cursor = nxt if isinstance(nxt, int) else cursor + 1


def receive_blob(
    chunks: Any,
    *,
    attachment_id: str,
    dest_store: Any,
    expected_hash: str,
) -> bool:
    """Reassemble a fetched blob into ``dest_store`` and verify its content hash.

    The plaintext chunks (from :func:`pull_iter` or any source) are fed straight
    into ``dest_store.seal_stream`` -- re-sealed under the receiver's own
    per-attachment subkey, one chunk at a time -- while a running SHA-256 is
    taken over exactly those bytes. If the digest does not match the manifest's
    ``content_hash`` (hex) the freshly written blob is DELETED and ``False`` is
    returned, so a corrupted or substituted transfer never survives. An empty
    or non-string ``expected_hash`` is treated as a verification failure (we do
    not land unverifiable bytes).
    """
    if not isinstance(expected_hash, str) or not expected_hash:
        return False
    digest = hashlib.sha256()

    def _tee() -> Iterator[bytes]:
        for c in chunks:
            digest.update(c)
            yield c

    try:
        dest_store.seal_stream(attachment_id, _tee())
    except Exception:
        logger.debug("blob receive seal_stream failed", exc_info=True)
        _safe_delete(dest_store, attachment_id)
        return False

    if digest.hexdigest().lower() != expected_hash.strip().lower():
        logger.warning(
            "blob content hash mismatch for %s; discarding", attachment_id
        )
        _safe_delete(dest_store, attachment_id)
        return False
    return True


def _safe_delete(store: Any, attachment_id: str) -> None:
    delete = getattr(store, "delete", None)
    if callable(delete):
        try:
            delete(attachment_id)
        except Exception:
            pass
