"""Automatic memory capture (M2).

The new memory store had every piece but the trigger: ``extraction`` extracts
durable facts and writes them through the coordinated ``MemoryStore`` (with the
double deduplication), but nothing called it outside the manual ``/extract``
route. This module is that trigger. After an assistant turn the executor calls
:func:`maybe_capture`, which -- when enabled and when the conversation has grown
enough -- fires the extraction in the background so facts accumulate on their
own.

Design:

* GATED. ``OPTI_AUTO_CAPTURE`` (default on) turns it off without a code change.
* THROTTLED. A capture fires only when the conversation has grown by
  ``AUTO_CAPTURE_MIN_NEW`` messages since the last one, tracked by an in-memory
  watermark keyed by conversation id. The watermark resets on restart; the
  store's deduplication makes a re-capture after restart harmless (it merges
  rather than duplicates), so persistence is not required here.
* FIRE-AND-FORGET. The extraction runs in a daemon thread and
  ``extract_and_store`` never raises, so the conversation path is never blocked
  or broken. A failure is swallowed.
* TYPED ONLY. The facts it writes come back in every later memory block, so
  the default runner hands the extraction the user's own typed words alone
  (:func:`typed_turns`): never an attached document, the assistant's reply, a
  question the model reworded, or a turn of no known origin. The executor
  offers it the conversation as the mirror reads it, each turn with its
  origin; a read without origins gives nothing, and with nothing typed the
  extraction is not called.
* ITS CONVERSATION SAID. A fact it writes keeps the conversation it was drawn
  from (the source ``auto-capture:<id>``), and goes when that conversation is
  deleted; a capture that lands after the deletion is refused by the store.

The dispatch is injectable (``runner``) so the gate/throttle logic is unit
tested without spawning a thread or loading the model.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable

# Every new module hardcodes this; it is never overridable.
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

# Messages of growth (user + assistant turns) before another capture fires.
AUTO_CAPTURE_MIN_NEW = 6

# conversation id -> message count at the last capture.
_capture_watermark: dict[str, int] = {}


def reset_auto_capture() -> None:
    """Clear the throttle watermark (used by tests and on store reset)."""
    _capture_watermark.clear()


def _auto_capture_enabled() -> bool:
    """Auto-capture is on unless ``OPTI_AUTO_CAPTURE`` is a falsey value."""
    raw = os.environ.get("OPTI_AUTO_CAPTURE", "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "")


# The turn-origin grammar as the mirror's reader holds it (``memory.probes``), copied whole so that this module, loaded
# alone, keeps no dependency: the copy is held equal to the reader's by its syntax tree.
_ORIGIN_BASES = ("typed", "refined", "document", "assistant", "legacy")
_ORIGIN_FLAGS = ("tool", "web")
_ORIGIN_ROLES = {
    "user": ("typed", "refined", "document", "legacy"),
    "assistant": ("assistant", "legacy"),
}


def _origin_defect(role, origin, segments, length):
    """Why a turn's origin lies outside the grammar, or None when it lies inside."""
    if not isinstance(origin, str) or not origin:
        return "an origin is a non-empty string"
    base, *flags = origin.split("+")
    if base not in _ORIGIN_BASES:
        return f"origin base {base[:24]!r} is not in the grammar"
    for flag in flags:
        if flag not in _ORIGIN_FLAGS:
            return f"origin flag {flag[:24]!r} is not in the grammar"
    if flags != sorted(set(flags)):
        return "origin flags are written once each, in order"
    if flags and base != "assistant":
        return f"a {base} origin carries no flag"
    allowed = _ORIGIN_ROLES.get(role, ("legacy",)) if isinstance(role, str) else ("legacy",)
    if base not in allowed:
        return f"role {str(role)[:24]!r} cannot carry {base}"
    if not isinstance(segments, (list, tuple)):
        return "segments are a list"
    if segments and base == "legacy":
        return "a legacy turn has no segments"
    end = 0
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 3:
            return "a segment is [start, end, base]"
        start, stop, label = segment
        if type(start) is not int or type(stop) is not int:
            return "segment bounds are integers"
        if not end <= start < stop <= length:
            return f"segment [{start}, {stop}] overlaps, is empty or leaves the content"
        if label == "legacy" or label not in allowed:
            return f"role {str(role)[:24]!r} carries no {str(label)[:24]} segment"
        end = stop
    return None


def typed_turns(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """The user's own typed words in ``messages``, as user turns for the extraction.

    A user turn gives its typed segments, or its whole content when it is
    typed and carries no segment. A document's segment, the words the
    executor writes between parts, a turn the model reworded (refined), a
    turn of no known origin (legacy), an assistant reply and any turn whose
    origin or segments cannot be read give nothing -- segments the store could
    not decode arrive as None, and such a turn is held legacy, as the mirror
    holds it. So does a declaration outside the turn-origin grammar (an
    unknown base, a legacy turn that carries segments, a label its role
    cannot carry): the mirror's reader holds it legacy, with no segment.
    """
    turns = []
    for message in messages or ():
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content, origin = message.get("content"), message.get("origin")
        if not isinstance(content, str) or not isinstance(origin, str):
            continue
        segments = message.get("segments")
        if not isinstance(segments, (list, tuple)):
            continue
        if _origin_defect("user", origin, segments, len(content)) is not None:
            continue
        if not segments:
            parts = [content] if origin == "typed" else []
        else:
            # Segments as the grammar writes them: [start, end, base], sorted
            # and disjoint, inside the content; one out of it and the turn
            # gives nothing.
            parts, end = [], 0
            for segment in segments:
                if not isinstance(segment, (list, tuple)) or len(segment) != 3:
                    parts = []
                    break
                start, stop, label = segment
                if type(start) is not int or type(stop) is not int or not end <= start < stop <= len(content):
                    parts = []
                    break
                end = stop
            else:
                # Words typed beside a document in one run of parts (pasted
                # text among the words) take their sense from it: none of
                # them is the user's alone. The memory's reader holds the
                # same rule (probes.beside_documents).
                run_start = 0
                while run_start < len(segments):
                    run_end = run_start + 1
                    while run_end < len(segments) and segments[run_end][0] == segments[run_end - 1][1]:
                        run_end += 1
                    run = segments[run_start:run_end]
                    if not any(label == "document" for _start, _stop, label in run):
                        parts.extend(content[start:stop] for start, stop, label in run if label == "typed")
                    run_start = run_end
        text = "\n".join(part.strip() for part in parts if part.strip())
        if text:
            turns.append({"role": "user", "content": text})
    return turns


def _default_runner(
    messages: list[dict[str, Any]],
    *,
    user_id: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
) -> None:
    """Run the extraction over the typed words in a daemon thread; swallow any failure.

    The extractor is imported lazily so importing this module stays light and
    free of the model/store import chain. With no typed word, nothing runs.
    The facts are written under the source ``auto-capture:<conversation_id>``.
    """
    typed = typed_turns(messages)
    if not typed:
        return
    source = f"auto-capture:{conversation_id}" if conversation_id else "auto-capture"

    def _job() -> None:
        try:
            from .extraction import extract_and_store

            # Typed words alone: a single turn of them is worth the model pass,
            # and the facts drawn from them are the user's own.
            extract_and_store(
                typed, source=source, user_id=user_id, model=model, min_messages=1, endorsed=True
            )
        except Exception:  # noqa: BLE001
            logger.debug("auto-capture extraction failed", exc_info=True)

    threading.Thread(target=_job, name="oo-auto-capture", daemon=True).start()


def maybe_capture(
    conversation_id: str | None,
    messages: list[dict[str, Any]] | None,
    *,
    user_id: str | None = None,
    model: str | None = None,
    min_new: int | None = None,
    runner: Callable[..., None] | None = None,
) -> bool:
    """Fire a background capture if enabled and the growth threshold is met.

    Returns ``True`` when a capture was dispatched, ``False`` otherwise. Never
    raises: a dispatch error is swallowed so the calling turn is never broken.
    """
    if not _auto_capture_enabled():
        return False
    if not conversation_id or not isinstance(messages, list) or not messages:
        return False

    threshold = AUTO_CAPTURE_MIN_NEW if min_new is None else int(min_new)
    count = len(messages)
    if count - _capture_watermark.get(conversation_id, 0) < threshold:
        return False

    # Advance the watermark before dispatch so a concurrent turn cannot
    # double-fire on the same growth.
    _capture_watermark[conversation_id] = count

    try:
        if runner is not None:
            runner(list(messages), user_id=user_id, model=model)
        else:
            # The default runner writes under the conversation it draws from.
            _default_runner(list(messages), user_id=user_id, model=model, conversation_id=conversation_id)
    except Exception:  # noqa: BLE001
        logger.debug("auto-capture dispatch failed", exc_info=True)
    return True


def forget_conversation(conversation_id: str) -> int:
    """Let go of a deleted conversation's watermark; 1 when there was one. Its facts are the store's to erase."""
    return 1 if _capture_watermark.pop(str(conversation_id), None) is not None else 0
