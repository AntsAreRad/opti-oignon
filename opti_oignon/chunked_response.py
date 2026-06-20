#!/usr/bin/env python3
"""
Chunked transfer encoding utilities for large responses (S159).

Provides a generator that breaks a JSON-serialisable payload into
fixed-size chunks suitable for ``StreamingResponse``.  Each chunk
is a valid UTF-8 byte sequence; the generator never splits a
multi-byte character across chunk boundaries.

Includes an optional progress callback so the frontend can display
a streaming indicator while a large RAG response is being delivered.

Usage::

    from opti_oignon.chunked_response import chunked_json_generator

    data = {"results": [...]}  # large payload
    gen = chunked_json_generator(data, chunk_size=4096)
    return StreamingResponse(gen, media_type="application/json")
"""

import json
import logging
import time
from collections.abc import Callable, Generator

# Hardcoded, never overridable
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

# Default chunk size in bytes
DEFAULT_CHUNK_SIZE = 4096


def chunked_json_generator(
    payload: dict | list,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
) -> Generator[bytes, None, None]:
    """Yield a JSON payload as fixed-size byte chunks.

    Parameters
    ----------
    payload : dict or list
        The data to serialise and stream.
    chunk_size : int
        Target size in bytes for each chunk.  The last chunk may be
        smaller.  Must be >= 64.
    on_progress : callable, optional
        Called with ``(bytes_sent, total_bytes)`` after each chunk is
        yielded.  Useful for logging or frontend progress reporting.

    Yields
    ------
    bytes
        UTF-8 encoded chunks of the JSON representation.
    """
    if chunk_size < 64:
        raise ValueError("chunk_size must be >= 64")

    start = time.monotonic()

    # Serialise the full payload to bytes once
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    total = len(raw)

    if total == 0:
        return

    sent = 0
    chunk_count = 0
    while sent < total:
        end = min(sent + chunk_size, total)
        # Avoid splitting a multi-byte UTF-8 character:
        # if the byte at `end` is a continuation byte (10xxxxxx),
        # back up until we find a leading byte.
        if end < total:
            while end > sent and (raw[end] & 0xC0) == 0x80:
                end -= 1
            if end == sent:
                # Extremely unlikely: entire chunk is continuation bytes.
                # Fall back to the original boundary.
                end = min(sent + chunk_size, total)

        chunk = raw[sent:end]
        yield chunk
        chunk_count += 1
        sent = end

        if on_progress is not None:
            try:
                on_progress(sent, total)
            except Exception:
                pass  # Never let callback errors break the stream

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.debug(
        "Chunked response: %d bytes in %d chunks (%.1f ms, chunk_size=%d)",
        total, chunk_count, elapsed_ms, chunk_size,
    )


def chunked_text_generator(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
) -> Generator[str, None, None]:
    """Yield a text string as fixed-size string chunks.

    Similar to ``chunked_json_generator`` but operates on plain text
    and yields ``str`` objects (suitable for ``StreamingResponse``
    with ``media_type="text/plain"``).

    Parameters
    ----------
    text : str
        The text to stream.
    chunk_size : int
        Target size in characters for each chunk.  Must be >= 64.
    on_progress : callable, optional
        Called with ``(chars_sent, total_chars)`` after each chunk.
    """
    if chunk_size < 64:
        raise ValueError("chunk_size must be >= 64")

    total = len(text)
    if total == 0:
        return

    sent = 0
    while sent < total:
        end = min(sent + chunk_size, total)
        # Avoid splitting a surrogate pair
        if end < total and text[end - 1:end + 1]:
            c = text[end - 1]
            if "\uD800" <= c <= "\uDBFF":
                end -= 1
        chunk = text[sent:end]
        yield chunk
        sent = end

        if on_progress is not None:
            try:
                on_progress(sent, total)
            except Exception:
                pass


# -- Module availability flag --
CHUNKED_RESPONSE_AVAILABLE = True
