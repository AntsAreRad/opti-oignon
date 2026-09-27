#!/usr/bin/env python3
"""Shared window and fixtures for the fact-check contract suites.

Every suite loads the package ``opti_oignon.factcheck`` from its files through
the shared isolation window, with the native loader declared unreachable, so
the reference path is what runs. Sources are built here with their SHA-256
computed from their text, exactly as a store records it at ingest; a contract
that needs a hash that does not match passes its own.

Every non-ASCII character a fixture needs is built with ``chr()``.

Local-only (the public distribution ships no tests).
"""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

MODULES = (
    "vocabulary", "markup", "passage", "scope", "sources",
    "record", "render", "decide", "canary", "checker",
)
PACKAGE = "opti_oignon.factcheck"
AS_OF = "2026-09-26"
READ_ON = "2026-09-27"
GRANT = "grant-1"


def load(extra_targets=None, packages=()):
    """Open a window on the package; returns (namespace, restore)."""
    targets = {f"{PACKAGE}.{name}": source("factcheck", f"{name}.py") for name in MODULES}
    targets.update(extra_targets or {})
    loaded, restore = isolate(
        targets=targets,
        packages=(PACKAGE, *packages),
        blocked=("opti_oignon.native",),
    )
    names = {key.rsplit(".", 1)[1]: module for key, module in loaded.items()}
    return SimpleNamespace(**names), restore


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk(fc, text, **fields):
    """A chunk whose hash is its text's, unless the caller passes one."""
    digest = fields.pop("sha256", None) or sha(text)
    return fc.sources.Chunk(text=text, sha256=digest, **fields)


def item(fc, source_id, texts, *, kind="library", author="third_party", consent=GRANT,
         recorded_at="2026-01-10", source_date="2025-11-02", chunk_fields=None, **fields):
    """A source item from one text or several, one chunk per text."""
    if isinstance(texts, str):
        texts = [texts]
    chunk_fields = chunk_fields or {}
    chunks = tuple(chunk(fc, text, **dict(chunk_fields)) for text in texts)
    return fc.sources.SourceItem(
        source_id=source_id, kind=kind, author=author, chunks=chunks, consent=consent,
        recorded_at=recorded_at, source_date=source_date, **fields,
    )


def config(fc):
    return fc.checker.load_config()


def run(fc, claim, items, *, as_of=AS_OF, read_on=READ_ON, cfg=None):
    return fc.decide.check(claim, items, as_of=as_of, read_on=read_on, config=cfg or config(fc))


def blank(fc, items, text):
    """The same items with every chunk's text replaced and its hash recomputed."""
    import dataclasses

    out = []
    for one in items:
        chunks = tuple(dataclasses.replace(c, text=text, sha256=sha(text)) for c in one.chunks)
        out.append(dataclasses.replace(one, chunks=chunks))
    return out
