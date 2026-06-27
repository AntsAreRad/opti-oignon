#!/usr/bin/env python3
"""Tests for embedder health visibility (M4).

The vector layer fails open: ``embed`` returns None when the embedder is absent
or failing, which silently degrades semantic recall (only keyword/canonical
retrieval keeps working). M4 makes that visible: ``MemoryVectorStore.health``
reports the embedder state, and ``embed`` logs once when no embedder is
configured instead of failing silently forever. This suite loads
``vector_store.py`` in isolation (no chromadb/ollama; a fake embedder injected,
a dummy collection to skip the Chroma build) and proves:

  * health is "unavailable" with no embedder, "ok" when embeddings flow (with
    the dimension), and "degraded" when the embedder is present but returns
    nothing or raises;
  * embed logs the "embedder unavailable" warning exactly once across repeated
    calls.

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import logging
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load():
    keys = ("opti_oignon", "opti_oignon.memory", "opti_oignon.memory.vector_store")
    saved = {k: sys.modules.get(k) for k in keys}
    for n in ("opti_oignon", "opti_oignon.memory"):
        pkg = types.ModuleType(n)
        pkg.__path__ = []
        sys.modules[n] = pkg
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.memory.vector_store", _OO / "memory" / "vector_store.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.memory.vector_store"] = mod
    spec.loader.exec_module(mod)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


class FakeEmbedder:
    def __init__(self, *, vector=None, raises=False):
        self._vector = vector
        self._raises = raises

    def embed_single(self, text):
        if self._raises:
            raise RuntimeError("ollama down")
        return self._vector


def _store(mod, embedder):
    # Pass a non-None placeholder so __init__ does NOT fall back to the real
    # default embedder (which exists on a host with Ollama), then set the
    # intended embedder explicitly. ``None`` here means "no embedder configured",
    # deterministically, on any host. A dummy collection skips the Chroma build.
    vs = mod.MemoryVectorStore(embedder=object(), collection=object())
    vs._embedder = embedder
    vs._warned_no_embedder = False
    return vs


def _degraded_store(mod):
    # Force the no-chromadb path deterministically on any host, then construct
    # WITHOUT injecting a collection so _build_collection runs and must yield a
    # None collection (degraded) instead of raising.
    mod._HAS_CHROMADB = False
    return mod.MemoryVectorStore(embedder=object())


def test_health_unavailable_without_embedder():
    mod, restore = _load()
    try:
        h = _store(mod, None).health()
        assert h["status"] == "unavailable"
        assert h["available"] is False
    finally:
        restore()


def test_health_ok_when_embeddings_flow():
    mod, restore = _load()
    try:
        h = _store(mod, FakeEmbedder(vector=[0.1, 0.2, 0.3])).health()
        assert h["status"] == "ok"
        assert h["available"] is True
        assert h["dim"] == 3
    finally:
        restore()


def test_health_degraded_when_embedder_returns_none():
    mod, restore = _load()
    try:
        h = _store(mod, FakeEmbedder(vector=None)).health()
        assert h["status"] == "degraded"
        assert h["available"] is False
    finally:
        restore()


def test_health_degraded_when_embedder_raises():
    mod, restore = _load()
    try:
        h = _store(mod, FakeEmbedder(raises=True)).health()
        assert h["status"] == "degraded"
        assert h["available"] is False
    finally:
        restore()


def test_embed_warns_once_without_embedder():
    mod, restore = _load()
    try:
        vs = _store(mod, None)

        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture()
        mod.logger.addHandler(handler)
        try:
            assert vs.embed("a") is None
            assert vs.embed("b") is None
            assert vs.embed("c") is None
        finally:
            mod.logger.removeHandler(handler)

        warned = [m for m in records if "embedder unavailable" in m]
        assert len(warned) == 1   # logged once, not on every call
    finally:
        restore()


# ---------------------------------------------------------------------------
# Degrade to canonical-only when chromadb is absent (fulfil the health()
# contract "recall is degraded, not down"): construction must NOT raise, CRUD
# calls are safe no-ops, similarity is empty, and health reports "unavailable".
# ---------------------------------------------------------------------------
def test_construct_without_chromadb_does_not_raise():
    mod, restore = _load()
    try:
        vs = _degraded_store(mod)          # must not raise
        assert vs.collection is None
    finally:
        restore()


def test_health_unavailable_without_collection():
    mod, restore = _load()
    try:
        h = _degraded_store(mod).health()
        assert h["status"] == "unavailable"
        assert h["available"] is False
        assert "chromadb" in h["detail"]   # the collection tier, not the embedder
    finally:
        restore()


def test_crud_are_safe_noops_without_collection():
    mod, restore = _load()
    try:
        vs = _degraded_store(mod)
        assert vs.add("id1", "hello") == "id1"   # canonical keeps it; vector is a no-op
        assert vs.get("id1") is None
        assert vs.count() == 0
        assert vs.update("id1", text="x") is False
        assert vs.delete("id1") is False
        assert vs.clear() == 0
    finally:
        restore()


def test_find_similar_empty_without_collection():
    mod, restore = _load()
    try:
        assert _degraded_store(mod).find_similar([0.1, 0.2, 0.3]) == []
    finally:
        restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
