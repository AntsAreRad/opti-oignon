#!/usr/bin/env python3
"""Tests for the store-backed /api/memory handlers (M3b).

The legacy /api/memory surface is now backed by the coordinated MemoryStore, so
the memory tab writes one store. This suite loads ``routes_memory.py`` with the
memory/conversation dependencies stubbed (the real FastAPI + schemas), then
drives the handler functions directly against a fake store and proves:

  * add_fact routes to ``store.add`` with the category mapped onto the canonical
    set and source="manual", returning the legacy MemoryFactSchema;
  * an empty fact is rejected (422);
  * list_facts maps store records onto MemoryFactSchema (fact <- text);
  * delete_fact soft-deletes (404 when absent);
  * clear_all soft-deletes every active record and counts them;
  * extract routes to the new extraction over the conversation messages and
    counts inserts (merges excluded);
  * /migrate calls the migration with force=True;
  * an unavailable store yields 503.

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _install_stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__path__ = []
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _load():
    names = [
        "opti_oignon", "opti_oignon.api", "opti_oignon.memory",
        "opti_oignon.api.deps", "opti_oignon.api.schemas",
        "opti_oignon.api.routes_memory",
        "opti_oignon.memory.dedup", "opti_oignon.memory.extraction",
        "opti_oignon.memory.migration", "opti_oignon.memory.canonical_store",
        "opti_oignon.conversation", "opti_oignon.memory.vector_store",
    ]
    saved = {k: sys.modules.get(k) for k in names}

    # Minimal fastapi stub: APIRouter whose decorators return the function
    # unchanged, an HTTPException carrying status_code, and a passthrough Depends.
    saved["fastapi"] = sys.modules.get("fastapi")

    class _APIRouter:
        def __init__(self, *a, **k):
            pass

        def _deco(self, *a, **k):
            def wrap(fn):
                return fn
            return wrap

        get = post = delete = patch = put = _deco

    class _HTTPException(Exception):
        def __init__(self, status_code=None, detail=None):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    def _Depends(x=None):
        return x

    _install_stub("fastapi", APIRouter=_APIRouter, Depends=_Depends,
                  HTTPException=_HTTPException)

    _install_stub("opti_oignon")
    _install_stub("opti_oignon.api")
    _install_stub("opti_oignon.memory")
    _install_stub("opti_oignon.api.deps", MEMORY_AVAILABLE=True, memory_manager=object())
    _install_stub("opti_oignon.memory.dedup", get_memory_store=lambda: None)
    _install_stub("opti_oignon.memory.extraction", extract_and_store=lambda *a, **k: [])
    _install_stub("opti_oignon.memory.migration",
                  migrate_legacy_to_store=lambda **k: {"forced": k.get("force")})
    _install_stub("opti_oignon.memory.canonical_store",
                  CATEGORIES=frozenset({"identity", "preference", "fact",
                                        "contact", "project", "goal"}))
    _install_stub("opti_oignon.conversation", conversation_manager=object())
    _install_stub(
        "opti_oignon.memory.vector_store",
        get_vector_store=lambda: types.SimpleNamespace(
            health=lambda: {"status": "ok", "available": True, "dim": 3}
        ),
    )

    # Real schemas (pydantic models the mapper builds).
    sp = importlib.util.spec_from_file_location(
        "opti_oignon.api.schemas", _OO / "api" / "schemas.py")
    schemas = importlib.util.module_from_spec(sp)
    sys.modules["opti_oignon.api.schemas"] = schemas
    sp.loader.exec_module(schemas)

    # Real routes_memory (imports resolve to the stubs + real schemas + FastAPI).
    rp = importlib.util.spec_from_file_location(
        "opti_oignon.api.routes_memory", _OO / "api" / "routes_memory.py")
    rm = importlib.util.module_from_spec(rp)
    sys.modules["opti_oignon.api.routes_memory"] = rm
    rp.loader.exec_module(rm)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return rm, schemas, restore


@dataclass
class FRecord:
    id: str
    text: str
    category: str = "fact"
    source: str = ""
    created_at: str = "t0"
    updated_at: str = "t0"
    active: bool = True
    use_count: int = 0


@dataclass
class FDec:
    action: str = "add"


class FakeStore:
    def __init__(self, records=None, *, merge=False):
        self._records = list(records or [])
        self.added = []
        self.soft_deleted = []
        self._merge = merge

    def add(self, text, category="fact", *, source="", user_id=None, embedding=None):
        self.added.append({"text": text, "category": category, "source": source})
        rec = FRecord(id="new", text=text, category=category, source=source)
        return (rec, FDec(action="merge" if self._merge else "add"))

    def list(self, *, category=None, active_only=True, user_id=None):
        return list(self._records)

    def soft_delete(self, fact_id, *, user_id=None):
        self.soft_deleted.append(fact_id)
        return any(r.id == fact_id for r in self._records) or fact_id == "exists"


def _http_status(fn, *a, **k):
    try:
        fn(*a, **k)
    except Exception as e:  # noqa: BLE001
        return getattr(e, "status_code", None)
    return None


def test_add_fact_routes_and_maps_category():
    rm, schemas, restore = _load()
    try:
        store = FakeStore()
        rm._get_store = lambda: store
        rm._STORE_OK = True
        out = rm.add_fact(schemas.MemoryAddRequest(fact="name is Leon", category="context"))
        assert store.added[0]["category"] == "fact"   # "context" mapped to default
        assert store.added[0]["source"] == "manual"
        assert out.fact == "name is Leon"
    finally:
        restore()


def test_add_fact_empty_422():
    rm, schemas, restore = _load()
    try:
        rm._get_store = lambda: FakeStore()
        rm._STORE_OK = True
        assert _http_status(rm.add_fact, schemas.MemoryAddRequest(fact="   ")) == 422
    finally:
        restore()


def test_list_facts_maps_records():
    rm, schemas, restore = _load()
    try:
        store = FakeStore([FRecord("a", "fact one"), FRecord("b", "fact two")])
        rm._get_store = lambda: store
        rm._STORE_OK = True
        out = rm.list_facts()
        assert [f.fact for f in out] == ["fact one", "fact two"]
        assert out[0].id == "a"
    finally:
        restore()


def test_delete_fact_soft_deletes():
    rm, schemas, restore = _load()
    try:
        store = FakeStore([FRecord("x", "t")])
        rm._get_store = lambda: store
        rm._STORE_OK = True
        out = rm.delete_fact("x")
        assert out["deleted"] is True
        assert store.soft_deleted == ["x"]
        assert _http_status(rm.delete_fact, "missing") == 404
    finally:
        restore()


def test_clear_all_soft_deletes_each():
    rm, schemas, restore = _load()
    try:
        store = FakeStore([FRecord("a", "1"), FRecord("b", "2"), FRecord("c", "3")])
        rm._get_store = lambda: store
        rm._STORE_OK = True
        out = rm.clear_all_facts()
        assert out["count"] == 3
        assert store.soft_deleted == ["a", "b", "c"]
    finally:
        restore()


def test_extract_routes_to_new_extraction():
    rm, schemas, restore = _load()
    try:
        rm._get_store = lambda: FakeStore()
        rm._STORE_OK = True
        rm._conv_manager = types.SimpleNamespace(
            get_context_messages=lambda cid: [{"role": "user", "content": "hi"}])
        rm._extract_and_store = lambda msgs, **k: [
            (FRecord("1", "a"), FDec("add")),
            (FRecord("2", "b"), FDec("merge")),
        ]
        out = rm.extract_facts("c1")
        assert out.facts_added == 1        # merge excluded
        assert out.conversation_id == "c1"
    finally:
        restore()


def test_migrate_route_forces():
    rm, schemas, restore = _load()
    try:
        captured = {}
        rm._migrate_legacy = lambda **k: captured.update(k) or {"ok": True}
        out = rm.migrate_memory()
        assert captured.get("force") is True
        assert out == {"ok": True}
    finally:
        restore()


def test_store_unavailable_503():
    rm, schemas, restore = _load()
    try:
        rm._STORE_OK = False
        rm._get_store = None
        assert _http_status(rm.add_fact, schemas.MemoryAddRequest(fact="x")) == 503
    finally:
        restore()


def test_health_endpoint_ok():
    rm, schemas, restore = _load()
    try:
        sys.modules["opti_oignon.memory.vector_store"].get_vector_store = (
            lambda: types.SimpleNamespace(health=lambda: {"status": "ok", "dim": 3})
        )
        out = rm.memory_health()
        assert out["degraded"] is False
        assert out["archive"] == "ok"
        assert out["embedder"]["status"] == "ok"
    finally:
        restore()


def test_health_endpoint_degraded():
    rm, schemas, restore = _load()
    try:
        sys.modules["opti_oignon.memory.vector_store"].get_vector_store = (
            lambda: types.SimpleNamespace(
                health=lambda: {"status": "degraded", "available": False}
            )
        )
        out = rm.memory_health()
        assert out["degraded"] is True
        assert out["archive"] == "degraded"
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
