#!/usr/bin/env python3
"""Tests for S174 -- the MemoryStore-backed memory API surface.

The API (FastAPI) is not importable in the sandbox, so the routes, schemas, app
registration, and frontend client are checked by file content. The store
semantics the endpoints depend on (list filters, edit, soft delete, restore) are
additionally verified at the store boundary in isolation, so the surface is
grounded in real behaviour and not only in text.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
MEM = OO / "memory"
ROUTES = OO / "api" / "routes_memory.py"
SCHEMAS = OO / "api" / "schemas.py"
APP = OO / "api" / "app.py"
CLIENT = ROOT / "frontend" / "src" / "lib" / "api" / "memories.ts"

sys.path.insert(0, str(ROOT / "tests"))
from _memory_fakes import FakeChromaCollection, FakeEmbedder  # noqa: E402


def _ensure_pkg():
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(OO)]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.memory" not in sys.modules:
        mpkg = types.ModuleType("opti_oignon.memory")
        mpkg.__path__ = [str(MEM)]
        sys.modules["opti_oignon.memory"] = mpkg


def _ensure_real(name: str):
    full = f"opti_oignon.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(OO / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_mem(name: str):
    full = f"opti_oignon.memory.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(MEM / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg()
_ensure_real("db_encryption")
_ensure_real("user_isolation")
canon = _ensure_mem("canonical_store")
vec = _ensure_mem("vector_store")
ded = _ensure_mem("dedup")


def _routes() -> str:
    return ROUTES.read_text(encoding="utf-8")


def _schemas() -> str:
    return SCHEMAS.read_text(encoding="utf-8")


# Backend routes (file content)


class TestRoutesFile:
    def test_memories_router_with_prefix(self):
        text = _routes()
        assert "memories_router" in text
        assert 'prefix="/api/memories"' in text

    def test_uses_memory_store(self):
        text = _routes()
        assert "from opti_oignon.memory.dedup import get_memory_store" in text
        assert "MEMORY_STORE_AVAILABLE" in text

    def test_has_list_endpoint(self):
        text = _routes()
        assert '@memories_router.get("", response_model=list[MemoryRecordSchema])' in text
        assert "def list_memories(" in text
        assert "store.list(" in text

    def test_has_edit_endpoint(self):
        text = _routes()
        assert '@memories_router.patch("/{fact_id}"' in text
        assert "def edit_memory(" in text
        assert "store.update(" in text

    def test_has_soft_delete_endpoint(self):
        text = _routes()
        assert '@memories_router.delete("/{fact_id}")' in text
        assert "def soft_delete_memory(" in text
        assert "store.soft_delete(" in text

    def test_has_restore_endpoint(self):
        text = _routes()
        assert '@memories_router.post("/{fact_id}/restore"' in text
        assert "def restore_memory(" in text
        assert "store.restore(" in text

    def test_per_user_threaded(self):
        text = _routes()
        # The auth subject is threaded to the store as the user id.
        assert 'current_user.get("sub")' in text
        assert "user_id=user_id" in text

    def test_category_validated(self):
        text = _routes()
        assert "_MEMORY_CATEGORIES" in text
        assert "Invalid category" in text

    def test_auth_dependency_guarded(self):
        text = _routes()
        assert "_get_current_user" in text
        assert "dependencies=_memories_auth_dep" in text


# Legacy surface intact (file content)


class TestLegacyIntact:
    def test_legacy_router_prefix_present(self):
        text = _routes()
        assert 'router = APIRouter(prefix="/api/memory", tags=["memory"])' in text

    def test_legacy_endpoints_present(self):
        text = _routes()
        for fn in ("def list_facts(", "def add_fact(", "def delete_fact(", "def clear_all_facts(", "def extract_facts("):
            assert fn in text

    def test_legacy_uses_memory_manager(self):
        text = _routes()
        assert "memory_manager" in text
        assert "MEMORY_AVAILABLE" in text


# Schemas (file content)


class TestSchemas:
    def test_record_schema_fields(self):
        text = _schemas()
        assert "class MemoryRecordSchema(BaseModel):" in text
        for field in ("id:", "text:", "category:", "source:", "created_at:", "updated_at:", "active:", "use_count:"):
            assert field in text

    def test_edit_request_schema(self):
        text = _schemas()
        assert "class MemoryEditRequest(BaseModel):" in text
        assert "text: str | None" in text
        assert "category: str | None" in text


# App registration (file content)


class TestAppRegistration:
    def test_imports_memories_router(self):
        text = APP.read_text(encoding="utf-8")
        assert "from .routes_memory import memories_router" in text

    def test_includes_memories_router(self):
        text = APP.read_text(encoding="utf-8")
        assert "app.include_router(memories_router)" in text

    def test_legacy_memory_router_still_registered(self):
        text = APP.read_text(encoding="utf-8")
        assert "from .routes_memory import router as memory_router" in text
        assert "app.include_router(memory_router)" in text


# Frontend client (file content)


class TestFrontendClient:
    def test_client_exists(self):
        assert CLIENT.exists()

    def test_exports_four_operations(self):
        text = CLIENT.read_text(encoding="utf-8")
        for fn in ("export async function listMemories", "export async function editMemory", "export async function softDeleteMemory", "export async function restoreMemory"):
            assert fn in text

    def test_uses_memories_paths(self):
        text = CLIENT.read_text(encoding="utf-8")
        assert "'/api/memories'" in text
        assert "/api/memories/${id}" in text
        assert "/api/memories/${id}/restore" in text

    def test_record_type_and_categories(self):
        text = CLIENT.read_text(encoding="utf-8")
        assert "export interface MemoryRecord" in text
        assert "MEMORY_CATEGORIES" in text
        for cat in ("identity", "preference", "fact", "contact", "project", "goal"):
            assert cat in text


# Store semantics the endpoints depend on (isolated runtime)


def _build(tmp_path, *, single_user_mode=True):
    canon_store = canon.CanonicalMemoryStore(
        tmp_path / "api.db", single_user_mode=single_user_mode
    )
    vstore = vec.MemoryVectorStore(
        collection=FakeChromaCollection(name=vec.COLLECTION_NAME),
        embedder=FakeEmbedder(dim=16),
    )
    return ded.MemoryStore(canon_store, vstore)


class TestStoreSemantics:
    def test_list_filters_by_category(self, tmp_path):
        store = _build(tmp_path)
        store.add("The user studies marine biology", "project")
        store.add("The user prefers dark mode", "preference")
        projects = store.list(category="project")
        assert len(projects) == 1 and projects[0].category == "project"

    def test_list_active_only_excludes_soft_deleted(self, tmp_path):
        store = _build(tmp_path)
        rec = store.add("The user prefers tea", "preference")[0]
        store.soft_delete(rec.id)
        assert store.list(active_only=True) == []
        assert len(store.list(active_only=False)) == 1

    def test_edit_changes_text(self, tmp_path):
        store = _build(tmp_path)
        rec = store.add("The user prefers tea", "preference")[0]
        updated = store.update(rec.id, text="The user prefers coffee")
        assert updated.text == "The user prefers coffee"

    def test_soft_delete_then_restore_round_trip(self, tmp_path):
        store = _build(tmp_path)
        rec = store.add("The user prefers tea", "preference")[0]
        assert store.soft_delete(rec.id) is True
        assert store.get(rec.id).active is False
        assert store.restore(rec.id) is True
        assert store.get(rec.id).active is True

    def test_soft_delete_missing_returns_false(self, tmp_path):
        store = _build(tmp_path)
        assert store.soft_delete("nope") is False

    def test_update_missing_returns_none(self, tmp_path):
        store = _build(tmp_path)
        assert store.update("nope", text="x") is None
