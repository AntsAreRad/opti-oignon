"""
S194 F7e -- user data management fix lot tests (UD-03).

Covers:
- The memories export/wipe legs target the user-scoped two-tier
  canonical store (real behavioral test on a temp store: user A wiped,
  user B intact, exact counts).
- The conversations legs detect the unscoped store explicitly (no
  swallowed TypeError) and remain forward-compatible.
- Export metadata and delete results carry the not_covered inventory;
  the route response model accepts it.

user_data_manager.py imports lazily inside functions, so it loads
standalone; the canonical store is loaded with the package-stub idiom
(relative .config import) plus a cleaned opti_oignon.db_utils stub.
"""

import importlib.util
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (_PROJECT / rel).read_text(encoding="utf-8")


def _load_plain(name, rel_path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(_PROJECT / rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_canonical_store():
    """Load canonical_store with stub parent packages, then clean up."""
    pkg = types.ModuleType("s194e_pkg")
    pkg.__path__ = [str(_PROJECT / "opti_oignon")]
    sys.modules.setdefault("s194e_pkg", pkg)
    mem = types.ModuleType("s194e_pkg.memory")
    mem.__path__ = [str(_PROJECT / "opti_oignon" / "memory")]
    sys.modules.setdefault("s194e_pkg.memory", mem)

    created = "opti_oignon" not in sys.modules
    if created:
        oo = types.ModuleType("opti_oignon")
        oo.__path__ = []
        sys.modules["opti_oignon"] = oo
    if "opti_oignon.db_utils" not in sys.modules:
        import sqlite3 as _sq3
        dbu = types.ModuleType("opti_oignon.db_utils")
        dbu.safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)
        dbu.get_encrypted_connection = lambda p, **kw: _sq3.connect(str(p), **kw)
        sys.modules["opti_oignon.db_utils"] = dbu
        sys.modules["opti_oignon"].db_utils = dbu

    try:
        spec = importlib.util.spec_from_file_location(
            "s194e_pkg.memory.canonical_store",
            str(_PROJECT / "opti_oignon" / "memory" / "canonical_store.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for stub in ("opti_oignon.db_utils", "opti_oignon"):
            entry = sys.modules.get(stub)
            if entry is not None and not getattr(entry, "__file__", None):
                del sys.modules[stub]


udm = _load_plain("s194e_udm", "opti_oignon/user_data_manager.py")
canon_mod = _load_canonical_store()


def _store(tmp):
    return canon_mod.CanonicalMemoryStore(
        db_path=Path(tmp) / "mem.db", single_user_mode=False
    )


class TestUD03MemoryWiring(unittest.TestCase):
    """Memories legs target the user-scoped canonical store."""

    def _patched(self, store):
        """Patch the canonical accessor; vector store absent."""
        orig_c = udm._get_canonical_memory_store
        orig_v = udm._get_vector_memory_store
        udm._get_canonical_memory_store = lambda: store
        udm._get_vector_memory_store = lambda: None
        return orig_c, orig_v

    def _restore(self, orig_c, orig_v):
        udm._get_canonical_memory_store = orig_c
        udm._get_vector_memory_store = orig_v

    def test_wipe_scopes_to_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            store.add("fact A1", user_id="alice")
            store.add("fact A2", user_id="alice")
            store.add("fact B1", user_id="bob")

            orig = self._patched(store)
            try:
                deleter = udm.UserDataDeleter()
                count = deleter._delete_memories("alice")
            finally:
                self._restore(*orig)

            self.assertEqual(count, 2)
            self.assertEqual(len(store.list(user_id="alice", active_only=False)), 0)
            self.assertEqual(len(store.list(user_id="bob", active_only=False)), 1)

    def test_export_scopes_to_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            store.add("fact A1", user_id="alice")
            store.add("fact B1", user_id="bob")

            orig = self._patched(store)
            try:
                exporter = udm.UserDataExporter()
                mems = exporter._export_memories("alice")
            finally:
                self._restore(*orig)

            self.assertEqual(len(mems), 1)
            self.assertEqual(mems[0].get("user_id"), "alice")

    def test_vector_clear_failure_does_not_break_count(self):
        class _BoomVec:
            def clear(self, *, user_id=None):
                raise RuntimeError("chroma down")

        with tempfile.TemporaryDirectory() as tmp:
            store = _store(tmp)
            store.add("fact A1", user_id="alice")
            orig_c = udm._get_canonical_memory_store
            orig_v = udm._get_vector_memory_store
            udm._get_canonical_memory_store = lambda: store
            udm._get_vector_memory_store = lambda: _BoomVec()
            try:
                count = udm.UserDataDeleter()._delete_memories("alice")
            finally:
                udm._get_canonical_memory_store = orig_c
                udm._get_vector_memory_store = orig_v
            self.assertEqual(count, 1)


class TestUD03ConversationsExplicitSkip(unittest.TestCase):
    """Unscoped conversation store is an explicit skip, not a TypeError."""

    class _RealShapeManager:
        """Mirrors the real ConversationManager surface."""

        def __init__(self):
            self.deleted = []

        def list_conversations(self, limit=50, offset=0):
            return [{"id": "c1"}, {"id": "c2"}]

        def delete_conversation(self, cid):
            self.deleted.append(cid)
            return True

    class _ScopedManager(_RealShapeManager):
        def list_conversations(self, limit=50, offset=0, user_id=None):
            return [{"id": f"{user_id}-c1"}]

    def test_unscoped_store_skipped_without_deleting(self):
        mgr = self._RealShapeManager()
        orig = udm._get_conversation_manager
        udm._get_conversation_manager = lambda: mgr
        try:
            count = udm.UserDataDeleter()._delete_conversations("alice")
            exported = udm.UserDataExporter()._export_conversations("alice")
        finally:
            udm._get_conversation_manager = orig
        self.assertEqual(count, 0)
        self.assertEqual(exported, [])
        self.assertEqual(mgr.deleted, [])

    def test_forward_compatible_with_scoped_store(self):
        mgr = self._ScopedManager()
        orig = udm._get_conversation_manager
        udm._get_conversation_manager = lambda: mgr
        try:
            count = udm.UserDataDeleter()._delete_conversations("alice")
        finally:
            udm._get_conversation_manager = orig
        self.assertEqual(count, 1)
        self.assertEqual(mgr.deleted, ["alice-c1"])


class TestUD03NotCoveredSurface(unittest.TestCase):
    """not_covered appears in results, export metadata, and the schema."""

    def test_delete_results_carry_not_covered(self):
        orig = {
            name: getattr(udm, name)
            for name in (
                "_get_conversation_manager", "_get_memory_manager",
                "_get_canonical_memory_store", "_get_vector_memory_store",
                "_get_rag_store", "_get_plugin_config_store",
                "_get_user_settings_store", "_get_user_key_manager",
                "_get_admin_audit",
            )
        }
        for name in orig:
            setattr(udm, name, lambda: None)
        try:
            results = udm.UserDataDeleter().delete_all("alice")
        finally:
            for name, fn in orig.items():
                setattr(udm, name, fn)
        self.assertIn("not_covered", results)
        self.assertIn(
            "conversations (store not user-scoped)", results["not_covered"]
        )
        self.assertGreaterEqual(len(results["not_covered"]), 8)

    def test_export_metadata_carries_not_covered(self):
        orig = {
            name: getattr(udm, name)
            for name in (
                "_get_conversation_manager", "_get_memory_manager",
                "_get_canonical_memory_store", "_get_rag_store",
                "_get_plugin_config_store", "_get_user_settings_store",
            )
        }
        for name in orig:
            setattr(udm, name, lambda: None)
        try:
            data = udm.UserDataExporter().export("alice")
        finally:
            for name, fn in orig.items():
                setattr(udm, name, fn)
        self.assertIn("not_covered", data["export_metadata"])

    def test_response_model_accepts_field(self):
        src = _read("opti_oignon/api/routes_users.py")
        block = src.split("class DeleteDataResponse")[1].split("class ")[0]
        self.assertIn("not_covered: list[str] = []", block)


if __name__ == "__main__":
    unittest.main()
