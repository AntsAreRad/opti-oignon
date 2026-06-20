#!/usr/bin/env python3
"""Tests for S174 -- S66 dual-layer memory integration.

The working layer (the compressed, budgeted block injected into the prompt) and
the recovery path (the full uncompressed archive, still searchable) are
exercised against a real coordinated store in isolation. The wiring into the
prompt-assembly path (executor._inject_memory) is checked by file content, since
the executor is not importable in the sandbox.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
MEM = OO / "memory"
EXECUTOR = OO / "executor.py"
SPEC = ROOT / "ODYSSEUS_SPEC.md"

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
_ensure_real("context_window")
canon = _ensure_mem("canonical_store")
vec = _ensure_mem("vector_store")
ded = _ensure_mem("dedup")
ret = _ensure_mem("retrieval")


def _build(tmp_path, *, single_user_mode=True):
    canon_store = canon.CanonicalMemoryStore(
        tmp_path / "dl.db", single_user_mode=single_user_mode
    )
    vstore = vec.MemoryVectorStore(
        collection=FakeChromaCollection(name=vec.COLLECTION_NAME),
        embedder=FakeEmbedder(dim=16),
    )
    store = ded.MemoryStore(canon_store, vstore)
    retriever = ret.MemoryRetriever(canon_store, vstore)
    return retriever, store, canon_store


@pytest.fixture(autouse=True)
def _reset_singletons():
    ret.reset_retriever()
    yield
    ret.reset_retriever()


@pytest.fixture
def bundle(tmp_path):
    return _build(tmp_path)


def _seed_distinct(store, n=5):
    topics = ["gardening", "cooking", "cycling", "painting", "reading", "hiking"]
    ids = []
    for i in range(n):
        rec = store.add(f"The user enjoys {topics[i % len(topics)]} topic{i}", "preference")[0]
        ids.append(rec.id)
    return ids


# Module surface


class TestSurface:
    def test_dual_layer_dataclass(self):
        assert hasattr(ret, "DualLayerMemory")

    def test_module_helpers(self):
        assert hasattr(ret, "working_memory_block")
        assert hasattr(ret, "recover_memories")

    def test_retriever_methods(self):
        for name in ("recent_memories", "working_block", "recover", "assemble_dual_layer"):
            assert hasattr(ret.MemoryRetriever, name)


# Recent memories (working layer without a query)


class TestRecentMemories:
    def test_only_active(self, bundle):
        retriever, store, _ = bundle
        ids = _seed_distinct(store, 3)
        store.soft_delete(ids[0])
        recents = retriever.recent_memories(top_n=10)
        assert all(m.id != ids[0] for m in recents)
        assert len(recents) == 2

    def test_orders_by_use_count(self, bundle):
        retriever, store, _ = bundle
        ids = _seed_distinct(store, 3)
        store.touch(ids[1])
        store.touch(ids[1])
        recents = retriever.recent_memories(top_n=10)
        assert recents[0].id == ids[1]

    def test_respects_top_n(self, bundle):
        retriever, store, _ = bundle
        _seed_distinct(store, 5)
        assert len(retriever.recent_memories(top_n=2)) == 2


# Working block (compressed layer)


class TestWorkingBlock:
    def test_empty_when_no_memories(self, bundle):
        retriever, _, _ = bundle
        assert retriever.working_block() == ""

    def test_block_lists_recent(self, bundle):
        retriever, store, _ = bundle
        _seed_distinct(store, 3)
        block = retriever.working_block(max_tokens=512)
        assert block.startswith("Relevant memories:")
        assert block.count("\n") == 3  # header + 3 facts

    def test_budget_truncates(self, bundle):
        retriever, store, _ = bundle
        _seed_distinct(store, 6)
        tight = retriever.working_block(max_tokens=25)
        wide = retriever.working_block(max_tokens=512)
        assert tight.count("\n") < wide.count("\n")

    def test_query_relevant_selection(self, bundle):
        retriever, store, _ = bundle
        _seed_distinct(store, 5)
        store.add("The user studies marine biology in Panama", "project")
        block = retriever.working_block("Where does the user study marine biology?", max_tokens=512)
        assert "marine biology" in block

    def test_block_is_unwrapped(self, bundle):
        retriever, store, _ = bundle
        _seed_distinct(store, 3)
        block = retriever.working_block(max_tokens=512)
        # The agent applies the untrusted-context wrapping in S175; the block
        # carries no wrapper here.
        assert "untrusted" not in block.lower()
        assert "```" not in block

    def test_mark_used_touches(self, bundle):
        retriever, store, canon_store = bundle
        ids = _seed_distinct(store, 2)
        retriever.working_block(max_tokens=512, mark_used=True)
        assert all(canon_store.get(i).use_count >= 1 for i in ids)


# Recovery path (full archive stays searchable)


class TestRecovery:
    def test_recover_finds_fact_dropped_from_block(self, bundle):
        retriever, store, _ = bundle
        # The unique fact is oldest and unused, so the budgeted working block
        # (top_n=2 of the touched facts) drops it; recovery still finds it.
        unique = store.add("The user enjoys kayaking on alpine rivers", "preference")[0]
        ids = _seed_distinct(store, 4)
        for i in ids:
            store.touch(i)
        block = retriever.working_block(top_n=2, max_tokens=512)
        assert "kayaking" not in block
        recovered = retriever.recover("kayaking", top_n=10)
        assert any(m.id == unique.id for m in recovered)

    def test_recover_only_active(self, bundle):
        retriever, store, _ = bundle
        rec = store.add("The user enjoys kayaking on alpine rivers", "preference")[0]
        store.soft_delete(rec.id)
        assert retriever.recover("kayaking", top_n=10) == []

    def test_recover_scoped_per_user(self, tmp_path):
        retriever, store, _ = _build(tmp_path, single_user_mode=False)
        store.add("The user enjoys kayaking", "preference", user_id="alice")
        bob_hits = retriever.recover("kayaking", user_id="bob", top_n=10)
        alice_hits = retriever.recover("kayaking", user_id="alice", top_n=10)
        assert bob_hits == []
        assert len(alice_hits) == 1


# Dual-layer assembly (the S66 invariant)


class TestDualLayer:
    def test_full_archive_larger_than_block(self, bundle):
        retriever, store, _ = bundle
        _seed_distinct(store, 6)
        dl = retriever.assemble_dual_layer(max_tokens=25)
        assert dl.total_active == 6
        assert len(dl.selected_ids) < dl.total_active
        assert dl.block

    def test_dropped_detail_recoverable(self, bundle):
        retriever, store, _ = bundle
        unique = store.add("The user enjoys kayaking on alpine rivers", "preference")[0]
        ids = _seed_distinct(store, 4)
        for i in ids:
            store.touch(i)
        dl = retriever.assemble_dual_layer(top_n=2, max_tokens=512)
        # The unique fact is in the full archive but not the working selection,
        # and is recoverable from the still-searchable archive.
        assert unique.id not in dl.selected_ids
        assert any(m.id == unique.id for m in retriever.recover("kayaking", top_n=10))

    def test_block_unwrapped_in_assembly(self, bundle):
        retriever, store, _ = bundle
        _seed_distinct(store, 3)
        dl = retriever.assemble_dual_layer(max_tokens=512)
        assert "untrusted" not in dl.block.lower()


# Module-level convenience over the singleton (injected, no real stores built)


class TestModuleHelpers:
    def test_working_memory_block_delegates(self, bundle):
        retriever, store, _ = bundle
        _seed_distinct(store, 3)
        ret._retriever = retriever
        block = ret.working_memory_block(max_tokens=512)
        assert block.startswith("Relevant memories:")

    def test_recover_memories_delegates(self, bundle):
        retriever, store, _ = bundle
        unique = store.add("The user enjoys kayaking on alpine rivers", "preference")[0]
        ret._retriever = retriever
        recovered = ret.recover_memories("kayaking", top_n=10)
        assert any(m.id == unique.id for m in recovered)


# Wiring into the prompt-assembly path (file content; executor not importable)


class TestExecutorWiring:
    def _executor_text(self) -> str:
        return EXECUTOR.read_text(encoding="utf-8")

    def test_imports_dual_layer_block(self):
        text = self._executor_text()
        assert "working_memory_block" in text
        assert "DUAL_LAYER_MEMORY_AVAILABLE" in text

    def test_inject_memory_takes_question(self):
        text = self._executor_text()
        assert "def _inject_memory(self, system_prompt: str, question" in text

    def test_inject_prefers_dual_layer_then_legacy(self):
        text = self._executor_text()
        # The dual-layer block is tried first; the legacy flat block is the
        # fallback when the new layer yields nothing.
        dual_at = text.find("_working_memory_block(question")
        legacy_at = text.find("_memory_manager.format_for_prompt(max_tokens=500)")
        assert dual_at != -1 and legacy_at != -1
        assert dual_at < legacy_at

    def test_call_site_passes_question(self):
        text = self._executor_text()
        assert "self._inject_memory(system_prompt, refined_question)" in text


# Spec alignment (file content)


class TestSpecAlignment:
    def test_spec_describes_dual_layer(self):
        text = SPEC.read_text(encoding="utf-8").lower()
        assert "dual-layer" in text or "dual layer" in text
        assert "s66" in text

    def test_retrieval_module_has_dual_layer(self):
        text = (MEM / "retrieval.py").read_text(encoding="utf-8")
        assert "working_block" in text
        assert "def recover" in text
        assert "DualLayerMemory" in text
