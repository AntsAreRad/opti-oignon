#!/usr/bin/env python3
"""Tests for S174 -- conservative curation with the fingerprint sidecar.

The fingerprint gating and idempotence, the deterministic near-duplicate
consolidation, the high-confidence LLM retirement gate, the soft-delete
preference, and the per-user scoping run against a real coordinated MemoryStore
(tmp SQLite canonical store, the vector layer with an injected cosine
collection, and a deterministic embedder). The sidecar path and the curator
chat callable are injected, so no ollama and no configured DATA_DIR are needed,
and the modules load in isolation via spec_from_file_location.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
MEM = OO / "memory"

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
cur = _ensure_mem("curation")


# Near-duplicates: Jaccard("...ecology", "...ecology daily") = 7/8 = 0.875 >= 0.85.
A_TEXT = "The user works in bioinformatics and ecology"
B_TEXT = "The user works in bioinformatics and ecology daily"
C_TEXT = "The user prefers dark mode"


def make_chat_fn(reply: str, *, calls: list | None = None):
    def chat_fn(model=None, messages=None, options=None):
        if calls is not None:
            calls.append({"model": model})
        return {"message": {"content": reply}}

    return chat_fn


def raising_chat_fn(model=None, messages=None, options=None):
    raise RuntimeError("curator unavailable")


# A near-duplicate pair cannot arise from two adds (the add-time Jaccard 0.6
# dedup would merge them). It arises from an edit: a distinct fact is updated to
# collide with an existing one. This helper reproduces that drift so curation,
# whose threshold (0.85) is stricter than the add-time 0.6, has something to
# consolidate. The kept fact (A_TEXT) is added first, so it is the older one.
_PLACEHOLDER = "Completely separate note regarding weather forecasts tonight"

# Explicit orthogonal vectors for every text used, so the add-time cosine dedup
# (0.92) never merges them (low-dim random vectors can collide by chance). The
# only add-time dedup signal is then Jaccard; consolidation drift is created
# purely through an edit, and curation catches it on the Jaccard 0.85 threshold.
_MAPPING = {
    A_TEXT: [1.0, 0.0, 0.0, 0.0],
    B_TEXT: [0.0, 1.0, 0.0, 0.0],
    C_TEXT: [0.0, 0.0, 1.0, 0.0],
    _PLACEHOLDER: [0.0, 0.0, 0.0, 1.0],
}


def _make_near_dup_pair(store, *, user_id=None):
    keep = store.add(A_TEXT, "project", user_id=user_id)[0]
    drifted = store.add(_PLACEHOLDER, "fact", user_id=user_id)[0]
    store.update(drifted.id, text=B_TEXT, user_id=user_id)
    return keep, store.get(drifted.id, user_id=user_id)


def _build(tmp_path, *, single_user_mode=True):
    canon_store = canon.CanonicalMemoryStore(
        tmp_path / "cur.db", single_user_mode=single_user_mode
    )
    embedder = FakeEmbedder(mapping=_MAPPING, dim=4)
    vstore = vec.MemoryVectorStore(
        collection=FakeChromaCollection(name=vec.COLLECTION_NAME), embedder=embedder
    )
    store = ded.MemoryStore(canon_store, vstore)
    return store


@pytest.fixture(autouse=True)
def _reset_singletons():
    cur.reset_curator()
    yield
    cur.reset_curator()


@pytest.fixture
def store(tmp_path):
    return _build(tmp_path)


@pytest.fixture
def curator(tmp_path, store):
    return cur.MemoryCurator(store, state_path=tmp_path / "tidy.json", model="m")


# Module sentinels


class TestModuleSentinels:
    def test_feature_available(self):
        assert cur.FEATURE_AVAILABLE is True

    def test_checkpoint_sentinel(self):
        assert cur.checkpoint_before_apply is True

    def test_singleton_helpers(self):
        assert hasattr(cur, "get_curator")
        assert hasattr(cur, "reset_curator")

    def test_constants(self):
        assert cur.STATE_FILENAME == "memory_tidy_state.json"
        assert cur.CONSOLIDATE_JACCARD >= 0.8
        assert cur.HIGH_CONFIDENCE >= 0.8


# Fingerprint


class TestFingerprint:
    def test_order_free(self, store):
        f1 = store.add(A_TEXT, "project")[0]
        f2 = store.add(C_TEXT, "preference")[0]
        fp_a = cur.compute_fingerprint([f1, f2])
        fp_b = cur.compute_fingerprint([f2, f1])
        assert fp_a == fp_b

    def test_changes_when_set_changes(self, store):
        f1 = store.add(A_TEXT, "project")[0]
        before = cur.compute_fingerprint([f1])
        f2 = store.add(C_TEXT, "preference")[0]
        after = cur.compute_fingerprint([f1, f2])
        assert before != after

    def test_curator_fingerprint_matches_active_set(self, store, curator):
        store.add(A_TEXT, "project")
        facts = store.list(active_only=True)
        assert curator.compute_fingerprint() == cur.compute_fingerprint(facts)


# Fingerprint gating and idempotence


class TestGating:
    def test_first_pass_not_skipped(self, store, curator):
        store.add(A_TEXT, "project")
        report = curator.curate()
        assert report.skipped is False

    def test_second_pass_skipped(self, store, curator):
        store.add(A_TEXT, "project")
        curator.curate()
        report = curator.curate()
        assert report.skipped is True

    def test_force_runs_when_unchanged(self, store, curator):
        store.add(A_TEXT, "project")
        curator.curate()
        report = curator.curate(force=True)
        assert report.skipped is False

    def test_needs_pass_after_add(self, store, curator):
        store.add(A_TEXT, "project")
        curator.curate()
        assert curator.needs_pass() is False
        store.add(C_TEXT, "preference")
        assert curator.needs_pass() is True

    def test_idempotent_after_consolidation(self, store, curator):
        _make_near_dup_pair(store)
        first = curator.curate(use_llm=False)
        assert first.skipped is False and first.consolidated == 1
        second = curator.curate(use_llm=False)
        assert second.skipped is True


# Consolidation


class TestConsolidation:
    def test_consolidates_near_duplicate(self, store, curator):
        _make_near_dup_pair(store)
        assert store.count() == 2
        report = curator.curate(use_llm=False)
        assert report.consolidated == 1
        assert report.retired == 1
        assert store.count() == 1

    def test_soft_delete_keeps_row(self, store, curator):
        keep, drifted = _make_near_dup_pair(store)
        curator.curate(use_llm=False)
        all_rows = store.list(active_only=False)
        assert len(all_rows) == 2
        retired = [r for r in all_rows if not r.active]
        assert len(retired) == 1
        # The older fact (keep) is retained; the drifted near-duplicate is retired.
        assert retired[0].id == drifted.id
        assert store.get(keep.id).active is True

    def test_strength_prefers_higher_use_count(self, store, curator):
        keep, drifted = _make_near_dup_pair(store)
        # Make the drifted fact stronger so it becomes the representative and the
        # originally-kept fact is retired instead.
        store.touch(drifted.id)
        store.touch(drifted.id)
        report = curator.curate(use_llm=False)
        assert report.retired_ids == [keep.id]
        assert store.get(drifted.id).active is True

    def test_distinct_not_consolidated(self, store, curator):
        store.add(A_TEXT, "project")
        store.add(C_TEXT, "preference")
        report = curator.curate(use_llm=False)
        assert report.consolidated == 0
        assert report.retired == 0
        assert store.count() == 2

    def test_find_consolidations_pure(self, store, curator):
        keep, drifted = _make_near_dup_pair(store)
        facts = store.list(active_only=True)
        pairs = curator.find_consolidations(facts)
        assert len(pairs) == 1
        assert pairs[0].keep_id == keep.id
        assert pairs[0].retire_id == drifted.id
        assert pairs[0].score >= cur.CONSOLIDATE_JACCARD


# LLM retirement gate


class TestLLMRetirement:
    def test_high_confidence_retired(self, store, tmp_path):
        a = store.add(A_TEXT, "project")[0]
        store.add(C_TEXT, "preference")
        reply = json.dumps({"retire": [{"id": a.id, "confidence": 0.95}]})
        curator = cur.MemoryCurator(store, state_path=tmp_path / "t.json", chat_fn=make_chat_fn(reply), model="m")
        report = curator.curate(use_llm=True)
        assert a.id in report.retired_ids
        assert store.get(a.id).active is False

    def test_low_confidence_kept(self, store, tmp_path):
        a = store.add(A_TEXT, "project")[0]
        reply = json.dumps({"retire": [{"id": a.id, "confidence": 0.3}]})
        curator = cur.MemoryCurator(store, state_path=tmp_path / "t.json", chat_fn=make_chat_fn(reply), model="m")
        report = curator.curate(use_llm=True)
        assert a.id not in report.retired_ids
        assert store.get(a.id).active is True

    def test_use_llm_false_does_not_call(self, store, tmp_path):
        a = store.add(A_TEXT, "project")[0]
        calls: list = []
        reply = json.dumps({"retire": [{"id": a.id, "confidence": 0.99}]})
        curator = cur.MemoryCurator(store, state_path=tmp_path / "t.json", chat_fn=make_chat_fn(reply, calls=calls), model="m")
        curator.curate(use_llm=False)
        assert calls == []
        assert store.get(a.id).active is True

    def test_raising_curator_is_swallowed(self, store, tmp_path):
        a = store.add(A_TEXT, "project")[0]
        curator = cur.MemoryCurator(store, state_path=tmp_path / "t.json", chat_fn=raising_chat_fn, model="m")
        report = curator.curate(use_llm=True)
        # No crash; the LLM proposed nothing actionable, so the fact survives.
        assert store.get(a.id).active is True
        assert report.skipped is False

    def test_unknown_id_ignored(self, store, tmp_path):
        store.add(A_TEXT, "project")
        reply = json.dumps({"retire": [{"id": "does-not-exist", "confidence": 0.99}]})
        curator = cur.MemoryCurator(store, state_path=tmp_path / "t.json", chat_fn=make_chat_fn(reply), model="m")
        report = curator.curate(use_llm=True)
        assert report.retired == 0


# Per-user scoping


class TestPerUser:
    def test_scopes_to_user(self, tmp_path):
        store = _build(tmp_path, single_user_mode=False)
        _make_near_dup_pair(store, user_id="alice")
        store.add(C_TEXT, "preference", user_id="bob")
        curator = cur.MemoryCurator(store, state_path=tmp_path / "t.json", model="m")
        report = curator.curate(user_id="alice", use_llm=False)
        assert report.consolidated == 1
        assert store.count(user_id="alice") == 1
        assert store.count(user_id="bob") == 1

    def test_state_keyed_per_user(self, tmp_path):
        store = _build(tmp_path, single_user_mode=False)
        store.add(A_TEXT, "project", user_id="alice")
        store.add(C_TEXT, "preference", user_id="bob")
        state_path = tmp_path / "t.json"
        curator = cur.MemoryCurator(store, state_path=state_path, model="m")
        curator.curate(user_id="alice", use_llm=False)
        data = json.loads(state_path.read_text())
        assert "alice" in data
        assert "bob" not in data


# Sidecar


class TestSidecar:
    def test_state_file_written(self, store, tmp_path):
        store.add(A_TEXT, "project")
        state_path = tmp_path / "tidy.json"
        curator = cur.MemoryCurator(store, state_path=state_path, model="m")
        curator.curate(use_llm=False)
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert isinstance(data, dict) and data

    def test_reset_state_all(self, store, curator):
        store.add(A_TEXT, "project")
        curator.curate(use_llm=False)
        assert curator.needs_pass() is False
        curator.reset_state()
        assert curator.needs_pass() is True

    def test_reset_state_one_user(self, tmp_path):
        store = _build(tmp_path, single_user_mode=False)
        store.add(A_TEXT, "project", user_id="alice")
        store.add(C_TEXT, "preference", user_id="bob")
        state_path = tmp_path / "t.json"
        curator = cur.MemoryCurator(store, state_path=state_path, model="m")
        curator.curate(user_id="alice", use_llm=False)
        curator.curate(user_id="bob", use_llm=False)
        curator.reset_state(user_id="alice")
        data = json.loads(state_path.read_text())
        assert "alice" not in data
        assert "bob" in data

    def test_missing_state_returns_empty(self, tmp_path, store):
        curator = cur.MemoryCurator(store, state_path=tmp_path / "absent.json", model="m")
        assert curator._load_state() == {}

    def test_corrupt_state_returns_empty(self, tmp_path, store):
        state_path = tmp_path / "bad.json"
        state_path.write_text("{ not json")
        curator = cur.MemoryCurator(store, state_path=state_path, model="m")
        assert curator._load_state() == {}


# Soft vs hard delete


class TestDeleteMode:
    def test_default_is_soft(self, store, curator):
        _make_near_dup_pair(store)
        curator.curate(use_llm=False)
        assert len(store.list(active_only=False)) == 2

    def test_hard_delete_removes_row(self, store, curator):
        _make_near_dup_pair(store)
        curator.curate(use_llm=False, hard_delete=True)
        assert len(store.list(active_only=False)) == 1


# Never raises


class TestNeverRaises:
    def test_curate_swallows_store_failure(self, tmp_path):
        class Boom:
            def resolve_user(self, user_id=None):
                return "local"

            def list(self, *a, **k):
                raise RuntimeError("store down")

        curator = cur.MemoryCurator(Boom(), state_path=tmp_path / "t.json", model="m")
        report = curator.curate()
        assert report.skipped is True
        assert report.retired == 0
