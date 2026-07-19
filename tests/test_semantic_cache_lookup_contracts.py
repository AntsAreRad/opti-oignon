#!/usr/bin/env python3
"""What the semantic cache promises about which entry may be served.

Sixteen pins, one promise: the cache never serves an answer the current
request did not earn. A candidate below the similarity threshold, in
another conversation, under another model, under another generation
context, or past its time-to-live is refused -- and each refusal is
proven against a control that serves the same entry the moment the
refusing condition is lifted, so a green here is never an empty cache in
disguise. The isolated mode's fail-closed refusal of unscoped traffic,
scoped invalidation, least-recently-used eviction with its under-capacity
guard, and time-to-live expiry round out the set.

These sixteen contracts were first asserted across five earlier suites
that each manufactured a package window by hand. They are re-asserted
here, unchanged in what they pin, on the shared isolation window -- with
the hardened connection layer seeded by a stand-in whose connector opens
a plain throwaway database, because what is under contract here is the
lookup semantics, not the at-rest posture (the fail-secure suite owns
that side). The five earlier files remain in the tree word for word and
are deselected in the runner's registry; nothing was edited or deleted.

Each contract loads the module fresh through the shared window, steers
the mode probe and the embedding backend by rebinding the module-level
seams, and drives only the public surface. No real database under the
repository, no model backend and no network is ever reached.
"""

import math
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.semantic_cache"

# Two unit vectors with cosine similarity exactly 0.5, and stable legacy keys.
_VEC_BASE = [1.0, 0.0]
_VEC_HALF = [0.5, math.sqrt(3) / 2]
_STORED_KEY = "stored-key"
_OTHER_KEY = "other-key"


def _working_connect(path, **kwargs):
    import sqlite3

    kwargs.pop("check_same_thread", None)
    return sqlite3.connect(str(path), check_same_thread=False)


def _load():
    """Load the real module with a working seeded connection layer."""
    data_dir = Path(tempfile.mkdtemp(prefix="cache_data_"))
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = data_dir

    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = _working_connect

    loaded, restore = isolate(
        targets={_TARGET: source("semantic_cache.py")},
        seeded={"opti_oignon.config": cfg, "opti_oignon.db_utils": db_utils},
    )
    return loaded[_TARGET], restore


def _fresh_cache(
    module,
    *,
    threshold=0.92,
    ttl=3600,
    max_entries=100,
    scope="conversation",
    exact=True,
    semantic=True,
    embeddings=True,
):
    """Build an enabled cache on a private temporary database.

    A non-existent config path forces the built-in defaults (no YAML read or
    write), then the relevant switches are set directly for the contract.
    """
    db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
    cache = module.SemanticCache(
        db_path=db_dir / "semantic_cache.db",
        config_path=db_dir / "absent.yaml",
        similarity_threshold=threshold,
        ttl_seconds=ttl,
        max_entries=max_entries,
        scope=scope,
    )
    cache._config["enabled"] = True
    cache._config["exact_match_enabled"] = exact
    cache._config["semantic_match_enabled"] = semantic
    cache.embeddings_available = embeddings
    return cache


def _embedding_table(mapping):
    """Return a fake embedding backend that maps exact query text to a vector."""

    def _fake(text, model=None):
        return mapping.get(text)

    return _fake


# ===========================================================================
# Pins first asserted in tests/test_semantic_cache_contracts.py
# ===========================================================================

def test_candidate_below_similarity_threshold_is_not_served():
    module, restore = _load()
    try:
        cache = _fresh_cache(module, threshold=0.92, scope="conversation")
        stored_query = "How do I sort a list in Python?"
        incoming_query = "What is the capital of France?"  # distinct -> exact miss

        module._is_bulbe = lambda: False
        module._get_embedding = _embedding_table(
            {stored_query: _VEC_BASE, incoming_query: _VEC_HALF}
        )

        cache.put(stored_query, "STORED ANSWER", conversation_id="c1")

        # cosine(incoming, stored) == 0.5 < threshold 0.92 -> the only
        # candidate is too far, so the lookup must miss rather than serve it.
        hit = cache.get(incoming_query, conversation_id="c1")
        assert hit is None, (
            "a candidate below the similarity threshold must not be served"
        )
    finally:
        restore()


def test_isolated_mode_unscoped_query_is_not_served_from_shared_bucket():
    module, restore = _load()
    try:
        cache = _fresh_cache(
            module, scope="global", semantic=False, embeddings=False
        )
        query = "shared bucket question"

        module._get_embedding = _embedding_table({})

        # Seed into the shared global bucket while not isolated.
        module._is_bulbe = lambda: False
        cache.put(query, "GLOBAL ANSWER", conversation_id=None)

        # Control: outside isolation the same query is retrievable, so the
        # refusal below is a real fail-closed and not merely an empty cache.
        served = cache.get(query, conversation_id=None)
        assert served is not None and served.response == "GLOBAL ANSWER"

        # In isolated mode with no conversation in scope: fail closed.
        module._is_bulbe = lambda: True
        hit = cache.get(query, conversation_id=None)
        assert hit is None, "isolated mode must not serve an unscoped global hit"
    finally:
        restore()


def test_stale_context_fingerprint_entry_is_not_served():
    module, restore = _load()
    try:
        cache = _fresh_cache(module, semantic=False, embeddings=False)
        query = "what does the document say?"

        module._is_bulbe = lambda: False
        module._get_embedding = _embedding_table({})

        cache.put(
            query,
            "ANSWER UNDER CONTEXT A",
            conversation_id="c1",
            context_fingerprint="context-A",
        )

        # Same query, but the current generation context differs: must miss.
        hit = cache.get(query, conversation_id="c1", context_fingerprint="context-B")
        assert hit is None, (
            "an entry under a different context fingerprint must not be served"
        )

        # Control: an identical fingerprint serves, proving the entry exists.
        same = cache.get(query, conversation_id="c1", context_fingerprint="context-A")
        assert same is not None and same.response == "ANSWER UNDER CONTEXT A"
    finally:
        restore()


def test_expired_entry_is_not_served_and_is_purged():
    module, restore = _load()
    try:
        clock = {"now": 1000.0}
        module.time = types.SimpleNamespace(time=lambda: clock["now"])

        cache = _fresh_cache(module, ttl=10, semantic=False, embeddings=False)
        query = "time-sensitive question"

        module._is_bulbe = lambda: False
        module._get_embedding = _embedding_table({})

        cache.put(query, "STALE ANSWER", conversation_id="c1")  # created == 1000

        # Advance the clock past the TTL: now - created == 15 >= ttl 10.
        clock["now"] = 1000.0 + 10 + 5
        hit = cache.get(query, conversation_id="c1")
        assert hit is None, "an entry past its TTL must not be served"

        # The expired entry must have been removed on lookup, not just hidden.
        assert cache.entry_count() == 0, "an expired entry must be purged on lookup"
    finally:
        restore()


# ===========================================================================
# Pins first asserted in tests/test_semantic_cache_fallback_contracts.py
# ===========================================================================

def test_below_threshold_embedding_is_not_returned():
    module, restore = _load()
    try:
        cache = _fresh_cache(module, threshold=0.92)
        cache.store_embedding(
            cache_key=_STORED_KEY,
            model="model-a",
            query_text="stored query",
            embedding=_VEC_BASE,
        )

        # cosine(incoming, stored) == 0.5 < threshold 0.92 -> the only
        # candidate is too far, so the search must miss.
        miss = cache.find_similar_by_embedding(
            _VEC_HALF, "model-a", exclude_key=_OTHER_KEY
        )
        assert miss is None, (
            "a candidate below the similarity threshold must not be returned"
        )

        # Control: an identical vector scores 1.0 and is returned, proving the
        # row is present and the search path reaches it.
        served = cache.find_similar_by_embedding(
            _VEC_BASE, "model-a", exclude_key=_OTHER_KEY
        )
        assert served is not None and served.cache_key == _STORED_KEY
    finally:
        restore()


def test_embedding_is_not_returned_across_models():
    module, restore = _load()
    try:
        cache = _fresh_cache(module, threshold=0.92)
        cache.store_embedding(
            cache_key=_STORED_KEY,
            model="model-a",
            query_text="stored query",
            embedding=_VEC_BASE,
        )

        # Same vector (cosine 1.0) but a different model: the model filter
        # must exclude the row despite a perfect vector match.
        miss = cache.find_similar_by_embedding(
            _VEC_BASE, "model-b", exclude_key=_OTHER_KEY
        )
        assert miss is None, (
            "an embedding stored under a different model must not be returned"
        )

        # Control: the same query under the matching model is returned.
        served = cache.find_similar_by_embedding(
            _VEC_BASE, "model-a", exclude_key=_OTHER_KEY
        )
        assert served is not None and served.cache_key == _STORED_KEY
    finally:
        restore()


def test_embedding_is_not_returned_across_context_fingerprints():
    module, restore = _load()
    try:
        cache = _fresh_cache(module, threshold=0.92)
        cache.store_embedding(
            cache_key=_STORED_KEY,
            model="model-a",
            query_text="stored query",
            embedding=_VEC_BASE,
            context_fingerprint="context-A",
        )

        # Same vector and model, but a different generation context is
        # requested: the fingerprint clause must exclude the row.
        miss = cache.find_similar_by_embedding(
            _VEC_BASE, "model-a", exclude_key=_OTHER_KEY,
            context_fingerprint="context-B",
        )
        assert miss is None, (
            "an embedding under a different context fingerprint must not be returned"
        )

        # Control: the matching fingerprint is returned, proving the row exists.
        served = cache.find_similar_by_embedding(
            _VEC_BASE, "model-a", exclude_key=_OTHER_KEY,
            context_fingerprint="context-A",
        )
        assert served is not None and served.cache_key == _STORED_KEY
    finally:
        restore()


# ===========================================================================
# Pins first asserted in tests/test_semantic_cache_management_contracts.py
# ===========================================================================

def test_invalidate_removes_only_the_targeted_conversation():
    module, restore = _load()
    try:
        module._is_bulbe = lambda: False
        cache = _fresh_cache(
            module, max_entries=100, semantic=False, embeddings=False
        )

        cache.put("first question", "ANSWER ONE", conversation_id="c1")
        cache.put("second question", "ANSWER TWO", conversation_id="c2")
        assert cache.entry_count() == 2, "control: both conversations stored"

        removed = cache.invalidate("c1")
        assert removed == 1, "invalidate must remove exactly the targeted conversation"
        assert cache.entry_count() == 1, "only the targeted conversation is removed"

        # The survivor is c2: its exact entry is still served, proving the
        # scope spared it rather than the store being wiped table-wide.
        survivor = cache.get("second question", conversation_id="c2")
        assert survivor is not None and survivor.response == "ANSWER TWO"
    finally:
        restore()


def test_eviction_removes_the_oldest_not_the_newest():
    module, restore = _load()
    try:
        clock = {"now": 1000.0}
        module.time = types.SimpleNamespace(time=lambda: clock["now"])
        module._is_bulbe = lambda: False
        cache = _fresh_cache(
            module, max_entries=2, semantic=False, embeddings=False
        )

        cache.put("oldest question", "OLDEST", conversation_id="c1")  # 1000
        clock["now"] = 1001.0
        cache.put("middle question", "MIDDLE", conversation_id="c1")  # 1001
        clock["now"] = 1002.0
        cache.put("newest question", "NEWEST", conversation_id="c1")  # evicts

        assert cache.entry_count() == 2, "eviction must cap the store at max_entries"

        # The least-recently-used row is evicted; the two most-recent survive.
        assert cache.get("oldest question", conversation_id="c1") is None, (
            "the least-recently-used entry must be evicted"
        )
        assert cache.get("newest question", conversation_id="c1") is not None, (
            "the most-recently-used entry must survive"
        )
        assert cache.get("middle question", conversation_id="c1") is not None
    finally:
        restore()


def test_expire_stale_removes_only_past_ttl_entries():
    module, restore = _load()
    try:
        clock = {"now": 1000.0}
        module.time = types.SimpleNamespace(time=lambda: clock["now"])
        module._is_bulbe = lambda: False
        cache = _fresh_cache(
            module, ttl=10, max_entries=100, semantic=False, embeddings=False
        )

        cache.put("aging question", "OLD", conversation_id="c1")   # created 1000
        clock["now"] = 1008.0
        cache.put("recent question", "NEW", conversation_id="c1")  # created 1008
        assert cache.entry_count() == 2, "control: both entries must be stored"

        # Advance past the TTL for the first entry only: age 15 >= 10 (stale)
        # against age 7 < 10 (fresh).
        clock["now"] = 1015.0
        removed = cache.expire_stale()
        assert removed == 1, "expire_stale must remove only entries past their TTL"
        assert cache.entry_count() == 1, "the fresh entry must survive expire_stale"

        survivor = cache.get("recent question", conversation_id="c1")
        assert survivor is not None and survivor.response == "NEW", (
            "the entry still inside its TTL must remain retrievable"
        )
    finally:
        restore()


def test_no_eviction_while_under_capacity():
    module, restore = _load()
    try:
        module._is_bulbe = lambda: False
        cache = _fresh_cache(
            module, max_entries=5, semantic=False, embeddings=False
        )

        # Two entries, well under the capacity of five. The eviction pass runs
        # after every write; while under capacity it must remove nothing.
        # Without the guard the excess count is negative and the delete's
        # negative limit drops every row, so this also proves a below-limit
        # cache is never wiped.
        cache.put("kept question one", "KEEP ONE", conversation_id="c1")
        cache.put("kept question two", "KEEP TWO", conversation_id="c1")

        assert cache.entry_count() == 2, "a store under capacity must not be evicted"
        assert cache.get("kept question one", conversation_id="c1") is not None
        assert cache.get("kept question two", conversation_id="c1") is not None
    finally:
        restore()


# ===========================================================================
# Pins first asserted in tests/test_semantic_cache_no_model_lookup_contracts.py
# ===========================================================================

def test_candidate_in_other_conversation_is_not_served_without_model_filter():
    module, restore = _load()
    try:
        cache = _fresh_cache(module, threshold=0.92, scope="conversation")
        stored_query = "How do I reverse a list in Python?"
        incoming_query = "What is the capital of France?"  # distinct -> exact miss

        module._is_bulbe = lambda: False
        # Both texts map to the same unit vector, so any row the SQL
        # pre-filter returns is an exact cosine match (1.0 >= threshold).
        module._get_embedding = _embedding_table(
            {stored_query: _VEC_BASE, incoming_query: _VEC_BASE}
        )

        # The candidate lives in conversation c2 and carries a concrete model.
        cache.put(
            stored_query,
            "ANSWER FROM CONVERSATION TWO",
            model="model-alpha",
            conversation_id="c2",
        )

        # A no-model lookup in conversation c1: the SQL conversation scope
        # excludes the c2 row before the cosine comparison.
        hit = cache.get(incoming_query, conversation_id="c1", model="")
        assert hit is None, "a candidate in another conversation must not be served"

        # Control: the identical no-model lookup in the matching conversation
        # serves the candidate, proving the refusal above is a real
        # conversation scope and not an empty cache, a below-threshold miss,
        # or a model-filter artifact.
        served = cache.get(incoming_query, conversation_id="c2", model="")
        assert served is not None
        assert served.response == "ANSWER FROM CONVERSATION TWO"
    finally:
        restore()


def test_expired_candidate_is_filtered_not_served_on_no_model_path():
    module, restore = _load()
    try:
        clock = {"now": 1000.0}
        module.time = types.SimpleNamespace(time=lambda: clock["now"])

        cache = _fresh_cache(module, threshold=0.92, ttl=10, scope="conversation")
        stored_query = "How do I read a CSV in Python?"
        incoming_query = "How tall is Mount Everest?"  # distinct -> exact miss

        module._is_bulbe = lambda: False
        module._get_embedding = _embedding_table(
            {stored_query: _VEC_BASE, incoming_query: _VEC_BASE}
        )

        cache.put(
            stored_query,
            "STALE ANSWER",
            model="model-alpha",
            conversation_id="c1",
        )  # created_at == 1000

        # Control: while still inside the TTL the candidate is served through
        # the no-model fuzzy path, so the refusal below is a real age filter
        # and not an empty cache.
        fresh = cache.get(incoming_query, conversation_id="c1", model="")
        assert fresh is not None and fresh.response == "STALE ANSWER"

        # Advance the clock past the TTL: now - created == 15 >= ttl 10.
        clock["now"] = 1000.0 + 10 + 5
        hit = cache.get(incoming_query, conversation_id="c1", model="")
        assert hit is None, (
            "a candidate past its TTL must not be served on the no-model path"
        )

        # The no-model branch filters the stale row out of the ranking but
        # does not purge it, so it stays available to a later query that lands
        # inside a refreshed window.
        assert cache.entry_count() == 1, (
            "the no-model path must filter, not purge, the stale row"
        )
    finally:
        restore()


# ===========================================================================
# Pins first asserted in tests/test_semantic_cache_semantic_tier_contracts.py
# ===========================================================================

def test_candidate_under_other_model_is_not_served():
    module, restore = _load()
    try:
        cache = _fresh_cache(module, threshold=0.92, scope="conversation")
        stored_query = "How do I reverse a string in Python?"
        incoming_query = "What is the tallest mountain on Earth?"  # exact miss

        module._is_bulbe = lambda: False
        # Both texts map to the same unit vector, so any row the SQL
        # pre-filter returns is an exact cosine match (1.0 >= threshold).
        module._get_embedding = _embedding_table(
            {stored_query: _VEC_BASE, incoming_query: _VEC_BASE}
        )

        cache.put(
            stored_query,
            "ANSWER FROM MODEL ALPHA",
            model="model-alpha",
            conversation_id="c1",
        )

        # The request is routed to a different model: the SQL scope filter
        # excludes the alpha-model row before the cosine comparison.
        hit = cache.get(incoming_query, conversation_id="c1", model="model-beta")
        assert hit is None, (
            "a candidate stored under another model must not be served"
        )

        # Control: the identical request routed to the matching model does
        # serve the candidate, proving the refusal above is a real scope
        # filter and not an empty cache or a below-threshold miss.
        served = cache.get(incoming_query, conversation_id="c1", model="model-alpha")
        assert served is not None and served.response == "ANSWER FROM MODEL ALPHA"
    finally:
        restore()


def test_expired_candidate_is_filtered_not_served_by_fuzzy_tier():
    module, restore = _load()
    try:
        clock = {"now": 1000.0}
        module.time = types.SimpleNamespace(time=lambda: clock["now"])

        cache = _fresh_cache(module, threshold=0.92, ttl=10, scope="conversation")
        stored_query = "How do I open a file in Python?"
        incoming_query = "Which planet is closest to the Sun?"  # exact miss

        module._is_bulbe = lambda: False
        module._get_embedding = _embedding_table(
            {stored_query: _VEC_BASE, incoming_query: _VEC_BASE}
        )

        cache.put(
            stored_query,
            "STALE ANSWER",
            model="model-alpha",
            conversation_id="c1",
        )  # created_at == 1000

        # Control: while still inside the TTL the candidate is served through
        # the fuzzy tier, so the refusal below is a real age filter and not an
        # empty cache. The distinct query text means the exact tier misses.
        fresh = cache.get(incoming_query, conversation_id="c1", model="model-alpha")
        assert fresh is not None and fresh.response == "STALE ANSWER"

        # Advance the clock past the TTL: now - created == 15 >= ttl 10.
        clock["now"] = 1000.0 + 10 + 5
        hit = cache.get(incoming_query, conversation_id="c1", model="model-alpha")
        assert hit is None, (
            "a candidate past its TTL must not be served by the fuzzy tier"
        )

        # The fuzzy tier filters the stale row out of the ranking but --
        # unlike the exact tier -- does not purge it, so it stays available to
        # a later query that lands inside a refreshed window.
        assert cache.entry_count() == 1, (
            "the fuzzy tier must filter, not purge, the stale row"
        )
    finally:
        restore()


def test_stale_context_fingerprint_candidate_is_skipped():
    module, restore = _load()
    try:
        cache = _fresh_cache(module, threshold=0.92, scope="conversation")
        stored_query = "What does the attached report conclude?"
        incoming_query = "How many moons does Jupiter have?"  # exact miss

        module._is_bulbe = lambda: False
        module._get_embedding = _embedding_table(
            {stored_query: _VEC_BASE, incoming_query: _VEC_BASE}
        )

        cache.put(
            stored_query,
            "ANSWER UNDER CONTEXT ALPHA",
            model="model-alpha",
            conversation_id="c1",
            context_fingerprint="context-alpha",
        )

        # The current generation context differs from the one the candidate
        # was stored under: it is skipped while ranking, so the fuzzy tier
        # misses even though the embedding is an exact match.
        hit = cache.get(
            incoming_query,
            conversation_id="c1",
            model="model-alpha",
            context_fingerprint="context-beta",
        )
        assert hit is None, (
            "a candidate under a different context fingerprint must be skipped"
        )

        # Control: the same request under the matching fingerprint serves the
        # candidate, proving it is present and the skip above is a real gate.
        same = cache.get(
            incoming_query,
            conversation_id="c1",
            model="model-alpha",
            context_fingerprint="context-alpha",
        )
        assert same is not None and same.response == "ANSWER UNDER CONTEXT ALPHA"
    finally:
        restore()
