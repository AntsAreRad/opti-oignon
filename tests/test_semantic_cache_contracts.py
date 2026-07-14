#!/usr/bin/env python3
"""Security contracts for the semantic response cache.

These tests pin the serve-versus-refuse behaviour of the live ``get`` path so
that a regression which loosens any one gate is caught. Each contract seeds the
cache with an entry that *would* be served if its gate were removed, then proves
that the intact gate withholds it:

  * a candidate below the cosine similarity threshold is never served, so an
    unrelated or poisoned entry cannot be returned through a loose fuzzy match;
  * in the network-isolated (Bulbe) mode an unscoped query never serves a hit
    from the shared global bucket, so one conversation's response cannot bleed
    into another;
  * an entry stored under a different generation-context fingerprint is not
    served when a fingerprint is supplied, so a response built under stale
    retrieval/memory context is never replayed;
  * an entry past its time-to-live is purged and missed, never served stale.

The module is loaded in isolation. The embedding backend and the security-mode
probe are replaced with deterministic fakes, and the real cosine arithmetic is
kept, so every gate decision is reproducible without a model server. Each test
receives its own freshly executed module instance, which keeps the fakes from
leaking between tests.
"""

import importlib.util
import math
import sys
import tempfile
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TARGET = _REPO_ROOT / "opti_oignon" / "semantic_cache.py"

# A unit vector and a second unit vector 60 degrees away: their cosine is
# exactly 0.5, comfortably below the default 0.92 threshold.
_VEC_BASE = [1.0, 0.0]
_VEC_HALF = [0.5, math.sqrt(3.0) / 2.0]


def _load_semantic_cache():
    """Flat-load the cache module with its package dependencies stubbed.

    A fresh module object is created on every call so that per-test monkey
    patches (embedding backend, mode probe, clock) stay isolated.
    """
    pkg = sys.modules.get("opti_oignon")
    if pkg is None:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = []
        sys.modules["opti_oignon"] = pkg
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = Path(tempfile.mkdtemp(prefix="cache_data_"))
    sys.modules["opti_oignon.config"] = cfg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.semantic_cache", str(_TARGET)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.semantic_cache"] = module
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def module():
    """The module under contract, loaded fresh for each clause.

    Without this, every clause in this file is an ERROR under pytest: the
    ``module`` parameter reads as a request for a fixture that does not exist,
    and pytest refuses the test before a single assertion runs. The clauses were
    written for the __main__ runner below, which passes the module positionally
    -- so they PASS there and have never once executed under pytest, the runner
    whose junitxml is this project's authority. A contract that only one runner
    can see is not a contract; it is a file that looks like one.
    """
    # The loader registers modules under the project namespace and hands none of
    # them back. That leak was DORMANT only because these clauses never ran: the
    # missing fixture made every one of them an ERROR under pytest. Wiring them
    # in without this would trade twenty-six errors for a fresh contamination
    # front -- a stub ``opti_oignon`` package sitting in the cache, served to
    # every suite that imports the project after this one.
    prefix = "opti_oignon"
    saved = {
        key: value for key, value in sys.modules.items()
        if key == prefix or key.startswith(prefix + ".")
    }
    saved_meta_path = list(sys.meta_path)
    try:
        yield _load_semantic_cache()
    finally:
        for key in [
            k for k in sys.modules
            if k == prefix or k.startswith(prefix + ".")
        ]:
            del sys.modules[key]
        sys.modules.update(saved)
        sys.meta_path[:] = saved_meta_path



def _fresh_cache(
    module,
    *,
    threshold=0.92,
    ttl=3600,
    scope="conversation",
    exact=True,
    semantic=True,
    embeddings=True,
):
    """Build a cache on a private temporary database with an explicit config.

    A non-existent config path forces the built-in defaults (no YAML read or
    write), then the relevant switches are set directly for the contract.
    """
    db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
    absent_config = db_dir / "absent.yaml"
    cache = module.SemanticCache(
        db_path=db_dir / "semantic_cache.db",
        config_path=absent_config,
        similarity_threshold=threshold,
        ttl_seconds=ttl,
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


# ---------------------------------------------------------------------------
# Contract 1 -- a below-threshold candidate is never served (no fuzzy over-match)
# ---------------------------------------------------------------------------

def test_candidate_below_similarity_threshold_is_not_served(module):
    cache = _fresh_cache(module, threshold=0.92, scope="conversation")
    stored_query = "How do I sort a list in Python?"
    incoming_query = "What is the capital of France?"  # distinct -> exact tier misses

    module._is_bulbe = lambda: False
    module._get_embedding = _embedding_table(
        {stored_query: _VEC_BASE, incoming_query: _VEC_HALF}
    )

    cache.put(stored_query, "STORED ANSWER", conversation_id="c1")

    # cosine(incoming, stored) == 0.5 < threshold 0.92 -> the only candidate is
    # too far, so the lookup must miss rather than return the stored answer.
    hit = cache.get(incoming_query, conversation_id="c1")
    assert hit is None, "a candidate below the similarity threshold must not be served"


# ---------------------------------------------------------------------------
# Contract 2 -- isolated mode does not serve an unscoped global hit (no bleed)
# ---------------------------------------------------------------------------

def test_isolated_mode_unscoped_query_is_not_served_from_shared_bucket(module):
    cache = _fresh_cache(module, scope="global", semantic=False, embeddings=False)
    query = "shared bucket question"

    module._get_embedding = _embedding_table({})

    # Seed into the shared global bucket while not isolated.
    module._is_bulbe = lambda: False
    cache.put(query, "GLOBAL ANSWER", conversation_id=None)

    # Control: outside isolation the same query is retrievable, so the refusal
    # below is a real fail-closed and not merely an empty cache.
    served = cache.get(query, conversation_id=None)
    assert served is not None and served.response == "GLOBAL ANSWER"

    # In isolated mode with no conversation in scope the lookup must fail closed.
    module._is_bulbe = lambda: True
    hit = cache.get(query, conversation_id=None)
    assert hit is None, "isolated mode must not serve an unscoped global hit"


# ---------------------------------------------------------------------------
# Contract 3 -- a stale generation-context entry is not served (fingerprint)
# ---------------------------------------------------------------------------

def test_stale_context_fingerprint_entry_is_not_served(module):
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
    assert hit is None, "an entry under a different context fingerprint must not be served"

    # Control: an identical fingerprint serves, proving the entry is present.
    same = cache.get(query, conversation_id="c1", context_fingerprint="context-A")
    assert same is not None and same.response == "ANSWER UNDER CONTEXT A"


# ---------------------------------------------------------------------------
# Contract 4 -- an expired entry is purged and missed (no serve-stale)
# ---------------------------------------------------------------------------

def test_expired_entry_is_not_served_and_is_purged(module):
    clock = {"now": 1000.0}
    module.time = types.SimpleNamespace(time=lambda: clock["now"])

    cache = _fresh_cache(module, ttl=10, semantic=False, embeddings=False)
    query = "time-sensitive question"

    module._is_bulbe = lambda: False
    module._get_embedding = _embedding_table({})

    cache.put(query, "STALE ANSWER", conversation_id="c1")  # created_at == 1000

    # Advance the clock past the TTL: now - created == 15 >= ttl 10.
    clock["now"] = 1000.0 + 10 + 5
    hit = cache.get(query, conversation_id="c1")
    assert hit is None, "an entry past its TTL must not be served"

    # The expired entry must have been removed on lookup, not merely hidden.
    assert cache.entry_count() == 0, "an expired entry must be purged on lookup"


_TESTS = [
    test_candidate_below_similarity_threshold_is_not_served,
    test_isolated_mode_unscoped_query_is_not_served_from_shared_bucket,
    test_stale_context_fingerprint_entry_is_not_served,
    test_expired_entry_is_not_served_and_is_purged,
]


def _main(argv):
    selected = set(argv) if argv else {t.__name__ for t in _TESTS}
    failures = 0
    for test in _TESTS:
        if test.__name__ not in selected:
            continue
        module = _load_semantic_cache()
        try:
            test(module)
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            print(f"FAIL {test.__name__} - {exc}")
            failures += 1
        except Exception as exc:  # pragma: no cover - surfaced as a failure
            print(f"ERROR {test.__name__} - {exc!r}")
            failures += 1
    print(f"{'-' * 48}\n{len(selected)} selected, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
