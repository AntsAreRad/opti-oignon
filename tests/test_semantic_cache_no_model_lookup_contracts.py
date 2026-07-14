#!/usr/bin/env python3
"""Security contracts for the fuzzy serve tier on the no-model lookup path.

The second-tier (embedding-similarity) lookup branches on whether the request
carries a model filter. When the filter is empty -- the single-turn path that
asks for any model -- a distinct SQL statement runs that scopes only by
conversation and time-to-live. The model-scoped branch is pinned elsewhere;
here every request passes an empty model so that branch is skipped and only the
no-model branch's own gates can withhold a candidate. Each contract seeds a
candidate whose embedding is an *exact* match (cosine 1.0, well above the
threshold), so only the gate under test can keep it from being served, then
proves the intact gate withholds it:

  * a candidate stored in another conversation is filtered out at the SQL layer
    before any cosine comparison, so a no-model lookup in one conversation is
    never served another conversation's response even though the embedding is
    an exact match;
  * a candidate whose age exceeds the time-to-live is filtered out at the SQL
    layer before any cosine comparison on the no-model path, and -- as on the
    model-scoped branch -- the row is left in place rather than purged, so it
    stays available to a later query that lands inside a refreshed window.

To reach the fuzzy tier the incoming query text always differs from the stored
query text (the exact tier misses on the hash) while both map to the same
embedding vector, so the candidate is a genuine cosine match that only the gate
under test can exclude. The distinct query text also keeps the exact tier's
expiry purge -- which is keyed on the incoming query hash -- from removing the
seeded row, so the age contract exercises the no-model branch's own filter
rather than a side effect of the exact tier. A concrete model is stored on the
seeded row while the lookup carries none, matching the real single-turn path
where model-tagged entries are queried without a model filter.

The module is loaded in isolation. The embedding backend and the security-mode
probe are replaced with deterministic fakes and the real cosine arithmetic is
kept, so every gate decision is reproducible without a model server. Each test
receives its own freshly executed module instance, which keeps the fakes from
leaking between tests.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TARGET = _REPO_ROOT / "opti_oignon" / "semantic_cache.py"

# A single unit vector: mapping two distinct query texts to it makes their
# cosine exactly 1.0, comfortably above the default 0.92 threshold, so the only
# thing that can withhold the candidate is the eligibility gate under test.
_VEC_BASE = [1.0, 0.0]


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
# Contract 1 -- a candidate in another conversation is filtered before ranking
# ---------------------------------------------------------------------------

def test_candidate_in_other_conversation_is_not_served_without_model_filter(module):
    cache = _fresh_cache(module, threshold=0.92, scope="conversation")
    stored_query = "How do I reverse a list in Python?"
    incoming_query = "What is the capital of France?"  # distinct -> exact miss

    module._is_bulbe = lambda: False
    # Both texts map to the same unit vector, so any row the SQL pre-filter
    # returns is an exact cosine match (1.0 >= threshold).
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

    # A no-model lookup in conversation c1: the SQL conversation scope excludes
    # the c2 row before the cosine comparison, so the fuzzy tier misses.
    hit = cache.get(incoming_query, conversation_id="c1", model="")
    assert hit is None, "a candidate in another conversation must not be served"

    # Control: the identical no-model lookup in the matching conversation serves
    # the candidate, proving the refusal above is a real conversation scope and
    # not an empty cache, a below-threshold miss, or a model-filter artifact.
    served = cache.get(incoming_query, conversation_id="c2", model="")
    assert served is not None and served.response == "ANSWER FROM CONVERSATION TWO"


# ---------------------------------------------------------------------------
# Contract 2 -- a candidate past its TTL is filtered before ranking (not purged)
# ---------------------------------------------------------------------------

def test_expired_candidate_is_filtered_not_served_on_no_model_path(module):
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

    # Control: while still inside the TTL the candidate is served through the
    # no-model fuzzy path, so the refusal below is a real age filter and not an
    # empty cache. The distinct query text means the exact tier misses.
    fresh = cache.get(incoming_query, conversation_id="c1", model="")
    assert fresh is not None and fresh.response == "STALE ANSWER"

    # Advance the clock past the TTL: now - created == 15 >= ttl 10.
    clock["now"] = 1000.0 + 10 + 5
    hit = cache.get(incoming_query, conversation_id="c1", model="")
    assert hit is None, "a candidate past its TTL must not be served on the no-model path"

    # The no-model branch filters the stale row out of the ranking but does not
    # purge it, so it stays available to a later query that lands inside a
    # refreshed window.
    assert cache.entry_count() == 1, "the no-model path must filter, not purge, the stale row"


_TESTS = [
    test_candidate_in_other_conversation_is_not_served_without_model_filter,
    test_expired_candidate_is_filtered_not_served_on_no_model_path,
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
