#!/usr/bin/env python3
"""Security contracts for the embedding-similarity fallback of the response cache.

These tests pin the serve-versus-refuse behaviour of the embedding-similarity
search that backs the exact cache on a miss -- the path reached through
``get_with_fallback`` -> ``find_similar`` -> ``find_similar_by_embedding`` over
the dedicated embeddings table. The two-tier ``get`` path is pinned elsewhere;
this file closes the older fallback search, which serves real responses on an
exact miss and otherwise carries no serve/refuse coverage.

Each contract seeds an embedding that *would* be returned if its gate were
removed, then proves the intact gate withholds it while a matching control is
still returned (so every refusal is a real fail-closed, not an empty table):

  * a candidate below the cosine similarity threshold is never returned, so an
    unrelated or poisoned embedding cannot surface through a loose fuzzy match;
  * an embedding stored under a different model is never returned for another
    model, so one model's cached answer cannot be served for a different model;
  * an embedding stored under a different generation-context fingerprint is not
    returned when a fingerprint is supplied, so a response built under stale
    retrieval/memory context is never replayed.

The module is loaded in isolation with its package dependencies stubbed. Real
cosine arithmetic is kept and vectors are supplied directly, so every gate
decision is reproducible without a model server. Each test receives its own
freshly executed module instance so state cannot leak between tests.
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
# exactly 0.5, comfortably below the default 0.92 threshold. An identical
# vector scores 1.0 and is used as the matching control.
_VEC_BASE = [1.0, 0.0]
_VEC_HALF = [0.5, math.sqrt(3.0) / 2.0]

_STORED_KEY = "stored-key"
# An excluded key that never collides with the stored row, so exclusion by key
# is never the reason a lookup misses -- only the gate under test can withhold.
_OTHER_KEY = "other-key"


def _load_semantic_cache():
    """Flat-load the cache module with its package dependencies stubbed.

    A fresh module object is created on every call so per-test state stays
    isolated.
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



def _fresh_cache(module, *, threshold=0.92):
    """Build an enabled cache on a private temporary database.

    A non-existent config path forces the built-in defaults (no YAML read or
    write); the cache is then enabled so the fallback search runs.
    """
    db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
    absent_config = db_dir / "absent.yaml"
    cache = module.SemanticCache(
        db_path=db_dir / "semantic_cache.db",
        config_path=absent_config,
        similarity_threshold=threshold,
    )
    cache._config["enabled"] = True
    return cache


# ---------------------------------------------------------------------------
# Contract 1 -- a below-threshold candidate is never returned (no fuzzy over-match)
# ---------------------------------------------------------------------------

def test_below_threshold_embedding_is_not_returned(module):
    cache = _fresh_cache(module, threshold=0.92)
    cache.store_embedding(
        cache_key=_STORED_KEY,
        model="model-a",
        query_text="stored query",
        embedding=_VEC_BASE,
    )

    # cosine(incoming, stored) == 0.5 < threshold 0.92 -> the only candidate is
    # too far, so the search must miss rather than return the stored embedding.
    miss = cache.find_similar_by_embedding(
        _VEC_HALF, "model-a", exclude_key=_OTHER_KEY
    )
    assert miss is None, "a candidate below the similarity threshold must not be returned"

    # Control: an identical vector scores 1.0 and is returned, proving the row
    # is present and the search path reaches it.
    served = cache.find_similar_by_embedding(
        _VEC_BASE, "model-a", exclude_key=_OTHER_KEY
    )
    assert served is not None and served.cache_key == _STORED_KEY


# ---------------------------------------------------------------------------
# Contract 2 -- an embedding under a different model is not returned (model bind)
# ---------------------------------------------------------------------------

def test_embedding_is_not_returned_across_models(module):
    cache = _fresh_cache(module, threshold=0.92)
    cache.store_embedding(
        cache_key=_STORED_KEY,
        model="model-a",
        query_text="stored query",
        embedding=_VEC_BASE,
    )

    # Same vector (cosine 1.0) but a different model: the model filter must
    # exclude the row, so the search misses despite a perfect vector match.
    miss = cache.find_similar_by_embedding(
        _VEC_BASE, "model-b", exclude_key=_OTHER_KEY
    )
    assert miss is None, "an embedding stored under a different model must not be returned"

    # Control: the same query under the matching model is returned.
    served = cache.find_similar_by_embedding(
        _VEC_BASE, "model-a", exclude_key=_OTHER_KEY
    )
    assert served is not None and served.cache_key == _STORED_KEY


# ---------------------------------------------------------------------------
# Contract 3 -- a stale generation-context embedding is not returned (fingerprint)
# ---------------------------------------------------------------------------

def test_embedding_is_not_returned_across_context_fingerprints(module):
    cache = _fresh_cache(module, threshold=0.92)
    cache.store_embedding(
        cache_key=_STORED_KEY,
        model="model-a",
        query_text="stored query",
        embedding=_VEC_BASE,
        context_fingerprint="context-A",
    )

    # Same vector and model, but a different generation context is requested:
    # the fingerprint clause must exclude the row, so the search misses.
    miss = cache.find_similar_by_embedding(
        _VEC_BASE, "model-a", exclude_key=_OTHER_KEY,
        context_fingerprint="context-B",
    )
    assert miss is None, "an embedding under a different context fingerprint must not be returned"

    # Control: the matching fingerprint is returned, proving the row is present.
    served = cache.find_similar_by_embedding(
        _VEC_BASE, "model-a", exclude_key=_OTHER_KEY,
        context_fingerprint="context-A",
    )
    assert served is not None and served.cache_key == _STORED_KEY


_TESTS = [
    test_below_threshold_embedding_is_not_returned,
    test_embedding_is_not_returned_across_models,
    test_embedding_is_not_returned_across_context_fingerprints,
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
