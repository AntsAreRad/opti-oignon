#!/usr/bin/env python3
"""Security contracts for the fuzzy (embedding-similarity) serve tier.

These tests pin the serve-versus-refuse behaviour of the second-tier lookup --
the branch that ranks stored candidates by cosine similarity after an exact-hash
miss. The exact-hash tier and the similarity threshold are pinned elsewhere;
here each contract seeds a candidate whose embedding is an *exact* match (cosine
1.0, well above the threshold) so that only the tier's own eligibility gate can
keep it from being served, then proves the intact gate withholds it:

  * a candidate stored under a different model is filtered out at the SQL layer
    before any cosine comparison, so a response produced for one engine is never
    served to a request routed to another;
  * a candidate whose age exceeds the time-to-live is filtered out at the SQL
    layer before any cosine comparison, so a stale response is never served
    through the fuzzy tier -- and, unlike the exact tier, the row is left in
    place rather than purged, so it stays available to a later in-window query;
  * a candidate stored under a different generation-context fingerprint is
    skipped while the candidates are ranked, so a response built under stale
    retrieval/memory context is never replayed by a fuzzy match.

To reach the fuzzy tier the incoming query text always differs from the stored
query text (the exact tier misses on the hash) while both map to the same
embedding vector, so the candidate is a genuine cosine match that only the gate
under test can exclude. The distinct query text also keeps the exact tier's
expiry purge -- which is keyed on the incoming query hash -- from removing the
seeded row, so the age contract exercises the fuzzy tier's own filter rather
than a side effect of the exact tier.

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
# Contract 1 -- a candidate under a different model is filtered before ranking
# ---------------------------------------------------------------------------

def test_candidate_under_other_model_is_not_served(module):
    cache = _fresh_cache(module, threshold=0.92, scope="conversation")
    stored_query = "How do I reverse a string in Python?"
    incoming_query = "What is the tallest mountain on Earth?"  # distinct -> exact miss

    module._is_bulbe = lambda: False
    # Both texts map to the same unit vector, so any row the SQL pre-filter
    # returns is an exact cosine match (1.0 >= threshold).
    module._get_embedding = _embedding_table(
        {stored_query: _VEC_BASE, incoming_query: _VEC_BASE}
    )

    cache.put(
        stored_query,
        "ANSWER FROM MODEL ALPHA",
        model="model-alpha",
        conversation_id="c1",
    )

    # The request is routed to a different model: the SQL scope filter excludes
    # the alpha-model row before the cosine comparison, so the fuzzy tier misses.
    hit = cache.get(incoming_query, conversation_id="c1", model="model-beta")
    assert hit is None, "a candidate stored under another model must not be served"

    # Control: the identical request routed to the matching model does serve the
    # candidate, proving the refusal above is a real scope filter and not an
    # empty cache or a below-threshold miss.
    served = cache.get(incoming_query, conversation_id="c1", model="model-alpha")
    assert served is not None and served.response == "ANSWER FROM MODEL ALPHA"


# ---------------------------------------------------------------------------
# Contract 2 -- a candidate past its TTL is filtered before ranking (not purged)
# ---------------------------------------------------------------------------

def test_expired_candidate_is_filtered_not_served_by_fuzzy_tier(module):
    clock = {"now": 1000.0}
    module.time = types.SimpleNamespace(time=lambda: clock["now"])

    cache = _fresh_cache(module, threshold=0.92, ttl=10, scope="conversation")
    stored_query = "How do I open a file in Python?"
    incoming_query = "Which planet is closest to the Sun?"  # distinct -> exact miss

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
    # fuzzy tier, so the refusal below is a real age filter and not an empty
    # cache. The distinct query text means the exact tier misses on the hash.
    fresh = cache.get(incoming_query, conversation_id="c1", model="model-alpha")
    assert fresh is not None and fresh.response == "STALE ANSWER"

    # Advance the clock past the TTL: now - created == 15 >= ttl 10.
    clock["now"] = 1000.0 + 10 + 5
    hit = cache.get(incoming_query, conversation_id="c1", model="model-alpha")
    assert hit is None, "a candidate past its TTL must not be served by the fuzzy tier"

    # The fuzzy tier filters the stale row out of the ranking but -- unlike the
    # exact tier -- does not purge it, so it stays available to a later query
    # that lands inside a refreshed window.
    assert cache.entry_count() == 1, "the fuzzy tier must filter, not purge, the stale row"


# ---------------------------------------------------------------------------
# Contract 3 -- a different-fingerprint candidate is skipped during ranking
# ---------------------------------------------------------------------------

def test_stale_context_fingerprint_candidate_is_skipped(module):
    cache = _fresh_cache(module, threshold=0.92, scope="conversation")
    stored_query = "What does the attached report conclude?"
    incoming_query = "How many moons does Jupiter have?"  # distinct -> exact miss

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

    # The current generation context differs from the one the candidate was
    # stored under: it is skipped while ranking, so the fuzzy tier misses even
    # though the embedding is an exact match.
    hit = cache.get(
        incoming_query,
        conversation_id="c1",
        model="model-alpha",
        context_fingerprint="context-beta",
    )
    assert hit is None, "a candidate under a different context fingerprint must be skipped"

    # Control: the same request under the matching fingerprint serves the
    # candidate, proving it is present and the skip above is a real gate.
    same = cache.get(
        incoming_query,
        conversation_id="c1",
        model="model-alpha",
        context_fingerprint="context-alpha",
    )
    assert same is not None and same.response == "ANSWER UNDER CONTEXT ALPHA"


_TESTS = [
    test_candidate_under_other_model_is_not_served,
    test_expired_candidate_is_filtered_not_served_by_fuzzy_tier,
    test_stale_context_fingerprint_candidate_is_skipped,
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
