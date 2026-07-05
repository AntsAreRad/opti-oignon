#!/usr/bin/env python3
"""Security contracts for the exact-match LLM response cache.

These tests pin the serve-versus-refuse behaviour of the live ``get``/``put``
path so that a regression which loosens any one gate is caught. Each contract
seeds the cache with an entry that *would* be served if its gate were removed,
then proves that the intact gate withholds it (and, where useful, a control
shows the entry is genuinely present so the refusal is a real fail and not an
empty cache):

  * an entry past its time-to-live is purged on lookup and missed, never served
    stale;
  * the model name is bound into the cache key, so a response generated for one
    model is never served to a request for a different model;
  * the conversation history is bound into the conversation cache key, so a
    response built under one conversation state is never served to a request
    whose history has diverged;
  * when the cache is at capacity a new insertion evicts the least-recently-used
    entry, so the store stays bounded and cannot grow without limit.

The module is loaded in isolation with its package dependencies stubbed: a
private temporary ``DATA_DIR`` is supplied through a stub config, and the
encrypted-connection helper is allowed to fall back to a real plain SQLite
database, so every gate decision is reproducible without a model server or an
encrypted store. Each test receives its own freshly executed module instance,
and each cache uses its own temporary database, which keeps state from leaking
between tests. The clock is replaced with a deterministic fake where time
matters (expiry and recency ordering) so the outcomes do not depend on wall
time.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TARGET = _REPO_ROOT / "opti_oignon" / "response_cache.py"


def _load_response_cache():
    """Flat-load the cache module with its package dependencies stubbed.

    A fresh module object is created on every call so that per-test fakes (the
    clock in particular) stay isolated.
    """
    pkg = sys.modules.get("opti_oignon")
    if pkg is None:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = []
        sys.modules["opti_oignon"] = pkg
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = Path(tempfile.mkdtemp(prefix="resp_cache_data_"))
    sys.modules["opti_oignon.config"] = cfg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.response_cache", str(_TARGET)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.response_cache"] = module
    spec.loader.exec_module(module)
    return module


def _fresh_cache(module, *, ttl=3600, max_entries=500):
    """Build a cache on a private temporary database with an explicit config."""
    db_dir = Path(tempfile.mkdtemp(prefix="resp_cache_db_"))
    return module.ResponseCache(
        db_path=db_dir / "response_cache.db",
        default_ttl=ttl,
        max_entries=max_entries,
    )


def _use_fake_clock(module, start=1000.0):
    """Replace the module clock with a settable fake; return the holder dict."""
    clock = {"now": start}
    module.time = types.SimpleNamespace(time=lambda: clock["now"])
    return clock


# ---------------------------------------------------------------------------
# Contract 1 -- an expired entry is purged and missed (no serve-stale)
# ---------------------------------------------------------------------------

def test_expired_entry_is_not_served_and_is_purged(module):
    clock = _use_fake_clock(module, start=1000.0)
    cache = _fresh_cache(module, ttl=10)

    key = cache.put("m", "system", "user content", "STALE ANSWER")  # created at 1000

    # Control: before expiry the entry is served, proving it is present.
    fresh_hit = cache.get(key)
    assert fresh_hit is not None and fresh_hit.response == "STALE ANSWER"

    # Advance past the TTL: now - created == 15 > ttl 10.
    clock["now"] = 1000.0 + 10 + 5
    hit = cache.get(key)
    assert hit is None, "an entry past its TTL must not be served"

    # The expired entry must have been removed on lookup, not merely hidden.
    assert cache.entry_count() == 0, "an expired entry must be purged on lookup"


# ---------------------------------------------------------------------------
# Contract 2 -- the model is bound into the key (no cross-model serve)
# ---------------------------------------------------------------------------

def test_response_is_not_served_across_models(module):
    cache = _fresh_cache(module)
    system_prompt = "system"
    user_content = "user content"

    cache.put("model-A", system_prompt, user_content, "ANSWER FROM A")

    # Control: the model-A key serves its answer, proving the entry is present.
    key_a = module.ResponseCache.make_cache_key("model-A", system_prompt, user_content)
    served = cache.get(key_a)
    assert served is not None and served.response == "ANSWER FROM A"

    # A request for a different model (same prompt and content) must miss: the
    # answer generated for model-A must never be served for model-B.
    key_b = module.ResponseCache.make_cache_key("model-B", system_prompt, user_content)
    assert key_b != key_a, "different models must produce different cache keys"
    hit = cache.get(key_b)
    assert hit is None, "a response cached for one model must not be served to another"


# ---------------------------------------------------------------------------
# Contract 3 -- the history is bound into the conversation key (no cross-state serve)
# ---------------------------------------------------------------------------

def test_response_is_not_served_across_conversation_states(module):
    cache = _fresh_cache(module)
    model = "m"
    system_prompt = "system"
    user_content = "current question"
    history_a = [{"role": "user", "content": "earlier turn A"}]
    history_b = [{"role": "user", "content": "earlier turn B"}]

    key_a = module.ResponseCache.make_conversation_cache_key(
        model, system_prompt, history_a, user_content
    )
    cache.put(model, system_prompt, user_content, "ANSWER UNDER STATE A", explicit_key=key_a)

    # Control: the same conversation state serves its answer.
    served = cache.get(key_a)
    assert served is not None and served.response == "ANSWER UNDER STATE A"

    # A request whose history has diverged (same model, prompt and current
    # message) must miss: the answer built under state A must not be replayed.
    key_b = module.ResponseCache.make_conversation_cache_key(
        model, system_prompt, history_b, user_content
    )
    assert key_b != key_a, "different histories must produce different conversation keys"
    hit = cache.get(key_b)
    assert hit is None, "a response cached for one conversation state must not be served to another"


# ---------------------------------------------------------------------------
# Contract 4 -- a full cache evicts the LRU entry on insert (bounded growth)
# ---------------------------------------------------------------------------

def test_full_cache_evicts_lru_on_insert(module):
    clock = _use_fake_clock(module, start=1000.0)
    cache = _fresh_cache(module, max_entries=3)

    keys = {}
    for i in range(3):
        clock["now"] = 1000.0 + i
        keys[i] = cache.put(f"m{i}", "system", f"content {i}", f"answer {i}")

    # Touch the oldest-inserted entry so it is no longer the LRU candidate; the
    # least-recently-used is now entry 1.
    clock["now"] = 1000.0 + 10
    assert cache.get(keys[0]) is not None

    # Insert a fourth entry while at capacity.
    clock["now"] = 1000.0 + 11
    cache.put("m9", "system", "content 9", "answer 9")

    # The store must stay bounded at the configured maximum.
    assert cache.entry_count() == 3, "a full cache must stay bounded on insert"

    # The least-recently-used entry must be the one evicted; the touched entry
    # and the new entry must survive.
    assert cache.get(keys[1]) is None, "the least-recently-used entry must be evicted"
    assert cache.get(keys[0]) is not None, "a recently-touched entry must not be evicted"


_TESTS = [
    test_expired_entry_is_not_served_and_is_purged,
    test_response_is_not_served_across_models,
    test_response_is_not_served_across_conversation_states,
    test_full_cache_evicts_lru_on_insert,
]


def _main(argv):
    selected = set(argv) if argv else {t.__name__ for t in _TESTS}
    failures = 0
    for test in _TESTS:
        if test.__name__ not in selected:
            continue
        module = _load_response_cache()
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
