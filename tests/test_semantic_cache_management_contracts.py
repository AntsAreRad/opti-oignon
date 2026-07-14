#!/usr/bin/env python3
"""Security contracts for the query cache management surface.

Beyond the read path, the query cache exposes management operations that remove
rows: scoped invalidation, least-recently-used eviction, and time-to-live
expiry. Each must remove exactly the rows it targets and keep the rest; a
loosened gate silently deletes entries a caller expected to keep (a scoped
invalidation that wipes every conversation, an eviction that discards the newest
instead of the oldest, an expiry that removes fresh rows). Each contract seeds a
store whose contents are known, exercises the operation, and asserts both the
returned count and the surviving rows, with a control confirming the store was
populated so a pass can never come from an empty table:

  * scoped invalidation removes only the targeted conversation and leaves every
    other conversation intact, so clearing one conversation never wipes the
    whole cache;
  * eviction removes the least-recently-used rows (oldest last-accessed) and
    keeps the most-recently-used, so a full cache discards stale entries rather
    than the ones just written;
  * eviction leaves the store untouched while it is at or under capacity, so a
    cache below its limit is never wiped by a spurious eviction pass (the guard
    is load-bearing: without it the excess count goes negative and the delete
    drops every row);
  * expiry removes only rows past their time-to-live and keeps rows still inside
    the window, so a maintenance sweep never drops a fresh entry.

The management operations do not rank by similarity, so the embedding tier is
disabled: the store is exercised through the exact path only, seeded rows are
matched by their query hash, and no embedding backend is needed. A monotonic
fake clock drives the last-accessed ordering and the age comparison so eviction
order and expiry are deterministic. The module is loaded in isolation and each
test receives its own freshly executed instance, which keeps the fake clock and
mode probe from leaking between tests.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TARGET = _REPO_ROOT / "opti_oignon" / "semantic_cache.py"


def _load_semantic_cache():
    """Flat-load the cache module with its package dependencies stubbed.

    A fresh module object is created on every call so that per-test monkey
    patches (mode probe, clock) stay isolated.
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
    ttl=3600,
    max_entries=100,
    scope="conversation",
    exact=True,
    semantic=False,
    embeddings=False,
):
    """Build a cache on a private temporary database with an explicit config.

    A non-existent config path forces the built-in defaults, then the relevant
    switches are set directly. The similarity tier is off by default so the
    store is exercised through the exact path only (no embedding backend).
    """
    db_dir = Path(tempfile.mkdtemp(prefix="cache_db_"))
    absent_config = db_dir / "absent.yaml"
    cache = module.SemanticCache(
        db_path=db_dir / "semantic_cache.db",
        config_path=absent_config,
        ttl_seconds=ttl,
        max_entries=max_entries,
        scope=scope,
    )
    cache._config["enabled"] = True
    cache._config["exact_match_enabled"] = exact
    cache._config["semantic_match_enabled"] = semantic
    cache.embeddings_available = embeddings
    return cache


# ---------------------------------------------------------------------------
# Contract 1 -- scoped invalidation removes only the targeted conversation
# ---------------------------------------------------------------------------

def test_invalidate_removes_only_the_targeted_conversation(module):
    module._is_bulbe = lambda: False
    cache = _fresh_cache(module, max_entries=100)

    cache.put("first question", "ANSWER ONE", conversation_id="c1")
    cache.put("second question", "ANSWER TWO", conversation_id="c2")
    assert cache.entry_count() == 2, "control: both conversations must be stored"

    removed = cache.invalidate("c1")
    assert removed == 1, "invalidate must remove exactly the targeted conversation"
    assert cache.entry_count() == 1, "only the targeted conversation must be removed"

    # The survivor is c2: its exact entry is still served, proving the scope
    # spared it rather than the store being wiped table-wide.
    survivor = cache.get("second question", conversation_id="c2")
    assert survivor is not None and survivor.response == "ANSWER TWO"


# ---------------------------------------------------------------------------
# Contract 2 -- eviction removes the least-recently-used, not the newest
# ---------------------------------------------------------------------------

def test_eviction_removes_the_oldest_not_the_newest(module):
    clock = {"now": 1000.0}
    module.time = types.SimpleNamespace(time=lambda: clock["now"])
    module._is_bulbe = lambda: False
    cache = _fresh_cache(module, max_entries=2)

    cache.put("oldest question", "OLDEST", conversation_id="c1")  # last_accessed 1000
    clock["now"] = 1001.0
    cache.put("middle question", "MIDDLE", conversation_id="c1")  # last_accessed 1001
    clock["now"] = 1002.0
    cache.put("newest question", "NEWEST", conversation_id="c1")  # 1002 -> evicts oldest

    assert cache.entry_count() == 2, "eviction must cap the store at max_entries"

    # The least-recently-used row is evicted; the two most-recent survive.
    assert cache.get("oldest question", conversation_id="c1") is None, (
        "the least-recently-used entry must be evicted"
    )
    assert cache.get("newest question", conversation_id="c1") is not None, (
        "the most-recently-used entry must survive"
    )
    assert cache.get("middle question", conversation_id="c1") is not None


# ---------------------------------------------------------------------------
# Contract 3 -- expiry removes only entries past their TTL
# ---------------------------------------------------------------------------

def test_expire_stale_removes_only_past_ttl_entries(module):
    clock = {"now": 1000.0}
    module.time = types.SimpleNamespace(time=lambda: clock["now"])
    module._is_bulbe = lambda: False
    cache = _fresh_cache(module, ttl=10, max_entries=100)

    cache.put("aging question", "OLD", conversation_id="c1")   # created_at 1000
    clock["now"] = 1008.0
    cache.put("recent question", "NEW", conversation_id="c1")  # created_at 1008
    assert cache.entry_count() == 2, "control: both entries must be stored"

    # Advance past the TTL for the first entry only: age 15 >= 10 (stale) vs
    # age 7 < 10 (fresh).
    clock["now"] = 1015.0
    removed = cache.expire_stale()
    assert removed == 1, "expire_stale must remove only entries past their TTL"
    assert cache.entry_count() == 1, "the fresh entry must survive expire_stale"

    survivor = cache.get("recent question", conversation_id="c1")
    assert survivor is not None and survivor.response == "NEW", (
        "the entry still inside its TTL must remain retrievable"
    )


# ---------------------------------------------------------------------------
# Contract 4 -- no eviction while the store is at or under capacity
# ---------------------------------------------------------------------------

def test_no_eviction_while_under_capacity(module):
    module._is_bulbe = lambda: False
    cache = _fresh_cache(module, max_entries=5)

    # Two entries, well under the capacity of five. The eviction pass runs after
    # every write; while under capacity it must remove nothing. Without the
    # guard the excess count is negative and the delete's negative limit drops
    # every row, so this also proves a below-limit cache is never wiped.
    cache.put("kept question one", "KEEP ONE", conversation_id="c1")
    cache.put("kept question two", "KEEP TWO", conversation_id="c1")

    assert cache.entry_count() == 2, "a store under capacity must not be evicted"
    assert cache.get("kept question one", conversation_id="c1") is not None
    assert cache.get("kept question two", conversation_id="c1") is not None


_TESTS = [
    test_invalidate_removes_only_the_targeted_conversation,
    test_eviction_removes_the_oldest_not_the_newest,
    test_expire_stale_removes_only_past_ttl_entries,
    test_no_eviction_while_under_capacity,
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
