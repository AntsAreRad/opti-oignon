#!/usr/bin/env python3
"""Security contracts for the response cache invalidation and stats surface.

These tests cover the eviction and reporting entry points, whose gates are a
scoping ``WHERE`` clause on a destructive statement or a time-to-live filter on a
count. A regression that loosens any one of them would silently over-delete or
over-report, so each contract seeds entries that must survive (or must be
excluded) and proves the intact gate holds:

  * ``invalidate`` removes exactly the keyed entry and leaves every other entry
    in place, so a single-key eviction can never become a table-wide wipe;
  * ``invalidate_model`` removes exactly the entries of the named model and
    leaves other models in place, so a model-scoped eviction stays scoped;
  * ``clear`` removes every entry and resets the live session counters, so a
    clear is a true reset and not a partial one that leaves stale entries or
    stale hit statistics behind;
  * ``get_stats`` counts only entries that are still within their time-to-live,
    so an expired entry that physically remains in the store is never reported
    as active.

The module is loaded in isolation with its package dependencies stubbed: a
private temporary ``DATA_DIR`` is supplied through a stub config, and the
encrypted-connection helper is allowed to fall back to a real plain SQLite
database, so every gate decision is reproducible without a model server or an
encrypted store. Each test receives its own freshly executed module instance,
and each cache uses its own temporary database, which keeps state from leaking
between tests. The clock is replaced with a deterministic fake where time
matters (the stats time-to-live filter) so the outcomes do not depend on wall
time. A raw entry count read directly from the store is used as a control, so a
refusal is always proven against a store that genuinely still holds the entry
rather than against an empty table.
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
# Contract 1 -- invalidate removes only the keyed entry (scoped deletion)
# ---------------------------------------------------------------------------

def test_invalidate_removes_only_the_keyed_entry(module):
    cache = _fresh_cache(module)

    key_a = cache.put("model-A", "system", "content A", "ANSWER A")
    key_b = cache.put("model-B", "system", "content B", "ANSWER B")
    assert key_a != key_b

    # Control: both entries are genuinely present before the invalidation.
    assert cache.entry_count() == 2

    removed = cache.invalidate(key_a)
    assert removed is True, "invalidating a present key must report a removal"

    # The keyed entry is gone; every other entry must remain. A gate that no
    # longer scopes the delete to the key would wipe the whole table here.
    assert cache.get(key_a) is None, "the invalidated entry must not be served"
    assert cache.get(key_b) is not None, "an unrelated entry must not be removed"
    assert cache.entry_count() == 1, "only the keyed entry may be removed"


# ---------------------------------------------------------------------------
# Contract 2 -- invalidate_model removes only that model's entries (scoped)
# ---------------------------------------------------------------------------

def test_invalidate_model_removes_only_that_model(module):
    cache = _fresh_cache(module)

    cache.put("model-A", "system", "content 1", "A1")
    cache.put("model-A", "system", "content 2", "A2")
    key_b = cache.put("model-B", "system", "content 3", "B1")

    # Control: all three entries are genuinely present.
    assert cache.entry_count() == 3

    removed = cache.invalidate_model("model-A")
    assert removed == 2, "exactly the two model-A entries must be removed"

    # The other model's entry must survive. A gate that no longer scopes the
    # delete to the model would remove model-B as well.
    assert cache.get(key_b) is not None, "another model's entry must not be removed"
    assert cache.entry_count() == 1, "only the named model's entries may be removed"


# ---------------------------------------------------------------------------
# Contract 3 -- clear empties the store and resets the session counters
# ---------------------------------------------------------------------------

def test_clear_empties_store_and_resets_counters(module):
    cache = _fresh_cache(module)

    key = cache.put("m", "system", "content", "ANSWER")
    cache.put("m2", "system", "other content", "ANSWER 2")

    # Drive the session counters off zero: one hit and one miss.
    assert cache.get(key) is not None
    assert cache.get("no-such-key") is None
    assert cache.session_hits > 0 and cache.session_misses > 0
    assert cache.entry_count() == 2

    removed = cache.clear()
    assert removed == 2, "clear must report every removed entry"

    # The store must be empty: a clear that no longer deletes would leave the
    # entries behind here.
    assert cache.entry_count() == 0, "clear must remove every entry"

    # The session statistics must be reset: a clear that skips the reset would
    # leave the earlier hit and miss counts standing here.
    assert cache.session_hits == 0, "clear must reset the session hit counter"
    assert cache.session_misses == 0, "clear must reset the session miss counter"


# ---------------------------------------------------------------------------
# Contract 4 -- get_stats counts only entries within their TTL (no stale count)
# ---------------------------------------------------------------------------

def test_get_stats_counts_only_active_entries(module):
    clock = _use_fake_clock(module, start=1000.0)
    cache = _fresh_cache(module)

    # One long-lived entry and one short-lived entry, both stored at t=1000.
    cache.put("fresh-model", "system", "fresh content", "FRESH", ttl=3600)
    cache.put("stale-model", "system", "stale content", "STALE", ttl=10)

    # Advance past the short TTL only: stale is expired (1000+10 <= 2000),
    # fresh is still active (1000+3600 > 2000).
    clock["now"] = 2000.0

    # Control: both entries physically remain in the store; nothing has purged
    # the expired one, so the stats filter is what must exclude it.
    assert cache.entry_count() == 2

    stats = cache.get_stats()
    assert stats.total_entries == 1, "an expired entry must not be counted as active"
    assert set(stats.entries_by_model) == {"fresh-model"}, (
        "only active entries may appear in the per-model breakdown"
    )


_TESTS = [
    test_invalidate_removes_only_the_keyed_entry,
    test_invalidate_model_removes_only_that_model,
    test_clear_empties_store_and_resets_counters,
    test_get_stats_counts_only_active_entries,
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
