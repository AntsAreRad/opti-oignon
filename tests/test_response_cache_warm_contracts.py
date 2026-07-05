#!/usr/bin/env python3
"""Contracts for the response cache's warm pre-population path.

``warm`` bulk-loads known question/response pairs so first requests hit the
cache. These contracts pin the two properties that keep that path safe:

  * a disabled cache warms nothing: the call reports zero and the store's
    raw row count is untouched, so "disabled" can never be worked around
    through the bulk loader;
  * a malformed entry is skipped without aborting the batch: the
    well-formed entries land and round-trip, the malformed ones leave no
    row behind, and the return value equals exactly the number stored --
    the count is real accounting, not optimism.

The module is loaded in isolation with its package dependencies stubbed
(a private temporary data directory through a stub config; the encrypted
connection helper falls back to plain SQLite), each contract gets its own
freshly executed module instance, and each cache uses its own temporary
database, so nothing leaks between contracts and no model server or
encrypted store is needed.
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

    A fresh module object is created on every call so per-contract state
    stays isolated.
    """
    pkg = sys.modules.get("opti_oignon")
    if pkg is None:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = []
        sys.modules["opti_oignon"] = pkg
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = Path(tempfile.mkdtemp(prefix="resp_cache_warm_data_"))
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
    db_dir = Path(tempfile.mkdtemp(prefix="resp_cache_warm_db_"))
    return module.ResponseCache(
        db_path=db_dir / "response_cache.db",
        default_ttl=ttl,
        max_entries=max_entries,
    )


def _entry(tag: str) -> dict:
    """A well-formed warm entry, distinguishable by ``tag``."""
    return {
        "model": f"model-{tag}",
        "system_prompt": f"system-{tag}",
        "user_content": f"question-{tag}",
        "response": f"answer-{tag}",
    }


# ---------------------------------------------------------------------------
# Contract 1 -- a disabled cache warms nothing (no count, no rows)
# ---------------------------------------------------------------------------

def test_disabled_cache_warms_nothing(module):
    cache = _fresh_cache(module)
    cache.enabled = False

    assert cache.entry_count() == 0, "a fresh cache must start empty"
    warmed = cache.warm([_entry("a"), _entry("b")])

    assert warmed == 0, (
        f"a disabled cache must report zero warmed entries, got {warmed}"
    )
    assert cache.entry_count() == 0, (
        "a disabled cache must write no rows through the bulk loader"
    )


# ---------------------------------------------------------------------------
# Contract 2 -- malformed entries are skipped; the count is real accounting
# ---------------------------------------------------------------------------

def test_malformed_entry_skipped_and_valid_ones_land(module):
    cache = _fresh_cache(module)

    missing_response = {
        "model": "model-x",
        "system_prompt": "system-x",
        "user_content": "question-x",
    }
    not_a_mapping = "garbage"

    warmed = cache.warm([
        _entry("a"),
        missing_response,
        not_a_mapping,
        _entry("b"),
    ])

    assert warmed == 2, (
        f"only the well-formed entries may be counted, got {warmed}"
    )
    assert cache.entry_count() == 2, (
        "exactly the well-formed entries must land in the store"
    )

    # A stored entry must genuinely round-trip (real rows, not a counter).
    key = cache.make_cache_key("model-a", "system-a", "question-a")
    hit = cache.get(key)
    assert hit is not None, "a warmed entry must be servable"
    assert hit.response == "answer-a", (
        f"the served response must match the warmed one, got {hit.response!r}"
    )


_TESTS = [
    test_disabled_cache_warms_nothing,
    test_malformed_entry_skipped_and_valid_ones_land,
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
