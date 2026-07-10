#!/usr/bin/env python3
"""Contracts for the fine-tune tracker's persistence and comparison safety.

The tracker is a passive registry: it stores fine-tuned variants and A/B
comparison reports in a local database and nothing reads them back to act.
The safety-relevant properties are that every filter and update reaches the
database as a bound parameter behind a static field allowlist, that a
corrupted stored payload degrades on load instead of crashing, that the
variant registry enforces its uniqueness rule, and that a comparison run is
exhaustive in its accounting and captures inference failures as data instead
of propagating them. These contracts pin those guards without pinning the
schema, the length heuristic, or the timeout value.

  * TK1 -- filter values are bound parameters: a hostile filter string
    returns no rows and leaves the registry intact, it is never spliced
    into the statement.
  * TK2 -- junk variant metadata degrades: a row whose metadata payload is
    not valid JSON loads with empty metadata instead of raising.
  * TK3 -- junk comparison prompts degrade: a row whose prompts payload is
    not valid JSON loads with an empty prompt list instead of raising.
  * TK4 -- updates respect the field allowlist: a field outside the
    allowlist is ignored, so the immutable identity columns cannot be
    rewritten through the update path.
  * TK5 -- the registry rejects a duplicate variant model name.
  * TK6 -- comparison accounting is exhaustive and persisted: wins plus
    ties equal the prompt count, and the stored row round-trips the result.
  * TK7 -- an inference failure is captured, not raised: the failing side
    is recorded as an error response and the comparison still completes.

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module is
loaded under a stand-in package with an empty search path, so storage
degrades to plain sqlite on a temporary database and no model backend is
required; inference is a deterministic in-process function.
"""

import importlib.util
import sqlite3
import sys
import tempfile
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_KEYS = (
    "opti_oignon",
    "opti_oignon.fine_tune_tracker",
    "opti_oignon.db_utils",
)


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load():
    """Load the real tracker under a stand-in package.

    Returns (module, restore). With an empty package search path the
    encrypted-connection helper is absent, so the tracker uses its plain
    sqlite fallback on the temporary databases these tests create; the
    parameter-binding and load-tolerance properties under test are
    identical either way.
    """
    saved = {k: sys.modules.get(k) for k in _KEYS}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.fine_tune_tracker", _OO / "fine_tune_tracker.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.fine_tune_tracker"] = mod
    spec.loader.exec_module(mod)
    pkg.fine_tune_tracker = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


def _make_tracker(mod):
    """Build a tracker on fresh temporary database and config paths."""
    tmp = Path(tempfile.mkdtemp(prefix="oo-ftt-"))
    return mod.FineTuneTracker(
        db_path=tmp / "variants.db",
        config_path=tmp / "fine_tune.yaml",
    )


def _register(mod, tracker, name="tuned", variant_model="tuned:latest"):
    variant = mod.FineTuneVariant(
        name=name, base_model="base:latest", variant_model=variant_model,
    )
    return tracker.register_variant(variant)


def _raw_update(tracker, statement, params):
    """Tamper with the stored payload directly, as a corrupted file would."""
    conn = sqlite3.connect(str(tracker._db_path))
    try:
        conn.execute(statement, params)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
def test_tk1_hostile_filter_value_is_parameterized():
    mod, restore = _load()
    try:
        tracker = _make_tracker(mod)
        _register(mod, tracker)

        hostile = "base:latest'; DROP TABLE variants; --"
        rows = tracker.list_variants(base_model=hostile)
        assert rows == [], (
            "a hostile filter value must match nothing; it reached the"
            " statement as text instead of a bound parameter"
        )

        rows = tracker.list_variants(base_model="base:latest")
        assert len(rows) == 1, (
            "the registry must be intact after the hostile filter"
        )
    finally:
        restore()


def test_tk2_junk_variant_metadata_degrades_to_empty():
    mod, restore = _load()
    try:
        tracker = _make_tracker(mod)
        variant = _register(mod, tracker)

        _raw_update(
            tracker,
            "UPDATE variants SET metadata = ? WHERE variant_id = ?",
            ("{this is not json", variant.variant_id),
        )
        loaded = tracker.get_variant(variant.variant_id)
        assert loaded is not None, "the row must still load"
        assert loaded.metadata == {}, (
            f"junk metadata must degrade to an empty mapping,"
            f" got {loaded.metadata!r}"
        )
    finally:
        restore()


def test_tk3_junk_comparison_prompts_degrade_to_empty():
    mod, restore = _load()
    try:
        tracker = _make_tracker(mod)
        variant = _register(mod, tracker)
        comparison = tracker.create_comparison(variant.variant_id, ["probe"])

        _raw_update(
            tracker,
            "UPDATE comparisons SET prompts_json = ?"
            " WHERE comparison_id = ?",
            ("[[[", comparison.comparison_id),
        )
        loaded = tracker.get_comparison(comparison.comparison_id)
        assert loaded is not None, "the row must still load"
        assert loaded.prompts == [], (
            f"junk prompts must degrade to an empty list,"
            f" got {loaded.prompts!r}"
        )
    finally:
        restore()


def test_tk4_update_ignores_fields_outside_allowlist():
    mod, restore = _load()
    try:
        tracker = _make_tracker(mod)
        variant = _register(mod, tracker)

        updated = tracker.update_variant(variant.variant_id, {
            "name": "renamed",
            "variant_model": "hijacked:latest",
        })
        assert updated is not None
        assert updated.name == "renamed", "allowlisted fields must apply"
        assert updated.variant_model == "tuned:latest", (
            f"the identity column was rewritten to"
            f" {updated.variant_model!r}; fields outside the allowlist"
            " must be ignored"
        )
    finally:
        restore()


def test_tk5_duplicate_variant_model_is_rejected():
    mod, restore = _load()
    try:
        tracker = _make_tracker(mod)
        _register(mod, tracker, name="first", variant_model="tuned:latest")

        try:
            _register(
                mod, tracker, name="second", variant_model="tuned:latest",
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "registering the same variant model twice must be rejected"
            )
    finally:
        restore()


def test_tk6_comparison_accounting_is_exhaustive_and_persisted():
    mod, restore = _load()
    try:
        tracker = _make_tracker(mod)
        variant = _register(mod, tracker)
        comparison = tracker.create_comparison(
            variant.variant_id, ["variant-wins", "base-wins", "even"],
        )

        def inference(model, prompt):
            tuned = model == variant.variant_model
            if prompt == "variant-wins":
                return "y" * (40 if tuned else 3)
            if prompt == "base-wins":
                return "y" * (3 if tuned else 40)
            return "same-length"

        result = tracker.run_comparison(
            comparison.comparison_id, inference_fn=inference,
        )
        assert result.status == mod.COMPARISON_STATUS_COMPLETED
        total = result.base_wins + result.variant_wins + result.ties
        assert total == 3, (
            f"accounting must be exhaustive over the prompts,"
            f" got {result.base_wins}+{result.variant_wins}+{result.ties}"
        )
        assert (result.base_wins, result.variant_wins, result.ties) == (1, 1, 1)

        reloaded = tracker.get_comparison(comparison.comparison_id)
        assert reloaded.status == mod.COMPARISON_STATUS_COMPLETED
        assert (
            reloaded.base_wins, reloaded.variant_wins, reloaded.ties,
        ) == (1, 1, 1), "the stored row must round-trip the accounting"
        assert len(reloaded.prompts) == 3
    finally:
        restore()


def test_tk7_inference_failure_is_captured_not_raised():
    mod, restore = _load()
    try:
        tracker = _make_tracker(mod)
        variant = _register(mod, tracker)
        comparison = tracker.create_comparison(variant.variant_id, ["probe"])

        def inference(model, prompt):
            if model == variant.variant_model:
                raise RuntimeError("backend refused")
            return "steady answer"

        result = tracker.run_comparison(
            comparison.comparison_id, inference_fn=inference,
        )
        assert result.status == mod.COMPARISON_STATUS_COMPLETED, (
            "one failing side must not abort the comparison"
        )
        assert result.prompts[0].variant_response.startswith("[Error:"), (
            f"the failure must be captured as an error response,"
            f" got {result.prompts[0].variant_response!r}"
        )
        assert result.prompts[0].base_response == "steady answer"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("TK1 hostile filter value is parameterized",
         test_tk1_hostile_filter_value_is_parameterized),
        ("TK2 junk variant metadata degrades to empty",
         test_tk2_junk_variant_metadata_degrades_to_empty),
        ("TK3 junk comparison prompts degrade to empty",
         test_tk3_junk_comparison_prompts_degrade_to_empty),
        ("TK4 update ignores fields outside allowlist",
         test_tk4_update_ignores_fields_outside_allowlist),
        ("TK5 duplicate variant model is rejected",
         test_tk5_duplicate_variant_model_is_rejected),
        ("TK6 comparison accounting is exhaustive and persisted",
         test_tk6_comparison_accounting_is_exhaustive_and_persisted),
        ("TK7 inference failure is captured not raised",
         test_tk7_inference_failure_is_captured_not_raised),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
