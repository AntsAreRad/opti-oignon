#!/usr/bin/env python3
"""Contracts for the published-prose CI guard.

The guard records what the framework publishes rather than predicting it, so
these contracts pin the recording, not the framework. Every input here is a
hand-built schema mapping: no application is imported, and the behaviour is
verified independently of whatever the live schema happens to contain today.

  * C1 -- the digest is deterministic. The same schema yields the same text,
    byte for byte, however many times it is built.
  * C2 -- a rewritten model class description moves its line, and moves only
    its line. This is the construct no prover here understood: the framework
    publishes a model docstring as the schema description.
  * C3 -- a rewritten field description moves its line.
  * C4 -- a rewritten route handler description moves its line.
  * C5 -- a rename moves a KEY even when the prose is untouched. A model or
    an endpoint cannot be renamed behind the digest's back.
  * C6 -- what the framework does not publish is not recorded. A description
    that never reaches the schema contributes no line, so purging it stays
    free.
  * C7 -- framework-generated text is excluded. Response descriptions and
    endpoint summaries are manufactured from names; recording them would add
    noise that can only move when a key already moved.
  * C8 -- the digest carries hashes, never prose. The recorded file must not
    become a second copy of the public surface.
  * C9 -- comparison is silent when nothing moved, and names exactly what
    moved otherwise, in three separate directions.
  * C10 -- the digest is stable under construction order. Two mappings built
    in different orders with the same content yield the same text.
  * C11 -- the recorded lines are globally sorted, not merely grouped by the
    order the schema was walked in. A new description then lands beside its
    neighbours and a reviewer reads a local diff instead of a reshuffle.

Local-only. Runs under pytest or via the __main__ runner. The guard script
lives under .github/, outside the importable package, and is loaded through
the shared isolation window.
"""

import copy
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = "_published_prose_guard_under_contract"
_GUARD_PATH = REPO / ".github" / "scripts" / "published_prose_guard.py"


def _load():
    """Load the guard through the shared window; returns (module, restore)."""
    loaded, restore = isolate(targets={_GUARD: _GUARD_PATH})
    return loaded[_GUARD], restore


def _schema():
    """A small generated schema, carrying one of each published kind."""
    return {
        "components": {
            "schemas": {
                "Widget": {
                    "description": "A widget as the client sees it.",
                    "properties": {
                        "size": {"description": "Width in millimetres."},
                        "tag": {"type": "string"},
                    },
                },
                "Gadget": {"type": "object"},
            },
        },
        "paths": {
            "/api/widgets": {
                "get": {
                    "operationId": "list_widgets",
                    "summary": "List Widgets",
                    "description": "Return every widget on record.",
                    "parameters": [
                        {"name": "limit", "description": "How many at most."},
                        {"name": "cursor"},
                    ],
                    "responses": {"200": {"description": "Successful Response"}},
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

def test_c1_digest_is_deterministic():
    guard, restore = _load()
    try:
        schema = _schema()
        first = guard.build_digest(schema)
        second = guard.build_digest(copy.deepcopy(schema))
        assert first == second, "the digest is not reproducible"
        assert first == guard.build_digest(schema), "a rebuild differs"
    finally:
        restore()


def test_c2_model_description_moves_its_line():
    guard, restore = _load()
    try:
        before = guard.build_digest(_schema())
        after_schema = _schema()
        after_schema["components"]["schemas"]["Widget"]["description"] = \
            "A widget as the client sees it today."
        appeared, gone, rewritten = guard.compare(
            before, guard.build_digest(after_schema))
        assert rewritten == ["schema Widget"], rewritten
        assert appeared == [] and gone == [], (appeared, gone)
    finally:
        restore()


def test_c3_field_description_moves_its_line():
    guard, restore = _load()
    try:
        before = guard.build_digest(_schema())
        after_schema = _schema()
        properties = after_schema["components"]["schemas"]["Widget"]["properties"]
        properties["size"]["description"] = "Width in centimetres."
        appeared, gone, rewritten = guard.compare(
            before, guard.build_digest(after_schema))
        assert rewritten == ["field Widget.size"], rewritten
        assert appeared == [] and gone == [], (appeared, gone)
    finally:
        restore()


def test_c4_handler_description_moves_its_line():
    guard, restore = _load()
    try:
        before = guard.build_digest(_schema())
        after_schema = _schema()
        after_schema["paths"]["/api/widgets"]["get"]["description"] = \
            "Return every widget still on record."
        appeared, gone, rewritten = guard.compare(
            before, guard.build_digest(after_schema))
        assert rewritten == ["op list_widgets"], rewritten
        assert appeared == [] and gone == [], (appeared, gone)
    finally:
        restore()


def test_c5_a_rename_cannot_hide():
    guard, restore = _load()
    try:
        before = guard.build_digest(_schema())
        after_schema = _schema()
        models = after_schema["components"]["schemas"]
        models["Doodad"] = models.pop("Widget")
        appeared, gone, rewritten = guard.compare(
            before, guard.build_digest(after_schema))
        assert rewritten == [], rewritten
        assert "schema Doodad" in appeared and "field Doodad.size" in appeared
        assert "schema Widget" in gone and "field Widget.size" in gone
    finally:
        restore()


def test_c6_unpublished_prose_is_not_recorded():
    guard, restore = _load()
    try:
        keys = guard.entries(guard.build_digest(_schema()))
        assert "schema Gadget" not in keys, "a model with no description"
        assert "field Widget.tag" not in keys, "a field with no description"
        assert "param list_widgets.cursor" not in keys, "a bare parameter"
        assert "schema Widget" in keys and "field Widget.size" in keys
        assert "op list_widgets" in keys and "param list_widgets.limit" in keys
    finally:
        restore()


def test_c7_framework_generated_text_is_excluded():
    guard, restore = _load()
    try:
        before = guard.build_digest(_schema())
        after_schema = _schema()
        operation = after_schema["paths"]["/api/widgets"]["get"]
        operation["summary"] = "Enumerate Widgets"
        operation["responses"]["200"]["description"] = "All good"
        assert guard.build_digest(after_schema) == before, \
            "manufactured text reached the digest"
    finally:
        restore()


def test_c8_the_digest_carries_no_prose():
    guard, restore = _load()
    try:
        digest = guard.build_digest(_schema())
        for prose in ("A widget as the client sees it.",
                      "Width in millimetres.",
                      "Return every widget on record.",
                      "How many at most."):
            assert prose not in digest, f"published prose leaked: {prose!r}"
    finally:
        restore()


def test_c9_comparison_is_silent_when_nothing_moved():
    guard, restore = _load()
    try:
        digest = guard.build_digest(_schema())
        assert guard.compare(digest, digest) == ([], [], [])
        after_schema = _schema()
        after_schema["components"]["schemas"]["Gadget"]["description"] = "New."
        appeared, gone, rewritten = guard.compare(
            digest, guard.build_digest(after_schema))
        assert appeared == ["schema Gadget"], appeared
        assert gone == [] and rewritten == [], (gone, rewritten)
    finally:
        restore()


def test_c10_digest_is_stable_under_construction_order():
    guard, restore = _load()
    try:
        straight = _schema()
        shuffled = _schema()
        models = shuffled["components"]["schemas"]
        shuffled["components"]["schemas"] = {
            "Gadget": models["Gadget"], "Widget": models["Widget"],
        }
        properties = models["Widget"]["properties"]
        models["Widget"]["properties"] = {
            "tag": properties["tag"], "size": properties["size"],
        }
        assert guard.build_digest(shuffled) == guard.build_digest(straight)
    finally:
        restore()


def test_c11_recorded_lines_are_globally_sorted():
    guard, restore = _load()
    try:
        recorded = [line for line in guard.build_digest(_schema()).splitlines()
                    if line and not line.startswith("#")]
        assert recorded == sorted(recorded), \
            "the digest is grouped by walk order, not sorted"
        assert len(recorded) > 1, "too few lines to establish an order"
    finally:
        restore()


# ---------------------------------------------------------------------------

def _run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
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
