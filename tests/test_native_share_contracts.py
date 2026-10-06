#!/usr/bin/env python3
"""Contracts for the share the native core serves of the probes' work.

The probes have a second implementation in the native core, asked at each
call and answering only for the generator it declares. A twin that never
answers is a silent zero: the work is served, by the reference, and nothing
says so. Every draw and every score is therefore counted once: served by the
core, or sent to the reference with the reason it was, so that the share the
core serves is a figure, and its zero a reading.

  * SN1 -- every draw is counted once: served, or sent to the reference with
    its reason -- no core, a core of another generator, a core that offers no
    draw.
  * SN2 -- every score is counted once in the same terms, and a call whose
    shapes are never handed to a core is counted as such.
  * SN3 -- a twin that serves is counted served: the counter leaves zero, and
    what it served is the reference's own.
  * SN4 -- a twin that declines, raises or answers in another shape is counted
    by that reason, and the reference serves the call.
  * SN5 -- every count is taken under the lock, so that the counts stay exact
    on an interpreter where increments race, threads included.
  * SN6 -- the runbook reports the share beside what it measured.
  * SN7 -- a row whose answer, key, negations or canonical flag is not of
    its kind is refused and counted, and the reference draws.
  * SN8 -- a draw that is no list of rows, a kind that is no name, or a
    decision with no key -- one the reference never draws -- is refused and
    counted, and the reference draws.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_SPAN = [
    {"turn_id": "n1", "role": "user", "origin": "typed",
     "text": "We keep the old cluster on 2026-03-04 with Alice Martin."},
    {"turn_id": "n2", "role": "assistant", "origin": "assistant", "text": "Noted: 16 Go per host."},
]
_TEXT = "We keep the old cluster on 2026-03-04 with Alice Martin. Noted: 16 Go per host."


def _window():
    loaded, restore = isolate(
        targets={
            "opti_oignon.memory.probes": source("memory", "probes.py"),
            "opti_oignon.memory.receipts": source("memory", "receipts.py"),
            "opti_oignon.memory.peels": source("memory", "peels.py"),
        },
        packages=("opti_oignon.memory",),
    )
    probes = loaded["opti_oignon.memory.probes"]
    probes._native = lambda: None
    return probes, loaded["opti_oignon.memory.peels"], restore


class _Core:
    """A native core of a stated generator: its draw and its score answer as told."""

    def __init__(self, version, draw=None, score=None):
        if version is not None:
            self.probe_generator_version = version
        if draw is not None:
            self.probe_generate = draw
        if score is not None:
            self.probe_score = score


def _rows(probes, drawn):
    """The reference's probes as the rows a twin of its generator returns, for a span of one piece per turn."""
    index = {turn["turn_id"]: i for i, turn in enumerate(_SPAN)}
    return [(index[q.turn_id], q.kind, q.answer, sorted(q.key), q.negations, q.canonical) for q in drawn]


def test_sn1_every_draw_is_counted_once_served_or_sent_to_the_reference_with_its_reason():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        for core, reason in ((None, "no core"), (_Core(probes.GENERATOR_VERSION - 1, draw=lambda *a: []),
                                                "another generator"),
                             (_Core(probes.GENERATOR_VERSION), "not offered")):
            probes._NATIVE_COUNTS.clear()
            probes._native = lambda core=core: core
            for _ in range(3):
                probes.generate_probes(_SPAN, gate.lexicon)
            assert probes.native_share()["draw"] == {reason: 3}, reason
    finally:
        restore()


def test_sn2_every_score_is_counted_once_and_a_call_never_handed_is_counted_as_such():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_SPAN, gate.lexicon)
        for core, reason in ((None, "no core"), (_Core(probes.GENERATOR_VERSION - 1), "another generator"),
                             (_Core(probes.GENERATOR_VERSION), "not offered")):
            probes._NATIVE_COUNTS.clear()
            probes._native = lambda core=core: core
            probes.score(drawn, _TEXT)
            probes.score(drawn, _TEXT)
            assert probes.native_share()["score"] == {reason: 2}, reason
        probes._NATIVE_COUNTS.clear()
        probes._native = lambda: _Core(probes.GENERATOR_VERSION, score=lambda *a: [])
        probes._native_failures(drawn, b"bytes are never handed")
        assert probes.native_share()["score"] == {"not handed": 1}
    finally:
        restore()


def test_sn3_a_twin_that_serves_is_counted_served_and_serves_the_references_own():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        reference = probes.generate_probes(_SPAN, gate.lexicon)
        failing = [i for i, q in enumerate(reference) if not probes.answers(q, "We keep the old cluster.")]
        assert reference and failing, "control: a draw and failures to serve"
        twin = _Core(probes.GENERATOR_VERSION, draw=lambda *a: _rows(probes, reference), score=lambda *a: list(failing))
        probes._native = lambda: twin
        probes._NATIVE_COUNTS.clear()
        assert probes.generate_probes(_SPAN, gate.lexicon) == reference
        assert probes.score(reference, "We keep the old cluster.").failures == [reference[i] for i in failing]
        assert probes.native_share() == {"draw": {"served": 1}, "score": {"served": 1}}
    finally:
        restore()


def test_sn4_a_twin_that_declines_raises_or_answers_in_another_shape_is_counted_by_that_reason():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        reference = probes.generate_probes(_SPAN, gate.lexicon)

        def raising(*args):
            raise ValueError("a text the core cannot take")

        for draw, reason in ((lambda *a: None, "declined"), (raising, "refused by the core"),
                             (lambda *a: [("one", "field")], "row refused"),
                             (lambda *a: [(9, "date", "2026-03-04", [], 0, True)], "row refused"),
                             (lambda *a: [(-1, "date", "2026-03-04", [], 0, True)], "row refused"),
                             (lambda *a: [(0, "weather", "rain", [], 0, False)], "row refused")):
            probes._native = lambda draw=draw: _Core(probes.GENERATOR_VERSION, draw=draw)
            probes._NATIVE_COUNTS.clear()
            assert probes.generate_probes(_SPAN, gate.lexicon) == reference, reason
            assert probes.native_share()["draw"] == {reason: 1}, reason
        odd = [reference[0].__class__(**{**reference[0].__dict__, "key": ["not", "a", "frozenset"]})]
        for probes_, score, reason in ((reference, lambda *a: None, "declined"), (reference, raising, "refused by the core"),
                                       (odd, lambda *a: [], "probe not handed")):
            probes._native = lambda score=score: _Core(probes.GENERATOR_VERSION, score=score)
            probes._NATIVE_COUNTS.clear()
            probes._native_failures(probes_, _TEXT)
            assert probes.native_share()["score"] == {reason: 1}, reason
    finally:
        restore()


def test_sn5_every_count_is_taken_under_the_lock():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        lock, held = probes._NATIVE_LOCK, []

        class Watched(type(probes._NATIVE_COUNTS)):
            def __setitem__(self, key, value):
                held.append(lock.locked())
                super().__setitem__(key, value)

        probes._NATIVE_COUNTS = Watched()
        drawn = probes.generate_probes(_SPAN, gate.lexicon)
        probes.score(drawn, _TEXT)
        assert len(held) == 2, "control: a draw and a score were counted"
        assert all(held), "a count is taken outside the lock"
        assert probes.native_share() == {"draw": {"no core": 1}, "score": {"no core": 1}}
        barrier, threads = threading.Barrier(4), []

        def work():
            barrier.wait()
            for _ in range(50):
                probes.generate_probes(_SPAN, gate.lexicon)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(held) and probes.native_share()["draw"] == {"no core": 201}, "threads count under the lock too"
    finally:
        restore()


def _runbook():
    """The runbook script, loaded from its file; the path it adds is taken back."""
    spec = importlib.util.spec_from_file_location("oo_onion_runbook", REPO / "scripts" / "onion_runbook.py")
    module = importlib.util.module_from_spec(spec)
    path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = path
    return module


def test_sn6_the_runbook_reports_the_native_share_beside_what_it_measured():
    probes, peels, restore = _window()
    try:
        runbook = _runbook()
        gate = peels.load_gate()
        probes._NATIVE_COUNTS.clear()
        drawn = probes.generate_probes(_SPAN, gate.lexicon)
        probes.score(drawn, _TEXT)
        assert runbook._native_entry() == {"draw": {"no core": 1}, "score": {"no core": 1}, "source": "measured"}
    finally:
        restore()


def test_sn7_a_row_whose_fields_are_not_of_their_kind_is_refused_and_the_reference_draws():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        reference = probes.generate_probes(_SPAN, gate.lexicon)
        assert reference, "control: the reference draws"
        rows = (
            (0, "date", "2026-03-04", ["date:2026-03-04"], None, True),
            (0, "date", "2026-03-04", None, 0, True),
            (0, "date", "2026-03-04", [], -1, True),
            (0, "date", "2026-03-04", [], True, True),
            (0, "date", "2026-03-04", [7], 0, True),
            (0, "date", None, [], 0, True),
            (0, "date", "2026-03-04", [], 0, "yes"),
        )
        drawn, counted = [], []
        for row in rows:
            probes._native = lambda row=row: _Core(probes.GENERATOR_VERSION, draw=lambda *a: [row])
            probes._NATIVE_COUNTS.clear()
            drawn.append(probes.generate_probes(_SPAN, gate.lexicon))
            counted.append(probes.native_share()["draw"])
    finally:
        restore()
    for row, got, count in zip(rows, drawn, counted):
        assert got == reference, row
        assert count == {"row refused": 1}, (row, count)


def test_sn8_a_draw_of_no_rows_a_kind_of_no_name_or_a_keyless_decision_is_refused_and_the_reference_draws():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        reference = probes.generate_probes(_SPAN, gate.lexicon)
        assert reference, "control: the reference draws"
        results = (5, [(0, ["decision"], "We keep it.", ["keep"], 0, False)],
                   [(0, "decision", "We keep the old cluster.", [], 0, False)])
        drawn, counted = [], []
        for result in results:
            probes._native = lambda result=result: _Core(probes.GENERATOR_VERSION, draw=lambda *a: result)
            probes._NATIVE_COUNTS.clear()
            drawn.append(probes.generate_probes(_SPAN, gate.lexicon))
            counted.append(probes.native_share()["draw"])
    finally:
        restore()
    for result, got, count in zip(results, drawn, counted):
        assert got == reference, result
        assert sum(count.values()) == 1 and "served" not in count, (result, count)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
