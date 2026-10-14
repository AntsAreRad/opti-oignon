#!/usr/bin/env python3
"""Contracts for the onion window's budgets, one table per rung, and the one count it is measured with.

The window the onion lays is bounded by a table of caps, one per rung of the
context ladder the resource governor admits contexts on: a table per rung,
no other, one reserve for all, and each layer no smaller at a higher rung. A
context composes under the largest rung at or under it. The ladder is read
from the governor's file and held to what was written, a list of distinct
positive whole numbers, which the governor reads more loosely: anything
else is refused by name. A caller that names no rung composes under the
home rung's table less the system prompt and the tail, which it does not
lay: the window before rungs.

The block is counted once, with the counter of the turn it rides: the
executor hands the librarian that counter, for the model the turn is sent
to, before it asks for the block, and the librarian counts every part of
that turn's block with it. A pin counts the Core as the conversation's last
turn counted it and leaves a share of its cap free, so a Core that lands
still fits a turn that counts it a third more.

  * OW1 -- a context composes under the largest rung at or under it; one
    under the smallest rung, or no whole number, is refused by name.
  * OW2 -- every misconfiguration of the tables is refused by name, every
    reason at once: a rung with no table, a table for no rung, a layer
    missing or unknown, a cap that is a boolean, a table that does not hold
    at its rung, a layer that shrinks as the rung grows, a home rung with no
    table, a reserve that is no whole number, a margin that is no share, the
    keys of the window before rungs, a key named twice where the last would
    win; a missing layer or an unread reserve hides no other reason.
  * OW3 -- a ladder the governor's file does not hold readably is refused
    by name: a file absent or no mapping, a ladder missing or empty, a rung
    that is no positive whole number, a rung named twice.
  * OW4 -- the shipped tables hold: the governor's ladder, each table within
    its rung, no layer shrinking, the chat floor a rung with a table, and a
    caller that names no rung composes under the window before rungs, cap
    for cap.
  * OW5 -- the block is counted with the counter the turn handed over, at
    every part and in the choice of its anchors and peels, per conversation;
    two turns of one conversation in flight each count their own block with
    their own counter; the default counts a conversation no turn handed one
    to; a counter that cannot be called or a model that is no name is
    refused by name.
  * OW6 -- a pin and a supersession are counted as the block will count
    them: what the default would let in and the turn's counter would not is
    refused, naming the model; nothing lands and the block is what it was;
    what the counter lets in composes.
  * OW7 -- an accepted proposal is counted the same way: refused by the
    turn's counter, or before any turn by the heaviest one, nothing lands
    and the proposal stays open.
  * OW8 -- an opened conversation shows the block its turn would send: the
    open counts with the counter the conversation's turn handed over.
  * OW9 -- a pin leaves a share of the Core's cap free (``core_margin``):
    a Core that lands under one count still composes when another model's
    turn counts it a third more; one counted further over its cap is refused
    by name -- the block is empty and the refusal counted -- never cut.
  * OW10 -- before any turn handed a counter over, a pin and a supersession
    count the Core as the heaviest counter a turn may hand over would, so a
    Core that lands then composes under the first turn's count; that
    heaviest count holds against every ratio and code adjustment the
    executor's counter ships.
  * OW11 -- a supersession that does not grow the Core is never refused: a
    Core a turn's counter puts over its cap steps down, entry by entry,
    until its block comes back; one that grows it is still held to its
    room.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source; the registry is blocked, and the
governor's module is not loaded: its file is read as a file.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_ONION_YAML = REPO / "opti_oignon" / "config" / "onion.yaml"
_GOVERNOR_YAML = REPO / "opti_oignon" / "config" / "resource_governor.yaml"
_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")


def _open(*, persisted=False):
    modules = _MODULES + (("onion_store",) if persisted else ())
    loaded, restore = isolate(
        targets={f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in modules},
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils", "opti_oignon.resource_governor"),
        packages=("opti_oignon.memory",),
    )
    lib = loaded["opti_oignon.memory.librarian"]
    lib.reset_librarian()
    return lib, loaded, restore


def _shipped():
    return yaml.safe_load(_ONION_YAML.read_text(encoding="utf-8"))


def _write(tmp_path, onion, ladder=None, name="onion.yaml"):
    """An onion file holding ``onion`` and a governor file holding ``ladder`` (the shipped one when None)."""
    onion_path = tmp_path / name
    onion_path.write_text(yaml.safe_dump(onion, sort_keys=False), encoding="utf-8")
    governor_path = tmp_path / f"governor-{name}"
    held = yaml.safe_load(_GOVERNOR_YAML.read_text(encoding="utf-8"))["ctx_ladder"] if ladder is None else ladder
    governor_path.write_text(yaml.safe_dump({"ctx_ladder": held}), encoding="utf-8")
    return onion_path, governor_path


def _refusal(composer, tmp_path, onion, ladder=None, name="onion.yaml"):
    onion_path, governor_path = _write(tmp_path, onion, ladder, name)
    with pytest.raises(composer.BudgetError) as caught:
        composer.load_budgets(onion_path, governor_path=governor_path)
    return str(caught.value)


def _words(n):
    return lambda text: n * len(text.split())


# ---------------------------------------------------------------------------
# OW1 -- the largest rung at or under the context
# ---------------------------------------------------------------------------
def test_ow1_a_context_composes_under_the_largest_rung_at_or_under_it_and_under_the_smallest_is_refused():
    lib, loaded, restore = _open()
    try:
        composer = loaded["opti_oignon.memory.composer"]
        budgets = composer.load_budgets()
        rungs = budgets.rungs
        assert len(rungs) >= 2 and list(rungs) == sorted(set(rungs))
        for low, high in zip(rungs, rungs[1:]):
            assert budgets.at(low).window == low
            assert budgets.at(high - 1).window == low, "between two rungs, the lower one"
        assert budgets.at(rungs[-1]).window == budgets.at(10 ** 9).window == rungs[-1]
        for num_ctx in (rungs[0] - 1, 0):
            with pytest.raises(composer.BudgetError, match=f"^num_ctx {num_ctx} is under the smallest rung {rungs[0]}"):
                budgets.at(num_ctx)
        for num_ctx in (-1, float(rungs[0]), True, str(rungs[0]), None):
            with pytest.raises(composer.BudgetError, match="is not a whole number of tokens"):
                budgets.at(num_ctx)
    finally:
        restore()


# ---------------------------------------------------------------------------
# OW2 -- every misconfiguration, by name, at once
# ---------------------------------------------------------------------------
def test_ow2_every_misconfiguration_of_the_tables_is_refused_by_name_every_reason_at_once(tmp_path):
    lib, loaded, restore = _open()
    try:
        composer = loaded["opti_oignon.memory.composer"]
        shipped = _shipped()
        onion_path, governor_path = _write(tmp_path, shipped, name="control.yaml")
        assert composer.load_budgets(onion_path, governor_path=governor_path) == composer.load_budgets(), (
            "control: the shipped file, written out again, loads as the shipped one")
        rungs = sorted(shipped["budgets"])
        low, high = rungs[0], rungs[-1]

        def with_tables(change, **top):
            tables = {rung: dict(row) for rung, row in shipped["budgets"].items()}
            change(tables)
            return {**shipped, **top, "budgets": tables}

        cases = {
            "a rung with no table": (with_tables(lambda t: t.pop(rungs[1])),
                                     f"rung {rungs[1]} of ctx_ladder {rungs} has no table"),
            "a table for no rung": (with_tables(lambda t: t.update({low + 1: dict(t[low])})),
                                    f"table {low + 1} is for no rung of ctx_ladder {rungs}"),
            "a missing layer": (with_tables(lambda t: t[low].pop("tail")), f"rung {low}: tail is missing"),
            "an unknown layer": (with_tables(lambda t: t[low].update({"history": 1})),
                                 f"rung {low}: unknown layer 'history'"),
            "a boolean cap": (with_tables(lambda t: t[low].update({"core": True})),
                              f"rung {low}: core: True is not a non-negative integer"),
            "a table over its rung": (with_tables(lambda t: t[low].update({"flesh": t[low]["flesh"] + shipped["reserve"]})),
                                      f"rung {low}: oversubscribed"),
            "a shrinking layer": (with_tables(lambda t: t[high].update({"turn": t[rungs[-2]]["turn"] - 1})),
                                  f"rung {high}: turn {shipped['budgets'][rungs[-2]]['turn'] - 1} is under rung "
                                  f"{rungs[-2]}'s {shipped['budgets'][rungs[-2]]['turn']}: a layer never shrinks"),
            "a home rung with no table": ({**shipped, "home_rung": low + 1}, f"home_rung {low + 1} is no rung with a table"),
            "a reserve that is no whole number": ({**shipped, "reserve": -1}, "reserve: -1 is not a non-negative integer"),
            "a margin that is no share": ({**shipped, "core_margin": 1.0}, "core_margin: 1.0 is not a share in [0, 1)"),
            "a boolean margin": ({**shipped, "core_margin": True}, "core_margin: True is not a share in [0, 1)"),
            "no margin": ({key: value for key, value in shipped.items() if key != "core_margin"},
                          "core_margin: None is not a share in [0, 1)"),
            "the window before rungs": ({**shipped, "window": 12288, "layers": {"core": 1024}},
                                        "window and layers are gone"),
            "the window alone": ({**shipped, "window": 12288}, "window and layers are gone"),
            "the layers alone": ({**shipped, "layers": {"core": 1024}}, "window and layers are gone"),
            "tables that are no mapping": ({**shipped, "budgets": [4096, 8192]}, "budgets: no table per rung"),
            "a table key of zero": (with_tables(lambda t: t.update({0: dict(t[low])})),
                                    "table key 0 is no positive whole number of tokens"),
            "a table key that is text": (with_tables(lambda t: t.update({str(low): dict(t[low])})),
                                         f"table key '{low}' is no positive whole number of tokens"),
            "a table key of one": (with_tables(lambda t: t.update({1: dict(t[low])})),
                                   f"table 1 is for no rung of ctx_ladder {rungs}"),
            "a table that is no mapping": (with_tables(lambda t: t.update({low: 5})),
                                           f"rung {low}: its table is not a mapping of layers"),
            "a cap that is text in a full table": (with_tables(lambda t: t[low].update({"core": "yes"})),
                                                   f"rung {low}: core: 'yes' is not a non-negative integer"),
            "a home rung on a table for no rung": (with_tables(lambda t: (t.update({low + 1: dict(t[low])}),
                                                                          t.update({high: t.pop(high)})),
                                                               home_rung=low + 1),
                                                   f"home_rung {low + 1} is no rung with a table"),
            "no tables": ({key: value for key, value in shipped.items() if key != "budgets"}, "budgets: no table per rung"),
        }
        for i, (case, (onion, words)) in enumerate(cases.items()):
            refusal = _refusal(composer, tmp_path, onion, name=f"case{i}.yaml")
            assert words in refusal, f"{case}: {refusal}"
            if case == "a margin that is no share":
                assert f"rung {low}: core_margin" not in refusal, f"the margin is named once, not per rung: {refusal}"
        top = _refusal(composer, tmp_path, with_tables(lambda t: t[high].update({"core": True})), name="top.yaml")
        assert f"rung {high}: core: True is not" in top and "never shrinks" not in top, (
            f"a table that does not read is compared with no other: {top}")
        onion_path = tmp_path / "zero" / "onion.yaml"
        onion_path.parent.mkdir()
        onion_path.write_text(yaml.safe_dump({**shipped, "core_margin": 0}, sort_keys=False), encoding="utf-8")
        zero = composer.load_budgets(onion_path, governor_path=governor_path)
        assert zero.at(low).core_room() == zero.at(low).core, "a margin of zero reads, and holds nothing back"
        for i, (body, words) in enumerate((("", "budgets: no table per rung"), ("- 4096\n- 8192\n", "holds no mapping"))):
            onion_path = tmp_path / f"bare{i}" / "onion.yaml"
            onion_path.parent.mkdir()
            onion_path.write_text(body, encoding="utf-8")
            with pytest.raises(composer.BudgetError, match=words):
                composer.load_budgets(onion_path, governor_path=governor_path)
        for margin in (1.0, -0.1, True, "0.5", 1.5):
            built = composer.Budget(window=600, reserve=60, core=60, receipts=60, peels=120, flesh=200, turn=100,
                                    core_margin=margin)
            assert f"core_margin: {margin!r} is not a share in [0, 1)" in built.validate(), margin
        for margin in (0, 0.0, 0.25, 0.99):
            built = composer.Budget(window=600, reserve=60, core=60, receipts=60, peels=120, flesh=200, turn=100,
                                    core_margin=margin)
            assert built.validate() == [], f"a budget built by hand with the margin {margin!r} holds"
        both = with_tables(lambda t: (t.pop(rungs[1]), t[low].update({"core": True})), home_rung=low + 1)
        refusal = _refusal(composer, tmp_path, both, name="both.yaml")
        for words in (f"rung {rungs[1]} of ctx_ladder", f"rung {low}: core: True", f"home_rung {low + 1}"):
            assert words in refusal, f"every reason at once: {refusal}"
        hidden = with_tables(lambda t: (t[low].pop("tail"), t[low].update({"core": "yes"}),
                                        t[high].update({"turn": t[rungs[-2]]["turn"] - 1}),
                                        t[rungs[1]].update({"flesh": t[rungs[1]]["flesh"] + shipped["reserve"]})),
                             reserve=2048.0)
        refusal = _refusal(composer, tmp_path, hidden, name="hidden.yaml")
        for words in (f"rung {low}: tail is missing", f"rung {low}: core: 'yes' is not", "reserve: 2048.0 is not",
                      f"rung {high}: turn"):
            assert words in refusal, f"a missing layer or an unread reserve hides no other reason: {refusal}"
        assert f"rung {rungs[1]}: oversubscribed" not in refusal, (
            f"a table filled to its rung is no fault of its own while the reserve does not read: {refusal}")
        written = _ONION_YAML.read_text(encoding="utf-8")
        row = next(line for line in written.splitlines() if line.lstrip().startswith(f"{rungs[1]}:"))
        hex_row = row.replace(f"{rungs[1]}:", f"{hex(rungs[1])}:", 1)
        for i, (twice, words) in enumerate((
                (written.replace(row, row + "\n" + row), f"budgets names {rungs[1]} twice"),
                (written.replace(row, row + "\n" + hex_row), f"budgets names {rungs[1]} twice"),
                (written.replace(row, row.replace("{system:", "{core: 1, system:")), f"the table of {rungs[1]} names 'core' twice"),
                (written + "\nreserve: 1024\n", "onion.yaml names 'reserve' twice"))):
            onion_path = tmp_path / f"twice{i}" / "onion.yaml"
            onion_path.parent.mkdir()
            onion_path.write_text(twice, encoding="utf-8")
            with pytest.raises(composer.BudgetError) as caught:
                composer.load_budgets(onion_path, governor_path=governor_path)
            assert words in str(caught.value) and "the last would win without a word" in str(caught.value), caught.value
        low_row = next(line for line in written.splitlines() if line.lstrip().startswith(f"{low}:"))
        merged = written.replace(low_row, low_row.replace(f"{low}:", f"{low}: &low", 1)).replace(
            row, row.replace("{system:", "{<<: *low, system:"))
        onion_path = tmp_path / "merged" / "onion.yaml"
        onion_path.parent.mkdir()
        onion_path.write_text(merged, encoding="utf-8")
        assert composer.load_budgets(onion_path, governor_path=governor_path) == composer.load_budgets(), (
            "a table merged from another, every layer then written out, reads as the shipped one: a merge is no key "
            "named twice")
    finally:
        restore()


# ---------------------------------------------------------------------------
# OW3 -- a ladder that does not read
# ---------------------------------------------------------------------------
def test_ow3_a_ladder_the_governors_file_does_not_hold_readably_is_refused_by_name(tmp_path):
    lib, loaded, restore = _open()
    try:
        composer = loaded["opti_oignon.memory.composer"]
        shipped = _shipped()
        rungs = sorted(shipped["budgets"])
        assert composer.read_ladder() == (tuple(rungs), ""), "control: the shipped ladder reads"
        cases = (
            ([], "missing or empty"),
            ("8192", "missing or empty"),
            (rungs[:-1] + [0], "holds a value that is no positive whole number"),
            (rungs[:-1] + [-rungs[-1]], "holds a value that is no positive whole number"),
            (rungs[:-1] + [float(rungs[-1])], "holds a value that is no positive whole number"),
            (rungs[:-1] + [True], "holds a value that is no positive whole number"),
            (rungs + [rungs[0]], "names a rung twice"),
        )
        for i, (ladder, words) in enumerate(cases):
            onion_path, governor_path = _write(tmp_path, shipped, ladder, name=f"ladder{i}.yaml")
            ladder_read, why = composer.read_ladder(governor_path)
            assert ladder_read is None and why.startswith(f"ctx_ladder unreadable in {governor_path.name}: ") and words in why, why
            with pytest.raises(composer.BudgetError, match="ctx_ladder unreadable"):
                composer.load_budgets(onion_path, governor_path=governor_path)
        one = tmp_path / "one.yaml"
        one.write_text("ctx_ladder: [1, 2]\n", encoding="utf-8")
        assert composer.read_ladder(one) == ((1, 2), ""), "a rung of one token is a positive whole number"
        absent = tmp_path / "absent.yaml"
        assert composer.read_ladder(absent) == (None, "ctx_ladder unreadable in absent.yaml: FileNotFoundError")
        listed = tmp_path / "listed.yaml"
        listed.write_text("- 4096\n- 8192\n", encoding="utf-8")
        assert composer.read_ladder(listed) == (None, "ctx_ladder unreadable in listed.yaml: missing or empty")
        onion_path, governor_path = _write(tmp_path, shipped, name="unordered.yaml")
        governor_path.write_text(yaml.safe_dump({"ctx_ladder": list(reversed(rungs))}), encoding="utf-8")
        assert composer.load_budgets(onion_path, governor_path=governor_path).rungs == tuple(rungs), (
            "the governor writes its ladder from the top; the onion reads it in either order")
    finally:
        restore()


# ---------------------------------------------------------------------------
# OW4 -- the shipped tables
# ---------------------------------------------------------------------------
def test_ow4_the_shipped_tables_hold_and_a_caller_that_names_no_rung_composes_under_the_window_before_rungs():
    lib, loaded, restore = _open()
    try:
        composer = loaded["opti_oignon.memory.composer"]
        governor = yaml.safe_load(_GOVERNOR_YAML.read_text(encoding="utf-8"))
        shipped = _shipped()
        budgets = composer.load_budgets()
        assert budgets.rungs == tuple(sorted(governor["ctx_ladder"]))
        assert governor["ctx_floor"]["chat"] in budgets.rungs, "the smallest context chat is admitted at has a table"
        for (low, below), (high, above) in zip(budgets.tables, budgets.tables[1:]):
            for layer in composer.LAYERS:
                assert getattr(below, layer) <= getattr(above, layer), f"{layer} never shrinks from {low} to {high}"
        for rung, budget in budgets.tables:
            assert sum(shipped["budgets"][rung].values()) + shipped["reserve"] <= rung, f"the table holds at {rung}"
        assert composer.load_budget() == composer.Budget(
            window=12288, reserve=2048, core=1024, receipts=512, peels=3072, flesh=5120, turn=512, system=0, tail=0,
            core_margin=0.25,
        ), "the window before rungs, cap for cap: a caller that names no rung composes as it always did"
        assert composer.load_budget().core_room() == 767, (
            "a pin leaves a quarter of the Core's cap free, and a token for the counts' rounding down")
        assert budgets.home == 16384 and budgets.at(16384).window == 16384
        import dataclasses

        for fixed, field in ((budgets, "reserve"), (budgets.at(16384), "core")):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(fixed, field, 1)
    finally:
        restore()


# ---------------------------------------------------------------------------
# OW5 and OW6 -- the one count
# ---------------------------------------------------------------------------
def _messages(n):
    # Turns of no origin that decide nothing, the wording of the queue's own
    # fixture: a summary that repeats them asserts no decision.
    out = []
    for i in range(1, n + 1):
        role = "user" if i % 2 else "assistant"
        out.append({"role": role, "content": f"Turn {i}: Alice reviewed service {i} on 2026-03-{i:02d} and we stated "
                                             f"that service {i} lives on the new cluster."})
    return out


def _small(composer, **over):
    fields = dict(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=140, turn=60)
    fields.update(over)
    return composer.Budget(**fields)


def _grown(lib, loaded, cid):
    """A conversation with a Core entry and more peels than its peels cap holds."""
    peels, core_store, composer = (loaded[f"opti_oignon.memory.{m}"] for m in ("peels", "core_store", "composer"))
    gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
    state = lib.state_for(cid)
    state.core.add("Answers are concise.", actor=core_store.USER)
    state.mirror(_messages(24))
    while lib.curate(state, lambda turns: " ".join(t["text"] for t in turns), gate=gate,
                     budget=_small(composer)).evicted:
        pass
    assert len(state.tree.all()) >= 5, "control: enough peels to overflow the peels cap"
    return state


def _peel_texts(block):
    texts, current = [], None
    for line in str(block).splitlines():
        if line.startswith("[data layer=peels"):
            current = []
        elif line == "[/data]" and current is not None:
            texts.append("\n".join(current))
            current = None
        elif current is not None:
            current.append(line)
    return texts


def _recording(loaded):
    """The selections of the window's peels module, each recording the counter it is handed."""
    peels = loaded["opti_oignon.memory.peels"]
    handed = []
    real = {"anchors": peels.select_anchors, "peels": peels.select_peels}

    def recorder(kind):
        def select(*args, estimate=None, **kwargs):
            handed.append((kind, estimate))
            return real[kind](*args, estimate=estimate, **kwargs)
        return select

    peels.select_anchors, peels.select_peels = recorder("anchors"), recorder("peels")
    return handed


def test_ow5_the_block_is_counted_with_the_counter_the_turn_handed_over_at_every_part():
    import threading

    lib, loaded, restore = _open()
    try:
        composer = loaded["opti_oignon.memory.composer"]
        budget = _small(composer)
        _grown(lib, loaded, "c1")
        _grown(lib, loaded, "c2")
        question = "service reviewed by alice on the new cluster"
        by_default = lib.memory_block("c1", question, budget=budget)
        assert by_default and lib.memory_block("c2", question, budget=budget) == by_default, "control: twins"
        seen = []

        def heavy(text):
            seen.append(text)
            return 4 * len(text.split())

        handed = _recording(loaded)
        lib.use_counter("c1", heavy, "big:7b")
        counted = lib.memory_block("c1", question, budget=budget)
        assert counted and counted != by_default, "the turn's counter changes what the block holds"
        assert [kind for kind, _estimate in handed] == ["anchors", "peels"], "control: both selections were asked"
        assert all(estimate("one two three") == 12 for _kind, estimate in handed), (
            "the anchors and the peels are chosen with the turn's counter")
        placed = _peel_texts(counted)
        assert 1 <= len(placed) < len(_peel_texts(by_default)), "fewer peels fit when each counts more"
        assert sum(4 * len(text.split()) for text in placed) <= budget.peels, "they fit their cap as the turn counts"
        assert "Answers are concise." in seen, "the Core is counted with it"
        assert lib.memory_block("c2", question, budget=budget) == by_default, "another conversation keeps its count"
        lib.use_counter("c1", composer.estimate_tokens, "small:1b")
        assert lib.memory_block("c1", question, budget=budget) == by_default, "the last counter handed over counts"
        lib.use_counter(None, heavy, "big:7b")
        lib.use_counter("", heavy, "big:7b")
        assert lib._counter(None)[1] is None and lib._counter("")[1] is None, "a turn of no conversation hands nothing"
        for count, model, words in ((None, "big:7b", "cannot be called"), ("4", "big:7b", "cannot be called"),
                                    (heavy, "", "a model by name"), (heavy, None, "a model by name")):
            with pytest.raises(TypeError, match=words):
                lib.use_counter("c1", count, model)
        refused = lib.counters().get("block", {}).get("refused_BudgetError", 0)
        lib.use_counter("c1", lambda text: None, "broken:7b")
        assert lib.memory_block("c1", question, budget=budget) == "", "a counter that gives no number lays nothing"
        assert lib.counters()["block"]["refused_BudgetError"] == refused + 1, "and the refusal is counted by name"
        with pytest.raises(composer.BudgetError, match="the counter of broken:7b gave None"):
            lib.pin("c1", "Dates are written in full.", actor="user", budget=budget)
        lib.use_counter("c1", lambda text: 2 ** 32, "huge:7b")
        with pytest.raises(composer.BudgetError, match="the counter of huge:7b gave 4294967296: no whole number"):
            lib.pin("c1", "Dates are written in full.", actor="user", budget=budget)
        lib.use_counter("c1", lambda text: 2 ** 32 - 1, "huge:7b")
        with pytest.raises(lib.LibrarianError, match="the pin would bring the Core to 4294967295 tokens"):
            lib.pin("c1", "Dates are written in full.", actor="user", budget=budget)

        blocks, both = {}, threading.Barrier(2)

        def turn(model, count):
            lib.use_counter("c1", count, model)
            both.wait(timeout=30)
            blocks[model] = lib.memory_block("c1", question, budget=budget)

        threads = [threading.Thread(target=turn, args=(model, count))
                   for model, count in (("big:7b", heavy), ("small:1b", composer.estimate_tokens))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert blocks == {"big:7b": counted, "small:1b": by_default}, (
            "two turns of one conversation in flight: each block is counted with its own turn's counter")

        rules = " ".join(f"rule{i}" for i in range(20))
        pinned, registered, overtaken = {}, threading.Event(), threading.Event()

        def stale():
            lib.use_counter("c3", heavy, "big:7b")
            registered.set()
            overtaken.wait(timeout=30)
            try:
                pinned["c3"] = bool(lib.pin("c3", rules, actor="user", budget=budget))
            except lib.LibrarianError as exc:
                pinned["c3"] = str(exc)

        thread = threading.Thread(target=stale)
        thread.start()
        registered.wait(timeout=30)
        lib.use_counter("c3", composer.estimate_tokens, "small:1b")
        overtaken.set()
        thread.join(timeout=60)
        assert pinned == {"c3": True}, (
            f"a pin counts with the last counter handed over, never a hand-over left on its thread: {pinned}")
        lib.reset_librarian()
        assert lib._counter("c3")[1] is None, "a reset forgets the counters handed over"
    finally:
        restore()


def test_ow6_a_pin_and_a_supersession_are_counted_as_the_block_will_count_them():
    lib, loaded, restore = _open()
    try:
        composer = loaded["opti_oignon.memory.composer"]
        budget = _small(composer, core=24)
        question = "anything"
        for cid in ("c1", "c2"):
            lib.pin(cid, "Answers cite their source.", actor="user", budget=budget)
        lib.use_counter("c1", _words(4), "big:7b")
        before = lib.memory_block("c1", question, budget=budget)
        assert before, "control: a block before the pin"
        pinned = "Dates are written in full."
        assert composer.estimate_tokens("Answers cite their source.\n" + pinned) <= budget.core, (
            "control: the default count would let it in")
        with pytest.raises(lib.LibrarianError) as refused:
            lib.pin("c1", pinned, actor="user", budget=budget)
        assert "as the counter of big:7b counts them" in str(refused.value) and str(budget.core) in str(refused.value)
        assert "against a cap of 24:" in str(refused.value), "with no share held back, the room is the cap, said so"
        assert [e.text for e in lib.core_entries("c1")] == ["Answers cite their source."], "nothing landed"
        assert lib.memory_block("c1", question, budget=budget) == before, "the block is what it was, not blank"
        assert lib.pin("c2", pinned, actor="user", budget=budget), "control: with no counter handed over it lands"
        (entry,) = [e for e in lib.core_entries("c1")]
        assert composer.estimate_tokens("Answers always cite their source in full.") <= budget.core, "control"
        with pytest.raises(lib.LibrarianError, match="as the counter of big:7b counts them"):
            lib.supersede("c1", entry.id, "Answers always cite their source in full.", actor="user", budget=budget)
        assert [e.text for e in lib.core_entries("c1")] == ["Answers cite their source."]
        lib.supersede("c1", entry.id, "Cite sources.", actor="user", budget=budget)
        block = lib.memory_block("c1", question, budget=budget)
        assert block and "Cite sources." in block, "what the counter lets in composes"
    finally:
        restore()


_TYPED = [
    {"role": "user", "origin": "typed", "segments": [],
     "content": "Alice moved the build to Berlin on 2026-03-04. We keep Docker on the build server."},
    {"role": "assistant", "origin": "assistant", "segments": [],
     "content": "Noted: the Berlin build runs 12 jobs a day, a sensible load for that machine. "
                "Bob checks the logs every morning."},
]
_DECISION = "We keep Docker on the build server."
_LOSSY = ("Alice moved the build to Berlin on 2026-03-04. The Berlin build runs 12 jobs a day. "
          "Bob checks the logs every morning.")


def test_ow7_an_accepted_proposal_is_counted_as_the_block_will_count_it():
    lib, loaded, restore = _open()
    try:
        peels, composer = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.composer"]
        gate = replace(peels.load_gate(), span_turns=2)
        ladder = replace(peels.load_ladder(), rho=0.5, proposals_per_day=5)
        tiny = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)
        lib._today = lambda: "2026-10-06"
        for cid in ("c1", "c2"):
            state = lib.state_for(cid)
            state.mirror(_TYPED)
            assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny, ladder=ladder).rung == "held"
        (offered,) = lib.proposals("c1")
        assert offered["text"] == _DECISION, "control: the decision is offered"
        core = composer.estimate_tokens(_DECISION) + 1
        budget = replace(tiny, core=core)
        lib.use_counter("c1", _words(4), "big:7b")
        with pytest.raises(lib.LibrarianError, match="as the counter of big:7b counts them"):
            lib.accept_proposal("c1", offered["id"], actor="user", budget=budget)
        assert lib.core_entries("c1") == [] and [p["id"] for p in lib.proposals("c1")] == [offered["id"]], (
            "nothing landed, and the proposal is still open")
        (other,) = lib.proposals("c2")
        with pytest.raises(lib.LibrarianError, match="as the heaviest counter a turn may hand over counts them"):
            lib.accept_proposal("c2", other["id"], actor="user", budget=budget)
        assert lib.core_entries("c2") == [], "before any turn, it is counted as the heaviest counter would"
        lib.use_counter("c2", composer.estimate_tokens, "small:1b")
        assert lib.accept_proposal("c2", other["id"], actor="user", budget=budget), (
            "control: under a turn's counter that counts it within the room, it lands")
    finally:
        restore()


# ---------------------------------------------------------------------------
# OW8 -- an open counts as the turn would
# ---------------------------------------------------------------------------
def test_ow8_an_opened_conversation_shows_the_block_its_turn_would_send(tmp_path):
    import sqlite3

    lib, loaded, restore = _open(persisted=True)
    try:
        composer, store_mod = loaded["opti_oignon.memory.composer"], loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "onion.db"
        cfg = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=4, temperature=0.1,
                                  num_predict=128, persist_path=str(path), require_encryption=False)
        lib._store[(str(path), False)] = store_mod.OnionStore(path, connect=lambda p: sqlite3.connect(str(p)),
                                                              require_encryption=False)
        peels = loaded["opti_oignon.memory.peels"]
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        budget = _small(composer)
        state = lib.state_for("c1", cfg)
        state.core.add("Answers are concise.", actor=loaded["opti_oignon.memory.core_store"].USER)
        state.mirror(_messages(24))
        while lib.curate(state, lambda turns: " ".join(t["text"] for t in turns), gate=gate, budget=budget).evicted:
            pass
        lib._save_state("c1", state, cfg)
        question = "service reviewed by alice on the new cluster"
        by_default = lib.open_onion("c1", question, config=cfg, budget=budget).block
        assert by_default == lib.memory_block("c1", question, budget=budget, config=cfg), "control: the same block"
        lib.use_counter("c1", _words(4), "big:7b")
        opened = lib.open_onion("c1", question, config=cfg, budget=budget).block
        assert opened and opened != by_default, "the open counts with the turn's counter"
        assert opened == lib.memory_block("c1", question, budget=budget, config=cfg), "the block the turn sends"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OW9 -- the share of the Core's cap held back
# ---------------------------------------------------------------------------
def test_ow9_a_core_that_lands_still_composes_when_another_model_counts_it_a_third_more():
    lib, loaded, restore = _open()
    try:
        composer = loaded["opti_oignon.memory.composer"]
        budget = composer.load_budget()
        assert (budget.core, budget.core_room()) == (1024, 767), "control: the shipped cap, and the room a pin has"
        rules = " ".join(f"rule{i}" for i in range(590))
        assert composer.estimate_tokens(rules) == 767, "control: the default counts it within the room"
        for cid in ("c1", "c2"):
            lib.use_counter(cid, composer.estimate_tokens, "small:1b")
        lib.pin("c1", rules, actor="user")
        with pytest.raises(lib.LibrarianError, match="767, its cap of 1024 less the share held back"):
            lib.pin("c2", " ".join(f"rule{i}" for i in range(592)), actor="user")
        assert lib.core_entries("c2") == [], "past the room, nothing lands"
        (entry,) = lib.core_entries("c1")
        laws = " ".join(f"law{i}" for i in range(590))
        lib.supersede("c1", entry.id, laws, actor="user")
        assert [e.text for e in lib.core_entries("c1") if not e.superseded_by] == [laws], "at the room, it lands"
        lib.use_counter("c1", lambda text: len(text.split()) * 26 // 15, "third:7b")
        block = lib.memory_block("c1", "anything")
        assert block and laws in block, "a turn that counts the Core a third more still has the block"
        refused = lib.counters().get("block", {}).get("refused_BudgetError", 0)
        lib.use_counter("c1", lambda text: len(text.split()) * 39 // 20, "half:7b")
        assert lib.memory_block("c1", "anything") == "", "a turn that counts it over its cap has no block"
        assert lib.counters()["block"]["refused_BudgetError"] == refused + 1, "and the refusal is counted by name"
        assert [e.text for e in lib.core_entries("c1") if not e.superseded_by] == [laws], "the Core is never cut"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OW10 -- before any turn, the heaviest count
# ---------------------------------------------------------------------------
_PROSE = "Le jardinier arrose les oignons du potager chaque matin, avant le lever du soleil, sans en oublier un seul."


def test_ow10_a_core_grown_before_any_turn_is_counted_as_the_heaviest_counter_would():
    import importlib.util
    import math

    lib, loaded, restore = _open()
    try:
        composer = loaded["opti_oignon.memory.composer"]
        budget = composer.load_budget()
        heavy = " ".join([_PROSE] * 30)
        assert composer.estimate_tokens(heavy) <= budget.core_room() < math.ceil(len(heavy) / 2.5), (
            "control: the default would let it in, the heaviest count would not")
        with pytest.raises(lib.LibrarianError, match="as the heaviest counter a turn may hand over counts them"):
            lib.pin("c1", heavy, actor="user")
        assert lib.core_entries("c1") == [], "nothing landed"
        light = " ".join([_PROSE] * 12)
        assert lib.pin("c1", light, actor="user"), "within the room as the heaviest count goes, it lands"
        (entry,) = lib.core_entries("c1")
        with pytest.raises(lib.LibrarianError, match="as the heaviest counter a turn may hand over counts them"):
            lib.supersede("c1", entry.id, heavy, actor="user")
        assert [e.text for e in lib.core_entries("c1")] == [light], "a supersession is counted the same way"
        lib.use_counter("c1", lambda text: math.ceil(len(text) / 3.0), "codellama:7b")
        block = lib.memory_block("c1", "anything")
        assert block and light in block, "and the first turn's block, counted by the heaviest family, holds it"

        spec = importlib.util.spec_from_file_location("context_counts_under_contract", source("context_manager.py"))
        counts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(counts)
        ratios = list(counts._FAMILY_CHARS_PER_TOKEN.values()) + [counts._DEFAULT_CHARS_PER_TOKEN]
        ratios += [limits.chars_per_token for limits in counts.DEFAULT_MODEL_LIMITS.values()]
        heaviest = min(ratios) / max(counts._CODE_LANGUAGE_ADJUSTMENTS.values())
        assert lib._FEWEST_CHARS_PER_TOKEN <= heaviest, (
            f"a shipped counter takes {heaviest} characters a token, fewer than the onion's bound")
        assert heaviest - lib._FEWEST_CHARS_PER_TOKEN < 0.01, "and the bound is that counter's, not a looser one"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OW11 -- a Core over its cap steps down
# ---------------------------------------------------------------------------
def test_ow11_a_supersession_that_does_not_grow_the_core_is_never_refused_so_a_core_over_its_cap_steps_down():
    lib, loaded, restore = _open()
    try:
        core_store = loaded["opti_oignon.memory.core_store"]
        state = lib.state_for("c1")
        # A Core admitted by words alone, as before this change: sixty rules of
        # ten words, 780 by the default count, under the old cap of 1024.
        for i in range(60):
            state.core.add(f"Rule {i}: answers name their source, their date, their author.", actor=core_store.USER)
        lib.use_counter("c1", _words(2), "heavy:7b")
        assert lib.memory_block("c1", "anything") == "", "control: the turn's counter puts the Core over its cap"
        entries = lib.core_entries("c1")
        with pytest.raises(lib.LibrarianError, match="the supersession would bring the Core to 1206 tokens"):
            lib.supersede("c1", entries[0].id, entries[0].text + " Every rule binds.", actor="user")
        for i, entry in enumerate(entries[:11]):
            lib.supersede("c1", entry.id, f"Cite{i} sources.", actor="user")
        block = lib.memory_block("c1", "anything")
        assert block and "Cite10 sources." in block, "eleven steps down, the Core is within its cap and the block is back"
        assert len(lib.core_entries("c1")) == 60 + 11, "the old entries stay, linked to their successors"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
