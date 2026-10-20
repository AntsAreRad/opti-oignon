#!/usr/bin/env python3
"""Contracts for the paired windows: a change of the onion's composition measured on the same states.

A drift A/B run's later states follow its own answers, so two trees run
end to end differ by their course as much as by their composition. The
bench keeps, at each probe turn, the onion arm's state before its window is
planned (``scripts/drift_ab.py --states``), and ``scripts/drift_pairs.py``
composes that window again under another tree, at the home rung the run
was pinned to (``--home-rung``).

  * DP1 -- THE ARM KEEPS EACH PROBE'S STATE BEFORE ITS PLAN: with a
    directory, each turn numbered among those kept leaves its store --
    kept before the turn's window is planned -- its turn, the conversation
    as stored before it, the system prompt and home rung it is planned
    under and whether a burst held it, and the window it sent with its
    answer, exactly the messages sent; no other turn anything; without a
    directory nothing is written anywhere; the host command takes only a
    new directory, and refuses a rung it cannot pin, asking no model.
  * DP2 -- A KEPT STATE COMPOSES AGAIN TO THE WINDOW IT SENT: loaded from a
    copy of its store under the same package, each kept state plans and
    composes the window the arm sent, byte for byte, the memory engaged;
    the record is never opened, and left as it was.
  * DP3 -- THE RUNG PINNED IN THE PROCESS: inside the pin the onion's
    loaders read home_rung R, the journal's tool room and result cap scaled
    to keep their value at each rung, and the machine's local file beside
    the copy as beside the file; after it -- whatever ends it -- the file
    again, its copy gone; a rung the loaders refuse, and a configuration
    without home_rung, refused by name, nothing pointed elsewhere.
  * DP4 -- TWO TREES COMPARED PROBE BY PROBE: the windows that differ by
    their messages, rung or context; the refusals; and, when both were
    asked, the probes each answers right and what the second gains and
    loses -- none of that when either was not asked; a probe that is not
    the same in both, refused.
  * DP5 -- COMPOSE, END TO END: every kept probe composed again under the
    home rung it was kept under, asked and read: the window the run sent,
    the verdict, the tree that composed; ``turns`` keeps some; a kept turn
    that is not this tree's probe, a rung that is not the states', and a
    model no backend serves are refused and write nothing.
  * DP6 -- WHAT THE LIBRARIAN READS, LAID ON THE SAME STATES: ``read`` asks
    the model as the librarian asks it, each span the kept states hold
    once, oldest first, with the subjects known before it, and writes each
    span's statements by its key; ``compose --statements`` lays them in
    each state before its window is planned and names the file it laid; a
    model no backend serves, a reading the model fails, a file that holds
    no statements by span or nothing of a span a state holds, and a tree
    whose librarian reads no statement are refused, and write nothing.
  * DP7 -- A KEPT STATE SAYS WHAT IT CARRIED READ: a state kept with what
    was read of its spans says, in each line ``compose`` writes, how many
    spans it carried read, their digest and whether this tree laid them;
    laid with ``--statements``, the file's take their place and the line
    says the file alone; ``read`` lays what a state carried aside whole, so
    no span is read again for the reader that read it.

Local-only (the public distribution ships no tests). The scripts load as
the drift A/B suite loads its own (tests/test_drift_ab_contracts.py); the
onion's modules come through the shared isolation window, and the model is
a recording seam.
"""

import dataclasses
import hashlib
import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_drift_ab_contracts as da  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_dp1_the_arm_keeps_each_probes_state_before_its_plan": 30.0,
    "test_dp2_a_kept_state_composes_again_to_the_window_it_sent": 20.0,
    "test_dp3_the_rung_pinned_in_the_process": 5.0,
    "test_dp4_two_trees_compared_probe_by_probe": 2.0,
    "test_dp5_compose_end_to_end": 20.0,
    "test_dp6_what_the_librarian_reads_laid_on_the_same_states": 30.0,
    "test_dp7_a_kept_state_says_what_it_carried_read": 45.0,
}

_ONION = ("probes", "drift", "core_store", "receipts", "composer", "peels", "librarian", "onion_store")
_PAIRS = Path(__file__).resolve().parent.parent / "scripts" / "drift_pairs.py"
_KEPT = (16, 26, 36)
_STORE = "opti_oignon.memory.onion_store"
_WRAP = "opti_oignon.agent.untrusted_context"


def _window():
    """The onion's modules, its store among them, and the untrusted-data envelope, each from its file, in the
    shared window."""
    targets = {f"opti_oignon.memory.{name}": source("memory", f"{name}.py") for name in _ONION}
    targets[_WRAP] = source("agent", "untrusted_context.py")
    return isolate(targets=targets, packages=("opti_oignon.memory", "opti_oignon.agent"))


def _pairs():
    spec = importlib.util.spec_from_file_location("drift_pairs_under_contract", _PAIRS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(lib):
    return lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=1, temperature=0.1,
                               num_predict=64, persist_path="", require_encryption=False)


def _kept_run(ab, lib, loaded, states, turns=40):
    """The bench's first ``turns`` through the onion arm, curated after each one to a small Flesh, the probes 16,
    26 and 36 kept under ``states``: the messages sent, turn by turn."""
    composer = loaded["opti_oignon.memory.composer"]
    sent = []

    def ask(messages):
        sent.append(messages)
        return "Noted."

    budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)
    arm = ab.onion_arm(ask, librarian=lib, config=_config(lib), summarize=da._faithful, wrap=loaded[_WRAP].wrap,
                       budget=budget, states=states, record=set(_KEPT))
    ab.run_arm(ab.TURNS[:turns], arm)
    return sent


class _Answering:
    """A backend that answers each probe with the fact it expects, and keeps the models it was asked for."""

    def __init__(self, ab):
        self._ab, self.models = ab, []

    def generate(self, model, messages, **kwargs):
        self.models.append(model)
        asked = messages[-1]["content"]
        turn = next(turn for turn in self._ab.TURNS if turn.probe and asked.endswith(turn.text))
        return SimpleNamespace(content=f"{turn.expects}.")


# ---------------------------------------------------------------------------
# DP1 -- the arm keeps each probe's state before its plan
# ---------------------------------------------------------------------------
def test_dp1_the_arm_keeps_each_probes_state_before_its_plan(tmp_path, monkeypatch):
    """DP1: the kept turns' store (before the plan), turn, conversation, system prompt, home rung and window, and
    nothing else; nothing anywhere without a directory; the host command takes only a new directory and a rung it
    can pin."""
    ab = da._script()
    loaded, restore = _window()
    lib, composer = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.composer"]
    try:
        states = tmp_path / "states"
        planned, plan = [], lib.plan_window

        def spying(conversation_id, **kwargs):
            text = kwargs["turn"]["content"]
            for turn in _KEPT:
                if text == ab.TURNS[turn - 1].text:
                    planned.append((turn, (states / f"t{turn:03d}.db").is_file(),
                                    (states / f"t{turn:03d}.window.json").exists()))
            return plan(conversation_id, **kwargs)

        lib.plan_window = spying
        sent = _kept_run(ab, lib, loaded, states)
        lib.plan_window = plan
        assert planned == [(turn, True, False) for turn in _KEPT], ("each state kept before its plan", planned)
        names = sorted(path.name for path in states.iterdir())
        assert names == sorted(f"t{turn:03d}{suffix}" for turn in _KEPT
                               for suffix in (".db", ".json", ".window.json")), ("the kept turns alone", names)
        for turn in _KEPT:
            meta = json.loads((states / f"t{turn:03d}.json").read_text(encoding="utf-8"))
            assert meta["question"] == ab.TURNS[turn - 1].text and meta["turn"] == turn, ("the turn", turn)
            assert len(meta["history"]) == 2 * (turn - 1) and meta["history"][-1]["origin"] == "assistant", (
                "the conversation as stored before the turn", turn)
            assert meta["system"] == ab.SYSTEM_PROMPT and meta["home_rung"] == composer.load_budgets().home, (
                "the system prompt and the home rung it is planned under", turn)
            assert meta["held"] is False, ("no burst holds a conversation curated on the answering thread", turn)
            window = json.loads((states / f"t{turn:03d}.window.json").read_text(encoding="utf-8"))
            assert window["messages"] == sent[turn - 1] and window["reply"] == "Noted.", (
                "the window kept is the one sent, with its answer", turn)
            store = loaded[_STORE].OnionStore(states / f"t{turn:03d}.db", connect=sqlite3.connect,
                                              require_encryption=False)
            assert store.conversations() == [meta["cid"]], ("the store holds the conversation", turn)
        lib.reset_librarian()
        bare = tmp_path / "none"
        bare.mkdir()
        monkeypatch.chdir(bare)
        sent_bare = []
        arm = ab.onion_arm(lambda messages: sent_bare.append(messages) or "Noted.", librarian=lib,
                           config=_config(lib), summarize=da._faithful, wrap=loaded[_WRAP].wrap, record=set(_KEPT))
        ab.run_arm(ab.TURNS[:20], arm)
        assert len(sent_bare) == 20 and not any(bare.iterdir()), "without a directory, nothing written"
        with pytest.raises(ValueError, match="directory"):
            ab.onion_arm(lambda messages: "", librarian=lib, config=_config(lib), summarize=da._faithful,
                         wrap=loaded[_WRAP].wrap, states="")
        asked = []
        for argv in (["--states", str(states)], ["--states", ""], ["--home-rung", "5000"]):
            code = ab.main(argv + ["--model", "m"], resolve=lambda name: asked.append(name))
            assert code == 2 and asked == [], ("refused before any model is asked", argv, code)
        assert sorted(path.name for path in states.iterdir()) == names, "the run's states left as they were"
    finally:
        lib.reset_librarian()
        restore()


# ---------------------------------------------------------------------------
# DP2 -- a kept state composes again to the window it sent
# ---------------------------------------------------------------------------
def test_dp2_a_kept_state_composes_again_to_the_window_it_sent(tmp_path):
    """DP2: each kept state, loaded from a copy of its store, composes the window sent, byte for byte; the record
    never opened, left as it was."""
    ab, pairs = da._script(), _pairs()
    loaded, restore = _window()
    lib, store = loaded["opti_oignon.memory.librarian"], loaded[_STORE]
    shipped = store.OnionStore
    try:
        states = tmp_path / "states"
        _kept_run(ab, lib, loaded, states)
        before = {path.name: path.read_bytes() for path in states.iterdir()}
        assert pairs.recorded(states) == list(_KEPT), "control: the kept turns, in order"
        opened = []

        class Opened(shipped):
            def __init__(self, path, **kwargs):
                opened.append(Path(path))
                super().__init__(path, **kwargs)

        store.OnionStore = Opened
        engaged = []
        for turn in _KEPT:
            meta, window = pairs.compose_recorded(lib, store, states, turn, wrap=ab._envelope(loaded[_WRAP].wrap),
                                                  counter=lambda _model: ab.estimate_tokens,
                                                  system=ab.SYSTEM_PROMPT)
            assert window.refusal is None and meta["turn"] == turn, (turn, window.refusal)
            kept = json.loads((states / f"t{turn:03d}.window.json").read_text(encoding="utf-8"))
            messages = [{"role": role, "content": content} for role, content, _label in window.messages]
            assert messages == kept["messages"] and (window.rung, window.num_ctx) == (kept["rung"], kept["num_ctx"]), (
                "the window the arm sent, composed again", turn)
            layers = dict(window.layers)
            engaged.append(layers["peels"] + layers["receipts"] > 0)
        assert all(engaged), ("control: the memory engaged at every kept probe", engaged)
        assert opened and not any(states in path.parents for path in opened), ("the record never opened", opened)
        assert {path.name: path.read_bytes() for path in states.iterdir()} == before, "the record left as it was"
    finally:
        store.OnionStore = shipped
        lib.reset_librarian()
        restore()


# ---------------------------------------------------------------------------
# DP3 -- the rung pinned in the process
# ---------------------------------------------------------------------------
def test_dp3_the_rung_pinned_in_the_process(tmp_path):
    """DP3: inside the pin, home_rung R, the tool room and result cap scaled, the local file read beside the copy;
    after it, whatever ends it, the file, its copy gone; a rung refused and a file without home_rung refused,
    nothing pointed elsewhere."""
    ab = da._script()
    loaded, restore = _window()
    composer = loaded["opti_oignon.memory.composer"]
    lib, peels = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.peels"]
    modules = (composer, lib, peels)
    shipped = [module._CONFIG for module in modules]
    try:
        machine = tmp_path / "config"
        machine.mkdir()
        shutil.copyfile(shipped[0], machine / "onion.yaml")
        (machine / "local.yaml").write_text('onion:\n  librarian:\n    model: "local:7b"\n', encoding="utf-8")
        for module in modules:
            module._CONFIG = machine / "onion.yaml"
        paths = [module._CONFIG for module in modules]
        journal = composer.load_journal()
        assert composer.load_budgets().home == 16384 and lib.load_config().model == "local:7b", (
            "control: the machine's file and its local file beside it")
        with ab.pinned_rung(None):
            assert [module._CONFIG for module in modules] == paths, "None pins nothing"
        with ab.pinned_rung(4096):
            pinned = composer.load_journal()
            copy = Path(composer._CONFIG)
            assert composer.load_budgets().home == 4096, "the loaders read home_rung 4096"
            assert pinned.tool_room == max(1, journal.tool_room * 4096 // 16384), (
                "the tool room scaled to keep its value at each rung", pinned.tool_room, journal.tool_room)
            assert pinned.tool_result_cap == max(1, journal.tool_result_cap * 4096 // 16384), (
                "the result cap scaled too", pinned.tool_result_cap)
            assert {module._CONFIG for module in modules} == {copy} and copy.parent != machine, (
                "the three readers read the one copy")
            assert lib.load_config().model == "local:7b", "the machine's local file read beside the copy"
        assert [module._CONFIG for module in modules] == paths and not copy.parent.exists(), (
            "after the pin, the file, its copy gone")
        with pytest.raises(RuntimeError):
            with ab.pinned_rung(8192):
                copy = Path(composer._CONFIG)
                raise RuntimeError("inside")
        assert [module._CONFIG for module in modules] == paths and not copy.parent.exists(), (
            "an error inside the pin leaves the file again")
        with pytest.raises(ab.RungRefused):
            with ab.pinned_rung(5000):
                pytest.fail("a rung the loaders refuse is never entered")
        assert [module._CONFIG for module in modules] == paths, "a refused rung points nothing elsewhere"
        with pytest.raises(ab.RungRefused, match="home_rung"):
            ab.pinned_text("journal:\n  margin: 0.15\n", 4096)
    finally:
        for module, path in zip(modules, shipped):
            module._CONFIG = path
        restore()


# ---------------------------------------------------------------------------
# DP4 -- two trees compared probe by probe
# ---------------------------------------------------------------------------
def test_dp4_two_trees_compared_probe_by_probe(tmp_path):
    """DP4: windows that differ by messages, rung or context; refusals; verdicts only when both were asked; a
    probe not the same in both refused."""
    pairs = _pairs()
    window = [{"role": "user", "content": "w"}]
    other = [{"role": "user", "content": "v"}]

    def row(turn, verdict="agrees", messages=window, num_ctx=4096, question=None, refusal=None, asked=True):
        out = {"turn": turn, "question": question or f"q{turn}", "rung": 4096, "refusal": refusal}
        if refusal is None:
            out.update(messages=messages, window_rung=4096, num_ctx=num_ctx)
        if asked and refusal is None:
            out["verdict"] = verdict
        return out

    def written(name, rows):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return path

    before = written("before", [row(16), row(26, "contradicts"), row(36), row(49), row(59)])
    after = written("after", [row(16), row(26, messages=other), row(36, "undecided", messages=other),
                              row(49, num_ctx=8192), row(69)])
    assert pairs.compare(before, after) == {
        "probes": [16, 26, 36, 49], "windows_differ": [26, 36, 49], "refused": {"before": {}, "after": {}},
        "asked": True, "before": 3, "after": 3, "gains": [26], "losses": [36]}, "probe by probe"
    refused = written("refused", [row(16), row(26, refusal="count")])
    read = pairs.compare(before, refused)
    assert read["windows_differ"] == [26] and read["refused"] == {"before": {}, "after": {26: "count"}}, read
    assert read["asked"] is False and read["gains"] is None and read["before"] is None, (
        "a probe with no verdict: no verdicts compared", read)
    windows_only = written("windows", [row(16, asked=False), row(26, asked=False)])
    read = pairs.compare(before, windows_only)
    assert read["asked"] is False and (read["before"], read["after"], read["losses"]) == (None, None, None), read
    other_question = written("other", [row(16, question="another")])
    with pytest.raises(pairs.Refused, match="probe 16"):
        pairs.compare(before, other_question)


# ---------------------------------------------------------------------------
# DP5 -- compose, end to end
# ---------------------------------------------------------------------------
def test_dp5_compose_end_to_end(tmp_path):
    """DP5: every kept probe composed under the home rung it was kept under, asked and read; some turns alone;
    refusals that write nothing."""
    ab, pairs = da._script(), _pairs()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        states = tmp_path / "states"
        _kept_run(ab, lib, loaded, states)
        lib.reset_librarian()
        backend = _Answering(ab)
        out = tmp_path / "composed.jsonl"
        assert pairs.compose(states, out, resolve=lambda name: backend) == 0
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert [r["turn"] for r in rows] == list(_KEPT), "every kept probe, in order"
        for r in rows:
            assert r["same_as_recorded"] is True and r["held"] is False, ("the window the run sent", r["turn"])
            assert r["verdict"] == "agrees" and r["model"] == "bench", ("asked and read", r["turn"], r.get("reply"))
            assert r["rung"] == 16384 and Path(r["tree"]) == _PAIRS.parent.parent, ("rung and tree", r["turn"])
        assert backend.models == ["bench"] * len(_KEPT), "the run's model, once per probe"
        some = tmp_path / "some.jsonl"
        assert pairs.compose(states, some, ask=False, turns=[26]) == 0
        only = [json.loads(line) for line in some.read_text(encoding="utf-8").splitlines()]
        assert [r["turn"] for r in only] == [26] and "verdict" not in only[0], "some turns, composed only"
        assert pairs.compare(out, some)["probes"] == [26], "the probes both composed"
        refused = {}
        refused["rung"] = pairs.compose(states, tmp_path / "rung.jsonl", rung=4096, ask=False)
        refused["model"] = pairs.compose(states, tmp_path / "model.jsonl", resolve=lambda name: None)
        meta = states / "t026.json"
        kept = json.loads(meta.read_text(encoding="utf-8"))
        meta.write_text(json.dumps(dict(kept, question="Quick check: something else?")), encoding="utf-8")
        refused["probe"] = pairs.compose(states, tmp_path / "probe.jsonl", ask=False)
        assert refused == {"rung": 2, "model": 2, "probe": 2}, refused
        assert not any(tmp_path.glob("rung.jsonl*")) and not any(tmp_path.glob("model.jsonl*")) and not any(
            tmp_path.glob("probe.jsonl*")), "a refusal writes nothing"
    finally:
        lib.reset_librarian()
        restore()


class _Reader:
    """A backend that reads, in each span, one statement of each typed turn that names Alice or Bob, and keeps what
    it was handed."""

    def __init__(self, prompt, fail=False):
        self._prompt, self._fail, self.read = prompt, fail, []

    def generate(self, model, messages, **kwargs):
        assert messages[0]["content"] == self._prompt, "the librarian's reading prompt"
        if self._fail:
            raise RuntimeError("the model went away")
        lines = [json.loads(line) for line in messages[1]["content"].split("\n")]
        known = lines[-1]["known"] if "known" in lines[-1] else []
        turns = [line for line in lines if "turn" in line]
        self.read.append((tuple(t["turn"] for t in turns), tuple(known)))
        found = [f"{t['turn']} | {name} | is named in | {name}'s" for t in turns if t["role"] == "user"
                 for name in ("Alice", "Bob") if f"{name}'s" in t["text"]]
        found += [f"{t['turn']} | {name} | lives in | {t['text'].split(' lives in ')[1].rstrip('.')}"
                  for t in turns if t["role"] == "user" for name in ("Alice", "Bob")
                  if t["text"].endswith(".") and f"{name} lives in " in t["text"]]
        return SimpleNamespace(content="\n".join(found) or "NONE")


# ---------------------------------------------------------------------------
# DP6 -- what the librarian reads, laid on the same states
# ---------------------------------------------------------------------------
def test_dp6_what_the_librarian_reads_laid_on_the_same_states(tmp_path, monkeypatch):
    """DP6: ``read`` reads each span once, oldest first, with the subjects known before it, and writes each span's
    statements by its key; ``compose --statements`` lays them before each plan and names the file; refusals that
    write nothing."""
    ab, pairs = da._script(), _pairs()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        states = tmp_path / "states"
        _kept_run(ab, lib, loaded, states)
        lib.reset_librarian()
        backend = _Reader(lib._STATEMENTS_PROMPT)
        out = tmp_path / "read.json"
        assert pairs.read(states, out, model="small:1b", resolve=lambda name: backend) == 0
        rows = json.loads(out.read_text(encoding="utf-8"))
        assert rows and len(backend.read) == len(rows), ("each span read once", len(backend.read), len(rows))
        firsts = [int(turns[0][1:]) for turns, _known in backend.read]
        assert firsts == sorted(firsts), "oldest first"
        said = {r[3] for found in rows.values() for r in found}
        assert {"Alice", "Bob"} <= said, ("control: both named", said)
        first_bob = next(n for n, (turns, _k) in enumerate(backend.read)
                         if any(r[3] == "Bob" for key in rows for r in rows[key] if r[0] in turns))
        assert all("Bob" not in known for _turns, known in backend.read[:first_bob + 1]) and all(
            "Bob" in known for _turns, known in backend.read[first_bob + 1:]), "the subjects known before each span"
        laid = []
        real = lib.plan_window

        def planned(conversation_id, **kwargs):
            state = lib.peek_state(conversation_id)
            laid.append({key: [list(dataclasses.astuple(s)) for s in found]
                         for key, found in state.statements.items()})
            return real(conversation_id, **kwargs)

        monkeypatch.setattr(lib, "plan_window", planned)
        section = lib._statements_section()
        assert not section.enabled, "control: off as shipped"
        monkeypatch.setattr(lib, "_statements_section", lambda: dataclasses.replace(section, enabled=True))
        composed = tmp_path / "composed.jsonl"
        assert pairs.compose(states, composed, ask=False, statements=out) == 0
        digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
        lines = [json.loads(line) for line in composed.read_text(encoding="utf-8").splitlines()]
        assert [r["turn"] for r in lines] == list(_KEPT) and {r["statements"] for r in lines} == {digest}, (
            "every probe composed, the file named")
        assert len(laid) == len(_KEPT) and all(found and all(rows[k] == v for k, v in found.items())
                                               for found in laid), "each state laid as read before its plan"
        assert len(laid[-1]) > len(laid[0]), "a later state holds more spans read"
        refused = {}
        refused["model"] = pairs.read(states, tmp_path / "model.json", model="small:1b", resolve=lambda name: None)
        refused["failed"] = pairs.read(states, tmp_path / "failed.json", model="small:1b",
                                       resolve=lambda name: _Reader(lib._STATEMENTS_PROMPT, fail=True))
        listed = tmp_path / "listed.json"
        listed.write_text("[]", encoding="utf-8")
        refused["listed"] = pairs.compose(states, tmp_path / "listed.jsonl", ask=False, statements=listed)
        short = tmp_path / "short.json"
        short.write_text(json.dumps({key: [["t1", 0]] for key in rows}), encoding="utf-8")
        refused["short"] = pairs.compose(states, tmp_path / "short.jsonl", ask=False, statements=short)
        partial = tmp_path / "partial.json"
        partial.write_text(json.dumps(dict(list(rows.items())[:1])), encoding="utf-8")
        refused["partial"] = pairs.compose(states, tmp_path / "partial.jsonl", ask=False, statements=partial)
        monkeypatch.setattr(lib, "_statements_section", lambda: dataclasses.replace(section, enabled=False))
        refused["off"] = pairs.compose(states, tmp_path / "off.jsonl", ask=False, statements=out)
        assert refused == {"model": 2, "failed": 2, "listed": 2, "short": 2, "partial": 2, "off": 2}, refused
        assert not any(tmp_path.glob("model.json*")) and not any(tmp_path.glob("failed.json*")) and not any(
            tmp_path.glob("listed.jsonl*")) and not any(tmp_path.glob("short.jsonl*")) and not any(
            tmp_path.glob("partial.jsonl*")) and not any(tmp_path.glob("off.jsonl*")), "a refusal writes nothing"
    finally:
        lib.reset_librarian()
        restore()


# ---------------------------------------------------------------------------
# DP7 -- a kept state says what it carried read
# ---------------------------------------------------------------------------
def test_dp7_a_kept_state_says_what_it_carried_read(tmp_path, monkeypatch):
    """DP7: each line says the spans a kept state carried read, their digest and whether this tree laid them; the
    file of ``--statements`` takes their place; ``read`` reads no span again for the reader that read it."""
    ab, pairs = da._script(), _pairs()
    loaded, restore = _window()
    lib, store = loaded["opti_oignon.memory.librarian"], loaded[_STORE]
    try:
        states = tmp_path / "states"
        _kept_run(ab, lib, loaded, states)
        lib.reset_librarian()
        out = tmp_path / "read.json"
        assert pairs.read(states, out, model="small:1b", resolve=lambda name: _Reader(lib._STATEMENTS_PROMPT)) == 0
        rows = json.loads(out.read_text(encoding="utf-8"))
        # The same states, each kept with what was read of its spans, by a reader of its own.
        carried = tmp_path / "carried"
        shutil.copytree(states, carried)
        for turn in _KEPT:
            meta = json.loads((carried / f"t{turn:03d}.json").read_text(encoding="utf-8"))
            kept = store.OnionStore(carried / f"t{turn:03d}.db", connect=sqlite3.connect, require_encryption=False)
            lib.reset_librarian()
            state = kept.load(meta["cid"], lib.OnionState())
            pairs.take_statements(lib, state, rows)
            state.readers = {key: "a" * 64 for key in state.statements}
            kept.save(meta["cid"], state)
        lib.reset_librarian()
        backend = _Reader(lib._STATEMENTS_PROMPT)
        again = tmp_path / "again.json"
        assert pairs.read(carried, again, model="small:1b", resolve=lambda name: backend) == 0
        assert json.loads(again.read_text(encoding="utf-8")) == rows and len(backend.read) == len(rows), (
            "each span read once: what a state carried laid aside whole", len(backend.read), len(rows))
        section = lib._statements_section()
        composed = {}
        for name, on in (("off", False), ("on", True)):
            monkeypatch.setattr(lib, "_statements_section", lambda on=on: dataclasses.replace(section, enabled=on))
            path = tmp_path / f"{name}.jsonl"
            assert pairs.compose(carried, path, ask=False) == 0
            composed[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        spans = [r["readings"]["spans"] for r in composed["off"]]
        assert spans == sorted(spans) and spans[0] > 0 and all(
            len(r["readings"]["digest"]) == 16 for r in composed["off"]), ("each state says what it carried", spans)
        assert {r["readings"]["laid"] for r in composed["off"]} == {False}, "laid only when this tree reads"
        assert [r["readings"] for r in composed["on"]] == [dict(r["readings"], laid=True) for r in composed["off"]]
        laid = tmp_path / "laid.jsonl"
        assert pairs.compose(carried, laid, ask=False, statements=out) == 0
        lines = [json.loads(line) for line in laid.read_text(encoding="utf-8").splitlines()]
        assert [r["turn"] for r in lines] == list(_KEPT) and all(
            "readings" not in r and r["statements"] == hashlib.sha256(out.read_bytes()).hexdigest()[:16]
            for r in lines), "the file laid in their place, said alone"
    finally:
        lib.reset_librarian()
        restore()
