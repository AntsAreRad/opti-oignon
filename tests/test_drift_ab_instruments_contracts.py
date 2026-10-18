#!/usr/bin/env python3
"""Contracts for what the drift A/B gives to read its runs: the gate's figures and, on request, the answers.

Measured on the maintainer's machine, two things in a report could not be
read from it: the librarian's summaries were refused for length nearly every
time, and on the GPU half the answers' sentences were left undecided. The
counts said so; nothing said why. The host command now keeps what a reader
needs, and nothing of anyone's:

  * DI1 -- THE ANSWERS, ON REQUEST: with ``--answers PATH``, once the run has
    measured, one JSON line per arm and turn: the turn's question, the arm's
    answer and each of its sentences with its verdict, which add up to the
    report's sentences, decided, undecided, agreements and contradictions,
    arm by arm. A run that breaks writes nothing there and leaves an earlier
    file as it was; a write that fails on its way leaves it too, with
    nothing beside it.
  * DI2 -- THE GATE'S FIGURES: the report's ``gate`` holds one figure per
    decision the gate took in the run -- the span's turns and words, the
    summary's, the words the peel shows, its length ratio, novelty and
    coverage, accepted or not, the motives -- and their spread, with how many
    could not be read; numbers and motive names, never a word of the
    conversation. So in the fast mode and in the launcher's, the real run's,
    whose bursts run on threads of their own.
  * DI3 -- PUT BACK AS IT WAS: the gate's decision is the package's own again
    once the command returns, whether the run measured or broke.
  * DI4 -- THE PROBES ON THEIR OWN: each arm scores the turns that ask for a
    fact apart -- asked, answered with the fact expected, contradicted,
    decided, undecided, and the share recalled -- and the report gives the
    two arms' difference of recall; the undecided answers to the other
    turns weigh nothing there. An arm that answers every probe right recalls
    all; one that answers with the facts the conversation retired
    contradicts exactly those. An answer that gives the retired value beside
    the current one recalls nothing; one that recalls the fact and errs about
    another subject still recalls it; an empty one is undecided, and one
    about another fact is decided.

The script is loaded as the drift A/B suite loads it; the onion's modules
come through the shared isolation window; the model is a stand-in that
answers one fact and one word, on a conversation shortened to its first
turns where the whole is not needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_drift_ab_contracts as da  # noqa: E402

BUDGET_S = {
    "test_di1_the_answers_go_to_the_file_asked_for_and_add_up_to_the_report": 8.0,
    "test_di2_the_report_holds_the_gates_figures_for_each_decision_and_never_a_word": 15.0,
    "test_di3_the_gates_decision_is_put_back_whether_the_run_measured_or_broke": 8.0,
    "test_di4_the_probes_are_scored_on_their_own_and_the_arms_recall_compared": 15.0,
}

_PEELS = "opti_oignon.memory.peels"


class _Answering(da._Backend):
    """A backend that answers every turn with one fact and one word, and every summary with one sentence."""

    def generate(self, model, messages, **kwargs):
        self.calls += 1
        if self._fail_after is not None and self.calls > self._fail_after:
            raise RuntimeError(f"model {model!r} not found (status code: 404)")
        return type("_Reply", (), {"content": "Alice lives in Paris. Noted."})()


class _Oracle(da._Backend):
    """A backend that answers each probe with the fact it expects -- or, told ``stale``, with the one that fact
    retired -- and every other turn and every summary with one word."""

    def __init__(self, served, turns, *, stale=False):
        super().__init__(served)
        retired = {turn.fact: turn.replaces for turn in turns if turn.replaces}
        self._answers = {turn.text: (retired.get(turn.expects, turn.expects) if stale else turn.expects) + "."
                         for turn in turns if turn.probe}

    def generate(self, model, messages, **kwargs):
        self.calls += 1
        last = str(messages[-1].get("content", "")) if messages else ""
        answer = next((said for text, said in self._answers.items() if last.endswith(text)), "Noted.")
        return type("_Reply", (), {"content": answer})()


def _shortened(ab, turns):
    ab.TURNS = ab.TURNS[:turns]
    return ab


def _run(ab, argv, backend, capsys):
    code = ab.main(argv, resolve=lambda model: backend)
    out = capsys.readouterr()
    return code, out


# ---------------------------------------------------------------------------
# DI1 -- the answers, on request
# ---------------------------------------------------------------------------
def test_di1_the_answers_go_to_the_file_asked_for_and_add_up_to_the_report(tmp_path, capsys):
    loaded, restore = da._window()
    try:
        lib = loaded["opti_oignon.memory.librarian"]
        served = ["llama3:latest", lib.load_config().model]
        ab = _shortened(da._script(), 20)
        path = tmp_path / "answers.jsonl"
        code, out = _run(ab, ["--model", "llama3", "--mode", "fast", "--answers", str(path)], _Answering(served),
                         capsys)
        assert code == 0, out.err
        report = json.loads(out.out)
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [(line["arm"], line["turn"]) for line in lines] == (
            [("plain", i) for i in range(1, 21)] + [("onion", i) for i in range(1, 21)]), "one line per arm and turn"
        assert any(turn.probe for turn in ab.TURNS), "control: the shortened conversation holds a probe"
        assert all(line["question"] == turn.text for line, turn in zip(lines, ab.TURNS * 2)), "the turn's question"
        assert all(line["answer"] == "Alice lives in Paris. Noted." for line in lines), "the arm's answer"
        assert all(line["expects"] == (turn.expects if turn.probe else None) for line, turn in zip(lines, ab.TURNS * 2)), (
            "a probe's line names the fact it expects")
        for arm in ("plain", "onion"):
            said = [s for line in lines if line["arm"] == arm for s in line["sentences"]]
            verdicts = [s["verdict"] for s in said]
            reading = report[arm]
            assert len(said) == reading["sentences"], (arm, len(said), reading["sentences"])
            assert verdicts.count("undecided") == reading["undecided"], (arm, verdicts)
            assert len(said) - verdicts.count("undecided") == reading["decided"], arm
            assert verdicts.count("contradicts") == len({(a["turn"], a["sentence"]) for a in reading["at"]}), arm
            assert verdicts.count("agrees") <= reading["agreements"], arm
        assert set(verdicts) <= {"contradicts", "agrees", "decided", "undecided"}, verdicts
        assert "undecided" in verdicts and ({"contradicts", "agrees", "decided"} & set(verdicts)), (
            "control: both kinds were read", set(verdicts))

        before = path.read_text(encoding="utf-8")
        code, out = _run(_shortened(da._script(), 12), ["--model", "llama3", "--answers", str(path)],
                         _Answering(served, fail_after=2), capsys)
        assert code == 2 and out.out == "", out.err
        assert path.read_text(encoding="utf-8") == before, "a run that breaks leaves an earlier file as it was"
        broken = {"plain": (["fine", "no line of this"], [[], [{"text": "x", "verdict": {"a set"}}]])}
        try:
            ab.write_answers(path, ab.TURNS, broken)
        except TypeError:
            pass
        else:
            raise AssertionError("control: the second line cannot be written")
        assert path.read_text(encoding="utf-8") == before, "a write that fails on its way leaves the file as it was"
        assert not path.with_name(path.name + ".partial").exists(), "and leaves nothing beside it"
        lib.reset_librarian()
    finally:
        restore()


# ---------------------------------------------------------------------------
# DI2 -- the gate's figures
# ---------------------------------------------------------------------------
def test_di2_the_report_holds_the_gates_figures_for_each_decision_and_never_a_word(capsys):
    loaded, restore = da._window()
    try:
        lib, peels = loaded["opti_oignon.memory.librarian"], loaded[_PEELS]
        served = ["llama3:latest", lib.load_config().model]
        decided, decide = [], peels.decide

        def counting(span, probes, text, gate, refs=()):
            decision = decide(span, probes, text, gate, refs)
            decided.append((len(span), decision.length_ratio, bool(decision.accepted),
                            tuple(peels.refusal_motives(decision, gate))))
            return decision

        peels.decide = counting
        try:
            code, out = _run(da._script(), ["--model", "llama3"], _Answering(served), capsys)
            assert code == 0, out.err
            launched, seen = json.loads(out.out)["gate"], len(decided)
            lib.reset_librarian()
            decided.clear()
            code, out = _run(da._script(), ["--model", "llama3", "--mode", "fast"], _Answering(served), capsys)
        finally:
            peels.decide = decide
        assert seen and launched["decisions"] == seen and launched["unread"] == 0, (
            "the launcher's bursts, on threads of their own, are all read", launched["decisions"], seen)
        assert code == 0, out.err
        report = json.loads(out.out)
        gate = report["gate"]
        assert decided, "control: the librarian handed the gate summaries on the bench's own run"
        assert gate["decisions"] == len(decided) == len(gate["each"]) and gate["unread"] == 0, (
            gate["decisions"], len(decided))
        each = gate["each"]
        assert [(f["span_turns"], f["accepted"], tuple(f["motives"])) for f in each] == [
            (turns, accepted, motives) for turns, _ratio, accepted, motives in decided], "each decision, in order"
        assert all(f["length_ratio"] is None and ratio is None or abs(f["length_ratio"] - ratio) < 1e-3
                   for f, (_t, ratio, _a, _m) in zip(each, decided)), "the ratio the gate held it to"
        assert all(f["span_words"] > 0 and f["shown_words"] >= f["summary_words"] >= 0 for f in each), each[:3]
        assert all(f["length_ratio"] is None or abs(f["length_ratio"] - f["shown_words"] / f["span_words"]) < 1e-3
                   for f in each), "the ratio is the words shown over the span's"
        motives = {}
        for _turns, _ratio, _accepted, found in decided:
            for motive in found:
                motives[motive] = motives.get(motive, 0) + 1
        assert gate["motives"] == motives and gate["accepted"] == sum(1 for d in decided if d[2]), gate["motives"]
        ratios = sorted(r for _t, r, _a, _m in decided if r is not None)
        assert gate["length_ratio"]["n"] == len(ratios) and gate["length_ratio"]["max"] == round(ratios[-1], 3)
        assert gate["max_length_ratio"] == peels.load_gate().max_length_ratio
        assert gate["over_length"] == sum(1 for r in ratios if r > gate["max_length_ratio"])

        keys = {"decisions", "unread", "accepted", "motives", "max_length_ratio", "over_length", "length_ratio",
                "span_words", "summary_words", "reference_words", "each", "n", "min", "median", "p90", "max", "mean",
                "span_turns", "shown_words", "novelty", "probe_coverage"}
        strings = []

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    strings.append(key)
                    assert key in keys or key in peels.REFUSAL_MOTIVES, ("a key of the figures' own", key)
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
            elif isinstance(node, str):
                strings.append(node)
                assert node in peels.REFUSAL_MOTIVES, ("the only words are motive names", node)
            else:
                assert node is None or isinstance(node, (bool, int, float)), node

        walk(gate)
        assert "Alice" not in json.dumps(gate) and "Noted" not in json.dumps(gate), "never a word of the run"

        # A decision whose figures cannot be read is still the gate's, and is counted, never silently lost.
        ab = da._script()
        verdict = object()
        stub = type("_Peels", (), {"decide": staticmethod(lambda *args: verdict)})
        failing = type("_Probes", (), {"word_count": staticmethod(lambda text: int("not a count"))})
        figures, unread = [], []
        recorded = ab.recording_gate(stub, failing, figures, unread)
        assert recorded([{"text": "x"}], (), "y", None) is verdict, "the decision stands"
        assert (figures, unread) == ([], ["ValueError"]), (figures, unread)
        assert ab.gate_report(figures, 1.0, unread)["unread"] == 1
        lib.reset_librarian()
    finally:
        restore()


# ---------------------------------------------------------------------------
# DI3 -- put back as it was
# ---------------------------------------------------------------------------
def test_di3_the_gates_decision_is_put_back_whether_the_run_measured_or_broke(capsys):
    loaded, restore = da._window()
    try:
        lib, peels = loaded["opti_oignon.memory.librarian"], loaded[_PEELS]
        served = ["llama3:latest", lib.load_config().model]
        original = peels.decide
        code, out = _run(_shortened(da._script(), 12), ["--model", "llama3", "--mode", "fast"], _Answering(served),
                         capsys)
        assert code == 0 and peels.decide is original, ("put back once the run measured", out.err)
        code, out = _run(_shortened(da._script(), 12), ["--model", "llama3", "--mode", "fast"],
                         _Answering(served, fail_after=6), capsys)
        assert code == 2 and peels.decide is original, ("put back once the run broke", out.err)
        lib.reset_librarian()
    finally:
        restore()


# ---------------------------------------------------------------------------
# DI4 -- the probes on their own
# ---------------------------------------------------------------------------
def test_di4_the_probes_are_scored_on_their_own_and_the_arms_recall_compared(capsys):
    loaded, restore = da._window()
    try:
        lib = loaded["opti_oignon.memory.librarian"]
        served = ["llama3:latest", lib.load_config().model]
        ab = da._script()
        probes = [turn for turn in ab.TURNS if turn.probe]
        retired = {turn.fact for turn in ab.TURNS if turn.replaces}
        moved = sum(1 for turn in probes if turn.expects in retired)
        assert len(probes) == 7 and moved == 3, "control: seven probes, three about a fact the conversation retired"
        assert all(any(statement == turn.expects for _id, statement in ab.held_facts(ab.TURNS, index + 1))
                   for index, turn in enumerate(ab.TURNS) if turn.probe), (
            "control: each probe expects a fact held at its turn, word for word, or a hedge would go unseen")

        code, out = _run(ab, ["--model", "llama3", "--mode", "fast"], _Oracle(served, ab.TURNS), capsys)
        assert code == 0, out.err
        report = json.loads(out.out)
        for arm in ("plain", "onion"):
            assert report[arm]["probes"] == {"asked": 7, "agrees": 7, "contradicts": 0, "decided": 0, "undecided": 0,
                                             "recall": 1.0}, (arm, report[arm]["probes"])
            assert report[arm]["undecided"] >= len(ab.TURNS) - len(probes), "the other turns' word weighs nothing"
        assert report["delta_probe_recall"] == 0.0
        lib.reset_librarian()

        code, out = _run(da._script(), ["--model", "llama3", "--mode", "fast"], _Oracle(served, ab.TURNS, stale=True),
                         capsys)
        assert code == 0, out.err
        report = json.loads(out.out)
        for arm in ("plain", "onion"):
            scored = report[arm]["probes"]
            assert (scored["asked"], scored["agrees"], scored["contradicts"]) == (7, 7 - moved, moved), (arm, scored)
            assert sum(scored[k] for k in ("agrees", "contradicts", "decided", "undecided")) == scored["asked"]
            assert scored["recall"] == round((7 - moved) / 7, 3), scored
        moved_turns = (
            ab.Turn("Alice lives in Berlin.", fact="Alice lives in Berlin"),
            ab.Turn("Bob lives in Lyon.", fact="Bob lives in Lyon"),
            ab.Turn("News: Alice lives in Oslo.", fact="Alice lives in Oslo", replaces="Alice lives in Berlin"),
            ab.Turn("Quick check: where does Alice live?", probe=True, expects="Alice lives in Oslo"),
        )
        tallies = {}
        for case, said in (
            ("right", "Alice lives in Oslo."),
            ("hedged", "Alice lives in Oslo. Alice lives in Berlin."),
            ("right, wrong elsewhere", "Alice lives in Oslo. Bob lives in Paris."),
            ("retired", "Alice lives in Berlin."),
            ("about another fact", "Bob lives in Lyon."),
            ("empty", ""),
        ):
            probes = ab.reading(moved_turns, ["Noted."] * 3 + [said], arm="unit", source="unit")["probes"]
            tallies[case] = next(k for k in ("agrees", "contradicts", "decided", "undecided") if probes[k])
        assert tallies == {"right": "agrees", "hedged": "contradicts", "right, wrong elsewhere": "agrees",
                           "retired": "contradicts", "about another fact": "decided", "empty": "undecided"}, tallies
        compared = ab.compare({"rate_per_1000_turns": 10.0, "probes": {"recall": 0.75}},
                              {"rate_per_1000_turns": 30.0, "probes": {"recall": 0.25}})
        assert (compared["delta_probe_recall"], compared["delta_rate_per_1000_turns"]) == (0.5, -20.0), (
            "onion minus plain, both", compared)
        lib.reset_librarian()
    finally:
        restore()
