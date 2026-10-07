#!/usr/bin/env python3
"""Contracts for the drift A/B the host runs: the onion against no onion.

The measure -- contradictions per 1000 turns, with the onion and without
-- needs a real model and is owed to the machine. What is held here is
everything around the model, so that the host run is one command whose
number means what it says.

  * DA1 -- the conversation is built to drift: it states facts in the
    templates' grammar, replaces some on known turns, asks about each one
    long after the history window has dropped it, and runs longer than the
    onion's window.
  * DA2 -- the reading judges each answer against the facts holding at its
    turn: a faithful answer agrees, a replaced fact restated is a
    contradiction, a sentence the templates cannot parse is undecided and
    never agreement, and no answer at all is no rate.
  * DA3 -- the two arms differ by the onion's block and by nothing else:
    over the same turns and the same model they send the same history
    window; the onion arm's system prompt carries the curated block,
    wrapped as untrusted data and holding Peels, and the plain arm's never
    does; the onion arm refuses a librarian that would write a store.
  * DA4 -- without a backend the host command prints no number and exits
    non-zero.
  * DA5 -- a model the backend lists as not served -- the answering one or
    the librarian's -- is refused by name before any request, with the
    served models named; a name without a tag matches its latest tag, and
    a backend that cannot list its models is not refused on that ground.
  * DA6 -- a request that fails once the run has started measures nothing:
    no number is printed and the command exits non-zero.
  * DA7 -- the pair is tried in the run's order before the first turn: a
    model that cannot answer beside the other -- the governor refusing the
    load, say -- is reported by name with its reason, and no turn is asked.
  * DA9 -- the onion arm hands the librarian each turn with the origin the
    executor gives it -- the user's words typed, the answer the
    assistant's -- so the measurement runs the queue production runs.
  * DA10 -- after a failed call the onion arm's curation goes on without
    the model, as a burst does: one failed call per turn, then none.

The rhythm of the chat path: the librarian's launcher fires a burst once
the conversation grew by ``min_new_turns``, on a thread of its own, while
the next turns are answered.

  * DA11 -- in the launcher's mode a burst fires only after the configured
    growth, as the chat path's launcher fires it, and runs off the
    answering thread.
  * DA12 -- the fast mode is the one the bench had: it curates after every
    turn, on the answering thread, before the next turn is asked.
  * DA13 -- both modes ask for the second summary, as the chat path does.
  * DA14 -- the report names its mode and carries the engagement counts --
    bursts, evictions by rung, refusals by motive, the block's tokens per
    turn -- with the generator's version and the lexicon's fingerprint.
  * DA15 -- an arm ends only once the bursts it launched have ended.
  * DA16 -- the onion arm refuses a counters path: the bench never adds
    its counts to the user's.
  * DA17 -- with the registry's summariser, each measured turn's curation
    in the fast mode is a run of its own, whose budget is the turn's.

Local-only (the public distribution ships no tests). The script is loaded
from its path; the onion's modules come through the shared isolation
window, and the model is a recording seam.
"""

import collections
import importlib.util
import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_SCRIPT = REPO / "scripts" / "drift_ab.py"
_ONION = ("probes", "drift", "core_store", "receipts", "composer", "peels", "librarian")


def _script():
    spec = importlib.util.spec_from_file_location("drift_ab_under_contract", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _window():
    targets = {f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _ONION}
    targets["opti_oignon.agent.untrusted_context"] = source("agent", "untrusted_context.py")
    return isolate(targets=targets, packages=("opti_oignon.memory", "opti_oignon.agent"))


def _faithful(turns):
    return " ".join(t["text"] for t in turns)


# ---------------------------------------------------------------------------
# DA1 -- a conversation built to drift
# ---------------------------------------------------------------------------
def test_da1_the_conversation_is_built_to_drift():
    ab = _script()
    loaded, restore = _window()
    try:
        drift = loaded["opti_oignon.memory.drift"]
        turns = ab.TURNS
        facts = [t for t in turns if t.fact]
        replaced = [t for t in turns if t.replaces]
        probes = [i for i, t in enumerate(turns) if t.probe]
        assert len(facts) >= 6 and len(replaced) >= 2 and len(probes) >= 6
        for t in facts + replaced:
            for statement in (t.fact, t.replaces):
                assert not statement or drift._parse(statement) is not None, f"{statement!r} is in the templates' grammar"
        for i in probes:
            expected = turns[i].expects
            held = [statement for _id, statement in ab.held_facts(turns, i + 1)]
            assert expected in held, f"probe {i} asks for a fact that holds then: {expected!r}"
            stated = max(j for j in range(i) if turns[j].fact == expected)
            between = sum(ab.estimate_tokens(turns[j].text) for j in range(stated + 1, i))
            assert between > ab.HISTORY_TOKENS, f"probe {i}: the history window has dropped {expected!r} by then"
        total = sum(ab.estimate_tokens(t.text) for t in turns)
        window = loaded["opti_oignon.memory.composer"].load_budget().window
        assert total > window, f"the conversation ({total} tokens) is longer than the onion's own window ({window})"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DA2 -- the reading
# ---------------------------------------------------------------------------
def test_da2_each_answer_is_judged_against_the_facts_holding_at_its_turn():
    ab = _script()
    Turn = ab.Turn
    turns = (
        Turn("Alice lives in Berlin.", fact="Alice lives in Berlin"),
        Turn("Tell me about the weather."),
        Turn("Alice moved. Alice lives in Oslo.", fact="Alice lives in Oslo", replaces="Alice lives in Berlin"),
        Turn("Where does Alice live?", probe=True, expects="Alice lives in Oslo"),
    )
    faithful = ab.reading(turns, ["Alice lives in Berlin.", "Sunny.", "Alice lives in Oslo.", "Alice lives in Oslo."], arm="onion", source="fixture")
    assert (faithful["contradictions"], faithful["agreements"], faithful["undecided"]) == (0, 3, 1), faithful
    assert faithful["rate_per_1000_turns"] == 0.0 and faithful["source"] == "fixture" and faithful["arm"] == "onion"
    drifted = ab.reading(turns, ["Alice lives in Berlin.", "Sunny.", "Alice lives in Oslo.", "Alice lives in Berlin."], arm="plain", source="fixture")
    assert drifted["contradictions"] == 1 and drifted["rate_per_1000_turns"] == 250.0, drifted
    assert [at["turn"] for at in drifted["at"]] == [4], "the replaced fact restated after its replacement"
    silent = ab.reading(turns, ["I cannot say.", "Sunny.", "I do not know.", "No idea."], arm="plain", source="fixture")
    assert (silent["contradictions"], silent["agreements"], silent["decided"]) == (0, 0, 0), "undecided is not agreement"
    assert silent["undecided"] == 4
    assert ab.reading(turns, [], arm="plain", source="fixture")["rate_per_1000_turns"] is None, "no answer, no rate"
    with pytest.raises(ValueError):
        ab.reading(turns, ["one"] * 5, arm="plain", source="fixture")


# ---------------------------------------------------------------------------
# DA3 -- the arms differ by the onion's block alone
# ---------------------------------------------------------------------------
def test_da3_the_arms_differ_by_the_onion_block_and_nothing_else(tmp_path):
    ab = _script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        peels = loaded["opti_oignon.memory.peels"]
        composer = loaded["opti_oignon.memory.composer"]
        uc = loaded["opti_oignon.agent.untrusted_context"]
        config = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=1, temperature=0.1,
                                     num_predict=64, persist_path="", require_encryption=False)
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)

        def recorder(calls):
            def ask(messages):
                calls.append(messages)
                return "Noted."
            return ask

        def wrap(block):
            return uc.wrap(block, source=uc.SOURCE_MEMORY)

        plain_calls, onion_calls = [], []
        turns = ab.TURNS[:40]
        ab.run_arm(turns, ab.plain_arm(recorder(plain_calls), history_tokens=300))
        onion = ab.onion_arm(recorder(onion_calls), librarian=lib, config=config, summarize=_faithful, wrap=wrap,
                             history_tokens=300, gate=gate, budget=budget)
        ab.run_arm(turns, onion)
        assert len(plain_calls) == len(onion_calls) == len(turns)
        for plain, with_onion in zip(plain_calls, onion_calls):
            assert plain[1:] == with_onion[1:], "the same history window and the same turn"
            assert plain[0] == {"role": "system", "content": ab.SYSTEM_PROMPT}, "the plain arm never carries a block"
            assert with_onion[0]["role"] == "system" and with_onion[0]["content"].startswith(ab.SYSTEM_PROMPT)
        blocks = [c[0]["content"] for c in onion_calls if c[0]["content"] != ab.SYSTEM_PROMPT]
        policy = wrap("x").splitlines()[0]
        assert blocks and all(policy in b for b in blocks), "the onion's block reached the model, wrapped as untrusted data"
        assert any("layer=peels" in b for b in blocks), "curation ran: a Peel reached the model"
        assert len(plain_calls[-1]) < 2 + 2 * 39, "the history window dropped the oldest turns"
        with pytest.raises(ValueError):
            ab.onion_arm(recorder([]), librarian=lib, config=replace(config, persist_path=str(tmp_path / "onion.db")),
                         summarize=_faithful, wrap=wrap)
    finally:
        lib.reset_librarian()
        restore()


# ---------------------------------------------------------------------------
# DA4 -- no backend, no number
# ---------------------------------------------------------------------------
def test_da4_without_a_backend_the_host_command_prints_no_number(capsys):
    ab = _script()
    code = ab.main(["--model", "absent:0b"], resolve=lambda model: None)
    out = capsys.readouterr()
    assert code == 2, "no backend, no run"
    assert out.out == "", "nothing that looks like a result"
    assert "absent:0b" in out.err and "nothing measured" in out.err



# ---------------------------------------------------------------------------
# DA5 -- a model that is not served is refused before any request
# ---------------------------------------------------------------------------
class _Listed:
    def __init__(self, name):
        self.name = name


class _Backend:
    """A backend that lists fixed models and answers, or fails, as told."""

    def __init__(self, served, *, fail_after=None):
        self._served = served
        self._fail_after = fail_after
        self.calls = 0

    def list_models(self):
        return None if self._served is None else [_Listed(name) for name in self._served]

    def generate(self, model, messages, **kwargs):
        self.calls += 1
        if self._fail_after is not None and self.calls > self._fail_after:
            raise RuntimeError(f"model {model!r} not found (status code: 404)")
        return type("_Reply", (), {"content": "Noted."})()


def test_da5_a_model_that_is_not_served_is_refused_by_name_before_any_request(capsys):
    loaded, restore = _window()
    try:
        ab = _script()
        librarian_model = loaded["opti_oignon.memory.librarian"].load_config().model
        backend = _Backend(["llama3:latest", librarian_model], fail_after=0)
        code = ab.main(["--model", "qwen3:32b"], resolve=lambda model: backend)
        out = capsys.readouterr()
        assert code == 2 and out.out == "", "refused, and nothing that looks like a result"
        assert "qwen3:32b" in out.err and "llama3:latest" in out.err and "--model" in out.err, out.err
        assert "nothing measured" in out.err and backend.calls == 0, (out.err, backend.calls)
        unserved = _Backend(["llama3:latest"], fail_after=0)
        code = ab.main(["--model", "llama3"], resolve=lambda model: unserved)
        out = capsys.readouterr()
        assert code == 2 and out.out == "", "the librarian's model is checked too"
        assert repr(librarian_model) in out.err and "--librarian-model" in out.err and unserved.calls == 0, out.err
        blind = _Backend(None, fail_after=0)
        code = ab.main(["--model", "llama3"], resolve=lambda model: blind)
        out = capsys.readouterr()
        assert blind.calls >= 1, "a backend that cannot list its models is asked, not refused"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DA6 -- a run that breaks measures nothing
# ---------------------------------------------------------------------------
def test_da6_a_request_that_fails_once_the_run_has_started_measures_nothing(capsys):
    loaded, restore = _window()
    try:
        ab = _script()
        librarian_model = loaded["opti_oignon.memory.librarian"].load_config().model
        backend = _Backend(["llama3:latest", librarian_model], fail_after=2)
        code = ab.main(["--model", "llama3"], resolve=lambda model: backend)
        out = capsys.readouterr()
        assert backend.calls == 3, "the run started, and broke on its third request"
        assert code == 2 and out.out == "", "a broken run prints no number"
        assert "nothing measured" in out.err and "404" in out.err, out.err
    finally:
        restore()


# ---------------------------------------------------------------------------
# DA7 -- the pair is tried before the first turn
# ---------------------------------------------------------------------------
class _Refusing(_Backend):
    """A backend whose governor refuses to load one model, counting requests by model."""

    def __init__(self, served, refuse):
        super().__init__(served)
        self._refuse = refuse
        self.by_model = collections.Counter()

    def generate(self, model, messages, **kwargs):
        self.by_model[model] += 1
        if model == self._refuse:
            raise RuntimeError(f"Not enough resources to load {model} (short by 1.4 GB)")
        return super().generate(model, messages, **kwargs)


def test_da7_the_pair_is_tried_before_the_first_turn_and_a_refusal_asks_no_turn(capsys):
    loaded, restore = _window()
    try:
        ab = _script()
        backend = _Refusing(["llama3:latest", "small:1b"], refuse="small:1b")
        code = ab.main(["--model", "llama3", "--librarian-model", "small:1b"], resolve=lambda model: backend)
        out = capsys.readouterr()
        assert code == 2 and out.out == "", "refused, and nothing that looks like a result"
        assert "small:1b" in out.err and "short by 1.4 GB" in out.err and "nothing measured" in out.err, out.err
        assert backend.by_model == {"llama3": 1, "small:1b": 1}, f"one trial each, and no turn: {dict(backend.by_model)}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DA8 -- what DA3 pinned, with the block in front of the turn, as on the chat
# path: the same system prompt, the same window, and the turn carrying the
# onion's block, its frames kept, before the question.
# ---------------------------------------------------------------------------
def test_da8_the_arms_differ_by_the_onion_block_in_front_of_the_turn_and_nothing_else(tmp_path):
    ab = _script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        peels = loaded["opti_oignon.memory.peels"]
        composer = loaded["opti_oignon.memory.composer"]
        uc = loaded["opti_oignon.agent.untrusted_context"]
        config = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=1, temperature=0.1,
                                     num_predict=64, persist_path="", require_encryption=False)
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)

        def recorder(calls):
            def ask(messages):
                calls.append(messages)
                return "Noted."
            return ask

        def wrap(block):
            return uc.wrap(block, source=uc.SOURCE_MEMORY, frames=True)

        plain_calls, onion_calls = [], []
        turns = ab.TURNS[:40]
        ab.run_arm(turns, ab.plain_arm(recorder(plain_calls), history_tokens=300))
        onion = ab.onion_arm(recorder(onion_calls), librarian=lib, config=config, summarize=_faithful, wrap=wrap,
                             history_tokens=300, gate=gate, budget=budget)
        ab.run_arm(turns, onion)
        assert len(plain_calls) == len(onion_calls) == len(turns)
        for plain, with_onion in zip(plain_calls, onion_calls):
            assert plain[:-1] == with_onion[:-1], "the same system prompt and the same history window"
            assert plain[0] == {"role": "system", "content": ab.SYSTEM_PROMPT}, "no block in the system message"
            assert plain[-1]["role"] == with_onion[-1]["role"] == "user"
            assert with_onion[-1]["content"].endswith(plain[-1]["content"]), "the question closes the turn"
        blocks = [
            o[-1]["content"][: -len(p[-1]["content"])]
            for p, o in zip(plain_calls, onion_calls) if o[-1]["content"] != p[-1]["content"]
        ]
        policy = wrap("x").splitlines()[0]
        assert blocks and all(policy in b for b in blocks), "the onion's block reached the model, wrapped as untrusted data"
        assert any("layer=peels" in b for b in blocks), "curation ran: a Peel reached the model"
        assert len(plain_calls[-1]) < 2 + 2 * 39, "the history window dropped the oldest turns"
        with pytest.raises(ValueError):
            ab.onion_arm(recorder([]), librarian=lib, config=replace(config, persist_path=str(tmp_path / "onion.db")),
                         summarize=_faithful, wrap=wrap)
    finally:
        lib.reset_librarian()
        restore()


def test_da9_the_onion_arm_hands_each_turn_with_the_origin_the_executor_gives_it():
    ab = _script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        peels, composer = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.composer"]
        config = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=1, temperature=0.1,
                                     num_predict=64, persist_path="", require_encryption=False)
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        roomy = composer.Budget(window=4000, reserve=200, core=300, receipts=300, peels=800, flesh=2000, turn=200)
        onion = ab.onion_arm(lambda messages: "Noted.", librarian=lib, config=config, summarize=_faithful,
                             wrap=lambda block: block, gate=gate, budget=roomy)
        ab.run_arm(ab.TURNS[:3], onion)
        turns = lib.state_for("drift-ab", config).flesh.turns()
        assert len(turns) == 6, "control: every turn is still in the Flesh"
        assert [t.get("origin") for t in turns] == ["typed", "assistant"] * 3, "typed, then the assistant's"
    finally:
        lib.reset_librarian()
        restore()


def test_da10_after_a_failed_call_the_onion_arm_goes_on_without_the_model():
    ab = _script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        peels, composer = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.composer"]
        config = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=1, temperature=0.1,
                                     num_predict=64, persist_path="", require_encryption=False)
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=1)
        tight = composer.Budget(window=4000, reserve=200, core=300, receipts=300, peels=800, flesh=1, turn=200)
        calls = []

        def down(turns):
            calls.append(len(turns))
            raise TimeoutError("no answer")

        onion = ab.onion_arm(lambda messages: "Noted: Bob moved the build to Berlin on 2026-03-04.", librarian=lib,
                             config=config, summarize=down, wrap=lambda block: block, gate=gate, budget=tight)
        ab.run_arm(ab.TURNS[:6], onion)
        steps = sum(lib.counters().get("eviction", {}).values())
        assert steps > 6, "control: turns ran more than one step"
        assert len(calls) == 6, "one failed call a turn, then the steps go on without the model"
    finally:
        lib.reset_librarian()
        restore()


# ---------------------------------------------------------------------------
# DA11-DA17 -- the chat path's rhythm, the second asking, the report
# ---------------------------------------------------------------------------
_LOSSY = "Something happened in this conversation."


def _rhythm(lib, loaded, *, min_new_turns=1, flesh=1, span_turns=2):
    peels, composer = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.composer"]
    config = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="5m", min_new_turns=min_new_turns,
                                 temperature=0.0, num_predict=64, persist_path="", require_encryption=False)
    gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=span_turns)
    budget = composer.Budget(window=4000, reserve=200, core=300, receipts=300, peels=800, flesh=flesh, turn=200)
    return config, gate, budget


def _bursts_on(lib):
    """The threads the bursts ran on, by wrapping the librarian's burst."""
    threads, real = [], lib._curation_burst

    def burst(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    lib._curation_burst = burst
    return threads


def test_da11_in_the_launcher_s_mode_a_burst_fires_after_the_configured_growth_off_the_answering_thread():
    ab = _script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        config, gate, budget = _rhythm(lib, loaded, min_new_turns=6, flesh=4000)
        answering, bursts = set(), _bursts_on(lib)

        def ask(messages):
            answering.add(threading.get_ident())
            return "Noted."

        onion = ab.onion_arm(ask, librarian=lib, config=config, summarize=_faithful, wrap=lambda block: block,
                             gate=gate, budget=budget, mode="launcher")
        ab.run_arm(ab.TURNS[:9], onion)
        assert len(bursts) == 3, "eighteen turns of growth, six at a time"
        counted = lib.counters()["burst"]
        assert counted.get("ran", 0) + counted.get("in_flight", 0) == 3, "each ran, or found the one before still in flight"
        assert not {t.ident for t in bursts} & answering, "off the answering thread"
    finally:
        lib.reset_librarian()
        restore()


def test_da12_the_fast_mode_curates_after_every_turn_on_the_answering_thread():
    ab = _script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        config, gate, budget = _rhythm(lib, loaded)
        answering, summarised = set(), set()

        def ask(messages):
            answering.add(threading.get_ident())
            return "Noted."

        def summarize(turns):
            summarised.add(threading.get_ident())
            return _faithful(turns)

        onion = ab.onion_arm(ask, librarian=lib, config=config, summarize=summarize, wrap=lambda block: block,
                             gate=gate, budget=budget, mode="fast")
        for turn in ab.TURNS[:4]:
            onion(turn.text)
            state = lib.state_for("drift-ab", config)
            assert state.flesh.tokens(lib.estimate_tokens) <= budget.flesh, "curated before the next turn"
        assert summarised == answering, "on the answering thread"
        assert "burst" not in lib.counters(), "no launcher in the fast mode"
    finally:
        lib.reset_librarian()
        restore()


@pytest.mark.parametrize("mode", ["fast", "launcher"])
def test_da13_both_modes_ask_for_the_second_summary(mode):
    ab = _script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        config, gate, budget = _rhythm(lib, loaded, min_new_turns=2)
        asked = []

        def reask(turns, missing):
            asked.append(missing)
            return _LOSSY

        onion = ab.onion_arm(lambda messages: "Noted: Bob moved the build to Berlin on 2026-03-04.", librarian=lib,
                             config=config, summarize=lambda turns: _LOSSY, reask=reask, wrap=lambda block: block,
                             gate=gate, budget=budget, mode=mode)
        ab.run_arm(ab.TURNS[:6], onion)
        assert sum(lib.counters().get("eviction", {}).values()) >= 2, "control: spans left"
        assert asked, "the second asking was made"
    finally:
        lib.reset_librarian()
        restore()


def test_da14_the_report_names_its_mode_and_carries_the_engagement_and_the_generator(capsys):
    loaded, restore = _window()
    try:
        ab = _script()
        lib = loaded["opti_oignon.memory.librarian"]
        probes = loaded["opti_oignon.memory.probes"]
        backend = _Backend(None)
        code = ab.main(["--model", "llama3", "--mode", "fast"], resolve=lambda model: backend)
        out = capsys.readouterr()
        assert code == 0, out.err
        report = json.loads(out.out)
        assert report["mode"] == "fast"
        assert report["generator"] == probes.GENERATOR_VERSION
        assert report["lexicon"] == loaded["opti_oignon.memory.peels"].load_gate().lexicon.fingerprint
        engagement = report["engagement"]
        assert engagement["counts"]["eviction"] and "refusal" in engagement["counts"]
        tokens = engagement["block_tokens"]
        assert len(tokens["per_turn"]) == report["turns"] and tokens["max"] >= tokens["mean"] > 0
        lib.reset_librarian()
    finally:
        restore()


def test_da15_an_arm_ends_only_once_the_bursts_it_launched_have_ended():
    ab = _script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        config, gate, budget = _rhythm(lib, loaded, min_new_turns=2)
        bursts = _bursts_on(lib)

        def slow(turns):
            time.sleep(0.05)
            return _faithful(turns)

        onion = ab.onion_arm(lambda messages: "Noted.", librarian=lib, config=config, summarize=slow,
                             wrap=lambda block: block, gate=gate, budget=budget, mode="launcher")
        ab.run_arm(ab.TURNS[:6], onion)
        assert bursts, "control: bursts were launched"
        assert not [t for t in bursts if t.is_alive()], "every burst ended before the arm did"
    finally:
        lib.reset_librarian()
        restore()


def test_da16_the_onion_arm_refuses_a_counters_path(tmp_path):
    ab = _script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        config, gate, budget = _rhythm(lib, loaded)
        with pytest.raises(ValueError, match="counters"):
            ab.onion_arm(lambda messages: "Noted.", librarian=lib,
                         config=replace(config, counters_path=str(tmp_path / "counts.json")), summarize=_faithful,
                         wrap=lambda block: block)
    finally:
        lib.reset_librarian()
        restore()


def test_da17_with_the_registry_s_summariser_each_fast_turn_is_a_run_of_its_own():
    ab = _script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        config, gate, budget = _rhythm(lib, loaded)
        runs, real = [], lib._Run

        class Counted(real):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                runs.append(self)

        lib._Run = Counted
        backend = _Backend(None)
        onion = ab.onion_arm(lambda messages: "Noted.", librarian=lib, config=config, summarize=None,
                             resolve=lambda model: backend, wrap=lambda block: block, gate=gate, budget=budget,
                             mode="fast")
        ab.run_arm(ab.TURNS[:4], onion)
        assert backend.calls >= 4, "control: the registry's summariser was asked"
        assert len(runs) == 4, "one run per measured turn"
    finally:
        lib.reset_librarian()
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
