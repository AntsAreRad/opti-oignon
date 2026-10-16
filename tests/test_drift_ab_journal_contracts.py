#!/usr/bin/env python3
"""Contracts for the drift A/B's onion arm, which sends each turn as the onion's journal composed it.

With the onion on, the chat path sends a conversation's turn as the window
the librarian planned and composed for it, in place of the history
window. The host command's onion arm does the same, so the A/B sets the
journal against the plain history window, turn by turn, on one model.

  * DA18 -- the onion arm sends, for each turn, exactly the messages
    compose_window composed for it, labels left behind; it opens on the
    bench's system prompt, as the plain arm does, which sends the system
    prompt, its history window and the turn; the onion arm's last message
    is the user's turn and ends with the turn's words; a persistence path
    is refused. It keeps, turn by turn, the window's layers summed, what
    the memory laid of them -- the Core, the receipts digest, the peels,
    the tail -- and the Flesh's.
  * DA19 -- the host command sends each window with the context it was
    composed for, and the plain arm's requests with none: the arm hands
    the window it composed to the ask, and both arms keep the same
    sampling and the same room for the answer, within the window's; the
    pair is tried at that context, the run's peak.
  * DA20 -- the host command plans the onion arm within the model's window
    and counts it with the model's counter, as the executor reads them
    from the context manager: every window composed under the rung that
    window holds and sent with it, every plan counted for the model the
    turn is sent to; an in-process engine with no governor to admit it
    bounds that window by the context it loads at, as on the chat path, and
    one a governor admits does not; a context manager that cannot be read
    leaves the executor's own 32768, still bounded.
  * DA21 -- the host command's report carries, beside the window's count,
    what the memory laid of it and the Flesh's, turn by turn: on the
    bench's own run the memory engages.

Local-only (the public distribution ships no tests). The script is loaded
as the drift A/B suite loads it (tests/test_drift_ab_contracts.py); the
onion's modules come through the shared isolation window, and the model is
a recording seam.
"""

import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_drift_ab_contracts as da  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_da18_the_onion_arm_sends_each_turn_as_the_journal_composed_it_and_opens_on_the_benchs_system_prompt": 12.0,
    "test_da19_the_host_command_sends_each_window_with_the_context_it_was_composed_for": 4.0,
    "test_da20_the_onion_arm_is_planned_within_the_models_window_and_counted_with_its_counter": 12.0,
    "test_da21_the_host_report_carries_what_the_memory_laid_beside_the_windows_count": 45.0,
}


_ONION = ("probes", "drift", "core_store", "receipts", "composer", "peels", "librarian")


def _window(seeded=None):
    """The onion's modules and the untrusted-data envelope, each from its file, in the shared window, beside the
    stand-ins ``seeded``."""
    targets = {f"opti_oignon.memory.{name}": source("memory", f"{name}.py") for name in _ONION}
    targets["opti_oignon.agent.untrusted_context"] = source("agent", "untrusted_context.py")
    return isolate(targets=targets, seeded=seeded, packages=("opti_oignon.memory", "opti_oignon.agent"))


def _config(lib):
    return lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=1, temperature=0.1,
                               num_predict=64, persist_path="", require_encryption=False)


def _composing(lib):
    """Every window the journal composes, kept in order as it is handed back."""
    composed, compose = [], lib.compose_window

    def composing(plan, **kwargs):
        window = compose(plan, **kwargs)
        composed.append(window)
        return window

    lib.compose_window = composing
    return composed


def _recorder(calls, reply="Noted."):
    def ask(messages):
        calls.append(messages)
        return reply
    return ask


# ---------------------------------------------------------------------------
# DA18 -- each turn sent as the journal composed it
# ---------------------------------------------------------------------------
def test_da18_the_onion_arm_sends_each_turn_as_the_journal_composed_it_and_opens_on_the_benchs_system_prompt(tmp_path):
    ab = da._script()
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        peels, composer = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.composer"]
        uc = loaded["opti_oignon.agent.untrusted_context"]
        config = _config(lib)
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)
        composed = _composing(lib)
        plain_calls, onion_calls = [], []
        turns = ab.TURNS[:40]
        ab.run_arm(turns, ab.plain_arm(_recorder(plain_calls), history_tokens=300))
        onion = ab.onion_arm(_recorder(onion_calls), librarian=lib, config=config, summarize=da._faithful,
                             wrap=uc.wrap, gate=gate, budget=budget)
        ab.run_arm(turns, onion)
        assert len(plain_calls) == len(onion_calls) == len(composed) == len(turns), (
            len(plain_calls), len(onion_calls), len(composed))
        system = {"role": "system", "content": ab.SYSTEM_PROMPT}
        history = []
        for index, (turn, plain, sent, window) in enumerate(zip(turns, plain_calls, onion_calls, composed)):
            assert window.refusal is None, (index, window.refusal)
            assert sent == [{"role": role, "content": content} for role, content, _label in window.messages], (
                "exactly the messages the journal composed for the turn", index)
            assert all(set(message) == {"role", "content"} for message in sent), ("labels left behind", index)
            assert plain == [system, *ab.history_window(history, 300), {"role": "user", "content": turn.text}], (
                "the plain arm sends the system prompt, its history window and the turn", index)
            assert sent[0] == plain[0] == system, ("both arms open on the bench's system prompt", index)
            assert sent[-1]["role"] == "user" and sent[-1]["content"].endswith(turn.text), (
                "the onion arm's last message is the turn, ending with its words", index)
            history.extend([{"role": "user", "content": turn.text}, {"role": "assistant", "content": "Noted."}])
        assert any(dict(window.layers)["peels"] > 0 for window in composed), "control: the journal laid peels"
        assert any(sent != plain for sent, plain in zip(onion_calls, plain_calls)), (
            "control: the journal is not the plain window")
        assert onion.block_tokens == [sum(count for _layer, count in window.layers) for window in composed], (
            "the report's block tokens are the window's layers, summed")
        laid = [dict(window.layers) for window in composed]
        assert onion.memory_tokens == [sum(layers[name] for name in ("core", "receipts", "peels", "tail"))
                                       for layers in laid], "what the memory laid: the Core, digest, peels and tail"
        assert onion.flesh_tokens == [layers["flesh"] for layers in laid], "the Flesh's, beside it"
        assert any(onion.memory_tokens) and any(onion.flesh_tokens), "control: both engaged"
        with pytest.raises(ValueError, match="persistence"):
            ab.onion_arm(_recorder([]), librarian=lib, config=replace(config, persist_path=str(tmp_path / "onion.db")),
                         summarize=da._faithful, wrap=uc.wrap)
    finally:
        lib.reset_librarian()
        restore()


# ---------------------------------------------------------------------------
# DA19 -- each window sent with the context it was composed for
# ---------------------------------------------------------------------------
class _Recording(da._Backend):
    """A backend that serves every model and keeps each request's model, messages and options."""

    def __init__(self):
        super().__init__(None)
        self.requests = []

    def generate(self, model, messages, **kwargs):
        self.requests.append({"model": model, "messages": messages, "options": dict(kwargs.get("options") or {})})
        return super().generate(model, messages, **kwargs)


def test_da19_the_host_command_sends_each_window_with_the_context_it_was_composed_for(capsys):
    loaded, restore = _window()
    lib = loaded["opti_oignon.memory.librarian"]
    try:
        ab = da._script()
        ab.TURNS = ab.TURNS[:6]
        composed = _composing(lib)
        backend = _Recording()
        code = ab.main(["--model", "llama3", "--mode", "fast"], resolve=lambda model: backend)
        out = capsys.readouterr()
        assert code == 0, out.err
        system = {"role": "system", "content": ab.SYSTEM_PROMPT}
        asked = [request for request in backend.requests
                 if request["model"] == "llama3" and request["messages"][:1] == [system]]
        plain, onion = asked[: len(ab.TURNS)], asked[len(ab.TURNS):]
        assert len(plain) == len(onion) == len(composed) == len(ab.TURNS), (len(asked), len(composed))
        for request, window in zip(onion, composed):
            assert request["messages"] == [{"role": role, "content": content}
                                           for role, content, _label in window.messages], "control: the onion arm"
            assert request["options"].get("num_ctx") == window.num_ctx, (request["options"], window.num_ctx)
            assert request["options"]["num_predict"] <= window.num_predict, "the answer's room is within the window's"
        assert all("num_ctx" not in request["options"] for request in plain), [r["options"] for r in plain]
        sampled = {tuple(sorted((key, value) for key, value in request["options"].items() if key != "num_ctx"))
                   for request in asked}
        assert len(sampled) == 1, ("both arms sample alike and keep the same room for the answer", sampled)
        (trial,) = [request for request in backend.requests
                    if request["model"] == "llama3" and request["messages"] == ab._TRIAL]
        assert trial["options"].get("num_ctx") == composed[0].num_ctx, ("the pair is tried at the run's peak",
                                                                        trial["options"])
        lib.reset_librarian()
    finally:
        restore()


# ---------------------------------------------------------------------------
# DA20 -- planned within the model's window, counted with its counter
# ---------------------------------------------------------------------------
def _context_manager(window, chars_per_token):
    """A context manager that gives every model ``window`` and counts a text at ``chars_per_token``, keeping the
    model each count was asked for."""
    asked = []
    module = types.ModuleType("opti_oignon.context_manager")
    module.get_model_limits = lambda model: SimpleNamespace(context_window=window, max_output=4096)

    def estimate_tokens(text, model=None):
        asked.append(model)
        return int(len(str(text)) / chars_per_token)

    module.estimate_tokens = estimate_tokens
    return module, asked


class _InProcess(_Recording):
    """An in-process engine, loaded at its own context."""

    name = "llama_cpp"
    _n_ctx = 4096


def test_da20_the_onion_arm_is_planned_within_the_models_window_and_counted_with_its_counter(capsys):
    # The model's window as the context manager gives it, bounded by an in-process engine's own context when no
    # governor admits it. The window seeds no governor: the bench reads it off, as the executor reads one it
    # cannot reach. A governor switched on admits the context instead, and the engine's own bounds nothing, as in
    # the executor's governed case (JW7). A context manager that cannot be read leaves 32768, still bounded.
    cases = ((_Recording, False, True, 8192), (_InProcess, False, True, 4096), (_InProcess, True, True, 8192),
             (_InProcess, False, False, 4096))
    for engine, governed, managed, rung in cases:
        manager, asked = _context_manager(8192, 3.5)
        loaded, restore = _window(seeded={"opti_oignon.context_manager": manager} if managed else None)
        lib = loaded["opti_oignon.memory.librarian"]
        try:
            ab = da._script()
            ab.TURNS = ab.TURNS[:6]
            assert ab._governor_on() is False, "control: the window seeds no governor"
            if governed:
                ab._governor_on = lambda: True
            composed, counters, plan = _composing(lib), [], lib.plan_window

            def planning(conversation_id, plan=plan, counters=counters, **kwargs):
                counters.append(kwargs["count"])
                return plan(conversation_id, **kwargs)

            lib.plan_window = planning
            backend = engine()
            code = ab.main(["--model", "llama3", "--mode", "fast"], resolve=lambda model, backend=backend: backend)
            out = capsys.readouterr()
            assert code == 0, out.err
            assert len(composed) == len(counters) == len(ab.TURNS), (len(composed), len(counters))
            assert [(window.rung, window.num_ctx) for window in composed] == [(rung, rung)] * len(ab.TURNS), (
                "planned within the model's window", engine.__name__, governed, managed,
                [(window.rung, window.num_ctx) for window in composed])
            system = {"role": "system", "content": ab.SYSTEM_PROMPT}
            onion = [request["options"].get("num_ctx") for request in backend.requests
                     if request["model"] == "llama3" and request["messages"][:1] == [system]][len(ab.TURNS):]
            assert onion == [rung] * len(ab.TURNS), (engine.__name__, onion)
            (trial,) = [request for request in backend.requests
                        if request["model"] == "llama3" and request["messages"] == ab._TRIAL]
            assert trial["options"].get("num_ctx") == rung, (engine.__name__, trial["options"])
            probe = "Alice lives in Berlin and works at Contoso."
            assert int(len(probe) / 3.5) != ab.estimate_tokens(probe), "control: the two counts differ"
            for count in counters:
                asked.clear()
                if managed:
                    assert count(probe) == int(len(probe) / 3.5), "the model's counter, as the executor hands it"
                    assert asked == ["llama3"], ("counted for the model the turn is sent to", asked)
                else:
                    assert count(probe) == ab.estimate_tokens(probe), "no context manager: the bench's estimate"
            lib.reset_librarian()
        finally:
            restore()


# ---------------------------------------------------------------------------
# DA21 -- what the memory laid, in the report
# ---------------------------------------------------------------------------
def test_da21_the_host_report_carries_what_the_memory_laid_beside_the_windows_count(capsys):
    loaded, restore = _window()
    try:
        ab = da._script()
        code = ab.main(["--model", "llama3", "--mode", "fast"], resolve=lambda model: da._Backend(None))
        out = capsys.readouterr()
        assert code == 0, out.err
        report = json.loads(out.out)
        engagement = report["engagement"]
        whole, memory, flesh = (engagement[key] for key in ("block_tokens", "memory_tokens", "flesh_tokens"))
        assert len(whole["per_turn"]) == len(memory["per_turn"]) == len(flesh["per_turn"]) == report["turns"]
        assert memory["max"] > 0, "the memory engaged: what it laid reached the window"
        assert all(laid + kept <= total for laid, kept, total in zip(memory["per_turn"], flesh["per_turn"],
                                                                     whole["per_turn"])), "parts of the whole"
        assert memory["max"] >= memory["mean"] and flesh["max"] >= flesh["mean"] > 0, (memory, flesh)
        loaded["opti_oignon.memory.librarian"].reset_librarian()
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
