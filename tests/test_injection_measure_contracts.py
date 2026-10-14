#!/usr/bin/env python3
"""A measurement of a served model by the injection bench says what it measured, how surely, and how long it took.

A served model samples: two runs of the same suite need not give the same
counts, and a count of four in sixteen says little without how far the
true rate may lie. The bench repeats its runs on request, gives four rates
of a model's rows their 95% Wilson interval and, from a client that times
its turns, the seconds the model took per run after one untimed warm-up
turn, and names the model and the decoding options it was handed. A
persona is deterministic, so its report carries none of this and stays as
it was:

  * IM1 -- REPEATING A RUN MULTIPLIES EVERY DENOMINATOR, AND A PERSONA'S
    COUNTS: ``repeat`` runs every run of the rows and of the memory
    injection that many times; one repeat is the report as it was, with no
    trace of the option; a repeat that is not a count of at least one is
    refused.
  * IM2 -- WITH A MODEL, THE REPORT SAYS WHAT IT MEASURED, AND WITHOUT ONE
    NOTHING MORE: each row of a model carries the intervals of its utility,
    utility under attack, targeted success and attempt rate, each of that
    rate's own counts; from a client that times its turns, the model's
    seconds per run (median, longest, total) summed over each run's turns,
    with the warm-up turn kept out and reported apart; from a client that
    does not, no seconds at all; the report names the model, its options
    and the repeats; a persona's report and rows have exactly the keys they
    had; a temperature without a model, below 0 or not finite, a repeat of
    none, and a repeat at a temperature of 0 are refused before any model
    is asked.
  * IM3 -- THE INTERVAL IS WILSON'S: known values at the bounds and in the
    middle, never outside [0, 1], always around the observed rate, and
    narrower as the runs grow.
  * IM4 -- THE REGISTRY CLIENT HANDS ITS OPTIONS AND TIMES ITS TURNS: a
    temperature reaches the engine beside the tools, on every turn, without
    the tools of one turn staying for the next; a client given no option
    hands the engine what it always did; each turn adds the seconds its
    ``generate`` took, a turn that failed included.

The bench is loaded through the shared isolation window of its own suite of
contracts, the client through its suite's window. Local-only.
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_agent_eval_client_through_registry_contracts as client_contracts  # noqa: E402
import test_injection_bench_contracts as bench_contracts  # noqa: E402

BUDGET_S = {
    "test_im1_repeating_a_run_multiplies_every_denominator_and_a_personas_counts": 10.0,
    "test_im2_with_a_model_the_report_says_what_it_measured_and_without_one_nothing_more": 10.0,
    "test_im3_the_interval_is_wilsons": 2.0,
    "test_im4_the_registry_client_hands_its_options_and_times_its_turns": 2.0,
}

_RATES = ("utility", "utility_under_attack", "targeted_asr", "attempt_rate")
_MEMORY = ("attempted", "landed_without_acceptance", "landed_after_review", "attempted_later", "carried_out_later")
_REPORT_KEYS = {"bench", "format", "suite", "machine_mode", "pairs", "rows", "memory", "confinement",
                "noninterference", "floors"}
_ROW_KEYS = {"persona", "mode_asked", "mode", "policy_asked", "policy", "responder", "utility", "utility_under_attack",
             "targeted_asr", "attempt_rate", "asr_by_kind", "attempts_by_kind", "asks_per_task", "asks_per_pair",
             "proposals_per_pair", "refusals_per_pair", "runs_stopped", "steps_dropped", "calls_enumerated",
             "answers_unread", "calls"}


# ---------------------------------------------------------------------------
# IM1 -- repeats
# ---------------------------------------------------------------------------
def _counts(rate):
    return rate["count"], rate["of"]


def test_im1_repeating_a_run_multiplies_every_denominator_and_a_personas_counts():
    bench, _, close = bench_contracts._window()
    try:
        suite = bench.load_suite()
        chosen = {"personas": ("obedient",), "settings": bench.parse_settings("daily:refuse_unendorsed:approve"),
                  "values": False, "memory_policies": ("refuse_unendorsed",)}
        once = bench.run_bench(suite, **chosen)
        again = bench.run_bench(suite, **chosen, repeat=1)
        thrice = bench.run_bench(suite, **chosen, repeat=3)
        refused = []
        for repeat in (0, -1, True, 2.0, "2"):
            with pytest.raises(ValueError):
                bench.run_bench(suite, **chosen, repeat=repeat)
            refused.append(repeat)
    finally:
        close()
    assert json.dumps(once, sort_keys=True) == json.dumps(again, sort_keys=True) and "repeat" not in once, once.keys()
    assert thrice["repeat"] == 3, thrice.get("repeat")
    row, rows3 = once["rows"][0], thrice["rows"][0]
    for name in _RATES:
        count, of = _counts(row[name])
        assert _counts(rows3[name]) == (3 * count, 3 * of), (name, row[name], rows3[name])
    for key in ("asr_by_kind", "attempts_by_kind"):
        for kind, rate in row[key].items():
            count, of = _counts(rate)
            assert _counts(rows3[key][kind]) == (3 * count, 3 * of), (key, kind)
    assert rows3["calls"] == 3 * row["calls"] and rows3["asks_per_pair"] == row["asks_per_pair"], (row, rows3)
    assert len(once["memory"]) == len(thrice["memory"]) == len(bench.REVIEWS), once["memory"]
    for first, third in zip(once["memory"], thrice["memory"]):
        for name in _MEMORY:
            count, of = _counts(first[name])
            assert _counts(third[name]) == (3 * count, 3 * of), (name, first[name], third[name])
    assert row["targeted_asr"]["of"] > 0 and row["attempt_rate"]["count"] > 0, ("control: counts to multiply", row)
    assert any(first["attempted_later"]["of"] for first in once["memory"]), ("control: a later session ran", once)
    assert refused == [0, -1, True, 2.0, "2"], refused


# ---------------------------------------------------------------------------
# IM2 -- what a model's report says
# ---------------------------------------------------------------------------
class _Timed:
    """A served model that does a task's one read, then answers; its turns take seconds from a script.

    A turn with no reply of its own in the conversation yet -- the warm-up,
    or the first turn of a run -- calls ``view`` on the mini suite's
    document and takes a quarter of a second more than the one before it;
    the next answers and takes one second.
    """

    def __init__(self, model, options=None):
        self.model = model
        self.options = dict(options or {})
        self.turn_seconds = []
        self.firsts = 0

    def stream(self, messages, tools=None):
        if not any(message.get("role") == "assistant" for message in messages):
            self.firsts += 1
            self.turn_seconds.append(0.25 * self.firsts)
            return [{"message": {"content": "", "tool_calls": [
                {"function": {"name": "view", "arguments": {"path": "doc.txt"}}}]}}]
        self.turn_seconds.append(1.0)
        return [{"message": {"content": "Done: I read doc.txt.", "tool_calls": []}}]


class _Untimed:
    """The same model, behind a client that records no time."""

    def stream(self, messages, tools=None):
        if not any(message.get("role") == "assistant" for message in messages):
            return [{"message": {"content": "", "tool_calls": [
                {"function": {"name": "view", "arguments": {"path": "doc.txt"}}}]}}]
        return [{"message": {"content": "Done: I read doc.txt.", "tool_calls": []}}]


def _timed_runner():
    runner = types.ModuleType("opti_oignon.agent_eval.runner")
    runner.made = []

    def client(model, options=None):
        made = _Timed(model, options)
        runner.made.append(made)
        return made

    runner.RegistryChatClient = client
    return runner


def test_im2_with_a_model_the_report_says_what_it_measured_and_without_one_nothing_more(tmp_path):
    runner = _timed_runner()
    bench, _, close = bench_contracts._window(seeded={"opti_oignon.agent_eval.runner": runner})
    try:
        mini = tmp_path / "mini.yaml"
        mini.write_text(json.dumps(bench_contracts._mini()), encoding="utf-8")
        served, fixed_out, persona = tmp_path / "served.json", tmp_path / "fixed.json", tmp_path / "persona.json"
        one = "daily:refuse_unendorsed:approve"
        code = bench.main(["--suite", str(mini), "--model", "local-model", "--temperature", "0.7", "--repeat", "2",
                           "--settings", one, "--json", str(served)])
        fixed_code = bench.main(["--suite", str(mini), "--model", "local-model", "--temperature", "0",
                                 "--settings", one, "--json", str(fixed_out)])
        persona_code = bench.main(["--suite", str(mini), "--persona", "obedient", "--json", str(persona)])
        built = len(runner.made)
        refused = {}
        for extra in (["--temperature", "0.2"], ["--repeat", "0"], ["--model", "local-model", "--repeat", "0"],
                      ["--model", "local-model", "--temperature", "-1"],
                      ["--model", "local-model", "--temperature", "nan"],
                      ["--model", "local-model", "--temperature", "inf"],
                      ["--model", "local-model", "--temperature", "0", "--repeat", "2"]):
            refused[" ".join(extra)] = bench.main(["--suite", str(mini), *extra, "--json", str(tmp_path / "x.json")])
        written = (tmp_path / "x.json").exists()
        report = json.loads(served.read_text(encoding="utf-8"))
        row = report["rows"][0]
        expected = {name: bench.wilson(row[name]["count"], row[name]["of"]) for name in _RATES}
        untimed = bench.run_bench(bench.load_suite(str(mini)), settings=bench.parse_settings(one),
                                  model_client=_Untimed(), memory=False)
        medians = (bench._seconds([3.0, 1.0, 2.0]), bench._seconds([4.0, 1.0, 3.0, 2.0]), bench._seconds([]))
    finally:
        close()
    assert code == 0 and fixed_code == 0, (code, fixed_code)
    assert report["model"] == {"name": "local-model", "options": {"temperature": 0.7}}, report["model"]
    assert report["repeat"] == 2 and runner.made[0].options == {"temperature": 0.7}, runner.made
    fixed = json.loads(fixed_out.read_text(encoding="utf-8"))
    assert fixed["model"]["options"] == {"temperature": 0.0} and "repeat" not in fixed, fixed["model"]
    assert _counts(row["utility"]) == (2, 2) and _counts(row["targeted_asr"]) == (0, 2), ("control: the model", row)
    assert row["intervals"] == expected and row["intervals"]["utility"] != row["intervals"]["targeted_asr"], row
    assert report["model_warm_up_seconds"] == 0.25, report.get("model_warm_up_seconds")
    assert row["model_seconds"] == {"runs": 4, "median": 1.875, "max": 2.25, "total": 7.5}, row["model_seconds"]
    assert medians == ({"runs": 3, "median": 2.0, "max": 3.0, "total": 6.0},
                       {"runs": 4, "median": 2.5, "max": 4.0, "total": 10.0},
                       {"runs": 0, "median": 0.0, "max": 0.0, "total": 0.0}), medians
    untimed_row = untimed["rows"][0]
    assert "intervals" in untimed_row and "model_seconds" not in untimed_row, sorted(untimed_row)
    assert "model_warm_up_seconds" not in untimed, sorted(untimed)
    plain = json.loads(persona.read_text(encoding="utf-8"))
    assert persona_code == 0 and set(plain) == _REPORT_KEYS, sorted(plain)
    assert all(set(each) == _ROW_KEYS for each in plain["rows"]), sorted(plain["rows"][0])
    assert set(refused.values()) == {2} and not written, (refused, written)
    assert all(not made.turn_seconds for made in runner.made[built:]), ("no model is asked on a refusal",
                                                                          [made.turn_seconds for made in runner.made])


# ---------------------------------------------------------------------------
# IM3 -- Wilson
# ---------------------------------------------------------------------------
def test_im3_the_interval_is_wilsons():
    bench, _, close = bench_contracts._window()
    try:
        known = {(count, total): bench.wilson(count, total)
                 for count, total in ((0, 16), (16, 16), (8, 16), (3, 10), (1, 4), (0, 0))}
        grid = [(count, total, bench.wilson(count, total)) for total in range(1, 41) for count in range(total + 1)]
        widths = [bench.wilson(n // 2, n) for n in (4, 16, 64)]
    finally:
        close()
    assert known == {(0, 16): [0.0, 0.1936], (16, 16): [0.8064, 1.0], (8, 16): [0.28, 0.72],
                     (3, 10): [0.1078, 0.6032], (1, 4): [0.0456, 0.6994], (0, 0): [0.0, 1.0]}, known
    outside = [(count, total, interval) for count, total, interval in grid
               if not 0.0 <= interval[0] <= count / total + 5e-5 or not count / total - 5e-5 <= interval[1] <= 1.0]
    assert outside == [] and len(grid) == 860, outside[:5]
    spans = [high - low for low, high in widths]
    assert spans[0] > spans[1] > spans[2] > 0, widths


# ---------------------------------------------------------------------------
# IM4 -- the registry client
# ---------------------------------------------------------------------------
class _Clock:
    """A monotonic clock that only the scripted engine moves."""

    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


def test_im4_the_registry_client_hands_its_options_and_times_its_turns():
    mod, scripted, restore = client_contracts._open()
    clock = _Clock()
    steps = iter((0.75, 1.5, 0.25, 0.5, 0.375))
    answer = scripted.chat

    def chat(**kwargs):
        clock.now += next(steps)
        if kwargs.get("messages") == [{"role": "user", "content": "fail"}]:
            raise RuntimeError("the engine stopped")
        return answer(**kwargs)

    scripted.chat = chat
    real_time = mod.time
    mod.time = clock
    try:
        tuned = mod.RegistryChatClient("m", options={"temperature": 0.0})
        list(tuned.stream(client_contracts._MSGS, client_contracts._TOOLS))
        list(tuned.stream(client_contracts._MSGS))
        plain = mod.RegistryChatClient("m")
        list(plain.stream(client_contracts._MSGS, client_contracts._TOOLS))
        list(plain.stream(client_contracts._MSGS))
        with pytest.raises(RuntimeError, match="stopped"):
            list(plain.stream([{"role": "user", "content": "fail"}]))
        timed = (list(tuned.turn_seconds), list(plain.turn_seconds))
    finally:
        mod.time = real_time
        restore()
    with_tools, without_tools, plain_tools, plain_bare = scripted.calls
    assert with_tools["options"] == {"temperature": 0.0}, with_tools
    assert with_tools["tools"] == client_contracts._TOOLS, with_tools
    assert without_tools["options"] == {"temperature": 0.0} and "tools" not in without_tools, without_tools
    assert plain_tools["options"] == {} and plain_tools["tools"] == client_contracts._TOOLS, plain_tools
    assert plain_bare["options"] is None and "tools" not in plain_bare, plain_bare
    assert timed == ([0.75, 1.5], [0.25, 0.5, 0.375]), timed
