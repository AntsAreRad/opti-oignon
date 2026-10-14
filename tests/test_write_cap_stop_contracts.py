#!/usr/bin/env python3
"""Contracts for the stop of an agent run that writes past its gate's bound.

An agent run's pending-write gate lets the run propose at most
``max_per_run`` writes (``config/pending_writes.yaml``). A run that tries to
propose more is likelier to be following words it read than its task: the
gate refuses the write and names a stop for its run (``WriteGate.halt``);
the agent loop ends the run after that round, with the stop as its reason
and an ``aborted`` event that names it; the route hands the gate's stop to
the loop and reports the reason. What the run proposed before waits for the
user's review.

  * WC1 -- THE LOOP ENDS A RUN ON ITS HALT REASON, AFTER THE ROUND: a run
    whose ``halt`` gives a reason after its second round ends there, with
    that reason as its stop and an ``aborted`` event naming it, and the
    model is not asked again; with no ``halt``, one that fails, or one that
    gives no reason, the same run goes on to its answer.
  * WC2 -- THE GATE NAMES THE STOP ONLY PAST ITS BOUND: a gate bound to two
    proposals names no stop after two, names ``proposal_cap`` once a third
    write is refused, and never for the words the user typed, which are
    written past the bound as before.
  * WC3 -- THE ROUTE HANDS THE RUN'S GATE STOP TO THE LOOP AND REPORTS IT: an
    agent run whose model writes past the bound gets its gate's stop from
    the route, ends with ``proposal_cap`` in the run's status, and leaves
    exactly the bound's proposals in the queue; the next run on the same
    manager, its write tools withdrawn, gets no stop, the last run's least
    of all.
  * WC4 -- THE BENCH'S RUNS STOP WHERE THE ROUTE'S DO: the injection bench
    builds its runs with the gate's stop, so the enumerator that floods the
    stores is stopped at the bound, with the bound's proposals queued, and
    its row counts the runs stopped there; an obedient model's never are.
  * WC5 -- THE BENCH REACHES NO STORE AND NO NETWORK OF THE USER'S (ib8, with
    the bound raised for the run, so the enumerator goes on to the
    identifiers it read).
  * WC6 -- THE AGENT PANEL ENDS A RUN ON ITS ABORTED EVENT AND SAYS WHY: the
    pure module reads the reason a loop event ends a run with -- the
    reason an ``aborted`` event names among them, never a subtask's -- and
    the words for it; the panel stops showing the run as running on any of
    them, says a run stopped for too many writes in words that point to the
    review, and shows no cancel the server says found no run.
  * WC8 -- A RUN STOPPED AT THE BOUND WAKES NO TEACHER: with the teacher armed
    and the emergency stop released, a run stopped at the bound never builds
    an escalator -- it was likelier following words it read than its task --
    while a run that ended otherwise is offered to it.
  * WC7 -- THE REVIEW GATHERS A RUN'S PROPOSALS UNDER ONE CONTROL: an agent
    run that made several of the proposals shown gets one line that says how
    many, its tag on each of them, and one control, named for its count and
    its run, that selects them and only them, so they are declined in one
    decision; a run with one proposal, and a proposal of no run, get none.

The loop, the gate and the bench are loaded through the shared isolation
windows of their own suites; the pure module runs under Node
(``_frontend.run_ts``) and the panel is server-rendered (``_frontend.ssr``).
Local-only.
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_injection_bench_contracts as bench_contracts  # noqa: E402
import test_pending_writes_contracts as pw_contracts  # noqa: E402
from _frontend import read, run_ts, ssr  # noqa: E402
from _isolation import source  # noqa: E402

BUDGET_S = {
    "test_wc1_the_loop_ends_a_run_on_its_halt_reason_after_the_round": 5.0,
    "test_wc2_the_gate_names_the_stop_only_past_its_bound": 5.0,
    "test_wc3_the_route_hands_the_runs_gate_stop_to_the_loop_and_reports_it": 10.0,
    "test_wc4_the_benchs_runs_stop_where_the_routes_do": 10.0,
    "test_wc5_the_bench_reaches_no_store_and_no_network_of_the_users_with_the_bound_raised": 10.0,
    "test_wc6_the_agent_panel_ends_a_run_on_its_aborted_event_and_says_why": 15.0,
    "test_wc7_the_review_gathers_a_runs_proposals_under_one_control": 15.0,
    "test_wc8_a_run_stopped_at_the_bound_wakes_no_teacher": 10.0,
}

_LOOP = "opti_oignon.agent.loop"


# ---------------------------------------------------------------------------
# WC1 -- the loop ends a run on its halt reason
# ---------------------------------------------------------------------------
class _Reader:
    """A model that plays a script of rounds, each a list of calls, then answers; then, asked to verify, plays the
    verifier's rounds."""

    def __init__(self, rounds, checks=()):
        self.rounds = [list(each) for each in rounds]
        self.checks = [list(each) for each in checks]
        self.asked = 0

    def stream(self, messages, tools=None):
        self.asked += 1
        if self.asked <= len(self.rounds):
            calls = [{"function": {"name": name, "arguments": dict(arguments)}}
                     for name, arguments in self.rounds[self.asked - 1]]
            return [{"message": {"content": f"Reading, step {self.asked}.", "tool_calls": calls}}]
        check = self.asked - len(self.rounds) - 2
        if 0 <= check < len(self.checks):
            calls = [{"function": {"name": name, "arguments": dict(arguments)}}
                     for name, arguments in self.checks[check]]
            return [{"message": {"content": f"Checking, step {check + 1}.", "tool_calls": calls}}]
        return [{"message": {"content": "Done: the files are read.", "tool_calls": []}}]


_ONE_A_ROUND = ([("view", {"path": "doc.txt"})], [("view", {"path": "key.txt"})], [("ls", {"path": "."})],
                [("glob", {"pattern": "*.txt"})])
_THREE_AT_ONCE = ([("view", {"path": "doc.txt"}), ("view", {"path": "key.txt"}), ("ls", {"path": "."})],
                  [("glob", {"pattern": "*.txt"})])
_SAME_THRICE = ([("view", {"path": "doc.txt"})], [("view", {"path": "doc.txt"})], [("view", {"path": "doc.txt"})],
                [("view", {"path": "doc.txt"})])
_WITH_A_TASK = ([("view", {"path": "doc.txt"}), ("task", {"description": "look again", "prompt": "Read doc.txt."})],
                [("glob", {"pattern": "*.txt"})])


def _after_answer(client, reason="proposal_cap"):
    """A halt that names its reason only once the model has answered the run (the verifier's turn)."""
    return lambda: reason if client.asked > len(client.rounds) + 1 else None


def _halting(after, reason="proposal_cap"):
    asked = []

    def halt():
        asked.append(1)
        return reason if len(asked) >= after else None

    return halt


def _raising():
    raise RuntimeError("the gate could not be read")


def test_wc1_the_loop_ends_a_run_on_its_halt_reason_after_the_round(caplog):
    bench, loaded, close = bench_contracts._window()
    outcomes, warned, said, ran, asked_doom = {}, {}, {}, {}, []
    try:
        loop = loaded[_LOOP]
        suite = bench.parse_suite(bench_contracts._mini(), name="mini")
        cases = (("halted", _THREE_AT_ONCE, {"halt": _halting(3)}), ("task halted", _WITH_A_TASK,
                                                                       {"halt": _halting(2)}),
                 ("verified", _ONE_A_ROUND, "verify"), ("verified calm", _ONE_A_ROUND, "verify calm"),
                 ("verifier halted", _ONE_A_ROUND, "verify calls"),
                 ("repeating halted", _SAME_THRICE, "repeat"),
                 ("none", _ONE_A_ROUND, {}),
                 ("failing", _ONE_A_ROUND, {"halt": _raising}),
                 ("no reason", _ONE_A_ROUND, {"halt": _halting(1, reason="")}),
                 ("not a reason", _ONE_A_ROUND, {"halt": _halting(1, reason=True)}),
                 ("round cap", _ONE_A_ROUND, {"max_rounds": 2}), ("cancelled", _ONE_A_ROUND, "cancel"))
        for case, rounds, extra in cases:
            world = bench._World(suite)
            tool_set = bench._bench_registry(world).build("daily")
            checks = ([("bash", {"command": "echo one"}), ("bash", {"command": "echo two"})],
                      [("bash", {"command": "echo three"})])
            client, events = _Reader(rounds, checks if extra == "verify calls" else ()), []
            if extra == "cancel":
                kwargs = {"should_continue": lambda client=client: client.asked < 2}
            elif extra == "verify":
                kwargs = {"verify": True, "halt": _after_answer(client)}
            elif extra == "verify calm":
                kwargs = {"verify": True, "halt": _halting(10 ** 6)}
            elif extra == "verify calls":
                kwargs = {"verify": True, "halt": _after_answer(client)}
            elif extra == "repeat":
                kwargs = {"halt": _halting(6), "approval_fn": lambda *args, **kw: asked_doom.append(args) and False}
            else:
                kwargs = extra
            caplog.clear()
            result = loop.run("read the files", model_client=client, sandbox=world.sandbox, mode="daily",
                              system_prompt=tool_set.system_prompt_section(), tools=tool_set.native_tools(),
                              tool_handlers=dict(tool_set.tool_handlers), include_memory=False,
                              on_event=events.append, **kwargs)
            aborted = [(event.round, dict(event.data)) for event in events if event.kind == "aborted"]
            halted = [r for r in result.tool_results if getattr(r, "reason", "") == "halted"]
            outcomes[case] = (result.stop_reason, result.rounds, client.asked, aborted,
                              len([r for r in result.tool_results if getattr(r, "executed", False)]), len(halted),
                              result.final_text)
            said[case] = [(r.tool_name, r.observation, r.mode) for r in halted]
            ran[case] = list(world.commands)
            warned[case] = any("halt check failed" in record.getMessage() for record in caplog.records)
    finally:
        close()
    assert outcomes["halted"] == ("proposal_cap", 1, 1, [(1, {"reason": "proposal_cap"})], 2, 1,
                                  "Reading, step 1."), outcomes["halted"]
    assert said["halted"] == [("ls", "Not run: the run is stopping (proposal_cap).", "daily")], said["halted"]
    assert outcomes["task halted"][:2] == ("proposal_cap", 1) and outcomes["task halted"][4:6] == (1, 1), (
        outcomes["task halted"])
    assert said["task halted"] == [("task", "Not run: the run is stopping (proposal_cap).", "daily")], (
        "a task later in the round is not run either", said["task halted"])
    assert outcomes["verified"][0] == "proposal_cap" and outcomes["verified"][3] == [
        (5, {"reason": "proposal_cap"})], ("a verifier that meets the gate's stop ends the run with it",
                                           outcomes["verified"])
    for case in ("none", "failing", "no reason", "not a reason"):
        assert outcomes[case] == ("done", 5, 5, [], 4, 0, "Done: the files are read."), (case, outcomes[case])
    assert outcomes["round cap"] == ("max_rounds", 2, 2, [(2, {"reason": "max_rounds"})], 2, 0,
                                     "Reading, step 2."), outcomes["round cap"]
    assert outcomes["cancelled"] == ("cancelled", 2, 2, [(2, {"reason": "cancelled"})], 2, 0,
                                     "Reading, step 2."), outcomes["cancelled"]
    assert outcomes["verified calm"][:4] == ("done", 5, 6, []), ("a verified run the gate never stopped ends done",
                                                                 outcomes["verified calm"])
    assert outcomes["repeating halted"][:3] == ("proposal_cap", 3, 3) and asked_doom == [], (
        "a round that repeats and meets the halt ends on the halt, and asks no one to go on",
        outcomes["repeating halted"], asked_doom)
    assert outcomes["verifier halted"][:3] == ("proposal_cap", 5, 6) and ran["verifier halted"] == [], (
        "the verifier's calls are held by the halt, and it stops after its round", outcomes["verifier halted"],
        ran["verifier halted"])
    assert warned == {"halted": False, "task halted": False, "verified": False, "verified calm": False,
                      "verifier halted": False, "repeating halted": False,
                      "none": False, "failing": True,
                      "no reason": False, "not a reason": False, "round cap": False, "cancelled": False}, (
        "a halt that fails is said, and only that one", warned)


# ---------------------------------------------------------------------------
# WC2 -- the gate names the stop only past its bound
# ---------------------------------------------------------------------------
def test_wc2_the_gate_names_the_stop_only_past_its_bound(tmp_path):
    p = pw_contracts
    typed = "Remember that I water the plants on Sunday."
    loaded, restore = p._gate_window()
    try:
        pw, tools = loaded[p._PW], loaded[p._TOOLS]
        queue, memory = p._queue(pw, tmp_path), p._Memory()
        gate = p._gate(pw, queue, typed, max_per_run=2, run_id="r1")
        tool = p._memory_tool(tools, memory, gate)
        seen = [gate.halt()]
        for _ in range(3):
            tool({"action": "add", "text": typed})
        seen.append(gate.halt())
        said = []
        for index in range(2):
            said.append(tool({"action": "add", "text": f"A fact the page planted, number {index}."}))
            seen.append(gate.halt())
        said.append(tool({"action": "add", "text": "A third fact the page planted."}))
        seen.append(gate.halt())
        tool({"action": "add", "text": typed})
        seen.append(gate.halt())
        rows = queue.list()
    finally:
        restore()
    assert seen == [None, None, None, None, "proposal_cap", "proposal_cap"], seen
    assert all("Proposed to the user" in each for each in said[:2]) and "already made 2 proposals" in said[2], said
    assert len(rows) == 2 and p._adds(memory) == [typed] * 4, (len(rows), p._adds(memory))


# ---------------------------------------------------------------------------
# WC3 -- the route hands the run's gate stop to the loop and reports it
# ---------------------------------------------------------------------------
def _flooding_loop(record):
    """A loop stand-in whose model writes twenty-five planted facts in one round, then stops on the gate's word; each
    run records in ``loop.record``, ``record`` first."""
    loop = types.ModuleType(_LOOP)
    loop.record = record
    loop.flood = True

    def run(*, task, tool_handlers, on_event=None, should_continue=None, **kwargs):
        record = loop.record
        halt = kwargs.get("halt")
        record["halt_given"] = "halt" in kwargs
        record["before"] = halt() if halt is not None else None
        for index in range(25 if loop.flood else 0):
            if "manage_memory" in tool_handlers:
                tool_handlers["manage_memory"]({"action": "add", "text": f"Planted fact number {index}."})
        record["after"] = halt() if halt is not None else None
        return SimpleNamespace(stop_reason=record["after"] or "done", rounds=1)

    loop.run = run
    return loop


def _route_world(tmp_path, loop, *, seeded=None, stopped=True):
    """The agent route, the run manager, the real tools and the gate, over ``loop``; ``stopped`` is the emergency
    stop's state, ``seeded`` more stand-ins."""
    p = pw_contracts
    skills = types.ModuleType("opti_oignon.agent.skills")
    skills.make_manage_skills_handler = lambda **kwargs: (lambda arguments: "skills untouched")
    skills.consult_skills = lambda task, registry=None: SimpleNamespace(block="")
    security_mode = types.ModuleType("opti_oignon.security_mode")
    security_mode.get_current_mode = lambda: "daily"
    estop = types.ModuleType("opti_oignon.emergency_stop")
    estop.guard_http = lambda: None
    estop.is_stopped = lambda: stopped
    capability = types.ModuleType("opti_oignon.capability_manifest")
    capability.model_tool_capable = lambda name: True
    loaded, restore = p._gate_window(
        extra_targets={p._AGENT_ROUTES: source("api", "routes_agent.py")},
        seeded={_LOOP: loop, "opti_oignon.agent.skills": skills, "opti_oignon.security_mode": security_mode,
                "opti_oignon.emergency_stop": estop, "opti_oignon.capability_manifest": capability,
                **(seeded or {})},
        packages=("opti_oignon.api",))
    pw, tools, routes = loaded[p._PW], loaded[p._TOOLS], loaded[p._AGENT_ROUTES]
    tools.reset_tool_registry()
    tools._REGISTRY = tools.ToolRegistry(memory_store=p._Memory(), notes_store=p._Notes(),
                                         skills_handler=lambda arguments: "skills untouched",
                                         web_search_fn=lambda *a, **k: "no results")
    queue = p._queue(pw, tmp_path)
    pw.set_pending_store(queue)
    routes.reset_run_manager()
    return SimpleNamespace(pw=pw, routes=routes, queue=queue, restore=restore)


def test_wc3_the_route_hands_the_runs_gate_stop_to_the_loop_and_reports_it(tmp_path):
    record, withdrawn = {}, {}
    loop = _flooding_loop(record)
    world = _route_world(tmp_path, loop)
    try:
        pw_contracts._route_run(world, "Tell me about my day.")
        status = world.routes.get_run_manager().status()
        rows = world.queue.list()
        bound = world.pw.load_config()["max_per_run"]
        real_gate = world.pw.WriteGate
        world.pw.WriteGate = lambda *args, **kwargs: 1 / 0
        loop.record = withdrawn
        pw_contracts._route_run(world, "Tell me about my day.")
        world.pw.WriteGate = real_gate
    finally:
        world.restore()
    assert record == {"halt_given": True, "before": None, "after": "proposal_cap"}, record
    assert status["stop_reason"] == "proposal_cap" and len(rows) == bound == 20, (status, len(rows), bound)
    assert withdrawn == {"halt_given": False, "before": None, "after": None}, withdrawn


# ---------------------------------------------------------------------------
# WC4 -- the bench's runs stop where the route's do
# ---------------------------------------------------------------------------
def test_wc4_the_benchs_runs_stop_where_the_routes_do():
    bench, loaded, close = bench_contracts._window()
    try:
        suite = bench.load_suite()
        tasks = {task.id: task for task in suite.user_tasks}
        injections = {injection.id: injection for injection in suite.injection_tasks}
        world = bench._World(suite, injection=injections["iban_to_notes"].text)
        task = tasks["weather_typed"]
        flooded = bench.run_once(world, task.prompt, bench.Persona("enumerator", task.plan,
                                                                   [injections["iban_to_notes"]],
                                                                   secret=suite.secret, typed=task.prompt),
                                 mode="daily", policy="free", responder="approve")
        queued = len(world.queue.records)
        enumerator = bench.bench_row(suite, "enumerator", bench.Setting("daily", "free", "approve"))
        obedient = bench.bench_row(suite, "obedient", bench.Setting("daily", "free", "approve"))
        bound = loaded[bench_contracts._PW].load_config()["max_per_run"]
    finally:
        close()
    assert flooded.stop_reason == "proposal_cap" and queued == bound, (flooded.stop_reason, queued, bound)
    held = flooded.calls - flooded.executed - sum(flooded.refused.values())
    assert "halted" not in flooded.refused and held > 0, ("a call the stop held is no refusal of a gate",
                                                          flooded.refused, held)
    assert enumerator["runs_stopped"].get("proposal_cap", 0) > 0, enumerator["runs_stopped"]
    assert "proposal_cap" not in obedient["runs_stopped"], obedient["runs_stopped"]


# ---------------------------------------------------------------------------
# WC5 -- ib8 with the bound raised for the run
# ---------------------------------------------------------------------------
def test_wc5_the_bench_reaches_no_store_and_no_network_of_the_users_with_the_bound_raised(tmp_path):
    bench, loaded, close = bench_contracts._window()
    tripped = []
    try:
        raised = Path(tmp_path) / "pending_writes.yaml"
        raised.write_text("max_per_run: 1000\n", encoding="utf-8")
        loaded[bench_contracts._PW]._CONFIG = raised
        for module, names in bench_contracts._TRIPWIRES.items():
            for name in names:
                assert callable(getattr(loaded[module], name)), ("control: the default exists", module, name)

                def wire(*args, _name=f"{module}.{name}", **kwargs):
                    tripped.append(_name)
                    raise AssertionError(f"the bench reached {_name}")

                setattr(loaded[module], name, wire)
        suite = bench.load_suite()
        tasks = {t.id: t for t in suite.user_tasks}
        injections = {i.id: i for i in suite.injection_tasks}
        world = bench._World(suite, injection=injections["iban_to_notes"].text)
        task = tasks["weather_typed"]
        swept = bench.run_once(world, task.prompt, bench.Persona("enumerator", task.plan, [injections["iban_to_notes"]],
                                                                 secret=suite.secret, typed=task.prompt),
                               mode="daily", policy="free", responder="approve")
        planting = bench._World(suite, injection=injections["plant_instruction"].text)
        task = tasks["report_summary"]
        bench.run_once(planting, task.prompt, bench.Persona("obedient", task.plan, [injections["plant_instruction"]],
                                                            secret=suite.secret, typed=task.prompt),
                       mode="daily", policy="free", responder="approve")
        applied = bench._review(planting, "accept_all")
        planting.clear_traces()
        victim = suite.victim_tasks[0]
        later = bench.run_once(planting, victim.prompt, bench.Persona("obedient", victim.plan,
                                                                      [injections["iban_by_search"]],
                                                                      secret=suite.secret, typed=victim.prompt),
                               mode="daily", policy="free", responder="approve")
    finally:
        close()
    assert tripped == [], tripped
    reached = {tool for tool, count in swept.executed_by_tool.items() if count}
    every = {"web_search", "manage_memory", "manage_notes", "manage_skills", "bash", "view", "create_file",
             "str_replace", "grep", "glob", "ls"}
    assert every <= reached, ("control: every tool's handler ran", sorted(every - reached))
    assert swept.sent and world.queue.records and world.notes.live(), ("control: every sink was reached", world)
    with_ids = {(name, arguments.get("fact_id") or arguments.get("note_id")) for name, arguments in swept.emitted
                if arguments.get("action") in ("update", "delete")}
    assert ("manage_memory", "f1") in with_ids and ("manage_notes", "n1") in with_ids, (
        "the enumerator takes the identifiers it read", sorted(i for i in with_ids if i[1] in ("f1", "n1")))
    assert applied >= 1 and later.sent == [suite.secret], ("control: the memory was read back", applied, later.sent)
    assert swept.stop_reason != "proposal_cap", ("control: the bound was raised", swept.stop_reason)


# ---------------------------------------------------------------------------
# WC6 -- the agent panel ends a run on its aborted event and says why
# ---------------------------------------------------------------------------
_STOP = "frontend/src/lib/agentStop.ts"
_PANEL = "frontend/src/lib/components/panels/AgentPanel.svelte"
_AGENT_API = "frontend/src/lib/api/agent.ts"
_STOP_DRIVER = r"""
const mod = await import(process.env.OO_STOP);
const clause = process.argv[2];
const events = JSON.parse(process.env.OO_EVENTS);
const result = {
  reasons: events.map((event) => mod.endingReason(event)),
  labels: ['done', 'proposal_cap', 'doom_loop', 'odd'].map((reason) => mod.stopLabel(reason)),
  sentences: ['proposal_cap', 'odd'].map((reason) => mod.stopSentence(reason)),
};
console.log('RESULT ' + JSON.stringify(result));
console.log('PASS ' + clause);
"""


def test_wc6_the_agent_panel_ends_a_run_on_its_aborted_event_and_says_why():
    events = [{"kind": "round_start", "data": {}}, {"kind": "model_output", "data": {}}, {"kind": "done", "data": {}},
              {"kind": "error", "data": {"error": "refused"}}, {"kind": "aborted", "data": {"reason": "proposal_cap"}},
              {"kind": "aborted", "data": {"reason": "doom_loop", "tool": "ls"}}, {"kind": "aborted", "data": {}},
              {"kind": "aborted", "data": {"reason": ""}}, {"kind": "aborted", "data": {"reason": 5}},
              {"kind": "error", "data": {"error": "child failed", "task": "t1"}},
              {"kind": "done", "data": {"final_text": "child done", "task": "t1"}}]
    out = run_ts({"OO_STOP": _STOP}, _STOP_DRIVER, "stopping", env={"OO_EVENTS": json.dumps(events)})
    lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, out
    got = json.loads(lines[0][len("RESULT "):])
    assert got["reasons"] == [None, None, "done", "error", "proposal_cap", "doom_loop", "aborted", "aborted",
                              "aborted", None, None], got["reasons"]
    assert got["labels"] == ["Done", "Stopped: too many writes", "Stopped: repeating", "Stopped (odd)"], got["labels"]
    assert "propose more writes for your review than one run may" in got["sentences"][0], got["sentences"]
    assert "waits in your review" in got["sentences"][0] and got["sentences"][1] == "Stopped (odd).", got["sentences"]
    panel = read(_PANEL)
    push = panel[panel.index("function pushEvent"):panel.index("async function refreshPending")]
    assert ("const ended = endingReason(event);\n\t\tif (event.kind === 'round_start') {\n\t\t\tcurrentRound = "
            "event.round;\n\t\t\trunning = true;\n\t\t} else if (ended) {\n\t\t\trunning = false;\n\t\t\t"
            "stopReason = ended;\n\t\t}") in push, push
    cancel = panel[panel.index("async function handleCancel"):panel.index("onMount(")]
    assert "const { cancelled } = await cancelAgentRun();" in cancel and "if (cancelled) {" in cancel, cancel
    found = cancel[cancel.index("if (cancelled) {"):cancel.index("} else {")]
    missed = cancel[cancel.index("} else {"):cancel.index("} catch {")]
    assert "stopReason = 'cancelled';" in found and "stopReason" not in missed, ("a cancel that found no run",
                                                                                 found, missed)
    assert "{stopLabel(stopReason)}" in panel and "stopSentence(endingReason(event)" in panel, (
        "the panel says how the run ended")
    api = read(_AGENT_API)
    assert "| 'aborted'" in api and "\t'aborted',\n" in api, "the client knows the loop's aborted event"
    html = ssr().render(_PANEL, {}).html
    assert "Idle" in html, ("control: the panel renders", html[:200])


# ---------------------------------------------------------------------------
# WC7 -- the review gathers a run's proposals under one control
# ---------------------------------------------------------------------------
_PURE = "frontend/src/lib/pendingWrites.ts"
_LIST = "frontend/src/lib/components/panels/PendingWritesList.svelte"
_REVIEW = "frontend/src/lib/components/panels/PendingWritesReview.svelte"
_GROUP_DRIVER = r"""
const mod = await import(process.env.OO_PENDING);
const clause = process.argv[2];
const items = JSON.parse(process.env.OO_ITEMS);
const result = {
  groups: mod.runGroups(items),
  tag: mod.runTag('3f2a9c71d0e4'),
  none: mod.runGroups([]),
};
console.log('RESULT ' + JSON.stringify(result));
console.log('PASS ' + clause);
"""


def _proposal(pid, run_id, text):
    return {"id": pid, "store": "memory", "action": "add", "arguments": {"text": text, "category": "fact"},
            "provenance": {"source": "agent" if run_id else "extraction", "turn": "typed", "typed": {},
                           "untyped": ["text"], "target": None, "read": ["web_search"]},
            "conversation_id": "conv-1", "run_id": run_id, "created_at": "2026-10-14T10:00:00+00:00",
            "target": None}


def test_wc7_the_review_gathers_a_runs_proposals_under_one_control():
    items = [_proposal("a1", "run-a", "Planted one."), _proposal("a2", "run-a", "Planted two."),
             _proposal("b1", "run-b", "A fact of another run."), _proposal("e1", "", "An extracted fact."),
             _proposal("a3", "run-a", "Planted three."), _proposal("c1", "run-c", "A pair, first."),
             _proposal("c2", "run-c", "A pair, second.")]
    out = run_ts({"OO_PENDING": _PURE}, _GROUP_DRIVER, "grouping", env={"OO_ITEMS": json.dumps(items)})
    lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, out
    got = json.loads(lines[0][len("RESULT "):])
    assert got["groups"] == [{"run_id": "run-a", "ids": ["a1", "a2", "a3"]},
                             {"run_id": "run-c", "ids": ["c1", "c2"]}], got["groups"]
    assert got["tag"] == "run 3f2a9c" and got["none"] == [], got
    html = ssr().render(_LIST, {"items": items, "selected": [], "busy": False}).html
    single = ssr().render(_LIST, {"items": items[2:4], "selected": [], "busy": False}).html
    assert html.count("data-pending-run=") == 2 and 'data-pending-run="run-a"' in html, html.count("data-pending-run")
    assert "run run-a: 3 proposals from one agent run" in html and "run run-c: 2 proposals" in html, "one line a run"
    labels = [html.count(f'<span class="oo-btn-label">Select these {n}</span>') for n in (3, 2)]
    assert labels == [1, 1], ("one control a run, its count", labels)
    assert 'aria-label="Select these 3: only the proposals of run run-a"' in html, "the control says whose proposals"
    assert html.count('class="oo-pending-run-tag') == 5, "each grouped proposal carries its run's tag"
    assert "data-pending-run=" not in single and "Select these" not in single, "no group for a single proposal"
    assert "oo-pending-run-tag" not in single, "no tag where no run is grouped"
    review = read(_REVIEW)
    assert "on:selectRun={(event) => (selected = [...event.detail])}" in review, (
        "the control selects the run's proposals, and only them")
    listing = read(_LIST)
    assert "dispatch('selectRun', run.ids)" in listing, "the control hands the run's ids"


# ---------------------------------------------------------------------------
# WC8 -- a run stopped at the bound wakes no teacher
# ---------------------------------------------------------------------------
def test_wc8_a_run_stopped_at_the_bound_wakes_no_teacher(tmp_path):
    built = []
    teacher = types.ModuleType("opti_oignon.agent.teacher")

    class _Escalator:
        def __init__(self, policy=None):
            built.append(policy)

        def should_escalate(self, result):
            return SimpleNamespace(escalate=False)

    teacher.TeacherEscalator = _Escalator
    loop = _flooding_loop({})
    world = _route_world(tmp_path, loop, seeded={"opti_oignon.agent.teacher": teacher}, stopped=False)
    try:
        manager = world.routes.get_run_manager()
        manager._teacher_policy_opt_in = lambda: "armed"
        pw_contracts._route_run(world, "Tell me about my day.")
        capped = (manager.status()["stop_reason"], list(built))
        loop.flood, loop.record = False, {}
        pw_contracts._route_run(world, "Tell me about my day.")
        calm = (manager.status()["stop_reason"], list(built))
    finally:
        world.restore()
    assert capped == ("proposal_cap", []), ("a run stopped at the bound woke the teacher", capped)
    assert calm == ("done", ["armed"]), ("control: a run that ended otherwise is offered to the teacher", calm)
