#!/usr/bin/env python3
"""Contracts for the network policy Daily ships with: a value the user did not type waits for their yes.

``opti_oignon/config/provenance.yaml`` says what a network call that carries
an unendorsed argument does in Daily. It ships ``ask_unendorsed``: the call
is put to the user, who sees each value whole with its label, and goes out
only on their yes; a value the user typed, whole, goes out unasked. A
missing file reads the same. ``free`` and ``refuse_unendorsed`` stay the
user's choice, read as written.

  * ND1 -- THE SHIPPED POLICY ASKS, AND A MISSING FILE READS THE SAME: the
    package's policy file holds the one setting, ``ask_unendorsed``; Daily
    reads it so and Bulbe refuses; a missing file reads ``ask_unendorsed``,
    and one that says ``free`` or ``refuse_unendorsed`` reads as it says.
  * ND2 -- UNDER THE SHIPPED POLICY, AN UNENDORSED NETWORK VALUE WAITS FOR
    THE USER: with the file as shipped, the chat's tool call and the agent's
    put a search for words the user did not type to the turn's way to ask,
    with each value's label and the call's class; nothing is sent on a no,
    the call runs on a yes, and a search for the words the user typed,
    whole, runs without a question.
  * ND3 -- A CALL SALVAGED FROM PROSE IS JUDGED LIKE ANY OTHER (pv8, with the
    policy file set to free for the run): a tool call rebuilt from the
    model's narration meets the turn's policy -- refused under a refusing
    turn, run under a free one.
  * ND4 -- A TOOL THE TABLE DOES NOT KNOW IS HELD AS NETWORK (pv12, with the
    policy file set to free for the run): Bulbe refuses it, a refusing Daily
    turn refuses its unendorsed call, a free one runs it.
  * ND5 -- THE CAPTURE ARM SAYS THE POLICY ITS LATER RUNS GOT: a later run
    is never looser than the machine's policy file, and the arm's report
    says the mode and the policy its later runs ran under, not only the one
    it asked for -- ``ask_unendorsed`` under the file as shipped, ``free``
    and ``refuse_unendorsed`` under files that say so -- and the policy took
    effect: the instruction kept without the filter is carried out where
    the call is let through, and nowhere under the refusing file.
  * ND6 -- EVERY ARM OF THE BENCH SAYS THE POLICY ITS RUNS GOT: under the
    file as shipped, a row, a memory row, a confinement row and a
    non-interference row asked under ``free`` each say they ran under
    ``ask_unendorsed``, and the floors keep their witnesses.
  * ND7 -- A QUESTION OF A CANCELLED RUN IS WITHDRAWN: a question the agent
    route's approval asks waits on the run's stop as well as the answer --
    the route hands its run manager's stop to the approval it starts the
    run with -- and once the run is cancelled the waiting question leaves
    the queue, withdrawn on the run's behalf (denied, by a queue that cannot
    withdraw), the call is denied at once rather than at the deadline, a
    yes that lands with the cancel runs nothing, and a run already stopped
    asks no one.

The gate is loaded through the shared isolation window of the gate's own
suite of contracts, the capture arm through its own suite's. Local-only.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_agent_run_mode_floor_contracts as route_contracts  # noqa: E402
import test_capture_arm_contracts as capture_contracts  # noqa: E402
import test_provenance_gate_contracts as gate_contracts  # noqa: E402

BUDGET_S = {
    "test_nd6_every_arm_of_the_bench_says_the_policy_its_runs_got": 20.0,
    "test_nd7_a_question_of_a_cancelled_run_is_withdrawn_and_a_later_yes_runs_nothing": 5.0,
    "test_nd1_the_shipped_policy_asks_and_a_missing_file_reads_the_same": 2.0,
    "test_nd2_under_the_shipped_policy_an_unendorsed_network_value_waits_for_the_user": 5.0,
    "test_nd3_a_tool_call_salvaged_from_the_models_prose_meets_the_turns_policy": 5.0,
    "test_nd4_a_tool_the_table_does_not_know_is_held_to_the_network_rules_of_the_turn": 5.0,
    "test_nd5_the_capture_arm_says_the_mode_and_the_policy_its_later_runs_got": 20.0,
}

_SHIPPED = gate_contracts._OO / "config" / "provenance.yaml"


# ---------------------------------------------------------------------------
# Contract ND1 -- the shipped policy asks, and a missing file reads the same
# ---------------------------------------------------------------------------
def test_nd1_the_shipped_policy_asks_and_a_missing_file_reads_the_same(tmp_path):
    g = gate_contracts
    loaded, restore = g._gate_window()
    try:
        prov = loaded[g._PROV]
        shipped_path = Path(prov.POLICY_FILE)
        shipped = (prov.read_network_policy(), prov.network_policy("daily"), prov.network_policy("bulbe"))
        absent = prov.read_network_policy(Path(tmp_path) / "absent.yaml")
        chosen = {}
        for name, text in (("free", g._FREE), ("refuse", g._REFUSE)):
            path = Path(tmp_path) / f"{name}.yaml"
            path.write_text(text, encoding="utf-8")
            chosen[name] = prov.read_network_policy(path)
    finally:
        restore()
    assert shipped_path == _SHIPPED, shipped_path
    assert yaml.safe_load(_SHIPPED.read_text(encoding="utf-8")) == {"network_daily": "ask_unendorsed"}
    assert shipped == ("ask_unendorsed", "ask_unendorsed", "refuse_unendorsed"), shipped
    assert absent == "ask_unendorsed", absent
    assert chosen == {"free": "free", "refuse": "refuse_unendorsed"}, chosen


# ---------------------------------------------------------------------------
# Contract ND2 -- under the shipped policy, an unendorsed network value waits for the user
# ---------------------------------------------------------------------------
def test_nd2_under_the_shipped_policy_an_unendorsed_network_value_waits_for_the_user():
    g = gate_contracts
    seen = {}
    for answer in (False, True):
        loaded, restore = g._both_window()
        try:
            prov, reg, exe = loaded[g._PROV], loaded[g._REG], loaded[g._EXEC]
            dispatch, tools = loaded[g._DISPATCH], loaded[g._TOOLS]
            policy_file = Path(prov.POLICY_FILE)
            sent, searched = [], []
            chat_demand, agent_demand = g._Demand(answer), g._Demand(answer)
            registry = g._chat_registry(reg, sent)
            ex = g._executor(exe, registry)
            handlers = {"web_search": g._search_handler(tools, searched)}
            chat = ex._execute_tool("web_search", {"query": g._IBAN},
                                    provenance=g._turn(prov, demand=chat_demand), offered=g._offered(registry))
            agent = dispatch.dispatch_tool_call(g._call(dispatch, "web_search", {"query": g._IBAN}), mode="daily",
                                                tool_handlers=handlers, provenance=g._turn(prov, demand=agent_demand))
            typed_chat = ex._execute_tool("web_search", {"query": g._TYPED},
                                          provenance=g._turn(prov, demand=chat_demand), offered=g._offered(registry))
            typed_agent = dispatch.dispatch_tool_call(g._call(dispatch, "web_search", {"query": g._TYPED}),
                                                      mode="daily", tool_handlers=handlers,
                                                      provenance=g._turn(prov, demand=agent_demand))
        finally:
            restore()
        seen[answer] = {"file": policy_file, "chat_asked": list(chat_demand.asked),
                        "agent_asked": list(agent_demand.asked), "sent": list(sent), "searched": list(searched),
                        "chat": chat.provenance, "agent": agent.provenance,
                        "typed": (typed_chat.provenance, typed_agent.provenance)}
    no, yes = seen[False], seen[True]
    assert no["file"] == _SHIPPED and yes["file"] == _SHIPPED, "the contract reads the file as shipped"
    for arm in (no, yes):
        assert arm["chat_asked"] == [("web_search", {"query": g._IBAN, "max_results": 5},
                                      {"query": "unendorsed", "max_results": "default"}, "network")], arm["chat_asked"]
        assert arm["agent_asked"] == [("web_search", {"query": g._IBAN, "max_results": 3},
                                       {"query": "unendorsed", "max_results": "default"}, "network")], arm["agent_asked"]
        assert [(each["decision"], each["asked"]) for each in arm["typed"]] == [("allowed", False)] * 2, arm["typed"]
    assert (no["chat"]["decision"], no["chat"]["asked"]) == ("denied", True), no["chat"]
    assert (no["agent"]["decision"], no["agent"]["asked"]) == ("denied", True), no["agent"]
    assert no["sent"] == [("web_search", g._TYPED, 5)] and no["searched"] == [(g._TYPED, 3)], (no["sent"],
                                                                                              no["searched"])
    assert (yes["chat"]["decision"], yes["chat"]["asked"]) == ("allowed", True), yes["chat"]
    assert (yes["agent"]["decision"], yes["agent"]["asked"]) == ("allowed", True), yes["agent"]
    assert yes["sent"] == [("web_search", g._IBAN, 5), ("web_search", g._TYPED, 5)], yes["sent"]
    assert yes["searched"] == [(g._IBAN, 3), (g._TYPED, 3)], yes["searched"]


# ---------------------------------------------------------------------------
# Contract ND3 -- a call salvaged from prose is judged like any other (pv8, the file set to free)
# ---------------------------------------------------------------------------
def test_nd3_a_tool_call_salvaged_from_the_models_prose_meets_the_turns_policy(tmp_path):
    g = gate_contracts
    outcomes = {}
    for policy in ("refuse_unendorsed", "free"):
        loaded, restore = g._chat_window()
        try:
            prov, reg, exe = loaded[g._PROV], loaded[g._REG], loaded[g._EXEC]
            g._policy(prov, tmp_path, g._FREE, name=f"salvage-{policy}.yaml")
            sent = []
            ex = g._executor(exe, g._chat_registry(reg, sent))
            g._scripted(ex, [])
            exe.transpile_intent = lambda candidate, message, names: [("web_search", {"query": g._IBAN})]
            ex._generate_final_response = lambda *args, **kwargs: "I will look up " + g._IBAN
            result = ex.execute_with_tools(message=g._TYPED, provenance=g._turn(prov, policy=policy))
        finally:
            restore()
        outcomes[policy] = (list(sent), [(c.success, (c.provenance or {}).get("decision")) for c in result.tool_calls])
    assert outcomes["free"] == ([("web_search", g._IBAN, 5)], [(True, "allowed")]), (
        f"witness: the salvage itself did not run: {outcomes['free']}")
    assert outcomes["refuse_unendorsed"] == ([], [(False, "unendorsed")]), outcomes["refuse_unendorsed"]


# ---------------------------------------------------------------------------
# Contract ND4 -- a tool the table does not know is held as network (pv12, the file set to free)
# ---------------------------------------------------------------------------
def test_nd4_a_tool_the_table_does_not_know_is_held_to_the_network_rules_of_the_turn(tmp_path):
    g = gate_contracts
    outcomes = {}
    for case, mode, policy in (("bulbe", "bulbe", None), ("daily refusing", "daily", "refuse_unendorsed"),
                               ("daily free", "daily", "free")):
        loaded, restore = g._chat_window(g._Machine(mode))
        try:
            prov, reg, exe = loaded[g._PROV], loaded[g._REG], loaded[g._EXEC]
            g._policy(prov, tmp_path, g._FREE, name=f"unknown-{case.replace(' ', '-')}.yaml")
            sent, hook = [], g._Hook(True)
            registry = g._chat_registry(reg, sent, extra=("fetch_page",))
            ex = g._executor(exe, registry)
            effect = prov.effect_of("fetch_page")
            result = ex._execute_tool("fetch_page", {"text": g._IBAN}, approval_fn=hook if mode == "bulbe" else None,
                                      provenance=g._turn(prov, policy=policy), offered=g._offered(registry))
        finally:
            restore()
        outcomes[case] = (effect, list(sent), (result.provenance or {}).get("decision"), len(hook.asked))
    assert outcomes == {"bulbe": ("network", [], "not_permitted", 0),
                        "daily refusing": ("network", [], "unendorsed", 0),
                        "daily free": ("network", [("fetch_page", g._IBAN)], "allowed", 0)}, outcomes


# ---------------------------------------------------------------------------
# Contract ND5 -- the capture arm says the policy its later runs got
# ---------------------------------------------------------------------------
def test_nd5_the_capture_arm_says_the_mode_and_the_policy_its_later_runs_got(tmp_path):
    c = capture_contracts
    reports, heads = {}, {}
    for case, text in (("shipped", None), ("free", "network_daily: free\n"),
                       ("refusing", "network_daily: refuse_unendorsed\n")):
        arm, loaded, close = c._window()
        try:
            prov = loaded[c._PROV]
            if text is None:
                prov.POLICY_FILE = _SHIPPED
            else:
                path = Path(tmp_path) / f"capture-{case}.yaml"
                path.write_text(text, encoding="utf-8")
                prov.POLICY_FILE = path
            reports[case] = arm.run_capture(loaded[c._BENCH].load_suite(), policy="free")
            heads[case] = arm._table(reports[case]).splitlines()[0]
        finally:
            close()
    ran = {case: (report["policy"], report["later_runs"]) for case, report in reports.items()}
    assert ran == {"shipped": ("free", {"mode": "daily", "policy": "ask_unendorsed"}),
                   "free": ("free", {"mode": "daily", "policy": "free"}),
                   "refusing": ("free", {"mode": "daily", "policy": "refuse_unendorsed"})}, ran
    assert heads["shipped"] == ("capture arm, suite injection_micro, later runs asked under free, run in daily "
                                "under ask_unendorsed"), heads["shipped"]
    carried = {case: sum(row["carried_out_later_without_the_filter"]["count"] for row in report["capture"])
               for case, report in reports.items()}
    assert carried["shipped"] > 0 and carried["free"] > 0 and carried["refusing"] == 0, (
        "the policy each later run got took effect", carried)


# ---------------------------------------------------------------------------
# Contract ND6 -- every arm of the bench says the policy its runs got
# ---------------------------------------------------------------------------
def test_nd6_every_arm_of_the_bench_says_the_policy_its_runs_got():
    b = capture_contracts
    arm, loaded, close = b._window()
    try:
        loaded[b._PROV].POLICY_FILE = _SHIPPED
        bench = loaded[b._BENCH]
        report = bench.run_bench(bench.load_suite(), personas=("obedient",),
                                 settings=bench.parse_settings("daily:free:approve"), memory_policies=("free",))
        table = bench._table(report)
    finally:
        close()
    asked_free = {"rows": [row["policy"] for row in report["rows"] if row["policy_asked"] == "free"],
                  "memory": [row["policy"] for row in report["memory"] if row["policy_asked"] == "free"],
                  "confinement": [row["policy"] for row in report["confinement"] if row["policy_asked"] == "free"],
                  "noninterference": [row["policy"] for row in report["noninterference"]
                                      if row["policy_asked"] == "free"]}
    assert asked_free == {"rows": ["ask_unendorsed"], "memory": ["ask_unendorsed", "ask_unendorsed"],
                          "confinement": ["ask_unendorsed"], "noninterference": ["ask_unendorsed"]}, asked_free
    strict = {row["policy"] for row in [*report["confinement"], *report["noninterference"]]
              if row["policy_asked"] == "refuse_unendorsed"}
    assert strict == {"refuse_unendorsed"}, strict
    memory_lines = [line for line in table.splitlines() if line.startswith("  memory free ")]
    assert len(memory_lines) == 2 and all(line.startswith(f"  memory {'free':<18} ran ask_unendorsed ")
                                          for line in memory_lines), memory_lines
    assert report["floors"]["network_values_confined_under_refusal"] is True, report["floors"]
    assert report["floors"]["network_trace_independent_of_the_secret"] is True, report["floors"]


# ---------------------------------------------------------------------------
# Contract ND7 -- a question of a cancelled run is withdrawn
# ---------------------------------------------------------------------------
class _Queue:
    """The approval queue as the agent route's approval sees it: one question at a time, answered by hand."""

    def __init__(self):
        import threading

        self.withdrawn = []
        self.answers = {}
        self.events = {}
        self._threading = threading

    def submit(self, conversation_id, tool_name, arguments, labels=None, effect=""):
        aid = f"q{len(self.events) + 1}"
        self.events[aid] = self._threading.Event()
        return aid, self.events[aid]

    def get_status(self, aid):
        return self.answers.get(aid, "pending")

    def answer(self, aid, status):
        self.answers[aid] = status
        self.events[aid].set()

    def withdraw(self, aid, reason="turn_stopped"):
        if self.answers.get(aid, "pending") != "pending":
            return False
        self.withdrawn.append((aid, reason))
        self.answers[aid] = "denied"
        self.events[aid].set()
        return True


class _Together(_Queue):
    """A queue whose person answers yes at the very moment the run is cancelled, before the wait looks."""

    def __init__(self, approved):
        super().__init__()
        self.approved = approved
        self.stop = None

    def submit(self, conversation_id, tool_name, arguments, labels=None, effect=""):
        aid, event = super().submit(conversation_id, tool_name, arguments, labels=labels, effect=effect)
        self.answers[aid] = self.approved
        event.set()
        self.stop.set()
        return aid, event

    def withdraw(self, aid, reason="turn_stopped"):
        self.withdrawn.append(aid)
        return False


class _Denying(_Queue):
    """A queue with no way to withdraw a question: it can only deny it."""

    withdraw = None

    def __init__(self):
        super().__init__()
        self.denied = []

    def deny(self, aid, user_id="admin"):
        self.denied.append((aid, user_id))
        self.answers[aid] = "denied"
        self.events[aid].set()
        return True


def test_nd7_a_question_of_a_cancelled_run_is_withdrawn_and_a_later_yes_runs_nothing():
    import threading
    import time

    g = gate_contracts
    loaded, restore = g._gate_window()
    try:
        allow = loaded[g._ALLOW]
        approved = allow.ApprovalStatus.APPROVED if hasattr(allow, "ApprovalStatus") else "approved"
        queue, stop = _Queue(), threading.Event()
        timer = threading.Timer(0.2, stop.set)
        timer.start()
        started = time.monotonic()
        cancelled = allow.request_approval("conv", "web_search", {"query": g._IBAN}, manager=queue, timeout=20.0,
                                           labels={"query": "unendorsed"}, effect="network", stop=stop)
        waited = time.monotonic() - started
        late = queue.withdraw("q1")
        asked_before = len(queue.events)
        after = allow.request_approval("conv", "web_search", {"query": g._IBAN}, manager=queue, timeout=0.3,
                                       stop=stop)
        asked_after = len(queue.events)
        together, both = _Together(approved), threading.Event()
        together.stop = both
        crossed = allow.request_approval("conv", "web_search", {"query": g._IBAN}, manager=together, timeout=20.0,
                                         stop=both)
        denying, alone = _Denying(), threading.Event()
        threading.Timer(0.1, alone.set).start()
        denied = allow.request_approval("conv", "web_search", {"query": g._IBAN}, manager=denying, timeout=20.0,
                                        stop=alone)
        live = threading.Event()
        threading.Timer(0.1, lambda: queue.answer("q2", approved)).start()
        yes = allow.request_approval("conv", "web_search", {"query": g._IBAN}, manager=queue, timeout=20.0,
                                     stop=live)
        plain = threading.Timer(0.1, lambda: queue.answer("q3", approved))
        plain.start()
        unstopped = allow.request_approval("conv", "web_search", {"query": g._IBAN}, manager=queue, timeout=20.0)
    finally:
        restore()
    route, manager, close = route_contracts._load("daily")
    handed = []
    try:
        manager.cancel_event = threading.Event()
        sys.modules["opti_oignon.agent.allowlists"].request_approval = (
            lambda *args, **kwargs: handed.append(kwargs.get("stop")) or False)
        request = SimpleNamespace(task="look up the weather", mode=None, model="any-model", conversation_id="c",
                                  verify=False, consult=False)
        route.router.handlers[("POST", "/run")](request)
        started = manager.start_calls[0][1]
        started["approval_fn"]("c", "web_search", {"query": g._IBAN}, labels={"query": "unendorsed"},
                               effect="network")
    finally:
        close()
    assert handed == [manager.cancel_event], ("the route's approval waits on its run's stop", handed)
    assert cancelled is False and waited < 5.0, (cancelled, waited)
    assert queue.withdrawn == [("q1", "run_cancelled")] and late is False, queue.withdrawn
    assert after is False and asked_after == asked_before, ("a stopped run asks no one", after, asked_after)
    assert crossed is False and together.withdrawn == ["q1"], ("a yes landing with the cancel runs nothing",
                                                                crossed, together.withdrawn)
    assert denied is False and denying.denied == [("q1", "run_cancelled")], ("a queue that cannot withdraw denies",
                                                                             denying.denied)
    assert yes is True and unstopped is True, ("control: a yes before any stop runs", yes, unstopped)
