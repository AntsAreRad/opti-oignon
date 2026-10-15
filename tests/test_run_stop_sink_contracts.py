#!/usr/bin/env python3
"""Contracts for the run's stop at the sink: a cancelled run runs nothing more, and the audit says so.

A person cancels an agent run. Until now the run read the cancel once a
round, before asking the model: the calls of the round in flight still ran,
those that need no approval at once, a task child ran its rounds to their
end, and a call a person approved just before the cancel still reached its
sink. The run's stop is now one: a cancel first, then the run's own halt
(the write gate's bound), read before each call of a round, again at each
call's sink once every gate has let it through, after the round, and at the
start of each round of a task child and of the verifier. A call stopped at
its sink does not run, and every approval it was given is withdrawn on the
run's behalf: the queue's audit keeps the person's approval and records the
withdrawal after it, as the chat's turn now does for an approval its stop
overtook.

  * SK1 -- WC1 word for word, but for the cancel: a run cancelled while the
    model answers its second round runs none of that round's calls -- each
    is not run, the model is told why -- and ends ``cancelled`` with one
    ``aborted`` event naming it (WC1 is deselected by name; it pinned the
    round's calls running after the cancel). A task child reads its run's
    cancel the same way, runs none of its calls and asks nothing more; a
    task its run stopped before its child's first round is said not run; a
    run cancelled during its last answer asks its verifier nothing; a
    cancel that cannot be read stops the run before it asks anything. The
    halt is now read at each call's sink as well, so WC1's stand-in halts,
    which count their reads, are set to the read that stops the same call:
    two reads for each call that runs, one more after its round.
  * SK2 -- in Bulbe, a call a person approved just before the cancel does
    not reach its sink: nothing runs, the round's next call is not even
    asked about, and the queue's audit says approved, then withdrawn on the
    run's behalf.
  * SK3 -- the queue withdraws an approved call once, after its approval,
    in both audits; a call denied, timed out or unknown is not withdrawn, a
    pending one is as before.
  * SK4 -- the chat's turn: an approval its stop overtook runs nothing and
    is withdrawn after it, in the queue's audit.
  * SK5 -- a call's approvals are its own: they are held only while the
    dispatch holds the call -- a thread that asks and never dispatches keeps
    nothing, and a withdrawal outside a dispatch starts no holding -- a
    thread withdraws only those its own call was given, and a later call's
    stop never withdraws an approval an earlier call ran with.
  * SK6 -- the run's stop is a cancel first, then the halt's reason; a
    cancel that cannot be read is a cancel; with no cancel to read it is the
    halt itself, unchanged.
  * SK7 -- a no is a no on both paths: the agent's approval answers no for a
    person's no and for a question left unanswered past its deadline, and
    holds nothing; the chat's hook answers no for a person's no with no stop
    in sight, and the audit names the person.

Local-only (the public distribution ships no tests). The loop, the dispatch
and the bench are loaded through the bench suite's shared isolation window,
the chat's route through the turn suite's, the queue from its file.
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_injection_bench_contracts as bench_contracts  # noqa: E402
import test_turn_stop_reach_contracts as turn_contracts  # noqa: E402
import test_write_cap_stop_contracts as wc  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_sk1_a_round_cancelled_while_the_model_answers_runs_none_of_its_calls": 5.0,
    "test_sk2_a_call_approved_just_before_the_cancel_does_not_reach_its_sink_and_the_audit_says_so": 5.0,
    "test_sk3_the_queue_withdraws_an_approved_call_once_after_its_approval_and_nothing_else": 2.0,
    "test_sk4_an_approval_the_chats_stop_overtook_is_withdrawn_after_it": 5.0,
    "test_sk5_a_calls_approvals_are_its_own": 5.0,
    "test_sk6_the_runs_stop_is_a_cancel_first_then_the_halt": 2.0,
    "test_sk7_a_no_is_a_no_on_both_paths": 5.0,
}

_LOOP = "opti_oignon.agent.loop"
_ALLOW = "opti_oignon.agent.allowlists"
_DISPATCH = "opti_oignon.agent.dispatch"
_APPROVAL = "opti_oignon.tool_call_approval"
_SIGNED = "opti_oignon.signed_audit_log"


def _queue_window(signed=None):
    """The approval queue from its file; ``signed`` stands in for the signed audit log when given."""
    import types

    seeded = {}
    if signed is not None:
        module = types.ModuleType(_SIGNED)
        module.chain_log = lambda **kwargs: signed.append(kwargs)
        seeded[_SIGNED] = module
    loaded, restore = isolate(targets={_APPROVAL: source("tool_call_approval.py")}, seeded=seeded, packages=())
    return loaded[_APPROVAL], restore


def _approving(approval):
    """A queue whose person says yes the moment a call is put to them."""

    class Approving(approval.ToolCallApprovalManager):
        def submit(self, *args, **kwargs):
            approval_id, event = super().submit(*args, **kwargs)
            self.approve(approval_id, "user")
            return approval_id, event

    queue = Approving()
    queue._reaper_active = True
    return queue


def _trail(queue, approval_id):
    """One call's decisions, oldest first: the queue's audit lists the newest first."""
    return [(entry["status"], entry["resolved_by"]) for entry in reversed(queue.audit_log(500))
            if entry["approval_id"] == approval_id]


# ---------------------------------------------------------------------------
# SK1 -- WC1, read again for a cancel
# ---------------------------------------------------------------------------
# A round that launches a task whose child is answered two commands.
_A_TASK_THAT_RUNS = ([("task", {"description": "run them", "prompt": "Run the two commands."})],
                     [("bash", {"command": "echo child one"}), ("bash", {"command": "echo child two"})])

def test_sk1_a_round_cancelled_while_the_model_answers_runs_none_of_its_calls(caplog):
    bench, loaded, close = bench_contracts._window()
    outcomes, warned, said, ran, asked_doom = {}, {}, {}, {}, []
    try:
        loop = loaded[_LOOP]
        suite = bench.parse_suite(bench_contracts._mini(), name="mini")
        children, current, child = {}, [None], loop._run_task_child

        def counted_child(*args, **kwargs):
            answer = child(*args, **kwargs)
            children.setdefault(current[0], []).append(answer)
            return answer

        loop._run_task_child = counted_child
        cases = (("halted", wc._THREE_AT_ONCE, {"halt": wc._halting(5)}),
                 ("task halted", wc._WITH_A_TASK, {"halt": wc._halting(3)}),
                 ("verified", wc._ONE_A_ROUND, "verify"), ("verified calm", wc._ONE_A_ROUND, "verify calm"),
                 ("verifier halted", wc._ONE_A_ROUND, "verify calls"),
                 ("repeating halted", wc._SAME_THRICE, "repeat"),
                 ("none", wc._ONE_A_ROUND, {}),
                 ("failing", wc._ONE_A_ROUND, {"halt": wc._raising}),
                 ("no reason", wc._ONE_A_ROUND, {"halt": wc._halting(1, reason="")}),
                 ("not a reason", wc._ONE_A_ROUND, {"halt": wc._halting(1, reason=True)}),
                 ("round cap", wc._ONE_A_ROUND, {"max_rounds": 2}), ("cancelled", wc._ONE_A_ROUND, "cancel"),
                 ("cancelled, three at once", wc._THREE_AT_ONCE, "cancel"),
                 ("cancelled in a task", _A_TASK_THAT_RUNS, "cancel"),
                 ("cancelled before the verifier", wc._ONE_A_ROUND, "cancel before verifying"),
                 ("halted before the task's first round", wc._WITH_A_TASK, {"halt": wc._halting(4)}),
                 ("a cancel that cannot be read", wc._ONE_A_ROUND, {"should_continue": wc._raising}))
        for case, rounds, extra in cases:
            current[0] = case
            world = bench._World(suite)
            tool_set = bench._bench_registry(world).build("daily")
            checks = ([("bash", {"command": "echo one"}), ("bash", {"command": "echo two"})],
                      [("bash", {"command": "echo three"})])
            client, events = wc._Reader(rounds, checks if extra == "verify calls" else ()), []
            if extra == "cancel":
                kwargs = {"should_continue": lambda client=client: client.asked < 2}
            elif extra == "cancel before verifying":
                kwargs = {"verify": True, "should_continue": lambda client=client: client.asked < 5}
            elif extra == "verify":
                kwargs = {"verify": True, "halt": wc._after_answer(client)}
            elif extra == "verify calm":
                kwargs = {"verify": True, "halt": wc._halting(10 ** 6)}
            elif extra == "verify calls":
                kwargs = {"verify": True, "halt": wc._after_answer(client)}
            elif extra == "repeat":
                kwargs = {"halt": wc._halting(9), "approval_fn": lambda *args, **kw: asked_doom.append(args) and False}
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
            said[case] += [(event.data["tool_name"], event.data.get("observation"), event.data.get("mode"))
                           for event in events if event.kind == "tool_result" and event.data.get("task")
                           and event.data.get("reason") == "halted"]
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
    assert outcomes["cancelled"] == ("cancelled", 2, 2, [(2, {"reason": "cancelled"})], 1, 1,
                                     "Reading, step 2."), outcomes["cancelled"]
    assert said["cancelled"] == [("view", "Not run: the run is stopping (cancelled).", "daily")], (
        "the call of the round the cancel came in is not run, and the model is told why", said["cancelled"])
    assert outcomes["cancelled, three at once"] == ("cancelled", 2, 2, [(2, {"reason": "cancelled"})], 3, 1,
                                                    "Reading, step 2."), outcomes["cancelled, three at once"]
    assert ran["cancelled in a task"] == [] and outcomes["cancelled in a task"][:3] == ("cancelled", 1, 2), (
        "a task child reads its run's cancel: none of its calls runs, and it asks nothing more",
        ran["cancelled in a task"], outcomes["cancelled in a task"])
    assert said["cancelled in a task"] == [("bash", "Not run: the run is stopping (cancelled).", "daily")] * 2, (
        said["cancelled in a task"])
    assert outcomes["cancelled before the verifier"][:4] == ("cancelled", 5, 5, [(5, {"reason": "cancelled"})]), (
        "a run cancelled during its last answer asks its verifier nothing", outcomes["cancelled before the verifier"])
    assert outcomes["halted before the task's first round"][:6] == (
        "proposal_cap", 1, 1, [(1, {"reason": "proposal_cap"})], 1, 1), outcomes["halted before the task's first round"]
    assert said["halted before the task's first round"] == [
        ("task", "Not run: the run is stopping (proposal_cap).", "daily")], (
        "a task its run stopped before its child's first round is said not run", said)
    assert children.get("halted before the task's first round") == [("", 0)], (
        "the child read the stop before its first round and used none", children)
    assert outcomes["a cancel that cannot be read"] == ("cancelled", 0, 0, [(0, {"reason": "cancelled"})], 0, 0, ""), (
        "a cancel that cannot be read stops the run before it asks anything", outcomes["a cancel that cannot be read"])
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
                      "no reason": False, "not a reason": False, "round cap": False, "cancelled": False,
                      "cancelled, three at once": False, "cancelled in a task": False,
                      "cancelled before the verifier": False, "halted before the task's first round": False,
                      "a cancel that cannot be read": False}, ("a halt that fails is said, and only that one", warned)


# ---------------------------------------------------------------------------
# SK2 -- a call approved just before the cancel
# ---------------------------------------------------------------------------
def test_sk2_a_call_approved_just_before_the_cancel_does_not_reach_its_sink_and_the_audit_says_so():
    approval, restore_queue = _queue_window()
    bench, loaded, close = bench_contracts._window("bulbe")
    try:
        loop, allow = loaded[_LOOP], loaded[_ALLOW]
        suite = bench.parse_suite(bench_contracts._mini(), name="mini")
        outcomes = {}
        for case in ("cancelled after the yes", "never cancelled"):
            queue, cancel, asked = _approving(approval), threading.Event(), []

            def approve(conversation_id, tool_name, arguments, *, queue=queue, cancel=cancel, case=case, **kwargs):
                asked.append(tool_name)
                ok = allow.request_approval(conversation_id, tool_name, arguments, manager=queue, timeout=5.0,
                                            stop=cancel)
                if case == "cancelled after the yes":
                    cancel.set()
                return ok

            world = bench._World(suite)
            tool_set = bench._bench_registry(world).build("bulbe")
            client, events = wc._Reader([[("bash", {"command": "echo one"}), ("bash", {"command": "echo two"})]]), []
            result = loop.run("run the two commands", model_client=client, sandbox=world.sandbox, mode="bulbe",
                              system_prompt=tool_set.system_prompt_section(), tools=tool_set.native_tools(),
                              tool_handlers=dict(tool_set.tool_handlers), include_memory=False,
                              on_event=events.append, approval_fn=approve,
                              should_continue=lambda cancel=cancel: not cancel.is_set())
            approvals = [entry["approval_id"] for entry in queue.audit_log(500) if entry["status"] == "approved"]
            outcomes[case] = {
                "stop": result.stop_reason,
                "ran": list(world.commands),
                "results": [(r.tool_name, r.executed, r.reason, r.observation) for r in result.tool_results],
                "asked": list(asked),
                "trails": [_trail(queue, approval_id) for approval_id in approvals],
                "aborted": [dict(event.data) for event in events if event.kind == "aborted"],
            }
    finally:
        close()
        restore_queue()
    calm = outcomes["never cancelled"]
    assert calm["stop"] == "done" and len(calm["ran"]) == 2 and calm["asked"] == ["bash", "bash"], (
        "control: in Bulbe each command is asked about and runs on the yes", calm)
    assert calm["trails"] == [[("approved", "user")], [("approved", "user")]], calm["trails"]
    cut = outcomes["cancelled after the yes"]
    assert cut["ran"] == [], f"nothing ran: {cut['ran']}"
    assert cut["results"] == [
        ("bash", False, "halted", "Not run: the run is stopping (cancelled)."),
        ("bash", False, "halted", "Not run: the run is stopping (cancelled)."),
    ], cut["results"]
    assert cut["asked"] == ["bash"], "the round's next call is not even asked about"
    assert cut["trails"] == [[("approved", "user"), ("denied", "run_cancelled")]], (
        "the audit keeps the yes and says the call was withdrawn on the run's behalf", cut["trails"])
    assert cut["stop"] == "cancelled" and cut["aborted"] == [{"reason": "cancelled"}], (cut["stop"], cut["aborted"])


# ---------------------------------------------------------------------------
# SK3 -- the queue withdraws an approved call once
# ---------------------------------------------------------------------------
def test_sk3_the_queue_withdraws_an_approved_call_once_after_its_approval_and_nothing_else():
    signed = []
    approval, restore = _queue_window(signed)
    try:
        queue = approval.ToolCallApprovalManager()
        queue._reaper_active = True
        given, _event = queue.submit("conv", "bash", {"command": "echo one"})
        assert queue.approve(given, "user") is True
        signed.clear()
        assert queue.withdraw(given, "run_cancelled") is True
        assert _trail(queue, given) == [("approved", "user"), ("denied", "run_cancelled")]
        assert queue.get_status(given) == approval.ApprovalStatus.DENIED, "its status is the withdrawal"
        assert [(entry["event_type"], entry["resolved_by"], entry["approval_id"]) for entry in signed] == [
            ("tool_call_denied", "run_cancelled", given)], f"the signed audit says so too: {signed}"
        assert queue.withdraw(given, "run_cancelled") is False and len(_trail(queue, given)) == 2, "once"
        refused, _ = queue.submit("conv", "bash", {"command": "echo two"})
        queue.deny(refused, "user")
        timed, _ = queue.submit("conv", "bash", {"command": "echo three"})
        with queue._lock:
            queue._resolve(timed, approval.ApprovalStatus.TIMEOUT, "timeout")
        signed.clear()
        for other in (refused, timed, "no-such-call"):
            assert queue.withdraw(other, "run_cancelled") is False, other
        assert _trail(queue, refused) == [("denied", "user")] and _trail(queue, timed) == [("timeout", "timeout")]
        assert signed == [], "nothing is recorded for a call that was not approved"
        waiting, event = queue.submit("conv", "bash", {"command": "echo four"})
        assert queue.withdraw(waiting, "turn_stopped") is True and event.is_set(), "a pending call, as before"
        assert _trail(queue, waiting) == [("denied", "turn_stopped")] and queue.pending() == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# SK4 -- the chat's turn
# ---------------------------------------------------------------------------
def test_sk4_an_approval_the_chats_stop_overtook_is_withdrawn_after_it():
    loaded, rc, _schemas, close = turn_contracts._load_routes_only(
        extra_targets={_APPROVAL: source("tool_call_approval.py")})
    try:
        approval = loaded[_APPROVAL]
        trails, allowed = {}, {}
        for case in ("approved", "approved, then stopped"):
            queue = approval.ToolCallApprovalManager()
            queue._reaper_active = True
            turn = rc.ChatTurn("conv-s")
            said = []

            def emit(event, queue=queue, turn=turn, case=case, said=said):
                said.append(event)
                if event[0] != "tool_call_pending":
                    return
                queue.approve(event[1]["approval_id"], "user")
                if case == "approved, then stopped":
                    turn.stop.set()

            hook = rc._make_approval_hook(queue, turn, "conv-s", emit, 30)
            allowed[case] = hook("echo", {})
            (asked,) = [event[1]["approval_id"] for event in said if event[0] == "tool_call_pending"]
            trails[case] = _trail(queue, asked)
    finally:
        close()
    assert allowed == {"approved": True, "approved, then stopped": False}, allowed
    assert trails["approved"] == [("approved", "user")], "control: a yes the stop did not overtake stays a yes"
    assert trails["approved, then stopped"] == [("approved", "user"), ("denied", "turn_stopped")], (
        "the audit keeps the yes and says the turn withdrew the call", trails["approved, then stopped"])


# ---------------------------------------------------------------------------
# SK5 -- a call's approvals are its own
# ---------------------------------------------------------------------------
def test_sk5_a_calls_approvals_are_its_own():
    approval, restore_queue = _queue_window()
    bench, loaded, close = bench_contracts._window()
    try:
        allow, dispatch = loaded[_ALLOW], loaded[_DISPATCH]
        queue = _approving(approval)
        assert allow.request_approval("conv", "bash", {"command": "echo alone"}, manager=queue, timeout=5.0)
        alone = queue.audit_log(1)[0]["approval_id"]
        assert allow.withdraw_approved("run_cancelled") == 0 and _trail(queue, alone) == [("approved", "user")], (
            "outside a dispatch nothing is held: a thread that asks and never dispatches keeps nothing")
        allow.request_approval("conv", "bash", {"command": "echo again"}, manager=queue, timeout=5.0)
        assert allow.withdraw_approved("run_cancelled") == 0, "a withdrawal outside a dispatch starts no holding"

        held, ready, go, counted = {}, threading.Event(), threading.Event(), {}

        def other_thread():
            allow.forget_approved()
            allow.request_approval("conv", "bash", {"command": "echo other"}, manager=queue, timeout=5.0)
            held["other"] = queue.audit_log(1)[0]["approval_id"]
            ready.set()
            go.wait(timeout=30)
            counted["other"] = allow.withdraw_approved("run_cancelled")
            allow.release_approved()

        thread = threading.Thread(target=other_thread)
        thread.start()
        ready.wait(timeout=30)
        allow.forget_approved()
        counted["main"] = allow.withdraw_approved("run_cancelled")
        allow.release_approved()
        go.set()
        thread.join(timeout=30)
        assert counted == {"main": 0, "other": 1}, f"each thread withdraws its own call's approvals: {counted}"
        assert _trail(queue, held["other"]) == [("approved", "user"), ("denied", "run_cancelled")]

        world = bench._World(bench.parse_suite(bench_contracts._mini(), name="mini"))

        def ask(conversation_id, tool_name, arguments, **kwargs):
            return allow.request_approval(conversation_id, tool_name, arguments, manager=queue, timeout=5.0)

        earlier = dispatch.dispatch_tool_call(dispatch.ToolCall(name="ls", arguments={"path": "."}, source="native"),
                                              mode="bulbe", sandbox=world.sandbox, approval_fn=ask)
        spent = queue.audit_log(1)[0]["approval_id"]
        later = dispatch.dispatch_tool_call(dispatch.ToolCall(name="ls", arguments={"path": "."}, source="native"),
                                            mode="daily", sandbox=world.sandbox, halt=lambda: "cancelled")
        assert earlier.executed and (later.executed, later.reason) == (False, "halted"), (earlier, later)
        assert _trail(queue, spent) == [("approved", "user")], (
            "a later call's stop never withdraws an approval an earlier call ran with")
        allow.request_approval("conv", "bash", {"command": "echo after"}, manager=queue, timeout=5.0)
        assert allow.withdraw_approved("run_cancelled") == 0, "a dispatch that returned holds nothing"
    finally:
        close()
        restore_queue()


# ---------------------------------------------------------------------------
# SK6 -- the run's stop
# ---------------------------------------------------------------------------
def test_sk6_the_runs_stop_is_a_cancel_first_then_the_halt():
    bench, loaded, close = bench_contracts._window()
    try:
        loop = loaded[_LOOP]

        def halt():
            return "proposal_cap"

        going = [True]
        assert loop._run_stop(None, None) is None and loop._run_stop(halt, None) is halt, (
            "with no cancel to read, the run's stop is its halt, unchanged")
        bare, both = loop._run_stop(None, lambda: going[0]), loop._run_stop(halt, lambda: going[0])
        assert (bare(), both()) == (None, "proposal_cap")
        assert loop._run_stop(halt, wc._raising)() == "cancelled", "a cancel that cannot be read stops the run"
        going[0] = False
        assert (bare(), both()) == ("cancelled", "cancelled"), "a cancel is read first"
    finally:
        close()


# ---------------------------------------------------------------------------
# SK7 -- a no is a no
# ---------------------------------------------------------------------------
def test_sk7_a_no_is_a_no_on_both_paths():
    approval, restore_queue = _queue_window()
    bench, loaded, close = bench_contracts._window()
    try:
        allow = loaded[_ALLOW]

        class Denying(approval.ToolCallApprovalManager):
            def submit(self, *args, **kwargs):
                approval_id, event = super().submit(*args, **kwargs)
                self.deny(approval_id, "user")
                return approval_id, event

        denying, silent = Denying(), approval.ToolCallApprovalManager()
        denying._reaper_active = silent._reaper_active = True
        allow.forget_approved()
        said_no = allow.request_approval("conv", "bash", {"command": "echo no"}, manager=denying, timeout=5.0)
        unanswered = allow.request_approval("conv", "bash", {"command": "echo late"}, manager=silent, timeout=0.2)
        held = allow.withdraw_approved("run_cancelled")
        allow.release_approved()
    finally:
        close()
        restore_queue()
    assert (said_no, unanswered, held) == (False, False, 0), (said_no, unanswered, held)

    loaded, rc, _schemas, close = turn_contracts._load_routes_only(
        extra_targets={_APPROVAL: source("tool_call_approval.py")})
    try:
        approval = loaded[_APPROVAL]
        queue = approval.ToolCallApprovalManager()
        queue._reaper_active = True
        turn, said = rc.ChatTurn("conv-n"), []

        def emit(event):
            said.append(event)
            if event[0] == "tool_call_pending":
                queue.deny(event[1]["approval_id"], "user")

        allowed = rc._make_approval_hook(queue, turn, "conv-n", emit, 30)("echo", {})
        (asked,) = [event[1]["approval_id"] for event in said if event[0] == "tool_call_pending"]
        trail = _trail(queue, asked)
    finally:
        close()
    assert allowed is False and not turn.stop.is_set(), "the chat's no, with no stop in sight, runs nothing"
    assert trail == [("denied", "user")], trail


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
