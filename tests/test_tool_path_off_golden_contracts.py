#!/usr/bin/env python3
"""Contracts for the chat's tool path with the onion memory off: what it sends, admits, asks, saves and yields.

With tools armed for every model the manifest does not refuse, the tools
pipeline is the path nearly every chat turn takes, and with the onion off it
runs beside the onion entirely. ``tests/executor_golden/v1/tool_path_off.json``
holds what that path did for a matrix of cases -- the tools pipeline's
decisions, calls and finals in their every shape, the other pipelines a turn
can take with tools armed and the saves each makes, a turn the web search
routes past the tools, one step of an execution pipeline, a governor that
admits and one that refuses, no backend at all, and the stops a turn can
meet -- turn by turn: every request the backend's heads received, the gate's
admissions, the approval hook's calls, the store's writes, the calls
remembered for the next turn, the non-empty yields, what the turn wrote into
its run, and the ordered trace of the seams it reached, the path's own calls
among them by the names of their keywords.
``tests/_tool_path_off_golden.py`` captures it in child processes under two
hash seeds; the file is written only on purpose, by its ``--write``, and its
digest is pinned here. A missing file fails, it is never skipped.

Each case also carries the facts that show its trigger fired -- a tool run,
a hint injected, a call salvaged, a row cut, a refusal met, a stop seen --
and a case whose trigger stops firing fails by its name, so no case can pass
by having gone quiet. Its script is consumed exactly: a path that asks the
engine once more, or once less, fails by name too. The set of cases is held
both ways, and the matrix -- each case's name and what it checks -- is
pinned by its digest here.

  * JW33a -- every case's turns are the golden's, byte for byte, and every
    case fired.
  * JW33b -- what the off path holds, read on every case: no request carries
    a budget (no ``num_ctx``, ``num_predict`` or ``max_tokens``) -- a rule
    that holds for the golden's governor only, whose decisions grant no
    context, so the gate's load accounting never runs; a decision goes at
    temperature 0.0 with the tool schemas or a format= schema, a final at
    0.3 with neither; under a governor, from the tool loop's first call
    (``stream_with_tools`` or ``execute_with_tools``) on, no ticket is taken
    or released and every request is admitted by the gate's ``direct``
    backstop, once, with no context requested; a ``chat`` admission is the
    chat executor's own, never more of them than its calls, and the pipeline
    runner's is the only other one; nothing only an onion turn reaches is
    reached, nor the librarian's curation, nor an admission with no
    governor seated; a pipeline that saves its own answer offers it to no
    capture, and writes the user's row, then the answer flagged
    ``assistant+tool`` with ``tool`` in its context; a stop or a close ends
    the turn where it is met. Each rule is shown to fail on a record that
    breaks it.
  * JW33c -- the two hash seeds capture the same bytes.
  * JW33d -- ``--check`` on a golden file that is absent fails by name and
    creates nothing, and a missing file is never read as agreeing.

Local-only (the public distribution ships no tests).
"""

import copy
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tool_path_off_golden as golden  # noqa: E402
from _isolation import REPO  # noqa: E402

# The digest of the golden file, written by ``tests/_tool_path_off_golden.py --write``.
_GOLDEN_SHA256 = "d1e6384f53c0de95c04db08a5cb5071e4250bd0ce7662ab6b616e94397fe8caf"
# The digest of the matrix: each case's name and the descriptions of its checks, in order.
_MATRIX_SHA256 = "d2f2dfb570ada952f279936678a9d0d68d95f0a153a463efd66e2bcc9026f334"
# The golden file in the tree, read before any contract points the module elsewhere.
_FIXTURE = golden.FIXTURE

# The first of JW33a to JW33d to run pays for the capture (about two seconds for both seeds, measured), whatever
# the order: each is given room for it. JW33d also starts the data firewall's check child, twice.
BUDGET_S = {
    "test_jw33a_every_case_sends_admits_asks_saves_and_yields_what_the_golden_holds": 20.0,
    "test_jw33b_the_off_path_sends_no_budget_holds_no_ticket_and_offers_no_save_to_the_write_side": 20.0,
    "test_jw33c_the_two_hash_seeds_capture_the_same_bytes": 20.0,
    "test_jw33d_a_check_on_an_absent_golden_file_fails_by_name_and_creates_nothing": 25.0,
}


@lru_cache(maxsize=1)
def _captured():
    """Each seed's capture as its child printed it, once per session, or why it failed: a capture that stalls
    costs one timeout, not one for each contract that reads it."""
    try:
        return golden.capture_in_children(timeout=60), None
    except Exception as exc:  # noqa: BLE001 - every contract that reads the capture reports its failure by name
        return None, f"{type(exc).__name__}: {exc}"


def _capture():
    captured, failure = _captured()
    assert failure is None, f"the capture failed: {failure}"
    return captured


def _pinned(path=None):
    fixture = Path(path) if path is not None else _FIXTURE
    assert fixture.is_file(), f"the golden file is missing: {fixture}; it is written on purpose, never skipped"
    text = fixture.read_text(encoding="ascii")
    assert hashlib.sha256(text.encode("ascii")).hexdigest() == _GOLDEN_SHA256, (
        "the golden file is not the one pinned here")
    return json.loads(text)


def _live():
    return json.loads(_capture()[golden.SEEDS[0]])


def _matrix():
    return hashlib.sha256(golden.canonical(
        [[case["name"], [description for description, _check in case["checks"]]] for case in golden.CASES]
    ).encode("ascii")).hexdigest()


# The fields a difference is read in first, and the digests that only point at one, read last.
_FIRST = ("requests", "admissions", "approvals", "writes", "history", "yields", "results", "trace", "messages",
          "options", "tools", "schema", "given", "think")
_LAST = ("engine", "fields", "request", "sha256")


def _order(keys):
    return sorted(keys, key=lambda key: (key in _LAST, _FIRST.index(key) if key in _FIRST else len(_FIRST), key))


def _difference(old, new, where):
    """The first place two records part, as its path and both sides: a field a person reads before a digest, and an
    item one side has and the other lacks by its place."""
    if type(old) is not type(new):
        return where, old, new
    if isinstance(old, dict):
        for key in _order(set(old) | set(new)):
            if old.get(key) != new.get(key):
                return _difference(old.get(key), new.get(key), f"{where}.{key}")
    elif isinstance(old, list):
        for index, (was, now) in enumerate(zip(old, new)):
            if was != now:
                return _difference(was, now, f"{where}[{index}]")
        if len(old) != len(new):
            index = min(len(old), len(new))
            return (f"{where}[{index}]", old[index] if index < len(old) else None,
                    new[index] if index < len(new) else None)
    return where, old, new


def _where(name, pinned, live):
    where, was, now = _difference(pinned, live, name)
    return f"{where}: {json.dumps(now)[:300]} where the golden has {json.dumps(was)[:300]}"


def _requests(case):
    return [request for turn in case["turns"] for request in turn["requests"]]


# ---------------------------------------------------------------------------
# JW33a -- the golden, case by case
# ---------------------------------------------------------------------------
def test_jw33a_every_case_sends_admits_asks_saves_and_yields_what_the_golden_holds():
    pinned, live = _pinned(), _live()
    assert _matrix() == _MATRIX_SHA256, "the matrix of cases and their checks is not the one pinned here"
    cases = live["golden"]["cases"]
    expected = {case["name"] for case in golden.CASES}
    assert set(cases) == expected and len(expected) == len(golden.CASES), (sorted(cases), sorted(expected))
    assert set(pinned["cases"]) == expected, ("the golden's cases are the matrix's", sorted(set(pinned["cases"])
                                                                                            ^ expected))
    for name in sorted(cases):
        assert cases[name] == pinned["cases"][name], _where(name, pinned["cases"][name], cases[name])
    quiet = []
    for case in golden.CASES:
        record = {"turns": cases[case["name"]]["turns"], "facts": live["facts"][case["name"]]}
        quiet += [f"{case['name']}: {description}" for description, check in case["checks"] if not check(record)]
    assert not quiet, f"a case whose trigger did not fire: {quiet}"
    assert sum(len(_requests(case)) for case in cases.values()) >= 60, "control: the matrix sends requests"


# ---------------------------------------------------------------------------
# JW33b -- what the off path holds, on every case
# ---------------------------------------------------------------------------
_BUDGET = ("num_ctx", "num_predict", "max_tokens")


def _request_faults(name, request):
    """What a request breaks of the off path's shape: a budget, or a decision or a final out of its form."""
    faults = []
    options = request["options"]
    if any(key in options for key in _BUDGET):
        faults.append(f"{name}: a request carries a budget: {options}")
    temperature = options.get("temperature")
    if temperature == 0.0 and not (bool(request["tools"]) != bool(request["schema"])):
        faults.append(f"{name}: a decision offers neither the tool schemas nor a format= schema, or both")
    if temperature == 0.3 and (request["tools"] or request["schema"]):
        faults.append(f"{name}: a final offers tools or a schema")
    return faults


# The tool loop's own entries: from the first of them on, the turn is the tool path's.
_LOOP = ("call.stream_with_tools", "call.execute_with_tools")


def _governor_faults(name, turn):
    """Under a governor, from the tool loop's first call on: no ticket taken or released, each request admitted
    once by the direct backstop with no context asked; a chat admission is the chat executor's own, never more of
    them than its calls (a think+tools turn's reasoning pass holds its ticket before the loop begins)."""
    trace = turn["trace"]
    kinds = [event[0] for event in trace]
    start = next((position for position, kind in enumerate(kinds) if kind in _LOOP), len(kinds))
    tail = trace[start:]
    faults = []
    if any(event[0].startswith("ticket.") for event in tail):
        faults.append(f"{name}: a ticket was taken or released on the tool path: {kinds[start:]}")
    requests = 0
    for position, event in enumerate(tail):
        if event[0] not in ("generate", "stream"):
            continue
        requests += 1
        following = [later for later in tail[position + 1:] if later[0] in ("admit", "generate", "stream")]
        admit = following[0] if following else None
        if admit is None or admit[0] != "admit" or admit[1].get("caller") != "direct" \
                or admit[1].get("requested_ctx") is not None:
            faults.append(f"{name}: a request is not admitted by the direct backstop alone: {following[:1]}")
    callers = [item["caller"] for item in turn["admissions"]]
    executor_calls = kinds.count("call.executor.execute")
    if callers.count("direct") != requests or set(callers) - {"direct", "pipeline", "chat"} \
            or callers.count("pipeline") > 1 or callers.count("chat") > executor_calls:
        faults.append(f"{name}: the admissions are not one backstop per request: {callers}")
    return faults


def _owned(turn):
    """Whether the turn's answer is the pipeline's to save, never the chat executor's."""
    return turn["results"].get("pipeline") in golden.PIPELINE_OWNED


def _save_faults(name, turn):
    """A pipeline's own save: the user's row, then the answer flagged assistant+tool, tool in its context."""
    rows = [[write["role"], write["origin"]] for write in turn["writes"]]
    if not rows:
        return []
    faults = []
    if rows != [["user", rows[0][1]], ["assistant", "assistant+tool"]] or rows[0][1] not in ("typed", "legacy"):
        faults.append(f"{name}: the save is not the user's row then the answer flagged assistant+tool: {rows}")
    elif "tool" not in (turn["writes"][1]["context"] or []):
        faults.append(f"{name}: the answer saved does not name the tool kind in its context")
    return faults


def _stop_faults(name, turn):
    """A stop ends the turn where it is raised; a close ends it where it is closed: nothing sent or written after."""
    kinds = [event[0] for event in turn["trace"]]
    faults = []
    for mark in ("stop", "close"):
        if mark in kinds:
            after = kinds[kinds.index(mark) + 1:]
            if any(kind in ("generate", "stream", "store.add", "tool") for kind in after):
                faults.append(f"{name}: the turn went on past its {mark}: {after}")
            if turn["writes"]:
                faults.append(f"{name}: a turn that met its {mark} was saved")
    return faults


def _case_faults(name, case, governed):
    faults = []
    for turn in case["turns"]:
        for request in turn["requests"]:
            faults += _request_faults(name, request)
        kinds = [event[0] for event in turn["trace"]]
        reached = sorted({kind for kind in kinds if kind.startswith("trip.") or kind == "curate"})
        if reached:
            faults.append(f"{name}: reached what only an onion turn reaches: {reached}")
        if governed:
            faults += _governor_faults(name, turn)
        elif turn["admissions"]:
            faults.append(f"{name}: an admission with no governor seated: {turn['admissions']}")
        if _owned(turn):
            if "capture" in kinds:
                faults.append(f"{name}: a pipeline's own save was offered to the capture")
            faults += _save_faults(name, turn)
        faults += _stop_faults(name, turn)
    return faults


def test_jw33b_the_off_path_sends_no_budget_holds_no_ticket_and_offers_no_save_to_the_write_side():
    cases = _live()["golden"]["cases"]
    governed = {case["name"] for case in golden.CASES if "governor" in case["options"]}
    faults = []
    for name, case in sorted(cases.items()):
        faults += _case_faults(name, case, name in governed)
    assert not faults, faults
    # Controls: each rule reads what it claims to read, and fails on a record that breaks it.
    assert len(governed) >= 5 and sum(len(turn["admissions"]) for name in governed
                                      for turn in cases[name]["turns"]) >= 20, ("control: governed cases read",
                                                                                sorted(governed))
    saved = [name for name, case in cases.items() if any(turn["writes"] for turn in case["turns"]) and
             all(_owned(turn) for turn in case["turns"])]
    assert len(saved) >= 20, ("control: the pipelines' own saves are read", saved)
    web = cases["web/executor-save"]["turns"][0]
    assert not _owned(web) and "capture" in [event[0] for event in web["trace"]], (
        "control: the chat executor's own save is offered to the capture, and the rule leaves it alone")
    admitted = cases["governor/admits-every-call"]
    turn = admitted["turns"][0]
    budget = copy.deepcopy(admitted)
    budget["turns"][0]["requests"][0]["options"]["num_ctx"] = 16384
    assert _case_faults("budget", budget, True), "control: a request carrying a budget is seen"
    bare = copy.deepcopy(turn["requests"][0])
    assert bare["options"]["temperature"] == 0.0 and bare["tools"], "control: the first request is a decision"
    bare["tools"] = None
    assert _request_faults("bare", bare), "control: a decision offering no tool schemas is seen"
    armed = copy.deepcopy(turn["requests"][-1])
    assert armed["options"]["temperature"] == 0.3 and not armed["tools"], "control: the last request is the final"
    armed["tools"] = [["read_file"], "0" * 16]
    assert _request_faults("armed", armed), "control: a final carrying the tool schemas is seen"
    # A ticket held around the tool loop, as a funnel holding one would leave it: one chat admission, the ticket
    # set before the loop's first call and released after its last request, the gate standing down on it, so no
    # backstop admission at all.
    loop = [event for event in turn["trace"] if event[0] != "admit"]
    first = next(position for position, event in enumerate(loop) if event[0] in _LOOP)
    last = max(position for position, event in enumerate(loop) if event[0] in ("generate", "stream"))
    around = dict(turn, admissions=[dict(turn["admissions"][0], caller="chat")],
                  trace=[["admit", {"caller": "chat"}], ["ticket.set", {}], *loop[first:last + 1], ["ticket.clear"],
                         *loop[last + 1:]])
    assert _governor_faults("around", around), "control: a ticket held around the tool loop is seen"
    assert any("ticket" in fault for fault in _governor_faults("around", around)), (
        "control: the ticket rule itself sees a ticket held around the tool loop")
    reasoned = cases["think-tools/governed"]["turns"][0]
    assert "ticket.clear" in [event[0] for event in reasoned["trace"]] and not _governor_faults("reasoned", reasoned), (
        "control: the reasoning pass's own ticket, held before the loop's first call, is the chat executor's")
    dropped = dict(turn, trace=[event for event in turn["trace"] if event[0] != "admit"], admissions=[])
    assert _governor_faults("dropped", dropped), "control: a request the backstop did not admit is seen"
    tripped = dict(turn, trace=turn["trace"] + [["trip.plan_window"]])
    assert _case_faults("tripped", {"turns": [tripped]}, True), "control: a journal plan asked on the off path"
    curated = dict(turn, trace=turn["trace"] + [["curate", {}]])
    assert _case_faults("curated", {"turns": [curated]}, True), "control: a save offered to the curation is seen"
    ungoverned = cases["tools/one-call-then-final"]["turns"][0]
    assert not ungoverned["admissions"], "control: an ungoverned turn holds no admission"
    admitted_alone = dict(ungoverned, admissions=[{"caller": "direct", "model": "m", "requested_ctx": None,
                                                    "admitted": True}])
    assert _case_faults("admitted alone", {"turns": [admitted_alone]}, False), (
        "control: an admission with no governor seated is seen")
    offered = dict(turn, trace=turn["trace"] + [["capture", {}]])
    assert _case_faults("offered", {"turns": [offered]}, True), "control: a pipeline's save offered to the capture"
    flagged = copy.deepcopy(turn)
    flagged["writes"][1]["origin"] = "assistant"
    assert _save_faults("flagged", flagged), "control: an answer saved without its tool flag is seen"
    unnamed = copy.deepcopy(turn)
    unnamed["writes"][1]["context"] = ["memory"]
    assert _save_faults("unnamed", unnamed), "control: an answer whose context does not name the tool is seen"
    stopped = cases["stop/mid-loop"]["turns"][0]
    went_on = dict(stopped, trace=stopped["trace"] + [["stream", {}]])
    assert _stop_faults("went on", went_on), "control: a turn that went on past its stop is seen"
    closed = cases["stop/final-closed-by-caller"]["turns"][0]
    assert "close" in [event[0] for event in closed["trace"]] and not _stop_faults("closed", closed), (
        "control: a close is read")
    reopened = dict(closed, trace=closed["trace"] + [["stream", {}]])
    assert _stop_faults("reopened", reopened), "control: a turn that went on past its close is seen"


# ---------------------------------------------------------------------------
# JW33c -- the two hash seeds
# ---------------------------------------------------------------------------
def test_jw33c_the_two_hash_seeds_capture_the_same_bytes():
    captured = _capture()
    assert sorted(captured) == sorted(golden.SEEDS) and len(golden.SEEDS) == 2
    first, second = (captured[seed] for seed in golden.SEEDS)
    assert len(json.loads(first)["golden"]["cases"]) == len(golden.CASES), "control: each seed captured every case"
    assert first == second, "the capture depends on the hash seed"


# ---------------------------------------------------------------------------
# JW33d -- a check on an absent golden file
# ---------------------------------------------------------------------------
def test_jw33d_a_check_on_an_absent_golden_file_fails_by_name_and_creates_nothing(tmp_path, monkeypatch, capsys):
    captured = dict(_capture())
    monkeypatch.setattr(golden, "capture_in_children", lambda *args, **kwargs: dict(captured))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    missing = tmp_path / "executor_golden" / "v1" / _FIXTURE.name
    monkeypatch.setattr(golden, "FIXTURE", missing)
    assert golden.main(["--check"]) == 1, "a check on an absent golden file did not fail"
    assert capsys.readouterr().out.splitlines()[-1:] == ["golden: ABSENT"], "the check did not name the absence"
    assert list(tmp_path.iterdir()) == [], "a check created something where the golden file would be"
    assert golden.compare(_FIXTURE.read_text(encoding="ascii"), missing) == "absent"
    with pytest.raises(AssertionError, match="the golden file is missing"):
        _pinned(missing)
    # Controls: the same check on the golden file in the tree agrees, and another text is a difference.
    monkeypatch.setattr(golden, "FIXTURE", _FIXTURE)
    assert golden.main(["--check"]) == 0 and capsys.readouterr().out.splitlines()[-1:] == ["golden: same"], (
        "control: a check on the golden file in the tree agrees")
    text = _FIXTURE.read_text(encoding="ascii")
    assert golden.compare(text + "\n") == "differs", "control: a different text is a difference"
    assert _FIXTURE.relative_to(REPO).as_posix() == "tests/executor_golden/v1/tool_path_off.json"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
