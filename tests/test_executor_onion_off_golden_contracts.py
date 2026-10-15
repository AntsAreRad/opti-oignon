#!/usr/bin/env python3
"""Contracts for the chat executor with the onion memory off: what it sends, and hands every seam, byte for byte.

The onion memory is off by default, and the chat path with it off is the
path every turn takes. ``tests/executor_golden/v1/onion_off.json`` holds what
that path did for a matrix of cases in two windows -- the onion wiring
contracts' own, isolated, and an integrated one where the context
optimizer, the summary tiers, the untrusted-data envelope and the
deduplicator run as they are -- turn by turn: every request the backend
received, the non-empty yields, the value the call returned, what it wrote
into its run, its window figures, and the ordered trace of the seams it
reached (the caches, the network check and its queue, the governor and its
ticket, the conversation store, the capture, the librarian's write side, and
every collaborator asked for context). ``tests/_onion_off_golden.py``
captures it in child processes under two hash seeds; the file is written
only on purpose, by its ``--write``, and its digest is pinned here. A missing
file fails, it is never skipped.

Each case also carries the facts that show its trigger fired -- a summary
made, a window dropped, a cache hit, a queue entered -- and a case whose
trigger stops firing fails by its name, so no case can pass by having gone
quiet. The set of cases is held both ways: a case the golden holds and the
matrix no longer runs fails, and the matrix itself -- each case's name and
what it checks -- is pinned by its digest here.

  * JW13a -- the isolated window: every case's turns are the golden's. The
    base case's requests are the onion wiring contracts' own, byte for byte,
    and the librarian's five postures (the counting fake off, the
    three-name fake off and on with nothing to say, the real librarian off,
    no librarian at all) send those very requests. A capability block is
    placed by the optimizer alone, and a governor that steps down sends the
    messages of one that admits.
  * JW13b -- the integrated window: every case's turns are the golden's; the
    five postures send the same requests; a capability block off the
    optimizer changes nothing.
  * JW13c -- the order of a turn: the cache, then the network check, then the
    admission, then the stream, each only when the one before let the turn
    through, after the refinement when the turn asks for one. A cache hit
    ends the turn at the cache, a queued request at the queue, a refusal at
    the admission; a stop raised during the lookup, the admission, the
    refinement or at the start of the stream's thread ends it there, with
    nothing after it but the rest of the lookup, the ticket's release and
    the ledger. A ticket is taken just before the call it admits (the slot
    named in between) and released just after; nothing an onion turn alone
    would call is ever reached.
  * JW13d -- the two hash seeds capture the same bytes.
  * JW13e -- the golden's options, replayed through the llama-server stream
    and generate heads, leave with no ``max_tokens``; and so does the shape
    of the benchmark judge's request, which carries ``num_predict``: a head
    that turned one into the other would change bytes with the onion off.

Local-only (the public distribution ships no tests).
"""

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_off_golden as golden  # noqa: E402
from _isolation import isolate, source  # noqa: E402

# The digest of the golden file, written by ``tests/_onion_off_golden.py --write``.
_GOLDEN_SHA256 = "708e63775c286b0ada3363a6594b11aeed9ec242aeedea702c4232e3a470e565"
# The digest of the matrix: each case's name and the descriptions of its checks, in order.
_MATRIX_SHA256 = "73ecd2974b5667464bce79bd272559b10fb7a44e5811fa8e812156654372fb9e"

# The first of JW13a to JW13d to run pays for the capture (about a second for both seeds, measured), whatever
# the order: each is given room for it.
BUDGET_S = {
    "test_jw13a_the_isolated_window_sends_the_golden_requests_and_the_wiring_contracts_own": 10.0,
    "test_jw13b_the_integrated_window_sends_the_golden_requests_and_reaches_its_seams_as_it_did": 10.0,
    "test_jw13c_a_turn_reaches_the_cache_the_network_the_admission_and_the_stream_in_that_order": 10.0,
    "test_jw13d_the_two_hash_seeds_capture_the_same_bytes": 10.0,
    "test_jw13e_the_golden_options_leave_the_llama_server_heads_with_no_max_tokens": 3.0,
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


def _pinned():
    assert golden.FIXTURE.is_file(), (
        f"the golden file is missing: {golden.FIXTURE}; it is written on purpose, never skipped")
    text = golden.FIXTURE.read_text(encoding="ascii")
    assert hashlib.sha256(text.encode("ascii")).hexdigest() == _GOLDEN_SHA256, (
        "the golden file is not the one pinned here")
    return json.loads(text)


def _live():
    return json.loads(_capture()[golden.SEEDS[0]])


def _matrix():
    return hashlib.sha256(golden.canonical(
        [[case["name"], [description for description, _check in case["checks"]]] for case in golden.CASES]
    ).encode("ascii")).hexdigest()


def _where(name, pinned, live):
    """The first place a case's turns part from the golden's, in words."""
    if len(pinned["turns"]) != len(live["turns"]):
        return f"{name}: {len(live['turns'])} turns, the golden has {len(pinned['turns'])}"
    for index, (old, new) in enumerate(zip(pinned["turns"], live["turns"])):
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                if key == "trace":
                    for step, (was, now) in enumerate(zip(old[key], new[key])):
                        if was != now:
                            return f"{name}, turn {index + 1}, trace step {step + 1}: {now} where the golden has {was}"
                return f"{name}, turn {index + 1}, {key}: {new.get(key)} where the golden has {old.get(key)}"
    return f"{name}: differs"


def _hold(env):
    """Every case of ``env`` is the golden's, and every one fired; returns the live capture's cases of ``env``."""
    pinned, live = _pinned(), _live()
    assert _matrix() == _MATRIX_SHA256, "the matrix of cases and their checks is not the one pinned here"
    cases = {name: case for name, case in live["golden"]["cases"].items() if name.startswith(env + "/")}
    expected = {case["name"] for case in golden.CASES if case["env"] == env}
    assert set(cases) == expected and expected, (sorted(cases), sorted(expected))
    held = {name for name in pinned["cases"] if name.startswith(env + "/")}
    assert held == expected, ("the golden's cases are the matrix's", sorted(held ^ expected))
    for name, case in sorted(cases.items()):
        assert name in pinned["cases"], f"{name}: not in the golden"
        assert case == pinned["cases"][name], _where(name, pinned["cases"][name], case)
    quiet = []
    for case in golden.CASES:
        if case["env"] != env:
            continue
        record = {"turns": cases[case["name"]]["turns"], "facts": live["facts"][case["name"]]}
        quiet += [f"{case['name']}: {description}" for description, check in case["checks"] if not check(record)]
    assert not quiet, f"a case whose trigger did not fire: {quiet}"
    return cases, live["facts"]


def _requests(case):
    return [[request["sha256"] for request in turn["requests"]] for turn in case["turns"]]


def _messages(case):
    return [[request["messages"] for request in turn["requests"]] for turn in case["turns"]]


# ---------------------------------------------------------------------------
# JW13a -- the isolated window
# ---------------------------------------------------------------------------
def test_jw13a_the_isolated_window_sends_the_golden_requests_and_the_wiring_contracts_own():
    cases, facts = _hold("isolated")
    postures = [f"isolated/librarian-{posture}" for posture in golden.POSTURES if posture != "counting-off"]
    for name in ["isolated/base", *postures]:
        assert facts[name]["scripted_digest_is_the_wiring_contracts"] is True, (
            f"{name}: the requests are not the onion wiring contracts' own")
        assert _requests(cases[name]) == _requests(cases["isolated/base"]), name
    assert all(turn["requests"] for turn in cases["isolated/base"]["turns"]), "control: the base case sent each turn"
    assert _requests(cases["isolated/manifest"]) == _requests(cases["isolated/manual"]), (
        "a capability block is placed by the optimizer alone")
    assert _messages(cases["isolated/governor-steps-down"]) == _messages(cases["isolated/governor-admits"]), (
        "a governor that steps down sends the messages of one that admits")
    assert _requests(cases["isolated/governor-steps-down"]) != _requests(cases["isolated/governor-admits"]), (
        "control: its admitted context is its own")


# ---------------------------------------------------------------------------
# JW13b -- the integrated window
# ---------------------------------------------------------------------------
def test_jw13b_the_integrated_window_sends_the_golden_requests_and_reaches_its_seams_as_it_did():
    cases, facts = _hold("integrated")
    base = _requests(cases["integrated/librarian-counting-off"])
    for posture in golden.POSTURES:
        assert _requests(cases[f"integrated/librarian-{posture}"]) == base, posture
    assert _requests(cases["integrated/manual"]) == base, "the postures send what the plain manual turn sends"
    assert _requests(cases["integrated/manifest"]) == base, "a capability block off the optimizer changes nothing"
    windows = [cases[name]["turns"][0]["window"]["strategy"] for name in ("integrated/manual", "integrated/optimizer")]
    assert windows == ["keep_all", "optimizer"], ("control: the two paths are each their own", windows)


# ---------------------------------------------------------------------------
# JW13c -- the order of a turn
# ---------------------------------------------------------------------------
_PHASE = {"semantic.get": 0, "cache.key": 0, "cache.get": 0, "semantic.fallback": 0,
          "network.online": 1, "network.enqueue": 1, "admit": 2, "stream": 3}


def _held_at(kinds, position):
    """Whether a ticket is held at ``position``: taken before it and not yet released."""
    return kinds[:position].count("ticket.set") > kinds[:position].count("ticket.clear")


def _order_faults(name, index, turn):
    trace = turn["trace"]
    kinds = [event[0] for event in trace]
    where = f"{name}, turn {index + 1}"
    faults = []
    # The refinement a turn asks for comes first: its admission, its ticket and its call, ahead of the cache.
    first_cache = next((position for position, kind in enumerate(kinds) if _PHASE.get(kind) == 0), len(kinds))
    prelude = max((position for position, kind in enumerate(kinds[:first_cache]) if kind == "generate"), default=-1)
    if prelude >= 0 and kinds[prelude + 1:prelude + 2] == ["ticket.clear"]:
        prelude += 1
    body = kinds[prelude + 1:]
    phases = [_PHASE[kind] for kind in body if kind in _PHASE]
    if phases != sorted(phases):
        faults.append(f"{where}: {[kind for kind in body if kind in _PHASE]} is out of order")
    if body.count("stream") > 1 or "generate" in body:
        faults.append(f"{where}: more than the one stream a turn sends: {body}")
    if any(kind.startswith("trip.") for kind in kinds):
        faults.append(f"{where}: an onion turn's own call was reached: {kinds}")
    hit = any(event[0] == "cache.get" and event[1].get("hit") for event in trace)
    if hit and any(_PHASE.get(kind, 0) > 0 for kind in body):
        faults.append(f"{where}: a cache hit went on past the cache")
    if "stop" in kinds:
        # A stop ends the turn where it is raised: what may follow is the rest of a lookup it interrupted, the
        # release of a ticket it found held, and the ledger.
        at = kinds.index("stop")
        allowed = {"ledger"}
        if _held_at(kinds, at):
            allowed.add("ticket.clear")
        if kinds[at - 1:at] == ["cache.key"]:
            allowed.add("cache.get")
        rest = kinds[at + 1:]
        if any(kind not in allowed for kind in rest) or any(rest.count(kind) > 1 for kind in allowed):
            faults.append(f"{where}: the turn went on past its stop: {rest}")
    if "network.enqueue" in body and ("admit" in body or "stream" in body):
        faults.append(f"{where}: a queued request went on past the queue")
    if "admit" in body and "stream" not in body and "stop" not in body and \
            not turn["results"].get("admission_refused"):
        faults.append(f"{where}: an admission that let the turn through opened no stream")
    for position, kind in enumerate(kinds):
        if kind == "ticket.set":
            following = [later for later in kinds[position + 1:] if later != "slot.choose"][:2]
            if following[:1] not in (["stream"], ["generate"]) and following != ["stop", "ticket.clear"]:
                faults.append(f"{where}: a ticket is not taken just before the call it admits")
        if kind == "ticket.clear" and not _held_at(kinds, position):
            faults.append(f"{where}: a ticket is released that was not held")
        if kind in ("stream", "generate") and _held_at(kinds, position) and \
                [later for later in kinds[position + 1:] if later != "stop"][:1] != ["ticket.clear"]:
            faults.append(f"{where}: a ticket is not released just after its call")
    return faults


def test_jw13c_a_turn_reaches_the_cache_the_network_the_admission_and_the_stream_in_that_order():
    cases = _live()["golden"]["cases"]
    faults, whole = [], 0
    for name, case in sorted(cases.items()):
        for index, turn in enumerate(case["turns"]):
            faults += _order_faults(name, index, turn)
            kinds = [event[0] for event in turn["trace"]]
            whole += all(kind in kinds for kind in ("cache.get", "network.online", "admit", "ticket.set", "stream"))
    assert not faults, faults
    assert whole >= 4, ("control: turns that reach all four steps are read", whole)
    hit = cases["isolated/cache-hit"]["turns"][1]
    assert _order_faults("a hit read on", 1, dict(hit, trace=hit["trace"] + [["stream", {}]])), (
        "control: a hit that went on to the stream is seen")
    tails = {name: [event[0] for event in cases[f"isolated/{name}"]["turns"][1]["trace"]][-5:]
             for name in ("stop-in-cache-lookup", "stop-in-admission", "stop-at-thread-start", "stop-in-refinement")}
    assert tails == {
        "stop-in-cache-lookup": ["memory.build", "cache.key", "stop", "cache.get", "ledger"],
        "stop-in-admission": ["cache.get", "network.online", "admit", "stop", "ledger"],
        "stop-at-thread-start": ["admit", "ticket.set", "stop", "ticket.clear", "ledger"],
        "stop-in-refinement": ["ticket.set", "generate", "stop", "ticket.clear", "ledger"],
    }, ("a stop is seen where it is raised", tails)
    for name in tails:
        stopped = cases[f"isolated/{name}"]["turns"][1]
        went_on = dict(stopped, trace=stopped["trace"] + [["network.online", {}]])
        assert _order_faults(name, 1, went_on), f"control: {name}, a turn that went on past its stop is seen"
    slotted = [event[0] for event in cases["integrated/named-llama-server-admitted"]["turns"][0]["trace"]]
    assert "slot.choose" in slotted and not _order_faults("slot", 0, cases[
        "integrated/named-llama-server-admitted"]["turns"][0]), "control: a slot named inside the ticket is read"
    refined = [event[0] for event in cases["isolated/refine"]["turns"][0]["trace"]]
    assert refined[:4] == ["admit", "ticket.set", "generate", "ticket.clear"], ("control: the prelude is read", refined)


# ---------------------------------------------------------------------------
# JW13d -- the two hash seeds
# ---------------------------------------------------------------------------
def test_jw13d_the_two_hash_seeds_capture_the_same_bytes():
    captured = _capture()
    assert sorted(captured) == sorted(golden.SEEDS) and len(golden.SEEDS) == 2
    first, second = (captured[seed] for seed in golden.SEEDS)
    assert len(json.loads(first)["golden"]["cases"]) == len(golden.CASES), "control: each seed captured every case"
    assert first == second, "the capture depends on the hash seed"


# ---------------------------------------------------------------------------
# JW13e -- the llama-server heads, under the golden's options
# ---------------------------------------------------------------------------
_BACKEND = "opti_oignon.inference_backend"
_SSE = [b'data: {"choices": [{"delta": {"content": "hi"}}]}\n', b"data: [DONE]\n"]
_BODY = json.dumps({"choices": [{"message": {"content": "hi"}}], "model": "m"}).encode("utf-8")
# What the heads may forward of a chat turn's request: the model, the messages, the stream switch, and the
# options they allow through.
_FORWARDED = {"model", "messages", "stream", "temperature", "top_p", "cache_prompt", "id_slot"}


class _Response:
    def __init__(self, body, lines):
        self._body, self._lines = body, list(lines)

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# The benchmark judge's request carries num_predict, and its timeout: the heads send it no max_tokens.
_JUDGE_OPTIONS = {"num_predict": 64, "timeout": 30}


def test_jw13e_the_golden_options_leave_the_llama_server_heads_with_no_max_tokens():
    pinned = _pinned()
    options = {json.dumps(request["options"], sort_keys=True)
               for case in pinned["cases"].values() for turn in case["turns"] for request in turn["requests"]}
    assert len(options) >= 3, ("control: the golden holds options with and without an admitted context", options)
    options.add(json.dumps(_JUDGE_OPTIONS, sort_keys=True))
    loaded, close = isolate(targets={_BACKEND: source("inference_backend.py")}, seeded={}, packages=("opti_oignon",))
    module = loaded[_BACKEND]
    sent = []

    def urlopen(request, timeout=None):
        sent.append(json.loads(request.data.decode("utf-8")))
        return _Response(_BODY, _SSE)

    real = module.urllib.request.urlopen
    module.urllib.request.urlopen = urlopen
    try:
        # A host off the machine is refused in Bulbe mode; the heads are read in Daily.
        module._live_mode = lambda: "daily"
        backend = module.LlamaServerBackend(host="http://fake:8080")
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        for text in sorted(options):
            list(backend.stream(model=golden.MODEL, messages=messages, options=json.loads(text), keep_alive="30m"))
            backend.generate(model=golden.MODEL, messages=messages, options=json.loads(text), keep_alive="30m")
    finally:
        module.urllib.request.urlopen = real
        close()
    assert len(sent) == 2 * len(options), ("control: each head sent each request", len(sent))
    assert any("temperature" in payload for payload in sent), "control: an allowed option is forwarded"
    for payload in sent:
        assert "max_tokens" not in payload, payload
        assert set(payload) <= _FORWARDED, sorted(set(payload) - _FORWARDED)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
