#!/usr/bin/env python3
"""The trust kernel's native twin decides every call as the gate does, in its shadow.

The gate (``provenance``) labels each argument of a tool call and decides the call. The native core holds a
second implementation of both (``rust/oo_core/src/trust.rs``), asked beside the gate at each call and never in its
place:

  * NT1 -- the twin labels every argument as the gate does, on calls drawn at random: typed units, texts that fold
    to one and texts that do not, the default in type and value and out of either, calls that reach the network
    and calls that stay on the machine; each call is counted agreed, and a twin made to disagree is counted;
  * NT2 -- the twin decides every call as the gate does, on calls drawn at random through every check: offered or
    not, each class in each mode, each policy of the file and of the turn, values too long to be shown, approvals
    and the answers a person gives; every reason the gate gives is reached;
  * NT3 -- a twin that disagrees changes no verdict, and is said: the call is decided as the gate decides it, the
    divergence is counted, and the log names the tool, its class, its mode and its labels, never a value;
  * NT4 -- a twin that is not there, of another shape, or failing, changes nothing either, and every call it was
    not asked, or that failed, is counted by its reason;
  * NT5 -- the twin's small tables are the gate's, entry by entry: the classes each mode permits, and the policy
    of a network call for each policy of the file and of the turn.

NT1, NT2 and NT5 need the built artefact, as the native core's contracts do (scripts/build_oo_core.sh run once).
The gate is loaded through the shared isolation window. Local-only.
"""

import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_nt1_the_twin_labels_every_argument_as_the_gate_does": 10.0,
    "test_nt2_the_twin_decides_every_call_as_the_gate_does": 10.0,
    "test_nt3_a_twin_that_disagrees_changes_no_verdict_and_is_said_without_a_value": 3.0,
    "test_nt4_a_twin_absent_of_another_shape_or_failing_changes_nothing_and_is_counted": 3.0,
    "test_nt5_the_twin_s_small_tables_are_the_gate_s_entry_by_entry": 3.0,
}

_REAL = object()
_E_ACUTE = chr(0xE9)
_COMBINING_ACUTE = chr(0x301)
_SPACES = (" ", "  ", "\t", "\n", chr(0xA0), chr(0x3000), " \t ", "\r\n")
_WORDS = ("hello world", "Caf" + _E_ACUTE + " au lait", "tab stop", "x", "MiXeD Case words", "the end.")
_TOOLS = ("web_search", "bash", "todo", "manage_memory", "unknown_tool", "view", "sandbox_bash")


def _gate(core=_REAL):
    """(provenance, restore): the gate in a window, its native core the one built, or ``core`` in its place."""
    loaded, restore = isolate(targets={"opti_oignon.memory.probes": source("memory", "probes.py"),
                                       "opti_oignon.pending_writes": source("pending_writes.py"),
                                       "opti_oignon.native": source("native", "__init__.py"),
                                       "opti_oignon.provenance": source("provenance.py")},
                              packages=("opti_oignon.memory",))
    native = loaded["opti_oignon.native"]
    if core is _REAL:
        if native.load() is None:
            restore()
            raise AssertionError("oo_core is not built: run scripts/build_oo_core.sh on this machine")
    else:
        native._loaded = core
    return loaded["opti_oignon.provenance"], restore


def _respelled(rng, text):
    """The same words, another spelling: other white space, the decomposed accent, padding at the ends."""
    words = text.replace(_E_ACUTE, "e" + _COMBINING_ACUTE) if rng.random() < 0.5 else text
    out = rng.choice(_SPACES).join(words.split(" "))
    return rng.choice(("", " ", "\n")) + out + rng.choice(("", " ", "\t"))


def _units(rng):
    chosen = rng.sample(_WORDS, rng.randrange(0, 4))
    return tuple(_respelled(rng, word).strip() if rng.random() < 0.3 else word for word in chosen)


def _value(rng, units):
    kind = rng.randrange(8)
    if units and kind == 0:
        return rng.choice(units)
    if units and kind in (1, 2):
        return _respelled(rng, rng.choice(units))
    if kind == 3:
        return rng.choice((0, 1, 2.0, True, False, None, [1, "a"], {"k": "v"}))
    if kind == 4:
        return "lone " + chr(0xDC80) + " surrogate"
    if kind == 5:
        return rng.choice(("", "   ", "\n"))
    return rng.choice(_WORDS) + rng.choice(("", " more", "!"))


def _default(rng, value):
    kind = rng.randrange(5)
    if kind == 0:
        return value
    if kind == 1 and isinstance(value, bool):
        return int(value)
    if kind == 1 and isinstance(value, int):
        return float(value)
    if kind == 2:
        return "default text"
    return None if kind == 3 else 7


def _call(rng, prov):
    """(tool, arguments, turn, defaults, network flag) drawn at random."""
    units = _units(rng)
    names = rng.sample(("query", "path", "text", "n", "flag"), rng.randrange(0, 4))
    arguments = {name: _value(rng, units) for name in names}
    defaults = {name: _default(rng, value) for name, value in arguments.items() if rng.random() < 0.5}
    return rng.choice(_TOOLS), arguments, prov.TurnProvenance(units=units), defaults, rng.random() < 0.1


class _Answer:
    def __init__(self, answer):
        self.answer = answer

    def __call__(self, tool_name, arguments, labels=None, effect=None):
        return self.answer


class _Skewed:
    """The built twin, one answer turned: its labels with the first typed one read unendorsed, or its decision
    reversed."""

    def __init__(self, core, *, labels=False, decision=False):
        self._core, self._labels, self._decision = core, labels, decision
        self.TRUST_FORMAT = core.TRUST_FORMAT

    def trust_labels(self, *args):
        rows = list(self._core.trust_labels(*args))
        if self._labels:
            for at, (name, label) in enumerate(rows):
                if label == "typed":
                    rows[at] = (name, "unendorsed")
                    break
        return rows

    def trust_decide(self, *args):
        allowed, reason, asked = self._core.trust_decide(*args)
        return (not allowed, reason, asked) if self._decision else (allowed, reason, asked)


# ---------------------------------------------------------------------------
# NT1 -- the labels
# ---------------------------------------------------------------------------
def test_nt1_the_twin_labels_every_argument_as_the_gate_does():
    rng = random.Random(1231)
    prov, restore = _gate()
    try:
        calls, typed = 3000, 0
        for _ in range(calls):
            tool, arguments, turn, defaults, network = _call(rng, prov)
            labels = prov.assess(tool, arguments, turn, defaults=defaults, network=network).labels
            typed += sum(tag == "typed" for tag in labels.values())
        share = prov.twin_share()["labels"]
        core = prov._twin()[0]
    finally:
        restore()
    assert typed > calls // 10, ("control: the draws reach typed labels", typed)
    assert share == {"agreed": calls}, share

    rng = random.Random(1232)
    prov, restore = _gate(_Skewed(core, labels=True))
    try:
        skewed = 0
        for _ in range(300):
            tool, arguments, turn, defaults, network = _call(rng, prov)
            labels = prov.assess(tool, arguments, turn, defaults=defaults, network=network).labels
            skewed += "typed" in labels.values()
        share = prov.twin_share()["labels"]
    finally:
        restore()
    assert skewed and share.get("diverged") == skewed, ("witness: a twin that disagrees is counted", share, skewed)


# ---------------------------------------------------------------------------
# NT2 -- the decisions
# ---------------------------------------------------------------------------
def _policies(tmp_path):
    files = {"absent": tmp_path / "absent.yaml"}
    for name, text in (("free", "network_daily: free\n"), ("ask", "network_daily: ask_unendorsed\n"),
                       ("refuse", "network_daily: refuse_unendorsed\n"), ("empty", ""),
                       ("other", "network_daily: maybe\n"), ("unnamed", "something: else\n")):
        files[name] = tmp_path / f"{name}.yaml"
        files[name].write_text(text, encoding="utf-8")
    return files


def test_nt2_the_twin_decides_every_call_as_the_gate_does(tmp_path):
    rng = random.Random(1233)
    files = _policies(tmp_path)
    prov, restore = _gate()
    try:
        calls, reasons = 3000, set()
        for _ in range(calls):
            tool, arguments, _turn, defaults, network = _call(rng, prov)
            if rng.random() < 0.15:
                arguments["query"] = "q" * rng.choice((1999, 2000, 2001, 4000))
            turn = prov.TurnProvenance(units=_turn.units,
                                       demand=_Answer(rng.random() < 0.6) if rng.random() < 0.7 else None,
                                       policy=rng.choice((None, None, "free", "ask_unendorsed", "refuse_unendorsed",
                                                          "bogus")))
            prov.POLICY_FILE = files[rng.choice(tuple(files))]
            assessment = prov.assess(tool, arguments, turn, defaults=defaults, network=network)
            offered = None if rng.random() < 0.5 else set(rng.sample(_TOOLS, rng.randrange(1, len(_TOOLS))))
            approval = _Answer(rng.random() < 0.6) if rng.random() < 0.3 else None
            verdict = prov.decide(tool, assessment, turn, mode=rng.choice(("daily", "daily", "bulbe", "weird")),
                                  offered=offered, approval=approval, approval_required=rng.random() < 0.3)
            reasons.add(verdict.reason)
        share = prov.twin_share()["decision"]
    finally:
        restore()
    assert reasons == {"allowed", "not_offered", "not_permitted", "unendorsed", "unshowable", "no_channel",
                       "denied"}, ("control: every reason the gate gives is reached", reasons)
    assert share == {"agreed": calls}, share


# ---------------------------------------------------------------------------
# NT3 -- a twin that disagrees
# ---------------------------------------------------------------------------
def _decided(prov, secret):
    turn = prov.TurnProvenance(units=("find cheap flights",), demand=_Answer(True))
    out = []
    for tool, arguments in (("web_search", {"query": secret}), ("web_search", {"query": "find cheap flights"}),
                            ("bash", {"command": secret})):
        assessment = prov.assess(tool, arguments, turn)
        verdict = prov.decide(tool, assessment, turn, mode="daily", approval_required=tool == "bash",
                              approval=_Answer(False))
        out.append((dict(assessment.labels), verdict.allowed, verdict.reason, verdict.asked))
    return out


def test_nt3_a_twin_that_disagrees_changes_no_verdict_and_is_said_without_a_value(tmp_path, caplog):
    secret = "SECRET-7f3a planted in an argument"
    prov, restore = _gate(None)
    try:
        prov.POLICY_FILE = tmp_path / "absent.yaml"
        alone = _decided(prov, secret)
    finally:
        restore()
    prov, restore = _gate()
    try:
        core = prov._twin()[0]
    finally:
        restore()
    prov, restore = _gate(_Skewed(core, labels=True, decision=True))
    try:
        prov.POLICY_FILE = tmp_path / "absent.yaml"
        with caplog.at_level(logging.WARNING):
            shadowed = _decided(prov, secret)
        share = prov.twin_share()
    finally:
        restore()
    assert shadowed == alone, (shadowed, alone)
    assert share["decision"] == {"diverged": 3} and share["labels"].get("diverged") == 1, share
    said = "\n".join(record.getMessage() for record in caplog.records)
    assert "diverge" in said and "web_search" in said and "bash" in said, said
    assert "SECRET" not in said and "cheap flights" not in said, said


# ---------------------------------------------------------------------------
# NT4 -- absent, another shape, failing
# ---------------------------------------------------------------------------
class _Shape:
    TRUST_FORMAT = 2

    def trust_labels(self, *args):
        raise AssertionError("a core of another shape is never asked")

    trust_decide = trust_labels


class _Failing:
    TRUST_FORMAT = 1

    def trust_labels(self, *args):
        raise RuntimeError("the core failed")

    trust_decide = trust_labels


def test_nt4_a_twin_absent_of_another_shape_or_failing_changes_nothing_and_is_counted(tmp_path, caplog):
    seen = {}
    for name, core in (("absent", None), ("another shape", _Shape()), ("failed", _Failing())):
        prov, restore = _gate(core)
        try:
            prov.POLICY_FILE = tmp_path / "absent.yaml"
            with caplog.at_level(logging.WARNING):
                seen[name] = (_decided(prov, "a value"), prov.twin_share())
        finally:
            restore()
    decided = {name: out for name, (out, _share) in seen.items()}
    assert decided["another shape"] == decided["absent"] and decided["failed"] == decided["absent"], decided
    for name, (_out, share) in seen.items():
        assert share == {"labels": {name: 3}, "decision": {name: 3}}, (name, share)
    assert "a value" not in "\n".join(record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# NT5 -- the small tables
# ---------------------------------------------------------------------------
def test_nt5_the_twin_s_small_tables_are_the_gate_s_entry_by_entry():
    prov, restore = _gate()
    try:
        core = prov._twin()[0]
        permits = {}
        for effect in prov.EFFECTS + ("bogus",):
            for mode in ("daily", "bulbe", "weird"):
                allowed, reason, _asked = core.trust_decide("t", None, effect, mode, [], None, None, [], 2000, False,
                                                            False, False, [])
                permits[(effect, mode)] = (allowed, prov.permitted(effect, mode), reason)
        policies = {}
        outcome = {"free": (True, "allowed", False), "ask_unendorsed": (True, "allowed", True),
                   "refuse_unendorsed": (False, "unendorsed", False)}
        for filed in prov.NETWORK_POLICIES:
            for explicit in (None,) + prov.NETWORK_POLICIES + ("bogus",):
                twin = core.trust_decide("web_search", None, "network", "daily", [("query", "unendorsed")], explicit,
                                         filed, [5], 2000, False, False, True, [True])
                policies[(filed, explicit)] = (twin, outcome[prov._policy("daily", explicit, filed)])
    finally:
        restore()
    assert len(permits) == 21 and all(allowed == want for allowed, want, _reason in permits.values()), permits
    assert len(policies) == 15 and all(twin == want for twin, want in policies.values()), policies
