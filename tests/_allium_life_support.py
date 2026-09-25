#!/usr/bin/env python3
"""Shared support for the componion's life contracts: beings built at the engine level, and ways to run them.

Nothing here reaches the platform: a contract opens the shared window with
``open_allium``, wraps it in ``Engine``, and builds a ``Being`` -- a genesis
and its facts, drawn from the chassis stream on a fixed key -- that it
advances through the reference's byte protocol:

* ``Being.advance`` sends one ``advance`` request, with only the facts after
  the state's minute and up to ``to``;
* ``cuts`` advances through a list of stops, one call per stop; ``slices``
  advances to one minute under a sequence of budgets, one call per budget,
  until the life gets there;
* ``analytic`` is the work a trace accounts for, by the law's unit table;
  ``add_traces`` sums the traces of several calls;
* ``injected`` swaps in law files the engine does not carry -- a stable pair
  ``fixture_s`` and ``fixture_s2`` built from the fixture law, or defective
  copies -- for the length of a ``with`` block, and gives the carried files
  back when it ends.

The author script's ``ceilings`` is loaded by path, inside an open window,
since it imports the engine's codec.
"""

import copy
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO  # noqa: E402

WALL = 1760000000
DAY = 1440
MAX_INT = (1 << 53) - 1
T_MAX = MAX_INT - 2880
SCRIPTS = REPO / "scripts"
ORGANS = ("chem", "clock", "soil", "stage")


class Engine:
    """The reference engine of one window, asked through its byte protocol."""

    def __init__(self, loaded):
        prefix = "opti_oignon.allium."
        self.loaded = loaded
        self.wire = loaded[prefix + "wire"]
        self.fx = loaded[prefix + "fx"]
        self.rng = loaded[prefix + "rng"]
        self.lawfiles = loaded[prefix + "lawfiles"]
        self.civil = loaded[prefix + "ref.civil"]
        self.lawdata = loaded[prefix + "ref.lawdata"]
        self.world = loaded[prefix + "ref.world"]
        self.protocol = loaded[prefix + "ref.protocol"]
        self.calls = 0

    def ask(self, request):
        """The reference's answer to one request, parsed."""
        self.calls += 1
        return self.wire.parse(self.protocol.call(self.wire.emit(request)))

    def law(self, name):
        return self.ask({"name": name, "op": "law", "v": 1})


class Being:
    """A genesis and its facts, in canonical order, at the engine level."""

    def __init__(self, engine, *, suite, index, law="fixture", wall=WALL, tz=0, weather="garden",
                 hemisphere="north", band="long", params=None):
        self.engine = engine
        draw = engine.rng.Stream(bytes(32), "test." + suite, index)
        words = [draw.next_u64() for _ in range(12)]
        self.being = "".join(format(word, "016x") for word in words[0:2])
        self.origin = format(words[2], "016x")
        seed = "".join(format(word, "016x") for word in words[3:7])
        owner = "".join(format(word, "016x") for word in words[7:9])
        info = engine.law(law)
        defaults = {name: spec["default"] for name, spec in engine.lawfiles.law(law)["params"].items()}
        self.law_name = law
        self.v = info["version"]
        self.digest = info["digest"]
        self.b = wall // 60
        self.tz_birth = tz
        self.genesis = {
            "being": self.being,
            "body": {
                "band": band,
                "birth": {"tz": tz, "wall": wall},
                "derive": 1,
                "hemisphere": hemisphere,
                "laws": {"name": law, "params": dict(params or defaults), "provisional": info["provisional"],
                         "sha256": info["digest"], "v": info["version"]},
                "owner": owner,
                "rhythm_consent": False,
                "seed": seed,
                "soil": "encrypted",
                "weather": weather,
            },
            "kind": "genesis",
            "laws": info["version"],
            "origin": self.origin,
            "oseq": 0,
            "t": 0,
        }
        self.facts = []
        self._oseq = {self.origin: 1}

    # -- facts ----------------------------------------------------------------

    def fact(self, t, kind, body, *, laws=None, origin=None):
        """Append a fact at minute ``t``; the list stays in canonical order ``(t, origin, oseq)``."""
        origin = origin or self.origin
        oseq = self._oseq.get(origin, 0)
        self._oseq[origin] = oseq + 1
        fact = {"being": self.being, "body": body, "kind": kind, "laws": self.v if laws is None else laws,
                "origin": origin, "oseq": oseq, "t": t}
        self.facts.append(fact)
        self.facts.sort(key=lambda f: (f["t"], f["origin"], f["oseq"]))
        return fact

    def act(self, t, act, **kw):
        return self.fact(t, "act", {"act": act}, **kw)

    def tz(self, t, offset, **kw):
        return self.fact(t, "tz", {"quarters": offset // 15}, **kw)

    def pin(self, t, **kw):
        return self.fact(t, "laws_pin", {}, **kw)

    def unpin(self, t, **kw):
        return self.fact(t, "laws_unpin", {}, **kw)

    def midnight(self, t, offset):
        """The next local midnight strictly after ``t`` under ``offset``."""
        return self.engine.civil.next_midnight(self.b, t, offset)

    def evolve(self, t, params, *, effective_from, to=None, source=None, **kw):
        """An ``evolve`` at ``t`` to law ``to`` (``(name, sha256, v)``, default the being's own law)."""
        name, digest, v = to or (self.law_name, self.digest, self.v)
        from_name, from_digest = source or (self.law_name, self.digest)
        body = {"effective_from": effective_from, "from": {"name": from_name, "sha256": from_digest},
                "params": dict(params), "to": {"name": name, "sha256": digest, "v": v}}
        return self.fact(t, "evolve", body, **kw)

    def params(self, **changes):
        out = dict(self.genesis["body"]["laws"]["params"])
        out.update(changes)
        return out

    # -- requests -------------------------------------------------------------

    def after(self, at, to):
        """The facts a call from ``at`` (``None`` for the genesis) to ``to`` sends."""
        low = -1 if at is None else at
        return [fact for fact in self.facts if low < fact["t"] <= to]

    def request(self, state, to, *, budget=MAX_INT, probe=None, facts=None):
        at = None if state is None else state["at"]
        request = {"budget": budget, "facts": self.after(at, to) if facts is None else facts,
                   "genesis": self.genesis, "op": "advance", "state": state, "to": to, "v": 1}
        if probe is not None:
            request["probe"] = probe
        return request

    def advance(self, to, state=None, *, budget=MAX_INT, probe=None):
        """One ``advance`` call; a refusal fails the contract with its words."""
        answer = self.engine.ask(self.request(state, to, budget=budget, probe=probe))
        assert "refused" not in answer, answer
        return answer

    def timeline(self, to, tstate=None, *, midnights_after=None):
        at = None if tstate is None else tstate["at"]
        kinds = ("evolve", "laws_pin", "laws_unpin", "tz")
        request = {"facts": [f for f in self.after(at, to) if f["kind"] in kinds], "from": tstate,
                   "genesis": self.genesis, "op": "timeline", "to": to, "v": 1}
        if midnights_after is not None:
            request["midnights_after"] = midnights_after
        answer = self.engine.ask(request)
        assert "refused" not in answer, answer
        return answer


# ---------------------------------------------------------------------------
# Ways to run a life
# ---------------------------------------------------------------------------

def cuts(being, stops, *, probe=None, state=None):
    """Advance through ``stops`` (increasing minutes), one call each; the answers, in order."""
    answers = []
    for stop in stops:
        answer = being.advance(stop, state, probe=probe)
        assert answer["done"] and answer["at"] == stop, (stop, answer["at"])
        state = answer["state"]
        answers.append(answer)
    return answers


def slices(being, to, budgets, *, probe=None, state=None):
    """Advance to ``to`` under the budgets drawn one per call from ``budgets``; the answers, in order."""
    answers = []
    for budget in budgets:
        answer = being.advance(to, state, budget=budget, probe=probe)
        state = answer["state"]
        answers.append(answer)
        if answer["done"]:
            return answers
    raise AssertionError("the budgets ran out before the life reached its minute")


def add_traces(traces):
    """The sum of several calls' traces, key by key."""
    total = {}
    for trace in traces:
        _add(total, trace)
    return total


def _add(total, part):
    for key, value in part.items():
        if isinstance(value, dict):
            _add(total.setdefault(key, {}), value)
        else:
            total[key] = total.get(key, 0) + value


def analytic(trace, law):
    """The work a trace accounts for by the law's unit table (the count the engine must equal)."""
    units = law["work"]["units"]
    count = (trace["visits"] * units["visit"] + trace["facts"] * units["fact"] + trace["fast_path_days"] * units["fast_path_day"]
             + trace["draws"] * units["draw"] + trace["env"] * units["env"])
    for organ, layers in trace["calls"].items():
        for layer in ("fast", "daily"):
            costs = units["organs"].get(organ, {}).get(layer)
            if costs is None:
                continue
            for state in ("awake", "dormant"):
                count += layers[layer][state] * costs[state]
    for act, n in trace["acts"].items():
        count += n * sum(units["act"].get(act, {}).values())
    return count


# ---------------------------------------------------------------------------
# Laws the engine does not carry
# ---------------------------------------------------------------------------

def stable_pair(engine):
    """Two stable laws built from the fixture: ``fixture_s``, and ``fixture_s2`` that migrates from it.

    ``fixture_s2`` carries a higher version, wider param ranges and a larger
    sugar ceiling; everything ``successor_ok`` asks to be the same is the same.
    """
    base = engine.lawfiles.law("fixture")
    first = copy.deepcopy(base)
    first["name"] = "fixture_s"
    first["provisional"] = False
    first["version"] = 1
    second = copy.deepcopy(first)
    second["name"] = "fixture_s2"
    second["version"] = 2
    second["succeeds"] = {"migrate": "identity", "name": "fixture_s", "sha256": engine.lawfiles.digest(first)}
    second["params"]["evap_awake"]["hi"] = 16384
    second["params"]["evap_awake"]["lo"] = 0
    second["params"]["sun_max"]["lo"] = 8192
    second["constants"]["chem"]["sugar_max"] = 327680
    return {"fixture_s": engine.wire.emit(first), "fixture_s2": engine.wire.emit(second)}


def defective(engine, edit, name="fixture_x"):
    """A copy of the fixture law under ``name``, with ``edit(law)`` applied; its file bytes."""
    law = copy.deepcopy(engine.lawfiles.law("fixture"))
    law["name"] = name
    edit(law)
    return {name: engine.wire.emit(law)}


@contextmanager
def injected(engine, files):
    """The window's law files with ``files`` (name -> bytes) added, for the length of the block."""
    lawfiles = engine.lawfiles
    carried = lawfiles.LAWS
    real = lawfiles.law_bytes

    def law_bytes(name):
        if name in files:
            return files[name]
        return real(name)

    lawfiles.LAWS = carried + tuple(name for name in files if name not in carried)
    lawfiles.law_bytes = law_bytes
    try:
        yield
    finally:
        lawfiles.LAWS = carried
        lawfiles.law_bytes = real


def load_script(name, module_name):
    """An authoring script loaded by path; call it inside an open window, since the script imports the codec."""
    path = SCRIPTS / name
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module
