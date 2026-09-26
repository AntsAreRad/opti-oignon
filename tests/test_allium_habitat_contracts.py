#!/usr/bin/env python3
"""Contracts for the habitat: the container a being lives in, and what the security mode changes.

A being lives in an encrypted pot, or in a glass jar when no key is
configured and the settings allow it. The mode is read once per action:
Daily serves everything; Bulbe -- or a mode that cannot be read, which
takes Bulbe's rules -- closes the lid of a pot and seals a jar, and gates
every capability the policy names. The mode changes what is served, never
the life.

  * HB3 -- one encrypted being, looked at by three gardens that read the
    mode as Daily, as Bulbe and as nothing at all (the reader raises): the
    same view hash, alive each time, the file untouched; the labels say the
    mode (``bulbe``, ``mode_unknown``); the Bulbe and unread forms differ at
    exactly that label line and share the closed rim, and the Daily form
    differs from the Bulbe one at the label line and the rim alone.
  * HB4 -- the Bulbe policy is a closed record: Daily and Bulbe are the
    table's, only exactly ``"daily"`` gets Daily, and the container and
    layer follow the soil and the mode. The gate calls a capability under
    Daily, and never under Bulbe, an unread mode or a garden switched off;
    an unknown capability is a ``KeyError``.
  * HB6 -- a glass jar is labelled on every form: the card discloses the jar
    before it asks, and the sow result, both tiers of ``show``, its JSON,
    the lab, the laws screen, ``keep verify``, ``keep laws diff``, a gesture,
    a name and a refused second pin all carry the jar's line. In Bulbe the
    jar is sealed: ``show`` says so with no drawing, a gesture, a law diff
    and a deep verification are refused with the same lines on stderr (the
    sealed line first), and the laws screen shows the doctrine and those
    lines.

Local-only (the public distribution ships no tests). The platform and the
terminal load through the shared isolation window with the platform's
configuration, keys, mode, audit log and user modules proven unreachable;
every seam is injected (``tests/_allium_store_support.py``,
``tests/_allium_garden_support.py``); the glass jar is judged by the real
``store.probe``.
"""

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_garden_support as garden  # noqa: E402
import _allium_store_support as support  # noqa: E402

BUDGET_S = {
    "test_hb3_the_mode_changes_the_served_form_and_never_the_life": 2.0,
    "test_hb4_the_bulbe_policy_is_closed_and_the_gate_calls_only_under_daily": 2.0,
    "test_hb6_a_glass_jar_is_labelled_on_every_form_and_sealed_in_bulbe": 2.0,
}
FIELDS = ("life", "glass_open", "taste", "voice", "initiatives", "dream_depth_change", "sync", "clear_export")
DAILY = (True, True, True, True, True, True, True, False)
BULBE = (True, False, False, False, False, False, False, False)
CAPABILITIES = ("taste", "voice", "initiatives", "sync", "dream_depth_change")
POT_OPEN = "   ." + "-" * 24 + "."
POT_SHUT = "   ." + "=" * 24 + "."


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def p(monkeypatch, tmp_path):
    window, restore = garden.open_garden(monkeypatch, tmp_path)
    try:
        yield window
    finally:
        restore()


def _jar(p, tmp_path, suite, mode="daily"):
    """Seams for a glass jar: no key, the jar allowed, Daily, the real probe."""
    return support.seams(p, tmp_path, suite=suite, anchor_secret=lambda: ("none", None, "nokey"),
                         persistence={"busy_timeout_ms": 5000, "path": "allium", "require_encryption": False},
                         probe=p.store.probe, mode=support.Mode(mode))


def _art(text):
    """Rows of a drawing: a pot's rim or body, a jar's walls, the sky's sun, the ground."""
    marks = ("(..)", "(--)", "(*)", "( )", '"  "', "____", "\\", "|")
    return [line for line in text.splitlines() if any(mark in line for mark in marks)]


# ---------------------------------------------------------------------------
# HB3 -- the mode changes the form, never the life
# ---------------------------------------------------------------------------
def test_hb3_the_mode_changes_the_served_form_and_never_the_life(p, tmp_path):
    say = p.wording.say
    given = support.seams(p, tmp_path, suite="hb3")
    target = support.store(p, given)
    try:
        being = support.sow(p, target)
        given["clock"].advance_days(1)
        being.append("act", {"act": "water"}, transport=support.cli(p))
        given["clock"].advance_days(4)
    finally:
        target.close()
    path = support.store_path(p, given)
    before = garden.sha256(path)
    readers = {"daily": support.Mode("daily"), "bulbe": support.Mode("bulbe"),
               "raiser": support.Mode(RuntimeError("the mode cannot be read"))}
    looks = {}
    for name, reader in readers.items():
        gardener = garden.garden(p, dict(given, mode=reader))()
        try:
            looks[name] = gardener.look()
        finally:
            gardener.close()
        assert reader.reads >= 1, name
    assert {look.status for look in looks.values()} == {"alive"}
    assert len({look.view.hash for look in looks.values()}) == 1, "the same life under every mode"
    assert looks["daily"].view.at == (given["clock"].wall - support.WALL) // 60, "shown at the minute of now"
    assert garden.sha256(path) == before, "looking wrote nothing"
    assert (looks["daily"].labels, looks["bulbe"].labels, looks["raiser"].labels) == (
        ("prototype",), ("prototype", "bulbe"), ("prototype", "mode_unknown"))
    assert (looks["daily"].habitat, looks["bulbe"].habitat, looks["raiser"].habitat) == (
        ("pot", "open"), ("pot", "bulbe"), ("pot", "bulbe"))
    assert (looks["daily"].mode, looks["bulbe"].mode, looks["raiser"].mode) == ("daily", "bulbe", "unknown")

    forms = {name: [line.text for line in p.describe.show(look, "ascii")] for name, look in looks.items()}
    bulbe, raiser, daily = forms["bulbe"], forms["raiser"], forms["daily"]
    assert len(bulbe) == len(raiser)
    differ = [i for i in range(len(bulbe)) if bulbe[i] != raiser[i]]
    assert [bulbe[i] for i in differ] == [say("label.bulbe").text], differ
    assert [raiser[i] for i in differ] == [say("label.mode_unknown").text], differ
    assert POT_SHUT in bulbe and POT_SHUT in raiser and POT_OPEN not in bulbe
    unlabelled = [line for line in bulbe if line != say("label.bulbe").text]
    assert len(unlabelled) == len(daily) == len(bulbe) - 1
    changed = [i for i in range(len(daily)) if daily[i] != unlabelled[i]]
    assert [(daily[i], unlabelled[i]) for i in changed] == [(POT_OPEN, POT_SHUT)], changed


# ---------------------------------------------------------------------------
# HB4 -- the policy, closed, and the gate
# ---------------------------------------------------------------------------
class _Capability:
    def __init__(self, name):
        self.name = name
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.name


def test_hb4_the_bulbe_policy_is_closed_and_the_gate_calls_only_under_daily(p, tmp_path):
    h = p.habitat
    assert tuple(h.ModePolicy._fields) == FIELDS
    assert (tuple(h.DAILY), tuple(h.BULBE)) == (DAILY, BULBE)
    assert h.policy("daily") == h.DAILY
    for mode in ("bulbe", "unknown", "Daily", " daily", "daily ", None, ""):
        assert h.policy(mode) == h.BULBE, repr(mode)
    assert tuple(h.CAPABILITIES) == CAPABILITIES
    layers = {("encrypted", "daily"): ("pot", "open"), ("encrypted", "bulbe"): ("pot", "bulbe"),
              ("encrypted", "unknown"): ("pot", "bulbe"), ("glass", "daily"): ("jar", "open"),
              ("glass", "bulbe"): ("jar", "sealed"), ("glass", "unknown"): ("jar", "sealed")}
    for (soil, mode), want in layers.items():
        assert tuple(h.layer(soil, mode)) == want, (soil, mode)

    counts = {}
    readers = {"daily": ("on", "daily"), "bulbe": ("on", "bulbe"),
               "raiser": ("on", RuntimeError("the mode cannot be read")), "off": ("off", "daily")}
    for name, (switch, mode) in readers.items():
        given = support.seams(p, tmp_path.joinpath(name), suite="hb4", mode=support.Mode(mode))
        capabilities = {cap: _Capability(cap) for cap in ("taste", "voice", "initiatives", "sync")}
        gardener = garden.garden(p, given, switch=switch)()
        try:
            answers = {cap: gardener.gated(cap, call) for cap, call in capabilities.items()}
            with pytest.raises(KeyError):
                gardener.gated("telepathy", _Capability("telepathy"))
        finally:
            gardener.close()
        counts[name] = {cap: call.calls for cap, call in capabilities.items()}
        if name == "daily":
            assert answers == {cap: cap for cap in capabilities}, answers
        else:
            assert set(answers.values()) == {None}, (name, answers)
    assert all(count >= 1 for count in counts["daily"].values()), counts["daily"]
    for name in ("bulbe", "raiser", "off"):
        assert set(counts[name].values()) == {0}, (name, counts[name])


# ---------------------------------------------------------------------------
# HB6 -- a glass jar, labelled everywhere, sealed in Bulbe
# ---------------------------------------------------------------------------
def test_hb6_a_glass_jar_is_labelled_on_every_form_and_sealed_in_bulbe(p, tmp_path):
    say = p.wording.say
    given = _jar(p, tmp_path, "hb6")
    factory = garden.garden(p, given, attended=True)
    produced = []

    def run(name, args, stream="stdout", code=0, input=None):
        result = garden.invoke(p, args, input=input, factory=factory)
        assert result.exit_code == code, (name, result.exit_code, result.stdout, result.stderr, result.exception)
        text = result.stdout if stream == "stdout" else result.stderr
        assert garden.says(text, p, "label.glass_jar"), (name, text)
        produced.append(name)
        return result

    sown = run("sow", ["sow"], input="Pip\nyes\n")
    rows = sown.stdout.splitlines()
    glass = garden.rows(p, say("sow.glass"))
    confirm = garden.rows(p, say("sow.ask.confirm"))
    assert rows.index(glass[0]) < rows.index(confirm[0]), "the jar is disclosed before the question"
    assert rows.index(confirm[0]) < rows.index(garden.rows(p, say("label.glass_jar"))[0]), "and labelled after"
    path = support.store_path(p, given, suffix=".glass.db")
    assert path.exists() and path.read_bytes()[:16] == support.MAGIC, "a glass jar: a store in clear"

    drawn = run("show ascii", ["show"])
    assert len(_art(drawn.stdout)) >= 5, ("witness: the probe finds the jar's drawing", drawn.stdout)
    run("show text", ["show", "--tier", "text"])
    served = run("show --json", ["show", "--json"])
    body = json.loads(served.stdout.strip().splitlines()[-1])
    assert "glass_jar" in body["labels"] and "label.glass_jar" in [line["key"] for line in body["lines"]], body
    assert body["habitat"] == {"container": "jar", "layer": "open"}, body
    run("lab", ["lab"])
    run("lab laws", ["lab", "laws"])
    run("keep verify", ["keep", "verify"])
    run("keep laws diff", ["keep", "laws", "diff"])
    run("care water", ["care", "water"])
    run("keep name", ["keep", "name"], input="Pip\n")
    run("keep laws pin", ["keep", "laws", "pin"])
    refused = run("a second pin", ["keep", "laws", "pin"], stream="stderr", code=1)
    assert garden.says(refused.stderr, p, "refuse.laws.pinned"), refused.stderr

    # In Bulbe the jar is sealed: said with no drawing, and a gesture and a law diff are refused.
    given["mode"].value = "bulbe"
    sealed = run("sealed show", ["show"])
    for key in ("label.bulbe", "status.sealed_bulbe"):
        assert garden.says(sealed.stdout, p, key), (key, sealed.stdout)
    assert _art(sealed.stdout) == [], _art(sealed.stdout)
    for name, args in (("sealed care water", ["care", "water"]), ("sealed diff", ["keep", "laws", "diff"])):
        result = run(name, args, stream="stderr", code=1)
        for key in ("label.bulbe", "status.sealed_bulbe"):
            assert garden.says(result.stderr, p, key), (name, key, result.stderr)
        assert result.stdout == "", (name, result.stdout)
    verify = garden.invoke(p, ["keep", "verify"], factory=factory)
    assert (verify.exit_code, verify.stdout) == (1, ""), (verify.stdout, verify.stderr, verify.exception)
    for key in ("label.glass_jar", "label.bulbe", "status.sealed_bulbe"):
        assert garden.says(verify.stderr, p, key), (key, verify.stderr)
    assert verify.stderr.startswith("Error: " + say("status.sealed_bulbe").text[:40]), verify.stderr
    laws = run("sealed lab laws", ["lab", "laws"])
    assert laws.stdout.startswith(garden.printed(p, say("doctrine"))), laws.stdout
    for key in ("label.bulbe", "status.sealed_bulbe"):
        assert garden.says(laws.stdout, p, key), (key, laws.stdout)
    assert len(produced) == 16 and len(set(produced)) == 16, produced


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
