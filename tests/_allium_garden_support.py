#!/usr/bin/env python3
"""Shared support for the garden's contracts: the window with the terminal, the test factory, the runner.

A garden contract opens the platform window with the terminal's modules
loaded after it (``open_garden``), builds its store seams with
``_allium_store_support.seams`` and hands ``oo garden`` a factory of its
own (``garden``): each command then builds one ``Garden`` over a fresh store
on the same seams -- a fresh process, as far as the store can tell -- with
the switch, the stop, the terminal test and the law injected. ``invoke``
runs ``oo garden`` through click's runner with that factory in the root
object, colour off unless asked.

The CLI reads its configuration file when its module loads, so the window
points ``XDG_CONFIG_HOME`` into the test's own directory and drops
``NO_COLOR`` and ``OO_API_URL`` before anything is loaded.

``look`` builds a synthetic ``Look`` for the pure scans; ``felt`` and
``being_info`` its parts. ``FileAudit`` is the in-memory audit log written
through to a JSON file, so a child process reads the same entries.
``child`` starts ``python -c`` in the hermetic environment of the footprint
contracts: an empty working directory, the configuration under the test's
directory, no colour, API, key or passphrase variable, no bytecode written.

The rendering helpers compare what a stream holds with what the catalogue
renders: ``rows`` wraps lines as the terminal does, ``shows`` finds a key's
rows whole in a stream, and ``flat`` reads a stream as one line of words, so
a refusal on stderr is found whatever its wrapping and its ``Error:``
prefix.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_store_support as support  # noqa: E402
from _allium_window import open_allium  # noqa: E402
from _isolation import REPO  # noqa: E402

# The exit of the show form for each status, as the served state machine gives it (frozen aside).
EXITS = {"disabled": 0, "stopped": 0, "unavailable": 1, "awaiting_soil": 2, "ready": 0, "sealed_bulbe": 0,
         "missing": 1, "retired_prototype": 0, "unreadable": 1, "resting": 0, "alive": 0}
# The variables a child process never inherits.
UNSET = ("NO_COLOR", "OO_API_URL", "OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE")
TIMEOUT_S = 60


class Window(support.Platform):
    """The modules of a garden window: the platform, and the terminal that drives it."""

    def __init__(self, loaded):
        super().__init__(loaded)
        self.main = loaded["opti_oignon.cli.main"]
        self.cli_garden = loaded["opti_oignon.cli.garden"]
        self.output = loaded["opti_oignon.cli.output"]


def open_garden(monkeypatch, tmp_path, **kw):
    """One window with the platform and the terminal loaded; ``(Window, restore)``.

    ``kw`` goes to ``open_allium``: ``blocked`` defaults to the store
    support's list, and a contract that seeds a platform dependency leaves
    it out.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(Path(tmp_path) / "xdg"))
    for name in ("NO_COLOR", "OO_API_URL"):
        monkeypatch.delenv(name, raising=False)
    kw.setdefault("blocked", support.BLOCKED)
    loaded, restore = open_allium(native=False, platform=True, cli=True, **kw)
    return Window(loaded), restore


class Factory:
    """The test's ``allium_service``: a new ``Garden`` per command, over a new store on the same seams.

    ``attended`` and ``switch`` are values or callables, read when the
    garden asks; ``stopped`` is the garden's own seam (``None`` for none).
    Every call is counted.
    """

    def __init__(self, p, given, *, attended=False, law="fixture", switch="on", stopped=None):
        self.p = p
        self.given = given
        self.attended = attended
        self.law = law
        self.switch = switch
        self.stopped = stopped
        self.calls = 0
        self.stores = 0
        self.attended_reads = 0
        self.switch_reads = 0

    def __call__(self):
        self.calls += 1
        return self.p.service.Garden(store_factory=self._store, switch=self._switch, stopped=self.stopped,
                                     attended=self._attended, law=self.law)

    def _store(self):
        self.stores += 1
        return self.p.store.Store(**self.given)

    def _switch(self):
        self.switch_reads += 1
        return self.switch() if callable(self.switch) else self.switch

    def _attended(self):
        self.attended_reads += 1
        return self.attended() if callable(self.attended) else self.attended


def garden(p, given, *, attended=False, law="fixture", switch="on", stopped=None):
    """The test factory for ``oo garden``, over the store seams ``given``."""
    return Factory(p, given, attended=attended, law=law, switch=switch, stopped=stopped)


def invoke(p, args, input=None, factory=None, color=False):
    """``oo garden ARGS`` through click's runner: stdout and stderr apart, the factory in the root object."""
    from click.testing import CliRunner

    runner = CliRunner(mix_stderr=False)
    argv = ([] if color else ["--no-color"]) + ["garden", *args]
    obj = {} if factory is None else {"allium_service": factory}
    return runner.invoke(p.main.cli, argv, input=input, obj=obj, color=color)


# ---------------------------------------------------------------------------
# Synthetic records
# ---------------------------------------------------------------------------
def felt(p, **fields):
    """A ``Felt`` of a seed on its day 0 in an autumn garden, full sun and wet soil, with ``fields`` changed."""
    values = {"name": None, "day": 0, "season": 3, "place": "garden", "light": "up", "soil": "wet", "life": "awake",
              "minute": 533, "daylength": 641, "sun": 65536, "sun_max": 65536, "jar": False, "layer": "open",
              "labels": ("prototype",), "local": "2025-10-09 08:53", "offset": "+00:00"}
    values.update(fields)
    return p.describe.Felt(**values)


def being_info(p, **fields):
    """A ``BeingInfo`` of a fixture being in an encrypted pot, with ``fields`` changed."""
    values = {"name": None, "soil": "encrypted", "law": "fixture", "v": 0, "provisional": True,
              "digest": "84f750633e8e", "tag": "0a1b2c3d", "events": 3, "weather": "garden"}
    values.update(fields)
    return p.service.BeingInfo(**values)


def look(p, status, **fields):
    """A synthetic ``Look`` of ``status``: every field absent unless given, the exit the status serves.

    What a status cannot be served without is filled: an alive being's
    habitat, felt state, identity and label, and the reason a store could
    not be opened (a terminal with no account, the one refused before any
    store is read).
    """
    values = {"status": status, "labels": (), "mode": "daily", "habitat": None, "view": None, "felt": None,
              "being": None, "reason": None, "offer": None, "seq": None, "exit": EXITS[status],
              "glass_allowed": False}
    if status == "alive":
        values.update(habitat=("pot", "open"), felt=felt(p), being=being_info(p), labels=("prototype",))
    if status == "unavailable":
        values.update(reason="account")
    values.update(fields)
    return p.service.Look(**values)


# ---------------------------------------------------------------------------
# The audit log a child reads
# ---------------------------------------------------------------------------
class FileAudit(support.MemoryAudit):
    """``MemoryAudit`` written through to a JSON file after every append, and read from it when it exists."""

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="ascii"))

    def append_event(self, *args, **kwargs):
        entry_id = super().append_event(*args, **kwargs)
        self.path.write_text(json.dumps(self.entries, sort_keys=True), encoding="ascii")
        return entry_id


# ---------------------------------------------------------------------------
# Children
# ---------------------------------------------------------------------------
def child_env(tmp_path):
    env = {key: value for key, value in os.environ.items() if key not in UNSET}
    env["XDG_CONFIG_HOME"] = str(Path(tmp_path) / "xdg")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # A child runs from an empty directory, where the package would resolve
    # through whatever install points at a checkout, which need not be the
    # tree under test (a second worktree is the case that showed it). The
    # tree's own root goes first.
    env["PYTHONPATH"] = os.pathsep.join([str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def child(tmp_path, code, *, cwd):
    """Start ``python -c code`` in the hermetic environment, in the empty directory ``cwd``; the process.

    The caller waits with ``finish``, so several children can run at once.
    """
    Path(cwd).mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([sys.executable, "-c", code], cwd=str(cwd), env=child_env(tmp_path),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(process):
    """``(returncode, stdout, stderr, report)``: ``report`` is the JSON of the last stderr line, or ``None``."""
    try:
        out, err = process.communicate(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        out, err = process.communicate()
        raise AssertionError("a child process ran past its timeout: " + err[-800:]) from None
    report = None
    lines = [line for line in err.splitlines() if line.strip()]
    if lines:
        try:
            report = json.loads(lines[-1])
        except ValueError:
            report = None
    return process.returncode, out, err, report


def canary(p, suite):
    return support.canary(p, suite)


# ---------------------------------------------------------------------------
# What a stream holds
# ---------------------------------------------------------------------------
def rows(p, *lines):
    """The rows the terminal prints for ``lines`` (``Line`` records), wrapped as it wraps them."""
    return p.describe.wrap(list(lines))


def printed(p, *lines):
    """The exact text ``lines`` print on one stream: each row, then a line end."""
    return "".join(row + "\n" for row in rows(p, *lines))


def shows(stream, p, key, /, **slots):
    """Whether ``stream`` holds the rows the catalogue's ``key`` prints, whole, consecutive and unprefixed."""
    wanted = rows(p, p.wording.say(key, **slots))
    lines = stream.splitlines()
    return any(lines[i:i + len(wanted)] == wanted for i in range(len(lines) - len(wanted) + 1))


def flat(stream):
    """A stream as one line of words, its ``Error:`` prefix dropped: what a refusal says, however it wraps."""
    return " ".join(stream.replace("Error: ", "", 1).split())


def says(stream, p, key, /, **slots):
    """Whether ``stream`` says the catalogue's ``key``, wherever its rows break."""
    return " ".join(p.wording.say(key, **slots).text.split()) in flat(stream)


def sha256(path):
    return support.sha256_file(path)


__all__ = ["REPO", "open_garden", "garden", "invoke", "felt", "being_info", "look", "FileAudit", "child",
           "finish", "canary", "rows", "printed", "shows", "flat", "says", "sha256", "Window", "Factory", "EXITS"]
