#!/usr/bin/env python3
"""Contracts for the launcher's frontend install: the lock decides.

``python -m opti_oignon`` (``opti_oignon/ui.py``) starts the frontend's dev server
from ``frontend/node_modules``. It used to install only when that directory
was missing, so a pull that changed ``frontend/package-lock.json`` (a
dependency added, such as the reply renderer's lexer) left an install that
no longer held what the code imports, while the launcher still said the
frontend was ready: the dev server's output goes nowhere.

npm records what it installed in ``node_modules/.package-lock.json``, the
hidden lockfile. It holds every package of the lock, less the lock's root
entry and the optional packages npm skipped because they are built for
another system (47 of 311 on this machine), so the two files are never equal
byte for byte; they are compared package by package. A build is named by its
version and, when the lock pins one, its digest: the same bytes fetched
through a registry mirror carry the mirror's source, and are the same build.
A package the lock pins without a digest (a git or a local one) is named by
its source instead.

  * LN1 -- the launcher holds the install to the lock before any port is
    touched: when ``node_modules`` is missing, when its hidden lockfile is
    missing or unreadable, or when a package differs from the lock (another
    version or digest, another source for a package pinned without a digest,
    one the lock pins and the install lacks, unless npm skips it as an
    optional or dev-optional package, or one the install holds and the lock
    no longer pins), it runs ``npm ci`` in ``frontend/`` with npm's output
    left on the terminal, and says why; when they agree, a mirror's source
    for a pinned digest included, it runs nothing; when ``npm ci`` fails, or
    npm cannot be run, it says so and stops. Its own body starts the dev
    server and runs no other npm command.

The launcher is loaded through the shared isolation window and driven up to
the step after the install (freeing the ports, which raises here), with
``subprocess``, the Ollama check and the signal handlers stood in for:
nothing is installed, no port is touched, nothing is reached on the network.

Local-only (the public distribution ships no tests).
"""

import ast
import copy
import json
import signal
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
_UI = REPO / "opti_oignon" / "ui.py"

_LOCK = {
    "name": "opti-oignon-frontend",
    "version": "2.2.0",
    "lockfileVersion": 3,
    "requires": True,
    "packages": {
        "": {
            "name": "opti-oignon-frontend",
            "version": "2.2.0",
            "dependencies": {"marked": "15.0.11"},
            "devDependencies": {"vite": "^5.4.0"},
        },
        "node_modules/marked": {
            "version": "15.0.11",
            "resolved": "https://registry.npmjs.org/marked/-/marked-15.0.11.tgz",
            "integrity": "sha512-marked-digest",
            "license": "MIT",
        },
        "node_modules/vite": {
            "version": "5.4.21",
            "resolved": "https://registry.npmjs.org/vite/-/vite-5.4.21.tgz",
            "integrity": "sha512-vite-digest",
            "dev": True,
        },
        "node_modules/@esbuild/darwin-arm64": {
            "version": "0.21.5",
            "resolved": "https://registry.npmjs.org/@esbuild/darwin-arm64/-/darwin-arm64-0.21.5.tgz",
            "integrity": "sha512-esbuild-digest",
            "cpu": ["arm64"],
            "dev": True,
            "optional": True,
            "os": ["darwin"],
        },
        "node_modules/fsevents": {
            "version": "2.3.3",
            "resolved": "https://registry.npmjs.org/fsevents/-/fsevents-2.3.3.tgz",
            "integrity": "sha512-fsevents-digest",
            "devOptional": True,
            "os": ["darwin"],
        },
        "node_modules/local-helper": {
            "version": "1.0.0",
            "resolved": "git+ssh://git@git.test/helper.git#aaaaaaaa",
        },
    },
}


def _hidden(lock):
    """The hidden lockfile npm writes for ``lock`` on a system the optional
    package is not built for."""
    hidden = copy.deepcopy(lock)
    packages = hidden["packages"]
    del packages[""]
    del packages["node_modules/@esbuild/darwin-arm64"]
    del packages["node_modules/fsevents"]
    return hidden


class _Stopped(Exception):
    """The launch reached the step after the install: freeing the ports."""


class _Run:
    """``subprocess.run`` stood in for: records each call and answers with
    ``returncode``, or raises ``raises``."""

    def __init__(self, returncode=0, raises=None):
        self.calls = []
        self.returncode = returncode
        self.raises = raises

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(command, self.returncode, "", "")


def _refuse_popen(*_args, **_kwargs):
    raise AssertionError("the launch started a process before the ports step")


def _launch(root, run):
    """Drives the real ``launch`` over the project at ``root`` until it frees
    the ports. Returns ``"ports"`` when it got there, or the exit code it
    stopped with."""
    loaded, restore = isolate(targets={"opti_oignon.ui": _UI})
    try:
        ui = loaded["opti_oignon.ui"]
        ui._find_project_root = lambda: root
        ui._check_ollama = lambda: True

        def stop(*_args, **_kwargs):
            raise _Stopped()

        ui._free_port = stop
        ui.signal = types.SimpleNamespace(
            signal=lambda *_args: None, SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM,
            SIGKILL=signal.SIGKILL,
        )
        ui.subprocess = types.SimpleNamespace(
            run=run, Popen=_refuse_popen, DEVNULL=subprocess.DEVNULL,
            PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired,
        )
        try:
            ui.launch()
        except _Stopped:
            return "ports"
        except SystemExit as exc:
            return exc.code
        return "returned"
    finally:
        restore()


def _project(tmp, lock=_LOCK, hidden=None, modules=True):
    """A project root with ``frontend/package-lock.json`` and, when asked,
    ``node_modules`` holding ``hidden`` as its hidden lockfile (a text is
    written as it is)."""
    root = Path(tmp)
    (root / "opti_oignon").mkdir()
    frontend = root / "frontend"
    frontend.mkdir()
    (frontend / "package-lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    if modules:
        (frontend / "node_modules").mkdir()
        if hidden is not None:
            text = hidden if isinstance(hidden, str) else json.dumps(hidden, indent=2)
            (frontend / "node_modules" / ".package-lock.json").write_text(text, encoding="utf-8")
    return root


def _with(packages, path, **fields):
    changed = copy.deepcopy(packages)
    changed["packages"][path].update(fields)
    return changed


def _without(packages, path):
    changed = copy.deepcopy(packages)
    del changed["packages"][path]
    return changed


def _stale_cases():
    hidden = _hidden(_LOCK)
    extra = copy.deepcopy(hidden)
    extra["packages"]["node_modules/left-pad"] = {
        "version": "1.3.0", "resolved": "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz",
        "integrity": "sha512-left-pad-digest",
    }
    return {
        "a package added to the lock": ({}, _without(hidden, "node_modules/marked"), "marked"),
        "another version": ({}, _with(hidden, "node_modules/marked", version="15.0.10"), "marked"),
        "another digest": ({}, _with(hidden, "node_modules/vite", integrity="sha512-other"), "vite"),
        "another source for a package pinned without a digest": (
            {}, _with(hidden, "node_modules/local-helper",
                      resolved="git+ssh://git@git.test/helper.git#bbbbbbbb"), "local-helper",
        ),
        "a package the lock no longer pins": ({}, extra, "left-pad"),
        "no hidden lockfile": ({}, None, ".package-lock.json"),
        "an unreadable hidden lockfile": ({}, "{ not json", ".package-lock.json"),
        "no node_modules": ({"modules": False}, None, "node_modules"),
    }


def test_ln1_the_launcher_holds_the_frontend_install_to_the_lock(capsys):
    # Its own body starts the dev server and runs no other npm command.
    launch = next(
        node for node in ast.parse(_UI.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "launch"
    )
    npm_commands = [
        [element.value for element in node.elts if isinstance(element, ast.Constant)]
        for node in ast.walk(launch)
        if isinstance(node, ast.List) and node.elts
        and isinstance(node.elts[0], ast.Constant) and node.elts[0].value == "npm"
    ]
    assert npm_commands == [["npm", "run", "dev", "--", "--port"]], (
        f"launch() starts the dev server and runs no npm command of its own: {npm_commands}"
    )

    # The install is the lock's: nothing runs, and the launch goes on.
    with tempfile.TemporaryDirectory() as tmp:
        run = _Run()
        assert _launch(_project(tmp, hidden=_hidden(_LOCK)), run) == "ports"
    assert run.calls == [], f"an install that is the lock's runs nothing: {run.calls}"
    assert "npm ci" not in capsys.readouterr().out

    # The same build, whatever source npm recorded for it: a mirror's source
    # for the digest the lock pins runs nothing either.
    mirrored = _with(_hidden(_LOCK), "node_modules/vite", resolved="https://mirror.test/vite.tgz")
    with tempfile.TemporaryDirectory() as tmp:
        run = _Run()
        assert _launch(_project(tmp, hidden=mirrored), run) == "ports"
    assert run.calls == [], f"a mirror's source for the same digest runs nothing: {run.calls}"
    assert "npm ci" not in capsys.readouterr().out

    # Each way the install can differ from the lock runs npm ci, once, in
    # frontend/, with npm's output on the terminal, and says why.
    for case, (layout, hidden, named) in _stale_cases().items():
        with tempfile.TemporaryDirectory() as tmp:
            root = _project(tmp, hidden=hidden, **layout)
            run = _Run()
            assert _launch(root, run) == "ports", f"{case}: the launch goes on after npm ci"
            assert [command for command, _kwargs in run.calls] == [["npm", "ci"]], (
                f"{case}: npm ci runs once: {run.calls}"
            )
            kwargs = run.calls[0][1]
            assert Path(kwargs.get("cwd", "")).resolve() == (root / "frontend").resolve(), (
                f"{case}: npm ci runs in frontend/: {kwargs}"
            )
            assert not kwargs.get("capture_output") and not {
                kwargs.get("stdout"), kwargs.get("stderr"),
            } & {subprocess.DEVNULL, subprocess.PIPE}, (
                f"{case}: npm's output stays on the terminal: {kwargs}"
            )
        said = capsys.readouterr().out
        assert "npm ci" in said and named in said, (
            f"{case}: the launcher says it runs npm ci, and why ({named}): {said!r}"
        )

    # A failed npm ci, or npm that cannot be run, stops the launch and says so.
    stale = _without(_hidden(_LOCK), "node_modules/marked")
    for case, run, words in (
        ("npm ci fails", _Run(returncode=1), "failed"),
        ("npm cannot be run", _Run(raises=FileNotFoundError(2, "No such file", "npm")), "npm"),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = _launch(_project(tmp, hidden=stale), run)
        said = capsys.readouterr().out
        assert outcome == 1, f"{case}: the launch stops with exit 1 before the ports: {outcome!r}"
        assert [command for command, _kwargs in run.calls] == [["npm", "ci"]]
        assert "[ERR]" in said and words in said, f"{case}: the launcher says so: {said!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly"]))
