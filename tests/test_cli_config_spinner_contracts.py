#!/usr/bin/env python3
"""Contracts for the ``oo`` CLI's configuration file and its spinner.

The CLI keeps apart two things it used to mix. The configuration a run
uses is the file plus the run's overrides -- ``NO_COLOR``, ``--no-color``,
``--api-url``, ``OO_API_URL`` -- while the file is only what the user wrote
there: ``oo config set`` edits the file, never the run. And a spinner owns
its line while it turns: nothing else is written to stderr until it has
stopped and erased it.

  * CK1 -- a command whose request fails stops its spinner before it says
    why: the error comes after the spinner's exit, on every command that
    waits behind one.
  * CK2 -- the spinner's exit stops its thread and clears its line before
    it returns, and nothing is drawn after it: what CK1's stand-in assumes.
  * CK3 -- ``oo config set`` writes the file's own values and the key it
    was given, ``oo config reset`` writes the defaults, and no override of
    the run reaches the file.
  * CK4 -- every key of cli.yaml is read alone: ``color: "false"`` is
    false, and an unreadable value falls back to its own default without
    touching the others.
  * CK5 -- ``oo config set`` refuses by name a colour, a timeout or an
    output format it cannot read, and a file it cannot read, leaving the
    file as it was.
  * CK6 -- ``oo ask --json-out`` waits behind a live spinner when colour
    is on, and stdout is exactly the JSON document.

Local-only (the public distribution ships no tests). The CLI modules are
loaded through the shared isolation window; the HTTP client, the spinner,
the error printer and stderr are seams.
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_CLI = ("config", "client", "output", "main")
_DEFAULT_URL = "http://localhost:8001"


def _path(tmp_path):
    return tmp_path / "xdg" / "opti-oignon" / "cli.yaml"


def _open(monkeypatch, tmp_path, cli_yaml=None):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in ("NO_COLOR", "OO_API_URL"):
        monkeypatch.delenv(name, raising=False)
    if cli_yaml is not None:
        _path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        _path(tmp_path).write_text(yaml.safe_dump(cli_yaml, sort_keys=False), encoding="utf-8")
    return isolate(
        targets={f"opti_oignon.cli.{m}": source("cli", f"{m}.py") for m in _CLI},
        packages=("opti_oignon.cli",),
    )


def _runner():
    from click.testing import CliRunner

    return CliRunner(mix_stderr=False)


def _file(tmp_path):
    return yaml.safe_load(_path(tmp_path).read_text(encoding="utf-8"))


class _Log:
    """One ordered record of what the spinner and the error printer did."""

    def __init__(self):
        self.events = []

    def spinner(self, message="", *, enabled=True):
        log = self

        class _Spinner:
            def __enter__(self):
                log.events.append(("enter", message, enabled))
                return self

            def __exit__(self, *exc):
                log.events.append(("exit", message))
                return False

        return _Spinner()

    def error(self, message, **_kw):
        self.events.append(("error", message))


class _Stderr:
    """A terminal stand-in for stderr: records every write, from any thread."""

    def __init__(self):
        self.writes = []
        self._lock = threading.Lock()

    def isatty(self):
        return True

    def write(self, text):
        with self._lock:
            self.writes.append(text)
        return len(text)

    def flush(self):
        pass


def _failing_client(main):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def _refuse(self, *args, **kwargs):
            raise main.CLIClientError("the backend refused")

        get = post = post_file = stream_chat = _refuse

    return _Client


def _waiting_commands(tmp_path):
    backup = tmp_path / "backup.json"
    backup.write_text(json.dumps({"settings": {}}), encoding="utf-8")
    note = tmp_path / "note.txt"
    note.write_text("An onion has layers.", encoding="utf-8")
    return (
        ["ask", "--json-out", "What is an onion?"],
        ["models"],
        ["status"],
        ["backup", "export", str(tmp_path / "exported.json")],
        ["backup", "import", str(backup)],
        ["rag", "ingest", str(note)],
        ["rag", "query", "What is an onion?"],
        ["redteam", "run"],
    )


# ---------------------------------------------------------------------------
# CK1 -- the error comes after the spinner
# ---------------------------------------------------------------------------
def test_ck1_a_failed_request_stops_the_spinner_before_the_error_is_written(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        main = loaded["opti_oignon.cli.main"]
        monkeypatch.setattr(main, "OOClient", _failing_client(main))
        commands = _waiting_commands(tmp_path)
        for args in commands:
            log = _Log()
            monkeypatch.setattr(main, "Spinner", log.spinner)
            monkeypatch.setattr(main, "echo_error", log.error)
            result = _runner().invoke(main.cli, args)
            kinds = [event[0] for event in log.events]
            assert result.exit_code == 1, (args, result.output)
            assert "enter" in kinds and "error" in kinds, f"{args}: the request waited behind a spinner and failed"
            assert kinds[-1] == "error" and kinds.count("enter") == kinds.count("exit"), (
                f"{args}: every spinner has stopped before the error is written: {log.events}"
            )
        assert len(commands) >= 8
    finally:
        restore()


# ---------------------------------------------------------------------------
# CK2 -- the real spinner's exit
# ---------------------------------------------------------------------------
def test_ck2_the_spinner_stops_and_clears_its_line_before_its_exit_returns(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        stderr = _Stderr()
        monkeypatch.setattr(sys, "stderr", stderr)
        spinner = out.Spinner("Working", enabled=True)
        with spinner:
            deadline = time.monotonic() + 3
            while not stderr.writes and time.monotonic() < deadline:
                time.sleep(0.01)
        drawn = list(stderr.writes)
        assert any("Working" in text for text in drawn), "the spinner drew while it turned"
        assert spinner._thread is not None and not spinner._thread.is_alive(), "its thread has stopped when the exit returns"
        assert drawn[-1] == "\r\033[K", "its line is cleared last"
        time.sleep(0.5)
        assert stderr.writes == drawn, "nothing is drawn after the exit"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CK3 -- the file, never the run
# ---------------------------------------------------------------------------
def test_ck3_config_set_and_reset_write_the_file_and_never_the_runs_overrides(monkeypatch, tmp_path):
    written = {"animations": False, "timeout": 30}
    loaded, restore = _open(monkeypatch, tmp_path, cli_yaml=written)
    try:
        main = loaded["opti_oignon.cli.main"]
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("OO_API_URL", "http://from-the-environment:1")
        result = _runner().invoke(main.cli, ["--no-color", "config", "set", "default_model", "llama3"])
        assert result.exit_code == 0, result.output
        assert _file(tmp_path) == {**written, "default_model": "llama3"}, "the file's own values and the key, nothing of the run"
        result = _runner().invoke(main.cli, ["--api-url", "http://from-the-flag:2", "config", "set", "timeout", "45"])
        assert result.exit_code == 0, result.output
        assert _file(tmp_path) == {**written, "default_model": "llama3", "timeout": 45}
        result = _runner().invoke(main.cli, ["--no-color", "--api-url", "http://from-the-flag:2", "config", "reset"])
        assert result.exit_code == 0, result.output
        reset = _file(tmp_path)
        assert reset["color"] is True and reset["api_url"] == _DEFAULT_URL, "reset writes the defaults, not the run's overrides"
        assert reset["animations"] is True and reset["timeout"] == 120 and "default_model" not in reset
    finally:
        restore()


# ---------------------------------------------------------------------------
# CK4 -- every key read alone
# ---------------------------------------------------------------------------
def test_ck4_every_key_of_cli_yaml_is_read_alone(monkeypatch, tmp_path):
    cases = (
        (
            {"color": "false", "timeout": "abc", "api_url": None, "default_model": 3, "output_format": "json", "animations": "no"},
            {"color": False, "timeout": 120, "api_url": _DEFAULT_URL, "default_model": None, "output_format": "json", "animations": False},
        ),
        (
            {"color": "off", "timeout": "45", "api_url": "http://remote:9000/", "default_model": "llama3"},
            {"color": False, "timeout": 45, "api_url": "http://remote:9000", "default_model": "llama3"},
        ),
        ({"color": "yes", "timeout": 0}, {"color": True, "timeout": 120}),
        ({"color": "banana", "timeout": True, "output_format": "xml"}, {"color": True, "timeout": 120, "output_format": "text"}),
        ({"color": False, "timeout": -5, "api_url": ""}, {"color": False, "timeout": 120, "api_url": _DEFAULT_URL}),
    )
    for written, expected in cases:
        loaded, restore = _open(monkeypatch, tmp_path, cli_yaml=written)
        try:
            cfg = loaded["opti_oignon.cli.config"].load_config()
            assert {key: getattr(cfg, key) for key in expected} == expected, f"{written!r}"
        finally:
            restore()


# ---------------------------------------------------------------------------
# CK5 -- refusals leave the file alone
# ---------------------------------------------------------------------------
def test_ck5_config_set_refuses_what_it_cannot_read_and_leaves_the_file(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path, cli_yaml={"timeout": 30})
    try:
        main = loaded["opti_oignon.cli.main"]
        path = _path(tmp_path)
        accepted = _runner().invoke(main.cli, ["config", "set", "color", "off"])
        assert accepted.exit_code == 0, accepted.output
        assert _file(tmp_path)["color"] is False
        before = path.read_bytes()
        for key, value in (("color", "banana"), ("timeout", "0"), ("timeout", "soon"), ("output_format", "xml")):
            refused = _runner().invoke(main.cli, ["config", "set", key, value])
            assert refused.exit_code == 1, (key, value, refused.output)
            assert key in refused.stderr, refused.stderr
            assert path.read_bytes() == before, f"{key}={value}: a refused value leaves the file as it was"
        accepted = _runner().invoke(main.cli, ["config", "set", "timeout", "45"])
        assert accepted.exit_code == 0 and _file(tmp_path)["timeout"] == 45
        path.write_text("- a list\n- not a mapping\n", encoding="utf-8")
        garbled = path.read_bytes()
        refused = _runner().invoke(main.cli, ["config", "set", "timeout", "45"])
        assert refused.exit_code == 1 and "cli.yaml" in refused.stderr, refused.stderr
        assert path.read_bytes() == garbled, "a file that cannot be read is not overwritten"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CK6 -- ask --json-out
# ---------------------------------------------------------------------------
def test_ck6_ask_json_out_waits_behind_a_live_spinner_and_prints_only_the_document(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        main = loaded["opti_oignon.cli.main"]

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            def stream_chat(self, text, model=None, **kwargs):
                return "An onion has layers."

        monkeypatch.setattr(main, "OOClient", _Client)
        document = {"model": "router", "prompt": "What is an onion?", "response": "An onion has layers."}
        for args, lit in (([], True), (["--no-color"], False)):
            log = _Log()
            monkeypatch.setattr(main, "Spinner", log.spinner)
            result = _runner().invoke(main.cli, [*args, "ask", "--json-out", "What is an onion?"])
            assert result.exit_code == 0, result.output
            assert result.stdout == json.dumps(document, indent=2) + "\n", "stdout is the document and nothing else"
            assert [e for e in log.events if e[0] == "enter"] == [("enter", "Generating", lit)], "the wait is shown when colour is on"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
