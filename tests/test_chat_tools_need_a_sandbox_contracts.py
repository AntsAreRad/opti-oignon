#!/usr/bin/env python3
"""The chat's file and code tools have no host-side handler.

``execute_code``, ``read_file``, ``write_file`` and ``list_files`` run only
inside a sandbox session. When none is attached to the conversation (the
quick sandbox is off, unavailable, or failed to start), each one refuses,
names the sandbox as the remedy, and touches nothing on this machine:

  * NS1 -- ``read_file`` returns no byte of a host file;
  * NS2 -- ``list_files`` lists no host directory entry;
  * NS3 -- ``write_file`` creates no host file, absolute path included;
  * NS4 -- ``execute_code`` refuses rather than running anything;
  * NS5 -- the quick sandbox routes the tools to its session for the turn,
    and switching it off brings the refusals back, not a host handler.

The registry module is loaded from its file in an isolation window and a
fresh registry is built for each contract. Local-only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.tool_registry"


def _registry():
    loaded, restore = isolate(targets={_TARGET: source("tool_registry.py")})
    module = loaded[_TARGET]
    registry = module.ToolRegistry()
    module._register_builtin_tools(registry)
    return registry, restore


def _reply(name, **args):
    registry, restore = _registry()
    try:
        return registry.get(name).handler(**args)
    finally:
        restore()


def _refused(reply):
    return reply.startswith("Refused:") and "sandbox" in reply


class _Session:
    """A quick sandbox session stand-in that answers from the sandbox."""

    def handle_read_file(self, path):
        return f"sandboxed read {path}"

    def handle_write_file(self, path, content):
        return f"sandboxed write {path}"

    def handle_list_files(self, path="."):
        return f"sandboxed list {path}"

    def handle_execute_code(self, code, language="python", timeout=30):
        return "sandboxed run"


def test_ns1_read_file_without_a_sandbox_session_reads_nothing_on_the_host(tmp_path):
    secret = tmp_path / "ns1-host.txt"
    secret.write_text("ns1-host-bytes", encoding="utf-8")
    reply = _reply("read_file", path=str(secret))
    assert "ns1-host-bytes" not in reply, "a host file reached the model"
    assert _refused(reply), reply


def test_ns2_list_files_without_a_sandbox_session_lists_nothing_on_the_host(tmp_path):
    (tmp_path / "ns2-host-entry.txt").write_text("", encoding="utf-8")
    reply = _reply("list_files", path=str(tmp_path))
    assert "ns2-host-entry" not in reply, "a host directory was listed"
    assert _refused(reply), reply


def test_ns3_write_file_without_a_sandbox_session_writes_nothing_on_the_host(tmp_path):
    target = tmp_path / "ns3-written.txt"
    reply = _reply("write_file", path=str(target), content="ns3")
    assert not target.exists(), "a host file was written"
    assert _refused(reply), reply


def test_ns4_execute_code_without_a_sandbox_session_refuses():
    reply = _reply("execute_code", code="print('ns4')")
    assert _refused(reply), reply


def test_ns5_switching_the_quick_sandbox_off_restores_the_refusals(tmp_path):
    secret = tmp_path / "ns5-host.txt"
    secret.write_text("ns5-host-bytes", encoding="utf-8")
    registry, restore = _registry()
    try:
        registry.set_quick_sandbox_mode(True, session=_Session())
        inside = registry.get("read_file").handler(path=str(secret))
        registry.set_quick_sandbox_mode(False)
        after = registry.get("read_file").handler(path=str(secret))
    finally:
        restore()
    assert inside == f"sandboxed read {secret}", inside
    assert "ns5-host-bytes" not in after and _refused(after), after
