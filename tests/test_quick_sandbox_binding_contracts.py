#!/usr/bin/env python3
"""Contracts for the conversation-binding bridge of the quick sandbox.

  * Contract 1 -- adoption: a session built for a bound workspace routes
    its tool calls to that existing workspace (uploaded files are
    readable, no fresh sandbox is created), and falls back to a fresh
    sandbox -- still fully functional -- when the bound workspace is
    gone.
  * Contract 2 -- announcement: on adoption the session rehydrates its
    file list from the adopted workspace (bounded by the announce cap)
    so the registry tools prompt tells the model which files are
    already there.
  * Contract 3 -- the manager bridge: get_or_create_session with a
    bound workspace returns an adopting session, is idempotent for the
    same pair, keeps plain conversations on fresh sandboxes, and a
    rebinding replaces the live session immediately.
  * Contract 4 -- ownership: destroying or expiring an adopting session
    never destroys the bound workspace (the binding layer owns it),
    while a session-owned sandbox is destroyed as before.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. The sandbox manager and the low-level
file handlers are replaced by in-memory stand-ins so every behavior is
deterministic and no real sandbox is ever spawned.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# In-memory sandbox world
# ---------------------------------------------------------------------------
class FakeSandbox:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.active = True
        self.owner_user_id = "local"
        self.files: dict[str, str] = {}


class FakeManager:
    def __init__(self):
        self.sessions: dict[str, FakeSandbox] = {}
        self.create_calls: list[str] = []
        self.destroy_calls: list[str] = []

    def seed(self, session_id: str, files: dict[str, str]) -> FakeSandbox:
        box = FakeSandbox(session_id)
        box.files.update(files)
        self.sessions[session_id] = box
        return box

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def create_sandbox(self, session_id: str, allow_degraded: bool = True):
        self.create_calls.append(session_id)
        box = FakeSandbox(session_id)
        self.sessions[session_id] = box
        return box

    def destroy_sandbox(self, session_id: str) -> bool:
        self.destroy_calls.append(session_id)
        box = self.sessions.pop(session_id, None)
        if box is not None:
            box.active = False
            return True
        return False

    def extract_files(self, session_id: str):
        box = self.sessions.get(session_id)
        if box is None:
            return []
        return [{"path": name} for name in sorted(box.files)]


def _fake_view(session_id, path, start_line=0, end_line=0,
               _sandbox_manager=None):
    box = _sandbox_manager.get_session(session_id)
    if box is None:
        return f"Error: unknown session {session_id}"
    if path in (".", ""):
        listing = "\n".join(sorted(box.files))
        return listing if listing else "(empty workspace)"
    if path in box.files:
        return box.files[path]
    return f"Error: Path not found: {path}"


def _fake_create_file(session_id, path, content, _sandbox_manager=None):
    box = _sandbox_manager.get_session(session_id)
    if box is None:
        return f"Error: unknown session {session_id}"
    box.files[path] = content
    return f"File created: {path}"


def _fake_bash(session_id, command, timeout=30, _sandbox_manager=None):
    return "Command success (return code: 0)"


# ---------------------------------------------------------------------------
# Isolated loading
# ---------------------------------------------------------------------------
def _load():
    keys = (
        "opti_oignon", "opti_oignon.sandbox_manager",
        "opti_oignon.file_tools", "opti_oignon.tool_calling",
        "opti_oignon.tool_registry", "opti_oignon.quick_sandbox",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    sm = types.ModuleType("opti_oignon.sandbox_manager")
    sm.SANDBOX_AVAILABLE = True
    sm.SandboxManager = FakeManager
    sm.SandboxSession = FakeSandbox
    sm.sandbox_manager = None
    sys.modules["opti_oignon.sandbox_manager"] = sm
    pkg.sandbox_manager = sm

    ft = types.ModuleType("opti_oignon.file_tools")
    ft.FILE_TOOLS_AVAILABLE = True
    ft._handle_sandbox_bash = _fake_bash
    ft._handle_sandbox_view = _fake_view
    ft._handle_sandbox_create_file = _fake_create_file
    sys.modules["opti_oignon.file_tools"] = ft
    pkg.file_tools = ft

    def _real(dotted: str, path: Path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg.tool_calling = _real(
        "opti_oignon.tool_calling", _OO / "tool_calling.py",
    )
    tr = _real("opti_oignon.tool_registry", _OO / "tool_registry.py")
    pkg.tool_registry = tr
    qs = _real("opti_oignon.quick_sandbox", _OO / "quick_sandbox.py")
    pkg.quick_sandbox = qs

    if not qs.QUICK_SANDBOX_AVAILABLE:
        raise RuntimeError("quick sandbox reports unavailable under stubs")

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return qs, tr, restore


# ---------------------------------------------------------------------------
# Contract 1 -- adoption routes tool calls to the bound workspace
# ---------------------------------------------------------------------------
def test_c1_adoption_and_fallback():
    qs, _tr, restore = _load()
    try:
        mgr = FakeManager()
        mgr.seed("ws1", {"notes.txt": "ping"})

        session = qs.QuickSandboxSession(
            "conv1", sandbox_mgr=mgr, existing_sandbox_id="ws1",
        )
        assert session.bound_sandbox_id == "ws1"
        assert session.handle_read_file("notes.txt") == "ping"
        assert mgr.create_calls == [], mgr.create_calls
        assert session.active is True
        assert session.session_id == "conv1"
        listing = session.handle_list_files(".")
        assert "notes.txt" in listing

        # Fallback: the bound workspace is gone -- a fresh sandbox is
        # created and the session stays fully functional.
        orphan = qs.QuickSandboxSession(
            "conv2", sandbox_mgr=mgr, existing_sandbox_id="ghost",
        )
        missing = orphan.handle_read_file("notes.txt")
        assert missing.startswith("Error: Path not found"), missing
        assert mgr.create_calls == ["conv2"], mgr.create_calls
        assert orphan.handle_write_file("own.txt", "mine") == (
            "File created: own.txt"
        )
        assert orphan.handle_read_file("own.txt") == "mine"
        assert orphan.bound_sandbox_id == "ghost"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- rehydration feeds the tools-prompt announcement
# ---------------------------------------------------------------------------
def test_c2_rehydration_and_announcement():
    qs, tr, restore = _load()
    try:
        mgr = FakeManager()
        mgr.seed("ws1", {"notes.txt": "ping", "brief.md": "hello"})

        session = qs.QuickSandboxSession(
            "conv1", sandbox_mgr=mgr, existing_sandbox_id="ws1",
        )
        session.handle_list_files(".")  # adoption is lazy: first call
        assert session.files_created == ["brief.md", "notes.txt"], (
            session.files_created
        )

        registry = tr.ToolRegistry()
        # The quick-sandbox mode replaces the handlers of EXISTING unsafe
        # tools; the tools prompt short-circuits on an empty registry, so
        # register them first, exactly as production does at boot.
        for _name in ("read_file", "write_file", "list_files"):
            registry.register(tr.ToolDefinition(
                name=_name,
                description=f"{_name} in the workspace.",
                parameters={},
                handler=lambda *a, **k: "",
            ))
        registry.set_quick_sandbox_mode(True, session=session)
        prompt = registry.get_tools_prompt()
        assert "Files already in your sandbox workspace: " in prompt
        assert "notes.txt" in prompt and "brief.md" in prompt

        # The announcement is bounded by the cap; the workspace itself
        # stays fully listable through the tools.
        cap = qs.QuickSandboxSession.ADOPTED_FILES_ANNOUNCE_CAP
        big = mgr.seed(
            "ws2", {f"data_{i:03d}.txt": "x" for i in range(cap + 10)},
        )
        assert len(big.files) == cap + 10
        wide = qs.QuickSandboxSession(
            "conv3", sandbox_mgr=mgr, existing_sandbox_id="ws2",
        )
        wide.handle_list_files(".")
        assert len(wide.files_created) == cap, len(wide.files_created)
        assert wide.files_created == sorted(big.files)[:cap]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- the manager bridge and immediate rebinding
# ---------------------------------------------------------------------------
def test_c3_manager_bridge_and_rebind():
    qs, _tr, restore = _load()
    try:
        mgr = FakeManager()
        mgr.seed("ws1", {"notes.txt": "ping"})
        manager = qs.QuickSandboxManager(sandbox_mgr=mgr)

        bound = manager.get_or_create_session(
            "convA", bound_sandbox_id="ws1",
        )
        assert bound.handle_read_file("notes.txt") == "ping"
        assert mgr.create_calls == [], mgr.create_calls
        again = manager.get_or_create_session(
            "convA", bound_sandbox_id="ws1",
        )
        assert again is bound

        # A plain conversation still gets its own fresh sandbox.
        plain = manager.get_or_create_session("convB")
        plain.handle_write_file("draft.txt", "wip")
        assert mgr.create_calls == ["convB"], mgr.create_calls
        assert plain.bound_sandbox_id is None

        # Rebinding replaces the live session immediately: the old own
        # sandbox is destroyed, the new session adopts the workspace.
        rebound = manager.get_or_create_session(
            "convB", bound_sandbox_id="ws1",
        )
        assert rebound is not plain
        assert "convB" in mgr.destroy_calls
        assert rebound.handle_read_file("notes.txt") == "ping"
        assert manager.get_or_create_session(
            "convB", bound_sandbox_id="ws1",
        ) is rebound
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- the bound workspace survives destroy and expiry
# ---------------------------------------------------------------------------
def test_c4_ownership_on_destroy_and_expiry():
    qs, _tr, restore = _load()
    try:
        mgr = FakeManager()
        mgr.seed("ws1", {"notes.txt": "ping"})
        manager = qs.QuickSandboxManager(sandbox_mgr=mgr)

        adopted = manager.get_or_create_session(
            "convA", bound_sandbox_id="ws1",
        )
        assert adopted.handle_read_file("notes.txt") == "ping"
        assert adopted.destroy() is True
        assert mgr.destroy_calls == [], mgr.destroy_calls
        assert mgr.get_session("ws1").active is True
        assert adopted.active is False

        # A session-owned sandbox is destroyed as before.
        own = qs.QuickSandboxSession("convO", sandbox_mgr=mgr)
        own.handle_write_file("draft.txt", "wip")
        assert own.destroy() is True
        assert mgr.destroy_calls == ["convO"], mgr.destroy_calls

        # Expiry cleanup detaches an adopting session without touching
        # the workspace; the next request re-adopts and still reads it.
        expiring = manager.get_or_create_session(
            "convE", bound_sandbox_id="ws1",
        )
        expiring.handle_read_file("notes.txt")
        expiring._last_activity -= expiring._auto_destroy_seconds + 1
        assert expiring.expired is True
        fresh = manager.get_or_create_session(
            "convE", bound_sandbox_id="ws1",
        )
        assert fresh is not expiring
        assert "ws1" not in mgr.destroy_calls
        assert mgr.get_session("ws1").active is True
        assert fresh.handle_read_file("notes.txt") == "ping"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
