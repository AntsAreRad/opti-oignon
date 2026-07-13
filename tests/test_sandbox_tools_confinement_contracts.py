#!/usr/bin/env python3
"""Sandbox tool-layer contracts: locked registry, confined reads, no exits.

The tool layer is the surface the model actually sees: four workspace tools
with the session identity pre-bound, plus read-only search helpers. While a
workspace is active the shared registry is switched into its locked mode so
the unsandboxed equivalents cannot be called around the workspace, and the
optional web-search shutoff honours the manager's configuration. The layer
exposes no exit toward the host: no approval, no copy-out, no clone, no
browse -- those live behind the human-driven facade only. Every read-only
helper resolves its path through the shared validator, never follows a
symbolic link, keeps results inside the workspace by real path, and clamps
its bounds. This suite pins that behavior:

  * TC1 -- starting a session locks the registry (the unsafe tools go
    disabled), the generated identity is unique and non-trivial, and
    stopping destroys the workspace and restores the registry;
  * TC2 -- attaching to an existing workspace applies the identical lock,
    and detaching releases the lock without destroying anything;
  * TC3 -- the definitions handed to the model are exactly the four
    session-bound workspace tools, each routing through the shared
    handlers with the bound identity, and the wrapper exposes no
    host-exit surface;
  * TC4 -- with the manager configured to shut off web search inside a
    workspace, starting disables it and stopping restores it;
  * TC5 -- the glob helper never returns a symbolic link and never
    returns a candidate whose real path leaves the workspace, even when
    the pattern walks through a linked directory;
  * TC6 -- the grep helper never reads through a symbolic link, skips
    binary and oversized files while counting them, and only workspace
    content can match;
  * TC7 -- the listing helper skips symbolic links, honours its clamped
    entry bound with an honest truncation marker, and propagates the
    validator's refusal for an escaping path.

Loads the tool layer and the real registry in isolation under a stand-in
package; every ``opti_oignon.*`` entry plus the model-client and yaml
entries is snapshotted and evicted first, and the only other seeds are a
recording path validator with real containment semantics and recording
stand-ins for the shared tool handlers. A meta-path guard refuses any
project submodule that was not seeded, so the load behaves identically
whether or not the project is installed. Local-only. Runs under pytest or
the __main__ runner.
"""

import importlib.util
import os
import re
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_UNSAFE = ("execute_code", "read_file", "write_file", "list_files")


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the test's
    back -- silently importing live code. This guard sits ahead of every
    finder and refuses the names that were not seeded, so a load behaves
    identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


class _PathValidator:
    """Recording path validator with real containment semantics."""

    def __init__(self):
        self.calls = []

    def __call__(self, workspace_root, requested_path):
        self.calls.append((str(workspace_root), str(requested_path)))
        if not requested_path:
            return False, "", "Empty path"
        workspace_real = os.path.realpath(workspace_root)
        if os.path.isabs(requested_path):
            return False, "", "absolute paths are refused by the stand-in"
        resolved = os.path.realpath(
            os.path.join(workspace_real, requested_path)
        )
        if (
            not resolved.startswith(workspace_real + os.sep)
            and resolved != workspace_real
        ):
            return False, "", "path escapes the workspace"
        return True, resolved, ""


class _Manager:
    """Recording workspace manager behind the tool layer."""

    def __init__(self, workspace=None, disable_web_search=False):
        self.workspace = workspace
        self.created = []
        self.destroyed = []
        self._sessions = {}
        self.config = SimpleNamespace(
            disable_web_search_in_sandbox=disable_web_search
        )

    def create_sandbox(self, session_id, allow_degraded=False):
        session = SimpleNamespace(
            session_id=session_id,
            active=True,
            isolation_backend=SimpleNamespace(value="fallback"),
        )
        self.created.append((session_id, allow_degraded))
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id):
        return self._sessions.get(session_id)

    def destroy_sandbox(self, session_id):
        self.destroyed.append(session_id)
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.active = False
        return True

    def get_workspace_path(self, session_id):
        return self.workspace


def _load():
    """Load the tool layer and the real registry under a stand-in package."""
    keys = ["ollama", "yaml"] + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # no client import exists; drift fails loud

    fake_yaml = types.ModuleType("yaml")

    def _no_yaml(*_args, **_kwargs):
        raise RuntimeError("yaml is disabled inside the isolation window")

    fake_yaml.safe_load = _no_yaml
    sys.modules["yaml"] = fake_yaml

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    sys.modules["opti_oignon"] = root

    validator = _PathValidator()
    manager_mod = types.ModuleType("opti_oignon.sandbox_manager")
    manager_mod.SANDBOX_AVAILABLE = True
    manager_mod.SandboxManager = type("SandboxManager", (), {})
    manager_mod.SandboxSession = type("SandboxSession", (), {})
    manager_mod.validate_sandbox_path = validator
    manager_mod.sandbox_manager = None
    sys.modules["opti_oignon.sandbox_manager"] = manager_mod
    root.sandbox_manager = manager_mod

    calls = []
    file_tools = types.ModuleType("opti_oignon.file_tools")

    def _bash(session_id, command, timeout=30, _sandbox_manager=None):
        calls.append(("bash", session_id, command, timeout))
        return "handled bash"

    def _view(session_id, path, start_line=0, end_line=0,
              _sandbox_manager=None):
        calls.append(("view", session_id, path, start_line, end_line))
        return "handled view"

    def _create_file(session_id, path, content, _sandbox_manager=None):
        calls.append(("create_file", session_id, path))
        return "handled create_file"

    def _str_replace(session_id, path, old_str, new_str="",
                     _sandbox_manager=None):
        calls.append(("str_replace", session_id, path))
        return "handled str_replace"

    file_tools._handle_sandbox_bash = _bash
    file_tools._handle_sandbox_view = _view
    file_tools._handle_sandbox_create_file = _create_file
    file_tools._handle_sandbox_str_replace = _str_replace
    sys.modules["opti_oignon.file_tools"] = file_tools
    root.file_tools = file_tools

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
        sys.modules.pop("ollama", None)
        sys.modules.pop("yaml", None)
        for k, v in saved.items():
            sys.modules[k] = v

    def _load_real(short_name, filename):
        full = f"opti_oignon.{short_name}"
        spec = importlib.util.spec_from_file_location(full, _OO / filename)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        setattr(root, short_name, mod)
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            restore()
            raise
        return mod

    registry_mod = _load_real("tool_registry", "tool_registry.py")
    tools_mod = _load_real("sandbox_tools", "sandbox_tools.py")

    return SimpleNamespace(
        tools=tools_mod, registry_mod=registry_mod, validator=validator,
        handler_calls=calls, restore=restore,
    )


def _registry(ctx, with_web_search=True):
    """A fresh real registry carrying the unsafe tools (all enabled)."""
    reg = ctx.registry_mod.ToolRegistry()
    for name in _UNSAFE:
        reg.register(
            ctx.registry_mod.ToolDefinition(name=name, description="stand-in")
        )
    if with_web_search:
        reg.register(
            ctx.registry_mod.ToolDefinition(
                name="web_search", description="stand-in"
            )
        )
    return reg


def _enabled(reg, names):
    return {name: reg.get(name).enabled for name in names}


# ---------------------------------------------------------------------------
# TC1 -- start locks the registry; stop destroys and restores
# ---------------------------------------------------------------------------
def test_tc1_start_locks_the_registry_and_stop_destroys_and_restores():
    ctx = _load()
    try:
        mgr = _Manager()
        reg = _registry(ctx)
        wrapper = ctx.tools.SandboxToolSession(
            sandbox_mgr=mgr, tool_registry=reg
        )
        sid = wrapper.start()
        assert re.fullmatch(r"tool-session-[0-9a-f]{12}", sid), (
            f"the generated identity must be non-trivial, got {sid!r}"
        )
        assert _enabled(reg, _UNSAFE) == {n: False for n in _UNSAFE}, (
            "every unsandboxed tool must be disabled while a workspace "
            f"is active, got {_enabled(reg, _UNSAFE)}"
        )
        assert reg.sandbox_mode is True
        assert reg.get("web_search").enabled is True, (
            "without the shutoff configuration, web search stays enabled"
        )

        other = ctx.tools.SandboxToolSession(
            sandbox_mgr=_Manager(), tool_registry=_registry(ctx)
        )
        assert other.start() != sid, (
            "two generated identities must never collide"
        )

        assert wrapper.stop() is True
        assert mgr.destroyed == [sid], (
            "stopping must destroy the workspace it created"
        )
        assert _enabled(reg, _UNSAFE) == {n: True for n in _UNSAFE}, (
            "stopping must restore the tools the lock disabled"
        )
        assert reg.sandbox_mode is False
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# TC2 -- attach applies the identical lock; detach never destroys
# ---------------------------------------------------------------------------
def test_tc2_attach_locks_identically_and_detach_never_destroys():
    ctx = _load()
    try:
        mgr = _Manager()
        existing = mgr.create_sandbox("workspace-under-test")
        reg = _registry(ctx)
        wrapper = ctx.tools.SandboxToolSession(
            sandbox_mgr=mgr, tool_registry=reg
        )

        try:
            wrapper.attach("no-such-workspace")
            raise AssertionError("attaching to an unknown workspace must "
                                 "be refused")
        except ValueError:
            pass

        assert wrapper.attach("workspace-under-test") == existing.session_id
        assert _enabled(reg, _UNSAFE) == {n: False for n in _UNSAFE}, (
            "attach must apply the identical registry lock"
        )

        assert wrapper.detach() is True
        assert mgr.destroyed == [], (
            "detach must never destroy the workspace it releases"
        )
        assert existing.active is True, (
            "the released workspace must stay alive"
        )
        assert _enabled(reg, _UNSAFE) == {n: True for n in _UNSAFE}, (
            "detach must restore the tools the lock disabled"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# TC3 -- exactly four session-bound tools, and no exit toward the host
# ---------------------------------------------------------------------------
def test_tc3_the_model_surface_is_four_bound_tools_and_no_host_exit():
    ctx = _load()
    try:
        mgr = _Manager()
        wrapper = ctx.tools.SandboxToolSession(
            sandbox_mgr=mgr, tool_registry=_registry(ctx)
        )
        sid = wrapper.start()

        definitions = wrapper.get_tool_definitions()
        names = [d.name for d in definitions]
        assert names == ["bash", "view", "create_file", "str_replace"], (
            f"the model surface must be exactly the four workspace tools, "
            f"got {names}"
        )

        by_name = {d.name: d for d in definitions}
        by_name["bash"].handler("echo ok")
        by_name["view"].handler("notes.txt")
        by_name["create_file"].handler("notes.txt", "body")
        by_name["str_replace"].handler("notes.txt", "old", "new")
        bound = [c[1] for c in ctx.handler_calls]
        assert bound == [sid] * 4, (
            "every tool must route through the shared handlers with the "
            f"bound identity, saw identities {bound}"
        )

        for exit_name in (
            "copy_out", "copy_out_file", "copy_out_batch", "approve_files",
            "reject_files", "apply_workspace_changes", "browse_host",
            "clone_directory", "upload_files",
        ):
            assert not hasattr(wrapper, exit_name), (
                f"the model-facing layer must expose no host exit, "
                f"found {exit_name!r}"
            )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# TC4 -- the configured web-search shutoff is applied and restored
# ---------------------------------------------------------------------------
def test_tc4_configured_web_search_shutoff_is_applied_and_restored():
    ctx = _load()
    try:
        mgr = _Manager(disable_web_search=True)
        reg = _registry(ctx)
        wrapper = ctx.tools.SandboxToolSession(
            sandbox_mgr=mgr, tool_registry=reg
        )
        wrapper.start()
        assert reg.get("web_search").enabled is False, (
            "the configured shutoff must disable web search while a "
            "workspace is active"
        )
        assert "web_search" in reg._disabled_by_sandbox

        wrapper.stop()
        assert reg.get("web_search").enabled is True, (
            "stopping must restore web search with the other tools"
        )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# TC5 -- glob never returns a link nor a candidate resolving outside
# ---------------------------------------------------------------------------
def test_tc5_glob_never_returns_links_nor_out_of_workspace_candidates():
    ctx = _load()
    try:
        with tempfile.TemporaryDirectory() as ws, \
                tempfile.TemporaryDirectory() as outside:
            (Path(ws) / "kept.txt").write_text("inside", encoding="ascii")
            (Path(outside) / "secret.txt").write_text(
                "outside", encoding="ascii"
            )
            os.symlink(
                str(Path(outside) / "secret.txt"), str(Path(ws) / "link.txt")
            )
            os.symlink(str(outside), str(Path(ws) / "doorway"))

            mgr = _Manager(workspace=ws)
            wrapper = ctx.tools.SandboxToolSession(
                sandbox_mgr=mgr, tool_registry=_registry(ctx)
            )
            wrapper.start()

            listing = wrapper.glob("*.txt")
            assert listing.startswith("1 file(s)"), listing
            assert "kept.txt" in listing
            assert "link.txt" not in listing, (
                "a symbolic link must never be returned by glob"
            )

            through = wrapper.glob("doorway/*.txt")
            assert through.startswith("0 file(s)"), (
                "a candidate resolving outside the workspace must be "
                f"excluded, got {through!r}"
            )
            assert "secret" not in through
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# TC6 -- grep reads only workspace content; links and blobs are skipped
# ---------------------------------------------------------------------------
def test_tc6_grep_never_reads_links_and_skips_binary_and_oversized():
    ctx = _load()
    try:
        with tempfile.TemporaryDirectory() as ws, \
                tempfile.TemporaryDirectory() as outside:
            (Path(ws) / "ok.txt").write_text(
                "alpha SIGNAL beta", encoding="ascii"
            )
            (Path(outside) / "secret.txt").write_text(
                "SIGNAL from the outside", encoding="ascii"
            )
            os.symlink(
                str(Path(outside) / "secret.txt"), str(Path(ws) / "link.txt")
            )
            (Path(ws) / "blob.dat").write_bytes(b"\x00SIGNAL")
            (Path(ws) / "big.txt").write_text(
                "SIGNAL" + "x" * (1024 * 1024), encoding="ascii"
            )

            mgr = _Manager(workspace=ws)
            wrapper = ctx.tools.SandboxToolSession(
                sandbox_mgr=mgr, tool_registry=_registry(ctx)
            )
            wrapper.start()

            report = wrapper.grep("SIGNAL")
            header = report.splitlines()[0]
            assert header.startswith("1 match(es) in 1 file(s)"), header
            assert "[2 file(s) skipped: binary or >1 MiB]" in header, header
            assert "ok.txt:1:" in report
            assert "link.txt" not in report, (
                "a symbolic link must never be read by grep"
            )
            assert "outside" not in report, (
                "content from outside the workspace must never match"
            )
    finally:
        ctx.restore()


# ---------------------------------------------------------------------------
# TC7 -- ls skips links, clamps its bound, and propagates refusals
# ---------------------------------------------------------------------------
def test_tc7_ls_skips_links_clamps_the_bound_and_propagates_refusals():
    ctx = _load()
    try:
        with tempfile.TemporaryDirectory() as ws, \
                tempfile.TemporaryDirectory() as outside:
            (Path(outside) / "secret.txt").write_text("x", encoding="ascii")
            for name in ("a.txt", "b.txt", "c.txt"):
                (Path(ws) / name).write_text("x", encoding="ascii")
            (Path(ws) / "sub").mkdir()
            os.symlink(
                str(Path(outside) / "secret.txt"), str(Path(ws) / "link.txt")
            )

            mgr = _Manager(workspace=ws)
            wrapper = ctx.tools.SandboxToolSession(
                sandbox_mgr=mgr, tool_registry=_registry(ctx)
            )
            wrapper.start()

            listing = wrapper.ls(".")
            lines = listing.splitlines()
            assert lines[0] == "dir 0 sub", lines
            assert [ln.split()[-1] for ln in lines[1:]] == [
                "a.txt", "b.txt", "c.txt"
            ], lines
            assert "link.txt" not in listing, (
                "a symbolic link must never be listed"
            )

            clamped = wrapper.ls(".", max_entries=0)
            clamped_lines = clamped.splitlines()
            assert clamped_lines == [
                "dir 0 sub", "[truncated at 1 entries]"
            ], (
                "the entry bound must be clamped to at least one and the "
                f"truncation stated honestly, got {clamped_lines}"
            )

            before = len(ctx.validator.calls)
            refused = wrapper.ls("../elsewhere")
            assert refused.startswith("Error: Path rejected"), refused
            assert ctx.validator.calls[before:] == [
                (ws, "../elsewhere")
            ], (
                "the escaping path must be refused by the shared validator, "
                f"saw {ctx.validator.calls[before:]}"
            )
    finally:
        ctx.restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
