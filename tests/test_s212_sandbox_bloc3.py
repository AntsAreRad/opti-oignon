#!/usr/bin/env python3
"""S212 -- Sandbox Workspace cycle, Bloc 3: diff-gated write-back.

Per-fix suite for SANDBOX_WORKSPACE_SPEC section 6: the diff over the 6.1
baseline (hash-driven added/modified/deleted, no-baseline conservative
posture, bounds that REFUSE rather than truncate), the approval refinement
(deletions confirmed in their OWN set, parallel to approved_paths, never
inside an approve-all; the S116 codes re-proven), and the apply-to-host
writer -- the cycle's highest-risk component, its path validation covered
by the ADVERSARIAL suite first: dot-dot and absolute request paths,
symlinked parent directories (outside AND inside the root), a symlinked
target file replaced as the link with the host target intact,
outside-allowlist explicit targets refused before existence, the no-target
and cloned-root-conflict refusals, allowlist narrowing failing secure, and
the load-bearing rule that confirming a live path can never delete it.
Temp-plus-rename observed; the cloned-mount round-trip (mount/<rel> ->
cloned_root/<rel>, the S212 in-session arbitration) proven exact; the
audit verbs (deletion_confirm, apply_write, apply_delete, apply_refused,
apply_summary) pinned; the routes' code ladders via the fastapi
TestClient; and the spec / cartography / FRONTEND_REDESIGN / yaml /
source registrations, with coding_agent.py and agent/dispatch.py pinned
edit-free.

Harness: the s210/s211 `_load_fresh` shape -- other suites in the sweep
(test_file_tools) pre-load the real package chain, so this file ALWAYS
execs its own module copies and re-pins them per test.
"""

import importlib.util
import io
import os
import re
import stat as stat_module
import sys
import types

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_OO = os.path.join(_ROOT, "opti_oignon")
_API = os.path.join(_OO, "api")
_AGENT = os.path.join(_OO, "agent")


def _ensure_pkg(name: str, path: str) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        sys.modules[name] = mod


_ensure_pkg("opti_oignon", _OO)
_ensure_pkg("opti_oignon.api", _API)
_ensure_pkg("opti_oignon.agent", _AGENT)


def _load_fresh(relpath: str, register: str, bind: dict | None = None):
    """ALWAYS exec this file's own copy; never reuse a pre-loaded module.

    The s210-documented sweep-order class: test_file_tools imports the whole
    real opti_oignon.api.app chain, pre-loading the canonical names; reusing
    those would split exception-class identity (WorkspaceReviewDrift raised
    from one copy, caught against another in the routes). Temporarily
    register `bind` plus the module's own name, exec the fresh copy, restore
    every touched sys.modules entry afterwards.
    """
    bind = dict(bind or {})
    touched = list(bind.keys()) + [register]
    saved = {name: sys.modules.get(name) for name in touched}
    try:
        for name, mod in bind.items():
            sys.modules[name] = mod
        path = os.path.join(_ROOT, relpath)
        spec = importlib.util.spec_from_file_location(register, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[register] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


_sm = _load_fresh(
    os.path.join("opti_oignon", "sandbox_manager.py"),
    register="opti_oignon.sandbox_manager",
)
_st = _load_fresh(
    os.path.join("opti_oignon", "sandbox_tools.py"),
    register="opti_oignon.sandbox_tools",
    bind={"opti_oignon.sandbox_manager": _sm},
)
_ws = _load_fresh(
    os.path.join("opti_oignon", "sandbox_workspace.py"),
    register="opti_oignon.sandbox_workspace",
    bind={
        "opti_oignon.sandbox_manager": _sm,
        "opti_oignon.sandbox_tools": _st,
    },
)

SandboxConfig = _sm.SandboxConfig


@pytest.fixture(autouse=True)
def _bind_module_copies(monkeypatch):
    """Bind THIS file's module copies for the duration of each test."""
    pairs = {
        "opti_oignon.sandbox_manager": _sm,
        "opti_oignon.sandbox_tools": _st,
        "opti_oignon.sandbox_workspace": _ws,
    }
    for name, mod in pairs.items():
        monkeypatch.setitem(sys.modules, name, mod)
        parent = sys.modules.get("opti_oignon")
        if parent is not None:
            monkeypatch.setattr(
                parent, name.rsplit(".", 1)[1], mod, raising=False
            )
    yield


@pytest.fixture(autouse=True)
def _fresh_stores():
    # The S212 surface gate: every test in this suite targets a Bloc 3
    # mechanic, so the suite must be RED against a pre-S212 tree (the
    # pristine proof). Touching the S212 exception base here makes that
    # explicit -- the attribute does not exist before this session.
    _ = _ws.WorkspaceDiffError
    _ws.reset_workspace_bindings()
    _ws.reset_workspace_manifests()
    yield
    _ws.reset_workspace_bindings()
    _ws.reset_workspace_manifests()


def _make_manager(tmp_path, share_root=None, **cfg_kw):
    defaults = dict(
        workspace_base=str(tmp_path / "sbx"),
        audit_db_path="audit.db",
        isolation_backend="tempdir",
        require_degraded_confirmation=False,
        strict_mode=False,
        idle_ttl_seconds=0,
    )
    if share_root is not None:
        defaults["host_share_roots"] = [str(share_root)]
    defaults.update(cfg_kw)
    return _sm.SandboxManager(config=SandboxConfig(**defaults))


@pytest.fixture()
def share_root(tmp_path):
    root = tmp_path / "share"
    root.mkdir()
    return root


@pytest.fixture()
def manager(tmp_path, share_root):
    return _make_manager(tmp_path, share_root=share_root)


def _ws_path(manager, sid):
    return manager.get_active_workspace_path(sid)


def _write_ws(manager, sid, rel, data: bytes):
    path = os.path.join(_ws_path(manager, sid), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _seed_clone(manager, sid, share_root, files: dict[str, bytes],
                src_name="src", dest_subdir=""):
    """Materialize files under share_root/<src_name>, clone for real, and
    record the baseline the way the route does (root + mount write-once)."""
    src = share_root / src_name
    src.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    result = manager.clone_directory(sid, str(src), dest_subdir=dest_subdir)
    _ws.get_workspace_manifests().record(
        sid,
        result["manifest"],
        cloned_root=result["cloned_root"],
        cloned_mount=result["dest"],
    )
    return result


def _seed_upload(manager, sid, files: dict[str, bytes]):
    """Upload files for real and record the baseline (no root, no mount)."""
    items = [(name, io.BytesIO(data), len(data)) for name, data in files.items()]
    result = manager.upload_files(sid, items)
    entries = {w["relative_path"]: w["sha256"] for w in result["written"]}
    _ws.get_workspace_manifests().record(sid, entries)
    return result


def _diff(manager, sid):
    return _ws.generate_workspace_diff(sid, manager=manager)


def _apply(manager, sid, diff_hash, **kw):
    return _ws.apply_workspace_changes(
        sid, diff_hash, manager=manager, **kw
    )


def _approval_rows(manager, sid):
    return [e["action"] for e in manager.audit.get_approval_log(sid)]


def _read(relpath: str) -> str:
    with open(os.path.join(_ROOT, relpath), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 1. The apply writer -- ADVERSARIAL PATH SUITE FIRST (spec section 14 risk)
# ---------------------------------------------------------------------------

class TestApplyAdversarialPaths:
    def test_request_path_validator_refusals(self):
        refuse = _ws._refuse_request_path
        assert refuse("") is not None
        assert refuse("a\x00b") is not None
        assert refuse("/etc/passwd") is not None
        assert refuse("..") is not None
        assert refuse("../evil") is not None
        assert refuse("a/../../evil") is not None
        assert refuse(".") is not None
        assert refuse("a/b.txt") is None
        assert refuse("plain.txt") is None

    def test_poisoned_baseline_dotdot_refused(self, tmp_path, manager,
                                              share_root):
        sid = "ws-adv1"
        manager.create_sandbox(sid)
        target = share_root / "out"
        target.mkdir()
        outside = tmp_path / "evil.txt"
        outside.write_text("precious")
        # A poisoned baseline key classifies "deleted" (absent from the
        # live walk); even confirmed, the writer must refuse the shape.
        _ws.get_workspace_manifests().record(sid, {"../evil.txt": "h" * 64})
        manager.confirm_deletions(sid, ["../evil.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert out["deleted"] == []
        assert any(
            r["path"] == "../evil.txt" and "escapes" in r["error"]
            for r in out["refused"]
        )
        assert outside.read_text() == "precious"

    def test_poisoned_baseline_absolute_refused(self, manager, share_root):
        sid = "ws-adv2"
        manager.create_sandbox(sid)
        target = share_root / "out"
        target.mkdir()
        _ws.get_workspace_manifests().record(sid, {"/etc/oo-evil": "h" * 64})
        manager.confirm_deletions(sid, ["/etc/oo-evil"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert out["deleted"] == []
        assert any(
            r["path"] == "/etc/oo-evil" and "absolute" in r["error"]
            for r in out["refused"]
        )

    def test_symlinked_parent_outside_root_refused(self, tmp_path, manager,
                                                   share_root):
        sid = "ws-adv3"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "sub/x.txt", b"X")
        target = share_root / "out"
        target.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(str(outside), str(target / "sub"))
        manager.approve_files(sid, ["sub/x.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert out["applied"] == []
        assert any(
            r["path"] == "sub/x.txt" and "symlink" in r["error"]
            for r in out["refused"]
        )
        assert os.listdir(outside) == []

    def test_symlinked_parent_inside_root_refused(self, manager, share_root):
        sid = "ws-adv4"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "sub/x.txt", b"X")
        target = share_root / "out"
        target.mkdir()
        other = target / "other"
        other.mkdir()
        os.symlink(str(other), str(target / "sub"))
        manager.approve_files(sid, ["sub/x.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        # Even a symlink currently pointing INSIDE the root is refused: it
        # could be retargeted, and a redirected write is a redirected write.
        assert out["applied"] == []
        assert any("symlink" in r["error"] for r in out["refused"])
        assert os.listdir(other) == []

    def test_symlinked_target_file_replaced_as_link(self, manager,
                                                    share_root):
        sid = "ws-adv5"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"NEW")
        target = share_root / "out"
        target.mkdir()
        secret = target / "secret.txt"
        secret.write_text("SECRET")
        os.symlink(str(secret), str(target / "a.txt"))
        manager.approve_files(sid, ["a.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert [e["path"] for e in out["applied"]] == ["a.txt"]
        dest = target / "a.txt"
        # os.replace swapped the LINK itself for the regular file; the
        # link's old target was never written through.
        assert not os.path.islink(str(dest))
        assert dest.read_text() == "NEW"
        assert secret.read_text() == "SECRET"

    def test_outside_allowlist_target_refused_before_existence(
        self, tmp_path, manager
    ):
        sid = "ws-adv6"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"A")
        manager.approve_files(sid, ["a.txt"])
        d = _diff(manager, sid)
        missing_outside = str(tmp_path / "nowhere" / "deep")
        with pytest.raises(PermissionError):
            _apply(manager, sid, d.diff_hash, target_dir=missing_outside)
        existing_outside = tmp_path / "real-outside"
        existing_outside.mkdir()
        with pytest.raises(PermissionError):
            _apply(manager, sid, d.diff_hash,
                   target_dir=str(existing_outside))

    def test_no_target_refusal(self, manager):
        sid = "ws-adv7"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"A")
        d = _diff(manager, sid)
        with pytest.raises(_ws.WorkspaceApplyTargetError):
            _apply(manager, sid, d.diff_hash)

    def test_cloned_root_conflict_refused(self, manager, share_root):
        sid = "ws-adv8"
        manager.create_sandbox(sid)
        _seed_clone(manager, sid, share_root, {"a.txt": b"v1"})
        other = share_root / "elsewhere"
        other.mkdir()
        d = _diff(manager, sid)
        with pytest.raises(_ws.WorkspaceApplyTargetError):
            _apply(manager, sid, d.diff_hash, target_dir=str(other))

    def test_allowlist_narrowing_fails_secure(self, tmp_path, manager,
                                              share_root, monkeypatch):
        sid = "ws-adv9"
        manager.create_sandbox(sid)
        _seed_clone(manager, sid, share_root, {"a.txt": b"v1"})
        _write_ws(manager, sid, "src/a.txt", b"v2")
        manager.approve_files(sid, ["src/a.txt"])
        d = _diff(manager, sid)
        narrowed = tmp_path / "other-root"
        narrowed.mkdir()
        monkeypatch.setattr(
            manager._config, "host_share_roots",
            [os.path.realpath(str(narrowed))],
        )
        with pytest.raises(PermissionError):
            _apply(manager, sid, d.diff_hash)
        assert (share_root / "src" / "a.txt").read_bytes() == b"v1"

    def test_confirming_live_path_never_deletes(self, manager, share_root):
        sid = "ws-adv10"
        manager.create_sandbox(sid)
        _seed_clone(manager, sid, share_root, {"keep.txt": b"K"})
        # The path is alive (unchanged); a recorded confirmation is inert.
        manager.confirm_deletions(sid, ["src/keep.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash)
        assert out["deleted"] == []
        assert (share_root / "src" / "keep.txt").read_bytes() == b"K"

    def test_missing_diff_hash_refused(self, manager):
        sid = "ws-adv11"
        manager.create_sandbox(sid)
        with pytest.raises(_ws.WorkspaceReviewDrift):
            _apply(manager, sid, "")

    def test_drift_after_review_refused(self, manager, share_root):
        sid = "ws-adv12"
        manager.create_sandbox(sid)
        target = share_root / "out"
        target.mkdir()
        _write_ws(manager, sid, "a.txt", b"v1")
        manager.approve_files(sid, ["a.txt"])
        d = _diff(manager, sid)
        _write_ws(manager, sid, "a.txt", b"v2-drift")
        with pytest.raises(_ws.WorkspaceReviewDrift):
            _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert not (target / "a.txt").exists()


# ---------------------------------------------------------------------------
# 2. Writer mechanics: temp-plus-rename, modes, deletions, honesty
# ---------------------------------------------------------------------------

class TestApplyWriterMechanics:
    def test_temp_plus_rename_observed(self, manager, share_root,
                                       monkeypatch):
        sid = "ws-mech1"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"DATA")
        manager.approve_files(sid, ["a.txt"])
        target = share_root / "out"
        target.mkdir()
        observed = []
        real_replace = os.replace

        def spy(src, dst, *a, **kw):
            observed.append((src, dst))
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(_ws.os, "replace", spy)
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert [e["path"] for e in out["applied"]] == ["a.txt"]
        assert len(observed) == 1
        src, dst = observed[0]
        assert os.path.dirname(src) == os.path.dirname(dst)
        assert os.path.basename(src).startswith(".oo-apply-")
        assert os.path.basename(src).endswith(".tmp")

    def test_no_temp_residue_on_failure(self, manager, share_root,
                                        monkeypatch):
        sid = "ws-mech2"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"DATA")
        manager.approve_files(sid, ["a.txt"])
        target = share_root / "out"
        target.mkdir()

        def boom(src, dst, *a, **kw):
            raise OSError("simulated rename failure")

        monkeypatch.setattr(_ws.os, "replace", boom)
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert out["applied"] == []
        assert any(r["path"] == "a.txt" for r in out["refused"])
        leftovers = [n for n in os.listdir(target) if ".oo-apply-" in n]
        assert leftovers == []

    def test_source_mode_preserved(self, manager, share_root):
        sid = "ws-mech3"
        manager.create_sandbox(sid)
        src_path = _write_ws(manager, sid, "a.sh", b"#!/bin/sh\n")
        os.chmod(src_path, 0o640)
        manager.approve_files(sid, ["a.sh"])
        target = share_root / "out"
        target.mkdir()
        d = _diff(manager, sid)
        _apply(manager, sid, d.diff_hash, target_dir=str(target))
        mode = stat_module.S_IMODE(os.lstat(str(target / "a.sh")).st_mode)
        assert mode == 0o640

    def test_setuid_bits_stripped(self, manager, share_root):
        sid = "ws-mech4"
        manager.create_sandbox(sid)
        src_path = _write_ws(manager, sid, "a.sh", b"#!/bin/sh\n")
        os.chmod(src_path, 0o4755)
        manager.approve_files(sid, ["a.sh"])
        target = share_root / "out"
        target.mkdir()
        d = _diff(manager, sid)
        _apply(manager, sid, d.diff_hash, target_dir=str(target))
        mode = stat_module.S_IMODE(os.lstat(str(target / "a.sh")).st_mode)
        assert mode == 0o755

    def test_content_and_byte_counts(self, manager, share_root):
        sid = "ws-mech5"
        manager.create_sandbox(sid)
        payload = b"x" * 70000  # spans more than one apply chunk
        _write_ws(manager, sid, "big.bin", payload)
        manager.approve_files(sid, ["big.bin"])
        target = share_root / "out"
        target.mkdir()
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert out["applied"][0]["bytes"] == len(payload)
        assert (target / "big.bin").read_bytes() == payload

    def test_dest_directory_collision_refused(self, manager, share_root):
        sid = "ws-mech6"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "thing", b"file-side")
        manager.approve_files(sid, ["thing"])
        target = share_root / "out"
        target.mkdir()
        (target / "thing").mkdir()
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert out["applied"] == []
        assert any(r["path"] == "thing" for r in out["refused"])
        assert (target / "thing").is_dir()

    def test_deletion_symlink_removed_as_link(self, manager, share_root):
        sid = "ws-mech7"
        manager.create_sandbox(sid)
        target = share_root / "out"
        target.mkdir()
        keep = target / "keep.txt"
        keep.write_text("KEEP")
        os.symlink(str(keep), str(target / "gone.txt"))
        _ws.get_workspace_manifests().record(sid, {"gone.txt": "h" * 64})
        manager.confirm_deletions(sid, ["gone.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert out["deleted"] == [{"path": "gone.txt", "action": "deleted"}]
        assert not os.path.lexists(str(target / "gone.txt"))
        assert keep.read_text() == "KEEP"

    def test_deletion_already_absent_honest(self, manager, share_root):
        sid = "ws-mech8"
        manager.create_sandbox(sid)
        target = share_root / "out"
        target.mkdir()
        _ws.get_workspace_manifests().record(sid, {"ghost.txt": "h" * 64})
        manager.confirm_deletions(sid, ["ghost.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert out["deleted"] == [
            {"path": "ghost.txt", "action": "already_absent"}
        ]

    def test_deletion_target_directory_refused(self, manager, share_root):
        sid = "ws-mech9"
        manager.create_sandbox(sid)
        target = share_root / "out"
        target.mkdir()
        (target / "adir").mkdir()
        _ws.get_workspace_manifests().record(sid, {"adir": "h" * 64})
        manager.confirm_deletions(sid, ["adir"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert out["deleted"] == []
        assert any(
            r["path"] == "adir" and "directory" in r["error"]
            for r in out["refused"]
        )
        assert (target / "adir").is_dir()

    def test_skipped_counts_honest(self, manager, share_root):
        sid = "ws-mech10"
        manager.create_sandbox(sid)
        target = share_root / "out"
        target.mkdir()
        _write_ws(manager, sid, "approved.txt", b"A")
        _write_ws(manager, sid, "unapproved.txt", b"U")
        _ws.get_workspace_manifests().record(sid, {"gone.txt": "h" * 64})
        manager.approve_files(sid, ["approved.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert out["skipped_unapproved"] == 1
        assert out["skipped_unconfirmed"] == 1
        assert not (target / "unapproved.txt").exists()
        assert (target / "approved.txt").exists()

    def test_nested_parents_created(self, manager, share_root):
        sid = "ws-mech11"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a/b/c/deep.txt", b"D")
        manager.approve_files(sid, ["a/b/c/deep.txt"])
        target = share_root / "out"
        target.mkdir()
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert [e["path"] for e in out["applied"]] == ["a/b/c/deep.txt"]
        assert (target / "a" / "b" / "c" / "deep.txt").read_bytes() == b"D"

    def test_touch_activity_on_apply(self, manager, share_root):
        sid = "ws-mech12"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"A")
        manager.approve_files(sid, ["a.txt"])
        target = share_root / "out"
        target.mkdir()
        before = manager.get_session(sid).last_activity
        d = _diff(manager, sid)
        _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert manager.get_session(sid).last_activity >= before


# ---------------------------------------------------------------------------
# 3. The cloned-mount round-trip (S212 in-session arbitration)
# ---------------------------------------------------------------------------

class TestMountRoundTrip:
    def test_round_trip_writes_originals(self, manager, share_root):
        sid = "ws-rt1"
        manager.create_sandbox(sid)
        _seed_clone(manager, sid, share_root, {"a.txt": b"v1"})
        _write_ws(manager, sid, "src/a.txt", b"v2")
        manager.approve_files(sid, ["src/a.txt"])
        d = _diff(manager, sid)
        assert d.cloned_mount == "src"
        out = _apply(manager, sid, d.diff_hash)
        assert [e["path"] for e in out["applied"]] == ["src/a.txt"]
        # mount/<rel> -> cloned_root/<rel>: the ORIGINAL file, never nested.
        assert (share_root / "src" / "a.txt").read_bytes() == b"v2"
        assert not (share_root / "src" / "src").exists()

    def test_outside_subtree_refused_under_cloned_root(self, manager,
                                                       share_root):
        sid = "ws-rt2"
        manager.create_sandbox(sid)
        _seed_clone(manager, sid, share_root, {"a.txt": b"v1"})
        _write_ws(manager, sid, "rogue.txt", b"R")
        manager.approve_files(sid, ["rogue.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash)
        assert out["applied"] == []
        assert any(
            r["path"] == "rogue.txt"
            and "outside the cloned subtree" in r["error"]
            for r in out["refused"]
        )
        assert not (share_root / "rogue.txt").exists()
        assert not (share_root / "src" / "rogue.txt").exists()

    def test_deletion_round_trip(self, manager, share_root):
        sid = "ws-rt3"
        manager.create_sandbox(sid)
        _seed_clone(manager, sid, share_root, {"a.txt": b"v1", "c.txt": b"C"})
        os.remove(os.path.join(_ws_path(manager, sid), "src", "c.txt"))
        manager.confirm_deletions(sid, ["src/c.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash)
        assert out["deleted"] == [{"path": "src/c.txt", "action": "deleted"}]
        assert not (share_root / "src" / "c.txt").exists()
        assert (share_root / "src" / "a.txt").exists()

    def test_root_without_mount_refuses_conservatively(self, manager,
                                                       share_root):
        sid = "ws-rt4"
        manager.create_sandbox(sid)
        src = share_root / "srcdir"
        src.mkdir()
        _write_ws(manager, sid, "x.txt", b"X")
        manager.approve_files(sid, ["x.txt"])
        _ws.get_workspace_manifests().record(
            sid, {"x.txt": "h" * 64}, cloned_root=os.path.realpath(str(src))
        )
        d = _diff(manager, sid)
        assert d.cloned_root is not None and d.cloned_mount is None
        out = _apply(manager, sid, d.diff_hash)
        assert out["applied"] == []
        assert all(
            "cloned mount unknown" in r["error"] for r in out["refused"]
        )

    def test_write_once_pair(self):
        store = _ws.get_workspace_manifests()
        store.record("w1", {"a": "h"}, cloned_root="/r1", cloned_mount="m1")
        store.record("w1", {"b": "h"}, cloned_root="/r2", cloned_mount="m2")
        assert store.get_cloned_root("w1") == "/r1"
        assert store.get_cloned_mount("w1") == "m1"
        store.drop("w1")
        assert store.get_cloned_root("w1") is None
        assert store.get_cloned_mount("w1") is None

    def test_dest_subdir_clone_round_trips(self, manager, share_root):
        sid = "ws-rt5"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "in/.keep", b"")
        _seed_clone(manager, sid, share_root, {"a.txt": b"v1"},
                    dest_subdir="in")
        _write_ws(manager, sid, "in/src/a.txt", b"v2")
        manager.approve_files(sid, ["in/src/a.txt"])
        d = _diff(manager, sid)
        assert d.cloned_mount == "in/src"
        out = _apply(manager, sid, d.diff_hash)
        assert [e["path"] for e in out["applied"]] == ["in/src/a.txt"]
        assert (share_root / "src" / "a.txt").read_bytes() == b"v2"


# ---------------------------------------------------------------------------
# 4. The diff: classification, postures, bounds, hash
# ---------------------------------------------------------------------------

class TestDiffClassification:
    def test_clone_baseline_classes(self, manager, share_root):
        sid = "ws-d1"
        manager.create_sandbox(sid)
        _seed_clone(manager, sid, share_root,
                    {"same.txt": b"S", "mod.txt": b"old", "gone.txt": b"G"})
        wsdir = _ws_path(manager, sid)
        _write_ws(manager, sid, "src/mod.txt", b"new")
        os.remove(os.path.join(wsdir, "src", "gone.txt"))
        _write_ws(manager, sid, "src/new.txt", b"N")
        d = _diff(manager, sid)
        kinds = {c.path: c.kind for c in d.entries}
        assert kinds == {
            "src/mod.txt": "modified",
            "src/gone.txt": "deleted",
            "src/new.txt": "added",
        }
        assert d.unchanged == 1
        assert d.baseline_present is True
        mod = next(c for c in d.entries if c.path == "src/mod.txt")
        assert mod.baseline_hash and mod.current_hash
        assert mod.baseline_hash != mod.current_hash
        gone = next(c for c in d.entries if c.path == "src/gone.txt")
        assert gone.current_hash == "" and gone.baseline_hash

    def test_upload_then_unmodified_is_unchanged(self, manager):
        sid = "ws-d2"
        manager.create_sandbox(sid)
        _seed_upload(manager, sid, {"u.txt": b"U"})
        d = _diff(manager, sid)
        assert d.entries == []
        assert d.unchanged == 1
        assert d.baseline_present is True
        assert d.cloned_root is None and d.cloned_mount is None

    def test_no_baseline_everything_added(self, manager):
        sid = "ws-d3"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"A")
        _write_ws(manager, sid, "b/b.txt", b"B")
        d = _diff(manager, sid)
        assert d.baseline_present is False
        assert d.cloned_root is None
        assert {c.kind for c in d.entries} == {"added"}
        assert {c.path for c in d.entries} == {"a.txt", "b/b.txt"}

    def test_binary_added_hash_only(self, manager):
        sid = "ws-d4"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "blob.bin", bytes(range(256)) * 4)
        d = _diff(manager, sid)
        entry = d.entries[0]
        assert entry.kind == "added"
        assert len(entry.current_hash) == 64

    def test_ws_symlink_skipped_never_followed(self, manager):
        sid = "ws-d5"
        manager.create_sandbox(sid)
        wsdir = _ws_path(manager, sid)
        os.symlink("/etc/passwd", os.path.join(wsdir, "evil-link"))
        d = _diff(manager, sid)
        assert d.skipped_symlinks == 1
        assert all(c.path != "evil-link" for c in d.entries)

    def test_fifo_skipped_counted(self, manager):
        sid = "ws-d6"
        manager.create_sandbox(sid)
        wsdir = _ws_path(manager, sid)
        os.mkfifo(os.path.join(wsdir, "pipe"))
        d = _diff(manager, sid)
        assert d.skipped_special == 1
        assert all(c.path != "pipe" for c in d.entries)

    def test_live_entry_bound_refuses(self, tmp_path, share_root):
        manager = _make_manager(tmp_path, share_root=share_root,
                                diff_max_entries=2)
        sid = "ws-d7"
        manager.create_sandbox(sid)
        for i in range(3):
            _write_ws(manager, sid, f"f{i}.txt", b"x")
        with pytest.raises(_ws.WorkspaceDiffBoundExceeded):
            _diff(manager, sid)

    def test_classified_bound_refuses(self, tmp_path, share_root):
        manager = _make_manager(tmp_path, share_root=share_root,
                                diff_max_entries=3)
        sid = "ws-d8"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "live.txt", b"L")
        baseline = {f"gone{i}.txt": "h" * 64 for i in range(4)}
        baseline["live.txt"] = _ws.manifest_hash_file(
            os.path.join(_ws_path(manager, sid), "live.txt")
        )
        _ws.get_workspace_manifests().record(sid, baseline)
        with pytest.raises(_ws.WorkspaceDiffBoundExceeded):
            _diff(manager, sid)

    def test_depth_bound_refuses(self, manager):
        sid = "ws-d9"
        manager.create_sandbox(sid)
        wsdir = _ws_path(manager, sid)
        deep = wsdir
        for i in range(_ws._WORKSPACE_WALK_MAX_DEPTH + 2):
            deep = os.path.join(deep, f"d{i}")
        os.makedirs(deep)
        with open(os.path.join(deep, "leaf.txt"), "wb") as fh:
            fh.write(b"x")
        with pytest.raises(_ws.WorkspaceDiffBoundExceeded):
            _diff(manager, sid)

    def test_diff_hash_deterministic(self, manager):
        sid = "ws-d10"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"A")
        h1 = _diff(manager, sid).diff_hash
        h2 = _diff(manager, sid).diff_hash
        assert h1 == h2

    def test_diff_hash_sensitive(self, manager):
        sid = "ws-d11"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"A")
        h1 = _diff(manager, sid).diff_hash
        _write_ws(manager, sid, "a.txt", b"B")
        h2 = _diff(manager, sid).diff_hash
        assert h1 != h2

    def test_diff_hash_order_independent(self):
        c1 = _ws.WorkspaceChange(path="a", kind="added", current_hash="1")
        c2 = _ws.WorkspaceChange(path="b", kind="deleted",
                                 baseline_hash="2")
        assert (_ws.compute_workspace_diff_hash([c1, c2])
                == _ws.compute_workspace_diff_hash([c2, c1]))

    def test_unknown_session_value_error(self, manager):
        with pytest.raises(ValueError):
            _diff(manager, "ws-missing")


# ---------------------------------------------------------------------------
# 5. The approval-kind machine (S116 re-proven, S212 parallel set)
# ---------------------------------------------------------------------------

class TestApprovalKindMachine:
    def test_s116_approve_semantics_reproved(self, manager):
        sid = "ws-a1"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"A")
        approved = manager.approve_files(sid, ["a.txt", "missing.txt"])
        assert approved == ["a.txt"]
        assert manager.is_file_approved(sid, "a.txt")
        assert not manager.is_file_approved(sid, "missing.txt")
        session = manager.get_session(sid)
        assert session.approval_state == _sm.ApprovalState.APPROVED

    def test_approve_cannot_take_deleted_entry(self, manager):
        sid = "ws-a2"
        manager.create_sandbox(sid)
        _ws.get_workspace_manifests().record(sid, {"gone.txt": "h" * 64})
        approved = manager.approve_files(sid, ["gone.txt"])
        assert approved == []
        assert not manager.is_file_approved(sid, "gone.txt")

    def test_confirm_deletions_records_and_audits(self, manager):
        sid = "ws-a3"
        manager.create_sandbox(sid)
        recorded = manager.confirm_deletions(sid, ["x.txt", ""])
        assert recorded == ["x.txt"]
        assert manager.get_confirmed_deletions(sid) == {"x.txt"}
        rows = manager.audit.get_approval_log(sid)
        assert any(e["action"] == "deletion_confirm" for e in rows)

    def test_confirmations_additive_and_copied(self, manager):
        sid = "ws-a4"
        manager.create_sandbox(sid)
        manager.confirm_deletions(sid, ["a"])
        manager.confirm_deletions(sid, ["b"])
        got = manager.get_confirmed_deletions(sid)
        assert got == {"a", "b"}
        got.add("tampered")
        assert manager.get_confirmed_deletions(sid) == {"a", "b"}

    def test_reject_clears_both_sets(self, manager):
        sid = "ws-a5"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"A")
        manager.approve_files(sid, ["a.txt"])
        manager.confirm_deletions(sid, ["gone.txt"])
        manager.reject_files(sid)
        session = manager.get_session(sid)
        assert session.approval_state == _sm.ApprovalState.REJECTED
        assert session.approved_paths == set()
        assert manager.get_confirmed_deletions(sid) == set()

    def test_copy_out_unapproved_still_permission_error(self, manager,
                                                        tmp_path):
        sid = "ws-a6"
        manager.create_sandbox(sid)
        _write_ws(manager, sid, "a.txt", b"A")
        dest = tmp_path / "exports"
        dest.mkdir()
        with pytest.raises(PermissionError):
            manager.copy_out_file(sid, "a.txt", str(dest))

    def test_confirm_unknown_session_value_error(self, manager):
        with pytest.raises(ValueError):
            manager.confirm_deletions("ws-missing", ["x"])


# ---------------------------------------------------------------------------
# 6. The apply audit rows
# ---------------------------------------------------------------------------

class TestApplyAudit:
    def test_full_verb_set_and_dest_column(self, manager, share_root):
        sid = "ws-au1"
        manager.create_sandbox(sid)
        target = share_root / "out"
        target.mkdir()
        _write_ws(manager, sid, "ok.txt", b"OK")
        _write_ws(manager, sid, "sub/blocked.txt", b"B")
        outside = share_root.parent / "elsewhere"
        outside.mkdir()
        os.symlink(str(outside), str(target / "sub"))
        (target / "gone.txt").write_text("G")
        _ws.get_workspace_manifests().record(sid, {"gone.txt": "h" * 64})
        manager.approve_files(sid, ["ok.txt", "sub/blocked.txt"])
        manager.confirm_deletions(sid, ["gone.txt"])
        d = _diff(manager, sid)
        out = _apply(manager, sid, d.diff_hash, target_dir=str(target))
        assert [e["path"] for e in out["applied"]] == ["ok.txt"]
        assert out["deleted"] == [{"path": "gone.txt", "action": "deleted"}]
        assert any(r["path"] == "sub/blocked.txt" for r in out["refused"])
        rows = manager.audit.get_approval_log(sid)
        actions = [e["action"] for e in rows]
        assert "deletion_confirm" in actions
        assert actions.count("apply_write") == 1
        assert actions.count("apply_delete") == 1
        assert actions.count("apply_refused") == 1
        assert actions.count("apply_summary") == 1
        target_real = os.path.realpath(str(target))
        for e in rows:
            if e["action"].startswith("apply_"):
                assert e["dest_dir"] == target_real


# ---------------------------------------------------------------------------
# 7. Routes: the code ladders via TestClient
# ---------------------------------------------------------------------------

_ROUTES_SANDBOX_CACHE: dict = {}


def _load_routes_sandbox():
    if "mod" in _ROUTES_SANDBOX_CACHE:
        return _ROUTES_SANDBOX_CACHE["mod"]
    schemas = _load_fresh(
        os.path.join("opti_oignon", "api", "schemas.py"),
        register="opti_oignon.api.schemas",
    )
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.SANDBOX_AVAILABLE = True
    deps.sandbox_manager = None  # patched per test on the routes module
    deps.FILE_TOOLS_AVAILABLE = False
    mod = _load_fresh(
        os.path.join("opti_oignon", "api", "routes_sandbox.py"),
        register="opti_oignon.api.routes_sandbox",
        bind={
            "opti_oignon.api.routes_auth": None,
            "opti_oignon.api.schemas": schemas,
            "opti_oignon.api.deps": deps,
            "opti_oignon.sandbox_manager": _sm,
            "opti_oignon.sandbox_workspace": _ws,
            "opti_oignon.sandbox_tools": _st,
        },
    )
    _ROUTES_SANDBOX_CACHE["mod"] = mod
    return mod


@pytest.fixture()
def api(manager, monkeypatch):
    rs = _load_routes_sandbox()
    monkeypatch.setattr(rs, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(rs, "sandbox_manager", manager)
    app = fastapi.FastAPI()
    app.include_router(rs.router)
    return rs, TestClient(app), manager


def _mp(*named):
    return [
        ("files", (name, io.BytesIO(data), "application/octet-stream"))
        for name, data in named
    ]


class TestDiffRoute:
    def test_200_shape(self, api, share_root):
        rs, client, manager = api
        manager.create_sandbox("ws-r1")
        _seed_clone(manager, "ws-r1", share_root, {"a.txt": b"v1"})
        _write_ws(manager, "ws-r1", "src/a.txt", b"v2")
        manager.approve_files("ws-r1", ["src/a.txt"])
        manager.confirm_deletions("ws-r1", ["phantom.txt"])
        resp = client.get("/api/sandbox/ws-r1/diff")
        assert resp.status_code == 200
        body = resp.json()
        assert body["baseline_present"] is True
        assert body["cloned_mount"] == "src"
        assert body["cloned_root"]
        assert body["diff_hash"]
        assert body["entries"] == [{
            "path": "src/a.txt",
            "kind": "modified",
            "size": 2,
            "baseline_hash": body["entries"][0]["baseline_hash"],
            "current_hash": body["entries"][0]["current_hash"],
        }]
        assert body["approved_paths"] == ["src/a.txt"]
        assert body["confirmed_deletions"] == ["phantom.txt"]

    def test_404_unknown_session(self, api):
        rs, client, manager = api
        resp = client.get("/api/sandbox/ws-none/diff")
        assert resp.status_code == 404

    def test_403_foreign_owner(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-r2", owner_user_id="someone-else")
        resp = client.get("/api/sandbox/ws-r2/diff")
        assert resp.status_code == 403

    def test_413_over_bound(self, tmp_path, share_root, monkeypatch):
        manager = _make_manager(tmp_path, share_root=share_root,
                                diff_max_entries=1)
        rs = _load_routes_sandbox()
        monkeypatch.setattr(rs, "SANDBOX_AVAILABLE", True)
        monkeypatch.setattr(rs, "sandbox_manager", manager)
        app = fastapi.FastAPI()
        app.include_router(rs.router)
        client = TestClient(app)
        manager.create_sandbox("ws-r3")
        _write_ws(manager, "ws-r3", "a.txt", b"A")
        _write_ws(manager, "ws-r3", "b.txt", b"B")
        resp = client.get("/api/sandbox/ws-r3/diff")
        assert resp.status_code == 413

    def test_503_partial_build(self, api, monkeypatch):
        rs, client, manager = api
        manager.create_sandbox("ws-r4")
        monkeypatch.setattr(rs, "WORKSPACE_BINDING_AVAILABLE", False)
        assert client.get("/api/sandbox/ws-r4/diff").status_code == 503
        assert client.post(
            "/api/sandbox/ws-r4/confirm-deletions", json={"paths": []}
        ).status_code == 503
        assert client.post(
            "/api/sandbox/ws-r4/apply", json={"diff_hash": "x"}
        ).status_code == 503


class TestConfirmDeletionsRoute:
    def test_per_path_refusals_in_200(self, api, share_root):
        rs, client, manager = api
        manager.create_sandbox("ws-c1")
        _seed_clone(manager, "ws-c1", share_root, {"gone.txt": b"G"})
        os.remove(os.path.join(_ws_path(manager, "ws-c1"),
                               "src", "gone.txt"))
        resp = client.post(
            "/api/sandbox/ws-c1/confirm-deletions",
            json={"paths": ["src/gone.txt", "src/alive... no",
                            "unknown.txt", ""]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["confirmed"] == ["src/gone.txt"]
        refused = {r["path"]: r["reason"] for r in body["refused"]}
        assert "unknown.txt" in refused
        assert "" in refused
        assert manager.get_confirmed_deletions("ws-c1") == {"src/gone.txt"}

    def test_live_path_refused(self, api, share_root):
        rs, client, manager = api
        manager.create_sandbox("ws-c2")
        _seed_clone(manager, "ws-c2", share_root, {"keep.txt": b"K"})
        resp = client.post(
            "/api/sandbox/ws-c2/confirm-deletions",
            json={"paths": ["src/keep.txt"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["confirmed"] == []
        assert body["refused"][0]["path"] == "src/keep.txt"
        assert "not classified as deleted" in body["refused"][0]["reason"]

    def test_404_unknown_session(self, api):
        rs, client, manager = api
        resp = client.post(
            "/api/sandbox/ws-none/confirm-deletions", json={"paths": ["x"]}
        )
        assert resp.status_code == 404


class TestApplyRoute:
    def test_full_round_trip_200(self, api, share_root):
        rs, client, manager = api
        manager.create_sandbox("ws-ap1")
        src = share_root / "proj"
        src.mkdir()
        (src / "a.txt").write_text("v1")
        (src / "old.txt").write_text("OLD")
        resp = client.post(
            "/api/sandbox/ws-ap1/clone", json={"src_path": str(src)}
        )
        assert resp.status_code == 200
        wsdir = _ws_path(manager, "ws-ap1")
        with open(os.path.join(wsdir, "proj", "a.txt"), "w") as fh:
            fh.write("v2")
        os.remove(os.path.join(wsdir, "proj", "old.txt"))
        diff1 = client.get("/api/sandbox/ws-ap1/diff").json()
        client.post(
            "/api/sandbox/ws-ap1/approve", json={"paths": ["proj/a.txt"]}
        )
        client.post(
            "/api/sandbox/ws-ap1/confirm-deletions",
            json={"paths": ["proj/old.txt"]},
        )
        resp = client.post(
            "/api/sandbox/ws-ap1/apply",
            json={"diff_hash": diff1["diff_hash"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["target"] == os.path.realpath(str(src))
        assert [e["path"] for e in body["applied"]] == ["proj/a.txt"]
        assert body["deleted"] == [
            {"path": "proj/old.txt", "action": "deleted", "bytes": 0}
        ]
        assert (src / "a.txt").read_text() == "v2"
        assert not (src / "old.txt").exists()

    def test_400_no_target_upload_only(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-ap2")
        client.post("/api/sandbox/ws-ap2/upload",
                    files=_mp(("a.txt", b"A")))
        diff = client.get("/api/sandbox/ws-ap2/diff").json()
        resp = client.post(
            "/api/sandbox/ws-ap2/apply",
            json={"diff_hash": diff["diff_hash"]},
        )
        assert resp.status_code == 400

    def test_403_outside_allowlist_target(self, api, tmp_path):
        rs, client, manager = api
        manager.create_sandbox("ws-ap3")
        _write_ws(manager, "ws-ap3", "a.txt", b"A")
        diff = client.get("/api/sandbox/ws-ap3/diff").json()
        outside = tmp_path / "definitely-outside"
        resp = client.post(
            "/api/sandbox/ws-ap3/apply",
            json={"diff_hash": diff["diff_hash"],
                  "target_dir": str(outside)},
        )
        assert resp.status_code == 403

    def test_409_stale_hash(self, api, share_root):
        rs, client, manager = api
        manager.create_sandbox("ws-ap4")
        target = share_root / "out"
        target.mkdir()
        _write_ws(manager, "ws-ap4", "a.txt", b"v1")
        diff = client.get("/api/sandbox/ws-ap4/diff").json()
        _write_ws(manager, "ws-ap4", "a.txt", b"v2")
        resp = client.post(
            "/api/sandbox/ws-ap4/apply",
            json={"diff_hash": diff["diff_hash"],
                  "target_dir": str(target)},
        )
        assert resp.status_code == 409

    def test_404_unknown_session(self, api):
        rs, client, manager = api
        resp = client.post(
            "/api/sandbox/ws-none/apply", json={"diff_hash": "x"}
        )
        assert resp.status_code == 404

    def test_413_over_bound(self, tmp_path, share_root, monkeypatch):
        manager = _make_manager(tmp_path, share_root=share_root,
                                diff_max_entries=1)
        rs = _load_routes_sandbox()
        monkeypatch.setattr(rs, "SANDBOX_AVAILABLE", True)
        monkeypatch.setattr(rs, "sandbox_manager", manager)
        app = fastapi.FastAPI()
        app.include_router(rs.router)
        client = TestClient(app)
        manager.create_sandbox("ws-ap5")
        _write_ws(manager, "ws-ap5", "a.txt", b"A")
        _write_ws(manager, "ws-ap5", "b.txt", b"B")
        resp = client.post(
            "/api/sandbox/ws-ap5/apply", json={"diff_hash": "whatever"}
        )
        assert resp.status_code == 413

    def test_destroy_still_last_with_new_paths(self):
        rs = _load_routes_sandbox()
        paths = [
            (r.path, sorted(r.methods)) for r in rs.router.routes
        ]
        flat = [p for p, _m in paths]
        assert "/api/sandbox/{session_id}/diff" in flat
        assert "/api/sandbox/{session_id}/confirm-deletions" in flat
        assert "/api/sandbox/{session_id}/apply" in flat
        last_path, last_methods = paths[-1]
        assert last_path == "/api/sandbox/{session_id}"
        assert last_methods == ["DELETE"]


# ---------------------------------------------------------------------------
# 8. Registrations: spec, cartography, frontend, yaml, pinned files
# ---------------------------------------------------------------------------

def _check_svelte_source(relpath: str) -> str:
    """Tag-balance + token hygiene by source (the project discipline)."""
    src = _read(relpath)
    markup = re.sub(r"<!--.*?-->", "", src, flags=re.S)
    style_bodies = re.findall(r"<style[^>]*>(.*?)</style>", markup, flags=re.S)
    script_bodies = re.findall(
        r"<script[^>]*>(.*?)</script>", markup, flags=re.S
    )
    markup_wo = re.sub(r"<script[^>]*>.*?</script>", "", markup, flags=re.S)
    markup_wo = re.sub(r"<style[^>]*>.*?</style>", "", markup_wo, flags=re.S)
    opens = len(re.findall(r"\{#(if|each|await|key)", markup_wo))
    closes = len(re.findall(r"\{/(if|each|await|key)\}", markup_wo))
    assert opens == closes, f"{relpath}: svelte block imbalance"
    void = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr"}
    stack: list[str] = []
    tag_re = (r"<(/?)([a-zA-Z][a-zA-Z0-9:.\-]*)"
              r"((?:\"[^\"]*\"|'[^']*'|\{[^}]*\}|[^>])*?)(/?)>")
    for m in re.finditer(tag_re, markup_wo):
        closing, name, selfclose = m.group(1), m.group(2).lower(), m.group(4)
        if name in void or selfclose == "/":
            continue
        if closing:
            assert stack and stack[-1] == name, (
                f"{relpath}: mismatched </{name}>"
            )
            stack.pop()
        else:
            stack.append(name)
    assert not stack, f"{relpath}: unclosed tags {stack}"
    for body in style_bodies + [markup_wo] + script_bodies:
        for m in re.finditer(r"#[0-9a-fA-F]{3,8}\b", body):
            ctx = body[max(0, m.start() - 60):m.start()]
            assert re.search(r"var\(--oo-[a-z0-9\-]+,\s*$", ctx), (
                f"{relpath}: raw hex outside var(--oo-*) fallback"
            )
    return src


class TestRegistrations:
    def test_spec_63_status(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "### 6.3 Status (S212)" in spec
        assert "The consumer LANDED at S212" in spec
        assert "mount/<rel>` maps onto `cloned_root/<rel>`" in spec
        assert "refuses 400 -- never guess" in spec

    def test_spec_cartography(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "generate_workspace_diff" in spec
        assert "apply_workspace_changes" in spec
        assert "EXTENDED S212: `GET /{id}/diff`" in spec
        assert "SandboxDiffReview.svelte" in spec
        assert "coding_agent.py` itself stays byte-identical" in spec
        assert "write-once `cloned_mount` pair" in spec

    def test_spec_section15_row(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "tests/test_s212_sandbox_bloc3.py" in spec
        assert "ADVERSARIAL path suite FIRST" in spec

    def test_frontend_redesign_rows(self):
        frd = _read("FRONTEND_REDESIGN_SPEC.md")
        assert "| `SandboxDiffReview.svelte` | NEW | S212 |" in frd
        # The S210/S211 pinned rows survive, note-extended.
        assert ("Workspace manager panel (Sandbox Workspace cycle, Bloc 1)"
                in frd)
        assert "S212 (Bloc 3): gains the Review-and-apply card" in frd
        assert ("the Review-and-apply card targets the same Select" in frd)

    def test_component_source_clean(self):
        src = _check_svelte_source(
            "frontend/src/lib/components/panels/SandboxDiffReview.svelte"
        )
        assert "getDiff" in src
        assert "confirmDeletions" in src
        assert "applyChanges" in src
        assert "never inside an approve-all" in src.lower() or \
            "never included in an approve-all" in src.lower()
        assert "diff_hash" in src

    def test_panel_mounts_review_child(self):
        src = _check_svelte_source(
            "frontend/src/lib/components/panels/SandboxPanel.svelte"
        )
        assert "import SandboxDiffReview from './SandboxDiffReview.svelte';" \
            in src
        assert "<SandboxDiffReview" in src
        assert "handleApplied" in src
        assert "Review and apply" in src

    def test_client_and_types(self):
        ts = _read("frontend/src/lib/api/sandbox.ts")
        for fn in ("export async function getDiff",
                   "export async function confirmDeletions",
                   "export async function applyChanges"):
            assert fn in ts
        assert "/diff" in ts and "/confirm-deletions" in ts and "/apply" in ts
        ty = _read("frontend/src/lib/types.ts")
        for iface in ("SandboxDiffEntry", "SandboxDiffResponse",
                      "SandboxConfirmDeletionsResponse",
                      "SandboxApplyRequest", "SandboxApplyResponse"):
            assert f"export interface {iface}" in ty
        assert "cloned_mount: string | null;" in ty

    def test_yaml_and_config_clamp(self):
        import yaml
        raw = yaml.safe_load(_read("opti_oignon/config/sandbox.yaml"))
        assert raw.get("diff_max_entries") == 50000
        assert SandboxConfig(diff_max_entries=0).diff_max_entries == 1
        assert SandboxConfig(
            diff_max_entries=10 ** 9
        ).diff_max_entries == 1000000

    def test_schemas_registered(self):
        src = _read("opti_oignon/api/schemas.py")
        for name in ("SandboxDiffEntry", "SandboxDiffResponse",
                     "SandboxConfirmDeletionsRequest",
                     "SandboxConfirmDeletionsResponse",
                     "SandboxApplyRequest", "SandboxApplyResponse"):
            assert f"class {name}(BaseModel)" in src

    def test_coding_agent_pinned_edit_free(self):
        src = _read("opti_oignon/coding_agent.py")
        assert "S212" not in src
        # The generalization did NOT land here: the content-coupled
        # machinery is untouched (difflib diff, plain write in apply).
        assert "difflib.unified_diff" in src
        assert 'with open(dest, "w", encoding="utf-8") as fh:' in src
        assert "generate_workspace_diff" not in src

    def test_dispatch_pinned_edit_free(self):
        src = _read(os.path.join("opti_oignon", "agent", "dispatch.py"))
        assert "S212" not in src
        for name in ("apply_workspace_changes", "generate_workspace_diff",
                     "confirm_deletions"):
            assert name not in src

    def test_workspace_module_conventions(self):
        assert _ws.checkpoint_before_apply is True
        assert callable(_ws.reset_workspace_manifests)
        assert callable(_ws.reset_workspace_bindings)
        for exc_name in ("WorkspaceDiffError", "WorkspaceDiffBoundExceeded",
                         "WorkspaceReviewDrift", "WorkspaceApplyTargetError"):
            assert issubclass(getattr(_ws, exc_name), Exception)

    def test_odysseus_spec_untouched(self):
        spec = _read("ODYSSEUS_SPEC.md")
        assert "S212" not in spec
