#!/usr/bin/env python3
"""S211 -- Sandbox Workspace cycle, Bloc 2: copy-in.

Per-fix suite for SANDBOX_WORKSPACE_SPEC section 5 (drag-and-drop upload,
allowlisted host browse, symlink-safe host clone) and the landed half of
section 6.1 (the baseline manifest recorded at copy-in).

Coverage: the upload path (caps summed BEFORE any write, whole-request 413
refusals with the workspace untouched, per-file sanitization and collision
refusals reported honestly, on-the-fly sha256); the browse confinement
(roots normalized at load with "/" refused, confinement BEFORE existence so
outside-roots answers 403 with no existence leak, symlinks displayed
without follow and targets undisclosed, hidden flagged not omitted); the
clone (exact pre-walk refusing on byte/file caps AND the remaining S210
quota before any copy, the depth bound refusing instead of undercounting,
symlinks and specials skipped and counted with targets never exposed,
destination collision 409, the manifest recorded and hash-stable, the
cloned root write-once); the WorkspaceManifests store; the routes via the
fastapi TestClient (multipart included); the audit rows; and the spec /
cartography / FRONTEND_REDESIGN / yaml / source registrations.

Harness: the s210 `_load_fresh` shape -- other suites in the sweep
(test_file_tools) pre-load the real package chain, so this file ALWAYS
execs its own module copies and re-pins them per test.
"""

import importlib.util
import io
import json
import os
import re
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
    those would split exception-class identity (WorkspaceQuotaExceeded
    raised from one copy, caught against another in the routes). Temporarily
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
_ssec = _load_fresh(
    os.path.join("opti_oignon", "sandbox_seccomp.py"),
    register="opti_oignon.sandbox_seccomp",
)

SandboxConfig = _sm.SandboxConfig


@pytest.fixture(autouse=True)
def _bind_module_copies(monkeypatch):
    """Bind THIS file's module copies for the duration of each test."""
    pairs = {
        "opti_oignon.sandbox_manager": _sm,
        "opti_oignon.sandbox_tools": _st,
        "opti_oignon.sandbox_workspace": _ws,
        "opti_oignon.sandbox_seccomp": _ssec,
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


def _read(relpath: str) -> str:
    with open(os.path.join(_ROOT, relpath), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Config keys, clamps, and share-root normalization
# ---------------------------------------------------------------------------

class TestConfigKeys:
    def test_defaults(self):
        cfg = SandboxConfig()
        assert cfg.upload_max_files == 64
        assert cfg.upload_max_file_bytes == 128 * 1024 ** 2
        assert cfg.clone_max_bytes == 512 * 1024 ** 2
        assert cfg.clone_max_files == 20000

    def test_clamps_correct_never_disable(self):
        cfg = SandboxConfig(
            upload_max_files=0,
            upload_max_file_bytes=1,
            clone_max_bytes=10 ** 18,
            clone_max_files=-5,
        )
        assert cfg.upload_max_files == 1
        assert cfg.upload_max_file_bytes == 1 * 1024 ** 2
        assert cfg.clone_max_bytes == 16 * 1024 ** 3
        assert cfg.clone_max_files == 1

    def test_share_roots_realpathed_and_root_refused(self, tmp_path):
        real = tmp_path / "ok"
        real.mkdir()
        cfg = SandboxConfig(host_share_roots=["/", str(real)])
        assert os.path.realpath(str(real)) in cfg.host_share_roots
        assert os.path.realpath(os.sep) not in cfg.host_share_roots

    def test_share_roots_nonexistent_dropped(self, tmp_path):
        cfg = SandboxConfig(
            host_share_roots=[str(tmp_path / "missing-dir")]
        )
        assert cfg.host_share_roots == []

    def test_share_roots_empty_defaults_to_home(self):
        roots = _sm._normalize_share_roots([])
        home = os.path.realpath(os.path.expanduser("~"))
        if home != os.path.realpath(os.sep) and os.path.isdir(home):
            assert roots == [home]
        else:  # pragma: no cover - root-homed edge container
            assert roots == []

    def test_share_roots_deduplicated(self, tmp_path):
        d = tmp_path / "dup"
        d.mkdir()
        cfg = SandboxConfig(host_share_roots=[str(d), str(d)])
        assert cfg.host_share_roots.count(os.path.realpath(str(d))) == 1

    def test_yaml_carries_the_five_keys(self):
        raw = _read(os.path.join("opti_oignon", "config", "sandbox.yaml"))
        for key in (
            "host_share_roots",
            "upload_max_files",
            "upload_max_file_bytes",
            "clone_max_bytes",
            "clone_max_files",
        ):
            assert key in raw

    def test_loader_reads_the_five_keys(self):
        src = _read(os.path.join("opti_oignon", "sandbox_manager.py"))
        loader = src.split("def _load_config", 1)[1].split("def ", 1)[0]
        for key in (
            "host_share_roots",
            "upload_max_files",
            "upload_max_file_bytes",
            "clone_max_bytes",
            "clone_max_files",
        ):
            assert key in loader


# ---------------------------------------------------------------------------
# Upload: sanitization, caps, per-file honesty, manifest hashes
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    @pytest.mark.parametrize("bad", ["", "   ", ".", "..", "a/b", "a\\b",
                                     "x\x00y"])
    def test_refusals(self, bad):
        ok, _reason = _sm.SandboxManager.sanitize_upload_filename(bad)
        assert ok is False

    def test_plain_name_passes(self):
        ok, clean = _sm.SandboxManager.sanitize_upload_filename(" notes.txt ")
        assert ok is True
        assert clean == "notes.txt"


class TestUploadManager:
    def test_writes_and_hashes(self, manager):
        manager.create_sandbox("ws-u")
        result = manager.upload_files(
            "ws-u", [("a.txt", io.BytesIO(b"hello"), 5)]
        )
        assert result["written_bytes"] == 5
        entry = result["written"][0]
        assert entry["relative_path"] == "a.txt"
        ws = manager.get_workspace_path("ws-u")
        assert open(os.path.join(ws, "a.txt"), "rb").read() == b"hello"
        assert entry["sha256"] == _ws.manifest_hash_file(
            os.path.join(ws, "a.txt")
        )

    def test_per_file_refusals_honest(self, manager):
        manager.create_sandbox("ws-u")
        manager.upload_files("ws-u", [("a.txt", io.BytesIO(b"x"), 1)])
        result = manager.upload_files(
            "ws-u",
            [
                ("../evil", io.BytesIO(b"x"), 1),
                ("a.txt", io.BytesIO(b"dup"), 3),
                ("ok.txt", io.BytesIO(b"fine"), 4),
            ],
        )
        reasons = {r["name"]: r["reason"] for r in result["refused"]}
        assert "../evil" in reasons
        assert "not overwritten" in reasons["a.txt"]
        assert [w["name"] for w in result["written"]] == ["ok.txt"]
        ws = manager.get_workspace_path("ws-u")
        assert open(os.path.join(ws, "a.txt"), "rb").read() == b"x"

    def test_dest_subdir_valid_and_invalid(self, manager):
        manager.create_sandbox("ws-u")
        result = manager.upload_files(
            "ws-u", [("b.txt", io.BytesIO(b"bb"), 2)], dest_subdir="sub/dir"
        )
        assert result["written"][0]["relative_path"] == "sub/dir/b.txt"
        with pytest.raises(ValueError):
            manager.upload_files(
                "ws-u", [("c.txt", io.BytesIO(b"c"), 1)],
                dest_subdir="../out",
            )

    def test_count_cap_refuses_whole_request(self, manager):
        manager.create_sandbox("ws-u")
        manager._config.upload_max_files = 1
        with pytest.raises(_sm.WorkspaceQuotaExceeded):
            manager.upload_files(
                "ws-u",
                [
                    ("a.txt", io.BytesIO(b"a"), 1),
                    ("b.txt", io.BytesIO(b"b"), 1),
                ],
            )
        ws = manager.get_workspace_path("ws-u")
        assert not os.path.exists(os.path.join(ws, "a.txt"))

    def test_per_file_cap_refuses_whole_request(self, manager):
        manager.create_sandbox("ws-u")
        manager._config.upload_max_file_bytes = 4
        with pytest.raises(_sm.WorkspaceQuotaExceeded):
            manager.upload_files(
                "ws-u",
                [
                    ("small.txt", io.BytesIO(b"ok"), 2),
                    ("big.txt", io.BytesIO(b"toolarge"), 8),
                ],
            )
        ws = manager.get_workspace_path("ws-u")
        assert not os.path.exists(os.path.join(ws, "small.txt"))

    def test_quota_summed_before_write(self, manager):
        manager.create_sandbox("ws-u")
        manager._config.disk_soft_limit_bytes = 10
        with pytest.raises(_sm.WorkspaceQuotaExceeded):
            manager.upload_files(
                "ws-u",
                [
                    ("a.txt", io.BytesIO(b"123456"), 6),
                    ("b.txt", io.BytesIO(b"123456"), 6),
                ],
            )
        ws = manager.get_workspace_path("ws-u")
        assert os.listdir(ws) == []

    def test_touch_activity_and_audit(self, manager):
        session = manager.create_sandbox("ws-u")
        before = session.last_activity
        manager.upload_files("ws-u", [("a.txt", io.BytesIO(b"x"), 1)])
        assert manager.get_session("ws-u").last_activity >= before
        actions = [
            r["action"] for r in manager.audit.get_approval_log("ws-u")
        ]
        assert "workspace_upload" in actions


# ---------------------------------------------------------------------------
# Host browse: confinement before existence, symlink/hidden policy, audit
# ---------------------------------------------------------------------------

class TestBrowse:
    def test_no_path_lists_roots(self, manager, share_root):
        listing = manager.browse_host(None)
        assert listing["path"] == ""
        assert str(share_root) in [
            os.path.realpath(r) for r in listing["roots"]
        ] or os.path.realpath(str(share_root)) in listing["roots"]
        assert all(e["type"] == "dir" for e in listing["entries"])

    def test_listing_types_and_hidden_flag(self, manager, share_root):
        (share_root / "sub").mkdir()
        (share_root / "f.txt").write_text("data")
        (share_root / ".secret").write_text("h")
        os.symlink("/etc/passwd", str(share_root / "leak"))
        listing = manager.browse_host(str(share_root))
        by_name = {e["name"]: e for e in listing["entries"]}
        assert by_name["sub"]["type"] == "dir"
        assert by_name["f.txt"]["type"] == "file"
        assert by_name["f.txt"]["size"] == 4
        assert by_name[".secret"]["hidden"] is True
        assert by_name["leak"]["type"] == "symlink"
        assert by_name["leak"]["size"] == 0

    def test_symlink_target_never_disclosed(self, manager, share_root):
        os.symlink("/etc/passwd", str(share_root / "leak"))
        listing = manager.browse_host(str(share_root))
        assert "/etc/passwd" not in json.dumps(listing)

    def test_outside_roots_403_before_existence(self, manager, tmp_path):
        outside_missing = str(tmp_path / "nope" / "missing")
        with pytest.raises(PermissionError):
            manager.browse_host(outside_missing)
        with pytest.raises(PermissionError):
            manager.browse_host("/etc")

    def test_inside_missing_is_404_class(self, manager, share_root):
        with pytest.raises(ValueError):
            manager.browse_host(str(share_root / "missing"))
        (share_root / "plain.txt").write_text("x")
        with pytest.raises(ValueError):
            manager.browse_host(str(share_root / "plain.txt"))

    def test_symlink_escape_resolves_outside_and_403s(
        self, manager, share_root
    ):
        os.symlink("/etc", str(share_root / "etclink"))
        with pytest.raises(PermissionError):
            manager.browse_host(str(share_root / "etclink"))

    def test_browse_audited_under_sentinel(self, manager, share_root):
        manager.browse_host(str(share_root))
        actions = [
            r["action"]
            for r in manager.audit.get_approval_log("host-browse")
        ]
        assert "host_browse" in actions

    def test_empty_roots_refuse_everything(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._config.host_share_roots = []
        with pytest.raises(PermissionError):
            mgr.browse_host(str(tmp_path))


# ---------------------------------------------------------------------------
# Clone: exact pre-walk, caps + quota, symlink safety, specials, manifest
# ---------------------------------------------------------------------------

def _make_tree(share_root):
    src = share_root / "proj"
    (src / "sub").mkdir(parents=True)
    (src / "f1.txt").write_text("one")
    (src / ".hidden").write_text("h")
    (src / "sub" / "f2.txt").write_text("two-two")
    return src


class TestClone:
    def test_happy_path_counts_and_manifest(self, manager, share_root):
        src = _make_tree(share_root)
        os.symlink("/etc/passwd", str(src / "leak"))
        manager.create_sandbox("ws-c")
        result = manager.clone_directory("ws-c", str(src))
        assert result["dest"] == "proj"
        assert result["copied_files"] == 3
        assert result["skipped_symlinks"] == 1
        assert result["cloned_root"] == os.path.realpath(str(src))
        ws = manager.get_workspace_path("ws-c")
        assert not os.path.lexists(os.path.join(ws, "proj", "leak"))
        assert set(result["manifest"]) == {
            "proj/f1.txt", "proj/.hidden", "proj/sub/f2.txt"
        }
        for rel, digest in result["manifest"].items():
            assert digest == _ws.manifest_hash_file(os.path.join(ws, rel))

    def test_internal_symlink_also_skipped(self, manager, share_root):
        src = _make_tree(share_root)
        os.symlink(str(src / "f1.txt"), str(src / "inlink"))
        manager.create_sandbox("ws-c")
        result = manager.clone_directory("ws-c", str(src))
        assert result["skipped_symlinks"] == 1
        ws = manager.get_workspace_path("ws-c")
        assert not os.path.lexists(os.path.join(ws, "proj", "inlink"))

    def test_special_files_skipped(self, manager, share_root):
        src = _make_tree(share_root)
        os.mkfifo(str(src / "pipe"))
        manager.create_sandbox("ws-c")
        result = manager.clone_directory("ws-c", str(src))
        assert result["skipped_special"] == 1
        ws = manager.get_workspace_path("ws-c")
        assert not os.path.lexists(os.path.join(ws, "proj", "pipe"))

    def test_byte_cap_refuses_before_any_copy(self, manager, share_root):
        src = _make_tree(share_root)
        manager.create_sandbox("ws-c")
        manager._config.clone_max_bytes = 2
        with pytest.raises(_sm.WorkspaceQuotaExceeded):
            manager.clone_directory("ws-c", str(src))
        ws = manager.get_workspace_path("ws-c")
        assert not os.path.exists(os.path.join(ws, "proj"))

    def test_file_count_cap_refuses(self, manager, share_root):
        src = _make_tree(share_root)
        manager.create_sandbox("ws-c")
        manager._config.clone_max_files = 2
        with pytest.raises(_sm.WorkspaceQuotaExceeded):
            manager.clone_directory("ws-c", str(src))

    def test_remaining_quota_also_bounds(self, manager, share_root):
        src = _make_tree(share_root)
        manager.create_sandbox("ws-c")
        manager._config.disk_soft_limit_bytes = 4
        with pytest.raises(_sm.WorkspaceQuotaExceeded) as exc:
            manager.clone_directory("ws-c", str(src))
        assert "quota" in str(exc.value)

    def test_depth_bound_refuses_not_undercounts(self, manager, share_root):
        deep = share_root / "deep"
        path = deep
        for i in range(_sm._CLONE_WALK_MAX_DEPTH + 2):
            path = path / f"d{i}"
        path.mkdir(parents=True)
        (path / "leaf.txt").write_text("x")
        manager.create_sandbox("ws-c")
        with pytest.raises(_sm.WorkspaceQuotaExceeded):
            manager.clone_directory("ws-c", str(deep))

    def test_collision_refused(self, manager, share_root):
        src = _make_tree(share_root)
        manager.create_sandbox("ws-c")
        manager.clone_directory("ws-c", str(src))
        with pytest.raises(FileExistsError):
            manager.clone_directory("ws-c", str(src))

    def test_dest_subdir(self, manager, share_root):
        src = _make_tree(share_root)
        manager.create_sandbox("ws-c")
        result = manager.clone_directory("ws-c", str(src), dest_subdir="in")
        assert result["dest"] == "in/proj"
        with pytest.raises(ValueError):
            manager.clone_directory("ws-c", str(src), dest_subdir="../o")

    def test_source_outside_roots_403(self, manager, tmp_path):
        manager.create_sandbox("ws-c")
        with pytest.raises(PermissionError):
            manager.clone_directory("ws-c", "/etc")

    def test_source_not_a_directory_404_class(self, manager, share_root):
        (share_root / "file.txt").write_text("x")
        manager.create_sandbox("ws-c")
        with pytest.raises(ValueError):
            manager.clone_directory("ws-c", str(share_root / "file.txt"))

    def test_clone_audited(self, manager, share_root):
        src = _make_tree(share_root)
        manager.create_sandbox("ws-c")
        manager.clone_directory("ws-c", str(src))
        actions = [
            r["action"] for r in manager.audit.get_approval_log("ws-c")
        ]
        assert "host_clone" in actions


# ---------------------------------------------------------------------------
# WorkspaceManifests store + manifest_hash_file
# ---------------------------------------------------------------------------

class TestManifestStore:
    def test_singleton_and_reset(self):
        a = _ws.get_workspace_manifests()
        assert _ws.get_workspace_manifests() is a
        _ws.reset_workspace_manifests()
        assert _ws.get_workspace_manifests() is not a

    def test_record_merges_and_counts(self):
        store = _ws.WorkspaceManifests()
        assert store.record("ws", {"a": "1"}) == 1
        assert store.record("ws", {"b": "2"}) == 2
        assert store.get_manifest("ws") == {"a": "1", "b": "2"}

    def test_cloned_root_write_once(self):
        store = _ws.WorkspaceManifests()
        store.record("ws", {"a": "1"}, cloned_root="/first")
        store.record("ws", {"b": "2"}, cloned_root="/second")
        assert store.get_cloned_root("ws") == "/first"

    def test_upload_only_has_no_root(self):
        store = _ws.WorkspaceManifests()
        store.record("ws", {"a": "1"})
        assert store.get_cloned_root("ws") is None

    def test_drop_forgets_everything(self):
        store = _ws.WorkspaceManifests()
        store.record("ws", {"a": "1"}, cloned_root="/r")
        store.drop("ws")
        assert store.get_manifest("ws") is None
        assert store.get_cloned_root("ws") is None

    def test_get_returns_a_copy(self):
        store = _ws.WorkspaceManifests()
        store.record("ws", {"a": "1"})
        got = store.get_manifest("ws")
        got["a"] = "tampered"
        assert store.get_manifest("ws") == {"a": "1"}

    def test_hash_stable_and_content_sensitive(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"payload")
        h1 = _ws.manifest_hash_file(str(f))
        h2 = _ws.manifest_hash_file(str(f))
        assert h1 == h2
        assert len(h1) == 64
        f.write_bytes(b"payload!")
        assert _ws.manifest_hash_file(str(f)) != h1


# ---------------------------------------------------------------------------
# Routes (fastapi TestClient, multipart included)
# ---------------------------------------------------------------------------

_ROUTES_SANDBOX_CACHE = {}


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


def _mp(*named: tuple[str, bytes]):
    """Build the TestClient multipart files payload."""
    return [
        ("files", (name, io.BytesIO(data), "application/octet-stream"))
        for name, data in named
    ]


class TestUploadRoute:
    def test_single_and_manifest_recorded(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-r")
        resp = client.post(
            "/api/sandbox/ws-r/upload", files=_mp(("a.txt", b"hello"))
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["uploaded_paths"] == ["a.txt"]
        assert body["uploaded_bytes"] == 5
        assert body["manifest_files"] == 1
        manifest = _ws.get_workspace_manifests().get_manifest("ws-r")
        assert "a.txt" in manifest

    def test_multi_with_dest_subdir(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-r")
        resp = client.post(
            "/api/sandbox/ws-r/upload",
            files=_mp(("a.txt", b"a"), ("b.txt", b"bb")),
            data={"dest_subdir": "in"},
        )
        assert resp.status_code == 200
        assert sorted(resp.json()["uploaded_paths"]) == [
            "in/a.txt", "in/b.txt"
        ]

    def test_per_file_refusals_in_200(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-r")
        client.post("/api/sandbox/ws-r/upload", files=_mp(("a.txt", b"x")))
        resp = client.post(
            "/api/sandbox/ws-r/upload",
            files=_mp(("a.txt", b"dup"), ("ok.txt", b"y")),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["uploaded_paths"] == ["ok.txt"]
        assert body["refused"][0]["name"] == "a.txt"

    def test_count_cap_413(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-r")
        manager._config.upload_max_files = 1
        resp = client.post(
            "/api/sandbox/ws-r/upload",
            files=_mp(("a.txt", b"a"), ("b.txt", b"b")),
        )
        assert resp.status_code == 413

    def test_per_file_cap_413(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-r")
        manager._config.upload_max_file_bytes = 2
        resp = client.post(
            "/api/sandbox/ws-r/upload", files=_mp(("big.txt", b"toolarge"))
        )
        assert resp.status_code == 413

    def test_quota_413_workspace_untouched(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-r")
        manager._config.disk_soft_limit_bytes = 3
        resp = client.post(
            "/api/sandbox/ws-r/upload", files=_mp(("a.txt", b"123456"))
        )
        assert resp.status_code == 413
        ws = manager.get_workspace_path("ws-r")
        assert os.listdir(ws) == []

    def test_unknown_404_foreign_403(self, api):
        rs, client, manager = api
        resp = client.post(
            "/api/sandbox/nope/upload", files=_mp(("a.txt", b"x"))
        )
        assert resp.status_code == 404
        manager.create_sandbox("ws-f", owner_user_id="someone-else")
        resp = client.post(
            "/api/sandbox/ws-f/upload", files=_mp(("a.txt", b"x"))
        )
        assert resp.status_code == 403

    def test_invalid_dest_subdir_400(self, api):
        rs, client, manager = api
        manager.create_sandbox("ws-r")
        resp = client.post(
            "/api/sandbox/ws-r/upload",
            files=_mp(("a.txt", b"x")),
            data={"dest_subdir": "../out"},
        )
        assert resp.status_code == 400


class TestBrowseRoute:
    def test_roots_then_listing(self, api, share_root):
        rs, client, _ = api
        resp = client.get("/api/sandbox/host/browse")
        assert resp.status_code == 200
        assert resp.json()["path"] == ""
        (share_root / "f.txt").write_text("data")
        resp = client.get(
            "/api/sandbox/host/browse", params={"path": str(share_root)}
        )
        assert resp.status_code == 200
        names = [e["name"] for e in resp.json()["entries"]]
        assert "f.txt" in names

    def test_outside_403_inside_missing_404(self, api, share_root):
        rs, client, _ = api
        resp = client.get(
            "/api/sandbox/host/browse", params={"path": "/etc"}
        )
        assert resp.status_code == 403
        resp = client.get(
            "/api/sandbox/host/browse",
            params={"path": str(share_root / "missing")},
        )
        assert resp.status_code == 404


class TestCloneRoute:
    def test_clone_200_and_manifest_with_root(self, api, share_root):
        rs, client, manager = api
        src = _make_tree(share_root)
        manager.create_sandbox("ws-r")
        resp = client.post(
            "/api/sandbox/ws-r/clone", json={"src_path": str(src)}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["dest"] == "proj"
        assert body["copied_files"] == 3
        assert body["manifest_files"] == 3
        store = _ws.get_workspace_manifests()
        assert store.get_cloned_root("ws-r") == os.path.realpath(str(src))
        assert "proj/f1.txt" in store.get_manifest("ws-r")

    def test_clone_codes(self, api, share_root):
        rs, client, manager = api
        src = _make_tree(share_root)
        manager.create_sandbox("ws-r")
        assert client.post(
            "/api/sandbox/nope/clone", json={"src_path": str(src)}
        ).status_code == 404
        manager.create_sandbox("ws-f", owner_user_id="someone-else")
        assert client.post(
            "/api/sandbox/ws-f/clone", json={"src_path": str(src)}
        ).status_code == 403
        assert client.post(
            "/api/sandbox/ws-r/clone", json={"src_path": "/etc"}
        ).status_code == 403
        assert client.post(
            "/api/sandbox/ws-r/clone",
            json={"src_path": str(share_root / "missing")},
        ).status_code == 404
        manager._config.clone_max_bytes = 2
        assert client.post(
            "/api/sandbox/ws-r/clone", json={"src_path": str(src)}
        ).status_code == 413
        manager._config.clone_max_bytes = 512 * 1024 ** 2
        assert client.post(
            "/api/sandbox/ws-r/clone", json={"src_path": str(src)}
        ).status_code == 200
        assert client.post(
            "/api/sandbox/ws-r/clone", json={"src_path": str(src)}
        ).status_code == 409
        assert client.post(
            "/api/sandbox/ws-r/clone",
            json={"src_path": str(src), "dest_subdir": "../o"},
        ).status_code == 400

    def test_refusal_audited(self, api, share_root):
        rs, client, manager = api
        src = _make_tree(share_root)
        manager.create_sandbox("ws-r")
        manager._config.clone_max_files = 1
        client.post("/api/sandbox/ws-r/clone", json={"src_path": str(src)})
        actions = [
            r["action"] for r in manager.audit.get_approval_log("ws-r")
        ]
        assert "host_clone_refused" in actions

    def test_destroy_drops_manifest(self, api, share_root):
        rs, client, manager = api
        src = _make_tree(share_root)
        manager.create_sandbox("ws-r")
        client.post("/api/sandbox/ws-r/clone", json={"src_path": str(src)})
        store = _ws.get_workspace_manifests()
        assert store.get_manifest("ws-r") is not None
        assert client.delete("/api/sandbox/ws-r").status_code == 200
        assert store.get_manifest("ws-r") is None
        assert store.get_cloned_root("ws-r") is None

    def test_destroy_still_last_route(self, api):
        rs, client, _ = api
        paths = [
            (r.path, sorted(r.methods))
            for r in rs.router.routes
        ]
        assert paths[-1] == ("/api/sandbox/{session_id}", ["DELETE"])
        flat = [p for p, _m in paths]
        assert "/api/sandbox/{session_id}/upload" in flat
        assert "/api/sandbox/host/browse" in flat
        assert "/api/sandbox/{session_id}/clone" in flat


# ---------------------------------------------------------------------------
# Registrations: spec, cartography, FRONTEND_REDESIGN, UI + client by source
# ---------------------------------------------------------------------------

class TestRegistrations:
    def test_spec_section_5_status_landed(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "### 5.3 Status (S211)" in spec
        assert "Bloc 2 is LANDED" in spec

    def test_spec_61_status_note_names_bloc3(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "Status note (S211)" in spec
        assert "Bloc 3 is the\nconsumer" in spec or "Bloc 3 is the consumer" in spec

    def test_spec_section_12_registers_the_landings(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        for needle in (
            "clone_directory",
            "WorkspaceManifests",
            "manifest_hash_file",
            "SandboxUploadZone.svelte",
            "SandboxHostExplorer.svelte",
            "apiUpload",
        ):
            assert needle in spec

    def test_spec_section_15_gains_the_s211_row(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "tests/test_s211_sandbox_bloc2.py" in spec

    def test_frontend_spec_registers_the_components(self):
        spec = _read("FRONTEND_REDESIGN_SPEC.md")
        assert re.search(
            r"SandboxUploadZone\.svelte`?\s*\|\s*NEW\s*\|\s*S211", spec
        )
        assert re.search(
            r"SandboxHostExplorer\.svelte`?\s*\|\s*NEW\s*\|\s*S211", spec
        )
        # The S210 rows survive (pinned by test_s210_sandbox_bloc1).
        assert re.search(r"SandboxPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S210", spec)
        assert re.search(
            r"SandboxWorkspaceList\.svelte`?\s*\|\s*NEW\s*\|\s*S210", spec
        )

    def test_components_exist_tag_balanced_token_only(self):
        for rel in (
            os.path.join(
                "frontend", "src", "lib", "components", "panels",
                "SandboxUploadZone.svelte",
            ),
            os.path.join(
                "frontend", "src", "lib", "components", "panels",
                "SandboxHostExplorer.svelte",
            ),
            os.path.join(
                "frontend", "src", "lib", "components", "panels",
                "SandboxPanel.svelte",
            ),
        ):
            src = _read(rel)
            for tag in ("script", "style", "section", "ul", "li", "div",
                        "span", "button", "p"):
                opens = len(re.findall(rf"<{tag}[\s>]", src))
                closes = len(re.findall(rf"</{tag}>", src))
                selfc = len(re.findall(rf"<{tag}\b[^>]*/>", src, re.S))
                assert opens == closes + selfc, f"{rel}: <{tag}> unbalanced"
            for hex_match in re.finditer(r"#[0-9a-fA-F]{3,8}\b", src):
                prefix = src[max(0, hex_match.start() - 60):hex_match.start()]
                assert re.search(r"var\(--oo-[a-z0-9-]+,\s*$", prefix), (
                    f"{rel}: hex outside var(--oo-*, ...) fallback"
                )

    def test_panel_mounts_the_copyin_card(self):
        src = _read(os.path.join(
            "frontend", "src", "lib", "components", "panels",
            "SandboxPanel.svelte",
        ))
        assert "SandboxUploadZone" in src
        assert "SandboxHostExplorer" in src
        assert "copyTargetId" in src

    def test_sandbox_ts_exports_the_three_functions(self):
        src = _read(os.path.join("frontend", "src", "lib", "api", "sandbox.ts"))
        for fn in ("uploadFiles", "browseHost", "cloneDirectory"):
            assert f"export async function {fn}" in src
        assert "apiUpload" in src

    def test_client_ts_exports_api_upload(self):
        src = _read(os.path.join("frontend", "src", "lib", "api", "client.ts"))
        assert "export async function apiUpload" in src
        assert "FormData" in src

    def test_types_ts_carries_the_s211_types(self):
        src = _read(os.path.join("frontend", "src", "lib", "types.ts"))
        for t in (
            "SandboxUploadResponse",
            "HostBrowseEntry",
            "HostBrowseResponse",
            "SandboxCloneRequest",
            "SandboxCloneResponse",
        ):
            assert f"export interface {t}" in src

    def test_workspace_module_conventions_hold(self):
        src = _read(os.path.join("opti_oignon", "sandbox_workspace.py"))
        assert "checkpoint_before_apply = True" in src
        for needle in (
            "class WorkspaceManifests",
            "def manifest_hash_file",
            "def get_workspace_manifests",
            "def reset_workspace_manifests",
        ):
            assert needle in src

    def test_manager_inject_directory_untouched_no_s211(self):
        src = _read(os.path.join("opti_oignon", "sandbox_manager.py"))
        body = src.split("def inject_directory", 1)[1]
        body = body.split("def _check_disk_quota", 1)[0]
        assert "S211" not in body
        assert "copytree" in body

    def test_no_model_trigger_surface(self):
        """The dispatch carries no S211 surface: copy-in is user-only."""
        src = _read(os.path.join("opti_oignon", "agent", "dispatch.py"))
        assert "S211" not in src
        for needle in ("upload_files", "clone_directory", "browse_host"):
            assert needle not in src
