#!/usr/bin/env python3
"""A plugin runs under bubblewrap, inside walls and limits the server sets.

A plugin is third-party code. Its worker process starts under bubblewrap
whenever the sandbox runs its commands there, sees only what it needs, and
lives under resource limits the server installs before it executes:

  * PB1 -- the posture follows the sandbox: bubblewrap in use gives
    ``bwrap``; without it, strict mode gives ``blocked`` and a lax one
    ``direct``; a sandbox whose state cannot be read gives ``blocked``;
  * PB2 -- a blocked posture starts no process, and the error names the
    cause and the remedy, up to the loader;
  * PB3 -- the walls: the plugin folder, the worker file and the interpreter
    are read-only; the only writable binds are the plugin's socket folder and
    its data folder; /tmp is a fresh tmpfs mounted before them; a protected
    folder inside a bound tree is masked; a bind that would expose the root
    or a home is refused;
  * PB4 -- the network namespace is kept only for a manifest that declares
    ``network_outbound``, and the loader hands the manifest's permissions on;
  * PB5 -- the worker's environment carries no secret of the server's, and
    the HMAC key is never on a command line;
  * PB6 -- the limits are the server's: the lower of the manifest and the
    ceiling, installed on the worker before it executes;
  * PB7 -- a worker whose limits are not in place refuses to serve, naming
    the limit;
  * PB8 -- the seccomp filter goes with a bubblewrap launch, and a filter
    that cannot be built refuses the launch when it is required;
  * PB9 -- a plugin that may write gets a private data folder (0700, under
    the data root, never through a link), seen by its hooks as
    ``metadata["data_dir"]``, a key only the worker sets;
  * PB10 -- the three shipped plugins that keep a store keep it in their
    data folder, and never in the shared temporary directory;
  * PB11 -- /api/health reports the plugin isolation;
  * PB12 -- under the real bubblewrap, a plugin reads nothing outside its
    binds, cannot write its own code, writes its data, has no network, has
    its own PID namespace, and nothing of it is left once it stops;
  * PB13 -- a plugin that reaches for the host package starts fast, on its
    fallback, refused by the worker's own guard even with the package on
    its path (it supersedes c4 of the worker's host-package suite, whose
    server was built on a socket path the worker no longer opens).

The posture and the launcher are faked at their seams (the posture provider,
``subprocess.Popen``); PB6, PB7, PB12 and PB13 run real worker processes.
PB12 needs a bubblewrap that can create namespaces and is skipped by name
otherwise. Package files loaded alone go through the shared isolation
window; the modules are resolved when a contract runs. Local-only.
"""

import importlib
import os
import platform
import re
import secrets
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

MIB = 1024 * 1024

ENTRY = (
    "def hook_pre_inference(context):\n"
    "    return {'marker': 'ran'}\n"
)

PROBE = '''
import os
import socket


def _attempt(action):
    try:
        action()
        return "done"
    except OSError as exc:
        return type(exc).__name__


def hook_pre_inference(context):
    asked = context.data
    out = {}

    def read_outside():
        with open(asked["outside"], encoding="utf-8") as fh:
            fh.read()

    def read_masked():
        with open(asked["masked"], encoding="utf-8") as fh:
            fh.read()

    def write_code():
        with open(os.path.join(asked["code_dir"], "written.txt"), "w") as fh:
            fh.write("x")

    def write_data():
        data_dir = context.metadata.get("data_dir")
        with open(os.path.join(data_dir, "kept.txt"), "w") as fh:
            fh.write("kept")

    def connect():
        socket.create_connection(("127.0.0.1", int(asked["port"])), timeout=2).close()

    out["outside"] = _attempt(read_outside)
    out["masked"] = _attempt(read_masked)
    out["code"] = _attempt(write_code)
    out["data"] = _attempt(write_data)
    out["net"] = _attempt(connect)
    out["pid"] = os.getpid()
    return out
'''

# The sandbox's shipped read-only system binds, restated here so no contract
# imports the sandbox manager (its import builds a singleton).
SYSTEM_BINDS = (
    "/usr", "/bin", "/lib", "/lib64",
    "/etc/ld.so.cache", "/etc/alternatives",
    "/etc/python3", "/etc/localtime", "/etc/ssl/certs",
)


def _iso():
    return importlib.import_module("opti_oignon.plugin_isolation")


def _sub():
    return importlib.import_module("opti_oignon.plugin_subprocess")


def _loader():
    return importlib.import_module("opti_oignon.plugin_loader")


def _worker_path():
    return Path(_sub().__file__).with_name("plugin_worker.py")


def _window(name, *parts):
    """One package source file, loaded alone in the shared isolation window."""
    loaded, restore = isolate(targets={name: source(*parts)})
    return loaded[name], restore


def _make_plugin(base, name, *, entry=ENTRY, permissions=(), resource_limits=None):
    plugin = base / "plugins" / name
    plugin.mkdir(parents=True)
    plugin = plugin.resolve()
    lines = [
        f"name: {name}",
        "version: 1.0.0",
        "author: tester",
        "description: isolation fixture",
        "entry_point: entry_point.py",
        "hooks:",
        "  - pre_inference",
    ]
    if permissions:
        lines.append("permissions:")
        lines.extend(f"  - {perm}" for perm in permissions)
    if resource_limits:
        lines.append("resource_limits:")
        lines.extend(f"  {key}: {value}" for key, value in resource_limits.items())
    (plugin / "manifest.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (plugin / "entry_point.py").write_text(entry, encoding="utf-8")
    return plugin


def _posture(mode, *, strict=True, seccomp=False, required=True, protected=()):
    return _iso().PluginPosture(
        mode=mode,
        strict=strict,
        reason=f"{mode} posture set by the contract",
        ro_binds=SYSTEM_BINDS,
        never_bind=frozenset({"/home", "/var", "/run"}),
        tmpfs_size_bytes=64 * MIB,
        seccomp_enabled=seccomp,
        seccomp_required=required,
        protected_dirs=tuple(str(p) for p in protected),
    )


def _manager(tmp_path, posture, ceilings=None):
    return _sub().PluginSubprocessManager(
        log_dir=tmp_path / "logs",
        data_root=tmp_path / "data",
        posture_provider=lambda: posture,
        limit_ceilings=ceilings,
        startup_timeout=30.0,
    )


def _record_popen(monkeypatch):
    """Stand in for a worker's launch: record what would run, start nothing.

    Any other process (a library probing the system, say) starts as usual.
    """
    calls = []
    real_popen = subprocess.Popen

    def fake_popen(argv, **kwargs):
        if not any(str(token).endswith("plugin_worker.py") for token in argv):
            return real_popen(argv, **kwargs)
        calls.append(dict(kwargs, argv=list(argv)))
        raise OSError("launch recorded by the contract, nothing started")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def _bind_pairs(argv):
    return [
        (token, argv[i + 1], argv[i + 2])
        for i, token in enumerate(argv)
        if token in ("--bind", "--ro-bind") and i + 2 < len(argv)
    ]


class _FakeSandbox:
    """The sandbox manager's posture surface, and nothing else."""

    def __init__(self, *, in_use, strict, backend="auto", info="bubblewrap 0.0 test"):
        self.bwrap_in_use = in_use
        self.strict_mode = strict
        self.config = types.SimpleNamespace(
            enabled=True,
            strict_mode=strict,
            isolation_backend=backend,
            bwrap_ro_binds=list(SYSTEM_BINDS),
            bwrap_never_bind=[],
            tmpfs_size_bytes=32 * MIB,
            seccomp_enabled=True,
            seccomp_required=True,
        )
        self._info = info

    def get_isolation_status(self):
        return {
            "isolation_level": "bwrap" if self.bwrap_in_use else "blocked",
            "bwrap_available": self.bwrap_in_use,
            "bwrap_info": self._info,
            "strict_mode": self.strict_mode,
            "backend": "bwrap" if self.bwrap_in_use else "tempdir",
        }


def _sandbox_module(manager, config=None):
    return types.SimpleNamespace(
        sandbox_manager=manager,
        _config=config,
        _HARDCODED_NEVER_BIND=frozenset({"/home", "/var"}),
    )


class _Unreadable:
    def __getattr__(self, name):
        raise RuntimeError(f"sandbox state unreadable: {name}")


def _proc_hard_limits(pid):
    rows = {}
    for line in Path(f"/proc/{pid}/limits").read_text().splitlines()[1:]:
        match = re.match(r"^(Max .+?)\s{2,}(\S+)\s+(\S+)", line)
        if match:
            hard = match.group(3)
            rows[match.group(1)] = float("inf") if hard == "unlimited" else int(hard)
    return rows


def _pids_with_env(marker):
    """Processes whose environment carries ``marker`` (readable ones only)."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if marker.encode() in environ.split(b"\0"):
            found.append(int(entry.name))
    return found


def _bwrap_can_run():
    exe = shutil.which("bwrap")
    if exe is None:
        return False, "bubblewrap is not installed"
    probe = [exe]
    for path in ("/usr", "/bin", "/lib", "/lib64"):
        if os.path.exists(path):
            probe += ["--ro-bind", path, path]
    probe += ["--dev", "/dev", "--proc", "/proc", "--unshare-pid",
              "--unshare-net", "--die-with-parent", "true"]
    try:
        rc = subprocess.run(probe, capture_output=True, timeout=20).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"bubblewrap cannot run here: {exc}"
    return rc == 0, f"bubblewrap cannot create namespaces here (rc={rc})"


def test_pb1_the_posture_follows_the_sandbox_and_fails_closed():
    iso = _iso()
    in_use = iso.resolve_posture(_sandbox_module(_FakeSandbox(in_use=True, strict=True)))
    assert in_use.mode == iso.MODE_BWRAP
    blocked = iso.resolve_posture(_sandbox_module(_FakeSandbox(in_use=False, strict=True)))
    assert blocked.mode == iso.MODE_BLOCKED
    assert "bubblewrap" in blocked.reason and "strict_mode" in blocked.reason
    lax = iso.resolve_posture(_sandbox_module(_FakeSandbox(in_use=False, strict=False)))
    assert lax.mode == iso.MODE_DIRECT and lax.strict is False
    tempdir = iso.resolve_posture(_sandbox_module(
        _FakeSandbox(in_use=False, strict=True, backend="tempdir")))
    assert tempdir.mode == iso.MODE_BLOCKED and "isolation_backend" in tempdir.reason
    off_strict = types.SimpleNamespace(enabled=False, strict_mode=True)
    assert iso.resolve_posture(_sandbox_module(None, off_strict)).mode == iso.MODE_BLOCKED
    off_lax = types.SimpleNamespace(enabled=False, strict_mode=False)
    assert iso.resolve_posture(_sandbox_module(None, off_lax)).mode == iso.MODE_DIRECT
    assert iso.resolve_posture(_sandbox_module(None, None)).mode == iso.MODE_BLOCKED
    assert iso.resolve_posture(_Unreadable()).mode == iso.MODE_BLOCKED


def test_pb2_a_blocked_posture_starts_nothing_and_names_cause_and_remedy(
        tmp_path, monkeypatch):
    iso = _iso()
    calls = _record_popen(monkeypatch)
    posture = iso.resolve_posture(_sandbox_module(_FakeSandbox(in_use=False, strict=True)))
    plugin = _make_plugin(tmp_path, "pbblocked", permissions=("filesystem_plugin_dir",))
    manager = _manager(tmp_path, posture)
    with pytest.raises(iso.PluginIsolationError) as refused:
        manager.start_plugin(
            "pbblocked", plugin, "entry_point.py", permissions=["filesystem_plugin_dir"])
    assert "bubblewrap" in str(refused.value) and "strict_mode" in str(refused.value)
    loader = _loader().PluginLoader(subprocess_manager=manager)
    with pytest.raises(_loader().PluginLoadError) as failed:
        loader.load_plugin(plugin)
    assert "strict_mode" in str(failed.value)
    assert calls == []
    assert not (tmp_path / "data" / "pbblocked").exists()


def test_pb3_the_plugin_reads_its_own_tree_and_writes_only_its_own_folders(tmp_path):
    iso = _iso()
    plugin = _make_plugin(tmp_path, "pbwalls")
    private = plugin / "private"
    private.mkdir()
    data = tmp_path / "data" / "pbwalls"
    data.mkdir(parents=True)
    data = data.resolve()
    worker = _worker_path().resolve()
    argv = iso.build_bwrap_argv(
        _posture(iso.MODE_BWRAP, protected=[private]),
        interpreter=sys.executable,
        worker_script=str(worker),
        plugin_dir=str(plugin),
        data_dir=str(data),
        allow_network=False,
    )
    pairs = _bind_pairs(argv)
    writable = [(src, dst) for flag, src, dst in pairs if flag == "--bind"]
    assert writable == [(str(data), str(data))]
    read_only = {src for flag, src, dst in pairs if flag == "--ro-bind"}
    assert str(plugin) in read_only and str(worker) in read_only
    assert str(worker.parent) not in read_only
    tmpfs_at = [i for i, token in enumerate(argv) if token == "--tmpfs"]
    first_tmp = argv.index("/tmp", tmpfs_at[0]) if tmpfs_at else -1
    assert argv[tmpfs_at[0] + 1] == "/tmp"
    for i, token in enumerate(argv):
        if token in ("--bind", "--ro-bind") and argv[i + 2].startswith("/tmp/"):
            assert i > first_tmp
    mask = argv.index(str(private))
    assert argv[mask - 1] == "--tmpfs" and mask > argv.index(str(plugin))
    for flag in ("--unshare-pid", "--unshare-ipc", "--unshare-uts",
                 "--unshare-cgroup", "--new-session", "--die-with-parent"):
        assert flag in argv
    assert argv[argv.index("--chdir") + 1] == str(plugin)
    assert argv[-2:] == [sys.executable, str(worker)]
    for too_wide in ("/", str(Path.home())):
        with pytest.raises(iso.PluginIsolationError):
            iso.build_bwrap_argv(
                _posture(iso.MODE_BWRAP),
                interpreter=sys.executable,
                worker_script=str(worker),
                plugin_dir=too_wide,
                data_dir=None,
                allow_network=False,
            )


def test_pb4_only_a_manifest_that_declares_network_keeps_it(tmp_path, monkeypatch):
    iso = _iso()
    calls = _record_popen(monkeypatch)
    manager = _manager(tmp_path,_posture(iso.MODE_BWRAP))
    loader = _loader().PluginLoader(subprocess_manager=manager)
    for name, permissions, unshared in (
        ("pbnonet", (), True),
        ("pbnet", ("network_outbound",), False),
    ):
        plugin = _make_plugin(tmp_path, name, permissions=permissions)
        with pytest.raises(_loader().PluginLoadError):
            loader.load_plugin(plugin)
        assert ("--unshare-net" in calls[-1]["argv"]) is unshared
    assert len(calls) >= 2


def test_pb5_no_server_secret_reaches_the_worker_and_no_key_is_on_a_command_line(
        tmp_path, monkeypatch):
    iso = _iso()
    monkeypatch.setenv("OPTI_ENCRYPTION_KEY", "server-only-value")
    calls = _record_popen(monkeypatch)
    plugin = _make_plugin(tmp_path, "pbenv")
    manager = _manager(tmp_path,_posture(iso.MODE_BWRAP))
    with pytest.raises(_sub().PluginSubprocessError):
        manager.start_plugin("pbenv", plugin, "entry_point.py")
    call = calls[-1]
    env = call["env"]
    assert "OPTI_ENCRYPTION_KEY" not in env
    assert all("server-only-value" not in value for value in env.values())
    key = env["OO_HMAC_KEY"]
    assert len(key) >= 64
    assert all(key not in token for token in call["argv"])
    assert env["HOME"] == "/tmp" and env["TMPDIR"] == "/tmp"


def test_pb6_the_worker_runs_under_the_lower_of_manifest_and_ceiling(tmp_path):
    iso = _iso()
    ceilings = iso.ProcessLimits(
        cpu_time_seconds=20,
        memory_bytes=512 * MIB,
        max_file_descriptors=48,
        max_processes=4096,
        max_file_size_bytes=8 * MIB,
    )
    asked = types.SimpleNamespace(
        cpu_time_seconds=999, memory_bytes=384 * MIB, max_file_descriptors=999,
    )
    effective = iso.effective_limits(asked, ceilings)
    assert (effective.cpu_time_seconds, effective.memory_bytes,
            effective.max_file_descriptors) == (20, 384 * MIB, 48)
    configured = tmp_path / "plugins.yaml"
    configured.write_text(
        "subprocess:\n"
        "  limit_ceilings:\n"
        "    cpu_time_seconds: 21\n"
        "    memory_bytes: 1\n"
        "    max_file_descriptors: 47\n"
        "    max_processes: 100000\n"
        "    max_file_size_bytes: 9437184\n",
        encoding="utf-8",
    )
    read = iso.load_ceilings(configured)
    assert (read.cpu_time_seconds, read.max_file_descriptors, read.max_file_size_bytes) == (
        21, 47, 9 * MIB)
    assert read.memory_bytes >= 64 * MIB and read.max_processes <= 8192
    plugin = _make_plugin(tmp_path, "pblimits", resource_limits={
        "cpu_time_seconds": 999, "memory_bytes": 2048 * MIB, "max_file_descriptors": 999,
    })
    manager = _manager(tmp_path,_posture(iso.MODE_DIRECT, strict=False), ceilings)
    loader = _loader().PluginLoader(subprocess_manager=manager)
    try:
        loader.load_plugin(plugin)
        hard = _proc_hard_limits(manager.get_process("pblimits").process.pid)
        assert hard["Max cpu time"] <= 20
        assert hard["Max address space"] <= 512 * MIB
        assert hard["Max open files"] <= 48
        assert hard["Max processes"] <= 4096
        assert hard["Max file size"] <= 8 * MIB
        assert hard["Max core file size"] <= 0
    finally:
        manager.stop_all()


def test_pb7_a_worker_without_its_limits_refuses_to_serve(tmp_path):
    plugin = _make_plugin(tmp_path, "pbnolimits")
    key = secrets.token_bytes(32)
    host_end, worker_end = socket.socketpair()
    # Limits a worker could set on itself: one that did so, instead of
    # finding them in place, would go on to serve.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "OO_PLUGIN_NAME": "pbnolimits",
        "OO_PLUGIN_DIR": str(plugin),
        "OO_PLUGIN_ENTRY": "entry_point.py",
        "OO_SOCKET_FD": str(worker_end.fileno()),
        "OO_HMAC_KEY": key.hex(),
        "OO_RLIMIT_CPU": "30",
        "OO_RLIMIT_MEM": str(1024 * MIB),
        "OO_RLIMIT_NOFILE": "64",
        "OO_RLIMIT_NPROC": "4096",
        "OO_RLIMIT_FSIZE": str(256 * MIB),
    }
    process = subprocess.Popen(
        [sys.executable, str(_worker_path())], env=env,
        pass_fds=(worker_end.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    worker_end.close()
    sub = _sub()
    try:
        sub.send_message(host_end, key, sub.make_rpc_request("ping", {}))
        answer = sub.recv_message(host_end, key, timeout=30)
    except (sub.PluginSubprocessError, OSError):
        answer = None
    finally:
        host_end.close()
    try:
        _out, err = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        _out, err = process.communicate()
    assert answer is None
    assert process.returncode != 0
    assert b"RLIMIT_CPU is not in place" in err


def test_pb8_the_seccomp_filter_goes_with_bwrap_and_its_absence_refuses_when_required(
        tmp_path, monkeypatch):
    iso = _iso()
    calls = _record_popen(monkeypatch)
    plugin = _make_plugin(tmp_path, "pbseccomp")
    if platform.machine() in ("x86_64", "amd64"):
        manager = _manager(tmp_path,_posture(iso.MODE_BWRAP, seccomp=True))
        with pytest.raises(_sub().PluginSubprocessError):
            manager.start_plugin("pbseccomp", plugin, "entry_point.py")
        argv = calls[-1]["argv"]
        fd = int(argv[argv.index("--seccomp") + 1])
        assert fd in calls[-1]["pass_fds"]
    seccomp = importlib.import_module("opti_oignon.sandbox_seccomp")

    def no_table(*_args, **_kwargs):
        raise seccomp.SeccompUnavailable("no syscall table in this contract")

    monkeypatch.setattr(seccomp, "build_filter_program", no_table)
    before = len(calls)
    required = _manager(tmp_path,_posture(iso.MODE_BWRAP, seccomp=True))
    with pytest.raises(iso.PluginIsolationError):
        required.start_plugin("pbseccomp", plugin, "entry_point.py")
    assert len(calls) == before
    optional = _manager(
        tmp_path, _posture(iso.MODE_BWRAP, seccomp=True, required=False))
    with pytest.raises(_sub().PluginSubprocessError):
        optional.start_plugin("pbseccomp", plugin, "entry_point.py")
    assert len(calls) == before + 1 and "--seccomp" not in calls[-1]["argv"]


def test_pb9_a_plugin_that_may_write_gets_a_private_data_folder(
        tmp_path, monkeypatch):
    iso = _iso()
    calls = _record_popen(monkeypatch)
    manager = _manager(tmp_path,_posture(iso.MODE_BWRAP))
    for name, permission in (("pbdata", "filesystem_plugin_dir"), ("pbwrite", "filesystem_write")):
        plugin = _make_plugin(tmp_path, name)
        with pytest.raises(_sub().PluginSubprocessError):
            manager.start_plugin(name, plugin, "entry_point.py", permissions=[permission])
        granted = Path(calls[-1]["env"]["OO_PLUGIN_DATA_DIR"])
        assert granted == tmp_path / "data" / name
        assert stat.S_IMODE(granted.stat().st_mode) == 0o700
        assert ("--bind", str(granted), str(granted)) in _bind_pairs(calls[-1]["argv"])
    plugin = _make_plugin(tmp_path, "pbreadonly")
    with pytest.raises(_sub().PluginSubprocessError):
        manager.start_plugin("pbreadonly", plugin, "entry_point.py", permissions=["network_outbound"])
    assert "OO_PLUGIN_DATA_DIR" not in calls[-1]["env"]
    assert not (tmp_path / "data" / "pbreadonly").exists()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "data" / "pblink").symlink_to(elsewhere)
    plugin = _make_plugin(tmp_path, "pblink")
    before = len(calls)
    with pytest.raises(iso.PluginIsolationError):
        manager.start_plugin("pblink", plugin, "entry_point.py", permissions=["filesystem_plugin_dir"])
    assert len(calls) == before
    worker, restore = _window("opti_oignon.plugin_worker", "plugin_worker.py")
    try:
        module = types.ModuleType("pb_probe_module")
        module.hook_pre_inference = lambda context: {"seen": context.metadata.get("data_dir")}
        forged = {"metadata": {"data_dir": "/forged-by-the-wire"}}
        granted_seen = worker.execute_hook(module, "pre_inference", forged, data_dir="/granted")
        bare_seen = worker.execute_hook(module, "pre_inference", forged)
    finally:
        restore()
    assert granted_seen == {"seen": "/granted"}
    assert bare_seen == {"seen": None}


STORE_PLUGINS = (
    ("scratchpad", "_get_db", "scratchpad.db"),
    ("task-extractor", "_get_db", "tasks.db"),
    ("github-connector", "_get_store", "github_auth.db"),
)


def _store(name, getter, metadata, connect):
    """Open a shipped plugin's store as its worker would: the host package out of reach."""
    target = f"opti_oignon._pb_store_{name.replace('-', '_')}"
    module, restore = _window(target, "plugins", name, "entry_point.py")
    try:
        module._safe_connect = connect
        store = getattr(module, getter)(types.SimpleNamespace(metadata=metadata, config={}))
        store.close()
    finally:
        restore()


def test_pb10_the_shipped_stores_live_in_the_data_folder_never_in_the_shared_temp(
        tmp_path, monkeypatch):
    shared = tmp_path / "shared-temp"
    shared.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(shared))
    for name, getter, filename in STORE_PLUGINS:
        opened = []

        def connect(path, **kwargs):
            opened.append(str(path))
            return sqlite3.connect(str(path), **kwargs)

        granted = tmp_path / "data" / name
        granted.mkdir(parents=True)
        _store(name, getter, {"data_dir": str(granted)}, connect)
        assert (granted / filename).exists()
        _store(name, getter, {}, connect)
        assert len(opened) >= 2
        assert all(not path.startswith(str(shared)) for path in opened)
    assert sorted(shared.iterdir()) == []


def test_pb11_health_reports_the_plugin_isolation(monkeypatch):
    iso = _iso()
    posture = iso.resolve_posture(_sandbox_module(_FakeSandbox(in_use=False, strict=True)))
    monkeypatch.setattr(iso, "resolve_posture", lambda sandbox_module=None: posture)
    app = importlib.import_module("opti_oignon.api.app")
    section = app._get_health_security_info()
    assert section["plugin_isolation"] == {
        "mode": iso.MODE_BLOCKED, "strict": True, "reason": posture.reason,
    }


HOST_REACHING = '''
try:
    from opti_oignon.db_utils import safe_connect  # noqa: F401
    REFUSAL = ""
except ImportError as exc:
    REFUSAL = str(exc)


def hook_pre_inference(context):
    return {"refusal": REFUSAL}
'''


def test_pb13_a_plugin_reaching_for_the_host_package_starts_fast_on_its_fallback(tmp_path):
    iso = _iso()
    plugin = _make_plugin(tmp_path, "pbhost", entry=HOST_REACHING)
    manager = _manager(tmp_path, _posture(iso.MODE_DIRECT, strict=False))
    try:
        started = time.perf_counter()
        # The adverse layout: the host package is on the worker's path, so
        # only the worker's own guard can be what refuses it.
        manager.start_plugin(
            "pbhost", plugin, "entry_point.py", env_extra={"PYTHONPATH": str(REPO)})
        elapsed = time.perf_counter() - started
        out = manager.call_hook("pbhost", "pre_inference", {"data": {}})
    finally:
        manager.stop_all()
    assert "isolation boundary" in out["refusal"]
    assert elapsed <= 5.0


def test_pb12_under_the_real_bubblewrap_the_walls_hold(tmp_path):
    can_run, why = _bwrap_can_run()
    if not can_run:
        pytest.skip(why)
    iso = _iso()
    name = f"pbprobe{secrets.token_hex(3)}"
    plugin = _make_plugin(tmp_path, name, entry=PROBE, permissions=("filesystem_plugin_dir",))
    masked = plugin / "private"
    masked.mkdir()
    (masked / "secret.txt").write_text("not for the plugin", encoding="utf-8")
    outside = tmp_path / "outside" / "secret.txt"
    outside.parent.mkdir()
    outside.write_text("not for the plugin", encoding="utf-8")
    posture = _posture(
        iso.MODE_BWRAP,
        seccomp=platform.machine() in ("x86_64", "amd64"),
        protected=[masked],
    )
    manager = _manager(tmp_path,posture)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(0.5)
    try:
        process = manager.start_plugin(
            name, plugin, "entry_point.py", permissions=["filesystem_plugin_dir"])
        assert len(_pids_with_env(f"OO_PLUGIN_NAME={name}")) >= 1
        out = manager.call_hook(name, "pre_inference", {"data": {
            "outside": str(outside),
            "masked": str(masked / "secret.txt"),
            "code_dir": str(plugin),
            "port": listener.getsockname()[1],
        }})
        assert out["outside"] != "done"
        assert out["masked"] != "done"
        assert out["code"] != "done" and not (plugin / "written.txt").exists()
        assert out["data"] == "done"
        assert (tmp_path / "data" / name / "kept.txt").read_text() == "kept"
        assert out["net"] != "done"
        with pytest.raises(OSError):
            listener.accept()
        assert out["pid"] <= 3
        assert manager.stop_plugin(name) is True
        assert process.process.poll() is not None
        assert _pids_with_env(f"OO_PLUGIN_NAME={name}") == []
    finally:
        listener.close()
        manager.stop_all()
