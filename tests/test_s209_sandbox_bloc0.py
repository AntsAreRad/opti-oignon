#!/usr/bin/env python3
"""S209 (Sandbox Workspace, Bloc 0): containment hardening.

Container-deliverable assertions only. We pin the bwrap argv, the seccomp BPF
bytes (against known-good encodings plus a small in-test cBPF interpreter), the
resource-cap clamps and the rlimit preexec wiring, the fail-secure refusal, the
config surface, the spec/cartography registrations, and the boundary comment by
source. The running containment -- caps actually binding, the kernel loading
the filter and killing syscalls, the namespaces absent from /proc, --size
honoured -- assures only on the host (BLOC0_HOST_ASSURANCE in
SHAKEDOWN_S198_HANDOFF.md). Nothing here executes bwrap.
"""

import os
import struct
import sys
import types

# Guarded stub: in CI ollama is installed and this is a no-op; locally it lets
# the isolated module load resolve the opti_oignon import chain.
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

import importlib.util

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


def _load(name, relpath):
    path = os.path.join(_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sm = _load(
    "sandbox_manager_s209", os.path.join("opti_oignon", "sandbox_manager.py")
)
_ss = _load(
    "sandbox_seccomp_s209", os.path.join("opti_oignon", "sandbox_seccomp.py")
)

_build_bwrap_command = _sm._build_bwrap_command
SandboxConfig = _sm.SandboxConfig
CommandResult = _sm.CommandResult


def _argv(command="echo hi", workspace="/tmp/ws", seccomp_fd=None, **cfg_kw):
    cfg = SandboxConfig(bwrap_ro_binds=["/usr"], **cfg_kw)
    return _build_bwrap_command(command, workspace, cfg, seccomp_fd=seccomp_fd)


@pytest.fixture(autouse=True)
def _seccomp_module_resolvable(monkeypatch):
    # Under the full sweep, an earlier test module may replace
    # sys.modules["opti_oignon"] with a non-package stub, which breaks the
    # source's lazy `from opti_oignon import sandbox_seccomp`. Bind our
    # isolated module copy so the import resolves regardless of sweep order.
    # This is a test-isolation guard only; the source is correct (in
    # production opti_oignon is always the real package).
    oo = sys.modules.get("opti_oignon")
    if oo is None:
        oo = types.ModuleType("opti_oignon")
        monkeypatch.setitem(sys.modules, "opti_oignon", oo)
    monkeypatch.setattr(oo, "sandbox_seccomp", _ss, raising=False)
    monkeypatch.setitem(sys.modules, "opti_oignon.sandbox_seccomp", _ss)
    yield


# ---------------------------------------------------------------------------
# argv: tmpfs size cap
# ---------------------------------------------------------------------------

class TestTmpfsSize:
    def test_size_precedes_tmpfs(self):
        cmd = _argv()
        assert "--size" in cmd and "--tmpfs" in cmd
        assert cmd.index("--size") < cmd.index("--tmpfs")

    def test_tmpfs_still_targets_tmp(self):
        # The S183 / sandbox_manager assertion cmd[index("--tmpfs")+1] == "/tmp"
        # must still hold after inserting --size.
        cmd = _argv()
        ti = cmd.index("--tmpfs")
        assert cmd[ti + 1] == "/tmp"

    def test_size_value_is_configured_bytes(self):
        cmd = _argv(tmpfs_size_bytes=128 * 1024 ** 2)
        si = cmd.index("--size")
        assert cmd[si + 1] == str(128 * 1024 ** 2)

    def test_size_value_is_clamped_in_argv(self):
        # An out-of-range tmpfs size is clamped (never disabled) and the argv
        # carries the clamped value.
        cmd = _argv(tmpfs_size_bytes=10 ** 15)
        si = cmd.index("--size")
        assert cmd[si + 1] == str(_sm._CAP_TMPFS_BYTES_MAX)


# ---------------------------------------------------------------------------
# argv: namespace cloistering
# ---------------------------------------------------------------------------

class TestNamespaceCloistering:
    def test_ipc_uts_cgroup_added(self):
        cmd = _argv()
        for flag in ("--unshare-ipc", "--unshare-uts", "--unshare-cgroup"):
            assert flag in cmd

    def test_net_and_pid_still_explicit(self):
        # Incremental shape, not --unshare-all: net and pid stay listed.
        cmd = _argv()
        assert "--unshare-net" in cmd
        assert "--unshare-pid" in cmd

    def test_not_unshare_all(self):
        # The user namespace / uid-gid mapping must be left untouched.
        cmd = _argv()
        assert "--unshare-all" not in cmd
        assert "--unshare-user" not in cmd

    def test_session_and_die_with_parent_preserved(self):
        cmd = _argv()
        assert "--new-session" in cmd
        assert "--die-with-parent" in cmd


# ---------------------------------------------------------------------------
# argv: seccomp flag
# ---------------------------------------------------------------------------

class TestSeccompFlag:
    def test_no_seccomp_flag_without_fd(self):
        cmd = _argv(seccomp_fd=None)
        assert "--seccomp" not in cmd

    def test_seccomp_flag_with_fd(self):
        cmd = _argv(seccomp_fd=9)
        assert "--seccomp" in cmd
        assert cmd[cmd.index("--seccomp") + 1] == "9"

    def test_command_still_terminates_with_bash_c(self):
        cmd = _argv(command="ls -la", seccomp_fd=5)
        assert cmd[-3:] == ["bash", "-c", "ls -la"]


# ---------------------------------------------------------------------------
# seccomp BPF: known-good encodings
# ---------------------------------------------------------------------------

class TestSeccompEncoding:
    def test_sock_filter_is_eight_bytes(self):
        assert _ss.SOCK_FILTER_SIZE == 8
        assert struct.calcsize("<HBBI") == 8

    def test_blob_length_matches_instruction_count(self):
        blob = _ss.build_filter_program("x86_64")
        assert len(blob) == 8 * _ss.program_instruction_count()

    def test_prologue_instructions(self):
        instrs = _ss.build_instructions()
        # [0] load arch word, [1] jeq x86_64 (skip kill on match),
        # [2] kill, [3] load nr, [4] jge x32 bit, [5] kill.
        assert instrs[0] == (_ss._LD_ABS_W, 0, 0, _ss._OFF_ARCH)
        assert instrs[1] == (_ss._JEQ_K, 1, 0, _ss.AUDIT_ARCH_X86_64)
        assert instrs[2] == (_ss._RET_K, 0, 0, _ss.SECCOMP_RET_KILL_PROCESS)
        assert instrs[3] == (_ss._LD_ABS_W, 0, 0, _ss._OFF_NR)
        assert instrs[4] == (_ss._JGE_K, 0, 1, _ss.X32_SYSCALL_BIT)
        assert instrs[5] == (_ss._RET_K, 0, 0, _ss.SECCOMP_RET_KILL_PROCESS)

    def test_tail_is_default_allow(self):
        instrs = _ss.build_instructions()
        assert instrs[-1] == (_ss._RET_K, 0, 0, _ss.SECCOMP_RET_ALLOW)

    def test_packed_prologue_bytes(self):
        blob = _ss.build_filter_program("x86_64")
        expected_first = struct.pack("<HBBI", _ss._LD_ABS_W, 0, 0, _ss._OFF_ARCH)
        assert blob[:8] == expected_first
        expected_arch = struct.pack(
            "<HBBI", _ss._JEQ_K, 1, 0, _ss.AUDIT_ARCH_X86_64
        )
        assert blob[8:16] == expected_arch

    def test_each_denied_syscall_has_jeq_then_kill(self):
        instrs = _ss.build_instructions()
        for nr in _ss.denied_syscall_numbers():
            jeq = (_ss._JEQ_K, 0, 1, nr)
            assert jeq in instrs
            i = instrs.index(jeq)
            assert instrs[i + 1] == (
                _ss._RET_K, 0, 0, _ss.SECCOMP_RET_KILL_PROCESS
            )

    def test_known_syscall_numbers(self):
        # Verified against /usr/include/x86_64-linux-gnu/asm/unistd_64.h.
        assert _ss.syscall_number("keyctl") == 250
        assert _ss.syscall_number("add_key") == 248
        assert _ss.syscall_number("request_key") == 249
        assert _ss.syscall_number("ptrace") == 101
        assert _ss.syscall_number("userfaultfd") == 323
        assert _ss.syscall_number("bpf") == 321
        assert _ss.syscall_number("mount") == 165
        assert _ss.syscall_number("unshare") == 272

    def test_required_core_set_is_denied(self):
        denied = set(_ss.DENIED_SYSCALLS)
        for name in (
            "keyctl", "add_key", "request_key", "ptrace",
            "userfaultfd", "bpf", "mount", "unshare",
        ):
            assert name in denied

    def test_arch_refusal_off_x86_64(self):
        with pytest.raises(_ss.SeccompUnavailable):
            _ss.build_filter_program("aarch64")

    def test_table_is_versioned(self):
        assert _ss.SYSCALL_TABLE_VERSION.startswith("x86_64")


# ---------------------------------------------------------------------------
# seccomp BPF: behaviour via a small cBPF interpreter
# ---------------------------------------------------------------------------

def _run_bpf(nr, arch=None, args=(0, 0, 0, 0, 0, 0)):
    """Interpret the built program over a synthetic seccomp_data record.

    Returns the SECCOMP_RET_* action the program would yield.
    """
    if arch is None:
        arch = _ss.AUDIT_ARCH_X86_64
    prog = _ss.build_instructions()
    # struct seccomp_data: int nr; __u32 arch; __u64 ip; __u64 args[6].
    data = struct.pack("<iI", nr, arch) + struct.pack("<Q", 0)
    for v in args:
        data += struct.pack("<Q", v)
    acc = 0
    pc = 0
    while True:
        code, jt, jf, k = prog[pc]
        if code == _ss._LD_ABS_W:
            acc = struct.unpack_from("<I", data, k)[0]
            pc += 1
        elif code == _ss._JEQ_K:
            pc += 1 + (jt if acc == k else jf)
        elif code == _ss._JGE_K:
            pc += 1 + (jt if acc >= k else jf)
        elif code == _ss._RET_K:
            return k
        else:  # pragma: no cover - guards against an unexpected opcode
            raise AssertionError(f"unexpected opcode {code:#x}")


class TestSeccompBehaviour:
    def test_denied_syscall_is_killed(self):
        for name in ("keyctl", "ptrace", "unshare", "mount", "bpf"):
            assert _run_bpf(_ss.syscall_number(name)) == (
                _ss.SECCOMP_RET_KILL_PROCESS
            )

    def test_ordinary_syscalls_allowed(self):
        # read=0, write=1, openat=257, clone=56 (deliberately not inspected),
        # exit_group=231.
        for nr in (0, 1, 257, 56, 231):
            assert _run_bpf(nr) == _ss.SECCOMP_RET_ALLOW

    def test_x32_range_is_killed(self):
        assert _run_bpf(_ss.X32_SYSCALL_BIT) == _ss.SECCOMP_RET_KILL_PROCESS
        assert _run_bpf(_ss.X32_SYSCALL_BIT | 1) == (
            _ss.SECCOMP_RET_KILL_PROCESS
        )

    def test_wrong_arch_is_killed(self):
        aarch64 = 0xC00000B7
        assert _run_bpf(0, arch=aarch64) == _ss.SECCOMP_RET_KILL_PROCESS


# ---------------------------------------------------------------------------
# Resource caps: clamp surface
# ---------------------------------------------------------------------------

class TestConfigCapSurface:
    def test_defaults(self):
        cfg = SandboxConfig()
        assert cfg.limits_enabled is True
        assert cfg.resource_backend == "rlimit"
        assert cfg.limit_memory_bytes == _sm._CAP_MEMORY_BYTES_DEFAULT
        assert cfg.limit_nproc == _sm._CAP_NPROC_DEFAULT
        assert cfg.limit_fsize_bytes == _sm._CAP_FSIZE_BYTES_DEFAULT
        assert cfg.limit_cpu_seconds == _sm._CAP_CPU_SECONDS_DEFAULT
        assert cfg.tmpfs_size_bytes == _sm._CAP_TMPFS_BYTES_DEFAULT
        assert cfg.seccomp_enabled is True
        assert cfg.seccomp_required is True

    def test_low_values_clamp_up(self):
        cfg = SandboxConfig(
            limit_memory_bytes=1,
            limit_nproc=1,
            limit_fsize_bytes=1,
            limit_cpu_seconds=0,
            tmpfs_size_bytes=1,
        )
        assert cfg.limit_memory_bytes == _sm._CAP_MEMORY_BYTES_MIN
        assert cfg.limit_nproc == _sm._CAP_NPROC_MIN
        assert cfg.limit_fsize_bytes == _sm._CAP_FSIZE_BYTES_MIN
        assert cfg.limit_cpu_seconds == _sm._CAP_CPU_SECONDS_MIN
        assert cfg.tmpfs_size_bytes == _sm._CAP_TMPFS_BYTES_MIN

    def test_high_values_clamp_down(self):
        cfg = SandboxConfig(
            limit_memory_bytes=10 ** 18,
            limit_nproc=10 ** 9,
            limit_fsize_bytes=10 ** 18,
            limit_cpu_seconds=10 ** 9,
            tmpfs_size_bytes=10 ** 18,
        )
        assert cfg.limit_memory_bytes == _sm._CAP_MEMORY_BYTES_MAX
        assert cfg.limit_nproc == _sm._CAP_NPROC_MAX
        assert cfg.limit_fsize_bytes == _sm._CAP_FSIZE_BYTES_MAX
        assert cfg.limit_cpu_seconds == _sm._CAP_CPU_SECONDS_MAX
        assert cfg.tmpfs_size_bytes == _sm._CAP_TMPFS_BYTES_MAX

    def test_non_integer_clamps_to_floor(self):
        cfg = SandboxConfig(limit_memory_bytes="not-a-number")
        assert cfg.limit_memory_bytes == _sm._CAP_MEMORY_BYTES_MIN

    def test_unknown_backend_falls_back_to_rlimit(self):
        cfg = SandboxConfig(resource_backend="bogus")
        assert cfg.resource_backend == "rlimit"

    def test_clamp_never_disables(self):
        # A degenerate config still yields positive, in-range caps.
        cfg = SandboxConfig(limit_nproc=-50, tmpfs_size_bytes=-1)
        assert cfg.limit_nproc >= _sm._CAP_NPROC_MIN
        assert cfg.tmpfs_size_bytes >= _sm._CAP_TMPFS_BYTES_MIN


# ---------------------------------------------------------------------------
# Resource caps: the rlimit preexec hook content
# ---------------------------------------------------------------------------

class TestRlimitHook:
    def test_hook_sets_all_limits_with_clamped_values(self, monkeypatch):
        cfg = SandboxConfig(
            limit_memory_bytes=512 * 1024 ** 2,
            limit_nproc=128,
            limit_fsize_bytes=64 * 1024 ** 2,
            limit_cpu_seconds=60,
        )
        recorded = {}

        def _rec(which, limits):
            recorded[which] = limits

        monkeypatch.setattr(_sm.resource, "setrlimit", _rec)
        hook = _sm._make_rlimit_preexec(cfg)
        hook()

        r = _sm.resource
        assert recorded[r.RLIMIT_AS] == (cfg.limit_memory_bytes,) * 2
        assert recorded[r.RLIMIT_NPROC] == (cfg.limit_nproc,) * 2
        assert recorded[r.RLIMIT_FSIZE] == (cfg.limit_fsize_bytes,) * 2
        assert recorded[r.RLIMIT_CPU] == (cfg.limit_cpu_seconds,) * 2
        assert recorded[r.RLIMIT_CORE] == (0, 0)

    def test_hook_uses_clamped_not_raw(self, monkeypatch):
        cfg = SandboxConfig(limit_memory_bytes=1)  # below floor -> clamps
        recorded = {}
        monkeypatch.setattr(
            _sm.resource, "setrlimit",
            lambda which, limits: recorded.__setitem__(which, limits),
        )
        _sm._make_rlimit_preexec(cfg)()
        assert recorded[_sm.resource.RLIMIT_AS] == (
            _sm._CAP_MEMORY_BYTES_MIN,
        ) * 2


# ---------------------------------------------------------------------------
# Resource caps: cgroup prefix + detection
# ---------------------------------------------------------------------------

class TestCgroupBackend:
    def test_systemd_prefix_shape(self):
        cfg = SandboxConfig(
            limit_memory_bytes=512 * 1024 ** 2, limit_nproc=200
        )
        prefix = _sm._systemd_run_prefix(cfg)
        assert prefix[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
        assert prefix[-1] == "--"
        joined = " ".join(prefix)
        assert f"MemoryMax={cfg.limit_memory_bytes}" in joined
        assert f"TasksMax={cfg.limit_nproc}" in joined

    def test_detect_returns_bool(self):
        assert isinstance(_sm._detect_systemd_run(), bool)


# ---------------------------------------------------------------------------
# _run_bwrap wiring (no bwrap executed: subprocess.run is faked)
# ---------------------------------------------------------------------------

@pytest.fixture()
def manager(tmp_path):
    cfg = SandboxConfig(
        workspace_base=str(tmp_path / "sbx"),
        audit_db_path="audit.db",
    )
    return _sm.SandboxManager(config=cfg)


class _FakeProc:
    def __init__(self):
        self.returncode = 0
        self.stdout = b"ok"
        self.stderr = b""


class TestRunBwrapWiring:
    def _fake_run(self, captured):
        def _run(argv, **kw):
            captured["argv"] = argv
            captured["kw"] = kw
            return _FakeProc()
        return _run

    def test_rlimit_backend_wires_preexec_and_passes_seccomp_fd(
        self, manager, monkeypatch
    ):
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "run", self._fake_run(captured))
        manager._config.limits_enabled = True
        manager._config.resource_backend = "rlimit"
        manager._config.seccomp_enabled = True
        res = manager._run_bwrap("echo hi", str(manager._config.workspace_base), 5)
        assert res.blocked is False
        assert callable(captured["kw"]["preexec_fn"])
        pass_fds = captured["kw"]["pass_fds"]
        assert len(pass_fds) == 1 and isinstance(pass_fds[0], int)
        argv = captured["argv"]
        assert "--seccomp" in argv
        assert argv[argv.index("--seccomp") + 1] == str(pass_fds[0])

    def test_limits_disabled_means_no_preexec(self, manager, monkeypatch):
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "run", self._fake_run(captured))
        manager._config.limits_enabled = False
        manager._run_bwrap("echo hi", str(manager._config.workspace_base), 5)
        assert captured["kw"]["preexec_fn"] is None

    def test_seccomp_disabled_means_no_flag(self, manager, monkeypatch):
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "run", self._fake_run(captured))
        manager._config.seccomp_enabled = False
        manager._run_bwrap("echo hi", str(manager._config.workspace_base), 5)
        assert "--seccomp" not in captured["argv"]
        assert captured["kw"]["pass_fds"] == ()

    def test_cgroup_backend_prefixes_and_drops_preexec(
        self, manager, monkeypatch
    ):
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "run", self._fake_run(captured))
        monkeypatch.setattr(_sm, "_detect_systemd_run", lambda: True)
        manager._config.limits_enabled = True
        manager._config.resource_backend = "cgroup"
        manager._run_bwrap("echo hi", str(manager._config.workspace_base), 5)
        assert captured["argv"][0] == "systemd-run"
        assert captured["kw"]["preexec_fn"] is None

    def test_cgroup_unavailable_falls_back_to_rlimit(
        self, manager, monkeypatch
    ):
        captured = {}
        monkeypatch.setattr(_sm.subprocess, "run", self._fake_run(captured))
        monkeypatch.setattr(_sm, "_detect_systemd_run", lambda: False)
        manager._config.limits_enabled = True
        manager._config.resource_backend = "cgroup"
        manager._run_bwrap("echo hi", str(manager._config.workspace_base), 5)
        # Never disabled: it falls back to the rlimit preexec.
        assert captured["argv"][0] != "systemd-run"
        assert callable(captured["kw"]["preexec_fn"])


# ---------------------------------------------------------------------------
# Fail-secure: refuse rather than launch unfiltered
# ---------------------------------------------------------------------------

class TestSeccompFailSecure:
    def _break_builder(self, monkeypatch):
        def _raise(*a, **k):
            raise _ss.SeccompUnavailable("forced for test")

        monkeypatch.setattr(_ss, "build_filter_program", _raise)

    def test_required_true_refuses_launch(self, manager, monkeypatch):
        self._break_builder(monkeypatch)
        # subprocess.run must NOT be reached on the refusal path.
        def _boom(*a, **k):
            raise AssertionError("subprocess.run reached on fail-secure path")
        monkeypatch.setattr(_sm.subprocess, "run", _boom)
        manager._config.seccomp_enabled = True
        manager._config.seccomp_required = True
        res = manager._run_bwrap(
            "echo hi", str(manager._config.workspace_base), 5
        )
        assert res.blocked is True
        assert res.return_code == -1
        assert "fail-secure" in res.block_reason

    def test_required_false_launches_unfiltered_with_warning(
        self, manager, monkeypatch, caplog
    ):
        self._break_builder(monkeypatch)
        captured = {}

        def _run(argv, **kw):
            captured["argv"] = argv
            captured["kw"] = kw
            return _FakeProc()

        monkeypatch.setattr(_sm.subprocess, "run", _run)
        manager._config.seccomp_enabled = True
        manager._config.seccomp_required = False
        with caplog.at_level("WARNING"):
            res = manager._run_bwrap(
                "echo hi", str(manager._config.workspace_base), 5
            )
        assert res.blocked is False
        assert "--seccomp" not in captured["argv"]
        assert captured["kw"]["pass_fds"] == ()
        assert any("seccomp" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Spec / cartography registrations and the boundary comment by source
# ---------------------------------------------------------------------------

def _read(relpath):
    with open(os.path.join(_ROOT, relpath), encoding="utf-8") as fh:
        return fh.read()


class TestRegistrationsAndComment:
    def test_new_module_carries_checkpoint_sentinel(self):
        assert _ss.checkpoint_before_apply is True

    def test_spec_registers_seccomp_module(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "sandbox_seccomp.py" in spec

    def test_spec_section3_status_landed(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "Status (S209)" in spec
        assert "BLOC0_HOST_ASSURANCE" in spec

    def test_spec_section15_has_s209_row(self):
        spec = _read("SANDBOX_WORKSPACE_SPEC.md")
        assert "test_s209_sandbox_bloc0.py" in spec

    def test_shakedown_has_host_assurance_list(self):
        sh = _read("SHAKEDOWN_S198_HANDOFF.md")
        assert "BLOC0_HOST_ASSURANCE" in sh

    def test_boundary_comment_in_source(self):
        src = _read(os.path.join("opti_oignon", "sandbox_manager.py"))
        assert "S209 boundary note" in src
        assert "bypassable" in src
        assert "namespace isolation" in src
