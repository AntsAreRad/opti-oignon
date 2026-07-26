#!/usr/bin/env python3
"""Seccomp denylist BPF builder for the bubblewrap sandbox.

Pure, dependency-free assembly of a classic cBPF seccomp program suitable for
``bwrap --seccomp <fd>``. The program is a moderate DENYLIST: it kills a set of
escape-class and legacy syscalls and allows everything else.

Honesty about assurance and scope:

- The bytes this module produces are deterministic and fully testable in any
  environment -- they are just packed ``struct sock_filter`` records. Whether
  the kernel actually loads the program and kills the listed syscalls assures
  only on the host, under a real bwrap launch. See the Bloc 0 host-assurance
  list in SHAKEDOWN_S198_HANDOFF.md.
- A denylist reduces kernel attack surface; it is NOT the sandbox boundary. The
  boundary is the namespace isolation that bwrap sets up. Kerckhoffs: the
  filter is open by design; its strength is the mechanism, not its secrecy.
- x86_64 only for now. The syscall-number table is versioned here and the
  builder REFUSES (raises ``SeccompUnavailable``) on any other architecture
  rather than emitting a filter against the wrong numbers. aarch64 is a
  documented future row.
- bwrap installs the filter last, immediately before exec'ing the child, after
  it has already created the namespaces and set up the mounts. Denying
  mount / unshare / setns in the child is therefore safe and is exactly the
  intent: block the nested-namespace / kernel-LPE escape path from inside the
  sandbox, not bwrap's own setup.

The program shape is an inline skip-over chain: every conditional that matches
falls through to an adjacent ``RET``, so there are no long jumps and the
encoding is trivially verifiable instruction by instruction.
"""

import platform
import struct

checkpoint_before_apply = True


# BPF instruction classes / sizes / modes / ops (linux/bpf_common.h).
_BPF_LD = 0x00
_BPF_W = 0x00
_BPF_ABS = 0x20
_BPF_JMP = 0x05
_BPF_JEQ = 0x10
_BPF_JGE = 0x30
_BPF_K = 0x00
_BPF_RET = 0x06

_LD_ABS_W = _BPF_LD | _BPF_W | _BPF_ABS   # 0x20: load word from a fixed offset
_JEQ_K = _BPF_JMP | _BPF_JEQ | _BPF_K     # 0x15: jump if accumulator == k
_JGE_K = _BPF_JMP | _BPF_JGE | _BPF_K     # 0x35: jump if accumulator >= k
_RET_K = _BPF_RET | _BPF_K                # 0x06: return constant k

# seccomp_data field offsets (linux/seccomp.h, struct seccomp_data):
#   int   nr;                    offset 0
#   __u32 arch;                  offset 4
#   __u64 instruction_pointer;   offset 8
#   __u64 args[6];               offset 16...
_OFF_NR = 0
_OFF_ARCH = 4

# Audit arch token for x86_64 (linux/audit.h, AUDIT_ARCH_X86_64).
AUDIT_ARCH_X86_64 = 0xC000003E

# The x32 ABI sets this bit in the syscall number on x86_64. A denylist that
# only matches the native numbers is bypassable through x32, so kill the whole
# x32 range outright.
X32_SYSCALL_BIT = 0x40000000

# seccomp return actions (linux/seccomp.h).
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ALLOW = 0x7FFF0000

# struct sock_filter is { __u16 code; __u8 jt; __u8 jf; __u32 k; }, packed
# little-endian on x86_64. bwrap reads the raw concatenation of these records
# from the fd and builds the sock_fprog itself.
_SOCK_FILTER = struct.Struct("<HBBI")
SOCK_FILTER_SIZE = 8

# Versioned x86_64 syscall-number table. Verified against
# /usr/include/x86_64-linux-gnu/asm/unistd_64.h (linux-libc-dev). Bump
# SYSCALL_TABLE_VERSION whenever the audited set changes.
SYSCALL_TABLE_VERSION = "x86_64/1"

_X86_64_NR = {
    # Required core (spec section 3).
    "keyctl": 250,
    "add_key": 248,
    "request_key": 249,
    "ptrace": 101,
    "userfaultfd": 323,
    "bpf": 321,
    "mount": 165,
    "unshare": 272,
    # Obscure / legacy escape, info-leak, and rarely-needed surface.
    "setns": 308,
    "kexec_load": 246,
    "kexec_file_load": 320,
    "init_module": 175,
    "finit_module": 313,
    "delete_module": 176,
    "iopl": 172,
    "ioperm": 173,
    "swapon": 167,
    "swapoff": 168,
    "reboot": 169,
    "pivot_root": 155,
    "mount_setattr": 442,
    "open_by_handle_at": 304,
    "name_to_handle_at": 303,
    "perf_event_open": 298,
    "process_vm_readv": 310,
    "process_vm_writev": 311,
    "quotactl": 179,
    "acct": 163,
    "personality": 135,
    "_sysctl": 156,
    "nfsservctl": 180,
}

# The denied set, in a stable documented order (table order).
DENIED_SYSCALLS = tuple(_X86_64_NR.keys())


class SeccompUnavailable(RuntimeError):
    """Raised when a seccomp filter cannot be built for this architecture."""


def syscall_number(name):
    """Return the x86_64 syscall number for a denied-set name."""
    return _X86_64_NR[name]


def denied_syscall_numbers():
    """Return the denied syscall numbers in stable order (x86_64)."""
    return tuple(_X86_64_NR[name] for name in DENIED_SYSCALLS)


def build_instructions():
    """Build the cBPF program as a list of ``(code, jt, jf, k)`` tuples."""
    prog = []
    # Validate architecture. On anything but x86_64 we cannot reason about the
    # syscall numbers, so kill (fail-secure). The Python-level guard in
    # build_filter_program refuses earlier; this is belt-and-braces in-filter.
    prog.append((_LD_ABS_W, 0, 0, _OFF_ARCH))
    prog.append((_JEQ_K, 1, 0, AUDIT_ARCH_X86_64))      # match -> skip the kill
    prog.append((_RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS))
    # Load the syscall number.
    prog.append((_LD_ABS_W, 0, 0, _OFF_NR))
    # Kill the entire x32 range (numbers with the x32 bit set).
    prog.append((_JGE_K, 0, 1, X32_SYSCALL_BIT))        # >= bit -> fall to kill
    prog.append((_RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS))
    # Deny each listed syscall.
    for nr in denied_syscall_numbers():
        prog.append((_JEQ_K, 0, 1, nr))                 # match -> fall to kill
        prog.append((_RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS))
    # Default allow.
    prog.append((_RET_K, 0, 0, SECCOMP_RET_ALLOW))
    return prog


def program_instruction_count():
    """Number of sock_filter instructions in the built program."""
    return len(build_instructions())


def build_filter_program(arch=None):
    """Return the packed BPF program bytes for ``bwrap --seccomp <fd>``.

    Raises ``SeccompUnavailable`` on any non-x86_64 architecture rather than
    emitting a filter built against the wrong syscall numbers.
    """
    machine = arch if arch is not None else platform.machine()
    if machine not in ("x86_64", "amd64"):
        raise SeccompUnavailable(
            f"no seccomp syscall table for architecture {machine!r} "
            f"(table version {SYSCALL_TABLE_VERSION}, x86_64 only)"
        )
    return b"".join(
        _SOCK_FILTER.pack(code, jt, jf, k)
        for (code, jt, jf, k) in build_instructions()
    )
