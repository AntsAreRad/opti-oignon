#!/usr/bin/env python3
"""Contracts against false-positive encryption reports in the LUKS detector.

The detector is advisory, but a wrong "encrypted" verdict is worse than
no verdict: it credits a security score the disk has not earned and
withholds the guidance an unencrypted system needs. Two ways the detector
could over-report encryption are pinned here:

  * Contract 1 -- unrelated crypt device: a crypt device that is NOT an
    ancestor of the root mountpoint (a backup drive, encrypted swap
    mounted elsewhere) must NOT make the root filesystem report
    encrypted. The lsblk verdict tracks whether "/" sits on or descends
    from a crypt device, not merely whether any crypt device exists.
  * Contract 2 -- LUKS-on-LVM is still encrypted: when "/" is an LVM
    volume layered on top of a crypt device (the common full-disk
    encryption layout), the root must report encrypted. This guards the
    ancestry fix against introducing a false negative.
  * Contract 3 -- dm/mapper root is not proof: in the /proc/mounts
    fallback, a root device under /dev/mapper or /dev/dm-* reports
    encrypted only when the dm UUID confirms a CRYPT- target. A plain LVM
    volume shares that namespace, so an unconfirmed dm device must report
    unencrypted (fail secure), not encrypted.
  * Contract 4 -- confirmed dm crypt is encrypted: a dm/mapper root whose
    UUID confirms CRYPT- reports encrypted. This guards the fallback fix
    against dropping a real detection.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. The detector module is loaded in isolation under a
stub package; subprocess and /proc/mounts are stubbed per clause.
"""

import importlib.util
import io
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load_detector_module():
    keys = ("opti_oignon", "opti_oignon.luks_detector")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.luks_detector", _OO / "luks_detector.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.luks_detector"] = mod
    spec.loader.exec_module(mod)
    pkg.luks_detector = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


def _lsblk_stub(mod, payload):
    """Replace the module's subprocess.run with a canned lsblk answer."""
    import json
    import subprocess as real_subprocess

    class _Completed:
        returncode = 0

    completed = _Completed()
    completed.stdout = json.dumps(payload)

    fake = types.SimpleNamespace(
        run=lambda *a, **k: completed,
        TimeoutExpired=real_subprocess.TimeoutExpired,
    )
    mod.subprocess = fake


def _serve_proc_mounts(content):
    """Return an open() replacement that serves canned /proc/mounts."""
    import builtins

    real_open = builtins.open

    def _fake_open(path, *args, **kwargs):
        if str(path) == "/proc/mounts":
            return io.StringIO(content)
        return real_open(path, *args, **kwargs)

    return _fake_open


# An unrelated crypt device: plain root on sda1, a separate encrypted
# backup drive mounted at /mnt/backup. No crypt device is an ancestor of
# "/".
_UNRELATED_CRYPT_TREE = {
    "blockdevices": [
        {
            "name": "sda", "type": "disk", "fstype": None, "mountpoint": None,
            "children": [
                {
                    "name": "sda1", "type": "part", "fstype": "ext4",
                    "mountpoint": "/",
                },
            ],
        },
        {
            "name": "sdb", "type": "disk", "fstype": None, "mountpoint": None,
            "children": [
                {
                    "name": "sdb1", "type": "part", "fstype": "crypto_LUKS",
                    "mountpoint": None,
                    "children": [
                        {
                            "name": "cryptbackup", "type": "crypt",
                            "fstype": "ext4", "mountpoint": "/mnt/backup",
                        },
                    ],
                },
            ],
        },
    ],
}

# LUKS-on-LVM: "/" is an LVM volume layered on top of a crypt device.
_LUKS_ON_LVM_TREE = {
    "blockdevices": [
        {
            "name": "nvme0n1", "type": "disk", "fstype": None,
            "mountpoint": None,
            "children": [
                {
                    "name": "nvme0n1p3", "type": "part",
                    "fstype": "crypto_LUKS", "mountpoint": None,
                    "children": [
                        {
                            "name": "luksroot", "type": "crypt",
                            "fstype": "LVM2_member", "mountpoint": None,
                            "children": [
                                {
                                    "name": "vg-root", "type": "lvm",
                                    "fstype": "ext4", "mountpoint": "/",
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# Contract 1 -- an unrelated crypt device does not encrypt the root
# ---------------------------------------------------------------------------
def test_c1_unrelated_crypt_device_does_not_encrypt_root():
    mod, restore = _load_detector_module()
    try:
        _lsblk_stub(mod, _UNRELATED_CRYPT_TREE)
        result = mod.check_luks_encryption()
        assert result.checked and result.method == "lsblk"
        assert not result.encrypted, (
            "a crypt device that is not an ancestor of '/' must not make "
            "the root report encrypted"
        )
        assert result.tips, "an unencrypted root must still carry tips"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- LUKS-on-LVM root is encrypted (no false negative)
# ---------------------------------------------------------------------------
def test_c2_luks_on_lvm_root_reports_encrypted():
    mod, restore = _load_detector_module()
    try:
        _lsblk_stub(mod, _LUKS_ON_LVM_TREE)
        result = mod.check_luks_encryption()
        assert result.checked and result.method == "lsblk"
        assert result.encrypted, (
            "a root LVM volume layered on a crypt device must report "
            "encrypted"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- dm/mapper root without a CRYPT- UUID is not encrypted
# ---------------------------------------------------------------------------
def test_c3_proc_mounts_dm_without_crypt_uuid_is_unencrypted():
    import builtins

    mod, restore = _load_detector_module()
    real_open = builtins.open
    try:
        mod._check_lsblk = lambda: None
        mod._check_dm_uuid = lambda name: False  # not a CRYPT- target
        builtins.open = _serve_proc_mounts(
            "/dev/mapper/vg-root / ext4 rw,relatime 0 0\n"
        )
        try:
            result = mod.check_luks_encryption()
        finally:
            builtins.open = real_open
        assert result.method == "proc_mounts"
        assert not result.encrypted, (
            "a /dev/mapper root without a confirmed CRYPT- UUID must not "
            "report encrypted (plain LVM shares that namespace)"
        )
        assert result.tips, "the unconfirmed case must carry tips"
    finally:
        builtins.open = real_open
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- dm/mapper root with a CRYPT- UUID is encrypted
# ---------------------------------------------------------------------------
def test_c4_proc_mounts_dm_with_crypt_uuid_is_encrypted():
    import builtins

    mod, restore = _load_detector_module()
    real_open = builtins.open
    try:
        mod._check_lsblk = lambda: None
        mod._check_dm_uuid = lambda name: True  # confirmed CRYPT- target
        builtins.open = _serve_proc_mounts(
            "/dev/mapper/cryptroot / ext4 rw,relatime 0 0\n"
        )
        try:
            result = mod.check_luks_encryption()
        finally:
            builtins.open = real_open
        assert result.method == "proc_mounts"
        assert result.encrypted, (
            "a /dev/mapper root with a confirmed CRYPT- UUID must report "
            "encrypted"
        )
    finally:
        builtins.open = real_open
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
