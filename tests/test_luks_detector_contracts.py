#!/usr/bin/env python3
"""Contracts for the LUKS / full-disk encryption detector.

The detector is advisory only: it reports whether the root filesystem
sits on LUKS/dm-crypt, carries actionable tips when it does not, and
never blocks anything by itself. Detection walks lsblk JSON, then
/proc/mounts, then dmsetup. These contracts pin the observable
behaviour the startup checklist consumes:

  * Contract 1 -- lsblk crypt walk: a nested lsblk tree containing a
    crypt device (or a crypto_LUKS fstype) reports encrypted via the
    lsblk method, lists the crypt devices, and carries no tips.
  * Contract 2 -- plain-disk lsblk tree reports unencrypted with
    actionable tips (encryption guidance is present).
  * Contract 3 -- fallback order: when lsblk yields nothing, the
    /proc/mounts result is used and attributed to that method.
  * Contract 4 -- nothing determinable: all three methods silent leaves
    the result unchecked with method "none", still carrying tips.
  * Contract 5 -- serialization: the result serializes to JSON with the
    fields the checklist and the API read.

Local-only (the public distribution ships no tests). Runs under pytest
or the __main__ runner. The detector module is loaded in isolation
under a stub package; subprocess output is stubbed per clause.
"""

import importlib.util
import json
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


_ENCRYPTED_TREE = {
    "blockdevices": [
        {
            "name": "nvme0n1", "type": "disk", "fstype": None,
            "mountpoint": None,
            "children": [
                {
                    "name": "nvme0n1p2", "type": "part",
                    "fstype": "crypto_LUKS", "mountpoint": None,
                    "children": [
                        {
                            "name": "cryptroot", "type": "crypt",
                            "fstype": "ext4", "mountpoint": "/",
                        },
                    ],
                },
            ],
        },
    ],
}

_PLAIN_TREE = {
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
    ],
}


# ---------------------------------------------------------------------------
# Contract 1 -- lsblk walk recognizes crypt devices in a nested tree
# ---------------------------------------------------------------------------
def test_c1_lsblk_walk_detects_crypt_devices():
    mod, restore = _load_detector_module()
    try:
        _lsblk_stub(mod, _ENCRYPTED_TREE)
        result = mod.check_luks_encryption()
        assert result.checked and result.method == "lsblk"
        assert result.encrypted, "a crypt device in the tree must report encrypted"
        assert "cryptroot" in result.encrypted_devices, (
            f"the crypt device must be listed, got {result.encrypted_devices}"
        )
        assert not result.tips, "an encrypted system needs no tips"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- plain disk reports unencrypted with actionable tips
# ---------------------------------------------------------------------------
def test_c2_plain_disk_reports_unencrypted_with_tips():
    mod, restore = _load_detector_module()
    try:
        _lsblk_stub(mod, _PLAIN_TREE)
        result = mod.check_luks_encryption()
        assert result.checked and result.method == "lsblk"
        assert not result.encrypted
        assert result.tips, "an unencrypted system must carry tips"
        joined = " ".join(result.tips).lower()
        assert "luks" in joined or "cryptsetup" in joined, (
            "the tips must contain concrete encryption guidance"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- fallback order: /proc/mounts is used when lsblk is silent
# ---------------------------------------------------------------------------
def test_c3_proc_mounts_fallback_is_attributed():
    mod, restore = _load_detector_module()
    try:
        mod._check_lsblk = lambda: None
        mod._check_proc_mounts = lambda: {
            "encrypted": True, "devices": ["dm-0"],
        }
        result = mod.check_luks_encryption()
        assert result.method == "proc_mounts", (
            f"the fallback method must be attributed, got {result.method!r}"
        )
        assert result.encrypted and result.encrypted_devices == ["dm-0"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- all methods silent leaves the result unchecked, with tips
# ---------------------------------------------------------------------------
def test_c4_all_methods_silent_is_unchecked_with_tips():
    mod, restore = _load_detector_module()
    try:
        mod._check_lsblk = lambda: None
        mod._check_proc_mounts = lambda: None
        mod._check_dmsetup = lambda: None
        result = mod.check_luks_encryption()
        assert not result.checked and result.method == "none"
        assert not result.encrypted, (
            "an undeterminable status must not claim encryption"
        )
        assert result.tips, "the undeterminable case still carries tips"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 5 -- serialization exposes the fields the checklist reads
# ---------------------------------------------------------------------------
def test_c5_result_serializes_with_expected_fields():
    mod, restore = _load_detector_module()
    try:
        _lsblk_stub(mod, _ENCRYPTED_TREE)
        result = mod.check_luks_encryption()
        payload = result.to_dict()
        json.dumps(payload)
        for field in (
            "checked", "encrypted", "method", "detail",
            "encrypted_devices", "tips",
        ):
            assert field in payload, f"to_dict must expose {field!r}"
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
