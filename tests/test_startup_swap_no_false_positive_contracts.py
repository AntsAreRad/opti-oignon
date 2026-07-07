#!/usr/bin/env python3
"""Contracts against a false-positive encrypted-swap report at startup.

The startup checklist warns when swap may be unencrypted, because RAM
contents (keys, conversations) can leak to disk. A swap device under
/dev/mapper or /dev/dm-* is NOT proof of encryption on its own: plain LVM
swap volumes share that namespace. These contracts pin that the advisory
only credits a swap device as encrypted when its dm UUID confirms a
CRYPT- target:

  * Contract 1 -- LVM swap is not encrypted: a /dev/mapper swap whose dm
    UUID does not confirm CRYPT- reports the swap check failed (a
    warning), not passed.
  * Contract 2 -- confirmed crypt swap passes: a /dev/mapper swap whose dm
    UUID confirms CRYPT- reports the swap check passed.
  * Contract 3 -- a plain partition swap still warns.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. The module is loaded in isolation under a stub
package; /proc/swaps and the dm-UUID probe are stubbed per clause.
"""

import importlib.util
import io
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_SWAP_HEADER = (
    "Filename                                Type            "
    "Size            Used            Priority\n"
)


def _load_checks_module():
    keys = ("opti_oignon", "opti_oignon.startup_checks")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.startup_checks", _OO / "startup_checks.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.startup_checks"] = mod
    spec.loader.exec_module(mod)
    pkg.startup_checks = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


def _serve_proc_swaps(content):
    """Return an open() replacement that serves canned /proc/swaps."""
    import builtins

    real_open = builtins.open

    def _fake_open(path, *args, **kwargs):
        if str(path) == "/proc/swaps":
            return io.StringIO(content)
        return real_open(path, *args, **kwargs)

    return _fake_open


def _run_swap_check(mod, swaps_content, dm_is_crypt):
    import builtins

    real_open = builtins.open
    mod._dm_device_is_crypt = lambda device: dm_is_crypt
    builtins.open = _serve_proc_swaps(swaps_content)
    try:
        return mod._check_encrypted_swap()
    finally:
        builtins.open = real_open


# ---------------------------------------------------------------------------
# Contract 1 -- LVM swap under /dev/mapper is not credited as encrypted
# ---------------------------------------------------------------------------
def test_c1_lvm_swap_is_not_encrypted():
    mod, restore = _load_checks_module()
    try:
        content = _SWAP_HEADER + (
            "/dev/mapper/vg-swap                     partition       "
            "8388604         0               -2\n"
        )
        item = _run_swap_check(mod, content, dm_is_crypt=False)
        assert not item.passed, (
            "a /dev/mapper swap without a confirmed CRYPT- UUID must warn, "
            "not pass"
        )
        assert item.severity == "warning"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- confirmed crypt swap passes
# ---------------------------------------------------------------------------
def test_c2_confirmed_crypt_swap_passes():
    mod, restore = _load_checks_module()
    try:
        content = _SWAP_HEADER + (
            "/dev/mapper/cryptswap                   partition       "
            "8388604         0               -2\n"
        )
        item = _run_swap_check(mod, content, dm_is_crypt=True)
        assert item.passed, (
            "a /dev/mapper swap with a confirmed CRYPT- UUID must pass"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- a plain partition swap still warns
# ---------------------------------------------------------------------------
def test_c3_plain_partition_swap_warns():
    mod, restore = _load_checks_module()
    try:
        content = _SWAP_HEADER + (
            "/dev/sda2                               partition       "
            "8388604         0               -2\n"
        )
        item = _run_swap_check(mod, content, dm_is_crypt=False)
        assert not item.passed, "a plain partition swap must warn"
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
