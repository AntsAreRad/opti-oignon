#!/usr/bin/env python3
"""Contracts for owner-only permissions on saved red team reports.

Red team reports name which attack categories bypass the defenses. On a
shared host, other local users must not be able to read them, and the
default umask must not leave them world-readable. These contracts pin
that ``save_report`` writes the report directory and every report file
owner-only:

  * Contract 1 -- directory is 0o700.
  * Contract 2 -- every written report file is 0o600.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. The module is loaded in isolation under a stub
package; a temporary output directory receives the reports.
"""

import importlib.util
import os
import shutil
import stat
import sys
import tempfile
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load_reports_module():
    keys = ("opti_oignon", "opti_oignon.redteam.reports")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.redteam.reports", _OO / "redteam" / "reports.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.redteam.reports"] = mod
    spec.loader.exec_module(mod)

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


class _FakeScore:
    """Minimal stand-in for a CampaignScore the report generators read."""

    total = 0
    total_bypasses = 0
    total_flags = 0
    total_blocks = 0
    overall_bypass_rate = 0.0
    overall_detection_rate = 0.0
    overall_block_rate = 0.0
    by_category: dict = {}
    by_target: dict = {}
    by_strategy: dict = {}

    def heatmap_data(self):
        return []


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


# ---------------------------------------------------------------------------
# Contract 1 -- the report directory is owner-only (0o700)
# ---------------------------------------------------------------------------
def test_c1_report_directory_is_owner_only():
    mod, restore = _load_reports_module()
    tmp = tempfile.mkdtemp(prefix="oo_rt_")
    out = os.path.join(tmp, "redteam_results")
    try:
        mod.save_report(_FakeScore(), output_dir=out)
        assert _mode(out) == 0o700, (
            f"report directory must be 0o700, got {oct(_mode(out))}"
        )
    finally:
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Contract 2 -- every report file is owner-only (0o600)
# ---------------------------------------------------------------------------
def test_c2_report_files_are_owner_only():
    mod, restore = _load_reports_module()
    tmp = tempfile.mkdtemp(prefix="oo_rt_")
    out = os.path.join(tmp, "redteam_results")
    try:
        saved = mod.save_report(
            _FakeScore(), output_dir=out,
            formats=["json", "text", "markdown"],
        )
        assert saved, "save_report must report the files it wrote"
        for fmt, path in saved.items():
            assert _mode(path) == 0o600, (
                f"{fmt} report must be 0o600, got {oct(_mode(path))} "
                f"for {path}"
            )
    finally:
        restore()
        shutil.rmtree(tmp, ignore_errors=True)


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
