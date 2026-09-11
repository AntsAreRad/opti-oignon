#!/usr/bin/env python3
"""Contracts for where the governor's VRAM capacity comes from.

The governor decides admission against a total VRAM figure. That figure was
only ever a number typed into YAML, refined downward by a ceiling learned from
failures -- and the shipped configuration leaves it null, so on an unconfigured
host the capacity is unknown and the VRAM half of every admission fails open.

A real probe exists in the tree: the host metrics collector reads the total and
used VRAM from nvidia-smi and reports -1.0, not 0.0, for anything it could not
read. The governor could not see it. These contracts wire that sensor in and
pin the two things that matter about the wiring:

  * VC1 -- with nothing configured, a probe that reports a real total gives the
    governor its capacity, and the snapshot names the probe as the source
    rather than presenting a measurement as configuration.
  * VC2 -- a probe that cannot read leaves the capacity unknown. The sentinel
    is negative on purpose; turning it into a capacity of zero would declare a
    card with no memory and refuse everything, and turning it into a capacity
    at all would be inventing one.
  * VC3 -- a configured capacity wins, and the probe is not consulted at all:
    an operator who wrote a number down is not overruled by a sensor.
  * VC4 -- a probe that raises is survived, and leaves the capacity unknown.

What is NOT pinned here, and is owed to the machine: any real VRAM figure.
Every probe in this file is injected. That nvidia-smi reports this host's card
correctly is not provable in a container and is not claimed.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; the database layer is seeded, and every store this suite
touches lives under pytest's tmp_path.
"""

import sqlite3
import sys
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _db_stub():
    """Seed for the database helper the governor's store reaches through."""
    module = types.ModuleType("opti_oignon.db_utils")
    module.safe_connect = lambda p, **kw: sqlite3.connect(
        str(p), check_same_thread=kw.get("check_same_thread", False),
    )
    return module


def _open():
    """Open the shared window on the governor alone."""
    loaded, restore = isolate(
        targets={
            "opti_oignon.resource_governor": source("resource_governor.py"),
        },
        seeded={"opti_oignon.db_utils": _db_stub()},
    )
    return loaded["opti_oignon.resource_governor"], restore


def _config(tmp_path, total_vram_gb):
    """Write a governor config carrying just the capacity under test."""
    # The real file carries these at the top level, not under a section key.
    body = "enabled: true\n"
    if total_vram_gb is None:
        body += "total_vram_gb: null\n"
    else:
        body += f"total_vram_gb: {total_vram_gb}\n"
    path = tmp_path / "resource_governor.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _governor(rg, tmp_path, total_vram_gb, probe):
    """Build a governor with an injected probe and a throwaway store."""
    return rg.ResourceGovernor(
        config_path=_config(tmp_path, total_vram_gb),
        db_path=tmp_path / "governor.db",
        warmup=None,
        registry=None,
        vram_probe=probe,
    )


class _Probe:
    """A VRAM probe that answers from a knob and counts its calls."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


# ---------------------------------------------------------------------------
# VC1 -- an unconfigured capacity is taken from the probe, and named
# ---------------------------------------------------------------------------
def test_vc1_an_unconfigured_capacity_comes_from_the_probe(tmp_path):
    rg, restore = _open()
    try:
        probe = _Probe(24576.0)  # 24 GiB reported in MiB
        gov = _governor(rg, tmp_path, None, probe)
        snapshot = gov._build_snapshot()
        assert probe.calls >= 1, (
            "the probe was actually consulted, so this is not vacuous"
        )
        assert snapshot.capacity_gb is not None, (
            "a probe that can read gives the governor a capacity"
        )
        assert abs(snapshot.capacity_gb - 24.0) < 0.01, (
            "the reading is carried across in gibibytes, not megabytes"
        )
        assert snapshot.capacity_source == "probe", (
            "the snapshot says the capacity was measured, not configured"
        )
        assert snapshot.vram_status == "ok", (
            "with a capacity known, the VRAM half is no longer disabled"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# VC2 -- a probe that cannot read leaves the capacity unknown
# ---------------------------------------------------------------------------
def test_vc2_an_unreadable_probe_leaves_the_capacity_unknown(tmp_path):
    rg, restore = _open()
    try:
        probe = _Probe(-1.0)  # the collector's own sentinel for "cannot say"
        gov = _governor(rg, tmp_path, None, probe)
        snapshot = gov._build_snapshot()
        assert probe.calls >= 1, (
            "the probe was actually consulted, so this is not vacuous"
        )
        assert snapshot.capacity_gb is None, (
            "a sentinel is not a reading: the capacity stays unknown"
        )
        assert snapshot.capacity_gb != 0.0, (
            "the sentinel never becomes a card with no memory"
        )
        assert snapshot.capacity_source == "unknown", (
            "an absent reading is reported as absent"
        )
        assert snapshot.vram_status == "disabled_capacity_unknown", (
            "the VRAM half stays disabled and fail-open, as before the wiring"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# VC3 -- a configured capacity wins and the probe is never consulted
# ---------------------------------------------------------------------------
def test_vc3_a_configured_capacity_is_not_overruled(tmp_path):
    rg, restore = _open()
    try:
        probe = _Probe(8192.0)  # a smaller card the operator did not declare
        gov = _governor(rg, tmp_path, 24.0, probe)
        snapshot = gov._build_snapshot()
        assert snapshot.capacity_gb == 24.0, (
            "the configured figure is the capacity"
        )
        assert snapshot.capacity_source == "config", (
            "the source names configuration, not the probe"
        )
        assert probe.calls == 0, (
            "the probe is not even asked when an operator has written a "
            "number down, so wiring it in costs nothing on a configured host"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# VC4 -- a probe that raises is survived
# ---------------------------------------------------------------------------
def test_vc4_a_raising_probe_leaves_the_capacity_unknown(tmp_path):
    rg, restore = _open()
    try:
        probe = _Probe(OSError("nvidia-smi is not on this host"))
        gov = _governor(rg, tmp_path, None, probe)
        snapshot = gov._build_snapshot()
        assert probe.calls >= 1, (
            "the probe was actually consulted, so this is not vacuous"
        )
        assert snapshot.capacity_gb is None, (
            "a probe that raises leaves the capacity unknown"
        )
        assert snapshot.capacity_source == "unknown", (
            "a failed reading is reported as absent, never as a figure"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    import tempfile

    tests = [
        ("VC1 unconfigured capacity comes from the probe", test_vc1_an_unconfigured_capacity_comes_from_the_probe),
        ("VC2 unreadable probe leaves capacity unknown", test_vc2_an_unreadable_probe_leaves_the_capacity_unknown),
        ("VC3 configured capacity is not overruled", test_vc3_a_configured_capacity_is_not_overruled),
        ("VC4 raising probe leaves capacity unknown", test_vc4_a_raising_probe_leaves_the_capacity_unknown),
    ]
    passed = 0
    for label, fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"PASS  {label}")
                passed += 1
            except Exception:  # noqa: BLE001 -- report and continue
                print(f"FAIL  {label}")
                traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
