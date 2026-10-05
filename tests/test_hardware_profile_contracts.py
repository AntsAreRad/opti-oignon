#!/usr/bin/env python3
"""What the governor knows about the machine it runs on.

The governor priced every admission against one figure: the total memory of
the first card nvidia-smi listed, read again by a new process at every
snapshot rebuild. A second card, an AMD card, an integrated GPU and the memory
other programs hold on a card were invisible to it; the system RAM was read by
three private copies of one parser, two of them reaching for a library no
manifest declares; and the RAM a split leaves to the rest of the machine was a
fixed 4 GiB, whatever the machine and whatever its neighbours went through.

  The profile -- every card, read once, refreshed off the caller's path:
    * HW1 -- the nvidia-smi reader returns every card it prints, and the
      overlay keeps reading the first one as before.
    * HW2 -- one device per nvidia-smi row, with its index, UUID, bus id,
      total and used memory; a field the tool cannot report is unknown.
    * HW3 -- AMD and Intel cards come from DRM sysfs: an AMD card with its own
      memory is discrete, one under the floor is integrated, Intel without a
      memory figure is integrated with an unknown total.
    * HW4 -- a card both sources list is counted once; without nvidia-smi,
      DRM's NVIDIA card has no known total.
    * HW5 -- an override names a card's kind by id, UUID or bus id.
    * HW6 -- auto selects the discrete cards of the first vendor that has any;
      the capacity is their sum.
    * HW7 -- auto honours CUDA_VISIBLE_DEVICES from the server's environment.
    * HW8 -- an explicit list selects exactly the named cards with a total.
    * HW9 -- the cards are read once; importing the profile reads nothing.
    * HW10 -- used memory is cached for its TTL, then refreshed in the
      background while the caller gets the cached figure and its age.
    * HW11 -- a failed refresh keeps the last figure; none read is unknown.
    * HW12 -- meminfo gives the total and available RAM; unknown is zero.
    * HW13 -- pressure stall information is parsed per resource, and its
      absence is named.
    * HW14 -- the profile file ships the decided defaults and holds its ranges.
    * HW15 -- no module of the package imports psutil any more.
    * HW16 -- smart_router and live_metrics read the RAM through the profile.
    * HW17 -- the governor's own meminfo reader agrees with the profile's.

  The governor that sees the machine:
    * HW18 -- an unconfigured capacity is the sum of the selected cards.
    * HW19 -- the safety margin is kept on every selected card.
    * HW20 -- the VRAM other programs hold is deducted, never below zero.
    * HW21 -- a configured capacity reads no card and counts no other program.
    * HW22 -- snapshot rebuilds never wait on nvidia-smi after the first read.
    * HW23 -- a null reserve is sized from the machine; a number stays fixed.
    * HW24 -- memory pressure multiplies the adaptive reserve, with hysteresis.
    * HW25 -- a split is priced against the effective reserve.
    * HW26 -- without pressure files, the snapshot says so and the reserve is
      its base.
    * HW27 -- the shipped file sizes the reserve from the machine, and the new
      blocks hold their ranges.
    * HW28 -- the routes show the machine and write the new keys in range only.
    * HW29 -- speculative decoding sizes its budget from the detected capacity.

  Readings that describe the card as it is:
    * HW30 -- a card reading taken before the engines last changed is not set
      against them: what other programs held is carried until a fresh reading.
    * HW31 -- a reading older than its max age is unknown, and a refresh that
      hangs is not doubled.
    * HW32 -- numeric CUDA_VISIBLE_DEVICES entries count only where the card
      order is known.
    * HW33 -- a bus id in any written form names its card; a name that
      matches no card is warned.
    * HW34 -- a UUID prefix names a card only when it names one.
    * HW35 -- a config write checks its pairs against the file it writes.
    * HW36 -- a first reading slower than the TTL does not start a second one
      at once.

  The pressure other programs suffer, which the background gate reads:
    * HW37 -- the CPU pressure others suffer is the highest of the user's
      other leaf cgroups.
    * HW38 -- neither the process's own cgroup, nor anything under it, nor a
      parent of other cgroups counts.
    * HW39 -- without a user root the profile falls back to the system-wide
      reading, and without that it says it does not know.
    * HW40 -- the profile says a machine has no card only when its PCI bus
      shows none: an integrated controller is none, a controller the DRM tree
      does not list is one all the same, an unreadable bus proves nothing.
    * HW41 -- the governor's snapshot says what the profile knows: no card,
      or a capacity unknown.

Everything here is proven in the container, on fixture trees and scripted
answers loaded through the shared isolation window: no card, no nvidia-smi,
no /proc of the host. What the cards of a real machine report, and what its
neighbours feel, is owed to the machine.
"""

import ast
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_HP = "opti_oignon.hardware_profile"
_RG = "opti_oignon.resource_governor"
_LM = "opti_oignon.live_metrics"
_SR = "opti_oignon.smart_router"
_SD = "opti_oignon.speculative_decoding"
_ROUTES = "opti_oignon.api.routes_governor"
# The two seams the governor resolves at the call: unreachable when the window
# opens, so neither an emergency stop nor a model window enters a decision.
_SEAMS = ("opti_oignon.context_manager", "opti_oignon.emergency_stop")
_MIB = 1024 ** 2
_GIB = 1024 ** 3
_CLOSERS = []


def _project_modules():
    """Every project entry of the module cache, by identity."""
    return {k: v for k, v in sys.modules.items() if k == "opti_oignon" or k.startswith("opti_oignon.")}


@pytest.fixture(autouse=True)
def _left_as_found():
    """No contract may leave a project module or the HTTP transport changed."""
    before = _project_modules()
    urlopen = urllib.request.urlopen
    yield
    urllib.request.urlopen = urlopen
    while _CLOSERS:
        _CLOSERS.pop()()
    assert _project_modules() == before, "every project module is left as the contract found it"


def _db_utils():
    """A db_utils stand-in whose safe_connect is plain sqlite."""
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(
        str(p), check_same_thread=kw.get("check_same_thread", False)
    )
    return db


def _open(*targets, seeded=None, blocked=(), packages=()):
    loaded, restore = isolate(
        targets=dict(targets), blocked=blocked, seeded=seeded or {}, packages=packages
    )
    _CLOSERS.append(restore)
    return loaded


def _hp():
    return _open((_HP, source("hardware_profile.py")))[_HP]


def _rg_hp(*extra, packages=()):
    loaded = _open(
        (_HP, source("hardware_profile.py")),
        (_RG, source("resource_governor.py")),
        *extra,
        seeded={"opti_oignon.db_utils": _db_utils()},
        blocked=_SEAMS,
        packages=packages,
    )
    return loaded


def _tmp():
    return Path(tempfile.mkdtemp(prefix="hw-"))


class _Query:
    """nvidia-smi as the profile asks it: every call recorded, answers scripted.

    Each answer is a list of rows or None (the tool could not answer); the
    last answer repeats once the others are spent.
    """

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, query, timeout):
        self.calls.append((query, timeout))
        if len(self.answers) > 1:
            return self.answers.pop(0)
        return self.answers[0] if self.answers else None


def _row(index, total_mib, used_mib, *, uuid=None, bus=None, name="NVIDIA Test Card"):
    """One nvidia-smi row: index, uuid, name, pci.bus_id, memory.total, memory.used."""
    return [
        str(index),
        uuid or f"GPU-{index:04d}",
        name,
        bus or f"00000000:{index + 1:02X}:00.0",
        str(total_mib),
        str(used_mib),
    ]


def _drm(root, cards):
    """A /sys/class/drm of our own; each card is a mapping of its sysfs files."""
    root.mkdir(parents=True, exist_ok=True)
    for card in cards:
        device = root / f"card{card['n']}" / "device"
        device.mkdir(parents=True)
        (device / "vendor").write_text(card["vendor"] + "\n", encoding="utf-8")
        (device / "device").write_text(card.get("device", "0x0000") + "\n", encoding="utf-8")
        (device / "uevent").write_text(
            f"DRIVER=test\nPCI_SLOT_NAME={card['slot']}\n", encoding="utf-8"
        )
        if card.get("total") is not None:
            (device / "mem_info_vram_total").write_text(f"{card['total']}\n", encoding="utf-8")
        if card.get("used") is not None:
            (device / "mem_info_vram_used").write_text(f"{card['used']}\n", encoding="utf-8")
    return str(root)


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class _Spawn:
    """Background work held until the contract runs it."""

    def __init__(self):
        self.pending = []

    def __call__(self, target):
        self.pending.append(target)

    def run(self):
        while self.pending:
            self.pending.pop(0)()


def _profile(hp, tmp, *, query=None, cards=(), environ=None, clock=None, spawn=None, config=None, pressure=None):
    return hp.HardwareProfile(
        config=config if config is not None else hp.ProfileConfig(),
        drm_root=_drm(tmp / "drm", list(cards)),
        pressure_root=pressure if pressure is not None else str(tmp / "no-pressure"),
        nvidia_query=query if query is not None else _Query(None),
        environ=environ if environ is not None else {},
        clock=clock if clock is not None else _Clock(),
        spawn=spawn if spawn is not None else _Spawn(),
    )


def _psi(directory, **resources):
    """A /proc/pressure of our own: one file per resource named."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("memory", "cpu", "io"):
        path = directory / name
        if name in resources:
            path.write_text(resources[name], encoding="utf-8")
        elif path.exists():
            path.unlink()
    return str(directory)


def _psi_line(kind, avg10, avg60=0.0, avg300=0.0, total=0):
    return f"{kind} avg10={avg10:.2f} avg60={avg60:.2f} avg300={avg300:.2f} total={total}\n"


def _meminfo(tmp, total_mib, available_mib):
    path = tmp / "meminfo"
    path.write_text(
        f"MemTotal:       {int(total_mib * 1024)} kB\n"
        "MemFree:        1024 kB\n"
        f"MemAvailable:   {int(available_mib * 1024)} kB\n",
        encoding="utf-8",
    )
    return str(path)


class _Warmup:
    """The warmup as the governor reads it: a keep_alive and a loaded set."""

    def __init__(self, loaded=None, keep_alive="10m"):
        self.keep_alive = keep_alive
        self._loaded = list(loaded or [])

    def get_loaded_models(self):
        return list(self._loaded)


def _resident(name, size_vram):
    """A loaded model that never reads as idle, so nothing is evictable."""
    return types.SimpleNamespace(
        name=name, size_vram=size_vram, size=None, expires_at=None, context_length=None, digest=None
    )


def _governor(rg, tmp, *, profile, body="enabled: true\ntotal_vram_gb: null\n", loaded=(), total_mib=65536.0, available_mib=60000.0, warmup=None):
    tmp.mkdir(parents=True, exist_ok=True)
    config = tmp / "resource_governor.yaml"
    config.write_text(body, encoding="utf-8")
    return rg.ResourceGovernor(
        config_path=str(config),
        db_path=str(tmp / "governor.db"),
        warmup=warmup if warmup is not None else _Warmup(loaded),
        registry=None,
        clock=lambda: 1000.0,
        meminfo_path=_meminfo(tmp, total_mib, available_mib),
        hardware=profile,
    )


# ---------------------------------------------------------------------------
# HW1-HW11 -- the profile: every card, read once, refreshed off the caller's path
# ---------------------------------------------------------------------------


def test_hw1_the_nvidia_smi_reader_returns_every_card_and_the_overlay_keeps_the_first():
    lm = _open((_LM, source("live_metrics.py")))[_LM]
    calls = []

    def run(argv, **kwargs):
        calls.append((list(argv), kwargs.get("timeout")))
        if argv[1].startswith("--query-gpu=index"):
            text = "0, GPU-a, A, 00000000:01:00.0, 8151, 1200\n1, GPU-b, B, 00000000:02:00.0, 12227, 300\n"
        else:
            text = "35, 4096, 8151, 51\n10, 1024, 12227, 44\n"
        return types.SimpleNamespace(returncode=0, stdout=text, stderr="")

    lm.subprocess = types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired)
    lm._nvidia_smi_available = lambda: True
    query = "index,uuid,name,pci.bus_id,memory.total,memory.used"
    assert lm.nvidia_smi_rows(query, 2.5) == [
        ["0", "GPU-a", "A", "00000000:01:00.0", "8151", "1200"],
        ["1", "GPU-b", "B", "00000000:02:00.0", "12227", "300"],
    ]
    assert calls[-1] == (["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"], 2.5)
    assert lm._query_gpu_metrics() == {
        "gpu_utilization_pct": 35.0,
        "gpu_memory_used_mb": 4096.0,
        "gpu_memory_total_mb": 8151.0,
        "gpu_temperature_c": 51.0,
    }
    # A host without the tool answers nothing, and spawns nothing to learn it.
    lm._nvidia_smi_available = lambda: False
    before = len(calls)
    assert lm.nvidia_smi_rows(query, 2.5) is None
    assert len(calls) == before
    assert not hasattr(lm, "read_total_vram_mb")


def test_hw2_one_device_per_nvidia_smi_row_with_its_identity_and_memory():
    hp = _hp()
    query = _Query([
        ["0", "GPU-1111", "NVIDIA GeForce RTX 5070", "00000000:01:00.0", "8151", "1200"],
        ["1", "GPU-2222", "NVIDIA RTX A2000", "00000000:0A:00.0", "[N/A]", "[N/A]"],
    ])
    profile = _profile(hp, _tmp(), query=query)
    first, second = profile.devices()
    assert (first.id, first.vendor, first.kind, first.index, first.uuid, first.bus_id, first.total_mib, first.name, first.source) == (
        "nvidia:0", "nvidia", "discrete", 0, "GPU-1111", "0000:01:00.0", 8151.0, "NVIDIA GeForce RTX 5070", "nvidia-smi",
    )
    assert (second.id, second.bus_id, second.total_mib) == ("nvidia:1", "0000:0a:00.0", None)
    assert first.to_dict()["total_gb"] == round(8151.0 / 1024.0, 3)
    assert second.to_dict()["total_gb"] is None
    assert query.calls[0][0] == hp.NVIDIA_QUERY == "index,uuid,name,pci.bus_id,memory.total,memory.used"
    assert query.calls[0][1] == hp.ProfileConfig().nvidia_smi_timeout_s


def test_hw3_amd_and_intel_cards_come_from_drm_sysfs_with_their_kind():
    hp = _hp()
    tmp = _tmp()
    cards = [
        {"n": 0, "vendor": "0x1002", "device": "0x744c", "slot": "0000:03:00.0", "total": 16 * _GIB, "used": 2 * _GIB},
        {"n": 1, "vendor": "0x1002", "device": "0x150e", "slot": "0000:c5:00.0", "total": 512 * _MIB, "used": 100 * _MIB},
        {"n": 2, "vendor": "0x8086", "device": "0x7d55", "slot": "0000:00:02.0"},
    ]
    profile = _profile(hp, tmp, cards=cards)
    drm = tmp / "drm"
    (drm / "card0-DP-1").mkdir()
    (drm / "renderD128" / "device").mkdir(parents=True)
    (drm / "renderD128" / "device" / "vendor").write_text("0x1002\n", encoding="utf-8")
    by_id = {d.id: d for d in profile.devices()}
    assert set(by_id) == {"pci:0000:03:00.0", "pci:0000:c5:00.0", "pci:0000:00:02.0"}
    discrete = by_id["pci:0000:03:00.0"]
    assert (discrete.vendor, discrete.kind, discrete.total_mib, discrete.bus_id, discrete.source) == (
        "amd", "discrete", 16384.0, "0000:03:00.0", "sysfs",
    )
    assert (by_id["pci:0000:c5:00.0"].kind, by_id["pci:0000:c5:00.0"].total_mib) == ("integrated", 512.0)
    intel = by_id["pci:0000:00:02.0"]
    assert (intel.vendor, intel.kind, intel.total_mib) == ("intel", "integrated", None)


def test_hw4_a_card_both_sources_list_is_counted_once_and_drm_alone_knows_no_total():
    hp = _hp()
    tmp = _tmp()
    drm_nvidia = [{"n": 0, "vendor": "0x10de", "device": "0x2c05", "slot": "0000:01:00.0"}]
    both = _profile(hp, tmp / "both", query=_Query([_row(0, 8151, 1200)]), cards=drm_nvidia)
    assert [d.id for d in both.devices()] == ["nvidia:0"]
    alone = _profile(hp, tmp / "alone", query=_Query(None), cards=drm_nvidia)
    (only,) = alone.devices()
    assert (only.id, only.vendor, only.total_mib, only.source) == ("pci:0000:01:00.0", "nvidia", None, "sysfs")
    assert alone.placement().capacity_mib is None


def test_hw5_an_override_names_a_card_kind_by_id_uuid_or_bus():
    hp = _hp()
    cards = [{"n": 1, "vendor": "0x1002", "slot": "0000:c5:00.0", "total": 512 * _MIB, "used": 0}]
    query = _Query([
        _row(0, 8151, 0, uuid="GPU-aaaa", bus="00000000:01:00.0"),
        _row(1, 4096, 0, uuid="GPU-bbbb", bus="00000000:02:00.0"),
    ])
    config = hp.ProfileConfig(
        kind_overrides={"pci:0000:c5:00.0": "discrete", "GPU-aaaa": "integrated", "0000:02:00.0": "integrated"}
    )
    profile = _profile(hp, _tmp(), query=query, cards=cards, config=config)
    assert {d.id: d.kind for d in profile.devices()} == {
        "nvidia:0": "integrated",
        "nvidia:1": "integrated",
        "pci:0000:c5:00.0": "discrete",
    }


def test_hw6_auto_selects_the_discrete_cards_of_the_first_vendor_that_has_any_and_sums_them():
    hp = _hp()
    tmp = _tmp()
    amd = [
        {"n": 0, "vendor": "0x1002", "slot": "0000:03:00.0", "total": 16 * _GIB, "used": 1 * _GIB},
        {"n": 1, "vendor": "0x1002", "slot": "0000:c5:00.0", "total": 512 * _MIB, "used": 0},
    ]
    mixed = _profile(hp, tmp / "mixed", query=_Query([_row(0, 8192, 1024), _row(1, 12288, 2048)]), cards=amd).placement()
    assert [d.id for d in mixed.devices] == ["nvidia:0", "nvidia:1"]
    assert (mixed.capacity_mib, mixed.used_mib) == (20480.0, 3072.0)
    amd_only = _profile(hp, tmp / "amd", query=_Query(None), cards=amd).placement()
    assert [d.id for d in amd_only.devices] == ["pci:0000:03:00.0"]
    assert (amd_only.capacity_mib, amd_only.used_mib) == (16384.0, 1024.0)
    integrated_only = _profile(hp, tmp / "igpu", query=_Query(None), cards=[amd[1]]).placement()
    assert integrated_only.devices == ()
    assert (integrated_only.capacity_mib, integrated_only.used_mib, integrated_only.used_age_s) == (None, None, None)


def test_hw7_auto_honours_cuda_visible_devices_from_the_server_environment():
    hp = _hp()
    tmp = _tmp()
    rows = [_row(0, 8192, 0, uuid="GPU-aaaa-1"), _row(1, 12288, 0, uuid="GPU-bbbb-2")]

    def selected(environ, sub):
        profile = _profile(hp, tmp / sub, query=_Query(rows), environ=environ)
        return [d.id for d in profile.placement().devices]

    assert selected({}, "unset") == ["nvidia:0", "nvidia:1"]
    assert selected({"CUDA_VISIBLE_DEVICES": "1"}, "index") == ["nvidia:1"]
    assert selected({"CUDA_VISIBLE_DEVICES": "GPU-aaaa"}, "uuid") == ["nvidia:0"]
    assert selected({"CUDA_VISIBLE_DEVICES": ""}, "empty") == []
    # The list stops at its first entry that names no card, as CUDA reads it.
    assert selected({"CUDA_VISIBLE_DEVICES": "1,bogus,0"}, "stop") == ["nvidia:1"]


def test_hw8_an_explicit_list_selects_exactly_the_named_cards_with_a_known_total():
    hp = _hp()
    rows = [_row(0, 8192, 100, uuid="GPU-aaaa"), _row(1, 12288, 200, uuid="GPU-bbbb", bus="00000000:02:00.0")]
    cards = [
        {"n": 0, "vendor": "0x1002", "slot": "0000:c5:00.0", "total": 512 * _MIB, "used": 50 * _MIB},
        {"n": 1, "vendor": "0x8086", "slot": "0000:00:02.0"},
    ]
    config = hp.ProfileConfig(devices=["GPU-bbbb", "pci:0000:c5:00.0", "nvidia:1", "0000:00:02.0", "nvidia:7"])
    placement = _profile(hp, _tmp(), query=_Query(rows), cards=cards, config=config).placement()
    assert [d.id for d in placement.devices] == ["nvidia:1", "pci:0000:c5:00.0"]
    assert (placement.capacity_mib, placement.used_mib) == (12800.0, 250.0)


def test_hw9_the_cards_are_read_once_and_importing_the_profile_reads_nothing():
    spawned = []
    real_run = subprocess.run

    def refuse(*args, **kwargs):
        spawned.append(args)
        raise AssertionError("no process may run while the profile is imported")

    asked = []
    collector = types.ModuleType(_LM)
    collector.nvidia_smi_rows = lambda query, timeout: asked.append(query)
    subprocess.run = refuse
    try:
        hp = _open((_HP, source("hardware_profile.py")), seeded={_LM: collector})[_HP]
    finally:
        subprocess.run = real_run
    assert spawned == []
    assert asked == []
    assert hp._profile is None
    query = _Query([_row(0, 8192, 100)])
    profile = _profile(hp, _tmp(), query=query)
    assert query.calls == []
    for _ in range(3):
        profile.devices()
        profile.placement()
    assert len(query.calls) == 1


def test_hw10_used_memory_is_cached_for_its_ttl_then_refreshed_off_the_callers_path():
    hp = _hp()
    query = _Query([_row(0, 8192, 1000)], [_row(0, 8192, 3000)])
    clock, spawn = _Clock(100.0), _Spawn()
    profile = _profile(hp, _tmp(), query=query, clock=clock, spawn=spawn, config=hp.ProfileConfig(vram_used_ttl_s=5.0))
    first = profile.placement()
    assert (first.used_mib, first.used_age_s) == (1000.0, 0.0)
    clock.t = 104.0
    assert profile.placement().used_mib == 1000.0
    assert len(query.calls) == 1 and spawn.pending == []
    clock.t = 106.0
    stale = profile.placement()
    assert (stale.used_mib, stale.used_age_s) == (1000.0, 6.0)
    assert len(query.calls) == 1
    profile.placement()
    assert len(spawn.pending) == 1
    spawn.run()
    assert len(query.calls) == 2
    clock.t = 106.5
    fresh = profile.placement()
    assert (fresh.used_mib, fresh.used_age_s) == (3000.0, 0.5)


def test_hw11_a_failed_refresh_keeps_the_last_figure_with_its_age_and_none_read_is_unknown():
    hp = _hp()
    tmp = _tmp()
    clock, spawn = _Clock(100.0), _Spawn()
    profile = _profile(hp, tmp / "kept", query=_Query([_row(0, 8192, 1000)], None), clock=clock, spawn=spawn)
    profile.placement()
    clock.t = 106.0
    profile.placement()
    spawn.run()
    clock.t = 107.0
    kept = profile.placement()
    assert (kept.used_mib, kept.used_age_s) == (1000.0, 7.0)
    assert [d.id for d in kept.devices] == ["nvidia:0"]
    never = _profile(hp, tmp / "never", query=_Query([_row(0, 8192, "[N/A]")])).placement()
    assert (never.capacity_mib, never.used_mib, never.used_age_s) == (8192.0, None, None)


def test_hw12_meminfo_gives_the_total_and_available_ram_and_unknown_is_zero():
    hp = _hp()
    tmp = _tmp()
    full = tmp / "full"
    full.write_text("MemTotal:       65536000 kB\nMemFree:  100 kB\nMemAvailable:   32768000 kB\n", encoding="utf-8")
    info = hp.read_meminfo(str(full))
    assert (info.total_mib, info.available_mib) == (64000.0, 32000.0)
    partial = tmp / "partial"
    partial.write_text("MemTotal: 2048 kB\n", encoding="utf-8")
    info = hp.read_meminfo(str(partial))
    assert (info.total_mib, info.available_mib) == (2.0, 0.0)
    garbled = tmp / "garbled"
    garbled.write_text("MemTotal: lots kB\nMemAvailable: 1024 kB\n", encoding="utf-8")
    info = hp.read_meminfo(str(garbled))
    assert (info.total_mib, info.available_mib) == (0.0, 1.0)
    gone = hp.read_meminfo(str(tmp / "absent"))
    assert (gone.total_mib, gone.available_mib) == (0.0, 0.0)


def test_hw13_pressure_stall_information_is_parsed_per_resource_and_absence_is_named():
    hp = _hp()
    tmp = _tmp()
    root = _psi(
        tmp / "pressure",
        memory=_psi_line("some", 12.5, 6.0, 1.0, 999) + _psi_line("full", 3.25, 1.0, 0.5, 10),
        cpu=_psi_line("some", 40.0, 20.0, 10.0),
        io="some avg10=oops avg60=1.00 avg300=0.00 total=1\nfull avg10=2.00 avg60=1.00 avg300=0.50 total=7\n",
    )
    psi = hp.read_pressure(root)
    assert psi == {
        "memory": {
            "some": {"avg10": 12.5, "avg60": 6.0, "avg300": 1.0},
            "full": {"avg10": 3.25, "avg60": 1.0, "avg300": 0.5},
        },
        "cpu": {"some": {"avg10": 40.0, "avg60": 20.0, "avg300": 10.0}},
        "io": {"full": {"avg10": 2.0, "avg60": 1.0, "avg300": 0.5}},
    }
    only_cpu = _psi(tmp / "only-cpu", cpu=_psi_line("some", 1.0))
    assert set(hp.read_pressure(only_cpu)) == {"cpu"}
    assert hp.read_pressure(str(tmp / "nowhere")) is None


def test_hw14_the_profile_file_parses_and_ships_the_decided_defaults(caplog):
    hp = _hp()
    tmp = _tmp()
    shipped = hp.load_config(Path(source("config", "hardware_profile.yaml")))
    assert shipped == hp.ProfileConfig()
    assert (shipped.devices, shipped.integrated_below_gb, shipped.kind_overrides, shipped.vram_used_ttl_s, shipped.nvidia_smi_timeout_s) == (
        "auto", 1.0, {}, 5.0, 5.0,
    )
    assert shipped.vram_used_max_age_s == 600.0
    assert hp.load_config(tmp / "missing.yaml") == hp.ProfileConfig()
    path = tmp / "hardware_profile.yaml"
    path.write_text(
        "devices: [nvidia:1, 'pci:0000:03:00.0']\nintegrated_below_gb: 2.5\nkind_overrides:\n"
        "  GPU-x: integrated\nvram_used_ttl_s: 1.0\nnvidia_smi_timeout_s: 2.0\nvram_used_max_age_s: 30.0\n",
        encoding="utf-8",
    )
    read = hp.load_config(path)
    assert (read.devices, read.integrated_below_gb, read.kind_overrides, read.vram_used_ttl_s, read.nvidia_smi_timeout_s) == (
        ["nvidia:1", "pci:0000:03:00.0"], 2.5, {"GPU-x": "integrated"}, 1.0, 2.0,
    )
    assert read.vram_used_max_age_s == 30.0
    path.write_text(
        "devices: 3\nintegrated_below_gb: -1\nkind_overrides:\n  GPU-x: purple\n"
        "vram_used_ttl_s: .nan\nnvidia_smi_timeout_s: 0\nvram_used_max_age_s: -5\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger=_HP):
        assert hp.load_config(path) == hp.ProfileConfig()
    warned = " ".join(r.getMessage() for r in caplog.records)
    for key in ("devices", "integrated_below_gb", "kind_overrides", "vram_used_ttl_s", "nvidia_smi_timeout_s", "vram_used_max_age_s"):
        assert key in warned, key


def test_hw15_no_module_of_the_package_imports_psutil_any_more():
    package = Path(source("resource_governor.py")).parent
    checked = 0
    for directory, subdirs, files in os.walk(package):
        # The package's data directory holds the maintainer's content, not
        # code: the walk never enters it.
        subdirs[:] = sorted(d for d in subdirs if d not in ("data", "__pycache__"))
        for name in sorted(f for f in files if f.endswith(".py")):
            path = Path(directory) / name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                assert all(n.split(".")[0] != "psutil" for n in names), path.relative_to(package)
            checked += 1
    assert checked > 100


def test_hw16_smart_router_and_live_metrics_read_the_ram_through_the_profile():
    reads = []
    fake = types.ModuleType(_HP)

    def read_meminfo(path="/proc/meminfo"):
        reads.append(path)
        return types.SimpleNamespace(total_mib=8000.0, available_mib=3000.0)

    fake.read_meminfo = read_meminfo
    loaded = _open((_SR, source("smart_router.py")), (_LM, source("live_metrics.py")), seeded={_HP: fake})
    assert loaded[_SR]._get_available_ram_mb() == 3000.0
    assert loaded[_LM]._get_system_memory() == (5000.0, 8000.0)
    assert reads == ["/proc/meminfo", "/proc/meminfo"]
    while _CLOSERS:
        _CLOSERS.pop()()
    # Without the profile, the RAM is unknown: the pre-flight excludes nothing.
    alone = _open((_SR, source("smart_router.py")), (_LM, source("live_metrics.py")), blocked=(_HP,))
    assert alone[_SR]._get_available_ram_mb() == 0.0
    assert alone[_LM]._get_system_memory() == (0.0, 0.0)


def test_hw17_the_governors_own_meminfo_reader_agrees_with_the_profiles():
    loaded = _rg_hp()
    rg, hp = loaded[_RG], loaded[_HP]
    tmp = _tmp()
    fixtures = {
        "full": "MemTotal: 65536000 kB\nMemAvailable: 32768000 kB\n",
        "partial": "MemTotal: 2048 kB\n",
        "garbled": "MemTotal: lots kB\nMemAvailable: 1024 kB\n",
        "empty": "",
    }
    for name, text in fixtures.items():
        path = tmp / name
        path.write_text(text, encoding="utf-8")
        info = hp.read_meminfo(str(path))
        assert rg._read_meminfo_mb(str(path)) == (info.available_mib, info.total_mib), name
    assert rg._read_meminfo_mb(str(tmp / "absent")) == (0.0, 0.0)
    assert not hasattr(rg, "_read_available_ram_mb")


# ---------------------------------------------------------------------------
# HW18-HW29 -- the governor that sees the machine
# ---------------------------------------------------------------------------


def test_hw18_an_unconfigured_capacity_is_the_sum_of_the_selected_cards():
    loaded = _rg_hp()
    rg, hp = loaded[_RG], loaded[_HP]
    tmp = _tmp()
    profile = _profile(hp, tmp, query=_Query([_row(0, 8192, 0), _row(1, 12288, 0)]))
    snap = _governor(rg, tmp, profile=profile).refresh(force=True)
    assert (snap.capacity_gb, snap.capacity_source, snap.device_count) == (20.0, "probe", 2)
    assert "S4-capacity-probe" in snap.sources
    assert [d["id"] for d in snap.to_dict()["devices"]] == ["nvidia:0", "nvidia:1"]


def test_hw19_the_safety_margin_is_kept_on_every_selected_card():
    loaded = _rg_hp()
    rg, hp = loaded[_RG], loaded[_HP]
    tmp = _tmp()
    body = (
        "enabled: true\ntotal_vram_gb: null\nsafety_margin_gb: 1.5\n"
        "weights_overrides:\n  models:\n    fits: 16.9\n    spills: 17.5\n"
    )
    two = _profile(hp, tmp / "two", query=_Query([_row(0, 8192, 0), _row(1, 12288, 0)]))
    gov = _governor(rg, tmp / "two", profile=two, body=body)
    gov.refresh(force=True)
    # 20 GiB on two cards, 1.5 kept on each: 17.0 free.
    assert gov.admit("fits", requested_ctx=None, caller="direct").reason == "fits"
    spilled = gov.admit("spills", requested_ctx=None, caller="direct")
    assert spilled.partial_offload is True
    assert spilled.vram_cost_gb == 17.0
    one = _profile(hp, tmp / "one", query=_Query([_row(0, 20480, 0)]))
    gov = _governor(rg, tmp / "one", profile=one, body=body)
    gov.refresh(force=True)
    assert gov.admit("spills", requested_ctx=None, caller="direct").reason == "fits"


def test_hw20_the_vram_other_programs_hold_is_deducted_and_never_below_zero():
    loaded = _rg_hp()
    rg, hp = loaded[_RG], loaded[_HP]
    tmp = _tmp()
    body = (
        "enabled: true\ntotal_vram_gb: null\nsafety_margin_gb: 1.5\n"
        "weights_overrides:\n  models:\n    small: 4.4\n    large: 4.6\n"
    )
    resident = [_resident("resident", 4 * _GIB)]
    busy = _profile(hp, tmp / "busy", query=_Query([_row(0, 12288, 6144)]))
    gov = _governor(rg, tmp / "busy", profile=busy, body=body, loaded=resident)
    snap = gov.refresh(force=True)
    # The card holds 6 GiB; the engines declare 4: the other 2 are someone else's.
    assert (snap.vram_in_use_gb, snap.vram_others_gb, snap.vram_available_gb) == (4.0, 2.0, 6.0)
    assert snap.to_dict()["vram_others_gb"] == 2.0
    pressure = gov.pressure_state()
    assert (pressure["effective_capacity_gb"], pressure["others_gb"], pressure["ratio"]) == (10.0, 2.0, 0.4)
    # 12 - 4 - 2 - 1.5 = 4.5 free.
    assert gov.admit("small", requested_ctx=None, caller="direct").reason == "fits"
    assert gov.admit("large", requested_ctx=None, caller="direct").partial_offload is True
    quiet = _profile(hp, tmp / "quiet", query=_Query([_row(0, 12288, 2048)]))
    gov = _governor(rg, tmp / "quiet", profile=quiet, body=body, loaded=resident)
    assert gov.refresh(force=True).vram_others_gb == 0.0


def test_hw21_a_configured_capacity_reads_no_card_and_counts_no_other_program():
    loaded = _rg_hp()
    rg, hp = loaded[_RG], loaded[_HP]
    tmp = _tmp()
    query = _Query([_row(0, 12288, 9000)])
    gov = _governor(rg, tmp, profile=_profile(hp, tmp, query=query), body="enabled: true\ntotal_vram_gb: 12.0\n")
    snap = gov.refresh(force=True)
    assert (snap.capacity_gb, snap.capacity_source, snap.vram_others_gb, snap.device_count) == (12.0, "config", 0.0, 0)
    assert query.calls == []


def test_hw22_snapshot_rebuilds_never_wait_on_nvidia_smi_after_the_first_reading():
    loaded = _rg_hp()
    rg, hp = loaded[_RG], loaded[_HP]
    tmp = _tmp()
    query = _Query([_row(0, 12288, 1000)])
    clock, spawn = _Clock(100.0), _Spawn()
    gov = _governor(rg, tmp, profile=_profile(hp, tmp, query=query, clock=clock, spawn=spawn))
    for _ in range(4):
        gov.refresh(force=True)
    assert len(query.calls) == 1
    clock.t = 200.0
    gov.refresh(force=True)
    assert len(query.calls) == 1
    assert len(spawn.pending) == 1


def test_hw23_a_null_reserve_is_sized_from_the_machine_and_a_number_stays_fixed():
    rg = _rg_hp()[_RG]
    tmp = _tmp()
    adaptive = "enabled: true\ntotal_vram_gb: 8.0\noffload:\n  ram_reserve_gb: null\n"
    fixed = "enabled: true\ntotal_vram_gb: 8.0\noffload:\n  ram_reserve_gb: 4.5\n"

    def reserve(body, total_mib, sub):
        gov = _governor(rg, tmp / sub, profile=None, body=body, total_mib=total_mib)
        snap = gov.refresh(force=True)
        return gov.effective_ram_reserve_gb(snap), gov.ram_reserve_state(snap)["mode"]

    assert reserve(adaptive, 65536.0, "64") == (4.0, "adaptive")
    assert reserve(adaptive, 16384.0, "16") == (2.0, "adaptive")
    assert reserve(adaptive, 262144.0, "256") == (8.0, "adaptive")
    # No total to size it from: the ceiling, the side that protects the rest.
    assert reserve(adaptive, 0.0, "unknown") == (8.0, "adaptive")
    assert reserve(fixed, 262144.0, "fixed") == (4.5, "fixed")


def test_hw24_memory_pressure_multiplies_the_adaptive_reserve_with_hysteresis():
    loaded = _rg_hp()
    rg, hp = loaded[_RG], loaded[_HP]
    tmp = _tmp()
    root = tmp / "pressure"
    _psi(root, memory=_psi_line("some", 0.0))
    adaptive = "enabled: true\ntotal_vram_gb: 8.0\noffload:\n  ram_reserve_gb: null\n"
    gov = _governor(rg, tmp / "gov", profile=_profile(hp, tmp / "gov", pressure=str(root)), body=adaptive)
    seen = []
    for avg10 in (2.0, 10.0, 7.0, 4.9, 7.0):
        _psi(root, memory=_psi_line("some", avg10))
        snap = gov.refresh(force=True)
        seen.append((gov.effective_ram_reserve_gb(snap), gov.ram_reserve_state(snap)["pressure_applied"]))
    assert seen == [(4.0, False), (8.0, True), (8.0, True), (4.0, False), (4.0, False)]
    _psi(root, memory=_psi_line("some", 50.0))
    fixed = _governor(
        rg, tmp / "fixed", profile=_profile(hp, tmp / "fixed", pressure=str(root)),
        body="enabled: true\ntotal_vram_gb: 8.0\noffload:\n  ram_reserve_gb: 4.5\n",
    )
    assert fixed.effective_ram_reserve_gb(fixed.refresh(force=True)) == 4.5
    off = _governor(
        rg, tmp / "off", profile=_profile(hp, tmp / "off", pressure=str(root)),
        body=adaptive + "host_pressure:\n  enabled: false\n",
    )
    assert off.effective_ram_reserve_gb(off.refresh(force=True)) == 4.0


def test_hw25_a_split_is_priced_against_the_effective_reserve():
    loaded = _rg_hp()
    rg, hp = loaded[_RG], loaded[_HP]
    tmp = _tmp()
    root = tmp / "pressure"
    _psi(root, memory=_psi_line("some", 0.0))
    body = (
        "enabled: true\ntotal_vram_gb: 10.0\nsafety_margin_gb: 1.5\noffload:\n  ram_reserve_gb: null\n"
        "weights_overrides:\n  models:\n    m: 11.0\n"
    )
    # 8.5 GiB on the GPU, 2.5 to RAM. 8 GiB available: 4.0 usable over a 4.0
    # reserve, none over the 8.0 the same reserve becomes under pressure.
    gov = _governor(
        rg, tmp / "gov", profile=_profile(hp, tmp / "gov", pressure=str(root)), body=body,
        total_mib=65536.0, available_mib=8192.0,
    )
    gov.refresh(force=True)
    assert gov.admit("m", requested_ctx=None, caller="direct").partial_offload is True
    _psi(root, memory=_psi_line("some", 25.0))
    gov.refresh(force=True)
    refused = gov.admit("m", requested_ctx=None, caller="direct")
    assert refused.admitted is False
    assert refused.reason == "vram_insufficient+ram_insufficient"
    assert refused.ram_shortfall_gb == 2.5


def test_hw26_without_pressure_files_the_snapshot_says_so_and_the_reserve_is_its_base():
    loaded = _rg_hp()
    rg, hp = loaded[_RG], loaded[_HP]
    tmp = _tmp()
    gov = _governor(
        rg, tmp, profile=_profile(hp, tmp, pressure=str(tmp / "nowhere")),
        body="enabled: true\ntotal_vram_gb: 8.0\noffload:\n  ram_reserve_gb: null\n",
    )
    snap = gov.refresh(force=True)
    assert snap.host_pressure is None
    assert snap.to_dict()["host_pressure"] is None
    assert gov.ram_reserve_state(snap) == {
        "mode": "adaptive",
        "gb": 4.0,
        "base_gb": 4.0,
        "pressure_applied": False,
        "pressure_known": False,
    }


def test_hw27_the_shipped_file_sizes_the_reserve_from_the_machine_and_the_new_blocks_hold_their_ranges():
    rg = _rg_hp()[_RG]
    tmp = _tmp()
    shipped = Path(source("config", "resource_governor.yaml"))
    raw = yaml.safe_load(shipped.read_text(encoding="utf-8"))
    assert raw["offload"]["ram_reserve_gb"] is None
    assert raw["ram_reserve"] == {"fraction": 0.0625, "floor_gb": 2.0, "ceiling_gb": 8.0, "pressure_factor": 2.0}
    assert raw["host_pressure"] == {"enabled": True, "memory_enter_some_avg10": 10.0, "memory_exit_some_avg10": 5.0}
    cfg = rg.load_config(shipped)

    def reserve_keys(c):
        return (
            c.offload_ram_reserve_gb,
            c.ram_reserve_fraction,
            c.ram_reserve_floor_gb,
            c.ram_reserve_ceiling_gb,
            c.ram_reserve_pressure_factor,
            c.host_pressure_enabled,
            c.host_pressure_memory_enter,
            c.host_pressure_memory_exit,
        )

    assert reserve_keys(cfg) == (None, 0.0625, 2.0, 8.0, 2.0, True, 10.0, 5.0)
    defaults = reserve_keys(rg.GovernorConfig())
    assert defaults == (4.0, 0.0625, 2.0, 8.0, 2.0, True, 10.0, 5.0)
    path = tmp / "resource_governor.yaml"
    path.write_text(
        "ram_reserve:\n  fraction: 1.5\n  floor_gb: -1\n  ceiling_gb: .inf\n  pressure_factor: 0.5\n"
        "host_pressure:\n  enabled: maybe\n  memory_enter_some_avg10: 101\n  memory_exit_some_avg10: -1\n",
        encoding="utf-8",
    )
    assert reserve_keys(rg.load_config(path)) == defaults
    # A crossed pair is set aside whole, for the defaults of both.
    path.write_text(
        "ram_reserve:\n  floor_gb: 9.0\n  ceiling_gb: 3.0\n"
        "host_pressure:\n  memory_enter_some_avg10: 4.0\n  memory_exit_some_avg10: 6.0\n",
        encoding="utf-8",
    )
    assert reserve_keys(rg.load_config(path)) == defaults
    path.write_text("offload:\n  ram_reserve_gb: null\n", encoding="utf-8")
    assert rg.load_config(path).offload_ram_reserve_gb is None
    assert rg.load_config(tmp / "missing.yaml").offload_ram_reserve_gb == 4.0


def test_hw28_the_routes_show_the_machine_and_write_the_new_keys_in_range_only():
    loaded = _rg_hp((_ROUTES, source("api", "routes_governor.py")), packages=("opti_oignon.api",))
    rg, hp, routes = loaded[_RG], loaded[_HP], loaded[_ROUTES]
    tmp = _tmp()
    profile = _profile(
        hp, tmp, query=_Query([_row(0, 12288, 6144)]),
        pressure=_psi(tmp / "pressure", memory=_psi_line("some", 1.0)),
    )
    gov = _governor(rg, tmp, profile=profile, body="enabled: true\ntotal_vram_gb: null\noffload:\n  ram_reserve_gb: null\n")
    status = routes.status_payload(gov)
    assert [d["id"] for d in status["hardware"]["devices"]] == ["nvidia:0"]
    assert status["hardware"]["selected"] == ["nvidia:0"]
    assert status["snapshot"]["vram_others_gb"] == 6.0
    assert status["snapshot"]["host_pressure"]["memory"]["some"]["avg10"] == 1.0
    assert status["ram_reserve"]["mode"] == "adaptive"
    view = routes.config_read_payload(gov)
    assert view["config"]["ram_reserve"] == {"fraction": 0.0625, "floor_gb": 2.0, "ceiling_gb": 8.0, "pressure_factor": 2.0}
    assert view["config"]["host_pressure"] == {"enabled": True, "memory_enter_some_avg10": 10.0, "memory_exit_some_avg10": 5.0}
    assert {
        "ram_reserve.fraction",
        "ram_reserve.floor_gb",
        "ram_reserve.ceiling_gb",
        "ram_reserve.pressure_factor",
        "host_pressure.enabled",
        "host_pressure.memory_enter_some_avg10",
        "host_pressure.memory_exit_some_avg10",
    } <= set(view["writable_keys"])
    path = Path(tempfile.mkdtemp(prefix="hw-api-")) / "resource_governor.yaml"
    path.write_text(Path(source("config", "resource_governor.yaml")).read_text(encoding="utf-8"), encoding="utf-8")
    routes.config_write_payload(
        rg.load_config(path),
        {"ram_reserve.fraction": 0.125, "host_pressure.memory_enter_some_avg10": 20.0, "offload.ram_reserve_gb": 3.0},
        path,
        lambda: None,
        lambda changes: None,
    )
    written = rg.load_config(path)
    assert (written.ram_reserve_fraction, written.host_pressure_memory_enter, written.offload_ram_reserve_gb) == (0.125, 20.0, 3.0)
    routes.config_write_payload(written, {"offload.ram_reserve_gb": None}, path, lambda: None, lambda changes: None)
    assert rg.load_config(path).offload_ram_reserve_gb is None
    before = path.read_bytes()
    for bad in (
        {"ram_reserve.fraction": -0.1},
        {"ram_reserve.fraction": 1.5},
        {"ram_reserve.pressure_factor": 0.5},
        {"ram_reserve.floor_gb": 9.0},
        {"host_pressure.memory_exit_some_avg10": 30.0},
        {"host_pressure.memory_enter_some_avg10": 101.0},
    ):
        with pytest.raises(routes.ConfigWriteError) as refused:
            routes.config_write_payload(rg.load_config(path), bad, path, lambda: None, lambda changes: None)
        assert refused.value.status_code == 400, bad
    assert path.read_bytes() == before


def test_hw29_speculative_decoding_sizes_its_budget_from_the_detected_capacity():
    tmp = _tmp()
    shipped = source("config", "speculative_decoding.yaml")

    def available(capacity_mib=None, blocked=()):
        seeded = {}
        if capacity_mib != "absent":
            fake = types.ModuleType(_HP)
            fake.get_hardware_profile = lambda: types.SimpleNamespace(
                placement=lambda: types.SimpleNamespace(capacity_mib=capacity_mib)
            )
            seeded = {_HP: fake}
        sd = _open((_SD, source("speculative_decoding.py")), seeded=seeded, blocked=blocked)[_SD]
        manager = sd.SpeculativeDecodingManager(
            config_path=shipped, stats_path=str(tmp / "stats.json"), availability_probe=lambda: None
        )
        result = (manager.get_vram_calculator().available_vram_gb, manager.get_draft_selector()._vram_calc.available_vram_gb)
        while _CLOSERS:
            _CLOSERS.pop()()
        return result

    # 8 GiB detected, 1.5 kept: 6.5 for both the calculator and the selector.
    assert available(8192.0) == (6.5, 6.5)
    # Nothing detected: the file's default, as its comment says.
    assert available(None) == (22.5, 22.5)
    assert available("absent", blocked=(_HP,)) == (22.5, 22.5)


# ---------------------------------------------------------------------------
# HW30-HW36 -- readings that describe the card as it is
# ---------------------------------------------------------------------------


def test_hw30_a_card_reading_taken_before_the_engines_last_changed_is_not_set_against_them():
    loaded = _rg_hp()
    rg, hp = loaded[_RG], loaded[_HP]
    tmp = _tmp()
    # A game holds 6 GiB throughout; an 8 GiB model loads, then is evicted.
    query = _Query([_row(0, 12288, 6144)], [_row(0, 12288, 14336)], [_row(0, 12288, 6144)])
    clock, spawn = _Clock(100.0), _Spawn()
    warmup = _Warmup([])
    gov = _governor(rg, tmp, profile=_profile(hp, tmp, query=query, clock=clock, spawn=spawn), warmup=warmup)
    first = gov.refresh(force=True)
    assert (first.vram_others_gb, first.vram_others_carried) == (6.0, False)
    warmup._loaded = [_resident("m", 8 * _GIB)]
    gov.invalidate_on_load("m", None)
    assert len(spawn.pending) == 1
    loading = gov.refresh(force=True)
    # The card was read before the load: 6 - 8 is not what others hold.
    assert loading.vram_in_use_gb == 8.0
    assert (loading.vram_others_gb, loading.vram_others_carried) == (6.0, True)
    spawn.run()
    loaded_now = gov.refresh(force=True)
    assert (loaded_now.vram_others_gb, loaded_now.vram_others_carried) == (6.0, False)
    warmup._loaded = []
    gov.invalidate_on_evict("m")
    evicting = gov.refresh(force=True)
    # Nor is the 14 a reading from before the eviction shows.
    assert (evicting.vram_others_gb, evicting.vram_others_carried) == (6.0, True)
    spawn.run()
    assert gov.refresh(force=True).vram_others_gb == 6.0


def test_hw31_a_card_reading_older_than_its_max_age_is_unknown_and_a_hung_refresh_is_not_doubled():
    hp = _hp()
    clock, spawn = _Clock(100.0), _Spawn()
    config = hp.ProfileConfig(vram_used_ttl_s=5.0, vram_used_max_age_s=60.0)
    profile = _profile(hp, _tmp(), query=_Query([_row(0, 8192, 1000)]), clock=clock, spawn=spawn, config=config)
    profile.placement()
    clock.t = 160.0
    assert profile.placement().used_mib == 1000.0
    # The refresh started at 160 never returns.
    clock.t = 161.0
    late = profile.placement()
    assert (late.capacity_mib, late.used_mib, late.used_age_s) == (8192.0, None, None)
    assert len(spawn.pending) == 1


def test_hw32_numeric_cuda_entries_count_only_where_the_card_order_is_known(caplog):
    hp = _hp()
    tmp = _tmp()
    rows = [_row(0, 8192, 0, name="NVIDIA GeForce GTX 1080"), _row(1, 24576, 0, name="NVIDIA GeForce RTX 4090")]

    def selected(environ, sub):
        return [d.id for d in _profile(hp, tmp / sub, query=_Query(rows), environ=environ).placement().devices]

    # CUDA puts the fastest card first unless told otherwise: index 0 may be either.
    with caplog.at_level(logging.WARNING, logger=_HP):
        assert selected({"CUDA_VISIBLE_DEVICES": "0"}, "fastest") == []
    assert "CUDA_DEVICE_ORDER" in " ".join(r.getMessage() for r in caplog.records)
    assert selected({"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, "pci") == ["nvidia:0"]
    assert selected({"CUDA_VISIBLE_DEVICES": "GPU-0001"}, "uuid") == ["nvidia:1"]


def test_hw33_a_bus_id_in_any_written_form_names_its_card_and_a_name_that_matches_none_is_warned(caplog):
    hp = _hp()
    rows = [_row(0, 8192, 0, bus="00000000:0A:00.0")]
    config = hp.ProfileConfig(
        devices=["00000000:0A:00.0", "PCI:0000:ff:00.0"],
        kind_overrides={"0000:0A:00.0": "integrated"},
    )
    with caplog.at_level(logging.WARNING, logger=_HP):
        placement = _profile(hp, _tmp(), query=_Query(rows), config=config).placement()
    assert [d.id for d in placement.devices] == ["nvidia:0"]
    assert placement.devices[0].kind == "integrated"
    assert "PCI:0000:ff:00.0" in " ".join(r.getMessage() for r in caplog.records)


def test_hw34_a_uuid_prefix_names_a_card_only_when_it_names_one():
    hp = _hp()
    tmp = _tmp()
    rows = [_row(0, 8192, 0, uuid="GPU-ab12-0001"), _row(1, 8192, 0, uuid="GPU-ab12-0002")]

    def selected(value, sub):
        environ = {"CUDA_VISIBLE_DEVICES": value}
        return [d.id for d in _profile(hp, tmp / sub, query=_Query(rows), environ=environ).placement().devices]

    assert selected("GPU-ab12-0002", "exact") == ["nvidia:1"]
    assert selected("GPU-ab12-00", "ambiguous") == []
    assert selected("GPU-ab12-0001,GPU-ab12", "stops") == ["nvidia:0"]


def test_hw35_a_config_write_checks_its_pairs_against_the_file_it_writes():
    loaded = _rg_hp((_ROUTES, source("api", "routes_governor.py")), packages=("opti_oignon.api",))
    rg, routes = loaded[_RG], loaded[_ROUTES]
    path = Path(tempfile.mkdtemp(prefix="hw-pairs-")) / "resource_governor.yaml"
    text = Path(source("config", "resource_governor.yaml")).read_text(encoding="utf-8")
    # A hand edit crossed the pair; the load set both sides aside.
    path.write_text(text.replace("  floor_gb: 2.0\n", "  floor_gb: 10.0\n").replace("  ceiling_gb: 8.0\n", "  ceiling_gb: 6.0\n"), encoding="utf-8")
    current = rg.load_config(path)
    assert (current.ram_reserve_floor_gb, current.ram_reserve_ceiling_gb) == (2.0, 8.0)
    before = path.read_bytes()
    with pytest.raises(routes.ConfigWriteError) as refused:
        routes.config_write_payload(current, {"ram_reserve.ceiling_gb": 9.0}, path, lambda: None, lambda changes: None)
    assert refused.value.status_code == 400
    assert path.read_bytes() == before
    routes.config_write_payload(
        current, {"ram_reserve.floor_gb": 4.0, "ram_reserve.ceiling_gb": 9.0}, path, lambda: None, lambda changes: None
    )
    written = rg.load_config(path)
    assert (written.ram_reserve_floor_gb, written.ram_reserve_ceiling_gb) == (4.0, 9.0)


def test_hw36_a_first_reading_slower_than_the_ttl_does_not_start_a_second_at_once():
    hp = _hp()
    clock, spawn = _Clock(100.0), _Spawn()

    def slow(query, timeout):
        # The first answer takes longer than the 5 s the reading is served for.
        clock.t += 7.0
        return [_row(0, 8192, 1000)]

    profile = _profile(hp, _tmp(), query=slow, clock=clock, spawn=spawn)
    first = profile.placement()
    assert (first.used_mib, first.used_age_s) == (1000.0, 0.0)
    assert spawn.pending == []


_USER_ROOT = "user.slice/user-1000.slice/user@1000.service"


def _cgroup(root, relative, some=None):
    """A cgroup directory under ``root``; with ``some``, its cpu.pressure."""
    path = Path(root).joinpath(*relative.split("/"))
    path.mkdir(parents=True, exist_ok=True)
    if some is not None:
        (path / "cpu.pressure").write_text(_psi_line("some", some) + _psi_line("full", 0.0), encoding="utf-8")
    return path


def _self_cgroup(tmp, relative):
    """A /proc/self/cgroup of our own, naming the process's cgroup."""
    path = tmp / "self-cgroup"
    path.write_text(f"0::/{relative}\n", encoding="utf-8")
    return path


def test_hw37_the_cpu_pressure_others_suffer_is_the_highest_of_the_users_other_leaf_cgroups():
    hp = _hp()
    tmp = _tmp()
    _cgroup(tmp / "cg", f"{_USER_ROOT}/app.slice/editor.scope", 3.5)
    _cgroup(tmp / "cg", f"{_USER_ROOT}/app.slice/browser.scope", 12.25)
    _cgroup(tmp / "cg", f"{_USER_ROOT}/session.slice/audio.service", 0.5)
    _cgroup(tmp / "cg", f"{_USER_ROOT}/app.slice/oo.service", 0.0)
    reading = hp.read_cgroup_cpu_pressure(tmp / "cg", _self_cgroup(tmp, f"{_USER_ROOT}/app.slice/oo.service"))
    assert reading == {"source": "cgroups", "some_avg10": 12.25, "cgroup": "app.slice/browser.scope", "count": 3}


def test_hw38_neither_the_process_own_cgroup_nor_anything_under_it_nor_a_parent_counts():
    hp = _hp()
    tmp = _tmp()
    # The slice carries the stalls of every cgroup under it, ours included.
    _cgroup(tmp / "cg", f"{_USER_ROOT}/app.slice", 60.0)
    _cgroup(tmp / "cg", f"{_USER_ROOT}/app.slice/oo.service", 50.0)
    _cgroup(tmp / "cg", f"{_USER_ROOT}/app.slice/oo.service/sandbox-1.scope", 40.0)
    _cgroup(tmp / "cg", f"{_USER_ROOT}/app.slice/editor.scope", 2.0)
    reading = hp.read_cgroup_cpu_pressure(tmp / "cg", _self_cgroup(tmp, f"{_USER_ROOT}/app.slice/oo.service"))
    assert reading == {"source": "cgroups", "some_avg10": 2.0, "cgroup": "app.slice/editor.scope", "count": 1}


def test_hw39_without_a_user_root_the_profile_falls_back_to_the_system_reading_or_says_it_does_not_know():
    hp = _hp()
    tmp = _tmp()
    _cgroup(tmp / "cg", "system.slice/oo.service", 30.0)
    _cgroup(tmp / "cg", "system.slice/other.service", 20.0)
    own = _self_cgroup(tmp, "system.slice/oo.service")
    assert hp.read_cgroup_cpu_pressure(tmp / "cg", own) is None

    def profile(pressure_root):
        return hp.HardwareProfile(
            config=hp.ProfileConfig(),
            drm_root=_drm(tmp / "drm", []),
            pressure_root=pressure_root,
            nvidia_query=_Query(None),
            environ={},
            clock=_Clock(),
            spawn=_Spawn(),
            cgroup_root=str(tmp / "cg"),
            self_cgroup=str(own),
        )

    system = _psi(tmp / "pressure", cpu=_psi_line("some", 7.5) + _psi_line("full", 1.0))
    assert profile(system).others_cpu_pressure() == {"source": "system", "some_avg10": 7.5, "cgroup": None, "count": None}
    assert profile(str(tmp / "no-pressure")).others_cpu_pressure() is None


# ---------------------------------------------------------------------------
# HW40-HW41 -- a machine with no card
# ---------------------------------------------------------------------------


def _pci(root, devices):
    """A /sys/bus/pci/devices of our own: (address, class, vendor) per device."""
    root.mkdir(parents=True, exist_ok=True)
    for address, klass, vendor in devices:
        device = root / address
        device.mkdir()
        (device / "class").write_text(klass + "\n", encoding="utf-8")
        (device / "vendor").write_text(vendor + "\n", encoding="utf-8")
    return str(root)


def _on_bus(hp, tmp, pci, *, cards=()):
    """A profile whose PCI bus is ``pci``, a fixture tree or a path that is none."""
    return hp.HardwareProfile(
        config=hp.ProfileConfig(),
        drm_root=_drm(tmp / "drm", list(cards)),
        pressure_root=str(tmp / "no-pressure"),
        nvidia_query=_Query(None),
        environ={},
        clock=_Clock(),
        spawn=_Spawn(),
        pci_root=pci,
    )


_BRIDGE = ("0000:00:00.0", "0x060000", "0x8086")
_IGPU = ("0000:00:02.0", "0x030000", "0x8086")
_HIDDEN = ("0000:01:00.0", "0x030200", "0x10de")


def test_hw40_the_profile_says_a_machine_has_no_card_only_when_its_bus_shows_none():
    """HW40 -- a bus whose only display controller is integrated (Intel), or
    that has none, is a machine with no card; an NVIDIA controller the DRM
    tree does not list (no nvidia-drm, nvidia-smi silent) is a card all the
    same, and a bus that cannot be read proves nothing. Reading an
    unreadable bus as empty -> RED."""
    hp = _hp()
    tmp = _tmp()
    igpu = {"n": 0, "vendor": "0x8086", "slot": "0000:00:02.0"}
    integrated = _on_bus(hp, tmp / "a", _pci(tmp / "a" / "pci", [_BRIDGE, _IGPU]), cards=[igpu])
    bare = _on_bus(hp, tmp / "b", _pci(tmp / "b" / "pci", [_BRIDGE]))
    hidden = _on_bus(hp, tmp / "c", _pci(tmp / "c" / "pci", [_BRIDGE, _IGPU, _HIDDEN]), cards=[igpu])
    unread = _on_bus(hp, tmp / "d", str(tmp / "d" / "no-bus"))
    assert [p.cards_absent() for p in (integrated, bare, hidden, unread)] == [True, True, False, False]


def test_hw41_the_governors_snapshot_says_what_the_profile_knows_of_the_cards():
    """HW41 -- with no capacity configured, a profile that knows the machine
    has no card makes a snapshot that says so; an NVIDIA controller on the
    bus whose memory nothing reads leaves the capacity unknown, not absent.
    Leaving the snapshot blind to the profile -> RED."""
    loaded = _rg_hp()
    hp, rg = loaded[_HP], loaded[_RG]
    tmp = _tmp()
    bare = _on_bus(hp, tmp / "a", _pci(tmp / "a" / "pci", [_BRIDGE]))
    hidden = _on_bus(hp, tmp / "b", _pci(tmp / "b" / "pci", [_BRIDGE, _HIDDEN]))
    seen = [_governor(rg, tmp / name, profile=p).refresh(force=True) for name, p in (("a", bare), ("b", hidden))]
    assert [(s.capacity_gb, s.cards_absent) for s in seen] == [(None, True), (None, False)]
