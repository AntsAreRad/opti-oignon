#!/usr/bin/env python3
"""Hardware profile: the cards, the memory and the pressure of this machine.

The governor prices admissions against the memory of the cards an inference
engine can use, and leaves the rest of the machine some room. This module is
where it learns what those are, on whatever machine it runs:

- Cards. Every NVIDIA card nvidia-smi lists (asked through the live-metrics
  collector, the one place the package runs that tool) and every card the
  kernel's DRM sysfs describes (AMD and Intel, read as files). A card has an
  id (``nvidia:<index>`` or ``pci:<bus id>``), a vendor, a kind (discrete,
  integrated or unknown), its own memory when the source reports it, and its
  used memory. An AMD card reporting less of its own memory than
  ``integrated_below_gb`` is taken as integrated: that memory is a carve-out
  of the system RAM. Intel cards are integrated unless an override says
  otherwise, and their memory is not read. A card both sources list is
  counted once, matched by its bus id.
- Selection. ``devices: auto`` selects the discrete cards with a known total
  of the first vendor that has any, NVIDIA first, then AMD, and honours
  CUDA_VISIBLE_DEVICES in the server's environment for NVIDIA cards: a
  UUID, or a prefix that names one card, needs no order; a number is an
  nvidia-smi index only under CUDA_DEVICE_ORDER=PCI_BUS_ID or when the
  cards are all one model, since CUDA otherwise counts the fastest first;
  and the list stops at its first entry that names no card. An explicit
  list selects exactly the cards it names, by id, UUID or bus id in any
  written form, that report a total; an entry that names none is warned.
  The placement capacity is the sum of the selected cards' totals. An
  engine running in another process, Ollama as a service, reads its own
  environment rather than this one: on such a host, name the cards it uses
  in the list.
- Refresh. The cards are read at the first question, never at import. Their
  used memory is served for ``vram_used_ttl_s``; a question asked after that
  gets the figure it has, with its age, while one background refresh reads
  the sources again, so no caller waits on nvidia-smi after the first
  reading. When the engines' holdings change (``invalidate_used``), the
  figures read before no longer pair with them and a fresh reading starts
  at once. A refresh that fails keeps the last figures and lets their age
  grow, until ``vram_used_max_age_s`` makes them unknown; a refresh that
  never returns is not doubled; a card no longer listed is forgotten.
- Memory and pressure. ``read_meminfo`` reads MemTotal and MemAvailable from
  /proc/meminfo, 0.0 for a field it cannot read, never a guess.
  ``read_pressure`` reads the kernel's pressure stall information
  (/proc/pressure/memory, cpu and io: the share of time some or all tasks
  were stalled waiting for that resource) and answers None where the kernel
  exposes none.

Every source is injectable -- the DRM and pressure roots, the nvidia-smi
query, the environment, the clock and the background runner -- so the
container proves the logic on fixtures; what a real machine's cards report is
measured on that machine.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "hardware_profile.yaml"

# The one question the profile asks nvidia-smi: one row per card.
NVIDIA_QUERY = "index,uuid,name,pci.bus_id,memory.total,memory.used"

_PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
# The order auto selection tries the vendors in.
_VENDOR_ORDER = ("nvidia", "amd", "intel")
_OVERRIDE_KINDS = ("discrete", "integrated")
_CARD_ENTRY = re.compile(r"^card\d+$")
_PRESSURE_RESOURCES = ("memory", "cpu", "io")
_PRESSURE_WINDOWS = ("avg10", "avg60", "avg300")
# The cgroup of a user's service manager, under which the user's programs run.
_USER_SERVICE = re.compile(r"^user@\d+\.service$")
_BYTES_PER_MIB = 1024.0 * 1024.0

# Sentinel distinguishing "not passed" from an explicit None injection.
_UNSET: Any = object()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProfileConfig:
    """hardware_profile.yaml, with its defaults."""

    devices: Any = "auto"
    integrated_below_gb: float = 1.0
    kind_overrides: dict[str, str] = field(default_factory=dict)
    vram_used_ttl_s: float = 5.0
    nvidia_smi_timeout_s: float = 5.0
    vram_used_max_age_s: float = 600.0


def _finite(raw: Any, key: str, default: float, *, low: float, low_open: bool) -> float:
    """``raw`` as a finite number at or above ``low`` (above, if open), else
    ``default`` with a warning naming the key."""
    value = None
    if not isinstance(raw, bool):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = None
    if value is not None and math.isfinite(value) and (value > low if low_open else value >= low):
        return value
    logger.warning(
        "%s %r is not a finite number %s %s; keeping %s",
        key, raw, "above" if low_open else "of at least", low, default,
    )
    return default


def load_config(config_path: str | Path | None = None) -> ProfileConfig:
    """Load hardware_profile.yaml; a missing or unreadable file is the defaults."""
    p = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    cfg = ProfileConfig()
    if not p.is_file():
        return cfg
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", p.name, exc)
        return cfg
    if not isinstance(raw, dict):
        logger.warning("%s root is not a mapping; using defaults", p.name)
        return cfg

    if "devices" in raw:
        devices = raw.get("devices")
        if devices == "auto":
            cfg.devices = "auto"
        elif isinstance(devices, list) and all(isinstance(d, str) and d for d in devices):
            cfg.devices = list(devices)
        else:
            logger.warning("devices %r is neither auto nor a list of card names; keeping auto", devices)
    if "integrated_below_gb" in raw:
        cfg.integrated_below_gb = _finite(
            raw.get("integrated_below_gb"), "integrated_below_gb", cfg.integrated_below_gb, low=0.0, low_open=False
        )
    overrides = raw.get("kind_overrides")
    if isinstance(overrides, dict):
        kept: dict[str, str] = {}
        for key, kind in overrides.items():
            if isinstance(key, str) and kind in _OVERRIDE_KINDS:
                kept[key] = kind
            else:
                logger.warning("kind_overrides %r: %r is neither discrete nor integrated; ignored", key, kind)
        cfg.kind_overrides = kept
    elif overrides is not None:
        logger.warning("kind_overrides is not a mapping; ignored")
    if "vram_used_ttl_s" in raw:
        cfg.vram_used_ttl_s = _finite(
            raw.get("vram_used_ttl_s"), "vram_used_ttl_s", cfg.vram_used_ttl_s, low=0.0, low_open=False
        )
    if "nvidia_smi_timeout_s" in raw:
        cfg.nvidia_smi_timeout_s = _finite(
            raw.get("nvidia_smi_timeout_s"), "nvidia_smi_timeout_s", cfg.nvidia_smi_timeout_s, low=0.0, low_open=True
        )
    if "vram_used_max_age_s" in raw:
        cfg.vram_used_max_age_s = _finite(
            raw.get("vram_used_max_age_s"), "vram_used_max_age_s", cfg.vram_used_max_age_s, low=0.0, low_open=True
        )
    return cfg


# ---------------------------------------------------------------------------
# What the profile reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuDevice:
    """One card, as its source describes it."""

    id: str
    vendor: str
    name: str
    kind: str
    total_mib: float | None
    bus_id: str | None = None
    uuid: str | None = None
    index: int | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vendor": self.vendor,
            "name": self.name,
            "kind": self.kind,
            "total_gb": round(self.total_mib / 1024.0, 3) if self.total_mib is not None else None,
            "bus_id": self.bus_id,
            "uuid": self.uuid,
            "index": self.index,
            "source": self.source,
        }


@dataclass(frozen=True)
class MemInfo:
    """System RAM in MiB; 0.0 is a field that could not be read."""

    total_mib: float = 0.0
    available_mib: float = 0.0


@dataclass(frozen=True)
class Placement:
    """The selected cards, their summed total, and their summed used memory.

    ``used_mib`` and ``used_age_s`` are None unless every selected card has
    a used figure younger than ``vram_used_max_age_s``; the age is that of
    the oldest figure in the sum. ``used_current`` is False when a figure
    was read before the engines' holdings last changed (``invalidate_used``):
    such a figure no longer pairs with what the engines declare now.
    """

    devices: tuple[GpuDevice, ...] = ()
    capacity_mib: float | None = None
    used_mib: float | None = None
    used_age_s: float | None = None
    used_current: bool = True


# ---------------------------------------------------------------------------
# Pure readers
# ---------------------------------------------------------------------------


def read_meminfo(path: str | Path = "/proc/meminfo") -> MemInfo:
    """MemTotal and MemAvailable in MiB; a field that cannot be read is 0.0."""
    fields = {"MemTotal:": 0.0, "MemAvailable:": 0.0}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return MemInfo()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in fields:
            try:
                value = float(parts[1])
            except ValueError:
                continue
            if math.isfinite(value) and value >= 0.0:
                fields[parts[0]] = value / 1024.0
    return MemInfo(total_mib=fields["MemTotal:"], available_mib=fields["MemAvailable:"])


def read_pressure(root: str | Path = "/proc/pressure") -> dict[str, dict[str, dict[str, float]]] | None:
    """The kernel's pressure stall information per resource, or None.

    ``{"memory": {"some": {"avg10": ..., "avg60": ..., "avg300": ...},
    "full": {...}}, "cpu": {...}, "io": {...}}``: the percentage of time, over
    the last 10, 60 and 300 seconds, that some (or all) non-idle tasks were
    stalled on that resource. A resource whose file is missing or unreadable
    is left out, a line that does not parse is skipped, and a kernel that
    exposes none of them answers None.
    """
    base = Path(root)
    found: dict[str, dict[str, dict[str, float]]] = {}
    for resource in _PRESSURE_RESOURCES:
        try:
            text = (base / resource).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines = _pressure_lines(text)
        if lines:
            found[resource] = lines
    return found or None


def _pressure_lines(text: str) -> dict[str, dict[str, float]]:
    """The "some" and "full" lines of one pressure file, each kept only when
    its three averages parse as finite numbers, none below zero."""
    lines: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] not in ("some", "full"):
            continue
        values: dict[str, float] | None = {}
        for item in parts[1:]:
            key, sep, raw = item.partition("=")
            if not sep or key not in _PRESSURE_WINDOWS:
                continue
            try:
                number = float(raw)
            except ValueError:
                values = None
                break
            if not math.isfinite(number) or number < 0.0:
                values = None
                break
            values[key] = number
        if values is not None and set(values) == set(_PRESSURE_WINDOWS):
            lines[parts[0]] = values
    return lines


def read_cgroup_cpu_pressure(
    cgroup_root: str | Path = "/sys/fs/cgroup", self_cgroup: str | Path = "/proc/self/cgroup"
) -> dict[str, Any] | None:
    """The CPU pressure the user's other programs suffer, or None.

    The process's own cgroup is the cgroup v2 line of ``self_cgroup``; the
    user's root is its nearest ancestor that is a user's service manager
    (``user@UID.service``). Every leaf cgroup under that root is read, other
    than the process's own and anything under it: a parent carries the stalls
    of every cgroup beneath it, the process's own among them, so only the
    leaves tell one program from another. The answer is the highest "some
    avg10" among them -- the share of the last ten seconds in which some task
    of that program waited for a CPU -- with the cgroup it was read in,
    relative to the root, and how many leaves were read. A process outside
    any user's service tree, or a tree with no readable leaf, answers None.
    """
    try:
        text = Path(self_cgroup).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    own = next((line[3:].strip("/ ") for line in text.splitlines() if line.startswith("0::")), "")
    parts = own.split("/") if own else []
    cut = next((i for i in range(len(parts) - 1, -1, -1) if _USER_SERVICE.match(parts[i])), None)
    if cut is None:
        return None
    root = Path(cgroup_root).joinpath(*parts[: cut + 1])
    own_relative = "/".join(parts[cut + 1 :])
    best: tuple[float, str] | None = None
    count = 0
    for directory, subdirs, _files in os.walk(root):
        relative = Path(directory).relative_to(root).as_posix()
        relative = "" if relative == "." else relative
        if own_relative and (relative == own_relative or relative.startswith(own_relative + "/")):
            subdirs[:] = []
            continue
        if subdirs:
            subdirs.sort()
            continue
        try:
            some = _pressure_lines((Path(directory) / "cpu.pressure").read_text(encoding="utf-8")).get("some")
        except (OSError, UnicodeDecodeError):
            continue
        if some is None:
            continue
        count += 1
        if best is None or some["avg10"] > best[0]:
            best = (some["avg10"], relative)
    if best is None:
        return None
    return {"source": "cgroups", "some_avg10": best[0], "cgroup": best[1], "count": count}


def _field(raw: Any) -> str | None:
    """A reported field, or None for an empty one or one the tool could not
    report ("[N/A]", "[Not Supported]")."""
    text = str(raw).strip() if raw is not None else ""
    if not text or text.startswith("[") or text.upper() == "N/A":
        return None
    return text


def _mib(raw: Any) -> float | None:
    text = _field(raw)
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _pci_address(text: str) -> str | None:
    """``text`` as a PCI address, 0000:01:00.0, or None when it is not one.

    The domain may be absent (01:00.0), written with eight digits as
    nvidia-smi prints it (00000000:01:00.0), or in either case.
    """
    parts = text.strip().lower().split(":")
    if len(parts) == 2:
        domain, bus, rest = "0", parts[0], parts[1]
    elif len(parts) == 3:
        domain, bus, rest = parts
    else:
        return None
    device, dot, function = rest.partition(".")
    if not dot:
        return None
    try:
        return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{int(device, 16):02x}.{int(function, 16):x}"
    except ValueError:
        return None


def _bus_id(raw: Any) -> str | None:
    """A reported PCI address, normalised; None for a field not reported."""
    text = _field(raw)
    if text is None:
        return None
    return _pci_address(text) or text.lower()


def _selector_key(text: Any) -> str:
    """A card name as written in the configuration, normalised for matching.

    A PCI address, with or without the ``pci:`` prefix, in any case or
    domain width, becomes ``pci:<address>``; ids and UUIDs stay as written.
    """
    raw = str(text).strip()
    if raw.lower().startswith("pci:"):
        address = _pci_address(raw[4:])
        return f"pci:{address}" if address else raw.lower()
    address = _pci_address(raw)
    return f"pci:{address}" if address else raw


def _card_keys(device: GpuDevice) -> set[str]:
    keys = {device.id}
    if device.uuid:
        keys.add(device.uuid)
    if device.bus_id:
        keys.add(f"pci:{device.bus_id}")
    return keys


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _bytes_as_mib(path: Path) -> float | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value / _BYTES_PER_MIB if math.isfinite(value) and value >= 0.0 else None


def _slot(device_dir: Path) -> str | None:
    text = _read_text(device_dir / "uevent")
    if text is None:
        return None
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "PCI_SLOT_NAME":
            return _bus_id(value)
    return None


def _overridden(device: GpuDevice, overrides: dict[str, str]) -> GpuDevice:
    """``device`` with the kind an override names it by, if any.

    ``overrides`` are keyed by ``_selector_key``.
    """
    for key in (device.id, device.uuid, f"pci:{device.bus_id}" if device.bus_id else None):
        if key and key in overrides:
            return replace(device, kind=overrides[key])
    return device


def _nvidia_cards(rows: list[list[str]], overrides: dict[str, str]) -> list[tuple[GpuDevice, float | None]]:
    """(device, used MiB) for every row of NVIDIA_QUERY with an index."""
    found = []
    for row in rows:
        if len(row) < 6:
            continue
        index_text = _field(row[0])
        if index_text is None or not index_text.isdigit():
            continue
        index = int(index_text)
        device = GpuDevice(
            id=f"nvidia:{index}",
            vendor="nvidia",
            name=_field(row[2]) or "NVIDIA card",
            kind="discrete",
            total_mib=_mib(row[4]),
            bus_id=_bus_id(row[3]),
            uuid=_field(row[1]),
            index=index,
            source="nvidia-smi",
        )
        found.append((_overridden(device, overrides), _mib(row[5])))
    return found


def _drm_cards(root: str | Path, integrated_below_mib: float, overrides: dict[str, str]) -> list[tuple[GpuDevice, float | None]]:
    """(device, used MiB) for every ``cardN`` entry of a DRM sysfs tree.

    Connector entries (``card0-DP-1``) and render nodes are not cards. AMD
    cards report their own memory; NVIDIA cards are left to nvidia-smi and
    Intel cards are integrated, and neither has its memory read here.
    """
    base = Path(root)
    try:
        entries = sorted(p for p in base.iterdir() if _CARD_ENTRY.match(p.name))
    except OSError:
        return []
    found = []
    for entry in entries:
        device_dir = entry / "device"
        vendor_text = _read_text(device_dir / "vendor")
        if vendor_text is None:
            continue
        vendor = _PCI_VENDORS.get(vendor_text.strip().lower(), "unknown")
        model = (_read_text(device_dir / "device") or "").strip()
        bus = _slot(device_dir)
        total = used = None
        if vendor == "amd":
            total = _bytes_as_mib(device_dir / "mem_info_vram_total")
            used = _bytes_as_mib(device_dir / "mem_info_vram_used")
            if total is None:
                kind = "unknown"
            else:
                kind = "integrated" if total < integrated_below_mib else "discrete"
        elif vendor == "intel":
            kind = "integrated"
        elif vendor == "nvidia":
            kind = "discrete"
        else:
            kind = "unknown"
        device = GpuDevice(
            id=f"pci:{bus}" if bus else f"drm:{entry.name}",
            vendor=vendor,
            name=f"{vendor} {model}".strip(),
            kind=kind,
            total_mib=total,
            bus_id=bus,
            source="sysfs",
        )
        found.append((_overridden(device, overrides), used))
    return found


def _pci_display_vendors(root: str | Path) -> list[str] | None:
    """The vendor of every display controller (PCI base class 0x03) the bus
    lists, as _PCI_VENDORS names it, "unknown" for any other; None when the
    bus, or a device's class, cannot be read, which proves nothing."""
    base = Path(root)
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return None
    vendors = []
    for entry in entries:
        text = _read_text(entry / "class")
        if text is None:
            return None
        try:
            code = int(text.strip(), 16)
        except ValueError:
            return None
        if code >> 16 != 0x03:
            continue
        vendor = (_read_text(entry / "vendor") or "").strip().lower()
        vendors.append(_PCI_VENDORS.get(vendor, "unknown"))
    return vendors


def _default_nvidia_query(query: str, timeout: float) -> list[list[str]] | None:
    """nvidia-smi's rows through the live-metrics collector, or None."""
    try:
        from opti_oignon.live_metrics import nvidia_smi_rows
    except Exception as exc:
        logger.debug("nvidia-smi reader unavailable: %s", exc)
        return None
    return nvidia_smi_rows(query, timeout)


def _default_spawn(target: Callable[[], None]) -> None:
    threading.Thread(target=target, name="oo-hardware-refresh", daemon=True).start()


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------


class HardwareProfile:
    """The cards of this machine, read once and refreshed off the caller's path."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        config: ProfileConfig | None = None,
        drm_root: str | Path = "/sys/class/drm",
        pressure_root: str | Path = "/proc/pressure",
        nvidia_query: Any = _UNSET,
        environ: Any = None,
        clock: Callable[[], float] = time.monotonic,
        spawn: Callable[[Callable[[], None]], None] | None = None,
        cgroup_root: str | Path = "/sys/fs/cgroup",
        self_cgroup: str | Path = "/proc/self/cgroup",
        pci_root: str | Path = "/sys/bus/pci/devices",
    ):
        self._config = config if config is not None else load_config(config_path)
        self._drm_root = drm_root
        self._pci_root = pci_root
        # The vendors of the display controllers on the bus, read with the
        # cards; None until read, or when the bus cannot be read.
        self._pci_vendors: list[str] | None = None
        self._pressure_root = pressure_root
        self._cgroup_root = cgroup_root
        self._self_cgroup = self_cgroup
        self._nvidia_query = _default_nvidia_query if nvidia_query is _UNSET else nvidia_query
        self._environ = os.environ if environ is None else environ
        self._clock = clock
        self._spawn = spawn if spawn is not None else _default_spawn
        self._lock = threading.Lock()
        self._first_read = threading.Lock()
        self._devices: list[GpuDevice] | None = None
        # Card id -> (used MiB or None, the clock reading it was taken at,
        # the generation of the engines' holdings it was read in).
        self._used: dict[str, tuple[float | None, float, int]] = {}
        self._generation = 0
        self._last_attempt: float | None = None
        self._refreshing = False
        self._again = False
        self._warned: set[str] = set()

    @property
    def config(self) -> ProfileConfig:
        return self._config

    # -- reading ---------------------------------------------------------------

    def _ask_nvidia(self) -> list[list[str]] | None:
        if self._nvidia_query is None:
            return None
        try:
            rows = self._nvidia_query(NVIDIA_QUERY, self._config.nvidia_smi_timeout_s)
        except Exception as exc:
            logger.debug("nvidia-smi query failed: %s", exc)
            return None
        if rows is None:
            return None
        return [list(row) for row in rows if isinstance(row, (list, tuple))]

    def _read(self) -> None:
        """Ask every source once and fold the answers into the state.

        The figures are stamped when the answers are in, with the
        generation of the engines' holdings the read started in: a change
        announced while the read ran leaves them not current.
        """
        with self._lock:
            generation = self._generation
        overrides = {_selector_key(k): v for k, v in self._config.kind_overrides.items()}
        rows = self._ask_nvidia()
        drm = _drm_cards(self._drm_root, self._config.integrated_below_gb * 1024.0, overrides)
        pci = _pci_display_vendors(self._pci_root)
        now = self._clock()
        with self._lock:
            self._pci_vendors = pci
            previous = list(self._devices or [])
            if rows is not None:
                answered = _nvidia_cards(rows, overrides)
                nvidia = [device for device, _ in answered]
                for device, used in answered:
                    self._used[device.id] = (used, now, generation)
            else:
                # The tool did not answer: the cards it listed last are kept,
                # with figures that grow older rather than vanish.
                nvidia = [d for d in previous if d.source == "nvidia-smi"]
            taken = {d.bus_id for d in nvidia if d.bus_id}
            others = []
            for device, used in drm:
                if device.bus_id and device.bus_id in taken:
                    continue
                others.append(device)
                self._used[device.id] = (used, now, generation)
            self._devices = nvidia + others
            listed = {d.id for d in self._devices}
            for key in [k for k in self._used if k not in listed]:
                del self._used[key]

    def _ensure_read(self) -> None:
        if self._devices is not None:
            return
        with self._first_read:
            if self._devices is None:
                self._read()
                # Stamped once the answer is in: a first reading slower than
                # the TTL must not start a second one at once.
                with self._lock:
                    self._last_attempt = self._clock()

    def _maybe_refresh(self) -> None:
        now = self._clock()
        with self._lock:
            if self._refreshing:
                return
            if self._last_attempt is not None and now - self._last_attempt <= self._config.vram_used_ttl_s:
                return
            self._refreshing = True
            self._last_attempt = now
        self._start_refresh()

    def _start_refresh(self) -> None:
        try:
            self._spawn(self._background_refresh)
        except Exception as exc:
            logger.debug("hardware refresh could not start: %s", exc)
            with self._lock:
                self._refreshing = False
                self._again = False

    def _background_refresh(self) -> None:
        try:
            self._read()
        except Exception as exc:
            logger.debug("hardware refresh failed: %s", exc)
        finally:
            with self._lock:
                again = self._again
                self._again = False
                if again:
                    self._last_attempt = self._clock()
                else:
                    self._refreshing = False
            if again:
                self._start_refresh()

    def invalidate_used(self) -> None:
        """The engines' holdings just changed: read the cards again.

        Figures read until now no longer pair with what the engines declare
        (``Placement.used_current`` turns False for them), and a fresh
        reading starts now, or right after the one in flight. A refresh
        that never returns is not doubled: its figures age out instead.
        """
        with self._lock:
            if self._devices is None:
                return
            self._generation += 1
            if self._refreshing:
                self._again = True
                return
            self._refreshing = True
            self._last_attempt = self._clock()
        self._start_refresh()

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        with self._lock:
            if key in self._warned:
                return
            self._warned.add(key)
        logger.warning(message, *args)

    # -- selection -------------------------------------------------------------

    def _cuda_visible(self, devices: list[GpuDevice]) -> list[GpuDevice] | None:
        """The NVIDIA cards CUDA_VISIBLE_DEVICES leaves visible, or None if unset.

        The list stops at its first entry that names no card, as CUDA reads
        it. A UUID, or a prefix of one that names a single card, needs no
        order. A number is an nvidia-smi index only where CUDA counts the
        same way: CUDA_DEVICE_ORDER=PCI_BUS_ID, or cards that are all the
        same model; otherwise CUDA puts the fastest first, which cannot be
        read here, and the number names no card.
        """
        raw = self._environ.get("CUDA_VISIBLE_DEVICES")
        if raw is None:
            return None
        nvidia = [d for d in devices if d.vendor == "nvidia"]
        ordered = (
            self._environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
            or len({d.name for d in nvidia}) <= 1
        )
        visible: list[GpuDevice] = []
        for token in (t.strip() for t in str(raw).split(",")):
            match = None
            if token.isdigit():
                if ordered:
                    match = next((d for d in nvidia if d.index == int(token)), None)
                else:
                    self._warn_once(
                        f"cuda-order:{raw}",
                        "CUDA_VISIBLE_DEVICES %r names cards by number, and CUDA orders"
                        " different cards fastest first: set CUDA_DEVICE_ORDER=PCI_BUS_ID"
                        " or name them by UUID; no NVIDIA card is counted until then",
                        raw,
                    )
            elif token.startswith(("GPU-", "MIG-")):
                found = [d for d in nvidia if d.uuid and d.uuid.startswith(token)]
                match = found[0] if len(found) == 1 else None
            if match is None:
                break
            if match not in visible:
                visible.append(match)
        return visible

    def _select(self, devices: list[GpuDevice]) -> list[GpuDevice]:
        choice = self._config.devices
        if isinstance(choice, list):
            chosen: list[GpuDevice] = []
            for selector in choice:
                key = _selector_key(selector)
                named = [d for d in devices if key in _card_keys(d) and d.total_mib is not None]
                if not named:
                    self._warn_once(
                        f"selector:{selector}",
                        "devices entry %r names no card with a known total; it selects nothing",
                        selector,
                    )
                for device in named:
                    if device not in chosen:
                        chosen.append(device)
            return chosen
        visible = self._cuda_visible(devices)
        candidates = [
            d
            for d in devices
            if d.kind == "discrete"
            and d.total_mib is not None
            and (d.vendor != "nvidia" or visible is None or d in visible)
        ]
        for vendor in _VENDOR_ORDER:
            group = [d for d in candidates if d.vendor == vendor]
            if group:
                return group
        return []

    # -- the questions ---------------------------------------------------------

    def devices(self) -> list[GpuDevice]:
        """Every card found, selected or not."""
        self._ensure_read()
        with self._lock:
            return list(self._devices or [])

    def cards_absent(self) -> bool:
        """Whether this machine has no card a model could be placed on.

        True only when the PCI bus could be read and lists no display
        controller but integrated ones (Intel), and no card the profile found
        is discrete: a controller the DRM tree does not list (an NVIDIA card
        without nvidia-drm) is a card all the same, and a bus that cannot be
        read proves nothing.
        """
        self._ensure_read()
        with self._lock:
            vendors = self._pci_vendors
            devices = list(self._devices or [])
        if vendors is None or any(vendor != "intel" for vendor in vendors):
            return False
        return not any(device.kind == "discrete" for device in devices)

    def _state(self) -> tuple[list[GpuDevice], dict[str, tuple[float | None, float, int]], int]:
        """One consistent copy of the cards, their figures and the generation."""
        with self._lock:
            return list(self._devices or []), dict(self._used), self._generation

    def _placement_of(self, devices: list[GpuDevice], used: dict, generation: int) -> Placement:
        selected = self._select(devices)
        if not selected:
            return Placement()
        capacity = float(sum(d.total_mib or 0.0 for d in selected))
        figures = [used.get(d.id) for d in selected]
        if any(f is None or f[0] is None for f in figures):
            return Placement(tuple(selected), capacity, None, None)
        age = max(0.0, self._clock() - min(f[1] for f in figures))
        if age > self._config.vram_used_max_age_s:
            # Too old to describe the cards: a refresh that never returned
            # leaves them unknown rather than frozen.
            return Placement(tuple(selected), capacity, None, None)
        current = all(f[2] == generation for f in figures)
        return Placement(tuple(selected), capacity, float(sum(f[0] for f in figures)), age, current)

    def placement(self) -> Placement:
        """The selected cards, their capacity and their used memory."""
        self._ensure_read()
        self._maybe_refresh()
        return self._placement_of(*self._state())

    def pressure(self) -> dict[str, dict[str, dict[str, float]]] | None:
        """The kernel's pressure stall information, read now."""
        return read_pressure(self._pressure_root)

    def others_cpu_pressure(self) -> dict[str, Any] | None:
        """The CPU pressure other programs suffer, read now, or None.

        From the user's other cgroups when the process runs in a user's
        service tree (read_cgroup_cpu_pressure); otherwise the system-wide
        reading, which counts this process's own stalls too and so errs
        towards yielding; None when neither can be read.
        """
        reading = read_cgroup_cpu_pressure(self._cgroup_root, self._self_cgroup)
        if reading is not None:
            return reading
        some = ((read_pressure(self._pressure_root) or {}).get("cpu") or {}).get("some")
        if some is None:
            return None
        return {"source": "system", "some_avg10": some["avg10"], "cgroup": None, "count": None}

    def to_dict(self) -> dict[str, Any]:
        self._ensure_read()
        self._maybe_refresh()
        devices, used, generation = self._state()
        placement = self._placement_of(devices, used, generation)
        return {
            "devices": [d.to_dict() for d in devices],
            "selected": [d.id for d in placement.devices],
            "selection": self._config.devices if isinstance(self._config.devices, str) else list(self._config.devices),
            "capacity_gb": round(placement.capacity_mib / 1024.0, 3) if placement.capacity_mib is not None else None,
            "used_gb": round(placement.used_mib / 1024.0, 3) if placement.used_mib is not None else None,
            "used_age_s": round(placement.used_age_s, 3) if placement.used_age_s is not None else None,
            "used_current": placement.used_current,
        }


_profile: HardwareProfile | None = None
_profile_lock = threading.Lock()


def get_hardware_profile() -> HardwareProfile:
    """The process-wide profile, built at the first call."""
    global _profile
    with _profile_lock:
        if _profile is None:
            _profile = HardwareProfile()
        return _profile


def reset_hardware_profile() -> None:
    """Drop the process-wide profile; the next call reads the machine again."""
    global _profile
    with _profile_lock:
        _profile = None
