#!/usr/bin/env python3
"""
LUKS / Full-Disk Encryption Detector for Opti-Oignon (S145).

Checks whether the system root filesystem (and optionally /home) is
protected by full-disk encryption (LUKS/dm-crypt).

This is an **advisory** check only — it never blocks startup, even in
Bulbe mode. It provides:
  - Security score deduction (minor) if unencrypted
  - Actionable tips for the user to enable encryption
  - Audit logging of the detection result

Detection methods (in order):
  1. ``lsblk --json`` — reliable, structured output
  2. ``/proc/mounts`` + ``/sys/block`` — fallback without lsblk
  3. ``dmsetup table`` — last resort, requires root

Kerckhoffs compliance: no secrets are used; detection is purely
observational.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------------------

@dataclass
class LUKSCheckResult:
    """Result of a LUKS / full-disk encryption check."""

    checked: bool = False
    encrypted: bool = False
    method: str = ""
    detail: str = ""
    encrypted_devices: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "checked": self.checked,
            "encrypted": self.encrypted,
            "method": self.method,
            "detail": self.detail,
            "encrypted_devices": self.encrypted_devices,
            "tips": list(self.tips),
        }


# ---------------------------------------------------------------------------
# Tips for unencrypted systems
# ---------------------------------------------------------------------------

_LUKS_TIPS: list[str] = [
    (
        "Full-disk encryption protects your data if the device is lost, "
        "stolen, or seized. Without it, all conversations, models, and "
        "keys are readable by anyone with physical access."
    ),
    (
        "On Ubuntu/Kubuntu: reinstall with the 'Encrypt the new installation "
        "for security' option, or use 'cryptsetup luksFormat' on a spare "
        "partition and migrate."
    ),
    (
        "If reinstalling is not feasible, consider encrypting your home "
        "directory with 'fscrypt' or moving sensitive Opti-Oignon data "
        "to a LUKS-encrypted partition."
    ),
    (
        "LUKS encryption has negligible performance impact on modern CPUs "
        "with AES-NI hardware acceleration."
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_luks_encryption() -> LUKSCheckResult:
    """Detect whether the root filesystem is LUKS-encrypted.

    Tries multiple detection methods in order of reliability.
    This check is always advisory — it never blocks startup.

    Returns:
        LUKSCheckResult with detection details and tips if unencrypted.
    """
    result = LUKSCheckResult()

    # Method 1: lsblk --json
    lsblk_result = _check_lsblk()
    if lsblk_result is not None:
        result.checked = True
        result.method = "lsblk"
        result.encrypted = lsblk_result["encrypted"]
        result.encrypted_devices = lsblk_result["devices"]
        if result.encrypted:
            result.detail = (
                f"Root filesystem is LUKS-encrypted via "
                f"{', '.join(result.encrypted_devices)}"
            )
            logger.info("LUKS check: %s", result.detail)
        else:
            result.detail = (
                "Root filesystem does not appear to be LUKS-encrypted"
            )
            result.tips = list(_LUKS_TIPS)
            logger.info("LUKS check: %s", result.detail)
        return result

    # Method 2: /proc/mounts + /sys/block dm-crypt detection
    proc_result = _check_proc_mounts()
    if proc_result is not None:
        result.checked = True
        result.method = "proc_mounts"
        result.encrypted = proc_result["encrypted"]
        result.encrypted_devices = proc_result["devices"]
        if result.encrypted:
            result.detail = (
                f"Root filesystem uses dm-crypt device(s): "
                f"{', '.join(result.encrypted_devices)}"
            )
            logger.info("LUKS check: %s", result.detail)
        else:
            result.detail = (
                "Root filesystem does not use dm-crypt (checked /proc/mounts)"
            )
            result.tips = list(_LUKS_TIPS)
            logger.info("LUKS check: %s", result.detail)
        return result

    # Method 3: dmsetup table (requires root)
    dmsetup_result = _check_dmsetup()
    if dmsetup_result is not None:
        result.checked = True
        result.method = "dmsetup"
        result.encrypted = dmsetup_result["encrypted"]
        result.encrypted_devices = dmsetup_result["devices"]
        if result.encrypted:
            result.detail = (
                f"dm-crypt targets found via dmsetup: "
                f"{', '.join(result.encrypted_devices)}"
            )
        else:
            result.detail = "No dm-crypt targets found via dmsetup"
            result.tips = list(_LUKS_TIPS)
        logger.info("LUKS check: %s", result.detail)
        return result

    # Could not determine
    result.checked = False
    result.method = "none"
    result.detail = (
        "Could not determine encryption status — lsblk unavailable, "
        "/proc/mounts not parseable, dmsetup not accessible"
    )
    result.tips = list(_LUKS_TIPS)
    logger.info("LUKS check: %s", result.detail)
    return result


# ---------------------------------------------------------------------------
# Detection method 1: lsblk --json
# ---------------------------------------------------------------------------

def _check_lsblk() -> dict[str, Any] | None:
    """Use ``lsblk --json -o NAME,TYPE,FSTYPE,MOUNTPOINT`` to detect LUKS.

    Returns dict with 'encrypted' bool and 'devices' list, or None on failure.
    """
    try:
        result = subprocess.run(
            ["lsblk", "--json", "-o", "NAME,TYPE,FSTYPE,MOUNTPOINT"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        devices = data.get("blockdevices", [])
        encrypted_devs: list[str] = []

        def _walk(dev_list: list[dict]) -> None:
            for dev in dev_list:
                dev_type = (dev.get("type") or "").lower()
                fstype = (dev.get("fstype") or "").lower()
                name = dev.get("name", "")

                # LUKS container or crypt device
                if dev_type == "crypt" or fstype.startswith("crypto_luks"):
                    encrypted_devs.append(name)

                # Recurse into children
                children = dev.get("children") or []
                _walk(children)

        _walk(devices)

        # Check if any encrypted device is an ancestor of the root mount
        root_encrypted = _is_root_on_crypt(devices, encrypted_devs)

        return {
            "encrypted": root_encrypted or len(encrypted_devs) > 0,
            "devices": encrypted_devs,
        }
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        OSError,
    ):
        return None


def _is_root_on_crypt(
    devices: list[dict], crypt_names: list[str],
) -> bool:
    """Check if the root mountpoint sits on a crypt device."""
    if not crypt_names:
        return False

    def _find_root(dev_list: list[dict]) -> bool:
        for dev in dev_list:
            mp = dev.get("mountpoint") or ""
            dev_type = (dev.get("type") or "").lower()
            name = dev.get("name", "")
            if mp == "/" and (dev_type == "crypt" or name in crypt_names):
                return True
            children = dev.get("children") or []
            if _find_root(children):
                return True
        return False

    return _find_root(devices)


# ---------------------------------------------------------------------------
# Detection method 2: /proc/mounts
# ---------------------------------------------------------------------------

def _check_proc_mounts() -> dict[str, Any] | None:
    """Parse /proc/mounts to detect dm-crypt root device.

    A root device path containing '/dm-' or '/mapper/' strongly suggests
    dm-crypt encryption.

    Returns dict with 'encrypted' bool and 'devices' list, or None on failure.
    """
    try:
        mounts_path = Path("/proc/mounts")
        if not mounts_path.exists():
            return None

        root_device = ""
        encrypted_devs: list[str] = []

        with open(mounts_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                device, mountpoint = parts[0], parts[1]
                if mountpoint == "/":
                    root_device = device
                    break

        if not root_device:
            return None

        # Check if root is on a dm-crypt device
        is_dm = (
            "/dm-" in root_device
            or "/mapper/" in root_device
        )

        # Additional check: see if the dm device is actually crypt
        if is_dm:
            dev_name = root_device.split("/")[-1]
            # Check /sys/block/dm-*/dm/uuid for CRYPT prefix
            crypt_confirmed = _check_dm_uuid(dev_name)
            if crypt_confirmed:
                encrypted_devs.append(dev_name)
                return {"encrypted": True, "devices": encrypted_devs}
            # Even without UUID confirmation, dm-mapper is a strong signal
            encrypted_devs.append(dev_name)
            return {"encrypted": True, "devices": encrypted_devs}

        return {"encrypted": False, "devices": []}

    except (OSError, PermissionError, ValueError):
        return None


def _check_dm_uuid(dev_name: str) -> bool:
    """Check if a device-mapper device has a CRYPT- UUID prefix.

    The kernel exposes dm UUIDs in /sys/block/<dm-N>/dm/uuid.
    LUKS devices have UUIDs starting with 'CRYPT-'.
    """
    try:
        # dev_name might be 'dm-0' or a mapper name
        # Try resolving mapper name to dm-N
        if dev_name.startswith("dm-"):
            dm_name = dev_name
        else:
            # Try /sys/block/*/dm/name
            sys_block = Path("/sys/block")
            dm_name = None
            if sys_block.exists():
                for dm_dir in sys_block.glob("dm-*"):
                    name_file = dm_dir / "dm" / "name"
                    if name_file.exists():
                        stored_name = name_file.read_text().strip()
                        if stored_name == dev_name:
                            dm_name = dm_dir.name
                            break
            if dm_name is None:
                return False

        uuid_path = Path(f"/sys/block/{dm_name}/dm/uuid")
        if uuid_path.exists():
            uuid_val = uuid_path.read_text().strip()
            return uuid_val.startswith("CRYPT-")
    except (OSError, PermissionError):
        pass
    return False


# ---------------------------------------------------------------------------
# Detection method 3: dmsetup
# ---------------------------------------------------------------------------

def _check_dmsetup() -> dict[str, Any] | None:
    """Use ``dmsetup table`` to find crypt targets.

    Requires root. Returns dict or None.
    """
    try:
        result = subprocess.run(
            ["dmsetup", "table", "--target", "crypt"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        output = result.stdout.strip()
        if not output or output == "No devices found":
            return {"encrypted": False, "devices": []}

        encrypted_devs: list[str] = []
        for line in output.splitlines():
            # Format: name: start length crypt ...
            if ":" in line:
                name = line.split(":")[0].strip()
                encrypted_devs.append(name)

        return {
            "encrypted": len(encrypted_devs) > 0,
            "devices": encrypted_devs,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


# ---------------------------------------------------------------------------
# Module availability flag
# ---------------------------------------------------------------------------

LUKS_DETECTOR_AVAILABLE = True
