#!/usr/bin/env python3
"""fetch_veilid_server.py -- fetch, verify, and stage the headless veilid-server.

Veilid sync (Theme 4) talks to a running headless ``veilid-server``. This script
stages that binary for the current platform: it downloads the pinned release
artifact, verifies it against a pinned SHA-256 (and, optionally, a signature),
and only then places it under the app's data directory with an exec bit. It is
an explicit, auditable, human-invoked step; it does no network at import and
never runs the binary it stages.

Usage:
    # Show what would be fetched, without touching the network:
    python scripts/fetch_veilid_server.py --print-plan

    # Stage the binary for this platform (pins must be filled in first):
    python scripts/fetch_veilid_server.py --sha <published-sha256>

    # Verify a binary already on disk against the pinned / given checksum:
    python scripts/fetch_veilid_server.py --verify /path/to/veilid-server --sha <sha256>

Security trade-offs (Kerckhoffs: security from pinned values and correct checks,
not from secrecy -- this script is open source):
  - The expected SHA-256 is pinned per artifact. The pinned slots ship empty on
    purpose: the deployer must paste the checksum Veilid publishes for the
    release, verified out of band. With no pinned (or given) checksum the script
    refuses to stage -- it never downloads-and-trusts an unverified binary.
  - Download is HTTPS-only; any other scheme is refused.
  - The checksum is verified on a temp file; on any mismatch the temp file is
    removed and nothing is staged. The exec bit is set only after verification.
  - Signature verification is a layered, optional step: when a sidecar signature
    and a pinned minisign public key and the ``minisign`` tool are all present,
    it is enforced; otherwise it is reported as skipped and the pinned checksum
    stands. Pass --require-signature to make a missing/failed signature fatal.
  - Staging is a Daily-mode setup step: sync itself is Daily-only, refused under
    Bulbe at the binding layer (see opti_oignon/veilid/guard.py). This script
    only places a file; it does not start the node or open any socket.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import platform
import shutil
import stat
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("fetch_veilid_server")

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The release this script pins. Bump together with the checksums below.
VEILID_SERVER_VERSION = "0.4.7"

# Official release host. Open and auditable; the trust anchor is the pinned
# checksum (and optional signature), not this URL.
RELEASE_BASE_URL = (
    "https://gitlab.com/veilid/veilid/-/releases"
    f"/v{VEILID_SERVER_VERSION}/downloads"
)

# Minisign public key Veilid publishes for its releases. Empty by default: the
# deployer pins the real key to enable signature enforcement. This is a public
# key, not a secret; it is here to be verified, in the open.
VEILID_MINISIGN_PUBKEY = ""

# The staged binary's name on disk.
BINARY_NAME = "veilid-server"


# Errors


class PackagingError(RuntimeError):
    """Base for any failure staging the veilid-server binary."""


class UnsupportedPlatform(PackagingError):
    """No release artifact is mapped for the current platform."""


class ChecksumUnset(PackagingError):
    """No pinned or supplied SHA-256 to verify against; staging is refused."""


class ChecksumMismatch(PackagingError):
    """The downloaded artifact did not match the expected SHA-256."""


class InsecureURL(PackagingError):
    """A non-HTTPS download URL was refused."""


class SignatureError(PackagingError):
    """Signature verification was required but did not succeed."""


@dataclass(frozen=True)
class ReleaseArtifact:
    """One platform's release artifact: where it is and what it must hash to."""

    platform_key: str
    url: str
    sha256: str  # pinned expected checksum; "" means unset -> refuse to stage
    filename: str = BINARY_NAME


def _artifact(platform_key: str, archive: str, sha256: str = "") -> ReleaseArtifact:
    return ReleaseArtifact(
        platform_key=platform_key,
        url=f"{RELEASE_BASE_URL}/{archive}",
        sha256=sha256,
        filename=BINARY_NAME,
    )


# Per-platform artifacts. The sha256 slots are intentionally empty: pin the
# published checksum (out-of-band verified) before staging on that platform.
ARTIFACTS: dict[str, ReleaseArtifact] = {
    "linux-x86_64": _artifact("linux-x86_64", "veilid-server-linux-x86_64", ""),
    "linux-aarch64": _artifact("linux-aarch64", "veilid-server-linux-aarch64", ""),
    "macos-x86_64": _artifact("macos-x86_64", "veilid-server-macos-x86_64", ""),
    "macos-aarch64": _artifact("macos-aarch64", "veilid-server-macos-aarch64", ""),
}


# Platform detection (pure; no network)


def detect_platform() -> str:
    """Map the host to an artifact key, or raise UnsupportedPlatform."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }.get(machine)
    osname = {"linux": "linux", "darwin": "macos"}.get(system)
    if osname is None or arch is None:
        raise UnsupportedPlatform(f"unsupported platform: {system}/{machine}")
    return f"{osname}-{arch}"


def artifact_for(platform_key: Optional[str] = None) -> ReleaseArtifact:
    """The release artifact for a platform key (default: the detected host)."""
    key = platform_key or detect_platform()
    artifact = ARTIFACTS.get(key)
    if artifact is None:
        raise UnsupportedPlatform(f"no artifact mapped for {key}")
    return artifact


# Staging location (guarded import of the app config)


def staging_dir(dest: Optional[Path] = None) -> Path:
    """Where the binary is staged: ``DATA_DIR/veilid/bin`` by default."""
    if dest is not None:
        return Path(dest)
    try:
        from opti_oignon.config import DATA_DIR

        base = Path(DATA_DIR)
    except Exception:  # pragma: no cover - constrained environments only
        base = Path.home() / ".opti-oignon" / "data"
    return base / "veilid" / "bin"


def staging_path(platform_key: Optional[str] = None, *, dest: Optional[Path] = None) -> Path:
    """The full path the binary would be staged to."""
    return staging_dir(dest) / artifact_for(platform_key).filename


# Checksum (pure; no network)


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """The SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str) -> bool:
    """Compare a file's SHA-256 to an expected hex digest (case-insensitive).

    Raises ChecksumUnset when ``expected`` is empty (nothing to verify against)
    and ChecksumMismatch when the digests differ; returns True on a match.
    """
    if not expected:
        raise ChecksumUnset("no expected SHA-256 supplied")
    actual = sha256_file(path)
    if actual.lower() != expected.strip().lower():
        raise ChecksumMismatch(
            f"checksum mismatch: expected {expected.strip().lower()}, got {actual}"
        )
    return True


# Signature (optional, layered; no network for the local verify)


def verify_signature(
    path: Path,
    *,
    signature: Optional[Path] = None,
    pubkey: Optional[str] = None,
    require: bool = False,
) -> str:
    """Best-effort minisign verification; returns a status string.

    Statuses: ``verified``, ``skipped-not-configured`` (no pinned key),
    ``skipped-no-signature`` (no sidecar file), ``skipped-no-tool`` (minisign
    absent). When ``require`` is set, any non-verified outcome raises
    SignatureError. This layers on top of the mandatory checksum; it is never
    the sole gate.
    """
    key = pubkey if pubkey is not None else VEILID_MINISIGN_PUBKEY
    sig = signature if signature is not None else path.with_suffix(path.suffix + ".minisig")

    def _maybe_fail(status: str) -> str:
        if require:
            raise SignatureError(f"signature not verified: {status}")
        logger.info("signature verification %s", status)
        return status

    if not key:
        return _maybe_fail("skipped-not-configured")
    if not Path(sig).is_file():
        return _maybe_fail("skipped-no-signature")
    if shutil.which("minisign") is None:
        return _maybe_fail("skipped-no-tool")
    import subprocess  # local-only; never at import

    try:
        subprocess.run(
            ["minisign", "-V", "-P", key, "-m", str(path), "-x", str(sig)],
            check=True,
            capture_output=True,
        )
    except Exception as exc:  # verification failed or tool errored
        raise SignatureError(f"minisign verification failed: {exc}") from exc
    return "verified"


# Download (the only network in this module)


def download(url: str, dest: Path, *, timeout: float = 60.0) -> None:
    """Stream an HTTPS URL to ``dest``. Refuses any non-HTTPS scheme.

    This is the sole network call in the module and is invoked only from
    ``stage`` / ``main``; importing the module performs no network at all.
    """
    if not url.lower().startswith("https://"):
        raise InsecureURL(f"refusing non-HTTPS download URL: {url}")
    logger.info("downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "opti-oignon-veilid-fetch"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https enforced above)
        with open(dest, "wb") as out:
            shutil.copyfileobj(response, out)


# Orchestration


def stage(
    platform_key: Optional[str] = None,
    *,
    dest: Optional[Path] = None,
    expected_sha: Optional[str] = None,
    downloader: Optional[Callable[[str, Path], None]] = None,
    require_signature: bool = False,
) -> Path:
    """Fetch, verify, and stage the binary; return the staged path.

    Fail-secure: a missing checksum (ChecksumUnset) or a mismatch
    (ChecksumMismatch) leaves nothing staged -- the temp download is removed and
    no exec bit is ever set on an unverified artifact.
    """
    artifact = artifact_for(platform_key)
    expected = expected_sha if expected_sha is not None else artifact.sha256
    if not expected:
        raise ChecksumUnset(
            f"no pinned SHA-256 for {artifact.platform_key}; refusing to stage an "
            "unverified binary. Pin the published checksum or pass --sha."
        )
    fetch = downloader if downloader is not None else download

    target_dir = staging_dir(dest)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / artifact.filename

    fd, tmp_name = tempfile.mkstemp(prefix=".veilid-server-", dir=str(target_dir))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        fetch(artifact.url, tmp)
        verify_checksum(tmp, expected)
        verify_signature(tmp, require=require_signature)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    mode = os.stat(target).st_mode
    os.chmod(target, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    logger.info("staged %s", target)
    return target


def plan(platform_key: Optional[str] = None, *, dest: Optional[Path] = None) -> dict[str, str]:
    """What a stage would do, with no network and no disk writes."""
    artifact = artifact_for(platform_key)
    return {
        "platform": artifact.platform_key,
        "version": VEILID_SERVER_VERSION,
        "url": artifact.url,
        "expected_sha256": artifact.sha256 or "(unset -- pin before staging)",
        "target": str(staging_path(platform_key, dest=dest)),
    }


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Network happens only when actually staging."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Fetch, verify, and stage veilid-server.")
    parser.add_argument("--platform", default=None, help="artifact key (default: detect)")
    parser.add_argument("--dest", default=None, help="staging directory override")
    parser.add_argument("--sha", default=None, help="expected SHA-256 override / pin")
    parser.add_argument(
        "--print-plan", action="store_true", help="show the plan; no network, no writes"
    )
    parser.add_argument("--verify", default=None, help="verify an existing binary, then exit")
    parser.add_argument(
        "--require-signature", action="store_true", help="fail on a missing/failed signature"
    )
    args = parser.parse_args(argv)

    try:
        if args.print_plan:
            for key, value in plan(args.platform, dest=_as_path(args.dest)).items():
                print(f"{key}: {value}")
            return 0
        if args.verify:
            expected = args.sha or artifact_for(args.platform).sha256
            verify_checksum(Path(args.verify), expected)
            print(f"verified: {args.verify}")
            return 0
        staged = stage(
            args.platform,
            dest=_as_path(args.dest),
            expected_sha=args.sha,
            require_signature=args.require_signature,
        )
        print(f"staged: {staged}")
        return 0
    except PackagingError as exc:
        logger.error("%s", exc)
        return 1


def _as_path(value: Optional[str]) -> Optional[Path]:
    return Path(value) if value else None


if __name__ == "__main__":
    sys.exit(main())
