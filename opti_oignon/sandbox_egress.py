#!/usr/bin/env python3
"""Scoped sandbox network egress: gate, provision phase, audit shapes (S213).

Bloc 4 of the Sandbox Workspace cycle (SANDBOX_WORKSPACE_SPEC section 8).
The sandbox network is the cycle's one new capability and its most
sensitive: off by default, per workspace, Daily-only, switched on only by
an explicit user action, never a config default, never model-triggerable,
and refused entirely under Bulbe.

This module is the binding-layer gate (the ``veilid/guard.py`` discipline):
the security mode is read live and per call, any failure to resolve it is
treated as Bulbe (fail-secure), and the gate is STRICTER than the Veilid
guard's bulbe-equality test -- egress is allowed only when the mode is
exactly ``daily``; an unset, unknown, or undeterminable mode is refused.
There is no parameter to bypass it and no configuration that relaxes it.

The egress mechanism shipped this bloc is the PROVISION PHASE (spec 8.4,
the preferred mode): a dedicated, network-on bwrap run that executes ONLY
dependency installation -- a hash-pinned requirements set installed with
``--require-hashes --only-binary=:all:`` into a workspace venv -- after
which the network is off again BY CONSTRUCTION: only the provision run
uses the network-on argv; every task run keeps ``--unshare-net``
unconditionally. Arbitrary task code never touches the network.

The PROXY ALLOWLIST mode (spec 8.4, the advanced option: slirp4netns or
pasta giving the namespace userspace connectivity, all egress forced
through a filtering CONNECT proxy, direct egress dropped, every request
logged) is PREPARED AND LABELLED here, not wired: it adds host-only
dependencies the container cannot exercise, so shipping routes against it
would be simulation. ``proxy_mode_available()`` reports whether the
userspace-network helpers exist on the host; the design contract stays in
the spec. Raw ``--share-net`` is permanently out of scope (spec 14).

No singleton lives here: the gate is stateless and read per call, so no
``reset_*`` is needed. Heavy imports are lazy and guarded; the module is
importlib-isolatable.
"""

from __future__ import annotations

import logging
import re
import shutil

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True


# Error hierarchy


class SandboxEgressError(RuntimeError):
    """Base for every controlled sandbox-egress failure."""


class SandboxNetworkDisabledInBulbe(SandboxEgressError):
    """Sandbox network egress was requested under Bulbe, where it is refused.

    Raised by :func:`assert_network_allowed` at the binding layer; the API
    route translates it into the same 403 the security-mode middleware
    returns for any Bulbe-blocked capability.
    """


class ProvisionValidationError(SandboxEgressError):
    """A provision request failed validation (paths or requirements set)."""


# The binding-layer gate


def current_mode() -> str:
    """The live security mode, fail-secure to ``bulbe`` when undeterminable.

    The import is lazy and per call so the gate always reflects the current
    mode and so this module collects without the backend; any failure to
    resolve the mode is treated as Bulbe (fail-secure), which refuses
    egress. Mirrors ``veilid/guard.current_mode``.
    """
    try:
        from opti_oignon.security_mode import get_current_mode

        return get_current_mode()
    except Exception:
        logger.warning(
            "Cannot determine security mode; treating as 'bulbe' (fail-secure)."
        )
        return "bulbe"


def network_allowed() -> bool:
    """True only when the mode is exactly ``daily``.

    Deliberately STRICTER than an is-bulbe test: an unset or unknown mode
    string is refused, per spec 8.3 ("an unset or unknown mode is treated
    as Bulbe").
    """
    return current_mode() == "daily"


def assert_network_allowed() -> None:
    """Raise :class:`SandboxNetworkDisabledInBulbe` unless egress is allowed.

    Called at the binding layer: before the network flag may be turned on
    and before every egress while it is on. There is no bypass parameter;
    the mode is read live and fail-secure.
    """
    if not network_allowed():
        raise SandboxNetworkDisabledInBulbe(
            "Sandbox network egress is disabled in Bulbe mode: the workspace "
            "network is a Daily-only, user-activated capability. Switch to "
            "Daily mode to enable it."
        )


# Workspace-relative path validation (the apply-writer discipline: these
# are client-adjacent inputs, not trusted keys)

_REL_PATH_MAX_LEN = 1024


def refuse_rel_path(path: str) -> str | None:
    """Refusal reason for a workspace-relative request path, else None.

    Same rules as the S212 apply writer's request-path validator: empty,
    ".", absolute, NUL, over-long, or any ``..`` segment after
    normalization is refused. Backslashes are refused outright rather than
    interpreted.
    """
    if not isinstance(path, str) or not path:
        return "empty path"
    if len(path) > _REL_PATH_MAX_LEN:
        return "path too long"
    if "\x00" in path:
        return "NUL byte in path"
    if "\\" in path:
        return "backslash in path"
    if path.startswith("/"):
        return "absolute path"
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if not parts:
        return "empty path"
    if any(p == ".." for p in parts):
        return "path traversal segment"
    return None


# Requirements validation (the supply-chain half of the provision phase)

# One exact, hash-pinned requirement: name[extras]==version followed by one
# or more --hash=sha256:<64 hex> options. Nothing else is accepted: no
# ranges, no URLs, no editable installs, no option lines -- an option line
# (--index-url, --find-links, --trusted-host, -e, -r ...) changes the
# supply chain and is refused per line, honestly.
_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?P<extras>\[[A-Za-z0-9._,\s-]+\])?"
    r"==(?P<version>[A-Za-z0-9._!+*-]+)"
    r"(?P<hashes>(?:\s+--hash=sha256:[0-9a-fA-F]{64})+)\s*$"
)

_REQUIREMENTS_MAX_BYTES = 262144
_REQUIREMENTS_MAX_LINES = 2000


def validate_requirements_text(text: str) -> tuple[list[str], list[dict]]:
    """Validate a requirements set as exact, hash-pinned, option-free.

    Logical lines are assembled first (trailing-backslash continuations,
    the layout ``pip hash`` emits); blank lines and ``#`` comments are
    ignored. Every remaining logical line must be a single
    ``name==version --hash=sha256:...`` requirement. Returns
    ``(accepted_names, refused)`` where each refusal carries the 1-based
    line number of the logical line's start, the offending text (bounded)
    and the reason. NOTHING is installed when ``refused`` is non-empty:
    the provision route refuses the whole set -- a partially-validated
    install would be a partially-open supply chain.
    """
    if len(text.encode("utf-8", errors="replace")) > _REQUIREMENTS_MAX_BYTES:
        return [], [{
            "line": 0,
            "text": "",
            "reason": "requirements file exceeds the size bound",
        }]

    raw_lines = text.splitlines()
    if len(raw_lines) > _REQUIREMENTS_MAX_LINES:
        return [], [{
            "line": 0,
            "text": "",
            "reason": "requirements file exceeds the line bound",
        }]

    # Assemble logical lines: a trailing backslash continues onto the next
    # physical line. Comments and blanks end (and never join) a logical
    # line.
    logical: list[tuple[int, str]] = []
    buf = ""
    buf_start = 0
    for idx, raw in enumerate(raw_lines, start=1):
        stripped = raw.strip()
        if buf:
            if stripped.endswith("\\"):
                buf += " " + stripped[:-1].strip()
                continue
            buf += " " + stripped
            logical.append((buf_start, buf.strip()))
            buf = ""
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buf = stripped[:-1].strip()
            buf_start = idx
            continue
        logical.append((idx, stripped))
    if buf:
        logical.append((buf_start, buf.strip()))

    accepted: list[str] = []
    refused: list[dict] = []
    for line_no, content in logical:
        if "\x00" in content:
            refused.append({
                "line": line_no,
                "text": content[:120],
                "reason": "NUL byte",
            })
            continue
        if content.startswith("-"):
            refused.append({
                "line": line_no,
                "text": content[:120],
                "reason": (
                    "pip option lines are refused: they change the supply "
                    "chain (use exact name==version --hash pins only)"
                ),
            })
            continue
        match = _REQUIREMENT_RE.match(content)
        if not match:
            refused.append({
                "line": line_no,
                "text": content[:120],
                "reason": (
                    "not an exact hash-pinned requirement "
                    "(name==version --hash=sha256:... required)"
                ),
            })
            continue
        accepted.append(match.group("name"))
    return accepted, refused


# The provision command (fixed shape, built server-side only)

_VENV_DIR_DEFAULT = ".venv"


def build_provision_command(requirements_rel: str, venv_rel: str = _VENV_DIR_DEFAULT) -> str:
    """Build the fixed provision command for a validated requirements set.

    The shape is the spec 8.4 contract verbatim: create (or clear) a
    workspace venv, then install the hash-pinned set with
    ``--require-hashes`` (every requirement must be ==-pinned and carry
    hashes; pip aborts otherwise) and ``--only-binary=:all:`` (no sdists,
    so no setup/build hooks execute at install time), with no cache, no
    prompts, and no dependency resolution outside the pinned set. Both
    paths are workspace-relative and must have passed
    :func:`refuse_rel_path`; the command never embeds caller text beyond
    those validated relative paths.

    Raises:
        ProvisionValidationError: if either path fails validation.
    """
    for label, rel in (("requirements_path", requirements_rel), ("venv_dir", venv_rel)):
        reason = refuse_rel_path(rel)
        if reason is not None:
            raise ProvisionValidationError(f"{label}: {reason}")
    req = f"/workspace/{requirements_rel}"
    venv = f"/workspace/{venv_rel}"
    return (
        f"python3 -m venv --clear '{venv}' && "
        f"'{venv}/bin/python' -m pip install "
        f"--require-hashes --only-binary=:all: "
        f"--no-cache-dir --no-input --disable-pip-version-check "
        f"-r '{req}'"
    )


# Proxy allowlist mode: PREPARED AND LABELLED, not wired (spec 8.4, the
# advanced option). The design contract: the workspace's network namespace
# gets userspace connectivity (slirp4netns or pasta, both rootless), all
# egress is forced through a filtering HTTP/HTTPS CONNECT proxy whose
# allowlist is user-defined, HTTP_PROXY/HTTPS_PROXY are set in the sandbox
# env, direct egress to anything but the proxy is dropped, and the proxy
# logs every request for the audit. The helpers and the proxy itself are
# HOST-ONLY dependencies; nothing here is reachable from any route or tool
# surface, and the container deliberately ships no simulated proxy.

_PROXY_HELPERS = ("pasta", "slirp4netns")


def proxy_mode_available() -> bool:
    """True when a userspace-network helper for the proxy mode is present.

    Detection only: reports whether ``pasta`` or ``slirp4netns`` exists on
    PATH. Shipping the mode (the namespace wiring and the filtering proxy)
    is host territory and stays out of this bloc's deliverable.
    """
    try:
        return any(shutil.which(helper) for helper in _PROXY_HELPERS)
    except Exception:  # pragma: no cover - which() is defensive here
        return False
