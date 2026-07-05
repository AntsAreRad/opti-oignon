#!/usr/bin/env python3
"""Sandbox binding doctor: one-command diagnosis of the workspace chain.

Runs the exact server-side path a conversation uses -- imports, sandbox
manager, workspace binding store, copy-in, quick-session adoption -- and
prints a PASS/FAIL verdict per link with a remediation hint. Everything
is ephemeral: the probe workspace is created, exercised and destroyed;
no existing workspace, binding or conversation is touched.

Usage (from the repository root):
    python3 scripts/sandbox_doctor.py

Exit code 0 when every link passes, 1 otherwise.
"""

from __future__ import annotations

import io
import sys
import uuid
from pathlib import Path

# Allow direct execution from the repository root without an installed
# package: scripts/ sits one level below the root that holds opti_oignon/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PROBE_CONV = f"doctor-conv-{uuid.uuid4().hex[:8]}"
PROBE_FILE = "doctor_probe.txt"
PROBE_TEXT = b"doctor probe payload\n"

_results: list[tuple[str, bool, str]] = []


def _report(name: str, ok: bool, detail: str) -> bool:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Imports: every optional subsystem the chat path relies on.
    # ------------------------------------------------------------------
    try:
        from opti_oignon.sandbox_manager import (
            SANDBOX_AVAILABLE,
            sandbox_manager,
        )
        _report(
            "sandbox_manager import", True,
            f"SANDBOX_AVAILABLE={SANDBOX_AVAILABLE}",
        )
    except Exception as exc:
        _report("sandbox_manager import", False, repr(exc))
        sandbox_manager = None
        SANDBOX_AVAILABLE = False

    try:
        from opti_oignon.sandbox_workspace import get_workspace_bindings
        bindings = get_workspace_bindings()
        _report("workspace binding store import", True, "store constructed")
    except Exception as exc:
        _report(
            "workspace binding store import", False,
            f"{exc!r} -- the /bind endpoint answers 503 in this state",
        )
        bindings = None

    try:
        from opti_oignon.quick_sandbox import (
            QUICK_SANDBOX_AVAILABLE,
            quick_sandbox_manager,
        )
        _report(
            "quick_sandbox import", True,
            f"QUICK_SANDBOX_AVAILABLE={QUICK_SANDBOX_AVAILABLE} "
            f"enabled={getattr(quick_sandbox_manager, 'enabled', '?')} "
            f"available={getattr(quick_sandbox_manager, 'available', '?')}",
        )
    except Exception as exc:
        _report("quick_sandbox import", False, repr(exc))
        quick_sandbox_manager = None
        QUICK_SANDBOX_AVAILABLE = False

    if not SANDBOX_AVAILABLE or sandbox_manager is None:
        _report(
            "chain", False,
            "sandbox manager unavailable; nothing further can run",
        )
        return _finish()

    status = {}
    try:
        status = sandbox_manager.get_status()
    except Exception:
        pass
    if status:
        print(
            f"       manager status: backend="
            f"{status.get('isolation_backend', '?')} "
            f"active={status.get('active_sessions', '?')}/"
            f"{status.get('max_sessions', '?')}"
        )

    # ------------------------------------------------------------------
    # 2. Ephemeral workspace: create, upload, list.
    # ------------------------------------------------------------------
    probe_ws = None
    try:
        probe = sandbox_manager.create_sandbox(
            f"doctor-ws-{uuid.uuid4().hex[:8]}", allow_degraded=True,
        )
        probe_ws = probe.session_id
        _report("workspace creation", True, f"session {probe_ws}")
    except Exception as exc:
        _report("workspace creation", False, repr(exc))
        return _finish()

    try:
        result = sandbox_manager.upload_files(
            probe_ws, [(PROBE_FILE, io.BytesIO(PROBE_TEXT), len(PROBE_TEXT))],
        )
        written = [w["relative_path"] for w in result["written"]]
        refused = result["refused"]
        ok = PROBE_FILE in written and not refused
        _report(
            "copy-in (upload_files)", ok,
            f"written={written} refused={refused}",
        )
    except Exception as exc:
        _report("copy-in (upload_files)", False, repr(exc))

    try:
        listed = [e.get("path") for e in sandbox_manager.extract_files(probe_ws)]
        _report(
            "workspace listing (extract_files)", PROBE_FILE in listed,
            f"listed={listed}",
        )
    except Exception as exc:
        _report("workspace listing (extract_files)", False, repr(exc))

    # ------------------------------------------------------------------
    # 3. Binding round-trip against the real store.
    # ------------------------------------------------------------------
    if bindings is not None:
        try:
            bindings.bind(PROBE_CONV, probe_ws, manager=sandbox_manager)
            resolved = bindings.get_sandbox_for(
                PROBE_CONV, manager=sandbox_manager,
            )
            _report(
                "binding round-trip (bind -> get_sandbox_for)",
                resolved == probe_ws,
                f"resolved={resolved}",
            )
        except Exception as exc:
            _report("binding round-trip", False, repr(exc))
    else:
        _report(
            "binding round-trip", False,
            "skipped: binding store unavailable",
        )

    # ------------------------------------------------------------------
    # 4. Quick-session adoption: the bridge a conversation actually uses.
    # ------------------------------------------------------------------
    if quick_sandbox_manager is not None and QUICK_SANDBOX_AVAILABLE:
        try:
            session = quick_sandbox_manager.get_or_create_session(
                PROBE_CONV, bound_sandbox_id=probe_ws,
            )
            content = session.handle_read_file(PROBE_FILE)
            adopted = PROBE_TEXT.decode().strip() in str(content)
            _report(
                "quick adoption (bound workspace readable)", adopted,
                f"read_file -> {content!r}",
            )
            announced = PROBE_FILE in session.files_created
            _report(
                "adoption announcement (files_created)", announced,
                f"files_created={session.files_created}",
            )
            session.destroy()
            alive = sandbox_manager.get_session(probe_ws)
            _report(
                "ownership (workspace survives wrapper destroy)",
                alive is not None and alive.active,
                "workspace still active" if alive else "workspace gone",
            )
        except TypeError as exc:
            _report(
                "quick adoption", False,
                f"{exc!r} -- the binding bridge patch is not applied",
            )
        except Exception as exc:
            _report("quick adoption", False, repr(exc))
    else:
        _report("quick adoption", False, "quick sandbox unavailable")

    # ------------------------------------------------------------------
    # 5. Cleanup: unbind and destroy the probe workspace.
    # ------------------------------------------------------------------
    try:
        if bindings is not None:
            bindings.unbind(PROBE_CONV, manager=sandbox_manager)
    except Exception:
        pass
    try:
        sandbox_manager.destroy_sandbox(probe_ws)
        _report("cleanup", True, f"probe workspace {probe_ws} destroyed")
    except Exception as exc:
        _report("cleanup", False, repr(exc))

    return _finish()


def _finish() -> int:
    failed = [name for name, ok, _ in _results if not ok]
    print()
    if failed:
        print(f"DOCTOR: {len(failed)} failing link(s): {', '.join(failed)}")
        return 1
    print("DOCTOR: all links pass -- the workspace chain is healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
