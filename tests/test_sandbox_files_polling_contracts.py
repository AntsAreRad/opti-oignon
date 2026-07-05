#!/usr/bin/env python3
"""Contracts for the sandbox file-list poll scheduler.

The chat-side sandbox file list polls the files API on a timer. The
scheduler decides, after every fetch outcome, whether to keep polling
and how long to wait -- it is the piece that turns a 404 into an honest
terminal state instead of an endless request spam, and that names the
difference between a workspace that was destroyed and one that never
existed. These contracts pin that state machine (a dependency-free
TypeScript module, executed here under Node's type stripping):

  * Contract 1 -- steady cadence: successful listings keep the base
    delay, with no drift across repeats.
  * Contract 2 -- never-born honesty: a 404 before any successful
    listing is 'absent' (no sandbox exists for this id) and polling
    stops (next delay is null) -- the anti-spam pin.
  * Contract 3 -- destroyed-after-life honesty: a 404 after at least
    one successful listing is 'expired', and polling stops.
  * Contract 4 -- transient failures back off exponentially from the
    base delay up to a hard cap, and a success resets the ladder.
  * Contract 5 -- a session change fully resets the machine: an
    immediate fetch is requested and the absent/expired distinction
    starts from scratch for the new id.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. Requires a Node runtime with TypeScript type
stripping (node >= 22.6); the module under test is imported directly
from the frontend source tree, no bundler and no package install.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_MODULE = _REPO / "frontend" / "src" / "lib" / "sandbox" / "filesPolling.ts"

_DRIVER = r"""
const { SandboxFilesPoller } = await import(process.env.MODULE_URL);

const clause = process.argv[2];

function eq(got, want, label) {
    const g = JSON.stringify(got);
    const w = JSON.stringify(want);
    if (g !== w) {
        console.log(`FAIL ${label}: got ${g} want ${w}`);
        process.exit(1);
    }
}

if (clause === 'steady') {
    const p = new SandboxFilesPoller({ baseDelayMs: 5000, maxDelayMs: 60000 });
    eq(p.current, 'idle', 'initial state');
    eq(p.onSuccess(), { state: 'live', nextDelayMs: 5000 }, 'first success');
    eq(p.onSuccess(), { state: 'live', nextDelayMs: 5000 }, 'second success');
    eq(p.onSuccess(), { state: 'live', nextDelayMs: 5000 }, 'third success');
    eq(p.everListed, true, 'everListed after success');
} else if (clause === 'absent') {
    const p = new SandboxFilesPoller({ baseDelayMs: 5000, maxDelayMs: 60000 });
    eq(p.everListed, false, 'everListed before any listing');
    eq(
        p.onNotFound(),
        { state: 'absent', nextDelayMs: null },
        'first-contact 404 is absent and stops polling',
    );
    eq(p.current, 'absent', 'state after first-contact 404');
} else if (clause === 'expired') {
    const p = new SandboxFilesPoller({ baseDelayMs: 5000, maxDelayMs: 60000 });
    p.onSuccess();
    eq(
        p.onNotFound(),
        { state: 'expired', nextDelayMs: null },
        '404 after a successful listing is expired and stops polling',
    );
    eq(p.current, 'expired', 'state after post-life 404');
} else if (clause === 'backoff') {
    const p = new SandboxFilesPoller({ baseDelayMs: 5000, maxDelayMs: 60000 });
    const got = [];
    for (let i = 0; i < 6; i++) {
        const d = p.onTransientError();
        eq(d.state, 'backoff', `transient state step ${i}`);
        got.push(d.nextDelayMs);
    }
    eq(got, [5000, 10000, 20000, 40000, 60000, 60000], 'capped ladder');
    eq(p.onSuccess(), { state: 'live', nextDelayMs: 5000 }, 'success resets');
    eq(
        p.onTransientError(),
        { state: 'backoff', nextDelayMs: 5000 },
        'ladder restarts from the base after a success',
    );
} else if (clause === 'session_change') {
    const p = new SandboxFilesPoller({ baseDelayMs: 5000, maxDelayMs: 60000 });
    p.onSuccess();
    p.onNotFound();
    eq(p.current, 'expired', 'terminal before the switch');
    eq(
        p.onSessionChange(),
        { state: 'idle', nextDelayMs: 0 },
        'switch requests an immediate fetch',
    );
    eq(p.everListed, false, 'the life memory is wiped by the switch');
    eq(
        p.onNotFound(),
        { state: 'absent', nextDelayMs: null },
        'a 404 on the new id is absent again, not expired',
    );
} else {
    console.log(`FAIL unknown clause: ${clause}`);
    process.exit(1);
}

console.log(`PASS ${clause}`);
"""


def _node_version() -> tuple[int, int]:
    exe = shutil.which("node")
    if exe is None:
        raise RuntimeError(
            "node is required for the frontend store contracts "
            "(>= 22.6 with TypeScript type stripping)"
        )
    out = subprocess.run(
        [exe, "--version"], capture_output=True, text=True, check=True,
    ).stdout.strip().lstrip("v")
    parts = out.split(".")
    return int(parts[0]), int(parts[1])


def _run_clause(clause: str) -> None:
    major, minor = _node_version()
    if (major, minor) < (22, 6):
        raise RuntimeError(
            f"node >= 22.6 required for type stripping (found {major}.{minor})"
        )
    if not _MODULE.exists():
        raise AssertionError(f"module under contract is absent: {_MODULE}")

    env = dict(os.environ)
    env["MODULE_URL"] = _MODULE.resolve().as_uri()
    with tempfile.TemporaryDirectory() as tmp:
        driver = Path(tmp) / "driver.mjs"
        driver.write_text(_DRIVER)
        attempts = ([], ["--experimental-strip-types"])
        proc = None
        for extra in attempts:
            proc = subprocess.run(
                ["node", *extra, str(driver), clause],
                capture_output=True, text=True, env=env,
            )
            if "ERR_UNKNOWN_FILE_EXTENSION" not in (proc.stderr or ""):
                break
        assert proc is not None
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 or f"PASS {clause}" not in proc.stdout:
            raise AssertionError(
                f"clause {clause} failed (rc={proc.returncode}):\n{out}"
            )


# ---------------------------------------------------------------------------
# Contract 1 -- steady cadence on success
# ---------------------------------------------------------------------------
def test_c1_steady_cadence_on_success():
    _run_clause("steady")


# ---------------------------------------------------------------------------
# Contract 2 -- never-born honesty: first-contact 404 is absent and stops
# ---------------------------------------------------------------------------
def test_c2_first_contact_404_is_absent_and_terminal():
    _run_clause("absent")


# ---------------------------------------------------------------------------
# Contract 3 -- destroyed-after-life honesty: 404 after life is expired
# ---------------------------------------------------------------------------
def test_c3_post_life_404_is_expired_and_terminal():
    _run_clause("expired")


# ---------------------------------------------------------------------------
# Contract 4 -- capped exponential backoff on transient failures
# ---------------------------------------------------------------------------
def test_c4_transient_backoff_capped_and_reset_by_success():
    _run_clause("backoff")


# ---------------------------------------------------------------------------
# Contract 5 -- session change resets the machine
# ---------------------------------------------------------------------------
def test_c5_session_change_resets_everything():
    _run_clause("session_change")


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
