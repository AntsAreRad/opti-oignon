#!/usr/bin/env python3
"""Contracts for the conversation-scoped workspace binding store.

The chat surface shows which workspace the active conversation is bound
to. That indication must be scoped to the conversation: switching
conversations clears it immediately, a late answer from a conversation
the user already left never paints the new one, and a bind or unbind
performed in the workspace panel updates the badge only when it targets
the conversation on screen. These contracts pin that store (a
dependency-free TypeScript module implementing the Svelte store
contract, executed here under Node's type stripping):

  * Contract 1 -- store contract: subscribe emits the current value
    immediately and unsubscribing stops further emissions.
  * Contract 2 -- refresh: refreshing for a conversation emits a
    loading state tied to that conversation, then the fetched binding.
  * Contract 3 -- anti-bleed race: a slow fetch for conversation A that
    resolves after the user switched to conversation B must not
    overwrite B's state (the stale response is dropped).
  * Contract 4 -- clear on switch: the instant a switch happens the
    previous binding disappears (no old workspace shown while the new
    fetch is in flight).
  * Contract 5 -- panel actions: applyBound / applyUnbound take effect
    only for the active conversation, and a bind action wins over any
    stale refresh still in flight.
  * Contract 6 -- no conversation: refreshing for null clears the
    store without calling any fetcher.
  * Contract 7 -- fetcher failure: a throwing fetcher lands on an
    honest unbound state, not a crash and not a stuck loading state.

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
_MODULE = _REPO / "frontend" / "src" / "lib" / "stores" / "workspaceBinding.ts"

_DRIVER = r"""
const { WorkspaceBindingStore } = await import(process.env.MODULE_URL);

const clause = process.argv[2];

function eq(got, want, label) {
    const g = JSON.stringify(got);
    const w = JSON.stringify(want);
    if (g !== w) {
        console.log(`FAIL ${label}: got ${g} want ${w}`);
        process.exit(1);
    }
}

function deferred() {
    let resolve;
    const promise = new Promise((r) => { resolve = r; });
    return { promise, resolve };
}

if (clause === 'subscribe') {
    const store = new WorkspaceBindingStore();
    const seen = [];
    const stop = store.subscribe((v) => seen.push(v));
    eq(seen.length, 1, 'immediate emission on subscribe');
    eq(
        seen[0],
        { conversationId: null, sessionId: null, loading: false },
        'initial value',
    );
    stop();
    await store.refreshFor('A', async () => ({ session_id: 'w1' }));
    eq(seen.length, 1, 'no emission after unsubscribe');
} else if (clause === 'refresh') {
    const store = new WorkspaceBindingStore();
    const seen = [];
    store.subscribe((v) => seen.push(v));
    await store.refreshFor('A', async () => ({ session_id: 'w1' }));
    eq(
        seen[1],
        { conversationId: 'A', sessionId: null, loading: true },
        'loading state tied to the conversation',
    );
    eq(
        store.snapshot,
        { conversationId: 'A', sessionId: 'w1', loading: false },
        'fetched binding lands',
    );
} else if (clause === 'late_response') {
    const store = new WorkspaceBindingStore();
    const slowA = deferred();
    const pA = store.refreshFor('A', () => slowA.promise);
    await store.refreshFor('B', async () => ({ session_id: 'wB' }));
    slowA.resolve({ session_id: 'wA' });
    await pA;
    eq(
        store.snapshot,
        { conversationId: 'B', sessionId: 'wB', loading: false },
        'a late response from a left conversation never paints the new one',
    );
} else if (clause === 'clear_on_switch') {
    const store = new WorkspaceBindingStore();
    await store.refreshFor('A', async () => ({ session_id: 'wA' }));
    eq(store.snapshot.sessionId, 'wA', 'A is bound before the switch');
    const pending = deferred();
    const p = store.refreshFor('B', () => pending.promise);
    eq(
        store.snapshot,
        { conversationId: 'B', sessionId: null, loading: true },
        'the old binding disappears the instant the switch happens',
    );
    pending.resolve({ session_id: null });
    await p;
} else if (clause === 'panel_actions') {
    const store = new WorkspaceBindingStore();
    await store.refreshFor('B', async () => ({ session_id: null }));
    store.applyBound('A', 'wA');
    eq(
        store.snapshot,
        { conversationId: 'B', sessionId: null, loading: false },
        'a bind for another conversation is ignored',
    );
    store.applyBound('B', 'wB');
    eq(store.snapshot.sessionId, 'wB', 'a bind for the active conversation lands');
    store.applyUnbound('A');
    eq(store.snapshot.sessionId, 'wB', 'an unbind for another conversation is ignored');
    store.applyUnbound('B');
    eq(store.snapshot.sessionId, null, 'an unbind for the active conversation lands');

    // A bind action wins over a stale refresh still in flight.
    const slow = deferred();
    const p = store.refreshFor('B', () => slow.promise);
    store.applyBound('B', 'wX');
    slow.resolve({ session_id: null });
    await p;
    eq(
        store.snapshot,
        { conversationId: 'B', sessionId: 'wX', loading: false },
        'the explicit bind is not overwritten by the stale fetch',
    );
} else if (clause === 'null_conversation') {
    const store = new WorkspaceBindingStore();
    await store.refreshFor('A', async () => ({ session_id: 'wA' }));
    let fetcherCalls = 0;
    await store.refreshFor(null, async () => { fetcherCalls += 1; return { session_id: 'never' }; });
    eq(fetcherCalls, 0, 'no fetch for a null conversation');
    eq(
        store.snapshot,
        { conversationId: null, sessionId: null, loading: false },
        'null conversation clears the store',
    );
} else if (clause === 'fetcher_failure') {
    const store = new WorkspaceBindingStore();
    await store.refreshFor('A', async () => { throw new Error('backend down'); });
    eq(
        store.snapshot,
        { conversationId: 'A', sessionId: null, loading: false },
        'a throwing fetcher lands on an honest unbound state',
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
# Contract 1 -- Svelte store contract
# ---------------------------------------------------------------------------
def test_c1_subscribe_contract():
    _run_clause("subscribe")


# ---------------------------------------------------------------------------
# Contract 2 -- refresh emits loading then the fetched binding
# ---------------------------------------------------------------------------
def test_c2_refresh_loading_then_binding():
    _run_clause("refresh")


# ---------------------------------------------------------------------------
# Contract 3 -- anti-bleed: a late response never paints the new conversation
# ---------------------------------------------------------------------------
def test_c3_late_response_is_dropped():
    _run_clause("late_response")


# ---------------------------------------------------------------------------
# Contract 4 -- the old binding disappears the instant a switch happens
# ---------------------------------------------------------------------------
def test_c4_clear_on_switch():
    _run_clause("clear_on_switch")


# ---------------------------------------------------------------------------
# Contract 5 -- panel bind/unbind actions are conversation-scoped and win
# ---------------------------------------------------------------------------
def test_c5_panel_actions_scoped_and_win_over_stale_fetch():
    _run_clause("panel_actions")


# ---------------------------------------------------------------------------
# Contract 6 -- a null conversation clears without fetching
# ---------------------------------------------------------------------------
def test_c6_null_conversation_clears():
    _run_clause("null_conversation")


# ---------------------------------------------------------------------------
# Contract 7 -- a throwing fetcher lands on an honest unbound state
# ---------------------------------------------------------------------------
def test_c7_fetcher_failure_is_honest():
    _run_clause("fetcher_failure")


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
