#!/usr/bin/env bash
# Opti-Oignon -- opencode side-by-side baseline (S230, AGT_SPEC 7.5)
#
# OPTIONAL, HOST-ONLY, ONE-SHOT, REFERENCE ONLY. This script runs the SAME
# micro-suite task fixtures through opencode (the pinned reference version)
# pointed at the local Ollama endpoint, in a throwaway directory per task,
# scored by the SAME checks, and stores the results with engine
# "opencode-baseline". It is a reference point for the gap question Route A
# depends on -- not a CI job, not a container path, never simulated.
#
# Attribution (AGT_SPEC Section 2; mechanisms read at the pin):
#   Repository: sst/opencode
#   Commit:     4519a1da329c1a4fc384054e7203ba7d06928205 (v1.16.2)
#   License:    MIT (Copyright (c) opencode contributors)
#   The opencode binary itself is a separate MIT-licensed work; running it
#   here uses it as an external tool and changes nothing about its license.
#
# Requirements (host): the opencode binary on PATH at the pinned version,
# a running local Ollama endpoint, python3, and this repository checkout.
#
# Usage:
#   bash scripts/agent_eval_opencode_baseline.sh --models qwen3:4b [--suite micro]
#
# Exit codes: 0 baseline completed, 1 task-level failures recorded,
# 2 environment or argument errors (including: opencode not installed).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PINNED_COMMIT="4519a1da329c1a4fc384054e7203ba7d06928205"

MODELS=""
SUITE="micro"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            MODELS="${2:-}"
            shift 2
            ;;
        --suite)
            SUITE="${2:-micro}"
            shift 2
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$MODELS" ]]; then
    echo "error: --models is required (comma-separated Ollama model names)" >&2
    exit 2
fi

# Host-only guard: this script never runs in the container path. The
# opencode binary is the hard requirement; its absence is a clean exit 2,
# never a simulation.
if ! command -v opencode >/dev/null 2>&1; then
    echo "error: opencode binary not found in PATH." >&2
    echo "This baseline is host-only and never simulated; install the" >&2
    echo "pinned version (commit $PINNED_COMMIT, v1.16.2) to run it." >&2
    exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 not found in PATH" >&2
    exit 2
fi

OPENCODE_VERSION="$(opencode --version 2>/dev/null || true)"
echo "opencode baseline: binary version: ${OPENCODE_VERSION:-unknown}"
echo "opencode baseline: pinned reference: $PINNED_COMMIT (v1.16.2, MIT)"
if [[ "$OPENCODE_VERSION" != *"1.16.2"* ]]; then
    echo "warning: binary version differs from the pinned reference;" >&2
    echo "results remain a reference point, not a like-for-like number." >&2
fi

cd "$PROJECT_ROOT"

# The Python half: materialize each task fixture into a throwaway
# directory, drive opencode non-interactively on the task prompt against
# the local Ollama endpoint, then score with the SAME checks and store
# rows with engine "opencode-baseline" in the eval results DB. Plain
# subprocess driving of an external binary on the host; nothing here
# touches the in-repo sandbox or agent surface.
exec python3 - "$MODELS" "$SUITE" <<'PYEOF'
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, ".")

from opti_oignon.agent_eval.store import EvalResultsStore
from opti_oignon.agent_eval.tasks import load_suite

models = [m.strip() for m in sys.argv[1].split(",") if m.strip()]
suite = sys.argv[2]
tasks = load_suite(suite)
store = EvalResultsStore()

run_id = f"opencode-{uuid.uuid4().hex[:12]}"
store.create_run(
    run_id,
    f"{suite} (engine: opencode-baseline)",
    models,
    1,
    governor_present=False,
    host_fingerprint="engine=opencode-baseline",
)

any_failure = False
for model in models:
    for task in tasks:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="oo-opencode-") as workdir:
            work = Path(workdir)
            for relpath, content in task.fixture.items():
                target = work / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            try:
                subprocess.run(
                    ["opencode", "run", "--model", f"ollama/{model}", task.prompt],
                    cwd=work,
                    timeout=task.timeout_s,
                    capture_output=True,
                )
                failure = "none"
            except subprocess.TimeoutExpired:
                failure = "timeout"
            except OSError as exc:
                print(f"error: opencode invocation failed: {exc}", file=sys.stderr)
                failure = "error"
            passed = False
            if failure == "none":
                codes = []
                for command in task.checks:
                    proc = subprocess.run(
                        command, shell=True, cwd=work, capture_output=True,
                        timeout=60,
                    )
                    codes.append(proc.returncode)
                    if proc.returncode != 0:
                        break
                passed = all(code == 0 for code in codes)
                failure = "none" if passed else "test_fail"
            if not passed:
                any_failure = True
            store.record_task(
                run_id,
                f"opencode-baseline/{model}",
                task.id,
                0,
                passed=passed,
                rounds=0,
                tool_calls=0,
                wall_s=round(time.monotonic() - started, 3),
                failure_class=failure,
                admitted="absent",
            )
            print(
                f"[{model}] {task.id}: "
                + ("passed" if passed else f"failed ({failure})")
            )

store.finish_run(run_id, "completed")
details = store.get_run_details(run_id)
print(json.dumps(details["summary"], indent=2, sort_keys=True))
print(f"baseline run stored: {run_id}")
sys.exit(1 if any_failure else 0)
PYEOF
