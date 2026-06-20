#!/usr/bin/env bash
# Opti-Oignon -- Agent Eval Harness, one-command host entry (S230, AGT_SPEC 7.4)
#
# Runs the whole (model, task, repeat) matrix against the local Ollama
# fleet and prints the register summary. The model-in-the-loop runs are
# host territory by design: a live Ollama endpoint, the real sandbox
# backend (bwrap on the recommended native deployment), real VRAM.
#
# Usage:
#   bash scripts/run_agent_eval.sh --models qwen3:4b,qwen3:14b
#   bash scripts/run_agent_eval.sh --models qwen3:4b --suite micro --repeats 3
#   bash scripts/run_agent_eval.sh --models qwen3:4b --no-evict
#
# All arguments are forwarded verbatim to the module entry:
#   python3 -m opti_oignon.agent_eval --models a,b --suite micro [--repeats N]
#
# Exit codes (forwarded): 0 run completed, 1 run failed or cancelled,
# 2 argument or suite errors.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 not found in PATH" >&2
    exit 2
fi

exec python3 -m opti_oignon.agent_eval "$@"
