#!/usr/bin/env bash
# Launch the Opti-Oignon backend in development mode.
#
# Usage:
#   scripts/dev_backend.sh [--host HOST] [--port PORT] [--reload]
#
# Defaults: host 127.0.0.1, port 8001, no auto-reload.
# Use --reload during active development to auto-restart on code changes.
# The frontend Vite dev server expects the backend on this port (see frontend/vite.config.ts).

set -euo pipefail

HOST="127.0.0.1"
PORT="8001"
RELOAD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --reload) RELOAD="--reload"; shift ;;
        -h|--help)
            grep "^#" "$0" | head -10 | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."

echo "Starting Opti-Oignon backend on http://${HOST}:${PORT}"
echo "Frontend Vite proxy expects backend on port 8001 (see frontend/vite.config.ts)"
echo

exec python3 -m uvicorn opti_oignon.api.app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    ${RELOAD}
