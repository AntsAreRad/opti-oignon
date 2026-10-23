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
            awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# The project root, found without running a program: launched by scripts/launch.sh -- under bash -p,
# which reads no BASH_ENV and takes no function from the environment --, this script holds the
# platform's secrets in its environment, and they reach no command but the server.
case "$0" in
    */*) cd "${0%/*}/.." ;;
    *) cd .. ;;
esac

echo "Starting Opti-Oignon backend on http://${HOST}:${PORT}"
echo "Frontend Vite proxy expects backend on port 8001 (see frontend/vite.config.ts)"
echo

# Told to stop, the server waits 10 s at most for the connections still open (a browser tab's
# stream) before it closes them and runs its own shutdown: scripts/launch.sh waits 15 s for it.
exec python3 -m uvicorn opti_oignon.api.app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --timeout-graceful-shutdown 10 \
    ${RELOAD}
