#!/usr/bin/env bash
# Launch the Opti-Oignon frontend in development mode.
#
# Usage:
#   scripts/dev_frontend.sh [--port PORT]
#
# Defaults: port 5173 (Vite default).
# The Vite dev server proxies /api requests to the backend on port 8001
# (see frontend/vite.config.ts). Launch the backend first with scripts/dev_backend.sh.

set -euo pipefail

PORT="5173"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        -h|--help)
            grep "^#" "$0" | head -10 | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/../frontend"

if [ ! -d "node_modules" ]; then
    echo "node_modules missing -- running 'npm install' first"
    npm install
fi

echo "Starting Opti-Oignon frontend on http://localhost:${PORT}"
echo "Backend proxy target: http://localhost:8001 (start with scripts/dev_backend.sh)"
echo

exec npm run dev -- --port "${PORT}"
