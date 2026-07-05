#!/usr/bin/env bash
# Opti-Oignon -- one-shot launcher.
#
# Starts the backend (uvicorn on 127.0.0.1:8001) and the frontend (Vite on
# 5173), waits until both accept connections, opens the app in your browser,
# and shuts both down cleanly when you press Ctrl+C or close the window.
#
# Tip: run this once from a terminal to confirm it works, then use the desktop
# icon (the .desktop file) for one-click launches.

set -uo pipefail

# --- Locate the project root, regardless of where this was launched from. ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8001"
FRONTEND_PORT="5173"
APP_URL="http://localhost:${FRONTEND_PORT}"

# --- Show a readable error and keep the window open instead of flashing shut.
fail() {
    echo
    echo "ERROR: $*" >&2
    echo
    read -rp "Press Enter to close..." _ || true
    exit 1
}

# --- Environment. When launched from a desktop icon the PATH is minimal.    ---
# --- The .desktop file runs this through an interactive shell so your conda  ---
# --- and node setup load normally. As a fallback (e.g. launched some other  ---
# --- way), try to activate a conda base where opti_oignon is installed.      ---
if ! command -v python3 >/dev/null 2>&1 || ! python3 -c "import opti_oignon" >/dev/null 2>&1; then
    for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
        if [ -f "${_conda}/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "${_conda}/etc/profile.d/conda.sh" && conda activate base || true
            break
        fi
    done
fi

# --- Preflight checks (clear messages instead of a silent failure). ---
command -v python3 >/dev/null 2>&1 \
    || fail "python3 not found on PATH."
python3 -c "import opti_oignon" >/dev/null 2>&1 \
    || fail "The 'opti_oignon' package is not importable in this environment.
Activate the environment where you ran 'pip install -e .' and try again."
command -v npm >/dev/null 2>&1 \
    || fail "npm not found on PATH (required for the frontend)."

# --- Clean shutdown: signal the whole process group on exit. ---
cleanup() {
    trap - INT TERM EXIT
    echo
    echo "Stopping Opti-Oignon..."
    kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- Wait for a TCP port to start accepting connections (bash /dev/tcp). ---
wait_port() {
    local host="$1" port="$2" label="$3" tries="${4:-60}"
    local i
    for ((i = 1; i <= tries; i++)); do
        if (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; then
            exec 3>&- 3<&-
            return 0
        fi
        sleep 1
    done
    echo "WARNING: ${label} did not come up on ${host}:${port} within ${tries}s." >&2
    return 1
}

echo "Project: $ROOT"
echo "Starting backend  (http://${BACKEND_HOST}:${BACKEND_PORT}) ..."
"$ROOT/scripts/dev_backend.sh" &

echo "Starting frontend (${APP_URL}) ..."
echo "(first launch may take a minute while the frontend installs dependencies)"
"$ROOT/scripts/dev_frontend.sh" &

echo "Waiting for services..."
wait_port "$BACKEND_HOST" "$BACKEND_PORT" "backend" 60
wait_port "127.0.0.1" "$FRONTEND_PORT" "frontend" 120

echo "Opening ${APP_URL}"
( xdg-open "$APP_URL" >/dev/null 2>&1 || true ) &

echo
echo "Opti-Oignon is running. Press Ctrl+C or close this window to stop everything."
wait
