#!/usr/bin/env -S bash -p
# Opti-Oignon -- one-shot launcher.
#
# Starts the backend (uvicorn on 127.0.0.1:8001) and, once it answers, the
# frontend (Vite on 5173, looked for at localhost, where the browser goes);
# opens the app in your browser, and stops both when you press Ctrl+C or close
# the window. Each runs in a session of its own, which the terminal neither
# interrupts nor hangs up, and each is told to stop (TERM) when the launcher
# ends, however it ends: on a signal or an error, the launcher tells each to
# stop, waits 15 s at most for everything it started to end, and kills what did
# not, saying so; killed itself, it leaves each told to stop. A backend or a
# frontend that stops is said with its exit status once the other one is
# stopped, so nothing scrolls the message away. It needs Linux: util-linux's
# setsid and setpriv (2.33 or later), and a /proc of its own.
#
# Run it as it is (scripts/launch.sh, or its desktop icon): its first line runs
# it under bash -p, which reads no startup file (BASH_ENV) and takes no function
# from the environment. `bash scripts/launch.sh` would read them first, with
# whatever secrets it was handed.
#
# The platform's two secrets, OPTI_KEYFILE_PASSPHRASE and OPTI_ENCRYPTION_KEY,
# go into the backend's environment alone: the frontend, the browser and every
# other command the launcher runs start without them, those an activated conda
# base sets included, and the bash that runs the backend's script, as this
# one, reads no startup file and takes no function from the environment, so
# that no other program runs where they are. A key file wrapped under a
# passphrase (see docs/security/encryption.md) is opened by the backend: the
# launcher asks for the passphrase on its terminal before it starts anything,
# and checks it (opti_oignon/keyfile_gate.py); started with
# OPTI_KEYFILE_PASSPHRASE set, it checks that one instead of asking. A
# passphrase given for a key file that is not wrapped, or not there yet, goes
# to the backend as it was given: random-key setup wraps the key it makes under
# it.
#
# Tip: run this once from a terminal to confirm it works, then use the desktop
# icon for one-click launches: a .desktop file with Terminal=true, so that the
# launcher has a terminal to ask on, and with
#   Exec=bash -ic '<project>/scripts/launch.sh'
# so that your shell's conda and node setup load and the launcher still runs by
# its first line (`bash -i <file>` would not).

set -uo pipefail
# No option inherited from the caller exports or traces what this launcher reads, ends it on the gate's "no", or
# gives its children process groups of their own before setsid does.
set +o allexport +o errexit +o monitor +o noglob +o xtrace +o verbose

# --- take_secrets: the secrets in the environment leave it, kept unexported for ---
# --- the backend alone; run at once, and again once a conda base is activated, ---
# --- whose activation may set them. A secret already taken stays as it was. An  ---
# --- empty passphrase is none.                                                    ---
declare +x _given="" _passphrase="" _key_given="" _key=""
take_secrets() {
    if [ -z "$_given" ] && [ -n "${OPTI_KEYFILE_PASSPHRASE:-}" ]; then
        _given=given
        _passphrase="$OPTI_KEYFILE_PASSPHRASE"
    fi
    if [ -z "$_key_given" ] && [ -n "${OPTI_ENCRYPTION_KEY:-}" ]; then
        _key_given=given
        _key="$OPTI_ENCRYPTION_KEY"
    fi
    unset OPTI_KEYFILE_PASSPHRASE OPTI_ENCRYPTION_KEY
}
take_secrets

# --- Locate the project root, regardless of where this was launched from. ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8001"
FRONTEND_HOST="localhost"
FRONTEND_PORT="5173"
APP_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}"
STOP_GRACE=15
started=no
backend=""
frontend=""

# --- running PID: whether PID is still this launcher's child and has not ended (its own process: what is watched). ---
# --- alive PID: whether it is, or anything is left in the process group it leads in its own session: what a stop   ---
# --- waits for.                                                                                                       ---
running() {
    local stat
    [ -n "$1" ] && { read -r stat < "/proc/$1/stat"; } 2>/dev/null || return 1
    stat=${stat##*) }
    set -- $stat
    [ "$1" != Z ] && [ "$2" = "$$" ]
}
alive() { running "$1" || { [ -n "$1" ] && kill -0 -- "-$1" 2>/dev/null; }; }

# --- ended SECONDS PID...: wait until nothing is left of what the PIDs lead, SECONDS at most. ---
ended() {
    local left=$(( $1 * 10 )) pid busy
    shift
    while (( left > 0 )); do
        busy=no
        for pid in "$@"; do
            if alive "$pid"; then busy=yes; fi
        done
        [ "$busy" = no ] && return 0
        sleep 0.1
        left=$((left - 1))
    done
    return 1
}

# --- stop_started: tell what this launcher started to stop -- each process group it leads, never ---
# --- the launcher's own --, wait STOP_GRACE s at most, and kill what is left, saying so.        ---
stop_started() {
    [ "$started" = yes ] || return 0
    local pid
    for pid in "$frontend" "$backend"; do
        if alive "$pid"; then kill -TERM -- "-$pid" 2>/dev/null || { running "$pid" && kill -TERM "$pid"; }; fi
    done
    if ended "$STOP_GRACE" "$frontend" "$backend"; then
        started=stopped
        return 0
    fi
    if alive "$frontend"; then
        echo "The frontend did not end within ${STOP_GRACE} s of being told to: it is killed."
        kill -KILL -- "-$frontend" 2>/dev/null || { running "$frontend" && kill -KILL "$frontend"; }
    fi
    if alive "$backend"; then
        echo "The backend did not end within ${STOP_GRACE} s of being told to: it is killed."
        kill -KILL -- "-$backend" 2>/dev/null || { running "$backend" && kill -KILL "$backend"; }
    fi
    started=stopped
}

# --- Show a readable error and keep the window open instead of flashing shut. Whatever was ---
# --- started is stopped first; until the message is out, no signal and no closed pipe     ---
# --- takes the launcher; then a signal ends it.                                           ---
fail() {
    trap '' INT TERM HUP PIPE
    stop_started
    echo
    echo "ERROR: $*" >&2
    echo
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    read -rp "Press Enter to close..." _ || true
    exit 1
}

# --- Environment. Launched from a desktop icon, the PATH may be minimal and ---
# --- your shell's setup not loaded. As a fallback, activate a conda base     ---
# --- where opti_oignon is installed.                                         ---
if ! command -v python3 >/dev/null 2>&1 || ! python3 -c "import opti_oignon" >/dev/null 2>&1; then
    for _conda in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
        if [ -f "${_conda}/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "${_conda}/etc/profile.d/conda.sh" && conda activate base || true
            break
        fi
    done
    # The activation ran in this shell: whatever options it left set, none exports or traces what follows.
    set +o allexport +o errexit +o monitor +o noglob +o xtrace +o verbose
    take_secrets
fi

# --- Preflight checks (clear messages instead of a silent failure). ---
command -v python3 >/dev/null 2>&1 \
    || fail "python3 not found on PATH."
python3 -c "import opti_oignon" >/dev/null 2>&1 \
    || fail "The 'opti_oignon' package is not importable in this environment.
Activate the environment where you ran 'pip install -e .' and try again."
command -v npm >/dev/null 2>&1 \
    || fail "npm not found on PATH (required for the frontend)."
command -v setsid >/dev/null 2>&1 && command -v setpriv >/dev/null 2>&1 \
    || fail "setsid or setpriv not found on PATH (util-linux, on Linux): the launcher starts the backend and the frontend in sessions of their own, told to stop when it ends."
setpriv --pdeathsig TERM -- true >/dev/null 2>&1 \
    || fail "This setpriv cannot have what it starts told to stop when the launcher ends (--pdeathsig: util-linux 2.33 or later). Nothing was started."
_self=""
{ read -r _self < /proc/self/stat; } 2>/dev/null
[ "${_self%% *}" = "$$" ] \
    || fail "This system's /proc does not number processes as this launcher does (not Linux, or a PID namespace without a /proc of its own): it could not watch what it starts. Nothing was started."

# --- A port that already answers: Opti-Oignon running already, or another program. ---
answers() { (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null; }
for _where in "${BACKEND_HOST}:${BACKEND_PORT}" "${FRONTEND_HOST}:${FRONTEND_PORT}"; do
    answers "${_where%:*}" "${_where##*:}" \
        && fail "Something already answers on ${_where}: Opti-Oignon may be running already. Nothing was started."
done

# --- The key file: a wrapped one opens under a passphrase checked before anything starts.     ---
# --- The gate answers 0 (yes) or 3 (no); any other status is "cannot tell". -E: no PYTHON*    ---
# --- variable changes what the gate imports; PYTHONUTF8, which changes how the backend decodes ---
# --- its environment, is handed on to it so that both read the passphrase alike.              ---
gate() { python3 -E ${PYTHONUTF8:+-X "utf8=$PYTHONUTF8"} -m opti_oignon.keyfile_gate "$@"; }
opens() {  # opens PASSPHRASE: the gate's answer, the passphrase on its standard input alone, one line
    printf '%s\n' "$1" | gate opens
    return "${PIPESTATUS[1]}"
}
# --- A passphrase of several lines given to the launcher is not taken: no terminal ---
# --- could type it again, and the gate reads one line.                              ---
case "$_passphrase" in
    *$'\n'*)
        echo "The passphrase this launcher was started with holds several lines: it is not taken."
        _given=""
        _passphrase="" ;;
esac
gate wrapped
case $? in
    0) _wrapped=yes ;;
    3) _wrapped=no ;;
    *) fail "Could not tell whether the key file is wrapped under a passphrase (the line above says why). Nothing was started." ;;
esac
if [ "$_wrapped" = yes ]; then
    _opened=no
    if [ -n "$_given" ]; then
        opens "$_passphrase"
        case $? in
            0) _opened=yes ;;
            3) echo "The passphrase this launcher was started with does not open the key file." ;;
            *) fail "Could not try the passphrase on the key file (the line above says why). Nothing was started." ;;
        esac
    fi
    if [ "$_opened" = no ]; then
        [ -t 0 ] || fail "The key file is wrapped under a passphrase, and this launcher has no terminal to ask for it on: start it from a terminal. Nothing was started."
        for _try in 1 2 3; do
            IFS= read -rsp "Passphrase of the key file: " _passphrase \
                || { echo; fail "No passphrase was read. Nothing was started."; }
            echo
            opens "$_passphrase"
            case $? in
                0) _opened=yes; break ;;
                3) echo "That passphrase does not open the key file." ;;
                *) fail "Could not try the passphrase on the key file (the line above says why). Nothing was started." ;;
            esac
        done
        [ "$_opened" = yes ] || fail "Three passphrases did not open the key file. Nothing was started."
    fi
    _given=given
fi

# --- Shutdown: on a signal or on exit, stop what was started, once. While it stops, the launcher ---
# --- hears no other signal (a closing window sends its hangup more than once) and outlives a pipe ---
# --- whose reader the same signal ended. A signal that comes while a child is being started waits ---
# --- until its PID is written down: the stop then reaches it. Its output's reader gone (PIPE) is a ---
# --- signal like the others.                                                                      ---
starting=no
held=""
cleanup() {
    trap '' INT TERM HUP PIPE
    trap - EXIT
    if [ "$started" = yes ]; then
        echo
        echo "Stopping Opti-Oignon..."
        stop_started
    fi
}
on_signal() {  # on_signal STATUS
    if [ "$starting" = yes ]; then
        held=$1
        return
    fi
    cleanup
    exit "$1"
}
trap 'on_signal 129' HUP
trap 'on_signal 130' INT
trap 'on_signal 141' PIPE
trap 'on_signal 143' TERM
trap cleanup EXIT

# --- up HOST PORT LABEL PID...: 0 once HOST:PORT answers; 1 as soon as one of the PIDs has ended. ---
# --- Says every 30 s that it waits.                                                               ---
up() {
    local host="$1" port="$2" label="$3" waited=0 pid
    shift 3
    while :; do
        for pid in "$@"; do
            running "$pid" || return 1
        done
        answers "$host" "$port" && return 0
        sleep 1
        waited=$((waited + 1))
        if (( waited % 30 == 0 )); then
            echo "Still waiting for the ${label} (${waited} s)..."
        fi
    done
}

# --- The backend, run in a subshell of its own: the secrets are exported there alone, never written ---
# --- on a command line, and the subshell becomes the backend, in a session of its own, told to stop ---
# --- (TERM) when the launcher ends, reading nothing from the terminal. Its script runs under bash -p ---
# --- (this launcher's bash): no BASH_ENV read, no function nor option taken from the environment,   ---
# --- so that no program but the server runs where the secrets are.                                  ---
start_backend() {
    if [ -n "$_given" ]; then export OPTI_KEYFILE_PASSPHRASE="$_passphrase"; fi
    if [ -n "$_key_given" ]; then export OPTI_ENCRYPTION_KEY="$_key"; fi
    exec setsid setpriv --pdeathsig TERM -- "$BASH" -p "$ROOT/scripts/dev_backend.sh" < /dev/null
}

echo "Project: $ROOT"
echo "Starting backend  (http://${BACKEND_HOST}:${BACKEND_PORT}) ..."
started=yes
starting=yes
start_backend &
backend=$!
starting=no
_passphrase=""
_key=""
[ -z "$held" ] || on_signal "$held"

# --- The backend first: the frontend starts once it answers, never beside a backend that stopped. ---
echo "Waiting for the backend..."
if ! up "$BACKEND_HOST" "$BACKEND_PORT" backend "$backend"; then
    wait "$backend"
    fail "The backend stopped before it answered (exit status $?): its last lines are above. The frontend was not started."
fi

echo "Starting frontend (${APP_URL}) ..."
echo "(first launch may take a minute while the frontend installs dependencies)"
starting=yes
setsid setpriv --pdeathsig TERM -- "$ROOT/scripts/dev_frontend.sh" < /dev/null &
frontend=$!
starting=no
[ -z "$held" ] || on_signal "$held"
if ! up "$FRONTEND_HOST" "$FRONTEND_PORT" frontend "$frontend" "$backend"; then
    if ! running "$backend"; then
        wait "$backend"
        fail "The backend stopped (exit status $?) while the frontend started: its last lines are above."
    fi
    wait "$frontend"
    fail "The frontend stopped before it answered (exit status $?): its last lines are above."
fi

echo "Opening ${APP_URL}"
# The browser in a session of its own: nothing the launcher stops reaches it, nor its end.
setsid xdg-open "$APP_URL" >/dev/null 2>&1 < /dev/null &

echo
echo "Opti-Oignon is running. Press Ctrl+C or close this window to stop everything."
while running "$backend" && running "$frontend"; do
    sleep 1
done
if running "$backend"; then
    wait "$frontend"
    fail "The frontend stopped (exit status $?): its last lines are above."
fi
wait "$backend"
fail "The backend stopped (exit status $?): its last lines are above."
