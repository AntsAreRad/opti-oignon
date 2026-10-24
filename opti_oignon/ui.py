#!/usr/bin/env python3
"""
Opti-Oignon UI Launcher
========================

Launches the full Opti-Oignon stack:
  1. Checks prerequisites (Python, Node, Ollama), and holds the frontend's
     installed dependencies to package-lock.json (npm ci when they differ)
  2. Frees ports if occupied (with user confirmation)
  3. Asks for the key file's passphrase when it is wrapped under one, and
     checks it (``opti_oignon.keyfile_gate``) before the backend starts
  4. Starts FastAPI backend, the platform's secrets (OPTI_KEYFILE_PASSPHRASE,
     OPTI_ENCRYPTION_KEY) in its environment alone, and waits until it
     answers or stops
  5. Starts SvelteKit frontend (npm run dev), without them
  6. Opens browser, without them

Usage:
    python -m opti_oignon [ui] [--port PORT]
"""

import functools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Default ports
DEFAULT_BACKEND_PORT = 8001
DEFAULT_FRONTEND_PORT = 5173

# The platform's two secrets, the key file's passphrase and the master key itself: taken out of this process's
# environment at once, handed to the backend alone.
_PASSPHRASE_VAR = "OPTI_KEYFILE_PASSPHRASE"
_KEY_VAR = "OPTI_ENCRYPTION_KEY"
_SECRET_VARS = (_PASSPHRASE_VAR, _KEY_VAR)
_PASSPHRASE_TRIES = 3
# The key file gate's answers; any other status, Python's own failures included, is "cannot tell".
_GATE_YES, _GATE_NO = 0, 3
# How long a stopped backend or frontend may take to end before it is killed (the backend waits 10 s at most for
# its open connections: scripts/dev_backend.sh).
_STOP_GRACE = 15
# prctl's option that sets the signal a child is sent when its parent ends (linux/prctl.h).
_PR_SET_PDEATHSIG = 1

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
NC = "\033[0m"


def _find_project_root() -> Path:
    """Find the project root directory (contains opti_oignon/ and frontend/)."""
    # Check common locations
    candidates = [
        Path(__file__).resolve().parent.parent,  # from opti_oignon/ui.py -> project root
        Path.cwd(),
        Path.home() / "opti-oignon",
    ]
    for p in candidates:
        if (p / "frontend").is_dir() and (p / "opti_oignon").is_dir():
            return p
    # Fallback: parent of the package directory
    return Path(__file__).resolve().parent.parent


def _port_pids(port: int) -> list[int]:
    """Return PIDs using the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return []


def _port_info(port: int) -> str:
    """Get human-readable info about what is using a port."""
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            return "\n".join(lines[:4])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "(could not determine)"


def _free_port(port: int, label: str) -> bool:
    """Free a port by killing occupying processes (with confirmation)."""
    pids = _port_pids(port)
    if not pids:
        return True

    print(f"\n{YELLOW}[!] Port {port} ({label}) is already in use:{NC}")
    print(_port_info(port))
    print()

    try:
        answer = input(f"    Kill these processes to free port {port}? [Y/n] ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer == "" or answer.lower() == "y":
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(1)

        # Force kill if still alive
        remaining = _port_pids(port)
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.5)

        if _port_pids(port):
            print(f"{RED}    Could not free port {port}.{NC}")
            return False

        print(f"{GREEN}    Port {port} freed.{NC}")
        return True
    else:
        print(f"{RED}    Aborted. Cannot start {label} on port {port}.{NC}")
        return False


def _check_ollama() -> bool:
    """Check if Ollama is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _lock_packages(path: Path) -> dict | None:
    """The ``packages`` map of an npm lockfile, or None when it cannot be read."""
    try:
        packages = json.loads(path.read_text(encoding="utf-8")).get("packages")
    except (OSError, ValueError, AttributeError):
        return None
    return packages if isinstance(packages, dict) else None


def _package_name(path: str) -> str:
    return path.rpartition("node_modules/")[2] or path


def _same_build(pinned: dict, have: dict) -> bool:
    """True when ``have`` is the build ``pinned`` names: the same version and,
    when either side records a digest, the same digest. The same bytes
    fetched through a registry mirror carry the mirror's source, so the
    source decides only for a package pinned without a digest (a git or a
    local one)."""
    if have.get("version") != pinned.get("version"):
        return False
    if pinned.get("integrity") or have.get("integrity"):
        return have.get("integrity") == pinned.get("integrity")
    return have.get("resolved") == pinned.get("resolved")


def _skippable(pinned: object) -> bool:
    """True for a package npm may leave out: an optional or dev-optional one,
    built for another system."""
    return isinstance(pinned, dict) and (
        pinned.get("optional") is True or pinned.get("devOptional") is True
    )


def _install_differences(frontend_dir: Path) -> list[str]:
    """How ``node_modules`` differs from ``package-lock.json``; empty when it
    holds what the lock pins.

    npm records what it installed in ``node_modules/.package-lock.json`` (the
    hidden lockfile): every package of the lock, less the lock's root entry
    and the optional or dev-optional packages it skipped because they are
    built for another system. So the two files are compared package by
    package, never byte for byte, and a build is named by its version and
    digest (``_same_build``).
    """
    modules = frontend_dir / "node_modules"
    if not modules.is_dir():
        return ["node_modules is missing"]
    lock = _lock_packages(frontend_dir / "package-lock.json")
    if lock is None:
        return ["package-lock.json cannot be read"]
    installed = _lock_packages(modules / ".package-lock.json")
    if installed is None:
        return ["node_modules/.package-lock.json is missing or unreadable"]
    differences = []
    for path, pinned in lock.items():
        if not path:
            continue
        have = installed.get(path)
        if have is None:
            if not _skippable(pinned):
                differences.append(f"{_package_name(path)} is not installed")
            continue
        if not isinstance(pinned, dict) or not isinstance(have, dict) or not _same_build(
            pinned, have
        ):
            version = have.get("version") if isinstance(have, dict) else None
            wanted = pinned.get("version") if isinstance(pinned, dict) else None
            differences.append(
                f"{_package_name(path)} {version} is installed, the lock pins {wanted}"
            )
    for path in installed:
        if path and path not in lock:
            differences.append(f"{_package_name(path)} is installed and the lock no longer pins it")
    return differences


def _install_frontend(frontend_dir: Path) -> bool:
    """Hold ``frontend/node_modules`` to ``package-lock.json`` before the dev
    server starts, and say what was done.

    When the install differs from the lock, ``npm ci`` replaces it, with
    npm's output left on the terminal. Without a lock, ``npm install`` runs
    only when ``node_modules`` is missing, as before. Returns False when npm
    failed or could not be run; the caller stops.
    """
    if not (frontend_dir / "package-lock.json").is_file():
        if (frontend_dir / "node_modules").is_dir():
            return True
        command = ["npm", "install"]
        print(f"{YELLOW}[>] Installing frontend dependencies (no package-lock.json)...{NC}")
    else:
        differences = _install_differences(frontend_dir)
        if not differences:
            return True
        command = ["npm", "ci"]
        shown = "; ".join(differences[:3])
        if len(differences) > 3:
            shown += f"; and {len(differences) - 3} more"
        print(f"{YELLOW}[>] The frontend's dependencies differ from package-lock.json: {shown}.{NC}")
        print("    Running npm ci, which reinstalls node_modules from the lock...")
    try:
        result = subprocess.run(command, cwd=str(frontend_dir))
    except OSError as exc:
        print(f"{RED}[ERR] Could not run npm ({exc}). Install Node.js and npm, then retry.{NC}")
        return False
    if result.returncode != 0:
        print(f"{RED}[ERR] {' '.join(command)} failed (exit {result.returncode}); npm's output is above.{NC}")
        return False
    print(f"{GREEN}    Done.{NC}")
    return True


def _update_vite_proxy(project_root: Path, backend_port: int) -> None:
    """Update vite.config.ts proxy target to match backend port."""
    vite_config = project_root / "frontend" / "vite.config.ts"
    if not vite_config.exists():
        return
    content = vite_config.read_text()
    import re
    new_content = re.sub(
        r"target:\s*'http://localhost:\d+'",
        f"target: 'http://localhost:{backend_port}'",
        content,
    )
    if new_content != content:
        vite_config.write_text(new_content)


def _gate(root: Path, action: str, passphrase: str | None = None) -> int:
    """The key file gate's answer (``opti_oignon.keyfile_gate``) for the key file of the project at ``root``, the
    one its backend reads: 0 yes, 3 no, anything else cannot tell, its reason said here.

    Asked in this process: run as a process of its own, as ``scripts/launch.sh``
    runs it, it would be one more interpreter launched, which the egress census
    charges as a network sink, and this module owes no more than it has. The
    key the gate opens to check a passphrase is wiped where the gate holds it,
    but a copy may stay in this process's memory while it runs.
    """
    try:
        from opti_oignon import keyfile_gate
    except ImportError as exc:
        print(f"keyfile_gate: it does not import here: {exc}")
        return -1
    try:
        if action == "wrapped":
            return keyfile_gate.YES if keyfile_gate.wrapped(root) else keyfile_gate.NO
        return keyfile_gate.YES if keyfile_gate.opens(passphrase or "", root) else keyfile_gate.NO
    except keyfile_gate.GateError as exc:
        print(f"keyfile_gate: {exc}")
        return keyfile_gate.CANNOT


def _backend_passphrase(given: str | None, root: Path) -> str | None:
    """The passphrase the backend is started with, checked before the backend starts.

    ``given`` (the one this launcher was started with, or None) as it was given
    when the key file is not wrapped or not there yet: random-key setup wraps
    the key it makes under it. For a wrapped key file, ``given`` when it opens
    it; otherwise one asked on the terminal, without echo, up to three times. A
    given passphrase of several lines is not taken: no terminal could type it
    again. When none can be had -- no terminal, three that do not open it, or a
    key file the gate cannot read or try -- it says so and exits: the backend is
    not started.
    """
    if given is not None and "\n" in given:
        print(f"{YELLOW}[!] The passphrase this launcher was started with holds several lines: it is not taken.{NC}")
        given = None
    answer = _gate(root, "wrapped")
    if answer == _GATE_NO:
        return given
    if answer != _GATE_YES:
        print(f"{RED}[ERR] Could not tell whether the key file is wrapped under a passphrase (the line above says "
              f"why). Nothing was started.{NC}")
        sys.exit(1)
    if given:
        answer = _gate(root, "opens", given)
        if answer == _GATE_YES:
            return given
        if answer != _GATE_NO:
            print(f"{RED}[ERR] Could not try the passphrase on the key file (the line above says why). Nothing "
                  f"was started.{NC}")
            sys.exit(1)
        print(f"{YELLOW}[!] The passphrase this launcher was started with does not open the key file.{NC}")
    if not sys.stdin.isatty():
        print(f"{RED}[ERR] The key file is wrapped under a passphrase, and this launcher has no terminal "
              f"to ask for it on: start it from a terminal. Nothing was started.{NC}")
        sys.exit(1)
    import getpass

    for _ in range(_PASSPHRASE_TRIES):
        try:
            passphrase = getpass.getpass("Passphrase of the key file: ")
        except EOFError:
            print(f"\n{RED}[ERR] No passphrase was read. Nothing was started.{NC}")
            sys.exit(1)
        except UnicodeError:
            # A byte the terminal's encoding cannot carry: no key file was wrapped under it, and it is not shown.
            print("    That passphrase does not open the key file.")
            continue
        answer = _gate(root, "opens", passphrase)
        if answer == _GATE_YES:
            return passphrase
        if answer != _GATE_NO:
            print(f"{RED}[ERR] Could not try the passphrase on the key file (the line above says why). Nothing "
                  f"was started.{NC}")
            sys.exit(1)
        print("    That passphrase does not open the key file.")
    print(f"{RED}[ERR] {_PASSPHRASE_TRIES} passphrases did not open the key file. Nothing was started.{NC}")
    sys.exit(1)


def _child_env(secrets: dict | None = None) -> dict:
    """This process's environment for a child, without the platform's secrets but those of ``secrets`` that are
    given: the backend's, and the backend's alone."""
    env = {name: value for name, value in os.environ.items() if name not in _SECRET_VARS}
    env.update({name: value for name, value in (secrets or {}).items() if value is not None})
    return env


def _status(code) -> str:
    """A process's end, as its exit status or the signal that ended it."""
    return f"ended by signal {-code}" if code is not None and code < 0 else f"exit status {code}"


def _group_left(pgid: int) -> bool:
    """Whether anything is left in the process group ``pgid``."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _say(text: str = "") -> None:
    """Print ``text`` at once. Once this launcher's output is gone -- a closed window, a pipe whose reader ended
    --, what is left to say goes nowhere, the flush at exit included, and nothing fails: what was started is stopped
    all the same. An output in use by what a signal interrupted takes nothing more: the line is dropped."""
    try:
        print(text, flush=True)
    except RuntimeError:
        pass
    except (OSError, ValueError):
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull, sys.stdout.fileno())
            finally:
                os.close(devnull)
        except (OSError, ValueError):
            pass


def _left(proc) -> bool:
    """Whether ``proc``, or anything in the process group it leads (it runs in a session of its own), is left."""
    return proc.poll() is None or _group_left(proc.pid)


def _stop_all(started) -> None:
    """Stop each ``(label, proc)`` of ``started`` by its process group: TERM to every group first, then all awaited
    together, ``_STOP_GRACE`` s at most, then KILL to every group still holding something; each said."""
    started = [(label, proc) for label, proc in started if proc is not None and _left(proc)]
    for _label, proc in started:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            proc.terminate()
    deadline = time.monotonic() + _STOP_GRACE
    while any(_left(proc) for _label, proc in started) and time.monotonic() < deadline:
        time.sleep(0.1)
    for label, proc in started:
        if _left(proc):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                proc.kill()
            proc.wait()
            _say(f"    {label} did not end within {_STOP_GRACE} s of being told to: killed.")
        else:
            _say(f"    {label} stopped.")


def _die_with_launcher(libc, launcher: int) -> None:
    """In a child, before it runs: the kernel tells it to stop (TERM) when this launcher ends, however it ends --
    a session of its own keeps the terminal's hangup from it. Its TERM goes back to the default first: the
    launcher's handler, its own until it runs, would take that TERM and drop it. A launcher that ended already,
    before the setting held, would never tell it: the child ends at once."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
        raise OSError("the parent-death signal could not be set")
    if os.getppid() != launcher:
        os._exit(128 + signal.SIGTERM)


def _death_hook():
    """What a child runs before it starts so that it is told to stop when this launcher ends: Linux's parent-death
    signal, its library loaded here, in the launcher, so that the child loads nothing. None elsewhere, where a
    launcher killed outright leaves what it started running."""
    if not sys.platform.startswith("linux"):
        return None
    import ctypes

    return functools.partial(_die_with_launcher, ctypes.CDLL(None, use_errno=True), os.getpid())


def _wait_for_backend(port: int, proc) -> bool:
    """True once the backend answers; False as soon as its process ``proc`` ended. Says every 30 s that it waits."""
    import urllib.request
    waited = 0
    while proc.poll() is None:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/api/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
        waited += 1
        if waited % 30 == 0:
            _say(f"\n    Still waiting for the backend ({waited} s)...")
    return False


def _get_version(port: int) -> str:
    """Get version from running backend."""
    try:
        import json
        import urllib.request
        req = urllib.request.Request(f"http://localhost:{port}/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return data.get("version", "?")
    except Exception:
        return "?"


def _open_browser(url: str) -> None:
    """Open URL in default browser."""
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def launch(port: int = 8000, share: bool = False, debug: bool = False) -> None:
    """Launch the full Opti-Oignon stack.

    Args:
        port: Backend port (default overridden to 8001).
        share: Ignored (legacy Gradio parameter).
        debug: Ignored (legacy Gradio parameter).
    """
    # The secrets handed to this launcher leave its environment at once: the
    # frontend, the browser and every command run from here start without them,
    # and they are zeroed in the environment this process was started with,
    # which a process of the same user reads in /proc for as long as it runs.
    from opti_oignon.process_secrets import withdraw

    given = withdraw(_PASSPHRASE_VAR) or None
    key_given = withdraw(_KEY_VAR) or None

    # Override default port 8000 to 8001
    backend_port = port if port != 8000 else DEFAULT_BACKEND_PORT
    frontend_port = DEFAULT_FRONTEND_PORT

    project_root = _find_project_root()
    frontend_dir = project_root / "frontend"

    backend_proc = None
    frontend_proc = None
    # A signal that comes while a child is being started waits until it is written down: the stop then reaches it.
    starting = False
    held = []

    def cleanup(code=0, why=None):
        """Stop what was started, each by its process group, say ``why`` last, and exit with ``code``. While it
        stops, the launcher hears no other signal: a closing window sends its hangup more than once."""
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), signal.SIG_IGN)
        started = [(label, proc) for label, proc in (("Frontend", frontend_proc), ("Backend", backend_proc))
                   if proc is not None]
        if started:
            _say(f"\n{YELLOW}[>] Stopping Opti-Oignon...{NC}")
            _stop_all(started)
            _say(f"{GREEN}[OK] Opti-Oignon stopped.{NC}")
        else:
            _say(f"\n{YELLOW}[>] Nothing was started.{NC}")
        if why:
            _say(f"\n{why}")
        sys.exit(code)

    def on_signal(signum, frame=None):
        """An interruption, a request to end, a hangup: everything stops, and the launcher ends with 128 and the
        signal's number -- once what is being started is written down."""
        if starting:
            held.append(signum)
            return
        cleanup(code=128 + signum)

    def heard(sig):
        # One ignored when this launcher started -- under nohup, as a background job -- stays ignored, as a shell
        # keeps it.
        if signal.getsignal(sig) != signal.SIG_IGN:
            signal.signal(sig, on_signal)

    # A closing window hangs this launcher up: what it started, in sessions of their own, is not, and is stopped
    # here.
    heard(signal.SIGINT)
    heard(signal.SIGTERM)
    if hasattr(signal, "SIGHUP"):  # not on every system
        heard(signal.SIGHUP)

    # -- Banner --
    print(f"{CYAN}")
    print("  ___        _   _    ___  _                       ")
    print(" / _ \\ _ __ | |_(_)  / _ \\(_) __ _ _ __   ___  _ __")
    print("| | | | '_ \\| __| | | | | | |/ _` | '_ \\ / _ \\| '_ \\")
    print("| |_| | |_) | |_| | | |_| | | (_| | | | | (_) | | | |")
    print(" \\___/| .__/ \\__|_|  \\___/|_|\\__, |_| |_|\\___/|_| |_|")
    print("      |_|                    |___/")
    print(f"{NC}")

    # -- Prerequisites --
    print(f"{YELLOW}[>] Checking prerequisites...{NC}")

    if not _check_ollama():
        print(f"{RED}[ERR] Ollama not reachable on localhost:11434{NC}")
        print("      Start Ollama with: ollama serve")
        sys.exit(1)
    print(f"    Ollama: {GREEN}running{NC}")

    if not frontend_dir.is_dir():
        print(f"{RED}[ERR] Frontend directory not found: {frontend_dir}{NC}")
        sys.exit(1)

    # The installed dependencies are the lock's, or the launch stops here.
    if not _install_frontend(frontend_dir):
        sys.exit(1)

    # -- Free ports --
    if not _free_port(backend_port, "backend"):
        sys.exit(1)
    if not _free_port(frontend_port, "frontend"):
        sys.exit(1)

    # -- The key file: a wrapped one opens under a passphrase checked now --
    passphrase = _backend_passphrase(given, project_root)
    given = None

    # -- Update vite proxy --
    _update_vite_proxy(project_root, backend_port)

    # -- Determine bind address (route through bind guard) --
    bind_host = "127.0.0.1"
    try:
        from opti_oignon.network_bind_guard import get_safe_bind_address
        bind_host = get_safe_bind_address("127.0.0.1")
    except ImportError:
        pass  # Guard not available; keep localhost default

    # -- Start backend --
    print(f"\n{CYAN}[>] Starting backend on port {backend_port}...{NC}")
    on_death = _death_hook()
    try:
        starting = True
        backend_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "opti_oignon.api.app:app",
                "--host", bind_host,
                "--port", str(backend_port),
                "--log-level", "warning",
                "--timeout-graceful-shutdown", "10",
            ],
            cwd=str(project_root),
            env=_child_env({_PASSPHRASE_VAR: passphrase, _KEY_VAR: key_given}),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            # Its errors are on this terminal: a backend that does not start says why.
            # A session of its own: the terminal neither interrupts nor hangs it up; cleanup stops it, and on Linux
            # it is told to stop when this launcher ends, however it ends.
            start_new_session=True,
            preexec_fn=on_death,
        )
        starting = False
        passphrase = key_given = None
        if held:
            cleanup(code=128 + held[0])

        sys.stdout.write("    Waiting")
        sys.stdout.flush()
        if _wait_for_backend(backend_port, backend_proc):
            version = _get_version(backend_port)
            print(f"\n{GREEN}[OK] Backend ready (v{version}){NC}")
        else:
            _stop_all([("What the backend left", backend_proc)])
            _say(f"\n{RED}[ERR] The backend stopped before it answered ({_status(backend_proc.returncode)}): "
                 f"its last lines are above. The frontend was not started.{NC}")
            sys.exit(1)

        # -- Start frontend --
        print(f"{CYAN}[>] Starting frontend on port {frontend_port}...{NC}")
        starting = True
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(frontend_port)],
            cwd=str(frontend_dir),
            env=_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=on_death,
        )
        starting = False
        if held:
            cleanup(code=128 + held[0])
        time.sleep(3)
        print(f"{GREEN}[OK] Frontend ready{NC}")

        # -- Ready --
        browser_url = f"http://localhost:{frontend_port}"

        print()
        print(f"{BOLD}{GREEN}========================================{NC}")
        print(f"{BOLD}{GREEN}  Opti-Oignon v{_get_version(backend_port)} is ready!{NC}")
        print(f"{GREEN}========================================{NC}")
        print()
        print(f"  {CYAN}Interface:{NC} {browser_url}")
        print(f"  {CYAN}API docs:{NC}  http://localhost:{backend_port}/docs")
        print()
        print(f"  {YELLOW}Ctrl+C to stop everything{NC}")
        print()

        _open_browser(browser_url)

        # -- Keep alive: one that stops is said once the other one is stopped --
        while True:
            if backend_proc.poll() is not None:
                cleanup(code=1, why=f"{RED}[!] The backend stopped ({_status(backend_proc.returncode)}): its last "
                                    f"lines are above.{NC}")
            if frontend_proc.poll() is not None:
                cleanup(code=1, why=f"{RED}[!] The frontend stopped ({_status(frontend_proc.returncode)}).{NC}")
            time.sleep(2)
    except KeyboardInterrupt:
        cleanup(code=130)
    except Exception as exc:  # noqa: BLE001 - whatever ends the launcher stops what it started
        # Its output gone -- a closed window, a pipe whose reader ended --, a child that could not start, anything
        # else: what was started is stopped, and why is said where it still can be.
        cleanup(code=1, why=f"{RED}[ERR] {type(exc).__name__}: {exc}{NC}")
