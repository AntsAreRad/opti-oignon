#!/usr/bin/env python3
"""
Opti-Oignon UI Launcher
========================

Launches the full Opti-Oignon stack:
  1. Checks prerequisites (Python, Node, Ollama), and holds the frontend's
     installed dependencies to package-lock.json (npm ci when they differ)
  2. Frees ports if occupied (with user confirmation)
  3. Starts FastAPI backend
  4. Starts SvelteKit frontend (npm run dev)
  5. Opens browser

Usage:
    python -m opti_oignon [ui] [--port PORT]
"""

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


def _wait_for_backend(port: int, timeout: int = 30) -> bool:
    """Wait for backend to be ready."""
    import urllib.request
    for _ in range(timeout):
        try:
            req = urllib.request.Request(f"http://localhost:{port}/api/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
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
    # Override default port 8000 to 8001
    backend_port = port if port != 8000 else DEFAULT_BACKEND_PORT
    frontend_port = DEFAULT_FRONTEND_PORT

    project_root = _find_project_root()
    frontend_dir = project_root / "frontend"

    backend_proc = None
    frontend_proc = None

    def cleanup(signum=None, frame=None):
        """Clean shutdown of both processes."""
        print(f"\n{YELLOW}[>] Stopping Opti-Oignon...{NC}")
        if frontend_proc and frontend_proc.poll() is None:
            frontend_proc.terminate()
            print("    Frontend stopped.")
        if backend_proc and backend_proc.poll() is None:
            backend_proc.terminate()
            print("    Backend stopped.")
        print(f"{GREEN}[OK] Opti-Oignon stopped.{NC}")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

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
    backend_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "opti_oignon.api.app:app",
            "--host", bind_host,
            "--port", str(backend_port),
            "--log-level", "warning",
        ],
        cwd=str(project_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    sys.stdout.write("    Waiting")
    sys.stdout.flush()
    if _wait_for_backend(backend_port):
        version = _get_version(backend_port)
        print(f"\n{GREEN}[OK] Backend ready (v{version}){NC}")
    else:
        print(f"\n{RED}[ERR] Backend did not start within 30s{NC}")
        if backend_proc.poll() is None:
            backend_proc.terminate()
        sys.exit(1)

    # -- Start frontend --
    print(f"{CYAN}[>] Starting frontend on port {frontend_port}...{NC}")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", str(frontend_port)],
        cwd=str(frontend_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
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

    # -- Keep alive --
    try:
        while True:
            # Check if processes are still running
            if backend_proc.poll() is not None:
                print(f"\n{RED}[!] Backend stopped unexpectedly.{NC}")
                cleanup()
            if frontend_proc.poll() is not None:
                print(f"\n{RED}[!] Frontend stopped unexpectedly.{NC}")
                cleanup()
            time.sleep(2)
    except KeyboardInterrupt:
        cleanup()
