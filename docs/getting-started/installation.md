# Installation

## Prerequisites

Before installing Opti-Oignon, make sure you have:

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Node.js 18+** (for the SvelteKit frontend)
- **Ollama** running locally (see [Ollama Setup](ollama-setup.md))
- **Linux** (Kubuntu/Ubuntu recommended; macOS works without the bwrap sandbox
  and without the one-click launcher, which needs Linux)

Optional but recommended:

- **bubblewrap** (`bwrap`) for kernel-level sandbox isolation
- **SQLCipher** for encrypted database support
- **liboqs-python** for post-quantum signature verification


## Option A: Docker Compose (recommended)

The simplest way to get everything running:

```bash
git clone https://github.com/AntsAreRad/opti-oignon.git
cd opti-oignon
docker compose up --build
```

Open [http://localhost:5173](http://localhost:5173) in your browser.


## Option B: Manual install

```bash
git clone https://github.com/AntsAreRad/opti-oignon.git
cd opti-oignon

# Install Python dependencies
pip install -e ".[all]"

# Install frontend dependencies
cd frontend && npm install && cd ..
```

Start the backend and frontend in separate terminals:

```bash
# Terminal 1: Backend
uvicorn opti_oignon.api.app:app --port 8001 --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173).


## Option C: One-click launcher

```bash
scripts/launch.sh
```

The launcher needs Linux: util-linux's `setsid` and `setpriv` (2.33 or
later), and a `/proc` of its own. Run it as it is, as above or from its
desktop icon: its first line runs it under `bash -p`, which reads no
`BASH_ENV` and takes no function from your environment, while
`bash scripts/launch.sh` would read them first. It starts the backend,
then, once it answers, the frontend, and opens the browser
automatically. Both run in sessions of their own, which the terminal
neither interrupts nor hangs up: Ctrl+C or closing the window tells the
launcher, which tells each to stop, waits 15 s at most for it to end,
and kills only what did not, saying so; and each is told to stop when
the launcher ends, however it ends. A backend that stops is said with its
exit status once the frontend is stopped, so its error stays in sight.
`python -m opti_oignon` does the same, on macOS too, where a launcher
killed outright leaves what it started running. When
the key file is wrapped under a passphrase (see
[Encryption](../security/encryption.md)), the launcher asks for it on its
terminal before it starts anything, checks it, and hands it to the backend
alone, as it does `OPTI_ENCRYPTION_KEY`: the frontend and the browser start
without them. A desktop icon for it needs `Terminal=true`, so that the
launcher has a terminal to ask on, and
`Exec=bash -ic '<project>/scripts/launch.sh'`, so that your shell's conda
and node setup load while the launcher still runs by its first line --
`bash -i <project>/scripts/launch.sh` would skip it.


## Verifying the installation

Once both services are running:

1. Open [http://localhost:5173](http://localhost:5173)
2. The onboarding overlay should appear on first run
3. It scans your Ollama models and recommends a system preset
4. Pick a preset (Minimal, Balanced, or Power) and start chatting

You can also check the backend health endpoint:

```bash
curl http://localhost:8001/api/health
```


## Troubleshooting

**Ollama not detected:** Make sure Ollama is running (`ollama serve`) and
accessible at `http://localhost:11434`. You can verify with
`curl http://localhost:11434/api/tags`.

**Port conflicts:** The backend defaults to port 8001 and the frontend to
5173. Change them with `--port` (uvicorn) or by editing `vite.config.ts`.
`scripts/launch.sh` refuses to start when either port already answers;
`python -m opti_oignon` offers to stop whatever holds them.

**Missing bwrap:** On Ubuntu/Debian, install with `sudo apt install bubblewrap`.
Without bwrap, the sandbox falls back to tempdir isolation (less secure but
functional), except while the startup checklist says a secret the backend
was started with could not be wiped from its environment: then no sandbox
command and no plugin runs outside bwrap.

**SQLCipher not found:** Install `sqlcipher` and `pysqlcipher3` for encrypted
database support. Without it, databases are stored unencrypted.
