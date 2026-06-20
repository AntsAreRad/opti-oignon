# Installation

## Prerequisites

Before installing Opti-Oignon, make sure you have:

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Node.js 18+** (for the SvelteKit frontend)
- **Ollama** running locally (see [Ollama Setup](ollama-setup.md))
- **Linux** (Kubuntu/Ubuntu recommended; macOS works without bwrap sandbox)

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
uvicorn opti_oignon.api.app:app --port 8000 --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173).


## Option C: One-click launcher

```bash
./launch.sh
```

The launcher starts both backend and frontend, checks for Ollama, and
opens the browser automatically.


## Verifying the installation

Once both services are running:

1. Open [http://localhost:5173](http://localhost:5173)
2. The onboarding overlay should appear on first run
3. It scans your Ollama models and recommends a system preset
4. Pick a preset (Minimal, Balanced, or Power) and start chatting

You can also check the backend health endpoint:

```bash
curl http://localhost:8000/api/health
```


## Troubleshooting

**Ollama not detected:** Make sure Ollama is running (`ollama serve`) and
accessible at `http://localhost:11434`. You can verify with
`curl http://localhost:11434/api/tags`.

**Port conflicts:** The backend defaults to port 8000 and the frontend to
5173. Change them with `--port` (uvicorn) or by editing `vite.config.ts`.

**Missing bwrap:** On Ubuntu/Debian, install with `sudo apt install bubblewrap`.
Without bwrap, the sandbox falls back to tempdir isolation (less secure but
functional).

**SQLCipher not found:** Install `sqlcipher` and `pysqlcipher3` for encrypted
database support. Without it, databases are stored unencrypted.
