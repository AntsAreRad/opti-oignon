# Contributing to Opti-Oignon

Thanks for considering a contribution. Opti-Oignon is a local-first AI
inference platform with defense-in-depth security. This document covers
how to set up a development environment, the conventions the project
follows, and how to submit changes.

## Development setup

Prerequisites on Linux (tested on Kubuntu and Ubuntu):

```bash
sudo apt install -y \
    build-essential cmake pkg-config \
    libsqlcipher-dev liboqs-dev \
    python3-dev python3-pip \
    libffi-dev libssl-dev nodejs npm
```

Clone and install:

```bash
git clone https://github.com/AntsAreRad/opti-oignon.git
cd opti-oignon
pip install --user ".[auth,sqlcipher,llama,dev,docs]"
cd frontend && npm install && cd ..
```

### Alternative: conda / mamba

On systems without the apt prerequisites (or where you prefer an isolated
environment), conda-forge provides SQLCipher and the build toolchain, so no
`sudo apt` step is needed:

```bash
conda create -n opti-oignon python=3.12 -y
conda activate opti-oignon
# SQLCipher + compilers from conda-forge (replaces the apt prerequisites):
conda install -c conda-forge sqlcipher cmake compilers nodejs -y
# Inside the env, plain pip (no --user) installs into the env:
pip install ".[auth,sqlcipher,llama,dev,docs]"
cd frontend && npm install && cd ..
```

Note: the `[all]` convenience extra installs only the pure-Python groups
(`auth,dev,docs`); the platform-specific `llama` and `sqlcipher` extras must be
requested explicitly (as above), since they need system or conda-forge
libraries.

Launch in dev mode:

```bash
# Terminal 1: backend (port 8001)
scripts/dev_backend.sh --reload

# Terminal 2: frontend (port 5173, proxies /api to backend)
scripts/dev_frontend.sh
```

## Code conventions

- All code comments and docstrings in English. No French in code or UI.
- No hardcoded hex colors in Svelte components. Use `--oo-*` CSS variables.
  Fallback values inside `var(--oo-foo, #hex)` are tolerated for resilience.
- `checkpoint_before_apply = True` hardcoded in every new module that
  modifies persistent state.
- Literal FastAPI routes registered before parametric catch-alls.
- AST verification after every Python file creation or modification.
- HTML tag balance checks on Svelte components.

## Testing

```bash
# Backend tests
scripts/run_tests.sh

# Coverage
scripts/run_coverage.sh

# End-to-end (Playwright)
scripts/run_e2e.sh

# Lint and typecheck
scripts/lint.sh
scripts/run_typecheck.sh
```

The project targets ~9500 backend tests across 152 files. New features
should ship with tests in `tests/test_<feature>.py`.

## Security

Opti-Oignon takes security seriously. The threat model, defense layers,
and disclosure process are documented in [SECURITY.md](SECURITY.md).
If you find a vulnerability, please follow the disclosure process there
rather than opening a public issue.

## Pull requests

1. Fork the repository and create a topic branch from `main`.
2. Keep changes focused -- one feature or fix per PR.
3. Add tests covering the change. CI must pass (lint, typecheck, tests, E2E).
4. Update CHANGELOG.md under the unreleased section.
5. Update relevant documentation in `docs/` (rendered with MkDocs).
6. Open the PR against `main` with a clear description of what and why.

The CI pipeline (`.github/workflows/ci.yml`) runs on every PR.
Branch protection details are in `BRANCH_PROTECTION.md`.

## Roadmap and sessions

Opti-Oignon is developed in numbered sessions (S1, S2, ...). Each session
ships an incremental release with a corresponding `PROMPT_SXX.md` and
`SESSION_TRACKING_*.md`. External contributors do not need to follow this
cadence -- it is a private development convention.

## License

By contributing, you agree that your contributions will be licensed under
the MIT License (see [LICENSE](LICENSE)).
