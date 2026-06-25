# Changelog

All notable changes to Opti-Oignon are documented in this file.
Security-relevant changes are marked with [SECURITY].

## 2.0.2 -- 2026-06-25

Data-integrity release: fixes a bug that caused agentic conversations to be
lost, plus related persistence hardening.

### Fixed

- Agentic conversations (those using the in-session sandbox and tool calls) were
  not saved: the history was empty after a page reload and the context token
  counter showed 0. The persistence call passed an unexpected keyword argument,
  raising an error that was silently swallowed, so every agentic turn was
  dropped. This affected anyone running 2.0.1.
- Turns that combined a reasoning pass with tool use persisted only the
  reasoning; the tool output was dropped on reload. The complete turn is now
  saved.

### Changed

- The streaming idle-disconnect timeout is now configurable through the
  `OPTI_IDLE_TIMEOUT_S` environment variable, and its default was raised from
  60 to 600 seconds so slower local models that stream in bursts are not cut off
  mid-response.

### Internal

- Conversation persistence now records the generating model on assistant
  messages and fails loudly (a logged warning with traceback) instead of
  silently, so a future persistence regression is visible.
- Added defensive persistence to the cascading and speculative generation
  pipelines so they cannot drop a turn if they are ever wired into a saved
  conversation.

## 2.0.1 -- 2026-06-23

Maintenance release: a version-reporting fix, a small frontend security
hardening, and repository cleanup. No functional changes to the application.

### Fixed

- The command line and the package metadata now both report 2.0.1. The version
  module had carried a stale internal version string over from pre-release
  development, so the 2.0.0 build reported the wrong number from `oo --version`.

### Security

- Removed unnecessary `{@html}` rendering in the benchmark panels, eliminating
  an unused HTML-injection surface in the frontend. [SECURITY]

### Internal

- The public continuous-integration pipeline (Python lint, frontend type-check
  and lint, install smoke test, security scan) now passes.
- Removed a dead pytest configuration block from `pyproject.toml` that referenced
  a test suite not shipped in the public distribution.

## 2.0.0 -- 2026-06-20

A complete rewrite and public re-release. Opti-Oignon began as a Gradio-based
local-LLM optimization framework (the 1.x line); 2.0.0 replaces that entirely
with a new local-first AI inference platform built on SvelteKit and FastAPI,
running against Ollama on your own hardware.

### Working in this release

- Private, streamed chat from local Ollama models over WebSocket.
- Accounts with registration, login, and optional 2FA (TOTP and WebAuthn). [SECURITY]
- Two security modes. Daily is the normal mode; Bulbe binds the backend to
  `127.0.0.1` at the socket layer and accepts cookie-only authentication, with a
  guarded, human-confirmed downgrade ceremony and fail-secure behavior when the
  mode cannot be determined. [SECURITY]
- Encryption at rest via SQLCipher. [SECURITY]
- Projects with RAG context backed by ChromaDB.

### Also included, still maturing

A broad backend surface is implemented and covered by the test suite but not yet
verified end to end on a fresh install: smart model routing, multi-model
consensus and cascading inference, a semantic cache, a sandboxed agent loop,
benchmark and performance dashboards, a resource governor, an LLM-powered red
team engine, RBAC and multi-user isolation, and an encrypted Notes tab. See the
Status section of the README for details.

### Not yet wired end to end

Veilid device-to-device sync is incomplete on the producer side, so nothing
moves between paired devices yet. Everything that depends on it -- remote
inference, collaborative Notes sync, and the mobile client -- is experimental.

### Security posture

Deny-by-default authentication, per-user data isolation, Argon2/bcrypt password
hashing, a disposable bubblewrap sandbox for any LLM-driven filesystem, shell,
or code tool, ML-DSA-65 post-quantum signatures on records intended for sync,
and a hash-chained audit log. Security follows Kerckhoffs's principle: it rests
on keys and correct implementation, not on secrecy of the code. [SECURITY]

---

Earlier history (the 1.x Gradio line) remains available in the git tags.
