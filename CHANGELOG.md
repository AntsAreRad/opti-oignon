# Changelog

All notable changes to Opti-Oignon are documented in this file.
Security-relevant changes are marked with [SECURITY].

## 2.1.0 -- 2026-06-27

A capability release on two fronts: an agentic robustness cycle that makes local
tool use far more reliable, and a memory-system overhaul that closes the
capture -> store -> injection loop on a single source of truth.

### Added

- Native model function-calling for agentic tool use: when a model advertises the
  capability, tool calls go through its native function-calling interface, with a
  JSON-schema-constrained path as the unconditional fallback otherwise.
- Automatic memory capture: after a turn is saved, facts are extracted and stored
  in the background every few messages, so memory accumulates without a manual
  `/extract`. Gated and throttled, fire-and-forget.
- Memory health endpoint: `GET /api/memory/health` reports the canonical, archive
  (semantic) and embedder tiers, so a degraded recall path is visible instead of
  silent.

### Changed

- Agentic tool loop hardened (the robustness cycle): enum-forcing for constrained
  arguments, intent-transpiler salvage and argument auto-repair for malformed tool
  calls, an error-feedback retry so a failed call self-heals, an anti-spin guard
  plus a verification pass to stop the agent looping, and capability-aware
  reasoning handling that avoids the think=True / HTTP 400 case on models that do
  not support it, with an explicit optimize toggle.
- Memory unified on one source of truth (the coordinated MemoryStore): the
  `/api/memory` surface (list/add/delete/clear/extract) is re-backed by the new
  store and mapped onto the existing schema, so the frontend is unchanged. The
  working block now keeps a salience floor -- durable facts are always injected,
  not dropped on an unrelated turn -- and marks injected facts as used.
- The memory vector (semantic) layer degrades gracefully: when chromadb is not
  installed it falls back to canonical keyword/recency recall instead of raising,
  so the memory tab, list and migration keep working without it; only similarity
  search is disabled, and health reports the archive tier as unavailable.

### Fixed

- The memory tab and the injector no longer read different stores: facts entered
  in the tab now surface in recall. Previously the tab wrote the legacy store
  while the injector read the new one, so tab-entered facts never appeared.
- An unrelated query no longer drops durable memories from the working block (the
  old retrieval path discarded every fact scoring zero).

### Internal

- One-shot legacy `memories.db` -> MemoryStore migration runs once at application
  boot: idempotent (the store's dedup merges a re-run), marker-guarded, and
  fail-safe -- a migration problem is logged and swallowed, never breaking
  startup.
- The silent no-embedder path now logs once and is surfaced by the health probe
  rather than degrading recall invisibly.

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
