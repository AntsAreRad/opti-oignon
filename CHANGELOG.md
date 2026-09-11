# Changelog

All notable changes to Opti-Oignon are documented in this file.
Security-relevant changes are marked with [SECURITY].

## Unreleased

The merge guards learn to say what they cannot prove, rather than reporting an
absence of proof as a verdict, and a new one measures what importing the
package costs.

### Added

- Onion memory, behind `enabled: false` in `opti_oignon/config/onion.yaml`.
  A bounded window -- Core, receipts, Peels, Flesh -- over an archive that is
  the only source of every compression: a span leaves the window only once
  its summary has answered the recall probes drawn from it, and only with a
  receipt the model can see. Off, the chat path is unchanged; on, the
  librarian curates off the interactive path through the inference registry
  with the model released after every burst. The figures the contracts
  produce are fixture readings and say so; `scripts/onion_runbook.py` takes
  the measured ones on the host. Documented in `docs/architecture/onion-memory.md`.
- `import_footprint_guard.py` observes the import in a subprocess, through an
  audit hook, and refuses any database, any file written into the caller's
  directory, or any heavy dependency that its ledger does not already carry.
  The ledger records the debt that predates the guard and **may only shrink**.
  The module count carries a ceiling because it is stable to the digit; wall
  time and resident memory are recorded and never enforced, because both move
  with the machine and a ceiling that moves with the machine forbids nothing.
- `registry_funnel_guard.py` keeps every inference request inside the backend
  registry, where admission, provenance, constrained decoding and the placement
  recipe live. It counts direct calls to the client library on the syntax tree,
  so prose never counts, in every spelling the library can be reached by. The
  modules that still call directly are a sealed ledger that **may only
  shrink**: an owed module that changes while still calling directly must
  migrate, and a module nobody owes for is refused outright.
- A static concordance check pairs the mobile bridge's declarations with its
  native entry points by name, arity and type. A type outside the agreed table
  is refused rather than mapped by default, and zero declarations on either
  side is refused rather than agreed.
- A driver for the interlanguage round trip, which compiles both sides and
  compares what comes back with what was sent. It reports three outcomes, not
  two: the third means the measurement did not run, and nothing downstream may
  read it as either a pass or a failure.
- Unit tests for the wire envelopes, in the source set that needs no device.

### Changed

- Tool calls are schema-constrained. A forced tool decision travels as one
  schema branch per tool -- the name as a constant, the tool's own parameter
  schema for the arguments, closed to unknown keys -- so a constrained
  sampler cannot produce a call the tool cannot take; the native and the
  constrained schema come from one builder. At execution, an argument of the
  wrong type is refused before the handler, named, and marked retryable.
- The tool executor asks the inference registry at every head -- native tool
  decision, forced decision, streamed and single-shot final answer -- with
  tool schemas and decision schemas travelling as engine options, so the
  registry's admission, provenance and schema handling apply to tool calls
  like any other request. No direct client path remains in the module; with
  no backend registered each head degrades by name. The agent-eval harness
  lays its scripted backend over the registry for the length of a run
  instead of rebinding a name on the module.
- `comment_only_guard.py` accepts a rename proven by reconstruction: a
  substitution of names that carry internal naming for names that do not,
  applied to the before side and required to reproduce the after side exactly.
  Injectivity is asked of each namespace separately, since two names that never
  denoted the same binding merge nothing. Outside Python, where no analyser
  exists, an unattributable change is now reported as unjudged -- neither
  accepted nor refused -- and listed by name, so an absence of checking stays
  visible instead of passing for a check that succeeded.
- `public_clean_guard.py`, `public_language_guard.py`, `published_prose_guard.py`,
  `summary_fidelity_guard.py`, `isolation_seal_guard.py` and `red_team_guard.py`
  are unchanged in this cycle and continue to gate merges.

### Fixed

- The browser specs no longer race the first-run dialog. The helper waited a
  fixed budget for it to appear and, when the budget ran out first, returned as
  though there were nothing to dismiss; the dialog then opened over the page and
  swallowed the next click, so the failure landed on an unrelated locator. No
  budget can tell "not yet" from "never", so the application now marks the
  moment it has decided and the helper waits for that mark.

## 2.2.0 -- 2026-07-28

The semantic cache's published surface loses its legacy naming, which renames
five HTTP paths and four schemas. That breaks any client that spoke the old
prefix; the bundled frontend ships updated in the same release, which is why
it lands as a minor rather than a major. The version register becomes the
single declarative source, and release verification learns to demand the
project's key rather than any key.

### Changed

- **Breaking.** The semantic cache endpoints move under
  `/api/cache/semcache/*`, and the four cache schemas follow
  (`SemCacheStatusResponse`, `SemCacheStatsSchema`, `SemCacheConfigUpdate`,
  `SemCacheClearRequest`). The retired prefix carried an internal iteration
  code, which is exactly why it goes. The bundled frontend is updated in
  the same release; external clients must adopt the new prefix.
- The version register (`opti_oignon/__version__.py`) is the only place a
  version is declared. The packaging manifest, the frontend package and the
  newest changelog heading are contract-pinned equal to it; the health
  dashboard default derives from it instead of restating it; code banners
  no longer carry a version at all. Each retired site had drifted to a
  different stale value, which is what this policy ends.
- The merge guard that rejects internal nomenclature now reads a session
  code in either case and through a camel-case continuation, while never
  charging platform names whose digits run into a lowercase letter.
- [SECURITY] **Breaking in Bulbe.** Bulbe now REFUSES to load a GGUF whose
  provenance does not verify -- including one that is simply not enrolled yet.
  Enrol existing models before switching to Bulbe. Configuration cannot weaken
  this; a security mode that cannot be resolved is treated as Bulbe. Daily is
  unchanged by default: it observes and logs without blocking. Models served
  through Ollama are not affected, as the gate sits on the in-process load seam.

### Fixed

- [SECURITY] `verify_release.sh` exit codes match their documented map: a
  checksum file that is absent in strict mode exits 3 (missing file), not
  2 (mismatch). Failure diagnostics no longer leak the exit code into the
  printed message.

### Added

- [SECURITY] `verify_release.sh` honours a pinned project fingerprint
  recorded in `scripts/release_key.fpr`: when present and no `--key` is
  given, a genuine signature from any other key is refused. Without a pin
  it now says out loud that a valid signature proves integrity, not
  identity.
- [SECURITY] Model weight provenance. GGUF files are now pinned to the sha256 of
  their bytes in a manifest sealed with ML-DSA-65 (or HMAC-SHA512 where liboqs is
  absent), and the llama.cpp in-process load seam verifies that pin before the
  bytes reach the native parser. The path guard already proved WHERE a model file
  sits and the SSRF guard proved WHERE it was fetched from; nothing proved WHAT it
  contained, and a trojaned or corrupted model was parsed by native code
  regardless.
- [SECURITY] `POST /api/backends/gguf/download` accepts an optional
  `expected_sha256`. It is verified against the partial file BEFORE the download is
  promoted to a loadable `.gguf`, so a mismatch never materialises a model. Taken
  from a model card rather than from the serving host, it is the only check on that
  path that is not trust-on-first-use. Successful downloads are enrolled in the
  manifest automatically.

- Merge guards under `.github/scripts/`, all wired into CI. `public_clean_guard`
  and `public_language_guard` hold the public surface to English prose free of
  internal working nomenclature; `comment_only_guard` catches files that shed
  that nomenclature without changing behaviour; `isolation_seal_guard` keeps the
  test-isolation ledger shrinking and never growing; `published_prose_guard`
  pins every published schema description to a recorded digest, so the API
  surface cannot drift silently; `summary_fidelity_guard` and `red_team_guard`
  hold measured floors for summary fidelity and for the defense layers, failing
  the build when a measurement falls below the written threshold.
- Red team engine under `opti_oignon/redteam/`: attack generation, strategies,
  scoring, and reports against named defense targets, with a local probe driver
  in `scripts/redteam_llm_probe.py`. Documented in `docs/redteam/`.
- A shared isolation window for contract suites, `tests/_isolation.py`. Suites
  that load a single module from file no longer hand-roll their own package
  window, which is what let a dead signature primitive stay invisible.
- A browser end-to-end harness: Playwright specs under `frontend/tests/e2e/`
  driven by `scripts/run_e2e.sh`, covering the health surface, the reported
  security mode, and the sign-in refusal path.
- `constraints.txt`, the pinned environment under which every guard that reads
  the published surface runs, and under which its digest is regenerated. A
  floating resolver could otherwise move the schema without a line of this
  repository changing.
- Release signing: `scripts/sign_release.sh` and `scripts/verify_release.sh`
  produce and check a detached GPG signature plus a sha256 over the archive.
- `capability_manifest.py`, per-request introspection of what a target model can
  actually call, and `context_ledger.py`, per-request context measurements that
  record numbers about a turn and never its words.
- An Android client skeleton under `android/`. It does not yet talk to a
  backend; device-to-device sync remains unwired.

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
