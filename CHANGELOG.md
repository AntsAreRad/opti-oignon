# Changelog

All notable changes to Opti-Oignon are documented in this file.
Security-relevant changes are marked with [SECURITY].

## Unreleased

The merge guards learn to say what they cannot prove, rather than reporting an
absence of proof as a verdict, and a new one measures what importing the
package costs.

### Added

- Two observation heads on the backend contract, defaulted rather than
  abstract: `loaded_models()` answers the models a backend reports resident,
  and `None` when it cannot say -- never an empty list, which would read as
  "nothing loaded" where the truth is "nobody looked"; `embed(model, text)`
  answers one vector, `None` when the backend has no embedding endpoint, and
  asks the governor first. Ollama implements both through its client;
  llama.cpp answers its in-process set; llama-server and the remote core
  answer unknown by name. The warmup and the governor's S1 read now say
  unknown instead of empty when nothing observed the loaded set.
- A request timeout travels as the engine option `timeout` and binds the
  transport -- a per-timeout client on Ollama, the request's own timeout on
  llama-server -- and never reaches the engine; a timeout that is not a
  number is refused before anything leaves. The counts an engine reports
  beside its answer (`eval_count`, `prompt_eval_count`, the durations)
  arrive on `extra` of the response and of the chunk that carries them, and
  are absent rather than invented when nothing was reported. Both existed
  only in the private clients the benchmark, judge and reasoning modules
  kept for themselves.
- `registry_clients.py`: the two clients the verification, note-action and
  agent routes now stand on, a one-shot text completion and a streamer in
  the agent loop's chunk shape, both sending their request through the
  inference registry. The host argument of the former signature is accepted
  and unused: where a model is served is the registry's to know.
- The core daemon (`oo core serve`): a resident loopback HTTP server over the
  inference registry, standard library only, with health, models, generate,
  stream as JSON lines, and a pack's admission ticket; off by default in
  `core.yaml`. When enabled, `core_client.RemoteCoreBackend` is registered
  in the calling process so the daemon serves its inference through the
  registry -- admission, provenance and schema apply across the process
  boundary. Refuses by name: non-loopback host, missing token, unknown
  route, malformed body, model without a backend.
- `core_boundary_guard.py` names the resident core module by module and holds
  the line the census drew: a core module that imports a module outside the
  core at module scope is refused unless the ledger already carries that
  leak; the ledger may only shrink; no core module reaches the inference
  client directly, the registry excepted. Documented in
  `docs/architecture/native-core.md`.
- `scripts/core_census.py`: a static census of the package -- per module,
  the project imports at module scope and inside functions, both transitive
  closures, third-party imports outside the standard library, database
  files named, direct client sites, and the map of every router the API
  includes -- labelled `source: static`, refusing an empty tree. The
  instrument the core/packs cut is measured with.
- The onion memory's native core, `rust/oo_core`: the four integrity hashes
  and the window assembly in Rust behind the Python surface, loaded at the
  call and never at import, with Python as the reference and the fallback.
  Built by `scripts/build_oo_core.sh` from a pinned crate; the artefact is
  never tracked. Contracts hold every hash byte-equal and every prompt
  field-equal to the reference.
- Inference-time compute (`opti_oignon/inference_compute.py`): N candidates
  for one prompt sampled through the inference registry under a budget of
  candidates and tokens (`inference_compute.yaml`), each verified -- tests run
  through a sandbox runner for code, strict-majority agreement for the rest
  -- and the first verified one chosen, stopping there. When nothing is
  verified the best-scored candidate is returned marked unverified; with no
  score the selection is empty and carries the verdicts. Not wired into any
  path yet.
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
- The user's surface on the onion Core: `GET /api/memory/onion/{conv}`
  lists a conversation's Core entries and open receipts, `POST .../pin`
  pins a statement as the user, `POST .../supersede` pins a successor and
  links the old entry to it, `POST .../recall/{key}` hands the verbatim
  span behind a receipt back and marks it resolved. The librarian carries
  the five entry points and forwards the actor to the store, so only a
  caller that says it is the user gets through; a pin that would push the
  Core over its cap is refused before it lands, where before an over-cap
  Core would have blanked the whole memory block at compose time in
  silence (that refusal is now logged as a warning, by conversation). Every
  mutation is saved through the onion store when one is configured. The
  model's tools have no path to any of it; the no-pipeline contract now
  names the two modules that import the librarian and checks the tools.
- `oo chat`: an interactive session in the calling process, one line per
  turn through the executor and the inference registry, with `/open`,
  `/close`, `/pin`, `/recall`, `/skill`, `/help` and `/quit`; the stream
  goes to stdout and every refusal to stderr by name. The librarian gains
  `close_onion`, which evicts the whole Flesh through the gate
  synchronously, stops at the first refused span and names it, then saves
  and returns the receipts digest and the Core root, and `open_onion`,
  which finds a saved conversation again or refuses by name. `/skill` puts
  a published skill's body into the turn's system prompt; a draft, an
  unknown name and an ambiguous one are refused. The session is
  in-process, so it opens the local conversation store, onion state and
  skill root: it keeps the registry funnel, not the separate-process pack
  protocol. The CLI reference now documents `oo core` too.
- The onion memory persists: `memory/onion_store.py` writes each
  conversation's Core, Cellar, receipts, Peels, Flesh and mirror cursor
  through the repository's connection seam, after every mirror and every
  accepted eviction, and reads it back when the process next sees the
  conversation. The read re-hashes every Core entry, Cellar span and Peel
  against its saved id and the root over the four stores against the saved
  root, and refuses by name the first row that no longer answers; a
  conversation the file does not know is unknown, never an empty state,
  and a refused one is logged by name and left absent rather than replaced
  by a fresh one. The table holds conversation text, so a connection that
  is not encrypted is refused by default (`persistence.require_encryption`
  in `onion.yaml`), where the rest of the tree opens plaintext with a
  warning outside Bulbe mode; the drift ledger gains the same refusal as an
  opt-in. `persistence.path` is empty in the shipped file: the maintainer
  sets it when switching the onion on. The Core has no pin surface yet, so
  what persists of it is what the tests pin; that surface is its own block.

### Changed

- The Ollama `host` in `backends.yaml` is where the requests go. It was
  written onto the backend and read by no request: every head asked a
  client that resolved `OLLAMA_HOST` or the library's default. Every head --
  health, listing, loaded set, model details, generate, stream, embeddings,
  eviction -- now goes through one client for the configured host, unless
  `OLLAMA_HOST` is set, which still wins; `endpoint()` follows the same
  order. The shipped value changes from `http://localhost:11434` to
  `http://127.0.0.1:11434`, the library's own default, so a machine that
  sets neither sends exactly where it did; a file that names another host
  now sends there.
- The registry-funnel guard follows the client through a function or a
  method that returns it: a request made on a helper's result is a site.
  Routing the Ollama heads through one transport had hidden the funnel's
  own calls from the census, and the same shape anywhere else would have
  passed as clean.
- The red team reaches its model through the inference registry, so every
  attack is admitted by the governor. Its loopback property moves to where
  the requests actually go: a new `endpoint()` head on the backend contract
  answers the base URL a backend's requests reach -- Ollama's as its client
  library resolves it from `OLLAMA_HOST`, llama.cpp in-process, llama-server
  and the core daemon their own -- and the generator, the multilingual
  strategy and the chat target refuse by name, before any request, a backend
  whose endpoint is unknown or off the local host. Their URL arguments are
  still checked and no longer route anything. The launcher's liveness probe
  is exempt from the funnel guard by name, and the raw ledger is empty.
- Every embedding goes through the inference registry. The embedder behind
  the RAG store, the project context and the memory's semantic recall
  posted to the server's embedding endpoints with its own HTTP transport;
  it now asks the registry's backend, one text through `embed` and a batch
  through `embed_many`, a new head on the backend contract: `None` where a
  backend has no embedding endpoint, one admission per batch on Ollama, and
  a refusal by name when the answer's count does not match the texts. Both
  embedding heads take a timeout. A misaligned or failed batch is still
  redone text by text. The model is verified against the registry's
  catalogue, and a catalogue nobody could read verifies nothing. Ollama
  versions without `/api/embed` are no longer served: the legacy
  `/api/embeddings` fallback is gone.
- The registry-funnel guard sees a request that never touches the client
  library: an endpoint of the inference server spelled in a module that
  imports an HTTP transport. It found six modules at nine sites. The
  project trigger detector's third level now asks the registry, with its
  half-second budget as the request timeout; the other five are sealed on
  a raw ledger that may only shrink, each waiting on a decision -- the RAG
  embedder sends batches, the red team enforces a loopback endpoint the
  registry does not check, the launcher's probe is not inference.
  `level3_ollama_url` in `projects.yaml` is no longer read.
- The tuner reaches a real backend. Four call sites asked the registry for
  `get_backend`, a method it does not have, inside broad handlers: every
  tuning run fell to the simulated benchmark, the llama.cpp benchmark
  answered "not available", and the speculative-decoding listing was always
  empty. They now ask `get`. The Ollama benchmark no longer posts to the
  server's HTTP endpoint itself: it asks the registry's Ollama backend, so
  every sweep point is admitted by the governor, the request carries a
  timeout, and a refusal comes back as a named error on that point. Rates
  are read from the counters the backend reports and are labelled measured
  only when both are present, as before. The `host` argument is accepted
  and unused.
- `list_models()` on the backend contract answers `None` when the backend
  cannot say and an empty list only when it looked and found nothing, the
  doctrine the two observation heads already carried. Every implementation
  answered an empty list before: Ollama with the client absent or the call
  failing, llama-server on a transport error or a body without a listing,
  the remote core with the daemon down, llama.cpp with no configured
  directory to scan (the production instance is built with none, so its
  listing was a constant empty), the test bridge without a scripted
  `list`. The daemon's `GET /models` carries unknown as a null listing
  with `known` false and a known empty with `known` true; the remote core
  reads it back. The registry's backend status reports an unknown count as
  null rather than zero, and its aggregate listing skips a backend that
  could not list without hiding the others. The shared catalogue readers
  propagate unknown, so their "empty only when one looked" claim is now
  true; the extraction falls to its first fallback model on an unknown
  listing instead of scanning nothing, and the health monitor skips a
  check it could not begin instead of marking nothing. Every other caller
  names what it does with unknown in its own log line, and none iterates
  a `None`.
- The model catalogue is read through the inference registry: sixteen
  modules asked the client library which models are installed and what a
  model is made of (`list()` and `show()`) at twenty-two sites, and the
  registry-funnel census could not see three of them -- two modules that
  bound the client module to an attribute at construction and asked
  through the attribute, one with a `chat` carrying images, and two
  request methods handed on uncalled. The census now follows the client
  into a bound attribute or name, counts an uncalled reference, and counts
  `list` and `show` beside the request methods; on the files as they were
  it reads 22 sites in 16 modules, on the tree it reads 0 over 355. Model
  management (`pull`, `delete`) has no head on the contract and is not
  counted, by decision. The catalogue is the two heads the contract already
  had: Ollama's `model_info` reads both answer forms and carries in `extra`,
  only when reported, the family list, the parameter text, the digest, the
  template, the modelfile, the license and the raw mapping; a list entry
  carries the size in bytes, the digest and the family list the same way.
  Three readers on `registry_clients.py` answer every module -- `None` when
  no backend is registered or the registered one could not read its
  listing, an empty list only when one looked and found nothing (the
  second half of that sentence became true one block later, see below).
  The vision pipeline describes through `generate` with the
  images; the dependency layer's model listing asks the active backend and
  no longer falls through to the client (the registry method it asked for
  never existed, so the fall-through ran every time); the CLI listing
  exits non-zero when no backend is registered instead of printing a
  connection error.
- The last eleven modules ask the inference registry instead of the client
  library, and the registry-funnel ledger is empty: the two benchmark
  transports and the judge (streaming, timeout as an option, the reported
  token count over the chunk count), the model warmup (loaded set through
  the new head, warm-up as one user message and a single token, residency
  renewed with no messages), the semantic cache's embedding, the cascade,
  the humanizer's rewrite pass, the reasoning engine (available exactly
  when the registry serves its default model, per-step timeout as an
  option), the fine-tune comparison, the benchmark route and the routing
  benchmark (model listings from the registry's backends). Each degrades
  by name without a backend. Four of them sent a completion prompt; it
  travels as one user message, the decision the previous block took once.
- `registry_funnel_guard.py` counts a `ps()` read as a client site now that
  the loaded set is a head on the contract, refuses an estate it could not
  scan instead of printing a zero, and names how many modules it read when
  it finds nothing owed. The runner, pre-cache and humanizer suites stand on
  the shared window and the registry bridge; the humanizer suite leaves the
  isolation-seal ledger.
- Nine more modules ask the inference registry instead of the client library:
  self-correction (four completion calls), the dynamic planner and its step
  executor, the agent base (chat, stream and the model list), speculative
  draft and verify, legacy fact extraction (two chats and the model list),
  and the five routes that built a client per request, some with a host of
  their own. Every one of those requests is now admitted, labelled and
  schema-checked; a model no backend serves gets a refusal by name or the
  documented degraded result (heuristic scores, the fallback plan, no facts,
  an error marker), never a client call. The funnel guard's ledger goes from
  21 modules and 36 direct sites to 11 and 12; `model_warmup` stays owed
  because its process listing has no registry surface yet.
- Importing `opti_oignon` no longer imports the API application. The package
  facade exports the same names through a module `__getattr__`: each is
  imported when first asked for, and names that are also submodule names
  (`config`, `router`, `executor`, ...) still resolve to the exported object.
  Measured on the import-footprint guard: 1634 modules, 2.1 s and 187 MiB at
  import before; 85 modules, 1 ms and 13 MiB after, with no database opened
  and no heavy dependency reached. The guard's ceiling drops from 1700 to
  200 modules and its ledger of import-time debt is empty.
- The coding agent reads its test verdict from the pytest summary instead
  of substrings: a passing run whose output mentions the word error passes,
  an error counts as a failure, and counts are recorded; a run with no
  summary is still treated as passed with a zero count. `fix_candidates` in
  `coding_agent.yaml` (default 1, the loop as it was) lets each fix attempt
  try several candidates, each applied, tested and undone by its inverse
  when it fails, so the first that passes stays.
- The agent-eval runner's chat client asks the inference registry for each
  turn, tool schemas as an engine option; `OllamaChatClient` stays as an
  alias of `RegistryChatClient`.
- Tool calls are schema-constrained. A forced tool decision travels as one
  schema branch per tool -- the name as a constant, the tool's own parameter
  schema for the arguments, closed to unknown keys -- so a constrained
  sampler cannot produce a call the tool cannot take; the native and the
  constrained schema come from one builder. At execution, an argument of the
  wrong type is refused before the handler, named, and marked retryable.
- The verification engine's fix requests and the consensus engine's model
  queries go through the inference registry; consensus reports itself
  available when the registry can serve a model, not when a client library
  imports.
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
