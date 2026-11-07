# Security

This document describes Opti-Oignon's security architecture, threat
model, deployment recommendations, and vulnerability reporting process.
For detailed per-topic documentation, see the
[Security Guide](docs/security/overview.md) in the MkDocs site.


## Security Philosophy

Opti-Oignon follows Kerckhoffs's principle throughout: security derives
from keys, cryptographic primitives, and human factors -- never from
code obscurity. The project is designed for open source release with no
security-through-obscurity assumptions.

Local-first is a security feature. All LLM inference, RAG indexing,
embedding generation, and data storage happen on the user's hardware.
No data leaves the machine unless the user explicitly enables optional
web search (which itself passes through PII sanitization).


## Threat Model

### What Opti-Oignon protects against

- Unauthorized access to conversations, RAG data, and user accounts
- LLM prompt injection via RAG context or plugin output
- Sandbox escape from LLM-driven filesystem tools
- Data exfiltration through crafted LLM tool calls
- Tampering with the audit log (SHA-512 hash-chain, keyed HMAC-SHA256)
- Man-in-the-middle on localhost (Bulbe mode socket-level binding)
- Supply chain attacks on releases (GPG signing, SHA-256 checksums)
- Dependency vulnerabilities (automated pip-audit monitoring)

### What is out of scope

- Physical access to an unlocked machine (LUKS detection is advisory)
- Attacks against Ollama itself (upstream responsibility)
- Browser-level exploits (CSP mitigates but cannot eliminate)
- Side-channel attacks on local model inference
- Denial of service against the local backend (single-user context)


## Defense Layers

Opti-Oignon implements six independent defense layers. Compromising one
does not bypass the others.

### Layer 1 -- Authentication and access control

Global deny-by-default middleware on all API endpoints. JWT session
cookies with server-side algorithm enforcement. Multi-user RBAC with
per-user data isolation (separate encryption keys derived via Argon2id).
Optional 2FA via WebAuthn/FIDO2 and TOTP with recovery codes. Login
timing oracle prevention (dummy bcrypt on invalid usernames). Rate
limiting with exponential lockout: a login checks one password at a time
for an address and for an account, so that tries sent together are each
counted, and a code sent at the second step is counted before it is
read. The address is the socket's, or the one a proxy listed under
`api.trusted_proxies` forwards (none as shipped): no route and not
uvicorn reads X-Forwarded-For, the Host gate does, from a listed proxy
only. CSRF double-submit cookie on the
state-changing endpoints of a signed-in browser session, the security
mode's included (in single-user Daily mode, which has no session cookie,
and for the few routes it exempts -- sign-in, registration, the session's
status and refresh, the health check itself (not the routes below it),
the schema -- the gate's origin check below is what refuses another
site).

The API answers only the names the machine is served as: a request, or a
websocket handshake, whose Host header is not `localhost`, `127.0.0.1` or
`[::1]` (any port) is refused before anything else runs, so a web page
that rebinds its own name to this machine (DNS rebinding) reaches nothing.
A machine served under other names lists them in `security.yaml`:

```yaml
api:
  hosts: ["oo.lan", "192.168.1.20"]
```

A page of another site can still send a request to the loopback address
itself (a form, a `fetch` that does not ask to read the answer, a
websocket): the same gate refuses every state-changing request (any method
but GET, HEAD and OPTIONS) and every websocket handshake that a page of
another site sends. A page is of this site when its `Origin` has the
request's scheme and the name and the port of its Host header (over
`https` for a name other than a loopback one: a page served under a listed
name over plain `http` changes nothing, serve remote access over TLS), or
when it is one of the origins listed under `cors.origins`. A page of
another port of this machine (a dev server, a local app) or of another
scheme is another site. A proxy that ends TLS in front of the API passes
the scheme on (`X-Forwarded-Proto`, from an address `api.trusted_proxies`
lists, below), or lists its pages' origin. The routes' own socket check
takes what the gate let through, and holds a socket the gate did not see
to a loopback name.
With no `Origin`, a request whose `Sec-Fetch-Site` is not `same-origin` or
`none` is another site's. A client that sends neither header is no current
browser (the CLI, curl) and is answered. A GET is not asked where it comes
from: a route must not change anything on a GET.

CORS answers only the origins `security.yaml` lists, with credentials; a
page of this site needs none, and a page of an origin not listed gets no
CORS header at all (the frontend's dev server answers no CORS of its own
either):

```yaml
cors:
  origins: ["http://localhost:5173"]
```

Each is an exact origin (`http` for a loopback name only, `https`
otherwise, no path, no wildcard); one entry that is not refuses the whole
list. `OPTI_CORS_ORIGINS` lists them when the file lists none; `"*"` lets
any page read the answers, without credentials, and lists no origin for
the gate. The frontend served through its dev server or its nginx proxy is
of the API's own origin; a build whose `VITE_API_URL` names the API's own
port is a page of another origin, to be listed.

A list that cannot be read leaves the loopback names alone. The names are
read when the server starts: a newly listed name needs a restart. A listed
name is only as safe as its resolution: on a local network, where another
device can answer for a `.local` or local DNS name, prefer an IP literal.
The garden's routes keep their own list (`api.hosts` in `allium.yaml`),
checked after this one: a name the garden serves is listed in both. A
reverse proxy in front of the API forwards the browser's Host with its
port (nginx: `proxy_set_header Host $http_host`; `$host` drops the port,
and a page served on another port than its scheme's default then changes
nothing); one that rewrites it to a loopback name (nginx's default with
`proxy_pass http://127.0.0.1:8001`) turns the name check off (a rebound
page reads the answers) and refuses every change and every socket of the
proxy's own pages, whose origin is not the Host it forwards.

A proxy whose `X-Forwarded-For` and `X-Forwarded-Proto` the API is to
read writes them itself (nginx: `proxy_set_header X-Forwarded-For
$proxy_add_x_forwarded_for;` and `proxy_set_header X-Forwarded-Proto
$scheme;`) and is listed by its address or network:

```yaml
api:
  trusted_proxies: ["127.0.0.1"]
```

The Host gate alone reads those headers, from a listed proxy only:
uvicorn reads none (every launcher of the tree starts it with
`--no-proxy-headers`, the backend image included). The scheme is the one
the proxy's single `X-Forwarded-Proto` says, and the client the last
address its `X-Forwarded-For` names that is no listed proxy -- read only
when it is an IP address written alone (no port, no name) and no
loopback one; otherwise the client is `unlisted-proxy`. A request that
carries a forwarded header from any other peer is handed on as that one
client too: never the loopback, so the localhost-only routes (device
certificates, remote access) refuse it, Bulbe's local-only check rejects
it (on every path but the health check, its sockets included), its
cookies are the remote ones, and the limiters count every such request
as one. A front that adds no forwarded header at all (nginx's
`proxy_pass` with no such line, a TCP forwarder such as `ssh -L` or
socat) is not seen: its clients are the front's own address, the
loopback for one on this machine, as before. A listed address can name
any client: list a network only when each of its hosts is a proxy, a
proxy only when it writes the header (one that relays the client's own
lets every client name its address), and `127.0.0.1` only when every
client of this machine may be taken at its word. An entry that is not an
address or a network as written (host bits set, an IPv4 address written
as IPv6, a scoped one), or a list that holds every address, lists none.
The list is read when the server starts: a newly listed proxy needs a
restart.

Remote access (Daily mode, `remote_access.enabled`) is served over mutual
TLS: each device holds a client certificate made on this machine. The
launcher (`opti-oignon api`) serves an address that is not a loopback name
only with the whole of it -- the server's key and certificate, the CA,
a client certificate asked --, and otherwise does not start. Every
connection is checked as soon as its handshake is done, before a request
on it is read, from the CRL and the devices' records on disk, and again
every second while it is open: a revoked certificate, one made again for
its device since (the new one replaces it), or one that cannot be checked
(a CRL or a record that does not load) is refused, after a restart too.
Revoking a device ends its open connections at once in the server that
revokes it, and within a second in another one (the settings page talks
to the loopback server, the devices to the mTLS one). A revocation marks
the device's record first, which refuses it, then writes the CRL: a CRL
that cannot be written then is said in the answer and undoes nothing.
The CRL lists every certificate revoked, or replaced since this version (one
replaced before it is refused by the devices' records alone): a revocation
keeps the ones it listed, and a CRL that does not load, which refuses
every device, is written again from the devices' records by the next
revocation or certificate made again for a device (revoking, through the
API, a device already revoked does it too). Revocations are one at a time
across processes where `fcntl` is (not on Windows). Starting uvicorn
directly skips the launcher and these checks.

### Layer 2 -- Encryption at rest

AES-256-GCM encryption for sensitive data fields, conversation messages
and their titles among them: a title is content, made from the words its
user typed -- the local model, when it refines one, only selects and
orders them, with a few function words -- or given by the user. SQLCipher
on all databases via the
centralized `safe_connect` wrapper. Per-user
encryption keys derived from the user's password via Argon2id KDF.
The audit chain is keyed with HMAC-SHA256, not signed (see below).
ML-DSA-65 post-quantum signatures cover backup exports, the model
provenance seal and Veilid device keys. `SecureBytes` uses mlock to
prevent key material
from being swapped to disk, with memset wipe on deallocation and a
SIGTERM handler that wipes all tracked key instances.

Closed exception (RS-01): the knowledge base -- the RAG store's chunks,
the memory's facts, the projects' and the documents' indexes -- is
encrypted at rest. Its texts, its metadata and its embedding vectors,
which give most of a short text back, live in one database of their own,
`opti_oignon/data/rag/vectors.db`, opened through `safe_connect` and
encrypted with SQLCipher like every other database, its journals with
it. ChromaDB, which kept them in plaintext files with no at-rest
encryption, is no longer installed with the platform, which reads
nothing of its stores: it only looks whether one is left. A ChromaDB
store left by an earlier version stays in plaintext until
`python3 -m opti_oignon.db_encryption migrate`, with the platform
stopped, moves its collections into the encrypted store: it reads
ChromaDB's database with the standard library, read-only, embeds each
text again with the model its namespace is queried with, which Ollama
must serve, proves each text paired with its vector, and commits all of
them or none. The command keeps the ChromaDB folders, with any item that
had no text to embed, in a `plaintext-<time>` folder, which git ignores,
to delete once the platform reads its knowledge base again; until then
the platform reads and writes nothing of the part of the knowledge base
that waits, so nothing deleted meanwhile comes back with the migration.
`python3 -m opti_oignon.db_encryption check` names each such store and
the command that moves it.

The Mycelium's stores (the search of a project's code and documents),
one `*.db` per corpus under `opti_oignon/data/mycelium/`, are opened
through `safe_connect` and encrypted with SQLCipher like every other
database, never in plaintext in Bulbe; one written in plaintext is taken
by the encryption command with the others. They keep no source's text,
but what they keep is drawn from it: each source's root, each file's
path (a document's id) and first sentence, a heading's path, the names a
part defines and names, the full-text index's tokens with their
positions, and each part's vector, which gives much of its text back. A
store is read through a read-only connection that writes nothing (beside
a store another program left in WAL, SQLite makes its `-wal` and `-shm`
on reading), and written by one pass at a time, in the rollback journal.
Forgetting a source merges the index whole and rewrites the file, and so
does the owner's indexing route after a pass that changed, renamed or
removed a file, so the dropped words leave the store's bytes once that
succeeds; the rewrite is owed from the transaction that dropped them
until it runs to its end, said when it did not, and done by the route's
next pass. In a store kept in plaintext, the rollback journals of those
passes and of the rewrite hold the dropped words and are removed, not
overwritten. The route never indexes the package's data folder or a
folder inside it. A citation the search reads back -- one a model hands
it -- is read through the store: under the root the store holds for the
source it names, only a file that source holds, of a source the reader
may read; never a root or a path the citation carries. The agent's
`mycelium_search` reads only the public and private sources rooted at
the folder its run's workspace was cloned from, read-only, a connection
opened again when the security mode changed; in Bulbe each call is
asked. The account wipe, for a single-user install's one user,
overwrites and removes every store; with several users, it names them as
not covered. The plaintext copies an encryption run keeps beside the
stores stay, as for every store. A project's files are indexed into
the project's own store, `project_<id>`: only the files its rows name,
private; a file removed from the project leaves that store, which is
rewritten; a project's deletion overwrites and removes its store, and
the sweep of every deletion removes the stores of projects that are
gone. The owner's route never writes a project's store. A project's
chat turn reads its store, the public and private sources only, never a
secret one; a store that cannot be read gives no passage, and the
knowledge base never answers in its place. The bridge that translates a
turn's question is made for that search alone and keeps no question past
it.

Known exception (MP-01): enforcement of the model provenance seal at
load is wired on the `llama_cpp` backend only. The Ollama backend (the
default) and the llama-server backend load model weights without an
integrity check, so under Bulbe the "only a verified model loads"
guarantee holds for `llama_cpp` but not for those two backends.
Extending the gate to the Ollama blob store is the remaining work, and
it is tracked as such. The startup security checklist reports the
active backend's provenance coverage as an advisory warning.


### Layer 3 -- Isolation and sandboxing

Sandbox Manager runs every LLM filesystem, shell, and code tool inside a
disposable bubblewrap (bwrap) sandbox. Containment: kernel namespaces
(unshare-net, unshare-pid, unshare-ipc/uts/cgroup), a seccomp-BPF syscall
denylist on every launch (fail-secure: if the filter cannot build and
seccomp is required, the launch is refused), resource caps via rlimits or a
transient cgroup scope (memory, process count, file size, CPU seconds, no
core dumps), a tmpfs size cap, a cleared environment, and read-only system
binds. If bwrap is unavailable, strict mode refuses execution rather than
falling back to the host.

The backend's own children start without the platform's secrets. The
application's entry takes the key file's passphrase and the master key
out of its environment before anything else runs, zeroes each where the
process was started with it -- what `/proc/<pid>/environ` shows a
process of the same user --, reads that back, and holds them in memory
locked where the system allows, wiped at exit; what a shell runs code
from (`BASH_ENV`, exported functions, an inherited `SHELLOPTS` or `PS4`)
leaves with them. Every program the backend starts directly -- the
sandbox's bubblewrap, the system tools of the security pages, a plugin's
worker -- is handed an environment built for it, and the ones that
inherit it -- the pool's workers, the multiprocessing tracker, the
compiler `find_library` runs -- start after the secrets and those
variables are gone. A secret that could not be zeroed refuses every
sandbox command and plugin that would run without bubblewrap, and the
startup checklist says so; while a secret is held the backend has no
core size to dump, every process start Python audits is checked as it is
made -- one whose environment or arguments hold a secret, or whose
environment names one of those shell variables, is refused --, and
Python code of its interpreter can put no secret back into its
environment, which the starts Python does not audit inherit. A census
guard checks every process start of the package against the same rule on
every commit, over the spellings it lists.

The hard-sandbox contract holds across the workspace cycle, clause
by clause: the sandbox stays a fully isolated, disposable environment (a
workspace is destroyed on demand, on TTL, or on conversation close); host
filesystem and network access stay zero by default; files enter only by
explicit copy-in (drag-and-drop upload or an allowlisted, symlink-safe host
clone -- $HOME plus configured roots, never /); results leave only after
human approval, refined into a diff model -- changes against the copy-in
baseline are approved individually, deletions are confirmed separately, the
apply writer is symlink-safe, and the applied set is hash-bound to the
reviewed diff. Auto-apply does not exist.

The optional sandbox network is default-off, per-workspace,
user-activated only (no configuration default, no tool surface, never
model-triggerable), and Daily-only at a fail-secure binding-layer gate: an
unset, unknown, or undeterminable mode is treated as Bulbe and refused. The
only shipped egress is a provision phase -- a server-built, hash-pinned
pip install into a workspace venv; task code never sees the network, and
the network is off again by construction after the run. Toggles, refusals,
and every provision run are audited (per-session log and hash-chain rows).

Plugin subprocess isolation runs each plugin in a separate process with
HMAC-signed JSON-RPC IPC over a socket pair the worker inherits, resource
limits the server installs before the worker starts (CPU, memory, file
descriptors, processes, file size; the worker refuses to serve without
them), and stdout/stderr capture. Whenever the sandbox runs under
bubblewrap, so does each worker: no network unless the manifest declares
`network_outbound`, and a filesystem holding only the interpreter, the
plugin's folder (read-only) and, for a plugin that may write, its private
data folder. Without bubblewrap, strict mode keeps plugins from starting.
Each worker additionally enforces a
host-package import boundary: a meta-path guard inside the worker
process rejects any import of the host application package with an
immediate, explicitly labelled isolation error, instead of letting such
an import stall on the worker's minimal environment. A plugin that
tries to reach back into the application from its worker fails fast and
its declared fallbacks engage; the standard library remains available,
since the process boundary itself is the isolation. The refusal is
deterministic by construction — it never depends on the environment
layout the worker happens to start with. The coding agent operates entirely inside the
sandbox; the apply phase is the only exit path and always requires explicit
human approval. Command validator with blocklists detects
base64-pipe-to-shell and write-then-execute patterns.

### Layer 4 -- LLM-specific defenses

RAG prompt injection defense via `RAGSanitizer`: pattern matching,
heuristic scoring, and confidence thresholds. Per-collection trust
levels control how much influence RAG context has on responses.
`augment_secure()` wrapper injects RAG context with injection attempt
audit logging. Search result sanitization strips potentially malicious
content. PII sanitizer removes emails, IPs, file paths, and hostnames
from outgoing web search queries. Search kill switch with circuit
breaker for anomaly detection.

### Layer 5 -- Audit and monitoring

SHA-512 hash-chain audit log where each entry includes the hash of its
predecessor (tamper-evident). Anchors are authenticated with a keyed
HMAC-SHA256 derived from the master encryption key -- NOT with a
post-quantum signature, and not with any asymmetric signature at all.

The distinction is load-bearing and is stated here rather than implied.
An HMAC is symmetric: the party that verifies holds the same secret as
the party that authenticates. The chain is therefore tamper-EVIDENT
against an attacker who does not hold the master key, and it is NOT
tamper-PROOF against one who does -- such an attacker can recompute
every anchor and produce a chain that verifies clean. Nor can an anchor
be verified by a third party, or by yourself on a machine that does not
already hold the key; there is no public half. Binding the chain to an
asymmetric key the host does not hold is the remaining work, and it is
tracked as such.

External anchoring via QR code export, signed
JSON for USB backup, and plain-text clipboard anchor. HMAC verification
on imported anchors. Startup security checklist with scoring API runs
on every backend start.

The Resource Governor is an availability control, not a
confidentiality or integrity boundary: it admits or refuses model loads
against measured VRAM/RAM, applies runtime backpressure, and manages
limits, failing open on missing measurement so it can never deny service
on its own uncertainty. Its surfaces are mode-free (identical in Daily
and Bulbe) and hold no secrets. Two advisory-only postures touch this
layer. The startup checklist gained a sixth check for the external-Ollama
limit configuration: it is advisory in every mode and can never block
startup -- a mismatch is a warning with actionable tips, an
externally-managed server degrades to "unknown" rather than a false
claim of enforcement. The optional in-process rlimits backstop is off by
default and carries an honest caveat: setrlimit is process-wide, so it
caps the entire Opti-Oignon process rather than the llama.cpp backend
alone, which is why admission accounting remains the primary control. The
host-bound cgroup recipe (scripts/ollama_cgroup_limits.sh) is print-only
reference material the operator applies deliberately; the application
never runs it. Eviction and config changes ride the signed audit chain.

### Layer 6 -- Automated security testing

LLM-powered red team engine tests Opti-Oignon's own defense layers
(not generic model benchmarking). 80 attack seeds across 8 categories:
injection, jailbreak, exfiltration, encoding bypass, privilege
escalation, resource abuse, PII leakage, sandbox escape. 9 obfuscation
strategies. 6 target adapters (RAGSanitizer, RAGAugmenter,
SearchSanitizer, PIISanitizer, Sandbox, Chat). Automated scheduling
with regression detection. Dependency vulnerability monitoring via
pip-audit with severity filtering.


## Security Modes

### Daily mode

Standard security with sensible defaults. Authentication can be relaxed
for localhost single-user convenience. All encryption and sandboxing
remain active. Suitable for local development.

### Bulbe mode

Maximum security configuration. Bulbe mode is a physical network
constraint, not a policy toggle -- it enforces localhost-only socket
binding at the OS level. The backend literally cannot accept connections
from external hosts. Additionally:

- Authentication is always enforced
- This machine's clients only: on every path but the health check, the
  sign-in and the security mode's own routes included, a client of
  another address or a request a proxy forwarded is refused, and its
  websocket closed before it is accepted (one it opened in Daily, at its
  first message once Bulbe began). Such a client may not enter Bulbe
  either -- the escalation answers 403, and the emergency stop stops
  without its drop to Bulbe, each refusal audited --, so that Bulbe is
  entered from where it can be left. A Bearer token is refused on every HTTP path but
  the health check, the sign-in routes and the schema (a websocket may
  still carry its token in its query); every cookie the API sets is
  SameSite=Strict, the session's included
- A second factor is asked at login of every account: a TOTP or a
  security key, a recovery code standing in for either; an account none
  of whose factors can be checked here is refused at login, and one that
  has none yet sets a TOTP up there before its first session (see
  docs/security/authentication.md)
- Web search is disabled
- Ollama bind guard blocks 0.0.0.0 exposure
- Per-turn RAM wipe of sensitive context
- Startup security checklist runs automatically with strict scoring

Conversation wipe is RAM-only by default: it zeroes in-memory buffers but does
not delete the persisted conversation rows, which remain SQLCipher-encrypted at
rest in Bulbe mode. Deleting stored history is a distinct operation; the
emergency-wipe endpoints accept an opt-in `purge_disk` flag for a full wipe that
also removes the persisted rows. The flag is off by default so an emergency RAM
wipe never destroys stored history unless explicitly requested.


## Deployment Recommendations

### Recommended: native installation

Native deployment on the host OS provides the strongest security
posture. It enables all six defense layers without compromise:

- Bubblewrap (bwrap) kernel namespaces for full sandbox isolation
- Direct socket-level localhost binding (Bulbe mode)
- LUKS full-disk encryption integration (advisory detection)
- No Docker daemon root trade-off

Full-disk encryption for RAG: the knowledge base is encrypted at rest
since RS-01 was closed, so LUKS is no longer required for it, and stays
advisable. What the platform decrypts in memory -- conversations,
retrieved chunks, the knowledge base's vectors, which its search keeps in
memory -- may be written to a swap device in plaintext: a swap kept in
memory (zram) or an encrypted swap keeps it off the disk. On Linux the
application detects LUKS and warns when it is absent.

### Optional: Docker deployment

Docker support is provided for convenience but is documented as
optional. Docker introduces trade-offs:

- The Docker daemon runs as root by default, which conflicts with
  Opti-Oignon's principle of minimal privilege
- Bubblewrap sandboxing is not available inside containers (falls back
  to tmpdir isolation)
- Bulbe mode socket binding is replaced by Docker network isolation
- Bulbe mode serves this machine's clients only, and a backend in a
  container sees every client at the bridge's address: from a browser,
  Bulbe can be neither entered nor used there

If Docker is used, the following mitigations are in place:

- All port bindings restricted to `127.0.0.1` (no `0.0.0.0`)
- Ollama container has no exposed host ports (internal network only)
- Backend runs as non-root user (`opti`, uid 1000) inside the container
- HEALTHCHECK on `/api/health` for liveness monitoring
- Podman rootless is suggested as an alternative to avoid Docker daemon
  root entirely

See `Dockerfile.backend` and `docker-compose.yml` for the full
configuration with inline security notices.


## Content Security Policy

CSP headers are applied to all HTTP responses via FastAPI middleware.
The policy enforces:

- No inline scripts (nonce-based execution)
- No `eval()` or equivalent
- `connect-src` restricted to localhost origins
- Currently in report-only mode for compatibility assessment


## Release Signing

Release archives can be signed with GPG for supply chain integrity
verification.

### Signing a release

```bash
# Sign an archive (auto-selects GPG key or specify --key)
./scripts/sign_release.sh opti-oignon-v2.2.0.zip

# Produces:
#   opti-oignon-v2.2.0.zip.sig    (detached GPG signature)
#   opti-oignon-v2.2.0.zip.sha256 (SHA-256 checksum)
```

### Verifying a release

```bash
# Import the public key (first time only)
gpg --import opti-oignon-release.pub

# Verify signature and checksum
./scripts/verify_release.sh opti-oignon-v2.2.0.zip

# Before trusting an imported key, check its fingerprint against the
# project's published fingerprint through an independent channel:
gpg --fingerprint --keyid-format=long

# Strict mode (fails on missing checksum file)
./scripts/verify_release.sh opti-oignon-v2.2.0.zip --strict
```

### Pinning the project key

A valid signature proves the archive matches a key in *your* keyring --
integrity, not identity. To make the verifier demand the project's key
specifically, record its full fingerprint (40 hex characters) in
`scripts/release_key.fpr`. When that file is present and no `--key` is
given, `verify_release.sh` refuses any signature from another key.
Obtain the fingerprint through an independent channel, never from the
archive being verified.

### CI/CD integration

The GitHub Actions release workflow (`.github/workflows/release.yml`)
supports GPG signing when the `GPG_PRIVATE_KEY` repository secret is
configured. Without the secret, releases are still created with SHA-256
checksums only. The workflow:

1. Validates that the git tag matches `__version__.py`
2. Builds the release archive
3. Signs with GPG if the secret is available
4. Generates SHA-256 checksums
5. Creates a GitHub Release with all artifacts


## Security Audit History

### Tooling-assisted audit cycle

Full codebase audit using pip-audit, npm audit, bandit, and a custom
18-check static scanner. 38 findings across 4 severity levels. All
critical and high findings remediated in the same cycle.

Key remediations:

- Dependency updates for pypdf, pyjwt, flask, requests, werkzeug
- Remaining f-string SQL patterns in analytics.py parameterized
- Cookie security flags enforced (HttpOnly; SameSite=Strict in Bulbe mode
  and for another machine's client)
- Path traversal guards on user-supplied file paths
- SSRF protections on user-controlled URL inputs

### Manual audit

4-round manual security audit during the hardening cycle.
22 findings, 20 fixed. Established the six-layer defense architecture.


## Vulnerability Reporting

If you discover a security vulnerability in Opti-Oignon, please
report it responsibly:

1. Do not open a public GitHub issue for security vulnerabilities
2. Email the maintainer with a description of the vulnerability,
   steps to reproduce, and potential impact
3. Allow reasonable time for a fix before public disclosure
4. Security fixes are prioritized and released as patch versions

The red team engine continuously tests the defense layers.
If you find a bypass that the red team does not detect, that is
especially valuable to report.


## Further Reading

- [Security Overview](docs/security/overview.md) -- defense-in-depth
  architecture summary
- [Bulbe Mode](docs/security/bulbe-mode.md) -- maximum security mode
- [Encryption](docs/security/encryption.md) -- AES-256-GCM, SQLCipher,
  SecureBytes, post-quantum signatures
- [Audit Chain](docs/security/audit-chain.md) -- hash-chain log,
  external anchoring, tamper detection
- [Authentication](docs/security/authentication.md) -- JWT, RBAC, 2FA,
  rate limiting
- [Red Team Guide](docs/redteam/running-audits.md) -- running LLM
  security audits
