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
- Tampering with the audit log (hash-chain with post-quantum signatures)
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
limiting with exponential lockout. CSRF double-submit cookie on all
state-changing endpoints.

### Layer 2 -- Encryption at rest

AES-256-GCM encryption for sensitive data fields. SQLCipher on all
databases via the centralized `safe_connect` wrapper. Per-user
encryption keys derived from the user's password via Argon2id KDF.
ML-DSA-65 post-quantum signatures on audit chain entries (Ed25519
fallback). `SecureBytes` class uses mlock to prevent key material
from being swapped to disk, with memset wipe on deallocation and a
SIGTERM handler that wipes all tracked key instances.

Known exception (RS-01): the RAG vector store (ChromaDB, under
`data/chroma_v2/`) is NOT encrypted at rest. ChromaDB has no native
at-rest encryption, so the ingested chunk text and embedding vectors
are stored in plaintext SQLite and index files. The RAG metadata
database (collections, documents, citations) is encrypted via
`safe_connect`, but the vector store is not. For a corpus of
sensitive documents this is a confidentiality gap against a second
local user, a copied backup, or a disk read without full-disk
encryption. Full-disk encryption (LUKS) is therefore a deployment
requirement for the RAG corpus (see Deployment Recommendations).
Application-layer encryption of the vector store (encrypt the chunk
text before storing, decrypt on retrieval) is a planned cycle
(ROADMAP_POST_S183, RAG-at-rest cycle).

### Layer 3 -- Isolation and sandboxing

Sandbox Manager runs every LLM filesystem, shell, and code tool inside a
disposable bubblewrap (bwrap) sandbox. Containment (S209): kernel namespaces
(unshare-net, unshare-pid, unshare-ipc/uts/cgroup), a seccomp-BPF syscall
denylist on every launch (fail-secure: if the filter cannot build and
seccomp is required, the launch is refused), resource caps via rlimits or a
transient cgroup scope (memory, process count, file size, CPU seconds, no
core dumps), a tmpfs size cap, a cleared environment, and read-only system
binds. If bwrap is unavailable, strict mode refuses execution rather than
falling back to the host.

The S73/S74 contract holds across the workspace cycle (S209-S213), clause
by clause: the sandbox stays a fully isolated, disposable environment (a
workspace is destroyed on demand, on TTL, or on conversation close); host
filesystem and network access stay zero by default; files enter only by
explicit copy-in (drag-and-drop upload or an allowlisted, symlink-safe host
clone -- $HOME plus configured roots, never /); results leave only after
human approval, refined into a diff model -- changes against the copy-in
baseline are approved individually, deletions are confirmed separately, the
apply writer is symlink-safe, and the applied set is hash-bound to the
reviewed diff. Auto-apply does not exist.

The optional sandbox network (S213) is default-off, per-workspace,
user-activated only (no configuration default, no tool surface, never
model-triggerable), and Daily-only at a fail-secure binding-layer gate: an
unset, unknown, or undeterminable mode is treated as Bulbe and refused. The
only shipped egress is a provision phase -- a server-built, hash-pinned
pip install into a workspace venv; task code never sees the network, and
the network is off again by construction after the run. Toggles, refusals,
and every provision run are audited (per-session log and hash-chain rows).

Plugin subprocess isolation runs each plugin in a separate process with
HMAC-signed JSON-RPC IPC, resource limits (CPU, memory, file descriptors),
and stdout/stderr capture. The coding agent operates entirely inside the
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
predecessor (tamper-evident). Entries are signed with ML-DSA-65
post-quantum signatures. External anchoring via QR code export, signed
JSON for USB backup, and plain-text clipboard anchor. HMAC verification
on imported anchors. Startup security checklist with scoring API runs
on every backend start.

The Resource Governor (S221-S227) is an availability control, not a
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
- 2FA is mandatory
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

Full-disk encryption requirement for RAG: if the RAG corpus holds
sensitive documents, LUKS full-disk encryption is required (not merely
advisory), because the ChromaDB vector store keeps ingested chunk text
and vectors in plaintext at rest (RS-01). On Linux the application
detects LUKS and warns when it is absent.

### Optional: Docker deployment

Docker support is provided for convenience but is documented as
optional. Docker introduces trade-offs:

- The Docker daemon runs as root by default, which conflicts with
  Opti-Oignon's principle of minimal privilege
- Bubblewrap sandboxing is not available inside containers (falls back
  to tmpdir isolation)
- Bulbe mode socket binding is replaced by Docker network isolation

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
./scripts/sign_release.sh opti-oignon-v3.3.0.zip

# Produces:
#   opti-oignon-v3.3.0.zip.sig    (detached GPG signature)
#   opti-oignon-v3.3.0.zip.sha256 (SHA-256 checksum)
```

### Verifying a release

```bash
# Import the public key (first time only)
gpg --import opti-oignon-release.pub

# Verify signature and checksum
./scripts/verify_release.sh opti-oignon-v3.3.0.zip

# Strict mode (fails on missing checksum file)
./scripts/verify_release.sh opti-oignon-v3.3.0.zip --strict
```

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

### S155-S156 audit cycle (v3.2.4-v3.2.5)

Full codebase audit using pip-audit, npm audit, bandit, and a custom
18-check static scanner. 38 findings across 4 severity levels. All
critical and high findings remediated in S156. Detailed results in
[docs/SECURITY_AUDIT_S155.md](docs/SECURITY_AUDIT_S155.md).

Key remediations:

- Dependency updates for pypdf, pyjwt, flask, requests, werkzeug
- Remaining f-string SQL patterns in analytics.py parameterized
- Cookie security flags enforced (HttpOnly, SameSite=Strict)
- Path traversal guards on user-supplied file paths
- SSRF protections on user-controlled URL inputs

### S136 manual audit (v3.0.0)

4-round manual security audit during the S126-S136 hardening cycle.
22 findings, 20 fixed. Established the six-layer defense architecture.


## Vulnerability Reporting

If you discover a security vulnerability in Opti-Oignon, please
report it responsibly:

1. Do not open a public GitHub issue for security vulnerabilities
2. Email the maintainer with a description of the vulnerability,
   steps to reproduce, and potential impact
3. Allow reasonable time for a fix before public disclosure
4. Security fixes are prioritized and released as patch versions

The red team engine (S147-S148) continuously tests the defense layers.
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
- [Security Audit Report](docs/SECURITY_AUDIT_S155.md) -- S155 findings
