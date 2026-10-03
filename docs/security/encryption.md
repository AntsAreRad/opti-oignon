# Encryption

## Encryption at rest

Opti-Oignon encrypts sensitive data at rest. All encryption and
decryption happens locally -- keys never leave your machine.


### Database encryption (SQLCipher)

Databases are opened through the `safe_connect` wrapper, which routes
each connection through the encrypted connection factory when SQLCipher
is installed. The factory applies SQLCipher 4 settings on every
connection: a key derived from the master encryption key, a 4096-byte
cipher page, HMAC-SHA512 page authentication and 256,000 KDF
iterations. Pages are encrypted with AES-256 in CBC mode, the SQLCipher
default. Encrypted stores include:

- Conversation history
- User accounts and session data
- Audit chain entries
- Plugin index, manifests and reviews
- RAG metadata (collections, documents, citations)

When the encryption module or SQLCipher is missing, Bulbe mode refuses
to open a database in plaintext. Daily mode opens it in plaintext and
logs a warning once per process; the startup security checklist does
not report it.

The RAG vector store (ChromaDB) is not encrypted at rest: chunk text and
embedding vectors are stored in plaintext files. Full-disk encryption is
required for a sensitive corpus; see `SECURITY.md`.


### Field-level encryption (AES-256-GCM)

Some secrets are encrypted individually with AES-256-GCM: a fresh
random 96-bit nonce for every encryption and a 128-bit tag that detects
tampering. This covers:

- Two-factor authentication secrets
- Note attachments (one subkey per attachment, streamed in
  authenticated frames)
- Backup archives
- The device's Veilid signing key

Keys are derived from passphrases with Argon2id, or with PBKDF2-SHA256
at 600,000 iterations when argon2-cffi is not installed. Each user's
encryption subkey is derived from that user's password with Argon2id.


### Key storage (SecureBytes)

Encryption keys in memory are held in `SecureBytes`, which locks their
pages with `mlock` so that the operating system does not swap key
material to disk, and overwrites the buffer with zeros when the object
is deleted or wiped explicitly. A SIGTERM handler wipes every tracked
key. Where `mlock` is unavailable, the buffer is still zeroed.


## Signatures

### Post-quantum signatures (ML-DSA-65)

ML-DSA-65 (FIPS 204, formerly CRYSTALS-Dilithium) is a post-quantum
signature scheme, designed to resist attacks by both classical and
quantum computers. Through `liboqs-python`, it covers:

- Backup exports
- The model provenance seal
- Veilid device keys and the records they sign

Without a usable liboqs, nothing falls back to a classical signature.
Veilid record signing refuses. Where a post-quantum signature is
required (the operator asked for one, or the mode demands it), a backup
export and a provenance seal are refused. Elsewhere a backup is
exported unsigned, still encrypted and authenticated with AES-256-GCM,
and the provenance seal is an HMAC-SHA512 MAC, which anyone holding the
key can forge and nobody else can verify. The startup security
checklist reports a missing primitive.

### What is not signed

- **The audit chain** is a SHA-512 hash chain whose anchor is keyed with
  HMAC-SHA256 under a key derived from the master encryption key. It is
  tamper-evident against anyone without that key, but it is not a
  signature: there is no public half, and a holder of the key can
  rewrite it. See [Audit Chain](audit-chain.md).
- **Releases** are signed with GPG only, when the release workflow holds
  a signing key; otherwise a release ships with SHA-256 checksums alone.
  There is no ML-DSA release signature and no Ed25519 fallback.


## Password hashing

User passwords are hashed with **bcrypt**, 12 rounds by default and
configurable in `config/auth.yaml`. Password verification uses bcrypt's
constant-time comparison.

For key derivation from passwords, **Argon2id** is used with
memory-hard parameters that resist GPU-based attacks.


## LUKS disk encryption

Opti-Oignon checks whether the underlying filesystem uses LUKS
full-disk encryption. This check is **advisory only** -- it provides
actionable tips if LUKS is not detected but never blocks startup, even
in Bulbe mode.

LUKS status is reported in the startup security checklist and the
security health endpoint.
