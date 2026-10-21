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

#### Turning database encryption on

Once a key loads, every connection the platform opens keys its
database, so a database still kept in plaintext no longer opens. The
databases are therefore encrypted together, with the platform stopped
-- the server, and every `oo` process:

```bash
python3 -m opti_oignon.db_encryption check     # each database's state
python3 -m opti_oignon.db_encryption encrypt   # all of them, or none
```

`encrypt` takes every `*.db` of the repository's `data/` and of
`opti_oignon/data/`, of the knowledge base's `rag/` folder, and the
sandbox's audit log where `config/sandbox.yaml` puts it. A plugin's own
databases (`plugin_data/<plugin>/`) are left in plaintext: the plugin
opens them in its own process, without the platform's key. Any other
`*.db` below the data directories -- a file added to a project, say --
is named and left as it is.

It refuses, changing nothing -- but each plaintext database's
write-ahead log folded into it -- while another run is under way, while
a process holds a database or runs the platform (read from `/proc`; the
shell that started the command, naming it, is not counted), when a key
file exists and does not load here (set `OPTI_KEYFILE_PASSPHRASE`), when
a database does not open under the key, or when an interrupted run left
something to look at first. A new key is held in memory while every
database is copied, encrypted, beside itself; each copy is proved before
anything moves -- its integrity checked, its objects, each table's rows
in order and a digest of them, its `user_version`, `application_id` and
`auto_vacuum`, its file mode. (The implicit row ids of a table without
an `INTEGER PRIMARY KEY` may be renumbered, in order; the platform never
refers to them from another table.) Only then is the key written to its
key file -- wrapped under `OPTI_KEYFILE_PASSPHRASE` when it is set --
and each database switched: its plaintext linked into a
`plaintext-<time>` folder (mode 0700) inside its directory, its
encrypted copy taking its name in one rename.

SIGTERM, SIGHUP and Ctrl-C are noted -- the command says so at once --
and the run stops at its next safe point -- before a copy, before the
processes are read again, before the key is written, before a database
is switched -- putting back every database it switched and removing the
key file it wrote; one that lands once the last database's switch has
begun lets the run finish, and is said. A database is put back before
its plaintext link goes: one that cannot be put back keeps its
plaintext, its folder and the key file, and the run says where each is.
A run killed outright (`kill -9`, `Ctrl-\`) still leaves every database
whole under its name, plaintext or encrypted under the key file: run the
command again before starting the platform. It finishes, or first names
what the killed run left -- a second name for a plaintext, a key file's
temp -- and what to do with it.

The command ends with 0 once every database is encrypted, 1 when it
refused and changed nothing, and 3 when a database could not be put
back. A directory it could not sync to the disk once every database was
switched is said: the switch stands; run `sync`.

The command says where the key file is. Every database now opens only
with it: keep a copy of it somewhere safe, off the machine and out of
any repository, before deleting the `plaintext-<time>` folders -- until
then they are the plaintext. `check` lists every such folder, and
`encrypt` names those an earlier run left too. A key taken from
`OPTI_ENCRYPTION_KEY`, or a key file wrapped under
`OPTI_KEYFILE_PASSPHRASE`, needs that variable set wherever the platform
starts. Field-level encryption and the sealing of the run journal follow
when encryption is set up in the security settings once the command has
run: setup then takes the existing key and makes no new one.

The server never does this itself: it holds its databases open, and a
database moved under an open connection loses what that connection
writes next. `POST /api/security/db-encryption/migrate` answers 409,
and `POST /api/security/encryption/setup` makes no key -- 409 -- while
a database is kept in plaintext.

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
- The [run journal](../agent/run-journal.md), when it is on: each line
  sealed under the journal's own subkey of the master key (a fresh nonce
  per line); with encryption off, Bulbe keeps none of it

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
