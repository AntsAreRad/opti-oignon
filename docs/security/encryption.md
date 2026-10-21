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
`opti_oignon/data/`, of the knowledge base's `rag/` folder -- followed
when it is a symbolic link --, and the sandbox's audit log where its
manager opens it: `audit_db_path` joined to `workspace_base`, from
`config/sandbox.yaml`. A plugin's own databases (`plugin_data/<plugin>/`)
are left in plaintext: the plugin opens them in its own process, without
the platform's key. Any other `*.db` below the data directories -- a file
added to a project, say -- is named and left as it is, and so is every
other file there that begins as a plaintext SQLite database --
ChromaDB's `chroma.sqlite3`, say --, and every such file, a `*.db`
included, below `RAG_HOME` when it is set to an absolute path (a relative
one is said, not walked): `check` and `encrypt` name each one, never
encrypted by this command, and name what they could not check -- a
symbolic link, never followed, and an entry they could not read.

A contract reads the package's sources: every binding of a directory of
databases (`DATA_DIR`, `DB_DIR` or `STORE_DIR`, with or without a leading
`_` or `DEFAULT_`) made in a module or a class, in every branch, is one
of these places or their `rag/` folder, and no environment variable moves
one; every ChromaDB directory and `RAG_HOME` lies below a place, and a
directory an environment variable moves counts only when the command
follows that variable -- it follows `RAG_HOME`. A binding made inside a
function is not read.

It refuses, changing nothing -- but each plaintext database's
write-ahead log folded into it -- while another run is under way, while
a process holds a database or runs the platform (read from `/proc`; the
shell that started the command, naming it, is not counted), when a key
file exists and does not load here (set `OPTI_KEYFILE_PASSPHRASE`), when
`OPTI_ENCRYPTION_KEY` is set and is no key -- 32 bytes, base64-encoded
with its padding: 44 characters ending in a single `=` -- (the platform
would pass over it to the key file), when a database does not open
under the key, when a
directory of databases is there and cannot be read or searched, when
the sandbox's audit log is configured below wherever the platform starts
or lies at the manager's default path while its configuration names
another (the manager opens that one only when its configuration does not
build), or when an interrupted run left something to look at first.
`check` says what it can see of these -- the processes, the key, the
directories it cannot read, the sandbox's log, what an interrupted run
left --; it does not see another run under way.

A new key is held in memory while every database is copied, encrypted,
beside itself; each copy is proved before
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
temp -- and what to do with it; `check` names them too. Git ignores
each of these where the platform keeps them, as it ignores the key file
and the databases of the `plaintext-<time>` folders.

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
run: setup then takes the existing key and makes no new one. The
settings say so: with a key here -- a key file, or
`OPTI_ENCRYPTION_KEY` -- they offer only to enable it and ask for no
passphrase, the server saying why when it cannot open it; a success
says whether the key was the one already here or made, where, whether
its key file is wrapped (the answer's `key`, `existing` or `made`, and
`wrapped`), and how many databases are still kept in plaintext
(`plaintext_stores`); a refusal shows the server's reason.

The server never does this itself: it holds its databases open, and a
database moved under an open connection loses what that connection
writes next. `POST /api/security/db-encryption/migrate` answers 409,
and `POST /api/security/encryption/setup` makes no key -- 409 -- while
a database is kept in plaintext.

The knowledge base -- the RAG store's chunks, the memory's facts, the
projects' and the documents' indexes, their texts, metadata and vectors --
lives in `opti_oignon/data/rag/vectors.db`, a database of the `rag/`
folder that `encrypt` takes like every other. ChromaDB, which kept it
before in plaintext files, is no longer read. With the platform stopped,
`python3 -m opti_oignon.db_encryption migrate` moves every ChromaDB store
below the data directories -- `rag/chroma_v2/`, `rag/chroma_db/` (below
`RAG_HOME` when it is set) and `project_chroma/`, each into a namespace of
its own -- into it, all of them or none: each collection is read through
chromadb, its telemetry off, and written in one transaction, which commits
only once every item's document, metadata and vector read back as they
were, and once the distances ChromaDB measures, on a sample of each
collection, are found again in the space it measured them in. Then each
ChromaDB folder is kept in a `plaintext-<time>` folder beside it, which
git ignores, to delete once the platform reads its knowledge base again.
Until a folder has moved, its namespace of the store is closed: the
platform reads and writes nothing of it, each call refused by name with
the command, so nothing it deletes meanwhile comes back with the
migration. A folder waits when it holds ChromaDB's database, when it, or
that file, is a symbolic link, or when it cannot be looked at; `migrate`
reads the same test, and refuses while a folder of the platform's modules
waits in a way it cannot move -- a link, a folder it cannot look at,
ChromaDB's store in a folder its walk could not list, the documents' and
the memory's folder under a relative `RAG_HOME` (an empty one included,
which the platform reads as its own directory) --, so the platform never
waits on what the command says it has nothing to move. It runs where the
platform runs: a `RAG_HOME` the platform has and the command's shell has
not names a folder the command never sees, and the platform names it as it
waits. A vector of the memory is moved only when an active fact of the
memory holds it with the very text the vector was made of: a fact deleted
or rewritten while the migration waited keeps no stale vector -- a run
that resumes one that committed and could not move its folders removes,
for each fact a folder still names, a copy the store holds with a text the
fact no longer reads, whichever folder it came from, and keeps one with
the text the fact reads, whoever wrote it --, and the active facts left
without a vector of their text -- those too, and those added meanwhile --
are counted and found by their words. It refuses, writing nothing and
moving nothing, while another run of `encrypt` or `migrate` is under way,
while a process holds the store or a ChromaDB file or runs the platform --
read again before the commit --, when chromadb does not import or does not
read a store, when a key file does not load here, when the store is in
plaintext while a key loads (`encrypt` first) or does not open, when the
memory's facts cannot be read, when a collection's space or dimension
differs from that of a collection the store holds with items in it -- an
empty one takes the source's --, and when a collection does not read
whole; an item the store holds already is kept, and counted, and a store
or a folder the run made is removed again. chromadb itself rewrites some
of its own files as it reads them. A signal before the commit stops the
run, nothing written; a folder that cannot be moved once the store has
committed is said, with what the store took from every folder, the command
ends with 3, and the next run moves it. `check` names the command beside
each ChromaDB store it moves; a ChromaDB store in another folder is named
and left.


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
