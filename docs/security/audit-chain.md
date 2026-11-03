# Audit Chain

## How it works

The audit chain is a tamper-evident log of security-relevant events,
stored in `data/audit_chain.db` (SQLCipher when available). Each entry
carries the SHA-512 hash of its own fields and of the previous entry's
hash, so a modification to a past entry breaks the chain at that entry.

A separate anchor file records the number of entries and the hash of the
last one, so that truncating the chain, or replacing it with a fresh
one, is detected too. The anchor is authenticated with HMAC-SHA256 under
a key derived from the master encryption key. Without a master key the
anchor is a plain SHA-256 checksum, which only detects accidental
corruption.


## What is logged

Among the events the chain records:

- Account events (registration, login, password change, user deletion,
  project sharing)
- Security mode changes and rejected non-local requests
- Tool call approvals and sandbox events (provisioning, network toggle)
- Emergency stops and conversation wipes
- Resource governor decisions
- Veilid synchronization, remote inference and lifecycle events
- TLS setup and skill changes


## Keyed, not signed

The chain is keyed with HMAC-SHA256, not signed: no ML-DSA-65 or
Ed25519 signature is involved. An HMAC is symmetric, so whoever verifies
holds the same secret as whoever wrote. The chain is therefore
tamper-evident against an attacker who does not hold the master key, and
not tamper-proof against one who does: such an attacker can recompute
every hash and the anchor, and produce a chain that verifies clean. See
`SECURITY.md`, layer 5.


## Verification

### On startup

When the backend starts, the audit log walks the chain and compares it
with the anchor file: the chain must reach the anchored height and hold
the anchored hash there. A broken link, or a walk that fails, is logged
as a warning, and the anchor is compared all the same. What the
anchor comparison sees is also written into the chain itself, as an
entry `audit_anchor_<verdict>` from `signed_audit_log`, and only then
does the anchor move, past that entry, so the verdict outlives the
anchor it was read from:

- Critical under a master key, a warning without one (the anchor is then
  advisory only): `tampered` (the anchor's tag fails; without a key,
  `checksum_mismatch`), `truncated` (the chain is shorter than the
  anchor, or was replaced by an empty one), `rewritten` (the entry at the
  anchored height is not the anchored hash, even when the chain
  verifies).
- A warning: `missing` (no anchor beside entries: a truncation before it
  was lost cannot be seen), `unrecognized` (not an anchor this version
  writes: legacy, foreign or damaged), `other_key` (written under another
  key: a key change, a relocated install, or an anchor made without the
  key; its numbers are said, not believed), `not_checked` (the anchor
  could not be looked at -- not a regular file, a link at its name, a
  file longer than any anchor, a read error -- or the comparison itself
  failed; a failed comparison is said once, and runs again at each
  connection until it concludes, the anchor left as it was meanwhile).

An anchor that verifies, and a chain grown past it, write nothing. A
process where a key is configured but does not load, beside an anchor
written under a key -- a tool run without `OPTI_KEYFILE_PASSPHRASE` --
writes nothing either, and leaves that anchor as it is for a process
that holds the key; its status says `key_not_loaded`. Where no key is
configured at all, such an anchor is another key's, said as
`other_key`. A process whose check ran under a key never writes an
anchor without one.

While the backend runs, each entry it writes first checks that the
chain still holds what it last anchored: a chain truncated, put back
from an older copy or rewritten meanwhile is said, as above, right after
that entry, and the anchor moves only past the verdict.

Under a key, the anchor names a truncation only while it is itself
intact: an anchor deleted, garbled, replaced or forged without the key
turns a truncation into one of the warnings above, which do not name it.
A chain and an anchor both deleted look like a first run, and an older
anchor put back with a chain truncated to its height passes: neither is
seen on the machine (see External anchors). The anchor is written whole
-- a new file, mode 0600, moved into place -- and never through a link
at its name.

### On demand

These routes are under `/api/security` and require a valid session:

- `POST /audit-chain/verify` walks the chain and reports the first
  broken entry, if any
- `GET /audit-chain/status` reports the chain's state, and under
  `anchor` what the startup check saw of the anchor
- `GET /audit-chain/export` downloads the full chain as CSV

### External anchors

The current anchor can be exported in three forms:

- **JSON file** -- the anchor with its HMAC tag
- **QR code** -- a PNG of the same anchor as compact JSON
- **Clipboard text** -- the same fields in readable form

`POST /audit/verify-anchor` checks a re-presented anchor: its tag under
the same key, the chain's integrity, and that the chain at the anchored
height still ends at the anchored hash. Growth after the anchor is fine;
truncation and rewrites are not.

An anchor kept off the machine shows later whether the chain was
rewritten behind your back. Checking it needs the same master key, so a
third party cannot verify it, and neither can a machine without that
key.


## Retention

Entries are kept indefinitely: nothing in the code deletes them, and
each takes a few hundred bytes. A truncated chain is seen at the next
start, or at the running backend's next entry, and named while its
anchor is intact (see On startup); whoever
holds the master key can truncate it unseen, and without a master key
anyone can. Old entries can be exported as CSV, but the
chain in the database must stay complete for verification to work.
