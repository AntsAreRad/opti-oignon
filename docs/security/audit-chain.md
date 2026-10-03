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

When the backend starts, the audit log walks the chain and compares its
tip with the anchor file. A broken link, a truncation or an altered
anchor is logged as a warning or a critical error.

### On demand

These routes are under `/api/security` and require a valid session:

- `POST /audit-chain/verify` walks the chain and reports the first
  broken entry, if any
- `GET /audit-chain/status` reports the chain's state
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
each takes a few hundred bytes. The chain cannot be truncated without
the anchor detecting it. Old entries can be exported as CSV, but the
chain in the database must stay complete for verification to work.
