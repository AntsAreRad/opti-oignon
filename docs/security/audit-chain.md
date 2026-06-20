# Audit Chain

## How it works

The audit chain is a tamper-evident log of security-relevant events.
Each entry contains a SHA-256 hash of the previous entry, forming a
hash chain similar to a blockchain. Any modification to a past entry
breaks the chain and is detected automatically.


## What is logged

The audit chain records:

- Authentication events (login, logout, failed attempts)
- Configuration changes (security settings, plugin toggles)
- Admin actions (user creation, role changes)
- Sandbox events (tool execution, apply/reject decisions)
- Encryption key operations (creation, rotation)
- Red team audit results
- Security checklist outcomes


## Signatures

Each audit chain entry is signed using ML-DSA-65 post-quantum
signatures (or Ed25519 as fallback). This ensures that entries cannot
be forged even by an attacker who gains database write access.

The signing key is stored in SecureBytes (mlock'd memory) and derived
from the admin password via Argon2id.


## Verification

### Automatic verification

The startup security checklist verifies the entire audit chain on
every boot. Any broken links or invalid signatures are reported in
the health endpoint.

### Manual verification

```bash
# Via the API
curl http://localhost:8000/api/security/audit/verify

# Via the CLI (if backend is running)
oo redteam status  # includes chain integrity in output
```

### External verification

The audit chain supports export to external formats for independent
verification:

- **QR code export** -- individual entries encoded as QR codes for
  offline verification or physical archival
- **Signed JSON export** -- full chain exported as signed JSON with
  embedded ML-DSA-65 signature for verification by external tools
- **Clipboard anchor** -- copy a chain anchor (hash + signature) to
  clipboard for pasting into external records

These exports allow you to verify the integrity of the audit chain
without trusting the Opti-Oignon software itself.


## Retention

Audit chain entries are retained indefinitely by default. The chain
cannot be truncated without breaking verification -- this is by design.
Storage impact is minimal (each entry is a few hundred bytes).

Old entries can be exported and archived, but the chain in the database
must remain complete for verification to work.
