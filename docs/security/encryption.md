# Encryption

## Encryption at rest

Opti-Oignon encrypts sensitive data at rest using industry-standard
algorithms. All encryption and decryption happens locally -- keys never
leave your machine.


### Database encryption (SQLCipher)

When SQLCipher is available, all SQLite databases are encrypted
transparently using AES-256 in CBC mode. This includes:

- Conversation history
- User accounts and session data
- Audit chain entries
- Plugin configuration
- RAG metadata

Database access uses the `safe_connect` wrapper, which centralizes
SQLCipher configuration (cipher page size, KDF iterations, WAL mode)
and ensures consistent encryption settings across all modules.

Without SQLCipher, databases are stored unencrypted. The startup
security checklist reports this as a warning.


### Field-level encryption (AES-256-GCM)

Sensitive fields (API keys, tokens, user secrets) are encrypted
individually using AES-256-GCM with per-field random nonces. This
provides both confidentiality and integrity verification.

The encryption module supports:

- Key derivation from passwords via Argon2id
- Key wrapping for multi-user scenarios
- Authenticated encryption (tamper detection)


### Key storage (SecureBytes)

Encryption keys in memory are stored using the `SecureBytes` class,
which uses `mlock` to prevent the operating system from swapping
key material to disk. SecureBytes instances are automatically zeroed
when garbage collected.


## Post-quantum signatures (ML-DSA-65)

The audit chain uses ML-DSA-65 (formerly CRYSTALS-Dilithium) for
digital signatures. This is a NIST-standardized post-quantum signature
scheme, meaning it remains secure against both classical and quantum
computer attacks.

ML-DSA-65 signatures are used for:

- Signing audit chain entries
- Release artifact verification (GPG + ML-DSA-65 dual signing)
- Anchor export verification

The implementation uses `liboqs-python`. When liboqs is not available,
the system falls back to Ed25519 signatures with a warning.


## Password hashing

User passwords are hashed using **bcrypt** with a configurable work
factor. Password verification uses constant-time comparison to prevent
timing attacks.

For key derivation (encrypting databases, deriving encryption keys
from passwords), **Argon2id** is used with memory-hard parameters
that resist GPU-based attacks.


## LUKS disk encryption

Opti-Oignon checks whether the underlying filesystem uses LUKS
full-disk encryption. This check is **advisory only** -- it provides
actionable tips if LUKS is not detected but never blocks startup, even
in Bulbe mode.

LUKS status is reported in the startup security checklist and the
security health endpoint.
