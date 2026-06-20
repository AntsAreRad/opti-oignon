# Security Overview

## Defense-in-depth

Opti-Oignon follows a defense-in-depth model with six independent
security layers. Each layer operates independently, so compromising one
does not bypass the others.

The security architecture follows **Kerckhoffs's principle**: security
derives from keys and human factors, never from code obscurity. The
project is designed to be open source with no security-through-obscurity
assumptions.


## The six defense layers

1. **Authentication and authorization** -- deny-by-default middleware,
   JWT session cookies, RBAC with per-user data isolation, optional 2FA
   (WebAuthn/FIDO2 + TOTP). See [Authentication](authentication.md).

2. **Encryption at rest** -- AES-256-GCM for sensitive data, SQLCipher
   for database encryption, `SecureBytes` for mlock'd key storage in
   memory. See [Encryption](encryption.md).

3. **Sandbox isolation** -- bubblewrap kernel namespaces for LLM tools
   and plugins (unshare-net, unshare-pid, read-only binds), command
   validator with blocklists, tempdir fallback when bwrap is unavailable.

4. **Network hardening** -- Bulbe mode enforces localhost-only socket
   binding, network bind guard monitors Ollama configuration, CSP
   headers in report-only mode. See [Bulbe Mode](bulbe-mode.md).

5. **Audit chain** -- hash-chain audit log with tamper detection,
   ML-DSA-65 post-quantum signatures, QR code and signed JSON export
   for external verification. See [Audit Chain](audit-chain.md).

6. **Automated security testing** -- LLM-powered red team engine that
   tests Opti-Oignon's own defense layers, dependency vulnerability
   monitoring, automated security scheduling. See the
   [Red Team Guide](../redteam/running-audits.md).


## Security modes

| Mode | Description | Use case |
|------|-------------|----------|
| Daily | Standard security, authentication optional for localhost | Local development |
| Bulbe | Maximum security, all layers enforced | Any networked use |

In **Daily mode**, the backend starts without requiring authentication
when accessed from localhost. This is convenient for single-user local
development but not suitable for any networked deployment.

In **Bulbe mode**, all six defense layers are enforced. Authentication
is required, Ollama is verified to bind to localhost only, and the
startup security checklist runs automatically. See
[Bulbe Mode](bulbe-mode.md) for details.


## Startup security checklist

On every startup (and on demand via the API), the backend runs a
security checklist that verifies:

- Authentication configuration
- Encryption status (SQLCipher availability, key storage)
- Sandbox availability (bwrap detected)
- Network binding (Ollama localhost check)
- Audit chain integrity
- LUKS disk encryption (advisory only, never blocks startup)
- Dependency vulnerability status
- Red team resistance score (if red team data is available)
- CSP middleware status

Results are available at `GET /api/security/health` and displayed in
the settings dashboard.
