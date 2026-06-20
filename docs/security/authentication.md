# Authentication

## Authentication model

Opti-Oignon uses a **deny-by-default** authentication model. The auth
middleware intercepts every API request and verifies the JWT session
cookie before allowing access. Only explicitly marked public endpoints
(health check, login, static assets) bypass authentication.

In Daily mode on localhost, authentication can be relaxed for
convenience. In Bulbe mode, authentication is always enforced.


## User accounts

User accounts are stored locally with bcrypt-hashed passwords. Each
account has a role that determines access:

| Role | Permissions |
|------|-------------|
| admin | Full access, user management, security settings |
| user | Chat, RAG, plugins, own conversation history |
| viewer | Read-only access to shared resources |


## Two-factor authentication (2FA)

Opti-Oignon supports two 2FA methods, which can be used independently
or together:

### WebAuthn / FIDO2

Hardware security key authentication using the WebAuthn standard.
Supports USB keys (YubiKey, SoloKeys), platform authenticators
(fingerprint, face ID), and passkeys.

Setup:

1. Go to **Settings > Security > 2FA**
2. Click **Register Security Key**
3. Follow the browser prompt to touch your key
4. Name the key for identification

Multiple keys can be registered for backup purposes.

### TOTP (Time-based One-Time Password)

Software-based 2FA compatible with any authenticator app (Google
Authenticator, Authy, KeePassXC, etc.).

Setup:

1. Go to **Settings > Security > 2FA**
2. Click **Enable TOTP**
3. Scan the QR code with your authenticator app
4. Enter the verification code to confirm

A validated TOTP code is single-use within its validity window: the consumed
time-step is recorded, so a code that was just used cannot be replayed during the
30-second window in which it would otherwise still verify.

### Recovery codes and app passwords

Recovery codes (one-time, shown once at generation) and app-specific passwords
(for CLI use) are never stored in clear: only a keyed hash is persisted. The hash
is an HMAC-SHA256 under a subkey derived from the master encryption key on its own
domain, so the master key is the only secret (Kerckhoffs). Without a master key
configured, the at-rest protection of these hashes rests on SQLCipher.

Migration (one-time, automatic). Recovery codes and app passwords created before
this re-keying were hashed under the previous scheme. They are not rejected:

- A pre-existing **app password** is transparently re-hashed to the new scheme the
  next time it is used (the plaintext is available at that moment).
- Pre-existing **recovery codes** cannot be re-hashed in place (their plaintext is
  not stored). After any successful 2FA, the 2FA status reports
  `recovery_reissue_required: true` until you regenerate your recovery codes; the
  regenerated set is keyed under the new scheme. Regenerate them from
  **Settings > Security > 2FA** to re-key.


## Session management

Sessions use JWT tokens stored in HTTP-only cookies with the following
properties:

- **Secure flag** -- cookies are only sent over HTTPS (when TLS is
  configured)
- **SameSite=Strict** -- prevents CSRF by restricting cross-origin
  cookie sending
- **Configurable expiry** -- default 24 hours, adjustable in security
  settings
- **Session fingerprinting** -- sessions are bound to the client's
  user agent and IP range to detect hijacking

Active sessions can be viewed and revoked from Settings > Security.


## RBAC (Role-Based Access Control)

Each user's data is isolated at the database level:

- Conversations belong to their creator and are invisible to other
  users
- RAG collections are scoped per-user
- Plugin configurations are per-user
- Admin users can see all users but not their conversation content

RBAC is enforced in the API middleware, not just the UI. Direct API
calls are subject to the same access checks.


## API authentication

For programmatic access, authenticate via the login endpoint:

```bash
# Login and get a session cookie
curl -c cookies.txt -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'

# Use the cookie for subsequent requests
curl -b cookies.txt http://localhost:8000/api/health
```

The CLI (`oo`) handles authentication automatically using the stored
session configuration.
