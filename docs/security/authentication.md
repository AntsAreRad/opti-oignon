# Authentication

## Authentication model

Opti-Oignon uses a **deny-by-default** authentication model. The auth
middleware intercepts every API request and verifies the JWT session
cookie before allowing access. Only explicitly marked public endpoints
(health check, login, static assets) bypass authentication.

In Daily mode on localhost, authentication can be relaxed for
convenience. In Bulbe mode, authentication is always enforced, and a
security mode that cannot be read is taken for Bulbe mode.

These checks fail closed. The middleware that requires a session checks
every HTTP request, serves nothing but the public paths when the account
store cannot be loaded, and the app does not start without it; every
WebSocket checks its own handshake, the session and the page's origin: a
page of another host, or of an opaque origin (`null`), is refused, and
only a client that sends no origin, which is no browser, is taken
without one. A login whose second factor cannot be checked -- the
two-factor module unreachable, its store unreadable -- is refused (503),
never let through without the factor. A ceremony that asks for a
password -- approving plugins, leaving Bulbe mode, re-enabling search --
takes the owner's, through the owner's own session: the owner is an
admin's account or, while no account is an admin's, the first
registered, so that an account registered later approves nothing with
its own password, and the session of single-user mode stands for the
owner when there is exactly one; the password is checked through the
login limiter, one check at a time per account and off the event loop,
so that a wrong one counts as a failed login. A wrong password refuses
with a 403, a session that is not the owner's too, an account the
limiter holds back, or one whose password is being checked already, with
a 429, and a password that cannot be checked with a 503 -- with bcrypt
5, one of more than 72 bytes among them; bcrypt 4 checks its first 72
bytes instead --, and nothing is approved, confirmed or re-enabled. A
remote-session guard that cannot be read gives the hardened cookies. Not
yet: the login limiter, when it cannot be loaded, limits nothing, and
the login itself checks tries sent together before it counts the first;
registration is open by default, so that on an install with no account
the first to register is the owner; and a configuration that gives every
new account the admin role (`users.default_role: admin`) makes each one
an owner.


## User accounts

User accounts are stored locally with bcrypt-hashed passwords. Each
account has a role that determines access:

| Role | Permissions |
|------|-------------|
| admin | Full access, user management, security settings |
| user | Chat, RAG, plugins, own conversation history |
| viewer | Read-only access to shared resources |

The security ceremonies -- approving plugins, leaving Bulbe mode,
re-enabling search -- and switching authentication on or off take the
platform's owner rather than a role: an admin's account or, while no
account is an admin's, the first registered. With no account at all, no
ceremony passes until one is registered.


## Two-factor authentication (2FA)

Opti-Oignon supports two 2FA methods, which can be used independently
or together, each described below.

A second factor belongs to the account that sets it up, and the login
asks it of that account alone: a security key, whose request options the
first step of the login carries, a TOTP code, or a recovery code -- only
what the first step offered; a code is tried as a recovery code only
when it has one's shape (sixteen hexadecimal characters), whichever
method names it, any other as a TOTP code, read in its plain form: the
spaces an app shows inside it ignored, fullwidth digits read as plain
ones. A security key's answer is checked against the keys the account
holds when it arrives, and of two answers that pass together, one signs
in. An app password is never a second factor, and no client signs in
with one yet. Adding or removing a factor asks the account's own
password first (403 for a wrong one, 429 for one held back). Setting a
TOTP up over an active one is refused: disable it first. In single-user
mode nothing asks a factor at login until authentication is on, and with
no account at all there is none to give a factor to. Deleting an account
deletes its factors, where the store can write. Factors set up before
they belonged to an account were stored under the identity `admin`,
never asked at login, and are removed at the first start of this version
that can write to the store, their counts audited: the owner is told to
set them up again.

### WebAuthn / FIDO2

Hardware security key authentication using the WebAuthn standard.
Supports USB keys (YubiKey, SoloKeys), platform authenticators
(fingerprint, face ID), and passkeys.

Setup:

1. Go to **Preferences > Account > Two-factor (WebAuthn)**
2. Click **Register New Key**
3. Name the key, and enter your account's password
4. Follow the browser prompt to touch your key

Multiple keys can be registered for backup purposes.

### TOTP (Time-based One-Time Password)

Software-based 2FA compatible with any authenticator app (Google
Authenticator, Authy, KeePassXC, etc.).

Setup:

1. Go to **Preferences > Account > Two-factor (TOTP)**
2. Click **Set Up Authenticator**, and enter your account's password
3. Scan the QR code with your authenticator app
4. Enter the verification code to confirm

A validated TOTP code is single-use within its validity window: the consumed
time-step is recorded, so a code that was just used cannot be replayed during the
30-second window in which it would otherwise still verify.

### Recovery codes and app passwords

Recovery codes (one-time, shown once at generation) and app-specific passwords
(meant for clients that sign in with one; none does yet) are never stored in clear:
only a keyed hash is persisted. The hash is an HMAC-SHA256 under a subkey derived
from the master encryption key on its own domain, so the master key is the only
secret (Kerckhoffs). Without a master key configured, the at-rest protection of
these hashes rests on SQLCipher.

Migration. Recovery codes and app passwords created before this re-keying were
hashed under the previous scheme, and every one of them was stored under the
identity `admin`: those are removed at the first start that can write to the
store (see above), and none is left after it. Should a recovery code under the
previous scheme still be met, it is accepted -- once, as any recovery code --
and the 2FA status reports `recovery_reissue_required: true` while such codes
remain unused; regenerating them from **Preferences > Account > Recovery codes**
keys the new set under the new scheme.


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

The web interface does not list active sessions: a session ends when its
token expires, or when you sign out from the account menu in
**Preferences > Account**.


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
curl -c cookies.txt -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'

# Use the cookie for subsequent requests
curl -b cookies.txt http://localhost:8001/api/health
```

The CLI (`oo`) handles authentication automatically using the stored
session configuration.
