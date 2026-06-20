# Post-install checklist

After installing Opti-Oignon (see [Installation](installation.md)), run through
this checklist to confirm the install is healthy and to choose a security mode
before first use.

## Verify the install

```bash
# 1. The CLI entry point is on PATH and reports the version.
oo --version

# 2. The package and its dependencies are consistent.
pip check

# 3. Ollama is installed and reachable (see Ollama Setup for details).
ollama list

# 4. The frontend builds (from the repository, if you installed from source).
cd frontend && npm run build && cd ..
```

If `oo --version` is not found, ensure your user scripts directory is on
`PATH` (for a `pip install --user` install) or that the conda/virtual
environment is activated.

## Choose a security mode

Opti-Oignon runs in one of two modes. The mode is resolved from two sources
that must agree; if they disagree, the system fails secure and defaults to
**Bulbe**.

- **Daily** -- frictionless everyday use with the baseline security layers
  active. Remote access can be enabled here (off by default).
- **Bulbe** -- maximum security. Every defense layer is active and the network
  binding is constrained to loopback at the socket level. Remote access is
  disabled regardless of configuration.

The human-readable source is `opti_oignon/config/security.yaml`. A minimal
example:

```yaml
# opti_oignon/config/security.yaml
# Primary, human-readable mode declaration. Valid values: daily | bulbe.
security_mode: daily

# Remote access is only consulted in Daily mode; it is always disabled in
# Bulbe mode. Leave disabled unless you have configured mTLS (see the
# Security Guide).
remote_access:
  enabled: false
```

Switching modes is asymmetric by design:

- **Escalation (Daily -> Bulbe)** is immediate and requires a single
  authenticated request.
- **Degradation (Bulbe -> Daily)** requires a multi-factor ceremony with a
  cooling period, so dropping out of maximum security is never accidental.

After editing `security.yaml`, confirm the effective mode:

```bash
oo --version           # confirms the CLI is wired
# then start the backend and check the security/status surface in the UI,
# or query the security status endpoint documented in the API Reference.
```

For the full model -- the signed mode lockfile, the fail-secure rules, and the
escalation/degradation ceremonies -- see the
[Bulbe Mode](../security/bulbe-mode.md) and
[Security Overview](../security/overview.md) pages.

## Next steps

- [First Run](first-run.md) -- start the backend and frontend and send your
  first message.
- [Ollama Setup](ollama-setup.md) -- pull and configure local models.
