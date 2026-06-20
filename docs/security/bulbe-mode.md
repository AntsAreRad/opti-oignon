# Bulbe Mode

## What is Bulbe mode

Bulbe mode is Opti-Oignon's maximum security configuration. The name
comes from the French word for "bulb" (as in onion bulb) -- the
hardened inner core of the system.

Bulbe mode is a **physical network constraint**, not just a policy
toggle. It enforces localhost-only socket binding at the OS level,
meaning the backend literally cannot accept connections from external
hosts.


## What Bulbe mode enforces

When Bulbe mode is active, the following constraints apply:

- **Localhost-only binding** -- the FastAPI backend binds to
  `127.0.0.1` only; the network bind guard verifies this at startup
- **Ollama bind guard** -- checks that Ollama is also bound to
  localhost (`127.0.0.1:11434`); blocks startup if Ollama is exposed
  externally
- **Mandatory authentication** -- all API endpoints require a valid
  JWT session cookie; no anonymous access
- **Full audit chain** -- every security-relevant action is logged
  to the hash-chain audit log with post-quantum signatures
- **Startup checklist** -- the full security checklist runs at startup
  and must pass all critical checks
- **LUKS advisory** -- disk encryption status is checked and reported
  (advisory only, does not block startup even in Bulbe mode)


## Enabling Bulbe mode

### From the UI

1. Go to **Settings > Advanced > Security**
2. Toggle **Bulbe mode** on
3. The backend restarts with all security layers enforced

### From the configuration

Set `bulbe_mode: true` in `config/security.yaml`:

```yaml
security:
  bulbe_mode: true
  require_auth: true
  audit_chain: true
```

### From the CLI

```bash
oo config set bulbe_mode true
```


## Network bind guard

The network bind guard is a runtime check that verifies socket binding
at the OS level. It inspects the actual listening addresses of both the
Opti-Oignon backend and Ollama.

If either service is detected as listening on `0.0.0.0` or an external
interface, Bulbe mode blocks startup with a clear error message
explaining what to fix.

This is distinct from a configuration check -- it verifies the actual
network state, not just config files.


## When to use Bulbe mode

- **Always** if the machine is on a shared network
- **Always** if multiple users access the instance
- **Recommended** even for single-user use on a laptop that connects
  to public Wi-Fi
- **Not needed** for air-gapped machines with a single user

Bulbe mode has negligible performance impact. The security checks run
at startup and add less than a second to boot time.
