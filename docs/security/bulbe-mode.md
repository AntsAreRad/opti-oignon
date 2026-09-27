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


## Web search

In Bulbe mode the platform's web search sends nothing. The searcher
behind the chat, the tools and the proxy health check opens one gate at
every request point -- the search itself and the proxy check's Tor exit
lookup -- and the gate asks the security mode first: anything but Daily
is refused by name, before any cached result is read, and a refusal is
never retried. The capability manifest does not offer the web tool.

The knowledge base's page ingestion, `POST /api/rag/ingest/url`, fetches
through the same gate, asked again before each redirect and after every
read of the page: outside Daily mode, and while the switch is engaged or
cannot be read, it is refused by name before any host name is resolved,
or where the reading stands, and nothing is recorded. It is refused the
same way when `web_ingestion.enabled` in `rag.yaml` is anything but true.
In Daily mode it reaches only a public address. User information in the
URL, a scheme other than http or https, a port other than 80 and 443
unless `rag.yaml` names it under `web_ingestion.allowed_ports`, and a host
that resolves to a loopback, private, link-local, site-local, unspecified,
multicast, reserved or shared (carrier-grade NAT) address -- an IPv4
address carried inside IPv6 included -- are refused. So are, on an IPv6
network where every device has a global address, this machine's own
addresses (a datagram socket binds only to an address the machine holds,
and the bind sends nothing) and any address on a network the machine
reaches without a gateway: its IPv6 addresses' prefixes and its on-link
routes, read from `/proc/net` at every request. A route without a gateway
counts as a link, a VPN's included, so a tunnel that routes a wide prefix
to its interface makes that prefix refused. The host is resolved once per
request and the connection goes only to answers that were checked, in the
resolver's order, so a name whose answer changes in between cannot steer
it to this machine or the local network; the request still names the host,
and TLS verifies that name. At most three redirects are followed, each
checked as the first request, and a redirect from https to http is
followed as a browser follows it. The page's size is capped as it is read,
and one time budget covers the whole fetch. The fetch connects directly: a
proxy named in the environment is not used, and neither is the web
search's own proxy (`web_search.yaml`), so the page's server sees this
machine's address even when searches go through Tor.

The search kill switch is recorded under `data/` and holds across
restarts; every reader treats a switch it cannot read as engaged.
Re-enabling it is refused in Bulbe mode. The tool registry's legacy view
and the agent's toolset still list `web_search` while the switch is
engaged in Daily mode; a call is refused by the same gate, and the
refusal is named in the tool result.

What this does not cover, each named with the work that owns it:

- **Code run by the code executor** -- its network is not confined; that
  belongs to the sandbox work.
- **Plugins** -- they are not confined; that belongs to plugin
  confinement.
- **Ollama's cloud search and fetch** -- no module calls them, and the
  registry-funnel guard refuses a module that would. If `OLLAMA_API_KEY`
  is in the application's environment, nothing at run time would stop
  one that did: keep it out of that environment, and set
  `OLLAMA_NO_CLOUD=1` where the Ollama server runs.
- **Name resolution time** -- a page fetch's time budget does not bound the
  system resolver, whose own timeouts do.
- **The router's public address, reached back from inside** -- a page may
  name the address the router shows the internet, which the router may
  answer itself (hairpin NAT); this machine cannot tell that address from
  any other public one. Nor is a network-specific NAT64 prefix read as
  carrying an IPv4 address: only the well-known `64:ff9b::/96` is.
- **A platform without `/proc/net`** -- the machine's own addresses are
  still refused, its links are not known. A kernel set to bind to any
  address (`ip_nonlocal_bind`) makes every address read as this machine's,
  and every page fetch is refused.


## Every outbound request

A request that leaves the process asks a gate first, by the class of where
it goes:

- **The web** -- a host the platform does not own. It may leave only in
  exactly Daily mode with the search kill switch released: the web gate
  above. Where the platform holds the connection, it reaches a public
  address only, by the page fetch's rules.
- **This machine** -- loopback, `localhost`, an unspecified address, or the
  process itself. The local rule answers: in Daily mode every endpoint is
  let through; in any other mode only this machine is.
- **The operator's own services elsewhere** -- a private or link-local
  address, named by the operator. The local rule answers them as it answers
  any endpoint off this machine.
- **Peers** -- the Veilid peers, reached through the local veilid-server,
  under their own gate, in exactly Daily mode.

The front door, `opti_oignon/egress.py`, asks these gates for every caller
that is not a gate itself: it has no rule of its own, and a gate it cannot
import is a refusal. What asks the web gate today:

- **The web search and the page fetch**, as described above.
- **The plugin marketplace's install** (`POST /api/plugins/marketplace/install`)
  -- refused outside Daily mode and while the switch is engaged or cannot be
  read, in its own words, before anything is fetched or written. The archive
  comes through the page fetch, so from a public address only, as bytes,
  named after the URL asked for. `install.allow_remote_install`,
  `install.max_download_size_mb`, `install.require_hash` and
  `install.timeout_s` in `plugin_marketplace.yaml` are read.
- **The marketplace's index refresh** -- the same gate and the same fetch;
  a refusal keeps the cached listing and serves it. The listing refreshes a
  stale index on its own only when `index.auto_refresh` is true.
- **The model downloader** (`POST /api/backends/gguf/download`) -- the gate
  before it starts, before every redirect and after every block it reads; a
  refusal answers 403 and leaves no partial file. It refuses every address
  the page fetch refuses, and writes only a `.gguf`, named by the last
  segment of the name it is given, inside a configured model directory.
- **The web-only routes, in Bulbe** -- the marketplace install, the GGUF
  download and the model pull (`POST /api/model-lifecycle/pull`) are refused
  by the middleware before their handler, with 403 and the web gate's words.
  The middleware decides on the path the router matches, by whole segments,
  and admits `/api/health` itself, not the routes below it.
- **The mode** -- every gate reads the mode that `security.yaml` and the
  lockfile hold: each read stats both files and reads them again when either
  changed, so a mode another process writes is the next one read.
- **The signature library** -- liboqs-python is imported only when its
  shared library loads; without it the package would clone and build liboqs
  from GitHub.

Not yet: the egress census guard counts every network sink in the package,
and 48 of them, in 15 modules its ledger names, still leave without asking
the web gate. They are the model lifecycle (the pull in Daily mode, its
callers other than the route, and the update check), the external vector
stores (Pinecone, Qdrant, Weaviate) and the local one, code that runs with
the network (the code executor, the sandbox, plugin workers, the dependency
monitor), the Ollama command line the context manager runs, the core
client, the token counter, the terminal interface and the Veilid client.
The local rule does not yet look at an environment proxy: a proxy named in
the application's environment still carries the requests to Ollama and to
a llama-server.


## Enabling Bulbe mode

### From the UI

1. Go to **Workshop > Security > Security mode**
2. Click **Escalate to Bulbe**
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
