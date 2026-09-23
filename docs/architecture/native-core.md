# The core and the packs

The census of the package (`scripts/core_census.py`) draws a line the
architecture now holds: a small resident **core** that serves one chat
turn with every guarantee the platform makes -- admission by the resource
governor, provenance on every figure, schema and tool lists through the
inference registry, the onion memory, authentication, the conversation
store, tool dispatch -- and **packs**, everything else, which are loaded
when a feature is used and never as a cost of importing the package.

## What the core is

The core is a named list in `.github/scripts/core_boundary_guard.py`:
the registry, the governor, the chat hub, the tool executor and its
dispatch, the onion memory's modules, authentication, the conversation
store, the configuration, and the primitives those pull at module scope
(encryption, secure bytes, the context manager, structured output,
response hygiene). Measured without the package facade, the registry, the
memory core and the tool dispatch pull nothing outside the core at module
scope; the chat hub pulls twenty-nine outside modules, most of them
guarded imports that decide a feature flag.

## What the guard holds

- A core module that imports a module outside the core at module scope is
  refused, unless the ledger already carries that leak. An import inside a
  function is not a leak: it is a cost paid where the feature is used.
- The ledger of leaks may only shrink. A leak the module no longer makes is
  reported stale and must come off, so the debt count stays honest.
- No core module reaches the inference client directly. The registry module
  is the one exemption: it is the funnel every request goes through.

The static census is the guard's counter, so the instrument that measured
the line and the guard that holds it cannot disagree. The census does not
see imports made by name at runtime; the package facade resolves its
exports that way, and its contract records the limit.

## The packs, and how they will reach the core

A pack is a feature that runs for a task and stops: benchmarks, the tuner,
the red team, fine-tuning, plugins, agent evaluation, media of notes. The
protocol they will follow when the core becomes a resident process:

- a pack is launched by the core with a ticket -- the governor admits or
  refuses before launch -- a budget of memory, time and tokens, and a
  lifetime; it ends with its task;
- a pack has no inference client of its own; it asks the core, with the
  schema and tool options the registry already carries, so admission,
  provenance and schema apply to a pack exactly as to a chat turn;
- a pack does not open the core's databases; it receives what it needs
  from the core and returns a result under its ticket.

Today every pack runs in the same process and reaches inference and the
model catalogue through the registry; the registry-funnel guard counts
what still does not, whatever name the client travels under, and also a
module that posts to the inference server's endpoint with an HTTP
transport of its own. None does: every embedding -- RAG, project
context, memory recall -- goes through the registry's `embed` and
`embed_many` heads, and the red team reaches its model through the
registry too, refusing any backend whose `endpoint` -- where its requests
actually go -- is not on the local host. The launcher's liveness probe is
the one exemption, by name: asking whether a process answers is not an
inference request. The
resident core process is the next step, Python first behind the same
surface, then Rust by strangling, as the memory's native core was born.

The terminal chat session (`oo chat`) is not a pack in this sense, and is
not presented as one. It runs in the calling process: the conversation
store, the onion state and the skill root it opens are local to it. What
it does satisfy is the funnel -- every request goes through the registry,
to the daemon when `core.yaml` enables it, and the session holds no
inference client of its own. A pack does not open the core's databases;
the session does, so the separate-process protocol above is still owed.

## The core daemon

`opti_oignon/core_daemon.py` is the resident process, in Python first: a
loopback HTTP server over the inference registry, in the standard library,
importing only the core. Its routes are `GET /health`, `GET /models`,
`POST /inference/generate`, `POST /inference/stream` (JSON lines) and
`POST /admission` (a pack's ticket: the governor admits or refuses, in its
own words). It binds the loopback and nothing else, requires the configured
bearer token on every route but health, and refuses by name an unknown
route, a body that is not JSON, a request without a model, a model no
backend serves. The listing route says whether anyone could look: a
backend that cannot read its catalogue crosses the wire as a null listing
with `known` false, a backend that looked and found nothing as an empty
listing with `known` true, and the client on the other side keeps the two
apart.

`opti_oignon/core_client.py` is the other half: a registry backend that
forwards every request to the daemon. When `core.yaml` enables the daemon,
`init_backends_from_config` registers and activates it in the calling
process, so that process -- the application first -- asks the daemon for
inference and is admitted, labelled and schema-checked like a chat turn.
The funnel crosses the process boundary.

Run it with `oo core serve`; ask it with `oo core status`. Off by default.

Owed to the host, never estimated: the daemon's resident memory at rest
(`ps -o rss= -p <pid>` after `oo core serve`, against the 42 and 249 MiB
the roadmap measured for a Python core floor and the application) and the
latency of one turn across the boundary. The native daemon comes by
strangling, behind this same surface.
