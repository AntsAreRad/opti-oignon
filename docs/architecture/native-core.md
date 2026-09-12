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

Today every pack runs in the same process and reaches inference through
the registry; the registry-funnel guard counts what still does not. The
resident core process is the next step, Python first behind the same
surface, then Rust by strangling, as the memory's native core was born.
