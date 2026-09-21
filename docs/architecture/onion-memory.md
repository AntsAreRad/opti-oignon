# Onion memory

The model always sees a small, bounded, deterministically assembled window,
backed by an unbounded archive that is the only legal source of every
compression. Every compression step is verified before it is trusted.

## Layers

| Layer | Content | Changes by | Cap (tokens) |
|---|---|---|---|
| Core | pinned invariants: persona, hard constraints, canonical decisions | supersession only, on an explicit user action | 1024 |
| Receipts | one line per evicted span, with its recall key | append and resolve | 512 |
| Peels | summaries that answered the recall probes drawn from their source spans | regenerated from the Cellar only | 3072 |
| Flesh | the last turns, verbatim | a turn leaves only through an eviction that leaves a receipt | 5120 |
| Cellar | the full archive; not in the window | append only | -- |

The caps and the generation reserve live in `opti_oignon/config/onion.yaml`
and add up to the window exactly; the composer refuses a file that does not.

## Invariants the contracts hold

- The composer is a pure function: the same registry state, budget and
  retrieval set give the same prompt, and the inputs are not touched.
- The Core bytes in every prompt are the registry's bytes, anchored by a
  hash; an entry whose bytes moved is refused by name, never repaired.
- No eviction without a receipt; no receipt whose key does not resolve to a
  Cellar span -- the digest checks every key before it renders a line.
- A span leaves the Flesh only once its candidate peel has answered the
  recall probes drawn from the span, class by class, at the configured
  thresholds. An empty probe set on a rich span is a defect, and an unknown
  rate is not a pass.
- A parent peel is summarised from the union of its children's Cellar
  spans, never from the children's text. A peel handed in that stands on
  anything else -- a dangling key, sources that are not its children's, a
  digest that no longer matches the archive -- is refused.
- Recalled content is framed as data with its provenance and wrapped as
  untrusted memory before it reaches the model; only the Core and the
  current turn bear instruction.

## The librarian

Off by default (`enabled: false`). Switched on, the executor offers each
saved conversation to the librarian beside the auto-capture; the librarian
mirrors it, and when it has grown by `min_new_turns` it evicts through the
gate off the interactive path, using the configured model through the
inference registry with `keep_alive: "0"`, so the model holds nothing while
idle. On the next turn, the librarian's block -- Core, receipts digest and
the peels selected for the question -- replaces today's working-memory
block when it has one, and today's block stands when it does not.

With a persistence path in `onion.yaml`, the librarian writes each
conversation's Core, Cellar, receipts, Peels and Flesh through
`opti_oignon/memory/onion_store.py` after every mirror and every accepted
eviction, and reads it back when the process next sees the conversation.
The read proves what it loads: every Core entry, Cellar span and Peel is
re-hashed against the id it was saved under, the root over the four
stores is recomputed against the saved one, and the first row that no
longer answers is refused by name. A conversation the file does not know
is unknown, never an empty state. The table holds conversation text, so a
connection that is not encrypted is refused unless the file allows it;
the rest of the tree opens plaintext with a warning outside Bulbe mode,
this store does not. Without a path the onion lives in the process and a
restart starts it over. The root covers Core, Cellar, receipts and Peels;
the drift ledger has its own table and its root is a later decision.

## Drift ledger

`opti_oignon/memory/ledger_store.py` holds facts with provenance, kind and
confidence. A fact is superseded, never edited: the only update the module
issues links a row to its successor. Every write runs the deterministic
contradiction templates against the active facts and records what they
find, so drift is a count -- contradictions per thousand turns -- rather
than an impression. The model judge for what the templates cannot read is
a host measurement.

## What is measured where

Everything above is proven in CI over injected summarisers, and the
figures those runs produce carry `source: fixture`. The numbers that need a
real model -- gate acceptance with the real summariser, compression
fidelity, the effective-context multiplier, summariser latency, and the
threshold sweep the calibration needs -- come from
`scripts/onion_runbook.py` on the host, labelled `source: measured`. The
script refuses to print a number it did not measure. The thresholds in
`onion.yaml` are the design's proposed defaults; on the fixtures, a single
wrong entity or date in a four-turn span passes at 0.7, and that finding is
recorded as a contract until the calibration replaces the numbers.

## The native core

The integrity primitives -- the four hashes -- and the window assembly
have a second implementation in Rust, `rust/oo_core`, built into
`opti_oignon/native/` by `scripts/build_oo_core.sh` and never tracked.
Python stays the reference and the fallback: the memory modules ask for
the native core at the call, never at import, and run the reference path
when it is absent. When it is present, every hash is byte-equal to the
reference and every assembled prompt is field-equal, and the contracts
that hold it to that run on every machine that builds it. The crate is
pinned (exact `pyo3`, committed `Cargo.lock`) so the artefact is
reproducible. Floats in a span are refused by the core and formatted by
the reference; the memory stores none.
