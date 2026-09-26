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
  thresholds. The probes read English and French: decisions, negations, names and whole words, accents included. An empty probe set on a rich span is a defect, and an unknown
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

The user's surface on the Core and the receipts is a memory route scoped
to the conversation: `GET /api/memory/onion/{conversation_id}` lists the
Core entries and the open receipts, `POST .../pin` pins a statement as the
user, `POST .../supersede` pins a successor and links the old entry to it,
`POST .../recall/{key}` hands the verbatim span back and marks the receipt
resolved. The route imports the librarian inside its handlers and passes
the user as actor; the store refuses any other actor, and a pin that would
push the Core over its cap is refused before it lands, because the
composer never cuts the Core and would otherwise blank the whole block.
The model's tools reach none of this; a contract on the tree names the
three modules that import the librarian -- the executor, the memory
routes and the terminal chat session. Recall has no actor gate by design:
it changes nothing in the Core, and the model's tool over it is not built.

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

Two verbs work on a whole conversation, and `oo chat` gives them to the
user as `/close` and `/open`. Closing runs the gate synchronously over
the entire Flesh, whatever its cap, one span at a time: every span the
gate accepts leaves with its receipt and its peel, the first it refuses
stays verbatim and is named with the probes that failed, and there is no
override. The state is saved, and the receipts digest and the Core root
it leaves are returned. Opening finds a saved conversation again through
the store and composes its block; a conversation the store does not hold,
or a configuration without a persistence path, is refused by name, and a
store that refuses the conversation is raised by name where the memory
block would have answered empty.

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

Drift against a no-onion baseline has a host command of its own,
`scripts/drift_ab.py`. It runs one scripted conversation -- facts stated,
replaced on known turns, and asked about long after the history window has
dropped them -- through two arms that differ the way the chat path does:
the same system prompt, history window and model, and in one arm the
onion's block, wrapped as untrusted data. The librarian runs in process
with persistence off, so neither arm reads or writes the data directory.
Each arm gets contradictions per 1000 turns against the facts holding at
each turn, the sentences that agree with them, and the sentences the
templates cannot decide, which are left to a model judge and never counted
as agreement.

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

The recall probes of the eviction gate -- drawn from a span, then scored
against a candidate summary -- have a native implementation too. The
reference is a set of Python regular expressions over Unicode classes; the
core reproduces them as hand-written scanners, and only for text whose
every code point it classes exactly as Python does: ASCII, Latin-1, Latin
Extended-A but the dotted capital I, General Punctuation and the euro
sign, which covers English and French. Any other code point sends the call
to the reference. The expressions travel with every call and the core
refuses any it does not reproduce, so a change to them on the Python side
takes effect at once, on the reference path, until the core is taught it.
The rates, the thresholds and the gate's wording stay in Python.

The same artefact links a second crate, `rust/allium`: the engine of the
componion, the companion onion, which the terminal's `oo garden` looks
after (below). It answers one byte protocol -- canonical JSON in,
canonical JSON out, never an
exception -- and `opti_oignon/allium/ref` answers it too, as the
reference. Before the native engine is used, its whole identity
(versions, limits, operations, refusal codes, and the digests of the law
and table files both engines read) must equal the reference's; an
artefact built from other files is not used, and that is said once. The
crate is `no_std`, holds no float and no hashed map, and releases the GIL
for the length of a call.

The engine's first organ is the genome: four operations found a genome
from a seed and a law, build one at a corner of the law's bounds, decode
one, and compile one into flat tables. A genome is never stored -- the
seed and the law are -- and a genome received from elsewhere is decoded as
hostile input, accepted inside the law's bounds or refused by name. The
laws and founder pools under `opti_oignon/allium/laws/` are written by
`scripts/allium_author_genome.py`; `scripts/allium_greenhouse.py` reports on
a cohort of founders without simulating a day, including the body levels and
the adult beard and moustache each founder's genes allow.

The second organ is the phonology: eight operations read a being's
language block from its genome, decode the sounds it can make, judge
whether forms are sayable, coin forms for concepts, give its first sound,
derive its 2048-form word list and check forms against a taboo list kept
as SHA-256 digests only. `scripts/allium_author_phon.py` writes the
phonology table and the fixture's witness digests;
`scripts/allium_taboo_digest.py` turns the owner's word list, read from
outside the repository, into the full law's digests.

A being's life is written in a journal, one file per person under the data
directory. Each fact has an id that does not depend on the device (the
digest of its envelope, the body entering only by its digest) and is
linked to the one before it, so a flipped byte, a reordered event or a cut
tail is refused by name at the event where it breaks, never repaired.
Every write goes through one membrane that derives the surface and the
actor from how the request arrived, not from what it claims, and that
holds each kind of event to its schema, its surfaces and a daily budget.

A being is not born without encryption: the store needs a readable master
key and SQLCipher, and without them it waits for its soil. Only when no key
is configured, in Daily mode, can the owner allow a glass jar (a journal in
clear) by setting `persistence.require_encryption` to false; a jar is
labelled everywhere, never opened in Bulbe mode, and stays open once born
until it is repotted. The being's birth is anchored in the signed audit log,
and the store carries a keyed anchor over its head, so an older copy, a
swapped file or a being whose file went missing is shown as such rather
than as a new garden. These anchors protect the being against the software
and against a file changed behind its back; they are not a lock against its
owner, and they do not detect a restore of the whole data directory, audit
log included, to an earlier day.

Forgetting is real but has a scope. A taught word lives sealed under its
own random key, and forgetting destroys the key and the sealed bytes, then
compacts the file; the event that it was forgotten stays in the journal,
without the word. What is gone is gone from the store's files, not from the
medium: an older backup or a disk image may still hold it. A broken journal
is never mended in silence; the owner can resume from the last verified
event, which is recorded as such, keeps every forgotten thing forgotten and
discards what could not be verified.

A being lives in real time. Its age is the minutes since its birth, read
from the wall clock, and its days are local days: whenever the machine's
UTC offset changes, the next write records the new offset (never a zone's
name), so a journey across time zones neither lives a day twice nor
skips one silently. Looking writes nothing: a view is the engine folding
the journal from the genesis to the minute asked, and the states kept at
a few local midnights are caches it may start from, which can all be
dropped without changing what it shows. Advancing the clock buys nothing:
a being whose clock jumps a month ahead lives the same month as one left
alone for a month -- the garden rains on it, a windowsill dries it out
and it sleeps -- because the engine sees only the facts and the minute; a
clock set back is noted in the journal once, and a late gesture lands on
the latest minute recorded. Each being keeps the law and the four params
its birth froze from the proposal in `config/allium.yaml`, so editing that
file moves no living being. A law update takes effect at the next local
midnight: the owner can apply the proposal to a being, or pin it to the
law and params it has; when a stable successor of its law is carried, the
next gesture on the being's home device writes the update to it, with the
params in force (or with those of an update the owner applied that has not
taken effect yet), and a view never does. The laws in this tree are
prototypes, and every being sown under one is labelled so wherever it is
shown; a prototype whose law is retired (listed in `laws/retired.json`) is
shown as a retired prototype, never as missing or broken, and a being
whose law file was edited in place does not open.

The first surface is the terminal. `oo garden` runs in the calling process
and reaches the componion through one service,
`opti_oignon/allium/service.py`, which the API serves as well (below). The
garden is off until `config/allium.yaml` says `enabled: true`, and
switched off it builds no store and reads no key, security mode or clock.
Each command is one action: the security mode, whether the machine has a
single account, and the wall clock are read once and hold for the whole
action. The terminal's account is derived by the service (`local` on a
single-user machine; a terminal on a machine with accounts has none and is
told so), and a being is opened only under that account; the single-user
answer is read from the auth settings and, when the auth store exists,
from one count of its accounts, without importing the auth module and
without writing in that store: while a WAL file or a journal lies beside
it, it is opened read-only, so pending frames are never checkpointed and a
hot journal is never rolled back (the answer is then "not single-user").
A look writes nothing in the being's store. After each write the terminal makes
-- a sowing, a gesture, a name, a law update, a pin or unpin, a resume --
the service settles the being in the same action (a settle that fails is
logged and changes nothing the write said), so there are kept states
for `keep verify` to check: it verifies the chain from the genesis,
replays every kept state on the reference engine, and compares the state
served now; a kept state that disagrees is named by its day and law
version and never replaced. An engine that stops on a fault freezes that
one look on the last kept state, labelled, and the next look computes
again. Sowing, naming, writing a law update, resuming and finishing a
sowing run only from an interactive terminal in the foreground, and every
sowing names the newest law the engine carries for sowing and does not
retire, never the test law.

Every form the garden prints comes from a closed catalogue
(`allium/wording.py`, rendered through `allium/describe.py`) and passes
the checks of `allium/ethics.py` before it is printed, a person's name, a
file path and a catalogue key being read as neutral words: printable ASCII; no
word of an inner state -- feeling, wanting, missing, fearing, thinking,
knowing, waiting, and the words of affect -- beyond the few verbs the
design allows a simulation (it predicts, learns, expects, sleeps, rests,
flowers, computes, and asks for); none of the phrases a companion uses to
hold a person (a streak, days away, come back, last chance, and the like);
and no served field named after an absence. The lists hold the forms they
name and no paraphrase of them, which `ethics.LAWS` says where it matters.
One sentence is exempt, by its
exact bytes, and every `show` closes with it: "This is a simulation of an
onion. It does not feel anything; what it does follows the laws its
laboratory names (oo garden lab laws). It will never ask you to come
back." These rules are code, not configuration, so no settings file can
loosen them. A refusal is said through a closed table, never with an
exception's text, and one that comes after a write says the write was
done. While a garden command runs, no log record of the platform reaches
the terminal. Under Bulbe's rules -- Bulbe mode, or a mode that cannot
be read -- the life goes on as in Daily and the forms say so; a glass jar
stays sealed and nothing of it is shown; and the capabilities the policy
in `allium/habitat.py` closes outside Daily (taste, voice, initiatives,
sync, a change of dream depth) are behind a gate that calls nothing
outside Daily, though none of them exists yet.

### The garden over the API

The API serves the componion read only, through one route,
`GET /api/allium/status`, which answers the terminal's own projection: the
status, its label codes, the being's codes at the served minute and the
lines that say them, unwrapped, the doctrine last. The router is always
mounted, so the published surface never depends on the settings file.
While `enabled` is not `true` the route answers `disabled` with no line at
all -- the terminal's form names the path of the settings file, which is
never served -- and neither the route nor the `allium` key of
`/api/health` imports anything of the being, whichever allowed name the
request was addressed by: both read the switch, and the route the names of
`api.hosts`, from the file themselves, by the garden's own rules. Only a
refused request imports the garden's catalogue and its nets, to say its
line. The API's garden is built at the first status request with the
switch on, one per process; it serves no gesture, refuses a caller that is
not given, reads this server's emergency stop, and takes its single-user
rule from the auth manager the server already runs, whose read opens the
auth store as the platform's own reads do.

Every garden route first checks where a request comes from. Its Host
header, exactly one, must name this machine as `127.0.0.1`, `localhost` or
`[::1]`, or a name listed in `api.hosts` in `config/allium.yaml`: exact
lowercase names or IP literals, with no port, scheme or wildcard, an IPv4
address written as its dotted quad. Ports are
never compared (a rebound name is the attacker's, the port is not), and
`X-Forwarded-Host` is never read. An Origin, when one is sent, must name
the same set, over https for a listed name; a `Sec-Fetch-Site` other than
`same-origin` or `none` is answered only beside such an Origin, so a page
of another site cannot make the garden look, even with an image pointed at
the loopback API. A refused request gets one closed line and a code, never
an echo of what it sent. A phone reads the status through a page served by
this machine. Through the Vite dev server on the local network, list the
PC's IP address (a name also needs Vite's `server.allowedHosts`). Behind
remote access over TLS, or a reverse proxy that forwards the Host, list the
name the page is served under. A proxy that rewrites the Host to a loopback
name (nginx's default, `proxy_set_header Host $proxy_host`) turns the name
check off, as the dev proxy's `changeOrigin: true` did: forward the
browser's Host (`proxy_set_header Host $host`) and give the proxy no
catch-all server. Prefer an IP literal to a name for the phone: a name is
only as safe as its resolution, and a device on the local network can
answer for a `.local` name or a local DNS name. A page on one origin
calling the API on another needs an https origin whose name is listed, and
the platform's CORS to allow it. The dev proxy keeps the browser's own Host
(`changeOrigin: false`), so the check holds through it; Vite's own host
check is off when `server.allowedHosts` is `true`, and when Vite serves
https. The phone app's own channel, remote inference, refuses the being's
names as a capability it never reaches.

A view in a request is capped: `api.python_cap` units of engine work with
the Python reference (200,000 by default), `api.native_cap` with the native
core (5,000,000). No request asks the engine for more than the cap at once,
and a view that does not finish within it is shown as of its last kept
state, labelled, with a line saying that the next write made in
`oo garden` computes it: the API never catches up and writes nothing in the
being's store, and each write the terminal makes settles the being in the
terminal's own process. A cap below one awake day of the laws the engine
carries is raised to that day, so the line holds. Building the native core
(`scripts/build_oo_core.sh`) removes the reference cap in practice. A look
can still cause one write of the platform's own: when the mode files have
changed to disagree, the garden's mode reading records the mismatch as
tamper evidence (the auth store's audit log and the signed audit chain),
once per change of the files; and the first look of a process may be its
first user of the signed audit log, which then creates its table and may
rewrite its anchor. The API holds one store for the life of its
process, so a key made readable after its first garden request is seen
only after a restart; restart the API after an upgrade as well. The web
writes nothing in this version: sowing is irreversible, and in the default
single-user mode, which has no login, the membrane keeps it to an attended
terminal; and no consent request exists yet that a page could grant.
