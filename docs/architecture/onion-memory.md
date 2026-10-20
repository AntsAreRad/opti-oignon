# Onion memory

The model always sees a small, bounded, deterministically assembled window,
backed by an unbounded archive that is the only legal source of every
compression. Every compression step is verified before it is trusted.

## Layers

| Layer | Content | Changes by | Cap at 16384 (tokens) |
|---|---|---|---|
| System | the instructions the turn is sent with; bears instruction | its caller, each turn | 1024 |
| Core | pinned invariants: persona, hard constraints, canonical decisions | supersession only, on an explicit user action | 1024 |
| Receipts | one line per evicted span, with its recall key | append and resolve | 512 |
| Peels | summaries that answered the recall probes drawn from their source spans | regenerated from the Cellar only | 3072 |
| Flesh | the last turns, verbatim | a turn leaves only through an eviction that leaves a receipt | 5120 |
| Tail | the sources a turn brings -- retrieval, web results, documents -- as data | its caller, each turn | 3072 |
| Turn | the current question; refused if over, never truncated | the user | 512 |
| Cellar | the full archive; not in the window | append only | -- |

The caps live in `opti_oignon/config/onion.yaml`, one table per rung of the
context ladder the resource governor admits contexts on (`ctx_ladder` in
`resource_governor.yaml`, read as a file), with one generation reserve for
all: each table and the reserve fit within its rung, and no layer is
smaller at a higher rung. The composer refuses by name, every reason at
once, a file that breaks any of this or names a key twice, and a ladder
that is not a list of distinct positive whole numbers, which the governor
reads more loosely. A context composes under the largest rung at or under
it; a caller that names no rung composes under the 16384 table without the
system prompt and the tail, which it does not lay: the window before rungs,
12288 tokens, cap for cap.

Every part of the window is counted once, with the counter of the turn it
is sent in. The executor hands the journal's plan its own counter, for the
model the turn goes to; the plan counts every part of that turn's window
with it, on the turn's own thread, so two turns of one conversation in
flight never count with each other's, and holds it as the conversation's
last counter. A pin, a supersession, an accepted proposal and an open
count with that counter; before any turn in this process, an open counts
with the default, and a pin, a supersession or an accepted proposal counts
the Core as the heaviest counter a turn may hand over would. A Core grown
by a pin, a supersession or an accepted proposal keeps a quarter of its
cap free (`core_margin`), and one token more for rounding: what lands
still fits a later turn that counts it up to a third more. A turn that
counts the Core over its cap at every rung the model's window allows is
refused by name, `core`, with the remedy of superseding it with a shorter
one, and the Core is never cut; a supersession that does not grow the
Core is never refused, so a Core that landed counted by words alone can
step down until the window is laid again.

## The journal

With the onion on, a turn of a conversation the store keeps that the chat
executor answers -- the direct, web-search, code-verify and think
pipelines, the tools pipeline, both phases of think and tools,
self-correct's first draft -- is sent as the onion's journal lays it, and
the executor composes nothing of that window itself: no optimizer, no
history window of its own, no working-memory block, no archive retrieval,
no context check of the turn's documents. It gathers the sources the turn
brings -- the retrieval, asked for within its share of the tail at the
conversation's own rung, the web results, and for a tool turn the calls
the conversation's earlier turns made -- and hands them raw to the
journal. A turn of no conversation keeps the path it had, as every turn
does with the onion off. A plain chat turn takes the tools pipeline by
default, its request arming the tools; its rounds run inside the one
window the journal lays for it (below). The other pipelines keep the
context they build themselves: the reasoning and consensus engines, the
cascading and speculative pipelines and self-correct's correction pass,
their own, or none; the reasoning engine, the consensus engine and the
correction pass say so in a status line.

The journal is a window as messages, the same bytes from one turn to the
next up to the current turn's message, so an engine that keeps its prompt
cache reuses the longest prefix. The system prompt, and the manifest when
there is one, open it, one system message each. One user message holds
the Core under its header, then one memory envelope with the peels of the
region -- the newest the tree accepted that fit the peels' cap -- and the
receipts digest, with the conversation's first turn joined to it when that
turn is the user's. The Flesh follows, each turn in its own role, its
documents in file envelopes at their places and its words bare. The last
user message holds the tail -- the memory the question brings, the
retrieval, the web results and the sentence that cites them, a tool turn's
earlier calls, each in its envelope and cut to whole items within its share
of the tail's cap (`journal.tail_shares`: memory 0.35, retrieval 0.25, web
0.1, tool 0.1, documents 0.2, every kind named), what each left out counted
-- the memory borrowing, past its own share, the shares of the kinds the
turn brings nothing of, the room the region leaves of the peels' cap and
the room the system prompt, the Core and the turn, laid whole, leave of
theirs, since at 4096 its own share does not hold its envelope: room it
gives up when the window passes its cap, before the Flesh gives, none of it
under a tail's room, and that a burst is not aimed on; the plan keeps of
the memory it chose, a recalled span included, what the composer's own
loop lays at that cap, and the window counts what it kept back as cut;
chosen by rank, so that a cap leaves out the last chosen first, the
memory is laid in the order of time, the newest nearest the question, a
recalled span after it, wherever that order holds within the cap; a peel
the region lays is among it, chosen by rank with the rest, so a fact and
its update the question reaches can stand together in the order of time
where the cap holds both, the region keeping it as it was -- a copy laid
only with room the window does not need: the first thing it gives up when
it passes its cap, before the borrowed room and the Flesh, and not
counted as cut where the region shows it; chosen after the rest, a
recalled span included, the words the user typed of a newer statement the
question reaches where an older item reaches it alike -- since a summary
keeps a fact's new value, not always that it is new: for each term of the
question an older item holds, the newest unit that holds it, word for
word, from the span of a peel it lays, at that span's place in time, or
from the Flesh, which stands farther from the question, laid again in the
tail after every item that says its time -- a copy that, as a copy of a
peel of the region, gives way first when the window passes its cap and
is counted as cut only where the window lays its turn nowhere -- never a
unit that only asks, one that reaches into an order of the user's, shown
whole where it stands, one within a segment its peel references, a turn
a peer sent or one a withdrawal reached -- then
the turn, its documents held to the document share and refused, never
cut, past it. Two user messages are
never adjacent. Each message carries the label of what it holds, and
the request's label is their union. The memory shows nothing a withdrawal
reached: a peel or a recalled span standing on a source the user withdrew
is left out, and the window says what it left out.

The executor reads the model's window as the context manager gives it --
its configuration, its table of models, what Ollama says of the model,
else its default of 8192; 32768 only without the context manager --
bounded on a llama-server engine by the smallest context of its slots -- a
listing that names none leaves the bound unknown, and the turn says so in
its ledger row -- and, with no governor, on an in-process engine by the
context it loads at. The plan picks the rung: the largest of the ladder
within `home_rung` and that window, or a larger one when the system prompt,
the Core, the turn's words or its documents need it, never past the window.
The resource governor, when on, admits the turn once, at that rung, before
the window is composed; the window is composed under the context admitted,
capped by the slot's, or, when no admission names one, under the rung or
the lower context an in-process engine loads at. A context under the rung
is a descent: the window is composed under the largest rung the context
holds, and the conversation keeps its rung. The answer always keeps
`journal.answer_floor` tokens: the window is laid under the count that
leaves them once the count is raised by `journal.margin` -- what the window
is accounted -- and the answer is held to what the context leaves past that
account. The executor checks that the window was composed for the context
it is sent with and that its account and the answer's room fit that
context, then sends it with both (`num_ctx` and `num_predict`; on
llama-server `max_tokens`, with `cache_prompt` so the prefix is kept): no
truncation is left to the engine. An answer that used all its room, or
that llama-server says it stopped at its own limit short of it, is the
outcome `length`, said in a status line and never in the answer's words,
and is not cached, since the cache key does not name the room; an answer
from an engine that reports no count is not cached either.

Each image of the current turn is charged `journal.image_tokens`, counted
on its message: the counter reads none. The current turn comes first: the
charge takes its room from the Flesh, raises the rung only when even the
whole Flesh cap cannot hold it, and is refused `document` past every rung
within the model's window.

### A turn of the tools pipeline

The agentic layer reads the switch once for the turn and hands the chat
executor the switch and the tool loop. A loop handed in without a journal,
or offered no tool the journal can lay, is refused by name before any
plan. The window's head is the system prompt, the manifest and one fixed
attribution message -- every journal turn handed a manifest carries the
same three, so the head stays the engine's to reuse. The plan charges the
schemas the turn may send beside its messages -- the native tool schemas,
or the structured decision's instruction a model that does not call tools
natively is sent, the heavier -- on the whole, counted with the turn's
counter: the charge takes its room from the Flesh after the images', raises
the rung only when even what they leave of the whole Flesh cap cannot hold
it, and is refused `tools` past every rung, before any admission. The
window is composed holding back `journal.tool_room` for the transcript its
rounds append, past the answer's floor, the Flesh giving first over a
shadow; at another rung than home, the room and `journal.tool_result_cap`
are held to the rung in proportion. The plan weighs that room: a rung whose
context, less the room and the answer's floor, cannot hold what never gives
-- the system prompt, the Core, the turn and the schemas -- is passed for
the next, and refused `tools` past every rung, so a window is never refused
its count once admitted. A tool turn is planned with no image: its rounds
send none, as the tools pipeline never did. The record a burst reads names
the Flesh room the schemas and the transcript leave, so the burst evicts
ahead of it.

The turn is admitted once, composed once, never served from nor stored in
a cache, and refused by name offline instead of queued: a queued turn would
run again without its tools. The stream's thread runs the tool executor's
rounds under the window and its ticket, every request through it: the
window's messages byte for byte, then the transcript, never rewritten --
native for a model the native replay is allowed for, each call's echo, with
the arguments it ran with, then its result in role `tool`; flat for any
other, per round one assistant message (the model's own decision text, or
the runtime's line naming its calls with the arguments they ran with) then
one user message holding the round's results. The model's own words the
transcript lays -- a decision's text, the reply a verification answers --
are defanged as every bare part is, and counted as laid. A structured
decision joins its instruction after the request's last message, or past a
message that is not a user's in a message of its own, its overhead counted;
each request's messages, what it is told past them stripped, are a prefix
of the next's. Every request carries the window's context and the room the
answer keeps past the window, the transcript as sent and what it is told,
the turn's keep_alive and the same tools, the final's included; on
llama-server its room as `max_tokens`, the prefix kept and the one slot.
Each call is judged by the turn's provenance and the names its manifest
offers, as off the journal. Each result is laid under a bare header in the
runtime's words naming its call by its place in the round and its tool --
its arguments are the echo's or the line's, never laid twice -- defanged
as a tail item is and inside its own tool envelope, and cut past what it
may take -- the most one result may take, or what the room leaves it --
with a line naming how much of it is shown. A round the room cannot hold
runs none of its calls, and a decision that used all its room is never
parsed: the loop ends by name, and the final keeps the answer's floor. A
round whose calls ran with more than they were decided with -- a default
filled -- and so pass the room is laid in the form the room held before
they ran: each call as decided, each result at its least.

A decision that calls nothing is the answer, after the tools as before
them, as the decision wrote it. When a guard ends the loop -- a call
repeated, a call that failed for good, the retries or the rounds spent, the
room spent -- the final is asked for, told in one fixed line joined after
its last message that no further tool can run; the line is counted, the
transcript keeps its room, and the same tools are offered for the head's
sake. Every final is told so, and so is the salvage's own request when a
guard ended the loop before any tool ran. A final that calls a tool anyway
is said and counted, and so is a salvage's candidate that is the answer;
one that says nothing else ends the turn in fixed words. The Ollama and
llama-server heads say how many tools an answer called, streamed or whole,
and the llama-server head's whole answer says what the server counted and
why it stopped, so a decision that used all its room is seen on either
engine.

Each request's label, read where it is sent, is the window's joined with
what its results carry -- `tool`, and `tool:` the digest of its whole
result, the entry the withdrawal grammar names a tool's output by, so a
withdrawal by value reaches it. The answer is saved
once with the union of every request's label, flagged `assistant+tool` only
when a result was among them, then offered to the capture and the curation;
next turn it is a Flesh turn like any other, laid bare with its label. A
final that fails, or a loop that finds no backend, ends the turn in fixed
words and saves nothing; a stop or a close makes no further call, saves
nothing and keeps the record of the calls that ran; an early exit gives
back its admission and the load it claimed. The calls the conversation's
earlier turns made are never replayed into the window as calls of this
turn -- no echo, no tool message, no flat replay: the agentic layer hands
them with the loop, and they ride the tail as its `tool` kind, newest first,
each one whole item -- the runtime's words, the call named as a transcript
names it with the arguments it ran with, then its whole result, a line of
it that opens as those records do neutralized -- in one tool envelope,
defanged, cut to whole items within the kind's share, the ones left out
counted, each under the label its result carried in its own turn; one
standing on a source the user withdrew is withheld. Their answers are the
window's Flesh.

### Think and tools, and the turns a pipeline saves itself

Think and tools reads the switch once for its turn and hands it to both of
its chat-executor calls, neither saving: the first pass, then the tool
phase, laid by the journal as the tools pipeline is -- its own window,
planned with the schemas, its rounds inside it -- never reading the
conversation's last rows nor replaying an earlier call. The tool phase's
answer is held until its loop ends and shown, as off the journal, only
when a tool ran. A first pass that ends by name -- a window or an admission
refused, a stream that failed, the turn queued offline -- ends the turn
there, said as the chat executor said it, and nothing is saved. A tool
phase that ends so, or whose chat-executor call raised, is said after a
blank line and joins nothing: the first pass is saved alone, flagged
`assistant`, and what the phase reported -- the refusal, the admission
refused, the stream that failed -- stays on the run, so a pipeline runner
still ends its step by it. Otherwise the pipeline saves the turn once --
the user's row, then the first pass joined to the tool phase's answer --
with the label the run reports, the union of the labels of the requests
sent on it, both phases' among them, flagged `assistant+tool` only when the
tool phase's answer stands in it.

Every save a pipeline makes of a turn it answered itself -- think and
tools, the reasoning and consensus engines, self-correct, the cascading and
speculative pipelines, the coding agent -- passes one door once its rows are
written (`executor.offer_saved_turn`): on the journal path only -- the
switch as the turn read it at its start -- the turn is offered once to the
capture and once to the curation, as the chat executor offers its own; off
it, nothing follows the rows. Every one of them but think and tools keeps
the label it always had -- the agentic pipelines' wide one: the question's
parts, the last ten turns, the `tool` kind and what the run reported -- and
the reasoning engine, the consensus engine and self-correct's correction
pass say in a status line, with the caveat's prefix and once they have
returned, that the onion's window is not theirs; an engine that raises
falls back to a reply the journal lays, and says nothing of the kind. On the
journal path the pipeline runner gives back its own admission before a step
the chat executor answers -- its ticket held and released, its load ended
-- since every chat-executor call of the step is admitted on its own: a
tools step holds one admission, not two. A reasoning or a consensus step
runs an engine of its own and keeps the runner's admission, its only one.

A chat-executor call whose stream fails, before or after its first words,
says so to its run (`stream_failed`), with the onion off or on, and so does
one queued offline (`offline_queued`). A pipeline that drafts through it
keeps no answer from it: think and tools runs no tool phase and saves
nothing; self-correct corrects nothing and saves nothing, ends its first
step failed and its second not run, and says so in fixed words; on the
journal path, a draft refused by name, refused its admission or queued
offline is neither corrected nor saved either, the turn stopped or not. A
pipeline step whose stream failed ends failed, and its text never reaches
the next step -- the chat executor's stream, the tools pipeline's answer cut
after its first words, its chat-executor call raising on the journal.

A Flesh found over its cap at the conversation's own rung -- a
conversation the process had not seen, one older than the onion, one grown
past its cap -- is caught up at the plan, with no model, while no burst
holds the conversation's slot: the turns past the newest
`journal.catch_up.reach` leave bare, a span at a time, then the queue runs
at most `max_steps` steps on the rungs that need no model, then bare spans
until the Flesh fits; every span leaves under its receipt and stays
recallable. With the slot held, nothing is evicted: the window lays the
overflow over a shadow -- its oldest turns set aside in stores of the
shadow's own, for one composition, with one line naming them, charged to
the room they freed, and the peels the shadow's queue made of them -- and
the conversation's state is not touched. A descent is laid over a shadow
the same way. A burst aims below the cap, at `journal.burst_mark` of it,
so a window rarely has a shadow to lay.

Each window is compared with the conversation's last. Its stable prefix
kept, no epoch begins; broken, an epoch begins, read by why -- the first
window, the model, the rung, the system prompt, the Core, an accepted
proposal, a resolved receipt, a take-back, a withdrawal, an eviction, a
region past its cap -- and any other cause is counted unexplained. The
window says where its first changed piece begins, and whether it is cold:
the first window, or one whose first piece changed, sharing no prefix with
the last.

A window that cannot be laid is refused by name, never with an
exception's words: a layer over its cap (`system`, `core`, `receipts`,
`peels`, `flesh`, `turn`, `window`), the turn's documents or images past
every rung (`document`), a tool turn's schemas past every rung or no tool
offered (`tools`), a tool turn offline (`offline`), a tool loop handed a
turn the journal does not lay (`journal`), a source of a shape the window
does not lay (`tail`), a context no rung holds (`rung`), no rung within the
model's window (`model`), an answer left under its floor (`count`), a
configuration that does not read (`config`), no envelope for the data
(`wrap`), a store that does not read (`store`), a plan or a composition
that raised (`raised`). The turn says one line -- the motive, its fixed
words and its whole numbers -- and leaves nothing behind: nothing is sent,
saved, captured, curated or cached, and an admission it took is given
back, as one is when a stop or a cache hit ends a turn before its stream.

The turn's ledger row carries the window's figures: the rung planned and
the one composed under, the context and the answer's room, the count and
what it is accounted, the epoch and its motive, where it first changed,
whether it was cold or a descent, what each kind of source left out,
whether the slot bound was read, and a refusal's motive and whole numbers.
For a cold window it also keeps the engine's own count of the whole
prompt, its cached prefix included, read on the stream's last chunk, on
Ollama and on llama-server alike -- for a tool turn, on its first request,
the window laid from its first byte. A tool turn's row adds the requests
its rounds sent, the results they cut, the schemas' charge and the
transcript's room its window was composed with, and how many tools its
final called.

## Invariants the contracts hold

- The composer is a pure function: the same registry state, budget and
  retrieval set give the same prompt, and the inputs are not touched.
- The Core bytes in every prompt are the registry's bytes, anchored by a
  hash; an entry whose bytes moved is refused by name, never repaired.
- No eviction without a receipt; no receipt whose key does not resolve to a
  Cellar span -- the digest checks every key before it renders a line.
- A peel stands for a span only once it has answered the recall probes
  drawn from the span, class by class, at the configured thresholds. The probes read English and French: decisions, negations, names and whole words, accents included. An empty probe set on a rich span is a defect, and an unknown
  rate is not a pass. A span no peel answers for leaves the Flesh with
  none: held, its receipt keeping the places in the Cellar of the fewest
  units that answer its probes (a unit that reaches into an order the user
  typed kept by each whole segment that order stands in), or bare -- a span
  the probes draw nothing from, and the spans the catch-up and a shadow set
  aside -- under a receipt that names it, its turns recallable from the
  Cellar.
- The same peel must say nothing its span does not hold. The second face
  of the gate reads the summary as a source is read and refuses, by name,
  a name, a date, a number or a code block its span lacks, inline code its
  span never wrote as inline code, and a sentence
  that decides with no typed decision of the span behind it; a sentence
  whose subject is a reporter -- the assistant, a document, a tool, from a
  closed list in `onion.yaml` -- tells another speaker's words and needs
  none. Nor may a peel give an order, since every later turn reads it: a
  clause that orders, in the forms the `directives` table of `onion.yaml`
  names and on a text folded for them (compatibility forms, format,
  combining and invisible characters, look-alike letters, hyphens, marks of
  emphasis, words spelled letter by letter, lines broken before a word in
  lower case, enumerators and tags) -- in the second person, as an
  obligation of the assistant, under a label that addresses it, after
  "Please", "Do not" or "Try to", as a lasting rule, with an injection's
  signature, as a request told of the user, or with a verb before its
  object -- is refused by name, the user's own restated or told included,
  and the repair drops it wherever it stands. The user's orders reach a peel
  only in the user's words, by reference: each whole segment the user typed
  that holds an order, with the condition, the quote, the list's line or
  the retraction typed with it, by its turn, its place and the digest of
  its bytes, read from the Cellar each time the peel is shown and shown
  only while it still answers to that digest. A span held keeps them by
  the same segments: an anchor that reaches into an order keeps each whole
  segment the order was typed in, never the order's sentence alone, and
  the memory block shows it as one item with every segment of that order
  and of each bare retraction typed after it, as a peel references them,
  or not at all; a fact typed beside an order keeps its own place, shown
  alone only where the block does not show the order's segment; an
  anchor that is a unit only asking, ending on a question mark and
  ordering nothing, ranks after every anchor that states, since an
  earlier question holds the words a question is asked by and not what
  answers it. A decision that reaches into an order is not offered to
  the Core. A span held before this rule keeps its anchors as
  they were made. No block a summary writes is
  honoured, the user's own words written into it included, and the model's
  own markers are taken out first. A decision is
  held clause by clause: a reporter at the head of a sentence exempts no
  clause that opens on a deciding subject of its own after a semicolon, a
  colon, a dash, a comma, an "and", a bracket or a subordinating word, a
  predicate coordinated to a typed decision needs one of its own, and a
  reporter that tells a decision as the user's needs one typed; a decision
  written under a list or after a label that quotes, negates or conditions
  it holds none. The table's comment names what it does not read: a bare
  imperative, an order told as a plan or a permission, an obligation of a
  model or an agent, among others. Two bounds come last, each refused with its figure: the share of
  the summary's content words its span holds no word for
  (`max_novelty`), and its words over its span's (`max_length_ratio`). A
  summary longer than its span saves nothing, and the verbatim turns stay.
- The probes are not taken on trust. The gate reads the span's facts again
  -- its names, dates, numbers, fenced blocks and typed decisions, piece by
  piece as the origins bound them, and never by the native core -- and a
  probe set that asks for a smaller share of them than the floor
  (`probe_floor`) is refused after the bounds, with its share and the facts
  it leaves unasked. A generator, a native twin or a caller that draws less
  cannot raise an acceptance rate.
- A parent peel is summarised from the union of its children's Cellar
  spans, never from the children's text. A peel handed in that stands on
  anything else -- a dangling key, sources that are not its children's, a
  digest that no longer matches the archive -- is refused.
- Recalled content is framed as data with its provenance and wrapped as
  untrusted memory before it reaches the model. What bears instruction is
  the system prompt and the manifest, the Core and the words of the
  conversation's turns -- the current one and the Flesh's, each laid bare
  in its own role, as it was said; every document, retrieval and web
  result rides in an envelope as data.

## The librarian

Off by default (`enabled: false`). A machine switches it on in
`opti_oignon/config/local.yaml`, which git ignores, so no tracked file
changes:

```yaml
onion:
  enabled: true
  librarian:
    model: "qwen3.5:4b"   # the model this machine has for its librarian
```

Only a YAML boolean switches, and it wins over the shipped switch both
ways; only a name that is not blank names the librarian's model, in place
of the shipped placeholder. Either stands without the other. Anything else
in that file leaves what the shipped file says, said once in the log, and a
path handed to the loader is read as it stands. Switched on, the executor offers each
saved conversation to the librarian beside the auto-capture, and so does
every pipeline that saves its own turn, on the journal path; the librarian
mirrors it, and when it has grown by `min_new_turns`, or once a journal
laid it and its Flesh passed the journal's mark, it evicts through the
gate off the interactive path, using the configured model through the
inference registry. Each turn of the conversation the executor answers is
sent as the journal lays it (above), and the working-memory block is never
composed for it; the pipelines that build their own context keep it.

The mirror knows each turn by what it is -- its role, its words, its origin
and its segments -- and never by a count. A retry, a synchronisation from
another device or a deletion leaves the conversation with as many turns as
before, or fewer, and other words in them: where the conversation first
differs from what the librarian holds, the turns that follow are taken back
from the Flesh, and a span already in the Cellar is superseded with every
later one. A superseded span keeps its receipt in the ledger and its turns
in the Cellar, unchanged and recallable, but its peel, its anchors and its
receipt line leave the block and the journal's window, and a proposal made
from it is closed. One
the user was offered still counts in that day's cap, so edits never offer
more in a day than the cap; made again in the same words from the turns
mirrored again, it takes the place of the offer the user already had, and
no more room. A decision the user accepted or declined is not offered
again from the same turn, word for word, when that turn is mirrored or sent
again: a copy deferred before the verdict is never offered, and one already
offered stays an offer of its own, decided one at a time. The
conversation is mirrored again from there, under turn ids never given
before, so a step that read a span before it was taken back evicts nothing.
A synchronisation makes every received turn legacy, so it supersedes the
whole Cellar and the conversation is summarised again, a burst at a time.

Each call to the model asks the resource governor for a ticket of its own,
as `librarian`, a background caller, for the context `librarian.num_ctx`
names; between two calls a burst holds none, and the governor's background
gate keeps the next call waiting while a chat turn is in flight. A span
whose prompt and answer do not fit the window it was admitted at, less
`librarian.window_margin` for the token estimate, is not sent; neither is a
call the governor does not admit, nor one past the run's budget of time on
the model (`librarian.run_budget_s`; with the call in flight, a run spends
its budget and one call at most): the step goes on down the rungs that need
no model. An admission whose call can no longer start is handed back to the
governor. The model stays resident for a burst or a close
(`librarian.keep_alive`) and is let go at its end through the governor,
which unloads it only when the background alone loaded it and no call is in
flight on it. The librarian's shipped model is none of the models the
shipped routes answer with; a machine may name one that is, and its calls
are then served by the chat's load. The governor's floor for `librarian`
(`ctx_floor`, the ladder's smallest rung) makes every rung acceptable to
it: a resident serves a call at the context it holds instead of refusing
the reload a larger `librarian.num_ctx` would take, so a librarian that
shares the chat's model runs at every rung, and a load of its own model
that does not fit at `librarian.num_ctx` steps down to the rung that fits,
in free memory only, as the background always loads. A span that does not
fit the window admitted, less the margin, is not sent. The engine keeps a
model for the keep-alive of the last call that ended on it, so a call on a
load the background did not make, or one a higher class has used since --
the chat's, or one loaded out of the governor's sight -- carries the
residence the foreground gives its loads, the warm-up's keep-alive that the
chat sends with each turn, in place of the run's shorter one: the librarian
never cuts the residence of a load the chat holds. A load the background
made and no higher class has used -- the librarian's own, or another
background caller's, the warm-up's preloading included -- keeps the run's
residence and is let go at the run's end; a pressure override still wins.

What the queue counts -- bursts, evictions by rung, refusals by motive, the
block's events, proposals, the mirror, the residence, and the native core's
share of the probes -- is kept across processes in a file of aggregates
(`counters.path`), names and numbers only, written whole under a lock and
never written over when it does not read as counts: at the end of each
burst and close, after each decision on a proposal and each listing that
opens deferred ones, when the terminal session ends by `/quit`, the end of
its input or an interruption, and from the chat path at a turn once
`counters.flush_every_s` has passed. What a process counted since its last
write is lost if it stops before the next, a hang-up or a kill included.
`GET /api/memory/onion/status` and the terminal's `/status` show it, each
count once.

The librarian asks its model for a summary in the language of the turns
that attributes each decision to its source: a decision is the user's
only if the user typed it, and what the assistant, a document or a tool
said is told with them as the subject, the form the second face accepts.
It asks the model to give no order and to restate none, whoever gave it:
the user's orders are kept apart, in the user's own words. The words the
peel shows as written are left out of the turns each call gives the model,
since a summary that retells them only lengthens the peel, and its
instructions hold a budget of words: a share (`budget_share`) of what the
queue's compression floor (`rho`) leaves the summary once those words are
counted, a target the gate does not enforce (it judges a summary by its
length bound; the repair alone holds a peel to the floor). A span whose
shown words already pass that floor is not summarised, and the queue counts
it (`brief`, `no_room`): its peel could not save enough, and a peel that
saves a twentieth of its span takes the place of others in a small window,
where a span held leaves anchors found again by what is asked. The model
is asked for plain prose that names, in each sentence, who said what it
tells: an answer told bare would make the assistant's claim a fact. What
the request changes on real conversations is a host measurement; on the
drift A/B it is in the changelog.

Switched on (`statements.enabled`, off as shipped), once a burst has
evicted what it could, the librarian reads what the spans it has not
read state, oldest first, at most `statements.per_burst` of them, under
the same run: one more call a span, handed the words the
user typed in it -- never the assistant's, a document's, a tool's or a
peer's -- and the subjects the conversation named before, as data, asking
for one line a fact, `turn | subject | relation | value`, each subject
named as the turns name it, never by a pronoun. No line is trusted until
the span's words bear it out: it is kept only when its turn is one the
user typed in the span; its value's words stand one after another, case
and accents aside, in one of that turn's typed sentences; its subject
holds none of the section's pronouns -- "you" and the words of a group
among them -- and is written so in a sentence the user typed in the span
up to that turn, or is a subject already known, or is the user in a turn
that holds a first-person word -- "Bob" is not written in "Bobby"; its
relation holds one to eight words; and the sentence it is placed in is
known: the first that holds its value and writes its subject, else the
one sentence of the turn that holds its value. A value two sentences
hold, the subject written in neither -- or the user's in two sentences
that say "I" -- places nothing, so a line the model wrote of one
sentence never lands in another's by its value alone. Every other line
is refused and counted by its motive, never kept. A span the user typed
nothing in states nothing, with no call; a span the window cannot hold
is passed over for the next and asked again by a later reading; any other call refused or failed stops the burst's
readings, and a model a step stopped asking is not asked to read. What
is read is kept in the process and never persisted, so a restart reads
the spans again; switched off, nothing is read.

What the newer words do with what the librarian read is the section's
`newer_words`. As shipped, `label`: each unit they lay again whose
statements the librarian placed in it says whom it speaks of, before its
words -- "(about Alice) News: the move is done, Alice lives in Oslo.",
"(about Alice) She lives in Oslo now." for a pronoun the librarian
resolved, each subject once in the order read, the user's as "the user".
A label shows no word the window could not: a subject is named only as a
sentence the user typed writes it, in a standing span no later than the
unit's and a turn no withdrawal reached -- one known only from a turn a
withdrawal reached names nothing; by its words alone -- letters with
their marks and digits in any script, composed, an apostrophe, a period or
a hyphen inside one -- at most `label_words` words and `label_chars`
characters, never one that reads as the user's; and, when the unit does
not write it -- the user's: a first-person word -- only on a unit that
refers by one of the section's pronouns, never on one that names its
subject itself. A word the user typed both in a turn a withdrawal reached
and in another still names: what the label says of a pronoun, that it
refers to that name, the reading may have drawn from the turn withdrawn.
The label is defanged with the words it stands before. Nothing is left
out; a unit of the Flesh, of a span not read, or one the librarian placed
nothing in, is laid bare. A pronoun the librarian misreads labels its
sentence with another's name: the label tells the model whom the sentence
speaks of, and it can be wrong as the reading is.

With `leave_out`, the reading takes back newer words instead. When a
question names a subject the statements know -- every word that names it
among the question's terms; never the user, whom "remind me" or "we"
addresses as often as it asks of -- a unit is no longer laid whose
statements, those placed in it, are of nothing the question refers to:
none is the user's when the question says "I" or names the user, none has
a subject that shares a word with the question, none is said of the
user's own relation ("my wife", who may be the subject asked under a
name), and the unit's own words name no subject asked. Asked where Bob
lives, "News: the move is done, Alice lives in Oslo." is not laid; asked
"Bob asked when the release is", the Harvest release's update is.
Nothing is laid in its place, so an older unit never stands in for a
newer one; a unit of the Flesh, of a span not read, or one the librarian
placed nothing in, is laid as before, and so is every unit for a
question that names no known subject. A reading that misnames a subject
can only take back a unit the newer words would lay, never add one; but
a question that refers to a subject by a word no statement of it holds
-- "the launch" for "the Harvest release", "the CEO" for "Dana", and as
much "releases" for "release" or "deploy" for "the deployment" -- while
it names another known subject loses that subject's newer words, which
a label never takes away.

The section ships off. A machine switches the reading on in `local.yaml`
(`onion: {statements: {enabled: true}}`, with the onion's own switch and
its librarian's model); only a YAML boolean switches, the shipped switch
standing otherwise, and a key the section does not take is said once.

The user's surface on the Core and the receipts is a memory route scoped
to the conversation: `GET /api/memory/onion/{conversation_id}` lists the
Core entries and the open receipts, `POST .../pin` pins a statement as the
user, `POST .../supersede` pins a successor and links the old entry to it,
`POST .../recall/{key}` hands the verbatim span back and leaves the receipt
open, and `POST .../resolve/{key}` closes it as the user. Reading is not
closing: no tool a model can reach imports either verb.
`GET .../proposals` lists the decisions the user typed that the queue
offers the Core, word for word, and `POST .../proposals/{id}/accept` and
`.../decline` decide one at a time, as the user. The route imports
the librarian inside its handlers and passes
the user as actor; the store refuses any other actor, and a pin that would
push the Core over its cap is refused before it lands, because the
composer never cuts the Core: over its cap, the Core blanks the block, and
every journal turn is laid at a larger rung, or refused `core` when no
rung holds it.
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
wrong entity or date in a four-turn span passes the probes at 0.7, and the
second face refuses it by name. The bounds are proposed defaults too: on
the fixtures a faithful summary reads at most a tenth new and one that
drifts two thirds or more, and a faithful summary that tells who said what
can be longer than a short span, which keeps the verbatim turns. The
runbook reports what the second face refuses, kind by kind, and the spread
of each bound's figures -- how many are over, the median, the ninetieth
percentile -- the figures their calibration is owed on.

A rate counts the probes a summary answers; it cannot count a fact no probe
was drawn for. The recall instrument, in `opti_oignon/memory/probes.py`
beside the generator it measures, reads a labelled set -- spans whose facts a reader marked by class, at the turn
that first writes them -- and counts, class by class, the facts the
generator draws a probe for: of their class, at their turn, with their
answer. A date or a number is answered by its canonical form, a name by its
writing, a code block by the marker of its body, and a decision by its
polarity and the acts, dates and numbers it was marked with; the other
words a decision is keyed by are the generator's, never compared. The
recall read on the fixture set the contracts hold, French and English, is
stated in `onion.yaml` under `gate.probe_recall` with the version of the
generator and the fingerprint of the lexicon it was read with, and a
contract reads it again; the facts the set marks as known misses are held
missed. Every gate decision, every fidelity reading and every rate of the
runbook carries that recall, or None and the reason when the generator or
the lexicon differs: the lexicon is the user's to change, and a figure read
on another one is never borrowed. It is a fixture figure, not a
calibration. `scripts/onion_runbook.py --probe-recall SET` reads it on a
labelled set of real conversations, kept off the repository, with no model
and counts only, never a writing or a turn of the set; that reading is owed
on the host.

The recall reads the generator against a reader's labels; the floor reads
every probe set against the span itself, with no label. On the fixtures
every probe set the generator draws asks for every fact the span holds, and
the shipped floor asks the same of a native twin, a caller or a later
generator. It is set from the fixtures, not calibrated: the runbook reports,
span by span, the spread of the share and how many spans fall under the
floor, and the facts left unasked counted kind by kind, never written out;
its calibration on real conversations is owed on the host.

Drift against a no-onion baseline has a host command of its own,
`scripts/drift_ab.py`. It runs one scripted conversation -- facts stated,
replaced on known turns, and asked about long after the history window has
dropped them -- through two arms that differ the way the chat path does:
the same model, the same system prompt first and the same turn last, and
between them the plain arm's history window or, in the onion arm, the
journal composed for the turn, planned within the model's window and
counted with the model's counter as the executor's, sent with the context
it was composed for; the report carries what the memory laid and the
Flesh beside the window's whole count.
The librarian runs in process
with persistence off, so neither arm reads or writes the data directory.
Each arm gets contradictions per 1000 turns against the facts holding at
each turn, the sentences that agree with them, and the sentences the
templates cannot decide, which are left to a model judge and never counted
as agreement.

A change of the onion's composition is measured on the same states, never
run against run: a run's later states follow its own answers, so two trees
run end to end differ by their course as much as by their composition, and
one run's verdict on a probe proves nothing. With `--states DIR`, a new
directory, the onion arm keeps at each probe turn its state before the
window is planned -- the bench's own conversation, in a store file in
plaintext under DIR, never the data directory -- with the turn, the
conversation as stored, the system prompt and home rung it is planned
under, whether a burst held it, and the window it sent; `--home-rung R`
plans the arm under `home_rung` R, in the process alone.
`scripts/drift_pairs.py compose` loads each kept state again from a copy
of its store and composes its probe's window under the tree it runs from,
at the home rung it was kept under, then asks the answering model with the
bench's sampling and reads the answer with that tree's reading, refusing a
kept turn that is not its probe; `compare` reads two such files probe by
probe -- the windows that differ, and, when both were asked, the probes
each tree answers right and what the second gains and loses. Under the
tree that kept them a state composes again to the window it sent, unless a
burst held the conversation when the run planned it. On Ollama each window
is asked of a model loaded for it alone, the bench's trial line asked
first so that a model left loaded answers no window warm: an engine that
reuses the prefix a window shares with the one asked before it, computed
in other batches, moved a 4B model's log-probabilities by about a nat on
the maintainer's machine, enough to turn a verdict. A verdict is one
greedy draw, and a small model flips it on a window that gives it no more
and no less to go on: with `--margins K` the model is asked through
Ollama's own API on this machine -- its own penalties and draft model held
off, since a penalty weighs on a value the window lays near its end after
the log-probabilities are read, and a draft model's tokens carry none --
each token of its reply with its K likeliest alternatives, and each probe
carries its margin, in nats: with a stale statement the probe's expected
one retired, at the token where the two part, the log-probability of going
the expected way less that of going the stale one's; with none, at each
token of the value, the least against every other token; a bound when one
side is not among those read. Each probe also names the token its margin is
read against: the stale value's, or, with none, the likeliest other -- often
the sentence form's own placeholder ("<"), a copy of the form rather than
another value. `compare` refuses two files asked otherwise and says how each
probe's margin moved, those read exactly that moved by a nat or more either
way, and, probe by probe, those read exactly on both sides that moved or
changed what they are read against, with both tokens: a margin lost to
another subject's value is not one lost to the copy of the form.
Measured on the maintainer's machine with K of 20
over the 91 kept probes, cold and unpenalized: under the change that lays
the user's words of a newer statement, a 4B model's update probes gained
2.8 to 9.1 nats and a 27B model's 3.4 to 9.2, one verdict of the 4B's and
two of the 27B's turned right, and the last question, on another subject,
lost 1.2 to 1.9 nats for the 4B and 1.0 for the 27B where no verdict
turned; the change that left out the anchors that only ask turned four of
the 4B's verdicts on margins that fell by 1.5, where run warm and
penalized it had turned eleven.

What the librarian reads is measured the same way, in a tree that
switches the section on: `drift_pairs.py read`
asks a model, as the librarian asks it, what the kept states' spans
state, once a span, and `compose --statements` lays what it read before
each window is planned. On the maintainer's machine a 4B librarian
(temperature 0, seed 1), handed the words the user typed, read the 401
spans of the 77 kept windows' states and kept 99 statements; it read
eight of the bench's nine facts and updates in all eleven sets, each with
its subject right, and the budget review's date in none -- an omission,
whose sentence lays as before. Ten windows changed, each by one unit of
another subject left out -- Alice's move for Bob's question, Bob's city
for Alice's, Carol's employer for Alice's -- every message before the
turn's unchanged; no verdict turned; the 4B's margins moved by +0.8 nat
in sum, Bob's city at 16384 from 6.3 nats ahead of Oslo to 8.5 ahead of
Paris, and the 27B's by +0.4. On the twelve questions of ten
conversations written to break a reading by surface -- a pronoun, "my
brother", "my wife" with her name, a value two subjects share, "they",
the first person, a product's team -- each typed fact a span of its own,
the 4B reader left out the two units of another subject and laid the ten
updates of the subject asked; it read nothing of "They live in Paris
now." nor of "it ships on Thursday now", laid as before. A 0.8B reader
left out no update of the subject asked, and two units of another
subject stayed laid.
Laid with whom they speak of instead (`label`, the same readings): 34
of the 77 kept windows changed, each by the labels of the units laid
again, every message before the turn's unchanged; no verdict turned; the
4B's margins moved by +9.5 nats in sum over its 27 -- from -0.7 to +2.7,
the Harvest release's date at 8192 +2.7 in four sets, Bob's city at
16384 +0.7 against Oslo with Alice's move still laid -- and the 27B's by
+0.02 over its 7. On seven windows whose update of the subject asked is a
pronoun's -- "She lives in Oslo now." after "Alice called me yesterday.",
in English and in French -- read by the 4B and asked of it, the labels
moved the margins by +4.8 nats in sum, one of them down by 0.2, no
verdict turned.

## The native core

The integrity primitives -- the four hashes -- and the window assembly
have a second implementation in Rust, `rust/oo_core`, built into
`opti_oignon/native/` by `scripts/build_oo_core.sh` and never tracked.
Python stays the reference and the fallback: the memory modules ask for
the native core at the call, never at import, and run the reference path
when it is absent. When it is present, every hash is byte-equal to the
reference and every assembled prompt is field-equal, and the contracts
that hold it to that run on every machine that builds it. The core counts
no token: it assembles the parts the composer counted, so the window has
one count, and a text it cannot carry, a lone surrogate, is assembled by
the reference. The crate is
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
