# Changelog

All notable changes to Opti-Oignon are documented in this file.
Security-relevant changes are marked with [SECURITY].

## Unreleased

The merge guards learn to say what they cannot prove, rather than reporting an
absence of proof as a verdict, and a new one measures what importing the
package costs.

### Added

- [SECURITY] Pasted text is a document, never the user's typed words. A
  page, a mail or a log the user pastes carries whatever its author wrote,
  orders included, and it was sent and saved as typed: its words could
  endorse a memory or notes write, a tool call's argument, or a decision of
  the memory. The composer now keeps, character by character, what was
  typed and what was pasted or dropped, through every edit (any input it
  does not know as typing -- a paste, a drop, an undo -- is held pasted),
  and sends the pasted ranges with the message (`pasted`, in code points;
  out of shape, or more than `config/chat.yaml` allows, the request is
  refused by name and nothing runs). The route saves each range as a
  document part of the turn. Words typed beside pasted ones take their
  sense from them ("Never" pasted before "share my location with Bob"
  typed), so a turn with a paste in its words has no typed unit at all:
  its words endorse no write, no argument and no decision, in every reader
  of the memory, the automatic capture, the pending writes and the
  provenance gate alike; the automatic capture no longer keeps such a
  turn's words, and the manual extraction proposes them with the rest,
  never writes them. A file attached after the words keeps its rule. A
  turn with a paste gives the coding agent no directive; a /code the user
  pasted starts nothing, on a retry either, which reads who wrote it from
  the stored turn (a legacy turn's is no one's, and a turn a hook rewrote
  is the hook's, whoever typed the words it rewrote: a fresh turn reads
  the command before the hooks run; such a retry starts the agent only
  when /code is typed again, and reads no directive the hook wrote), and
  the composer no longer shows the coding agent for it; a hook's rewrite of
  words that held a paste is saved as no one's (legacy). `oo ask` sends
  what it reads (`-f`, standard input) as a document beside the typed
  prompt, which keeps its standing (`-f` used to replace the prompt, and
  `--pipe` both the prompt and the file); with `--json-out`, `prompt` is
  the typed prompt alone and `documents` names what was read beside it (a
  server older than attached files drops the document); `oo chat` holds a
  line read from a pipe or a file as pasted, and refuses a command read
  that way, but for /help and /quit (a mail piped in could carry
  "/accept"). A turn without a paste is composed byte for byte as before.
  Not told apart yet: text pasted at an interactive terminal's prompt, an
  agent run's task (typed as its caller vouches), a message a client posts
  to a branch, and text a phone keyboard or an operating system's writing
  tool inserts as if typed (owed to the machine).
- [SECURITY] The provenance gate (`opti_oignon/provenance.py`,
  `config/provenance.yaml`). A tool call is the model's choice, and so is
  every argument it carries: words the model read in a page, a document, a
  tool result or the memory reach an argument as easily as the user's own,
  and a call that reaches the network can carry them out of the machine.
  Both passage points -- the chat's tool executor and the agent's dispatch
  -- now label each argument of every call as it will run, after its name is
  repaired, its default filled and its value coerced: `typed` when, folded,
  it equals a part the user typed in the current turn, whole (for an agent
  run, its task); `default` when it equals the tool's own declared value,
  type and all; `unendorsed` otherwise. Every tool has an effect class from
  a closed table (none, session, sandbox, deferred, network, approved), and
  a tool the table does not know is held as networked. In Daily, the
  network policy says what a network call carrying an unendorsed argument
  does: `free`, the default, sends it as before; `ask_unendorsed` holds it
  for the user, who sees each value whole with its label (a value too long
  to be shown whole is refused instead of asked); `refuse_unendorsed`
  refuses it. Only a missing file, or one that says `free`, is free: a file
  that is present but cannot be read, names no policy, or gives a value
  outside the three reads as `refuse_unendorsed`, and a policy given with a
  turn can only tighten the file's. An endorsed network argument is handed
  to its sink as the typed part's own characters, never the model's
  spelling, so a planted instruction cannot hide a few bits of its own in
  spacing or a Unicode form (the search may still scrub personal data from
  it). An argument that stays on the machine is labelled `typed` only when
  it already is the typed part, character for character. Each result
  carries its labels, its class and the decision. The provenance of a turn
  travels with each call, never on the shared executor. Not measured here:
  how often real models make the gate ask or refuse, and what a demand costs
  in time; both are owed by the machine.
- `oo approve <id>` and `oo deny <id>`, and `oo ask` shows a tool call held
  for the user's answer -- each value whole, with its label, every
  character that could rewrite the line written as its escape -- asks at
  the keyboard when there is one, and says how the call ended; an answer
  that comes too late no longer ends the reply. The approval drawer, the
  agent panel (which used to show a one-line summary only) and the approval
  queue show each value whole up to 2,000 characters, labelled or not, with
  the value's own length and lines as the queue counted them, and say how
  long a longer one is, never cutting inside an escape. Every character a
  screen would hide is written as its escape -- controls, direction
  overrides, zero-width spaces, every blank but the plain space (a tab, a
  no-break space), and every code point Unicode lets a screen draw as
  nothing (variation selectors, tag characters, fillers), each of which
  could carry bits unseen -- with a backslash doubled, so what is shown
  determines what is sent. A list or a mapping is shown as its JSON, read
  by JSON's own rule, up to 2,000 characters as shown, and its cut says
  both lengths; a number JSON cannot carry (NaN, an infinity), a browser
  would print otherwise (a negative zero) or cannot read exactly (a whole
  number past 2**53) is shown as its text. An argument's name is shown the
  same way, on one line. The terminal prints each argument's name, label
  and length before its value, every line of the value behind a bar, a
  name that is not a plain identifier behind a bar of its own, and cuts
  every row to the narrowest width it can learn (`COLUMNS`, and every
  terminal the process is attached to, standard error piped or not; 80 when
  it can learn none), so that, on a screen at least that wide, no value,
  name or summary can print a row that passes for the terminal's own.
- The write census guard (`.github/scripts/write_census_guard.py`). A gate
  on a store's writes protects only the writes that go through it, so the
  guard finds every write into a store a model reads back -- the facts with
  their rows and vectors, the frozen legacy memory, the extractor, the
  notes, the skills, the Core, the onion's peels, receipts, cellar and live
  turns, the transcript and its branches, the two response caches, the
  projects and their index, the ingested documents and their index, the
  prompt templates, the coding agent's working memory and checkpoints --
  and holds each one housed in its own store, gated, exempt with its reason,
  or owed. It reads the package's own code, statically, and catches its
  drift in any form ordinary code takes; code written to hide a write from
  a static reader (a class rebuilt by a metaclass, a name assembled from
  strings) is beyond it, and the guard lists those forms rather than claim
  them. It follows a store object through a module's bindings and across
  modules: what a module exports or re-exports, by name or by a star, a
  package's lazy export table, a module looked up by its name or relayed by
  another module, an argument handed to another module's function, class or
  method or to a dispatch table, or handed with a function to a task, a
  thread or a pool, an object of any class that keeps a store on itself or
  hands one back, a dataclass or a named tuple that holds one, a dependency
  or a context manager that yields one, a subclass, a property, a container
  of stores; and a private member of a store reached outside its house is a
  site too. A gate counts by dominance on the syntax
  tree, never by line order: an early exit on a refused verdict, a guarding
  condition, a raising check, the user's approval, a filter whose result is
  all that is written, or the gate's own body -- never a function, a lambda
  or a generator nested in it, nor a writer it hands out; a function is
  gated by its callers only when every reference to it, anywhere in the
  package -- through re-exports, star imports, decorators and ``getattr``
  -- is a call from a gated position; and a gate whose authority is its caller's (an
  approval, the Core's actor, an acceptance) holds only when a route or the
  terminal client reaches it. Six kinds of exemption are checked by a
  predicate (only a route handler reaches the write; nothing reads the
  store back; the write binds its whole context; it adds no content; it
  runs only as a program; its receiver is a store the module built in a
  temporary place); the others argue their reason in prose, cover the
  sites they name by function and method and no other, and the green line
  counts them apart. Every module that opens a database is a store's house
  or is named outside the census with the reason, and every method of a
  store that spells an SQL write -- in its own strings or in a constant --
  or reaches one of its writes, through a private helper too, is in the
  table or set aside with its reason; two were not, and now are: the
  conversation store's import of the legacy JSON history, a write its
  module's self-test alone runs, and the semantic cache's lookup, which
  counts its hit as the response cache's does. On this tree: 198 write
  sites in 22 stores -- 41 housed, 23 gated behind 11 gates each proven by
  contracts the selection rule runs, 119 exempt (84 by a checked
  predicate), and 15 owed in 6 modules: the pre-cache stores answers with
  no context fingerprint; a
  synced fact, note, skill or conversation lands without the gate that
  admitted it on its peer; the documents' auto-refresh re-ingests a changed
  file with no gate; a coding checkpoint's plan, written by the model, comes
  back to the agent when the user resumes the task; and a caption or a
  transcript is written back on a request that carries the user's approval
  but runs the model again, so the text written is not the text the user
  approved. Each owed module is sealed by its digest: the debt may only
  shrink, and a module that changes while it owes must pay. The guard runs
  in CI and in the ladder's guard tier.
- Two contracts prove what nothing tested: a caption or a transcript drawn
  from a note's attachment is written back to the note only on a request
  that carries the user's approval.
- The review of pending writes. The Memory panel gains a "To review"
  section, and the Notes panel the same for notes: each write the agent
  proposed, or each fact the manual extraction drew from anything but the
  user's typed words, with what accepting it would do, each of its words
  marked typed by you or not, what an update would change, and what the
  agent had read before proposing it. A checkbox per proposal and one for
  all; "Accept selected" applies the batch in order, each proposal once and
  exactly as proposed, and "Decline selected" writes nothing. A failed
  write stays waiting. Served by `GET /api/pending-writes`,
  `POST /api/pending-writes/accept` and `POST /api/pending-writes/decline`
  over a queue beside the other stores, encrypted at rest the same way. A
  run makes at most 20 proposals (`max_per_run` in
  `config/pending_writes.yaml`); a proposal already waiting is never queued
  twice, and a write the user declined in a conversation is not proposed
  again there (another conversation may ask again; a run with no
  conversation keeps no refusal). A write that reached
  the store before failing is decided rather than put back, a change whose
  fact or note is gone is reported with nothing saved, and an acceptance cut
  short by the end of the process is completed by the next review after
  `stale_claim_seconds` -- settled if it had landed, written once if not,
  never put back. Each proposal names what the run had read before it:
  web results, files, command output, and the facts, notes or skills it
  looked up. The per-user wipe deletes the queue and the export (format
  1.2) carries it. The manual extraction answers with
  `facts_proposed` beside `facts_added`, and both its passes, like the
  automatic capture, ask the model even when there is a single turn to read
  (the extractor had required two, so a conversation of one typed question
  had its facts read by the pattern fallback alone).
- The chat's loader, drawn from what the server did. While a reply is
  written, one status line stands where it will appear: its words come from
  one closed table that has passed the garden's two ethics nets, the onion
  stands beside them until the first token, the seconds show from the
  fifth, and after 30 seconds without any frame, pings included, the motion
  stops and the line says how long the server has been silent; the next
  frame resumes it.
  A run the server executes (an execution pipeline, a reasoning strategy, a
  consensus, a self-correction) shows a card instead, held at the top of
  the thread while the run is open: the run's name and the step it is at,
  one plant per step on one soil line, and one line per step with its
  state, its sub-progress and its duration in words. A plant grows only on
  a fraction the server sent and never shrinks, and each of its looks is
  held at least 400 ms. Every end has its own drawing and its own words: a
  failed step keeps its plant as it stood, in straw, with the machine's tin
  tag on its stem; a stopped one keeps it too, in the stop's own ink; a
  step skipped or not run grows nothing; and a step still open when the
  stream closes is drawn in dots, never as failed. After `done`, the
  reply's footer carries one summary line, a disclosure, read only from the
  steps' record `done` carries: a reply without that record, a reloaded one
  included, has none. No step text is parsed, and a status the word table
  does not map changes nothing visible. One status region speaks for the
  whole stream, the last of a burst after 400 ms of calm and at most one a
  second, never the seconds; the reply loses its `aria-live`, and the
  thread is a named region rather than a log. A Stop now reads the end: the
  cancel is sent first and the socket is read on until `done` or `error`,
  or 10 seconds, so the server's closing frames arrive and the run says
  where it stopped; a stopped reply keeps its text, marked `stopped`, and an
  error keeps the partial reply beside it. A connection lost before the
  first frame is retried up to three times, and the line says so. The
  composer's input stays usable while a reply streams; sending still waits.
  Sending and retrying read the stream through one reader, which knows
  `pipeline_step`, `ping` and the tool-call approval frames, and one reset
  clears every streaming store on send, retry, done, an error and a Stop.
  Two palette roles, `dried` and `machine`, join the three palettes.
  `StreamingIndicator`, `VisionDelegationIndicator` and `OnionLoader` are
  removed. Not yet drawn: the sub-lines of a nested run, the words of the
  pipeline a step really ran as in the unfolded summary line, and a drawing
  held still while a tool call waits on an approval, a request waits in the
  queue or the stream reconnects. Owed to the machine: a real run on
  Ollama, a Stop and a refusal mid-step, a server cut, the palettes at
  several widths and pixel ratios, forced colours, reduced motion and a
  screen reader's order.
- A `pipeline_step` frame on the chat stream, so the interface can draw each
  step of a run from what the server did rather than from status text. An
  execution pipeline, the three reasoning strategies, a consensus and a
  forced self-correction announce their steps before running them, then
  report every change of state (pending, running, done, failed, skipped,
  cancelled, not run) with a sequence number per reply, a duration measured
  on a monotonic clock, the agentic pipeline an execution step really ran
  as, and a fraction only where its units are the step's whole work: the
  models of a consensus query, the samples or sub-steps of a nested run. A
  step whose work raised, or which returned one of the server's own fixed
  error messages, ends failed, read from a signal and never from the reply
  text; Stop, Stop all and a resource refusal never end a step failed, but
  cancelled or not run, with the reason. The frame is never dropped by
  backpressure; on Stop and on an error every open step is closed before
  the stream says so, and `done` carries the last state of every step. The
  "Step k done" status is no longer sent (it was sent after a failed step
  too); the "Step i/N" status stays for the terminal and the logs. With it:
  a resource refusal inside a pipeline step now ends the run, where the
  next step used to receive the refusal text as its previous analysis; the
  backpressure trim could lose an event appended while it ran, a critical
  one included, and now replaces only what it read; reasoning steps and
  consensus answers reach the socket once instead of twice; a status that
  only names vision is no longer tagged as an image analysis; and a
  structured chunk of an unknown kind is dropped with a log line instead of
  being appended to the reply text. Unchanged, and said: a self-correction
  stopped after its first draft has already saved that draft. The frame is
  documented in the API reference.

- A command palette. Ctrl+K opens it, and so does Search in the sidebar,
  expanded, on the rail and in the phone's drawer: one dialog, mounted once
  by the layout of both spaces, that reaches every place and every action
  of the interface from one field. It lists the pages of both spaces (the
  componion's once it has a page); every command, the nine default
  shortcuts among them, "Stop this reply" and "Show notifications" (which
  goes to Preferences and opens the notification history there), each
  beside the keys that run it for this reader, the reader's own keys when
  they changed them (the sidebar's hint beside Search shows them too); each
  of the fifty settings groups that has a page, found by the words the
  settings page's own search reads (its title, its other names, its
  description, the page that holds it and the old section that listed it),
  an embedded group opening its host's page and a retired one never listed;
  and the chats: the recent ones while nothing is typed, then the server's
  search over titles and messages, asked once the typing pauses, the
  request before it aborted when new words are asked or the palette closes,
  and an answer to words no longer asked dropped, in whatever order the
  answers come back. When the server found more chats than the palette
  shows, a last entry opens the chats index on the same words, where every
  match is listed with its limit notice. The words meet a name whole, at
  its start, at the start of a later word, then by a command's or a group's
  other names, then by the name's letters in order, case and accents
  aside; the chats the server found are kept after those; each group shows
  six at most and the groups follow their best match. A command that
  cannot run where the reader is stays in the list, in the muted ink, with
  its reason beside it ("Open a chat to export it", "No reply is being
  written", "Open a chat to send a message"); opening it says why in the
  status line, and runs nothing. The field is a combobox over a listbox of
  labelled groups: focus stays in the field, the active entry is named to
  assistive technology and drawn with a ring in the focus ink. The active
  entry is held by what it is, not by its place: the entry the reader moved
  to stays active when the list is ranked again under it (a chat the server
  found jumping to the top no longer takes the reader's Enter), and
  otherwise the first entry that can run is, never a disabled one; each
  entry's id in the page follows what it is, so a new best match is read
  out. The arrows move and wrap, Home and End take the first and the last,
  Enter opens, Escape closes. A polite status line says how many results
  are listed once the list has stood still a moment, that nothing matches,
  or that a slow search of the chats is still running; a burst of keys is
  read once.

  Stop all sits in the palette's head, beside its close button, in every
  state and at every width, so the stop stays one tap or click away while
  the palette covers the page: above the field, where a phone's keyboard,
  raised by the field, never covers it. Typing "stop", "emergency" or
  "halt" finds a Stop all entry too, which opens that control's
  confirmation and puts focus on its first action; nothing stops until the
  reader chooses, and the palette never stops anything itself. On a phone
  every entry is a 44 px target, a disabled command's reason wraps under
  its name instead of being cut, and the field's type is 16 px at least,
  so the phone does not zoom into it. The ds Modal gains an actions slot in
  its head for this, and on a touch screen its close button is a 44 px
  target. The list of shortcuts and the export dialog, which the keys and
  the palette open, hold Stop all in their head as well (the list only
  while it is open); the modal dialogs that do not yet, sixteen of them,
  are named in a ledger that only shrinks. Escape pressed in the stop's
  confirmation shuts it and goes no further, so the palette, a dialog or
  the phone's drawer around it stays open.

  Every shortcut is a command of one registry, which says for each where it
  cannot run and why; the shortcut handler starts from it, applies the
  reader's own keys over it as before, and runs every command through one
  runner that asks for that reason before any handler does, the same runner
  the palette uses: the palette judges its commands from inside and runs
  the one chosen in that same context once it has closed, so what it shows
  disabled is what the runner refuses. A key pressed while the palette or
  the list of shortcuts is open runs only what closes them: Ctrl+Enter
  typed in the palette's field no longer sends the draft of the chat under
  it, and Ctrl+N no longer opens a chat behind it. On the sign-in,
  registration and component gallery pages, which have no shell and no
  palette, only the theme and closing a dialog run: Ctrl+K no longer
  leaves the palette open with nothing on screen, to pop up after sign-in,
  and the palette shuts its store when it leaves the page. The root layout
  hands the handler nothing, and nothing a shortcut runs reads the page's
  markup: Ctrl+K no longer looks the sidebar's field up by a data
  attribute, and that field is gone, its place taken by the Search entry
  (the chats index keeps its own search). A key the application binds with
  a modifier is the application's even where its command cannot run, so
  Ctrl+K pressed in the open palette does not fall through to the browser;
  a plain key (? or Escape) that runs nothing keeps its default. The list
  of shortcuts (?) names each by its command and says the palette holds
  them all. The form field can now be a combobox's field, and a
  conversations request carries an abort signal down to fetch.

  svelte-check still reports 34 errors, none in the new files. Border-token
  reads fall by 2 (the shortcut list's key caps sit on the sunken ground
  with the edge). The palette, its store, its modules and runner, the
  shortcut handler, the store of its list, the store of the keys each
  command runs by and the store of the notification history's panel join
  the files held to the surface rules. What the contracts prove is what the
  templates emit when compiled for the server and what the ranking, the
  registry, the sources, the conversation source, the follower that keeps
  it in step and the active entry and status line answer under Node; focus
  returning to the opener, the arrows, Home and End in a browser, a screen
  reader reading the active entry and the status line, Stop all above a
  phone's keyboard, and the latency of a search on a real store are
  checked on the machine.

- A fact-check core, `opti_oignon/factcheck`, that decides whether a claim is
  supported by the evidence it is handed, and says "supported" in one case
  only: a whole sentence of an admitted source, valid at the date the claim is
  about, restated verbatim under a fixed fold (typographic quotes, dashes,
  spaces, ligatures and the ellipsis; never case, digits or NFKC), located
  again by the host in a chunk whose SHA-256 it recomputes, in a window that
  does not qualify it. "Supported" means supported by that passage of that
  source, read on that date. Everything else is "not enough evidence" with
  every reason found, or "out of scope" with its reason; a claim inside a
  longer sentence ("It is false that ...", or the tail of a sentence cut at
  an initial such as "George W."), a moved negation, one digit, one added word
  or a splice of two sentences is never supported, and no evidence is never
  "contradicted", a verdict this core does not give at all. The window a
  source sentence is read in is its heading path, the heading its store
  gives, its lead-ins (a parent list item, a label set above a paragraph, a
  raw HTML heading), and the sentences or list items beside it: a qualifying
  word ("Myths", "wrongly", "refuted"), a denial, struck-through text, an
  attribution, or a frame of condition, forecast, narrower population or
  negation set above it blocks support, the marker and its place recorded;
  so does a context a chunk may have cut. A model's words are never evidence:
  a model or unknown author, a model-quoted range, a search excerpt, a source
  without consent and a retracted one are refused by name and recorded. A
  past decision of the owner that his drift ledger shows replaced is
  "conflicting, superseded", with both dates, on the ledger as it really
  stores supersession (a link to the successor and no end date, so the end is
  derived from the successor's start, and a successor with no date, or dated
  before what it replaced, lets the replaced decision stand at no date); a
  present state that was replaced, and a decision carrying its own date, are
  "no longer held"; a decision replaced more than once is dated by the one
  that holds, each one in between named with its date. The owner's pronouns
  are rewritten by who wrote each side, and only the opening subject, its
  auxiliary and that person's possessives: the assistant's "You decided ..."
  and its French form are the owner's own claims, while "we pay you" never
  equals "you pay us" and the assistant's own "I" is never the owner. A claim
  about the present found only in undated sources, or in sources dated after
  the date it is about, is not enough evidence; every verdict line shows each
  cited source's own date, or "undated", beside the date read and the date the
  claim is checked for, and a decision supported at a past date says when it
  was replaced since. A check for a date after the day it runs is refused.
  Markdown answers become one claim per sentence with offsets into the answer,
  and what is not checked (code, headings, tables, images, struck-through
  text, markup the reader does not know) comes back with its reason and is
  counted in the per-answer summary, which has no percentage and leads with
  the most severe verdict. Every verdict is a record in canonical JSON with no
  float, its id the SHA-256 of that JSON, the same in whatever order the
  sources come, with the digests of the rules and the configuration; a replay
  says whether the record reproduces, or whether a source or the rules
  changed; the record calls itself a digest, not a signature. Every verdict
  other than "supported" sets the claim as written beside each passage
  examined, each differing run of words named: a sentence one number away is
  shown with that number beside the claim's. A canary of 101 planted errors
  and positive controls, in English and French, runs through the checker's own
  check at every construction; one item out of place and the checker refuses
  every call, naming it. The vocabulary is closed for the whole design, so
  later readers (numbers and dates with their precision, restricted
  contradictions, a calibrated entailment model) add rules, not words. The
  core imports the standard library alone at module level, reaches no model,
  store or network, and nothing in the application imports it yet;
  `opti_oignon/config/factcheck.yaml` holds its limits, all proposals. In the
  container the canary runs in 0.07 s and a check over 2,000,000
  characters in 0.30 s on plain text and 0.88 s on markdown
  notes, 0.79 s and 1.31 s when every sentence
  of them restates the claim; the machine's figures are owed.
  `docs/architecture/fact-check.md` says what it decides, what it never says
  and what it cannot see. Seventeen contracts, each red at birth only by the
  package's absence and proven by its directed mutations.
- One shell for the whole interface, and two spaces in it. Use holds the
  pages a person works in (chats, notes, projects, preferences) and the
  Workshop holds the operator pages, all under `/workshop`: system status,
  models and inference, knowledge, extensions, verify, benchmarks,
  observability, network and sync, security and backup. The shell is mounted
  once, by the layout both spaces sit under, so moving between pages no
  longer mounts it again; the page sits on a sheet beside the sidebar. The
  sidebar shows the onion mark and the name (drawn inline in the accent ink,
  no longer an image that needed a filter at night), New chat, Search
  (which opens the command palette), the destinations of the space on
  screen, the six most recent chats, then Preferences, the switch between
  the two spaces (which goes
  back to the page last open in that space) and a status card: the inference
  backend and its state, "Server unreachable" when the API itself does not
  answer and "Ollama unavailable" (by the backend's name) when the server
  answers and its backend does not, read once a minute and only while the
  page is visible; the security grade; the word Bulbe in Bulbe mode alone;
  and a pill while tool calls wait on an approval. Every destination the
  sidebar, the route announcer and the old addresses name is read from one
  table, and which entry is current is decided in one place: the page itself
  is marked, and a conversation marks Chats as its section and its own
  recent row as the page.

  The emergency stop, Stop all, is one control with a visible label, drawn
  wherever the shell stands: in the status card; in the 72 px rail a
  collapsed sidebar becomes on a desktop (the sidebar is no longer unmounted
  when it collapses); on a phone in a header outside the drawer, above every
  layer that is not a dialog (a side panel on a phone now stands over the
  page below it, with a control that closes it), and in the drawer's own
  card while the drawer is open; and in the approvals drawer, a modal
  dialog that leaves the rest of the page out of reach. Its confirmation is
  fixed to the viewport and placed from its button, so neither the rail nor
  the card clips it, and on a touch screen its actions and Resume are 44 px
  targets like the button. It always renders now: disabled with its reason
  written beside it when the server says it cannot stop, and enabled while
  its state is unknown, which a status that cannot be read makes it again,
  a failed request then saying so; before, it vanished whenever its status
  had not been read. Its state is read by one poller, one request at a time
  whoever asks, instead of one per mounted control, and only the copy the
  reader acted on announces what changed. Its confirmation keeps both
  actions (stop, or stop and switch to Bulbe), the steps that failed, and
  the stopped pill with Resume; under the pointer it lifts to the second
  surface and keeps 4.5:1 in every palette. The other dialogs (export, a
  chat's rename or delete, the first-run overlay) still cover it while they
  are open. The approvals drawer and the export dialog are mounted once, by
  the shell, and the count of the tool calls waiting on an approval shows
  in the status card, the rail and the phone header alike, so an approval
  can be answered and a conversation exported from any page (the export
  shortcut no longer depends on the chat page listening for a window
  event). The global fifteen-second health poll is gone, and so is the
  health store the old dashboard alone still read; the stop's status and
  the backend's state are read by their stores, each with a timeout set
  again after its answer.

  On a phone the drawer is a modal dialog with its own close control: while
  it is open the header and the page behind it are inert, focus moves into
  it and back to its opener when it shuts. At every width the shell keeps
  clear of the safe areas (a phone on its side draws the desktop's sidebar),
  and the drawer's links are 44 px targets. An address no page serves (a
  mistyped Workshop page, a stale link) is answered inside the shell, with
  the sidebar and Stop all, instead of by the framework's bare error page.

  Every address the interface used to serve still lands: `/settings` with
  its old `section`, `tab`, `g` and `q` parameters, `/health`, `/benchmark`,
  `/verify`, `/claims`, `/verify-answer` and `/verify-citations` redirect
  permanently, in the route's load, to the page that holds what they held,
  their query kept; an unknown one goes to Preferences, never to a missing
  page, and a settings group named by `g` is found wherever it now lives.
  The root sends its reader to the chats index for now, temporarily. The
  settings are split between Preferences (appearance, keyboard, account,
  chats and memory) and the Workshop's pages, one settings hub drawing the
  groups each page holds; its search still reads every group's title,
  description and synonyms, and now the name of the page holding it and of
  the old section it sat in, across both spaces; each result links to the
  page holding it, Enter opens the first, Escape clears, and the words stay
  in the address. The palette switcher, the account menu and the
  notification history, which the old header held, are in Preferences,
  redrawn to the surface rules; the network page shows the server's
  reachability (the inference server, its latency, the offline queue, the
  last error), read when the page is shown and on request. The chats' side
  panels are hosted by the chat frame, drawn around a conversation only.
  Each run in the benchmarks' history links to its detail (every model's
  accuracy, code, structure and speed), which the old sidebar's runs list
  alone used to open, and the history, now the latest fifty runs, filters
  by a run's id, profile or models. The Workshop is drawn
  compact under a band in the cool tint whatever the reader's density, and
  is marked as its own space; Use follows the reader's density.

  Chats, where the old dashboard stood, is an index of every conversation,
  newest first, grouped by the day it last changed: today, yesterday, the
  previous seven days, earlier; it stands alone, the chat frame's model,
  preset and context bars drawn only around a conversation. Its search is
  the server's, over titles and messages: the words travel in the address,
  up to 200 matches come back, and when a search fills that limit the page
  says so instead of passing a partial
  list off as the whole. Without words the listing comes fifty at a time,
  from an offset, and "Show more chats" reads the next page (a chat deleted
  meanwhile does not make it skip one). Each row keeps its actions in sight,
  a menu button always drawn, never revealed by the pointer alone: rename
  and delete ask in a dialog that shows why the server refused, if it does,
  and export opens the shell's export dialog; a rename or a delete reaches
  the sidebar's recent chats too. The dashboard, the old sidebar's list of
  conversations (nothing mounted it since the sidebar was rebuilt) and the
  health store only the dashboard still read are removed. The guides name
  the new places: the twenty-three paths into the old settings page, in
  eleven documents, now read Preferences or the Workshop (the plugin
  marketplace, two-factor setup, Bulbe mode, the system preset, the keyboard
  shortcuts, backups, the knowledge base); where no page did what a guide
  described (active sessions, a red-team page, a pipeline override), the
  guide says what the interface does instead. The branch-protection guide
  keeps its path, which leads through the repository host's own settings.

  svelte-check reports 34 errors, down from 77: the settings page's 42 went
  with it, and the shell's one. The ratchets fall with the rebuilt and
  removed files: style attributes by 46, border-token reads by 27, hand-made
  buttons by 23, lines between rows by 10, capitals and wide tracking by 10,
  type under 12 px by 7, raw requests outside the API layer by 6, hand-made
  fields by 4, status washes outside the primitives by 4, interval timers by
  3 (one poll is gone, the health store's; the stop's and the backend's
  reads remain, as timeouts set again after each answer), colour literals by
  3, the one hover-only reveal of the old chat list and one French comment
  line. The shell, the sidebar, the status card, the stop, the phone header,
  the settings hub, the navigation table, the root and route layouts, the
  chats index, the settings search and its catalog, the three moved header
  controls, the network reachability, the stop's read rules and the
  catch-all join the files held to the surface rules. What the contracts
  prove is what the templates emit when compiled for the server, what the
  pure modules answer under Node and what the styles declare; the shell
  kept across a space switch, the redirects firing in a browser, the
  keyboard (the drawer's focus among it), an accessibility pass in each
  palette, no horizontal scroll at 393 px (with approvals waiting and the
  machine stopped), the rail's confirmation in view, and the stop in one
  tap on a phone on every page, a side panel open, are checked on the
  machine.

- The interface's primitives, its own icons, and the rules its surfaces are
  held to. The button forwards what it is to assistive technology --
  `pressed`, `expanded`, `haspopup` and `controls` become `aria-pressed`,
  `aria-expanded`, `aria-haspopup` and `aria-controls`, each omitted when not
  given -- takes a pill or a round shape, and draws a check when pressed,
  beside its label or at the corner of an icon alone, where it keeps the
  accent ink on its own tint and a line over a pixel wide; a pressed quiet
  button keeps its tint under the pointer. Five primitives join it in
  `frontend/src/lib/ds`: an icon button (a 36 px circle on a desktop, 44 px
  as a phone's target); a toggle chip, whose pressed state is the accent
  tint and a check in place of its icon, with an optional visible note such
  as "auto"; a menu, placed by floating-ui, whose keys (the arrows,
  wrapping and skipping disabled items, Home, End, Escape, which returns
  focus to the trigger, and Tab, and on the closed trigger the down and up
  arrows alone) are decided by one pure module; a native checkbox named by
  its visible label alone, its description said after it, with a mixed
  state that is the input's own; and a side panel, a labelled complementary
  region beside the page, never a dialog, whose edge is a focusable
  separator that says its width in pixels, which the arrow keys move by
  16 px, Home and End take to its narrowest and widest, and a press
  without a drag steps through three widths, so a pointer resizes it
  without dragging. An icon alone never renders without a name: the icon
  button, a button drawing an icon alone and a menu refuse a missing or
  blank one. Every shell that mounts the app shell (chat, notes, projects,
  benchmark, settings, health and verify) hosts its right panel in the side
  panel, so that panel can now be resized from the keyboard, and its bar no
  longer lights in the warning wash. Icons are drawn once, as compact path
  data: the forty-one the interface needs, thirty from the approved
  drawings and eleven drawn in the same hand (close, the up chevron, a
  warning, an error, information, a straight line, the two feedback thumbs,
  delete, download and attach); every icon the icon primitive draws is
  stroked at 1.5 now, not 2, and the older icon package draws only a name
  the set does not hold yet. An error toast shows the octagon, no longer
  the cross its dismiss button draws.

  The primitives and the component gallery are held to six rules, checked
  on every run over a list that only grows: a toned ground (a surface, a
  tint, the accent fill) carries the edge token, which the day and night
  palettes draw transparent and high contrast draws; no border token draws
  a line (the dialog's header and footer, the tab list's rule and the
  toast's status border are gone, the dialog's parts set apart by space, a
  toast's kind said by its icon); no capitals, no title case and no wide
  tracking (the button's tracking is gone); matte; a selection is never a
  colour alone: a pressed control draws its check, a selected tab is set at
  weight 600 (an underline in the mark token, or the fill, which forced
  colours draw in the system's selection colours), and so is a selected
  option; and focus stays visible: the tab panel keeps its focus ring, and
  the option under the keyboard in a select, whose focus stays on the
  field, draws its own ring in the focus ink. The gallery
  (`/dev/components`, development only) shows every primitive in its
  states and every icon by name, in the palette and density chosen at its
  top. The ratchets fall with it: border-token reads by 12, capitals and
  wide tracking by 3, lines between rows by 4, style attributes by 2 (the
  panel's width is now run-time geometry its primitive writes, and the
  handle's hand-made hover colour is gone), status washes outside the
  primitives by 2 and hand-made buttons by 2. Sixteen contracts; the ones
  that render do so compiled for the server, since the app renders only in
  the browser: the menu's keyboard, the panel's resize by keys, drag and
  press, the icons as drawn, forced colours and an accessibility pass over
  the gallery in each palette are checked on the machine. The rest of the
  interface adopts the primitives as each surface is rebuilt; until then
  its own buttons, fields and lines stand, counted.

- An assistant reply is rendered from its markdown: headings, lists and
  task lists, quotes, tables, links, emphasis, code blocks, and line breaks
  (a single newline breaks the line, as a chat reply means it). marked's
  lexer turns the reply into tokens and nothing else of marked runs: each
  token becomes a node of a closed set, shown through Svelte's text
  interpolation, so no reply reaches the page as markup. The frontend holds
  none of the raw HTML sinks (`{@html}`, with or without a space after the
  brace, `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `Function(` and
  `eval(`, bare or on an object), and the lint rule against `{@html}` stays
  on; the one other call that writes markup, `document.write` in the
  recovery codes' print window, is pinned where it is. Raw HTML in a reply
  shows as the text it is, a block of it with its lines and indentation;
  character references in text decode once, through a fixed table and every
  numeric reference, while code keeps them as written; a link is a link only
  when it is an absolute http, https or mailto address, and it opens in a
  new tab with no referrer -- anything else shows as its source; an image
  is never fetched and shows as "Image: alt (url)". A code block is named by
  its language and carries a Copy that writes that block's code and says
  "Copied" once the clipboard took it ("Copy failed" otherwise), absent only
  while the block is still arriving; keywords and defined names are coloured
  for Python, JavaScript and TypeScript, Rust, shell and SQL. A table scrolls
  sideways in a region the keyboard can focus. Headings start one level
  below a screen-reader heading each reply now carries. While a reply
  streams it is lexed at most once per animation frame, and its caret, a
  2 px bar that reduced motion stills, follows its last character inside
  its last block. A long finished reply shows its first whole blocks and a
  "Show the rest" toggle, and the blocks it hides are not rendered, so a
  code block or a table is never cut; a reply that is one long block is
  shown whole. The row of code-copy buttons under a reply is gone. Any
  message that is not the assistant's -- a user's, or one of another role
  -- stays plain text, and a long one shows its first lines and a toggle,
  as before. The renderer cannot fail or hold the page: marked's lexer is
  super-linear on some inputs (six kilobytes of unclosed link openers took
  over six seconds to lex), so the work is estimated from marked's
  block pass before the inline pass runs, and a reply over the limit, one
  marked cannot lex, or one nested deeper than the tree's cap of 32 levels
  (past it, a level is its source as text) is shown as plain text with every
  character; a lex that still turns out slow is not repeated for the rest of
  that stream or when the reply is shown again. Replies are set in a serif
  stack; four provisional tokens (`--oo-bg-code`, `--oo-code-keyword`,
  `--oo-code-name`, `--oo-font-serif`) carry the reply's colours and type
  until the palettes are rebuilt, and no size in the renderer computes under
  12 px. Thirty hostile replies are rendered on every run, finished and
  streaming, and must emit nothing that loads or runs. The contracts render
  the components compiled for the server, since the app itself renders only
  in the browser: how a long reply feels while it streams, the clipboard and
  screen readers are checked on the machine, not here. Twenty-five
  contracts. The chat page's script grows by about 87 KB (24 KB gzipped),
  marked and the renderer, in the chat route's own chunk.

- A measuring harness for the frontend; nothing in the interface changes.
  Sixteen per-file ratchets count what the interface has to pay down, each
  in every spelling that draws the same thing: hand-made buttons (a
  `<button>` or `role="button"`) and fields outside the primitives, `style=`
  attributes and `style:` directives, type under 12 px (px, rem or em, a
  class, a declaration, a directive or a `font` shorthand, and a text-scale
  token declared under 12 px), `rgb(` literals, hover-only reveals (a
  `group-hover:` utility no focus variant sets too), raw `fetch(` (bare or
  through the global object) outside the API layer and inside it outside
  its client, French comment lines (with a plain reading as their floor),
  `setInterval(` sites, symbol glyphs (raw, as a numeric or named reference,
  a script or CSS escape, or built from a number), border tokens read
  directly or through an alias, capitals and wide tracking, lines between
  rows (the classes, and the CSS that draws the same line), gradients (CSS
  and the Tailwind utilities) and page reloads. They read the files `git
  ls-files --cached --others --exclude-standard frontend/src` lists, so a
  file being written is counted and an ignored one is not, and a further
  contract refuses a file of a kind no census reads. Each ledger is a
  literal dict in its test file, born equal to today's count, and one
  engine in `tests/_frontend.py` holds them all: no file above its entry, a
  file not in the ledger counts 0, an entry above its count fails with
  "lower the ledger", no entry higher than the committed one at `HEAD`, a
  listing that is never empty, and a positive sample every probe must
  count, so a probe gone blind is red instead of a quiet zero. A moved file
  carries its entry when git's rename detection pairs it with the file it
  was, whether or not the move was staged (the untracked files are shown to
  git in a scratch copy of the index, never in the index itself); a ledger
  renamed, or moved to another test file, is compared with the one `HEAD`
  holds. Eight contracts prove those rules on synthetic repositories.
  svelte-check's errors are now counted per file -- 78 in 16 files today,
  42 of them in the settings page -- so a new error in one file can no
  longer hide behind a repair in another. Each run plants one error in a
  component and one in a module, under `src/lib` and under `src/routes`, on
  its copy and must count all four, and it refuses a run that printed no
  `COMPLETED` summary or whose total differs from the error lines read: the
  two ways a global count reads zero. The CI svelte-check step is not
  changed. `bash scripts/ladder.sh t1` now also lints the frontend (0
  errors, over at least one component and one module) and builds it (`vite
  build` exits 0 and writes its index page), both on a copy under `$TMPDIR`
  whose dependencies are linked one package at a time, with the reports in
  the run's own scratch directory, so nothing is written in the tree;
  without `frontend/node_modules` the step is a named OWED skip, and `bash
  scripts/ladder.sh frontend` runs it alone. The ladder's time budgets now
  read the frontend suites as well as the componion's. Twenty-six
  contracts.

- [SECURITY] The componion is served over the API, read only.
  `GET /api/allium/status` answers the terminal's own projection -- the
  status, its label codes, the being's codes at the served minute and the
  lines that say them, the doctrine last -- through one router under
  `/api/allium` and schemas of its own (`api/schemas_allium.py`, every model
  named `Allium*`, so no published component is renamed). The router is
  always mounted. While the garden is switched off the route answers
  `disabled` with no line (the terminal's form names the settings file's
  path, which is never served), and neither the route nor the new `allium`
  key of `/api/health` imports anything of the being, whichever allowed name
  the request was addressed by: both read the switch, and the route the
  names of `api.hosts`, from `config/allium.yaml` themselves, one `stat` per
  read, by the garden's own rules; only a refused request imports the
  garden's catalogue and its nets, to say its line. Every garden
  route first checks the raw Host header (exactly one, naming `127.0.0.1`,
  `localhost`, `[::1]` or a name listed in the new `api.hosts`, where an
  address spelled otherwise than as its dotted quad refuses the list; ports
  are not compared and `X-Forwarded-Host` is never read), the Origin when one is
  sent (the same names, over https for a listed one; `null` refused) and
  `Sec-Fetch-Site` (anything but `same-origin` or `none` is answered only
  beside an accepted Origin). A rebound name, a page of another site and an
  image pointed at the loopback API are refused before the user dependency
  or the garden runs. The platform's user dependency comes next, and the
  garden's caller is built from its principal alone: a validated token, from
  the cookie or as Bearer, is a session, and the single-user principal is
  the local web surface. Every refusal is a closed body,
  `{"detail": <one of the garden's lines>, "refusal": <code>}`: nothing of
  the request or of an exception is repeated, and a 401 keeps its status.
  The API's garden is built at the first status request with the switch on,
  one per process, and its store is closed at shutdown; it serves no gesture,
  refuses a caller that is not given, reads this server's emergency stop, and
  takes its single-user rule from the auth manager the server already runs,
  whose read opens the auth store as the platform's own reads do (so a look
  can checkpoint that store's pending frames, as they can). A view in a
  request is capped
  (`api.python_cap`, 200,000 units of engine work with the Python reference;
  `api.native_cap`, 5,000,000 with the native core): every request to the
  engine asks at most the cap, the fold from the genesis included (`life`
  now carries the cap to it and to the frozen path, where it was uncapped),
  and a view that does not finish is shown as of its last kept state,
  labelled, followed by the new line "The next write made in oo garden
  computes it."; a cap below one awake day of the laws the engine carries is
  raised to that day, logged, so the line holds. A look never catches up and
  writes nothing in the being's store. The one write it can cause is the
  platform's tamper evidence: after the mode files change to disagree, the
  mode reading records the mismatch in the auth store's audit log and in
  the signed audit chain, once per change of the files. Three shared
  lines are reworded to hold on every surface: the two emergency-stop lines
  ("in this server", presupposing no onion) and the missing-account line.
  For a process that lives long and runs many threads, the engine handshake
  is answered once for every thread, the switch and `api` readers keep their
  caches in one assignment, and the mode reading keeps a manager of its own,
  so looking at the garden no longer makes the whole server re-read its
  security mode. Around the being: remote inference refuses the fields
  `allium`, `componion`, `garden` and `pet` as a capability it never reaches;
  while a plugin loads in process (the fallback when its subprocess cannot
  run), the sandbox refuses the being and its two entry modules (the router
  and the terminal's commands) to code with a plugin frame anywhere on its
  stack, loaded or not -- by a statement, `__import__`, a relative import
  whose package the plugin forged, or importlib's own entries -- and a
  plugin that renames itself is still plugin code. It is a rule of the load,
  not a boundary against a hostile plugin: hooks run after the load with no
  import restriction, and a loader of the plugin's own can execute a file by
  its path (both left to the plugins audit). Vite's dev proxy keeps the
  browser's Host (`changeOrigin: false`), so the check holds through it. The
  inference backend gains a passive light sink -- a record of a finished
  request's raw token counts, each with its source -- with no sink
  registered and no head calling it yet. `docs/api-reference.md` counts 531
  endpoints and the published prose digest records the new route and
  schemas. Nineteen contracts. Owed to the machine: the first prototype
  sowing on the real key file, read through the running API and through the
  dev proxy; a capped view's time at the default caps, reference and native;
  the first status request of an API process on the real record (the chain,
  the audit log's chain, the cipher integrity check, the engine handshake;
  when that request is the process's first user of the signed audit log, it
  creates its table and may rewrite the chain's anchor);
  the auth manager's store reads per request, and whether one checkpoints
  the auth store; the store held open while
  `oo garden` writes; the browsers and the phone (the Fetch Metadata they
  send to the loopback names, an image from another site refused, the phone
  reading with `api.hosts` set, a page of another origin refused); an API
  kept running across an upgrade; the frontend's degraded ratio with one more
  module key; and the light counts from a real engine, once a sink exists.

- The componion's first surface: `oo garden`, in the terminal's own process.
  It reaches the componion through one service,
  `opti_oignon/allium/service.py`, the one the API is to serve as well, so
  the store and its life are no longer uncalled. The garden is off until
  `config/allium.yaml` says `enabled: true` (anything else, and a file that
  cannot be read, is off); switched off, every look and every write says so
  and names the file, and nothing builds a store or reads a key, the
  security mode or the clock. Five subtopics are listed: `show` (the text,
  or the text and a drawing of 32 columns by 9 rows, or one line of JSON),
  `sow`, `care greet|water|warm|play`, `lab` with `lab laws`, and `keep`
  (`verify`, `name`, `laws diff|apply|pin|unpin`, `resume`, `finish`);
  `lang`, `tray`, `share` and `keep celebrate|bury` are hidden and say that
  they are not in this version. Each command is one action: the security
  mode, the single-user answer and the clock are read once for it
  (`Store.action`), the being is opened only under the account the service
  derives for the terminal, and every outcome is a served status with its
  own form, stream and exit. A look writes nothing in the being's store,
  nor in the auth store it reads the single-user answer from: that store is
  opened read-only while a WAL file or a journal lies beside it (the keyed
  connector gains a read-only open), so pending frames are never
  checkpointed and a hot journal is never rolled back.
  Each write the terminal makes is followed by a settle in the same action,
  so `keep verify` has kept states to check: it verifies the chain from the
  genesis, replays every kept state on the reference engine and compares the
  state served now, and names a divergence by its day and law version
  without replacing anything. An engine that stops on a fault freezes that
  one look on the last kept state, labelled, with exit 1. Text never comes
  from the command line: every parameter is a closed, quiet type, and
  extras, unknown options and unknown subtopics are refused without being
  repeated; a name or an answer is read from stdin, one line, after its
  question is printed. Sowing, naming, the law writes, resume and finish run
  only from an interactive terminal in the foreground. A sowing names only
  the newest law of the sowing channel the engine carries and has not
  retired, never the test law, and writes no rhythm consent; its card says
  why, and says before the question that the seed is a prototype and that
  there is no way yet to compost it or put it to rest. Every form comes from
  a closed catalogue of 139 templates and passes the ethics nets before it
  is printed: printable ASCII, a closed class of inner-state words of which
  only the design's verbs of a simulation pass, phrases grouped by the
  psychological law they would break, and no served field named after an
  absence; one sentence, the doctrine every `show` closes with, is exempt by
  its exact bytes. The lists are code, and a refusal is said through a
  closed table, never with an exception's text: its first line says what
  was refused, and one that comes after a write says the write was done.
  No platform log record reaches the terminal while a garden command runs
  (`oo` configures no logging, so a warning would otherwise print outside
  the catalogue). Under Bulbe's rules, or with
  a mode that cannot be read, the life goes on and a label says so, and a
  glass jar stays sealed. The single-user answer is read from the auth
  settings and, when the auth store exists, from one count of its accounts,
  without importing the auth module. `scripts/allium_garden_gallery.py`
  prints sample forms from fixed values, without a store. Two fixes on the
  way, neither released: the store's default data directory was read from
  the package's settings object instead of the configuration module, so the
  production store could only answer `unavailable`; and the test process's
  data firewall now redirects a SQLite connect by `file:` URI as it does a
  connect by path. Nineteen contracts, and one for the firewall; the cold
  footprints of the CLI's import, of a garden switched off (predicted, then
  measured equal), of a seamed `show` and of the production look behind the
  data firewall's empty mirror are frozen. Owed to the machine: the gallery and the forms in each
  terminal used (IDE, tmux, ssh; light and dark backgrounds; the drawing's
  backslashes, backquotes and quotes, the blank sky row, the 78-column wrap,
  the unsplit path line, and the `Error:` colour under `--no-color`,
  `NO_COLOR` and `color: false`); a readable key and a security mode read as
  Daily from each of those terminals; the terminal test true in a foreground
  interactive shell and false in a pipe, in the background and under a
  sandbox; a cold `oo garden` switched off, then `show` with the production
  seams once a first prototype is sown, and that the single-user read leaves
  a SQLCipher auth store's bytes and file list unchanged (the container
  proves it on a plaintext store: closed cleanly, with frames never
  checkpointed, and with a hot journal); deep verification
  of a real life over a month and a year, with the native engine present;
  the chain verification's cost per command on a journal of 20,000 events;
  and the settle after a gesture on a year-old being with no kept state,
  native against reference.

- The componion's life on the platform: the store now lives a being in
  time. Each journaled write reads the wall clock and the machine's UTC
  offset once; a late write lands on the latest recorded minute, a clock
  set back by more than `life.skew_note_min` is noted once with a `clock`
  fact, and an offset that differs from the one in force is recorded as a
  `tz` fact in quarter hours, never a zone's name (unless an offset another
  device recorded at that minute ends it: the reading then waits for the
  next minute); a wall past the last minute a fact may carry is refused
  like one before birth. Both go before the
  gesture, in its own transaction, and every fact carries the law version
  the engine's law timeline has in force at its minute. A view writes
  nothing -- no fact, checkpoint, meta or anchor: it reads the facts in
  canonical order, the genesis first, then asks the engine outside the
  store's lock, packed under the engine's input limit without ever
  splitting a minute; a view whose budget of work does not get there
  serves a stored state labelled `catching_up`, with an estimate of the
  work still owed, never a partial one. `settle` keeps states at the local
  midnights the new `life:` section retains (the last 14 local days, the
  Mondays of the last 52 weeks, every first of a month, the latest) and
  drops the rest and every row of another engine; a gesture drops the
  checkpoints from its minute on, and a checkpoint a later-landed fact made
  stale is skipped by a view, never repaired, and replaced by the next
  settle, so dropping every checkpoint changes no view; one whose bytes do
  not hold the state it names is refused `divergence`. The engine never
  runs under the store's lock for a view, `law_state`, `laws_diff` or a
  settle. Advancing the clock buys nothing: a clock jumped forward and an
  honest absence give the same facts and the same state, in the garden and
  on the windowsill. The new `laws:` section of `config/allium.yaml` is the
  proposal a birth freezes into the genesis (four params, the hemisphere,
  the day-length band, the weather); a malformed value is refused by name
  where it is used, the engine lives the genesis's first minute before it
  is written (a genesis it refuses is never written), and a wall clock
  before the second day of 1970 is refused. After the birth the proposal
  moves nothing by itself: `laws_diff` shows what it would change and a
  confirmation, `laws_apply` writes the law update the confirmation names
  (refused `pinned`, `params`, `nothing` or `confirm` by name, and
  extending a pending update rather than cancelling it), `laws_pin` and
  `laws_unpin` hold and release the law and params in force, and
  `law_state` says what is in force; a gesture on the being's home device
  writes the update to a carried stable successor of its law, with the
  params in force (or those of an update the owner confirmed that is still
  pending, which it extends), at the next local midnight of the offset its
  own write records, within the day's budget of law updates, and a view
  never does. A generic append of the law
  kinds is refused `producer`. Law identity is a name and a digest: a
  being whose law is not carried under the digest, version and provisional
  flag its genesis names -- a law file edited in place included -- reads
  `unavailable`, a prototype whose law the register of retired laws names
  reads `retired_prototype`, and a fact may carry only the genesis's law
  version or one an `evolve` in the trunk goes to. Every form a prototype
  being is served in carries the `prototype` label. Fifteen contracts; the
  stable laws that exercise law updates are injected in the reference only,
  since both carried laws are prototypes. The latencies on a real keyed
  store (the sowing dry run, the law timeline per write, a view after a
  month away, a settle over a year, retention on a store months old) and
  the machine's real UTC offset across a daylight-saving change and during
  travel are owed to the machine, and nothing in the application calls the
  store yet.

- The componion's life in time, in both engines. A being now lives: three
  new engine operations, answered byte for byte by the reference and the
  Rust twin. `advance` folds a being from its genesis and its facts to a
  minute, or until a budget of work runs out, re-checking every fact (kind,
  body, redaction, daily budget, law field) and returning the state, its
  digest, the environment at that minute, notes and, on request, a trace of
  the work; `grid` gives the local time, the civil date and the next
  local midnight of explicit minutes under explicit offsets; `timeline`
  folds the law kinds alone and says which law and params are in force,
  what is pending and where the daily firings fall -- the one law timeline
  in the tree. Time is a fifteen-minute layer aligned to UTC and a daily
  layer that fires once per rise of the local day index, so a move of the
  time zone never lives a day twice and a skipped day is not lived; the
  offset comes from `tz` facts in quarter hours, and the calendar is
  integer arithmetic equal to `datetime` from 1970 to 2199. Under the
  provisional law the being is a breathing seed: a three-gene clock
  entrained by light, a reserve of sugar and fructan, a soil that takes
  rain and waters, and a stage that sleeps through drought and from the
  first day of winter and wakes after its rest, joined by a four-channel
  bus read one step late, so the order of the organs never matters; rain
  is drawn from the being and the local day alone, so a life cut or sliced
  anywhere is the same life. While the being sleeps, the organs the law
  declares quiescent are never called, the soil and the stage step once a
  day and the rest catch up once at the wake: this fast path reaches the
  same bytes as stepping every minute, and a dormant windowsill year on the
  fixture law costs 1460 units instead of 36135. Each law now carries its
  organ code revision, its world (year, seasons, a provisional sine
  daylength per band, rain), its organs' constants, the ranges of four
  params a genesis freezes, and a unit table with per-day caps and day
  ceilings recomputed by contract (4209 and 4591 units an awake day, 7 a
  dormant one, which counts the visit a sleeping being pays for a pending
  `evolve` a later time zone move took off its midnight); the
  journal's reserved kinds get their bodies (`evolve`, the pins, `tz`,
  `clock`), and `evolve` and the pins a producer of their own. An
  `evolve` takes effect at the next local midnight and a pin holds the
  law and params in force; a conflict is a note, never a refusal that
  would replay forever, while a law the engine does not carry, a change of
  law no migration joins and a law field never in force are refused by
  name. The fixture genome gains the reserve's three enzymes, and a
  register of retired prototype laws starts empty. The native core's
  release build now checks integer overflow, and a panic inside the engine
  is caught and answered as `engine_panic`. Six golden thirty-day lives
  are committed in hexadecimal and replayed in both engines;
  `scripts/allium_author_life.py` writes them and refuses to re-record a
  life whose answer changed under an unchanged law digest, so a change to
  what an organ does must bump the law's code revision. A new ladder tier,
  `bash scripts/ladder.sh life` (not part of `all`), lives ten-year
  lives on the native core with the reference replaying sampled windows,
  and exits 3 as owed where the core is not built. Twenty-five contracts,
  two of them in that tier and two that succeed the genome's equivalence
  contracts, whose enzyme column and corpus floor the three new enzyme loci
  moved. `scripts/allium_bench.py` measures the cost
  of a unit of work in each engine and is for the machine only: no figure
  of it is claimed here. The store's side of it -- the recorder, views,
  checkpoints and laws on the platform -- is the entry above.

- The componion's journal and store. A being's life is a chain of facts, one
  file per person, each fact identified by the digest of its envelope in
  both engines (a new engine operation computes the ids of envelopes whose
  body is only a digest, so a forgotten fact still verifies). One membrane
  admits every write, deriving the surface and actor from the transport
  and holding each kind to a closed table of schemas, surfaces and daily
  budgets that both laws pin by digest. The store refuses a birth without
  a readable key and SQLCipher (a labelled glass jar only when the owner
  allows it, no key is configured and the mode is Daily, never opened in
  Bulbe), checks the file header before trusting the cipher, anchors each
  birth in the signed audit log and its head under a keyed anchor, and
  refuses a flipped byte, a reordered event, a truncation, an older copy
  or a missing file by name. Forgetting a taught word destroys its random
  key and sealed bytes and compacts the file; resuming a broken journal is
  an explicit, recorded act that never brings a forgotten thing back.

- The componion's body is read from its own genes. The eight continuous
  shape loci of the full law (bulb width and height, hat height, leaf
  stiffness, eye spacing, lean and gaze, skin lines, speckles) get alleles
  on a lattice of multiples of 8192, so a genotype compiles to an exact
  step and the greenhouse reads each trait through fixed cuts halfway
  between steps: every level is reachable, the middle levels are the
  common ones, and no founder's look depends on another founder. Wide eyes
  are dominant and strong skin lines recessive; a missing locus reads as
  its default and never as a mark. The founder window is unchanged, and no
  genotype of either law can cross a cut under it. Only the full law's
  pool moves; the fixture law was already on the lattice and stays byte
  for byte.

- The componion's beard and moustache genes, in the full law. Thirteen shape
  loci, each placed after every other locus of its pair so that no founder
  draw and no promoter moves: four additive melanin loci and a red locus
  give the colours (platinum to dark brown, red glints on carriers,
  venetian to auburn), a dominant cream allele keeps the cream beard every
  young gnome is born with (about half of all founders, the commonest
  beard by far), three loci time and cap the greying, and four decide
  whether a moustache grows (about three in ten) and its form, size and
  thickness. The shape kind's trait box widens to 0..31 in both laws; code
  15 stays free. The greenhouse reports the adult beard and moustache each
  founder allows, reading a missing locus as the cream beard with no red
  and no moustache rather than inventing a colour. Nothing draws a beard
  yet: the renderer and the greying clock are later work, and the
  frequencies are proposals, not a sowing.

- The componion's phonology, in both engines. A being's language starts
  from sixty-three bytes of its genome: which of twenty-seven sounds it can
  make (with floors every language keeps: the corner vowels, a stop, a
  nasal, six consonants), how it builds syllables, how long its words run.
  Eight operations: the table as the engine read it, the language block of
  a genome (or of a seed alone), the phonology a block decodes to, whether
  forms are sayable and where their syllables split, coinages for concepts
  with a sound-symbolic bias (small and sharp things lean to i, e, s and
  voiceless stops; large and round ones to o, u, m and l), the being's first
  sound, a list of 2048 sayable six-letter forms per block with six of them
  carrying the first 66 bits of any digest, and a taboo check. The taboo
  list holds SHA-256 digests only, matched on whole forms and on every
  substring of three to eight letters; a rejected candidate is reported by
  its digest, never spelled, and the work charged does not depend on what
  the list holds. The full law's list is empty until its owner supplies
  one through `scripts/allium_taboo_digest.py`, which reads the words from
  outside the repository, keeps only their digests and refuses a list that
  would leave a block without its short word list. The fixture law carries
  three witness digests the contracts derive from the generator itself, so
  no word is written in the tree. Both laws pin their phonology and taboo
  tables by digest.

- The componion's genome, in both engines. A genome is diploid: pairs of
  chromosomes of fixed 16-byte records whose order is itself data, since a
  gene's promoter is the run of regulatory records just before it. Sixteen
  kinds of gene are laid out, seven of them reserved for the brain, the
  language, the shape, the clock and the cold that later work will read;
  their layout and bounds are fixed now so that a genome written today
  stays valid. Two laws carry it: a small fixture law (one pair, 29 loci)
  for the contracts, and a first provisional full law (eight pairs, 175
  loci, 5640 bytes a genome). A genome is never stored: it is founded from
  a 32-byte seed and the law's founder pool, one keyed stream per locus and
  homolog so that a change at one locus moves no other, and compiled on
  demand into flat tables -- dominance by each locus's mode, promoters as
  edges, a decay per species, units converted. The decoder takes a genome
  as hostile input: it accepts inside the law's box or refuses by name with
  the existing codes, and a field that could carry the token light into the
  being's chemistry is refused. Each law records the bound analysis of the
  factors a genome can supply, recomputed by contract. An authoring script
  writes both laws and both pools from one table, and a greenhouse script
  reports on a cohort of founders without simulating a day. The Rust twin
  answers the four new operations byte for byte as the reference does, over
  some 1700 requests per run; its genome code denies unwrap, expect, panic,
  indexing and ``abs``. The reference codec now reads and writes a plain
  string in one step rather than character by character, with the same
  refusals at the same positions: the equivalence contracts had run past
  their time budget on it.

- The chassis of the componion's engine (the componion: the companion
  onion), `opti_oignon/allium` and
  `rust/allium`: a being that will be a pure function of its facts needs
  two engines that cannot disagree, so every piece is written twice, a
  Python reference and a Rust twin, and proven byte for byte equal. A
  canonical JSON wire (printable ASCII, sorted unique keys, integers within
  2^53 - 1, no float, no whitespace, a closed set of eleven refusal codes);
  Q16.16 fixed point with primitives that never raise and never panic --
  a correction saturates and counts an alarm; randomness addressed by
  content (SHA-256 keys, SplitMix64 streams, rejection without modulo
  bias); law and table files digested from their canonical re-emission, so
  indentation never counts and a native core built from other files fails
  its handshake and is not used. The twin is `no_std`, forbids unsafe code,
  denies arithmetic with side effects, links `sha2` alone and releases the
  GIL for the length of a call. Twenty-one contracts, each within two
  seconds; the ladder now reads their measured durations against the
  budget each suite declares, and runs the engine's own `cargo test` and
  `cargo clippy` (owed where clippy is not installed, as here). Nothing in
  the application calls the engine yet.
- [SECURITY] The test process no longer reaches the maintainer's data.
  Measured first: 253 tests touched `data/`, `opti_oignon/data/` or a
  database file of the tree through real modules -- the existence check
  of the master key, loaded on any machine that has one; appends to the
  signed audit chain; the application's own databases -- and a sandbox
  that hid only `data/` answered otherwise than the maintainer's machine.
  `tests/_data_firewall.py`, installed by the conftest before any suite is
  collected, redirects every path in those places to a session mirror that
  starts with the tracked files as HEAD holds them, taken from git: the
  real places are never touched, and the session ends with the count of
  paths kept off them (505 on the first run), a figure to bring down suite
  by suite. Its reach to the child processes a contract starts came later,
  under Fixed.
- A suite-wide check, `tests/conftest.py`, fails at its teardown any
  contract that leaves a stand-in project module, a neutralised project
  entry or a replaced `urllib.request.urlopen` it did not find. A stand-in
  left behind is what made the facade contract pass or fail with the order
  of the suites before it, and five suites left the transport replaced. The
  check reads the state before the contract's fixtures are set up and
  checks it after they are torn down, so a suite that closes its own window
  is not charged; the whole tree measured zero of each before it was
  switched on.
- [SECURITY] The sync gate shows what it lets in. The live approval of a
  received skill now leads with the name it lands under and the SHA-256 of
  its text -- the prompt cuts every value to 60 characters, so they come
  first -- and the pending list carries both, still never a body. The text
  itself is read on demand, one record at a time, through
  `GET /api/sync/deferred/review`: exactly as it would land, from an
  envelope that still decodes against its content hash, and refused by name
  for anything else; the sync panel's "Show text" loads it before you
  approve. The digest shown is the one `/adopt` asks for once the skill has
  landed. The skills API says, for every published skill, whether it was
  written here, received and adopted, or received and never adopted, and
  the skills panel marks the last.
- `scripts/drift_ab.py`, the host command for drift with the onion against
  drift without it: one scripted conversation whose facts change on known
  turns, two arms that differ only by the onion's block (same model, same
  history window, the librarian in process with persistence off), and per
  arm the contradictions per 1000 turns against the facts holding at each
  turn, the agreements, and what the deterministic templates could not
  decide. It prints no number without a backend. Before anything is asked,
  a model the backend lists as not installed -- the answering one
  (`--model`) or the librarian's (`--librarian-model`) -- is refused by name
  with the served models listed; the pair is then tried in the run's order,
  one minimal request each, so a pair the governor cannot load together is
  refused before the first turn; and a request that fails once the run has
  started ends it without a number. The measurement itself is the host's
  to take.
- `oo ask --json-out` shows its "Generating" spinner on stderr while it
  waits, as every other waiting command does; stdout is still exactly the
  JSON document. The spinner was there and could never turn on.
- The eviction gate's recall probes -- drawn from a span, scored against a
  candidate summary -- have a native implementation in `oo_core`, used
  when the artefact is built. It answers exactly as the Python reference
  or not at all: only for text whose every code point it classes as Python
  does (ASCII, Latin-1, Latin Extended-A but U+0130, General Punctuation,
  the euro sign: English and French), and only for the module's regular
  expressions as they are written, which travel with each call. Anything
  else is scanned by the reference. The gate's rates, thresholds and
  messages stay in Python.
- `oo chat` animates the waits you really sit through, in one line of ASCII
  on stderr: an onion that breathes from Enter to the first answer, sprouts
  once the executor's keepalive says the request is with the model, sheds
  its peels during `/close` and regrows during `/open`, and waves goodbye
  on `/quit`. Nothing shows before `animation_delay_ms`; the line is erased
  before any output, so stdout is byte-identical with or without it; no
  escape sequence is written, so Ctrl-C or a kill leaves the terminal
  clean, and Ctrl-C still exits 130. It is off when stderr is not a
  terminal, with `NO_COLOR`, `--no-color`, `TERM=dumb`, a terminal of 32
  columns or fewer, or `animations: false`. Four keys in `cli.yaml`
  (`animations`, `animation_interval_ms`, `animation_delay_ms`,
  `animation_stop_ms`), settable with `oo config set`; a bad value falls
  back alone instead of resetting the whole file.
- Two observation heads on the backend contract, defaulted rather than
  abstract: `loaded_models()` answers the models a backend reports resident,
  and `None` when it cannot say -- never an empty list, which would read as
  "nothing loaded" where the truth is "nobody looked"; `embed(model, text)`
  answers one vector, `None` when the backend has no embedding endpoint, and
  asks the governor first. Ollama implements both through its client;
  llama.cpp answers its in-process set; llama-server and the remote core
  answer unknown by name. The warmup and the governor's S1 read now say
  unknown instead of empty when nothing observed the loaded set.
- A request timeout travels as the engine option `timeout` and binds the
  transport -- a per-timeout client on Ollama, the request's own timeout on
  llama-server -- and never reaches the engine; a timeout that is not a
  number is refused before anything leaves. The counts an engine reports
  beside its answer (`eval_count`, `prompt_eval_count`, the durations)
  arrive on `extra` of the response and of the chunk that carries them, and
  are absent rather than invented when nothing was reported. Both existed
  only in the private clients the benchmark, judge and reasoning modules
  kept for themselves.
- `registry_clients.py`: the two clients the verification, note-action and
  agent routes now stand on, a one-shot text completion and a streamer in
  the agent loop's chunk shape, both sending their request through the
  inference registry. The host argument of the former signature is accepted
  and unused: where a model is served is the registry's to know.
- The core daemon (`oo core serve`): a resident loopback HTTP server over the
  inference registry, standard library only, with health, models, generate,
  stream as JSON lines, and a pack's admission ticket; off by default in
  `core.yaml`. When enabled, `core_client.RemoteCoreBackend` is registered
  in the calling process so the daemon serves its inference through the
  registry -- admission, provenance and schema apply across the process
  boundary. Refuses by name: non-loopback host, missing token, unknown
  route, malformed body, model without a backend.
- `core_boundary_guard.py` names the resident core module by module and holds
  the line the census drew: a core module that imports a module outside the
  core at module scope is refused unless the ledger already carries that
  leak; the ledger may only shrink; no core module reaches the inference
  client directly, the registry excepted. Documented in
  `docs/architecture/native-core.md`.
- `scripts/core_census.py`: a static census of the package -- per module,
  the project imports at module scope and inside functions, both transitive
  closures, third-party imports outside the standard library, database
  files named, direct client sites, and the map of every router the API
  includes -- labelled `source: static`, refusing an empty tree. The
  instrument the core/packs cut is measured with.
- The onion memory's native core, `rust/oo_core`: the four integrity hashes
  and the window assembly in Rust behind the Python surface, loaded at the
  call and never at import, with Python as the reference and the fallback.
  Built by `scripts/build_oo_core.sh` from a pinned crate; the artefact is
  never tracked. Contracts hold every hash byte-equal and every prompt
  field-equal to the reference.
- Inference-time compute (`opti_oignon/inference_compute.py`): N candidates
  for one prompt sampled through the inference registry under a budget of
  candidates and tokens (`inference_compute.yaml`), each verified -- tests run
  through a sandbox runner for code, strict-majority agreement for the rest
  -- and the first verified one chosen, stopping there. When nothing is
  verified the best-scored candidate is returned marked unverified; with no
  score the selection is empty and carries the verdicts. Not wired into any
  path yet.
- Onion memory, behind `enabled: false` in `opti_oignon/config/onion.yaml`.
  A bounded window -- Core, receipts, Peels, Flesh -- over an archive that is
  the only source of every compression: a span leaves the window only once
  its summary has answered the recall probes drawn from it, and only with a
  receipt the model can see. Off, the chat path is unchanged; on, the
  librarian curates off the interactive path through the inference registry
  with the model released after every burst. The figures the contracts
  produce are fixture readings and say so; `scripts/onion_runbook.py` takes
  the measured ones on the host. Documented in `docs/architecture/onion-memory.md`.
- `import_footprint_guard.py` observes the import in a subprocess, through an
  audit hook, and refuses any database, any file written into the caller's
  directory, or any heavy dependency that its ledger does not already carry.
  The ledger records the debt that predates the guard and **may only shrink**.
  The module count carries a ceiling because it is stable to the digit; wall
  time and resident memory are recorded and never enforced, because both move
  with the machine and a ceiling that moves with the machine forbids nothing.
- `registry_funnel_guard.py` keeps every inference request inside the backend
  registry, where admission, provenance, constrained decoding and the placement
  recipe live. It counts direct calls to the client library on the syntax tree,
  so prose never counts, in every spelling the library can be reached by. The
  modules that still call directly are a sealed ledger that **may only
  shrink**: an owed module that changes while still calling directly must
  migrate, and a module nobody owes for is refused outright.
- A static concordance check pairs the mobile bridge's declarations with its
  native entry points by name, arity and type. A type outside the agreed table
  is refused rather than mapped by default, and zero declarations on either
  side is refused rather than agreed.
- A driver for the interlanguage round trip, which compiles both sides and
  compares what comes back with what was sent. It reports three outcomes, not
  two: the third means the measurement did not run, and nothing downstream may
  read it as either a pass or a failure.
- Unit tests for the wire envelopes, in the source set that needs no device.
- The user's surface on the onion Core: `GET /api/memory/onion/{conv}`
  lists a conversation's Core entries and open receipts, `POST .../pin`
  pins a statement as the user, `POST .../supersede` pins a successor and
  links the old entry to it, `POST .../recall/{key}` hands the verbatim
  span behind a receipt back and marks it resolved. The librarian carries
  the five entry points and forwards the actor to the store, so only a
  caller that says it is the user gets through; a pin that would push the
  Core over its cap is refused before it lands, where before an over-cap
  Core would have blanked the whole memory block at compose time in
  silence (that refusal is now logged as a warning, by conversation). Every
  mutation is saved through the onion store when one is configured. The
  model's tools have no path to any of it; the no-pipeline contract now
  names the two modules that import the librarian and checks the tools.
- `oo chat`: an interactive session in the calling process, one line per
  turn through the executor and the inference registry, with `/open`,
  `/close`, `/pin`, `/recall`, `/skill`, `/help` and `/quit`; the stream
  goes to stdout and every refusal to stderr by name. The librarian gains
  `close_onion`, which evicts the whole Flesh through the gate
  synchronously, stops at the first refused span and names it, then saves
  and returns the receipts digest and the Core root, and `open_onion`,
  which finds a saved conversation again or refuses by name. `/skill` puts
  a published skill's body into the turn's system prompt; a draft, an
  unknown name and an ambiguous one are refused. The session is
  in-process, so it opens the local conversation store, onion state and
  skill root: it keeps the registry funnel, not the separate-process pack
  protocol. The CLI reference now documents `oo core` too.
- The onion memory persists: `memory/onion_store.py` writes each
  conversation's Core, Cellar, receipts, Peels, Flesh and mirror cursor
  through the repository's connection seam, after every mirror and every
  accepted eviction, and reads it back when the process next sees the
  conversation. The read re-hashes every Core entry, Cellar span and Peel
  against its saved id and the root over the four stores against the saved
  root, and refuses by name the first row that no longer answers; a
  conversation the file does not know is unknown, never an empty state,
  and a refused one is logged by name and left absent rather than replaced
  by a fresh one. The table holds conversation text, so a connection that
  is not encrypted is refused by default (`persistence.require_encryption`
  in `onion.yaml`), where the rest of the tree opens plaintext with a
  warning outside Bulbe mode; the drift ledger gains the same refusal as an
  opt-in. `persistence.path` is empty in the shipped file: the maintainer
  sets it when switching the onion on. The Core has no pin surface yet, so
  what persists of it is what the tests pin; that surface is its own block.

### Changed

- [SECURITY] The chat refuses a tool its turn did not offer, whether the
  model decided on it or it was rebuilt from the model's prose, and a tool
  whose class the machine's mode does not permit, the mode read at each
  call. A person is asked about a call once it is resolved, so they approve
  the arguments that will run, with their labels.
- [SECURITY] A chat turn in Bulbe whose approval hook cannot be set up
  refuses every tool call and says so; it used to run them unasked. A call
  that must be asked and has no way to reach the user is refused, with the
  reason.
- [SECURITY] The agent's run manager runs in the machine's security mode
  when none is asked, and never in a looser one; it defaulted to Daily. The
  dispatch reads the machine's mode again at each call, so a running agent
  is held to Bulbe from its next call once the machine escalates, and the
  teacher publishes a draft, and the run's own skill tool writes a skill,
  only if the machine is still in Daily once the person has answered --
  through the run's own gate, or the run's approval queue when it has none.
  A call that cannot run (no handler, no sandbox) is refused
  before anyone is asked about it. An evaluation run takes the machine's
  mode too, and a call that needs a person is refused at once with that
  reason, instead of waiting on an approval queue nobody reads.
- [SECURITY] On every turn the chat composes, its own web search (the web
  search switch, the search step of a pipeline) sends the user's own words
  -- typed, or a hook's rewrite of them -- and nothing else: never the words
  of a file attached to the turn, which it used to send with the question,
  nor a vision model's description of an image, nor a later pipeline step's
  prompt with the model's analysis. A turn with no words of the user's (one
  stored before turns had origins, a file alone) is not searched. It meets
  the network policy like a tool call: a hook's rewrite, or the question of
  a caller that vouches for no turn (sent as it is under free), is asked
  about or refused under the strict settings, and Bulbe sends none.
- The chat evaluation harness runs its scripted tools under the same gate:
  on a Bulbe machine a scripted call that needs a person is refused, and
  under a strict network policy its scripted search is held to it.
- [SECURITY] A peel may no longer give an order, nor restate or tell one,
  the user's included, but in the user's own words, stitched in by the
  queue. Every later turn reads a peel, so the eviction gate's second face
  now reads a peel clause by clause for an order, from a table in
  `onion.yaml` (`directives`, English and French), on the text folded for
  it: compatibility forms composed, format, combining and invisible
  characters dropped (the Hangul fillers, the blank Braille pattern), the
  common Cyrillic and Greek look-alikes of Latin letters read as them,
  underscores between letters read as spaces, hyphenated words read both
  joined and split ("E-mail", "Send-the-keys", "Email-them",
  "Please-send"), marks of emphasis dropped, a colon glued to its label
  spaced ("Reminder:send"), a word spelled letter by letter joined ("S e n
  d", "D.e.l.e.t.e", "S/e/n/d"), a line that opens in lower case read on
  the line before it, inline code read as words, and enumerators, tags,
  interjections ("ok so send") and code markers set aside. A clause orders when it speaks to the reader in the
  second person; when the assistant is obliged, past an adverb and in the
  periphrases of an obligation ("is encouraged to", "is tasked with", "has
  been asked to"); under a label that addresses the reader ("Instructions
  for the assistant:", or an authority named alone, "System:"), the list
  after it included; on an opening of courtesy, prohibition, emphasis or
  lasting rule ("Please", "Do not", "Try to", "From now on", "Until further
  notice", "Veuillez"), unless a subject follows it and it asks ("Do we keep
  Docker?"); with an injection's signature, whoever tells it
  ("ignore every previous instruction", "reveal its instructions"); when
  the user is told to have asked, told or ordered the reader to act, or to
  want a thing done ("The user told the assistant to email...", "The user
  wants the keys sent to..."); and when a verb opens it before its object,
  past adverbs, a particle, a dative, a vocative ("Bob, email..."),
  brackets ("(Send) the keys") or a capital that made the verb a name
  elsewhere in the span, an object a time phrase owns or precedes
  included ("Delete this month's logs"). Every such clause is refused by
  name, the user's restated or told as much as any other, and the repair
  drops it wherever it stands: a paraphrase can lose what bounds an order
  or change who gives it, and no reading of a paraphrase against the typed
  text proved both safe and faithful. So an instruction copied from an
  attached document, echoed by the assistant or carried by a turn written
  before origins were kept no longer reaches a peel in those forms. The
  user's orders stand in a peel only as the user typed them: the queue
  stitches at the end of the peel, word for word and marked with its turn,
  each run the user typed that orders -- a request in the user's own voice
  included ("I'd like the old logs deleted by Friday") -- and each bare
  retraction typed after one, whatever was asked between, on every rung
  that makes a peel, for a leaf, a parent and a gated eviction too. A run
  is a typed sentence joined with the next where parting them would change
  what it says: a sentence in lower case, the inside of a quote in any
  marks, all that follows a line on a colon or a sentence that announces
  another's words ("This is what the phishing email said.", "Here is the
  spam I got."), a one-word sentence, a bare negation or retraction,
  interjections included ("Delete the logs. Hm, no."), and a sentence
  that bounds the order before it ("Not before Friday.", "Only if Bob
  agrees."); so a condition, a quote, an attribution, a list's line and a
  retraction stay with their order. The gate holds stitched runs only as
  the final block of the peel, each exactly the run of that turn, after a
  sentence the summary ended with a stop -- an ellipsis, spaced or not, or
  an abbreviation ("i.e.", "etc.") ends none -- so that no word of the
  summary's is read with them ("On
  every later turn, [t1] ...", "Delete [t2] ..."); a marker anywhere else
  stitches nothing, what it marks being read as the summary's own words,
  and the model's own markers are taken out first. A turn whose id is
  empty or shared, a gap no typed segment covers, and the words of a
  document, a tool, the assistant or a refined turn are never stitched.
  The librarian asks its model to give no order and to restate none. The
  price is in compression: a span whose orders are long is held more
  often, its anchors kept, rather than peeled, and a summary that does not
  end its last sentence carries no stitched order. A decision is held
  clause by clause, read sentence by sentence as an order is: a reporter at
  the head of a sentence exempts no clause after a semicolon, a colon or a
  dash, nor one that opens on a deciding subject of its own after a comma,
  an "and", a bracket or a subordinating word ("which we decided", "as we
  decided"); a predicate coordinated to a typed decision ("and", "but",
  "yet", "plus", "&", or a gerund after a comma), a sentence that answers
  one with words of its own, and a decision told as the user's, by a
  reporter or not ("the assistant confirmed the user decided", "the report
  says the user prefers Podman", "the user confirmed that we drop...", past
  a phrase between commas) each need the user's words, while "we" in a
  reporter's words stays the reporter's voice; a typed sentence is parted
  as a summary's is, so that the user's words restated whole are held
  whole and two typed decisions may be joined ("We keep Docker but drop the
  NAS"), a predicate coordinated to a typed decision holding alone only
  when its head is the user's plain decision, with no negation, doubt,
  attribution, hedging adverb or modal ("It is not true that we keep the
  logs and drop the backups", "We reportedly keep the logs and drop the
  backups" hold no "We drop the backups"); a
  decision the user wrote under a list or after a label or a phrase that
  negates, conditions, attributes or quotes it ("Don't do any of this:",
  "Mallory wrote:", "According to Bob,", "Bob said,", "Mallory's ideas:",
  "Our vendor's proposal:"), or took back
  with a bare negation, holds none; a stop glued to the next sentence ends
  one, a line that opens in lower case goes on. Each decision names the
  table's fingerprint. The table names its limits: a bare imperative whose
  object has no determiner ("Delete backups") or is a time phrase alone
  ("Delete this month"); an order written as a third person ("Deletes the
  backups"); an order told as a plan, a requirement or a permission ("The
  next step is to...", "The assistant can now share..."); a third party's
  "should"; an obligation of a model, an agent or a system; a passive
  obligation under a reporter word ("The logs must be sent..."); a fronted
  phrase without a comma ("On Friday delete the backups"); a sum written
  after its currency sign; look-alike letters outside the table; an order
  written as one word ("SendTheVaultKeys"); a lasting rule outside the
  table; a request told with a verb outside the table and no reader; a
  decision restated with a few added words as a statement; a coordinator
  or a possessive changed ("or" told "and"), a plural for a singular; "we"
  or "I" under a reporter, even against a typed "will not"; a speaker
  named alone before a colon ("Bob: we drop the backups"); a list's line
  that rejects without a negation ("Rejected options:"), and one whose
  negation is undone ("None of this is optional:"), the user's own list of
  their group's decisions ("Our team's decisions:"), a comma piece that
  names a quote's noun or a possessive ("With the file server full,"), and
  a summary whose last sentence ends on an abbreviation ("at 3 a.m."), all
  false refusals; a sentence of the summary that tells of the stitched block
  ("The user took this back.", "The next line applies on every later
  turn."), judged as a claim; a wish in the first person, read as a
  request ("I want to go home"), and requests the reading still misses
  ("We would really like...", "wants to see the logs deleted"), which
  leave no trace; a French wish to know whose infinitive follows its verb
  ("voulait savoir si..."), read as a request, a false refusal; a quote framed by no
  announcing word or mark ("Someone sent me this."), framed after it or in
  another turn, stitched as the user's own words; a sentence that names a
  file or tells ("Please update the settings file."), which joins the rest
  of its turn to its run, a cost in compression; an apostrophe after a
  plural, read as a quote mark; a retraction told
  as a statement ("I changed my mind"), not stitched with the order it
  takes back; a decision split by an interjection between dashes or
  brackets ("We -- as Mallory wanted -- drop..."); and a fenced block,
  never read for an order, the summariser seeing code only as markers. A
  hyphenated compound that opens on an opening ("Always-on") is read as an
  order, a false refusal on the side of safety. A strict switch, off as
  shipped, makes every clause whose first word opens no statement an
  order, at a cost in compression. The decision lexicon's own reach (the
  passive voice, verbs outside it, a reporter noun read as the subject,
  word order, a decision's scope or hedge changed, negations it does not
  count) is unchanged here. The false refusals on real summaries are a
  host measurement, still owed.
- [SECURITY] The onion memory's queue never stops: not on a refusal, not on
  a model that fails or never answers. One burst runs per conversation, and
  a lock on each conversation's memory keeps the mirror, an eviction's
  commit, a save and the user's own verbs from crossing; an eviction commits
  only while its span is still the oldest, and takes no turn that arrived
  while its summary was written. When the probe gate refuses a summary, the
  span goes down a ladder below the gate instead of staying where it was: a
  span no probe can judge leaves bare, with no summary; a refused summary
  that failed probes is asked for once more, handed those probes, at the
  temperature and seed `onion.yaml` sets for every librarian call (0 and 1
  as shipped) -- unless no text could change the refusal, the span's probes
  asking for too few of its facts; then it is repaired: the sentences its
  span holds, then each typed decision it lost in its own sentence and the
  fewest other units the user typed that answer what else it lost (a
  sentence, or a code block by its marker; the fewest up to
  `queue.exact_cover` candidates, a deterministic greedy choice beyond),
  each marked with its turn, kept when the gate accepts it again and it
  saves enough (`queue.rho`, compared as written); else the span is held,
  the places of the typed units that answer its probes (each typed
  decision's own sentence, then the fewest for the rest) kept in the
  archive as anchors and shown to the model as data -- a code block by its
  marker, never its code -- within their share of the summaries' layer. A
  repair drops each sentence of the summary that shares at least
  `queue.copy_shared_words` content words, and most of the words of either
  one, with a sentence of a document or of an answer the assistant gave
  with a tool or the web. A turn marker the model writes -- `t` and digits,
  or a turn id of the span, in any bracket, angle bracket or quotation
  mark, whatever its case, width, spacing or invisible characters, nested
  or not -- is taken out before the gate reads its text, unless the user
  typed it so. Only words the user typed are ever stitched into a summary
  or kept as an anchor; the archive keeps every turn whole, for recall. A
  refusal is marked by a hash of what failed and of the call refused, never
  its words, and a span that comes back with the same mark is not asked for
  again. A call that fails or outlives its deadline
  (`librarian.call_timeout_s`, held by the librarian whatever the backend
  does) is counted and logged by its class, and the burst or close that met
  it goes on without the model; a close says when it ended so, and no call
  to a model starts while an abandoned one of it still hangs. Receipts
  carry no word of their span -- their key, turns, kind and origins -- and
  the oldest fold into one line under the receipts' cap, so the memory
  block never goes blank for having evicted too much. Closing a
  conversation runs the same ladder and waits for a burst in flight. The
  store names the schema a file was written with: an older file is
  migrated once, conversation by conversation, each root proved first, the
  receipts' lines written again without the words they held and the root
  recomputed, each under a savepoint of its own; a conversation that does
  not prove -- its root, a span its receipts name, a row that does not
  decode -- is left as it was and refused by name on its own load; an
  error of the database itself rolls the whole migration back, to be tried
  again at the next opening; and a file of a newer schema is refused by
  name and left as it was. The temperature of the librarian is read from
  `onion.yaml` alone.
- Proposals to the Core. A decision the user typed, as the probes read it,
  that a held span keeps is offered to the Core in its own words and never
  written there by itself: the terminal session lists them with
  `/proposals`, and the user takes or sets aside one at a time (`/accept
  ID`, `/decline ID`), its exact words shown with its turn and origin. At
  most `queue.proposals_per_day` are offered per conversation and per day
  (UTC); a decision past the cap is deferred with the conversation and
  offered on a later day, when the proposals are listed or a step runs;
  what a listing opens is saved. A proposal whose place no longer holds a
  typed decision is neither shown nor accepted, a deferred one of them
  takes none of a day's room, and one moved in the file is refused by
  name. What the queue does is counted in the process -- bursts and their
  breaks, evictions by rung, refusals by motive, failed calls, receipts
  folded (each once), anchors left out or no longer placed (at each
  composition), refused memory blocks, proposals -- without a word of a
  conversation. The onion arm of `scripts/drift_ab.py` hands the librarian
  each turn with the origin the executor gives it and stops asking a model
  that failed, as a burst does.
- [SECURITY] The onion's mirror knows each turn by what it is -- its role,
  its words, its origin and its segments -- and no longer by a count. A
  retry that replaces an answer, a synchronisation that writes the
  conversation again from another device (every turn legacy there), and
  turns deleted from the end leave as many turns as before, or fewer, with
  other words in them: the mirror now follows them. Where the conversation
  first differs from what the onion holds, the turns that follow are taken
  back from the window, and a span already in the archive is superseded
  with every later one: its receipt stays in the ledger and its span in the
  archive, byte for byte, recallable by the user, but its summary, its
  anchors and its receipt line leave the memory block, and a proposal
  made from it is superseded too and can no longer be accepted. One the
  user was offered still counts in that day's cap, so edits never offer
  more in a day than the cap; made again in the same words from the turns
  mirrored again, it takes the place of the offer the user already had,
  and no more room. A decision the user accepted or declined is not
  offered again from the same turn, word for word, when that turn is
  mirrored or sent again: a copy deferred before the verdict is never
  offered, and one already offered stays an offer of its own, decided one
  at a time. The conversation is then mirrored again from there under turn
  ids never given before, across a save and a load as well, so a step whose
  span was taken back while its summary was written evicts nothing; a Flesh
  row the mirror cannot read as a turn is taken back the same way. A
  synchronisation makes every received turn legacy, so it supersedes the
  whole archive and the conversation is summarised again, a burst at a
  time. A refusal mark is kept only while its span is the next one asked:
  a conversation holds one at most.
- [SECURITY] The librarian's calls are governed. Each call asks the resource
  governor for a ticket of its own, as `librarian`, a background caller, for
  the context `librarian.num_ctx` names, and holds it while the backend
  answers; between two calls it holds none, and the governor's background
  gate keeps the next one waiting while a chat turn is in flight. A span
  whose prompt and answer do not fit the window -- the context asked, then
  the context admitted, less `librarian.window_margin` for the token
  estimate -- is not sent, and the step goes on without the model; so does
  one the governor does not admit, and the burst or close then asks no
  more. Once the calls of a burst or a close have taken
  `librarian.run_budget_s` seconds (300 as shipped), no other call starts:
  a run spends its budget and one call at most, so a model that answers
  just under its deadline no longer costs a close two deadlines per span.
  A call that cannot start, an earlier one to its model still hanging,
  asks the governor for nothing, and one admitted that can no longer start
  hands its admission back. The model stays resident for a burst or a
  close (`librarian.keep_alive`, now `5m`) and is let go at its end
  through the governor, which unloads it only when the background alone
  loaded it and no call is in flight on it. The shipped librarian model is
  none of the models the shipped routes answer with, and `onion.yaml` says
  so.
- What the onion's queue counts is kept across processes in a file of
  aggregates (`counters.path` in `onion.yaml`, under the data directory):
  names and numbers, never a word of a conversation nor its id, with the
  native core's share of the probes; written whole or not at all, merged
  under a lock with what other processes wrote, one write at a time within
  a process, and never written over when it does not read as counts. They
  are written at the end of each burst and close, after each decision on a
  proposal and each listing that opens deferred ones, when the terminal
  session ends -- `/quit`, the end of its input or an interruption -- and
  from the chat path at a turn once
  `counters.flush_every_s` has passed (60 as shipped); what a process
  counted since its last write is lost if it stops before the next. The
  host runbook asks each of its calls as a run of its own and reports a
  call the governor does not admit, or one that does not fit, by its name,
  never timed as a call of the model.
  `GET /api/memory/onion/status` and the terminal's `/status` show the
  counts and whether the onion is on. The
  proposals have their routes: `GET /api/memory/onion/{conversation_id}/proposals`
  lists them word for word, and `POST .../proposals/{id}/accept` and
  `.../decline` decide one, as the user.
- `scripts/drift_ab.py` curates at the chat path's rhythm by default: the
  librarian's launcher fires a burst once the conversation grew by
  `min_new_turns`, on a thread of its own, while the next turns are
  answered, and the arm ends once its bursts have. `--mode fast` keeps the
  curation after every turn. Both ask for the second summary, and the
  report names its mode and carries the engagement -- bursts, evictions by
  rung, refusals by motive, the block's tokens at each turn -- with the
  probe generator's version and the decision lexicon's fingerprint.
- [SECURITY] Every conversation turn is saved with the origin of its words:
  typed by the user, the user's question as the model rewrote it (refined), a
  document the user attached, the assistant's answer -- flagged when tools or
  web results stood behind it -- or legacy, for a turn written before origins
  were kept, synced from another device or imported. An attached document is
  a part of its own, and the words the executor writes between the question
  and the document belong to no one. An origin outside the closed grammar is
  refused by name before anything is written. Origins and segments are bounds
  on content, never text, and the model never sees them; they travel into the
  onion memory, where the archive, its receipts and its summaries answer for
  the union of their turns' origins.
- [SECURITY] A file attached in the chat travels beside the typed words, no
  longer wrapped into them: each is saved as a document part of its own,
  under a line the executor writes naming it, so no file's text is saved as
  the user's words, whatever it contains. A request is held to the bounds
  `config/chat.yaml` sets -- files per turn, bytes per file, characters per
  name -- and one over them is refused by name before anything runs or is
  written; files with no typed words still make a turn. Words a hook or the
  vision step rewrote are saved as refined. The turn as the chat route
  composed it reaches every path that saves it -- the executor, the agentic
  pipelines, the execution pipelines and the coding agent -- and vouches for
  its own text only: a prompt a pipeline step or the coding agent composes
  from it is saved as legacy. A retry re-creates
  the turn with the origin and segments it was saved with, read as the
  grammar admits them, and never more trusted. The `pre_inference` hooks are
  shown the files beside the words -- hidden, as the words are, from a
  plugin without the `inference_content` permission -- and may rewrite
  their text within the bounds of `config/chat.yaml`, which stays a document
  part; a hook that edits what it is shown in place and then fails changes
  nothing. On the coding path, the hook of each model call hides the words
  and the files from such a plugin too. The coding agent reads its
  directives from the typed words
  alone, on a retry too, and every phase it runs reads the whole turn.
  Routing reads the files with the words; on the plain chat path the web
  search and the memory read the words alone. The chat page shows the user's
  message as it is saved.
- [SECURITY] Only typed text decides. A decision probe is drawn from the
  user's own words alone -- never from the assistant's, a document's, a
  rewritten question or a legacy turn -- and a native twin that draws one
  elsewhere is overruled. A decision needs no marker: "we keep", "on garde"
  or "let's go with" are read by a bilingual lexicon in `onion.yaml`, each
  act in a class, and a summary keeps a decision only with an act of each of
  its classes, its dates and numbers, and the same polarity. A decision is
  read however its text is written: accents typed or not, in either Unicode
  form, its acts included; a marker of two words split by a line break or a
  no-break space; an elided word read whole ("prevu d'utiliser"). An
  accented marker decides wherever it opens a word, and never inside one
  ("helicopter", "undecidable"); an English word that opens with one
  ("decide between") is read as a decision, the safe side, where the span
  stays verbatim. Negations are counted word by word: one written as a word
  in inline code ("`will not`", "`won't`") inverts as a plain one, and one
  fused into a flag such as `--no-cache` is none, in prose as in code.
- The recall probes read French and English as they are written: a date in
  any common form to its canonical day, month or day of a month, never a
  relative one; a number to its value, its written precision kept, with a
  unit from a closed table (Go is GB, nothing converted), and a writing that
  reads two ways kept as written; a name by the span it stands in, a word at
  the head of a sentence only where the span capitalises it elsewhere, a
  part of a full name as its alias; list items without their bullets; a
  fenced code block as one artifact keyed by the digest of its body, nothing
  inside it probed, inline code never. Every figure names the version of the
  generator that drew its probes.
- [SECURITY] The eviction gate has a second face. A summary that says what
  its span does not hold -- a name, a date, a number, a code block, or a
  sentence that decides with no typed decision behind it -- is refused by
  name, whatever its probes answer; a sentence whose subject is the
  assistant, a document or a tool reports another speaker's words. Two bounds
  join it, each refused with its figure: the share of the summary's content
  words its span holds no word for (`max_novelty`), and its length against
  its span's (`max_length_ratio`).
- The gate no longer takes its probe set on trust. It reads the span's facts
  again, piece by piece and never through the native core, and refuses a set
  that asks for a smaller share of them than `probe_floor`, naming the facts
  left unasked: a generator, a twin or a caller that draws less can no
  longer raise an acceptance rate. Every decision carries that coverage and
  the recall of the probes measured on a labelled fixture set
  (`gate.probe_recall`), stated with the generator's version and the
  lexicon's fingerprint and never borrowed by another;
  `scripts/onion_runbook.py --probe-recall SET` reads the recall on a labelled
  set of real conversations, counts only.
- A code block summarised away is read back by the key its marker names:
  `POST /api/memory/onion/{conv_id}/code/{digest}`, or `/recall code:KEY` in
  the terminal session; a malformed, unknown or ambiguous key is refused by
  name, and nothing changes.
- Peel selection reads words in any script, folded as the probes fold them,
  with the function and question words of both languages aside: a French
  query no longer breaks at its accents.
- What the native core serves of the probes' work is counted, call by call,
  with the reason the reference served the rest, and reported by the
  runbook. The core's probe twin, written for an earlier generator, is not
  asked until it declares the current one.
- [SECURITY] No system message carries data any more. The working memory,
  project retrieval, web results, archive snippets and every summary of
  earlier turns ride the user role, each wrapped as untrusted data under its
  own source label and joined to the turn it belongs to, on the context
  optimizer, the manual pipeline and the single turn alike; the system
  message is the instruction head alone. Project text and archive snippets
  were not wrapped at all and now are; with no wrapper they are withheld.
  Runs of user messages are joined, so a chat template written for strictly
  alternating turns never meets two in a row. The `stable_prefix` switch in
  `context_optimizer.yaml` no longer moves anything: the head is stable on
  every path, and the switch only asks a llama-server engine to reuse its
  prompt cache. Cache fingerprints cover the same head and tail as before.
- [SECURITY] A summary is made from conversation turns only, never from an
  earlier summary. The manual pipeline no longer restores a stored summary
  beside the archive it loads in full: a summary stands only for turns the
  window lets go, computed from the archived turns, with the frozen tier
  segments inside that span composed as they are. The cumulative merge and
  the tier rollup are gone; the composition keeps the first segment and the
  newest that fit, and says how many it left out. Its room is reserved before
  any turn is cut and bounded by a share of the window (`compose_share`), so
  no verbatim turn is cut to make room for a summary and a small window keeps
  its recent turns. The segments are weighed as the engine counts, and the
  summary is fitted as it is placed, envelope included: what an estimate
  misses comes off the segments, counted as left out, never off the turns.
  Once a summary is placed, the soft limit is a target rather than a cut,
  and the hard limit still bounds the window. The three summarizers hand
  their model one JSON object per turn, so no turn can forge another. The
  pipeline's cutting steps keep a summary while an older turn can still go.
- [SECURITY] A frame marker of the onion's composer -- an opening `[data` with
  an attribute, or a closing `[/data]`, with or without attributes -- inside a
  recalled segment, the Core and the turn included, or inside any other
  wrapped block, is defanged: only the composer's own frames reach a window.
  A bare `[data]` (an index, a list, a section header) is ordinary text and
  is left alone. Unified retrieval skips what the turn's data blocks already
  carry, as it did when they sat in the system prompt. The response cache
  key covers the whole turn, so a user turn left without a reply no longer
  shares a key with the history without it.
- The live summarizer's model, fallback chain, temperature, output cap,
  timeout, input cap and message threshold, and the tier budgets, are read
  from `compression.yaml` (`live_summary`, `summary_tiers`); each value is
  checked and a wrong one is refused by its full name. A file that cannot be
  read, decoded or built leaves the live summary unavailable, with the reason,
  where it used to stop the pipeline from importing. The shipped values are
  the former ones. The named summary model is now tried before its
  fallbacks.
- [SECURITY] Reading a receipt's span no longer closes the receipt. Closing
  it is the user's verb: `POST /api/memory/onion/{conv_id}/resolve/{key}`,
  or `/resolve KEY` in the terminal session; any other actor is refused by
  name, and no module a model can reach imports either verb.
- The drift A/B harness places the onion's block where the chat path does,
  in front of the turn, with its frames.
- The public-clean guard reads every tracked line, the source trees
  included: their standing debt is paid, so its rule now holds everywhere.
  The contract that pinned the old boundary is superseded by name.
- The context optimizer's window statistics name their strategy
  `optimizer`; the value used to carry a work code. The last eight lines of
  work codes and process words under the source trees, in four test
  suites' docstrings, are gone, and so is a work-block word left in a
  published schema description.
- Four test suites load their modules through the shared isolation window
  instead of windows of their own, with every test and assertion as it
  was; the isolation seal owes for 87 suites, down from 91.
- Comments, docstrings, a tooltip and five published API descriptions no
  longer number the work in French "blocks": 155 lines in 34 files. A French
  comment in the chat view is now English. The public-clean guard charges
  that word, capitalised as the numbering spelled it; in lower case it is
  ordinary French for a block of code, and the language guard's business.
- Comments, docstrings, one log message and one local name in ten files no
  longer carry work codes, internal document names or process words. Each
  file is proven by the comment-only guard: its executable shape is
  unchanged, or the change is a string purge or a proven rename.
- Opti-Oignon is licensed under the GNU Affero General Public License,
  version 3 only (AGPL-3.0-only), from this change on. What was published
  before it, releases 2.1.0 and earlier included, stays under the MIT
  License. The licence text, the README notice, the contribution terms, the
  package metadata (`pyproject.toml` and both crates) and the package's own
  `__license__` change together.
- The public-clean guard holds markdown names in capitals to a closed list
  of the public document names the tree writes: the documents it carries,
  the release notes it generates, the skill format's file and two fixture
  names its contracts write. Any other such name of four characters or more
  is charged on the lines the guard reads and as a tracked file's name:
  internal documents live outside the repository and are never named from
  inside it. A settings hint no longer points to an installation document
  the tree does not carry.
- The public-clean guard holds every line outside its scan trees to its
  whole rule, standing lines as well as added ones: at the root, under
  `docs/`, under `.github/` or in any other directory, a session code, an
  internal document reference or a process word fails it, since the debt
  there is zero. Inside the scan trees the standing debt is still left to
  the diff pass until it is paid. A tree git cannot read now fails the
  guard instead of passing it on nothing read.
- Comments, docstrings and ignore rules no longer name internal planning
  documents or work-block codes: 70 lines in 52 files. Two of them were
  published, an endpoint description and a schema description, and the
  prose digest records the change.
- The public-clean guard reads the whole tracked tree as well as the diff:
  no tracked path may carry a session code or the name of the tool used to
  write the tree, and no tracked line may name that tool. A file at the root
  sat outside every scan tree of the diff pass; a stale checksum manifest
  named after a work block had shipped there, and is removed. The ignore
  entries for that tool's local files leave the tracked ignore file for each
  clone's own exclude file, the ladder finds its root from its own location,
  and its directed-mutation tier reads the blade register's path from the
  local git config key `oo.bladeRegister` (a named skip when unset).

- The Workshop's settings pages are quiet lists: each group is a row,
  titled once, and one group is open at a time. Only the open group loads
  its panel, so entering Models and inference no longer starts thirteen
  panels' reads, and the model health poll, at once; a closed group's panel
  never asks the server anything (its code may still come with the page's).
  A group whose panel was used on this visit (a field typed in, a choice
  clicked, a switch pressed, a file dropped) stays mounted, hidden, once
  closed, so what it held is there when it reopens: a draft, a queue of
  files, an ingest's progress. A search on the page hides the groups
  instead of dropping them, so a draft outlives it too; leaving the page
  still drops what was not saved. The address names the open group (`?g=`): the command
  palette, the settings search and a shared link open it, focused and
  scrolled into view, and a link to one of the inference pipeline's
  dashboards opens the group that shows them. With no group named, a page's
  only group opens, and a page of several opens none. Preferences' groups
  do not fold. Each group's title is drawn once, by a new header of the
  design system (`PanelHeader`): the whole row is its target, and its
  description is read as the button's description, not as part of the
  heading. Twenty-nine panels repeated their group's title above their own
  content; none does now, nor writes a heading at or above its group's
  level, and a Workshop page no longer repeats its own title as a section
  heading. The design system's row of tabs scrolls within itself when it is
  wider than its place, with room for the focus ring on every side, and
  keeps the selected tab in view, a link to a far tab included; a dialog no
  longer opens with its focus on the controls in its head.

  Observability lists the inference pipeline once. Its group, now called
  "Inference pipeline", holds an overview and one row of tabs (Telemetry,
  Telemetry history, Profiler, Performance dashboard) drawn from the
  catalog, which marks the four dashboards as embedded in it; they used to
  be both tabs of that group and groups of their own on the same page. The
  overview says each state in words ("Collecting", "12 requests profiled",
  "History off") where a coloured dot stood. A control that changes the
  tab (an overview's "Open", a model picked in the profiler, the history's
  filter cleared) hands focus to the selected tab, and clearing the filter
  draws the history again unfiltered, where it used to stay filtered. The knowledge base no longer
  holds the retrieval dashboard in a sub-tab, nor the installed plugins the
  marketplace: each is mounted once, as its own group. Model assignment
  lives only in Models and inference, where it now says the roles could not
  be read (with a retry) instead of "No role assignments found.", shows a
  refused save under its role, and keeps the editor open on the reader's
  choices when a save fails. A group's feature gate names a key the server
  serves and the panel's own routes check: the knowledge base, the
  retrieval dashboard, the installed plugins and the marketplace were gated
  by keys the health map never served, so they could never show as
  unavailable; the telemetry history and the plugin allowlist carry none,
  their routes checking a flag the health map does not serve. The history's
  retention could not be saved (its client called a helper it never
  imported); it can. "Fine-Tune export" reads "Fine-tune export".

  The Security page opens on the grade the status card's badge shows: its
  letter, its score and how many checks passed, and, behind a disclosure,
  each check with its points and its detail, whether sessions use httpOnly
  cookies, and the ten most recent security events (sign-in activity:
  logins, failed logins, registrations and password changes; sandbox
  blocks, sign-in lockouts and detected search injections). Only a panel no page
  reached showed them; it is deleted, with the unreachable claim verifier
  (the Verify page's pairs mode does its work), that verifier's client and
  an unused focus trap. The badge reads the grade through the API client,
  so an expired session now takes the reader to the sign-in page, as every
  request does, where the badge used to hide itself.

  Stop all sits in the head of every dialog of the Workshop's pages (a
  run's detail, fine-tune's and the knowledge base's deletes, the
  documents' two deletes, hardening's confirmation, the marketplace's
  reviews, remote access's two confirmations, device sync's rename) and of
  the first-run dialog. That dialog no longer reloads the page after a
  preset: once a preset is applied, closing it ("Get started", Escape or
  its close button) reads again what the preset changed (the chat's models
  and default model, the feature map, the backends, the chat control bar's
  switches), and so do applying a preset and "Reload from disk" at Models
  and inference, which refreshed none of it before. It cannot be closed
  while the preset is being applied. Its presets are one radio group; the
  recommended one says so in a word, and the dialog opens on the chosen
  one. The grade's, the roles' and the first-run dialog's retries keep
  focus while they read again, and what replaces a pressed control takes
  it.

  The benchmarks page is one row of tabs over the quality evaluation (Run,
  Leaderboard, Head-to-head, Trends, Compare, History, Profiles), where two
  rows used to nest, with a second "Run" and a second "History" for the
  older suite engine. The tab lives in the address, and a run opened from
  History closes back onto History. What left the interface with the older
  engine's pages: its keyword-scored suites, the temperature and timeout of
  a run, its live table of results, and the list of its past runs with
  their deletion, for which the evaluation has no equivalent. Those runs
  stay on the server (`GET` and `DELETE /api/benchmark/runs`), and none of
  its routes changed; an old `/benchmark?tab=models` link lands on Run.
  System status's section that times the server's own components is now
  "Component latency", with a "Measure" button: it was titled "Benchmarks",
  and it benchmarks no model.
- Run `npm ci` in `frontend/` after pulling this change: the reply renderer
  adds `marked`, the frontend's third runtime dependency, and an install
  made before it does not hold it, so the app fails to load. The launcher
  (`python -m opti_oignon`) now does this by itself. It used to install only
  when `node_modules` was missing, so an install older than the lock stayed
  as it was while the launcher still said the frontend was ready; it now
  compares what `node_modules` holds (npm's own record of it,
  `node_modules/.package-lock.json`) with `package-lock.json` package by
  package -- version and digest, or the source for a package pinned without
  a digest, so the same build fetched through a registry mirror is not
  reinstalled; an optional or dev-optional package npm skips on this system
  excepted -- and runs `npm ci` when they differ, saying why, with npm's
  output left on the terminal. A failed install, or no npm, stops the
  launch before any port is touched. The frontend's Dockerfile installs
  from the lock with `npm ci` too. `scripts/dev_frontend.sh` and
  `scripts/run_e2e.sh` still install only when `node_modules` is missing:
  run `npm ci` yourself before them after this pull.
- The merge guards read the native crates. `public_clean_guard.py` and
  `comment_only_guard.py` covered five published trees and `rust/` was not
  one of them, and `public_language_guard.py` read Python alone: the 10,598
  lines added under `rust/` since `origin/main` (10,472 of them in 22 `.rs`
  files) reached none of them, a contract running the clean guard's patterns
  over the two crates in the meantime. All three read `rust/` now. The
  release recipes' non-ASCII and nomenclature censuses take their trees
  from the clean guard and so reach `rust/` as well, but they charge only
  the paths a release manifest lists, and the tracked manifest lists no
  `rust/` path yet: until one does, they charge nothing there. A contract
  holds every tracked file under `rust/` to printable ASCII, tab and newline
  instead, the rule one source directory of the engine already had. The
  clean guard is a regex over added lines and needed one entry in its list.
  The other two needed a Rust lexer, which lives in the comment-only guard
  and which the language guard loads by path, so the two read a Rust file
  the same way: block comments nest, a raw string holds any quote behind
  its hashes, an apostrophe opens a char, a lifetime or a label by what
  follows it, an identifier glued to a literal is that literal's suffix and
  never the prefix of a raw string, one byte-order mark and a shebang line
  are skipped by the compiler's own rule, and only a CR LF pair ends a line.
  Each of those, got wrong, turns every quote after it round and reads code
  as comment, so that a changed value passes as a purged comment or a French
  sentence as code; each was measured on a file the compiler accepts. A
  comment removed from between two tokens leaves a space, so a purge cannot
  fuse two names into one, and a comment, string or char the lexer cannot
  close is refused by the comment-only guard and reported unparsable by the
  language guard. The language guard reads one file kind per tree, `.py`
  under the Python trees and `.rs` under `rust/`; it skips Cargo's build
  directory (a `target` beside a `Cargo.toml`, where a third-party build
  script writes Rust sources of its own), reads a `//` comment line by line
  and a run of `///` or `//!` lines, a block comment and the value of a
  `doc` attribute as one passage each (the value only inside an attribute,
  a `concat!` of strings included; a `format!` argument named `doc` is
  data), and names `rust/` unreadable, failing the run, when the lexer
  cannot be loaded. The comment-only guard now refuses a TOML file holding a
  multi-line string, since its hash model would read the lines of the value
  as comments and the crates' `Cargo.toml` files are in its reach. The debt
  found under `rust/` was zero -- no internal nomenclature, no French in 405
  passages, no character outside ASCII in the 26 tracked files -- so no
  ledger gains an entry.
- The clean and language guards read every added line of a diff, and all of
  it. Both read `git diff` in text mode and cut it with `splitlines`, which
  also breaks a line at a lone CR, a vertical tab or a form feed: the rest
  of that added line came back without its `+` and was never read, so an
  internal code placed after one passed the clean guard, in every tree. And
  both took any line opening with `+++ ` for a file header, which is what an
  added line whose own text opens with `++ ` looks like in a diff: its code
  was never read, and the language guard handed the hunks after it to
  another path. The diff is now read as bytes, cut at newlines alone, and a
  header is recognised only before a file's first hunk. On the current diff
  over `origin/main` (97,615 added lines) both readers return exactly what
  they returned before.
- Still outside the guards: TOML comments and string literals (data, as in
  Python) for the language guard, the prose of the TypeScript, Svelte,
  Kotlin and shell trees, any file under `rust/` that is not `.rs` (a
  module a `#[path]` attribute points at, a file an `include!` or an
  `include_str!` pulls in), and a doc a macro assembles out of its
  arguments. A doc comment is free in the comment-only guard, like an
  internal docstring, although a fenced example in one is compiled and run
  by `cargo test`; the crates hold none today. Under `rust/`, as in every
  tree outside Python, the line-level prover accepts a line whose literal
  lost its code while the code beside it changed, and a movement it cannot
  attribute is NOT JUDGED, which does not fail the run: for a Rust file the
  comment-only guard fails only on a source it cannot lex. A renamed file
  escapes it, in every tree.
- The interface has three palettes, day, night and high contrast, and one
  way to reach them. Each palette is one file of 26 named roles and its
  colour scheme (`frontend/src/styles/theme-{day,night,high-contrast}.css`,
  under `[data-oo-theme="<id>"]`), and one derivation layer
  (`styles/theme.css`) maps every token a component reads onto those roles,
  once; the Anthracite, Parchment, Slate and Linen files are gone, and the
  high contrast palette is a new, warmer dark set rather than pure black.
  The choices are "Match system" (the default: night, day when the system
  asks for a light scheme, and high contrast whenever it asks for more
  contrast, whatever its scheme), Day, Night and High contrast. What was
  stored before migrates without a write: Anthracite and Slate become Night,
  Parchment and Linen Day, and a light or dark theme stored alone becomes
  Match system when it is what the system shows anyway, and Day or Night
  otherwise. One module, `lib/theme/apply.ts`, resolves the stored choices
  and the system into what `<html>` carries (the palette, the `dark`
  class, one density class, the root font size, the motion classes and the
  componion switch) and writes nothing else; the inline script in
  `app.html` makes the same decisions before the first paint, inside a
  try/catch, over a static night palette that stands when it cannot run, and
  sets the browser's theme-color to the palette's ground. A contract runs
  both over 1,500 combinations of stored values, system settings, blocked
  storage and text sizes and requires the same root. Under Match system a
  change of the system's scheme or contrast applies at once. Nothing stores
  the old binary theme any more (the root layout rewrote it on every visit,
  so following the system stopped after the first one), and the palette
  choice is stored only when the user makes it. At startup the old key is
  removed where it decides nothing (a palette was chosen, or it is what the
  system shows); where it still pins (unlike the system), the choice it
  means is held for the visit, so a change of the system does not move the
  choice shown. The accent colour builder is
  retired, with its API module; custom accents it saved were never applied
  at startup, and the server's theme endpoint is untouched. The theme
  switcher, the appearance settings and the component gallery preview a
  palette as an element carrying its attribute, with no colour of their
  own.
  Text size is now the root font size (92, 100, 109 or 118 percent), and
  the text and space tokens are in rem, so it scales every one of them and
  the multiplier token is gone; sizes in px (the Tailwind px text classes,
  fixed widths, touch targets) do not follow it, by design, and the
  composer's field keeps a 16 px floor. Compact density no longer sets text
  under 12 px. Radii follow a rounder scale (8 to 30 px, and 999 for a
  pill), and the font stacks name IBM Plex Sans and Mono and Source Serif 4
  first (each falls back to the system's fonts until they are shipped).
  Tailwind's `rounded-sm` and `rounded` now mean 8 px (they meant 2 and
  4), so small swatches read rounder.
  `tailwind.config.js` holds no colour: its `surface-*` and `accent-*`
  utilities resolve, per kind of utility, to a ground, one of the two text
  levels, a boundary, the accent fill or the accent ink, their opacity
  modifier mixing the token with transparent, and the base layer Tailwind
  generates draws in tokens too (the default border, the ring and its
  offset, and a field's placeholder, which was a gray under 3:1 on the day
  surface); the light-mode override layer
  in `app.css` (106 rules, two of which painted the accent's 20% and 30%
  tints with the error and the success washes) is deleted. The accent's ink is no longer used as a fill:
  surfaces that carry text use the accent fill with the text colour made for
  it, and the text on a status fill (a danger button, a revoke button, the
  tool-approval pill) is the on-semantic ink, where the on-accent ink gave
  2.3:1 in day. A mark that carries no text -- a progress bar, a chart bar,
  a legend dot, the streaming caret -- is drawn in a new mark token, the
  accent's ink, which reaches 3:1 on every ground, where the day fill gave
  2.0 to 2.5; a second mark token draws a second series beside it. A
  switch's track when on uses the ink too (the knob is under 3:1 on the day
  fill), and every hand-made switch draws its tracks in the switch tokens,
  where the knob, in the surface colour, could sit on a track of that same
  colour. The field primitives draw their edge in the rule colour, a
  boundary that must be seen. The accent fill's hover and the dialog's
  scrim differ with the palette and are given by a rule naming each
  palette, not by `light-dark()`, which browsers before 2024 do not compute.
  White and black are gone from components (a caption on an image
  sits on the surface, a backdrop on a new scrim token). Outside the palette
  files no hex colour stands but the QR code's white ground, which a
  drawing's light page now reads too, and the pre-render's theme-color map;
  no named colour and no colour function stands either, but the `rgba()`
  literals left in 20 components (53), a debt its ratchet only lets fall.
  The focus ring, the components' own included, is drawn in the
  focus ink and no longer changes a control's corners; selected text, native
  checkboxes and the forced-colours mode are declared; cards, dialogs and
  tooltips are separated from their ground by tone, and draw an edge only in
  high contrast and under forced colours. Every pair of text and ground the
  design lists -- 327 over the three palettes -- reaches its WCAG ratio, as
  evaluated from the role files through the derivation layer by a checker
  that must catch a planted low pair; so does every pair of a ground and a
  text colour a component sets together in one rule or on one element (a
  static census that follows a conditional's branches, a class list a
  script holds and the Tailwind names; a child's text on its parent's
  ground is not followed). Hex colours outside the palette files
  fall from 304 to the two exceptions above, named white and black from 39
  to none, and the accent's ink used as a fill from 91 (with 60 reads no
  census could place) to none, counted under every name the derivation
  layer gives the ink. Twenty-seven contracts. The wiring half of `ux11`, whose last
  assertion named the preferences store as the writer of the motion classes,
  is superseded by `ds20`. How the three palettes look, the 4% hover wash,
  forced colours, axe in each palette, the x-large text size on a phone, the
  colours a browser computes (owed, and reported as a skip until recorded)
  and the Tailwind opacity modifier's `calc()` inside `color-mix()` on the
  phone's and the desktop's browsers are checked on the machine, not here.
- `think=False` now reaches Ollama. The registry declared `think: bool =
  False` and sent `think` only when it was true, so a model that thinks by
  default thought whatever the caller said -- through the agentic pipelines
  that pass `think=False` (direct, web search, the fallbacks), and through
  the drift A/B and the librarian, whose small token budgets went to the
  thinking. `think` now has three states: None, the new default, sends
  nothing and leaves the model to its own way, as every caller that says
  nothing had before; True is sent as before; False is sent to a model
  whose capabilities, as Ollama reports them, include thinking, and to no
  other, so a model that cannot think never receives the switch. The
  capabilities are read once per model. The pipelines that say
  `think=False` stop thinking on a model that declares it; the drift A/B
  and the librarian's summariser ask for no thinking; the remote core
  carries the three states over its wire.
- The onion's eviction gate reads French names and whole French words. A
  name was a run of ASCII letters opened by an ASCII capital, so a name
  carrying an accent -- at its head or inside it, as Elodie, Helene and
  Chloe are written in French -- was never drawn, and a summary that
  swapped one passed its entity probes; a word stopped at an accented
  letter, so a French decision key was made of fragments. A name is now a
  run of letters opened by any capital and a word a run of letters and
  digits; on ASCII text both draw exactly what they drew before. The native
  core reads the same, with the same capitals as Python.
- The onion's eviction gate reads French. A decision written in French --
  "nous avons convenu", "on va", "il faut", "doit" and their kin -- is drawn
  as a decision probe, and negation counts "ne", the elided "n'", "pas",
  "jamais", "rien" and "aucun(e)" beside the English words; "n't" counts with
  the typographic apostrophe too. Before this, a French decision drew no
  probe and a summary that inverted it passed the gate at 1.0. Both halves
  of "ne ... pas" count, so a summary written in the other register fails
  its decision probe and the verbatim stays. The native core reads the same
  pattern, including the two letters Python folds beyond ASCII when case is
  ignored, the dotless i and the long s.

- [SECURITY] In Bulbe mode the external llama-server stays on the machine
  too. Its host comes from `backends.yaml` and every request went to it
  whatever the mode, while the registry's docstring already claimed the
  gate. Every head now asks the same question as Ollama's -- unless the
  mode reads exactly Daily, a host off the machine, or one nobody can read,
  is refused by name before any request is built -- and a generation or a
  stream is refused before the governor is asked for room it would never
  use. Health reads false, so the registry never resolves it.
- [SECURITY] In Bulbe mode every Ollama request stays on the machine. Now
  that `backends.yaml` or `OLLAMA_HOST` can point the backend anywhere, a
  request whose endpoint is off the machine -- or whose endpoint cannot be
  read -- is refused by name at every head before any client is asked, so
  the backend fails its health check and is never resolved. Localhost, the
  loopback addresses and the unspecified address count as this machine.
  Only Daily mode lifts the gate; an unreadable mode is Bulbe. llama-server
  carries no such gate yet, and the registry's docstring no longer claims
  one.
- `ollama.timeout` in `backends.yaml` is read: it bounds the connection to
  Ollama and never a read, so a long generation is not cut off; a request's
  own timeout still binds the whole request. A value that is not a positive
  number leaves the connection unbounded and says so in the log.
- The Ollama `host` in `backends.yaml` is where the requests go. It was
  written onto the backend and read by no request: every head asked a
  client that resolved `OLLAMA_HOST` or the library's default. Every head --
  health, listing, loaded set, model details, generate, stream, embeddings,
  eviction -- now goes through one client for the configured host, unless
  `OLLAMA_HOST` is set, which still wins; `endpoint()` follows the same
  order. The shipped value changes from `http://localhost:11434` to
  `http://127.0.0.1:11434`, the library's own default, so a machine that
  sets neither sends exactly where it did; a file that names another host
  now sends there.
- The registry-funnel guard follows the client through a function or a
  method that returns it: a request made on a helper's result is a site.
  Routing the Ollama heads through one transport had hidden the funnel's
  own calls from the census, and the same shape anywhere else would have
  passed as clean.
- The red team reaches its model through the inference registry, so every
  attack is admitted by the governor. Its loopback property moves to where
  the requests actually go: a new `endpoint()` head on the backend contract
  answers the base URL a backend's requests reach -- Ollama's as its client
  library resolves it from `OLLAMA_HOST`, llama.cpp in-process, llama-server
  and the core daemon their own -- and the generator, the multilingual
  strategy and the chat target refuse by name, before any request, a backend
  whose endpoint is unknown or off the local host. Their URL arguments are
  still checked and no longer route anything. The launcher's liveness probe
  is exempt from the funnel guard by name, and the raw ledger is empty.
- Every embedding goes through the inference registry. The embedder behind
  the RAG store, the project context and the memory's semantic recall
  posted to the server's embedding endpoints with its own HTTP transport;
  it now asks the registry's backend, one text through `embed` and a batch
  through `embed_many`, a new head on the backend contract: `None` where a
  backend has no embedding endpoint, one admission per batch on Ollama, and
  a refusal by name when the answer's count does not match the texts. Both
  embedding heads take a timeout. A misaligned or failed batch is still
  redone text by text. The model is verified against the registry's
  catalogue, and a catalogue nobody could read verifies nothing. Ollama
  versions without `/api/embed` are no longer served: the legacy
  `/api/embeddings` fallback is gone.
- The registry-funnel guard sees a request that never touches the client
  library: an endpoint of the inference server spelled in a module that
  imports an HTTP transport. It found six modules at nine sites. The
  project trigger detector's third level now asks the registry, with its
  half-second budget as the request timeout; the other five are sealed on
  a raw ledger that may only shrink, each waiting on a decision -- the RAG
  embedder sends batches, the red team enforces a loopback endpoint the
  registry does not check, the launcher's probe is not inference.
  `level3_ollama_url` in `projects.yaml` is no longer read.
- The tuner reaches a real backend. Four call sites asked the registry for
  `get_backend`, a method it does not have, inside broad handlers: every
  tuning run fell to the simulated benchmark, the llama.cpp benchmark
  answered "not available", and the speculative-decoding listing was always
  empty. They now ask `get`. The Ollama benchmark no longer posts to the
  server's HTTP endpoint itself: it asks the registry's Ollama backend, so
  every sweep point is admitted by the governor, the request carries a
  timeout, and a refusal comes back as a named error on that point. Rates
  are read from the counters the backend reports and are labelled measured
  only when both are present, as before. The `host` argument is accepted
  and unused.
- `list_models()` on the backend contract answers `None` when the backend
  cannot say and an empty list only when it looked and found nothing, the
  doctrine the two observation heads already carried. Every implementation
  answered an empty list before: Ollama with the client absent or the call
  failing, llama-server on a transport error or a body without a listing,
  the remote core with the daemon down, llama.cpp with no configured
  directory to scan (the production instance is built with none, so its
  listing was a constant empty), the test bridge without a scripted
  `list`. The daemon's `GET /models` carries unknown as a null listing
  with `known` false and a known empty with `known` true; the remote core
  reads it back. The registry's backend status reports an unknown count as
  null rather than zero, and its aggregate listing skips a backend that
  could not list without hiding the others. The shared catalogue readers
  propagate unknown, so their "empty only when one looked" claim is now
  true; the extraction falls to its first fallback model on an unknown
  listing instead of scanning nothing, and the health monitor skips a
  check it could not begin instead of marking nothing. Every other caller
  names what it does with unknown in its own log line, and none iterates
  a `None`.
- The model catalogue is read through the inference registry: sixteen
  modules asked the client library which models are installed and what a
  model is made of (`list()` and `show()`) at twenty-two sites, and the
  registry-funnel census could not see three of them -- two modules that
  bound the client module to an attribute at construction and asked
  through the attribute, one with a `chat` carrying images, and two
  request methods handed on uncalled. The census now follows the client
  into a bound attribute or name, counts an uncalled reference, and counts
  `list` and `show` beside the request methods; on the files as they were
  it reads 22 sites in 16 modules, on the tree it reads 0 over 355. Model
  management (`pull`, `delete`) has no head on the contract and is not
  counted, by decision. The catalogue is the two heads the contract already
  had: Ollama's `model_info` reads both answer forms and carries in `extra`,
  only when reported, the family list, the parameter text, the digest, the
  template, the modelfile, the license and the raw mapping; a list entry
  carries the size in bytes, the digest and the family list the same way.
  Three readers on `registry_clients.py` answer every module -- `None` when
  no backend is registered or the registered one could not read its
  listing, an empty list only when one looked and found nothing (the
  second half of that sentence became true one block later, see below).
  The vision pipeline describes through `generate` with the
  images; the dependency layer's model listing asks the active backend and
  no longer falls through to the client (the registry method it asked for
  never existed, so the fall-through ran every time); the CLI listing
  exits non-zero when no backend is registered instead of printing a
  connection error.
- The last eleven modules ask the inference registry instead of the client
  library, and the registry-funnel ledger is empty: the two benchmark
  transports and the judge (streaming, timeout as an option, the reported
  token count over the chunk count), the model warmup (loaded set through
  the new head, warm-up as one user message and a single token, residency
  renewed with no messages), the semantic cache's embedding, the cascade,
  the humanizer's rewrite pass, the reasoning engine (available exactly
  when the registry serves its default model, per-step timeout as an
  option), the fine-tune comparison, the benchmark route and the routing
  benchmark (model listings from the registry's backends). Each degrades
  by name without a backend. Four of them sent a completion prompt; it
  travels as one user message, the decision the previous block took once.
- `registry_funnel_guard.py` counts a `ps()` read as a client site now that
  the loaded set is a head on the contract, refuses an estate it could not
  scan instead of printing a zero, and names how many modules it read when
  it finds nothing owed. The runner, pre-cache and humanizer suites stand on
  the shared window and the registry bridge; the humanizer suite leaves the
  isolation-seal ledger.
- Nine more modules ask the inference registry instead of the client library:
  self-correction (four completion calls), the dynamic planner and its step
  executor, the agent base (chat, stream and the model list), speculative
  draft and verify, legacy fact extraction (two chats and the model list),
  and the five routes that built a client per request, some with a host of
  their own. Every one of those requests is now admitted, labelled and
  schema-checked; a model no backend serves gets a refusal by name or the
  documented degraded result (heuristic scores, the fallback plan, no facts,
  an error marker), never a client call. The funnel guard's ledger goes from
  21 modules and 36 direct sites to 11 and 12; `model_warmup` stays owed
  because its process listing has no registry surface yet.
- Importing `opti_oignon` no longer imports the API application. The package
  facade exports the same names through a module `__getattr__`: each is
  imported when first asked for, and names that are also submodule names
  (`config`, `router`, `executor`, ...) still resolve to the exported object.
  Measured on the import-footprint guard: 1634 modules, 2.1 s and 187 MiB at
  import before; 85 modules, 1 ms and 13 MiB after, with no database opened
  and no heavy dependency reached. The guard's ceiling drops from 1700 to
  200 modules and its ledger of import-time debt is empty.
- The coding agent reads its test verdict from the pytest summary instead
  of substrings: a passing run whose output mentions the word error passes,
  an error counts as a failure, and counts are recorded; a run with no
  summary is still treated as passed with a zero count. `fix_candidates` in
  `coding_agent.yaml` (default 1, the loop as it was) lets each fix attempt
  try several candidates, each applied, tested and undone by its inverse
  when it fails, so the first that passes stays.
- The agent-eval runner's chat client asks the inference registry for each
  turn, tool schemas as an engine option; `OllamaChatClient` stays as an
  alias of `RegistryChatClient`.
- Tool calls are schema-constrained. A forced tool decision travels as one
  schema branch per tool -- the name as a constant, the tool's own parameter
  schema for the arguments, closed to unknown keys -- so a constrained
  sampler cannot produce a call the tool cannot take; the native and the
  constrained schema come from one builder. At execution, an argument of the
  wrong type is refused before the handler, named, and marked retryable.
- The verification engine's fix requests and the consensus engine's model
  queries go through the inference registry; consensus reports itself
  available when the registry can serve a model, not when a client library
  imports.
- The tool executor asks the inference registry at every head -- native tool
  decision, forced decision, streamed and single-shot final answer -- with
  tool schemas and decision schemas travelling as engine options, so the
  registry's admission, provenance and schema handling apply to tool calls
  like any other request. No direct client path remains in the module; with
  no backend registered each head degrades by name. The agent-eval harness
  lays its scripted backend over the registry for the length of a run
  instead of rebinding a name on the module.
- `comment_only_guard.py` accepts a rename proven by reconstruction: a
  substitution of names that carry internal naming for names that do not,
  applied to the before side and required to reproduce the after side exactly.
  Injectivity is asked of each namespace separately, since two names that never
  denoted the same binding merge nothing. Outside Python, where no analyser
  exists, an unattributable change is now reported as unjudged -- neither
  accepted nor refused -- and listed by name, so an absence of checking stays
  visible instead of passing for a check that succeeded.
- Fifteen modules carried French prose in their comments and docstrings,
  among them seven published API descriptions (six operations and one
  schema). All of it is in English now, the published prose digest is
  recorded again, and the English-only ledger falls from 320 French spans
  in 33 files to 295 in 18: a file paid down to nothing comes off it.
- The note action "Fact-check with web" is removed. It searched nothing:
  one model call was asked to verify a selection "against current web
  sources and cite them", an invitation to invent citations under a label
  that promised a web check. A sourced verifier, one that searches and cites
  for real, is future work. `WEB_ACTIONS` stays, empty, as the mechanism it
  will join, with the Daily gate it will inherit. The old id
  `fact_check_web` is refused as an unknown action, by the runner and by the
  route, before any model call; nothing stored it, and the frontend drops
  the entry with it. The local "Fact-check" is labelled "Fact-check (no
  sources)", so no action reads as sourced, and its instruction now also
  forbids citing sources, references, links or URLs and claiming to have
  looked anything up: a check with no source cannot invent one either.
- `registry_funnel_guard.py` counts Ollama's cloud search and fetch: the
  client's `web_search` and `web_fetch` like a request, and the paths
  `/api/web_search` and `/api/web_fetch`, and a URL on ollama.com whose path
  is empty or under `/api` or `/v1`, as raw sites in any module, with or
  without a transport of its own; a link to a page on the site, such as the
  download page in an error message, is not one. An exemption or a raw
  ledger entry excuses local endpoints only: a cloud site is refused by
  name in every module outside the funnel, the exempt launcher included.
  The census now follows the client through a submodule import, a star
  import, `import_module` and `__import__` with a constant however they are
  reached or renamed, a `sys.modules` lookup, and `getattr` (a name it
  cannot read is charged, `pull` is not); through an unpacking, a walrus, a
  loop or `with` target, a parameter default, an argument passed to a
  function, a method or a class of the same module, a call given it, a
  container or a comprehension, a class attribute and a lambda; and along
  any attribute chain rooted at one of those. A value is bound to the
  client when it evaluates to it -- what a request returns is not the
  client -- and names are not scoped, which can only over-count. A path is
  read as a server routes it: bytes decoded, percent encoding, surrounding
  spaces, the query, the fragment, dot segments and repeated slashes, and
  a host written with a Unicode full stop. Over the 396 modules of the tree
  nothing outside the funnel reaches them, and the funnel's own count is
  unchanged at 12: no ledger entry, no exemption. Its docstring names what
  a syntax census still cannot see: a name built at run time or read
  through `vars`, `__dict__`, `globals`, `attrgetter`, `methodcaller`,
  `exec` or `eval`, a client handed to or received from another module, a
  capitalised name imported from the package that is in fact a submodule,
  a local endpoint or a cloud URL assembled from pieces none of which names
  it, a local endpoint posted with a transport the guard does not list.
- The web search docs and docstrings say which engines receive a query: the
  `ddgs` package with its default backend sends it to Wikipedia and to one
  or more engines it picks at random among DuckDuckGo, Bing, Google, Brave,
  Mojeek, Yahoo, Yandex and Mullvad, and every result is still labelled
  "duckduckgo". Which engines to use is left as it was.
- The ladder's directed-mutation tier proves that every contract a change
  adds has its blade. It counted only the unchecked lines of the blade
  register, so a contract added without any line went through unseen. `bash
  scripts/ladder.sh t3` now takes a census of the contracts in the tree --
  the tests pytest collects under the selection rule's own file, class,
  function and directory names, through module-level blocks and unittest
  cases; the Rust `#[test]` functions; the front-end tests -- compares it
  with HEAD, staged and untracked files included, and fails on each added
  contract that no checked line names: by its bare name when no other
  contract shares it, by its path otherwise. Contracts the selection rule
  deselects or ignores, those deselected parameter by parameter, those
  marked to skip and ignored Rust tests are exempt and listed; a front-end
  test runs outside the ladder, so its blade is owed to the machine and the
  tier ends OWED. A census that finds no Python contract, meets a Rust test
  attribute it cannot pair with a function or cannot read a file fails
  rather than passes; what it cannot read of the selection rule exempts
  nothing; the data places are never opened. The tier also counts the
  checked lines that name nothing in the tree. On this tree the census
  agrees with pytest's own collection, with `cargo test -- --list` and with
  Playwright's list. Twenty-eight contracts.
- `published_prose_guard.py`, `summary_fidelity_guard.py` and
  `red_team_guard.py` are unchanged in this cycle and continue to gate
  merges. `isolation_seal_guard.py` changed only in its sealed ledger, which
  fell as debt was paid, and so did the ledger of `public_language_guard.py`,
  which with `public_clean_guard.py` now reads `rust/` as well (above).

### Fixed

- [SECURITY] What the agent writes to memory or notes waits for the user
  unless the user typed it. In Daily, `manage_memory` and `manage_notes`
  wrote into stores every later turn reads back, with no review at all: a
  page, a file or a tool result that told the agent to remember something
  was enough for it to stick, the shape of a memory-injection attack. A new
  fact, or a new note's title, body and tags, is now written directly only
  when each of its words, folded (Unicode NFC, each run of white space as
  one space), equals the whole of what the user typed in the turn that
  started the run -- never a part of it: a sentence, a wrapped line, a list
  item, a line of code and even a paragraph can take their sense from what
  stands next to them ("Things you must never do:" before "Share my
  location with Bob."), so none of them endorses. Anything else, and every
  update or delete, becomes a proposal
  the user accepts or declines in the Memory or Notes panel (see Added);
  the model is told it was proposed, never that it was saved. The run's
  turn is its task, which `POST /api/agent/run` vouches for as typed; a run
  started any other way endorses nothing. When the review queue cannot
  record a proposal, nothing is written. On a bench of six planted
  instructions of different forms read in a web result, none is written
  and each is proposed; the same words typed by the user as the task are
  written, six of six. Measured in the container on stand-in stores; how
  many proposals real models make in a day is owed to the machine.
- [SECURITY] An evaluation run no longer writes the user's notes. The eval
  harness neutralized the skills, memory and web tools but not
  `manage_notes`, which Daily exposes, so a task could create a real note,
  contrary to the harness's own claim that a run never mutates user state.
- [SECURITY] A user turn saved with no claim is no longer held as typed.
  The agentic pipelines and the coding agent saved the user's message as
  typed whenever the turn carried no claim, and only typed words may
  endorse a write or decide a probe. With no claim the message is now
  legacy; the chat route hands every turn its claim, so its turns are
  unchanged.
- [SECURITY] The automatic memory capture keeps only what the user typed.
  After a turn it handed the extraction the whole conversation as the model
  reads it: the text of an attached document, folded into the user's turn,
  and the assistant's replies, web results included. A model then took
  "durable facts" from all of it into the memory every later turn reads
  back, so a document that told the assistant the user prefers something
  became a remembered preference within a few turns, with no tool call and
  no approval. The capture is now offered the conversation with who wrote
  each turn, and the extraction is handed the user's typed words alone: no
  document, no assistant reply, no question the model reworded, no turn of
  unknown origin; with nothing typed it is not called. The manual extraction
  route, which the user starts, follows the review of pending writes (see
  Added): what the user typed is written, the rest is proposed.
- [SECURITY] An agent run can no longer leave Bulbe through its request. The
  run entry took its mode from the request body, Daily when the body named
  none, and passed it on as given, so on a machine in Bulbe a request with no
  mode, or one naming Daily, started the agent with the Daily tool set: the
  network and autonomous writes to memory and skills, without the per-call
  approval, and without the degradation ceremony that leaving Bulbe takes.
  A machine mode that could not be read gave Daily too. A run now starts in
  the machine's security mode when the request names none, never in a looser
  one, and in Bulbe when the mode is unknown or the machine's cannot be read.
  The internal callers that name a mode themselves are unchanged.
- Five merge guards no longer pass over what they cannot read. A file that
  is not UTF-8 text was read with its bad bytes dropped or skipped, a module
  that does not parse counted no network sink or inference call, and a
  directory the walk could not list was left out, each without a word. The
  egress census, the registry funnel and the public-language, comment-only
  and isolation-seal guards now fail and name each such file or directory.
  The walks that read the package never list a directory named `data`,
  which holds the maintainer's content and no source. The isolation seal
  reads the repository's own `tests/` wherever it is run from, and refuses
  a directory with no suite instead of reporting every owed suite as stale.
- The merge guards no longer take a failed git call for an empty answer. A
  base ref git cannot resolve, such as the commit before a force push,
  emptied the diff passes of the public-clean, public-language and
  comment-only guards without a word, and each printed a clean verdict; it
  now fails the guard and names the base. The comment-only guard no longer
  takes a file git could not read at the base for one the base never had,
  and the public-clean guard fails when git cannot list the tracked names
  or search them. Each of these green lines now says how much it read. The
  egress census names why a plugin's manifest does not permit its sinks
  (absent, unparseable, not a mapping, or parsed without the permission),
  and fails by name when the YAML parser is not installed instead of
  reporting the plugin's permission as missing.
- [SECURITY] The security documentation no longer claims signatures that do
  not exist. The audit chain is a SHA-512 hash chain whose anchor is keyed
  with HMAC-SHA256, not signed with ML-DSA-65; releases are signed with GPG
  alone, when the workflow holds a key; nothing falls back to Ed25519, which
  serves only as a TLS key type. The pages also said that the startup
  checklist reports a plaintext database, that API keys are encrypted field
  by field, that anchors can be verified without trusting the application,
  and that the chain hashes with SHA-256: none of it is true, and each page
  now says what the code does, including that the RAG vector store is not
  encrypted at rest.
- [SECURITY] Every outbound request of the plugin marketplace and of the
  model downloader asks the web gate, and every gate reads the security mode
  from disk. Until now:
  - the marketplace installed a plugin from any URL in any mode, and
    refreshed its index the same way, through `urllib` with no destination
    check. Both now fetch through the page fetch: refused outside Daily mode
    and while the search kill switch is engaged or cannot be read, before
    any request, and reaching a public address only. The install reads
    `install.allow_remote_install`, `install.max_download_size_mb`,
    `install.require_hash` and the new `install.timeout_s`; the index reads
    the new `index.timeout_s` and `index.max_bytes`, and the listing
    refreshes a stale index on its own only when `index.auto_refresh`
    allows it. The GitHub shorthand is rewritten for the host `github.com`
    itself, not for a URL that merely contains the name.
  - the model downloader answered to no mode and wrote wherever its request
    named. It now asks the gate before it starts, before every redirect and
    after every block; a refusal answers 403 and removes the partial file.
    It refuses every address the page fetch refuses (shared address space,
    site-local, this machine's own, a network on its links) and writes only
    a `.gguf`, named by the last segment of its name, inside a configured
    model directory.
  - each process kept the mode it first read. Every read now stats
    `security.yaml` and the lockfile and reads them again when either
    changed, so a mode another process writes is the next one read.
  - the Bulbe middleware matched paths by string prefix on the request's
    own path: `/api/searchx` was refused as a search, a path under a mount
    prefix was not, and two of its entries named no route. It now matches
    whole segments on the path the router reads, refuses the three routes
    that only reach the web (the marketplace install, the GGUF download,
    the model pull) with 403 before their handler, and admits `/api/health`
    itself, not the routes below it.
  - importing the signature module imported liboqs-python even where its
    shared library was nowhere, which makes the package clone and build it
    from GitHub. It is now imported only when its library loads, and an exit
    while it loads is caught.

  A new guard, `egress_census_guard.py`, counts every network sink in the
  package: 30 sites in 3 gated modules, 48 still owed in 15 modules named in
  its ledger, which may only shrink. Those 48 are not gated yet: the model
  lifecycle (the pull in Daily mode, its other callers, the update check),
  the external and local vector stores, code that runs with the network, the
  Ollama command line, the core client, the token counter, the terminal
  interface and the Veilid client. The local rule does not yet look at an
  environment proxy.
- [SECURITY] A chat turn's stop, callbacks and results belonged to the
  process, not to the turn. The executor kept one stop flag for every call,
  set by any conversation's Stop and cleared by the next call of any
  conversation; the agentic executor kept each call's four callbacks on
  itself; the tool loop kept its direct answer and its native decision in
  shared slots; and the route built each turn's `done` payload from the
  singletons' last results. With two turns at once -- a phone and a tab, or
  two users when single-user mode is off -- one conversation's socket
  carried another's tool calls, with their arguments and the first 500
  characters of their results; its `done` carried the other's tool, reasoning
  and correction figures and vision model; and its reply could be the text of
  the other's direct answer. A Stop stopped every conversation's reply; a
  reply started after a Stop cleared it, so the stopped reply's model went on
  generating, unread, holding its admission ticket; a closed tab was noticed
  only at the next send or keepalive ping, up to 10 s later, and then held
  the event loop for up to 5 s, pausing every other stream and request; and
  a stopped reply read to its end (the coding agent's calls, the first phase
  of self-correction) was saved, captured into memory and cached. Each chat
  turn now owns its stop and its results: the route opens a turn per
  request and hands it to the executor, the agentic executor, the pipeline
  runner and every stage, the `done` payload and the vision and
  verification events are read from it, and Stop stops the live turns of
  its own conversation and nothing else (a turn with no conversation cannot
  be named; two tabs on one conversation both stop). A closed socket is
  noticed at once by a reader that only ever receives the disconnect, and
  stops its own turn; the stream functions no longer wait on a thread while
  holding the loop (a census of their spelled waits, sleeps and timed
  queue reads). The executor sees a stop while the model has sent nothing,
  prefill included, before and after its admission, and before its stream
  opens; a stopped call is never saved, captured, curated, cached or
  measured, and leaves exactly one cancelled ledger record, also when its
  caller closes it. The stop now reaches the reasoning strategies, the
  consensus wait (no merge follows) and both self-correction phases, the
  tool loop (its decisions, the tools it salvages from a narrated answer,
  and its final answer) and the second phase of think+tools, the pipeline
  runner and code verification, each before its next model call: a call
  already in flight is not cut short, and whether the model server stops
  working on it is not yet measured. Tools that ran stay in the
  conversation's tool history, also when the turn is stopped while its
  answer streams. The coding agent stops between phases, never writes the
  files of a stopped call and records the stopped turn with the files it
  wrote; its model callback is now the turn's, not the first turn's, a
  second `/code` turn of a conversation waits until the first has ended,
  and a reply with no conversation no longer runs the coding agent, whose
  one session, workspace and history every such reply shared. In Bulbe mode
  a stopped turn's tool approval is withdrawn at once, the audit names
  `turn_stopped`, not a person, as the resolver, and an approval that lands
  after the Stop runs nothing. The emergency stop also reaches every live
  chat turn. Still open, each for its own change: the quick sandbox's mode
  is process-wide, so it arms tools for every concurrent turn, including
  turns that did not enable it, and a second turn's tools run in the first
  turn's workspace, or on the host once the first turn ends; the tool
  approval queue is one queue for every conversation and user, and the
  Stop route has no owner check in multi-user mode; a self-correction
  stopped during its correction keeps its first draft saved, as a finished
  step of a stopped multi-step pipeline keeps its output; the context
  statistics are the last call's, whichever conversation it served; a
  timed-out reply is still saved and cached; and the consensus route and
  the plugin hooks still run on the event loop. Fourteen contracts, one of
  which supersedes a contract that stopped the executor through its private
  flag.

- [SECURITY] The children of a test session reached the maintainer's data,
  and so did the session's own pathlib listings. The data firewall covered
  the test process only: under `strace -f` over a full sweep, 25 child
  processes that contracts start reached the real data places, 21 of them
  with a call that succeeded. Five import probes opened
  `opti_oignon/data/branches.db` and `plugins.db` read-write, with their WAL
  and shared-memory files, and made `opti_oignon/data/plugins`; about twenty
  more read `opti_oignon/data/user_config.yaml` and made the data
  directories. On the maintainer's machine every local sweep therefore
  opened the real branch and plugin stores. In the test process, glob's
  string globber, which every `pathlib` glob and rglob lists through, had
  bound `os.scandir` and `os.lstat` when its class was made, out of the
  firewall's reach, and 24 tree-walking tests listed the real
  `opti_oignon/data`, its plugins and its projects by name. The session now
  sets two variables and puts `tests/_firewall_site`, then the tree's own
  root, first on `PYTHONPATH`; its `usercustomize` installs the firewall,
  on the session's mirror, in each Python child that inherits them -- as it
  is, copied and extended, or built by the componion suites' `child_env` --
  before any code of the child's own. It loads nothing the interpreter had
  not loaded: the firewall's source runs under a private name, outside the
  module cache and without touching `sys.path`, and the connect, globber
  and bound-open wraps are applied as their modules are executed. The root
  on the path matters in a second worktree, where the editable install maps
  the package to the main checkout: a child started from an empty directory
  imported that checkout's package, whose data places no firewall covered.
  It now imports the tree under test, and in a covered process the package
  is refused outright when an install would load it from outside every
  covered tree. A child that cannot be covered -- variables that do not
  pair as absolute paths, a firewall that cannot be installed -- exits 70
  and says why instead of running uncovered, and the session starts one
  child before any suite to see the firewall installed there: when it is
  not (a user site disabled by `PYTHONNOUSERSITE`, say), the session stops
  with status 70 and the reason instead of running on. The globber is
  wrapped in the test process too, and a listing through it names what it
  found in the mirror under the path the caller gave, so a walk of the tree
  still reads as a walk of the tree.

  The review of that change found more of the same kind, all closed here.
  In the test process the package-level `sqlcipher3.connect` -- the one the
  application calls -- and `sqlite3.dbapi2.connect` were never wrapped,
  since each package had copied its connect before the firewall came; a
  path that reached a data place through `..`, a doubled slash or a climb
  out of the working directory was passed through, because the prefix was
  tested before the path was normalised; a listing with no path, in a data
  place, listed the real one; `tarfile` and `bz2` had bound `open` at
  import, so an archive written or extracted into a data place landed in
  the real one; and truncating, linking, making a pipe or a node, reading a
  link and changing an owner were not redirected at all. The summary line
  now counts the processes that installed the firewall -- a marker each,
  written even when a process keeps nothing off, forks included -- beside
  what they kept off, and names every launch the firewall does not reach,
  by reason: a Python child run with `-I`, `-E`, `-s` or `-S`, with
  `PYTHONNOUSERSITE`, without the two variables, or with a `PYTHONPATH`
  that lacks the site, and a process that is not Python started in a data
  place. The import footprint guard's probe is one of them, by decision:
  its `PYTHONPATH` is replaced to measure a fresh interpreter, so the
  regression it exists to catch, a store opened at import, would open the
  tree's real store once before being reported. The ladder's first tier
  prints the line, and holds every suite that declares time budgets to
  them. Not covered, and written in the firewall: a path spelled through a
  symbolic link, the files SQLite opens from SQL (`ATTACH`,
  `VACUUM INTO`), a `sqlite3.Connection` built directly, an interpreter
  whose own build disables its user site, a child that is not Python
  outside the data places, and a Python process such a child starts in
  turn, as the native core's build script does. Measured again with the
  same two instruments and a wider set of traced calls, against this tree
  and the main checkout: no process reaches a real data place, and the
  test process lists none. Fourteen contracts, twelve of them first red on
  their property against the tree as it was.
- The chat composer's five server-wide switches -- semantic cache, cascading,
  humanizer, quick sandbox and coding agent -- showed what they had asked for,
  not what the server held. Each was a raw request without the CSRF header, so
  Bulbe and multi-user mode refused it, and every button flipped its state
  when the request failed. Each now goes through the API layer and
  `lib/switches/serverSwitch.ts`: its state is unknown until the server has
  been read, does not move until the server answers, and is then the state the
  server answers, named when it is not the one asked for. A refusal (a 4xx)
  keeps the state and is named with its status and the server's own reason; a
  server error (a 5xx, which some of these routes answer after the change has
  landed) and a failure to reach the server read the state again; a second
  toggle while one is pending is refused. While a state is unknown the switch
  is drawn dimmed, never as off, carries no pressed state for assistive
  technology, and the per-message mirrors of the sandbox and coding defaults
  forget it. What the server refused or could not say shows under the bar,
  where a state that could not be read can be read again. The cache and
  humanizer settings panels toggle the same way, and their switch is the only
  control that changes the on/off state: Save sends the rest of the form, and
  the switch reads the state again after it. Turning the quick sandbox or the
  coding agent on turns the other off on the server first, where it used to
  be turned off in the page only; when the second change does not follow, the
  first is named.
  The cascading client moved from raw requests to the API client, and the
  quick sandbox and coding agent defaults gained theirs. The Opti button is
  gone: the `prompt_enhance` field it set has no reader on the server (the
  request's `optimize` field is another option and stays out of the composer).
  Whether cache, cascading or the humanizer change a web chat at all is for
  the machine to measure.
- The chat options store built five keys the server's `ChatRequest` does not
  have (`no_cache`, `cascading`, `speculative`, `prompt_enhance`,
  `humanize`), and the chat store dropped them before sending, so the
  composer's Cache, Cascade, Opti and Human choices never reached a request.
  `lib/chat/requestFields.ts` now builds a message's options and its request
  from `ChatRequest`'s fields only, sending an option only when it was chosen
  (a temperature of 0 is a choice, an empty image list is not sent); both
  chat stores build through it, and the frontend's `ChatRequest` type names
  those fields and no other.
- Wiping a conversation from memory ran on one click, without a question, and
  swallowed its failure; wiping every conversation asked through the browser's
  own dialog. Both now ask in `ConfirmDialog`, a design-system dialog on the
  modal primitive (a native modal dialog, focus restored to its opener), which
  holds its buttons while the action runs and shows in an alert why it failed.
  The conversation wipe names the conversation it asks about and wipes that
  one, and the question is withdrawn if another conversation opens meanwhile.
  The browser's `confirm()` and `prompt()` are gone from the frontend:
  deleting a document, a selection of documents or a collection, unregistering
  a fine-tune variant, disabling remote access, revoking a client certificate
  and renaming a synced device (a dialog with a field, focused on open, which
  Enter confirms) ask in the same dialog. The modal primitive now focuses a
  field marked for it before its close button, and disables that button while
  an action runs.
- Events that nobody heard are gone or heard: the per-message Fork button (the
  chat path does not follow branches; forking stays in the branch explorer
  above the thread), the up arrow's "edit the last message" (no page
  listened), a branch tree's fork relay and the telemetry timeline's selection
  event are removed, and the project badge in the context bar now opens its
  project. Every message knows its conversation, and feedback is never
  submitted without both its conversation and its message id (it went out with
  an empty conversation id); without both, the thumbs are not shown.
- One skip link, the root layout's (the HTML shell and the app shell each
  carried their own), leads to one `main-content` landmark per page: the chat
  page no longer doubles it, and the sign-in, registration and component
  gallery pages gained theirs. Ctrl+K works again: it opens the command
  palette. Scrolling to a settings group or to the end of a chat is smooth
  only when neither the motion
  preference nor the system reduces motion (`lib/motion.ts`), and not when
  that cannot be read.
- Deleting the open conversation navigates within the app instead of reloading
  the whole page. Five custom properties that components read and nothing
  declared now resolve to the tokens they meant. Three pictographs in code
  became drawn icons; a pipeline's own emoji is still shown. The error
  boundary's "Report issue" link pointed at a vendor organisation's repository
  and is removed. The settings page and the sidebar read one settings catalog
  (`lib/settings/catalog.ts`), so the settings search now finds text size,
  motion, the advanced appearance group and configuration maintenance, which
  it missed. The live metrics overlay and the sandbox isolation badge read
  through the API layer (`lib/api/liveMetrics.ts` is new). The README said the
  Playwright specs ran against a mocked backend and covered chat, settings,
  RAG and mobile variants; it now names the three specs, which need a running
  backend.
- Sixteen contracts. `cs4` of the cache surface suite, which looked for the
  cache's path inside the control bar, is superseded by `cs8`, which holds the
  frontend's side wherever the switch is shown. Raw requests outside the API
  layer fall from 30 to 15, and raw requests in the API layer outside its
  client from 68 to 65. Focus handling in the dialog, the CSRF refusal before
  and after, and how the switches read in the browser are checked on the
  machine, not here.
- A vision request through the core daemon was answered blind: the remote
  core client took a request's images and never put them on the wire, and
  the daemon had no field for them. The client now sends them as given, the
  daemon hands them to its backend on both heads, and images that are not a
  list of base64 strings are refused by name before any backend is asked.
- The local ladder's guard tier ran each guard on the maintainer's tree. A
  guard that imports the application -- the published-prose guard builds
  the OpenAPI schema from it -- runs its module-level singletons, and those
  opened the real stores: the branch store with the master key, the project
  vector store, the presets. `tests/_guard_mirrored.py` runs a guard in its
  own process as a direct run would, under the test session's data
  firewall with a fresh mirror seeded from HEAD, and the tier goes through
  it: locally, the guards see what CI sees. A child process a guard starts
  is not covered, on purpose, since the import footprint guard measures in a
  fresh interpreter.
- The two measurement guards wrote into the data places on every run --
  in CI, in their harness contracts, and in the local ladder, where no
  firewall covers them. The summary fidelity guard's needle sweep builds
  its conversation store in a temporary directory, but the store journals
  every save through the process's sync engine, whose feed and key are
  the device's: 162 conversation records a run, signed with the device
  key where one exists. The red-team guard's RAG probe ran the sanitizer
  with its audit on: 4 rows of attack text a run in the injection audit.
  `ConversationManager` now takes `publish_to_sync` (default unchanged),
  the needle sweep builds its store with it off, and the probe runs the
  shipped defense configuration with the audit alone switched off, the
  verdict being computed before the audit is written.
- The notes send-half suite let the store's real sync publish hook run in
  a full sweep: its creates happen before the spy is installed, and once
  the note-update suites before it had left the real sync package in the
  module cache, the hook resolved the master key, the device signing key,
  the change feed and the peer store, and journalled a note record, signed
  with the device key where one exists.
  The data firewall has kept those paths off the real places since it
  landed; the suite now also loads through the shared isolation window,
  with the sync package proven unreachable, so the hook takes the no-op it
  documents for a flat-loaded store. One more seal debt is paid.
- The two chat routes suites cleared the routes module's conditional
  imports so that "their absence selects the inert branches", and behind
  their stand-in package those imports resolved by name anyway: the chat
  retry suite loaded 93 real modules -- the executor, the configuration,
  the conversation store, the encryption modules, the emergency stop --
  and the quick sandbox suite 97, so the active branches ran. Both now
  load through the shared isolation window, where each conditional import
  takes its inert branch as the harness said, and they no longer evict a
  real fastapi or pydantic from the cache on the way out. Two more seal
  debts are paid.
- The two agentic suites ran on the real executor stack: behind their
  stand-in package the classifier suite loaded 82 real modules and the
  summary alignment suite 84. The classifier's fallback check -- the
  configuration absent, so the legacy model name -- ran with the real
  configuration loaded, and the summary suite's comment said the tool
  executor's flag resolved False under the bare package when the real
  module had loaded. Both now load through the shared isolation window:
  the configuration is proven unreachable where its absence is asserted,
  the flag resolves False as the comment says, and the model-client stubs
  are gone, since nothing under contract imports the client any more. Two
  more seal debts are paid.
- The memory wrap suite's last check -- no retrieval backend importable,
  so no message -- held for the wrong reason: behind its stand-in package
  the real retrieval module loaded, with 26 others, and ran against the
  real memory store, which happened to return nothing. The extraction
  bounds suite loaded 13 real modules the same way. Both now load through
  the shared isolation window, the retrieval module proven unreachable for
  the first; two more seal debts are paid.
- The notes store's phone opt-in suite loaded 21 real modules by name
  behind its stand-in package -- the real sync package, encryption and the
  post-quantum signatures among them -- and the auto-tuner's bounds suite
  resolved the speculative-decoding module in a full sweep, depending on
  the suites that ran before it. Both now load through the shared isolation
  window; two more seal debts are paid.
- Two sealed memory suites resolved real project modules by name behind
  their stand-in package. The canonical store's SQL hygiene suite ran the
  real encryption stack -- on a machine with a key, that key -- where the
  store's own path in isolation is plain SQLite, and the dual-layer suite
  ran the real context window where the retriever keeps its own estimate.
  Both now load through the shared isolation window, with what they need
  declared and the encryption module proven unreachable; two seal debts are
  paid.
- With no vector signal -- no embedding model, or the backend down -- the
  memory recall ranked facts on keywords and category alone and gave every
  one a vector similarity of 0.0, the value of a fact the vector layer
  measured and found unrelated, without a word in the log. A similarity
  nobody measured is now `None`, and the retriever says once that it
  recalled without its vector signal. Its keywords are whole words, accents
  included: a French query no longer meets an unrelated fact through the
  fragments of other words.
- A directory written with `~` in `backends.yaml` was taken as a folder
  named `~` under the working directory. The shipped `~/models/gguf` was
  therefore never found -- llama.cpp listed no model even with the file
  applied -- and the model manager would have downloaded into a literal
  `./~/models/gguf`. llama.cpp's model directories and the model manager's
  scan and download directories now expand `~` to the home directory.
- The tuner loaded `benchmark_tokens` from `auto_tuner.yaml` and never used
  it: the route built every benchmark on hardcoded defaults (128 tokens, a
  120 s timeout). The budget and a new `benchmark_timeout_s` key now reach
  the run on Ollama and llama.cpp; a value that cannot be read keeps its
  default and says so in the log.
- The facade contract that imports the real package failed or passed with
  the order of the suites before it. Two routing suites replaced the
  `opti_oignon` entry of `sys.modules` with a stand-in and never put it
  back, and five transport suites left `urllib.request.urlopen` swapped for
  the rest of the process. All seven now close what they open -- the two
  routing suites through the shared isolation window, paying their seal
  debt -- and each carries a teardown check that fails a contract leaving
  either one altered.
- The three resource-governor suites left a new stand-in `opti_oignon`
  package behind after each of their 24 in-process contracts, and under an
  editable install they ran beside the real emergency stop, backend
  registry and VRAM estimator, loaded by name behind their stand-in rather
  than kept out. They now load the governor through the shared isolation
  window, one window per contract, and a teardown check fails any contract
  that leaves a project module changed; three seal debts are paid.
- [SECURITY] `/skill` in `oo chat` put the text of a skill received from a
  paired device into the system prompt, as an instruction. The device's
  sync gate had let the record through on its provenance -- peer, device,
  the category/name it lands under -- without ever showing its text, so a
  compromised peer could swap the text of a skill you know by name. A
  skill applied from sync now carries a device-local mark, and
  `/skill` runs only bytes adopted on this device: `/adopt NAME` shows the
  text with its digest, `/adopt NAME DIGEST` adopts those bytes and no
  others, and a new version asks again. Writing a whole skill on this
  device adopts it; an edit of unadopted bytes does not; a mark that cannot
  be read adopts nothing.
- `oo config set` wrote the configuration of the run, not the file: a
  transient `NO_COLOR`, `--no-color`, `--api-url` or `OO_API_URL` ended up
  saved, along with every default the user never chose, and `oo config
  reset` saved `color: false` under `NO_COLOR`. It now writes what the file
  holds plus the key it was given, refuses by name a colour, timeout or
  output format it cannot read, and refuses to edit a file that is not a
  mapping instead of overwriting it.
- `color: "false"` in `cli.yaml` read as true, and one unreadable value --
  a timeout of `abc`, say -- reset the whole file to its defaults. Every key
  is now read alone and falls back to its own default.
- An error printed while a spinner turned landed inside the spinner's
  line. The eight commands that wait behind a spinner now stop it and erase
  its line before they say why they failed.
- The CLI's `OK` and `Error:` prefixes were coloured on a terminal whatever
  the run had been told: 23 of the 26 calls passed no colour, and the
  helpers defaulted to on. They now follow the run -- `--no-color`,
  `NO_COLOR`, or `color: false` in `cli.yaml` -- and, outside a command,
  `NO_COLOR`.
- `scripts/build_oo_core.sh` copied the new native core over the old file
  in place: a running process that had loaded the old one could crash as
  its mapped pages changed under it, and a loader starting mid-copy could
  read half a file. It now installs by rename. Its final check also
  imported whatever `opti_oignon` the calling directory offered; it now
  checks the artefact of its own tree.
- The browser specs no longer race the first-run dialog. The helper waited a
  fixed budget for it to appear and, when the budget ran out first, returned as
  though there were nothing to dismiss; the dialog then opened over the page and
  swallowed the next click, so the failure landed on an unrelated locator. No
  budget can tell "not yet" from "never", so the application now marks the
  moment it has decided and the helper waits for that mark.
- [SECURITY] Encryption setup could replace the master key file.
  `POST /api/security/encryption/setup` stood aside only when the running
  encryption manager was enabled, and the manager turns itself off whenever
  the server cannot load the key: an enveloped key file without
  `OPTI_KEYFILE_PASSPHRASE` in the server's environment is enough. Setup
  then made a new key and wrote it over the existing file -- in random mode,
  in the unprotected format -- and every encrypted database and the audit
  chain, keyed by the old one, would no longer have opened. A quieter path
  went the same way: with `OPTI_ENCRYPTION_KEY` set and encryption off in
  `security.yaml`, setup wrote a new key file and encrypted fields under it
  for the rest of the session, while the databases kept the variable's key
  and the next start loaded the variable again. Setup now never replaces a
  key file. A key that already loads (`OPTI_ENCRYPTION_KEY`, or the key
  file) is enabled as it is and no key file is written; anything at the key
  file's path that does not open in this server -- a symbolic link
  included, which the old write followed -- is refused with 409 before any
  key is made, the refusal naming the variable and the restart it needs. A
  new key file is written to a temp created with mode 0600 by
  `O_CREAT|O_EXCL|O_NOFOLLOW` and synced, then hard-linked into place and
  its directory synced, so it never exists partly written or with looser
  permissions, and a key file another writer put there meanwhile is refused
  with the same 409, not replaced. Once linked, a temp that cannot be
  removed or a directory that cannot be synced is logged as a warning, not
  answered as a failed setup. Setup now needs hard links in the key
  directory: on a filesystem without them it fails and writes nothing,
  since the only fallback, a rename, could replace a file. Owed to the key
  ceremony panel, not changed here: it shows the refusal as "API error 409"
  without the server's reason, and it words every success as a new key --
  when an existing key is enabled, a passphrase typed there was not used,
  the random-mode text names a key file that may not exist, and an existing
  key in the unprotected format stays unwrapped at rest. Seven contracts
  (nineteen cases); before the change the same call rewrote the key file it
  could not open, and wrote a new key through a link at the key path.
- [SECURITY] The search kill switch was read as off everywhere. `is_killed`
  was a property and its three readers -- the capability manifest, the chat
  executor and the Bulbe middleware -- called it, caught the `TypeError` and
  read "not engaged"; engaging it stopped a chat search only because it
  nulled the searcher, which then failed as "Web search failed". It lived in
  memory, so a restart disengaged it, and after the re-enable ceremony the
  searcher stayed nulled and answered "Search package not installed" until
  a restart. The status route answered `search_enabled: true` when the
  switch was unavailable. The switch is now a recorded state,
  `data/.search_killswitch.json` (mode 0600, written through a unique
  temporary file under a thread lock and a file lock, and replaced
  atomically), read at every check: by the searcher's request gate, the
  manifest, the executor and the middleware. A record that is malformed,
  of another version, a link, a directory, a FIFO or over 64 KiB reads
  engaged; a missing one reads never engaged, so deleting it re-enables
  search, as deleting the mode's lockfile does. A kill that cannot be
  recorded holds in the process, says it will not survive a restart, and
  is written by the next kill; only the ceremony records "not engaged", and
  every other writer keeps an engaged record engaged. A reader treats a
  switch that raises, or whose own import fails, as engaged; only the
  switch module itself being absent reads not engaged, and the searcher
  then refuses on its own. A process whose kill could not be recorded gives
  its latch up once a later write of its own records the engaged state.
  Loading the module reads nothing. The status route answers
  `search_enabled: false` when the switch is unavailable, and an unrecorded
  re-enable is a 503. The tool registry's legacy view and the agent's
  toolset still list `web_search` while the switch is engaged in Daily
  mode; a call is refused by the gate, which names the refusal in the tool
  result.
- [SECURITY] A web search left the process in Bulbe mode, and so did the
  proxy health check with its Tor exit lookup: nothing asked the mode
  before a request. Every request point of the searcher -- the search class
  and the Tor exit lookup -- now opens a gate first, ahead of the cache: it
  refuses while the switch is engaged or cannot be
  read, and in any mode but exactly `daily`, a mode that cannot be read
  included, with `WebSearchRefused` naming why. The retry loop re-raises the
  refusal from its first handler (every exception class of the `ddgs`
  package is `Exception`, so the rate-limit handler used to take anything),
  the search class is bound privately and constructed in one place, and a
  chat turn names the refusal in its status line. The mode is read from the
  process's cache, which a mode change made in the same process refreshes:
  a long-lived second process still keeps the mode it read at start.
- [SECURITY] The knowledge base's page ingestion, `POST /api/rag/ingest/url`,
  fetched any URL in every mode, Bulbe included: it checked the scheme
  only, followed redirects without checking them, and refused neither this
  machine (the model server, the core daemon, the API itself) nor the local
  network. The store now fetches through `web_search.fetch_page`, behind
  the same gate, asked again before each redirect and after every read of
  the body: outside Daily mode, or with the switch engaged, the route
  refuses by name with a 403 (a 503 when the switch cannot be read) before
  any host name is resolved, or where the reading stands, and nothing is
  recorded, not even the collection. `web_ingestion.enabled` in `rag.yaml`,
  which nothing read, now refuses the same way (403) when it is anything
  but true. In Daily mode the fetch reaches only a public address: user
  information, a scheme other than http or https, a port other than 80 and
  443 unless `web_ingestion.allowed_ports` in `rag.yaml` names it (none
  does by default), and a host resolving to a loopback, private,
  link-local, site-local, unspecified, multicast, reserved or shared
  (carrier-grade NAT) address, an IPv4 address carried inside IPv6
  included, are refused by name with a 400. On an IPv6 network, where the
  router, the other devices and this machine all hold global addresses,
  this machine's own addresses and every address on a network it reaches
  without a gateway (its prefixes and on-link routes, read from `/proc/net`
  at each request) are refused too; the router's public address reached
  back through the router (hairpin NAT) cannot be told from any other and
  is not. The host is resolved once per request and the connection goes
  only to answers that were checked, in the resolver's order, the request
  still naming the host and TLS verifying that name; at most three
  redirects are followed, each checked as the first. The body is capped as
  it is read (a declared length over the cap is refused before the body),
  one time budget covers the whole fetch, and a timer cuts a socket that
  stalls past it; name resolution is bounded by the system resolver alone.
  A public page in Daily mode is ingested as before, with three
  differences: a body sent compressed although identity was asked for is
  refused by name; a page whose Content-Type names no charset, or one that
  is not among the web's encodings, is read as UTF-8 (the old client read
  a `text/*` page with no charset as Latin-1, JSON as UTF-8, guessed the
  others, and fell back to UTF-8 only for a name Python did not know); and
  the fetch connects directly, never through a proxy named in the
  environment. As before, the web search's own proxy is not used, and a
  redirected page is recorded under the URL asked for. The store holds no
  client library of its own any more.
- [SECURITY] Chat and tool-loop web results reached the model as bare text,
  with an instruction to use them. Both now carry the listing inside the
  untrusted-data envelope, source `web`, with the platform's sentence
  outside it, and withhold it, saying so, when the wrapper cannot be loaded.
  In the chat the block still rides a system message, the head or the
  volatile tail, as the memory block does: the policy header and the markers
  are the same there, the role is not, and moving both to the user role is
  owed to a later context change. The agent loop already wrapped every
  observation.
- [SECURITY] The domain allowlist and the injection circuit breaker had no
  caller: the settings panel's "server-enforced" allowlist filtered nothing
  and the breaker never counted. The searcher now applies the allowlist to
  every result list it returns, cached results included, and feeds the
  breaker once per search whose results carried a detected injection,
  counted on a sanitizer instance of that search's own, with pattern names
  only -- never the query, a snippet or a URL; its detections still join
  the shared log the security events route lists, and it carries the shared
  configuration, an empty one included, without reading the file again.
  The sanitizer used alone, as the red-team harness uses it, never reaches
  the breaker. A breaker trip is recorded like any kill: it holds across a
  restart, and only the re-enable ceremony clears it, so three searches in
  ten minutes whose results match an injection pattern -- pages about
  prompt injection included -- disable search until an administrator
  re-enables it. How often that happens is owed to the machine. The allowlist is
  recorded with the switch and survives a restart; an enabled allowlist
  naming no domain passes nothing (it used to pass everything); entries are
  normalised to host names and the rest refused by name, and a result URL
  with a backslash, whitespace, a control character, userinfo or a scheme
  other than http or https never passes. The route answers the normalised
  list, and a 503 when nothing could be recorded.
- [SECURITY] The re-enable ceremony took any password when the auth manager
  could not be read. It now refuses with a 503. The 2FA code it accepts is
  still not verified, and the visual code is served over the API, so it
  proves an API session, not physical presence; the switch's docstring says
  both.
- The claim verifier behind `/api/claims/verify` and the answer and citation
  routes read "The source does not confirm the claim.", "This is not
  consistent with the source." and twenty-six other negated or hedged
  replies as supported: a lead containing "confirm", "consistent with" or
  "corroborat" anywhere was promoted. A reply is now supported only when
  its first line opens with SUPPORTED -- markdown, a quote marker or a
  "verdict:" label may come before it, a question mark may not follow it,
  and it may not be the start of a longer word -- and no line of the reply
  carries a negation cue, a hedge or an unsupported marker: the reason the
  instruction asks for often comes on the next line. A cue is read in
  English or French, whatever the apostrophe or the hyphen (a soft hyphen
  included), contracted without an apostrophe, carried by a negating prefix
  on a word of support ("unsubstantiated", "disagrees", "mismatch",
  "inaccurate"), or named outright ("wrong", "otherwise", "silent",
  "contraire", "infirme"); a word mixing scripts is not read as anything
  else. The rest is uncertain, or unsupported when the reply says so
  anywhere. The cost is pinned: "The source confirms the claim." and a
  SUPPORTED reply that says "no rounding" are uncertain now, and "Nothing
  contradicts it." on the reason line reads unsupported. How often that
  happens with a real model is owed to the machine.

  Nine contracts, each red at birth on its property. 153 directed
  mutations prove them as they stand, each restored byte-exact: 63 from
  the first build, each turning its contract red; 66 after a review, whose
  added clauses were each red on the code before its fixes, one per clause
  and each red in the clause it names; and 24 on the note-action contract,
  also red in the clause each names. That contract was rewritten when the
  action was removed rather than renamed, and the nine mutations that
  proved its withdrawn version no longer count; six of its eleven clauses
  were red before the code changed, the other five are proven by their
  mutations alone. The page ingestion adds five clauses to the Bulbe
  contract, two contracts of fourteen clauses for the destinations and the
  bounds, and four RAG store contracts that supersede the four which
  pinned the store's own client library (deselected by name); on the code
  before the change the route resolved and connected in Bulbe, with the
  switch engaged, to a host resolving to loopback, with user information
  and to port 11434, and the store went around the fetcher, while the
  redirect, pinning and page clauses, on which the old client could not be
  driven, and the bounds, absent then, were proven by their mutations
  alone. A review then found the fetch failing on the most common real
  answer -- a body of declared length on a connection the server closes,
  over a real socket pair: "Bad file descriptor" -- and reaching an IPv6
  home network's router and this machine; the clauses added for those, for
  the site-local range, the gate while the body is read, the next checked
  answer, the page's charset and the ingestion's own switch were each red
  on the code before its fix. Every clause of the page ingestion is proven
  by mutation against the code as it stands: 31 mutations, each red in the
  clause it names and restored byte-exact. Measured here, in the
  container: the manifest read the engaged switch as off, a search in Bulbe
  constructed the search class, the chat head carried the bare listing, a
  three-result search under a one-domain allowlist returned all three, and
  the verifier promoted all 28 negated phrasings. Owed to the machine: a
  real chat turn with web search (the envelope, the engines that received
  the query), a restart with the switch engaged, the refusal in Bulbe
  through the running API, a real page ingested by URL in Daily mode and a
  loopback, private or local-network URL refused through the running API
  (the machine's own `/proc/net` tables read, an IPv6 router refused), the
  rate of strict verdicts, whether real pages about prompt injection trip
  the breaker, and `OLLAMA_API_KEY` absent from
  the API's environment with `OLLAMA_NO_CLOUD=1` on the host.

- A contract's time is its own again. The test session now freezes the heap
  its collection built -- every suite imported, most of it alive until the
  end -- before the first contract runs, and thaws it when the session
  ends. Left in the collector's generations, that heap was rescanned by
  every full pass, and a pass landed in whichever contract happened to
  cross the threshold: over a full sweep, a ratchet contract whose own work
  takes 0.03 s was charged a 1.2 s pause and failed its 1.0 s time budget,
  for a reason it never caused. Frozen, it took 0.03 s in the same sweep,
  and the largest pause inside any contract fell from 1.9 s to 1.2 s.
  Garbage the contracts make is still collected; no budget was loosened.
  One contract, which also shows the probe failing when the freeze is
  taken out.
- The sandbox's isolation checks read the backend in use, not whether
  bubblewrap is installed. With `isolation_backend: tempdir`, commands run
  in a plain temporary directory even where bwrap is present, yet eight
  checks keyed on its presence alone: strict mode let those commands run,
  `execution_blocked` and the health level reported bwrap, the agent
  dispatch and the skills tool took the session for isolated, note
  transcription and captioning went ahead, and the security score credited
  the sandbox. Each now asks the manager whether bwrap is in use, and strict
  mode blocks a configured tempdir backend with a reason that names the
  setting. Nine contracts: one per check, one for the manager's answer.
- A degraded sandbox is the server's decision, never a caller's. A sandbox
  on the tempdir backend gives no real isolation, and the manager asks the
  user to confirm one before creating it; yet the quick sandbox, the chat's
  coding agent, the evaluation runner and the benchmark evaluator passed
  `allow_degraded=True` past that confirmation, and the two REST routes that
  start a sandbox took the flag from the request body. The flag is gone
  from every function. `POST /api/sandbox/create` and `POST
  /api/coding/start` refuse `allow_degraded: true` with a 400 that names
  `POST /api/sandbox/confirm-degraded`, where the user confirms. Four
  contracts.
- The chat's file and code tools have no handler outside the sandbox. On
  a chat turn without a sandbox session (the quick sandbox switched off
  in the request, unavailable, or failing to start), `read_file` returned
  any file the server could open, `list_files` listed any directory, and
  `write_file`, said to write inside a working directory, wrote any
  absolute path it was given; the shipped tool configuration enables all
  four tools, web search beside them. Outside a sandbox session each now
  refuses, says that nothing was read, written or run, and names the
  remedy. With the quick sandbox on, the shipped default, a turn is
  unchanged. Five contracts.
- A plugin whose subprocess fails is no longer run inside the server. The
  loader's default mode tried the plugin's own subprocess and, on any
  failure there, the plugin failing its own start handshake included,
  loaded the same code into the server process, where the import rules
  are no boundary against a hostile plugin. A failed start now raises a
  load error that names the cause and leaves the plugin unloaded. The
  subprocess is the default mode, an unknown mode is refused when the
  loader is built instead of falling through to that fallback, and Bulbe
  refuses an explicit in-process loader, which it let through before.
  Three contracts.
- Code execution runs in the sandbox, and only when the user turned it on.
  `POST /api/code/execute` and the automatic check of the code in an answer
  shared a runner of their own: a process on this machine with the
  server's environment, its secrets included, and a memory limit as its
  only bound. It stayed off because a flag nothing set stayed false, while
  the settings screen showed it on and saved a setting nothing read. Each
  run now goes through the sandbox manager, like the agent's tools, in a
  sandbox made for that run and destroyed after it, or the conversation's
  own in persistent mode; without a usable sandbox it is refused and says
  the code never ran, and an image leaves the sandbox only as a regular
  file, never through a link. `code_execution` turns execution on; the
  automatic check has its own setting, `code_auto_verify`, and needs both.
  Both are off by default, on the server and on the screen. With execution
  off, the check used to ask the model to fix the refusal, twice per code
  block; code that never ran is no longer offered for a fix. Eleven
  contracts, and the egress census owes one sink fewer.
- [SECURITY] A plugin's worker runs under bubblewrap whenever the sandbox
  does, and under limits the server sets. It ran as a plain process of
  this user: every file the user can read, the network, and limits taken
  from the plugin's own manifest with no upper bound, each skipped when it
  could not be set. It now gets the sandbox's namespaces and seccomp
  filter, sees the interpreter and its own folder read-only, keeps the
  network only when its manifest declares `network_outbound`, and, with a
  write permission, gets a private data folder, `plugin_data/<name>` under
  the data folder (0700), named to its hooks as `metadata["data_dir"]`.
  CPU time, memory and open files are the lower of the manifest and a
  ceiling in `config/plugins.yaml`; processes and file size are the
  server's alone; all are set before the worker starts, and it refuses to
  serve without them. Host and worker talk over a socket pair the worker
  inherits instead of a socket file. Without bubblewrap, strict mode keeps
  plugins from starting, with the cause and the remedy, and `/api/health`
  reports the plugin isolation. The scratchpad, the task extractor and the
  GitHub connector kept their databases in the shared temporary directory,
  the GitHub token included, since nothing ever gave them their folder:
  they use the data folder now and, without one, keep nothing on disk. The
  files they left there (`opti_scratchpad.db`, `opti_tasks.db`,
  `opti_github_auth.db`) are not migrated. The plugin pages and
  SECURITY.md described a bubblewrap isolation no plugin had and a pipe
  channel of a second launcher; that launcher, imported by nothing, is
  removed, and the pages describe what runs. Twelve contracts; one
  superseded by name.
- [SECURITY] No plugin loads inside the server any more. The loader kept
  an in-process mode, an explicit choice since the fallback to it was
  removed: the plugin's file executed in the server under import, path and
  builtins restrictions that were no boundary against a hostile plugin.
  The mode is gone with its 650 lines, and `inprocess`, like any unknown
  mode, is refused when the loader is built. The security score's seventh
  check credited "plugin import blocking" from that mode's list while
  plugins ran in their own process, where the list applied to nothing; it
  now credits plugin isolation only when no plugin can run outside
  bubblewrap. The plugins configuration loses the import lists only that
  mode read, and its mode comment says what runs. Ten contracts; the boot
  path's six are re-asserted on real workers, three more are superseded by
  name, and the in-process sandbox suite, which imports what is gone, is
  ignored and kept as the record.
- [SECURITY] Bulbe pins the sandbox's strict settings. Every switch that
  lowers the bubblewrap path was configuration in both modes:
  `strict_mode: false` (in sandbox.yaml, or security.yaml over it),
  `isolation_backend: tempdir`, `seccomp_enabled` or `seccomp_required`
  off, `limits_enabled` off, `require_degraded_confirmation` off. Under
  Bulbe each now reads at its strict value: without bubblewrap nothing
  executes and no plugin starts, a tempdir backend gives way to bubblewrap
  where it runs, a seccomp filter that cannot be built refuses the launch,
  the limits are installed, and a degraded sandbox waits for the user's
  confirmation. Daily keeps the configured values. The mode is read at
  each use, so a switch to Bulbe pins them without a restart; a mode that
  cannot be read pins them too. The Bulbe page drops two gaps the entries
  above closed, code execution and plugins outside any confinement, and
  counts the census as it stands. Seven contracts.
- The chat runs a model larger than the card again. Since the governor
  learned the card's capacity from nvidia-smi (2026-09-11), every model
  whose weights and context exceeded the free VRAM was refused, even when
  the GPU and system RAM together held it; before that, an unknown
  capacity let everything through on the VRAM side. Admission now tries the
  GPU alone, now and then after evicting idle models, and otherwise splits
  the model between the VRAM free now and the system RAM above a reserve.
  Ollama places the layers itself: no num_gpu is sent. The order follows
  `offload.prefer` in resource_governor.yaml: `context`, the default, keeps
  the requested context and splits before stepping down the ladder; `speed`
  steps down on the GPU alone first and splits last. `offload.min_gpu_share`
  (0.0) refuses a split that would leave the GPU less than that share of
  the cost, `offload.ram_reserve_gb` (4.0) is the RAM a split leaves to the
  rest of the machine, and `offload.enabled: false` gives back the earlier
  decisions. The four keys are held to their ranges when the file is read,
  and the governor's config routes show and write them. A refusal names
  both shortfalls, the VRAM the GPU alone lacks and the RAM a split lacks;
  RAM that cannot be read means no split. A split is recorded in the
  decisions ring as `partial_offload`, without a new column. llama.cpp in
  process, which puts every layer on the GPU unless `n_gpu_layers` says
  otherwise, refuses a split by name before loading anything instead of
  failing for memory inside the engine; llama-server is unchanged. The KV
  cost of a context now comes from the model's own geometry when Ollama or
  the GGUF header describes it (layers, KV heads, key and value lengths, at
  f16), under any configured override, rather than the flat 0.5 GiB per
  1024 tokens. The loaded view keeps the total size Ollama reports beside
  its VRAM part, the cost the governor learns is that total, and a model
  split or held in RAM counts as loaded. Twenty-four contracts; eight
  earlier ones are superseded by name, each replaced in its own file.
- The governor sees every card the engines use, and what the rest of the
  machine needs. Its capacity was the total of the first card nvidia-smi
  printed, read by a new process at every snapshot rebuild, sometimes on
  the request path; a second card, an AMD card and the memory other
  programs held on a card did not exist for it. A hardware profile
  (`opti_oignon/hardware_profile.py`, `hardware_profile.yaml`) now reads
  every NVIDIA card nvidia-smi lists and every card DRM sysfs describes,
  each with an id, a kind and its memory. With `total_vram_gb` null, the
  capacity is the sum of the cards it selects: `devices: auto` takes the
  discrete cards of the first vendor that has any, NVIDIA then AMD,
  honouring CUDA_VISIBLE_DEVICES in the server's environment (a number
  there counts only under CUDA_DEVICE_ORDER=PCI_BUS_ID or with cards of one
  model, since CUDA otherwise counts the fastest first), and leaves out an
  AMD card whose own memory is under `integrated_below_gb` (1.0); a list
  names the cards instead, by id, UUID or bus id in any written form. The
  safety margin is kept on each selected card, and the memory other
  programs hold on them, the cards' used memory less what the engines
  declare, is not counted as free: not by admission, the dynamic context,
  the eviction it plans or the pressure signal. That deduction is taken
  only from a reading made since the engines last loaded or released a
  model; in between, the last paired figure is carried and the cards are
  read again at once, and a reading older than `vram_used_max_age_s`
  (600 s) is unknown. The cards are read at the first question, and their
  used memory is read again in the background every `vram_used_ttl_s` (5
  s), so no admission waits on nvidia-smi after the first reading. The RAM a
  split leaves to
  the rest of the machine is sized from the machine: the shipped
  `offload.ram_reserve_gb` is null, which keeps 1/16 of the total RAM,
  between 2 and 8 GB (`ram_reserve`), and doubles it while the kernel's
  pressure stall information reports memory pressure (`host_pressure`:
  entered when tasks waited on memory 10 per cent of the last ten seconds,
  left under 5); a number there is still a fixed reserve, and a file that
  names none keeps 4.0. The status route shows the cards, what other
  programs hold, the pressure and the reserve; the config routes write the
  new keys in range and refuse a write that crosses the floor and the
  ceiling or the two pressure marks, read from the file being written.
  smart_router and the live metrics read the RAM through the profile, the
  governor keeps its own reader so it still loads alone, held to the same
  answers, and no module imports psutil any more: no manifest declared it.
  A host without /proc/meminfo now reads its RAM as unknown, so no split is
  priced there. Speculative decoding sizes its draft budget from the
  detected cards, and from its configured 24 GB only when none is
  detected. Thirty-six contracts; go17, which pinned the shipped 4.0, is
  superseded by name by go25 in its own file.
- The governor asks the engine that will serve a call, charges a model
  already loaded only what the call adds, and plans an eviction at the
  price it admitted. Admission asks the backend that will serve the model
  first for its geometry and size, and the decision names it: one the
  caller names, else the registry's last resolution for the model, read
  from its cache (`BackendRegistry.cached_backend`) so admission calls no
  engine for it; a name no backend carries serves nothing. An engine may
  declare what one request costs
  (`InferenceBackend.cost_model`): a model with no KV cache is charged none
  at any context and is never stepped down for memory, a declared
  per-request state is charged once, and declared weights replace the
  estimate while an operator override still replaces both. No engine
  declares anything yet, so every model still prices as a token generator,
  and a declaration that does not hold together is ignored. A resident
  model asked at or below the context it holds is admitted at that context
  and charged nothing; its KV used to be charged again on top of the memory
  it held, so a full card split it with nothing on the GPU; a draft that
  loads beside it is still charged. Asked more, it is priced as a reload:
  its weights are what it holds less the KV of its loaded context, its VRAM
  and RAM are credited, the context ladder stops at the context it holds,
  which is the last step where the caller's floor allows it, and the
  dynamic context counts what the reload frees. With its loaded context
  unknown, it is priced as before. A model is never its own eviction
  candidate. The eviction a conditional grant plans uses the grant's own
  price, credit included only while the model is still loaded; it used to
  price the load without the operator's overrides or the model's KV
  coefficient, and could evict nothing for a load the admission had found
  too large. A grant the admission did not price is priced the admission's
  way. Twenty-two contracts.
- The governor places a split model layer by layer, as the model's own
  file weighs it, and tells the engine how many layers go to the GPU. It
  reads the tensor table of the file the serving engine loads -- the GGUF
  file llama.cpp names, or the blob Ollama's modelfile names, a vision
  projector set aside -- once per file identity and only when a split is
  priced (`model_manager.read_gguf_tensors`: versions 2 and 3, the 34
  tensor types ggml defines with their exact block sizes, metadata stepped
  over and never kept, every bound held against a damaged or hostile file,
  and each refusal named from a closed set). The last layers that fit the
  VRAM free now go to the GPU, each with its share of the KV cache, as
  llama.cpp and Ollama place them; what the tensors and the KV do not
  explain stays on the GPU, a model priced under its own tensors is charged
  them, and a draft loading beside it is charged on top. The decision
  carries the count (`num_gpu`, and `gpu_layers`): Ollama is sent it with
  the context it was priced for, unless the caller names its own -- a call
  that tells another context, or none, is placed by Ollama itself -- and a
  resident model's calls at the context its split load used keep sending
  it; a load, an eviction or the model leaving the loaded view ends that.
  The split prices the KV of one sequence, and Ollama keeps a cache for
  `num_parallel` of them at once: it is told the count only when
  `ollama_limits.num_parallel` is 1; unnamed, as shipped, or more, it places
  the layers itself. llama.cpp in process loads at the admitted
  context and layer count, an explicit `n_gpu_layers` being a ceiling the
  count can only lower; it loads a model it holds again only for a longer
  context on the GPU alone, closing the old copy first and never while a
  call runs on it; it refuses a split only when no count could be told; and
  its loaded view carries the context each model is held at. A new
  `split_speed` block, null as shipped, takes the GPU and RAM bandwidths
  and a `max_slowdown`: the decision reports how many times slower than on
  the GPU alone the split should run, a slower split does not hold and the
  context ladder goes on, and a speed that cannot be told never refuses. A
  refusal reached after a planned split is figured on the plan's total and
  names `no_layer_fits` or `split_too_slow` when they apply. There is no
  plan, and the split is the even one given before, when the file cannot
  be read, when the call names no context, or when the operator or the
  engine names the model's weights. Ollama's streaming head now refuses
  malformed options before admission, as generation does, so a refused
  stream leaves no load accounted. Forty-one contracts.
- The governor knows who asks. Every caller belongs to one of three
  classes, named in `resource_governor.yaml` (`classes.callers`):
  interactive (chat, pipeline: a person is waiting on the answer), user
  (benchmark, agent_eval, the direct backstop, and any caller the file does
  not name) and background (warm-ups, indexing, tuning, consolidation);
  every decision carries its class. The background never evicts: it counts
  no eviction credit, so it is never granted on condition of an eviction;
  it never reloads a resident for more context, and is served by the
  resident at the context it holds when its own floor allows; and it splits
  a model between VRAM and RAM only when `classes.background.allow_split`
  says so (shipped false). Where the free memory cannot show a fit, a user
  is admitted fail-open, since the engine's own LRU would then evict for
  it, while the background is refused at once, without waiting: a card
  whose free memory cannot be read, or a load whose cost cannot be told. A
  machine whose PCI bus lists no display controller but integrated Intel
  ones is checked against its RAM, the reserve kept; a card the DRM tree
  does not list still counts, and a bus that cannot be read proves nothing.
  A background load that names no context is priced and loaded at the one
  the governor names: the context last admitted to an interactive call on
  that model (read back from the decision ring, so it survives a restart),
  else the model's `max_output`, else the smallest step of the context
  ladder; with none of them it is refused. The background also leaves free
  the memory of every load admitted, in any class, that the engine does not
  show yet: it joins a pending load of the model it asks, at that load's
  context and loading nothing, and is refused until a later try a longer
  context than that load's, or a decision during which another load was
  admitted. A pending load stops counting once the engine shows it, once
  its call has given its ticket back and a later view still does not show
  it, or after `background_gate.pending_load_max_s` (600 s). A call on a
  resident at the context it holds still loads nothing and is admitted.
  The background is also evicted first: a resident only the background
  loaded, with no call in flight on it, is an eviction candidate before any
  other, whatever its idle time, and a ticket of a higher class held on it
  makes it that class's. Holding and releasing a ticket now counts the
  calls in flight per class; a ticket held longer than
  `background_gate.in_flight_max_s` (900 s; 0 or less is refused) is
  counted as a leak and no longer held. A background gate holds every
  background admission while an interactive call is in flight, waiting, or
  admitted and not yet held by any thread (for at most
  `background_gate.admitted_grace_s`, 10 s); while a user caller waits in
  the queue; and while the CPU pressure other programs suffer is at or
  above 10 (some avg10, percent) until it falls below 5: the highest among
  the user's other leaf cgroups under
  `user@UID.service`, the process's own cgroup and everything under it left
  out, or the system-wide reading where they cannot be told apart
  (`hardware_profile.read_cgroup_cpu_pressure`); a reading nobody can take
  holds nothing, and only the background pays for the reading. A held
  admission is recorded with its reason, and no background decision enters
  the refusal-rate window that drives backpressure. The queue
  (`admit_or_wait`) is now a priority queue: class first, then arrival; no
  caller passes a waiter of a higher class; within a class a waiter is
  passed by later callers that fit at most `queue.max_bypass` times (2); a
  caller that may not try waits without trying; every wake honours the
  emergency stop first, whether the waiter may try or not; and a waiter's
  retries are written neither to the decision ring nor to the refusal
  window, only its entry and its outcome. Each class has its own depth and
  wait (`classes.<class>.depth` and `wait_s`, else `queue.depth` and
  `queue.wait_s`); the background waits by default, eight deep, two
  minutes, and a caller may shorten its own wait. The warm-up is the first
  background caller: it asks as `warmup`, waits at most its `timeout`,
  loads under its own ticket and sends Ollama the context it was admitted
  at; its keepalive ping sends the context the resident holds, never
  waits, and is skipped while held; when the admission itself raises, both
  send nothing and say why. Resuming after an emergency stop answers at
  once, the warm-up waiting its turn on a thread of its own. `/status`
  gains a `scheduling` section (calls in flight and waiting by class, the
  gate's state, reason and reading, and the loads admitted and not yet
  seen); `queue.max_bypass` and the gate's scalars are writable through the
  config route, in range and with the CPU marks kept in order; the class
  tables stay read-only. Owed to the machine: what the gate spares a
  desktop under load, the context Ollama loads at when a call sends none,
  and the PCI reading with a card present. Sixty-eight contracts.
- A configuration write no longer changes the rules under a caller already
  waiting, nor forgets the pressure the governor has seen. A caller still
  holding the governor the write replaced is answered by the new one; a
  waiter's turn, the background gate it waits on and its refusal at the end
  of its wait follow the file just written (its deadline does not move). The
  refusal-rate window, the side each pressure hysteresis is on (memory, and
  the CPU pressure other programs suffer), the sustained-pressure timer and
  the warm-up keep_alive a sustained pressure shortened are shared with the
  rebuilt governor, under the one lock both hold: a write under sustained
  pressure used to forget the keep_alive to restore, which then stayed short
  until a restart. The readings themselves are taken again, and the new
  file's marks judge them. Two loads of one model not yet seen now count
  once against the background, at the larger of the two. Nine contracts.
- The hardware profile reads the CPUs as the kernel describes them. The
  cores the server may use are the CPUs online and in its affinity, each set
  of SMT siblings folded into one physical core; the profile also names the
  L3 domains, the NUMA nodes and the tightest `cpu.max` quota from the
  server's cgroup up to the root. The cores are ranked by the first kernel
  source that parts them (CPPC highest performance, AMD's preferred-core
  ranking, readable with preferred cores off, the scheduler's capacity, the
  highest frequency), and a new performance class starts where a rank falls
  more than `cpu_class_gap` (0.15) below the one above it; a flat or
  incomplete source is passed over, and with none the cores form one class,
  still ordered for the reserve. `cpu_classes` names the classes outright,
  and a reading is kept `topology_ttl_s` (60 s), so an affinity or a quota
  changed at run time is seen. A second view reads the machine itself: every
  online CPU, whatever the server's affinity, and no quota, the CPUs an
  engine in a process of its own may run on. An AMD display controller
  counts as integrated when its DRM card at the same PCI address does: a
  machine with an APU alone is judged on its RAM, like a machine with no
  card. Fourteen contracts.
- The governor plans the CPU threads an engine computes with, when it
  computes on the CPU: a model split between the GPU and system RAM, or any
  model on a machine with no card. The count is the machine's physical cores
  less a reserve left to the user's programs (`threads.reserve_fraction`,
  0.125 rounded up, between `reserve_floor` 1 and `reserve_ceiling` 4, never
  the last core), never under one, and no more than the fastest class's
  cores with `fast_cores_only` (shipped off until a machine measures it
  faster); `threads.models` names a model's own count. That count is told to
  an engine in a process of its own (Ollama), which neither the server's
  affinity nor its cgroup quota binds. llama.cpp computes in the server's
  own process and loads within the server's own cap: the cores of its
  affinity less the reserve, never past its `cpu.max` quota rounded down
  (one throttled thread stalls the others at the engine's barriers). Every
  decision carries `threads`, `threads_batch` and where they come from: the
  plan, an override, or the count a resident was loaded with. With the plan
  off or the CPUs unreadable no count is told, and neither is a load the GPU
  holds whole. Ollama is told `num_thread` by both generation heads, `embed`
  and `embed_many` (these two send options only when there is a count; the
  warm-up and its ping go through a generation head), unless the call names
  its own; llama.cpp gets `n_threads` and `n_threads_batch`, unless
  `backends.yaml` names its own (`n_threads_batch: null` follows
  `n_threads`, as llama.cpp does). The count is pinned with the resident,
  like its layers: a resident served as it is keeps the count its load was
  told, so no call reloads a model for a thread count; a pending load keeps
  its count for the call that joins it. Known limits: an engine under CPU
  limits of its own (a container's cpuset or quota) is told the machine's
  count, which `threads.models` can lower per model; another client of the
  same Ollama that does not repeat `num_thread` reloads the model; a restart
  of the server forgets the pins, so the first call to a resident loaded
  with a count reloads it once; the status shows a llama.cpp resident at the
  count admitted, before the server's cap. Twenty-two contracts.
- Background work runs in worker processes that leave the machine to its
  user. `background_pool.py` starts them with `spawn` (no Unix socket, a
  persistent pool); before any task, each puts every one of its threads in
  SCHED_IDLE, enters the idle I/O class (`ioprio_set`, with the syscall
  numbers of x86-64 and of the generic table arm64 and riscv64 share, for a
  64-bit interpreter only; a 32-bit interpreter on a 64-bit kernel, whose
  numbers differ, is said unsupported) and takes the CPUs of the cores
  outside the reserve, the reserve being the highest-ranked cores (fastest
  class, highest rank, lowest CPU). A kernel refusal leaves the worker at
  normal priority and is named by its errno, the other steps done all the
  same. The server's own threads are never lowered, and the worker setup
  refuses to run in the server's process. There is one worker per core
  outside the reserve, at most `threads.background.max_workers` (4), never
  more than the quota less the reserve, never under one;
  `in_flight_per_worker` (2) tasks are queued per worker, and the workers
  exit after `idle_shutdown_s` (120 s) without a task, to start again with
  the next one. Stopping the server ends the workers at once instead of
  running the tasks still queued, those of executors retired without waiting
  too, and closes the server's end of their result pipes, so a result cut
  halfway holds nothing. With the plan missing or off, or processes that
  cannot start (said, and remembered), the work runs on the thread that asks
  for it. Indexing is the first user: the chunking of each file runs in the
  pool, and a worker holds a whole document as it parses it, so a file is
  sent only while the estimated parses of the files in flight, of every
  indexing job and its own included, fit the room the governor gives the
  background (the RAM available less the reserve a split leaves the machine,
  none under memory pressure); the estimate is the file's size times
  `threads.background.parse_expansion` for its kind (4 by default, 10 for a
  PDF, 30 for a Word document, 50 for a spreadsheet). A job with nothing in
  flight waits for room, and is served before a job with files in flight
  sends more; a file larger than the room goes once nothing is in flight; a
  parse that runs on after its job is cancelled keeps its room until it
  ends. The job stores the chunks in order, each file under a governor
  ticket for the embedding model as the background caller `index`; a refusal
  a wait can lift is asked again after `held_retry_s` (30 s), a final
  refusal fails the file with its reason. The pool is shared by every job: a
  task that breaks retires only the workers it ran on, and each file that
  may have killed them is chunked again alone, in a worker no other task
  shares, so only a file whose own worker dies fails, by name; a file whose
  task an executor retired under it is sent again; a worker the pool's own
  shutdown ends blames no file. A cancel is seen within half a second while
  the job waits for a chunking or for its admission (the governor's queue
  takes the job's cancel): the files in flight go back to the queue, nothing
  chunked after the cancel is stored, and an admission granted as the cancel
  lands is handed back. Whatever stops a job, every file it holds goes back
  to the queue. `ingest_file` now creates the collection after the chunking.
  Known limits: on a machine whose GPU memory cannot be read, indexing fails
  every file unless the embedding model is already loaded (the background is
  refused a load it cannot price); if the kernel refuses SCHED_IDLE, the
  pool keeps its workers at normal priority, said; results come back from
  the workers by pickle, so the pool is no security boundary, as the
  server's own thread was not; an `index` caller the configuration would
  move out of the background class fails the job with the pool's reason;
  under a quota an idle worker draws from the server's own quota, hence the
  bound of the quota less the reserve; each worker imports the server's main
  module again (under `oo` the entry script imports only the command line;
  not proven when the server is launched by path); the parse factors are
  estimates, not measurements. Forty-four contracts.
- The auto-tuner measures the thread count, and the governor keeps what it
  confirms. `parameter_space.threads: auto`, now the default, sweeps the
  plan's count, each of `threads_fractions` (0.5, 0.75) of it rounded up,
  and the fastest class's core count, never past the plan; a list is swept
  as written, and an unreadable value falls back to auto, named in the log.
  The baseline runs at the plan's count. Each trial takes a governor ticket
  as the background caller `tuner`, naming its engine, holds it through the
  call and sends the context it was admitted at: a sweep now waits for the
  machine and no longer evicts. A held trial is an error named as such; a
  final refusal ends the sweep, and so does a baseline that could not be
  measured. Each result carries its engine, its placement (`cpu`,
  `split:<GPU layers>`, or none when the card holds the model whole or the
  split's layers are not counted) and the count applied. The best thread
  trial (the smaller count on a tie) is confirmed and kept only when the
  confirmation has no error, every thread trial was measured, the engine
  applies a count per call (Ollama does; llama.cpp fixes it at load), one
  engine and one placement ran every trial, the placement is known, the
  count is within the plan, and the confirmed generation rate is no lower
  than the baseline's; otherwise a last trial runs at the plan's count. When
  the last trial to reach the engine ran at another count (the run
  cancelled, refused or failed, or its settling trial held by the governor),
  the governor pins what a fresh load would get, the plan or a count kept
  for that model, engine and placement, so the next call reloads the model
  once at it; a run that changed no pin, or whose trials reached no engine,
  changes none, and a run reads and restores pins only in a running
  governor, starting none. The fastest class's candidate counts the
  machine's cores, as the plan does. A kept count is stored per model,
  engine and placement with a fingerprint of the machine's cores, and is
  planned from then on when all three match and the fingerprint holds,
  within the current plan; a model's own count still wins. A call that sends
  a resident its own count re-pins it, since Ollama reloads for any other.
  The tuner's status route no longer fails on `auto` (the schema takes a
  list or the word, with its fractions), and a tuner profile carries the
  count it kept, or why it kept none. Known limits: a model that must be
  split is measured only when already loaded (the background does not
  split); the sweep measures at the background's context (the model's last
  interactive context when there is one); the first load after a start does
  not know its engine yet, so a kept count applies from the next load;
  threads are measured at the tuner's other defaults (batch 2048, flash
  attention); a split whose layers are not counted, and llama.cpp, keep
  nothing; the fingerprint names neither the RAM nor the card. Eighteen
  contracts.
- The governor's `/status` shows the cores and the background. `threads`
  gives the plan, the machine's physical cores, the reserve, the server's
  quota and its own cap, the background's budget, the pins, and the counts
  the tuner kept with whether each holds on this machine (at most
  `threads.status_limit`, 50, newest first); `background` gives the pool's
  state (reading it creates no pool), the I/O class policy of the nearest
  cgroup that names one, and what the idle I/O class does on each disk the
  indexing reads: it takes effect under bfq, is deferred under mq-deadline
  (whose aging is given), and does nothing under kyber or none; a policy
  that promotes it takes effect under bfq and mq-deadline only. oo never
  changes a disk's scheduler. A section that cannot be read says it is
  unavailable while the rest of the status stands. The hardware profile's
  view carries the CPUs as the topology reads them. An indexing job notes
  the device of each file it sends the pool, and its route names the disks
  behind each device once (a partition's disk, the disks under device-mapper
  and md, a mount's device, found by the file's path where its device number
  shows on no mount, as on a btrfs subvolume), in device order. Known
  limits: a btrfs over several disks names only the device its mount gives;
  overlayfs, ZFS and network filesystems are unknown; the disks are those of
  the files sent since the server started. Owed to the machine: what the
  idle CPU and I/O classes spare a loaded desktop, the thread count each
  model runs best at, and whether the fast cores alone run faster. Eleven
  contracts.

## 2.2.0 -- 2026-07-28

The semantic cache's published surface loses its legacy naming, which renames
five HTTP paths and four schemas. That breaks any client that spoke the old
prefix; the bundled frontend ships updated in the same release, which is why
it lands as a minor rather than a major. The version register becomes the
single declarative source, and release verification learns to demand the
project's key rather than any key.

### Changed

- **Breaking.** The semantic cache endpoints move under
  `/api/cache/semcache/*`, and the four cache schemas follow
  (`SemCacheStatusResponse`, `SemCacheStatsSchema`, `SemCacheConfigUpdate`,
  `SemCacheClearRequest`). The retired prefix carried an internal iteration
  code, which is exactly why it goes. The bundled frontend is updated in
  the same release; external clients must adopt the new prefix.
- The version register (`opti_oignon/__version__.py`) is the only place a
  version is declared. The packaging manifest, the frontend package and the
  newest changelog heading are contract-pinned equal to it; the health
  dashboard default derives from it instead of restating it; code banners
  no longer carry a version at all. Each retired site had drifted to a
  different stale value, which is what this policy ends.
- The merge guard that rejects internal nomenclature now reads a session
  code in either case and through a camel-case continuation, while never
  charging platform names whose digits run into a lowercase letter.
- [SECURITY] **Breaking in Bulbe.** Bulbe now REFUSES to load a GGUF whose
  provenance does not verify -- including one that is simply not enrolled yet.
  Enrol existing models before switching to Bulbe. Configuration cannot weaken
  this; a security mode that cannot be resolved is treated as Bulbe. Daily is
  unchanged by default: it observes and logs without blocking. Models served
  through Ollama are not affected, as the gate sits on the in-process load seam.

### Fixed

- [SECURITY] `verify_release.sh` exit codes match their documented map: a
  checksum file that is absent in strict mode exits 3 (missing file), not
  2 (mismatch). Failure diagnostics no longer leak the exit code into the
  printed message.

### Added

- [SECURITY] `verify_release.sh` honours a pinned project fingerprint
  recorded in `scripts/release_key.fpr`: when present and no `--key` is
  given, a genuine signature from any other key is refused. Without a pin
  it now says out loud that a valid signature proves integrity, not
  identity.
- [SECURITY] Model weight provenance. GGUF files are now pinned to the sha256 of
  their bytes in a manifest sealed with ML-DSA-65 (or HMAC-SHA512 where liboqs is
  absent), and the llama.cpp in-process load seam verifies that pin before the
  bytes reach the native parser. The path guard already proved WHERE a model file
  sits and the SSRF guard proved WHERE it was fetched from; nothing proved WHAT it
  contained, and a trojaned or corrupted model was parsed by native code
  regardless.
- [SECURITY] `POST /api/backends/gguf/download` accepts an optional
  `expected_sha256`. It is verified against the partial file BEFORE the download is
  promoted to a loadable `.gguf`, so a mismatch never materialises a model. Taken
  from a model card rather than from the serving host, it is the only check on that
  path that is not trust-on-first-use. Successful downloads are enrolled in the
  manifest automatically.

- Merge guards under `.github/scripts/`, all wired into CI. `public_clean_guard`
  and `public_language_guard` hold the public surface to English prose free of
  internal working nomenclature; `comment_only_guard` catches files that shed
  that nomenclature without changing behaviour; `isolation_seal_guard` keeps the
  test-isolation ledger shrinking and never growing; `published_prose_guard`
  pins every published schema description to a recorded digest, so the API
  surface cannot drift silently; `summary_fidelity_guard` and `red_team_guard`
  hold measured floors for summary fidelity and for the defense layers, failing
  the build when a measurement falls below the written threshold.
- Red team engine under `opti_oignon/redteam/`: attack generation, strategies,
  scoring, and reports against named defense targets, with a local probe driver
  in `scripts/redteam_llm_probe.py`. Documented in `docs/redteam/`.
- A shared isolation window for contract suites, `tests/_isolation.py`. Suites
  that load a single module from file no longer hand-roll their own package
  window, which is what let a dead signature primitive stay invisible.
- A browser end-to-end harness: Playwright specs under `frontend/tests/e2e/`
  driven by `scripts/run_e2e.sh`, covering the health surface, the reported
  security mode, and the sign-in refusal path.
- `constraints.txt`, the pinned environment under which every guard that reads
  the published surface runs, and under which its digest is regenerated. A
  floating resolver could otherwise move the schema without a line of this
  repository changing.
- Release signing: `scripts/sign_release.sh` and `scripts/verify_release.sh`
  produce and check a detached GPG signature plus a sha256 over the archive.
- `capability_manifest.py`, per-request introspection of what a target model can
  actually call, and `context_ledger.py`, per-request context measurements that
  record numbers about a turn and never its words.
- An Android client skeleton under `android/`. It does not yet talk to a
  backend; device-to-device sync remains unwired.

## 2.1.0 -- 2026-06-27

A capability release on two fronts: an agentic robustness cycle that makes local
tool use far more reliable, and a memory-system overhaul that closes the
capture -> store -> injection loop on a single source of truth.

### Added

- Native model function-calling for agentic tool use: when a model advertises the
  capability, tool calls go through its native function-calling interface, with a
  JSON-schema-constrained path as the unconditional fallback otherwise.
- Automatic memory capture: after a turn is saved, facts are extracted and stored
  in the background every few messages, so memory accumulates without a manual
  `/extract`. Gated and throttled, fire-and-forget.
- Memory health endpoint: `GET /api/memory/health` reports the canonical, archive
  (semantic) and embedder tiers, so a degraded recall path is visible instead of
  silent.

### Changed

- Agentic tool loop hardened (the robustness cycle): enum-forcing for constrained
  arguments, intent-transpiler salvage and argument auto-repair for malformed tool
  calls, an error-feedback retry so a failed call self-heals, an anti-spin guard
  plus a verification pass to stop the agent looping, and capability-aware
  reasoning handling that avoids the think=True / HTTP 400 case on models that do
  not support it, with an explicit optimize toggle.
- Memory unified on one source of truth (the coordinated MemoryStore): the
  `/api/memory` surface (list/add/delete/clear/extract) is re-backed by the new
  store and mapped onto the existing schema, so the frontend is unchanged. The
  working block now keeps a salience floor -- durable facts are always injected,
  not dropped on an unrelated turn -- and marks injected facts as used.
- The memory vector (semantic) layer degrades gracefully: when chromadb is not
  installed it falls back to canonical keyword/recency recall instead of raising,
  so the memory tab, list and migration keep working without it; only similarity
  search is disabled, and health reports the archive tier as unavailable.

### Fixed

- The memory tab and the injector no longer read different stores: facts entered
  in the tab now surface in recall. Previously the tab wrote the legacy store
  while the injector read the new one, so tab-entered facts never appeared.
- An unrelated query no longer drops durable memories from the working block (the
  old retrieval path discarded every fact scoring zero).

### Internal

- One-shot legacy `memories.db` -> MemoryStore migration runs once at application
  boot: idempotent (the store's dedup merges a re-run), marker-guarded, and
  fail-safe -- a migration problem is logged and swallowed, never breaking
  startup.
- The silent no-embedder path now logs once and is surfaced by the health probe
  rather than degrading recall invisibly.

## 2.0.2 -- 2026-06-25

Data-integrity release: fixes a bug that caused agentic conversations to be
lost, plus related persistence hardening.

### Fixed

- Agentic conversations (those using the in-session sandbox and tool calls) were
  not saved: the history was empty after a page reload and the context token
  counter showed 0. The persistence call passed an unexpected keyword argument,
  raising an error that was silently swallowed, so every agentic turn was
  dropped. This affected anyone running 2.0.1.
- Turns that combined a reasoning pass with tool use persisted only the
  reasoning; the tool output was dropped on reload. The complete turn is now
  saved.

### Changed

- The streaming idle-disconnect timeout is now configurable through the
  `OPTI_IDLE_TIMEOUT_S` environment variable, and its default was raised from
  60 to 600 seconds so slower local models that stream in bursts are not cut off
  mid-response.

### Internal

- Conversation persistence now records the generating model on assistant
  messages and fails loudly (a logged warning with traceback) instead of
  silently, so a future persistence regression is visible.
- Added defensive persistence to the cascading and speculative generation
  pipelines so they cannot drop a turn if they are ever wired into a saved
  conversation.

## 2.0.1 -- 2026-06-23

Maintenance release: a version-reporting fix, a small frontend security
hardening, and repository cleanup. No functional changes to the application.

### Fixed

- The command line and the package metadata now both report 2.0.1. The version
  module had carried a stale internal version string over from pre-release
  development, so the 2.0.0 build reported the wrong number from `oo --version`.

### Security

- Removed unnecessary `{@html}` rendering in the benchmark panels, eliminating
  an unused HTML-injection surface in the frontend. [SECURITY]

### Internal

- The public continuous-integration pipeline (Python lint, frontend type-check
  and lint, install smoke test, security scan) now passes.
- Removed a dead pytest configuration block from `pyproject.toml` that referenced
  a test suite not shipped in the public distribution.

## 2.0.0 -- 2026-06-20

A complete rewrite and public re-release. Opti-Oignon began as a Gradio-based
local-LLM optimization framework (the 1.x line); 2.0.0 replaces that entirely
with a new local-first AI inference platform built on SvelteKit and FastAPI,
running against Ollama on your own hardware.

### Working in this release

- Private, streamed chat from local Ollama models over WebSocket.
- Accounts with registration, login, and optional 2FA (TOTP and WebAuthn). [SECURITY]
- Two security modes. Daily is the normal mode; Bulbe binds the backend to
  `127.0.0.1` at the socket layer and accepts cookie-only authentication, with a
  guarded, human-confirmed downgrade ceremony and fail-secure behavior when the
  mode cannot be determined. [SECURITY]
- Encryption at rest via SQLCipher. [SECURITY]
- Projects with RAG context backed by ChromaDB.

### Also included, still maturing

A broad backend surface is implemented and covered by the test suite but not yet
verified end to end on a fresh install: smart model routing, multi-model
consensus and cascading inference, a semantic cache, a sandboxed agent loop,
benchmark and performance dashboards, a resource governor, an LLM-powered red
team engine, RBAC and multi-user isolation, and an encrypted Notes tab. See the
Status section of the README for details.

### Not yet wired end to end

Veilid device-to-device sync is incomplete on the producer side, so nothing
moves between paired devices yet. Everything that depends on it -- remote
inference, collaborative Notes sync, and the mobile client -- is experimental.

### Security posture

Deny-by-default authentication, per-user data isolation, Argon2/bcrypt password
hashing, a disposable bubblewrap sandbox for any LLM-driven filesystem, shell,
or code tool, ML-DSA-65 post-quantum signatures on records intended for sync,
and a hash-chained audit log. Security follows Kerckhoffs's principle: it rests
on keys and correct implementation, not on secrecy of the code. [SECURITY]

---

Earlier history (the 1.x Gradio line) remains available in the git tags.
