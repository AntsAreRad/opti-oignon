# Changelog

All notable changes to Opti-Oignon are documented in this file.
Security-relevant changes are marked with [SECURITY].

## Unreleased

The merge guards learn to say what they cannot prove, rather than reporting an
absence of proof as a verdict, and a new one measures what importing the
package costs.

### Added

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
  by suite. A child process a contract starts is not covered.
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
- `public_clean_guard.py`, `published_prose_guard.py`, `summary_fidelity_guard.py`
  and `red_team_guard.py` are unchanged in this cycle and continue to gate
  merges. `isolation_seal_guard.py` and `public_language_guard.py` changed only
  in their sealed ledgers, which fell as debt was paid.

### Fixed

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
  gallery pages gained theirs. Ctrl+K focuses the conversation search again:
  the handler queried a placeholder the field no longer carries, and now
  queries the field's `data-oo-search` attribute. Scrolling to a settings
  group or to the end of a chat is smooth only when neither the motion
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
