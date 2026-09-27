# Fact check: the passage core

`opti_oignon/factcheck` decides whether a claim is supported by the evidence
it is handed. "Supported" means supported by this passage of this source,
read on this date; it never means true. The core reaches no model, no store
and no network: it imports the standard library alone at module level, reads
its YAML lazily, and imports nothing from the application. Nothing in the
application calls it yet; retrieval of the owner's notes, ledger and library,
an entailment model and a surface in the interface are later work, each with
its own conditions.

```python
from opti_oignon.factcheck.checker import FactChecker

checker = FactChecker()                  # runs the canary; refuses every call if it fails
verdict = checker.check(claim, items, as_of="2026-09-26", read_on="2026-09-27")
verdict.value, verdict.basis, verdict.reasons, verdict.text, verdict.record
claims = checker.claims_from_text(answer_markdown, lang="en", origin={"author": "assistant"})
```

`as_of` is the date the claim is about and `read_on` the day the check runs;
the core reads no clock. A check for a date after the day it runs is refused
by name: sources are read as they stand that day and cannot speak for a later
one. `decide.check` may be called directly; its record then says the canary
did not run, since only a `FactChecker` runs it.

## The verdicts

| Verdict | When | Basis |
|---|---|---|
| supported | a whole sentence of an admitted source, valid at the date, restated verbatim under fold v1, in a window that does not qualify it | deterministic |
| conflicting | the claim reports a past decision of the owner, with no date of its own, a fact of his ledger restates it, and a later decision replaced it (reason `superseded`, with both dates) | deterministic |
| not enough evidence | the claim is checkable and none of the above holds; every reason found is kept, the leading one first | none |
| out of scope | nothing checkable remains once the wrapper is stripped | none |
| contradicted | never given by this core | -- |

"Contradicted" is part of the closed vocabulary and never produced here: a
source that differs from the claim is not enough evidence, shown beside the
claim. A contradiction by rule needs the same event, which this core cannot
establish; that reader is later work, restricted to a closed list of
single-valued statements. No evidence is never "contradicted".

The vocabulary is closed: five verdicts, three bases, and a fixed list of
reasons per verdict, complete for the whole design (reasons only a later
reader gives are already named). A verdict built with anything else, or with a
reason under the wrong verdict, raises. No code is a word that claims truth,
and no text the core builds holds one, in English or French, inflected or not,
outside the quoted words of a source.

## The passage check

A passage counts only where the host finds it: for one quote against one
chunk, the host recomputes the chunk's SHA-256 (a mismatch refuses the chunk,
`chunk_changed`), refuses a chunk that is not NFC (`chunk_not_nfc`, never
repaired), folds the quote (empty is `quote_empty`, under 12 characters is
`quote_too_short`), folds the chunk with an alignment map, and returns every
aligned occurrence at the chunk's own code points (none is
`quote_not_found`), the first 16 of them and a flag past that. A span never
crosses a chunk: each chunk is searched on its own, after its own hash check,
and the check scans each chunk on its own too, so a sentence split across two
chunks supports nothing.

Fold v1, one code point at a time:

| Class | Becomes |
|---|---|
| every code point `str.isspace()` accepts | one space; runs collapse |
| U+00AD, U+200B, U+200C, U+200D, U+2060, U+FEFF | removed |
| single quotes and apostrophes (U+2018 to U+201B, U+02BC, U+2032, U+00B4, U+0060, U+FF07) | `'` |
| double quotes (U+201C to U+201F, U+00AB, U+00BB, U+2033, U+FF02) | `"` |
| hyphens, dashes, minus (U+2010 to U+2015, U+2212, U+FE63, U+FF0D) | `-` |
| U+2026 | `...` |
| ligatures U+FB00 to U+FB06 | `ff`, `fi`, `fl`, `ffi`, `ffl`, `st`, `st` |

Never case folding, never digit folding, never NFKC: compatibility forms turn
ten with a superscript nine into 109. An expansion is atomic, so a quote that
starts inside a ligature is not an occurrence; a removed code point is never
at the edge of a span. The check only ever locates a whole sentence of a
chunk, so the alignment rule is proven on the passage search itself, and the
hash and NFC refusals on the check's path are those of admission.

The only support is a whole sentence: the claim and a source sentence are
compared with their final `.`, `!` or `;` and closing quotes aside, and their
first letter lowered. A claim found inside a longer sentence ("It is false
that ...") is never equal. The splitter never ends a sentence at a period
after a single letter (an initial: "George W. Bush", "U.S.") or after an
abbreviation of its closed list (titles, company and month abbreviations,
"approx.", "a.m.", and the French ones); a sentence that follows a period
after any other capitalised word of one to three letters (a Roman numeral
excepted) may be the tail of the one before, and is never read as whole
(`context_incomplete`). A sentence that restates the claim is located again by
the passage check, and its span is the host's. When a sentence restates the
claim and a check fails, the verdict gives that check's reason (a source not
valid at the date, a passage that repeats a model's words, a window that
qualifies it); "not restated" is said only when no sentence of any source
searched restates the claim.

## What counts as evidence

Every source handed in is recorded with its admission, never dropped.

| Refusal | When |
|---|---|
| `author_model` | a model wrote it: an assistant turn, a model-extracted memory, an agent's note |
| `author_unknown` | no author is recorded; the core never guesses |
| `author_model_quoted` | per span: the span overlaps text found equal to a stored assistant turn |
| `snippet` | a search engine's excerpt; a web source is a fetched page only |
| `no_consent` | no consent grant |
| `retracted` | kept as a flag, never used |
| `successor_missing` | a replaced fact whose successor was not handed in |
| `chunk_changed`, `chunk_not_nfc`, `chunk_too_large`, `over_budget` | the passage check and the limits |

A supporting passage is also refused for a number read from a table, OCR or
multi-column extraction (`extraction_uncertain`; the hash proves only that
the extracted text has not changed), from a copy ingested before the
source's correction (`ingested_before_correction`), and, for a claim about
the world, from a source that cites a retracted work (`cites_retracted`).
Preprints and corrected sources are admitted and shown as such.

The tiers are the owner's decisions (ledger, Core), his notes, his library
and the web. The owner's own claims count only his decisions and notes; when
nothing else was handed in, the text says that no decision or note of his
was. A claim about the world counts every tier the same, and the tiers travel
into the verdict: a claim supported by the owner's notes alone reads "Your
notes say ... Not checked against any other source."

## Whose claim it is

A claim is the owner's own when its opening subject is the first person with a
decision marker, or, in the assistant's text, the second person with one;
anything else is about the world. An own claim meets the owner's own sources
after a closed rewrite that knows who wrote each side: in the owner's text
the first person is the owner, in the assistant's text the second person is.
Only the subject opening the sentence, the auxiliary right after it and the
possessives of that person are rewritten into one owner token. So the
assistant's "You decided to hold the offsite in Berlin." and the owner's
"We decided to hold the offsite in Berlin." are one claim, and so are their
French forms; but "we pay you" and "you pay us", "my car" and "your car", a
note the owner wrote in the second person, and the assistant speaking of
itself in the first person, never are. French "on" is a subject only before a
verb of a closed list and never in a claim marked English, so "On 12 May ..."
opens a date.

## Time

A source starts at its "valid from", else at the date it was recorded
("recorded on"), else it is treated as started and shown "validity unknown".
For a claim about the present, a source does not speak for any date before
its own date: its start is its own date when that is later. It ends at its
"valid until", or, for a fact of the ledger or the Core replaced by a fact
handed in, at the start of its successor when that comes first: the drift
ledger links a replaced fact to its successor and never writes an end date,
so a rule that read only the end date would keep a replaced decision valid
forever. When the successor has no date, or one before the replaced fact's
own, the dates cannot say when the replaced fact held: it is valid at no
date, and both are shown "validity unknown". A note or document carrying a
successor is flagged `supersession_coarse` and the link is ignored. A source
not valid at the date never supports.

- A past decision the owner replaced ("We decided to hold the offsite in
  Berlin.", replaced by "... in Oslo.") is conflicting, `superseded`: "Your
  decision recorded on 2026-06-02 (...) says this: ...; a later decision
  recorded on 2026-07-10 (...) replaced it: ...". Before the replacement it
  is supported, and the text adds that it was later replaced, when, and by
  what. A decision that carries its own date ("in June 2026"), or that is
  worded in the present, is not enough evidence, `no_longer_held`, instead:
  this core does not read the claim's own date.
- A present state that was replaced ("The offsite is in Berlin.") is not
  enough evidence, `no_longer_held`, "held until" the replacement, which is
  quoted; never conflicting.
- A decision replaced more than once is dated by the decision that holds
  now, which is the one quoted, and each decision in between is named with
  its date.
- A source whose explicit end has passed, with no successor, gives
  `expired` with its date.
- A claim about the present ("current", "latest", "now", and their French
  forms) found only in undated sources is `source_undated`. Every verdict
  line carries each cited source's own date, or "undated", apart from the
  date read and the date the claim is checked for.

## The window

A source sentence is read with its window: the sentences before and after in
its paragraph (for a list item, the items beside it), every heading on its
path above it in the chunk and the heading the store gives for the chunk,
and what leads into it: for a list item, each parent item and the line that
leads into the list; for a paragraph, a one-line label set above it (a line
that ends with a colon or is wholly bold, or a raw HTML line, whose text is
read).

- A qualifying marker there ("myth", "false", "hypothesis", "to verify",
  "wrongly", "refuted", "not sure" ... and their French forms), a neighbour
  that is a bare denial ("No, it is not."), or a sentence struck through,
  gives `context_qualified`, with the marker and where it stood.
- A heading or lead-in that frames the sentence gives its own reason: a
  condition ("if", "unless" ...) `conditional`, a possibility or a forecast
  ("may", "will", "forecast", "scenario" ...) `evidence_hedged`, a narrower
  population ("in mice", "in vitro" ...) `population_narrower`, unless the
  claim itself carries a marker of the same class; a negation ("not",
  "no longer" ...) always gives `context_qualified`. The modal "may" counts
  only in lowercase, so a heading "May 2026" frames nothing.
- An attribution ("according to", "said", "claims" ...) anywhere in the
  window, or a quotation, gives `attributed`.
- The context is incomplete (`context_incomplete`) for a list item with
  neither lead-in nor heading, a first sentence that opens in lowercase, the
  first sentence of a chunk not known to start its source (its item has
  more than one chunk, its offset is not 0 and its locator does not say
  `starts_block`), the last sentence of such a chunk when it lacks final
  punctuation (unless the locator says `ends_block`), and a sentence after
  an uncertain period.

## Scope

Decided on the claim alone: a line of code, a heading, a table row, an image,
struck-through text and markup the reader does not know are out of scope; so
are a sentence with no word, one over 600 characters and a question. An
opinion opener ("I think", with the "that" after it, and its French forms) is
stripped and the rest is checked; an advice opener ("you should", "make
sure") is stripped, and the rest is checked only when it carries a digit or a
name, else it is an `instruction`; a wrapper with no word after it ("I
think.") is `empty`. A subject that refers to something named elsewhere is
`not_standalone`: a pronoun, a demonstrative standing alone ("This is ..."),
a possessive ("Its capital ..."), an anaphoric opener ("Another study",
"Both drugs", "One of them", "Such a device"), and, in a claim about the
world, a first- or second-person subject or possessive, which names whoever
wrote it. A claim about the world opening on "the", "this" or their French
forms, with no name in it, is `subject_unresolved`; a month or a weekday is
not a name.

`claims_from_text` reads a markdown answer through a closed subset (fenced
and indented code, headings, lists, blockquotes, emphasis, strikethrough,
inline code, links, images, tables; anything else is "markup not read") and
returns one claim per sentence with its offsets into the answer, markup
included; what is out of scope comes back with its reason, never dropped. A
source that is not markdown is read for its lines, paragraphs, `#` heading
lines and list-item lines only, and its text is never changed.

## The record

Every verdict carries a record in canonical JSON (sorted keys, UTF-8, no
float): the claim and how it was read, every source with its admission and
validity, every located span with its window, every candidate sentence with
its passage outcome (a passage located past the cap says so) and what blocked
it, the verdict with all its reasons, the text, and the digests of the rules
(fold table, splitter, markdown subset, lexicons, templates, the nearest
sentence's floor, Unicode version) and of the configuration. The judge is
null, with its reason (`no_judge`): this core has none. Its id is the SHA-256
of that JSON; the creation time and the path that computed it sit outside the
digest. The same inputs give the same id, in whatever order sources are
handed in. `replay` recomputes a record from its chunks and says
`reproduced`, `source_changed`, `rules_changed` or `record_differs`. The
record is reproducible, not tamper-evident: it says `integrity: digest_only`.

Every verdict other than "supported" sets side by side the claim as written,
with its offsets, and each source passage examined, with its source, tier,
chunk hash and offsets, and each differing run of words named:

- a sentence that restated the claim but was blocked carries what blocked
  it;
- a replaced decision carries both sides, the runs that differ named;
- a chunk where no sentence restates the claim is shown by its nearest
  sentence, the one sharing the most of the claim's words, when it shares at
  least half of them, with each differing run of words ("1608" beside
  "1607"); the text names the nearest of them all; a chunk with no sentence
  that near is shown with the number of sentences compared;
- a source not read as evidence is named with why, its text not examined.

The runs are words as written after the fold; typed readings of numbers,
dates, units and entities are later work.

## The canary

One hundred and one planted errors and positive controls, in English and
French, across eight classes (passage, restatement, provenance, time, scope,
context, person, and a claim with only irrelevant or blank evidence), run
through the checker's own check function at every construction. If one item
comes out otherwise than planted, the checker is refused: every call returns
a refusal naming the failing items. Its digest and outcome are in every
record. The canary proves the core able to refuse on the classes it holds;
it measures nothing beyond them.

## What it cannot see

- Every paraphrase, and every restatement that is not a whole sentence:
  such claims read "not restated as a whole sentence".
- A claim of fewer than 12 characters after the fold: the passage floor
  never locates it, so it is never supported.
- Support that needs two windows or two chunks; a word hyphenated across a
  line break; spacing around French guillemets.
- Negations and hedges outside a whole-sentence comparison and the window's
  lexicons; a qualifying, framing or attributing word in the window that does
  not in fact bear on this sentence (it blocks, the safe direction: a
  heading such as "What you will need" blocks what follows it).
- A sentence after an uncertain period, the first sentence of a chunk not
  known to start its source, and the unpunctuated last one of a chunk not
  known to end it: all blocked, the safe direction.
- Pronoun, possessive and definite subjects without the sentence that names
  them; a name in the subject is necessary, not sufficient. A claim about the
  world in the first or second person.
- An own claim whose owner is not the opening subject ("In June we
  decided ..."), and an equivalence carried by an object pronoun ("you
  entrust yourselves", "we entrust ourselves").
- Extraction errors present at ingest in an unflagged chunk; poisoned
  sources the owner owns; cherry-picking and true-but-misleading claims.

## Configuration

`opti_oignon/config/factcheck.yaml` holds the rules version, the limits
(claim length, the 12-character quote floor, chunk size, 64 sources, 32
chunks per source, 2,000,000 characters per check, 16 occurrences) and the
canary's refusal switch. Every number is a proposal, not a measurement. In
the container, the canary ran in 0.07 s and one check over 2,000,000
characters took 0.30 s on plain text and 0.88 s on markdown
notes; 0.79 s and 1.31 s when every sentence
of them restated the claim; 0.40 s and 0.89 s when
every sentence was one number away from it (a load average near 3, other work running). The machine's figures are owed.
