# Sandboxed Agent

The Opti-Oignon agent (Theme 3, Odysseus Core) is a multi-turn loop that plans,
calls tools, reads the results, and continues until it produces a final answer or
reaches a round cap. Its autonomy is bounded by two hard constraints: a disposable
sandbox for every side-effecting tool, and human approval for sensitive actions.

## The loop

Each round, the agent streams a model response, dispatches any tool calls, and
feeds the tool output back as an untrusted observation. When the model returns an
answer with no tool calls, the loop stops. A bounded verifier pass can review the
final answer before it is returned. The loop is built so that nothing -- a model
error, a refused tool, a failed dispatch -- ever raises into the conversation
path; failures become observations or a terminal result.

## Tools and the sandbox

The agent's filesystem, shell, and code tools (`bash`, `view`, `create_file`,
`str_replace`) run **only** inside a disposable bubblewrap sandbox with no access
to the host filesystem or network. Files are copied in explicitly; results are
copied out only after review. Tool output is always wrapped as untrusted data:
the model is told to treat everything between the untrusted-data markers as
information to reason about, never as instructions to obey. The non-sandbox tools
(`web_search`, `manage_memory`, `manage_notes`, `manage_skills`) are
handler-backed and follow the same untrusted-output discipline. Since the workspace cycle the
sandbox behind these tools can be a named, conversation-bound workspace with
explicit copy-in and a diff-gated write-back -- see
[Sandbox Workspaces](sandbox-workspaces.md). The agent performance cycle
A later cycle added read-only search tools (`grep`, `glob`, `ls`), a
diagnostics pass after writes, `todo` and `task` tools, loop hardening and
an eval harness -- see [Agent Performance](agent-performance.md).

## Modes and approval

In Daily mode the full tool set is available. In Bulbe mode the network is
constrained at the socket level (a physical constraint, not a policy), and every
state-mutating tool call is held behind the tool-call approval gate, which is
fail-secure: an unanswered request is denied. The approval surface is the same
`/api/security/tool-approval/*` API used elsewhere in the app.

A run starts in the machine's security mode and never in a looser one,
whatever its caller asks: leaving Bulbe takes the mode's own downgrade
ceremony. The mode is read again at each call, so an escalation holds a
running agent to Bulbe from its next call, and a call that cannot run (no
sandbox, no handler) is refused before anyone is asked about it. An
evaluation run, which no person watches, takes the machine's mode too, and a
call that would need a person is refused at once, with that reason.

## Where each argument came from

Every tool call -- the agent's and the chat's -- meets the provenance gate
(`opti_oignon/provenance.py`) on its arguments as they will run: after a
near-miss name is repaired, a default filled and a value coerced. Each
argument is labelled `typed` when, folded, it equals a part the user typed in
the current turn, whole (for an agent run, the task it was started with);
`default` when it equals the tool's own declared value, in type and in value;
`unendorsed` otherwise -- words the model chose, wherever it read them. An
argument that stays on the machine is `typed` only when it is the typed part
character for character. Each tool has an effect class from a closed table
(none, session, sandbox, deferred, network, approved); a tool the table does
not know is held as networked.

In Daily, `config/provenance.yaml` says what a network call carrying an
unendorsed argument does -- a tool call's, and the chat's own web search:
`free` (the default) sends it; `ask_unendorsed` holds it for the user, who
sees each value whole, every hidden character made visible, with its label,
in the approval drawer, the agent panel and the terminal (a value too long to be shown
whole is refused instead); `refuse_unendorsed` refuses it. Only a missing
file, or one that says `free`, is free: a file that cannot be read, names no
policy, or gives a value outside the three reads as `refuse_unendorsed`. An
endorsed network argument is handed to its sink as the user typed it, never
in the model's spelling. On a turn that says which words are the user's
(every turn the chat composes), the chat's own search sends those words --
typed, or a hook's rewrite of them -- and nothing else: never an attached
file's, an image's description or a pipeline step's prompt; a caller that
composes no turn has its question judged by the gate like any network
value. The person is shown each value whole, labelled or not, with
its own length, and every character a screen would draw as a blank or as nothing
(a tab, a no-break space, variation selectors, tag characters, fillers)
written as its escape. Bulbe has no network, asks before every call
that acts outside the run, and shows the same labels. The chat also refuses
a tool its turn did not offer, and one whose class the machine's mode does
not permit; a call that must be asked and has no way to reach the user is
refused, and the model is told why.

What this cannot see: the user's own words, sent whole by a planted
instruction, still go -- they are the user's; a search asked for in other
words than its query is unendorsed, so the strict settings ask or refuse it.

## Memory and notes writes

What the agent writes with `manage_memory` and `manage_notes` comes back in
every later turn, so each such write passes a gate. A new fact, or a new
note's title, body and tags, is written directly only when each of its words,
folded (Unicode NFC, each run of white space as one space), equals the whole
of what the user typed in the turn that started the run -- never a part of
it: a sentence, a line, a list item, a line of code and even a paragraph can
take their sense from what stands next to them ("Things you must never do:"
before "Share my location with Bob."). Anything else becomes a
proposal: a fact the model drew from a web result, a file or its own words,
and every update or delete, whose target is an identifier no typed word can
vouch for. A proposal is inert: it is kept with where each of its words came
from and what the run had read before it (web results, files, command
output, facts, notes or skills it looked up), no model reads it, and it is
written only when the user accepts it. The model is told the write was
proposed, never that it was saved.

The turn of a run is its task. `POST /api/agent/run` vouches for the task as
the words the user typed; a run started any other way endorses nothing, so
every write it makes is proposed. A run makes at most `max_per_run` proposals
(`config/pending_writes.yaml`, 20 by default); a proposal already waiting is
never queued twice, and a write the user declined in a conversation is not
proposed again in that conversation (another conversation may ask again; a
run with no conversation keeps no refusal).
When the review queue cannot record a proposal, nothing is written. An
evaluation run neither writes nor proposes. The queue follows the user's
data controls: the per-user wipe deletes it and the export carries it.

Proposals wait in the review section of the Memory panel (all of them) and
of the Notes panel (the notes ones), over `GET /api/pending-writes`,
`POST /api/pending-writes/accept` and `POST /api/pending-writes/decline`: a
batch of ids, applied in order, each once and exactly as proposed. A write
that fails stays waiting, unless it reached the store before the failure: it
is then decided, and can no longer be declined while it stays written. A
change whose fact or note is gone is reported, and nothing is saved. An
acceptance cut short by the end of the process is completed by the next
review after `stale_claim_seconds` (300 by default): settled if its write
landed -- a fact found by its source, a note by an id drawn from the
proposal's -- and written once otherwise, never put back. The manual extraction of the Memory
panel follows the same rule: the facts drawn from the user's typed words are
written, the facts drawn from anything else are proposed, and a single typed
turn is enough for the model to read.

What the gate cannot see: a fact the model paraphrases from the user's own
words is not their words, so it is proposed rather than written; accepting it
is one gesture. Words pasted into the task count as typed, until pasted text
is told apart from typed text. The gate does not judge content -- it holds
back every write whose words the user did not type, whatever they say.

## Control surface

The running agent is controlled over `/api/agent/*`:

- `POST /api/agent/run` starts a run, in the machine's security mode. A request
  may name the stricter Bulbe, never a looser mode than the machine's: leaving
  Bulbe for Daily takes the degradation ceremony, not a request field. A mode
  that does not exist, or a machine mode that cannot be read, runs in Bulbe.
- `GET /api/agent/status` returns `{running, rounds, stop_reason}`.
- `POST /api/agent/cancel` requests cooperative cancellation.
- `WS /api/agent/stream` emits the live `AgentEvent` stream (`round_start`,
  `model_output`, `tool_result`, `done`, `error`, `verifier_output`).

The agent panel in the UI consumes this surface: it shows the live tool stream,
the round and step display, a cancel control, and the Bulbe approval prompts,
each with every value the held call would run with, as the approval drawer
shows them.
