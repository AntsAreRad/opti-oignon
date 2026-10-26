# Chat

## Sending messages

Type your message in the input area and press `Ctrl+Enter` to send.
Responses stream in real time via WebSocket.

The chat interface supports:

- **Markdown rendering** of replies: headings, lists and task lists,
  quotes, tables, links and code blocks. Your own messages are shown as you
  typed them. Raw HTML in a reply is shown as text, images are never
  loaded (a reply shows "Image: alt (url)"), and links open in a new tab.
- **Code blocks** with their language, keywords and defined names coloured
  for Python, JavaScript and TypeScript, Rust, shell and SQL, and a Copy
  button in each block
- **Long messages** collapse: a long reply shows its first whole blocks, a
  long message of yours its first lines, each with "Show the rest"
- **Multi-turn conversations** with full history context
- **Conversation branching** -- fork a conversation at any message to
  explore alternative paths: select the message, then fork it from the
  branch explorer above the thread


## Questions before a search

In Daily mode, a call that would send words you did not type out of the
machine waits for your yes: a search the model words its own way, or words
it read in a page, a file or a tool result. The approval drawer shows the
call as it would run, each value whole with where it came from. A call
whose every value is one you typed, whole, or the tool's own default goes
without a question. A no, or no answer
within 30 seconds, refuses the call, and the reply goes on without tools
for the rest of that turn. The network policy that asks is
`ask_unendorsed`, in `config/provenance.yaml`; `refuse_unendorsed` refuses
such calls without asking, and `free` sends them unasked.


## Stopping a reply

Stop ends this conversation's reply and nothing else. Replies in other
conversations, and requests from a paired phone, go on.

- The reply on screen stops at once. The reasoning, consensus,
  self-correction and tool stages stop before their next model call: a
  call already in flight in one of those stages is not cut short, and the
  stop takes effect when it returns. Whether the model server itself stops
  working on a stopped call is not yet measured.
- The coding agent stops between phases and never writes files from a
  stopped reply. A test command already running finishes. The turn is
  recorded as stopped, with the files it wrote. A new `/code` message in
  the same conversation waits until the previous one has stopped. A reply
  that has no conversation does not run the coding agent.
- A tool waiting for your approval -- every tool call in Bulbe mode, and in
  Daily a network call carrying words you did not type -- is denied and
  withdrawn from the approval list when its reply is stopped, and an
  approval given after the Stop does not run the tool.
- Tools that already ran stay in the conversation's tool history, so the
  next reply knows what they did.
- Closing the tab stops the reply as Stop does.
- A stopped reply is not saved; the page keeps the partial text. Two
  exceptions remain: in self-correction, the first draft, written before
  the correction starts and not shown, is saved even when the correction
  is stopped; in a pipeline of several steps, each step that finished
  before the Stop keeps its output.
- The emergency stop still stops everything at once.


## Pipelines

Opti-Oignon routes each query through one of nine pipeline types,
selected automatically based on query analysis:

- **Direct** -- simple question-answer, single model call
- **Chain-of-thought** -- step-by-step reasoning before answering
- **Tools** -- function calling with sandboxed filesystem tools
- **Think+tools** -- reasoning followed by tool use
- **Code verification** -- generates code then validates it
- **Web search** -- augments the response with web results. The query
  goes through the ddgs package, which sends it to Wikipedia and to one
  or more search engines it picks at random. The results reach the model
  wrapped as untrusted data, in front of your question, never in a system
  message. It sends the words you typed and nothing else; words you pasted,
  or a hook's rewrite of yours, wait for your yes first. Refused in Bulbe
  mode and while the search kill switch is engaged.
- **Reasoning** -- advanced strategies (Decompose-and-Solve,
  Tree-of-Thought, Self-Consistency)
- **Consensus** -- multiple models vote on the best answer
  (Best-of-N, Weighted Vote, LLM Merge)
- **Self-correction** -- iterative refinement loop

The pipeline is shown in the response header. The pipelines themselves
are managed in the chat's **Pipelines** panel.


## Smart routing

The smart router selects which model handles each query. It considers:

- **Capability profiles** -- 15+ numeric dimensions per model (coding,
  math, creativity, reasoning, etc.)
- **Context window** -- ensures the conversation fits the model's limit
- **Model health** -- excludes slow or unresponsive models
- **Learned preferences** -- ML-based routing trained on your feedback
  history (thumbs up/down)

In Balanced and Power presets, routing enables **cascading inference**:
a fast small model handles simple queries, and only complex ones are
escalated to larger models.


## Coding agent

For code-related tasks, Opti-Oignon can activate its autonomous coding
agent. The agent operates in a sandboxed environment and follows this
loop:

1. Plans the task based on your request
2. Generates code
3. Runs tests in the sandbox
4. Auto-fixes failures (up to a configurable retry limit)
5. Presents unified diffs for your review

The apply phase always requires explicit human approval. No code
changes reach your filesystem without your confirmation.

Working memory persists context across steps, and cascading
auto-escalates to stronger models on repeated failures.


## Conversation management

- **New chat:** `Ctrl+N`
- **Search chats, pages, commands and settings:** `Ctrl+K` (the command
  palette; the sidebar's Search opens it too)
- **Export conversation:** `Ctrl+Shift+E`
- **Toggle sidebar:** `Ctrl+B`

Conversations are stored locally in SQLite (encrypted with SQLCipher
when available). You can export and import conversations via the
backup system (Workshop > Backup > Backup & restore, or the `oo backup`
CLI).

Deleting a conversation from the app stops its live replies -- even when
the deletion is then refused --, removes its messages and forgets what
the platform drew from it: the onion memory's copy, the facts the model
drew from it -- by the extraction you asked of it, or by the automatic
capture --, its branches and their turns, the facts proposed in it, the
feedback given in it, what the semantic cache kept of it, under the
global scope too, its links to projects -- the outputs it gave stay with
the project --, the artifacts, tool calls and coding session the process
held of it, and the agent runs it started. While another process writes
to the conversation store, the deletion fails before anything is
forgotten. What the databases delete is overwritten, whichever write
freed it, and the write-ahead logs that held its words are emptied (a
read in progress keeps one until the next deletion); the run journal
writes its days again without the conversation's runs. Within the
process, the stores then refuse anything new about the conversation, a
write already on its way included, and its tool calls are no longer
recorded; a coding session a reply that outlived the deletion opens
again expires with its sandbox; another process sharing the stores is
not held back. A store that keeps the conversation's words and cannot
forget -- locked by another process, or not opened -- refuses the
deletion and names itself; the conversation stays, every store takes its
writes again, and a deletion asked again finishes the work. A
conversation that comes back under its id, a peer's later edit winning
over its deletion, is taken again by every store. The facts you made
yours stay: written by hand, asked of the agent, a proposal you
accepted, or a fact drawn from it that you wrote again, edited or
confirmed; so do the facts of the memory store of earlier versions, and
those the automatic capture kept before it said which conversation it
drew from -- where they came from is not known. A deletion does not yet
forget the response cache (which serves an answer for an hour and keeps
it until a later lookup cleans it up), or, with the sync package, the
change feed's copies; and a conversation an older version deleted keeps
what these stores drew from it, its words perhaps still in their free
pages: nothing sweeps them yet.


## Memory

The memory panel lists the facts the assistant keeps about you; each
turn's memory block is composed from them alone. Deleting a fact moves
it to the trash: it leaves the window at once, and the panel's trash
shows it with the day it is erased, until you restore it. After 30 days
(`trash_days` in `opti_oignon/config/memory.yaml`) it is erased for
good: its text overwritten, and paired devices told. A fact in the trash
is restored before it is edited. You can erase a fact in the trash at
once, or empty the trash; Clear all moves every fact to the trash. A
fact deleted more than 30 days before this version is erased at its
first start.

The memory store of earlier versions (`memories.db`) is moved into the
facts at the first start, and then emptied: a fact deleted since never
comes back from it. A fact that store kept out of service lands in the
trash, unless that store also kept it in service, or you confirmed it as
yours since.


## Conversation branches

Fork any conversation at a specific message to explore alternative
responses without losing the original thread: select the message, then
use Fork in the branch explorer above the thread. Branches are listed
there and can be merged or deleted independently.
