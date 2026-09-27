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
- In Bulbe mode, a tool waiting for approval is denied and withdrawn from
  the approval list when its reply is stopped, and an approval given after
  the Stop does not run the tool.
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
  wrapped as untrusted data, still inside a system message. Refused in
  Bulbe mode and while the search kill switch is engaged.
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


## Conversation branches

Fork any conversation at a specific message to explore alternative
responses without losing the original thread: select the message, then
use Fork in the branch explorer above the thread. Branches are listed
there and can be merged or deleted independently.
