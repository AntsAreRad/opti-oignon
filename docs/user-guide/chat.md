# Chat

## Sending messages

Type your message in the input area and press `Ctrl+Enter` to send.
Responses stream in real time via WebSocket.

The chat interface supports:

- **Markdown rendering** in both input and output
- **Code blocks** with syntax highlighting and copy button
- **Multi-turn conversations** with full history context
- **Conversation branching** -- fork a conversation at any message to
  explore alternative paths


## Pipelines

Opti-Oignon routes each query through one of nine pipeline types,
selected automatically based on query analysis:

- **Direct** -- simple question-answer, single model call
- **Chain-of-thought** -- step-by-step reasoning before answering
- **Tools** -- function calling with sandboxed filesystem tools
- **Think+tools** -- reasoning followed by tool use
- **Code verification** -- generates code then validates it
- **Web search** -- augments the response with web results
- **Reasoning** -- advanced strategies (Decompose-and-Solve,
  Tree-of-Thought, Self-Consistency)
- **Consensus** -- multiple models vote on the best answer
  (Best-of-N, Weighted Vote, LLM Merge)
- **Self-correction** -- iterative refinement loop

The pipeline is shown in the response header. You can override it in
Settings > Advanced > Pipelines.


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
- **Search conversations:** `Ctrl+K`
- **Export conversation:** `Ctrl+Shift+E`
- **Toggle sidebar:** `Ctrl+B`

Conversations are stored locally in SQLite (encrypted with SQLCipher
when available). You can export and import conversations via the
backup system (Settings > Advanced > Backup, or `oo backup` CLI).


## Conversation branches

Fork any conversation at a specific message to explore alternative
responses without losing the original thread. Branches are visible in
the sidebar and can be merged or deleted independently.
