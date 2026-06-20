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
(`web_search`, `manage_memory`, `manage_skills`) are handler-backed and follow
the same untrusted-output discipline. Since the workspace cycle (S209-S213) the
sandbox behind these tools can be a named, conversation-bound workspace with
explicit copy-in and a diff-gated write-back -- see
[Sandbox Workspaces](sandbox-workspaces.md). The agent performance cycle
(S228-S230) added read-only search tools (`grep`, `glob`, `ls`), a
diagnostics pass after writes, `todo` and `task` tools, loop hardening and
an eval harness -- see [Agent Performance](agent-performance.md).

## Modes and approval

In Daily mode the full tool set is available. In Bulbe mode the network is
constrained at the socket level (a physical constraint, not a policy), and every
state-mutating tool call is held behind the tool-call approval gate, which is
fail-secure: an unanswered request is denied. The approval surface is the same
`/api/security/tool-approval/*` API used elsewhere in the app.

## Control surface

The running agent is controlled over `/api/agent/*`:

- `POST /api/agent/run` starts a run.
- `GET /api/agent/status` returns `{running, rounds, stop_reason}`.
- `POST /api/agent/cancel` requests cooperative cancellation.
- `WS /api/agent/stream` emits the live `AgentEvent` stream (`round_start`,
  `model_output`, `tool_result`, `done`, `error`, `verifier_output`).

The agent panel in the UI consumes this surface: it shows the live tool stream,
the round and step display, a cancel control, and the Bulbe approval prompts.
