# Agent Performance

The Agent Performance cycle (AGT, S222-S231, released as v3.10.0) makes the
sandboxed agent loop measurably better at multi-step coding and filesystem
work with local models. Everything below rides the existing security
posture: every tool goes through the approval-gated, mode-aware dispatch,
and the disposable bubblewrap sandbox boundary is unchanged.

## Search tools: grep, glob, ls

Three read-only tools let the model explore the workspace without shell
round-trips. They are trusted host-side reads confined to the workspace
(the same path validation as `view`), with deterministic sorted output,
size caps and truncated flags, a binary sniff and a 1 MiB skip on `grep`,
and symlinks never followed or read. `glob` orders results by modification
time (newest first, name as the tiebreak); `ls` lists directories first,
then files, each name-sorted.

## Diagnostics after writes

After a successful `create_file` or `str_replace` on a `.py` or `.svelte`
file inside a real bubblewrap session, the loop runs a diagnostics pass and
appends findings to the tool result, so the model sees its own mistakes
immediately. Python goes through a linter ladder (ruff, else pyflakes, else
`py_compile`) executed inside the sandbox like any other command, under the
signed audit chain; Svelte gets a tag-balance check. Findings are capped
(25 lines / 4096 bytes), never block the write, and a clean write stays
byte-identical. Outside bwrap (for example the test container) the pass is
skipped entirely.

## Planning and delegation: todo, task

`todo` is a lightweight plan tracker the model updates as it works -- a
replacement-list of steps surfaced to the UI as `todo_updated` events with
`{todos, total, completed}`. It holds nothing at rest and is the first tool
available in Bulbe mode without a per-call prompt. `task` runs a bounded
depth-1 subagent: a child loop with a fresh context that returns only a
structured summary to the parent, the child's rounds debited from the
parent's budget so delegation cost stays visible and bounded.

## Loop hardening

Oversized tool observations are capped, and the full content spills to
`.agent/spill/` inside the workspace, retrievable with the read tools
(spill files are ordinary session writes under the diff walk). Old
observations are pruned deterministically behind a protect window that
keeps recent rounds intact. A doom-loop detector watches a sliding window
of repeated identical tool calls and intervenes. Conservative `str_replace`
misses get a hint instead of a silent failure, and a two-rounds reminder
keeps the model on the task. When the resource governor has admitted the
run, the truncation caps consume the admitted context size; otherwise
conservative static caps hold.

## The eval harness

The harness measures the loop end to end on your local models: the `micro`
suite is 12 sandbox-run micro-tasks auto-scored by tests passing inside the
sandbox. Admission rides the governor contract -- a task is admitted,
refused, or skipped, never silently downsized -- and degrades honestly when
the governor is absent. Results land in a dedicated store
(`data/agent_eval_results.db`) with honest provenance columns
(`governor_present`, `admitted`, `admitted_ctx`, `failure_class`).

Run it on the host:

```bash
bash scripts/run_agent_eval.sh --models qwen3:4b,qwen3:14b --suite micro
```

The API mirror is five endpoints under `/api/agent-eval` (`run`, `status`,
`results/{run_id}`, `history`, `cancel`), behind the same auth as the rest
of the API.

The numbers only mean something on a real machine: the model-in-the-loop
runs themselves, eviction effectiveness on real VRAM, and real bwrap spill
and diagnostics behaviour are host-assured, directed by
`HOST_SHAKEDOWN_S231.md` at the repository root. An optional side-by-side
opencode baseline script exists for the Route A gap question
(reference-only; it never simulates results).
