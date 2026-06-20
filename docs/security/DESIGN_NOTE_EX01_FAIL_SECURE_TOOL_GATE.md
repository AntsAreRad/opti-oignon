# Design note: fail-secure default for the tool-execution approval gate (EX-01)

Status: recorded design decision, not yet implemented. Raised in
AUDIT_FINDINGS_S184 (EX-01). This note captures the analysis and the proposed
direction so a future session can implement it deliberately.

## Summary

The tool-execution approval gate is **fail-open by construction**: when no
approval callable is wired, tool calls execute without any human gate. The
secure behaviour (block every tool call until human approval in Bulbe mode) is
arranged by the *caller* (`routes_chat`), not by the executor itself. Any caller
that does not wire the gate runs tools ungated, including in Bulbe mode. This
contradicts the project's max-security, fail-secure posture and should be
inverted: absent an explicit gate, the executor should consult the security
policy and default-deny side-effectful tool calls in Bulbe mode.

## Current behaviour (as of v3.6.0)

Gate resolution and enforcement live in `opti_oignon/tool_executor.py`,
`ToolExecutor._execute_tool` (around L341):

```python
hook = approval_fn if approval_fn is not None else self.pre_tool_call_hook
if hook is not None:
    approved = hook(tool_name, arguments)
    if not approved:
        return ToolCallResult(..., result="Tool call denied by approval gate",
                              success=False, ...)
    # on hook exception -> deny (fail-secure within the gate)
# if hook is None -> fall through and execute the tool with NO gate
```

Two things are already correct here:

- EX-02 ordering: an explicit per-invocation `approval_fn` takes precedence over
  the legacy process-wide `self.pre_tool_call_hook` attribute, so concurrent
  Bulbe sessions cannot clobber or drop each other's gate.
- Within the gate, a hook that raises is treated as a denial (fail-secure on
  error).

The gap is the `hook is None` branch: it executes the tool ungated.

The live request path arms the gate only in Bulbe mode. In
`opti_oignon/api/routes_chat.py`:

- `_approval_fn = None` is the default, documented as "None means no gate
  (Daily / no policy)" (around L436).
- The Bulbe gate is installed only when the security policy reports
  `tool_call_approval_required` (around L584), which `security_mode.ModePolicy`
  sets to `True` for `MODE_BULBE` and `False` for `MODE_DAILY`
  (`opti_oignon/security_mode.py`, L119/L139). The hook blocks each tool call
  until a human approves, with a short timeout that auto-denies.

So the production chat path is wired: in Bulbe mode it gates, in Daily mode it
does not. The risk is everything *outside* that one wiring site.

## Risk

1. **Security is opt-in per call site, not a default.** The executor's safe
   behaviour depends on every caller remembering to pass `approval_fn` (and to
   derive it from the policy). The default is the unsafe one.

2. **Concrete unwired caller: pipelines.** `opti_oignon/pipelines.py`
   (`run_pipeline`, around L488/L611) threads an `on_tool_call` *observation*
   callback through, but does **not** pass an `approval_fn` *gate*. A
   pipeline-driven tool call therefore reaches `_execute_tool` with `hook is
   None` and runs ungated -- even when the system is in Bulbe mode. The Bulbe
   gate armed in `routes_chat` does not cover this path.

3. **Future call sites inherit the hazard.** Any new agentic entry point,
   plugin, scheduled task, or sync path that calls
   `tool_executor.execute_with_tools(...)` / `agentic_executor.execute(...)`
   without wiring the gate will silently run tools ungated. A reviewer must
   notice the omission; the type system does not (the parameter is
   `Callable | None = None`).

4. **Inconsistent with the rest of the codebase.** The Veilid sync engine
   (`opti_oignon/veilid/sync_engine.py`, around L389) already implements the
   correct posture for sensitive records: anything not positively approved by an
   injected `approval_fn` is *deferred* (held, not applied) -- fail-secure. And
   `security_mode` itself fails secure at the mode level ("if the two sources
   disagree, default to Bulbe"). The tool-execution path is the outlier.

The current *production* exposure is limited because the primary chat path is
wired correctly, and the executor's filesystem tools run under the S73/S74
disposable sandbox regardless. But "fail-open unless the caller opts in" is the
wrong default for this project.

## Proposal

Invert the default: when no explicit gate is wired, the executor consults the
security policy itself.

1. **Tool side-effect taxonomy.** Classify every registered tool by side effect,
   e.g. `read_only`, `state_mutating`, `network`. Store it in the tool registry
   (a `side_effect` field on the tool descriptor). Unknown / unclassified tools
   are treated as unsafe.

2. **Default policy gate in `_execute_tool`.** Change the `hook is None` branch
   from "execute ungated" to "resolve a default gate from
   `security_mode.get_policy()`":
   - In Bulbe mode (or whenever `tool_call_approval_required` is set), deny tools
     classified as `state_mutating` or `network`, and allow `read_only` tools.
     Unknown side effect -> deny (default-deny on unknown).
   - In Daily mode, preserve current behaviour (ungated) unless a stricter
     project-wide default is chosen (see open question below).
   - Import `security_mode` lazily/optionally (as other modules do) so tests and
     standalone use do not require it.

3. **No double-gating.** The default policy gate must fire only when no explicit
   `approval_fn` is wired. EX-02's existing precedence already guarantees that an
   explicit human gate wins; the default gate is strictly the `hook is None`
   fallback.

Net effect: an unwired caller in Bulbe mode default-denies side-effectful tools
instead of running them, and the human-approval gate wired by `routes_chat`
continues to take precedence where present.

## Trade-offs and open questions

- **Misclassification / unknown tools.** Default-deny on unknown is the safe
  choice but can break legitimate read-only agentic use in Bulbe if a tool is
  unclassified. A complete, reviewed taxonomy mitigates this; it is a
  prerequisite, not an afterthought.
- **No-policy default (the core decision to settle).** When `security_mode` is
  unavailable (policy cannot be resolved), what is the default for
  side-effectful tools? Options: (a) treat absence-of-policy as Daily
  (ungated) -- least surprising, weakest; (b) treat absence-of-policy as
  deny-side-effectful -- strongest, but can surprise non-Bulbe/standalone users
  and tests. Recommendation leans toward (b) for a max-security project, gated
  behind the taxonomy so read-only flows still work, but this must be decided
  explicitly before implementation.
- **Hot-path coupling.** This adds a `security_mode` dependency to the tool
  execution path. Keep it a lazy import with a cached policy read to avoid a
  per-call cost.
- **Interactions.** Pairs naturally with the auth-hardening lot (A4 / S187) and
  the red-team bloc (Bloc 6), both of which touch policy and adversarial tool
  use. The taxonomy is also useful input to red-team scoring (which tools are
  "dangerous").

## Recommendation

Defer implementation to a dedicated hardening pass (auth lot S187, or the
red-team bloc), because it is a security-policy change to the tool-execution hot
path and requires (1) a reviewed tool side-effect taxonomy and (2) a settled
answer on the no-policy default. The decision is recorded here so it is not lost;
the smallest correct first step is the taxonomy plus flipping the `hook is None`
branch to consult the policy, with `read_only` allowed and everything else
denied in Bulbe.
