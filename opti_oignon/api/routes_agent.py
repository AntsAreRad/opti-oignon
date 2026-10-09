#!/usr/bin/env python3
"""Live API route for the sandboxed agent loop (Theme 3 / Odysseus Core).

Wires the agent half of Odysseus into a running agent: the streaming loop
(``agent.loop``), the per-mode tool set (``agent.tools``), the approval-gated
SKILL.md registry and its ``manage_skills`` tool (``agent.skills``), the
teacher-draft publish path, and the working-memory block. It exposes the
contract the agent panel consumes (frontend api/agent.ts):

- ``GET  /api/agent/status``  -> ``{running, rounds, stop_reason}``
- ``POST /api/agent/cancel``  -> ``{cancelled}``
- ``POST /api/agent/run``     -> start a run (the wiring entry point)
- ``WS   /api/agent/stream``  -> a live AgentEvent JSON stream

It also mounts the SKILL.md registry surface that the skills-manager panel
consumes (frontend api/skills.ts, Goal 0, closing the carry-over):

- ``GET    /api/agent/skills``                       -> ``{skills: [...]}``
- ``GET    /api/agent/skills/{category}/{name}``     -> one skill, with its body
- ``POST   /api/agent/skills/{category}/{name}/publish`` -> publish a draft
- ``DELETE /api/agent/skills/{category}/{name}``     -> ``{deleted}``

The skills routes read the on-disk ``SkillRegistry``; the path segments are
sanitised and contained inside the registry itself, and the handlers map errors
to HTTP codes rather than raising into the response path.

Bulbe approvals are NOT duplicated here: the tool-call approval surface is
reused verbatim from the existing ``/api/security/tool-approval/*`` API, which
the loop's ``approval_fn`` and the ``manage_skills`` gate already drive.

Design for testability and isolation: the run engine, ``AgentRunManager``, is a
plain object (threading plus an event broadcast) with no web dependency, so the
end-to-end wiring is exercised in isolation with an injected model client and
sandbox. The FastAPI surface is a thin wrapper, guarded so the module loads even
where FastAPI is absent; the agent imports are likewise guarded.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any, Callable

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Guarded agent imports (isolatable: the agent package needs no web stack).
try:
    from opti_oignon.agent import loop as agent_loop
    from opti_oignon.agent import skills as agent_skills
    from opti_oignon.agent import tools as agent_tools

    _AGENT_OK = True
except Exception:  # pragma: no cover - constrained environments only
    agent_loop = None  # type: ignore[assignment]
    agent_skills = None  # type: ignore[assignment]
    agent_tools = None  # type: ignore[assignment]
    _AGENT_OK = False

# Emergency-stop admission guard (a stopped system refuses honestly)
try:
    from opti_oignon import emergency_stop as _emergency_stop
except Exception:  # pragma: no cover - constrained environments only
    _emergency_stop = None  # type: ignore[assignment]


# Teacher escalation wiring (driver-side). The chokepoints themselves live
# in ``agent.teacher`` (decision + escalation, never raising) and
# ``agent.skills.publish_teacher_draft`` (human gate first, sandbox-tested)
# and keep their own contracts; the driver only decides WHEN to consult
# them. Event kinds are stable strings: clients key on them.
EVENT_TEACHER_GUIDANCE = "teacher_guidance"
EVENT_TEACHER_DRAFT = "teacher_draft"

# The failure context handed to the teacher is size-bounded here; the
# escalation module wraps it as untrusted data itself.
_TEACHER_CONTEXT_MAX_CHARS = 4000
_TEACHER_FAILED_RESULTS_TAIL = 5


def _clip_teacher_text(text: Any) -> str:
    """Clip a context string to the bounded tail (the freshest part)."""
    value = str(text or "")
    if len(value) <= _TEACHER_CONTEXT_MAX_CHARS:
        return value
    return value[-_TEACHER_CONTEXT_MAX_CHARS:]


def _teacher_failure_observations(result: Any) -> str:
    """The bounded tail of failed tool observations from a run result."""
    parts: list[str] = []
    tail = list(getattr(result, "tool_results", []) or [])
    for item in tail[-_TEACHER_FAILED_RESULTS_TAIL:]:
        if bool(getattr(item, "executed", False)):
            continue
        text = str(getattr(item, "observation", "") or "")
        if text:
            parts.append(text)
    return _clip_teacher_text("\n".join(parts))


# The run engine (no web dependency)

# The tools whose writes persist into stores later turns read back. A run
# binds them to a pending-write gate of its own (opti_oignon.pending_writes).
_PERSISTENT_WRITES = frozenset({"manage_memory", "manage_notes"})

# The tools whose results bring the run something to read: the web, files,
# a command's output, a subtask's answer. A write tool's own result is only
# its confirmation, so it is never named as read before a proposal.
_READING_TOOLS = frozenset({"web_search", "view", "grep", "glob", "ls", "bash", "task"})

# The read actions of the state tools: what they return (facts, note bodies,
# skill bodies) is read as surely as a web page, so each counts as read.
_READ_ACTIONS = {
    "manage_memory": frozenset({"list", "get"}),
    "manage_notes": frozenset({"list", "get"}),
    "manage_skills": frozenset({"list", "index", "view", "view_ref", "search"}),
}


class AgentRunManager:
    """Drives one agent run at a time and fans its events out to subscribers.

    A run executes ``agent.loop.run`` on a background thread with a context-bound
    ``manage_skills`` handler, memory and notes writes bound to a pending-write
    gate of the run's own, and, optionally, the skills most relevant to the
    task prepended as untrusted context. ``status`` / ``cancel`` are safe to call
    concurrently. Cancellation is cooperative: it sets a flag the loop checks
    between rounds (``should_continue``), so the run stops cleanly. Subscribers
    are plain callables receiving JSON payloads, which keeps the engine free of
    any event-loop dependency; the WebSocket endpoint adapts that to asyncio.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._rounds = 0
        self._stop_reason = ""
        self._subscribers: set[Callable[[str], None]] = set()
        # ATL-02: the conversation-bound SandboxToolSession this run
        # attached, if any. One run at a time, so a single slot suffices;
        # detached (never destroyed) when the run ends.
        self._owned_sandbox: Any = None
        # The active run's task, mode and gate bindings: the post-run
        # teacher hook forwards these to the gated publish entry so a
        # draft is judged by the run's own human gate, never a substitute.
        self._teacher_ctx: dict[str, Any] = {}
        # The tools whose results the active run has read, each once, in
        # the order first read: what a proposal says the agent had seen.
        self._read: list[str] = []

    # status / control

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "rounds": self._rounds,
                "stop_reason": self._stop_reason,
            }

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def rounds(self) -> int:
        with self._lock:
            return self._rounds

    def cancel(self) -> dict[str, bool]:
        """Request cancellation of the active run (cooperative, fail-safe)."""
        with self._lock:
            was_running = self._running
            if was_running:
                self._cancel.set()
        return {"cancelled": bool(was_running)}

    # event subscription (decoupled from asyncio)

    def subscribe(self, callback: Callable[[str], None]) -> Callable[[str], None]:
        with self._lock:
            self._subscribers.add(callback)
        return callback

    def unsubscribe(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            self._subscribers.discard(callback)

    def _broadcast(self, payload: str) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for cb in subscribers:
            try:
                cb(payload)
            except Exception:  # pragma: no cover - a bad subscriber must not break the run
                logger.debug("agent stream subscriber failed", exc_info=True)

    def _on_event(self, event: Any) -> None:
        try:
            rnd = int(getattr(event, "round", 0))
            with self._lock:
                if rnd > self._rounds:
                    self._rounds = rnd
            if getattr(event, "kind", "") == "tool_result":
                self._note_read(getattr(event, "data", {}) or {})
            payload = json.dumps(
                {
                    "kind": getattr(event, "kind", ""),
                    "round": rnd,
                    "data": getattr(event, "data", {}) or {},
                }
            )
        except Exception:  # pragma: no cover - defensive
            return
        self._broadcast(payload)

    def _note_read(self, data: dict) -> None:
        """Record a reading tool whose result the run read; one that did not run returned nothing to read."""
        name = data.get("tool_name")
        if not data.get("executed") or not isinstance(name, str) or not name:
            return
        if name not in _READING_TOOLS:
            return
        self._record_read(name)

    def _record_read(self, name: str) -> None:
        with self._lock:
            if name not in self._read:
                self._read.append(name)

    def _reading(self, name: str, handler: Callable[[dict], str]) -> Callable[[dict], str]:
        """``handler`` that records ``name`` as read each time it serves one of its read actions."""
        actions = _READ_ACTIONS.get(name, frozenset())

        def wrapped(arguments: dict) -> str:
            out = handler(arguments)
            if str((arguments or {}).get("action", "")).strip().lower() in actions:
                self._record_read(name)
            return out

        return wrapped

    def _read_so_far(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._read)

    def _gate_persistent_writes(
        self,
        handlers: dict[str, Any],
        *,
        task: str,
        turn_origin: str,
        conversation_id: str,
        run_id: str,
        user_id: str | None,
    ) -> dict[str, Any]:
        """Bind the run's persistent-write tools to a pending-write gate of the run's own.

        The task's typed units endorse a write when the caller vouches that
        the task is the words the user typed (``turn_origin`` typed, as the
        agent route does); a run no caller vouches for endorses nothing.
        Every other write is proposed to the user, naming the tools the run
        had read before it. A gate that cannot be built withdraws the write
        tools from the run rather than leave them writing.
        """
        if not any(name in handlers for name in _PERSISTENT_WRITES):
            return handlers
        try:
            from opti_oignon import pending_writes

            gate = pending_writes.WriteGate(
                pending_writes.Endorsers.for_turn(task, turn_origin),
                conversation_id=conversation_id,
                run_id=run_id,
                read=self._read_so_far,
                user_id=user_id,
            )
            bound = agent_tools.bind_write_gate(handlers, gate)
            return {name: self._reading(name, handler) if name in _PERSISTENT_WRITES else handler
                    for name, handler in bound.items()}
        except Exception:
            logger.warning("pending-write gate unavailable; the run's write tools are withdrawn", exc_info=True)
            return {name: handler for name, handler in handlers.items() if name not in _PERSISTENT_WRITES}

    # run lifecycle

    def start(
        self,
        task: str,
        *,
        model_client: Any,
        mode: str | None = None,
        conversation_id: str = "",
        sandbox: Any = None,
        system_prompt: str = "",
        approval_fn: Callable[[str, str, dict], bool] | None = None,
        memory_provider: Callable[..., str] | None = None,
        memory_query: str | None = None,
        user_id: str | None = None,
        include_memory: bool = True,
        verify: bool = False,
        registry: Any = None,
        approval_manager: Any = None,
        consult: bool = True,
        max_rounds: int | None = None,
        turn_origin: str = "legacy",
    ) -> dict[str, Any]:
        """Assemble and launch a run; refuse if one is already running.

        ``mode`` is what the caller asks for; the run gets the machine's mode
        when none is asked, and never a looser one. ``turn_origin`` is who
        wrote ``task``, in the turn-origin grammar: the caller that received
        the user's own typed words says typed, and the task, whole, then
        endorses the run's memory and notes writes and the arguments of its
        tool calls. Any other caller leaves it legacy: every such write is
        proposed, and no argument of the run is endorsed.
        """
        if not _AGENT_OK:
            return {"started": False, "reason": "agent_unavailable"}
        mode = _run_mode(mode)
        with self._lock:
            if self._running:
                return {"started": False, "reason": "already_running"}
            self._cancel.clear()
            self._rounds = 0
            self._stop_reason = ""
            self._running = True
        try:
            tool_set = agent_tools.build_tool_set(mode)
            handlers = dict(tool_set.tool_handlers)
            # ATL-02: when the conversation is bound to a workspace,
            # attach that workspace's SandboxToolSession and inject it into
            # the run (and so into dispatch.dispatch_tool_call) instead of
            # the per-run create/destroy. Explicit binding only: no bound
            # workspace means sandbox stays as passed (usually None) and the
            # dispatch refuses sandboxed tools exactly as before. The
            # attached session is detached (never destroyed) when the run
            # ends; the set_sandbox_mode lockout rides attach/detach.
            owned_sandbox = None
            if sandbox is None and conversation_id:
                try:
                    from opti_oignon import sandbox_workspace as _ws

                    owned_sandbox = _ws.attach_session_for_conversation(
                        conversation_id
                    )
                except Exception:  # pragma: no cover - binding is optional
                    logger.debug(
                        "workspace binding resolution failed", exc_info=True
                    )
                if owned_sandbox is not None:
                    sandbox = owned_sandbox
            with self._lock:
                self._owned_sandbox = owned_sandbox
            # Bind manage_skills (Daily only) to this run's conversation, sandbox,
            # and gate so its writes go through the right human approval.
            # The person decides inside the handler, which may take the
            # approval's whole timeout: a yes counts only if the machine is
            # still Daily once they answered.
            if agent_tools.TOOL_MANAGE_SKILLS in handlers:
                handlers[agent_tools.TOOL_MANAGE_SKILLS] = self._reading(
                    agent_tools.TOOL_MANAGE_SKILLS,
                    agent_skills.make_manage_skills_handler(
                        registry=registry,
                        approval_fn=_still_daily(approval_fn, mode, approval_manager),
                        sandbox=sandbox,
                        conversation_id=conversation_id,
                        manager=approval_manager,
                    ),
                )
            with self._lock:
                self._read = []
            handlers = self._gate_persistent_writes(
                handlers,
                task=task,
                turn_origin=turn_origin,
                conversation_id=conversation_id,
                run_id=uuid.uuid4().hex[:12],
                user_id=user_id,
            )
            native = tool_set.native_tools()
            prompt = system_prompt or agent_tools.system_prompt_section_for(mode)
            # Consult learned procedures relevant to the task; wrapped as untrusted.
            if consult:
                consultation = agent_skills.consult_skills(task, registry=registry)
                if consultation.block:
                    prompt = prompt + "\n\n" + consultation.block
        except Exception:
            with self._lock:
                self._running = False
                self._stop_reason = "error"
            self._detach_owned_sandbox()
            logger.exception("agent run setup failed")
            return {"started": False, "reason": "setup_error"}

        with self._lock:
            self._teacher_ctx = {
                "task": task,
                "mode": mode,
                "conversation_id": conversation_id,
                "approval_fn": approval_fn,
                "approval_manager": approval_manager,
                "sandbox": sandbox,
            }
        kwargs: dict[str, Any] = dict(
            task=task,
            model_client=model_client,
            sandbox=sandbox,
            mode=mode,
            conversation_id=conversation_id,
            system_prompt=prompt,
            tools=native,
            approval_fn=approval_fn,
            tool_handlers=handlers,
            include_memory=include_memory,
            memory_provider=memory_provider,
            memory_query=memory_query if memory_query is not None else task,
            user_id=user_id,
            verify=verify,
            provenance=_run_provenance(task, turn_origin, approval_fn, conversation_id),
        )
        if max_rounds is not None:
            kwargs["max_rounds"] = max_rounds
        self._thread = threading.Thread(target=self._run, kwargs=kwargs, daemon=True)
        self._thread.start()
        return {"started": True}

    def _run(self, **kwargs: Any) -> None:
        try:
            result = agent_loop.run(
                on_event=self._on_event,
                should_continue=lambda: not self._cancel.is_set(),
                **kwargs,
            )
            with self._lock:
                self._stop_reason = getattr(result, "stop_reason", "")
                self._rounds = getattr(result, "rounds", self._rounds)
            self._teacher_post_run(result)
        except BaseException:  # the loop is built not to raise; be defensive anyway
            with self._lock:
                self._stop_reason = "error"
            logger.exception("agent run thread crashed")
        finally:
            self._detach_owned_sandbox()
            with self._lock:
                self._running = False

    def _detach_owned_sandbox(self) -> None:
        """Release the conversation-bound session, never destroying it.

        detach() re-enables the tools the set_sandbox_mode lockout disabled
        and leaves the workspace and its files intact: the binding owns the
        workspace lifetime, not the run.
        """
        with self._lock:
            owned = self._owned_sandbox
            self._owned_sandbox = None
        if owned is None:
            return
        try:
            owned.detach()
        except Exception:  # pragma: no cover - release must not raise
            logger.debug("workspace detach failed", exc_info=True)

    # Teacher escalation (post-run driver hook)

    @staticmethod
    def _teacher_policy_opt_in() -> Any:
        """The escalation policy, armed only by an explicit opt-in.

        Waking the escalation path is a deployment decision: the agent
        configuration must load AND carry an explicit truthy
        ``teacher.enabled`` flag. An absent or falsy flag, or an
        unreadable configuration, leaves the path dormant (None) -- the
        escalation module's own policy default is deliberately not
        consulted here.
        """
        try:
            from opti_oignon.agent import config_loader as agent_config

            cfg = agent_config.get_agent_config()
            teacher_cfg = dict(getattr(cfg, "teacher", {}) or {})
            if not bool(teacher_cfg.get("enabled", False)):
                return None
            return cfg.teacher_policy()
        except Exception:  # an unreadable configuration stays dormant
            logger.debug("teacher opt-in resolution failed", exc_info=True)
            return None

    @staticmethod
    def _teacher_estop_blocked() -> bool:
        """True when the emergency stop is engaged or indeterminable.

        The hook calls a model and may submit a draft to the approval
        gate, so an engaged stop skips it -- and an unavailable stop
        module skips it too: an indeterminable stop state never wakes
        the path (fail closed).
        """
        if _emergency_stop is None:
            return True
        try:
            return bool(_emergency_stop.is_stopped())
        except Exception:  # pragma: no cover - defensive
            return True

    def _teacher_post_run(self, result: Any) -> None:
        """Consult the pinned teacher chokepoints after the loop returns.

        Armed only by the explicit configuration opt-in; skipped for a
        cancelled run and under an engaged (or indeterminable) emergency
        stop. The guidance surfaces as a run event; a proposed draft is
        submitted only through the gated publish entry, carrying the
        run's own approval gate, sandbox, conversation and approval
        manager, and only in the daily mode (mirroring the skill tool's
        exposure). Never raises into the run thread.
        """
        try:
            with self._lock:
                ctx = dict(self._teacher_ctx)
            policy = self._teacher_policy_opt_in()
            if policy is None:
                return
            if self._cancel.is_set():
                return
            if self._teacher_estop_blocked():
                return
            from opti_oignon.agent import teacher as agent_teacher

            escalator = agent_teacher.TeacherEscalator(policy=policy)
            decision = escalator.should_escalate(result)
            if not getattr(decision, "escalate", False):
                return
            model = str(getattr(policy, "teacher_model", "") or "")
            client = _OllamaModelClient(model) if model else None
            outcome = escalator.escalate(
                str(ctx.get("task", "")),
                attempts=_clip_teacher_text(
                    getattr(result, "final_text", "")
                ),
                observations=_teacher_failure_observations(result),
                teacher_client=client,
            )
            rounds = int(getattr(result, "rounds", 0) or 0)
            self._on_event(agent_loop.AgentEvent(
                kind=EVENT_TEACHER_GUIDANCE,
                round=rounds,
                data={
                    "escalated": bool(getattr(outcome, "escalated", False)),
                    "reason": str(getattr(outcome, "reason", "")),
                    "has_draft": getattr(outcome, "draft", None) is not None,
                    "teacher_model": str(
                        getattr(outcome, "teacher_model", "")
                    ),
                },
            ))
            draft = getattr(outcome, "draft", None)
            if draft is None or not bool(getattr(outcome, "escalated", False)):
                return
            # The machine's mode at publication, never looser than the run's:
            # a machine that escalated during the run publishes nothing, and
            # neither does one that escalated while a person decided.
            run_mode = str(ctx.get("mode", "")).strip().lower() or None
            if run_mode is None or _run_mode(run_mode) != "daily":
                return
            publication = agent_skills.publish_teacher_draft(
                draft,
                approval_fn=_still_daily(ctx.get("approval_fn"), run_mode, ctx.get("approval_manager")),
                sandbox=ctx.get("sandbox"),
                conversation_id=str(ctx.get("conversation_id", "")),
                manager=ctx.get("approval_manager"),
            )
            self._on_event(agent_loop.AgentEvent(
                kind=EVENT_TEACHER_DRAFT,
                round=rounds,
                data={
                    "published": bool(
                        getattr(publication, "published", False)
                    ),
                    "reason": str(getattr(publication, "reason", "")),
                    "name": str(getattr(draft, "name", "")),
                },
            ))
        except Exception:  # the hook never breaks the run
            logger.debug("teacher post-run hook failed", exc_info=True)

    def join(self, timeout: float | None = None) -> None:
        """Wait for the run thread to finish (used by tests)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)


# Module-level engine (one running agent per process; reset for tests)

_MANAGER: AgentRunManager | None = None


def get_run_manager() -> AgentRunManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = AgentRunManager()
    return _MANAGER


def reset_run_manager() -> None:
    global _MANAGER
    _MANAGER = None


# Model-client adapter (backend glue; guarded)


class _OllamaModelClient:
    """The model client a run is built over: the registry's stream client.

    The name is kept for the resolver below. Each turn is one request
    through the inference registry, tool schemas travelling as an engine
    option, handed to the loop as ``{"message": {"content", "tool_calls"}}``
    chunks, the shape it reads. The import is lazy so this module loads
    without the registry, exactly as it loaded without the client before.
    """

    def __init__(self, model: str, *, host: str | None = None) -> None:
        self._model = model
        self._host = host

    def stream(self, messages: list[dict[str, Any]], tools: Any = None):
        from opti_oignon.registry_clients import ModelStreamClient

        yield from ModelStreamClient(self._model, host=self._host).stream(messages, tools)


def _resolve_model_client(model: str | None) -> Any:
    """Build a model client for a run, or None when none can be resolved."""
    if not model:
        return None
    try:
        return _OllamaModelClient(model)
    except Exception:  # pragma: no cover - defensive
        return None


# Named refusal reasons for the run entry's model-capability gate. Stable
# strings: clients key on them and tests pin them by identity.
REASON_MODEL_NOT_TOOL_CAPABLE = "model_not_tool_capable"
REASON_TOOL_CAPABILITY_UNAVAILABLE = "tool_capability_unavailable"


def _model_capability_refusal(model: str) -> str | None:
    """Why ``model`` must not drive the tool loop, or None when it may.

    The agent loop is tool-bound by construction, so the run entry poses
    the tool-calling requirement itself. The verdict is the capability
    manifest's public predicate -- the single source of truth, never a
    local reimplementation: a model with an explicit negative verdict is
    refused by name, and a model with no profile passes (the textual
    fallback protocol drives tools for models without native function
    calling). Fail-secure: when the predicate cannot be imported the
    capability is indeterminable, and an indeterminable capability under
    this intrinsic requirement refuses by name rather than silently
    starting a loop whose model may answer tool-less.
    """
    try:
        from opti_oignon.capability_manifest import model_tool_capable
    except Exception:
        return REASON_TOOL_CAPABILITY_UNAVAILABLE
    if not model_tool_capable(model):
        return REASON_MODEL_NOT_TOOL_CAPABLE
    return None


def _resolve_memory_provider() -> Callable[..., str] | None:
    """The working-memory block provider, guarded."""
    try:
        from opti_oignon.memory import working_memory_block

        return working_memory_block
    except Exception:  # pragma: no cover - memory backend optional here
        return None


def _run_mode(requested: str | None) -> str:
    """The mode a run gets: the machine's security mode when the request
    names none, and never a looser one. Leaving Bulbe for Daily takes the
    degradation ceremony of ``security_mode``, never a field of a request.
    An unknown mode, or a machine mode that cannot be read, is Bulbe."""
    try:
        from opti_oignon.agent import allowlists
    except Exception:
        return "bulbe"
    return allowlists.floor_mode(requested)


def _still_daily(approval_fn: Any, run_mode: str | None, manager: Any = None) -> Any:
    """The run's approval gate, read again once a person has answered: a yes counts only if the machine is still Daily.

    A person may take up to the approval's timeout to decide, and the machine
    may escalate meanwhile. A run with no gate of its own asks the approval
    queue (``manager``, or the default one), the entry's own default, and
    its answer is read the same way.
    """
    if not callable(approval_fn):

        def approval_fn(conversation_id: str, tool_name: str, arguments: dict | None = None, **kwargs: Any) -> bool:
            try:
                from opti_oignon.agent import allowlists
            except Exception:
                return False
            return allowlists.request_approval(conversation_id, tool_name, arguments, manager=manager, **kwargs)

    def gate(*args: Any, **kwargs: Any) -> bool:
        if not run_mode:
            return False
        return bool(approval_fn(*args, **kwargs)) and _run_mode(run_mode) == "daily"

    gate.__wrapped__ = approval_fn  # type: ignore[attr-defined]
    return gate


def _run_provenance(task: str, turn_origin: str, approval_fn: Any, conversation_id: str) -> Any:
    """The run's provenance: the task, whole, when its caller vouches it is typed, and its way to ask the user.

    The way to ask is the run's approval function, shown the label of each
    argument and the call's class. None when the gate is not loaded or the
    provenance cannot be built; the dispatch then endorses nothing.
    """
    try:
        from opti_oignon import provenance
    except Exception:
        return None
    demand = None
    if callable(approval_fn):

        def demand(tool_name: str, arguments: dict, labels: dict | None = None, effect: str | None = None) -> bool:
            if provenance.accepts_labels(approval_fn):
                return bool(approval_fn(conversation_id, tool_name, arguments, labels=labels, effect=effect))
            return bool(approval_fn(conversation_id, tool_name, arguments))

    try:
        return provenance.TurnProvenance.of_turn(task, turn_origin, demand=demand)
    except Exception:
        logger.warning("the run's provenance cannot be built; no argument of the run is endorsed", exc_info=True)
        return None


def _resolve_approval_fn() -> Callable[[str, str, dict], bool] | None:
    """Reuse the existing tool-call approval gate as the loop's approval_fn."""
    try:
        from opti_oignon.agent import allowlists

        def _gate(conversation_id: str, tool_name: str, arguments: dict, labels: dict | None = None,
                  effect: str | None = None) -> bool:
            return allowlists.request_approval(conversation_id, tool_name, arguments, labels=labels, effect=effect)

        return _gate
    except Exception:  # pragma: no cover - defensive
        return None


# Skills registry logic (web-free; the FastAPI handlers are thin wrappers)
#
# These functions take a resolved registry and return plain payloads matching
# the contract frontend/src/lib/api/skills.ts defines. Keeping them off the
# FastAPI surface lets the isolation harness exercise list / view / publish /
# delete against a real SkillRegistry rooted at a temp dir, without the web
# stack. A missing skill or draft raises SkillNotFound, which the web layer maps
# to a 404; the path segments are sanitised and contained inside the registry
# itself, so a traversal payload resolves to a slug or to a clean miss.


class SkillNotFound(Exception):
    """A requested skill (published or draft) does not exist in the registry."""


def _skill_payload(skill: Any, *, with_body: bool = False, registry: Any = None) -> dict[str, Any]:
    """Serialise a skill for the wire: metadata, plus the body on a single view.

    A published skill also says what its bytes are to this device --
    ``local``, ``adopted``, or ``unadopted`` when it arrived from a paired
    device and was never shown and adopted here -- when the registry can say.
    """
    data = dict(skill.to_dict())
    if with_body:
        data["body"] = skill.body
    sync_state = getattr(registry, "sync_state", None)
    if sync_state is not None and data.get("status") == "published":
        try:
            data["sync_state"] = sync_state(skill.name, skill.category)
        except Exception:  # noqa: BLE001 - a state that cannot be read is left out
            logger.debug("skill sync state unreadable for %s/%s", skill.category, skill.name, exc_info=True)
    return data


def skills_list_payload(registry: Any, *, include_drafts: bool = True) -> dict[str, Any]:
    """The registry index payload: published skills, plus drafts by default."""
    skills = registry.list(include_drafts=include_drafts)
    return {"skills": [_skill_payload(s, registry=registry) for s in skills]}


def skill_view_payload(registry: Any, category: str, name: str) -> dict[str, Any]:
    """One skill with its full body; the published one if present, else its draft."""
    skill = registry.get(name, category, draft=False)
    if skill is None:
        skill = registry.get(name, category, draft=True)
    if skill is None:
        raise SkillNotFound(f"{category}/{name}")
    return _skill_payload(skill, with_body=True, registry=registry)


def skill_publish_payload(registry: Any, category: str, name: str) -> dict[str, Any]:
    """Promote a draft to published; raise SkillNotFound when no draft exists."""
    published = registry.publish(name, category)
    if published is None:
        raise SkillNotFound(f"{category}/{name}")
    return _skill_payload(published, with_body=True)


def skill_delete_payload(registry: Any, category: str, name: str) -> dict[str, bool]:
    """Delete a skill: the published one if present, else the draft. Always returns."""
    if registry.exists(name, category, draft=False):
        deleted = registry.delete(name, category, draft=False)
    else:
        deleted = registry.delete(name, category, draft=True)
    return {"deleted": bool(deleted)}


# FastAPI surface (guarded; thin wrappers over the engine)

try:
    import asyncio

    from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel

    router = APIRouter(prefix="/api/agent", tags=["agent"])

    class AgentRunRequest(BaseModel):
        task: str
        mode: str | None = None
        model: str = ""
        conversation_id: str = ""
        verify: bool = False
        consult: bool = True

    def _require_agent() -> None:
        if not _AGENT_OK:
            raise HTTPException(status_code=503, detail="Agent loop not available")

    @router.get("/status")
    def agent_status() -> dict[str, Any]:
        """The current run snapshot."""
        return get_run_manager().status()

    @router.post("/cancel")
    def agent_cancel() -> dict[str, bool]:
        """Cancel the active run (cooperative)."""
        return get_run_manager().cancel()

    @router.post("/run")
    def agent_run(request: AgentRunRequest) -> dict[str, Any]:
        """Start a run: the wiring entry point for the agent panel."""
        if _emergency_stop is not None:
            _emergency_stop.guard_http()  # Refused, not hung
        _require_agent()
        if not request.task.strip():
            raise HTTPException(status_code=422, detail="task cannot be empty")
        model_client = _resolve_model_client(request.model or None)
        if model_client is None:
            raise HTTPException(status_code=503, detail="No model client available")
        refusal = _model_capability_refusal(request.model)
        if refusal is not None:
            raise HTTPException(status_code=422, detail=refusal)
        # The task is the words the client sent as the user's own, as the
        # chat route holds its message: this route vouches for them as
        # typed, so the task, whole, endorses the run's writes and the
        # arguments of its calls -- never a sentence of it.
        result = get_run_manager().start(
            request.task,
            model_client=model_client,
            mode=_run_mode(request.mode),
            conversation_id=request.conversation_id,
            approval_fn=_resolve_approval_fn(),
            memory_provider=_resolve_memory_provider(),
            verify=request.verify,
            consult=request.consult,
            turn_origin="typed",
        )
        if not result.get("started"):
            raise HTTPException(status_code=409, detail=result.get("reason", "run_not_started"))
        return result

    @router.websocket("/stream")
    async def agent_stream(websocket: WebSocket) -> None:
        """Forward the live AgentEvent stream over a WebSocket."""
        await websocket.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str] = asyncio.Queue()

        def _push(payload: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, payload)

        manager = get_run_manager()
        manager.subscribe(_push)
        try:
            while True:
                payload = await queue.get()
                await websocket.send_text(payload)
        except WebSocketDisconnect:
            pass
        except Exception:  # pragma: no cover - transport hiccup
            logger.debug("agent stream send failed", exc_info=True)
        finally:
            manager.unsubscribe(_push)

    # Skills registry surface (Goal 0: closes the carry-over).
    #
    # Thin wrappers over the module-level skills logic below. Each resolves the
    # registry (503 when the agent package is absent), then delegates; a missing
    # skill maps to 404 (SkillNotFound), any other fault to 500. The logic is web
    # free so it is exercised in isolation; these wrappers are the FastAPI seam.

    def _resolve_skill_registry() -> Any:
        if not _AGENT_OK or agent_skills is None:
            raise HTTPException(status_code=503, detail="Skills registry not available")
        try:
            return agent_skills.get_skill_registry()
        except Exception:  # pragma: no cover - registry resolution is defensive
            logger.exception("skill registry resolution failed")
            raise HTTPException(status_code=503, detail="Skills registry not available")

    @router.get("/skills")
    def list_skills(include_drafts: bool = True) -> dict[str, Any]:
        """List published skills and, by default, the agent-proposed drafts."""
        registry = _resolve_skill_registry()
        try:
            return skills_list_payload(registry, include_drafts=include_drafts)
        except Exception:  # pragma: no cover - registry read is defensive
            logger.exception("skills list failed")
            raise HTTPException(status_code=500, detail="Failed to list skills")

    @router.get("/skills/{category}/{name}")
    def get_skill(category: str, name: str) -> dict[str, Any]:
        """One skill with its full body; the published one if present, else its draft."""
        registry = _resolve_skill_registry()
        try:
            return skill_view_payload(registry, category, name)
        except SkillNotFound:
            raise HTTPException(status_code=404, detail="Skill not found")
        except Exception:  # pragma: no cover - registry read is defensive
            logger.exception("skill fetch failed")
            raise HTTPException(status_code=500, detail="Failed to read skill")

    @router.post("/skills/{category}/{name}/publish")
    def publish_skill(category: str, name: str) -> dict[str, Any]:
        """Promote a draft to published: the human approval of an agent proposal."""
        registry = _resolve_skill_registry()
        try:
            return skill_publish_payload(registry, category, name)
        except SkillNotFound:
            raise HTTPException(status_code=404, detail="No draft to publish")
        except Exception:  # pragma: no cover - registry write is defensive
            logger.exception("skill publish failed")
            raise HTTPException(status_code=500, detail="Failed to publish skill")

    @router.delete("/skills/{category}/{name}")
    def delete_skill(category: str, name: str) -> dict[str, bool]:
        """Delete a skill: the published one if present, else the draft."""
        registry = _resolve_skill_registry()
        try:
            return skill_delete_payload(registry, category, name)
        except Exception:  # pragma: no cover - registry write is defensive
            logger.exception("skill delete failed")
            raise HTTPException(status_code=500, detail="Failed to delete skill")

except Exception:  # pragma: no cover - FastAPI absent (e.g. isolated tests)
    router = None  # type: ignore[assignment]


def register(app: Any) -> bool:
    """Register the agent router on a FastAPI app. Returns False when unavailable."""
    if router is None:
        return False
    try:
        app.include_router(router)
        return True
    except Exception:  # pragma: no cover - defensive
        logger.exception("failed to register agent router")
        return False
