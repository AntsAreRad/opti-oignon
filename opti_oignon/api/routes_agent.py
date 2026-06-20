#!/usr/bin/env python3
"""Live API route for the sandboxed agent loop (S177, Theme 3 / Odysseus Core).

Wires the agent half of Odysseus into a running agent: the streaming loop
(``agent.loop``), the per-mode tool set (``agent.tools``), the approval-gated
SKILL.md registry and its ``manage_skills`` tool (``agent.skills``), the
teacher-draft publish path, and the S66 working-memory block. It exposes the
contract the agent panel consumes (frontend api/agent.ts):

- ``GET  /api/agent/status``  -> ``{running, rounds, stop_reason}``
- ``POST /api/agent/cancel``  -> ``{cancelled}``
- ``POST /api/agent/run``     -> start a run (the wiring entry point)
- ``WS   /api/agent/stream``  -> a live AgentEvent JSON stream

It also mounts the SKILL.md registry surface that the skills-manager panel
consumes (frontend api/skills.ts, S178 Goal 0, closing the S177 carry-over):

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

# S215: emergency-stop admission guard (a stopped system refuses honestly)
try:
    from opti_oignon import emergency_stop as _emergency_stop
except Exception:  # pragma: no cover - constrained environments only
    _emergency_stop = None  # type: ignore[assignment]


# The run engine (no web dependency)


class AgentRunManager:
    """Drives one agent run at a time and fans its events out to subscribers.

    A run executes ``agent.loop.run`` on a background thread with a context-bound
    ``manage_skills`` handler and, optionally, the skills most relevant to the
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
        # S210 (ATL-02): the conversation-bound SandboxToolSession this run
        # attached, if any. One run at a time, so a single slot suffices;
        # detached (never destroyed) when the run ends.
        self._owned_sandbox: Any = None

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

    # run lifecycle

    def start(
        self,
        task: str,
        *,
        model_client: Any,
        mode: str = "daily",
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
    ) -> dict[str, Any]:
        """Assemble and launch a run; refuse if one is already running."""
        if not _AGENT_OK:
            return {"started": False, "reason": "agent_unavailable"}
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
            # S210 (ATL-02): when the conversation is bound to a workspace,
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
            if agent_tools.TOOL_MANAGE_SKILLS in handlers:
                handlers[agent_tools.TOOL_MANAGE_SKILLS] = agent_skills.make_manage_skills_handler(
                    registry=registry,
                    approval_fn=approval_fn,
                    sandbox=sandbox,
                    conversation_id=conversation_id,
                    manager=approval_manager,
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
        except BaseException:  # the loop is built not to raise; be defensive anyway
            with self._lock:
                self._stop_reason = "error"
            logger.exception("agent run thread crashed")
        finally:
            self._detach_owned_sandbox()
            with self._lock:
                self._running = False

    def _detach_owned_sandbox(self) -> None:
        """Release the conversation-bound session, never destroying it (S210).

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
    """Bridge an Ollama chat stream to the loop's ``stream(messages, tools)``.

    Ollama already yields chunks shaped as ``{"message": {"content", "tool_calls"}}``,
    which is the loop's expected stream shape. The ``ollama`` import is lazy so
    this module loads without it; resolution failure surfaces as a 503.
    """

    def __init__(self, model: str, *, host: str | None = None) -> None:
        self._model = model
        self._host = host

    def stream(self, messages: list[dict[str, Any]], tools: Any = None):
        import ollama

        client = ollama.Client(host=self._host) if self._host else ollama.Client()
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools
        for chunk in client.chat(**kwargs):
            yield chunk


def _resolve_model_client(model: str | None) -> Any:
    """Build a model client for a run, or None when none can be resolved."""
    if not model:
        return None
    try:
        return _OllamaModelClient(model)
    except Exception:  # pragma: no cover - defensive
        return None


def _resolve_memory_provider() -> Callable[..., str] | None:
    """The S66 working-memory block provider, guarded."""
    try:
        from opti_oignon.memory import working_memory_block

        return working_memory_block
    except Exception:  # pragma: no cover - memory backend optional here
        return None


def _resolve_approval_fn() -> Callable[[str, str, dict], bool] | None:
    """Reuse the existing tool-call approval gate as the loop's approval_fn."""
    try:
        from opti_oignon.agent import allowlists

        def _gate(conversation_id: str, tool_name: str, arguments: dict) -> bool:
            return allowlists.request_approval(conversation_id, tool_name, arguments)

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


def _skill_payload(skill: Any, *, with_body: bool = False) -> dict[str, Any]:
    """Serialise a skill for the wire: metadata, plus the body on a single view."""
    data = dict(skill.to_dict())
    if with_body:
        data["body"] = skill.body
    return data


def skills_list_payload(registry: Any, *, include_drafts: bool = True) -> dict[str, Any]:
    """The registry index payload: published skills, plus drafts by default."""
    skills = registry.list(include_drafts=include_drafts)
    return {"skills": [_skill_payload(s) for s in skills]}


def skill_view_payload(registry: Any, category: str, name: str) -> dict[str, Any]:
    """One skill with its full body; the published one if present, else its draft."""
    skill = registry.get(name, category, draft=False)
    if skill is None:
        skill = registry.get(name, category, draft=True)
    if skill is None:
        raise SkillNotFound(f"{category}/{name}")
    return _skill_payload(skill, with_body=True)


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
        mode: str = "daily"
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
            _emergency_stop.guard_http()  # S215: refused, not hung
        _require_agent()
        if not request.task.strip():
            raise HTTPException(status_code=422, detail="task cannot be empty")
        model_client = _resolve_model_client(request.model or None)
        if model_client is None:
            raise HTTPException(status_code=503, detail="No model client available")
        result = get_run_manager().start(
            request.task,
            model_client=model_client,
            mode=request.mode,
            conversation_id=request.conversation_id,
            approval_fn=_resolve_approval_fn(),
            memory_provider=_resolve_memory_provider(),
            verify=request.verify,
            consult=request.consult,
        )
        if not result.get("started"):
            raise HTTPException(status_code=409, detail=result.get("reason", "run_not_started"))
        return result

    @router.websocket("/stream")
    async def agent_stream(websocket: WebSocket) -> None:
        """Forward the live AgentEvent stream over a WebSocket."""
        await websocket.accept()
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[str]" = asyncio.Queue()

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

    # Skills registry surface (S178 Goal 0: closes the S177 carry-over).
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
