#!/usr/bin/env python3
"""
API routes for streaming chat over WebSocket.

Provides real-time generation, cancellation, and retry
endpoints for the SvelteKit frontend.
"""

import asyncio
import json
import logging
import os
import threading
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .deps import (
    ANALYZER_AVAILABLE,
    CONVERSATION_AVAILABLE,
    EXECUTOR_AVAILABLE,
    PRESET_AVAILABLE,
    ROUTER_AVAILABLE,
    analyzer,
    conversation_manager,
    executor,
    preset_manager,
)
from .deps import (
    router as model_router,
)
from .schemas import (
    ChatCancelRequest,
    ChatRequest,
    ChatRetryRequest,
    ConsensusConfigResponse,
    ConsensusRequest,
)

# Import conditionnel du tool executor
try:
    from opti_oignon.tool_executor import tool_executor as _tool_executor
    TOOL_EXECUTOR_AVAILABLE = True
except ImportError:
    TOOL_EXECUTOR_AVAILABLE = False
    _tool_executor = None

# Emergency-stop admission guard (a stopped system refuses honestly)
try:
    from opti_oignon import emergency_stop as _emergency_stop
except Exception:
    _emergency_stop = None

# PIP-06: conditional import of the execution-pipeline system.
# The PipelineRunner rides the agentic executor; CRUD lives in
# routes_exec_pipelines; this is the execution seam (finished here).
try:
    from opti_oignon.pipelines import (
        get_pipeline_runner,
        get_pipeline_store,
    )
    EXEC_PIPELINES_AVAILABLE = True
except ImportError:
    EXEC_PIPELINES_AVAILABLE = False
    get_pipeline_runner = None
    get_pipeline_store = None

# Import conditionnel de l'executeur agentique
try:
    from opti_oignon.agentic_executor import (
        PIPELINE_CONSENSUS,  # noqa: F401
        PIPELINE_DIRECT,  # noqa: F401
        PIPELINE_REASONING,  # noqa: F401
        PIPELINE_SELF_CORRECT,  # noqa: F401
        PIPELINE_THINK_TOOLS,  # noqa: F401
        PIPELINE_TOOLS,  # noqa: F401
    )
    from opti_oignon.agentic_executor import (
        agentic_executor as _agentic_executor,
    )
    AGENTIC_EXECUTOR_AVAILABLE = True
except ImportError:
    AGENTIC_EXECUTOR_AVAILABLE = False
    _agentic_executor = None

# Import conditionnel du consensus engine
try:
    from opti_oignon.consensus import (
        ConsensusEngine,
        ConsensusResult,  # noqa: F401
    )
    from opti_oignon.consensus import (
        ModelResponse as ConsensusModelResponse,  # noqa: F401
    )
    from opti_oignon.consensus import (
        consensus_engine as _consensus_engine,
    )
    CONSENSUS_ENGINE_AVAILABLE = True
except ImportError:
    CONSENSUS_ENGINE_AVAILABLE = False
    _consensus_engine = None

# Import conditionnel du plugin hook system
try:
    from opti_oignon.plugin_hooks import hook_manager as _hook_manager
    PLUGIN_HOOKS_AVAILABLE = True
except ImportError:
    PLUGIN_HOOKS_AVAILABLE = False
    _hook_manager = None

# Import conditionnel du quick sandbox
try:
    from opti_oignon.quick_sandbox import (
        QUICK_SANDBOX_AVAILABLE,
    )
    from opti_oignon.quick_sandbox import (
        quick_sandbox_manager as _quick_sandbox_manager,
    )
    from opti_oignon.tool_registry import tool_registry as _tool_registry
except ImportError:
    QUICK_SANDBOX_AVAILABLE = False
    _quick_sandbox_manager = None
    _tool_registry = None

# The conversation <-> workspace binding store; guarded so the chat route
# still loads if the module is absent. Resolution failures fall back to
# the pre-bridge behavior (a fresh per-conversation sandbox).
try:
    from opti_oignon.sandbox_workspace import (
        get_workspace_bindings as _get_workspace_bindings,
    )
except ImportError:
    _get_workspace_bindings = None

# Import conditionnel du tool call approval (Bulbe mode)
try:
    from opti_oignon.tool_call_approval import ApprovalStatus as _ApprovalStatus
    from opti_oignon.tool_call_approval import tool_call_approval as _tool_call_approval
    TOOL_CALL_APPROVAL_AVAILABLE = True
except ImportError:
    TOOL_CALL_APPROVAL_AVAILABLE = False
    _tool_call_approval = None
    _ApprovalStatus = None

# Import security mode for policy check
try:
    from opti_oignon.security_mode import get_policy as _get_security_policy
    SECURITY_POLICY_AVAILABLE = True
except ImportError:
    SECURITY_POLICY_AVAILABLE = False
    _get_security_policy = None

# Import backpressure buffer for slow client detection
try:
    from opti_oignon.sse_backpressure import BackpressureBuffer
    BACKPRESSURE_AVAILABLE = True
except ImportError:
    BACKPRESSURE_AVAILABLE = False
    BackpressureBuffer = None  # type: ignore[assignment,misc]

# Configurable backpressure defaults
_BP_MAX_SIZE = 100
_BP_SLOW_THRESHOLD = 0.8
# Idle disconnect: stop a stream the client has stopped consuming. Raised from
# the original 60s and made overridable (OPTI_IDLE_TIMEOUT_S) so slow local
# models -- e.g. a MoE that spills to RAM and streams in bursts -- are not cut
# off mid-thought. This measures the gap since the last consumed event, not the
# total duration, so it still catches a genuinely dead client.
_BP_IDLE_TIMEOUT = float(os.environ.get("OPTI_IDLE_TIMEOUT_S", "600"))

# RFC 6455 WebSocket close codes for graceful server-side shutdown.
# 1011 = internal error (server hit an unexpected condition); 1003 = the
# client sent data the server cannot accept (malformed / invalid request).
WS_CLOSE_INTERNAL_ERROR = 1011
WS_CLOSE_INVALID_DATA = 1003

# Import conditionnel du chat coding agent
try:
    from opti_oignon.chat_coding_agent import (
        CHAT_CODING_AVAILABLE,
        ChatCodingSession,
        CodingEvent,
        LLMCallContext,
        LLMCallResult,
    )
    from opti_oignon.chat_coding_agent import (
        chat_coding_manager as _chat_coding_manager,
    )
    from opti_oignon.chat_coding_agent import (
        parse_directives as _parse_coding_directives,
    )
except ImportError:
    CHAT_CODING_AVAILABLE = False
    _chat_coding_manager = None
    ChatCodingSession = None
    CodingEvent = None
    LLMCallResult = None
    LLMCallContext = None
    _parse_coding_directives = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Registre des flags d'annulation par conversation
# Each conversation en cours a un threading.Event
_cancel_events: dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


def _get_cancel_event(conversation_id: str) -> threading.Event:
    """Retrieve or create a cancellation Event for a conversation."""
    with _cancel_lock:
        if conversation_id not in _cancel_events:
            _cancel_events[conversation_id] = threading.Event()
        return _cancel_events[conversation_id]


def _cleanup_cancel_event(conversation_id: str) -> None:
    """Delete the cancellation Event after generation ends."""
    with _cancel_lock:
        _cancel_events.pop(conversation_id, None)


def _resolve_model_and_route(message: str, request: ChatRequest):
    """Resolve the model and perform routing.

    Priorite: force_model > preset > auto-routing.
    Passe les images pour detection de vision auto-routing.

    Returns:
        Tuple of (routing_result, error_message). error_message is None on success.
    """
    # 1. Model force dans la request
    force_model = request.model

    # 2. Resolution via preset
    if not force_model and request.preset and PRESET_AVAILABLE and preset_manager:
        try:
            p = preset_manager.get(request.preset)
            if p and p.model:
                force_model = p.model
        except Exception as e:
            logger.debug(f"Erreur resolution preset {request.preset}: {e}")

    # 3. Auto-detection par mots-cles du preset
    if (
        not force_model
        and request.use_presets
        and PRESET_AVAILABLE
        and preset_manager
    ):
        try:
            detected = preset_manager.find_by_keywords(message)
            if detected and detected.model:
                force_model = detected.model
        except Exception:
            pass

    # 4. Routage via analyzer + router
    if not ANALYZER_AVAILABLE or not ROUTER_AVAILABLE:
        return None, "Analyzer or Router not available"

    try:
        analysis = analyzer.analyze(message)
        # Pass the images and the message through for vision detection
        routing = model_router.route(
            analysis,
            force_model=force_model if force_model else None,
            images=request.images,
            message=message,
        )
        # Appliquer temperature forcee si specifiee
        if request.temperature is not None:
            routing.temperature = request.temperature
        # Make sure the images are part of the routing input
        if request.images and not routing.images:
            routing.images = request.images
        return routing, None
    except Exception as e:
        return None, f"Routing error: {e}"


async def _send_token(ws: WebSocket, token_type: str, content: str = "",
                      metadata: dict | None = None) -> bool:
    """Send a ChatToken via WebSocket. Returns False if connection is closed."""
    try:
        data = {"type": token_type, "content": content}
        if metadata:
            data["metadata"] = metadata
        await ws.send_json(data)
        return True
    except Exception:
        return False


async def _stream_response(
    websocket: WebSocket,
    conversation_id: str,
    message: str,
    request: ChatRequest,
) -> None:
    """Generate and stream the LLM response via WebSocket.

    Orchestre le routage, l'appel a l'executor, et l'envoi
    des tokens au client. Gere les modes think et web_search.
    """
    start_time = time.time()
    _last_send_time = start_time  # Track last WS send for keepalive

    # Check les prerequis
    if not EXECUTOR_AVAILABLE or executor is None:
        await _send_token(websocket, "error", "Executor module not available")
        return

    # Routage
    routing, routing_error = _resolve_model_and_route(message, request)
    if routing_error or routing is None:
        await _send_token(websocket, "error", routing_error or "Routing failed")
        return

    # Preparer l'annulation
    cancel_event = _get_cancel_event(conversation_id)
    cancel_event.clear()

    # Resetr l'executor
    executor.reset()

    # Activate quick sandbox mode if requested
    _qs_session = None
    _qs_active = False
    if (
        QUICK_SANDBOX_AVAILABLE
        and _quick_sandbox_manager is not None
        and _tool_registry is not None
    ):
        # Determine if quick sandbox should be active for this request
        qs_requested = getattr(request, 'quick_sandbox', None)
        qs_enabled = (
            qs_requested is True
            or (qs_requested is None and _quick_sandbox_manager.enabled)
        )
        if qs_enabled and _quick_sandbox_manager.available:
            try:
                _bound_workspace_id = None
                if _get_workspace_bindings is not None and conversation_id:
                    try:
                        _bound_workspace_id = (
                            _get_workspace_bindings().get_sandbox_for(
                                conversation_id
                            )
                        )
                    except Exception:
                        _bound_workspace_id = None
                _qs_session = _quick_sandbox_manager.get_or_create_session(
                    request_id=conversation_id or None,
                    bound_sandbox_id=_bound_workspace_id,
                )
                _tool_registry.set_quick_sandbox_mode(
                    True, session=_qs_session
                )
                _qs_active = True
                logger.info(
                    "Quick sandbox activated for conversation %s",
                    conversation_id,
                )
            except Exception as exc:
                logger.warning("Quick sandbox activation failed: %s", exc)
                _qs_session = None

    # Detect chat coding agent activation
    _cc_active = False
    _cc_message = message  # possibly stripped of /code prefix
    if (
        CHAT_CODING_AVAILABLE
        and _chat_coding_manager is not None
    ):
        cc_requested = getattr(request, 'chat_coding', None)
        # Detect /code slash command
        _msg_stripped = message.strip()
        _code_prefix = False
        if _msg_stripped.startswith("/code ") or _msg_stripped == "/code":
            _code_prefix = True
            _cc_message = _msg_stripped[5:].strip() or _msg_stripped

        cc_enabled = (
            cc_requested is True
            or _code_prefix
            or (cc_requested is None and _chat_coding_manager.enabled)
        )
        if cc_enabled and _chat_coding_manager.available:
            _cc_active = True
            # When Code Agent is ON, Quick Sandbox is implicitly disabled
            # (the coding session owns its own sandbox)
            if _qs_active and _tool_registry is not None:
                _tool_registry.set_quick_sandbox_mode(False)
                _qs_active = False

    # Send the routing metadata
    await _send_token(websocket, "metadata", metadata={
        "conversation_id": conversation_id,
        "model": routing.model,
        "task_type": routing.task_type,
        "temperature": routing.temperature,
        "prompt_variant": routing.prompt_variant,
        "think": request.think or False,
        "web_search": request.web_search or False,
        # Raison de routage transparente
        "routing_reason": routing.routing_reason,
        # Vision routing
        "vision_routed": getattr(routing, "vision_routed", False),
        "has_images": bool(getattr(routing, "images", None)),
        # Quick sandbox status
        "quick_sandbox": _qs_active,
        # Chat coding agent status
        "chat_coding": _cc_active,
    })

    # Chat Coding Agent execution path
    # When active, replaces the normal generation flow entirely.
    # The coding agent runs plan -> implement -> test -> fix in the sandbox,
    # streaming CodingEvents via WebSocket, with full pipeline capabilities
    # (vision, tools, plugins, web search) at each LLM call.
    if _cc_active:
        await _stream_chat_coding(
            websocket=websocket,
            conversation_id=conversation_id,
            message=_cc_message,
            request=request,
            routing=routing,
            start_time=start_time,
        )
        _cleanup_cancel_event(conversation_id)
        return

    # Executer la generation dans un thread (l'executor est synchrone)
    full_response = ""
    thinking_content = ""
    error_occurred = False
    generation_done = threading.Event()
    chunks = []
    # Backpressure tracking for slow client detection
    _bp_dropped = 0
    _bp_slow_logged = 0
    _bp_last_consumer_time = time.time()

    # Determiner si on utilise l'AgenticExecutor
    use_agentic = (
        AGENTIC_EXECUTOR_AVAILABLE
        and _agentic_executor is not None
        and _agentic_executor.available
    )

    # PIP-06: resolve the requested execution pipeline up front so an
    # unknown id or a missing prerequisite refuses honestly instead of
    # silently falling through to the plain chat path. The chat-coding branch
    # above takes precedence by construction (it returns before this point).
    _exec_pipeline_obj = None
    if getattr(request, "exec_pipeline", None):
        if not EXEC_PIPELINES_AVAILABLE or get_pipeline_store is None:
            await _send_token(
                websocket, "error",
                "Execution pipelines module not available",
            )
            return
        if not use_agentic:
            await _send_token(
                websocket, "error",
                "Execution pipelines require the agentic executor",
            )
            return
        _exec_pipeline_obj = get_pipeline_store().get(request.exec_pipeline)
        if _exec_pipeline_obj is None:
            await _send_token(
                websocket, "error",
                f"Unknown execution pipeline: {request.exec_pipeline}",
            )
            return

    def _on_tool_call_callback(tool_call_result):
        """Callback emitting tool calls in real time."""
        chunks.append(("tool_call", tool_call_result))
        # Fire tool_call hooks so plugins can react
        # redact_sensitive=True applies per-plugin data redaction
        if PLUGIN_HOOKS_AVAILABLE and _hook_manager and _hook_manager.has_hooks("tool_call"):
            try:
                _hook_manager.execute(
                    "tool_call",
                    conversation_id=conversation_id,
                    model=routing.model if routing else None,
                    data={
                        "tool_name": getattr(tool_call_result, "tool_name", ""),
                        "arguments": getattr(tool_call_result, "arguments", {}),
                        "result": getattr(tool_call_result, "result", ""),
                        "success": getattr(tool_call_result, "success", True),
                    },
                    redact_sensitive=True,
                )
            except Exception as exc:
                logger.debug("tool_call hook dispatch failed: %s", exc)

    def _on_reasoning_step_callback(reasoning_step):
        """Callback emitting the reasoning steps."""
        chunks.append(("reasoning_step", reasoning_step))

    def _on_consensus_model_callback(model_response):
        """Callback to emit individual consensus responses."""
        chunks.append(("consensus_model_done", model_response))

    # EX-02: per-request tool-approval gate. Bound to this request and
    # passed into the executor call rather than mutated onto a shared singleton,
    # so overlapping Bulbe sessions cannot clobber or drop each other's gate.
    # Assigned in the Bulbe branch below; None means no gate (Daily / no policy).
    _approval_fn = None

    def _generate():
        """Generation thread: calls executor.execute() and collects chunks."""
        nonlocal full_response, error_occurred
        # A live turn pins the quick sandbox for the whole generation: the
        # workspace must survive long inferences between tool calls (the
        # inactivity timeout is a between-turns notion). The finally below
        # guarantees the release on every exit path.
        if _qs_active and _qs_session is not None:
            _qs_session.begin_turn()
        try:
            # Retrieve the images from routing
            _images = getattr(routing, "images", None) or (request.images if request else None)

            # Status callback to capture vision delegation events
            # Also emit general status messages for intermediate feedback
            def _on_status(msg):
                # Always emit as generic status for StreamingIndicator
                chunks.append(("status", {"message": msg}))
                # Additionally tag vision-specific statuses
                if "Analyzing image" in msg or "vision" in msg.lower():
                    chunks.append(("vision_delegation", {
                        "status": "analyzing",
                        "message": msg,
                    }))

            if _exec_pipeline_obj is not None:
                # PIP-06: execution-pipeline run via the executor-backed
                # PipelineRunner (the seam, finished). The runner resets
                # and drives the agentic executor per step; the approval gate
                # (EX-02) is forwarded so Bulbe semantics hold per step.
                _agentic_executor.reset()
                gen = get_pipeline_runner().execute(
                    pipeline=_exec_pipeline_obj,
                    message=message,
                    routing=routing,
                    conversation_id=conversation_id if conversation_id else None,
                    on_status=_on_status,
                    on_tool_call=_on_tool_call_callback,
                    on_reasoning_step=_on_reasoning_step_callback,
                    on_consensus_model=_on_consensus_model_callback,
                    approval_fn=_approval_fn,
                )
            elif use_agentic:
                # Execution via AgenticExecutor
                _agentic_executor.reset()
                gen = _agentic_executor.execute(
                    message=message,
                    routing=routing,
                    conversation_id=conversation_id if conversation_id else None,
                    think=request.think if request.think is not None else None,
                    web_search=request.web_search if request.web_search is not None else None,
                    # Consensus multi-model
                    consensus=request.consensus if request.consensus is not None else None,
                    consensus_models=request.consensus_models,
                    consensus_strategy=request.consensus_strategy,
                    # Auto-correction
                    self_correct=request.self_correct if hasattr(request, 'self_correct') and request.self_correct is not None else None,
                    # Lot 5: opt-in tool-model optimization
                    optimize=getattr(request, 'optimize', None),
                    on_tool_call=_on_tool_call_callback,
                    on_reasoning_step=_on_reasoning_step_callback,
                    on_consensus_model=_on_consensus_model_callback,
                    # Pipe status for intermediate feedback
                    on_status=_on_status,
                    # EX-02: per-request tool-approval gate
                    approval_fn=_approval_fn,
                )
            else:
                # Execution classique via Executor
                gen = executor.execute(
                    question=message,
                    routing=routing,
                    document=None,
                    refine=False,
                    on_status=_on_status,
                    conversation_id=conversation_id if conversation_id else None,
                    think=request.think if request.think else False,
                    web_search=request.web_search if request.web_search else False,
                    images=_images,
                )
            for chunk in gen:
                if cancel_event.is_set():
                    if use_agentic:
                        _agentic_executor.cancel()
                    else:
                        executor.cancel()
                    chunks.append(("cancel", None))
                    break
                if chunk:
                    # PIP-06: pipeline step-boundary tuples from the
                    # PipelineRunner; relayed as light status, never
                    # concatenated into the response text.
                    if isinstance(chunk, tuple) and len(chunk) == 3:
                        if chunk[0] == "pipeline_step_end":
                            chunks.append((
                                "status",
                                {"message": f"Step {chunk[1] + 1} done"},
                            ))
                        # pipeline_step_start: on_status already emitted
                        # "Step i/N: label" from the runner.
                        continue
                    # Distinguish thinking chunks from regular chunks
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        chunk_type, chunk_content = chunk
                        if chunk_type == "thinking":
                            chunks.append(("thinking", chunk_content))
                        elif chunk_type == "reasoning_step":
                            # Etape de raisonnement
                            chunks.append(("reasoning_step", chunk_content))
                        elif chunk_type == "reasoning_done":
                            # Fin du raisonnement
                            chunks.append(("reasoning_done", chunk_content))
                        elif chunk_type == "consensus_model_done":
                            # Reponse individuelle de consensus
                            chunks.append(("consensus_model_done", chunk_content))
                        elif chunk_type == "consensus_done":
                            # Fin du consensus
                            chunks.append(("consensus_done", chunk_content))
                        elif chunk_type == "correction_step":
                            # Etape d'auto-correction
                            chunks.append(("correction_step", chunk_content))
                        elif chunk_type == "correction_done":
                            # Fin de l'auto-correction
                            chunks.append(("correction_done", chunk_content))
                        else:
                            full_response += chunk_content
                            chunks.append(("chunk", chunk_content))
                    else:
                        full_response += chunk
                        chunks.append(("chunk", chunk))

            # Emit vision delegation completion if it occurred
            _vm = getattr(executor, 'last_vision_meta', {})
            if _vm.get("delegated"):
                chunks.append(("vision_delegation", {
                    "status": "done",
                    "vision_model": _vm.get("vision_model", ""),
                    "description_length": _vm.get("description_length", 0),
                    "duration_ms": _vm.get("duration_ms", 0),
                }))

        except Exception as e:
            logger.error(f"Generation error: {e}")
            chunks.append(("error", str(e)))
            error_occurred = True
        finally:
            if _qs_active and _qs_session is not None:
                _qs_session.end_turn()
            generation_done.set()

    # Fire pre_inference hooks before starting generation
    # redact_sensitive=True applies per-plugin data redaction
    if PLUGIN_HOOKS_AVAILABLE and _hook_manager and _hook_manager.has_hooks("pre_inference"):
        try:
            pre_report = _hook_manager.execute(
                "pre_inference",
                conversation_id=conversation_id,
                model=routing.model,
                data={"message": message, "model": routing.model},
                redact_sensitive=True,
            )
            # Allow hooks to modify the message (e.g. chain-of-thought-enforcer)
            if pre_report.final_data.get("message") and pre_report.final_data["message"] != message:
                message = pre_report.final_data["message"]
                logger.debug("pre_inference hooks modified message (conv=%s)", conversation_id[:8] if conversation_id else "?")
        except Exception as exc:
            logger.warning("pre_inference hook dispatch failed: %s", exc)

    gen_thread = threading.Thread(target=_generate, daemon=True)

    # In Bulbe mode, arm a pre-execution approval gate so every tool
    # call blocks until human approval.
    # EX-02: the gate is bound to this request (assigned to _approval_fn
    # and passed into the executor call above) instead of mutated onto the
    # shared ToolExecutor singleton, so overlapping Bulbe sessions cannot
    # clobber or drop each other's gate.
    if (
        TOOL_CALL_APPROVAL_AVAILABLE
        and _tool_call_approval is not None
        and SECURITY_POLICY_AVAILABLE
        and _get_security_policy is not None
    ):
        try:
            policy = _get_security_policy()
            if getattr(policy, "tool_call_approval_required", False):
                _conv_id = conversation_id or ""

                def _approval_hook(tool_name: str, arguments: dict) -> bool:
                    """Block until human approves or 30s timeout (auto-deny)."""
                    aid, event = _tool_call_approval.submit(
                        conversation_id=_conv_id,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                    # Emit pending event to WebSocket via chunks
                    chunks.append(("tool_call_pending", {
                        "approval_id": aid,
                        "tool_name": tool_name,
                        "arguments_summary": _tool_call_approval._pending.get(aid, None)
                            and _tool_call_approval._pending[aid].arguments_summary or "",
                        "risk_level": _tool_call_approval._pending.get(aid, None)
                            and _tool_call_approval._pending[aid].risk_level or "low",
                    }))
                    # Wait for approval (blocks this thread)
                    from opti_oignon.tool_call_approval import DEFAULT_TIMEOUT_SECONDS
                    event.wait(timeout=DEFAULT_TIMEOUT_SECONDS + 2)
                    status = _tool_call_approval.get_status(aid)
                    approved = (
                        status is not None
                        and status == _ApprovalStatus.APPROVED
                    )
                    # Emit resolution event
                    chunks.append(("tool_call_resolved", {
                        "approval_id": aid,
                        "tool_name": tool_name,
                        "approved": approved,
                    }))
                    return approved

                # EX-02: bind the gate to this request instead of
                # mutating the shared singleton. _approval_fn is forwarded to
                # the executor call, which threads it down to _execute_tool.
                _approval_fn = _approval_hook
                logger.info("Tool call approval gate armed (Bulbe mode)")
        except Exception as exc:
            logger.warning("Failed to install tool call approval hook: %s", exc)

    gen_thread.start()

    # Streaming loop: sends chunks progressively
    # Critical event types that must never be dropped by backpressure
    _BP_CRITICAL_EVENTS = frozenset({
        "error", "cancel", "tool_call_pending", "tool_call_resolved",
    })
    sent_index = 0
    while not generation_done.is_set() or sent_index < len(chunks):
        # Backpressure -- detect slow client and drop oldest
        # non-critical events when the pending queue exceeds the limit.
        pending = len(chunks) - sent_index
        if pending > _BP_MAX_SIZE:
            # Find non-critical events to drop from the front of the
            # pending window, preserving critical events and the most
            # recent events.
            drop_target = pending - _BP_MAX_SIZE
            dropped_now = 0
            new_chunks = chunks[:sent_index]  # already consumed
            skipped = 0
            for i in range(sent_index, len(chunks)):
                ev_type = chunks[i][0]
                if skipped < drop_target and ev_type not in _BP_CRITICAL_EVENTS:
                    skipped += 1
                    dropped_now += 1
                else:
                    new_chunks.append(chunks[i])
            chunks[:] = new_chunks
            _bp_dropped += dropped_now
            if dropped_now > 0:
                _bp_slow_logged += 1
                if _bp_slow_logged <= 5 or _bp_slow_logged % 20 == 0:
                    logger.warning(
                        "Backpressure: dropped %d non-critical events "
                        "(total dropped: %d, pending: %d/%d)",
                        dropped_now, _bp_dropped,
                        len(chunks) - sent_index, _BP_MAX_SIZE,
                    )

        # Idle timeout -- disconnect if consumer has not
        # progressed for longer than the configured timeout.
        if (
            not generation_done.is_set()
            and (time.time() - _bp_last_consumer_time) > _BP_IDLE_TIMEOUT
            and sent_index > 0  # only after at least one event consumed
        ):
            logger.warning(
                "Client idle timeout (%.0fs) -- disconnecting "
                "(conv=%s, dropped=%d)",
                _BP_IDLE_TIMEOUT,
                conversation_id[:8] if conversation_id else "?",
                _bp_dropped,
            )
            cancel_event.set()
            executor.cancel()
            generation_done.wait(timeout=5.0)
            if _qs_active and _tool_registry is not None:
                _tool_registry.set_quick_sandbox_mode(False)
            _cleanup_cancel_event(conversation_id)
            return

        # Send les chunks en attente
        while sent_index < len(chunks):
            event_type, content = chunks[sent_index]
            sent_index += 1
            _bp_last_consumer_time = time.time()

            if event_type == "chunk":
                alive = await _send_token(websocket, "token", content)
                _last_send_time = time.time()
                if not alive:
                    cancel_event.set()
                    executor.cancel()
                    generation_done.wait(timeout=5.0)
                    if _qs_active and _tool_registry is not None:
                        _tool_registry.set_quick_sandbox_mode(False)
                    _cleanup_cancel_event(conversation_id)
                    return
            elif event_type == "thinking":
                # Send the thinking content over the WebSocket
                thinking_content += content
                alive = await _send_token(websocket, "thinking", content)
                if not alive:
                    cancel_event.set()
                    executor.cancel()
                    generation_done.wait(timeout=5.0)
                    if _qs_active and _tool_registry is not None:
                        _tool_registry.set_quick_sandbox_mode(False)
                    _cleanup_cancel_event(conversation_id)
                    return
            elif event_type == "cancel":
                await _send_token(websocket, "token", "\n\n[Generation cancelled]")
                break
            elif event_type == "tool_call":
                # Emettre les appels d'outils en temps reel
                tc = content  # content est un ToolCallResult ici
                if tc is not None:
                    await _send_token(websocket, "tool_call", "", metadata={
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments if hasattr(tc, 'arguments') else {},
                        "status": "complete" if tc.success else "error",
                        "result_preview": tc.result[:500] if hasattr(tc, 'result') and tc.result else "",
                        "execution_time": tc.execution_time if hasattr(tc, 'execution_time') else 0,
                        "success": tc.success if hasattr(tc, 'success') else True,
                        "reasoning": tc.reasoning if hasattr(tc, 'reasoning') else "",
                    })
            elif event_type == "reasoning_step":
                # Emettre les etapes de raisonnement
                step = content
                if step is not None:
                    await _send_token(websocket, "reasoning_step", "", metadata={
                        "step_number": step.step_number if hasattr(step, 'step_number') else 0,
                        "title": step.title if hasattr(step, 'title') else "",
                        "content": step.content[:1000] if hasattr(step, 'content') and step.content else "",
                        "duration_ms": step.duration_ms if hasattr(step, 'duration_ms') else 0,
                    })
            elif event_type == "reasoning_done":
                # Emit the end of the reasoning
                reasoning_result = content
                if reasoning_result is not None:
                    await _send_token(websocket, "reasoning_done", "", metadata={
                        "strategy": reasoning_result.strategy if hasattr(reasoning_result, 'strategy') else "",
                        "steps_count": len(reasoning_result.steps) if hasattr(reasoning_result, 'steps') else 0,
                        "confidence": reasoning_result.confidence if hasattr(reasoning_result, 'confidence') else 0,
                        "total_duration_ms": reasoning_result.total_duration_ms if hasattr(reasoning_result, 'total_duration_ms') else 0,
                    })
            elif event_type == "consensus_model_done":
                # Emit an individual model response
                model_resp = content
                if model_resp is not None:
                    await _send_token(websocket, "consensus_model_done", "", metadata={
                        "model": model_resp.model if hasattr(model_resp, 'model') else "",
                        "content": model_resp.content[:2000] if hasattr(model_resp, 'content') and model_resp.content else "",
                        "duration_ms": model_resp.duration_ms if hasattr(model_resp, 'duration_ms') else 0,
                        "success": model_resp.success if hasattr(model_resp, 'success') else False,
                        "error": model_resp.error if hasattr(model_resp, 'error') else "",
                        "quality_tier": model_resp.quality_tier if hasattr(model_resp, 'quality_tier') else "medium",
                    })
            elif event_type == "consensus_done":
                # Emit the final consensus result
                consensus_result = content
                if consensus_result is not None:
                    await _send_token(websocket, "consensus_done", "", metadata={
                        "strategy": consensus_result.strategy if hasattr(consensus_result, 'strategy') else "",
                        "selected_model": consensus_result.selected_model if hasattr(consensus_result, 'selected_model') else "",
                        "confidence": consensus_result.confidence if hasattr(consensus_result, 'confidence') else 0,
                        "total_duration_ms": consensus_result.total_duration_ms if hasattr(consensus_result, 'total_duration_ms') else 0,
                        "average_agreement": consensus_result.comparison.average_agreement if hasattr(consensus_result, 'comparison') and consensus_result.comparison else 0,
                        "models_count": len(consensus_result.individual_responses) if hasattr(consensus_result, 'individual_responses') else 0,
                    })
            elif event_type == "correction_step":
                # Emettre une etape d'auto-correction
                step_info = content
                if step_info is not None:
                    await _send_token(websocket, "correction_step", "", metadata=step_info)
            elif event_type == "correction_done":
                # Emettre le result final d'auto-correction
                correction_result = content
                if correction_result is not None:
                    await _send_token(websocket, "correction_done", "", metadata={
                        "was_corrected": correction_result.was_corrected if hasattr(correction_result, 'was_corrected') else False,
                        "iterations_performed": correction_result.iterations_performed if hasattr(correction_result, 'iterations_performed') else 0,
                        "compliance_before": correction_result.compliance_before if hasattr(correction_result, 'compliance_before') else 1.0,
                        "compliance_after": correction_result.compliance_after if hasattr(correction_result, 'compliance_after') else 1.0,
                        "quality_before": correction_result.quality_before if hasattr(correction_result, 'quality_before') else 1.0,
                        "quality_after": correction_result.quality_after if hasattr(correction_result, 'quality_after') else 1.0,
                        "total_duration_ms": correction_result.total_duration_ms if hasattr(correction_result, 'total_duration_ms') else 0,
                    })
            elif event_type == "status":
                # Emit intermediate status for StreamingIndicator
                if isinstance(content, dict):
                    await _send_token(websocket, "status", "", metadata=content)
            elif event_type == "vision_delegation":
                # Emit vision delegation status (analyzing / done)
                if isinstance(content, dict):
                    await _send_token(websocket, "vision_delegation", "", metadata=content)
            elif event_type == "tool_call_pending":
                # Tool call awaiting human approval (Bulbe mode)
                if isinstance(content, dict):
                    await _send_token(websocket, "tool_call_pending", "", metadata=content)
            elif event_type == "tool_call_resolved":
                # Tool call approval resolved (approved/denied/timeout)
                if isinstance(content, dict):
                    await _send_token(websocket, "tool_call_resolved", "", metadata=content)
            elif event_type == "error":
                await _send_token(websocket, "error", content)
                if _qs_active and _tool_registry is not None:
                    _tool_registry.set_quick_sandbox_mode(False)
                _cleanup_cancel_event(conversation_id)
                return

        # Wait briefly before rechecking.
        # SSE-05: poll without blocking the asyncio event loop. The
        # generation runs on a daemon thread and signals via generation_done;
        # awaiting asyncio.sleep yields to the loop instead of blocking it on a
        # threading.Event (the sibling _stream_chat_coding loop already does this).
        if not generation_done.is_set():
            await asyncio.sleep(0.05)

            # Keepalive: send ping every 10s during long inferences
            # to prevent WebSocket timeout (1006). This happens when
            # search + think + opti are combined and take 30+ seconds.
            elapsed_since_last = time.time() - _last_send_time
            if elapsed_since_last > 10.0:
                try:
                    await websocket.send_json({"type": "ping", "timestamp": time.time()})
                    _last_send_time = time.time()
                    # SSE-04: a successful keepalive ping proves the client
                    # socket is still draining, so refresh the consumer timer too.
                    # Without this, the idle-timeout below would mistake a slow
                    # producer (long tool/search/think phase, no events) for a
                    # slow client and cancel a legitimate long generation.
                    _bp_last_consumer_time = time.time()
                except Exception:
                    cancel_event.set()
                    executor.cancel()
                    return

    # Attendre la fin du thread
    gen_thread.join(timeout=10.0)

    # Calculer la duration
    duration_ms = int((time.time() - start_time) * 1000)

    # Emit code verification results
    # Retrieve from AgenticExecutor if used, otherwise from Executor
    _verification_results = []
    if use_agentic and _agentic_executor is not None:
        _verification_results = _agentic_executor.last_verification_results or []
    elif (
        EXECUTOR_AVAILABLE
        and executor is not None
        and hasattr(executor, 'last_verification_results')
    ):
        _verification_results = executor.last_verification_results or []

    for vr in _verification_results:
        await _send_token(websocket, "verification", "", metadata={
            "status": vr.status,
            "iterations": vr.iterations,
            "language": vr.language,
            "errors": vr.errors_encountered[:3],
            "fixes": vr.fixes_applied[:3],
            "execution_output": vr.execution_output[:500] if vr.execution_output else "",
        })

    # Emit tool call results (those not already emitted in real-time)
    # Les appels emis via callback sont already envoyes; ici on emet ceux de l'executor legacy
    if (
        not use_agentic
        and TOOL_EXECUTOR_AVAILABLE
        and _tool_executor is not None
        and hasattr(executor, '_last_tool_calls')
        and executor._last_tool_calls
    ):
        for tc in executor._last_tool_calls:
            await _send_token(websocket, "tool_call", "", metadata={
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
                "status": "complete" if tc.success else "error",
                "result_preview": tc.result[:500] if tc.result else "",
                "execution_time": tc.execution_time,
                "success": tc.success,
                "reasoning": tc.reasoning,
            })

    # Fire post_inference hooks — plugins can annotate/modify the response
    # redact_sensitive=True applies per-plugin data redaction
    plugin_annotations: list[dict] = []
    if PLUGIN_HOOKS_AVAILABLE and _hook_manager and _hook_manager.has_hooks("post_inference"):
        try:
            post_report = _hook_manager.execute(
                "post_inference",
                conversation_id=conversation_id,
                model=routing.model,
                data={
                    "response": full_response,
                    "message": message,
                    "model": routing.model,
                    "duration_ms": duration_ms,
                },
                redact_sensitive=True,
            )
            # Collect plugin annotations (e.g. fact-checker results, auto-tldr)
            for result in post_report.results:
                if result.success and result.modified_data:
                    ann = result.modified_data.get("annotation")
                    if ann:
                        plugin_annotations.append({
                            "plugin": result.plugin_name,
                            "data": ann,
                        })
                    # Allow plugins to append text to response
                    suffix = result.modified_data.get("response_suffix")
                    if suffix and isinstance(suffix, str):
                        full_response += suffix
            if post_report.failed > 0:
                logger.warning(
                    "post_inference: %d/%d hooks failed",
                    post_report.failed, post_report.total_hooks,
                )
        except Exception as exc:
            logger.warning("post_inference hook dispatch failed: %s", exc)

    # Send le message "done"
    done_metadata = {
        "conversation_id": conversation_id,
        "model": routing.model,
        "duration_ms": duration_ms,
        "cancelled": cancel_event.is_set(),
        # Routing reason in the done payload as well
        "routing_reason": routing.routing_reason,
    }
    # Include backpressure stats if any events were dropped
    if _bp_dropped > 0:
        done_metadata["backpressure"] = {
            "events_dropped": _bp_dropped,
            "slow_warnings": _bp_slow_logged,
        }
    # Include vision delegation info in done metadata
    _final_vision_meta = getattr(executor, 'last_vision_meta', {})
    if _final_vision_meta.get("delegated"):
        done_metadata["vision_delegation"] = {
            "vision_model": _final_vision_meta.get("vision_model", ""),
            "description_length": _final_vision_meta.get("description_length", 0),
            "duration_ms": _final_vision_meta.get("duration_ms", 0),
        }
    if thinking_content:
        done_metadata["thinking"] = thinking_content
    # PIP-06: record which execution pipeline ran, if any
    if _exec_pipeline_obj is not None:
        done_metadata["exec_pipeline"] = _exec_pipeline_obj.id
    # Add the agentic information
    if use_agentic and _agentic_executor is not None:
        done_metadata["pipeline"] = _agentic_executor.last_pipeline
        done_metadata["tool_calls_count"] = len(_agentic_executor.last_tool_calls)
        done_metadata["verifications_count"] = len(_agentic_executor.last_verification_results)
        # Add the reasoning information
        if _agentic_executor.last_reasoning_result is not None:
            rr = _agentic_executor.last_reasoning_result
            done_metadata["reasoning"] = {
                "strategy": rr.strategy,
                "steps_count": len(rr.steps),
                "confidence": rr.confidence,
                "total_duration_ms": rr.total_duration_ms,
            }
        # Add the self-correction information
        if _agentic_executor.last_correction_result is not None:
            cr = _agentic_executor.last_correction_result
            done_metadata["correction"] = {
                "was_corrected": cr.was_corrected,
                "iterations_performed": cr.iterations_performed,
                "compliance_before": cr.compliance_before,
                "compliance_after": cr.compliance_after,
                "quality_before": cr.quality_before,
                "quality_after": cr.quality_after,
                "total_duration_ms": cr.total_duration_ms,
            }
    # Include plugin annotations if any
    if plugin_annotations:
        done_metadata["plugin_annotations"] = plugin_annotations
    # Include quick sandbox metadata if sandbox was used
    if _qs_active and _qs_session is not None:
        sandbox_files = _qs_session.get_sandbox_files()
        done_metadata["sandbox_active"] = True
        # The servable id: the adopted workspace id when bound, the
        # session's own id otherwise. The files/preview/approve routes
        # are keyed by this id, not by the conversation-keyed one.
        done_metadata["sandbox_session_id"] = _qs_session.effective_sandbox_id
        done_metadata["sandbox_files"] = sandbox_files
        done_metadata["sandbox_files_created"] = _qs_session.files_created
    await _send_token(websocket, "done", full_response, metadata=done_metadata)

    # Nettoyage
    # Deactivate quick sandbox mode (restore original tool handlers)
    if _qs_active and _tool_registry is not None:
        try:
            _tool_registry.set_quick_sandbox_mode(False)
        except Exception as exc:
            logger.warning("Quick sandbox deactivation failed: %s", exc)
    # EX-02: the approval gate was request-scoped (passed into the
    # executor call), so there is no shared singleton attribute to reset here.
    _cleanup_cancel_event(conversation_id)


# ---------------------------------------------------------------------------
# Chat Coding Agent — rich LLM callback + streaming
# ---------------------------------------------------------------------------

def _build_rich_llm_callback(
    routing,
    conversation_id: str,
):
    """Build a rich LLM callback that wraps the full chat pipeline.

    The returned callback gives the coding agent access to:
    - Vision delegation: images analyzed by vision-capable model
    - Web search: search and inject documentation
    - Tool calls: all registered tools via agentic executor
    - Plugin hooks: pre/post inference
    - Full conversation context (already in the messages array)

    Signature: (messages, model, LLMCallContext) -> LLMCallResult
    """
    def _rich_callback(
        messages: list[dict[str, str]],
        model: str,
        context: "LLMCallContext",
    ) -> "LLMCallResult":
        """Rich LLM callback wrapping the full pipeline."""
        if LLMCallResult is None:
            return None  # type: ignore

        result = LLMCallResult()
        _images = context.images if context else None

        # 1. Vision delegation: process images before LLM call
        user_msg = messages[-1]["content"] if messages else ""
        try:
            from opti_oignon.vision_pipeline import (
                vision_pipeline as _vp,
            )
            if _vp is not None and _images:
                user_msg_new, _images, vmeta = _vp.process(
                    message=user_msg,
                    images=_images,
                    current_model=model,
                    on_status=lambda s: logger.debug("Vision: %s", s),
                )
                if user_msg_new != user_msg:
                    messages = list(messages)
                    messages[-1] = {"role": "user", "content": user_msg_new}
                    user_msg = user_msg_new
                result.vision_meta = vmeta
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Vision in coding agent: %s", exc)

        # SR-02: the coding-agent path had a dead web-search block here
        # that imported and called a search-augmentation helper defined nowhere
        # in the codebase (and a flag that is not exported), so the import always
        # raised ImportError and the block never ran. Removed. The live
        # web_search path is in executor.py (web_search_engine), used by the
        # standard chat pipeline.

        # 3. Plugin pre_inference hooks
        plugin_annotations: list[dict] = []
        if PLUGIN_HOOKS_AVAILABLE and _hook_manager:
            try:
                if _hook_manager.has_hooks("pre_inference"):
                    _hook_manager.execute(
                        "pre_inference",
                        conversation_id=conversation_id,
                        model=model,
                        data={"message": user_msg, "model": model},
                    )
            except Exception as exc:
                logger.debug("pre_inference in coding: %s", exc)

        # 4. LLM call via executor with coding system prompt as suffix.
        # We use the standard executor (not agentic) because:
        # - The coding agent handles its own tool calls to the sandbox
        # - Agentic tool routing would conflict with sandbox routing
        # - executor.execute() natively handles vision delegation + images
        # The coding phase instructions (system prompt) are passed as suffix.
        full_text = ""
        tool_calls_meta: list[dict] = []

        # Extract coding system prompt from messages[0] (injected by
        # ChatCodingSession._build_conversation_messages)
        coding_system = ""
        if messages and messages[0].get("role") == "system":
            coding_system = messages[0]["content"]

        try:
            gen = executor.execute(
                question=user_msg,
                routing=routing,
                conversation_id=conversation_id,
                think=context.think if context else False,
                web_search=False,  # already handled above
                images=_images,
                system_prompt_suffix=coding_system,
            )
            for chunk in gen:
                if isinstance(chunk, tuple):
                    if chunk[0] == "thinking":
                        result.thinking += chunk[1]
                elif isinstance(chunk, str):
                    full_text += chunk
        except Exception as exc:
            logger.warning("LLM call in coding agent: %s", exc)
            result.error = str(exc)

        result.text = full_text
        result.tool_calls = tool_calls_meta

        # 5. Plugin post_inference hooks
        # redact_sensitive=True applies per-plugin data redaction
        if PLUGIN_HOOKS_AVAILABLE and _hook_manager:
            try:
                if _hook_manager.has_hooks("post_inference"):
                    post_report = _hook_manager.execute(
                        "post_inference",
                        conversation_id=conversation_id,
                        model=model,
                        data={
                            "response": full_text,
                            "message": user_msg,
                            "model": model,
                        },
                        redact_sensitive=True,
                    )
                    for pr in post_report.results:
                        if pr.success and pr.modified_data:
                            ann = pr.modified_data.get("annotation")
                            if ann:
                                plugin_annotations.append({
                                    "plugin": pr.plugin_name,
                                    "data": ann,
                                })
                            suffix = pr.modified_data.get("response_suffix")
                            if suffix and isinstance(suffix, str):
                                result.text += suffix
            except Exception as exc:
                logger.debug("post_inference in coding: %s", exc)

        result.plugin_annotations = plugin_annotations
        return result

    return _rich_callback


async def _stream_chat_coding(
    websocket: WebSocket,
    conversation_id: str,
    message: str,
    request: "ChatRequest",
    routing,
    start_time: float,
) -> None:
    """Execute the chat coding agent and stream CodingEvents via WebSocket.

    Replaces the normal generation flow when chat_coding is active.
    The coding agent runs plan -> implement -> test -> fix in its sandbox,
    with full pipeline capabilities at each LLM call.
    """
    import asyncio

    if _chat_coding_manager is None:
        await _send_token(websocket, "error", "Chat coding agent not available")
        return

    # Build the rich LLM callback
    rich_callback = _build_rich_llm_callback(
        routing=routing,
        conversation_id=conversation_id,
    )

    # Get or create a coding session for this conversation
    try:
        session = _chat_coding_manager.get_or_create_session(
            conversation_id=conversation_id,
            llm_call=rich_callback,
        )
    except RuntimeError as exc:
        await _send_token(websocket, "error", str(exc))
        return

    logger.info(
        "Chat coding active for conv %s (session=%s, turn=%d)",
        conversation_id[:8], session.session_id, session.turn_count + 1,
    )

    # Parse directives from the message
    directives = (
        _parse_coding_directives(message)
        if _parse_coding_directives else None
    )

    # Run the coding pipeline in a thread (LLM calls are synchronous)
    events: list = []
    execution_done = threading.Event()
    final_result: dict = {}

    def _run_coding():
        nonlocal final_result
        try:
            gen = session.execute_task(
                message=(
                    directives.cleaned_message if directives else message
                ),
                model=routing.model,
                directives=directives,
                images=request.images if request else None,
                web_search=bool(request.web_search) if request else False,
                think=bool(request.think) if request else False,
            )
            for event in gen:
                events.append(event)
                # Capture final result from coding_done event
                if event.event_type == "coding_done":
                    final_result = event.data or {}
        except Exception as exc:
            if CodingEvent is not None:
                events.append(CodingEvent(
                    event_type="coding_error",
                    content=str(exc),
                ))
            logger.error("Chat coding error: %s", exc)
        finally:
            execution_done.set()

    coding_thread = threading.Thread(target=_run_coding, daemon=True)
    coding_thread.start()

    # Stream events to WebSocket as they arrive
    sent_index = 0
    _last_send_time = time.time()
    full_response_text = ""

    while not execution_done.is_set() or sent_index < len(events):
        while sent_index < len(events):
            event = events[sent_index]
            sent_index += 1

            event_type = event.event_type
            metadata = dict(event.data) if event.data else {}
            metadata["event_content"] = event.content

            if event_type == "coding_done":
                full_response_text = event.content
            elif event_type == "coding_error":
                await _send_token(websocket, "error", event.content)
                execution_done.wait(timeout=5.0)
                return

            alive = await _send_token(
                websocket, event_type, event.content, metadata=metadata
            )
            _last_send_time = time.time()
            if not alive:
                execution_done.wait(timeout=5.0)
                return

        if not execution_done.is_set():
            await asyncio.sleep(0.05)
            if time.time() - _last_send_time > 10.0:
                try:
                    await websocket.send_json({
                        "type": "ping", "timestamp": time.time()
                    })
                    _last_send_time = time.time()
                except Exception:
                    return

    coding_thread.join(timeout=10.0)

    # Send "done" with sandbox metadata
    duration_ms = int((time.time() - start_time) * 1000)
    sandbox_files = session.get_sandbox_files_for_ui()

    done_metadata = {
        "conversation_id": conversation_id,
        "model": routing.model,
        "duration_ms": duration_ms,
        "cancelled": False,
        "routing_reason": routing.routing_reason,
        "chat_coding": True,
        "coding_result": final_result,
        "sandbox_active": True,
        "sandbox_session_id": session.session_id,
        "sandbox_files": sandbox_files,
        "sandbox_files_created": list(session.sandbox_state.files),
        "turn_count": session.turn_count,
    }

    if hasattr(session, '_last_vision_meta') and session._last_vision_meta:
        done_metadata["vision_delegation"] = session._last_vision_meta
    if (
        hasattr(session, '_last_plugin_annotations')
        and session._last_plugin_annotations
    ):
        done_metadata["plugin_annotations"] = session._last_plugin_annotations

    await _send_token(
        websocket, "done", full_response_text, metadata=done_metadata
    )


# ---------------------------------------------------------------------------
# WebSocket: /api/chat/stream
# ---------------------------------------------------------------------------

@router.websocket("/stream")
async def chat_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for chat streaming.

    S136 audit fix: authenticates before processing.
    """
    await websocket.accept()

    # Audit fix: authenticate WebSocket connection
    try:
        from .routes_auth import authenticate_websocket
        user = await authenticate_websocket(websocket)
        if user is None:
            await _send_token(websocket, "error", "Authentication required")
            await websocket.close(code=4001)
            return
    except Exception:
        await _send_token(websocket, "error", "Authentication failed")
        await websocket.close(code=4001)
        return

    # Emergency-stop admission guard -- refused, not hung
    if _emergency_stop is not None and _emergency_stop.is_stopped():
        await _send_token(
            websocket, "error", _emergency_stop.refusal_payload()["message"]
        )
        await websocket.close()
        return

    try:
        # Receive request from client
        raw_data = await websocket.receive_json()

        # Validate request
        try:
            request = ChatRequest(**raw_data)
        except (ValidationError, TypeError) as e:
            await _send_token(websocket, "error", f"Invalid request: {e}")
            await websocket.close()
            return

        if not request.message.strip():
            await _send_token(websocket, "error", "Empty message")
            await websocket.close()
            return

        # Creer ou valider la conversation
        conversation_id = request.conversation_id or ""

        if not conversation_id and CONVERSATION_AVAILABLE and conversation_manager:
            try:
                conv = conversation_manager.create_conversation(title="New conversation")
                conversation_id = conv.id
            except Exception as e:
                logger.error(f"Erreur creation conversation: {e}")
                conversation_id = ""
        elif conversation_id and CONVERSATION_AVAILABLE and conversation_manager:
            # Check que la conversation existe
            conv = conversation_manager.get_conversation(conversation_id)
            if conv is None:
                await _send_token(
                    websocket, "error",
                    f"Conversation not found: {conversation_id}"
                )
                await websocket.close()
                return

        # Lancer le streaming
        await _stream_response(websocket, conversation_id, request.message, request)

    except WebSocketDisconnect:
        logger.debug("Client WebSocket disconnected")
    except json.JSONDecodeError:
        try:
            await _send_token(websocket, "error", "Invalid JSON")
            await websocket.close(code=WS_CLOSE_INVALID_DATA)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Erreur WebSocket stream: {e}")
        try:
            await _send_token(websocket, "error", str(e))
            await websocket.close(code=WS_CLOSE_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket: /api/chat/retry
# ---------------------------------------------------------------------------

@router.websocket("/retry")
async def chat_retry(websocket: WebSocket) -> None:
    """WebSocket endpoint to regenerate the last response.

    S136 audit fix: authenticates before processing.
    """
    await websocket.accept()

    # Audit fix: authenticate WebSocket connection
    try:
        from .routes_auth import authenticate_websocket
        user = await authenticate_websocket(websocket)
        if user is None:
            await _send_token(websocket, "error", "Authentication required")
            await websocket.close(code=4001)
            return
    except Exception:
        await _send_token(websocket, "error", "Authentication failed")
        await websocket.close(code=4001)
        return

    # Emergency-stop admission guard -- refused, not hung
    if _emergency_stop is not None and _emergency_stop.is_stopped():
        await _send_token(
            websocket, "error", _emergency_stop.refusal_payload()["message"]
        )
        await websocket.close()
        return

    try:
        raw_data = await websocket.receive_json()

        # Validate request
        try:
            retry_req = ChatRetryRequest(**raw_data)
        except (ValidationError, TypeError) as e:
            await _send_token(websocket, "error", f"Invalid request: {e}")
            await websocket.close()
            return

        conv_id = retry_req.conversation_id

        if not CONVERSATION_AVAILABLE or conversation_manager is None:
            await _send_token(websocket, "error", "Conversation module not available")
            await websocket.close()
            return

        # Check que la conversation existe
        conv = conversation_manager.get_conversation(conv_id)
        if conv is None:
            await _send_token(websocket, "error", f"Conversation not found: {conv_id}")
            await websocket.close()
            return

        # Retrieve the messages to find the last user message
        messages = conversation_manager.get_messages(conv_id)
        if not messages:
            await _send_token(websocket, "error", "No messages in conversation")
            await websocket.close()
            return

        # Recover the model of the turn being retried before the history is
        # rewound: an explicit request override wins, then the conversation's
        # last used model, then the newest assistant message that recorded
        # one. None (no model anywhere) keeps the default routing behavior.
        retry_model = retry_req.model or conv.model
        if not retry_model:
            for msg in reversed(messages):
                if msg.role == "assistant" and getattr(msg, "model", None):
                    retry_model = msg.model
                    break

        # Supprimer le dernier message assistant
        conversation_manager.delete_last_message(conv_id, role="assistant")

        # Trouver le dernier message utilisateur
        last_user_message = None
        # Re-lire les messages apres suppression
        messages = conversation_manager.get_messages(conv_id)
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_message = msg.content
                break

        if not last_user_message:
            await _send_token(websocket, "error", "No user message found for retry")
            await websocket.close()
            return

        # Supprimer also le dernier user message (executor va le re-creer)
        conversation_manager.delete_last_message(conv_id, role="user")

        # Construire une ChatRequest pour le re-envoi
        chat_request = ChatRequest(
            conversation_id=conv_id,
            message=last_user_message,
            model=retry_model,
        )

        # Stream la new reponse
        await _stream_response(websocket, conv_id, last_user_message, chat_request)

    except WebSocketDisconnect:
        logger.debug("Client WebSocket disconnected (retry)")
    except json.JSONDecodeError:
        try:
            await _send_token(websocket, "error", "Invalid JSON")
            await websocket.close(code=WS_CLOSE_INVALID_DATA)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Erreur WebSocket retry: {e}")
        try:
            await _send_token(websocket, "error", str(e))
            await websocket.close(code=WS_CLOSE_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# POST: /api/chat/cancel
# ---------------------------------------------------------------------------

@router.post("/cancel")
async def cancel_generation(body: ChatCancelRequest) -> dict:
    """Annule une generation en cours pour une conversation donnee.

    Positionne le flag d'annulation que la boucle de streaming verifie.
    """
    conv_id = body.conversation_id
    with _cancel_lock:
        event = _cancel_events.get(conv_id)

    if event is None:
        return JSONResponse(
            status_code=404,
            content={"detail": "No active generation for this conversation"},
        )

    event.set()
    # Annuler also cote executor (pour interrompre ollama.chat)
    if EXECUTOR_AVAILABLE and executor is not None:
        executor.cancel()
    # Annuler also l'executeur agentique
    if AGENTIC_EXECUTOR_AVAILABLE and _agentic_executor is not None:
        _agentic_executor.cancel()

    logger.info(f"Annulation demandee pour conversation {conv_id[:8]}...")
    return {"status": "cancelled", "conversation_id": conv_id}


# ---------------------------------------------------------------------------
# POST: /api/chat/consensus
# ---------------------------------------------------------------------------

@router.post("/consensus")
async def run_consensus(body: ConsensusRequest) -> dict:
    """Run a multi-model consensus (synchronous mode).

    Queries N models in parallel, compares the responses, and
    returns the consensus result with all individual responses
    and the comparison.
    """
    if not CONSENSUS_ENGINE_AVAILABLE or _consensus_engine is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Consensus engine not available"},
        )

    try:
        result = _consensus_engine.run_consensus(
            query=body.message,
            models=body.models,
            strategy=body.strategy,
            system_prompt=body.system_prompt,
            temperature=body.temperature,
        )

        return ConsensusEngine.result_to_dict(result)

    except Exception as e:
        logger.error(f"Consensus error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Consensus execution failed: {str(e)}"},
        )


# ---------------------------------------------------------------------------
# GET: /api/chat/consensus/config
# ---------------------------------------------------------------------------

@router.get("/consensus/config")
async def get_consensus_config() -> dict:
    """Return the current consensus configuration."""
    if not CONSENSUS_ENGINE_AVAILABLE or _consensus_engine is None:
        return ConsensusConfigResponse(available=False).model_dump()

    config = _consensus_engine.config
    return ConsensusConfigResponse(
        default_models=config.default_models,
        strategy=config.strategy,
        max_models=config.max_models,
        timeout_per_model=config.timeout_per_model,
        min_agreement_threshold=config.min_agreement_threshold,
        available=_consensus_engine.available,
    ).model_dump()

# ---------------------------------------------------------------------------
# Chat Coding Agent endpoints
# ---------------------------------------------------------------------------

@router.get("/coding/status")
async def get_chat_coding_status() -> dict:
    """Get chat coding agent system status."""
    if not CHAT_CODING_AVAILABLE or _chat_coding_manager is None:
        return {
            "enabled": False,
            "available": False,
            "session_timeout_minutes": 60,
            "max_concurrent_sessions": 3,
            "active_sessions": 0,
            "auto_test": True,
            "max_fix_retries": 3,
        }
    return _chat_coding_manager.get_status()


@router.post("/coding/toggle")
async def toggle_chat_coding(request: dict) -> dict:
    """Enable or disable the chat coding agent."""
    if not CHAT_CODING_AVAILABLE or _chat_coding_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Chat coding agent not available"},
        )
    enabled = request.get("enabled", False)
    _chat_coding_manager.enabled = bool(enabled)
    return _chat_coding_manager.get_status()


@router.get("/coding/sessions")
async def list_chat_coding_sessions() -> dict:
    """List active chat coding sessions."""
    if not CHAT_CODING_AVAILABLE or _chat_coding_manager is None:
        return {"sessions": []}
    return {"sessions": _chat_coding_manager.list_sessions()}


@router.delete("/coding/{conversation_id}")
async def destroy_chat_coding_session(conversation_id: str) -> dict:
    """Destroy a chat coding session for a conversation."""
    if not CHAT_CODING_AVAILABLE or _chat_coding_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Chat coding agent not available"},
        )
    destroyed = _chat_coding_manager.destroy_session(conversation_id)
    if not destroyed:
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"No active coding session for {conversation_id}"
            },
        )
    return {"status": "destroyed", "conversation_id": conversation_id}


@router.get("/coding/{conversation_id}/status")
async def get_chat_coding_session_status(conversation_id: str) -> dict:
    """Get status of a specific chat coding session."""
    if not CHAT_CODING_AVAILABLE or _chat_coding_manager is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Chat coding agent not available"},
        )
    session = _chat_coding_manager.get_session(conversation_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"No active coding session for {conversation_id}"
            },
        )
    return session.get_status()


@router.post("/coding/cleanup")
async def cleanup_chat_coding_sessions() -> dict:
    """Clean up expired chat coding sessions."""
    if not CHAT_CODING_AVAILABLE or _chat_coding_manager is None:
        return {"cleaned": 0}
    count = _chat_coding_manager.cleanup_expired()
    return {"cleaned": count}
