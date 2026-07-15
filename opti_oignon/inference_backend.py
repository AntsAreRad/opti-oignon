#!/usr/bin/env python3
"""
INFERENCE BACKEND ABSTRACTION -- OPTI-OIGNON S105
=================================================

Backend-agnostic inference layer allowing Opti-Oignon to run
models through different engines (Ollama, llama.cpp, etc.)
without changing the inference pipeline.

Architecture:
    InferenceBackend (ABC)
      |-- OllamaBackend       (wraps existing ollama-python)
      |-- LlamaCppBackend     (direct llama-cpp-python GGUF loading)
    BackendRegistry (singleton) -- manages available backends

Author: Leon
"""

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

try:
    import ollama as _ollama_module
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    _ollama_module = None

try:
    from llama_cpp import Llama as _LlamaCpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    _LlamaCpp = None

# Telemetry integration (S112) -- lazy import to avoid circular deps.
_telemetry_collector = None  # type: Any
_TELEMETRY_CHECKED = False


def _get_telemetry() -> Any:
    """Lazy-load the telemetry singleton. Returns None if unavailable."""
    global _telemetry_collector, _TELEMETRY_CHECKED
    if _TELEMETRY_CHECKED:
        return _telemetry_collector
    _TELEMETRY_CHECKED = True
    try:
        from opti_oignon.telemetry import get_telemetry
        _telemetry_collector = get_telemetry()
    except Exception:
        _telemetry_collector = None
    return _telemetry_collector


# Resource Governor mechanical seam (S224, spec Section 4.1) -- lazy and
# fail-open by construction: resolved per call (never cached, so a
# test-seeded or standalone module is reused as-is), absent or unavailable
# means proceed unguarded (the S216 availability-control posture).


def _resolve_resource_governor() -> Any:
    """Lazy governor resolver; None means unguarded."""
    try:
        import sys as _sys

        mod = _sys.modules.get("opti_oignon.resource_governor")
        if mod is None:
            from opti_oignon import resource_governor as mod  # type: ignore
        if mod is None or not getattr(mod, "FEATURE_AVAILABLE", False):
            return None
        return mod
    except Exception:
        return None


def _governor_admission(model: str, options: dict | None) -> None:
    """The internal hook at the four generate/stream heads (S224).

    Additive and internal: generate/stream signatures DO NOT change. A
    funnel-held ticket (resource_governor.ticket_scope) stands the gate
    down; a ticketless call gets the fast cached admit-or-refuse backstop.
    Module absent, disabled by config, or any governor error: proceed
    unguarded. Only the governor's own typed GovernorRefusal propagates.
    """
    rg = _resolve_resource_governor()
    if rg is None:
        return
    try:
        rg.backend_admission_gate(model, options)
    except Exception as exc:
        refusal = getattr(rg, "GovernorRefusal", None)
        if refusal is not None and isinstance(exc, refusal):
            raise
        logger.debug("Governor gate failed open: %s", exc)


# Model integrity seam. Deliberately the OPPOSITE posture to the governor gate
# above, and the contrast is the point: an absent resource governor means an
# unguarded but otherwise correct load, so it fails open. An absent integrity
# proof does not mean "load unverified" -- it means no proof exists, so it
# fails secure.


def _provenance_mode() -> str:
    """The live security mode, fail-secure to bulbe when undeterminable."""
    try:
        from opti_oignon.security_mode import get_current_mode

        return str(get_current_mode() or "").strip().lower() or "bulbe"
    except Exception:
        return "bulbe"


def _provenance_guard(gguf_path: Path) -> None:
    """Verify the model's pinned digest before its bytes reach llama.cpp.

    The path guard in _resolve_model_path proved WHERE the file is; it never
    proved WHAT it contains. This gate does, and it raises to refuse.

    An unresolvable provenance module is itself a refusal whenever the mode
    enforces. Swallowing that import would reintroduce, on the one seam that
    hands raw bytes to a native parser, exactly the silent fail-open shape
    this gate exists to remove.

    Raises:
        ProvenanceRefusal: When the model's provenance does not verify.
        RuntimeError: When verification is unavailable and the mode enforces.
    """
    try:
        from opti_oignon import model_provenance as _provenance
    except Exception as exc:
        if _provenance_mode() != "daily":
            raise RuntimeError(
                "Model provenance verification is unavailable and the current "
                "security mode enforces it; refusing to load "
                f"{gguf_path.name}"
            ) from exc
        logger.warning(
            "Model provenance unavailable; load continues unverified: %s", exc
        )
        return

    _provenance.guard_model_load(gguf_path)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class BackendModelInfo:
    """Unified model information across backends."""

    __slots__ = (
        "name", "backend", "size", "family", "parameter_size",
        "quantization_level", "context_length", "modified_at",
        "path", "extra",
    )

    def __init__(
        self,
        name: str,
        backend: str,
        size: str | None = None,
        family: str | None = None,
        parameter_size: str | None = None,
        quantization_level: str | None = None,
        context_length: int | None = None,
        modified_at: str | None = None,
        path: str | None = None,
        extra: dict | None = None,
    ):
        self.name = name
        self.backend = backend
        self.size = size
        self.family = family
        self.parameter_size = parameter_size
        self.quantization_level = quantization_level
        self.context_length = context_length
        self.modified_at = modified_at
        self.path = path
        self.extra = extra or {}

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "backend": self.backend,
            "size": self.size,
            "family": self.family,
            "parameter_size": self.parameter_size,
            "quantization_level": self.quantization_level,
            "context_length": self.context_length,
            "modified_at": self.modified_at,
            "path": self.path,
            "extra": self.extra,
        }


class ChatResponse:
    """Unified non-streaming chat response."""

    __slots__ = ("content", "thinking", "model", "done", "total_duration", "extra")

    def __init__(
        self,
        content: str,
        thinking: str | None = None,
        model: str = "",
        done: bool = True,
        total_duration: int | None = None,
        extra: dict | None = None,
    ):
        self.content = content
        self.thinking = thinking
        self.model = model
        self.done = done
        self.total_duration = total_duration
        self.extra = extra or {}

    def to_dict(self) -> dict:
        """Serialize to dictionary matching ollama response format."""
        result = {
            "message": {"role": "assistant", "content": self.content},
            "model": self.model,
            "done": self.done,
        }
        if self.thinking:
            result["message"]["thinking"] = self.thinking
        if self.total_duration is not None:
            result["total_duration"] = self.total_duration
        return result


class StreamChunk:
    """Unified streaming chunk."""

    __slots__ = ("content", "thinking", "done", "model")

    def __init__(
        self,
        content: str = "",
        thinking: str = "",
        done: bool = False,
        model: str = "",
    ):
        self.content = content
        self.thinking = thinking
        self.done = done
        self.model = model

    def to_dict(self) -> dict:
        """Serialize to dictionary matching ollama chunk format."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.thinking:
            msg["thinking"] = self.thinking
        return {
            "message": msg,
            "done": self.done,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class InferenceBackend(ABC):
    """Abstract interface for inference backends.

    Every backend must implement these methods so the executor
    can switch engines transparently.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique backend identifier (e.g. 'ollama', 'llama_cpp')."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g. 'Ollama', 'llama.cpp')."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable and functional."""
        ...

    @abstractmethod
    def list_models(self) -> list[BackendModelInfo]:
        """List all models available through this backend."""
        ...

    @abstractmethod
    def model_info(self, model_name: str) -> BackendModelInfo | None:
        """Return detailed information about a specific model."""
        ...

    @abstractmethod
    def generate(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        keep_alive: str = "30m",
        think: bool = False,
        images: list | None = None,
    ) -> ChatResponse:
        """Non-streaming inference.

        Args:
            model: Model name or GGUF filename.
            messages: Chat messages in OpenAI format
                      [{"role": "user", "content": "..."}].
            options: Engine options (temperature, top_p, etc.).
            keep_alive: Keep-alive duration (Ollama-specific, ignored by others).
            think: Enable thinking/chain-of-thought output.
            images: Optional base64-encoded images for vision models.

        Returns:
            ChatResponse with the complete generation.
        """
        ...

    @abstractmethod
    def stream(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        keep_alive: str = "30m",
        think: bool = False,
        images: list | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Streaming inference.

        Same parameters as generate(), but yields StreamChunk objects.
        """
        ...


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaBackend(InferenceBackend):
    """Backend wrapping the existing ollama-python library.

    This is a transparent wrapper: all current Ollama functionality
    keeps working exactly as before.
    """

    def __init__(self, host: str = "http://localhost:11434"):
        self._host = host

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama"

    def health_check(self) -> bool:
        """Check Ollama connectivity."""
        if not OLLAMA_AVAILABLE:
            return False
        try:
            _ollama_module.list()
            return True
        except Exception:
            return False

    def unload_all(self) -> int:
        """Evict every model currently loaded in Ollama (frees VRAM).

        S215: Ollama exposes no dedicated unload endpoint; the documented
        eviction is a generate call with ``keep_alive=0``, which unloads the
        model immediately. Loaded models are enumerated via ``ps()`` (both
        the dict and the object response forms are handled, the CC-01
        class). Requires no privileges; stopping a systemd-managed Ollama
        service remains a documented host action. Per-model failures are
        logged and skipped so one stuck model never blocks the rest.

        Returns the number of successful eviction requests.
        """
        if not OLLAMA_AVAILABLE:
            return 0
        try:
            ps_response = _ollama_module.ps()
        except Exception as exc:
            logger.debug("Ollama ps failed during unload_all: %s", exc)
            return 0
        if isinstance(ps_response, dict):
            raw_models = ps_response.get("models", []) or []
        else:
            raw_models = getattr(ps_response, "models", []) or []
        count = 0
        for m in raw_models:
            if isinstance(m, dict):
                name = m.get("name") or m.get("model")
            else:
                name = getattr(m, "name", None) or getattr(m, "model", None)
            if not name:
                continue
            try:
                _ollama_module.generate(model=name, keep_alive=0)
                count += 1
            except Exception as exc:
                logger.warning("Ollama unload failed for %s: %s", name, exc)
        if count:
            logger.info("Requested Ollama eviction for %d model(s)", count)
        return count

    def unload_model(self, model_name: str) -> bool:
        """Evict ONE model from Ollama (S225: the unload_all idiom
        narrowed to a single name for the governor's targeted eviction).

        Same documented primitive: a generate call with ``keep_alive=0``
        unloads the named model immediately. Returns True when the
        eviction request was accepted; False when ollama is unavailable
        or the request failed (logged, never raised).
        """
        if not OLLAMA_AVAILABLE:
            return False
        try:
            _ollama_module.generate(model=model_name, keep_alive=0)
            logger.info("Requested Ollama eviction for %s", model_name)
            return True
        except Exception as exc:
            logger.warning("Ollama unload failed for %s: %s", model_name, exc)
            return False

    def list_models(self) -> list[BackendModelInfo]:
        """List Ollama models via ollama.list()."""
        if not OLLAMA_AVAILABLE:
            return []
        try:
            response = _ollama_module.list()
            raw_models = []
            if hasattr(response, "models"):
                raw_models = response.models or []
            elif isinstance(response, dict):
                raw_models = response.get("models", [])
            else:
                raw_models = list(response) if response else []

            results = []
            for m in raw_models:
                results.append(self._parse_ollama_model(m))
            return results
        except Exception as exc:
            logger.debug("Ollama list_models failed: %s", exc)
            return []

    def model_info(self, model_name: str) -> BackendModelInfo | None:
        """Get model details via ollama.show()."""
        if not OLLAMA_AVAILABLE:
            return None
        try:
            info = _ollama_module.show(model_name)
            ctx_length = None
            if isinstance(info, dict):
                mi = info.get("model_info", {})
                if isinstance(mi, dict):
                    for k, v in mi.items():
                        if "context_length" in k:
                            ctx_length = int(v)
                            break
                details = info.get("details", {})
                family = details.get("family") if isinstance(details, dict) else None
                param_size = details.get("parameter_size") if isinstance(details, dict) else None
                quant = details.get("quantization_level") if isinstance(details, dict) else None
            else:
                family = getattr(getattr(info, "details", None), "family", None)
                param_size = getattr(getattr(info, "details", None), "parameter_size", None)
                quant = getattr(getattr(info, "details", None), "quantization_level", None)

            return BackendModelInfo(
                name=model_name,
                backend=self.name,
                family=family,
                parameter_size=param_size,
                quantization_level=quant,
                context_length=ctx_length,
            )
        except Exception as exc:
            logger.debug("Ollama model_info(%s) failed: %s", model_name, exc)
            return None

    def generate(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        keep_alive: str = "30m",
        think: bool = False,
        images: list | None = None,
    ) -> ChatResponse:
        """Non-streaming chat via ollama.chat()."""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama is not installed (pip install ollama)")

        # S224: governor admission hook (after the availability guard so
        # the "not installed" error semantics stay exactly as pinned).
        _governor_admission(model, options)

        # S112: telemetry start.
        tel = _get_telemetry()
        rid = tel.on_inference_start(model, messages) if tel else ""
        t0 = time.time()

        if images:
            messages = _inject_images(messages, images)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": options or {},
            "keep_alive": keep_alive,
        }
        if think:
            kwargs["think"] = True

        response = _ollama_module.chat(**kwargs)

        msg = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        thinking_text = msg.get("thinking", "") if isinstance(msg, dict) else getattr(msg, "thinking", "")
        total_dur = response.get("total_duration") if isinstance(response, dict) else getattr(response, "total_duration", None)

        # S112: telemetry end.
        if tel and rid:
            tokens_out = len(content.split()) if content else 0
            tokens_in = sum(len(str(m.get("content", "")).split()) for m in messages)
            latency = (time.time() - t0) * 1000.0
            tel.on_inference_end(
                request_id=rid, model=model,
                tokens_in=tokens_in, tokens_out=tokens_out,
                latency_ms=latency,
            )

        return ChatResponse(
            content=content,
            thinking=thinking_text or None,
            model=model,
            total_duration=total_dur,
        )

    def stream(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        keep_alive: str = "30m",
        think: bool = False,
        images: list | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Streaming chat via ollama.chat(stream=True)."""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama is not installed (pip install ollama)")

        # S224: governor admission hook. A generator head runs at first
        # iteration; the funnel's ticket is thread-local, so funnels set it
        # on the consuming thread (see resource_governor.ticket_scope).
        _governor_admission(model, options)

        # S112: telemetry start.
        tel = _get_telemetry()
        rid = tel.on_inference_start(model, messages) if tel else ""
        t0 = time.time()
        token_count = 0

        if images:
            messages = _inject_images(messages, images)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": options or {},
            "stream": True,
            "keep_alive": keep_alive,
        }
        if think:
            kwargs["think"] = True

        stream_iter = _ollama_module.chat(**kwargs)

        for chunk in stream_iter:
            msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
            content = ""
            thinking_text = ""

            if isinstance(msg, dict):
                content = msg.get("content", "") or ""
                thinking_text = msg.get("thinking", "") or ""
            else:
                content = getattr(msg, "content", "") or ""
                thinking_text = getattr(msg, "thinking", "") or ""

            done = chunk.get("done", False) if isinstance(chunk, dict) else getattr(chunk, "done", False)

            # S112: telemetry per-token.
            if content and tel and rid:
                tel.on_token_generated(rid, count=1)
                token_count += 1

            yield StreamChunk(
                content=content,
                thinking=thinking_text,
                done=bool(done),
                model=model,
            )

        # S112: telemetry end.
        if tel and rid:
            tokens_in = sum(len(str(m.get("content", "")).split()) for m in messages)
            latency = (time.time() - t0) * 1000.0
            tel.on_inference_end(
                request_id=rid, model=model,
                tokens_in=tokens_in, tokens_out=token_count,
                latency_ms=latency,
            )

    # -- internal helpers --

    @staticmethod
    def _parse_ollama_model(model_data) -> BackendModelInfo:
        """Parse an Ollama model object into BackendModelInfo."""
        if isinstance(model_data, dict):
            name = model_data.get("name", model_data.get("model", "unknown"))
            size = model_data.get("size")
            modified = model_data.get("modified_at")
            details = model_data.get("details", {})
            family = details.get("family") if isinstance(details, dict) else None
            param_size = details.get("parameter_size") if isinstance(details, dict) else None
            quant = details.get("quantization_level") if isinstance(details, dict) else None
        else:
            name = getattr(model_data, "model", None) or getattr(model_data, "name", "unknown")
            size = getattr(model_data, "size", None)
            modified = getattr(model_data, "modified_at", None)
            details = getattr(model_data, "details", None)
            family = getattr(details, "family", None) if details else None
            param_size = getattr(details, "parameter_size", None) if details else None
            quant = getattr(details, "quantization_level", None) if details else None

        if modified and not isinstance(modified, str):
            try:
                modified = str(modified)
            except Exception:
                modified = None

        size_str = None
        if size:
            try:
                s = int(size)
                if s >= 1_000_000_000:
                    size_str = f"{s / 1_000_000_000:.1f}GB"
                elif s >= 1_000_000:
                    size_str = f"{s / 1_000_000:.1f}MB"
                else:
                    size_str = f"{s}B"
            except (ValueError, TypeError):
                size_str = str(size)

        return BackendModelInfo(
            name=str(name),
            backend="ollama",
            size=size_str,
            family=family,
            parameter_size=param_size,
            quantization_level=quant,
            modified_at=modified,
        )


# ---------------------------------------------------------------------------
# llama.cpp backend
# ---------------------------------------------------------------------------

def _is_within_dir(base: Path, candidate: Path) -> bool:
    """Return True if candidate is contained within base, both resolved.

    Containment is decided with os.path.commonpath on the resolved paths,
    not str.startswith. startswith has a sibling-prefix pitfall (it accepts
    /models/main-evil as "inside" /models/main); commonpath does not, and
    resolving first accounts for '..' and symlinks. This mirrors the coding
    agent apply-boundary guard (_is_within_target).
    """
    try:
        base_r = str(base.resolve())
        cand_r = str(candidate.resolve())
    except OSError:
        return False
    try:
        return os.path.commonpath([base_r, cand_r]) == base_r
    except ValueError:
        # Raised for paths on different drives or a mix of absolute/relative.
        return False


def _resolve_ggml_kv_type(name: str) -> int | None:
    """Resolve a KV-cache type name (e.g. "q8_0") to the installed
    llama-cpp-python's GGML_TYPE_* constant (S259).

    Fail-open by design: an absent library or an unexposed constant
    answers None and the caller skips the knob with a warning -- a perf
    knob must never block a load.
    """
    try:
        import llama_cpp  # type: ignore[import-not-found]
    except ImportError:
        return None
    constant = getattr(llama_cpp, f"GGML_TYPE_{str(name).upper()}", None)
    return int(constant) if isinstance(constant, int) else None


class LlamaCppBackend(InferenceBackend):
    """Backend using llama-cpp-python to load GGUF files directly.

    This allows running models without an Ollama server. Models are
    loaded from local .gguf files specified in the configuration.

    Speculative decoding: this backend runs models in-process via
    llama-cpp-python and does NOT apply llama.cpp speculative decoding -- the
    -md / --draft-* flags target an external llama-server, which this backend
    does not launch. The speculative-decoding config, draft selection, VRAM
    budgeting and acceptance stats live in SpeculativeDecodingManager
    (opti_oignon.speculative_decoding); the S70 prompt-level draft-verify path
    lives in opti_oignon.speculative. S259 wires the external path: the argv
    is materialised by speculative_decoding.build_llama_server_command and the
    running server is consumed through LlamaServerBackend below (launching the
    process stays host-side, per INFERENCE_PERF_S259.md -- this codebase never
    spawns the server itself).

    Thread safety (IB-02): a per-model load lock serializes concurrent
    first-use loads of the same model, so each model is constructed
    exactly once (no double GGUF load, no race on the loaded-models
    dict). A per-model inference lock serializes calls to a single Llama
    instance (which is not safe for concurrent use) while allowing
    generation on distinct models to proceed in parallel.
    """

    def __init__(
        self,
        model_dirs: list[str] | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        flash_attn: bool = False,
        type_k: str | None = None,
        type_v: str | None = None,
    ):
        self._model_dirs = [Path(d) for d in (model_dirs or [])]
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        # S259 perf knobs, inert by default: flash attention and KV-cache
        # quantization type names (e.g. "q8_0"), resolved to ggml type
        # constants at load time, fail-open when the installed
        # llama-cpp-python does not expose the requested constant.
        self._flash_attn = bool(flash_attn)
        self._type_k = type_k
        self._type_v = type_v
        self._loaded_models: dict[str, Any] = {}
        # IB-02: guard for the per-model lock dicts below. Held only while
        # creating-and-registering a missing lock, never during a load or
        # an inference call, so it cannot serialize the hot path.
        self._locks_guard = threading.Lock()
        # Per-model load lock: serializes concurrent first-use loads of the
        # SAME model so it is constructed exactly once.
        self._load_locks: dict[str, threading.Lock] = {}
        # Per-model inference lock: a Llama instance is not safe for
        # concurrent calls, so calls to the same instance are serialized;
        # distinct models can generate in parallel.
        self._inference_locks: dict[str, threading.Lock] = {}

    @property
    def name(self) -> str:
        return "llama_cpp"

    @property
    def display_name(self) -> str:
        return "llama.cpp"

    def health_check(self) -> bool:
        """Check if llama-cpp-python is importable."""
        return LLAMA_CPP_AVAILABLE

    def list_models(self) -> list[BackendModelInfo]:
        """Scan configured directories for .gguf files."""
        results = []
        seen = set()
        for d in self._model_dirs:
            if not d.is_dir():
                continue
            for gguf_path in sorted(d.glob("*.gguf")):
                if gguf_path.name in seen:
                    continue
                seen.add(gguf_path.name)
                info = _parse_gguf_filename(gguf_path)
                info.path = str(gguf_path)
                results.append(info)
        return results

    def model_info(self, model_name: str) -> BackendModelInfo | None:
        """Return info about a specific GGUF model."""
        gguf_path = self._resolve_model_path(model_name)
        if gguf_path is None:
            return None
        info = _parse_gguf_filename(gguf_path)
        info.path = str(gguf_path)
        return info

    def generate(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        keep_alive: str = "30m",
        think: bool = False,
        images: list | None = None,
    ) -> ChatResponse:
        """Non-streaming inference via llama-cpp-python."""
        # S224: governor admission hook (additive, signature untouched).
        _governor_admission(model, options)

        # S112: telemetry start.
        tel = _get_telemetry()
        rid = tel.on_inference_start(model, messages) if tel else ""
        t0 = time.time()

        llm = self._get_or_load(model)
        opts = options or {}
        temperature = opts.get("temperature", 0.7)

        formatted = _format_messages_for_llama_cpp(messages)

        with self._lock_for(self._inference_locks, model):
            result = llm.create_chat_completion(
                messages=formatted,
                temperature=temperature,
                max_tokens=opts.get("num_predict", 2048),
                top_p=opts.get("top_p", 0.9),
                stream=False,
            )

        content = ""
        if result and "choices" in result and result["choices"]:
            msg = result["choices"][0].get("message", {})
            content = msg.get("content", "")

        # S112: telemetry end.
        if tel and rid:
            tokens_out = len(content.split()) if content else 0
            tokens_in = sum(len(str(m.get("content", "")).split()) for m in messages)
            latency = (time.time() - t0) * 1000.0
            tel.on_inference_end(
                request_id=rid, model=model,
                tokens_in=tokens_in, tokens_out=tokens_out,
                latency_ms=latency,
            )

        return ChatResponse(
            content=content,
            model=model,
        )

    def stream(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        keep_alive: str = "30m",
        think: bool = False,
        images: list | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Streaming inference via llama-cpp-python."""
        # S224: governor admission hook (additive, signature untouched).
        _governor_admission(model, options)

        # S112: telemetry start.
        tel = _get_telemetry()
        rid = tel.on_inference_start(model, messages) if tel else ""
        t0 = time.time()
        token_count = 0

        llm = self._get_or_load(model)
        opts = options or {}
        temperature = opts.get("temperature", 0.7)

        formatted = _format_messages_for_llama_cpp(messages)

        with self._lock_for(self._inference_locks, model):
            stream_iter = llm.create_chat_completion(
                messages=formatted,
                temperature=temperature,
                max_tokens=opts.get("num_predict", 2048),
                top_p=opts.get("top_p", 0.9),
                stream=True,
            )

            for chunk in stream_iter:
                delta = {}
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "") or ""
                done = False
                if "choices" in chunk and chunk["choices"]:
                    done = chunk["choices"][0].get("finish_reason") is not None

                # S112: telemetry per-token.
                if content and tel and rid:
                    tel.on_token_generated(rid, count=1)
                    token_count += 1

                yield StreamChunk(
                    content=content,
                    done=done,
                    model=model,
                )

        # S112: telemetry end.
        if tel and rid:
            tokens_in = sum(len(str(m.get("content", "")).split()) for m in messages)
            latency = (time.time() - t0) * 1000.0
            tel.on_inference_end(
                request_id=rid, model=model,
                tokens_in=tokens_in, tokens_out=token_count,
                latency_ms=latency,
            )

    # -- internal helpers --

    def _lock_for(
        self, registry: dict[str, threading.Lock], model_name: str
    ) -> threading.Lock:
        """Return the per-model lock from `registry`, creating it once.

        The guard is held only to create-and-register a missing lock
        (microseconds), never during a load or an inference call, so it
        cannot serialize the hot path. Double-checked so a lock is created
        exactly once per model.
        """
        lock = registry.get(model_name)
        if lock is None:
            with self._locks_guard:
                lock = registry.get(model_name)
                if lock is None:
                    lock = threading.Lock()
                    registry[model_name] = lock
        return lock

    def _resolve_model_path(self, model_name: str) -> Path | None:
        """Find a .gguf file by name across configured directories.

        S136 audit fix: NEVER accepts absolute paths or paths outside
        configured model_dirs.  Previously, an attacker could send
        model='/tmp/evil.gguf' via the API and load arbitrary GGUF files,
        potentially exploiting llama.cpp vulnerabilities or loading
        trojaned models.
        """
        # S136: reject absolute paths and path traversal
        if os.path.isabs(model_name) or ".." in model_name:
            logger.warning(
                "Rejected model path (absolute or traversal): %s", model_name
            )
            return None

        for d in self._model_dirs:
            candidate = d / model_name
            # Verify resolved path stays within the model directory
            resolved = candidate.resolve()
            dir_resolved = d.resolve()
            if not _is_within_dir(d, candidate):
                logger.warning(
                    "Model path traversal blocked: %s -> %s (outside %s)",
                    model_name, resolved, dir_resolved,
                )
                continue
            if resolved.is_file() and resolved.suffix == ".gguf":
                return resolved
            if not model_name.endswith(".gguf"):
                candidate_gguf = d / f"{model_name}.gguf"
                resolved_gguf = candidate_gguf.resolve()
                if (
                    _is_within_dir(d, candidate_gguf)
                    and resolved_gguf.is_file()
                ):
                    return resolved_gguf
        return None

    def _get_or_load(self, model_name: str) -> Any:
        """Get a cached model or load it from disk.

        Thread safety (IB-02): the fast path returns an already-loaded
        model without taking any contended lock, so inference on a loaded
        model is never blocked by an unrelated load. A first use acquires
        a per-model load lock and re-checks the cache (double-checked
        locking), so the model is constructed exactly once even under
        concurrent first use -- no double GGUF load, no race on the
        _loaded_models dict.
        """
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is not installed "
                "(pip install llama-cpp-python)"
            )

        # Fast path: dict.get is atomic under the GIL; no lock for a hit.
        cached = self._loaded_models.get(model_name)
        if cached is not None:
            return cached

        # Slow path: serialize loads of THIS model so it loads exactly once.
        with self._lock_for(self._load_locks, model_name):
            cached = self._loaded_models.get(model_name)
            if cached is not None:
                return cached

            gguf_path = self._resolve_model_path(model_name)
            if gguf_path is None:
                raise FileNotFoundError(
                    f"GGUF model not found: {model_name}. "
                    f"Searched directories: {[str(d) for d in self._model_dirs]}"
                )

            # The path is now proven to sit inside a configured model
            # directory. Nothing has yet proven the bytes are the bytes we
            # pinned, and they are about to be parsed by native code, so the
            # integrity gate runs here: after containment, before the load.
            #
            # This is the ONLY backend wired to the gate. model_provenance
            # PROVENANCE_GATED_BACKENDS records that fact, and the escalation
            # preflight keys off it. If the gate is added to another backend,
            # that set must gain its name, or a fortress will escalate into a
            # brick the preflight no longer sees.
            _provenance_guard(gguf_path)

            logger.info("Loading GGUF model: %s", gguf_path)
            start = time.time()

            kwargs: dict[str, Any] = {
                "model_path": str(gguf_path),
                "n_ctx": self._n_ctx,
                "n_gpu_layers": self._n_gpu_layers,
                "verbose": False,
            }
            if self._n_threads is not None:
                kwargs["n_threads"] = self._n_threads
            # S259 perf knobs: flash attention is a plain boolean; the
            # KV-cache type names resolve against the installed
            # llama-cpp-python's GGML_TYPE_* constants. Fail-open: an
            # unresolvable name is skipped with a warning, never blocking
            # the load (the unquantized default then applies).
            if self._flash_attn:
                kwargs["flash_attn"] = True
            for knob, value in (("type_k", self._type_k),
                                ("type_v", self._type_v)):
                if not value:
                    continue
                resolved = _resolve_ggml_kv_type(value)
                if resolved is None:
                    logger.warning(
                        "KV cache type %r for %s not exposed by "
                        "llama-cpp-python; using the default",
                        value,
                        knob,
                    )
                else:
                    kwargs[knob] = resolved

            # S226 (R-03): optional, off-by-default process-wide rlimits,
            # applied at most once per process BEFORE the first in-process
            # load (resource_governor.yaml, rlimits.enabled). Caveat: the
            # limits cap the ENTIRE process, not this backend alone (why
            # the knob is off by default). Fail-open: an absent module or
            # a raising applier never blocks the load.
            try:
                from opti_oignon.resource_governor import (
                    apply_llamacpp_rlimits,
                )

                apply_llamacpp_rlimits()
            except Exception:
                logger.debug("rlimit hook unavailable; load continues")

            llm = _LlamaCpp(**kwargs)
            elapsed = time.time() - start
            logger.info(
                "GGUF model loaded in %.1fs: %s", elapsed, gguf_path.name
            )

            self._loaded_models[model_name] = llm
            return llm

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory.

        IB-04 (S215 pick-up): pop-based removal -- the previous
        ``in``-then-``del`` was a TOCTOU that could raise ``KeyError``
        under concurrent unload of the same name.
        """
        unloaded = self._loaded_models.pop(model_name, None) is not None
        if unloaded:
            logger.info("Unloaded GGUF model: %s", model_name)
        return unloaded

    def unload_all(self) -> int:
        """Unload all cached models.

        IB-04 (S215 pick-up): pop-based drain over a key snapshot, so a
        concurrent unload of one name cannot raise and the count reflects
        what this call actually removed.
        """
        count = 0
        for name in list(self._loaded_models.keys()):
            if self._loaded_models.pop(name, None) is not None:
                count += 1
        logger.info("Unloaded %d model(s)", count)
        return count


# ---------------------------------------------------------------------------
# llama-server backend (S259): the external-process seam
# ---------------------------------------------------------------------------

class LlamaServerBackend(InferenceBackend):
    """Backend speaking to an EXTERNAL llama-server over HTTP (S259).

    This is the process-isolated counterpart to LlamaCppBackend: the
    server runs outside the Opti-Oignon process (an inference OOM can
    never take the API down), carries the llama.cpp performance surface
    this codebase cannot reach in-process (speculative decoding via the
    --draft-* flags, MTP self-drafting models, KV-cache quantization,
    flash attention), and is consumed here through the server's
    OpenAI-compatible endpoints using only the stdlib.

    Launching the server is deliberately NOT this class's job: the argv
    is built by speculative_decoding.build_llama_server_command and the
    process lifecycle is host-side (INFERENCE_PERF_S259.md), the same
    external_advisory posture as the governor's ollama_limits. Every
    method degrades honestly when the server is unreachable: health is
    False, listings are empty, lookups are None, and generation raises
    RuntimeError -- never a silent fallback to another engine.
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:8080",
        timeout_s: float = 5.0,
    ):
        self._host = host.rstrip("/")
        self._timeout_s = max(0.05, float(timeout_s))

    @property
    def name(self) -> str:
        return "llama_server"

    @property
    def display_name(self) -> str:
        return "llama.cpp server"

    # -- transport ---------------------------------------------------------

    def _request(
        self,
        path: str,
        payload: dict | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        """One guarded HTTP round trip; raises RuntimeError when the
        server is unreachable or answers a non-JSON body."""
        url = f"{self._host}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(
                req, timeout=timeout_s or self._timeout_s
            ) as resp:
                body = resp.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"llama-server unreachable at {self._host}: {exc}"
            ) from exc
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"llama-server returned a non-JSON body from {path}"
            ) from exc

    # -- ABC surface --------------------------------------------------------

    def health_check(self) -> bool:
        try:
            self._request("/health")
            return True
        except RuntimeError:
            return False

    def list_models(self) -> list[BackendModelInfo]:
        try:
            data = self._request("/v1/models")
        except RuntimeError:
            return []
        items = data.get("data") if isinstance(data, dict) else None
        out: list[BackendModelInfo] = []
        for item in items or []:
            if isinstance(item, dict) and item.get("id"):
                out.append(
                    BackendModelInfo(
                        name=str(item["id"]),
                        backend=self.name,
                        path=str(item.get("id")),
                    )
                )
        return out

    def model_info(self, model_name: str) -> BackendModelInfo | None:
        for info in self.list_models():
            if info.name == model_name:
                return info
        return None

    def generate(
        self,
        model: str,
        messages: list[dict] | None = None,
        options: dict | None = None,
        keep_alive: str = "30m",
        think: bool = False,
        images: list | None = None,
        prompt: str | None = None,
    ) -> ChatResponse:
        """Non-streaming chat through /v1/chat/completions.

        ``keep_alive`` and ``images`` ride the unified signature but are
        Ollama-side concepts; the server ignores them. ``prompt`` is a
        convenience wrapper for a single user message.
        """
        msgs = list(messages or [])
        if prompt is not None:
            msgs.append({"role": "user", "content": str(prompt)})
        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "stream": False,
        }
        for key in ("temperature", "top_p", "max_tokens"):
            if options and key in options:
                payload[key] = options[key]
        start = time.time()
        data = self._request(
            "/v1/chat/completions", payload, timeout_s=max(self._timeout_s, 30.0)
        )
        choices = data.get("choices") if isinstance(data, dict) else None
        content = ""
        if choices and isinstance(choices[0], dict):
            content = str(
                (choices[0].get("message") or {}).get("content") or ""
            )
        return ChatResponse(
            content=content,
            model=str(data.get("model", model)) if isinstance(data, dict) else model,
            done=True,
            total_duration=int((time.time() - start) * 1e9),
            extra={"backend": self.name},
        )

    def stream(
        self,
        model: str,
        messages: list[dict] | None = None,
        options: dict | None = None,
        keep_alive: str = "30m",
        think: bool = False,
        images: list | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Streaming chat through the server's SSE channel."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages or []),
            "stream": True,
        }
        for key in ("temperature", "top_p", "max_tokens"):
            if options and key in options:
                payload[key] = options[key]
        url = f"{self._host}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self._timeout_s)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"llama-server unreachable at {self._host}: {exc}"
            ) from exc
        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk_body = line[len("data:"):].strip()
                if chunk_body == "[DONE]":
                    yield StreamChunk(done=True, model=model)
                    return
                try:
                    data = json.loads(chunk_body)
                except ValueError:
                    continue
                choices = data.get("choices") or []
                delta = (
                    choices[0].get("delta", {})
                    if choices and isinstance(choices[0], dict)
                    else {}
                )
                content = str(delta.get("content") or "")
                if content:
                    yield StreamChunk(content=content, model=model)
        yield StreamChunk(done=True, model=model)


# ---------------------------------------------------------------------------
# Backend registry (singleton)
# ---------------------------------------------------------------------------

class BackendRegistry:
    """Manages available inference backends and active backend selection.

    Singleton pattern: use get_backend_registry() to obtain the instance.
    """

    def __init__(self):
        self._backends: dict[str, InferenceBackend] = {}
        self._active_name: str | None = None
        self._lock = threading.Lock()
        # BR: model -> backend name, for resolve_backend (positive resolutions
        # only; cleared whenever the backend set changes).
        self._route_cache: dict[str, str] = {}

    def register(self, backend: InferenceBackend) -> None:
        """Register a backend."""
        with self._lock:
            self._backends[backend.name] = backend
            self._route_cache.clear()
            logger.info("Registered inference backend: %s", backend.display_name)

    def unregister(self, name: str) -> bool:
        """Remove a backend from the registry."""
        with self._lock:
            if name in self._backends:
                if self._active_name == name:
                    self._active_name = None
                del self._backends[name]
                self._route_cache.clear()
                return True
            return False

    def get(self, name: str) -> InferenceBackend | None:
        """Get a backend by name."""
        return self._backends.get(name)

    def backends(self) -> list[InferenceBackend]:
        """Snapshot of the registered backend objects (S215).

        Taken under the lock so callers (the emergency-stop unload step)
        iterate a stable list while registration may happen concurrently.
        """
        with self._lock:
            return list(self._backends.values())

    @property
    def active(self) -> InferenceBackend | None:
        """Return the currently active backend."""
        if self._active_name:
            return self._backends.get(self._active_name)
        for b in self._backends.values():
            try:
                if b.health_check():
                    return b
            except Exception:
                continue
        return None

    @property
    def active_name(self) -> str | None:
        """Return the name of the active backend."""
        return self._active_name

    def activate(self, name: str) -> bool:
        """Set the active backend by name."""
        if name not in self._backends:
            logger.warning("Cannot activate unknown backend: %s", name)
            return False
        with self._lock:
            self._active_name = name
            logger.info("Active inference backend: %s", name)
            return True

    def resolve_backend(self, model: str) -> InferenceBackend | None:
        """Select the backend that should serve ``model`` (BR, per-model routing).

        Health-gated probe of each registered backend's ``model_info(model)``: a
        backend that recognises the model is a candidate. The active backend is
        preferred when it is itself a candidate (stability -- no needless
        switch); otherwise the first healthy recogniser is returned. When no
        backend recognises the model the active backend is returned, so a
        single-backend deployment behaves exactly as before (backward
        compatible). Returns None only when there is no usable backend at all,
        matching ``active`` -- the executor's existing ``if backend:`` guard then
        falls through to its direct path.

        Resolutions are cached (model -> backend name); the whole cache is
        cleared on register/unregister (a topology change), a cache hit is
        re-``health_check``ed so a backend that died since caching is never
        served, and only positive resolutions are cached (an unrecognised model
        is not pinned to the fallback -- a later pull may make a backend
        recognise it). ``model_info`` may hit the network for Ollama or the
        filesystem for llama.cpp, so the cache removes that cost on the hot path.
        In Bulbe an HTTP backend fails ``health_check`` and is never resolved.
        """
        cached_name = self._route_cache.get(model)
        if cached_name is not None:
            cached = self._backends.get(cached_name)
            if cached is not None:
                try:
                    if cached.health_check():
                        return cached
                except Exception:
                    pass
            self._route_cache.pop(model, None)
        active = self.active
        candidates: list[InferenceBackend] = []
        for backend in self.backends():
            try:
                if not backend.health_check():
                    continue
                if backend.model_info(model) is not None:
                    candidates.append(backend)
            except Exception:
                continue
        if not candidates:
            return active
        if active is not None and any(active is c for c in candidates):
            chosen = active
        else:
            chosen = candidates[0]
        self._route_cache[model] = chosen.name
        return chosen

    def list_backends(self) -> list[dict]:
        """Return status of all registered backends."""
        result = []
        for b in self._backends.values():
            healthy = False
            try:
                healthy = b.health_check()
            except Exception:
                pass
            result.append({
                "name": b.name,
                "display_name": b.display_name,
                "healthy": healthy,
                "active": b.name == self._active_name,
                "model_count": len(b.list_models()) if healthy else 0,
            })
        return result

    def all_models(self) -> list[BackendModelInfo]:
        """List models from all healthy backends."""
        models = []
        for b in self._backends.values():
            try:
                if b.health_check():
                    models.extend(b.list_models())
            except Exception:
                continue
        return models


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry_instance: BackendRegistry | None = None
_registry_lock = threading.Lock()


def get_backend_registry() -> BackendRegistry:
    """Return the global BackendRegistry singleton.

    On first call, registers OllamaBackend and (if available)
    LlamaCppBackend with default settings. Configuration from
    backends.yaml is applied by init_backends_from_config().
    """
    global _registry_instance
    if _registry_instance is not None:
        return _registry_instance

    with _registry_lock:
        if _registry_instance is not None:
            return _registry_instance

        registry = BackendRegistry()

        # Always register Ollama backend
        if OLLAMA_AVAILABLE:
            registry.register(OllamaBackend())
            registry.activate("ollama")

        # Register llama.cpp backend if available
        if LLAMA_CPP_AVAILABLE:
            registry.register(LlamaCppBackend())
            if not OLLAMA_AVAILABLE:
                registry.activate("llama_cpp")

        _registry_instance = registry
        return _registry_instance


def init_backends_from_config(config_path: str | None = None) -> BackendRegistry:
    """Initialize backends from backends.yaml configuration.

    Loads configuration and applies settings to the registry.
    Called during application startup.
    """
    registry = get_backend_registry()

    cfg = _load_backend_config(config_path)
    if not cfg:
        return registry

    # Apply Ollama settings
    ollama_cfg = cfg.get("ollama", {})
    if ollama_cfg and OLLAMA_AVAILABLE:
        ollama_backend = registry.get("ollama")
        if ollama_backend and isinstance(ollama_backend, OllamaBackend):
            host = ollama_cfg.get("host", "http://localhost:11434")
            ollama_backend._host = host

    # Apply llama.cpp settings
    llama_cfg = cfg.get("llama_cpp", {})
    if llama_cfg:
        model_dirs = llama_cfg.get("model_dirs", [])
        n_ctx = llama_cfg.get("n_ctx", 4096)
        n_gpu_layers = llama_cfg.get("n_gpu_layers", -1)
        n_threads = llama_cfg.get("n_threads")
        # S259 perf knobs (inert when absent).
        flash_attn = bool(llama_cfg.get("flash_attn", False))
        type_k = llama_cfg.get("type_k")
        type_v = llama_cfg.get("type_v")

        if LLAMA_CPP_AVAILABLE:
            llama_backend = LlamaCppBackend(
                model_dirs=model_dirs,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                flash_attn=flash_attn,
                type_k=type_k,
                type_v=type_v,
            )
            registry.register(llama_backend)
        elif model_dirs:
            logger.info(
                "llama-cpp-python not installed; GGUF model directories "
                "configured but llama.cpp backend unavailable"
            )

    # S259: register the external llama-server seam when configured.
    # Registration is config presence, not reachability -- availability
    # is the backend's health_check, asked at use time; the process
    # itself is launched host-side (INFERENCE_PERF_S259.md), never here.
    server_cfg = cfg.get("llama_server", {})
    if isinstance(server_cfg, dict) and server_cfg:
        registry.register(
            LlamaServerBackend(
                host=str(server_cfg.get("host", "http://127.0.0.1:8080")),
                timeout_s=float(server_cfg.get("timeout_s", 5.0) or 5.0),
            )
        )

    # Apply default backend selection
    default_backend = cfg.get("default_backend")
    if default_backend and registry.get(default_backend):
        registry.activate(default_backend)

    return registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_images(messages: list[dict], images: list) -> list[dict]:
    """Inject images into the last user message (ollama >= 0.5 convention).

    Returns a shallow copy of messages to avoid mutating the original.
    """
    messages = [m.copy() for m in messages]
    for msg in reversed(messages):
        if msg.get("role") == "user":
            msg["images"] = images
            break
    return messages


def _format_messages_for_llama_cpp(messages: list[dict]) -> list[dict]:
    """Ensure messages are in the format llama-cpp-python expects.

    llama-cpp-python uses the OpenAI-compatible format:
    [{"role": "...", "content": "..."}]
    """
    formatted = []
    for msg in messages:
        formatted.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        })
    return formatted


def _parse_gguf_filename(path: Path) -> BackendModelInfo:
    """Extract model info from a GGUF filename.

    Common naming convention: ModelName-Size-Quant.gguf
    e.g. llama-3.1-8b-instruct-Q4_K_M.gguf
    """
    stem = path.stem
    size_bytes = None
    try:
        size_bytes = path.stat().st_size
    except OSError:
        pass

    size_str = None
    if size_bytes:
        if size_bytes >= 1_000_000_000:
            size_str = f"{size_bytes / 1_000_000_000:.1f}GB"
        elif size_bytes >= 1_000_000:
            size_str = f"{size_bytes / 1_000_000:.1f}MB"
        else:
            size_str = f"{size_bytes}B"

    # Try to extract quantization from filename
    quant = None
    for token in stem.split("-"):
        token_up = token.upper()
        if token_up.startswith("Q") and any(c.isdigit() for c in token_up):
            quant = token_up
            break
        if token_up in ("F16", "F32", "BF16"):
            quant = token_up
            break

    modified = None
    try:
        mtime = path.stat().st_mtime
        import datetime
        modified = datetime.datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        pass

    return BackendModelInfo(
        name=path.name,
        backend="llama_cpp",
        size=size_str,
        quantization_level=quant,
        modified_at=modified,
        path=str(path),
    )


def _load_backend_config(config_path: str | None = None) -> dict:
    """Load backends.yaml configuration."""
    if config_path:
        p = Path(config_path)
    else:
        p = Path(__file__).parent / "config" / "backends.yaml"

    if not p.is_file():
        logger.debug("No backends.yaml found at %s", p)
        return {}

    try:
        import yaml
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded backend config from %s", p)
        return data
    except Exception as exc:
        logger.warning("Failed to load backends.yaml: %s", exc)
        return {}
