#!/usr/bin/env python3
"""
INFERENCE BACKEND ABSTRACTION -- OPTI-OIGNON
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

# Telemetry integration -- lazy import to avoid circular deps.
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


# Resource Governor mechanical seam (spec Section 4.1) -- lazy and
# fail-open by construction: resolved per call (never cached, so a
# test-seeded or standalone module is reused as-is), absent or unavailable
# means proceed unguarded (the availability-control posture).


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
    """The internal hook at the six generate/stream heads and the embedding head.

    Six, not four: the external llama-server backend was the one that talks
    to a process the governor cannot see, and it was also the only one that
    never asked before sending. It asks now. The Ollama embedding head asks
    too: an embedding loads a model like any other request.

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

def _client_field(entry, name):
    """A field of a client answer, mapping or object; ``None`` when absent."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get(name)
    return getattr(entry, name, None)


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


class BackendLoadedModel:
    """One model a backend reports as resident right now.

    The five fields are the ones the warmup used to read from the client's
    ``ps()``; ``None`` in any of them means the backend did not say, never
    zero. ``size_vram`` is bytes.
    """

    __slots__ = ("name", "backend", "size_vram", "expires_at", "context_length", "digest")

    def __init__(
        self,
        name: str,
        backend: str,
        size_vram: int | None = None,
        expires_at: float | None = None,
        context_length: int | None = None,
        digest: str | None = None,
    ):
        self.name = name
        self.backend = backend
        self.size_vram = size_vram
        self.expires_at = expires_at
        self.context_length = context_length
        self.digest = digest

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "backend": self.backend,
            "size_vram": self.size_vram,
            "expires_at": self.expires_at,
            "context_length": self.context_length,
            "digest": self.digest,
        }


# What ``endpoint`` answers for a backend whose requests never leave the process.
ENDPOINT_IN_PROCESS = "in-process"


def _ollama_host_parser() -> Any:
    """The client library's own resolution of its host, or None when it cannot be read."""
    try:
        from ollama._client import _parse_host
    except Exception:  # noqa: BLE001 - an unreadable resolution is unknown
        return None
    return _parse_host


def _field(entry: Any, name: str, default: Any = None) -> Any:
    """A field of a client entry in either of its shapes: mapping or object."""
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _loaded_model_from_ps(entry: Any, backend: str) -> BackendLoadedModel | None:
    """One ``ps()`` entry as a loaded-model record, in either response form.

    A name is taken from ``name`` then ``model``; an entry naming nothing is
    skipped. ``size_vram`` is coerced through ``__int__`` (the client's
    ``ByteSize``), ``expires_at`` from a datetime to a timestamp.
    """
    name = _field(entry, "name") or _field(entry, "model")
    if not name:
        return None
    size_vram = _field(entry, "size_vram")
    if size_vram is not None and hasattr(size_vram, "__int__"):
        size_vram = int(size_vram)
    expires_at = _field(entry, "expires_at")
    if expires_at is not None:
        if hasattr(expires_at, "timestamp"):
            expires_at = expires_at.timestamp()
        elif isinstance(expires_at, (int, float)):
            expires_at = float(expires_at)
        else:
            expires_at = None
    digest = _field(entry, "digest")
    return BackendLoadedModel(
        name=str(name),
        backend=backend,
        size_vram=size_vram,
        expires_at=expires_at,
        context_length=_field(entry, "context_length"),
        digest=str(digest) if digest else None,
    )


class ChatResponse:
    """Unified non-streaming chat response."""

    __slots__ = (
        "content", "thinking", "model", "done", "total_duration", "extra",
        "tool_calls",
    )

    def __init__(
        self,
        content: str,
        thinking: str | None = None,
        model: str = "",
        done: bool = True,
        total_duration: int | None = None,
        extra: dict | None = None,
        tool_calls: list | None = None,
    ):
        self.content = content
        self.thinking = thinking
        self.model = model
        self.done = done
        self.total_duration = total_duration
        self.extra = extra or {}
        # The calls the model made, normalised to name and arguments. Empty
        # when it made none -- an answer that called nothing says so.
        self.tool_calls = list(tool_calls or [])

    def to_dict(self) -> dict:
        """Serialize to dictionary matching ollama response format."""
        result = {
            "message": {"role": "assistant", "content": self.content},
            "model": self.model,
            "done": self.done,
        }
        if self.thinking:
            result["message"]["thinking"] = self.thinking
        if self.tool_calls:
            # The client's own shape, so parse_native_tool_calls keeps
            # working unchanged on a response that came through the registry.
            result["message"]["tool_calls"] = [
                {"function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in self.tool_calls
            ]
        if self.total_duration is not None:
            result["total_duration"] = self.total_duration
        return result


class StreamChunk:
    """Unified streaming chunk.

    ``extra`` carries what the engine reported on this chunk and nothing
    else: the final Ollama chunk names its token counts and durations, an
    earlier chunk names nothing and carries an empty dict. A consumer that
    counts tokens reads the reported count when there is one and knows,
    from its absence, when it is only counting chunks.
    """

    __slots__ = ("content", "thinking", "done", "model", "extra")

    def __init__(
        self,
        content: str = "",
        thinking: str = "",
        done: bool = False,
        model: str = "",
        extra: dict | None = None,
    ):
        self.content = content
        self.thinking = thinking
        self.done = done
        self.model = model
        self.extra = extra or {}

    def to_dict(self) -> dict:
        """Serialize to dictionary matching ollama chunk format."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.thinking:
            msg["thinking"] = self.thinking
        out = {
            "message": msg,
            "done": self.done,
            "model": self.model,
        }
        out.update(self.extra)
        return out


# The counts and durations an engine reports beside its answer. Copied onto
# ``extra`` when present, never defaulted: a missing count stays missing.
_REPORTED_FIELDS = (
    "eval_count", "prompt_eval_count", "total_duration", "load_duration",
    "prompt_eval_duration", "eval_duration",
)


def _reported(entry: Any) -> dict:
    """The reported fields present on a client response or chunk."""
    out: dict[str, Any] = {}
    for key in _REPORTED_FIELDS:
        value = _field(entry, key)
        if value is not None:
            out[key] = value
    return out


# Constrained decoding travels as an engine option, beside temperature and
# top_p, rather than as a new argument on the abstract interface: the schema
# IS an engine option, every existing caller keeps working, and each backend
# translates the one request into its own dialect instead of each caller
# learning three.
SCHEMA_OPTION = "schema"
# A tool list travels the same way, for the same reasons: native function
# calling is what makes an agent's tool selection reliable, and it existed
# only as a direct client call the registry never saw.
TOOLS_OPTION = "tools"
# A request timeout travels the same way and is bound to the transport, never
# forwarded to the engine: the modules that kept a client of their own did so
# for a per-timeout client, and a hung model call that blocks a pipeline for
# good is the defect that client existed to close.
TIMEOUT_OPTION = "timeout"


def _pop_timeout(engine_options: dict) -> tuple[dict, float | None]:
    """Take the transport timeout out of the engine options.

    Returns ``(options without it, seconds or None)``. A timeout that is not
    a number is refused before anything leaves, like an unusable schema.
    """
    opts = dict(engine_options or {})
    raw = opts.pop(TIMEOUT_OPTION, None)
    if raw is None:
        return opts, None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"{TIMEOUT_OPTION} must be a number of seconds, got {type(raw).__name__}"
        )
    return opts, float(raw)


def _split_extras(
    options: dict | None,
) -> tuple[dict, dict | None, list | None]:
    """Separate the schema and the tool list from the engine options.

    Returns ``(options without either, the schema or None, the tools or
    None)``. The caller's dict is copied, never mutated: taking either out in
    place would silently disarm every later reuse of the same options.

    A schema that is not an object, or a tool list that is not a list, is
    refused here rather than forwarded. Each engine would ignore an unusable
    value in its own way -- and an unconstrained answer that was supposed to
    be constrained, or a model that was never offered the tools it was
    supposed to choose from, is exactly the kind of silence this repository
    treats as a defect.
    """
    opts = dict(options or {})
    schema = opts.pop(SCHEMA_OPTION, None)
    tools = opts.pop(TOOLS_OPTION, None)
    if schema is not None and not isinstance(schema, dict):
        raise ValueError(
            f"{SCHEMA_OPTION} must be a JSON schema object, got "
            f"{type(schema).__name__}"
        )
    if tools is not None and not isinstance(tools, list):
        raise ValueError(
            f"{TOOLS_OPTION} must be a list of tool schemas, got "
            f"{type(tools).__name__}"
        )
    return opts, schema, tools


def _normalise_tool_calls(raw: Any) -> list[dict]:
    """Every engine's tool-call shape, reduced to name and arguments.

    The client library and the OpenAI-compatible surfaces both nest a
    ``function`` with a ``name`` and ``arguments``; the latter delivers the
    arguments as a JSON string, the former as a dict or an object. Anything
    without a name is dropped, and arguments that do not parse to an object
    become an empty one -- a call the model made is still reported, with the
    arguments it managed to express.
    """
    calls: list[dict] = []
    for call in raw or []:
        fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        if not name:
            continue
        args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": str(name), "arguments": args})
    return calls


def _response_format(schema: dict) -> dict:
    """The OpenAI-compatible spelling both llama.cpp surfaces accept."""
    return {"type": "json_object", "schema": schema}


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
    def list_models(self) -> list[BackendModelInfo] | None:
        """List all models available through this backend.

        ``None`` when the backend cannot say -- the client is absent, the
        request failed, nothing could be scanned -- and an empty list only
        when it looked and found nothing. The same doctrine as the two
        observation heads below: an empty list here would read as "nothing
        installed" where the truth is "nobody looked".
        """
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

    # The two heads below are defaulted, not abstract, and the default is the
    # honest answer: a backend that has not been taught to observe its loaded
    # set, or has no embedding endpoint, says so with ``None``. An empty list
    # here would read as "nothing is loaded" where the truth is "nobody
    # looked"; the governor and the warmup treat ``None`` as unknown.

    def loaded_models(self) -> list[BackendLoadedModel] | None:
        """The models this backend reports as resident right now.

        ``None`` when the backend cannot say; an empty list only when it
        looked and found nothing.
        """
        return None

    def embed(self, model: str, text: str, timeout: float | None = None) -> list[float] | None:
        """One embedding vector for ``text`` from ``model``.

        ``None`` when this backend has no embedding endpoint. A backend that
        has one and fails lets the failure propagate, as generate() does.
        ``timeout`` binds the transport when given.
        """
        return None

    def embed_many(
        self, model: str, texts: list[str], timeout: float | None = None,
    ) -> list[list[float]] | None:
        """One vector per text, in order, from one request.

        ``None`` when this backend has no embedding endpoint. An answer
        whose count does not match the texts is refused by name rather than
        handed on: a caller that zips vectors with texts would pair a text
        with another's vector.
        """
        return None

    def endpoint(self) -> str | None:
        """The base URL this backend's requests actually go to.

        ``ENDPOINT_IN_PROCESS`` when they never leave the process, ``None``
        when the backend cannot say -- never a configured value the
        requests do not read.
        """
        return None


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
        # One client per requested timeout, built on first use and kept: the
        # transport binding the benchmark and reasoning modules used to keep
        # for themselves, now held here for every caller.
        self._clients: dict[float, Any] = {}
        self._clients_lock = threading.Lock()

    def _client_for(self, timeout: float) -> Any:
        """The cached client bound to ``timeout`` seconds."""
        with self._clients_lock:
            client = self._clients.get(timeout)
            if client is None:
                client = _ollama_module.Client(timeout=timeout)
                self._clients[timeout] = client
            return client

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

        Ollama exposes no dedicated unload endpoint; the documented
        eviction is a generate call with ``keep_alive=0``, which unloads the
        model immediately. Loaded models are enumerated via ``ps()`` (both
        the dict and the object response forms are handled, the CC-01
        class). Requires no privileges; stopping a systemd-managed Ollama
        service remains a documented host action. Per-model failures are
        logged and skipped so one stuck model never blocks the rest.

        Returns the number of successful eviction requests.
        """
        loaded = self.loaded_models()
        if not loaded:
            return 0
        count = 0
        for m in loaded:
            try:
                _ollama_module.generate(model=m.name, keep_alive=0)
                count += 1
            except Exception as exc:
                logger.warning("Ollama unload failed for %s: %s", m.name, exc)
        if count:
            logger.info("Requested Ollama eviction for %d model(s)", count)
        return count

    def loaded_models(self) -> list[BackendLoadedModel] | None:
        """The loaded set through ``ps()``, both response forms.

        ``None`` when the client is absent or ``ps()`` fails: an unknown,
        not an empty set. The warmup used to read this from the client
        itself and answer ``[]`` in both cases; the registry says which.
        """
        if not OLLAMA_AVAILABLE:
            return None
        try:
            ps_response = _ollama_module.ps()
        except Exception as exc:
            logger.debug("Ollama ps failed: %s", exc)
            return None
        raw_models = _field(ps_response, "models") or []
        out: list[BackendLoadedModel] = []
        for entry in raw_models:
            record = _loaded_model_from_ps(entry, self.name)
            if record is not None:
                out.append(record)
        return out

    def endpoint(self) -> str | None:
        """Where the client library sends, resolved the way it resolves it.

        The client is built without a host, so it reads ``OLLAMA_HOST``, or
        its own default when that is unset. ``_host`` is written from
        ``backends.yaml`` and read by no request, so it is not the answer.
        """
        if not OLLAMA_AVAILABLE:
            return None
        parse = _ollama_host_parser()
        if parse is None:
            return None
        try:
            return str(parse(os.environ.get("OLLAMA_HOST")))
        except Exception:  # noqa: BLE001 - an unreadable resolution is unknown
            return None

    def _embed_client(self, timeout: float | None) -> Any:
        return _ollama_module if timeout is None else self._client_for(float(timeout))

    def embed(self, model: str, text: str, timeout: float | None = None) -> list[float] | None:
        """One vector through the client's ``embed``, after the governor.

        ``None`` without the client or when the client answers no vector; a
        client failure propagates. Admission is asked first: an embedding
        loads a model like any other request.
        """
        if not OLLAMA_AVAILABLE:
            return None
        _governor_admission(model, None)
        result = self._embed_client(timeout).embed(model=model, input=text)
        vectors = _field(result, "embeddings") or []
        if not vectors:
            return None
        return list(vectors[0])

    def embed_many(
        self, model: str, texts: list[str], timeout: float | None = None,
    ) -> list[list[float]] | None:
        """The whole batch through one ``embed`` of the client, after one admission."""
        if not OLLAMA_AVAILABLE:
            return None
        texts = list(texts)
        if not texts:
            return []
        _governor_admission(model, None)
        result = self._embed_client(timeout).embed(model=model, input=texts)
        vectors = _field(result, "embeddings") or []
        if len(vectors) != len(texts):
            raise ValueError(
                f"{model} answered {len(vectors)} vector(s) for {len(texts)} text(s): "
                f"refused, so no text is paired with another's vector"
            )
        return [list(v) for v in vectors]

    def unload_model(self, model_name: str) -> bool:
        """Evict ONE model from Ollama (the unload_all idiom
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

    def list_models(self) -> list[BackendModelInfo] | None:
        """List Ollama models via ollama.list(); ``None`` when nobody could look."""
        if not OLLAMA_AVAILABLE:
            return None
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
            logger.debug("Ollama list_models failed, listing unknown: %s", exc)
            return None

    def model_info(self, model_name: str) -> BackendModelInfo | None:
        """Get model details via ollama.show().

        The typed fields stay as they were. What the client reports beyond
        them travels in ``extra`` under its own name -- ``families``,
        ``parameters`` (the modelfile parameter text), ``digest``,
        ``template``, ``modelfile``, ``license`` and the raw ``model_info``
        mapping -- and only when the client reported it: a caller that
        used to read ``show()`` finds the same fields here, and never an
        invented one. Both response forms are read, the mapping and the
        ``ShowResponse`` object, whose mapping attribute is ``modelinfo``.
        """
        if not OLLAMA_AVAILABLE:
            return None
        try:
            info = _ollama_module.show(model_name)
        except Exception as exc:
            logger.debug("Ollama model_info(%s) failed: %s", model_name, exc)
            return None
        return self._parse_ollama_show(model_name, info, self.name)

    @staticmethod
    def _parse_ollama_show(model_name: str, info, backend_name: str) -> BackendModelInfo:
        """Fold a ``show()`` answer, mapping or object, into BackendModelInfo."""
        mapping = _client_field(info, "model_info")
        if not hasattr(mapping, "items"):
            mapping = _client_field(info, "modelinfo")
        if not hasattr(mapping, "items"):
            mapping = {}
        ctx_length = None
        for key, value in mapping.items():
            if "context_length" in str(key):
                try:
                    ctx_length = int(value)
                except (TypeError, ValueError):
                    ctx_length = None
                break
        details = _client_field(info, "details")
        extra: dict = {}
        families = _client_field(details, "families")
        if isinstance(families, (list, tuple)):
            extra["families"] = [str(f) for f in families]
        for key in ("parameters", "digest", "template", "modelfile", "license"):
            value = _client_field(info, key)
            if value:
                extra[key] = value
        if mapping:
            extra["model_info"] = dict(mapping)
        return BackendModelInfo(
            name=model_name,
            backend=backend_name,
            family=_client_field(details, "family"),
            parameter_size=_client_field(details, "parameter_size"),
            quantization_level=_client_field(details, "quantization_level"),
            context_length=ctx_length,
            extra=extra,
        )

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

        # Before the admission hook and before any telemetry: a malformed
        # schema is a refusal, and a refusal must not leave a started request
        # behind it.
        engine_options, schema, tools = _split_extras(options)
        engine_options, timeout = _pop_timeout(engine_options)

        # Governor admission hook (after the availability guard so
        # the "not installed" error semantics stay exactly as pinned).
        _governor_admission(model, options)

        # Telemetry start.
        tel = _get_telemetry()
        rid = tel.on_inference_start(model, messages) if tel else ""
        t0 = time.time()

        if images:
            messages = _inject_images(messages, images)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": engine_options,
            "keep_alive": keep_alive,
        }
        if think:
            kwargs["think"] = True
        if schema is not None:
            kwargs["format"] = schema
        if tools is not None:
            kwargs["tools"] = tools

        transport = _ollama_module if timeout is None else self._client_for(timeout)
        response = transport.chat(**kwargs)

        msg = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        raw_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        thinking_text = msg.get("thinking", "") if isinstance(msg, dict) else getattr(msg, "thinking", "")
        total_dur = response.get("total_duration") if isinstance(response, dict) else getattr(response, "total_duration", None)

        # Telemetry end.
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
            extra=_reported(response),
            tool_calls=_normalise_tool_calls(raw_calls),
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

        # Governor admission hook. A generator head runs at first
        # iteration; the funnel's ticket is thread-local, so funnels set it
        # on the consuming thread (see resource_governor.ticket_scope).
        _governor_admission(model, options)

        # Telemetry start.
        tel = _get_telemetry()
        rid = tel.on_inference_start(model, messages) if tel else ""
        t0 = time.time()
        token_count = 0

        if images:
            messages = _inject_images(messages, images)

        engine_options, schema, tools = _split_extras(options)
        engine_options, timeout = _pop_timeout(engine_options)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": engine_options,
            "stream": True,
            "keep_alive": keep_alive,
        }
        if think:
            kwargs["think"] = True
        if schema is not None:
            kwargs["format"] = schema
        if tools is not None:
            kwargs["tools"] = tools

        transport = _ollama_module if timeout is None else self._client_for(timeout)
        stream_iter = transport.chat(**kwargs)

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

            # Telemetry per-token.
            if content and tel and rid:
                tel.on_token_generated(rid, count=1)
                token_count += 1

            yield StreamChunk(
                content=content,
                thinking=thinking_text,
                done=bool(done),
                model=model,
                extra=_reported(chunk),
            )

        # Telemetry end.
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

        # What the client reports beyond the typed fields, under its own
        # name and only when reported: the size in bytes the formatted
        # ``size`` was made from, the digest, the family list.
        extra: dict = {}
        if size:
            try:
                extra["size_bytes"] = int(size)
            except (ValueError, TypeError):
                pass
        digest = _client_field(model_data, "digest")
        if digest:
            extra["digest"] = str(digest)
        families = _client_field(details, "families")
        if isinstance(families, (list, tuple)):
            extra["families"] = [str(f) for f in families]

        return BackendModelInfo(
            name=str(name),
            backend="ollama",
            size=size_str,
            family=family,
            parameter_size=param_size,
            quantization_level=quant,
            modified_at=modified,
            extra=extra,
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
    llama-cpp-python's GGML_TYPE_* constant.

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
    (opti_oignon.speculative_decoding); the prompt-level draft-verify path
    lives in opti_oignon.speculative. The external path is wired here: the argv
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
        # Perf knobs, inert by default: flash attention and KV-cache
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

    def endpoint(self) -> str | None:
        return ENDPOINT_IN_PROCESS

    @property
    def display_name(self) -> str:
        return "llama.cpp"

    def health_check(self) -> bool:
        """Check if llama-cpp-python is importable."""
        return LLAMA_CPP_AVAILABLE

    def list_models(self) -> list[BackendModelInfo] | None:
        """Scan configured directories for .gguf files.

        ``None`` when no configured directory exists, because nothing was
        scanned; an empty list only after a real scan that found nothing.
        """
        results = []
        seen = set()
        scanned = False
        for d in self._model_dirs:
            if not d.is_dir():
                continue
            scanned = True
            for gguf_path in sorted(d.glob("*.gguf")):
                if gguf_path.name in seen:
                    continue
                seen.add(gguf_path.name)
                info = _parse_gguf_filename(gguf_path)
                info.path = str(gguf_path)
                results.append(info)
        if not scanned:
            logger.debug("llama.cpp list_models: no configured model directory exists, listing unknown")
            return None
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
        # Governor admission hook (additive, signature untouched).
        _governor_admission(model, options)

        # Telemetry start.
        tel = _get_telemetry()
        rid = tel.on_inference_start(model, messages) if tel else ""
        t0 = time.time()

        llm = self._get_or_load(model)
        opts, schema, tools = _split_extras(options)
        temperature = opts.get("temperature", 0.7)

        formatted = _format_messages_for_llama_cpp(messages)

        completion_kwargs: dict[str, Any] = {
            "messages": formatted,
            "temperature": temperature,
            "max_tokens": opts.get("num_predict", 2048),
            "top_p": opts.get("top_p", 0.9),
            "stream": False,
        }
        if schema is not None:
            completion_kwargs["response_format"] = _response_format(schema)
        if tools is not None:
            completion_kwargs["tools"] = tools

        with self._lock_for(self._inference_locks, model):
            result = llm.create_chat_completion(**completion_kwargs)

        content = ""
        raw_calls = None
        if result and "choices" in result and result["choices"]:
            msg = result["choices"][0].get("message", {})
            content = msg.get("content", "")
            raw_calls = msg.get("tool_calls")

        # Telemetry end.
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
            tool_calls=_normalise_tool_calls(raw_calls),
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
        # Governor admission hook (additive, signature untouched).
        _governor_admission(model, options)

        # Telemetry start.
        tel = _get_telemetry()
        rid = tel.on_inference_start(model, messages) if tel else ""
        t0 = time.time()
        token_count = 0

        llm = self._get_or_load(model)
        opts, schema, tools = _split_extras(options)
        temperature = opts.get("temperature", 0.7)

        formatted = _format_messages_for_llama_cpp(messages)

        completion_kwargs: dict[str, Any] = {
            "messages": formatted,
            "temperature": temperature,
            "max_tokens": opts.get("num_predict", 2048),
            "top_p": opts.get("top_p", 0.9),
            "stream": True,
        }
        if schema is not None:
            completion_kwargs["response_format"] = _response_format(schema)
        if tools is not None:
            completion_kwargs["tools"] = tools

        with self._lock_for(self._inference_locks, model):
            stream_iter = llm.create_chat_completion(**completion_kwargs)

            for chunk in stream_iter:
                delta = {}
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "") or ""
                done = False
                if "choices" in chunk and chunk["choices"]:
                    done = chunk["choices"][0].get("finish_reason") is not None

                # Telemetry per-token.
                if content and tel and rid:
                    tel.on_token_generated(rid, count=1)
                    token_count += 1

                yield StreamChunk(
                    content=content,
                    done=done,
                    model=model,
                )

        # Telemetry end.
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

        Audit fix: NEVER accepts absolute paths or paths outside
        configured model_dirs.  Previously, an attacker could send
        model='/tmp/evil.gguf' via the API and load arbitrary GGUF files,
        potentially exploiting llama.cpp vulnerabilities or loading
        trojaned models.
        """
        # Reject absolute paths and path traversal
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
            # Perf knobs: flash attention is a plain boolean; the
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

            # Optional, off-by-default process-wide rlimits,
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

        pop-based removal -- the previous
        ``in``-then-``del`` was a TOCTOU that could raise ``KeyError``
        under concurrent unload of the same name.
        """
        unloaded = self._loaded_models.pop(model_name, None) is not None
        if unloaded:
            logger.info("Unloaded GGUF model: %s", model_name)
        return unloaded

    def unload_all(self) -> int:
        """Unload all cached models.

        pop-based drain over a key snapshot, so a
        concurrent unload of one name cannot raise and the count reflects
        what this call actually removed.
        """
        count = 0
        for name in list(self._loaded_models.keys()):
            if self._loaded_models.pop(name, None) is not None:
                count += 1
        logger.info("Unloaded %d model(s)", count)
        return count

    def loaded_models(self) -> list[BackendLoadedModel] | None:
        """The in-process set: a known answer, empty when nothing is loaded.

        VRAM size and expiry are not observed by this backend, so they stay
        ``None`` rather than a zero that would read as measured.
        """
        return [
            BackendLoadedModel(name=name, backend=self.name)
            for name in list(self._loaded_models.keys())
        ]


# ---------------------------------------------------------------------------
# llama-server backend: the external-process seam
# ---------------------------------------------------------------------------

class LlamaServerBackend(InferenceBackend):
    """Backend speaking to an EXTERNAL llama-server over HTTP.

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

    def endpoint(self) -> str | None:
        return self._host

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

    def list_models(self) -> list[BackendModelInfo] | None:
        """The server's listing; ``None`` when it could not be read."""
        try:
            data = self._request("/v1/models")
        except RuntimeError as exc:
            logger.debug("llama-server list_models failed, listing unknown: %s", exc)
            return None
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            logger.debug("llama-server list_models: the body carries no listing, listing unknown")
            return None
        out: list[BackendModelInfo] = []
        for item in items:
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
        for info in self.list_models() or []:
            if info.name == model_name:
                return info
        return None

    def slots(self) -> list[dict]:
        """The server's slot listing, read-only.

        A plain GET on the slots endpoint parsed as JSON; an unreachable
        server or a non-list body answers an empty list. Observability
        may degrade to silence, never to an exception -- the generation
        paths keep their own contract of raising instead.
        """
        try:
            data = self._request("/slots")
        except RuntimeError:
            return []
        return data if isinstance(data, list) else []

    def loaded_models(self) -> list[BackendLoadedModel] | None:
        """Unknown, by name: the slot listing does not say which model is
        resident and the model listing says what is served, not what is
        loaded. A reachable server still answers ``None`` here, not ``[]``.
        """
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
        engine_options, schema, tools = _split_extras(options)
        engine_options, timeout = _pop_timeout(engine_options)
        # Admission before anything leaves. A refusal that arrives after the
        # request has gone to the server is a log line, not a refusal.
        _governor_admission(model, options)
        options = engine_options
        payload: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "stream": False,
        }
        if schema is not None:
            payload["response_format"] = _response_format(schema)
        if tools is not None:
            payload["tools"] = tools
        # ``cache_prompt`` is the server's prompt-KV reuse switch and
        # ``id_slot`` names the slot whose cache is reused: both are
        # forwarded verbatim when the caller sets them, never invented.
        # Choosing the slot is a policy the caller owns, not this seam.
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "cache_prompt",
            "id_slot",
        ):
            if options and key in options:
                payload[key] = options[key]
        start = time.time()
        data = self._request(
            "/v1/chat/completions", payload,
            timeout_s=timeout if timeout is not None else max(self._timeout_s, 30.0),
        )
        choices = data.get("choices") if isinstance(data, dict) else None
        content = ""
        raw_calls = None
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "")
            raw_calls = message.get("tool_calls")
        return ChatResponse(
            content=content,
            model=str(data.get("model", model)) if isinstance(data, dict) else model,
            done=True,
            total_duration=int((time.time() - start) * 1e9),
            extra={"backend": self.name},
            tool_calls=_normalise_tool_calls(raw_calls),
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
        engine_options, schema, tools = _split_extras(options)
        engine_options, timeout = _pop_timeout(engine_options)
        # Same gate as the whole-answer head, and for the same reason. A
        # generator body runs at first iteration, so the caller's first
        # ``next`` is where admission is decided.
        _governor_admission(model, options)
        options = engine_options
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages or []),
            "stream": True,
        }
        if schema is not None:
            payload["response_format"] = _response_format(schema)
        if tools is not None:
            payload["tools"] = tools
        # Same forwarding contract as the non-streaming path: the
        # prompt-KV switch and the slot number ride only when the caller
        # set them.
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "cache_prompt",
            "id_slot",
        ):
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
            resp = urllib.request.urlopen(
                req, timeout=timeout if timeout is not None else self._timeout_s
            )
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
        """Snapshot of the registered backend objects.

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
        matching ``active`` -- and a caller that receives None refuses the
        request by name. Nothing falls through to the client behind the
        registry's back; the executor used to, and the path it took could
        only run in a state where the client was absent too.

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
            # An unknown listing is an unknown count, never zero: a healthy
            # backend that could not list says so with None on the wire.
            model_count: int | None = 0
            if healthy:
                listed = b.list_models()
                model_count = None if listed is None else len(listed)
            result.append({
                "name": b.name,
                "display_name": b.display_name,
                "healthy": healthy,
                "active": b.name == self._active_name,
                "model_count": model_count,
            })
        return result

    def all_models(self) -> list[BackendModelInfo]:
        """List models from all healthy backends.

        A backend whose listing is unknown contributes nothing and hides
        nothing else; the aggregate is what could be read.
        """
        models = []
        for b in self._backends.values():
            try:
                if b.health_check():
                    listed = b.list_models()
                    if listed is None:
                        logger.debug("Backend %s could not list its models; skipped in the aggregate", b.name)
                        continue
                    models.extend(listed)
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
        # Perf knobs (inert when absent).
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

    # Register the external llama-server seam when configured.
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

    # The core daemon, when core.yaml enables it: this process then asks the
    # daemon for every request, through this registry, so the funnel crosses
    # the process boundary. A malformed core configuration is reported, not
    # silently replaced by a local backend.
    try:
        from opti_oignon.core_client import register_from_config as _register_core

        _register_core(registry)
    except Exception as exc:  # noqa: BLE001 - reported, the local backends stand
        logger.warning("core daemon client not registered: %s", exc)

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
