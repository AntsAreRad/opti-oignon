#!/usr/bin/env python3
"""
MODEL WARM-UP & KEEPALIVE - OPTI-OIGNON v1.4.0 (Session 24, F2)
=================================================================

Manage warm-up and keep-alive in VRAM for models Ollama.

Ollama unloads models after ~5 min of inactivity by default.
This module makes it possible to:
- Check which models are loaded in VRAM (ollama.ps())
- Warm up a model with a minimal request
- Keep models loaded via a periodic keepalive thread
- Integrate the keep_alive parameter into executor calls

NOTE: There are two complementary keepalive mechanisms:
  1. **Ollama keepalive** (ce module): Parameter `keep_alive` sent to
     ollama.chat()/ollama.generate() to prevent VRAM unloading.
     This module's thread sends periodic pings to renew
     the time before expiration (default 30min).
  2. **Gradio keepalive** (in executor.py): Yields empty strings
     while a model is loading to prevent the SSE timeout
     de Gradio (~30s). Warm-up reduces the time required
     of this mechanism by loading models before the first request.

Usage:
    from opti_oignon.model_warmup import model_warmup

    # Check status
    status = model_warmup.get_loaded_models()

    # Warm up un model
    model_warmup.warmup("qwen3:32b")

    # Launch the keepalive in the background
    model_warmup.start_keepalive(["qwen3:32b"], interval=240)

    # Combo: warmup + keepalive (ideal au demarrage de l'UI)
    model_warmup.warmup_in_background(
        ["qwen3:32b", "qwen3-coder:30b"],
        start_keepalive=True,
    )

Author: Leon
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Conditional ollama import
try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    _ollama = None  # type: ignore
    OLLAMA_AVAILABLE = False


# --- Dataclasses ---

@dataclass
class LoadedModel:
    """Information about a model currently loaded in VRAM.

    Attributes:
        name: Model name (e.g. "qwen3:32b")
        size_vram: Size in VRAM in bytes
        expires_at: Timestamp d'expiration du keepalive
        context_length: Supported context length
        digest: Model hash
    """

    name: str
    size_vram: int = 0
    expires_at: float | None = None
    context_length: int | None = None
    digest: str | None = None


@dataclass
class WarmupResult:
    """Result of a warm-up operation.

    Attributes:
        model: Model name
        success: True if the model was loaded
        duration: Loading time in seconds
        error: Message d'erreur si failed
        already_loaded: True if the model was already in VRAM
    """

    model: str
    success: bool
    duration: float = 0.0
    error: str | None = None
    already_loaded: bool = False


@dataclass
class WarmupStats:
    """Global warm-up system statistics.

    Attributes:
        total_warmups: Total number of warmups performed
        total_keepalives: Total number of keepalive pings sent
        warmup_errors: Number of warm-up errors
        keepalive_errors: Number of keepalive errors
        avg_warmup_time: Average warm-up time (secondes)
        models_warmed: Set of models warmed at least once
        keepalive_running: True if the keepalive thread is active
        keepalive_models: List of models kept alive
        keepalive_interval: Keepalive interval in seconds
    """

    total_warmups: int = 0
    total_keepalives: int = 0
    warmup_errors: int = 0
    keepalive_errors: int = 0
    avg_warmup_time: float = 0.0
    models_warmed: set = field(default_factory=set)
    keepalive_running: bool = False
    keepalive_models: list = field(default_factory=list)
    keepalive_interval: int = 0


# --- Default keepalive duration (sent to Ollama) ---
DEFAULT_KEEP_ALIVE = "30m"
# Interval between keepalive pings (seconds)
DEFAULT_KEEPALIVE_INTERVAL = 240  # 4 minutes (before the 5min default expiration)
# Minimal prompt for warm-up
WARMUP_PROMPT = "hi"


class ModelWarmup:
    """Manager for warm-up and keep-alive of models Ollama.

    Check loaded models, warm them up if needed,
    and maintains a keepalive thread to prevent their unloading.

    Thread-safe via locks internes.
    """

    def __init__(
        self,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        keepalive_interval: int = DEFAULT_KEEPALIVE_INTERVAL,
    ):
        """Initialize the model warmup manager.

        Args:
            keep_alive: Duration string for Ollama's keep_alive param (e.g. "30m")
            keepalive_interval: Seconds between keepalive pings
        """
        self._keep_alive = keep_alive
        self._keepalive_interval = keepalive_interval

        # Stats internes
        self._total_warmups = 0
        self._total_keepalives = 0
        self._warmup_errors = 0
        self._keepalive_errors = 0
        self._warmup_times: list[float] = []
        self._models_warmed: set = set()

        # Thread keepalive
        self._keepalive_thread: threading.Thread | None = None
        self._keepalive_stop = threading.Event()
        self._keepalive_models: list[str] = []
        self._lock = threading.Lock()

        # Optional callback for events
        self._on_warmup: Callable | None = None
        self._on_keepalive: Callable | None = None

    @property
    def keep_alive(self) -> str:
        """Current keep_alive duration string sent to Ollama."""
        return self._keep_alive

    @keep_alive.setter
    def keep_alive(self, value: str):
        """Set the keep_alive duration.

        Args:
            value: Duration string (e.g. "30m", "1h", "0" to disable)
        """
        self._keep_alive = value

    @property
    def keepalive_interval(self) -> int:
        """Seconds between keepalive pings."""
        return self._keepalive_interval

    @keepalive_interval.setter
    def keepalive_interval(self, value: int):
        """Set the keepalive ping interval.

        Args:
            value: Interval in seconds (minimum 30)
        """
        self._keepalive_interval = max(30, value)

    @property
    def is_keepalive_running(self) -> bool:
        """True if the keepalive background thread is active."""
        return (
            self._keepalive_thread is not None
            and self._keepalive_thread.is_alive()
        )

    def get_loaded_models(self) -> list[LoadedModel]:
        """Query Ollama for models currently loaded in VRAM.

        Returns:
            List of LoadedModel with VRAM info, empty if Ollama unavailable
        """
        if not OLLAMA_AVAILABLE:
            logger.debug("Ollama unavailable, no models loaded")
            return []

        try:
            ps_response = _ollama.ps()
            models = []

            # Handle both dict and object (ProcessResponse) forms: a dict has
            # .get, the object exposes .models. The previous unconditional
            # ps_response.get(...) raised AttributeError on the object form
            # (newer ollama clients) -> the outer except returned [] and the
            # intended getattr fallback below was never reached, so
            # is_model_loaded was always False and warmup never skipped.
            if isinstance(ps_response, dict):
                raw_models = ps_response.get("models", []) or []
            else:
                raw_models = getattr(ps_response, "models", []) or []

            for m in raw_models:
                # Handle dict or object
                if isinstance(m, dict):
                    name = m.get("name", m.get("model", "unknown"))
                    size_vram = m.get("size_vram", 0)
                    expires_at = m.get("expires_at", None)
                    context_length = m.get("context_length", None)
                    digest = m.get("digest", None)
                else:
                    name = getattr(m, "name", None) or getattr(m, "model", "unknown")
                    size_vram = getattr(m, "size_vram", 0) or 0
                    expires_at = getattr(m, "expires_at", None)
                    context_length = getattr(m, "context_length", None)
                    digest = getattr(m, "digest", None)

                # Convert expires_at to timestamp if it is a datetime
                expires_ts = None
                if expires_at is not None:
                    if hasattr(expires_at, "timestamp"):
                        expires_ts = expires_at.timestamp()
                    elif isinstance(expires_at, (int, float)):
                        expires_ts = float(expires_at)

                # Convert size_vram if it is a ByteSize object
                if hasattr(size_vram, "__int__"):
                    size_vram = int(size_vram)

                models.append(LoadedModel(
                    name=str(name),
                    size_vram=size_vram,
                    expires_at=expires_ts,
                    context_length=context_length,
                    digest=str(digest) if digest else None,
                ))

            logger.debug(f"{len(models)} models loaded in VRAM")
            return models

        except Exception as e:
            logger.warning(f"Erreur lors de ollama.ps(): {e}")
            return []

    def is_model_loaded(self, model: str) -> bool:
        """Check if a specific model is currently loaded in VRAM.

        Args:
            model: Model name to check (e.g. "qwen3:32b")

        Returns:
            True if the model is loaded
        """
        loaded = self.get_loaded_models()
        return any(m.name == model for m in loaded)

    def warmup(
        self,
        model: str,
        force: bool = False,
        timeout: float = 120.0,
    ) -> WarmupResult:
        """Warm up a model by loading it into VRAM.

        Sends a minimal prompt to force Ollama to load the model.
        If the model is already loaded, skips unless force=True.

        Args:
            model: Model name to warm up
            force: If True, warm up even if already loaded
            timeout: Maximum seconds to wait for loading

        Returns:
            WarmupResult with success status and timing
        """
        if not OLLAMA_AVAILABLE:
            return WarmupResult(
                model=model,
                success=False,
                error="Ollama not available",
            )

        # Check si already loaded
        if not force and self.is_model_loaded(model):
            logger.info(f"Model {model} already loaded in VRAM")
            return WarmupResult(
                model=model,
                success=True,
                already_loaded=True,
            )

        # Send a minimal request to force loading
        start = time.time()
        try:
            _ollama.generate(
                model=model,
                prompt=WARMUP_PROMPT,
                keep_alive=self._keep_alive,
                options={"num_predict": 1},  # Generate a single token
            )
            duration = time.time() - start

            with self._lock:
                self._total_warmups += 1
                self._warmup_times.append(duration)
                self._models_warmed.add(model)

            logger.info(f"Model {model} warmed up in {duration:.1f}s")

            if self._on_warmup:
                try:
                    self._on_warmup(model, duration, True)
                except Exception:
                    pass

            return WarmupResult(
                model=model,
                success=True,
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start
            with self._lock:
                self._warmup_errors += 1

            error_msg = str(e)
            logger.warning(f"Warm-up failed for {model}: {error_msg}")

            if self._on_warmup:
                try:
                    self._on_warmup(model, duration, False)
                except Exception:
                    pass

            return WarmupResult(
                model=model,
                success=False,
                duration=duration,
                error=error_msg,
            )

    def warmup_batch(
        self,
        models: list[str],
        force: bool = False,
        timeout: float = 120.0,
    ) -> list[WarmupResult]:
        """Warm up multiple models sequentially.

        Args:
            models: List of model names to warm up
            force: If True, warm up even if already loaded
            timeout: Maximum seconds per model

        Returns:
            List of WarmupResult for each model
        """
        results = []
        for model in models:
            result = self.warmup(model, force=force, timeout=timeout)
            results.append(result)
        return results

    def send_keepalive(self, model: str) -> bool:
        """Send a keepalive ping to keep a model loaded in VRAM.

        Uses ollama.generate with empty prompt and keep_alive parameter.

        Args:
            model: Model name to keep alive

        Returns:
            True if ping successful
        """
        if not OLLAMA_AVAILABLE:
            return False

        try:
            _ollama.generate(
                model=model,
                prompt="",
                keep_alive=self._keep_alive,
                options={"num_predict": 0},
            )
            with self._lock:
                self._total_keepalives += 1

            logger.debug(f"Keepalive ping: {model}")
            return True

        except Exception as e:
            with self._lock:
                self._keepalive_errors += 1

            logger.warning(f"Keepalive failed for {model}: {e}")
            return False

    def start_keepalive(
        self,
        models: list[str],
        interval: int | None = None,
        warmup_first: bool = True,
    ) -> threading.Thread:
        """Start a background thread that periodically pings models.

        Keeps specified models loaded in VRAM by sending periodic
        keepalive requests before Ollama's auto-unload timer expires.

        Args:
            models: Model names to keep alive
            interval: Seconds between pings (default: self.keepalive_interval)
            warmup_first: If True, warm up models before starting loop

        Returns:
            The background thread (already started)
        """
        # Arreter le thread existant
        self.stop_keepalive()

        if interval is not None:
            self._keepalive_interval = max(30, interval)

        self._keepalive_models = list(models)
        self._keepalive_stop.clear()

        def _keepalive_loop():
            """Background keepalive loop."""
            # Initial warm-up if requested
            if warmup_first:
                for model in self._keepalive_models:
                    if self._keepalive_stop.is_set():
                        return
                    self.warmup(model)

            # Ping loop
            while not self._keepalive_stop.is_set():
                # Wait the interval (with stop check)
                if self._keepalive_stop.wait(timeout=self._keepalive_interval):
                    break  # stop_keepalive() a ete appele

                for model in self._keepalive_models:
                    if self._keepalive_stop.is_set():
                        return
                    self.send_keepalive(model)

                    if self._on_keepalive:
                        try:
                            self._on_keepalive(model)
                        except Exception:
                            pass

        self._keepalive_thread = threading.Thread(
            target=_keepalive_loop,
            daemon=True,
            name="model-keepalive",
        )
        self._keepalive_thread.start()
        logger.info(
            f"Keepalive started for {models} "
            f"(interval={self._keepalive_interval}s)"
        )
        return self._keepalive_thread

    def stop_keepalive(self) -> None:
        """Stop the background keepalive thread."""
        if self._keepalive_thread is not None:
            self._keepalive_stop.set()
            self._keepalive_thread.join(timeout=5.0)
            self._keepalive_thread = None
            self._keepalive_models = []
            logger.info("Keepalive arrete")

    def warmup_in_background(
        self,
        models: list[str],
        callback: Callable | None = None,
        delay: float = 1.0,
        start_keepalive: bool = False,
    ) -> threading.Thread:
        """Launch warmup in a background thread with optional keepalive.

        Args:
            models: Model names to warm up
            callback: Optional callable(results: list) called when done
            delay: Delay in seconds before starting
            start_keepalive: If True, start keepalive loop after warmup

        Returns:
            The background thread (already started)
        """
        def _bg_warmup():
            if delay > 0:
                time.sleep(delay)

            results = self.warmup_batch(models)

            if callback:
                try:
                    callback(results)
                except Exception as e:
                    logger.warning(f"Erreur callback warmup: {e}")

            # Demarrer le keepalive if requested
            if start_keepalive:
                successful = [r.model for r in results if r.success]
                if successful:
                    self.start_keepalive(successful, warmup_first=False)

        thread = threading.Thread(
            target=_bg_warmup,
            daemon=True,
            name="model-warmup-bg",
        )
        thread.start()
        logger.info(f"Background warmup launched for {models}")
        return thread

    def get_stats(self) -> WarmupStats:
        """Get warmup and keepalive statistics.

        Returns:
            WarmupStats with counters and averages
        """
        with self._lock:
            avg_time = (
                sum(self._warmup_times) / len(self._warmup_times)
                if self._warmup_times
                else 0.0
            )
            return WarmupStats(
                total_warmups=self._total_warmups,
                total_keepalives=self._total_keepalives,
                warmup_errors=self._warmup_errors,
                keepalive_errors=self._keepalive_errors,
                avg_warmup_time=avg_time,
                models_warmed=set(self._models_warmed),
                keepalive_running=self.is_keepalive_running,
                keepalive_models=list(self._keepalive_models),
                keepalive_interval=self._keepalive_interval,
            )

    def get_vram_summary(self) -> dict[str, Any]:
        """Get a summary of VRAM usage by loaded models.

        Returns:
            Dict with total_vram, model_count, and per-model details
        """
        loaded = self.get_loaded_models()
        total_vram = sum(m.size_vram for m in loaded)
        return {
            "model_count": len(loaded),
            "total_vram_bytes": total_vram,
            "total_vram_gb": total_vram / (1024**3) if total_vram > 0 else 0.0,
            "models": [
                {
                    "name": m.name,
                    "vram_gb": m.size_vram / (1024**3) if m.size_vram else 0.0,
                    "context_length": m.context_length,
                }
                for m in loaded
            ],
        }

    def get_warmup_report(self) -> str:
        """Generate a formatted report of warmup status.

        Returns:
            Multi-line text report
        """
        stats = self.get_stats()
        loaded = self.get_loaded_models()

        lines = ["Model Warmup Status:"]
        lines.append(f"  Loaded in VRAM: {len(loaded)}")
        for m in loaded:
            vram_gb = m.size_vram / (1024**3) if m.size_vram else 0.0
            lines.append(f"    {m.name} ({vram_gb:.1f} GB VRAM)")

        lines.append(f"  Total warmups: {stats.total_warmups}")
        if stats.avg_warmup_time > 0:
            lines.append(f"  Avg warmup time: {stats.avg_warmup_time:.1f}s")
        lines.append(f"  Keepalive: {'running' if stats.keepalive_running else 'stopped'}")
        if stats.keepalive_running:
            lines.append(f"    Models: {', '.join(stats.keepalive_models)}")
            lines.append(f"    Interval: {stats.keepalive_interval}s")
            lines.append(f"    Pings sent: {stats.total_keepalives}")

        if stats.warmup_errors > 0 or stats.keepalive_errors > 0:
            lines.append(f"  Errors: {stats.warmup_errors} warmup, {stats.keepalive_errors} keepalive")

        return "\n".join(lines)

    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        with self._lock:
            self._total_warmups = 0
            self._total_keepalives = 0
            self._warmup_errors = 0
            self._keepalive_errors = 0
            self._warmup_times.clear()
            self._models_warmed.clear()

    def set_callbacks(
        self,
        on_warmup: Callable | None = None,
        on_keepalive: Callable | None = None,
    ) -> None:
        """Set event callbacks.

        Args:
            on_warmup: Called after warmup(model, duration, success)
            on_keepalive: Called after each keepalive ping(model)
        """
        self._on_warmup = on_warmup
        self._on_keepalive = on_keepalive

    def __del__(self):
        """Clean shutdown of the keepalive thread."""
        try:
            self.stop_keepalive()
        except Exception:
            pass


# Singleton module-level
model_warmup = ModelWarmup()

# Flag de disponibilite
MODEL_WARMUP_AVAILABLE = True
