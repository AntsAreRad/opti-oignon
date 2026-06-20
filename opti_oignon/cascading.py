#!/usr/bin/env python3
"""
Cascading Inference Engine -- S69

Reduces latency by routing queries through progressively larger models,
stopping at the first model whose response meets a quality threshold.

3-tier cascade (default):
  1. Fast tier   -- small model, handles simple queries
  2. Standard tier -- medium model, handles most queries
  3. Power tier  -- large model, handles complex queries

Each tier generates a response, scores it via heuristic quality evaluation,
and either serves it (if score >= threshold) or escalates to the next tier.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality evaluation import (from self_correction.py)
# ---------------------------------------------------------------------------

try:
    from opti_oignon.self_correction import compute_heuristic_quality, QualityResult
    QUALITY_EVAL_AVAILABLE = True
except ImportError:
    QUALITY_EVAL_AVAILABLE = False
    QualityResult = None

# ---------------------------------------------------------------------------
# Ollama import
# ---------------------------------------------------------------------------

try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    _ollama = None

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "cascading.yaml"


@dataclass
class CascadeTierConfig:
    """Configuration for a single cascade tier."""
    name: str
    model: str
    threshold: float
    max_tokens: int = 4096
    temperature: float = 0.5


@dataclass
class CascadeTierResult:
    """Result from a single tier attempt."""
    tier_name: str
    model: str
    response: str
    score: float
    latency_ms: float
    escalation_reason: str | None = None


@dataclass
class CascadeResult:
    """Final result of a cascading inference run."""
    final_response: str
    model_used: str
    tier_index: int
    tier_name: str
    score: float
    attempts: list[CascadeTierResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    escalation_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CascadingInference
# ---------------------------------------------------------------------------

class CascadingInference:
    """3-tier cascading inference engine with quality-based escalation.

    Iterates through tiers from fastest to most powerful, stopping at the
    first tier whose response score meets the configured threshold.
    """

    def __init__(self, config_path: str | Path | None = None):
        """Initialize from YAML config.

        Args:
            config_path: Path to cascading.yaml. If None, uses default location.
        """
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._raw_config: dict = {}
        self._enabled: bool = False
        self._tiers: list[CascadeTierConfig] = []
        self._max_retries_per_tier: int = 1
        self._timeout_per_tier: int = 30
        self._score_weights: dict[str, float] = {
            "completeness": 0.4,
            "coherence": 0.4,
            "hallucination_penalty": 0.2,
        }
        self._last_result: CascadeResult | None = None
        self._load_config()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                self._raw_config = raw
                self._enabled = raw.get("enabled", False)
                self._max_retries_per_tier = raw.get("max_retries_per_tier", 1)
                self._timeout_per_tier = raw.get("timeout_per_tier_seconds", 30)

                # Load score weights
                sw = raw.get("score_weights", {})
                if isinstance(sw, dict):
                    self._score_weights.update(sw)

                # Load tiers
                self._tiers = []
                for tier_data in raw.get("tiers", []):
                    if isinstance(tier_data, dict) and "name" in tier_data and "model" in tier_data:
                        self._tiers.append(CascadeTierConfig(
                            name=tier_data["name"],
                            model=tier_data["model"],
                            threshold=tier_data.get("threshold", 0.0),
                            max_tokens=tier_data.get("max_tokens", 4096),
                            temperature=tier_data.get("temperature", 0.5),
                        ))

                logger.info(
                    "Cascading config loaded: enabled=%s, tiers=%d",
                    self._enabled, len(self._tiers),
                )
            except Exception as e:
                logger.error("Failed to load cascading config: %s", e)
                self._set_defaults()
        else:
            logger.warning("Cascading config not found at %s, using defaults", self._config_path)
            self._set_defaults()

    def _set_defaults(self) -> None:
        """Set default tier configuration."""
        self._enabled = False
        self._tiers = [
            CascadeTierConfig(name="fast", model="qwen3:8b", threshold=0.7, max_tokens=2048),
            CascadeTierConfig(name="standard", model="qwen3:32b", threshold=0.6, max_tokens=4096),
            CascadeTierConfig(name="power", model="deepseek-r1:32b", threshold=0.0, max_tokens=8192),
        ]
        self._max_retries_per_tier = 1
        self._timeout_per_tier = 30

    def _save_config(self) -> None:
        """Persist current config to YAML."""
        data = {
            "enabled": self._enabled,
            "tiers": [
                {
                    "name": t.name,
                    "model": t.model,
                    "threshold": t.threshold,
                    "max_tokens": t.max_tokens,
                    "temperature": t.temperature,
                }
                for t in self._tiers
            ],
            "max_retries_per_tier": self._max_retries_per_tier,
            "timeout_per_tier_seconds": self._timeout_per_tier,
            "score_weights": dict(self._score_weights),
        }
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            logger.debug("Cascading config saved to %s", self._config_path)
        except Exception as e:
            logger.error("Failed to save cascading config: %s", e)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    @property
    def tiers(self) -> list[CascadeTierConfig]:
        return list(self._tiers)

    @property
    def tier_count(self) -> int:
        return len(self._tiers)

    @property
    def last_result(self) -> CascadeResult | None:
        return self._last_result

    @property
    def max_retries_per_tier(self) -> int:
        return self._max_retries_per_tier

    @property
    def timeout_per_tier(self) -> int:
        return self._timeout_per_tier

    @property
    def score_weights(self) -> dict[str, float]:
        return dict(self._score_weights)

    # ------------------------------------------------------------------
    # Tier management
    # ------------------------------------------------------------------

    def update_tiers(self, tiers: list[dict]) -> None:
        """Update tier configuration from a list of dicts.

        Args:
            tiers: List of tier dicts with keys: name, model, threshold,
                   max_tokens (optional), temperature (optional).
        """
        new_tiers = []
        for t in tiers:
            if isinstance(t, dict) and "name" in t and "model" in t:
                new_tiers.append(CascadeTierConfig(
                    name=t["name"],
                    model=t["model"],
                    threshold=t.get("threshold", 0.0),
                    max_tokens=t.get("max_tokens", 4096),
                    temperature=t.get("temperature", 0.5),
                ))
        if new_tiers:
            self._tiers = new_tiers
            logger.info("Cascading tiers updated: %d tiers", len(self._tiers))

    def update_config(
        self,
        enabled: bool | None = None,
        tiers: list[dict] | None = None,
        max_retries_per_tier: int | None = None,
        timeout_per_tier_seconds: int | None = None,
        score_weights: dict[str, float] | None = None,
    ) -> dict:
        """Update cascading configuration and persist to YAML.

        Args:
            enabled: Toggle cascading on/off.
            tiers: New tier definitions.
            max_retries_per_tier: Max retries within a tier.
            timeout_per_tier_seconds: Timeout per tier.
            score_weights: Quality score weights.

        Returns:
            Current config as dict.
        """
        if enabled is not None:
            self._enabled = bool(enabled)
        if tiers is not None:
            self.update_tiers(tiers)
        if max_retries_per_tier is not None:
            self._max_retries_per_tier = max(0, int(max_retries_per_tier))
        if timeout_per_tier_seconds is not None:
            self._timeout_per_tier = max(1, int(timeout_per_tier_seconds))
        if score_weights is not None and isinstance(score_weights, dict):
            self._score_weights.update(score_weights)

        self._save_config()
        return self.get_config()

    def get_config(self) -> dict:
        """Return current configuration as dict."""
        return {
            "enabled": self._enabled,
            "tiers": [
                {
                    "name": t.name,
                    "model": t.model,
                    "threshold": t.threshold,
                    "max_tokens": t.max_tokens,
                    "temperature": t.temperature,
                }
                for t in self._tiers
            ],
            "max_retries_per_tier": self._max_retries_per_tier,
            "timeout_per_tier_seconds": self._timeout_per_tier,
            "score_weights": dict(self._score_weights),
        }

    # ------------------------------------------------------------------
    # Quality evaluation
    # ------------------------------------------------------------------

    def evaluate_quality(self, query: str, response: str) -> float:
        """Evaluate response quality and return overall score.

        Uses compute_heuristic_quality from self_correction if available,
        otherwise falls back to a basic length/structure heuristic.

        Args:
            query: Original user query.
            response: LLM response to evaluate.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        if not response or not response.strip():
            return 0.0

        if QUALITY_EVAL_AVAILABLE:
            try:
                result = compute_heuristic_quality(query, response)
                # Apply custom weights if configured
                w = self._score_weights
                score = (
                    result.completeness_score * w.get("completeness", 0.4)
                    + result.coherence_score * w.get("coherence", 0.4)
                    + (1.0 - result.hallucination_risk) * w.get("hallucination_penalty", 0.2)
                )
                return round(min(1.0, max(0.0, score)), 3)
            except Exception as e:
                logger.debug("Quality evaluation error, using fallback: %s", e)

        # Fallback: basic heuristic
        return self._basic_quality_score(query, response)

    @staticmethod
    def _basic_quality_score(query: str, response: str) -> float:
        """Basic quality score when self_correction is unavailable."""
        words = response.split()
        word_count = len(words)
        if word_count < 3:
            return 0.1
        if word_count < 10:
            return 0.3

        q_words = len(query.split())
        length_score = min(1.0, word_count / max(q_words * 2, 30))

        # Check for sentence structure
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        structure_score = min(1.0, len(sentences) / 3)

        return round((length_score * 0.6 + structure_score * 0.4), 3)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        query: str,
        tier: CascadeTierConfig,
        task_type: str | None = None,
    ) -> str:
        """Call LLM for a specific tier.

        Args:
            query: User query.
            tier: Tier configuration.
            task_type: Optional task type for system prompt selection.

        Returns:
            LLM response text.

        Raises:
            RuntimeError: If Ollama is unavailable or call fails.
        """
        if not OLLAMA_AVAILABLE or _ollama is None:
            raise RuntimeError("Ollama is not available")

        system_prompt = (
            "You are a helpful assistant. Respond clearly and concisely."
        )

        try:
            response = _ollama.chat(
                model=tier.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                options={
                    "temperature": tier.temperature,
                    "num_predict": tier.max_tokens,
                },
                keep_alive="30m",
            )
            # Handle both dict and object response formats
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "")
            if hasattr(response, "message"):
                msg = response.message
                if hasattr(msg, "content"):
                    return msg.content or ""
                if isinstance(msg, dict):
                    return msg.get("content", "")
            return ""
        except Exception as e:
            raise RuntimeError(f"LLM call failed for {tier.model}: {e}") from e

    # ------------------------------------------------------------------
    # Cascade
    # ------------------------------------------------------------------

    def cascade(
        self,
        query: str,
        task_type: str | None = None,
        llm_call: Any = None,
    ) -> CascadeResult:
        """Run cascading inference across configured tiers.

        Iterates tiers from fastest to most powerful, stopping at the first
        tier whose response score meets the tier threshold.

        Args:
            query: User query to process.
            task_type: Optional task type hint (for future routing integration).
            llm_call: Optional callable(query, tier) -> str for testing.
                      If None, uses self._call_llm().

        Returns:
            CascadeResult with final response and all tier attempt details.
        """
        if not self._tiers:
            result = CascadeResult(
                final_response="[ERR] No tiers configured for cascading inference.",
                model_used="none",
                tier_index=-1,
                tier_name="none",
                score=0.0,
            )
            self._last_result = result
            return result

        call_fn = llm_call if llm_call is not None else self._call_llm
        attempts: list[CascadeTierResult] = []
        escalation_reasons: list[str] = []
        total_start = time.time()

        for tier_idx, tier in enumerate(self._tiers):
            retries = 0
            max_attempts = 1 + self._max_retries_per_tier

            while retries < max_attempts:
                tier_start = time.time()
                try:
                    response = call_fn(query, tier)
                    latency_ms = (time.time() - tier_start) * 1000

                    score = self.evaluate_quality(query, response)

                    tier_result = CascadeTierResult(
                        tier_name=tier.name,
                        model=tier.model,
                        response=response,
                        score=score,
                        latency_ms=round(latency_ms, 1),
                    )

                    if score >= tier.threshold:
                        # Quality threshold met -- serve this response
                        tier_result.escalation_reason = None
                        attempts.append(tier_result)
                        total_ms = (time.time() - total_start) * 1000

                        result = CascadeResult(
                            final_response=response,
                            model_used=tier.model,
                            tier_index=tier_idx,
                            tier_name=tier.name,
                            score=score,
                            attempts=attempts,
                            total_latency_ms=round(total_ms, 1),
                            escalation_reasons=escalation_reasons,
                        )
                        self._last_result = result
                        logger.info(
                            "Cascade resolved at tier %d (%s), score=%.3f, latency=%.0fms",
                            tier_idx, tier.name, score, total_ms,
                        )
                        return result

                    # Score below threshold -- decide retry or escalate
                    reason = (
                        f"Tier '{tier.name}' ({tier.model}): "
                        f"score {score:.3f} < threshold {tier.threshold}"
                    )
                    tier_result.escalation_reason = reason

                    if retries < self._max_retries_per_tier:
                        # Retry within same tier
                        retries += 1
                        logger.debug(
                            "Cascade tier %s retry %d/%d: score=%.3f",
                            tier.name, retries, self._max_retries_per_tier, score,
                        )
                        attempts.append(tier_result)
                        continue

                    # Max retries reached -- escalate
                    escalation_reasons.append(reason)
                    attempts.append(tier_result)
                    logger.info(
                        "Cascade escalating from tier %d (%s): score=%.3f < %.3f",
                        tier_idx, tier.name, score, tier.threshold,
                    )
                    break

                except Exception as e:
                    latency_ms = (time.time() - tier_start) * 1000
                    reason = f"Tier '{tier.name}' ({tier.model}): error -- {e}"
                    tier_result = CascadeTierResult(
                        tier_name=tier.name,
                        model=tier.model,
                        response="",
                        score=0.0,
                        latency_ms=round(latency_ms, 1),
                        escalation_reason=reason,
                    )
                    attempts.append(tier_result)

                    if retries < self._max_retries_per_tier:
                        retries += 1
                        logger.debug(
                            "Cascade tier %s error retry %d/%d: %s",
                            tier.name, retries, self._max_retries_per_tier, e,
                        )
                        continue

                    escalation_reasons.append(reason)
                    logger.warning(
                        "Cascade tier %s failed after %d attempts: %s",
                        tier.name, max_attempts, e,
                    )
                    break

        # All tiers exhausted -- return best attempt
        total_ms = (time.time() - total_start) * 1000

        best = max(attempts, key=lambda a: a.score) if attempts else None
        if best and best.response:
            result = CascadeResult(
                final_response=best.response,
                model_used=best.model,
                tier_index=next(
                    (i for i, t in enumerate(self._tiers) if t.name == best.tier_name),
                    len(self._tiers) - 1,
                ),
                tier_name=best.tier_name,
                score=best.score,
                attempts=attempts,
                total_latency_ms=round(total_ms, 1),
                escalation_reasons=escalation_reasons,
            )
        else:
            result = CascadeResult(
                final_response="[ERR] All cascade tiers failed to produce a response.",
                model_used="none",
                tier_index=-1,
                tier_name="none",
                score=0.0,
                attempts=attempts,
                total_latency_ms=round(total_ms, 1),
                escalation_reasons=escalation_reasons,
            )

        self._last_result = result
        logger.warning(
            "Cascade exhausted all %d tiers, best score=%.3f, latency=%.0fms",
            len(self._tiers), result.score, total_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current status for API consumption."""
        last = None
        if self._last_result is not None:
            last = {
                "model_used": self._last_result.model_used,
                "tier_name": self._last_result.tier_name,
                "tier_index": self._last_result.tier_index,
                "score": self._last_result.score,
                "total_latency_ms": self._last_result.total_latency_ms,
                "tiers_attempted": len(self._last_result.attempts),
            }
        return {
            "enabled": self._enabled,
            "tier_count": len(self._tiers),
            "tiers": [
                {
                    "name": t.name,
                    "model": t.model,
                    "threshold": t.threshold,
                    "max_tokens": t.max_tokens,
                    "temperature": t.temperature,
                }
                for t in self._tiers
            ],
            "last_result": last,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

CASCADING_AVAILABLE = True

try:
    cascading_inference = CascadingInference()
except Exception as _init_err:
    logger.error("Failed to initialize CascadingInference: %s", _init_err)
    cascading_inference = None
    CASCADING_AVAILABLE = False
