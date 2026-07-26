#!/usr/bin/env python3
"""
CONSENSUS ENGINE - OPTI-OIGNON v1.5.4
==============================================

Multi-model consensus engine -- queries several models
in parallel, compares their responses, and selects or merges
the best response for increased reliability.

Trois strategies:
1. Best-of-N: Select the response closest to consensus
2. Weighted Vote: Weights by model quality tier
3. LLM Merge: a judge model synthesizes the best parts

Author: Leon
"""

import logging
import time
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Conditional import of the YAML config
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Conditional Ollama import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Conditional import of model profiles
try:
    from .model_profiles import ModelProfile, profile_manager
    PROFILES_AVAILABLE = True
except ImportError:
    PROFILES_AVAILABLE = False
    profile_manager = None
    ModelProfile = None

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class ModelResponse:
    """A single model's response."""
    model: str
    content: str
    duration_ms: int = 0
    success: bool = True
    error: str = ""
    quality_tier: str = "medium"


@dataclass
class ConsensusComparison:
    """Result of the inter-model comparison."""
    agreement_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    average_agreement: float = 0.0
    areas_of_agreement: list[str] = field(default_factory=list)
    areas_of_disagreement: list[str] = field(default_factory=list)


@dataclass
class ConsensusResult:
    """Full result of the multi-model consensus."""
    strategy: str  # "best_of_n", "weighted_vote", "llm_merge"
    selected_response: str = ""
    selected_model: str = ""
    confidence: float = 0.0
    individual_responses: list[ModelResponse] = field(default_factory=list)
    comparison: ConsensusComparison | None = None
    total_duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Strategies valides
STRATEGY_BEST_OF_N = "best_of_n"
STRATEGY_WEIGHTED_VOTE = "weighted_vote"
STRATEGY_LLM_MERGE = "llm_merge"

VALID_STRATEGIES = {STRATEGY_BEST_OF_N, STRATEGY_WEIGHTED_VOTE, STRATEGY_LLM_MERGE}

# Default weights by quality tier
DEFAULT_QUALITY_WEIGHTS = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
}


@dataclass
class ConsensusConfig:
    """Configuration of the consensus engine."""
    default_models: list[str] = field(default_factory=lambda: [
        "qwen3:32b", "deepseek-r1:32b", "nemotron-3-nano:30b",
    ])
    strategy: str = STRATEGY_BEST_OF_N
    judge_model: str | None = None
    max_models: int = 3
    timeout_per_model: int = 60
    min_agreement_threshold: float = 0.3
    temperature: float = 0.3
    quality_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_QUALITY_WEIGHTS))

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.strategy not in VALID_STRATEGIES:
            logger.warning(
                f"Invalid strategy '{self.strategy}', "
                f"fallback '{STRATEGY_BEST_OF_N}'"
            )
            self.strategy = STRATEGY_BEST_OF_N
        if self.max_models < 1:
            self.max_models = 1
        if self.timeout_per_model < 5:
            self.timeout_per_model = 5

    @classmethod
    def from_yaml(cls, path: str) -> "ConsensusConfig":
        """Load configuration from a YAML file.

        Args:
            path: YAML file path

        Returns:
            ConsensusConfig initialisee
        """
        if not YAML_AVAILABLE:
            logger.warning("PyYAML unavailable, using default config")
            return cls()

        p = Path(path)
        if not p.exists():
            logger.info(f"Config file missing: {path}, using default config")
            return cls()

        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data or "consensus" not in data:
                return cls()
            cfg = data["consensus"]
            _default_models = ["qwen3:32b", "deepseek-r1:32b", "nemotron-3-nano:30b"]
            return cls(
                default_models=cfg.get("default_models", _default_models),
                strategy=cfg.get("strategy", STRATEGY_BEST_OF_N),
                judge_model=cfg.get("judge_model"),
                max_models=cfg.get("max_models", 3),
                timeout_per_model=cfg.get("timeout_per_model", 60),
                min_agreement_threshold=cfg.get("min_agreement_threshold", 0.3),
                temperature=cfg.get("temperature", 0.3),
                quality_weights=cfg.get("quality_weights", dict(DEFAULT_QUALITY_WEIGHTS)),
            )
        except Exception as e:
            logger.warning(f"Consensus config read error: {e}")
            return cls()


# =============================================================================
# MOTEUR DE CONSENSUS
# =============================================================================

class ConsensusEngine:
    """Moteur de consensus multi-model.

    Queries several models in parallel, compares the responses,
    and selects or merges the best result.

    Usage:
        engine = ConsensusEngine()
        result = engine.run_consensus("What is X?", models=["m1", "m2", "m3"])
        result = engine.run_consensus("What is X?", strategy="llm_merge")
    """

    def __init__(
        self,
        config: ConsensusConfig | None = None,
        default_model: str = "qwen3:32b",
    ):
        """Initialize the consensus engine.

        Args:
            config: Configuration (loads YAML by default)
            default_model: Default model if none configured
        """
        if config is not None:
            self._config = config
        else:
            # Look for the config file in the standard locations
            config_path = Path(__file__).parent / "config" / "consensus.yaml"
            self._config = ConsensusConfig.from_yaml(str(config_path))

        self._default_model = default_model
        self._last_result: ConsensusResult | None = None

        logger.info(
            f"ConsensusEngine initialise: strategy={self._config.strategy}, "
            f"models={self._config.default_models}, "
            f"max_models={self._config.max_models}"
        )

    # ----------------------------------------------------------------
    # Proprietes
    # ----------------------------------------------------------------

    @property
    def config(self) -> ConsensusConfig:
        """Current configuration."""
        return self._config

    @property
    def last_result(self) -> ConsensusResult | None:
        """Last consensus result."""
        return self._last_result

    @property
    def available(self) -> bool:
        """Whether consensus is functional."""
        return OLLAMA_AVAILABLE

    # ----------------------------------------------------------------
    # Parallel execution of the models
    # ----------------------------------------------------------------

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Appelle un model LLM via Ollama.

        Args:
            messages: Messages to send
            model: Model name
            temperature: Temperature de generation

        Returns:
            Response text
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama is not available")

        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
            stream=False,
        )
        # Handle both dict and object (ChatResponse) response formats, mirroring
        # CascadingInference._call_llm: newer ollama clients return an object
        # exposing `.message.content`, older ones a dict. The previous dict-only
        # `response.get(...)` raised AttributeError on the object form, which
        # `_query_model` swallowed -> every consensus model query failed.
        content = ""
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "") or ""
        elif hasattr(response, "message"):
            msg = response.message
            if hasattr(msg, "content"):
                content = msg.content or ""
            elif isinstance(msg, dict):
                content = msg.get("content", "") or ""
        return content.strip()

    def _get_model_quality_tier(self, model_name: str) -> str:
        """Retrieve the quality tier of a model via profiles.

        Args:
            model_name: Model name

        Returns:
            Quality tier: "high", "medium", or "low"
        """
        if not PROFILES_AVAILABLE or profile_manager is None:
            return "medium"
        try:
            profile = profile_manager.get_profile(model_name)
            if profile is not None:
                return getattr(profile, "quality_tier", "medium")
        except Exception:
            pass
        return "medium"

    def _query_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> ModelResponse:
        """Query a model and return its response.

        Args:
            model: Model name
            messages: Conversation messages
            temperature: Temperature

        Returns:
            ModelResponse with the result
        """
        start = time.time()
        try:
            content = self._call_llm(
                messages=messages,
                model=model,
                temperature=temperature,
            )
            duration = int((time.time() - start) * 1000)
            quality_tier = self._get_model_quality_tier(model)
            return ModelResponse(
                model=model,
                content=content,
                duration_ms=duration,
                success=True,
                quality_tier=quality_tier,
            )
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            logger.warning(f"Failed model {model}: {e}")
            return ModelResponse(
                model=model,
                content="",
                duration_ms=duration,
                success=False,
                error=str(e),
            )

    def query_models_parallel(
        self,
        messages: list[dict[str, str]],
        models: list[str] | None = None,
        temperature: float | None = None,
        on_model_done: Callable[[ModelResponse], None] | None = None,
    ) -> list[ModelResponse]:
        """Query several models in parallel via threading.

        Args:
            messages: Conversation messages
            models: Liste de models (default: config.default_models)
            temperature: Temperature (default: config.temperature)
            on_model_done: Callback called when a model finishes

        Returns:
            List of each model's responses
        """
        _models = (models or self._config.default_models)[:self._config.max_models]
        _temp = temperature if temperature is not None else self._config.temperature

        if not _models:
            return []

        responses: list[ModelResponse] = []

        with ThreadPoolExecutor(max_workers=len(_models)) as pool:
            futures = {
                pool.submit(self._query_model, m, messages, _temp): m
                for m in _models
            }

            for future in as_completed(futures, timeout=self._config.timeout_per_model + 10):
                model_name = futures[future]
                try:
                    resp = future.result(timeout=self._config.timeout_per_model)
                    responses.append(resp)
                    if on_model_done:
                        try:
                            on_model_done(resp)
                        except Exception:
                            pass
                except TimeoutError:
                    logger.warning(f"Timeout for model {model_name}")
                    responses.append(ModelResponse(
                        model=model_name,
                        content="",
                        success=False,
                        error="timeout",
                    ))
                except Exception as e:
                    logger.warning(f"Future error {model_name}: {e}")
                    responses.append(ModelResponse(
                        model=model_name,
                        content="",
                        success=False,
                        error=str(e),
                    ))

        return responses

    # ----------------------------------------------------------------
    # Comparison of the responses
    # ----------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set:
        """Tokenize a text into a set of normalized words.

        Filter short words (<=3 characters) to reduce noise.

        Args:
            text: Raw text

        Returns:
            Ensemble de mots significatifs
        """
        words = set(text.lower().split())
        return {w for w in words if len(w) > 3}

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Compute the Jaccard similarity between two sets.

        Args:
            set_a: Premier ensemble
            set_b: Second ensemble

        Returns:
            Jaccard score between 0.0 and 1.0
        """
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def compare_responses(
        self,
        responses: list[ModelResponse],
    ) -> ConsensusComparison:
        """Compare all model responses against each other.

        Compute a Jaccard similarity matrix and identify
        the areas of agreement and disagreement.

        Args:
            responses: List of the models' responses

        Returns:
            ConsensusComparison with the matrix and the scores
        """
        valid = [r for r in responses if r.success and r.content]
        if len(valid) < 2:
            return ConsensusComparison(
                average_agreement=1.0 if len(valid) == 1 else 0.0,
            )

        # Tokenize each response
        word_sets = {r.model: self._tokenize(r.content) for r in valid}

        # Similarity matrix
        matrix: dict[str, dict[str, float]] = {}
        all_scores = []

        for r_i in valid:
            matrix[r_i.model] = {}
            for r_j in valid:
                if r_i.model == r_j.model:
                    matrix[r_i.model][r_j.model] = 1.0
                    continue
                score = self._jaccard_similarity(
                    word_sets[r_i.model],
                    word_sets[r_j.model],
                )
                matrix[r_i.model][r_j.model] = round(score, 3)
                all_scores.append(score)

        avg_agreement = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Identify agreement areas: words present in all responses
        if word_sets:
            all_words = list(word_sets.values())
            common_words = all_words[0]
            for ws in all_words[1:]:
                common_words = common_words & ws
            areas_agreement = sorted(common_words)[:20]  # Top 20 mots partages

            # Disagreement areas: words present in a single response
            unique_per_model = {}
            for model_name, ws in word_sets.items():
                others = set()
                for other_name, other_ws in word_sets.items():
                    if other_name != model_name:
                        others |= other_ws
                unique = ws - others
                if unique:
                    unique_per_model[model_name] = sorted(unique)[:5]

            areas_disagreement = []
            for model_name, uniq in unique_per_model.items():
                areas_disagreement.append(f"{model_name}: {', '.join(uniq)}")
        else:
            areas_agreement = []
            areas_disagreement = []

        return ConsensusComparison(
            agreement_matrix=matrix,
            average_agreement=round(avg_agreement, 3),
            areas_of_agreement=areas_agreement,
            areas_of_disagreement=areas_disagreement,
        )

    # ----------------------------------------------------------------
    # Strategies de consensus
    # ----------------------------------------------------------------

    def _best_of_n(
        self,
        responses: list[ModelResponse],
        comparison: ConsensusComparison,
    ) -> tuple[str, str, float]:
        """Best-of-N strategy: select the most central response.

        The response with the highest average similarity to the others
        is chosen as the most representative of the consensus.

        Args:
            responses: Responses of the models
            comparison: Result of the comparison

        Returns:
            Tuple (content, model, confidence)
        """
        valid = [r for r in responses if r.success and r.content]
        if not valid:
            return "", "", 0.0
        if len(valid) == 1:
            return valid[0].content, valid[0].model, 0.5

        # Compute each model's average score in the matrix
        model_scores = {}
        for r in valid:
            if r.model in comparison.agreement_matrix:
                scores = [
                    v for k, v in comparison.agreement_matrix[r.model].items()
                    if k != r.model
                ]
                model_scores[r.model] = sum(scores) / len(scores) if scores else 0.0
            else:
                model_scores[r.model] = 0.0

        # Select the model with the best score
        best_model = max(model_scores, key=model_scores.get)
        best_response = next(r for r in valid if r.model == best_model)
        confidence = model_scores[best_model]

        return best_response.content, best_model, round(confidence, 3)

    def _weighted_vote(
        self,
        responses: list[ModelResponse],
        comparison: ConsensusComparison,
    ) -> tuple[str, str, float]:
        """Weighted Vote strategy: weighted by the quality tier.

        Combine the similarity with the quality-tier weights
        to select the best response.

        Args:
            responses: Responses of the models
            comparison: Result of the comparison

        Returns:
            Tuple (content, model, confidence)
        """
        valid = [r for r in responses if r.success and r.content]
        if not valid:
            return "", "", 0.0
        if len(valid) == 1:
            return valid[0].content, valid[0].model, 0.5

        weights = self._config.quality_weights

        # Compute each model's weighted score
        model_scores = {}
        for r in valid:
            quality_weight = weights.get(r.quality_tier, 0.7)

            # Average agreement score in the matrix
            if r.model in comparison.agreement_matrix:
                agreement_scores = [
                    v for k, v in comparison.agreement_matrix[r.model].items()
                    if k != r.model
                ]
                avg_agreement = (
                    sum(agreement_scores) / len(agreement_scores)
                    if agreement_scores else 0.0
                )
            else:
                avg_agreement = 0.0

            # Final score: agreement * quality weight
            model_scores[r.model] = avg_agreement * quality_weight

        best_model = max(model_scores, key=model_scores.get)
        best_response = next(r for r in valid if r.model == best_model)
        confidence = model_scores[best_model]

        return best_response.content, best_model, round(confidence, 3)

    def _llm_merge(
        self,
        responses: list[ModelResponse],
        original_query: str,
    ) -> tuple[str, str, float]:
        """LLM Merge strategy: a judge model merges the responses.

        Send all responses to a judge model that synthesizes
        the best parts of each.

        Args:
            responses: Responses of the models
            original_query: The user's original question

        Returns:
            Tuple (merged content, judge model, confidence)
        """
        valid = [r for r in responses if r.success and r.content]
        if not valid:
            return "", "", 0.0
        if len(valid) == 1:
            return valid[0].content, valid[0].model, 0.5

        judge = self._config.judge_model or self._default_model

        # Build the synthesis prompt
        responses_text = ""
        for i, r in enumerate(valid, 1):
            responses_text += f"\n--- Response {i} (from {r.model}) ---\n{r.content}\n"

        merge_prompt = (
            f"Multiple AI models were asked the following question:\n"
            f"\"{original_query}\"\n\n"
            f"Here are their responses:{responses_text}\n"
            f"---\n\n"
            f"Synthesize the best parts of all responses into a single, "
            f"comprehensive, and accurate answer. Keep the strongest arguments "
            f"and facts from each response. If responses contradict each other, "
            f"note the disagreement. Respond directly with the merged answer."
        )

        try:
            merged = self._call_llm(
                messages=[{"role": "user", "content": merge_prompt}],
                model=judge,
                temperature=self._config.temperature,
            )
            # Confidence based on the number of merged responses
            confidence = min(0.9, 0.5 + 0.1 * len(valid))
            return merged, f"[merge:{judge}]", round(confidence, 3)
        except Exception as e:
            logger.warning(f"Failed LLM merge with {judge}: {e}")
            # Fallback: return the first valid response
            return valid[0].content, valid[0].model, 0.3

    # ----------------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------------

    def run_consensus(
        self,
        query: str,
        models: list[str] | None = None,
        strategy: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        on_model_done: Callable[[ModelResponse], None] | None = None,
    ) -> ConsensusResult:
        """Execute the full multi-model consensus.

        1. Queries N models in parallel
        2. Compares the responses
        3. Applies the consensus strategy

        Args:
            query: User question
            models: Liste de models (default: config.default_models)
            strategy: Consensus strategy (default: config.strategy)
            system_prompt: Optional system prompt
            temperature: Temperature (default: config.temperature)
            on_model_done: Callback when a model finishes

        Returns:
            ConsensusResult complet
        """
        start_time = time.time()
        _strategy = strategy or self._config.strategy
        if _strategy not in VALID_STRATEGIES:
            _strategy = STRATEGY_BEST_OF_N

        # Prepare the messages
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        # 1) Query models in parallel
        responses = self.query_models_parallel(
            messages=messages,
            models=models,
            temperature=temperature,
            on_model_done=on_model_done,
        )

        # 2) Compare the responses
        comparison = self.compare_responses(responses)

        # 3) Apply the strategy
        if _strategy == STRATEGY_WEIGHTED_VOTE:
            content, selected_model, confidence = self._weighted_vote(
                responses, comparison,
            )
        elif _strategy == STRATEGY_LLM_MERGE:
            content, selected_model, confidence = self._llm_merge(
                responses, query,
            )
        else:
            content, selected_model, confidence = self._best_of_n(
                responses, comparison,
            )

        total_duration = int((time.time() - start_time) * 1000)

        result = ConsensusResult(
            strategy=_strategy,
            selected_response=content,
            selected_model=selected_model,
            confidence=confidence,
            individual_responses=responses,
            comparison=comparison,
            total_duration_ms=total_duration,
            metadata={
                "models_queried": [r.model for r in responses],
                "models_succeeded": [r.model for r in responses if r.success],
                "models_failed": [r.model for r in responses if not r.success],
                "average_agreement": comparison.average_agreement,
                "strategy_used": _strategy,
            },
        )

        self._last_result = result
        return result

    # ----------------------------------------------------------------
    # Execution with streaming (for agentic_executor integration)
    # ----------------------------------------------------------------

    def execute_consensus(
        self,
        query: str,
        models: list[str] | None = None,
        strategy: str | None = None,
        system_prompt: str | None = None,
        on_model_done: Callable[[ModelResponse], None] | None = None,
    ) -> Generator:
        """Execute consensus and yield results for streaming.

        Yields tuples for streaming:
        - ("consensus_model_done", ModelResponse) for each model
        - ("consensus_done", ConsensusResult) a la fin
        - str for the final response tokens

        Args:
            query: User question
            models: Liste de models
            strategy: Consensus strategy
            system_prompt: Optional system prompt
            on_model_done: Callback additionnel

        Yields:
            Chunks de streaming
        """
        model_responses_streamed: list[ModelResponse] = []

        def _model_callback(resp: ModelResponse):
            model_responses_streamed.append(resp)
            if on_model_done:
                on_model_done(resp)

        result = self.run_consensus(
            query=query,
            models=models,
            strategy=strategy,
            system_prompt=system_prompt,
            on_model_done=_model_callback,
        )

        # Yield each model response
        for resp in result.individual_responses:
            yield ("consensus_model_done", resp)

        # Yield the final result
        yield ("consensus_done", result)

        # Yield the selected response as tokens
        if result.selected_response:
            yield result.selected_response

    # ----------------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------------

    @staticmethod
    def result_to_dict(result: ConsensusResult) -> dict[str, Any]:
        """Convert a ConsensusResult to a serializable dictionary.

        Args:
            result: Consensus result

        Returns:
            Dictionnaire JSON-serialisable
        """
        individual = []
        for r in result.individual_responses:
            individual.append({
                "model": r.model,
                "content": r.content,
                "duration_ms": r.duration_ms,
                "success": r.success,
                "error": r.error,
                "quality_tier": r.quality_tier,
            })

        comparison_dict = None
        if result.comparison:
            comparison_dict = {
                "agreement_matrix": result.comparison.agreement_matrix,
                "average_agreement": result.comparison.average_agreement,
                "areas_of_agreement": result.comparison.areas_of_agreement,
                "areas_of_disagreement": result.comparison.areas_of_disagreement,
            }

        return {
            "strategy": result.strategy,
            "selected_response": result.selected_response,
            "selected_model": result.selected_model,
            "confidence": result.confidence,
            "individual_responses": individual,
            "comparison": comparison_dict,
            "total_duration_ms": result.total_duration_ms,
            "metadata": result.metadata,
        }


# =============================================================================
# SINGLETON
# =============================================================================

# Singleton instance, loaded from the YAML config
consensus_engine = ConsensusEngine()
