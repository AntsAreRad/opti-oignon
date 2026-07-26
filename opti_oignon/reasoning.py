#!/usr/bin/env python3
"""
REASONING ENGINE - OPTI-OIGNON v1.5.3
=============================================

Moteur de raisonnement avance offrant trois strategies:

1. Decompose-and-Solve:
   Decompose une request complexe en sous-etapes, les resout
   sequentially, then aggregates results into a final response.

2. Tree-of-Thought (simplifie):
   Generates N candidate approaches, evaluates each via a scoring
   prompt, then selects and develops the best one.

3. Self-Consistency:
   Execute la meme request N fois avec des temperatures variees,
   compares responses for consistency, and returns the most
   representative avec un score de confiance.

Author: Leon
"""

import json
import logging
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import conditionnel de la config YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import conditionnel d'Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


def _reply_text(response: Any) -> str:
    """Pull the assistant text from an ollama-style chat response.

    Handles the dict shape and the object shape returned by newer
    ollama-python (a ChatResponse is not subscriptable and has no .get), so
    an object-form response no longer raises an AttributeError swallowed
    into the strategy fallbacks. Mirrors memory/legacy._reply_text
    (the dict-vs-object class).
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(response.get("content") or "")
    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "content", "") or "")
    return ""


# =============================================================================
# DATACLASSES DE RESULTAT
# =============================================================================

@dataclass
class ReasoningStep:
    """Etape individuelle dans un processus de raisonnement."""
    step_number: int
    title: str
    content: str
    duration_ms: int = 0


@dataclass
class ReasoningResult:
    """Result complet of a raisonnement multi-etapes."""
    strategy: str  # "decompose", "tree_of_thought", "self_consistency"
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    total_duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeBranch:
    """Branche dans l'arbre de pensee (Tree-of-Thought)."""
    branch_id: int
    approach: str
    score: float = 0.0
    elaboration: str = ""


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ReasoningConfig:
    """Configuration du moteur de raisonnement."""
    max_sub_steps: int = 5
    tree_branches: int = 3
    self_consistency_runs: int = 3
    temperature_variance: float = 0.1
    base_temperature: float = 0.3
    min_complexity_for_reasoning: str = "complex"
    decompose_model: str | None = None
    evaluate_model: str | None = None
    timeout_per_step: int = 60
    # Strategy used by the live pipeline when the caller
    # does not force one ("decompose", "tree_of_thought", "self_consistency").
    default_strategy: str = "decompose"

    @classmethod
    def from_yaml(cls, path: str) -> "ReasoningConfig":
        """Load configuration from a YAML file."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, using default config")
            return cls()

        try:
            config_path = Path(path)
            if not config_path.exists():
                logger.debug(f"Fichier config absent: {path}")
                return cls()

            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            reasoning_data = data.get("reasoning", data)
            return cls(
                max_sub_steps=reasoning_data.get("max_sub_steps", 5),
                tree_branches=reasoning_data.get("tree_branches", 3),
                self_consistency_runs=reasoning_data.get("self_consistency_runs", 3),
                temperature_variance=reasoning_data.get("temperature_variance", 0.1),
                base_temperature=reasoning_data.get("base_temperature", 0.3),
                min_complexity_for_reasoning=reasoning_data.get(
                    "min_complexity_for_reasoning", "complex"
                ),
                decompose_model=reasoning_data.get("decompose_model"),
                evaluate_model=reasoning_data.get("evaluate_model"),
                timeout_per_step=reasoning_data.get("timeout_per_step", 60),
                default_strategy=reasoning_data.get(
                    "default_strategy", "decompose"
                ),
            )
        except Exception as e:
            logger.warning(f"Erreur loading config reasoning: {e}")
            return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningConfig":
        """Create a configuration from a dictionary."""
        reasoning_data = data.get("reasoning", data)
        return cls(
            max_sub_steps=reasoning_data.get("max_sub_steps", 5),
            tree_branches=reasoning_data.get("tree_branches", 3),
            self_consistency_runs=reasoning_data.get("self_consistency_runs", 3),
            temperature_variance=reasoning_data.get("temperature_variance", 0.1),
            base_temperature=reasoning_data.get("base_temperature", 0.3),
            min_complexity_for_reasoning=reasoning_data.get(
                "min_complexity_for_reasoning", "complex"
            ),
            decompose_model=reasoning_data.get("decompose_model"),
            evaluate_model=reasoning_data.get("evaluate_model"),
            timeout_per_step=reasoning_data.get("timeout_per_step", 60),
            default_strategy=reasoning_data.get(
                "default_strategy", "decompose"
            ),
        )


# =============================================================================
# PROMPTS SYSTEME
# =============================================================================

_DECOMPOSE_PROMPT = """You are a reasoning assistant. Your task is to decompose a complex question into clear sub-steps.

INSTRUCTIONS:
1. Analyze the user's question carefully.
2. Break it down into {max_steps} or fewer logical sub-steps.
3. Each sub-step should be a focused, answerable question or task.
4. Respond ONLY with a JSON array of objects, each with "title" and "question" keys.

Example output:
[
  {{"title": "Define the problem scope", "question": "What exactly is being asked?"}},
  {{"title": "Identify key factors", "question": "What are the main variables involved?"}},
  {{"title": "Analyze relationships", "question": "How do these factors interact?"}}
]

Do NOT include any text outside the JSON array."""

_SOLVE_STEP_PROMPT = """You are answering a sub-question as part of a larger reasoning process.

ORIGINAL QUESTION: {original_question}

PREVIOUS STEPS AND ANSWERS:
{previous_context}

CURRENT SUB-STEP: {step_title}
SUB-QUESTION: {step_question}

Provide a clear, focused answer to this sub-question. Be concise but thorough."""

_SYNTHESIZE_PROMPT = """You are synthesizing a final answer from a step-by-step reasoning process.

ORIGINAL QUESTION: {original_question}

REASONING STEPS AND ANSWERS:
{steps_context}

Provide a comprehensive final answer that integrates all the reasoning steps above. Be clear and well-structured."""

_TREE_GENERATE_PROMPT = """You are exploring different approaches to solve a problem.

QUESTION: {question}

Generate {n_branches} distinct approaches to answer this question. Each approach should take a different angle or methodology.

Respond ONLY with a JSON array of objects, each with "approach" key describing the strategy.

Example:
[
  {{"approach": "Analyze from a historical perspective..."}},
  {{"approach": "Use a quantitative comparison..."}},
  {{"approach": "Consider the ethical implications..."}}
]

Do NOT include any text outside the JSON array."""

_TREE_EVALUATE_PROMPT = """You are evaluating an approach to answering a question.

QUESTION: {question}

APPROACH: {approach}

Rate this approach on a scale of 0.0 to 1.0 based on:
- Relevance to the question
- Feasibility of producing a good answer
- Depth and thoroughness potential

Respond with ONLY a JSON object: {{"score": 0.X, "justification": "brief reason"}}"""

_TREE_ELABORATE_PROMPT = """You are developing the best approach to answer a question.

QUESTION: {question}

SELECTED APPROACH: {approach}
(This approach was selected as the most promising among {n_branches} candidates.)

Now fully develop this approach into a comprehensive answer. Be thorough and well-structured."""


# =============================================================================
# MOTEUR DE RAISONNEMENT
# =============================================================================

class ReasoningEngine:
    """Moteur de raisonnement multi-strategies.

    Provides three reasoning strategies for complex queries:
    - decompose: Break down and solve step-by-step
    - tree_of_thought: Explore multiple approaches
    - self_consistency: Multiple runs for consistency check
    """

    def __init__(
        self,
        config: ReasoningConfig | None = None,
        default_model: str = "qwen3:32b",
    ):
        """Initialize the reasoning engine.

        Args:
            config: Reasoning configuration (or None for defaults)
            default_model: Default Ollama model
        """
        self._config = config or ReasoningConfig()
        self._default_model = default_model
        self._last_result: ReasoningResult | None = None
        # Per-timeout ollama clients so timeout_per_step is
        # actually enforced on every call.
        self._clients: dict[int, Any] = {}

    # ----------------------------------------------------------------
    # Proprietes
    # ----------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Indique si le moteur de raisonnement est operationnel."""
        return OLLAMA_AVAILABLE

    @property
    def config(self) -> ReasoningConfig:
        """Configuration actuelle."""
        return self._config

    @property
    def last_result(self) -> ReasoningResult | None:
        """Dernier result de raisonnement."""
        return self._last_result

    # ----------------------------------------------------------------
    # Utilitaire LLM
    # ----------------------------------------------------------------

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        timeout: int | None = None,
    ) -> str:
        """Appel synchrone au LLM via Ollama.

        Args:
            messages: Messages au format Ollama
            model: Model a utiliser
            temperature: Temperature de generation
            timeout: Timeout en secondes

        Returns:
            Texte de la reponse
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama non disponible")

        _model = model or self._default_model
        _timeout = timeout or self._config.timeout_per_step

        try:
            # Route through a timeout-bound client so
            # timeout_per_step is enforced (it was computed but never used;
            # a hung model call blocked the executor pipeline indefinitely).
            client = self._get_client(_timeout)
            response = client.chat(
                model=_model,
                messages=messages,
                options={"temperature": temperature},
            )
            # Both-form parse (dict / object ollama-python).
            content = _reply_text(response)
            return content.strip()
        except Exception as e:
            logger.error(f"Erreur appel LLM ({_model}): {e}")
            raise

    def _get_client(self, timeout: int) -> Any:
        """Return a cached ollama client bound to the given timeout (RSN-02)."""
        client = self._clients.get(timeout)
        if client is None:
            client = ollama.Client(timeout=timeout)
            self._clients[timeout] = client
        return client

    def _parse_json_response(self, text: str) -> Any:
        """Parse une reponse JSON du LLM, avec nettoyage.

        Gere les cas ou le LLM enrobe le JSON dans des blocs markdown.
        """
        cleaned = text.strip()

        # Retirer les blocs markdown ```json ... ```
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Retirer la premiere et derniere ligne si ce sont des fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Retirer un eventuel prefixe texte avant le JSON
        # Chercher le debut du JSON ([ ou {)
        for i, ch in enumerate(cleaned):
            if ch in ("[", "{"):
                cleaned = cleaned[i:]
                break

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fall back to the in-house tolerant repair before
        # giving up (trailing prose, truncation, single quotes, comments).
        # A parse failure here silently degrades the strategy to a
        # single-step / single-branch run, so repairing directly raises the
        # strategy success rate on local models.
        try:
            from .json_repair import repair_json
            return repair_json(cleaned)
        except Exception as e:
            logger.warning(f"Erreur parsing JSON LLM (after repair): {e}")
            logger.debug(f"Texte brut: {text[:500]}")
            return None

    # ----------------------------------------------------------------
    # Strategie 1: Decompose-and-Solve
    # ----------------------------------------------------------------

    def decompose_and_solve(
        self,
        question: str,
        model: str | None = None,
        max_steps: int | None = None,
        on_step: Callable | None = None,
    ) -> ReasoningResult:
        """Decompose a request into sub-steps and solve them sequentially.

        Args:
            question: Question complexe de l'utilisateur
            model: Model a utiliser (default: config ou default_model)
            max_steps: Nombre max de sous-etapes (default: config.max_sub_steps)
            on_step: Callback appele apres each etape resolue
                Signature: on_step(step: ReasoningStep)

        Returns:
            ReasoningResult avec les etapes et la reponse finale
        """
        start_time = time.time()
        _model = model or self._config.decompose_model or self._default_model
        _max_steps = max_steps or self._config.max_sub_steps

        result = ReasoningResult(strategy="decompose")

        # Phase 1: Decomposition de la question
        decompose_prompt = _DECOMPOSE_PROMPT.format(max_steps=_max_steps)
        try:
            decompose_response = self._call_llm(
                messages=[
                    {"role": "system", "content": decompose_prompt},
                    {"role": "user", "content": question},
                ],
                model=_model,
                temperature=0.1,
            )

            sub_steps = self._parse_json_response(decompose_response)
            if not isinstance(sub_steps, list) or not sub_steps:
                # Fallback: traiter comme une seule etape
                sub_steps = [{"title": "Direct answer", "question": question}]

            # Limiter au max
            sub_steps = sub_steps[:_max_steps]

        except Exception as e:
            logger.error(f"Decomposition echouee: {e}")
            # Fallback: reponse directe
            sub_steps = [{"title": "Direct answer", "question": question}]

        # Phase 2: Resoudre each sous-etape
        previous_context = ""
        for i, step_data in enumerate(sub_steps):
            step_start = time.time()
            step_title = step_data.get("title", f"Step {i + 1}")
            step_question = step_data.get("question", question)

            solve_prompt = _SOLVE_STEP_PROMPT.format(
                original_question=question,
                previous_context=previous_context or "(no previous steps)",
                step_title=step_title,
                step_question=step_question,
            )

            try:
                step_answer = self._call_llm(
                    messages=[
                        {"role": "system", "content": solve_prompt},
                        {"role": "user", "content": step_question},
                    ],
                    model=_model,
                    temperature=self._config.base_temperature,
                )
            except Exception as e:
                step_answer = f"[Error solving step: {e}]"

            step_duration = int((time.time() - step_start) * 1000)

            step = ReasoningStep(
                step_number=i + 1,
                title=step_title,
                content=step_answer,
                duration_ms=step_duration,
            )
            result.steps.append(step)

            # Accumuler le contexte pour les etapes suivantes
            previous_context += f"\nStep {i + 1} ({step_title}): {step_answer}\n"

            # Callback
            if on_step is not None:
                try:
                    on_step(step)
                except Exception as e:
                    logger.debug(f"Callback on_step echoue: {e}")

        # Phase 3: Synthese finale
        steps_context = "\n".join(
            f"Step {s.step_number} ({s.title}): {s.content}"
            for s in result.steps
        )
        synthesize_prompt = _SYNTHESIZE_PROMPT.format(
            original_question=question,
            steps_context=steps_context,
        )

        try:
            final_answer = self._call_llm(
                messages=[
                    {"role": "system", "content": synthesize_prompt},
                    {"role": "user", "content": question},
                ],
                model=_model,
                temperature=self._config.base_temperature,
            )
            result.final_answer = final_answer
        except Exception:
            # Fallback: concatener les reponses des etapes
            result.final_answer = "\n\n".join(
                f"**{s.title}**: {s.content}" for s in result.steps
            )

        result.confidence = min(1.0, len(result.steps) / max(1, _max_steps))
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        result.metadata = {
            "model": _model,
            "sub_steps_requested": _max_steps,
            "sub_steps_generated": len(result.steps),
        }

        self._last_result = result
        return result

    # ----------------------------------------------------------------
    # Strategie 2: Tree-of-Thought
    # ----------------------------------------------------------------

    def tree_of_thought(
        self,
        question: str,
        model: str | None = None,
        n_branches: int | None = None,
        on_step: Callable | None = None,
    ) -> ReasoningResult:
        """Explore N candidate approaches and select the best.

        Args:
            question: Question a resoudre
            model: Model a utiliser
            n_branches: Nombre de branches a explorer (default: config.tree_branches)
            on_step: Callback apres each etape

        Returns:
            ReasoningResult avec les branches et la reponse elaboree
        """
        start_time = time.time()
        _model = model or self._config.evaluate_model or self._default_model
        _n_branches = n_branches or self._config.tree_branches

        result = ReasoningResult(strategy="tree_of_thought")

        # Phase 1: Generer les approches
        gen_prompt = _TREE_GENERATE_PROMPT.format(
            question=question,
            n_branches=_n_branches,
        )

        try:
            gen_response = self._call_llm(
                messages=[
                    {"role": "system", "content": gen_prompt},
                    {"role": "user", "content": question},
                ],
                model=_model,
                temperature=0.7,
            )

            branches_data = self._parse_json_response(gen_response)
            if not isinstance(branches_data, list) or not branches_data:
                branches_data = [{"approach": f"Direct analysis of: {question}"}]

            branches_data = branches_data[:_n_branches]

        except Exception as e:
            logger.error(f"Generation de branches echouee: {e}")
            branches_data = [{"approach": f"Direct analysis of: {question}"}]

        # Enregistrer l'etape de generation
        gen_step = ReasoningStep(
            step_number=1,
            title="Generate approaches",
            content=f"Generated {len(branches_data)} candidate approaches.",
            duration_ms=int((time.time() - start_time) * 1000),
        )
        result.steps.append(gen_step)
        if on_step:
            try:
                on_step(gen_step)
            except Exception:
                pass

        # Phase 2: Evaluer each branche
        branches: list[TreeBranch] = []
        for i, bd in enumerate(branches_data):
            eval_start = time.time()  # noqa: F841
            approach = bd.get("approach", f"Approach {i + 1}")

            eval_prompt = _TREE_EVALUATE_PROMPT.format(
                question=question,
                approach=approach,
            )

            score = 0.5  # Default score
            try:
                eval_response = self._call_llm(
                    messages=[
                        {"role": "system", "content": eval_prompt},
                        {"role": "user", "content": approach},
                    ],
                    model=_model,
                    temperature=0.1,
                )

                eval_data = self._parse_json_response(eval_response)
                if isinstance(eval_data, dict):
                    score = float(eval_data.get("score", 0.5))
                    score = max(0.0, min(1.0, score))  # Clamper

            except Exception as e:
                logger.warning(f"Evaluation branche {i} echouee: {e}")

            branch = TreeBranch(
                branch_id=i,
                approach=approach,
                score=score,
            )
            branches.append(branch)

        # Enregistrer l'etape d'evaluation
        eval_step = ReasoningStep(
            step_number=2,
            title="Evaluate approaches",
            content="; ".join(
                f"Branch {b.branch_id}: score={b.score:.2f}"
                for b in branches
            ),
            duration_ms=int((time.time() - start_time) * 1000) - gen_step.duration_ms,
        )
        result.steps.append(eval_step)
        if on_step:
            try:
                on_step(eval_step)
            except Exception:
                pass

        # Phase 3: Selectionner la meilleure branche et l'elaborer
        best_branch = max(branches, key=lambda b: b.score) if branches else None

        if best_branch is not None:
            elab_prompt = _TREE_ELABORATE_PROMPT.format(
                question=question,
                approach=best_branch.approach,
                n_branches=_n_branches,
            )

            elab_start = time.time()
            try:
                elaboration = self._call_llm(
                    messages=[
                        {"role": "system", "content": elab_prompt},
                        {"role": "user", "content": question},
                    ],
                    model=_model,
                    temperature=self._config.base_temperature,
                )
                best_branch.elaboration = elaboration
                result.final_answer = elaboration
            except Exception as e:
                logger.error(f"Elaboration echouee: {e}")
                result.final_answer = best_branch.approach

            elab_step = ReasoningStep(
                step_number=3,
                title=f"Elaborate best approach (score: {best_branch.score:.2f})",
                content=best_branch.elaboration or best_branch.approach,
                duration_ms=int((time.time() - elab_start) * 1000),
            )
            result.steps.append(elab_step)
            if on_step:
                try:
                    on_step(elab_step)
                except Exception:
                    pass
        else:
            result.final_answer = "[No valid approaches generated]"

        result.confidence = best_branch.score if best_branch else 0.0
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        result.metadata = {
            "model": _model,
            "branches_requested": _n_branches,
            "branches_generated": len(branches),
            "best_branch_id": best_branch.branch_id if best_branch else -1,
            "best_branch_score": best_branch.score if best_branch else 0.0,
            "branches": [
                {"id": b.branch_id, "approach": b.approach[:100], "score": b.score}
                for b in branches
            ],
        }

        self._last_result = result
        return result

    # ----------------------------------------------------------------
    # Strategie 3: Self-Consistency
    # ----------------------------------------------------------------

    def self_consistency(
        self,
        question: str,
        model: str | None = None,
        n_runs: int | None = None,
        on_step: Callable | None = None,
    ) -> ReasoningResult:
        """Execute the same request N times and measure coherence.

        Args:
            question: Question a poser plusieurs fois
            model: Model a utiliser
            n_runs: Nombre d'executions (default: config.self_consistency_runs)
            on_step: Callback apres each run

        Returns:
            ReasoningResult avec la reponse la plus representative
        """
        start_time = time.time()
        _model = model or self._default_model
        _n_runs = n_runs or self._config.self_consistency_runs

        result = ReasoningResult(strategy="self_consistency")
        answers: list[str] = []

        # Generer N reponses avec des temperatures legerement variees
        base_temp = self._config.base_temperature
        variance = self._config.temperature_variance

        for i in range(_n_runs):
            run_start = time.time()
            # Temperature variee autour de la base
            temp = base_temp + (i - _n_runs // 2) * variance
            temp = max(0.0, min(1.5, temp))  # Clamper

            answer = ""
            try:
                answer = self._call_llm(
                    messages=[
                        {"role": "user", "content": question},
                    ],
                    model=_model,
                    temperature=temp,
                )
                answers.append(answer)
            except Exception as e:
                logger.warning(f"Run {i} echouee: {e}")
                answers.append("")

            run_duration = int((time.time() - run_start) * 1000)
            step = ReasoningStep(
                step_number=i + 1,
                title=f"Run {i + 1} (temp={temp:.2f})",
                content=answer if answer else "[failed]",
                duration_ms=run_duration,
            )
            result.steps.append(step)
            if on_step:
                try:
                    on_step(step)
                except Exception:
                    pass

        # Calculer la coherence entre les reponses
        valid_answers = [a for a in answers if a]
        if not valid_answers:
            result.final_answer = "[All runs failed]"
            result.confidence = 0.0
        elif len(valid_answers) == 1:
            result.final_answer = valid_answers[0]
            result.confidence = 0.5
        else:
            # Calculer la similarite par mots partages (heuristique simple)
            # et selectionner la reponse la plus "centrale"
            best_answer, agreement_score = self._select_most_consistent(
                valid_answers,
            )
            result.final_answer = best_answer
            result.confidence = agreement_score

        result.total_duration_ms = int((time.time() - start_time) * 1000)
        result.metadata = {
            "model": _model,
            "runs_requested": _n_runs,
            "runs_successful": len(valid_answers),
            "runs_failed": _n_runs - len(valid_answers),
            "agreement_score": result.confidence,
        }

        self._last_result = result
        return result

    def _select_most_consistent(
        self,
        answers: list[str],
    ) -> tuple[str, float]:
        """Select the most coherent response from N responses.

        Utilise une heuristique de similarite basee sur les mots partages.
        La reponse qui partage le plus de mots avec les autres est choisie.

        Returns:
            Tuple (meilleure reponse, score d'accord 0.0-1.0)
        """
        if not answers:
            return "", 0.0
        if len(answers) == 1:
            return answers[0], 1.0

        # Tokeniser each reponse en ensemble de mots (normalises)
        word_sets = []
        for answer in answers:
            words = set(answer.lower().split())
            # Retirer les mots tres courts (articles, etc.)
            words = {w for w in words if len(w) > 3}
            word_sets.append(words)

        # Calculer le score Jaccard moyen de each reponse par rapport aux autres
        scores = []
        for i, ws_i in enumerate(word_sets):
            if not ws_i:
                scores.append(0.0)
                continue
            pair_scores = []
            for j, ws_j in enumerate(word_sets):
                if i == j or not ws_j:
                    continue
                intersection = len(ws_i & ws_j)
                union = len(ws_i | ws_j)
                jaccard = intersection / union if union > 0 else 0.0
                pair_scores.append(jaccard)
            avg_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
            scores.append(avg_score)

        # Selectionner la reponse avec le meilleur score moyen
        best_idx = scores.index(max(scores)) if scores else 0
        agreement = max(scores) if scores else 0.0

        return answers[best_idx], round(agreement, 3)

    # ----------------------------------------------------------------
    # Execution avec streaming (pour integration agentic_executor)
    # ----------------------------------------------------------------

    def execute_reasoning(
        self,
        question: str,
        strategy: str | None = None,
        model: str | None = None,
        on_step: Callable | None = None,
    ) -> Generator:
        """Execute reasoning and yield results for streaming.

        Yields tuples for streaming:
        - ("reasoning_step", ReasoningStep) for each step
        - ("reasoning_done", ReasoningResult) at the end
        - str for the tokens of the final answer

        Args:
            question: Question to reason about
            strategy: "decompose", "tree_of_thought", "self_consistency",
                or None to use config.default_strategy
            model: Model to use
            on_step: Optional callback for each step
        """
        # Resolve the strategy from config when the caller
        # does not force one, so tree_of_thought / self_consistency are
        # reachable from the live path via reasoning.yaml (the executor
        # previously hardcoded "decompose").
        _strategy = strategy or self._config.default_strategy or "decompose"

        steps_so_far: list[ReasoningStep] = []

        def _step_callback(step: ReasoningStep):
            steps_so_far.append(step)
            if on_step:
                on_step(step)

        # Dispatcher vers la strategie
        if _strategy == "tree_of_thought":
            reasoning_result = self.tree_of_thought(
                question=question,
                model=model,
                on_step=_step_callback,
            )
        elif _strategy == "self_consistency":
            reasoning_result = self.self_consistency(
                question=question,
                model=model,
                on_step=_step_callback,
            )
        else:
            # Default: decompose_and_solve
            reasoning_result = self.decompose_and_solve(
                question=question,
                model=model,
                on_step=_step_callback,
            )

        # Yield les etapes individuelles
        for step in reasoning_result.steps:
            yield ("reasoning_step", step)

        # Yield la reponse finale
        yield ("reasoning_done", reasoning_result)

        # Yield la reponse texte pour le streaming normal
        if reasoning_result.final_answer:
            yield reasoning_result.final_answer

    # ----------------------------------------------------------------
    # Detection de complexite
    # ----------------------------------------------------------------

    @staticmethod
    def should_use_reasoning(message: str, complexity: str = "auto") -> bool:
        """Determine si une request necessiterait du raisonnement avance.

        Heuristique basee sur la longueur, les mots-cles, et la
        structure de la request.

        Args:
            message: Message de l'utilisateur
            complexity: "auto", "simple", "moderate", "complex"

        Returns:
            True si le raisonnement avance est recommande
        """
        if complexity == "complex":
            return True
        if complexity == "simple":
            return False

        msg_lower = message.lower()

        # Mots-cles indiquant un raisonnement complexe
        reasoning_keywords = [
            "step by step", "etape par etape",
            "analyze in detail", "analyse en detail",
            "compare and contrast", "compare et contraste",
            "what are the pros and cons", "avantages et inconvenients",
            "explain the reasoning", "explique le raisonnement",
            "break down", "decompose",
            "think through", "reflechis",
            "evaluate different", "evalue differentes",
            "consider all", "considere tous",
            "systematic", "systematique",
            "comprehensive analysis", "analyse complete",
            "multi-step", "multi-etape",
        ]

        keyword_match = any(kw in msg_lower for kw in reasoning_keywords)

        # Heuristique de longueur: questions longues et structurees
        word_count = len(message.split())
        is_long = word_count > 50

        # Presence de questions multiples (? multiples)
        question_marks = message.count("?")
        multi_question = question_marks >= 2

        return keyword_match or (is_long and multi_question)


# =============================================================================
# DEFAULT CONFIGURATION AND SINGLETON
# =============================================================================

# Chercher le fichier de config
_CONFIG_PATHS = [
    Path(__file__).parent / "config" / "reasoning.yaml",
    Path("reasoning.yaml"),
]

_default_config = ReasoningConfig()
for _path in _CONFIG_PATHS:
    if _path.exists():
        _default_config = ReasoningConfig.from_yaml(str(_path))
        break

reasoning_engine = ReasoningEngine(config=_default_config)
