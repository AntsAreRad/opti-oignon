#!/usr/bin/env python3
"""
PIPELINES - OPTI-OIGNON v1.5.7
==============================

Customisable execution pipelines built on the pipeline types of
the agentic executor. Lets users create sequences of steps
(THINK -> CODE_VERIFY -> SELF_CORRECT, etc.) and save them
as YAML.

Each step references an AgenticExecutor pipeline type
(direct, tools, think, code_verify, web_search, reasoning,
consensus, self_correct) with optional parameters and a
model override.

Author: Leon
"""

import copy
import logging
import re
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _resolve_emergency_stop():
    """Lazily resolve the emergency-stop module.

    Monkeypatchable proof seam, mirroring the emergency-stop pattern. Module-absent
    returns None and the pipeline runs unguarded: an availability control,
    not a security boundary (the documented posture).
    """
    try:
        from opti_oignon import emergency_stop
        return emergency_stop
    except Exception:
        return None


def _resolve_resource_governor():
    """Lazily resolve the resource governor.

    Same posture as the estop seam above: sys.modules is consulted first
    (so a test-seeded or standalone-loaded module is reused as-is),
    module-absent or unavailable returns None and the pipeline runs
    unguarded -- an availability control, fail-open by construction.
    """
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


# Repertoire de config des pipelines
_CONFIG_DIR = Path(__file__).parent / "config" / "pipelines"
# Repertoire de data pour les pipelines custom
_DATA_DIR = Path(__file__).parent / "data"

# Types de step disponibles (correspondent aux pipelines de l'agentic executor)
VALID_STEP_TYPES = [
    "direct",
    "tools",
    "think",
    "think_tools",
    "web_search",
    "code_verify",
    "reasoning",
    "consensus",
    "self_correct",
]

# Descriptions des types de step
STEP_TYPE_DESCRIPTIONS = {
    "direct": "Reponse directe sans traitement special",
    "tools": "Execution avec outils disponibles (ReAct loop)",
    "think": "Mode reflexion approfondie (chain-of-thought)",
    "think_tools": "Reflexion + outils combines",
    "web_search": "Search web + synthese",
    "code_verify": "Generation de code avec verification",
    "reasoning": "Raisonnement structure (Decompose/ToT/Self-Consistency)",
    "consensus": "Multi-model avec vote et fusion",
    "self_correct": "Auto-correction iterative",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExecutionStep:
    """
    A step in an execution pipeline.

    Attributes:
        step_type: Type de pipeline agentic (direct, tools, think, etc.)
        label: Nom d'affichage de l'etape
        model_override: Model specifique pour this etape (None = default)
        parameters: Parametres specifiques au step_type
        condition: Condition d'execution optionnelle (ex: "if_code_detected")
        pass_previous_output: Si True, injecte le result precedent dans le prompt
    """
    step_type: str
    label: str = ""
    model_override: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    condition: str | None = None
    pass_previous_output: bool = True

    def __post_init__(self):
        # Default label = capitalized type
        if not self.label:
            self.label = self.step_type.replace("_", " ").title()

    def validate(self) -> list[str]:
        """Validate the step. Return list of errors (empty if OK)."""
        errors = []
        if self.step_type not in VALID_STEP_TYPES:
            errors.append(
                f"Type '{self.step_type}' invalide. "
                f"Valides: {', '.join(VALID_STEP_TYPES)}"
            )
        if not self.label.strip():
            errors.append("Label requis")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialise en dictionnaire pour YAML."""
        d: dict[str, Any] = {
            "step_type": self.step_type,
            "label": self.label,
        }
        if self.model_override:
            d["model_override"] = self.model_override
        if self.parameters:
            d["parameters"] = self.parameters
        if self.condition:
            d["condition"] = self.condition
        if not self.pass_previous_output:
            d["pass_previous_output"] = False
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionStep":
        """Create from a dictionary."""
        return cls(
            step_type=data.get("step_type", "direct"),
            label=data.get("label", ""),
            model_override=data.get("model_override"),
            parameters=data.get("parameters", {}),
            condition=data.get("condition"),
            pass_previous_output=data.get("pass_previous_output", True),
        )


@dataclass
class ExecutionPipeline:
    """
    Pipeline d'execution compose de plusieurs etapes.

    Attributes:
        id: Identifiant unique
        name: Display name
        description: Pipeline description
        steps: Liste ordonnee des etapes
        created_at: Date de creation ISO
        updated_at: Date de derniere modification ISO
        is_builtin: True = lecture seule (defini dans config/pipelines/)
    """
    id: str
    name: str
    description: str = ""
    steps: list[ExecutionStep] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    is_builtin: bool = False

    def __post_init__(self):
        # Convert les dicts en ExecutionStep si required
        if self.steps and isinstance(self.steps[0], dict):
            self.steps = [ExecutionStep.from_dict(s) for s in self.steps]
        # Default dates
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def validate(self) -> list[str]:
        """Validate the complete pipeline. Return list of errors."""
        errors = []
        if not self.id:
            errors.append("ID requis")
        elif not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", self.id):
            errors.append(
                "ID invalide: lettres, chiffres, _ et - uniquement, "
                "doit commencer par une lettre"
            )
        if not self.name.strip():
            errors.append("Nom requis")
        if not self.steps:
            errors.append("Au moins une etape requise")
        for i, step in enumerate(self.steps):
            step_errors = step.validate()
            for e in step_errors:
                errors.append(f"Etape {i + 1}: {e}")
        return errors

    @property
    def step_count(self) -> int:
        """Nombre d'etapes."""
        return len(self.steps)

    @property
    def step_types_summary(self) -> str:
        """Resume des types d'etapes (ex: 'think -> code_verify')."""
        return " -> ".join(s.step_type for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Serialise en dictionnaire pour YAML."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_builtin": self.is_builtin,
        }

    @classmethod
    def from_dict(
        cls, pipeline_id: str, data: dict[str, Any], is_builtin: bool = False
    ) -> "ExecutionPipeline":
        """Create from a dictionary."""
        steps_data = data.get("steps", [])
        steps = [ExecutionStep.from_dict(s) for s in steps_data]
        return cls(
            id=pipeline_id,
            name=data.get("name", pipeline_id),
            description=data.get("description", ""),
            steps=steps,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            is_builtin=is_builtin,
        )


# =============================================================================
# PIPELINE STORE
# =============================================================================

class PipelineStore:
    """
    Manager de stockage des pipelines d'execution.

    Load builtin pipelines from config/pipelines/*.yaml
    et les pipelines custom depuis data/execution_pipelines.yaml.
    """

    def __init__(
        self,
        config_dir: Path | None = None,
        data_dir: Path | None = None,
    ):
        """Initialize the store."""
        self._config_dir = config_dir or _CONFIG_DIR
        self._data_dir = data_dir or _DATA_DIR
        self._custom_file = self._data_dir / "execution_pipelines.yaml"
        self._pipelines: dict[str, ExecutionPipeline] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all pipelines (builtin + custom)."""
        self._pipelines = {}
        self._load_builtin()
        self._load_custom()
        logger.info(
            f"PipelineStore: {len(self._pipelines)} execution pipelines loaded "
            f"({sum(1 for p in self._pipelines.values() if p.is_builtin)} builtin, "
            f"{sum(1 for p in self._pipelines.values() if not p.is_builtin)} custom)"
        )

    def _load_builtin(self) -> None:
        """Load builtin pipelines from config/pipelines/."""
        if not self._config_dir.exists():
            logger.debug(f"Repertoire config pipelines absent: {self._config_dir}")
            return
        for yaml_file in sorted(self._config_dir.glob("*.yaml")):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                # Each fichier peut contenir un ou plusieurs pipelines
                # Format: {pipeline_id: {name, description, steps, ...}}
                if "id" in data:
                    # Format simple: un seul pipeline par fichier
                    pid = data["id"]
                    self._pipelines[pid] = ExecutionPipeline.from_dict(
                        pid, data, is_builtin=True
                    )
                else:
                    # Format multi: {id1: {}, id2: {}}
                    for pid, pdata in data.items():
                        if isinstance(pdata, dict):
                            self._pipelines[pid] = ExecutionPipeline.from_dict(
                                pid, pdata, is_builtin=True
                            )
            except Exception as e:
                logger.error(f"Erreur loading {yaml_file}: {e}")

    def _load_custom(self) -> None:
        """Charge les pipelines custom depuis data/execution_pipelines.yaml."""
        if not self._custom_file.exists():
            return
        try:
            with open(self._custom_file, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for pid, pdata in data.items():
                if isinstance(pdata, dict):
                    # Les custom ne peuvent pas ecraser les builtin
                    if pid not in self._pipelines:
                        self._pipelines[pid] = ExecutionPipeline.from_dict(
                            pid, pdata, is_builtin=False
                        )
                    else:
                        logger.warning(
                            f"Pipeline custom '{pid}' ignore: "
                            f"ID already utilise par un builtin"
                        )
        except Exception as e:
            logger.error(f"Erreur loading custom pipelines: {e}")

    def _save_custom(self) -> None:
        """Save custom pipelines as YAML."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        custom = {
            pid: p.to_dict()
            for pid, p in self._pipelines.items()
            if not p.is_builtin
        }
        try:
            with open(self._custom_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    custom, f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
        except Exception as e:
            logger.error(f"Custom pipeline save error: {e}")

    # -- CRUD --

    def list_all(self) -> list[ExecutionPipeline]:
        """Liste tous les pipelines."""
        return list(self._pipelines.values())

    def list_builtin(self) -> list[ExecutionPipeline]:
        """Liste les pipelines builtin."""
        return [p for p in self._pipelines.values() if p.is_builtin]

    def list_custom(self) -> list[ExecutionPipeline]:
        """Liste les pipelines custom."""
        return [p for p in self._pipelines.values() if not p.is_builtin]

    def get(self, pipeline_id: str) -> ExecutionPipeline | None:
        """Retrieve a pipeline by ID."""
        return self._pipelines.get(pipeline_id)

    def create(self, pipeline: ExecutionPipeline) -> bool:
        """Create a new custom pipeline."""
        if pipeline.id in self._pipelines:
            logger.error(f"Pipeline '{pipeline.id}' existe already")
            return False
        errors = pipeline.validate()
        if errors:
            logger.error(f"Pipeline invalide: {errors}")
            return False
        pipeline.is_builtin = False
        now = datetime.now().isoformat()
        pipeline.created_at = now
        pipeline.updated_at = now
        self._pipelines[pipeline.id] = pipeline
        self._save_custom()
        return True

    def update(self, pipeline_id: str, pipeline: ExecutionPipeline) -> bool:
        """Update an existing custom pipeline."""
        existing = self._pipelines.get(pipeline_id)
        if not existing:
            logger.error(f"Pipeline '{pipeline_id}' non trouve")
            return False
        if existing.is_builtin:
            logger.error("Cannot modify a builtin pipeline")
            return False
        errors = pipeline.validate()
        if errors:
            logger.error(f"Pipeline invalide: {errors}")
            return False
        pipeline.is_builtin = False
        pipeline.created_at = existing.created_at
        pipeline.updated_at = datetime.now().isoformat()
        self._pipelines[pipeline_id] = pipeline
        self._save_custom()
        return True

    def delete(self, pipeline_id: str) -> bool:
        """Delete a custom pipeline."""
        existing = self._pipelines.get(pipeline_id)
        if not existing:
            return False
        if existing.is_builtin:
            logger.error("Cannot delete a builtin pipeline")
            return False
        del self._pipelines[pipeline_id]
        self._save_custom()
        return True

    def duplicate(
        self, source_id: str, new_id: str
    ) -> ExecutionPipeline | None:
        """Duplicate a pipeline (builtin or custom) as custom."""
        source = self._pipelines.get(source_id)
        if not source:
            return None
        if new_id in self._pipelines:
            logger.error(f"Pipeline '{new_id}' existe already")
            return None
        # Copie profonde
        new_steps = [
            ExecutionStep.from_dict(s.to_dict()) for s in source.steps
        ]
        now = datetime.now().isoformat()
        new_pipeline = ExecutionPipeline(
            id=new_id,
            name=f"{source.name} (copy)",
            description=source.description,
            steps=new_steps,
            created_at=now,
            updated_at=now,
            is_builtin=False,
        )
        self._pipelines[new_id] = new_pipeline
        self._save_custom()
        return new_pipeline

    def get_step_types(self) -> list[dict[str, str]]:
        """Return available step types with descriptions."""
        return [
            {"type": st, "description": STEP_TYPE_DESCRIPTIONS.get(st, "")}
            for st in VALID_STEP_TYPES
        ]


# =============================================================================
# PIPELINE RUNNER
# =============================================================================

class PipelineRunner:
    """
    Execute an execution pipeline by chaining steps
    via the AgenticExecutor.

    Each step is executed sequentially. The result of each
    step is injected into the next step's prompt if
    pass_previous_output is True.

    Optionally uses SmartRouter for automatic per-step
    model selection when no explicit model_override is set.
    """

    def __init__(self, agentic_executor=None, smart_router=None):
        """Initialize the runner.

        Args:
            agentic_executor: AgenticExecutor instance (None = lazy import)
            smart_router: SmartRouter instance for auto model selection (None = singleton)
        """
        self._executor = agentic_executor
        self._smart_router = smart_router

    def _get_executor(self):
        """Retrieve the agentic executor (lazy import)."""
        if self._executor is not None:
            return self._executor
        try:
            from .agentic_executor import agentic_executor
            return agentic_executor
        except ImportError:
            return None

    def _get_smart_router(self):
        """Retrieve the smart router."""
        if self._smart_router is not None:
            return self._smart_router
        try:
            from .smart_router import smart_router
            return smart_router
        except ImportError:
            return None

    def execute(
        self,
        pipeline: ExecutionPipeline,
        message: str,
        routing: Any,
        conversation_id: str | None = None,
        on_status: Callable[[str], None] | None = None,
        on_step_start: Callable[[int, ExecutionStep], None] | None = None,
        on_step_end: Callable[[int, ExecutionStep, str], None] | None = None,
        on_tool_call: Callable | None = None,
        on_reasoning_step: Callable | None = None,
        on_consensus_model: Callable | None = None,
        on_correction_step: Callable | None = None,
        approval_fn: Callable | None = None,
    ) -> Generator:
        """Run the pipeline step by step.

        Yields streaming chunks. Steps are chained: the result of
        each step is passed to the next one.

        Also yields special tuples:
        - ("pipeline_step_start", int, ExecutionStep) at step start
        - ("pipeline_step_end", int, str) at step end with the result

        Args:
            pipeline: Pipeline to run
            message: Original user message
            routing: RoutingResult from the router
            conversation_id: Conversation ID
            on_status: Status callback
            on_step_start: Step-start callback (index, step)
            on_step_end: Step-end callback (index, step, result)
            on_tool_call: Callback for tool calls
            on_reasoning_step: Callback for reasoning steps
            on_consensus_model: Callback for consensus answers
            on_correction_step: Callback for corrections
            approval_fn: Per-request tool-approval gate forwarded to the
                executor at every step

        Yields:
            Streaming chunks (str or tuples)
        """
        executor = self._get_executor()
        if executor is None:
            yield "[ERR] AgenticExecutor not available"
            return

        if not pipeline.steps:
            yield "[ERR] Pipeline has no steps"
            return

        # Contexte passe entre les etapes
        current_input = message
        accumulated_output = ""

        for step_idx, step in enumerate(pipeline.steps):
            # Inter-step emergency-stop check. A stop
            # engaged mid-pipeline must end the whole run honestly, not let
            # the runner chain into the next step after the inner cancel.
            _estop = _resolve_emergency_stop()
            if _estop is not None and _estop.is_stopped():
                logger.warning(
                    "PipelineRunner: emergency stop engaged, aborting at "
                    f"step {step_idx + 1}/{len(pipeline.steps)}"
                )
                yield "\n[ERR] Pipeline aborted: emergency stop engaged"
                return

            # Evaluer la condition de l'etape
            if step.condition and not self._evaluate_condition(
                step.condition, current_input, accumulated_output
            ):
                logger.info(
                    f"PipelineRunner: etape {step_idx + 1} ignoree "
                    f"(condition '{step.condition}' non remplie)"
                )
                continue

            # Signaler le debut de l'etape
            yield ("pipeline_step_start", step_idx, step)
            if on_step_start:
                on_step_start(step_idx, step)
            if on_status:
                on_status(
                    f"Step {step_idx + 1}/{len(pipeline.steps)}: "
                    f"{step.label}"
                )

            # Construire le prompt pour this etape
            if step_idx > 0 and step.pass_previous_output and accumulated_output:
                step_prompt = (
                    f"Based on the following previous analysis:\n\n"
                    f"---\n{accumulated_output}\n---\n\n"
                    f"Original question: {message}\n\n"
                    f"Now continue with: {step.label}"
                )
            elif step_idx > 0 and not step.pass_previous_output:
                # pass_previous_output=False means the step
                # runs on the original message. current_input carries the
                # previous step output here and would silently drop the
                # original question.
                step_prompt = message
            else:
                step_prompt = current_input

            # Executer l'etape via l'agentic executor
            step_output = ""
            step_failed = False
            try:
                # Configurer les overrides pour this etape
                think_override = None
                web_search_override = None
                consensus_override = None
                self_correct_override = None

                if step.step_type in ("think", "think_tools"):
                    think_override = True
                elif step.step_type == "web_search":
                    web_search_override = True
                elif step.step_type == "consensus":
                    consensus_override = True
                elif step.step_type == "self_correct":
                    self_correct_override = True

                # Model override
                step_routing = routing
                if step.model_override:
                    # Use explicit model override from step config
                    step_routing = self._override_model(
                        routing, step.model_override
                    )
                else:
                    # Try smart routing for automatic model selection
                    sr = self._get_smart_router()
                    if sr and sr.enabled:
                        step_routing = sr.override_routing(
                            routing, step.step_type
                        )

                # Parametres specifiques
                consensus_models = step.parameters.get("models")
                consensus_strategy = step.parameters.get("strategy")

                # Per-step resource admission, beside the
                # estop check above and never replacing it. The
                # step's resolved model is admitted with pipeline
                # semantics (floor 4096, downsize-then-refuse); a refusal
                # aborts the WHOLE run honestly with the established
                # prefix. requested_ctx stays None here: the inner
                # executor admission measures the real prompt.
                _governor = _resolve_resource_governor()
                if _governor is not None:
                    _gov_decision = None
                    try:
                        _gov_decision = _governor.get_resource_governor().admit(
                            step_routing.model, None, caller="pipeline"
                        )
                    except Exception as exc:
                        logger.debug(
                            "PipelineRunner: admission failed open: %s", exc
                        )
                    if _gov_decision is not None and not _gov_decision.admitted:
                        logger.warning(
                            "PipelineRunner: resource admission refused for "
                            f"{step_routing.model} at step "
                            f"{step_idx + 1}/{len(pipeline.steps)}: "
                            f"{_gov_decision.reason}"
                        )
                        yield (
                            "\n[ERR] Pipeline aborted: resource admission "
                            f"refused for {step_routing.model} "
                            f"({_gov_decision.reason})"
                        )
                        return

                for chunk in executor.execute(
                    message=step_prompt,
                    routing=step_routing,
                    conversation_id=conversation_id,
                    think=think_override,
                    web_search=web_search_override,
                    consensus=consensus_override,
                    consensus_models=consensus_models,
                    consensus_strategy=consensus_strategy,
                    self_correct=self_correct_override,
                    on_status=on_status,
                    on_tool_call=on_tool_call,
                    on_reasoning_step=on_reasoning_step,
                    on_consensus_model=on_consensus_model,
                    on_correction_step=on_correction_step,
                    # The per-request approval gate holds at every step
                    approval_fn=approval_fn,
                ):
                    # Collecter la sortie texte
                    if isinstance(chunk, str):
                        step_output += chunk
                    yield chunk

            except Exception as e:
                error_msg = f"\n[ERR] Step {step_idx + 1} failed: {e}"
                logger.error(
                    f"PipelineRunner: erreur etape {step_idx + 1}: {e}"
                )
                yield error_msg
                step_output = error_msg
                step_failed = True

            # Signaler la fin de l'etape
            yield ("pipeline_step_end", step_idx, step_output)
            if on_step_end:
                on_step_end(step_idx, step, step_output)

            # Mettre a jour le contexte
            # A failed step must not poison the chain context;
            # the next step keeps the last good context instead of receiving
            # the error text as "previous analysis".
            if not step_failed:
                accumulated_output = step_output
                current_input = step_output

    def _evaluate_condition(
        self, condition: str, current_input: str, prev_output: str
    ) -> bool:
        """Evalue une condition simple.

        Conditions supportees:
        - "if_code_detected": si du code est detecte dans l'input
        - "if_long_input": si l'input depasse 500 caracteres
        - "always": toujours True
        """
        condition = condition.strip().lower()
        if condition == "always" or not condition:
            return True
        if condition == "if_code_detected":
            # Heuristique simple: detecter des patterns de code
            code_patterns = [
                "```", "def ", "class ", "import ", "function ",
                "const ", "let ", "var ", "if (", "for (",
            ]
            text = current_input + prev_output
            return any(p in text for p in code_patterns)
        if condition == "if_long_input":
            return len(current_input) > 500
        # Unrecognized condition => True (execute by default)
        logger.warning(f"Condition non reconnue: {condition}")
        return True

    def _override_model(self, routing: Any, model: str) -> Any:
        """Create a copy of routing with a different model."""
        try:
            new_routing = copy.copy(routing)
            new_routing.model = model
            return new_routing
        except Exception:
            return routing


# =============================================================================
# SINGLETON
# =============================================================================

_pipeline_store: PipelineStore | None = None
_pipeline_runner: PipelineRunner | None = None


def get_pipeline_store() -> PipelineStore:
    """Return the PipelineStore singleton."""
    global _pipeline_store
    if _pipeline_store is None:
        _pipeline_store = PipelineStore()
    return _pipeline_store


def get_pipeline_runner() -> PipelineRunner:
    """Return the PipelineRunner singleton."""
    global _pipeline_runner
    if _pipeline_runner is None:
        _pipeline_runner = PipelineRunner()
    return _pipeline_runner
