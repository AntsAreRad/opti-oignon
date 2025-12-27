#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORCHESTRATOR - Multi-Agent System
==================================

Main orchestrator that coordinates specialized agents
and executes pipelines.

Features:
- Execution of preconfigured pipelines
- Multi-agent coordination
- Orchestration pattern management
- Logging and metrics
- User intervention

Author: Léon
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Generator, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .base import (
    BaseAgent, AgentRole, AgentOutput,
    StepResult, StepStatus, PipelineResult, PipelineStatus,
    get_agent_config, is_multi_agent_enabled, set_multi_agent_enabled,
)
from .specialists import (
    create_coder_agent, create_reviewer_agent,
    create_explainer_agent, create_planner_agent,
    CoderAgent, ReviewerAgent, ExplainerAgent, PlannerAgent,
)

logger = logging.getLogger("Orchestrator")

# Dynamic pipeline support
try:
    from .dynamic_pipeline import (
        DynamicPipelinePlanner,
        DynamicPipelineExecutor,
        DynamicPipelinePlan,
        PipelineStep,
        plan_pipeline,
        get_planner,
        get_executor,
    )
    DYNAMIC_PIPELINE_AVAILABLE = True
except ImportError:
    DYNAMIC_PIPELINE_AVAILABLE = False
    DynamicPipelinePlanner = None

# Custom pipeline support
try:
    from ..pipeline_manager import get_pipeline_manager, Pipeline as CustomPipeline
    PIPELINE_MANAGER_AVAILABLE = True
except ImportError:
    PIPELINE_MANAGER_AVAILABLE = False
    get_pipeline_manager = None


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class Orchestrator:
    """
    Main orchestrator of the multi-agent system.
    
    Coordinates specialized agents and executes pipelines
    to solve complex tasks.
    
    Example:
        orchestrator = Orchestrator()
        result = orchestrator.run_pipeline("data_analysis", "Analyze this dataset")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Custom configuration (optional)
        """
        self.config = config or get_agent_config()
        self.global_config = self.config.get("global", {})
        
        # State
        self._status = PipelineStatus.IDLE
        self._cancel_event = threading.Event()
        self._current_pipeline: Optional[str] = None
        self._current_step: int = 0
        
        # Agents (created on demand)
        self._agents: Dict[str, BaseAgent] = {}
        
        # History
        self._history: List[PipelineResult] = []
        self._history_dir = Path(
            self.config.get("history", {}).get("directory", "data/agent_history")
        )
        self._history_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        self._on_step_start: Optional[Callable] = None
        self._on_step_complete: Optional[Callable] = None
        self._on_token: Optional[Callable] = None

        # Dynamic pipeline
        # FIX: Read enabled state from config
        dynamic_config = self.config.get("dynamic_pipeline", {})
        self._dynamic_enabled = dynamic_config.get("enabled", True)  # True by default now
        self._auto_execute_dynamic = dynamic_config.get("defaults", {}).get("auto_execute", True)
        
        logger.info(f"Orchestrator initialized (dynamic_pipeline: {self._dynamic_enabled})")

    # -------------------------------------------------------------------------
    # DYNAMIC PIPELINE METHODS
    # -------------------------------------------------------------------------

    def set_dynamic_pipeline_enabled(self, enabled: bool) -> None:
        """
        Active/désactive le mode pipeline dynamique.
        
        Args:
            enabled: True pour activer, False pour désactiver
        """
        self._dynamic_enabled = enabled
        logger.info(f"Dynamic pipeline {'enabled' if enabled else 'disabled'}")

    def is_dynamic_pipeline_enabled(self) -> bool:
        """Vérifie si le mode dynamique est activé."""
        return self._dynamic_enabled and DYNAMIC_PIPELINE_AVAILABLE

    def set_auto_execute_dynamic(self, auto_execute: bool) -> None:
        """
        Configure l'exécution automatique ou avec confirmation.
        
        Args:
            auto_execute: True pour exécuter sans confirmation, False pour demander
        """
        self._auto_execute_dynamic = auto_execute

    def get_dynamic_planner(self) -> 'DynamicPipelinePlanner':
        """Récupère le planificateur dynamique (lazy init)."""
        if not DYNAMIC_PIPELINE_AVAILABLE:
            raise RuntimeError("Dynamic pipeline module not available")
        
        if self._dynamic_planner is None:
            self._dynamic_planner = DynamicPipelinePlanner(self.config)
        return self._dynamic_planner

    def get_dynamic_executor(self) -> 'DynamicPipelineExecutor':
        """Récupère l'exécuteur dynamique (lazy init)."""
        if not DYNAMIC_PIPELINE_AVAILABLE:
            raise RuntimeError("Dynamic pipeline module not available")
        
        if self._dynamic_executor is None:
            self._dynamic_executor = DynamicPipelineExecutor(self.config)
        return self._dynamic_executor

    def plan_dynamic(
        self, 
        user_input: str, 
        context: str = "",
        fast_mode: bool = False,
    ) -> 'DynamicPipelinePlan':
        """
        Génère un plan de pipeline dynamique.
        
        Args:
            user_input: Requête utilisateur
            context: Contexte additionnel (document, code)
            fast_mode: Mode rapide (moins précis)
            
        Returns:
            DynamicPipelinePlan avec les étapes recommandées
        """
        planner = self.get_dynamic_planner()
        planner.fast_mode = fast_mode
        return planner.plan(user_input, context)

    def run_dynamic_pipeline(
        self,
        user_input: str,
        context: Optional[Dict] = None,
        plan: Optional['DynamicPipelinePlan'] = None,
        stream: bool = True,
        auto_plan: bool = True,
    ) -> Generator[str, None, PipelineResult]:
        """
        Exécute un pipeline dynamique avec planification automatique.
        
        Si aucun plan n'est fourni et auto_plan=True, génère d'abord
        un plan optimal avant exécution.
        
        Args:
            user_input: Input utilisateur
            context: Contexte additionnel
            plan: Plan pré-généré (optionnel)
            stream: Active le streaming
            auto_plan: Si True et pas de plan, génère automatiquement
            
        Yields:
            Tokens de la réponse (si streaming)
            
        Returns:
            PipelineResult final
        """
        if not self.is_dynamic_pipeline_enabled():
            # Fallback vers mode standard
            yield "[!] Dynamic pipeline disabled, using standard routing\n"
            detected = self.detect_pipeline(user_input)
            pipeline_id = detected or "quick"
            result = self.run_pipeline(pipeline_id, user_input, context, stream=False)
            yield result.final_output or ""
            return result
        
        # Génère le plan si nécessaire
        if plan is None and auto_plan:
            yield "[>] Analyzing and planning pipeline...\n"
            document = context.get("document", "") if context else ""
            plan = self.plan_dynamic(user_input, document)
            yield f"[OK] Plan generated: {plan.step_count} steps, {plan.complexity.value}\n\n"
        
        if plan is None:
            yield "[ERR] No plan available\n"
            return PipelineResult(
                pipeline_name="dynamic_error",
                status=PipelineStatus.FAILED,
                steps=[],
                final_output="No plan available",
            )
        
        # Vérifie si multi-agent nécessaire
        planner = self.get_dynamic_planner()
        if not planner.should_use_pipeline(plan):
            yield f"[INFO] Single model sufficient for this task\n"
            # Exécute avec un seul modèle
            step = plan.recommended_pipeline[0]
            yield f"[>] Using {step.model} directly...\n\n"
            
            # Exécution simple
            try:
                from .executor import executor as simple_executor
                from .router import RoutingResult
                
                routing = RoutingResult(
                    model=step.model,
                    task_type=step.agent_type,
                    temperature=0.4,
                    prompt_variant="standard",
                    timeout=step.timeout,
                )
                
                full_response = ""
                for token in simple_executor.execute(
                    user_input,
                    routing,
                    document=context.get("document") if context else None,
                    refine=False,
                ):
                    full_response += token
                    yield token
                
                return PipelineResult(
                    pipeline_name="dynamic_single",
                    status=PipelineStatus.COMPLETED,
                    steps=[StepResult(
                        step_name=step.agent_type,
                        step_index=0,
                        status=StepStatus.COMPLETED,
                        output=AgentOutput(
                            content=full_response,
                            agent_name=step.agent_type,
                            model_used=step.model,
                            role=AgentRole.GENERATOR,
                        ),
                    )],
                    final_output=full_response,
                )
                
            except Exception as e:
                logger.error(f"Single model execution error: {e}")
                yield f"\n[ERR] Error: {e}"
                return PipelineResult(
                    pipeline_name="dynamic_error",
                    status=PipelineStatus.FAILED,
                    steps=[],
                    final_output=f"Error: {e}",
                )
        
        # Exécution multi-étapes
        yield plan.format_preview(show_models=True) + "\n\n"
        yield "═" * 40 + "\n"
        yield "**EXECUTION**\n"
        yield "═" * 40 + "\n\n"
        
        executor = self.get_dynamic_executor()
        steps_results = []
        final_output = ""
        start_time = time.time()
        
        # Callbacks internes
        def on_step_start(step_num, agent_type, model):
            if self._on_step_start:
                self._on_step_start(f"Step {step_num}: {agent_type}", step_num - 1)
        
        def on_step_complete(step_num, output, time_taken):
            step_result = StepResult(
                step_name=f"dynamic_step_{step_num}",
                step_index=step_num - 1,
                status=StepStatus.COMPLETED,
                output=AgentOutput(
                    content=output,
                    agent_name=plan.recommended_pipeline[step_num - 1].agent_type,
                    model_used=plan.recommended_pipeline[step_num - 1].model,
                    role=AgentRole.GENERATOR,
                    execution_time=time_taken,
                ),
            )
            steps_results.append(step_result)
            if self._on_step_complete:
                self._on_step_complete(step_result)
        
        # Exécute
        try:
            gen = executor.execute(
                plan=plan,
                user_input=user_input,
                context=context,
                stream=stream,
                on_step_start=on_step_start,
                on_step_complete=on_step_complete,
            )
            
            for token in gen:
                final_output += token
                yield token
            
            # Récupère le résultat final du générateur
            result_dict = gen.send(None) if hasattr(gen, 'send') else {}
            
        except StopIteration as e:
            result_dict = e.value or {}
        except Exception as e:
            logger.error(f"Dynamic pipeline execution error: {e}")
            yield f"\n\n[ERR] Execution error: {e}"
            return PipelineResult(
                pipeline_name="dynamic_error",
                status=PipelineStatus.FAILED,
                steps=steps_results,
                final_output=final_output,
            )
        
        total_time = time.time() - start_time
        
        yield f"\n\n{'═' * 40}\n"
        yield f"**Completed in {total_time:.1f}s** ({len(steps_results)} steps)\n"
        
        return PipelineResult(
            pipeline_name=f"dynamic_{plan.complexity.value}",
            status=PipelineStatus.COMPLETED,
            steps=steps_results,
            final_output=result_dict.get("final_output", final_output),
            total_time=total_time,
            metadata={"plan": plan.to_dict()},
        )
    
    @property
    def status(self) -> PipelineStatus:
        """Return current status."""
        return self._status
    
    # -------------------------------------------------------------------------
    # AGENT MANAGEMENT
    # -------------------------------------------------------------------------
    
    def get_agent(self, agent_type: str) -> BaseAgent:
        """
        Get or create an agent of the specified type.
        
        Args:
            agent_type: "coder", "reviewer", "explainer", "planner", etc.
            
        Returns:
            Agent instance
        """
        if agent_type in self._agents:
            return self._agents[agent_type]
        
        # Create agent
        agent_config = self.config.get("agents", {}).get(agent_type, {})
        
        factories = {
            "coder": create_coder_agent,
            "reviewer": create_reviewer_agent,
            "explainer": create_explainer_agent,
            "planner": create_planner_agent,
            "writer": lambda cfg: create_explainer_agent(
                self.config.get("agents", {}).get("writer", cfg)
            ),
        }
        
        factory = factories.get(agent_type)
        if factory is None:
            logger.warning(f"Unknown agent type: {agent_type}, using 'coder'")
            factory = create_coder_agent
            agent_config = self.config.get("agents", {}).get("coder", {})
        
        agent = factory(agent_config)
        self._agents[agent_type] = agent
        
        return agent
    
    def list_agents(self) -> List[str]:
        """List available agent types."""
        return list(self.config.get("agents", {}).keys())
    
    # -------------------------------------------------------------------------
    # PIPELINE MANAGEMENT
    # -------------------------------------------------------------------------
    
    def list_pipelines(self) -> List[Dict[str, str]]:
        """
        List available pipelines (builtin + custom).
        
        Returns:
            List of pipeline info dicts
        """
        result = []
        
        # Use PipelineManager if available (includes custom pipelines)
        if PIPELINE_MANAGER_AVAILABLE and get_pipeline_manager:
            try:
                pm = get_pipeline_manager()
                for pipeline in pm.list_all():
                    result.append({
                        "id": pipeline.id,
                        "name": pipeline.name,
                        "description": pipeline.description,
                        "emoji": pipeline.emoji,
                        "steps": pipeline.step_count,
                        "is_builtin": pipeline.is_builtin,
                    })
                return result
            except Exception as e:
                logger.warning(f"PipelineManager error: {e}, falling back to config")
        
        # Fallback to config.yaml only
        pipelines = self.config.get("pipelines", {})
        for pipe_id, pipe_config in pipelines.items():
            result.append({
                "id": pipe_id,
                "name": pipe_config.get("name", pipe_id),
                "description": pipe_config.get("description", ""),
            })
        return result
    
    def detect_pipeline(self, query: str) -> Optional[str]:
        """
        Détecte automatiquement le pipeline approprié pour la requête.
        
        Analyse la requête utilisateur et compare avec les mots-clés
        de chaque pipeline (builtin + custom).
        
        Args:
            query: Question ou requête de l'utilisateur
            
        Returns:
            ID du pipeline détecté ou None si aucun match significatif
            
        Example:
            >>> orch = Orchestrator()
            >>> orch.detect_pipeline("J'ai une erreur dans mon code R")
            'debug'
            >>> orch.detect_pipeline("Analyse ces données de biodiversité")
            'data_analysis'
        """
        if not query:
            return None
        
        query_lower = query.lower()
        
        # Récupérer la config d'auto-détection
        auto_detection = self.config.get("auto_detection", {})
        if not auto_detection.get("enabled", True):
            logger.debug("Auto-détection désactivée dans la config")
            return None
        
        # Try PipelineManager first (includes custom pipelines with weighted scoring)
        if PIPELINE_MANAGER_AVAILABLE and get_pipeline_manager:
            try:
                pm = get_pipeline_manager()
                results = pm.find_by_keywords_with_scores(query, min_matches=1)
                if results:
                    best_pipeline, score, matches = results[0]
                    logger.info(f"Pipeline détecté: '{best_pipeline.id}' ({matches} keywords, score: {score:.2f})")
                    return best_pipeline.id
            except Exception as e:
                logger.warning(f"PipelineManager detection error: {e}, falling back to config")
        
        # Fallback: use config.yaml pipelines
        priority_order = auto_detection.get("priority", [])
        pipelines = self.config.get("pipelines", {})
        if not priority_order:
            priority_order = list(pipelines.keys())
        
        # Scores de match pour chaque pipeline
        scores: Dict[str, Tuple[int, float]] = {}  # (matches, score)
        
        for pipe_id in priority_order:
            pipe_config = pipelines.get(pipe_id)
            if not pipe_config:
                continue
            
            # Récupérer les keywords d'auto-détection
            auto_detect = pipe_config.get("auto_detect")
            if not auto_detect:
                continue
            
            keywords = auto_detect.get("keywords", [])
            if not keywords:
                continue
            
            # Calculer le score : nombre de keywords trouvés
            matches = sum(1 for kw in keywords if kw.lower() in query_lower)
            
            if matches > 0:
                # Score basé sur le nombre absolu de matches + ratio
                ratio = matches / len(keywords)
                score = matches + ratio
                
                scores[pipe_id] = (matches, score)
                logger.debug(f"Pipeline '{pipe_id}': {matches} keywords matched, score={score:.2f}")
        
        if not scores:
            logger.debug("Aucun pipeline détecté automatiquement")
            return None
        
        # Trouver le meilleur score
        best_pipeline = max(scores, key=lambda x: scores[x][1])
        matches, score = scores[best_pipeline]
        
        if matches >= 1:
            logger.info(f"Pipeline détecté: '{best_pipeline}' ({matches} keywords, score: {score:.2f})")
            return best_pipeline
        
        return None
    
    def get_pipeline_config(self, pipeline_id: str) -> Optional[Dict]:
        """
        Get pipeline configuration (supports builtin + custom).
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            Pipeline config dict or None
        """
        # Try PipelineManager first (includes custom pipelines)
        if PIPELINE_MANAGER_AVAILABLE and get_pipeline_manager:
            try:
                pm = get_pipeline_manager()
                pipeline = pm.get(pipeline_id)
                if pipeline:
                    return pipeline.to_config_dict()
            except Exception as e:
                logger.warning(f"PipelineManager error: {e}, falling back to config")
        
        # Fallback to config.yaml
        return self.config.get("pipelines", {}).get(pipeline_id)
    
    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------
    
    def set_callbacks(
        self,
        on_step_start: Optional[Callable[[str, int], None]] = None,
        on_step_complete: Optional[Callable[[StepResult], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ):
        """
        Set callbacks for pipeline execution.
        
        Args:
            on_step_start: Called when a step starts (name, index)
            on_step_complete: Called when a step completes
            on_token: Called for each streaming token
        """
        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._on_token = on_token
    
    # -------------------------------------------------------------------------
    # PIPELINE EXECUTION
    # -------------------------------------------------------------------------
    
    def run_pipeline(
        self,
        pipeline_id: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> PipelineResult:
        """
        Run a pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            user_input: User input/question
            context: Additional context
            stream: Enable streaming
            
        Returns:
            PipelineResult with all outputs
        """
        pipeline_config = self.get_pipeline_config(pipeline_id)
        if not pipeline_config:
            return PipelineResult(
                pipeline_name=pipeline_id,
                status=PipelineStatus.FAILED,
                steps=[],
                final_output=f"Pipeline not found: {pipeline_id}",
            )
        
        self._status = PipelineStatus.RUNNING
        self._current_pipeline = pipeline_id
        self._cancel_event.clear()
        
        start_time = time.time()
        steps_results = []
        context = context or {}
        context["original_input"] = user_input
        
        steps = pipeline_config.get("steps", [])
        
        for i, step_config in enumerate(steps):
            if self._cancel_event.is_set():
                break
            
            step_name = step_config.get("name", f"Step {i+1}")
            agent_type = step_config.get("agent", "coder")
            
            # Callback
            if self._on_step_start:
                self._on_step_start(step_name, i)
            
            # Get agent
            if agent_type == "auto":
                agent = self._select_auto_agent(user_input, context)
            else:
                agent = self.get_agent(agent_type)
            
            # Build prompt
            prompt = self._build_step_prompt(
                step_config, 
                user_input, 
                steps_results,
                context
            )
            
            # Execute
            try:
                output = agent.execute(
                    prompt=prompt,
                    role=self._get_role_for_step(step_config),
                    context=context,
                    stream=stream,
                    on_token=self._on_token if stream else None,
                )
                
                step_result = StepResult(
                    step_name=step_name,
                    step_index=i,
                    status=StepStatus.COMPLETED,
                    output=output,
                )
                
                # Update context
                context["previous_output"] = output.content
                
            except Exception as e:
                logger.error(f"Step '{step_name}' error: {e}")
                step_result = StepResult(
                    step_name=step_name,
                    step_index=i,
                    status=StepStatus.FAILED,
                    error=str(e),
                )
            
            steps_results.append(step_result)
            
            # Callback
            if self._on_step_complete:
                self._on_step_complete(step_result)
        
        # Build result
        total_time = time.time() - start_time
        final_output = context.get("previous_output", "")
        
        status = PipelineStatus.COMPLETED
        if self._cancel_event.is_set():
            status = PipelineStatus.CANCELLED
        elif any(s.status == StepStatus.FAILED for s in steps_results):
            status = PipelineStatus.FAILED
        
        result = PipelineResult(
            pipeline_name=pipeline_id,
            status=status,
            steps=steps_results,
            final_output=final_output,
            total_time=total_time,
            total_tokens=sum(s.output.token_count for s in steps_results if s.output),
        )
        
        self._status = PipelineStatus.IDLE
        self._current_pipeline = None
        self._history.append(result)
        
        return result
    
    def _build_step_prompt(
        self,
        step_config: Dict,
        original_input: str,
        previous_steps: List[StepResult],
        context: Dict,
    ) -> str:
        """Build the prompt for a step."""
        template_name = step_config.get("prompt_template", "direct")
        templates = self.config.get("prompt_templates", {})
        template = templates.get(template_name, "{input}")
        
        # Build substitution dict
        subs = {
            "input": original_input,
            "original_input": original_input,
            "previous_output": context.get("previous_output", ""),
            "context": json.dumps(context, default=str)[:2000],
        }
        
        # Apply template
        try:
            return template.format(**subs)
        except KeyError:
            return original_input
    
    def _get_role_for_step(self, step_config: Dict) -> AgentRole:
        """Determine the agent role for a step."""
        step_name = step_config.get("name", "").lower()
        
        if "review" in step_name or "verify" in step_name:
            return AgentRole.VERIFIER
        elif "plan" in step_name:
            return AgentRole.DECOMPOSER
        elif "explain" in step_name or "interpret" in step_name:
            return AgentRole.SYNTHESIZER
        else:
            return AgentRole.GENERATOR
    
    def _select_auto_agent(self, input_text: str, context: Dict) -> BaseAgent:
        """Auto-select the best agent for the input."""
        input_lower = input_text.lower()
        
        if any(k in input_lower for k in ["code", "function", "script", "implement"]):
            return self.get_agent("coder")
        elif any(k in input_lower for k in ["review", "check", "verify", "bug"]):
            return self.get_agent("reviewer")
        elif any(k in input_lower for k in ["explain", "what", "why", "how"]):
            return self.get_agent("explainer")
        elif any(k in input_lower for k in ["plan", "steps", "strategy"]):
            return self.get_agent("planner")
        else:
            return self.get_agent("coder")
    
    # -------------------------------------------------------------------------
    # CONTROL
    # -------------------------------------------------------------------------
    
    def cancel(self):
        """Cancel current execution."""
        self._cancel_event.set()
        logger.info("Cancellation requested")
    
    def is_running(self) -> bool:
        """Check if a pipeline is running."""
        return self._status == PipelineStatus.RUNNING


# =============================================================================
# GLOBAL INSTANCE AND CONVENIENCE FUNCTIONS
# =============================================================================

_orchestrator: Optional[Orchestrator] = None


def get_orchestrator(config: Optional[Dict] = None) -> Orchestrator:
    """Get the global orchestrator instance (singleton)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(config)
    return _orchestrator


def run_pipeline(
    pipeline_id: str,
    user_input: str,
    context: Optional[Dict] = None,
    stream: bool = False,
) -> PipelineResult:
    """
    Convenience function to run a pipeline.
    
    Args:
        pipeline_id: Pipeline ID
        user_input: User input
        context: Additional context
        stream: Enable streaming
        
    Returns:
        PipelineResult
    """
    return get_orchestrator().run_pipeline(pipeline_id, user_input, context, stream)


def run_auto(
    user_input: str,
    context: Optional[Dict] = None,
    stream: bool = False,
) -> PipelineResult:
    """
    Run with automatic pipeline detection.
    
    Args:
        user_input: User input
        context: Additional context
        stream: Enable streaming
        
    Returns:
        PipelineResult
    """
    orch = get_orchestrator()
    
    # Simple auto-detection
    input_lower = user_input.lower()
    
    keywords_to_pipeline = {
        "debug": ["error", "bug", "traceback", "crash", "erreur"],
        "data_analysis": ["analysis", "data", "correlation", "analyse", "données"],
        "scientific_writing": ["abstract", "methods", "discussion", "article"],
        "code_with_tests": ["test", "function", "implement"],
    }
    
    for pipeline_id, keywords in keywords_to_pipeline.items():
        if any(k in input_lower for k in keywords):
            if orch.get_pipeline_config(pipeline_id):
                return orch.run_pipeline(pipeline_id, user_input, context, stream)
    
    # Fallback to quick
    return orch.run_pipeline("quick", user_input, context, stream)


def is_dynamic_pipeline_available() -> bool:
    """Vérifie si le module de pipeline dynamique est disponible."""
    return DYNAMIC_PIPELINE_AVAILABLE


def run_dynamic(
    user_input: str,
    context: Optional[Dict] = None,
    stream: bool = True,
) -> Generator[str, None, PipelineResult]:
    """
    Fonction de commodité pour exécuter un pipeline dynamique.
    
    Args:
        user_input: Input utilisateur
        context: Contexte additionnel
        stream: Active le streaming
        
    Yields:
        Tokens de la réponse
        
    Returns:
        PipelineResult final
    """
    orch = get_orchestrator()
    return orch.run_dynamic_pipeline(user_input, context, stream=stream)


# =============================================================================
# CLI FOR TESTS
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Orchestrator Test ===\n")
    
    orch = Orchestrator()
    
    print(f"Status: {orch.status.value}")
    print(f"Multi-agent enabled: {is_multi_agent_enabled()}")
    
    print("\nAvailable agents:")
    for agent in orch.list_agents():
        print(f"  - {agent}")
    
    print("\nAvailable pipelines:")
    for pipeline in orch.list_pipelines():
        print(f"  - {pipeline['id']}: {pipeline['name']}")
    
    print("\n✅ Orchestrator functional")
