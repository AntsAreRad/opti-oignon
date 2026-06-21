#!/usr/bin/env python3
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

import json
import logging
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Optional

from .base import (
    AgentOutput,
    AgentRole,
    BaseAgent,
    PipelineResult,
    PipelineStatus,
    StepResult,
    StepStatus,
    get_agent_config,
    is_multi_agent_enabled,
)
from .specialists import (
    create_coder_agent,
    create_explainer_agent,
    create_planner_agent,
    create_reviewer_agent,
)

logger = logging.getLogger("Orchestrator")

# Dynamic pipeline support
try:
    from .dynamic_pipeline import (
        DynamicPipelineExecutor,
        DynamicPipelinePlan,
        DynamicPipelinePlanner,
        PipelineStep,  # noqa: F401
        get_executor,  # noqa: F401
        get_planner,  # noqa: F401
        plan_pipeline,  # noqa: F401
    )
    DYNAMIC_PIPELINE_AVAILABLE = True
except ImportError:
    DYNAMIC_PIPELINE_AVAILABLE = False
    DynamicPipelinePlanner = None

# Custom pipeline support
try:
    from ..pipeline_manager import Pipeline as CustomPipeline  # noqa: F401
    from ..pipeline_manager import get_pipeline_manager
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

    def __init__(self, config: dict | None = None):
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
        self._current_pipeline: str | None = None
        self._current_step: int = 0

        # Agents (created on demand)
        self._agents: dict[str, BaseAgent] = {}

        # History
        self._history: list[PipelineResult] = []
        self._history_dir = Path(
            self.config.get("history", {}).get("directory", "data/agent_history")
        )
        self._history_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks
        self._on_step_start: Callable | None = None
        self._on_step_complete: Callable | None = None
        self._on_token: Callable | None = None

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
        Enable/disable dynamic pipeline mode.

        Args:
            enabled: True to enable, False to disable
        """
        self._dynamic_enabled = enabled
        logger.info(f"Dynamic pipeline {'enabled' if enabled else 'disabled'}")

    def is_dynamic_pipeline_enabled(self) -> bool:
        """Check if dynamic mode is enabled."""
        return self._dynamic_enabled and DYNAMIC_PIPELINE_AVAILABLE

    def set_auto_execute_dynamic(self, auto_execute: bool) -> None:
        """
        Configure automatic or confirmation-required execution.

        Args:
            auto_execute: True to execute without confirmation, False to ask
        """
        self._auto_execute_dynamic = auto_execute

    def get_dynamic_planner(self) -> 'DynamicPipelinePlanner':
        """Retrieve the dynamic planner (lazy init)."""
        if not DYNAMIC_PIPELINE_AVAILABLE:
            raise RuntimeError("Dynamic pipeline module not available")

        if self._dynamic_planner is None:
            self._dynamic_planner = DynamicPipelinePlanner(self.config)
        return self._dynamic_planner

    def get_dynamic_executor(self) -> 'DynamicPipelineExecutor':
        """Retrieve the dynamic executor (lazy init)."""
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
        Generate a dynamic pipeline plan.

        Args:
            user_input: User request
            context: Contexte additionnel (document, code)
            fast_mode: Fast mode (less accurate)

        Returns:
            DynamicPipelinePlan with recommended steps
        """
        planner = self.get_dynamic_planner()
        planner.fast_mode = fast_mode
        return planner.plan(user_input, context)

    def run_dynamic_pipeline(
        self,
        user_input: str,
        context: dict | None = None,
        plan: Optional['DynamicPipelinePlan'] = None,
        stream: bool = True,
        auto_plan: bool = True,
    ) -> Generator[str, None, PipelineResult]:
        """
        Execute a dynamic pipeline with automatic planning.

        If no plan is provided and auto_plan=True, first generates
        an optimal plan before execution.

        Args:
            user_input: Input utilisateur
            context: Contexte additionnel
            plan: Pre-generated plan (optional)
            stream: Active le streaming
            auto_plan: If True and no plan, auto-generate

        Yields:
            Response tokens (if streaming)

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

        # Generate plan if needed
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

        # Check if multi-agent needed
        planner = self.get_dynamic_planner()
        if not planner.should_use_pipeline(plan):
            yield "[INFO] Single model sufficient for this task\n"
            # Execute with a single model
            step = plan.recommended_pipeline[0]
            yield f"[>] Using {step.model} directly...\n\n"

            # Simple execution
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

        # Multi-step execution
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

        # Execute
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

            # Retrieve final result from generator
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

    def list_agents(self) -> list[str]:
        """List available agent types."""
        return list(self.config.get("agents", {}).keys())

    # -------------------------------------------------------------------------
    # PIPELINE MANAGEMENT
    # -------------------------------------------------------------------------

    def list_pipelines(self) -> list[dict[str, str]]:
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

    def detect_pipeline(self, query: str) -> str | None:
        """
        Automatically detect the appropriate pipeline for the request.

        Analyzes the user request and compares with keywords
        de each pipeline (builtin + custom).

        Args:
            query: User question or request

        Returns:
            Detected pipeline ID or None if no significant match

        Example:
            >>> orch = Orchestrator()
            >>> orch.detect_pipeline("J'ai une erreur dans mon code R")
            'debug'
            >>> orch.detect_pipeline("Analyze this biodiversity data")
            'data_analysis'
        """
        if not query:
            return None

        query_lower = query.lower()

        # Retrieve auto-detection config
        auto_detection = self.config.get("auto_detection", {})
        if not auto_detection.get("enabled", True):
            logger.debug("Auto-detection disabled in config")
            return None

        # Try PipelineManager first (includes custom pipelines with weighted scoring)
        if PIPELINE_MANAGER_AVAILABLE and get_pipeline_manager:
            try:
                pm = get_pipeline_manager()
                results = pm.find_by_keywords_with_scores(query, min_matches=1)
                if results:
                    best_pipeline, score, matches = results[0]
                    logger.info(f"Pipeline detected: '{best_pipeline.id}' ({matches} keywords, score: {score:.2f})")
                    return best_pipeline.id
            except Exception as e:
                logger.warning(f"PipelineManager detection error: {e}, falling back to config")

        # Fallback: use config.yaml pipelines
        priority_order = auto_detection.get("priority", [])
        pipelines = self.config.get("pipelines", {})
        if not priority_order:
            priority_order = list(pipelines.keys())

        # Scores de match pour each pipeline
        scores: dict[str, tuple[int, float]] = {}  # (matches, score)

        for pipe_id in priority_order:
            pipe_config = pipelines.get(pipe_id)
            if not pipe_config:
                continue

            # Retrieve auto-detection keywords
            auto_detect = pipe_config.get("auto_detect")
            if not auto_detect:
                continue

            keywords = auto_detect.get("keywords", [])
            if not keywords:
                continue

            # Compute score: number of keywords found
            matches = sum(1 for kw in keywords if kw.lower() in query_lower)

            if matches > 0:
                # Score based on absolute match count + ratio
                ratio = matches / len(keywords)
                score = matches + ratio

                scores[pipe_id] = (matches, score)
                logger.debug(f"Pipeline '{pipe_id}': {matches} keywords matched, score={score:.2f}")

        if not scores:
            logger.debug("No pipeline detected automatically")
            return None

        # Trouver le meilleur score
        best_pipeline = max(scores, key=lambda x: scores[x][1])
        matches, score = scores[best_pipeline]

        if matches >= 1:
            logger.info(f"Pipeline detected: '{best_pipeline}' ({matches} keywords, score: {score:.2f})")
            return best_pipeline

        return None

    def get_pipeline_config(self, pipeline_id: str) -> dict | None:
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
        on_step_start: Callable[[str, int], None] | None = None,
        on_step_complete: Callable[[StepResult], None] | None = None,
        on_token: Callable[[str], None] | None = None,
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
        context: dict[str, Any] | None = None,
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
        step_config: dict,
        original_input: str,
        previous_steps: list[StepResult],
        context: dict,
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

    def _get_role_for_step(self, step_config: dict) -> AgentRole:
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

    def _select_auto_agent(self, input_text: str, context: dict) -> BaseAgent:
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

_orchestrator: Orchestrator | None = None


def get_orchestrator(config: dict | None = None) -> Orchestrator:
    """Get the global orchestrator instance (singleton)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(config)
    return _orchestrator


def run_pipeline(
    pipeline_id: str,
    user_input: str,
    context: dict | None = None,
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
    context: dict | None = None,
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
        "data_analysis": ["analysis", "data", "correlation", "analyse", "data"],
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
    """Check if the dynamic pipeline module is available."""
    return DYNAMIC_PIPELINE_AVAILABLE


def run_dynamic(
    user_input: str,
    context: dict | None = None,
    stream: bool = True,
) -> Generator[str, None, PipelineResult]:
    """
    Convenience function to execute a dynamic pipeline.

    Args:
        user_input: Input utilisateur
        context: Contexte additionnel
        stream: Active le streaming

    Yields:
        Response tokens

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
