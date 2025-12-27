#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MULTI-AGENT SYSTEM - OPTI-OIGNON 1.0
====================================

Multi-model orchestration module for complex tasks.

This system allows combining multiple specialized LLM models
to solve problems that exceed the capabilities of a
single model.

IMPORTANT: The system can be enabled/disabled!
    - Check: is_multi_agent_enabled()
    - Enable: set_multi_agent_enabled(True)
    - Disable: set_multi_agent_enabled(False)

Architecture:
    ┌─────────────────────────────────────────┐
    │           ORCHESTRATOR                  │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────────┐  │
    │  │Coder│ │Review│ │Expl.│ │Planner  │  │
    │  └─────┘ └─────┘ └─────┘ └─────────┘  │
    │              ↓                         │
    │         PIPELINES                      │
    │  [data_analysis] [debug] [writing]    │
    └─────────────────────────────────────────┘

Basic usage:
    from opti_oignon.agents import run_auto, is_multi_agent_enabled
    
    if is_multi_agent_enabled():
        result = run_auto("Analyze this dataset")
        print(result.final_output)

Advanced usage:
    from opti_oignon.agents import Orchestrator
    
    orch = Orchestrator()
    orch.set_callbacks(
        on_step_start=lambda name, i: print(f"Step {i}: {name}"),
        on_token=lambda t: print(t, end=""),
    )
    result = orch.run_pipeline("data_analysis", "My problem...")

Author: Léon
Version: 2.0
"""

# Base imports
from .base import (
    # Data classes
    AgentOutput,
    StepResult,
    StepStatus,
    PipelineResult,
    PipelineStatus,
    AgentRole,
    BaseAgent,
    
    # Configuration
    get_agent_config,
    load_agent_config,
    
    # Multi-agent control (IMPORTANT!)
    is_multi_agent_enabled,
    set_multi_agent_enabled,
)

# Specialized agents
from .specialists import (
    CoderAgent,
    ReviewerAgent,
    ExplainerAgent,
    PlannerAgent,
    create_coder_agent,
    create_reviewer_agent,
    create_explainer_agent,
    create_planner_agent,
    create_agent,
    list_agents,
)

# Orchestrator
from .orchestrator import (
    Orchestrator,
    get_orchestrator,
    run_pipeline,
    run_auto,
)

# Dynamic Pipeline
try:
    from .dynamic_pipeline import (
        DynamicPipelinePlanner,
        DynamicPipelineExecutor,
        DynamicPipelinePlan,
        PipelineStep,
        PlanComplexity,
        plan_pipeline,
        execute_dynamic_pipeline,
        get_planner,
        get_executor,
    )
    DYNAMIC_PIPELINE_AVAILABLE = True
except ImportError:
    DYNAMIC_PIPELINE_AVAILABLE = False

# Version
__version__ = "2.0.0"

# Public exports
__all__ = [
    # Version
    "__version__",
    
    # System control (MOST IMPORTANT!)
    "is_multi_agent_enabled",
    "set_multi_agent_enabled",
    
    # Main functions
    "run_auto",
    "run_pipeline",
    "get_orchestrator",
    
    # Main classes
    "Orchestrator",
    "BaseAgent",
    
    # Specialized agents
    "CoderAgent",
    "ReviewerAgent",
    "ExplainerAgent",
    "PlannerAgent",
    
    # Agent factories
    "create_coder_agent",
    "create_reviewer_agent",
    "create_explainer_agent",
    "create_planner_agent",
    "create_agent",
    "list_agents",
    
    # Dataclasses
    "AgentOutput",
    "StepResult",
    "StepStatus",
    "PipelineResult",
    "PipelineStatus",
    "AgentRole",
    
    # Configuration
    "get_agent_config",
    "load_agent_config",
]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_run(query: str, stream: bool = False) -> str:
    """
    Quickly execute a query and return the response.
    
    This is the simplest way to use the multi-agent system.
    
    Args:
        query: Your question or problem
        stream: Enable streaming (not implemented here)
        
    Returns:
        The final response as a string
        
    Example:
        >>> from opti_oignon.agents import quick_run
        >>> answer = quick_run("How to debug this R code?")
        >>> print(answer)
    """
    if not is_multi_agent_enabled():
        # Simple mode: just use the coder
        coder = create_coder_agent()
        output = coder.execute(
            prompt=query,
            role=AgentRole.GENERATOR,
            context={"original_input": query},
        )
        return output.content
    
    result = run_auto(query, stream=stream)
    return result.final_output


def enable_multi_agent() -> bool:
    """Enable the multi-agent system."""
    return set_multi_agent_enabled(True)


def disable_multi_agent() -> bool:
    """Disable the multi-agent system."""
    return set_multi_agent_enabled(False)


def status() -> dict:
    """
    Return the current system status.
    
    Returns:
        Dict with status information
    """
    orch = get_orchestrator()
    return {
        "multi_agent_enabled": is_multi_agent_enabled(),
        "orchestrator_status": orch.status.value,
        "available_agents": list_agents(),
        "available_pipelines": [p["id"] for p in orch.list_pipelines()],
        "version": __version__,
    }


# =============================================================================
# DISPLAY ON LOAD
# =============================================================================

def _print_status():
    """Display status on load (optional)."""
    import logging
    logger = logging.getLogger("MultiAgent")
    
    enabled = is_multi_agent_enabled()
    status_emoji = "✅" if enabled else "❌"
    
    logger.debug(f"Multi-Agent System v{__version__} {status_emoji}")


# Display status on load (if debug)
try:
    _print_status()
except:
    pass
