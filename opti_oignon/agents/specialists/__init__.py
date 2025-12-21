#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPECIALIZED AGENTS - OPTI-OIGNON 1.0
====================================

This module exposes the specialized agents of the multi-agent system.

Available agents:
- CoderAgent: R/Python/Bash code generation
- ReviewerAgent: Code verification and critique
- ExplainerAgent: Explanation and simplification
- PlannerAgent: Planning and decomposition

Usage:
    from opti_oignon.agents.specialists import create_coder_agent
    
    coder = create_coder_agent()
    code = coder.generate_code("R function to calculate Shannon")
"""

from .coder import CoderAgent, create_coder_agent
from .reviewer import ReviewerAgent, create_reviewer_agent, ReviewResult
from .explainer import ExplainerAgent, create_explainer_agent
from .planner import PlannerAgent, create_planner_agent, Plan, Step

__all__ = [
    # Agent classes
    "CoderAgent",
    "ReviewerAgent",
    "ExplainerAgent",
    "PlannerAgent",
    
    # Factories
    "create_coder_agent",
    "create_reviewer_agent",
    "create_explainer_agent",
    "create_planner_agent",
    
    # Useful dataclasses
    "ReviewResult",
    "Plan",
    "Step",
]


def create_agent(agent_type: str, config: dict = None):
    """
    Generic factory to create an agent by its type.
    
    Args:
        agent_type: "coder", "reviewer", "explainer", or "planner"
        config: Custom configuration (optional)
        
    Returns:
        Instance of the appropriate agent
        
    Raises:
        ValueError: If agent type is unknown
    """
    factories = {
        "coder": create_coder_agent,
        "reviewer": create_reviewer_agent,
        "explainer": create_explainer_agent,
        "planner": create_planner_agent,
    }
    
    factory = factories.get(agent_type.lower())
    if factory is None:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(factories.keys())}")
    
    return factory(config)


def list_agents() -> list:
    """List available agent types."""
    return ["coder", "reviewer", "explainer", "planner"]
