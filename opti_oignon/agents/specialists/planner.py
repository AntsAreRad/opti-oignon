#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLANNER AGENT - OPTI-OIGNON 1.0
===============================

Specialized agent for planning and task decomposition.

Author: Léon
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from ..base import BaseAgent, AgentRole, get_agent_config


@dataclass
class Step:
    """A step in a plan."""
    number: int
    description: str
    details: str = ""
    dependencies: List[int] = field(default_factory=list)


@dataclass
class Plan:
    """A complete plan."""
    objective: str
    steps: List[Step] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    estimated_time: str = ""


class PlannerAgent(BaseAgent):
    """
    Planning and decomposition agent.
    
    Specialties:
    - Task planning
    - Problem decomposition
    - Strategy development
    - Workflow design
    """
    
    def get_system_prompt(self, role: AgentRole, context: Dict[str, Any]) -> str:
        """Generate system prompt based on role and context."""
        
        base_prompt = """You are an expert strategic planner.

## YOUR APPROACH
1. **Analytical**: Understand the full scope
2. **Systematic**: Break down into clear steps
3. **Practical**: Focus on actionable items
4. **Anticipatory**: Identify risks and dependencies

## LANGUAGE RULE
Respond in the same language as the user's question.
If they ask in French, respond in French. If they ask in English, respond in English.
"""
        
        if role == AgentRole.DECOMPOSER:
            return base_prompt + """

Decompose the task into:
1. Clear objective
2. Numbered steps
3. Dependencies between steps
4. Estimated complexity
5. Potential risks"""
        elif role == AgentRole.PROPOSER:
            return base_prompt + "\n\nPropose multiple solution approaches with pros/cons."
        else:
            return base_prompt
    
    def create_plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        """
        Create a plan for a task.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Structured Plan
        """
        ctx = context or {}
        
        output = self.execute(
            prompt=f"Create a detailed plan for:\n\n{task}",
            role=AgentRole.DECOMPOSER,
            context=ctx,
        )
        
        # Parse output into Plan (simplified)
        content = output.content
        
        # Extract numbered items as steps
        steps = []
        import re
        step_matches = re.findall(r'(\d+)[.):]\s*(.+?)(?=\n\d+[.):]\s*|\Z)', content, re.DOTALL)
        for i, (num, desc) in enumerate(step_matches[:10]):  # Limit to 10 steps
            steps.append(Step(
                number=int(num) if num.isdigit() else i + 1,
                description=desc.strip()[:200],
            ))
        
        return Plan(
            objective=task,
            steps=steps,
        )


def create_planner_agent(config: Optional[Dict] = None) -> PlannerAgent:
    """
    Factory to create a planner agent.
    
    Args:
        config: Custom configuration (optional)
        
    Returns:
        Configured PlannerAgent instance
    """
    if config is None:
        full_config = get_agent_config()
        config = full_config.get("agents", {}).get("planner", {})
    
    return PlannerAgent(name="planner", config=config)
