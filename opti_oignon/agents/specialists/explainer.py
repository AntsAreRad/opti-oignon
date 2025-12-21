#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPLAINER AGENT - OPTI-OIGNON 1.0
=================================

Specialized agent for explanation and simplification.

Author: Léon
"""

from typing import Dict, Any, Optional
from ..base import BaseAgent, AgentRole, get_agent_config


class ExplainerAgent(BaseAgent):
    """
    Explanation and vulgarization agent.
    
    Specialties:
    - Code explanation
    - Concept documentation
    - Result interpretation
    - Technical simplification
    """
    
    def get_system_prompt(self, role: AgentRole, context: Dict[str, Any]) -> str:
        """Generate system prompt based on role and context."""
        
        base_prompt = """You are an expert at explaining complex technical concepts.

## YOUR APPROACH
1. **Clear**: Use simple, accessible language
2. **Structured**: Organize explanations logically
3. **Concrete**: Use examples and analogies
4. **Adapted**: Match the audience's level

## LANGUAGE RULE
Respond in the same language as the user's question.
If they ask in French, respond in French. If they ask in English, respond in English.
"""
        
        if role == AgentRole.SYNTHESIZER:
            return base_prompt + "\n\nSynthesize the information into a clear, concise summary."
        else:
            return base_prompt
    
    def explain(
        self,
        content: str,
        level: str = "intermediate",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Explain content at specified level.
        
        Args:
            content: Content to explain
            level: "beginner", "intermediate", or "expert"
            context: Additional context
            
        Returns:
            Explanation text
        """
        ctx = context or {}
        ctx["explanation_level"] = level
        
        prompt = f"Explain this at {level} level:\n\n{content}"
        
        output = self.execute(
            prompt=prompt,
            role=AgentRole.SYNTHESIZER,
            context=ctx,
        )
        
        return output.content


def create_explainer_agent(config: Optional[Dict] = None) -> ExplainerAgent:
    """
    Factory to create an explainer agent.
    
    Args:
        config: Custom configuration (optional)
        
    Returns:
        Configured ExplainerAgent instance
    """
    if config is None:
        full_config = get_agent_config()
        config = full_config.get("agents", {}).get("explainer", {})
    
    return ExplainerAgent(name="explainer", config=config)
