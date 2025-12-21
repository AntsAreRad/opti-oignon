#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CODER AGENT - OPTI-OIGNON 1.0
=============================

Specialized agent for code generation (R, Python, Bash).

Author: Léon
"""

from typing import Dict, Any, Optional
from ..base import BaseAgent, AgentRole, get_agent_config


class CoderAgent(BaseAgent):
    """
    Code generation agent.
    
    Specialties:
    - R code (tidyverse, bioinformatics)
    - Python code (data science, type hints)
    - Bash scripts
    - Code refactoring
    """
    
    def get_system_prompt(self, role: AgentRole, context: Dict[str, Any]) -> str:
        """Generate system prompt based on role and context."""
        
        base_prompt = """You are an expert code generator specializing in R, Python, and Bash.

## YOUR RULES
1. **Clean code**: Well-structured, commented, following best practices
2. **Error handling**: Include tryCatch/try-except when relevant
3. **Documentation**: Clear comments and docstrings
4. **Type safety**: Use type hints (Python) and validation

## LANGUAGE RULE
Respond in the same language as the user's question.
If they ask in French, respond in French. If they ask in English, respond in English.
"""
        
        if role == AgentRole.GENERATOR:
            return base_prompt + "\n\nGenerate complete, functional code for the request."
        elif role == AgentRole.REVISOR:
            previous = context.get("previous_output", "")
            return base_prompt + f"""

You are revising code based on feedback:
{previous}

Incorporate the feedback and produce improved code."""
        else:
            return base_prompt
    
    def generate_code(
        self,
        request: str,
        language: str = "auto",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate code for a request.
        
        Args:
            request: Code generation request
            language: "r", "python", "bash", or "auto"
            context: Additional context
            
        Returns:
            Generated code
        """
        ctx = context or {}
        ctx["language_hint"] = language
        
        output = self.execute(
            prompt=request,
            role=AgentRole.GENERATOR,
            context=ctx,
        )
        
        return output.content


def create_coder_agent(config: Optional[Dict] = None) -> CoderAgent:
    """
    Factory to create a coder agent.
    
    Args:
        config: Custom configuration (optional)
        
    Returns:
        Configured CoderAgent instance
    """
    if config is None:
        full_config = get_agent_config()
        config = full_config.get("agents", {}).get("coder", {})
    
    return CoderAgent(name="coder", config=config)
