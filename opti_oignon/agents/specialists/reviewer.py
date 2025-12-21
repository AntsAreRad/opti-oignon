#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REVIEWER AGENT - OPTI-OIGNON 1.0
================================

Specialized agent for code review and verification.

Author: Léon
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from ..base import BaseAgent, AgentRole, get_agent_config


@dataclass
class ReviewResult:
    """Result of a code review."""
    score: float  # 0-100
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    is_valid: bool = True
    summary: str = ""


class ReviewerAgent(BaseAgent):
    """
    Code review and verification agent.
    
    Specialties:
    - Code review
    - Bug detection
    - Logic verification
    - Best practices check
    """
    
    def get_system_prompt(self, role: AgentRole, context: Dict[str, Any]) -> str:
        """Generate system prompt based on role and context."""
        
        base_prompt = """You are an expert code reviewer with extensive experience.

## YOUR APPROACH
1. **Thorough**: Check syntax, logic, edge cases
2. **Constructive**: Provide actionable feedback
3. **Precise**: Point to specific lines/issues
4. **Helpful**: Suggest improvements

## LANGUAGE RULE
Respond in the same language as the user's question.
If they ask in French, respond in French. If they ask in English, respond in English.
"""
        
        if role == AgentRole.VERIFIER:
            return base_prompt + """

Review the code and provide:
1. Score (0-100)
2. Issues found
3. Suggestions for improvement
4. Overall verdict (VALID/NEEDS_WORK)"""
        elif role == AgentRole.CRITIC:
            return base_prompt + "\n\nProvide critical analysis of the approach and solution."
        else:
            return base_prompt
    
    def review_code(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """
        Review code and return structured feedback.
        
        Args:
            code: Code to review
            context: Additional context
            
        Returns:
            ReviewResult with score and feedback
        """
        ctx = context or {}
        ctx["code_to_review"] = code
        
        output = self.execute(
            prompt=f"Review this code:\n\n```\n{code}\n```",
            role=AgentRole.VERIFIER,
            context=ctx,
        )
        
        # Parse output (simplified)
        content = output.content
        score = 75.0  # Default score
        
        # Try to extract score
        if "score:" in content.lower():
            try:
                import re
                match = re.search(r'score[:\s]*(\d+)', content.lower())
                if match:
                    score = float(match.group(1))
            except:
                pass
        
        is_valid = "VALID" in content.upper() and "INVALID" not in content.upper()
        
        return ReviewResult(
            score=score,
            is_valid=is_valid,
            summary=content[:500],
        )


def create_reviewer_agent(config: Optional[Dict] = None) -> ReviewerAgent:
    """
    Factory to create a reviewer agent.
    
    Args:
        config: Custom configuration (optional)
        
    Returns:
        Configured ReviewerAgent instance
    """
    if config is None:
        full_config = get_agent_config()
        config = full_config.get("agents", {}).get("reviewer", {})
    
    return ReviewerAgent(name="reviewer", config=config)
