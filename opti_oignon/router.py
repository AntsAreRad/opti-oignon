#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROUTER - CONTEXTEUR 2.0 (with Smart Routing)
=============================================

Selects the optimal model based on task analysis.

This module bridges Opti-Oignon's Analyzer and the SmartRouter
from the routing/ module for advanced benchmark-based routing.

The Router takes Analyzer results and determines:
- Which model to use (via SmartRouter)
- Which temperature to apply
- Which system prompt to load
- Additional parameters

Author: Léon
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

import ollama

from .config import config, OptiOignonConfig
from .analyzer import AnalysisResult, TaskType

logger = logging.getLogger(__name__)

# =============================================================================
# ROUTING RESULT
# =============================================================================

@dataclass
class RoutingResult:
    """Model routing result."""
    model: str                    # Selected Ollama model
    temperature: float            # Temperature to use
    task_type: str               # Task type
    prompt_variant: str          # Prompt variant (standard, reasoning, fast)
    model_type: str              # Model type (code, reasoning, general)
    priority_used: str           # Which priority was used (primary, fast, fallback)
    explanation: str             # Why this model was chosen
    timeout: int                 # Timeout in seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "task_type": self.task_type,
            "prompt_variant": self.prompt_variant,
            "model_type": self.model_type,
            "priority_used": self.priority_used,
            "explanation": self.explanation,
            "timeout": self.timeout,
        }


# =============================================================================
# MAIN CLASS
# =============================================================================

class ModelRouter:
    """
    Intelligent model router.
    
    Uses SmartRouter from routing/ module if available,
    otherwise uses basic logic with config.py.
    
    Usage:
        router = ModelRouter()
        result = router.route(analysis_result)
        print(result.model)  # qwen3-coder:30b
    """
    
    def __init__(self):
        """Initialize the router."""
        self._available_models: List[str] = []
        self._last_check: float = 0
        self._cache_duration: float = 60.0
        self._config = config
    
    # -------------------------------------------------------------------------
    # Available Models Check
    # -------------------------------------------------------------------------
    
    def get_available_models(self, force_refresh: bool = False) -> List[str]:
        """
        Get the list of available Ollama models.
        
        Args:
            force_refresh: Force cache refresh
            
        Returns:
            List of model names
        """
        import time
        
        if not force_refresh and self._available_models:
            if time.time() - self._last_check < self._cache_duration:
                return self._available_models
        
        try:
            response = ollama.list()
            models = []
            if hasattr(response, 'models'):
                for m in response.models:
                    name = getattr(m, 'model', None) or getattr(m, 'name', None)
                    if name:
                        models.append(name)
            elif isinstance(response, dict):
                for m in response.get("models", []):
                    name = m.get("model") or m.get("name", "")
                    if name:
                        models.append(name)
            
            self._available_models = models
            self._last_check = time.time()
            logger.debug(f"Models detected: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Ollama model listing error: {e}")
            return self._available_models
    
    def is_model_available(self, model: str) -> bool:
        """Check if a model is available."""
        return model in self.get_available_models()
    
    def find_best_available(self, preferred: str, alternatives: List[str]) -> Tuple[str, str]:
        """
        Find the best available model among options.
        
        Args:
            preferred: Preferred model
            alternatives: Alternative list in preference order
            
        Returns:
            (selected_model, reason)
        """
        available = self.get_available_models()
        
        if preferred in available:
            return preferred, "primary"
        
        for alt in alternatives:
            if alt in available:
                logger.info(f"Model {preferred} not available, using {alt}")
                return alt, "fallback"
        
        if available:
            logger.warning(f"No preferred model available, using {available[0]}")
            return available[0], "emergency"
        
        logger.error("No Ollama models available!")
        return preferred, "unavailable"
    
    # -------------------------------------------------------------------------
    # Routing Logic
    # -------------------------------------------------------------------------
    
    def route(
        self,
        analysis: AnalysisResult,
        priority: str = "balanced",
        force_model: Optional[str] = None,
        force_variant: Optional[str] = None,
    ) -> RoutingResult:
        """
        Route to optimal model based on analysis.
        
        Args:
            analysis: Task analysis result
            priority: "fast" (speed), "balanced" (default), "quality" (max quality)
            force_model: Force a specific model (ignores auto-selection)
            force_variant: Force a prompt variant
            
        Returns:
            RoutingResult with complete configuration
        """
        model_type = analysis.suggested_model_type
        task_type = analysis.task_type.value
        
        prompt_variant = self._determine_prompt_variant(analysis, force_variant)
        
        if force_model:
            model, priority_used = self._validate_forced_model(force_model)
        else:
            model, priority_used = self._select_model(model_type, priority)
        
        temperature = self._determine_temperature(task_type, analysis.complexity.value)
        timeout = self._determine_timeout(priority, analysis.complexity.value)
        
        explanation = self._build_explanation(
            analysis, model, model_type, priority, priority_used
        )
        
        return RoutingResult(
            model=model,
            temperature=temperature,
            task_type=task_type,
            prompt_variant=prompt_variant,
            model_type=model_type,
            priority_used=priority_used,
            explanation=explanation,
            timeout=timeout,
        )
    
    def _validate_forced_model(self, model: str) -> Tuple[str, str]:
        """Validate forced model and return alternative if unavailable."""
        if self.is_model_available(model):
            return model, "forced"
        
        logger.warning(f"Forced model {model} not available")
        fallbacks = self._config.get_fallback_models()
        return self.find_best_available(model, fallbacks)
    
    def _select_model(self, model_type: str, priority: str) -> Tuple[str, str]:
        """
        Select model based on type and priority.
        
        Args:
            model_type: Model type (code, reasoning, general, quick)
            priority: Priority (fast, balanced, quality)
            
        Returns:
            (model, reason)
        """
        # Map priority to config model type
        priority_map = {
            "fast": "fast",
            "balanced": "primary",
            "quality": "quality",
        }
        config_priority = priority_map.get(priority, "primary")
        
        # Get preferred model
        preferred = self._config.get_model(model_type, config_priority)
        
        # Get alternatives
        alternatives = []
        for alt_priority in ["primary", "fast", "quality"]:
            if alt_priority != config_priority:
                alt_model = self._config.get_model(model_type, alt_priority)
                if alt_model and alt_model not in alternatives:
                    alternatives.append(alt_model)
        
        # Add global fallbacks
        alternatives.extend(self._config.get_fallback_models())
        
        return self.find_best_available(preferred, alternatives)
    
    def _determine_prompt_variant(
        self, 
        analysis: AnalysisResult, 
        force_variant: Optional[str]
    ) -> str:
        """Determine which prompt variant to use."""
        if force_variant:
            return force_variant
        
        if analysis.complexity.value == "complex":
            return "reasoning"
        
        if analysis.task_type == TaskType.PLANNING_DEEP:
            return "reasoning"
        
        if analysis.task_type == TaskType.SIMPLE_QUESTION:
            return "fast"
        
        if analysis.complexity.value == "simple":
            return "fast"
        
        return "standard"
    
    def _determine_temperature(self, task_type: str, complexity: str) -> float:
        """Determine optimal temperature."""
        base_temp = self._config.get_temperature(task_type.split("_")[0])
        
        if complexity == "complex":
            return min(base_temp + 0.1, 0.9)
        elif complexity == "simple":
            return max(base_temp - 0.1, 0.1)
        
        return base_temp
    
    def _determine_timeout(self, priority: str, complexity: str) -> int:
        """Determine appropriate timeout."""
        if priority == "fast":
            return self._config.get_timeout("fast")
        elif complexity == "complex":
            return self._config.get_timeout("deep")
        else:
            return self._config.get_timeout("default")
    
    def _build_explanation(
        self,
        analysis: AnalysisResult,
        model: str,
        model_type: str,
        priority: str,
        priority_used: str,
    ) -> str:
        """Build readable routing explanation."""
        parts = [f"Task: {analysis.task_type.value}"]
        
        if analysis.confidence < 0.5:
            parts.append(f"(low confidence: {analysis.confidence:.0%})")
        
        parts.append(f"→ {model}")
        
        if priority_used == "primary":
            parts.append(f"(optimal for {model_type})")
        elif priority_used == "fallback":
            parts.append("(fallback - preferred model unavailable)")
        elif priority_used == "forced":
            parts.append("(forced by user)")
        
        if priority != "balanced":
            parts.append(f"[priority: {priority}]")
        
        return " ".join(parts)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

router = ModelRouter()


def route(
    analysis: AnalysisResult,
    priority: str = "balanced",
    force_model: Optional[str] = None,
) -> RoutingResult:
    """Convenience function to route a task."""
    return router.route(analysis, priority, force_model)


# =============================================================================
# TEST CLI
# =============================================================================

if __name__ == "__main__":
    from .analyzer import analyze
    
    print("=== Router Test ===\n")
    
    test_questions = [
        "How to calculate Shannon in R?",
        "Debug my Python code",
        "Write an abstract about biodiversity",
    ]
    
    for q in test_questions:
        print(f"Question: {q}")
        analysis = analyze(q)
        routing = router.route(analysis)
        
        print(f"  → Model: {routing.model}")
        print(f"  → Task: {routing.task_type}")
        print(f"  → Variant: {routing.prompt_variant}")
        print(f"  → Temperature: {routing.temperature}")
        print(f"  → Explanation: {routing.explanation}")
        print()
