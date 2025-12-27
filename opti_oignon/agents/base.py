#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASE - MULTI-AGENT SYSTEM
=========================

Base classes for the multi-model orchestration system.

This module defines:
- Dataclasses for results and steps
- Base Agent class
- Common interfaces

Author: Léon
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Generator, Callable
from enum import Enum
from datetime import datetime
from pathlib import Path
import threading
import logging
import time
import json
import yaml
import ollama

# Logging configuration
logger = logging.getLogger("MultiAgent")


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AgentRole(Enum):
    """Possible roles for an agent in a pipeline."""
    GENERATOR = "generator"
    VERIFIER = "verifier"
    PROPOSER = "proposer"
    ARBITER = "arbiter"
    DECOMPOSER = "decomposer"
    SOLVER = "solver"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"
    REVISOR = "revisor"


class StepStatus(Enum):
    """Status of an execution step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class PipelineStatus(Enum):
    """Global status of a pipeline."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class AgentOutput:
    """
    Agent output after execution.
    
    Contains generated content, metadata, and metrics.
    """
    content: str                          # Generated content
    agent_name: str                        # Agent name
    model_used: str                        # Ollama model used
    role: AgentRole                        # Role played in the pipeline
    confidence: float = 0.0               # Confidence score (0-1)
    execution_time: float = 0.0           # Execution time in seconds
    token_count: int = 0                  # Number of generated tokens
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "agent_name": self.agent_name,
            "model_used": self.model_used,
            "role": self.role.value,
            "confidence": self.confidence,
            "execution_time": self.execution_time,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentOutput':
        """Create an instance from a dictionary."""
        data["role"] = AgentRole(data["role"])
        return cls(**data)


@dataclass
class StepResult:
    """
    Result of a pipeline step.
    
    Encapsulates agent output and step information.
    """
    step_name: str                         # Step name
    step_index: int                        # Position in the pipeline
    status: StepStatus                     # Step status
    output: Optional[AgentOutput] = None   # Agent output
    error: Optional[str] = None            # Error message if failed
    requires_revision: bool = False        # If the next step should revise
    user_intervention: Optional[str] = None # Possible user intervention
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "step_name": self.step_name,
            "step_index": self.step_index,
            "status": self.status.value,
            "output": self.output.to_dict() if self.output else None,
            "error": self.error,
            "requires_revision": self.requires_revision,
            "user_intervention": self.user_intervention,
        }


@dataclass
class PipelineResult:
    """
    Complete result of a pipeline.
    
    Contains all intermediate results and the final result.
    """
    pipeline_name: str
    status: PipelineStatus
    steps: List[StepResult]
    final_output: Optional[str] = None
    total_time: float = 0.0
    total_tokens: int = 0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "final_output": self.final_output,
            "total_time": self.total_time,
            "total_tokens": self.total_tokens,
            "confidence_score": self.confidence_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    def get_step(self, name: str) -> Optional[StepResult]:
        """Get a step by its name."""
        for step in self.steps:
            if step.step_name == name:
                return step
        return None
    
    @property
    def is_success(self) -> bool:
        """Check if the pipeline succeeded."""
        return self.status == PipelineStatus.COMPLETED


# =============================================================================
# BASE AGENT CLASS
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    An agent encapsulates:
    - One or more Ollama models
    - Specialties (task types)
    - Execution parameters
    - Generation logic
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        shared_state: Optional[Dict] = None,
    ):
        """
        Initialize an agent.
        
        Args:
            name: Unique agent name
            config: Agent configuration (from YAML)
            shared_state: Shared state between agents (optional)
        """
        self.name = name
        self.config = config
        self.shared_state = shared_state or {}
        
        # Extract configuration
        self.description = config.get("description", "")
        self.models = config.get("models", {})
        self.temperature = config.get("temperature", 0.5)
        self.max_tokens = config.get("max_tokens", 2048)
        self.timeout = config.get("timeout", 120)
        self.specialties = config.get("specialties", [])
        
        # Internal state
        self._cancel_event = threading.Event()
        self._current_role: Optional[AgentRole] = None
        
        logger.debug(f"Agent '{name}' initialized with primary model: {self.models.get('primary')}")
    
    # -------------------------------------------------------------------------
    # Abstract methods (to be implemented by subclasses)
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def get_system_prompt(self, role: AgentRole, context: Dict[str, Any]) -> str:
        """
        Generate system prompt adapted to role and context.
        
        Args:
            role: Agent role in the pipeline
            context: Execution context
            
        Returns:
            Formatted system prompt
        """
        pass
    
    # -------------------------------------------------------------------------
    # Model selection methods
    # -------------------------------------------------------------------------
    
    def get_model(self, priority: str = "primary") -> str:
        """
        Get the model according to priority.
        
        Args:
            priority: "primary", "quality", or "fast"
            
        Returns:
            Ollama model name
        """
        model = self.models.get(priority)
        if model:
            return model
        # Fallback to primary
        return self.models.get("primary", "qwen3-coder:30b")
    
    def is_available(self, model: str) -> bool:
        """Check if a model is available on Ollama."""
        try:
            models_list = ollama.list()
            available = [m.get("name", m.get("model", "")) for m in models_list.get("models", [])]
            # Normalize names (remove :latest if present)
            available_normalized = []
            for m in available:
                available_normalized.append(m)
                if ":" in m:
                    available_normalized.append(m.split(":")[0])
            return model in available_normalized or model.split(":")[0] in available_normalized
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    def select_available_model(self) -> str:
        """
        Select the first available model.
        
        Iterates through models in priority order and returns
        the first one available on Ollama.
        
        Returns:
            Available model name
        """
        for priority in ["primary", "quality", "fast"]:
            model = self.models.get(priority)
            if model and self.is_available(model):
                return model
        
        # No configured model available, use fallback
        fallback = "qwen3-coder:30b"
        logger.warning(f"Agent '{self.name}': No configured model available, using {fallback}")
        return fallback
    
    # -------------------------------------------------------------------------
    # Execution methods
    # -------------------------------------------------------------------------
    
    def execute(
        self,
        prompt: str,
        role: AgentRole,
        context: Dict[str, Any],
        model_priority: str = "primary",
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> AgentOutput:
        """
        Execute the agent on a given prompt.
        
        Args:
            prompt: User prompt
            role: Agent role in the pipeline
            context: Execution context (previous outputs, etc.)
            model_priority: Model priority ("primary", "quality", "fast")
            stream: If True, stream tokens
            on_token: Callback called for each token (if stream=True)
            
        Returns:
            AgentOutput with the result
        """
        self._cancel_event.clear()
        self._current_role = role
        
        # Select model
        model = self.get_model(model_priority)
        if not self.is_available(model):
            model = self.select_available_model()
        
        # Generate system prompt
        system_prompt = self.get_system_prompt(role, context)
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Execute
        start_time = time.time()
        content = ""
        token_count = 0
        
        try:
            if stream and on_token:
                # Streaming mode
                response_stream = ollama.chat(
                    model=model,
                    messages=messages,
                    options={"temperature": self.temperature},
                    stream=True,
                )
                
                for chunk in response_stream:
                    if self._cancel_event.is_set():
                        content += "\n[Cancelled]"
                        break
                    
                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]
                        content += token
                        token_count += 1
                        on_token(token)
                    
                    # Check timeout
                    if time.time() - start_time > self.timeout:
                        content += "\n[Timeout]"
                        break
            else:
                # Non-streaming mode
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    options={"temperature": self.temperature},
                )
                content = response["message"]["content"]
                token_count = len(content.split())  # Approximation
                
        except Exception as e:
            logger.error(f"Agent '{self.name}' error: {e}")
            content = f"Error: {str(e)}"
        
        execution_time = time.time() - start_time
        
        # Calculate confidence (simple heuristic)
        confidence = self._estimate_confidence(content)
        
        self._current_role = None
        
        return AgentOutput(
            content=content,
            agent_name=self.name,
            model_used=model,
            role=role,
            confidence=confidence,
            execution_time=execution_time,
            token_count=token_count,
            metadata={
                "temperature": self.temperature,
                "system_prompt_length": len(system_prompt),
            }
        )
    
    def execute_streaming(
        self,
        prompt: str,
        role: AgentRole,
        context: Dict[str, Any],
        model_priority: str = "primary",
    ) -> Generator[str, None, AgentOutput]:
        """
        Execute the agent with streaming and return a generator.
        
        Yields:
            Tokens as they come
            
        Returns:
            Final AgentOutput (accessible via .value after generator exhaustion)
        """
        self._cancel_event.clear()
        self._current_role = role
        
        model = self.get_model(model_priority)
        if not self.is_available(model):
            model = self.select_available_model()
        
        system_prompt = self.get_system_prompt(role, context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        start_time = time.time()
        content = ""
        token_count = 0
        
        try:
            response_stream = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": self.temperature},
                stream=True,
            )
            
            for chunk in response_stream:
                if self._cancel_event.is_set():
                    content += "\n[Cancelled]"
                    yield "\n[Cancelled]"
                    break
                
                if "message" in chunk and "content" in chunk["message"]:
                    token = chunk["message"]["content"]
                    content += token
                    token_count += 1
                    yield token
                
                if time.time() - start_time > self.timeout:
                    content += "\n[Timeout]"
                    yield "\n[Timeout]"
                    break
                    
        except Exception as e:
            error_msg = f"\nError: {str(e)}"
            content += error_msg
            yield error_msg
        
        execution_time = time.time() - start_time
        confidence = self._estimate_confidence(content)
        self._current_role = None
        
        return AgentOutput(
            content=content,
            agent_name=self.name,
            model_used=model,
            role=role,
            confidence=confidence,
            execution_time=execution_time,
            token_count=token_count,
            metadata={"temperature": self.temperature}
        )
    
    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    
    def _estimate_confidence(self, content: str) -> float:
        """
        Estimate a confidence score for the output.
        
        Heuristic based on:
        - Response length
        - Presence of code (if expected)
        - Absence of error messages
        
        Args:
            content: Generated content
            
        Returns:
            Score between 0 and 1
        """
        score = 0.5  # Base score
        
        # Penalty for too short responses
        if len(content) < 50:
            score -= 0.2
        elif len(content) > 200:
            score += 0.1
        
        # Penalty for errors
        error_indicators = ["error", "impossible", "i cannot", "sorry", "erreur"]
        for indicator in error_indicators:
            if indicator.lower() in content.lower():
                score -= 0.15
        
        # Bonus for structured code
        if "```" in content:
            score += 0.15
        
        # Bonus for clear structure
        if any(marker in content for marker in ["###", "**", "1.", "- "]):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def cancel(self) -> None:
        """Cancel the current execution."""
        self._cancel_event.set()
        logger.info(f"Agent '{self.name}': Cancellation requested")
    
    def is_cancelled(self) -> bool:
        """Check if cancellation is in progress."""
        return self._cancel_event.is_set()
    
    def can_handle(self, task_type: str) -> bool:
        """Check if the agent can handle this task type."""
        return task_type in self.specialties
    
    def __repr__(self) -> str:
        return f"<Agent '{self.name}' model={self.get_model()} specialties={self.specialties}>"


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_agent_config(config_path: Optional[str] = None) -> Dict:
    """
    Load agent configuration from YAML file.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Agent config not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error loading agent config: {e}")
        return {}


# Global config instance
_agent_config: Optional[Dict] = None


def get_agent_config() -> Dict:
    """Return the global agent configuration (singleton)."""
    global _agent_config
    if _agent_config is None:
        _agent_config = load_agent_config()
    return _agent_config


def is_multi_agent_enabled() -> bool:
    """
    Check if multi-agent is enabled.
    
    IMPORTANT: This function allows enabling/disabling the system.
    
    Returns:
        True if multi-agent is enabled
    """
    config = get_agent_config()
    return config.get("global", {}).get("enabled", False)


def set_multi_agent_enabled(enabled: bool) -> bool:
    """
    Enable or disable the multi-agent system.
    
    Args:
        enabled: True to enable, False to disable
        
    Returns:
        True if modification succeeded
    """
    global _agent_config
    config_path = Path(__file__).parent / "config.yaml"
    
    try:
        # Reload config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        # Modify
        if "global" not in config:
            config["global"] = {}
        config["global"]["enabled"] = enabled
        
        # Save
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # Update cache
        _agent_config = config
        
        logger.info(f"Multi-agent {'enabled' if enabled else 'disabled'}")
        return True
        
    except Exception as e:
        logger.error(f"Error modifying config: {e}")
        return False


# =============================================================================
# CLI FOR TESTS
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== Multi-agent base module test ===\n")
    
    # Load config
    config = get_agent_config()
    print(f"Configuration loaded: {len(config)} sections")
    print(f"Multi-agent enabled: {is_multi_agent_enabled()}")
    
    # List configured agents
    agents = config.get("agents", {})
    print(f"\nConfigured agents: {len(agents)}")
    for name, cfg in agents.items():
        print(f"   • {name}: {cfg.get('description', 'N/A')}")
        print(f"     Primary model: {cfg.get('models', {}).get('primary', '?')}")
    
    # List pipelines
    pipelines = config.get("pipelines", {})
    print(f"\nConfigured pipelines: {len(pipelines)}")
    for name, cfg in pipelines.items():
        print(f"   • {name}: {cfg.get('name', name)}")
    
    print("\n[OK] Base module functional")
