#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTEXT MANAGER - OPTI-OIGNON 1.1
=================================

Intelligent context management for LLM interactions.

This module handles:
- Dynamic model limit fetching from Ollama
- Token estimation
- Context validation before execution
- Smart truncation
- Warning generation for UI

Integrates with existing config.py and models.yaml structure.

VERIFIED SOURCES (December 2025):
- Qwen3-Coder-30B: 262K (HuggingFace, Ollama library)
- DeepSeek-R1-Distill-Qwen-32B: 128K (DeepSeek GitHub)
- Gemma 3 27B: 128K (Google AI docs)
- Nemotron 3 Nano 30B: 1M (NVIDIA research)

Author: Léon
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import subprocess
import re
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelLimits:
    """Context limits for a specific model."""
    context_window: int  # Total context capacity in tokens
    max_output: int  # Maximum output tokens
    recommended_input: int  # Recommended input size for best performance
    chars_per_token: float  # Approximation for token estimation
    display_name: str = ""  # Human-readable name
    source: str = "default"  # Where this info came from (ollama, config, verified, default)
    
    @property
    def safe_input(self) -> int:
        """Input size that leaves room for output."""
        return self.context_window - self.max_output


@dataclass
class ContextCheck:
    """Result of context validation."""
    total_tokens: int
    prompt_tokens: int
    document_tokens: int
    system_tokens: int
    context_limit: int
    output_reserve: int
    available_for_input: int
    recommended_input: int
    exceeds_limit: bool
    exceeds_recommended: bool
    truncation_needed: bool
    suggested_truncation: int  # Tokens to remove
    usage_percent: float
    warning_message: Optional[str] = None
    model: str = ""
    
    @property
    def is_safe(self) -> bool:
        """Whether the context is safe to use."""
        return not self.exceeds_limit
    
    @property
    def status_emoji(self) -> str:
        """Get status emoji based on usage."""
        if self.exceeds_limit:
            return "🔴"
        elif self.exceeds_recommended:
            return "🟡"
        elif self.usage_percent > 75:
            return "🟠"
        elif self.usage_percent > 50:
            return "🟢"
        return "🟢"
    
    @property
    def status_class(self) -> str:
        """Get CSS class for styling."""
        if self.exceeds_limit:
            return "danger"
        elif self.exceeds_recommended or self.usage_percent > 75:
            return "warning"
        return "safe"


# =============================================================================
# DEFAULT MODEL CONFIGURATIONS (VERIFIED)
# =============================================================================

DEFAULT_MODEL_LIMITS: Dict[str, ModelLimits] = {
    # Source: HuggingFace Qwen/Qwen3-Coder-30B-A3B-Instruct, Ollama library
    "qwen3-coder:30b": ModelLimits(
        context_window=262144,
        max_output=8192,
        recommended_input=240000,
        chars_per_token=3.8,
        display_name="Qwen3 Coder 30B",
        source="verified"
    ),
    
    # Source: DeepSeek-AI GitHub, HuggingFace
    "deepseek-r1:32b": ModelLimits(
        context_window=131072,
        max_output=16384,
        recommended_input=110000,
        chars_per_token=3.5,
        display_name="DeepSeek R1 32B",
        source="verified"
    ),
    
    # Source: Google AI docs, HuggingFace
    "gemma3:27b": ModelLimits(
        context_window=131072,
        max_output=8192,
        recommended_input=120000,
        chars_per_token=4.0,
        display_name="Gemma 3 27B",
        source="verified"
    ),
    
    # Source: Meta official documentation
    "llama3.3:latest": ModelLimits(
        context_window=131072,
        max_output=16384,
        recommended_input=110000,
        chars_per_token=3.8,
        display_name="Llama 3.3 70B",
        source="verified"
    ),
    
    # Source: Mistral AI documentation
    "mistral-small3.2:latest": ModelLimits(
        context_window=131072,
        max_output=16384,
        recommended_input=110000,
        chars_per_token=3.5,
        display_name="Mistral Small 3.2 24B",
        source="verified"
    ),
    
    # Source: NVIDIA research, HuggingFace
    "nemotron-3-nano:30b": ModelLimits(
        context_window=1048576,
        max_output=16384,
        recommended_input=1000000,
        chars_per_token=3.2,
        display_name="Nemotron 3 Nano 30B",
        source="verified"
    ),
    
    # Source: QwenLM GitHub
    "qwen3:32b": ModelLimits(
        context_window=131072,
        max_output=8192,
        recommended_input=120000,
        chars_per_token=3.5,
        display_name="Qwen3 32B",
        source="verified"
    ),
    
    # Source: Mistral AI documentation
    "devstral-small-2:latest": ModelLimits(
        context_window=262144,
        max_output=16384,
        recommended_input=240000,
        chars_per_token=3.5,
        display_name="Devstral Small 2 24B",
        source="verified"
    ),
    
    # Source: QwenLM documentation
    "qwen2.5-coder:7b": ModelLimits(
        context_window=131072,
        max_output=8192,
        recommended_input=120000,
        chars_per_token=4.0,
        display_name="Qwen2.5 Coder 7B",
        source="verified"
    ),
    
    "qwen2.5-coder:14b": ModelLimits(
        context_window=32768,
        max_output=8192,
        recommended_input=24000,
        chars_per_token=3.8,
        display_name="Qwen2.5 Coder 14B",
        source="verified"
    ),
    
    "qwen2.5-coder:32b-instruct-q5_k_m": ModelLimits(
        context_window=131072,
        max_output=8192,
        recommended_input=120000,
        chars_per_token=3.5,
        display_name="Qwen2.5 Coder 32B Q5_K_M",
        source="verified"
    ),
    
    "wizard-math:13b": ModelLimits(
        context_window=32768,
        max_output=4096,
        recommended_input=28000,
        chars_per_token=4.0,
        display_name="Wizard Math 13B",
        source="verified"
    ),
    
    "dolphin-mixtral:8x7b": ModelLimits(
        context_window=16384,
        max_output=8192,
        recommended_input=12000,
        chars_per_token=4.0,
        display_name="Dolphin Mixtral 8x7B",
        source="verified"
    ),
    
    "qwen3-vl:32b": ModelLimits(
        context_window=1048576,
        max_output=32768,
        recommended_input=1000000,
        chars_per_token=3.2,
        display_name="Qwen3 VL 32B",
        source="verified"
    ),
    
    "goekdenizguelmez/JOSIEFIED-Qwen3:30b": ModelLimits(
        context_window=32768,
        max_output=8192,
        recommended_input=24000,
        chars_per_token=4.0,
        display_name="JOSIEFIED Qwen3 30B",
        source="verified"
    ),
    
    # Fallback for unknown models
    "_default": ModelLimits(
        context_window=8192,
        max_output=4096,
        recommended_input=4000,
        chars_per_token=4.0,
        display_name="Unknown Model",
        source="default"
    ),
}


# =============================================================================
# CONTEXT MANAGER CLASS
# =============================================================================

class ContextManager:
    """
    Intelligent context management for LLMs.
    
    Handles model limit detection, token estimation, and context validation.
    Prioritizes dynamic Ollama information when available.
    
    Usage:
        manager = ContextManager()
        check = manager.check_context(
            prompt="Your question",
            document="Optional document",
            system_prompt="System prompt",
            model="qwen3-coder:30b"
        )
        if check.exceeds_limit:
            truncated, _ = manager.smart_truncate(document, check.suggested_truncation)
    """
    
    def __init__(self, models_config: Optional[Dict] = None):
        """
        Initialize the context manager.
        
        Args:
            models_config: Optional dict from models.yaml for additional limits
        """
        self._cache: Dict[str, ModelLimits] = {}
        self._ollama_cache: Dict[str, Dict] = {}
        self._config_limits: Dict[str, Dict] = models_config or {}
    
    @classmethod
    def from_config(cls) -> 'ContextManager':
        """
        Create a ContextManager using the existing config.py infrastructure.
        
        Returns:
            Configured ContextManager instance
        """
        try:
            from .config import config
            models_data = config._models_config.get("models", {})
            return cls(models_config=models_data)
        except ImportError:
            logger.debug("Could not import config, using defaults only")
            return cls()
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
            return cls()
    
    # -------------------------------------------------------------------------
    # Ollama Integration
    # -------------------------------------------------------------------------
    
    def _fetch_ollama_info(self, model: str) -> Optional[Dict]:
        """
        Fetch model information from Ollama.
        
        Args:
            model: Model name (e.g., "qwen3-coder:30b")
            
        Returns:
            Dictionary with model info, or None if unavailable
        """
        if model in self._ollama_cache:
            return self._ollama_cache[model]
        
        try:
            result = subprocess.run(
                ["ollama", "show", model],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.debug(f"Ollama show failed for {model}: {result.stderr}")
                return None
            
            output = result.stdout
            info = self._parse_ollama_output(output)
            self._ollama_cache[model] = info
            return info
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Ollama show timed out for {model}")
        except FileNotFoundError:
            logger.debug("Ollama not found in PATH")
        except Exception as e:
            logger.debug(f"Error fetching Ollama info for {model}: {e}")
        
        return None
    
    def _parse_ollama_output(self, output: str) -> Dict:
        """
        Parse the output of 'ollama show' command.
        
        Args:
            output: Raw output from ollama show
            
        Returns:
            Dictionary with parsed model info
        """
        info = {}
        
        # Extract context length
        context_match = re.search(r'context length\s+(\d+)', output, re.IGNORECASE)
        if context_match:
            info['context_length'] = int(context_match.group(1))
        
        # Extract parameters count
        params_match = re.search(r'parameters\s+([\d.]+)([BMK])', output, re.IGNORECASE)
        if params_match:
            value = float(params_match.group(1))
            unit = params_match.group(2).upper()
            multiplier = {'K': 1e3, 'M': 1e6, 'B': 1e9}.get(unit, 1)
            info['parameters'] = int(value * multiplier)
        
        # Extract architecture
        arch_match = re.search(r'architecture\s+(\w+)', output, re.IGNORECASE)
        if arch_match:
            info['architecture'] = arch_match.group(1)
        
        # Extract quantization
        quant_match = re.search(r'quantization\s+(\S+)', output, re.IGNORECASE)
        if quant_match:
            info['quantization'] = quant_match.group(1)
        
        return info
    
    # -------------------------------------------------------------------------
    # Model Limits
    # -------------------------------------------------------------------------
    
    def get_model_limits(self, model: str) -> ModelLimits:
        """
        Get context limits for a model.
        
        Priority:
        1. Cached result
        2. Config file limits (models.yaml) - user's custom config
        3. Built-in verified defaults
        4. Dynamic Ollama info (for unknown models)
        5. Universal fallback
        
        Args:
            model: Model name
            
        Returns:
            ModelLimits dataclass
        """
        # Check cache first
        if model in self._cache:
            return self._cache[model]
        
        limits = None
        
        # Try config file first (models.yaml - user's config takes priority)
        if model in self._config_limits:
            cfg = self._config_limits[model]
            if isinstance(cfg, dict) and 'context_window' in cfg:
                limits = ModelLimits(
                    context_window=cfg.get("context_window", 8192),
                    max_output=cfg.get("max_output", 4096),
                    recommended_input=cfg.get("recommended_input", 4000),
                    chars_per_token=cfg.get("chars_per_token", 4.0),
                    display_name=cfg.get("display_name", model),
                    source="config"
                )
                logger.debug(f"Got limits for {model} from config")
        
        # Try built-in verified defaults
        if limits is None and model in DEFAULT_MODEL_LIMITS:
            limits = DEFAULT_MODEL_LIMITS[model]
        
        # Try Ollama for unknown models
        if limits is None:
            ollama_info = self._fetch_ollama_info(model)
            if ollama_info and 'context_length' in ollama_info:
                context_window = ollama_info['context_length']
                max_output = min(context_window // 4, 32768)
                recommended_input = context_window - max_output - 1000
                
                limits = ModelLimits(
                    context_window=context_window,
                    max_output=max_output,
                    recommended_input=recommended_input,
                    chars_per_token=3.5,
                    display_name=model,
                    source="ollama"
                )
                logger.info(f"Got limits for {model} from Ollama: {context_window} context")
        
        # Try to find a similar model (same base name)
        if limits is None:
            base_name = model.split(":")[0] if ":" in model else model
            for key, default_limits in DEFAULT_MODEL_LIMITS.items():
                if key.startswith(base_name):
                    limits = default_limits
                    logger.debug(f"Using similar model limits from {key} for {model}")
                    break
        
        # Ultimate fallback
        if limits is None:
            limits = DEFAULT_MODEL_LIMITS["_default"]
            logger.warning(f"Using default limits for unknown model: {model}")
        
        # Cache the result
        self._cache[model] = limits
        return limits
    
    # -------------------------------------------------------------------------
    # Token Estimation
    # -------------------------------------------------------------------------
    
    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate the number of tokens in text.
        
        Uses character-to-token ratio based on model type.
        
        Args:
            text: Text to estimate
            model: Optional model for more accurate estimation
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        chars_per_token = 4.0
        
        if model:
            limits = self.get_model_limits(model)
            chars_per_token = limits.chars_per_token
        
        # Basic estimation
        estimated = len(text) / chars_per_token
        
        # Adjust for code (tends to have more tokens per char)
        code_indicators = ['def ', 'function(', 'class ', 'import ', 'library(', '<-', '```']
        if any(ind in text for ind in code_indicators):
            estimated *= 1.15
        
        return int(estimated)
    
    # -------------------------------------------------------------------------
    # Context Checking
    # -------------------------------------------------------------------------
    
    def check_context(
        self,
        prompt: str,
        document: str = "",
        system_prompt: str = "",
        model: str = ""
    ) -> ContextCheck:
        """
        Check if the context respects model limits.
        
        Args:
            prompt: User's question/prompt
            document: Optional document/code content
            system_prompt: System prompt being used
            model: Model to check against
            
        Returns:
            ContextCheck with detailed validation results
        """
        limits = self.get_model_limits(model)
        
        # Estimate tokens for each part
        prompt_tokens = self.estimate_tokens(prompt, model)
        document_tokens = self.estimate_tokens(document, model)
        system_tokens = self.estimate_tokens(system_prompt, model)
        
        total_tokens = prompt_tokens + document_tokens + system_tokens
        
        # Calculate available space
        available_for_input = limits.context_window - limits.max_output
        
        # Check limits
        exceeds_limit = total_tokens > available_for_input
        exceeds_recommended = total_tokens > limits.recommended_input
        
        # Calculate usage
        usage_percent = (total_tokens / available_for_input) * 100 if available_for_input > 0 else 100
        
        # Calculate truncation needs
        truncation_needed = exceeds_limit
        suggested_truncation = max(0, total_tokens - available_for_input + 500)
        
        # Generate warning message
        warning_message = None
        if exceeds_limit:
            warning_message = (
                f"Context exceeds limit! ~{total_tokens:,} tokens > "
                f"{available_for_input:,} available. "
                f"Need to remove ~{suggested_truncation:,} tokens."
            )
        elif exceeds_recommended:
            overage = total_tokens - limits.recommended_input
            warning_message = (
                f"Context exceeds recommended size by ~{overage:,} tokens. "
                f"Performance may be degraded."
            )
        elif usage_percent > 75:
            warning_message = (
                f"Context is {usage_percent:.0f}% full. "
                f"Consider summarizing for best results."
            )
        
        return ContextCheck(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            document_tokens=document_tokens,
            system_tokens=system_tokens,
            context_limit=limits.context_window,
            output_reserve=limits.max_output,
            available_for_input=available_for_input,
            recommended_input=limits.recommended_input,
            exceeds_limit=exceeds_limit,
            exceeds_recommended=exceeds_recommended,
            truncation_needed=truncation_needed,
            suggested_truncation=suggested_truncation,
            usage_percent=usage_percent,
            warning_message=warning_message,
            model=model
        )
    
    # -------------------------------------------------------------------------
    # Smart Truncation
    # -------------------------------------------------------------------------
    
    def smart_truncate(
        self,
        text: str,
        max_tokens: int,
        model: Optional[str] = None,
        preserve_start: int = 500,
        preserve_end: int = 200
    ) -> Tuple[str, int]:
        """
        Intelligently truncate text while preserving important parts.
        
        Keeps the beginning (context/setup) and end (recent/relevant)
        while removing content from the middle.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens to allow
            model: Model for token estimation
            preserve_start: Tokens to preserve at start
            preserve_end: Tokens to preserve at end
            
        Returns:
            Tuple of (truncated_text, tokens_removed)
        """
        current_tokens = self.estimate_tokens(text, model)
        
        if current_tokens <= max_tokens:
            return text, 0
        
        limits = self.get_model_limits(model) if model else DEFAULT_MODEL_LIMITS["_default"]
        chars_per_token = limits.chars_per_token
        
        # Calculate character positions
        start_chars = int(preserve_start * chars_per_token)
        end_chars = int(preserve_end * chars_per_token)
        
        # Ensure we don't exceed text length
        if start_chars + end_chars >= len(text):
            target_chars = int(max_tokens * chars_per_token)
            truncated = text[:target_chars]
            tokens_removed = current_tokens - self.estimate_tokens(truncated, model)
            return truncated + "\n\n[... truncated ...]", tokens_removed
        
        # Calculate middle section
        target_middle_tokens = max_tokens - preserve_start - preserve_end
        target_middle_chars = max(0, int(target_middle_tokens * chars_per_token))
        
        start_part = text[:start_chars]
        end_part = text[-end_chars:]
        
        middle_start = start_chars
        middle_end = len(text) - end_chars
        original_middle = text[middle_start:middle_end]
        
        if target_middle_chars > 0 and len(original_middle) > target_middle_chars:
            truncated_middle = original_middle[:target_middle_chars]
            marker = "\n\n[... content truncated to fit context window ...]\n\n"
        else:
            truncated_middle = ""
            marker = "\n\n[... middle section removed to fit context window ...]\n\n"
        
        truncated = start_part + truncated_middle + marker + end_part
        tokens_removed = current_tokens - self.estimate_tokens(truncated, model)
        
        return truncated, tokens_removed
    
    # -------------------------------------------------------------------------
    # Action Suggestions
    # -------------------------------------------------------------------------
    
    def suggest_action(self, check: ContextCheck) -> List[str]:
        """
        Suggest actions based on context check results.
        
        Args:
            check: ContextCheck result
            
        Returns:
            List of suggested actions
        """
        suggestions = []
        
        if check.exceeds_limit:
            suggestions.append("⚠️ Document must be truncated to proceed")
            suggestions.append("💡 Consider summarizing the document first")
            suggestions.append("🔄 Try a model with larger context window")
            
            if check.document_tokens > 100000:
                suggestions.append("📊 For very large docs: nemotron-3-nano:30b (1M context)")
            elif check.document_tokens > 50000:
                suggestions.append("📊 For large docs: qwen3-coder:30b (256K context)")
        
        elif check.exceeds_recommended:
            suggestions.append("⚡ Response quality may be reduced")
            suggestions.append("💡 Consider splitting into smaller queries")
        
        elif check.usage_percent > 75:
            suggestions.append("📊 Context is getting full")
            suggestions.append("💡 Leave room for detailed responses")
        
        return suggestions
    
    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------
    
    def format_context_display(self, check: ContextCheck) -> str:
        """
        Format context check for UI display as Markdown.
        
        Args:
            check: ContextCheck result
            
        Returns:
            Formatted Markdown string
        """
        # Create progress bar
        bar_length = 20
        filled = int(bar_length * min(check.usage_percent, 100) / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        lines = [
            f"**{check.status_emoji} Context Usage** • `{check.model}`",
            "",
            f"Input: ~{check.total_tokens:,} / {check.available_for_input:,} tokens",
            f"`[{bar}]` {check.usage_percent:.0f}%",
        ]
        
        if check.document_tokens > 100:
            lines.extend([
                "",
                f"*Prompt: ~{check.prompt_tokens:,} • "
                f"Doc: ~{check.document_tokens:,} • "
                f"System: ~{check.system_tokens:,}*"
            ])
        
        if check.warning_message:
            lines.extend(["", f"⚠️ {check.warning_message}"])
        
        if check.exceeds_limit:
            lines.extend([
                "",
                "**Suggestions:**",
                "• Summarize the document first",
                "• Enable auto-truncation",
                "• Use a larger context model"
            ])
        
        return "\n".join(lines)
    
    def format_context_status_line(self, check: ContextCheck) -> str:
        """
        Format a single-line status for UI header.
        
        Args:
            check: ContextCheck result
            
        Returns:
            Short status string
        """
        return f"{check.status_emoji} ~{check.total_tokens:,} tokens ({check.usage_percent:.0f}%)"
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._cache.clear()
        self._ollama_cache.clear()
    
    def list_available_models(self) -> List[str]:
        """List available models from Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            models = []
            for line in result.stdout.strip().split('\n')[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            
            return models
            
        except Exception as e:
            logger.debug(f"Error listing models: {e}")
            return []
    
    def get_all_limits(self) -> Dict[str, ModelLimits]:
        """Get limits for all known models."""
        all_limits = {}
        
        # Start with defaults
        for name, limits in DEFAULT_MODEL_LIMITS.items():
            if name != "_default":
                all_limits[name] = limits
        
        # Add config models
        for name in self._config_limits:
            if name not in all_limits and name != "_default":
                all_limits[name] = self.get_model_limits(name)
        
        # Add installed models from Ollama
        for model in self.list_available_models():
            if model not in all_limits:
                all_limits[model] = self.get_model_limits(model)
        
        return all_limits


# =============================================================================
# GLOBAL INSTANCE (lazy initialization)
# =============================================================================

_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get the global context manager instance (lazy initialization)."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager.from_config()
    return _context_manager


def reset_context_manager() -> None:
    """Reset the global context manager (useful for testing)."""
    global _context_manager
    _context_manager = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def check_context(
    prompt: str,
    document: str = "",
    system_prompt: str = "",
    model: str = ""
) -> ContextCheck:
    """Convenience function to check context."""
    return get_context_manager().check_context(prompt, document, system_prompt, model)


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """Convenience function to estimate tokens."""
    return get_context_manager().estimate_tokens(text, model)


def smart_truncate(
    text: str,
    max_tokens: int,
    model: Optional[str] = None,
    preserve_start: int = 500,
    preserve_end: int = 200
) -> Tuple[str, int]:
    """Convenience function for smart truncation."""
    return get_context_manager().smart_truncate(
        text, max_tokens, model, preserve_start, preserve_end
    )


def get_model_limits(model: str) -> ModelLimits:
    """Convenience function to get model limits."""
    return get_context_manager().get_model_limits(model)


def format_context_indicator(
    model: str,
    question: str = "",
    document: str = "",
    system_prompt: str = ""
) -> str:
    """
    Generate formatted context indicator for UI display.
    
    Args:
        model: Model name
        question: User's question
        document: Document content
        system_prompt: System prompt
        
    Returns:
        Markdown formatted string
    """
    if not model:
        return ""
    
    try:
        check = check_context(
            prompt=question,
            document=document,
            system_prompt=system_prompt,
            model=model
        )
        return get_context_manager().format_context_display(check)
    except Exception as e:
        logger.debug(f"Context check error: {e}")
        return ""


def get_quick_context_status(model: str, document: str) -> str:
    """
    Get a quick one-line context status.
    
    Args:
        model: Model name
        document: Document content
        
    Returns:
        Short status string
    """
    if not document or not model:
        return ""
    
    try:
        tokens = estimate_tokens(document, model)
        limits = get_model_limits(model)
        percent = (tokens / limits.recommended_input) * 100 if limits.recommended_input > 0 else 0
        
        if percent > 100:
            return f"🔴 Doc: ~{tokens:,} tokens (exceeds limit)"
        elif percent > 75:
            return f"🟡 Doc: ~{tokens:,} tokens ({percent:.0f}% of limit)"
        else:
            return f"🟢 Doc: ~{tokens:,} tokens ({percent:.0f}% of limit)"
    except Exception:
        return ""


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Context Manager Test ===\n")
    
    manager = ContextManager()
    
    # Test with common models
    test_models = [
        "qwen3-coder:30b",
        "deepseek-r1:32b",
        "gemma3:27b",
        "nemotron-3-nano:30b",
    ]
    
    print("--- Model Limits ---")
    for model in test_models:
        limits = manager.get_model_limits(model)
        print(f"\n{model}:")
        print(f"  Context: {limits.context_window:,} tokens")
        print(f"  Max Output: {limits.max_output:,} tokens")
        print(f"  Recommended Input: {limits.recommended_input:,} tokens")
        print(f"  Source: {limits.source}")
    
    # Test context check
    print("\n--- Context Check Test ---")
    test_doc = "species,count\n" + "sp1,100\n" * 1000
    
    check = manager.check_context(
        prompt="Write a function to calculate Shannon index",
        document=test_doc,
        system_prompt="You are an R expert.",
        model="qwen3-coder:30b"
    )
    
    print(f"\nTotal: ~{check.total_tokens:,} tokens")
    print(f"Usage: {check.usage_percent:.1f}%")
    print(f"Status: {check.status_emoji}")
    
    print("\n--- Formatted Display ---")
    print(manager.format_context_display(check))
    
    # Test truncation
    print("\n--- Truncation Test ---")
    long_doc = "data row " * 50000  # ~150K characters
    truncated, removed = manager.smart_truncate(
        long_doc,
        max_tokens=10000,
        model="qwen3-coder:30b"
    )
    print(f"Original: ~{len(long_doc):,} chars")
    print(f"Truncated: ~{len(truncated):,} chars")
    print(f"Tokens removed: ~{removed:,}")
