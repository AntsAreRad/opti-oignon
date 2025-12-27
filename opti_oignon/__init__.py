#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTI-OIGNON - Local LLM Optimization Suite
==========================================

A comprehensive framework for optimizing local LLMs running on Ollama,
featuring intelligent routing, RAG, and multi-agent orchestration.

Features:
    - Intelligent task detection and model routing
    - Customizable system prompts per task type
    - RAG (Retrieval-Augmented Generation) integration
    - Multi-agent pipeline orchestration
    - Dynamic pipeline planning
    - Custom pipeline management
    - Conversation history and export
    - Modern dark-mode Gradio interface

Usage:
    # Start the UI
    python -m opti_oignon
    
    # Or import components
    from opti_oignon import analyzer, router, executor

Author: Léon
Version: 1.2.10
License: MIT
"""

__version__ = "1.2.0"
__author__ = "Léon"
__license__ = "MIT"

# Core components
from .config import config, DATA_DIR, CONFIG_DIR
from .analyzer import analyzer, analyze, AnalysisResult
from .router import router, RoutingResult
from .executor import executor, execute, get_prompt
from .presets import preset_manager, Preset
from .history import history

# Pipeline Manager (new in 1.2.0)
try:
    from .pipeline_manager import (
        get_pipeline_manager,
        Pipeline,
        PipelineStep,
    )
    PIPELINE_MANAGER_AVAILABLE = True
except ImportError:
    PIPELINE_MANAGER_AVAILABLE = False
    get_pipeline_manager = None
    Pipeline = None
    PipelineStep = None

# Dynamic Pipeline (optionnel)
try:
    from .dynamic_pipeline_ui import (
        should_use_dynamic_pipeline,
        process_with_dynamic_pipeline,
        get_dynamic_pipeline_status,
        format_status_for_ui,
        DYNAMIC_PIPELINE_AVAILABLE,
    )
except ImportError:
    DYNAMIC_PIPELINE_AVAILABLE = False
    should_use_dynamic_pipeline = None
    process_with_dynamic_pipeline = None

# Convenience exports
__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Configuration
    "config",
    "DATA_DIR",
    "CONFIG_DIR",
    
    # Core components
    "analyzer",
    "analyze",
    "AnalysisResult",
    "router",
    "RoutingResult",
    "executor",
    "execute",
    "get_prompt",
    "preset_manager",
    "Preset",
    "history",
    
    # Pipeline Manager
    "PIPELINE_MANAGER_AVAILABLE",
    "get_pipeline_manager",
    "Pipeline",
    "PipelineStep",
    
    # Dynamic Pipeline
    "DYNAMIC_PIPELINE_AVAILABLE",
    "should_use_dynamic_pipeline",
    "process_with_dynamic_pipeline",
]


def main():
    """Main entry point - launches the UI."""
    from .ui import launch
    launch()
