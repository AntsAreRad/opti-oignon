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
    - Conversation history and export
    - Modern dark-mode Gradio interface

Usage:
    # Start the UI
    python -m opti_oignon
    
    # Or import components
    from opti_oignon import analyzer, router, executor

Author: Léon
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Léon"
__license__ = "MIT"

# Core components
from .config import config, DATA_DIR, CONFIG_DIR
from .analyzer import analyzer, analyze, AnalysisResult
from .router import router, RoutingResult
from .executor import executor, execute, get_prompt
from .presets import preset_manager, Preset
from .history import history

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
]


def main():
    """Main entry point - launches the UI."""
    from .ui import launch
    launch()
