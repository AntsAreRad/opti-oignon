#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point for running the package directly with python -m.

This allows both:
    python -m opti_oignon              # Launches UI by default
    python -m opti_oignon ui           # Explicit UI launch
    python -m opti_oignon benchmark    # Run benchmark
    python -m opti_oignon --help       # Show help

Usage Examples:
    python -m opti_oignon
    python -m opti_oignon --port 8080
    python -m opti_oignon benchmark --quick --confirm
    python -m opti_oignon rag index ./docs
"""

from .main import main

if __name__ == "__main__":
    main()
