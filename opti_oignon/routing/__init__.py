#!/usr/bin/env python3
"""Routing package.

The active per-step selection logic (``SmartRouter``) lives in
``opti_oignon.smart_router``; the analyzer-driven dispatcher (``ModelRouter``)
lives in ``opti_oignon.router``. This package provides the offline model
benchmark used to populate model profiles and routing config: see the
``benchmark`` submodule (``opti_oignon.routing.benchmark.ModelBenchmark``),
imported directly by callers.

Author: Léon
"""

# This package intentionally re-exports nothing. SmartRouter and the dispatcher
# live at the top level (opti_oignon.smart_router / opti_oignon.router), and the
# benchmark is imported from the `benchmark` submodule directly; the previous
# __all__ advertised names ("SmartRouter", "router") that were never bound here,
# so `from opti_oignon.routing import SmartRouter` would have failed.
__all__: list[str] = []
