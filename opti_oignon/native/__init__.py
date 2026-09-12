#!/usr/bin/env python3
"""The native core's loader: asked at the call, never at import.

``oo_core`` is a Rust extension built by ``scripts/build_oo_core.sh`` into
this directory and never tracked. The Python modules that can use it ask
:func:`load` when they need it; a missing artefact is ``None`` and the
reference path runs. Importing this package loads nothing: the artefact is
found and loaded inside :func:`load`, once, and cached.
"""

import importlib.util
import logging
from pathlib import Path

checkpoint_before_apply = True

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_UNSET = object()
_loaded = _UNSET


def _artefact():
    for candidate in sorted(_HERE.glob("oo_core*.so")):
        return candidate
    return None


def load():
    """The native module, or None when it is not built. Loaded once."""
    global _loaded
    if _loaded is not _UNSET:
        return _loaded
    path = _artefact()
    if path is None:
        _loaded = None
        return None
    try:
        spec = importlib.util.spec_from_file_location("opti_oignon.native.oo_core", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _loaded = module
    except Exception as exc:  # noqa: BLE001 - a broken artefact is absence, said once
        logger.warning("oo_core present but not loadable (%s); the reference path runs", exc)
        _loaded = None
    return _loaded


def available():
    return load() is not None


def reset():
    """Forget the cached artefact, for the contracts that build it."""
    global _loaded
    _loaded = _UNSET
