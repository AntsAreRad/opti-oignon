#!/usr/bin/env python3
"""The native core's loader: asked at the call, never at import.

``oo_core`` is a Rust extension built by ``scripts/build_oo_core.sh`` into
this directory and never tracked. The Python modules that can use it ask
:func:`load` when they need it; a missing artefact is ``None`` and the
reference path runs. Importing this package loads nothing: the artefact is
found and loaded inside :func:`load`, once, and cached.

An extension built for one interpreter may crash another at its first call
(one built under 3.13 does, under 3.14), before any check of its own could
run. So the artefact carries the ABI of the interpreter it was built for
(its ``SOABI``, ``cpython-313-x86_64-linux-gnu``: version, ABI flags,
platform), stamped by its build, and :func:`load` reads the stamp in the
file before it loads anything: an artefact without one, or built for
another ABI, is not loaded, the reference path runs, and the loader says so
once.
"""

import importlib.util
import logging
import sysconfig
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


_STAMP = b"\noo_core built for python: "


def stamp(path):
    """The ABI (``SOABI``) of the interpreter an artefact was built for, read from the file without loading it, or
    None when it carries no stamp."""
    data = Path(path).read_bytes()
    at = data.find(_STAMP)
    if at < 0:
        return None
    start = at + len(_STAMP)
    end = data.find(b"\n", start)
    tag = data[start:end] if end > start else b""
    return tag.decode("ascii", "replace") if tag else None


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
        built_for = stamp(path)
    except OSError as exc:
        built_for = f"unreadable ({exc})"
    here = sysconfig.get_config_var("SOABI")
    if not here or built_for != here:
        logger.warning("oo_core present but built for %s, not for this interpreter (%s); the reference path runs: "
                       "rebuild it with scripts/build_oo_core.sh", built_for or "no stamped interpreter", here)
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
