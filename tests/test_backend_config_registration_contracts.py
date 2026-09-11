#!/usr/bin/env python3
"""Contracts that the registry is configured at boot, and that the external
server can be registered from configuration.

Two halves of one hole. ``init_backends_from_config`` was complete, correct,
and never called: the configuration file was never applied, the in-process
backend was built with no model directories, and the external server backend
was never instantiated in production. And even had it been called, the
shipped configuration carried no section for the external server, so nothing
would have registered. A registry that nobody configures is the other face of
a registry that everybody bypasses.

  * BC1 -- a configuration carrying a server section registers the external
    server backend with the host it names; one without does not.
  * BC2 -- the shipped configuration carries that section, so the recipes
    the placement block writes have a backend to be served by.
  * BC3 -- the application lifespan calls the initialiser, so the file is
    applied on every launch path that starts the API.

Nothing here reaches a server: registration is configuration presence, and
availability is the backend's own health check, asked at use time.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"
_SHIPPED = REPO / "opti_oignon" / "config" / "backends.yaml"
_APP = REPO / "opti_oignon" / "api" / "app.py"


def _open():
    loaded, restore = isolate(
        targets={_BACKEND: source("inference_backend.py")},
        packages=("opti_oignon",),
    )
    return loaded[_BACKEND], restore


# ---------------------------------------------------------------------------
# BC1 -- a server section registers the server backend
# ---------------------------------------------------------------------------
def test_bc1_a_server_section_registers_the_server_backend(tmp_path):
    mod, restore = _open()
    try:
        with_section = tmp_path / "with.yaml"
        with_section.write_text(
            "llama_server:\n  host: http://fake:9\n  timeout_s: 2\n",
            encoding="utf-8",
        )
        registry = mod.init_backends_from_config(str(with_section))
        backend = registry.get("llama_server")
        assert backend is not None, "the section registers the backend"
        assert isinstance(backend, mod.LlamaServerBackend)
        assert backend._host == "http://fake:9", "with the host the section names"
    finally:
        restore()

    mod, restore = _open()
    try:
        without = tmp_path / "without.yaml"
        without.write_text("ollama:\n  host: http://localhost:11434\n", encoding="utf-8")
        registry = mod.init_backends_from_config(str(without))
        assert registry.get("llama_server") is None, (
            "no section, no backend: registration is configuration presence"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# BC2 -- the shipped configuration carries the section
# ---------------------------------------------------------------------------
def test_bc2_the_shipped_configuration_carries_the_server_section():
    import yaml

    raw = yaml.safe_load(_SHIPPED.read_text(encoding="utf-8")) or {}
    section = raw.get("llama_server")
    assert isinstance(section, dict) and section, (
        "backends.yaml declares the external server, so the placement "
        "recipes have a backend to be served by"
    )
    assert "host" in section, "and names where it listens"


# ---------------------------------------------------------------------------
# BC3 -- the lifespan applies the configuration
# ---------------------------------------------------------------------------
def test_bc3_the_lifespan_calls_the_initialiser():
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    lifespan = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "lifespan"
    )
    called = {
        (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
        for n in ast.walk(lifespan) if isinstance(n, ast.Call)
    }
    assert "init_backends_from_config" in called, (
        "the lifespan applies backends.yaml on every launch path that starts "
        "the API; a configuration nobody applies is decoration"
    )
