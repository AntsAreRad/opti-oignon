"""S189 phase F2 -- Smart routing core (item 1) regression tests.

Covers the one applied fix:

- RTR-02: ``opti_oignon/routing/__init__.py`` previously declared
  ``__all__ = ["SmartRouter", "router"]`` while importing neither symbol, so a
  ``from opti_oignon.routing import SmartRouter`` would have raised. The package
  is now honest: it re-exports nothing (``__all__ == []``) and its docstring no
  longer claims SmartRouter lives here.

The package ``__init__`` has no imports, so it is loaded in isolation under a
dotless name (no parent-package import, no heavy chain). The other F2 item-1
findings (SMR-01 cache/pre-flight, SMR-02 double-classify, RTR-01 dual selection
subsystems + stale router docstring) are recorded only and are not asserted here.
"""

import importlib.util
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[1]
_ROUTING_INIT = _REPO / "opti_oignon" / "routing" / "__init__.py"


def _load_routing_init_isolated():
    # Dotless name -> importlib does not import a parent package; the file has no
    # imports of its own, so exec is side-effect free.
    spec = importlib.util.spec_from_file_location(
        "opti_oignon_routing_init_probe", _ROUTING_INIT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_routing_all_is_empty():
    module = _load_routing_init_isolated()
    assert hasattr(module, "__all__")
    assert module.__all__ == [], (
        "routing package must re-export nothing; SmartRouter is in "
        "opti_oignon.smart_router, the dispatcher in opti_oignon.router"
    )


def test_routing_all_no_longer_advertises_unbound_names():
    module = _load_routing_init_isolated()
    for name in ("SmartRouter", "router"):
        assert name not in module.__all__
        # And the package genuinely does not bind them.
        assert not hasattr(module, name)


def test_routing_docstring_does_not_claim_smartrouter_here():
    content = _ROUTING_INIT.read_text(encoding="utf-8")
    # The corrected docstring points SmartRouter at opti_oignon.smart_router.
    assert "opti_oignon.smart_router" in content
    # The stale "Components:\n- SmartRouter: Main routing logic" line is gone.
    assert "Main routing logic" not in content
