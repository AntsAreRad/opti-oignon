#!/usr/bin/env python3
"""Contracts for the core census: the static instrument that measures what
the package is made of, before the core/packs cut is drawn.

The census never executes the package. It reads every module's syntax
tree, resolves the project imports it names -- eager at module scope,
lazy inside a function -- follows them to a transitive closure, classes
the third-party imports as heavy-eager or heavy-lazy, lists the databases
a module names, counts the direct client sites the funnel guard counts,
and maps every router the API includes to the module that defines it. A
census that finds nothing is an error, never a result.

  * CC1 -- the instrument is capable on the real tree: hundreds of modules,
    non-zero closures, the heavy set the footprint guard already names.
  * CC2 -- two runs agree to the field.
  * CC3 -- on a fixture package with a known graph the census is exact:
    closures, eager against lazy, heavy against stdlib, databases.
  * CC4 -- the API map names every included router and resolves each to
    a module; an empty root is refused.

Local-only (the public distribution ships no tests). The script is
imported by name from ``scripts/``; it imports nothing from the package.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO  # noqa: E402

_SCRIPTS_DIR = str(REPO / "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
try:
    import core_census  # noqa: E402
finally:
    sys.path.remove(_SCRIPTS_DIR)

_PACKAGE = REPO / "opti_oignon"


def _fixture(tmp_path):
    pkg = tmp_path / "pkgx"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text(
        "import pkgx.b\n\n\ndef later():\n    from pkgx import c\n    return c\n", encoding="utf-8")
    (pkg / "b.py").write_text("from .sub import d\n", encoding="utf-8")
    (pkg / "c.py").write_text("def use():\n    import pandas\n    return pandas\n", encoding="utf-8")
    (pkg / "sub" / "d.py").write_text("import numpy\nimport json\n\nfrom .. import e\n", encoding="utf-8")
    (pkg / "e.py").write_text('PATH = "data/foo.db"\nOTHER = "notes.db"\n', encoding="utf-8")
    (pkg / "f.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "g.py").write_text("try:\n    import chromadb\nexcept ImportError:\n    chromadb = None\n", encoding="utf-8")
    return pkg


# ---------------------------------------------------------------------------
# CC1 -- capable on the real tree
# ---------------------------------------------------------------------------
def test_cc1_the_instrument_is_capable_on_the_real_tree():
    report = core_census.census(_PACKAGE)
    assert report["source"] == "static"
    modules = report["modules"]
    assert len(modules) >= 300
    assert sum(1 for m in modules.values() if m["closure_eager"]) >= 100, "eager closures are non-zero across the tree"
    app = modules["opti_oignon.api.app"]
    assert len(app["closure_eager"]) >= 60, "the API pulls most of the tree eagerly"
    heavy_eager = set().union(*(set(m["heavy_eager"]) for m in modules.values()))
    assert {"fastapi", "pydantic", "numpy"} <= heavy_eager, "the footprint guard's heavy set is seen"
    assert sum(len(m["databases"]) for m in modules.values()) >= 1
    assert sum(m["direct_client_sites"] for m in modules.values()) >= 1, "the funnel debt is counted here too"
    assert report["totals"]["modules"] == len(modules) and report["totals"]["lines"] >= 100000
    assert "json" not in heavy_eager, "the standard library is not heavy"
    assert not (heavy_eager & set(sys.stdlib_module_names)), "no standard-library name is counted as heavy"


# ---------------------------------------------------------------------------
# CC2 -- deterministic
# ---------------------------------------------------------------------------
def test_cc2_two_runs_agree_to_the_field():
    first = core_census.census(_PACKAGE)
    second = core_census.census(_PACKAGE)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert list(first["modules"]) == sorted(first["modules"]), "modules are listed in one order"


# ---------------------------------------------------------------------------
# CC3 -- exact on a fixture
# ---------------------------------------------------------------------------
def test_cc3_the_census_is_exact_on_a_fixture_package(tmp_path):
    report = core_census.census(_fixture(tmp_path))
    m = report["modules"]
    assert set(m) == {"pkgx", "pkgx.a", "pkgx.b", "pkgx.c", "pkgx.sub", "pkgx.sub.d", "pkgx.e", "pkgx.f", "pkgx.g"}
    # ``from pkg import name`` runs the package's __init__ as well as the
    # module it names: both are dependencies, and the census counts both.
    assert m["pkgx.a"]["imports_eager"] == ["pkgx.b"] and m["pkgx.a"]["imports_lazy"] == ["pkgx", "pkgx.c"]
    assert m["pkgx.a"]["closure_eager"] == ["pkgx", "pkgx.b", "pkgx.e", "pkgx.sub", "pkgx.sub.d"], "eager follows eager only, transitively"
    assert m["pkgx.a"]["closure_all"] == ["pkgx", "pkgx.b", "pkgx.c", "pkgx.e", "pkgx.sub", "pkgx.sub.d"]
    assert m["pkgx.b"]["imports_eager"] == ["pkgx.sub", "pkgx.sub.d"], "a relative import resolves against the package"
    assert m["pkgx.sub.d"]["imports_eager"] == ["pkgx", "pkgx.e"], "a parent-relative import resolves too"
    assert m["pkgx.sub.d"]["heavy_eager"] == ["numpy"] and m["pkgx.sub.d"]["heavy_lazy"] == []
    assert m["pkgx.c"]["heavy_eager"] == [] and m["pkgx.c"]["heavy_lazy"] == ["pandas"]
    assert m["pkgx.g"]["heavy_eager"] == ["chromadb"], "a guarded import at module scope is still eager"
    assert m["pkgx.e"]["databases"] == ["data/foo.db", "notes.db"]
    assert m["pkgx.f"]["closure_all"] == [] and m["pkgx.f"]["heavy_eager"] == []
    assert m["pkgx.sub.d"]["lines"] == 4 and report["totals"]["modules"] == 9
    assert all(mod["direct_client_sites"] == 0 for mod in m.values())


# ---------------------------------------------------------------------------
# CC4 -- the API map, and the refusal of nothing
# ---------------------------------------------------------------------------
def test_cc4_the_api_map_names_every_router_and_an_empty_root_is_refused(tmp_path):
    report = core_census.census(_PACKAGE)
    routers = report["api"]["routers"]
    app_text = (_PACKAGE / "api" / "app.py").read_text(encoding="utf-8")
    assert len(routers) == app_text.count("include_router("), "one entry per included router"
    for router in routers:
        assert router["module"] in report["modules"], f"{router['name']} resolves to a module"
        assert router["closure_eager"] >= 0
    assert len({r["name"] for r in routers}) == len(routers)
    assert report["api"]["app_closure_eager"] == len(report["modules"]["opti_oignon.api.app"]["closure_eager"])
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(ValueError, match="no module"):
        core_census.census(empty)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
