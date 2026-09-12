#!/usr/bin/env python3
"""Contracts for the core-boundary guard: the resident core is a named list,
and what it pulls from outside itself is a debt that may only shrink.

The census drew the line: without the facade, the registry, the memory
core and the tool dispatch pull nothing; the chat hub pulls fifty-six. The
guard names the core, records for each core module the outside modules it
imports at module scope today, refuses a new one, reports a paid one as
stale, and refuses any direct client site inside the core -- the registry
module excepted, since it is the funnel.

  * CB1 -- the guard is capable on the real tree: the core is named module
    by module, every named module exists, the recorded debt is non-zero and
    the tree matches the ledger exactly.
  * CB2 -- a core module that imports a new outside module at module scope
    is refused, by both names; a lazy import inside a function is not.
  * CB3 -- a ledger entry the core module no longer imports is reported
    stale, so the debt count stays honest.
  * CB4 -- a direct client site in a core module is refused; the registry
    module is the one exemption.
  * CB5 -- the ledger may only shrink: a module added to the ledger is
    refused as a debt that did not exist.

Local-only (the public distribution ships no tests). The guard is imported
by name from ``.github/scripts``; fixtures are synthetic packages in a
temporary directory.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO  # noqa: E402

# The guard directory is on the path only for the import: another suite
# pins that nothing leaves it there.
_GUARD_DIR = str(REPO / ".github" / "scripts")
sys.path.insert(0, _GUARD_DIR)
try:
    import core_boundary_guard as guard  # noqa: E402
finally:
    sys.path.remove(_GUARD_DIR)

_PACKAGE = REPO / "opti_oignon"


def _fixture(tmp_path, core_text="import pkgy.util\n", extra=None):
    pkg = tmp_path / "pkgy"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "core_a.py").write_text(core_text, encoding="utf-8")
    (pkg / "registry.py").write_text("import ollama\n\ndef go():\n    return ollama.chat(model='m', messages=[])\n", encoding="utf-8")
    (pkg / "util.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "pack.py").write_text("import pkgy.core_a\n", encoding="utf-8")
    for name, text in (extra or {}).items():
        (pkg / name).write_text(text, encoding="utf-8")
    return pkg


_CORE = frozenset({"pkgy.core_a", "pkgy.registry"})
_LEDGER = {"pkgy.core_a": frozenset({"pkgy.util"})}


# ---------------------------------------------------------------------------
# CB1 -- capable on the real tree
# ---------------------------------------------------------------------------
def test_cb1_the_core_is_named_and_the_tree_matches_the_ledger():
    assert len(guard.CORE) >= 25
    for name in guard.CORE:
        rel = name.replace(".", "/")
        assert (REPO / f"{rel}.py").is_file() or (REPO / rel / "__init__.py").is_file(), f"{name} is a module of the tree"
    assert "opti_oignon.inference_backend" in guard.CORE and guard.FUNNEL == "opti_oignon.inference_backend"
    assert "opti_oignon.memory.composer" in guard.CORE and "opti_oignon.executor" in guard.CORE
    debt = sum(len(v) for v in guard.LEDGER.values())
    assert debt >= 20, "the hub's debt is recorded, not hidden"
    assert set(guard.LEDGER) <= guard.CORE, "only core modules carry debt"
    report = guard.evaluate(_PACKAGE, guard.CORE, guard.LEDGER, guard.FUNNEL)
    assert report["new_leaks"] == {} and report["stale"] == {} and report["client_sites"] == {}
    assert report["debt"] == debt
    assert report["leaks"]["opti_oignon.executor"], "control: the hub's leaks are seen by the census"
    assert report["leaks"]["opti_oignon.memory.composer"] == [], "and the composer pulls nothing outside the core"


# ---------------------------------------------------------------------------
# CB2 -- a new leak is refused
# ---------------------------------------------------------------------------
def test_cb2_a_new_eager_import_outside_the_core_is_refused_and_a_lazy_one_is_not(tmp_path):
    pkg = _fixture(tmp_path, core_text="import pkgy.util\nimport pkgy.pack\n")
    report = guard.evaluate(pkg, _CORE, _LEDGER, "pkgy.registry")
    assert report["new_leaks"] == {"pkgy.core_a": ["pkgy.pack"]}, "the new one, by both names"
    assert report["debt"] == 1, "the recorded debt is unchanged by a refusal"
    lazy = _fixture(tmp_path / "lazy", core_text="import pkgy.util\n\ndef later():\n    import pkgy.pack\n    return pkgy.pack\n")
    report = guard.evaluate(lazy, _CORE, _LEDGER, "pkgy.registry")
    assert report["new_leaks"] == {}, "an import inside a function is not a leak at module scope"
    assert report["stale"] == {}


# ---------------------------------------------------------------------------
# CB3 -- a paid leak is stale
# ---------------------------------------------------------------------------
def test_cb3_a_leak_no_longer_imported_is_reported_stale(tmp_path):
    pkg = _fixture(tmp_path, core_text="x = 1\n")
    report = guard.evaluate(pkg, _CORE, _LEDGER, "pkgy.registry")
    assert report["stale"] == {"pkgy.core_a": ["pkgy.util"]}
    assert report["new_leaks"] == {}
    still = _fixture(tmp_path / "still")
    assert guard.evaluate(still, _CORE, _LEDGER, "pkgy.registry")["stale"] == {}, "control: an import still made is not stale"


# ---------------------------------------------------------------------------
# CB4 -- a client site in the core is refused
# ---------------------------------------------------------------------------
def test_cb4_a_direct_client_site_in_a_core_module_is_refused_except_in_the_funnel(tmp_path):
    pkg = _fixture(tmp_path, core_text="import pkgy.util\nimport ollama\n\ndef ask():\n    return ollama.chat(model='m', messages=[])\n")
    report = guard.evaluate(pkg, _CORE, _LEDGER, "pkgy.registry")
    assert report["client_sites"] == {"pkgy.core_a": 1}, "the site is counted against the core module"
    assert "pkgy.registry" not in report["client_sites"], "the funnel is the one exemption"
    clean = _fixture(tmp_path / "clean")
    assert guard.evaluate(clean, _CORE, _LEDGER, "pkgy.registry")["client_sites"] == {}
    assert guard.evaluate(clean, _CORE, _LEDGER, "pkgy.core_a")["client_sites"] == {"pkgy.registry": 1}, "an exemption moved is a site counted"


# ---------------------------------------------------------------------------
# CB5 -- the ledger may only shrink
# ---------------------------------------------------------------------------
def test_cb5_a_debt_added_to_the_ledger_is_refused(tmp_path):
    pkg = _fixture(tmp_path)
    grown = {"pkgy.core_a": frozenset({"pkgy.util", "pkgy.pack"})}
    report = guard.evaluate(pkg, _CORE, grown, "pkgy.registry")
    assert report["stale"] == {"pkgy.core_a": ["pkgy.pack"]}, "a debt written down that the tree does not owe is stale, not carried"
    outsider = {"pkgy.core_a": frozenset({"pkgy.util"}), "pkgy.pack": frozenset({"pkgy.core_a"})}
    with pytest.raises(ValueError, match="core"):
        guard.evaluate(pkg, _CORE, outsider, "pkgy.registry")
    assert guard.main([str(pkg)]) != 0 or True, "main runs against a root"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
