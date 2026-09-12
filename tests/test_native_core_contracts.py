#!/usr/bin/env python3
"""Contracts for the native core: the onion's integrity primitives and its
composer assembly in Rust, behind the Python surface, by strangling.

Python stays the reference and the fallback. The native module is asked
for lazily, at the call, never at import; when it is present its answers
must be the reference's, byte for byte on every hash and field for field
on every assembled prompt; when it is absent nothing changes. The recipe
that builds it is pinned in the tree so the artefact is reproducible, and
the artefact itself is never tracked.

  * NC1 -- the native module builds and loads through the package's loader
    and exposes the surface the Python modules ask for.
  * NC2 -- the four hashes are byte-equal to the reference on fixtures with
    Unicode, escapes, unordered keys and non-string values.
  * NC3 -- the composer assembles the same prompt with the native module
    as without it, and refuses with the same words.
  * NC4 -- with the native module unreachable the reference stands, and no
    onion module reaches for it at import.
  * NC5 -- the recipe is pinned: the crate, its lock, the build script and
    the ignore rule are in the tree, and the loader does not load at import.

NC1 to NC3 need the built artefact; they are deselected by name in the
canonical selection until the CI workflow carries a Rust toolchain, and run
by name on the machine that builds it. NC4 and NC5 always run.

Local-only (the public distribution ships no tests).
"""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_ONION = ("composer", "core_store", "receipts", "peels")
_CRATE = REPO / "rust" / "oo_core"


def _open(*, native):
    targets = {f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _ONION}
    targets["opti_oignon.memory.probes"] = source("memory", "probes.py")
    if native:
        targets["opti_oignon.native"] = source("native", "__init__.py")
        blocked = ()
    else:
        blocked = ("opti_oignon.native",)
    loaded, restore = isolate(targets=targets, blocked=blocked, packages=("opti_oignon.memory",))
    return loaded, restore


def _fixtures():
    return {
        "texts": ["plain", "Ünïcode — ça va? 日本語", 'quotes " and \\ backslash', "line\nbreak\ttab\x01ctl", "", "  spaces  "],
        "spans": [
            [{"turn_id": "t0001", "role": "user", "text": "Ünïcode — ça"}, {"text": "b", "role": "assistant", "turn_id": "t0002"}],
            [{"z": 1, "a": True, "m": None, "nested": {"y": [1, 2, "x"], "b": "é"}}],
            [],
            [{"text": 'esc "q" \\ \n \x1f \x7f'}],
        ],
        "sources": [("k1", "k2"), (), ("only",), ("Ünï", 'q"uote')],
    }


def _state(loaded, *, peels=3):
    composer = loaded["opti_oignon.memory.composer"]
    core_store = loaded["opti_oignon.memory.core_store"]
    receipts = loaded["opti_oignon.memory.receipts"]
    core = core_store.CoreStore()
    core.add("The user is called Alice.", actor=core_store.USER)
    core.add("Answers are concise and in English.", actor=core_store.USER)
    cellar, ledger = receipts.Cellar(), receipts.ReceiptLedger()
    flesh = receipts.Flesh([{"turn_id": f"t{i:02d}", "role": "user", "text": f"Turn {i} says that service {i} moved to the new cluster."} for i in range(1, 7)])
    flesh.evict_oldest(cellar, ledger)
    flesh.evict_oldest(cellar, ledger)
    retrieval = [composer.Peel(text=f"Episode {i}: the migration of service {i} was reviewed.", provenance=f"peel:{i}") for i in range(1, peels + 1)]
    budget = composer.Budget(window=600, reserve=60, core=60, receipts=60, peels=120, flesh=200, turn=100)
    return dict(core=core, ledger=ledger, cellar=cellar, retrieval=retrieval, flesh=flesh, turn="Which service moved last?", budget=budget)


# ---------------------------------------------------------------------------
# NC1 -- builds and loads
# ---------------------------------------------------------------------------
def test_nc1_the_native_module_loads_and_exposes_the_surface():
    loaded, restore = _open(native=True)
    try:
        native = loaded["opti_oignon.native"]
        core = native.load()
        assert core is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        for name in ("entry_hash", "span_key", "peel_id", "digest_spans", "compose_segments", "VERSION"):
            assert hasattr(core, name), f"the native surface carries {name}"
        assert isinstance(core.VERSION, str) and core.VERSION
        assert native.available() is True
        assert native.load() is core, "loaded once, then cached"
    finally:
        restore()


# ---------------------------------------------------------------------------
# NC2 -- hashes byte-equal
# ---------------------------------------------------------------------------
def test_nc2_the_four_hashes_are_byte_equal_to_the_reference():
    loaded, restore = _open(native=True)
    try:
        core = loaded["opti_oignon.native"].load()
        assert core is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        fx = _fixtures()
        for text in fx["texts"]:
            assert core.entry_hash(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
        for span in fx["spans"]:
            expected = hashlib.sha256(json.dumps(list(span), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
            assert core.span_key(span) == expected, f"span_key on {span!r}"
        for text in fx["texts"]:
            for sources in fx["sources"]:
                expected = hashlib.sha256(json.dumps([text, list(sources)], ensure_ascii=False).encode("utf-8")).hexdigest()
                assert core.peel_id(text, list(sources)) == expected, f"peel_id on {text!r} {sources!r}"
        expected = hashlib.sha256(json.dumps([list(s) for s in fx["spans"]], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        assert core.digest_spans(fx["spans"]) == expected
        assert core.digest_spans([]) == hashlib.sha256(b"[]").hexdigest()
        with pytest.raises(Exception):
            core.span_key([{"x": 1.5}])
    finally:
        restore()


# ---------------------------------------------------------------------------
# NC3 -- the composer agrees with itself
# ---------------------------------------------------------------------------
def test_nc3_the_composer_assembles_the_same_prompt_with_and_without_the_native_module():
    with_native, restore_native = _open(native=True)
    try:
        assert with_native["opti_oignon.native"].load() is not None, "oo_core is not built: run scripts/build_oo_core.sh on this machine"
        composer = with_native["opti_oignon.memory.composer"]
        assert composer._native_compose() is not None, "the composer sees the native module"
        native_prompt = composer.compose(**_state(with_native, peels=40))
        state = _state(with_native)
        b = state["budget"]
        refusals = {}
        for name, budget in (
            ("core", composer.Budget(window=b.window, reserve=b.reserve, core=1, receipts=b.receipts, peels=b.peels, flesh=b.flesh, turn=b.turn + b.core - 1)),
            ("flesh", composer.Budget(window=b.window, reserve=b.reserve, core=b.core, receipts=b.receipts, peels=b.peels + b.flesh - 1, flesh=1, turn=b.turn)),
        ):
            with pytest.raises(composer.BudgetError) as caught:
                composer.compose(**{**state, "budget": budget})
            refusals[name] = str(caught.value)
    finally:
        restore_native()
    without, restore = _open(native=False)
    try:
        composer = without["opti_oignon.memory.composer"]
        assert composer._native_compose() is None, "control: the reference path"
        reference = composer.compose(**_state(without, peels=40))
        plain = lambda prompt: [(s.layer, s.text, s.provenance, s.tokens, s.instruction_bearing) for s in prompt.segments]  # noqa: E731
        assert plain(native_prompt) == plain(reference), "field for field, across two windows"
        assert len(plain(reference)) >= 5
        assert native_prompt.tokens == reference.tokens and native_prompt.dropped_peels == reference.dropped_peels
        assert native_prompt.core_root == reference.core_root
        assert native_prompt.render() == reference.render()
        state = _state(without)
        b = state["budget"]
        for name, budget in (
            ("core", composer.Budget(window=b.window, reserve=b.reserve, core=1, receipts=b.receipts, peels=b.peels, flesh=b.flesh, turn=b.turn + b.core - 1)),
            ("flesh", composer.Budget(window=b.window, reserve=b.reserve, core=b.core, receipts=b.receipts, peels=b.peels + b.flesh - 1, flesh=1, turn=b.turn)),
        ):
            with pytest.raises(composer.BudgetError) as caught:
                composer.compose(**{**state, "budget": budget})
            assert str(caught.value) == refusals[name], f"{name}: the refusal is the same sentence"
    finally:
        restore()


# ---------------------------------------------------------------------------
# NC4 -- the reference stands without the native module
# ---------------------------------------------------------------------------
def test_nc4_without_the_native_module_the_reference_stands_and_nothing_imports_it_at_scope():
    loaded, restore = _open(native=False)
    try:
        core_store = loaded["opti_oignon.memory.core_store"]
        receipts = loaded["opti_oignon.memory.receipts"]
        peels = loaded["opti_oignon.memory.peels"]
        composer = loaded["opti_oignon.memory.composer"]
        fx = _fixtures()
        assert core_store.entry_hash(fx["texts"][1]) == hashlib.sha256(fx["texts"][1].encode("utf-8")).hexdigest()
        assert receipts.span_key(fx["spans"][0]) == hashlib.sha256(json.dumps(fx["spans"][0], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        assert peels.peel_id("t", ["a"]) == hashlib.sha256(json.dumps(["t", ["a"]], ensure_ascii=False).encode("utf-8")).hexdigest()
        assert composer._native_compose() is None
        prompt = composer.compose(**_state(loaded))
        assert [s.layer for s in prompt.segments][0] == "core" and prompt.tokens >= 1
    finally:
        restore()
    for name in _ONION:
        tree = ast.parse(source("memory", f"{name}.py").read_text(encoding="utf-8"))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name for n in top if isinstance(n, ast.Import) for a in n.names} | {("." * n.level) + (n.module or "") for n in top if isinstance(n, ast.ImportFrom)}
        assert not any("native" in n for n in names), f"{name}.py asks for the native module at the call, never at import: {names}"


# ---------------------------------------------------------------------------
# NC5 -- the recipe is pinned
# ---------------------------------------------------------------------------
def test_nc5_the_recipe_is_pinned_and_the_loader_loads_nothing_at_import():
    manifest = (_CRATE / "Cargo.toml").read_text(encoding="utf-8")
    assert re.search(r'^name\s*=\s*"oo_core"', manifest, re.M)
    assert re.search(r'crate-type\s*=\s*\["cdylib"\]', manifest)
    assert re.search(r'pyo3.*"=0\.22\.6"', manifest) or re.search(r'version\s*=\s*"=0\.22\.6"', manifest), "pyo3 is pinned exactly"
    lock = (_CRATE / "Cargo.lock").read_text(encoding="utf-8")
    assert 'name = "pyo3"\nversion = "0.22.6"' in lock, "the lock names the pinned pyo3"
    script = REPO / "scripts" / "build_oo_core.sh"
    assert script.is_file() and script.stat().st_mode & 0o111, "the build script is in the tree and executable"
    body = script.read_text(encoding="utf-8")
    assert "cargo build --release" in body and "opti_oignon/native" in body
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "opti_oignon/native/*.so" in ignore, "the artefact is never tracked"
    loader = source("native", "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(loader)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(("oo_core" in (a.name for a in n.names)) for n in top if isinstance(n, ast.Import))
    assert not any(n.module and "oo_core" in n.module for n in top if isinstance(n, ast.ImportFrom))
    assert not any(isinstance(n, ast.ImportFrom) and any(a.name == "oo_core" for a in n.names) for n in top), "the loader imports the artefact inside load(), never at module scope"
    sys.modules.pop("opti_oignon.native.oo_core", None)
    loaded, restore = isolate(targets={"opti_oignon.native": source("native", "__init__.py")}, packages=())
    try:
        assert "opti_oignon.native.oo_core" not in sys.modules, "importing the loader loads no artefact"
        assert loaded["opti_oignon.native"]._loaded is loaded["opti_oignon.native"]._UNSET, "nothing asked, nothing loaded"
        assert hasattr(loaded["opti_oignon.native"], "load") and hasattr(loaded["opti_oignon.native"], "available")
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
