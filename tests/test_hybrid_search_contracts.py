#!/usr/bin/env python3
"""Contracts for the hybrid retrieval engine and for what it is advertised as.

The engine fuses a vector route with a keyword route. Nothing about that
fusion was pinned before, so these clauses hold its arithmetic and its
ordering, and they hold the honesty of the capability the API publishes
about it:

  * Keyword scoring is a real BM25, not a term counter: a term that occurs
    in few documents outweighs one that occurs in all of them, and a long
    document is not rewarded merely for being long.
  * Normalisation lands in the unit range, preserves order, and states what
    it does when every score is identical instead of dividing by zero.
  * Fusion is the stated convex combination, and its two endpoints really
    are pure vector and pure keyword.
  * Fusion merges the two routes by chunk id -- one entry per chunk, both
    component scores kept, and the entry names which routes reached it.
  * Fusion ORDER is reproducible. Results carrying equal scores must come
    back in the same order on every process, or no threshold measured on
    this engine means anything from one run to the next.
  * The vector route asks the store for raw hits: the store's own ordering
    heuristic is switched off, because the fusion below replaces it.
  * Reach is not presence. The capability layer says whether any product
    surface routes a query through this engine, and that statement is
    DERIVED from the tree here -- wiring a caller without saying so fails,
    and saying so without a caller fails too.
  * What the API reports about the engine is bounded by that routed truth,
    so an install can never advertise a retrieval path it cannot take.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
idiom: the module under test is loaded from its source file inside the
shared isolation window.
"""

import ast
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_OO = _ROOT / "opti_oignon"

_ENGINE = "opti_oignon.rag_hybrid_search"

# The names that, when CALLED outside the engine module, mean a product
# surface can reach the engine. A string reference in a lazy-attribute table
# is not a call and does not make the engine reachable.
_ROUTING_CALLS = ("get_hybrid_engine", "HybridSearchEngine")


def _engine_module():
    """Load the engine from source inside the shared window."""
    return isolate(targets={_ENGINE: source("rag_hybrid_search.py")})


def _vec(**scores):
    """A vector-route result table: chunk id -> entry."""
    return {
        cid: {"score": value, "content": cid, "source_file": "f", "file_type": "md"}
        for cid, value in scores.items()
    }


def _kw(**scores):
    """A keyword-route result table: chunk id -> entry."""
    return {
        cid: {"score": value, "content": cid, "source_file": "f", "file_type": "md"}
        for cid, value in scores.items()
    }


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

def test_h1_keyword_scoring_rewards_the_rare_term():
    """A term in one document outranks a term in every document."""
    loaded, restore = _engine_module()
    try:
        scorer = loaded[_ENGINE].BM25Scorer()
        chunks = [
            {"chunk_id": "rare", "content": "common shibboleth"},
            {"chunk_id": "plain", "content": "common common"},
            {"chunk_id": "other", "content": "common filler"},
            {"chunk_id": "more", "content": "common padding"},
        ]
        ranked = dict(scorer.score_chunks("shibboleth common", chunks))

        assert ranked["rare"] > ranked["plain"], (
            "a term carried by one document must outweigh one carried by all"
        )
        # A document that matches nothing but the ubiquitous term is not zero,
        # but the ubiquitous term must contribute far less than the rare one.
        assert ranked["plain"] > 0.0
        assert ranked["rare"] > 2 * ranked["plain"]
    finally:
        restore()


def test_h2_keyword_scoring_normalises_by_document_length():
    """Same term count, longer document, lower score."""
    loaded, restore = _engine_module()
    try:
        scorer = loaded[_ENGINE].BM25Scorer()
        padding = " ".join(f"filler{i}" for i in range(200))
        chunks = [
            {"chunk_id": "short", "content": "quokka"},
            {"chunk_id": "long", "content": "quokka " + padding},
        ]
        ranked = dict(scorer.score_chunks("quokka", chunks))

        assert ranked["short"] > ranked["long"], (
            "length normalisation must penalise the padded document"
        )

        # And the penalty is the b parameter's doing: switch it off and the
        # two documents score the same.
        flat = dict(loaded[_ENGINE].BM25Scorer(b=0.0).score_chunks("quokka", chunks))
        assert abs(flat["short"] - flat["long"]) < 1e-9
    finally:
        restore()


def test_h3_normalisation_is_bounded_order_preserving_and_states_the_flat_case():
    """Scores land in the unit range; ties do not divide by zero."""
    loaded, restore = _engine_module()
    try:
        scorer = loaded[_ENGINE].BM25Scorer()

        out = scorer.normalize_scores([("a", 4.0), ("b", 2.0), ("c", 0.0)])
        values = [v for _, v in out]
        assert values == [1.0, 0.5, 0.0]
        assert all(0.0 <= v <= 1.0 for v in values)
        assert [cid for cid, _ in out] == ["a", "b", "c"], "order must survive"

        # Every score identical and non-zero: the midpoint, not a crash.
        flat = scorer.normalize_scores([("a", 3.0), ("b", 3.0)])
        assert [v for _, v in flat] == [0.5, 0.5]

        # Every score identical and zero: zero, not the midpoint.
        zero = scorer.normalize_scores([("a", 0.0), ("b", 0.0)])
        assert [v for _, v in zero] == [0.0, 0.0]

        assert scorer.normalize_scores([]) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def test_h4_fusion_is_the_stated_combination_and_honours_its_endpoints():
    """alpha weights the two routes; 1.0 and 0.0 are pure routes."""
    loaded, restore = _engine_module()
    try:
        engine = loaded[_ENGINE].HybridSearchEngine()
        vector = _vec(only_v=0.8, both=0.4)
        keyword = _kw(only_k=0.9, both=1.0)

        fused = {r.chunk_id: r for r in engine._fuse_scores(vector, keyword, 0.25)}
        assert abs(fused["both"].fused_score - (0.25 * 0.4 + 0.75 * 1.0)) < 1e-9
        assert abs(fused["only_v"].fused_score - (0.25 * 0.8)) < 1e-9
        assert abs(fused["only_k"].fused_score - (0.75 * 0.9)) < 1e-9

        pure_vector = engine._fuse_scores(vector, keyword, 1.0)
        assert pure_vector[0].chunk_id == "only_v"
        assert abs(pure_vector[0].fused_score - 0.8) < 1e-9

        pure_keyword = engine._fuse_scores(vector, keyword, 0.0)
        assert pure_keyword[0].chunk_id == "both"
        assert abs(pure_keyword[0].fused_score - 1.0) < 1e-9
    finally:
        restore()


def test_h5_fusion_merges_the_two_routes_and_names_them():
    """One entry per chunk, both component scores kept, route named."""
    loaded, restore = _engine_module()
    try:
        engine = loaded[_ENGINE].HybridSearchEngine()
        fused = engine._fuse_scores(
            _vec(only_v=0.8, both=0.4), _kw(only_k=0.9, both=1.0), 0.5,
        )

        assert len(fused) == 3, "a chunk reached twice must not appear twice"
        by_id = {r.chunk_id: r for r in fused}

        assert by_id["both"].search_mode == "hybrid"
        assert by_id["both"].vector_score == 0.4
        assert by_id["both"].keyword_score == 1.0

        assert by_id["only_v"].search_mode == "vector"
        assert by_id["only_v"].keyword_score == 0.0

        assert by_id["only_k"].search_mode == "keyword"
        assert by_id["only_k"].vector_score == 0.0
    finally:
        restore()


def test_h6_fusion_order_is_reproducible_across_processes():
    """Equal scores come back in a stated order, not in cache order.

    The result table is keyed by chunk id. Iterating a set of those keys
    hands back a different order in every process, so two hosts running the
    same query on the same index disagree on what the top hit is whenever
    scores tie. A threshold measured on one of them means nothing on the
    other. The order has to be stated: score first, chunk id to break ties.
    """
    loaded, restore = _engine_module()
    try:
        engine = loaded[_ENGINE].HybridSearchEngine()
        tied = {f"doc{i}::0": 0.5 for i in range(8)}
        fused = engine._fuse_scores(_vec(**tied), {}, 1.0)

        ids = [r.chunk_id for r in fused]
        assert ids == sorted(ids), (
            "results carrying equal scores must be ordered by chunk id, "
            "not by whatever order the key set happened to iterate in"
        )

        # Ties break only AFTER the score has decided.
        mixed = _vec(**{"zz::0": 0.9, "aa::0": 0.5, "bb::0": 0.5})
        assert [r.chunk_id for r in engine._fuse_scores(mixed, {}, 1.0)] == [
            "zz::0", "aa::0", "bb::0",
        ]
    finally:
        restore()


def test_h7_vector_route_asks_the_store_for_raw_hits():
    """The store's own ordering heuristic is off; fusion replaces it."""
    loaded, restore = _engine_module()
    try:
        seen = {}

        class _Response:
            results = ()

        class _Store:
            def query(self, **kwargs):
                seen.update(kwargs)
                return _Response()

        engine = loaded[_ENGINE].HybridSearchEngine(store=_Store())
        engine._vector_search(query="q", collection="c", n_results=5)

        assert seen, "the vector route never reached the store"
        assert seen["rerank"] is False, (
            "the store must not re-order underneath the fusion"
        )
        assert seen["track_citations"] is False, (
            "citations belong to the hybrid layer, not to the store call"
        )
        assert seen["min_score"] == 0.0, (
            "filtering before fusion would drop chunks the keyword route wants"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Reach, and what is advertised about it
# ---------------------------------------------------------------------------

def _routing_call_sites():
    """Every package file that CALLS its way into the engine.

    Derived from the tree by parsing it, never listed by hand: a roster
    written out beside the code is stale the first time someone wires a
    caller and forgets it. The engine's own file is excluded -- it defines
    these names, it does not reach for them.
    """
    engine_file = _OO / "rag_hybrid_search.py"
    sites = []
    for path in sorted(_OO.rglob("*.py")):
        if path == engine_file:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in _ROUTING_CALLS:
                sites.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    return sites


def _declared_constant(path, name):
    """Read a module-level constant WITHOUT importing the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} declares no {name}")


def test_h8_the_routed_statement_matches_the_tree():
    """Reach is declared, and the declaration is checked against the tree."""
    declared = _declared_constant(_OO / "api" / "deps.py", "HYBRID_SEARCH_ROUTED")
    sites = _routing_call_sites()

    assert isinstance(declared, bool), "reach is a statement, not a maybe"
    assert declared == bool(sites), (
        "the declared reach and the tree disagree: declared "
        f"{declared!r}, call sites {sites or 'none'}. Wiring a caller means "
        "flipping the statement; flipping it means wiring a caller."
    )


def test_h9_the_reported_capability_is_bounded_by_reach():
    """The API cannot advertise a retrieval path no query can take."""
    app_source = (_OO / "api" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_source)

    published = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "hybrid_search":
                published = value
    assert published is not None, "the capability report names no hybrid entry"

    names = {n.id for n in ast.walk(published) if isinstance(n, ast.Name)}
    assert "HYBRID_SEARCH_ROUTED" in names, (
        "the reported capability is derived from module presence alone; "
        "presence is not reach, and an install would advertise a retrieval "
        "path that no product surface can take"
    )

    # The name has to arrive from the capability layer, or the expression
    # above raises at request time instead of reporting anything.
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "HYBRID_SEARCH_ROUTED" in imported, (
        "the capability report reads a name it never imports"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# Derived from the module, not listed beside it. A roster written out by hand
# goes stale the moment a clause is added and nobody notices.
_CLAUSES = sorted(name for name in dict(globals()) if name.startswith("test_h"))


def _main() -> int:
    passed = 0
    for name in _CLAUSES:
        try:
            globals()[name]()
        except Exception:
            print(f"FAIL {name}:")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
            passed += 1
    total = len(_CLAUSES)
    print(f"{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(_main())
