#!/usr/bin/env python3
"""Contracts for the Peels of the onion memory: a summary tree that is only
ever regenerated from the Cellar.

A peel is a summary of one or more Cellar spans. It carries the keys of the
spans it summarises, a digest of their content, and the probe score it
earned at creation. A parent peel is summarised from the union of its
children's spans, never from the children's text, so a photocopy of a
photocopy cannot be produced by the builder and is refused by the verifier
when handed in from outside. Selection at query time is deterministic and
keyword-based here; the vector layer is a host decision and is not claimed.

  * PT1 -- a leaf is built from a Cellar span: it points at the span, its
    digest is the span's, its id is the hash of its text and sources, and
    it verifies.
  * PT2 -- a parent recompresses from source: the summariser receives the
    Cellar turns of the children, not their text; sources are the union.
  * PT3 -- pointer refusal: a source that resolves to no span, a child that
    is not in the tree, a parent whose sources are not its children's, each
    refused by name.
  * PT4 -- divergence refusal: a digest that no longer matches the spans, a
    text that no longer matches the id, each refused.
  * PT5 -- selection: the peel about the queried subject ranks first, an
    ancestor and its descendant are never both selected, the selection fits
    the cap in whole items, and a query matching nothing selects nothing.
  * PT6 -- the tree verifies as a whole: one bad peel refuses the tree.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _open():
    loaded, restore = isolate(
        targets={
            "opti_oignon.memory.probes": source("memory", "probes.py"),
            "opti_oignon.memory.receipts": source("memory", "receipts.py"),
            "opti_oignon.memory.peels": source("memory", "peels.py"),
        },
        packages=("opti_oignon.memory",),
    )
    return loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.receipts"], restore


def _span(n):
    return [
        {"turn_id": f"s{n}a", "role": "user", "text": f"Service {n} was migrated by Alice on 2026-0{n}-10 with 1{n}0 connections."},
        {"turn_id": f"s{n}b", "role": "assistant", "text": f"We agreed that service {n} stays on the new cluster."},
    ]


def _faithful(span):
    return " ".join(t["text"] for t in span)


def _gate(peels):
    return peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)


def _forest(peels, receipts):
    """Three leaves over three spans and one parent over the first two."""
    cellar, tree = receipts.Cellar(), peels.PeelTree()
    keys = [cellar.store(_span(n)) for n in (1, 2, 3)]
    leaves = []
    for key in keys:
        peel, decision = peels.build_leaf(key, cellar, _faithful, _gate(peels), tree)
        assert decision.accepted, decision.reason
        leaves.append(peel)
    parent, decision = peels.build_parent([leaves[0].id, leaves[1].id], cellar, _faithful, _gate(peels), tree)
    assert decision.accepted, decision.reason
    return cellar, tree, keys, leaves, parent


# ---------------------------------------------------------------------------
# PT1 -- a leaf points at its span
# ---------------------------------------------------------------------------
def test_pt1_a_leaf_points_at_its_cellar_span_and_verifies():
    peels, receipts, restore = _open()
    try:
        cellar, tree = receipts.Cellar(), peels.PeelTree()
        key = cellar.store(_span(1))
        peel, decision = peels.build_leaf(key, cellar, _faithful, _gate(peels), tree)
        assert decision.accepted and peel is not None
        assert peel.level == 0 and peel.children == ()
        assert peel.sources == (key,)
        canonical = json.dumps([cellar.get(key)], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert peel.source_digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert peel.id == peels.peel_id(peel.text, peel.sources)
        assert peel.probes_total >= 1 and peel.probes_passed == peel.probes_total
        assert peels.verify_peel(peel, cellar, tree) is None
        assert tree.get(peel.id) == peel and tree.leaves() == [peel]
    finally:
        restore()


# ---------------------------------------------------------------------------
# PT2 -- a parent recompresses from source
# ---------------------------------------------------------------------------
def test_pt2_a_parent_is_summarised_from_the_cellar_turns_never_from_child_text():
    peels, receipts, restore = _open()
    try:
        cellar, tree = receipts.Cellar(), peels.PeelTree()
        keys = [cellar.store(_span(n)) for n in (1, 2)]
        seen = []

        def recording(span):
            seen.append([t["turn_id"] for t in span])
            return _faithful(span)

        leaves = [peels.build_leaf(k, cellar, recording, _gate(peels), tree)[0] for k in keys]
        seen.clear()
        parent, decision = peels.build_parent([lf.id for lf in leaves], cellar, recording, _gate(peels), tree)
        assert decision.accepted, decision.reason
        assert seen == [["s1a", "s1b", "s2a", "s2b"]], "the summariser saw the Cellar turns, once, in order"
        assert parent.level == 1
        assert parent.children == (leaves[0].id, leaves[1].id)
        assert parent.sources == (keys[0], keys[1])
        assert peels.verify_peel(parent, cellar, tree) is None
        assert tree.roots() == [parent, ]
    finally:
        restore()


# ---------------------------------------------------------------------------
# PT3 -- pointer refusal
# ---------------------------------------------------------------------------
def test_pt3_a_pointer_that_does_not_resolve_is_refused_by_name():
    peels, receipts, restore = _open()
    try:
        cellar, tree, keys, leaves, parent = _forest(peels, receipts)
        from dataclasses import replace
        bad_key = "e" * 64
        stray = replace(leaves[2], sources=(bad_key,), id=peels.peel_id(leaves[2].text, (bad_key,)))
        with pytest.raises(peels.PeelIntegrityError) as caught:
            peels.verify_peel(stray, cellar, tree)
        assert bad_key in str(caught.value)
        orphan = replace(parent, children=(leaves[0].id, "f" * 64), id=peels.peel_id(parent.text, parent.sources))
        with pytest.raises(peels.PeelIntegrityError) as caught:
            peels.verify_peel(orphan, cellar, tree)
        assert "f" * 64 in str(caught.value)
        # A parent whose sources are not the union of its children's: a summary
        # of something other than what it claims to stand on.
        narrowed = replace(parent, sources=(keys[0],), id=peels.peel_id(parent.text, (keys[0],)))
        with pytest.raises(peels.PeelIntegrityError) as caught:
            peels.verify_peel(narrowed, cellar, tree)
        assert "children" in str(caught.value)
        # A parent handed in with peel ids as sources: a summary of summaries.
        stacked = replace(parent, sources=(leaves[0].id, leaves[1].id),
                          id=peels.peel_id(parent.text, (leaves[0].id, leaves[1].id)))
        with pytest.raises(peels.PeelIntegrityError):
            peels.verify_peel(stacked, cellar, tree)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PT4 -- divergence refusal
# ---------------------------------------------------------------------------
def test_pt4_a_peel_that_diverged_from_its_spans_or_its_id_is_refused():
    peels, receipts, restore = _open()
    try:
        cellar, tree, keys, leaves, parent = _forest(peels, receipts)
        from dataclasses import replace
        assert peels.verify_peel(leaves[0], cellar, tree) is None, "control"
        diverged = replace(leaves[0], source_digest="0" * 64)
        with pytest.raises(peels.PeelIntegrityError) as caught:
            peels.verify_peel(diverged, cellar, tree)
        assert "digest" in str(caught.value)
        tampered = replace(leaves[0], text=leaves[0].text + " and Bob")
        with pytest.raises(peels.PeelIntegrityError) as caught:
            peels.verify_peel(tampered, cellar, tree)
        assert leaves[0].id in str(caught.value)
    finally:
        restore()


# ---------------------------------------------------------------------------
# PT5 -- selection
# ---------------------------------------------------------------------------
def test_pt5_selection_ranks_the_subject_first_never_stacks_and_fits_the_cap():
    peels, receipts, restore = _open()
    try:
        cellar, tree, keys, leaves, parent = _forest(peels, receipts)
        chosen = peels.select_peels(tree, "what happened with service 3", cap=200)
        assert len(chosen) >= 1
        assert chosen[0].text == leaves[2].text, "the leaf about service 3 ranks first"
        assert chosen[0].provenance.startswith("peel:") and chosen[0].score > 0
        chosen = peels.select_peels(tree, "service 1 migrated by alice", cap=200)
        ids = [c.provenance for c in chosen]
        assert ids and len(ids) == len(set(ids))
        texts = {c.text for c in chosen}
        assert not (parent.text in texts and (leaves[0].text in texts or leaves[1].text in texts)), (
            "an ancestor and its descendant are never both selected"
        )
        tight = peels.select_peels(tree, "service 1 migrated by alice", cap=peels.estimate_tokens(leaves[0].text))
        assert len(tight) == 1 and sum(peels.estimate_tokens(c.text) for c in tight) <= peels.estimate_tokens(leaves[0].text)
        assert peels.select_peels(tree, "quarterly tax filing", cap=200) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# PT6 -- the tree verifies as a whole
# ---------------------------------------------------------------------------
def test_pt6_one_bad_peel_refuses_the_whole_tree():
    peels, receipts, restore = _open()
    try:
        cellar, tree, keys, leaves, parent = _forest(peels, receipts)
        assert tree.verify(cellar) is None, "control: the forest verifies"
        assert len(tree.all()) == 4
        from dataclasses import replace
        bad = replace(leaves[2], source_digest="1" * 64)
        tree.add(bad)
        with pytest.raises(peels.PeelIntegrityError):
            tree.verify(cellar)
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
