#!/usr/bin/env python3
"""
S183 / W-02: every <label> in the touched panels must be associated with a
control -- either via a for= attribute or by wrapping a labelable control --
so the Svelte a11y-label-has-associated-control warning no longer fires.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MA = ROOT / "frontend/src/lib/components/panels/ModelAssignment.svelte"
BP = ROOT / "frontend/src/lib/components/panels/benchmark/BenchmarkProfiles.svelte"

_LABEL = re.compile(r"<label\b([^>]*)>(.*?)</label>", re.DOTALL)
_CONTROL = re.compile(r"<(input|select|textarea)\b")


def _orphan_labels(src: str):
    orphans = []
    for m in _LABEL.finditer(src):
        attrs, body = m.group(1), m.group(2)
        if "for=" in attrs:
            continue
        if _CONTROL.search(body):
            continue
        orphans.append(m.group(0)[:80])
    return orphans


class TestNoOrphanLabels:
    def test_model_assignment_labels_associated(self):
        orphans = _orphan_labels(MA.read_text(encoding="utf-8"))
        assert orphans == [], f"orphan labels: {orphans}"

    def test_benchmark_profiles_labels_associated(self):
        orphans = _orphan_labels(BP.read_text(encoding="utf-8"))
        assert orphans == [], f"orphan labels: {orphans}"


class TestModelAssignmentAssociation:
    def test_three_selects_have_ids(self):
        src = MA.read_text(encoding="utf-8")
        for role in ("primary", "fast", "quality"):
            assert f"id={{`edit-{role}-${{role.role}}`}}" in src

    def test_three_labels_have_for(self):
        src = MA.read_text(encoding="utf-8")
        assert src.count('<label class="edit-label" for=') == 3
        assert '<label class="edit-label">' not in src


class TestBenchmarkProfilesGroups:
    def test_group_headings_are_spans(self):
        src = BP.read_text(encoding="utf-8")
        assert '<span class="bv2-label">Categories</span>' in src
        assert '<span class="bv2-label">Weights</span>' in src
        assert '<label class="bv2-label">Categories' not in src
        assert '<label class="bv2-label">Weights' not in src

    def test_field_labels_still_have_for(self):
        # Regression guard: the real field labels keep their association.
        src = BP.read_text(encoding="utf-8")
        assert '<label class="bv2-label" for="editor-name">' in src
        assert '<label class="bv2-label" for="editor-desc">' in src
