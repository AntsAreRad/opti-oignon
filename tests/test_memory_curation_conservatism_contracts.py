#!/usr/bin/env python3
"""Curation contracts: the tidy pass is conservative, gated, and recoverable.

The curation pass consolidates near-duplicate facts and may act on model
retirement proposals. It has no production caller yet; these contracts pin
the chokepoint so any future wiring (a scheduler, a route) inherits a proven
posture: removal is rare, high-confidence, soft by default, and always flows
through the coordinated store.

  * CR1 -- consolidation fires only at the strict near-duplicate threshold
    (well above the add-time one) and keeps the strongest fact, reinforcing
    it; a merely similar pair is left alone;
  * CR2 -- a model proposal below the confidence bar is ignored, and a
    proposed id outside the audited set is ignored (no fabricated targets);
  * CR3 -- the default removal channel is the soft delete (recoverable);
    the hard channel requires an explicit opt-in;
  * CR4 -- removals are fault-isolated: one failing removal is skipped and
    the rest still apply, without raising;
  * CR5 -- an unchanged fact set short-circuits the pass (no-op, zero store
    mutations) and a completed pass is idempotent on immediate re-run;
  * CR6 -- the pass never raises into its caller: a failing store, a failing
    model call, or an unwritable sidecar all degrade to a report;
  * CR7 -- proposal parsing is hardened: reasoning tags and code fences are
    stripped, malformed payloads read as empty, and lenient entry forms
    normalise without error.

Loads the dedup module (for the shared similarity helper) then the curation
module in isolation; the model client import is blocked so resolution stays
deterministic. Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_MEMORY = _REPO / "opti_oignon" / "memory"

_MODULES = ("dedup", "curation")


def _load():
    """Load dedup then curation under a stand-in package.

    Every ``opti_oignon.*`` entry plus the model client entry is snapshotted
    and evicted first, so a previously imported real module cannot leak into
    the isolation window through a lazy import, then restored afterwards.
    """
    keys = ["ollama"] + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # imports of the client fail deterministically

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    memory = types.ModuleType("opti_oignon.memory")
    memory.__path__ = []
    sys.modules["opti_oignon"] = root
    sys.modules["opti_oignon.memory"] = memory

    loaded = {}
    for m in _MODULES:
        full = f"opti_oignon.memory.{m}"
        spec = importlib.util.spec_from_file_location(full, _MEMORY / f"{m}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        setattr(memory, m, mod)
        spec.loader.exec_module(mod)
        loaded[m] = mod

    def restore():
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
        sys.modules.pop("ollama", None)
        for k, v in saved.items():
            sys.modules[k] = v

    return loaded["curation"], restore


class _Fact:
    def __init__(self, fid, text, use_count=0, created_at="2026-01-01"):
        self.id = fid
        self.text = text
        self.category = "fact"
        self.use_count = use_count
        self.created_at = created_at
        self.active = True


class _Store:
    """Recorder coordinated store; per-id failures are injectable."""

    def __init__(self, facts, fail_ids=()):
        self.facts = {f.id: f for f in facts}
        self.fail_ids = set(fail_ids)
        self.soft = []
        self.hard = []
        self.touched = []

    def resolve_user(self, user_id=None):
        return "local"

    def list(self, *, active_only=True, user_id=None):
        return [f for f in self.facts.values() if f.active or not active_only]

    def touch(self, fact_id, *, user_id=None):
        self.touched.append(fact_id)
        return True

    def soft_delete(self, fact_id, *, user_id=None):
        if fact_id in self.fail_ids:
            raise RuntimeError("injected removal failure")
        self.soft.append(fact_id)
        self.facts[fact_id].active = False
        return True

    def hard_delete(self, fact_id, *, user_id=None):
        if fact_id in self.fail_ids:
            raise RuntimeError("injected removal failure")
        self.hard.append(fact_id)
        self.facts.pop(fact_id, None)
        return True


class _RaisingStore:
    def resolve_user(self, user_id=None):
        return "local"

    def list(self, *, active_only=True, user_id=None):
        raise RuntimeError("store down")


_NEAR_A = "the user brews single origin ethiopian coffee at home"
_NEAR_B = "the user brews single origin ethiopian coffee at home daily"
_MID_A = "the user plays classical guitar on quiet sunday evenings"
_MID_B = "the user plays classical guitar on quiet sunday mornings"


def test_cr1_consolidation_needs_the_strict_threshold_and_keeps_the_strongest():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            keep = _Fact("keep", _NEAR_A, use_count=5)
            dupe = _Fact("dupe", _NEAR_B, use_count=0)
            mid1 = _Fact("mid1", _MID_A, use_count=2)
            mid2 = _Fact("mid2", _MID_B, use_count=0)
            store = _Store([keep, dupe, mid1, mid2])
            cur = mod.MemoryCurator(store, state_path=Path(td) / "tidy.json")
            pairs = cur.find_consolidations(store.list())
            assert [(p.keep_id, p.retire_id) for p in pairs] == [("keep", "dupe")]
            report = cur.curate(force=True, use_llm=False)
            assert report.retired_ids == ["dupe"]
            assert "mid1" not in report.retired_ids
            assert "mid2" not in report.retired_ids
            assert "keep" in store.touched
        finally:
            restore()


def test_cr2_low_confidence_and_fabricated_proposals_are_ignored():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            facts = [
                _Fact("solo1", "alpha bravo charlie delta"),
                _Fact("solo2", "echo foxtrot golf hotel"),
                _Fact("solo3", "india juliet kilo lima"),
            ]
            store = _Store(facts)

            def chat(**_kw):
                return {
                    "message": {
                        "content": (
                            '{"retire": ['
                            '{"id": "solo1", "confidence": 0.3},'
                            ' {"id": "ghost", "confidence": 0.99},'
                            ' {"id": "solo2", "confidence": 0.9}]}'
                        )
                    }
                }

            cur = mod.MemoryCurator(
                store, chat_fn=chat, state_path=Path(td) / "tidy.json"
            )
            report = cur.curate(force=True, use_llm=True)
            assert report.retired_ids == ["solo2"]
            assert "solo1" not in report.retired_ids
            assert "ghost" not in report.retired_ids
        finally:
            restore()


def test_cr3_soft_delete_is_the_default_hard_is_an_explicit_opt_in():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = _Store([_Fact("keep", _NEAR_A, use_count=3), _Fact("dupe", _NEAR_B)])
            cur = mod.MemoryCurator(store, state_path=Path(td) / "a.json")
            cur.curate(force=True, use_llm=False)
            assert "dupe" in store.soft
            assert store.hard == []

            store2 = _Store([_Fact("keep", _NEAR_A, use_count=3), _Fact("dupe", _NEAR_B)])
            cur2 = mod.MemoryCurator(store2, state_path=Path(td) / "b.json")
            cur2.curate(force=True, use_llm=False, hard_delete=True)
            assert "dupe" in store2.hard
            assert store2.soft == []
        finally:
            restore()


def test_cr4_one_failing_removal_is_skipped_and_the_rest_apply():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            pair_one = "alpha bravo charlie delta echo foxtrot"
            pair_two = "hotel india juliet kilo lima mike"
            facts = [
                _Fact("k1", pair_one, use_count=3),
                _Fact("d1", pair_one + " golf"),
                _Fact("k2", pair_two, use_count=3),
                _Fact("d2", pair_two + " november"),
            ]
            store = _Store(facts, fail_ids={"d1"})
            cur = mod.MemoryCurator(store, state_path=Path(td) / "tidy.json")
            report = cur.curate(force=True, use_llm=False)
            assert report.retired == 1
            removed = store.soft + store.hard
            assert "d2" in removed
            assert "d1" not in removed
            assert sorted(report.retired_ids) == ["d1", "d2"]
        finally:
            restore()


def test_cr5_unchanged_fingerprint_short_circuits_and_a_pass_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            facts = [
                _Fact("s1", "alpha bravo charlie"),
                _Fact("s2", "delta echo foxtrot"),
            ]
            store = _Store(facts)
            cur = mod.MemoryCurator(store, state_path=Path(td) / "tidy.json")
            first = cur.curate(force=True, use_llm=False)
            assert first.skipped is False
            baseline = (len(store.touched), len(store.soft), len(store.hard))
            second = cur.curate(use_llm=False)
            assert second.skipped is True
            assert (len(store.touched), len(store.soft), len(store.hard)) == baseline
            assert cur.needs_pass() is False
        finally:
            restore()


def test_cr6_the_pass_never_raises_into_its_caller():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            cur = mod.MemoryCurator(
                _RaisingStore(), state_path=Path(td) / "tidy.json"
            )
            report = cur.curate(force=True, use_llm=False)
            assert isinstance(report, mod.CurationReport)

            def bad_chat(**_kw):
                raise RuntimeError("model down")

            store = _Store([_Fact("keep", _NEAR_A, use_count=3), _Fact("dupe", _NEAR_B)])
            cur2 = mod.MemoryCurator(
                store, chat_fn=bad_chat, state_path=Path(td) / "b.json"
            )
            report2 = cur2.curate(force=True, use_llm=True)
            assert "dupe" in report2.retired_ids

            blocker = Path(td) / "blocker"
            blocker.write_text("a file, not a directory", encoding="utf-8")
            store3 = _Store([_Fact("s1", "alpha bravo charlie")])
            cur3 = mod.MemoryCurator(
                store3, state_path=blocker / "nested" / "tidy.json"
            )
            report3 = cur3.curate(force=True, use_llm=False)
            assert isinstance(report3, mod.CurationReport)
        finally:
            restore()


def test_cr7_proposal_parsing_is_hardened_against_malformed_replies():
    mod, restore = _load()
    try:
        parse = mod.parse_curation_response
        hidden = '<think>{"retire":[{"id":"evil","confidence":0.99}]}</think>ok done'
        assert parse(hidden) == []
        fenced = '```json\n{"retire":[{"id":"a","confidence":0.9}]}\n```'
        assert parse(fenced) == [("a", 0.9)]
        assert parse('["b", {"id": "c"}]') == [("b", 1.0), ("c", 1.0)]
        assert parse('{"retire":[{"id":"d","confidence":"high"}]}') == [("d", 1.0)]
        assert parse("not json {broken") == []
        assert parse("") == []
        assert parse('{"retire": "nope"}') == []
        assert parse('[{"noid": 1}]') == []
    finally:
        restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
