#!/usr/bin/env python3
"""Contracts that an artefact's recipe and benchmark are sealed with it.

A manifest entry pinned one thing: the bytes of a model file, by digest. That
proves WHAT was loaded and nothing about HOW it was laid out or how it
performed. This block built both of those elsewhere -- a placement plan that
resolves to a reproducible command, and benchmark numbers that carry their own
provenance -- and here they are bound to the artefact they describe, under the
same seal, so that changing either without re-sealing is detectable and a
placement can be run again from the manifest alone.

The binding refuses one thing on purpose. A benchmark whose source is
``simulated`` or ``unknown`` is not a measurement of this artefact, and a seal
on it would turn an invented number into an attested one. That is the exact
transformation the provenance label was introduced to prevent, so the seal is
where it becomes load-bearing.

  * AM1 -- a placement plan recorded with a model is under the seal: altering
    it afterwards makes the seal fail.
  * AM2 -- a benchmark labelled measured is recorded, with its source.
  * AM3 -- a benchmark labelled simulated, or unknown, or carrying no source at
    all, is refused by name and the manifest on disk is left exactly as it was.
  * AM4 -- a model recorded with neither keeps the entry shape it always had,
    so existing manifests and readers are untouched.

Nothing here populates the repository's real manifest, which lives under a
directory this session may not touch, and nothing here enables enforcement,
which is an operator's decision. Every manifest in this file is a temporary
file sealed with a throwaway key.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_PLACEMENT = {"n_gpu_layers": 12, "override_tensor": r"\.ffn_.*_exps\.=CPU"}
_MEASURED = {"source": "measured", "tokens_per_second_tg": 41.2}


def _open():
    loaded, restore = isolate(
        targets={
            "opti_oignon.model_provenance": source("model_provenance.py"),
        },
        blocked=("opti_oignon.pqc_signatures",),
    )
    return loaded["opti_oignon.model_provenance"], restore


def _keys(mp):
    return mp.SealKeys(scheme=mp.SCHEME_HMAC, sign_key=b"k", verify_key=b"k")


def _model(tmp_path):
    path = tmp_path / "target.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 64)
    return path


def _sealed_ok(mp, manifest, keys):
    return mp.verify_seal(mp._payload_of(manifest), manifest["seal"], keys)


# ---------------------------------------------------------------------------
# AM1 -- a placement plan is under the seal
# ---------------------------------------------------------------------------
def test_am1_a_placement_plan_is_sealed_with_the_model(tmp_path):
    mp, restore = _open()
    try:
        keys = _keys(mp)
        manifest_path = tmp_path / "manifest.json"
        mp.record_model(
            _model(tmp_path), manifest_path=manifest_path, keys=keys,
            placement=_PLACEMENT,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["entries"]["target.gguf"]
        assert entry["placement"] == _PLACEMENT, (
            "the plan is recorded beside the digest it belongs to"
        )
        assert _sealed_ok(mp, manifest, keys) == mp.REASON_VERIFIED, (
            "the manifest verifies as written"
        )
        manifest["entries"]["target.gguf"]["placement"]["n_gpu_layers"] = 99
        assert _sealed_ok(mp, manifest, keys) != mp.REASON_VERIFIED, (
            "altering the plan without re-sealing breaks the seal: the "
            "placement is attested, not merely stored"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AM2 -- a measured benchmark is recorded with its source
# ---------------------------------------------------------------------------
def test_am2_a_measured_benchmark_is_recorded(tmp_path):
    mp, restore = _open()
    try:
        keys = _keys(mp)
        manifest_path = tmp_path / "manifest.json"
        mp.record_model(
            _model(tmp_path), manifest_path=manifest_path, keys=keys,
            benchmark=_MEASURED,
        )
        entry = json.loads(manifest_path.read_text(encoding="utf-8"))[
            "entries"
        ]["target.gguf"]
        assert entry["benchmark"]["source"] == "measured", (
            "the benchmark keeps the source it was recorded with"
        )
        assert entry["benchmark"]["tokens_per_second_tg"] == 41.2, (
            "and its figure"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AM3 -- an unmeasured benchmark is refused and the manifest untouched
# ---------------------------------------------------------------------------
def test_am3_an_unmeasured_benchmark_is_refused(tmp_path):
    mp, restore = _open()
    try:
        keys = _keys(mp)
        manifest_path = tmp_path / "manifest.json"
        mp.record_model(_model(tmp_path), manifest_path=manifest_path, keys=keys)
        before = manifest_path.read_bytes()

        for bad in (
            {"source": "simulated", "tokens_per_second_tg": 34.0},
            {"source": "unknown", "tokens_per_second_tg": 34.0},
            {"tokens_per_second_tg": 34.0},
        ):
            raised = ""
            try:
                mp.record_model(
                    _model(tmp_path), manifest_path=manifest_path, keys=keys,
                    benchmark=bad,
                )
            except mp.ProvenanceError as exc:
                raised = str(exc)
            label = bad.get("source", "missing")
            assert label in raised or "source" in raised, (
                f"a benchmark whose source is {label!r} is refused, naming why"
            )
            assert manifest_path.read_bytes() == before, (
                "and the manifest on disk is byte-identical: a refusal writes "
                "nothing"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AM4 -- an entry with neither keeps its old shape
# ---------------------------------------------------------------------------
def test_am4_a_bare_entry_keeps_its_shape(tmp_path):
    mp, restore = _open()
    try:
        keys = _keys(mp)
        manifest_path = tmp_path / "manifest.json"
        mp.record_model(_model(tmp_path), manifest_path=manifest_path, keys=keys)
        entry = json.loads(manifest_path.read_text(encoding="utf-8"))[
            "entries"
        ]["target.gguf"]
        assert set(entry) == {"sha256", "size", "recorded_at"}, (
            "a model recorded with no plan and no benchmark carries exactly "
            "the keys it always did, so nothing that reads a manifest today "
            "sees a shape it does not expect"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    import tempfile

    tests = [
        ("AM1 placement is sealed with the model", test_am1_a_placement_plan_is_sealed_with_the_model),
        ("AM2 measured benchmark is recorded", test_am2_a_measured_benchmark_is_recorded),
        ("AM3 unmeasured benchmark is refused", test_am3_an_unmeasured_benchmark_is_refused),
        ("AM4 bare entry keeps its shape", test_am4_a_bare_entry_keeps_its_shape),
    ]
    passed = 0
    for label, fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"PASS  {label}")
                passed += 1
            except Exception:  # noqa: BLE001 -- report and continue
                print(f"FAIL  {label}")
                traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
