#!/usr/bin/env python3
"""Enrolling the weights a host already holds.

Under enforcement the load seam refuses any model whose bytes are not pinned.
A fresh host has no manifest, so the instant it enters the fortress it refuses
every model it owns as unpinned. The pin was only ever written as a side effect
of DOWNLOADING a model, so a host that already held its weights had no way to
enrol them short of fetching them again.

``enroll_models`` is that handle. These contracts hold it to the same posture
the rest of the module keeps:

  * it pins the BYTES on disk, because enrolment is the act of deciding these
    bytes are the right ones (this is what separates it from re-sealing, which
    renews a decision already made and must never re-hash);
  * it refuses rather than downgrade -- required and no key means raise and
    write NOTHING, because a partial manifest reads as a readiness the host does
    not have;
  * and once a model is enrolled, the gate that refused it under enforcement
    lets it through. That last one is the whole point, so it is asserted end to
    end against the real load gate.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _raise_no_key():
    raise FileNotFoundError("PQC keypair file not found")


def _load_provenance(*, required=True, key=b"PRIVATE", hmac_secret=b"h" * 32):
    """The provenance module over stand-in signing and encryption modules.

    Mirrors the fortress-readiness harness: a matched sign/verify pair so a seal
    written here verifies here, and a required flag so the refusal paths can be
    exercised without a fortress on disk.
    """
    pqc = types.ModuleType("opti_oignon.pqc_signatures")
    pqc.PQC_AVAILABLE = True
    pqc.is_pqc_enabled = lambda: required
    pqc.pqc_required = lambda: required
    pqc.load_pqc_keypair = (lambda: (b"PUBLIC", key)) if key else _raise_no_key
    pqc.sign_bytes = lambda data, private: b"SIG:" + data[:8]
    pqc.verify_bytes = lambda data, sig, public: sig == b"SIG:" + data[:8]

    encryption = types.ModuleType("opti_oignon.encryption")
    encryption.load_keyfile = lambda: (hmac_secret, b"salt", "kdf")

    loaded, restore = isolate(
        targets={"opti_oignon.model_provenance": source("model_provenance.py")},
        seeded={
            "opti_oignon.pqc_signatures": pqc,
            "opti_oignon.encryption": encryption,
        },
    )
    return loaded["opti_oignon.model_provenance"], restore


def _model(tmp_path, name, body):
    p = tmp_path / name
    p.write_bytes(body)
    return p


def test_a1_enrolling_pins_the_bytes_on_disk_under_one_seal(tmp_path):
    prov, restore = _load_provenance(required=True)
    try:
        m1 = _model(tmp_path, "llama.gguf", b"\x00GGUF one")
        m2 = _model(tmp_path, "qwen.gguf", b"\x00GGUF two two")
        manifest = tmp_path / "model_provenance.json"

        result = prov.enroll_models([m1, m2], manifest_path=manifest)

        assert result["count"] == 2
        assert result["scheme"] == prov.SCHEME_PQC

        entries = prov.load_manifest(manifest)["entries"]
        assert set(entries) == {"llama.gguf", "qwen.gguf"}
        assert entries["llama.gguf"]["sha256"] == prov.compute_digest(m1), (
            "the pin must be the digest of the BYTES on disk. Anything else is "
            "not a provenance pin, it is a label."
        )
        assert entries["qwen.gguf"]["sha256"] == prov.compute_digest(m2)

        # One seal over the whole set, and it verifies where it was written.
        manifest_obj = prov.load_manifest(manifest)
        keys = prov.resolve_seal_keys()
        assert (
            prov.verify_seal(
                prov._payload_of(manifest_obj), manifest_obj["seal"], keys
            )
            == prov.REASON_VERIFIED
        )
    finally:
        restore()


def test_a2_enrolling_refuses_rather_than_writing_a_manifest_it_cannot_sign(
    tmp_path,
):
    prov, restore = _load_provenance(required=True, key=None)
    try:
        m1 = _model(tmp_path, "llama.gguf", b"\x00GGUF one")
        manifest = tmp_path / "model_provenance.json"

        with pytest.raises(prov.ProvenanceError):
            prov.enroll_models([m1], manifest_path=manifest)

        assert not manifest.exists(), (
            "a signature was required, none could be produced, and a manifest "
            "was written anyway. A partial manifest is read as readiness the "
            "host does not have -- the escalation preflight would wave it "
            "through."
        )
    finally:
        restore()


def test_a3_enrolling_nothing_writes_nothing(tmp_path):
    prov, restore = _load_provenance(required=True)
    try:
        manifest = tmp_path / "model_provenance.json"
        result = prov.enroll_models([], manifest_path=manifest)
        assert result["count"] == 0
        assert not manifest.exists(), (
            "enrolling zero models materialised a manifest. An empty sealed "
            "manifest changes the seal scheme from nothing to a signature, which "
            "would quietly satisfy the very check that guards the fortress."
        )
    finally:
        restore()


def test_a4_an_enrolled_model_passes_the_gate_that_refused_it(tmp_path):
    """The handle earns its place only if the gate changes its mind.

    Before enrolment, a fortress refuses the model as unpinned. After it, the
    same gate on the same bytes allows the load. This is the end-to-end proof
    that the migration path actually leads down the cliff and not off it.
    """
    prov, restore = _load_provenance(required=True)
    try:
        gguf = _model(tmp_path, "llama.gguf", b"\x00GGUF the real weights")
        manifest = tmp_path / "model_provenance.json"

        # Before: no manifest, fortress enforces, the model is refused.
        before = prov.verify_model(
            gguf, manifest_path=manifest, mode="bulbe", config={}
        )
        assert before.allowed is False
        assert before.reason == prov.REASON_MANIFEST_MISSING

        prov.enroll_models([gguf], manifest_path=manifest)

        # After: the same gate on the same bytes lets it through.
        after = prov.verify_model(
            gguf, manifest_path=manifest, mode="bulbe", config={}
        )
        assert after.allowed is True, (
            "the model was enrolled and the fortress still refuses it. The "
            "handle wrote a manifest the gate cannot verify -- the migration "
            "leads off the cliff, not down it."
        )
        assert after.reason == prov.REASON_VERIFIED
    finally:
        restore()


def test_a5_a_tampered_byte_after_enrolment_is_caught(tmp_path):
    """Enrolment pins bytes, so changing them must break the pin.

    If this passed while the bytes changed, the pin would be pinning the name
    and not the content -- exactly the substitution the manifest exists to
    refuse.
    """
    prov, restore = _load_provenance(required=True)
    try:
        gguf = _model(tmp_path, "llama.gguf", b"\x00GGUF the real weights")
        manifest = tmp_path / "model_provenance.json"
        prov.enroll_models([gguf], manifest_path=manifest)

        gguf.write_bytes(b"\x00GGUF swapped weights")
        decision = prov.verify_model(
            gguf, manifest_path=manifest, mode="bulbe", config={}
        )
        assert decision.allowed is False
        assert decision.reason == prov.REASON_DIGEST_MISMATCH
    finally:
        restore()
