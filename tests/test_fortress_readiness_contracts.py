#!/usr/bin/env python3
"""A requirement with no way to comply is a trap, not a policy.

The fortress REQUIRES the post-quantum signature. There it is a property of the
mode, like the socket bind, and no configuration file may switch it off. That
requirement is now wired and it is right. What was never wired is the way to
SATISFY it, and three gaps between them made a cliff with no path down.

  * A host holding a manifest sealed with a MAC cannot turn that MAC into a
    signature. The seal is only ever written as a SIDE EFFECT of enrolling a
    model, so the only way to re-seal was to download every model again. A
    fortress reads a MAC as a downgrade and refuses every model it owns.

  * Escalation says of itself that adding security is always safe. It is not.
    A host with no signing key becomes, the instant it escalates, a host that
    refuses every model and every backup. Escalation is immediate; the way back
    is a ceremony with a cooldown.

  * The boot checklist calls the primitive GREEN on exactly that host, because
    it asks whether the primitive RESOLVED -- and it resolves on a machine with
    no key at all. That is the same wrong question the seal used to ask, and it
    is the question this estate has now been wrong about twice.

So: one function that says what stands between this host and a signature it can
produce. One operation that re-seals what is pinned WITHOUT re-pinning it. A
readiness check placed where refusing is free -- at the escalation, not at the
boot, because a critical boot check would take down the very endpoints that
repair the condition it is complaining about.

And a floor under all of it. The emergency stop drops to the fortress, and it
must never be refused. A panic button that can say no is not a panic button.
"""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _write_keypair(tmp_path, *, algorithm, public=b"PUB", private=b"PRIV"):
    import base64

    payload = {
        "public_key": base64.urlsafe_b64encode(public).decode("ascii"),
        "private_key": base64.urlsafe_b64encode(private).decode("ascii"),
    }
    if algorithm is not None:
        payload["algorithm"] = algorithm
    fpath = tmp_path / ".pqc_keypair"
    fpath.write_text(json.dumps(payload), encoding="ascii")
    return fpath


# ---------------------------------------------------------------------------
# What stands between this host and a signature it can produce
# ---------------------------------------------------------------------------


def _load_signing(tmp_path, *, available=True, algorithm="ML-DSA-65", reason=None):
    loaded, restore = isolate(
        targets={"opti_oignon.pqc_signatures": source("pqc_signatures.py")},
    )
    mod = loaded["opti_oignon.pqc_signatures"]
    mod.PQC_AVAILABLE = available
    mod.PQC_MECHANISM = algorithm if available else None
    mod.PQC_UNAVAILABLE_REASON = reason
    mod._PQC_ALGORITHM = algorithm if available else None
    mod._DEFAULT_KEYPAIR_PATH = tmp_path / ".pqc_keypair"
    return mod, restore


def test_f1_a_dead_primitive_is_a_blocker_and_the_key_question_is_moot(tmp_path):
    mod, restore = _load_signing(
        tmp_path, available=False, reason="liboqs offers no usable mechanism."
    )
    try:
        blockers = mod.signing_blockers()
        assert blockers, "the primitive did not resolve and nothing stands in the way?"
        assert "liboqs" in " ".join(blockers)
    finally:
        restore()


def test_f2_a_resolved_primitive_with_no_key_is_a_blocker(tmp_path):
    mod, restore = _load_signing(tmp_path)
    try:
        blockers = mod.signing_blockers()
        assert blockers, (
            "THE trap. The primitive resolves, so every check that asks whether "
            "it resolved says green -- on a host that cannot sign a single byte "
            "because it holds no key. A fortress here refuses every model it owns."
        )
        assert ".pqc_keypair" in " ".join(blockers), (
            "the operator must be able to act on the reason without reading the "
            "source: say WHICH file is missing"
        )
    finally:
        restore()


def test_f3_a_key_minted_under_another_mechanism_is_a_blocker(tmp_path):
    mod, restore = _load_signing(tmp_path)
    try:
        _write_keypair(tmp_path, algorithm="Dilithium3")
        assert mod.signing_blockers(), (
            "the key on disk was minted under a different algorithm. Its bytes "
            "would be rejected deep inside the signer, or worse: a signature "
            "nobody can verify. Cannot-be-shown-to-agree is a blocker."
        )
    finally:
        restore()


def test_f4_a_resolved_primitive_with_an_agreeing_key_blocks_nothing(tmp_path):
    mod, restore = _load_signing(tmp_path)
    try:
        _write_keypair(tmp_path, algorithm="ML-DSA-65")
        assert mod.signing_blockers() == [], (
            "guards the refusal against over-reach: a host that CAN sign must "
            "not be told it cannot"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Re-sealing: the missing half of the requirement
# ---------------------------------------------------------------------------


def _load_provenance(*, required=True, key=b"PRIVATE", hmac_secret=b"h" * 32):
    pqc = types.ModuleType("opti_oignon.pqc_signatures")
    pqc.PQC_AVAILABLE = True
    pqc.is_pqc_enabled = lambda: required
    pqc.pqc_required = lambda: required
    pqc.load_pqc_keypair = (
        (lambda: (b"PUBLIC", key)) if key else _raise_no_key
    )
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


def _raise_no_key():
    raise FileNotFoundError("PQC keypair file not found")


def _mac_sealed_manifest(prov, tmp_path, entries):
    """A manifest as a Daily host leaves it: pinned, and sealed with a MAC."""
    payload = {"version": prov.MANIFEST_VERSION, "entries": entries}
    keys = prov.SealKeys(scheme=prov.SCHEME_HMAC, sign_key=b"h" * 32, verify_key=b"h" * 32)
    sealed = dict(payload)
    sealed["seal"] = prov.compute_seal(payload, keys)
    path = tmp_path / "model_provenance.json"
    prov.save_manifest(sealed, path)
    return path


_ENTRIES = {
    "llama.gguf": {"sha256": "a" * 64, "size": 10, "recorded_at": 1.0},
    "qwen.gguf": {"sha256": "b" * 64, "size": 20, "recorded_at": 2.0},
}


def test_f5_the_manifest_says_what_it_is_sealed_with(tmp_path):
    prov, restore = _load_provenance()
    try:
        assert prov.manifest_seal_scheme(tmp_path / "absent.json") is None
        path = _mac_sealed_manifest(prov, tmp_path, _ENTRIES)
        assert prov.manifest_seal_scheme(path) == prov.SCHEME_HMAC
    finally:
        restore()


def test_f6_re_sealing_turns_the_mac_into_a_signature_that_verifies(tmp_path):
    prov, restore = _load_provenance(required=True)
    try:
        path = _mac_sealed_manifest(prov, tmp_path, _ENTRIES)
        result = prov.reseal_manifest(manifest_path=path)

        assert result["scheme"] == prov.SCHEME_PQC
        assert prov.manifest_seal_scheme(path) == prov.SCHEME_PQC

        manifest = prov.load_manifest(path)
        keys = prov.resolve_seal_keys()
        assert prov.verify_seal(
            prov._payload_of(manifest), manifest["seal"], keys
        ) == prov.REASON_VERIFIED, (
            "the manifest was re-sealed and the host still cannot verify it. A "
            "seal that does not verify where it was written is not a seal."
        )
    finally:
        restore()


def test_f7_re_sealing_never_re_pins(tmp_path):
    """The digests are carried across VERBATIM. This is the whole safety of it.

    A re-seal that re-hashed the files on disk would bless whatever is there
    now -- which is precisely the substitution the manifest exists to refuse. An
    entry means somebody looked at those bytes and decided they were the right
    ones, and renewing that decision is not this operation's to make.
    """
    prov, restore = _load_provenance(required=True)
    try:
        path = _mac_sealed_manifest(prov, tmp_path, _ENTRIES)
        prov.reseal_manifest(manifest_path=path)

        entries = prov.load_manifest(path)["entries"]
        assert entries == _ENTRIES, (
            "re-sealing moved a pin. The operation that exists to change the "
            "SCHEME has changed the CLAIM, and a swapped model would now carry "
            "a valid signature over its own digest."
        )
    finally:
        restore()


def test_f8_re_sealing_refuses_rather_than_writing_a_mac_a_fortress_will_reject(tmp_path):
    prov, restore = _load_provenance(required=True, key=None)
    try:
        path = _mac_sealed_manifest(prov, tmp_path, _ENTRIES)
        before = path.read_bytes()

        with pytest.raises(prov.ProvenanceError):
            prov.reseal_manifest(manifest_path=path)

        assert path.read_bytes() == before, (
            "the re-seal could not produce a signature and wrote SOMETHING "
            "anyway. A partial write here leaves the operator believing the "
            "migration happened."
        )
    finally:
        restore()


def test_f9_there_is_nothing_to_re_seal_and_that_is_said_plainly(tmp_path):
    prov, restore = _load_provenance(required=True)
    try:
        with pytest.raises(prov.ProvenanceError):
            prov.reseal_manifest(manifest_path=tmp_path / "absent.json")
    finally:
        restore()


# ---------------------------------------------------------------------------
# Escalation: the one place where refusing is free
# ---------------------------------------------------------------------------


def _load_mode(
    *,
    blockers=(),
    scheme="mldsa65",
    broken=False,
    backend="ollama",
    manifest_entries=None,
):
    seeded = {}
    if not broken:
        pqc = types.ModuleType("opti_oignon.pqc_signatures")
        pqc.signing_blockers = lambda: list(blockers)
        prov = types.ModuleType("opti_oignon.model_provenance")
        prov.SCHEME_PQC = "mldsa65"
        prov.manifest_seal_scheme = lambda: scheme
        # The absent-manifest brick check reads these two. Their defaults
        # (an ungated backend, no manifest) leave every earlier contract
        # untouched: an ungated backend short-circuits the check to empty.
        prov.backend_enforces_provenance = lambda b: b == "llama_cpp"
        prov.load_manifest = lambda path=None: (
            {"entries": manifest_entries} if manifest_entries else None
        )
        seeded = {
            "opti_oignon.pqc_signatures": pqc,
            "opti_oignon.model_provenance": prov,
        }

    loaded, restore = isolate(
        targets={"opti_oignon.security_mode": source("security_mode.py")},
        seeded=seeded,
    )
    mod = loaded["opti_oignon.security_mode"]

    written = []
    mod._load_signing_key = lambda: b"k" * 32
    mod._write_yaml_mode = lambda mode: written.append(mode)
    mod._write_lockfile = lambda mode, user_id, key: 1.0
    mod._audit_log = lambda *a, **k: None
    mod._default_backend = lambda: backend

    manager = mod.SecurityModeManager()
    manager._cached_mode = mod.MODE_DAILY
    return mod, manager, written, restore


def test_f10_a_host_that_cannot_sign_is_not_walked_into_the_fortress():
    mod, manager, written, restore = _load_mode(
        blockers=["no signing keypair on disk (data/.pqc_keypair)"]
    )
    try:
        result = manager.escalate_to_bulbe("leon")

        assert result["success"] is False, (
            "escalation says of itself that adding security is always safe. It "
            "escalated a host that will now refuse every model it owns and every "
            "backup it exports, and the way back out is a ceremony with a cooldown."
        )
        assert written == [], "it refused and wrote the mode anyway"
        remedy = json.dumps(result)
        assert "generate-keys" in remedy and "reseal" in remedy, (
            "a refusal that offers no remedy is a wall. Name the two calls."
        )
    finally:
        restore()


def test_f11_the_emergency_stop_is_never_refused():
    """force -- a panic button that can say no is not a panic button.

    The emergency stop drops to the fortress and RAISES when the escalation
    reports failure. A readiness check that could refuse it would turn the panic
    path into an exception on the very host that needed it most.
    """
    mod, manager, written, restore = _load_mode(
        blockers=["no signing keypair on disk (data/.pqc_keypair)"]
    )
    try:
        result = manager.escalate_to_bulbe("emergency-stop", force=True)
        assert result["success"] is True
        assert written == [mod.MODE_BULBE]
    finally:
        restore()


def test_f12_a_ready_host_escalates():
    mod, manager, written, restore = _load_mode(blockers=(), scheme="mldsa65")
    try:
        result = manager.escalate_to_bulbe("leon")
        assert result["success"] is True
        assert written == [mod.MODE_BULBE]
    finally:
        restore()


def test_f13_a_mac_sealed_manifest_is_itself_a_blocker():
    mod, manager, written, restore = _load_mode(
        blockers=(), scheme="hmac-sha512"
    )
    try:
        result = manager.escalate_to_bulbe("leon")
        assert result["success"] is False, (
            "the key is there and the manifest is still sealed with a MAC. A "
            "fortress reads that as a downgrade and refuses every model: having "
            "minted the key is only half of the migration."
        )
        assert written == []
    finally:
        restore()


def test_f14_a_broken_tree_may_not_manufacture_a_refusal():
    """Nothing seeded: the window refuses the names, so readiness is unknowable.

    Refusing here would pin the host in Daily on the strength of the check's own
    inability to run, and buy no security at all with it.
    """
    mod, manager, written, restore = _load_mode(broken=True)
    try:
        result = manager.escalate_to_bulbe("leon")
        assert result["success"] is True
        assert written == [mod.MODE_BULBE]
    finally:
        restore()


# ---------------------------------------------------------------------------
# The plainest brick: a gated backend and no manifest at all
# ---------------------------------------------------------------------------


def test_f18_a_gated_backend_with_no_manifest_blocks_escalation():
    """The absent manifest the downgrade check never looked at.

    A fresh host has no manifest. On a fortress the load seam enforces and an
    unpinned model is refused, so a gated backend escalated here loads nothing.
    The downgrade check above only ever fires on a manifest that EXISTS with the
    wrong scheme -- it is blind to the host that has no manifest at all.
    """
    mod, manager, written, restore = _load_mode(
        blockers=(), scheme=None, backend="llama_cpp", manifest_entries=None
    )
    try:
        result = manager.escalate_to_bulbe("leon")
        assert result["success"] is False, (
            "the load backend verifies provenance and there is no manifest. In "
            "Bulbe every model is refused as unpinned -- escalation walks the "
            "host straight into a brick and the preflight said nothing."
        )
        assert written == []
        assert "enroll-models" in json.dumps(result), (
            "a refusal with no remedy is a wall. Name the enrolment call that "
            "writes the manifest whose absence is the blocker."
        )
    finally:
        restore()


def test_f19_a_gated_backend_with_a_manifest_is_not_falsely_bricked():
    mod, manager, written, restore = _load_mode(
        blockers=(),
        scheme="mldsa65",
        backend="llama_cpp",
        manifest_entries={"llama.gguf": {"sha256": "a" * 64}},
    )
    try:
        result = manager.escalate_to_bulbe("leon")
        assert result["success"] is True, (
            "the manifest exists with pins and is sealed with a signature; the "
            "gated backend has something to verify. The absent-manifest blocker "
            "fired on a host that was ready -- a false brick."
        )
        assert written == [mod.MODE_BULBE]
    finally:
        restore()


def test_f20_an_ungated_backend_with_no_manifest_is_not_a_blocker():
    """The scope line. Ollama does not call the gate, so escalation does not
    brick it -- and a check that refused here would pin the host in Daily and
    buy no security. The weakness that Ollama loads unverified weights in a
    fortress is a documented gap, not an escalation wall.
    """
    mod, manager, written, restore = _load_mode(
        blockers=(), scheme=None, backend="ollama", manifest_entries=None
    )
    try:
        result = manager.escalate_to_bulbe("leon")
        assert result["success"] is True, (
            "an ungated backend was refused escalation over a manifest its load "
            "path never reads. That buys no security and only walls off Daily."
        )
        assert written == [mod.MODE_BULBE]
    finally:
        restore()


def test_f21_the_gated_backend_set_is_measured_not_guessed():
    """Which backends actually call the gate is a FACT about the tree.

    Only LlamaCppBackend is wired to it today. Claiming Ollama or llama-server
    gate would make the preflight block escalations that do not brick; claiming
    none do would blind it to the brick that llama.cpp causes. The predicate
    must match where the guard is really wired.
    """
    prov, restore = _load_provenance(required=True)
    try:
        assert prov.backend_enforces_provenance("llama_cpp") is True
        assert prov.backend_enforces_provenance("ollama") is False, (
            "Ollama has no provenance gate. Saying it does turns escalations "
            "that do not brick into refusals."
        )
        assert prov.backend_enforces_provenance("llama_server") is False
        assert prov.backend_enforces_provenance(None) is False
    finally:
        restore()


def test_f22_the_default_backend_is_read_from_config(tmp_path):
    loaded, restore = isolate(
        targets={"opti_oignon.security_mode": source("security_mode.py")},
        seeded={},
    )
    mod = loaded["opti_oignon.security_mode"]
    try:
        cfg = tmp_path / "backends.yaml"
        cfg.write_text("default_backend: llama_cpp\n", encoding="utf-8")
        mod._BACKENDS_YAML = cfg
        assert mod._default_backend() == "llama_cpp"

        cfg.write_text("other: 1\n", encoding="utf-8")
        assert mod._default_backend() is None, (
            "no default_backend key must read as None, never as a guess that "
            "could brick or blind the preflight"
        )

        mod._BACKENDS_YAML = tmp_path / "absent.yaml"
        assert mod._default_backend() is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# The boot checklist: ask the question whose answer matters
# ---------------------------------------------------------------------------


def _load_checks(*, blockers=(), required=True, available=True):
    pqc = types.ModuleType("opti_oignon.pqc_signatures")
    pqc.PQC_AVAILABLE = available
    pqc.signing_blockers = lambda: list(blockers)
    pqc.pqc_posture = lambda: {
        "requested": required,
        "required": required,
        "available": available,
        "mechanism": "ML-DSA-65" if available else None,
        "reason": None,
        "degraded": required and not available,
    }
    loaded, restore = isolate(
        targets={"opti_oignon.startup_checks": source("startup_checks.py")},
        seeded={"opti_oignon.pqc_signatures": pqc},
    )
    return loaded["opti_oignon.startup_checks"], restore


def test_f15_a_fortress_with_no_key_is_not_a_green_checklist():
    checks, restore = _load_checks(
        blockers=["no signing keypair on disk (data/.pqc_keypair)"], required=True
    )
    try:
        item = checks._check_pqc_primitive()
        assert item.passed is False, (
            "the primitive resolved and the checklist reported green, on a host "
            "that cannot sign a byte. That is the identical wrong question the "
            "seal asked: whether it RESOLVED, not whether it could be USED."
        )
        assert item.severity != "critical", (
            "and it must NOT be critical. A critical check refuses the boot, and "
            "the boot carries the two endpoints that repair this exact condition. "
            "A check must never take down the exit it is telling you to take."
        )
    finally:
        restore()


def test_f16_the_checklist_names_the_remedy():
    checks, restore = _load_checks(
        blockers=["no signing keypair on disk (data/.pqc_keypair)"], required=True
    )
    try:
        item = checks._check_pqc_primitive()
        assert item.tips, "a refusal that offers no remedy is a wall"
    finally:
        restore()


def test_f17_a_host_that_can_sign_still_passes():
    """Guards the new question against over-reach."""
    checks, restore = _load_checks(blockers=(), required=True)
    try:
        item = checks._check_pqc_primitive()
        assert item.passed is True
        assert "ML-DSA-65" in item.detail
    finally:
        restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
