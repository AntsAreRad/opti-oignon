"""The post-quantum requirement must be asked, not merely computed.

``pqc_required`` answers one question: may this host substitute something else
for a post-quantum signature? Fortress mode says no -- there the signature is a
property of the MODE, not of a policy file, exactly as the socket bind is. The
function computes that answer correctly. Nothing in production ever asked it.

The two consumers each asked a DIFFERENT question instead, and each got a
different wrong answer from it:

  * the backup signer asked ``is_pqc_enabled`` -- configuration alone. With no
    ``pqc`` block in the policy file, that reads False, and the signer returned
    without signing. A backup left the machine unsigned in fortress mode while
    every posture check upstream reported green, because every one of them was
    checking whether the PRIMITIVE resolved, and it had.

  * the model provenance seal asked the same, and on False did something worse
    than nothing: it fell back to a symmetric MAC. A MAC is not a weaker
    signature, it is a different security property -- forgeable by whoever holds
    the shared secret, and unverifiable by anyone who does not. The quiet
    substitution is the precise failure the posture refusal exists to prevent,
    performed by a module that never consulted it.

These contracts pin the consumers, not the predicate. The stand-in poses the
exact situation the estate was in: the primitive is ALIVE, the policy file is
SILENT, and the mode requires a signature. A consumer that reads the policy and
not the requirement fails here. A consumer that over-reaches -- refusing, or
signing, when nothing was ever promised -- fails here too: a refusal nobody
asked for is a denial of service, and the requirement is not a licence for one.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


class _PQCUnavailable(RuntimeError):
    pass


def _pqc_stand_in(
    *,
    required,
    enabled=False,
    available=True,
    keypair_exists=True,
    keypair=(b"PUB", b"PRIV"),
    sign_raises=False,
    load_raises=False,
):
    """Stand in for the whole signing module, refusal included.

    ``required`` and ``enabled`` are deliberately independent. Their disagreement
    -- required and not enabled -- is the situation a silent policy file puts
    every fortress host in, and it is the one the estate could not see.
    """
    module = types.ModuleType("opti_oignon.pqc_signatures")
    module.PQC_AVAILABLE = available
    module.PQC_MECHANISM = "ML-DSA-65" if available else None
    module.PQCUnavailable = _PQCUnavailable
    module.pqc_required = lambda: required
    module.pqc_requested = lambda: enabled
    module.is_pqc_enabled = lambda: bool(available and enabled)
    module.pqc_keypair_exists = lambda path=None: keypair_exists

    def _load(path=None):
        if load_raises:
            raise ValueError("the keypair could not be used")
        return keypair

    module.load_pqc_keypair = _load

    def _sign(payload, private_key):
        if sign_raises:
            raise RuntimeError("the signer failed mid-flight")
        return b"SIGNATURE"

    module.sign_backup = _sign
    module.sign_bytes = _sign
    module.verify_backup = lambda *a, **k: True
    module.verify_bytes = lambda *a, **k: True

    def _assert():
        if required and not available:
            raise _PQCUnavailable(
                "Post-quantum signing was required and no usable mechanism "
                "resolved. Refusing to substitute a symmetric MAC."
            )

    module.assert_pqc_posture = _assert
    module.pqc_posture = lambda: {
        "requested": enabled,
        "required": required,
        "available": available,
        "mechanism": module.PQC_MECHANISM,
        "reason": None,
        "degraded": required and not available,
    }
    return module


# ---------------------------------------------------------------------------
# The backup signer
# ---------------------------------------------------------------------------


def _load_backup_manager(pqc):
    version = types.ModuleType("opti_oignon.__version__")
    version.__version__ = "0.0.0-isolated"

    loaded, restore = isolate(
        targets={"opti_oignon.backup_manager": source("backup_manager.py")},
        seeded={
            "opti_oignon.pqc_signatures": pqc,
            "opti_oignon.__version__": version,
        },
    )
    return loaded["opti_oignon.backup_manager"], restore


def _export(mod):
    mgr = mod.BackupManager()
    return mgr.export_sections(["semantic_cache"])


def test_e1_fortress_signs_even_when_the_policy_file_is_silent():
    """E1 -- required and not enabled: the backup is SIGNED.

    This is the estate's real posture: liboqs resolves, the policy file carries
    no pqc block, and the mode is a fortress. Today the signer reads the policy,
    sees nothing, and returns. The backup leaves unsigned.
    """
    pqc = _pqc_stand_in(required=True, enabled=False)
    mod, restore = _load_backup_manager(pqc)
    try:
        data = _export(mod)
        assert "_pqc_signature" in data, (
            "the mode required a signature and the primitive was there; a "
            "silent policy file must not be able to strip it"
        )
    finally:
        restore()


def test_e2_nothing_promised_stays_a_no_op():
    """E2 -- not required, not enabled: unsigned, and no refusal.

    The requirement is not a licence to refuse. Guards the change against
    over-reach: a machine that never asked for signing must still export.
    """
    pqc = _pqc_stand_in(required=False, enabled=False)
    mod, restore = _load_backup_manager(pqc)
    try:
        data = _export(mod)
        assert "_pqc_signature" not in data
    finally:
        restore()


def test_e3_required_without_a_keypair_refuses():
    """E3 -- required, primitive alive, no key: the export REFUSES."""
    pqc = _pqc_stand_in(required=True, enabled=False, keypair_exists=False)
    mod, restore = _load_backup_manager(pqc)
    try:
        with pytest.raises(RuntimeError):
            _export(mod)
    finally:
        restore()


def test_e4_a_signer_that_fails_mid_flight_refuses_rather_than_degrades():
    """E4 -- required, key present, the signer raises: REFUSAL, not a warning.

    The asymmetry this replaces was absurd. A promise broken by ABSENCE (no key)
    raised. A promise broken by BREAKAGE (the signer blew up) logged a warning
    and exported the document unsigned, under a message that called it valid.
    The second is the more dangerous of the two: an absent key is discoverable,
    a swallowed exception is not.
    """
    pqc = _pqc_stand_in(required=True, enabled=False, sign_raises=True)
    mod, restore = _load_backup_manager(pqc)
    try:
        with pytest.raises(RuntimeError):
            _export(mod)
    finally:
        restore()


def test_e5_a_key_that_cannot_be_loaded_refuses_when_required():
    """E5 -- required and the keypair cannot be used: REFUSAL.

    A keypair minted under a mechanism the host no longer resolves is exactly
    this case, and it is not hypothetical: the rename that killed the primitive
    is what mints the mismatch.
    """
    pqc = _pqc_stand_in(required=True, enabled=False, load_raises=True)
    mod, restore = _load_backup_manager(pqc)
    try:
        with pytest.raises(RuntimeError):
            _export(mod)
    finally:
        restore()


# ---------------------------------------------------------------------------
# The reporting surface
# ---------------------------------------------------------------------------


def test_e10_the_status_report_says_what_the_machine_will_actually_do():
    """E10 -- effective_enabled follows the REQUIREMENT, not the policy file.

    The machine may be right and the dashboard still wrong, and that is its own
    defect. A fortress signs whatever the policy file says; a status endpoint
    that reads the file and reports "off" while the signer signs is a report
    nobody can use. Worse, it will be believed on the day it says "off" and IS
    off -- the credibility is borrowed from the times it was accidentally right.
    """
    loaded, restore = isolate(
        targets={"opti_oignon.pqc_signatures": source("pqc_signatures.py")},
    )
    mod = loaded["opti_oignon.pqc_signatures"]
    try:
        mod.pqc_required = lambda: True  # a fortress, with the policy file mute
        assert mod.get_pqc_status()["effective_enabled"] is True, (
            "the signer will sign; a report that says otherwise is not a report"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# The model provenance seal
# ---------------------------------------------------------------------------


def _load_model_provenance(pqc, hmac_secret=b"K" * 32):
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


def test_e6_fortress_seals_with_a_signature_not_a_mac():
    """E6 -- required and not enabled: the seal scheme is post-quantum.

    Today the seal reads the policy, sees nothing, and reaches for the HMAC
    secret. The manifest is then sealed with a primitive that whoever holds the
    local key can forge -- which is the whole of what a provenance seal is meant
    to prevent.
    """
    pqc = _pqc_stand_in(required=True, enabled=False)
    mod, restore = _load_model_provenance(pqc)
    try:
        keys = mod.resolve_seal_keys()
        assert keys is not None
        assert keys.scheme == mod.SCHEME_PQC, (
            "the mode required a signature; a symmetric MAC is not a weaker "
            "signature, it is a different security property"
        )
    finally:
        restore()


def test_e7_a_required_seal_hands_back_no_keys_rather_than_mac_keys():
    """E7 -- required and the keypair is unusable: NO keys, never MAC keys.

    The fallback is the defect. Degrading from a publicly verifiable signature
    to a forgeable MAC, on a warning line, is the quiet substitution the posture
    refusal was written to make impossible.

    No keys, and not an exception. The two callers already know what to do with
    nothing: sealing refuses, classifying reports it and lets the enforcement
    policy refuse the model. Raising here would trade the defect for a denial of
    service on every model load.
    """
    pqc = _pqc_stand_in(required=True, enabled=False, load_raises=True)
    mod, restore = _load_model_provenance(pqc)
    try:
        assert mod.resolve_seal_keys() is None
    finally:
        restore()


def test_e9_a_keypair_that_yields_nothing_refuses_when_required():
    """E9 -- required, the key file loads and hands back no key material.

    The other road to the same refusal. It exists so the refusal has TWO
    contracts reaching it by different causes: a single site that only one test
    can reach is a site that stops being tested the day that test changes shape.
    """
    pqc = _pqc_stand_in(required=True, enabled=False, keypair=(b"", b""))
    mod, restore = _load_model_provenance(pqc)
    try:
        assert mod.resolve_seal_keys() is None
    finally:
        restore()


def test_e8_a_mac_is_still_right_when_nothing_was_promised():
    """E8 -- not required: the HMAC seal stands. No over-reach."""
    pqc = _pqc_stand_in(required=False, enabled=False)
    mod, restore = _load_model_provenance(pqc)
    try:
        keys = mod.resolve_seal_keys()
        assert keys is not None
        assert keys.scheme == mod.SCHEME_HMAC
    finally:
        restore()


def test_e11_a_fortress_without_a_key_refuses_the_MODEL_not_the_machinery():
    """E11 -- classifying stays pure: it reports, it does not raise.

    This is the contract that stops the fix from becoming the next defect. A
    fortress with no post-quantum keypair must refuse the model -- and it must
    do so as a DECISION, through the enforcement seam, not as an exception
    escaping up the load path. classify_model says of itself: no policy, no
    raise. That is load-bearing. Break it and every model load in a fortress
    dies on an unhandled error, which buys no security and costs the mode.
    """
    pqc = _pqc_stand_in(required=True, enabled=False, load_raises=True)
    mod, restore = _load_model_provenance(pqc)
    try:
        reason, digest = mod.classify_model(
            "/nonexistent/model.gguf", {"entries": {}, "seal": {}}
        )
        assert reason == mod.REASON_KEY_UNAVAILABLE
        assert digest is None
    finally:
        restore()


def test_e12_the_sealing_path_still_refuses_outright(tmp_path):
    """E12 -- the other side of E11: with nothing to sign with, sealing REFUSES.

    Reporting is right for the reader. It would be very wrong for the writer: a
    manifest that records a model without sealing it is a manifest that promises
    nothing, and writing one silently is how provenance decays into decoration.
    """
    pqc = _pqc_stand_in(required=True, enabled=False, load_raises=True)
    mod, restore = _load_model_provenance(pqc)
    try:
        model = tmp_path / "model.gguf"
        model.write_bytes(b"weights")
        with pytest.raises(mod.ProvenanceError):
            mod.record_model(model, manifest_path=tmp_path / "manifest.json")
    finally:
        restore()
