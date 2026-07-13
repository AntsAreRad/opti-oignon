#!/usr/bin/env python3
"""Contracts for post-quantum mechanism resolution.

The signing primitive is the project's root of trust: provenance manifests,
backup integrity and peer records all rest on it. It was nonetheless carried in
a hardcoded mechanism name, and when the library renamed that mechanism at
standardisation the primitive went out in place -- signing refused, provenance
fell back to a symmetric MAC, and NOT ONE contract noticed, because every suite
that touches post-quantum signing sets the availability flag itself. They prove
the logic on both branches and never once ask the installed library whether the
mechanism exists.

That is the gap these clauses close:

  * Q1 -- the preferred mechanism RESOLVES IN THE INSTALLED LIBRARY. Not in a
    stand-in, not on a flag the test set: in the library the host actually has.
    This is the clause that fails the day a rename lands, and it is the only one
    here that refuses to be told what the answer is;
  * Q2 -- resolution prefers the standardised name, falls back to the legacy one
    for older builds, and never answers with a mechanism the build does not
    offer;
  * Q3 -- a key envelope records the mechanism that ACTUALLY resolved, never the
    declared constant: the two names are different algorithms whose signatures
    do not interverify, so an envelope naming the wrong one is unverifiable;
  * Q4 -- intent and availability stay DISTINCT. Asked-for-and-absent is
    degraded, never "disabled": collapsing the two erases the operator's ask and
    is exactly how a dead primitive passes for an unconfigured one;
  * Q5 -- a broken promise refuses. A symmetric MAC is not a weaker signature,
    it is a different property -- forgeable by anyone holding the secret -- so a
    requested-but-absent primitive raises rather than substituting quietly;
  * Q6 -- the record signer and the backup signer speak of ONE mechanism. Two
    constants drift, and an envelope then claims what it was not signed with.

Q1 alone reaches the live host. The rest run inside the shared isolation window
against a stand-in library, so what they assert about resolution does not depend
on which liboqs the runner happens to carry.

Local-only. Runs under pytest or the __main__ runner.
"""

import sys
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_PQC = "opti_oignon.pqc_signatures"


def _fake_oqs(offered, constructible=None):
    """A stand-in liboqs offering exactly ``offered``.

    ``constructible`` narrows what Signature() will actually build, so a build
    that lists a mechanism it cannot construct is representable -- that is the
    shape a half-broken install has.
    """
    if constructible is None:
        constructible = offered
    module = types.ModuleType("oqs")
    module.get_enabled_sig_mechanisms = lambda: list(offered)

    class _Signature:
        def __init__(self, name, *args):
            if name not in constructible:
                raise ValueError(f"mechanism not enabled: {name}")
            self.name = name

    module.Signature = _Signature
    return module


def _load(oqs_module):
    """Load the signing backend against a stand-in library."""
    seeded = {} if oqs_module is None else {"oqs": oqs_module}
    loaded, restore = isolate(
        targets={_PQC: source("pqc_signatures.py")},
        seeded=seeded,
    )
    return loaded[_PQC], restore


# Q1 -- the live host. The only clause that refuses to be told the answer.


def test_q1_the_preferred_mechanism_resolves_in_the_installed_library():
    try:
        import oqs  # noqa: F401
    except Exception:
        # The library is genuinely optional, and a build whose shared object is
        # missing is as absent as one never installed. Either way there is no
        # primitive to interrogate, and an absence is not a lie. What this
        # clause refuses is a library that IS there and offers nothing usable.
        return

    from opti_oignon.pqc_signatures import (
        _MECHANISM_PREFERENCE,
        PQC_AVAILABLE,
        PQC_MECHANISM,
    )

    offered = list(oqs.get_enabled_sig_mechanisms())
    assert PQC_MECHANISM is not None, (
        "liboqs is installed and offers "
        f"{[m for m in offered if 'DSA' in m or 'ilithium' in m]}, but none of "
        f"the preferred mechanisms {list(_MECHANISM_PREFERENCE)} resolved. The "
        "signing primitive is DEAD on this host: record signing refuses and "
        "provenance falls back to a symmetric MAC that anyone holding the key "
        "can forge. A hardcoded mechanism name does not survive a rename."
    )
    assert PQC_MECHANISM in offered, (
        f"resolved {PQC_MECHANISM!r}, which this liboqs does not offer"
    )
    assert PQC_AVAILABLE is True, (
        "a mechanism resolved but the backend still reports unavailable"
    )


# Q2 -- preference, fallback, and never a mechanism the build lacks


def test_q2_resolution_prefers_the_standard_name_and_falls_back():
    mod, restore = _load(_fake_oqs(["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]))
    try:
        assert mod.PQC_MECHANISM == "ML-DSA-65", (
            "the standardised name must win when the build offers it"
        )
        assert mod.PQC_AVAILABLE is True
    finally:
        restore()

    mod, restore = _load(_fake_oqs(["Dilithium2", "Dilithium3", "Dilithium5"]))
    try:
        assert mod.PQC_MECHANISM == "Dilithium3", (
            "an older build offering only the legacy name must still work: the "
            "fallback is what keeps existing hosts signing"
        )
        assert mod.PQC_AVAILABLE is True
    finally:
        restore()

    mod, restore = _load(_fake_oqs(["Falcon-512", "SPHINCS+-SHA2-128f-simple"]))
    try:
        assert mod.PQC_MECHANISM is None, (
            "a build offering neither preferred mechanism must resolve to "
            "nothing, never to a name it cannot construct"
        )
        assert mod.PQC_AVAILABLE is False, (
            "a dead primitive must not report itself available"
        )
    finally:
        restore()


# Q3 -- the envelope names the mechanism that actually resolved


def test_q3_the_key_envelope_records_the_resolved_mechanism(tmp_path):
    mod, restore = _load(_fake_oqs(["Dilithium3"]))
    try:
        assert mod._PQC_ALGORITHM == "Dilithium3", (
            "every call site -- keygen, sign, verify, envelope -- must speak of "
            "the mechanism that resolved, not the one that was preferred"
        )
        path = mod.save_pqc_keypair(b"pub", b"priv", path=tmp_path / "kp")
        import json

        envelope = json.loads(path.read_text())
        assert envelope["algorithm"] == "Dilithium3", (
            "an envelope naming a mechanism it was not signed with is "
            "unverifiable: the two names are different algorithms and their "
            "signatures do not interverify"
        )
    finally:
        restore()


# Q4 -- intent and availability are not the same boolean


def test_q4_a_requested_but_absent_primitive_reports_degraded():
    mod, restore = _load(_fake_oqs(["Falcon-512"]))  # nothing usable
    try:
        mod._load_pqc_config = lambda: {"backup_signatures": True}

        assert mod.pqc_requested() is True, (
            "the operator's ask is read from configuration alone; runtime "
            "availability must not erase it"
        )
        assert mod.PQC_AVAILABLE is False
        assert mod.is_pqc_enabled() is False

        posture = mod.pqc_posture()
        assert posture["degraded"] is True, (
            "asked for and absent is a BROKEN PROMISE, not a configuration. A "
            "single boolean that answers False to both is how a dead primitive "
            "passes for an unconfigured one -- which is precisely what happened"
        )
        assert posture["mechanism"] is None
    finally:
        restore()

    mod, restore = _load(_fake_oqs(["ML-DSA-65"]))
    try:
        mod._load_pqc_config = lambda: {"backup_signatures": False}
        posture = mod.pqc_posture()
        assert posture["degraded"] is False, (
            "not asked for is not degraded: an unconfigured primitive is a "
            "choice, and it must not raise a false alarm"
        )
    finally:
        restore()


# Q5 -- a broken promise refuses rather than substituting


def test_q5_a_broken_promise_refuses_and_never_substitutes():
    mod, restore = _load(_fake_oqs(["Falcon-512"]))
    try:
        mod._load_pqc_config = lambda: {"backup_signatures": True}
        raised = None
        try:
            mod.assert_pqc_posture()
        except mod.PQCUnavailable as exc:
            raised = exc
        assert raised is not None, (
            "a symmetric MAC is not a weaker signature, it is a different "
            "property: forgeable by anyone holding the secret. Asked for and "
            "absent must refuse, never substitute quietly"
        )
    finally:
        restore()

    mod, restore = _load(_fake_oqs(["ML-DSA-65"]))
    try:
        mod._load_pqc_config = lambda: {"backup_signatures": True}
        mod.assert_pqc_posture()  # asked for AND available: silent
        mod._load_pqc_config = lambda: {"backup_signatures": False}
        mod.assert_pqc_posture()  # not asked for: silent
    finally:
        restore()


# Q6 -- one mechanism, not two constants


def test_q6_the_record_signer_and_the_backup_signer_agree():
    # Loaded through the window rather than out of the ambient cache. Two
    # neighbouring suites write PQC_AVAILABLE onto the REAL backend module and
    # never put it back, so a clause that reads the cache reads whatever ran
    # before it. What this clause asserts is a property of the code, not of the
    # execution order, and it says so by manufacturing its own package.
    records = types.ModuleType("opti_oignon.veilid.records")
    records.SyncRecord = object
    records.canonical_record_bytes = lambda *args, **kwargs: b""

    loaded, restore = isolate(
        targets={
            _PQC: source("pqc_signatures.py"),
            "opti_oignon.veilid.signing": source("veilid", "signing.py"),
        },
        seeded={
            "oqs": _fake_oqs(["ML-DSA-65"]),
            "opti_oignon.veilid.records": records,
        },
        packages=("opti_oignon.veilid",),
    )
    try:
        pqc = loaded[_PQC]
        signing = loaded["opti_oignon.veilid.signing"]

        assert signing.SIGNING_ALGORITHM in pqc._MECHANISM_PREFERENCE, (
            "the declared record-signing mechanism must be one the backend "
            "knows how to resolve; a constant nobody resolves is a constant "
            "that dies unnoticed"
        )
        assert pqc.PQC_MECHANISM == "ML-DSA-65"
        assert signing._resolved_algorithm() == pqc.PQC_MECHANISM, (
            "two constants drift, and an envelope then claims a mechanism it "
            "was not signed with. There is ONE resolved mechanism, and the "
            "record signer reads it rather than restating it"
        )
    finally:
        restore()


def _main():
    import tempfile

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                if fn.__code__.co_argcount:
                    fn(Path(tmp))
                else:
                    fn()
                print(f"PASS {fn.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {fn.__name__}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
