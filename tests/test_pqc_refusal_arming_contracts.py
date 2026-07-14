#!/usr/bin/env python3
"""Contracts for the arming of the post-quantum refusal.

The backend already knows how to refuse. ``assert_pqc_posture`` raises when the
primitive was asked for and did not resolve, and it says exactly why a symmetric
MAC is not a weaker signature but a different security property. That refusal is
correct, and until now it was never fired: nothing in production called it, and
three separate paths could disarm it before it was ever reached.

  * The configuration read that GATES the refusal failed open. A security.yaml
    that is present but unreadable returned an empty mapping, so the intent read
    False, so the refusal never triggered. A disk error was enough to turn the
    fortress into a machine that quietly substitutes a forgeable MAC.
  * The mode did not enter the question at all. Bulbe is a physical constraint
    rather than a policy -- the socket bind is not configurable under Bulbe, it
    is physical -- and a fortress does not ask politely for its root of trust in
    a config file.
  * The posture carried no REASON. liboqs absent and liboqs offering nothing
    produced an identical report, and the operator could not act on either.

These clauses arm it, and pin the symmetric half just as hard: a primitive
nobody asked for, on a host that is not a fortress, must raise no alarm at all.
An over-eager refusal is a denial of service the operator did not ask for, and
it is how a fail-closed control gets switched off for good.

Local-only. Runs under pytest or the __main__ runner.
"""

import sys
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_PQC = "opti_oignon.pqc_signatures"

_LIVE = ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87")
_DEAD = ("Falcon-512",)  # a library that offers none of the preferred names


def _oqs(offered):
    module = types.ModuleType("oqs")
    module.get_enabled_sig_mechanisms = lambda: list(offered)

    class _Signature:
        def __init__(self, name, *args):
            if name not in offered:
                raise ValueError(f"mechanism not enabled: {name}")
            self.name = name

    module.Signature = _Signature
    return module


def _mode(mode):
    module = types.ModuleType("opti_oignon.security_mode")
    module.get_current_mode = lambda: mode
    return module


def _load(*, offered=_LIVE, mode="daily"):
    """Load the backend against a stand-in library and a driven mode."""
    seeded = {"opti_oignon.security_mode": _mode(mode)}
    seeded["oqs"] = _oqs(offered) if offered is not None else None
    return isolate(targets={_PQC: source("pqc_signatures.py")}, seeded=seeded)


# ---------------------------------------------------------------------------
# R1 -- an unreadable configuration cannot disarm the refusal
# ---------------------------------------------------------------------------


def test_r1_an_unreadable_configuration_reads_as_requested():
    loaded, restore = _load(offered=_DEAD, mode="daily")
    try:
        pqc = loaded[_PQC]
        # None is what _load_pqc_config returns for a file that is PRESENT and
        # cannot be parsed -- distinct from {} for a file that is absent.
        pqc._load_pqc_config = lambda: None

        assert pqc.pqc_requested() is True, (
            "the configuration could not be read, so the operator's intent is "
            "UNKNOWN -- and the empty mapping the old code returned made it "
            "read as 'never asked for'. A config read that fails open disarms "
            "the very refusal it gates: a disk error was enough to let a "
            "forgeable MAC be substituted for a signature, in silence."
        )
        assert pqc.pqc_posture()["degraded"] is True
        raised = None
        try:
            pqc.assert_pqc_posture()
        except Exception as exc:  # noqa: BLE001 - the type is the module's own
            raised = exc
        assert raised is not None, (
            "an unreadable policy on a host with no signing primitive must "
            "refuse, not proceed"
        )
    finally:
        restore()


def test_r1b_an_absent_configuration_is_a_default_not_an_unknown():
    loaded, restore = _load(offered=_DEAD, mode="daily")
    try:
        pqc = loaded[_PQC]
        pqc._load_pqc_config = lambda: {}  # absent file: a real default

        assert pqc.pqc_requested() is False, (
            "an ABSENT security.yaml is a choice with a documented default. "
            "Only a file that is present and unreadable is an unknown, and "
            "conflating the two turns every default install into an alarm."
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# R2, R3 -- Bulbe requires the primitive, whatever the configuration says
# ---------------------------------------------------------------------------


def test_r2_bulbe_requires_the_primitive_even_when_config_declines():
    loaded, restore = _load(offered=_DEAD, mode="bulbe")
    try:
        pqc = loaded[_PQC]
        pqc._load_pqc_config = lambda: {"backup_signatures": False}

        assert pqc.pqc_requested() is False, (
            "intent is read from configuration ALONE and stays honest: the "
            "operator did not ask"
        )
        assert pqc.pqc_required() is True, (
            "Bulbe is a physical constraint, not a policy. The socket bind is "
            "not configurable under Bulbe; it is physical. The root of trust "
            "cannot be less than that. A fortress does not ask politely for it "
            "in a config file -- whoever wants to run without the primitive "
            "runs Daily."
        )
        assert pqc.pqc_posture()["degraded"] is True
    finally:
        restore()


def test_r3_a_dead_primitive_under_bulbe_refuses():
    loaded, restore = _load(offered=_DEAD, mode="bulbe")
    try:
        pqc = loaded[_PQC]
        pqc._load_pqc_config = lambda: {"backup_signatures": False}
        raised = None
        try:
            pqc.assert_pqc_posture()
        except Exception as exc:  # noqa: BLE001
            raised = exc
        assert raised is not None, (
            "the fortress has no signature primitive and proceeds anyway. "
            "Backups leave unsigned and the provenance seal falls back to a "
            "MAC that anyone holding the key can forge."
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# R4 -- the SYMMETRIC half: no alarm where none is owed
# ---------------------------------------------------------------------------


def test_r4_daily_and_unasked_raises_no_alarm():
    loaded, restore = _load(offered=_DEAD, mode="daily")
    try:
        pqc = loaded[_PQC]
        pqc._load_pqc_config = lambda: {"backup_signatures": False}

        assert pqc.pqc_required() is False
        assert pqc.pqc_posture()["degraded"] is False, (
            "not asked for, and not a fortress: an unconfigured primitive is a "
            "choice. A refusal here is a denial of service the operator never "
            "asked for, and it is how a fail-closed control gets turned off "
            "for good."
        )
        pqc.assert_pqc_posture()  # must be silent
    finally:
        restore()


# ---------------------------------------------------------------------------
# R5 -- the posture carries an actionable REASON
# ---------------------------------------------------------------------------


def test_r5_the_posture_says_why_and_distinguishes_the_two_deaths():
    absent, restore_a = _load(offered=None, mode="daily")
    try:
        reason_absent = absent[_PQC].pqc_posture()["reason"]
        assert reason_absent, (
            "liboqs is not installed and the posture records no reason. The "
            "operator sees 'available: false' and cannot tell a missing "
            "optional package from a signing primitive that died in place."
        )
        assert "liboqs" in reason_absent
    finally:
        restore_a()

    offers_nothing, restore_b = _load(offered=_DEAD, mode="daily")
    try:
        pqc = offers_nothing[_PQC]
        reason_dead = pqc.pqc_posture()["reason"]
        assert reason_dead, "a library that offers nothing usable is a reason too"
        assert "Falcon-512" in reason_dead and "ML-DSA-65" in reason_dead, (
            f"the reason must name what was looked for AND what the host "
            f"actually offers, or it cannot be acted on: {reason_dead!r}"
        )
        assert reason_dead != reason_absent, (
            "an absent library and a library that offers nothing are different "
            "situations with different remedies, and they must not report the "
            "same thing"
        )
    finally:
        restore_b()


# ---------------------------------------------------------------------------
# R6 -- a dead primitive declares NO mechanism
# ---------------------------------------------------------------------------


def test_r6_a_dead_primitive_names_no_mechanism():
    loaded, restore = _load(offered=_DEAD, mode="daily")
    try:
        pqc = loaded[_PQC]
        assert pqc.PQC_MECHANISM is None
        assert pqc._PQC_ALGORITHM is None, (
            "nothing resolved, and the module still names ML-DSA-65 -- the "
            "mechanism that was HOPED for. The status surface then reports an "
            "algorithm this host cannot honour, which is the exact way a dead "
            "root of trust goes on looking alive."
        )
        pqc._load_pqc_config = lambda: {}
        assert pqc.get_pqc_status()["algorithm"] is None
        assert pqc.get_pqc_status()["reason"], (
            "the status surface must carry the reason to the operator"
        )
    finally:
        restore()


def test_r6b_a_live_primitive_names_the_mechanism_that_resolved():
    loaded, restore = _load(offered=("Dilithium3",), mode="daily")
    try:
        pqc = loaded[_PQC]
        assert pqc._PQC_ALGORITHM == "Dilithium3", (
            "the fallback must still resolve; refusing to name a hoped-for "
            "mechanism must not mean refusing to name a resolved one"
        )
    finally:
        restore()


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
