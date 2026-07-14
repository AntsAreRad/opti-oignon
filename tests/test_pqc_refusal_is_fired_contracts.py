#!/usr/bin/env python3
"""Contracts for the two places the post-quantum refusal must actually fire.

``assert_pqc_posture`` was correct and inert. It raised on a broken promise --
signing asked for, primitive absent -- and no production path had ever called
it. A refusal that is armed and never fired is a comment with a docstring.

It belongs at exactly two places, and they answer different questions.

  * THE BOOT. The startup checklist already turns a critical check into a
    refusal to serve, so a degraded posture is a critical check. This is the
    question "may this host run at all", and under Bulbe the answer is no: a
    fortress with no root of trust has nothing left to be a fortress with.
  * THE BACKUP SIGNER. This is where the substitution actually HAPPENS. The
    signer was a documented no-op -- "if PQC is not enabled or keys are not
    available, this is a no-op" -- so a backup left the machine unsigned while
    the caller believed otherwise. A symmetric MAC is not a weaker signature; it
    is a different security property, forgeable by whoever holds the shared
    secret. Silence there is the whole defect.

  * B1 -- a degraded posture is a CRITICAL startup check.
  * B2 -- and the aggregate refuses the boot, naming the check.
  * B3 -- a resolved primitive passes and names the mechanism in force.
  * B4 -- the check that cannot even inspect the module is a WARNING, never a
    refusal: a broken import is a machinery failure, not a security verdict, and
    it must not brick a boot on its own.
  * B5 -- the backup signer REFUSES rather than exporting unsigned.
  * B6 -- and stays silent when nothing was promised.

Local-only. Runs under pytest or the __main__ runner.
"""

import sys
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


class _PQCUnavailable(RuntimeError):
    pass


def _pqc_stand_in(*, available, mechanism=None, reason=None, required=True):
    module = types.ModuleType("opti_oignon.pqc_signatures")
    module.PQC_AVAILABLE = available
    module.PQC_MECHANISM = mechanism
    module.PQC_UNAVAILABLE_REASON = reason
    module.PQCUnavailable = _PQCUnavailable
    module.pqc_required = lambda: required
    module.pqc_requested = lambda: required
    module.is_pqc_enabled = lambda: bool(available and required)
    module.pqc_keypair_exists = lambda path=None: False
    module.load_pqc_keypair = lambda path=None: (b"", b"")
    module.sign_backup = lambda *a, **k: b""
    module.verify_backup = lambda *a, **k: True

    def _assert():
        if required and not available:
            raise _PQCUnavailable(
                "Post-quantum signing was required and no usable mechanism "
                "resolved. Refusing to substitute a symmetric MAC."
            )

    module.assert_pqc_posture = _assert
    module.pqc_posture = lambda: {
        "requested": required,
        "required": required,
        "available": available,
        "mechanism": mechanism,
        "reason": reason,
        "degraded": required and not available,
    }
    return module


# ---------------------------------------------------------------------------
# The boot
# ---------------------------------------------------------------------------


def _load_checks(**posture):
    return isolate(
        targets={"opti_oignon.startup_checks": source("startup_checks.py")},
        seeded={"opti_oignon.pqc_signatures": _pqc_stand_in(**posture)},
    )


def test_b1_a_degraded_posture_is_a_critical_startup_check():
    loaded, restore = _load_checks(
        available=False, required=True,
        reason="liboqs is installed but offers none of ML-DSA-65, Dilithium3.",
    )
    try:
        item = loaded["opti_oignon.startup_checks"]._check_pqc_primitive()
        assert item.passed is False
        assert item.severity == "critical", (
            f"the signature primitive was required and did not resolve, and "
            f"the checklist rates it {item.severity!r}. A warning is served "
            f"anyway: backups leave unsigned and the provenance seal falls "
            f"back to a MAC that whoever holds the key can forge."
        )
        assert "liboqs" in item.detail, (
            "the operator must be able to act on the detail without reading "
            "the source"
        )
        assert item.tips, "a refusal that offers no remedy is a wall"
    finally:
        restore()


def test_b2_the_aggregate_refuses_the_boot_and_names_the_check():
    loaded, restore = _load_checks(available=False, required=True, reason="liboqs absent")
    try:
        checks = loaded["opti_oignon.startup_checks"]
        result = checks.run_startup_checks(force=True)

        assert "pqc_primitive" in [c.name for c in result.checks], (
            "the check exists and the checklist never runs it: a check nobody "
            "calls is a comment"
        )
        assert result.blocked is True
        assert "pqc_primitive" in (result.block_reason or ""), (
            f"the boot is refused and the reason does not name the primitive: "
            f"{result.block_reason!r}"
        )

        raised = None
        try:
            checks.enforce_boot_checks()
        except checks.StartupBlockedError as exc:
            raised = exc
        assert raised is not None, (
            "the aggregate says blocked and the boot guard serves anyway: a "
            "verdict that never reaches the lifespan is decoration"
        )
    finally:
        restore()


def test_b3_a_resolved_primitive_passes_and_names_the_mechanism():
    loaded, restore = _load_checks(available=True, mechanism="ML-DSA-65", required=True)
    try:
        item = loaded["opti_oignon.startup_checks"]._check_pqc_primitive()
        assert item.passed is True
        assert item.severity != "critical"
        assert "ML-DSA-65" in item.detail, (
            "the checklist must state WHICH mechanism is in force. 'PQC: ok' "
            "is exactly what let a dead constant pass for a live primitive"
        )
    finally:
        restore()


def test_b4_an_uninspectable_module_warns_and_never_refuses_the_boot():
    # Nothing seeded for the signing module: the window refuses the name, so the
    # check cannot import it at all. That is a broken tree, not a posture.
    loaded, restore = isolate(
        targets={"opti_oignon.startup_checks": source("startup_checks.py")},
    )
    try:
        checks = loaded["opti_oignon.startup_checks"]
        item = checks._check_pqc_primitive()
        assert item.severity != "critical", (
            "the check could not even import the module it inspects, and it "
            "returned a verdict that REFUSES THE BOOT. A machinery failure is "
            "not a security verdict; a check that cannot run must never brick "
            "a machine on its own."
        )
        assert checks.run_startup_checks(force=True).blocked is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# The backup signer -- where the substitution actually happens
# ---------------------------------------------------------------------------


def _load_backup_manager(**posture):
    return isolate(
        targets={"opti_oignon.backup_manager": source("backup_manager.py")},
        seeded={"opti_oignon.pqc_signatures": _pqc_stand_in(**posture)},
    )


def _sign(manager_module, backup):
    cls = manager_module.BackupManager
    # A bare instance: the signing path touches no state the constructor builds,
    # and constructing a real manager would drag a database in.
    instance = object.__new__(cls)
    return cls._sign_backup_pqc(instance, backup)


def test_b5_the_backup_signer_refuses_rather_than_exporting_unsigned():
    loaded, restore = _load_backup_manager(
        available=False, required=True,
        reason="liboqs is installed but offers nothing usable",
    )
    try:
        backup = {"data": "payload"}
        raised = None
        try:
            _sign(loaded["opti_oignon.backup_manager"], backup)
        except _PQCUnavailable as exc:
            raised = exc
        assert raised is not None, (
            "post-quantum signing was required, the primitive did not resolve, "
            "and the signer returned quietly. The backup leaves this machine "
            "UNSIGNED while the caller believes it is signed -- the exact "
            "broken promise assert_pqc_posture was written to refuse, and it "
            "was never called."
        )
        assert "_pqc_signature" not in backup
    finally:
        restore()


def test_b6_the_backup_signer_is_silent_when_nothing_was_promised():
    loaded, restore = _load_backup_manager(available=False, required=False)
    try:
        backup = {"data": "payload"}
        _sign(loaded["opti_oignon.backup_manager"], backup)  # must not raise
        assert "_pqc_signature" not in backup, (
            "nobody asked for a signature and none was added: correct, and it "
            "must stay a no-op. A refusal here would be a denial of service "
            "the operator never asked for."
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
