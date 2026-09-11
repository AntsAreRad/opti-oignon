#!/usr/bin/env python3
"""Contracts that an acceptance rate is reported only once it exists.

Speculative decoding is worth having only if the fraction of drafted tokens
the target accepts can be seen. The apparatus for seeing it is complete here:
a per-run record, a bounded history, persistence, a rolling window, an
endpoint. What it never had is a producer. The telemetry path carries a
``speculative_data`` parameter that no caller passes, and the log-parsing path
has no caller at all -- so ``total_draft_tokens`` stays at zero and the rate is
reported as ``0.0``.

That zero is the defect. It is indistinguishable from a target that accepted
none of the tokens drafted for it, which is a real and very different thing:
one says the feature was never exercised, the other says it was exercised and
is worthless. A number nobody produced must not be spelled the same way as a
number somebody measured.

  * AR1 -- a rate nobody produced is unknown, not zero.
  * AR2 -- the probe is proven able to answer: a recorded run yields a real
    positive rate, so AR1's unknown is an absence and not an inert reader.
  * AR3 -- a MEASURED zero stays zero. This is the whole point: drafting ten
    tokens and having none accepted is a result, and it must not be confused
    with never having drafted at all.
  * AR4 -- the rolling window follows the same rule as the overall rate.
  * AR5 -- the distinction survives serialisation; an unknown rate does not
    become a rounded zero on its way out.
  * AR6 -- the log line is a real producer: a llama-server acceptance line
    fed to the parser records a run and moves the rate off unknown.
  * AR7 -- availability is measured or unknown, never a literal. The status
    reported ``"available": True`` unconditionally, next to the name of the
    backend it requires, with no backend, no model and no card.

Nothing here measures a real acceptance rate, and none is claimed: that needs
a target and a draft model on the host, and it stays owed. Every producer in
this file is a log line or a recorded run handed in by the test.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; every stats file lives under pytest's tmp_path, never the
repository's data directory.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _open():
    """Open the shared window on the speculative-decoding module."""
    loaded, restore = isolate(
        targets={
            "opti_oignon.speculative_decoding": source(
                "speculative_decoding.py",
            ),
        },
        blocked=("opti_oignon.inference_backend",),
    )
    return loaded["opti_oignon.speculative_decoding"], restore


def _manager(sd, tmp_path, probe=None):
    """A manager whose stats live in a throwaway file."""
    return sd.SpeculativeDecodingManager(
        stats_path=str(tmp_path / "acceptance.json"),
        availability_probe=probe,
    )


# ---------------------------------------------------------------------------
# AR1 -- a rate nobody produced is unknown
# ---------------------------------------------------------------------------
def test_ar1_an_unproduced_rate_is_unknown():
    sd, restore = _open()
    try:
        stats = sd.AcceptanceStats()
        assert stats.total_draft_tokens == 0, (
            "nothing has been drafted, which is the state under test"
        )
        assert stats.overall_acceptance_rate is None, (
            "a rate nobody produced is unknown, not zero"
        )
        assert stats.overall_acceptance_rate != 0.0, (
            "and is not spelled the same way as a measured zero"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AR2 -- the reader is proven able to answer
# ---------------------------------------------------------------------------
def test_ar2_a_recorded_run_yields_a_real_rate():
    sd, restore = _open()
    try:
        stats = sd.AcceptanceStats()
        stats.record_run(draft_tokens=16, accepted=12, speedup=2.0)
        rate = stats.overall_acceptance_rate
        assert rate is not None, "a produced rate is not unknown"
        assert abs(rate - 0.75) < 1e-9, (
            "twelve of sixteen is the rate, so the reader really reads"
        )
        assert stats.total_runs == 1, "the run was counted"
    finally:
        restore()


# ---------------------------------------------------------------------------
# AR3 -- a measured zero stays zero
# ---------------------------------------------------------------------------
def test_ar3_a_measured_zero_is_not_an_unknown():
    sd, restore = _open()
    try:
        stats = sd.AcceptanceStats()
        stats.record_run(draft_tokens=10, accepted=0)
        rate = stats.overall_acceptance_rate
        assert rate is not None, (
            "ten tokens were drafted and none accepted: that is a result, "
            "not an absence of one"
        )
        assert rate == 0.0, "and the result is zero"
        assert stats.last_acceptance_rate == 0.0, (
            "the same holds for the last run's own rate"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AR4 -- the rolling window follows the same rule
# ---------------------------------------------------------------------------
def test_ar4_the_rolling_window_follows_the_same_rule():
    sd, restore = _open()
    try:
        stats = sd.AcceptanceStats()
        assert stats.get_rolling_acceptance_rate(10) is None, (
            "an empty window has no rate to give"
        )
        stats.record_run(draft_tokens=8, accepted=4)
        rolling = stats.get_rolling_acceptance_rate(10)
        assert rolling is not None, "a window with a run has a rate"
        assert abs(rolling - 0.5) < 1e-9, "and it is the rate of that run"
    finally:
        restore()


# ---------------------------------------------------------------------------
# AR5 -- the distinction survives serialisation
# ---------------------------------------------------------------------------
def test_ar5_an_unknown_rate_does_not_serialise_as_zero():
    sd, restore = _open()
    try:
        empty = sd.AcceptanceStats().to_dict()
        assert empty["overall_acceptance_rate"] is None, (
            "the unknown reaches the serialised form as an unknown"
        )
        assert empty["rolling_acceptance_rate"] is None, (
            "and so does the rolling one"
        )

        stats = sd.AcceptanceStats()
        stats.record_run(draft_tokens=4, accepted=1)
        produced = stats.to_dict()
        assert produced["overall_acceptance_rate"] == 0.25, (
            "a produced rate still serialises as a number, so the None above "
            "is the absence and not the field being broken"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AR6 -- a log line is a real producer
# ---------------------------------------------------------------------------
def test_ar6_a_log_line_produces_a_rate(tmp_path):
    sd, restore = _open()
    try:
        mgr = _manager(sd, tmp_path)
        status = mgr.get_status()
        assert status["stats"]["overall_acceptance_rate"] is None, (
            "the manager starts with nothing produced"
        )

        took = mgr.process_log_line(
            "speculative: accepted 12, drafted 16"
        )
        assert took is True, "the line carried acceptance data and was taken"

        after = mgr.get_status()["stats"]
        assert after["overall_acceptance_rate"] == 0.75, (
            "the rate now exists and is the one the line reported"
        )
        assert after["total_runs"] == 1, "the run was recorded"

        ignored = mgr.process_log_line("just an ordinary server line")
        assert ignored is False, (
            "a line carrying nothing is refused rather than recorded as a "
            "run of zero, which would poison the rate it is meant to report"
        )
        assert mgr.get_status()["stats"]["total_runs"] == 1, (
            "and the count is unchanged by it"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AR7 -- availability is measured or unknown, never a literal
# ---------------------------------------------------------------------------
def test_ar7_availability_is_never_a_literal(tmp_path):
    sd, restore = _open()
    try:
        unknown = _manager(sd, tmp_path, probe=lambda: None).get_status()
        assert unknown["available"] is None, (
            "a probe that cannot tell leaves availability unknown"
        )
        assert unknown["available"] is not True, (
            "and unknown is never reported as available"
        )
        assert unknown["availability_basis"] == "unknown", (
            "the status says on what basis it is answering"
        )

        absent = _manager(sd, tmp_path, probe=lambda: False).get_status()
        assert absent["available"] is False, (
            "a probe that says the backend is absent is believed"
        )
        assert absent["availability_basis"] == "probe", (
            "and the basis names the probe"
        )

        present = _manager(sd, tmp_path, probe=lambda: True).get_status()
        assert present["available"] is True, (
            "a probe that says the backend is there is believed too, so this "
            "contract is about provenance and not about always refusing"
        )
        assert present["availability_basis"] == "probe"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    import tempfile

    simple = [
        ("AR1 unproduced rate is unknown", test_ar1_an_unproduced_rate_is_unknown),
        ("AR2 recorded run yields a real rate", test_ar2_a_recorded_run_yields_a_real_rate),
        ("AR3 measured zero stays zero", test_ar3_a_measured_zero_is_not_an_unknown),
        ("AR4 rolling window follows the rule", test_ar4_the_rolling_window_follows_the_same_rule),
        ("AR5 unknown does not serialise as zero", test_ar5_an_unknown_rate_does_not_serialise_as_zero),
    ]
    with_tmp = [
        ("AR6 a log line produces a rate", test_ar6_a_log_line_produces_a_rate),
        ("AR7 availability is never a literal", test_ar7_availability_is_never_a_literal),
    ]
    passed = 0
    total = len(simple) + len(with_tmp)
    for label, fn in simple:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    for label, fn in with_tmp:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"PASS  {label}")
                passed += 1
            except Exception:  # noqa: BLE001 -- report and continue
                print(f"FAIL  {label}")
                traceback.print_exc()
    print(f"\n{passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
