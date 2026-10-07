"""Contracts for the host runbook's measurement under the librarian's governed calls.

The runbook asks the librarian's real model over the baseline corpora, a
summary per span, and prints what it measured. Each call of the librarian
now asks the resource governor for a ticket, is not sent when its span does
not fit the window, and belongs to a run with a budget of time on the model.
A measurement is not a run of the queue: its length is set by its corpora.

  * RN1 -- a call the governor does not admit, or that does not fit the
    window, is reported by its name in the report, never raised: the
    measurement goes on and prints what it did measure.
  * RN2 -- every measured call is a run of its own: no budget of a run
    stops the measurement part way.
  * RN3 -- a refused call is no call of the model: the latency the report
    prints is that of the calls that reached it, and none when none did.

Local-only (the public distribution ships no tests). The runbook is loaded
from its file; the onion's modules come through the shared isolation
window, and the model is a stand-in.
"""

import importlib.util
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian", "baseline")


def _open():
    loaded, restore = isolate(
        targets={f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _MODULES},
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils", "opti_oignon.resource_governor"),
        packages=("opti_oignon.memory",),
    )
    lib = loaded["opti_oignon.memory.librarian"]
    lib.reset_librarian()
    return lib, loaded, restore


def _runbook():
    """The runbook script, loaded from its file; the path it adds is taken back."""
    spec = importlib.util.spec_from_file_location("oo_onion_runbook_rn", REPO / "scripts" / "onion_runbook.py")
    module = importlib.util.module_from_spec(spec)
    path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = path
    return module


def _faithful(turns):
    return " ".join(t["text"] for t in turns)


def test_rn1_a_refused_call_is_reported_by_its_name_and_the_measurement_goes_on():
    lib, loaded, restore = _open()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        runbook = _runbook()
        calls = []

        def summarize(turns):
            calls.append(len(turns))
            if len(calls) % 3 == 0:
                raise peels.CallRefused("not_admitted", "the background gate holds")
            if len(calls) % 5 == 0:
                raise peels.CallRefused("over_window", "too long for the window")
            return _faithful(turns)

        config = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="5m", min_new_turns=8,
                                     temperature=0.0, num_predict=64)
        report = runbook._measure(summarize, config)
        assert len(calls) > 5, "control: the measurement asked many times"
        refused = report["refused_calls"]
        assert refused.get("not_admitted", 0) >= 1 and refused.get("over_window", 0) >= 1
        assert report["corpora"] and all("sweep" in entry for entry in report["corpora"].values())
    finally:
        restore()


class _Slow:
    def generate(self, model, messages, options=None, keep_alive="", think=False):
        time.sleep(0.02)
        return types.SimpleNamespace(content="A summary.")


def test_rn2_every_measured_call_is_a_run_of_its_own():
    lib, loaded, restore = _open()
    try:
        runbook = _runbook()
        config = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="5m", min_new_turns=8,
                                     temperature=0.0, num_predict=64, run_budget_s=0.01)
        summarize = runbook._each_call_a_run(config, resolve=lambda model: _Slow())
        turn = [{"turn_id": "t0001", "role": "user", "text": "Alice reviewed service 1."}]
        answers = [summarize(turn) for _ in range(5)]
        assert answers == ["A summary."] * 5, "no budget of a run stopped the measurement"
        assert runbook._each_call_a_run(config, resolve=lambda model: None) is None, "no backend, no summariser"
    finally:
        restore()


def test_rn3_the_latency_printed_is_that_of_the_calls_that_reached_the_model():
    lib, loaded, restore = _open()
    try:
        peels = loaded["opti_oignon.memory.peels"]
        runbook = _runbook()
        config = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="5m", min_new_turns=8,
                                     temperature=0.0, num_predict=64)

        def refused(turns):
            raise peels.CallRefused("not_admitted", "the background gate holds")

        report = runbook._measure(refused, config)
        assert report["refused_calls"]["not_admitted"] >= 2, "control: every call refused"
        assert "summariser_seconds" not in report, "no latency for a model never asked"
        reached = []

        def half(turns):
            if len(reached) % 2:
                reached.append(False)
                raise peels.CallRefused("not_admitted", "the background gate holds")
            reached.append(True)
            return _faithful(turns)

        report = runbook._measure(half, config)
        assert report["summariser_seconds"]["calls"] == reached.count(True)
    finally:
        restore()
