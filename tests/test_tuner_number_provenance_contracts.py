#!/usr/bin/env python3
"""Contracts for the provenance of every number the tuner emits.

The tuner reports token rates. Three different paths produce them and they are
not equally trustworthy: the Ollama path reads counters the server itself
reported, the llama.cpp path derives a token count from a character count and
a prompt rate from a constant multiple of the generation rate, and the mock
path invents both from a formula plus a random term. All three arrived at the
API in the same shape, so once a result was stored a fabricated rate and a
measured one were indistinguishable.

These contracts pin the label, never the rate. They do not assert that any
particular number is correct -- that is machine work, and it is owed. They
assert only that a number carries an honest account of where it came from, and
that the account survives averaging, aggregation into a profile, and the round
trip to disk and across the API boundary.

  * NP1 -- a simulated run says it is simulated.
  * NP2 -- averaging trials keeps the label instead of dropping it.
  * NP3 -- a profile built from simulated results is itself simulated.
  * NP4 -- the label survives serialisation and rehydration.
  * NP5 -- a record written before provenance existed does not claim to be
    measured; it rehydrates as unknown.
  * NP6 -- rates taken from server-reported counters are labelled measured.
  * NP7 -- rates derived from a character count and a constant multiple are
    labelled estimated, not measured.
  * NP8 -- mixing sources aggregates to the least trustworthy one, so a single
    simulated trial cannot be laundered by measured neighbours.
  * NP9 -- the API schema carries the label and defaults to claiming nothing.
  * NP10 -- the label survives the expression the results route actually uses.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window, with the inference backend declared unreachable and proven
so before anything runs: every contract here drives an injected transport, so
a contract that quietly reached the real registry would be testing something
other than what it says it tests.
"""

import sys
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------
def _open():
    """Open the shared window on the tuner and on the API schema module.

    Returns ``(tuner, schemas, restore)``. The schema module imports only
    ``typing`` and ``pydantic``, so it reaches no sibling; the tuner reaches
    the backend registry from exactly one lazy call site, which the block
    below declares unreachable.
    """
    loaded, restore = isolate(
        targets={
            "opti_oignon.auto_tuner": source("auto_tuner.py"),
            "opti_oignon.api.schemas": source("api", "schemas.py"),
        },
        blocked=("opti_oignon.inference_backend",),
        packages=("opti_oignon.api",),
    )
    return (
        loaded["opti_oignon.auto_tuner"],
        loaded["opti_oignon.api.schemas"],
        restore,
    )


def _labelled_benchmark(mod, source_label):
    """A deterministic benchmark whose every result carries ``source_label``."""
    def _bench(params):
        speed = 30.0 + params.get("threads", 4) * 0.1
        return mod.BenchmarkResult(
            params=params,
            tokens_per_second_tg=speed,
            tokens_per_second_pp=speed * 1.5,
            total_time_ms=1.0,
            source=source_label,
        )

    return _bench


# ---------------------------------------------------------------------------
# Injected transports
# ---------------------------------------------------------------------------
class _FakeResponse:
    """Minimal stand-in for a requests response carrying Ollama metadata."""

    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeRequests(types.ModuleType):
    """Stand-in ``requests`` module; records the calls it was handed."""

    def __init__(self, payload):
        super().__init__("requests")
        self._payload = payload
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(self._payload)


class _FakeChatResponse:
    """A backend reply that carries content and no timing metadata at all."""

    def __init__(self, content):
        self.content = content


class _FakeLlamaCppBackend:
    """A llama.cpp stand-in that answers without reporting any counters."""

    def __init__(self, content):
        self._content = content
        self.calls = []

    def generate(self, model=None, messages=None, options=None):
        self.calls.append({
            "model": model, "messages": messages, "options": options,
        })
        return _FakeChatResponse(self._content)


def _install_requests(payload):
    """Install a fake ``requests`` module; returns (fake, restore).

    ``requests`` is not a project name, so the window's guard does not speak
    for it; the suite seeds it here and takes it back off afterwards.
    """
    saved = sys.modules.get("requests")
    fake = _FakeRequests(payload)
    sys.modules["requests"] = fake

    def restore():
        if saved is None:
            sys.modules.pop("requests", None)
        else:
            sys.modules["requests"] = saved

    return fake, restore


# ---------------------------------------------------------------------------
# NP1 -- a simulated run says it is simulated
# ---------------------------------------------------------------------------
def test_np1_a_mock_benchmark_labels_itself_simulated():
    mod, _schemas, restore = _open()
    try:
        result = mod.create_mock_benchmark_fn()({"batch_size": 1024})
        assert result.source == mod.SOURCE_SIMULATED, (
            "a result invented by the mock path is labelled simulated"
        )
        assert result.source != mod.SOURCE_MEASURED, (
            "an invented rate never presents itself as a measurement"
        )
        assert result.to_dict()["source"] == mod.SOURCE_SIMULATED, (
            "the label reaches the serialised form, not just the object"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# NP2 -- averaging trials keeps the label
# ---------------------------------------------------------------------------
def test_np2_averaging_preserves_the_label():
    mod, _schemas, restore = _open()
    try:
        tuner = mod.AutoTuner(
            config=mod.TunerConfig(warmup_runs=0, trials_per_param=3),
            param_space=mod.ParameterSpace(),
            benchmark_fn=_labelled_benchmark(mod, mod.SOURCE_SIMULATED),
        )
        averaged = tuner._run_averaged({"threads": 4})
        assert averaged.tokens_per_second_tg > 0.0, (
            "the averaged result carries a rate, so the label is not vacuous"
        )
        assert averaged.source == mod.SOURCE_SIMULATED, (
            "averaging several trials keeps their provenance"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# NP3 -- a profile built from simulated results is itself simulated
# ---------------------------------------------------------------------------
def test_np3_a_profile_inherits_the_provenance_of_its_results():
    mod, _schemas, restore = _open()
    try:
        tuner = mod.AutoTuner(
            config=mod.TunerConfig(warmup_runs=0, trials_per_param=1),
            param_space=mod.ParameterSpace(),
            benchmark_fn=_labelled_benchmark(mod, mod.SOURCE_SIMULATED),
        )
        profile = tuner.run("stand-in", mod.TunerJob())
        assert profile.best_tg_speed > 0.0, (
            "the profile carries a rate, so the label is not vacuous"
        )
        assert profile.source == mod.SOURCE_SIMULATED, (
            "a profile built from invented results is itself invented"
        )
        assert profile.to_dict()["source"] == mod.SOURCE_SIMULATED, (
            "the label reaches the serialised profile"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# NP4 -- the label survives the round trip to disk
# ---------------------------------------------------------------------------
def test_np4_the_label_survives_serialisation():
    mod, _schemas, restore = _open()
    try:
        profile = mod.TunerProfile(
            model_name="stand-in",
            best_tg_speed=41.0,
            source=mod.SOURCE_ESTIMATED,
        )
        rehydrated = mod.TunerProfile.from_dict(profile.to_dict())
        assert rehydrated.source == mod.SOURCE_ESTIMATED, (
            "a stored profile rehydrates with the provenance it was stored with"
        )
        assert rehydrated.best_tg_speed == profile.best_tg_speed, (
            "the rate survives the same round trip, so the pair stays together"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# NP5 -- a record predating provenance does not claim to be measured
# ---------------------------------------------------------------------------
def test_np5_a_record_without_provenance_rehydrates_as_unknown():
    mod, _schemas, restore = _open()
    try:
        legacy = {
            "model_name": "stand-in",
            "best_params": {"threads": 6},
            "best_tg_speed": 34.0,
            "best_pp_speed": 51.0,
        }
        profile = mod.TunerProfile.from_dict(legacy)
        assert profile.source == mod.SOURCE_UNKNOWN, (
            "a profile written before provenance existed rehydrates as unknown"
        )
        assert profile.source != mod.SOURCE_MEASURED, (
            "an unlabelled record is never promoted to a measurement"
        )
        assert profile.best_tg_speed == 34.0, (
            "the stored rate is preserved; only its standing is withheld"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# NP6 -- server-reported counters are labelled measured
# ---------------------------------------------------------------------------
def test_np6_server_counters_are_labelled_measured():
    mod, _schemas, restore = _open()
    fake, restore_requests = _install_requests({
        "eval_count": 128,
        "eval_duration": 4_000_000_000,
        "prompt_eval_count": 32,
        "prompt_eval_duration": 500_000_000,
        "message": {"content": "stand-in reply"},
    })
    try:
        bench = mod.create_ollama_benchmark_fn("stand-in-model")
        result = bench({"threads": 6, "batch_size": 2048})
        assert fake.calls, (
            "the transport was actually exercised, so the label is not vacuous"
        )
        assert result.error == "", "the stand-in transport answered cleanly"
        assert result.tokens_per_second_tg == 32.0, (
            "the generation rate is the server's own counter, 128 over 4 s"
        )
        assert result.tokens_per_second_pp == 64.0, (
            "the prompt rate is the server's own counter, 32 over 0.5 s"
        )
        assert result.source == mod.SOURCE_MEASURED, (
            "rates read from counters the server reported are measured"
        )
    finally:
        restore_requests()
        restore()


# ---------------------------------------------------------------------------
# NP7 -- a character estimate is labelled estimated, not measured
# ---------------------------------------------------------------------------
def test_np7_a_character_estimate_is_labelled_estimated():
    mod, _schemas, restore = _open()
    try:
        backend = _FakeLlamaCppBackend("x" * 400)
        bench = mod.create_llamacpp_benchmark_fn(
            "stand-in-model", backend=backend,
        )
        result = bench({"threads": 6, "batch_size": 2048})
        assert backend.calls, (
            "the stand-in backend was actually called, so this is not vacuous"
        )
        assert result.error == "", (
            "the estimate path answered without reaching the real registry, "
            "which the window has proven unreachable"
        )
        assert result.tokens_per_second_tg > 0.0, (
            "a rate was produced, so the label describes a real emission"
        )
        assert result.source == mod.SOURCE_ESTIMATED, (
            "a token count derived from characters is an estimate, not a "
            "measurement"
        )
        assert result.source != mod.SOURCE_MEASURED, (
            "a constant multiple never presents itself as a measurement"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# NP8 -- mixing sources aggregates to the least trustworthy
# ---------------------------------------------------------------------------
def test_np8_mixed_provenance_aggregates_to_the_weakest():
    mod, _schemas, restore = _open()
    try:
        seen = {"n": 0}

        def _bench(params):
            seen["n"] += 1
            label = (
                mod.SOURCE_MEASURED if seen["n"] == 1
                else mod.SOURCE_SIMULATED
            )
            return mod.BenchmarkResult(
                params=params,
                tokens_per_second_tg=30.0 + params.get("threads", 4) * 0.1,
                tokens_per_second_pp=45.0,
                total_time_ms=1.0,
                source=label,
            )

        tuner = mod.AutoTuner(
            config=mod.TunerConfig(warmup_runs=0, trials_per_param=1),
            param_space=mod.ParameterSpace(),
            benchmark_fn=_bench,
        )
        profile = tuner.run("stand-in", mod.TunerJob())
        assert seen["n"] > 1, (
            "more than one trial ran, so the mixture was real"
        )
        assert profile.source == mod.SOURCE_SIMULATED, (
            "one invented trial among measured ones makes the profile invented"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# NP9 -- the API schema carries the label and defaults to no claim
# ---------------------------------------------------------------------------
def test_np9_the_api_schema_defaults_to_no_claim():
    _mod, schemas, restore = _open()
    try:
        assert "source" in schemas.TunerProfileSchema.model_fields, (
            "the profile a client receives has somewhere to carry provenance"
        )
        bare = schemas.TunerProfileSchema()
        assert bare.source == "unknown", (
            "a response built without stating provenance claims nothing"
        )
        assert bare.source != "measured", (
            "the default is never a measurement"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# NP10 -- the label survives the exact expression the route uses
# ---------------------------------------------------------------------------
def test_np10_the_label_survives_the_route_expression():
    mod, schemas, restore = _open()
    try:
        profile = mod.TunerProfile(
            model_name="stand-in",
            best_tg_speed=34.0,
            source=mod.SOURCE_SIMULATED,
        )
        # The expression is the one in the results route, verbatim.
        payload = schemas.TunerProfileSchema(**profile.to_dict())
        assert payload.best_tg_speed == 34.0, (
            "the rate crosses the boundary, so the label is not vacuous"
        )
        assert payload.source == mod.SOURCE_SIMULATED, (
            "an invented rate is still labelled invented at the API boundary"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("NP1 mock labels itself simulated", test_np1_a_mock_benchmark_labels_itself_simulated),
        ("NP2 averaging preserves the label", test_np2_averaging_preserves_the_label),
        ("NP3 profile inherits provenance", test_np3_a_profile_inherits_the_provenance_of_its_results),
        ("NP4 label survives serialisation", test_np4_the_label_survives_serialisation),
        ("NP5 unlabelled record is unknown", test_np5_a_record_without_provenance_rehydrates_as_unknown),
        ("NP6 server counters are measured", test_np6_server_counters_are_labelled_measured),
        ("NP7 character estimate is estimated", test_np7_a_character_estimate_is_labelled_estimated),
        ("NP8 mixed provenance takes the weakest", test_np8_mixed_provenance_aggregates_to_the_weakest),
        ("NP9 API schema defaults to no claim", test_np9_the_api_schema_defaults_to_no_claim),
        ("NP10 label survives the route expression", test_np10_the_label_survives_the_route_expression),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
