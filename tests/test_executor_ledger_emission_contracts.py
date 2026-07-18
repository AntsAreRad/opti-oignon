#!/usr/bin/env python3
"""What the execution hub promises about recording exactly one row per call.

The hub's pipeline computes a request's whole measurement story -- token
figures, cache verdicts, admission decisions, timings -- and used to let
it evaporate. The ledger wiring keeps it, and these contracts pin the two
properties the wiring lives or dies by.

One record per call, whichever exit is taken. A plain completion, a
cache hit served before generation, an admission refusal, an offline
enqueue, a mid-stream cancellation, a timeout, a backend error -- each
path leaves exactly one row, labelled with its outcome, carrying a fresh
request id and the figures that were on hand at that exit. The token
fields of a single-turn completion are pinned to the arithmetic of the
estimator fallback, so a drift in what gets measured is a red contract,
not a quiet skew. The counter's label upgrade is pinned from both sides:
an exact-capable counter rewrites the total and its method; a disabled
one is never even asked.

The sink can never touch the chat path. With the ledger module absent
the hub completes as if the wiring did not exist; with a recorder that
throws on every write the hub still completes and still made exactly one
attempt. Observability that can take down the observed path would be
worse than none, and that refusal is asserted here, not assumed.

This suite loads the real hub source -- the first one to do so. The
window seeds the three hard imports (a scripted inference client, a
configuration object, the routing result class) plus the ledger recorder,
and per contract the cache, governor, or network stand-ins; every other
project name is neutralised, so all optional stages report themselves
absent and the pipeline runs its bare shape. Nothing real is reached:
no model, no database, no network.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.executor"
_ABSENT = object()


class _Recorder:
    """Captures every ledger write as raw keyword fields."""

    def __init__(self, error=None):
        self.records = []
        self.error = error

    def record(self, **fields):
        self.records.append(fields)
        if self.error is not None:
            raise self.error
        return True


class _ScriptedOllama:
    """Plays one scripted chat stream and records every call."""

    def __init__(self):
        self.calls = []
        self.stream_factory = lambda: iter(
            [{"message": {"content": "Hello"}}, {"message": {"content": " world"}}]
        )
        self.error = None

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.stream_factory()


def _routing(**overrides):
    fields = {
        "model": "test-model:1b",
        "task_type": "general",
        "temperature": 0.2,
        "prompt_variant": "standard",
        "timeout": 30,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _load(*, ledger=_ABSENT, extra_seeded=None, extra_blocked=()):
    """Load the real hub with a scripted environment.

    ledger        -- a recorder seeded behind ``get_context_ledger``; pass
                     None to block the ledger name entirely (the absence
                     posture). The default seeds a fresh recorder.
    extra_seeded  -- additional project stand-ins for one contract.
    extra_blocked -- additional names whose absence the contract asserts.
    """
    recorder = _Recorder() if ledger is _ABSENT else ledger

    ollama_stub = types.ModuleType("ollama")
    scripted = _ScriptedOllama()
    ollama_stub.chat = scripted.chat

    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_model=lambda *a, **k: "test-model:1b",
        get_temperature=lambda *a, **k: 0.2,
    )

    router = types.ModuleType("opti_oignon.router")

    class RoutingResult:  # noqa: D401 - stand-in only
        pass

    router.RoutingResult = RoutingResult

    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
    }
    blocked = list(extra_blocked)

    if recorder is None:
        blocked.append("opti_oignon.context_ledger")
    else:
        ledger_mod = types.ModuleType("opti_oignon.context_ledger")
        ledger_mod.get_context_ledger = lambda: recorder
        seeded["opti_oignon.context_ledger"] = ledger_mod

    seeded.update(extra_seeded or {})

    had_ollama = "ollama" in sys.modules
    prev_ollama = sys.modules.get("ollama")
    sys.modules["ollama"] = ollama_stub

    loaded, win_restore = isolate(
        targets={_TARGET: source("executor.py")},
        blocked=blocked,
        seeded=seeded,
    )

    def restore():
        win_restore()
        if had_ollama:
            sys.modules["ollama"] = prev_ollama
        else:
            sys.modules.pop("ollama", None)

    return loaded[_TARGET], scripted, recorder, restore


def _drive(gen):
    """Consume a hub generator, returning (chunks, return_value)."""
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        return chunks, stop.value


# ---------------------------------------------------------------------------
# One record per call
# ---------------------------------------------------------------------------

def test_e1_plain_completion_emits_exactly_one_completed_record():
    mod, scripted, recorder, restore = _load()
    try:
        ex = mod.Executor()
        chunks, (refined, response) = _drive(
            ex.execute("What is a monoid?", _routing(), refine=False)
        )
        assert response == "Hello world"
        assert len(recorder.records) == 1
        rec = recorder.records[0]
        assert rec["outcome"] == "completed"
        assert rec["model"] == "test-model:1b"
        assert rec["caller"] == "chat"
        assert rec["conversation_id"] is None
        assert isinstance(rec["request_id"], str) and len(rec["request_id"]) == 32
        assert rec["duration_ms"] >= 0
        assert rec["token_method"] == "estimated"
        assert rec["cache_stored"] is False
        assert rec["retrieval_count"] == 0

        _drive(ex.execute("Another question", _routing(), refine=False))
        assert len(recorder.records) == 2
        assert recorder.records[1]["request_id"] != rec["request_id"]
    finally:
        restore()


def test_e2_single_turn_token_figures_are_the_estimator_fallback_arithmetic():
    mod, scripted, recorder, restore = _load()
    try:
        ex = mod.Executor()
        question = "What is a monoid in category theory?"
        _drive(ex.execute(question, _routing(), refine=False))
        rec = recorder.records[0]
        system_prompt = ex.get_system_prompt("general", "standard")
        expected_system = len(system_prompt) // 4
        expected_user = len(question) // 4
        assert rec["tokens_system"] == expected_system
        assert rec["tokens_user"] == expected_user
        assert rec["tokens_total"] == expected_system + expected_user
    finally:
        restore()


def test_e3_semantic_tier_hit_records_the_match_and_skips_generation():
    entry = SimpleNamespace(
        match_type="semantic",
        similarity=0.93,
        query_hash="deadbeef",
        response="cached answer",
    )
    sc = types.ModuleType("opti_oignon.semantic_cache")
    sc.semantic_cache = SimpleNamespace(
        enabled=True, get=lambda *a, **k: entry
    )
    mod, scripted, recorder, restore = _load(
        extra_seeded={"opti_oignon.semantic_cache": sc}
    )
    try:
        ex = mod.Executor()
        chunks, (refined, response) = _drive(
            ex.execute("What is a monoid?", _routing(), refine=False)
        )
        assert response == "cached answer"
        assert scripted.calls == []
        assert len(recorder.records) == 1
        rec = recorder.records[0]
        assert rec["outcome"] == "cache_hit"
        assert rec["cache_hit"] is True
        assert rec["cache_hit_type"] == "semantic"
        assert abs(rec["cache_similarity"] - 0.93) < 1e-9
        assert isinstance(rec["tokens_total"], int)
    finally:
        restore()


def test_e4_admission_refusal_records_the_decision_figures():
    decision = SimpleNamespace(
        admitted=False,
        action="refuse",
        reason="not enough memory",
        num_ctx=None,
        keep_alive=None,
        conditional_on_eviction=False,
        refusal_payload=lambda: {"message": "refused: not enough memory"},
    )
    gov = types.ModuleType("opti_oignon.resource_governor")
    gov.get_resource_governor = lambda: SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        admit=lambda model, requested_ctx=None, caller="chat", extra_models=None: decision,
    )
    gov.set_active_ticket = lambda decision: None
    gov.clear_active_ticket = lambda: None
    gov.ticket_scope = lambda decision: None
    mod, scripted, recorder, restore = _load(
        extra_seeded={"opti_oignon.resource_governor": gov}
    )
    try:
        ex = mod.Executor()
        chunks, (refined, response) = _drive(
            ex.execute("What is a monoid?", _routing(), refine=False)
        )
        assert response.startswith("[ERR]")
        assert scripted.calls == []
        assert len(recorder.records) == 1
        rec = recorder.records[0]
        assert rec["outcome"] == "governor_refused"
        assert rec["gov_action"] == "refuse"
        assert rec["gov_admitted"] is False
        assert isinstance(rec["gov_requested_ctx"], int)
        assert rec["gov_requested_ctx"] > 4096
        assert rec["gov_reason"] == "not enough memory"
    finally:
        restore()


def test_e5_offline_enqueue_records_its_outcome_without_generation():
    nm = types.ModuleType("opti_oignon.network_manager")
    nm.network_manager = SimpleNamespace(is_online=False)
    sq = types.ModuleType("opti_oignon.sync_queue")
    sq.sync_queue = SimpleNamespace(
        enqueue=lambda query, task_type, model: SimpleNamespace(id="q1")
    )
    mod, scripted, recorder, restore = _load(
        extra_seeded={
            "opti_oignon.network_manager": nm,
            "opti_oignon.sync_queue": sq,
        }
    )
    try:
        ex = mod.Executor()
        chunks, (refined, response) = _drive(
            ex.execute("What is a monoid?", _routing(), refine=False)
        )
        assert "queued" in response
        assert scripted.calls == []
        assert len(recorder.records) == 1
        assert recorder.records[0]["outcome"] == "offline_queued"
    finally:
        restore()


def test_e6_mid_stream_cancellation_is_one_cancelled_record():
    mod, scripted, recorder, restore = _load()
    try:
        ex = mod.Executor()

        def side_effect_stream():
            yield {"message": {"content": "Hi"}}
            ex._cancel_event.set()
            yield {"message": {"content": " more"}}
            yield {"message": {"content": " again"}}

        scripted.stream_factory = side_effect_stream
        chunks, (refined, response) = _drive(
            ex.execute("What is a monoid?", _routing(), refine=False)
        )
        assert "[Generation cancelled]" in response
        assert len(recorder.records) == 1
        assert recorder.records[0]["outcome"] == "cancelled"
    finally:
        restore()


def test_e11_timeout_is_one_timeout_record():
    mod, scripted, recorder, restore = _load()
    try:
        ex = mod.Executor()
        chunks, (refined, response) = _drive(
            ex.execute("What is a monoid?", _routing(timeout=0), refine=False)
        )
        assert "[Timeout reached]" in response
        assert len(recorder.records) == 1
        assert recorder.records[0]["outcome"] == "timeout"
    finally:
        restore()


def test_e12_backend_error_is_one_error_record():
    mod, scripted, recorder, restore = _load()
    try:
        scripted.error = RuntimeError("model exploded")
        ex = mod.Executor()
        chunks, (refined, response) = _drive(
            ex.execute("What is a monoid?", _routing(), refine=False)
        )
        assert "[ERR] Error:" in response
        assert len(recorder.records) == 1
        assert recorder.records[0]["outcome"] == "error"
        assert recorder.records[0]["cache_stored"] is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# The sink can never touch the chat path
# ---------------------------------------------------------------------------

def test_e7_absent_ledger_leaves_the_hub_untouched():
    mod, scripted, recorder, restore = _load(ledger=None)
    try:
        assert mod.CONTEXT_LEDGER_AVAILABLE is False
        ex = mod.Executor()
        chunks, (refined, response) = _drive(
            ex.execute("What is a monoid?", _routing(), refine=False)
        )
        assert response == "Hello world"
        assert refined == "What is a monoid?"
    finally:
        restore()


def test_e8_throwing_recorder_never_reaches_the_caller_and_is_tried_once():
    recorder = _Recorder(error=RuntimeError("ledger disk gone"))
    mod, scripted, seeded_recorder, restore = _load(ledger=recorder)
    try:
        ex = mod.Executor()
        chunks, (refined, response) = _drive(
            ex.execute("What is a monoid?", _routing(), refine=False)
        )
        assert response == "Hello world"
        assert len(recorder.records) == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# The counter's label upgrade, from both sides
# ---------------------------------------------------------------------------

def test_e9_exact_capable_counter_rewrites_the_total_and_its_method():
    tc = types.ModuleType("opti_oignon.token_counter")
    tc.get_token_counter = lambda: SimpleNamespace(
        exact_enabled=True,
        count_messages=lambda messages, model: SimpleNamespace(
            tokens=421, method="exact"
        ),
    )
    mod, scripted, recorder, restore = _load(
        extra_seeded={"opti_oignon.token_counter": tc}
    )
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is a monoid?", _routing(), refine=False))
        rec = recorder.records[0]
        assert rec["token_method"] == "exact"
        assert rec["tokens_total"] == 421
    finally:
        restore()


def test_e10_disabled_counter_is_never_asked_and_the_label_stays_estimated():
    consulted = []

    tc = types.ModuleType("opti_oignon.token_counter")
    tc.get_token_counter = lambda: SimpleNamespace(
        exact_enabled=False,
        count_messages=lambda messages, model: consulted.append(1)
        or SimpleNamespace(tokens=999, method="exact"),
    )
    mod, scripted, recorder, restore = _load(
        extra_seeded={"opti_oignon.token_counter": tc}
    )
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is a monoid?", _routing(), refine=False))
        rec = recorder.records[0]
        assert consulted == [], "a disabled counter was consulted"
        assert rec["token_method"] == "estimated"
        assert isinstance(rec["tokens_total"], int)
    finally:
        restore()
