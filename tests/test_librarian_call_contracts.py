"""Contracts for the librarian's call: a good neighbour to the machine and to the person at it.

The queue asks a model for summaries off the interactive path, in runs: a
burst after the conversation grew, a close the user asked for. Each call is
asked of the resource governor first, as the background it is, and holds
the ticket it was given while the backend answers; between two calls the
run lets the ticket go, so an interactive call is never kept waiting behind
a whole burst. The model stays resident for the run and is let go at its
end. A span larger than the window the call was admitted at is not sent,
and a run spends at most the time ``onion.yaml`` gives it on the model:

  * LA1 -- every call asks the governor first, as the librarian, for the
    context ``onion.yaml`` names; the backend is asked only once admitted,
    on the thread that holds the ticket, and the ticket is let go when the
    backend returns.
  * LA2 -- the governor's defaults and its shipped file name the librarian
    a background caller.
  * LA3 -- a run asks one ticket per call and never holds two: it lets each
    go before it asks the next.
  * LA4 -- a call carries the context its admission priced, and the run's
    residence or the keep-alive the admission sets under pressure.
  * LA5 -- a span too large for the window, its prompt and the answer's
    room counted with the margin, is not sent and no ticket is asked for
    it: the step goes on without the model, counted ``over_window``, and
    the next span that fits is asked.
  * LA6 -- admitted at a smaller context than the span needs, the call is
    not sent and the admission is handed back.
  * LA7 -- a call the governor does not admit is not made, counted
    ``not_admitted``, and the run asks the model no more; a governor that
    raises admits nothing.
  * LA8 -- no call starts once a run's calls have taken its budget: a model
    that answers just under its deadline costs a close its budget and one
    call at most, and the close still empties the Flesh, counted ``spent``.
  * LA9 -- a run keeps the model resident and lets it go at its end, once,
    through the governor's release of the background's guests, and only
    when the model was asked.
  * LA10 -- of two runs on one model, the first to end does not let it go
    while the other holds it; the last one does.
  * LA11 -- under random faults -- an error, a deadline, a refusal, a
    governor that raises -- every ticket is let go once its call returns,
    and every run lets go of the model's residence however it ends.
  * LA12 -- the librarian's shipped model is none of the models the shipped
    routes answer with, and ``onion.yaml`` says so.
  * LA13 -- the window, its margin and the run's budget come from
    ``onion.yaml`` alone: a file that omits one, or sets one out of range,
    is refused by name.
  * LA14 -- OQ34 word for word, with the three refusals of the call: every
    motive of the closed list leaves zero (OQ34 is deselected by name; its
    list had eight motives).
  * LA15 -- with no governor to ask, the call is made as before: no ticket,
    the context of ``onion.yaml``, nothing let go.
  * LA16 -- LB1 word for word, but for the residence: the shipped keep-alive
    holds the model for a run, not for no time at all (LB1 is deselected by
    name; it held the unload after every call that D6 measured as the cost).
  * LA17 -- a call that cannot start, an earlier call to its model still
    hanging, asks the governor for nothing: every ticket admitted is held by
    a call or handed back.
  * LA18 -- an admission whose call can no longer start, another call to
    its model having hung while it waited, is handed back.
  * LA19 -- under random faults -- another call hanging while one waits for
    its admission, a window admitted too small, a ticket refused on entry,
    a backend that fails or outlives its deadline -- every admission is held
    by its call or handed back, once, never both; the same runs find a call
    that drops its admission.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source; the registry is blocked, the backend
and the governor are stand-ins injected by each contract.
"""

import random
import sys
import threading
import time
import types
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
_ONION_YAML = REPO / "opti_oignon" / "config" / "onion.yaml"


def _open():
    loaded, restore = isolate(
        targets={f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _MODULES},
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils", "opti_oignon.resource_governor"),
        packages=("opti_oignon.memory",),
    )
    lib = loaded["opti_oignon.memory.librarian"]
    lib.reset_librarian()
    return lib, loaded, restore


def _config(lib, **over):
    fields = dict(enabled=True, model="fake:1b", keep_alive="5m", min_new_turns=4, temperature=0.0, num_predict=64,
                  num_ctx=4096, window_margin=0.25, run_budget_s=300.0, call_timeout_s=5.0)
    fields.update(over)
    return lib.LibrarianConfig(**fields)


class _Governor:
    """A governor that admits as told, hands out tickets, and records who asked, what was held and let go."""

    def __init__(self, *, admit=True, num_ctx=None, keep_alive=None, raises=None, guest=True):
        self.asked, self.handed_back, self.released, self.threads, self.scoped = [], [], [], [], []
        self.held = self.most_held = 0
        self.admit, self.num_ctx, self.keep_alive, self.raises, self.guest = admit, num_ctx, keep_alive, raises, guest
        self._lock = threading.Lock()

    def admit_or_wait(self, model, requested_ctx=None, caller="benchmark", **_kw):
        self.asked.append((model, requested_ctx, caller))
        if self.raises is not None:
            raise self.raises
        return types.SimpleNamespace(admitted=self.admit, model=model, num_ctx=self.num_ctx or requested_ctx,
                                     keep_alive=self.keep_alive, ticket_id=f"k{len(self.asked)}",
                                     reason="" if self.admit else "background_gate")

    def end_pending_load(self, ticket_id):
        self.handed_back.append(ticket_id)

    def release_guest(self, model):
        self.released.append(model)
        return self.guest


def _governance(gov):
    """The governor's module as the librarian asks it: the governor, and the ticket held on the calling thread."""

    @contextmanager
    def ticket_scope(decision):
        with gov._lock:
            gov.held += 1
            gov.most_held = max(gov.most_held, gov.held)
            gov.threads.append(threading.get_ident())
            gov.scoped.append(decision.ticket_id)
        try:
            yield
        finally:
            with gov._lock:
                gov.held -= 1

    return types.SimpleNamespace(get_resource_governor=lambda: gov, ticket_scope=ticket_scope)


class _Backend:
    """A backend that answers ``answer``, fails as told, and records each call with what was held around it."""

    def __init__(self, answer="", *, gov=None, fail=None, sleep=0.0, clock=None, takes=0.0):
        self.answer, self.gov, self.fail, self.sleep, self.clock, self.takes = answer, gov, fail, sleep, clock, takes
        self.calls = []

    def generate(self, model, messages, options=None, keep_alive="30m", think=False, images=None):
        self.calls.append({"model": model, "options": dict(options or {}), "keep_alive": keep_alive,
                           "thread": threading.get_ident(), "held": None if self.gov is None else self.gov.held})
        if self.clock is not None:
            self.clock.t += self.takes
        if self.sleep:
            time.sleep(self.sleep)
        if self.fail is not None:
            raise self.fail
        return types.SimpleNamespace(content=self.answer or _faithful_of(messages))


def _faithful(turns):
    return " ".join(t["text"] for t in turns)


def _faithful_of(messages):
    import json

    lines = messages[-1]["content"].splitlines()
    return " ".join(json.loads(line)["text"] for line in lines if line.startswith("{") and '"turn"' in line)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _line(i, words=0):
    filler = " ".join(["alongside"] * words)
    return f"Turn {i}: Alice reviewed service {i} on 2026-03-{i % 28 + 1:02d} and service {i} lives on cluster {i}. {filler}".strip()


def _messages(n, first=1, words=0):
    return [{"role": "user" if i % 2 else "assistant", "content": _line(i, words)} for i in range(first, first + n)]


def _gate(loaded, span_turns=2):
    return loaded["opti_oignon.memory.peels"].Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=span_turns)


def _budget(loaded, flesh=1):
    composer = loaded["opti_oignon.memory.composer"]
    return composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=flesh, turn=60)


def _run(lib, config, gov, clock=None):
    return lib._Run(config, governance=_governance(gov), clock=clock)


def _callers(lib, config, backend, run):
    resolve = lambda model: backend  # noqa: E731
    return lib.registry_summarizer(config, resolve=resolve, run=run), lib.registry_reasker(config, resolve=resolve, run=run)


def _count(lib, event, motive):
    return lib.counters().get(event, {}).get(motive, 0)


# ---------------------------------------------------------------------------
# LA1-LA4 -- a ticket per call, held where the backend is asked
# ---------------------------------------------------------------------------
def test_la1_every_call_asks_the_governor_first_and_holds_its_ticket_on_the_calling_thread():
    lib, loaded, restore = _open()
    try:
        gov = _Governor()
        backend = _Backend(gov=gov)
        config = _config(lib)
        summarize, _reask = _callers(lib, config, backend, _run(lib, config, gov))
        summarize([{"turn_id": "t0001", "role": "user", "text": _line(1)}])
        assert gov.asked == [("fake:1b", 4096, "librarian")]
        assert len(backend.calls) == 1 and backend.calls[0]["held"] == 1, "asked while its ticket is held"
        assert gov.threads == [backend.calls[0]["thread"]], "on the thread that asks the backend"
        assert gov.held == 0, "let go once the backend returned"
    finally:
        restore()


def test_la2_the_governor_names_the_librarian_a_background_caller():
    loaded, restore = isolate(
        targets={"opti_oignon.resource_governor": source("resource_governor.py")},
        blocked=("opti_oignon.context_manager", "opti_oignon.emergency_stop"),
        seeded={"opti_oignon.db_utils": types.SimpleNamespace(safe_connect=None)},
    )
    try:
        rg = loaded["opti_oignon.resource_governor"]
        assert rg._DEFAULT_CALLER_CLASSES.get("librarian") == "background"
        shipped = rg.load_config(Path(source("config", "resource_governor.yaml")))
        assert shipped.class_of("librarian") == "background"
    finally:
        restore()


def test_la3_a_run_asks_one_ticket_per_call_and_never_holds_two():
    lib, loaded, restore = _open()
    try:
        gov = _Governor()
        backend = _Backend("A summary that loses every fact.", gov=gov)
        config = _config(lib)
        run = _run(lib, config, gov)
        summarize, reask = _callers(lib, config, backend, run)
        state = lib.state_for("c1")
        state.mirror(_messages(6))
        lib._curation_burst("c1", config=config, summarize=summarize, reask=reask, gate=_gate(loaded),
                            budget=_budget(loaded), run=run)
        assert len(backend.calls) >= 4, "control: several calls in the run"
        assert len(gov.asked) == len(backend.calls), "one ticket per call"
        assert gov.most_held == 1 and gov.held == 0
    finally:
        restore()


def test_la4_a_call_carries_the_context_its_admission_priced_and_the_run_s_residence():
    lib, loaded, restore = _open()
    try:
        config = _config(lib, num_ctx=8192)
        turn = [{"turn_id": "t0001", "role": "user", "text": _line(1)}]
        gov = _Governor(num_ctx=2048)
        backend = _Backend(gov=gov)
        _callers(lib, config, backend, _run(lib, config, gov))[0](turn)
        assert backend.calls[0]["options"]["num_ctx"] == 2048, "the context the admission priced"
        assert backend.calls[0]["keep_alive"] == "5m", "the run's residence"
        pressed = _Governor(keep_alive="30s")
        backend = _Backend(gov=pressed)
        _callers(lib, config, backend, _run(lib, config, pressed))[0](turn)
        assert backend.calls[0]["keep_alive"] == "30s", "the keep-alive the admission sets under pressure"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LA5-LA6 -- the window
# ---------------------------------------------------------------------------
def test_la5_a_span_too_large_for_the_window_is_not_sent_and_the_next_that_fits_is():
    lib, loaded, restore = _open()
    try:
        gov = _Governor()
        backend = _Backend(gov=gov)
        config = _config(lib, num_ctx=1024, num_predict=64)
        summarize, reask = _callers(lib, config, backend, _run(lib, config, gov))
        state = lib.state_for("c1")
        state.mirror(_messages(2, words=400) + _messages(2, first=3))
        first = lib.curate(state, summarize, reask=reask, gate=_gate(loaded), budget=_budget(loaded))
        assert first.evicted and "over_window" in first.refused
        assert gov.asked == [] and backend.calls == [], "no ticket asked, nothing sent"
        assert _count(lib, "refusal", "over_window") == 1
        second = lib.curate(state, summarize, reask=reask, gate=_gate(loaded), budget=_budget(loaded))
        assert second.rung == "accepted" and len(backend.calls) == 1, "the next span that fits is asked"
    finally:
        restore()


def test_la6_admitted_at_a_smaller_context_than_the_span_needs_the_call_is_not_sent_and_handed_back():
    lib, loaded, restore = _open()
    try:
        gov = _Governor(num_ctx=256)
        backend = _Backend(gov=gov)
        config = _config(lib, num_ctx=8192)
        summarize, reask = _callers(lib, config, backend, _run(lib, config, gov))
        state = lib.state_for("c1")
        state.mirror(_messages(2, words=150))
        outcome = lib.curate(state, summarize, reask=reask, gate=_gate(loaded), budget=_budget(loaded))
        assert "over_window" in outcome.refused and backend.calls == []
        assert gov.handed_back == ["k1"], "the load the admission counted on will not happen"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LA7-LA8 -- refused, spent: the run goes on without the model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gov", [_Governor(admit=False), _Governor(raises=RuntimeError("governor down"))],
                         ids=["refused", "raises"])
def test_la7_a_call_the_governor_does_not_admit_is_not_made_and_the_run_asks_no_more(gov):
    lib, loaded, restore = _open()
    try:
        backend = _Backend(gov=gov)
        config = _config(lib)
        run = _run(lib, config, gov)
        summarize, reask = _callers(lib, config, backend, run)
        state = lib.state_for("c1")
        state.mirror(_messages(8))
        steps = lib._curation_burst("c1", config=config, summarize=summarize, reask=reask, gate=_gate(loaded),
                                    budget=_budget(loaded), run=run)
        assert steps >= 3, "control: the burst went on"
        assert backend.calls == [] and len(gov.asked) == 1, "one admission asked, none after it"
        assert _count(lib, "refusal", "not_admitted") == 1 and _count(lib, "burst", "breaker") == 1
    finally:
        restore()


def test_la8_a_model_just_under_its_deadline_costs_a_close_its_budget_and_one_call_at_most():
    lib, loaded, restore = _open()
    try:
        clock, gov = _Clock(), _Governor()
        backend = _Backend("A summary that loses every fact.", gov=gov, clock=clock, takes=100.0)
        config = _config(lib, run_budget_s=250.0, call_timeout_s=120.0)
        run = _run(lib, config, gov, clock=clock)
        summarize, reask = _callers(lib, config, backend, run)
        state = lib.state_for("c1")
        state.mirror(_messages(16))
        closing = lib.close_onion("c1", config=config, summarize=summarize, reask=reask, gate=_gate(loaded), run=run)
        assert len(backend.calls) == 3, "100, 200 and 300 seconds: the fourth call finds the budget spent"
        assert closing.remaining == 0 and closing.evicted == 8, "the close still empties the Flesh"
        assert _count(lib, "refusal", "spent") == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# LA9-LA11 -- residence for the run, let go at its end
# ---------------------------------------------------------------------------
def test_la9_a_run_keeps_the_model_resident_and_lets_it_go_at_its_end_once_when_it_asked():
    lib, loaded, restore = _open()
    try:
        gov = _Governor()
        backend = _Backend(gov=gov)
        config = _config(lib)
        run = _run(lib, config, gov)
        summarize, reask = _callers(lib, config, backend, run)
        state = lib.state_for("c1")
        state.mirror(_messages(6))
        lib._curation_burst("c1", config=config, summarize=summarize, reask=reask, gate=_gate(loaded),
                            budget=_budget(loaded), run=run)
        assert backend.calls and {c["keep_alive"] for c in backend.calls} == {"5m"}
        assert gov.released == ["fake:1b"], "let go once, at the end of the run"
        assert _count(lib, "residence", "released") == 1
        idle = _Governor()
        quiet = _run(lib, config, idle)
        summarize, reask = _callers(lib, config, _Backend(gov=idle), quiet)
        lib.state_for("c2").mirror(_messages(1))
        lib._curation_burst("c2", config=config, summarize=summarize, reask=reask, gate=_gate(loaded),
                            budget=_budget(loaded, flesh=5000), run=quiet)
        assert idle.asked == [] and idle.released == [], "a run that asked nothing lets nothing go"
    finally:
        restore()


def test_la10_of_two_runs_on_one_model_the_last_to_end_lets_it_go():
    lib, loaded, restore = _open()
    try:
        gov = _Governor()
        config = _config(lib)
        first, second = _run(lib, config, gov), _run(lib, config, gov)
        first.hold()
        second.hold()
        _callers(lib, config, _Backend(gov=gov), first)[0]([{"turn_id": "t0001", "role": "user", "text": _line(1)}])
        first.release()
        assert gov.released == [], "the other run still holds the model"
        second.release()
        assert gov.released == ["fake:1b"]
    finally:
        restore()


def _faulty(rng, gov):
    """A backend that answers, fails, or outlives its deadline, at random."""
    pick = rng.random()
    if pick < 0.3:
        return _Backend(gov=gov, fail=ConnectionError("backend down"))
    if pick < 0.5:
        return _Backend(gov=gov, sleep=0.05)
    return _Backend("A summary that loses every fact.", gov=gov)


def test_la11_under_random_faults_every_ticket_and_every_residence_is_let_go():
    lib, loaded, restore = _open()
    try:
        faults = 0
        for seed in range(40):
            rng = random.Random(seed)
            lib.reset_librarian()
            gov = rng.choice([_Governor(), _Governor(), _Governor(admit=False),
                              _Governor(raises=RuntimeError("governor down"))])
            backend = _faulty(rng, gov)
            config = _config(lib, call_timeout_s=0.01)
            run = _run(lib, config, gov)
            summarize, reask = _callers(lib, config, backend, run)
            state = lib.state_for("c1")
            state.mirror(_messages(rng.randint(2, 10)))
            try:
                if rng.random() < 0.5:
                    lib._curation_burst("c1", config=config, summarize=summarize, reask=reask, gate=_gate(loaded),
                                        budget=_budget(loaded), run=run)
                else:
                    lib.close_onion("c1", config=config, summarize=summarize, reask=reask, gate=_gate(loaded), run=run)
            except Exception:  # noqa: BLE001 - a run that ends by an error still lets go
                faults += 1
            deadline = time.monotonic() + 2.0
            while gov.held and time.monotonic() < deadline:
                time.sleep(0.01)
            assert gov.held == 0, f"seed {seed}: a ticket still held"
            assert lib._holders == {}, f"seed {seed}: a residence still held"
        assert faults == 0, "no run raised"
    finally:
        restore()


class _Meanwhile(_Governor):
    """A governor whose admissions meet a fault at random: another call hangs meanwhile, or the window shrinks."""

    def __init__(self, rng, lib, stuck, **kw):
        super().__init__(**kw)
        self.rng, self.lib, self.stuck = rng, lib, stuck

    def admit_or_wait(self, model, requested_ctx=None, caller="benchmark", **kw):
        decision = super().admit_or_wait(model, requested_ctx, caller, **kw)
        pick = self.rng.random()
        if pick < 0.2:
            # While this call waited, another run's call to the model was given up and still hangs.
            other = threading.Thread(target=self.stuck.wait, daemon=True)
            other.start()
            self.lib._hung.setdefault(model, set()).add(other)
        elif pick < 0.35:
            decision.num_ctx = 96  # admitted at a window no span fits
        return decision


def _refusing_entry(gov, rng):
    """The governor's module, whose ticket is refused on entry at random."""
    real = _governance(gov)

    @contextmanager
    def ticket_scope(decision):
        if rng.random() < 0.15:
            raise RuntimeError("the ticket cannot be held")
        with real.ticket_scope(decision):
            yield

    return types.SimpleNamespace(get_resource_governor=real.get_resource_governor, ticket_scope=ticket_scope)


def _admissions(lib, loaded, seeds):
    """Random runs under faults; by name, each admission not held by its call or handed back once."""
    found = []
    for seed in seeds:
        rng = random.Random(seed)
        lib.reset_librarian()
        stuck = threading.Event()
        gov = _Meanwhile(rng, lib, stuck)
        backend = _faulty(rng, gov)
        config = _config(lib, call_timeout_s=0.01)
        run = lib._Run(config, governance=_refusing_entry(gov, rng))
        summarize, reask = _callers(lib, config, backend, run)
        state = lib.state_for("c1")
        state.mirror(_messages(rng.randint(2, 10)))
        try:
            if rng.random() < 0.5:
                lib._curation_burst("c1", config=config, summarize=summarize, reask=reask, gate=_gate(loaded),
                                    budget=_budget(loaded), run=run)
            else:
                lib.close_onion("c1", config=config, summarize=summarize, reask=reask, gate=_gate(loaded), run=run)
        finally:
            stuck.set()
        deadline = time.monotonic() + 2.0
        while gov.held and time.monotonic() < deadline:
            time.sleep(0.01)
        for n in range(1, len(gov.asked) + 1):
            held, back = f"k{n}" in gov.scoped, gov.handed_back.count(f"k{n}")
            if held and back:
                found.append("an admission held and handed back")
            elif not held and not back:
                found.append("an admission neither held nor handed back")
            elif back > 1:
                found.append("an admission handed back twice")
    return found


def test_la19_under_random_faults_every_admission_is_held_or_handed_back_once():
    lib, loaded, restore = _open()
    try:
        assert _admissions(lib, loaded, range(40)) == []
        real = lib._ask
        lib._ask = lambda *a, unheld=None, **k: real(*a, **k)
        try:
            dropped = _admissions(lib, loaded, range(40))
        finally:
            lib._ask = real
        assert "an admission neither held nor handed back" in dropped, "witness: the runs find a dropped admission"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LA12-LA13 -- the shipped files
# ---------------------------------------------------------------------------
def test_la12_the_librarian_s_shipped_model_answers_no_route_and_onion_yaml_says_so():
    import yaml

    onion = yaml.safe_load(_ONION_YAML.read_text(encoding="utf-8"))
    models = yaml.safe_load((REPO / "opti_oignon" / "config" / "models.yaml").read_text(encoding="utf-8"))
    answering = {str(m) for route in models["routing"].values() for m in route.values()}
    answering |= {str(m) for m in models["fallback_order"]}
    assert len(answering) >= 5, "control: the shipped routes name their models"
    assert str(onion["librarian"]["model"]) not in answering
    assert "never one that answers" in _ONION_YAML.read_text(encoding="utf-8"), "the file says why"


@pytest.mark.parametrize("key, bad", [("num_ctx", 0), ("window_margin", 1.5), ("run_budget_s", -1)])
def test_la13_the_window_its_margin_and_the_run_s_budget_come_from_onion_yaml_alone(tmp_path, key, bad):
    lib, loaded, restore = _open()
    try:
        shipped = _ONION_YAML.read_text(encoding="utf-8")
        assert f"\n  {key}:" in shipped, "control: the shipped file states it"
        cfg = lib.load_config()
        assert cfg.validate() == [] and getattr(cfg, key) > 0
        omitted = tmp_path / "omitted.yaml"
        omitted.write_text("\n".join(line for line in shipped.splitlines() if not line.startswith(f"  {key}:")),
                           encoding="utf-8")
        with pytest.raises(lib.LibrarianError, match=key):
            lib.load_config(omitted)
        assert any(key in error for error in replace(cfg, **{key: bad}).validate())
    finally:
        restore()


# ---------------------------------------------------------------------------
# LA14 -- OQ34, with the refusals of the call
# ---------------------------------------------------------------------------
_TYPED = [
    {"role": "user", "origin": "typed", "segments": [],
     "content": "Alice moved the build to Berlin on 2026-03-04. We keep Docker on the build server."},
    {"role": "assistant", "origin": "assistant", "segments": [],
     "content": "Noted: the Berlin build runs 12 jobs a day, a sensible load for that machine. "
                "Bob checks the logs every morning."},
]
_DECISION = "We keep Docker on the build server."
_LOSSY = "Alice moved the build to Berlin on 2026-03-04. The Berlin build runs 12 jobs a day. Bob checks the logs every morning."


def test_la14_every_motive_of_the_closed_list_leaves_zero_the_refusals_of_the_call_included():
    lib, loaded, restore = _open()
    try:
        peels, probes = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.probes"]
        coded = [{"role": "user", "origin": "typed", "segments": [],
                  "content": _DECISION + "\n\n```bash\ndocker compose up -d\n```"}, _TYPED[1]]
        yaml_gate = replace(peels.load_gate(), span_turns=2)
        tiny = _budget(loaded)

        def down(turns):
            raise ConnectionError("backend down")

        def governed(gov, **over):
            config = _config(lib, **over)
            run = _run(lib, config, gov)
            return run, _callers(lib, config, _Backend(_LOSSY, gov=gov), run)[0]

        spent_run, spent = governed(_Governor())
        spent_run.spent = spent_run.config.run_budget_s
        cases = {
            "decision": (_TYPED, lambda turns: _LOSSY),
            "unsupported": (_TYPED, lambda turns: _LOSSY + " Carol approved a budget of 900 euros."),
            "episodic": (_TYPED, lambda turns: _DECISION),
            "code": (coded, lambda turns: _DECISION),
            "novelty": (_TYPED, lambda turns: _DECISION + " Zebras juggle quantum marmalade across violet "
                                              "orchestras beneath crimson lanterns."),
            "length": (_TYPED, lambda turns: " ".join([_faithful(turns)] * 3)),
            "coverage": (_TYPED, lambda turns: _LOSSY),
            "call_failed": (_TYPED, down),
            "not_admitted": (_TYPED, governed(_Governor(admit=False))[1]),
            "over_window": (_TYPED, governed(_Governor(), num_ctx=64)[1]),
            "spent": (_TYPED, spent),
        }
        drawn = probes.generate_probes
        for motive, (turns, summarize) in cases.items():
            if motive == "coverage":
                probes.generate_probes = lambda span, lexicon=None: [p for p in drawn(span, lexicon) if p.kind != "entity"]
            state = lib.state_for(motive)
            state.mirror(turns)
            lib.curate(state, summarize, gate=yaml_gate, budget=tiny, ladder=peels.load_ladder())
            probes.generate_probes = drawn
            assert _count(lib, "refusal", motive) >= 1, f"{motive}: counted when a step meets it"
        assert set(cases) == set(peels.REFUSAL_MOTIVES), "no motive of the closed list is one no step can meet"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LA15-LA16 -- no governor; the shipped residence
# ---------------------------------------------------------------------------
def test_la15_with_no_governor_to_ask_the_call_is_made_as_before():
    lib, loaded, restore = _open()
    try:
        backend = _Backend()
        config = _config(lib, num_ctx=4096)
        run = lib._Run(config)
        summarize = lib.registry_summarizer(config, resolve=lambda model: backend, run=run)
        run.hold()
        summarize([{"turn_id": "t0001", "role": "user", "text": _line(1)}])
        run.release()
        assert len(backend.calls) == 1 and backend.calls[0]["options"]["num_ctx"] == 4096
        assert _count(lib, "residence", "released") == 0, "nothing to let go through"
        assert lib._holders == {}
    finally:
        restore()


def test_la17_a_call_that_cannot_start_asks_the_governor_for_nothing():
    lib, loaded, restore = _open()
    try:
        gov = _Governor()
        config = _config(lib, call_timeout_s=0.05)
        turn = [{"turn_id": "t0001", "role": "user", "text": _line(1)}]
        hung = _Backend(gov=gov, sleep=0.5)
        first = _callers(lib, config, hung, _run(lib, config, gov))[0]
        with pytest.raises(TimeoutError):
            first(turn)
        second = _callers(lib, config, _Backend(gov=gov), _run(lib, config, gov))[0]
        with pytest.raises(TimeoutError, match="has not returned"):
            second(turn)
        assert len(gov.asked) == 1, "the call that could not start asked for no ticket"
        assert {f"k{i}" for i in range(1, len(gov.asked) + 1)} <= set(gov.scoped) | set(gov.handed_back)
    finally:
        deadline = time.monotonic() + 2.0
        while lib._hung and any(t.is_alive() for ts in lib._hung.values() for t in ts) and time.monotonic() < deadline:
            time.sleep(0.01)
        restore()


def test_la18_an_admission_whose_call_can_no_longer_start_is_handed_back():
    lib, loaded, restore = _open()
    stuck = threading.Event()
    try:
        config = _config(lib)

        class _Waiting(_Governor):
            def admit_or_wait(self, model, requested_ctx=None, caller="benchmark", **kw):
                # While this call waits, another run's call to the model is given up and still hangs.
                other = threading.Thread(target=stuck.wait, daemon=True)
                other.start()
                lib._hung.setdefault(model, set()).add(other)
                return super().admit_or_wait(model, requested_ctx, caller, **kw)

        gov = _Waiting()
        backend = _Backend(gov=gov)
        summarize = _callers(lib, config, backend, _run(lib, config, gov))[0]
        with pytest.raises(TimeoutError, match="has not returned"):
            summarize([{"turn_id": "t0001", "role": "user", "text": _line(1)}])
        assert backend.calls == [] and gov.scoped == []
        assert gov.handed_back == ["k1"], "the admission no call holds is handed back"
    finally:
        stuck.set()
        restore()


def test_la16_the_configuration_is_the_yamls_the_onion_is_off_and_the_model_resident_for_a_run():
    import yaml

    lib, loaded, restore = _open()
    try:
        raw = yaml.safe_load(_ONION_YAML.read_text(encoding="utf-8"))
        cfg = lib.load_config()
        assert raw["enabled"] is False and cfg.enabled is False, "the maintainer turns the onion on"
        assert lib.onion_enabled() is False
        assert cfg.model == str(raw["librarian"]["model"])
        assert cfg.keep_alive == str(raw["librarian"]["keep_alive"]) != "0", "resident for a run, let go at its end"
        assert cfg.min_new_turns == int(raw["librarian"]["min_new_turns"]) >= 1
        assert cfg.validate() == []
        assert _config(lib, min_new_turns=0).validate() != []
        assert _config(lib, num_predict=0).validate() != []
        assert lib.onion_enabled(path=REPO / "does-not-exist.yaml") is False, "an unreadable file is off, not on"
    finally:
        restore()
