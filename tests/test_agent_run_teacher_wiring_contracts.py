#!/usr/bin/env python3
"""Wiring contracts for the teacher escalation on the agent run driver.

The teacher chokepoints are proven elsewhere: the escalation module never
raises and wraps its failure context as untrusted data, and a teacher
draft reaches the published skill tree only through the gated, sandbox-
tested publish entry. This suite pins WHERE they are triggered from -- the
run driver, after the loop returns -- and that waking them widens nothing:

  * Contract T1 -- EXPLICIT OPT-IN: the escalation path stays dormant
    unless the agent configuration carries an explicit enabled teacher
    flag; an absent or falsy flag means zero teacher activity;
  * Contract T2 -- ESCALATION THROUGH THE PINNED CHOKEPOINT: armed, an
    eligible run consults the escalator's decision with the run's own
    result, then escalates once with the run task, a bounded failure
    context drawn from the result, and a client built for the configured
    teacher model; the guidance surfaces as a run event;
  * Contract T3 -- CANCELLATION SKIPS: a cancelled run never escalates,
    even when armed and otherwise eligible;
  * Contract T4 -- STOP STATE SKIPS, FAIL CLOSED: an engaged emergency
    stop skips the escalation, and an indeterminable stop state (module
    unavailable) skips it too;
  * Contract T5 -- PUBLICATION ONLY THROUGH THE GATED ENTRY: a proposed
    draft reaches publication only through the pinned publish entry,
    carrying the run's own approval gate, sandbox, conversation and
    approval manager -- never a substitute;
  * Contract T6 -- PUBLICATION IS DAILY-ONLY: outside the daily mode the
    draft is never submitted for publication (mirrors the skill tool's
    exposure), while the guidance itself still surfaces;
  * Contract T7 -- THE HOOK NEVER BREAKS THE RUN: a raising escalation
    leaves the run outcome intact (stop reason preserved, run finished);
  * Contract T8 -- SENTINEL: armed, an ineligible run consults the
    decision and honors it: no escalation, no events.

Loads the agent REST facade in isolation under a stand-in package; every
``opti_oignon.*`` entry plus the web-framework entries is snapshotted and
evicted first, and the seeds are deterministic recorders: a loop stub
returning a scripted result, an escalator recorder, a publish recorder, a
controllable configuration and emergency-stop stub. A meta-path guard
refuses any project submodule that was not seeded, so the load behaves
identically whether or not the project is installed. Local-only. Runs
under pytest or the __main__ runner.
"""

import importlib.util
import json
import sys
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the
    test's back -- silently importing live code. This guard sits ahead of
    every finder and refuses the names that were not seeded, so a load
    behaves identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


_FACADE_KEYS = (
    "fastapi", "pydantic",
    "opti_oignon", "opti_oignon.agent", "opti_oignon.agent.loop",
    "opti_oignon.agent.skills", "opti_oignon.agent.tools",
    "opti_oignon.agent.teacher", "opti_oignon.agent.config_loader",
    "opti_oignon.emergency_stop",
    "opti_oignon.api", "opti_oignon.api.routes_agent",
)


class _AgentEvent:
    def __init__(self, kind="", round=0, data=None):
        self.kind = kind
        self.round = round
        self.data = data or {}


def _default_result():
    failed = SimpleNamespace(
        tool_name="read_file", executed=False,
        observation="tool refused: path outside the workspace",
        reason="refused",
    )
    return SimpleNamespace(
        final_text="student attempt text",
        rounds=3,
        stop_reason="error",
        tool_results=[failed],
        verifier=None,
    )


def _load_driver(*, teacher_cfg, result=None, seed_estop=True,
                 estop_stopped=False, config_raises=False,
                 escalate_raises=False, outcome=None, during_run=None):
    """Load the run driver alone against scripted teacher-side siblings.

    Returns ``(module, state, restore)``. ``state`` carries the recorders:
    run_calls, escalators (constructed policies), should_calls,
    escalate_calls, publish_calls, events (decoded run events).
    """
    saved = {k: sys.modules.get(k) for k in _FACADE_KEYS}

    # The keys the window governs are EMPTIED, not merely remembered. A guard on
    # the meta path is consulted only on a cache MISS: a key another module left
    # in sys.modules short-circuits the import machinery before the guard is ever
    # asked -- and pytest imports EVERY test module at collection, long before the
    # first test runs, so a module-level import anywhere in the suite lands here.
    #
    # This clause is why it matters. T4's second face deliberately declines to
    # seed opti_oignon.emergency_stop, so that the driver cannot determine the
    # stop state and must fail CLOSED. A live emergency_stop surviving in the
    # cache answers "not stopped" instead, the driver escalates, and the proof of
    # a fail-closed path silently becomes its opposite. A contract that a polluted
    # cache can invert is not a contract.
    for key in _FACADE_KEYS:
        sys.modules.pop(key, None)

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    state = {
        "run_calls": [],
        "escalators": [],
        "should_calls": [],
        "escalate_calls": [],
        "publish_calls": [],
        "events": [],
    }

    fastapi = types.ModuleType("fastapi")

    class _Router:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def _deco(self, *args, **kwargs):
            def wrap(fn):
                return fn
            return wrap

        get = post = put = delete = websocket = _deco

    fastapi.APIRouter = _Router
    fastapi.HTTPException = type("HTTPException", (Exception,), {})
    fastapi.WebSocket = object
    fastapi.WebSocketDisconnect = type("WebSocketDisconnect", (Exception,), {})
    sys.modules["fastapi"] = fastapi

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    sys.modules["pydantic"] = pydantic

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    agent_pkg = types.ModuleType("opti_oignon.agent")
    agent_pkg.__path__ = []
    sys.modules["opti_oignon.agent"] = agent_pkg
    pkg.agent = agent_pkg

    run_result = result if result is not None else _default_result()

    loop_mod = types.ModuleType("opti_oignon.agent.loop")
    loop_mod.AgentEvent = _AgentEvent

    def _run(**kwargs):
        state["run_calls"].append(kwargs)
        if during_run is not None:
            during_run()
        return run_result

    loop_mod.run = _run
    sys.modules["opti_oignon.agent.loop"] = loop_mod
    agent_pkg.loop = loop_mod

    skills_mod = types.ModuleType("opti_oignon.agent.skills")

    class _Consultation:
        block = ""

    skills_mod.consult_skills = lambda task, registry=None: _Consultation()
    skills_mod.make_manage_skills_handler = (
        lambda **kwargs: (lambda arguments: "")
    )

    def _publish(draft, *, registry=None, approval_fn=None, sandbox=None,
                 conversation_id="", manager=None):
        state["publish_calls"].append({
            "draft": draft,
            "approval_fn": approval_fn,
            "sandbox": sandbox,
            "conversation_id": conversation_id,
            "manager": manager,
        })
        return SimpleNamespace(published=True, reason="published")

    skills_mod.publish_teacher_draft = _publish
    sys.modules["opti_oignon.agent.skills"] = skills_mod
    agent_pkg.skills = skills_mod

    tools_mod = types.ModuleType("opti_oignon.agent.tools")
    tools_mod.TOOL_MANAGE_SKILLS = "manage_skills"

    class _ToolSet:
        def __init__(self):
            self.tool_handlers = {"manage_skills": lambda arguments: ""}

        def native_tools(self):
            return []

    tools_mod.build_tool_set = lambda mode: _ToolSet()
    tools_mod.system_prompt_section_for = lambda mode: "prompt"
    sys.modules["opti_oignon.agent.tools"] = tools_mod
    agent_pkg.tools = tools_mod

    default_outcome = SimpleNamespace(
        escalated=True, reason="escalated", guidance="teacher guidance",
        draft=None, teacher_model="test-teacher",
    )
    esc_outcome = outcome if outcome is not None else default_outcome

    teacher_mod = types.ModuleType("opti_oignon.agent.teacher")

    class _Escalator:
        def __init__(self, teacher_client=None, policy=None):
            state["escalators"].append(policy)
            self._policy = policy

        def should_escalate(self, run_outcome):
            state["should_calls"].append(run_outcome)
            eligible = getattr(run_outcome, "stop_reason", "") in (
                "error", "max_rounds",
            )
            return SimpleNamespace(
                escalate=eligible,
                reason="model_error" if eligible else "no_escalation",
            )

        def escalate(self, task, *, attempts="", observations="",
                     teacher_client=None, on_event=None):
            if escalate_raises:
                raise RuntimeError("teacher exploded")
            state["escalate_calls"].append({
                "task": task,
                "attempts": attempts,
                "observations": observations,
                "client": teacher_client,
            })
            return esc_outcome

    teacher_mod.TeacherEscalator = _Escalator
    sys.modules["opti_oignon.agent.teacher"] = teacher_mod
    agent_pkg.teacher = teacher_mod

    config_mod = types.ModuleType("opti_oignon.agent.config_loader")

    def _get_agent_config():
        if config_raises:
            raise RuntimeError("config unavailable")
        teacher = dict(teacher_cfg or {})
        return SimpleNamespace(
            teacher=teacher,
            teacher_policy=lambda: SimpleNamespace(
                enabled=bool(teacher.get("enabled", True)),
                teacher_model=str(
                    teacher.get("teacher_model", "test-teacher")
                ),
                failure_threshold=2,
            ),
        )

    config_mod.get_agent_config = _get_agent_config
    sys.modules["opti_oignon.agent.config_loader"] = config_mod
    agent_pkg.config_loader = config_mod

    if seed_estop:
        estop = types.ModuleType("opti_oignon.emergency_stop")
        estop.is_stopped = lambda: bool(estop_stopped)
        estop.guard_http = lambda: None
        estop.status = lambda: {"stopped": bool(estop_stopped)}
        sys.modules["opti_oignon.emergency_stop"] = estop
        pkg.emergency_stop = estop

    api_pkg = types.ModuleType("opti_oignon.api")
    api_pkg.__path__ = []
    sys.modules["opti_oignon.api"] = api_pkg
    pkg.api = api_pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.api.routes_agent", _OO / "api" / "routes_agent.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.api.routes_agent"] = mod
    spec.loader.exec_module(mod)
    api_pkg.routes_agent = mod

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, state, restore


def _start_and_join(mod, state, *, mode="daily", approval_fn=None,
                    sandbox=None, approval_manager=None,
                    conversation_id="conv-1", task="fix the failing step"):
    manager = mod.AgentRunManager()
    manager.subscribe(lambda payload: state["events"].append(
        json.loads(payload)
    ))
    launched = manager.start(
        task,
        model_client=object(),
        mode=mode,
        conversation_id=conversation_id,
        sandbox=sandbox if sandbox is not None else object(),
        approval_fn=approval_fn,
        approval_manager=approval_manager,
        include_memory=False,
        consult=False,
    )
    assert launched.get("started") is True, "the run must launch"
    manager.join(timeout=10.0)
    assert manager.is_running() is False, "the run must finish"
    return manager


def _events_of_kind(state, kind):
    return [e for e in state["events"] if e.get("kind") == kind]


# ---------------------------------------------------------------------------
# Contract T1 -- explicit opt-in only
# ---------------------------------------------------------------------------
def test_t1_absent_or_falsy_opt_in_keeps_the_path_dormant():
    for cfg in ({}, {"enabled": False, "teacher_model": "test-teacher"}):
        mod, state, restore = _load_driver(teacher_cfg=cfg)
        try:
            _start_and_join(mod, state)
            assert state["escalators"] == [], (
                "without an explicit enabled flag the escalator must "
                "never be constructed"
            )
            assert state["escalate_calls"] == []
            assert state["publish_calls"] == []
            assert _events_of_kind(state, "teacher_guidance") == []
        finally:
            restore()


# ---------------------------------------------------------------------------
# Contract T2 -- escalation through the pinned chokepoint
# ---------------------------------------------------------------------------
def test_t2_armed_eligible_run_escalates_once_through_the_chokepoint():
    mod, state, restore = _load_driver(
        teacher_cfg={"enabled": True, "teacher_model": "test-teacher"},
    )
    try:
        _start_and_join(mod, state, task="fix the failing step")

        assert len(state["escalators"]) == 1, (
            "the driver must consult the pinned escalator exactly once"
        )
        policy = state["escalators"][0]
        assert getattr(policy, "teacher_model", "") == "test-teacher", (
            "the configured policy must reach the escalator"
        )
        assert len(state["should_calls"]) == 1
        assert getattr(state["should_calls"][0], "stop_reason", "") == (
            "error"
        ), "the decision must see the run's own result"

        assert len(state["escalate_calls"]) == 1
        call = state["escalate_calls"][0]
        assert call["task"] == "fix the failing step"
        assert "student attempt text" in call["attempts"], (
            "the student attempt must reach the teacher as failure context"
        )
        assert "outside the workspace" in call["observations"], (
            "the failed tool observation must reach the teacher"
        )
        assert getattr(call["client"], "_model", None) == "test-teacher", (
            "the teacher client must be built for the configured model"
        )

        guidance = _events_of_kind(state, "teacher_guidance")
        assert len(guidance) == 1, "the guidance must surface as a run event"
        assert guidance[0]["data"].get("escalated") is True
        assert guidance[0]["data"].get("reason") == "escalated"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract T3 -- a cancelled run never escalates
# ---------------------------------------------------------------------------
def test_t3_cancelled_run_skips_escalation():
    holder = {}
    mod, state, restore = _load_driver(
        teacher_cfg={"enabled": True},
        during_run=lambda: holder["manager"].cancel(),
    )
    try:
        manager = mod.AgentRunManager()
        holder["manager"] = manager
        manager.subscribe(lambda payload: state["events"].append(
            json.loads(payload)
        ))
        launched = manager.start(
            "fix the failing step",
            model_client=object(),
            mode="daily",
            conversation_id="conv-1",
            sandbox=object(),
            include_memory=False,
            consult=False,
        )
        assert launched.get("started") is True
        manager.join(timeout=10.0)

        assert state["escalators"] == [], (
            "a cancelled run must never wake the teacher"
        )
        assert state["escalate_calls"] == []
        assert _events_of_kind(state, "teacher_guidance") == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract T4 -- the stop state skips, fail closed
# ---------------------------------------------------------------------------
def test_t4_engaged_or_unavailable_stop_skips_escalation():
    # Face 1: an engaged stop skips.
    mod, state, restore = _load_driver(
        teacher_cfg={"enabled": True}, estop_stopped=True,
    )
    try:
        _start_and_join(mod, state)
        assert state["escalators"] == [], (
            "an engaged emergency stop must skip the escalation"
        )
        assert state["escalate_calls"] == []
    finally:
        restore()

    # Face 2: an indeterminable stop state skips too.
    mod, state, restore = _load_driver(
        teacher_cfg={"enabled": True}, seed_estop=False,
    )
    try:
        _start_and_join(mod, state)
        assert state["escalators"] == [], (
            "an indeterminable stop state must skip the escalation "
            "(fail closed)"
        )
        assert state["escalate_calls"] == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract T5 -- publication only through the gated entry, with the
# run's own bindings
# ---------------------------------------------------------------------------
def test_t5_daily_draft_publishes_only_through_the_gated_entry():
    draft = SimpleNamespace(name="retry-with-backoff", category="general")
    outcome = SimpleNamespace(
        escalated=True, reason="escalated", guidance="guidance",
        draft=draft, teacher_model="test-teacher",
    )
    gate = lambda conversation_id, tool_name, arguments: True  # noqa: E731
    box = object()
    approvals = object()
    mod, state, restore = _load_driver(
        teacher_cfg={"enabled": True}, outcome=outcome,
    )
    try:
        _start_and_join(
            mod, state, mode="daily", approval_fn=gate, sandbox=box,
            approval_manager=approvals, conversation_id="conv-9",
        )

        assert len(state["publish_calls"]) == 1, (
            "the draft must be submitted through the pinned publish entry"
        )
        call = state["publish_calls"][0]
        assert call["draft"] is draft
        assert call["approval_fn"] is gate, (
            "the publication must carry the run's own approval gate"
        )
        assert call["sandbox"] is box, (
            "the publication must carry the run's own sandbox"
        )
        assert call["manager"] is approvals, (
            "the publication must carry the run's own approval manager"
        )
        assert call["conversation_id"] == "conv-9"

        drafts = _events_of_kind(state, "teacher_draft")
        assert len(drafts) == 1
        assert drafts[0]["data"].get("published") is True
        assert drafts[0]["data"].get("name") == "retry-with-backoff"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract T6 -- publication is daily-only
# ---------------------------------------------------------------------------
def test_t6_outside_daily_the_draft_is_never_submitted():
    draft = SimpleNamespace(name="retry-with-backoff", category="general")
    outcome = SimpleNamespace(
        escalated=True, reason="escalated", guidance="guidance",
        draft=draft, teacher_model="test-teacher",
    )
    mod, state, restore = _load_driver(
        teacher_cfg={"enabled": True}, outcome=outcome,
    )
    try:
        _start_and_join(mod, state, mode="bulbe")

        assert state["publish_calls"] == [], (
            "outside the daily mode the draft must never be submitted "
            "for publication"
        )
        assert _events_of_kind(state, "teacher_draft") == []
        assert len(_events_of_kind(state, "teacher_guidance")) == 1, (
            "the guidance itself still surfaces outside daily"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract T7 -- the hook never breaks the run
# ---------------------------------------------------------------------------
def test_t7_a_raising_escalation_leaves_the_run_outcome_intact():
    result = _default_result()
    result.stop_reason = "max_rounds"
    mod, state, restore = _load_driver(
        teacher_cfg={"enabled": True}, result=result, escalate_raises=True,
    )
    try:
        manager = _start_and_join(mod, state)
        assert manager.status().get("stop_reason") == "max_rounds", (
            "a raising escalation must never replace the run's own "
            "stop reason"
        )
        assert state["publish_calls"] == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract T8 -- sentinel: the decision is honored
# ---------------------------------------------------------------------------
def test_t8_armed_ineligible_run_is_left_alone():
    result = _default_result()
    result.stop_reason = "done"
    result.tool_results = []
    mod, state, restore = _load_driver(
        teacher_cfg={"enabled": True}, result=result,
    )
    try:
        _start_and_join(mod, state)
        assert len(state["should_calls"]) == 1, (
            "the driver must consult the decision"
        )
        assert state["escalate_calls"] == [], (
            "a negative decision must be honored: no escalation"
        )
        assert state["events"] == [], "no events for an ineligible run"
        assert state["publish_calls"] == []
    finally:
        restore()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except BaseException:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
