#!/usr/bin/env python3
"""
Agent eval runner -- S230 (AGT_SPEC Section 7.2).

EvalRunner on the BenchmarkRunner idiom: a background thread, is_busy, one
run at a time, cancel; start_run(models, suite, repeats). Per (model, task,
repeat) the runner creates a FRESH SandboxToolSession (the S73/S74
disposable shape at its purest), materializes the task fixture, drives the
Lot 1/2 agent loop, executes the checks in the same sandbox, and destroys
the session. Nothing persists between tasks except the result row.

Admission (RESOURCE_GOVERNOR_SPEC 4.3 and Section 8, the benchmark-runner
S224/S225 idiom mirrored): when the resource_governor module is importable
and enabled, the runner admits each model (caller "agent_eval") before its
first task; a refusal records every (task, repeat) row as failure_class
"not_admitted" and SKIPS the model -- NEVER a silent downsize, because
silently altered num_ctx poisons the numbers. Evict-between-models is the
default where the eviction seam exists (keep_alive 0 through the backend
registry; host-effective only). Without the governor the degradation is
honest: governor_present false on the run row, admitted "absent" on every
task row, the static requested_ctx used as-is -- comparable WITHIN a run,
marked unadmitted ACROSS runs, visible never masked.

Ticket composition (the S229 6.6 seam): the runner holds
ticket_scope(decision) around each agent run, so the loop's thread-local
ticket read lights the FED branch live -- the admitted num_ctx is the one
the truncation caps and prune thresholds derive from. admitted_num_ctx is
deliberately NOT also passed explicitly: one seam (DI-S230 1).

State and egress neutralization (DI-S230 8): the model sees the production
Daily tool surface verbatim (all twelve schemas), but the manage_skills,
manage_memory and web_search handlers are replaced with honest refusal
stubs -- an eval run never mutates user state and never reaches the
network through the tool surface. The eval path runs Daily semantics with
no approval_fn, so the doom-loop corrective/abort branch is the live one
and "doom_loop" is a first-class failure class.

Host-assured (named, never simulated in the container): the
model-in-the-loop runs themselves, eviction effectiveness between models
on real VRAM, real bwrap spill and diagnostics behaviour, and the opencode
side-by-side baseline script.
"""

import logging
import re
import threading
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Module conventions (project-wide).
checkpoint_before_apply = True

from opti_oignon.agent_eval.store import (  # noqa: E402
    FAILURE_CLASSES,
    EvalResultsStore,
)
from opti_oignon.agent_eval.tasks import (  # noqa: E402
    TaskSpec,
    load_suite,
    max_requested_ctx,
)

# The Lot 1/2 agent surface the runner CONSUMES (read-only; guarded so a
# partial agent build degrades to FEATURE_AVAILABLE False instead of an
# import error at app startup).
try:
    from opti_oignon.agent import loop as agent_loop
    from opti_oignon.agent import tools as agent_tools
    from opti_oignon.file_tools import _handle_sandbox_create_file
    from opti_oignon.sandbox_tools import SandboxToolSession

    _AGENT_OK = True
except Exception:  # pragma: no cover - partial build degradation
    agent_loop = None  # type: ignore[assignment]
    agent_tools = None  # type: ignore[assignment]
    _handle_sandbox_create_file = None  # type: ignore[assignment]
    SandboxToolSession = None  # type: ignore[assignment]
    _AGENT_OK = False

# Ticket plumbing (the 6.6 composition). Imported from the real governor
# module when present; the governor RESOLUTION for admit() goes through
# sys.modules first (the benchmark idiom) so tests can stub decisions
# while the thread-local scope stays the real one the loop reads.
try:
    from opti_oignon.resource_governor import ticket_scope as _ticket_scope

    _TICKET_OK = True
except Exception:  # pragma: no cover - governor module absent

    from contextlib import contextmanager

    @contextmanager
    def _ticket_scope(decision):  # type: ignore[misc]
        yield

    _TICKET_OK = False

FEATURE_AVAILABLE = _AGENT_OK

# Per-check command timeout inside the sandbox (DI-S230 5). Micro tasks
# are small by construction; a check that needs more than this is not a
# micro check.
CHECK_TIMEOUT_S = 60

# Tools whose handlers are neutralized on the eval path (DI-S230 8).
_DISABLED_TOOLS = ("manage_skills", "manage_memory", "web_search")

# Spill references live in the transcript as workspace-relative paths
# (the 6.1 truncation stubs and the 6.2 prune stubs both carry them).
_SPILL_REF_RE = re.compile(r"\.agent/spill/[A-Za-z0-9_./-]+")

_DIAGNOSTICS_MARKER = "[diagnostics]"


# ---------------------------------------------------------------------------
# Governor plumbing (the benchmark-runner S224/S225 idiom, mirrored --
# never imported from benchmark_runner, which stays unedited)
# ---------------------------------------------------------------------------


def _resolve_resource_governor() -> Any:
    """Lazy governor resolver; None means unguarded (fail-open).

    sys.modules is consulted first so a test-seeded or standalone-loaded
    module is reused as-is, then the package import; any error degrades
    to None (the availability-control posture).
    """
    try:
        import sys as _sys

        mod = _sys.modules.get("opti_oignon.resource_governor")
        if mod is None:
            from opti_oignon import resource_governor as mod  # type: ignore
        if mod is None or not getattr(mod, "FEATURE_AVAILABLE", False):
            return None
        return mod
    except Exception:
        return None


def _governor_enabled(module: Any) -> bool:
    """Whether the resolved governor is present AND enabled."""
    if module is None:
        return False
    try:
        return bool(module.get_resource_governor().config.enabled)
    except Exception:
        return False


def _admit_model(model: str, requested_ctx: int) -> Any:
    """Per-model admission with eval semantics (spec 4.3 / 7.2):
    admit or refuse, NEVER downsize. None when the governor is absent or
    disabled (the honest-degradation path). Mirrors the benchmark
    runner's S225 entry: admit_or_wait when present degrades to plain
    admit() with the shipped default (nobody enrolled in the queue).
    """
    governor_module = _resolve_resource_governor()
    if governor_module is None:
        return None
    try:
        governor = governor_module.get_resource_governor()
        if not governor.config.enabled:
            return None
        admit_fn = getattr(governor, "admit_or_wait", governor.admit)
        decision = admit_fn(model, requested_ctx, caller="agent_eval")
        if (
            getattr(decision, "admitted", False)
            and getattr(decision, "load_expected", False)
        ):
            governor.invalidate_on_load(model, getattr(decision, "num_ctx", None))
        return decision
    except Exception as exc:
        logger.debug("Eval admission failed open: %s", exc)
        return None


def _evict_loaded_models() -> int:
    """Best-effort eviction between models (the S224 benchmark idiom):
    the backend registry's existing unload idiom (keep_alive=0, the S215
    primitive), then the governor snapshot invalidated. Host-effective
    only; every path fails open. Returns the number unloaded (0 on any
    failure).
    """
    count = 0
    try:
        import sys as _sys

        ib = _sys.modules.get("opti_oignon.inference_backend")
        if ib is None:
            from opti_oignon import inference_backend as ib  # type: ignore
        registry = ib.get_backend_registry()
        for backend in registry.backends():
            unload = getattr(backend, "unload_all", None)
            if callable(unload):
                try:
                    count += int(unload() or 0)
                except Exception as exc:
                    logger.debug("Evict-between unload failed: %s", exc)
    except Exception as exc:
        logger.debug("Evict-between unavailable: %s", exc)
    governor_module = _resolve_resource_governor()
    if governor_module is not None:
        try:
            governor_module.get_resource_governor().invalidate_on_evict(None)
        except Exception:
            pass
    return count


# ---------------------------------------------------------------------------
# Model client (the routes_agent production bridge, mirrored locally so the
# harness has no dependency on a web-layer private)
# ---------------------------------------------------------------------------


class OllamaChatClient:
    """Bridge an Ollama chat stream to the loop's ``stream(messages, tools)``.

    Ollama yields chunks shaped as ``{"message": {"content", "tool_calls"}}``,
    the loop's expected stream shape. The import is lazy so the module loads
    without ollama installed.
    """

    def __init__(self, model: str, *, host: str | None = None) -> None:
        self._model = model
        self._host = host

    def stream(self, messages: list[dict[str, Any]], tools: Any = None):
        import ollama

        client = ollama.Client(host=self._host) if self._host else ollama.Client()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        for chunk in client.chat(**kwargs):
            yield chunk


def _default_client_factory(model: str) -> Any:
    return OllamaChatClient(model)


# ---------------------------------------------------------------------------
# Tool surface (the AgentRunManager wiring, with DI-S230 8 neutralization)
# ---------------------------------------------------------------------------


def _disabled_tool_handler(name: str) -> Callable[[dict[str, Any]], str]:
    def _handler(arguments: dict[str, Any]) -> str:
        return f"Error: {name} is disabled in the eval harness"

    return _handler


def _build_eval_surface() -> tuple[Any, dict[str, Any], str]:
    """(native_tools, handlers, system_prompt) for one eval run.

    The Daily schemas verbatim (the production surface the model sees);
    the three stateful/egress handlers replaced with honest refusal stubs.
    """
    tool_set = agent_tools.build_tool_set("daily")
    handlers = dict(tool_set.tool_handlers)
    for name in _DISABLED_TOOLS:
        if name in handlers:
            handlers[name] = _disabled_tool_handler(name)
    native = tool_set.native_tools()
    prompt = agent_tools.system_prompt_section_for("daily")
    return native, handlers, prompt


# ---------------------------------------------------------------------------
# Pure helpers (container-testable in isolation)
# ---------------------------------------------------------------------------


def _host_fingerprint_lite() -> str:
    """A short, human-readable host descriptor for cross-run comparison.

    Deliberately lite (DI-S230 11): platform facts only, no GPU probe
    (real VRAM behaviour is host-assured territory).
    """
    import json
    import platform

    return json.dumps(
        {
            "system": platform.system(),
            "machine": platform.machine(),
            "cpus": __import__("os").cpu_count() or 0,
            "python": platform.python_version(),
        },
        sort_keys=True,
    )


def _extract_spill_refs(messages: list[dict[str, Any]]) -> str | None:
    """Collect .agent/spill/ paths from the final transcript (DI-S230 9).

    Spill paths exist only in the transcript: the 6.1 truncation stubs
    ("full output: <path>") and the 6.2 prune stubs ("spill: <path>")
    both carry them. Ordered, de-duplicated, comma-joined; None when the
    run spilled nothing.
    """
    seen: list[str] = []
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for match in _SPILL_REF_RE.findall(content):
            if match not in seen:
                seen.append(match)
    return ",".join(seen) if seen else None


def _diagnostics_seen(tool_results: list[Any]) -> bool:
    """Whether any tool observation carried a [diagnostics] block.

    The DispatchResults keep full pre-cap text (the S229 posture), so the
    scan is exact regardless of transcript truncation.
    """
    for result in tool_results or []:
        observation = getattr(result, "observation", "")
        if isinstance(observation, str) and _DIAGNOSTICS_MARKER in observation:
            return True
    return False


def _classify_outcome(
    stop_reason: str,
    deadline_hit: bool,
    check_codes: list[int] | None,
    any_blocked: bool,
    calls_attempted: int,
    calls_executed: int,
) -> tuple[bool, str]:
    """(passed, failure_class) per DI-S230 6/7, deterministic.

    Order: error (loop error or a blocked check -- an environment refusal
    is not a task failure), doom_loop, timeout, then the checks verdict:
    all zero -> passed/none; otherwise refusal when the gate refused every
    attempted call (work was attempted, none executed), else test_fail.
    ``check_codes`` is None when the checks were not executed (an earlier
    classification short-circuits them).
    """
    if stop_reason == "error":
        return False, "error"
    if stop_reason == "doom_loop":
        return False, "doom_loop"
    if deadline_hit:
        return False, "timeout"
    if any_blocked:
        return False, "error"
    if check_codes is not None and all(code == 0 for code in check_codes):
        return True, "none"
    if calls_attempted > 0 and calls_executed == 0:
        return False, "refusal"
    return False, "test_fail"


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Drives eval runs: one at a time, background thread, cancellable."""

    def __init__(
        self,
        store: EvalResultsStore | None = None,
        db_path: Any = None,
        sandbox_manager: Any = None,
        client_factory: Callable[[str], Any] | None = None,
        surface_factory: Callable[[], tuple[Any, dict[str, Any], str]] | None = None,
    ):
        self._store = store or EvalResultsStore(db_path)
        self._sandbox_manager = sandbox_manager
        self._client_factory = client_factory or _default_client_factory
        self._surface_factory = surface_factory or _build_eval_surface
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._busy = False
        self._progress: dict[str, Any] = {}

    @property
    def store(self) -> EvalResultsStore:
        return self._store

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def status(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self._progress)
            snapshot["busy"] = self._busy
        return snapshot

    def cancel(self) -> bool:
        """Request cooperative cancellation of the active run."""
        with self._lock:
            if not self._busy:
                return False
        self._cancel.set()
        return True

    # -- entry points --------------------------------------------------------

    def start_run(
        self,
        models: list[str],
        suite: str = "micro",
        repeats: int = 1,
        evict_between: bool = True,
        tasks: list[TaskSpec] | None = None,
    ) -> str:
        """Start a run on a background thread; one run at a time."""
        run_id, task_list = self._prepare(models, suite, repeats, tasks)
        thread = threading.Thread(
            target=self._execute_run,
            args=(run_id, list(models), suite, task_list, int(repeats), evict_between),
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()
        return run_id

    def run_sync(
        self,
        models: list[str],
        suite: str = "micro",
        repeats: int = 1,
        evict_between: bool = True,
        tasks: list[TaskSpec] | None = None,
    ) -> str:
        """Run synchronously (the CLI and test entry); returns the run id."""
        run_id, task_list = self._prepare(models, suite, repeats, tasks)
        self._execute_run(
            run_id, list(models), suite, task_list, int(repeats), evict_between
        )
        return run_id

    def _prepare(
        self,
        models: list[str],
        suite: str,
        repeats: int,
        tasks: list[TaskSpec] | None,
    ) -> tuple[str, list[TaskSpec]]:
        if not FEATURE_AVAILABLE:
            raise RuntimeError("agent surface unavailable; eval runner disabled")
        if not models or not all(isinstance(m, str) and m.strip() for m in models):
            raise ValueError("models must be a non-empty list of names")
        if int(repeats) < 1:
            raise ValueError("repeats must be >= 1")
        task_list = list(tasks) if tasks is not None else load_suite(suite)
        if not task_list:
            raise ValueError("the suite resolved to zero tasks")
        with self._lock:
            if self._busy:
                raise RuntimeError("an eval run is already in progress")
            self._busy = True
            self._cancel.clear()
            run_id = f"eval-{uuid.uuid4().hex[:12]}"
            self._progress = {
                "run_id": run_id,
                "suite": suite,
                "model": "",
                "task_id": "",
                "completed": 0,
                "total": len(models) * len(task_list) * int(repeats),
            }
        return run_id, task_list

    # -- execution -----------------------------------------------------------

    def _execute_run(
        self,
        run_id: str,
        models: list[str],
        suite: str,
        tasks: list[TaskSpec],
        repeats: int,
        evict_between: bool,
    ) -> None:
        governor_module = _resolve_resource_governor()
        governor_present = _governor_enabled(governor_module)
        try:
            self._store.create_run(
                run_id,
                suite,
                models,
                repeats,
                governor_present,
                _host_fingerprint_lite(),
            )
        except Exception:
            logger.exception("eval run row creation failed")
            with self._lock:
                self._busy = False
            return

        status = "completed"
        error = ""
        admission_ctx = max_requested_ctx(tasks)
        try:
            for model_index, model in enumerate(models):
                if self._cancel.is_set():
                    status = "cancelled"
                    break
                self._set_progress(model=model, task_id="")

                decision = (
                    _admit_model(model, admission_ctx) if governor_present else None
                )
                if decision is not None and not getattr(decision, "admitted", False):
                    # Refused: every (task, repeat) row records the skip
                    # honestly; never a silent downsize (spec 7.2).
                    reason = getattr(decision, "reason", "")
                    logger.info(
                        "Eval admission refused for %s (%s); model skipped",
                        model,
                        reason,
                    )
                    for task in tasks:
                        for repeat in range(repeats):
                            self._store.record_task(
                                run_id,
                                model,
                                task.id,
                                repeat,
                                passed=False,
                                rounds=0,
                                tool_calls=0,
                                wall_s=0.0,
                                failure_class="not_admitted",
                                admitted="refused",
                                admitted_ctx=None,
                            )
                            self._bump_progress()
                    continue

                admitted = "yes" if decision is not None else "absent"
                admitted_ctx = (
                    getattr(decision, "num_ctx", None)
                    if decision is not None
                    else None
                )
                client = self._client_factory(model)

                cancelled = False
                for task in tasks:
                    for repeat in range(repeats):
                        if self._cancel.is_set():
                            cancelled = True
                            break
                        self._set_progress(model=model, task_id=task.id)
                        row = self._run_one(
                            model, task, repeat, decision, client
                        )
                        if row is None:
                            # Cancel landed mid-task: the in-flight task is
                            # not recorded (DI-S230 6).
                            cancelled = True
                            break
                        self._store.record_task(
                            run_id,
                            model,
                            task.id,
                            repeat,
                            passed=row["passed"],
                            rounds=row["rounds"],
                            tool_calls=row["tool_calls"],
                            wall_s=row["wall_s"],
                            failure_class=row["failure_class"],
                            admitted=admitted,
                            admitted_ctx=admitted_ctx,
                            spill_ref=row["spill_ref"],
                            diagnostics_seen=row["diagnostics_seen"],
                        )
                        self._bump_progress()
                    if cancelled:
                        break
                if cancelled:
                    status = "cancelled"
                    break
                if evict_between and model_index < len(models) - 1:
                    _evict_loaded_models()
        except Exception as exc:  # the worker never raises out
            logger.exception("eval run failed")
            status = "failed"
            error = str(exc)
        finally:
            try:
                self._store.finish_run(run_id, status, error)
            except Exception:
                logger.exception("eval run finish write failed")
            with self._lock:
                self._busy = False

    def _run_one(
        self,
        model: str,
        task: TaskSpec,
        repeat: int,
        decision: Any,
        client: Any,
    ) -> dict[str, Any] | None:
        """One (model, task, repeat): fresh sandbox, fixture, run, checks.

        Returns the row dict, or None when cancellation landed mid-task.
        """
        started = time.monotonic()
        deadline = started + float(task.timeout_s)

        session = SandboxToolSession(
            sandbox_mgr=self._sandbox_manager, tool_registry=None
        )
        try:
            session.start(allow_degraded=True)
        except Exception as exc:
            logger.error("eval sandbox start failed: %s", exc)
            return self._error_row(started, f"sandbox start failed: {exc}")

        try:
            manager = session.sandbox_manager
            session_id = session.session_id

            # Fixture materialization through the path-confined create_file
            # handler (DI-S230 4): the exact write seam the session itself
            # uses, minus the diagnostics suffix (a feedback feature, not a
            # materialization feature).
            for relpath, content in task.fixture.items():
                result = _handle_sandbox_create_file(
                    session_id, relpath, content, _sandbox_manager=manager
                )
                if isinstance(result, str) and result.startswith("Error"):
                    return self._error_row(
                        started, f"fixture write failed: {result}"
                    )

            native, handlers, system_prompt = self._surface_factory()

            cancel_event = self._cancel

            def _should_continue() -> bool:
                return (
                    not cancel_event.is_set()
                    and time.monotonic() < deadline
                )

            with _ticket_scope(decision):
                run_result = agent_loop.run(
                    task.prompt,
                    model_client=client,
                    sandbox=session,
                    mode="daily",
                    conversation_id="",
                    system_prompt=system_prompt,
                    tools=native,
                    approval_fn=None,
                    tool_handlers=handlers,
                    max_rounds=task.max_rounds,
                    include_memory=False,
                    should_continue=_should_continue,
                    verify=False,
                )

            if cancel_event.is_set():
                return None

            stop_reason = run_result.stop_reason
            deadline_hit = (
                stop_reason == agent_loop.STOP_CANCELLED
                and time.monotonic() >= deadline
            )

            calls_attempted = len(run_result.tool_results)
            calls_executed = sum(
                1
                for r in run_result.tool_results
                if getattr(r, "executed", False)
            )

            check_codes: list[int] | None = None
            any_blocked = False
            run_checks = stop_reason in (
                agent_loop.STOP_DONE,
                agent_loop.STOP_MAX_ROUNDS,
            )
            if run_checks:
                check_codes = []
                for command in task.checks:
                    result = manager.execute_command(
                        session_id, command, timeout=CHECK_TIMEOUT_S
                    )
                    if getattr(result, "blocked", False):
                        any_blocked = True
                        logger.warning(
                            "eval check blocked (%s): %s",
                            getattr(result, "block_reason", ""),
                            command,
                        )
                        break
                    code = int(getattr(result, "return_code", 1))
                    if getattr(result, "timed_out", False):
                        code = code if code != 0 else 1
                    check_codes.append(code)
                    if code != 0:
                        break

            passed, failure_class = _classify_outcome(
                stop_reason,
                deadline_hit,
                check_codes,
                any_blocked,
                calls_attempted,
                calls_executed,
            )

            return {
                "passed": passed,
                "rounds": run_result.rounds,
                "tool_calls": calls_attempted,
                "wall_s": round(time.monotonic() - started, 3),
                "failure_class": failure_class,
                "spill_ref": _extract_spill_refs(run_result.messages),
                "diagnostics_seen": _diagnostics_seen(run_result.tool_results),
            }
        except Exception as exc:
            logger.exception("eval task crashed")
            return self._error_row(started, str(exc))
        finally:
            try:
                session.stop()
            except Exception:
                logger.debug("eval sandbox stop failed", exc_info=True)

    @staticmethod
    def _error_row(started: float, message: str) -> dict[str, Any]:
        logger.debug("eval error row: %s", message)
        return {
            "passed": False,
            "rounds": 0,
            "tool_calls": 0,
            "wall_s": round(time.monotonic() - started, 3),
            "failure_class": "error",
            "spill_ref": None,
            "diagnostics_seen": False,
        }

    # -- progress ------------------------------------------------------------

    def _set_progress(self, **fields: Any) -> None:
        with self._lock:
            self._progress.update(fields)

    def _bump_progress(self) -> None:
        with self._lock:
            self._progress["completed"] = self._progress.get("completed", 0) + 1


# Module-level singleton (the established get/reset pair for routers and
# tests).
_runner_lock = threading.Lock()
_runner: EvalRunner | None = None


def get_eval_runner() -> EvalRunner:
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = EvalRunner()
        return _runner


def reset_eval_runner() -> None:
    global _runner
    with _runner_lock:
        _runner = None


__all__ = [
    "CHECK_TIMEOUT_S",
    "FAILURE_CLASSES",
    "FEATURE_AVAILABLE",
    "EvalRunner",
    "OllamaChatClient",
    "get_eval_runner",
    "reset_eval_runner",
]
