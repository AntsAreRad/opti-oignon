#!/usr/bin/env python3
"""
Opti-Oignon agent eval harness -- S230 (AGT_SPEC Section 7).

The micro-task eval harness for the agent surface: TaskSpec suites scored
by in-sandbox checks (7.1), an EvalRunner on the BenchmarkRunner idiom
with the resource-governor admission contract (7.2), a dedicated results
store separate from the benchmark register (7.3), and the API/CLI entry
points (7.4). The opencode side-by-side baseline (7.5) is a host-only
reference script under scripts/, never a container path.

Everything about the harness except the model's own behaviour is
container-provable; the model-in-the-loop runs are host territory by
design, made one-command (scripts/run_agent_eval.sh).
"""

# Module conventions (project-wide).
checkpoint_before_apply = True

from opti_oignon.agent_eval.runner import (  # noqa: E402
    CHECK_TIMEOUT_S,
    FEATURE_AVAILABLE,
    EvalRunner,
    OllamaChatClient,
    get_eval_runner,
    reset_eval_runner,
)
from opti_oignon.agent_eval.store import (  # noqa: E402
    FAILURE_CLASSES,
    RUN_STATUSES,
    EvalResultsStore,
)
from opti_oignon.agent_eval.tasks import (  # noqa: E402
    DEFAULT_MAX_ROUNDS,
    DEFAULT_REQUESTED_CTX,
    DEFAULT_TIMEOUT_S,
    SUITES_DIR,
    TaskSpec,
    load_suite,
    max_requested_ctx,
    resolve_suite_path,
)

__all__ = [
    "CHECK_TIMEOUT_S",
    "DEFAULT_MAX_ROUNDS",
    "DEFAULT_REQUESTED_CTX",
    "DEFAULT_TIMEOUT_S",
    "FAILURE_CLASSES",
    "FEATURE_AVAILABLE",
    "RUN_STATUSES",
    "SUITES_DIR",
    "EvalResultsStore",
    "EvalRunner",
    "OllamaChatClient",
    "TaskSpec",
    "get_eval_runner",
    "load_suite",
    "max_requested_ctx",
    "reset_eval_runner",
    "resolve_suite_path",
]
