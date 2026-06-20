#!/usr/bin/env python3
"""
Agent eval CLI -- S230 (AGT_SPEC Section 7.4).

Usage:
    python -m opti_oignon.agent_eval --models qwen3:4b,qwen3:14b --suite micro
    python -m opti_oignon.agent_eval --models qwen3:4b --suite micro --repeats 3

One command runs the whole (model, task, repeat) matrix synchronously and
prints the register summary. scripts/run_agent_eval.sh wraps this entry as
the one-command host path. Exit codes: 0 the run completed, 1 the run
ended failed or cancelled, 2 argument or suite errors.

The model-in-the-loop runs are host territory (a live Ollama endpoint and
the local fleet); the CLI structure itself is container-testable through
the injectable runner seam.
"""

import argparse
import sys

# Module conventions (project-wide).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

from opti_oignon.agent_eval.runner import EvalRunner  # noqa: E402
from opti_oignon.agent_eval.tasks import load_suite  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m opti_oignon.agent_eval",
        description="Run the agent micro-task eval harness (AGT_SPEC 7).",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated Ollama model names (e.g. qwen3:4b,qwen3:14b)",
    )
    parser.add_argument(
        "--suite",
        default="micro",
        help="Suite name under agent_eval/suites/ or a YAML path (default: micro)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeats per (model, task) cell (default: 1)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Results database path (default: data/agent_eval_results.db)",
    )
    parser.add_argument(
        "--no-evict",
        action="store_true",
        help="Skip the evict-between-models default (host-effective seam)",
    )
    return parser


def _print_summary(details: dict) -> None:
    run = details["run"]
    governor = "present" if run["governor_present"] else "absent"
    print(
        f"Run {run['run_id']}  suite={run['suite']}  status={run['status']}"
        f"  governor={governor}"
    )
    if run.get("error"):
        print(f"  error: {run['error']}")
    summary = details.get("summary", {})
    for model in run.get("models", []):
        bucket = summary.get(model)
        if bucket is None:
            print(f"  {model}: no task rows recorded")
            continue
        line = (
            f"  {model}: {bucket['passed']}/{bucket['total']} passed;"
            f" rounds avg {bucket['rounds_avg']};"
            f" wall avg {bucket['wall_avg_s']}s"
        )
        print(line)
        failures = bucket.get("failures") or {}
        if failures:
            parts = " ".join(
                f"{name}={count}" for name, count in sorted(failures.items())
            )
            print(f"    failures: {parts}")


def main(argv: list[str] | None = None, runner: EvalRunner | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("error: --models must name at least one model", file=sys.stderr)
        return 2
    if args.repeats < 1:
        print("error: --repeats must be >= 1", file=sys.stderr)
        return 2

    try:
        load_suite(args.suite)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    eval_runner = runner or EvalRunner(db_path=args.db)
    try:
        run_id = eval_runner.run_sync(
            models=models,
            suite=args.suite,
            repeats=args.repeats,
            evict_between=not args.no_evict,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    details = eval_runner.store.get_run_details(run_id)
    if details is None:
        print(f"error: run {run_id} not found in the store", file=sys.stderr)
        return 1
    _print_summary(details)
    return 0 if details["run"]["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover - module entry
    sys.exit(main())
