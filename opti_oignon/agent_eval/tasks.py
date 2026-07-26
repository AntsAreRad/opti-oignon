#!/usr/bin/env python3
"""
Agent eval task model and suite loader.

A TaskSpec is one micro-task: a prompt, a fixture (relative path to file
content, materialized into a fresh sandbox workspace), and checks (commands
run IN the sandbox after the agent run; every command exiting 0 scores the
task as passed). Suites are YAML files under agent_eval/suites/; micro.yaml
is the v1 suite (12 tasks across the capability axes the AGT lots target).

Scoring is auto, by the checks passing: no judge model, no rubric. The
harness measures capability, not taste.

Structural validation happens at load time so a malformed suite fails
loudly before any sandbox is created. Fixture relpaths are validated here
(relative, no traversal) AND again at the write seam (the path-confined
create_file handler); checks must be deterministic and blocklist-clean
(the sandbox CommandValidator applies on every backend).
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module conventions (project-wide): every new module ships with the
# checkpoint discipline hardcoded and a feature sentinel.
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

try:
    import yaml

    _YAML_OK = True
except ImportError:  # pragma: no cover - PyYAML is a core dependency
    yaml = None  # type: ignore[assignment]
    _YAML_OK = False

# Suite directory: YAML files shipped with the package.
SUITES_DIR = Path(__file__).parent / "suites"

# TaskSpec defaults (AGT_SPEC 7.1, verbatim).
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_MAX_ROUNDS = 10
DEFAULT_REQUESTED_CTX = 8192


@dataclass
class TaskSpec:
    """One micro-task (AGT_SPEC 7.1).

    ``fixture`` maps workspace-relative paths to file content; it is
    materialized into a FRESH sandbox workspace before the agent run.
    ``checks`` are commands executed in the same sandbox after the run;
    the task passes only when every check exits 0.
    """

    id: str
    title: str
    prompt: str
    fixture: dict[str, str] = field(default_factory=dict)
    checks: list[str] = field(default_factory=list)
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_rounds: int = DEFAULT_MAX_ROUNDS
    requested_ctx: int = DEFAULT_REQUESTED_CTX

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "prompt": self.prompt,
            "fixture": dict(self.fixture),
            "checks": list(self.checks),
            "timeout_s": self.timeout_s,
            "max_rounds": self.max_rounds,
            "requested_ctx": self.requested_ctx,
        }


def _validate_fixture_path(relpath: str) -> str | None:
    """Return an error string when ``relpath`` is not a safe relative path.

    The write seam (the path-confined create_file handler) re-validates
    against the live workspace; this structural pass rejects the obvious
    shapes early so a bad suite never reaches a sandbox.
    """
    if not isinstance(relpath, str) or not relpath.strip():
        return "fixture path must be a non-empty string"
    if relpath.startswith("/") or relpath.startswith("\\"):
        return f"fixture path must be relative: {relpath!r}"
    if "\\" in relpath:
        return f"fixture path must use forward slashes: {relpath!r}"
    parts = relpath.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return f"fixture path contains traversal or empty segments: {relpath!r}"
    if os.path.isabs(relpath):  # defense-in-depth on exotic forms
        return f"fixture path must be relative: {relpath!r}"
    return None


def _coerce_task(raw: Any, index: int, seen_ids: set[str]) -> TaskSpec:
    """Validate one raw task mapping and build the TaskSpec.

    Raises ValueError with a message naming the task (by id when present,
    by index otherwise) on any structural problem.
    """
    where = f"task #{index}"
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: each task must be a mapping")

    task_id = raw.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"{where}: 'id' must be a non-empty string")
    task_id = task_id.strip()
    where = f"task {task_id!r}"
    if task_id in seen_ids:
        raise ValueError(f"{where}: duplicate task id")
    seen_ids.add(task_id)

    title = raw.get("title", "")
    if not isinstance(title, str):
        raise ValueError(f"{where}: 'title' must be a string")

    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{where}: 'prompt' must be a non-empty string")

    fixture_raw = raw.get("fixture", {}) or {}
    if not isinstance(fixture_raw, dict):
        raise ValueError(f"{where}: 'fixture' must be a mapping of path to content")
    fixture: dict[str, str] = {}
    for relpath, content in fixture_raw.items():
        err = _validate_fixture_path(relpath)
        if err is not None:
            raise ValueError(f"{where}: {err}")
        if not isinstance(content, str):
            raise ValueError(
                f"{where}: fixture content for {relpath!r} must be a string"
            )
        fixture[relpath] = content

    checks_raw = raw.get("checks")
    if not isinstance(checks_raw, list) or not checks_raw:
        raise ValueError(f"{where}: 'checks' must be a non-empty list of commands")
    checks: list[str] = []
    for check in checks_raw:
        if not isinstance(check, str) or not check.strip():
            raise ValueError(f"{where}: every check must be a non-empty string")
        checks.append(check)

    def _number(key: str, default: float, minimum: float) -> float:
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: '{key}' must be a number")
        if value < minimum:
            raise ValueError(f"{where}: '{key}' must be >= {minimum}")
        return float(value)

    timeout_s = _number("timeout_s", DEFAULT_TIMEOUT_S, 1.0)

    def _integer(key: str, default: int, minimum: int) -> int:
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: '{key}' must be an integer")
        if value < minimum:
            raise ValueError(f"{where}: '{key}' must be >= {minimum}")
        return value

    max_rounds = _integer("max_rounds", DEFAULT_MAX_ROUNDS, 1)
    requested_ctx = _integer("requested_ctx", DEFAULT_REQUESTED_CTX, 1)

    return TaskSpec(
        id=task_id,
        title=title,
        prompt=prompt,
        fixture=fixture,
        checks=checks,
        timeout_s=timeout_s,
        max_rounds=max_rounds,
        requested_ctx=requested_ctx,
    )


def resolve_suite_path(suite: str) -> Path:
    """Resolve a suite name or path to a YAML file path.

    A bare name (no separator, no .yaml/.yml suffix) resolves under the
    package suites directory; anything else is treated as a filesystem
    path. The file must exist either way.
    """
    if not isinstance(suite, str) or not suite.strip():
        raise ValueError("suite must be a non-empty string")
    suite = suite.strip()
    candidate = Path(suite)
    looks_like_path = (
        "/" in suite or suite.endswith(".yaml") or suite.endswith(".yml")
    )
    path = candidate if looks_like_path else SUITES_DIR / f"{suite}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"suite not found: {path}")
    return path


def load_suite(suite: str) -> list[TaskSpec]:
    """Load and validate a suite, returning its TaskSpec list in order."""
    if not _YAML_OK:  # pragma: no cover - PyYAML is a core dependency
        raise RuntimeError("PyYAML is required to load eval suites")
    path = resolve_suite_path(suite)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"suite {path.name}: top level must be a mapping")
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError(f"suite {path.name}: 'tasks' must be a non-empty list")
    seen: set[str] = set()
    tasks = [
        _coerce_task(entry, index, seen) for index, entry in enumerate(tasks_raw)
    ]
    logger.debug("Loaded suite %s: %d tasks", path.name, len(tasks))
    return tasks


def max_requested_ctx(tasks: list[TaskSpec]) -> int:
    """The per-model admission request: the suite's largest requested_ctx.

    One admission per model covers every task of that model (AGT_SPEC 7.2:
    admit before each model's first task and between models), so the
    request must fit the most demanding task.
    """
    if not tasks:
        return DEFAULT_REQUESTED_CTX
    return max(task.requested_ctx for task in tasks)
