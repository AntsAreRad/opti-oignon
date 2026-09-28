#!/usr/bin/env python3
"""The step event of a reply and the recorder that emits it.

A reply that runs an execution pipeline, a reasoning strategy, a consensus
or a self-correction is made of steps. The interface draws each step from
``pipeline_step`` frames, and every frame is built here:

* ``build`` makes a frame's metadata. It is closed and total: every field
  of the schema is present, typed and checked against the frame's state,
  and anything else raises ``ValueError``. The label and the reason are cut
  to their caps rather than refused.
* ``StepRecorder`` is one reply's state machine. It numbers the frames and
  sends each one to its sink under one lock, measures a step's duration on
  the monotonic clock, fills a parent step's progress from its nested run,
  and closes every open step on a stop or an error. An event the transition
  rule does not allow raises and sends nothing.

Every emitter calls the recorder through ``emit``, which does nothing
without a recorder and turns a recorder error into a log line: a step
frame never breaks a reply.

Standard library only: the core imports it inside the functions that emit.
"""

import logging
import threading
import time

checkpoint_before_apply = True

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
LABEL_CAP = 120
REASON_CAP = 300

# The first wave. ``coding`` joins with its emitter; ``research`` and
# ``search`` join with the first server block that makes a search unfold.
KINDS = ("exec_pipeline", "reasoning", "consensus", "self_correct")
REASONING_NAMES = ("decompose", "tree_of_thought", "self_consistency")
UNITS = ("sub_step", "model", "sample")

STATES = ("pending", "running", "done", "failed", "skipped", "cancelled", "not_run")
TERMINAL = ("done", "failed", "skipped", "cancelled", "not_run")
_RAN = ("done", "failed", "cancelled")
_WITH_REASON = ("failed", "not_run", "cancelled")

FIELDS = (
    "v", "seq", "run", "kind", "name", "pipeline_id", "parent", "index",
    "total", "label", "step_type", "state", "progress", "reason", "ran_as",
    "duration_ms",
)
_REQUIRED = ("seq", "run", "kind", "name", "index", "label", "state")
_OPTIONAL = ("pipeline_id", "parent", "total", "step_type", "progress",
             "reason", "ran_as", "duration_ms")


def _int(value, key: str, low: int) -> int:
    if type(value) is not int or value < low:
        raise ValueError(f"{key} must be an integer >= {low}, not {value!r}")
    return value


def _text(value, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string, not {value!r}")
    return value


def _parent(value) -> dict:
    if not isinstance(value, dict) or set(value) != {"run", "index"}:
        raise ValueError(f"parent must be {{run, index}}, not {value!r}")
    return {"run": _text(value["run"], "parent.run"),
            "index": _int(value["index"], "parent.index", 0)}


def _progress(value) -> dict:
    if not isinstance(value, dict) or set(value) != {"done", "total", "unit"}:
        raise ValueError(f"progress must be {{done, total, unit}}, not {value!r}")
    done = _int(value["done"], "progress.done", 0)
    total = _int(value["total"], "progress.total", 1)
    if done > total:
        raise ValueError(f"progress.done {done} is above its total {total}")
    if value["unit"] not in UNITS:
        raise ValueError(f"progress.unit {value['unit']!r} has no emitter")
    return {"done": done, "total": total, "unit": value["unit"]}


def build(**fields) -> dict:
    """The metadata of one ``pipeline_step`` frame, every field present."""
    unknown = sorted(set(fields) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(unknown)}")
    missing = [key for key in _REQUIRED if key not in fields]
    if missing:
        raise ValueError(f"missing field(s): {', '.join(missing)}")
    f = dict.fromkeys(_OPTIONAL)
    f.update(fields)

    kind, name, state = f["kind"], _text(f["name"], "name"), f["state"]
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} has no emitter")
    if kind == "reasoning" and name not in REASONING_NAMES:
        raise ValueError(f"reasoning run named {name!r}")
    if kind in ("consensus", "self_correct") and name != kind:
        raise ValueError(f"{kind} run named {name!r}")
    if state not in STATES:
        raise ValueError(f"state {state!r} is not one of the seven")
    pipeline = kind == "exec_pipeline"

    for key in ("pipeline_id", "step_type"):
        if f[key] is not None:
            if not pipeline:
                raise ValueError(f"{key} belongs to a pipeline step")
            _text(f[key], key)
    index = _int(f["index"], "index", 0)
    total = f["total"]
    if total is not None and index >= _int(total, "total", 1):
        raise ValueError(f"index {index} is not below the total {total}")
    label = f["label"]
    if not isinstance(label, str):
        raise ValueError(f"label must be a string, not {label!r}")

    progress = f["progress"]
    if progress is not None:
        if state != "running":
            raise ValueError(f"progress on a {state} step")
        progress = _progress(progress)
    reason = f["reason"]
    if reason is not None:
        if state not in _WITH_REASON:
            raise ValueError(f"reason on a {state} step")
        if not isinstance(reason, str):
            raise ValueError(f"reason must be a string, not {reason!r}")
        reason = reason[:REASON_CAP]
    if f["ran_as"] is not None:
        if not pipeline or state not in _RAN:
            raise ValueError(f"ran_as on a {state} {kind} step")
        _text(f["ran_as"], "ran_as")
    if f["duration_ms"] is not None:
        if state not in _RAN:
            raise ValueError(f"duration_ms on a {state} step")
        _int(f["duration_ms"], "duration_ms", 0)

    return {
        "v": SCHEMA_VERSION,
        "seq": _int(f["seq"], "seq", 1),
        "run": _text(f["run"], "run"),
        "kind": kind,
        "name": name,
        "pipeline_id": f["pipeline_id"],
        "parent": None if f["parent"] is None else _parent(f["parent"]),
        "index": index,
        "total": total,
        "label": label[:LABEL_CAP],
        "step_type": f["step_type"],
        "state": state,
        "progress": progress,
        "reason": reason,
        "ran_as": f["ran_as"],
        "duration_ms": f["duration_ms"],
    }


def _copy(frame: dict) -> dict:
    out = dict(frame)
    for key in ("parent", "progress"):
        if out[key] is not None:
            out[key] = dict(out[key])
    return out


class StepRecorder:
    """One reply's steps: the transition rule, the sequence and the sink.

    A step is ``(run, index)``. Its events are ``pending``, then
    ``running`` and ``progress`` updates, then one terminal state through
    ``end``; a pending step may end ``skipped`` or ``not_run`` without
    running. Anything else raises, and so does every event once ``close``
    has sealed the recorder. ``seq`` is one counter for the whole reply,
    assigned with the send.
    """

    def __init__(self, sink) -> None:
        self._sink = sink
        self._lock = threading.Lock()
        self._seq = 0
        self._runs: dict[str, dict] = {}
        self._last: dict[tuple, dict] = {}
        self._started: dict[tuple, float] = {}
        self._sealed = False

    # -- the checks -------------------------------------------------------

    def _open(self, event: str) -> None:
        if self._sealed:
            raise ValueError(f"{event} after the recorder was closed")

    def _run(self, run) -> dict:
        if run not in self._runs:
            raise ValueError(f"unknown run {run!r}")
        return self._runs[run]

    def _state(self, run, index, event: str, allowed: tuple) -> dict | None:
        self._run(run)
        last = self._last.get((run, index))
        state = None if last is None else last["state"]
        if state not in allowed:
            raise ValueError(f"step {index} of {run}: {event} after {state or 'nothing'}")
        return last

    # -- the send ---------------------------------------------------------

    def _emit(self, run, index, state, *, label=None, step_type=None,
              progress=None, reason=None, ran_as=None, duration_ms=None) -> dict:
        r = self._runs[run]
        last = self._last.get((run, index))
        frame = build(
            seq=self._seq + 1, run=run, kind=r["kind"], name=r["name"],
            pipeline_id=r["pipeline_id"], parent=r["parent"], index=index,
            total=r["total"], label=last["label"] if last else label,
            step_type=last["step_type"] if last else step_type, state=state,
            progress=progress, reason=reason, ran_as=ran_as,
            duration_ms=duration_ms)
        self._sink(_copy(frame))
        self._seq = frame["seq"]
        self._last[(run, index)] = frame
        return frame

    def _finish(self, run, index, state, reason=None, ran_as=None) -> None:
        start = self._started.get((run, index))
        duration = None if start is None else int((time.monotonic() - start) * 1000)
        self._emit(run, index, state, reason=reason, ran_as=ran_as, duration_ms=duration)
        self._started.pop((run, index), None)

    def _feed_parent(self, run) -> None:
        """A nested run's ends are its parent step's progress, once the
        nested run's total is known."""
        r = self._runs[run]
        parent = r["parent"]
        if parent is None or r["total"] is None:
            return
        last = self._last.get((parent["run"], parent["index"]))
        if last is None or last["state"] != "running":
            return
        ended = sum(1 for (owner, _), frame in self._last.items()
                    if owner == run and frame["state"] in TERMINAL)
        unit = "sample" if r["name"] == "self_consistency" else "sub_step"
        self._update_progress(parent["run"], parent["index"],
                       {"done": ended, "total": r["total"], "unit": unit})

    def _update_progress(self, run, index, progress: dict) -> None:
        last = self._last[(run, index)]
        before = last["progress"]
        if before is not None and progress["done"] < before["done"]:
            raise ValueError(f"step {index} of {run}: progress went back "
                             f"from {before['done']} to {progress['done']}")
        self._emit(run, index, "running", progress=progress)

    # -- the events -------------------------------------------------------

    def start_run(self, kind, name, total=None, pipeline_id=None) -> str:
        """A new run of the reply; nested under the innermost running step."""
        with self._lock:
            self._open("start_run")
            run = f"r{len(self._runs) + 1}"
            build(seq=1, run=run, kind=kind, name=name, index=0, total=total,
                  pipeline_id=pipeline_id, label="", state="pending")
            parent = None
            for owner in reversed(list(self._runs)):
                running = [index for (o, index), frame in self._last.items()
                           if o == owner and frame["state"] == "running"]
                if running:
                    parent = {"run": owner, "index": running[-1]}
                    break
            self._runs[run] = {"kind": kind, "name": name, "total": total,
                               "pipeline_id": pipeline_id, "parent": parent}
            return run

    def set_total(self, run, total) -> None:
        """The run's step count, once, from null (a plan that arrives)."""
        with self._lock:
            self._open("set_total")
            r = self._run(run)
            if r["total"] is not None:
                raise ValueError(f"{run} already has a total of {r['total']}")
            _int(total, "total", 1)
            highest = max((index for owner, index in self._last if owner == run), default=-1)
            if highest >= total:
                raise ValueError(f"{run} already has step {highest}, not below {total}")
            r["total"] = total

    def pending(self, run, index, label, step_type=None) -> None:
        with self._lock:
            self._open("pending")
            self._state(run, index, "pending", (None,))
            self._emit(run, index, "pending", label=label, step_type=step_type)

    def running(self, run, index) -> None:
        with self._lock:
            self._open("running")
            self._state(run, index, "running", ("pending",))
            self._emit(run, index, "running")
            self._started[(run, index)] = time.monotonic()

    def progress(self, run, index, done, total, unit) -> None:
        with self._lock:
            self._open("progress")
            self._state(run, index, "progress", ("running",))
            self._update_progress(run, index, _progress({"done": done, "total": total, "unit": unit}))

    def end(self, run, index, state, reason=None, ran_as=None) -> None:
        """One terminal state: from ``running`` done, failed or cancelled;
        from ``pending`` skipped or not_run."""
        with self._lock:
            self._open("end")
            if state in _RAN:
                allowed = ("running",)
            elif state in TERMINAL:
                allowed = ("pending",)
            else:
                raise ValueError(f"{state!r} is not a terminal state")
            self._state(run, index, state, allowed)
            self._finish(run, index, state, reason=reason, ran_as=ran_as)
            self._feed_parent(run)

    def close(self, cause, reason=None) -> None:
        """Close every open step, innermost run first, and seal.

        ``stop``: running -> cancelled, pending -> not_run.
        ``error``: running -> failed, pending -> not_run.
        """
        with self._lock:
            self._open("close")
            if cause not in ("stop", "error"):
                raise ValueError(f"close on {cause!r}, not stop or error")
            if reason is not None and not isinstance(reason, str):
                raise ValueError(f"reason must be a string, not {reason!r}")
            ended = "cancelled" if cause == "stop" else "failed"
            for run in reversed(list(self._runs)):
                open_steps = sorted(index for (owner, index), frame in self._last.items()
                                    if owner == run and frame["state"] not in TERMINAL)
                for index in open_steps:
                    state = self._last[(run, index)]["state"]
                    self._finish(run, index, ended if state == "running" else "not_run",
                                 reason=reason)
            self._sealed = True

    def summary(self) -> list[dict]:
        """The last frame of every step, in first-emission order."""
        with self._lock:
            return [_copy(frame) for frame in self._last.values()]


def emit(steps, event: str, *args, **kwargs):
    """Call one recorder event; None without a recorder or on its error."""
    if steps is None:
        return None
    try:
        return getattr(steps, event)(*args, **kwargs)
    except Exception as exc:
        logger.warning(f"pipeline_step {event} refused: {exc}")
        return None
