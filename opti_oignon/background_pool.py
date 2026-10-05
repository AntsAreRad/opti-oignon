#!/usr/bin/env python3
"""Background work in worker processes that yield to everything else.

Background CPU work, the chunking of an indexing job first, runs in worker
processes, never on a thread of the server. Each worker, as it starts and
before any task, puts every thread it has in SCHED_IDLE, in the idle I/O
class, and on the CPUs the governor's background plan gives it (the cores
outside the user's reserve); every thread it starts later inherits all
three. A step the kernel refuses is left as it was and said, by its errno
name, in the worker's entry report.

Why processes: a thread of the server lowered to the idle class would stall
the interactive path behind any lock it holds, and every thread a library
spawned from it would inherit the class. A worker has its own interpreter
lock and its own library pools, so neither can happen. Workers start with
"spawn", a fresh interpreter that imports only what a task names; never by
fork of a multi-threaded server, and not by "forkserver", whose Unix socket
a sandbox may refuse.

The pool starts its workers with its first task and closes them after the
plan's idle delay without one, so an idle server holds no worker. It serves
background callers only. A task sent alone runs in a one-worker executor of
its own, so a worker that dies there names its task; a broken task retires
the executor it was sent to and no later one, since the workers are shared
by every caller. With no plan (no governor), a plan that says off,
or workers that cannot start, a task runs on the caller's thread, as all
background work did before the pool, and the pool says so and why. As the
interpreter exits, every pool closes at once, before the standard library
would wait for the tasks already queued: those not started are cancelled
and the workers terminated, so stopping the server never waits for a
chunking.

The status names, for each device the background reads a file from, the
disks behind it and what each does with the idle I/O class: read from the
kernel (the scheduler each disk's queue has selected, the cgroup's
io.prio.class), never assumed.

Not proven here: what a background index spares the user's programs on a
busy machine, the I/O class's effect, which a disk whose scheduler is
"none" does not honour, and the disks of a file on a real btrfs, read here
from a mount table written as the kernel writes it.
"""

from __future__ import annotations

import atexit
import ctypes
import errno
import functools
import logging
import multiprocessing
import os
import platform
import re
import struct
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BACKGROUND_CLASS = "background"
IOPRIO_CLASS_IDLE = 3
_IOPRIO_CLASS_SHIFT = 13
_IOPRIO_WHO_PROCESS = 1
# ioprio_set and ioprio_get by machine, where the number is known: x86-64,
# and the generic table arm64 and riscv64 share. Any other machine is told
# "unsupported" rather than handed a number that may name another call.
_IOPRIO_CALLS = {
    "x86_64": (251, 252),
    "aarch64": (30, 31),
    "arm64": (30, 31),
    "riscv64": (30, 31),
}
# The width of this interpreter's pointers, in bytes. The machine names the
# kernel, not the interpreter: a 32-bit interpreter on a 64-bit kernel calls
# through the 32-bit table, where the numbers above name other calls (252 is
# exit_group on i386). The table holds for a 64-bit interpreter only.
_POINTER_SIZE = struct.calcsize("P")

# What this process's entry set or was refused; empty outside a worker.
_ENTRY: dict[str, Any] = {}
_LIBC: Any = None
# The class a caller has when there is no governor to ask.
_NO_GOVERNOR = object()


class Unsupported(Exception):
    """The machine has no known call for a step of the entry."""


def ioprio_calls() -> tuple[int, int] | None:
    """(ioprio_set, ioprio_get) for this machine and this interpreter, or
    None where unknown: the table's numbers are the 64-bit ABI's, so an
    interpreter of another width is given none, whatever the machine."""
    if _POINTER_SIZE != 8:
        return None
    return _IOPRIO_CALLS.get(platform.machine())


def _syscall(*args: int) -> int:
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(None, use_errno=True)
    return _LIBC.syscall(*args)


def _ioprio_set(tid: int, klass: int) -> None:
    calls = ioprio_calls()
    if calls is None:
        raise Unsupported(
            f"no I/O priority call is known for {platform.machine()} with a {_POINTER_SIZE * 8}-bit interpreter"
        )
    if _syscall(calls[0], _IOPRIO_WHO_PROCESS, tid, klass << _IOPRIO_CLASS_SHIFT) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _ioprio_class(tid: int) -> int | None:
    """The I/O class of thread ``tid``, or None where it cannot be read."""
    calls = ioprio_calls()
    if calls is None:
        return None
    value = _syscall(calls[1], _IOPRIO_WHO_PROCESS, tid)
    return None if value < 0 else value >> _IOPRIO_CLASS_SHIFT


class _KernelOps:
    """The calls the entry makes: one step on one thread at a time."""

    def threads(self) -> list[int]:
        return sorted(int(name) for name in os.listdir("/proc/self/task"))

    def set_idle(self, tid: int) -> None:
        os.sched_setscheduler(tid, os.SCHED_IDLE, os.sched_param(0))

    def set_io_idle(self, tid: int) -> None:
        _ioprio_set(tid, IOPRIO_CLASS_IDLE)

    def set_cpus(self, tid: int, cpus: tuple[int, ...]) -> None:
        os.sched_setaffinity(tid, cpus)


_KERNEL = _KernelOps()


def _refusal(exc: BaseException) -> str:
    if isinstance(exc, Unsupported):
        return f"unsupported: {exc}"
    code = getattr(exc, "errno", None)
    return f"refused: {errno.errorcode.get(code, code)}"


def _enter_background(cpus: tuple[int, ...], owner_pid: int, reports: Any = None, ops: Any = None) -> dict[str, Any]:
    """Enter the background, in a worker, before any task.

    Every thread of the process goes to SCHED_IDLE, to the idle I/O class,
    and onto ``cpus`` (left as inherited when empty). A step the kernel
    refuses is left as it was and named in the outcome, by its errno; the
    other steps are still made, and a thread gone meanwhile is passed. The
    outcome is kept for the census and put on ``reports`` for the owner.

    Refuses, before any call, to run in the process that owns the pool:
    nothing here ever lowers a thread of the server.
    """
    if os.getpid() == owner_pid:
        raise RuntimeError("the background entry never runs in the process that owns the pool")
    ops = ops if ops is not None else _KERNEL
    cpus = tuple(cpus)
    outcome: dict[str, Any] = {"pid": os.getpid(), "policy": "idle", "io": "idle"}
    outcome["cpus"] = "set" if cpus else "inherited"
    steps: list[tuple[str, Callable[[int], None]]] = [("policy", ops.set_idle), ("io", ops.set_io_idle)]
    if cpus:
        steps.append(("cpus", lambda tid: ops.set_cpus(tid, cpus)))
    for tid in ops.threads():
        for key, step in steps:
            try:
                step(tid)
            except ProcessLookupError:
                continue
            except (OSError, Unsupported) as exc:
                if outcome[key] in ("idle", "set"):
                    outcome[key] = _refusal(exc)
    _ENTRY.clear()
    _ENTRY.update(outcome)
    said = {k: v for k, v in outcome.items() if isinstance(v, str) and v.startswith(("refused", "unsupported"))}
    if said:
        logger.warning("Background worker %s entered with %s", outcome["pid"], said)
    if reports is not None:
        try:
            reports.put(dict(outcome))
        except Exception as exc:  # noqa: BLE001 - the report is a courtesy; the entry stands
            logger.debug("Background worker entry not reported: %s", exc)
    return outcome


def census() -> dict[str, Any]:
    """This process as the kernel sees it: its pid, its entry outcome (empty
    outside a worker), and each thread's scheduling policy, I/O class and
    CPUs, read from /proc/self/task."""
    threads = []
    for tid in _KERNEL.threads():
        try:
            threads.append(
                {
                    "tid": tid,
                    "policy": os.sched_getscheduler(tid),
                    "io_class": _ioprio_class(tid),
                    "cpus": sorted(os.sched_getaffinity(tid)),
                }
            )
        except ProcessLookupError:
            continue
    return {"pid": os.getpid(), "entry": dict(_ENTRY), "threads": threads}


def _inline(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Future:
    """``fn`` run on this thread, as a completed future."""
    future: Future = Future()
    future.set_running_or_notify_cancel()
    try:
        result = fn(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - the future carries it to the caller
        future.set_exception(exc)
    else:
        future.set_result(result)
    return future


def _governor() -> Any:
    try:
        from opti_oignon.resource_governor import get_resource_governor

        return get_resource_governor()
    except Exception as exc:  # noqa: BLE001 - absence is an answer here
        logger.debug("Resource governor unavailable to the background pool: %s", exc)
        return None


def _governor_plan() -> Any:
    governor = _governor()
    if governor is None:
        return None
    try:
        return governor.plan_background()
    except Exception as exc:  # noqa: BLE001 - no plan: the work runs inline
        logger.warning("Background plan unavailable: %s", exc)
        return None


def _governor_class(caller: str) -> Any:
    governor = _governor()
    if governor is None:
        return _NO_GOVERNOR
    try:
        return governor.caller_class(caller)
    except Exception as exc:  # noqa: BLE001 - unclassed: refused where it matters
        logger.warning("Caller class unavailable to the background pool: %s", exc)
        return None


def _alive(process: Any) -> bool:
    try:
        return bool(process.is_alive())
    except Exception:  # noqa: BLE001 - a process that cannot say is not counted
        return False


def _remains(executor: Any) -> tuple[list[Any], Any]:
    """An executor's worker processes and its result queue, read before its
    shutdown, which forgets both."""
    processes = list((getattr(executor, "_processes", None) or {}).values())
    return processes, getattr(executor, "_result_queue", None)


def _end(remains: tuple[list[Any], Any]) -> None:
    """End an executor's workers at once, then close the server's copy of
    the write end of its result pipe, so that its manager, were it waiting
    inside a frame a worker left half sent, meets the end of the pipe."""
    processes, results = remains
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except Exception as exc:  # noqa: BLE001 - one worker left leaves the others ended
            logger.debug("Background worker not ended at exit: %s", exc)
    writer = getattr(results, "_writer", None)
    if writer is not None:
        try:
            writer.close()
        except Exception as exc:  # noqa: BLE001 - the manager then waits on its own
            logger.debug("Background result pipe not closed at exit: %s", exc)


def _spawn_executor(workers: int, initializer: Callable[..., Any], initargs: tuple) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )


class BackgroundPool:
    """Worker processes for background callers, started on demand.

    ``plan`` returns the background plan (None: no plan, tasks run inline);
    ``class_of`` names a caller's admission class; ``executor_factory``
    builds the executor from (workers, initializer, initargs). Each
    defaults to the governor and a "spawn" process pool.
    """

    def __init__(
        self,
        *,
        plan: Callable[[], Any] | None = None,
        class_of: Callable[[str], Any] | None = None,
        executor_factory: Callable[[int, Callable[..., Any], tuple], Any] | None = None,
    ):
        self._plan_fn = plan if plan is not None else _governor_plan
        self._class_of = class_of if class_of is not None else _governor_class
        self._factory = executor_factory if executor_factory is not None else _spawn_executor
        self._lock = threading.Lock()
        self._executor: Any = None
        self._retiring: list[Any] = []
        # The worker processes and result queue of each executor retired
        # without waiting whose workers may still run a task: its shutdown
        # forgot both, and the exit still has to end them.
        self._orphans: list[tuple[list[Any], Any]] = []
        self._reports: Any = None
        self._entries: dict[int, dict[str, Any]] = {}
        self._plan: Any = None
        self._mode = "idle"
        self._reason: str | None = None
        self._failed: str | None = None
        self._pending = 0
        self._generation = 0
        self._timer: threading.Timer | None = None
        self._closed = False
        # The generation of the executor each task was sent to, so a broken
        # task retires its own executor and never a later one.
        self._origin: weakref.WeakKeyDictionary[Future, int] = weakref.WeakKeyDictionary()
        # The one-worker executors of tasks sent alone, each with the queue
        # its worker reports its entry on.
        self._alone: dict[Any, Any] = {}
        _LIVE_POOLS.add(self)

    # -- tasks ---------------------------------------------------------------

    def submit(self, fn: Callable[..., Any], /, *args: Any, caller: str, **kwargs: Any) -> Future:
        """Run ``fn(*args, **kwargs)`` for ``caller``: in a worker, or on this
        thread while the pool runs inline. A caller the governor does not
        class as background is refused, and nothing reaches a worker."""
        klass = self._background_class(caller)
        for _attempt in range(2):
            closing = None
            with self._lock:
                if self._closed:
                    raise RuntimeError("the background pool is shut down")
                executor = self._start()
                if executor is None:
                    break
                if klass is _NO_GOVERNOR:
                    raise PermissionError(f"no governor to class caller {caller!r}")
                self._cancel_timer()
                generation = self._generation
                try:
                    future = executor.submit(fn, *args, **kwargs)
                except BrokenProcessPool:
                    closing = self._detach("idle")
                except Exception as exc:  # noqa: BLE001 - the workers cannot run: inline, said
                    closing = self._detach("inline")
                    self._failed = self._reason = f"the workers did not start: {exc}"
                else:
                    self._pending += 1
                    self._origin[future] = generation
            if closing is not None:
                self._retire(closing, wait=False)
                continue
            future.add_done_callback(functools.partial(self._task_done, generation))
            return future
        else:
            raise BrokenProcessPool("the background workers broke twice as the task was sent")
        return _inline(fn, args, kwargs)

    def in_flight_limit(self) -> int:
        """How many tasks a caller may keep in the pool at once: the plan's
        workers times its tasks per worker, or one while it runs inline."""
        with self._lock:
            plan = self._plan if self._executor is not None else self._read_plan()
            failed = self._failed
        workers = getattr(plan, "workers", 0) or 0
        if failed or workers < 1:
            return 1
        return max(1, workers * max(1, int(getattr(plan, "in_flight", 1) or 1)))

    def submit_alone(self, fn: Callable[..., Any], /, *args: Any, caller: str, **kwargs: Any) -> Future:
        """Run ``fn(*args, **kwargs)`` for ``caller`` in a worker of its own:
        a one-worker executor that no other task shares, entering as every
        worker does on the plan's CPUs, and retired once the task is done.
        A worker that dies there was killed by this task and no other. On
        this thread while the pool runs inline; a caller is refused as by
        ``submit``. A worker of its own that cannot start turns the pool
        inline, said, as ``submit``'s do."""
        klass = self._background_class(caller)
        made: tuple[Any, Any] | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("the background pool is shut down")
            plan = self._worker_plan()
            if plan is not None:
                if klass is _NO_GOVERNOR:
                    raise PermissionError(f"no governor to class caller {caller!r}")
                executor = reports = None
                try:
                    reports = multiprocessing.get_context("spawn").SimpleQueue()
                    executor = self._factory(1, _enter_background, (tuple(plan.cpus), os.getpid(), reports))
                    future = executor.submit(fn, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - the workers cannot run: inline, said
                    self._failed = f"the workers did not start: {exc}"
                    if self._executor is None:
                        self._mode, self._reason = "inline", self._failed
                    made = (executor, reports)
                    plan = None
                else:
                    self._alone[executor] = reports
        if plan is None:
            if made is not None:
                self._discard(*made)
            return _inline(fn, args, kwargs)
        future.add_done_callback(functools.partial(self._alone_done, executor))
        return future

    def reset(self, broken: Future) -> None:
        """Retire the executor the task ``broken`` was sent to, if it is
        still the current one: the next task starts fresh workers. A task
        of an executor already replaced, one sent alone or one run inline
        retires nothing, so a job that sees its task break never retires
        the fresh executor that other tasks, of any job, already run in."""
        with self._lock:
            try:
                origin = self._origin.get(broken)
            except TypeError:  # not a task of this pool
                origin = None
            current = self._executor is not None and origin == self._generation
            executor = self._detach("idle") if current else None
        if executor is not None:
            self._retire(executor, wait=False)

    @property
    def closed(self) -> bool:
        """Whether the pool is shut down: a task it broke after that broke
        of the closing, and says nothing of its file."""
        with self._lock:
            return self._closed

    def shutdown(self) -> None:
        """Close the workers, those of tasks sent alone too, and wait for
        them to exit; no task after. The workers of executors retired
        without waiting are ended, as the exit ends them."""
        with self._lock:
            self._closed = True
            executor = self._detach("closed") if self._executor is not None else None
            self._mode = "closed"
            alone = list(self._alone)
            orphans = list(self._orphans)
            self._orphans.clear()
        _LIVE_POOLS.discard(self)
        if executor is not None:
            self._retire(executor, wait=True)
        for own in alone:
            self._retire_alone(own)
        for remains in orphans:
            _end(remains)
            for process in remains[0]:
                try:
                    process.join()
                except Exception as exc:  # noqa: BLE001 - one worker left leaves the others joined
                    logger.debug("Background worker not joined at shutdown: %s", exc)

    def _abandon(self) -> None:
        """Close the pool at once, as the interpreter exits: no task after,
        the tasks not yet started cancelled, and every worker terminated
        rather than waited for, those of executors retired without waiting
        too. No executor is joined here; the standard library's own exit
        hook joins their manager threads next. A worker ended while it sent
        its result leaves part of a frame in the result pipe, which the
        manager reads for as long as any write end is open: the server's own
        copy is closed once the workers are ended, so the manager meets the
        end of the pipe and ends."""
        with self._lock:
            self._closed = True
            if self._executor is not None:
                self._detach("closed")
            self._mode = "closed"
            executors = [*self._retiring, *self._alone]
            ending = list(self._orphans)
            self._orphans.clear()
        _LIVE_POOLS.discard(self)
        for executor in executors:
            try:
                remains = _remains(executor)  # read before the shutdown, which forgets them
                executor.shutdown(wait=False, cancel_futures=True)
                ending.append(remains)
            except Exception as exc:  # noqa: BLE001 - one executor left leaves the others closed
                logger.debug("Background executor not closed at exit: %s", exc)
        for remains in ending:
            _end(remains)

    # -- what it is doing ------------------------------------------------------

    def worker_pids(self) -> tuple[int, ...]:
        """The pids of the live workers, closing ones and those of tasks
        sent alone included (each executor's process table, read; empty
        when no worker runs)."""
        with self._lock:
            executors = [self._executor, *self._retiring, *self._alone]
            self._orphans = [remains for remains in self._orphans if any(_alive(p) for p in remains[0])]
            orphaned = [p for remains in self._orphans for p in remains[0]]
        pids: set[int] = set()
        for executor in executors:
            processes = getattr(executor, "_processes", None) or {}
            pids.update(pid for pid, process in list(processes.items()) if process.is_alive())
        pids.update(p.pid for p in orphaned if _alive(p) and p.pid is not None)
        return tuple(sorted(pids))

    def status(self) -> dict[str, Any]:
        """The pool's mode ("idle", "process", "inline" or "closed"), why it
        runs inline, its plan, its tasks in flight (those sent alone
        included), and what each live worker's entry set or was refused."""
        alive = self.worker_pids()
        with self._lock:
            self._drain()
            plan = self._plan
            return {
                "mode": "process" if self._mode == "idle" and self._alone else self._mode,
                "reason": self._reason,
                "plan": None
                if plan is None
                else {
                    "workers": plan.workers,
                    "cpus": list(plan.cpus),
                    "reserved": list(plan.reserved),
                    "source": plan.source,
                },
                "pending": self._pending + len(self._alone),
                "workers": [dict(self._entries[pid]) for pid in alive if pid in self._entries],
            }

    # -- inside the lock -------------------------------------------------------

    def _read_plan(self) -> Any:
        self._plan = self._plan_fn()
        return self._plan

    def _worker_plan(self) -> Any:
        """The plan workers run on: the current executor's, else the one
        read now; None while the pool runs inline (the reason said)."""
        if self._executor is not None:
            return self._plan
        if self._failed:
            self._mode, self._reason = "inline", self._failed
            return None
        plan = self._read_plan()
        if plan is None:
            self._mode, self._reason = "inline", "no governor to plan the background"
            return None
        if (getattr(plan, "workers", 0) or 0) < 1:
            self._mode, self._reason = "inline", f"the background plan says {getattr(plan, 'source', 'off')}"
            return None
        return plan

    def _start(self) -> Any:
        """The executor, started if need be; None while the pool runs inline
        (the reason said)."""
        if self._executor is not None:
            return self._executor
        plan = self._worker_plan()
        if plan is None:
            return None
        try:
            reports = multiprocessing.get_context("spawn").SimpleQueue()
            executor = self._factory(plan.workers, _enter_background, (tuple(plan.cpus), os.getpid(), reports))
        except Exception as exc:  # noqa: BLE001 - the workers cannot run: inline, said
            self._failed = f"the workers did not start: {exc}"
            self._mode, self._reason = "inline", self._failed
            return None
        self._executor, self._reports = executor, reports
        self._generation += 1
        self._mode, self._reason = "process", None
        return executor

    def _detach(self, mode: str) -> Any:
        executor, self._executor = self._executor, None
        self._cancel_timer()
        self._generation += 1
        self._pending = 0
        self._entries.clear()
        self._mode = mode
        if self._reports is not None:
            self._reports.close()
            self._reports = None
        if executor is not None:
            self._retiring.append(executor)
        return executor

    def _drain(self) -> None:
        for reports in (self._reports, *self._alone.values()):
            while reports is not None:
                try:
                    if reports.empty():
                        break
                    entry = reports.get()
                except (OSError, EOFError, ValueError):
                    break
                if isinstance(entry, dict) and isinstance(entry.get("pid"), int):
                    self._entries[entry["pid"]] = entry

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    # -- outside the lock ------------------------------------------------------

    def _background_class(self, caller: str) -> Any:
        """The class of ``caller``; a caller the governor does not class as
        background is refused."""
        klass = self._class_of(caller)
        if klass is not _NO_GOVERNOR and klass != BACKGROUND_CLASS:
            raise PermissionError(
                f"caller {caller!r} is of class {klass!r}: the background pool serves background callers only"
            )
        return klass

    def _task_done(self, generation: int, _future: Future) -> None:
        with self._lock:
            if generation != self._generation or self._executor is None:
                return
            self._pending = max(0, self._pending - 1)
            if self._pending == 0 and not self._closed:
                idle_s = float(getattr(self._plan, "idle_s", 0.0) or 0.0)
                timer = threading.Timer(idle_s, self._close_if_idle, args=(self._generation,))
                timer.daemon = True
                timer.name = "background-pool-idle"
                self._cancel_timer()
                self._timer = timer
                timer.start()

    def _close_if_idle(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self._pending or self._executor is None:
                return
            executor = self._detach("idle")
        self._retire(executor, wait=True)

    def _retire(self, executor: Any, *, wait: bool) -> None:
        """Shut ``executor`` down; it stays among the live workers' owners
        until its shutdown returns (joined, when ``wait``). Retired without
        waiting, its workers may still run a task: they are kept, with its
        result queue, for the exit to end."""
        remains = _remains(executor)
        try:
            executor.shutdown(wait=wait, cancel_futures=True)
        finally:
            with self._lock:
                if executor in self._retiring:
                    self._retiring.remove(executor)
                # Only workers still alive are kept: one gone holds nothing
                # the exit needs, and its pipes go with it.
                self._orphans = [kept for kept in self._orphans if any(_alive(p) for p in kept[0])]
                if not wait and any(_alive(p) for p in remains[0]):
                    self._orphans.append(remains)

    def _alone_done(self, executor: Any, _future: Future) -> None:
        """A task sent alone is done: its executor is retired on a helper
        thread, never on the thread this callback runs on (the executor's
        own, where a shutdown that waits would wait for itself)."""
        try:
            threading.Thread(
                target=self._retire_alone, args=(executor,), name="background-pool-alone", daemon=True
            ).start()
        except RuntimeError:  # no new thread as the interpreter exits
            self._retire_alone(executor, wait=False)

    def _retire_alone(self, executor: Any, wait: bool = True) -> None:
        """Shut the executor of a task sent alone down; it stays among the
        live workers' owners until its shutdown returns."""
        try:
            executor.shutdown(wait=wait, cancel_futures=True)
        except Exception as exc:  # noqa: BLE001 - its one worker goes either way
            logger.debug("Background executor of a task sent alone not shut down: %s", exc)
        finally:
            with self._lock:
                reports = self._alone.pop(executor, None)
            if reports is not None:
                reports.close()

    def _discard(self, executor: Any, reports: Any) -> None:
        """Drop what a worker of its own that did not start left behind: its
        executor, and the worker process it may have spawned before failing,
        ended at once (left, it would block on its queue and hold the
        exit)."""
        try:
            if executor is not None:
                remains = _remains(executor)  # read before the shutdown, which forgets them
                executor.shutdown(wait=False, cancel_futures=True)
                _end(remains)
            if reports is not None:
                reports.close()
        except Exception as exc:  # noqa: BLE001 - nothing ran there
            logger.debug("Background executor that did not start not dropped: %s", exc)


_POOL: BackgroundPool | None = None
_POOL_LOCK = threading.Lock()
# Every pool not yet shut down, closed at once as the interpreter exits.
_LIVE_POOLS: weakref.WeakSet[BackgroundPool] = weakref.WeakSet()


def _close_live_pools() -> None:
    """Close every live pool as the interpreter exits (BackgroundPool's
    ``_abandon``): no task after, and no queued task waited for."""
    for pool in list(_LIVE_POOLS):
        try:
            pool._abandon()
        except Exception as exc:  # noqa: BLE001 - one pool left open leaves the others closed
            logger.debug("Background pool not closed at exit: %s", exc)


def _register_exit_hook(hook: Callable[[], None]) -> None:
    """Run ``hook`` as the interpreter exits, before the standard library
    waits for the tasks of its process pools.

    concurrent.futures.process registers that wait with threading's own
    exit hooks, which run before any atexit callback and in the reverse
    order of their registration; this module imports it first, so a hook
    registered there after it runs before it. Where that private call is
    missing, atexit is the fallback, which runs after the wait."""
    register = getattr(threading, "_register_atexit", None)
    if register is not None:
        try:
            register(hook)
            return
        except RuntimeError:  # the interpreter is already shutting down
            pass
    atexit.register(hook)


_register_exit_hook(_close_live_pools)


def get_background_pool() -> BackgroundPool:
    """The server's background pool, closed at interpreter exit by the
    module's exit hook, before the standard library would wait for the
    tasks already queued."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = BackgroundPool()
        return _POOL


def background_state() -> dict[str, Any]:
    """The server's pool for the status surface: the pool's own status once
    background work has asked for it, "unused" before. Reading it never
    creates the pool."""
    with _POOL_LOCK:
        pool = _POOL
    if pool is None:
        return {
            "mode": "unused",
            "reason": "no background work has asked for the pool since the server started",
            "plan": None,
            "pending": 0,
            "workers": [],
        }
    return pool.status()


# ---------------------------------------------------------------------------
# The disks the background reads, and what each does with the idle I/O class
# ---------------------------------------------------------------------------

# What a scheduler does with a request in the idle I/O class. Only bfq and
# mq-deadline order requests by class (the kernel's block/ioprio document):
# bfq serves the classes in strict order, the idle class with a thin share
# that keeps it from starving; mq-deadline serves a lower class after the
# higher ones unless it has waited its queue's prio_aging_expire. kyber
# schedules by the kind of request and reads no priority; "none" schedules
# nothing.
_IDLE_CLASS = {"bfq": "honored", "mq-deadline": "deferred", "kyber": "no_effect", "none": "no_effect"}
# A cgroup's io.prio.class (blk-ioprio): these leave a request of the idle
# class idle; promote-to-rt, and none-to-rt its older name, make every
# request that is not real-time a real-time one.
_KEEPS_IDLE = frozenset({"no-change", "restrict-to-be", "idle"})
_TO_REAL_TIME = frozenset({"promote-to-rt", "none-to-rt"})
# How deep a stack of mappings and arrays is followed down to its disks.
_STACK_DEPTH = 8
_MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")


def device_disks(
    device: int,
    *,
    path: str | os.PathLike[str] | None = None,
    sys_root: str | os.PathLike[str] = "/sys",
    mountinfo: str | os.PathLike[str] = "/proc/self/mountinfo",
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    self_cgroup: str | os.PathLike[str] = "/proc/self/cgroup",
) -> dict[str, Any]:
    """The disks behind ``device`` (a file's st_dev), and what each does
    with the idle I/O class the background workers read in.

    A partition is read on its disk; a device-mapper mapping or an md array
    on the disks under it (the request it sends down is a clone, and a
    clone carries the class); a file system with no device number of its
    own (btrfs, say) on the device its mount names. That mount is the one
    whose mountinfo line carries ``device``, else, ``path`` given (a file
    on ``device``), the one that holds the path: a btrfs subvolume's files
    carry a device number of their own that no line shows (a line carries
    the superblock's). Each disk names the scheduler its queue has selected
    and, under mq-deadline, the aging in milliseconds after which a request
    of a lower class is served anyway. The policy is the io.prio.class of
    the nearest cgroup of this process that names one. ``idle_class`` is
    the disks' effect, "mixed" where they differ; a device with no disk to
    read is "unknown", and says why. Nothing is assumed: what is not read
    is unknown.
    """
    number = f"{os.major(device)}:{os.minor(device)}"
    policy = io_class_policy(cgroup_root=cgroup_root, self_cgroup=self_cgroup)
    found: dict[str, Any] = {"device": number, "disks": [], "policy": policy, "idle_class": "unknown", "reason": None}
    try:
        node, reason = _block_node(device, Path(sys_root), Path(mountinfo), path)
        leaves = _leaf_disks(node) if node is not None else []
    except OSError as exc:
        node, reason, leaves = None, f"{number}: {exc.strerror or exc}", []
    if node is not None and not leaves:
        reason = f"no disk under {number}"
    if not leaves:
        found["reason"] = reason
        return found
    for leaf in sorted(leaves, key=lambda path: path.name):
        scheduler, aging = _scheduler_of(leaf)
        found["disks"].append(
            {"name": leaf.name, "scheduler": scheduler, "aging_ms": aging, "idle_class": _idle_effect(scheduler, policy)}
        )
    effects = {disk["idle_class"] for disk in found["disks"]}
    found["idle_class"] = effects.pop() if len(effects) == 1 else "mixed"
    return found


def io_class_policy(
    *,
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    self_cgroup: str | os.PathLike[str] = "/proc/self/cgroup",
) -> str | None:
    """The io.prio.class of the nearest cgroup of this process that names
    one, from its own up to the root: the policy its requests take. None
    where none names one, or the cgroup cannot be read."""
    try:
        lines = Path(self_cgroup).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    relative = next((line[3:] for line in lines if line.startswith("0::")), None)
    if relative is None:
        return None
    root = Path(cgroup_root)
    here = root / relative.strip().lstrip("/")
    while True:
        try:
            named = (here / "io.prio.class").read_text(encoding="utf-8").strip()
        except OSError:
            named = ""
        if named:
            return named
        if here == root or root not in here.parents:
            return None
        here = here.parent


def _block_node(
    device: int, sys_root: Path, mountinfo: Path, path: str | os.PathLike[str] | None = None
) -> tuple[Path | None, str | None]:
    """The sysfs directory of the block device behind ``device``, or None
    and why. A device with no number of its own is read on the mount whose
    line carries it, else on the mount that holds ``path``."""
    major, minor = os.major(device), os.minor(device)
    if major != 0:
        node = sys_root / "dev" / "block" / f"{major}:{minor}"
        if not node.exists():
            return None, f"the kernel lists no block device {major}:{minor}"
        return node.resolve(), None
    mount = _mount_source(mountinfo, f"0:{minor}")
    if mount is None and path is not None:
        real = os.path.realpath(os.fspath(path))
        mount = _mount_holding(mountinfo, real)
        if mount is None:
            return None, f"no mount names device 0:{minor}, and no mount point holds {real}"
    if mount is None:
        return None, f"no mount names device 0:{minor}"
    fstype, source = mount
    if not source.startswith("/dev/"):
        return None, f"{fstype} {source} has no block device"
    node = _named_block(sys_root, source)
    if node is None:
        return None, f"{fstype} {source}: the kernel lists no block device of that name"
    return node, None


def _mount_lines(mountinfo: Path) -> list[tuple[str, str, str, str]]:
    """(device number, mount point, file system type, source) of each line
    of a mountinfo table, in its order, the octal escapes undone."""
    lines = []
    for line in mountinfo.read_text(encoding="utf-8").splitlines():
        parts = line.split(" - ", 1)
        head = parts[0].split()
        tail = parts[1].split() if len(parts) == 2 else []
        if len(head) >= 5 and len(tail) >= 2:
            lines.append((head[2], _unescaped(head[4]), tail[0], _unescaped(tail[1])))
    return lines


def _unescaped(field: str) -> str:
    return _MOUNT_ESCAPE.sub(lambda m: chr(int(m.group(1), 8)), field)


def _mount_source(mountinfo: Path, number: str) -> tuple[str, str] | None:
    """(file system type, source) of the first mount whose files carry the
    device number ``number``, from a mountinfo table; None if none does."""
    for found, _point, fstype, source in _mount_lines(mountinfo):
        if found == number:
            return fstype, source
    return None


def _mount_holding(mountinfo: Path, real: str) -> tuple[str, str] | None:
    """(file system type, source) of the mount that holds the real path
    ``real``: the one whose mount point is the longest that prefixes it by
    whole components, the later line winning on the same point (a later
    mount shadows an earlier one); None if no mount point holds it."""
    held: tuple[str, str] | None = None
    longest = -1
    for _number, point, fstype, source in _mount_lines(mountinfo):
        holds = real == point or real.startswith(point.rstrip("/") + "/")
        if holds and len(point) >= longest:
            held, longest = (fstype, source), len(point)
    return held


def _named_block(sys_root: Path, source: str) -> Path | None:
    """The sysfs directory of the block device a mount names by its node
    (or a link to one), or by a mapping's name under /dev/mapper; None."""
    blocks = sys_root / "class" / "block"
    for name in (os.path.basename(os.path.realpath(source)), os.path.basename(source)):
        if name and (blocks / name).exists():
            return (blocks / name).resolve()
    if source.startswith("/dev/mapper/"):
        wanted = source[len("/dev/mapper/"):]
        for entry in sorted(blocks.glob("dm-*")):
            try:
                if (entry / "dm" / "name").read_text(encoding="utf-8").strip() == wanted:
                    return entry.resolve()
            except OSError:
                continue
    return None


def _leaf_disks(node: Path, depth: int = 0, seen: set[Path] | None = None) -> list[Path]:
    """The whole disks under the block device at ``node``: itself when it
    is one, its disk when it is a partition, and the disks under each
    device a mapping or an array sits on (its slaves), each once."""
    seen = set() if seen is None else seen
    if depth > _STACK_DEPTH or node in seen:
        return []
    seen.add(node)
    slaves = node / "slaves"
    under = sorted(slaves.iterdir()) if slaves.is_dir() else []
    if under:
        leaves: list[Path] = []
        for entry in under:
            leaves.extend(_leaf_disks(entry.resolve(), depth + 1, seen))
        return leaves
    if (node / "partition").is_file():
        return _leaf_disks(node.parent, depth + 1, seen)
    return [node]


def _scheduler_of(disk: Path) -> tuple[str | None, int | None]:
    """The scheduler a disk's queue has selected and, under mq-deadline,
    the aging its queue names (prio_aging_expire, in milliseconds); None
    for what cannot be read."""
    try:
        names = (disk / "queue" / "scheduler").read_text(encoding="utf-8").split()
    except OSError:
        return None, None
    chosen = [name[1:-1] for name in names if len(name) > 2 and name[0] == "[" and name[-1] == "]"]
    scheduler = chosen[0] if len(chosen) == 1 else (names[0] if len(names) == 1 else None)
    aging = None
    if scheduler == "mq-deadline":
        try:
            aging = int((disk / "queue" / "iosched" / "prio_aging_expire").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            aging = None
    return scheduler, aging


def _idle_effect(scheduler: str | None, policy: str | None) -> str:
    """What ``scheduler`` does with a request the background sends in the
    idle class, once ``policy`` (a cgroup's io.prio.class) has applied."""
    effect = _IDLE_CLASS.get(scheduler, "unknown") if scheduler else "unknown"
    if policy is None or policy in _KEEPS_IDLE:
        return effect
    if policy in _TO_REAL_TIME:
        return "promoted_to_rt" if effect in ("honored", "deferred") else effect
    return "unknown"
