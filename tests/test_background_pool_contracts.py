#!/usr/bin/env python3
"""Background work in processes that yield to everything else.

The indexing job chunked every file on a thread of the server, at the
server's own priority and on any core, so a large folder took CPU and disk
from the user's programs and from the server's interactive work alike; the
embeddings it asked for were admitted as a direct call, a class that may
evict. Lowering a thread of the server instead would have been worse: an
idle-class thread holding a lock the interactive path waits on stalls that
path, and every thread a library spawns from it inherits the class.

  The workers:
    * BW1 -- every thread of a worker is SCHED_IDLE: its first, and one a
      task starts after the worker's entry.
    * BW2 -- no thread of the process that owns the pool is idle, by policy
      or by I/O class, after the pool served; the entry refuses to run in
      that process.
    * BW3 -- every thread of a worker is in the idle I/O class.
    * BW4 -- every thread of a worker runs on the CPUs the plan gives it and
      on no other; a plan that names none leaves the affinity inherited.
    * BW5 -- a step the kernel refuses changes nothing, the other steps are
      still tried, and the entry says each refusal by its errno name; a
      machine with no known I/O call says so.
    * BW6 -- a file chunked in a worker gives what the chunker gives on the
      job's thread, at the settings the job passes.
    * BW7 -- a worker holds neither the vector store nor any project module
      past the pool and the chunker.
    * BW19 -- the I/O priority numbers are the interpreter's ABI's: a
      32-bit interpreter on a 64-bit kernel is told "unsupported" and makes
      no I/O priority call.

  The job:
    * BW8 -- a worker dying on a file fails that file alone, by name; the
      files in flight with it are chunked again and stored; the job
      completes, and the pool serves again.
    * BW9 -- each file is stored on the job's thread under the ticket the
      governor admitted for the "index" caller and the store's embedding
      model.
    * BW10 -- a refusal a wait can lift is asked again, after the plan's
      pause, until admitted, and the file waits rather than fails; a refusal
      no wait can lift fails the file with its reason.
    * BW11 -- a cancelled job gives the files in flight back to the queue
      and does not wait for their chunking.
    * BW25 -- a file whose task comes back cancelled is sent again and
      stored, never failed.
    * BW26 -- after a task breaks, the files that may have killed the worker
      are each sent in a worker of their own, one at a time.
    * BW27 -- a job cancelled while it waits for the chunking at the head of
      its flight leaves at once, gives the file back, and stores nothing.
    * BW28 -- a job cancelled while the governor's queue holds its admission
      leaves the queue at once, gives the file back, and stores nothing.
    * BW29 -- a governor whose admission takes no cancel is asked without
      it, and the file is stored under the ticket it admits.
    * BW30 -- while the governor says memory is short for the background,
      the job keeps one file in flight.
    * BW31 -- a job that fails mid-flight gives every file it holds back to
      the queue, and says why it failed.
    * BW32 -- an admission that lands as the job is cancelled is handed
      back: the load it admitted is ended, nothing is stored, and the file
      goes back to the queue.
    * BW33 -- a file sent alone whose worker the pool's own shutdown ends is
      not blamed: it goes back to the queue, and the job says the pool is
      shut down.
    * BW34 -- a worker ended halfway through sending its result does not
      hold the exit: the pool closes the server's write end of the result
      pipe once its workers are ended.
    * BW35 -- an executor retired without waiting is ended at exit with its
      running task.
    * BW36 -- each file is sent against an estimate of the memory its parse
      takes, charged with those in flight against the room the governor
      gives the background; a file larger than the room goes alone.
    * BW37 -- the room is shared by every job of the engine: a job waits for
      room rather than send beside another job's files that fill it.
    * BW38 -- a file sent alone, suspected of killing its worker, is charged
      against the room like any other.
    * BW39 -- a worker of its own whose executor fails as its task is sent
      is ended, not left blocked to hold the exit.
    * BW40 -- the workers of executors retired without waiting are kept only
      while they live, and the pool's shutdown ends those still alive.
    * BW41 -- a cancelled job keeps the room of a parse that runs on until
      it ends.
    * BW42 -- a job waiting for room is served before a job that already
      has files in flight sends more.

  The pool:
    * BW12 -- a caller of the interactive or user class is refused, and
      nothing reaches a worker; a background caller is served.
    * BW13 -- no worker exists before the first task; every worker exits
      after the plan's idle delay; none is left at shutdown.
    * BW14 -- a pool that cannot start, a plan that says off, or no governor
      to plan: the task runs on the caller's thread with the same result,
      and the pool says so and why.
    * BW22 -- a process that returns from main with tasks queued exits
      without running them: the pool closes itself before the standard
      library waits for an executor's tasks.
    * BW23 -- a broken task retires the executor it came from while that
      executor is current, and no other.
    * BW24 -- a task sent alone runs in a one-worker executor of its own,
      shut down once the task is done.

  The status:
    * BW15 -- the server's pool state says "unused" until background work
      asks for the pool, then is the pool's own status; reading it creates
      no pool.
    * BW16 -- the disks behind a device are read through partitions,
      device-mapper mappings, md arrays and mounts with no device number of
      their own, each with the scheduler its queue has selected; no disk is
      assumed, and a device with none says why.
    * BW17 -- what each disk does with the idle class is its scheduler's:
      honoured under bfq, deferred under mq-deadline up to its aging, no
      effect under kyber or none; a cgroup io.prio.class that promotes the
      class to real time is said.
    * BW18 -- the indexing job names the disks of the files it sent the
      background, each device once, through the job's route.
    * BW20 -- a file whose device number no mount shows (a btrfs
      subvolume's) is read on the mount that holds its path; a path no
      listed mount holds gives no disk and says why.
    * BW21 -- the job reads each device's disks with the path of the first
      file it sent from that device.

Proven in the container: real worker processes under the real kernel (the
policy, the I/O class and the CPUs are read back from /proc), and a child
interpreter that imports the real module and exits, with a scripted plan,
a stand-in store and a stand-in governor. What a background index spares
the user's programs on a busy machine is owed to the machine, and so are
the I/O class's effect, which a disk whose scheduler is "none" does not
honour, what one file in flight spares the memory when the governor says
it is short, and the disks of a file on a real btrfs, whose mount table the
contracts write as the kernel writes it.
"""

import contextlib
import dataclasses
import errno
import functools
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import types
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bw_tasks  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

_POOL = "opti_oignon.background_pool"
_CHUNKER = "opti_oignon.rag_chunker"
_INGEST = "opti_oignon.rag.batch_ingest"
_RG = "opti_oignon.resource_governor"
_CLOSERS = []
_WAIT_S = 60.0
# The refusals the stand-in governor holds for final, as the real one does.
_FINAL = frozenset({"background_capacity_unknown", "background_cost_unknown", "background_ctx_unknown"})


def _db_utils():
    """A db_utils stand-in whose safe_connect is plain sqlite."""
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(str(p), check_same_thread=kw.get("check_same_thread", False))
    return db


def _open(*, governor=None, ingest=False):
    """The pool and the chunker, and the indexing job when asked, in one
    window. The governor is the stand-in module given, else unreachable; the
    vector store and the sandbox manager are always unreachable."""
    targets = {_POOL: source("background_pool.py"), _CHUNKER: source("rag_chunker.py")}
    seeded = {"opti_oignon.db_utils": _db_utils()}
    blocked = ("opti_oignon.rag_store", "opti_oignon.sandbox_manager")
    if governor is not None:
        seeded[_RG] = governor
    else:
        blocked += (_RG,)
    packages = ()
    if ingest:
        targets[_INGEST] = source("rag", "batch_ingest.py")
        packages = ("opti_oignon.rag",)
    loaded, restore = isolate(targets=targets, blocked=blocked, seeded=seeded, packages=packages)
    _CLOSERS.append(restore)
    return loaded


def _project_modules():
    """Every project entry of the module cache, by identity."""
    return {k: v for k, v in sys.modules.items() if k == "opti_oignon" or k.startswith("opti_oignon.")}


@pytest.fixture(autouse=True)
def _left_as_found():
    """No contract may leave a project module changed."""
    before = _project_modules()
    yield
    while _CLOSERS:
        _CLOSERS.pop()()
    assert _project_modules() == before, "every project module is left as the contract found it"


def _plan(workers=1, cpus=(), *, in_flight=2, idle_s=60.0, held_retry_s=0.01, source="plan", reserved=()):
    """A background plan as the governor writes it."""
    return types.SimpleNamespace(
        workers=workers,
        cpus=tuple(cpus),
        reserved=tuple(reserved),
        in_flight=in_flight,
        idle_s=idle_s,
        held_retry_s=held_retry_s,
        source=source,
    )


def _live(bp, plan):
    """A pool of real worker processes on ``plan``, serving any caller."""
    return bp.BackgroundPool(plan=lambda: plan, class_of=lambda caller: "background")


def _decision(admitted=True, reason="", is_estop=False):
    return types.SimpleNamespace(admitted=admitted, reason=reason, is_estop=is_estop, caller="index", model="embed-m")


class _Gov:
    """The governor as the pool and the job ask it: a plan, the callers'
    classes, and an admission that answers from a script, then admits."""

    CLASSES = {"index": "background", "warmup": "background", "chat": "interactive", "direct": "user"}

    def __init__(self, plan=None, answers=()):
        self.plan = plan
        self.answers = list(answers)
        self.asked = []

    def plan_background(self):
        return self.plan

    def caller_class(self, caller):
        return self.CLASSES.get(caller, "user")

    def admit_or_wait(self, model, requested_ctx=None, caller="benchmark", **_kw):
        self.asked.append((model, caller))
        return self.answers.pop(0) if self.answers else _decision()


def _governor_module(gov):
    """A resource_governor stand-in: the governor, a thread-local ticket
    scope, and the final-refusal test the real module exports."""
    mod = types.ModuleType(_RG)
    local = threading.local()

    @contextlib.contextmanager
    def ticket_scope(decision):
        previous = getattr(local, "ticket", None)
        local.ticket = decision
        try:
            yield
        finally:
            local.ticket = previous

    mod.get_resource_governor = lambda: gov
    mod.ticket_scope = ticket_scope
    mod.get_active_ticket = lambda: getattr(local, "ticket", None)
    mod.refusal_is_final = lambda d: (not d.admitted) and (bool(d.is_estop) or d.reason in _FINAL)
    return mod


class _Store:
    """The vector store as the job reads it: its chunker's settings, its
    embedding model, and a storage half that records what it was handed,
    on which thread and under which ticket."""

    embedding_model = "embed-m"

    def __init__(self, rg=None, on_store=None):
        self._rg = rg
        self._on_store = on_store
        self.stored = []

    def chunker_settings(self):
        return (120, 20)

    def store_chunked(self, result, collection=None, metadata=None):
        self.stored.append(
            types.SimpleNamespace(
                name=Path(result.source_file).name,
                ticket=self._rg.get_active_ticket() if self._rg is not None else None,
                thread=threading.get_ident(),
                chunks=len(result.chunks),
                collection=collection,
                metadata=dict(metadata or {}),
            )
        )
        if self._on_store is not None:
            self._on_store()
        return types.SimpleNamespace(doc_id=result.doc_id, chunk_count=len(result.chunks))


class _SpyEvent(threading.Event):
    """A cancel event that records each wait the job makes on it."""

    def __init__(self):
        super().__init__()
        self.waits = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return super().wait(timeout)


_PARAGRAPH = (
    "The background index reads each file once, cuts it where its structure allows, "
    "and hands the pieces to the store, one file after another. "
)


def _texts(names):
    """Text files long enough to chunk, one per name, in a fresh directory."""
    folder = Path(tempfile.mkdtemp(prefix="bw-files-"))
    paths = []
    for n, name in enumerate(names):
        path = folder / name
        path.write_text("\n\n".join(f"File {n}, part {i}. " + _PARAGRAPH * 2 for i in range(6)), encoding="utf-8")
        paths.append(path)
    return paths


def _corpus():
    """One file of each kind the chunker cuts its own way."""
    folder = Path(tempfile.mkdtemp(prefix="bw-corpus-"))
    files = {
        "notes.txt": "\n\n".join(f"Paragraph {i}. " + _PARAGRAPH * 3 for i in range(30)),
        "guide.md": "# Guide\n\n" + "\n\n".join(f"## Section {i}\n\n" + _PARAGRAPH * 4 for i in range(12)),
        "tool.py": "\n\n".join(f"def step_{i}(x):\n    \"\"\"Step {i}.\"\"\"\n    return x + {i}\n" for i in range(40)),
        "table.csv": "name,size,kind\n" + "\n".join(f"item{i},{i * 7},kind{i % 5}" for i in range(300)),
        "data.json": json.dumps({f"key{i}": {"value": i, "text": _PARAGRAPH} for i in range(60)}, indent=2),
    }
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return [folder / name for name in files]


def _job(loaded, files, *, pool, store, chunk_task=None):
    """An indexing job over ``files`` that chunks through ``pool`` and stores
    into ``store``; the engine and the job's id."""
    ingest = loaded[_INGEST]
    engine = ingest.BatchIngestEngine(data_dir=tempfile.mkdtemp(prefix="bw-job-"))
    engine._get_rag_store = lambda: store
    engine._pool = lambda: pool
    if chunk_task is not None:
        engine._chunk_task = chunk_task
    job = engine.create_batch_job([str(f) for f in files], collection="c")
    return ingest, engine, job.job_id


def _files(engine, job_id):
    return {f.filename: f for f in engine.db.get_files_for_job(job_id)}


def _until(predicate, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


# ---------------------------------------------------------------------------
# BW1-BW7 -- the workers
# ---------------------------------------------------------------------------


def test_bw1_every_thread_of_a_worker_is_sched_idle_its_first_and_one_born_later():
    """BW1 -- a worker sets SCHED_IDLE on its threads as it enters, before
    any task, so its first thread is idle and a thread a task starts later
    inherits the policy. Skipping the policy at the entry -> RED."""
    bp = _open()[_POOL]
    pool = _live(bp, _plan(workers=1))
    try:
        report = pool.submit(_bw_tasks.census_after_thread, caller="index").result(timeout=_WAIT_S)
    finally:
        pool.shutdown()
    threads = report["threads"]
    assert report["pid"] != os.getpid()
    assert len(threads) >= 2, "the census sees the worker's first thread and the late one"
    assert report["late_thread"] in {t["tid"] for t in threads}
    assert all(t["policy"] == os.SCHED_IDLE for t in threads), threads
    assert report["entry"]["policy"] == "idle"


def test_bw2_no_thread_of_the_owning_process_is_idle_and_the_entry_refuses_to_run_there():
    """BW2 -- the pool's own process keeps every thread's policy and I/O
    class after the pool served (the census that reads it reads an idle
    worker in the same contract), and the entry, asked to run in the process
    that owns the pool, refuses before any kernel call. An entry that runs
    in the owner -> RED."""
    bp = _open()[_POOL]
    pool = _live(bp, _plan(workers=2))
    try:
        worker = pool.submit(bp.census, caller="index").result(timeout=_WAIT_S)
        for _ in range(3):
            pool.submit(bp.census, caller="index").result(timeout=_WAIT_S)
        own = bp.census()
    finally:
        pool.shutdown()
    assert worker["pid"] != os.getpid()
    assert all(t["policy"] == os.SCHED_IDLE for t in worker["threads"]), "the census reads an idle thread"
    assert own["pid"] == os.getpid()
    assert len(own["threads"]) >= 2, "the pool's own threads were alive at the census"
    assert all(t["policy"] != os.SCHED_IDLE for t in own["threads"]), own["threads"]
    assert all(t["io_class"] != bp.IOPRIO_CLASS_IDLE for t in own["threads"]), own["threads"]
    calls = []
    with pytest.raises(RuntimeError):
        bp._enter_background((0,), os.getpid(), None, _Ops(calls))
    assert calls == []


def test_bw3_every_thread_of_a_worker_is_in_the_idle_io_class():
    """BW3 -- a worker sets the idle I/O class on its threads as it enters,
    and a thread born later inherits it. Skipping the I/O class at the
    entry -> RED."""
    bp = _open()[_POOL]
    if bp.ioprio_calls() is None:
        pytest.skip("no I/O priority call is known for this machine")
    pool = _live(bp, _plan(workers=1))
    try:
        report = pool.submit(_bw_tasks.census_after_thread, caller="index").result(timeout=_WAIT_S)
    finally:
        pool.shutdown()
    threads = report["threads"]
    assert len(threads) >= 2
    assert all(t["io_class"] == bp.IOPRIO_CLASS_IDLE for t in threads), threads
    assert report["entry"]["io"] == "idle"


def test_bw4_every_thread_of_a_worker_runs_on_the_plans_cpus_and_no_other():
    """BW4 -- the CPUs the plan gives are every worker thread's affinity,
    the late thread's too; a plan that names no CPU leaves the affinity the
    worker inherited. Skipping the affinity at the entry -> RED."""
    bp = _open()[_POOL]
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("a single usable CPU leaves no set to narrow to")
    cpus = tuple(allowed[1:])
    pool = _live(bp, _plan(workers=1, cpus=cpus))
    try:
        narrowed = pool.submit(_bw_tasks.census_after_thread, caller="index").result(timeout=_WAIT_S)
    finally:
        pool.shutdown()
    pool = _live(bp, _plan(workers=1, cpus=()))
    try:
        inherited = pool.submit(_bw_tasks.census_after_thread, caller="index").result(timeout=_WAIT_S)
    finally:
        pool.shutdown()
    assert len(narrowed["threads"]) >= 2
    assert all(sorted(t["cpus"]) == list(cpus) for t in narrowed["threads"]), narrowed["threads"]
    assert narrowed["entry"]["cpus"] == "set"
    assert all(sorted(t["cpus"]) == allowed for t in inherited["threads"]), inherited["threads"]
    assert inherited["entry"]["cpus"] == "inherited"


class _Ops:
    """Kernel calls that record, and refuse or lack the steps named."""

    def __init__(self, calls, *, refuse=(), lack=(), unsupported=None, threads=(101, 102)):
        self.calls = calls
        self.refuse = set(refuse)
        self.lack = set(lack)
        self.unsupported = unsupported
        self._threads = list(threads)

    def threads(self):
        return list(self._threads)

    def set_idle(self, tid):
        self._step("set_idle", tid)

    def set_io_idle(self, tid):
        self._step("set_io_idle", tid)

    def set_cpus(self, tid, cpus):
        self._step("set_cpus", tid)

    def _step(self, name, tid):
        self.calls.append((name, tid))
        if name in self.refuse:
            raise PermissionError(errno.EPERM, "refused")
        if name in self.lack:
            raise self.unsupported(f"no call for {name} here")


class _Box:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_bw5_a_refused_step_changes_nothing_the_others_are_tried_and_each_refusal_is_said():
    """BW5 -- the kernel refusing the policy leaves the entry standing: the
    I/O class and the CPUs are still set on every thread, the outcome names
    the refusal by its errno, and the outcome is reported to the owner; a
    machine with no known I/O call says so, and a plan with no CPU makes no
    affinity call. A refusal that stops the entry -> RED."""
    bp = _open()[_POOL]
    calls = []
    box = _Box()
    outcome = bp._enter_background((3, 4), -1, box, _Ops(calls, refuse={"set_idle"}))
    assert outcome["policy"] == "refused: EPERM"
    assert (outcome["io"], outcome["cpus"]) == ("idle", "set")
    assert [c for c in calls if c[0] == "set_io_idle"] == [("set_io_idle", 101), ("set_io_idle", 102)]
    assert [c for c in calls if c[0] == "set_cpus"] == [("set_cpus", 101), ("set_cpus", 102)]
    assert box.items == [outcome]
    lacking = []
    outcome = bp._enter_background((), -1, None, _Ops(lacking, lack={"set_io_idle"}, unsupported=bp.Unsupported))
    assert outcome["policy"] == "idle"
    assert outcome["io"].startswith("unsupported")
    assert outcome["cpus"] == "inherited"
    assert not any(name == "set_cpus" for name, _tid in lacking)


def test_bw6_a_file_chunked_in_a_worker_gives_what_the_chunker_gives_on_the_jobs_thread():
    """BW6 -- for each kind of file, the worker's result equals the
    chunker's at the same settings, chunk ids included, and the settings
    the job passes are the ones the worker cuts at. A worker that cuts at
    its own defaults -> RED."""
    loaded = _open()
    bp, chunker = loaded[_POOL], loaded[_CHUNKER]
    files = _corpus()
    pool = _live(bp, _plan(workers=2))
    try:
        for path in files:
            here = chunker.RAGChunker(chunk_size=120, chunk_overlap=20).chunk_file(path, doc_id="d1")
            there = pool.submit(chunker.chunk_file_task, str(path), "d1", 120, 20, caller="index").result(
                timeout=_WAIT_S
            )
            assert dataclasses.asdict(there) == dataclasses.asdict(here), path.name
            assert [c.chunk_id for c in there.chunks] == [c.chunk_id for c in here.chunks]
        coarse = pool.submit(chunker.chunk_file_task, str(files[0]), "d1", 500, 50, caller="index").result(
            timeout=_WAIT_S
        )
        fine = pool.submit(chunker.chunk_file_task, str(files[0]), "d1", 120, 20, caller="index").result(
            timeout=_WAIT_S
        )
    finally:
        pool.shutdown()
    assert len(fine.chunks) > len(coarse.chunks) > 1


def test_bw7_a_worker_holds_neither_the_vector_store_nor_any_project_module_past_the_pool_and_the_chunker():
    """BW7 -- after chunking as the job does, a worker's project modules are
    the package, the pool and the chunker, and no vector database is
    loaded. A chunk task that lives in the store -> RED."""
    bp = _open()[_POOL]
    path = _texts(["lean.txt"])[0]
    pool = _live(bp, _plan(workers=1))
    try:
        modules = pool.submit(_bw_tasks.chunk_then_modules, str(path), "d1", 500, 50, caller="index").result(
            timeout=_WAIT_S
        )
    finally:
        pool.shutdown()
    project = {m for m in modules if m == "opti_oignon" or m.startswith("opti_oignon.")}
    assert {"opti_oignon.background_pool", "opti_oignon.rag_chunker"} <= project, "the census reads the worker"
    assert project <= {
        "opti_oignon",
        "opti_oignon.__version__",
        "opti_oignon.background_pool",
        "opti_oignon.rag_chunker",
    }, sorted(project)
    assert not any(m == "chromadb" or m.startswith("chromadb.") for m in modules)


# ---------------------------------------------------------------------------
# BW8-BW11 -- the job
# ---------------------------------------------------------------------------


def test_bw8_a_worker_dying_on_a_file_fails_that_file_alone_and_the_job_completes():
    """BW8 -- a worker killed while chunking breaks every file in flight
    with it: each is chunked again alone, the one that kills its worker
    alone fails with its name, every other file is stored, the job
    completes, and the pool serves the next task. Failing every file in
    flight with the dead worker -> RED."""
    gov = _Gov(plan=_plan(workers=2, in_flight=2))
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    bp = loaded[_POOL]
    pool = bp.BackgroundPool(plan=lambda: gov.plan, class_of=gov.caller_class)
    store = _Store(rg)
    files = _texts(["a.txt", "b.txt", "die.txt", "c.txt", "d.txt"])
    try:
        ingest, engine, job_id = _job(
            loaded, files, pool=pool, store=store, chunk_task=functools.partial(_bw_tasks.chunk_or_die, "die.txt")
        )
        engine._worker_loop(job_id, threading.Event())
        after = pool.submit(bp.census, caller="index").result(timeout=_WAIT_S)
    finally:
        pool.shutdown()
    by_name = _files(engine, job_id)
    assert engine.get_job(job_id).status == ingest.JobStatus.COMPLETED.value
    assert by_name["die.txt"].status == ingest.FileStatus.ERROR.value
    assert "die.txt" in by_name["die.txt"].error_message and "died" in by_name["die.txt"].error_message
    assert sorted(s.name for s in store.stored) == ["a.txt", "b.txt", "c.txt", "d.txt"]
    assert all(by_name[n].status == ingest.FileStatus.DONE.value for n in ("a.txt", "b.txt", "c.txt", "d.txt"))
    assert after["pid"] != os.getpid()


def test_bw9_each_file_is_stored_on_the_jobs_thread_under_the_index_ticket_for_the_embedding_model():
    """BW9 -- the job asks the governor, through the queue that lets a
    background caller wait, for the store's embedding model as the "index"
    caller, once per file, and stores each file on its own thread under the
    ticket that admission returned. Storing outside the ticket's scope -> RED."""
    decisions = [_decision(), _decision(), _decision()]
    gov = _Gov(plan=_plan(workers=0, source="disabled"), answers=decisions)
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    pool = loaded[_POOL].BackgroundPool()
    store = _Store(rg)
    ingest, engine, job_id = _job(loaded, _texts(["a.txt", "b.txt", "c.txt"]), pool=pool, store=store)
    engine._worker_loop(job_id, threading.Event())
    pool.shutdown()
    assert gov.asked == [("embed-m", "index")] * 3
    assert [s.ticket for s in store.stored] == decisions
    assert all(a is b for a, b in zip((s.ticket for s in store.stored), decisions))
    assert all(s.thread == threading.get_ident() for s in store.stored)
    assert rg.get_active_ticket() is None
    assert all(s.metadata["batch_job_id"] == job_id and s.collection == "c" for s in store.stored)


def test_bw10_a_refusal_a_wait_can_lift_is_asked_again_and_one_no_wait_can_lift_fails_the_file():
    """BW10 -- two refusals a wait can lift, then an admission: the file is
    stored under that admission after two pauses of the plan's length; a
    refusal no wait can lift fails the next file with its reason, and
    nothing is stored for it. Failing a file on the first refusal -> RED."""
    held = _decision(False, "background_gate_closed")
    admitted = _decision()
    final = _decision(False, "background_cost_unknown")
    gov = _Gov(plan=_plan(workers=0, source="disabled", held_retry_s=0.25), answers=[held, held, admitted, final])
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    pool = loaded[_POOL].BackgroundPool()
    store = _Store(rg)
    ingest, engine, job_id = _job(loaded, _texts(["a.txt", "b.txt"]), pool=pool, store=store)
    cancel = _SpyEvent()
    engine._worker_loop(job_id, cancel)
    pool.shutdown()
    by_name = _files(engine, job_id)
    assert len(gov.asked) == 4
    assert [s.name for s in store.stored] == ["a.txt"]
    assert store.stored[0].ticket is admitted
    assert cancel.waits == [0.25, 0.25]
    assert by_name["a.txt"].status == ingest.FileStatus.DONE.value
    assert by_name["b.txt"].status == ingest.FileStatus.ERROR.value
    assert "background_cost_unknown" in by_name["b.txt"].error_message
    assert engine.get_job(job_id).status == ingest.JobStatus.COMPLETED.value


class _StallingPool:
    """A pool whose first task runs at once and whose others never start."""

    def __init__(self):
        self.futures = []

    def in_flight_limit(self):
        return 3

    def submit(self, fn, /, *args, caller):
        future = Future()
        if not self.futures:
            future.set_result(fn(*args))
        self.futures.append(future)
        return future

    def reset(self):
        pass


def test_bw11_a_cancelled_job_gives_its_files_in_flight_back_and_does_not_wait_for_them():
    """BW11 -- cancelled while two files are still being chunked, the job
    cancels their tasks, gives both back to the queue, leaves the file it
    never sent queued, and ends cancelled without waiting for a chunking
    that never finishes. Leaving the files in flight marked processing ->
    RED."""
    gov = _Gov(plan=_plan(workers=0, source="disabled"))
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    pool = _StallingPool()
    cancel = threading.Event()
    store = _Store(rg, on_store=cancel.set)
    ingest, engine, job_id = _job(loaded, _texts(["a.txt", "b.txt", "c.txt", "d.txt"]), pool=pool, store=store)
    runner = threading.Thread(target=engine._worker_loop, args=(job_id, cancel), daemon=True)
    runner.start()
    runner.join(_WAIT_S)
    assert not runner.is_alive(), "the job does not wait for a chunking that never finishes"
    by_name = _files(engine, job_id)
    assert [s.name for s in store.stored] == ["a.txt"]
    assert len(pool.futures) == 3
    assert all(f.cancelled() for f in pool.futures[1:])
    assert by_name["a.txt"].status == ingest.FileStatus.DONE.value
    assert [by_name[n].status for n in ("b.txt", "c.txt", "d.txt")] == [ingest.FileStatus.QUEUED.value] * 3
    assert engine.get_job(job_id).status == ingest.JobStatus.CANCELLED.value


# ---------------------------------------------------------------------------
# BW12-BW14 -- the pool
# ---------------------------------------------------------------------------


class _Executor:
    """An executor that runs each task at once, or fails to submit."""

    def __init__(self, fail=None):
        self.sent = []
        self.fail = fail

    def submit(self, fn, /, *args, **kwargs):
        if self.fail is not None:
            raise self.fail
        self.sent.append(fn)
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass


def test_bw12_a_caller_of_the_interactive_or_user_class_is_refused_and_a_background_one_served():
    """BW12 -- the pool asks the class of each caller: an interactive or a
    user caller is refused with nothing sent to a worker, a background
    caller is served. Serving any caller -> RED."""
    bp = _open()[_POOL]
    made = []

    def factory(workers, initializer, initargs):
        made.append(_Executor())
        return made[-1]

    classes = {"chat": "interactive", "direct": "user", "index": "background"}
    pool = bp.BackgroundPool(plan=lambda: _plan(workers=1), class_of=classes.get, executor_factory=factory)
    for caller in ("chat", "direct", "nobody"):
        with pytest.raises(PermissionError):
            pool.submit(len, "abc", caller=caller)
    assert all(executor.sent == [] for executor in made)
    assert pool.submit(len, "abc", caller="index").result(timeout=_WAIT_S) == 3
    assert [executor.sent for executor in made] == [[len]]
    pool.shutdown()


def test_bw13_no_worker_before_the_first_task_none_after_the_idle_delay_and_none_at_shutdown():
    """BW13 -- the pool starts its workers with its first task, closes them
    once the plan's idle delay passes without a task, starts afresh for the
    next, and leaves no worker process after shutdown. A pool that keeps
    its workers when idle -> RED."""
    bp = _open()[_POOL]
    pool = _live(bp, _plan(workers=1, idle_s=0.3))
    try:
        assert pool.worker_pids() == ()
        assert pool.status()["mode"] == "idle"
        first = pool.submit(bp.census, caller="index").result(timeout=_WAIT_S)
        assert first["pid"] in pool.worker_pids()
        assert pool.status()["mode"] == "process"
        assert _until(lambda: pool.worker_pids() == (), 20.0), "the workers close after the idle delay"
        assert not Path(f"/proc/{first['pid']}").exists()
        assert pool.status()["mode"] == "idle"
        second = pool.submit(bp.census, caller="index").result(timeout=_WAIT_S)
        assert second["pid"] != first["pid"]
        pool.shutdown()
        assert pool.worker_pids() == ()
        assert not Path(f"/proc/{second['pid']}").exists()
        with pytest.raises(RuntimeError):
            pool.submit(bp.census, caller="index")
    finally:
        pool.shutdown()


def test_bw14_a_pool_that_cannot_start_says_off_or_has_no_governor_runs_inline_and_says_why():
    """BW14 -- a factory that raises, an executor that cannot submit, a plan
    that says off, and a pool with no governor to plan each run the task on
    the caller's thread, return its result, and say "inline" with the
    reason. A pool that drops the task -> RED."""
    bp = _open()[_POOL]

    def refused(workers, initializer, initargs):
        raise OSError(errno.EPERM, "spawn refused")

    background = {"class_of": lambda caller: "background"}
    cases = [
        (bp.BackgroundPool(plan=lambda: _plan(workers=1), executor_factory=refused, **background), "spawn refused"),
        (
            bp.BackgroundPool(
                plan=lambda: _plan(workers=1),
                executor_factory=lambda w, i, a: _Executor(fail=OSError("no process")),
                **background,
            ),
            "no process",
        ),
        (bp.BackgroundPool(plan=lambda: _plan(workers=0, source="disabled"), **background), "disabled"),
        (bp.BackgroundPool(), "governor"),
    ]
    here = threading.get_ident()
    for pool, why in cases:
        assert pool.submit(threading.get_ident, caller="index").result(timeout=_WAIT_S) == here
        status = pool.status()
        assert status["mode"] == "inline", status
        assert why in status["reason"], status
        assert pool.worker_pids() == ()
        pool.shutdown()


# ---------------------------------------------------------------------------
# BW15-BW18 -- the status
# ---------------------------------------------------------------------------

_ROUTES_RAG = "opti_oignon.api.routes_rag"


def _tmpdir(prefix):
    """A temporary directory the contract's teardown removes."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    _CLOSERS.append(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


def _open_job_routes(governor):
    """The pool, the chunker, the indexing job and the RAG routes in one
    window; the governor is the stand-in module given."""
    loaded, restore = isolate(
        targets={
            _POOL: source("background_pool.py"),
            _CHUNKER: source("rag_chunker.py"),
            _INGEST: source("rag", "batch_ingest.py"),
            _ROUTES_RAG: source("api", "routes_rag.py"),
        },
        blocked=("opti_oignon.rag_store", "opti_oignon.sandbox_manager"),
        seeded={"opti_oignon.db_utils": _db_utils(), _RG: governor},
        packages=("opti_oignon.rag", "opti_oignon.api"),
    )
    _CLOSERS.append(restore)
    return loaded


def _disk_seams(tmp):
    """A /sys, a mount table and a cgroup tree of our own, as the disk
    reader takes them: an NVMe disk under none with a partition; a SATA disk
    under bfq whose partition a dm-crypt mapping sits on; a disk under
    mq-deadline with its aging, one without it, one under kyber, one under
    a scheduler of another name; an md array over the aged disk and the
    NVMe partition. Two btrfs mounts with no device number of their own
    name the partition and the mapping; tmpfs and a network share name no
    block device. The process sits in user.slice/oo.service, and no cgroup
    names an io.prio.class."""
    sys_root = tmp / "sys"

    def disk(where, scheduler, aging=None):
        queue = sys_root / "devices" / where / "queue"
        queue.mkdir(parents=True)
        (queue / "scheduler").write_text(scheduler + "\n", encoding="utf-8")
        if aging is not None:
            (queue / "iosched").mkdir()
            (queue / "iosched" / "prio_aging_expire").write_text(f"{aging}\n", encoding="utf-8")
        return queue.parent

    def partition(whole, name, number):
        part = whole / name
        part.mkdir()
        (part / "partition").write_text(f"{number}\n", encoding="utf-8")
        return part

    def link(at, target):
        at.parent.mkdir(parents=True, exist_ok=True)
        at.symlink_to(os.path.relpath(target, at.parent))

    nvme = disk("pci0/nvme/block/nvme0n1", "[none] mq-deadline kyber bfq")
    nvme_p2 = partition(nvme, "nvme0n1p2", 2)
    sda = disk("pci0/ata1/block/sda", "mq-deadline kyber [bfq] none")
    sda1 = partition(sda, "sda1", 1)
    sdb = disk("pci0/ata2/block/sdb", "[mq-deadline] kyber bfq none", aging=10000)
    sdc = disk("pci0/ata3/block/sdc", "mq-deadline [kyber] bfq none")
    sdd = disk("pci0/ata4/block/sdd", "[mq-deadline] none")
    vda = disk("pci0/virtio/block/vda", "[elevator-x] none")
    dm = disk("virtual/block/dm-0", "none")
    (dm / "dm").mkdir()
    (dm / "dm" / "name").write_text("cryptroot\n", encoding="utf-8")
    link(dm / "slaves" / "sda1", sda1)
    md = disk("virtual/block/md0", "none")
    link(md / "slaves" / "sdb", sdb)
    link(md / "slaves" / "nvme0n1p2", nvme_p2)
    numbers = {"259:2": nvme_p2, "8:1": sda1, "8:16": sdb, "8:32": sdc, "8:48": sdd, "252:0": vda, "253:0": dm, "9:0": md}
    for number, target in numbers.items():
        link(sys_root / "dev" / "block" / number, target)
    for target in (nvme, nvme_p2, sda, sda1, sdb, sdc, sdd, vda, dm, md):
        link(sys_root / "class" / "block" / target.name, target)
    mountinfo = tmp / "mountinfo"
    mountinfo.write_text(
        "22 1 0:45 / /home rw,relatime shared:1 - btrfs /dev/nvme0n1p2 rw,ssd\n"
        "23 1 0:46 / /vault rw,relatime shared:2 master:7 - btrfs /dev/mapper/cryptroot rw\n"
        "24 1 0:47 / /tmp rw,nosuid shared:3 - tmpfs tmpfs rw\n"
        "25 1 0:48 / /net rw - nfs4 server:/export rw\n",
        encoding="utf-8",
    )
    cgroup_root = tmp / "cg"
    (cgroup_root / "user.slice" / "oo.service").mkdir(parents=True)
    self_cgroup = tmp / "self-cgroup"
    self_cgroup.write_text("0::/user.slice/oo.service\n", encoding="utf-8")
    return {
        "sys_root": str(sys_root),
        "mountinfo": str(mountinfo),
        "cgroup_root": str(cgroup_root),
        "self_cgroup": str(self_cgroup),
    }


def test_bw15_the_state_is_unused_until_the_server_pool_exists_then_its_own_status_and_reading_it_creates_none():
    """BW15 -- before any background work asked for the server's pool, its
    state says "unused", with no plan, nothing pending and no worker, and
    reading it creates no pool; once the pool exists, the state is the
    pool's own status: its mode, its plan with the CPUs it leaves to the
    user, and what each live worker's entry set. A state that creates the
    pool to read it -> RED."""
    bp = _open()[_POOL]
    assert bp._POOL is None
    unused = bp.background_state()
    assert (unused["mode"], unused["plan"], unused["pending"], unused["workers"]) == ("unused", None, 0, [])
    assert unused["reason"]
    assert bp._POOL is None
    pool = _live(bp, _plan(workers=1, reserved=(0, 12)))
    bp._POOL = pool
    try:
        assert bp.background_state() == pool.status()
        pid = pool.submit(bp.census, caller="index").result(timeout=_WAIT_S)["pid"]
        state = bp.background_state()
        assert (state["mode"], state["plan"]["reserved"], state["plan"]["workers"]) == ("process", [0, 12], 1)
        assert [(w["pid"], w["policy"], w["io"]) for w in state["workers"]] == [(pid, "idle", "idle")]
    finally:
        bp._POOL = None
        pool.shutdown()


def test_bw16_the_disks_behind_a_device_are_read_through_partitions_mappings_arrays_and_mounts_never_assumed():
    """BW16 -- a partition is read on its disk; a device-mapper mapping or
    an md array on the disks under it, each once, by name; a file system
    with no device number of its own on the device its mount names, by its
    node or by its mapping's name; each disk with the scheduler its queue
    has selected. A mount with no block device (tmpfs, a network share), a
    device no mount names, or one the kernel does not list gives no disk
    and says why. Reading the mapping's own queue in place of the disks
    under it -> RED."""
    bp = _open()[_POOL]
    paths = _disk_seams(_tmpdir("bw16-"))

    def disks(major, minor):
        found = bp.device_disks(os.makedev(major, minor), **paths)
        return [(d["name"], d["scheduler"]) for d in found["disks"]], found

    assert disks(259, 2)[0] == [("nvme0n1", "none")]
    assert disks(259, 2)[1]["device"] == "259:2"
    assert disks(8, 1)[0] == [("sda", "bfq")]
    assert disks(253, 0)[0] == [("sda", "bfq")]
    assert disks(9, 0)[0] == [("nvme0n1", "none"), ("sdb", "mq-deadline")]
    assert disks(0, 45)[0] == [("nvme0n1", "none")]
    assert disks(0, 46)[0] == [("sda", "bfq")]
    for (major, minor), why in (((0, 47), "tmpfs"), ((0, 48), "nfs4"), ((0, 99), "0:99"), ((8, 99), "8:99")):
        named, found = disks(major, minor)
        assert (named, found["idle_class"]) == ([], "unknown"), (major, minor)
        assert why in found["reason"], found


def test_bw17_what_each_disk_does_with_the_idle_class_is_its_schedulers_and_a_cgroup_promoting_it_is_said():
    """BW17 -- the idle I/O class is honoured under bfq, deferred under
    mq-deadline up to the aging its queue names, and has no effect under
    kyber or none; a scheduler of another name is unknown, and disks that
    disagree are "mixed". The io.prio.class of the nearest cgroup that
    names one is read: promote-to-rt (or its older name none-to-rt) turns
    the workers' idle requests into real-time ones on a disk whose
    scheduler orders classes, said "promoted_to_rt"; restrict-to-be, idle
    and no-change leave the idle class as it is; a policy of another name
    is unknown. Ignoring a cgroup that promotes the class -> RED."""
    bp = _open()[_POOL]
    paths = _disk_seams(_tmpdir("bw17-"))

    def seen(major, minor):
        return bp.device_disks(os.makedev(major, minor), **paths)

    effects = {name: seen(*number) for name, number in (
        ("nvme", (259, 2)), ("sda", (8, 1)), ("sdb", (8, 16)), ("sdc", (8, 32)), ("sdd", (8, 48)), ("vda", (252, 0)),
    )}
    assert {name: found["idle_class"] for name, found in effects.items()} == {
        "nvme": "no_effect", "sda": "honored", "sdb": "deferred", "sdc": "no_effect", "sdd": "deferred", "vda": "unknown",
    }
    assert [found["disks"][0]["aging_ms"] for found in (effects["sdb"], effects["sdd"], effects["sda"])] == [10000, None, None]
    assert effects["sda"]["policy"] is None
    array = seen(9, 0)
    assert (array["idle_class"], [d["idle_class"] for d in array["disks"]]) == ("mixed", ["no_effect", "deferred"])
    slice_file = Path(paths["cgroup_root"]) / "user.slice" / "io.prio.class"
    for policy, bfq, none in (
        ("promote-to-rt", "promoted_to_rt", "no_effect"),
        ("none-to-rt", "promoted_to_rt", "no_effect"),
        ("restrict-to-be", "honored", "no_effect"),
        ("idle", "honored", "no_effect"),
        ("no-change", "honored", "no_effect"),
        ("lift-all", "unknown", "unknown"),
    ):
        slice_file.write_text(policy + "\n", encoding="utf-8")
        assert (seen(8, 1)["policy"], seen(8, 1)["idle_class"], seen(259, 2)["idle_class"]) == (policy, bfq, none), policy
    slice_file.write_text("promote-to-rt\n", encoding="utf-8")
    (Path(paths["cgroup_root"]) / "user.slice" / "oo.service" / "io.prio.class").write_text(
        "restrict-to-be\n", encoding="utf-8"
    )
    assert (seen(8, 1)["policy"], seen(8, 1)["idle_class"]) == ("restrict-to-be", "honored")


def test_bw18_the_job_names_the_disks_of_the_files_it_sends_the_background_each_device_once_through_its_route():
    """BW18 -- the indexing job notes the device of each file it sends the
    pool, and its route names the disks behind each device once, in device
    order, as the disk reader gives them; a job that has sent nothing names
    none. Naming the disk of the job's first file alone -> RED."""
    gov = _Gov(plan=_plan(workers=0, source="disabled"))
    rg = _governor_module(gov)
    loaded = _open_job_routes(rg)
    ingest, routes = loaded[_INGEST], loaded[_ROUTES_RAG]
    tmp = _tmpdir("bw18-")
    files = []
    for name in ("a.txt", "b.txt", "c.txt"):
        path = tmp / name
        path.write_text("\n\n".join(f"{name}, part {i}. " + _PARAGRAPH * 2 for i in range(4)), encoding="utf-8")
        files.append(path)
    engine = ingest.BatchIngestEngine(data_dir=tmp / "jobs")
    pool = loaded[_POOL].BackgroundPool()
    engine._get_rag_store = lambda: _Store(rg)
    engine._pool = lambda: pool
    job_id = engine.create_batch_job([str(f) for f in files], collection="c").job_id
    devices = {str(files[0]): os.makedev(259, 2), str(files[1]): os.makedev(8, 1), str(files[2]): os.makedev(259, 2)}
    engine._device_of = lambda path: devices[str(path)]
    engine._disks_of = lambda device, path=None: {
        "device": f"{os.major(device)}:{os.minor(device)}", "disks": [], "policy": None, "idle_class": "no_effect",
        "reason": None,
    }
    assert engine.job_disks(job_id) == []
    engine._worker_loop(job_id, threading.Event())
    pool.shutdown()
    named = engine.job_disks(job_id)
    assert [d["device"] for d in named] == ["8:1", "259:2"]
    assert all(d["idle_class"] == "no_effect" for d in named)
    routes._get_batch_engine = lambda: engine
    assert routes.get_ingest_job(job_id).disks == named


# ---------------------------------------------------------------------------
# BW19 -- the workers, continued
# ---------------------------------------------------------------------------


def test_bw19_a_32_bit_interpreter_on_a_64_bit_kernel_says_unsupported_and_makes_no_io_priority_call(monkeypatch):
    """BW19 -- the I/O priority numbers are those of the 64-bit ABI, so they
    stand for a 64-bit interpreter only: with the kernel's machine "x86_64"
    and 4-byte pointers, the entry's I/O step says "unsupported", naming the
    machine and the interpreter's width, and neither the entry nor the
    census makes a system call; with 8-byte pointers the x86-64 numbers
    stand. Choosing the numbers by the kernel's machine alone -> RED."""
    bp = _open()[_POOL]
    made = []

    def no_syscall(*args):
        made.append(args)
        raise AssertionError(f"a system call was made with {args}")

    class _IoStep(_Ops):
        """The recorded steps, the I/O one through the module's own call."""

        def set_io_idle(self, tid):
            self.calls.append(("set_io_idle", tid))
            bp._KERNEL.set_io_idle(tid)

    monkeypatch.setattr(bp, "platform", types.SimpleNamespace(machine=lambda: "x86_64"))
    monkeypatch.setattr(bp, "_syscall", no_syscall)
    monkeypatch.setattr(bp, "_POINTER_SIZE", 4, raising=False)
    calls = []
    outcome = bp._enter_background((), -1, None, _IoStep(calls))
    report = bp.census()
    assert made == []
    assert [c for c in calls if c[0] == "set_io_idle"] == [("set_io_idle", 101), ("set_io_idle", 102)]
    assert outcome["io"].startswith("unsupported"), outcome
    assert "x86_64" in outcome["io"] and "32-bit" in outcome["io"], outcome
    assert bp.ioprio_calls() is None
    assert report["threads"] and all(t["io_class"] is None for t in report["threads"]), report["threads"]
    monkeypatch.setattr(bp, "_POINTER_SIZE", 8)
    assert bp.ioprio_calls() == (251, 252)


# ---------------------------------------------------------------------------
# BW20-BW21 -- the status, continued
# ---------------------------------------------------------------------------


def test_bw20_a_file_whose_device_no_mount_shows_is_read_on_the_mount_that_holds_its_path():
    """BW20 -- a btrfs subvolume's files carry a device number of their own
    that no mountinfo line shows (a line carries the superblock's): such a
    file is read on the mount whose mount point is the longest that holds
    its real path by whole components, the mount point's octal escapes
    undone and a later line winning on the same point, and that mount's
    source names the partition and its disk; a path no listed mount point
    holds gives no disk and says why. Reading the device number alone ->
    RED."""
    bp = _open()[_POOL]
    tmp = _tmpdir("bw20-")
    paths = _disk_seams(tmp)
    base = os.path.realpath(tmp)
    table = tmp / "mountinfo-subvolumes"
    table.write_text(
        f"30 1 0:31 / {base} rw - ext4 /dev/sdc rw\n"
        f"31 30 0:45 /@home {base}/my\\040home rw,relatime shared:1 - btrfs /dev/sdd rw\n"
        f"32 30 0:45 /@home {base}/my\\040home rw,relatime shared:1 - btrfs /dev/nvme0n1p2 rw,ssd\n"
        f"33 32 0:47 / {base}/my\\040home/no rw,nosuid - tmpfs tmpfs rw\n",
        encoding="utf-8",
    )
    paths["mountinfo"] = str(table)
    held = os.path.join(base, "my home", "notes", "a.txt")
    found = bp.device_disks(os.makedev(0, 61), path=held, **paths)
    assert [(d["name"], d["scheduler"]) for d in found["disks"]] == [("nvme0n1", "none")], found
    assert (found["device"], found["idle_class"], found["reason"]) == ("0:61", "no_effect", None)
    outside = bp.device_disks(os.makedev(0, 61), path="/nowhere/a.txt", **paths)
    assert (outside["disks"], outside["idle_class"]) == ([], "unknown")
    assert "0:61" in outside["reason"] and "/nowhere/a.txt" in outside["reason"], outside


def test_bw21_the_job_reads_each_devices_disks_with_the_path_of_the_first_file_it_sent_from_it():
    """BW21 -- the job keeps, for each device it sent the background a file
    from, the path of the first such file, and the disks of the device are
    read with that path, so a file system whose files carry a device number
    no mount shows is found by where the file lies. Reading the disks by the
    device number alone -> RED."""
    gov = _Gov(plan=_plan(workers=0, source="disabled"))
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    pool = loaded[_POOL].BackgroundPool()
    files = [str(path.resolve()) for path in _texts(["a.txt", "b.txt", "c.txt"])]
    ingest, engine, job_id = _job(loaded, files, pool=pool, store=_Store(rg))
    devices = {files[0]: os.makedev(0, 61), files[1]: os.makedev(0, 62), files[2]: os.makedev(0, 61)}
    engine._device_of = lambda path: devices[str(path)]
    asked = []

    def disks_of(device, path=None):
        asked.append((device, path))
        return {"device": f"0:{os.minor(device)}", "disks": [], "policy": None, "idle_class": "unknown", "reason": "x"}

    engine._disks_of = disks_of
    engine._worker_loop(job_id, threading.Event())
    pool.shutdown()
    named = engine.job_disks(job_id)
    assert [d["device"] for d in named] == ["0:61", "0:62"]
    assert asked == [(os.makedev(0, 61), files[0]), (os.makedev(0, 62), files[1])]


# ---------------------------------------------------------------------------
# BW22-BW24 -- the pool, continued
# ---------------------------------------------------------------------------

# A process that queues eight two-second tasks on two real workers, waits
# until a worker has entered and a task runs, says when it returns from
# main, and returns.
_EXIT_CHILD = """
import time
import types

from opti_oignon import background_pool

plan = types.SimpleNamespace(
    workers=2, cpus=(), reserved=(), in_flight=4, idle_s=60.0, held_retry_s=0.01, source="plan"
)
pool = background_pool.BackgroundPool(plan=lambda: plan, class_of=lambda caller: "background")
futures = [pool.submit(time.sleep, 2.0, caller="index") for _ in range(8)]
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline:
    if pool.status()["workers"] and any(f.running() for f in futures):
        break
    time.sleep(0.02)
print("running", sum(f.running() for f in futures), repr(time.monotonic()), flush=True)
"""


def test_bw22_a_process_that_returns_from_main_with_tasks_queued_exits_without_running_them():
    """BW22 -- the pool closes itself as the interpreter exits, before the
    standard library waits for each executor's tasks: a process (the real
    module, imported from this tree) that queued eight two-second tasks on
    two workers and returns from main once one runs exits within four
    seconds of returning, with status 0, where running the queued tasks
    would hold it about eight. A pool closed by an atexit hook, which runs
    after the standard library's wait -> RED."""
    tmp = _tmpdir("bw22-")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (env.get("PYTHONPATH"), str(REPO)) if p)
    out_path, err_path = tmp / "out.txt", tmp / "err.txt"
    with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
        child = subprocess.Popen([sys.executable, "-c", _EXIT_CHILD], cwd=str(tmp), env=env, stdout=out, stderr=err)
        try:
            child.wait(timeout=_WAIT_S)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        ended = time.monotonic()
    said = out_path.read_text(encoding="utf-8").split()
    assert said[:1] == ["running"] and int(said[1]) >= 1, err_path.read_text(encoding="utf-8")[-2000:]
    assert child.returncode == 0, err_path.read_text(encoding="utf-8")[-2000:]
    assert ended - float(said[2]) < 4.0, f"the process lived {ended - float(said[2]):.1f} s after main returned"


class _Held:
    """An executor that keeps each task pending (cancelled by a shutdown
    that cancels) or runs it at once, may refuse its n-th task as broken,
    and records its shutdowns."""

    def __init__(self, *, run=False, breaks_at=None):
        self.run = run
        self.breaks_at = breaks_at
        self.futures = []
        self.shutdowns = []

    def submit(self, fn, /, *args, **kwargs):
        if self.breaks_at is not None and len(self.futures) + 1 == self.breaks_at:
            raise BrokenProcessPool("a worker of this executor died")
        future = Future()
        if self.run:
            future.set_result(fn(*args, **kwargs))
        self.futures.append(future)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdowns.append((wait, cancel_futures))
        if cancel_futures:
            for future in self.futures:
                future.cancel()


def test_bw23_a_broken_task_retires_the_executor_it_came_from_while_current_and_no_other():
    """BW23 -- told which task broke, the pool retires that task's executor
    only while it is the current one: a task of an executor the pool has
    already replaced leaves the current executor serving, its tasks
    untouched, while a broken task of the current executor retires it and
    the next task starts a fresh one. A reset that retires whatever executor
    is current -> RED."""
    bp = _open()[_POOL]
    made = []

    def factory(workers, initializer, initargs):
        made.append(_Held(breaks_at=None if made else 2))
        return made[-1]

    pool = bp.BackgroundPool(
        plan=lambda: _plan(workers=2), class_of=lambda caller: "background", executor_factory=factory
    )
    try:
        first = pool.submit(len, "a", caller="index")
        second = pool.submit(len, "b", caller="index")
        assert len(made) == 2 and made[0].shutdowns, "the first executor broke on its second task and was replaced"
        pool.reset(first)
        third = pool.submit(len, "c", caller="index")
        assert len(made) == 2 and made[1].futures == [second, third]
        assert made[1].shutdowns == [] and not second.cancelled()
        pool.reset(third)
        assert made[1].shutdowns and second.cancelled()
        pool.submit(len, "d", caller="index")
        assert len(made) == 3
    finally:
        pool.shutdown()


def test_bw24_a_task_sent_alone_runs_in_a_one_worker_executor_of_its_own_shut_down_once_done():
    """BW24 -- a task sent alone runs in an executor of its own: the factory
    is asked for one worker, entering on the plan's CPUs, the shared
    executor is not used, and that executor is shut down once the task is
    done while the shared one serves on. A task sent alone through the
    shared executor -> RED."""
    bp = _open()[_POOL]
    made = []

    def factory(workers, initializer, initargs):
        made.append((workers, initializer, initargs[0], _Held(run=True)))
        return made[-1][3]

    pool = bp.BackgroundPool(
        plan=lambda: _plan(workers=2, cpus=(3, 5)), class_of=lambda caller: "background", executor_factory=factory
    )
    try:
        assert pool.submit(len, "ab", caller="index").result(timeout=_WAIT_S) == 2
        assert pool.submit_alone(len, "abc", caller="index").result(timeout=_WAIT_S) == 3
        assert [(w, i, c) for w, i, c, _e in made] == [(2, bp._enter_background, (3, 5)), (1, bp._enter_background, (3, 5))]
        shared, alone = made[0][3], made[1][3]
        assert len(shared.futures) == 1 and len(alone.futures) == 1
        assert _until(lambda: alone.shutdowns != [], 5.0), "the executor of its own is shut down once the task is done"
        assert shared.shutdowns == []
    finally:
        pool.shutdown()


# ---------------------------------------------------------------------------
# BW25-BW30 -- the job, continued
# ---------------------------------------------------------------------------


class _ScriptedPool:
    """A pool that runs each task on the sending thread, except the tasks
    its script answers otherwise, and records each send: the entry used,
    the file, and the files the store holds by then."""

    def __init__(self, store, *, limit=3, answer=None):
        self.store = store
        self.limit = limit
        self.answer = answer or (lambda entry, name, count: None)
        self.calls = []
        self.resets = []

    def in_flight_limit(self):
        return self.limit

    def submit(self, fn, /, *args, caller):
        return self._send("submit", fn, args)

    def submit_alone(self, fn, /, *args, caller):
        return self._send("submit_alone", fn, args)

    def reset(self, broken=None):
        self.resets.append(broken)

    def _send(self, entry, fn, args):
        name = os.path.basename(args[0])
        self.calls.append((entry, name, sorted(s.name for s in self.store.stored)))
        future = Future()
        answer = self.answer(entry, name, sum(1 for call in self.calls if call[1] == name))
        if answer == "cancelled":
            future.cancel()
        elif answer == "broken":
            future.set_exception(BrokenProcessPool("a worker of the executor died"))
        else:
            future.set_result(fn(*args))
        return future


def test_bw25_a_file_whose_task_comes_back_cancelled_is_sent_again_and_stored_never_failed():
    """BW25 -- a task that comes back cancelled (its executor retired under
    it) says nothing of its file: the file is sent again and stored, the
    files after it too, and the job completes. Failing a file whose task was
    cancelled -> RED."""
    gov = _Gov(plan=_plan(workers=0, source="disabled"))
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    store = _Store(rg)
    pool = _ScriptedPool(store, answer=lambda entry, name, count: "cancelled" if (name, count) == ("b.txt", 1) else None)
    ingest, engine, job_id = _job(loaded, _texts(["a.txt", "b.txt", "c.txt"]), pool=pool, store=store)
    engine._worker_loop(job_id, threading.Event())
    by_name = _files(engine, job_id)
    assert [name for _entry, name, _held in pool.calls].count("b.txt") == 2
    assert sorted(s.name for s in store.stored) == ["a.txt", "b.txt", "c.txt"]
    assert {n: f.status for n, f in by_name.items()} == dict.fromkeys(by_name, ingest.FileStatus.DONE.value), {
        n: (f.status, f.error_message) for n, f in by_name.items()
    }
    assert engine.get_job(job_id).status == ingest.JobStatus.COMPLETED.value


def test_bw26_after_a_task_breaks_the_suspects_are_each_sent_in_a_worker_of_their_own_one_at_a_time():
    """BW26 -- the three files in flight break together, as the tasks of an
    executor whose worker died do: each is then sent through the pool's
    worker of its own, the next only once the one before is stored, so a
    death there names its file; the files after them go back to the shared
    workers, and every file is stored. Sending the suspects through the
    shared executor -> RED."""
    gov = _Gov(plan=_plan(workers=0, source="disabled"))
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    store = _Store(rg)
    pool = _ScriptedPool(store, answer=lambda entry, name, count: "broken" if count == 1 and name < "d" else None)
    names = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]
    ingest, engine, job_id = _job(loaded, _texts(names), pool=pool, store=store)
    engine._worker_loop(job_id, threading.Event())
    assert [(entry, name) for entry, name, _held in pool.calls[:3]] == [("submit", n) for n in names[:3]]
    assert pool.calls[3:6] == [
        ("submit_alone", "a.txt", []),
        ("submit_alone", "b.txt", ["a.txt"]),
        ("submit_alone", "c.txt", ["a.txt", "b.txt"]),
    ]
    assert [(entry, name) for entry, name, _held in pool.calls[6:]] == [("submit", "d.txt"), ("submit", "e.txt")]
    assert sorted(s.name for s in store.stored) == names
    assert engine.get_job(job_id).status == ingest.JobStatus.COMPLETED.value


class _StuckPool:
    """A pool whose tasks never complete until the contract completes them."""

    def __init__(self):
        self.futures = []

    def in_flight_limit(self):
        return 1

    def submit(self, fn, /, *args, caller):
        self.futures.append(Future())
        return self.futures[-1]

    submit_alone = submit

    def reset(self, broken=None):
        pass


def test_bw27_a_job_cancelled_while_it_waits_for_the_head_of_its_flight_leaves_at_once():
    """BW27 -- cancelled while it waits for the chunking at the head of its
    flight, one that never ends, the job stops waiting, cancels that task,
    gives its file back to the queue, stores nothing and ends cancelled,
    within 5 s of the cancel. Waiting for the head's result with no eye on
    the cancel -> RED."""
    gov = _Gov(plan=_plan(workers=0, source="disabled"))
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    store = _Store(rg)
    pool = _StuckPool()
    cancel = threading.Event()
    ingest, engine, job_id = _job(loaded, _texts(["a.txt", "b.txt"]), pool=pool, store=store)
    runner = threading.Thread(target=engine._worker_loop, args=(job_id, cancel), daemon=True)
    runner.start()
    try:
        assert _until(lambda: pool.futures, _WAIT_S), "the job sent its first file"
        cancel.set()
        runner.join(5.0)
        ended = not runner.is_alive()
    finally:
        for future in pool.futures:
            if not future.done():
                future.set_exception(RuntimeError("completed by the contract"))
        runner.join(_WAIT_S)
    assert ended, "the job ends within 5 s of its cancel"
    by_name = _files(engine, job_id)
    assert store.stored == []
    assert len(pool.futures) == 1 and pool.futures[0].cancelled()
    assert [by_name[n].status for n in ("a.txt", "b.txt")] == [ingest.FileStatus.QUEUED.value] * 2
    assert engine.get_job(job_id).status == ingest.JobStatus.CANCELLED.value


class _QueueGov(_Gov):
    """A governor whose queue holds the index until the cancel it is given
    is set (then refuses it, "cancelled") or the contract releases it, for
    30 s at most."""

    def __init__(self, plan):
        super().__init__(plan=plan)
        self.entered = threading.Event()
        self.release = threading.Event()

    def admit_or_wait(self, model, requested_ctx=None, caller="benchmark", cancel=None, **_kw):
        self.asked.append((model, caller))
        self.entered.set()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not self.release.is_set():
            if cancel is not None and cancel.is_set():
                return _decision(False, "cancelled")
            time.sleep(0.01)
        return _decision(False, "background_gate_closed")


def test_bw28_a_job_cancelled_while_its_admission_waits_leaves_the_queue_and_gives_the_file_back():
    """BW28 -- the job hands the governor its cancel as it asks to be
    admitted, so a cancel while the governor's queue holds the file's
    admission takes the job out of the queue: the job ends within 5 s, the
    file goes back to the queue, and nothing is stored. Asking for the
    admission with no cancel to honour -> RED."""
    gov = _QueueGov(plan=_plan(workers=0, source="disabled"))
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    pool = loaded[_POOL].BackgroundPool()
    store = _Store(rg)
    cancel = threading.Event()
    ingest, engine, job_id = _job(loaded, _texts(["a.txt", "b.txt"]), pool=pool, store=store)
    runner = threading.Thread(target=engine._worker_loop, args=(job_id, cancel), daemon=True)
    runner.start()
    try:
        assert gov.entered.wait(_WAIT_S), "the job asked for its admission"
        cancel.set()
        runner.join(5.0)
        ended = not runner.is_alive()
    finally:
        gov.release.set()
        runner.join(_WAIT_S)
        pool.shutdown()
    assert ended, "the job ends within 5 s of its cancel"
    by_name = _files(engine, job_id)
    assert store.stored == []
    assert [by_name[n].status for n in ("a.txt", "b.txt")] == [ingest.FileStatus.QUEUED.value] * 2
    assert engine.get_job(job_id).status == ingest.JobStatus.CANCELLED.value


class _OlderGov(_Gov):
    """A governor whose admission takes no cancel."""

    def admit_or_wait(self, model, requested_ctx=None, caller="benchmark"):
        self.asked.append((model, caller))
        return self.answers.pop(0) if self.answers else _decision()


def test_bw29_a_governor_whose_admission_takes_no_cancel_is_asked_without_it_and_its_ticket_held():
    """BW29 -- a governor whose admission takes no cancel is asked again
    without one: each file is admitted once, as the "index" caller, and
    stored under the ticket that admission returned. Failing open on that
    governor, the file stored under no ticket -> RED."""
    decisions = [_decision(), _decision()]
    gov = _OlderGov(plan=_plan(workers=0, source="disabled"), answers=decisions)
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    pool = loaded[_POOL].BackgroundPool()
    store = _Store(rg)
    ingest, engine, job_id = _job(loaded, _texts(["a.txt", "b.txt"]), pool=pool, store=store)
    engine._worker_loop(job_id, threading.Event())
    pool.shutdown()
    assert gov.asked == [("embed-m", "index")] * 2
    assert len(store.stored) == 2
    assert all(s.ticket is d for s, d in zip(store.stored, decisions))


class _MemoryGov(_Gov):
    """A governor that says why memory is short for the background, or
    None when it is not."""

    def __init__(self, plan, short):
        super().__init__(plan=plan)
        self.short = short
        self.memory_asked = 0

    def background_memory_short(self):
        self.memory_asked += 1
        return self.short


def test_bw30_while_the_governor_says_memory_is_short_the_job_keeps_one_file_in_flight():
    """BW30 -- before it sends a file while another is in flight, the job
    asks the governor whether memory is short for the background: told
    why, it keeps one file in flight (each worker holds a whole document as
    it parses it), as the job did before the pool; told it is not, it keeps
    up to the pool's limit. Every file is stored either way. Sending up to
    the limit whatever the governor says of memory -> RED."""
    names = [f"f{n}.txt" for n in range(10)]
    gov = _MemoryGov(_plan(workers=0, source="disabled"), "memory_pressure")
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)

    def most_held():
        store = _Store(rg)
        pool = _ScriptedPool(store, limit=8)
        ingest, engine, job_id = _job(loaded, _texts(names), pool=pool, store=store)
        engine._worker_loop(job_id, threading.Event())
        assert sorted(s.name for s in store.stored) == names
        return max(sent - len(stored) for sent, (_entry, _name, stored) in enumerate(pool.calls, start=1))

    assert most_held() == 1
    assert gov.memory_asked > 0
    gov.short = None
    assert most_held() == 8


class _RefusingPool(_StuckPool):
    """A pool whose tasks never complete and that refuses the third task
    sent to it, as a pool shut down under the job does."""

    def in_flight_limit(self):
        return 8

    def submit(self, fn, /, *args, caller):
        if len(self.futures) == 2:
            raise RuntimeError("the background pool is shut down")
        return super().submit(fn, *args, caller=caller)


def test_bw31_a_job_that_fails_mid_flight_gives_every_file_it_holds_back_to_the_queue():
    """BW31 -- the pool refuses the job's third file: the job fails with
    that reason, and no file is left "processing": the file being sent and
    the two in flight, their tasks cancelled, go back to the queue, beside
    the file never taken. Leaving the files the job held marked processing
    -> RED."""
    gov = _Gov(plan=_plan(workers=0, source="disabled"))
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    store = _Store(rg)
    pool = _RefusingPool()
    names = ["a.txt", "b.txt", "c.txt", "d.txt"]
    ingest, engine, job_id = _job(loaded, _texts(names), pool=pool, store=store)
    engine._worker_loop(job_id, threading.Event())
    by_name = _files(engine, job_id)
    job = engine.get_job(job_id)
    assert (job.status, job.error_message) == (ingest.JobStatus.FAILED.value, "the background pool is shut down")
    assert [by_name[n].status for n in names] == [ingest.FileStatus.QUEUED.value] * 4
    assert len(pool.futures) == 2 and all(f.cancelled() for f in pool.futures)
    assert store.stored == []


class _LateGov(_Gov):
    """A governor that admits the index as the job's cancel is set (the
    cancel lands while it answers), and records each load it is told will
    not happen."""

    def __init__(self, plan, cancel):
        super().__init__(plan=plan)
        self.cancel = cancel
        self.ended = []

    def admit_or_wait(self, model, requested_ctx=None, caller="benchmark", cancel=None, **_kw):
        self.asked.append((model, caller))
        self.cancel.set()
        decision = _decision()
        decision.ticket_id = "t-1"
        return decision

    def end_pending_load(self, ticket_id):
        self.ended.append(ticket_id)


def test_bw32_an_admission_that_lands_as_the_job_is_cancelled_is_handed_back_and_nothing_stored():
    """BW32 -- the governor admits a file's embeddings in the moment the job
    is cancelled: the job stores nothing, gives its files back to the queue,
    and tells the governor the load it admitted will not happen
    (end_pending_load, with the decision's ticket), so the background does
    not count a load nobody makes until it expires. Dropping an admission
    the job will not use -> RED."""
    cancel = threading.Event()
    gov = _LateGov(_plan(workers=0, source="disabled"), cancel)
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    store = _Store(rg)
    pool = _ScriptedPool(store)
    ingest, engine, job_id = _job(loaded, _texts(["a.txt", "b.txt"]), pool=pool, store=store)
    engine._worker_loop(job_id, cancel)
    by_name = _files(engine, job_id)
    assert store.stored == []
    assert gov.ended == ["t-1"]
    assert [by_name[n].status for n in ("a.txt", "b.txt")] == [ingest.FileStatus.QUEUED.value] * 2
    assert engine.get_job(job_id).status == ingest.JobStatus.CANCELLED.value


class _ClosingPool(_ScriptedPool):
    """A scripted pool that is shut down as it sends a file alone: the task
    it breaks then broke by the pool's own closing (the exit hook ends every
    worker), and the pool says it is closed."""

    def __init__(self, store, **kwargs):
        super().__init__(store, **kwargs)
        self.closed = False

    def _send(self, entry, fn, args):
        future = super()._send(entry, fn, args)
        if entry == "submit_alone":
            self.closed = True
        return future


def test_bw33_a_file_sent_alone_whose_worker_the_pools_shutdown_ends_is_not_blamed():
    """BW33 -- a file breaks its shared task and is sent alone, and the pool
    is shut down while it runs there (the server stops): its worker died of
    the shutdown, not of the file, so the file goes back to the queue rather
    than failing by name, the files already chunked are stored, and the job
    says the pool is shut down; a real pool says it is closed once shut
    down, and not before. Failing a file whose worker the shutdown ended
    -> RED."""
    gov = _Gov(plan=_plan(workers=0, source="disabled"))
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    store = _Store(rg)
    pool = _ClosingPool(store, answer=lambda entry, name, count: "broken" if name == "a.txt" else None)
    ingest, engine, job_id = _job(loaded, _texts(["a.txt", "b.txt"]), pool=pool, store=store)
    engine._worker_loop(job_id, threading.Event())
    by_name = _files(engine, job_id)
    assert [entry for entry, name, _held in pool.calls if name == "a.txt"] == ["submit", "submit_alone"]
    assert (by_name["a.txt"].status, by_name["b.txt"].status) == (
        ingest.FileStatus.QUEUED.value, ingest.FileStatus.DONE.value
    )
    job = engine.get_job(job_id)
    assert (job.status, job.error_message) == (ingest.JobStatus.FAILED.value, "the background pool is shut down")
    real = loaded[_POOL].BackgroundPool(plan=lambda: None, class_of=lambda caller: "background")
    assert real.closed is False
    real.shutdown()
    assert real.closed is True


# ---------------------------------------------------------------------------
# BW34-BW36 -- the exit, and memory for the job
# ---------------------------------------------------------------------------

# A process whose one worker writes the head of a result frame into its result
# pipe and sleeps; it says once the frame is out, then returns from main.
_HALF_SENT_CHILD = """
import os
import sys
import time
import types

sys.path.insert(0, {tests!r})
import _bw_tasks
from opti_oignon import background_pool

plan = types.SimpleNamespace(
    workers=1, cpus=(), reserved=(), in_flight=2, idle_s=60.0, held_retry_s=0.01, source="plan"
)
pool = background_pool.BackgroundPool(plan=lambda: plan, class_of=lambda caller: "background")
future = pool.submit(_bw_tasks.half_a_result, {marker!r}, 60.0, caller="index")
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline and not os.path.exists({marker!r}):
    time.sleep(0.02)
print("sent", os.path.exists({marker!r}), repr(time.monotonic()), flush=True)
"""

# A process whose executor is retired, without waiting, once its one worker
# runs a thirty-second task; it says so, then returns from main.
_ORPHAN_CHILD = """
import os
import sys
import time
import types

sys.path.insert(0, {tests!r})
import _bw_tasks
from opti_oignon import background_pool

plan = types.SimpleNamespace(
    workers=1, cpus=(), reserved=(), in_flight=2, idle_s=60.0, held_retry_s=0.01, source="plan"
)
pool = background_pool.BackgroundPool(plan=lambda: plan, class_of=lambda caller: "background")
first = pool.submit(_bw_tasks.mark_and_sleep, {marker!r}, 30.0, caller="index")
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline and not os.path.exists({marker!r}):
    time.sleep(0.02)
pool.reset(first)
print("retired", os.path.exists({marker!r}), repr(time.monotonic()), flush=True)
"""


def _run_child(script, tmp, *, wait_s):
    """``script`` run by a child Python on this tree's modules: (its words,
    its return code, when it ended, the end of its stderr); a child still
    running after ``wait_s`` is killed."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (env.get("PYTHONPATH"), str(REPO)) if p)
    out_path, err_path = tmp / "out.txt", tmp / "err.txt"
    with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
        child = subprocess.Popen(
            [sys.executable, "-c", script], cwd=str(tmp), env=env, stdout=out, stderr=err, start_new_session=True
        )
        try:
            child.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            pass
        finally:
            if child.poll() is None:
                # The child's own session: its workers go with it.
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        ended = time.monotonic()
    said = out_path.read_text(encoding="utf-8").split()
    return said, child.returncode, ended, err_path.read_text(encoding="utf-8")[-2000:]


def test_bw34_a_worker_ended_halfway_through_sending_its_result_does_not_hold_the_exit():
    """BW34 -- the pool closes at exit by ending its workers, and a worker
    ended while it sends its result leaves part of a frame in the result
    pipe, which the executor's reader then waits to finish for as long as
    any write end of the pipe is open: the pool closes the server's own copy
    once its workers are ended, so the reader meets the end of the pipe. A
    process whose one worker wrote half a result and sleeps exits within
    five seconds of returning from main, with status 0. Leaving the server's
    write end of the result pipe open -> RED."""
    tmp = _tmpdir("bw34-")
    marker = tmp / "sent"
    script = _HALF_SENT_CHILD.format(tests=str(Path(__file__).resolve().parent), marker=str(marker))
    said, code, ended, err = _run_child(script, tmp, wait_s=20.0)
    assert said[:2] == ["sent", "True"], err
    assert ended - float(said[2]) < 5.0, f"the process lived {ended - float(said[2]):.1f} s after main returned"
    assert code == 0, err


def test_bw35_an_executor_retired_without_waiting_is_ended_at_exit_with_its_running_task():
    """BW35 -- an executor the pool retires without waiting (a task that
    broke, a start that failed) keeps its workers until their tasks end; at
    exit the pool ends them too, as it ends its current ones: a process
    whose retired executor still runs a thirty-second task exits within four
    seconds of returning from main, with status 0. Losing the workers of an
    executor retired without waiting -> RED."""
    tmp = _tmpdir("bw35-")
    marker = tmp / "running"
    script = _ORPHAN_CHILD.format(tests=str(Path(__file__).resolve().parent), marker=str(marker))
    said, code, ended, err = _run_child(script, tmp, wait_s=20.0)
    assert said[:2] == ["retired", "True"], err
    assert ended - float(said[2]) < 4.0, f"the process lived {ended - float(said[2]):.1f} s after main returned"
    assert code == 0, err


class _RoomGov(_Gov):
    """A governor that gives background work ``room`` bytes of memory and
    says memory is not short."""

    def __init__(self, plan, room):
        super().__init__(plan=plan)
        self.room = room

    def background_memory_room(self):
        return self.room

    def background_memory_short(self):
        return None


def _sized(names_and_sizes):
    """Text files of the sizes given, in bytes, in a fresh directory."""
    folder = Path(tempfile.mkdtemp(prefix="bw-sized-"))
    _CLOSERS.append(lambda: shutil.rmtree(folder, ignore_errors=True))
    paths = []
    line = "The background reads this file once and cuts it where it can.\n"
    for name, size in names_and_sizes:
        path = folder / name
        path.write_text((line * (size // len(line) + 1))[:size], encoding="utf-8")
        paths.append(path)
    return paths


def _most_in_flight(calls):
    """The most files sent and not yet stored at any send, from a scripted
    pool's record of each send (the files the store held by then)."""
    return max(index + 1 - len(held) for index, (_entry, _name, held) in enumerate(calls))


def test_bw36_each_file_is_sent_against_an_estimate_of_the_memory_its_parse_takes():
    """BW36 -- a file goes to the pool against an estimate of the memory its
    parse takes (its size times the plan's factor for its kind), charged with
    the estimates already in flight against the room the governor gives the
    background (the RAM available less the reserve), so a burst of large
    documents never parses at once: with room for two files of 100 kB, no
    more than two are in flight; a file larger than the room is sent alone,
    and nothing is sent while it is in flight; every file is stored. Sending
    up to the pool's limit whatever the files weigh -> RED."""
    plan = _plan(workers=0, source="disabled")
    plan.parse_expansion = {"default": 1.0}
    gov = _RoomGov(plan, room=250_000)
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    store = _Store(rg)
    pool = _ScriptedPool(store, limit=8)
    sizes = [("a.txt", 100_000), ("b.txt", 100_000), ("c.txt", 100_000), ("e.txt", 300_000), ("f.txt", 100_000)]
    ingest, engine, job_id = _job(loaded, _sized(sizes), pool=pool, store=store)
    engine._worker_loop(job_id, threading.Event())
    assert _most_in_flight(pool.calls) == 2
    large = next(index for index, call in enumerate(pool.calls) if call[1] == "e.txt")
    assert large - len(pool.calls[large][2]) == 0, "nothing else is in flight as the large file is sent"
    assert all("e.txt" in held for _entry, _name, held in pool.calls[large + 1:]), (
        "nothing is sent while the large file is in flight"
    )
    assert sorted(s.name for s in store.stored) == sorted(name for name, _size in sizes)
    assert engine.get_job(job_id).status == ingest.JobStatus.COMPLETED.value


# ---------------------------------------------------------------------------
# BW37-BW40 -- the room across jobs, and workers left behind
# ---------------------------------------------------------------------------


class _HeldPool:
    """A pool whose tasks stay in flight until the contract releases them,
    and that records each file sent, by either entry."""

    def __init__(self, limit=8):
        self.limit = limit
        self.sent = []
        self.futures = {}
        self.lock = threading.Lock()

    def in_flight_limit(self):
        return self.limit

    def submit(self, fn, /, *args, caller):
        future = Future()
        future.set_running_or_notify_cancel()
        with self.lock:
            self.sent.append(os.path.basename(args[0]))
            self.futures[os.path.basename(args[0])] = (future, fn, args)
        return future

    submit_alone = submit

    def reset(self, broken=None):
        pass

    def release(self, name):
        future, fn, args = self.futures[name]
        future.set_result(fn(*args))


def test_bw37_the_memory_room_is_shared_by_every_job_of_the_engine():
    """BW37 -- two indexing jobs of one engine share the room the governor
    gives the background: while a file of the first fills it, the second
    job, nothing of its own in flight yet, waits for room rather than send
    beside it, and sends once that file has left; every file is stored.
    Charging each job's files against the room alone -> RED."""
    plan = _plan(workers=0, source="disabled")
    plan.parse_expansion = {"default": 1.0}
    gov = _RoomGov(plan, room=250_000)
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    store = _Store(rg)
    pool = _HeldPool()
    engine = loaded[_INGEST].BatchIngestEngine(data_dir=str(_tmpdir("bw37-")))
    engine._get_rag_store = lambda: store
    engine._pool = lambda: pool
    first = engine.create_batch_job([str(p) for p in _sized([("a1.txt", 200_000)])], collection="c")
    second = engine.create_batch_job([str(p) for p in _sized([("b1.txt", 100_000)])], collection="c")
    one = threading.Thread(target=engine._worker_loop, args=(first.job_id, threading.Event()), daemon=True)
    two = threading.Thread(target=engine._worker_loop, args=(second.job_id, threading.Event()), daemon=True)
    one.start()
    assert _until(lambda: pool.sent == ["a1.txt"], _WAIT_S), "the first job sends its file"
    two.start()
    time.sleep(1.0)
    held = list(pool.sent)
    pool.release("a1.txt")
    assert _until(lambda: "b1.txt" in pool.sent, _WAIT_S), "the second job sends once the room frees"
    pool.release("b1.txt")
    one.join(_WAIT_S)
    two.join(_WAIT_S)
    assert held == ["a1.txt"], "the second job waits while the first one's file fills the room"
    assert sorted(s.name for s in store.stored) == ["a1.txt", "b1.txt"]


def test_bw38_a_file_sent_alone_is_charged_against_the_room_like_any_other():
    """BW38 -- a file whose shared task broke is sent alone, in a worker of
    its own, and its estimated parse is charged against the room as any
    file's is: no file is sent beside it past the room, so the next file
    waits until it has left. Sending a suspect with no estimate -> RED."""
    plan = _plan(workers=0, source="disabled")
    plan.parse_expansion = {"default": 1.0}
    gov = _RoomGov(plan, room=250_000)
    rg = _governor_module(gov)
    loaded = _open(governor=rg, ingest=True)
    store = _Store(rg)
    pool = _ScriptedPool(store, answer=lambda entry, name, count: "broken" if (entry, name) == ("submit", "a.txt") else None)
    ingest, engine, job_id = _job(loaded, _sized([("a.txt", 200_000), ("b.txt", 100_000)]), pool=pool, store=store)
    engine._worker_loop(job_id, threading.Event())
    assert [(entry, name) for entry, name, _held in pool.calls] == [
        ("submit", "a.txt"), ("submit_alone", "a.txt"), ("submit", "b.txt")
    ]
    assert "a.txt" in pool.calls[2][2], "the next file is sent only once the suspect has left"
    assert sorted(s.name for s in store.stored) == ["a.txt", "b.txt"]


class _Mortal:
    """A worker process the pool may end: alive until it is terminated."""

    def __init__(self, pid):
        self.pid = pid
        self.alive = True
        self.terminated = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def join(self, timeout=None):
        pass


class _Tracked(_Held):
    """An executor with one worker process and a result pipe, both of which
    its shutdown forgets, as the standard library's does."""

    def __init__(self, pid, **kwargs):
        super().__init__(**kwargs)
        self.process = _Mortal(pid)
        self.writer = types.SimpleNamespace(closed=False)
        self.writer.close = lambda: setattr(self.writer, "closed", True)
        self._processes = {pid: self.process}
        self._result_queue = types.SimpleNamespace(_writer=self.writer)

    def shutdown(self, wait=True, *, cancel_futures=False):
        super().shutdown(wait, cancel_futures=cancel_futures)
        self._processes = None
        self._result_queue = None


def test_bw39_a_worker_of_its_own_whose_executor_fails_as_its_task_is_sent_is_ended():
    """BW39 -- the executor of a worker of its own may fail as the task is
    sent (a thread that cannot start) after its worker process was spawned,
    which would block on its queue and hold the exit: the pool ends that
    worker and closes the server's end of its result pipe, and the task runs
    on the caller's thread, said. Dropping the executor alone -> RED."""
    bp = _open()[_POOL]
    made = []

    class _Failing(_Tracked):
        def submit(self, fn, /, *args, **kwargs):
            raise RuntimeError("can't start new thread")

    def factory(workers, initializer, initargs):
        made.append(_Failing(200 + len(made)))
        return made[-1]

    pool = bp.BackgroundPool(plan=lambda: _plan(workers=1), class_of=lambda caller: "background", executor_factory=factory)
    assert pool.submit_alone(len, "abc", caller="index").result(timeout=_WAIT_S) == 3
    assert (made[0].process.terminated, made[0].writer.closed) == (True, True)
    assert pool.status()["mode"] == "inline"


def test_bw40_retired_workers_are_kept_while_they_live_and_ended_at_shutdown():
    """BW40 -- an executor retired without waiting leaves its worker to end
    its task; the pool keeps such a worker only while it lives (one that has
    exited is dropped as the next executor is retired, not only when the
    status is read), and its shutdown ends those still alive, closing the
    server's end of their result pipes, as the exit does. Keeping every
    retired worker, or leaving the live ones past a shutdown -> RED."""
    bp = _open()[_POOL]
    made = []

    def factory(workers, initializer, initargs):
        made.append(_Tracked(100 + len(made)))
        return made[-1]

    pool = bp.BackgroundPool(plan=lambda: _plan(workers=1), class_of=lambda caller: "background", executor_factory=factory)
    first = pool.submit(len, "a", caller="index")
    pool.reset(first)
    made[0].process.alive = False
    second = pool.submit(len, "b", caller="index")
    pool.reset(second)
    assert [remains[0][0].pid for remains in pool._orphans] == [101]
    pool.shutdown()
    assert (made[1].process.terminated, made[1].writer.closed) == (True, True)
    assert (made[0].process.terminated, pool._orphans) == (False, [])


def _room_engine(room, pool):
    """An engine whose jobs chunk through ``pool`` under a governor that
    gives the background ``room`` bytes, each file weighing its size."""
    plan = _plan(workers=0, source="disabled")
    plan.parse_expansion = {"default": 1.0}
    rg = _governor_module(_RoomGov(plan, room=room))
    loaded = _open(governor=rg, ingest=True)
    store = _Store(rg)
    engine = loaded[_INGEST].BatchIngestEngine(data_dir=str(_tmpdir("bw-room-")))
    engine._get_rag_store = lambda: store
    engine._pool = lambda: pool
    return engine, store


def test_bw41_a_cancelled_job_keeps_the_room_of_a_parse_that_runs_on_until_it_ends():
    """BW41 -- a job cancelled while a file's parse runs gives the file back
    to the queue, but a running task cannot be cancelled and the parse goes
    on in its worker: its room stays charged until the parse ends, so a job
    started next waits for it rather than parse beside it past the room.
    Freeing the room of a parse that still runs -> RED."""
    pool = _HeldPool()
    engine, store = _room_engine(250_000, pool)
    first = engine.create_batch_job([str(p) for p in _sized([("a1.txt", 200_000)])], collection="c")
    second = engine.create_batch_job([str(p) for p in _sized([("b1.txt", 100_000)])], collection="c")
    cancel = threading.Event()
    one = threading.Thread(target=engine._worker_loop, args=(first.job_id, cancel), daemon=True)
    one.start()
    assert _until(lambda: pool.sent == ["a1.txt"], _WAIT_S), "the first job sends its file"
    cancel.set()
    one.join(_WAIT_S)
    assert not one.is_alive(), "the cancelled job ends while its file's parse runs on"
    two = threading.Thread(target=engine._worker_loop, args=(second.job_id, threading.Event()), daemon=True)
    two.start()
    time.sleep(1.0)
    held = list(pool.sent)
    pool.release("a1.txt")
    assert _until(lambda: "b1.txt" in pool.sent, _WAIT_S), "the next job sends once the parse has ended"
    pool.release("b1.txt")
    two.join(_WAIT_S)
    assert held == ["a1.txt"], "the next job waits while the cancelled job's parse still holds the room"
    assert [s.name for s in store.stored] == ["b1.txt"]


def test_bw42_a_job_waiting_for_room_is_served_before_another_sends_more():
    """BW42 -- the room is served to the job that waits for it before a job
    that already has files in flight sends more: with room for 250 kB, a
    job holding two files of 100 kB defers its third while another job waits
    to send one of 200 kB, which goes as soon as the first job's files have
    left; the third goes after it. Letting a job with files in flight take
    every room another job waits for -> RED."""
    pool = _HeldPool()
    engine, store = _room_engine(250_000, pool)
    first = engine.create_batch_job(
        [str(p) for p in _sized([("a1.txt", 100_000), ("a2.txt", 100_000), ("a3.txt", 100_000)])], collection="c"
    )
    second = engine.create_batch_job([str(p) for p in _sized([("b1.txt", 200_000)])], collection="c")
    one = threading.Thread(target=engine._worker_loop, args=(first.job_id, threading.Event()), daemon=True)
    two = threading.Thread(target=engine._worker_loop, args=(second.job_id, threading.Event()), daemon=True)
    released = set()

    def release(name):
        if name not in released and name in pool.futures:
            released.add(name)
            pool.release(name)

    try:
        one.start()
        assert _until(lambda: pool.sent == ["a1.txt", "a2.txt"], _WAIT_S), "the first job fills the room"
        two.start()
        time.sleep(0.5)
        release("a1.txt")
        time.sleep(0.5)
        release("a2.txt")
        assert _until(lambda: "b1.txt" in pool.sent, 5.0), "the waiting job is served before the other sends more"
        release("b1.txt")
        assert _until(lambda: "a3.txt" in pool.sent, _WAIT_S)
        release("a3.txt")
        one.join(_WAIT_S)
        two.join(_WAIT_S)
    finally:
        for name in ("a1.txt", "a2.txt", "a3.txt", "b1.txt"):
            if _until(lambda: name in pool.futures, 0.5):
                release(name)
    assert pool.sent == ["a1.txt", "a2.txt", "b1.txt", "a3.txt"]
    assert sorted(s.name for s in store.stored) == ["a1.txt", "a2.txt", "a3.txt", "b1.txt"]
