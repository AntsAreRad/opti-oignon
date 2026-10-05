#!/usr/bin/env python3
"""Tasks the background-pool contracts send to a worker process.

A worker is a fresh interpreter: it imports a task by its module and never
sees a test's patches, so whatever a contract wants a worker to do has to be
a function of an importable module. Each function here imports the package
inside the call, so loading this file in the test process reaches no project
module.

Stdlib-only at import. Used by contract suites; never by the package.
"""

import os
import sys
import threading


def census_after_thread():
    """The worker's census, taken while a thread the task started is alive,
    with that thread's id: a thread born after the worker's entry."""
    from opti_oignon import background_pool

    ready = threading.Event()
    release = threading.Event()

    def _hold():
        ready.set()
        release.wait(30.0)

    thread = threading.Thread(target=_hold, name="bw-late-thread")
    thread.start()
    ready.wait(30.0)
    try:
        report = background_pool.census()
        report["late_thread"] = thread.native_id
        return report
    finally:
        release.set()
        thread.join(30.0)


def chunk_or_die(marker, filepath, doc_id, chunk_size, chunk_overlap):
    """The job's chunk task, except that the worker dies on the file named
    ``marker``, as a worker the kernel kills or a parser that crashes does."""
    if os.path.basename(filepath) == marker:
        os._exit(70)
    from opti_oignon.rag_chunker import chunk_file_task

    return chunk_file_task(filepath, doc_id, chunk_size, chunk_overlap)


def chunk_then_modules(filepath, doc_id, chunk_size, chunk_overlap):
    """Chunk a file as the job's task does, then name every module the
    worker holds."""
    from opti_oignon.rag_chunker import chunk_file_task

    chunk_file_task(filepath, doc_id, chunk_size, chunk_overlap)
    return sorted(sys.modules)


def mark_and_sleep(marker, seconds):
    """Say in the file ``marker`` that the task runs in its worker, then
    sleep ``seconds``."""
    import time

    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("running")
    time.sleep(seconds)


def half_a_result(marker, seconds):
    """Write the head of a result frame into this worker's result pipe and no
    more, as a worker ended halfway through sending a large result leaves
    it, say so in the file ``marker``, and sleep: the owner's reader then
    waits inside the frame for bytes that never come. The pipe is the
    worker's result queue, a local of the standard library's worker loop
    that calls this task."""
    import struct
    import time

    frame = sys._getframe(1)
    while frame is not None and "result_queue" not in frame.f_locals:
        frame = frame.f_back
    writer = frame.f_locals["result_queue"]._writer
    writer._send(struct.pack("!i", 1 << 20) + b"x" * 1024)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("sent")
    time.sleep(seconds)
