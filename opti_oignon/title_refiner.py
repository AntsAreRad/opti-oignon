"""Rule (b): a conversation's automatic title refined by the local model once an answer is sent; off as shipped.

Before an answer's ``done`` (routes_chat), ``maybe_refine`` hands the title rule (a) drew to a background thread,
when titles.yaml's ``refine.enabled`` holds or the machine's local.yaml says so. The thread asks the local model
titles.yaml or local.yaml names, or else the one the conversation was answered with, for a short title, handing it
the words rule (a) keeps of the title's turn and nothing else -- never the answer, a document, a pasted part, nor a
word the rule drops (the store's ``title_state``). Once admitted, the title is read again, and the stop and the cancel
checked after it, right before the call starts: a title that is no longer the one refined -- renamed, deleted, its
turn withdrawn or gone, its words changed -- is not asked about. The answer is held to the rule's guard
(``conversation_titles.accept_refined``: every word exactly one of those or a function word, none the rule would
drop) and written over the rule's title only while that title is still the one
refined (the store's ``commit_refined_title``, a compare-and-set). Refused, lost or late, the rule's title stays.
Each title is asked once in a process; a rename forgets the conversation's.

Only a backend whose own requests stay on this machine is called, whatever the security mode
(``inference_backend.is_on_this_machine``; ``local_refusal`` lets a request leave in Daily): in the process, or to
localhost, a loopback or an unspecified address. One that cannot say where its requests go, or hands them to
another process that chooses (``forwards_requests``: the core daemon), is never called. A server on this machine
that sends a request on -- a proxy, a model it serves from elsewhere -- is judged by its own address. The call is the
background's: admitted by the resource governor as caller ``titler``, its ticket held on the thread that asks the
backend, an admission no call holds handed back; a governor that cannot be had admits nothing. Past its deadline the
call is given up, and no other call to that model starts while one given up lives. With the emergency stop engaged,
or unreadable, nothing is dispatched, called or written. Each wait for the model -- in the queue, for the backend's
resolution, at the governor, for the answer -- is bounded by ``refine.timeout_s`` (a resolution given up counts as a
hung call: one at most a model); the store is read as every turn reads it. A title is ``pending`` until its job ends,
is cancelled or outlives its deadline, whichever comes first. The rename and the deletion routes cancel the job.

Nothing here logs a word: what the process counted (``counts``) are numbers. The registry, the governor and the
emergency stop are imported when a job needs them, never at import.
"""

import collections
import importlib
import logging
import threading
import time
from dataclasses import dataclass, field

from . import conversation_titles as _titles

logger = logging.getLogger(__name__)

# Who asks, as the resource governor's caller table names it: the background.
CALLER = "titler"
# The option that carries the deadline to a backend, as the librarian sends it.
_TIMEOUT_OPTION = "timeout"
# The waits a job may live through, each bounded by refine.timeout_s: the queue, the backend's resolution, the
# governor, the answer.
_WAITS = 4

_clock = time.monotonic
_lock = threading.Lock()
# The jobs waiting, oldest first; every job queued or running, by conversation.
_queue: collections.deque = collections.deque()
_jobs: dict = {}
# The threads draining the queue.
_draining = [0]
# The titles asked in this process, by (conversation, title_turn), oldest first.
_attempted: collections.OrderedDict = collections.OrderedDict()
# The calls given up at their deadline, by model, while their backend has not returned.
_hung: dict = {}
_counts: collections.Counter = collections.Counter()


@dataclass
class _Job:
    conv_id: str
    title_turn: str
    basis: str
    title: str
    words: list
    model: str
    queued_at: float
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)


def _counted(outcome):
    with _lock:
        _counts[outcome] += 1
    logger.debug("title refinement: %s", outcome)


def counts():
    """What the process counted, by outcome: numbers, never a word."""
    with _lock:
        return dict(_counts)


def reset():
    """Forget every job, attempt, hung call and count (a test's fresh start); a job running writes nothing."""
    with _lock:
        for job in _jobs.values():
            job.cancelled.set()
        _queue.clear()
        _jobs.clear()
        _attempted.clear()
        _hung.clear()
        _counts.clear()


def _module(name):
    """The package's module ``name``, or None where it cannot be loaded: asked at the call, never at import."""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - absence is an answer
        logger.debug("title refinement: %s unavailable (%s)", name, type(exc).__name__)
        return None


def _stopped():
    """Whether the emergency stop is engaged; True when it cannot be read: nothing starts on what cannot be told."""
    stop = _module("opti_oignon.emergency_stop")
    if stop is None:
        return True
    try:
        return bool(stop.is_stopped())
    except Exception:  # noqa: BLE001 - an unreadable stop stops
        return True


def _handed(words, limit):
    """The kept words that fit in ``limit`` characters joined by spaces, from the first: what the model is handed."""
    handed, size = [], 0
    for word in words:
        grown = size + len(word) + (1 if handed else 0)
        if grown > limit:
            break
        handed.append(word)
        size = grown
    return handed


def maybe_refine(manager, conv_id, *, runner=None):
    """Queue the refinement of the conversation's title when it may be refined; whether a job was queued.

    Never raises and never waits on a model: at most a read of the store. Nothing is queued when titles or the
    refinement are off or their configuration unreadable, when the emergency stop is engaged or unreadable, when the
    title is not the rule's own (``title_state``: a placeholder, a manual, legacy or refined title, a turn that does
    not decrypt), when this title was asked already in this process or a job of the conversation is queued or
    running, when no model is named, when not even the first kept word fits in ``refine.max_input_chars``, or when
    the queue is full. ``runner(manager, cfg)`` starts a drain of the queue; a background thread when not given.
    """
    try:
        cfg = manager.titles_config()
        if cfg is None or not cfg.enabled or not cfg.refine.enabled or not conv_id:
            return False
        if _stopped():
            _counted("stopped")
            return False
        state = manager.title_state(conv_id)
        if state is None:
            return False
        title_turn, basis, title, words, model = state
        model = cfg.refine.model or model
        handed = _handed(words, cfg.refine.max_input_chars)
        if not model or not handed:
            _counted("not_asked")
            return False
        now = _clock()
        with _lock:
            if (conv_id, title_turn) in _attempted or conv_id in _jobs:
                return False
            if len(_queue) >= cfg.refine.max_queued:
                _counts["dropped_queue_full"] += 1
                return False
            _attempted[(conv_id, title_turn)] = True
            while len(_attempted) > cfg.refine.attempted_cap:
                _attempted.popitem(last=False)
            job = _Job(conv_id, title_turn, basis, title, handed, model, now,
                       now + _WAITS * cfg.refine.timeout_s)
            _queue.append(job)
            _jobs[conv_id] = job
            _counts["dispatched"] += 1
            start = _draining[0] < cfg.refine.max_in_flight
            if start:
                _draining[0] += 1
        if start:
            try:
                (runner or _default_runner)(manager, cfg)
            except Exception:  # noqa: BLE001 - a drain that cannot start: what no drain will run is taken back
                with _lock:
                    _draining[0] -= 1
                    # Another drain still running runs the queue; with none, every queued job is taken back, its
                    # title to be asked again, never left queued without a drain.
                    taken = list(_queue) if _draining[0] == 0 else []
                    for each in taken:
                        _queue.remove(each)
                        if _jobs.get(each.conv_id) is each:
                            del _jobs[each.conv_id]
                        _attempted.pop((each.conv_id, each.title_turn), None)
                    _counts["taken_back"] += len(taken)
                raise
        return True
    except Exception as exc:  # noqa: BLE001 - nothing here may break an answer
        logger.debug("title refinement not dispatched (%s)", type(exc).__name__)
        return False


def _default_runner(manager, cfg):
    threading.Thread(target=_drain, args=(manager, cfg), name="oo-titler", daemon=True).start()


def _drain(manager, cfg):
    """Run the queued jobs one after the other until none is left."""
    while True:
        with _lock:
            if not _queue:
                _draining[0] -= 1
                return
            job = _queue.popleft()
        try:
            _run(manager, cfg, job)
        except Exception as exc:  # noqa: BLE001 - a job's failure is counted, never raised
            _counted("error")
            logger.debug("title refinement failed (%s)", type(exc).__name__)
        finally:
            with _lock:
                if _jobs.get(job.conv_id) is job:
                    del _jobs[job.conv_id]


def pending(conv_id):
    """Whether a refinement of the conversation's title is queued or running, not cancelled, before its deadline."""
    with _lock:
        job = _jobs.get(conv_id)
        return job is not None and not job.cancelled.is_set() and _clock() < job.deadline


def cancel(conv_id):
    """Cancel the conversation's refinement and forget its titles asked; whether a job was cancelled. Never raises.

    A queued job never runs; a running one writes nothing. Its call, already sent, cannot be taken back: the store's
    compare-and-set would refuse a late answer anyway.
    """
    try:
        with _lock:
            for asked in [asked for asked in _attempted if asked[0] == conv_id]:
                del _attempted[asked]
            job = _jobs.pop(conv_id, None)
            if job is None:
                return False
            job.cancelled.set()
            queued = job in _queue
            if queued:
                _queue.remove(job)
        if queued:
            # A job already running counts its own end when it sees the cancel.
            _counted("cancelled")
        return True
    except Exception as exc:  # noqa: BLE001 - a cancel never fails the route that asks it
        logger.debug("title refinement not cancelled (%s)", type(exc).__name__)
        return False


def _within(seconds, model, name, work):
    """``(finished, value)``: ``work()`` on a thread of its own, waited for ``seconds``. Past them it is given up and
    kept among the model's hung threads, so no other call to that model starts while it lives; an error is
    ``(True, None)``."""
    out = {}

    def run():
        try:
            out["value"] = work()
        except BaseException as exc:  # noqa: BLE001 - handed back below as no value
            out["error"] = exc

    worker = threading.Thread(target=run, name=name, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        with _lock:
            _hung.setdefault(model, set()).add(worker)
        return False, None
    if "error" in out:
        logger.debug("title refinement: %s failed (%s)", name, type(out["error"]).__name__)
        return True, None
    return True, out.get("value")


def _hanging(model):
    """Whether a call or a resolution given up for ``model`` still lives; those that returned are forgotten."""
    with _lock:
        hanging = {thread for thread in _hung.get(model, ()) if thread.is_alive()}
        _hung[model] = hanging
    return bool(hanging)


def _backend(model, seconds):
    """``(finished, backend)``: the backend the registry resolves for ``model``, None when there is none; not finished
    when its resolution -- which may ask the engine -- outlives ``seconds``."""
    backends = _module("opti_oignon.inference_backend")
    if backends is None:
        return True, None
    return _within(seconds, model, "oo-titler-resolve", lambda: backends.get_backend_registry().resolve_backend(model))


def _on_this_machine(backend):
    """Whether ``backend`` says its requests stay on this machine: in the process, or to localhost, a loopback or an
    unspecified address. A backend that cannot say, or a check that cannot be made, is elsewhere."""
    backends = _module("opti_oignon.inference_backend")
    if backends is None:
        return False
    try:
        # A backend that hands its requests to another process says only their first hop; one that does not say
        # whether it does is taken for one that does.
        if getattr(backend, "forwards_requests", True) is not False:
            return False
        endpoint = backend.endpoint()
        if not isinstance(endpoint, str) or not endpoint:
            return False
        return endpoint == backends.ENDPOINT_IN_PROCESS or bool(backends.is_on_this_machine(endpoint))
    except Exception:  # noqa: BLE001 - what cannot be told is elsewhere
        return False


def _run(manager, cfg, job):
    refine = cfg.refine
    if job.cancelled.is_set():
        return _counted("cancelled")
    if _clock() > job.queued_at + refine.timeout_s:
        return _counted("expired")
    if _stopped():
        return _counted("stopped")
    if _hanging(job.model):
        return _counted("hung")
    finished, backend = _backend(job.model, refine.timeout_s)
    if not finished:
        return _counted("resolve_timeout")
    if backend is None:
        return _counted("no_backend")
    if not _on_this_machine(backend):
        return _counted("refused_off_machine")

    def current():
        """Whether the title is still the one refined, read again from the store."""
        state = manager.title_state(job.conv_id)
        return state is not None and state[:2] == (job.title_turn, job.basis)

    answer = _ask(backend, refine, job, current)
    if answer is None:
        return None
    title = _titles.accept_refined(answer, job.words, cfg.derive, refine)
    if title is None:
        return _counted("guard_refused")
    if title == job.title:
        return _counted("unchanged")
    if job.cancelled.is_set() or _stopped():
        return _counted("cancelled")
    if manager.commit_refined_title(job.conv_id, job.title_turn, job.basis, title):
        return _counted("committed")
    return _counted("cas_lost")


def _ask(backend, refine, job, current):
    """The model's answer; None, counted, when no call may start, the call is not admitted, the title is no longer
    the one refined once admitted (``current``), or the call outlives its deadline or fails. The ticket is held on the
    thread that asks the backend, and an admission no call holds is handed back."""
    if _hanging(job.model):
        return _counted("hung")
    governance = _module("opti_oignon.resource_governor")
    try:
        governor = governance.get_resource_governor() if governance is not None else None
    except Exception:  # noqa: BLE001 - a governor that cannot be had admits nothing
        governor = None
    if governor is None:
        return _counted("not_admitted")
    try:
        decision = governor.admit_or_wait(job.model, requested_ctx=refine.num_ctx, caller=CALLER,
                                          wait_s=float(refine.timeout_s), cancel=job.cancelled)
    except Exception:  # noqa: BLE001 - a governor that raises admits nothing
        return _counted("not_admitted")
    if not getattr(decision, "admitted", False):
        return _counted("not_admitted")

    def hand_back():
        handed = getattr(governor, "end_pending_load", None)
        if callable(handed):
            try:
                handed(decision.ticket_id)
            except Exception as exc:  # noqa: BLE001 - the load expires by itself
                logger.debug("title refinement: an admission not handed back (%s)", type(exc).__name__)

    try:
        still = current()
    except Exception as exc:  # noqa: BLE001 - a title that cannot be read again is not asked about
        logger.debug("title refinement: the title not read again (%s)", type(exc).__name__)
        still = False
    if not still:
        hand_back()
        return _counted("stale")
    # After the store's read, which may wait: the stop or a cancel that came meanwhile starts nothing.
    if job.cancelled.is_set() or _stopped():
        hand_back()
        return _counted("cancelled")
    window = getattr(decision, "num_ctx", None)
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        window = refine.num_ctx
    options = {"temperature": refine.temperature, "seed": refine.seed, "num_predict": refine.num_predict,
               "num_ctx": window, _TIMEOUT_OPTION: refine.timeout_s}
    messages = [{"role": "system", "content": refine.system_prompt},
                {"role": "user", "content": " ".join(job.words)}]
    keep_alive = getattr(decision, "keep_alive", None) or refine.keep_alive
    answer, entered = {}, []

    def call():
        try:
            with governance.ticket_scope(decision):
                entered.append(True)
                answer["response"] = backend.generate(model=job.model, messages=messages, options=options,
                                                      keep_alive=keep_alive, think=False)
        except BaseException as exc:  # noqa: BLE001 - handed back to the job below
            answer["error"] = exc
        finally:
            if not entered:
                hand_back()

    worker = threading.Thread(target=call, name="oo-titler-call", daemon=True)
    try:
        worker.start()
    except BaseException:
        hand_back()
        raise
    worker.join(refine.timeout_s)
    if worker.is_alive():
        with _lock:
            _hung.setdefault(job.model, set()).add(worker)
        return _counted("timeout")
    if "error" in answer:
        return _counted("error")
    return str(getattr(answer.get("response"), "content", "") or "")
