"""The chat's search of a project's Mycelium store: the passages one turn is given, in place of the knowledge base.

``passages(project_id, paths, query)`` answers a ``Turn`` whose ``state`` says where the turn's
project context comes from:

* ``none``: the project has no store (never indexed, or no file the Mycelium reads): the knowledge base serves it;
* ``rag``: the store holds none of the files the project's rows name, a secret's name aside: the knowledge base
  serves the whole project, said;
* ``both`` (262): the store holds some of them and not others (a PDF, a document, a sheet, an image, a file not
  indexed yet, one the Mycelium does not read): the passages of the files it holds, as in ``ok``, and the paths of
  the others (``uncovered``), which the knowledge base serves, said;
* ``unreadable``: the store is there and cannot be opened or searched: nothing of the Mycelium, said in the turn and
  in the log once per project, reason and process, and never the knowledge base in its place;
* ``ok``: the passages, widened as the agent's tool widens them, each with the name the file was added under, its
  lines and their state now (fresh, modified, gone).

The search reads the ``public`` and ``private`` sources, never a ``secret`` one, in every mode. It keeps the bridge
(the question in English and rewritten) and the judge when their settings are read. The question's vector waits
``VECTOR_S`` at most; the whole search ``DEADLINE_S``, counted from the store's opening: past it, a search of the
lexical stages alone (names and full text, no bridge, no vector, no judge) answers, said, and the first one ends on
its own thread, closing its own store -- stopped (263): from the deadline on it sends nothing more to the model or the
encoder (the bridge's and the judge's next questions refused, the vector and the judging stages left out), a call
already sent running to its end, its answer unused.
Each search makes its own bridge, which keeps no question past it: nothing of a conversation's questions stays in this
process once its search ends (a search past the deadline ends after its turn). Read only: nothing here writes a
store."""
import logging
import threading
import time
from dataclasses import dataclass, field

from . import place, projects
from .store import StoreError

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

ALLOW = ("public", "private")       # never "secret", in either mode
DEFAULT_K = 8
DEADLINE_S = 10.0                   # the whole search of a turn
VECTOR_S = 5.0                      # the question's vector
LEXICAL = ("symbols", "lexical")
MAX_UNCOVERED_SAID = 5              # the files the knowledge base serves, named in a turn

_lock = threading.Lock()
_said = set()                       # (project, why) an unreadable store was said for


@dataclass
class Turn:
    state: str
    passages: list = field(default_factory=list)    # {"name", "first", "last", "text", "state"}
    said: str = ""
    uncovered: tuple = ()       # in "both": the paths of the project's files the store does not hold


def _settings_or_none(load, make):
    try:
        return make(load())
    except Exception as exc:  # noqa: BLE001 - a stage whose settings cannot be read is silent
        logger.info("mycelium: a stage of the chat's search is off (%s)", type(exc).__name__)
        return None


def _bridge():
    """A bridge of this search alone (its cache dies with it); None when its settings are off."""
    from .bridge import Bridge, load_settings

    return _settings_or_none(load_settings, Bridge)


def _reranker():
    from .rerank import Reranker, load_settings

    return _settings_or_none(load_settings, Reranker)


def _unreadable(project_id, why):
    with _lock:
        first = (project_id, why) not in _said
        _said.add((project_id, why))
    if first:
        logger.warning("project %s: its Mycelium store cannot be read (%s): its turns get no project passage",
                       project_id, why)
    return Turn("unreadable", said=f"the project's Mycelium store cannot be read ({why}): no project passage")


def _covered(corpus, paths):
    """``(how many of the files the rows name the store holds, the paths of those it does not)``, a secret's name
    aside on both sides (never indexed, never searched here, never handed to the knowledge base by this rule); ``None``
    when there is no store."""
    try:
        store = place.open_to_read(corpus)
    except place.NoStore:
        return None
    try:
        held = {row[0] for row in store.db.execute("SELECT path FROM file")}
    finally:
        store.close()
    from .index import secret_name

    named = {}
    for path in paths:
        name = path.rsplit("/", 1)[-1]
        if not (secret_name(name) or secret_name(projects._original(name))):
            named[path] = name
    uncovered = tuple(sorted(path for path, name in named.items() if name not in held))
    return len(named) - len(uncovered), uncovered


class _Stopping:
    """A model's asker (the bridge's, the judge's) that sends nothing once the turn's deadline has passed (``stop``
    set): each later question raises the asker's own error, so the stage is left out and said, as when the model does
    not answer; the bridge's asker checks the stop again once admitted, just before the question leaves. A question
    already sent runs to its end: a request in flight is not cut."""

    def __init__(self, asker, stop):
        import inspect

        self.asker, self.stop = asker, stop
        try:    # the bridge's own asker checks the stop again once admitted, just before its question leaves
            self.passes_stop = "stop" in inspect.signature(asker.ask).parameters
        except (TypeError, ValueError):
            self.passes_stop = False

    def ask(self, prompt, num_predict, system=None, timeout_s=None):
        from .bridge import AskError

        if self.stop.is_set():
            raise AskError("the turn's deadline passed: not asked")
        if self.passes_stop:
            return self.asker.ask(prompt, num_predict, system=system, timeout_s=timeout_s, stop=self.stop)
        return self.asker.ask(prompt, num_predict, system=system, timeout_s=timeout_s)


def _search(corpus, query, k, lexical, stop=None):
    """One search: (passages, silences) -- the store opened and closed here, on the thread that searches; ``stop``,
    once set, ends its calls to the model and the encoder (``_Stopping``, ``Searcher.stop``)."""
    from .search import Searcher
    from .tool import widened

    store = place.open_to_read(corpus)
    try:
        if lexical:
            searcher = Searcher(store, allow=ALLOW, timeout_s=VECTOR_S)
            found = searcher.find(query, k, stages=LEXICAL)
        else:
            reranker, bridge = _reranker(), _bridge()
            if stop is not None:
                for model in (bridge, reranker):
                    if model is not None:
                        model.asker = _Stopping(model.asker, stop)
            searcher = Searcher(store, allow=ALLOW, timeout_s=VECTOR_S, bridge=bridge, reranker=reranker, stop=stop)
            found = searcher.find(query, k, rerank=reranker is not None)
        out = []
        for span in widened(searcher, found.hits):
            if span["lines"] is None:
                continue
            out.append({"name": projects._original(span["path"].rsplit("/", 1)[-1]), "first": span["first"],
                        "last": span["last"], "text": "\n".join(span["lines"][span["first"] - 1:span["last"]]),
                        "state": span["state"]})
        return out, dict(found.silent)
    finally:
        store.close()


def passages(project_id, paths, query, k=DEFAULT_K):
    """The turn's passages from the project's store (see the module's doc); never raises."""
    started = time.monotonic()      # the deadline counts the store's opening too
    try:
        corpus = projects.corpus_of(project_id)
    except StoreError:
        return Turn("none")
    try:
        covered = _covered(corpus, list(paths))
    except Exception as exc:  # noqa: BLE001 - a store that does not open: said once, nothing else
        return _unreadable(project_id, type(exc).__name__)
    if covered is None:
        return Turn("none")
    held, uncovered = covered
    if uncovered and not held:
        return Turn("rag", said="the project holds files its Mycelium store does not: the knowledge base serves it")
    done, result, stop = threading.Event(), {}, threading.Event()

    def full():
        try:
            result["full"] = _search(corpus, query, k, lexical=False, stop=stop)
        except Exception as exc:  # noqa: BLE001 - named below
            result["error"] = exc
        finally:
            done.set()

    threading.Thread(target=full, name="mycelium-turn", daemon=True).start()
    said = ""
    if done.wait(max(0.0, DEADLINE_S - (time.monotonic() - started))) and "full" in result:
        found, silent = result["full"]
    elif "error" in result:
        return _unreadable(project_id, type(result["error"]).__name__)
    else:
        stop.set()          # the full search sends nothing more to the model or the encoder
        try:
            found, silent = _search(corpus, query, k, lexical=True)
        except Exception as exc:  # noqa: BLE001 - a store that does not answer: said once, nothing else
            return _unreadable(project_id, type(exc).__name__)
        said = f"the search took longer than {DEADLINE_S:g} s: the lexical stages answered"
    if silent:
        logger.info("project %s: stages left out of its turn's search: %s", project_id, ", ".join(sorted(silent)))
    if uncovered:
        names = [projects._original(path.rsplit("/", 1)[-1]) for path in uncovered]
        shown = ", ".join(names[:MAX_UNCOVERED_SAID]) + (f" and {len(names) - MAX_UNCOVERED_SAID} more"
                                                          if len(names) > MAX_UNCOVERED_SAID else "")
        told = (f"the project holds {len(names)} file(s) its Mycelium store does not ({shown}): those are left to "
                "the knowledge base")
        return Turn("both", passages=found, said="; ".join(part for part in (said, told) if part), uncovered=uncovered)
    return Turn("ok", passages=found, said=said)
