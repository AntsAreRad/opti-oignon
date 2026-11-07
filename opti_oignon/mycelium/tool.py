"""The Mycelium as the agent's tool: ``mycelium_search(query, k?)``, and the card the agent's prompt carries.

The sources a run searches are those rooted where the run's workspace was cloned from (``workspace_root``: the host
folder ``sandbox_workspace`` recorded for the session, its real path), public or private, in the store that holds
them: never another source of that store -- a folder elsewhere the owner indexed beside it --, never another store,
never a secret source. A run with no workspace, a workspace not cloned from a host folder, or a folder no store holds,
is told so, with the owner's route that indexes a folder; a store the platform's opener refuses is named as such,
never as a folder not indexed. A store is read read-only (``place.open_to_read``); it is kept open for the process
while its file is the one opened and the machine's security mode the one it was opened under, opened again otherwise
(so the platform's opener judges it again -- a plaintext store is refused in Bulbe), and closed as soon as its file is
gone -- always under its lock, never under a running search. One search at a time reads a store (its lock): its
connection is one. One folder indexed under two sources of a store gives each passage once.

A search asks the judge (``rerank``) when its settings can be read, and the bridge for a French question; each that
cannot be had leaves the search to the others, said in one line. Each passage is widened as the frozen recipe widened
them (myc5x, D8): 20 lines at least, around the part found, and the passages of one file that meet or overlap merged,
until none do, and held to the lines the file has now; it is cited ``path:first-last`` with the state of its lines
(``fresh``, ``modified``, ``gone``) and given while the answer has room -- at most ``MAX_BYTES`` bytes and ``MAX_LINES``
lines in all, under the agent loop's caps on an observation (16384 bytes, 256 lines), a line longer than
``MAX_LINE_CHARS`` cut and said: a passage that does not fit is given its first lines that do, its citation saying
which, a passage of which no line fits is named and the next ones still tried, and the passages past the bound are
named by their citations. When the judge said that no passage
answers, the answer says that verdict first, and that the judge is wrong about one time in four or five on a question
the corpus does answer. What the tool gives back is data: the agent's loop wraps every tool result as untrusted
before the model reads it. A handler never raises."""
import json
import logging
import os
import threading

from . import place

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

TOOL = "mycelium_search"
ALLOW = ("public", "private")
DEFAULT_K = 5
MAX_K = 10
MIN_LINES = 20          # a passage of 20 lines at least (myc5x, D8)
MAX_BYTES = 14000       # the whole answer, under the agent loop's 16384 bytes of an observation (obs_max_bytes)
MAX_LINES = 200         # the whole answer, under the agent loop's 256 lines of an observation (obs_max_lines)
MAX_LINE_CHARS = 400    # a longer line (minified code, a data line) is cut and said
MAX_NAMED = 20          # citations named past the bound, at most
MAX_QUERY = 2000
ROUTE = "python -m opti_oignon mycelium index CORPUS FOLDER"
NO_WORKSPACE = ("No workspace is bound to this run, so no Mycelium corpus is searched. The owner indexes a folder with "
                f"`{ROUTE}`, and a run whose workspace was cloned from that folder searches it.")
UPLOADED = ("This run's workspace was not cloned from a host folder, so no Mycelium corpus belongs to it. A run whose "
            f"workspace is cloned from a folder the owner indexed with `{ROUTE}` searches it.")
NO_CORPUS = ("No Mycelium corpus is indexed for this run's workspace (cloned from {root}). The owner indexes it with "
             f"`{ROUTE}`.")
UNREADABLE = ("No readable Mycelium corpus holds this run's workspace (cloned from {root}); the store of {names} could "
              "not be opened ({why}) and may hold it: indexing again does not help, the store is refused as it is.")
JUDGED_NONE = ("The judge read the first passages and found that none of them answers this question. The judge is "
               "wrong about one time in four or five on questions the corpus does answer: the closest passages "
               "follow.")

_open = {}              # store path -> (device, inode, security mode, Store)
_lock = threading.Lock()


def _workspace(sandbox):
    """('none', None), ('uploaded', None) or ('cloned', the real path of the host folder it was cloned from)."""
    if sandbox is None:
        return "none", None
    try:
        session_id = sandbox.session_id
        path = sandbox.sandbox_manager.get_workspace_path(session_id)
        if not session_id or not path:
            return "none", None
        from opti_oignon import sandbox_workspace

        root = sandbox_workspace.get_workspace_manifests().get_cloned_root(session_id, path)
    except Exception:  # noqa: BLE001 - a run with no readable workspace searches nothing
        return "none", None
    return ("cloned", os.path.realpath(root)) if root else ("uploaded", None)


def workspace_root(sandbox):
    """The real path of the host folder the run's workspace was cloned from, or None."""
    return _workspace(sandbox)[1]


def _mode():
    try:
        from opti_oignon import provenance

        return provenance.machine_mode()
    except Exception:  # noqa: BLE001 - a mode that cannot be read is the tighter one
        return "bulbe"


def _close(store):
    try:
        store.close()
    except Exception:  # noqa: BLE001 - a connection to a removed file closes as it can
        pass


def _close_after(store):
    """Close ``store`` once no search holds it (its lock)."""
    with store.lock:
        _close(store)


def _evict_gone():
    """Every store kept open whose file is gone or replaced: forgotten, then closed once no search holds it."""
    stale = []
    with _lock:
        for path, held in list(_open.items()):
            try:
                now = os.stat(path)
                gone = held[:2] != (now.st_dev, now.st_ino)
            except OSError:
                gone = True
            if gone:
                stale.append(_open.pop(path)[3])
    for store in stale:
        _close_after(store)


def _store(corpus):
    """The corpus's store, read-only, kept while its file is the one opened under the same security mode; None when
    it is gone."""
    path = str(place.path_for(corpus))
    mode = _mode()
    try:
        held_file = os.stat(path)
    except OSError:
        held_file = None
    stale = None
    try:
        with _lock:
            held = _open.get(path)
            if held is not None and held_file is not None and \
                    held[:3] == (held_file.st_dev, held_file.st_ino, mode):
                return held[3]
            if held is not None:
                stale = _open.pop(path)[3]
            if held_file is None:
                return None
            store = place.open_to_read(corpus)
            _open[path] = (held_file.st_dev, held_file.st_ino, mode, store)
            return store
    finally:
        if stale is not None:
            _close_after(stale)


def corpus_for(root, refused=None):
    """(corpus, store, the names of its public or private sources rooted at ``root``, compared by real path), or
    None. A store that cannot be read is passed over, said in the log and named in ``refused`` (corpus -> why)."""
    _evict_gone()
    if not root:
        return None
    for corpus in place.corpora():
        try:
            store = _store(corpus)
        except Exception as exc:  # noqa: BLE001 - one unreadable store hides no other
            logger.warning("mycelium: the store of %s cannot be read for the agent's tool (%s)", corpus,
                           type(exc).__name__)
            if refused is not None:
                refused[corpus] = str(exc)[:200] or type(exc).__name__
            continue
        if store is None:
            continue
        with store.lock:
            sources = store.sources()
        names = tuple(s["name"] for s in sources
                      if s["sensitivity"] in ALLOW and os.path.realpath(s["root"]) == root)
        if names:
            return corpus, store, names
    return None


def _said_path(path):
    """A host path as the tool says it: quoted, its control characters escaped."""
    return json.dumps(str(path))


def _not_found(root, refused):
    """Why no corpus answers for ``root``: none holds it, or the stores that may were refused."""
    if refused:
        names = ", ".join(sorted(refused))
        why = "; ".join(f"{name}: {refused[name]}" for name in sorted(refused))
        return UNREADABLE.format(root=_said_path(root), names=names, why=why)
    return NO_CORPUS.format(root=_said_path(root))


def card(sandbox):
    """The lines the agent's tools section carries about what this run searches, or why nothing."""
    state, root = _workspace(sandbox)
    if state == "none":
        return "Mycelium: " + NO_WORKSPACE
    if state == "uploaded":
        return "Mycelium: " + UPLOADED
    refused = {}
    found = corpus_for(root, refused)
    if found is None:
        return "Mycelium: " + _not_found(root, refused)
    corpus, store, names = found
    with store.lock:
        lines = []
        for source in store.sources():
            if source["name"] not in names:
                continue
            files = store.db.execute("SELECT COUNT(*) FROM file WHERE source = ?", (source["id"],)).fetchone()[0]
            last = next(iter(store.episodes(source["id"], 1)), None)
            lines.append(f"  - {source['name']} ({source['kind']}, {files} files"
                         + (f", indexed {last['at']})" if last else ")"))
    return ("Mycelium: mycelium_search searches what the owner indexed of this run's workspace, in the corpus `"
            + corpus + "`, by symbols, full text and meaning; its passages are cited path:first-last, quote the "
            "citations you use. A passage marked modified or gone no longer says what the file says now. Its "
            "sources:\n" + "\n".join(lines))


def _settings_or_none(load, make):
    try:
        return make(load())
    except Exception as exc:  # noqa: BLE001 - a stage whose settings cannot be read is silent
        logger.info("mycelium: a stage of the agent's tool is off (%s)", exc)
        return None


def _searcher(store, names):
    from .bridge import Bridge
    from .bridge import load_settings as bridge_settings
    from .rerank import Reranker
    from .rerank import load_settings as rerank_settings
    from .search import Searcher

    return Searcher(store, allow=ALLOW, bridge=_settings_or_none(bridge_settings, Bridge),
                    reranker=_settings_or_none(rerank_settings, Reranker), sources=names)


def _count(value):
    if value is None:
        return DEFAULT_K
    if type(value) is not int:
        raise ValueError("k is a whole number of passages")
    return max(1, min(MAX_K, value))


def widened(searcher, hits):
    """The passages, each widened to MIN_LINES lines at least around its part and held to the lines the file has now,
    those of one file -- one folder's file, whatever source of the store names it -- that meet or overlap merged until
    none do, in the order their first part came; each with the state of its lines read now."""
    from .search import source_lines

    spans, lines_of = [], {}
    for hit in hits:
        root = searcher.held_root(hit)
        key = (os.path.realpath(root) if root and not root.startswith("documents:") else root, hit["path"])
        if key not in lines_of:
            lines_of[key] = source_lines({"root": root, "path": hit["path"]}, searcher.readers) if root else None
        lines = lines_of[key]
        total = len(lines) if lines is not None else hit["last"]
        first, last = hit["first"], hit["last"]
        missing = MIN_LINES - (last - first + 1)
        if missing > 0:
            first = max(1, first - missing // 2)
            last = min(total, first + MIN_LINES - 1)
            first = max(1, min(first, last - MIN_LINES + 1))
        last = min(last, max(total, 1))
        first = min(first, last)
        spans.append({"key": key, "path": hit["path"], "first": first, "last": last, "parts": [hit], "lines": lines})
    merged = True
    while merged:
        merged = False
        for i, span in enumerate(spans):
            for other in spans[i + 1:]:
                if other["key"] == span["key"] and other["first"] <= span["last"] + 1 \
                        and other["last"] >= span["first"] - 1:
                    span["first"], span["last"] = min(span["first"], other["first"]), max(span["last"], other["last"])
                    span["parts"].extend(other["parts"])
                    spans.remove(other)
                    merged = True
                    break
            if merged:
                break
    for span in spans:
        state = "fresh"
        if span["lines"] is None:
            state = "gone"
        elif any(searcher.verify(part) != "fresh" for part in span["parts"]):
            state = "modified"
        span["state"] = state
    return spans


def _size(text):
    return len(text.encode("utf-8"))


def _shown(line):
    """A line as the answer gives it: a line longer than MAX_LINE_CHARS cut, its cut said."""
    if len(line) <= MAX_LINE_CHARS:
        return line
    return line[:MAX_LINE_CHARS] + f" [... {len(line) - MAX_LINE_CHARS} more characters of this line]"


def render(searcher, found, judged):
    """The tool's answer: the judge's verdict when it found none, the silences, then the widened passages while the
    answer has room (MAX_BYTES, MAX_LINES in all, every head and the closing line counted): a passage that does not fit
    gives its first lines that do, a passage of which no line fits is named and the next ones tried, and once a
    passage is cut the others are named."""
    out = []
    if judged is False:
        out.append(JUDGED_NONE)
    if found.silent:
        said = "; ".join(f"{k}: {v}" for k, v in sorted(found.silent.items()))
        out.append("Stages left out of this search: " + said[:1000])
    if not found.hits:
        out.append("No passage of the corpus matches this search.")
        return "\n".join(out)
    # The closing line names at most MAX_NAMED citations: its room is kept from the start.
    closing_room = 200 + MAX_NAMED * 120
    used_bytes = sum(_size(block) + 2 for block in out) + closing_room
    used_lines = sum(block.count("\n") + 2 for block in out) + 2
    named, cut = [], False
    for span in widened(searcher, found.hits):
        cite = f"{span['path']}:{span['first']}-{span['last']}"
        if cut:
            named.append(cite)
            continue
        body = [] if span["lines"] is None else [_shown(line) for line in span["lines"][span["first"] - 1:span["last"]]]
        head_room = _size(cite) + 160
        given, size = [], head_room
        for line in body:
            if used_bytes + size + _size(line) + 1 > MAX_BYTES or used_lines + len(given) + 3 > MAX_LINES:
                break
            given.append(line)
            size += _size(line) + 1
        if body and not given:
            named.append(cite)          # not a line of it fits: named, and the next passage tried
            continue
        if len(given) < len(body):
            last = span["first"] + len(given) - 1
            head = (f"[{span['path']}:{span['first']}-{last}] ({span['state']}; lines {last + 1}-{span['last']} of "
                    "this passage left out: the answer's bound)")
            cut = True
        else:
            head = f"[{cite}] ({span['state']})"
        block = head + ("\n" + "\n".join(given) if given else "")
        out.append(block)
        used_bytes += _size(block) + 2
        used_lines += len(given) + 2
    if named:
        more = len(named) - MAX_NAMED
        out.append("Passages left out by the answer's bound: " + ", ".join(named[:MAX_NAMED])
                   + (f", and {more} more" if more > 0 else ""))
    return "\n\n".join(out)


def make_handler(sandbox):
    """The run's handler of ``mycelium_search``: what the owner indexed of the run's workspace, searched (see the
    module's doc); never raises."""
    state, root = _workspace(sandbox)

    def handle(arguments):
        try:
            arguments = arguments if isinstance(arguments, dict) else {}
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return "mycelium_search needs a query: what to look for, in words or identifiers."
            query = query[:MAX_QUERY]
            try:
                k = _count(arguments.get("k"))
            except ValueError as exc:
                return f"mycelium_search: {exc}."
            if state == "none":
                return NO_WORKSPACE
            if state == "uploaded":
                return UPLOADED
            refused = {}
            held = corpus_for(root, refused)
            if held is None:
                return _not_found(root, refused)
            _corpus, store, names = held
            with store.lock:        # one search at a time on the store's one connection
                searcher = _searcher(store, names)
                found = searcher.find(query, k, rerank=searcher.reranker is not None)
                judged = searcher.reranker.found(found.hits) if searcher.reranker is not None else None
                return render(searcher, found, judged)
        except Exception as exc:  # noqa: BLE001 - a tool's failure is said to the model, never raised into the run
            logger.warning("mycelium: the agent's search failed (%s)", type(exc).__name__)
            return f"The Mycelium could not search this time ({type(exc).__name__})."

    return handle
