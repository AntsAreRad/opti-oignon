"""The traits of a search's candidate files: what the value head (``value.py``) orders them by.

For a question, the search's own lists (``Searcher.lists``: the symbols, the full text, the full text rewritten by
the bridge, the vectors) are fused as the search fuses them; the files of the fused parts, in that order and each
once, are the candidates, ``depth`` at most, each named by its source and its path (two sources may hold the same
path; a folder is a source's). Each candidate is a row of the traits ``TRAITS`` names, read from the
store and from those lists, no model asked beyond the ones the search already asks:

  dense_best      the greatest cosine of the question to the file's parts, over the whole matrix
  dense_rank      1 / (1 + the file's rank among the files of the vectors' list); 0 when it is not there
  lexical_rank    the same among the files of the full text's list
  lexical_best    the greatest score of the file's parts in the full text's list
  rewritten_rank  the same rank among the files of the rewritten full text's list
  defines_named   1 when the file defines an identifier the question names (the symbols' list)
  fused_best      the greatest fused score of the file's parts
  centroid        the cosine of the question to the mean of the file's parts' vectors
  folder          the cosine of the question to the mean of the parts' vectors of the file's folder
  parts           log(1 + the file's parts)
  links           log(1 + the other files that name a symbol of at least four characters the file defines; a name
                  defined more often than the search's own bound, MAX_DEFINITIONS, links nothing)
  kind_*          the file's kind, by the language the store read it in (py, ts -- JavaScript too --, svelte, md,
                  yaml, rs; another: none of them)
  test            1 when the file is a test (under a tests folder, or named test_*)
  path_words      the share of the question's words (as the words-matching stages read it) in the file's path

A list the search left silent (no encoder active, the bridge away, the turn's deadline) leaves its traits at 0, and
``silent`` says why, as the search says it. The store's tables -- each file's parts, its links, the sums of its and
its folder's vectors -- are read in one reading of the store (one transaction, under the store's lock: a pass that
commits meanwhile waits), and kept for the store's file (whichever opening of it a searcher reads: the turn opens one
per search) and the sensitivities and sources read, while the store's generation and a digest of its sources, files
and parts hold (another store's bytes in the same file are another store); a store's file gone takes its tables
with it. Past the turn's deadline they are not begun, and the traits they give are 0, said (``silent["traits"]``);
tables read while the store's matrix could not be are said so and not kept. A source the searcher may not read
gives no candidate, no link and no vector to a mean."""
import hashlib
import math
import os
import re
import threading
from collections import OrderedDict, namedtuple
from contextlib import contextmanager

from .search import MAX_DEFINITIONS, fuse

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

KINDS = ("py", "ts", "svelte", "md", "yaml", "rs")
# The store's languages (``chunk.language``) as kinds.
KIND_OF = {"python": "py", "ts": "ts", "svelte": "svelte", "markdown": "md", "yaml": "yaml", "rust": "rs"}
TRAITS = ("dense_best", "dense_rank", "lexical_rank", "lexical_best", "rewritten_rank", "defines_named", "fused_best",
          "centroid", "folder", "parts", "links", *(f"kind_{k}" for k in KINDS), "test", "path_words")
DEPTH = 40
MIN_NAME = 4
WORD = re.compile(r"[a-z0-9]+")

Candidates = namedtuple("Candidates", "files rows silent")

# Per store's file, per (sensitivities, sources) read: (generation, the tables of ``_tables``); the latest MAX_HELD.
_HELD = OrderedDict()
_HELD_LOCK = threading.Lock()
MAX_HELD = 8


def _folder(path):
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _is_test(path):
    name = path.rsplit("/", 1)[-1]
    return path.startswith("tests/") or "/tests/" in path or name.startswith("test_")


@contextmanager
def _one_reading(db):
    """The statements inside read one state of the store: one transaction (opened here unless one is open), ended
    without a write."""
    own = not db.in_transaction
    if own:
        db.execute("BEGIN")
    try:
        yield
    finally:
        if own:
            db.execute("ROLLBACK")


def _stamp(store):
    """The store's generation and a digest of its sources, files and parts: what the tables were read from."""
    digest = hashlib.sha256()
    for row in store.db.execute("SELECT id, name, root, sensitivity FROM source ORDER BY id"):
        digest.update(repr(row).encode())
    for row in store.db.execute("SELECT id, source, path, sha256 FROM file ORDER BY id"):
        digest.update(repr(row).encode())
    digest.update(repr(tuple(store.db.execute("SELECT COUNT(*), MIN(id), MAX(id), TOTAL(id) FROM part").fetchone()))
                  .encode())
    return store.generation(), digest.hexdigest()


def _tables(searcher):
    """The store's tables (``_read``) in one reading of it, under its lock; when the lock is not had within the
    searcher's ``timeout_s``, none, ``complete`` False."""
    store = searcher.store
    if not store.lock.acquire(timeout=searcher.timeout_s):
        return {"file_of": {}, "named": {}, "parts": {}, "links": {}, "norms": {}, "folder_norms": {},
                "matrix_rows": None, "complete": False, "stamp": None}
    try:
        with _one_reading(store.db):
            tables = _read(searcher)
            tables["stamp"] = _stamp(store)
            return tables
    finally:
        store.lock.release()


def _read(searcher):
    """What the traits read of the store beyond the search's lists, for what ``searcher`` may read: each part's file
    (every part, with or without a vector), each file's source, path and language and its parts, its links, and the
    norms of the sums of the matrix's rows of each file and of each folder -- with them, the cosine of a mean to the
    question is the sum of its rows' cosines over that norm. ``complete`` is False when the matrix could not be read
    (its store held past the searcher's ``timeout_s``)."""
    import numpy as np

    store = searcher.store
    allowed, values = searcher._filter()
    file_of, named, parts = {}, {}, {}
    for part_id, file_id, source, path, lang in store.db.execute(
            f"SELECT p.id, f.id, s.name, f.path, f.lang FROM part p JOIN file f ON f.id = p.file "
            f"JOIN source s ON s.id = f.source WHERE {allowed}", values):
        file_of[part_id] = file_id
        named[file_id] = (source, path, lang)
        parts[file_id] = parts.get(file_id, 0) + 1
    # The links: as the search's symbols stage, a name defined more often than MAX_DEFINITIONS says little.
    definers, defined = {}, {}
    for name, part_id in store.db.execute(
            f"SELECT y.name, y.part FROM symbol y JOIN part p ON p.id = y.part JOIN file f ON f.id = p.file "
            f"JOIN source s ON s.id = f.source WHERE length(y.name) >= {MIN_NAME} AND {allowed}", values):
        defined[name] = defined.get(name, 0) + 1
        definers.setdefault(name, set()).add(file_of.get(part_id))
    namers = {}
    for name, part_id in store.db.execute(
            f"SELECT r.name, r.part FROM ref r JOIN part p ON p.id = r.part JOIN file f ON f.id = p.file "
            f"JOIN source s ON s.id = f.source WHERE length(r.name) >= {MIN_NAME} AND {allowed}", values):
        if defined.get(name, MAX_DEFINITIONS + 1) > MAX_DEFINITIONS:
            continue
        namer = file_of.get(part_id)
        for definer in definers[name]:
            if definer is not None and namer is not None and definer != namer:
                namers.setdefault(definer, set()).add(namer)
    links = {file_id: len(found) for file_id, found in namers.items()}
    matrix_rows, norms, folder_norms, complete = None, {}, {}, True
    held = searcher._matrix()
    if held is None:
        complete = False
    elif held[0] is not None and held[1]:
        _active, ids, matrix = held
        files = [file_of.get(i) for i in ids]
        file_index = {f: n for n, f in enumerate(sorted({f for f in files if f is not None}))}
        places = sorted({_place(named[f]) for f in file_index})
        folder_index = {d: n for n, d in enumerate(places)}
        index = np.array([file_index[f] if f is not None else -1 for f in files])
        # Summed run by run of rows of one file (a file's parts are rows in a row) in the matrix's own precision,
        # without a copy of it, then in double precision over the runs.
        starts = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
        run_sums = np.add.reduceat(matrix, starts, axis=0).astype(np.float64)
        run_files = index[starts]
        kept = run_files >= 0
        sums = np.zeros((len(file_index), matrix.shape[1]))
        np.add.at(sums, run_files[kept], run_sums[kept])
        folder_sums = np.zeros((len(places), matrix.shape[1]))
        by_folder = np.array([folder_index[_place(named[f])] for f in sorted(file_index, key=file_index.get)],
                             dtype=np.int64)
        if len(by_folder):
            np.add.at(folder_sums, by_folder, sums)
        norms = {f: float(np.linalg.norm(sums[n])) for f, n in file_index.items()}
        folder_norms = {d: float(np.linalg.norm(folder_sums[n])) for d, n in folder_index.items()}
        matrix_rows = (list(ids), files)
    return {"file_of": file_of, "named": named, "parts": parts, "links": links, "norms": norms,
            "folder_norms": folder_norms, "matrix_rows": matrix_rows, "complete": complete}


def _place(named):
    """A file's folder, as its source's: (source, folder)."""
    return named[0], _folder(named[1])


def _key(searcher):
    """The store's file (its real path, its device and inode: a store made again at the same path is another) and
    what the searcher may read; None when the file cannot be looked at (the tables are then read, never kept)."""
    try:
        seen = os.stat(searcher.store.path)
    except OSError:
        return None
    return (os.path.realpath(searcher.store.path), seen.st_dev, seen.st_ino, tuple(sorted(set(searcher.allow))),
            None if searcher.sources is None else tuple(sorted(set(searcher.sources))))


def _held(searcher, build=True):
    """``_tables`` of ``searcher``, read again only when the store's stamp (``_stamp``) moved; None when they are not
    held and ``build`` is False. Each call lets go the tables of a store whose file is gone or is another file."""
    key = _key(searcher)
    with _HELD_LOCK:
        for other in list(_HELD):
            try:
                seen = os.stat(other[0])
                gone = (seen.st_dev, seen.st_ino) != other[1:3]
            except OSError:
                gone = True
            if gone:
                del _HELD[other]
    if key is not None:
        with _HELD_LOCK:
            held = _HELD.get(key)
        if held is not None:
            store = searcher.store
            if store.lock.acquire(timeout=searcher.timeout_s):
                try:
                    with _one_reading(store.db):
                        stamp = _stamp(store)
                finally:
                    store.lock.release()
                if stamp == held[0]:
                    with _HELD_LOCK:
                        if key in _HELD:
                            _HELD.move_to_end(key)
                    return held[1]
    if not build:
        return None
    tables = _tables(searcher)
    if key is not None and tables["complete"]:
        with _HELD_LOCK:
            _HELD[key] = (tables["stamp"], tables)
            _HELD.move_to_end(key)
            while len(_HELD) > MAX_HELD:
                _HELD.popitem(last=False)
    return tables


def _files_alone(searcher, part_ids):
    """The files of ``part_ids`` alone, for a search past its deadline: no part counted, no link, no vector."""
    allowed, values = searcher._filter()
    file_of, named = {}, {}
    ids = sorted({int(i) for i in part_ids})
    for at in range(0, len(ids), 500):
        chunk = ids[at:at + 500]
        for part_id, file_id, source, path, lang in searcher.store.db.execute(
                f"SELECT p.id, f.id, s.name, f.path, f.lang FROM part p JOIN file f ON f.id = p.file "
                f"JOIN source s ON s.id = f.source WHERE p.id IN ({','.join('?' * len(chunk))}) AND {allowed}",
                (*chunk, *values)):
            file_of[part_id] = file_id
            named[file_id] = (source, path, lang)
    return {"file_of": file_of, "named": named, "parts": {}, "links": {}, "norms": {}, "folder_norms": {},
            "matrix_rows": None, "complete": False}


def _file_order(ranked, file_of):
    order = []
    for part_id, _score in ranked:
        file_id = file_of.get(part_id)
        if file_id is not None and file_id not in order:
            order.append(file_id)
    return order


def rows(searcher, query, depth=DEPTH) -> Candidates:
    """The candidate files of ``query`` (the fused list's files, in its order, ``depth`` at most), each as (source,
    path), and a row of ``TRAITS`` for each, as floats."""
    listed = searcher.lists(query)
    silent = dict(listed.silent)
    order, fused, _why = fuse(listed.lists)
    late = searcher.stop is not None and searcher.stop.is_set()
    tables = _held(searcher, build=not late)
    if tables is None:
        tables = _files_alone(searcher, order)
        silent["traits"] = ("the search's deadline passed: the store's tables are left out (the files' parts, their "
                            "links, the means of their vectors)")
    elif not tables["complete"]:
        silent["traits"] = "the store's vectors could not be read for the traits: their traits are left at 0"
    file_of, named = tables["file_of"], tables["named"]
    candidates = []
    for part_id in order if depth > 0 else ():
        file_id = file_of.get(part_id)
        if file_id is not None and file_id not in candidates:
            candidates.append(file_id)
            if len(candidates) >= depth:
                break
    lists = listed.lists
    ranks = {stage: _file_order(lists.get(stage, []), file_of) for stage in ("dense", "lexical", "rewritten")}
    lexical_best = {}
    for part_id, score in lists.get("lexical", []):
        file_id = file_of.get(part_id)
        if file_id is not None:
            lexical_best[file_id] = max(lexical_best.get(file_id, float(score)), float(score))
    defining = {file_of.get(part_id) for part_id, _s in lists.get("symbols", [])}
    fused_best = {}
    for part_id, score in fused.items():
        file_id = file_of.get(part_id)
        if file_id is not None:
            fused_best[file_id] = max(fused_best.get(file_id, 0.0), score)
    dense_best, sums, folder_sums = {}, {}, {}
    if listed.scores is not None and tables["matrix_rows"] is not None:
        ids, files = tables["matrix_rows"]
        scored_ids, scores = listed.scores
        if scored_ids != ids:
            silent["traits"] = "the store moved between the search and its tables: the vectors' traits are left at 0"
        else:
            for file_id, score in zip(files, scores):
                if file_id is None:
                    continue
                score = float(score)
                dense_best[file_id] = max(dense_best.get(file_id, score), score)
                sums[file_id] = sums.get(file_id, 0.0) + score
                place = _place(named[file_id])
                folder_sums[place] = folder_sums.get(place, 0.0) + score
    words = set(WORD.findall(listed.worded.lower()))
    out = []
    for file_id in candidates:
        source, path, lang = named[file_id]
        kind = KIND_OF.get(lang, "")
        path_words = set(WORD.findall(path.lower().replace("_", " ")))
        norm = tables["norms"].get(file_id, 0.0)
        place = _place(named[file_id])
        folder_norm = tables["folder_norms"].get(place, 0.0)

        def rank(stage, file_id=file_id):
            files = ranks[stage]
            return 1.0 / (1 + files.index(file_id)) if file_id in files else 0.0

        out.append([
            dense_best.get(file_id, 0.0),
            rank("dense"),
            rank("lexical"),
            lexical_best.get(file_id, 0.0),
            rank("rewritten"),
            1.0 if file_id in defining else 0.0,
            fused_best.get(file_id, 0.0),
            sums[file_id] / norm if file_id in sums and norm > 0 else 0.0,
            folder_sums[place] / folder_norm if place in folder_sums and folder_norm > 0 else 0.0,
            math.log1p(tables["parts"].get(file_id, 0)),
            math.log1p(tables["links"].get(file_id, 0)),
            *(1.0 if kind == k else 0.0 for k in KINDS),
            1.0 if _is_test(path) else 0.0,
            len(words & path_words) / max(len(words), 1),
        ])
    return Candidates([named[f][:2] for f in candidates], out, silent)
