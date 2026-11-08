"""A project's files in the Mycelium: one store per project, ``project_<id>``, held to the files its rows name.

A pass reads, as the one source ``files`` of the project's store, private, the files the project's rows name in its
files folder (``<projects folder>/<id>/files``) -- never the folder's other entries: a file an interrupted removal left
on the disk with no row never enters the store --, and none the indexer does not read (a PDF, a docx, an image, a name
a secret's file bears, read without the id the projects put before it: those stay with the knowledge base's index of
the project). What the store held of a file the
rows no longer name leaves it at that pass, and the store is rewritten so that its words leave the pages
(``Store.rewrite``). A project with no file left has no store: it is overwritten and removed, as a forgotten
project's is (``forget``); the sweep (``sweep``) removes the stores of the projects that are gone and, from a live
project's store, the files its rows no longer name, and runs a rewrite a store still owes. Only a corpus named
``project_`` and a project's id
(``PROJECT_ID``, the projects' own rule) is a project's: the owner's route refuses those names, and the sweep never
touches another. After an index pass comes the vector pass, as the owner's route runs it: one at a time on the
install (``place.one_vector_pass``, never waited for), each request admitted by the resource governor as background
work; vectors held are left to a later pass.

Nothing here raises into its caller: each function returns what it could not do, named, each line starting with the
inventory's label of the project indexes (never a file's words), and says it in the log. A deletion's caller names
those lines and its next sweep takes them up again."""
import contextlib
import logging
import re
from pathlib import Path

from . import place
from .store import StoreError

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

PREFIX = "project_"
SOURCE = "files"
# A project's id, as the projects' store makes them (``projects._ID``): a contract ties the two.
PROJECT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{3}\Z")
# The inventory's label of the project indexes (store_inventory), which every line said here starts with.
LABEL = "project indexes (chunks of the project files)"


def corpus_of(project_id):
    """The corpus of a project's store; an id that is not a project's raises (``StoreError``)."""
    if not isinstance(project_id, str) or not PROJECT_ID.match(project_id):
        raise StoreError(f"{project_id!r} is not a project's id")
    return PREFIX + project_id


def project_of(corpus):
    """The project whose store ``corpus`` is, or None: a corpus named otherwise is an owner's."""
    if isinstance(corpus, str) and corpus.startswith(PREFIX) and PROJECT_ID.match(corpus[len(PREFIX):]):
        return corpus[len(PREFIX):]
    return None


def _original(name):
    """The name a project's file was added under: its stored name without the id the projects put before it
    (``<file id>_<name>``)."""
    head, sep, rest = name.partition("_")
    return rest if sep and rest and PROJECT_ID.match(head) else name


def _left(project_id, what):
    logger.warning("project %s: its Mycelium store: %s", project_id, what)
    return [f"{LABEL} (project {project_id}'s Mycelium store: {what})"]


def _folder_and_names(paths):
    """The folder of ``paths`` (the paths the rows store, ``<files folder>/<name>``) and their names; (None, set())
    when there is none or they lie in more than one folder."""
    folders, names = set(), set()
    for path in paths:
        path = Path(path)
        folders.add(path.parent)
        names.add(path.name)
    if len(folders) != 1:
        return None, set()
    return folders.pop(), names


def index(project_id, paths, vectors=True):
    """A pass over the project's files ``paths`` (the paths its rows store) into its store, then its vectors unless
    ``vectors`` is False; what was not done, named. No file the indexer reads (none, or only skipped kinds and
    secrets' names): the store is removed (``forget``)."""
    from . import vectors as encoder
    from .index import Indexer, kept

    try:
        corpus = corpus_of(project_id)
    except StoreError as exc:
        return _left(project_id, f"not indexed, {exc}")
    paths = list(paths)
    folder, names = _folder_and_names(paths)
    if paths and folder is None:
        return _left(project_id, "not indexed, its files lie in more than one folder")
    names = {name for name in names if kept(name) and kept(_original(name))}
    if not names:
        return forget(project_id)       # no file the indexer reads: the project has no store
    try:
        settings = encoder.load_settings()
    except encoder.VectorError as exc:
        return _left(project_id, f"not indexed, the Mycelium's settings are refused ({type(exc).__name__})")
    try:
        store = place.open_to_write(corpus)
    except Exception as exc:  # noqa: BLE001 - a key, a sealed file, a full disk: named, nothing indexed
        return _left(project_id, f"not indexed, it could not be opened ({type(exc).__name__})")
    left = []
    try:
        indexer = Indexer(store, header=settings.header)
        try:
            counts = indexer.index(SOURCE, folder, paths=sorted(names))
        except Exception as exc:  # noqa: BLE001 - nothing of a failed pass stays (its rollback)
            return _left(project_id, f"not indexed, nothing of the pass kept ({type(exc).__name__})")
        if counts.get("unreadable"):
            left += _left(project_id, f"{counts['unreadable']} file(s) not read, the next pass reads them again")
        left += _rewritten(store, project_id)
        if vectors and settings.enabled:
            left += _vectors(indexer, settings, project_id)
    finally:
        store.close()
    return left


def vectors(project_id, holding=contextlib.nullcontext):
    """The vector pass of the project's existing store (none is made), as the owner's route runs it; what was not
    done, named. Vectors held are left to a later pass. The store is opened under ``holding`` (the project's own lock,
    ``projects.holding_project``), so a deletion never meets the opening and makes it again; it is embedded outside
    it."""
    from . import vectors as encoder
    from .index import Indexer

    try:
        corpus = corpus_of(project_id)
        settings = encoder.load_settings()
    except (StoreError, encoder.VectorError) as exc:
        return _left(project_id, f"no vector pass ({type(exc).__name__})")
    if not settings.enabled:
        return []
    try:
        with holding():
            store = place.open_to_write(corpus, make=False)
    except place.NoStore:
        return []
    except Exception as exc:  # noqa: BLE001 - named
        return _left(project_id, f"no vector pass, it could not be opened ({type(exc).__name__})")
    try:
        return _vectors(Indexer(store, header=settings.header), settings, project_id)
    finally:
        store.close()


def _rewritten(store, project_id):
    """The store rewritten when a pass owes it (a file changed or left): its dropped words leave the pages."""
    try:
        owed = store.meta("rewrite")
    except Exception as exc:  # noqa: BLE001 - unread: rewritten anyway, no harm
        logger.warning("project %s: whether its Mycelium store owes a rewrite was not read (%s)", project_id,
                       type(exc).__name__)
        owed = True
    if owed and not store.rewrite():
        return _left(project_id, "not rewritten, the words of its removed files stay in its pages until it is")
    return []


def _vectors(indexer, settings, project_id):
    from . import vectors as encoder

    try:
        with place.one_vector_pass():
            counts = indexer.embed(encoder.Embedder(settings.encoder, settings.timeout_s), settings.batch)
    except place.AnotherVectorPass:
        return []
    except Exception as exc:  # noqa: BLE001 - what the pass wrote is kept; the next one takes up the rest
        return _left(project_id, f"its vector pass failed ({type(exc).__name__})")
    if counts["missing"]:
        return _left(project_id, "its vector pass stopped, the encoder is missing")
    return []


def drop(project_id, paths):
    """The files ``paths`` (as their rows store them) dropped from the project's store, the store rewritten; what
    was not done, named. A store that is not there holds none, and is not made."""
    try:
        corpus = corpus_of(project_id)
    except StoreError as exc:
        return _left(project_id, f"not reached, {exc}")
    folder, gone = _folder_and_names(paths)
    if folder is None:
        return []
    held = _held(corpus)
    if held is None:
        return _left(project_id, "not read, the files it holds are unknown")
    if not held & gone:
        return []
    keep = held - gone
    if not keep:
        return forget(project_id)
    return index(project_id, [folder / name for name in sorted(keep)], vectors=False)


def _held(corpus, owed=None):
    """The names of the files the store of ``corpus`` holds (an empty set when there is no store); None unread. With
    ``owed`` (a list), whether the store owes a rewrite is appended to it."""
    try:
        store = place.open_to_read(corpus)
    except place.NoStore:
        return set()
    except Exception:  # noqa: BLE001 - named by the caller
        return None
    try:
        if owed is not None:
            owed.append(bool(store.meta("rewrite")))
        return {row[0] for row in store.db.execute(
            "SELECT file.path FROM file JOIN source ON source.id = file.source WHERE source.name = ?", (SOURCE,))}
    except Exception:  # noqa: BLE001 - named by the caller
        return None
    finally:
        store.close()


def _rewrite(project_id):
    """A rewrite the project's store still owes, run (one that failed when its pass dropped words)."""
    try:
        store = place.open_to_write(corpus_of(project_id), make=False)
    except place.NoStore:
        return []
    except Exception as exc:  # noqa: BLE001 - named
        return _left(project_id, f"not rewritten, it could not be opened ({type(exc).__name__})")
    try:
        return _rewritten(store, project_id)
    finally:
        store.close()


def forget(project_id):
    """The project's store, and the files SQLite keeps beside it, overwritten and removed; what stayed, named."""
    try:
        corpus = corpus_of(project_id)
    except StoreError as exc:
        return _left(project_id, f"not reached, {exc}")
    try:
        place.remove_store(corpus)
    except OSError as exc:
        return _left(project_id, f"not removed ({type(exc).__name__})")
    return []


def sweep(live, claim=None):
    """What earlier deletions left in the projects' stores, removed: each project's store whose project ``live``
    does not hold (``project id -> the paths its rows store``), and from a live project's store the files its rows no
    longer name, and a rewrite a store still owes run; what stayed, named. A corpus that is not a project's
    (``project_of``) is never touched. ``claim(project id)``, when given, takes that project's own lock without
    waiting, as a context manager, or gives None when another thread holds it: that project is left to its pass
    running now, which drops what its rows do not name (265) -- never read nor written beside it."""
    left = []
    for corpus in place.corpora():
        project_id = project_of(corpus)
        if project_id is None:
            continue
        held_lock = contextlib.nullcontext() if claim is None else claim(project_id)
        if held_lock is None:
            continue
        with held_lock:
            left += _swept(corpus, project_id, live)
    return left


def _swept(corpus, project_id, live):
    """One project's store swept (see ``sweep``)."""
    if project_id not in live:
        return forget(project_id)
    owed = []
    held = _held(corpus, owed)
    if held is None:
        return _left(project_id, "not read, the files it holds are unknown")
    named = {Path(path).name for path in live[project_id]}
    if held - named:
        folder, _names = _folder_and_names(live[project_id])
        return (forget(project_id) if folder is None or not held & named
                else index(project_id, [folder / name for name in sorted(held & named)], vectors=False))
    if any(owed):
        return _rewrite(project_id)
    return []
