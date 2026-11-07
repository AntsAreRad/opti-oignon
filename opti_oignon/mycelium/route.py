"""The owner's route to the Mycelium: ``python -m opti_oignon mycelium index|forget|status`` (``main.cmd_mycelium``).

``index CORPUS FOLDER`` indexes a folder into the corpus's store (``place.open_to_write``) -- never a project's store,
``project_`` and a project's id, which its project keeps (``projects``) --, as the source named as the
corpus unless ``--source`` names another (by the corpus's rule), private unless ``--sensitivity`` says otherwise. A
name outside the rule, a folder that is missing, not a folder, that this process cannot list, or that is the
package's data folder or inside it (its real path: a link into it is refused too), and settings (``mycelium.yaml``)
that are refused -- they name the header the parts are embedded with --, refuse the pass by name before any store is
opened. A store that owes a rewrite -- the pass changed, renamed or removed a file, or an earlier one did and its
rewrite never ran to its end -- is rewritten
after the pass (``Store.rewrite``: the dropped words leave its bytes); then comes the vector pass, unless
``--no-vectors`` or the settings turn the vectors off: one at a time on the install (``place.one_vector_pass``), every
request admitted by the resource governor as background work (``vectors``). ``forget CORPUS SOURCE`` forgets a source
of an existing store and rewrites it. ``status [CORPUS]`` reads each store read-only (``place.open_to_read``):
nothing written, nothing made -- but beside a store another program left in WAL, where SQLite itself makes its -wal
and -shm on reading, until a pass of the route brings the store back to the rollback journal.

The exit code is 0 when what was asked was done, and 1 when it was refused or not done in full: a file or a folder
the pass could not read, words of a dropped file or a forgotten source left in the store's pages, a pass under
settings that are refused, the encoder missing, a store that cannot be read (the status says refused settings, and
exits 0 when it said every store). Vectors held -- by the governor, a busy engine, another pass writing
the store, another vector pass -- are background work left to the next pass (0). What is said is counts and names,
never a source's words."""
import logging
import os
from pathlib import Path

from . import place, projects
from .store import SENSITIVITIES, StoreError

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

logger = logging.getLogger(__name__)


def _say(line):
    print(line, flush=True)


def _refused(message):
    _say(f"[ERR] {message}")
    return 1


def _counted(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _owners(corpus):
    """``corpus`` when it is the owner's to write: a name the rule admits and no project's store (``project_`` and a
    project's id: its project keeps it); any other refused by name."""
    place.path_for(corpus)
    if projects.project_of(corpus) is not None:
        raise StoreError(f"{corpus!r} is a project's store: its project keeps it, the route never writes it")
    return corpus


def _source_name(name):
    """A source of the route is named as a corpus is (``place.CORPUS``); any other name refused by name."""
    if not isinstance(name, str) or not place.CORPUS.match(name):
        raise StoreError(f"a source is named by lowercase letters, digits, '_' and '-', from a letter or a digit, at "
                         f"most 100 of them, not {name!r}")
    return name


def _indexable(folder):
    """The folder to index, refused by name when it is missing, not a folder, or the package's data folder or inside
    it (the stores and the user's data are never indexed)."""
    root = Path(folder)
    if not root.exists():
        raise StoreError(f"{folder} does not exist: nothing indexed")
    if not root.is_dir():
        raise StoreError(f"{folder} is not a folder: nothing indexed")
    if not os.access(root, os.R_OK | os.X_OK):
        raise StoreError(f"{folder} cannot be listed by this process: nothing indexed")
    data = os.path.realpath(place.folder().parent)
    real = os.path.realpath(root)
    if real == data or real.startswith(data.rstrip(os.sep) + os.sep):
        raise StoreError(f"{folder} is the package's data folder or lies inside it: never indexed")
    return root


def _settings():
    """(the vectors' settings, None), or (None, why they are refused)."""
    from . import vectors

    try:
        return vectors.load_settings(), None
    except vectors.VectorError as exc:
        return None, exc


def run(args):
    """The action ``main`` parsed, run; its exit code."""
    action = getattr(args, "mycelium_action", None)
    if action == "index":
        return index(args.corpus, args.folder, args.source, args.sensitivity, not args.no_vectors)
    if action == "forget":
        return forget(args.corpus, args.source)
    if action == "status":
        return status(args.corpus)
    raise ValueError(f"no Mycelium action {action!r}")


def index(corpus, folder, source=None, sensitivity=None, vectors=True):
    """A pass of the route over ``folder`` into ``corpus``'s store (see the module's doc); its exit code."""
    from .index import Indexer

    try:
        _owners(corpus)
        name = _source_name(corpus if source is None else source)
        if sensitivity is not None and sensitivity not in SENSITIVITIES:
            raise StoreError(f"sensitivity is one of {', '.join(SENSITIVITIES)}, not {sensitivity!r}")
        root = _indexable(folder)
    except StoreError as exc:
        return _refused(str(exc))
    settings, unset = _settings()
    if unset is not None:
        return _refused(f"the Mycelium's settings are refused, nothing indexed: {unset}")
    try:
        store = place.open_to_write(corpus)
    except StoreError as exc:
        return _refused(str(exc))
    except Exception as exc:  # noqa: BLE001 - a key, a sealed file, a full disk: said by name, nothing indexed
        return _refused(f"the store of {corpus!r} cannot be opened to be written: {exc}")
    try:
        indexer = Indexer(store, header=settings.header)
        try:
            counts = indexer.index(name, root, sensitivity=sensitivity)
        except StoreError as exc:
            return _refused(str(exc))
        except Exception as exc:  # noqa: BLE001 - nothing of a failed pass stays (its rollback): said by name
            return _refused(f"the pass over {root} failed, nothing of it kept: {type(exc).__name__}: {exc}")
        _say(f"[OK] mycelium: {name} indexed into {corpus}: {counts['added']} added, {counts['changed']} changed, "
             f"{counts['renamed']} renamed, {counts['removed']} removed, {counts['unchanged']} unchanged, "
             f"{_counted(counts['parts'], 'part')} written ({counts['seconds']} s)")
        code = 0
        if counts.get("unreadable"):
            _say(f"[!] {_counted(counts['unreadable'], 'file')} could not be read: the next pass reads them again")
            code = 1
        if counts.get("blind"):
            _say(f"[!] {_counted(counts['blind'], 'folder')} could not be listed: what the store held under them stays "
                 f"as it was, and a file new there waits for a pass that can list it")
            code = 1
        try:
            owed = store.meta("rewrite")
        except Exception as exc:  # noqa: BLE001 - another connection holding the store past the wait: said
            _say(f"[!] whether the store owes a rewrite could not be read ({type(exc).__name__}): the next pass of the "
                 f"route reads it again")
            owed, code = None, 1
        if owed:
            if store.rewrite():
                _say("[OK] the store was merged and rewritten: the dropped words left its pages")
            else:
                _say("[!] the store was not rewritten: dropped or forgotten words stay in its pages until a pass of "
                     "the route rewrites it")
                code = 1
        if vectors:
            code = max(code, _vectors(indexer, settings))
        return code
    finally:
        store.close()


def _vectors(indexer, settings):
    """The vector pass after the route's index pass; its exit code."""
    from . import vectors

    if not settings.enabled:
        _say("[>] the vectors are off (mycelium.yaml): none asked")
        return 0
    try:
        with place.one_vector_pass():
            counts = indexer.embed(vectors.Embedder(settings.encoder, settings.timeout_s), settings.batch)
    except place.AnotherVectorPass as exc:
        _say(f"[!] {exc}")
        return 0
    except StoreError as exc:
        _say(f"[!] {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - what the pass wrote is kept; the next one takes up the rest
        _say(f"[!] the vector pass failed: {type(exc).__name__}: {exc}")
        return 1
    if counts["missing"]:
        _say(f"[!] the vector pass stopped: {indexer.last_stop} ({counts['embedded']} embedded before)")
        return 1
    if counts["held"]:
        _say(f"[!] the vector pass was held: {indexer.last_stop} ({counts['embedded']} embedded; the next pass takes "
             f"up the rest)")
        return 0
    _say(f"[OK] vectors under {settings.encoder.key}: {counts['embedded']} embedded, {counts['refused']} refused by "
         f"the engine, {_counted(counts.get('uncovered', 0), 'text')} left without one"
         + ("; now the encoder read" if counts.get("activated") else ""))
    return 0


def forget(corpus, source):
    """A source of ``corpus``'s existing store forgotten, the store rewritten; its exit code."""
    from .index import Indexer

    try:
        store = place.open_to_write(_owners(corpus), make=False)
    except StoreError as exc:
        return _refused(str(exc))
    except Exception as exc:  # noqa: BLE001 - a key, a sealed file: said by name, nothing forgotten
        return _refused(f"the store of {corpus!r} cannot be opened to be written: {exc}")
    try:
        try:
            counts = Indexer(store).forget(source)
        except StoreError as exc:
            return _refused(str(exc))
        except Exception as exc:  # noqa: BLE001 - a reader holding the store at the commit: nothing forgotten, said
            return _refused(f"{source} was not forgotten, nothing changed: {type(exc).__name__}: {exc}")
        _say(f"[OK] mycelium: forgot {source} in {corpus}: {_counted(counts['files'], 'file')}, "
             f"{_counted(counts['parts'], 'part')}, {_counted(counts['vectors'], 'vector')}")
        if counts["merged"]:
            _say("[OK] the store was merged and rewritten: the forgotten words left its pages")
            return 0
        _say("[!] the store was not rewritten: the forgotten words stay in its pages until a pass of the route "
             "rewrites it")
        return 1
    finally:
        store.close()


def status(corpus=None):
    """What each store holds -- or ``corpus``'s alone --, read-only; its exit code."""
    if corpus is not None:
        try:
            place.path_for(corpus)
        except StoreError as exc:
            return _refused(str(exc))
    names = [corpus] if corpus is not None else place.corpora()
    if not names:
        _say(f"[>] mycelium: no corpus is indexed ({place.folder()})")
        return 0
    settings, unset = _settings()
    code = 0
    for name in names:
        try:
            store = place.open_to_read(name)
        except Exception as exc:  # noqa: BLE001 - a store that cannot be read is said, the others still are
            _say(f"[ERR] corpus {name}: {exc}")
            code = 1
            continue
        try:
            _said(name, store, settings, unset)
        except Exception as exc:  # noqa: BLE001 - a store a writer holds now: said, the others still are
            _say(f"[ERR] corpus {name} cannot be read now: {type(exc).__name__}: {exc}")
            code = 1
        finally:
            store.close()
    return code


def _said(name, store, settings, unset):
    from .index import Indexer

    path = place.path_for(name)
    _say(f"corpus {name}: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    indexer = Indexer(store, header=settings.header if settings else "full")
    db = store.db
    for source in store.sources():
        files = db.execute("SELECT COUNT(*) FROM file WHERE source = ?", (source["id"],)).fetchone()[0]
        parts = db.execute("SELECT COUNT(*) FROM part p JOIN file f ON f.id = p.file WHERE f.source = ?",
                           (source["id"],)).fetchone()[0]
        last = next(iter(store.episodes(source["id"], 1)), None)
        _say(f"  source {source['name']}: {source['kind']} {source['root']}, {source['sensitivity']}, "
             f"{_counted(files, 'file')}, {_counted(parts, 'part')}, "
             + (f"last pass {last['at']}" if last else "never indexed"))
        if source["kind"] != "folder":
            continue
        try:
            now = indexer.status(source["name"], source["root"])
        except Exception as exc:  # noqa: BLE001 - a folder moved or unreadable: said, the rest still is
            _say(f"    a pass now: its folder cannot be read ({type(exc).__name__})")
            continue
        about = "" if now["estimate_seconds"] is None else f" (about {now['estimate_seconds']} s)"
        _say(f"    a pass now: {now['new']} new, {now['touched']} touched, {now['gone']} gone{about}")
    if unset is not None:
        _say(f"  vectors: the settings are refused: {unset}")
    else:
        held = indexer.vector_status(settings.encoder)
        read = "none read yet" if held["active"] is None else f"read under {held['active']}"
        _say(f"  vectors: {read}; {held['with_vector']} of {_counted(held['texts'], 'text')} have one under "
             f"{held['encoder']}" + ("; the last vector pass was held" if held["held"] else "")
             + ("; the last vector pass found no encoder" if held["missing"] else ""))
    if store.meta("rewrite"):
        _say("  the store's rewrite is owed: words a pass or a forgetting dropped may stay in its pages until a pass of "
             "the route rewrites it")
