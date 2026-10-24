#!/usr/bin/env python3
"""Move ChromaDB's stores into the knowledge base's store, the platform stopped: all of them, or none.

ChromaDB kept the knowledge base's collections in folders of its own, in plaintext. ``migrate_all`` reads each of
them through chromadb itself -- the one reader that knows every version of its files -- and writes every collection
into the knowledge base's store (``vector_db``), encrypted under the platform's key where one loads, in ONE
transaction, proved before it commits: each item's document, metadata and vector bit for bit; and, on a sample of
each collection's vectors, the distances ChromaDB measures to its nearest found again by the store, in the space it
measured them in. Then each folder is moved whole into a folder of plaintext copies beside it, to delete once the
platform reads its knowledge base again. Until a folder has moved, its namespace of the store is closed
(``vector_db.MigrationPending``): the platform neither reads nor writes there meanwhile.

WHERE EACH COLLECTION GOES. A folder is known by its name: ``chroma_v2``, the knowledge base's collections, to the
namespace ``knowledge``; ``chroma_db``, the documents' index, to ``documents`` -- but the memory's collection,
``oo_memories``, which shared that folder, to ``memory``; ``project_chroma``, the projects' index, to ``projects``. A
ChromaDB store in another folder is said and left as it is. A collection keeps the space ChromaDB measured it in:
its configuration's, its metadata's, or ChromaDB's default, squared L2.

WHAT THE MEMORY STILL HOLDS. The memory's facts live in their own store, the vector layer mirrors them, and the
platform kept the facts while the vectors waited: a vector is copied only when an active fact of the memory holds
it with the very text the vector was made of; the others -- a fact deleted, or rewritten, since -- are left out,
and counted, and so are the active facts left with no vector, found by their words alone.

WHAT IS REFUSED, NOTHING WRITTEN AND NOTHING MOVED. Another run of this command or of ``encrypt`` under way; a
process that runs the platform, or holds the knowledge base's store or a ChromaDB file open, read again before the
commit; chromadb that does not import, or a store it does not read; a key file that does not load here; the
knowledge base's store in plaintext while a key loads (``encrypt`` first), or one that does not open; the memory's
facts that cannot be read; a folder of the platform's modules the platform would wait on and this command cannot
move -- a symbolic link, a folder that cannot be looked at, a ``RAG_HOME`` that is relative (``vector_db
.chroma_state``, the test the platform's wait reads); a collection whose space or dimension differs from the one of
that name the store holds with items in it; a ChromaDB collection that does not read whole, an item without its
vector, a value the store does not keep; a copy that does not prove. A store, and a folder, this run made are
removed again. An item whose id the store holds already is kept as it is, and counted. The command runs where the
platform runs: a ``RAG_HOME`` the platform has and this shell has not names a folder it never sees, which the
platform names when it waits. chromadb itself rewrites some of its own files as it reads them (its
index's lengths, its schema): what it keeps stays readable by it. A ChromaDB store behind a symbolic link is never
reached: ``check`` says the link, never followed.

A SIGNAL. SIGTERM, SIGHUP and SIGINT are noted and said; the run stops at its next safe point -- before a collection
is read, before the transaction commits -- and writes nothing. Once it has committed, it moves the folders whatever
comes: a run killed between the commit and the moves leaves the folders where they were, their namespaces closed,
and the next run finds every item kept and moves them. A folder that cannot be moved stops the run
(``EncryptionStuck``), each folder said where it is.
"""

from __future__ import annotations

import os
import time
from array import array
from pathlib import Path
from typing import Any

from opti_oignon import db_encryption as dbe
from opti_oignon import vector_db
from opti_oignon.db_utils import safe_connect

# The folders ChromaDB kept its collections in, by name, and the namespace their collections go to: the ones
# ``check`` names the command for.
SOURCES = dbe._CHROMA_FOLDERS
# A collection that goes to a namespace of its own, whatever folder kept it.
OWN = {"oo_memories": "memory"}
# The memory's facts, in the package's place of data: a vector is copied only when an active fact holds it.
FACTS = "memory_facts.db"
# How many items one read of ChromaDB returns.
_PAGE = 1000
# How many vectors of each collection are queried back, how many neighbours of each are compared, and how far
# apart two measures of one distance may be, relative to the distance, ChromaDB's float32 against this store's.
_SAMPLE = 8
_NEAREST = 5
_TOLERANCE = 1e-4


def target(places: Any = None) -> Path:
    """The knowledge base's store: in the store folder ``rag`` of the package's place of data."""
    return Path((dbe._places() if places is None else places)[-1]) / "rag" / vector_db.FILENAME


def sources(places: Any = None) -> tuple:
    """([(folder, namespace)] of the ChromaDB stores this command moves; [folder] of the ones it does not know), as
    ``plaintext_others`` finds them below the places: a folder of plaintext copies is passed over."""
    known, unknown = [], []
    for path in dbe.plaintext_others(places):
        folder = path.parent
        if path.name != dbe._CHROMA_FILE or dbe._in_a_kept_copy(folder, places):
            continue
        if folder.name in SOURCES:
            known.append((folder, SOURCES[folder.name]))
        else:
            unknown.append(folder)
    return known, unknown


def _alone(place: Path) -> Any:
    """Hold the lock ``encrypt`` holds, on ``place`` itself: one run of either at a time."""
    import fcntl

    try:
        fd = os.open(place, os.O_RDONLY)
    except OSError as exc:
        raise dbe.EncryptionRefused(f"the run's lock cannot be held on {place} ({type(exc).__name__}): nothing was "
                                    "written") from None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        why = ("another run of encrypt or migrate is under way" if isinstance(exc, BlockingIOError)
               else f"the run's lock cannot be held on {place} ({type(exc).__name__})")
        raise dbe.EncryptionRefused(f"{why}: nothing was written") from None
    return dbe._Held(fd)


def migrate_all(*, places: Any = None, proc: Any = None) -> dict:
    """Move every ChromaDB store below the places into the knowledge base's store, the platform stopped.

    Returns what was done: ``moved``, for each folder (its path, the folder
    of plaintext copies it was moved into, its namespace, its collections,
    the items written, the items kept, the items left out); ``unknown``,
    the ChromaDB stores left as they are; ``target``, the knowledge base's
    store, and whether it is ``encrypted``; the ``signal`` that came after
    the commit, if one did. Refused by name (``EncryptionRefused``),
    nothing written and nothing moved.
    """
    import signal

    found, unknown = sources(places)
    done = {"moved": [], "unknown": [str(folder) for folder in unknown], "target": str(target(places)),
            "encrypted": None, "signal": None}
    stuck = _unmovable(places, [folder for folder, _namespace in found])
    if stuck:
        raise dbe.EncryptionRefused("this command cannot move what the platform waits on: " + "; ".join(stuck)
                                    + ": nothing was written")
    if not found:
        return done
    lock = _alone(Path((dbe._places() if places is None else places)[0]))
    noted: list = []
    previous = {}
    notice = dbe._notice_fd()

    def note(signum, _frame):
        first = not noted
        noted.append(signum)
        if first and notice is not None:
            try:
                os.write(notice, f"signal {signum} noted: the run stops at its next safe point and writes nothing, "
                                 "or finishes if it has committed\n".encode())
            except OSError:
                pass

    try:
        for name in dbe._ENDING:
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                if signal.getsignal(signum) is signal.SIG_IGN:
                    continue
                previous[signum] = signal.signal(signum, note)
            except ValueError:
                pass
        return _migrate(found, done, places, proc, noted)
    finally:
        try:
            for signum, handler in previous.items():
                signal.signal(signum, signal.SIG_DFL if handler is None else handler)
        finally:
            try:
                if notice is not None:
                    os.close(notice)
            finally:
                lock.close()


def _migrate(found: list, done: dict, places: Any, proc: Any, noted: list) -> dict:
    store = Path(done["target"])
    watched = [store, *(folder / dbe._CHROMA_FILE for folder, _namespace in found)]

    def no_holder():
        held = dbe.store_holders(watched, proc=proc)
        if held:
            named = ", ".join(f"process {pid} {why}" for pid, why in sorted(held.items()))
            raise dbe.EncryptionRefused(f"stop the platform first ({named}): nothing was written")

    no_holder()
    key = _key()
    if key is not None and store.exists() and dbe._plain(store):
        raise dbe.EncryptionRefused(f"{store} is in plaintext while a key loads: run `{dbe.ENCRYPT_COMMAND}` first: "
                                    "nothing was written")
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as exc:  # noqa: BLE001 - whatever stops chromadb's import stops the read
        raise dbe.EncryptionRefused(f"chromadb does not import here ({type(exc).__name__}: {exc}): the ChromaDB "
                                    "stores cannot be read: nothing was written") from None
    facts: dict = {}
    batches, read = [], []
    for folder, namespace in found:
        names = []
        try:
            client = chromadb.PersistentClient(path=str(folder), settings=Settings(anonymized_telemetry=False))
            collections = [client.get_collection(entry) if isinstance(entry, str) else entry
                           for entry in client.list_collections()]
            for collection in collections:
                if noted:
                    raise dbe.EncryptionRefused(f"signal {noted[0]} noted before {collection.name} of {folder} was "
                                                "read: nothing was written")
                destination = OWN.get(collection.name, namespace)
                if destination == "memory" and "texts" not in facts:
                    facts["texts"] = _active_facts(places)
                batch = _read(collection, folder, destination, facts.get("texts"))
                _prove_distances(collection, batch)
                batches.append(batch)
                names.append(collection.name)
        except dbe.EncryptionRefused:
            raise
        except Exception as exc:  # noqa: BLE001 - a store chromadb does not read is refused, never half read
            raise dbe.EncryptionRefused(f"{folder} does not read through chromadb ({type(exc).__name__}: {exc}): "
                                        "nothing was written") from None
        read.append((folder, namespace, names))

    def before_commit():
        if noted:
            raise dbe.EncryptionRefused(f"signal {noted[0]} noted before the knowledge base's store committed: "
                                        "nothing was written")
        no_holder()

    existed, folder_existed = store.exists(), store.parent.exists()
    try:
        counts = vector_db.load(store, batches, before_commit=before_commit)
    except BaseException as exc:
        if not existed and store.exists() and not dbe.store_holders([store], proc=proc):
            for name in ("", "-wal", "-shm", "-journal"):
                try:
                    store.with_name(store.name + name).unlink()
                except FileNotFoundError:
                    pass
            if not folder_existed:
                try:
                    store.parent.rmdir()
                except OSError:
                    pass
        if isinstance(exc, (dbe.EncryptionRefused, KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, Exception):
            raise dbe.EncryptionRefused(f"{exc}" if isinstance(exc, ValueError) else
                                        f"the knowledge base's store at {store} took nothing "
                                        f"({type(exc).__name__}: {exc}): nothing was written") from None
        raise
    done["encrypted"] = key is not None
    if "texts" in facts:
        # Counted from what the store holds now, whoever wrote it. The store has committed and the folders move
        # whatever comes now: one that does not read back leaves the count unknown, said as such.
        try:
            got = vector_db.Client(store, "memory").get_collection("oo_memories").get(include=["documents"])
            held = dict(zip(got["ids"], got["documents"]))
            done["facts_without_vector"] = sum(1 for fact, text in facts["texts"].items() if held.get(fact) != text)
        except Exception as exc:
            done["facts_uncounted"] = f"{type(exc).__name__}: {exc}"
    # Committed: the folders move whatever comes now.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    kept_in: dict = {}
    at, each = 0, []
    for folder, namespace, names in read:
        each.append((folder, namespace, [
            {"name": batch["name"], "namespace": batch["namespace"], "written": count[0], "kept": count[1],
             "left_out": batch["left_out"], "removed": count[2]}
            for batch, count in zip(batches[at:at + len(names)], counts[at:at + len(names)])]))
        at += len(names)
    for number, (folder, namespace, collections) in enumerate(each):
        try:
            if folder.parent not in kept_in:
                kept_in[folder.parent] = dbe._fresh_folder(folder.parent, f"plaintext-{stamp}")
            os.rename(folder, kept_in[folder.parent] / folder.name)
        except (OSError, dbe.EncryptionRefused) as exc:
            moved = "; ".join(f"{entry['folder']} moved into {entry['kept_in']}" for entry in done["moved"])
            if folder.parent in kept_in:
                # A folder of plaintext copies made for this folder alone goes again: nothing was kept in it.
                try:
                    kept_in[folder.parent].rmdir()
                except OSError:
                    pass
            # What the store took from the folders left, said with the rest of what was done.
            done["unmoved"] = [{"folder": str(left), "namespace": space, "collections": entries}
                               for left, space, entries in each[number:]]
            done["signal"] = noted[0] if noted else None
            stopped = dbe.EncryptionStuck(
                f"the knowledge base's store holds every collection, but {folder} could not be moved into a folder of "
                f"plaintext copies ({type(exc).__name__}: {exc}): it stays where it is, its namespace closed, and the "
                f"next run of migrate moves it{'; ' + moved if moved else ''}")
            stopped.done = done
            raise stopped from None
        done["moved"].append({"folder": str(folder), "kept_in": str(kept_in[folder.parent]), "namespace": namespace,
                              "collections": collections,
                              "written": sum(entry["written"] for entry in collections),
                              "kept": sum(entry["kept"] for entry in collections),
                              "left_out": sum(entry["left_out"] for entry in collections)})
    for parent in kept_in:
        dbe._fsync_dir(parent)
    done["signal"] = noted[0] if noted else None
    return done


def _key() -> Any:
    """The databases' key that loads, or None; refused by name when a key file is here and does not load, or when
    SQLCipher is missing beside one."""
    from opti_oignon import encryption

    if dbe.SQLCIPHER_AVAILABLE:
        return dbe._loaded_key()
    if encryption.keyfile_present() or dbe._environment_key_refused() is not None or encryption._secret(
            encryption._ENV_KEY_NAME):
        raise dbe.EncryptionRefused("SQLCipher is not installed and a key is set: the knowledge base's store would be "
                                    "written in plaintext beside encrypted stores: nothing was written")
    return None


def _active_facts(places: Any) -> dict:
    """{id: text} of the memory's active facts, read from its store; none when it holds none. Refused by name when
    it is there and does not read."""
    path = Path((dbe._places() if places is None else places)[-1]) / FACTS
    if not path.exists():
        return {}
    try:
        conn = safe_connect(path, read_only=True)
        try:
            return dict(conn.execute("SELECT id, text FROM memory_facts WHERE active = 1").fetchall())
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - facts that cannot be read cannot say which vectors are theirs
        hint = (f"; it is in plaintext while a key loads: run `{dbe.ENCRYPT_COMMAND}` first"
                if dbe._plain(path) and dbe._get_db_encryption_key() is not None else "")
        raise dbe.EncryptionRefused(f"the memory's facts at {path} do not read here ({type(exc).__name__}: {exc})"
                                    f"{hint}: which of its vectors are still the memory's cannot be told: nothing "
                                    "was written") from None


def _unmovable(places: Any, found: list) -> list:
    """Each folder of the platform's modules the platform would wait on (``vector_db.chroma_state``) and this run
    would not move, with why: a symbolic link; a folder that cannot be looked at; ChromaDB's store, which the walk of
    the places did not reach (a folder it could not list); or the documents' and the memory's folder below a
    ``RAG_HOME`` that is relative -- empty included, which the platform reads as its own directory -- and that this
    command cannot place."""
    package = Path((dbe._places() if places is None else places)[-1])
    home = os.environ.get("RAG_HOME")
    if home is not None and not Path(home).is_absolute():
        stuck = [f"the documents' index and the memory's folder below RAG_HOME, which is relative ({home!r}): this "
                 "command cannot look below it; set it to the absolute path the platform reads"]
        folders = [package / "rag" / "chroma_v2", package / "project_chroma"]
    else:
        stuck = []
        folders = [package / "rag" / "chroma_v2", (Path(home) if home else package / "rag") / "chroma_db",
                   package / "project_chroma"]
    for folder in folders:
        state, detail = vector_db.chroma_state(folder)
        what = "file" if Path(detail).name == vector_db.CHROMA_FILE else "folder"
        if state == "link":
            stuck.append(f"{folder} (a symbolic link, {detail}, never followed: put the {what} it leads to in its "
                         "place)")
        elif state == "unreadable":
            stuck.append(f"{folder} (it cannot be looked at, {detail})")
        elif state == "store" and os.path.realpath(folder) not in {os.path.realpath(path) for path in found}:
            stuck.append(f"{folder} (ChromaDB's store, which the walk of the places did not reach: a folder above it "
                         "cannot be listed)")
    return stuck


def _space(collection: Any) -> str:
    """The space ChromaDB measures ``collection`` in: its configuration's, its metadata's, or ChromaDB's default,
    squared L2."""
    configuration = getattr(collection, "configuration_json", None) or {}
    for index in ("hnsw", "spann"):
        part = configuration.get(index) if isinstance(configuration, dict) else None
        if isinstance(part, dict) and part.get("space"):
            return str(part["space"])
    metadata = collection.metadata or {}
    if metadata.get(vector_db.SPACE_KEY):
        return str(metadata[vector_db.SPACE_KEY])
    return "l2"


def _read(collection: Any, folder: Path, namespace: str, facts: Any) -> dict:
    """Every item of ``collection`` read whole, as a batch of ``vector_db.load``; refused by name otherwise. In the
    memory's namespace an item is kept only when an active fact holds it with the text it was made of; the others
    are left out, and counted."""
    space = _space(collection)
    if space not in vector_db.SPACES:
        raise dbe.EncryptionRefused(f"{collection.name} of {folder} is measured in {space}, which the knowledge base's "
                                    "store does not measure: nothing was written")
    metadata = dict(collection.metadata or {})
    metadata[vector_db.SPACE_KEY] = space
    total = collection.count()
    items: list = []
    seen = 0
    for offset in range(0, total, _PAGE):
        page = collection.get(include=["documents", "metadatas", "embeddings"], limit=_PAGE, offset=offset)
        ids = list(page["ids"])
        documents = list(page["documents"]) if page.get("documents") is not None else [None] * len(ids)
        metadatas = list(page["metadatas"]) if page.get("metadatas") is not None else [None] * len(ids)
        embeddings = page.get("embeddings")
        embeddings = [None] * len(ids) if embeddings is None else list(embeddings)
        for item_id, document, item_metadata, embedding in zip(ids, documents, metadatas, embeddings):
            seen += 1
            if embedding is None:
                raise dbe.EncryptionRefused(f"{item_id} of {collection.name} in {folder} has no vector: nothing was "
                                            "written")
            values = [float(x) for x in embedding]
            vector = array("f", values)
            if vector.tolist() != values:
                raise dbe.EncryptionRefused(f"{item_id} of {collection.name} in {folder} has a vector float32 does not "
                                            "hold exactly: nothing was written")
            items.append((str(item_id), document, None if item_metadata is None else dict(item_metadata), vector))
    if seen != total or len({item[0] for item in items}) != total:
        raise dbe.EncryptionRefused(f"{collection.name} of {folder} does not read whole ({seen} items read, "
                                    f"{total} counted): nothing was written")
    kept = (items if namespace != "memory" or facts is None else
            [item for item in items if item[0] in facts and facts[item[0]] == item[1]])
    # Each item of the memory names the text its fact reads now, or None: the store keeps no other copy of it, which
    # an earlier run that committed and could not move its folder made, from whichever folder.
    return {"namespace": namespace, "name": collection.name, "metadata": metadata, "source": str(folder),
            "items": kept, "space": space, "read": items, "left_out": len(items) - len(kept),
            "drop": ({} if namespace != "memory" or facts is None else
                     {item[0]: facts.get(item[0]) for item in items})}


def _prove_distances(collection: Any, batch: dict) -> None:
    """For a sample of the collection's vectors, the distance ChromaDB measures to each of its nearest is the one
    this store measures between the same two vectors, in the space the batch is written in: the space read, and the
    store's measure of it, proved; refused by name otherwise."""
    items = batch["read"]
    if not items:
        return
    vectors = [item[3] for item in items]
    index = {item[0]: at for at, item in enumerate(items)}
    nearest = vector_db.exact_searcher(batch["space"], vectors)
    n = min(_NEAREST, len(vectors))
    for at in sorted(range(len(vectors)), key=lambda i: items[i][0])[:_SAMPLE]:
        theirs = collection.query(query_embeddings=[list(vectors[at])], n_results=n, include=["distances"])
        ours = dict(nearest(vectors[at], len(vectors)))
        for item_id, distance in zip(theirs["ids"][0], theirs["distances"][0]):
            if str(item_id) not in index:
                raise dbe.EncryptionRefused(f"{collection.name} of {batch['source']}: ChromaDB returns {item_id}, "
                                            "which it did not list: nothing was written")
            distance, mine = float(distance), ours[index[str(item_id)]]
            if abs(mine - distance) > _TOLERANCE * max(1.0, abs(mine), abs(distance)):
                raise dbe.EncryptionRefused(f"{collection.name} of {batch['source']}: ChromaDB measures {distance} "
                                            f"where the knowledge base's store measures {mine} in {batch['space']}: "
                                            "nothing was written")
