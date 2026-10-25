#!/usr/bin/env python3
"""Move ChromaDB's stores into the knowledge base's store, the platform stopped: all of them, or none.

ChromaDB kept the knowledge base's collections in folders of its own, in plaintext. ``migrate_all`` reads each
folder's database, ``chroma.sqlite3``, with the standard library alone and read-only -- chromadb is never imported
--: every collection's name, metadata and space, every item's id, text and metadata, in the order ChromaDB keeps
them. The vectors are not there to be trusted (ChromaDB keeps them in index files of its own, which this command
never opens): each text is embedded again by the embedder the module that queries its namespace embeds with, with
the model those queries use -- the configured one under the tag the server lists, never the faster one or another
it finds --, stored as that module stores it, and a sample of each collection is embedded once more, alone, which
proves each of those texts paired with its vector. Then every collection is written into the knowledge base's store
(``vector_db``), encrypted under the platform's key where one loads, in ONE transaction, each item read back before
it commits. Then each folder is moved whole into a folder of plaintext copies beside it, to delete once the platform
reads its knowledge base again. Until a folder has moved, its namespace of the store is closed
(``vector_db.MigrationPending``): the platform neither reads nor writes there meanwhile.

WHAT IS READ. A database whose ``migrations`` rows are the ones ChromaDB 1.3.7 records, or the first rows of each of
its folders of migrations, as an earlier version left them; one database of collections; one metadata segment for
each collection; each value of a metadata in exactly one of its columns; no write ChromaDB recorded and never
applied. Any other is refused by name. A collection keeps the space chromadb 1.3.7 measures it in, read as 1.3.7
reads the file: the space its schema records, or, before the schema, its configuration under ``vector_index``, when
that is not the default; else the one its index's segment records; else squared L2. Before the schema, a collection
whose metadata says another space is refused by name. An item without text is counted and said, and stays in the
folder of plaintext copies alone: with no text there is no vector to compute. The texts reach the embedders' server
through the inference registry, as the platform's own texts do when it indexes them.

WHERE EACH COLLECTION GOES. A folder is known by its name: ``chroma_v2``, the knowledge base's collections, to the
namespace ``knowledge``; ``chroma_db``, the documents' index, to ``documents`` -- but the memory's collection,
``oo_memories``, which shared that folder, to ``memory``; ``project_chroma``, the projects' index, to ``projects``. A
ChromaDB store in another folder is said and left as it is.

WHAT THE MEMORY STILL HOLDS. The memory's facts live in their own store, the vector layer mirrors them, and the
platform kept the facts while the vectors waited: a vector is copied only when an active fact of the memory holds
it with the very text the vector was made of; the others -- a fact deleted, or rewritten, since -- are left out,
and counted, and so are the active facts left with no vector, found by their words alone.

WHAT IS REFUSED, NOTHING WRITTEN AND NOTHING MOVED. Another run of this command or of ``encrypt`` under way; a
process that runs the platform, or holds the knowledge base's store or a ChromaDB file open, read again before the
commit; a ChromaDB database this command does not read, or one whose write-ahead log holds writes with no index of
it beside; a key file that does not load here; the knowledge base's store in plaintext while a key loads
(``encrypt`` first), or one that does not open; the memory's facts that cannot be read; an embedder whose model is
not served here, or would be replaced by another, one that gives no vector for a text, vectors of two dimensions in
one collection, a text whose vector its second embedding does not find again; a folder of the platform's modules the
platform would wait on and this command cannot move -- a symbolic link, a folder that cannot be looked at, a
``RAG_HOME`` that is relative (``vector_db.chroma_state``, the test the platform's wait reads); a collection whose
space or dimension differs from the one of that name the store holds with items in it; a copy that does not read
back. A store, and a folder, this run made are removed again. An item whose id the store holds already is kept as it
is, and counted. The command runs where the platform runs: a ``RAG_HOME`` the platform has and this shell has not
names a folder it never sees, which the platform names when it waits. No byte of ChromaDB's files is written: each
database is opened read-only, its shared index read-only too -- or, in write-ahead-log mode with no index beside it,
as immutable --, and no file is made beside it. A ChromaDB store behind a symbolic link is never reached: ``check``
says the link, never followed.

A SIGNAL. SIGTERM, SIGHUP and SIGINT are noted and said; the run stops at its next safe point -- before a store is
read, between two requests for vectors, before the transaction commits -- and writes nothing. Once it has committed,
it moves the folders whatever comes: a run killed between the commit and the moves leaves the folders where they
were, their namespaces closed, and the next run finds every item kept and moves them. A folder that cannot be moved
stops the run (``EncryptionStuck``), each folder said where it is.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
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
# The rows ChromaDB 1.3.7 records in its database's ``migrations``, folder by folder: a database holding them, or
# the first of them in each folder, as an earlier version left it, is read; any other row is refused by name.
_KNOWN = {
    "embeddings_queue": ("00001-embeddings.sqlite.sql", "00002-embeddings-queue-config.sqlite.sql"),
    "metadb": ("00001-embedding-metadata.sqlite.sql", "00002-embedding-metadata.sqlite.sql",
               "00003-full-text-tokenize.sqlite.sql", "00004-metadata-indices.sqlite.sql",
               "00005-max-seq-id-int.sqlite.sql"),
    "sysdb": ("00001-collections.sqlite.sql", "00002-segments.sqlite.sql", "00003-collection-dimension.sqlite.sql",
              "00004-tenants-databases.sqlite.sql", "00005-remove-topic.sqlite.sql",
              "00006-collection-segment-metadata.sqlite.sql", "00007-collection-config.sqlite.sql",
              "00008-maintenance-log.sqlite.sql", "00009-segment-collection-not-null.sqlite.sql",
              "00010-collection-schema.sqlite.sql"),
}
# The tables a ChromaDB database of collections holds whatever its version.
_TABLES = ("migrations", "collections", "collection_metadata", "segments", "segment_metadata", "embeddings",
           "embedding_metadata")
# The metadata key ChromaDB keeps an item's text under, and the prefix of the keys it keeps for itself.
_DOCUMENT = "chroma:document"
_OWN_KEYS = "chroma:"
# How many texts one request for vectors carries: a signal is looked at between two requests.
_REQUEST = 64
# How many items of each collection are embedded again, alone, beside its first, its last and the last of each
# request; and how far apart one text's two vectors may lie, relative to the longer, and still be taken for the same
# text's: a vector of another text lies much farther.
_SAMPLE = 8
_PAIRED = 1e-2


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


def migrate_all(*, places: Any = None, proc: Any = None, say: Any = None) -> dict:
    """Move every ChromaDB store below the places into the knowledge base's store, the platform stopped.

    Returns what was done: ``moved``, for each folder (its path, the folder
    of plaintext copies it was moved into, its namespace, its collections,
    the items written, the items kept, the items left out, the items
    without text, the model and the dimension of their vectors);
    ``unknown``, the ChromaDB stores left as they are; ``target``, the
    knowledge base's store, and whether it is ``encrypted``; the ``signal``
    that came after the commit, if one did. ``say(line)``, when given, is
    told before the vectors are computed how many texts each model embeds.
    Refused by name (``EncryptionRefused``), nothing written and nothing
    moved.
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
        return _migrate(found, done, places, proc, noted, say)
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


def _migrate(found: list, done: dict, places: Any, proc: Any, noted: list, say: Any) -> dict:
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
    facts: dict = {}
    batches, read = [], []
    for folder, namespace in found:
        if noted:
            raise dbe.EncryptionRefused(f"signal {noted[0]} noted before {folder} was read: nothing was written")
        names = []
        for collection in _read_store(folder):
            destination = OWN.get(collection["name"], namespace)
            if destination == "memory" and "texts" not in facts:
                facts["texts"] = _active_facts(places)
            batches.append(_batch(collection, folder, destination, facts.get("texts")))
            names.append(collection["name"])
        read.append((folder, namespace, names))
    # Every store read, before a single vector is asked for: each namespace's embedder answers with its own model.
    embedders = {name: _embedder(name) for name in sorted({batch["namespace"] for batch in batches if batch["items"]})}
    _announce(batches, embedders, say)
    for batch in batches:
        if batch["items"]:
            _embed(batch, embedders[batch["namespace"]], noted)
            _prove_pairing(batch, embedders[batch["namespace"]])

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
             "left_out": batch["left_out"], "removed": count[2], "without_text": batch["without_text"],
             "model": batch.get("model"), "dimension": batch.get("dimension")}
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
                              "left_out": sum(entry["left_out"] for entry in collections),
                              "without_text": sum(entry["without_text"] for entry in collections)})
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


def _read_store(folder: Path) -> list:
    """[{name, metadata, space, items}] of the ChromaDB database in ``folder``, each item (id, text or None, metadata
    or None) in the order ChromaDB keeps them: read with the standard library, read-only, its shared index read-only
    too, in one read transaction, the schema untrusted; refused by name when it is not a database this command
    knows, nothing written."""
    path = folder / dbe._CHROMA_FILE
    log, index = Path(f"{path}-wal"), Path(f"{path}-shm")
    if log.exists() and log.stat().st_size and not index.exists():
        raise dbe.EncryptionRefused(
            f"{path} has a write-ahead log and no index of it beside: read-only, the log would go unread; fold it "
            f"into the database first, the platform stopped (`sqlite3 {path} 'PRAGMA wal_checkpoint(TRUNCATE)'`), "
            "then run migrate again: nothing was written")
    try:
        conn = sqlite3.connect(_reader_uri(path, log), uri=True)
    except (sqlite3.Error, OSError) as exc:
        raise dbe.EncryptionRefused(f"{path} does not open ({type(exc).__name__}: {exc}): nothing was "
                                    "written") from None
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA trusted_schema = OFF")
        if hasattr(conn, "setconfig") and hasattr(sqlite3, "SQLITE_DBCONFIG_DEFENSIVE"):
            conn.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
        conn.execute("BEGIN")
        return _collections(conn, path)
    except dbe.EncryptionRefused:
        raise
    except Exception as exc:  # noqa: BLE001 - a database this reader does not read is refused, never half read
        raise dbe.EncryptionRefused(f"{path} does not read as a ChromaDB database ({type(exc).__name__}: {exc}): "
                                    "nothing was written") from None
    finally:
        conn.close()


def _reader_uri(path: Path, log: Path) -> str:
    """The URI that opens ChromaDB's database and writes no byte beside it: read-only, its shared index read-only
    when there is one (``db_encryption._read_only_uri``); a database in write-ahead-log mode whose log is absent or
    empty -- nothing to read in it -- opened as immutable, for read-only alone would make the log, and the index when
    there is none, and leave them behind."""
    uri = dbe._read_only_uri(str(path))
    with open(path, "rb") as handle:
        header = handle.read(20)
    if len(header) == 20 and 2 in (header[18], header[19]) and (not log.exists() or not log.stat().st_size):
        uri += "&immutable=1"
    return uri


def _collections(conn: Any, path: Path) -> list:
    """What ``_read_store`` returns, from ``conn``, in its read transaction."""

    def refuse(why):
        raise dbe.EncryptionRefused(f"{path}: {why}: nothing was written")

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    missing = [table for table in (*_TABLES, "embeddings_queue", "max_seq_id") if table not in tables]
    if missing:
        refuse(f"no table {', '.join(missing)}: not a ChromaDB database this command knows")
    recorded: dict = {}
    for directory, version, filename in conn.execute("SELECT dir, version, filename FROM migrations"):
        recorded.setdefault(directory, {})[version] = filename
    for directory, versions in sorted(recorded.items(), key=lambda pair: str(pair[0])):
        known = _KNOWN.get(directory, ())
        expected = {at + 1: name for at, name in enumerate(known[:len(versions)])}
        if versions != expected:
            odd = sorted((str(version), str(name)) for version, name in versions.items()
                         if expected.get(version) != name)
            refuse(f"written by a ChromaDB this command does not know: migration {directory} {odd[0][0]} {odd[0][1]}")

    def columns(table):
        return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}

    held = columns("collections")
    if "database_id" in held:
        databases = conn.execute("SELECT COUNT(DISTINCT database_id) FROM collections").fetchone()[0]
        if databases > 1:
            refuse(f"its collections lie in {databases} databases: this command reads a store of one")
    configured = "config_json_str" if "config_json_str" in held else "NULL"
    schemas = "schema_str" if "schema_str" in held else "NULL"
    rows = conn.execute(f"SELECT id, name, {configured}, {schemas} FROM collections ORDER BY rowid").fetchall()
    names = [row[1] for row in rows]
    if len(set(names)) != len(names):
        refuse("two of its collections share a name")
    values = "str_value, int_value, float_value" + (", bool_value" if "bool_value" in columns("collection_metadata")
                                                     else "")
    item_values = "m.string_value, m.int_value, m.float_value" + (
        ", m.bool_value" if "bool_value" in columns("embedding_metadata") else "")
    segments: dict = {}
    indexes: dict = {}
    for segment, scope, owner in conn.execute("SELECT id, scope, collection FROM segments"):
        if scope == "METADATA":
            segments.setdefault(owner, []).append(segment)
        elif scope == "VECTOR":
            indexes.setdefault(owner, []).append(segment)
    segment_values = "str_value, int_value, float_value" + (
        ", bool_value" if "bool_value" in columns("segment_metadata") else "")
    topics = [row[0] for row in conn.execute("SELECT DISTINCT topic FROM embeddings_queue")]
    found = []
    for collection_id, name, configuration, schema in rows:
        metadata = _values(conn.execute(f"SELECT key, {values} FROM collection_metadata WHERE collection_id = ?",
                                        (collection_id,)), f"the metadata of {name}", path)
        own = segments.get(collection_id, [])
        if len(own) != 1:
            refuse(f"{name} has {len(own)} metadata segments: this command reads a collection of one")
        _applied(conn, path, name, collection_id, own[0], topics)
        # The space its index's segment records, as ChromaDB before its configuration wrote it.
        recorded = {str(spaces[vector_db.SPACE_KEY]) for spaces in (
            _values(conn.execute(f"SELECT key, {segment_values} FROM segment_metadata WHERE segment_id = ?",
                                 (index,)), f"the index's metadata of {name}", path)
            for index in indexes.get(collection_id, ())) if spaces.get(vector_db.SPACE_KEY)}
        if len(recorded) > 1:
            refuse(f"the index segments of {name} record {len(recorded)} spaces: this command reads one")
        space = _space_of(schema, configuration, next(iter(recorded), None), metadata, name, path)
        # A collection without metadata has none, as chromadb gives it back.
        found.append({"name": name, "metadata": metadata or None, "space": space,
                      "items": _items(conn, path, name, own[0], item_values)})
    return found


def _values(rows: Any, what: str, path: Path) -> dict:
    """{key: value} of metadata rows, each (key, then its value's columns: string, integer, real, and the boolean
    where the version has it): each value in exactly one column, of that column's type, a boolean given back as
    one; refused by name otherwise."""
    kinds = (str, int, float, int)
    found = {}
    for key, *held in rows:
        set_ = [(at, value) for at, value in enumerate(held) if value is not None]
        if not isinstance(key, str) or len(set_) != 1:
            raise dbe.EncryptionRefused(f"{path}: {what} holds {key!r} with {len(set_)} values: nothing was written")
        at, value = set_[0]
        if type(value) is not kinds[at] or (at == 3 and value not in (0, 1)):
            raise dbe.EncryptionRefused(f"{path}: {what} holds {key!r} as a {type(value).__name__} in its "
                                        f"{('string', 'integer', 'real', 'boolean')[at]} column: nothing was written")
        found[key] = bool(value) if at == 3 else value
    return found


def _items(conn: Any, path: Path, name: str, segment: str, values: str) -> list:
    """[(id, text or None, metadata or None)] of the metadata segment ``segment``, in the order of its rows."""
    rows = conn.execute("SELECT id, embedding_id FROM embeddings WHERE segment_id = ? ORDER BY id",
                        (segment,)).fetchall()
    held: dict = {}
    for row_id, *entry in conn.execute(f"SELECT m.id, m.key, {values} FROM embedding_metadata m JOIN embeddings e "
                                       "ON e.id = m.id WHERE e.segment_id = ? ORDER BY m.id, m.key", (segment,)):
        held.setdefault(row_id, []).append(entry)
    items = []
    for row_id, item_id in rows:
        if not isinstance(item_id, str):
            raise dbe.EncryptionRefused(f"{path}: an item of {name} has the id {item_id!r}, not a string: nothing "
                                        "was written")
        metadata = _values(held.get(row_id, ()), f"{item_id} of {name}", path)
        text = metadata.pop(_DOCUMENT, None)
        if text is not None and not isinstance(text, str):
            raise dbe.EncryptionRefused(f"{path}: the text of {item_id} of {name} is not a string: nothing was "
                                        "written")
        own = sorted(key for key in metadata if key.startswith(_OWN_KEYS))
        if own:
            raise dbe.EncryptionRefused(f"{path}: {item_id} of {name} holds {own[0]!r}, a key ChromaDB keeps for "
                                        "itself, which the knowledge base's store has no place for: nothing was "
                                        "written")
        items.append((item_id, text, metadata or None))
    return items


def _applied(conn: Any, path: Path, name: str, collection_id: str, segment: str, topics: list) -> None:
    """Refused by name when ChromaDB's log of writes holds one for the collection past the last its metadata segment
    applied: the database would not say what the collection holds."""
    mine = [topic for topic in topics
            if isinstance(topic, str) and (topic == collection_id or topic.endswith("/" + collection_id))]
    if not mine:
        return
    newest = conn.execute(f"SELECT MAX(seq_id) FROM embeddings_queue WHERE topic IN ({','.join('?' * len(mine))})",
                          mine).fetchone()[0]
    if newest is None:
        return
    row = conn.execute("SELECT seq_id FROM max_seq_id WHERE segment_id = ?", (segment,)).fetchone()
    applied = None if row is None else row[0]
    if isinstance(applied, bytes):
        # Before ChromaDB kept it as an integer: big-endian, its last eight bytes, as its own migration reads it.
        applied = int.from_bytes(applied[-8:], "big")
    if not isinstance(newest, int) or (applied is not None and not isinstance(applied, int)):
        raise dbe.EncryptionRefused(f"{path}: the log of writes of {name} is not numbered as ChromaDB numbers it: "
                                    "nothing was written")
    if applied is None or newest > applied:
        raise dbe.EncryptionRefused(
            f"{path}: ChromaDB recorded writes to {name} it never applied (its log reaches {newest}, the collection "
            f"{applied}): open the store once with the ChromaDB that wrote it, then run migrate again: nothing was "
            "written")


def _space_of(schema: Any, configuration: Any, segment: Any, metadata: dict, name: str, path: Path) -> str:
    """The space a collection is measured in, as chromadb 1.3.7 reads it from the file: the space its schema records
    -- its embedding's index first, then its defaults --, or, before the schema, the one its configuration says under
    ``vector_index`` (1.x), unless that is ChromaDB's default, squared L2; then the space its index's segment records
    (``segment``, written by the versions before the configuration); then squared L2. 1.3.7 reads no other shape of
    the configuration, nor, before the schema, the collection's metadata: a metadata that says another space than
    the file's is refused by name, which one the store was measured in being told by no part of it."""
    parsed_configuration = (_json_object(configuration, f"the configuration of {name}", path)
                            if configuration not in (None, "") else {})
    configured = None
    if schema not in (None, ""):
        parsed = _json_object(schema, f"the schema of {name}", path)
        keys = parsed.get("keys")
        for holder in ((keys.get("#embedding") if isinstance(keys, dict) else None), parsed.get("defaults")):
            lists = holder.get("float_list") if isinstance(holder, dict) else None
            index = lists.get("vector_index") if isinstance(lists, dict) else None
            config = index.get("config") if isinstance(index, dict) else None
            if isinstance(config, dict) and config.get("space"):
                configured = str(config["space"])
                break
    else:
        index = parsed_configuration.get("vector_index")
        hnsw = index.get("hnsw") if isinstance(index, dict) else None
        if isinstance(hnsw, dict) and hnsw.get("space"):
            configured = str(hnsw["space"])
    space = configured if configured not in (None, "l2") else (segment or configured or "l2")
    said = metadata.get(vector_db.SPACE_KEY)
    if schema in (None, "") and said and str(said) != space:
        raise dbe.EncryptionRefused(
            f"{path}: the space of {name} is said two ways ({said} by its metadata, {space} by the rest of its file), "
            "and which one ChromaDB measured it in cannot be told: the folder stays as it is: nothing was written")
    return space


def _json_object(text: Any, what: str, path: Path) -> dict:
    """``text`` read as a JSON object; refused by name otherwise."""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if not isinstance(parsed, dict):
        raise dbe.EncryptionRefused(f"{path}: {what} is not a JSON object: nothing was written")
    return parsed


def _batch(collection: dict, folder: Path, namespace: str, facts: Any) -> dict:
    """A batch of ``vector_db.load`` for one collection read, its vectors still to compute: its items with a text,
    and in the memory's namespace only those an active fact holds with that very text. The items without a text,
    and in the memory the others, are left out, and counted."""
    space = collection["space"]
    if space not in vector_db.SPACES:
        raise dbe.EncryptionRefused(f"{collection['name']} of {folder} is measured in {space}, which the knowledge "
                                    "base's store does not measure: nothing was written")
    metadata = dict(collection["metadata"] or {})
    metadata[vector_db.SPACE_KEY] = space
    read = collection["items"]
    texted = [item for item in read if item[1]]
    kept = (texted if namespace != "memory" or facts is None else
            [item for item in texted if item[0] in facts and facts[item[0]] == item[1]])
    # Each item of the memory names the text its fact reads now, or None: the store keeps no other copy of it, which
    # an earlier run that committed and could not move its folder made, from whichever folder.
    return {"namespace": namespace, "name": collection["name"], "metadata": metadata, "source": str(folder),
            "items": kept, "space": space, "without_text": len(read) - len(texted),
            "left_out": len(texted) - len(kept),
            "drop": ({} if namespace != "memory" or facts is None else
                     {item[0]: facts.get(item[0]) for item in read})}


def _embedder(namespace: str) -> Any:
    """The embedder the module that queries ``namespace`` embeds its queries with, its configuration a copy of the
    module's (the check of its model may change what it checks), proved to serve that model itself -- not the faster
    one, nor another it finds --: refused by name otherwise, before any vector is asked for."""
    import dataclasses

    try:
        made = _module_embedder(namespace)
    except Exception as exc:  # noqa: BLE001 - an embedder that cannot be made computes no vector
        raise dbe.EncryptionRefused(f"the embedder of the namespace {namespace} cannot be made "
                                    f"({type(exc).__name__}: {exc}): nothing was written") from None
    made.config = dataclasses.replace(made.config)
    wanted = made.config.model
    try:
        served = made._verify_model()
    except Exception:  # noqa: BLE001 - a check that fails proves nothing
        served = False
    if not served or made.config.model.split(":")[0] != wanted.split(":")[0]:
        other = f" (it would embed with {made.config.model})" if served else ""
        raise dbe.EncryptionRefused(
            f"the namespace {namespace} is queried with the embedding model {wanted}, which {_served_at(wanted)} does "
            f"not serve{other}: serve it (`ollama pull {wanted.split(':')[0]}`), then run migrate again: nothing was "
            "written")
    return made


def _module_embedder(namespace: str) -> Any:
    """The embedder ``namespace``'s module makes for its queries, made as it makes it: the knowledge base's store's
    with its own model and server (``rag_store``); the documents' index's, the memory's and the projects' with the
    RAG configuration's -- the retriever's, ``memory.vector_store._default_embedder``'s and ``ProjectIndexer``'s
    alike --, without importing the memory or the projects, which open their own stores as they load."""
    from opti_oignon.rag.embeddings import OllamaEmbeddings

    if namespace == "knowledge":
        from opti_oignon import rag_store
        from opti_oignon.rag.config import EmbeddingConfig

        return OllamaEmbeddings(EmbeddingConfig(model=rag_store.EMBEDDING_MODEL, ollama_url=rag_store.OLLAMA_URL))
    if namespace in ("documents", "memory", "projects"):
        from opti_oignon.rag.config import get_config

        return OllamaEmbeddings(get_config().embedding)
    raise ValueError(f"no module queries the namespace {namespace}")


def _served_at(model: str) -> str:
    """Where the inference registry would ask for ``model``'s vectors, said."""
    try:
        from opti_oignon.rag.embeddings import _backend_for

        backend = _backend_for(model)
        endpoint = None if backend is None else backend.endpoint()
    except Exception:  # noqa: BLE001 - where cannot be told: said so
        return "the inference registry"
    return f"the inference registry's backend at {endpoint}" if endpoint else "the inference registry"


def _announce(batches: list, embedders: dict, say: Any) -> None:
    """How many texts each model embeds, in how many collections, told to ``say`` before the first vector."""
    if say is None:
        return
    counted: dict = {}
    for batch in batches:
        if batch["items"]:
            model = embedders[batch["namespace"]].config.model
            texts, collections = counted.get(model, (0, 0))
            counted[model] = (texts + len(batch["items"]), collections + 1)
    for model, (texts, collections) in sorted(counted.items()):
        say(f"embedding {texts} text(s) of {collections} collection(s) with {model}, through {_served_at(model)}")


def _as_stored(namespace: str, values: list) -> list:
    """A vector as ``namespace``'s module stores it: the projects' L2-normalized, as ``project_context`` normalizes
    each at its index and at its query alike (``_l2_normalize``), the others as the embedder gives it."""
    if namespace != "projects":
        return values
    norm = math.sqrt(sum(x * x for x in values))
    return values if norm <= 0.0 else [x / norm for x in values]


def _embed(batch: dict, embedder: Any, noted: list) -> None:
    """Each item of ``batch`` given the vector its text has now, ``_REQUEST`` texts a request: a signal noted between
    two requests stops the run; a text without a vector, a value that is not a finite number, or vectors of two
    dimensions refuse it, nothing written."""
    model, done = embedder.config.model, []
    items = batch["items"]
    for at in range(0, len(items), _REQUEST):
        if noted:
            raise dbe.EncryptionRefused(f"signal {noted[0]} noted while the vectors of {batch['name']} were computed: "
                                        "nothing was written")
        part = items[at:at + _REQUEST]
        vectors = embedder.embed_batch([item[1] for item in part])
        if vectors is None or len(vectors) != len(part):
            raise dbe.EncryptionRefused(f"{batch['name']} of {batch['source']}: {model} gave "
                                        f"{0 if vectors is None else len(vectors)} vector(s) for {len(part)} text(s): "
                                        "nothing was written")
        for item, vector in zip(part, vectors):
            values = _as_stored(batch["namespace"], [float(x) for x in vector] if vector else [])
            if not values or not all(math.isfinite(x) for x in values):
                raise dbe.EncryptionRefused(f"{item[0]} of {batch['name']} in {batch['source']}: {model} gave no "
                                            "vector of finite numbers for its text: nothing was written")
            done.append((item[0], item[1], item[2], array("f", values)))
    dimensions = sorted({len(item[3]) for item in done})
    if len(dimensions) > 1:
        raise dbe.EncryptionRefused(f"{batch['name']} of {batch['source']}: {model} gave vectors of "
                                    f"{dimensions[0]} and of {dimensions[-1]} dimensions: nothing was written")
    batch["items"], batch["model"], batch["dimension"] = done, model, dimensions[0] if dimensions else None


def _prove_pairing(batch: dict, embedder: Any) -> None:
    """Each text of a sample of ``batch`` -- its first item, its last, the last of each request for vectors, and the
    first ``_SAMPLE`` of its ids -- embedded once more, alone, gives a vector within ``_PAIRED`` of the one computed for
    that item, relative to the longer of the two: a vector paired with another text than its own, a request answered
    out of its order, is found so, and refuses the run, nothing written."""
    items = batch["items"]
    if not items:
        return
    by_id = sorted(range(len(items)), key=lambda at: items[at][0])[:_SAMPLE]
    for at in sorted({0, len(items) - 1, *by_id, *range(_REQUEST - 1, len(items), _REQUEST)}):
        item_id, text, _metadata, vector = items[at]
        again = embedder.embed_single(text)
        mine = _as_stored(batch["namespace"], [float(x) for x in again] if again else [])
        if len(mine) != len(vector) or not all(math.isfinite(x) for x in mine):
            raise dbe.EncryptionRefused(f"{item_id} of {batch['name']} in {batch['source']}: its text embedded again, "
                                        f"alone, gives no vector of {len(vector)} finite numbers: nothing was written")
        apart = math.sqrt(sum((a - b) ** 2 for a, b in zip(mine, vector)))
        longer = max(math.sqrt(sum(a * a for a in mine)), math.sqrt(sum(b * b for b in vector)))
        if apart > _PAIRED * longer:
            raise dbe.EncryptionRefused(
                f"{item_id} of {batch['name']} in {batch['source']}: its text embedded again, alone, lies "
                f"{apart / longer:.3g} from the vector computed for it, relative to the longer: the texts and the "
                "vectors are not paired: nothing was written")
