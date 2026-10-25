#!/usr/bin/env python3
"""Contracts for the move of ChromaDB's stores into the knowledge base's store, read from ChromaDB's own file.

``python3 -m opti_oignon.db_encryption migrate`` (``chroma_migration``)
reads each ChromaDB folder's ``chroma.sqlite3`` with the standard library,
read-only, embeds every text again with the embedder its namespace is
queried with, proves a sample of texts paired with their vectors, writes
every collection into the knowledge base's store in one transaction and
moves each folder into a folder of plaintext copies beside it. chromadb is
never imported. The stores are the ones the real chromadb 1.3.7 wrote
(``tests/chroma_golden/1.3.7``: each database dumped as text, its full-text
index left out, and what chromadb returned for it), or databases made here
in their layout, from their schema and their migrations, for the cases they
do not hold. The embedders are stand-ins: each namespace's model its own,
each text's vector a digest of the model and the text.

  * CZ1 -- THE READER GIVES WHAT CHROMADB GAVE: for every golden store, each
    collection's name, metadata (typed) and space, each item's id, text and
    metadata, in chromadb's order; an earlier version's layout, and 0.5 and
    0.6's configuration, read as theirs.
  * CZ2 -- NO BYTE OF CHROMADB'S FILES WRITTEN: a read, a refusal, a move
    leave every file as it was; a write through the reader's connection is
    refused by SQLite.
  * CZ3 -- A DATABASE THIS READER DOES NOT KNOW IS REFUSED BY NAME: an
    unknown migration, a gap, no metadata segment or two, a key ChromaDB keeps
    for itself, a value in no column or two or of another type, a second
    database, writes recorded and never applied, a file without ChromaDB's
    tables, a write-ahead log with no index, a configuration that is no JSON
    object: each named, nothing changed; a sequence number kept as a blob
    reads.
  * CZ4 -- EACH NAMESPACE EMBEDDED AS ITS MODULE EMBEDS: each namespace's
    vectors are its module's embedder's for each text; the module's own
    configuration untouched by the check of its model.
  * CZ5 -- AN EMBEDDER THAT CANNOT EMBED REFUSES THE RUN, NOTHING WRITTEN: a
    model not served, or served only by falling back to another, refuses
    before any vector is asked for; no embedder, one that cannot be made; a
    text with no vector, a value not finite, fewer vectors than texts,
    vectors of two dimensions, a dimension the store's collection does not
    hold: each named.
  * CZ6 -- EACH TEXT PROVED PAIRED WITH ITS VECTOR: vectors shifted by one,
    or the last two swapped, refuse the run; a second embedding within
    float noise passes; one that gives nothing refuses.
  * CZ7 -- AN ITEM WITHOUT TEXT: counted, said, not written, kept in the
    folder of plaintext copies; an empty text is none.
  * CZ8 -- WHERE EACH COLLECTION GOES, ENCRYPTED, KEPT BESIDE: each folder to
    its namespace, the memory's collection to its own, another folder left,
    a folder of copies passed over; the store encrypted, or in plaintext with
    no key, said; each folder moved whole into a folder of plaintext copies.
  * CZ9 -- EACH REFUSAL CHANGES NOTHING: the platform running, a process
    holding a file, the store in plaintext under a key, a space or a
    dimension a held collection does not have, a space the store does not
    measure, a copy that does not read back.
  * CZ10 -- WHAT THE PLATFORM WROTE SINCE IS KEPT, A STUCK RUN FINISHED: an id
    the store holds is kept; a folder that cannot be moved after the commit
    stops the run, said in order, and the next run moves it.
  * CZ11 -- A SIGNAL: before a store is read, between two requests for
    vectors, before the commit: nothing changed; after the commit the run
    goes on.
  * CZ12 -- GIT TAKES NO PLAINTEXT COPY.
  * CZ13 -- WHAT THE MEMORY STILL HOLDS: a vector only for an active fact
    holding the very text; facts that do not read refuse; a copy with no
    text whose fact is gone goes.
  * CZ14 -- A RESUMED RUN UNDOES WHAT IT MUST NOT KEEP: a fact rewritten or
    deleted between two runs has its copy removed; a vector the store holds
    with the text the fact reads is kept; the count said not counted when
    the store does not read back.
  * CZ15 -- TWO SOURCES OF THE MEMORY'S COLLECTION: they move together, the
    current text kept, whichever folder comes first, and after a run stopped
    between them.
  * CZ16 -- NO HALF RUN: a holder seen before the commit; a database that
    does not read; a store that does not open under the key: each refused by
    name, the folder the run made removed.
  * CZ17 -- WHAT WAITS: a repository below a folder named like a copy, a
    ``RAG_HOME`` spelled another way or through a link, move as any other.
  * CZ18 -- THE REAL CHAIN: in a process of its own, the network refused, the
    real mode, key file and audit chain, the golden stores rebuilt, the
    embedder's class replaced: every collection moved, encrypted, in its
    space, one ``lockfile_hmac_failure`` recorded.
  * CZ19 -- SAID BY CHECK, ONE RUN AT A TIME: ``check`` names the command for
    a ChromaDB store the command moves, never for another folder or a folder
    of plaintext copies; another run of ``encrypt`` or ``migrate`` under way
    refuses, nothing changed.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import ast
import dataclasses
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
import signal
import site
import sqlite3
import subprocess
import sys
import types
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_KEY = bytes(range(1, 33))
_OTHER_KEY = bytes(range(33, 65))
_PLAIN = b"SQLite format 3\x00"
_REPO = Path(__file__).resolve().parent.parent
_GOLDEN = Path(__file__).resolve().parent / "chroma_golden" / "1.3.7"
# The knowledge base's store embeds with a model of its own; the documents, the memory and the projects with the
# RAG configuration's.
_MODELS = {"knowledge": "m-knowledge", "documents": "m-rag", "memory": "m-rag", "projects": "m-rag"}
BUDGET_S = {
    "test_cz1_the_reader_gives_what_chromadb_gave_for_every_golden_store": 6.0,
    "test_cz2_reading_writes_no_byte_of_chromadbs_files": 6.0,
    "test_cz3_a_database_this_reader_does_not_know_is_refused_by_name": 12.0,
    "test_cz4_each_namespace_is_embedded_by_the_embedder_its_module_queries_with": 6.0,
    "test_cz5_an_embedder_that_cannot_embed_refuses_the_run_nothing_written": 8.0,
    "test_cz6_each_text_is_proved_paired_with_its_vector_before_the_commit": 8.0,
    "test_cz7_an_item_without_text_is_counted_said_and_kept_in_the_copy": 6.0,
    "test_cz8_each_folder_goes_to_its_namespace_encrypted_and_is_kept_beside": 8.0,
    "test_cz9_each_refusal_changes_nothing": 8.0,
    "test_cz10_an_id_the_store_holds_is_kept_and_a_stuck_run_is_finished_by_the_next": 8.0,
    "test_cz11_a_signal_before_the_commit_or_between_two_requests_changes_nothing": 8.0,
    "test_cz12_every_file_the_command_keeps_in_plaintext_is_one_git_ignores": 8.0,
    "test_cz13_the_memory_keeps_only_what_an_active_fact_holds_as_it_reads": 6.0,
    "test_cz14_a_resumed_run_removes_what_a_rewritten_or_deleted_fact_no_longer_holds": 6.0,
    "test_cz15_two_folders_holding_the_memory_move_together_the_current_text_kept": 8.0,
    "test_cz16_a_holder_seen_before_the_commit_and_every_failure_a_refusal_by_name": 6.0,
    "test_cz17_what_waits_moves_as_the_place_spells_it": 6.0,
    "test_cz18_migrate_with_the_real_mode_and_chain_on_the_golden_stores": 60.0,
    "test_cz19_check_names_the_command_for_the_stores_it_moves_and_one_run_at_a_time": 6.0,
}


def _expected():
    return json.loads((_GOLDEN / "expected.json").read_text(encoding="ascii"))


def _vector(model, text):
    """The stand-in embedder's vector for ``text`` under ``model``: four values float32 holds exactly."""
    digest = hashlib.sha256(f"{model}\0{text}".encode()).digest()
    return [(byte - 128) / 64 for byte in digest[:4]]


def _stored(namespace, model, text):
    """The vector the store holds for ``text``: the stand-in embedder's, L2-normalized in the projects' namespace as
    the projects normalize theirs, as float32 keeps it."""
    values = _vector(model, text)
    if namespace == "projects":
        norm = math.sqrt(sum(x * x for x in values))
        values = [x / norm for x in values]
    return [float(x) for x in array("f", values)]


def _embedding_modules(state):
    """The modules the migration makes its embedders through, stood in: each namespace's model its own."""

    @dataclasses.dataclass
    class EmbeddingConfig:
        model: str = "m-default"
        fast_model: str = "nomic-embed-text"
        ollama_url: str = "http://localhost:11434"
        timeout: int = 30

    class OllamaEmbeddings:
        def __init__(self, config=None):
            self.config = config if config is not None else EmbeddingConfig()
            self._model_verified = False
            state.made.append(self)

        def _verify_model(self):
            return state.verify(self)

        def embed_batch(self, texts):
            state.batches.append((self.config.model, list(texts)))
            return state.batch(self.config.model, list(texts))

        def embed_single(self, text):
            state.singles.append((self.config.model, text))
            return state.single(self.config.model, text)

    state.rag = EmbeddingConfig(model=_MODELS["documents"])
    config = types.ModuleType("opti_oignon.rag.config")
    config.EmbeddingConfig = EmbeddingConfig
    config.get_config = lambda: state.configured()
    embeddings = types.ModuleType("opti_oignon.rag.embeddings")
    embeddings.OllamaEmbeddings = OllamaEmbeddings
    embeddings._backend_for = lambda model: SimpleNamespace(endpoint=lambda: "http://localhost:11434")
    rag_store = types.ModuleType("opti_oignon.rag_store")
    rag_store.EMBEDDING_MODEL, rag_store.OLLAMA_URL = _MODELS["knowledge"], "http://localhost:11434"
    return {"opti_oignon.rag.config": config, "opti_oignon.rag.embeddings": embeddings,
            "opti_oignon.rag_store": rag_store}


def _world(tmp_path, monkeypatch, *, key=_KEY):
    """The command in a window: the places, the process table and the key redirected, the embedders stand-ins."""
    pytest.importorskip("sqlcipher3")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
        monkeypatch.delenv(name, raising=False)
    row = sqlite3.Row
    state = SimpleNamespace(made=[], batches=[], singles=[], served=set(_MODELS.values()))
    state.verify = lambda embedder: embedder.config.model.split(":")[0] in state.served
    state.batch = lambda model, texts: [_vector(model, text) for text in texts]
    state.single = _vector
    state.configured = lambda: SimpleNamespace(embedding=state.rag)
    seeded = _embedding_modules(state)
    loaded, restore = isolate(targets={
        "opti_oignon.encryption": source("encryption.py"),
        "opti_oignon.db_encryption": source("db_encryption.py"),
        "opti_oignon.db_utils": source("db_utils.py"),
        "opti_oignon.vector_db": source("vector_db.py"),
        "opti_oignon.chroma_migration": source("chroma_migration.py"),
    }, blocked=("opti_oignon.secure_bytes",), seeded=seeded, packages=("opti_oignon.rag",))

    def closed():
        try:
            restore()
        finally:
            sqlite3.Row = row

    try:
        dbe = loaded["opti_oignon.db_encryption"]
        places = (tmp_path / "data", tmp_path / "package" / "data")
        for place in places:
            place.mkdir(parents=True)
        proc = tmp_path / "proc"
        proc.mkdir()
        loaded["opti_oignon.encryption"]._DEFAULT_KEYFILE = tmp_path / "keys" / ".keyfile"
        dbe._STORE_PLACES = places
        dbe._LONE_STORES = []
        dbe._PROC = proc
        dbe._get_db_encryption_key = lambda: key
        dbe._loaded_key = lambda: key
        world = SimpleNamespace(dbe=dbe, cm=loaded["opti_oignon.chroma_migration"], vdb=loaded["opti_oignon.vector_db"],
                                dbu=loaded["opti_oignon.db_utils"], places=places, proc=proc, restore=closed,
                                store=places[1] / "rag" / "vectors.db", embed=state)
    except BaseException:
        closed()
        raise
    return world


def _columns(value):
    """The four value columns ChromaDB keeps a metadata value in: string, integer, real, boolean."""
    if isinstance(value, bool):
        return None, None, None, int(value)
    if isinstance(value, int):
        return None, value, None, None
    if isinstance(value, float):
        return None, None, value, None
    return value, None, None, None


def _fresh(folder):
    path = folder / "chroma.sqlite3"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("", "-wal", "-shm", "-journal"):
        if Path(f"{path}{name}").exists():
            Path(f"{path}{name}").unlink()
    return path


def _golden(place, relative, of=None):
    """The golden store of ``of`` (a folder chromadb 1.3.7 wrote) rebuilt in ``place/relative``, with the files it
    kept beside its database (their bytes made up)."""
    of = relative if of is None else of
    folder = place / relative
    conn = sqlite3.connect(str(_fresh(folder)))
    try:
        conn.executescript((_GOLDEN / of / "chroma.sql").read_text(encoding="ascii"))
    finally:
        conn.close()
    for beside in _expected()["files"][of]["beside"]:
        if beside != "chroma.sqlite3":
            (folder / beside).parent.mkdir(parents=True, exist_ok=True)
            (folder / beside).write_bytes(b"\x00" * 32)
    return folder


def _store(place, relative, collections):
    """A ChromaDB 1.3.7 database in ``place/relative``, its schema and migrations the golden's, holding
    ``collections``: each (name, metadata or None, items), each item (id, text or None, metadata or None), the
    way chromadb keeps them -- the text under ``chroma:document``, each value in its column, every write applied."""
    folder = place / relative
    conn = sqlite3.connect(str(_fresh(folder)))
    try:
        conn.executescript((_GOLDEN / "project_chroma" / "chroma.sql").read_text(encoding="ascii"))
        template = conn.execute("SELECT schema_str FROM collections").fetchone()[0]
        database = conn.execute("SELECT id FROM databases").fetchone()[0]
        for table in ("embedding_metadata", "embeddings", "max_seq_id", "embeddings_queue", "segment_metadata",
                      "segments", "collection_metadata", "collections"):
            conn.execute(f"DELETE FROM {table}")
        seq = 0
        for number, (name, metadata, items) in enumerate(collections, start=1):
            collection, own, index = (f"{number:08x}-0000-4000-8000-{kind:012x}" for kind in (1, 2, 3))
            # Its schema records the space its metadata says, or squared L2, as chromadb records it at creation.
            schema = json.loads(template)
            for holder in (schema["defaults"], schema["keys"]["#embedding"]):
                holder["float_list"]["vector_index"]["config"]["space"] = (metadata or {}).get("hnsw:space", "l2")
            conn.execute("INSERT INTO collections (id, name, dimension, database_id, config_json_str, schema_str) "
                         "VALUES (?, ?, ?, ?, '{}', ?)",
                         (collection, name, 4 if items else None, database, json.dumps(schema)))
            for key, value in (metadata or {}).items():
                conn.execute("INSERT INTO collection_metadata (collection_id, key, str_value, int_value, float_value, "
                             "bool_value) VALUES (?, ?, ?, ?, ?, ?)", (collection, key, *_columns(value)))
            conn.executemany("INSERT INTO segments (id, type, scope, collection) VALUES (?, ?, ?, ?)",
                             [(own, "urn:chroma:segment/metadata/sqlite", "METADATA", collection),
                              (index, "urn:chroma:segment/vector/hnsw-local-persisted", "VECTOR", collection)])
            for item_id, text, item_metadata in items:
                seq += 1
                row = conn.execute("INSERT INTO embeddings (segment_id, embedding_id, seq_id) VALUES (?, ?, ?)",
                                   (own, item_id, seq)).lastrowid
                entries = dict(item_metadata or {})
                if text is not None:
                    entries["chroma:document"] = text
                for key, value in entries.items():
                    conn.execute("INSERT INTO embedding_metadata (id, key, string_value, int_value, float_value, "
                                 "bool_value) VALUES (?, ?, ?, ?, ?, ?)", (row, key, *_columns(value)))
            if items:
                conn.execute("INSERT INTO max_seq_id (segment_id, seq_id) VALUES (?, ?)", (own, seq))
            (folder / index).mkdir(exist_ok=True)
            (folder / index / "data_level0.bin").write_bytes(b"\x00" * 64)
        conn.commit()
    finally:
        conn.close()
    return folder


def _sql(folder, *statements):
    conn = sqlite3.connect(str(folder / "chroma.sqlite3"))
    try:
        for statement in statements:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def _facts(world, *, active=(), inactive=()):
    """The memory's store of facts in the package's place, under the window's key: (id, text), active or not."""
    conn = world.dbu.safe_connect(world.places[1] / "memory_facts.db")
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS memory_facts (id TEXT PRIMARY KEY, text TEXT NOT NULL DEFAULT '', "
                     "active INTEGER NOT NULL DEFAULT 1)")
        conn.executemany("INSERT INTO memory_facts (id, text, active) VALUES (?, ?, ?)",
                         [(fact, text, 1) for fact, text in active] + [(fact, text, 0) for fact, text in inactive])
        conn.commit()
    finally:
        conn.close()


def _rewrite(world, statement):
    conn = world.dbu.safe_connect(world.places[1] / "memory_facts.db")
    try:
        conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def _tree(*roots):
    """Every file below ``roots``, with its bytes, and every folder."""
    found = {}
    for root in roots:
        for top, dirs, files in os.walk(root):
            for name in dirs:
                found[str(Path(top) / name) + "/"] = None
            for name in files:
                path = Path(top) / name
                found[str(path)] = path.read_bytes()
    return found


def _held(world, namespace, name):
    collection = world.vdb.Client(world.store, namespace).get_collection(name)
    got = collection.get(include=["documents", "metadatas", "embeddings"])
    return collection.space, list(zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]))


def _typed(value):
    """``value`` with the type of each scalar in it: True and 1 are not the same value of a metadata."""
    if isinstance(value, dict):
        return sorted((key, _typed(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_typed(item) for item in value]
    return type(value).__name__, value


def _refused(world, said):
    before = _tree(*world.places)
    with pytest.raises(world.dbe.EncryptionRefused, match=said):
        world.cm.migrate_all()
    assert _tree(*world.places) == before, ("the tree as it was", said)


def _kept(package, relative, besides=()):
    """The folder of plaintext copies a run moved ``relative``'s folder into, the folders ``besides`` aside."""
    parent = package / relative
    found = [p for p in parent.parent.iterdir()
             if p.name.startswith("plaintext-") and p.name not in besides and (p / parent.name).is_dir()]
    assert len(found) == 1, found
    return found[0] / parent.name


def _stuck(monkeypatch, world, folder):
    rename = os.rename

    def stuck(src, dst):
        if Path(src) == folder:
            raise OSError(5, "Input/output error")
        return rename(src, dst)

    monkeypatch.setattr(world.cm.os, "rename", stuck)
    return rename


# ---------------------------------------------------------------------------
# CZ1 -- the reader gives what chromadb gave
# ---------------------------------------------------------------------------
def test_cz1_the_reader_gives_what_chromadb_gave_for_every_golden_store(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        expected = _expected()
        assert expected["chromadb"] == "1.3.7" and len(expected["files"]["rag/chroma_v2"]["migrations"]) == 17
        for relative, listed in expected["stores"].items():
            read = {entry["name"]: entry for entry in world.cm._read_store(_golden(world.places[1], relative))}
            assert sorted(read) == sorted(entry["name"] for entry in listed), relative
            for want in listed:
                got = read[want["name"]]
                where = (relative, want["name"])
                assert _typed(got["metadata"]) == _typed(want["metadata"]), where
                # The space a query of chromadb measures in, found from its distances; for an empty collection,
                # the one its configuration gives.
                assert got["space"] == (want["measured"] or want["configuration"]["hnsw"]["space"]), where
                assert [item[0] for item in got["items"]] == want["ids"], ("chromadb's order", where)
                assert [item[1] for item in got["items"]] == want["documents"], where
                assert _typed([item[2] for item in got["items"]]) == _typed(want["metadatas"]), where
        inner = {entry["name"]: entry for entry in world.cm._read_store(world.places[1] / "rag/chroma_v2")}["inner"]
        assert [item[0] for item in inner["items"]] == ["i9", "i1"], "the order they came in, not sorted"
        # An earlier version's layout: its last migration not recorded yet, the column it adds not there -- the
        # legacy golden, whose collections hold no schema, but for the two whose schema it keeps.
        legacy = {entry["name"]: entry for entry in world.cm._read_store(world.places[1] / "legacy")}
        older = _golden(tmp_path / "older", "legacy")
        kept = "('legacy_schema_l2_segment_cosine', 'legacy_schema_ip_segment_cosine')"
        _sql(older, f"DELETE FROM embedding_metadata WHERE id IN (SELECT e.id FROM embeddings e JOIN segments s ON "
                    f"s.id = e.segment_id JOIN collections c ON c.id = s.collection WHERE c.name IN {kept})",
             f"DELETE FROM embeddings WHERE segment_id IN (SELECT s.id FROM segments s JOIN collections c ON "
             f"c.id = s.collection WHERE c.name IN {kept})",
             f"DELETE FROM segment_metadata WHERE segment_id IN (SELECT s.id FROM segments s JOIN collections c ON "
             f"c.id = s.collection WHERE c.name IN {kept})",
             f"DELETE FROM segments WHERE collection IN (SELECT id FROM collections WHERE name IN {kept})",
             f"DELETE FROM collections WHERE name IN {kept}",
             "DELETE FROM migrations WHERE dir = 'sysdb' AND version = 10",
             "ALTER TABLE collections DROP COLUMN schema_str")
        assert {entry["name"]: entry for entry in world.cm._read_store(older)} == {
            name: entry for name, entry in legacy.items() if name not in kept}, "read as its own"
        # Before the schema, as 1.3.7 reads it: its configuration under vector_index unless the default, then its
        # index's segment; no other shape of the configuration.
        for configuration, segment, space in (
                ('{"vector_index": {"hnsw": {"space": "ip"}}}', None, "ip"),
                ('{"vector_index": {"hnsw": {"space": "ip"}}}', "cosine", "ip"),
                ('{"vector_index": {"hnsw": {"space": "l2"}}}', "cosine", "cosine"),
                ('{"hnsw_configuration": {"space": "ip"}}', None, "l2"), ('{"hnsw": {"space": "ip"}}', None, "l2"),
                ("{}", "ip", "ip"), (None, None, "l2")):
            _sql(older, "UPDATE collections SET config_json_str = ? WHERE name = 'legacy_top_ip'".replace(
                "?", "NULL" if configuration is None else "'" + configuration + "'"),
                 "DELETE FROM segment_metadata WHERE segment_id = (SELECT s.id FROM segments s JOIN collections c ON "
                 "c.id = s.collection WHERE c.name = 'legacy_top_ip' AND s.scope = 'VECTOR')",
                 *([] if segment is None else [
                     "INSERT INTO segment_metadata (segment_id, key, str_value) SELECT s.id, 'hnsw:space', "
                     f"'{segment}' FROM segments s JOIN collections c ON c.id = s.collection WHERE "
                     "c.name = 'legacy_top_ip' AND s.scope = 'VECTOR'"]))
            got = {entry["name"]: entry for entry in world.cm._read_store(older)}["legacy_top_ip"]
            assert got["space"] == space, (configuration, segment, got["space"])
        # Before the schema, a metadata that says another space than the file: which one is not told, refused.
        _sql(older, "INSERT INTO collection_metadata (collection_id, key, str_value) SELECT id, 'hnsw:space', "
                    "'cosine' FROM collections WHERE name = 'legacy_nested_ip'")
        with pytest.raises(world.dbe.EncryptionRefused, match="the space of legacy_nested_ip is said two ways \\("
                                                              "cosine by its metadata, ip by the rest of its file\\)"):
            world.cm._read_store(older)
    finally:
        world.restore()
    # Moved, each collection of the spaces and the legacy goldens measured in the space chromadb measures it in.
    world = _world(tmp_path / "moved", monkeypatch)
    try:
        _golden(world.places[1], "rag/chroma_v2", of="spaces")
        _golden(world.places[1], "project_chroma", of="legacy")
        assert world.cm.migrate_all()["moved"]
        for namespace, store in (("knowledge", "spaces"), ("projects", "legacy")):
            for want in _expected()["stores"][store]:
                assert _held(world, namespace, want["name"])[0] == want["measured"], (store, want["name"])
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ2 -- no byte of ChromaDB's files written
# ---------------------------------------------------------------------------
def test_cz2_reading_writes_no_byte_of_chromadbs_files(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        folders = [_golden(package, relative) for relative in ("rag/chroma_v2", "rag/chroma_db", "project_chroma")]
        before = _tree(*folders)
        for folder in folders:
            world.cm._read_store(folder)
        assert _tree(*folders) == before, "a read writes nothing, creates nothing"
        collections, tried = world.cm._collections, []

        def probing(conn, path):
            assert conn.in_transaction, "every read in one read transaction"
            # The database itself opened read-only, whatever the connection's own switch says.
            conn.execute("PRAGMA query_only = OFF")
            for statement in ("CREATE TABLE probe (x)", "DELETE FROM migrations",
                              "UPDATE embeddings SET embedding_id = 'x'"):
                with pytest.raises(sqlite3.Error):
                    conn.execute(statement)
                tried.append(statement)
            conn.execute("PRAGMA query_only = ON")
            return collections(conn, path)

        monkeypatch.setattr(world.cm, "_collections", probing)
        world.cm._read_store(folders[0])
        monkeypatch.setattr(world.cm, "_collections", collections)
        assert len(tried) == 3 and _tree(*folders) == before, "the reader's connection refuses every write"
        _facts(world, active=[("m1", "a fact"), ("m3", "un fait " + chr(0xE0) + " garder")])
        assert world.cm.migrate_all()["moved"]
        for folder in folders:
            kept = _kept(package, folder.relative_to(package))
            assert _tree(kept) == {key.replace(str(folder), str(kept), 1): value for key, value in before.items()
                                   if key.startswith(str(folder))}, ("moved whole, byte for byte", folder)
    finally:
        world.restore()
    # A database in write-ahead-log mode, closed clean: read, refused, moved, nothing made beside it.
    world = _world(tmp_path / "wal", monkeypatch)
    try:
        package = world.places[1]
        folder = _golden(package, "rag/chroma_v2")
        _sql(folder, "PRAGMA journal_mode = WAL")
        assert (folder / "chroma.sqlite3").read_bytes()[18:20] == b"\x02\x02" and not (
            folder / "chroma.sqlite3-shm").exists(), "control: in WAL mode, no index and no log beside it"
        before = _tree(folder)
        world.cm._read_store(folder)
        assert _tree(folder) == before, "read: no index, no log made beside it"
        (folder / "chroma.sqlite3-wal").write_bytes(b"")
        world.cm._read_store(folder)
        assert _tree(folder) == {**before, str(folder / "chroma.sqlite3-wal"): b""}, "an empty log, nothing in it to read"
        (folder / "chroma.sqlite3-wal").unlink()
        (folder / "chroma.sqlite3-shm").write_bytes(b"")
        world.cm._read_store(folder)
        assert _tree(folder) == {**before, str(folder / "chroma.sqlite3-shm"): b""}, "an index and no log: none made"
        (folder / "chroma.sqlite3-shm").unlink()
        world.embed.served = set()
        _refused(world, "is queried with the embedding model m-knowledge")
        world.embed.served = set(_MODELS.values())
        assert world.cm.migrate_all()["moved"]
        kept = _kept(package, "rag/chroma_v2")
        assert _tree(kept) == {key.replace(str(folder), str(kept), 1): value for key, value in before.items()}, (
            "moved as it was")
        # A log that holds writes, its index beside -- as a writer that stopped left them: read without a byte of
        # the index written. (Taken as a copy while the writer holds them: a writer in this process would share its
        # index with the reader's connection.)
        live = _golden(tmp_path / "wal-live", "project_chroma")
        copy = package / "project_chroma"
        copy.mkdir(parents=True)
        writer = sqlite3.connect(str(live / "chroma.sqlite3"))
        try:
            writer.execute("PRAGMA journal_mode = WAL")
            writer.execute("UPDATE embedding_metadata SET string_value = 'x = 1, in the log' WHERE "
                           "key = 'chroma:document' AND string_value = 'x = 1'")
            writer.commit()
            for name in ("chroma.sqlite3", "chroma.sqlite3-wal", "chroma.sqlite3-shm"):
                shutil.copy2(live / name, copy / name)
        finally:
            writer.close()
        assert (copy / "chroma.sqlite3-wal").stat().st_size and (copy / "chroma.sqlite3-shm").exists(), "control"
        before = _tree(copy)
        read = world.cm._read_store(copy)
        assert [item[1] for item in read[0]["items"]][0] == "x = 1, in the log", "the writes the log holds, read"
        assert _tree(copy) == before, "the index read-only: not a byte of it written"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ3 -- a database this reader does not know is refused by name
# ---------------------------------------------------------------------------
def test_cz3_a_database_this_reader_does_not_know_is_refused_by_name(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        papers = ("(SELECT id FROM collections WHERE name = 'papers')")
        segment = f"(SELECT id FROM segments WHERE scope = 'METADATA' AND collection = {papers})"
        first = f"(SELECT MIN(id) FROM embeddings WHERE segment_id = {segment})"
        cases = (
            (["INSERT INTO migrations VALUES ('sysdb', 11, '00011-future.sqlite.sql', '', '')"],
             "written by a ChromaDB this command does not know: migration sysdb 11 00011-future.sqlite.sql"),
            (["INSERT INTO migrations VALUES ('rustdb', 1, '00001-rust.sqlite.sql', '', '')"],
             "this command does not know: migration rustdb 1 00001-rust.sqlite.sql"),
            (["DELETE FROM migrations WHERE dir = 'sysdb' AND version = 5"],
             "this command does not know: migration sysdb 10 00010-collection-schema.sqlite.sql"),
            ([f"DELETE FROM segments WHERE id = {segment}"], "papers has 0 metadata segments"),
            ([f"INSERT INTO segments VALUES ('extra', 'urn:chroma:segment/metadata/sqlite', 'METADATA', {papers})"],
             "papers has 2 metadata segments"),
            ([f"INSERT INTO embedding_metadata (id, key, string_value) VALUES ({first}, 'chroma:uri', 'x')"],
             "p1 of papers holds 'chroma:uri', a key ChromaDB keeps for itself"),
            ([f"INSERT INTO embedding_metadata (id, key) VALUES ({first}, 'empty')"],
             "p1 of papers holds 'empty' with 0 values"),
            ([f"UPDATE embedding_metadata SET int_value = 1 WHERE id = {first} AND key = 'source_file'"],
             "p1 of papers holds 'source_file' with 2 values"),
            ([f"UPDATE embedding_metadata SET int_value = 'x', string_value = NULL WHERE id = {first} AND key = 'page'"],
             "p1 of papers holds 'page' as a str in its integer column"),
            ([f"UPDATE embedding_metadata SET bool_value = 2 WHERE id = {first} AND key = 'edited'"],
             "p1 of papers holds 'edited' as a int in its boolean column"),
            ([f"UPDATE embedding_metadata SET int_value = 5, string_value = NULL WHERE id = {first} "
              "AND key = 'chroma:document'"], "the text of p1 of papers is not a string"),
            ([f"UPDATE embeddings SET embedding_id = X'01' WHERE id = {first}"],
             "an item of papers has the id b'.x01', not a string"),
            (["INSERT INTO databases VALUES ('11111111-0000-0000-0000-000000000000', 'other', 'default_tenant')",
              "UPDATE collections SET database_id = '11111111-0000-0000-0000-000000000000' WHERE name = 'bulk'"],
             "its collections lie in 2 databases"),
            ([f"INSERT INTO embeddings_queue (seq_id, operation, topic, id) VALUES (5000, 0, "
              f"'persistent://default/default/' || {papers}, 'p10')"],
             "ChromaDB recorded writes to papers it never applied \\(its log reaches 5000, the collection 12\\)"),
            ([f"DELETE FROM max_seq_id WHERE segment_id = {segment}"],
             "ChromaDB recorded writes to papers it never applied \\(its log reaches 12, the collection None\\)"),
            (["UPDATE collections SET config_json_str = 'not json' WHERE name = 'papers'"],
             "the configuration of papers is not a JSON object"),
            (["UPDATE collections SET config_json_str = '[1]' WHERE name = 'papers'"],
             "the configuration of papers is not a JSON object"),
        )
        for statements, said in cases:
            _sql(_golden(package, "rag/chroma_v2"), *statements)
            _refused(world, said)
        _golden(package, "rag/chroma_v2")
        conn = sqlite3.connect(str(package / "rag/chroma_v2/chroma.sqlite3"))
        try:
            conn.execute("DROP TABLE migrations")
            conn.commit()
        finally:
            conn.close()
        _refused(world, "no table migrations: not a ChromaDB database this command knows")
        _golden(package, "rag/chroma_v2")
        (package / "rag/chroma_v2/chroma.sqlite3-wal").write_bytes(b"\x37\x7f\x06\x82" + b"\x00" * 28)
        _refused(world, "has a write-ahead log and no index of it beside: read-only, the log would go unread; fold it")
        (package / "rag/chroma_v2/chroma.sqlite3-wal").unlink()
        # A sequence number kept as a blob, as before ChromaDB kept it as an integer: read as its own migration reads it.
        _golden(package, "rag/chroma_v2")
        _sql(package / "rag/chroma_v2", f"UPDATE max_seq_id SET seq_id = X'000000000000000c' WHERE segment_id = {segment}")
        assert [entry["name"] for entry in world.cm._read_store(package / "rag/chroma_v2")], "control: it reads"
        _sql(package / "rag/chroma_v2", f"UPDATE max_seq_id SET seq_id = X'000000000000000b' WHERE segment_id = {segment}")
        _refused(world, "its log reaches 12, the collection 11")
        # Two index segments of one collection that record two spaces.
        _sql(_golden(package, "project_chroma", of="legacy"),
             "INSERT INTO segments (id, type, scope, collection) SELECT 'second-index', "
             "'urn:chroma:segment/vector/hnsw-local-persisted', 'VECTOR', id FROM collections "
             "WHERE name = 'legacy_segment_cosine'",
             "INSERT INTO segment_metadata (segment_id, key, str_value) VALUES ('second-index', 'hnsw:space', 'ip')")
        _refused(world, "the index segments of legacy_segment_cosine record 2 spaces")
        shutil.rmtree(package / "project_chroma")
        _golden(package, "rag/chroma_v2")
        _facts(world)
        assert world.cm.migrate_all()["moved"], "control: the golden itself moves"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ4 -- each namespace embedded as its module embeds
# ---------------------------------------------------------------------------
def test_cz4_each_namespace_is_embedded_by_the_embedder_its_module_queries_with(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        for relative in ("rag/chroma_v2", "rag/chroma_db", "project_chroma"):
            _golden(package, relative)
        _facts(world, active=[("m1", "a fact"), ("m3", "un fait " + chr(0xE0) + " garder")])
        checked = world.embed.verify

        def tagged(embedder):
            # The check of a model names it in full, as the real one does: the module's own name must not change.
            embedder.config.model += ":latest"
            return checked(embedder)

        world.embed.verify = tagged
        done = world.cm.migrate_all()
        assert len(done["moved"]) == 3
        assert world.embed.rag.model == "m-rag", "the RAG configuration as it was"
        for namespace, name in (("knowledge", "papers"), ("knowledge", "bulk"), ("knowledge", "inner"),
                                ("documents", "documents"), ("memory", "oo_memories"), ("projects", "project_p1")):
            model = _MODELS[namespace] + ":latest"
            _space, items = _held(world, namespace, name)
            assert items and all(vector == _stored(namespace, model, text) for _id, text, _metadata, vector in items), (
                namespace, name, "each vector its namespace's embedder's for its text, as its module stores it")
        asked = {model for model, _texts in world.embed.batches}
        assert asked == {"m-knowledge:latest", "m-rag:latest"}, asked
        entry = next(e for moved in done["moved"] for e in moved["collections"] if e["name"] == "papers")
        assert (entry["model"], entry["dimension"]) == ("m-knowledge:latest", 4), entry
    finally:
        world.restore()
    # How each module embeds its queries, read from its source: the migration embeds as they do, or this goes red.
    store = _functions(source("rag_store.py"))
    assert {"EmbeddingConfig(model=self._embedding_model, ollama_url=self._ollama_url)", "OllamaEmbeddings(cfg)"} <= (
        _calls(_method(source("rag_store.py"), "RAGVectorStore", "_get_embedder"))), (
        "the knowledge base's store embeds with its own model and server")
    for function in (store["get_rag_store"], _method(source("rag_store.py"), "RAGVectorStore", "__init__")):
        assert [ast.unparse(default) for default in function.args.defaults][-2:] == [
            "EMBEDDING_MODEL", "OLLAMA_URL"], (function.name, "the platform's store made with them")
    assert "OllamaEmbeddings()" in _calls(_functions(source("memory", "vector_store.py"))["_default_embedder"])
    assert {"get_rag_config()", "OllamaEmbeddings(rag_config.embedding)"} <= _calls(
        _method(source("project_context.py"), "ProjectIndexer", "_get_embedder"))
    imported = {(node.module, alias.name, alias.asname)
                for node in ast.walk(ast.parse(source("project_context.py").read_text(encoding="utf-8")))
                if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert ("opti_oignon.rag.config", "get_config", "get_rag_config") in imported, "the RAG configuration's"
    for part in ("retriever.py", "indexer.py"):
        found = _calls(ast.parse(source("rag", part).read_text(encoding="utf-8")))
        assert "OllamaEmbeddings(self.config.embedding)" in found and "get_config()" in found, part
    embedder = _method(source("rag", "embeddings.py"), "OllamaEmbeddings", "__init__")
    assert "self.config = config or get_config().embedding" in ast.unparse(embedder), "a default made the same way"
    # The projects normalize what they store and what they query; no other module normalizes.
    assert {"_l2_normalize(e)", "_l2_normalize(query_embedding)"} <= _calls(
        ast.parse(source("project_context.py").read_text(encoding="utf-8")))
    normalize = ast.unparse(_functions(source("project_context.py"))["_l2_normalize"])
    assert "if norm <= 0.0:" in normalize and "return [x / norm for x in vec]" in normalize, normalize
    for path in (source("rag_store.py"), source("rag", "indexer.py"), source("rag", "retriever.py"),
                 source("memory", "vector_store.py")):
        found = [call for call in _calls(ast.parse(path.read_text(encoding="utf-8")))
                 if "normaliz" in call.split("(")[0].lower()]
        assert found == [], (path.name, found)


def _functions(path):
    """{name: the definition of that name} of the functions at the top of the module at ``path``, and of the methods
    whose name no other definition shares."""
    found = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(node.name, []).append(node)
    return {name: nodes[0] for name, nodes in found.items() if len(nodes) == 1}


def _method(path, cls, name):
    """The method ``name`` of the class ``cls`` in the module at ``path``."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == name)
    raise AssertionError(f"no class {cls} in {path}")


def _calls(node):
    return {ast.unparse(call) for call in ast.walk(node) if isinstance(call, ast.Call)}


# ---------------------------------------------------------------------------
# CZ5 -- an embedder that cannot embed refuses the run, nothing written
# ---------------------------------------------------------------------------
def test_cz5_an_embedder_that_cannot_embed_refuses_the_run_nothing_written(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        _golden(package, "rag/chroma_v2")
        _golden(package, "rag/chroma_db")
        _facts(world, active=[("m1", "a fact")])
        served, state = set(world.embed.served), world.embed
        state.served = served - {"m-rag"}
        _refused(world, "the namespace documents is queried with the embedding model m-rag, which the "
                        "inference registry's backend at http://localhost:11434 does not serve: serve it "
                        "\\(`ollama pull m-rag`\\), then run migrate again")
        assert state.batches == [], "refused before a single vector is asked for"
        state.served = served
        checked = state.verify

        def fallback(embedder):
            if embedder.config.model == "m-knowledge":
                embedder.config.model = "nomic-embed-text:latest"
                return True
            return checked(embedder)

        state.verify = fallback
        _refused(world, "the namespace knowledge is queried with the embedding model m-knowledge, which .* does not "
                        "serve \\(it would embed with nomic-embed-text:latest\\)")
        state.verify = lambda embedder: (_ for _ in ()).throw(OSError("connection refused"))
        _refused(world, "the namespace documents is queried with the embedding model m-rag, which")
        state.verify = checked
        assert state.batches == []
        state.configured = lambda: (_ for _ in ()).throw(ImportError("no numpy"))
        _refused(world, "the embedder of the namespace documents cannot be made \\(ImportError: no numpy\\)")
        state.configured = lambda: SimpleNamespace(embedding=state.rag)
        batch = state.batch
        for broken, said in (
                (lambda model, texts: [None if text == "dogs bark" or text.startswith("caf") else _vector(model, text)
                                       for text in texts], "p2 of papers in .*: m-knowledge gave no vector of finite"),
                (lambda model, texts: [[math.nan] * 4 if text == "cats purr, rewritten" else _vector(model, text)
                                       for text in texts], "p1 of papers in .*: m-knowledge gave no vector of finite"),
                (lambda model, texts: [_vector(model, text) for text in texts][:-1],
                 "of .*: m-[a-z]+ gave (\\d+) vector\\(s\\) for (?!\\1 )\\d+ text\\(s\\)"),
                (lambda model, texts: [_vector(model, text)[:3] if text == "t5" else _vector(model, text)
                                       for text in texts], "bulk of .*: m-knowledge gave vectors of 3 and of 4 "
                                                           "dimensions")):
            state.batch = broken
            _refused(world, said)
        state.batch = batch
        held = world.vdb.Client(world.store, "knowledge").create_collection("papers", metadata={"hnsw:space": "cosine"})
        held.add(ids=["other"], embeddings=[[1.0, 0.0]])
        _refused(world, "knowledge/papers holds vectors of dimension 2 in .* and of dimension 4 in")
        held.delete(ids=["other"])
        assert world.cm.migrate_all()["moved"], "control: each cause gone, the same tree moves"
    finally:
        world.restore()
    # A namespace with no text to embed needs no model: an empty collection, an item without text.
    world = _world(tmp_path / "nothing", monkeypatch)
    try:
        _golden(world.places[1], "rag/chroma_v2")
        _store(world.places[1], "project_chroma", [("project_p", None, []), ("project_q", None, [("q", None, None)])])
        world.embed.served = {"m-knowledge"}
        assert world.cm.migrate_all()["moved"], "the projects' model never asked for"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ6 -- each text proved paired with its vector
# ---------------------------------------------------------------------------
def test_cz6_each_text_is_proved_paired_with_its_vector_before_the_commit(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        _golden(package, "rag/chroma_v2")
        state = world.embed
        batch, single = state.batch, state.single
        state.batch = lambda model, texts: [_vector(model, text) for text in texts[1:] + texts[:1]]
        _refused(world, "of papers in .*: its text embedded again, alone, lies .* from the vector computed for it, "
                        "relative to the longer: the texts and the vectors are not paired")

        # The last two of the collection swapped: its last item, sampled whatever its id, finds it.
        state.batch = lambda model, texts: (
            [_vector(model, text) for text in texts[:-2]] + [_vector(model, texts[-1]), _vector(model, texts[-2])]
            if texts[-1] == "t1001" else [_vector(model, text) for text in texts])
        _refused(world, "bulk-1001 of bulk in .*: its text embedded again, alone, lies")
        # The last two of one request swapped, in the middle of the collection: the request's last item finds it.
        state.batch = lambda model, texts: (
            [_vector(model, text) for text in texts[:-2]] + [_vector(model, texts[-1]), _vector(model, texts[-2])]
            if texts[-1] == "t192" else [_vector(model, text) for text in texts])
        _refused(world, "bulk-0192 of bulk in .*: its text embedded again, alone, lies")
        state.batch = batch
        state.single = lambda model, text: None if text == "t0" else single(model, text)
        _refused(world, "bulk-0000 of bulk in .*: its text embedded again, alone, gives no vector of 4 finite numbers")
        state.single = lambda model, text: [value * 1.05 for value in single(model, text)]
        # 5 % apart, relative to the longer: 0.0476, past the tolerance of float noise.
        _refused(world, "of papers in .*: its text embedded again, alone, lies 0.0476 from the vector computed for it")
        state.single = lambda model, text: [value * (1 + 1e-4) for value in single(model, text)]
        state.singles.clear()
        assert world.cm.migrate_all()["moved"], "a second embedding within float noise of the first proves"
        bulk = [text for model, text in state.singles if text.startswith("t")]
        assert {"t0", "t7", "t64", "t960", "t1001"} <= set(bulk) and len(bulk) == 24, (
            "its first, its last, its first eight ids, the last of each request", bulk)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ7 -- an item without text
# ---------------------------------------------------------------------------
def test_cz7_an_item_without_text_is_counted_said_and_kept_in_the_copy(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        _golden(package, "rag/chroma_v2")
        _store(package, "project_chroma", [("project_p", {"project_id": "p"},
                                            [("a", "x = 1", None), ("b", "", {"source_file": "b"})])])
        assert world.dbe.main(["migrate"]) == 0
        said = capsys.readouterr().out
        assert "  papers, into the namespace knowledge: 7 item(s) written, their vectors computed again with " \
               "m-knowledge (4 dimensions), 0 already there kept, 1 without text, not written: ChromaDB's folder " \
               "keeps them\n" in said, said
        assert "  project_p, into the namespace projects: 1 item(s) written, their vectors computed again with " \
               "m-rag (4 dimensions), 0 already there kept, 1 without text, not written" in said, said
        assert "embedding 1010 text(s) of 3 collection(s) with m-knowledge, through the inference registry's " \
               "backend at http://localhost:11434\n" in said, said
        assert "p5" not in [item[0] for item in _held(world, "knowledge", "papers")[1]]
        assert [item[0] for item in _held(world, "projects", "project_p")[1]] == ["a"], "an empty text is none"
        kept = {entry["name"]: entry for entry in world.cm._read_store(_kept(package, "rag/chroma_v2"))}
        assert ("p5", None, {"source_file": "/no-text.pdf"}) in kept["papers"]["items"], "kept in the copy"
    finally:
        world.restore()
    # What each model embeds is told before the first request for vectors.
    world = _world(tmp_path / "order", monkeypatch)
    try:
        _golden(world.places[1], "project_chroma")
        events, batch = [], world.embed.batch
        world.embed.batch = lambda model, texts: events.append("request") or batch(model, texts)
        assert world.cm.migrate_all(say=events.append)["moved"]
        assert events[0].startswith("embedding 2 text(s) of 1 collection(s) with m-rag") and events[1:] == [
            "request"], events
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ8 -- where each collection goes, encrypted, kept beside
# ---------------------------------------------------------------------------
def test_cz8_each_folder_goes_to_its_namespace_encrypted_and_is_kept_beside(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        repo, package = world.places
        knowledge = _golden(package, "rag/chroma_v2")
        index = _golden(package, "rag/chroma_db")
        projects = _golden(package, "project_chroma")
        elsewhere = _store(repo, "rag/chroma_db", [("documents", None, [("d2", "more", None)])])
        unknown = _store(package, "someone_elses_chroma", [("theirs", None, [])])
        _facts(world, active=[("m1", "a fact")])
        kept = _store(package, "rag/plaintext-20260101T000000Z/chroma_v2", [("old", None, [])])
        copied = _store(package, "exports/plaintext-20260101T000000Z/chroma_v2", [("old", None, [])])
        assert copied / "chroma.sqlite3" in world.dbe.plaintext_others(), "control: a copy outside a store folder"
        found, other = world.cm.sources()
        assert sorted((str(f), n) for f, n in found) == sorted([
            (str(knowledge), "knowledge"), (str(index), "documents"), (str(projects), "projects"),
            (str(elsewhere), "documents")]), found
        assert other == [unknown], "a folder the command does not know is left; a folder of copies passed over"
        before = _tree(unknown, kept.parent, copied.parent)
        assert world.dbe.main(["migrate"]) == 0
        said = capsys.readouterr().out
        assert _tree(unknown, kept.parent, copied.parent) == before, "left as they were"
        with world.vdb._connection(world.store, write=False) as conn:
            held = sorted(conn.execute("SELECT namespace, name FROM collections").fetchall())
            records = conn.execute("SELECT source, namespace, collection, copied, kept FROM migrations "
                                   "WHERE collection = 'papers'").fetchall()
        assert held == [("documents", "documents"), ("knowledge", "bulk"), ("knowledge", "empty"),
                        ("knowledge", "inner"), ("knowledge", "papers"), ("memory", "oo_memories"),
                        ("projects", "project_p1")], held
        assert sorted(i[0] for i in _held(world, "documents", "documents")[1]) == ["/n.txt-0", "/n.txt-1", "d2"], (
            "two folders, one collection")
        assert [i[0] for i in _held(world, "memory", "oo_memories")[1]] == ["m1"], "the memory's collection, its own"
        assert world.store.read_bytes()[:16] != _PLAIN, "the store is encrypted"
        assert _held(world, "knowledge", "inner")[0] == "ip" and _held(world, "knowledge", "empty") == ("cosine", [])
        assert world.vdb.Client(world.store, "knowledge").get_collection("inner").metadata == {
            "hnsw:space": "ip", "owner": "L" + chr(0xE9) + "a", "count": 3, "ratio": 0.5, "public": False}
        _space, papers = _held(world, "knowledge", "papers")
        assert [(i[0], i[1], i[2]) for i in papers][:2] == [
            ("p1", "cats purr, rewritten", {"source_file": "/a.pdf", "page": 3, "parent_doc_id": "d1", "edited": True}),
            ("p2", "caf" + chr(0xE9) + " again", {"flag": True, "parent_doc_id": "d2", "source_file": "/b.pdf"})]
        assert records == [(str(knowledge), "knowledge", "papers", 7, 0)], records
        for folder, relative in ((knowledge, "rag/chroma_v2"), (index, "rag/chroma_db"), (projects, "project_chroma")):
            copy = _kept(package, relative, besides=(kept.parent.name,))
            assert not folder.exists() and (copy / "chroma.sqlite3").is_file(), folder
            assert copy.parent in world.dbe.plaintext_folders() and oct(copy.parent.stat().st_mode & 0o777) == "0o700"
            assert f"{folder}: " in said and f"its folder is kept in {copy.parent}" in said, said
        assert f"the knowledge base's store: {world.store}, encrypted under the key that loads" in said
        assert world.dbe.main(["migrate"]) == 0 and "no ChromaDB store to move" in capsys.readouterr().out
    finally:
        world.restore()
    world = _world(tmp_path / "no-key", monkeypatch, key=None)
    try:
        _golden(world.places[1], "project_chroma")
        assert world.dbe.main(["migrate"]) == 0
        assert world.store.read_bytes()[:16] == _PLAIN, "no key loads: in plaintext, as every store of the platform"
        assert "in plaintext: no key loads, as every store of the platform" in capsys.readouterr().out
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ9 -- each refusal changes nothing
# ---------------------------------------------------------------------------
def test_cz9_each_refusal_changes_nothing(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        _golden(package, "rag/chroma_v2")
        entry = world.proc / "4242"
        (entry / "fd").mkdir(parents=True)
        (entry / "cmdline").write_bytes(b"python3\0-m\0uvicorn\0opti_oignon.api.app:app\0")
        _refused(world, "stop the platform first \\(process 4242 runs the platform")
        (entry / "cmdline").write_bytes(b"sqlite3\0")
        os.symlink(package / "rag" / "chroma_v2" / "chroma.sqlite3", entry / "fd" / "3")
        _refused(world, "process 4242 holds chroma.sqlite3 open")
        os.unlink(entry / "fd" / "3")
        world.store.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(world.store))
        try:
            conn.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY)")
            conn.commit()
        finally:
            conn.close()
        _refused(world, "is in plaintext while a key loads: run `python3 -m opti_oignon.db_encryption encrypt`")
        for name in os.listdir(world.store.parent):
            if name.startswith("vectors.db"):
                os.unlink(world.store.parent / name)
        held = world.vdb.Client(world.store, "knowledge")
        held.create_collection("papers", metadata={"hnsw:space": "l2"}).add(ids=["other"], embeddings=[[1.0, 0.0]])
        _refused(world, "knowledge/papers is measured in l2 in .* and in cosine in")
        held.delete_collection("papers")
        _sql(package / "rag/chroma_v2", "UPDATE collections SET schema_str = json_set(schema_str, "
                                        "'$.defaults.float_list.vector_index.config.space', 'manhattan', "
                                        "'$.keys.\"#embedding\".float_list.vector_index.config.space', 'manhattan') "
                                        "WHERE name = 'papers'")
        _refused(world, "papers of .* is measured in manhattan, which the knowledge base's store does not measure")
        _golden(package, "rag/chroma_v2")
        insert = world.vdb.Collection._insert

        def altered(conn, row, item_id, vector, metadata, document):
            if item_id == "p3":
                vector = array("f", [value + 0.5 for value in vector])
            return insert(conn, row, item_id, vector, metadata, document)

        world.vdb.Collection._insert = staticmethod(altered)
        _refused(world, "p3 does not read back from .* as it was handed in")
        world.vdb.Collection._insert = staticmethod(insert)
        held.create_collection("papers", metadata={"hnsw:space": "l2"})
        assert world.cm.migrate_all()["moved"], "control: the same tree moves once each cause is gone"
        assert _held(world, "knowledge", "papers")[0] == "cosine", "an empty collection takes the space it is given"
        assert world.vdb.Client(world.store, "knowledge").get_collection("papers").metadata["hnsw:space"] == "cosine"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ10 -- what the platform wrote since is kept, a stuck run finished
# ---------------------------------------------------------------------------
def test_cz10_an_id_the_store_holds_is_kept_and_a_stuck_run_is_finished_by_the_next(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        folder = _store(package, "rag/chroma_v2", [("papers", {"hnsw:space": "cosine"}, [
            ("p1", "cats purr", {"page": 2}), ("p2", "dogs bark", None), ("p3", "birds sing", None),
            ("p4", "fish swim", None)])])
        # The memory's folder, read first: its facts never judge another namespace's items.
        _store(package, "rag/chroma_db", [("oo_memories", {"hnsw:space": "cosine"},
                                           [("f1", "Lives in Lyon", {"user_id": "local"})])])
        _facts(world, active=[("f1", "Lives in Lyon")])
        since = world.vdb.Client(world.store, "knowledge").create_collection("papers", metadata={"hnsw:space": "cosine"})
        since.add(ids=["p2"], embeddings=[[0.0, 0.0, 1.0, 0.0]], documents=["written since"])
        rename = _stuck(monkeypatch, world, folder)
        assert world.dbe.main(["migrate"]) == 3
        said = capsys.readouterr()
        assert f"the knowledge base's store holds every collection, but {folder} could not be moved" in said.err
        assert f"{folder}: 1 collection(s) written into the knowledge base's store, the folder not moved: the next " \
               "run of migrate moves it\n  papers, into the namespace knowledge: 3 item(s) written, their vectors " \
               "computed again with m-knowledge (4 dimensions), 1 already there kept\n" in said.out, said.out
        order = []

        class Buffered(io.StringIO):
            def flush(self):
                if self.getvalue():
                    order.append("summary")
                super().flush()

        class Errors(io.StringIO):
            def write(self, text):
                order.append("stopped" if text.startswith("stopped part way") else "other")
                return super().write(text)

        class Closed(io.TextIOBase):
            def write(self, text):
                raise BrokenPipeError(32, "Broken pipe")

            def fileno(self):
                raise io.UnsupportedOperation("fileno")

        with monkeypatch.context() as patch:
            patch.setattr(sys, "stdout", Buffered())
            patch.setattr(sys, "stderr", Errors())
            assert world.dbe.main(["migrate"]) == 3
        assert "summary" in order and order.index("summary") < order.index("stopped"), order
        with monkeypatch.context() as patch:
            patch.setattr(sys, "stdout", Closed())
            assert world.dbe.main(["migrate"]) == 3, "a stdout closed early costs neither the line nor the 3"
        assert "what this run did is said only up to here (BrokenPipeError: [Errno 32] Broken pipe)\nstopped part " \
               "way: " in capsys.readouterr().err
        with monkeypatch.context() as patch:
            patch.setattr(world.dbe, "_migration_summary", lambda done: (_ for _ in ()).throw(KeyboardInterrupt()))
            try:
                code = world.dbe.main(["migrate"])
            except KeyboardInterrupt:
                code = "the interrupt escaped"
            assert code == 3
        assert "what this run did is said only up to here (KeyboardInterrupt)\nstopped part way: " in (
            capsys.readouterr().err), "an interrupt while it is said: the same"
        space, items = _held(world, "knowledge", "papers")
        assert [i[0] for i in items] == ["p2", "p1", "p3", "p4"] and items[0][1] == "written since", (
            "what the platform wrote since is kept as it is", items)
        monkeypatch.setattr(world.cm.os, "rename", rename)
        assert world.dbe.main(["migrate"]) == 0
        assert "  papers, into the namespace knowledge: 0 item(s) written, 4 already there kept" in (
            capsys.readouterr().out)
        assert not folder.exists() and _held(world, "knowledge", "papers") == (space, items)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ11 -- a signal
# ---------------------------------------------------------------------------
def test_cz11_a_signal_before_the_commit_or_between_two_requests_changes_nothing(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        folder = _golden(package, "rag/chroma_v2")
        projects = _golden(package, "project_chroma")
        before, handler = _tree(*world.places), signal.getsignal(signal.SIGTERM)
        read = world.cm._read_store

        def reading(place):
            got = read(place)
            os.kill(os.getpid(), signal.SIGTERM)
            return got

        monkeypatch.setattr(world.cm, "_read_store", reading)
        assert world.dbe.main(["migrate"]) == 1
        err = capsys.readouterr().err
        assert "signal 15 noted before " in err and " was read: nothing was written" in err, err
        assert _tree(*world.places) == before and signal.getsignal(signal.SIGTERM) == handler, "nothing; put back"
        monkeypatch.setattr(world.cm, "_read_store", read)
        state, batch = world.embed, world.embed.batch

        def interrupting(model, texts):
            if texts[0] == "t0":
                os.kill(os.getpid(), signal.SIGTERM)
            return batch(model, texts)

        state.batch = interrupting
        assert world.dbe.main(["migrate"]) == 1
        err = capsys.readouterr().err
        assert "signal 15 noted while the vectors of bulk were computed: nothing was written" in err, err
        assert sum(1 for _model, texts in state.batches if texts[0] == "t0") == 1, "stopped at the next request"
        assert _tree(*world.places) == before and not world.store.exists()
        state.batch = batch
        load = world.cm.vector_db.load

        def loading(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGTERM)
            return load(*args, **kwargs)

        monkeypatch.setattr(world.cm.vector_db, "load", loading)
        assert world.dbe.main(["migrate"]) == 1
        assert "signal 15 noted before the knowledge base's store committed: nothing was written" in (
            capsys.readouterr().err)
        assert _tree(*world.places) == before and not world.store.exists()
        monkeypatch.setattr(world.cm.vector_db, "load", load)
        fresh = world.dbe._fresh_folder

        def after(place, name):
            os.kill(os.getpid(), signal.SIGTERM)
            return fresh(place, name)

        monkeypatch.setattr(world.dbe, "_fresh_folder", after)
        rename = _stuck(monkeypatch, world, folder)
        assert world.dbe.main(["migrate"]) == 3
        assert "signal 15 came after the commit: the run went on\nstopped part way: " in capsys.readouterr().err
        monkeypatch.setattr(world.cm.os, "rename", rename)
        assert world.dbe.main(["migrate"]) == 0
        assert "signal 15 came after the commit: the run finished" in capsys.readouterr().err
        assert not folder.exists() and not projects.exists()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ12 -- git takes no plaintext copy
# ---------------------------------------------------------------------------
def test_cz12_every_file_the_command_keeps_in_plaintext_is_one_git_ignores(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        repo, package = world.places
        for place, relative in ((repo, "rag/chroma_db"), (repo, "project_chroma"), (package, "rag/chroma_v2"),
                                (package, "rag/chroma_db"), (package, "project_chroma")):
            _store(place, relative, [("documents", None, [("d", "x", None)])])
        _facts(world)
        done = world.cm.migrate_all()
        assert len(done["moved"]) == 5, done
        world.dbe._STORE_PLACES = None
        real = [Path(place).relative_to(_REPO) for place in world.dbe._places()]
        rooms = {repo: real[0], package: real[1]}
        paths = []
        for entry in done["moved"]:
            kept = Path(entry["kept_in"]) / Path(entry["folder"]).name
            place = repo if repo in kept.parents else package
            paths += [str(rooms[place] / path.relative_to(place)) for path in sorted(kept.rglob("*")) if path.is_file()]
        assert len(paths) == 10, ("control: each folder's database and index", paths)
        probe = "opti_oignon/chroma_migration.py"
        answer = subprocess.run(["git", "-C", str(_REPO), "check-ignore", "--no-index", "-v", "-n", *paths, probe],
                                capture_output=True, text=True, timeout=60)
        rules = dict(reversed(line.split("\t", 1)) for line in answer.stdout.splitlines())

        def ignored(path):
            source_, _, rest = rules.get(path, "::").partition(":")
            return bool(source_) and not rest.partition(":")[2].startswith("!")

        assert not ignored(probe) and probe in rules, ("control: a source file is not ignored", answer.stdout,
                                                       answer.stderr)
        assert [path for path in paths if not ignored(path)] == [], ("git would take these", answer.stdout)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ13 -- what the memory still holds
# ---------------------------------------------------------------------------
def _memories():
    return [("oo_memories", {"hnsw:space": "cosine"}, [
        ("f-active", "Allergic to penicillin", {"user_id": "local"}), ("f-soft", "b", {"user_id": "local"}),
        ("f-gone", "c", {"user_id": "local"}), ("f-edited", "Allergic to amoxicillin", {"user_id": "local"})])]


def test_cz13_the_memory_keeps_only_what_an_active_fact_holds_as_it_reads(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        _store(package, "rag/chroma_db", _memories() + [("documents", None, [("d1", "doc", None)])])
        _facts(world, active=[("f-active", "Allergic to penicillin"), ("f-edited", "Allergic to aspirin"),
                              ("f-added", "Lives in Lyon")], inactive=[("f-soft", "b")])
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert [(i[0], i[3]) for i in _held(world, "memory", "oo_memories")[1]] == [
            ("f-active", _vector("m-rag", "Allergic to penicillin"))], (
            "a deleted fact's vector stays out, and a rewritten one's, which holds the old text")
        assert "  oo_memories, into the namespace memory: 1 item(s) written, their vectors computed again with " \
               "m-rag (4 dimensions), 0 already there kept, 3 left out: no active fact of the memory holds them " \
               "as they read now" in out, out
        assert "2 active fact(s) of the memory hold no vector of their text as it reads now: recall finds them by " \
               "their words" in out, out
        assert [i[0] for i in _held(world, "documents", "documents")[1]] == ["d1"], "the documents have no such owner"
        assert all(texts != ["b"] and "c" not in texts for _model, texts in world.embed.batches), (
            "a text no active fact holds is never sent to the embedder")
        _store(package, "rag/chroma_db", _memories())
        facts = package / "memory_facts.db"
        kept = facts.read_bytes()
        facts.write_bytes(b"no database of facts, only these words")
        _refused(world, "the memory's facts at .* do not read here")
        facts.unlink()
        done = world.cm.migrate_all()
        assert done["moved"][0]["left_out"] == 4 and done["moved"][0]["written"] == 0, (
            "no store of facts: no fact holds a vector")
        facts.write_bytes(kept)
    finally:
        world.restore()
    world = _world(tmp_path / "gone", monkeypatch)
    try:
        package = world.places[1]
        _store(package, "rag/chroma_db", [("oo_memories", {"hnsw:space": "cosine"},
                                           [("f1", "Lives in Lyon", {"user_id": "local"})])])
        _facts(world, inactive=[("f1", "Lives in Lyon")])
        held = world.vdb.Client(world.store, "memory").get_or_create_collection("oo_memories",
                                                                                metadata={"hnsw:space": "cosine"})
        held.add(ids=["f1"], embeddings=[[1.0, 0.0, 0.0, 0.0]])
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert _held(world, "memory", "oo_memories")[1] == [], "a copy with no text whose fact is gone: removed"
        assert ", 1 removed: the store held them with a text no active fact of the memory holds now" in out, out
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ14 -- a resumed run undoes what it must not keep
# ---------------------------------------------------------------------------
def test_cz14_a_resumed_run_removes_what_a_rewritten_or_deleted_fact_no_longer_holds(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        folder = _store(package, "rag/chroma_db", [("oo_memories", {"hnsw:space": "cosine"}, [
            ("f1", "Allergic to penicillin", {"user_id": "local"}), ("f2", "Lives in Lyon", {"user_id": "local"}),
            ("f3", "Plays the cello", {"user_id": "local"})])])
        _facts(world, active=[("f1", "Allergic to penicillin"), ("f2", "Lives in Lyon"), ("f3", "Plays the cello")])
        rename = _stuck(monkeypatch, world, folder)
        assert world.dbe.main(["migrate"]) == 3
        said = capsys.readouterr()
        assert [i[0] for i in _held(world, "memory", "oo_memories")[1]] == ["f1", "f2", "f3"], "control: committed"
        assert not [kept.name for kept in (package / "rag").iterdir() if kept.name.startswith("plaintext-")], (
            "the folder of plaintext copies made for it, gone again", said.out)
        _rewrite(world, "UPDATE memory_facts SET text = 'Allergic to aspirin' WHERE id = 'f1'")
        _rewrite(world, "UPDATE memory_facts SET active = 0 WHERE id = 'f2'")
        _rewrite(world, "INSERT INTO memory_facts (id, text, active) VALUES ('f4', 'Reads in Spanish', 1)")
        world.vdb.Client(world.store, "memory").get_collection("oo_memories").add(
            ids=["f4"], embeddings=[[0.5, 0.5, 0.5, 0.5]], documents=["Reads in Spanish"])
        monkeypatch.setattr(world.cm.os, "rename", rename)
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert [i[0] for i in _held(world, "memory", "oo_memories")[1]] == ["f3", "f4"], (
            "the rewritten fact's old vector and the deleted fact's are gone; what something else wrote stays")
        assert "  oo_memories, into the namespace memory: 0 item(s) written, 1 already there kept, 2 left out: no " \
               "active fact of the memory holds them as they read now, 2 removed: the store held them with a text " \
               "no active fact of the memory holds now" in out, out
        assert "1 active fact(s) of the memory hold no vector of their text as it reads now" in out, out
        assert not folder.exists()
    finally:
        world.restore()
    world = _world(tmp_path / "kept", monkeypatch)
    try:
        package = world.places[1]
        _store(package, "rag/chroma_db", [("oo_memories", {"hnsw:space": "cosine"},
                                           [("f1", "Allergic to penicillin", {"user_id": "local"})])])
        _facts(world, active=[("f1", "Allergic to aspirin"), ("f5", "Speaks German")])
        held = world.vdb.Client(world.store, "memory").get_or_create_collection("oo_memories",
                                                                                metadata={"hnsw:space": "cosine"})
        held.add(ids=["f1", "f5"], embeddings=[[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
                 documents=["Allergic to aspirin", "Speaks Dutch"])
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert _held(world, "memory", "oo_memories")[1] == [("f1", "Allergic to aspirin", None, [0.0, 1.0, 0.0, 0.0]),
                                                            ("f5", "Speaks Dutch", None, [1.0, 1.0, 0.0, 0.0])], (
            "the platform's vector for the fact as it reads now, kept; one no folder names, left as it is")
        assert "1 left out" in out and "removed" not in out, out
        folder = _store(package, "rag/chroma_db", [("oo_memories", {"hnsw:space": "cosine"},
                                                    [("f1", "Allergic to penicillin", {"user_id": "local"})])])
        found = world.cm.vector_db.Client.get_collection

        def gone(self, name):
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(world.cm.vector_db.Client, "get_collection", gone)
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        monkeypatch.setattr(world.cm.vector_db.Client, "get_collection", found)
        assert "the active facts of the memory that hold no vector of their text: not counted, the store did not " \
               "read back (OSError: [Errno 5] Input/output error)" in out, out
        assert not folder.exists()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ15 -- two sources of the memory's collection
# ---------------------------------------------------------------------------
def test_cz15_two_folders_holding_the_memory_move_together_the_current_text_kept(tmp_path, monkeypatch, capsys):
    def memories(text):
        return [("oo_memories", {"hnsw:space": "cosine"}, [("f1", text, {"user_id": "local"})])]

    world = _world(tmp_path, monkeypatch)
    try:
        repo, package = world.places
        for first, second in (("Allergic to aspirin", "Allergic to penicillin"),
                              ("Allergic to penicillin", "Allergic to aspirin")):
            _store(repo, "rag/chroma_db", memories(first))
            _store(package, "rag/chroma_db", memories(second))
            if not (package / "memory_facts.db").exists():
                _facts(world, active=[("f1", "Allergic to aspirin")])
            assert world.dbe.main(["migrate"]) == 0, capsys.readouterr().err
            capsys.readouterr()
            assert _held(world, "memory", "oo_memories")[1] == [
                ("f1", "Allergic to aspirin", {"user_id": "local"}, _vector("m-rag", "Allergic to aspirin"))], (
                "the current copy, whichever came first")
            world.vdb.Client(world.store, "memory").delete_collection("oo_memories")
    finally:
        world.restore()
    lyon = "  oo_memories, into the namespace memory: "
    left = "1 left out: no active fact of the memory holds them as they read now"
    gone = "1 removed: the store held them with a text no active fact of the memory holds now"
    world = _world(tmp_path / "deleted", monkeypatch)
    try:
        repo, package = world.places
        first = _store(repo, "rag/chroma_db", memories("Lives in Lyon"))
        second = _store(package, "rag/chroma_db", memories("Lives in Paris"))
        _facts(world, active=[("f1", "Lives in Lyon")])
        rename = _stuck(monkeypatch, world, second)
        assert world.dbe.main(["migrate"]) == 3
        said = capsys.readouterr()
        assert not first.exists() and second.is_dir(), "control: the first folder moved, the second stayed"
        assert f"{second}: 1 collection(s) written into the knowledge base's store, the folder not moved" in said.out
        _rewrite(world, "UPDATE memory_facts SET active = 0 WHERE id = 'f1'")
        monkeypatch.setattr(world.cm.os, "rename", rename)
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert _held(world, "memory", "oo_memories")[1] == [], "the deleted fact's copy, made from the other folder"
        assert f"{lyon}0 item(s) written, 0 already there kept, {left}, {gone}\n" in out, out
        assert not second.exists()
    finally:
        world.restore()
    world = _world(tmp_path / "reverted", monkeypatch)
    try:
        repo, package = world.places
        first = _store(repo, "rag/chroma_db", memories("Lives in Lyon"))
        second = _store(package, "rag/chroma_db", memories("Lives in Paris"))
        _facts(world, active=[("f1", "Lives in Paris")])
        rename = _stuck(monkeypatch, world, first)
        assert world.dbe.main(["migrate"]) == 3
        capsys.readouterr()
        assert first.is_dir() and second.is_dir(), "control: stopped at the first folder"
        _rewrite(world, "UPDATE memory_facts SET text = 'Lives in Lyon' WHERE id = 'f1'")
        monkeypatch.setattr(world.cm.os, "rename", rename)
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert _held(world, "memory", "oo_memories")[1] == [
            ("f1", "Lives in Lyon", {"user_id": "local"}, _vector("m-rag", "Lives in Lyon"))], (
            "the copy of the text the fact reads now")
        assert f"{lyon}1 item(s) written, their vectors computed again with m-rag (4 dimensions), 0 already " \
               f"there kept, {gone}\n" in out, out
        assert "hold no vector" not in out and not first.exists() and not second.exists()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ16 -- no half run
# ---------------------------------------------------------------------------
def test_cz16_a_holder_seen_before_the_commit_and_every_failure_a_refusal_by_name(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        holders, calls = world.dbe.store_holders, []

        def late_holder(paths, proc=None):
            calls.append(list(paths))
            return {4242: "runs the platform (uvicorn)"} if len(calls) == 2 else holders(paths, proc=proc)

        _store(world.places[1], "project_chroma", [("project_p", None, [("p", "x", None)])])
        monkeypatch.setattr(world.dbe, "store_holders", late_holder)
        _refused(world, "stop the platform first \\(process 4242 runs the platform \\(uvicorn\\)\\)")
        assert len(calls) == 3 and calls[2] == [world.store], ("read again before the commit, then the store it "
                                                                "made removed once no one holds it", calls)
        assert not world.store.parent.exists(), "the folder the run made, removed"
        monkeypatch.setattr(world.dbe, "store_holders", holders)
        _golden(world.places[1], "rag/chroma_v2")
        broken = world.places[1] / "rag/chroma_v2/chroma.sqlite3"
        broken.write_bytes(broken.read_bytes()[:4096])
        _refused(world, "chroma.sqlite3 does not read as a ChromaDB database \\(DatabaseError: ")
        _golden(world.places[1], "rag/chroma_v2")
        world.dbe._get_db_encryption_key = lambda: _OTHER_KEY
        world.vdb.Client(world.store, "knowledge").create_collection("other")
        world.dbe._get_db_encryption_key = lambda: _KEY
        _refused(world, "the knowledge base's store at .* took nothing \\(RuntimeError: SQLCipher key verification")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ17 -- what waits
# ---------------------------------------------------------------------------
def test_cz17_what_waits_moves_as_the_place_spells_it(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path / "plaintext-archive", monkeypatch)
    try:
        folder = _golden(world.places[1], "project_chroma")
        assert world.cm.sources()[0] == [(folder, "projects")]
        assert world.dbe._migration_hint(folder / "chroma.sqlite3") is not None, "check names the command for it"
        assert world.dbe.main(["migrate"]) == 0 and not folder.exists()
        assert [i[0] for i in _held(world, "projects", "project_p1")[1]] == ["p1::a::0", "p1::b::0"]
    finally:
        world.restore()
    capsys.readouterr()
    world = _world(tmp_path / "spelled", monkeypatch)
    try:
        package = world.places[1]
        (tmp_path / "spelled" / "alias").symlink_to(package, target_is_directory=True)
        for spelling in (package / "rag" / ".." / "rag", tmp_path / "spelled" / "alias" / "rag"):
            folder = _store(package, "rag/chroma_db", [("documents", None, [("d", "x", None)])])
            monkeypatch.setenv("RAG_HOME", str(spelling))
            assert world.dbe.main(["migrate"]) == 0, (spelling, capsys.readouterr().err)
            capsys.readouterr()
            assert not folder.exists(), spelling
            assert world.vdb.chroma_state(spelling / "chroma_db") == ("none", ""), "the platform's folder moved"
    finally:
        world.restore()
    world = _world(tmp_path / "linked", monkeypatch)
    try:
        package = world.places[1]
        real = tmp_path / "linked" / "big-disk" / "rag"
        _store(real.parent, "rag/chroma_db", [("documents", None, [("d", "x", None)])])
        (package / "rag").symlink_to(real, target_is_directory=True)
        monkeypatch.setenv("RAG_HOME", str(real))
        assert world.dbe.main(["migrate"]) == 0, capsys.readouterr().err
        capsys.readouterr()
        assert world.vdb.chroma_state(real / "chroma_db") == ("none", ""), "the platform's folder moved"
        assert [i[0] for i in _held(world, "documents", "documents")[1]] == ["d"]
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ19 -- said by check, one run at a time
# ---------------------------------------------------------------------------
def test_cz19_check_names_the_command_for_the_stores_it_moves_and_one_run_at_a_time(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        repo, package = world.places
        known = _golden(package, "rag/chroma_db")
        unknown = _store(package, "other_chroma", [])
        _store(package, "rag/plaintext-20260101T000000Z/chroma_v2", [])
        exported = _store(package, "exports/plaintext-20260101T000000Z/chroma_v2", [])
        assert exported / "chroma.sqlite3" in world.dbe.plaintext_others(), "control: check lists that copy"
        _facts(world, active=[("m1", "a fact")])
        assert world.dbe.main(["check"]) == 0
        out = capsys.readouterr().out
        hint = ("`python3 -m opti_oignon.db_encryption migrate`, the platform stopped, moves its collections into the "
                "knowledge base's store, encrypted with the other stores")
        assert f"{known}: {hint}" in out, out
        assert f"{unknown}/chroma.sqlite3: a ChromaDB vector store" in out and f"{unknown}: {hint}" not in out
        assert out.count(hint) == 1, "never for a folder of plaintext copies"
        assert world.dbe.MIGRATE_COMMAND == "python3 -m opti_oignon.db_encryption migrate"
        fd = os.open(repo, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = _tree(*world.places)
            assert world.dbe.main(["migrate"]) == 1
            assert "another run of encrypt or migrate is under way: nothing was written" in capsys.readouterr().err
            assert _tree(*world.places) == before
        finally:
            os.close(fd)
        assert world.dbe.main(["migrate"]) == 0 and not known.exists()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CZ18 -- the real chain
# ---------------------------------------------------------------------------
_REAL = r'''
import array, hashlib, json, os, socket, sqlite3, sys
from pathlib import Path
repo, root, golden = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
for name in ("OPTI_ENCRYPTION_KEY", "RAG_HOME"):
    os.environ.pop(name, None)
os.environ["OPTI_KEYFILE_PASSPHRASE"] = "a phrase only this contract knows"


def refuse(*args, **kwargs):
    raise OSError("the network is refused in this contract")


socket.socket.connect = refuse
socket.socket.connect_ex = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse
sys.meta_path[:] = [finder for finder in sys.meta_path if "editable" not in repr(finder).lower()]
sys.path.insert(0, repo)
from opti_oignon import (chroma_migration as cm, db_encryption as dbe, encryption as enc, security_mode as sm,
                         signed_audit_log as sal, vector_db as vdb)
from opti_oignon.rag import config as ragcfg, embeddings as emb
modules = (cm, dbe, enc, sm, sal, vdb, emb, ragcfg)
assert all(module.__file__.startswith(repo) for module in modules), [module.__file__ for module in modules]
# The RAG configuration makes its folders as it is made: in this run's own, never the package's place of data.
home = root / "rag-home"
ragcfg.set_config(ragcfg.RAGConfig(rag_home=home, cache_dir=home / "cache", logs_dir=home / "logs"))


def vector(model, text):
    digest = hashlib.sha256(f"{model}\0{text}".encode("utf-8")).digest()
    return [(byte - 128) / 64 for byte in digest[:4]]


def stored(namespace, model, text):
    values = vector(model, text)
    if namespace == "projects":
        norm = sum(x * x for x in values) ** 0.5
        values = [x / norm for x in values]
    return [float(x) for x in array.array("f", values)]


# The embedder's class itself replaced: whatever the migration embeds goes through it, or meets the refused socket.
emb.OllamaEmbeddings._verify_model = lambda self: True
emb.OllamaEmbeddings.embed_batch = lambda self, texts: [vector(self.config.model, text) for text in texts]
emb.OllamaEmbeddings.embed_single = lambda self, text: vector(self.config.model, text)
emb._backend_for = lambda model: None
data, package, proc = root / "data", root / "package" / "data", root / "proc"
for folder in (data, package, proc):
    folder.mkdir(parents=True, exist_ok=True)
dbe._STORE_PLACES, dbe._LONE_STORES, dbe._PROC = (data, package), [], proc
enc._DEFAULT_KEYFILE = sm._DEFAULT_KEYFILE = data / ".keyfile"
sm._DATA_DIR, sm._LOCKFILE_PATH, sm._SECURITY_YAML = data, data / ".security_mode_lock", root / "security.yaml"
sal.signed_audit_log = sal.SignedAuditLog(data / "audit_chain.db")
sm.security_mode_manager = sm.SecurityModeManager()
# The key file wrapped under the passphrase; the mode Daily, from security.yaml alone.
enc.save_keyfile(bytes(range(1, 33)), kdf_name="random")
sm._SECURITY_YAML.write_text("security_mode: daily\n", encoding="utf-8")
# The memory's facts, encrypted under the key that loads; the golden stores rebuilt from their dumps.
conn = vdb.safe_connect(package / "memory_facts.db")
conn.execute("CREATE TABLE memory_facts (id TEXT PRIMARY KEY, text TEXT NOT NULL, active INTEGER NOT NULL)")
conn.execute("INSERT INTO memory_facts VALUES ('m1', 'a fact', 1)")
conn.commit()
conn.close()
for relative in ("rag/chroma_v2", "rag/chroma_db", "project_chroma"):
    (package / relative).mkdir(parents=True)
    rebuilt = sqlite3.connect(str(package / relative / "chroma.sqlite3"))
    rebuilt.executescript((golden / relative / "chroma.sql").read_text(encoding="ascii"))
    rebuilt.close()
# Then the lockfile, signed under the key before this one, as on the machine; the command starts afresh.
sm._write_lockfile("daily", "leon", bytes(range(32)))
sm.security_mode_manager = sm.SecurityModeManager()
code = dbe.main(["migrate"])
events = [event["event_type"] for event in sal.signed_audit_log.get_events(limit=50)]
held = {}
target = package / "rag" / vdb.FILENAME
for namespace, name in (("knowledge", "papers"), ("knowledge", "inner"), ("documents", "documents"),
                        ("memory", "oo_memories"), ("projects", "project_p1")):
    collection = vdb.Client(target, namespace).get_collection(name)
    got = collection.get(include=["documents", "embeddings"])
    held[f"{namespace}/{name}"] = [collection.space, got["ids"],
                                   all(list(v) == stored(namespace, m, t) for v, t, m in zip(
                                       got["embeddings"], got["documents"],
                                       [cm._module_embedder(namespace).config.model] * len(got["ids"])))]
print("RESULT " + json.dumps({"code": code, "events": events, "chain": list(sal.signed_audit_log.verify_chain()),
                              "mode": sm.get_current_mode(), "held": held, "head": target.read_bytes()[:16].hex()}))
'''


def test_cz18_migrate_with_the_real_mode_and_chain_on_the_golden_stores(tmp_path):
    pytest.importorskip("sqlcipher3")
    # Its home in a temporary folder, its user site kept: where this interpreter finds numpy.
    env = dict(os.environ, HOME=str(tmp_path / "home"), PYTHONUSERBASE=site.getuserbase(), PYTHONDONTWRITEBYTECODE="1")
    (tmp_path / "home").mkdir()
    try:
        child = subprocess.run([sys.executable, "-c", _REAL, str(_REPO), str(tmp_path / "root"), str(_GOLDEN)],
                               capture_output=True, text=True, timeout=180, env=env)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"the run never ended (stdout: {str(exc.stdout)[-1500:]!r}; stderr: {str(exc.stderr)[-1500:]!r})")
    lines = [line for line in child.stdout.splitlines() if line.startswith("RESULT ")]
    assert child.returncode == 0 and lines, (child.returncode, child.stdout[-2000:], child.stderr[-3000:])
    result = json.loads(lines[-1][len("RESULT "):])
    assert result["code"] == 0 and result["mode"] == "bulbe", result
    assert result["events"] == ["lockfile_hmac_failure"] and result["chain"] == [True, None, 1], result
    assert bytes.fromhex(result["head"]) != _PLAIN, "the store is encrypted"
    expected = {entry["name"]: entry for listed in _expected()["stores"].values() for entry in listed}
    for key, (space, ids, paired) in result["held"].items():
        name = key.split("/", 1)[1]
        want = [i for i, text in zip(expected[name]["ids"], expected[name]["documents"]) if text]
        if name == "oo_memories":
            want = ["m1"]
        assert space == expected[name]["configuration"]["hnsw"]["space"] and ids == want and paired, (key, result)
    for folder in ("chroma_v2", "chroma_db", "project_chroma"):
        assert f"{folder}: " in child.stdout and "collection(s) moved into the knowledge base's store" in child.stdout
    assert "encrypted under the key that loads" in child.stdout, child.stdout
    assert child.stderr.count("Lockfile HMAC verification FAILED") <= 1, child.stderr[-3000:]
