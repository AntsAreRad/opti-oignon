"""The Mycelium's store: one SQLite file per corpus, opened through ``db_utils.safe_connect`` (SQLCipher under the
platform's key; refused in plaintext in Bulbe), or through the ``connect`` a caller hands in -- never a connection of
the store's own making.

Tables: ``source`` (a corpus: its name, root, kind, sensitivity), ``file`` (path, size, mtime_ns, sha256 of the
content, language, the first sentence it opens on), ``part`` (kind, symbol, lines, sha256 of its text and of the
text it is embedded as), ``vector`` (by the text embedded and the encoder: parts that embed the same text share one;
float16 unit vectors), ``fts`` (FTS5 without content: the index keeps tokens, never the text, which stays in the
source and is read again by path and lines), ``symbol`` (what a part defines), ``ref`` (what a part names: the
links between parts are ``ref`` joined to ``symbol``, so a removed part leaves no link behind), ``episode`` (each
indexing pass, and what it did).

What the store keeps of a source, then, is not its text but: each file's first sentence (its module docstring's, or
its heading's), a Markdown section's heading path, the names a part defines and names, and the index's tokens with
their positions -- from which a part's stemmed words could be read again in order --, and each part's vector, from
which a model built for it can guess much of the text it was embedded from. The index keeps no text, but a
deleted row is only marked until its segment is merged, and freed pages are zeroed (``PRAGMA secure_delete``): after
a pass that removes a file the index is merged by a bounded number of pages (``Store.purge``), so the removed words
leave the store's bytes as it merges -- with that pass, in a small store -- and a changed file's old words at the
index's next merge. ``Store.purge(None)`` merges it whole.

A store written by a newer schema is refused by name, and so is a database of another kind (tables, no store's meta
table: never stamped); a store cut by another version of the cutter is cut again at the next pass, and so is a store
of schema 1 (its parts had no embedded text yet). The active encoder's card is kept in ``meta``: the dense stage
reads its vectors, and only its, until another encoder covers every part. An episode holds counts, the meta table
its own keys: neither takes a source's words. One pass at a time writes (``lock``)."""
import json
import logging
import os
import re
import threading
import time

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
TOKENIZE = "porter unicode61 remove_diacritics 2 tokenchars '_'"
SENSITIVITIES = ("public", "private", "secret")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS source(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, root TEXT NOT NULL, kind TEXT NOT NULL,
  sensitivity TEXT NOT NULL CHECK (sensitivity IN {SENSITIVITIES!r}), generation INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS file(
  id INTEGER PRIMARY KEY, source INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE, path TEXT NOT NULL,
  size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL, lang TEXT NOT NULL,
  generation INTEGER NOT NULL, module_doc TEXT NOT NULL DEFAULT '', UNIQUE(source, path));
CREATE INDEX IF NOT EXISTS file_sha ON file(sha256);
CREATE TABLE IF NOT EXISTS part(
  id INTEGER PRIMARY KEY, file INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE, kind TEXT NOT NULL,
  symbol TEXT NOT NULL, first INTEGER NOT NULL, last INTEGER NOT NULL, sha256 TEXT NOT NULL,
  embed_sha TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS part_file ON part(file);
CREATE INDEX IF NOT EXISTS part_embed ON part(embed_sha);
CREATE TABLE IF NOT EXISTS vector(
  embed_sha TEXT NOT NULL, encoder TEXT NOT NULL, dim INTEGER NOT NULL, data BLOB NOT NULL,
  PRIMARY KEY(embed_sha, encoder));
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(path, body, words, said, content='', contentless_delete=1,
  tokenize="{TOKENIZE}");
CREATE TABLE IF NOT EXISTS symbol(
  part INTEGER NOT NULL REFERENCES part(id) ON DELETE CASCADE, name TEXT NOT NULL, qualname TEXT NOT NULL,
  kind TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS symbol_name ON symbol(name);
CREATE INDEX IF NOT EXISTS symbol_part ON symbol(part);
CREATE TABLE IF NOT EXISTS ref(
  part INTEGER NOT NULL REFERENCES part(id) ON DELETE CASCADE, name TEXT NOT NULL, kind TEXT NOT NULL,
  module TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ref_name ON ref(name);
CREATE INDEX IF NOT EXISTS ref_part ON ref(part);
CREATE TABLE IF NOT EXISTS episode(
  id INTEGER PRIMARY KEY, at TEXT NOT NULL, action TEXT NOT NULL, source INTEGER, detail TEXT NOT NULL);
"""


class StoreError(RuntimeError):
    """The store cannot be used as asked."""


def _safe_connect(path, read_only=False):
    from opti_oignon.db_utils import safe_connect

    if read_only:
        return safe_connect(path, check_same_thread=False, read_only=True)
    return safe_connect(path, check_same_thread=False)


# The tables a store makes, the full-text index's own included: a database holding any other is not a store.
OWN_TABLES = {"meta", "source", "file", "part", "symbol", "ref", "episode", "vector", "fts", "fts_config",
              "fts_data", "fts_docsize", "fts_idx"}
# The meta table's keys: the schema, the vectors' count of changes and the active encoder's card, and per source the
# cutter that cut it, the header its texts are embedded with and when its last pass began.
META_KEY = re.compile(r"(schema|vectors|encoder|cutter:\d+|header:\d+|pass:\d+)\Z")
# The longest value the meta table keeps: the active encoder's card, as stored (``encoder_value``), is the longest.
MAX_VALUE = 4096


def encoder_value(key, card):
    """The active encoder as the meta table keeps it."""
    return json.dumps({"key": key, "card": card}, sort_keys=True)


class Store:
    def __init__(self, path, connect=None, read_only=False):
        """The store at ``path``, made when new; with ``read_only``, an existing store read and nothing written --
        opened through the platform's read-only connection (or ``connect``, which must open it so), no table made,
        no stamp, no migration: a missing file, a newer schema, a schema-1 store and a database of another kind are
        refused by name, left as they were."""
        self.path = str(path)
        self.read_only = read_only
        self.lock = threading.RLock()
        if read_only:
            if not os.path.isfile(self.path):
                raise StoreError(f"no store at {self.path}")
            try:
                self.db = connect(self.path) if connect else _safe_connect(self.path, read_only=True)
            except Exception as exc:  # noqa: BLE001 - a wrong key, a hot journal, a file sealed otherwise: said
                raise StoreError(f"{self.path} cannot be opened to be read: {exc}") from exc
            try:
                self._readable()
            except BaseException:
                self.db.close()
                raise
            return
        self.db = (connect or _safe_connect)(self.path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA secure_delete = ON")
        held = self._version()
        if held is not None and held > SCHEMA_VERSION:
            raise StoreError(f"the store is at schema {held}, this code reads {SCHEMA_VERSION} and before")
        if held == 1:
            # Schema 1 had no embedded text: the column comes, and every source is cut again at its next pass; in
            # one transaction with the new stamp, so a crash leaves schema 1 whole, and the schema read again inside
            # it, so a store another opener moved meanwhile is not moved twice.
            self.db.execute("BEGIN IMMEDIATE")
            if self._version() == 1:
                self.db.execute("ALTER TABLE part ADD COLUMN embed_sha TEXT NOT NULL DEFAULT ''")
                self.db.execute("DELETE FROM meta WHERE key LIKE 'cutter:%'")
                self.db.execute("UPDATE meta SET value = '2' WHERE key = 'schema'")
            self.db.commit()
        tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if held is None and tables - OWN_TABLES:
            raise StoreError(f"{self.path} holds tables a store does not make ({', '.join(sorted(tables - OWN_TABLES))}"
                             f"): not a Mycelium store, left untouched")
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
        self.db.commit()

    @classmethod
    def at(cls, path, read_only=False):
        """The store at ``path``, as ``Store(path, read_only=read_only)``: the package makes a Mycelium store through
        this. The garden's census of its own store (AF13 of tests/test_allium_surface_contracts.py) counts every call
        spelled ``Store(...)`` in the package as the garden's, and this class is another: so spelled, the census stays
        closed and exact."""
        return cls(path, read_only=read_only)

    def _readable(self):
        """A store opened read-only: refused unless it is one this code reads as it is."""
        held = self._version()
        if held is None:
            raise StoreError(f"{self.path} has no schema stamp: not a Mycelium store, left untouched")
        if held > SCHEMA_VERSION:
            raise StoreError(f"the store is at schema {held}, this code reads {SCHEMA_VERSION} and before")
        if held < SCHEMA_VERSION:
            raise StoreError(f"the store is at schema {held}: it is read once a pass has written it again")
        tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if tables - OWN_TABLES or "source" not in tables:
            raise StoreError(f"{self.path} holds tables a store does not make, or lacks its own: not a Mycelium "
                             f"store, left untouched")

    def _version(self):
        try:
            row = self.db.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()
        except Exception as exc:  # noqa: BLE001 - the drivers' error classes differ; the message says which
            if "no such table" in str(exc):
                return None     # a new file: no meta table yet
            raise StoreError(f"the store's schema cannot be read: {exc}") from exc
        return int(row[0]) if row else None

    def meta(self, key, value=None):
        """The value kept under ``key``, one of the store's own keys (``META_KEY``); with ``value`` (short), that
        value kept."""
        if not META_KEY.match(str(key)):
            raise StoreError(f"{key!r} is not a key the store keeps")
        if value is not None and len(str(value)) > MAX_VALUE:
            raise StoreError(f"the value under {key!r} is longer than a store's setting")
        if value is not None:
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, str(value)))
            return str(value)
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def purge(self, pages=None):
        """The full-text index merged -- by at most ``pages`` pages, even with few segments, or whole when None --,
        so what was deleted from the merged segments is gone from their pages, not only marked."""
        if pages is None:
            self.db.execute("INSERT INTO fts(fts) VALUES ('optimize')")
        else:
            self.db.execute("INSERT INTO fts(fts, rank) VALUES ('merge', ?)", (-int(pages),))

    def close(self):
        self.db.close()

    def source(self, name, root, kind="folder", sensitivity=None):
        """The id of the source ``name``, made if new (private unless a sensitivity is named); a known name keeps its
        root and its kind, and its sensitivity unless another is named -- taken then at once: its generation moves,
        so no reader keeps what it allowed before."""
        with self.lock:     # the row and the commit wait for a pass that holds the store
            if sensitivity is not None and sensitivity not in SENSITIVITIES:
                raise StoreError(f"sensitivity is one of {', '.join(SENSITIVITIES)}, not {sensitivity!r}")
            row = self.db.execute("SELECT id, root, kind, sensitivity FROM source WHERE name = ?", (name,)).fetchone()
            if row:
                if row[1] != str(root):
                    raise StoreError(f"source {name!r} is rooted at {row[1]}, not {root}")
                if row[2] != kind:
                    raise StoreError(f"source {name!r} is a {row[2]} source, not a {kind} one")
                if sensitivity is not None and row[3] != sensitivity:
                    self.db.execute("UPDATE source SET sensitivity = ?, generation = generation + 1 WHERE id = ?",
                                    (sensitivity, row[0]))
                    self.db.commit()
                return row[0]
            cur = self.db.execute("INSERT INTO source(name, root, kind, sensitivity) VALUES (?, ?, ?, ?)",
                                  (name, str(root), kind, sensitivity or "private"))
            self.db.commit()
            return cur.lastrowid

    def forget(self, name):
        """The source ``name`` forgotten, in one transaction under the lock: its parts' tokens, its row (its files,
        parts, names and references go with it), the vectors no part embeds any more, its own settings and episodes;
        an episode 'forget' with its counts. Then the index is merged whole and the file rewritten (``VACUUM``), so
        its words leave the store's bytes -- the pages a deleted row leaves zeroed, but the copies of rows the tables
        moved while they were written stay in their pages' free space until the file is rewritten. The vectors'
        count of changes moves even when no vector went: the generation never comes back to a value read before,
        though SQLite may give a name indexed again the same id. An unknown name is refused. Returns the counts and
        ``merged``: False when the merge or the rewrite failed (a full disk, a statement still running on the
        connection, a reader that came between the two), said in the log -- the source is forgotten, its words stay
        in the bytes until the next one succeeds. A reader in the middle of a statement when the forgetting begins
        makes the forgetting itself wait, then fail with the driver's error (locked), nothing forgotten. In a store
        kept in plaintext, the rollback journals of both hold the forgotten words, and are removed, not
        overwritten."""
        with self.lock:
            row = self.db.execute("SELECT id FROM source WHERE name = ?", (name,)).fetchone()
            if row is None:
                raise StoreError(f"no source named {name!r} in this store")
            source = row[0]
            try:
                parts = [r[0] for r in self.db.execute(
                    "SELECT p.id FROM part p JOIN file f ON f.id = p.file WHERE f.source = ?", (source,))]
                files = self.db.execute("SELECT COUNT(*) FROM file WHERE source = ?", (source,)).fetchone()[0]
                self.db.executemany("DELETE FROM fts WHERE rowid = ?", [(i,) for i in parts])
                self.db.execute("DELETE FROM source WHERE id = ?", (source,))
                self.db.executemany("DELETE FROM meta WHERE key = ?",
                                    [(f"{kind}:{source}",) for kind in ("cutter", "header", "pass")])
                self.db.execute("DELETE FROM episode WHERE source = ?", (source,))
                vectors = self.orphan_vectors()
                self.vectors_changed()
                counts = {"files": files, "parts": len(parts), "vectors": vectors}
                self.episode("forget", None, counts)
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            try:
                self.purge(None)
                self.db.commit()
                self.db.execute("VACUUM")
                counts["merged"] = True
            except Exception as exc:  # noqa: BLE001 - the source is forgotten; its words wait for the next merge
                self.db.rollback()
                counts["merged"] = False
                logger.warning("mycelium: the store was not merged and rewritten after a source was forgotten, its "
                               "words stay in the store's pages until it is (%s)", exc)
            return counts

    def sources(self):
        return [dict(zip(("id", "name", "root", "kind", "sensitivity", "generation"), row))
                for row in self.db.execute("SELECT id, name, root, kind, sensitivity, generation FROM source")]

    def episode(self, action, source, detail):
        """A pass's record: its action and its counts -- numbers only, never a source's words."""
        if not isinstance(detail, dict) or not all(isinstance(k, str) and type(v) in (int, float, bool)
                                                   for k, v in detail.items()):
            raise StoreError("an episode holds counts (numbers), never text")
        self.db.execute("INSERT INTO episode(at, action, source, detail) VALUES (?, ?, ?, ?)",
                        (time.strftime("%Y-%m-%dT%H:%M:%S"), action, source, json.dumps(detail, sort_keys=True)))

    def episodes(self, source=None, limit=20):
        rows = self.db.execute("SELECT id, at, action, source, detail FROM episode WHERE ? IS NULL OR source = ? "
                               "ORDER BY id DESC LIMIT ?", (source, source, limit)).fetchall()
        return [{"id": r[0], "at": r[1], "action": r[2], "source": r[3], "detail": json.loads(r[4])} for r in rows]

    def drop_parts(self, file_id):
        """Every part of a file, its tokens, symbols and references (its vectors stay while another part embeds the
        same text: ``orphan_vectors``)."""
        ids = [r[0] for r in self.db.execute("SELECT id FROM part WHERE file = ?", (file_id,))]
        self.db.executemany("DELETE FROM fts WHERE rowid = ?", [(i,) for i in ids])
        self.db.execute("DELETE FROM part WHERE file = ?", (file_id,))

    def orphan_vectors(self):
        """Vectors no part embeds any more: dropped."""
        dropped = self.db.execute("DELETE FROM vector WHERE embed_sha NOT IN (SELECT embed_sha FROM part)").rowcount
        if dropped:
            self.vectors_changed()
        return dropped

    def vectors_changed(self):
        """A vector written or dropped: readers that hold a matrix read it again."""
        self.meta("vectors", int(self.meta("vectors") or 0) + 1)

    def generation(self):
        """A value that changes whenever a part, a vector, a source's sensitivity or the active encoder changes:
        each source's own generation (a sum could be met again by two sources moving), the vectors' count of
        changes, the encoder's card."""
        sources = tuple(self.db.execute("SELECT id, generation, sensitivity FROM source ORDER BY id"))
        return sources, self.meta("vectors"), self.meta("encoder")

    def active_encoder(self):
        """{key, card} of the encoder whose vectors the dense stage reads, or None."""
        held = self.meta("encoder")
        return json.loads(held) if held else None

    def activate(self, key, card):
        """The encoder kept under ``key`` becomes the one read; every other encoder's vectors go."""
        self.meta("encoder", encoder_value(key, card))
        self.db.execute("DELETE FROM vector WHERE encoder != ?", (key,))
        self.vectors_changed()

    def stats(self):
        def one(query):
            return self.db.execute(query).fetchone()[0]

        return {"sources": one("SELECT COUNT(*) FROM source"), "files": one("SELECT COUNT(*) FROM file"),
                "parts": one("SELECT COUNT(*) FROM part"), "vectors": one("SELECT COUNT(*) FROM vector"),
                "symbols": one("SELECT COUNT(*) FROM symbol"),
                "refs": one("SELECT COUNT(*) FROM ref"), "episodes": one("SELECT COUNT(*) FROM episode")}
