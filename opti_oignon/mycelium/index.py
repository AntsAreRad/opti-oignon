"""Indexing a source into the Mycelium's store: never everything again, only what changed.

A pass lists the source's files (what git's index tracks for a repository, read from the index file itself
(``tracked``), never a process; else a walk; a ``data/`` never; a symbolic link, or a path that resolves outside the
source's root, never), and for each: the same size and mtime_ns -> untouched; else its sha256: the same -> only its
times are kept; a file gone whose sha256 shows up under a new path -> renamed (its row kept, cut again under its new
name); not text any more -> removed; else cut again (``chunk.cut``), its old parts dropped and the new ones written
with their tokens, definitions and references. A store cut by another version of the cutter cuts every file again.
The pass is one episode, written with what it did and how long it took.

A source that is not a folder -- the onion's Cellar, notes, an archive -- is indexed the same way from
``{doc_id: text}`` (``index_documents``), its citations checked through the reader it leaves behind.

Vectors are a pass of their own (``embed``), after the index: the parts whose embedded text has no vector under
the encoder, read again from their source (a part whose lines changed since is left to the next index pass),
embedded the shortest first, each batch written as it comes -- unless its text left the store while the engine
answered. The store's lock is held to plan the pass and to write each batch, never while the engine answers. A
pass held by the governor, or one whose encoder is missing or fails, keeps what it wrote and says so in its
episode (a flag; why, in the log and ``vector_status``); the next one takes up the rest. While another encoder is
read, the encoder becomes the one the search reads once a pass leaves no part without a vector under it; then the
other encoder's vectors go. On a store with no encoder read, it becomes the one read once it covers some part and
no part lacks a vector but the texts the engine refused in that pass (counted)."""
import hashlib
import logging
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from .chunk import CHUNKER_VERSION, cut, language, module_doc
from .lexis import subwords
from .tracked import tracked_files
from .vectors import EncoderMissing, Held, embedded_text, text_sha, to_blob

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2", ".ttf", ".zip", ".gz", ".pdf",
                 ".bin", ".so", ".jar", ".keystore", ".db", ".sqlite", ".myc", ".npy", ".pyc", ".lock")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "data", "build", "dist", ".svelte-kit",
             "target", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
MAX_BYTES = 2_000_000
DOCUMENTS = "documents:"
# Files that hold secrets by their name or their kind: never listed, tracked or not.
SECRET_NAMES = {".env", ".envrc", ".netrc", ".npmrc", ".pypirc", ".git-credentials", ".pgpass", ".htpasswd", "id_rsa",
                "id_ed25519", "id_ecdsa", "id_dsa", "credentials.json", "secrets.yml", "secrets.yaml"}
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks", ".der", ".kdbx", ".gpg", ".asc", ".tfstate",
                   ".tfstate.backup", ".tfvars")
# Folders every file of which is a secret: cloud and cluster credentials, SSH and GnuPG keys, a registry's logins.
SECRET_DIRS = {".aws", ".ssh", ".gnupg", ".kube", ".docker"}
# After a pass that removed files, the index is merged by at most this many pages (outside the pass): the removed
# words leave the store's bytes as the index merges, at a cost bounded whatever the store's size.
MERGE_PAGES = 4000
# A pass trusts a file's size and time to say it is unchanged only when the time is older than the pass before by
# this much: a file rewritten at the same size within the clock's tick -- or under a filesystem that keeps times to
# the second or two -- is read again (git's "racily clean" files).
RACY_NS = 2_000_000_000


class RootUnreadable(RuntimeError):
    """A source's root that cannot be listed (gone, unmounted, not readable): never read as an empty source."""


# A document of a source this indexer never held whole, and was not given: its text is not known here.
UNKNOWN = type("Unknown", (), {"__slots__": (), "__repr__": lambda self: "UNKNOWN"})()


def secret_name(rel):
    name = Path(rel).name.lower()
    return (name in SECRET_NAMES or name.startswith(".env.") or name.endswith(SECRET_SUFFIXES)
            or (name.startswith("client_secret") and name.endswith(".json"))
            or any(part.lower() in SECRET_DIRS for part in Path(rel).parts[:-1]))


def kept(rel):
    """Whether a listing keeps ``rel``: no skipped kind of file, no secret, nothing under a skipped folder."""
    return (not rel.lower().endswith(SKIP_SUFFIXES) and not secret_name(rel)
            and not any(part in SKIP_DIRS for part in Path(rel).parts[:-1]))


def repository_top(root):
    """The top of the git repository that holds ``root`` (``root`` itself or a folder above it), or None."""
    here = Path(root).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def listing(root):
    """The source's files, relative paths: what git tracks under it when it is (in) a repository that tracks
    something there (an index that cannot be read is refused by name, ``tracked.IndexUnreadable``), else a walk
    that skips dotfiles and dot-folders; never a ``data/``, never a file named for a secret. A root that cannot be
    listed is refused by name (``RootUnreadable``)."""
    return listing_and_blind(root)[0]


def looked_at(root):
    """Refused by name (``RootUnreadable``) when ``root`` is no folder that can be listed: gone, unmounted, closed."""
    if not Path(root).is_dir() or not os.access(root, os.R_OK | os.X_OK):
        raise RootUnreadable(f"{root} cannot be listed (gone, unmounted or not readable): its source is left as it "
                             f"is")


def listing_and_blind(root):
    """``listing``, and the folders a walk could not read (relative paths): a pass keeps what it held under them."""
    root = Path(root)
    looked_at(root)
    blind = []
    top = repository_top(root)
    paths = None
    if top is not None:
        prefix = root.resolve().relative_to(top).as_posix()
        prefix = "" if prefix == "." else prefix + "/"
        paths = [p[len(prefix):] for p in tracked_files(top) if p.startswith(prefix)]
        if not paths and prefix:
            paths = None        # a folder the repository does not track: walked as any folder
    if paths is None:
        paths = []
        def unread(error):
            if getattr(error, "filename", None):
                if Path(error.filename) == root:
                    raise RootUnreadable(f"{root} cannot be listed ({error}): its source is left as it is")
                blind.append(str(Path(error.filename).relative_to(root)))

        for base, dirs, names in os.walk(root, onerror=unread):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(names):
                if not name.startswith("."):
                    paths.append(str((Path(base) / name).relative_to(root)))
    return [p for p in paths if kept(p)], blind


def inside(root, full):
    """Whether ``full`` is a file of ``root`` itself: not a symbolic link, and resolving under ``root`` (a link
    may point anywhere, a directory of the path may have become one)."""
    try:
        return not full.is_symlink() and full.resolve().is_relative_to(Path(root).resolve())
    except OSError:
        return False


def file_lines(root, rel):
    """The lines of ``rel`` under ``root`` as the index read them (``\\r`` kept), or None: not inside the root (a
    link, or a path that resolves out of it), not a regular file (a FIFO is never opened for good), larger than
    MAX_BYTES, not text."""
    data = file_bytes(root, rel)
    if data is None or b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8").split("\n")
    except UnicodeDecodeError:
        return None


def file_bytes(root, rel):
    """The bytes of ``rel`` under ``root``, read as a pass and a citation read them, or None: not inside the root,
    not a regular file, larger than MAX_BYTES; opened without following a link, so a file swapped for one after it
    was checked is not read through it."""
    full = Path(root) / rel
    if not inside(root, full):
        return None
    try:
        fd = os.open(full, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        held = os.fstat(fd)
        if not stat.S_ISREG(held.st_mode) or held.st_size > MAX_BYTES:
            return None
        # O_NOFOLLOW guards the last name alone: a folder above it swapped for a link is caught by where the open
        # file really is (Linux's /proc; elsewhere the check before the open stands alone).
        opened = f"/proc/self/fd/{fd}"
        if os.path.exists(opened) and not Path(os.path.realpath(opened)).is_relative_to(Path(root).resolve()):
            return None
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(MAX_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    return None if len(data) > MAX_BYTES else data


def path_words(rel):
    return " ".join(rel.replace("/", " ").replace(".", " ").replace("_", " ").replace("-", " ").split())


def text_lines(data):
    """The lines of a file's bytes, or None when they are not text (a null byte, or not UTF-8)."""
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8").split("\n")
    except UnicodeDecodeError:
        return None


def _reader(mapping, whole):
    """A source of documents read by id: its text; None when the source holds no such document; ``UNKNOWN`` when
    the indexer never held the whole source and was not given this one."""

    def read(doc_id):
        if doc_id in mapping:
            return mapping[doc_id]
        return None if whole else UNKNOWN

    return read


class Indexer:
    def __init__(self, store, header="full"):
        self.store = store
        self.header = header
        self.last_stop = None   # why the last vector pass stopped (held, missing), said in the log too
        self.readers = {}
        self._documents = {}    # name -> (the documents held, as given; whether they are the whole source)

    def index(self, name, root, kind="folder", sensitivity=None, paths=None):
        """A pass over a folder (its root kept absolute); ``sensitivity`` None keeps the source's own (private for a
        new one). One pass at a time on a store: its lock is held from the source's row to the commit."""
        root = Path(os.path.abspath(root))
        with self.store.lock:
            source = self.store.source(name, root, kind, sensitivity)
            try:
                return self._index(source, root, paths)
            except BaseException:
                # Nothing of a pass that fails stays: its drops and updates would otherwise be committed by the
                # next write on this connection, a file left with its new sha and no parts.
                self.store.db.rollback()
                raise

    def _index(self, source, root, paths):
        started = time.perf_counter()
        began_ns = time.time_ns()
        store, db = self.store, self.store.db
        if paths is not None:
            looked_at(root)
            listed, blind = [p for p in paths if kept(p)], []
        else:
            listed, blind = listing_and_blind(root)
        last = store.meta(f"pass:{source}")
        trusted = int(last) - RACY_NS if last else None
        held = {r[0]: r[1:] for r in db.execute(
            "SELECT path, id, size, mtime_ns, sha256 FROM file WHERE source = ?", (source,))}
        generation = db.execute("SELECT generation FROM source WHERE id = ?", (source,)).fetchone()[0] + 1
        counts = {"added": 0, "changed": 0, "renamed": 0, "removed": 0, "unchanged": 0, "touched": 0,
                  "skipped": 0, "unreadable": 0, "parts": 0, "recut": 0}
        if held and not listed:
            raise RootUnreadable(f"{root} lists nothing where its source holds {len(held)} files (a drive unmounted, "
                                 f"its mount point left): refused; forget the source to empty it")
        recut = bool(held) and (store.meta(f"cutter:{source}") != str(CHUNKER_VERSION)
                                or store.meta(f"header:{source}") != self.header)
        seen, fresh = set(), []
        unlooked = 0    # files that could not be looked at: kept as they were held
        for rel in listed:
            try:
                rel.encode("utf-8")
            except UnicodeEncodeError:
                counts["skipped"] += 1      # a name in bytes that are not UTF-8 cannot be kept or cited
                continue
            full = root / rel
            try:
                st = full.stat()
            except FileNotFoundError:
                continue                    # gone: removed below
            except OSError:
                seen.add(rel)               # it could not be looked at (a folder closed): what was held of it stays
                unlooked += 1
                continue
            if not inside(root, full) or not full.is_file() or st.st_size > MAX_BYTES:
                counts["skipped"] += 1
                continue
            seen.add(rel)
            old = held.get(rel)
            if (old and old[1] == st.st_size and old[2] == st.st_mtime_ns and not recut and trusted is not None
                    and st.st_mtime_ns < trusted):
                counts["unchanged"] += 1
                continue
            data = file_bytes(root, rel)
            if data is None:
                counts["unreadable"] += 1   # gone from the index as a removed file is (a link put in its place too)
                seen.discard(rel)
                continue
            lines = text_lines(data)
            sha = hashlib.sha256(data).hexdigest()
            if lines is None:
                # Not text (any more): what was indexed of it goes, as for a file removed.
                counts["skipped"] += 1
                seen.discard(rel)
                continue
            if old and old[3] == sha and recut:
                store.drop_parts(old[0])
                db.execute("UPDATE file SET size = ?, mtime_ns = ?, generation = ? WHERE id = ?",
                           (st.st_size, st.st_mtime_ns, generation, old[0]))
                counts["parts"] += self._write_parts(old[0], rel, lines)
                counts["recut"] += 1
                continue
            if old and old[3] == sha:
                if old[1] == st.st_size and old[2] == st.st_mtime_ns:
                    counts["unchanged"] += 1    # read again for its recent time alone: nothing to write
                    continue
                db.execute("UPDATE file SET size = ?, mtime_ns = ? WHERE id = ?", (st.st_size, st.st_mtime_ns,
                                                                                   old[0]))
                counts["touched"] += 1
                continue
            fresh.append((rel, st, sha, lines, old))
        # A file under a folder the walk could not read is not gone: what was held of it stays.
        gone = {rel: row for rel, row in held.items() if rel not in seen
                and not any(rel.startswith(b.rstrip("/") + "/") for b in blind)}
        by_sha = {row[3]: rel for rel, row in gone.items()}
        for rel, st, sha, lines, old in fresh:
            moved = by_sha.pop(sha, None) if old is None else None
            if moved is not None:
                # The same file under a new path: its row (and id) kept, its parts cut again, since a module's
                # name and its tokens carry the path.
                file_id = gone.pop(moved)[0]
                store.drop_parts(file_id)
                db.execute("UPDATE file SET path = ?, size = ?, mtime_ns = ?, lang = ?, generation = ? WHERE id = ?",
                           (rel, st.st_size, st.st_mtime_ns, language(rel), generation, file_id))
                counts["parts"] += self._write_parts(file_id, rel, lines)
                counts["renamed"] += 1
                continue
            if old is None:
                file_id = db.execute("INSERT INTO file(source, path, size, mtime_ns, sha256, lang, generation) "
                                     "VALUES (?, ?, ?, ?, ?, ?, ?)", (source, rel, st.st_size, st.st_mtime_ns, sha,
                                                                      language(rel), generation)).lastrowid
                counts["added"] += 1
            else:
                file_id = old[0]
                store.drop_parts(file_id)
                db.execute("UPDATE file SET size = ?, mtime_ns = ?, sha256 = ?, generation = ? WHERE id = ?",
                           (st.st_size, st.st_mtime_ns, sha, generation, file_id))
                counts["changed"] += 1
            counts["parts"] += self._write_parts(file_id, rel, lines)
        for rel, row in gone.items():
            store.drop_parts(row[0])
            db.execute("DELETE FROM file WHERE id = ?", (row[0],))
            counts["removed"] += 1
        counts["vectors_dropped"] = store.orphan_vectors()
        db.execute("UPDATE source SET generation = ? WHERE id = ?", (generation, source))
        if not blind and not unlooked:
            # Only a pass that looked at every file says the source is cut by this cutter, and when it began.
            store.meta(f"cutter:{source}", CHUNKER_VERSION)
            store.meta(f"header:{source}", self.header)
            store.meta(f"pass:{source}", began_ns)
        counts["unreadable"] += unlooked
        counts["seconds"] = round(time.perf_counter() - started, 2)
        store.episode("index", source, counts)
        db.commit()
        if counts["removed"]:
            self._merge()
        return counts

    def _merge(self):
        """After a pass that removed files, outside it: the index merged by at most MERGE_PAGES pages, so the removed
        words leave the store's bytes as it merges (``Store.purge(None)`` merges it whole). A merge that fails -- a
        full disk -- is said, and the pass it follows stands."""
        try:
            self.store.purge(MERGE_PAGES)
            self.store.db.commit()
        except Exception as exc:  # noqa: BLE001 - the pass is committed; its merge waits for the next one
            self.store.db.rollback()
            logger.warning("mycelium: the index was not merged after a removal, its removed words stay until it is "
                           "(%s)", exc)

    def index_documents(self, name, documents, sensitivity=None, complete=True):
        """A source that is not a folder: ``documents`` is ``{doc_id: text}``, the whole source when ``complete``
        (a document not given is removed), else only what changed; cut by its id's suffix (none: text). Its
        citations are checked through the mapping itself, kept as the source's reader until the next pass (a pass
        that is not complete adds to the reader it finds: the documents it was not given stay readable; an indexer
        that never held the whole source says it cannot check them). A document whose id or text cannot be kept is
        skipped. The reader changes only once the pass is committed. A cutter of another version cuts every document
        again; only a complete pass marks the source cut by it. ``sensitivity`` None keeps the source's own."""
        with self.store.lock:
            source = self.store.source(name, DOCUMENTS + name, "documents", sensitivity)
            try:
                return self._index_documents(name, source, documents, complete)
            except BaseException:
                self.store.db.rollback()
                raise

    def _index_documents(self, name, source, documents, complete):
        started = time.perf_counter()
        store, db = self.store, self.store.db
        held = {r[0]: r[1:] for r in db.execute("SELECT path, id, sha256 FROM file WHERE source = ?", (source,))}
        generation = db.execute("SELECT generation FROM source WHERE id = ?", (source,)).fetchone()[0] + 1
        counts = {"added": 0, "changed": 0, "removed": 0, "unchanged": 0, "parts": 0, "recut": 0, "skipped": 0}
        recut = bool(held) and (store.meta(f"cutter:{source}") != str(CHUNKER_VERSION)
                                or store.meta(f"header:{source}") != self.header)
        given = {}
        for doc_id, text in dict(documents).items():
            try:
                if not isinstance(doc_id, str) or not isinstance(text, str):
                    raise TypeError("a document is a text under a name")
                doc_id.encode("utf-8")
                text.encode("utf-8")
            except (UnicodeEncodeError, TypeError):
                counts["skipped"] += 1      # an id or a text that cannot be kept or cited: the others are
                continue
            given[doc_id] = text
        before, whole = self._documents.get(name, ({}, False))
        mapping = dict(given) if complete else {**before, **given}
        whole = complete or whole
        for doc_id, text in given.items():
            data = text.encode("utf-8")
            sha = hashlib.sha256(data).hexdigest()
            old = held.get(doc_id)
            if old and old[1] == sha and not recut:
                counts["unchanged"] += 1
                continue
            if old and old[1] == sha:
                store.drop_parts(old[0])
                db.execute("UPDATE file SET generation = ? WHERE id = ?", (generation, old[0]))
                counts["parts"] += self._write_parts(old[0], doc_id, text.split("\n"))
                counts["recut"] += 1
                continue
            if old is None:
                file_id = db.execute("INSERT INTO file(source, path, size, mtime_ns, sha256, lang, generation) "
                                     "VALUES (?, ?, ?, 0, ?, ?, ?)", (source, doc_id, len(data), sha,
                                                                     language(doc_id), generation)).lastrowid
                counts["added"] += 1
            else:
                file_id = old[0]
                store.drop_parts(file_id)
                db.execute("UPDATE file SET size = ?, sha256 = ?, generation = ? WHERE id = ?",
                           (len(data), sha, generation, file_id))
                counts["changed"] += 1
            counts["parts"] += self._write_parts(file_id, doc_id, text.split("\n"))
        if complete:
            for doc_id, row in held.items():
                if doc_id not in given:
                    store.drop_parts(row[0])
                    db.execute("DELETE FROM file WHERE id = ?", (row[0],))
                    counts["removed"] += 1
        counts["vectors_dropped"] = store.orphan_vectors()
        db.execute("UPDATE source SET generation = ? WHERE id = ?", (generation, source))
        if complete or not held:
            # A complete pass cut every document by this cutter; so did a first one, whatever it was given.
            store.meta(f"cutter:{source}", CHUNKER_VERSION)
            store.meta(f"header:{source}", self.header)
        counts["seconds"] = round(time.perf_counter() - started, 2)
        store.episode("index", source, counts)
        db.commit()
        self._documents[name] = (mapping, whole)
        self.readers[name] = _reader(mapping, whole)
        if counts["removed"]:
            self._merge()
        return counts

    def status(self, name, root, paths=None):
        """What a pass over ``root`` would do, read from the files' sizes and times alone (nothing hashed, nothing
        written), and how long it would take, from the passes before: {new, touched, gone, estimate_seconds}."""
        db = self.store.db
        row = db.execute("SELECT id FROM source WHERE name = ?", (name,)).fetchone()
        held = {} if row is None else {r[0]: (r[1], r[2]) for r in db.execute(
            "SELECT path, size, mtime_ns FROM file WHERE source = ?", (row[0],))}
        listed = paths if paths is not None else listing(Path(root))
        new = touched = 0
        seen = set()
        for rel in listed:
            try:
                st = (Path(root) / rel).stat()
            except OSError:
                continue
            seen.add(rel)
            if rel not in held:
                new += 1
            elif held[rel] != (st.st_size, st.st_mtime_ns):
                touched += 1
        rate = None
        for episode in self.store.episodes(None if row is None else row[0], 10):
            d = episode["detail"]
            work = d.get("added", 0) + d.get("changed", 0) + d.get("renamed", 0) + d.get("recut", 0)
            if work and d.get("seconds") is not None:
                rate = d["seconds"] / work
                break
        return {"new": new, "touched": touched, "gone": len(set(held) - seen),
                "estimate_seconds": None if rate is None else round(rate * (new + touched), 1)}

    def _write_parts(self, file_id, rel, lines):
        db = self.store.db
        doc = module_doc(rel, lines)
        db.execute("UPDATE file SET module_doc = ? WHERE id = ?", (doc, file_id))
        words_of_path = path_words(rel)
        n = 0
        for part in cut(rel, lines):
            body = "\n".join(lines[part.first - 1:part.last])
            sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            embed_sha = text_sha(embedded_text(self.header, rel, part.kind, part.symbol, doc, body))
            part_id = db.execute("INSERT INTO part(file, kind, symbol, first, last, sha256, embed_sha) "
                                 "VALUES (?, ?, ?, ?, ?, ?, ?)", (file_id, part.kind, part.symbol, part.first,
                                                                  part.last, sha, embed_sha)).lastrowid
            db.execute("INSERT INTO fts(rowid, path, body, words, said) VALUES (?, ?, ?, ?, ?)",
                       (part_id, words_of_path, body, subwords(body + " " + part.symbol), "\n".join(part.said)))
            db.executemany("INSERT INTO symbol(part, name, qualname, kind) VALUES (?, ?, ?, ?)",
                           [(part_id, *d) for d in part.defines])
            db.executemany("INSERT INTO ref(part, name, kind, module) VALUES (?, ?, ?, ?)",
                           [(part_id, *r) for r in part.refs])
            n += 1
        return n

    def _part_texts(self, key):
        """(embed_sha, text) for each embedded text that has no vector under ``key``, read again from its source
        (only those files are read); a text whose lines changed since its pass is left out (the next index pass
        cuts its file again)."""
        db = self.store.db
        rows = db.execute(
            "SELECT p.embed_sha, f.path, p.kind, p.symbol, p.first, p.last, f.module_doc, s.root, s.kind "
            "FROM part p JOIN file f ON f.id = p.file JOIN source s ON s.id = f.source WHERE p.embed_sha != '' "
            "AND p.embed_sha NOT IN (SELECT embed_sha FROM vector WHERE encoder = ?) ORDER BY f.id, p.first",
            (key,)).fetchall()
        lines_of = {}
        for embed_sha, rel, kind, symbol, first, last, doc, root, source_kind in rows:
            if (root, rel) not in lines_of:
                lines_of = {(root, rel): self._source_lines(root, source_kind, rel)}
            lines = lines_of[(root, rel)]
            if lines is None:
                continue
            text = embedded_text(self.header, rel, kind, symbol, doc, "\n".join(lines[first - 1:last]))
            if text_sha(text) == embed_sha:
                yield embed_sha, text

    def _source_lines(self, root, source_kind, rel):
        """A part's file read again as a citation reads it (``file_lines``), a document through the reader this
        indexer holds; None when it cannot be (a document this indexer does not know included)."""
        if source_kind == "documents":
            reader = self.readers.get(root[len(DOCUMENTS):])
            text = reader(rel) if reader is not None else None
            return None if text is None or text is UNKNOWN else text.split("\n")
        return file_lines(root, rel)

    def embed(self, embedder, batch=32):
        """Vectors for every embedded text that has none under ``embedder``'s encoder (see the module's doc)."""
        started = time.perf_counter()
        store, db = self.store, self.store.db
        key = embedder.encoder.key
        with store.lock:
            wanted = {}
            for embed_sha, text in self._part_texts(key):
                wanted.setdefault(embed_sha, text)
        todo = sorted(wanted.items(), key=lambda item: (len(item[1]), item[0]))
        # An episode holds numbers: why a pass stopped is said in the log and kept here, never written.
        counts = {"embedded": 0, "refused": 0, "held": False, "missing": False}
        refused = set()     # the texts the engine refused in this pass
        self.last_stop = None
        for at in range(0, len(todo), batch):
            chunk = todo[at:at + batch]
            try:
                vectors, _refused = embedder.texts([text for _sha, text in chunk])
            except Held as exc:
                counts["held"], self.last_stop = True, str(exc)
                logger.info("mycelium: the vector pass under %s is held: %s", key, exc)
                break
            except EncoderMissing as exc:
                counts["missing"], self.last_stop = True, str(exc)
                logger.warning("mycelium: the vector pass under %s stopped: %s", key, exc)
                break
            with self._writing():
                for (embed_sha, _text), vector in zip(chunk, vectors):
                    if vector is None:
                        counts["refused"] += 1
                        refused.add(embed_sha)
                        continue
                    # A text that left the store while the engine answered (an index pass meanwhile) is not kept.
                    counts["embedded"] += db.execute(
                        "INSERT OR REPLACE INTO vector(embed_sha, encoder, dim, data) SELECT ?, ?, ?, ? "
                        "WHERE EXISTS (SELECT 1 FROM part WHERE embed_sha = ?)",
                        (embed_sha, key, len(vector), to_blob(vector), embed_sha)).rowcount
                store.vectors_changed()
        with self._writing():
            # While another encoder is read, this one replaces it only once it covers every part: its vectors go
            # at the change. On a store with no encoder read, the texts the engine refused in this very pass and
            # still lacking are set aside, if it covers anything. A part this indexer could not read again
            # (another header, a reader it does not hold, a file changed since) is never set aside.
            lacking = self._lacking(key)
            counts["uncovered"] = len(lacking - refused)
            active = store.active_encoder()
            if not counts["held"] and not counts["missing"] and (active is None or active["key"] != key):
                ready = not lacking if active is not None else not counts["uncovered"] and self._covers(key)
                if ready:
                    store.activate(key, embedder.encoder.card())
                    counts["activated"] = True
            counts["seconds"] = round(time.perf_counter() - started, 2)
            store.episode("embed", None, counts)
        return counts

    @contextmanager
    def _writing(self):
        """The store's lock held while one batch of writes is made and committed, rolled back if it fails."""
        with self.store.lock:
            try:
                yield
                self.store.db.commit()
            except BaseException:
                self.store.db.rollback()
                raise

    def _lacking(self, key):
        """The embedded texts with no vector under ``key`` ('' for parts with no embedded text yet)."""
        return {r[0] for r in self.store.db.execute(
            "SELECT DISTINCT embed_sha FROM part WHERE embed_sha = '' OR embed_sha NOT IN "
            "(SELECT embed_sha FROM vector WHERE encoder = ?)", (key,))}

    def _covers(self, key):
        """Whether any part has a vector under ``key``."""
        return self.store.db.execute(
            "SELECT EXISTS (SELECT 1 FROM part p JOIN vector v ON v.embed_sha = p.embed_sha AND v.encoder = ?)",
            (key,)).fetchone()[0] == 1

    def vector_status(self, encoder):
        """What the vectors are, written nothing: the encoder read, the parts and those with a vector under
        ``encoder``, and what the last vector pass said (held, missing)."""
        db = self.store.db
        texts = db.execute("SELECT COUNT(DISTINCT embed_sha) FROM part WHERE embed_sha != ''").fetchone()[0]
        covered = db.execute("SELECT COUNT(DISTINCT p.embed_sha) FROM part p JOIN vector v ON v.embed_sha = "
                             "p.embed_sha AND v.encoder = ?", (encoder.key,)).fetchone()[0]
        last = next((e["detail"] for e in self.store.episodes(None, 50) if e["action"] == "embed"), None)
        active = self.store.active_encoder()
        return {"active": None if active is None else active["key"], "encoder": encoder.key, "texts": texts,
                "with_vector": covered, "held": bool(last and last.get("held")),
                "missing": bool(last and last.get("missing")), "why": self.last_stop}
