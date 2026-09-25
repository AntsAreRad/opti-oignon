"""The componion's store: one SQLite file per person, and the only way into a being's journal.

Each person's being lives in one file under the data directory, named by the
owner's tag and the soil: ``<tag>.db`` is encrypted soil (SQLCipher through
the platform's own keyed connection), ``<tag>.glass.db`` a glass jar -- a
store in clear, allowed only by the YAML option, only when no key is
configured and only in Daily mode, and never opened in Bulbe.

Every dependency comes in through a seam of ``Store`` (the data directory,
the settings, both connects, the probe, the cipher's availability, the
anchor key, the audit log, the entropy source, the payload cipher, the
clock, the mode) and the platform defaults are imported only when a seam is
not given. Importing this module opens nothing.

A file is judged before it is trusted:

* an existing encrypted-name file needs SQLCipher and a readable key before
  it is connected, and its header must not be the database magic; after the
  connection it must answer a cipher version, and the first open in each
  process runs the page check;
* a new file is created empty, its schema committed, and only then probed:
  a file written in clear is refused (``PlaintextRefused``) and deleted with
  its journals, whatever its size.

Every write runs under ``BEGIN IMMEDIATE`` with the busy timeout from the
settings, rewrites the store anchor before it commits, and never retries: a
busy store is refused ``busy``. ``journal_mode`` is DELETE, ``secure_delete``
ON and ``temp_store`` MEMORY, and a destruction is followed by ``VACUUM``.

A birth is anchored in the signed audit log before the store is linked into
place, so a store missing after its birth is ``missing`` and never ``ready``,
and a being is never adopted by another account. Looking (``status``) writes
nothing and creates nothing.

Outside the journal, each being keeps local layers the trunk read never
touches: rhythm rows (the light hook's presence hours, with their minute
sealed inside), heard rows per season, and checkpoints of reducer states
(kept once, read back, never overwritten). Every destructible key is drawn
at random and lives only in its store; destroying one deletes its rows in the
same transaction, then VACUUM runs and a cross-anchor records the
destruction.

A refused store is refused the same way however often it is opened, and
nothing is repaired behind the person's back: ``resume`` runs only when the
person confirms the numbers the status showed, sets aside what failed to
verify, and enforces every recorded destruction again.

Refusals are named (``StoreRefused``, ``ChainRefused``, ``PlaintextRefused``,
``ResumeRefused``; the membrane's own are ``MembraneRefused``). None of them
is a wire code.
"""

import errno
import hashlib
import hmac
import logging
import os
import stat
import threading
from pathlib import Path
from typing import NamedTuple

checkpoint_before_apply = True

logger = logging.getLogger(__name__)

MAGIC = b"SQLite format 3\x00"
HEADER_BYTES = 16
MIN_FILE_BYTES = 512
MAX_INT = (1 << 53) - 1
SCHEMA_VERSION = 1
NOKEY = "nokey"
SOILS = ("encrypted", "glass")
SUFFIXES = {
    "autobio_encrypted": ".autobio.db",
    "autobio_glass": ".autobio.glass.db",
    "encrypted": ".db",
    "glass": ".glass.db",
    "sowing_encrypted": ".sowing.db",
    "sowing_glass": ".sowing.glass.db",
}
LOCK_NAME = ".sowing.lock"
JOURNALS = ("-journal", "-wal", "-shm")
# The explicit index comes last: VACUUM recreates tables before indexes, so
# with the index last every root page keeps its number across a VACUUM.
SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS facts (eid TEXT PRIMARY KEY, being TEXT NOT NULL, t INTEGER NOT NULL, "
    "kind TEXT NOT NULL, origin TEXT NOT NULL, oseq INTEGER NOT NULL, laws INTEGER NOT NULL, "
    "body_sha256 TEXT NOT NULL, UNIQUE (origin, oseq))",
    "CREATE TABLE IF NOT EXISTS bodies (eid TEXT PRIMARY KEY REFERENCES facts (eid), body BLOB, redacted_by TEXT)",
    "CREATE TABLE IF NOT EXISTS links (seq INTEGER PRIMARY KEY, prev TEXT NOT NULL, "
    "eid TEXT NOT NULL UNIQUE REFERENCES facts (eid), link TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS keys (name TEXT PRIMARY KEY, key BLOB NOT NULL) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS payloads (ref TEXT PRIMARY KEY, eid TEXT NOT NULL, ct BLOB NOT NULL) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS checkpoints (being TEXT NOT NULL, t INTEGER NOT NULL, engine TEXT NOT NULL, "
    "laws INTEGER NOT NULL, through TEXT NOT NULL, state_hash TEXT NOT NULL, blob BLOB NOT NULL, "
    "PRIMARY KEY (being, t, engine, laws, through))",
    "CREATE TABLE IF NOT EXISTS overflow (day INTEGER NOT NULL, kind TEXT NOT NULL, dropped INTEGER NOT NULL, "
    "PRIMARY KEY (day, kind))",
    "CREATE TABLE IF NOT EXISTS rhythm_facts (rseq INTEGER PRIMARY KEY, ct BLOB NOT NULL)",
    "CREATE TABLE IF NOT EXISTS heard (season INTEGER NOT NULL, hash TEXT NOT NULL, ct BLOB NOT NULL, "
    "PRIMARY KEY (season, hash))",
    "CREATE INDEX IF NOT EXISTS facts_budget ON facts (kind, t)",
)
# Refusals of an existing store that make it unreadable; the others make it unavailable. ``law`` is not
# among them: a law the engine does not carry, or its own table damaged, is a cause outside the store.
UNREADABLE = ("soil", "pages", "owner", "unsown", "local")
# What the heard layer takes: a word of lowercase letters and apostrophes, and a small object.
HEARD_WORD_MAX = 24
HEARD_DATA_MAX = 512
LAWS_MAX = 65535
T_MAX = "SELECT MAX(f.t) FROM facts f JOIN links l ON l.eid = f.eid"
ANCHOR_LABEL = b"opti-oignon-allium-anchor-v1"
KEYID_LABEL = b"opti-oignon-allium-anchor-keyid-v1"
_HEX = "0123456789abcdef"


# ---------------------------------------------------------------------------
# Refusals and records
# ---------------------------------------------------------------------------
class StoreRefused(ValueError):
    """A store action refused by name; ``code`` is one of ``CODES``."""

    CODES = ("path", "plaintext", "no_soil", "key", "cipher", "soil", "pages", "exists", "anchor_unwritten",
             "audit", "unanchored", "foreign", "unsown", "sealed", "busy", "divergence", "owner", "law", "local",
             "unclaimed")

    def __init__(self, code, detail=""):
        if code not in self.CODES:
            raise ValueError(f"unknown store refusal: {code}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class PlaintextRefused(StoreRefused):
    """A new file that was written in clear; the message names the cause."""

    def __init__(self, detail):
        super().__init__("plaintext", detail)


class ChainRefused(StoreRefused):
    """The being's record breaks at event ``seq``, for ``reason``.

    ``kept_seq`` is the last event whose checks all passed, ``through`` the
    furthest event anything local or anchored names, ``discarded`` what a
    resume from ``kept_seq`` would set aside, and ``birth`` the genesis wall
    clock when it could be read.
    """

    REASONS = ("genesis", "seq", "link", "eid", "body", "redaction", "being", "laws", "orphan", "anchor",
               "truncated", "older", "fork", "local", "destroyed")

    def __init__(self, seq, reason, kept_seq=-1, kept_t=0, through=-1, discarded=0, detail="", birth=None,
                 gen=None):
        if reason not in self.REASONS:
            raise ValueError(f"unknown chain refusal: {reason}")
        message = f"the record breaks at event #{seq}: {reason}"
        ValueError.__init__(self, f"{message} ({detail})" if detail else message)
        self.code = reason
        self.reason = reason
        self.detail = detail
        self.seq = seq
        self.kept_seq = kept_seq
        self.kept_t = kept_t
        self.through = through
        self.discarded = discarded
        self.birth = birth
        # The write generation the refused verification read: a resume acts only on that state.
        self.gen = gen


class ResumeRefused(ValueError):
    """A resume or a finish that is not due (``nothing``) or not confirmed (``confirm``)."""

    CODES = ("nothing", "confirm")

    def __init__(self, code, detail=""):
        if code not in self.CODES:
            raise ValueError(f"unknown resume refusal: {code}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class Status(NamedTuple):
    """What a person's being is, as far as the store can tell without writing anything."""

    status: str
    labels: tuple
    reason: object
    detail: object
    hints: tuple
    offer: object
    snapshot: object


class Resume(NamedTuple):
    """An offer to resume from the last sound event, and what it would set aside."""

    kept_seq: int
    kept_days: int
    discarded: int


class Finish(NamedTuple):
    """An offer to finish an interrupted sowing, confirmed by the being's short tag."""

    being: str


class Checkpoint(NamedTuple):
    """A reducer state kept at one place in the chain, as the store holds it."""

    being: str
    t: int
    engine: str
    laws: int
    through: str
    state_hash: str
    blob: bytes


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex(value, length):
    if not isinstance(value, str) or len(value) != length:
        return False
    for char in value:
        if char not in _HEX:
            return False
    return True


def _heard_word(value):
    """A heard word: 1..=24 lowercase letters or apostrophes."""
    if not isinstance(value, str) or not 1 <= len(value) <= HEARD_WORD_MAX:
        return False
    for char in value:
        if not ("a" <= char <= "z" or char == "'"):
            return False
    return True


_ENGINE_ID = {}


def _engine_id():
    """The digest of the engine's identity, which a checkpoint is keyed by; computed once per process."""
    if "sha256" not in _ENGINE_ID:
        from .ref import protocol

        _ENGINE_ID["sha256"] = hashlib.sha256(protocol.engine_info()).hexdigest()
    return _ENGINE_ID["sha256"]


def _inflates_to(blob, canonical):
    """Whether a kept blob inflates to exactly ``canonical``; never more than one byte past it is inflated."""
    import zlib

    try:
        inflater = zlib.decompressobj()
        out = inflater.decompress(bytes(blob), len(canonical) + 1)
    except Exception:  # noqa: BLE001 - a blob that does not inflate holds no state
        return False
    return out == canonical and inflater.eof and not inflater.unused_data


def _envelopes(rows):
    from . import wire

    for being, digest, kind, laws, origin, oseq, t, body in rows:
        envelope = {"being": being, "body": digest, "kind": kind, "laws": laws, "origin": origin,
                    "oseq": oseq, "t": t}
        yield envelope, (None if body is None else wire.parse(bytes(body)))


# ---------------------------------------------------------------------------
# Files, the probe, connections
# ---------------------------------------------------------------------------
def probe(path, conn=None):
    """``"encrypted"``, ``"plaintext"`` or ``"short"``: the header first, then the connection.

    A file shorter than 512 bytes, or than its 16-byte header, is short. A
    header that is the database magic is plaintext whatever a connection
    says. Without a connection, any other header is taken as encrypted;
    with one, it must also answer a cipher version. A symbolic link is never
    followed: it reads as short.
    """
    header, size = _header(path)
    if len(header) < HEADER_BYTES or size < MIN_FILE_BYTES:
        return "short"
    if header == MAGIC:
        return "plaintext"
    if conn is None:
        return "encrypted"
    try:
        version = bool(conn.execute("PRAGMA cipher_version").fetchall())
    except Exception:  # noqa: BLE001 - a connection that cannot answer answers no
        version = False
    return "encrypted" if version else "plaintext"


_PROBE = probe


def _header(path):
    """``(first 16 bytes, size)`` of a file opened without following a link; ``(b"", 0)`` when it cannot be."""
    try:
        handle = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return b"", 0
    try:
        return os.read(handle, HEADER_BYTES), os.fstat(handle).st_size
    except OSError:
        return b"", 0
    finally:
        os.close(handle)


def _is_link(path):
    """Whether a name is a symbolic link; a name that cannot be looked at counts as one."""
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False
    except OSError:
        return True


def store_file(directory, tag, name):
    """The path of ``<tag><suffix>`` in ``directory``; only the directory is resolved, never the name.

    A store name that is a symbolic link is refused ``path``: nothing is ever
    opened or deleted as a store through a link, whether it names a file
    outside the directory or another person's store inside it. A sowing temp
    name is returned as it is, link or not: whoever sorts the temp files
    removes a link itself, never what it names.
    """
    if not _is_hex(tag, 32):
        raise StoreRefused("path", "an owner tag is 32 lowercase hex characters")
    try:
        home = Path(directory).resolve()
    except (OSError, RuntimeError):
        raise StoreRefused("path", "the store directory cannot be resolved") from None
    path = home.joinpath(tag + SUFFIXES[name])
    if not name.startswith("sowing_") and _is_link(path):
        raise StoreRefused("path", "the store file is a symbolic link")
    return path


def _remove_with_journals(path):
    for suffix in ("",) + JOURNALS:
        try:
            os.unlink(str(path) + suffix)
        except FileNotFoundError:
            pass


def _close_quietly(conn):
    if conn is None:
        return
    try:
        conn.close()
    except Exception:  # noqa: BLE001 - a connection that cannot even close is dropped all the same
        pass


def _mapped(conn, exc):
    """The store's refusal for a database exception, or ``None`` when it is not one.

    ``OperationalError`` is tested first (it is a ``DatabaseError``): locked
    or busy is ``busy``; anything else, and every other ``DatabaseError`` or
    a ``MemoryError``, is ``pages`` and closes the connection for good.
    """
    operational = getattr(conn, "OperationalError", None)
    database = getattr(conn, "DatabaseError", None)
    if isinstance(operational, type) and isinstance(exc, operational):
        text = str(exc).lower()
        if "locked" in text or "busy" in text:
            return StoreRefused("busy", str(exc))
        _close_quietly(conn)
        return StoreRefused("pages", str(exc))
    if (isinstance(database, type) and isinstance(exc, database)) or isinstance(exc, MemoryError):
        _close_quietly(conn)
        return StoreRefused("pages", str(exc))
    return None


def _guarded(conn, work):
    try:
        return work()
    except StoreRefused:
        raise
    except Exception as exc:  # noqa: BLE001 - mapped below, or raised as it was
        mapped = _mapped(conn, exc)
        if mapped is None:
            raise
        raise mapped from None


def _transaction(conn, work):
    """Run ``work(conn)`` in one ``BEGIN IMMEDIATE`` transaction; any exception rolls back and is raised."""

    def run():
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = work(conn)
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            raise
        return result

    return _guarded(conn, run)


def _put(conn, key, value):
    from . import wire

    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, wire.emit(value)))


def _read_meta(conn):
    from . import chain

    return chain.parse_meta(conn.execute("SELECT key, value FROM meta").fetchall())


def _head(conn):
    row = conn.execute("SELECT seq, link FROM links ORDER BY seq DESC LIMIT 1").fetchone()
    return (row[0], row[1])


def _insert_fact(conn, *, being, origin, oseq, t, kind, body, laws, head):
    """Write a fact's rows and its link after ``head``; ``(eid, seq, link)``. Its identity is the engine's."""
    from . import chain, engine, wire
    from .membrane import MembraneRefused

    fact = {"being": being, "body": body, "kind": kind, "laws": laws, "origin": origin, "oseq": oseq, "t": t}
    answer = wire.parse(engine.call(wire.emit({"fact": fact, "op": "fact_id", "v": 1})))
    if "refused" in answer:
        raise MembraneRefused("body", f"body {answer.get('detail')}")
    eid = answer["eid"]
    seq = head[0] + 1
    link = chain.link(eid, head[1], seq)
    conn.execute("INSERT INTO facts (eid, being, t, kind, origin, oseq, laws, body_sha256) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (eid, being, t, kind, origin, oseq, laws, answer["body"]))
    conn.execute("INSERT INTO bodies (eid, body, redacted_by) VALUES (?, ?, NULL)", (eid, wire.emit(body)))
    conn.execute("INSERT INTO links (seq, prev, eid, link) VALUES (?, ?, ?, ?)", (seq, head[1], eid, link))
    return eid, seq, link


def _rewrite_anchor(conn, *, being, owner, soil, gen, key_id, anchor_key):
    """Rewrite ``meta.anchor`` over the head, the generation and the local tables as they now are."""
    from . import chain

    head_seq, head_link = _head(conn)
    local = chain.local_digest(chain.read_local(conn), _read_meta(conn))
    _put(conn, "anchor", chain.anchor_value(being=being, gen=gen, head_seq=head_seq, head_link=head_link,
                                            local=local, owner=owner, soil=soil, key_id=key_id,
                                            anchor_key=anchor_key))


def _make_directory(directory):
    """Create the store directory (and what is missing above it) as 0o700; tighten it if it is open."""
    missing = []
    current = Path(directory)
    while not current.is_dir():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    try:
        for folder in reversed(missing):
            try:
                os.mkdir(folder, 0o700)
            except FileExistsError:
                pass
            os.chmod(folder, 0o700)
        _tighten(directory)
    except OSError:
        raise StoreRefused("path", "the store directory cannot be created") from None


def _tighten(directory):
    if stat.S_IMODE(os.stat(directory).st_mode) & 0o077:
        os.chmod(directory, 0o700)


# ---------------------------------------------------------------------------
# Platform defaults, imported only when a seam is not given
# ---------------------------------------------------------------------------
def _default_connect():
    try:
        from opti_oignon.db_utils import safe_connect
    except Exception:  # noqa: BLE001 - no keyed connection means no encrypted soil
        return _no_encrypted_connect
    return safe_connect


def _no_encrypted_connect(path, **kwargs):
    raise StoreRefused("cipher", "the platform's keyed connection is not reachable")


def _plain_connect(path, check_same_thread=True, timeout=5):
    import sqlite3

    return sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=timeout)


def _default_cipher_available():
    try:
        from opti_oignon import db_encryption
    except Exception:  # noqa: BLE001 - unknown is unavailable
        return False
    return db_encryption.SQLCIPHER_AVAILABLE is True


def _default_anchor_secret():
    """``("readable", anchor_key, key_id)``, ``("none", None, "nokey")`` or ``("unreadable", None, "nokey")``."""
    try:
        from opti_oignon import encryption
    except Exception:  # noqa: BLE001 - a key module that cannot load cannot say there is no key
        return ("unreadable", None, NOKEY)
    try:
        configured = bool(os.environ.get(encryption._ENV_KEY_NAME)) or encryption._DEFAULT_KEYFILE.exists()
    except Exception:  # noqa: BLE001 - a missing attribute reads as configured
        configured = True
    try:
        secure = encryption.get_encryption_key()
    except Exception:  # noqa: BLE001 - a key that cannot be read is not a key
        secure = None
    if not secure:
        return ("unreadable", None, NOKEY) if configured else ("none", None, NOKEY)
    try:
        with secure as master:
            raw = master.as_bytes()
            anchor_key = hmac.new(raw, ANCHOR_LABEL, hashlib.sha256).digest()
            key_id = hmac.new(raw, KEYID_LABEL, hashlib.sha256).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return ("unreadable", None, NOKEY)
    return ("readable", anchor_key, key_id)


def _default_clock():
    import time

    return time.time_ns() // 1000000000


def _no_stage(name):
    return None


def _key_shape(value):
    if isinstance(value, tuple) and len(value) == 3:
        state, key, key_id = value
        if state == "readable" and isinstance(key, (bytes, bytearray)) and len(key) == 32 and _is_hex(key_id, 16):
            return ("readable", bytes(key), key_id)
        if state == "none" and key is None:
            return ("none", None, NOKEY)
    return ("unreadable", None, NOKEY)


def _status(status, *, labels=(), reason=None, detail=None, hints=(), offer=None):
    return Status(status, tuple(labels), reason, detail, tuple(hints), offer, None)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class Store:
    """The stores of one data directory: one per person, opened, sown and looked at through here.

    One instance per process. It keeps one connection per open being under
    one lock, calls ``anchor_secret`` once, and remembers per being what it
    learned from the audit log and that the pages were checked. What it
    remembers never outranks the audit log: every verification reads the
    anchors written since the last one it read, and an owner found without a
    birth is looked up again before anything is decided on it, so a store
    that has been running for days decides as a fresh process would.
    """

    def __init__(self, *, single_user, data_dir=None, persistence=None, connect=None, plain_connect=None,
                 probe=None, cipher_available=None, anchor_secret=None, audit=None, audit_present=None,
                 entropy=None, cipher=None, clock=None, mode=None, stage=None):
        if not callable(single_user):
            raise TypeError("single_user is a callable: whether the platform runs for one person")
        from . import mode as modes

        self._single_user = single_user
        self._data_dir = data_dir
        self._persistence = persistence
        self._connect = connect if connect is not None else _default_connect()
        self._plain_connect = plain_connect if plain_connect is not None else _plain_connect
        self._probe = probe if probe is not None else _PROBE
        self._cipher_available = cipher_available if cipher_available is not None else _default_cipher_available
        self._anchor_secret = anchor_secret if anchor_secret is not None else _default_anchor_secret
        self._audit = audit
        self._audit_present = audit_present
        self._entropy = entropy if entropy is not None else os.urandom
        self._cipher = cipher
        self._clock = clock if clock is not None else _default_clock
        self._mode = mode if mode is not None else modes.live_mode
        self._stage = stage if stage is not None else _no_stage
        self._lock = threading.RLock()
        self._keys = None
        self._beings = {}
        self._pages = {}
        self._trusted = {}
        self._sown = {}
        self._anchored = {}

    # -- reading the seams ---------------------------------------------------

    def _read_mode(self):
        try:
            value = self._mode()
        except Exception:  # noqa: BLE001 - a mode that cannot be read is Bulbe
            return "bulbe"
        return "daily" if value == "daily" else "bulbe"

    def _read_clock(self):
        try:
            return self._clock()
        except Exception:  # noqa: BLE001 - refused where the reading is used
            return None

    def _single_user_now(self):
        try:
            return self._single_user() is True
        except Exception:  # noqa: BLE001 - unknown is not single-user: an account is then required
            return False

    def _key_state(self):
        if self._keys is None:
            try:
                value = self._anchor_secret()
            except Exception:  # noqa: BLE001
                value = None
            self._keys = _key_shape(value)
        return self._keys

    def _cipher_ok(self):
        try:
            return bool(self._cipher_available())
        except Exception:  # noqa: BLE001
            return False

    def _settings(self):
        from . import settings

        if self._persistence is not None:
            return settings.normalise(self._persistence)
        return settings.persistence()

    def _soil_keys(self, soil):
        """``(key_id, anchor_key, seal_key)`` for a soil: ``nokey`` for glass, the readable key otherwise."""
        from . import anchors

        if soil == "glass":
            return (NOKEY, None, None)
        state, key, key_id = self._key_state()
        if state != "readable":
            raise StoreRefused("key", "the master key is not readable")
        return (key_id, key, anchors.seal_key(key))

    def _cipher_pair(self):
        if self._cipher is not None:
            return self._cipher
        try:
            from opti_oignon import encryption

            return (encryption.encrypt_bytes, encryption.decrypt_bytes)
        except Exception:  # noqa: BLE001
            from .membrane import MembraneRefused

            raise MembraneRefused("payload", "no cipher backend") from None

    def _draw(self, n):
        value = self._entropy(n)
        if not isinstance(value, (bytes, bytearray)) or len(value) != n:
            raise ValueError("the entropy source answered the wrong number of bytes")
        return bytes(value)

    def _directory(self):
        """``(data directory, store directory)``, both resolved; nothing is created here."""
        base = self._data_dir
        if base is None:
            try:
                from opti_oignon import config

                base = config.DATA_DIR
            except Exception:  # noqa: BLE001
                raise StoreRefused("path", "no data directory: the platform's configuration is not reachable") from None
        parts = self._settings()["path"].split("/")
        try:
            root = Path(base).resolve()
            directory = root.joinpath(*parts).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            raise StoreRefused("path", "the store directory cannot be resolved") from None
        if directory == root or root not in directory.parents:
            raise StoreRefused("path", "the store directory is not inside the data directory")
        return root, directory

    # -- the audit log -------------------------------------------------------

    def _audit_log(self):
        if self._audit is None:
            try:
                from opti_oignon import signed_audit_log

                instance = signed_audit_log.signed_audit_log
            except Exception:  # noqa: BLE001
                instance = None
            if instance is None:
                raise StoreRefused("audit", "the signed audit log is not available")
            self._audit = instance
        return self._audit

    def _audit_here(self, audit):
        if self._audit_present is not None:
            try:
                return bool(self._audit_present())
            except Exception:  # noqa: BLE001 - unknown counts as present
                return True
        try:
            path = audit._db_path
        except AttributeError:
            return True
        try:
            return os.path.exists(path)
        except (TypeError, ValueError):
            return True

    def _trust(self, audit, owner):
        if self._trusted.get(owner):
            return
        try:
            result = audit.verify_chain()
        except Exception:  # noqa: BLE001
            raise StoreRefused("audit", "the audit chain cannot be verified") from None
        if not (isinstance(result, tuple) and len(result) == 3 and result[0] is True and result[1] is None):
            raise StoreRefused("audit", "the audit chain does not verify")
        self._trusted[owner] = True

    def _sow_read(self, owner):
        """The owner's birth as the audit log records it now, and the key it was sown under (``None`` for none)."""
        from . import anchors

        audit = self._audit_log()
        if not self._audit_here(audit):
            return ("none",), None
        self._trust(audit, owner)
        key_state, key, key_id = self._key_state()
        try:
            return anchors.sow_birth(audit, owner, key_id, key if key_state == "readable" else None)
        except StoreRefused:
            raise
        except Exception:  # noqa: BLE001
            raise StoreRefused("audit", "the audit log cannot be read") from None

    def _sow_state(self, owner, refresh=False):
        """The owner's birth as the audit log records it.

        An open or a foreign birth is remembered and re-read when asked;
        ``("none",)`` never is, so a birth another process anchored since this
        store last looked is seen before anything is decided on its absence.
        """
        if not refresh and owner in self._sown:
            return self._sown[owner]
        state, _key = self._sow_read(owner)
        if state[0] in ("open", "foreign"):
            self._sown[owner] = state
        else:
            self._sown.pop(owner, None)
        return state

    def _anchors_of(self, tag, owner, soil, fresh=None):
        """The cross-anchors of a being, brought up to date with the audit log.

        Only the entries newer than the newest one this store already read are
        read, and merged into what it remembers, so an anchor another process
        wrote is never missed. ``fresh`` holds the tags already brought up to
        date during the current verification, which reads each once.
        """
        from . import anchors
        from .membrane import MembraneRefused

        if fresh is not None and tag in fresh:
            return self._anchored[tag]
        known = self._anchored.get(tag)
        audit = self._audit_log()
        if not self._audit_here(audit):
            if known is not None and known["seen"] is not None:
                raise StoreRefused("audit", "the audit log is gone after its anchors were read")
            found = {"latest": None, "latest_id": None, "seen": None, "union": anchors.empty_union()}
        else:
            self._trust(audit, owner)
            key_id, _anchor_key, seal = self._soil_keys(soil)
            opener = None
            if key_id != NOKEY:
                try:
                    opener = self._cipher_pair()[1]
                except MembraneRefused:
                    raise StoreRefused("audit", "no cipher backend opens the anchors") from None
            try:
                found = anchors.anchors(audit, tag, key_id, seal, opener, known=known)
            except StoreRefused:
                raise
            except Exception:  # noqa: BLE001
                raise StoreRefused("audit", "the audit log cannot be read") from None
        self._anchored[tag] = found
        if fresh is not None:
            fresh[tag] = True
        return found

    def _remember_anchor(self, tag, entry):
        from . import anchors

        found = self._anchored.get(tag)
        if found is None:
            return
        self._anchored[tag] = dict(found, latest=(entry["gen"], entry["seq"], entry["head"]),
                                   union=anchors.merge(found["union"], entry))

    def _cross_seq(self, tag, owner=None, soil=None, fresh=None):
        """The seq the latest cross-anchor of a being names, or -1; brought up to date with the audit log first.

        It only widens what a refusal says it would set aside, so an audit
        log that cannot be read here reads as what this store remembers: the
        lookup that decides (step 9 of verification) refuses it by name.
        """
        found = None
        if owner is not None:
            try:
                found = self._anchors_of(tag, owner, soil, fresh)
            except StoreRefused:
                found = None
        if found is None:
            found = self._anchored.get(tag)
        latest = found["latest"] if found else None
        return latest[1] if latest else -1

    # -- connections ---------------------------------------------------------

    def _open_raw(self, connect, path, settings, failure):
        timeout = (settings["busy_timeout_ms"] + 999) // 1000
        try:
            conn = connect(str(path), check_same_thread=False, timeout=timeout)
        except StoreRefused:
            raise
        except Exception:  # noqa: BLE001 - the keyed connect raises on a wrong key or a foreign file
            raise failure from None
        try:
            conn.isolation_level = None
        except Exception:  # noqa: BLE001
            _close_quietly(conn)
            raise failure from None
        return conn

    def _pragmas(self, conn, settings):
        conn.execute(f"PRAGMA busy_timeout = {int(settings['busy_timeout_ms'])}").fetchall()
        conn.execute("PRAGMA journal_mode = DELETE").fetchall()
        conn.execute("PRAGMA secure_delete = ON").fetchall()
        conn.execute("PRAGMA temp_store = MEMORY").fetchall()
        conn.execute("PRAGMA foreign_keys = ON").fetchall()

    def _warm(self):
        from . import engine, wire

        engine.call(wire.emit({"op": "engine", "v": 1}))

    def _page_check(self, conn, path, soil):
        key = str(path)
        if self._pages.get(key):
            return
        if soil == "encrypted":
            # Only after the connection answered a cipher version: on plain SQLite this pragma answers
            # nothing for any file, which would read as a pass.
            if conn.execute("PRAGMA cipher_integrity_check").fetchall() != []:
                raise StoreRefused("pages", "the page check found damaged pages")
        elif [tuple(row) for row in conn.execute("PRAGMA quick_check").fetchall()] != [("ok",)]:
            raise StoreRefused("pages", "the page check found damaged pages")
        self._pages[key] = True

    def _connect_existing(self, path, soil, mode, settings):
        """A connection to an existing store file, judged in order before anything is trusted."""
        if soil == "glass":
            if mode != "daily":
                raise StoreRefused("sealed", "a glass jar is sealed in Bulbe")
            if self._probe(path) != "plaintext":
                raise StoreRefused("soil", "the glass jar is not a plain database")
            conn = self._open_raw(self._plain_connect, path, settings,
                                  StoreRefused("soil", "the glass jar cannot be opened"))
            answer, refusal = "plaintext", StoreRefused("soil", "the glass jar answers a cipher")
        else:
            if not self._cipher_ok():
                raise StoreRefused("cipher", "SQLCipher is not available in this Python")
            state = self._key_state()[0]
            if state != "readable":
                raise StoreRefused("key", "the master key cannot be read" if state == "unreadable"
                                   else "no master key is configured")
            verdict = self._probe(path)
            if verdict == "short":
                raise StoreRefused("soil", "short file")
            if verdict != "encrypted":
                raise StoreRefused("soil", "plaintext under an encrypted name")
            conn = self._open_raw(self._connect, path, settings, StoreRefused("key", "key verification failed"))
            answer, refusal = "encrypted", StoreRefused("cipher", "the connection answered no cipher version")
        try:
            if self._probe(path, conn) != answer:
                raise refusal
            _guarded(conn, lambda: self._pragmas(conn, settings))
            self._warm()
            _guarded(conn, lambda: self._page_check(conn, path, soil))
        except BaseException:
            _close_quietly(conn)
            raise
        return conn

    def _verify(self, conn, path, tag, soil, prior):
        """Verification in full: the chain (steps 1-7), then the audit log (steps 8-10)."""
        from . import chain

        key_id, anchor_key, _seal = self._soil_keys(soil)
        fresh = {}

        def cross_seq(being_tag):
            return self._cross_seq(being_tag, tag, soil, fresh)

        verified = _guarded(conn, lambda: chain.verify(conn, name_tag=tag, name_soil=soil, key_id=key_id,
                                                       anchor_key=anchor_key, cross_seq=cross_seq, prior=prior))
        owner = verified.meta["owner"]
        self._check_sown(owner, verified.being_tag)
        found = self._anchors_of(verified.being_tag, owner, soil, fresh)
        _guarded(conn, lambda: chain.check_cross(verified, found["latest"], lambda seq: chain.link_at(conn, seq)))
        _guarded(conn, lambda: chain.check_destroyed(verified, found["union"],
                                                     lambda eid: chain.body_is_null(conn, eid)))
        return verified

    def _check_sown(self, owner, being_tag):
        """Step 8: the latest open birth of this account names this very being."""
        sown = self._sow_state(owner)
        if sown != ("open", being_tag):
            sown = self._sow_state(owner, refresh=True)
        if sown[0] == "foreign":
            raise StoreRefused("foreign", "this account's being was sown under another key")
        if sown[0] != "open":
            raise StoreRefused("unanchored", "no open birth of this account is anchored")
        if sown[1] != being_tag:
            raise StoreRefused("unsown", "the anchored birth names another being")

    # -- after a commit: the small transactions, VACUUM, the cross-anchor ------

    def _small_work(self, soil, change):
        """A small committing transaction on meta: ``change`` runs, the generation moves, the anchor is rewritten."""

        def work(conn):
            meta = _read_meta(conn)
            change(conn, meta)
            gen = meta["gen"] + 1
            _put(conn, "gen", gen)
            key_id, anchor_key, _seal = self._soil_keys(soil)
            _rewrite_anchor(conn, being=meta["being"], owner=meta["owner"], soil=soil, gen=gen, key_id=key_id,
                            anchor_key=anchor_key)

        return work

    def _vacuum_on(self, conn, soil, write):
        """VACUUM after a destruction, whose own transaction already wrote ``vacuum_owed = 1``.

        Only a VACUUM that ran clears the debt, in a small transaction of its
        own (``write(work)`` runs one transaction on the same connection). A
        VACUUM that fails, or never runs because the process stops, leaves the
        debt on record: the status says ``vacuum_owed`` and the next write
        runs it again. It is never said to be done when it was not.
        """
        try:
            conn.execute("VACUUM")
        except Exception as exc:  # noqa: BLE001 - owed, said, and retried at the next write
            logger.warning("the componion's store could not be vacuumed after a destruction: %s", exc)
            return
        try:
            write(self._small_work(soil, lambda conn, meta: _put(conn, "vacuum_owed", 0)))
        except StoreRefused as refusal:
            logger.warning("the componion's store was vacuumed and its debt is not yet cleared: %s", refusal)

    def _anchor_across(self, soil, being_tag, why, info, pending, write):
        """Write a cross-anchor to the audit log, then settle ``meta.cross`` and the pending records.

        The journal never waits on the audit log: a failure is logged, the
        records stay pending, and the anchor is retried at the next write.
        """
        from . import anchors

        union = anchors.empty_union()
        for item in pending:
            if isinstance(item, dict) and all(name in item for name in ("destroyed", "ended", "forgot", "rhythm_floor")):
                union = anchors.merge(union, item)
        entry = anchors.record(destroyed=union["destroyed"], ended=info["ended"], forgot=union["forgot"],
                               gen=info["gen"], head=info["head"], rhythm_floor=info["floor"], seq=info["seq"],
                               why=why)
        try:
            key_id, _anchor_key, seal = self._soil_keys(soil)
            seal_fn = self._cipher_pair()[0] if key_id != NOKEY else None
            written = anchors.write_anchor(self._audit_log(), tag=being_tag, key_id=key_id, seal=seal,
                                           seal_fn=seal_fn, entry=entry)
        except Exception as exc:  # noqa: BLE001 - the journal never waits on the audit log
            logger.warning("the componion's cross-anchor was not written (%s); it is retried at the next write", exc)
            return
        if not written:
            logger.warning("the componion's cross-anchor was not accepted; it is retried at the next write")
            return
        self._remember_anchor(being_tag, entry)

        def settle(conn, meta):
            cross = meta.get("cross") if isinstance(meta.get("cross"), dict) else {}
            day = cross.get("day", -1) if _is_int(cross.get("day", -1)) else -1
            _put(conn, "cross", {"day": info["day"] if info["day"] > day else day, "gen": info["gen"],
                                 "seq": info["seq"]})
            current = meta.get("cross_pending", [])
            if isinstance(current, list) and current[:len(pending)] == pending:
                current = current[len(pending):]
            _put(conn, "cross_pending", current)

        try:
            write(self._small_work(soil, settle))
        except StoreRefused as refusal:
            logger.warning("the componion's cross-anchor was written and not yet recorded: %s", refusal)

    def _open_being(self, tag, soil, path, mode, settings):
        key = str(path)
        being = self._beings.get(key)
        try:
            if being is not None and being._conn is not None:
                try:
                    being._verified = self._verify(being._conn, path, tag, soil, being._verified)
                except BaseException:
                    self._forget(being)
                    raise
                return being
            conn = self._connect_existing(path, soil, mode, settings)
            try:
                verified = self._verify(conn, path, tag, soil, None)
            except BaseException:
                _close_quietly(conn)
                raise
        except StoreRefused as refusal:
            if refusal.code == "pages":
                self._pages.pop(key, None)
            raise
        being = BeingStore(self, path=path, tag=tag, soil=soil, conn=conn, verified=verified)
        self._beings[key] = being
        return being

    def _forget(self, being):
        """Close a being's connection and drop it from this store."""
        key = str(being.path)
        if self._beings.get(key) is being:
            del self._beings[key]
        _close_quietly(being._conn)
        being._conn = None

    def _forget_path(self, path):
        being = self._beings.get(str(path))
        if being is not None:
            self._forget(being)

    # -- looking -------------------------------------------------------------

    def _choose_soil(self, mode, settings):
        state = self._key_state()[0]
        cipher = self._cipher_ok()
        if state == "readable" and cipher:
            return "encrypted"
        if state == "unreadable":
            raise StoreRefused("key", "a key is configured and this process cannot read it")
        if state == "readable":
            raise StoreRefused("cipher", "a key is readable and SQLCipher is not available")
        if mode == "daily" and settings["require_encryption"] is False:
            return "glass"
        raise StoreRefused("no_soil", "no key is configured: the being waits for its soil")

    def _inspect_temp(self, directory, tag, soil, mode, settings):
        """``(path, state, being_tag)`` of a sowing file: absent, link, leftover, sealed, garbage or complete."""
        from . import chain

        temp = store_file(directory, tag, "sowing_" + soil)
        if _is_link(temp):
            return temp, "link", None
        if not temp.exists():
            return temp, "absent", None
        final = store_file(directory, tag, soil)
        try:
            if final.exists() and os.path.samefile(temp, final):
                return temp, "leftover", None
        except OSError:
            pass
        if soil == "glass" and mode != "daily":
            return temp, "sealed", None
        conn = None
        try:
            conn = self._connect_existing(temp, soil, mode, settings)
            key_id, anchor_key, _seal = self._soil_keys(soil)
            verified = _guarded(conn, lambda: chain.verify(conn, name_tag=tag, name_soil=soil, key_id=key_id,
                                                           anchor_key=anchor_key))
        except Exception:  # noqa: BLE001 - a sowing file that does not verify is garbage
            return temp, "garbage", None
        finally:
            self._pages.pop(str(temp), None)
            _close_quietly(conn)
        return temp, "complete", verified.being_tag

    def _finish_offer(self, directory, tag, being_tag, mode, settings):
        for soil in SOILS:
            _temp, state, found = self._inspect_temp(directory, tag, soil, mode, settings)
            if state == "complete" and found == being_tag:
                return Finish(being_tag[:8])
        return None

    def _leftover(self, directory, tag):
        for soil in SOILS:
            temp = store_file(directory, tag, "sowing_" + soil)
            final = store_file(directory, tag, soil)
            try:
                if temp.exists() and final.exists() and os.path.samefile(temp, final):
                    return True
            except OSError:
                pass
        return False

    def _resume_offer(self, refusal):
        kept_days = 0
        wall = self._read_clock()
        if _is_int(refusal.birth) and _is_int(wall) and _is_int(refusal.kept_t):
            now_t = (wall - refusal.birth) // 60
            kept_days = ((now_t if now_t > 0 else 0) - refusal.kept_t) // 1440
        return Resume(refusal.kept_seq, kept_days, refusal.discarded)

    def _outcome(self, tag, soil, path, mode, settings, hints):
        try:
            being = self._open_being(tag, soil, path, mode, settings)
        except ChainRefused as refusal:
            offer = self._resume_offer(refusal) if refusal.kept_seq >= 0 else None
            return _status("unreadable", reason=refusal.reason, detail=str(refusal), hints=hints,
                           offer=offer), None, refusal
        except StoreRefused as refusal:
            if refusal.code in UNREADABLE:
                name = "unreadable"
            elif refusal.code == "sealed":
                name = "sealed_bulbe"
            else:
                name = "unavailable"
            return _status(name, reason=refusal.code, detail=str(refusal), hints=hints), None, refusal
        labels = []
        hints = list(hints)
        if soil == "glass":
            labels.append("glass_jar")
            if (self._key_state()[0] == "readable" and self._cipher_ok()) or settings["require_encryption"]:
                hints.append("repot")
        if being.provisional:
            labels.append("prototype")
        meta = being._verified.meta
        if meta.get("vacuum_owed") == 1:
            hints.append("vacuum_owed")
        pending = meta.get("cross_pending")
        if isinstance(pending, list) and pending:
            hints.append("anchor_owed")
        return _status("alive", labels=labels, hints=hints), being, None

    def _examine(self, user, mode):
        """``(Status, being or None, refusal or None)`` for one account; writes and creates nothing."""
        from . import anchors
        from .membrane import LOCAL_USER

        def unavailable(refusal):
            return _status("unavailable", reason=refusal.code, detail=str(refusal)), None, refusal

        try:
            settings = self._settings()
            _root, directory = self._directory()
        except StoreRefused as refusal:
            return unavailable(refusal)
        if not self._single_user_now() and user == LOCAL_USER:
            return unavailable(StoreRefused("unclaimed", "the local being is no account's once accounts exist"))
        try:
            tag = anchors.owner_tag(user)
            encrypted = store_file(directory, tag, "encrypted")
            glass = store_file(directory, tag, "glass")
            hints = ["leftover"] if self._leftover(directory, tag) else []
        except StoreRefused as refusal:
            return unavailable(refusal)
        if encrypted.exists() and glass.exists():
            refusal = StoreRefused("soil", "two stores: one encrypted, one glass")
            return _status("unreadable", reason="soil", detail=str(refusal), hints=hints), None, refusal
        if glass.exists():
            if mode != "daily":
                self._forget_path(glass)
                refusal = StoreRefused("sealed", "a glass jar is sealed in Bulbe")
                return _status("sealed_bulbe", reason="sealed", detail=str(refusal), hints=hints), None, refusal
            return self._outcome(tag, "glass", glass, mode, settings, hints)
        if encrypted.exists():
            return self._outcome(tag, "encrypted", encrypted, mode, settings, hints)
        state = self._key_state()[0]
        if state == "unreadable":
            return unavailable(StoreRefused("key", "a key is configured and this process cannot read it"))
        if state == "readable" and not self._cipher_ok():
            return unavailable(StoreRefused("cipher", "a key is readable and SQLCipher is not available"))
        try:
            sown = self._sow_state(tag, refresh=True)
        except StoreRefused as refusal:
            return unavailable(refusal)
        if sown[0] == "foreign":
            return unavailable(StoreRefused("foreign", "this account's being was sown under another key"))
        if sown[0] == "open":
            detail = f"the store of a sown being is missing: {encrypted.name} or {glass.name}"
            offer = self._finish_offer(directory, tag, sown[1], mode, settings)
            return _status("missing", detail=detail, hints=hints, offer=offer), None, StoreRefused("exists", detail)
        try:
            self._choose_soil(mode, settings)
        except StoreRefused as refusal:
            if refusal.code == "no_soil":
                return _status("awaiting_soil", reason="no_soil", detail=str(refusal), hints=hints), None, refusal
            return unavailable(refusal)
        return _status("ready", hints=hints), None, None

    def status(self, user):
        """What ``user``'s being is; writes nothing and creates nothing."""
        with self._lock:
            return self._examine(user, self._read_mode())[0]

    def open(self, user):
        """``user``'s being; ``None`` only when there is none (``ready``); otherwise the refusal."""
        with self._lock:
            status, being, refusal = self._examine(user, self._read_mode())
        if status.status == "alive":
            return being
        if status.status == "ready":
            return None
        raise refusal

    def unclaimed(self, *, transport):
        """The local being left without an account, ``[(owner_tag, short being tag)]``: listed, never adopted.

        The listing is read from the audit log alone. Outside Daily a birth in
        a glass jar (sown under ``nokey``) is left out, as the jar itself is
        sealed there.
        """
        from . import anchors
        from .membrane import LOCAL_USER, MembraneRefused, surface_of

        with self._lock:
            mode = self._read_mode()
            now = self._read_clock()
            surface = surface_of(transport, now)
            principal = transport.principal if isinstance(transport.principal, dict) else {}
            admin = surface == "web_session" and principal.get("role") == "admin"
            attended = surface == "cli_tty" and transport.attended is True
            if not (admin or attended):
                raise MembraneRefused("owner", "only an administrator lists the unclaimed being")
            owner = anchors.owner_tag(LOCAL_USER)
            state, key = self._sow_read(owner)
            if state[0] != "open" or (key == NOKEY and mode != "daily"):
                return []
            return [(owner, state[1][:8])]

    def close(self):
        """Close every connection this store holds."""
        with self._lock:
            for being in list(self._beings.values()):
                self._forget(being)

    # -- sowing --------------------------------------------------------------

    def _take_lock(self, directory):
        import fcntl

        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            handle = os.open(directory.joinpath(LOCK_NAME), flags, 0o600)
        except OSError:
            raise StoreRefused("path", "the sowing lock cannot be opened") from None
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(handle)
            raise StoreRefused("busy", "a sowing is in progress") from None
        return handle

    def _release_lock(self, handle):
        import fcntl

        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)

    def _sort_temps(self, directory, tag, mode, settings):
        """Step 7: a leftover or linked name is unlinked, garbage deleted; a finishable sowing is returned.

        A temp name that is a symbolic link is removed itself: what it names is
        never opened, and never deleted.
        """
        finishable = None
        for soil in SOILS:
            temp, state, found = self._inspect_temp(directory, tag, soil, mode, settings)
            if state in ("leftover", "link"):
                os.unlink(temp)
            elif state == "garbage":
                _remove_with_journals(temp)
            elif state == "complete":
                if self._sow_state(tag, refresh=True) == ("open", found):
                    finishable = (temp, soil, found)
                else:
                    _remove_with_journals(temp)
        return finishable

    def _schema(self, conn):
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in SCHEMA:
                conn.execute(statement)
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            raise

    def _plaintext_cause(self, path, conn, soil, mode):
        header, size = _header(path)
        if len(header) < HEADER_BYTES or size < MIN_FILE_BYTES:
            cause = "the file is shorter than a database header"
        elif soil == "glass":
            cause = "the glass jar was not written in clear"
        else:
            try:
                answers = bool(conn.execute("PRAGMA cipher_version").fetchall())
            except Exception:  # noqa: BLE001
                answers = False
            if not answers:
                cause = "the connection answered no cipher version (no SQLCipher binding in this Python)"
            else:
                cause = "the connection is not keyed: the file was written in clear"
        if self._key_state()[0] == "none" and mode == "daily":
            cause += "; with no key configured, persistence.require_encryption: false would allow a glass jar"
        return cause

    def _plant(self, temp, *, soil, tag, pin, wall, tz_minutes, rhythm_consent, sowing, mode, settings):
        """Steps 9 to 12: the file, its schema and probe, the draws, the genesis, the seed anchor."""
        from . import anchors, membrane

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            os.close(os.open(temp, flags, 0o600))
        except FileExistsError:
            raise StoreRefused("exists", "a sowing file is in the way") from None
        conn = None
        try:
            if soil == "encrypted":
                conn = self._open_raw(self._connect, temp, settings, StoreRefused("key", "key verification failed"))
            else:
                conn = self._open_raw(self._plain_connect, temp, settings,
                                      StoreRefused("soil", "the glass jar cannot be opened"))
            _guarded(conn, lambda: self._schema(conn))
            if self._probe(temp, conn) != ("encrypted" if soil == "encrypted" else "plaintext"):
                raise PlaintextRefused(self._plaintext_cause(temp, conn, soil, mode))
            _guarded(conn, lambda: self._pragmas(conn, settings))
            self._warm()
            # Only now draw: a refused soil never draws.
            being = self._draw(16).hex()
            origin = self._draw(8).hex()
            secret = self._draw(32)
            seed = self._draw(32).hex()
            key_id, anchor_key, _seal = self._soil_keys(soil)
            genesis = {
                "band": sowing["band"],
                "birth": {"tz": tz_minutes, "wall": wall},
                "derive": 1,
                "hemisphere": sowing["hemisphere"],
                "laws": {"name": pin["name"], "params": {name: spec["default"] for name, spec in pin["params"].items()},
                         "provisional": pin["provisional"], "sha256": pin["sha256"], "v": pin["version"]},
                "owner": tag,
                "rhythm_consent": rhythm_consent,
                "seed": seed,
                "soil": soil,
                "weather": sowing["weather"],
            }
            membrane.check_body(pin["table"]["kinds"]["genesis"]["body"], genesis)
            _guarded(conn, lambda: self._write_genesis(conn, soil=soil, tag=tag, being=being, origin=origin,
                                                       secret=secret, genesis=genesis, laws=pin["version"],
                                                       key_id=key_id, anchor_key=anchor_key))
        except BaseException:
            _close_quietly(conn)
            _remove_with_journals(temp)
            raise
        _close_quietly(conn)
        being_tag = anchors.being_tag(secret)
        try:
            written = anchors.write_sow(self._audit_log(), being=being_tag, key_id=key_id, anchor_key=anchor_key,
                                        owner=tag)
        except Exception:  # noqa: BLE001 - an entry that was not accepted anchors nothing
            written = False
        if not written:
            _remove_with_journals(temp)
            raise StoreRefused("anchor_unwritten", "the birth could not be anchored in the audit log; nothing was sown")
        self._sown[tag] = ("open", being_tag)
        self._anchored[being_tag] = {"latest": None, "latest_id": None, "seen": None, "union": anchors.empty_union()}
        return being_tag

    def _write_genesis(self, conn, *, soil, tag, being, origin, secret, genesis, laws, key_id, anchor_key):
        from .chain import ZERO

        conn.execute("BEGIN IMMEDIATE")
        try:
            meta = {
                "being": being, "cross": {"day": -1, "gen": 0, "seq": -1}, "cross_pending": [], "gen": 1,
                "heard_ended": [], "origin": origin, "oseq_next:" + origin: 1, "owner": tag, "rhythm_floor": 0,
                "schema": SCHEMA_VERSION, "segment": 0, "soil": soil, "vacuum_owed": 0,
            }
            for key in sorted(meta):
                _put(conn, key, meta[key])
            conn.execute("INSERT INTO keys (name, key) VALUES ('being_secret', ?)", (secret,))
            _insert_fact(conn, being=being, origin=origin, oseq=0, t=0, kind="genesis", body=genesis, laws=laws,
                         head=(-1, ZERO))
            _rewrite_anchor(conn, being=being, owner=tag, soil=soil, gen=1, key_id=key_id, anchor_key=anchor_key)
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            raise

    def _link(self, temp, final):
        """Step 13: link the anchored file into place; after the anchor, a failure never deletes it."""
        self._stage("link")
        try:
            os.link(temp, final)
        except FileExistsError:
            raise StoreRefused("exists", "a store already has this name") from None
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.EXDEV, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)):
                raise
            if final.exists():
                raise StoreRefused("exists", "a store already has this name") from None
            os.rename(temp, final)
            return
        os.unlink(temp)

    def _person(self, transport, action):
        """The surface check and the actor of a verb; ``(user, owner tag, clock reading)``."""
        from . import anchors
        from .membrane import VERBS, MembraneRefused, actor_of, surface_of

        now = self._read_clock()
        surface = surface_of(transport, now)
        if surface not in VERBS[action]:
            raise MembraneRefused("surface", f"{action} is not allowed from {surface}")
        user = actor_of(transport, self._single_user_now(), now)
        return user, anchors.owner_tag(user), now

    def sow(self, *, transport, law, tz_minutes, rhythm_consent, hemisphere=None, band=None, weather=None):
        """Sow a being for the account behind ``transport``, and return it open.

        Every refusal is named; nothing is created before the soil is chosen,
        nothing is drawn before the file is proven encrypted (or a glass jar),
        and the store is never linked into place before its birth is anchored.

        ``hemisphere``, ``band`` and ``weather`` are the being's identity,
        frozen into its genesis with the law's default params; a missing one
        reads north, long and garden.
        """
        from .membrane import MembraneRefused, law_pin

        with self._lock:
            mode = self._read_mode()
            user, tag, wall = self._person(transport, "sow")
            pin = law_pin(law)
            if not _is_int(tz_minutes) or not -840 <= tz_minutes <= 840 or tz_minutes % 15:
                raise MembraneRefused("clock", "the time zone is whole quarter hours within 14 hours of UTC")
            if not _is_int(wall) or not 0 <= wall <= MAX_INT:
                raise MembraneRefused("clock", "unreadable")
            if not isinstance(rhythm_consent, bool):
                raise MembraneRefused("body", "body rhythm_consent bool")
            sowing = {"band": "long" if band is None else band,
                      "hemisphere": "north" if hemisphere is None else hemisphere,
                      "weather": "garden" if weather is None else weather}
            genesis_schema = pin["table"]["kinds"]["genesis"]["body"]
            for field in sorted(sowing):
                if not isinstance(sowing[field], str) or sowing[field] not in genesis_schema[field]["of"]:
                    raise MembraneRefused("body", f"body {field} symbol")
            settings = self._settings()
            soil = self._choose_soil(mode, settings)
            _root, directory = self._directory()
            _make_directory(directory)
            handle = self._take_lock(directory)
            try:
                if self._sort_temps(directory, tag, mode, settings) is not None:
                    raise StoreRefused("exists", "an interrupted sowing can be finished")
                status = self._examine(user, mode)[0]
                if status.status != "ready":
                    raise StoreRefused("exists", status.status)
                temp = store_file(directory, tag, "sowing_" + soil)
                final = store_file(directory, tag, soil)
                self._plant(temp, soil=soil, tag=tag, pin=pin, wall=wall, tz_minutes=tz_minutes,
                            rhythm_consent=rhythm_consent, sowing=sowing, mode=mode, settings=settings)
                self._link(temp, final)
            finally:
                self._release_lock(handle)
            status, being, refusal = self._examine(user, mode)
            if being is None:
                raise refusal if refusal is not None else StoreRefused("exists", status.status)
            return being

    def finish_sowing(self, *, transport, confirm):
        """Link an interrupted, anchored sowing into place, once the person confirms its short tag."""
        with self._lock:
            mode = self._read_mode()
            user, tag, _now = self._person(transport, "finish_sowing")
            settings = self._settings()
            _root, directory = self._directory()
            if not directory.is_dir():
                raise ResumeRefused("nothing", "no interrupted sowing to finish")
            handle = self._take_lock(directory)
            try:
                finishable = self._sort_temps(directory, tag, mode, settings)
                if finishable is None:
                    raise ResumeRefused("nothing", "no interrupted sowing to finish")
                temp, soil, found = finishable
                if confirm != found[:8]:
                    raise ResumeRefused("confirm", "the confirmation names another being")
                self._link(temp, store_file(directory, tag, soil))
            finally:
                self._release_lock(handle)
            status, being, refusal = self._examine(user, mode)
            if being is None:
                raise refusal if refusal is not None else StoreRefused("exists", status.status)
            return being

    # -- restore, never repair -----------------------------------------------

    def resume(self, *, transport, confirm):
        """Resume a refused being from its last sound event, once the person confirms what they were shown.

        ``confirm`` is ``(kept_seq, discarded)`` as the ``Resume`` offer gave
        them. Verification runs again first: a being that opens has nothing
        to resume (``ResumeRefused("nothing")``), other numbers than the
        store's own now are ``ResumeRefused("confirm")``, and a refusal a
        resume cannot answer is raised as it is. Then, in one transaction,
        the tail above the kept event is set aside, a forget whose forgetter
        went with it is issued again, every recorded destruction is enforced,
        what the structural checks would refuse is discarded, the checkpoints
        are dropped, and a ``resumed`` fact is written past every event
        anything names. VACUUM and a cross-anchor follow. Nothing runs by
        itself: only this call writes, and only when the numbers agree.
        """
        from . import anchors, membrane

        with self._lock:
            mode = self._read_mode()
            user, tag, now = self._person(transport, "resume")
            status, being, refusal = self._examine(user, mode)
            if being is not None:
                raise ResumeRefused("nothing", "the being opens: there is nothing to resume")
            if refusal is None:
                raise ResumeRefused("nothing", "there is no store to resume")
            if not isinstance(refusal, ChainRefused) or refusal.kept_seq < 0:
                raise refusal
            if (not isinstance(confirm, (tuple, list)) or len(confirm) != 2 or not _is_int(confirm[0])
                    or not _is_int(confirm[1]) or tuple(confirm) != (refusal.kept_seq, refusal.discarded)):
                raise ResumeRefused("confirm", "the numbers confirmed are not the ones the store shows now")
            settings = self._settings()
            _root, directory = self._directory()
            path, soil = self._store_of(directory, tag)
            try:
                _tighten(directory)
            except OSError:
                raise StoreRefused("path", "the store directory cannot be tightened") from None
            conn = self._connect_existing(path, soil, mode, settings)
            try:
                secret, genesis = _guarded(conn, lambda: self._resume_reads(conn))
                wall = membrane.recorder_wall(now, genesis["birth"]["wall"])
                being_tag = anchors.being_tag(secret)
                self._check_sown(tag, being_tag)
                found = self._anchors_of(being_tag, tag, soil)
                pin = membrane.law_pin(genesis["laws"]["name"])
                self._stage("resume")
                info = _transaction(conn, lambda c: self._resume_in(
                    c, refusal=refusal, tag=tag, soil=soil, genesis=genesis, pin=pin, found=found, wall=wall))

                def write(work):
                    return _transaction(conn, work)

                self._vacuum_on(conn, soil, write)
                self._anchor_across(soil, being_tag, "resume", info, info["pending"], write)
            except StoreRefused as refused:
                if refused.code == "pages":
                    self._pages.pop(str(path), None)
                raise
            finally:
                _close_quietly(conn)
            self._forget_path(path)
            status, being, refusal = self._examine(user, mode)
            if being is None:
                raise refusal if refusal is not None else StoreRefused("exists", status.status)
            return being

    def _store_of(self, directory, tag):
        """The existing store file of an account and its soil (the glass name first, as looking reads it)."""
        glass = store_file(directory, tag, "glass")
        if glass.exists():
            return glass, "glass"
        return store_file(directory, tag, "encrypted"), "encrypted"

    def _resume_reads(self, conn):
        """The being's secret and its genesis, read before the resume transaction."""
        from . import chain

        row = conn.execute("SELECT key FROM keys WHERE name = 'being_secret'").fetchone()
        if row is None or not isinstance(row[0], bytes) or len(row[0]) != 32:
            raise StoreRefused("local", "the being's secret is gone: nothing can be resumed")
        first = conn.execute("SELECT b.body FROM links l JOIN bodies b ON b.eid = l.eid WHERE l.seq = 0").fetchone()
        genesis = None if first is None or first[0] is None else chain._strict_object(first[0])
        laws = genesis.get("laws") if isinstance(genesis, dict) else None
        birth = genesis.get("birth") if isinstance(genesis, dict) else None
        if (not isinstance(laws, dict) or not isinstance(laws.get("name"), str) or not _is_int(laws.get("v"))
                or not isinstance(birth, dict) or not _is_int(birth.get("wall"))):
            raise StoreRefused("local", "the genesis cannot be read: nothing can be resumed")
        return bytes(row[0]), genesis

    def _resume_in(self, conn, *, refusal, tag, soil, genesis, pin, found, wall):
        """The resume transaction; what the cross-anchor after it needs."""
        from . import anchors, chain, membrane, wire

        meta = _read_meta(conn)
        if meta.get("gen") != refusal.gen:
            raise ResumeRefused("confirm", "the store changed since it was verified: look at it again")
        if meta.get("owner") != tag:
            raise StoreRefused("owner", "the store names another owner")
        if meta.get("soil") != soil:
            raise StoreRefused("soil", "the store names another soil")
        origin = meta.get("origin")
        if not _is_hex(origin, 16) or not _is_hex(meta.get("being"), 32) or not _is_int(meta.get("segment")):
            raise StoreRefused("local", "the store's own identity cannot be read")
        floor = meta.get("rhythm_floor", 0)
        ended = meta.get("heard_ended", [])
        if not _is_int(floor) or not isinstance(ended, list) or not all(_is_int(item) for item in ended):
            raise StoreRefused("local", "the rhythm floor or the ended seasons cannot be read")
        # Every record, anchored or pending, is enforced; a pending record that is not whole cannot be.
        union = found["union"]
        pending = []
        current = meta.get("cross_pending", [])
        for item in current if isinstance(current, list) else []:
            if chain.pending_whole(item):
                union = anchors.merge(union, item)
                pending.append(item)
            else:
                logger.warning("a pending destruction record that is not whole was set aside by the resume")
        done = chain.resume_discard(conn, kept_seq=refusal.kept_seq, kinds=pin["table"]["kinds"], union=union,
                                    floor=floor, ended=ended)
        prev = chain.link_at(conn, refusal.kept_seq)
        t = membrane.recorder_t(wall, genesis["birth"]["wall"], conn.execute(T_MAX).fetchone()[0])
        name = "oseq_next:" + origin
        oseq = meta.get(name, 0) if _is_int(meta.get(name, 0)) else 0
        most = conn.execute("SELECT MAX(oseq) FROM facts WHERE origin = ?", (origin,)).fetchone()[0]
        if _is_int(most) and most + 1 > oseq:
            oseq = most + 1
        removed = done["removed"]
        body = {"digest": hashlib.sha256(wire.emit(removed)).hexdigest(), "removed": len(removed)}
        membrane.check_body(pin["table"]["kinds"]["resumed"]["body"], body)
        laws = genesis["laws"]["v"]
        # Past every event anything names, so an anchor inside the gap is satisfied by this resume.
        _eid, seq, link = _insert_fact(conn, being=meta["being"], origin=origin, oseq=oseq, t=t, kind="resumed",
                                       body=body, laws=laws, head=(refusal.through, prev))
        oseq += 1
        for target in done["reissue"]:
            forgetter, seq, link = _insert_fact(conn, being=meta["being"], origin=origin, oseq=oseq, t=t,
                                                kind="lang_forget", body={"target": target}, laws=laws,
                                                head=(seq, link))
            conn.execute("UPDATE bodies SET redacted_by = ? WHERE eid = ?", (forgetter, target))
            oseq += 1
        latest = found["latest"]
        gen = (latest[0] if latest is not None and latest[0] > meta["gen"] else meta["gen"]) + 1
        record = {"destroyed": done["destroyed"], "ended": done["ended"], "forgot": done["reissue"],
                  "rhythm_floor": done["rhythm_floor"]}
        pending.append(record)
        _put(conn, name, oseq)
        _put(conn, "segment", meta["segment"] + 1)
        _put(conn, "rhythm_floor", done["rhythm_floor"])
        _put(conn, "heard_ended", done["ended"])
        _put(conn, "cross_pending", pending)
        _put(conn, "vacuum_owed", 1)
        _put(conn, "gen", gen)
        key_id, anchor_key, _seal = self._soil_keys(soil)
        _rewrite_anchor(conn, being=meta["being"], owner=tag, soil=soil, gen=gen, key_id=key_id,
                        anchor_key=anchor_key)
        logger.info("the componion's being was resumed from event #%d: %d events set aside",
                    refusal.kept_seq, len(removed))
        return {"cross": meta.get("cross"), "day": membrane.day_of(t), "destroyed": True, "ended": done["ended"],
                "floor": done["rhythm_floor"], "gen": gen, "head": link, "pending": pending, "seq": seq, "t": t,
                "vacuum_owed": 1}


# ---------------------------------------------------------------------------
# One being
# ---------------------------------------------------------------------------
class BeingStore:
    """One open being: its connection, what its verification read, and the membrane's writes into it."""

    def __init__(self, store, *, path, tag, soil, conn, verified):
        genesis = verified.genesis
        self._store = store
        self._conn = conn
        self._verified = verified
        self._tightened = False
        self.path = path
        self.tag = tag
        self.soil = soil
        self.being = verified.meta["being"]
        self.origin = verified.meta["origin"]
        self.owner = verified.meta["owner"]
        self.being_tag = verified.being_tag
        self.birth_wall = genesis["birth"]["wall"]
        self.laws = genesis["laws"]["v"]
        self.law = genesis["laws"]["name"]
        self.provisional = genesis["laws"]["provisional"]
        self.rhythm_consent = genesis["rhythm_consent"]

    # -- the connection and its transactions ---------------------------------

    def _live(self):
        if self._conn is None:
            raise StoreRefused("busy", "this being's store is closed; open it again")
        return self._conn

    def _write(self, work):
        """Run ``work(conn)`` in one ``BEGIN IMMEDIATE`` transaction; any exception rolls back."""
        conn = self._live()
        try:
            return _transaction(conn, work)
        except StoreRefused as refusal:
            if refusal.code == "pages":
                self._store._pages.pop(str(self.path), None)
                self._store._forget(self)
            raise

    def _gate(self):
        """Read the mode once for this action: a glass jar outside Daily is sealed, and its connection closed."""
        store = self._store
        mode = store._read_mode()
        if self.soil == "glass" and mode != "daily":
            store._forget(self)
            raise StoreRefused("sealed", "a glass jar is sealed in Bulbe")
        return mode

    def _head(self, conn):
        return _head(conn)

    def _meta(self, conn):
        return _read_meta(conn)

    def _put(self, conn, key, value):
        _put(conn, key, value)

    def _insert_fact(self, conn, *, origin, oseq, t, kind, body, head):
        return _insert_fact(conn, being=self.being, origin=origin, oseq=oseq, t=t, kind=kind, body=body,
                            laws=self.laws, head=head)

    def _rewrite_anchor(self, conn, gen):
        key_id, anchor_key, _seal = self._store._soil_keys(self.soil)
        _rewrite_anchor(conn, being=self.being, owner=self.owner, soil=self.soil, gen=gen, key_id=key_id,
                        anchor_key=anchor_key)

    def _bump(self, conn, meta):
        """The generation moves and the anchor is rewritten over what this transaction wrote; the vacuum debt."""
        gen = meta["gen"] + 1
        _put(conn, "gen", gen)
        self._rewrite_anchor(conn, gen)
        return meta.get("vacuum_owed", 0)

    def _tighten(self):
        if not self._tightened:
            try:
                _tighten(self.path.parent)
            except OSError:
                raise StoreRefused("path", "the store directory cannot be tightened") from None
            self._tightened = True

    # -- the membrane's write ------------------------------------------------

    def append(self, kind, body, *, transport, grant_ref=None, payload=None):
        """Journal one fact through the membrane: ``Appended``, ``Dropped("budget")``, or a named refusal."""
        from . import membrane

        store = self._store
        with store._lock:
            self._gate()
            self._live()
            now = store._read_clock()
            entry = membrane.admit(kind, body, transport=transport, grant_ref=grant_ref, payload=payload, now=now,
                                   single_user=store._single_user_now(), owner=self.owner,
                                   table=self._verified.pin["table"])
            wall = membrane.recorder_wall(now, self.birth_wall)
            seal = store._cipher_pair()[0] if payload is not None else None
            self._tighten()
            outcome, after = self._write(lambda conn: self._append_in(conn, kind, entry, body, payload, wall, seal))
            self._after(after)
            return outcome

    def _append_in(self, conn, kind, entry, body, payload, wall, seal):
        from . import membrane

        store = self._store
        store._stage("head")
        head = _head(conn)
        t_max = conn.execute(T_MAX).fetchone()[0]
        meta = _read_meta(conn)
        t = membrane.recorder_t(wall, self.birth_wall, t_max)
        day = membrane.day_of(t)
        low = day * membrane.MINUTES_A_DAY
        count = conn.execute("SELECT COUNT(*) FROM facts WHERE kind = ? AND t >= ? AND t < ?",
                             (kind, low, low + membrane.MINUTES_A_DAY)).fetchone()[0]
        gen = meta["gen"] + 1
        if count >= self._verified.pin["budgets"][kind]:
            conn.execute("INSERT INTO overflow (day, kind, dropped) VALUES (?, ?, 1) "
                         "ON CONFLICT (day, kind) DO UPDATE SET dropped = dropped + 1", (day, kind))
            _put(conn, "gen", gen)
            self._rewrite_anchor(conn, gen)
            return membrane.Dropped("budget"), self._after_info(conn, meta, day, t, gen, False)
        name = "oseq_next:" + self.origin
        oseq = meta[name]
        written = dict(body)
        ref = key = None
        if payload is not None:
            ref = store._draw(16).hex()
            key = store._draw(32)
            written["payload"] = ref
        eid, seq, _link = self._insert_fact(conn, origin=self.origin, oseq=oseq, t=t, kind=kind, body=written,
                                            head=head)
        if payload is not None:
            sealed = bytes(seal(key, bytes.fromhex(ref) + payload.encode("ascii")))
            conn.execute("INSERT INTO keys (name, key) VALUES (?, ?)", ("payload:" + ref, key))
            conn.execute("INSERT INTO payloads (ref, eid, ct) VALUES (?, ?, ?)", (ref, eid, sealed))
        destruction = self._side_effect(conn, kind, written, eid, meta)
        _put(conn, name, oseq + 1)
        _put(conn, "gen", gen)
        if destruction is not None:
            _put(conn, "cross_pending", list(meta.get("cross_pending", [])) + [destruction])
            _put(conn, "vacuum_owed", 1)
        self._rewrite_anchor(conn, gen)
        outcome = membrane.Appended(eid, seq, oseq, t)
        return outcome, self._after_info(conn, meta, day, t, gen, destruction is not None)

    def _side_effect(self, conn, kind, body, eid, meta):
        """What a forget destroys, inside its own transaction; ``None`` for every other kind."""
        from . import chain

        floor = meta.get("rhythm_floor", 0)
        ended = list(meta.get("heard_ended", []))
        if kind == "lang_forget":
            done = chain.forget_target(conn, target=body["target"], forgetter=eid,
                                       kinds=self._verified.pin["table"]["kinds"])
            return {"destroyed": done["destroyed"], "ended": ended, "forgot": done["forgot"], "rhythm_floor": floor}
        if kind == "forget_rhythm":
            done = chain.forget_rhythm(conn, floor=floor)
            _put(conn, "rhythm_floor", done["rhythm_floor"])
            return {"destroyed": done["destroyed"], "ended": ended, "forgot": [], "rhythm_floor": done["rhythm_floor"]}
        return None

    def _after_info(self, conn, before, day, t, gen, destroyed):
        head_seq, head_link = _head(conn)
        after = _read_meta(conn)
        return {"cross": before.get("cross"), "day": day, "destroyed": destroyed,
                "ended": after.get("heard_ended", []), "floor": after.get("rhythm_floor", 0), "gen": gen,
                "head": head_link, "pending": after.get("cross_pending", []), "seq": head_seq, "t": t,
                "vacuum_owed": after.get("vacuum_owed", 0)}

    # -- the local layers, outside the journal -------------------------------

    def _local_key(self, conn, name):
        """A destructible key of the local layers: drawn at its first use, never computed from another key."""
        row = conn.execute("SELECT key FROM keys WHERE name = ?", (name,)).fetchone()
        if row is not None:
            return bytes(row[0])
        key = self._store._draw(32)
        conn.execute("INSERT INTO keys (name, key) VALUES (?, ?)", (name, key))
        return key

    def rhythm_put(self, transport, observed_hour, active):
        """Seal one presence hour in the rhythm layer, never in the journal; its ``rseq``.

        Only the light hook writes it, only for the account that owns the
        being, and only when the person consented at sowing. Its minute is
        sealed inside the row with the hour: no column gives either.
        """
        from . import anchors, membrane

        store = self._store
        with store._lock:
            self._gate()
            self._live()
            now = store._read_clock()
            surface = membrane.surface_of(transport, now)
            if surface not in membrane.MATRIX["presence_hour"]:
                raise membrane.MembraneRefused("surface", f"presence_hour is not written from {surface}")
            if anchors.owner_tag(membrane.actor_of(transport, store._single_user_now(), now)) != self.owner:
                raise membrane.MembraneRefused("owner", "this account does not own this being")
            if self.rhythm_consent is not True:
                raise membrane.MembraneRefused("consent", "the rhythm was not consented to at sowing")
            body = {"active": active, "observed_hour": observed_hour}
            membrane.check_body(self._verified.pin["table"]["kinds"]["presence_hour"]["body"], body)
            wall = membrane.recorder_wall(now, self.birth_wall)
            seal = store._cipher_pair()[0]
            self._tighten()
            rseq, owed = self._write(lambda conn: self._rhythm_in(conn, body, wall, seal))
            self._after_local(owed)
            return rseq

    def _rhythm_in(self, conn, body, wall, seal):
        from . import chain, membrane, wire

        t = membrane.recorder_t(wall, self.birth_wall, conn.execute(T_MAX).fetchone()[0])
        meta = _read_meta(conn)
        rseq = chain.next_rseq(conn, meta.get("rhythm_floor", 0))
        key = self._local_key(conn, "rhythm")
        sealed = bytes(seal(key, wire.emit({"body": body, "rseq": rseq, "t": t})))
        conn.execute("INSERT INTO rhythm_facts (rseq, ct) VALUES (?, ?)", (rseq, sealed))
        return rseq, self._bump(conn, meta)

    def heard_note(self, season, word, data):
        """Seal what was heard of one word in one season; the season's key is drawn at its first word.

        The word itself is kept only as an HMAC under the season's key, so
        ending the season destroys the only way to recognise it.
        """
        from . import wire
        from .membrane import MembraneRefused

        store = self._store
        with store._lock:
            self._gate()
            self._live()
            if not _is_int(season) or not 0 <= season <= MAX_INT:
                raise MembraneRefused("body", "body season int")
            if not _heard_word(word):
                raise MembraneRefused("body", "body word")
            try:
                size = len(wire.emit(data)) if isinstance(data, dict) else None
            except Exception:  # noqa: BLE001 - data the codec refuses is not data
                size = None
            if size is None or size > HEARD_DATA_MAX:
                raise MembraneRefused("body", "body data")
            seal = store._cipher_pair()[0]
            self._tighten()
            owed = self._write(lambda conn: self._heard_in(conn, season, word, data, seal))
            self._after_local(owed)

    def _heard_in(self, conn, season, word, data, seal):
        from . import wire
        from .membrane import MembraneRefused

        meta = _read_meta(conn)
        if season in meta.get("heard_ended", []):
            raise MembraneRefused("body", "body season ended")
        key = self._local_key(conn, f"heard:{season}")
        digest = hmac.new(key, word.encode("ascii"), hashlib.sha256).hexdigest()
        sealed = bytes(seal(key, wire.emit({"data": data, "hash": digest, "season": season})))
        conn.execute("INSERT OR REPLACE INTO heard (season, hash, ct) VALUES (?, ?, ?)", (season, digest, sealed))
        return self._bump(conn, meta)

    def heard_end_season(self, season):
        """End a heard season: its key and every row of it destroyed in one transaction, then VACUUM and an anchor."""
        from .membrane import MembraneRefused

        store = self._store
        with store._lock:
            self._gate()
            self._live()
            if not _is_int(season) or not 0 <= season <= MAX_INT:
                raise MembraneRefused("body", "body season int")
            self._tighten()
            after = self._write(lambda conn: self._end_in(conn, season))
            if after is not None:
                self._after(after)

    def _end_in(self, conn, season):
        from . import chain

        meta = _read_meta(conn)
        ended = list(meta.get("heard_ended", []))
        done = chain.end_season(conn, season)
        if season in ended and not done["destroyed"] and not done["rows"]:
            return None
        if season not in ended:
            ended = sorted(ended + [season])
        _put(conn, "heard_ended", ended)
        record = {"destroyed": done["destroyed"], "ended": ended, "forgot": [],
                  "rhythm_floor": meta.get("rhythm_floor", 0)}
        _put(conn, "cross_pending", list(meta.get("cross_pending", [])) + [record])
        _put(conn, "vacuum_owed", 1)
        gen = meta["gen"] + 1
        _put(conn, "gen", gen)
        self._rewrite_anchor(conn, gen)
        # No day of life is read for this write: the daily anchor stays with the journal's own writes.
        return self._after_info(conn, meta, -1, self._verified.head_t, gen, True)

    # -- checkpoints ---------------------------------------------------------

    def checkpoint_put(self, t, laws, through_eid, state):
        """Keep a reducer state at one place in the chain: ``INSERT OR IGNORE``, then read back; never overwritten.

        The state's identity is the digest of its canonical bytes; the row
        is keyed by the being, the minute, the engine's identity, the law
        version and the event it runs through. The minute is never before the
        minute of that event, so a forget from a minute on reaches every state
        that could hold what it forgets (``local`` otherwise). The row read back
        must hold this state -- its digest, and a blob that inflates to exactly
        its canonical bytes -- or it is refused ``divergence``, and the kept one
        stays. The same state kept again changes nothing.
        """
        import zlib

        from . import wire

        store = self._store
        with store._lock:
            self._gate()
            self._live()
            if (not _is_int(t) or not 0 <= t <= MAX_INT or not _is_int(laws) or not 0 <= laws <= LAWS_MAX
                    or not _is_hex(through_eid, 64)):
                raise StoreRefused("local", "a checkpoint's minute, law version or event is malformed")
            try:
                canonical = wire.emit(state)
            except Exception:  # noqa: BLE001 - a state the codec refuses has no identity
                raise StoreRefused("local", "a checkpoint's state is not canonical") from None
            state_hash = hashlib.sha256(canonical).hexdigest()
            blob = zlib.compress(canonical, 9)
            engine = _engine_id()
            self._tighten()
            made, owed = self._write(lambda conn: self._checkpoint_in(conn, t, laws, through_eid, engine,
                                                                      state_hash, canonical, blob))
            self._after_local(owed)
            return made

    def _checkpoint_in(self, conn, t, laws, through, engine, state_hash, canonical, blob):
        event = conn.execute("SELECT f.t FROM links l JOIN facts f ON f.eid = l.eid WHERE l.eid = ?",
                             (through,)).fetchone()
        if event is None:
            raise StoreRefused("local", "a checkpoint names an event that is not linked")
        if t < event[0]:
            raise StoreRefused("local", "a checkpoint's minute is before the event it runs through")
        cursor = conn.execute("INSERT OR IGNORE INTO checkpoints (being, t, engine, laws, through, state_hash, blob) "
                              "VALUES (?, ?, ?, ?, ?, ?, ?)", (self.being, t, engine, laws, through, state_hash, blob))
        inserted = cursor.rowcount == 1
        row = conn.execute("SELECT state_hash, blob FROM checkpoints WHERE being = ? AND t = ? AND engine = ? "
                           "AND laws = ? AND through = ?", (self.being, t, engine, laws, through)).fetchone()
        if row is None or row[0] != state_hash or not _inflates_to(row[1], canonical):
            raise StoreRefused("divergence", f"checkpoint at t={t} differs")
        owed = self._bump(conn, _read_meta(conn)) if inserted else None
        return Checkpoint(self.being, t, engine, laws, through, state_hash, bytes(row[1])), owed

    # -- after the commit ----------------------------------------------------

    def _after(self, info):
        """VACUUM after a destruction (or when owed), then a cross-anchor when one is due."""
        verified = self._verified
        if info["seq"] == verified.head + 1:
            verified.head, verified.head_link, verified.head_t = info["seq"], info["head"], info["t"]
        if info["destroyed"] or info["vacuum_owed"] == 1:
            self._vacuum()
        cross = info["cross"] if isinstance(info["cross"], dict) else {}
        last_day = cross.get("day", -1) if _is_int(cross.get("day", -1)) else -1
        pending = info["pending"] if isinstance(info["pending"], list) else []
        if pending:
            self._cross_anchor("destroy", info, pending)
        elif info["day"] > last_day:
            self._cross_anchor("day", info, pending)

    def _after_local(self, owed):
        """After a local-layer write: only a VACUUM still owed.

        No cross-anchor is written for these writes: an audit entry carries
        the time it was written, and a rhythm or a heard row keeps its time
        sealed.
        """
        if owed == 1:
            self._vacuum()

    def _vacuum(self):
        self._store._vacuum_on(self._live(), self.soil, self._write)

    def _cross_anchor(self, why, info, pending):
        self._store._anchor_across(self.soil, self.being_tag, why, info, pending, self._write)

    # -- reading -------------------------------------------------------------

    def head(self):
        """``(seq, link)`` of the head."""
        with self._store._lock:
            self._gate()
            conn = self._live()
            return _guarded(conn, lambda: _head(conn))

    def trunk(self):
        """``(envelope, body or None)`` in seq order, from the facts, bodies and links alone.

        The rows are read when this is called; what it returns yields them.
        """
        with self._store._lock:
            self._gate()
            conn = self._live()
            rows = _guarded(conn, lambda: conn.execute(
                "SELECT f.being, f.body_sha256, f.kind, f.laws, f.origin, f.oseq, f.t, b.body FROM links l "
                "JOIN facts f ON f.eid = l.eid LEFT JOIN bodies b ON b.eid = l.eid ORDER BY l.seq").fetchall())
        return _envelopes(rows)

    def verify(self):
        """Verify the whole chain again from genesis; nothing is written, and a refusal closes this being."""
        store = self._store
        with store._lock:
            self._gate()
            conn = self._live()
            try:
                self._verified = store._verify(conn, self.path, self.tag, self.soil, None)
            except BaseException:
                store._forget(self)
                raise

    def unseal(self, ref):
        """The word a payload reference seals; a forgotten or unknown one is refused ``payload``."""
        from .membrane import MembraneRefused

        store = self._store
        with store._lock:
            self._gate()
            if not _is_hex(ref, 32):
                raise MembraneRefused("payload", "forgotten or unknown")
            conn = self._live()
            row = _guarded(conn, lambda: conn.execute(
                "SELECT p.ct, k.key FROM payloads p LEFT JOIN keys k ON k.name = ? WHERE p.ref = ?",
                ("payload:" + ref, ref)).fetchone())
        if row is None or row[1] is None:
            raise MembraneRefused("payload", "forgotten or unknown")
        plain = bytes(store._cipher_pair()[1](bytes(row[1]), bytes(row[0])))
        if plain[:16] != bytes.fromhex(ref):
            raise MembraneRefused("payload", "the sealed payload belongs to another row")
        return plain[16:].decode("ascii")

    def close(self):
        with self._store._lock:
            self._store._forget(self)
