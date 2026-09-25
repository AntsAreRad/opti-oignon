#!/usr/bin/env python3
"""Shared support for the componion's store contracts: seams, stand-ins and file-level tools.

Every seam the store takes is injected from here, so no contract reaches the
platform's configuration, keys, mode, audit log or data: the store is built
on ``tmp_path`` with

* ``sqlite_seam()`` -- the standard library's connect, with
  ``secure_delete`` forced OFF at connect so that what the store sets is what
  gets measured; ``answering_seam()`` wraps it in ``CipherAnswering``, which
  answers ``PRAGMA cipher_version`` as SQLCipher does and is keyed by
  nothing; ``keyed_connect()`` is real SQLCipher with a raw key;
* ``CountingEntropy`` -- draws from the chassis stream on a fixed key and
  counts every call, so a refusal that draws is seen;
* ``Clock``, ``Mode`` and ``MemoryAudit`` -- stand-ins with the real
  signatures; ``load_audit`` loads the real signed audit log on ``tmp_path``
  instead, in a window of its own that blocks the key and the configuration;
* ``TestCipher`` -- AES-256-GCM in the byte layout of the platform's cipher.

The file tools count bytes over a store and its journals, edit a store
through a connection of the test's own, rewrite its anchor, and splice a
table's root page back from an older copy.

Use the loaded audit only inside a window that blocks the key module: it
asks for the key when it writes its own anchor.
"""

import hashlib
import json
import os
import sqlite3
import sys
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_window import open_allium  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BLOCKED = tuple("opti_oignon." + name for name in (
    "config", "db_utils", "db_encryption", "encryption", "signed_audit_log", "security_mode",
    "user_isolation", "auth",
))
WALL = 1760000000
DAY_S = 1440 * 60
KEY = hashlib.sha256(b"a fabricated anchor key").digest()
KEY_ID = "5eed00a1b2c3d4e5"
WRONG_KEY = hashlib.sha256(b"another anchor key").digest()
MAGIC = b"SQLite format 3\x00"
ACTS = ("greet", "play", "touch", "warm", "water")
JOURNALS = ("-journal", "-wal", "-shm")


class Platform:
    """The modules of one window: the reference, the seam and the platform."""

    def __init__(self, loaded):
        self.loaded = loaded
        prefix = "opti_oignon.allium."
        self.wire = loaded[prefix + "wire"]
        self.rng = loaded[prefix + "rng"]
        self.lawfiles = loaded[prefix + "lawfiles"]
        self.engine = loaded[prefix + "engine"]
        self.journal = loaded[prefix + "ref.journal"]
        self.settings = loaded[prefix + "settings"]
        self.mode = loaded[prefix + "mode"]
        self.chain = loaded[prefix + "chain"]
        self.membrane = loaded[prefix + "membrane"]
        self.anchors = loaded[prefix + "anchors"]
        self.store = loaded[prefix + "store"]


def open_platform(*, seeded=None, blocked=BLOCKED, extra=None):
    """One window with the platform loaded and every platform dependency proven unreachable.

    ``extra`` loads real platform modules a contract names (and leaves out of ``blocked``).
    """
    loaded, restore = open_allium(native=False, platform=True, blocked=blocked, seeded=seeded, extra=extra)
    return Platform(loaded), restore


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------
class _Answer:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class CipherAnswering:
    """A standard-library connection that answers ``PRAGMA cipher_version`` and is keyed by nothing."""

    def __init__(self, conn):
        object.__setattr__(self, "_conn", conn)

    def execute(self, sql, *args):
        if " ".join(sql.split()).lower() == "pragma cipher_version":
            return _Answer([("4.12.0 community",)])
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        setattr(self._conn, name, value)


class Connector:
    """A connect seam that counts its calls and keeps every connection it made."""

    def __init__(self, opener, wrap=None):
        self.calls = 0
        self.connections = []
        self._opener = opener
        self._wrap = wrap

    def __call__(self, path, check_same_thread=True, timeout=5.0):
        self.calls += 1
        conn = self._opener(str(path), check_same_thread, timeout)
        if self._wrap is not None:
            conn = self._wrap(conn)
        self.connections.append(conn)
        return conn


def _stdlib(path, check_same_thread, timeout):
    conn = sqlite3.connect(path, check_same_thread=check_same_thread, timeout=timeout)
    conn.execute("PRAGMA secure_delete = OFF")
    return conn


def sqlite_seam():
    """The standard library's connect with ``secure_delete`` OFF: a plaintext soil."""
    return Connector(_stdlib)


def answering_seam():
    """A plaintext soil whose connection nevertheless answers a cipher version."""
    return Connector(_stdlib, wrap=CipherAnswering)


def raising_seam(message):
    """A connect that raises, as the platform's keyed connect does on a wrong key."""

    def opener(path, check_same_thread, timeout):
        raise RuntimeError(message)

    return Connector(opener)


def keyed_connect(key_hex, keyed=True):
    """Real SQLCipher, keyed with a raw 32-byte key (``keyed=False``: no key at all)."""
    import sqlcipher3

    def opener(path, check_same_thread, timeout):
        conn = sqlcipher3.connect(path, check_same_thread=check_same_thread, timeout=timeout)
        if keyed:
            conn.execute("PRAGMA key = \"x'" + key_hex + "'\"")
            conn.execute("SELECT count(*) FROM sqlite_master").fetchall()
        return conn

    return Connector(opener)


def encrypted_probe(path, conn=None):
    """The probe of the journal contracts' fixture: every file answers as encrypted."""
    return "encrypted"


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------
class CountingEntropy:
    """Bytes from the chassis stream on a fixed key; every call and every chunk is recorded."""

    def __init__(self, rng, suite, index=0):
        self._stream = rng.Stream(bytes(32), "test." + suite, index)
        self._lock = threading.Lock()
        self.calls = 0
        self.drawn = 0
        self.chunks = []

    def __call__(self, n):
        with self._lock:
            out = bytearray()
            while len(out) < n:
                out += self._stream.next_u64().to_bytes(8, "big")
            chunk = bytes(out[:n])
            self.calls += 1
            self.drawn += n
            self.chunks.append(chunk)
        return chunk


class Clock:
    """A wall clock in whole seconds that moves only when told."""

    def __init__(self, wall=WALL):
        self.wall = wall
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.wall

    def advance_days(self, days):
        self.wall += days * DAY_S


class Mode:
    """A mode reader answering a fixed value (or raising it), counting its reads."""

    def __init__(self, value="daily"):
        self.value = value
        self.reads = 0

    def __call__(self):
        self.reads += 1
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


class MemoryAudit:
    """An audit log in memory with the real one's signatures: newest first, details decoded."""

    def __init__(self):
        self.entries = []
        self.calls = 0
        self._lock = threading.Lock()

    def append_event(self, event_type, source="", action="", severity="INFO", details=None):
        with self._lock:
            self.calls += 1
            entry_id = len(self.entries) + 1
            self.entries.append({
                "action": action,
                "details": json.loads(json.dumps(details or {})),
                "event_type": event_type,
                "id": entry_id,
                "severity": severity,
                "source": source,
            })
            return entry_id

    def get_events(self, limit=50, offset=0, event_type=None, severity=None, after=None, before=None):
        with self._lock:
            self.calls += 1
            rows = [json.loads(json.dumps(entry)) for entry in reversed(self.entries)
                    if event_type is None or entry["event_type"] == event_type]
        return rows[offset:offset + limit]

    def verify_chain(self):
        with self._lock:
            self.calls += 1
            return (True, None, len(self.entries))


class TestCipher:
    """AES-256-GCM in the byte layout of the platform's cipher: version, nonce, ciphertext, tag."""

    __test__ = False
    VERSION = 0x02

    def seal(self, key, plaintext):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = os.urandom(12)
        return bytes([self.VERSION]) + nonce + AESGCM(bytes(key)).encrypt(nonce, bytes(plaintext), None)

    def open(self, key, data):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        data = bytes(data)
        if len(data) < 29 or data[0] != self.VERSION:
            raise ValueError("not a sealed value")
        return AESGCM(bytes(key)).decrypt(data[1:13], data[13:], None)

    def pair(self):
        return (self.seal, self.open)


def _stdlib_safe_connect(db_path, *, check_same_thread=True, timeout=5.0):
    return sqlite3.connect(str(db_path), check_same_thread=check_same_thread, timeout=timeout)


def load_audit(tmp_path):
    """The real signed audit log on ``tmp_path``, loaded in a window of its own and closed at once.

    Its connection layer is a standard-library stand-in, the key module and
    the configuration are proven unreachable while it loads, and the
    singleton it builds at import (on the repository's own data path) is
    never used.
    """
    stub = types.ModuleType("opti_oignon.db_utils")
    stub.safe_connect = _stdlib_safe_connect
    loaded, restore = isolate(
        targets={"opti_oignon.signed_audit_log": source("signed_audit_log.py")},
        seeded={"opti_oignon.db_utils": stub},
        blocked=("opti_oignon.encryption", "opti_oignon.config"),
    )
    try:
        module = loaded["opti_oignon.signed_audit_log"]
    finally:
        restore()
    return module.SignedAuditLog(db_path=Path(tmp_path).joinpath("audit", "audit_chain.db"))


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------
def seams(platform, tmp_path, *, suite, index=0, **overrides):
    """The journal contracts' fixture: an encrypted-name plain file, a fabricated key, Daily."""
    out = {
        "single_user": lambda: True,
        "data_dir": Path(tmp_path).joinpath("data"),
        "persistence": {"busy_timeout_ms": 5000, "path": "allium", "require_encryption": True},
        "connect": sqlite_seam(),
        "plain_connect": sqlite_seam(),
        "probe": encrypted_probe,
        "cipher_available": lambda: True,
        "anchor_secret": lambda: ("readable", KEY, KEY_ID),
        "audit": MemoryAudit(),
        "entropy": CountingEntropy(platform.rng, suite, index),
        "cipher": TestCipher().pair(),
        "clock": Clock(),
        "mode": Mode("daily"),
    }
    out.update(overrides)
    return out


def store(platform, given):
    """A new store on the given seams: a fresh process, as far as the store can tell."""
    return platform.store.Store(**given)


def cli(platform, attended=False):
    return platform.membrane.Transport("cli", attended=attended)


def principal(sub, now, *, role="user", lifetime=3600):
    return {"exp": now + lifetime, "iat": now - 60, "role": role, "sub": sub, "type": "access"}


def web(platform, sub, now, **kwargs):
    return platform.membrane.Transport("web", principal=principal(sub, now, **kwargs))


def sow(platform, target, transport=None, *, law="fixture", rhythm_consent=False):
    return target.sow(transport=transport or cli(platform), law=law, tz_minutes=0,
                      rhythm_consent=rhythm_consent)


def build(being, n, transport):
    """``n`` acts appended through the membrane."""
    return [being.append("act", {"act": ACTS[i % len(ACTS)]}, transport=transport) for i in range(n)]


def directory(given):
    return Path(given["data_dir"]).resolve().joinpath("allium")


def store_path(platform, given, user="local", suffix=".db"):
    return directory(given).joinpath(platform.anchors.owner_tag(user) + suffix)


def land(being, origin, oseq, t, kind, body):
    """Land a fact from another device the way sync will: its rows, then the anchor over them."""

    def transaction(conn):
        head = being._head(conn)
        written = being._insert_fact(conn, origin=origin, oseq=oseq, t=t, kind=kind, body=body, head=head)
        meta = being._meta(conn)
        name = "oseq_next:" + origin
        being._put(conn, name, max(meta.get(name, 0), oseq + 1))
        gen = meta["gen"] + 1
        being._put(conn, "gen", gen)
        being._rewrite_anchor(conn, gen)
        return written

    return being._write(transaction)


def canary(platform, suite, length=20):
    """A word no fixture holds: lowercase letters from the chassis stream, generated at run time."""
    stream = platform.rng.Stream(bytes(32), "test." + suite + ".canary", 0)
    return "".join(chr(0x61 + stream.below(26)) for _ in range(length))


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def edit(path, change):
    """Run ``change(conn)`` on a connection of the test's own, commit and close."""
    conn = sqlite3.connect(str(path))
    try:
        result = change(conn)
        conn.commit()
    finally:
        conn.close()
    return result


def read(path, sql, params=()):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def counts(path):
    """Every table's row count."""
    conn = sqlite3.connect(str(path))
    try:
        names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        return {name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in names}
    finally:
        conn.close()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count_bytes(path, needle):
    """Occurrences of ``needle`` in the file, its write-ahead log and its shared memory."""
    total = 0
    for suffix in ("", "-wal", "-shm"):
        part = Path(str(path) + suffix)
        if part.exists():
            total += part.read_bytes().count(needle)
    return total


def remove_with_journals(path):
    for suffix in ("",) + JOURNALS:
        part = Path(str(path) + suffix)
        if part.exists():
            part.unlink()


def root_page(path, table, connect=None):
    """``(root page number, page size)`` of a table, read through ``connect``."""
    conn = (connect or sqlite3.connect)(str(path))
    try:
        page = conn.execute("SELECT rootpage FROM sqlite_master WHERE name = ?", (table,)).fetchone()[0]
        size = conn.execute("PRAGMA page_size").fetchone()[0]
    finally:
        conn.close()
    # SQLCipher answers the page size as text.
    return int(page), int(size)


def splice(path, old_bytes, table, connect=None):
    """Write a table's root page back from an older copy of the file, as raw bytes."""
    path = Path(path)
    old = path.with_name(path.name + ".old-copy")
    old.write_bytes(old_bytes)
    try:
        old_page = root_page(old, table, connect)
    finally:
        remove_with_journals(old)
    page, size = root_page(path, table, connect)
    assert (page, size) == old_page, f"the root page of {table} moved: {old_page} -> {(page, size)}"
    start = (page - 1) * size
    data = bytearray(path.read_bytes())
    data[start:start + size] = old_bytes[start:start + size]
    path.write_bytes(bytes(data))
    return page


def rewrite_anchor(platform, path, *, seq, key_id, anchor_key):
    """Rewrite ``meta.anchor`` over the file's own rows, local tables and meta, naming event ``seq``."""
    chain = platform.chain
    wire = platform.wire

    def change(conn):
        meta = {key: wire.parse(bytes(value)) for key, value in conn.execute("SELECT key, value FROM meta")}
        head_link = conn.execute("SELECT link FROM links WHERE seq = ?", (seq,)).fetchone()[0]
        local = chain.local_digest(chain.read_local(conn), meta)
        value = chain.anchor_value(being=meta["being"], gen=meta["gen"], head_seq=seq, head_link=head_link,
                                   local=local, owner=meta["owner"], soil=meta["soil"], key_id=key_id,
                                   anchor_key=anchor_key)
        conn.execute("UPDATE meta SET value = ? WHERE key = 'anchor'", (wire.emit(value),))

    edit(path, change)
