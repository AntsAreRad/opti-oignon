#!/usr/bin/env python3
"""The test process sees the maintainer's data as a fresh checkout would.

Real personal data lives in three kinds of place inside the tree: ``data/``
(the master key, the security mode, the signed audit chain),
``opti_oignon/data/`` (conversations, plugins, the governor's decisions,
the sync change feed) and database files wherever they lie. A contract must
never read or write any of it. Measured before this module existed, 253 of
them reached for it through real modules -- the master key by its
existence check, the audit chain by an append -- and a sandbox that hid
only ``data/`` answered differently from the maintainer's machine.

While a ``DataFirewall`` is installed, every path inside those places is
redirected to a mirror under a temporary directory. The mirror starts with
the tracked files of the data places as HEAD holds them, taken from git and
never from the places themselves, and nothing else: a read finds only what
this session wrote, a write lands in the mirror, and the real place is
never touched. The redirection covers the file functions the standard
library routes through -- ``open``, the ``os`` calls that take a path, and
the SQLite and SQLCipher connects, by path or by a ``file:`` URI whose
query is kept -- in this process only: a child process
a contract starts is not covered. A path relative to a directory
descriptor is left alone, since it names something the descriptor already
reached.

Each redirected path is counted against the running test, so the figure can
be driven down suite by suite as suites become hermetic.

Import-safe and stdlib-only.
"""

import builtins
import collections
import io
import os
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile

# The data places, relative to the root of the tree, as git spells them.
PLACES = ("data", "opti_oignon/data")
DATABASE_SUFFIXES = (".db", ".db-journal", ".db-wal", ".db-shm", ".sqlite", ".sqlite3")

# The os functions whose first argument is a path, and the two that take two.
_ONE_PATH = (
    "open", "stat", "lstat", "access", "listdir", "scandir", "mkdir", "rmdir",
    "remove", "unlink", "chmod", "utime",
)
_TWO_PATHS = ("rename", "replace")
_NOT_A_FILE = (":memory:", "")


class DataFirewall:
    """Redirects the data places of one tree to a mirror while it is installed."""

    def __init__(self, root, *, mirror=None, seed=True):
        self.root = os.path.normpath(os.path.abspath(os.fspath(root)))
        self._places = tuple(os.path.join(self.root, *place.split("/")) for place in PLACES)
        self.mirror = os.fspath(mirror) if mirror is not None else None
        self._own_mirror = mirror is None
        self._seed = seed
        self._saved = {}
        self.current = None
        self.redirected = collections.defaultdict(set)

    # -- which paths ------------------------------------------------------------

    def _relative(self, full):
        """The path of ``full`` inside the tree when it must be redirected, else None."""
        if not full.startswith(self.root + os.sep):
            return None
        relative = full[len(self.root) + 1:]
        for place in self._places:
            if full == place or full.startswith(place + os.sep):
                return relative
        if relative.endswith(DATABASE_SUFFIXES):
            return relative
        return None

    def target(self, path):
        """``path`` itself, or where it lies in the mirror when it is in a data place."""
        if path is None or isinstance(path, int):
            return path
        try:
            raw = os.fspath(path)
        except TypeError:
            return path
        text = os.fsdecode(raw)
        full = text if os.path.isabs(text) else os.path.join(os.getcwd(), text)
        if not full.startswith(self.root):
            return path
        relative = self._relative(os.path.normpath(full))
        if relative is None:
            return path
        self.redirected[self.current or "(while collecting)"].add(relative)
        mirrored = os.path.join(self.mirror, relative)
        return os.fsencode(mirrored) if isinstance(raw, bytes) else mirrored

    # -- the mirror -------------------------------------------------------------

    def _seed_from_head(self):
        """Copy the tracked files of the data places as HEAD holds them, if git can say."""
        git = ["git", "-C", self.root]
        try:
            listed = subprocess.run(
                [*git, "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", *PLACES],
                capture_output=True, check=True, timeout=30,
            ).stdout
            names = [name for name in listed.decode("utf-8", "surrogateescape").split("\0") if name]
            if not names:
                return
            archive = subprocess.run(
                [*git, "archive", "--format=tar", "HEAD", "--", *names],
                capture_output=True, check=True, timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return  # not a git tree: the mirror stays empty
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(self.mirror, filter="data")

    # -- install ----------------------------------------------------------------

    def _one_path(self, original):
        target = self.target

        def redirected(*args, **kwargs):
            if kwargs.get("dir_fd") is None:
                if args:
                    args = (target(args[0]), *args[1:])
                elif "path" in kwargs:
                    kwargs["path"] = target(kwargs["path"])
            return original(*args, **kwargs)

        return redirected

    def _two_paths(self, original):
        target = self.target

        def redirected(src, dst, *args, **kwargs):
            if kwargs.get("src_dir_fd") is None:
                src = target(src)
            if kwargs.get("dst_dir_fd") is None:
                dst = target(dst)
            return original(src, dst, *args, **kwargs)

        return redirected

    def _uri_target(self, database):
        """A ``file:`` URI whose file lies in a data place, rewritten to the mirror with its query kept; else as given.

        The query (``mode=ro``, say) and the fragment travel unchanged, so a
        read-only open of the mirror stays read-only. An in-memory URI, one
        that is not ``file:``, and one naming a file elsewhere pass untouched.
        """
        from urllib.parse import quote, unquote, urlsplit

        text = os.fsdecode(os.fspath(database))
        if not text.startswith("file:"):
            return database
        parts = urlsplit(text)
        path = unquote(parts.path)
        if path in _NOT_A_FILE or path.startswith(":memory:"):
            return database
        moved = self.target(path)
        if moved is path:
            return database
        os.makedirs(os.path.dirname(moved), exist_ok=True)
        rebuilt = "file:" + quote(moved) + ("?" + parts.query if parts.query else "")
        rebuilt += "#" + parts.fragment if parts.fragment else ""
        return os.fsencode(rebuilt) if isinstance(database, bytes) else rebuilt

    def _connect(self, original):
        target = self.target
        uri_target = self._uri_target

        def redirected(database, *args, **kwargs):
            named = isinstance(database, (str, bytes, os.PathLike))
            if named and kwargs.get("uri"):
                database = uri_target(database)
            elif named and os.fsdecode(os.fspath(database)) not in _NOT_A_FILE:
                moved = target(database)
                if moved is not database:
                    os.makedirs(os.path.dirname(os.fsdecode(moved)), exist_ok=True)
                database = moved
            return original(database, *args, **kwargs)

        return redirected

    def install(self):
        """Redirect, for this process, until ``uninstall``. Installing twice is a no-op."""
        if self._saved:
            return
        if self.mirror is None:
            self.mirror = tempfile.mkdtemp(prefix="oo-data-mirror-")
        for place in PLACES:
            os.makedirs(os.path.join(self.mirror, *place.split("/")), exist_ok=True)
        if self._seed:
            self._seed_from_head()
        target = self.target
        original_open = builtins.open

        def redirected_open(file, *args, **kwargs):
            return original_open(target(file), *args, **kwargs)

        self._saved["open"] = original_open
        builtins.open = io.open = redirected_open
        for name in _ONE_PATH:
            original = getattr(os, name)
            self._saved["os." + name] = original
            setattr(os, name, self._one_path(original))
        for name in _TWO_PATHS:
            original = getattr(os, name)
            self._saved["os." + name] = original
            setattr(os, name, self._two_paths(original))
        self._saved["sqlite3.connect"] = sqlite3.connect
        sqlite3.connect = self._connect(sqlite3.connect)
        try:
            import sqlcipher3.dbapi2 as cipher
        except Exception:  # noqa: BLE001 - no SQLCipher, nothing more to cover
            cipher = None
        if cipher is not None:
            self._saved["cipher"] = (cipher, cipher.connect)
            cipher.connect = self._connect(cipher.connect)

    def uninstall(self):
        """Put every function back; remove the mirror when this firewall made it."""
        if not self._saved:
            return
        builtins.open = io.open = self._saved.pop("open")
        if "cipher" in self._saved:
            module, connect = self._saved.pop("cipher")
            module.connect = connect
        sqlite3.connect = self._saved.pop("sqlite3.connect")
        for key, original in list(self._saved.items()):
            setattr(os, key[len("os."):], original)
        self._saved.clear()
        if self._own_mirror and self.mirror:
            shutil.rmtree(self.mirror, ignore_errors=True)

    def summary(self):
        """One line for the end of the session: how much was kept off the real places."""
        paths = sum(len(found) for found in self.redirected.values())
        return (
            f"data firewall: {paths} path(s) from {len(self.redirected)} test(s) or imports "
            "kept off the maintainer's data and served from the mirror"
        )
