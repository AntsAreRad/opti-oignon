"""Where the Mycelium's stores lie, and how they are opened and wiped.

One store per corpus, ``<data folder>/mycelium/<corpus>.db``, the package's data folder read at each call: a store
folder of the encryption command (``db_encryption._STORE_FOLDERS``), which takes a store written in plaintext there as
it takes every ``*.db`` of the platform. A corpus is named by lowercase letters, digits, '_' and '-', from a letter or a
digit, at most 100 of them: a name never leaves the folder, and any other is refused by name. A store's own name that is
a symbolic link is refused (the folder, like every store folder, may be one). Reading a corpus makes no folder and no
file and writes nothing (``open_to_read``: ``Store(..., read_only=True)``) -- but beside a store another program left in
WAL, where SQLite makes its -wal and -shm on reading until the store is written again; writing makes the folder, the
owner's alone, and the store (``open_to_write``), unless only an existing store is to be written. One vector pass runs
at a time on the install's stores (``one_vector_pass``: the folder itself held, no file made). The account wipe
overwrites and removes every store, the files SQLite keeps beside one while it writes and what an interrupted encryption
left beside it (``remove_every_store``; one store and its files: ``remove_store``; a file another name also holds, as an interrupted encryption's ``.restoring``
holds its kept copy's bytes, is removed without its overwrite, said in the log); the copies an encryption run keeps in
``plaintext-*`` folders are not removed, as for every store of the platform. The stores are the install's, not a user's:
the wipe asks it for a single-user install's one user."""
import errno
import logging
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path

from .store import Store, StoreError

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

SUFFIX = ".db"
# What SQLite keeps beside a store while it writes, and what an interrupted encryption leaves: removed with it.
COMPANIONS = ("-journal", "-wal", "-shm", ".encrypting", ".restoring")
CORPUS = re.compile(r"[a-z0-9][a-z0-9_-]{0,99}\Z")


class NoStore(StoreError):
    """A corpus with no store: nothing is indexed under that name."""


class AnotherVectorPass(StoreError):
    """Another vector pass holds the install's stores: this one is left to a later pass."""


def folder():
    """The stores' folder: the package's data folder as it is at the call, joined to the Mycelium's."""
    from ..config import DATA_DIR

    return Path(DATA_DIR) / "mycelium"


def path_for(corpus):
    """The store of ``corpus``, a name the rule admits (``CORPUS``); any other refused by name."""
    if not isinstance(corpus, str) or not CORPUS.match(corpus):
        raise StoreError(f"a corpus is named by lowercase letters, digits, '_' and '-', from a letter or a digit, at "
                         f"most 100 of them, not {corpus!r}")
    return folder() / (corpus + SUFFIX)


def corpora():
    """The corpora whose stores the folder holds, sorted; an absent folder holds none, and is not made."""
    try:
        entries = list(os.scandir(folder()))
    except FileNotFoundError:
        return []
    return sorted(entry.name[:-len(SUFFIX)] for entry in entries
                  if entry.name.endswith(SUFFIX) and CORPUS.match(entry.name[:-len(SUFFIX)])
                  and entry.is_file(follow_symlinks=False))


def _not_a_link(path):
    if os.path.islink(path):
        raise StoreError(f"{path} is a symbolic link: a store is never reached through one")


def open_to_read(corpus):
    """The store of ``corpus``, opened read-only: nothing made, nothing written; a corpus with no store refused by
    its name (``NoStore``)."""
    path = path_for(corpus)
    _not_a_link(path)
    if not path.is_file():
        raise NoStore(f"no store for the corpus {corpus!r}: nothing is indexed under that name")
    return Store.at(path, read_only=True)


def open_to_write(corpus, make=True):
    """The store of ``corpus``, opened to be written, made with its folder (the owner's alone) when new; with
    ``make`` False, only an existing store: a corpus with no store refused by its name (``NoStore``), nothing made."""
    path = path_for(corpus)
    _not_a_link(path)
    if not make and not path.is_file():
        raise NoStore(f"no store for the corpus {corpus!r}: nothing is indexed under that name")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return Store.at(path)


@contextmanager
def one_vector_pass():
    """One vector pass at a time on the install's stores, whatever the process: an exclusive flock on the stores'
    folder itself -- no file is made for it, and never a descriptor on a store, whose closing would let go of the
    locks SQLite holds on it in this process --, taken without waiting, let go when the pass ends or its process
    does. Another holder, in this process or another: ``AnotherVectorPass``; a folder that cannot be locked (a file
    system or a platform without flock): refused by name."""
    where = folder()
    try:
        import fcntl
    except ImportError:
        raise StoreError(f"the vector passes' lock cannot be held on {where}: this platform has no flock") from None
    try:
        fd = os.open(where, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise StoreError(f"the vector passes' lock cannot be held on {where} ({type(exc).__name__})") from None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AnotherVectorPass(f"another vector pass is under way on the stores of {where}: the vectors are left "
                                    f"to the next pass") from None
        except OSError as exc:
            raise StoreError(f"the vector passes' lock cannot be held on {where} "
                             f"({errno.errorcode.get(exc.errno, type(exc).__name__)})") from None
        yield
    finally:
        os.close(fd)


def remove_every_store():
    """The account wipe: every store of the folder and the files beside it (``COMPANIONS``), overwritten, then
    removed; a file that is not a store's is kept. Raises naming what stayed; an absent folder holds none, and is
    not made."""
    try:
        entries = list(os.scandir(folder()))
    except FileNotFoundError:
        return
    stayed = []
    for entry in sorted(entries, key=lambda e: e.name):
        name = next((entry.name[:-len(c)] for c in COMPANIONS if entry.name.endswith(c)), entry.name)
        if not name.endswith(SUFFIX):
            continue
        try:
            _overwrite_and_remove(Path(entry.path))
        except OSError as exc:
            logger.warning("A store of the Mycelium could not be removed: %s", type(exc).__name__)
            stayed.append(entry.name)
    if stayed:
        raise OSError(f"stores of the Mycelium that stayed: {', '.join(stayed)}")


def remove_store(corpus):
    """The store of ``corpus`` and the files beside it (``COMPANIONS``) overwritten, then removed; a store that is
    not there is not made. Raises naming what stayed."""
    path = path_for(corpus)
    stayed = []
    for name in [path.name] + [path.name + companion for companion in COMPANIONS]:
        try:
            _overwrite_and_remove(path.parent / name)
        except OSError as exc:
            logger.warning("A store of the Mycelium could not be removed: %s", type(exc).__name__)
            stayed.append(name)
    if stayed:
        raise OSError(f"stores of the Mycelium that stayed: {', '.join(stayed)}")


def _overwrite_and_remove(path: Path, said: list | None = None) -> bool:
    """Overwrite a file's bytes in place, then remove it; whether it was there. A filesystem that writes in place no
    longer holds them; a copy-on-write filesystem, a snapshot or a backup may. A link is refused, never followed.

    A file another link also names -- a backup made of links, say -- is removed without its overwrite: those bytes
    are the other copy's too. One this process may not write is made writable first, being the process's own; one
    that stays unwritable is removed without its overwrite. Each removal without its overwrite is said: in ``said``
    when it is given, in the log otherwise."""
    # Never blocks: a FIFO with no reader refuses the open rather than wait for one.
    nofollow = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    reason = None
    try:
        fd = os.open(path, os.O_WRONLY | nofollow)
    except FileNotFoundError:
        return False
    except PermissionError:
        try:
            fd, reason = _made_writable(path, nofollow)
        except FileNotFoundError:
            return False
    if fd is not None:
        try:
            held = os.fstat(fd)
            if not stat.S_ISREG(held.st_mode):
                reason = "not a regular file"
            elif held.st_nlink > 1:
                reason = "other links hold its bytes"
            else:
                left = held.st_size
                block = b"\0" * 65536
                while left > 0:
                    left -= os.write(fd, block[:min(len(block), left)])
                os.fsync(fd)
        finally:
            os.close(fd)
    os.unlink(path)
    if reason is not None:
        if said is None:
            logger.warning("a file removed without its overwrite (%s)", reason)
        else:
            said.append(reason)
    return True


def _made_writable(path: Path, nofollow: int) -> tuple:
    """(a descriptor that writes ``path``, None) once the file is made readable and writable by its owner -- it is
    this process's own --, or (None, why) when it cannot be; only a regular file no other link names is changed."""
    try:
        held = os.lstat(path)
    except FileNotFoundError:
        raise
    except OSError as exc:
        return None, type(exc).__name__
    if not stat.S_ISREG(held.st_mode):
        return None, "not a regular file"
    if held.st_nlink > 1:
        return None, "other links hold its bytes"
    try:
        os.chmod(path, stat.S_IMODE(held.st_mode) | stat.S_IRUSR | stat.S_IWUSR)
        return os.open(path, os.O_WRONLY | nofollow), None
    except OSError as exc:
        return None, type(exc).__name__
