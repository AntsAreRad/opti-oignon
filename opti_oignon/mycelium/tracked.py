"""What a repository tracks, read from git's own index file: no process is started, nothing reaches a remote.

The index (``.git/index``, or the one a worktree's ``.git`` file points to) is read as git writes it: versions 2,
3 and 4, the trailing checksum checked (all zeros when git was told to skip it). Only regular files are listed: a
submodule is a directory, and a symbolic link may point outside the repository, so neither is. An index split in
two (``core.splitIndex``), of an unknown version or of a repository whose objects are not named by SHA-1, a
checksum that does not hold, a file cut short: refused by name (``IndexUnreadable``), never guessed from the
working tree."""
import hashlib
import struct
from pathlib import Path

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

_REGULAR = (0o100644, 0o100755)
_EXTENDED = 0x4000
_NAME_MASK = 0x0FFF


class IndexUnreadable(RuntimeError):
    """The repository's index cannot be read as git wrote it."""


def git_dir(root):
    """The git directory of ``root``: ``root/.git``, or the directory its ``.git`` file names; None when none."""
    dot = Path(root) / ".git"
    if dot.is_dir():
        return dot
    if dot.is_file():
        line = dot.read_text(encoding="utf-8", errors="replace").strip()
        if line.startswith("gitdir:"):
            target = Path(line[len("gitdir:"):].strip())
            return target if target.is_absolute() else (Path(root) / target)
    return None


def _varint(data, at):
    """git's offset varint (index version 4): the value and where it ends."""
    byte = data[at]
    value = byte & 0x7F
    at += 1
    while byte & 0x80:
        byte = data[at]
        value = ((value + 1) << 7) | (byte & 0x7F)
        at += 1
    return value, at


def tracked_files(root):
    """The paths, relative and in index order, of the regular files git tracks under ``root``."""
    directory = git_dir(root)
    if directory is None:
        raise IndexUnreadable(f"{root} is not a git repository")
    if not directory.is_dir():
        raise IndexUnreadable(f"{directory} does not exist: the repository moved or is gone (an empty listing would "
                              f"remove every file of the source)")
    path = directory / "index"
    common = directory
    try:
        common = (directory / (directory / "commondir").read_text(encoding="utf-8").strip()).resolve()
    except OSError:
        pass
    try:
        config = (common / "config").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        config = ""
    if "objectformat" in config.replace(" ", "") and "sha256" in config:
        raise IndexUnreadable(f"{directory} names its objects by SHA-256: its index is not read here")
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise IndexUnreadable(f"{path} cannot be read: {exc}") from exc
    if len(data) < 32 or data[:4] != b"DIRC":
        raise IndexUnreadable(f"{path} is not a git index")
    version, count = struct.unpack(">II", data[4:12])
    if version not in (2, 3, 4):
        raise IndexUnreadable(f"{path} is a git index of version {version}, read here: 2, 3 and 4")
    body, checksum = data[:-20], data[-20:]
    if checksum != bytes(20) and hashlib.sha1(body).digest() != checksum:
        raise IndexUnreadable(f"{path} does not hold its checksum (written while it was read?)")
    at, previous, out, seen = 12, b"", [], set()
    try:
        for _ in range(count):
            start = at
            mode = struct.unpack(">I", body[at + 24:at + 28])[0]
            flags = struct.unpack(">H", body[at + 60:at + 62])[0]
            at += 62
            if version >= 3 and flags & _EXTENDED:
                at += 2
            if version == 4:
                strip, at = _varint(body, at)
                end = body.index(b"\0", at)
                if strip > len(previous):
                    raise IndexUnreadable(f"{path}: an entry strips more than the path before it")
                name = previous[:len(previous) - strip] + body[at:end]
                at = end + 1
            else:
                length = flags & _NAME_MASK
                end = at + length if length < _NAME_MASK else body.index(b"\0", at)
                name = body[at:end]
                at = start + ((end - start) // 8 + 1) * 8
            previous = name
            if mode in _REGULAR and name not in seen:
                seen.add(name)
                out.append(name.decode("utf-8", "surrogateescape"))
    except (struct.error, ValueError, IndexError) as exc:
        raise IndexUnreadable(f"{path} is cut short or malformed: {exc}") from exc
    while at + 8 <= len(body):
        signature, size = body[at:at + 4], struct.unpack(">I", body[at + 4:at + 8])[0]
        if signature == b"link":
            raise IndexUnreadable(f"{path} is split in two (core.splitIndex): its shared part is not read here")
        at += 8 + size
    return out
