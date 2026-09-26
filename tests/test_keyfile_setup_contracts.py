#!/usr/bin/env python3
"""Contracts for encryption setup and the master key file it writes.

The master key file keys every encrypted database and the signed audit
chain. ``POST /api/security/encryption/setup`` wrote it with ``write_text``
and never looked at what was already there. When the server could not load
an existing key file -- an enveloped one without OPTI_KEYFILE_PASSPHRASE in
the server's environment is enough -- the manager disabled itself, setup
ran, and the key every database was encrypted under was replaced (in random
mode, by a key in the unprotected format). Setup now never replaces a key
file, and a new one is written whole, with mode 0600 from its creation.

  * KF1 -- a key file that exists and does not open in this server is
    refused with 409 naming OPTI_KEYFILE_PASSPHRASE, in both modes, before
    any key is made: the key file, its directory and security.yaml are left
    as they were, and the manager stays off.
  * KF2 -- a key that loads (an enveloped key file with its passphrase, an
    unprotected key file, or OPTI_ENCRYPTION_KEY) is enabled as it is, in
    both modes: the manager encrypts under that key, security.yaml is
    switched on, and the key directory is unchanged byte for byte.
  * KF3 -- with no key file, setup creates one through a temp opened
    O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW with mode 0600 and synced, then
    hard-linked into place, the directory synced after the link; the key
    file is 0600, holds the key the manager encrypts with, and no temp is
    left.
  * KF4 -- a key file that appears after the route has looked (a rival
    writer) is refused with the same 409 and is never replaced, whether it
    appears while the key is being made (and then no temp holding key bytes
    is created at all) or between the synced temp and its move into place.
  * KF5 -- a symbolic link at the key path, dangling or pointing at a key
    file that does not open here, is refused with the same 409 before any
    key is made; the link and what it points at are left as they were.
  * KF6 -- once the key file is linked into place, a temp that cannot be
    removed or a directory that cannot be synced does not fail setup: it
    answers 200 with the key file whole, the manager on and security.yaml
    switched on, and a warning names what was left.
  * KF7 -- a link that fails for another reason (EPERM, as on a filesystem
    without hard links) is a failed setup that wrote nothing: 500, the key
    directory empty, the manager off, security.yaml as it was. No fallback
    writes the key file some other way.

Local-only (the public distribution ships no tests). The real
``encryption.py`` and ``api/routes_security.py`` are loaded through the
shared isolation window with the real web framework; the handler is called
as a function and the exception FastAPI would turn into a response is read
directly. The key file lives under the test's temporary directory
(``_DEFAULT_KEYFILE`` redirected), and so does security.yaml (the route's
path and the manager's config reader both redirected); OPTI_ENCRYPTION_KEY
and OPTI_KEYFILE_PASSPHRASE are cleared unless a clause sets them, and the
KDF runs with the module's own parameters (46 ms per derivation measured
here). The key wrapper is the module's own fallback: ``secure_bytes``
installs signal handlers and overwrites the caller's bytes when it is
imported, so the window proves it unreachable. The encryption module's
``os`` is a pass-through that records what it opens, syncs, closes, links
and unlinks in the key directory, plants a rival where a clause asks, and
makes a sync, a link or an unlink fail where a clause asks. security.yaml
starts with a comment line the route's own write never produces, so a
refused setup that wrote it is seen even when the setting it wrote is the
one already there. Key files the clauses start from are written here, in the on-disk format,
and a key file setup writes is read back the same way (the format, Argon2id
and AES-GCM), never through the module's own loader.
"""

import base64
import errno
import hashlib
import json
import logging
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from argon2.low_level import Type, hash_secret_raw
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_PASSPHRASE = "a passphrase typed into setup"
_WRAPPING = "the passphrase the key file is wrapped under"
_REFUSAL_WORDS = ("OPTI_KEYFILE_PASSPHRASE", "Setup never replaces a key file.")
_CREATE_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)


class _RecordingOS:
    """The encryption module's ``os``: every call passes through to the real one.

    What is opened, synced, closed, linked or unlinked in the key directory
    (or the directory itself) is recorded in order, by path. When
    ``after_temp`` is set, it runs once, right after a file opened inside the
    key directory is closed: between the temp and its move into place.
    ``refuse`` maps ``"fsync"``, ``"link"`` or ``"unlink"`` to ``(predicate,
    error)``: the call on a path the predicate accepts (for a link, its
    destination) raises ``error`` instead of reaching the real one, and is
    recorded as refused.
    """

    def __init__(self, real, keydir):
        self._real = real
        self._keydir = os.fspath(keydir)
        self._paths = {}
        self.events = []
        self.after_temp = None
        self.refuse = {}

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _watched(self, path):
        path = os.fspath(path)
        return path == self._keydir or os.path.dirname(path) == self._keydir

    def _refused(self, name, path):
        rule = self.refuse.get(name)
        if path is None or rule is None or not rule[0](os.fspath(path)):
            return
        self.events.append(("refused", name, os.fspath(path)))
        raise rule[1]

    def open(self, path, flags, mode=0o777, *args, **kwargs):
        fd = self._real.open(path, flags, mode, *args, **kwargs)
        if self._watched(path):
            self._paths[fd] = os.fspath(path)
            self.events.append(("open", os.fspath(path), flags, mode))
        return fd

    def fsync(self, fd):
        self._refused("fsync", self._paths.get(fd))
        self._real.fsync(fd)
        if fd in self._paths:
            self.events.append(("fsync", self._paths[fd]))

    def close(self, fd):
        path = self._paths.pop(fd, None)
        self._real.close(fd)
        if path is None:
            return
        self.events.append(("close", path))
        if path != self._keydir and self.after_temp is not None:
            plant, self.after_temp = self.after_temp, None
            plant()

    def link(self, src, dst, *args, **kwargs):
        self._refused("link", dst)
        self._real.link(src, dst, *args, **kwargs)
        self.events.append(("link", os.fspath(src), os.fspath(dst)))

    def unlink(self, path, *args, **kwargs):
        if self._watched(path):
            self._refused("unlink", path)
        self._real.unlink(path, *args, **kwargs)
        if self._watched(path):
            self.events.append(("unlink", os.fspath(path)))


def _b64(data):
    return base64.urlsafe_b64encode(data).decode("ascii")


def _kek(enc, passphrase, salt, kdf):
    if kdf == "pbkdf2":
        return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, enc._PBKDF2_ITERATIONS, dklen=32)
    return hash_secret_raw(
        secret=passphrase.encode("utf-8"), salt=salt, time_cost=enc._ARGON2_TIME_COST,
        memory_cost=enc._ARGON2_MEMORY_COST, parallelism=enc._ARGON2_PARALLELISM,
        hash_len=32, type=Type.ID,
    )


def _envelope(enc, key, passphrase):
    """An enveloped key file's bytes, as the on-disk format has them."""
    salt = os.urandom(16)
    nonce = os.urandom(12)
    blob = bytes([2]) + nonce + AESGCM(_kek(enc, passphrase, salt, "argon2id")).encrypt(nonce, key, None)
    payload = {"version": "envelope-v1", "kdf": "argon2id", "kek_salt": _b64(salt), "blob": _b64(blob)}
    return (json.dumps(payload) + "\n").encode("ascii")


def _unprotected(key):
    """A key file in the unprotected format: the key, an empty salt, the kdf name."""
    return ("\n".join([_b64(key), "", "random"]) + "\n").encode("ascii")


def _key_in(enc, path, passphrase=None):
    """The key a key file holds, read from its bytes by the format alone."""
    text = path.read_bytes().decode("ascii").strip()
    try:
        envelope = json.loads(text)
    except ValueError:
        envelope = None
    if isinstance(envelope, dict):
        kek = _kek(enc, passphrase, base64.urlsafe_b64decode(envelope["kek_salt"]), envelope["kdf"])
        blob = base64.urlsafe_b64decode(envelope["blob"])
        assert blob[0] == 2, "an envelope's blob starts with the format byte"
        return AESGCM(kek).decrypt(blob[1:13], blob[13:], None)
    return base64.urlsafe_b64decode(text.split("\n")[0])


def _format_of(path):
    if not path.exists():
        return "absent"
    try:
        json.loads(path.read_bytes())
    except ValueError:
        return "in the unprotected format"
    return "an envelope"


def _opened_under(key, token):
    """Decrypt a field the manager encrypted, with ``key`` and AES-GCM alone."""
    assert token.startswith("ENC2:"), f"the manager did not encrypt: {token!r}"
    raw = base64.urlsafe_b64decode(token[len("ENC2:"):])
    assert raw[0] == 2, "an encrypted field starts with the format byte"
    return AESGCM(key).decrypt(raw[1:13], raw[13:], None).decode("utf-8")


def _put(path, data):
    """Write a key file the way a previous setup left it: whole, mode 0600."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _tree(directory):
    """Every entry of ``directory`` with the md5 of its bytes."""
    if not directory.exists():
        return {}
    return {entry.name: hashlib.md5(entry.read_bytes()).hexdigest() for entry in sorted(directory.iterdir())}


def _config_reader(path):
    def read():
        if not path.exists():
            return {}
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("encryption", {})
    return read


def _open(tmp_path, monkeypatch, *, config_enabled):
    """The two real modules in a window, every path and variable redirected."""
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE"):
        monkeypatch.delenv(name, raising=False)
    loaded, restore = isolate(
        targets={
            "opti_oignon.encryption": source("encryption.py"),
            "opti_oignon.api.routes_security": source("api", "routes_security.py"),
        },
        blocked=("opti_oignon.secure_bytes",),
        packages=("opti_oignon.api",),
    )
    try:
        enc = loaded["opti_oignon.encryption"]
        routes = loaded["opti_oignon.api.routes_security"]
        keydir = tmp_path / "data"
        keydir.mkdir()
        config = tmp_path / "security.yaml"
        # The comment line is one the route's own write never produces, so a
        # refused setup that wrote security.yaml anyway is seen even when it
        # wrote the same setting back.
        config.write_text(
            "# security settings as the test left them\n"
            + yaml.safe_dump({"encryption": {"enabled": config_enabled}}),
            encoding="utf-8",
        )
        enc._DEFAULT_KEYFILE = keydir / ".keyfile"
        enc._load_encryption_config = _config_reader(config)
        routes._SECURITY_YAML_PATH = str(config)
        recorder = _RecordingOS(os, keydir)
        enc.os = recorder
    except BaseException:
        restore()
        raise
    return SimpleNamespace(
        enc=enc, routes=routes, keydir=keydir, keyfile=keydir / ".keyfile",
        config=config, os=recorder, restore=restore,
    )


def _setup(server, mode):
    """Call the setup handler; ``(200, answer)`` or ``(status, detail)`` of its refusal."""
    body = {"mode": mode}
    if mode == "passphrase":
        body["passphrase"] = _PASSPHRASE
    request = server.routes.EncryptionSetupRequest(**body)
    try:
        return 200, server.routes.setup_encryption(request)
    except server.routes.HTTPException as exc:
        return exc.status_code, exc.detail


def _count_key_making(server):
    """Record every call that makes a key (a random one, or one derived from a passphrase)."""
    made = []
    for name in ("generate_key", "derive_key_from_passphrase"):
        real = getattr(server.enc, name)

        def recording(*args, _real=real, _name=name, **kwargs):
            made.append(_name)
            return _real(*args, **kwargs)

        setattr(server.enc, name, recording)
    return made


def _config_enabled(server):
    return (yaml.safe_load(server.config.read_text(encoding="utf-8")) or {}).get("encryption", {}).get("enabled")


# ---------------------------------------------------------------------------
# KF1 -- a key file that does not open here is refused, and kept
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["passphrase", "random"])
def test_kf1_a_key_file_that_does_not_open_here_is_refused_and_kept(tmp_path, monkeypatch, mode):
    server = _open(tmp_path, monkeypatch, config_enabled=True)
    try:
        key = os.urandom(32)
        _put(server.keyfile, _envelope(server.enc, key, _WRAPPING))
        before = _tree(server.keydir)
        config_before = server.config.read_bytes()
        made = _count_key_making(server)

        status, answer = _setup(server, mode)

        assert _tree(server.keydir) == before, (
            f"setup in {mode} mode replaced a key file this server cannot open "
            f"(the key file is now {_format_of(server.keyfile)}): every database "
            f"keyed by the old one would stop opening"
        )
        assert status == 409, f"a key file that does not open here is refused with 409, not {status}: {answer}"
        for words in _REFUSAL_WORDS:
            assert words in answer, f"the refusal says {words!r}: {answer!r}"
        assert made == [], f"the refusal comes before any key is made: {made}"
        assert server.config.read_bytes() == config_before, "a refused setup leaves security.yaml as it was"
        assert server.enc.get_encryption_manager().enabled is False, "a refused setup leaves the manager off"
        assert _key_in(server.enc, server.keyfile, _WRAPPING) == key
    finally:
        server.restore()


# ---------------------------------------------------------------------------
# KF2 -- a key that loads is enabled as it is, and nothing is written
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["passphrase", "random"])
@pytest.mark.parametrize("origin", ["enveloped", "unprotected", "environment"])
def test_kf2_a_key_that_loads_is_enabled_as_it_is(tmp_path, monkeypatch, origin, mode):
    server = _open(tmp_path, monkeypatch, config_enabled=False)
    try:
        key = os.urandom(32)
        if origin == "enveloped":
            _put(server.keyfile, _envelope(server.enc, key, _WRAPPING))
            monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", _WRAPPING)
        elif origin == "unprotected":
            _put(server.keyfile, _unprotected(key))
        else:
            monkeypatch.setenv("OPTI_ENCRYPTION_KEY", _b64(key))
        before = _tree(server.keydir)

        status, answer = _setup(server, mode)

        assert _tree(server.keydir) == before, (
            f"setup in {mode} mode wrote in the key directory although a key loads "
            f"({origin}): {sorted(before)} became {sorted(_tree(server.keydir))}, "
            f"the key file now {_format_of(server.keyfile)}"
        )
        assert status == 200, f"a key that loads is enabled, not refused: {status} {answer}"
        assert answer["setup"] is True
        assert answer["detail"] == "Encryption enabled with the existing key"
        assert answer["status"]["enabled"] is True
        manager = server.enc.get_encryption_manager()
        assert _opened_under(key, manager.encrypt("a field")) == "a field", (
            "the manager encrypts under the key that loaded"
        )
        assert _config_enabled(server) is True, "security.yaml is switched on"
    finally:
        server.restore()


# ---------------------------------------------------------------------------
# KF3 -- a new key file is created 0600, synced, and linked into place whole
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["passphrase", "random"])
def test_kf3_a_new_key_file_is_created_0600_and_linked_whole(tmp_path, monkeypatch, mode):
    server = _open(tmp_path, monkeypatch, config_enabled=False)
    try:
        status, answer = _setup(server, mode)

        events = server.os.events
        creates = [event for event in events if event[0] == "open" and event[1] != str(server.keydir)]
        assert creates, (
            f"no file was opened in the key directory through os.open: the key file "
            f"({_format_of(server.keyfile)}) was written without O_CREAT|O_EXCL and a mode"
        )
        assert len(creates) == 1, f"one temp is created: {creates}"
        _, temp, flags, mode_bits = creates[0]
        assert flags & _CREATE_FLAGS == _CREATE_FLAGS, (
            f"the temp is opened O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW ({_CREATE_FLAGS:#o}), not {flags:#o}"
        )
        assert mode_bits == 0o600, f"the temp is created with mode 0600, not {mode_bits:#o}"
        assert temp != str(server.keyfile), "the key file itself is never the file being written"
        order = [event[:3] if event[0] == "link" else event[:2] for event in events]
        for step in (("fsync", temp), ("link", temp, str(server.keyfile)), ("fsync", str(server.keydir))):
            assert step in order, f"{step[0]} of {step[1:]} was not done: {order}"
        written = order.index(("fsync", temp))
        linked = order.index(("link", temp, str(server.keyfile)))
        synced = order.index(("fsync", str(server.keydir)))
        assert written < linked < synced, (
            f"the temp is synced, then linked to the key file, then the directory is synced: {order}"
        )
        assert sorted(os.listdir(server.keydir)) == [".keyfile"], (
            f"no temp is left: {sorted(os.listdir(server.keydir))}"
        )
        info = os.stat(server.keyfile)
        assert stat.S_IMODE(info.st_mode) == 0o600, f"the key file is 0600, not {stat.S_IMODE(info.st_mode):#o}"
        assert info.st_nlink == 1, "the temp's name is gone: the key file has one link"
        assert status == 200 and answer["setup"] is True, f"{status} {answer}"
        assert answer["detail"] == "Encryption configured successfully"
        key = _key_in(server.enc, server.keyfile, _PASSPHRASE if mode == "passphrase" else None)
        manager = server.enc.get_encryption_manager()
        assert _opened_under(key, manager.encrypt("a field")) == "a field", (
            "the key file holds the key the manager encrypts with"
        )
        assert _config_enabled(server) is True
    finally:
        server.restore()


# ---------------------------------------------------------------------------
# KF4 -- a key file that appears after the route has looked is never replaced
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("moment", ["making-random", "making-passphrase", "before-link"])
def test_kf4_a_rival_key_file_is_refused_and_never_replaced(tmp_path, monkeypatch, moment):
    server = _open(tmp_path, monkeypatch, config_enabled=False)
    try:
        rival = b"a key file another writer put in place\n"
        planted = []

        def plant():
            if not planted:
                _put(server.keyfile, rival)
                planted.append(moment)

        if moment == "before-link":
            server.os.after_temp = plant
            mode = "random"
        else:
            mode = moment.split("-")[1]
            name = "generate_key" if mode == "random" else "derive_key_from_passphrase"
            real = getattr(server.enc, name)

            def making(*args, **kwargs):
                plant()
                return real(*args, **kwargs)

            setattr(server.enc, name, making)
        config_before = server.config.read_bytes()

        status, answer = _setup(server, mode)

        assert planted, (
            f"the rival was never planted: nothing was opened, written and closed in the "
            f"key directory before the key file appeared (it is {_format_of(server.keyfile)})"
        )
        assert server.keyfile.read_bytes() == rival, (
            f"the rival key file ({moment}) was replaced; it is now {_format_of(server.keyfile)}"
        )
        assert status == 409, f"a key file that appeared meanwhile is refused with 409, not {status}: {answer}"
        for words in _REFUSAL_WORDS:
            assert words in answer, f"the refusal says {words!r}: {answer!r}"
        assert sorted(os.listdir(server.keydir)) == [".keyfile"], (
            f"no temp is left: {sorted(os.listdir(server.keydir))}"
        )
        if moment != "before-link":
            creates = [event for event in server.os.events if event[0] == "open"]
            assert creates == [], (
                f"with the key file already there, no temp holding key bytes is created: {creates}"
            )
        assert server.enc.get_encryption_manager().enabled is False, "a refused setup leaves the manager off"
        assert server.config.read_bytes() == config_before, "a refused setup leaves security.yaml as it was"
    finally:
        server.restore()


# ---------------------------------------------------------------------------
# KF5 -- a link at the key path is refused, and neither it nor its target moves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("points_at", ["nothing", "an envelope"])
def test_kf5_a_link_at_the_key_path_is_refused_and_left_as_it_is(tmp_path, monkeypatch, points_at):
    server = _open(tmp_path, monkeypatch, config_enabled=False)
    try:
        target = tmp_path / "elsewhere" / ".keyfile"
        target.parent.mkdir()
        if points_at == "an envelope":
            _put(target, _envelope(server.enc, os.urandom(32), _WRAPPING))
        target_before = target.read_bytes() if target.exists() else None
        os.symlink(target, server.keyfile)
        made = _count_key_making(server)

        status, answer = _setup(server, "random")

        assert status == 409, (
            f"a link at the key path pointing at {points_at} is refused with 409, not {status}: {answer}"
        )
        for words in _REFUSAL_WORDS:
            assert words in answer, f"the refusal says {words!r}: {answer!r}"
        assert made == [], f"a link at the key path is refused before any key is made: {made}"
        assert os.path.islink(server.keyfile) and os.readlink(server.keyfile) == str(target), (
            "the link at the key path is left as it was"
        )
        target_after = target.read_bytes() if target.exists() else None
        assert target_after == target_before, (
            f"what the link points at is left as it was: it was "
            f"{'absent' if target_before is None else 'an envelope'} and is now "
            f"{_format_of(target)}"
        )
        assert sorted(os.listdir(server.keydir)) == [".keyfile"], (
            f"nothing is left beside the link: {sorted(os.listdir(server.keydir))}"
        )
    finally:
        server.restore()


# ---------------------------------------------------------------------------
# KF6 -- once the key file is in place, a failed cleanup does not fail setup
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("step", ["directory-sync", "temp-unlink"])
def test_kf6_a_key_file_in_place_stands_when_cleanup_after_the_link_fails(tmp_path, monkeypatch, caplog, step):
    server = _open(tmp_path, monkeypatch, config_enabled=False)
    try:
        keydir = str(server.keydir)
        if step == "directory-sync":
            server.os.refuse["fsync"] = (lambda path: path == keydir, OSError(errno.EINVAL, "Invalid argument"))
        else:
            server.os.refuse["unlink"] = (
                lambda path: os.path.basename(path) != ".keyfile",
                OSError(errno.EACCES, "Permission denied"),
            )

        with caplog.at_level(logging.WARNING):
            status, answer = _setup(server, "random")

        refused = [event for event in server.os.events if event[0] == "refused"]
        assert refused, f"the {step} failure was never planted: {server.os.events}"
        assert status == 200 and answer["setup"] is True, (
            f"the key file was linked into place whole, yet setup answered {status}: {answer}"
        )
        assert answer["detail"] == "Encryption configured successfully"
        assert os.path.isfile(server.keyfile), (
            f"the key file setup answered for is in place: {sorted(os.listdir(server.keydir))}"
        )
        key = _key_in(server.enc, server.keyfile)
        manager = server.enc.get_encryption_manager()
        assert _opened_under(key, manager.encrypt("a field")) == "a field", (
            "the key file holds the key the manager encrypts with"
        )
        assert _config_enabled(server) is True, "security.yaml is switched on"
        named = refused[0][2]
        warnings = [
            record.getMessage() for record in caplog.records
            if record.levelno >= logging.WARNING and record.name == server.enc.logger.name
        ]
        assert any(named in message for message in warnings), (
            f"a warning names what the failed {step} left ({named}): {warnings}"
        )
        # The key directory's path is a prefix of every message naming the key
        # file (the unprotected-format warning is one), so the warning is also
        # read for what it says went wrong.
        why = "could not be synced" if step == "directory-sync" else "could not be removed"
        assert any(named in message and why in message for message in warnings), (
            f"a warning says that {named} {why}: {warnings}"
        )
    finally:
        server.restore()


# ---------------------------------------------------------------------------
# KF7 -- a link that fails for another reason writes nothing, and says so
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["passphrase", "random"])
def test_kf7_a_key_file_that_cannot_be_linked_is_a_failed_setup_that_wrote_nothing(tmp_path, monkeypatch, mode):
    server = _open(tmp_path, monkeypatch, config_enabled=False)
    try:
        keyfile = str(server.keyfile)
        # EPERM is what link(2) answers on a filesystem without hard links.
        server.os.refuse["link"] = (lambda path: path == keyfile, OSError(errno.EPERM, "Operation not permitted"))
        config_before = server.config.read_bytes()

        status, answer = _setup(server, mode)

        assert [event for event in server.os.events if event[0] == "refused"], (
            f"the link failure was never planted: {server.os.events}"
        )
        assert status == 500, (
            f"a key file that could not be linked into place is a failed setup, not {status}: {answer} "
            f"(the key file is {_format_of(server.keyfile)})"
        )
        assert sorted(os.listdir(server.keydir)) == [], (
            f"nothing is left in the key directory, neither the temp nor a key file written "
            f"another way: {sorted(os.listdir(server.keydir))}"
        )
        assert server.enc.get_encryption_manager().enabled is False, "a failed setup leaves the manager off"
        assert server.config.read_bytes() == config_before, "a failed setup leaves security.yaml as it was"
    finally:
        server.restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly"]))
