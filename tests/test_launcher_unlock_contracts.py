#!/usr/bin/env python3
"""Contracts for starting a platform whose key file is wrapped: the launchers ask, check, and hand the platform's
secrets to the backend alone.

Since the stores are encrypted, the backend needs the key file's passphrase to open its key file. Neither launcher
asked for it: from the desktop icon (``scripts/launch.sh``) the backend stopped at once, its error scrolled away by
the frontend's proxy errors, and ``python -m opti_oignon`` (``opti_oignon/ui.py``) threw the backend's output
away. Started with ``OPTI_KEYFILE_PASSPHRASE`` or ``OPTI_ENCRYPTION_KEY`` set, both handed them to the frontend's
dev server -- every package of ``node_modules`` -- and to the browser.

  * UL1 -- ``python3 -m opti_oignon.keyfile_gate wrapped`` exits 0 for a key file wrapped under a passphrase, 3
    for none or one in the unprotected format as the loader reads it (its base64's ``-`` and ``_`` included), 4 for
    one it cannot read or parse -- a symbolic link that leads nowhere, an envelope cut short, a file in neither
    format, base64 that does not decode included --, saying why; any other use is refused (4). 1 and 2, Python's own
    failures, are never answers.
  * UL2 -- ``... opens`` reads the passphrase on its standard input, as bytes decoded the way the environment is,
    and exits 0 when it opens the key file to a 32-byte key -- a passphrase beyond ASCII included --, 3 when the
    envelope, of the right lengths, does not open under it -- an empty one, one UTF-8 cannot carry, one with other
    spaces, a byte of its salt or of its wrapped key changed for another, never taking the environment's for it --,
    4 when the key file is not there, not wrapped, damaged in a way its format shows (a length, a version, a
    derivation unknown or that cannot run here, each said for what it is), or opens to another size, should the
    loader ever give one; nothing it writes holds the passphrase. A refused passphrase is a ``ValueError`` with the
    loader's own message.
  * UL3 -- ``launch.sh``, the key file wrapped and no passphrase given, asks on its terminal without echo before
    it starts anything, takes what is typed as it is typed, says a wrong one does not open the key file and asks
    again; the one that opens it is in the backend's environment alone: under no name in the frontend's, the
    browser opener's, nor any ``python3`` the launcher runs for its own checks -- which see it on the gate's
    standard input only, the gate run with ``-E`` --, and on no command line; nor does any program the bash that
    runs the backend's script would run at its start -- a ``BASH_ENV``, a function from the environment -- see it,
    the launcher itself run as its desktop icon runs it, by its own first line, under ``bash -p``.
  * UL4 -- ``launch.sh`` starts nothing when three passphrases do not open the key file, when no passphrase is
    read, when it has no terminal to ask on, when the gate cannot tell or cannot try -- whatever status it gives
    but 0 and 3 --, when a port already answers, where the browser would go included, when ``/proc`` does not
    number processes as the launcher does, or when its ``setpriv`` cannot set the parent-death signal; each is
    said. Interrupted at its prompt, it ends (130), the terminal's echo back on. Without a terminal, a passphrase
    given that opens the key file is enough.
  * UL5 -- ``launch.sh`` started with the passphrase set does not ask when that one opens the key file; when it
    does not, or is empty, or holds several lines -- which is said and never reaches the gate --, it asks. The
    backend alone receives the passphrase, and ``OPTI_ENCRYPTION_KEY`` when it is given: a ``BASH_ENV`` or a
    function in the launcher's environment runs nowhere where they are, the launcher's own bash included.
  * UL6 -- ``launch.sh`` starts the frontend only once the backend answers, and looks for it at ``localhost``,
    where the browser goes; the backend, the frontend and the browser opener run in sessions of their own, the
    browser opener without holding the launcher; a backend that stops before it answers, or while the frontend
    starts, and a frontend that stops before it answers or later are said with their exit status, and the browser
    is not opened before both answer; whatever runs -- each process group the launcher started, never its own --
    is stopped, and its whole group has ended, before the message, which a pipe after the launcher keeps and an
    interruption meanwhile does not cut short; once it is said, an interruption ends the launcher (130), and so does
    a closing window (129).
  * UL7 -- ``launch.sh`` with a key file that is not wrapped, or none, asks nothing, and a passphrase given goes
    to the backend as it was given -- random-key setup wraps the key it makes under it --, unless it is empty or
    holds several lines.
  * UL8 -- ``ui.py`` takes both secrets out of its environment before anything it runs, its install and port
    steps included; the passphrase -- the one given when it opens a wrapped key file, else one asked without echo;
    the one given, as it is, when the key file is not wrapped -- and the master key are in the backend's
    environment alone, never the frontend's nor the browser's; the gate, asked in its own process about the
    project's own key file, is handed each passphrase it tries, the environment already without the secrets; the
    backend and the frontend run in sessions of their own, reading nothing from the terminal, each, on Linux, told
    by the kernel to stop when the launcher ends -- its TERM back to the default first, and ended at once should
    the launcher have ended before that held --, elsewhere started without it; the backend is given 10 s to end its
    connections once told to stop; an interruption, a request to end and a hangup are each handled from the start,
    but one ignored when the launcher started, which stays ignored.
  * UL9 -- ``ui.py`` starts nothing without a terminal, after three passphrases that do not open the key file,
    when none is read, when the gate cannot tell, cannot try or does not import, or when it is interrupted while
    it asks (130); each is said. A typed byte the terminal's encoding cannot carry opens nothing and is never
    shown.
  * UL10 -- ``ui.py`` leaves the backend's errors on the terminal; a backend that stops before it answers is said
    with its exit status once what its group left is stopped, the frontend never started; a slow one is waited for
    at its own address, said every 30 s; a backend or a frontend that stops later, an interruption, a request to
    end, a hangup and anything else that ends the launcher -- a frontend that cannot start, its output gone --
    stop every process group it started -- each told first, then all awaited together for 15 s, a group whose
    leader ended included, then what is left killed, and said where it still can be, an output in use dropping
    its lines --, no other signal heard meanwhile, and the launcher ends with its status, the reason said last; an
    end by a signal is said as such; a signal that comes while a child is being started reaches it.
  * UL11 -- ``launch.sh`` whose window closes (HUP, to the launcher or to its whole process group, once or twice),
    interrupted (INT, its output's reader included), told to end (TERM) or whose output's reader is gone (PIPE)
    tells what it started to stop, waits for all of it to end -- a backend or a frontend slow to end included; what
    does not end in 15 s is killed, its whole group, a group whose leader ended included, and that is said --, and
    ends with 128 and the signal's number. Killed itself, it leaves what it started told to stop.
  * UL12 -- ``launch.sh`` that activates a conda base whose activation sets the secrets takes them as given, never
    over the ones it was handed: the backend alone receives them; an activation that leaves options set in the
    launcher's shell neither exports nor shows what the launcher holds.
  * UL13 -- ``launch.sh`` started with ``allexport``, ``errexit``, ``monitor`` and ``xtrace`` among its shell
    options -- by ``bash -o``: its own first line takes none from the environment --, on a controlling terminal,
    neither exports nor shows the passphrase it reads, neither ends on
    the gate's "no" nor loses track of the backend; the gate decodes as ``PYTHONUTF8`` says, as the backend does.
  * UL14 -- ``scripts/dev_backend.sh`` finds the project root without running a program, wherever it is started
    from and however it is named, runs no program but the server -- ``python3 -m uvicorn`` on 127.0.0.1:8001,
    given 10 s to end its connections once told to stop --, and ``--help`` prints its header alone.

The gate is loaded through the shared isolation window, its key files under ``tmp_path``. ``launch.sh`` runs as it
is in ``bwrap`` with a network and a process tree of its own -- its ports are free there whatever runs on this
machine, and whatever it leaves ends with it -- on a pseudo-terminal, with stand-ins: ``python3``, ``xdg-open`` and
the two dev scripts note what is in their environment -- through ``/usr/bin/env`` itself, so that a bash of
their own reads no BASH_ENV where the backend's secrets are --, ``python3`` its arguments too; ``env`` notes its
arguments; ``xdg-open`` waits as a real one may; the dev scripts take half a second to end once told to and their
listeners a second, so only a launcher that waits for the whole group sees them all end -- longer, or never, where
a case says so; ``npm`` answers only that it is there. ``dev_backend.sh`` runs as it is, a ``python3`` stand-in
alone on its PATH. ``ui.py`` is loaded through the window, the gate, ``subprocess``, ``getpass``, the network, the
browser, the clock and the process groups stood in for; the hook its children run before they start runs for real.
Where bwrap cannot give the launcher a network of its own, the launcher's contracts fail: they cannot be checked.

Local-only (the public distribution ships no tests).
"""

import base64
import ctypes
import functools
import getpass
import io
import json
import os
import pty
import select
import shlex
import shutil
import signal
import subprocess
import sys
import termios
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ENC, GATE, UI = "opti_oignon.encryption", "opti_oignon.keyfile_gate", "opti_oignon.ui"
VAR, KEY_VAR = "OPTI_KEYFILE_PASSPHRASE", "OPTI_ENCRYPTION_KEY"
K1 = bytes(range(32))
PHRASE = "a launcher test phrase, never a real one"
WRONG = "Wr0ng-Phrase-17"
MASTER = "a-test-master-key-never-a-real-one"
ODD = " a back\\slash, and spaces at both ends "
ACCENTED = "une phrase \u00e0 accents, \u00e9t\u00e9 comme hiver"
SECRETS = (PHRASE, WRONG, MASTER, ODD.strip())
UNSET = "<unset>"
PROMPT = "Passphrase of the key file: "


def _source(name: str) -> Path:
    return REPO / "opti_oignon" / name


# --------------------------------------------------------------------------- the gate


def _gate_world(tmp_path, monkeypatch):
    """encryption and the gate in a window, the key file under ``tmp_path``."""
    pytest.importorskip("cryptography")
    for name in (KEY_VAR, VAR):
        monkeypatch.delenv(name, raising=False)
    loaded, restore = isolate(targets={"opti_oignon.process_secrets": _source("process_secrets.py"),
                                       ENC: _source("encryption.py"), GATE: _source("keyfile_gate.py")},
                              blocked=("opti_oignon.secure_bytes", "opti_oignon.security_mode"))
    try:
        enc = loaded[ENC]
        data = tmp_path / "data"
        data.mkdir()
        enc._DEFAULT_KEYFILE = data / ".keyfile"
        return SimpleNamespace(enc=enc, gate=loaded[GATE], keyfile=enc._DEFAULT_KEYFILE, restore=restore)
    except BaseException:
        restore()
        raise


def _wrap(enc, path: Path, key: bytes, phrase: str, pbkdf2: bool = False) -> None:
    """A key file wrapping ``key`` -- of any size -- under ``phrase``, in the envelope setup writes; with ``pbkdf2``,
    derived as setup derives where argon2-cffi does not import."""
    kek, salt, kdf = enc.derive_key_from_passphrase(phrase, force_pbkdf2=pbkdf2)
    payload = {"version": enc._KEYFILE_ENVELOPE_VERSION, "kdf": kdf,
               "kek_salt": base64.urlsafe_b64encode(salt).decode("ascii"),
               "blob": base64.urlsafe_b64encode(enc.encrypt_bytes(kek, key)).decode("ascii")}
    path.write_text(json.dumps(payload) + "\n", encoding="ascii")
    os.chmod(path, 0o600)


def _replace(path: Path, text: str) -> None:
    path.unlink()
    path.write_text(text, encoding="ascii")
    os.chmod(path, 0o600)


def test_ul1_the_gate_tells_whether_the_key_file_is_wrapped(tmp_path, monkeypatch, capsys):
    w = _gate_world(tmp_path, monkeypatch)
    try:
        gate = w.gate
        # launch.sh reads these answers by number; 1 and 2, Python's own failures, are never answers.
        assert (gate.YES, gate.NO, gate.CANNOT) == (0, 3, 4)
        assert gate.main(["wrapped"]) == gate.NO
        # The unprotected format, as the loader reads it: a key whose base64 holds "-" and "_" included.
        for key in (K1, bytes([0xFB, 0xFF] * 16)):
            w.enc.save_keyfile(key, allow_unprotected=True)
            assert gate.main(["wrapped"]) == gate.NO, key.hex()
            assert w.enc.keyfile_wrapped() is False
            w.keyfile.unlink()
        w.enc.save_keyfile(K1, passphrase=PHRASE)
        assert gate.main(["wrapped"]) == gate.YES
        assert w.enc.keyfile_wrapped() is True
        held = w.keyfile.read_text(encoding="ascii")
        capsys.readouterr()
        if os.geteuid() != 0:
            os.chmod(w.keyfile, 0)
            try:
                assert gate.main(["wrapped"]) == gate.CANNOT
            finally:
                os.chmod(w.keyfile, 0o600)
            assert "cannot be read" in capsys.readouterr().err
        # Nested past what the parser follows, cut short, or neither format: cannot tell, never "not wrapped".
        for name, text in (("nested", "[" * 100_000), ("cut", held[: len(held) // 2]), ("neither", "!\n"),
                           ("undecodable", "abc\n")):
            _replace(w.keyfile, text)
            assert gate.main(["wrapped"]) == gate.CANNOT, name
            with pytest.raises(gate.GateError):
                gate.wrapped()
            assert "cannot be read" in capsys.readouterr().err, name
        # A symbolic link that leads nowhere is there, and cannot be read.
        w.keyfile.unlink()
        w.keyfile.symlink_to(tmp_path / "nowhere")
        assert gate.main(["wrapped"]) == gate.CANNOT
        capsys.readouterr()
        for refused in ([], ["opens", "now"], ["unwrap"], ["wrapped", "wrapped"]):
            assert gate.main(refused) == gate.CANNOT
            assert "usage" in capsys.readouterr().err
    finally:
        w.restore()


def test_ul2_the_gate_opens_only_under_the_passphrase_on_its_input(tmp_path, monkeypatch, capsys):
    w = _gate_world(tmp_path, monkeypatch)
    try:
        gate, enc = w.gate, w.enc
        enc.save_keyfile(K1, passphrase=PHRASE)
        held = w.keyfile.read_text(encoding="ascii")

        def given(data: bytes):
            monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data), encoding="latin-1"))
            return gate.main(["opens"])

        line = PHRASE.encode()
        assert given(line + b"\n") == gate.YES
        assert given(line) == gate.YES
        for name, data in (("wrong", WRONG.encode() + b"\n"), ("leading", b" " + line + b"\n"),
                           ("trailing", line + b" \n"), ("empty", b"\n"), ("nothing", b""),
                           ("undecodable", b"\xe9" + line + b"\n")):
            assert given(data) == gate.NO, name
        # The environment's passphrase is never taken for the one on its input.
        monkeypatch.setenv(VAR, PHRASE)
        assert given(b"\n") == gate.NO
        assert VAR not in os.environ
        monkeypatch.setenv(VAR, PHRASE)
        assert given(WRONG.encode() + b"\n") == gate.NO
        monkeypatch.setenv(VAR, PHRASE)
        assert gate.opens("") is False
        monkeypatch.delenv(VAR, raising=False)
        said = capsys.readouterr()
        assert not any(secret in said.out + said.err for secret in SECRETS)
        assert "0xe9" not in said.out + said.err
        # A refused passphrase is a ValueError carrying the loader's own message.
        assert issubclass(enc.PassphraseRefused, ValueError)
        with pytest.raises(enc.PassphraseRefused, match="^GCM authentication failed"):
            enc.load_keyfile(passphrase=WRONG)
        envelope = json.loads(held)
        blob = base64.urlsafe_b64decode(envelope["blob"])
        salt = base64.urlsafe_b64decode(envelope["kek_salt"])
        # A byte of the wrapped key or of the salt changed for another fails as a wrong passphrase does: nothing
        # can tell them apart.
        flipped = blob[:-1] + bytes([blob[-1] ^ 1])
        _replace(w.keyfile, json.dumps(dict(envelope, blob=base64.urlsafe_b64encode(flipped).decode("ascii"))) + "\n")
        assert given(line + b"\n") == gate.NO
        flipped = bytes([salt[0] ^ 1]) + salt[1:]
        _replace(w.keyfile, json.dumps(dict(envelope, kek_salt=base64.urlsafe_b64encode(flipped).decode("ascii")))
                 + "\n")
        assert given(line + b"\n") == gate.NO
        # It cannot try a key file that is not there, not wrapped, damaged in a way its format shows, of a
        # derivation that cannot run here, or that opens to anything but a 32-byte key.
        w.keyfile.unlink()
        assert given(line + b"\n") == gate.CANNOT
        enc.save_keyfile(K1, allow_unprotected=True)
        assert given(line + b"\n") == gate.CANNOT
        assert "not wrapped" in capsys.readouterr().err

        def b64(data):
            return base64.urlsafe_b64encode(data).decode("ascii")

        broken = {
            "salt not text": dict(envelope, kek_salt=7),
            "salt short": dict(envelope, kek_salt=b64(salt[:-1])),
            "salt long": dict(envelope, kek_salt=b64(salt + b"\x00")),
            "blob gone": {field: value for field, value in envelope.items() if field != "blob"},
            "blob empty": dict(envelope, blob=""),
            "blob short": dict(envelope, blob=b64(blob[:4])),
            "blob cut": dict(envelope, blob=b64(blob[:-16])),
            "blob longer": dict(envelope, blob=b64(blob + b"more")),
            "blob version": dict(envelope, blob=b64(b"\x01" + blob[1:])),
            "kdf unknown": dict(envelope, kdf="scrypt"),
            "kdf none": {field: value for field, value in envelope.items() if field != "kdf"},
        }
        for name, payload in broken.items():
            _replace(w.keyfile, json.dumps(payload) + "\n")
            assert given(line + b"\n") == gate.CANNOT, name
            err = capsys.readouterr().err
            assert "cannot be opened here" in err, name
            # A derivation it does not know is said as such.
            assert ("names a key derivation this version does not know" in err) == name.startswith("kdf"), name
        if envelope["kdf"] == "argon2id":
            _replace(w.keyfile, held)
            monkeypatch.setattr(enc, "_ARGON2_AVAILABLE", False)
            assert given(line + b"\n") == gate.CANNOT
            assert "argon2" in capsys.readouterr().err
            monkeypatch.setattr(enc, "_ARGON2_AVAILABLE", True)
        # Should the loader ever give anything but a 32-byte key, the gate does not take it.
        _replace(w.keyfile, held)
        loader = enc.load_keyfile
        monkeypatch.setattr(enc, "load_keyfile", lambda *_args, **_kwargs: (bytearray(16), None, "argon2id"))
        assert given(line + b"\n") == gate.CANNOT
        assert "opens to 16 bytes" in capsys.readouterr().err
        monkeypatch.setattr(enc, "load_keyfile", loader)
        w.keyfile.unlink()
        _wrap(enc, w.keyfile, bytes(16), PHRASE)
        assert given(line + b"\n") == gate.CANNOT
        said = capsys.readouterr()
        assert "cannot be opened here" in said.err and PHRASE not in said.out + said.err
        # A passphrase beyond ASCII opens the key file wrapped under it, read from the bytes typed.
        w.keyfile.unlink()
        enc.save_keyfile(K1, passphrase=ACCENTED)
        assert given(ACCENTED.encode("utf-8") + b"\n") == gate.YES
        assert given(ACCENTED.encode("latin-1") + b"\n") == gate.NO
        # One wrapped with PBKDF2 -- made where argon2-cffi did not import -- opens too, under its own alone.
        w.keyfile.unlink()
        _wrap(enc, w.keyfile, K1, PHRASE, pbkdf2=True)
        assert json.loads(w.keyfile.read_text(encoding="ascii"))["kdf"] == "pbkdf2"
        assert given(line + b"\n") == gate.YES
        assert given(WRONG.encode() + b"\n") == gate.NO
        # The key file of a project named by its root is the one asked about.
        other = tmp_path / "other"
        (other / "data").mkdir(parents=True)
        assert gate.wrapped(other) is False
        assert gate.wrapped() is True
    finally:
        w.restore()


# --------------------------------------------------------------------------- launch.sh


@functools.lru_cache(maxsize=1)
def _bwrap() -> str | None:
    """bwrap, when it can give a process a network and a process tree of its own here."""
    found = shutil.which("bwrap")
    if found is None:
        return None
    probe = subprocess.run([found, "--unshare-net", "--unshare-pid", "--die-with-parent", "--ro-bind", "/", "/",
                            "--dev-bind", "/dev", "/dev", "--proc", "/proc", "true"],
                           capture_output=True, timeout=30, check=False)
    return found if probe.returncode == 0 else None


@pytest.fixture
def bwrap():
    """bwrap, or the contract fails: one that cannot be checked here is not one that holds."""
    found = _bwrap()
    if found is None:
        pytest.fail("bwrap cannot give the launcher a network and a process tree of its own here: "
                    "these contracts cannot be checked")
    return found


_PYTHON3 = r"""#!/bin/bash
N='@NOTES@'
{ printf 'argv:'; printf ' [%s]' "$@"; printf '\n'; printf 'env:%s\n' "${OPTI_KEYFILE_PASSPHRASE-<unset>}"; } >> "$N/python3.log"
/usr/bin/env >> "$N/python3.environ"
while :; do
    case "${1-}" in
        -E) shift ;;
        -X) shift 2 ;;
        *) break ;;
    esac
done
if [ "${1-}" = -c ]; then
    if [ -e "$N/import_fails" ] && [ ! -e "$N/activated" ]; then exit 1; fi
    exit 0
fi
if [ "${1-}" = -m ] && [ "${2-}" = opti_oignon.keyfile_gate ]; then
    case "${3-}" in
        wrapped)
            s=$(cat "$N/wrapped")
            [ "$s" = 0 ] || [ "$s" = 3 ] || echo "keyfile_gate: the stand-in cannot tell ($s)" >&2
            exit "$s" ;;
        opens)
            IFS= read -r line
            printf 'stdin:%s\n' "$line" >> "$N/python3.log"
            s=$(cat "$N/opens")
            if [ "$s" != check ]; then
                echo "keyfile_gate: the stand-in cannot try ($s)" >&2
                exit "$s"
            fi
            [ "$line" = "$(cat "$N/phrase")" ] && exit 0
            exit 3 ;;
    esac
fi
echo "stand-in python3: not expected: $*" >&2
exit 9
"""

_ENV = r"""#!/bin/bash
N='@NOTES@'
printf 'argv:%s\n' "$*" >> "$N/env.log"
exec /usr/bin/env "$@"
"""

_LISTEN = r"""import http.server
import signal
import socket
import sys
import threading
import time

name, host, port, note, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]


def stop(*_args):
    time.sleep(1.0)
    with open(note, "a", encoding="ascii") as out:
        out.write(name + " listener stopped\n")
    raise SystemExit(0)


def beat():
    while True:
        with open(note + "." + name + ".beats", "a", encoding="ascii") as out:
            out.write("beat\n")
        time.sleep(0.2)


signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == "deaf" else stop)
threading.Thread(target=beat, daemon=True).start()
http.server.HTTPServer.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
http.server.HTTPServer((host, port), http.server.BaseHTTPRequestHandler).serve_forever()
"""

_BACKEND = r"""#!/bin/bash
N='@NOTES@'
# As the real backend does (Python ignores it), a closed pipe on its output does not end it.
trap '' PIPE
mode=$(cat "$N/backend")
delay=0.5
[ "$mode" = slow-stop ] && delay=3
[ "$mode" = stuck-stop ] && delay=60
trap 'sleep '"$delay"'; echo stopped >> "$N/backend.events"; exit 143' TERM
printf '%s\n' "${OPTI_KEYFILE_PASSPHRASE-<unset>}" > "$N/backend.env"
/usr/bin/env > "$N/backend.environ"
echo "$(awk '{print $6}' /proc/$$/stat) $(awk '{print $6}' /proc/$PPID/stat)" > "$N/backend.sid"
if [ "$mode" = dies ]; then
    echo "stand-in backend: it cannot start" >&2
    exit 3
fi
[ "$mode" = slow ] && sleep 2
listening=up
case "$mode" in stuck-stop|deaf-child) listening=deaf ;; esac
'@PY@' -I '@BIN@/listen.py' backend 127.0.0.1 8001 "$N/listeners" "$listening" >/dev/null 2>&1 &
while [ ! -e "$N/stop_backend" ]; do
    if [ "$mode" = dies-later ] && [ -e "$N/frontend.env" ]; then
        echo "stand-in backend: it stops" >&2
        exit 6
    fi
    if [ -e "$N/signal" ]; then
        sent=$(cat "$N/signal")
        rm -f "$N/signal"
        group=$(awk '{print $5}' /proc/$PPID/stat)
        case "$sent" in
            *-twice) kill -"${sent%-twice}" -- "-$group"; sleep 0.3; kill -"${sent%-twice}" -- "-$group" ;;
            *-group) kill -"${sent%-group}" -- "-$group" ;;
            *) kill -"$sent" "$PPID" ;;
        esac
    fi
    sleep 0.1
done
exit 7
"""

_FRONTEND = r"""#!/bin/bash
N='@NOTES@'
# As the real frontend does (node handles it), a closed pipe on its output does not end it.
trap '' PIPE
mode=$(cat "$N/frontend")
delay=0.5
[ "$mode" = slow-stop ] && delay=3
if [ "$mode" = interrupts ]; then
    trap 'kill -INT "$PPID"; sleep 0.5; echo stopped >> "$N/frontend.events"; exit 143' TERM
else
    trap 'sleep '"$delay"'; echo stopped >> "$N/frontend.events"; exit 143' TERM
fi
printf '%s\n' "${OPTI_KEYFILE_PASSPHRASE-<unset>}" > "$N/frontend.env"
/usr/bin/env > "$N/frontend.environ"
echo "$(awk '{print $6}' /proc/$$/stat) $(awk '{print $6}' /proc/$PPID/stat)" > "$N/frontend.sid"
if (exec 3<>/dev/tcp/127.0.0.1/8001) 2>/dev/null; then echo answered; else echo silent; fi > "$N/frontend.saw"
if [ "$mode" = dies ]; then
    echo "stand-in frontend: it cannot start" >&2
    exit 5
fi
[ "$mode" = slow ] && sleep 3
listening=up
[ "$mode" = deaf-child ] && listening=deaf
'@PY@' -I '@BIN@/listen.py' frontend "$(cat "$N/frontend_host")" 5173 "$N/listeners" "$listening" >/dev/null 2>&1 &
while [ ! -e "$N/stop_frontend" ]; do sleep 0.1; done
exit 5
"""

_XDG_OPEN = r"""#!/bin/bash
N='@NOTES@'
{ printf 'argv:%s\n' "$*"; printf 'env:%s\n' "${OPTI_KEYFILE_PASSPHRASE-<unset>}"; } > "$N/xdg.log"
/usr/bin/env > "$N/xdg.environ"
echo "$(awk '{print $6}' /proc/$$/stat) $(awk '{print $6}' /proc/$PPID/stat)" > "$N/xdg.sid"
sleep 30
"""

_NPM = "#!/bin/bash\nexit 0\n"

_OLD_SETPRIV = "#!/bin/bash\necho \"setpriv: unrecognized option '--pdeathsig'\" >&2\nexit 1\n"

_CONDA = r"""conda() {
    if [ "${1-}" = activate ]; then
        export OPTI_KEYFILE_PASSPHRASE="$(cat '@NOTES@/conda_phrase')"
        export OPTI_ENCRYPTION_KEY="$(cat '@NOTES@/conda_key')"
        touch '@NOTES@/activated'
        # An activation script that leaves options of its own set in the shell that sourced it.
        if [ -e '@NOTES@/noisy' ]; then set -o allexport; set -o xtrace; fi
    fi
}
"""


def _stubs(tmp_path: Path, *, wrapped="0", opens="check", backend="up", frontend="up", frontend_host="127.0.0.1",
           phrase=PHRASE, conda=None, noisy=False, old_setpriv=False) -> Path:
    """The launcher as it is, beside stand-ins that note what they are given; returns the notes' folder. ``conda``
    is the passphrase a conda base's activation sets, or None for no conda base -- ``noisy``, an activation that
    leaves allexport and xtrace set --; ``old_setpriv``, a setpriv that has no --pdeathsig."""
    root, notes, bin_, home = tmp_path / "root", tmp_path / "notes", tmp_path / "bin", tmp_path / "home"
    for folder in (root / "scripts", notes, bin_, home):
        folder.mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "launch.sh", root / "scripts" / "launch.sh")
    for name, value in (("wrapped", wrapped), ("opens", opens), ("phrase", phrase), ("backend", backend),
                        ("frontend", frontend), ("frontend_host", frontend_host), ("conda_key", MASTER),
                        ("conda_phrase", conda or "")):
        (notes / name).write_text(value, encoding="ascii")
    files = [(bin_ / "python3", _PYTHON3), (bin_ / "npm", _NPM), (bin_ / "xdg-open", _XDG_OPEN),
             (bin_ / "env", _ENV), (bin_ / "listen.py", _LISTEN), (root / "scripts" / "dev_backend.sh", _BACKEND),
             (root / "scripts" / "dev_frontend.sh", _FRONTEND)]
    if old_setpriv:
        files.append((bin_ / "setpriv", _OLD_SETPRIV))
    if noisy:
        (notes / "noisy").touch()
    if conda is not None:
        (notes / "import_fails").touch()
        (home / "miniconda3" / "etc" / "profile.d").mkdir(parents=True)
        files.append((home / "miniconda3" / "etc" / "profile.d" / "conda.sh", _CONDA))
    for path, text in files:
        path.write_text(text.replace("@NOTES@", str(notes)).replace("@PY@", sys.executable)
                        .replace("@BIN@", str(bin_)), encoding="ascii")
        path.chmod(0o755)
    return notes


class _Term:
    """The master side of the launcher's pseudo-terminal: everything it showed, and what is typed."""

    def __init__(self, master: int, slave_name: str | None = None):
        self.master, self.slave_name = master, slave_name
        self.shown = b""
        self.mark = 0
        self.closed = False

    def _read(self, timeout: float) -> bool:
        ready, _, _ = select.select([self.master], [], [], timeout)
        if not ready:
            return True
        try:
            chunk = os.read(self.master, 4096)
        except OSError:
            chunk = b""
        self.shown += chunk
        return bool(chunk)

    def expect(self, text: str, timeout: float = 20.0) -> None:
        """Wait until ``text`` shows after what was already expected."""
        wanted, end = text.encode(), time.monotonic() + timeout
        while wanted not in self.shown[self.mark:]:
            left = end - time.monotonic()
            if left <= 0 or not self._read(left):
                raise AssertionError(f"the launcher never showed {text!r}; it showed:\n{self.text}")
        self.mark = self.shown.index(wanted, self.mark) + len(wanted)

    def send(self, text: str) -> None:
        os.write(self.master, text.encode())

    def drain(self) -> None:
        while self._read(0.2) and len(self.shown) < 1_000_000:
            pass

    @property
    def text(self) -> str:
        return self.shown.decode(errors="replace")


def _launch(tmp_path: Path, *, env=None, tty=True, prelude="", hosts=None, pipe=None, reader="/usr/bin/cat",
            ctty=False, linger=None, own_proc=True, shell=None):
    """Run the launcher as its desktop icon does -- by its own first line -- or by ``shell``, in its own network and
    process tree, on a pseudo-terminal unless ``tty`` is False --
    its controlling terminal with ``ctty``; ``hosts`` is the /etc/hosts it sees; ``pipe``, a file its output goes
    to through ``reader``, in its group; ``linger``, seconds the process tree outlives the launcher, whose status is
    then noted in ``status``; ``own_proc`` False, this machine's /proc instead of one of its own."""
    environ = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "HOME": str(tmp_path / "home"), "LANG": "C.UTF-8"}
    environ.update(env or {})
    launcher = shlex.quote(str(tmp_path / "root" / "scripts" / "launch.sh"))
    if shell is not None:
        launcher = f"{shell} {launcher}"
    run = "setsid -c -w" if ctty else "setsid -w"
    if pipe is not None:
        # The pipeline's own shell outlives a signal, as an interactive shell does: the launcher and the reader
        # take it with their own dispositions.
        inner = prelude + f"exec {run} bash -o pipefail -c " + shlex.quote(
            f"trap : INT TERM HUP; {launcher} 2>&1 | {reader} > " + shlex.quote(str(pipe)))
    elif linger is not None:
        inner = prelude + f"{run} {launcher}; echo $? > {shlex.quote(str(tmp_path / 'status'))}; sleep {linger}"
    else:
        inner = prelude + f"exec {run} {launcher}"
    command = [_bwrap(), "--unshare-net", "--unshare-pid", "--die-with-parent", "--ro-bind", "/", "/"]
    if hosts is not None:
        (tmp_path / "hosts").write_text(hosts, encoding="ascii")
        command += ["--ro-bind", str(tmp_path / "hosts"), "/etc/hosts"]
    # The test's own shell, under -p, reads none of the hooks a case puts in the launcher's environment.
    command += ["--dev-bind", "/dev", "/dev"] + (["--proc", "/proc"] if own_proc else []) + [
        "--bind", str(tmp_path), str(tmp_path), "/bin/bash", "-p", "-c", inner]
    if not tty:
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=environ, start_new_session=True), None
    master, slave = pty.openpty()
    name = os.ttyname(slave)
    try:
        proc = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, env=environ,
                                start_new_session=True, close_fds=True)
    finally:
        os.close(slave)
    return proc, _Term(master, name)


def _end(proc, term, grace: float = 30.0) -> int | None:
    """Reap the launcher, killing what is left after ``grace`` seconds -- at once with 0, as a test that failed
    cleans up. Returns its exit status, None when it had to be killed."""
    try:
        if proc.poll() is None and grace:
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
            return None
        return proc.returncode
    finally:
        if term is not None and not term.closed:
            term.drain()
            os.close(term.master)
            term.closed = True


def _raw(notes: Path, name: str) -> str | None:
    """A note as written, its last newline taken off."""
    path = notes / name
    if not path.exists():
        return None
    text = path.read_text(encoding="ascii")
    return text[:-1] if text.endswith("\n") else text


def _note(notes: Path, name: str) -> str | None:
    text = _raw(notes, name)
    return None if text is None else text.strip()


def _calls(notes: Path) -> list[str]:
    return (_raw(notes, "python3.log") or "").splitlines()


def _typed(notes: Path) -> list[str]:
    """The lines the gate's stand-in read on its standard input, in order."""
    return [line[len("stdin:"):] for line in _calls(notes) if line.startswith("stdin:")]


def _until(check, what: str, timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while not check():
        if time.monotonic() > end:
            raise AssertionError(f"waited {timeout} s for {what}")
        time.sleep(0.05)


def _running(term, notes: Path) -> None:
    term.expect("Opti-Oignon is running.")
    _until(lambda: (notes / "xdg.log").exists(), "the browser opener")


def _stopped(notes: Path, *names: str) -> None:
    """Each of ``names`` (backend, frontend) and its listener, the slowest of its group, ended on the group's TERM."""
    listeners = (_raw(notes, "listeners") or "").splitlines()
    for name in names:
        assert _note(notes, f"{name}.events") == "stopped", name
        assert f"{name} listener stopped" in listeners, name


def _beats(notes: Path, name: str) -> int:
    return len((_raw(notes, f"listeners.{name}.beats") or "").splitlines())


def _close_on_backend_stop(proc, term, notes: Path) -> int | None:
    """Stop the backend stand-in, see the launcher say it and stop the frontend's whole group first, close the
    window."""
    (notes / "stop_backend").touch()
    term.expect("The backend stopped (exit status 7)")
    term.expect("Press Enter to close...")
    _stopped(notes, "frontend")
    term.send("\n")
    return _end(proc, term)


def _nothing_started(notes: Path) -> None:
    for name in ("backend.env", "frontend.env", "xdg.log"):
        assert not (notes / name).exists(), name


def _hooks(notes: Path) -> dict:
    """A BASH_ENV and a function exported as ``echo``, each noting the passphrase in the environment of every bash
    that takes it: what a shell's setup may hand the launcher."""
    hook = notes.parent / "bash_env.sh"
    hook.write_text(f"printf '%s\\n' \"${{{VAR}-<unset>}}\" >> {shlex.quote(str(notes / 'bash_env.log'))}\n",
                    encoding="ascii")
    return {"BASH_ENV": str(hook),
            "BASH_FUNC_echo%%": f"() {{ builtin printf '%s\\n' \"${{{VAR}-<unset>}}\" >> "
                                f"{shlex.quote(str(notes / 'fn.log'))}; builtin echo \"$@\"; }}"}


def _untouched_by_hooks(notes: Path) -> None:
    """Both hooks ran -- the stand-ins' bash took them -- and none saw a secret."""
    for name in ("bash_env.log", "fn.log"):
        seen = (_raw(notes, name) or "").splitlines()
        assert seen, name
        assert not any(secret in line for line in seen for secret in SECRETS), name


def _alone_in_the_backend(notes: Path, phrase: str, key: str | None = None) -> None:
    """``phrase`` (and ``key``) in the backend's environment; no secret under any name in any other's, nor on a
    command line; the gate run with -E."""
    secrets = SECRETS + (() if phrase == UNSET else (phrase,))
    assert _raw(notes, "backend.env") == phrase
    for name in ("frontend.environ", "xdg.environ", "python3.environ"):
        seen = _raw(notes, name) or ""
        assert not any(secret in seen for secret in secrets), name
    held = sorted(line for line in (_raw(notes, "backend.environ") or "").splitlines()
                  if any(secret in line for secret in secrets))
    assert held == sorted(([] if phrase == UNSET else [f"{VAR}={phrase}"]) + ([] if key is None else
                                                                            [f"{KEY_VAR}={key}"]))
    assert (_raw(notes, "xdg.log") or "").splitlines() == ["argv:http://localhost:5173", "env:" + UNSET]
    calls = _calls(notes)
    assert all(line == "env:" + UNSET for line in calls if line.startswith("env:"))
    gate_calls = [line for line in calls if "[opti_oignon.keyfile_gate]" in line]
    assert gate_calls and all(line.startswith("argv: [-E]") for line in gate_calls)
    lines = [line for line in calls if line.startswith("argv:")] + (_raw(notes, "env.log") or "").splitlines()
    assert not any(secret in line for line in lines for secret in secrets)


def test_ul3_the_launcher_asks_for_the_passphrase_and_hands_it_to_the_backend_alone(tmp_path, bwrap):
    notes = _stubs(tmp_path)
    proc, term = _launch(tmp_path, env=_hooks(notes))
    try:
        term.expect(PROMPT)
        _nothing_started(notes)
        term.send(WRONG + "\n")
        term.expect("That passphrase does not open the key file.")
        term.expect(PROMPT)
        term.send(PHRASE + "\n")
        _running(term, notes)
        _alone_in_the_backend(notes, PHRASE)
        assert _typed(notes) == [WRONG, PHRASE]
        assert _close_on_backend_stop(proc, term, notes) == 1
    finally:
        _end(proc, term, grace=0)
    _untouched_by_hooks(notes)
    # Typed without echo: neither passphrase ever showed.
    assert PHRASE not in term.text and WRONG not in term.text
    # What is typed is taken as it is typed: spaces at its ends, a backslash.
    (tmp_path / "odd").mkdir()
    notes = _stubs(tmp_path / "odd", phrase=ODD)
    proc, term = _launch(tmp_path / "odd")
    try:
        term.expect(PROMPT)
        term.send(ODD + "\n")
        _running(term, notes)
        _alone_in_the_backend(notes, ODD)
        assert _typed(notes) == [ODD]
        assert _close_on_backend_stop(proc, term, notes) == 1
    finally:
        _end(proc, term, grace=0)


def test_ul4_the_launcher_starts_nothing_it_cannot_open(tmp_path, bwrap):
    def refused(name, said, *, typed=(), prelude="", hosts=None, own_proc=True, **stubs):
        base = tmp_path / name
        base.mkdir()
        notes = _stubs(base, **stubs)
        proc, term = _launch(base, prelude=prelude, hosts=hosts, own_proc=own_proc)
        try:
            for text in typed:
                term.expect(PROMPT)
                term.send(text)
            for text in said:
                term.expect(text)
            term.expect("Press Enter to close...")
            term.send("\n")
            assert _end(proc, term) == 1, name
        finally:
            _end(proc, term, grace=0)
        _nothing_started(notes)
        return notes, term

    refused("three", ["Three passphrases did not open the key file. Nothing was started."],
            typed=[WRONG + "\n"] * 3)
    refused("none-read", ["No passphrase was read. Nothing was started."], typed=["\x04"])
    for status in ("4", "1"):
        refused(f"cannot-tell-{status}", [f"keyfile_gate: the stand-in cannot tell ({status})",
                                          "Could not tell whether the key file is wrapped under a passphrase"],
                wrapped=status)
        refused(f"cannot-try-{status}", [f"keyfile_gate: the stand-in cannot try ({status})",
                                         "Could not try the passphrase on the key file"],
                typed=[PHRASE + "\n"], opens=status)
    # A port that already answers, before the key file is even looked at -- the frontend's where the browser goes.
    for name, host, port, hosts, shown in (("8001", "127.0.0.1", 8001, None, "127.0.0.1:8001"),
                                           ("5173", "127.0.0.1", 5173, None, "localhost:5173"),
                                           ("5173-v6", "::1", 5173, "::1 localhost\n", "localhost:5173")):
        listen = (f"{shlex.quote(sys.executable)} -I -m http.server {port} --bind {host} >/dev/null 2>&1 & "
                  f"for i in $(seq 200); do (exec 3<>/dev/tcp/{host}/{port}) 2>/dev/null && break; "
                  "sleep 0.05; done; ")
        notes, term = refused(f"port-{name}", [f"Something already answers on {shown}: Opti-Oignon may be "
                                               "running already. Nothing was started."], prelude=listen,
                              hosts=hosts)
        assert not any("keyfile_gate" in line for line in _calls(notes))
        assert PROMPT not in term.text
    # A setpriv that cannot set the parent-death signal: nothing could be told to stop when the launcher ends.
    refused("old-setpriv", ["This setpriv cannot have what it starts told to stop when the launcher ends",
                            "Nothing was started."], old_setpriv=True)
    # Interrupted at its prompt: nothing started, 130, the terminal's echo back on.
    base = tmp_path / "interrupted"
    base.mkdir()
    notes = _stubs(base)
    proc, term = _launch(base, ctty=True)
    try:
        term.expect(PROMPT)
        term.send("\x03")
        proc.wait(timeout=30)
        tty = os.open(term.slave_name, os.O_RDWR | os.O_NOCTTY)
        try:
            assert termios.tcgetattr(tty)[3] & termios.ECHO
        finally:
            os.close(tty)
        assert _end(proc, term) == 130
    finally:
        _end(proc, term, grace=0)
    _nothing_started(notes)
    # A /proc that does not number processes as the launcher does -- not its own process tree's --: it could not
    # watch what it starts.
    notes, term = refused("foreign-proc", ["does not number processes as this launcher does", "Nothing was started."],
                          own_proc=False)
    assert not any("keyfile_gate" in line for line in _calls(notes))
    # No terminal to ask on: refused, unless a passphrase given opens the key file.
    base = tmp_path / "no-terminal"
    base.mkdir()
    notes = _stubs(base)
    proc, _ = _launch(base, tty=False)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 1
    assert b"no terminal to ask for it on" in out and b"Nothing was started." in out
    _nothing_started(notes)
    base = tmp_path / "no-terminal-given"
    base.mkdir()
    notes = _stubs(base)
    proc, _ = _launch(base, tty=False, env={VAR: PHRASE})
    try:
        _until(lambda: (notes / "xdg.log").exists(), "the browser opener")
        _alone_in_the_backend(notes, PHRASE)
        (notes / "stop_backend").touch()
        out, _ = proc.communicate(timeout=60)
    finally:
        _end(proc, None, grace=0)
    assert proc.returncode == 1 and b"The backend stopped (exit status 7)" in out


def test_ul5_a_passphrase_given_to_the_launcher_is_checked_not_asked(tmp_path, bwrap):
    (tmp_path / "opens").mkdir()
    notes = _stubs(tmp_path / "opens")
    # A BASH_ENV and a function in its environment: the launcher, holding both secrets until it takes them, runs
    # neither.
    proc, term = _launch(tmp_path / "opens", env={VAR: PHRASE, KEY_VAR: MASTER, **_hooks(notes)})
    try:
        _running(term, notes)
        _alone_in_the_backend(notes, PHRASE, MASTER)
        assert _typed(notes) == [PHRASE]
        assert _close_on_backend_stop(proc, term, notes) == 1
    finally:
        _end(proc, term, grace=0)
    _untouched_by_hooks(notes)
    assert PROMPT not in term.text
    cases = (("wrong", WRONG, "The passphrase this launcher was started with does not open the key file.", [WRONG]),
             ("lines", PHRASE + "\nmore", "The passphrase this launcher was started with holds several lines: "
                                          "it is not taken.", []),
             ("empty", "", None, []))
    for name, given, said, tried in cases:
        (tmp_path / name).mkdir()
        notes = _stubs(tmp_path / name)
        proc, term = _launch(tmp_path / name, env={VAR: given})
        try:
            if said is not None:
                term.expect(said)
            term.expect(PROMPT)
            _nothing_started(notes)
            term.send(PHRASE + "\n")
            _running(term, notes)
            _alone_in_the_backend(notes, PHRASE)
            assert _close_on_backend_stop(proc, term, notes) == 1
        finally:
            _end(proc, term, grace=0)
        assert _typed(notes) == tried + [PHRASE], name


def test_ul6_the_backend_first_and_a_stop_said_in_sight(tmp_path, bwrap):
    def run(name, *, hosts=None, ctty=False, **stubs):
        (tmp_path / name).mkdir()
        notes = _stubs(tmp_path / name, **stubs)
        return notes, *_launch(tmp_path / name, env={VAR: PHRASE}, hosts=hosts, ctty=ctty)

    # A backend that stops before it answers: said with its status, the frontend never started; once that is said,
    # an interruption ends the launcher.
    notes, proc, term = run("dies", backend="dies", ctty=True)
    try:
        term.expect("stand-in backend: it cannot start")
        term.expect("The backend stopped before it answered (exit status 3)")
        term.expect("The frontend was not started.")
        term.expect("Press Enter to close...")
        term.send("\x03")
        assert _end(proc, term) == 130
    finally:
        _end(proc, term, grace=0)
    assert _raw(notes, "backend.env") == PHRASE and not (notes / "frontend.env").exists()
    assert "Starting frontend" not in term.text
    # Its window closed at that prompt: the hangup ends it (129).
    notes, proc, term = run("dies-closed", backend="dies", ctty=True)
    try:
        term.expect("Press Enter to close...")
        os.close(term.master)
        term.closed = True
        assert _end(proc, None) == 129
    finally:
        _end(proc, term, grace=0)
    # A slow backend: the frontend starts once it answers. What the launcher starts -- the backend, the
    # frontend, the browser opener -- runs in a session of its own, which the terminal neither interrupts nor hangs
    # up; the browser opener, which waits, holds nothing up.
    notes, proc, term = run("slow", backend="slow")
    try:
        _running(term, notes)
        assert _note(notes, "frontend.saw") == "answered"
        for name in ("backend", "frontend", "xdg"):
            own, parents = (_note(notes, f"{name}.sid") or " ").split(" ")
            assert own and parents and own != parents, name
        assert _close_on_backend_stop(proc, term, notes) == 1
    finally:
        _end(proc, term, grace=0)
    # A backend that stops while the frontend starts: said, the browser never opened, the frontend stopped first.
    notes, proc, term = run("dies-later", backend="dies-later", frontend="slow")
    try:
        term.expect("The backend stopped (exit status 6) while the frontend started")
        term.expect("Press Enter to close...")
        # The frontend, still starting, had no listener yet; the backend's outlived it, and was stopped by its group.
        assert _note(notes, "frontend.events") == "stopped"
        assert "backend listener stopped" in (_raw(notes, "listeners") or "")
        term.send("\n")
        assert _end(proc, term) == 1
    finally:
        _end(proc, term, grace=0)
    assert not (notes / "xdg.log").exists()
    # A frontend that stops before it answers: said with its status, the backend's whole group stopped, no browser.
    notes, proc, term = run("frontend-dies", frontend="dies")
    try:
        term.expect("stand-in frontend: it cannot start")
        term.expect("The frontend stopped before it answered (exit status 5)")
        term.expect("Press Enter to close...")
        _stopped(notes, "backend")
        term.send("\n")
        assert _end(proc, term) == 1
    finally:
        _end(proc, term, grace=0)
    assert not (notes / "xdg.log").exists()
    # A frontend that stops later: said, the backend's whole group stopped before the message.
    notes, proc, term = run("frontend-stops")
    try:
        _running(term, notes)
        (notes / "stop_frontend").touch()
        term.expect("The frontend stopped (exit status 5)")
        term.expect("Press Enter to close...")
        _stopped(notes, "backend")
        term.send("\n")
        assert _end(proc, term) == 1
    finally:
        _end(proc, term, grace=0)
    # Interrupted while it stops what it started for an error: the stop and the message go on whole.
    notes, proc, term = run("interrupted", frontend="interrupts")
    try:
        _running(term, notes)
        (notes / "stop_backend").touch()
        term.expect("The backend stopped (exit status 7)")
        term.expect("Press Enter to close...")
        _stopped(notes, "frontend")
        term.send("\n")
        assert _end(proc, term) == 1
    finally:
        _end(proc, term, grace=0)
    # The frontend looked for where the browser goes: localhost, here ::1 alone.
    notes, proc, term = run("v6", frontend_host="::1", hosts="::1 localhost\n127.0.0.1 v4only\n")
    try:
        _running(term, notes)
        assert _close_on_backend_stop(proc, term, notes) == 1
    finally:
        _end(proc, term, grace=0)
    # Its output through a pipe in its own group: nothing it stops takes the pipe, and the message reaches it.
    (tmp_path / "piped").mkdir()
    notes = _stubs(tmp_path / "piped", backend="dies")
    out = tmp_path / "piped" / "out.txt"
    proc, term = _launch(tmp_path / "piped", env={VAR: PHRASE}, pipe=out)
    try:
        _until(lambda: "Press Enter to close..." in (out.read_text(encoding="ascii") if out.exists() else ""),
               "the message through the pipe")
        term.send("\n")
        assert _end(proc, term) == 1
    finally:
        _end(proc, term, grace=0)
    assert "ERROR: The backend stopped before it answered (exit status 3)" in out.read_text(encoding="ascii")


def test_ul7_a_key_file_not_wrapped_is_asked_nothing(tmp_path, bwrap):
    cases = (("given", {VAR: PHRASE}, PHRASE, None),
             ("none", {}, UNSET, None),
             ("empty", {VAR: ""}, UNSET, None),
             ("lines", {VAR: PHRASE + "\nmore"}, UNSET,
              "The passphrase this launcher was started with holds several lines: it is not taken."))
    for name, env, held, said in cases:
        (tmp_path / name).mkdir()
        notes = _stubs(tmp_path / name, wrapped="3")
        proc, term = _launch(tmp_path / name, env=env)
        try:
            if said is not None:
                term.expect(said)
            _running(term, notes)
            if held != UNSET:
                _alone_in_the_backend(notes, held)
            else:
                assert _raw(notes, "backend.env") == UNSET, name
            assert _typed(notes) == [], name
            assert _close_on_backend_stop(proc, term, notes) == 1
        finally:
            _end(proc, term, grace=0)
        assert PROMPT not in term.text, name


def test_ul11_the_launcher_stops_what_it_started_on_a_signal(tmp_path, bwrap):
    # A closing window hangs up the launcher's whole process group, as a terminal does, once or twice: what it
    # started, in sessions of their own, is told to stop by the launcher, not hung up. Whichever ends last, the
    # backend or the frontend, is waited for.
    for name, sent, backend, frontend, status in (
            ("HUP", "HUP", "up", "up", 129), ("HUP-group", "HUP-group", "up", "up", 129),
            ("HUP-twice", "HUP-twice", "slow-stop", "up", 129), ("INT", "INT", "up", "slow-stop", 130),
            ("TERM", "TERM", "slow-stop", "up", 143)):
        (tmp_path / name).mkdir()
        notes = _stubs(tmp_path / name, backend=backend, frontend=frontend)
        proc, term = _launch(tmp_path / name, env={VAR: PHRASE})
        try:
            _running(term, notes)
            (notes / "signal").write_text(sent, encoding="ascii")
            term.expect("Stopping Opti-Oignon...")
            assert _end(proc, term) == status, name
        finally:
            _end(proc, term, grace=0)
        _stopped(notes, "backend", "frontend")
    # Interrupted with its output's reader, which the interruption ends too: no closed pipe stops the launcher.
    (tmp_path / "piped").mkdir()
    notes = _stubs(tmp_path / "piped")
    proc, term = _launch(tmp_path / "piped", env={VAR: PHRASE}, pipe=tmp_path / "piped" / "out.txt")
    try:
        _until(lambda: (notes / "xdg.log").exists(), "the browser opener")
        (notes / "signal").write_text("INT-group", encoding="ascii")
        _end(proc, term)
    finally:
        _end(proc, term, grace=0)
    _stopped(notes, "backend", "frontend")
    # Its output's reader gone, nothing else: the launcher stops what it started, as on any other signal.
    (tmp_path / "reader-gone").mkdir()
    notes = _stubs(tmp_path / "reader-gone")
    proc, term = _launch(tmp_path / "reader-gone", env={VAR: PHRASE}, pipe=tmp_path / "reader-gone" / "out.txt",
                         reader="/usr/bin/sed '/Waiting for the backend/q'")
    try:
        assert _end(proc, term) == 141
    finally:
        _end(proc, term, grace=0)
    assert _note(notes, "backend.events") == "stopped"
    if (notes / "frontend.env").exists():
        assert _note(notes, "frontend.events") == "stopped"
    # What does not end in 15 s is killed, its whole group, and that is said: here a backend and a frontend whose
    # leaders ended while their groups kept a member deaf to TERM. Once the launcher has ended, nothing of either is
    # left.
    (tmp_path / "stuck").mkdir()
    notes = _stubs(tmp_path / "stuck", backend="deaf-child", frontend="deaf-child")
    proc, term = _launch(tmp_path / "stuck", env={VAR: PHRASE}, linger=6)
    try:
        _running(term, notes)
        (notes / "signal").write_text("TERM", encoding="ascii")
        term.expect("Stopping Opti-Oignon...")
        told = time.monotonic()
        term.expect("The frontend did not end within 15 s of being told to: it is killed.", timeout=40)
        assert time.monotonic() - told >= 14
        term.expect("The backend did not end within 15 s of being told to: it is killed.")
        _until(lambda: (tmp_path / "stuck" / "status").exists(), "the launcher's end")
        assert _note(tmp_path / "stuck", "status") == "143"
        beats = (_beats(notes, "backend"), _beats(notes, "frontend"))
        time.sleep(0.8)
        assert (_beats(notes, "backend"), _beats(notes, "frontend")) == beats
        _end(proc, term)
    finally:
        _end(proc, term, grace=0)
    assert _note(notes, "backend.events") == "stopped" and _note(notes, "frontend.events") == "stopped"
    assert not _raw(notes, "listeners")
    # Killed itself, the launcher leaves what it started told to stop.
    (tmp_path / "killed").mkdir()
    notes = _stubs(tmp_path / "killed")
    proc, term = _launch(tmp_path / "killed", env={VAR: PHRASE}, linger=4)
    try:
        _running(term, notes)
        (notes / "signal").write_text("KILL", encoding="ascii")
        _end(proc, term)
    finally:
        _end(proc, term, grace=0)
    assert _note(notes, "backend.events") == "stopped" and _note(notes, "frontend.events") == "stopped"


def test_ul12_secrets_a_conda_base_sets_are_taken_as_given(tmp_path, bwrap):
    (tmp_path / "taken").mkdir()
    notes = _stubs(tmp_path / "taken", conda=PHRASE)
    proc, term = _launch(tmp_path / "taken")
    try:
        _running(term, notes)
        assert (notes / "activated").exists()
        _alone_in_the_backend(notes, PHRASE, MASTER)
        assert _typed(notes) == [PHRASE]
        assert _close_on_backend_stop(proc, term, notes) == 1
    finally:
        _end(proc, term, grace=0)
    assert PROMPT not in term.text
    # An activation that leaves allexport and xtrace set: neither exports nor shows what the launcher holds.
    (tmp_path / "noisy").mkdir()
    notes = _stubs(tmp_path / "noisy", conda=PHRASE, noisy=True)
    proc, term = _launch(tmp_path / "noisy")
    try:
        _running(term, notes)
        assert (notes / "activated").exists()
        _alone_in_the_backend(notes, PHRASE, MASTER)
        assert _close_on_backend_stop(proc, term, notes) == 1
    finally:
        _end(proc, term, grace=0)
    assert PHRASE not in term.text and MASTER not in term.text
    # Never over the passphrase the launcher was handed.
    (tmp_path / "handed").mkdir()
    notes = _stubs(tmp_path / "handed", conda=WRONG)
    proc, term = _launch(tmp_path / "handed", env={VAR: PHRASE})
    try:
        _running(term, notes)
        assert (notes / "activated").exists()
        _alone_in_the_backend(notes, PHRASE, MASTER)
        assert _typed(notes) == [PHRASE]
        assert _close_on_backend_stop(proc, term, notes) == 1
    finally:
        _end(proc, term, grace=0)
    assert PROMPT not in term.text


def test_ul13_inherited_shell_options_neither_export_nor_show_the_passphrase(tmp_path, bwrap):
    notes = _stubs(tmp_path)
    proc, term = _launch(tmp_path, env={"PYTHONUTF8": "0"}, ctty=True,
                         shell="bash -o allexport -o errexit -o monitor -o xtrace")
    try:
        term.expect(PROMPT)
        term.send(WRONG + "\n")
        term.expect("That passphrase does not open the key file.")
        term.expect(PROMPT)
        term.send(PHRASE + "\n")
        _running(term, notes)
        _alone_in_the_backend(notes, PHRASE)
        gate_calls = [line for line in _calls(notes) if "[opti_oignon.keyfile_gate]" in line]
        assert all(line.startswith("argv: [-E] [-X] [utf8=0] [-m]") for line in gate_calls)
        assert _close_on_backend_stop(proc, term, notes) == 1
    finally:
        _end(proc, term, grace=0)
    assert PHRASE not in term.text and WRONG not in term.text


_RECORDER = r"""#!/bin/bash
{ printf 'cwd:%s\n' "$(pwd -P)"; printf 'argv:'; printf ' [%s]' "$@"; printf '\n'; } >> '@NOTES@/python3.log'
"""


def test_ul14_the_backend_script_runs_no_program_but_the_server(tmp_path):
    root, notes, bin_ = tmp_path / "root", tmp_path / "notes", tmp_path / "bin"
    for folder in (root / "scripts", notes, bin_, tmp_path / "elsewhere"):
        folder.mkdir(parents=True)
    script = root / "scripts" / "dev_backend.sh"
    shutil.copy2(REPO / "scripts" / "dev_backend.sh", script)
    (bin_ / "python3").write_text(_RECORDER.replace("@NOTES@", str(notes)), encoding="ascii")
    (bin_ / "python3").chmod(0o755)
    bash = shutil.which("bash") or "/bin/bash"
    # Its PATH holds python3 alone: any other program it ran would not be found, and that would be said.
    alone = {"PATH": str(bin_), "HOME": str(tmp_path), "LANG": "C.UTF-8"}
    for how, args, cwd in (("by its path", [bash, str(script)], tmp_path / "elsewhere"),
                           ("by its name", [bash, "dev_backend.sh"], root / "scripts")):
        done = subprocess.run(args, cwd=cwd, env=alone, capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, (how, done.stderr)
        assert "not found" not in done.stderr, how
    server = ("argv: [-m] [uvicorn] [opti_oignon.api.app:app] [--host] [127.0.0.1] [--port] [8001] "
              "[--timeout-graceful-shutdown] [10]")
    assert _calls(notes) == [f"cwd:{os.path.realpath(root)}", server] * 2
    # Traced, it runs no subshell, and no command but bash's own and the server: none by an absolute path either.
    traced = subprocess.run([bash, "-x", str(script)], cwd=tmp_path / "elsewhere", env=alone, capture_output=True,
                            text=True, timeout=60, check=True).stderr.splitlines()
    assert not any(line.startswith("++") for line in traced)
    ran = [line[2:].split()[0] for line in traced if line.startswith("+ ")]
    assert all(word in ("set", "[[", "case", "shift", "cd", "echo", "exec") or "=" in word for word in ran), ran
    assert traced[-1].startswith("+ exec python3 -m uvicorn ")
    # --help prints its header, without its comment marks, and nothing past it.
    shown = subprocess.run([bash, str(script), "--help"], cwd=tmp_path, env={"PATH": "/usr/bin:/bin"},
                           capture_output=True, text=True, timeout=60, check=True).stdout
    assert shown.startswith("Launch the Opti-Oignon backend in development mode.\n") and "Usage:" in shown
    assert shown.rstrip("\n").endswith("(see frontend/vite.config.ts).") and "#" not in shown


# --------------------------------------------------------------------------- ui.py


class _Proc:
    """A stand-in process leading a process group: ``polls`` answers its leader's first polls, then ``status``
    every later one; ``stuck``, its group ignores TERM; ``lingers``, its group keeps a member once its leader
    ends; ``deaf_child``, its leader ends on TERM while its group keeps a member deaf to it."""

    last_pid = 1000

    def __init__(self, polls, status, stuck=False, lingers=False, deaf_child=False):
        self.polls, self.status, self.stuck, self.lingers = list(polls), status, stuck, lingers
        self.deaf_child = deaf_child
        self.returncode, self.group, self.signals = None, True, []
        _Proc.last_pid += 1
        self.pid = _Proc.last_pid

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        answer = self.polls.pop(0) if self.polls else self.status
        if answer is not None:
            self.returncode = answer
            self.group = self.lingers
        return answer

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.signals.append("kill")


class _Busy(io.TextIOBase):
    """An output in use by what a signal interrupted: every write re-enters its buffer, as Python says."""

    def write(self, _text):
        raise RuntimeError("reentrant call inside <_io.BufferedWriter name='<stdout>'>")

    def flush(self):
        raise RuntimeError("reentrant call inside <_io.BufferedWriter name='<stdout>'>")


class _Answer:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _ui(tmp_path, monkeypatch, *, given=None, key=None, tty=True, typed=(), wrapped=0, opens="check",
        backend="up", frontend="up", silent_polls=0, stuck=False, lingers=False, signal_at=None,
        backend_end=9, deaf_child=False, again=None, in_popen=None, output=None, platform=None, ignored=()):
    """Run ``ui.launch()`` with everything around it stood in for, the gate, the clock and the process groups
    included; ``signal_at`` is a signal its handler takes at the first pause once both run -- its output gone just
    before, a closed window's terminal or a pipe without a reader, by ``output`` --; ``again``, signals sent once
    more at the first pause of the stop; ``in_popen``, the signal taken while a child (``uvicorn``, ``npm``) is
    being started; ``platform``, the system it runs on; ``ignored``, signals ignored when it starts. Returns what it
    did and how it ended."""
    for name in (VAR, KEY_VAR):
        monkeypatch.delenv(name, raising=False)
    if given is not None:
        monkeypatch.setenv(VAR, given)
    if key is not None:
        monkeypatch.setenv(KEY_VAR, key)
    root = tmp_path / "project"
    (root / "frontend").mkdir(parents=True)
    (root / "opti_oignon").mkdir()
    did = SimpleNamespace(popen=[], browser=[], asked=0, probes=0, code=None, frontend_proc=None, backend_proc=None,
                          gate=[], steps=[], handlers={}, groups=[], procs={}, now=0.0, signalled=False,
                          again=False, gone=None, flushed=None, installed={})
    typed = list(typed)

    def ask(prompt="Password: "):
        did.asked += 1
        if not typed:
            raise EOFError
        answer = typed.pop(0)
        if answer == "^C":
            did.handlers[signal.SIGINT](signal.SIGINT, None)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def urlopen(request, *_args, **_kwargs):
        did.probes += 1
        if request.full_url != "http://localhost:8001/api/health":
            raise urllib.error.URLError(f"nothing answers at {request.full_url}")
        if backend == "up" and did.probes > silent_polls:
            return _Answer()
        raise urllib.error.URLError("nothing answers here")

    def run(args, **_kwargs):
        raise AssertionError(f"ui.py ran a command it should not: {list(args)}")

    class GateError(Exception):
        pass

    def is_wrapped(root_asked=None):
        did.gate.append(SimpleNamespace(action="wrapped", passphrase=None, root=root_asked, env=dict(os.environ)))
        if wrapped == 4:
            raise GateError("the key file cannot be read: a stand-in")
        return wrapped == 0

    def opens_it(passphrase, root_asked=None):
        did.gate.append(SimpleNamespace(action="opens", passphrase=passphrase, root=root_asked,
                                        env=dict(os.environ)))
        if opens != "check":
            raise GateError("the key file cannot be opened here: a stand-in")
        return passphrase == PHRASE

    module = types.ModuleType(GATE)
    module.YES, module.NO, module.CANNOT = 0, 3, 4
    module.GateError, module.wrapped, module.opens = GateError, is_wrapped, opens_it

    def popen(args, **kwargs):
        did.popen.append((list(args), kwargs))
        if "uvicorn" in args:
            if backend == "dies":
                made = _Proc([None], 3, lingers=lingers)
            else:
                made = _Proc([None] * (silent_polls + (6 if frontend == "stops" else 30)), backend_end)
            did.backend_proc = made
        else:
            if frontend == "missing":
                raise FileNotFoundError(2, "No such file or directory", "npm")
            if frontend == "refused":
                raise subprocess.SubprocessError("Exception occurred in preexec_fn.")
            made = did.frontend_proc = _Proc([None, None] if frontend == "stops" else [],
                                             5 if frontend == "stops" else None, stuck=stuck,
                                             lingers=lingers and frontend == "stops", deaf_child=deaf_child)
        did.procs[made.pid] = made
        taken = (in_popen or {}).get("uvicorn" if "uvicorn" in args else "npm")
        if taken is not None:
            did.handlers[taken](taken, None)
        return made

    def killpg(pid, sig):
        proc = did.procs[pid]
        if sig == 0:
            if proc.group == "denied":
                raise PermissionError(pid)
            if not proc.group:
                raise ProcessLookupError(pid)
            return
        did.groups.append((pid, sig, did.now))
        if sig == signal.SIGTERM and not proc.stuck:
            proc.group = proc.deaf_child
            proc.returncode = -15 if proc.returncode is None else proc.returncode
        if sig == signal.SIGKILL:
            proc.group = False
            proc.returncode = -9

    def gone(kind):
        """The launcher's output, gone: a terminal whose window closed (EIO), or a pipe whose reader ended
        (EPIPE)."""
        if kind == "busy":
            did.gone = _Busy()
            sys.stdout = did.gone
            return
        if kind == "tty":
            master, fd = os.openpty()
            os.close(master)
            buffering = 1  # a terminal's output goes out line by line
        else:
            reader, fd = os.pipe()
            os.close(reader)
            buffering = -1  # a pipe's, in blocks
        did.gone = open(fd, "w", buffering=buffering, encoding="utf-8")
        sys.stdout = did.gone

    def pause(seconds):
        did.now += seconds
        if (signal_at is not None and not did.signalled and seconds == 2 and did.frontend_proc is not None):
            did.signalled = True
            if output is not None:
                gone(output)
            did.handlers[signal_at](signal_at, None)
        elif again and did.signalled and not did.again and seconds == 0.1:
            did.again = True
            for sig in again:
                if callable(did.handlers[sig]):
                    did.handlers[sig](sig, None)

    def handle(sig, handler):
        did.handlers[sig] = handler
        did.installed.setdefault(sig, handler)

    def step(name, answer):
        def seen(*_args, **_kwargs):
            did.steps.append((name, dict(os.environ)))
            return answer
        return seen

    monkeypatch.setattr(getpass, "getpass", ask)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: tty))
    loaded, restore = isolate(targets={"opti_oignon.process_secrets": _source("process_secrets.py"), UI: _source("ui.py")},
                              seeded={} if wrapped == "import" else {GATE: module})
    try:
        ui = loaded[UI]
        ui._find_project_root = lambda: root
        ui._check_ollama = step("ollama", True)
        ui._install_frontend = step("install", True)
        ui._free_port = step("ports", True)
        ui._update_vite_proxy = step("proxy", None)
        ui._get_version = lambda _port: "test"
        ui._open_browser = lambda url: did.browser.append((url, dict(os.environ)))
        ui.signal = SimpleNamespace(signal=handle, SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM,
                                    SIGHUP=signal.SIGHUP, SIGKILL=signal.SIGKILL, SIG_IGN=signal.SIG_IGN,
                                    SIG_DFL=signal.SIG_DFL,
                                    getsignal=lambda sig: signal.SIG_IGN if sig in ignored else signal.SIG_DFL)
        ui.time = SimpleNamespace(sleep=pause, monotonic=lambda: did.now)
        ui.subprocess = SimpleNamespace(Popen=popen, run=run, DEVNULL=subprocess.DEVNULL, PIPE=subprocess.PIPE,
                                        TimeoutExpired=subprocess.TimeoutExpired)
        did.die_with_launcher = getattr(ui, "_die_with_launcher", None)
        did.ui = ui
        system, shown = sys.platform, sys.stdout
        try:
            if platform is not None:
                sys.platform = platform
            ui.launch()
        except SystemExit as exc:
            did.code = exc.code
        finally:
            sys.platform, sys.stdout = system, shown
            if did.gone is not None:
                # As Python does at exit: what is left to say is flushed -- into /dev/null once the output is gone.
                try:
                    did.gone.flush()
                    did.flushed = True
                except (OSError, RuntimeError):
                    did.flushed = False
                try:
                    did.gone.close()
                except (OSError, RuntimeError):
                    pass
    finally:
        restore()
    did.root = root
    did.left_in_environ = [name for name in (VAR, KEY_VAR) if name in os.environ]
    did.backend = next((kw for args, kw in did.popen if "uvicorn" in args), None)
    did.frontend = next((kw for args, kw in did.popen if "npm" in args), None)
    return did


def _clean(env: dict) -> bool:
    """No secret, under any name, in ``env``."""
    return not ({VAR, KEY_VAR} & set(env)) and not any(secret in value for value in env.values()
                                                     for secret in SECRETS)


def test_ul8_the_python_launcher_hands_the_secrets_to_the_backend_alone(tmp_path, monkeypatch, capsys):
    cases = (
        ("given", dict(given=PHRASE, key=MASTER), PHRASE, MASTER, 0, [PHRASE]),
        ("asked", dict(typed=(WRONG, PHRASE)), PHRASE, None, 2, [WRONG, PHRASE]),
        ("given-wrong", dict(given=WRONG, typed=(PHRASE,)), PHRASE, None, 1, [WRONG, PHRASE]),
        ("empty", dict(given="", typed=(PHRASE,)), PHRASE, None, 1, [PHRASE]),
        ("lines", dict(given=PHRASE + "\nmore", typed=(PHRASE,)), PHRASE, None, 1, [PHRASE]),
        ("not-wrapped", dict(wrapped=3, given=PHRASE), PHRASE, None, 0, []),
        ("not-wrapped-none", dict(wrapped=3), None, None, 0, []),
        ("not-wrapped-empty", dict(wrapped=3, given="", key=""), None, None, 0, []),
    )
    for name, kwargs, held, key, asked, tried in cases:
        did = _ui(tmp_path / name, monkeypatch, **kwargs)
        said = capsys.readouterr().out
        assert did.left_in_environ == [], name
        assert did.asked == asked, name
        assert did.backend["env"].get(VAR) == held, name
        assert did.backend["env"].get(KEY_VAR) == key, name
        assert _clean(did.frontend["env"]), name
        for started in (did.backend, did.frontend):
            assert started.get("start_new_session") is True and started.get("stdin") == subprocess.DEVNULL, name
            hook = started.get("preexec_fn")
            assert isinstance(hook, functools.partial) and hook.func is did.die_with_launcher is not None, name
            assert hook.args[1] == os.getpid(), name
        server = next(args for args, _kwargs in did.popen if "uvicorn" in args)
        assert server[server.index("--timeout-graceful-shutdown") + 1] == "10", name
        # Each handled from the start: a handler of its own, before anything runs.
        assert set(did.installed) == {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}, name
        assert all(callable(handler) for handler in did.installed.values()), name
        assert did.browser and all(_clean(env) for _url, env in did.browser), name
        assert [step for step, _env in did.steps] == ["ollama", "install", "ports", "ports", "proxy"], name
        assert all(_clean(env) for _step, env in did.steps), name
        # The gate: asked about the project's own key file whether it is wrapped, then each passphrase tried, the
        # environment already without the secrets.
        assert [call.action for call in did.gate] == ["wrapped"] + ["opens"] * len(tried), name
        assert all(call.root == did.root for call in did.gate), name
        assert [call.passphrase for call in did.gate[1:]] == tried, name
        assert all(_clean(call.env) for call in did.gate), name
        assert not any(secret in said for secret in SECRETS), name
        if name == "given-wrong":
            assert "The passphrase this launcher was started with does not open the key file." in said
        if name == "lines":
            assert "The passphrase this launcher was started with holds several lines: it is not taken." in said
    # The hook has the kernel tell the child to stop (TERM) when this launcher ends; with that launcher gone already
    # -- here, a parent that is not the launcher it names --, the child ends at once and runs nothing.
    hook = did.backend["preexec_fn"]
    libc = ctypes.CDLL(None, use_errno=True)
    read, write = os.pipe()

    def probe():
        hook()
        told = ctypes.c_int(0)
        libc.prctl(2, ctypes.byref(told))  # PR_GET_PDEATHSIG
        os.write(write, str(told.value).encode("ascii"))

    true = shutil.which("true") or "/bin/true"
    try:
        child = subprocess.Popen([true], preexec_fn=probe, pass_fds=(write,))
    finally:
        os.close(write)
    with os.fdopen(read, "rb") as seen:
        assert child.wait(timeout=30) == 0 and seen.read() == str(int(signal.SIGTERM)).encode("ascii")
    orphan = subprocess.Popen([true], preexec_fn=functools.partial(hook.func, hook.args[0], hook.args[1] + 1))
    assert orphan.wait(timeout=30) == 128 + signal.SIGTERM
    # The kernel's setting first, the check of the launcher after it: a launcher that ends between the two is still
    # seen; a setting refused is an error, never a child left unwatched.
    calls = []

    class Libc:
        def __init__(self, answer):
            self.answer = answer

        def prctl(self, option, sig):
            calls.append(("prctl", option, int(sig)))
            return self.answer

    class Ended(Exception):
        pass

    def end(code):
        raise Ended(code)

    with monkeypatch.context() as patched:
        patched.setattr(os, "getppid", lambda: calls.append(("getppid",)) or 1)
        patched.setattr(os, "_exit", end)
        patched.setattr(did.ui, "signal", SimpleNamespace(
            signal=lambda sig, handler: calls.append(("signal", int(sig), handler)), SIGTERM=signal.SIGTERM,
            SIG_DFL=signal.SIG_DFL))
        with pytest.raises(Ended):
            did.die_with_launcher(Libc(0), os.getpid())
        # Its TERM back to the default before anything: the launcher's handler, its own until it runs, would drop it.
        assert calls == [("signal", int(signal.SIGTERM), signal.SIG_DFL), ("prctl", 1, int(signal.SIGTERM)),
                         ("getppid",)]
        with pytest.raises(OSError):
            did.die_with_launcher(Libc(-1), os.getpid())
    # Elsewhere than on Linux there is no such signal: nothing runs before a child starts.
    for system in ("darwin", "freebsd14"):
        did = _ui(tmp_path / system, monkeypatch, given=PHRASE, platform=system)
        capsys.readouterr()
        assert did.backend.get("preexec_fn") is None and did.frontend.get("preexec_fn") is None, system
        assert did.backend.get("start_new_session") is True and did.backend["env"].get(VAR) == PHRASE, system
    # A signal ignored when it started -- under nohup, as a background job -- stays ignored, as a shell keeps it: no
    # handler for it, and one for the others.
    did = _ui(tmp_path / "nohup", monkeypatch, given=PHRASE, ignored=(signal.SIGHUP,))
    capsys.readouterr()
    assert not callable(did.installed.get(signal.SIGHUP)) and callable(did.installed.get(signal.SIGINT))
    did = _ui(tmp_path / "background", monkeypatch, given=PHRASE,
              ignored=(signal.SIGINT, signal.SIGTERM, signal.SIGHUP))
    capsys.readouterr()
    assert not any(callable(did.installed.get(sig)) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP))


def test_ul9_the_python_launcher_starts_nothing_it_cannot_open(tmp_path, monkeypatch, capsys):
    cases = (
        ("no-terminal", dict(tty=False), "no terminal to ask for it on"),
        ("three", dict(typed=(WRONG, WRONG, WRONG)), "3 passphrases did not open the key file"),
        ("none-read", dict(), "No passphrase was read"),
        ("cannot-tell", dict(wrapped=4, given=PHRASE), "Could not tell whether the key file is wrapped"),
        ("gate-missing", dict(wrapped="import", given=PHRASE), "Could not tell whether the key file is wrapped"),
        ("cannot-try", dict(opens=4, given=PHRASE), "Could not try the passphrase on the key file"),
        ("cannot-try-typed", dict(opens=4, typed=(PHRASE,)), "Could not try the passphrase on the key file"),
    )
    for name, kwargs, reason in cases:
        did = _ui(tmp_path / name, monkeypatch, **kwargs)
        said = capsys.readouterr().out
        assert did.code == 1, name
        assert reason in said and "Nothing was started." in said, name
        assert did.popen == [] and did.browser == [], name
        assert did.left_in_environ == [], name
    # Interrupted while it asks: 130, and nothing was started.
    did = _ui(tmp_path / "interrupted", monkeypatch, typed=("^C",))
    said = capsys.readouterr().out
    assert did.code == 130 and "Nothing was started." in said
    assert did.popen == [] and did.browser == []
    # A typed byte the terminal's encoding cannot carry opens nothing, is never shown, and the next try is asked.
    undecodable = UnicodeDecodeError("utf-8", b"\xe9", 0, 1, "invalid continuation byte")
    did = _ui(tmp_path / "undecodable", monkeypatch, typed=(undecodable, PHRASE))
    said = capsys.readouterr().out
    assert "That passphrase does not open the key file." in said and "0xe9" not in said
    assert did.asked == 2 and did.backend["env"].get(VAR) == PHRASE


def test_ul10_the_python_launcher_shows_why_the_backend_stopped(tmp_path, monkeypatch, capsys):
    did = _ui(tmp_path / "dies", monkeypatch, given=PHRASE, backend="dies", lingers=True)
    said = capsys.readouterr().out
    assert did.code == 1
    assert "The backend stopped before it answered (exit status 3)" in said
    assert "The frontend was not started." in said
    assert did.frontend is None and did.browser == []
    assert did.backend.get("stderr") not in (subprocess.DEVNULL, subprocess.PIPE)
    assert did.probes <= 1
    # What its group left is stopped too, before the message.
    assert [(pid, sig) for pid, sig, _when in did.groups] == [(did.backend_proc.pid, signal.SIGTERM)]
    assert said.index("What the backend left stopped.") < said.index("The backend stopped before it answered")
    # A slow backend is waited for at its own address, said every 30 s, never given up on; once it stops, that is
    # said too, the frontend's group stopped, and the launcher fails.
    did = _ui(tmp_path / "slow", monkeypatch, given=PHRASE, silent_polls=61)
    said = capsys.readouterr().out
    assert "Still waiting for the backend (30 s)" in said and "Still waiting for the backend (60 s)" in said
    assert said.count("Still waiting for the backend") == 2
    assert did.frontend is not None and did.backend["env"].get(VAR) == PHRASE
    assert "The backend stopped (exit status 9)" in said
    assert did.code == 1 and [(pid, sig) for pid, sig, _when in did.groups] == [
        (did.frontend_proc.pid, signal.SIGTERM)]
    assert said.index("Frontend stopped.") < said.index("The backend stopped (exit status 9)")
    # One ended by a signal is said so, never as a negative status; one that exited is said with its status.
    did = _ui(tmp_path / "killed", monkeypatch, given=PHRASE, backend_end=-9)
    said = capsys.readouterr().out
    assert "The backend stopped (ended by signal 9)" in said and "status -9" not in said
    did = _ui(tmp_path / "exited", monkeypatch, given=PHRASE, backend_end=0)
    assert "The backend stopped (exit status 0)" in capsys.readouterr().out
    # A frontend that stops later, its group keeping a member: said with its status, the backend's group and what
    # the frontend's left both stopped, and the launcher fails.
    did = _ui(tmp_path / "frontend-stops", monkeypatch, given=PHRASE, frontend="stops", lingers=True)
    said = capsys.readouterr().out
    assert "The frontend stopped (exit status 5)" in said
    assert did.code == 1
    assert {(pid, sig) for pid, sig, _when in did.groups} == {(did.frontend_proc.pid, signal.SIGTERM),
                                                              (did.backend_proc.pid, signal.SIGTERM)}
    assert said.index("Backend stopped.") < said.index("The frontend stopped (exit status 5)")
    # An interruption, a request to end, a hangup: both groups stopped, the launcher ends with 128 and the signal.
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        did = _ui(tmp_path / f"signal-{sig}", monkeypatch, given=PHRASE, signal_at=sig)
        said = capsys.readouterr().out
        assert did.code == 128 + sig, sig
        assert {(pid, s) for pid, s, _when in did.groups} == {(did.frontend_proc.pid, signal.SIGTERM),
                                                              (did.backend_proc.pid, signal.SIGTERM)}, sig
        assert "Frontend stopped." in said and "Backend stopped." in said, sig
    # A group that does not end in time is killed after 15 s, the other told at once, and that is said.
    did = _ui(tmp_path / "stuck", monkeypatch, given=PHRASE, stuck=True, signal_at=signal.SIGTERM)
    said = capsys.readouterr().out
    told = {(pid, sig): when for pid, sig, when in did.groups}
    frontend_pid, backend_pid = did.frontend_proc.pid, did.backend_proc.pid
    assert told[(backend_pid, signal.SIGTERM)] == told[(frontend_pid, signal.SIGTERM)]
    assert told[(frontend_pid, signal.SIGKILL)] - told[(frontend_pid, signal.SIGTERM)] >= 15
    assert (backend_pid, signal.SIGKILL) not in told
    assert "Frontend did not end within 15 s of being told to: killed." in said and "Backend stopped." in said
    assert did.code == 143
    # A group whose leader ended on TERM while it kept a member deaf to it is killed too, after 15 s, and said.
    did = _ui(tmp_path / "deaf-child", monkeypatch, given=PHRASE, deaf_child=True, signal_at=signal.SIGTERM)
    said = capsys.readouterr().out
    told = {(pid, sig): when for pid, sig, when in did.groups}
    assert told[(did.frontend_proc.pid, signal.SIGKILL)] - told[(did.frontend_proc.pid, signal.SIGTERM)] >= 15
    assert "Frontend did not end within 15 s of being told to: killed." in said
    # A group this launcher may not signal still holds something: it is never taken for ended.
    denied = _Proc([], None)
    denied.group = "denied"
    did.procs[denied.pid] = denied
    assert did.ui._group_left(denied.pid) is True
    # No other signal is heard while it stops -- a closing window's second hangup included --: one stop, the first
    # signal's status.
    did = _ui(tmp_path / "again", monkeypatch, given=PHRASE, stuck=True, signal_at=signal.SIGTERM,
              again=(signal.SIGHUP, signal.SIGINT, signal.SIGTERM))
    said = capsys.readouterr().out
    assert did.again and did.code == 143 and said.count("Stopping Opti-Oignon...") == 1
    # Its output gone when the signal comes -- a closed window's terminal, a pipe whose reader ended --: everything
    # is stopped all the same, and the launcher ends with its status.
    for sig, output in ((signal.SIGHUP, "tty"), (signal.SIGINT, "pipe")):
        did = _ui(tmp_path / f"gone-{output}", monkeypatch, given=PHRASE, signal_at=sig, output=output)
        assert did.code == 128 + sig and did.flushed, output
        assert {(pid, s) for pid, s, _when in did.groups} == {(did.frontend_proc.pid, signal.SIGTERM),
                                                              (did.backend_proc.pid, signal.SIGTERM)}, output
    capsys.readouterr()
    # Its output in use by what the signal interrupted: the stop goes on without its lines, with the signal's status.
    did = _ui(tmp_path / "busy", monkeypatch, given=PHRASE, signal_at=signal.SIGTERM, output="busy")
    assert did.code == 143
    assert {(pid, s) for pid, s, _when in did.groups} == {(did.frontend_proc.pid, signal.SIGTERM),
                                                          (did.backend_proc.pid, signal.SIGTERM)}
    # A signal that comes while a child is being started waits until it is written down, then stops it.
    did = _ui(tmp_path / "starting", monkeypatch, given=PHRASE, in_popen={"uvicorn": signal.SIGINT})
    said = capsys.readouterr().out
    assert did.code == 130 and "Nothing was started." not in said and did.frontend is None
    assert [(pid, sig) for pid, sig, _when in did.groups] == [(did.backend_proc.pid, signal.SIGTERM)]
    did = _ui(tmp_path / "starting-frontend", monkeypatch, given=PHRASE, in_popen={"npm": signal.SIGTERM})
    capsys.readouterr()
    assert did.code == 143 and did.browser == []
    assert {(pid, sig) for pid, sig, _when in did.groups} == {(did.frontend_proc.pid, signal.SIGTERM),
                                                              (did.backend_proc.pid, signal.SIGTERM)}
    # Anything else that ends the launcher -- a frontend that cannot start, whatever the error -- stops what was
    # started first, then says why.
    for frontend, error in (("missing", "FileNotFoundError"), ("refused", "SubprocessError")):
        did = _ui(tmp_path / f"frontend-{frontend}", monkeypatch, given=PHRASE, frontend=frontend)
        said = capsys.readouterr().out
        assert did.code == 1 and [(pid, sig) for pid, sig, _when in did.groups] == [
            (did.backend_proc.pid, signal.SIGTERM)], frontend
        assert said.index("Backend stopped.") < said.index(f"[ERR] {error}"), frontend
