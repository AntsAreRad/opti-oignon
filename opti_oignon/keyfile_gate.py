#!/usr/bin/env python3
"""The launchers' gate to the key file: whether it is wrapped, and whether a passphrase opens it.

``python3 -m opti_oignon.keyfile_gate wrapped`` exits 0 when the platform's key file (``data/.keyfile``) is
wrapped under a passphrase, 3 when it is not -- no key file, or one in the legacy unprotected format --, and 4
when it cannot tell, saying why on stderr.

``python3 -m opti_oignon.keyfile_gate opens`` reads one line on its standard input, the passphrase -- never an
argument nor a variable --, as bytes decoded the way Python decodes the environment the backend reads its
passphrase from, and exits 0 when it opens the key file to a 32-byte key, 3 when the key file's envelope, well
formed, does not open under it, and 4 when it cannot try -- a damaged envelope, a key derivation this Python cannot
run --, saying why. A byte changed for another in the envelope's salt, or in its blob past the version byte, fails
as a wrong passphrase does, and is answered 3: nothing can tell the two apart. An empty passphrase opens nothing: the
key file's loader would take OPTI_KEYFILE_PASSPHRASE
from the environment instead, and the gate answers for the passphrase it was given alone; nor does one that UTF-8
cannot carry, under which no key file was ever wrapped.

1 and 2 are never answers: they are what Python exits with when it fails before answering -- a module it cannot
find, an error nobody caught --, and a launcher reads every status but 0 and 3 as "cannot tell". Nothing the gate
writes holds the passphrase. ``scripts/launch.sh`` runs the gate; ``opti_oignon/ui.py`` calls ``wrapped`` and
``opens``, whose every failure is a ``GateError``.
"""

from __future__ import annotations

import os
import sys

YES, NO, CANNOT = 0, 3, 4
_KEY_BYTES = 32
_PASSPHRASE_VAR = "OPTI_KEYFILE_PASSPHRASE"
_USAGE = "usage: python3 -m opti_oignon.keyfile_gate wrapped|opens (the passphrase on the standard input)"


class GateError(Exception):
    """The gate cannot answer: the key file cannot be read, is not wrapped, or opens to no 32-byte key."""


def _keyfile(root):
    """The key file asked about: the platform's own, or that of the project whose root is ``root``."""
    from opti_oignon import encryption

    return None if root is None else encryption.keyfile_under(root)


def wrapped(root=None) -> bool:
    """Whether the key file -- the platform's, or that of the project at ``root`` -- is wrapped under a passphrase;
    ``GateError`` when it cannot tell."""
    try:
        from opti_oignon import encryption

        return encryption.keyfile_wrapped(_keyfile(root))
    except Exception as exc:  # noqa: BLE001 - whatever stops the answer is "cannot tell", never "no"
        raise GateError(f"the key file cannot be read, or is in no format the platform knows: {exc}") from exc


def opens(passphrase: str, root=None) -> bool:
    """Whether ``passphrase`` opens the key file -- the platform's, or that of the project at ``root`` -- to a
    32-byte key.

    False for an empty passphrase and for one the key file's envelope does not open under; ``GateError`` when the
    key file is not there, is not wrapped, cannot be read or parsed, or opens to anything but a 32-byte key.
    """
    if not passphrase:
        return False
    try:
        passphrase.encode("utf-8")
    except UnicodeError:
        return False
    if not wrapped(root):
        raise GateError("the key file is not wrapped under a passphrase")
    from opti_oignon import encryption

    try:
        key, _salt, _kdf = encryption.load_keyfile(_keyfile(root), passphrase=passphrase)
    except encryption.PassphraseRefused:
        return False
    except Exception as exc:  # noqa: BLE001 - only a refused passphrase is "no"; anything else, "cannot try"
        raise GateError(f"the key file cannot be opened here: {type(exc).__name__}: {exc}") from exc
    try:
        size = len(key)
    finally:
        if hasattr(key, "wipe"):
            key.wipe()
    if size != _KEY_BYTES:
        raise GateError(f"the key file opens to {size} bytes, not to a {_KEY_BYTES}-byte key")
    return True


def main(argv: list[str] | None = None) -> int:
    """``wrapped`` or ``opens``: the exit status is the answer (see the module's docstring)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in (["wrapped"], ["opens"]):
        print(_USAGE, file=sys.stderr)
        return CANNOT
    # The gate answers for the passphrase on its input alone, never for one in its environment.
    from opti_oignon.process_secrets import withdraw

    withdraw(_PASSPHRASE_VAR)
    try:
        if args == ["wrapped"]:
            return YES if wrapped() else NO
        line = sys.stdin.buffer.readline()
        return YES if opens(os.fsdecode(line[:-1] if line.endswith(b"\n") else line)) else NO
    except Exception as exc:  # noqa: BLE001 - a GateError, or anything that stopped the answer
        print(f"keyfile_gate: {exc}", file=sys.stderr)
        return CANNOT


if __name__ == "__main__":
    sys.exit(main())
