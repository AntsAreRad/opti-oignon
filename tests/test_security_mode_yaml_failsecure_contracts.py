#!/usr/bin/env python3
"""Contracts for fail-secure reading of the security mode from YAML.

The current security mode is read from ``security.yaml``. The reader must
never hand a caller a value that is not a recognized mode: a hand-edited
file carrying an arbitrary string, and a corrupt or unreadable file, both
have to resolve to the restrictive interpretation (Bulbe) rather than
propagate a bogus value or silently fall to the permissive Daily mode.
A missing file is a different case -- it is the documented default state
of a fresh install, so it stays Daily by design.

  * Contract 1 -- a recognized mode is returned verbatim (the guard does
    not over-correct): a file naming ``daily`` yields ``daily`` and a
    file naming ``bulbe`` yields ``bulbe``.
  * Contract 2 -- an unrecognized mode string resolves to ``bulbe``: an
    arbitrary value is never propagated to a caller.
  * Contract 3 -- an unreadable or malformed file resolves to ``bulbe``:
    the exception path fails secure, not to Daily.
  * Contract 4 -- a missing file stays ``daily``: the documented default
    of a fresh install is preserved (this is not the unknown-mode case).

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. The module is loaded in isolation under a stub
package and the config path is redirected to a temporary file, so the
real security.yaml is never read.
"""

import importlib.util
import sys
import tempfile
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_MOD_NAME = "opti_oignon.security_mode"


def _load_security_mode():
    """Load the security_mode module in isolation under a stub package."""
    saved = {
        name: sys.modules.get(name)
        for name in ("opti_oignon", _MOD_NAME)
    }
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        _MOD_NAME, _OO / "security_mode.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod, saved


def _restore(saved):
    sys.modules.pop(_MOD_NAME, None)
    for name, value in saved.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def _with_yaml(mod, text):
    """Point the module's config path at a temp file holding ``text``."""
    fd = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fd.write(text)
    fd.close()
    path = Path(fd.name)
    mod._SECURITY_YAML = path
    return path


def test_c1_recognized_mode_is_returned_verbatim():
    mod, saved = _load_security_mode()
    paths = []
    try:
        for name in (mod.MODE_DAILY, mod.MODE_BULBE):
            p = _with_yaml(mod, f"security_mode: {name}\n")
            paths.append(p)
            got = mod._read_yaml_mode()
            assert got == name, (
                f"a recognized mode {name!r} must be returned as-is, got {got!r}"
            )
    finally:
        for p in paths:
            p.unlink(missing_ok=True)
        _restore(saved)


def test_c2_unrecognized_mode_resolves_to_bulbe():
    mod, saved = _load_security_mode()
    paths = []
    try:
        for bogus in ("NOT_A_MODE", "Daily", "bulbe ", "", "admin"):
            p = _with_yaml(mod, f"security_mode: {bogus!r}\n")
            paths.append(p)
            got = mod._read_yaml_mode()
            assert got == mod.MODE_BULBE, (
                f"an unrecognized mode {bogus!r} must fail secure to "
                f"{mod.MODE_BULBE!r}, got {got!r}"
            )
            assert got in mod.VALID_MODES, (
                "the reader must never return a value outside VALID_MODES"
            )
    finally:
        for p in paths:
            p.unlink(missing_ok=True)
        _restore(saved)


def test_c3_malformed_file_resolves_to_bulbe():
    mod, saved = _load_security_mode()
    paths = []
    try:
        # A YAML flow sequence left unclosed makes safe_load raise.
        p = _with_yaml(mod, "security_mode: [unclosed\n")
        paths.append(p)
        got = mod._read_yaml_mode()
        assert got == mod.MODE_BULBE, (
            f"a malformed file must fail secure to {mod.MODE_BULBE!r}, "
            f"got {got!r}"
        )
    finally:
        for p in paths:
            p.unlink(missing_ok=True)
        _restore(saved)


def test_c4_missing_file_stays_daily():
    mod, saved = _load_security_mode()
    try:
        # Point at a path that does not exist.
        with tempfile.TemporaryDirectory() as td:
            mod._SECURITY_YAML = Path(td) / "does_not_exist.yaml"
            got = mod._read_yaml_mode()
        assert got == mod.MODE_DAILY, (
            "a missing config file is the documented default and must stay "
            f"{mod.MODE_DAILY!r}, got {got!r}"
        )
    finally:
        _restore(saved)


_TESTS = [
    test_c1_recognized_mode_is_returned_verbatim,
    test_c2_unrecognized_mode_resolves_to_bulbe,
    test_c3_malformed_file_resolves_to_bulbe,
    test_c4_missing_file_stays_daily,
]


def _main():
    passed = 0
    for test in _TESTS:
        try:
            test()
        except Exception:
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__}")
            passed += 1
    print(f"{passed}/{len(_TESTS)} passed")
    return 0 if passed == len(_TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
