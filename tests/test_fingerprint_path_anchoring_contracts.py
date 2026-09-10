#!/usr/bin/env python3
"""Contracts for anchoring the preference store's path.

The store's path was configured as a bare filename, so it resolved against
whatever directory the caller happened to be in. Importing the package from
the repository root wrote the database into the repository root; importing it
from a temporary directory wrote it there; importing it from a user's home
would write it there. The module anchors its own configuration file on
``__file__`` and then reads an unanchored data path out of it, which is the
shape of the mistake: half the work was already done.

That is one entry on the import footprint guard's ledger, and this is what
takes it off. The ledger may only shrink, so removing an entry is the only
legitimate direction and it has to be earned by the code first.

  * FA1 -- the resolved path is absolute.
  * FA2 -- it sits under the package's data directory, beside the other
    stores rather than beside whoever ran the program.
  * FA3 -- an absolute path in the configuration is honoured as given. An
    operator who names a location means it, and anchoring it again would
    silently move their data.
  * FA4 -- the guard's ledger no longer carries a file, which is the shrink
    this change is for.
  * FA5 -- importing the package writes nothing into the caller's working
    directory. Measured by running the import, not asserted about it.

The anchoring is applied where the CONFIGURATION is read, not inside the
store. A caller that hands the store a path means that path: two contracts in
this tree pass a relative name and expect it opened rather than relocated,
and they refused the first version of this change. They were right to -- an
anchor inside the store would have made every test that names its own
database write into the package's data directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GUARD = REPO / ".github" / "scripts" / "import_footprint_guard.py"
_TARGET = "opti_oignon.session_fingerprint"


def _module():
    """Load the module under contract through the shared window."""
    loaded, restore = isolate(targets={_TARGET: source("session_fingerprint.py")})
    return loaded[_TARGET], restore


def _guard():
    loaded, restore = isolate(targets={"_footprint": GUARD})
    return loaded["_footprint"], restore


def test_fa1_the_resolved_path_is_absolute():
    module, restore = _module()
    try:
        resolved = Path(module._anchor("fingerprint.db"))
        assert resolved.is_absolute(), (
            f"a bare filename resolves against the caller's directory: "
            f"{resolved}"
        )
    finally:
        restore()


def test_fa2_the_resolved_path_sits_in_the_package_data_directory():
    module, restore = _module()
    try:
        resolved = Path(module._anchor("fingerprint.db"))
        expected = Path(module.__file__).resolve().parent / "data"
        assert resolved.parent == expected, (
            f"expected the store beside the others in {expected}, got "
            f"{resolved.parent}"
        )
        assert resolved.name == "fingerprint.db"
    finally:
        restore()


def test_fa3_an_absolute_configured_path_is_honoured_as_given():
    module, restore = _module()
    try:
        chosen = "/var/lib/opti-oignon/elsewhere.db"
        assert module._anchor(chosen) == chosen, (
            "an operator who names a location means it; anchoring it again "
            "would move their data without saying so"
        )
    finally:
        restore()


def test_fa4_the_guard_ledger_no_longer_carries_a_file():
    guard, restore = _guard()
    try:
        assert guard.LEDGER["files"] == frozenset(), (
            "the debt this change pays must come off the ledger, or the "
            "count stops meaning anything"
        )
    finally:
        restore()


def test_fa5_the_import_writes_nothing_into_the_callers_directory():
    """Measured by running the import in a subprocess, not asserted about."""
    guard, restore = _guard()
    try:
        seen = guard.observe()
        assert seen is not None, (
            "the probe did not run, so nothing was measured; that is not the "
            "same as a clean working directory"
        )
        assert seen["files"] == [], (
            f"importing the package still writes into the caller's "
            f"directory: {seen['files']}"
        )
        # Proven capable: the probe does report the rest, so an empty file
        # list is a measurement rather than a probe that saw nothing.
        assert seen["modules"] > 0 and seen["databases"], seen
    finally:
        restore()
