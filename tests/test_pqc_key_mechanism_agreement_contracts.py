"""A keypair carries the algorithm that minted it. The loader must READ it.

The keypair file records its own provenance:

    {"algorithm": ..., "public_key": ..., "private_key": ...}

The loader parsed that field, wrote it to a debug log, and returned the bytes.
The module states the principle against itself sixty lines earlier -- a log line
is not a reason, because nothing above it can read a log -- and then does the
thing it forbids.

The consequence is not theoretical. The two mechanism names the estate accepts
are not two spellings of one algorithm: their keys are different lengths and
their signatures do not interoperate. A keypair minted before the standards body
renamed the primitive is minted under the OTHER name. Hand those bytes to the
resolved mechanism and the backend rejects them -- deep inside the signer, past
every posture guard, in the exact place a swallowed exception used to turn the
rejection into an unsigned document that the caller believed was signed.

So the agreement is checked where the bytes are read, and a disagreement is a
refusal. An unknown provenance is a disagreement too: a key that does not say
what minted it cannot be shown to match, and cannot-be-shown-to-match is the
only reading that is never wrong.

The last contract keeps a dead branch dead. The backup manager carries an
``except ImportError`` fallback whose refusal fires unconditionally, and it is
unreachable only because this module imports the standard library and nothing
else -- the backend binding and the YAML parser are both deferred into the
functions that need them. One import hoisted to the top of the file arms that
fallback, and a machine whose signing module is merely absent stops being able
to export a backup at all. The branch is dead on purpose. This pins it dead.
"""

import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_SOURCE = source("pqc_signatures.py")


def _load():
    loaded, restore = isolate(
        targets={"opti_oignon.pqc_signatures": _SOURCE},
    )
    return loaded["opti_oignon.pqc_signatures"], restore


def _write_keypair(tmp_path, *, algorithm, public=b"PUB", private=b"PRIV"):
    import base64

    payload = {
        "public_key": base64.urlsafe_b64encode(public).decode("ascii"),
        "private_key": base64.urlsafe_b64encode(private).decode("ascii"),
    }
    if algorithm is not None:
        payload["algorithm"] = algorithm

    fpath = tmp_path / ".pqc_keypair"
    fpath.write_text(json.dumps(payload), encoding="ascii")
    return fpath


def test_a1_a_key_minted_under_another_mechanism_is_refused(tmp_path):
    """A1 -- the file says one algorithm, the host resolved another: REFUSE.

    This is the live shape of the defect. The rename that killed the primitive
    is precisely what leaves a host holding a key from before it.
    """
    mod, restore = _load()
    try:
        mod._PQC_ALGORITHM = "ML-DSA-65"
        fpath = _write_keypair(tmp_path, algorithm="Dilithium3")
        with pytest.raises(ValueError):
            mod.load_pqc_keypair(fpath)
    finally:
        restore()


def test_a2_a_key_that_does_not_say_what_minted_it_is_refused(tmp_path):
    """A2 -- no algorithm field: unknown provenance is not agreement."""
    mod, restore = _load()
    try:
        mod._PQC_ALGORITHM = "ML-DSA-65"
        fpath = _write_keypair(tmp_path, algorithm=None)
        with pytest.raises(ValueError):
            mod.load_pqc_keypair(fpath)
    finally:
        restore()


def test_a3_an_agreeing_key_loads(tmp_path):
    """A3 -- the file and the resolved mechanism agree: the bytes come back.

    Guards the refusal against over-reach. A correct key must still load.
    """
    mod, restore = _load()
    try:
        mod._PQC_ALGORITHM = "ML-DSA-65"
        fpath = _write_keypair(
            tmp_path, algorithm="ML-DSA-65", public=b"PUB", private=b"PRIV"
        )
        public, private = mod.load_pqc_keypair(fpath)
        assert (public, private) == (b"PUB", b"PRIV")
    finally:
        restore()


def test_a4_no_mechanism_resolved_means_no_key_can_be_loaded(tmp_path):
    """A4 -- nothing resolved: there is no mechanism for a key to agree WITH."""
    mod, restore = _load()
    try:
        mod._PQC_ALGORITHM = None
        fpath = _write_keypair(tmp_path, algorithm="ML-DSA-65")
        with pytest.raises(ValueError):
            mod.load_pqc_keypair(fpath)
    finally:
        restore()


def test_a5_the_signing_module_imports_on_the_standard_library_alone():
    """A5 -- keeps the backup manager's ImportError fallback unreachable.

    That fallback refuses unconditionally: a machine whose signing module will
    not import can export no backup at all. It is a cliff, and it is safe only
    while nothing can push this module off it. Every third-party import here is
    deferred into the function that needs it, and that is a load-bearing
    property, not a style. One hoisted import turns a broken optional dependency
    into a machine that cannot back itself up.
    """
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    stdlib = sys.stdlib_module_names

    offenders = []
    for node in tree.body:  # module level only -- deferred imports are the point
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".", 1)[0]
            if root and root not in stdlib and root != "opti_oignon":
                offenders.append(name)

    assert not offenders, (
        "a top-level third-party import here arms the backup manager's "
        f"unconditional ImportError refusal: {offenders}"
    )


def test_a6_the_status_report_shows_a_disagreeing_key_before_it_bites(tmp_path):
    """A6 -- the dashboard surfaces a stale key rather than waiting for a refusal.

    The refusal above is correct and it is also LATE: it fires when the operator
    is already trying to export something. The agreement is knowable at rest, so
    it is reported at rest.
    """
    mod, restore = _load()
    try:
        mod._PQC_ALGORITHM = "ML-DSA-65"
        mod._DEFAULT_KEYPAIR_PATH = _write_keypair(tmp_path, algorithm="Dilithium3")
        status = mod.get_pqc_status()
        assert status["key_algorithm"] == "Dilithium3"
        assert status["key_algorithm_agrees"] is False
    finally:
        restore()
