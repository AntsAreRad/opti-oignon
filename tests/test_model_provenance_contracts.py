#!/usr/bin/env python3
"""Model provenance: the hash-pin contracts.

``model_provenance.py`` is the module that decides whether the bytes about to
be handed to the native GGUF parser are the bytes we pinned. The path guard in
``inference_backend`` proves WHERE a model file is; the SSRF guard in
``model_manager`` proves WHERE it came from. This module is the only thing
that proves WHAT it contains, so its failure modes are the interesting part:
every outcome that is not a verified pin must refuse under enforcement, and no
configuration may weaken the fortress mode.

The module is split into three surfaces on purpose, and the split is what
makes each contract independently probeable:

  * POLICY (``enforcement_mode``) -- mode plus configuration to enforce/warn.
  * CLASSIFICATION (``classify_model``) -- pure; a file plus a manifest to a
    reason. No policy, no raise.
  * VERDICT (``decide``) -- reason plus enforcement to allowed/refused.

Contracts pinned:

  Digest and canonical bytes:
    * A1 the digest is a streaming sha256 and does not depend on chunk size --
      the pin has to be portable between the writer and the reader.
    * A2 the canonical byte recipe is key-order independent, so a seal survives
      any JSON round-trip that reorders keys (json.load will).

  Policy (``enforcement_mode``):
    * A3 any mode that is not exactly "daily" enforces -- Bulbe, but also a
      mode that is empty, misspelled or invented. The inequality is the
      fail-secure hinge; enumerating the modes that enforce would leave the
      unknown ones failing open.
    * A4 Bulbe enforces even when configuration asks for warn: no config key
      can weaken the fortress.
    * A5 (control) Daily with default configuration observes without blocking,
      so an installation whose models are not yet enrolled keeps working.

  Classification (``classify_model``):
    * A6 a digest mismatch is named as such -- the substituted-weights case.
    * A7 a model absent from the manifest is unpinned, never verified.
    * A8 a seal presenting a WEAKER scheme than the host requires is refused
      before any cryptographic work happens. Accepting whichever scheme the
      artefact happens to name is the cryptographic-agility footgun: it lets
      an attacker holding only the weaker secret strip the stronger seal.
    * A9 a tampered manifest breaks its seal -- the seal covers the entries.
    * A10 an unreadable model file refuses; it never falls through to a pass.
    * A15 no key material at all refuses; it never falls through to a pass.

  Verdict (``decide``) and the gate:
    * A11 under enforcement exactly one reason passes: a verified pin.
    * A12 under observation nothing blocks.
    * A13 ``guard_model_load`` RAISES on a refused decision -- without the
      raise, a refusal is a log line and the load proceeds anyway.

  Round trip:
    * A14 a model enrolled by ``record_model`` classifies as verified: the
      writer and the reader agree on the pin.

Isolation follows the house idiom: canonical dotted names, an empty-path
package stand-in, and a meta-path guard sealing the window. The module's
top-level imports are stdlib only, so nothing needs seeding; its lazy imports
of the keyring, the signature backend and the security mode are all either
injected or deliberately left to fail, which is itself the fail-secure path
under test.

Local-only. Runs under pytest or the __main__ runner.
"""

import hashlib
import importlib.util
import json
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the test's
    back -- silently importing live code. This guard sits ahead of every
    finder and refuses the names that were not seeded, so a load behaves
    identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


# The module's lazy imports. They must be UNRESOLVABLE inside the window: the
# fail-secure paths under test are precisely what happens when the keyring, the
# signature backend or the security mode cannot be reached.
_BLOCKED = (
    "opti_oignon.security_mode",
    "opti_oignon.pqc_signatures",
    "opti_oignon.encryption",
)


def _load():
    """Load model_provenance in isolation; returns (module, restore).

    The meta-path guard alone does not seal the window. Python consults
    sys.modules BEFORE any finder, so a module some earlier test already
    imported for real resolves straight out of the cache and the guard is
    never asked. Blocking each name with None closes that: an import of a
    None-valued key raises ImportError before the finders run at all. Without
    it this suite passes in isolation and lies inside a full run -- the mode
    would resolve to whatever a previous test left behind.
    """
    keys = ("opti_oignon", "opti_oignon.model_provenance") + _BLOCKED
    saved = {k: sys.modules.get(k) for k in keys}

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    for name in _BLOCKED:
        sys.modules[name] = None

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.model_provenance", _OO / "model_provenance.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.model_provenance"] = mod
    spec.loader.exec_module(mod)
    pkg.model_provenance = mod

    def restore():
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


# ---------------------------------------------------------------------------
# Fixtures: a model file, a key, a sealed manifest. Everything injected --
# no ambient keyfile, no ambient security mode, no ambient manifest is ever
# consulted, so the suite is deterministic on any host.
# ---------------------------------------------------------------------------

_KEY = b"k" * 32
_BODY = b"GGUF" + bytes(range(256)) * 12
_NAME = "pinned-model-Q4_K_M.gguf"


def _hmac_keys(mod):
    return mod.SealKeys(
        scheme=mod.SCHEME_HMAC, sign_key=_KEY, verify_key=_KEY
    )


def _write_model(tmp: Path, body: bytes = _BODY, name: str = _NAME) -> Path:
    path = tmp / name
    path.write_bytes(body)
    return path


def _sealed_manifest(mod, entries: dict) -> dict:
    """A manifest whose seal genuinely covers its entries."""
    payload = {"version": mod.MANIFEST_VERSION, "entries": entries}
    manifest = dict(payload)
    manifest["seal"] = mod.compute_seal(payload, _hmac_keys(mod))
    return manifest


def _entry_for(body: bytes = _BODY) -> dict:
    return {
        "sha256": hashlib.sha256(body).hexdigest(),
        "size": len(body),
        "recorded_at": 1.0,
    }


# ---------------------------------------------------------------------------
# A1-A2: the primitives everything else stands on
# ---------------------------------------------------------------------------


def test_a1_digest_is_streaming_and_chunk_independent(tmp_path):
    mod, restore = _load()
    try:
        path = _write_model(tmp_path)
        expected = hashlib.sha256(_BODY).hexdigest()

        # A tiny chunk forces many iterations of the read loop; a chunk larger
        # than the file forces exactly one. Both must agree with the whole-file
        # digest, or the pin written by one side would not match the other.
        assert mod.compute_digest(path, chunk_size=7) == expected
        assert mod.compute_digest(path, chunk_size=1 << 20) == expected
        assert mod.compute_digest(path, chunk_size=len(_BODY)) == expected
    finally:
        restore()


def test_a2_canonical_bytes_are_key_order_independent(tmp_path):
    mod, restore = _load()
    try:
        one = {"version": 1, "entries": {"b": 2, "a": 1}}
        two = {"entries": {"a": 1, "b": 2}, "version": 1}

        # The two dicts are the same object logically and differ only in the
        # order their keys were inserted. json.load will happily hand back
        # either order, so a seal computed over a non-canonical serialisation
        # would break for a reason that has nothing to do with tampering.
        assert mod.canonical_bytes(one) == mod.canonical_bytes(two)

        # And the seal must therefore survive a real JSON round trip.
        keys = _hmac_keys(mod)
        seal = mod.compute_seal(one, keys)
        reordered = json.loads(json.dumps(two, sort_keys=False))
        assert mod.verify_seal(reordered, seal, keys) == mod.REASON_VERIFIED
    finally:
        restore()


# ---------------------------------------------------------------------------
# A3-A5: policy. Mode decides; configuration may never weaken it.
# ---------------------------------------------------------------------------


def test_a3_unnameable_mode_enforces(tmp_path):
    mod, restore = _load()
    try:
        # An invented mode, an empty one, and a misspelling of the fortress:
        # none of them is "daily", so all of them enforce. This is the point of
        # writing the test as an inequality -- a whitelist of enforcing modes
        # would let exactly these cases through.
        for mode in ("chaos", "", "bulb", "BULBE_TYPO", "unknown"):
            assert mod.enforcement_mode(mode=mode, config={}) == mod.ENFORCE

        # And a mode that cannot be resolved at all is Bulbe: the security
        # module is unreachable inside the isolation window, which is exactly
        # the shape of a broken or partial installation.
        assert mod.current_mode() == mod.MODE_BULBE
    finally:
        restore()


def test_a4_bulbe_enforces_against_a_config_asking_for_warn(tmp_path):
    mod, restore = _load()
    try:
        asked = {"enforcement": "warn"}
        assert (
            mod.enforcement_mode(mode=mod.MODE_BULBE, config=asked)
            == mod.ENFORCE
        )
    finally:
        restore()


def test_a5_daily_default_observes_without_blocking(tmp_path):
    mod, restore = _load()
    try:
        assert mod.enforcement_mode(mode=mod.MODE_DAILY, config={}) == mod.WARN
        assert (
            mod.enforcement_mode(
                mode=mod.MODE_DAILY, config={"enforcement": "enforce"}
            )
            == mod.ENFORCE
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# A6-A10, A15: classification. Pure -- no policy is applied here.
# ---------------------------------------------------------------------------


def test_a6_digest_mismatch_is_named(tmp_path):
    mod, restore = _load()
    try:
        path = _write_model(tmp_path)
        # The manifest pins a digest that is not this file's: the substituted
        # weights case, and the whole reason the module exists.
        entry = _entry_for(b"the bytes we actually pinned")
        manifest = _sealed_manifest(mod, {_NAME: entry})

        reason, digest = mod.classify_model(
            path, manifest, keys=_hmac_keys(mod)
        )
        assert reason == mod.REASON_DIGEST_MISMATCH
        # The refused decision still reports the digest it actually computed,
        # so an audit record can say WHAT it saw. Whether that primitive is a
        # true sha256 is clause A1's business, and pinning it again here would
        # only make a probe on A1 redden this clause too.
        assert digest is not None and len(digest) == 64
    finally:
        restore()


def test_a7_absent_model_is_unpinned_never_verified(tmp_path):
    mod, restore = _load()
    try:
        path = _write_model(tmp_path)
        manifest = _sealed_manifest(mod, {"some-other-model.gguf": _entry_for()})

        reason, _ = mod.classify_model(path, manifest, keys=_hmac_keys(mod))
        assert reason == mod.REASON_UNPINNED
    finally:
        restore()


def test_a8_seal_scheme_downgrade_is_refused(tmp_path):
    mod, restore = _load()
    try:
        path = _write_model(tmp_path)
        manifest = _sealed_manifest(mod, {_NAME: _entry_for()})

        # The host requires ML-DSA-65. The manifest carries an HMAC seal that
        # would verify perfectly well against the HMAC secret -- and that is
        # exactly why it must be refused: an attacker holding only the weaker
        # secret must not be able to strip the stronger seal and substitute one
        # it can forge. The scheme is checked before any crypto runs, so the
        # verifier below is never even reached.
        pqc_keys = mod.SealKeys(
            scheme=mod.SCHEME_PQC, sign_key=b"priv", verify_key=b"pub"
        )

        def _never_called(data, sig, pub):
            raise AssertionError("crypto ran before the scheme was checked")

        reason, _ = mod.classify_model(
            path, manifest, keys=pqc_keys, verifier=_never_called
        )
        assert reason == mod.REASON_SEAL_DOWNGRADE
    finally:
        restore()


def test_a9_tampered_manifest_breaks_its_seal(tmp_path):
    mod, restore = _load()
    try:
        path = _write_model(tmp_path)
        manifest = _sealed_manifest(mod, {_NAME: _entry_for()})

        # Re-pin the model to the digest of the substituted bytes, leaving the
        # seal untouched. This is the attack the seal exists to stop: without
        # it, rewriting the pin would be enough to make any file "verified".
        manifest["entries"][_NAME]["sha256"] = hashlib.sha256(b"evil").hexdigest()

        reason, _ = mod.classify_model(path, manifest, keys=_hmac_keys(mod))
        assert reason == mod.REASON_SEAL_INVALID
    finally:
        restore()


def test_a10_unreadable_model_refuses_rather_than_passing(tmp_path):
    mod, restore = _load()
    try:
        missing = tmp_path / _NAME  # pinned, but never written to disk
        manifest = _sealed_manifest(mod, {_NAME: _entry_for()})

        reason, digest = mod.classify_model(
            missing, manifest, keys=_hmac_keys(mod)
        )
        assert reason == mod.REASON_FILE_UNREADABLE
        assert digest is None
    finally:
        restore()


def test_a15_absent_key_material_refuses_rather_than_passing(tmp_path):
    mod, restore = _load()
    try:
        path = _write_model(tmp_path)
        manifest = _sealed_manifest(mod, {_NAME: _entry_for()})

        # No keyring, no keyfile: the seal cannot be checked at all. An
        # unverifiable seal is not a verified one.
        mod.resolve_seal_keys = lambda: None

        reason, _ = mod.classify_model(path, manifest, keys=None)
        assert reason == mod.REASON_KEY_UNAVAILABLE
    finally:
        restore()


# ---------------------------------------------------------------------------
# A11-A13: the verdict and the gate
# ---------------------------------------------------------------------------


def test_a11_under_enforcement_only_a_verified_pin_passes(tmp_path):
    mod, restore = _load()
    try:
        refusing = (
            mod.REASON_DIGEST_MISMATCH,
            mod.REASON_UNPINNED,
            mod.REASON_SEAL_INVALID,
            mod.REASON_SEAL_DOWNGRADE,
            mod.REASON_SEAL_MISSING,
            mod.REASON_MANIFEST_MISSING,
            mod.REASON_MANIFEST_UNREADABLE,
            mod.REASON_KEY_UNAVAILABLE,
            mod.REASON_FILE_UNREADABLE,
        )
        for reason in refusing:
            assert mod.decide(reason, mod.ENFORCE) is False
        assert mod.decide(mod.REASON_VERIFIED, mod.ENFORCE) is True
    finally:
        restore()


def test_a12_under_observation_nothing_blocks(tmp_path):
    mod, restore = _load()
    try:
        for reason in (
            mod.REASON_DIGEST_MISMATCH,
            mod.REASON_UNPINNED,
            mod.REASON_SEAL_INVALID,
            mod.REASON_MANIFEST_MISSING,
            mod.REASON_VERIFIED,
        ):
            assert mod.decide(reason, mod.WARN) is True
    finally:
        restore()


def test_a13_guard_raises_on_a_refused_decision(tmp_path):
    mod, restore = _load()
    try:
        refused = mod.ProvenanceDecision(
            allowed=False,
            reason=mod.REASON_DIGEST_MISMATCH,
            enforcement=mod.ENFORCE,
            model=_NAME,
        )
        allowed = mod.ProvenanceDecision(
            allowed=True,
            reason=mod.REASON_VERIFIED,
            enforcement=mod.ENFORCE,
            model=_NAME,
        )

        # The decision is injected, so this clause depends on the raise and on
        # nothing else. Without the raise a refusal is only a log line and the
        # load proceeds regardless -- which is the failure this pins shut.
        mod.verify_model = lambda *a, **k: refused
        try:
            mod.guard_model_load(tmp_path / _NAME)
        except mod.ProvenanceRefusal as exc:
            assert exc.decision.reason == mod.REASON_DIGEST_MISMATCH
        else:
            raise AssertionError("a refused decision did not raise")

        mod.verify_model = lambda *a, **k: allowed
        assert mod.guard_model_load(tmp_path / _NAME).allowed is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# A14: the writer and the reader agree
# ---------------------------------------------------------------------------


def test_a14_recorded_model_classifies_as_verified(tmp_path):
    mod, restore = _load()
    try:
        path = _write_model(tmp_path)
        manifest_path = tmp_path / "provenance.json"

        # Signer and verifier are injected and ignore the data they are handed,
        # so this clause exercises the enrolment plumbing -- the digest that is
        # written and the name it is filed under -- and NOT the canonical byte
        # recipe, which is clause A2's business. Keeping them apart is what
        # lets a probe on either one redden exactly one clause.
        pqc_keys = mod.SealKeys(
            scheme=mod.SCHEME_PQC, sign_key=b"priv", verify_key=b"pub"
        )
        signer = lambda data, key: b"SEALED"  # noqa: E731
        verifier = lambda data, sig, pub: sig == b"SEALED"  # noqa: E731

        expected = mod.compute_digest(path)
        recorded = mod.record_model(
            path, manifest_path=manifest_path, keys=pqc_keys, signer=signer
        )
        assert recorded["sha256"] == expected

        written = mod.load_manifest(manifest_path)
        reason, digest = mod.classify_model(
            path, written, keys=pqc_keys, verifier=verifier
        )
        assert reason == mod.REASON_VERIFIED
        assert digest == expected
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _main() -> int:
    import tempfile

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"PASS {fn.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {fn.__name__}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
