#!/usr/bin/env python3
"""Contracts for the provenance writers: none carries an entry the manifest's seal does not vouch for here.

The load gate checks the seal before it reads an entry. The three writers did not: a download's pin
(``record_model``), an enrolment (``enroll_models``) and a re-seal (``reseal_manifest``) read the entries without
checking the seal and sealed them all again under the host's key -- an entry a process without the key wrote, or a
manifest whose seal was deleted, was vouched for at the next one. Each now reads through ``read_manifest``, which
gives entries only from bytes whose seal verifies here.

Old manifests are written here by the OLD recipe, inline, never by the module: the canonical bytes, an HMAC-SHA512
or a signature over them, ``json.dump`` sorted with an indent of two. An outside writer loads the JSON, edits it and
writes it back in the same layout: it cannot seal. The signing stand-in binds the whole data and the keypair, so a
tampered payload never verifies by accident.

  * ML1 -- A DOWNLOAD PIN CARRIES NO ENTRY AN OUTSIDE WRITER ADDED: ``seal_invalid``, nothing written.
  * ML2 -- AN ENROLMENT CARRIES NONE EITHER: ``seal_invalid``, nothing written.
  * ML3 -- A RE-SEAL NEVER SEALS A MANIFEST WHOSE SEAL WAS DELETED: ``seal_missing``, nothing written.
  * ML4 -- A RE-SEAL NEVER CALLS A FORGED SEAL OF THE HOST'S SCHEME SEALED: ``seal_invalid``, nothing written.
  * ML5 -- A MAC THAT VERIFIES IS LIFTED TO THE HOST'S SIGNATURE, ITS ENTRIES VERBATIM; one under another key is
    ``seal_invalid``, nothing written.
  * ML6 -- A SEAL UNDER A SCHEME THIS HOST HOLDS NO KEY FOR IS ``key_unavailable``, NAMING IT, for every writer;
    never a fresh start.
  * ML7 -- A MANIFEST THAT DOES NOT PARSE AS A JSON OBJECT IS ``manifest_unreadable`` FOR EVERY WRITER, NEVER
    OVERWRITTEN, NEVER CALLED ABSENT.
  * ML8 -- AN ENROLMENT NEVER RE-PINS A PINNED MODEL WHOSE BYTES CHANGED, UNLESS IT IS NAMED: ``digest_mismatch``
    naming it, nothing written, the gate still refusing it; ``repin`` re-pins it.
  * ML9 -- A PIN THAT STILL HOLDS IS KEPT VERBATIM, ONLY THE NEW IS PINNED, AND A RUN WITH NOTHING NEW WRITES
    NOTHING.
  * ML10 -- WHAT IS SEALED IS WHAT WAS VERIFIED: an entry written on disk after the check is not carried.
  * ML11 -- STARTING AGAIN IS EXPLICIT: the old file set aside, byte for byte, under a name git ignores; the new
    manifest holds exactly the models handed in; what was dropped said; refused on a manifest that verifies, on an
    absent one and on one that cannot be read.
  * ML12 -- ON THE REAL KEY FILE, A MANIFEST SEALED UNDER ITS OPENING BYTES (THE OLD FALLBACK) VOUCHES FOR NOTHING:
    refused with the passphrase and without it; started again, it is set aside and the model on disk pinned under
    the key that loads.
  * ML13 -- THE ENROLMENT SCRIPT REFUSES AN UNVERIFIED MANIFEST BY ITS REASON: exit 1, ``--start-again`` named,
    nothing written, "(already pinned)" never said of an entry that did not verify.
  * ML14 -- THE RE-SEAL SCRIPT NEVER SAYS "ALREADY SEALED" OF A SEAL IT DID NOT CHECK: exit 1, ``seal_invalid``.
  * ML15 -- THE LOAD GATE TAKES NONE OF THE WRITERS' LATITUDE: before the lift, a MAC manifest on a host that
    requires a signature still reads ``seal_downgrade`` at load.

Local-only (the public distribution ships no tests). Loaded through the shared isolation window; every manifest in
tmp, the module's default paths pointed at tmp as a net.
"""

import base64
import hashlib
import hmac
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

PROV = "opti_oignon.model_provenance"
BUDGET_S = {
    "test_ml1_a_download_pin_carries_no_entry_an_outside_writer_added": 2.0,
    "test_ml2_an_enrolment_carries_no_entry_an_outside_writer_added": 2.0,
    "test_ml3_a_reseal_never_seals_a_manifest_whose_seal_was_deleted": 2.0,
    "test_ml4_a_reseal_never_calls_a_forged_seal_of_the_hosts_scheme_sealed": 2.0,
    "test_ml5_a_mac_that_verifies_is_lifted_to_the_signature_and_one_under_another_key_refused": 2.0,
    "test_ml6_a_seal_this_host_holds_no_key_for_is_key_unavailable_never_a_fresh_start": 2.0,
    "test_ml7_a_manifest_that_does_not_parse_is_unreadable_never_overwritten_never_absent": 2.0,
    "test_ml8_an_enrolment_never_repins_a_changed_model_unless_it_is_named": 2.0,
    "test_ml9_a_pin_that_holds_is_kept_verbatim_and_nothing_new_writes_nothing": 2.0,
    "test_ml10_what_is_sealed_is_what_was_verified": 2.0,
    "test_ml11_starting_again_is_explicit_and_sets_the_old_file_aside_untouched": 4.0,
    "test_ml12_on_the_real_key_file_the_old_fallback_seal_vouches_for_nothing": 8.0,
    "test_ml13_the_enrolment_script_refuses_an_unverified_manifest_by_its_reason": 4.0,
    "test_ml14_the_reseal_script_never_says_already_sealed_of_a_seal_it_did_not_check": 4.0,
    "test_ml15_the_load_gate_takes_none_of_the_writers_latitude": 2.0,
}

# The key file's key in the stand-in worlds, and another 32-byte key.
_K = b"k" * 32
_OTHER = b"o" * 32
# The keypair of the signing stand-in.
_PUBLIC, _PRIVATE = b"PUBLIC-ML", b"PRIVATE-ML"
_PAIRS = {_PUBLIC: _PRIVATE}


def _sign(data, private):
    return hashlib.sha512(b"ML" + bytes(private) + data).digest()


def _verify(data, signature, public):
    private = _PAIRS.get(bytes(public))
    return private is not None and hmac.compare_digest(bytes(signature), _sign(data, private))


def _world(tmp_path, *, signature=False, keypair=True, key=_K, primitive=True):
    """model_provenance over stand-ins: a signing pair that binds the whole data and its keypair, required when
    ``signature``, checking nothing when the ``primitive`` is absent, as the real one does; a key file that loads
    ``key`` (None: it does not load); the default paths in tmp."""
    pqc = types.ModuleType("opti_oignon.pqc_signatures")
    pqc.PQC_AVAILABLE = primitive
    pqc.is_pqc_enabled = lambda: signature
    pqc.pqc_required = lambda: signature

    def load_pqc_keypair():
        if not keypair:
            raise FileNotFoundError("PQC keypair file not found")
        return _PUBLIC, _PRIVATE

    pqc.load_pqc_keypair = load_pqc_keypair
    pqc.sign_bytes = _sign
    pqc.verify_bytes = _verify if primitive else (lambda data, signature, public: False)
    encryption = types.ModuleType("opti_oignon.encryption")

    def load_keyfile():
        if key is None:
            raise ValueError("wrapped under a passphrase this process lacks")
        return key, b"salt", "kdf"

    encryption.load_keyfile = load_keyfile
    loaded, restore = isolate(targets={PROV: source("model_provenance.py")},
                              seeded={"opti_oignon.pqc_signatures": pqc, "opti_oignon.encryption": encryption})
    try:
        prov = loaded[PROV]
        data = tmp_path / "data"
        data.mkdir()
        prov._DATA_DIR, prov.DEFAULT_MANIFEST_PATH = data, data / "model_provenance.json"
    except BaseException:
        restore()
        raise
    return SimpleNamespace(prov=prov, restore=restore, data=data, manifest=data / "model_provenance.json")


# The old recipe, inline: never the module's own functions.
def _old_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _old_payload(entries):
    return {"version": 1, "entries": entries}


def _entry(path, recorded_at=1.0):
    body = path.read_bytes()
    return {"sha256": hashlib.sha256(body).hexdigest(), "size": len(body), "recorded_at": recorded_at}


def _old_hmac(payload, key):
    return {"scheme": "hmac-sha512", "value": hmac.new(key, _old_bytes(payload), hashlib.sha512).hexdigest()}


def _old_pqc(payload, private=_PRIVATE):
    return {"scheme": "mldsa65", "value": base64.b64encode(_sign(_old_bytes(payload), private)).decode("ascii")}


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return path.read_bytes()


def _old_write(path, payload, seal):
    obj = dict(payload)
    if seal is not None:
        obj["seal"] = seal
    return _dump(path, obj)


def _outside(path, change):
    """An outside writer: it loads the JSON, edits it and writes it back in the same layout; it cannot seal."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    change(obj)
    return _dump(path, obj)


def _evil(obj):
    obj["entries"]["evil.gguf"] = {"sha256": "e" * 64, "size": 4, "recorded_at": 2.0}


def _model(folder, name, body):
    path = folder / name
    path.write_bytes(body)
    return path


def _files(folder):
    return sorted(p.name for p in folder.iterdir())


def test_ml1_a_download_pin_carries_no_entry_an_outside_writer_added(tmp_path):
    w = _world(tmp_path)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        fresh = _model(tmp_path, "fresh.gguf", b"GGUF fresh")
        payload = _old_payload({"llama.gguf": _entry(llama)})
        _old_write(w.manifest, payload, _old_hmac(payload, _K))
        keys = prov.resolve_seal_keys()
        assert prov.read_manifest(w.manifest, keys=keys).status == prov.REASON_VERIFIED, "control: it verifies"
        before = _outside(w.manifest, _evil)
        with pytest.raises(prov.ManifestNotVerified) as refused:
            prov.record_model(fresh, manifest_path=w.manifest)
        assert refused.value.reason == prov.REASON_SEAL_INVALID and str(refused.value).startswith("seal_invalid:")
        assert "evil.gguf" in refused.value.names and refused.value.found == 2
        assert w.manifest.read_bytes() == before and _files(w.data) == ["model_provenance.json"], "nothing written"
    finally:
        w.restore()


def test_ml2_an_enrolment_carries_no_entry_an_outside_writer_added(tmp_path):
    w = _world(tmp_path)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        payload = _old_payload({"llama.gguf": _entry(llama)})
        _old_write(w.manifest, payload, _old_hmac(payload, _K))
        before = _outside(w.manifest, _evil)
        with pytest.raises(prov.ManifestNotVerified) as refused:
            prov.enroll_models([llama], manifest_path=w.manifest)
        assert refused.value.reason == prov.REASON_SEAL_INVALID
        assert "--start-again" in str(refused.value) and "nothing was written" in str(refused.value)
        assert w.manifest.read_bytes() == before and _files(w.data) == ["model_provenance.json"]
    finally:
        w.restore()


def test_ml3_a_reseal_never_seals_a_manifest_whose_seal_was_deleted(tmp_path):
    w = _world(tmp_path, signature=True)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        payload = _old_payload({"llama.gguf": _entry(llama)})
        _old_write(w.manifest, payload, _old_pqc(payload))
        before = _outside(w.manifest, lambda obj: obj.pop("seal"))
        with pytest.raises(prov.ManifestNotVerified) as refused:
            prov.reseal_manifest(manifest_path=w.manifest)
        assert refused.value.reason == prov.REASON_SEAL_MISSING
        assert w.manifest.read_bytes() == before and _files(w.data) == ["model_provenance.json"]
    finally:
        w.restore()


def test_ml4_a_reseal_never_calls_a_forged_seal_of_the_hosts_scheme_sealed(tmp_path):
    w = _world(tmp_path, signature=True)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        payload = _old_payload({"llama.gguf": _entry(llama)})
        forged = {"scheme": "mldsa65", "value": base64.b64encode(b"not a signature").decode("ascii")}
        before = _old_write(w.manifest, payload, forged)
        with pytest.raises(prov.ManifestNotVerified) as refused:
            prov.reseal_manifest(manifest_path=w.manifest)
        assert refused.value.reason == prov.REASON_SEAL_INVALID
        assert w.manifest.read_bytes() == before
        # Control: the same entries under a signature that verifies are left as they are, said.
        _old_write(w.manifest, payload, _old_pqc(payload))
        left = prov.reseal_manifest(manifest_path=w.manifest)
        assert left["written"] is False and left["previous_scheme"] == "mldsa65"
    finally:
        w.restore()


def test_ml5_a_mac_that_verifies_is_lifted_to_the_signature_and_one_under_another_key_refused(tmp_path):
    w = _world(tmp_path, signature=True)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        entries = {"llama.gguf": _entry(llama)}
        payload = _old_payload(entries)
        _old_write(w.manifest, payload, _old_hmac(payload, _K))
        lifted = prov.reseal_manifest(manifest_path=w.manifest)
        assert lifted == {"scheme": "mldsa65", "previous_scheme": "hmac-sha512", "entries": 1, "written": True}
        written = json.loads(w.manifest.read_text(encoding="utf-8"))
        assert written["entries"] == entries, "the entries verbatim"
        assert written["seal"]["scheme"] == "mldsa65"
        assert _verify(_old_bytes(_old_payload(entries)), base64.b64decode(written["seal"]["value"]), _PUBLIC)
        # The same entries under a MAC another 32-byte key made: refused, nothing written.
        before = _old_write(w.manifest, payload, _old_hmac(payload, _OTHER))
        with pytest.raises(prov.ManifestNotVerified) as refused:
            prov.reseal_manifest(manifest_path=w.manifest)
        assert refused.value.reason == prov.REASON_SEAL_INVALID and w.manifest.read_bytes() == before
        # An enrolment with nothing new lifts the MAC that verifies too: written, said, the entries verbatim.
        _old_write(w.manifest, payload, _old_hmac(payload, _K))
        moved = prov.enroll_models([llama], manifest_path=w.manifest)
        assert moved["written"] is True and moved["enrolled"] == [] and moved["kept"] == ["llama.gguf"]
        assert moved["previous_scheme"] == "hmac-sha512" and moved["scheme"] == "mldsa65"
        written = json.loads(w.manifest.read_text(encoding="utf-8"))
        assert written["entries"] == entries and written["seal"]["scheme"] == "mldsa65"
    finally:
        w.restore()


def test_ml6_a_seal_this_host_holds_no_key_for_is_key_unavailable_never_a_fresh_start(tmp_path):
    # No keypair; then a keypair file with no post-quantum primitive to check with.
    for place, keypair, primitive in (("gone", False, True), ("unchecked", True, False)):
        folder = tmp_path / place
        folder.mkdir()
        w = _world(folder, signature=False, keypair=keypair, primitive=primitive)
        try:
            prov = w.prov
            llama = _model(folder, "llama.gguf", b"GGUF llama")
            payload = _old_payload({"llama.gguf": _entry(llama)})
            before = _old_write(w.manifest, payload, _old_pqc(payload))
            assert prov.resolve_seal_keys().scheme == "hmac-sha512", "control: this host seals with a MAC"
            for write in (lambda: prov.record_model(llama, manifest_path=w.manifest),
                          lambda: prov.enroll_models([llama], manifest_path=w.manifest),
                          lambda: prov.reseal_manifest(manifest_path=w.manifest)):
                with pytest.raises(prov.ManifestNotVerified) as refused:
                    write()
                assert refused.value.reason == prov.REASON_KEY_UNAVAILABLE, (place, str(refused.value))
                assert "mldsa65" in str(refused.value), place
                assert w.manifest.read_bytes() == before and _files(w.data) == ["model_provenance.json"]
        finally:
            w.restore()


def test_ml7_a_manifest_that_does_not_parse_is_unreadable_never_overwritten_never_absent(tmp_path):
    w = _world(tmp_path)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        for body in (b"{not json", b"[1, 2, 3]\n"):
            w.manifest.write_bytes(body)
            for write in (lambda: prov.record_model(llama, manifest_path=w.manifest),
                          lambda: prov.enroll_models([llama], manifest_path=w.manifest),
                          lambda: prov.reseal_manifest(manifest_path=w.manifest)):
                with pytest.raises(prov.ManifestNotVerified) as refused:
                    write()
                assert refused.value.reason == prov.REASON_MANIFEST_UNREADABLE, body
                assert "no model provenance manifest" not in str(refused.value).lower()
                assert w.manifest.read_bytes() == body and _files(w.data) == ["model_provenance.json"]
    finally:
        w.restore()


def test_ml8_an_enrolment_never_repins_a_changed_model_unless_it_is_named(tmp_path):
    w = _world(tmp_path)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        fresh = _model(tmp_path, "fresh.gguf", b"GGUF fresh")
        payload = _old_payload({"llama.gguf": _entry(llama)})
        before = _old_write(w.manifest, payload, _old_hmac(payload, _K))
        llama.write_bytes(b"GGUF llama, other weights")
        with pytest.raises(prov.PinnedFileChanged) as refused:
            prov.enroll_models([llama, fresh], manifest_path=w.manifest)
        assert refused.value.models == ["llama.gguf"] and refused.value.reason == prov.REASON_DIGEST_MISMATCH
        assert w.manifest.read_bytes() == before, "nothing written: the fresh model waits too"
        decision = prov.verify_model(llama, manifest_path=w.manifest, mode="bulbe")
        assert decision.reason == prov.REASON_DIGEST_MISMATCH and decision.allowed is False
        result = prov.enroll_models([llama, fresh], manifest_path=w.manifest, repin=["llama.gguf", "ghost.gguf"])
        assert result["repinned"] == ["llama.gguf"] and result["repin_unused"] == ["ghost.gguf"]
        entries = json.loads(w.manifest.read_text(encoding="utf-8"))["entries"]
        assert entries["llama.gguf"]["sha256"] == hashlib.sha256(llama.read_bytes()).hexdigest()
        assert set(entries) == {"llama.gguf", "fresh.gguf"}
    finally:
        w.restore()


def test_ml9_a_pin_that_holds_is_kept_verbatim_and_nothing_new_writes_nothing(tmp_path):
    w = _world(tmp_path)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        fresh = _model(tmp_path, "fresh.gguf", b"GGUF fresh")
        held = _entry(llama, recorded_at=1.0)
        payload = _old_payload({"llama.gguf": held})
        _old_write(w.manifest, payload, _old_hmac(payload, _K))
        result = prov.enroll_models([llama, fresh], manifest_path=w.manifest)
        assert result["kept"] == ["llama.gguf"] and [e["model"] for e in result["enrolled"]] == ["fresh.gguf"]
        entries = json.loads(w.manifest.read_text(encoding="utf-8"))["entries"]
        assert entries["llama.gguf"] == held, "kept verbatim, recorded_at 1.0 included"
        once = w.manifest.read_bytes()
        again = prov.enroll_models([llama, fresh], manifest_path=w.manifest)
        assert again["written"] is False and again["enrolled"] == [] and w.manifest.read_bytes() == once
        # Zero models: the whole shape of a result, nothing written.
        nothing = prov.enroll_models([], manifest_path=w.manifest, repin=["x.gguf"])
        assert nothing == {"enrolled": [], "count": 0, "scheme": None, "kept": [], "repinned": [],
                           "repin_unused": ["x.gguf"], "previous_scheme": None, "written": False}
        assert w.manifest.read_bytes() == once
    finally:
        w.restore()


def test_ml10_what_is_sealed_is_what_was_verified(tmp_path):
    w = _world(tmp_path, signature=True)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        fresh = _model(tmp_path, "fresh.gguf", b"GGUF fresh")
        payload = _old_payload({"llama.gguf": _entry(llama)})
        checked = prov.verify_seal

        def then_written(*args, **kwargs):
            verdict = checked(*args, **kwargs)
            if verdict == prov.REASON_VERIFIED:
                _outside(w.manifest, _evil)
            return verdict

        prov.verify_seal = then_written
        for write in (lambda: prov.enroll_models([fresh], manifest_path=w.manifest),
                      lambda: prov.reseal_manifest(manifest_path=w.manifest)):
            _old_write(w.manifest, payload, _old_hmac(payload, _K))
            write()
            sealed = json.loads(w.manifest.read_text(encoding="utf-8"))
            assert "evil.gguf" not in sealed["entries"] and sealed["seal"]["scheme"] == "mldsa65", sealed
    finally:
        w.restore()


def test_ml11_starting_again_is_explicit_and_sets_the_old_file_aside_untouched(tmp_path):
    w = _world(tmp_path)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        fresh = _model(tmp_path, "fresh.gguf", b"GGUF fresh")
        payload = _old_payload({"llama.gguf": _entry(llama)})
        _old_write(w.manifest, payload, _old_hmac(payload, _OTHER))
        before = _outside(w.manifest, _evil)
        result = prov.enroll_models([llama, fresh], manifest_path=w.manifest, start_again=True)
        assert result["dropped"] == {"reason": "seal_invalid", "count": 2, "names": ["evil.gguf", "llama.gguf"]}
        aside = Path(result["set_aside"])
        assert aside.parent == w.data and aside.name.startswith("model_provenance.unverified-")
        assert aside.name.endswith("Z.json") and aside.read_bytes() == before, "the old bytes, untouched"
        reading = prov.read_manifest(w.manifest, keys=prov.resolve_seal_keys())
        assert reading.status == prov.REASON_VERIFIED and set(reading.entries) == {"llama.gguf", "fresh.gguf"}
        # The name it is set aside under is one git ignores, where the manifest lives.
        probe = f"data/{aside.name}"
        answer = subprocess.run(["git", "-C", str(REPO), "check-ignore", "--no-index", "-q", probe],
                                capture_output=True, timeout=60)
        assert answer.returncode == 0, ("git would carry the set-aside manifest", probe)
        # Refused, nothing moved: on a manifest that verifies, on an absent one, on one that cannot be read.
        listing = _files(w.data)
        with pytest.raises(prov.ProvenanceError):
            prov.enroll_models([llama], manifest_path=w.manifest, start_again=True)
        absent = w.data / "elsewhere.json"
        with pytest.raises(prov.ManifestNotVerified) as refused:
            prov.enroll_models([llama], manifest_path=absent, start_again=True)
        assert refused.value.reason == prov.REASON_MANIFEST_MISSING and not absent.exists()
        locked = w.data / "locked.json"
        locked.write_bytes(b"{not json")
        os.chmod(locked, 0)
        try:
            if os.access(locked, os.R_OK):
                pytest.skip("this process reads a file it has no permission on")
            with pytest.raises(prov.ManifestNotVerified) as refused:
                prov.enroll_models([llama], manifest_path=locked, start_again=True)
            assert refused.value.reason == prov.REASON_MANIFEST_UNREADABLE and "EACCES" in str(refused.value)
        finally:
            os.chmod(locked, 0o600)
        assert _files(w.data) == sorted([*listing, "locked.json"]), "nothing set aside on a refusal"
    finally:
        w.restore()


def _real_world(tmp_path, monkeypatch):
    """The real key file (encryption and the secrets it takes out of the environment) beside model_provenance; no
    post-quantum seal asked for or available."""
    pytest.importorskip("cryptography")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE"):
        monkeypatch.delenv(name, raising=False)
    pqc = types.ModuleType("opti_oignon.pqc_signatures")
    pqc.PQC_AVAILABLE = False
    pqc.is_pqc_enabled = lambda: False
    pqc.pqc_required = lambda: False

    def no_keypair():
        raise RuntimeError("no post-quantum keypair in these contracts")

    pqc.load_pqc_keypair = no_keypair
    loaded, restore = isolate(targets={"opti_oignon.process_secrets": source("process_secrets.py"),
                                       "opti_oignon.encryption": source("encryption.py"),
                                       PROV: source("model_provenance.py")},
                              blocked=("opti_oignon.secure_bytes", "opti_oignon.security_mode"),
                              seeded={"opti_oignon.pqc_signatures": pqc})
    try:
        enc, prov = loaded["opti_oignon.encryption"], loaded[PROV]
        data = tmp_path / "data"
        data.mkdir()
        enc._DEFAULT_KEYFILE = data / ".keyfile"
        prov._DATA_DIR, prov.DEFAULT_MANIFEST_PATH = data, data / "model_provenance.json"
    except BaseException:
        restore()
        raise
    return SimpleNamespace(enc=enc, prov=prov, restore=restore, data=data, keyfile=data / ".keyfile",
                           manifest=data / "model_provenance.json")


def test_ml12_on_the_real_key_file_the_old_fallback_seal_vouches_for_nothing(tmp_path, monkeypatch):
    w = _real_world(tmp_path, monkeypatch)
    try:
        prov, phrase, key = w.prov, "a passphrase for ML12", bytes(range(1, 33))
        model = _model(tmp_path, "pinned-Q4_K_M.gguf", b"GGUF" + bytes(range(256)) * 4)
        w.enc.save_keyfile(key, passphrase=phrase)
        opening = w.keyfile.read_bytes()[:32]
        payload = _old_payload({model.name: _entry(model), "ghost.gguf": {"sha256": "g" * 64, "size": 1,
                                                                           "recorded_at": 1.0}})
        before = _old_write(w.manifest, payload, _old_hmac(payload, opening))
        # Without the passphrase: no key to seal with, nothing written.
        for write in (lambda: prov.reseal_manifest(manifest_path=w.manifest),
                      lambda: prov.enroll_models([model], manifest_path=w.manifest)):
            with pytest.raises(prov.ProvenanceError):
                write()
            assert w.manifest.read_bytes() == before
        # With it: the seal under the opening bytes vouches for nothing.
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", phrase)
        for write in (lambda: prov.reseal_manifest(manifest_path=w.manifest),
                      lambda: prov.enroll_models([model], manifest_path=w.manifest)):
            with pytest.raises(prov.ManifestNotVerified) as refused:
                write()
            assert refused.value.reason == prov.REASON_SEAL_INVALID and w.manifest.read_bytes() == before
        result = prov.enroll_models([model], manifest_path=w.manifest, start_again=True)
        assert Path(result["set_aside"]).read_bytes() == before and result["dropped"]["count"] == 2
        written = json.loads(w.manifest.read_text(encoding="utf-8"))
        assert set(written["entries"]) == {model.name}, "ghost.gguf is gone"
        assert written["seal"] == _old_hmac(_old_payload(written["entries"]), key), "sealed under the key that loads"
    finally:
        w.restore()


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_contract", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = path
    return module


def _backends(tmp_path, models):
    """A backends.yaml in tmp naming one folder of models."""
    target = tmp_path / "backends.yaml"
    target.write_text(f"llama_cpp:\n  model_dirs:\n    - {models}\n", encoding="utf-8")
    return target


def test_ml13_the_enrolment_script_refuses_an_unverified_manifest_by_its_reason(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path)
    try:
        models = tmp_path / "models"
        models.mkdir()
        llama = _model(models, "llama.gguf", b"GGUF llama")
        fresh = _model(models, "fresh.gguf", b"GGUF fresh")
        payload = _old_payload({"llama.gguf": _entry(llama)})
        _old_write(w.manifest, payload, _old_hmac(payload, _K))
        before = _outside(w.manifest, _evil)
        script = _script("enroll_models")
        argv = ["enroll_models.py", "--manifest", str(w.manifest), "--backends", str(_backends(tmp_path, models))]
        monkeypatch.setattr(sys, "argv", argv)
        assert script.main() == 1
        said = capsys.readouterr().out
        assert "REFUSED (seal_invalid)" in said and "--start-again" in said, said
        assert "(already pinned)" not in said, said
        assert w.manifest.read_bytes() == before
        # Control: on the manifest as sealed, the pin that verified is said, and the fresh model pinned.
        _old_write(w.manifest, payload, _old_hmac(payload, _K))
        assert script.main() == 0
        said = capsys.readouterr().out
        assert "llama.gguf (already pinned)" in said and f"pinned fresh.gguf {_entry(fresh)['sha256']}" in said, said
    finally:
        w.restore()


def test_ml14_the_reseal_script_never_says_already_sealed_of_a_seal_it_did_not_check(tmp_path, monkeypatch, capsys):
    w = _world(tmp_path)
    try:
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        payload = _old_payload({"llama.gguf": _entry(llama)})
        before = _old_write(w.manifest, payload, {"scheme": "hmac-sha512", "value": "0" * 128})
        script = _script("reseal_model_manifest")
        monkeypatch.setattr(sys, "argv", ["reseal_model_manifest.py", "--manifest", str(w.manifest)])
        assert script.main() == 1
        said = capsys.readouterr().out
        assert "REFUSED (seal_invalid)" in said and "Already sealed" not in said, said
        assert w.manifest.read_bytes() == before
        # Control: a seal that verifies under the host's scheme is said to be there.
        _old_write(w.manifest, payload, _old_hmac(payload, _K))
        assert script.main() == 0 and "Already sealed with hmac-sha512" in capsys.readouterr().out
    finally:
        w.restore()


def test_ml15_the_load_gate_takes_none_of_the_writers_latitude(tmp_path):
    w = _world(tmp_path, signature=True)
    try:
        prov = w.prov
        llama = _model(tmp_path, "llama.gguf", b"GGUF llama")
        payload = _old_payload({"llama.gguf": _entry(llama)})
        _old_write(w.manifest, payload, _old_hmac(payload, _K))
        decision = prov.verify_model(llama, manifest_path=w.manifest, mode="bulbe")
        assert decision.reason == prov.REASON_SEAL_DOWNGRADE and decision.allowed is False
        # Control: the writers would lift it -- that latitude is theirs alone.
        assert prov.read_manifest(w.manifest, keys=prov.resolve_seal_keys()).status == prov.REASON_VERIFIED
        prov.reseal_manifest(manifest_path=w.manifest)
        assert prov.verify_model(llama, manifest_path=w.manifest, mode="bulbe").reason == prov.REASON_VERIFIED
    finally:
        w.restore()
