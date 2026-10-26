#!/usr/bin/env python3
"""Contracts for the escalation preflight: it judges the provenance manifest as the fortress's load gate will.

In Bulbe the load gate of a gated backend takes one seal: a signature that verifies under the keypair's public
half, over a manifest that pins the model. The preflight judged the seal by the scheme it CLAIMED and the
presence of entries: a signature that does not verify -- made under a keypair since rotated, or over entries
changed since -- passed, and the host escalated into refusing every model it loads. It now reads the manifest
as the gate will (``read_manifest_for_fortress``) and names what it found and a way out that works here.

Manifests are written by the OLD recipe, inline in the writers' suite, never by the module.

  * EV1 -- A SIGNATURE THAT DOES NOT VERIFY BLOCKS A GATED BACKEND: ``seal_invalid``, starting again named,
    nothing written.
  * EV2 -- A MAC THAT VERIFIES IS A DOWNGRADE: on a host that seals with a MAC the flag that asks for the
    signature is named; on one that seals with the signature, the re-seal it names lifts it and the host then
    escalates.
  * EV3 -- A MAC THAT DOES NOT VERIFY IS NEVER SENT TO THE RE-SEAL, which refuses it: ``seal_invalid``,
    starting again named.
  * EV4 -- A MANIFEST THAT CANNOT BE READ, CARRIES NO SEAL, NAMES AN UNKNOWN SCHEME OR IS MALFORMED IS NAMED BY
    ITS REASON, never "no provenance manifest": one nested past what the parser reads too, and a folder, a pipe or
    a device at the path, refused without being read and never waited on.
  * EV5 -- A SIGNATURE THAT VERIFIES OVER PINS ESCALATES.
  * EV6 -- NO MANIFEST, OR ONE THAT PINS NOTHING -- under a signature or a MAC --, BLOCKS A GATED BACKEND AND
    NAMES THE ENROLMENT, never the re-seal.
  * EV7 -- AN UNGATED DEFAULT BACKEND ESCALATES WHATEVER ITS MANIFEST IS, and the manifest is not read.
  * EV8 -- THE PREFLIGHT AGREES WITH THE GATE: it blocks exactly the states in which Bulbe's gate refuses the
    pinned model, a malformed manifest excepted (the preflight is stricter there, said).
  * EV9 -- A READING THAT RAISES MANUFACTURES NO REFUSAL, A WAY OUT THAT RAISES UNMAKES NONE, AND THE EMERGENCY
    STOP IS NEVER REFUSED BY THE MANIFEST.
  * EV10 -- EVERY WAY OUT IT NAMES, FOLLOWED ON THE HOST IT WAS NAMED ON, LETS THE HOST ESCALATE: on a host that
    seals with a MAC or with the signature, with its keypair or without, its primitive gone; its first alternative
    and its last; one that writes a seal is the last one needed, none weakens a signature that verified, and none
    sets aside a manifest that still verifies here. Bringing back the keypair that sealed a manifest restores that
    pair; making a keypair makes a new one, under which nothing sealed before verifies.
  * EV11 -- THE READER HOLDS NO DESCRIPTOR ONCE IT HAS ANSWERED, whatever stands at the path.
  * EV12 -- THE WRITERS REFUSE WHAT THE PREFLIGHT REFUSES, WITH A WAY OUT THAT WORKS: a pin, an enrolment, a fresh
    start and a re-seal name a folder, a pipe or a device and say to move it away; a manifest nested past the
    parser is started again; a signature they cannot check for want of the primitive is left as it is.

Local-only (the public distribution ships no tests). The real ``security_mode`` and ``model_provenance`` loaded
together through the shared isolation window, every manifest in tmp.
"""

import hmac
import logging
import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from test_model_manifest_writers_verify_contracts import (  # noqa: E402
    _K,
    _OTHER,
    _PRIVATE,
    _PUBLIC,
    _entry,
    _evil,
    _files,
    _model,
    _old_hmac,
    _old_payload,
    _old_pqc,
    _old_write,
    _outside,
    _sign,
)

PROV = "opti_oignon.model_provenance"
MODE = "opti_oignon.security_mode"
BUDGET_S = {
    "test_ev1_a_signature_that_does_not_verify_blocks_a_gated_backend": 2.0,
    "test_ev2_a_mac_that_verifies_is_a_downgrade_whose_named_way_out_works_on_this_host": 2.0,
    "test_ev3_a_mac_that_does_not_verify_is_never_sent_to_the_reseal": 2.0,
    "test_ev4_a_manifest_that_cannot_be_judged_is_named_by_its_reason": 3.0,
    "test_ev5_a_signature_that_verifies_over_pins_escalates": 2.0,
    "test_ev6_no_manifest_or_one_that_pins_nothing_blocks_and_names_the_enrolment": 2.0,
    "test_ev7_an_ungated_backend_escalates_whatever_its_manifest_and_it_is_not_read": 2.0,
    "test_ev8_the_preflight_blocks_exactly_where_the_gate_refuses": 3.0,
    "test_ev9_a_reading_that_raises_refuses_nothing_and_the_emergency_stop_is_never_refused": 2.0,
    "test_ev10_every_way_out_it_names_followed_on_its_host_lets_the_host_escalate": 6.0,
    "test_ev11_the_reader_holds_no_descriptor_once_it_has_answered": 3.0,
    "test_ev12_the_writers_refuse_what_the_preflight_refuses_with_a_way_out_that_works": 3.0,
}

# The model a gated backend would load, and its name in the manifest.
_NAME = "llama.gguf"
# How long a reading may take before it is called a wait on what stands at the path.
_HANG_S = 3.0


def _world(tmp_path, *, signature=False, keypair=True, primitive=True, backend="llama_cpp"):
    """security_mode and model_provenance over ML's stand-ins: the signing pair that binds the data and its
    keypair, required when ``signature`` (``world.flags.required`` switches it, as the flag does), its keypair
    loading when ``keypair`` -- and refused, as the real loader refuses a key whose mechanism no longer resolves,
    when the primitive is gone (``_primitive``) --; the key file loading ``_K`` while ``world.flags.keyfile``; the
    default backend ``backend``; the mode, lockfile and audit writes recorded, never made."""
    flags = SimpleNamespace(required=signature, keypair=keypair, primitive=primitive, keyfile=True,
                            pair=(_PUBLIC, _PRIVATE))
    # Every keypair this world has seen: a signature verifies under the public half of the pair that made it.
    pairs = {_PUBLIC: _PRIVATE}
    pqc = types.ModuleType("opti_oignon.pqc_signatures")
    pqc.PQC_AVAILABLE = primitive
    pqc.is_pqc_enabled = lambda: flags.required
    pqc.pqc_required = lambda: flags.required

    def load_pqc_keypair():
        if not flags.keypair:
            raise FileNotFoundError("PQC keypair file not found")
        if not flags.primitive:
            raise ValueError("the keypair was minted under ML-DSA-65; this host resolves no mechanism")
        return flags.pair

    def sign_bytes(data, private):
        if not flags.primitive:
            raise RuntimeError("the post-quantum primitive is not available")
        return _sign(data, private)

    pqc.load_pqc_keypair = load_pqc_keypair
    pqc.sign_bytes = sign_bytes
    def verify_bytes(data, signature, public):
        private = pairs.get(bytes(public))
        return bool(flags.primitive and private is not None
                    and hmac.compare_digest(bytes(signature), _sign(data, private)))

    pqc.verify_bytes = verify_bytes
    pqc.signing_blockers = lambda: []
    encryption = types.ModuleType("opti_oignon.encryption")

    def load_keyfile():
        if not flags.keyfile:
            raise ValueError("wrapped under a passphrase this process lacks")
        return _K, b"salt", "kdf"

    encryption.load_keyfile = load_keyfile
    loaded, restore = isolate(
        targets={PROV: source("model_provenance.py"), MODE: source("security_mode.py")},
        seeded={"opti_oignon.pqc_signatures": pqc, "opti_oignon.encryption": encryption},
    )
    try:
        prov, mod = loaded[PROV], loaded[MODE]
        data = tmp_path / "data"
        data.mkdir(parents=True)
        prov._DATA_DIR, prov.DEFAULT_MANIFEST_PATH = data, data / "model_provenance.json"
        models = tmp_path / "models"
        models.mkdir(parents=True)
        written = []
        mod._load_signing_key = lambda: b"s" * 32
        mod._write_yaml_mode = lambda mode: written.append(mode)
        mod._write_lockfile = lambda mode, user_id, key: 1.0
        mod._audit_log = lambda *a, **k: None
        mod._default_backend = lambda: backend
        mod._LOCKFILE_PATH = data / ".security_mode_lock"
    except BaseException:
        restore()
        raise
    return SimpleNamespace(prov=prov, mod=mod, pqc=pqc, restore=restore, data=data,
                           manifest=data / "model_provenance.json",
                           model=_model(models, _NAME, b"weights of the model"), written=written, flags=flags,
                           pairs=pairs, sealer=(_PUBLIC, _PRIVATE))


def _primitive(world, alive):
    world.flags.primitive = alive
    world.pqc.PQC_AVAILABLE = alive


def _manager(world):
    manager = world.mod.SecurityModeManager()
    manager._cached_mode = world.mod.MODE_DAILY
    return manager


def _pins(world):
    return _old_payload({_NAME: _entry(world.model)})


def _state(world, name):
    """Write the manifest state ``name`` by the old recipe (``missing`` writes nothing)."""
    payload = _pins(world)
    empty = _old_payload({})
    if name in ("signed", "no_keypair", "dead_primitive"):
        _old_write(world.manifest, payload, _old_pqc(payload))
        if name == "no_keypair":
            world.flags.keypair = False
        if name == "dead_primitive":
            _primitive(world, False)
    elif name == "another_keypair":
        # Sealed under a keypair since rotated away: its pair is known, not the one this host loads.
        world.sealer = (b"PUBLIC-ROTATED", b"PRIVATE-ROTATED")
        world.pairs[b"PUBLIC-ROTATED"] = b"PRIVATE-ROTATED"
        _old_write(world.manifest, payload, _old_pqc(payload, private=b"PRIVATE-ROTATED"))
    elif name == "entry_added_after":
        _old_write(world.manifest, payload, _old_pqc(payload))
        _outside(world.manifest, _evil)
    elif name == "mac":
        _old_write(world.manifest, payload, _old_hmac(payload, _K))
    elif name == "mac_empty":
        _old_write(world.manifest, empty, _old_hmac(empty, _K))
    elif name == "mac_another_key":
        _old_write(world.manifest, payload, _old_hmac(payload, _OTHER))
    elif name == "keyfile_locked":
        _old_write(world.manifest, payload, _old_hmac(payload, _K))
        world.flags.keyfile = False
    elif name == "unsealed":
        _old_write(world.manifest, payload, None)
    elif name == "unknown_scheme":
        _old_write(world.manifest, payload, {"scheme": "rot13", "value": "uryyb"})
    elif name == "not_json":
        world.manifest.write_bytes(b'{"version": 1, "entries": ')
    elif name == "nested":
        world.manifest.write_bytes(b'{"version": 1, "entries": ' + b"[" * 100000 + b"]" * 100000 + b"}")
    elif name == "a_folder":
        world.manifest.mkdir()
    elif name == "a_fifo":
        os.mkfifo(world.manifest)
    elif name == "a_device":
        world.manifest.symlink_to(os.devnull)
    elif name == "malformed":
        bad = _old_payload({_NAME: _entry(world.model), "other.gguf": {"sha256": 7, "size": 1}})
        _old_write(world.manifest, bad, _old_pqc(bad))
    elif name == "signed_empty":
        _old_write(world.manifest, empty, _old_pqc(empty))
    else:
        assert name == "missing", name


def _snapshot(world):
    held = world.manifest.read_bytes() if world.manifest.is_file() else None
    return held, _files(world.data)


def _guarded(world, call):
    """``call()`` in a thread: a reading that waits on what stands at the manifest's path is let go after
    ``_HANG_S`` -- a writer opens the pipe -- and the wait is the failure, never a suite that hangs."""
    box = {}

    def run():
        try:
            box["result"] = call()
        except BaseException as exc:  # noqa: BLE001 - carried to the test's own thread
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(_HANG_S)
    if worker.is_alive():
        try:
            os.close(os.open(world.manifest, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
        worker.join(_HANG_S)
        raise AssertionError(f"the preflight waited on what stands at {world.manifest.name}")
    if "error" in box:
        raise box["error"]
    return box["result"]


def _escalated(world, user="leon", force=False):
    before = _snapshot(world)
    result = _guarded(world, lambda: _manager(world).escalate_to_bulbe(user, force=force))
    assert _snapshot(world) == before, "the preflight wrote to the data folder"
    return result


def _blocker(result):
    """The one blocker a refused escalation gives for the manifest (the signing stand-in gives none)."""
    assert result["success"] is False and result["error"] == "fortress_not_ready", result
    assert len(result["blockers"]) == 1, ("one reading of the manifest, one blocker", result["blockers"])
    assert "post-quantum signature" not in result["message"], (
        f"this host signs; only its manifest blocks, and the message says it cannot sign: {result['message']}")
    assert "the remedy lists the calls" in result["message"], (
        f"not every blocker names its way out (the signing ones do not): the message must send to the remedy list. "
        f"{result['message']}")
    return result["blockers"][0]


def test_ev1_a_signature_that_does_not_verify_blocks_a_gated_backend(tmp_path):
    for name in ("another_keypair", "entry_added_after"):
        world = _world(tmp_path / name)
        try:
            _state(world, name)
            result = _escalated(world)
            blocker = _blocker(result)
            assert world.written == [], "refused, and the mode written anyway"
            assert "seal_invalid" in blocker and "mldsa65" in blocker, (
                f"{name}: a signature that does not verify under the keypair was read as ready, or misnamed: the "
                f"gate refuses every model it pins. {blocker}")
            assert "--start-again" in blocker and "re-seal" not in blocker, (
                f"{name}: the way out of a seal that does not verify is to start again once the weights are "
                f"checked; a re-seal refuses it. {blocker}")
        finally:
            world.restore()


def test_ev2_a_mac_that_verifies_is_a_downgrade_whose_named_way_out_works_on_this_host(tmp_path):
    world = _world(tmp_path / "daily_mac", signature=False)
    try:
        _state(world, "mac")
        blocker = _blocker(_escalated(world))
        assert "seal_downgrade" in blocker and "hmac-sha512" in blocker, blocker
        assert "backup_signatures: true" in blocker and "re-seal" in blocker, (
            "on a host that seals with a MAC, a re-seal keeps the MAC and says there is nothing to do: the way out "
            f"must ask for the signature first. {blocker}")
        assert world.prov.reseal_manifest()["written"] is False, "the host seals with a MAC: the re-seal keeps it"
    finally:
        world.restore()

    world = _world(tmp_path / "signing_host", signature=True)
    try:
        _state(world, "mac")
        blocker = _blocker(_escalated(world))
        assert "seal_downgrade" in blocker and "re-seal" in blocker and "backup_signatures" not in blocker, blocker
        lifted = world.prov.reseal_manifest()
        assert lifted["written"] is True and lifted["scheme"] == "mldsa65", lifted
        result = _escalated(world)
        assert result["success"] is True and world.written == [world.mod.MODE_BULBE], (
            f"the way out the blocker named was taken and the host is still refused: {result}")
    finally:
        world.restore()


def test_ev3_a_mac_that_does_not_verify_is_never_sent_to_the_reseal(tmp_path):
    world = _world(tmp_path, signature=True)
    try:
        _state(world, "mac_another_key")
        blocker = _blocker(_escalated(world))
        assert "seal_invalid" in blocker and "hmac-sha512" in blocker, blocker
        assert "re-seal" not in blocker and "reseal" not in blocker, (
            f"a MAC that does not verify here was sent to the re-seal, which refuses it: {blocker}")
        assert "--start-again" in blocker, blocker
        refused = None
        try:
            world.prov.reseal_manifest()
        except world.prov.ManifestNotVerified as exc:
            refused = exc.reason
        assert refused == "seal_invalid", "the re-seal itself must refuse what the blocker kept it from"
    finally:
        world.restore()


def test_ev4_a_manifest_that_cannot_be_judged_is_named_by_its_reason(tmp_path):
    expected = {
        "not_json": ("manifest_unreadable", "does not parse"),
        "nested": ("manifest_unreadable", "nests deeper than the JSON parser reads"),
        "a_folder": ("manifest_unreadable", "not a regular file but a folder"),
        "a_fifo": ("manifest_unreadable", "not a regular file but a pipe"),
        "a_device": ("manifest_unreadable", "not a regular file but a device"),
        "unsealed": ("seal_missing", "no seal"),
        "unknown_scheme": ("seal_invalid", "no scheme this code knows"),
        "malformed": ("manifest_malformed", "not an object of objects"),
    }
    for name, (reason, said) in expected.items():
        world = _world(tmp_path / name)
        try:
            _state(world, name)
            blocker = _blocker(_escalated(world))
            assert f"({reason}: " in blocker and said in blocker, f"{name}: {blocker}"
            assert "no provenance manifest" not in blocker and "pins no model" not in blocker, (
                f"{name}: a manifest that exists was called absent or empty, and its remedy (enrol) refuses it: "
                f"{blocker}")
            if name.startswith("a_"):
                assert "move away what stands at that path" in blocker and "start again" not in blocker, (
                    f"{name}: no file stands at the path to start again from, and a mode change makes none: "
                    f"{blocker}")
            if name == "malformed":
                assert "nothing more is pinned" in blocker and "every model it loads" not in blocker, (
                    f"the gate may load a model whose own entry is sound: the blocker must not say every one is "
                    f"refused. {blocker}")
        finally:
            world.restore()


def test_ev5_a_signature_that_verifies_over_pins_escalates(tmp_path):
    world = _world(tmp_path)
    try:
        _state(world, "signed")
        result = _escalated(world)
        assert result["success"] is True and world.written == [world.mod.MODE_BULBE], (
            f"a manifest whose signature verifies over the pin of the model was refused: a false brick. {result}")
    finally:
        world.restore()


def test_ev6_no_manifest_or_one_that_pins_nothing_blocks_and_names_the_enrolment(tmp_path):
    for name, said in (("missing", "has no provenance manifest"), ("signed_empty", "pins no model"),
                       ("mac_empty", "(seal_downgrade: ")):
        world = _world(tmp_path / name)
        try:
            _state(world, name)
            blocker = _blocker(_escalated(world))
            assert said in blocker, f"{name}: {blocker}"
            assert "enroll-models" in blocker and "scripts/enroll_models.py" in blocker, (
                f"{name}: a refusal with no remedy is a wall; enrolling writes the pins the gate needs. {blocker}")
            assert "reseal_model_manifest.py" not in blocker, (
                f"{name}: there is no pin to carry; a re-seal leaves a manifest that pins nothing. {blocker}")
            assert "--start-again" not in blocker and "start again" not in blocker, (
                f"{name}: starting again refuses an absent manifest and one that verifies; the way out is to enrol. "
                f"{blocker}")
            assert world.written == []
        finally:
            world.restore()


def test_ev7_an_ungated_backend_escalates_whatever_its_manifest_and_it_is_not_read(tmp_path):
    for backend, name in (("ollama", "another_keypair"), ("ollama", "mac"), ("llama_server", "a_folder"),
                          (None, "unsealed")):
        world = _world(tmp_path / f"{backend}_{name}", backend=backend)
        try:
            _state(world, name)
            reads = []
            reader = world.prov.read_manifest
            world.prov.read_manifest = lambda *a, **k: reads.append(a) or reader(*a, **k)
            result = _escalated(world)
            assert result["success"] is True and world.written == [world.mod.MODE_BULBE], (
                f"{backend!r} never reads the manifest at load, and was refused over it ({name}): that buys no "
                f"security and only holds the host in Daily. {result}")
            assert reads == [], f"{backend!r}: the manifest was read for a backend whose load never reads it"
        finally:
            world.restore()

    world = _world(tmp_path / "gated", backend="llama_cpp")
    try:
        _state(world, "signed")
        reads = []
        reader = world.prov.read_manifest
        world.prov.read_manifest = lambda *a, **k: reads.append(a) or reader(*a, **k)
        assert _escalated(world)["success"] is True
        assert len(reads) == 1, "the counter must see the one reading of a gated backend, or it proves nothing"
    finally:
        world.restore()


def test_ev8_the_preflight_blocks_exactly_where_the_gate_refuses(tmp_path):
    states = ("missing", "signed", "another_keypair", "entry_added_after", "mac", "mac_empty", "mac_another_key",
              "unsealed", "unknown_scheme", "not_json", "nested", "a_folder", "a_fifo", "a_device", "signed_empty",
              "no_keypair", "dead_primitive", "keyfile_locked", "malformed")
    seen = {}
    for name in states:
        world = _world(tmp_path / name)
        try:
            _state(world, name)
            blocks = bool(_guarded(world, world.mod.SecurityModeManager._provenance_blockers))
            world.flags.required = True
            reason, _digest = world.prov.classify_model(world.model, world.prov.load_manifest())
            seen[name] = (blocks, reason)
        finally:
            world.restore()
    assert seen["signed"] == (False, "verified") and seen["missing"] == (True, "manifest_missing"), seen
    for name, (blocks, reason) in seen.items():
        if name == "malformed":
            assert blocks and reason == "verified", (
                "the one state the preflight judges more strictly than the gate: the writers refuse a malformed "
                f"manifest, though the gate loads a model whose own entry is sound. {seen[name]}")
            continue
        assert blocks == (reason != "verified"), (
            f"{name}: the preflight {'blocks' if blocks else 'passes'} and Bulbe's gate says {reason!r} of the "
            f"pinned model")


def test_ev9_a_reading_that_raises_refuses_nothing_and_the_emergency_stop_is_never_refused(tmp_path, caplog):
    world = _world(tmp_path / "raises")
    try:
        _state(world, "another_keypair")

        def broken(*a, **k):
            raise RuntimeError("the reader broke")

        world.prov.read_manifest_for_fortress = broken
        with caplog.at_level(logging.ERROR, logger=MODE):
            result = _escalated(world)
        assert result["success"] is True and world.written == [world.mod.MODE_BULBE], (
            f"a check that could not run manufactured a refusal: {result}")
        assert any("could not be judged" in record.getMessage() and "the reader broke" in record.getMessage()
                   for record in caplog.records), "a reading that raised and was never said"
    finally:
        world.restore()

    world = _world(tmp_path / "advice")
    try:
        _state(world, "another_keypair")

        def no_advice(reading):
            raise RuntimeError("the advice broke")

        world.prov.fortress_way_out = no_advice
        with caplog.at_level(logging.ERROR, logger=MODE):
            blocker = _blocker(_escalated(world))
        assert "seal_invalid" in blocker and "could not be named here" in blocker, (
            f"a way out that could not be named unmade the verdict it was asked for, or hid it: {blocker}")
        assert world.written == [], "the verdict stood, and the mode was written anyway"
        assert any("could not be named" in record.getMessage() and "the advice broke" in record.getMessage()
                   for record in caplog.records), "a way out that raised and was never said"
    finally:
        world.restore()

    world = _world(tmp_path / "stop")
    try:
        _state(world, "another_keypair")
        result = _escalated(world, user="emergency-stop", force=True)
        assert result["success"] is True and world.written == [world.mod.MODE_BULBE], (
            f"the emergency stop was refused over the manifest: a panic button that can say no. {result}")
    finally:
        world.restore()


# The hosts a way out is followed on, and what the way out of an absent manifest must say first on each, and must
# not.
_HOSTS = {
    "a host that seals with a MAC": (
        {"signature": False}, ("backup_signatures: true",), ("make a keypair", "post-quantum primitive")),
    "a host that seals with the signature": (
        {"signature": True}, (), ("backup_signatures", "make a keypair", "post-quantum primitive")),
    "a MAC host with no keypair": (
        {"signature": False, "keypair": False}, ("make a keypair", "backup_signatures: true"),
        ("post-quantum primitive",)),
    "a signing host with no keypair": (
        {"signature": True, "keypair": False}, ("make a keypair",), ("backup_signatures", "post-quantum primitive")),
    "a host whose primitive is gone": (
        {"signature": False, "primitive": False}, ("bring back the post-quantum primitive",),
        ("backup_signatures", "make a keypair")),
}
_EVERY = ("missing", "signed", "signed_empty", "another_keypair", "entry_added_after", "mac", "mac_empty",
          "mac_another_key", "keyfile_locked", "unsealed", "unknown_scheme", "not_json", "nested", "malformed",
          "a_folder", "a_fifo", "a_device", "no_keypair", "dead_primitive")
_FEWER = ("missing", "signed", "another_keypair", "mac", "mac_another_key", "a_folder", "dead_primitive")


def _follow(world, way, take):
    """Take the way out as an operator reads it -- its ``take`` alternative, "first" or "last", each step in the
    order it is said. True when it wrote a seal."""
    alternatives = way.split("; or, ")
    step = alternatives[0] if take == "first" else alternatives[-1]
    if step.startswith("bring back the post-quantum primitive"):
        _primitive(world, True)
        return False
    if step.startswith("bring back the keypair that sealed it"):
        world.flags.pair, world.flags.keypair = world.sealer, True
        return False
    if step.startswith("set OPTI_KEYFILE_PASSPHRASE so the key file loads"):
        world.flags.keyfile = True
        return False
    if step.startswith("make it a readable file, or move away what stands at that path"):
        if world.manifest.is_dir() and not world.manifest.is_symlink():
            world.manifest.rmdir()
        else:
            world.manifest.unlink()
        return False
    if "make a keypair (POST /api/security/pqc/generate-keys)" in step:
        # A new pair: what the old one sealed does not verify under it.
        world.pairs[b"PUBLIC-NEW"] = b"PRIVATE-NEW"
        world.flags.pair, world.flags.keypair = (b"PUBLIC-NEW", b"PRIVATE-NEW"), True
    if "set pqc: backup_signatures: true in security.yaml" in step:
        world.flags.required = True
    if "--start-again" in step:
        world.prov.enroll_models([world.model], start_again=True)
    elif "scripts/reseal_model_manifest.py" in step:
        world.prov.reseal_manifest()
    elif "scripts/enroll_models.py" in step:
        world.prov.enroll_models([world.model])
    else:
        raise AssertionError(f"a way out no operator can take: {step}")
    return True


def test_ev10_every_way_out_it_names_followed_on_its_host_lets_the_host_escalate(tmp_path):
    for host, (settings, said, unsaid) in _HOSTS.items():
        states = _EVERY if host in ("a host that seals with a MAC", "a host that seals with the signature") else _FEWER
        for name in states:
            for take in ("first", "last"):
                label = f"{name}, on {host}, taking the {take} alternative"
                world = _world(tmp_path / host.replace(" ", "_") / name / take, **settings)
                try:
                    _state(world, name)
                    ways = []
                    for _round in range(5):
                        result = _escalated(world)
                        if result["success"]:
                            break
                        way = _blocker(result).split(" To escalate, ", 1)[1].rstrip(".")
                        ways.append(way)
                        before = world.prov.read_manifest_for_fortress()
                        try:
                            wrote = _follow(world, way, take)
                        except AssertionError:
                            raise
                        except Exception as exc:
                            raise AssertionError(f"{label}: the way out {way!r} failed: {exc!r}") from exc
                        after = world.prov.read_manifest_for_fortress()
                        if before.status == "verified":
                            assert after.status == "verified", (
                                f"{label}: the way out {way!r} weakened a signature that verified: {after.status}")
                        if wrote:
                            assert after.status == "verified" and after.entries, (
                                f"{label}: the way out {way!r} wrote a seal the fortress still refuses "
                                f"({after.status}): each one that writes must be the last")
                    assert result["success"] is True and world.written == [world.mod.MODE_BULBE], (
                        f"{label}: followed {ways}, and the host is still refused: {result}")
                    if name == "missing":
                        assert all(words in ways[0] for words in said), (
                            f"{label}: before it enrols, the way out must ask for what this host lacks: {ways[0]}")
                        assert not any(words in ways[0] for words in unsaid), (
                            f"{label}: the way out asks for what this host has, or cannot do yet: {ways[0]}")
                    _primitive(world, True)
                    world.flags.keypair = world.flags.keyfile = True
                    for aside in sorted(world.data.glob("*.unverified-*")):
                        assert world.prov.read_manifest_for_fortress(aside).status != "verified", (
                            f"{label}: {aside.name} was set aside and still verifies as a signature: {ways}")
                finally:
                    world.restore()


def _descriptors():
    return len(os.listdir("/proc/self/fd"))


def test_ev11_the_reader_holds_no_descriptor_once_it_has_answered(tmp_path):
    for name in ("signed", "not_json", "nested", "a_folder", "a_fifo", "a_device", "missing"):
        world = _world(tmp_path / name)
        try:
            _state(world, name)
            keys = world.prov.SealKeys(scheme="hmac-sha512", sign_key=_K, verify_key=_K)

            def readings():
                for _ in range(25):
                    world.prov.read_manifest_for_fortress()
                    world.prov.read_manifest(keys=keys)

            _guarded(world, readings)
            before = _descriptors()
            _guarded(world, readings)
            assert _descriptors() == before, f"{name}: fifty readings left descriptors open"
        finally:
            world.restore()
    # The count sees a descriptor that stays open, or it proves nothing.
    before = _descriptors()
    held = os.open(tmp_path, os.O_RDONLY)
    try:
        assert _descriptors() == before + 1, "the count of open descriptors does not see one held open"
    finally:
        os.close(held)


def _refused(world, call):
    """The refusal ``call`` raises, by its reason; a writer that wrote instead fails the contract."""
    try:
        _guarded(world, call)
    except Exception as exc:
        if isinstance(exc, world.prov.ManifestNotVerified):
            return exc
        raise
    raise AssertionError("the writer wrote over what it should refuse")


def test_ev12_the_writers_refuse_what_the_preflight_refuses_with_a_way_out_that_works(tmp_path):
    writers = {
        "a download's pin": lambda world: world.prov.record_model(world.model),
        "an enrolment": lambda world: world.prov.enroll_models([world.model]),
        "a fresh start": lambda world: world.prov.enroll_models([world.model], start_again=True),
        "a re-seal": lambda world: world.prov.reseal_manifest(),
    }
    for name in ("a_folder", "a_fifo", "a_device"):
        world = _world(tmp_path / name)
        try:
            _state(world, name)
            before = _snapshot(world)
            for writer, write in writers.items():
                refused = _refused(world, lambda: write(world))
                said = str(refused)
                assert refused.reason == "manifest_unreadable" and "not a regular file" in said, (name, writer, said)
                assert "move away what stands at that path" in said and "start again from" not in said, (
                    f"{writer}, {name}: no file stands at the path; the way out is to move it away: {said}")
            assert _snapshot(world) == before, f"{name}: a refusal wrote"
        finally:
            world.restore()

    world = _world(tmp_path / "nested")
    try:
        _state(world, "nested")
        said = str(_refused(world, lambda: world.prov.record_model(world.model)))
        assert "nests deeper" in said and "--start-again" in said, said
        started = _guarded(world, lambda: world.prov.enroll_models([world.model], start_again=True))
        assert started["written"] is True and started["set_aside"], (
            f"the fresh start a nested manifest is told to take refused it: {started}")
    finally:
        world.restore()

    world = _world(tmp_path / "dead_primitive")
    try:
        _state(world, "dead_primitive")
        before = _snapshot(world)
        said = str(_refused(world, lambda: world.prov.record_model(world.model)))
        assert "bring back the post-quantum primitive" in said and "start again" not in said, (
            f"a signature no primitive here can check was offered a fresh start, which sets it aside: {said}")
        assert _snapshot(world) == before, "a refusal wrote"
    finally:
        world.restore()
