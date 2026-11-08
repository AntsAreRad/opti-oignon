"""Contracts of the Ollama models' provenance (MP-01): each Ollama model pinned by the digest of its manifest under
the same seal as the GGUF files, every request for it checked against the digest the engine lists now, refused under
enforcement (Bulbe always), observed in Daily by default.

  * OV1 -- THE CLASSIFICATION: a pinned digest listed as pinned is verified; another digest under the same name is a
    mismatch; a name never pinned is unpinned; a name the engine does not list cannot be compared; a bare name is its
    ``latest`` tag.
  * OV2 -- THE POLICY: Bulbe refuses every model that is not verified (the gate raises) and no config weakens it; Daily
    observes by default, each model and reason said once per process however many requests ask.
  * OV3 -- THE ENROLMENT: the engine's listed models pinned under one seal, their layers read from the engine's folder;
    a manifest on the disk that does not hash to the listed digest, or a layer that does not hash to its name, is
    refused by name and nothing is written; a folder that cannot be read pins the digest alone and says so; a pinned
    model whose digest changed is never re-pinned on the way, unless named.
  * OV4 -- THE LAYERS: a blob rewritten under its own name, or one that cannot be read, is named; sound blobs give
    nothing.
  * OV5 -- THE SEAM: the Ollama backend's chat, stream, embedding and batch embedding ask the gate before anything is
    started -- no admission, no request -- and send nothing for a refused model; a verified model passes; the engine's
    list is asked once for several requests within its few seconds.
  * OV6 -- THE GUARDED BACKENDS (supersedes F21 of tests/test_fortress_readiness_contracts.py): Ollama is one of them,
    beside llama_cpp, llama-server and no backend are not; a provenance module that cannot be imported refuses the
    request whenever the mode enforces.
  * OV7 -- THE ESCALATION (supersedes EV7 of tests/test_escalation_preflight_verifies_contracts.py): an ungated
    backend (llama-server) escalates whatever its manifest, which is not read; an Ollama host -- named, or the
    registry's default when none is named -- is held in Daily, told to enrol its Ollama models, while its manifest does
    not verify for the fortress or pins no Ollama model, and escalates once one is pinned under the signature; a
    llama_cpp host whose manifest pins only Ollama models is held too.
  * OV8 -- THE ENROLMENT ROUTE finds the registered Ollama backend even when another one is active.
  * OV9 -- ONE MODE PER REQUEST: the gate follows the mode the backend reads for the request (as its host rule
    does), never a second reading of its own.
The suite runs the real package: the key a test key, the manifest and the engine's folder in a test folder, the mode
named per contract; the engine is a stand-in transport, never reached.
"""
import hashlib
import json
import logging
import sys
import types
from pathlib import Path

import pytest

BUDGET_S = {
    "test_ov1_the_classification": 3.0,
    "test_ov2_the_policy": 3.0,
    "test_ov3_the_enrolment": 4.0,
    "test_ov4_the_layers": 3.0,
    "test_ov4b_a_layer_rewritten_after_its_enrolment": 3.0,
    "test_ov8_the_enrolment_route_finds_the_ollama_backend": 3.0,
    "test_ov9_one_mode_per_request": 3.0,
    "test_ov5_the_seam": 4.0,
    "test_ov6_the_guarded_backends": 3.0,
    "test_ov7_the_escalation": 8.0,
}

_KEY = b"k" * 32


class _World(types.SimpleNamespace):
    pass


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The real provenance module over a test manifest and key, an engine folder holding one model, the mode Bulbe
    unless a contract names another."""
    from opti_oignon import model_provenance as mp

    keys = mp.SealKeys(scheme=mp.SCHEME_HMAC, sign_key=_KEY, verify_key=_KEY)
    monkeypatch.setattr(mp, "resolve_seal_keys", lambda: keys)
    monkeypatch.setattr(mp, "DEFAULT_MANIFEST_PATH", tmp_path / "model_provenance.json")
    monkeypatch.setattr(mp, "current_mode", lambda: "bulbe")
    monkeypatch.setattr(mp, "_load_config", lambda: {})
    monkeypatch.setattr(mp, "_said_ollama", set())
    # The layers' verdict (256) of this test's own: a check in one contract never refuses a model in the next.
    monkeypatch.setattr(mp, "_layers", {"state": "pending", "started_at": None, "finished_at": None, "checked": 0,
                                        "manifest": None})
    monkeypatch.setattr(mp, "_verdicts", {})
    monkeypatch.setattr(mp, "_unchecked", {})
    monkeypatch.setattr(mp, "_said_unchecked", set())
    models = tmp_path / "ollama"
    monkeypatch.setenv("OLLAMA_MODELS", str(models))
    w = _World(mp=mp, keys=keys, models=models, tmp=tmp_path)
    w.digest = _model(models, "llama3", "latest", b"the weights of llama3")
    return w


def _model(models, name, tag, weights, config=b'{"model_format": "gguf"}'):
    """A model in the engine's folder: its config and weights as blobs named by their sha256, its manifest; the
    manifest's sha256, the digest the engine lists for it."""
    blobs = models / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    digests = []
    for data in (config, weights):
        digest = hashlib.sha256(data).hexdigest()
        (blobs / f"sha256-{digest}").write_bytes(data)
        digests.append(digest)
    manifest = json.dumps({"schemaVersion": 2, "config": {"digest": f"sha256:{digests[0]}"},
                           "layers": [{"digest": f"sha256:{digests[1]}"}]}).encode()
    path = models / "manifests" / "registry.ollama.ai" / "library" / name / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(manifest)
    return hashlib.sha256(manifest).hexdigest()


def test_ov1_the_classification(world):
    """OV1 -- enrolled, ``llama3`` listed with its digest is verified (bare or tagged); another digest is a mismatch;
    ``mistral`` never enrolled is unpinned; a digest the engine does not give cannot be compared. A model passed on
    another's digest, or a bare name read apart from its tag -> RED."""
    mp = world.mp
    assert mp.enroll_ollama_models([("llama3", world.digest)])["count"] == 1
    manifest = mp.load_manifest()
    assert mp.classify_ollama("llama3", world.digest, manifest)[0] == mp.REASON_VERIFIED
    assert mp.classify_ollama("llama3:latest", "sha256:" + world.digest, manifest)[0] == mp.REASON_VERIFIED
    assert mp.classify_ollama("llama3", "0" * 64, manifest)[0] == mp.REASON_DIGEST_MISMATCH
    assert mp.classify_ollama("mistral", world.digest, manifest)[0] == mp.REASON_UNPINNED
    assert mp.classify_ollama("llama3", None, manifest)[0] == mp.REASON_NOT_LISTED


def test_ov2_the_policy(world, monkeypatch, caplog):
    """OV2 -- in Bulbe, an unpinned model and a changed one are refused by the gate, a config asking to warn changes
    nothing; in Daily, by default, they pass, said once per model and reason over three requests. A refused model
    sent, or a log line per request -> RED."""
    mp = world.mp
    mp.enroll_ollama_models([("llama3", world.digest)])
    for name, digest in (("mistral", "1" * 64), ("llama3", "2" * 64)):
        with pytest.raises(mp.ProvenanceRefusal):
            mp.guard_ollama(name, digest)
        with pytest.raises(mp.ProvenanceRefusal):
            mp.guard_ollama(name, digest, config={"enforcement": "warn"})
    assert mp.guard_ollama("llama3", world.digest).allowed
    monkeypatch.setattr(mp, "current_mode", lambda: "daily")
    with caplog.at_level(logging.WARNING, logger="opti_oignon.model_provenance"):
        for _ in range(3):
            assert mp.guard_ollama("mistral", "1" * 64).allowed
    said = [r for r in caplog.records if "mistral" in r.getMessage() and "observing" in r.getMessage()]
    assert len(said) == 1, [r.getMessage() for r in caplog.records]


def test_ov3_the_enrolment(world):
    """OV3 -- enrolment pins the listed model with its layers under the seal; a manifest that does not hash to the
    listed digest, or a rewritten layer, refuses by name and writes nothing; no engine folder pins the digest alone,
    said; a changed digest is refused unless re-pinned by name. A model pinned past a broken layer, or a pin moved
    silently -> RED."""
    mp = world.mp
    result = mp.enroll_ollama_models([("llama3", world.digest)])
    entry = mp.load_manifest()["entries"]["ollama:llama3:latest"]
    assert entry["sha256"] == world.digest and len(entry["layers"]) == 2 and result["layers_unchecked"] == []
    before = (world.tmp / "model_provenance.json").read_bytes()
    _model(world.models, "qwen", "latest", b"the weights of qwen")
    with pytest.raises(mp.ProvenanceError, match="hashes to"):
        mp.enroll_ollama_models([("qwen", "3" * 64)])       # the engine lists another digest than its manifest's
    other = _model(world.models, "phi", "latest", b"the weights of phi")
    phi_blob = world.models / "blobs" / f"sha256-{hashlib.sha256(b'the weights of phi').hexdigest()}"
    phi_blob.write_bytes(b"rewritten phi")
    with pytest.raises(mp.ProvenanceError, match="phi:latest"):
        mp.enroll_ollama_models([("phi", other)])
    assert (world.tmp / "model_provenance.json").read_bytes() == before
    result = mp.enroll_ollama_models([("gemma", "4" * 64)], models_dir=world.tmp / "absent")
    assert result["layers_unchecked"] == ["gemma:latest"] and result["count"] == 1
    with pytest.raises(mp.PinnedFileChanged):
        mp.enroll_ollama_models([("gemma", "5" * 64)], models_dir=world.tmp / "absent")
    result = mp.enroll_ollama_models([("gemma", "5" * 64)], models_dir=world.tmp / "absent", repin=["gemma"])
    assert result["repinned"] == ["gemma:latest"]
    with pytest.raises(mp.PinnedFileChanged, match="enroll-ollama-models"):
        mp.enroll_ollama_models([("gemma", "6" * 64)], models_dir=world.tmp / "absent")
    # A name that climbs out of the engine's folder is no model's (review 255-1).
    assert mp.read_ollama_manifest("../../../../etc/passwd") is None
    with pytest.raises(mp.ProvenanceError, match="not a model's name"):
        mp.enroll_ollama_models([("library/../../../outside", "7" * 64)])
    # The GGUF pins are kept beside the Ollama ones; a manifest that does not verify is refused, untouched.
    gguf = world.tmp / "m.gguf"
    gguf.write_bytes(b"GGUF weights")
    mp.enroll_models([gguf])
    mp.enroll_ollama_models([("llama3", world.digest)])
    assert {"m.gguf", "ollama:llama3:latest"} <= set(mp.load_manifest()["entries"])
    forged = world.tmp / "model_provenance.json"
    forged.write_text(json.dumps({"version": 1, "entries": {}, "seal": {"scheme": "hmac-sha512", "value": "0"}}))
    held = forged.read_bytes()
    with pytest.raises(mp.ProvenanceError):
        mp.enroll_ollama_models([("llama3", world.digest)])
    assert forged.read_bytes() == held


def test_ov4_the_layers(world):
    """OV4 -- the layers of the model as enrolled: sound, nothing named; one blob rewritten under its name and one
    removed: both named. A rewritten layer passed as sound -> RED."""
    mp = world.mp
    read = mp.read_ollama_manifest("llama3")
    assert read is not None and read[0] == world.digest
    layers = read[1]
    assert mp.verify_ollama_layers(layers) == []
    (world.models / "blobs" / f"sha256-{layers[1]}").write_bytes(b"other weights")
    (world.models / "blobs" / f"sha256-{layers[0]}").unlink()
    bad = mp.verify_ollama_layers(layers)
    assert len(bad) == 2 and any("unreadable" in b for b in bad) and any("hash to" in b for b in bad), bad


def test_ov4b_a_layer_rewritten_after_its_enrolment(world):
    """OV4 (review 255-1) -- a model enrolled, then a layer rewritten in place: the check on demand names it, and the
    enrolment run again refuses it rather than keep it. A layer rewritten after its enrolment never seen -> RED."""
    mp = world.mp
    mp.enroll_ollama_models([("llama3", world.digest)])
    assert mp.verify_ollama_pins() == []
    layers = mp.read_ollama_manifest("llama3")[1]
    (world.models / "blobs" / f"sha256-{layers[1]}").write_bytes(b"weights rewritten in place")
    found = mp.verify_ollama_pins()
    assert len(found) == 1 and found[0].startswith("llama3:latest: ") and "hash to" in found[0], found
    with pytest.raises(mp.ProvenanceError, match="llama3:latest"):
        mp.enroll_ollama_models([("llama3", world.digest)])


class _Transport:
    """The engine, a stand-in: its list (one model, its digest), its chat and its embedding, each call counted."""

    def __init__(self, calls, digest):
        self.calls, self.digest = calls, digest

    def list(self):
        self.calls.append("list")
        return {"models": [{"model": "llama3:latest", "digest": self.digest}]}

    def chat(self, **kwargs):
        self.calls.append("chat")
        if kwargs.get("stream"):
            return iter([{"message": {"content": "ok"}, "done": True}])
        return {"message": {"content": "ok"}, "done": True}

    def embed(self, **kwargs):
        self.calls.append("embed")
        texts = kwargs["input"] if isinstance(kwargs["input"], list) else [kwargs["input"]]
        return {"embeddings": [[0.1, 0.2] for _ in texts]}


def test_ov5_the_seam(world, monkeypatch):
    """OV5 -- the Ollama backend in Bulbe: for ``llama3`` not enrolled, chat, stream, embed and batch embed raise the
    refusal and start nothing (no admission, no chat, no embedding, not even the engine's list); enrolled, all four
    pass; the engine's list asked once for the requests of a few seconds. A request sent for a refused model, or an admission before the gate ->
    RED."""
    from opti_oignon import inference_backend as ib

    calls = []
    backend = ib.OllamaBackend()
    monkeypatch.setattr(backend, "_transport", lambda timeout=None: _Transport(calls, world.digest))
    monkeypatch.setattr(ib, "OLLAMA_AVAILABLE", True)
    # Hermetic: the mode is the world's, never the lock's; the speculative stats are a file of this test's own.
    from opti_oignon import speculative_decoding as sd

    monkeypatch.setattr(ib, "_provenance_mode", lambda: "bulbe")
    monkeypatch.setattr(sd, "_RESULTS_PATH", world.tmp / "speculative_stats.json")
    monkeypatch.setattr(sd, "_manager", None)
    monkeypatch.setattr(ib, "_governor_admission", lambda model, options=None: calls.append("admit"))
    monkeypatch.setattr(ib, "_governor_note_threads", lambda *args, **kwargs: None)
    asks = (lambda: backend.generate("llama3", [{"role": "user", "content": "hi"}]),
            lambda: list(backend.stream("llama3", [{"role": "user", "content": "hi"}])),
            lambda: backend.embed("llama3", "hi"),
            lambda: backend.embed_many("llama3", ["hi", "there"]))
    for ask in asks:
        with pytest.raises(world.mp.ProvenanceRefusal):
            ask()
    assert calls == [], calls       # an unpinned model is refused without asking the engine anything
    world.mp.enroll_ollama_models([("llama3", world.digest)])
    calls.clear()
    backend._digests = {}
    for ask in asks:
        try:
            ask()
        except world.mp.ProvenanceRefusal:
            raise
        except Exception:  # noqa: BLE001 - what the stand-in engine's answer lacks is not this contract's
            pass
    assert calls.count("list") == 1 and "chat" in calls and "embed" in calls, calls
    # Past its few seconds, the engine's list is asked again (a model swapped under its name is seen).
    import time

    monkeypatch.setattr(backend, "DIGEST_TTL_S", 0.05)
    time.sleep(0.1)
    backend.embed("llama3", "hi")
    assert calls.count("list") == 2, calls
    # A host off the machine in Bulbe is refused for its host, before its provenance is asked (review 255-1).
    monkeypatch.setattr(backend, "_bulbe_refusal", lambda: "Bulbe mode: Ollama requests stay on this machine")
    with pytest.raises(RuntimeError, match="stay on this machine"):
        backend.generate("llama3", [{"role": "user", "content": "hi"}])


def test_ov6_the_guarded_backends(world, monkeypatch):
    """OV6 -- Ollama is a guarded backend beside llama_cpp, and the one with no name is not; with the provenance module
    unimportable, a request in Bulbe is refused before it leaves. An Ollama load the preflight does not count, or a
    request sent without its gate -> RED."""
    from opti_oignon import inference_backend as ib
    from opti_oignon import model_provenance as mp

    assert mp.backend_enforces_provenance("ollama") and mp.backend_enforces_provenance("llama_cpp")
    assert not mp.backend_enforces_provenance("llama_server") and not mp.backend_enforces_provenance(None)
    calls = []
    backend = ib.OllamaBackend()
    monkeypatch.setattr(backend, "_transport", lambda timeout=None: _Transport(calls, world.digest))
    monkeypatch.setattr(ib, "OLLAMA_AVAILABLE", True)
    monkeypatch.setattr(ib, "_provenance_mode", lambda: "bulbe")
    import opti_oignon

    monkeypatch.delattr(opti_oignon, "model_provenance", raising=False)
    monkeypatch.setitem(sys.modules, "opti_oignon.model_provenance", None)
    with pytest.raises(RuntimeError, match="refusing to send a request"):
        backend.embed("llama3", "hi")
    assert "embed" not in calls, calls


def test_ov7_the_escalation(tmp_path):
    """OV7 -- llama-server escalates whatever its manifest, never reading it; an Ollama host (named, or the registry's
    default when backends.yaml names none) whose manifest pins only a GGUF file, or is sealed under another keypair,
    is held in Daily and told to enrol its Ollama models; pinned under the signature, it escalates; a llama_cpp host
    whose manifest pins only Ollama models is held. A host escalated into refusing every request, or an ungated one
    held over a manifest it never reads -> RED."""
    import test_escalation_preflight_verifies_contracts as ev

    for backend, name in (("llama_server", "a_folder"),):
        world = ev._world(tmp_path / f"{backend}_{name}", backend=backend)
        try:
            ev._state(world, name)
            reads = []
            reader = world.prov.read_manifest
            world.prov.read_manifest = lambda *a, **k: reads.append(a) or reader(*a, **k)
            result = ev._escalated(world)
            assert result["success"] is True and world.written == [world.mod.MODE_BULBE], (backend, result)
            assert reads == [], backend
        finally:
            world.restore()
    for backend, name, said in (("ollama", "signed", "pins no Ollama model"), ("ollama", "another_keypair", "refused"),
                                 (None, "signed", "pins no Ollama model")):
        # None: no backend named in backends.yaml -- the registry activates Ollama (review 255-1).
        world = ev._world(tmp_path / f"{backend}_{name}", backend=backend)
        try:
            ev._state(world, name)
            result = ev._escalated(world)
            assert result["success"] is False and world.written == [], (backend, name, result)
            assert said in str(result), (backend, name, result)
            if name == "signed":
                assert "enroll-ollama-models" in str(result), result
        finally:
            world.restore()
    # A llama_cpp host whose manifest pins only Ollama models has no GGUF pinned: held (review 255-1).
    world = ev._world(tmp_path / "llama_cpp_ollama_only", backend="llama_cpp")
    try:
        payload = ev._old_payload({"ollama:llama3:latest": {"sha256": "a" * 64, "kind": "ollama", "layers": [],
                                                             "recorded_at": 1.0}})
        ev._old_write(world.manifest, payload, ev._old_pqc(payload))
        result = ev._escalated(world)
        assert result["success"] is False and "pins no GGUF model" in str(result), result
    finally:
        world.restore()
    world = ev._world(tmp_path / "ollama_pinned", backend="ollama")
    try:
        entries = {ev._NAME: ev._entry(world.model),
                   "ollama:llama3:latest": {"sha256": "a" * 64, "kind": "ollama", "layers": [], "recorded_at": 1.0}}
        payload = ev._old_payload(entries)
        ev._old_write(world.manifest, payload, ev._old_pqc(payload))
        result = ev._escalated(world)
        assert result["success"] is True and world.written == [world.mod.MODE_BULBE], result
    finally:
        world.restore()


def test_ov8_the_enrolment_route_finds_the_ollama_backend(world, monkeypatch):
    """OV8 -- the enrolment route with llama_cpp active and Ollama registered (per-model routing): it enrols what
    the Ollama backend lists. The only way out of a Bulbe refusal saying there is nothing to enrol -> RED."""
    import asyncio

    from opti_oignon import inference_backend as ib
    from opti_oignon.api import routes_security as rs

    class Ollama:
        def listed_digests(self):
            return {"llama3:latest": world.digest}

    class Registry:
        def get(self, name):
            return Ollama() if name == "ollama" else None

        def active_name(self):
            return "llama_cpp"

    monkeypatch.setattr(ib, "get_backend_registry", lambda: Registry())
    monkeypatch.setattr(rs, "PROVENANCE_SEAL_AVAILABLE", True)
    answer = asyncio.run(rs.enroll_ollama_models_route(None))
    assert answer["count"] == 1 and answer["enrolled"][0]["model"] == "llama3:latest", answer


def test_ov9_one_mode_per_request(world, monkeypatch):
    """OV9 -- the provenance module would read Bulbe, the request's mode reads Daily: ``llama3`` not enrolled is
    observed and sent; the other way round, refused before anything starts. A gate reading the mode a second time,
    apart from the request's (two readers that can disagree; a suite fixing one leaves the other on the data) ->
    RED."""
    from opti_oignon import inference_backend as ib
    from opti_oignon import speculative_decoding as sd

    calls = []
    backend = ib.OllamaBackend()
    monkeypatch.setattr(backend, "_transport", lambda timeout=None: _Transport(calls, world.digest))
    monkeypatch.setattr(ib, "OLLAMA_AVAILABLE", True)
    monkeypatch.setattr(sd, "_RESULTS_PATH", world.tmp / "speculative_stats.json")
    monkeypatch.setattr(sd, "_manager", None)
    monkeypatch.setattr(ib, "_governor_admission", lambda model, options=None: calls.append("admit"))
    monkeypatch.setattr(ib, "_governor_note_threads", lambda *args, **kwargs: None)
    monkeypatch.setattr(ib, "_provenance_mode", lambda: "daily")
    assert backend.embed("llama3", "hi") is not None and "embed" in calls, calls
    calls.clear()
    monkeypatch.setattr(world.mp, "current_mode", lambda: "daily")
    monkeypatch.setattr(ib, "_provenance_mode", lambda: "bulbe")
    with pytest.raises(world.mp.ProvenanceRefusal):
        backend.embed("llama3", "hi")
    assert calls == [], calls
