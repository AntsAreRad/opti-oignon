"""Contracts of the Ollama layers checked without anyone asking (MP-01, after 255): the gate compares the digest the
engine lists, never the layers' bytes, so a blob rewritten in place under its name keeps the manifest's digest. A
check hashes every pinned model's layers in the background, at start and every day, and on demand; per model, the
last thing seen of it and when it was read is the verdict the gate reads.

  * LV1 -- THE VERDICT: a layer rewritten after its enrolment, or a blob gone from a folder that reads, the check
    passed: Bulbe refuses that model (``layers_changed``) and passes the others; Daily observes it, said once over
    several requests; a verdict seen under another pin is not this pin's.
  * LV2 -- BEFORE THE FIRST CHECK: the state is pending and the gate has the digest alone; after it, done, the model
    named.
  * LV3 -- CANNOT LOOK: a blob or a folder that cannot be read, or a pass with no key, is not checked, never refused,
    said once -- and never lifts a refusal already seen; a layer rewritten beside one that cannot be read is seen.
  * LV4 -- WHO SAW LAST: the model's enrolment (its layers hashed again and holding) lifts its verdict as of its
    reading, one that could not hash them lifts nothing; a finding replaces a model's verdict only when that model was
    read after it: a slow pass neither writes back a lifted refusal nor lifts a newer one, a change read after a lift
    is kept, a slow enrolment does not lift a change seen after it read the model, and a model re-pinned while a pass
    runs is not said changed by it.
  * LV5 -- THE BACKGROUND: started once per process -- by the API's lifespan (stopped at its end), the core daemon and
    the CLI's chat -- on its own thread at nice 19, a check at once and one every interval; a check that raises is
    said and the next one runs; a stop then a start during a pass leaves a checker running.
  * LV6 -- THE ROUTES AND THE TEXTS: the check on demand and the enrolment hash off the event loop; the state route
    reads the verdict; SECURITY.md and the CHANGELOG say the background check and the window between two checks.
  * LV7 -- A PASS THAT ENDS: a pipe at a blob or at a manifest is seen (not a regular file), never waited on; a
    blob held under a lease is seen, never waited on; a blob not the size its manifest names is seen without being
    read (a huge sparse file never holds the pass); a manifest that is not an object is seen; a model whose check
    raises is not checked, and the pass goes on to the others.
The suite runs the real package: the key a test key, the manifest and the engine's folder in a test folder, the mode
named per contract; no engine is reached.
"""
import ast
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ollama_provenance_contracts import _KEY, _model  # noqa: E402

BUDGET_S = {
    "test_lv1_the_verdict": 3.0,
    "test_lv2_before_the_first_check": 3.0,
    "test_lv3_cannot_look": 3.0,
    "test_lv4_who_saw_last": 6.0,
    "test_lv5_the_background": 6.0,
    "test_lv6_the_routes_and_the_texts": 6.0,
    "test_lv7_a_pass_that_ends": 6.0,
    "test_lv7b_what_cannot_hold_the_pass": 12.0,
}

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = b'{"model_format": "gguf"}'     # the config blob _model writes for every model


class _World(types.SimpleNamespace):
    pass


def _blob(models, data):
    return models / "blobs" / f"sha256-{hashlib.sha256(data).hexdigest()}"


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The real provenance module over a test manifest and key, an engine folder holding two models, the mode Bulbe
    unless a contract names another, and a verdict of this test's own (pending)."""
    from opti_oignon import model_provenance as mp

    keys = mp.SealKeys(scheme=mp.SCHEME_HMAC, sign_key=_KEY, verify_key=_KEY)
    monkeypatch.setattr(mp, "resolve_seal_keys", lambda: keys)
    monkeypatch.setattr(mp, "DEFAULT_MANIFEST_PATH", tmp_path / "model_provenance.json")
    monkeypatch.setattr(mp, "current_mode", lambda: "bulbe")
    monkeypatch.setattr(mp, "_load_config", lambda: {})
    monkeypatch.setattr(mp, "_said_ollama", set())
    monkeypatch.setattr(mp, "_said_unchecked", set())
    monkeypatch.setattr(mp, "_verdicts", {})
    monkeypatch.setattr(mp, "_unchecked", {})
    monkeypatch.setattr(mp, "_checker", {})
    monkeypatch.setattr(mp, "_layers", {"state": "pending", "started_at": None, "finished_at": None, "checked": 0,
                                        "manifest": None})
    models = tmp_path / "ollama"
    monkeypatch.setenv("OLLAMA_MODELS", str(models))
    w = _World(mp=mp, models=models, tmp=tmp_path)
    w.llama = _model(models, "llama3", "latest", b"the weights of llama3")
    w.mistral = _model(models, "mistral", "latest", b"the weights of mistral")
    mp.enroll_ollama_models([("llama3", w.llama), ("mistral", w.mistral)])
    w.blob = _blob(models, b"the weights of llama3")
    w.mistral_blob = _blob(models, b"the weights of mistral")
    w.config_blob = _blob(models, _CONFIG)
    w.original = w.blob.read_bytes()
    yield w
    mp.stop_layer_checks()


def _rewrite(world):
    """llama3's weights rewritten in place, under their own name: the manifest and its digest unchanged."""
    world.blob.write_bytes(b"other weights, same name")


def _refused(world, name, digest):
    """The reason the gate refuses ``name`` in Bulbe for, or None when it passes."""
    try:
        world.mp.guard_ollama(name, digest)
    except world.mp.ProvenanceRefusal as refusal:
        return refusal.decision.reason
    return None


def test_lv1_the_verdict(world, monkeypatch, caplog):
    """LV1 -- llama3's weights rewritten after its enrolment, the check passed: in Bulbe the gate refuses llama3 by
    ``layers_changed`` and passes mistral; in Daily llama3 passes, said once over three requests; mistral's weights
    gone from the folder, the next check refuses it too. A changed layer that still passes in Bulbe, a sound model
    refused, a line per request, or a vanished blob read as not checked -> RED."""
    mp = world.mp
    _rewrite(world)
    result = mp.check_ollama_layers()
    assert set(result["changed"]) == {"llama3:latest"}, result
    assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED
    assert mp.guard_ollama("mistral", world.mistral).allowed
    monkeypatch.setattr(mp, "current_mode", lambda: "daily")
    caplog.clear()      # the check's own line said before; the requests' lines counted below
    with caplog.at_level(logging.WARNING, logger="opti_oignon.model_provenance"):
        for _ in range(3):
            decision = mp.guard_ollama("llama3", world.llama)
            assert decision.allowed and decision.reason == mp.REASON_LAYERS_CHANGED
    said = [r for r in caplog.records if "llama3" in r.getMessage() and "layers_changed" in r.getMessage()]
    assert len(said) == 1, [r.getMessage() for r in caplog.records]
    monkeypatch.setattr(mp, "current_mode", lambda: "bulbe")
    world.mistral_blob.unlink()
    mp.check_ollama_layers()
    assert _refused(world, "mistral", world.mistral) == mp.REASON_LAYERS_CHANGED, mp.layer_check_state()
    # A change seen under another pin (the model re-pinned since) is not this pin's.
    mp._verdicts["mistral:latest"] = (time.monotonic(), "changed", "0" * 64, "seen under an older pin")
    assert mp.guard_ollama("mistral", world.mistral).allowed


def test_lv2_before_the_first_check(world, monkeypatch):
    """LV2 -- the weights rewritten, no check yet: the state is pending and Bulbe passes llama3 on its digest alone;
    while the pass runs, running; after it, done, two models read, llama3 named. A state that says done before any
    check or while one runs, or a check that does not name the model -> RED."""
    mp = world.mp
    _rewrite(world)
    state = mp.layer_check_state()
    assert state["state"] == "pending" and state["finished_at"] is None, state
    assert mp.guard_ollama("llama3", world.llama).allowed
    reached, go, real = _hold(monkeypatch, mp, "_layer_check", "before")
    passing = threading.Thread(target=mp.check_ollama_layers, name="lv-pass")
    passing.start()
    assert reached.wait(10)
    assert mp.layer_check_state()["state"] == "running", mp.layer_check_state()
    go.set()
    passing.join(10)
    monkeypatch.setattr(mp, "_layer_check", real)
    state = mp.layer_check_state()
    assert state["state"] == "done" and state["checked"] == 2 and state["finished_at"] is not None, state
    assert list(state["changed"]) == ["llama3:latest"] and "hash to" in state["changed"]["llama3:latest"], state


def test_lv3_cannot_look(world, monkeypatch, caplog):
    """LV3 -- llama3's blob unreadable: not checked, not refused, said once over two checks. Rewritten and refused,
    then made unreadable, or checked with no key: still refused. mistral's weights rewritten while its config cannot
    be read: refused. The whole folder elsewhere: neither model refused for it. A refusal for what could not be read,
    or a pass that could not look lifting a refusal -> RED."""
    if os.geteuid() == 0:
        pytest.skip("the rights of a file do not stop root")
    mp = world.mp
    world.blob.chmod(0)
    try:
        with caplog.at_level(logging.WARNING, logger="opti_oignon.model_provenance"):
            mp.check_ollama_layers()
            mp.check_ollama_layers()
        state = mp.layer_check_state()
        assert state["changed"] == {} and "llama3:latest" in state["unchecked"], state
        assert "PermissionError" in state["unchecked"]["llama3:latest"], state
        assert mp.guard_ollama("llama3", world.llama).allowed
        said = [r for r in caplog.records if "llama3" in r.getMessage() and "not checked" in r.getMessage()]
        assert len(said) == 1, [r.getMessage() for r in caplog.records]
    finally:
        world.blob.chmod(0o600)
    _rewrite(world)
    mp.check_ollama_layers()
    assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED
    world.blob.chmod(0)
    try:
        mp.check_ollama_layers()
        assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED, mp.layer_check_state()
    finally:
        world.blob.chmod(0o600)
    keys = mp.resolve_seal_keys
    monkeypatch.setattr(mp, "resolve_seal_keys", lambda: None)
    mp.check_ollama_layers()
    assert mp.layer_check_state()["state"] == "unavailable"
    monkeypatch.setattr(mp, "resolve_seal_keys", keys)
    assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED     # the pass with no key lifted nothing
    world.mistral_blob.write_bytes(b"mistral rewritten")
    world.config_blob.chmod(0)
    try:
        mp.check_ollama_layers()
        assert _refused(world, "mistral", world.mistral) == mp.REASON_LAYERS_CHANGED, mp.layer_check_state()
    finally:
        world.config_blob.chmod(0o600)
    world.blob.write_bytes(world.original)
    world.mistral_blob.write_bytes(b"the weights of mistral")
    mp.check_ollama_layers()
    monkeypatch.setenv("OLLAMA_MODELS", str(world.tmp / "elsewhere"))
    mp.check_ollama_layers()
    state = mp.layer_check_state()
    assert state["changed"] == {} and set(state["unchecked"]) == {"llama3:latest", "mistral:latest"}, state
    assert mp.guard_ollama("llama3", world.llama).allowed and mp.guard_ollama("mistral", world.mistral).allowed


def _hold(monkeypatch, mp, attribute, when):
    """``mp.<attribute>`` held ``when`` ("before" or "after") its real call, in the pass's own thread alone (named
    ``lv-pass``): (reached, go, real); the hold ends when ``go`` is set."""
    reached, go = threading.Event(), threading.Event()
    real = getattr(mp, attribute)

    def held(*args, **kwargs):
        mine = threading.current_thread().name == "lv-pass" and (
            attribute != "_model_finding" or args[0] == "llama3:latest")
        if mine and when == "before":
            reached.set()
            assert go.wait(10)
        result = real(*args, **kwargs)
        if mine and when == "after":
            reached.set()
            assert go.wait(10)
        return result

    monkeypatch.setattr(mp, attribute, held)
    return reached, go, real


def test_lv4_who_saw_last(world, monkeypatch):
    """LV4 -- llama3 refused, its weights put back and enrolled: it passes; refused again, an enrolment that could not
    read the folder lifts nothing. Then three passes held mid-way: one that read llama3 changed before its repair and
    enrolment ends -- llama3 still passes; one that read llama3 sound before it was rewritten and a check on demand
    refused it -- still refused; one begun before llama3's repair, enrolment and new rewrite, reading llama3 after
    them -- refused.
    A verdict kept after the enrolment, a blind enrolment lifting, or a slow pass undoing what a later reading saw
    -> RED."""
    mp = world.mp
    _rewrite(world)
    mp.check_ollama_layers()
    assert _refused(world, "llama3", world.llama)
    world.blob.write_bytes(world.original)
    mp.enroll_ollama_models([("llama3", world.llama)])
    assert mp.guard_ollama("llama3", world.llama).allowed
    _rewrite(world)
    mp.check_ollama_layers()
    monkeypatch.setenv("OLLAMA_MODELS", str(world.tmp / "elsewhere"))
    mp.enroll_ollama_models([("llama3", world.llama)])
    monkeypatch.setenv("OLLAMA_MODELS", str(world.models))
    assert _refused(world, "llama3", world.llama), mp.layer_check_state()
    # 1. A pass that read llama3 changed, held; llama3 repaired and enrolled; the pass ends: llama3 passes.
    reached, go, real = _hold(monkeypatch, mp, "_layer_check", "after")
    passing = threading.Thread(target=mp.check_ollama_layers, name="lv-pass")
    passing.start()
    assert reached.wait(10)
    monkeypatch.setattr(mp, "_layer_check", real)
    world.blob.write_bytes(world.original)
    mp.enroll_ollama_models([("llama3", world.llama)])
    go.set()
    passing.join(10)
    assert not passing.is_alive() and mp.guard_ollama("llama3", world.llama).allowed, mp.layer_check_state()
    # 2. A pass that read llama3 sound, held right after reading it; llama3 rewritten and refused on demand; the pass
    # ends: still refused (its reading began before the one on demand).
    reached, go, real = _hold(monkeypatch, mp, "_model_finding", "after")
    passing = threading.Thread(target=mp.check_ollama_layers, name="lv-pass")
    passing.start()
    assert reached.wait(10)
    _rewrite(world)
    mp.verify_ollama_pins()
    assert _refused(world, "llama3", world.llama)
    go.set()
    passing.join(10)
    monkeypatch.setattr(mp, "_model_finding", real)
    assert not passing.is_alive() and _refused(world, "llama3", world.llama), mp.layer_check_state()
    # 3. A pass begun, held before it reads any model; llama3 repaired and enrolled (lifted), then rewritten; the
    # pass then reads it: refused (it read llama3 after the lift, though it began before).
    reached, go, real = _hold(monkeypatch, mp, "read_manifest", "after")
    passing = threading.Thread(target=mp.check_ollama_layers, name="lv-pass")
    passing.start()
    assert reached.wait(10)
    world.blob.write_bytes(world.original)
    mp.enroll_ollama_models([("llama3", world.llama)])
    assert mp.guard_ollama("llama3", world.llama).allowed
    _rewrite(world)
    go.set()
    passing.join(10)
    assert not passing.is_alive() and _refused(world, "llama3", world.llama), mp.layer_check_state()

    monkeypatch.setattr(mp, "read_manifest", real)
    # 4. An enrolment held after it hashed llama3 sound; llama3 rewritten and refused on demand; the enrolment ends:
    # still refused (the lift is as of its reading).
    world.blob.write_bytes(world.original)
    mp.enroll_ollama_models([("llama3", world.llama)])
    reached, go, real = _hold(monkeypatch, mp, "_clear_layer_verdicts", "before")
    enrolling = threading.Thread(target=mp.enroll_ollama_models, args=([("llama3", world.llama)],), name="lv-pass")
    enrolling.start()
    assert reached.wait(10)
    _rewrite(world)
    mp.verify_ollama_pins()
    assert _refused(world, "llama3", world.llama)
    go.set()
    enrolling.join(10)
    assert not enrolling.is_alive() and _refused(world, "llama3", world.llama), mp.layer_check_state()
    # 5. A pass held after it read the pins, before any model; llama3 replaced by other weights and re-pinned by name;
    # the pass then reads the new manifest against the old pin: no change said of llama3, and the new pin passes.
    reached, go, real = _hold(monkeypatch, mp, "read_manifest", "after")
    passing = threading.Thread(target=mp.check_ollama_layers, name="lv-pass")
    passing.start()
    assert reached.wait(10)
    renewed = _model(world.models, "llama3", "latest", b"llama3, its next version")
    mp.enroll_ollama_models([("llama3", renewed)], repin=["llama3"])
    go.set()
    passing.join(10)
    monkeypatch.setattr(mp, "read_manifest", real)
    assert not passing.is_alive()
    assert "llama3:latest" not in mp.layer_check_state()["changed"], mp.layer_check_state()
    assert mp.guard_ollama("llama3", renewed).allowed

def test_lv5_the_background(world, monkeypatch, caplog):
    """LV5 -- started: a thread named for the check, set to nice 19 by its own native id, a check at once, another
    after the interval; a check that raises is said and the next one runs; a second start starts nothing; a stop then
    a start while a pass runs leaves a checker running; stopped, the thread ends. The API's lifespan starts it before
    it serves and stops it after; the core daemon's serve and the CLI's chat session start it. A check that never runs
    alone, a failure that ends the thread, a process serving the gate with no check, or a stop and start leaving none
    -> RED."""
    mp = world.mp
    calls, priorities = [], []
    real = mp.check_ollama_layers
    real_priority = os.setpriority

    def counted(**kwargs):
        calls.append(time.monotonic())
        if len(calls) == 2:
            raise OSError("a disk that fails once")
        return real(**kwargs)

    def priority(which, who, value):
        priorities.append((which, who, value))
        return real_priority(which, who, value)

    monkeypatch.setattr(mp, "check_ollama_layers", counted)
    monkeypatch.setattr(os, "setpriority", priority)
    # A check at once, not after the interval.
    monkeypatch.setattr(mp, "LAYER_CHECK_INTERVAL_S", 3600.0)
    assert mp.start_layer_checks() is True
    deadline = time.monotonic() + 5
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(calls) == 1, calls
    first = mp._checker["thread"]
    mp.stop_layer_checks()
    first.join(5)
    assert not first.is_alive()
    monkeypatch.setattr(mp, "LAYER_CHECK_INTERVAL_S", 0.05)
    _rewrite(world)
    with caplog.at_level(logging.ERROR, logger="opti_oignon.model_provenance"):
        assert mp.start_layer_checks() is True
        assert mp.start_layer_checks() is False
        deadline = time.monotonic() + 5
        while (len(calls) < 4 or "llama3:latest" not in mp.layer_check_state()["changed"]) \
                and time.monotonic() < deadline:
            time.sleep(0.01)
    thread = mp._checker["thread"]
    assert thread.name == "ollama-layer-check" and thread.daemon and thread.is_alive()
    assert len(calls) >= 3, calls
    assert any("a disk that fails once" in r.getMessage() for r in caplog.records)
    # The last pass to end said done (a pass may be under way meanwhile: the state then says running).
    assert mp._layers["state"] == "done" and "llama3:latest" in mp.layer_check_state()["changed"]
    assert mp.layer_check_state()["state"] in ("done", "running")
    if hasattr(os, "PRIO_PROCESS"):
        assert (os.PRIO_PROCESS, thread.native_id, 19) in priorities, priorities
    mp.stop_layer_checks()
    thread.join(5)
    assert not thread.is_alive()
    # A stop then a start while a pass is under way: the old thread ends after its pass, a new one runs.
    entered, go = threading.Event(), threading.Event()

    def slow(**kwargs):
        entered.set()
        go.wait(10)
        return real(**kwargs)

    monkeypatch.setattr(mp, "check_ollama_layers", slow)
    assert mp.start_layer_checks() is True
    assert entered.wait(5)
    old = mp._checker["thread"]
    mp.stop_layer_checks()
    assert mp.start_layer_checks() is True
    go.set()
    old.join(5)
    assert not old.is_alive() and mp._checker["thread"].is_alive()
    mp.stop_layer_checks()
    mp._checker["thread"].join(5)

    def body(relative, name, kind=ast.FunctionDef):
        text = (_ROOT / relative).read_text(encoding="utf-8")
        node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, kind) and n.name == name)
        return ast.get_source_segment(text, node)

    # Read from their files, never imported (an import of the app reads the data places at once).
    source = body("opti_oignon/api/app.py", "lifespan", ast.AsyncFunctionDef)
    assert source.index("start_layer_checks()") < source.index("    yield\n") < source.index("stop_layer_checks()")
    serve = body("opti_oignon/core_daemon.py", "serve")
    assert serve.index("start_layer_checks()") < serve.index("serve_forever()")
    chat = body("opti_oignon/cli/main.py", "_default_chat_session")
    assert chat.index("start_layer_checks()") < chat.index("return ChatSession(")


def test_lv6_the_routes_and_the_texts(world, monkeypatch):
    """LV6 -- the check on demand and the enrolment run off the event loop; the check's verdict is the one the state
    route then reads, which reads nothing on the disk; SECURITY.md and the CHANGELOG say the background check, its
    interval and the window between two checks. A route that hashes on the loop, or texts that still say nothing runs
    it -> RED."""
    mp = world.mp
    from opti_oignon import inference_backend as ib
    from opti_oignon.api import routes_security as rs

    monkeypatch.setattr(rs, "PROVENANCE_SEAL_AVAILABLE", True)
    seen = {}
    real_verify, real_enroll = mp.verify_ollama_pins, mp.enroll_ollama_models

    def verify(**kwargs):
        seen["verify"] = threading.current_thread()
        return real_verify(**kwargs)

    def enroll(*args, **kwargs):
        seen["enroll"] = threading.current_thread()
        return real_enroll(*args, **kwargs)

    class Ollama:
        def listed_digests(self):
            seen["list"] = threading.current_thread()
            return {"llama3:latest": world.llama}

    class Registry:
        def get(self, name):
            return Ollama() if name == "ollama" else None

    monkeypatch.setattr(mp, "verify_ollama_pins", verify)
    monkeypatch.setattr(mp, "enroll_ollama_models", enroll)
    monkeypatch.setattr(ib, "get_backend_registry", lambda: Registry())
    _rewrite(world)

    async def ask(route, *args):
        seen["loop"] = threading.current_thread()
        return await route(*args)

    answer = asyncio.run(ask(rs.verify_ollama_models_route))
    assert answer["success"] is False and answer["problems"][0].startswith("llama3:latest: "), answer
    assert seen["verify"] is not seen["loop"], seen
    with pytest.raises(rs.HTTPException):         # llama3's rewritten layer refuses its enrolment
        asyncio.run(ask(rs.enroll_ollama_models_route, rs.EnrollOllamaModelsRequest()))
    assert seen["enroll"] is not seen["loop"] and seen["list"] is not seen["loop"], seen
    monkeypatch.setenv("OLLAMA_MODELS", str(world.tmp / "elsewhere"))
    state = asyncio.run(rs.ollama_layers_route())
    assert state["state"] == "done" and list(state["changed"]) == ["llama3:latest"], state
    security = " ".join((_ROOT / "SECURITY.md").read_text(encoding="utf-8").split())
    assert "hashed in the background at start and every day" in security
    assert "between two checks" in security
    changelog = " ".join((_ROOT / "CHANGELOG.md").read_text(encoding="utf-8").split("\n## ")[1].split())
    assert "GET /api/security/pqc/ollama-layers" in changelog and "between two checks" in changelog


def test_lv7_a_pass_that_ends(world, monkeypatch):
    """LV7 -- a pipe put at llama3's blob: the pass ends within seconds and llama3 is refused as not a regular file;
    mistral's manifest made ``null``: refused as not a manifest, llama3 still read; a check of llama3 that raises:
    llama3 not checked, mistral still read. A pass that waits on a pipe, or one model ending the pass -> RED."""
    mp = world.mp
    world.blob.unlink()
    os.mkfifo(world.blob)
    passing = threading.Thread(target=mp.check_ollama_layers, daemon=True)
    passing.start()
    passing.join(5)
    assert not passing.is_alive(), "the pass waits on a pipe"
    assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED
    assert "a pipe" in mp.layer_check_state()["changed"]["llama3:latest"], mp.layer_check_state()
    world.blob.unlink()
    world.blob.write_bytes(world.original)
    manifest = world.models / "manifests" / "registry.ollama.ai" / "library" / "mistral" / "latest"
    sound = manifest.read_bytes()
    manifest.write_bytes(b"null")
    result = mp.check_ollama_layers()
    assert "not an object" in result["changed"].get("mistral:latest", ""), result
    assert "llama3:latest" not in result["changed"] and "llama3:latest" not in result["unchecked"], result
    manifest.write_bytes(sound)
    real = mp._model_finding

    def failing(name, *args):
        if name == "llama3:latest":
            raise RuntimeError("a check that fails")
        return real(name, *args)

    monkeypatch.setattr(mp, "_model_finding", failing)
    result = mp.check_ollama_layers()
    assert "a check that fails" in result["unchecked"].get("llama3:latest", ""), result
    assert result["checked"] == 2 and "mistral:latest" not in result["unchecked"], result
    # A pipe at a manifest: seen, never waited on.
    manifest = world.models / "manifests" / "registry.ollama.ai" / "library" / "mistral" / "latest"
    sound = manifest.read_bytes()
    manifest.unlink()
    os.mkfifo(manifest)
    passing = threading.Thread(target=mp.check_ollama_layers, daemon=True)
    passing.start()
    passing.join(5)
    assert not passing.is_alive(), "the pass waits on a pipe"
    assert "a pipe" in mp.layer_check_state()["changed"].get("mistral:latest", ""), mp.layer_check_state()
    manifest.unlink()
    manifest.write_bytes(sound)


def _sized_model(models, name, weights):
    """A model whose manifest names its layers' sizes, as the engine writes them; its listed digest."""
    blobs = models / "blobs"
    layers = []
    for data in (_CONFIG, weights):
        digest = hashlib.sha256(data).hexdigest()
        (blobs / f"sha256-{digest}").write_bytes(data)
        layers.append({"digest": f"sha256:{digest}", "size": len(data)})
    manifest = json.dumps({"schemaVersion": 2, "config": layers[0], "layers": layers[1:]}).encode()
    path = models / "manifests" / "registry.ollama.ai" / "library" / name / "latest"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(manifest)
    return hashlib.sha256(manifest).hexdigest()


_LEASE = """
import fcntl, os, signal, sys, time
signal.signal(signal.SIGIO, signal.SIG_IGN)
fd = os.open(sys.argv[1], os.O_RDWR)
fcntl.fcntl(fd, fcntl.F_SETLEASE, fcntl.F_WRLCK)
print("held", flush=True)
time.sleep(30)
"""


def test_lv7b_what_cannot_hold_the_pass(world):
    """LV7 -- phi's weights made a 64 GiB sparse file: the pass ends within seconds, phi refused by its size (read
    without hashing), llama3 still read; llama3's blob rewritten and held under a write lease by another process: the
    pass ends at once and llama3 is refused, never read as not checked. A pass held by a huge file, or a lease that
    turns a rewritten blob into 'not checked' -> RED."""
    if not hasattr(os, "truncate"):
        pytest.skip("no sparse files here")
    mp = world.mp
    phi = _sized_model(world.models, "phi", b"the weights of phi")
    mp.enroll_ollama_models([("phi", phi)])
    os.truncate(_blob(world.models, b"the weights of phi"), 64 * 1024 ** 3)
    passing = threading.Thread(target=mp.check_ollama_layers, daemon=True)
    started = time.monotonic()
    passing.start()
    passing.join(10)
    assert not passing.is_alive(), "a huge sparse file holds the pass"
    assert time.monotonic() - started < 10
    state = mp.layer_check_state()
    assert "its size is" in state["changed"].get("phi:latest", ""), state
    assert _refused(world, "phi", phi) == mp.REASON_LAYERS_CHANGED
    assert "llama3:latest" not in state["changed"] and "llama3:latest" not in state["unchecked"], state
    if not sys.platform.startswith("linux"):
        return
    _rewrite(world)
    holder = subprocess.Popen([sys.executable, "-c", _LEASE, str(world.blob)], stdout=subprocess.PIPE, text=True)
    try:
        if holder.stdout.readline().strip() != "held":
            pytest.skip("no write lease could be taken here")
        result = mp.check_ollama_layers()
        assert "lease" in result["changed"].get("llama3:latest", ""), result
        assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED
    finally:
        holder.kill()
        holder.wait(5)
