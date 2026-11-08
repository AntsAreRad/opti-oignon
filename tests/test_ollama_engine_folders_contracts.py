"""Contracts of the layer check reading the engine's folder (MP-01, after 257): the background check of the pinned
Ollama layers (256) and the enrolment read the folder this process names; the official Linux install keeps the
engine's models under its own account in /usr/share/ollama/.ollama/models, which the application did not read, so
every pinned model there was "not checked". When OLLAMA_MODELS is not set, both folders are read now, and a model is
judged in each folder whose manifest holds its pinned digest.

  * MD1 -- THE SYSTEM INSTALL'S FOLDER: OLLAMA_MODELS unset, the models in the system install's folder: the enrolment
    hashes their layers there (none left unchecked), the check finds them holding, and a layer rewritten there is
    refused under Bulbe while the other model passes. The folders read: the one given; else the one OLLAMA_MODELS
    names when set; else ~/.ollama/models and the system install's.
  * MD2 -- BOTH FOLDERS HOLD THE PIN: the model is judged in each; a layer rewritten in either copy is a change, the
    folder named.
  * MD3 -- AN OLD COPY: a manifest at another digest beside a folder that holds the pinned one is never a refusal, but
    it keeps the verdict from "holds" (not checked, the way out said), so a change seen before stays until it is
    removed; the old copy alone is a change, as when one folder is read.
  * MD4 -- A FOLDER THAT CANNOT BE LOOKED AT: beside a copy that holds, a folder whose manifest cannot be read keeps
    the verdict from "holds" (not checked, never refused); the enrolment then says the model's layers not checked
    and lifts no change seen before.
  * MD5 -- THE ENROLMENT: a pinned model kept, its layer rewritten in the system install's folder, is refused, nothing
    written, even with a sound copy in ~/.ollama/models; a model whose only manifest found is at another digest is
    refused, the folder named with the way out; a pipe at a manifest beside the copy that holds keeps the model, lifts
    nothing, and says it (``kept_unchecked``); a layer that cannot be read leaves the model's layers not checked, never
    a refusal.
  * MD7 -- A MANIFEST THAT IS NOT A FILE: a pipe at the model's manifest is a change wherever it is, beside a copy that
    holds in either folder; a manifest that does not parse beside one that holds is an old copy (not checked).
  * MD6 -- THE TEXTS: SECURITY.md says both folders are read when OLLAMA_MODELS is not set and the engine under
    another account covered when its folder reads; the CHANGELOG says it; the startup check of 257 names the folders
    the layer check reads.
The suite runs the real package: the key a test key, the manifest and both folders in a test folder (the system
install's folder moved there through the module's constant, HOME too), the mode named per contract; no engine is
reached, no folder outside the test's own is read.
"""
import hashlib
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ollama_provenance_contracts import _KEY, _model  # noqa: E402

BUDGET_S = {
    "test_md1_the_system_installs_folder": 3.0,
    "test_md2_both_folders_hold_the_pin": 3.0,
    "test_md3_an_old_copy": 3.0,
    "test_md4_a_folder_that_cannot_be_looked_at": 3.0,
    "test_md5_the_enrolment": 3.0,
    "test_md6_the_texts": 3.0,
    "test_md7_a_manifest_that_is_not_a_file": 3.0,
}

_ROOT = Path(__file__).resolve().parents[1]


class _World(types.SimpleNamespace):
    pass


def _blob(models, data):
    return models / "blobs" / f"sha256-{hashlib.sha256(data).hexdigest()}"


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The real provenance module over a test manifest and key, OLLAMA_MODELS unset, HOME and the system install's
    folder in the test's own folder, two models in the system install's folder, enrolled; the mode Bulbe."""
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
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    system = tmp_path / "system"
    monkeypatch.setattr(mp, "OLLAMA_SYSTEM_MODELS_DIR", system)
    w = _World(mp=mp, system=system, home=tmp_path / "home" / ".ollama" / "models", tmp=tmp_path)
    w.llama = _model(system, "llama3", "latest", b"the weights of llama3")
    w.mistral = _model(system, "mistral", "latest", b"the weights of mistral")
    w.enrolled = mp.enroll_ollama_models([("llama3", w.llama), ("mistral", w.mistral)])
    w.blob = _blob(system, b"the weights of llama3")
    yield w
    mp.stop_layer_checks()


def _refused(world, name, digest):
    """The reason the gate refuses ``name`` in Bulbe for, or None when it passes."""
    try:
        world.mp.guard_ollama(name, digest)
    except world.mp.ProvenanceRefusal as refusal:
        return refusal.decision.reason
    return None


def test_md1_the_system_installs_folder(world, monkeypatch, tmp_path):
    mp = world.mp
    assert mp.layer_check_folders() == [world.home, world.system]
    assert world.enrolled["layers_unchecked"] == [] and world.enrolled["count"] == 2, world.enrolled
    result = mp.check_ollama_layers()
    assert result["changed"] == {} and result["unchecked"] == {}, result
    assert mp.guard_ollama("llama3", world.llama).allowed
    world.blob.write_bytes(b"other weights, same name")
    result = mp.check_ollama_layers()
    assert set(result["changed"]) == {"llama3:latest"} and str(world.system) in result["changed"]["llama3:latest"], (
        result)
    assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED
    assert mp.guard_ollama("mistral", world.mistral).allowed
    # The folders read: the one given, else the one OLLAMA_MODELS names, else both.
    assert mp.layer_check_folders(tmp_path / "given") == [tmp_path / "given"]
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "named"))
    assert mp.layer_check_folders() == [tmp_path / "named"]
    monkeypatch.setenv("OLLAMA_MODELS", "")
    assert mp.layer_check_folders() == [world.home, world.system], "an empty OLLAMA_MODELS names no folder"
    monkeypatch.setattr(mp.sys, "platform", "darwin")
    monkeypatch.delenv("OLLAMA_MODELS")
    assert mp.layer_check_folders() == [world.home]


def test_md2_both_folders_hold_the_pin(world):
    mp = world.mp
    assert _model(world.home, "llama3", "latest", b"the weights of llama3") == world.llama
    result = mp.check_ollama_layers()
    assert result["changed"] == {} and result["unchecked"] == {}, result
    copy = _blob(world.home, b"the weights of llama3")
    copy.write_bytes(b"other weights, same name")
    result = mp.check_ollama_layers()
    assert set(result["changed"]) == {"llama3:latest"}, result
    why = result["changed"]["llama3:latest"]
    assert str(world.home) in why and str(world.system) not in why, why
    assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED


def test_md3_an_old_copy(world):
    mp = world.mp
    old = _model(world.home, "llama3", "latest", b"the weights of an older llama3")
    assert old != world.llama
    result = mp.check_ollama_layers()
    assert result["changed"] == {} and "llama3:latest" in result["unchecked"], result
    why = result["unchecked"]["llama3:latest"]
    assert f"{world.home}: its manifest on the disk hashes to {old}" in why and "an old copy" in why, why
    assert mp.guard_ollama("llama3", world.llama).allowed, "an old copy is never a refusal"
    # A change seen before stays while the old copy is there: it keeps the verdict from "holds".
    world.blob.write_bytes(b"other weights, same name")
    assert set(mp.check_ollama_layers()["changed"]) == {"llama3:latest"}
    world.blob.write_bytes(b"the weights of llama3")
    assert "llama3:latest" in mp.check_ollama_layers()["unchecked"]
    assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED
    home_manifest = world.home / "manifests" / "registry.ollama.ai" / "library" / "llama3" / "latest"
    home_manifest.unlink()
    assert mp.check_ollama_layers()["changed"] == {} and mp.guard_ollama("llama3", world.llama).allowed
    # The old copy alone: what was found does not hold.
    home_manifest.write_bytes(b'{"schemaVersion": 2, "layers": []}')
    old = hashlib.sha256(home_manifest.read_bytes()).hexdigest()
    manifest = world.system / "manifests" / "registry.ollama.ai" / "library" / "llama3" / "latest"
    manifest.unlink()
    result = mp.check_ollama_layers()
    assert set(result["changed"]) == {"llama3:latest"}, result
    assert result["changed"]["llama3:latest"] == (f"{world.home}: its manifest on the disk hashes to {old}, pinned "
                                                  f"{world.llama}"), result


def test_md4_a_folder_that_cannot_be_looked_at(world):
    mp = world.mp
    assert _model(world.home, "llama3", "latest", b"the weights of llama3") == world.llama
    world.blob.write_bytes(b"other weights, same name")
    assert set(mp.check_ollama_layers()["changed"]) == {"llama3:latest"}
    shut = world.system / "manifests"
    shut.chmod(0o000)
    try:
        result = mp.check_ollama_layers()
        assert "llama3:latest" in result["unchecked"] and result["changed"] == {}, result
        assert str(world.system) in result["unchecked"]["llama3:latest"], result
        assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED, "not checked lifts nothing"
        enrolled = mp.enroll_ollama_models([("llama3", world.llama)])
        assert enrolled["kept"] == ["llama3:latest"], enrolled
        assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED, (
            "an enrolment that could not look at every folder lifts no change")
        mp.stop_layer_checks()
        fresh = mp.enroll_ollama_models([("mistral2", _model(world.home, "mistral2", "latest", b"m2"))])
        assert fresh["layers_unchecked"] == ["mistral2:latest"] and fresh["count"] == 1, fresh
    finally:
        shut.chmod(0o755)


def test_md5_the_enrolment(world):
    mp = world.mp
    world.blob.write_bytes(b"other weights, same name")
    before = (world.tmp / "model_provenance.json").read_bytes()
    with pytest.raises(mp.ProvenanceError, match="not enrolled") as raised:
        mp.enroll_ollama_models([("llama3", world.llama)])
    assert str(world.system) in str(raised.value), str(raised.value)
    assert (world.tmp / "model_provenance.json").read_bytes() == before
    # Only an old copy at another digest: refused, the folder named with the way out.
    gemma = _model(world.home, "gemma", "latest", b"the weights of gemma")
    with pytest.raises(mp.ProvenanceError) as raised:
        mp.enroll_ollama_models([("gemma", "7" * 64)])
    said = str(raised.value)
    assert f"its manifest in {world.home} on the disk hashes to {gemma}, the engine lists {'7' * 64}" in said, said
    assert "point OLLAMA_MODELS at the folder the engine reads" in said, said
    assert (world.tmp / "model_provenance.json").read_bytes() == before
    # A sound copy in ~/.ollama/models does not hide the rewritten one in the system install's folder.
    assert _model(world.home, "llama3", "latest", b"the weights of llama3") == world.llama
    with pytest.raises(mp.ProvenanceError, match="not enrolled") as raised:
        mp.enroll_ollama_models([("llama3", world.llama)])
    assert str(world.system) in str(raised.value), str(raised.value)
    # A change seen, the layer put back, a pipe at the other folder's manifest: kept, nothing lifted, said.
    assert set(mp.check_ollama_layers()["changed"]) == {"llama3:latest"}
    world.blob.write_bytes(b"the weights of llama3")
    home_manifest = world.home / "manifests" / "registry.ollama.ai" / "library" / "llama3" / "latest"
    home_manifest.unlink()
    os.mkfifo(home_manifest)
    result = mp.enroll_ollama_models([("llama3", world.llama)])
    assert result["kept"] == ["llama3:latest"] and result["kept_unchecked"] == ["llama3:latest"], result
    assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED, "an enrolment beside a pipe lifted"
    home_manifest.unlink()
    # A layer this process cannot read: the model enrolled, its layers not checked -- never a refusal.
    digest = _model(world.system, "phi", "latest", b"the weights of phi")
    blob = _blob(world.system, b"the weights of phi")
    blob.chmod(0o000)
    try:
        if not os.access(blob, os.R_OK):
            fresh = mp.enroll_ollama_models([("phi", digest)])
            assert fresh["count"] == 1 and fresh["layers_unchecked"] == ["phi:latest"], fresh
    finally:
        blob.chmod(0o644)


def test_md6_the_texts(world, monkeypatch):
    mp = world.mp
    security = " ".join((_ROOT / "SECURITY.md").read_text(encoding="utf-8").split())
    for said in ("both `~/.ollama/models` and, on Linux, the system install's", "an old copy, never a refusal",
                 "covered when its folder can be read"):
        assert said in security, said
    changelog = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    added = changelog.split("### Added\n", 1)[1].split("\n### ", 1)[0]
    entry = " ".join(next((entry for entry in added.split("\n- ") if "MD1-MD7" in entry), "").split())
    assert "OLLAMA_MODELS" in entry and "/usr/share/ollama/.ollama/models" in entry, entry[:300]

    from opti_oignon import startup_checks as sc
    monkeypatch.setattr(sc, "_reach_walk", [])
    monkeypatch.setattr(mp, "_euid", lambda: os.geteuid() + 1)
    monkeypatch.setattr(mp, "_capabilities", lambda: set())
    monkeypatch.setattr(mp, "_carried_groups", lambda: set())
    monkeypatch.setattr(mp, "_missing_groups", lambda euid: (set(), None))
    monkeypatch.setattr(mp, "_access", lambda path, mode: False if mode & os.W_OK else os.access(path, mode))
    item = sc._check_ollama_models_writable()
    assert item.detail.startswith(f"No way for this account to write it found: {world.system}"), item.detail
    assert "The background check of the layers reads it." in item.detail and not item.tips, item
    monkeypatch.setenv("OLLAMA_MODELS", str(world.tmp / "named"))
    monkeypatch.setattr(sc, "_reach_walk", [])
    item = sc._check_ollama_models_writable()
    assert f"The background check of the layers reads {world.tmp / 'named'}, not this folder." in item.detail, item
    named = world.tmp / "named"
    _model(named, "llama3", "latest", b"the weights of llama3")
    monkeypatch.setattr(sc, "_reach_walk", [])
    item = sc._check_ollama_models_writable()
    assert f"The background check of the layers reads {named}, not {world.system}." in item.detail, item.detail


def test_md7_a_manifest_that_is_not_a_file(world):
    mp = world.mp
    path = ("manifests", "registry.ollama.ai", "library", "llama3", "latest")
    for holding, piped in ((world.system, world.home), (world.home, world.system)):
        if holding == world.home:
            assert _model(world.home, "llama3", "latest", b"the weights of llama3") == world.llama
        pipe = piped.joinpath(*path)
        pipe.parent.mkdir(parents=True, exist_ok=True)
        if pipe.exists():
            pipe.unlink()
        os.mkfifo(pipe)
        result = mp.check_ollama_layers()
        assert set(result["changed"]) == {"llama3:latest"}, (holding, result)
        assert "pipe" in result["changed"]["llama3:latest"] and str(piped) in result["changed"]["llama3:latest"]
        assert _refused(world, "llama3", world.llama) == mp.REASON_LAYERS_CHANGED
        pipe.unlink()
    # A manifest that does not parse beside the copy that holds: an old copy, not checked, never refused.
    for folder in (world.home, world.system):
        if not folder.joinpath(*path).exists():
            assert _model(folder, "llama3", "latest", b"the weights of llama3") == world.llama
    world.home.joinpath(*path).write_bytes(b"not a manifest")
    world.mp._verdicts.clear()
    result = mp.check_ollama_layers()
    assert result["changed"] == {} and "an old copy" in result["unchecked"].get("llama3:latest", ""), result
    assert mp.guard_ollama("llama3", world.llama).allowed
