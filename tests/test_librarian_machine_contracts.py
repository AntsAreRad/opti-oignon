#!/usr/bin/env python3
"""Contracts for the librarian on the machine it runs on: its model, its context, its neighbour's residence.

The librarian asks a model for summaries off the interactive path, as a
background caller of the resource governor. Three things depend on the
machine rather than on the shipped files: which model it has (the shipped
name is a placeholder), whether that model is the one the chat already
holds, and how long that resident should stay:

  * LM1 -- THE MACHINE'S OWN MODEL: ``onion: {librarian: {model: <name>}}``
    in ``local.yaml``, which git ignores, names the librarian's model,
    stripped, with or without the switch; it sets no switch and says
    nothing of one. A blank name, a number, a list, a mapping without a
    model and a key neither section takes leave the shipped model, said
    once each; a path given to the loader is read as it stands, and the
    shipped file stays whole.
  * LM2 -- EVERY RUNG: the shipped file states, and the governor's defaults
    hold, a floor for the librarian at the ladder's smallest rung, so a
    model resident at 4096 asked 8192 by the librarian serves it at 4096,
    loading nothing, and a load of its own model that fits only at 4096 is
    admitted there, in free memory; without that floor the first is refused
    (``background_never_reloads``), which is how M2 measured nothing at
    4096, and the second too.
  * LM3 -- NEVER SHORTENS A RESIDENCE: the engine keeps a model for the
    keep-alive of the last call that ended on it, so a background call on a
    load the background did not make -- the chat's, or one loaded out of the
    governor's sight -- carries the residence the foreground gives its loads
    (the warm-up's keep-alive, which the chat sends with each turn), and the
    librarian sends it in place of its run's shorter one. A load the
    background made, and no higher class has used, keeps its run's
    residence, a pressure override still wins, a foreground call carries
    nothing (its own path picks it), and with no warm-up to read nothing is
    carried.
  * LM4 -- THE REAL WIRING: the real governor, on its shipped file, behind
    the librarian's run: the backend is asked at the context the resident
    holds, with the chat's residence, while the ticket the governor handed
    out is the one active on its thread, and let go once it returned.

The governor and the onion's modules are loaded from source through the
shared isolation window, with a seeded ``db_utils``; snapshots are injected,
so no reading of the machine enters a decision.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
import tempfile
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_lm1_the_machine_names_its_librarians_model_in_its_own_file": 5.0,
    "test_lm2_a_resident_serves_the_librarian_at_the_context_it_holds": 5.0,
    "test_lm3_a_librarians_call_never_shortens_the_residence_of_the_model_it_shares": 5.0,
    "test_lm4_the_real_governor_serves_the_real_librarian_at_the_residents_context_and_residence": 5.0,
}

_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
_LIBRARIAN = "opti_oignon.memory.librarian"
_COMPOSER = "opti_oignon.memory.composer"
_PEELS = "opti_oignon.memory.peels"
_WRAPPER = "opti_oignon.agent.untrusted_context"
_RG = "opti_oignon.resource_governor"
_SHIPPED_ONION = REPO / "opti_oignon" / "config" / "onion.yaml"
_GIB = 1024 ** 3


def _db_utils():
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(str(p), check_same_thread=kw.get("check_same_thread", False))
    return db


def _window(tmp_path):
    """The onion's modules and the governor from source; ``onion.yaml`` a copy of the shipped one in ``tmp_path``."""
    targets = {f"opti_oignon.memory.{name}": source("memory", f"{name}.py") for name in _MODULES}
    targets[_WRAPPER] = source("agent", "untrusted_context.py")
    targets[_RG] = source("resource_governor.py")
    loaded, restore = isolate(
        targets=targets,
        blocked=("opti_oignon.inference_backend", "opti_oignon.native", "opti_oignon.context_manager",
                 "opti_oignon.emergency_stop"),
        seeded={"opti_oignon.db_utils": _db_utils()},
        packages=("opti_oignon.memory", "opti_oignon.agent"),
    )
    onion = tmp_path / "onion.yaml"
    onion.write_text(_SHIPPED_ONION.read_text(encoding="utf-8"), encoding="utf-8")
    for name in (_COMPOSER, _PEELS, _LIBRARIAN):
        loaded[name]._CONFIG = onion
    loaded[_LIBRARIAN].reset_librarian()
    loaded[_LIBRARIAN]._LOCAL_SAID.clear()
    return loaded, onion, restore


def _local(tmp_path, text):
    path = tmp_path / "local.yaml"
    if text is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(text, encoding="utf-8")


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class _Hardware:
    """A hardware profile that reads no pressure at all."""

    def others_cpu_pressure(self):
        return None

    def pressure(self):
        return None

    def placement(self):
        return None


_WARMUP = types.SimpleNamespace(keep_alive="30m")


def _shipped_governor(rg, *, clock=None, warmup=_WARMUP):
    """A real governor on its shipped file, ``m`` resident at 4096 on a roomy card; its warm-up keeps the
    foreground's loads for 30 minutes, as the chat's turns ask."""
    cfg = rg.load_config(Path(source("config", "resource_governor.yaml")))
    clock = clock or _Clock()
    gov = rg.ResourceGovernor(
        config_path="/nonexistent-resource-governor-config",
        db_path=str(Path(tempfile.mkdtemp()) / "governor.db"),
        warmup=warmup,
        registry=None,
        clock=clock,
        meminfo_path="/nonexistent-meminfo",
        hardware=_Hardware(),
    )
    gov._config = cfg
    held = rg.LoadedModelView(name="m", size_vram_bytes=4 * _GIB, context_length=4096,
                              expires_at=time.time() + 1500.0)
    gov._snapshot = rg.ResourceSnapshot(taken_at=clock(), ttl_s=9999.0, loaded=[held], capacity_gb=24.0,
                                        vram_in_use_gb=4.0, ram_available_mb=64000.0)
    return gov, cfg


class _Backend:
    """A backend that answers each span word for word and records what each call carried, and under which ticket."""

    def __init__(self, active=None):
        self.calls = []
        self.active = active

    def generate(self, model, messages, options=None, keep_alive="30m", think=False, images=None):
        self.calls.append({"model": model, "num_ctx": dict(options or {}).get("num_ctx"), "keep_alive": keep_alive,
                           "thread": threading.get_ident(),
                           "ticket": None if self.active is None else self.active()})
        return types.SimpleNamespace(content="Alice reviewed service 1.")


def _span():
    return [{"turn_id": "t0001", "role": "user", "text": "Alice reviewed service 1 on 2026-03-02."}]


def _config(lib, **over):
    fields = dict(enabled=True, model="m", keep_alive="5m", min_new_turns=4, temperature=0.0, num_predict=64,
                  num_ctx=8192, window_margin=0.25, run_budget_s=300.0, call_timeout_s=5.0)
    fields.update(over)
    return lib.LibrarianConfig(**fields)


# ---------------------------------------------------------------------------
# LM1 -- the machine's own model
# ---------------------------------------------------------------------------
def test_lm1_the_machine_names_its_librarians_model_in_its_own_file(tmp_path, caplog):
    """LM1: ``onion.librarian.model`` in local.yaml names the librarian's model; nothing else does."""
    loaded, onion, restore = _window(tmp_path)
    lib = loaded[_LIBRARIAN]
    try:
        shipped_text = onion.read_text(encoding="utf-8")
        shipped = lib.load_config(onion).model
        assert shipped and shipped != "qwen3.5:4b", "the shipped model is a placeholder of its own"
        seen = {}
        for case, text in (
            ("no local file", None),
            ("a model alone", "onion:\n  librarian:\n    model: 'qwen3.5:4b'\n"),
            ("a model and the switch", "onion:\n  enabled: true\n  librarian:\n    model: '  qwen3.5:4b  '\n"),
            ("a null librarian", "onion:\n  enabled: true\n  librarian:\n"),
            ("a blank name", "onion:\n  librarian:\n    model: '   '\n"),
            ("a number", "onion:\n  librarian:\n    model: 4\n"),
            ("a list", "onion:\n  librarian:\n    model: [qwen3.5:4b]\n"),
            ("no model key", "onion:\n  librarian:\n    name: qwen3.5:4b\n"),
            ("not a mapping", "onion:\n  librarian: qwen3.5:4b\n"),
            ("a key the section does not take", "onion:\n  librarian:\n    model: qwen3.5:4b\n    num_ctx: 4096\n"),
        ):
            _local(tmp_path, text)
            config = lib.load_config()
            seen[case] = (config.model, config.enabled)
        assert seen == {
            "no local file": (shipped, False),
            "a model alone": ("qwen3.5:4b", False),
            "a model and the switch": ("qwen3.5:4b", True),
            "a null librarian": (shipped, True),
            "a blank name": (shipped, False),
            "a number": (shipped, False),
            "a list": (shipped, False),
            "no model key": (shipped, False),
            "not a mapping": (shipped, False),
            "a key the section does not take": ("qwen3.5:4b", False),
        }, ("only a name under onion.librarian.model names the librarian's model", seen)

        lib._LOCAL_SAID.clear()
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _local(tmp_path, "onion:\n  librarian:\n    model: qwen3.5:4b\n")
            lib.load_config()
            lib.load_config()
        assert not [r for r in caplog.records if "local.yaml" in r.getMessage()], (
            "a model named without the switch sets no switch and says nothing of one")
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _local(tmp_path, "onion:\n  librarian:\n    model: 7\n")
            lib.load_config()
            lib.load_config()
            _local(tmp_path, "onion:\n  enable: true\n  librarian:\n    model: qwen3.5:4b\n")
            lib.load_config()
            lib.load_config()
            _local(tmp_path, "onion:\n  enabled:\n  librarian:\n    model: qwen3.5:4b\n")
            config = lib.load_config()
            lib.load_config()
        said = [r.getMessage() for r in caplog.records if "local.yaml" in r.getMessage()]
        assert len(said) == 3, ("each reason is said once", said)
        assert "onion.librarian.model" in said[0] and "enable" in said[1], said
        assert "onion.enabled" in said[2] and (config.model, config.enabled) == ("qwen3.5:4b", False), (
            "a switch written but not a boolean is said, beside a model too", said)

        _local(tmp_path, "onion:\n  librarian:\n    model: qwen3.5:4b\n")
        assert lib.load_config(onion).model == shipped, "a path given to the loader is read as it stands"
        assert onion.read_text(encoding="utf-8") == shipped_text, "the shipped file stays whole"
        onion.write_text(shipped_text.replace("\nlibrarian:\n", "\nlibrarian_gone:\n", 1), encoding="utf-8")
        try:
            lib.load_config()
        except lib.LibrarianError:
            pass
        else:
            raise AssertionError("a local model does not stand in for a shipped file that lost its librarian")
    finally:
        restore()


# ---------------------------------------------------------------------------
# LM2 -- served at the context the resident holds
# ---------------------------------------------------------------------------
def test_lm2_a_resident_serves_the_librarian_at_the_context_it_holds(tmp_path):
    """LM2: the shipped floor lets the chat's resident serve the librarian at 4096; without it, a refusal."""
    loaded, _onion, restore = _window(tmp_path)
    rg = loaded[_RG]
    try:
        import yaml

        defaults = rg.GovernorConfig()
        gov, shipped = _shipped_governor(rg)
        written = yaml.safe_load(Path(source("config", "resource_governor.yaml")).read_text(encoding="utf-8"))
        assert written["ctx_floor"].get("librarian") == min(written["ctx_ladder"]), (
            "the shipped file states the librarian's floor itself", written["ctx_floor"])
        for name, cfg in (("shipped file", shipped), ("defaults", defaults)):
            assert cfg.ctx_floor.get("librarian") == min(cfg.ctx_ladder), (
                f"the {name} give the librarian a floor at the ladder's smallest rung", cfg.ctx_floor, cfg.ctx_ladder)
        assert shipped.class_of("librarian") == "background"
        served = gov.admit("m", 8192, caller="librarian")
        assert (served.admitted, served.action, served.num_ctx, served.load_expected) == (
            True, "downsize", 4096, False), ("served by the resident at the context it holds", served)
        # A load of the librarian's own model, 4 GiB of weights, on a card with 6.5 GiB to spare: 8192 needs
        # 8 GiB, 4096 needs 6.
        shipped.weights_override_models = {"lib": 4.0}
        gov._snapshot = rg.ResourceSnapshot(taken_at=gov._clock(), ttl_s=9999.0, loaded=[], capacity_gb=8.0,
                                            vram_in_use_gb=0.0, ram_available_mb=64000.0)
        loaded_small = gov.admit("lib", 8192, caller="librarian")
        assert (loaded_small.admitted, loaded_small.num_ctx, loaded_small.load_expected) == (True, 4096, True), (
            "a load of its own that fits only at 4096 is admitted there", loaded_small)
        shipped.ctx_floor.pop("librarian")
        assert gov.admit("lib", 8192, caller="librarian").admitted is False, "without its floor, refused"
        held = rg.LoadedModelView(name="m", size_vram_bytes=4 * _GIB, context_length=4096,
                                  expires_at=time.time() + 1500.0)
        gov._snapshot = rg.ResourceSnapshot(taken_at=gov._clock(), ttl_s=9999.0, loaded=[held], capacity_gb=24.0,
                                            vram_in_use_gb=4.0, ram_available_mb=64000.0)
        refused = gov.admit("m", 8192, caller="librarian")
        assert (refused.admitted, refused.reason) == (False, "background_never_reloads"), (
            "without its floor the librarian is refused the reload", refused)
    finally:
        restore()


# ---------------------------------------------------------------------------
# LM3 -- never shortens a residence
# ---------------------------------------------------------------------------
def test_lm3_a_librarians_call_never_shortens_the_residence_of_the_model_it_shares(tmp_path):
    """LM3: a background call on a load it does not own carries the foreground's residence; its own guest does not."""
    loaded, _onion, restore = _window(tmp_path)
    rg, lib = loaded[_RG], loaded[_LIBRARIAN]
    try:
        gov, shipped = _shipped_governor(rg)
        unseen = gov.admit("m", 4096, caller="librarian")
        assert (unseen.reason, unseen.keep_alive) == ("fits_resident", "30m"), (
            "a load out of the governor's sight keeps the foreground's residence", unseen)
        gov.note_loaded_by("m", "interactive")
        chats = gov.admit("m", 8192, caller="librarian")
        assert (chats.action, chats.num_ctx, chats.keep_alive) == ("downsize", 4096, "30m"), (
            "the chat's load keeps the chat's residence", chats)
        gov.note_loaded_by("m", "background")
        guest = gov.admit("m", 4096, caller="librarian")
        assert (guest.admitted, guest.keep_alive) == (True, None), ("its own guest keeps its run's residence", guest)

        gov.note_loaded_by("m", "interactive")
        held = rg.LoadedModelView(name="m", size_vram_bytes=4 * _GIB, context_length=4096,
                                  expires_at=time.time() + 1500.0)
        gov._snapshot = rg.ResourceSnapshot(taken_at=gov._clock(), ttl_s=9999.0, loaded=[held], capacity_gb=24.0,
                                            vram_in_use_gb=23.0, ram_available_mb=64000.0)
        pressed = gov.admit("m", 4096, caller="librarian")
        assert pressed.keep_alive == shipped.pressure_keep_alive and pressed.keep_alive != "30m", (
            "a pressure override still wins", pressed)
        gov._snapshot = rg.ResourceSnapshot(taken_at=gov._clock(), ttl_s=9999.0, loaded=[held], capacity_gb=24.0,
                                            vram_in_use_gb=4.0, ram_available_mb=64000.0)
        # Last: an interactive admission holds the background behind it for its grace.
        assert gov.admit("m", 4096, caller="chat").keep_alive is None, "the foreground's own path picks it itself"
        bare, _cfg = _shipped_governor(rg, warmup=None)
        assert bare.admit("m", 4096, caller="librarian").keep_alive is None, "no warm-up, nothing to carry"

        class _Governor:
            def __init__(self, decision):
                self.decision = decision

            def admit_or_wait(self, model, requested_ctx=None, caller="benchmark", **_kw):
                return self.decision

            def end_pending_load(self, ticket_id):
                pass

            def release_guest(self, model):
                return False

        def _sent(keep_alive):
            decision = types.SimpleNamespace(admitted=True, model="m", num_ctx=4096, keep_alive=keep_alive,
                                             ticket_id="k1")
            governance = types.SimpleNamespace(get_resource_governor=lambda: _Governor(decision),
                                               ticket_scope=rg.ticket_scope)
            backend = _Backend()
            config = _config(lib)
            run = lib._Run(config, governance=governance)
            lib.registry_summarizer(config, resolve=lambda model: backend, run=run)(_span())
            return backend.calls[0]["keep_alive"]

        assert _sent("30m") == "30m", "the librarian sends the residence its admission carries"
        assert _sent(None) == "5m", "and its run's own when the admission carries none"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LM4 -- the real wiring
# ---------------------------------------------------------------------------
def test_lm4_the_real_governor_serves_the_real_librarian_at_the_residents_context_and_residence(tmp_path):
    """LM4: the shipped governor behind the librarian's run: asked at 4096, with the chat's residence."""
    loaded, _onion, restore = _window(tmp_path)
    rg, lib = loaded[_RG], loaded[_LIBRARIAN]
    try:
        gov, _shipped = _shipped_governor(rg)
        gov.note_loaded_by("m", "interactive")
        seen, after = [], []

        @contextmanager
        def _scope(decision):
            seen.append(decision)
            with rg.ticket_scope(decision):
                yield
            after.append((threading.get_ident(), rg.get_active_ticket()))

        backend = _Backend(active=rg.get_active_ticket)
        config = _config(lib)
        run = lib._Run(config, governance=types.SimpleNamespace(get_resource_governor=lambda: gov,
                                                                ticket_scope=_scope))
        lib.registry_summarizer(config, resolve=lambda model: backend, run=run)(_span())
        assert len(backend.calls) == 1, backend.calls
        call = backend.calls[0]
        assert call["num_ctx"] == 4096, ("asked at the context the resident holds, no reload", call)
        assert call["keep_alive"] == "30m", ("the chat's residence kept, not the run's 5 minutes", call)
        assert len(seen) == 1 and seen[0].caller == "librarian" and seen[0].action == "downsize", seen
        assert call["ticket"] is seen[0], ("the backend answers under the ticket the governor handed out", call)
        assert after == [(call["thread"], None)], ("let go on the thread that held it, once it returned", after)
    finally:
        restore()
