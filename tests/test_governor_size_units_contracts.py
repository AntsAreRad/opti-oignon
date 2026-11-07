#!/usr/bin/env python3
"""Contracts for the parameter sizes the resource governor reads in millions.

Ollama shows the small models' sizes in millions of parameters: ``271.00M``
for embeddinggemma-2, ``307.58M`` for embeddinggemma, ``595.78M`` for
qwen3-embedding. The governor read only billions, so such a label sized the
model at nothing, its static estimate was skipped, and with no cost observed
or learned on a new machine a background load of the Mycelium's encoder was
refused for good (``background_cost_unknown``): the Mycelium's vector pass
never started (found by the replay of the shipped Mycelium, 7 November).
  * GS1 -- THE LABELS: billions (``7B``, ``3.2b``, ``7``, a number), millions
    (``271.00M``) and thousands (``1.5K``) are read as the billions they
    make; a label that is no size (none, empty, ``M``, ``-3M``, ``1.2.3M``,
    ``7G``) is 0.0, never a size.
  * GS2 -- THE ESTIMATE: an engine that says ``271.00M`` at ``Q8_0`` and
    nothing else gives the governor a static estimate of that size, its basis
    said; one that says no size gives none.
  * GS3 -- THE ENCODER IS ADMITTED: the index's background load of an encoder
    sized in millions, never seen loaded, on a card with room, is no longer
    refused as a cost unknown.
Each contract opens the shared isolation window of ``tests/_isolation.py`` on
the governor's source and the static table's (``speculative_decoding``), with
a seeded ``db_utils``; snapshots are injected and
the store is a temporary file, so no reading of the machine enters a decision.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_gs1_the_labels_in_billions_millions_and_thousands": 3.0,
    "test_gs2_an_engine_sized_in_millions_gives_a_static_estimate": 5.0,
    "test_gs3_the_encoder_sized_in_millions_is_admitted_in_the_background": 5.0,
}

_RG = "opti_oignon.resource_governor"


def _db_utils():
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(str(p), check_same_thread=kw.get("check_same_thread", False))
    return db


def _open():
    # The static table is read from its home module, loaded from source in the window too (it imports no project
    # module): the governor's estimate is the shipped table's figure, never a stand-in's.
    loaded, restore = isolate(
        targets={"opti_oignon.speculative_decoding": source("speculative_decoding.py"), _RG: source("resource_governor.py")},
        blocked=("opti_oignon.context_manager", "opti_oignon.emergency_stop"),
        seeded={"opti_oignon.db_utils": _db_utils()},
    )
    return loaded[_RG], restore


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class _Hardware:
    def others_cpu_pressure(self):
        return None

    def pressure(self):
        return None

    def placement(self):
        return None


class _Engine:
    """An engine that says each model's parameter size and quantization, as Ollama's ``show`` does, and no file."""

    name = "ollama"

    def __init__(self, sizes):
        self._sizes = sizes

    def model_info(self, model):
        said = self._sizes.get(model)
        if said is None:
            return None
        size, quant = said
        return types.SimpleNamespace(parameter_size=size, quantization_level=quant, path=None, size=None, extra={})


class _Registry:
    def __init__(self, *engines):
        self._engines = list(engines)

    def backends(self):
        return list(self._engines)

    def resolve_backend(self, model):
        return self._engines[0] if self._engines else None


def _governor(rg, registry):
    gov = rg.ResourceGovernor(
        config_path="/nonexistent-resource-governor-config",
        db_path=str(Path(tempfile.mkdtemp()) / "governor.db"),
        warmup=None,
        registry=registry,
        clock=_Clock(),
        meminfo_path="/nonexistent-meminfo",
        hardware=_Hardware(),
    )
    gov._config = rg.load_config(Path(source("config", "resource_governor.yaml")))
    gov.kv_geometry = lambda model, engine=None: None
    return gov


def _snap(rg, gov, capacity):
    gov._snapshot = rg.ResourceSnapshot(taken_at=1000.0, ttl_s=9999.0, loaded=[], capacity_gb=capacity,
                                        vram_in_use_gb=0.0, ram_available_mb=64000.0, ram_total_mb=96000.0,
                                        sources=["S1"])


def test_gs1_the_labels_in_billions_millions_and_thousands():
    """GS1 -- each label read as the billions it makes; a label that is no size is 0.0. ``271.00M`` read as nothing,
    or as 271 billions, -> RED."""
    rg, restore = _open()
    try:
        parse = rg._parse_parameter_size_b
        for label, billions in (("7B", 7.0), ("3.2b", 3.2), ("7", 7.0), (7, 7.0), (7.5, 7.5), ("271.00M", 0.271),
                                ("595.78M", 0.59578), ("307.58 M", 0.30758), ("1.5K", 0.0000015),
                                ("1500M", 1.5), (" 4.7B ", 4.7)):
            assert abs(parse(label) - billions) < 1e-12, (label, parse(label))
        for label in (None, "", "M", "B", "-3M", "1.2.3M", "7G", "abc", "7MB", [7]):
            assert parse(label) == 0.0, (label, parse(label))
    finally:
        restore()


def test_gs2_an_engine_sized_in_millions_gives_a_static_estimate():
    """GS2 -- an engine that says ``271.00M`` at ``Q8_0``, with no file and no size: the governor's estimate is its
    static table's figure for 0.271 billion at Q8_0, basis ``static_table``; a model the engine sizes at nothing gives
    no estimate. An encoder sized in millions left unknown -> RED."""
    rg, restore = _open()
    try:
        engine = _Engine({"embeddinggemma-2": ("271.00M", "Q8_0"), "blank": ("", "Q8_0")})
        gov = _governor(rg, _Registry(engine))
        _snap(rg, gov, 8.0)
        estimate, basis = gov.estimate_model_vram_gb("embeddinggemma-2")
        assert basis == "static_table" and estimate is not None, (estimate, basis)
        assert abs(estimate - gov._estimator.estimate_model_vram(0.271, "Q8_0")) < 1e-9, estimate
        assert 0.1 < estimate < 1.0, estimate
        assert gov.estimate_model_vram_gb("blank") == (None, "unknown")
    finally:
        restore()


def test_gs3_the_encoder_sized_in_millions_is_admitted_in_the_background():
    """GS3 -- the Mycelium's vector pass asks as the index (a background caller) for an encoder sized ``271.00M``,
    never seen loaded nor learned, on an 8 GiB card with nothing loaded: the decision is no refusal for a cost
    unknown. The pass refused for good on a machine that never loaded its encoder -> RED."""
    rg, restore = _open()
    try:
        engine = _Engine({"embeddinggemma-2": ("271.00M", "Q8_0")})
        gov = _governor(rg, _Registry(engine))
        _snap(rg, gov, 8.0)
        decision = gov.admit("embeddinggemma-2", caller="index", engine="ollama")
        assert getattr(decision, "reason", "") != "background_cost_unknown", decision
        assert decision.admitted, decision
    finally:
        restore()
