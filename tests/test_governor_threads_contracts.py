#!/usr/bin/env python3
"""The CPU threads an engine computes with, planned from the machine's cores.

No admission said how many CPU threads an engine should compute with. A
model split between the GPU and system RAM, or loaded on a machine with no
card, ran with whatever count its engine picked: Ollama's own choice, or
llama-cpp-python's defaults (half the logical CPUs for the tokens, all of
them for the prompt, SMT siblings included), with no core left to the user's
programs and no regard for a cgroup quota. And Ollama reloads a resident
model for any num_thread other than the one it was loaded with, so a count
that changed from one call to the next would cost a reload each time.

  The plan, carried by the decision:
    * GT1 -- a split load carries the physical cores less the reserve, for
      the tokens and for the prompt, said "plan".
    * GT2 -- a load the GPU holds whole carries no count: the engine picks.
    * GT3 -- a machine with no card computes on the CPU and carries the
      plan; a card whose memory is unknown carries none.
    * GT4 -- never an SMT sibling, never past the cgroup quota, never under
      one thread. Superseded by GT35 and GT36: the server's own quota binds
      what runs in its process, not an engine in a process of its own.
    * GT5 -- the reserve is the file's fraction of the physical cores,
      rounded up, within its floor and ceiling, and never every core.
    * GT6 -- "fast cores only" counts no more than the fastest class, and is
      off by default.
    * GT7 -- a model the file names takes its own count, whatever the CPUs.
    * GT8 -- the plan is the model's, not the class's: a warm-up or a
      background load gets the interactive count.

  The count a resident model was loaded with:
    * GT9 -- the decisions for a resident model carry the count its load was
      told, said "pinned", whatever the plan says by then.
    * GT10 -- the pin ends with the resident, and the next load takes the
      plan again.
    * GT11 -- a caller's own num_thread is the count pinned, and Ollama is
      told nothing on top of it.
    * GT12 -- a call joining a pending load carries that load's count.
    * GT13 -- unknown CPUs, the plan switched off, or the governor disabled:
      no count, and the engine picks its own.
    * GT14 -- the shipped file's threads block holds the decided defaults,
      reads back as written, and holds its ranges.

  The engine heads that tell it:
    * GT15 -- the Ollama generate head tells the count the admission carries.
    * GT16 -- the Ollama stream head tells it too.
    * GT17 -- the Ollama embed head tells it, and sends no options without one.
    * GT18 -- the Ollama embed_many head tells it, likewise.
    * GT19 -- the warm-up tells the plan's count, and the keep-alive ping the
      count the resident was pinned at.
    * GT20 -- a caller's own num_thread is sent as it is, and no count is
      sent where the admission carries none.
    * GT21 -- llama.cpp in process loads with n_threads and n_threads_batch
      from the admission, unless backends.yaml names its own.

  The background's own budget:
    * GT22 -- the background workers get the CPUs of the cores outside the
      reserve, the highest-ranked cores; one worker per core outside it,
      within the file's ceiling and the quota less the reserve.
    * GT23 -- the shipped file's background block holds the decided
      defaults, reads back as written, and holds its ranges.
    * GT24 -- the governor names a caller's class as its file does, and
      tells a refusal no wait can lift from one a wait can.

  Where the model computes, and the count measured there:
    * GT25 -- the decision says where the CPU computes: "split:<n>", n the
      layers on the GPU; "cpu" on a machine with no card; None where the
      card holds the model whole; a resident and a joining call say the
      placement of their load.
    * GT26 -- a count kept for (model, engine, placement) is that triple's
      plan, said "measured"; another triple, a model the file names, a plan
      fallen under the count or another machine take the plan. Superseded
      by GT37, whose plan falls under the count by its reserve.
    * GT27 -- kept counts outlive the governor; a row that does not hold
      together is ignored; no count past the plan, for an unknown placement
      or engine, or without the CPUs, is written.
    * GT28 -- a count a call sends to an Ollama resident is the count pinned
      from then on: Ollama reloads the model for it.

  The threads on the status surface:
    * GT29 -- the threads section is the governor's own calls: the plan's
      count and source, the physical cores, the reserve, the fast-cores
      switch, the quota, and the background's budget with the CPUs it
      leaves to the user. Superseded by GT38, which adds the server's own
      cap and plans past the server's quota.
    * GT30 -- each resident's pinned count, prompt count and placement,
      until its pin ends.
    * GT31 -- the kept counts, newest first, each said measured on the CPUs
      read now or not; a row that does not hold together is not listed; the
      list stops at the file's limit.
    * GT32 -- the status route carries the threads section and the
      background pool's state, creates no pool, and a section that fails
      says so alone.

  A waiter's cancel, and memory for background work:
    * GT33 -- a waiter whose caller cancels leaves the queue at its next
      wake, refused "cancelled", its place given back; a cancel already set
      enqueues nothing.
    * GT34 -- background work is told memory is short while the kernel
      reports memory pressure or the RAM available is under the reserve.

  Whose CPUs bound the count:
    * GT35 -- an engine in a process of its own is planned on the machine's
      physical cores less the reserve, whatever the server's affinity and
      quota; never an SMT sibling, never under one thread.
    * GT36 -- the server's own cap (its affinity's cores less the reserve,
      within its quota) bounds what computes in its process: llama.cpp in
      process loads within it; an Ollama head tells the plan as it is.
    * GT37 -- GT26 again, the plan falling under the kept count by its
      reserve.
    * GT38 -- GT29 again, with the server's own cap, the plan made past the
      server's quota.
    * GT39 -- the sweep's fastest-class candidate is counted on the
      machine's cores, as the plan it measures against is.
    * GT40 -- a refusal a caller's own cancel made stays out of the
      refusal-rate window.

  Memory for background work, continued:
    * GT41 -- the room the governor gives background work is the RAM
      available less the reserve; none under memory pressure; None where
      the RAM cannot be read.
    * GT42 -- the parse factors per kind of file hold their shipped values
      and their ranges, and the background plan carries them.

Everything here is proven in the container, on a scripted CPU topology and
hand-set snapshots, through the shared isolation window. The count that
makes a given engine fastest on a given machine, and what a reserve of cores
spares the user's other programs, are owed to the machine.
"""

import json
import logging
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import types
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_RG = "opti_oignon.resource_governor"
_BACKEND = "opti_oignon.inference_backend"
_WARMUP = "opti_oignon.model_warmup"
# The two seams the governor resolves at the call: unreachable when the window
# opens, so neither an emergency stop nor a model window enters a decision.
_SEAMS = ("opti_oignon.context_manager", "opti_oignon.emergency_stop")
_GIB = 1024 ** 3
_CLOSERS = []


def _db_utils():
    """A db_utils stand-in whose safe_connect is plain sqlite."""
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(
        str(p), check_same_thread=kw.get("check_same_thread", False)
    )
    return db


def _open(*extra, packages=()):
    """The governor and ``extra`` (name, path) modules after it, in one window."""
    targets = {_RG: source("resource_governor.py")}
    for name, path in extra:
        targets[name] = path
    loaded, restore = isolate(
        targets=targets,
        blocked=_SEAMS,
        seeded={"opti_oignon.db_utils": _db_utils()},
        packages=packages,
    )
    _CLOSERS.append(restore)
    return loaded


def _rg():
    return _open()[_RG]


def _tmpdir(prefix):
    """A temporary directory the contract's teardown removes."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    _CLOSERS.append(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


def _project_modules():
    """Every project entry of the module cache, by identity."""
    return {k: v for k, v in sys.modules.items() if k == "opti_oignon" or k.startswith("opti_oignon.")}


@pytest.fixture(autouse=True)
def _left_as_found():
    """No contract may leave a project module or the HTTP transport changed."""
    before = _project_modules()
    urlopen = urllib.request.urlopen
    yield
    urllib.request.urlopen = urlopen
    while _CLOSERS:
        _CLOSERS.pop()()
    assert _project_modules() == before, "every project module is left as the contract found it"


class _Clock:
    """A fixed monotonic stand-in: a hand-set snapshot never goes stale."""

    def __call__(self) -> float:
        return 1000.0


def _meminfo(directory: Path, ram_mb: float) -> str:
    path = directory / "meminfo"
    path.write_text(f"MemTotal:       {int(ram_mb * 1024)} kB\nMemAvailable:   {int(ram_mb * 1024)} kB\n", encoding="utf-8")
    return str(path)


def _topology(physical=12, *, smt=True, fast=4, quota=None):
    """A CPU topology as the governor reads it: the usable CPUs, the
    physical cores with their classes (the first ``fast`` in class 0), and
    the cgroup quota."""
    cores = tuple(
        types.SimpleNamespace(
            cpus=(n, n + physical) if smt else (n,), rank=None, perf_class=0 if n < fast else 1
        )
        for n in range(physical)
    )
    usable = tuple(sorted(cpu for core in cores for cpu in core.cpus))
    return types.SimpleNamespace(
        usable=usable,
        cores=cores,
        physical=physical,
        smt=smt,
        l3=(),
        numa=(),
        quota_cpus=quota,
        class_source="acpi_cppc",
    )


class _Hardware:
    """A hardware profile that answers the CPUs, the server's own view and
    the machine's (the same unless given apart); its cards read nothing."""

    def __init__(self, topology=None, *, cards_absent=False, machine=None):
        self.topology = topology
        self.machine = machine
        self._absent = cards_absent

    def cpu_topology(self):
        return self.topology

    def machine_cpu_topology(self):
        return self.machine if self.machine is not None else self.topology

    def cards_absent(self):
        return self._absent

    def placement(self):
        return None

    def pressure(self):
        return None

    def others_cpu_pressure(self):
        return None


class _Warmup:
    """The warmup as the governor reads it: a keep_alive and a loaded set."""

    def __init__(self, loaded=None, keep_alive="10m"):
        self.keep_alive = keep_alive
        self._loaded = list(loaded or [])

    def get_loaded_models(self):
        return list(self._loaded)


def _resident(name, size_vram, size=None, context_length=None):
    return types.SimpleNamespace(
        name=name, size_vram=size_vram, size=size, expires_at=None, context_length=context_length, digest=None
    )


def _config(rg, *, weights=None, capacity=10.0, **fields):
    """10 GiB card, 1.5 GiB margin: 8.5 GiB free on an empty card."""
    cfg = rg.GovernorConfig(
        total_vram_gb=capacity,
        safety_margin_gb=1.5,
        kv_coefficient=0.5,
        ctx_ladder=[8192, 4096, 2048],
        ctx_floor={"chat": 2048},
    )
    cfg.weights_override_models = dict(weights or {})
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


def _governor(rg, config=None, *, hardware=None, warmup=None, ram_mb=64000.0):
    tmp = _tmpdir("gt-")
    gov = rg.ResourceGovernor(
        config_path=str(tmp / "missing.yaml"),
        db_path=str(tmp / "governor.db"),
        warmup=warmup,
        registry=None,
        clock=_Clock(),
        meminfo_path=_meminfo(tmp, ram_mb),
        vram_probe=None,
        hardware=hardware,
    )
    if config is not None:
        gov._config = config
    return gov


def _snapshot(rg, gov, *, capacity=10.0, in_use=0.0, loaded=None, ram_mb=64000.0, cards_absent=False):
    """A hand-set snapshot the admission reads as it is."""
    gov._snapshot = rg.ResourceSnapshot(
        taken_at=1000.0,
        ttl_s=9999.0,
        loaded=list(loaded or []),
        capacity_gb=capacity,
        vram_in_use_gb=in_use,
        ram_available_mb=ram_mb,
        cards_absent=cards_absent,
    )
    return gov._snapshot


def _split_admission(rg, gov, model="big"):
    """9.0 GiB of weights and 2.0 of KV at 4096: 8.5 on the GPU, 2.5 in RAM."""
    _snapshot(rg, gov)
    decision = gov.admit(model, requested_ctx=4096, caller="chat")
    assert decision.partial_offload is True
    return decision


# ---------------------------------------------------------------------------
# GT1-GT8 -- the plan, carried by the decision
# ---------------------------------------------------------------------------


def test_gt1_a_split_load_carries_the_physical_cores_less_the_reserve():
    """GT1 -- a load the GPU cannot hold whole is split, and the CPU computes
    its share: the decision carries a count for the tokens and one for the
    prompt, both the physical cores less the reserve (twelve cores, an
    eighth reserved rounded up: two), said "plan", and its record says so.
    Leaving the engine to pick its own -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg, weights={"big": 9.0}), hardware=_Hardware(_topology(12)))
    decision = _split_admission(rg, gov)
    assert (decision.threads, decision.threads_batch, decision.threads_source) == (10, 10, "plan")
    record = decision.to_dict()
    assert (record["threads"], record["threads_batch"], record["threads_source"]) == (10, 10, "plan")


def test_gt2_a_load_the_gpu_holds_whole_carries_no_count():
    """GT2 -- a model the GPU holds whole computes there: no count is
    carried, and the engine picks its own. Planning threads for every load
    -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg, weights={"small": 4.0}), hardware=_Hardware(_topology(12)))
    _snapshot(rg, gov)
    decision = gov.admit("small", requested_ctx=4096, caller="chat")
    assert decision.admitted is True and decision.gpu_share is None
    assert (decision.threads, decision.threads_batch, decision.threads_source) == (None, None, None)


def test_gt3_a_machine_with_no_card_carries_the_plan_and_an_unknown_card_none():
    """GT3 -- on a machine the profile knows has no card, every load
    computes on the CPU and carries the plan; with a card whose memory is
    unknown, where the engine places the model is unknown, and no count is
    carried. Planning for the unknown card, or not for the CPU alone -> RED."""
    rg = _rg()
    cpu_only = _governor(rg, _config(rg, weights={"m": 4.0}, capacity=None), hardware=_Hardware(_topology(12), cards_absent=True))
    _snapshot(rg, cpu_only, capacity=None, cards_absent=True)
    unknown = _governor(rg, _config(rg, weights={"m": 4.0}, capacity=None), hardware=_Hardware(_topology(12)))
    _snapshot(rg, unknown, capacity=None)
    seen = [gov.admit("m", requested_ctx=4096, caller="chat") for gov in (cpu_only, unknown)]
    assert [d.admitted for d in seen] == [True, True]
    assert [(d.threads, d.threads_batch, d.threads_source) for d in seen] == [(10, 10, "plan"), (None, None, None)]


def test_gt4_never_an_smt_sibling_never_past_the_quota_never_under_one_thread():
    """GT4 -- twelve cores of two siblings plan ten threads, not twenty-two;
    a cgroup quota of 4 CPUs caps the count at 4, one of 1.5 at 1, one of
    0.4 still leaves 1. Counting the logical CPUs, or ignoring the quota
    -> RED."""
    rg = _rg()
    counts = []
    for topology in (
        _topology(12, smt=True),
        _topology(12, smt=False),
        _topology(12, quota=4.0),
        _topology(12, quota=1.5),
        _topology(12, quota=0.4),
    ):
        gov = _governor(rg, _config(rg), hardware=_Hardware(topology))
        counts.append(gov.plan_threads("m")[:2])
    assert counts == [(10, 10), (10, 10), (4, 4), (1, 1), (1, 1)]


def test_gt5_the_reserve_is_the_files_fraction_rounded_up_within_its_floor_and_ceiling():
    """GT5 -- the reserve is reserve_fraction of the physical cores, rounded
    up, held between reserve_floor and reserve_ceiling, and never every core:
    an eighth of 64 is 8, held at 4; of 8 is 1; of 2 rounds up to 1; one core
    keeps its only one. Half of 12 with a floor of 0 is 6, and a fraction of
    0 reserves none. Truncating the fraction, or reserving the last core
    -> RED."""
    rg = _rg()

    def threads(physical, **fields):
        gov = _governor(rg, _config(rg, **fields), hardware=_Hardware(_topology(physical, smt=False, fast=physical)))
        return gov.plan_threads("m")[0]

    assert [threads(n) for n in (64, 12, 8, 2, 1)] == [60, 10, 7, 1, 1]
    custom = dict(threads_reserve_fraction=0.5, threads_reserve_floor=0, threads_reserve_ceiling=16)
    assert threads(12, **custom) == 6
    assert threads(12, threads_reserve_fraction=0.0, threads_reserve_floor=0) == 12
    # The plan's own count is never under one, so a reserve of the last core
    # would not show in it; the background's budget shows it.
    one = _governor(rg, _config(rg), hardware=_Hardware(_topology(1, smt=False, fast=1)))
    assert (one.thread_reserve(1), one.plan_background().reserved) == (0, ())


def test_gt6_fast_cores_only_counts_no_more_than_the_fastest_class_and_is_off_by_default():
    """GT6 -- on four fast cores and eight compact ones, the default plans
    ten threads; with fast_cores_only, no more than the four fast ones; on a
    machine of one class it changes nothing. Its default only changes once a
    machine has measured it. Counting the fast cores by default -> RED."""
    rg = _rg()
    hybrid = _Hardware(_topology(12, fast=4))
    uniform = _Hardware(_topology(12, fast=12))
    assert rg.GovernorConfig().threads_fast_cores_only is False
    plain = _governor(rg, _config(rg), hardware=hybrid).plan_threads("m")
    fast = _governor(rg, _config(rg, threads_fast_cores_only=True), hardware=hybrid).plan_threads("m")
    alike = _governor(rg, _config(rg, threads_fast_cores_only=True), hardware=uniform).plan_threads("m")
    assert (plain[0], fast[0], alike[0]) == (10, 4, 10)


def test_gt7_a_model_the_file_names_takes_its_own_count_whatever_the_cpus():
    """GT7 -- threads.models names a model's count: it is carried as it is,
    said "override", for that model only, even where the CPUs cannot be
    read. Planning over the operator's count -> RED."""
    rg = _rg()
    cfg = _config(rg, weights={"big": 9.0, "other": 9.0}, threads_models={"big": 6})
    gov = _governor(rg, cfg, hardware=_Hardware(_topology(12)))
    named = _split_admission(rg, gov, "big")
    other = _split_admission(rg, gov, "other")
    blind = _governor(rg, cfg, hardware=_Hardware(None))
    assert (named.threads, named.threads_batch, named.threads_source) == (6, 6, "override")
    assert (other.threads, other.threads_source) == (10, "plan")
    assert blind.plan_threads("big") == (6, 6, "override")


def test_gt8_the_plan_is_the_models_not_the_class_that_loads_it():
    """GT8 -- a warm-up loads a model for the person who will chat with it,
    and an embedding model serves both the searches and the indexing: on a
    machine with no card, the interactive chat, the warm-up and the
    background indexing each load it with the same count. A smaller count
    for the background here would stay pinned to the resident -> RED."""
    rg = _rg()
    seen = []
    for caller in ("chat", "warmup", "index"):
        gov = _governor(rg, _config(rg, weights={"m": 4.0}, capacity=None), hardware=_Hardware(_topology(12), cards_absent=True))
        _snapshot(rg, gov, capacity=None, cards_absent=True)
        decision = gov.admit("m", requested_ctx=4096, caller=caller)
        seen.append((decision.admitted, decision.admission_class, decision.threads))
    assert seen == [(True, "interactive", 10), (True, "background", 10), (True, "background", 10)]


# ---------------------------------------------------------------------------
# GT9-GT14 -- the count a resident model was loaded with
# ---------------------------------------------------------------------------


def test_gt9_a_resident_model_carries_the_count_its_load_was_told_whatever_the_plan_says_now():
    """GT9 -- a split load told ten threads is pinned at the gate; the CPUs
    then read eight cores (the plan would say seven), and the decisions for
    the resident model still carry ten, said "pinned": Ollama would reload
    it for any other num_thread. Planning the resident again -> RED."""
    rg = _rg()
    hardware = _Hardware(_topology(12))
    gov = _governor(rg, _config(rg, weights={"big": 9.0}), hardware=hardware)
    first = _split_admission(rg, gov)
    rg._account_load(gov, "big", first, {"num_ctx": 4096})
    hardware.topology = _topology(8)
    resident = rg.LoadedModelView(
        name="big", size_vram_bytes=int(8.5 * _GIB), size_bytes=int(11 * _GIB), context_length=4096
    )
    _snapshot(rg, gov, in_use=8.5, loaded=[resident])
    held = gov.admit("big", requested_ctx=4096, caller="chat")
    assert held.reason == "fits_resident"
    assert (held.threads, held.threads_batch, held.threads_source) == (10, 10, "pinned")


def test_gt10_the_pin_ends_with_the_resident_and_the_next_load_takes_the_plan_again():
    """GT10 -- the pin lives while the loaded view shows the model; once it
    has shown it and shows it no more, the pin ends, and the next load of
    the model takes the plan of the CPUs read then. Keeping the pin past
    its resident -> RED."""
    rg = _rg()
    hardware = _Hardware(_topology(12))
    warmup = _Warmup()
    gov = _governor(rg, _config(rg, weights={"big": 9.0}), hardware=hardware, warmup=warmup)
    first = _split_admission(rg, gov)
    rg._account_load(gov, "big", first, {"num_ctx": 4096})
    assert gov.pinned_threads("big") == (10, 10)
    warmup._loaded = [_resident("big", int(8.5 * _GIB), size=int(11 * _GIB), context_length=4096)]
    gov.refresh(force=True)
    assert gov.pinned_threads("big") == (10, 10)
    warmup._loaded = []
    gov.refresh(force=True)
    assert gov.pinned_threads("big") is None
    hardware.topology = _topology(8)
    again = _split_admission(rg, gov)
    assert (again.threads, again.threads_source) == (7, "plan")


def test_gt11_a_callers_own_num_thread_is_the_count_pinned_and_nothing_is_told_on_top():
    """GT11 -- a call that names its own num_thread (the tuner's trials) is
    told nothing on top of it, and its count is the one pinned, so the next
    call keeps the model as that load placed it; a call that names none is
    told the decision's. Pinning the plan over the count sent -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg, weights={"big": 9.0}), hardware=_Hardware(_topology(12)))
    decision = _split_admission(rg, gov)
    assert decision.ollama_threads({"num_ctx": 4096, "num_thread": 3}) is None
    assert decision.ollama_threads({"num_ctx": 4096}) == 10
    assert decision.ollama_threads(None) == 10
    rg._account_load(gov, "big", decision, {"num_ctx": 4096, "num_thread": 3})
    assert gov.pinned_threads("big") == (3, 3)


def test_gt12_a_call_joining_a_pending_load_carries_that_loads_count():
    """GT12 -- a background call for a model still loading for another
    admission joins that load: it carries the count that load was admitted
    with, not the plan of the CPUs read since, or Ollama would reload the
    model under it. Planning the joining call again -> RED."""
    rg = _rg()
    hardware = _Hardware(_topology(12), cards_absent=True)
    gov = _governor(rg, _config(rg, weights={"embed": 1.0}, capacity=None), hardware=hardware)
    _snapshot(rg, gov, capacity=None, cards_absent=True)
    first = gov.admit("embed", requested_ctx=2048, caller="index")
    assert (first.admitted, first.load_expected, first.threads) == (True, True, 10)
    hardware.topology = _topology(8)
    joined = gov.admit("embed", requested_ctx=2048, caller="index")
    assert joined.reason == "fits_pending"
    assert (joined.threads, joined.threads_batch, joined.threads_source) == (10, 10, "pinned")


def test_gt13_unknown_cpus_the_plan_off_or_the_governor_disabled_carry_no_count():
    """GT13 -- CPUs the profile cannot read, no profile at all, threads
    switched off in the file, or the governor disabled: a split carries no
    count, and the engine picks its own, as before the plan. Guessing a
    count from the logical CPUs -> RED."""
    rg = _rg()
    cases = {
        "unknown": _governor(rg, _config(rg, weights={"big": 9.0}), hardware=_Hardware(None)),
        "no profile": _governor(rg, _config(rg, weights={"big": 9.0}), hardware=None),
        "off": _governor(rg, _config(rg, weights={"big": 9.0}, threads_enabled=False), hardware=_Hardware(_topology(12))),
    }
    seen = {name: _split_admission(rg, gov) for name, gov in cases.items()}
    assert {name: (d.threads, d.threads_batch, d.threads_source) for name, d in seen.items()} == {
        name: (None, None, None) for name in cases
    }
    assert cases["unknown"].plan_threads("big") == (None, None, "unknown")
    assert cases["off"].plan_threads("big") == (None, None, "disabled")
    disabled = _governor(rg, _config(rg, weights={"big": 9.0}, enabled=False), hardware=_Hardware(_topology(12)))
    _snapshot(rg, disabled)
    stood_down = disabled.admit("big", requested_ctx=4096, caller="chat")
    assert (stood_down.reason, stood_down.threads) == ("governor_disabled", None)


def test_gt14_the_shipped_threads_block_holds_the_decided_defaults_and_its_ranges(caplog):
    """GT14 -- the shipped file plans threads (enabled), reserves an eighth
    of the physical cores between 1 and 4, counts every class, and names no
    model; a block written reads back as written; a fraction outside [0, 1),
    a floor or ceiling that is not a whole number at or above zero, a
    ceiling under the floor, a model count under one, or a block that is not
    a mapping is warned by name and its default kept. Accepting a reserve of
    every core -> RED."""
    rg = _rg()
    shipped = rg.load_config(Path(source("config", "resource_governor.yaml")))
    fields = ("threads_enabled", "threads_reserve_fraction", "threads_reserve_floor", "threads_reserve_ceiling",
              "threads_fast_cores_only", "threads_models")
    defaults = (True, 0.125, 1, 4, False, {})
    assert tuple(getattr(shipped, f) for f in fields) == defaults
    assert tuple(getattr(rg.GovernorConfig(), f) for f in fields) == defaults
    path = _tmpdir("gt-cfg-") / "resource_governor.yaml"
    path.write_text(
        "threads:\n  enabled: false\n  reserve_fraction: 0.25\n  reserve_floor: 2\n  reserve_ceiling: 6\n"
        "  fast_cores_only: true\n  models:\n    qwen3:8b: 6\n",
        encoding="utf-8",
    )
    read = rg.load_config(path)
    assert tuple(getattr(read, f) for f in fields) == (False, 0.25, 2, 6, True, {"qwen3:8b": 6})
    for body, key in (
        ("threads:\n  reserve_fraction: 1.0\n", "reserve_fraction"),
        ("threads:\n  reserve_fraction: -0.1\n", "reserve_fraction"),
        ("threads:\n  reserve_floor: -1\n", "reserve_floor"),
        ("threads:\n  reserve_floor: 1.5\n", "reserve_floor"),
        ("threads:\n  reserve_ceiling: true\n", "reserve_ceiling"),
        ("threads:\n  reserve_floor: 3\n  reserve_ceiling: 2\n", "reserve_ceiling"),
        ("threads:\n  models:\n    m: 0\n", "models"),
        ("threads:\n  models:\n    m: true\n", "models"),
        ("threads:\n  models: [m]\n", "models"),
        ("threads: 4\n", "threads"),
    ):
        path.write_text(body, encoding="utf-8")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=_RG):
            cfg = rg.load_config(path)
        assert tuple(getattr(cfg, f) for f in fields) == defaults, body
        assert key in " ".join(r.getMessage() for r in caplog.records), body


# ---------------------------------------------------------------------------
# GT15-GT21 -- the engine heads that tell it
# ---------------------------------------------------------------------------


class _Client:
    """The Ollama client as the heads ask it: every call recorded."""

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        if kwargs.get("stream"):
            return iter([{"message": {"content": "local"}, "done": True}])
        return {"message": {"content": "local"}}

    def embed(self, **kwargs):
        self.calls.append(("embed", kwargs))
        texts = kwargs["input"] if isinstance(kwargs["input"], list) else [kwargs["input"]]
        return {"embeddings": [[0.5, 0.25] for _ in texts]}


class _GateGovernor:
    """What backend_admission_gate touches, with its admission scripted."""

    def __init__(self, decision):
        self.config = types.SimpleNamespace(enabled=True)
        self._decision = decision
        self.loads = []
        self.pins = []

    def admit(self, model, requested, caller="direct"):
        return self._decision

    def _record_admission(self, decision):
        pass

    def invalidate_on_load(self, model, num_ctx):
        self.loads.append((model, num_ctx))

    def pin_threads(self, model, threads, threads_batch):
        self.pins.append((model, threads, threads_batch))


def _ollama_window(*extra):
    """The governor, the engine module and ``extra`` in one window, with a
    recording Ollama client and the daily mode."""
    loaded = _open((_BACKEND, source("inference_backend.py")), *extra)
    rg, mod = loaded[_RG], loaded[_BACKEND]
    mod._live_mode = lambda: "daily"
    client = _Client()
    mod.OLLAMA_AVAILABLE = True
    mod._ollama_module = client
    return loaded, rg, mod, client


def _told(rg, model="big", threads=10):
    """An admitted load whose plan computes on the CPU with ``threads``."""
    return rg.AdmissionDecision(
        admitted=True,
        model=model,
        num_ctx=4096,
        action="admit",
        reason="partial_offload",
        caller="direct",
        requested_ctx=4096,
        load_expected=True,
        gpu_share=8.5 / 11.0,
        vram_cost_gb=8.5,
        ram_cost_gb=2.5,
        threads=threads,
        threads_batch=threads,
        threads_source="plan" if threads else None,
    )


_HI = [{"role": "user", "content": "hi"}]


def test_gt15_the_generate_head_tells_ollama_the_count_the_admission_carries():
    """GT15 -- the generate head sends the admission's count as num_thread,
    beside the context, and the gate pins it to the model. Sending the
    context alone (the resident would then reload at the next count) -> RED."""
    _, rg, mod, client = _ollama_window()
    gate = _GateGovernor(_told(rg))
    rg._governor = gate
    try:
        assert mod.OllamaBackend().generate("big", _HI, options={"num_ctx": 4096}).content == "local"
        assert client.calls[0][1]["options"] == {"num_ctx": 4096, "num_thread": 10}
        assert gate.pins == [("big", 10, 10)]
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gt16_the_stream_head_tells_ollama_the_count_too():
    """GT16 -- the streaming head sends the same num_thread as the
    non-streaming one. Telling it on one head only -> RED."""
    _, rg, mod, client = _ollama_window()
    rg._governor = _GateGovernor(_told(rg))
    try:
        chunks = list(mod.OllamaBackend().stream("big", _HI, options={"num_ctx": 4096}))
        assert [c.content for c in chunks] == ["local"]
        sent = client.calls[0][1]
        assert (sent["stream"], sent["options"]) == (True, {"num_ctx": 4096, "num_thread": 10})
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gt17_the_embed_head_tells_the_count_and_sends_no_options_without_one():
    """GT17 -- an embedding loads a model like any other request: with a
    count, the embed head sends it as the options' num_thread; without one,
    it sends no options at all, as before. Embedding without the count the
    model was loaded with -> RED."""
    _, rg, mod, client = _ollama_window()
    try:
        rg._governor = _GateGovernor(_told(rg, "emb"))
        assert mod.OllamaBackend().embed("emb", "text") == [0.5, 0.25]
        rg._governor = _GateGovernor(_told(rg, "emb", threads=None))
        assert mod.OllamaBackend().embed("emb", "text") == [0.5, 0.25]
        assert client.calls == [
            ("embed", {"model": "emb", "input": "text", "options": {"num_thread": 10}}),
            ("embed", {"model": "emb", "input": "text"}),
        ]
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gt18_the_embed_many_head_tells_the_count_likewise():
    """GT18 -- a batch of embeddings, one admission: the count is sent once,
    with the whole batch; without one, no options. Telling the single
    embedding only -> RED."""
    _, rg, mod, client = _ollama_window()
    try:
        rg._governor = _GateGovernor(_told(rg, "emb"))
        assert mod.OllamaBackend().embed_many("emb", ["a", "b"]) == [[0.5, 0.25], [0.5, 0.25]]
        rg._governor = _GateGovernor(_told(rg, "emb", threads=None))
        assert len(mod.OllamaBackend().embed_many("emb", ["a"])) == 1
        assert client.calls == [
            ("embed", {"model": "emb", "input": ["a", "b"], "options": {"num_thread": 10}}),
            ("embed", {"model": "emb", "input": ["a"]}),
        ]
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gt19_the_warmup_tells_the_plan_and_the_keepalive_the_count_the_resident_was_pinned_at():
    """GT19 -- on a machine with no card, the warm-up is admitted as the
    background and loads with the plan's count, which the gate pins; the
    CPUs then read eight cores, and the keep-alive ping of the resident
    still sends the pinned count, so Ollama renews the model as it is.
    Pinging with the plan of the moment -> RED."""
    loaded, rg, mod, client = _ollama_window((_WARMUP, source("model_warmup.py")))
    warm = loaded[_WARMUP]
    hardware = _Hardware(_topology(12), cards_absent=True)
    gov = _governor(rg, _config(rg, weights={"m": 2.0}, capacity=None), hardware=hardware)
    _snapshot(rg, gov, capacity=None, cards_absent=True)
    rg._governor = gov
    backend = mod.OllamaBackend()
    warm._resolve_backend = lambda model=None: backend
    try:
        assert warm.ModelWarmup().warmup("m", force=True, timeout=0.0).success is True
        warmed = client.calls[0][1]["options"]
        assert (warmed["num_predict"], warmed["num_thread"]) == (1, 10)
        hardware.topology = _topology(8)
        resident = rg.LoadedModelView(name="m", size_bytes=4 * _GIB, context_length=warmed["num_ctx"])
        _snapshot(rg, gov, capacity=None, cards_absent=True, loaded=[resident])
        assert warm.ModelWarmup().send_keepalive("m") is True
        pinged = client.calls[1][1]["options"]
        assert (pinged["num_predict"], pinged["num_ctx"], pinged["num_thread"]) == (0, warmed["num_ctx"], 10)
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gt20_a_callers_own_count_is_sent_as_it_is_and_none_is_sent_without_one():
    """GT20 -- a call naming its own num_thread sends it unchanged, and that
    count is the one pinned; an admission that carries no count sends none,
    as before the plan. Overriding the caller's count, or telling one the
    admission never planned -> RED."""
    _, rg, mod, client = _ollama_window()
    try:
        gate = _GateGovernor(_told(rg))
        rg._governor = gate
        mod.OllamaBackend().generate("big", _HI, options={"num_ctx": 4096, "num_thread": 3})
        rg._governor = _GateGovernor(_told(rg, threads=None))
        mod.OllamaBackend().generate("big", _HI, options={"num_ctx": 4096})
        assert [call[1]["options"] for call in client.calls] == [
            {"num_ctx": 4096, "num_thread": 3},
            {"num_ctx": 4096},
        ]
        assert gate.pins == [("big", 3, 3)]
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


class _Llama:
    """The in-process engine's model class; constructing it is a load."""

    built = []

    def __init__(self, **kwargs):
        _Llama.built.append(kwargs)

    def create_chat_completion(self, **kwargs):
        return {"choices": [{"message": {"content": "served"}}]}


def test_gt21_llama_cpp_loads_with_both_counts_from_the_admission_unless_the_operator_names_its_own():
    """GT21 -- llama.cpp in process loads a model with n_threads and
    n_threads_batch from the admission; an operator's n_threads names both
    (llama.cpp's own rule for the prompt), an operator's n_threads_batch the
    prompt's alone; an admission with no count passes neither, and the
    library's defaults stand. Leaving the prompt to every logical CPU
    -> RED."""
    _, rg, mod, _client = _ollama_window()
    mod.LLAMA_CPP_AVAILABLE = True
    mod._LlamaCpp = _Llama
    mod._provenance_guard = lambda path: None
    models = _tmpdir("gt-models-")
    (models / "m.gguf").write_bytes(b"GGUF")
    cpu = rg.AdmissionDecision(
        admitted=True, model="m.gguf", num_ctx=4096, action="admit", reason="fits_ram", caller="direct",
        load_expected=True, threads=10, threads_batch=10, threads_source="plan",
    )
    _Llama.built = []
    try:
        for backend, decision in (
            (mod.LlamaCppBackend(model_dirs=[str(models)]), cpu),
            (mod.LlamaCppBackend(model_dirs=[str(models)], n_threads=6), cpu),
            (mod.LlamaCppBackend(model_dirs=[str(models)], n_threads_batch=12), cpu),
            (mod.LlamaCppBackend(model_dirs=[str(models)]), rg.AdmissionDecision(
                admitted=True, model="m.gguf", num_ctx=4096, action="admit", reason="fits", caller="direct",
                load_expected=True,
            )),
        ):
            rg._governor = _GateGovernor(decision)
            assert backend.generate("m.gguf", _HI).content == "served"
        seen = [(built.get("n_threads"), built.get("n_threads_batch")) for built in _Llama.built]
        assert seen == [(10, 10), (6, 6), (10, 12), (None, None)]
        assert "n_threads" not in _Llama.built[3] and "n_threads_batch" not in _Llama.built[3]
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


# ---------------------------------------------------------------------------
# GT22-GT24 -- the background's own budget
# ---------------------------------------------------------------------------


def _ranked_topology(quota=None):
    """Twelve cores with SMT: four fast ones whose ranks put core 2 first
    and core 0 second, then eight compact ones."""
    cores = tuple(
        types.SimpleNamespace(
            cpus=(n, n + 12),
            rank=(210.0 if n == 2 else 200.0 - n) if n < 4 else 150.0 - n,
            perf_class=0 if n < 4 else 1,
        )
        for n in range(12)
    )
    return types.SimpleNamespace(
        usable=tuple(range(24)), cores=cores, physical=12, smt=True, l3=(), numa=(), quota_cpus=quota,
        class_source="acpi_cppc",
    )


def test_gt22_the_background_plan_leaves_the_highest_ranked_cores_to_the_user_and_counts_the_rest():
    """GT22 -- the background workers get every CPU of the cores outside the
    reserve, the reserve being the highest-ranked cores of the fastest
    class, SMT siblings with them; one worker per core outside the reserve,
    no more than the file's ceiling nor the quota less the reserve, never
    under one; the plan off gives no worker, unreadable CPUs one worker
    that keeps its inherited CPUs. A reserve of the lowest CPUs instead of
    the highest ranks -> RED."""
    rg = _rg()
    topology = _ranked_topology()
    plan = _governor(rg, _config(rg), hardware=_Hardware(topology)).plan_background()
    assert plan.source == "plan"
    assert plan.reserved == (0, 2, 12, 14)
    assert plan.cpus == tuple(sorted(set(range(24)) - {0, 2, 12, 14}))
    assert plan.workers == 4
    assert (plan.in_flight, plan.idle_s, plan.held_retry_s) == (2, 120.0, 30.0)
    wide = _governor(rg, _config(rg, threads_background_max_workers=16), hardware=_Hardware(topology))
    assert wide.plan_background().workers == 10
    for quota, workers in ((5.5, 3), (3.5, 1), (1.0, 1)):
        quoted = _governor(rg, _config(rg, threads_background_max_workers=16), hardware=_Hardware(_ranked_topology(quota)))
        assert quoted.plan_background().workers == workers, quota
    single = types.SimpleNamespace(
        usable=(0,), cores=(types.SimpleNamespace(cpus=(0,), rank=None, perf_class=0),), physical=1, smt=False,
        l3=(), numa=(), quota_cpus=None, class_source="uniform",
    )
    alone = _governor(rg, _config(rg), hardware=_Hardware(single)).plan_background()
    assert (alone.workers, alone.cpus, alone.reserved) == (1, (0,), ())
    off = _governor(rg, _config(rg, threads_background_enabled=False), hardware=_Hardware(topology)).plan_background()
    assert (off.workers, off.cpus, off.reserved, off.source) == (0, (), (), "disabled")
    unknown = _governor(rg, _config(rg), hardware=_Hardware(None)).plan_background()
    assert (unknown.workers, unknown.cpus, unknown.reserved, unknown.source) == (1, (), (), "unknown")


def test_gt23_the_shipped_background_block_holds_the_decided_defaults_and_its_ranges(caplog):
    """GT23 -- the shipped file runs the background in workers (enabled), at
    most four, two tasks queued per worker, closed after 120 s idle, asking
    again 30 s after a refusal a wait can lift; a block written reads back
    as written; a count under one, a delay under zero or not a number, or a
    block that is not a mapping is warned by name and its default kept.
    Accepting a ceiling of zero workers -> RED."""
    rg = _rg()
    fields = (
        "threads_background_enabled",
        "threads_background_max_workers",
        "threads_background_in_flight",
        "threads_background_idle_s",
        "threads_background_held_retry_s",
    )
    defaults = (True, 4, 2, 120.0, 30.0)
    shipped = rg.load_config(Path(source("config", "resource_governor.yaml")))
    assert tuple(getattr(shipped, f) for f in fields) == defaults
    assert tuple(getattr(rg.GovernorConfig(), f) for f in fields) == defaults
    folder = _tmpdir("gt-bg-")
    written = folder / "written.yaml"
    written.write_text(
        "threads:\n  background:\n    enabled: false\n    max_workers: 6\n    in_flight_per_worker: 3\n"
        "    idle_shutdown_s: 45\n    held_retry_s: 5\n",
        encoding="utf-8",
    )
    assert tuple(getattr(rg.load_config(written), f) for f in fields) == (False, 6, 3, 45.0, 5.0)
    wrong = folder / "wrong.yaml"
    wrong.write_text(
        "threads:\n  background:\n    max_workers: 0\n    in_flight_per_worker: -1\n"
        "    idle_shutdown_s: -3\n    held_retry_s: soon\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        assert tuple(getattr(rg.load_config(wrong), f) for f in fields) == defaults
    for key in ("max_workers", "in_flight_per_worker", "idle_shutdown_s", "held_retry_s"):
        assert any(f"threads.background.{key}" in r.getMessage() for r in caplog.records), key
    caplog.clear()
    flat = folder / "flat.yaml"
    flat.write_text("threads:\n  background: 4\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert tuple(getattr(rg.load_config(flat), f) for f in fields) == defaults
    assert any("threads.background" in r.getMessage() for r in caplog.records)


def test_gt24_the_governor_names_a_callers_class_as_its_file_does_and_tells_a_final_refusal():
    """GT24 -- the governor answers a caller's class from its file (the
    shipped classes, and a file that moves a caller), an unnamed caller
    being a user; a refusal no wait can lift (a cost, a card or a context
    unknown, or the emergency stop) is final, one a wait can lift and an
    admission are not. Holding every refusal final -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg))
    assert [gov.caller_class(c) for c in ("index", "warmup", "chat", "direct", "nobody")] == [
        "background", "background", "interactive", "user", "user",
    ]
    moved = _config(rg)
    moved.caller_classes = {**moved.caller_classes, "index": "user"}
    assert _governor(rg, moved).caller_class("index") == "user"
    decision = rg.AdmissionDecision
    for reason in ("background_cost_unknown", "background_capacity_unknown", "background_ctx_unknown"):
        assert rg.refusal_is_final(decision(admitted=False, model="m", reason=reason))
    assert rg.refusal_is_final(decision(admitted=False, model="m", reason="stopped", is_estop=True))
    assert not rg.refusal_is_final(decision(admitted=False, model="m", reason="background_gate_closed"))
    assert not rg.refusal_is_final(decision(admitted=True, model="m", reason="fits"))


# ---------------------------------------------------------------------------
# GT25-GT28 -- where the model computes, and the count measured there
# ---------------------------------------------------------------------------

# Forty layers of 8 KV heads, keys and values of 128: 163840 bytes a token,
# 0.625 GiB at 4096 tokens. Nine GiB of weights then cost 9.625, and 8.5 of
# them on the GPU are 35 of the 40 layers.
_GEOMETRY = {"layers": 40, "kv_bytes_per_token": 163840, "kv_bytes_per_layer": [4096] * 40}


class _Engines:
    """A backend registry naming its engines, whose cached resolution sends
    every model to the first."""

    def __init__(self, *names):
        self._backends = [types.SimpleNamespace(name=name) for name in names]

    def backends(self):
        return list(self._backends)

    def cached_backend(self, model):
        return self._backends[0]


def test_gt25_the_decision_says_where_the_model_computes():
    """GT25 -- a split says "split:35", the 35 layers it puts on the GPU; a
    model the card holds whole says None; on a machine with no card, "cpu".
    The resident a split load made, served as it is, says the placement of
    that load, and a call joining a pending load says that load's. Keying a
    measured count on the model alone, or losing a resident's placement
    -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg, weights={"big": 9.0, "small": 4.0}), hardware=_Hardware(_topology(12)))
    gov._geometry["big"] = dict(_GEOMETRY)
    split = _split_admission(rg, gov)
    assert (split.gpu_layers, split.placement, split.to_dict()["placement"]) == (35, "split:35", "split:35")
    _snapshot(rg, gov)
    whole = gov.admit("small", requested_ctx=4096, caller="chat")
    assert (whole.admitted, whole.placement) == (True, None)
    rg._account_load(gov, "big", split, {"num_ctx": 4096})
    resident = rg.LoadedModelView(
        name="big", size_vram_bytes=int(8.5 * _GIB), size_bytes=int(9.625 * _GIB), context_length=4096
    )
    _snapshot(rg, gov, in_use=8.5, loaded=[resident])
    held = gov.admit("big", requested_ctx=4096, caller="chat")
    assert (held.reason, held.placement) == ("fits_resident", "split:35")
    cpu = _governor(rg, _config(rg, weights={"embed": 1.0}, capacity=None), hardware=_Hardware(_topology(12), cards_absent=True))
    _snapshot(rg, cpu, capacity=None, cards_absent=True)
    first = cpu.admit("embed", requested_ctx=2048, caller="index")
    joined = cpu.admit("embed", requested_ctx=2048, caller="index")
    assert (first.placement, joined.reason, joined.placement) == ("cpu", "fits_pending", "cpu")


def test_gt26_a_count_measured_for_this_model_engine_and_placement_is_planned_said_measured():
    """GT26 -- seven threads kept for ("m", "ollama", "split:35") are the
    plan for that exact triple, said "measured", and the split admission of
    "m" served by Ollama carries them; the model alone, another engine or
    another placement take the plan's ten; a model the file names keeps its
    own count; once the plan falls under the kept count (a quota of six) the
    plan wins, and on another machine the kept count is not read. Reading a
    count for the model alone, or past the reserve -> RED."""
    rg = _rg()
    hardware = _Hardware(_topology(12))
    gov = _governor(rg, _config(rg, weights={"m": 9.0}, threads_models={"named": 6}), hardware=hardware)
    assert gov.record_thread_optimum("m", "ollama", "split:35", threads=7, threads_batch=7, tg=12.0, base_tg=10.0)
    assert gov.record_thread_optimum("named", "ollama", "cpu", threads=4, threads_batch=4, tg=9.0, base_tg=8.0)
    assert gov.plan_threads("m", "ollama", "split:35") == (7, 7, "measured")
    assert [gov.plan_threads("m", *key) for key in ((None, None), ("llama_cpp", "split:35"), ("ollama", "split:30"))] == [
        (10, 10, "plan")
    ] * 3
    assert gov.plan_threads("named", "ollama", "cpu") == (6, 6, "override")
    gov._registry_override = _Engines("ollama")
    gov._geometry["m"] = dict(_GEOMETRY)
    decision = _split_admission(rg, gov, "m")
    assert (decision.engine, decision.placement) == ("ollama", "split:35")
    assert (decision.threads, decision.threads_batch, decision.threads_source) == (7, 7, "measured")
    hardware.topology = _topology(12, quota=6.0)
    assert gov.plan_threads("m", "ollama", "split:35") == (6, 6, "plan")
    hardware.topology = _topology(16)
    assert gov.plan_threads("m", "ollama", "split:35") == (14, 14, "plan")


def test_gt27_kept_counts_outlive_the_governor_and_a_row_that_does_not_hold_is_ignored():
    """GT27 -- a kept count is written with this machine's fingerprint and
    read back by a governor built again on the same file. A row that does
    not hold together (a count of zero, a fraction, a word, a prompt count
    of zero) is ignored and the plan is used. The governor writes no count
    past its plan, none for an unknown placement or engine, and none when
    the CPUs cannot be read; the refused writes change nothing. Trusting a
    row as it is stored, or writing one the plan would refuse -> RED."""
    rg = _rg()
    tmp = _tmpdir("gt27-")

    def build(topology):
        return rg.ResourceGovernor(
            config_path=str(tmp / "missing.yaml"),
            db_path=str(tmp / "governor.db"),
            warmup=None,
            registry=None,
            clock=_Clock(),
            meminfo_path=_meminfo(tmp, 64000.0),
            vram_probe=None,
            hardware=_Hardware(topology),
        )

    first = build(_topology(12))
    assert first.record_thread_optimum("m", "ollama", "cpu", threads=8, threads_batch=8, tg=12.0, base_tg=10.0) is True
    again = build(_topology(12))
    assert again.plan_threads("m", "ollama", "cpu") == (8, 8, "measured")
    row = again.store.get_thread_optimum("m", "ollama", "cpu")
    assert (row["threads"], row["threads_batch"], row["tg"], row["base_tg"]) == (8, 8, 12.0, 10.0)
    assert row["fingerprint"] == again.cpu_fingerprint() == first.cpu_fingerprint() and row["fingerprint"]
    conn = sqlite3.connect(str(tmp / "governor.db"))
    for model, threads, batch in (("zero", 0, 8), ("fraction", 2.5, 8), ("word", "eight", 8), ("prompt", 8, 0)):
        conn.execute(
            "INSERT OR REPLACE INTO thread_optima (model, engine, placement, fingerprint, threads, threads_batch,"
            " tg, base_tg, measured_at) VALUES (?, 'ollama', 'cpu', ?, ?, ?, 12.0, 10.0, 1.0)",
            (model, again.cpu_fingerprint(), threads, batch),
        )
    conn.commit()
    conn.close()
    for model in ("zero", "fraction", "word", "prompt"):
        assert again.store.get_thread_optimum(model, "ollama", "cpu") is None, model
        assert again.plan_threads(model, "ollama", "cpu") == (10, 10, "plan"), model
    refused = [
        again.record_thread_optimum("m", "ollama", "cpu", threads=11, threads_batch=11, tg=13.0, base_tg=10.0),
        again.record_thread_optimum("m", "ollama", None, threads=6, threads_batch=6, tg=13.0, base_tg=10.0),
        again.record_thread_optimum("m", "", "cpu", threads=6, threads_batch=6, tg=13.0, base_tg=10.0),
        build(None).record_thread_optimum("m", "ollama", "cpu", threads=6, threads_batch=6, tg=13.0, base_tg=10.0),
    ]
    assert refused == [False, False, False, False]
    assert again.plan_threads("m", "ollama", "cpu") == (8, 8, "measured")


def test_gt28_a_count_a_call_sends_to_a_resident_is_the_count_pinned_from_then_on():
    """GT28 -- Ollama reloads a resident model for a num_thread other than
    the one it holds: a call to the resident a split load made that sends
    its own count (a tuner's trial) leaves that count pinned, with the
    placement of the load, and the next decision for the resident carries
    it; a call that sends none is told that count and changes nothing.
    Keeping the first load's count while the engine holds another -> RED."""
    _, rg, mod, client = _ollama_window()
    gov = _governor(rg, _config(rg, weights={"big": 9.0}), hardware=_Hardware(_topology(12)))
    gov._geometry["big"] = dict(_GEOMETRY)
    first = _split_admission(rg, gov)
    rg._account_load(gov, "big", first, {"num_ctx": 4096})
    assert gov.pinned_threads("big") == (10, 10)
    resident = rg.LoadedModelView(
        name="big", size_vram_bytes=int(8.5 * _GIB), size_bytes=int(9.625 * _GIB), context_length=4096
    )
    _snapshot(rg, gov, in_use=8.5, loaded=[resident])
    rg._governor = gov
    try:
        mod.OllamaBackend().generate("big", _HI, options={"num_ctx": 4096, "num_thread": 6})
        assert client.calls[-1][1]["options"]["num_thread"] == 6
        assert (gov.pinned_threads("big"), gov.pinned_placement("big")) == ((6, 6), "split:35")
        held = gov.admit("big", requested_ctx=4096, caller="chat")
        assert (held.threads, held.threads_source, held.placement) == (6, "pinned", "split:35")
        mod.OllamaBackend().generate("big", _HI, options={"num_ctx": 4096})
        assert client.calls[-1][1]["options"]["num_thread"] == 6
        assert gov.pinned_threads("big") == (6, 6)
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


# ---------------------------------------------------------------------------
# GT29-GT32 -- the threads on the status surface
# ---------------------------------------------------------------------------

_ROUTES = "opti_oignon.api.routes_governor"
_POOL = "opti_oignon.background_pool"


def test_gt29_the_status_shows_the_plan_the_reserve_and_the_background_budget_as_the_governor_plans_them():
    """GT29 -- the threads section of the status is the governor's own
    calls, as plain data: the plan's count and source (plan_threads with no
    model), the physical cores, the reserve (thread_reserve), the
    fast-cores switch, the quota, and the background's budget with the CPUs
    it leaves to the user (plan_background). Unreadable CPUs say "unknown"
    with no count and no reserve; the plan off says "disabled" while the
    background keeps its own budget. A section that works the reserve out
    again instead of asking the governor -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg, threads_reserve_fraction=0.25), hardware=_Hardware(_ranked_topology(quota=7.5)))
    state = gov.threads_state()
    assert json.loads(json.dumps(state)) == state
    assert (state["available"], state["enabled"]) == (True, True)
    assert state["plan"] == {"threads": 7, "threads_batch": 7, "source": "plan"}
    assert (state["physical"], state["reserve"], state["quota_cpus"], state["fast_cores_only"]) == (12, 3, 7.5, False)
    background = state["background"]
    assert (background["workers"], background["source"], background["in_flight"]) == (4, "plan", 2)
    assert background["reserved"] == [0, 1, 2, 12, 13, 14]
    assert background["cpus"] == sorted(set(range(24)) - {0, 1, 2, 12, 13, 14})
    clamped = _governor(rg, _config(rg, threads_reserve_fraction=0.5), hardware=_Hardware(_ranked_topology()))
    assert clamped.threads_state()["reserve"] == clamped.thread_reserve(12) == 4
    unknown = _governor(rg, _config(rg), hardware=_Hardware(None)).threads_state()
    assert unknown["plan"] == {"threads": None, "threads_batch": None, "source": "unknown"}
    assert (unknown["physical"], unknown["reserve"], unknown["background"]["source"]) == (None, None, "unknown")
    off = _governor(rg, _config(rg, threads_enabled=False), hardware=_Hardware(_topology(12))).threads_state()
    assert (off["enabled"], off["plan"]["source"], off["background"]["source"]) == (False, "disabled", "plan")


def test_gt30_the_status_shows_each_residents_pinned_count_and_placement_until_its_pin_ends():
    """GT30 -- a model whose load was told a count shows that count, its
    prompt count and the placement it computes at, as the decisions for the
    resident carry them, and whether the loaded view has shown it since;
    once the view no longer shows it, the model leaves the list. Showing the
    layer pins in place of the thread pins -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg, weights={"big": 9.0}), hardware=_Hardware(_topology(12)))
    gov._geometry["big"] = dict(_GEOMETRY)
    first = _split_admission(rg, gov)
    rg._account_load(gov, "big", first, {"num_ctx": 4096})
    assert gov.threads_state()["pins"] == [
        {"model": "big", "threads": 10, "threads_batch": 10, "placement": "split:35", "seen": False}
    ]
    resident = rg.LoadedModelView(
        name="big", size_vram_bytes=int(8.5 * _GIB), size_bytes=int(9.625 * _GIB), context_length=4096
    )
    gov._release_pins([resident])
    assert [pin["seen"] for pin in gov.threads_state()["pins"]] == [True]
    gov._release_pins([])
    assert gov.threads_state()["pins"] == []


def test_gt31_the_status_lists_the_kept_counts_newest_first_within_the_files_limit_said_of_this_machine_or_not(caplog):
    """GT31 -- the counts kept in the store are listed newest first, each
    with its model, engine, placement, counts and rates, and whether the
    CPUs read now are the shape it was measured on; a row that does not
    hold together is not listed and takes no place in the list, which stops
    at the file's threads.status_limit (50 shipped; a limit under one or
    not a whole number is warned by name and 50 kept). Listing the rows as
    they are stored -> RED."""
    rg = _rg()
    tmp = _tmpdir("gt31-")
    gov = rg.ResourceGovernor(
        config_path=str(tmp / "missing.yaml"),
        db_path=str(tmp / "governor.db"),
        warmup=None,
        registry=None,
        clock=_Clock(),
        meminfo_path=_meminfo(tmp, 64000.0),
        vram_probe=None,
        hardware=_Hardware(_topology(12)),
    )
    here = gov.cpu_fingerprint()
    conn = sqlite3.connect(str(tmp / "governor.db"))
    conn.executemany(
        "INSERT INTO thread_optima (model, engine, placement, fingerprint, threads, threads_batch, tg, base_tg,"
        " measured_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("a", "ollama", "cpu", here, 8, 8, 12.0, 10.0, 3.0),
            ("b", "llama_cpp", "split:30", here, 6, 6, 9.0, 8.0, 2.0),
            ("c", "ollama", "cpu", "measured-elsewhere", 4, 4, 5.0, 4.0, 4.0),
            ("bad", "ollama", "cpu", here, 0, 8, 12.0, 10.0, 5.0),
        ],
    )
    conn.commit()
    conn.close()
    measured = gov.threads_state()["measured"]
    assert [(m["model"], m["this_machine"]) for m in measured] == [("c", False), ("a", True), ("b", True)]
    assert measured[1] == {
        "model": "a", "engine": "ollama", "placement": "cpu", "threads": 8, "threads_batch": 8,
        "tg": 12.0, "base_tg": 10.0, "measured_at": 3.0, "this_machine": True,
    }
    gov._config = _config(rg, threads_status_limit=2)
    assert [m["model"] for m in gov.threads_state()["measured"]] == ["c", "a"]
    shipped = rg.load_config(Path(source("config", "resource_governor.yaml")))
    assert shipped.threads_status_limit == rg.GovernorConfig().threads_status_limit == 50
    for raw in ("0", "-3", "2.5", "many", "true"):
        path = tmp / f"limit-{raw}.yaml"
        path.write_text(f"threads:\n  status_limit: {raw}\n", encoding="utf-8")
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert rg.load_config(path).threads_status_limit == 50, raw
        assert "threads.status_limit" in caplog.text, raw
    path = tmp / "limit-7.yaml"
    path.write_text("threads:\n  status_limit: 7\n", encoding="utf-8")
    assert rg.load_config(path).threads_status_limit == 7


def test_gt32_the_status_route_carries_the_threads_and_the_background_and_a_failing_section_says_so_alone():
    """GT32 -- the governor's status carries the threads section as the
    governor gives it and the background pool's state, and reading it
    creates no pool; a threads section whose reading fails, or a pool module
    that cannot be reached, says it is unavailable while the rest of the
    status stands. A status that fails whole when one section does -> RED."""
    loaded = _open((_POOL, source("background_pool.py")), (_ROUTES, source("api", "routes_governor.py")),
                   packages=("opti_oignon.api",))
    rg, bp, routes = loaded[_RG], loaded[_POOL], loaded[_ROUTES]
    gov = _governor(rg, _config(rg), hardware=_Hardware(_topology(12)))
    _snapshot(rg, gov)
    status = routes.status_payload(gov)
    assert status["threads"] == gov.threads_state()
    assert status["threads"]["plan"]["threads"] == 10
    assert (status["background"]["available"], status["background"]["mode"], bp._POOL) == (True, "unused", None)

    def broken():
        raise RuntimeError("no plan")

    gov.plan_background = broken
    partial = routes.status_payload(gov)
    assert partial["threads"] == {"available": False}
    assert (partial["scheduling"], partial["hardware"]) == (status["scheduling"], status["hardware"])
    alone = _open((_ROUTES, source("api", "routes_governor.py")), packages=("opti_oignon.api",))[_ROUTES]
    unreachable = alone.status_payload(gov)
    assert unreachable["background"] == {"available": False}
    assert unreachable["scheduling"] == status["scheduling"]


# ---------------------------------------------------------------------------
# GT33-GT34 -- a waiter's cancel, and memory for background work
# ---------------------------------------------------------------------------


def _until(predicate, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_gt33_a_waiter_whose_caller_cancels_leaves_the_queue_at_its_next_wake_refused_cancelled():
    """GT33 -- a caller that waits in the queue may hand the governor its
    cancel: once it is set, the waiter leaves at its next wake (the queue's
    own slice), refused "cancelled", and its place is given back, rather
    than waiting out its class's wait while the work it serves is gone; a
    cancel already set when the first try is refused enqueues nothing.
    Waiting out the wait whatever the cancel -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg, weights={"m": 50.0}, queue_enabled_per_caller={"index": True}, queue_wait_s=60.0))
    _snapshot(rg, gov)
    cancel = threading.Event()
    answers = []
    waiter = threading.Thread(
        target=lambda: answers.append(gov.admit_or_wait("m", caller="index", cancel=cancel)), daemon=True
    )
    waiter.start()
    assert _until(lambda: len(gov._waiters) == 1, 5.0), "the background caller waits in the queue"
    cancel.set()
    left = time.monotonic()
    waiter.join(timeout=5.0)
    assert not waiter.is_alive() and time.monotonic() - left < 2.0
    assert (answers[0].admitted, answers[0].reason) == (False, "cancelled")
    assert gov._waiters == []
    entered = [d for d in gov._store.recent_decisions(50) if d["decision"] == "queue"]
    assert len(entered) == 1, "the first waiter's entry is in the ring"
    again = gov.admit_or_wait("m", caller="index", cancel=cancel)
    assert (again.admitted, again.reason, gov._waiters) == (False, "cancelled", [])
    assert [d for d in gov._store.recent_decisions(50) if d["decision"] == "queue"] == entered


def test_gt34_background_work_is_told_memory_is_short_under_pressure_or_under_the_reserve():
    """GT34 -- the governor tells background work whether memory is short,
    so that it keeps one task at a time: "memory_pressure" while the kernel
    reports memory pressure (the hysteresis the RAM reserve reads),
    "ram_under_reserve" while the RAM available is under the reserve a split
    leaves the rest of the machine, None otherwise, and None where the RAM
    cannot be read. Saying memory is never short -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg))
    snapshot = _snapshot(rg, gov, ram_mb=48000.0)
    snapshot.ram_total_mb = 65536.0
    reserve_mb = gov.effective_ram_reserve_gb(snapshot) * 1024.0
    assert 1024.0 < reserve_mb < 48000.0
    assert gov.background_memory_short() is None
    snapshot.memory_pressure_active = True
    assert gov.background_memory_short() == "memory_pressure"
    snapshot.memory_pressure_active = False
    snapshot.ram_available_mb = reserve_mb - 1.0
    assert gov.background_memory_short() == "ram_under_reserve"
    snapshot.ram_total_mb, snapshot.ram_available_mb = 0.0, 0.0
    assert gov.background_memory_short() is None


# ---------------------------------------------------------------------------
# GT35-GT38 -- whose CPUs bound the count
# ---------------------------------------------------------------------------


def test_gt35_an_engine_in_its_own_process_is_planned_on_the_machines_cores_not_the_servers_limits():
    """GT35 -- supersedes GT4: the count a decision carries is told to an
    engine that computes in a process of its own (Ollama), which neither the
    server's affinity nor its cgroup quota binds: the machine's physical
    cores less the reserve, never an SMT sibling, never under one thread. A
    server held to four cores and three CPUs of quota, on a machine of
    twelve, plans ten for the engine; twelve cores of two siblings plan ten,
    not twenty-two; a quota in the machine's view counts for nothing; one
    core plans one. Planning the engine within the server's own limits
    -> RED."""
    rg = _rg()
    server = _topology(4, quota=3.0)
    split = _governor(rg, _config(rg, weights={"big": 9.0}), hardware=_Hardware(server, machine=_topology(12)))
    decision = _split_admission(rg, split)
    assert (decision.threads, decision.threads_batch, decision.threads_source) == (10, 10, "plan")
    counts = []
    for machine in (
        _topology(12, smt=True),
        _topology(12, smt=False),
        _topology(12, quota=0.4),
        _topology(1, smt=False, fast=1),
    ):
        gov = _governor(rg, _config(rg), hardware=_Hardware(server, machine=machine))
        counts.append(gov.plan_threads("m")[:2])
    assert counts == [(10, 10), (10, 10), (10, 10), (1, 1)]


def test_gt36_the_servers_own_cap_bounds_what_computes_in_its_process_and_only_that():
    """GT36 -- the server's own cap is the plan's rule on the server's own
    CPUs: the physical cores of its affinity less the reserve, never past
    its cgroup quota rounded down, never under one thread (a quota of four
    CPUs caps twelve cores at four, one of 1.5 at one, one of 0.4 at one; an
    affinity of four cores leaves three); unreadable CPUs, no cap.
    llama.cpp computes in the server's process and loads within that cap,
    an operator's own n_threads standing as named; an Ollama head tells the
    plan's count as it is. Loading llama.cpp past the server's own limits
    -> RED."""
    rg = _rg()
    caps = [
        _governor(rg, _config(rg), hardware=_Hardware(server, machine=_topology(12))).server_threads()
        for server in (_topology(12, quota=4.0), _topology(12, quota=1.5), _topology(12, quota=0.4), _topology(4))
    ]
    assert caps == [4, 1, 1, 3]
    assert _governor(rg, _config(rg), hardware=_Hardware(None)).server_threads() is None
    _, rg, mod, client = _ollama_window()
    mod.LLAMA_CPP_AVAILABLE = True
    mod._LlamaCpp = _Llama
    mod._provenance_guard = lambda path: None
    models = _tmpdir("gt-models-")
    (models / "m.gguf").write_bytes(b"GGUF")
    cpu = rg.AdmissionDecision(
        admitted=True, model="m.gguf", num_ctx=4096, action="admit", reason="fits_ram", caller="direct",
        load_expected=True, threads=10, threads_batch=10, threads_source="plan",
    )
    _Llama.built = []
    try:
        for backend in (
            mod.LlamaCppBackend(model_dirs=[str(models)]),
            mod.LlamaCppBackend(model_dirs=[str(models)], n_threads=6),
        ):
            gate = _GateGovernor(cpu)
            gate.server_threads = lambda: 3
            rg._governor = gate
            assert backend.generate("m.gguf", _HI).content == "served"
        assert [(built.get("n_threads"), built.get("n_threads_batch")) for built in _Llama.built] == [(3, 3), (6, 6)]
        gate = _GateGovernor(_told(rg))
        gate.server_threads = lambda: 3
        rg._governor = gate
        assert mod.OllamaBackend().generate("big", _HI, options={"num_ctx": 4096}).content == "local"
        assert client.calls[-1][1]["options"] == {"num_ctx": 4096, "num_thread": 10}
    finally:
        rg.clear_active_ticket()
        rg.reset_resource_governor()


def test_gt37_a_count_measured_for_this_model_engine_and_placement_is_planned_unless_past_the_plan():
    """GT37 -- supersedes GT26, whose plan fell under the kept count by the
    server's quota, which no longer bounds an engine in a process of its
    own: seven threads kept for ("m", "ollama", "split:35") are the plan for
    that exact triple, said "measured", and the split admission of "m"
    served by Ollama carries them; the model alone, another engine or
    another placement take the plan's ten; a model the file names keeps its
    own count; once the plan falls under the kept count (half the cores
    reserved) the plan wins, and on another machine the kept count is not
    read. Reading a count for the model alone, or past the plan -> RED."""
    rg = _rg()
    hardware = _Hardware(_topology(12))
    gov = _governor(rg, _config(rg, weights={"m": 9.0}, threads_models={"named": 6}), hardware=hardware)
    assert gov.record_thread_optimum("m", "ollama", "split:35", threads=7, threads_batch=7, tg=12.0, base_tg=10.0)
    assert gov.record_thread_optimum("named", "ollama", "cpu", threads=4, threads_batch=4, tg=9.0, base_tg=8.0)
    assert gov.plan_threads("m", "ollama", "split:35") == (7, 7, "measured")
    assert [gov.plan_threads("m", *key) for key in ((None, None), ("llama_cpp", "split:35"), ("ollama", "split:30"))] == [
        (10, 10, "plan")
    ] * 3
    assert gov.plan_threads("named", "ollama", "cpu") == (6, 6, "override")
    gov._registry_override = _Engines("ollama")
    gov._geometry["m"] = dict(_GEOMETRY)
    decision = _split_admission(rg, gov, "m")
    assert (decision.engine, decision.placement) == ("ollama", "split:35")
    assert (decision.threads, decision.threads_batch, decision.threads_source) == (7, 7, "measured")
    gov._config.threads_reserve_fraction, gov._config.threads_reserve_ceiling = 0.5, 8
    assert gov.plan_threads("m", "ollama", "split:35") == (6, 6, "plan")
    gov._config.threads_reserve_fraction, gov._config.threads_reserve_ceiling = 0.125, 4
    hardware.topology = _topology(16)
    assert gov.plan_threads("m", "ollama", "split:35") == (14, 14, "plan")


def test_gt38_the_status_shows_the_plan_the_servers_own_cap_and_the_background_budget():
    """GT38 -- supersedes GT29, which planned within the server's quota: the
    threads section of the status is the governor's own calls, as plain
    data: the plan's count and source (plan_threads with no model), the
    physical cores, the reserve (thread_reserve), the fast-cores switch, the
    server's quota and its own cap (server_threads), and the background's
    budget with the CPUs it leaves to the user (plan_background). Unreadable
    CPUs say "unknown" with no count, no reserve and no cap; the plan off
    says "disabled", with no cap, while the background keeps its own
    budget. A section that works the cap out again instead of asking the
    governor -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg, threads_reserve_fraction=0.25), hardware=_Hardware(_ranked_topology(quota=7.5)))
    state = gov.threads_state()
    assert json.loads(json.dumps(state)) == state
    assert (state["available"], state["enabled"]) == (True, True)
    assert state["plan"] == {"threads": 9, "threads_batch": 9, "source": "plan"}
    assert (state["physical"], state["reserve"], state["quota_cpus"], state["fast_cores_only"]) == (12, 3, 7.5, False)
    assert state["server_threads"] == gov.server_threads() == 7
    background = state["background"]
    assert (background["workers"], background["source"], background["in_flight"]) == (4, "plan", 2)
    assert background["reserved"] == [0, 1, 2, 12, 13, 14]
    assert background["cpus"] == sorted(set(range(24)) - {0, 1, 2, 12, 13, 14})
    clamped = _governor(rg, _config(rg, threads_reserve_fraction=0.5), hardware=_Hardware(_ranked_topology()))
    assert clamped.threads_state()["reserve"] == clamped.thread_reserve(12) == 4
    unknown = _governor(rg, _config(rg), hardware=_Hardware(None)).threads_state()
    assert (unknown["physical"], unknown["reserve"], unknown["server_threads"], unknown["background"]["source"]) == (
        None, None, None, "unknown"
    )
    off = _governor(rg, _config(rg, threads_enabled=False), hardware=_Hardware(_topology(12))).threads_state()
    assert (off["enabled"], off["plan"]["source"], off["server_threads"], off["background"]["source"]) == (
        False, "disabled", None, "plan"
    )


def test_gt39_the_fastest_class_candidate_is_counted_on_the_machines_cores():
    """GT39 -- the sweep's fastest-class candidate is the machine's fast
    cores, as the plan it measures against is the machine's: a server held to
    two cores, on a machine of twelve whose four first are fast, sweeps 4,
    5, 8 and 10, never the server's own count of two. Counting the fast class
    on the server's own CPUs -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg), hardware=_Hardware(_topology(2, fast=2), machine=_topology(12, fast=4)))
    assert gov.thread_candidates("m", [0.5, 0.75]) == (10, [4, 5, 8, 10], "plan")


def test_gt40_a_refusal_a_callers_own_cancel_made_stays_out_of_the_refusal_window():
    """GT40 -- a user-class caller whose cancel is set as its first try is
    refused does not wait: it is refused "cancelled", and that refusal is its
    own doing, not the machine's, so only its first try enters the
    refusal-rate window the pressure reads. Counting a caller's own cancel as
    a refusal of the machine's -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg, weights={"m": 500.0}, queue_enabled_per_caller={"benchmark": True}, queue_wait_s=60.0))
    _snapshot(rg, gov)
    before = len(gov._refusal_events)
    cancel = threading.Event()
    cancel.set()
    decision = gov.admit_or_wait("m", caller="benchmark", cancel=cancel)
    assert (decision.admitted, decision.reason, decision.admission_class) == (False, "cancelled", "user")
    assert len(gov._refusal_events) == before + 1


def test_gt41_background_work_is_given_the_ram_available_less_the_reserve_as_its_room():
    """GT41 -- the room the governor gives background work is the RAM
    available less the reserve a split leaves the machine, in bytes and
    never under zero; none while the kernel reports memory pressure; None
    where the RAM cannot be read, which then bounds nothing. Giving the
    background the RAM available whole -> RED."""
    rg = _rg()
    gov = _governor(rg, _config(rg))
    snapshot = _snapshot(rg, gov, ram_mb=48000.0)
    snapshot.ram_total_mb = 65536.0
    reserve_mb = gov.effective_ram_reserve_gb(snapshot) * 1024.0
    assert gov.background_memory_room() == int((48000.0 - reserve_mb) * 1024 * 1024)
    snapshot.ram_available_mb = reserve_mb / 2
    assert gov.background_memory_room() == 0
    snapshot.ram_available_mb = 48000.0
    snapshot.memory_pressure_active = True
    assert gov.background_memory_room() == 0
    snapshot.memory_pressure_active = False
    snapshot.ram_total_mb = 0.0
    assert gov.background_memory_room() is None


_PARSE_FACTORS = {"default": 4.0, ".pdf": 10.0, ".docx": 30.0, ".doc": 30.0, ".xlsx": 50.0, ".xls": 50.0}


def test_gt42_the_parse_factors_hold_their_shipped_values_and_ranges_and_the_plan_carries_them(caplog):
    """GT42 -- threads.background.parse_expansion says, per kind of file, how
    many times its size the parse of a file takes in memory: the shipped
    file and the defaults say 4 for any file, 10 for a PDF, 30 for a Word
    document, 50 for a spreadsheet; the file's table is laid over those
    (a named kind takes its factor, the others keep theirs); a factor under
    one, one that is not a number, or a key that is neither "default" nor
    an extension is warned by name and dropped; a value that is not a table
    keeps the defaults; the background plan carries the factors. Accepting a
    factor under one -> RED."""
    rg = _rg()
    shipped = rg.load_config(Path(source("config", "resource_governor.yaml")))
    assert shipped.threads_background_parse_expansion == _PARSE_FACTORS
    assert rg.GovernorConfig().threads_background_parse_expansion == _PARSE_FACTORS
    folder = _tmpdir("gt-parse-")
    written = folder / "written.yaml"
    written.write_text("threads:\n  background:\n    parse_expansion:\n      .pdf: 12\n      .odt: 25.5\n", encoding="utf-8")
    laid = {**_PARSE_FACTORS, ".pdf": 12.0, ".odt": 25.5}
    assert rg.load_config(written).threads_background_parse_expansion == laid
    wrong = folder / "wrong.yaml"
    wrong.write_text(
        "threads:\n  background:\n    parse_expansion:\n      default: 0.5\n      .pdf: many\n      pdf: 9\n"
        "      .xlsx: 60\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        assert rg.load_config(wrong).threads_background_parse_expansion == {**_PARSE_FACTORS, ".xlsx": 60.0}
    for key in ("default", ".pdf", "pdf"):
        assert any(f"threads.background.parse_expansion.{key}" in r.getMessage() for r in caplog.records), key
    caplog.clear()
    flat = folder / "flat.yaml"
    flat.write_text("threads:\n  background:\n    parse_expansion: 8\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert rg.load_config(flat).threads_background_parse_expansion == _PARSE_FACTORS
    assert any("threads.background.parse_expansion" in r.getMessage() for r in caplog.records)
    gov = _governor(rg, rg.load_config(written), hardware=_Hardware(_topology(12)))
    assert gov.plan_background().parse_expansion == laid
