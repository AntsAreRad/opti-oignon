"""Contracts of the threads a run leaves behind and of the route that runs the startup checklist again (260): the live
metrics' sampler, a singleton started by the first reading, was never stopped -- neither when the API's lifespan ended
nor between two contracts -- and its thread read the memory through ``opti_oignon.hardware_profile`` at every interval,
so a window that stands that module in counted a reading more when the thread fell into it (HW16, the chain of 257,
03 h 35). And ``GET /api/security/startup-checks?force=true`` ran the whole synchronous checklist on the event loop.

  * TB1 -- THE SAMPLER STOPS WITH THE APP: the real lifespan, its startup's heavy actions spied, ends with the live
    metrics' sampler stopped and forgotten; the next reading starts a new one; a lifespan in which the live metrics
    were never imported ends without importing them.
  * TB2 -- HW16 ON ITS OWN THREAD (supersedes HW16 of tests/test_hardware_profile_contracts.py, which counted every
    thread's readings): the smart router and the live metrics read the RAM through the profile, two readings on the
    contract's thread; a reading another thread makes meanwhile is said apart, never counted.
  * TB3 -- THE FORCED ROUTE OFF THE LOOP: while a forced run of the checklist waits, the event loop serves on.
TB1 drives the real application lifespan with the startup boot guard suite's harness; TB2 loads the smart router and
the live metrics on the shared isolation window as HW16 did; TB3 calls the real route's coroutine.
"""
import asyncio
import sys
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_hardware_profile_contracts as hp_t  # noqa: E402
import test_startup_boot_guard_contracts as bg_t  # noqa: E402
from _isolation import source  # noqa: E402

BUDGET_S = {
    "test_tb1_the_sampler_stops_with_the_app": 8.0,
    "test_tb2_hw16_on_its_own_thread": 2.0,
    "test_tb3_the_forced_route_off_the_loop": 3.0,
}


def test_tb1_the_sampler_stops_with_the_app():
    harness, restore = bg_t._real_app_harness()
    try:
        from opti_oignon import live_metrics as lm

        undo = bg_t._install_real_stub_checks(harness["sc"])
        lm.reset_live_metrics()
        sampler = lm.LiveMetricsCollector(config=lm.LiveMetricsConfig(enabled=True, sample_interval_ms=50))
        sampler.start()
        lm._collector = sampler
        try:
            assert sampler.is_running and sampler._thread is not None and sampler._thread.is_alive()
            thread = sampler._thread
            assert asyncio.run(bg_t._enter_and_exit_lifespan(harness["app"])) is True
            assert lm._collector is None, "the lifespan ended with the live metrics' sampler still held"
            assert not sampler.is_running and not thread.is_alive(), "the sampler's thread outlived the app"
            fresh = lm.get_live_metrics(auto_start=False)
            assert fresh is not sampler and not fresh.is_running
            # Never imported, nothing started: the shutdown does not import the live metrics to stop nothing.
            import opti_oignon

            held = sys.modules.pop("opti_oignon.live_metrics")
            del opti_oignon.live_metrics        # the package's attribute too: an import would find it there
            try:
                assert asyncio.run(bg_t._enter_and_exit_lifespan(harness["app"])) is True
                assert "opti_oignon.live_metrics" not in sys.modules, "the shutdown imported the live metrics"
            finally:
                sys.modules["opti_oignon.live_metrics"] = held
                opti_oignon.live_metrics = held
        finally:
            lm.reset_live_metrics()
            undo()
    finally:
        restore()


def test_tb2_hw16_on_its_own_thread():
    reads = []
    fake = types.ModuleType(hp_t._HP)

    def read_meminfo(path="/proc/meminfo"):
        reads.append((threading.get_ident(), path))
        return types.SimpleNamespace(total_mib=8000.0, available_mib=3000.0)

    fake.read_meminfo = read_meminfo
    loaded = hp_t._open((hp_t._SR, source("smart_router.py")), (hp_t._LM, source("live_metrics.py")),
                        seeded={hp_t._HP: fake})
    try:
        # Another thread reads through the module the window stands in, as a sampler left running would.
        other = threading.Thread(target=lambda: sys.modules[hp_t._HP].read_meminfo())
        other.start()
        other.join(5)
        assert loaded[hp_t._SR]._get_available_ram_mb() == 3000.0
        assert loaded[hp_t._LM]._get_system_memory() == (5000.0, 8000.0)
        mine = [path for ident, path in reads if ident == threading.get_ident()]
        assert mine == ["/proc/meminfo", "/proc/meminfo"], reads
        assert [ident for ident, _path in reads if ident != threading.get_ident()] != [], (
            "the other thread's reading is said apart, never counted")
    finally:
        while hp_t._CLOSERS:
            hp_t._CLOSERS.pop()()
    # Without the profile, the RAM is unknown: the pre-flight excludes nothing.
    alone = hp_t._open((hp_t._SR, source("smart_router.py")), (hp_t._LM, source("live_metrics.py")),
                       blocked=(hp_t._HP,))
    try:
        assert alone[hp_t._SR]._get_available_ram_mb() == 0.0
        assert alone[hp_t._LM]._get_system_memory() == (0.0, 0.0)
    finally:
        while hp_t._CLOSERS:
            hp_t._CLOSERS.pop()()


def test_tb3_the_forced_route_off_the_loop(monkeypatch):
    from opti_oignon import startup_checks as sc
    from opti_oignon.api import routes_security as rs

    ticked = threading.Event()
    seen = []

    def slow_checklist(force=False):
        seen.append(ticked.wait(2.0))      # True only when the loop ran the ticker while this waited
        return types.SimpleNamespace(to_dict=lambda: {"forced": force})

    monkeypatch.setattr(sc, "run_startup_checks", slow_checklist)

    async def ticker():
        await asyncio.sleep(0.01)
        ticked.set()

    async def both():
        return await asyncio.gather(rs.get_startup_checks(force=True), ticker())

    answer, _ = asyncio.run(both())
    assert answer == {"forced": True}, answer
    assert seen == [True], "the forced checklist ran on the event loop: nothing else was served while it waited"
