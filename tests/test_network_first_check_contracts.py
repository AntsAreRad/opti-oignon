#!/usr/bin/env python3
"""Contracts for the network manager's first check: none at import.

The network manager ran a health check of the inference engine when its module
was loaded, so importing the executor (which imports it) reached the engine:
the call outside any contract that the engine guard counted, and CV19 of 239
whenever it imported first. The first check now runs when the API starts, or at
the first read of ``is_online`` on a path with no start, once.
  * NI1 -- IMPORT REACHES NOTHING: loading the module asks the engine nothing,
    lists no model, starts no thread, and its manager says it has not checked.
  * NI2 -- ONE FIRST CHECK: ``first_check`` checks once and the status says
    what it found; a second call checks nothing; a manager read first through
    ``is_online`` checks then, once, even when eight threads read it at once.
  * NI3 -- A CHECK THAT FAILS IS AN OFFLINE STATUS: an engine whose health
    check raises leaves the manager checked and offline, ``first_check``
    raising nothing, and the next read checking nothing more.
  * NI4 -- THE API CHECKS AT ITS START: the real lifespan runs the first check
    once before it serves, and a first check that raises does not stop it.
NI1-NI3 open the shared isolation window of ``tests/_isolation.py`` on the
module's source, its registry clients a stand-in that counts every call; NI4
drives the real app's lifespan through the boot guard suite's harness, its
startup checks stood in.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_ni1_import_reaches_nothing": 3.0,
    "test_ni2_one_first_check": 3.0,
    "test_ni3_a_check_that_fails_is_an_offline_status": 3.0,
    "test_ni4_the_api_checks_at_its_start": 8.0,
}

_NM = "opti_oignon.network_manager"


class _Engine:
    def __init__(self, calls, healthy=True, raises=False):
        self.calls, self.healthy, self.raises = calls, healthy, raises
        self.delay = 0.0

    def health_check(self):
        self.calls.append("health")
        if self.delay:
            import time

            time.sleep(self.delay)      # the readers all stand at the check before the first one ends
        if self.raises:
            raise ConnectionError("the engine refused the connection")
        return self.healthy


def _clients(calls, engine):
    mod = types.ModuleType("opti_oignon.registry_clients")

    def backend_for(*args, **kwargs):
        calls.append("backend_for")
        return engine

    def installed_model_names(*args, **kwargs):
        calls.append("list")
        return ["mxbai-embed-large:latest"]

    mod.backend_for = backend_for
    mod.installed_model_names = installed_model_names
    return mod


def _open(healthy=True, raises=False):
    calls = []
    engine = _Engine(calls, healthy=healthy, raises=raises)
    threads_before = threading.active_count()
    loaded, restore = isolate(targets={_NM: source("network_manager.py")},
                              seeded={"opti_oignon.registry_clients": _clients(calls, engine)})
    loaded[_NM].engine = engine
    return loaded[_NM], calls, threads_before, restore


def test_ni1_import_reaches_nothing():
    """NI1 -- the module loaded: no health check, no listing, no thread; the singleton has not checked. A check run
    at import -> RED."""
    mod, calls, threads_before, restore = _open()
    try:
        assert calls == [], calls
        assert mod.network_manager is not None and mod.network_manager.checked is False
        assert mod.network_manager._poll_thread is None
        assert threading.active_count() == threads_before
    finally:
        restore()


def test_ni2_one_first_check(tmp_path):
    """NI2 -- ``first_check`` checks once (the engine's health, then the listing of the embedding model) and the
    status says online; a second call and later reads check nothing; a fresh manager read first through ``is_online``
    checks then, once, eight threads reading it at once; one polled first (the /poll route, the polling thread) reads
    without a second check. A check repeated, or a read answering before any check -> RED."""
    mod, calls, _threads, restore = _open()
    try:
        manager = mod.NetworkManager(config_path=tmp_path / "net.yaml")
        status = manager.first_check()
        assert status.online is True and manager.checked is True
        assert calls.count("health") == 1, calls
        manager.first_check()
        assert manager.is_online is True
        assert calls.count("health") == 1, calls
        fresh = mod.NetworkManager(config_path=tmp_path / "net.yaml")
        calls.clear()
        mod.engine.delay = 0.2
        gate = threading.Barrier(8)
        seen = []

        def read():
            gate.wait()
            seen.append(fresh.is_online)

        readers = [threading.Thread(target=read) for _ in range(8)]
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join(10)
        assert seen == [True] * 8, seen
        assert calls.count("health") == 1, calls
        polled = mod.NetworkManager(config_path=tmp_path / "net.yaml")
        mod.engine.delay = 0.0
        calls.clear()
        polled.poll_once()          # as the /poll route or the polling thread would, before any read
        assert polled.checked is True and polled.is_online is True
        assert calls.count("health") == 1, calls
    finally:
        restore()


def test_ni3_a_check_that_fails_is_an_offline_status(tmp_path):
    """NI3 -- an engine whose health check raises, and a poll that raises itself: ``first_check`` raises nothing,
    the manager is checked and offline, and the next read checks nothing more. A failed check raised into its caller,
    or retried at each read -> RED."""
    mod, calls, _threads, restore = _open(raises=True)
    try:
        manager = mod.NetworkManager(config_path=tmp_path / "net.yaml")
        status = manager.first_check()
        assert status.online is False and manager.checked is True
        assert manager.is_online is False
        assert calls.count("health") == 1, calls

        polls = []

        def broken():
            polls.append("poll")
            raise RuntimeError("the poll itself failed")

        other = mod.NetworkManager(config_path=tmp_path / "net.yaml")
        other.poll_once = broken
        status = other.first_check()
        assert status.online is False and other.checked is True
        assert [other.is_online, other.is_online] == [False, False]
        other.first_check()
        assert polls == ["poll"], polls
    finally:
        restore()


async def _serve(app, state):
    """The lifespan entered (the API then serves: ``served`` is set, never reset) and left."""
    async with app.router.lifespan_context(app):
        state["served"] = True
    return True


def test_ni4_the_api_checks_at_its_start():
    """NI4 -- the real app's lifespan, its startup checks passing stand-ins: the network manager's first check runs
    once, before the API serves, off the event loop; a first check that raises leaves the startup complete. An API
    that serves before its first check (a check after it, or left running beside it), a check on the event loop, or a
    boot stopped by it -> RED."""
    import test_startup_boot_guard_contracts as bg_t

    harness, restore = bg_t._real_app_harness()
    import opti_oignon.network_manager as nm

    real = nm.network_manager.first_check
    counted = []
    state = {"served": False}

    def where():
        try:
            asyncio.get_running_loop()
            return "on the event loop"
        except RuntimeError:
            return "off the loop"

    try:
        undo = bg_t._install_real_stub_checks(harness["sc"])
        try:
            def first():
                place = where()
                time.sleep(0.2)     # a check left running beside the API sees it serve before it ends
                counted.append(("first", state["served"], place))

            nm.network_manager.first_check = first
            assert asyncio.run(_serve(harness["app"], state)) is True
            assert counted == [("first", False, "off the loop")], counted
            state["served"] = False

            def broken():
                counted.append("broken")
                raise RuntimeError("the first check failed")

            nm.network_manager.first_check = broken
            assert asyncio.run(_serve(harness["app"], state)) is True
            assert counted[1:] == ["broken"], counted
        finally:
            undo()
    finally:
        nm.network_manager.first_check = real
        restore()
