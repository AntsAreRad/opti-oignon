#!/usr/bin/env python3
"""Async client wrapper bridging Veilid's async API to the app.

Veilid's Python API is asyncio-native; most of Opti-Oignon's surface (the node
lifecycle, the sync route handlers) is not. This wrapper owns a single asyncio
event loop on a dedicated daemon thread and submits Veilid coroutines to it, so
the framework's async work never runs on -- and never blocks -- the FastAPI
event loop.

Two surfaces are offered:

- A synchronous connector surface (``connect`` / ``attach`` / ``detach`` /
  ``shutdown`` / ``attachment_state``) that the node drives from its own
  background thread. Each call submits to the dedicated loop and waits with a
  timeout. These must be called off the main event loop (the node thread is);
  calling them from inside the FastAPI loop would block it.
- An async surface (``aconnect`` / ``aattach`` / ``adetach`` / ``ashutdown``)
  that a future async route can ``await`` from the FastAPI loop without blocking
  it: the work is scheduled on the dedicated loop and awaited via a wrapped
  future.

Fail-secure throughout: every operation is timeout-bounded (no hang can wedge a
caller), and only typed VeilidError instances escape -- a timeout surfaces as
VeilidTimeout, any underlying error is wrapped as VeilidError, and the
best-effort ``attachment_state`` read never raises. The heavy ``veilid`` import
is lazy and lives in the connect path, so this module collects without it.

The Veilid coroutines target the documented API (``api_connector`` ->
``attach`` / ``detach`` / ``get_state`` / ``release``); the api factory is
injectable so the loop-bridge, the timeouts, and the fail-secure behaviour are
exercised with a fake, without the framework or a live server.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import threading
from typing import Any, Callable

from opti_oignon.veilid.guard import (
    VeilidError,
    VeilidTimeout,
    VeilidUnavailable,
    veilid_available,
)

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

DEFAULT_TIMEOUT = 15.0


class VeilidClient:
    """A loop-bridge connector for one Veilid node.

    Implements the connector protocol the node drives (connect / attach / detach
    / shutdown / attachment_state). The dedicated loop is created lazily on the
    first operation and torn down on shutdown, so a stopped client can start
    again. The api object is owned solely by the dedicated loop; the sync and
    async surfaces both route work to it, never crossing loops.
    """

    def __init__(
        self,
        *,
        api_factory: Callable[[Callable[[Any], None]], Any] | None = None,
        host: str | None = None,
        port: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._api: Any = None
        self._api_factory = api_factory
        self._host = host
        self._port = port
        self._timeout = float(timeout)
        self._attach_lock = threading.Lock()
        self._last_attachment = ""

    # Dedicated loop / thread

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if (
                self._loop is not None
                and self._thread is not None
                and self._thread.is_alive()
            ):
                return self._loop
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=self._run_loop,
                args=(loop,),
                name="veilid-client-loop",
                daemon=True,
            )
            self._loop = loop
            self._thread = thread
            thread.start()
            return loop

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:  # pragma: no cover - teardown housekeeping
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
            except Exception:
                pass
            loop.close()

    def _submit(self, coro: Any, timeout: float | None = None) -> Any:
        """Run a coroutine on the dedicated loop from a worker thread, bounded."""
        budget = self._timeout if timeout is None else float(timeout)
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(budget)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise VeilidTimeout(f"veilid operation timed out after {budget}s") from exc
        except VeilidError:
            raise
        except Exception as exc:
            raise VeilidError(f"veilid operation failed: {exc}") from exc

    async def _await_on_loop(self, coro: Any, timeout: float | None = None) -> Any:
        """Await a dedicated-loop coroutine from another loop, without blocking it."""
        budget = self._timeout if timeout is None else float(timeout)
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), budget)
        except (concurrent.futures.TimeoutError, asyncio.TimeoutError) as exc:
            future.cancel()
            raise VeilidTimeout(f"veilid operation timed out after {budget}s") from exc
        except VeilidError:
            raise
        except Exception as exc:
            raise VeilidError(f"veilid operation failed: {exc}") from exc

    def _stop_loop(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:  # pragma: no cover - loop already gone
            pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._timeout)

    # Update callback (records the latest attachment state; never raises)

    def _on_update(self, update: Any) -> None:
        try:
            attachment = getattr(update, "attachment", None)
            state = getattr(attachment, "state", None)
            name = getattr(state, "name", None) or (str(state) if state is not None else None)
            if name:
                with self._attach_lock:
                    self._last_attachment = str(name)
        except Exception:  # pragma: no cover - the callback must never raise
            logger.debug("veilid update parse failed", exc_info=True)

    def _resolve_factory(self) -> Callable[[Callable[[Any], None]], Any]:
        if self._api_factory is not None:
            return self._api_factory
        host, port = self._host, self._port

        async def _factory(callback: Callable[[Any], None]) -> Any:
            import veilid  # heavy, optional; never imported at module load

            if host is not None and port is not None:
                return await veilid.api_connector(callback, host, port)
            return await veilid.api_connector(callback)

        return _factory

    # Coroutines (run on the dedicated loop)

    async def _aconnect(self) -> None:
        factory = self._resolve_factory()
        api = factory(self._on_update)
        if inspect.isawaitable(api):
            api = await api
        with self._lock:
            previous, self._api = self._api, api
        if previous is not None:
            # VLD-02: a reconnect must not leak the prior api connection. The
            # new api is already installed, so the release is best-effort and
            # a failing release never breaks the reconnect.
            try:
                await previous.release()
            except Exception:  # pragma: no cover - release is best-effort
                logger.debug("previous veilid api release failed", exc_info=True)
        with self._attach_lock:
            self._last_attachment = ""

    async def _aattach(self) -> None:
        api = self._api
        if api is None:
            raise VeilidError("attach called before connect")
        await api.attach()

    async def _adetach(self) -> None:
        api = self._api
        if api is None:
            raise VeilidError("detach called before connect")
        await api.detach()

    async def _ashutdown(self) -> None:
        with self._lock:
            api = self._api
            self._api = None
        if api is not None:
            await api.release()

    async def _aapp_call(self, target: Any, message: Any) -> Any:
        """Send a request to a peer over a private route and return its reply.

        Drives the documented ``app_call`` request/response primitive: ``target``
        is the peer's routing key (or a resolved route), ``message`` the request
        bytes, and the reply is the peer's answer bytes. Raises a typed
        VeilidError when called before connect.
        """
        api = self._api
        if api is None:
            raise VeilidError("app_call called before connect")
        return await api.app_call(target, message)

    # Synchronous connector surface (driven by the node, off the main loop)

    def connect(self) -> None:
        if self._api_factory is None and not veilid_available():
            raise VeilidUnavailable("veilid framework not installed")
        self._submit(self._aconnect())

    def attach(self) -> None:
        self._submit(self._aattach())

    def detach(self) -> None:
        self._submit(self._adetach())

    def shutdown(self) -> None:
        """Release the api and stop the dedicated loop. Always tears the loop down."""
        with self._lock:
            has_loop = (
                self._loop is not None
                and self._thread is not None
                and self._thread.is_alive()
            )
            has_api = self._api is not None
        if not has_loop and not has_api:
            return
        try:
            if has_loop:
                self._submit(self._ashutdown())
        finally:
            self._stop_loop()

    def attachment_state(self) -> str:
        """The last attachment state Veilid reported; instant and never raises."""
        with self._attach_lock:
            return self._last_attachment

    def app_call(self, target: Any, message: Any, *, timeout: float | None = None) -> Any:
        """Send a request to a peer over a private route, bounded; return the reply.

        Submits the request/response coroutine to the dedicated loop and waits with
        a timeout, off the main event loop (the node thread is). Fail-secure: a
        stall surfaces as VeilidTimeout, any underlying error is wrapped as
        VeilidError, so a hostile or unreachable peer can never wedge the caller.
        """
        return self._submit(self._aapp_call(target, message), timeout)

    # Async surface (awaitable from the FastAPI loop without blocking it)

    async def aconnect(self) -> None:
        if self._api_factory is None and not veilid_available():
            raise VeilidUnavailable("veilid framework not installed")
        await self._await_on_loop(self._aconnect())

    async def aattach(self) -> None:
        await self._await_on_loop(self._aattach())

    async def adetach(self) -> None:
        await self._await_on_loop(self._adetach())

    async def ashutdown(self) -> None:
        with self._lock:
            has_loop = (
                self._loop is not None
                and self._thread is not None
                and self._thread.is_alive()
            )
        if has_loop:
            try:
                await self._await_on_loop(self._ashutdown())
            finally:
                self._stop_loop()

    async def aapp_call(
        self, target: Any, message: Any, *, timeout: float | None = None
    ) -> Any:
        """Await a request/response over a private route from the FastAPI loop.

        Schedules the coroutine on the dedicated loop and awaits it via a wrapped
        future, never blocking the caller's loop; the same fail-secure, timeout-
        bounded behaviour as the synchronous surface.
        """
        return await self._await_on_loop(self._aapp_call(target, message), timeout)

    # Introspection (for tests / status)

    def is_loop_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()
