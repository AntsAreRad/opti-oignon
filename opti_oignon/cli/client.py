#!/usr/bin/env python3
"""
HTTP / WebSocket client wrapper -- Opti-Oignon CLI.

Provides a thin layer around ``httpx`` (sync HTTP) and ``websockets``
(async WS) that every CLI command uses to talk to the backend.

Design decisions
----------------
* Sync HTTP for simple REST calls (models, status, backup, rag, config).
* Async WebSocket for streaming chat (``oo ask``).
* All methods raise ``CLIClientError`` on transport / HTTP errors so the
  command layer can print user-friendly messages.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from .config import CLIConfig


class CLIClientError(Exception):
    """Raised when a backend request fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class OOClient:
    """Synchronous + async client for the Opti-Oignon backend API."""

    config: CLIConfig

    # -- internal helpers ---------------------------------------------------

    def _url(self, path: str) -> str:
        """Build a full HTTP URL from a relative path."""
        return f"{self.config.api_url}{path}"

    def _ws_url(self, path: str) -> str:
        """Build a full WebSocket URL from a relative path."""
        return f"{self.config.ws_base}{path}"

    def _http(self) -> httpx.Client:
        """Create an httpx client with the configured timeout."""
        return httpx.Client(timeout=self.config.timeout)

    def _handle_response(self, resp: httpx.Response) -> Any:
        """Raise CLIClientError for non-2xx responses, otherwise return JSON."""
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise CLIClientError(
                f"HTTP {resp.status_code}: {detail}",
                status_code=resp.status_code,
            )
        try:
            return resp.json()
        except Exception:
            return resp.text

    # -- REST helpers -------------------------------------------------------

    def get(self, path: str, **params: Any) -> Any:
        """Perform a GET request and return parsed JSON."""
        with self._http() as client:
            try:
                resp = client.get(self._url(path), params=params or None)
            except httpx.ConnectError as exc:
                raise CLIClientError(
                    f"Cannot connect to {self.config.api_url} -- is the backend running?"
                ) from exc
            except httpx.TimeoutException as exc:
                raise CLIClientError("Request timed out") from exc
            return self._handle_response(resp)

    def post(self, path: str, json_body: dict | None = None, **kwargs: Any) -> Any:
        """Perform a POST request with a JSON body."""
        with self._http() as client:
            try:
                resp = client.post(self._url(path), json=json_body, **kwargs)
            except httpx.ConnectError as exc:
                raise CLIClientError(
                    f"Cannot connect to {self.config.api_url} -- is the backend running?"
                ) from exc
            except httpx.TimeoutException as exc:
                raise CLIClientError("Request timed out") from exc
            return self._handle_response(resp)

    def delete(self, path: str, **params: Any) -> Any:
        """Perform a DELETE request and return parsed JSON."""
        with self._http() as client:
            try:
                resp = client.delete(self._url(path), params=params or None)
            except httpx.ConnectError as exc:
                raise CLIClientError(
                    f"Cannot connect to {self.config.api_url} -- is the backend running?"
                ) from exc
            except httpx.TimeoutException as exc:
                raise CLIClientError("Request timed out") from exc
            return self._handle_response(resp)

    def post_file(self, path: str, filepath: str,
                  extra_fields: dict[str, str] | None = None) -> Any:
        """Upload a file via multipart/form-data POST."""
        import os
        fname = os.path.basename(filepath)
        with open(filepath, "rb") as fh:
            files = {"file": (fname, fh)}
            data = extra_fields or {}
            with self._http() as client:
                try:
                    resp = client.post(self._url(path), files=files, data=data)
                except httpx.ConnectError as exc:
                    raise CLIClientError(
                        f"Cannot connect to {self.config.api_url} -- is the backend running?"
                    ) from exc
                except httpx.TimeoutException as exc:
                    raise CLIClientError("Request timed out") from exc
                return self._handle_response(resp)

    # -- WebSocket streaming ------------------------------------------------

    def stream_chat(
        self,
        message: str,
        model: str | None = None,
        on_token: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        on_metadata: Callable[[dict], None] | None = None,
        on_approval: Callable[[dict], bool | None] | None = None,
        on_resolved: Callable[[dict], None] | None = None,
        pasted: list | None = None,
        documents: list | None = None,
    ) -> str:
        """Send a chat message and stream tokens back.

        Connects to the backend WebSocket chat endpoint, sends a
        ``ChatRequest`` JSON, and processes incoming ``ChatToken``
        messages until a ``done`` or ``error`` frame arrives.

        Parameters
        ----------
        message : str
            The user prompt.
        model : str, optional
            Override model (``None`` = use smart router / config default).
        on_token : callable, optional
            Called with each content token string as it arrives.
        on_thinking : callable, optional
            Called with each thinking token.
        on_metadata : callable, optional
            Called with the metadata dict on the ``done`` frame.
        on_approval : callable, optional
            Called with a tool call the backend holds for the user's
            answer: its id, tool, arguments, the label of each argument and
            its class. True approves it, False denies it, None leaves it
            waiting for an answer from elsewhere. An answer that comes too
            late (the call already decided) is dropped; the stream goes on.
        on_resolved : callable, optional
            Called with how a held call ended: its id, tool, and whether it
            was approved, whoever answered.
        pasted : list, optional
            The ranges of ``message`` the user pasted rather than typed, as
            [start, end] in code points; the server saves each as a document
            part of the words.
        documents : list, optional
            What was read rather than typed -- a file, standard input -- as
            ``{"filename", "content"}`` objects sent beside the message; the
            server saves each as a document part of the turn.

        Returns
        -------
        str
            The full accumulated response text.
        """
        extra = {key: value for key, value in (("pasted", pasted), ("documents", documents)) if value}
        return asyncio.run(
            self._ws_stream(message, model, on_token, on_thinking, on_metadata, on_approval, on_resolved, **extra)
        )

    def answer_approval(self, approval_id: str, allowed: bool) -> Any:
        """Approve or deny a tool call the backend holds for the user's answer."""
        verdict = "approve" if allowed else "deny"
        return self.post(f"/api/security/tool-approval/{approval_id}/{verdict}")

    async def _ws_stream(
        self,
        message: str,
        model: str | None,
        on_token: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
        on_metadata: Callable[[dict], None] | None,
        on_approval: Callable[[dict], bool | None] | None = None,
        on_resolved: Callable[[dict], None] | None = None,
        pasted: list | None = None,
        documents: list | None = None,
    ) -> str:
        try:
            import websockets
        except ImportError:
            raise CLIClientError(
                "The 'websockets' package is required for streaming. "
                "Install it with: pip install websockets"
            )

        ws_url = self._ws_url("/api/chat/stream")
        request_payload = {"message": message}
        effective_model = model or self.config.default_model
        if effective_model:
            request_payload["model"] = effective_model
        if pasted:
            request_payload["pasted"] = [list(item) for item in pasted]
        if documents:
            request_payload["documents"] = [{"filename": d["filename"], "content": d["content"]} for d in documents]

        full_text = ""
        try:
            async with websockets.connect(ws_url) as ws:
                await ws.send(json.dumps(request_payload))
                async for raw in ws:
                    try:
                        frame = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    frame_type = frame.get("type", "")
                    content = frame.get("content", "")
                    metadata = frame.get("metadata")
                    if frame_type == "token":
                        full_text += content
                        if on_token:
                            on_token(content)
                    elif frame_type == "thinking":
                        if on_thinking:
                            on_thinking(content)
                    elif frame_type == "metadata":
                        if on_metadata and metadata:
                            on_metadata(metadata)
                    elif frame_type == "tool_call_pending":
                        # Asked off the event loop: the answer may wait on a
                        # person at the keyboard while frames keep arriving.
                        if on_approval and metadata and metadata.get("approval_id"):
                            allowed = await asyncio.to_thread(on_approval, metadata)
                            if allowed is not None:
                                try:
                                    await asyncio.to_thread(
                                        self.answer_approval, str(metadata["approval_id"]), bool(allowed)
                                    )
                                except CLIClientError:
                                    # Too late: the call was already decided
                                    # (its time ran out, or another surface
                                    # answered). How it ended comes next.
                                    pass
                    elif frame_type == "tool_call_resolved":
                        if on_resolved and metadata:
                            on_resolved(metadata)
                    elif frame_type == "done":
                        if on_metadata and metadata:
                            on_metadata(metadata)
                        break
                    elif frame_type == "error":
                        raise CLIClientError(f"Backend error: {content}")
        except CLIClientError:
            raise
        except ConnectionRefusedError as exc:
            raise CLIClientError(
                f"Cannot connect to {ws_url} -- is the backend running?"
            ) from exc
        except Exception as exc:
            raise CLIClientError(f"WebSocket error: {exc}") from exc

        return full_text
