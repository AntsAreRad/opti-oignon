#!/usr/bin/env python3
"""The remote core backend: the registry member that reaches the core daemon.

Registered in a process's inference registry when ``core.yaml`` enables
the daemon, it makes the daemon serve that process's inference. To the
registry it is a backend like any other; every request it carries crosses
the process boundary over the loopback and lands in the daemon's own
registry, where admission, provenance and schema apply. It answers with
the daemon's answers, reports itself unhealthy when the daemon is absent
and never raises for that, and refuses a request by name rather than hang.
"""

import json
import logging
import urllib.error
import urllib.request

# The daemon is on the loopback: no proxy applies, whatever the environment
# says, and the opener is this module's own rather than the global urlopen.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

from opti_oignon.inference_backend import (
    BackendModelInfo,
    ChatResponse,
    InferenceBackend,
    StreamChunk,
)

checkpoint_before_apply = True

logger = logging.getLogger(__name__)

NAME = "core"


class RemoteCoreBackend(InferenceBackend):
    """Every request forwarded to the core daemon over the loopback."""

    def __init__(self, base_url, token="", timeout_s=30.0):
        self.base_url = str(base_url).rstrip("/")
        self.token = token or ""
        self.timeout_s = float(timeout_s)

    @property
    def name(self):
        return NAME

    @property
    def display_name(self):
        return "core daemon (loopback)"

    def _request(self, path, payload=None, method=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method or ("POST" if data is not None else "GET"),
            headers={"Content-Type": "application/json"},
        )
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            return _OPENER.open(request, timeout=self.timeout_s)
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", "replace")
            raise RuntimeError(f"core daemon at {self.base_url} refused {path}: {err.code} {body}") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RuntimeError(f"core daemon unreachable at {self.base_url}: {exc}") from None

    def health_check(self):
        try:
            with self._request("/health") as resp:
                return bool(json.loads(resp.read().decode("utf-8")).get("ok"))
        except Exception as exc:  # noqa: BLE001 - unhealthy is an answer, not an error
            logger.debug("core daemon health: %s", exc)
            return False

    def list_models(self):
        """The daemon's listing; ``None`` when it could not be read or the daemon does not know."""
        try:
            with self._request("/models") as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - no daemon, listing unknown
            logger.debug("core daemon models: %s", exc)
            return None
        models = payload.get("models") if isinstance(payload, dict) else None
        if models is None or payload.get("known") is False:
            return None
        return [BackendModelInfo(name=str(m.get("name", "")), backend=NAME) for m in models]

    def model_info(self, model_name):
        # The daemon's registry resolves the model; this backend recognises
        # every name so that, when it is active, every request crosses to it.
        return BackendModelInfo(name=model_name, backend=NAME)

    def loaded_models(self):
        # The daemon has no route for its loaded set yet, so the answer is
        # unknown, said as such; a route is a decision, not a default.
        return None

    def embed(self, model, text, timeout=None):
        # Embeddings do not cross the boundary yet, for the same reason.
        return None

    def embed_many(self, model, texts, timeout=None):
        return None

    @staticmethod
    def _payload(model, messages, options, keep_alive, think):
        return {"model": model, "messages": messages, "options": options, "keep_alive": keep_alive, "think": bool(think)}

    def generate(self, model, messages, options=None, keep_alive="30m", think=False, images=None):
        with self._request("/inference/generate", self._payload(model, messages, options, keep_alive, think)) as resp:
            wire = json.loads(resp.read().decode("utf-8"))
        return ChatResponse(
            content=wire.get("content", "") or "",
            thinking=wire.get("thinking"),
            model=wire.get("model", model) or model,
            done=bool(wire.get("done", True)),
            total_duration=wire.get("total_duration"),
            extra=dict(wire.get("extra") or {}),
            tool_calls=list(wire.get("tool_calls") or []),
        )

    def stream(self, model, messages, options=None, keep_alive="30m", think=False, images=None):
        with self._request("/inference/stream", self._payload(model, messages, options, keep_alive, think)) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                yield StreamChunk(
                    content=chunk.get("content", "") or "",
                    thinking=chunk.get("thinking", "") or "",
                    done=bool(chunk.get("done", False)),
                    model=chunk.get("model", model) or model,
                )


def register_from_config(registry, path=None):
    """Register and activate the remote backend when ``core.yaml`` enables the daemon.

    Returns the backend, or None when the daemon is disabled. A malformed
    configuration raises: a process that expected the daemon must not fall
    back to a local backend in silence.
    """
    from opti_oignon.core_daemon import load_config

    config = load_config(path)
    if not config.enabled:
        return None
    backend = RemoteCoreBackend(config.base_url, token=config.token, timeout_s=config.timeout_s)
    registry.register(backend)
    registry.activate(backend.name)
    return backend
