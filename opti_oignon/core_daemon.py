#!/usr/bin/env python3
"""The core daemon: the resident process that serves inference to the rest.

A loopback HTTP server over the inference registry, in the standard
library, importing only the core. A request from another process reaches
a backend exactly as a chat turn does -- through the registry -- so the
governor's admission, the provenance labels and the schema and tool
options apply to it unchanged. The routes are few and flat: health, one
generate, one stream as JSON lines, one admission for a pack's ticket, a
model list. It fails closed: an unknown route, a body that is not JSON, a
request without a model, a model no backend serves and a missing token
are each a refusal by name, and a host that is not the loopback is refused
before the socket is bound.

Python first, behind the surface the registry already has; the native
daemon comes by strangling, as the memory's native core did.
"""

import hmac
import json
import logging
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

checkpoint_before_apply = True

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).resolve().parent / "config" / "core.yaml"
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
SOURCE = "core-daemon"


class CoreError(ValueError):
    """The daemon cannot run as configured."""


@dataclass(frozen=True)
class CoreConfig:
    enabled: bool
    host: str
    port: int
    token: str
    timeout_s: float

    def validate(self):
        errors = []
        if self.host not in _LOOPBACK:
            errors.append(f"host: {self.host!r} is not the loopback; the daemon binds nothing else")
        if not isinstance(self.port, int) or not 0 <= self.port <= 65535:
            errors.append(f"port: {self.port!r} is not a port")
        if not isinstance(self.token, str):
            errors.append("token: must be a string, empty for none")
        try:
            if float(self.timeout_s) <= 0:
                errors.append(f"timeout_s: {self.timeout_s!r} is not positive")
        except (TypeError, ValueError):
            errors.append(f"timeout_s: {self.timeout_s!r} is not a number")
        return errors

    def validate_or_raise(self):
        errors = self.validate()
        if errors:
            raise CoreError("; ".join(errors))
        return self

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"


def load_config(path=None):
    """The daemon's configuration from ``core.yaml``, refused when malformed."""
    import yaml

    raw = yaml.safe_load(Path(path or _CONFIG).read_text(encoding="utf-8")) or {}
    section = raw.get("core") or {}
    try:
        config = CoreConfig(
            enabled=bool(section.get("enabled", False)),
            host=str(section.get("host", "127.0.0.1")),
            port=int(section.get("port", 7411)),
            token=str(section.get("token", "") or ""),
            timeout_s=float(section.get("timeout_s", 30)),
        )
    except (TypeError, ValueError) as exc:
        raise CoreError(f"core configuration is malformed: {exc!r}") from exc
    return config.validate_or_raise()


def _registry():
    from opti_oignon.inference_backend import get_backend_registry

    return get_backend_registry()


def _admission(model, options):
    from opti_oignon.inference_backend import _governor_admission

    _governor_admission(model, options)


def _response_wire(response, model):
    return {
        "content": getattr(response, "content", "") or "",
        "thinking": getattr(response, "thinking", None),
        "model": getattr(response, "model", "") or model,
        "done": bool(getattr(response, "done", True)),
        "total_duration": getattr(response, "total_duration", None),
        "extra": dict(getattr(response, "extra", None) or {}),
        "tool_calls": list(getattr(response, "tool_calls", None) or []),
    }


class CoreService:
    """The daemon's routes, free of the transport: each returns (status, payload)."""

    def __init__(self, token=""):
        self._token = token or ""

    def authorized(self, header):
        if not self._token:
            return True
        if not header or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[len("Bearer "):], self._token)

    def health(self):
        try:
            registry = _registry()
            active = registry.active_name or (registry.active.name if registry.active else None)
        except Exception as exc:  # noqa: BLE001 - health reports, it does not raise
            return 503, {"ok": False, "source": SOURCE, "error": f"registry unavailable: {exc!r}"}
        return 200, {"ok": True, "source": SOURCE, "backend": active}

    def models(self):
        try:
            backend = _registry().active
            if backend is None:
                return 200, {"models": [], "source": SOURCE}
            models = backend.list_models()
        except Exception as exc:  # noqa: BLE001
            return 503, {"error": f"models unavailable: {exc!r}"}
        return 200, {"models": [m.to_dict() if hasattr(m, "to_dict") else {"name": getattr(m, "name", str(m))} for m in models], "source": SOURCE}

    @staticmethod
    def _request(payload):
        if not isinstance(payload, dict):
            return None, (400, {"error": "the body must be a JSON object"})
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            return None, (400, {"error": "model: a model name is required"})
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None, (400, {"error": "messages: a list of messages is required"})
        options = payload.get("options")
        if options is not None and not isinstance(options, dict):
            return None, (400, {"error": "options: must be an object"})
        return dict(
            model=model, messages=messages, options=options,
            keep_alive=str(payload.get("keep_alive", "30m")), think=bool(payload.get("think", False)),
        ), None

    def _backend(self, model):
        backend = _registry().resolve_backend(model)
        if backend is None:
            return None, (503, {"error": f"no inference backend is registered in the registry for {model!r}"})
        return backend, None

    def generate(self, payload):
        request, error = self._request(payload)
        if error:
            return error
        backend, error = self._backend(request["model"])
        if error:
            return error
        try:
            response = backend.generate(**request)
        except Exception as exc:  # noqa: BLE001 - the backend's refusal, by name
            return 502, {"error": f"backend refused: {exc}"}
        return 200, _response_wire(response, request["model"])

    def stream(self, payload):
        """(status, lines): each line a JSON chunk; the status is decided before the first."""
        request, error = self._request(payload)
        if error:
            return error[0], iter([error[1]])
        backend, error = self._backend(request["model"])
        if error:
            return error[0], iter([error[1]])

        def lines():
            for chunk in backend.stream(**request):
                yield {
                    "content": getattr(chunk, "content", "") or "",
                    "thinking": getattr(chunk, "thinking", "") or "",
                    "done": bool(getattr(chunk, "done", False)),
                    "model": getattr(chunk, "model", "") or request["model"],
                }
        return 200, lines()

    def admission(self, payload):
        if not isinstance(payload, dict):
            return 400, {"error": "the body must be a JSON object"}
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            return 400, {"error": "model: a model name is required"}
        options = {}
        if payload.get("requested_ctx") is not None:
            options["num_ctx"] = int(payload["requested_ctx"])
        try:
            _admission(model, options or None)
        except Exception as exc:  # noqa: BLE001 - the governor's refusal, in its words
            return 200, {"admitted": False, "model": model, "reason": str(exc)}
        return 200, {"admitted": True, "model": model, "reason": ""}


class _Handler(BaseHTTPRequestHandler):
    server_version = "opti-oignon-core/0.1"

    def log_message(self, format, *args):  # noqa: A002 - the base class's signature
        logger.debug("core daemon: " + format, *args)

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw.decode("utf-8") or "null"), None
        except (ValueError, UnicodeDecodeError):
            return None, (400, {"error": "the body is not JSON"})

    def do_GET(self):  # noqa: N802 - the base class's name
        service = self.server.service
        if self.path == "/health":
            status, payload = service.health()
            return self._send(status, payload)
        if not service.authorized(self.headers.get("Authorization")):
            return self._send(401, {"error": "a bearer token is required on this route"})
        if self.path == "/models":
            status, payload = service.models()
            return self._send(status, payload)
        return self._send(404, {"error": f"no such route: {self.path}"})

    def do_POST(self):  # noqa: N802 - the base class's name
        service = self.server.service
        if not service.authorized(self.headers.get("Authorization")):
            return self._send(401, {"error": "a bearer token is required on this route"})
        if self.path not in ("/inference/generate", "/inference/stream", "/admission"):
            return self._send(404, {"error": f"no such route: {self.path}"})
        payload, error = self._body()
        if error:
            return self._send(*error)
        if self.path == "/inference/generate":
            return self._send(*service.generate(payload))
        if self.path == "/admission":
            return self._send(*service.admission(payload))
        status, lines = service.stream(payload)
        if status != 200:
            return self._send(status, next(lines))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for line in lines:
            data = (json.dumps(line) + "\n").encode("utf-8")
            self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")


def make_server(config, service):
    """A bound server on the loopback; refuses any other host before binding."""
    config.validate_or_raise()
    server = ThreadingHTTPServer((config.host, config.port), _Handler)
    server.daemon_threads = True
    server.service = service
    return server


def serve(config_path=None):
    """Run the daemon in the foreground over the configured registry."""
    config = load_config(config_path)
    if not config.enabled:
        raise CoreError("the core daemon is disabled in core.yaml; set core.enabled to true")
    from opti_oignon.inference_backend import init_backends_from_config

    init_backends_from_config()
    server = make_server(config, CoreService(token=config.token))
    logger.info("core daemon listening on %s", config.base_url)
    try:
        server.serve_forever()
    finally:
        server.server_close()
