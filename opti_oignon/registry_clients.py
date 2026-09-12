#!/usr/bin/env python3
"""Shared clients over the inference registry for the routes that used to
build a client of their own.

Five routes built a one-shot client per request from the client library,
some with a host of their own -- a bypass by construction: a request that
picks its own host is admitted by no governor and labelled by nothing.
These two clients keep the routes' surface -- a callable that answers a
message list with text, a streamer the agent loop reads -- and send every
request through the registry. The host argument is accepted for the
former signature and unused: where a model is served is the registry's.
"""

import logging

checkpoint_before_apply = True

logger = logging.getLogger(__name__)


def _resolve_backend(model=None):
    """The registry's backend for ``model``, the active one without, or None."""
    try:
        from opti_oignon.inference_backend import get_backend_registry
    except Exception as exc:  # noqa: BLE001 - absence is an answer here
        logger.debug("Inference registry unavailable: %s", exc)
        return None
    try:
        registry = get_backend_registry()
        return registry.resolve_backend(model) if model else registry.active
    except Exception as exc:  # noqa: BLE001 - a broken registry is absence
        logger.debug("Inference registry could not resolve %s: %s", model, exc)
        return None


# ---------------------------------------------------------------------------
# The model catalogue, through the registry
# ---------------------------------------------------------------------------
#
# Twelve modules used to ask the client library which models are installed
# and what a model is made of -- ``list()`` and ``show()`` -- and each folded
# the client's two answer shapes on its own. The registry answers both on
# the backend contract; these three read it so no module keeps a client for
# the catalogue alone. The distinction that matters is kept on purpose:
# ``None`` is "no backend is registered, nobody looked", an empty list is a
# backend that looked and found nothing.

def backend_for(model=None):
    """The registry's backend for ``model``, the active one without, or None."""
    return _resolve_backend(model)


def installed_models():
    """The active backend's model records, or None when no backend is registered."""
    backend = _resolve_backend()
    if backend is None:
        logger.debug("No inference backend is registered; no model catalogue to read")
        return None
    return list(backend.list_models() or [])


def installed_model_names():
    """The names the active backend serves, or None when no backend is registered."""
    records = installed_models()
    if records is None:
        return None
    names = []
    for record in records:
        name = getattr(record, "name", None) or (record.get("name") if isinstance(record, dict) else None)
        if name:
            names.append(str(name))
    return names


def describe_model(model):
    """The registry's ``model_info`` for ``model``: None when no backend is registered or none knows it."""
    backend = _resolve_backend(model)
    if backend is None:
        logger.debug("No inference backend is registered; nothing to describe %s with", model)
        return None
    return backend.model_info(model)


def _require_backend(model):
    backend = _resolve_backend(model)
    if backend is None:
        raise RuntimeError(
            f"no inference backend is registered in the registry for {model!r}; "
            "refusing rather than calling a client of its own"
        )
    return backend


class OneShotChatClient:
    """A callable over the registry: a message list in, the assistant's text out."""

    def __init__(self, model, *, host=None):
        self._model = model
        self._host = host  # accepted for the former signature; the registry decides where

    def __call__(self, messages):
        backend = _require_backend(self._model)
        response = backend.generate(model=self._model, messages=list(messages), options=None)
        return str(getattr(response, "content", "") or "")


class ModelStreamClient:
    """A streamer over the registry in the loop's shape: one chunk per turn.

    A turn is one ``generate`` with the tool schemas as an engine option,
    handed back as ``{"message": {"content", "tool_calls"}}`` -- the client
    shape the loop reads, tool calls as the backend delivered them.
    """

    def __init__(self, model, *, host=None):
        self._model = model
        self._host = host

    def stream(self, messages, tools=None):
        backend = _require_backend(self._model)
        options = {"tools": list(tools)} if tools else None
        response = backend.generate(model=self._model, messages=list(messages), options=options)
        to_dict = getattr(response, "to_dict", None)
        if callable(to_dict):
            yield to_dict()
            return
        yield {"message": {
            "content": getattr(response, "content", "") or "",
            "tool_calls": list(getattr(response, "tool_calls", None) or []),
        }}
