"""The seam between the being's callers and its two engines.

``call(request_bytes) -> response_bytes`` answers with the native engine when
the handshake agrees, and with the Python reference otherwise. The handshake
compares the native engine's whole identity -- its version, the wire version,
the digest of every law and table file it embeds, its limits, operations and
refusal codes -- with the reference's, byte for byte. A native core built
from other law files, or an older one left installed, would answer for a
different world: it is not used, and that is said once.

The native core is looked up at the first call, never at import.
"""

import logging

checkpoint_before_apply = True

logger = logging.getLogger(__name__)

_state = {"checked": False, "native": None}


def _load_native():
    try:
        from opti_oignon import native
    except ImportError:
        return None
    return native.load()


def handshake(module):
    """True when the native module is present and its identity is the reference's."""
    from .ref import protocol

    engine = getattr(module, "allium_engine", None)
    answer = getattr(module, "allium_call", None)
    if engine is None or answer is None:
        return False
    try:
        return bytes(engine()) == protocol.engine_info()
    except Exception:  # noqa: BLE001 - a module that cannot say who it is is not used
        return False


def _native_call():
    if not _state["checked"]:
        _state["checked"] = True
        module = _load_native()
        if module is not None and handshake(module):
            _state["native"] = module.allium_call
        elif module is not None and getattr(module, "allium_call", None) is not None:
            logger.warning(
                "the native engine does not answer for the reference's world "
                "(a stale build or other law files): the reference answers"
            )
    return _state["native"]


def native_in_use():
    """Whether calls go to the native engine (after the first handshake)."""
    return _native_call() is not None


def call(request):
    """Answer one request, natively when the handshake agreed."""
    answer = _native_call()
    if answer is not None:
        return bytes(answer(bytes(request)))
    from .ref import protocol

    return protocol.call(bytes(request))


def reset():
    """Forget the handshake, for the contracts that swap the native core."""
    _state["checked"] = False
    _state["native"] = None
