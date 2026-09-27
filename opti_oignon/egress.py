"""The front door for outbound requests: it asks the gates and names their answers.

Every outbound request asks a gate before it leaves: the web gate
(``opti_oignon.web_gate``) for a host the platform does not own, the local
rule (``inference_backend.local_refusal``) for this machine and the
operator's own services. This module has no rule of its own. Each question
imports its gate at the call and returns what the gate answers, with a text
that carries the caller's label. A gate that cannot be imported, for any
reason, is a refusal named ``unreadable``: a request whose gate cannot be
asked does not leave.

Standard library only, and nothing is read at import.
"""

checkpoint_before_apply = True

_WEB_TEXTS = {
    "mode": "{label} is refused outside Daily mode.",
    "kill_switch": "{label} is refused while the kill switch is engaged.",
    "unreadable": "{label} is refused: the kill switch cannot be read.",
}


class EgressRefused(RuntimeError):
    """An outbound request refused by a gate, carrying the gate and the refusal by name."""

    def __init__(self, gate: str, refusal: str, text: str):
        super().__init__(text)
        self.gate = gate
        self.refusal = refusal
        self.text = text


def _web(label: str) -> tuple[str, str] | None:
    try:
        from opti_oignon import web_gate
        refusal = web_gate.search_refusal()
    except Exception:
        refusal = "unreadable"
    if refusal is None:
        return None
    template = _WEB_TEXTS.get(refusal)
    text = template.format(label=label) if template else f"{label} is refused by the web gate ({refusal})."
    return refusal, text


def _local(label: str, endpoint: str | None) -> tuple[str, str] | None:
    try:
        from opti_oignon import inference_backend
        text = inference_backend.local_refusal(label, endpoint)
    except Exception:
        return "unreadable", f"{label} is refused: the local rule cannot be read."
    return None if text is None else ("rule", text)


def web_refusal(label: str = "Web request") -> tuple[str, str] | None:
    """The web gate's refusal as ``(name, text)``, or None when a web request may leave now."""
    return _web(label)


def local_refusal(label: str, endpoint: str | None) -> str | None:
    """The local rule's refusal text for a request to ``endpoint``, or None."""
    found = _local(label, endpoint)
    return None if found is None else found[1]


def require_web(label: str) -> None:
    """Return when a web request may leave now; raise its refusal otherwise.

    Raises:
        EgressRefused: The web gate refused, or cannot be asked.
    """
    found = _web(label)
    if found is not None:
        raise EgressRefused("web", found[0], found[1])


def require_local(label: str, endpoint: str | None) -> None:
    """Return when a request to ``endpoint`` may leave now; raise its refusal otherwise.

    Raises:
        EgressRefused: The local rule refused, or cannot be asked.
    """
    found = _local(label, endpoint)
    if found is not None:
        raise EgressRefused("local", found[0], found[1])


def require_delegated(label: str, endpoint: str | None) -> None:
    """For a request another program sends on: ask where it goes, then what it causes.

    The local rule is asked first, for ``endpoint``; the web gate second.

    Raises:
        EgressRefused: Either gate refused, or cannot be asked.
    """
    require_local(label, endpoint)
    require_web(label)
