"""The web gate: whether a request may leave the process for the web now.

One rule for every path that reaches the web: exactly Daily mode, the search
kill switch released. The page fetch and the web search ask it at each
request; every other path asks it through the front door,
``opti_oignon.egress``. Standard library only, and nothing is read at import:
the switch and the mode are asked at each call.
"""

checkpoint_before_apply = True

_REFUSALS = {
    "kill_switch": "Web search is refused: the search kill switch is engaged.",
    "mode": "Web search is refused outside Daily mode.",
    "unreadable": "Web search is refused: the kill switch cannot be read.",
}


class WebSearchRefused(RuntimeError):
    """A web request refused by the gate, carrying the refusal by name."""

    def __init__(self, refusal: str):
        super().__init__(_REFUSALS[refusal])
        self.refusal = refusal


def search_refusal() -> str | None:
    """Why no web request may leave the process now, or None.

    ``kill_switch`` while the switch is engaged; ``unreadable`` when it cannot
    be read, its module absent included; ``mode`` in any mode but exactly
    ``"daily"``, and when the mode cannot be read. Both are asked at every
    request: the switch reads its record again, and the mode answers from
    the process's cache, which a mode change made in this process
    refreshes.
    """
    try:
        from opti_oignon.search_killswitch import search_killswitch
        if search_killswitch.is_killed():
            return "kill_switch"
    except Exception:
        return "unreadable"
    try:
        from opti_oignon.security_mode import get_current_mode
        mode = get_current_mode()
    except Exception:
        return "mode"
    return None if isinstance(mode, str) and mode == "daily" else "mode"
