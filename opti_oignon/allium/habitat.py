"""The habitat of a being: its container (a pot or a glass jar), and the rules of the security mode.

The mode changes what is served, never the life. Under Daily everything is
served; under Bulbe -- or a mode that cannot be read, which takes Bulbe's
rules -- a pot's lid is closed, a glass jar is sealed and not opened at all,
and every capability the policy names is gated off. The life goes on under
both: the same journal and the same minute give the same view.

``ModePolicy`` is a closed record of booleans, one per capability, and
``DAILY`` and ``BULBE`` are its two values. Only the exact word ``"daily"``
gets Daily. The policy is code: no configuration reaches it, so none can
loosen it. A capability allowed here still needs its own grant where it is
used.

``layer(soil, mode)`` gives ``(container, layer)``: a ``"pot"`` for
encrypted soil and a ``"jar"`` for glass; ``"open"`` under Daily, and
otherwise ``"bulbe"`` for a pot (its lid closed) and ``"sealed"`` for a jar.

Nothing is imported at module level but the standard library.
"""

from typing import NamedTuple

checkpoint_before_apply = True


class ModePolicy(NamedTuple):
    """What a mode allows: the life, a glass jar opened, and each capability. Never the grants themselves."""

    life: bool
    glass_open: bool
    taste: bool
    voice: bool
    initiatives: bool
    dream_depth_change: bool
    sync: bool
    clear_export: bool


# Daily: everything a grant may then allow; an export in clear never.
DAILY = ModePolicy(life=True, glass_open=True, taste=True, voice=True, initiatives=True, dream_depth_change=True,
                   sync=True, clear_export=False)
# Bulbe: the life goes on, and nothing else.
BULBE = ModePolicy(life=True, glass_open=False, taste=False, voice=False, initiatives=False,
                   dream_depth_change=False, sync=False, clear_export=False)
# The capabilities a gate may be asked about.
CAPABILITIES = ("taste", "voice", "initiatives", "sync", "dream_depth_change")


def policy(mode):
    """``DAILY`` for exactly ``"daily"``; ``BULBE`` for anything else, an unread mode included."""
    return DAILY if mode == "daily" and isinstance(mode, str) else BULBE


def allows(mode, capability):
    """Whether the policy of ``mode`` allows ``capability``; ``KeyError`` for a capability it does not name."""
    if capability not in CAPABILITIES:
        raise KeyError(capability)
    return getattr(policy(mode), capability) is True


def layer(soil, mode):
    """``(container, layer)`` for a being's soil under ``mode``."""
    container = "jar" if soil == "glass" else "pot"
    if policy(mode) is DAILY:
        return (container, "open")
    return (container, "sealed" if container == "jar" else "bulbe")
