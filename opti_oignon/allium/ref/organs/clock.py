"""The clock: three genes that repress one another in a ring, entrained by light.

The reference, mirrored in the twin. ``clock_m`` is repressed by
``clock_e``, ``clock_d`` by ``clock_m`` and ``clock_e`` by ``clock_d``. At
each awake fast step, for each gene: its repression is ``hill_down`` of the
gene before it in the ring, its production ``alpha`` times that repression,
and its decay ``beta`` times its own level; light, the constant gain times
the sun, adds to ``clock_m``. Each level is then held to ``0..=CMAX``, and
every level that would have gone above ``CMAX`` is counted as a clip in the
trace. Thirteen units a step.

The constants (``alpha``, ``beta``, ``k``, ``n`` per gene, and ``light``)
come from the genome and live in the organ's state. The clock is quiescent
in dormancy: while the being sleeps its step does nothing and costs nothing,
and its one ``jump`` at the wake is the identity.
"""

from ... import fx

checkpoint_before_apply = True

CHANNEL = "circadian"
# The gene each gene is repressed by: m by e, d by m, e by d.
RING = (2, 0, 1)


def init(constants, k):
    """The clock at a being's first minute: its levels from the law, its constants from the genome."""
    return {"k": k, "p": list(constants["init"])}


def fast(org, cx):
    """One fast step: the three levels move; 13 units awake, nothing while dormant."""
    if cx.dormant:
        return
    work = cx.work
    k = org["k"]
    p = org["p"]
    alpha, beta, ks, ns = k["alpha"], k["beta"], k["k"], k["n"]
    prod = [0, 0, 0]
    deg = [0, 0, 0]
    for i in range(3):
        repression = fx.hill_down(p[RING[i]], ks[i], ns[i], work)
        prod[i] = fx.mul(alpha[i], repression, work)
        deg[i] = fx.mul(beta[i], p[i], work)
    light = fx.mul(k["light"], cx.sun, work)
    raw = (p[0] + prod[0] + light - deg[0], p[1] + prod[1] - deg[1], p[2] + prod[2] - deg[2])
    levels = []
    for value in raw:
        if value > fx.CMAX:
            cx.clips += 1
        levels.append(fx.sat(value, 0, fx.CMAX, work))
    org["p"] = levels


def jump(org, cx, since, wake):
    """The dormancy from minute ``since`` to the wake at ``wake``, applied at once: nothing changes."""
    return None


def publish(org, cx):
    """The circadian channel: ``clock_m`` shifted down by three, at most ``ONE``."""
    return {CHANNEL: min(fx.ONE, org["p"][0] >> 3)}
