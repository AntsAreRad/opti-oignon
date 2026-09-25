"""The reserve: sugar made by light, burnt by respiration, stored as fructan and drawn back.

The reference, mirrored in the twin. At each awake fast step:

* photosynthesis ``P`` is the rate ``ps`` times the saturation of the sun
  and of the soil moisture (read from the bus), held to the room left for
  sugar;
* respiration ``R`` is the rate ``r`` times the saturation of the sugar,
  held to the sugar there is after ``P``;
* above ``theta_s``, sugar is stored as fructan at the rate ``sy``, held to
  the sugar left and to the room left for fructan;
* below ``theta_h``, fructan is drawn back as sugar at the rate ``hy``, held
  to the fructan above its ``core``.

Fifteen units a step. The ledger counters ``made`` and ``burnt`` add ``P``
and ``R`` (saturating at the wire's largest integer), so that sugar plus
fructan always equals their starting sum plus ``made`` minus ``burnt``; the
metabolism of the step, ``P + R``, is kept for the bus. The rates and the
two saturation constants come from the genome and live in the organ's
state. The reserve is quiescent in dormancy.
"""

from ... import fx, wire

checkpoint_before_apply = True

CHANNEL = "metab"


def init(constants, k):
    """The reserve at a being's first minute."""
    return {"burnt": 0, "fructan": constants["fructan0"], "k": k, "made": 0, "metab": 0,
            "sugar": constants["sugar0"]}


def fast(org, cx):
    """One fast step of the reserve; 15 units awake, nothing while dormant."""
    if cx.dormant:
        return
    work = cx.work
    c = cx.constants["chem"]
    k = org["k"]
    sugar = org["sugar"]
    fructan = org["fructan"]
    made = fx.mul(k["ps"], fx.mul(fx.mm(cx.sun, k["km_ps"], work), fx.mm(cx.bus["moisture"], c["k_w"], work),
                                  work), work)
    made = fx.sat(made, 0, c["sugar_max"] - sugar, work)
    burnt = fx.mul(k["r"], fx.mm(sugar, k["km_r"], work), work)
    burnt = fx.sat(burnt, 0, sugar + made, work)
    left = sugar + made - burnt
    stored = fx.mul(k["sy"], fx.sat(left - c["theta_s"], 0, fx.I32_MAX, work), work)
    stored = fx.sat(stored, 0, left, work)
    stored = fx.sat(stored, 0, c["fructan_max"] - fructan, work)
    drawn = fx.mul(k["hy"], fx.sat(c["theta_h"] - left, 0, fx.I32_MAX, work), work)
    drawn = fx.sat(drawn, 0, fructan - c["core"], work)
    org["sugar"] = left - stored + drawn
    org["fructan"] = fructan + stored - drawn
    org["made"] = min(wire.MAX_INT, org["made"] + made)
    org["burnt"] = min(wire.MAX_INT, org["burnt"] + burnt)
    org["metab"] = made + burnt


def jump(org, cx, since, wake):
    """The dormancy from minute ``since`` to the wake at ``wake``, applied at once: nothing changes."""
    return None


def publish(org, cx):
    """The metabolism channel: the step's ``P + R`` shifted up by ``metab_shift``, at most ``ONE``."""
    return {CHANNEL: min(fx.ONE, org["metab"] << cx.constants["chem"]["metab_shift"])}
