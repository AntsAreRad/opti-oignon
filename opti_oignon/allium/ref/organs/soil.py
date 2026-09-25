"""The soil: one moisture level, filled by rain and care, emptied by evaporation.

The reference, mirrored in the twin. At each daily firing the day's rain,
times the ``rain_gain`` param, is added and the evaporation of the being's
state is taken away (``evap_dormant`` while the bus says the being sleeps,
``evap_awake`` otherwise), the level held to ``0..=m_max``: two units, and
two more for the weather's draw in the garden. A ``water`` adds the law's
dose (one unit), and what overflows drains. The soil is never quiescent: it
dries while the being sleeps.
"""

from ... import fx
from . import weather

checkpoint_before_apply = True

CHANNEL = "moisture"


def init(constants, k):
    """The soil at a being's first minute."""
    return {"m": constants["m0"]}


def daily(org, cx, day):
    """One daily firing: rain in, evaporation out."""
    c = cx.constants["soil"]
    work = cx.work
    gained = fx.mul(weather.rain(cx, day), cx.params["rain_gain"], work)
    lost = cx.params["evap_dormant"] if cx.bus["dormant"] else cx.params["evap_awake"]
    org["m"] = fx.sat(org["m"] + gained - lost, 0, c["m_max"], work)


def on_fact(org, fact, cx):
    """A ``water`` adds the dose; nothing else reaches the soil."""
    if fact["kind"] == "act" and fact["body"]["act"] == "water":
        c = cx.constants["soil"]
        org["m"] = fx.sat(org["m"] + c["dose"], 0, c["m_max"], cx.work)


def publish(org, cx):
    """The moisture channel: the level itself."""
    return {CHANNEL: org["m"]}
