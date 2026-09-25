"""The stage: dormancy from drought and from winter, and the wake.

The reference, mirrored in the twin. At each daily firing, awake: a day
whose moisture (read from the bus) is under ``theta_dry`` adds one to the
run of dry days, any other day ends the run; ``d_enter`` dry days in a row
put the being to sleep for at least ``rest_dry`` days, and the first day of
winter (the season turning to winter, not every winter day) puts it to sleep
for at least ``rest_winter`` days. Dormant: one day of rest is done; once
the rest is over, a day out of winter whose moisture reaches ``theta_wet``
wakes the being. One unit a firing, awake or dormant.

A ``water`` wakes a being whose rest is over when it slept from drought, or
outside winter; a ``warm`` wakes one whose winter rest is over. Neither
costs a unit, and no other gesture reaches the stage. The entry minute is
kept in ``since`` until the next entry. The stage is never quiescent, and
has no ``jump``: it is stepped at every daily firing.
"""

from ... import fx
from .. import civil

checkpoint_before_apply = True

CHANNEL = "dormant"


def init(constants, k):
    """The stage at a being's first minute: awake; its season is set at the end of minute 0's offsets."""
    return {"cause": "none", "dormant": False, "dry": 0, "rest": 0, "season": 0, "since": 0}


def _sleep(org, cause, rest, minute):
    org["dormant"] = True
    org["cause"] = cause
    org["rest"] = rest
    org["since"] = minute
    org["dry"] = 0


def _wake(org):
    org["dormant"] = False
    org["cause"] = "none"
    org["dry"] = 0


def daily(org, cx, day):
    """One daily firing: the run of dry days and the entries awake, the rest and the wake dormant; 1 unit."""
    c = cx.constants["stage"]
    season = civil.season(day, cx.world, cx.hemisphere)
    if not org["dormant"]:
        dry = fx.sat(org["dry"] + 1 if cx.bus["moisture"] < c["theta_dry"] else 0, 0, c["rest_max"], cx.work)
        if dry >= c["d_enter"]:
            _sleep(org, "dry", c["rest_dry"], cx.minute)
        elif season == 0 and org["season"] != 0:
            _sleep(org, "winter", c["rest_winter"], cx.minute)
        else:
            org["dry"] = dry
        org["season"] = season
        return
    org["rest"] = fx.sat(org["rest"] - 1, 0, c["rest_max"], cx.work)
    org["season"] = season
    if org["rest"] == 0 and season != 0 and cx.bus["moisture"] >= c["theta_wet"]:
        _wake(org)


def on_fact(org, fact, cx):
    """A ``water`` or a ``warm`` may end a dormancy whose rest is over; nothing costs a unit."""
    if fact["kind"] != "act" or not org["dormant"] or org["rest"] != 0:
        return
    act = fact["body"]["act"]
    if act == "water":
        if org["cause"] == "dry" or civil.season(cx.day, cx.world, cx.hemisphere) != 0:
            _wake(org)
    elif act == "warm" and org["cause"] == "winter":
        _wake(org)


def publish(org, cx):
    """The dormancy channel: ``ONE`` while the being sleeps, else 0."""
    return {CHANNEL: fx.ONE if org["dormant"] else 0}
