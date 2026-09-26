#!/usr/bin/env python3
"""The garden's gallery: the forms ``oo garden`` prints, from fixed values.

``python3 scripts/allium_garden_gallery.py [--tier text|ascii]``

Prints a header, then sample forms rendered by the garden's own
description (``allium.describe``) and catalogue (``allium.wording``) from
fixed values: seeds in a pot and in a glass jar, awake, breathing and gone
dormant by winter and by drought, under Daily, Bulbe and a mode that cannot
be read, a view still catching up and one frozen on an engine fault; then
the form of every status that is not a being shown. Each form is wrapped as
the terminal wraps it, and each caption passes the garden's own nets before
it is printed.

No onion is read: no store is opened, no clock and no mode are read, and no
file is read but the package's own. It is the first look at the garden in a
real terminal -- the backslashes, backquotes and quotes of the drawing, the
blank sky row, the 78-column wrap and the unsplit path line.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opti_oignon.allium import describe, ethics, service, store  # noqa: E402

HEADER = "Sample forms from fixed values; no onion was read."

# A seed on its day 0 in an autumn garden, full sun and wet soil, awake; each sample changes some of it.
BASE = {"name": None, "day": 0, "season": 3, "place": "garden", "light": "up", "soil": "wet", "life": "awake",
        "minute": 533, "daylength": 641, "sun": 65536, "sun_max": 65536, "jar": False, "layer": "open",
        "labels": ("prototype",), "local": "2025-10-09 08:53", "offset": "+00:00"}
WINTER = {"name": "Pip", "day": 52, "season": 0, "light": "down", "life": "dormant_winter", "minute": 413,
          "daylength": 492, "sun": 0, "local": "2025-11-30 06:53"}
SAMPLES = (
    ("A seed in a pot in the garden, autumn, day 0, in full sun: awake.", {}),
    ("Named, in winter, at night: gone dormant when winter came.", WINTER),
    ("A windowsill with no water given, Bulbe's rules: gone dormant in a dry spell.",
     {"day": 18, "place": "windowsill", "light": "down", "soil": "dry", "life": "dormant_dry", "minute": 173,
      "daylength": 560, "sun": 0, "layer": "bulbe", "labels": ("prototype", "bulbe"), "local": "2025-10-27 02:53"}),
    ("In a glass jar, spring, at sunrise, damp soil: breathing.",
     {"name": "Pip", "day": 7, "season": 1, "light": "rise", "soil": "damp", "life": "breathing", "minute": 390,
      "daylength": 700, "sun": 30000, "jar": True, "labels": ("prototype", "glass_jar"),
      "local": "2025-10-16 06:30"}),
    ("In a glass jar, shown as of its minute 0 while the rest is computed.",
     {"jar": True, "labels": ("prototype", "glass_jar", "catching_up")}),
    ("The winter sample again, with a security mode that cannot be read.",
     dict(WINTER, layer="bulbe", labels=("prototype", "mode_unknown"))),
    ("Frozen on the state kept before an engine fault.",
     {"name": "Pip", "day": 12, "light": "set", "soil": "damp", "life": "breathing", "minute": 1000,
      "daylength": 600, "sun": 20000, "labels": ("prototype", "frozen"), "local": "2025-10-21 16:40",
      "offset": "+02:00"}),
)
BEING = service.BeingInfo(name=None, soil="encrypted", law="v0_1", v=0, provisional=True, digest="0" * 12,
                          tag="0a1b2c3d", events=3, weather="garden")


def _look(status, **fields):
    values = {"status": status, "labels": (), "mode": "daily", "habitat": None, "view": None, "felt": None,
              "being": None, "reason": None, "offer": None, "seq": None, "exit": describe.STATUS_FORMS[status][1],
              "glass_allowed": False}
    values.update(fields)
    return service.Look(**values)


STATUSES = (
    ("Switched off.", _look("disabled")),
    ("Its settings file cannot be read.", _look("disabled", reason="unreadable")),
    ("The emergency stop is on in the server.", _look("stopped")),
    ("No key for an encrypted store, under Bulbe's rules, the glass jar allowed.",
     _look("awaiting_soil", mode="bulbe", glass_allowed=True)),
    ("Nothing sown yet.", _look("ready")),
    ("A store that cannot be opened: its key cannot be read.", _look("unavailable", reason="key")),
    ("A being that cannot be computed now: the clock.",
     _look("unavailable", reason="clock", being=BEING, labels=("prototype",))),
    ("A glass jar under Bulbe's rules.", _look("sealed_bulbe", labels=("prototype", "glass_jar", "bulbe"))),
    ("A store missing after an interrupted sowing.",
     _look("missing", labels=("prototype",), offer=store.Finish("0a1b2c3d"))),
    ("A store missing, with nothing to finish.", _look("missing", labels=("prototype",))),
    ("A prototype whose world was retired.", _look("retired_prototype", labels=("retired",))),
    ("A record broken at an event.",
     _look("unreadable", labels=("prototype",), seq=3, offer=store.Resume(2, 0, 4))),
    ("A record with no verified event left.", _look("unreadable", labels=("prototype",))),
)


def _caption(caption):
    """``-- caption``, once the caption passes the garden's own nets: it is printed beside the garden's words."""
    if ethics.check(caption):
        raise ValueError(f"a caption fails the garden's nets: {caption}")
    return "-- " + caption


def gallery(tier="ascii"):
    """The rows the gallery prints in ``tier``: the header, then each sample under its caption."""
    rows = [HEADER]
    for caption, fields in SAMPLES:
        felt = describe.Felt(**dict(BASE, **fields))
        rows += ["", _caption(caption)]
        rows += describe.wrap(describe.show_form(felt, tier))
    for caption, look in STATUSES:
        rows += ["", _caption(caption)]
        rows += describe.wrap(describe.show(look, tier))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="The forms oo garden prints, from fixed values.")
    parser.add_argument("--tier", choices=("text", "ascii"), default="ascii")
    args = parser.parse_args(argv)
    sys.stdout.write("\n".join(gallery(args.tier)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
