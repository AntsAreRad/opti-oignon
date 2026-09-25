"""The weather: the rain a local day brings, a pure function of the being and the day.

The reference, mirrored in the twin. Weather is the being's identity, frozen
in its genesis: on a ``windowsill`` no rain falls; in the ``garden`` one
64-bit word decides each local day's rain. The word is drawn from the key
of the being's seed in the domain ``world.weather``, addressed by the
being's sixteen bytes (two big-endian 64-bit halves) and the local day, never
by a running counter: a life cut into any slices, or advanced along any
path, draws the same rain on the same day, and a clone from the same seed,
which has another being id, has weather of its own.

Of the word, the top sixteen bits decide whether the day is wet, against
the season's chance of rain; the next sixteen scale the season's largest
rain. A draw costs two units, the key and the word, and is counted only
where it is made.
"""

from ... import rng
from .. import civil

checkpoint_before_apply = True

DOMAIN = "world.weather"
DOMAINS = (DOMAIN,)
DRAW = 2


def rain(cx, day):
    """The rain of local day ``day`` (Q16), and the draw it cost; 0 on a windowsill, where nothing is drawn."""
    if cx.weather != "garden":
        return 0
    cx.work.units += DRAW
    cx.draws += 1
    word = rng.Stream.from_key(rng.key(cx.seed, DOMAIN, (cx.being_hi, cx.being_lo, day))).next_u64()
    season = civil.season(day, cx.world, cx.hemisphere)
    table = cx.world["rain"]
    if (word >> 48) >= table["p_wet"][season]:
        return 0
    return (((word >> 32) & 0xFFFF) * table["max"][season]) >> 16
