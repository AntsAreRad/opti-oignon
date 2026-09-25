"""Civil time for the componion's engine: local time, the calendar, the season and the light.

The reference. Its Rust twin is ``rust/allium/src/civil.rs``.

A being lives in minutes since its birth: minute ``t`` is ``(wall - birth.wall)
div 60``, and its instant lies inside UTC minute ``b + t``, where ``b =
birth.wall div 60`` is the birth minute. Every offset is a whole number of
quarter hours within fourteen hours either side, and the offset in force at a
minute is decided by the caller (the reducer folds it from the journal; the
``grid`` operation takes it as a list). Then:

* the local minute is ``b + t + z``; the local day is its day number since
  1970-01-01 in local time, and the minute of the day is the remainder;
* a fast boundary is a minute whose UTC minute is a multiple of 15, so a
  local midnight, under any quarter-hour offset, is always one;
* the next local midnight strictly after ``t`` is ``t + 1 + ((-(b + t + 1 +
  z)) mod 1440)``;
* the calendar is Howard Hinnant's, in integers: ``days_from_civil`` and
  ``civil_from_days`` map a proleptic Gregorian date to its day number and
  back, for every integer day.

A world (the law data's ``world`` section) gives each local day a position
in its year: the civil year, or a fixed year of ``days`` days; the southern
hemisphere is shifted by ``south_shift``. The season is the quarter of the
year position shifted by ``season_shift`` (0 winter, 1 spring, 2 summer, 3
autumn). The daylength is a sine of the year position around ``equinox``,
read from the frozen table, and the sun at a local minute ramps up after
sunrise and down before sunset over ``ramp`` minutes.

Every modulo is Euclidean and every division by a positive divisor floors,
as Python's ``%`` and ``//`` do and as the twin's ``rem_euclid`` and
``div_euclid`` do. Calendar and season arithmetic cost no unit; the light of
one minute costs what its primitives count: one ``sin_b`` for the daylength
and three ``sat`` for the sun.
"""

from .. import fx, wire

checkpoint_before_apply = True

DAY = 1440
FAST = 15
TZ_STEP = 15
TZ_LIMIT = 840
# The earliest birth wall: it keeps every local day number at or above zero.
WALL_MIN = 86400
# The largest minute a fact, a ``to`` or an ``at`` may carry, so that a next
# local midnight and an ``effective_from`` stay within the wire's integers.
T_MAX = wire.MAX_INT - 2 * DAY
SEASONS = 4
TURN = 1024


def offset_ok(z):
    """Whether ``z`` is an offset: a multiple of 15 minutes within fourteen hours either side."""
    return -TZ_LIMIT <= z <= TZ_LIMIT and z % TZ_STEP == 0


def days_from_civil(y, m, d):
    """The day number since 1970-01-01 of the proleptic Gregorian date ``y-m-d``."""
    y -= 1 if m <= 2 else 0
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m - 3 if m > 2 else m + 9) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def civil_from_days(z):
    """The proleptic Gregorian date ``(y, m, d)`` of day number ``z`` since 1970-01-01."""
    z += 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return (y + (1 if m <= 2 else 0), m, d)


def local(b, t, z):
    """The local day and the minute of that day, at minute ``t`` of a life born in UTC minute ``b``, under offset ``z``."""
    minute = b + t + z
    return minute // DAY, minute % DAY


def next_midnight(b, t, z):
    """The first local midnight strictly after minute ``t``, under offset ``z``."""
    return t + 1 + (-(b + t + 1 + z)) % DAY


def year_position(day, world, hemisphere):
    """The position ``p`` of local day ``day`` in its year, and the year's length ``Y``."""
    year = world["year"]
    if year["kind"] == "civil":
        y = civil_from_days(day)[0]
        start = days_from_civil(y, 1, 1)
        p = day - start
        length = days_from_civil(y + 1, 1, 1) - start
    else:
        length = year["days"]
        p = day % length
    if hemisphere == "south":
        p = (p + world["south_shift"]) % length
    return p, length


def season(day, world, hemisphere):
    """The season of local day ``day``: 0 winter, 1 spring, 2 summer, 3 autumn."""
    p, length = year_position(day, world, hemisphere)
    return (((p + world["season_shift"]) % length) * SEASONS) // length


def daylength(day, world, hemisphere, band, sine, work):
    """The minutes of light of local day ``day``, for a being of ``band`` (1 unit: the sine)."""
    p, length = year_position(day, world, hemisphere)
    light = world["daylength"]
    bam = (((p - light["equinox"]) % length) * TURN) // length
    return light["mean"] + ((light["amp"][band] * fx.sin_b(bam, sine, work)) >> 15)


def sun(minute, dl, world, sun_max, work):
    """The sun at local minute ``minute`` of a day with ``dl`` minutes of light, in ``0..=sun_max`` (3 units)."""
    ramp = world["daylength"]["ramp"]
    rise = 720 - dl // 2
    after_rise = fx.sat(minute - rise, 0, ramp, work)
    before_set = fx.sat(rise + dl - minute, 0, ramp, work)
    light = fx.sat(after_rise, 0, before_set, work)
    return (light * sun_max) // ramp
