#![deny(clippy::indexing_slicing, clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::todo, clippy::unimplemented)]
//! Civil time, the twin of `opti_oignon/allium/ref/civil.py`: local time,
//! the next local midnight and the calendar, in integers.
//!
//! Minute `t` of a life born in UTC minute `b` lies inside UTC minute
//! `b + t`. Under offset `z` its local minute is `b + t + z`, its local day
//! the day number of that minute since 1970-01-01, and its minute of the day
//! the remainder. The next local midnight strictly after `t` is
//! `t + 1 + ((-(b + t + 1 + z)) mod 1440)`. The calendar is Howard
//! Hinnant's: a proleptic Gregorian date and its day number, both ways.
//!
//! A world (the law data's `world` section) gives each local day a position
//! in its year: the civil year, or a fixed year of `days` days; the southern
//! hemisphere is shifted by `south_shift`. The season is the quarter of the
//! year position shifted by `season_shift` (0 winter, 1 spring, 2 summer, 3
//! autumn). The daylength is a sine of the year position around `equinox`,
//! read from the frozen table, and the sun at a local minute ramps up after
//! sunrise and down before sunset over `ramp` minutes. Calendar and season
//! arithmetic cost no unit; the light of one minute costs what its
//! primitives count: one `sin_b` for the daylength and three `sat` for the
//! sun.
//!
//! Every modulo is `rem_euclid` and every division by a positive divisor is
//! `div_euclid`, as Python's `%` and `//` are. Every operation is checked and
//! a result that does not fit gives `None`, which the caller maps to
//! `engine_panic`. For what a request may carry (a wall up to 2^53 - 1, a
//! minute up to `T_MAX`, an offset within fourteen hours) no intermediate
//! comes near the i64 bounds, so the reference, which has no such refusal,
//! is never contradicted.

use crate::fx::{self, Work};
use crate::lawdata::World;
use crate::ocj::MAX_INT;

pub const DAY: i64 = 1440;
pub const FAST: i64 = 15;
pub const TZ_STEP: i64 = 15;
pub const TZ_MIN: i64 = -840;
pub const TZ_MAX: i64 = 840;
/// The earliest birth wall: it keeps every local day number at or above zero.
pub const WALL_MIN: i64 = 86_400;
/// The largest minute a request may carry, so that a next local midnight stays within the wire's integers.
pub const T_MAX: i64 = MAX_INT - 2880;
pub const SEASONS: i64 = 4;
/// A binary-angle turn of the sine table.
const TURN: i64 = 1024;

fn floor_div(a: i64, b: i64) -> Option<i64> {
    a.checked_div_euclid(b)
}

fn modulo(a: i64, b: i64) -> Option<i64> {
    a.checked_rem_euclid(b)
}

/// Whether `z` is an offset: a multiple of 15 minutes within fourteen hours either side.
pub fn offset_ok(z: i64) -> bool {
    (TZ_MIN..=TZ_MAX).contains(&z) && modulo(z, TZ_STEP) == Some(0)
}

/// The day number since 1970-01-01 of the proleptic Gregorian date `y-m-d`.
pub fn days_from_civil(y: i64, m: i64, d: i64) -> Option<i64> {
    let y = if m <= 2 { y.checked_sub(1)? } else { y };
    let era = floor_div(y, 400)?;
    let yoe = y.checked_sub(era.checked_mul(400)?)?;
    let shifted = if m > 2 { m.checked_sub(3)? } else { m.checked_add(9)? };
    let doy = floor_div(shifted.checked_mul(153)?.checked_add(2)?, 5)?.checked_add(d)?.checked_sub(1)?;
    let doe = yoe
        .checked_mul(365)?
        .checked_add(floor_div(yoe, 4)?)?
        .checked_sub(floor_div(yoe, 100)?)?
        .checked_add(doy)?;
    era.checked_mul(146_097)?.checked_add(doe)?.checked_sub(719_468)
}

/// The proleptic Gregorian date `(y, m, d)` of day number `z` since 1970-01-01.
pub fn civil_from_days(z: i64) -> Option<(i64, i64, i64)> {
    let z = z.checked_add(719_468)?;
    let era = floor_div(z, 146_097)?;
    let doe = z.checked_sub(era.checked_mul(146_097)?)?;
    let yoe = floor_div(
        doe.checked_sub(floor_div(doe, 1460)?)?
            .checked_add(floor_div(doe, 36_524)?)?
            .checked_sub(floor_div(doe, 146_096)?)?,
        365,
    )?;
    let y = yoe.checked_add(era.checked_mul(400)?)?;
    let doy = doe.checked_sub(
        yoe.checked_mul(365)?
            .checked_add(floor_div(yoe, 4)?)?
            .checked_sub(floor_div(yoe, 100)?)?,
    )?;
    let mp = floor_div(doy.checked_mul(5)?.checked_add(2)?, 153)?;
    let d = doy.checked_sub(floor_div(mp.checked_mul(153)?.checked_add(2)?, 5)?)?.checked_add(1)?;
    let m = if mp < 10 { mp.checked_add(3)? } else { mp.checked_sub(9)? };
    let y = if m <= 2 { y.checked_add(1)? } else { y };
    Some((y, m, d))
}

/// The local day and the minute of that day, at minute `t` of a life born in UTC minute `b`, under offset `z`.
pub fn local(b: i64, t: i64, z: i64) -> Option<(i64, i64)> {
    let minute = b.checked_add(t)?.checked_add(z)?;
    Some((floor_div(minute, DAY)?, modulo(minute, DAY)?))
}

/// The first local midnight strictly after minute `t`, under offset `z`.
pub fn next_midnight(b: i64, t: i64, z: i64) -> Option<i64> {
    let after = b.checked_add(t)?.checked_add(1)?.checked_add(z)?;
    t.checked_add(1)?.checked_add(modulo(after.checked_neg()?, DAY)?)
}

/// The position `p` of local day `day` in its year, and the year's length `Y`.
pub fn year_position(day: i64, world: &World, south: bool) -> Option<(i64, i64)> {
    let (p, length) = if world.civil {
        let (y, _, _) = civil_from_days(day)?;
        let start = days_from_civil(y, 1, 1)?;
        let length = days_from_civil(y.checked_add(1)?, 1, 1)?.checked_sub(start)?;
        (day.checked_sub(start)?, length)
    } else {
        (modulo(day, world.days)?, world.days)
    };
    if south {
        return Some((modulo(p.checked_add(world.south_shift)?, length)?, length));
    }
    Some((p, length))
}

/// The season of local day `day`: 0 winter, 1 spring, 2 summer, 3 autumn.
pub fn season(day: i64, world: &World, south: bool) -> Option<i64> {
    let (p, length) = year_position(day, world, south)?;
    floor_div(modulo(p.checked_add(world.season_shift)?, length)?.checked_mul(SEASONS)?, length)
}

/// The minutes of light of local day `day`, for a being whose band has amplitude `amp` (1 unit: the sine).
pub fn daylength(day: i64, world: &World, south: bool, amp: i64, sine: &[i128], work: &mut Work) -> Option<i64> {
    let (p, length) = year_position(day, world, south)?;
    let bam = floor_div(modulo(p.checked_sub(world.equinox)?, length)?.checked_mul(TURN)?, length)?;
    let wave = fx::sin_b(i128::from(bam), sine, work);
    let swing = i128::from(amp).checked_mul(wave)?.checked_shr(15)?;
    i64::try_from(i128::from(world.mean).checked_add(swing)?).ok()
}

/// The sun at local minute `minute` of a day with `dl` minutes of light, in `0..=sun_max` (3 units).
pub fn sun(minute: i64, dl: i64, world: &World, sun_max: i64, work: &mut Work) -> Option<i64> {
    let ramp = i128::from(world.ramp);
    let rise = 720_i64.checked_sub(floor_div(dl, 2)?)?;
    let after_rise = fx::sat(i128::from(minute.checked_sub(rise)?), 0, ramp, work);
    let before_set = fx::sat(i128::from(rise.checked_add(dl)?.checked_sub(minute)?), 0, ramp, work);
    let light = fx::sat(after_rise, 0, before_set, work);
    i64::try_from(light.checked_mul(i128::from(sun_max))?.checked_div_euclid(ramp)?).ok()
}
