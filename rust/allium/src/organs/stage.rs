//! The stage, the twin of `opti_oignon/allium/ref/organs/stage.py`: dormancy
//! from drought and from winter, and the wake.
//!
//! At each daily firing, awake: a day whose moisture (read from the bus) is
//! under `theta_dry` adds one to the run of dry days, any other day ends the
//! run; `d_enter` dry days in a row put the being to sleep for at least
//! `rest_dry` days, and the first day of winter (the season turning to
//! winter, not every winter day) puts it to sleep for at least `rest_winter`
//! days. Dormant: one day of rest is done; once the rest is over, a day out
//! of winter whose moisture reaches `theta_wet` wakes the being. One unit a
//! firing, awake or dormant. A `water` wakes a being whose rest is over when
//! it slept from drought, or outside winter; a `warm` wakes one whose winter
//! rest is over. Neither costs a unit. The entry minute is kept in `since`
//! until the next entry. The stage is never quiescent, and has no `jump`.

use crate::civil;
use crate::fx::{self, Work, ONE};
use crate::lawdata::{StageConsts, World};

pub const CHANNEL: &str = "dormant";

/// Why the being sleeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    Dry,
    None,
    Winter,
}

impl Cause {
    pub fn name(self) -> &'static str {
        match self {
            Cause::Dry => "dry",
            Cause::None => "none",
            Cause::Winter => "winter",
        }
    }

    pub fn from_name(name: &str) -> Option<Cause> {
        match name {
            "dry" => Some(Cause::Dry),
            "none" => Some(Cause::None),
            "winter" => Some(Cause::Winter),
            _ => None,
        }
    }
}

/// The stage's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage {
    pub cause: Cause,
    pub dormant: bool,
    pub dry: i64,
    pub rest: i64,
    pub season: i64,
    pub since: i64,
}

/// The stage at a being's first minute: awake; its season is set at the end of minute 0's offsets.
pub fn init() -> Stage {
    Stage { cause: Cause::None, dormant: false, dry: 0, rest: 0, season: 0, since: 0 }
}

fn sleep(org: &mut Stage, cause: Cause, rest: i64, minute: i64) {
    org.dormant = true;
    org.cause = cause;
    org.rest = rest;
    org.since = minute;
    org.dry = 0;
}

fn wake(org: &mut Stage) {
    org.dormant = false;
    org.cause = Cause::None;
    org.dry = 0;
}

/// One daily firing at minute `minute` for local day `day`; 1 unit.
pub fn daily(
    org: &mut Stage,
    c: &StageConsts,
    sky: (&World, bool),
    moisture: i64,
    day: i64,
    minute: i64,
    work: &mut Work,
) -> Option<()> {
    let (world, south) = sky;
    let season = civil::season(day, world, south)?;
    if !org.dormant {
        let run = if moisture < c.theta_dry { i128::from(org.dry).checked_add(1)? } else { 0 };
        let dry = i64::try_from(fx::sat(run, 0, i128::from(c.rest_max), work)).ok()?;
        if dry >= c.d_enter {
            sleep(org, Cause::Dry, c.rest_dry, minute);
        } else if season == 0 && org.season != 0 {
            sleep(org, Cause::Winter, c.rest_winter, minute);
        } else {
            org.dry = dry;
        }
        org.season = season;
        return Some(());
    }
    let rest = i128::from(org.rest).checked_sub(1)?;
    org.rest = i64::try_from(fx::sat(rest, 0, i128::from(c.rest_max), work)).ok()?;
    org.season = season;
    if org.rest == 0 && season != 0 && moisture >= c.theta_wet {
        wake(org);
    }
    Some(())
}

/// A `water` or a `warm` may end a dormancy whose rest is over; nothing costs a unit.
pub fn on_act(org: &mut Stage, act: &str, sky: (&World, bool), day: i64) -> Option<()> {
    if !org.dormant || org.rest != 0 {
        return Some(());
    }
    let (world, south) = sky;
    if act == "water" {
        if org.cause == Cause::Dry || civil::season(day, world, south)? != 0 {
            wake(org);
        }
    } else if act == "warm" && org.cause == Cause::Winter {
        wake(org);
    }
    Some(())
}

/// The dormancy channel: `ONE` while the being sleeps, else 0.
pub fn publish(org: &Stage) -> i64 {
    if org.dormant {
        i64::try_from(ONE).unwrap_or(0)
    } else {
        0
    }
}
