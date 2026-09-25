//! The soil, the twin of `opti_oignon/allium/ref/organs/soil.py`: one
//! moisture level, filled by rain and care, emptied by evaporation.
//!
//! At each daily firing the day's rain, times the `rain_gain` param, is
//! added and the evaporation of the being's state is taken away
//! (`evap_dormant` while the bus says the being sleeps, `evap_awake`
//! otherwise), the level held to `0..=m_max`: two units, and two more for the
//! weather's draw in the garden. A `water` adds the law's dose (one unit),
//! and what overflows drains. The soil is never quiescent.

use alloc::collections::BTreeMap;
use alloc::string::String;

use crate::fx::{self, Work};
use crate::lawdata::{SoilConsts, World};
use crate::organs::weather::{self, Weather};

pub const CHANNEL: &str = "moisture";

/// The soil's state: its moisture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Soil {
    pub m: i64,
}

/// The soil at a being's first minute.
pub fn init(c: &SoilConsts) -> Soil {
    Soil { m: c.m0 }
}

fn param(params: &BTreeMap<String, i64>, name: &str) -> Option<i128> {
    params.get(name).map(|value| i128::from(*value))
}

/// One daily firing on local day `day`: rain in, evaporation out.
#[allow(clippy::too_many_arguments)]
pub fn daily(
    org: &mut Soil,
    c: &SoilConsts,
    params: &BTreeMap<String, i64>,
    bus_dormant: i64,
    sky: (&Weather, &World),
    day: i64,
    work: &mut Work,
    draws: &mut u64,
) -> Option<()> {
    let (weather_of, world) = sky;
    let rain = weather::rain(weather_of, world, day, work, draws)?;
    let gained = fx::mul(rain, param(params, "rain_gain")?, work);
    let lost = if bus_dormant != 0 { param(params, "evap_dormant")? } else { param(params, "evap_awake")? };
    let level = i128::from(org.m).checked_add(gained)?.checked_sub(lost)?;
    org.m = i64::try_from(fx::sat(level, 0, i128::from(c.m_max), work)).ok()?;
    Some(())
}

/// A `water` adds the dose; nothing else reaches the soil.
pub fn on_act(org: &mut Soil, c: &SoilConsts, act: &str, work: &mut Work) -> Option<()> {
    if act == "water" {
        let level = i128::from(org.m).checked_add(i128::from(c.dose))?;
        org.m = i64::try_from(fx::sat(level, 0, i128::from(c.m_max), work)).ok()?;
    }
    Some(())
}

/// The moisture channel: the level itself.
pub fn publish(org: &Soil) -> i64 {
    org.m
}
