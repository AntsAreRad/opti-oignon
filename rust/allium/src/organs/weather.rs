//! The weather, the twin of `opti_oignon/allium/ref/organs/weather.py`: the
//! rain a local day brings, a pure function of the being and the day.
//!
//! Weather is the being's identity, frozen in its genesis: on a windowsill
//! no rain falls; in the garden one 64-bit word decides each local day's
//! rain. The word is drawn from the key of the being's seed in the domain
//! `world.weather`, addressed by the being's sixteen bytes (two big-endian
//! 64-bit halves) and the local day, never by a running counter, so a life
//! cut into any slices draws the same rain on the same day. Of the word, the
//! top sixteen bits decide whether the day is wet, against the season's
//! chance of rain; the next sixteen scale the season's largest rain. A draw
//! costs two units, the key and the word.

use crate::civil;
use crate::fx::Work;
use crate::lawdata::World;
use crate::rng;

pub const DOMAIN: &str = "world.weather";
pub const DRAW: u64 = 2;

/// What the rain of a being depends on besides the day.
#[derive(Clone, Debug)]
pub struct Weather {
    pub garden: bool,
    pub south: bool,
    pub seed: [u8; 32],
    pub being_hi: u64,
    pub being_lo: u64,
}

/// The rain of local day `day` (Q16); 0 on a windowsill, where nothing is drawn.
pub fn rain(weather: &Weather, world: &World, day: i64, work: &mut Work, draws: &mut u64) -> Option<i128> {
    if !weather.garden {
        return Some(0);
    }
    work.units = work.units.checked_add(DRAW)?;
    *draws = draws.checked_add(1)?;
    let key = rng::key(&weather.seed, DOMAIN, &[weather.being_hi, weather.being_lo, u64::try_from(day).ok()?]);
    let word = rng::Stream::from_key(&key).next_u64();
    let season = usize::try_from(civil::season(day, world, weather.south)?).ok()?;
    let chance = u64::try_from(*world.p_wet.get(season)?).ok()?;
    if word.checked_shr(48)? >= chance {
        return Some(0);
    }
    let most = u64::try_from(*world.rain_max.get(season)?).ok()?;
    let scale = word.checked_shr(32)? & 0xFFFF;
    Some(i128::from(scale.checked_mul(most)?.checked_shr(16)?))
}
