//! The clock, the twin of `opti_oignon/allium/ref/organs/clock.py`: three
//! genes that repress one another in a ring, entrained by light.
//!
//! `clock_m` is repressed by `clock_e`, `clock_d` by `clock_m` and `clock_e`
//! by `clock_d`. At each awake fast step, for each gene: its repression is
//! `hill_down` of the gene before it in the ring, its production `alpha`
//! times that repression, and its decay `beta` times its own level; light,
//! the constant gain times the sun, adds to `clock_m`. Each level is then
//! held to `0..=CMAX`, and every level that would have gone above `CMAX` is
//! counted as a clip. Thirteen units a step. The clock is quiescent in
//! dormancy: its step does nothing and costs nothing while the being
//! sleeps, and its one `jump` at the wake is the identity.

use crate::fx::{self, Work, CMAX, ONE};
use crate::lawdata::ClockK;

pub const CHANNEL: &str = "circadian";

/// The clock's state: its genome constants and its three levels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clock {
    pub k: ClockK,
    pub p: [i64; 3],
}

/// The clock at a being's first minute: its levels from the law, its constants from the genome.
pub fn init(levels: [i64; 3], k: ClockK) -> Clock {
    Clock { k, p: levels }
}

/// One gene's production and decay: its repressor's level, its own level, and its constants.
fn gene(repressor: i64, own: i64, alpha: i64, beta: i64, k: i64, n: i64, work: &mut Work) -> (i128, i128) {
    let repression = fx::hill_down(i128::from(repressor), i128::from(k), i128::from(n), work);
    let production = fx::mul(i128::from(alpha), repression, work);
    let decay = fx::mul(i128::from(beta), i128::from(own), work);
    (production, decay)
}

/// One fast step: the three levels move; 13 units awake, nothing while dormant.
pub fn fast(org: &mut Clock, sun: i64, dormant: bool, work: &mut Work, clips: &mut u64) -> Option<()> {
    if dormant {
        return Some(());
    }
    let [m, d, e] = org.p;
    let k = org.k;
    let [alpha_m, alpha_d, alpha_e] = k.alpha;
    let [beta_m, beta_d, beta_e] = k.beta;
    let [k_m, k_d, k_e] = k.k;
    let [n_m, n_d, n_e] = k.n;
    // The ring: m is repressed by e, d by m, e by d.
    let (prod_m, deg_m) = gene(e, m, alpha_m, beta_m, k_m, n_m, work);
    let (prod_d, deg_d) = gene(m, d, alpha_d, beta_d, k_d, n_d, work);
    let (prod_e, deg_e) = gene(d, e, alpha_e, beta_e, k_e, n_e, work);
    let light = fx::mul(i128::from(k.light), i128::from(sun), work);
    let raw = [
        i128::from(m).checked_add(prod_m)?.checked_add(light)?.checked_sub(deg_m)?,
        i128::from(d).checked_add(prod_d)?.checked_sub(deg_d)?,
        i128::from(e).checked_add(prod_e)?.checked_sub(deg_e)?,
    ];
    let mut levels = [0_i64; 3];
    for (value, level) in raw.iter().zip(levels.iter_mut()) {
        if *value > CMAX {
            *clips = clips.checked_add(1)?;
        }
        *level = i64::try_from(fx::sat(*value, 0, CMAX, work)).ok()?;
    }
    org.p = levels;
    Some(())
}

/// The dormancy from minute `since` to the wake at `wake`, applied at once: nothing changes.
pub fn jump(_org: &mut Clock, _since: i64, _wake: i64) {}

/// The circadian channel: `clock_m` shifted down by three, at most `ONE`.
pub fn publish(org: &Clock) -> i64 {
    let [m, _, _] = org.p;
    i64::try_from(ONE.min(i128::from(m).checked_shr(3).unwrap_or(0))).unwrap_or(0)
}
