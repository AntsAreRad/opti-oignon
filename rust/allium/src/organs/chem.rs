//! The reserve, the twin of `opti_oignon/allium/ref/organs/chem.py`: sugar
//! made by light, burnt by respiration, stored as fructan and drawn back.
//!
//! At each awake fast step, photosynthesis `P` is the rate `ps` times the
//! saturation of the sun and of the soil moisture (read from the bus), held
//! to the room left for sugar; respiration `R` is the rate `r` times the
//! saturation of the sugar, held to the sugar there is after `P`; above
//! `theta_s`, sugar is stored as fructan at the rate `sy`, held to the sugar
//! left and to the room left for fructan; below `theta_h`, fructan is drawn
//! back as sugar at the rate `hy`, held to the fructan above its `core`.
//! Fifteen units a step. The ledger counters `made` and `burnt` add `P` and
//! `R`, saturating at the wire's largest integer, and the step's `P + R` is
//! kept for the bus. The reserve is quiescent in dormancy.

use crate::fx::{self, Work, I32_MAX, ONE};
use crate::lawdata::{ChemConsts, ChemK};
use crate::ocj::MAX_INT;

pub const CHANNEL: &str = "metab";

/// The reserve's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chem {
    pub burnt: i64,
    pub fructan: i64,
    pub k: ChemK,
    pub made: i64,
    pub metab: i64,
    pub sugar: i64,
}

/// The reserve at a being's first minute.
pub fn init(c: &ChemConsts, k: ChemK) -> Chem {
    Chem { burnt: 0, fructan: c.fructan0, k, made: 0, metab: 0, sugar: c.sugar0 }
}

/// One fast step of the reserve; 15 units awake, nothing while dormant.
pub fn fast(org: &mut Chem, c: &ChemConsts, sun: i64, moisture: i64, dormant: bool, work: &mut Work) -> Option<()> {
    if dormant {
        return Some(());
    }
    let k = org.k;
    let sugar = i128::from(org.sugar);
    let fructan = i128::from(org.fructan);
    let light = fx::mm(i128::from(sun), i128::from(k.km_ps), work);
    let water = fx::mm(i128::from(moisture), i128::from(c.k_w), work);
    let inner = fx::mul(light, water, work);
    let made = fx::mul(i128::from(k.ps), inner, work);
    let made = fx::sat(made, 0, i128::from(c.sugar_max).checked_sub(sugar)?, work);
    let saturation = fx::mm(sugar, i128::from(k.km_r), work);
    let burnt = fx::mul(i128::from(k.r), saturation, work);
    let burnt = fx::sat(burnt, 0, sugar.checked_add(made)?, work);
    let left = sugar.checked_add(made)?.checked_sub(burnt)?;
    let excess = fx::sat(left.checked_sub(i128::from(c.theta_s))?, 0, I32_MAX, work);
    let stored = fx::mul(i128::from(k.sy), excess, work);
    let stored = fx::sat(stored, 0, left, work);
    let stored = fx::sat(stored, 0, i128::from(c.fructan_max).checked_sub(fructan)?, work);
    let lack = fx::sat(i128::from(c.theta_h).checked_sub(left)?, 0, I32_MAX, work);
    let drawn = fx::mul(i128::from(k.hy), lack, work);
    let drawn = fx::sat(drawn, 0, fructan.checked_sub(i128::from(c.core))?, work);
    let wide = i128::from(MAX_INT);
    org.sugar = i64::try_from(left.checked_sub(stored)?.checked_add(drawn)?).ok()?;
    org.fructan = i64::try_from(fructan.checked_add(stored)?.checked_sub(drawn)?).ok()?;
    org.made = i64::try_from(wide.min(i128::from(org.made).checked_add(made)?)).ok()?;
    org.burnt = i64::try_from(wide.min(i128::from(org.burnt).checked_add(burnt)?)).ok()?;
    org.metab = i64::try_from(made.checked_add(burnt)?).ok()?;
    Some(())
}

/// The dormancy from minute `since` to the wake at `wake`, applied at once: nothing changes.
pub fn jump(_org: &mut Chem, _since: i64, _wake: i64) {}

/// The metabolism channel: the step's `P + R` shifted up by `metab_shift`, at most `ONE`.
pub fn publish(org: &Chem, c: &ChemConsts) -> Option<i64> {
    let shift = u32::try_from(c.metab_shift).ok()?;
    let raised = i128::from(org.metab).checked_shl(shift)?;
    i64::try_from(ONE.min(raised)).ok()
}
