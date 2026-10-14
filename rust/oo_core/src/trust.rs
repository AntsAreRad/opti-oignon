//! The trust kernel's twin: the label of each argument of a tool call, and the gate's decision on the call, as
//! the Python reference (`opti_oignon/provenance.py`) computes them.
//!
//! Python stays the reference, and it decides. This twin is asked beside it, in its shadow: Python hands over
//! what it read, and every decision the twin would have taken otherwise is counted and said on the Python side.
//! What Python folds (Unicode NFC, each run of white space one space, the ends stripped) arrives here folded, as
//! bytes: the twin compares bytes and holds no Unicode table of its own. What Python reads from the world -- the
//! policy file, the answers a person gave -- arrives as Python read it, and the twin takes it as given.

use pyo3::prelude::*;
use std::collections::HashMap;

/// The shape of what the twin takes and answers. Python asks the twin only when this is the shape it hands over.
pub const FORMAT: u32 = 1;

const TYPED: &str = "typed";
const DEFAULT: &str = "default";
const UNENDORSED: &str = "unendorsed";

const NETWORK: &str = "network";
const DAILY: &str = "daily";

const FREE: &str = "free";
const ASK_UNENDORSED: &str = "ask_unendorsed";
const REFUSE_UNENDORSED: &str = "refuse_unendorsed";
/// Loosest first: a policy is tightened by moving right, never left.
const POLICIES: [&str; 3] = [FREE, ASK_UNENDORSED, REFUSE_UNENDORSED];

const ALLOWED: &str = "allowed";
const NOT_OFFERED: &str = "not_offered";
const NOT_PERMITTED: &str = "not_permitted";
const NO_CHANNEL: &str = "no_channel";
const DENIED: &str = "denied";
const UNENDORSED_REFUSED: &str = "unendorsed";
const UNSHOWABLE: &str = "unshowable";
/// The twin's own answer when a network call needs the policy file's reading and Python handed none: it never
/// guesses one.
const FILED_MISSING: &str = "filed_missing";

/// Whether `mode` permits a call of class `effect`. Daily permits the six classes; Bulbe, and any mode the
/// table does not know, permits `none`, `session` and `sandbox` alone.
fn permitted(effect: &str, mode: &str) -> bool {
    if mode == DAILY {
        matches!(effect, "none" | "session" | "sandbox" | "deferred" | "network" | "approved")
    } else {
        matches!(effect, "none" | "session" | "sandbox")
    }
}

fn rank(policy: &str) -> usize {
    POLICIES.iter().position(|known| *known == policy).unwrap_or(POLICIES.len() - 1)
}

/// A known policy as one of the three, an unknown one as the tightest.
fn known(policy: &str) -> &'static str {
    POLICIES.iter().copied().find(|p| *p == policy).unwrap_or(REFUSE_UNENDORSED)
}

/// The policy a network call obeys: Bulbe refuses; in Daily, the file's, tightened by the turn's; a policy the
/// turn names outside the three reads as the tightest. None when Daily needs the file's and none was handed over.
fn network_policy(mode: &str, explicit: Option<&str>, filed: Option<&str>) -> Option<&'static str> {
    if mode != DAILY {
        return Some(REFUSE_UNENDORSED);
    }
    let filed = known(filed?);
    let Some(explicit) = explicit else {
        return Some(filed);
    };
    let given = known(explicit);
    Some(if rank(filed) >= rank(given) { filed } else { given })
}

/// The label of each argument of a call, in the order given.
///
/// `units` are the typed units of the turn, each as (folded, as typed), in the turn's order: a fold endorses as
/// the first unit that folds to it. Each argument is (name, folded value or None when the value is not text, the
/// value as given, whether it equals the declared default in type and in value). A text whose fold is a unit's is
/// `typed` when the call reaches the network, which takes the unit's own characters, and otherwise only when it
/// is the unit character for character, `unendorsed` if not; any other value is `default` when it equals the
/// default, and `unendorsed` otherwise.
#[pyfunction]
pub fn trust_labels(
    network: bool,
    units: Vec<(Vec<u8>, Vec<u8>)>,
    arguments: Vec<(String, Option<Vec<u8>>, Vec<u8>, bool)>,
) -> Vec<(String, &'static str)> {
    let mut by_fold: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();
    for (folded, unit) in units {
        by_fold.entry(folded).or_insert(unit);
    }
    arguments
        .into_iter()
        .map(|(name, folded, given, default_equal)| {
            let unit = folded.as_ref().filter(|f| !f.is_empty()).and_then(|f| by_fold.get(f));
            let label = match unit {
                Some(unit) if network || given == *unit => TYPED,
                Some(_) => UNENDORSED,
                None if default_equal => DEFAULT,
                None => UNENDORSED,
            };
            (name, label)
        })
        .collect()
}

/// The gate's decision on one call: (allowed, the reason, whether a person was asked).
///
/// `labels` are the call's (name, label); `unendorsed_lengths` how many characters each unendorsed value takes on
/// the approval surfaces; `filed` the policy file's reading when Python read it; `answers` what each question put
/// to a person returned, in the order asked. The order of the checks is the reference's: offered, permitted in
/// the mode, the network policy for a network call that carries an unendorsed argument, the length a person can
/// be shown, the approval the call needs, then the question the policy asks.
#[pyfunction]
#[pyo3(signature = (tool, offered, effect, mode, labels, explicit, filed, unendorsed_lengths, shown_limit, approval_required, has_approval, has_demand, answers))]
#[allow(clippy::too_many_arguments)]
pub fn trust_decide(
    tool: &str,
    offered: Option<Vec<String>>,
    effect: &str,
    mode: &str,
    labels: Vec<(String, String)>,
    explicit: Option<String>,
    filed: Option<String>,
    unendorsed_lengths: Vec<usize>,
    shown_limit: usize,
    approval_required: bool,
    has_approval: bool,
    has_demand: bool,
    answers: Vec<bool>,
) -> (bool, &'static str, bool) {
    if let Some(offered) = &offered {
        if !offered.iter().any(|name| name == tool) {
            return (false, NOT_OFFERED, false);
        }
    }
    if !permitted(effect, mode) {
        return (false, NOT_PERMITTED, false);
    }
    let carries_unendorsed = labels.iter().any(|(_, label)| label == UNENDORSED);
    let policy = if effect == NETWORK && carries_unendorsed {
        match network_policy(mode, explicit.as_deref(), filed.as_deref()) {
            Some(policy) => policy,
            None => return (false, FILED_MISSING, false),
        }
    } else {
        FREE
    };
    if policy == REFUSE_UNENDORSED {
        return (false, UNENDORSED_REFUSED, false);
    }
    let asking = approval_required || policy == ASK_UNENDORSED;
    if effect == NETWORK && asking && unendorsed_lengths.iter().any(|&length| length > shown_limit) {
        return (false, UNSHOWABLE, false);
    }
    let mut answers = answers.into_iter();
    let mut asked = false;
    if approval_required {
        if !(has_approval || has_demand) {
            return (false, NO_CHANNEL, false);
        }
        asked = true;
        if !answers.next().unwrap_or(false) {
            return (false, DENIED, true);
        }
    }
    if policy == ASK_UNENDORSED && !asked {
        if !has_demand {
            return (false, NO_CHANNEL, false);
        }
        asked = true;
        if !answers.next().unwrap_or(false) {
            return (false, DENIED, true);
        }
    }
    (true, ALLOWED, asked)
}
