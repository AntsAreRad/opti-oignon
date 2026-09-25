#![deny(clippy::indexing_slicing, clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::todo, clippy::unimplemented)]
//! The life, the twin of `opti_oignon/allium/ref/world.py`: a being's state,
//! advanced from its genesis and facts, and the law timeline.
//!
//! `advance` folds a life. From the genesis (`state: null`) or from a state
//! it returned before, it takes the facts after that state, in canonical
//! order `(t, origin, oseq)`, and lives every minute up to `to`, or until
//! its budget of work units runs out after a whole processed minute or a
//! whole day of the fast path. The state it returns at minute `at` depends
//! only on the genesis, the facts with `t <= at` and `at`.
//!
//! Processing an event minute `e` (a fast boundary, a minute that holds
//! facts, or the `effective_from` of a pending `evolve`): the visit; a due
//! `evolve`; the facts of `e` in three passes (the `tz` facts, then the law
//! kinds, then every other kind to each organ in organ order), each fact
//! first passing the law-field check and its day's budget or cap; at a fast
//! boundary the fast layer; when the local day is above the high-water mark
//! the daily layer; then every organ publishes on the bus, a quiescent organ
//! 0 while the being sleeps. Every read is of the bus published at the
//! previous processed minute, so the order of the organs never matters.
//!
//! While the being is dormant, production lives a span a local midnight at a
//! time, running only the non-quiescent organs' daily steps; the span ends
//! before the next fact and before a pending `effective_from`. That minute
//! costs a visit: at a local midnight it stands in for the fast path's day,
//! but an `effective_from` a later `tz` fact moved off its midnight is met by
//! one visit more even while the being sleeps, and the law's `dormant_day`
//! ceiling counts it. A due `evolve` adds no visit only awake, where every
//! fast boundary is visited anyway. With
//! `probe.fast_path: false` every minute is stepped instead, and the quiescent
//! organs are called as identities; the states are the same.
//!
//! `timeline` folds the law kinds alone with the same functions, and
//! computes the daily firing minutes arithmetically.
//!
//! Laws. This engine carries no migration: every change of law, whether an
//! `evolve` names one or a pending change read from a state comes due, is
//! refused `unknown_law "migration"` once its target is known to be carried,
//! which is the reference's answer for every pair of carried laws.
//!
//! Every check runs in the reference's order, so a request with several
//! defects is refused for the same one. Nothing here indexes, unwraps or
//! panics; an arithmetic check that fails where the reference has no
//! refusal is `engine_panic`, and none is reachable from what a request may
//! carry.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use crate::civil;
use crate::fx::Work;
use crate::journal;
use crate::lawdata::{self, ChemK, ClockK, Life, Organ};
use crate::laws;
use crate::ocj::{self, obj, s, Refused, Value, MAX_INT};
use crate::organs::chem::{self, Chem};
use crate::organs::clock::{self, Clock};
use crate::organs::compile as organ_compile;
use crate::organs::genome;
use crate::organs::soil::{self, Soil};
use crate::organs::stage::{self, Cause, Stage};
use crate::organs::weather::Weather;
use crate::protocol::{bad, fields, int, text};

const DAY: i64 = civil::DAY;
const FAST: i64 = civil::FAST;
const T_MAX: i64 = civil::T_MAX;
const SCHEMA: i64 = 1;
const VISIT: u64 = 1;
const FACT: u64 = 1;
const FAST_PATH_DAY: u64 = 1;
const ONE: i64 = 1 << 16;
const CMAX: i64 = 8 << 16;
const I32_MAX: i64 = (1 << 31) - 1;
const LAW_KINDS: [&str; 3] = ["evolve", "laws_pin", "laws_unpin"];
const REDUCER_KINDS: [&str; 4] = ["evolve", "laws_pin", "laws_unpin", "tz"];
const STATE_KEYS: [&str; 16] = [
    "at", "being", "budget", "bus", "day", "genome", "law", "n", "organs", "params", "pending", "pinned", "schema", "seen",
    "through", "tz",
];
const TSTATE_KEYS: [&str; 11] = ["at", "budget", "day", "law", "params", "pending", "pinned", "schema", "seen", "through", "tz"];
const CHANNELS: [&str; 4] = ["circadian", "dormant", "metab", "moisture"];
const CAUSES: [&str; 3] = ["dry", "none", "winter"];
const PROBE_KEYS: [&str; 3] = ["order", "fast_path", "trace"];

type Answer = Result<Value, Refused>;

fn broken() -> Refused {
    bad("engine_panic", "life")
}

fn ck<T>(value: Option<T>) -> Result<T, Refused> {
    value.ok_or_else(broken)
}

fn add(a: i64, b: i64) -> Result<i64, Refused> {
    ck(a.checked_add(b))
}

fn sub(a: i64, b: i64) -> Result<i64, Refused> {
    ck(a.checked_sub(b))
}

fn floor_div(a: i64, b: i64) -> Result<i64, Refused> {
    ck(a.checked_div_euclid(b))
}

fn modulo(a: i64, b: i64) -> Result<i64, Refused> {
    ck(a.checked_rem_euclid(b))
}

fn tick(counter: &mut u64, n: u64) -> Result<(), Refused> {
    *counter = ck(counter.checked_add(n))?;
    Ok(())
}

fn number(n: u64) -> Result<Value, Refused> {
    i64::try_from(n).map(Value::Int).map_err(|_| broken())
}

fn member(key: &str, value: Value) -> (String, Value) {
    (String::from(key), value)
}

// ---------------------------------------------------------------------------
// Reading request values
// ---------------------------------------------------------------------------

fn int_in(value: Option<&Value>, low: i64, high: i64) -> bool {
    lawdata::int_in(value, low, high)
}

fn is_hex(value: Option<&Value>, length: usize) -> bool {
    matches!(text(value), Some(given) if ocj::is_hex(given, length))
}

fn exactly(value: Option<&Value>, names: &[&str]) -> bool {
    lawdata::exactly(value, names)
}

fn triple(value: Option<&Value>, low: i64, high: i64) -> bool {
    match value {
        Some(Value::Arr(items)) => items.len() == 3 && items.iter().all(|item| int_in(Some(item), low, high)),
        _ => false,
    }
}

fn get_int(value: &Value, key: &str) -> Result<i64, Refused> {
    ck(int(value.get(key)))
}

fn get_text<'v>(value: &'v Value, key: &str) -> Result<&'v str, Refused> {
    ck(text(value.get(key)))
}

fn get_triple(value: Option<&Value>) -> Result<[i64; 3], Refused> {
    match value {
        Some(Value::Arr(items)) => match items.as_slice() {
            [Value::Int(a), Value::Int(b), Value::Int(c)] => Ok([*a, *b, *c]),
            _ => Err(broken()),
        },
        _ => Err(broken()),
    }
}

fn int_map(value: Option<&Value>) -> Result<BTreeMap<String, i64>, Refused> {
    let mut out = BTreeMap::new();
    if let Some(Value::Obj(members)) = value {
        for (key, item) in members {
            out.insert(key.clone(), ck(int(Some(item)))?);
        }
        return Ok(out);
    }
    Err(broken())
}

// ---------------------------------------------------------------------------
// The state and its schema
// ---------------------------------------------------------------------------

fn chem_ok(org: Option<&Value>) -> bool {
    if !exactly(org, &["burnt", "fructan", "k", "made", "metab", "sugar"]) {
        return false;
    }
    let k = org.and_then(|o| o.get("k"));
    if !exactly(k, &["hy", "km_ps", "km_r", "ps", "r", "sy"]) {
        return false;
    }
    let rate = |name: &str| k.and_then(|k| k.get(name));
    ["hy", "ps", "r", "sy"].iter().all(|name| int_in(rate(name), 0, ONE))
        && ["km_ps", "km_r"].iter().all(|name| int_in(rate(name), 1, CMAX))
        && ["burnt", "fructan", "made", "metab", "sugar"].iter().all(|name| int_in(org.and_then(|o| o.get(name)), 0, MAX_INT))
}

fn clock_ok(org: Option<&Value>) -> bool {
    if !exactly(org, &["k", "p"]) || !triple(org.and_then(|o| o.get("p")), 0, CMAX) {
        return false;
    }
    let k = org.and_then(|o| o.get("k"));
    if !exactly(k, &["alpha", "beta", "k", "light", "n"]) {
        return false;
    }
    let part = |name: &str| k.and_then(|k| k.get(name));
    triple(part("alpha"), 0, I32_MAX)
        && triple(part("beta"), 0, I32_MAX)
        && triple(part("k"), 1, CMAX)
        && triple(part("n"), 1, 4)
        && int_in(part("light"), 0, I32_MAX)
}

fn stage_ok(org: Option<&Value>) -> bool {
    if !exactly(org, &["cause", "dormant", "dry", "rest", "season", "since"]) {
        return false;
    }
    let part = |name: &str| org.and_then(|o| o.get(name));
    matches!(text(part("cause")), Some(cause) if CAUSES.contains(&cause))
        && matches!(part("dormant"), Some(Value::Bool(_)))
        && int_in(part("dry"), 0, MAX_INT)
        && int_in(part("rest"), 0, MAX_INT)
        && int_in(part("season"), 0, civil::SEASONS.saturating_sub(1))
        && int_in(part("since"), 0, T_MAX)
}

fn organs_ok(organs: Option<&Value>) -> bool {
    if !exactly(organs, &lawdata::CODE_ORGANS) {
        return false;
    }
    let part = |name: &str| organs.and_then(|o| o.get(name));
    let soil_org = part("soil");
    chem_ok(part("chem"))
        && clock_ok(part("clock"))
        && stage_ok(part("stage"))
        && exactly(soil_org, &["m"])
        && int_in(soil_org.and_then(|s| s.get("m")), 0, MAX_INT)
}

fn body_ok(schema: Option<&Value>, value: Option<&Value>) -> bool {
    match (schema, value) {
        (Some(schema), Some(value)) => journal::check_body(schema, value, "").is_ok(),
        _ => false,
    }
}

fn budget_ok(budget: Option<&Value>, table: &Value) -> bool {
    if !exactly(budget, &["counts", "day"]) || !int_in(budget.and_then(|b| b.get("day")), 0, MAX_INT) {
        return false;
    }
    let Some(Value::Obj(counts)) = budget.and_then(|b| b.get("counts")) else {
        return false;
    };
    let kinds = table.get("kinds");
    counts.iter().all(|(kind, count)| {
        let entry = kinds.and_then(|k| k.get(kind));
        entry.is_some()
            && text(entry.and_then(|e| e.get("scope"))) == Some("trunk")
            && kind != "genesis"
            && int_in(Some(count), 1, MAX_INT)
    })
}

fn bus_ok(bus: Option<&Value>) -> bool {
    exactly(bus, &CHANNELS) && CHANNELS.iter().all(|name| int_in(bus.and_then(|b| b.get(name)), 0, ONE))
}

fn law_ok(law: Option<&Value>) -> bool {
    exactly(law, &["name", "sha256", "v"])
        && matches!(text(law.and_then(|l| l.get("name"))), Some(name) if (1..=32).contains(&name.len()))
        && is_hex(law.and_then(|l| l.get("sha256")), 64)
        && int_in(law.and_then(|l| l.get("v")), 0, 65535)
}

fn seen_ok(seen: Option<&Value>) -> bool {
    let Some(Value::Arr(items)) = seen else {
        return false;
    };
    if items.is_empty() {
        return false;
    }
    let mut last: i64 = -1;
    for item in items {
        match int(Some(item)) {
            Some(v) if (0..=65535).contains(&v) && v > last => last = v,
            _ => return false,
        }
    }
    true
}

fn through_ok(through: Option<&Value>) -> bool {
    match through {
        Some(Value::Null) => true,
        Some(Value::Arr(items)) => match items.as_slice() {
            [t, origin, oseq] => int_in(Some(t), 0, T_MAX) && is_hex(Some(origin), 16) && int_in(Some(oseq), 0, MAX_INT),
            _ => false,
        },
        _ => false,
    }
}

fn state_check(key: &str, value: Option<&Value>, table: &Value) -> bool {
    match key {
        "at" => int_in(value, 0, T_MAX),
        "being" => is_hex(value, 32),
        "budget" => budget_ok(value, table),
        "bus" => bus_ok(value),
        "day" => int_in(value, 0, MAX_INT),
        "genome" => is_hex(value, 64),
        "law" => law_ok(value),
        "n" => int_in(value, 0, MAX_INT),
        "organs" => organs_ok(value),
        "params" => body_ok(lawdata::params_schema(table), value),
        "pending" => {
            matches!(value, Some(Value::Null))
                || body_ok(table.get("kinds").and_then(|k| k.get("evolve")).and_then(|e| e.get("body")), value)
        }
        "pinned" => matches!(value, Some(Value::Bool(_))),
        "schema" => int(value) == Some(SCHEMA),
        "seen" => seen_ok(value),
        "through" => through_ok(value),
        "tz" => matches!(int(value), Some(z) if civil::offset_ok(z)),
        _ => false,
    }
}

/// The first defect of a state (or a timeline state) as the detail of its refusal, `None` when it is whole.
fn state_defect(value: &Value, keys: &[&str], table: &Value) -> Option<String> {
    if !matches!(value, Value::Obj(_)) {
        return Some(String::from("state"));
    }
    if !exactly(Some(value), keys) {
        return Some(String::from("state fields"));
    }
    for key in keys {
        if !state_check(key, value.get(key), table) {
            let mut detail = String::from("state ");
            detail.push_str(key);
            return Some(detail);
        }
    }
    let v = int(value.get("law").and_then(|l| l.get("v")));
    let in_seen = match value.get("seen") {
        Some(Value::Arr(items)) => items.iter().any(|item| int(Some(item)) == v),
        _ => false,
    };
    if !in_seen {
        return Some(String::from("state seen"));
    }
    if let (Some(Value::Arr(through)), Some(at)) = (value.get("through"), int(value.get("at"))) {
        if let Some(t) = through.first().and_then(|t| int(Some(t))) {
            if t > at {
                return Some(String::from("state through"));
            }
        }
    }
    None
}

/// The law a state names, as the reducer reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Law {
    name: String,
    sha256: String,
    v: i64,
}

impl Law {
    fn value(&self) -> Value {
        obj(vec![member("name", s(&self.name)), member("sha256", s(&self.sha256)), member("v", Value::Int(self.v))])
    }

    fn read(value: Option<&Value>) -> Result<Law, Refused> {
        let value = ck(value)?;
        Ok(Law { name: String::from(get_text(value, "name")?), sha256: String::from(get_text(value, "sha256")?), v: get_int(value, "v")? })
    }
}

/// A registered change of law or params, waiting for its minute.
#[derive(Clone, Debug)]
struct Pending {
    effective_from: i64,
    from_name: String,
    from_sha: String,
    params: BTreeMap<String, i64>,
    to: Law,
}

impl Pending {
    fn value(&self) -> Value {
        obj(vec![
            member("effective_from", Value::Int(self.effective_from)),
            member("from", obj(vec![member("name", s(&self.from_name)), member("sha256", s(&self.from_sha))])),
            member("params", params_value(&self.params)),
            member("to", self.to.value()),
        ])
    }

    fn read(value: &Value) -> Result<Pending, Refused> {
        let from = ck(value.get("from"))?;
        Ok(Pending {
            effective_from: get_int(value, "effective_from")?,
            from_name: String::from(get_text(from, "name")?),
            from_sha: String::from(get_text(from, "sha256")?),
            params: int_map(value.get("params"))?,
            to: Law::read(value.get("to"))?,
        })
    }
}

fn params_value(params: &BTreeMap<String, i64>) -> Value {
    Value::Obj(params.iter().map(|(name, value)| (name.clone(), Value::Int(*value))).collect())
}

/// What the law timeline folds: the part of a state `advance` and `timeline` share.
#[derive(Clone, Debug)]
struct Core {
    at: i64,
    budget_counts: BTreeMap<String, i64>,
    budget_day: i64,
    day: i64,
    law: Law,
    params: BTreeMap<String, i64>,
    pending: Option<Pending>,
    pinned: bool,
    seen: Vec<i64>,
    through: Option<(i64, String, i64)>,
    tz: i64,
}

impl Core {
    fn members(&self) -> Vec<(String, Value)> {
        let counts = Value::Obj(self.budget_counts.iter().map(|(kind, n)| (kind.clone(), Value::Int(*n))).collect());
        let through = match &self.through {
            Some((t, origin, oseq)) => Value::Arr(vec![Value::Int(*t), s(origin), Value::Int(*oseq)]),
            None => Value::Null,
        };
        vec![
            member("at", Value::Int(self.at)),
            member("budget", obj(vec![member("counts", counts), member("day", Value::Int(self.budget_day))])),
            member("day", Value::Int(self.day)),
            member("law", self.law.value()),
            member("params", params_value(&self.params)),
            member("pending", self.pending.as_ref().map_or(Value::Null, Pending::value)),
            member("pinned", Value::Bool(self.pinned)),
            member("schema", Value::Int(SCHEMA)),
            member("seen", Value::Arr(self.seen.iter().map(|v| Value::Int(*v)).collect())),
            member("through", through),
            member("tz", Value::Int(self.tz)),
        ]
    }

    fn read(value: &Value) -> Result<Core, Refused> {
        let budget = ck(value.get("budget"))?;
        let through = match value.get("through") {
            Some(Value::Arr(items)) => match items.as_slice() {
                [Value::Int(t), Value::Str(origin), Value::Int(oseq)] => Some((*t, origin.clone(), *oseq)),
                _ => return Err(broken()),
            },
            _ => None,
        };
        let seen = match value.get("seen") {
            Some(Value::Arr(items)) => items.iter().map(|item| ck(int(Some(item)))).collect::<Result<Vec<i64>, Refused>>()?,
            _ => return Err(broken()),
        };
        Ok(Core {
            at: get_int(value, "at")?,
            budget_counts: int_map(budget.get("counts"))?,
            budget_day: get_int(budget, "day")?,
            day: get_int(value, "day")?,
            law: Law::read(value.get("law"))?,
            params: int_map(value.get("params"))?,
            pending: match value.get("pending") {
                Some(Value::Null) | None => None,
                Some(pending) => Some(Pending::read(pending)?),
            },
            pinned: matches!(value.get("pinned"), Some(Value::Bool(true))),
            seen,
            through,
            tz: get_int(value, "tz")?,
        })
    }
}

/// The bus: the channels the organs published at the last processed minute.
#[derive(Clone, Copy, Debug, Default)]
struct Bus {
    circadian: i64,
    dormant: i64,
    metab: i64,
    moisture: i64,
}

/// A being's whole state.
#[derive(Clone, Debug)]
struct State {
    core: Core,
    being: String,
    bus: Bus,
    genome: String,
    n: i64,
    chem: Chem,
    clock: Clock,
    soil: Soil,
    stage: Stage,
}

fn chem_k_value(k: &ChemK) -> Value {
    obj(vec![
        member("hy", Value::Int(k.hy)),
        member("km_ps", Value::Int(k.km_ps)),
        member("km_r", Value::Int(k.km_r)),
        member("ps", Value::Int(k.ps)),
        member("r", Value::Int(k.r)),
        member("sy", Value::Int(k.sy)),
    ])
}

fn ints(values: &[i64]) -> Value {
    Value::Arr(values.iter().map(|v| Value::Int(*v)).collect())
}

fn clock_k_value(k: &ClockK) -> Value {
    obj(vec![
        member("alpha", ints(&k.alpha)),
        member("beta", ints(&k.beta)),
        member("k", ints(&k.k)),
        member("light", Value::Int(k.light)),
        member("n", ints(&k.n)),
    ])
}

impl State {
    fn value(&self) -> Value {
        let mut members = self.core.members();
        members.push(member("being", s(&self.being)));
        members.push(member(
            "bus",
            obj(vec![
                member("circadian", Value::Int(self.bus.circadian)),
                member("dormant", Value::Int(self.bus.dormant)),
                member("metab", Value::Int(self.bus.metab)),
                member("moisture", Value::Int(self.bus.moisture)),
            ]),
        ));
        members.push(member("genome", s(&self.genome)));
        members.push(member("n", Value::Int(self.n)));
        let chem_org = &self.chem;
        let stage_org = &self.stage;
        members.push(member(
            "organs",
            obj(vec![
                member(
                    "chem",
                    obj(vec![
                        member("burnt", Value::Int(chem_org.burnt)),
                        member("fructan", Value::Int(chem_org.fructan)),
                        member("k", chem_k_value(&chem_org.k)),
                        member("made", Value::Int(chem_org.made)),
                        member("metab", Value::Int(chem_org.metab)),
                        member("sugar", Value::Int(chem_org.sugar)),
                    ]),
                ),
                member("clock", obj(vec![member("k", clock_k_value(&self.clock.k)), member("p", ints(&self.clock.p))])),
                member("soil", obj(vec![member("m", Value::Int(self.soil.m))])),
                member(
                    "stage",
                    obj(vec![
                        member("cause", s(stage_org.cause.name())),
                        member("dormant", Value::Bool(stage_org.dormant)),
                        member("dry", Value::Int(stage_org.dry)),
                        member("rest", Value::Int(stage_org.rest)),
                        member("season", Value::Int(stage_org.season)),
                        member("since", Value::Int(stage_org.since)),
                    ]),
                ),
            ]),
        ));
        obj(members)
    }

    fn read(value: &Value) -> Result<State, Refused> {
        let bus = ck(value.get("bus"))?;
        let organs = ck(value.get("organs"))?;
        let chem_value = ck(organs.get("chem"))?;
        let chem_k = ck(chem_value.get("k"))?;
        let clock_value = ck(organs.get("clock"))?;
        let clock_k = ck(clock_value.get("k"))?;
        let stage_value = ck(organs.get("stage"))?;
        Ok(State {
            core: Core::read(value)?,
            being: String::from(get_text(value, "being")?),
            bus: Bus {
                circadian: get_int(bus, "circadian")?,
                dormant: get_int(bus, "dormant")?,
                metab: get_int(bus, "metab")?,
                moisture: get_int(bus, "moisture")?,
            },
            genome: String::from(get_text(value, "genome")?),
            n: get_int(value, "n")?,
            chem: Chem {
                burnt: get_int(chem_value, "burnt")?,
                fructan: get_int(chem_value, "fructan")?,
                k: ChemK {
                    hy: get_int(chem_k, "hy")?,
                    km_ps: get_int(chem_k, "km_ps")?,
                    km_r: get_int(chem_k, "km_r")?,
                    ps: get_int(chem_k, "ps")?,
                    r: get_int(chem_k, "r")?,
                    sy: get_int(chem_k, "sy")?,
                },
                made: get_int(chem_value, "made")?,
                metab: get_int(chem_value, "metab")?,
                sugar: get_int(chem_value, "sugar")?,
            },
            clock: Clock {
                k: ClockK {
                    alpha: get_triple(clock_k.get("alpha"))?,
                    beta: get_triple(clock_k.get("beta"))?,
                    k: get_triple(clock_k.get("k"))?,
                    light: get_int(clock_k, "light")?,
                    n: get_triple(clock_k.get("n"))?,
                },
                p: get_triple(clock_value.get("p"))?,
            },
            soil: Soil { m: get_int(ck(organs.get("soil"))?, "m")? },
            stage: Stage {
                cause: ck(Cause::from_name(get_text(stage_value, "cause")?))?,
                dormant: matches!(stage_value.get("dormant"), Some(Value::Bool(true))),
                dry: get_int(stage_value, "dry")?,
                rest: get_int(stage_value, "rest")?,
                season: get_int(stage_value, "season")?,
                since: get_int(stage_value, "since")?,
            },
        })
    }
}

/// The stocks and levels of a state within the bounds of the law in force.
fn bounds_ok(organs: Option<&Value>, life: &Life) -> bool {
    let part = |organ: &str, key: &str| int(organs.and_then(|o| o.get(organ)).and_then(|o| o.get(key)));
    let (Some(sugar), Some(fructan), Some(m), Some(dry), Some(rest)) =
        (part("chem", "sugar"), part("chem", "fructan"), part("soil", "m"), part("stage", "dry"), part("stage", "rest"))
    else {
        return false;
    };
    let c = &life.chem;
    let rest_max = life.stage.rest_max;
    sugar <= c.sugar_max
        && c.core <= fructan
        && fructan <= c.fructan_max
        && m <= life.soil.m_max
        && dry <= rest_max
        && rest <= rest_max
}

/// The carried law a state names as in force: refused by name, digest or version, then by soundness.
fn state_law(law: Option<&Value>) -> Result<Life, Refused> {
    let law = ck(law)?;
    let name = get_text(law, "name")?;
    let (law_value, digest) = lawdata::law_file(name)?;
    if text(law.get("sha256")) != Some(digest.as_str()) {
        return Err(bad("unknown_law", "law digest"));
    }
    if int(law.get("v")) != int(law_value.get("version")) || int(law_value.get("version")).is_none() {
        return Err(bad("unknown_law", "law version"));
    }
    lawdata::life(name)
}

// ---------------------------------------------------------------------------
// The checks of a request, in the order that decides which defect is named
// ---------------------------------------------------------------------------

/// A genesis checked against the carried law it names.
struct Genesis<'r> {
    being: &'r str,
    b: i64,
    birth_tz: i64,
    band: &'r str,
    weather: Weather,
    params: BTreeMap<String, i64>,
    provisional: bool,
}

fn seed_of(hex: &str) -> Result<[u8; 32], Refused> {
    ck(ocj::from_hex(hex).and_then(|bytes| <[u8; 32]>::try_from(bytes).ok()))
}

fn halves(being: &str) -> Result<(u64, u64), Refused> {
    let bytes = ck(ocj::from_hex(being))?;
    let (hi, lo) = match bytes.as_slice() {
        [a, b, c, d, e, f, g, h, rest @ ..] => ([*a, *b, *c, *d, *e, *f, *g, *h], rest),
        _ => return Err(broken()),
    };
    let lo = ck(<[u8; 8]>::try_from(lo).ok())?;
    Ok((u64::from_be_bytes(hi), u64::from_be_bytes(lo)))
}

fn genesis_of(request: &Value) -> Result<(Genesis<'_>, Life), Refused> {
    let genesis = ck(request.get("genesis"))?;
    journal::check_envelope(genesis, false)?;
    if text(genesis.get("kind")) != Some("genesis") || int(genesis.get("t")) != Some(0) || int(genesis.get("oseq")) != Some(0) {
        return Err(bad("bad_fact", "genesis"));
    }
    let body = ck(genesis.get("body"))?;
    let laws = body.get("laws");
    let (name, sha) = match laws {
        Some(laws @ Value::Obj(_)) => match (text(laws.get("name")), text(laws.get("sha256"))) {
            (Some(name), Some(sha)) => (name, sha),
            _ => return Err(bad("bad_fact", "genesis laws")),
        },
        _ => return Err(bad("bad_fact", "genesis laws")),
    };
    let (_law, digest) = lawdata::law_file(name)?;
    if sha != digest {
        return Err(bad("unknown_law", "law digest"));
    }
    let life = lawdata::life(name)?;
    let schema = life
        .table
        .get("kinds")
        .and_then(|kinds| kinds.get("genesis"))
        .and_then(|entry| entry.get("body"))
        .ok_or_else(|| bad("unknown_law", "journal"))?;
    journal::check_body(schema, body, "")?;
    let laws = ck(laws)?;
    let provisional = matches!(laws.get("provisional"), Some(Value::Bool(true)));
    if int(laws.get("v")) != Some(life.version) || provisional != life.provisional {
        return Err(bad("bad_fact", "genesis laws"));
    }
    let birth = ck(body.get("birth"))?;
    let wall = get_int(birth, "wall")?;
    let birth_tz = get_int(birth, "tz")?;
    if wall < civil::WALL_MIN || modulo(birth_tz, civil::TZ_STEP)? != 0 {
        return Err(bad("bad_fact", "birth"));
    }
    let params = int_map(laws.get("params"))?;
    if let Some(name) = lawdata::params_defect(&params, &life)? {
        let mut detail = String::from("params ");
        detail.push_str(name);
        return Err(bad("bad_fact", &detail));
    }
    if int(genesis.get("laws")) != int(laws.get("v")) {
        return Err(bad("chain", "laws"));
    }
    let being = get_text(genesis, "being")?;
    let (being_hi, being_lo) = halves(being)?;
    let weather = Weather {
        garden: text(body.get("weather")) == Some("garden"),
        south: text(body.get("hemisphere")) == Some("south"),
        seed: seed_of(get_text(body, "seed")?)?,
        being_hi,
        being_lo,
    };
    Ok((
        Genesis {
            being,
            b: floor_div(wall, 60)?,
            birth_tz,
            band: get_text(body, "band")?,
            weather,
            params,
            provisional,
        },
        life,
    ))
}

fn to_of(request: &Value) -> Result<i64, Refused> {
    match int(request.get("to")) {
        Some(to) if (0..=T_MAX).contains(&to) => Ok(to),
        _ => Err(bad("bad_request", "to")),
    }
}

/// `(order, fast_path, trace)` from the optional probe; the default is the law's order, the fast path, no trace.
fn probe_of(request: &Value, life: &Life) -> Result<(Vec<Organ>, bool, bool), Refused> {
    let Some(probe) = request.get("probe") else {
        return Ok((life.organs.clone(), true, false));
    };
    let refuse = || bad("bad_request", "probe");
    if !matches!(probe, Value::Obj(_)) || probe.keys().iter().any(|key| !PROBE_KEYS.contains(key)) {
        return Err(refuse());
    }
    let mut order = life.organs.clone();
    if let Some(given) = probe.get("order") {
        let names = lawdata::distinct_names(Some(given)).ok_or_else(refuse)?;
        let mut sorted_names = names.clone();
        sorted_names.sort_unstable();
        let mut law_names: Vec<&str> = life.organs.iter().map(|organ| organ.name()).collect();
        law_names.sort_unstable();
        if sorted_names != law_names {
            return Err(refuse());
        }
        order = names.iter().map(|name| Organ::from_name(name).ok_or_else(refuse)).collect::<Result<Vec<Organ>, Refused>>()?;
    }
    let mut flags = [true, false];
    for (key, flag) in ["fast_path", "trace"].iter().zip(flags.iter_mut()) {
        match probe.get(key) {
            None => {}
            Some(Value::Bool(given)) => *flag = *given,
            Some(_) => return Err(refuse()),
        }
    }
    let [fast_path, trace] = flags;
    Ok((order, fast_path, trace))
}

/// The state a request resumes from, checked, and the law in force; `None` from the genesis.
fn resumed<'r>(request: &'r Value, key: &str, keys: &[&str], genesis: &Genesis<'_>, life: &Life, to: i64) -> Result<Option<(&'r Value, Life)>, Refused> {
    let value = ck(request.get(key))?;
    if matches!(value, Value::Null) {
        return Ok(None);
    }
    if let Some(defect) = state_defect(value, keys, &life.table) {
        return Err(bad("bad_request", &defect));
    }
    let whole = keys.len() == STATE_KEYS.len();
    if whole && text(value.get("being")) != Some(genesis.being) {
        return Err(bad("bad_request", "state being"));
    }
    let in_force = state_law(value.get("law"))?;
    if whole && !bounds_ok(value.get("organs"), &in_force) {
        return Err(bad("bad_request", "state organs"));
    }
    if to < get_int(value, "at")? {
        return Err(bad("bad_request", "to"));
    }
    Ok(Some((value, in_force)))
}

/// A checked fact of a life.
struct Fact<'r> {
    t: i64,
    origin: &'r str,
    oseq: i64,
    kind: &'r str,
    laws: i64,
    body: &'r Value,
}

/// What the kinds table says of one fact of a life, in the reference's order; every refusal is `bad_fact`.
fn check_fact(fact: &Value, table: &Value) -> Result<(), Refused> {
    let kind = text(fact.get("kind")).unwrap_or("");
    let entry = table.get("kinds").and_then(|kinds| kinds.get(kind)).ok_or_else(|| bad("bad_fact", "kind"))?;
    if text(entry.get("scope")) != Some("trunk") {
        return Err(bad("bad_fact", "scope"));
    }
    if kind == "genesis" {
        return Err(bad("bad_fact", "genesis"));
    }
    let schema = match entry.get("body") {
        None | Some(Value::Null) => return Err(bad("bad_fact", "reserved")),
        Some(schema) => schema,
    };
    match fact.get("body") {
        Some(body @ Value::Obj(_)) => journal::check_body(schema, body, ""),
        _ => match entry.get("redact_by") {
            None | Some(Value::Null) => Err(bad("bad_fact", "redaction")),
            Some(_) => Ok(()),
        },
    }
}

/// The facts of a request, each checked, then in canonical order after the state, and none after `to`.
fn facts_of<'r>(request: &'r Value, genesis: &Genesis<'_>, table: &Value, floor: Option<&Core>, to: i64, timeline: bool) -> Result<Vec<Fact<'r>>, Refused> {
    let items = match request.get("facts") {
        Some(Value::Arr(items)) => items,
        _ => return Err(bad("bad_request", "facts")),
    };
    if items.len() > crate::protocol::LIMIT_ITEMS {
        return Err(bad("limit", "items"));
    }
    let mut out: Vec<Fact<'r>> = Vec::with_capacity(items.len());
    for fact in items {
        let digest_body = matches!(fact.get("body"), Some(Value::Str(_)));
        journal::check_envelope(fact, digest_body)?;
        let t = get_int(fact, "t")?;
        if !(0..=T_MAX).contains(&t) {
            return Err(bad("bad_fact", "t"));
        }
        let body = ck(fact.get("body"))?;
        if matches!(body, Value::Obj(_)) && ocj::emit(body)?.len() > journal::BODY_LIMIT {
            return Err(bad("limit", "body size"));
        }
        if text(fact.get("being")) != Some(genesis.being) {
            return Err(bad("bad_fact", "being"));
        }
        check_fact(fact, table)?;
        let kind = get_text(fact, "kind")?;
        if timeline && !REDUCER_KINDS.contains(&kind) {
            return Err(bad("bad_request", "timeline kind"));
        }
        let origin = get_text(fact, "origin")?;
        let oseq = get_int(fact, "oseq")?;
        let key = (t, origin, oseq);
        if let Some(core) = floor {
            let below_through = match &core.through {
                Some((through_t, through_origin, through_oseq)) => key <= (*through_t, through_origin.as_str(), *through_oseq),
                None => false,
            };
            if t <= core.at || below_through {
                return Err(bad("chain", "order"));
            }
        }
        if let Some(previous) = out.last() {
            if key <= (previous.t, previous.origin, previous.oseq) {
                return Err(bad("chain", "order"));
            }
        }
        if t > to {
            return Err(bad("bad_request", "to"));
        }
        out.push(Fact { t, origin, oseq, kind, laws: get_int(fact, "laws")?, body });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The fold both drivers share: a fact consumed, the law kinds, the due evolve
// ---------------------------------------------------------------------------

/// The carried law a change goes to, from the law it names as its source; refused by name otherwise.
///
/// The target is carried, under its digest and its version; a change of law
/// (another name or digest than the source) is refused `migration`: this
/// engine carries no migration, which is the reference's answer for every
/// pair of carried laws.
fn approach(target: &Law, source_name: &str, source_sha: &str) -> Result<Life, Refused> {
    let (target_value, digest) = lawdata::law_file(&target.name)?;
    if target.sha256 != digest {
        return Err(bad("unknown_law", "law digest"));
    }
    if int(target_value.get("version")) != Some(target.v) {
        return Err(bad("unknown_law", "law version"));
    }
    let target_life = lawdata::life(&target.name)?;
    if target.name != source_name || target.sha256 != source_sha {
        return Err(bad("unknown_law", "migration"));
    }
    Ok(target_life)
}

/// The law in force and the notes, folding facts; `advance` and `timeline` differ in their drivers only.
struct Fold {
    life: Life,
    b: i64,
    notes: Vec<(&'static str, i64)>,
    facts: u64,
    noted: u64,
}

impl Fold {
    fn note(&mut self, code: &'static str, t: i64) {
        self.notes.push((code, t));
    }

    fn noted_one(&mut self) -> Result<(), Refused> {
        tick(&mut self.noted, 1)
    }

    /// One fact consumed (one unit): its law field, then its day's budget or cap; whether it is folded.
    fn consume(&mut self, st: &mut Core, fact: &Fact<'_>, work: &mut Work) -> Result<bool, Refused> {
        tick(&mut work.units, FACT)?;
        tick(&mut self.facts, 1)?;
        if !st.seen.contains(&fact.laws) {
            return Err(bad("chain", "laws"));
        }
        let day = floor_div(fact.t, DAY)?;
        if st.budget_day != day {
            st.budget_counts = BTreeMap::new();
            st.budget_day = day;
        }
        let count = st.budget_counts.get(fact.kind).copied().unwrap_or(0);
        let limit = ck(self.life.limits.get(fact.kind).copied())?;
        if count >= limit {
            self.note("budget", fact.t);
            self.noted_one()?;
            return Ok(false);
        }
        st.budget_counts.insert(String::from(fact.kind), add(count, 1)?);
        if fact.laws != st.law.v {
            self.note("laws_stale", fact.t);
        }
        Ok(true)
    }

    /// The first pass: a `tz` fact moves the offset in force.
    fn tz_fact(&mut self, st: &mut Core, fact: &Fact<'_>, work: &mut Work) -> Result<(), Refused> {
        if self.consume(st, fact, work)? {
            st.tz = ck(get_int(fact.body, "quarters")?.checked_mul(civil::TZ_STEP))?;
        }
        Ok(())
    }

    /// The second pass: a pin, an unpin, or an `evolve` registered against the offset after the first pass.
    fn law_fact(&mut self, st: &mut Core, fact: &Fact<'_>, e: i64, work: &mut Work) -> Result<(), Refused> {
        if !self.consume(st, fact, work)? {
            return Ok(());
        }
        match fact.kind {
            "laws_pin" => {
                if st.pinned {
                    self.note("pin_twice", e);
                    self.noted_one()?;
                } else {
                    st.pinned = true;
                }
                Ok(())
            }
            "laws_unpin" => {
                if !st.pinned {
                    self.note("unpin_unpinned", e);
                    self.noted_one()?;
                } else {
                    st.pinned = false;
                }
                Ok(())
            }
            _ => self.register(st, fact.body, e),
        }
    }

    fn register(&mut self, st: &mut Core, body: &Value, e: i64) -> Result<(), Refused> {
        let target = Law::read(body.get("to"))?;
        let source = ck(body.get("from"))?;
        let (source_name, source_sha) = (get_text(source, "name")?, get_text(source, "sha256")?);
        let target_life = approach(&target, source_name, source_sha)?;
        let effective_from = get_int(body, "effective_from")?;
        if effective_from != ck(civil::next_midnight(self.b, e, st.tz))? {
            self.note("evolve_when", e);
            return self.noted_one();
        }
        let params = int_map(body.get("params"))?;
        if lawdata::params_defect(&params, &target_life)?.is_some() {
            self.note("evolve_params", e);
            return self.noted_one();
        }
        if st.pending.is_some() {
            self.note("evolve_superseded", e);
        }
        st.pending = Some(Pending {
            effective_from,
            from_name: String::from(source_name),
            from_sha: String::from(source_sha),
            params,
            to: target,
        });
        Ok(())
    }

    /// The pending `evolve` at its minute: dropped when pinned or from another law, else applied.
    fn due(&mut self, st: &mut Core, e: i64) -> Result<(), Refused> {
        let pending = ck(st.pending.take())?;
        if st.pinned {
            self.note("evolve_pinned", e);
            return Ok(());
        }
        if pending.from_name != st.law.name || pending.from_sha != st.law.sha256 {
            self.note("evolve_from", e);
            return Ok(());
        }
        if pending.to.name != st.law.name || pending.to.sha256 != st.law.sha256 {
            // A pending change read from a state is approached as its registration was, and refused.
            approach(&pending.to, &pending.from_name, &pending.from_sha)?;
            return Err(bad("unknown_law", "migration"));
        }
        if pending.to.v != st.law.v {
            // The same law under another version: approached too, so a state this engine returns never
            // names a law it would refuse.
            approach(&pending.to, &pending.from_name, &pending.from_sha)?;
        }
        let v = pending.to.v;
        st.law = pending.to;
        st.params = pending.params;
        if !st.seen.contains(&v) {
            st.seen.push(v);
            st.seen.sort_unstable();
        }
        Ok(())
    }
}

fn capped_notes(notes: &[(&'static str, i64)], at: i64) -> Value {
    let limit = crate::protocol::LIMIT_NOTES;
    let note = |code: &str, t: i64| obj(vec![member("code", s(code)), member("t", Value::Int(t))]);
    if notes.len() <= limit {
        return Value::Arr(notes.iter().map(|(code, t)| note(code, *t)).collect());
    }
    let mut out: Vec<Value> = notes.iter().take(limit.saturating_sub(1)).map(|(code, t)| note(code, *t)).collect();
    out.push(note("truncated", at));
    Value::Arr(out)
}

// ---------------------------------------------------------------------------
// advance
// ---------------------------------------------------------------------------

/// The calls of one organ: daily awake and dormant, fast awake and dormant, jumps.
#[derive(Clone, Copy, Debug, Default)]
struct Calls {
    daily: [u64; 2],
    fast: [u64; 2],
    jump: u64,
}

/// The stepped path, the fast path, and the organs' layers.
struct Reducer {
    fold: Fold,
    order: Vec<Organ>,
    fast_path: bool,
    work: Work,
    calls: BTreeMap<Organ, Calls>,
    acts: BTreeMap<String, u64>,
    visits: u64,
    env: u64,
    fast_path_days: u64,
    fast_path_skipped: u64,
    draws: u64,
    clips: u64,
    sine: Vec<i128>,
    weather: Weather,
    amp: i64,
}

fn slot(dormant: bool) -> usize {
    usize::from(dormant)
}

fn bump(pair: &mut [u64; 2], dormant: bool) -> Result<(), Refused> {
    let cell = ck(pair.get_mut(slot(dormant)))?;
    tick(cell, 1)
}

impl Reducer {
    fn new(life: Life, genesis: &Genesis<'_>, order: Vec<Organ>, fast_path: bool) -> Result<Reducer, Refused> {
        let calls = life.organs.iter().map(|organ| (*organ, Calls::default())).collect();
        let amp = ck(life.world.amp(genesis.band))?;
        Ok(Reducer {
            fold: Fold { life, b: genesis.b, notes: Vec::new(), facts: 0, noted: 0 },
            order,
            fast_path,
            work: Work::default(),
            calls,
            acts: BTreeMap::new(),
            visits: 0,
            env: 0,
            fast_path_days: 0,
            fast_path_skipped: 0,
            draws: 0,
            clips: 0,
            sine: lawdata::sine()?,
            weather: genesis.weather.clone(),
            amp,
        })
    }

    fn quiescent(&self, organ: Organ) -> bool {
        self.fold.life.is_quiescent(organ)
    }

    fn calls_of(&mut self, organ: Organ) -> Result<&mut Calls, Refused> {
        ck(self.calls.get_mut(&organ))
    }

    fn publish(&self, st: &mut State) -> Result<(), Refused> {
        let dormant = st.stage.dormant;
        let life = &self.fold.life;
        for organ in &life.organs {
            let masked = dormant && life.is_quiescent(*organ);
            let value = match organ {
                Organ::Chem => ck(chem::publish(&st.chem, &life.chem))?,
                Organ::Clock => clock::publish(&st.clock),
                Organ::Soil => soil::publish(&st.soil),
                Organ::Stage => stage::publish(&st.stage),
            };
            let value = if masked { 0 } else { value };
            match organ {
                Organ::Chem => st.bus.metab = value,
                Organ::Clock => st.bus.circadian = value,
                Organ::Soil => st.bus.moisture = value,
                Organ::Stage => st.bus.dormant = value,
            }
        }
        Ok(())
    }

    fn jumps(&mut self, st: &mut State, since: i64, wake: i64) -> Result<(), Refused> {
        let order = self.order.clone();
        for organ in order {
            if self.quiescent(organ) {
                match organ {
                    Organ::Chem => chem::jump(&mut st.chem, since, wake),
                    Organ::Clock => clock::jump(&mut st.clock, since, wake),
                    Organ::Soil | Organ::Stage => {}
                }
                tick(&mut self.calls_of(organ)?.jump, 1)?;
            }
        }
        Ok(())
    }

    fn fast(&mut self, st: &mut State, e: i64, day: i64) -> Result<(), Refused> {
        let dormant = st.stage.dormant;
        let mut sun = 0_i64;
        if !dormant {
            tick(&mut self.env, 1)?;
            let minute = modulo(add(add(self.fold.b, e)?, st.core.tz)?, DAY)?;
            let world = &self.fold.life.world;
            let light = ck(civil::daylength(day, world, self.weather.south, self.amp, &self.sine, &mut self.work))?;
            let sun_max = ck(st.core.params.get("sun_max").copied())?;
            sun = ck(civil::sun(minute, light, world, sun_max, &mut self.work))?;
        }
        let order = self.order.clone();
        for organ in order {
            if !matches!(organ, Organ::Chem | Organ::Clock) || (dormant && self.fast_path && self.quiescent(organ)) {
                continue;
            }
            match organ {
                Organ::Chem => ck(chem::fast(&mut st.chem, &self.fold.life.chem, sun, st.bus.moisture, dormant, &mut self.work))?,
                _ => ck(clock::fast(&mut st.clock, sun, dormant, &mut self.work, &mut self.clips))?,
            }
            bump(&mut self.calls_of(organ)?.fast, dormant)?;
        }
        Ok(())
    }

    /// The daily layer at minute `e` for local day `day`; whether the being woke.
    fn daily(&mut self, st: &mut State, e: i64, day: i64) -> Result<bool, Refused> {
        let dormant = st.stage.dormant;
        let since = st.stage.since;
        let order = self.order.clone();
        for organ in order {
            if !matches!(organ, Organ::Soil | Organ::Stage) || (dormant && self.fast_path && self.quiescent(organ)) {
                continue;
            }
            let life = &self.fold.life;
            match organ {
                Organ::Soil => ck(soil::daily(
                    &mut st.soil,
                    &life.soil,
                    &st.core.params,
                    st.bus.dormant,
                    (&self.weather, &life.world),
                    day,
                    &mut self.work,
                    &mut self.draws,
                ))?,
                _ => ck(stage::daily(&mut st.stage, &life.stage, (&life.world, self.weather.south), st.bus.moisture, day, e, &mut self.work))?,
            }
            bump(&mut self.calls_of(organ)?.daily, dormant)?;
        }
        st.core.day = day;
        let woke = dormant && !st.stage.dormant;
        if woke {
            self.jumps(st, since, e)?;
        }
        Ok(woke)
    }

    fn minute(&mut self, st: &mut State, e: i64, here: &[Fact<'_>], first: bool) -> Result<(), Refused> {
        tick(&mut self.work.units, VISIT)?;
        tick(&mut self.visits, 1)?;
        if st.core.pending.as_ref().is_some_and(|pending| pending.effective_from <= e) {
            self.fold.due(&mut st.core, e)?;
        }
        for fact in here {
            if fact.kind == "tz" {
                self.fold.tz_fact(&mut st.core, fact, &mut self.work)?;
            }
        }
        let b = self.fold.b;
        let day = floor_div(add(add(b, e)?, st.core.tz)?, DAY)?;
        if first {
            st.core.day = day;
            st.stage.season = ck(civil::season(day, &self.fold.life.world, self.weather.south))?;
        }
        for fact in here {
            if LAW_KINDS.contains(&fact.kind) {
                self.fold.law_fact(&mut st.core, fact, e, &mut self.work)?;
            }
        }
        let asleep = st.stage.dormant;
        let since = st.stage.since;
        for fact in here {
            if REDUCER_KINDS.contains(&fact.kind) || !self.fold.consume(&mut st.core, fact, &mut self.work)? {
                continue;
            }
            if fact.kind != "act" {
                continue;
            }
            let act = get_text(fact.body, "act")?;
            let count = self.acts.entry(String::from(act)).or_insert(0);
            tick(count, 1)?;
            let order = self.order.clone();
            for organ in order {
                let life = &self.fold.life;
                match organ {
                    Organ::Soil => ck(soil::on_act(&mut st.soil, &life.soil, act, &mut self.work))?,
                    Organ::Stage => ck(stage::on_act(&mut st.stage, act, (&life.world, self.weather.south), day))?,
                    Organ::Chem | Organ::Clock => {}
                }
            }
        }
        if asleep && !st.stage.dormant {
            self.jumps(st, since, e)?;
        }
        if let Some(last) = here.last() {
            st.core.through = Some((last.t, String::from(last.origin), last.oseq));
            st.n = add(st.n, ck(i64::try_from(here.len()).ok())?)?;
        }
        if modulo(add(b, e)?, FAST)? == 0 {
            self.fast(st, e, day)?;
        }
        if day > st.core.day {
            self.daily(st, e, day)?;
        }
        self.publish(st)
    }

    /// Live up to `to` or until the budget runs out; whether `to` was reached.
    fn run(&mut self, st: &mut State, facts: &[Fact<'_>], to: i64, budget: u64, fresh: bool) -> Result<bool, Refused> {
        let b = self.fold.b;
        let mut rest = facts;
        if fresh {
            let count = rest.iter().take_while(|fact| fact.t == 0).count();
            let (here, after) = rest.split_at_checked(count).ok_or_else(broken)?;
            self.minute(st, 0, here, true)?;
            rest = after;
            st.core.at = 0;
            if self.work.units >= budget && st.core.at < to {
                return Ok(false);
            }
        }
        while st.core.at < to {
            let at = st.core.at;
            let next_t = rest.first().map(|fact| fact.t);
            let effective = st.core.pending.as_ref().map(|pending| pending.effective_from);
            if self.fast_path && st.stage.dormant {
                let mut span = to;
                if let Some(next) = next_t {
                    span = span.min(sub(next, 1)?);
                }
                // Asleep, the due evolve's minute is visited (see the module's fast path).
                if let Some(effective) = effective {
                    span = span.min(sub(effective, 1)?);
                }
                if span > at {
                    let z = st.core.tz;
                    let mut m = ck(civil::next_midnight(b, at, z))?;
                    let mut woke = false;
                    while m <= span {
                        let day = floor_div(add(add(b, m)?, z)?, DAY)?;
                        if day > st.core.day {
                            tick(&mut self.work.units, FAST_PATH_DAY)?;
                            tick(&mut self.fast_path_days, 1)?;
                            woke = self.daily(st, m, day)?;
                            self.publish(st)?;
                            if woke {
                                st.core.at = m;
                                break;
                            }
                            if self.work.units >= budget && m < span {
                                st.core.at = m;
                                return Ok(false);
                            }
                        } else {
                            tick(&mut self.fast_path_skipped, 1)?;
                        }
                        m = add(m, DAY)?;
                    }
                    if !woke {
                        st.core.at = span;
                    }
                    continue;
                }
            }
            let mut e = add(add(at, 1)?, modulo(ck(add(add(b, at)?, 1)?.checked_neg())?, FAST)?)?;
            if let Some(next) = next_t {
                e = e.min(next);
            }
            if let Some(effective) = effective {
                if at < effective && effective < e {
                    e = effective;
                }
            }
            if e > to {
                st.core.at = to;
                break;
            }
            let count = rest.iter().take_while(|fact| fact.t == e).count();
            let (here, after) = rest.split_at_checked(count).ok_or_else(broken)?;
            self.minute(st, e, here, false)?;
            rest = after;
            st.core.at = e;
            if self.work.units >= budget && e < to {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn trace(&self) -> Result<Value, Refused> {
        let mut acts = Vec::with_capacity(self.acts.len());
        for (act, n) in &self.acts {
            acts.push((act.clone(), number(*n)?));
        }
        let mut calls = Vec::with_capacity(self.calls.len());
        for (organ, counted) in &self.calls {
            let pair = |counts: &[u64; 2]| -> Result<Value, Refused> {
                let [awake, dormant] = *counts;
                Ok(obj(vec![member("awake", number(awake)?), member("dormant", number(dormant)?)]))
            };
            calls.push((
                String::from(organ.name()),
                obj(vec![member("daily", pair(&counted.daily)?), member("fast", pair(&counted.fast)?), member("jump", number(counted.jump)?)]),
            ));
        }
        Ok(obj(vec![
            member("acts", obj(acts)),
            member("calls", obj(calls)),
            member("clock_clip", number(self.clips)?),
            member("draws", number(self.draws)?),
            member("env", number(self.env)?),
            member("facts", number(self.fold.facts)?),
            member("noted", number(self.fold.noted)?),
            member("fast_path_days", number(self.fast_path_days)?),
            member("fast_path_skipped", number(self.fast_path_skipped)?),
            member("visits", number(self.visits)?),
        ]))
    }
}

/// The being's genome, founded under the genesis law from its seed, compiled; the organs' constants and the work.
fn founded(genesis_life: &Life, seed: &[u8; 32]) -> Result<(String, ChemK, ClockK, u64), Refused> {
    let (law, lawview) = lawdata::genome_law(Some(&genesis_life.name))?;
    let alleles = lawdata::genome_pool(&law, &lawview)?;
    let digest = ocj::hex(&laws::digest(&law)?);
    let (data, _chosen, found_work) = genome::found(seed, &lawview, &alleles)?;
    let (tables, compile_work) = organ_compile::compile_genome(&data, &lawview, &digest)?;
    let (chem_k, clock_k) = lawdata::consts(&tables, genesis_life)?;
    Ok((genome::sha256_hex(&data), chem_k, clock_k, ck(found_work.checked_add(compile_work))?))
}

fn initial_core(genesis: &Genesis<'_>, life: &Life) -> Result<Core, Refused> {
    Ok(Core {
        at: 0,
        budget_counts: BTreeMap::new(),
        budget_day: 0,
        day: floor_div(add(genesis.b, genesis.birth_tz)?, DAY)?,
        law: Law { name: life.name.clone(), sha256: life.digest.clone(), v: life.version },
        params: genesis.params.clone(),
        pending: None,
        pinned: false,
        seen: vec![life.version],
        through: None,
        tz: genesis.birth_tz,
    })
}

/// The state before minute 0 is processed, and the work the genome operations reported.
fn initial(genesis: &Genesis<'_>, life: &Life) -> Result<(State, u64), Refused> {
    let (genome_sha, chem_k, clock_k, work) = founded(life, &genesis.weather.seed)?;
    let state = State {
        core: initial_core(genesis, life)?,
        being: String::from(genesis.being),
        bus: Bus::default(),
        genome: genome_sha,
        n: 0,
        chem: chem::init(&life.chem, chem_k),
        clock: clock::init(life.clock.init, clock_k),
        soil: soil::init(&life.soil),
        stage: stage::init(),
    };
    Ok((state, work))
}

/// The environment at the state's minute, at no cost.
fn env(st: &State, reducer: &Reducer) -> Result<Value, Refused> {
    let world = &reducer.fold.life.world;
    let south = reducer.weather.south;
    let mut work = Work::default();
    let (day, minute) = ck(civil::local(reducer.fold.b, st.core.at, st.core.tz))?;
    let light = ck(civil::daylength(day, world, south, reducer.amp, &reducer.sine, &mut work))?;
    let (y, m, d) = ck(civil::civil_from_days(day))?;
    let sun_max = ck(st.core.params.get("sun_max").copied())?;
    Ok(obj(vec![
        member("civil", Value::Arr(vec![Value::Int(y), Value::Int(m), Value::Int(d)])),
        member("day", Value::Int(day)),
        member("daylength", Value::Int(light)),
        member("minute", Value::Int(minute)),
        member("season", Value::Int(ck(civil::season(day, world, south))?)),
        member("sun", Value::Int(ck(civil::sun(minute, light, world, sun_max, &mut work))?)),
    ]))
}

/// The `advance` operation: a life folded up to `to` or its budget.
pub fn op_advance(request: &Value, size: usize) -> Answer {
    fields(request, &["budget", "facts", "genesis", "op", "state", "to", "v"], &["probe"])?;
    let (genesis, life) = genesis_of(request)?;
    let to = to_of(request)?;
    let budget = match int(request.get("budget")) {
        Some(budget) if (1..=MAX_INT).contains(&budget) => ck(u64::try_from(budget).ok())?,
        _ => return Err(bad("bad_request", "budget")),
    };
    let (order, fast_path, tracing) = probe_of(request, &life)?;
    let resumed_from = resumed(request, "state", &STATE_KEYS, &genesis, &life, to)?;
    let (state, in_force) = match resumed_from {
        Some((value, in_force)) => (Some(State::read(value)?), in_force),
        None => (None, life.clone()),
    };
    let facts = facts_of(request, &genesis, &life.table, state.as_ref().map(|st| &st.core), to, false)?;
    let mut overhead = ck(u64::try_from(size).ok().and_then(|n| n.checked_add(63)).map(|n| n / 64))?;
    let fresh = state.is_none();
    let mut state = match state {
        Some(state) => state,
        None => {
            let (state, genome_work) = initial(&genesis, &life)?;
            overhead = ck(overhead.checked_add(genome_work))?;
            state
        }
    };
    let mut reducer = Reducer::new(in_force, &genesis, order, fast_path)?;
    if fresh {
        reducer.publish(&mut state)?;
    }
    let done = reducer.run(&mut state, &facts, to, budget, fresh)?;
    let environment = env(&state, &reducer)?;
    let state_value = state.value();
    let hash = ocj::hex(&Sha256::digest(ocj::emit(&state_value)?));
    let mut members = vec![
        member("alarm", number(reducer.work.alarm)?),
        member("at", Value::Int(state.core.at)),
        member("done", Value::Bool(done)),
        member("env", environment),
        member("hash", Value::Str(hash)),
        member("notes", capped_notes(&reducer.fold.notes, state.core.at)),
        member("overhead", number(overhead)?),
        member("provisional", Value::Bool(genesis.provisional)),
        member("state", state_value),
        member("work", number(reducer.work.units)?),
    ];
    if tracing {
        members.push(member("trace", reducer.trace()?));
    }
    Ok(obj(members))
}

// ---------------------------------------------------------------------------
// timeline
// ---------------------------------------------------------------------------

/// The law kinds folded alone, and the daily firing minutes computed per stretch of constant offset.
struct Timeline {
    fold: Fold,
    work: Work,
    /// The minutes after which firing minutes are listed, and how many may be.
    window: Option<(i64, usize)>,
    midnights: Vec<Value>,
    cursor: i64,
}

impl Timeline {
    /// A daily firing at minute `m`: the mark moves, and the minute is listed when it is in the window.
    fn fire(&mut self, st: &mut Core, m: i64, day: i64) -> Result<(), Refused> {
        st.day = day;
        if let Some((low, most)) = self.window {
            if m > low {
                if self.midnights.len() >= most {
                    return Err(bad("limit", "items"));
                }
                let (y, mo, d) = ck(civil::civil_from_days(day))?;
                self.midnights.push(Value::Arr(vec![
                    Value::Int(m),
                    Value::Int(day),
                    Value::Arr(vec![Value::Int(y), Value::Int(mo), Value::Int(d)]),
                ]));
                tick(&mut self.work.units, 1)?;
            }
        }
        Ok(())
    }

    /// Every local midnight in `(cursor, end]` under the offset in force whose day is above the mark.
    fn rise(&mut self, st: &mut Core, end: i64) -> Result<(), Refused> {
        if end <= self.cursor {
            return Ok(());
        }
        let b = i128::from(self.fold.b);
        let z = i128::from(st.tz);
        let day_len = i128::from(DAY);
        let wide_end = i128::from(end);
        let mut m = i128::from(ck(civil::next_midnight(self.fold.b, self.cursor, st.tz))?);
        let mut day = ck(b.checked_add(m).and_then(|v| v.checked_add(z)).and_then(|v| v.checked_div_euclid(day_len)))?;
        let mark = i128::from(st.day);
        if day <= mark {
            let skip = ck(mark.checked_sub(day).and_then(|v| v.checked_add(1)))?;
            m = ck(skip.checked_mul(day_len).and_then(|v| v.checked_add(m)))?;
            day = ck(mark.checked_add(1))?;
        }
        let last = ck(b.checked_add(wide_end).and_then(|v| v.checked_add(z)).and_then(|v| v.checked_div_euclid(day_len)))?;
        if m <= wide_end {
            match self.window {
                None => st.day = ck(i64::try_from(last).ok())?,
                Some((low, most)) => {
                    let low = i128::from(low);
                    if m <= low {
                        let skip = ck(low.checked_sub(m).and_then(|v| v.checked_div_euclid(day_len)).and_then(|v| v.checked_add(1)))?;
                        m = ck(skip.checked_mul(day_len).and_then(|v| v.checked_add(m)))?;
                        day = ck(day.checked_add(skip))?;
                    }
                    if m <= wide_end {
                        let listed = ck(i128::try_from(self.midnights.len()).ok())?;
                        let coming = ck(wide_end.checked_sub(m).and_then(|v| v.checked_div_euclid(day_len)).and_then(|v| v.checked_add(1)))?;
                        if ck(listed.checked_add(coming))? > ck(i128::try_from(most).ok())? {
                            return Err(bad("limit", "items"));
                        }
                    }
                    while m <= wide_end {
                        self.fire(st, ck(i64::try_from(m).ok())?, ck(i64::try_from(day).ok())?)?;
                        m = ck(m.checked_add(day_len))?;
                        day = ck(day.checked_add(1))?;
                    }
                    st.day = st.day.max(ck(i64::try_from(last).ok())?);
                }
            }
        }
        self.cursor = end;
        Ok(())
    }

    fn through(st: &mut Core, here: &[Fact<'_>]) {
        if let Some(last) = here.last() {
            st.through = Some((last.t, String::from(last.origin), last.oseq));
        }
    }

    fn run(&mut self, st: &mut Core, facts: &[Fact<'_>], to: i64, fresh: bool) -> Result<(), Refused> {
        let mut rest = facts;
        if fresh {
            let count = rest.iter().take_while(|fact| fact.t == 0).count();
            let (here, after) = rest.split_at_checked(count).ok_or_else(broken)?;
            for fact in here {
                if fact.kind == "tz" {
                    self.fold.tz_fact(st, fact, &mut self.work)?;
                }
            }
            st.day = floor_div(add(self.fold.b, st.tz)?, DAY)?;
            for fact in here {
                if LAW_KINDS.contains(&fact.kind) {
                    self.fold.law_fact(st, fact, 0, &mut self.work)?;
                }
            }
            Timeline::through(st, here);
            rest = after;
        }
        while let Some(first) = rest.first() {
            let e = first.t;
            let count = rest.iter().take_while(|fact| fact.t == e).count();
            let (here, after) = rest.split_at_checked(count).ok_or_else(broken)?;
            if let Some(effective) = st.pending.as_ref().map(|pending| pending.effective_from) {
                if effective <= e {
                    self.fold.due(st, effective)?;
                }
            }
            self.rise(st, sub(e, 1)?)?;
            let mut moved = false;
            for fact in here {
                if fact.kind == "tz" {
                    self.fold.tz_fact(st, fact, &mut self.work)?;
                    moved = true;
                }
            }
            if moved {
                let day = floor_div(add(add(self.fold.b, e)?, st.tz)?, DAY)?;
                if day > st.day {
                    self.fire(st, e, day)?;
                }
                self.cursor = e;
            }
            for fact in here {
                if LAW_KINDS.contains(&fact.kind) {
                    self.fold.law_fact(st, fact, e, &mut self.work)?;
                }
            }
            Timeline::through(st, here);
            rest = after;
        }
        if let Some(effective) = st.pending.as_ref().map(|pending| pending.effective_from) {
            if effective <= to {
                self.fold.due(st, effective)?;
            }
        }
        self.rise(st, to)?;
        st.at = to;
        Ok(())
    }
}

/// The `timeline` operation: the law kinds folded up to `to`, with the daily firing minutes.
pub fn op_timeline(request: &Value) -> Answer {
    fields(request, &["facts", "from", "genesis", "op", "to", "v"], &["midnights_after"])?;
    let (genesis, life) = genesis_of(request)?;
    let to = to_of(request)?;
    let after = match request.get("midnights_after") {
        None => None,
        Some(value) => match int(Some(value)) {
            Some(after) if (0..=MAX_INT).contains(&after) => Some(after),
            _ => return Err(bad("bad_request", "midnights_after")),
        },
    };
    let resumed_from = resumed(request, "from", &TSTATE_KEYS, &genesis, &life, to)?;
    let (core, in_force) = match resumed_from {
        Some((value, in_force)) => (Some(Core::read(value)?), in_force),
        None => (None, life.clone()),
    };
    let facts = facts_of(request, &genesis, &life.table, core.as_ref(), to, true)?;
    let fresh = core.is_none();
    let mut core = match core {
        Some(core) => core,
        None => initial_core(&genesis, &life)?,
    };
    let window = after.map(|after| (after.max(core.at), crate::protocol::LIMIT_ITEMS));
    let cursor = core.at;
    let mut fold = Timeline {
        fold: Fold { life: in_force, b: genesis.b, notes: Vec::new(), facts: 0, noted: 0 },
        work: Work::default(),
        window,
        midnights: Vec::new(),
        cursor,
    };
    fold.run(&mut core, &facts, to, fresh)?;
    let mut members = vec![
        member("midnight", Value::Int(ck(civil::next_midnight(genesis.b, to, core.tz))?)),
        member("notes", capped_notes(&fold.fold.notes, to)),
        member("state", obj(core.members())),
        member("work", number(fold.work.units)?),
    ];
    if window.is_some() {
        members.push(member("midnights", Value::Arr(fold.midnights)));
    }
    Ok(obj(members))
}
