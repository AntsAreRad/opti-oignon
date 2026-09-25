#![deny(clippy::indexing_slicing, clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::todo, clippy::unimplemented)]
//! The law data a life runs on, the twin of `opti_oignon/allium/ref/lawdata.py`.
//!
//! A carried law is one this engine embeds (`laws::LAWS`); its identity is
//! its name and the digest of its canonical bytes. Besides its genome
//! section it carries the sections its life runs on: the organ code
//! revision, the organs and the channel each writes, the organs never
//! called while the being is dormant, the world (the year, the seasons, the
//! provisional daylength, the rain), each organ's constants, the ranges of
//! the params a genesis freezes, and the unit table with the per-day caps
//! of the budget-exempt kinds.
//!
//! `life(name)` checks a law's journal pin (the kinds table it names, by
//! digest, and a daily budget for every budgeted kind), then its life
//! sections, as the reference does and in its order; a defect refuses
//! `unknown_law` with the detail `journal`, `life law` or, for an organ code
//! this engine does not implement, `code`. Only the carried laws reach this
//! engine, so these refusals are a second net: the reference, with laws
//! injected, is where they are exercised. The kinds table's own soundness is
//! the reference's alone; the handshake proves the embedded table by its
//! digest. Nothing is cached: the files are parsed at every call.
//!
//! `consts` derives, from a genome's compiled tables, the constants the
//! organs keep in the state: the clock's production, decay, repression and
//! light gain, and the reserve's reaction rates. The genome's law and founder
//! pool are read here too, for the genome operations and for a life's first
//! minute alike.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::laws;
use crate::ocj::{self, Refused, Value, MAX_INT};
use crate::organs::genome::{self, Allele, View};
use crate::protocol::{bad, int, text};

/// The organ code revisions this engine implements.
pub const CODES: [&str; 1] = ["seed_1"];
/// The organs of `seed_1`, in the order of their names.
pub const CODE_ORGANS: [&str; 4] = ["chem", "clock", "soil", "stage"];
/// The channel each organ of `seed_1` writes.
const CODE_BUS: [(&str, &str); 4] = [("circadian", "clock"), ("dormant", "stage"), ("metab", "chem"), ("moisture", "soil")];
/// The trunk kinds outside the membrane's daily budgets; all but the genesis are capped per day of life instead.
pub const EXEMPT: [&str; 5] = ["clock", "genesis", "owner", "resumed", "tz"];
const BUDGET_MAX: i64 = 4096;
const LIFE_KEYS: [&str; 8] = ["bus", "code", "constants", "organs", "params", "quiescent_in_dormancy", "work", "world"];
const BANDS: [&str; 3] = ["long", "medium", "short"];
const SEASONS: usize = 4;
const CIVIL_YEAR: i64 = 365;
const CHEM_KEYS: [&str; 14] = [
    "core",
    "enzymes",
    "fructan0",
    "fructan_max",
    "hyd_scale",
    "k_w",
    "metab_shift",
    "ps_scale",
    "resp_scale",
    "sugar0",
    "sugar_max",
    "syn_scale",
    "theta_h",
    "theta_s",
];
/// Each reaction of the reserve, the scale its summed turnover is taken at, and its rate's name.
const ROLES: [(&str, &str, &str); 4] = [
    ("hydrolysis", "hyd_scale", "hy"),
    ("photosynthesis", "ps_scale", "ps"),
    ("respiration", "resp_scale", "r"),
    ("synthesis", "syn_scale", "sy"),
];
const FRUCTAN_LIMIT: i64 = 1 << 30;
const STAGE_KEYS: [&str; 6] = ["d_enter", "rest_dry", "rest_max", "rest_winter", "theta_dry", "theta_wet"];
const REST_LIMIT: i64 = 100_000;
/// The tables a law's journal pin may name.
pub const JOURNAL_TABLES: [&str; 1] = ["journal_v1"];
const ONE: i64 = 1 << 16;
const CMAX: i64 = 8 << 16;

// ---------------------------------------------------------------------------
// Reading parsed values
// ---------------------------------------------------------------------------

pub(crate) fn int_in(value: Option<&Value>, low: i64, high: i64) -> bool {
    matches!(int(value), Some(n) if (low..=high).contains(&n))
}

/// An object whose keys are exactly `names`.
pub(crate) fn exactly(value: Option<&Value>, names: &[&str]) -> bool {
    match value {
        Some(Value::Obj(members)) => {
            members.iter().all(|(key, _)| names.contains(&key.as_str()))
                && names.iter().all(|name| members.iter().any(|(key, _)| key == name))
        }
        _ => false,
    }
}

/// A list of strings, none twice.
pub(crate) fn distinct_names(value: Option<&Value>) -> Option<Vec<&str>> {
    let Some(Value::Arr(items)) = value else {
        return None;
    };
    let mut names: Vec<&str> = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Value::Str(name) if !names.contains(&name.as_str()) => names.push(name.as_str()),
            _ => return None,
        }
    }
    Some(names)
}

/// Printable ASCII, 1..=`most` characters, no space at either end.
fn is_text(value: Option<&Value>, most: usize) -> bool {
    match text(value) {
        Some(given) => {
            (1..=most).contains(&given.len())
                && given.bytes().all(|b| (0x20..=0x7E).contains(&b))
                && !given.starts_with(' ')
                && !given.ends_with(' ')
        }
        None => false,
    }
}

fn is_hex(value: Option<&Value>, length: usize) -> bool {
    matches!(text(value), Some(given) if ocj::is_hex(given, length))
}

fn sorted(mut names: Vec<&str>) -> Vec<&str> {
    names.sort_unstable();
    names
}

// ---------------------------------------------------------------------------
// Carried files
// ---------------------------------------------------------------------------

/// The parsed carried law `name` and its digest; `unknown_law "law"` for a law this engine does not carry.
pub fn law_file(name: &str) -> Result<(Value, String), Refused> {
    let file = laws::law(name).ok_or_else(|| bad("unknown_law", "law"))?;
    let law = laws::parse_file(file)?;
    let digest = ocj::hex(&laws::digest(&law)?);
    Ok((law, digest))
}

/// A carried law and its genome codec view; `unknown_law "genome law"` when its genome is unsound.
pub fn genome_law(name: Option<&str>) -> Result<(Value, View), Refused> {
    let file = match name.and_then(laws::law) {
        Some(file) => file,
        None => return Err(bad("unknown_law", "law")),
    };
    let law = laws::parse_file(file)?;
    if !genome::validate_law(&law).is_empty() {
        return Err(bad("unknown_law", "genome law"));
    }
    let lawview = genome::view(&law).ok_or_else(|| bad("unknown_law", "genome law"))?;
    Ok((law, lawview))
}

/// The founder alleles a genome law pins, checked; refused by name when the pin or the pool is unsound.
pub fn genome_pool(law: &Value, lawview: &View) -> Result<BTreeMap<i64, Vec<Allele>>, Refused> {
    let pin = match law.get("founders") {
        Some(pin @ Value::Obj(_)) => pin,
        _ => return Err(bad("unknown_law", "founders digest")),
    };
    let file = match text(pin.get("name")).and_then(laws::founders) {
        Some(file) => file,
        None => return Err(bad("unknown_law", "founders digest")),
    };
    let pool = laws::parse_file(file)?;
    let digest = ocj::hex(&laws::digest(&pool)?);
    if text(pin.get("sha256")) != Some(digest.as_str()) {
        return Err(bad("unknown_law", "founders digest"));
    }
    if !genome::validate_pool(law, lawview, &pool).is_empty() {
        return Err(bad("unknown_law", "founders"));
    }
    genome::pool_alleles(&pool).ok_or_else(|| bad("unknown_law", "founders"))
}

/// The frozen Q15 sine table.
pub fn sine() -> Result<Vec<i128>, Refused> {
    laws::sine()
}

// ---------------------------------------------------------------------------
// The journal pin
// ---------------------------------------------------------------------------

fn journal_refused() -> Refused {
    bad("unknown_law", "journal")
}

/// The kinds table a law pins and the daily budgets of the budgeted kinds; refused `journal`.
fn journal_pin(law: &Value) -> Result<(Value, BTreeMap<String, i64>), Refused> {
    let pin = law.get("journal");
    if !exactly(pin, &["budgets", "table"]) {
        return Err(journal_refused());
    }
    let pin = pin.ok_or_else(journal_refused)?;
    let named = pin.get("table");
    if !exactly(named, &["name", "sha256"]) {
        return Err(journal_refused());
    }
    let named = named.ok_or_else(journal_refused)?;
    let name = match text(named.get("name")) {
        Some(name) if JOURNAL_TABLES.contains(&name) => name,
        _ => return Err(journal_refused()),
    };
    let file = laws::table(name).ok_or_else(journal_refused)?;
    let table = laws::parse_file(file).map_err(|_| journal_refused())?;
    let digest = ocj::hex(&laws::digest(&table).map_err(|_| journal_refused())?);
    if text(named.get("sha256")) != Some(digest.as_str()) {
        return Err(journal_refused());
    }
    // The table's own soundness is the reference's: the handshake proves the embedded table by its digest.
    let Some(Value::Obj(kinds)) = table.get("kinds") else {
        return Err(journal_refused());
    };
    let mut budgeted: Vec<&str> = Vec::new();
    for (kind, entry) in kinds {
        let scope = text(entry.get("scope")).ok_or_else(journal_refused)?;
        if scope == "trunk" && !EXEMPT.contains(&kind.as_str()) {
            budgeted.push(kind.as_str());
        }
    }
    budgeted.sort_unstable();
    let budgets = match pin.get("budgets") {
        Some(budgets @ Value::Obj(_)) => budgets,
        _ => return Err(journal_refused()),
    };
    let mut out = BTreeMap::new();
    for kind in &budgeted {
        match int(budgets.get(kind)) {
            Some(n) if (1..=BUDGET_MAX).contains(&n) => {
                out.insert(String::from(*kind), n);
            }
            _ => return Err(journal_refused()),
        }
    }
    for kind in budgets.keys() {
        if !budgeted.contains(&kind) {
            return Err(journal_refused());
        }
    }
    Ok((table, out))
}

/// The params a genesis freezes and an `evolve` carries, as the kinds table bounds them.
pub fn params_schema(table: &Value) -> Option<&Value> {
    table.get("kinds")?.get("genesis")?.get("body")?.get("laws")?.get("fields")?.get("params")?.get("fields")
}

// ---------------------------------------------------------------------------
// The life sections
// ---------------------------------------------------------------------------

/// The kind of each locus of the law's genome, by name; nothing for a genome section it cannot read.
fn loci_kinds(law: &Value) -> BTreeMap<&str, &str> {
    let mut kinds = BTreeMap::new();
    if let Some(Value::Arr(loci)) = law.get("genome").and_then(|section| section.get("loci")) {
        for entry in loci {
            if let (Some(name), Some(kind)) = (text(entry.get("name")), text(entry.get("kind"))) {
                kinds.insert(name, kind);
            }
        }
    }
    kinds
}

fn world_ok(world: Option<&Value>) -> bool {
    if !exactly(world, &["daylength", "rain", "season_shift", "south_shift", "year"]) {
        return false;
    }
    let Some(world) = world else {
        return false;
    };
    let year = world.get("year");
    let length = if exactly(year, &["kind"]) && text(year.and_then(|y| y.get("kind"))) == Some("civil") {
        CIVIL_YEAR
    } else if exactly(year, &["days", "kind"])
        && text(year.and_then(|y| y.get("kind"))) == Some("fixed")
        && int_in(year.and_then(|y| y.get("days")), 4, 1000)
    {
        int(year.and_then(|y| y.get("days"))).unwrap_or(0)
    } else {
        return false;
    };
    let light = world.get("daylength");
    if !exactly(light, &["amp", "equinox", "mean", "ramp"]) {
        return false;
    }
    let Some(light) = light else {
        return false;
    };
    let top = length.saturating_sub(1);
    for value in [light.get("equinox"), world.get("season_shift"), world.get("south_shift")] {
        if !int_in(value, 0, top) {
            return false;
        }
    }
    let Some(mean) = int(light.get("mean")) else {
        return false;
    };
    if !int_in(light.get("ramp"), 1, 120) || !exactly(light.get("amp"), &BANDS) {
        return false;
    }
    for band in BANDS {
        let Some(amp) = int(light.get("amp").and_then(|amp| amp.get(band))) else {
            return false;
        };
        if !(0..=720).contains(&amp) || amp > mean || mean > 1440_i64.saturating_sub(amp) {
            return false;
        }
    }
    let rain = world.get("rain");
    if !exactly(rain, &["max", "p_wet"]) {
        return false;
    }
    for key in ["max", "p_wet"] {
        let Some(Value::Arr(values)) = rain.and_then(|rain| rain.get(key)) else {
            return false;
        };
        if values.len() != SEASONS || !values.iter().all(|value| int_in(Some(value), 0, ONE)) {
            return false;
        }
    }
    true
}

fn chem_ok(chem: Option<&Value>, loci: &BTreeMap<&str, &str>) -> bool {
    if !exactly(chem, &CHEM_KEYS) {
        return false;
    }
    let Some(chem) = chem else {
        return false;
    };
    let mut values: BTreeMap<&str, i64> = BTreeMap::new();
    for key in CHEM_KEYS {
        if key == "enzymes" {
            continue;
        }
        match int(chem.get(key)) {
            Some(n) => {
                values.insert(key, n);
            }
            None => return false,
        }
    }
    let get = |key: &str| values.get(key).copied().unwrap_or(0);
    let (core, fructan0, fructan_max) = (get("core"), get("fructan0"), get("fructan_max"));
    if !(0 <= core && core <= fructan0 && fructan0 <= fructan_max && fructan_max <= FRUCTAN_LIMIT) {
        return false;
    }
    let (sugar0, sugar_max) = (get("sugar0"), get("sugar_max"));
    if !(0 <= sugar0 && sugar0 <= sugar_max && sugar_max <= CMAX) {
        return false;
    }
    if !(0..=sugar_max).contains(&get("theta_h")) || !(0..=sugar_max).contains(&get("theta_s")) {
        return false;
    }
    if !(1..=ONE).contains(&get("k_w")) || !(0..=15).contains(&get("metab_shift")) {
        return false;
    }
    let enzymes = chem.get("enzymes");
    let roles: Vec<&str> = ROLES.iter().map(|(role, _, _)| *role).collect();
    if !exactly(enzymes, &roles) {
        return false;
    }
    for (role, scale_key, _) in ROLES {
        let Some(names) = distinct_names(enzymes.and_then(|e| e.get(role))) else {
            return false;
        };
        let scale = get(scale_key);
        if !(1..=4).contains(&names.len()) || scale < 0 {
            return false;
        }
        if !names.iter().all(|name| loci.get(name) == Some(&"enz")) {
            return false;
        }
        let count = i128::try_from(names.len()).unwrap_or(i128::MAX);
        let turnover = count.saturating_mul(65535).saturating_mul(i128::from(scale)).checked_shr(16).unwrap_or(i128::MAX);
        if turnover > i128::from(ONE) {
            return false;
        }
    }
    true
}

fn clock_ok(clock: Option<&Value>, loci: &BTreeMap<&str, &str>) -> bool {
    if !exactly(clock, &["genes", "init", "light"]) {
        return false;
    }
    let Some(clock) = clock else {
        return false;
    };
    let Some(genes) = distinct_names(clock.get("genes")) else {
        return false;
    };
    if genes.len() != 3 || !genes.iter().all(|name| loci.get(name) == Some(&"tf")) {
        return false;
    }
    match clock.get("init") {
        Some(Value::Arr(init)) if init.len() == 3 && init.iter().all(|value| int_in(Some(value), 0, CMAX)) => {}
        _ => return false,
    }
    matches!(text(clock.get("light")), Some(light) if loci.get(light) == Some(&"rec"))
}

fn soil_ok(soil: Option<&Value>) -> bool {
    if !exactly(soil, &["dose", "m0", "m_max"]) || !int_in(soil.and_then(|s| s.get("m_max")), 1, ONE) {
        return false;
    }
    let m_max = int(soil.and_then(|s| s.get("m_max"))).unwrap_or(0);
    int_in(soil.and_then(|s| s.get("m0")), 0, m_max) && int_in(soil.and_then(|s| s.get("dose")), 0, m_max)
}

fn stage_ok(stage: Option<&Value>, m_max: i64) -> bool {
    if !exactly(stage, &STAGE_KEYS) || !int_in(stage.and_then(|s| s.get("rest_max")), 1, REST_LIMIT) {
        return false;
    }
    let field = |key: &str| stage.and_then(|s| s.get(key));
    let rest_max = int(field("rest_max")).unwrap_or(0);
    for key in ["d_enter", "rest_dry", "rest_winter"] {
        if !int_in(field(key), 1, rest_max) {
            return false;
        }
    }
    let (Some(dry), Some(wet)) = (int(field("theta_dry")), int(field("theta_wet"))) else {
        return false;
    };
    0 <= dry && dry <= wet && wet <= m_max
}

fn params_ok(params: Option<&Value>, table: &Value) -> bool {
    let Some(Value::Obj(bounds)) = params_schema(table) else {
        return false;
    };
    let names: Vec<&str> = bounds.iter().map(|(name, _)| name.as_str()).collect();
    if !exactly(params, &names) {
        return false;
    }
    let Some(Value::Obj(given)) = params else {
        return false;
    };
    for (name, spec) in given {
        if !exactly(Some(spec), &["default", "hi", "lo"]) {
            return false;
        }
        let (Some(lo), Some(default), Some(hi)) = (int(spec.get("lo")), int(spec.get("default")), int(spec.get("hi"))) else {
            return false;
        };
        let bound = bounds.iter().find(|(key, _)| key == name).map(|(_, bound)| bound);
        let (Some(low), Some(high)) = (int(bound.and_then(|b| b.get("lo"))), int(bound.and_then(|b| b.get("hi")))) else {
            return false;
        };
        if !(low <= lo && lo <= default && default <= hi && hi <= high) {
            return false;
        }
    }
    true
}

/// Whether `value` has the unit table's exact shape, with non-negative integer leaves.
fn units_ok(units: Option<&Value>) -> bool {
    let leaf = |value: Option<&Value>| int_in(value, 0, MAX_INT);
    let layer = |value: Option<&Value>| exactly(value, &["awake", "dormant"]) && leaf(value.and_then(|v| v.get("awake"))) && leaf(value.and_then(|v| v.get("dormant")));
    if !exactly(units, &["act", "draw", "env", "fact", "fast_path_day", "organs", "visit"]) {
        return false;
    }
    let get = |key: &str| units.and_then(|u| u.get(key));
    let act = get("act");
    let water = act.and_then(|a| a.get("water"));
    if !exactly(act, &["water"]) || !exactly(water, &["soil"]) || !leaf(water.and_then(|w| w.get("soil"))) {
        return false;
    }
    for key in ["draw", "env", "fact", "fast_path_day", "visit"] {
        if !leaf(get(key)) {
            return false;
        }
    }
    let organs = get("organs");
    if !exactly(organs, &CODE_ORGANS) {
        return false;
    }
    for (organ, key) in [("chem", "fast"), ("clock", "fast"), ("soil", "daily"), ("stage", "daily")] {
        let entry = organs.and_then(|o| o.get(organ));
        if !exactly(entry, &[key]) || !layer(entry.and_then(|e| e.get(key))) {
            return false;
        }
    }
    true
}

fn work_ok(work: Option<&Value>, table: &Value) -> bool {
    if !exactly(work, &["caps", "ceilings", "proof", "units"]) {
        return false;
    }
    let get = |key: &str| work.and_then(|w| w.get(key));
    let kinds = table.get("kinds");
    let capped: Vec<&str> = EXEMPT
        .iter()
        .copied()
        .filter(|kind| *kind != "genesis" && text(kinds.and_then(|k| k.get(kind)).and_then(|e| e.get("scope"))) == Some("trunk"))
        .collect();
    let caps = get("caps");
    if !exactly(caps, &capped) {
        return false;
    }
    if !capped.iter().all(|kind| int_in(caps.and_then(|c| c.get(kind)), 1, MAX_INT)) {
        return false;
    }
    if !units_ok(get("units")) {
        return false;
    }
    let (ceilings, proof) = (get("ceilings"), get("proof"));
    if !exactly(ceilings, &["awake_day", "dormant_day"]) || !exactly(proof, &["awake_day", "dormant_day"]) {
        return false;
    }
    ["awake_day", "dormant_day"]
        .iter()
        .all(|key| int(ceilings.and_then(|c| c.get(key))).is_some() && text(proof.and_then(|p| p.get(key))).is_some())
}

/// What is wrong with a law's life sections: `None`, `"code"` or `"life law"`.
fn life_defect(law: &Value, table: &Value) -> Option<&'static str> {
    if LIFE_KEYS.iter().any(|key| law.get(key).is_none()) {
        return Some("life law");
    }
    if !matches!(text(law.get("code")), Some(code) if CODES.contains(&code)) {
        return Some("code");
    }
    if !int_in(law.get("version"), 0, 65535) || !matches!(law.get("provisional"), Some(Value::Bool(_))) {
        return Some("life law");
    }
    let Some(organs) = distinct_names(law.get("organs")) else {
        return Some("life law");
    };
    if sorted(organs.clone()) != CODE_ORGANS.to_vec() {
        return Some("life law");
    }
    let Some(quiescent) = distinct_names(law.get("quiescent_in_dormancy")) else {
        return Some("life law");
    };
    if !quiescent.iter().all(|name| organs.contains(name)) {
        return Some("life law");
    }
    let bus = law.get("bus");
    let channels: Vec<&str> = CODE_BUS.iter().map(|(channel, _)| *channel).collect();
    if !exactly(bus, &channels)
        || !CODE_BUS.iter().all(|(channel, writer)| text(bus.and_then(|b| b.get(channel))) == Some(*writer))
    {
        return Some("life law");
    }
    if !world_ok(law.get("world")) {
        return Some("life law");
    }
    let constants = law.get("constants");
    let loci = loci_kinds(law);
    if !exactly(constants, &CODE_ORGANS) {
        return Some("life law");
    }
    let part = |key: &str| constants.and_then(|c| c.get(key));
    if !(chem_ok(part("chem"), &loci) && clock_ok(part("clock"), &loci) && soil_ok(part("soil"))) {
        return Some("life law");
    }
    let m_max = int(part("soil").and_then(|s| s.get("m_max"))).unwrap_or(0);
    if !stage_ok(part("stage"), m_max) {
        return Some("life law");
    }
    if !params_ok(law.get("params"), table) || !work_ok(law.get("work"), table) {
        return Some("life law");
    }
    if let Some(succeeds) = law.get("succeeds") {
        if !exactly(Some(succeeds), &["migrate", "name", "sha256"])
            || text(succeeds.get("migrate")) != Some("identity")
            || !is_text(succeeds.get("name"), 32)
            || !is_hex(succeeds.get("sha256"), 64)
        {
            return Some("life law");
        }
    }
    None
}

// ---------------------------------------------------------------------------
// A sound life, in the form the reducer reads
// ---------------------------------------------------------------------------

/// An organ of `seed_1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Organ {
    Chem,
    Clock,
    Soil,
    Stage,
}

impl Organ {
    pub fn name(self) -> &'static str {
        match self {
            Organ::Chem => "chem",
            Organ::Clock => "clock",
            Organ::Soil => "soil",
            Organ::Stage => "stage",
        }
    }

    pub fn from_name(name: &str) -> Option<Organ> {
        match name {
            "chem" => Some(Organ::Chem),
            "clock" => Some(Organ::Clock),
            "soil" => Some(Organ::Soil),
            "stage" => Some(Organ::Stage),
            _ => None,
        }
    }
}

/// The law's world: the year, the seasons, the provisional daylength and the rain.
#[derive(Clone, Debug)]
pub struct World {
    /// The civil year, or a fixed year of `days` days.
    pub civil: bool,
    pub days: i64,
    pub equinox: i64,
    pub mean: i64,
    pub ramp: i64,
    pub amp_long: i64,
    pub amp_medium: i64,
    pub amp_short: i64,
    pub rain_max: [i64; 4],
    pub p_wet: [i64; 4],
    pub season_shift: i64,
    pub south_shift: i64,
}

impl World {
    /// The amplitude of the daylength for a being of `band`.
    pub fn amp(&self, band: &str) -> Option<i64> {
        match band {
            "long" => Some(self.amp_long),
            "medium" => Some(self.amp_medium),
            "short" => Some(self.amp_short),
            _ => None,
        }
    }
}

/// The reserve's constants.
#[derive(Clone, Debug)]
pub struct ChemConsts {
    pub core: i64,
    pub fructan0: i64,
    pub fructan_max: i64,
    pub k_w: i64,
    pub metab_shift: i64,
    pub sugar0: i64,
    pub sugar_max: i64,
    pub theta_h: i64,
    pub theta_s: i64,
    /// Per role, in the order of `ROLES`: its loci, and the scale its turnover is taken at.
    pub roles: Vec<(Vec<String>, i64)>,
}

/// The clock's constants: the genes it reads, its levels at a being's first minute, its light receptor.
#[derive(Clone, Debug)]
pub struct ClockConsts {
    pub genes: Vec<String>,
    pub init: [i64; 3],
    pub light: String,
}

#[derive(Clone, Copy, Debug)]
pub struct SoilConsts {
    pub dose: i64,
    pub m0: i64,
    pub m_max: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct StageConsts {
    pub d_enter: i64,
    pub rest_dry: i64,
    pub rest_max: i64,
    pub rest_winter: i64,
    pub theta_dry: i64,
    pub theta_wet: i64,
}

/// The range of one param.
#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub lo: i64,
    pub hi: i64,
}

/// A carried law whose journal pin and life sections are sound, in the form the reducer reads.
#[derive(Clone, Debug)]
pub struct Life {
    pub name: String,
    pub digest: String,
    pub version: i64,
    pub provisional: bool,
    pub law: Value,
    pub table: Value,
    /// The per-day limit of every trunk kind but the genesis: its budget, or its cap.
    pub limits: BTreeMap<String, i64>,
    pub organs: Vec<Organ>,
    pub quiescent: Vec<Organ>,
    pub world: World,
    pub chem: ChemConsts,
    pub clock: ClockConsts,
    pub soil: SoilConsts,
    pub stage: StageConsts,
    pub params: BTreeMap<String, Range>,
}

impl Life {
    pub fn is_quiescent(&self, organ: Organ) -> bool {
        self.quiescent.contains(&organ)
    }
}

fn life_panic() -> Refused {
    bad("engine_panic", "life law")
}

fn field(value: Option<&Value>, key: &str) -> Result<i64, Refused> {
    int(value.and_then(|v| v.get(key))).ok_or_else(life_panic)
}

fn quad(value: Option<&Value>) -> Result<[i64; 4], Refused> {
    match value {
        Some(Value::Arr(items)) => match items.as_slice() {
            [Value::Int(a), Value::Int(b), Value::Int(c), Value::Int(d)] => Ok([*a, *b, *c, *d]),
            _ => Err(life_panic()),
        },
        _ => Err(life_panic()),
    }
}

fn organ_list(value: Option<&Value>) -> Result<Vec<Organ>, Refused> {
    let names = distinct_names(value).ok_or_else(life_panic)?;
    names.iter().map(|name| Organ::from_name(name).ok_or_else(life_panic)).collect()
}

/// The typed life of a sound law; every value read here was checked by `life_defect`.
fn build(name: &str, digest: String, law: Value, table: Value, budgets: BTreeMap<String, i64>) -> Result<Life, Refused> {
    let world_value = law.get("world");
    let year = world_value.and_then(|w| w.get("year"));
    let civil = text(year.and_then(|y| y.get("kind"))) == Some("civil");
    let light = world_value.and_then(|w| w.get("daylength"));
    let amp = light.and_then(|l| l.get("amp"));
    let rain = world_value.and_then(|w| w.get("rain"));
    let world = World {
        civil,
        days: if civil { CIVIL_YEAR } else { field(year, "days")? },
        equinox: field(light, "equinox")?,
        mean: field(light, "mean")?,
        ramp: field(light, "ramp")?,
        amp_long: field(amp, "long")?,
        amp_medium: field(amp, "medium")?,
        amp_short: field(amp, "short")?,
        rain_max: quad(rain.and_then(|r| r.get("max")))?,
        p_wet: quad(rain.and_then(|r| r.get("p_wet")))?,
        season_shift: field(world_value, "season_shift")?,
        south_shift: field(world_value, "south_shift")?,
    };
    let constants = law.get("constants");
    let chem_value = constants.and_then(|c| c.get("chem"));
    let enzymes = chem_value.and_then(|c| c.get("enzymes"));
    let mut roles = Vec::with_capacity(ROLES.len());
    for (role, scale_key, _) in ROLES {
        let names = distinct_names(enzymes.and_then(|e| e.get(role))).ok_or_else(life_panic)?;
        roles.push((names.iter().map(|name| String::from(*name)).collect(), field(chem_value, scale_key)?));
    }
    let chem = ChemConsts {
        core: field(chem_value, "core")?,
        fructan0: field(chem_value, "fructan0")?,
        fructan_max: field(chem_value, "fructan_max")?,
        k_w: field(chem_value, "k_w")?,
        metab_shift: field(chem_value, "metab_shift")?,
        sugar0: field(chem_value, "sugar0")?,
        sugar_max: field(chem_value, "sugar_max")?,
        theta_h: field(chem_value, "theta_h")?,
        theta_s: field(chem_value, "theta_s")?,
        roles,
    };
    let clock_value = constants.and_then(|c| c.get("clock"));
    let init = match clock_value.and_then(|c| c.get("init")) {
        Some(Value::Arr(items)) => match items.as_slice() {
            [Value::Int(a), Value::Int(b), Value::Int(c)] => [*a, *b, *c],
            _ => return Err(life_panic()),
        },
        _ => return Err(life_panic()),
    };
    let clock = ClockConsts {
        genes: distinct_names(clock_value.and_then(|c| c.get("genes")))
            .ok_or_else(life_panic)?
            .iter()
            .map(|gene| String::from(*gene))
            .collect(),
        init,
        light: String::from(text(clock_value.and_then(|c| c.get("light"))).ok_or_else(life_panic)?),
    };
    let soil_value = constants.and_then(|c| c.get("soil"));
    let soil = SoilConsts { dose: field(soil_value, "dose")?, m0: field(soil_value, "m0")?, m_max: field(soil_value, "m_max")? };
    let stage_value = constants.and_then(|c| c.get("stage"));
    let stage = StageConsts {
        d_enter: field(stage_value, "d_enter")?,
        rest_dry: field(stage_value, "rest_dry")?,
        rest_max: field(stage_value, "rest_max")?,
        rest_winter: field(stage_value, "rest_winter")?,
        theta_dry: field(stage_value, "theta_dry")?,
        theta_wet: field(stage_value, "theta_wet")?,
    };
    let mut params = BTreeMap::new();
    if let Some(Value::Obj(given)) = law.get("params") {
        for (param, spec) in given {
            params.insert(param.clone(), Range { lo: field(Some(spec), "lo")?, hi: field(Some(spec), "hi")? });
        }
    }
    let mut limits = budgets;
    if let Some(Value::Obj(caps)) = law.get("work").and_then(|w| w.get("caps")) {
        for (kind, cap) in caps {
            limits.insert(kind.clone(), int(Some(cap)).ok_or_else(life_panic)?);
        }
    }
    Ok(Life {
        name: String::from(name),
        digest,
        version: int(law.get("version")).ok_or_else(life_panic)?,
        provisional: matches!(law.get("provisional"), Some(Value::Bool(true))),
        organs: organ_list(law.get("organs"))?,
        quiescent: organ_list(law.get("quiescent_in_dormancy"))?,
        law,
        table,
        limits,
        world,
        chem,
        clock,
        soil,
        stage,
        params,
    })
}

/// Carried law `name` as a `Life`; `unknown_law` `law`, `journal`, `life law` or `code` otherwise.
pub fn life(name: &str) -> Result<Life, Refused> {
    let (law, digest) = law_file(name)?;
    life_of(name, digest, law)
}

/// A parsed law as a `Life`: its journal pin, then its life sections; `unknown_law` by name otherwise.
pub fn life_of(name: &str, digest: String, law: Value) -> Result<Life, Refused> {
    let (table, budgets) = journal_pin(&law)?;
    if let Some(defect) = life_defect(&law, &table) {
        return Err(bad("unknown_law", defect));
    }
    build(name, digest, law, table, budgets)
}

/// The first param, by name, outside the law's range; `None` when every one is inside.
pub fn params_defect<'a>(params: &BTreeMap<String, i64>, life: &'a Life) -> Result<Option<&'a str>, Refused> {
    for (name, range) in &life.params {
        let value = params.get(name).copied().ok_or_else(life_panic)?;
        if value < range.lo || value > range.hi {
            return Ok(Some(name.as_str()));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Constants a genome gives
// ---------------------------------------------------------------------------

/// The reserve's rates and saturation constants, from the genome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChemK {
    pub hy: i64,
    pub km_ps: i64,
    pub km_r: i64,
    pub ps: i64,
    pub r: i64,
    pub sy: i64,
}

/// The clock's production, decay, repression constants and light gain, from the genome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockK {
    pub alpha: [i64; 3],
    pub beta: [i64; 3],
    pub k: [i64; 3],
    pub light: i64,
    pub n: [i64; 3],
}

fn column(tables: &Value, table: &str, name: &str) -> Result<Vec<i64>, Refused> {
    let packed = tables.get(table).and_then(|t| t.get(name)).ok_or_else(life_panic)?;
    let (_, values) = ocj::unpack_bulk(packed)?;
    values.into_iter().map(|value| i64::try_from(value).map_err(|_| life_panic())).collect()
}

fn at(values: &[i64], index: i64) -> Result<i64, Refused> {
    usize::try_from(index).ok().and_then(|i| values.get(i)).copied().ok_or_else(life_panic)
}

/// The organs' genome-derived constants from compiled tables under `life`.
///
/// The clock: for each gene named in `constants.clock.genes`, the TF row
/// whose gene's locus has that name (the last such row) gives `alpha` and
/// `beta`, and the gene's first promoter edge gives `k` and `n`; `light` is
/// the gain of the first REC row named `constants.clock.light`. An absent
/// row gives zeros, with `k = ONE` and `n = 1`; a present gene without an
/// edge gives `k = ONE` and `n = 1` too. The reserve: each rate is the summed
/// `kcat` of the role's loci present, times its scale, shifted down by 16;
/// `km_ps` and `km_r` are the `km` of the first present locus of
/// photosynthesis and respiration, else `ONE`.
pub fn consts(tables: &Value, life: &Life) -> Result<(ChemK, ClockK), Refused> {
    let mut names: BTreeMap<i64, &str> = BTreeMap::new();
    if let Some(Value::Arr(loci)) = life.law.get("genome").and_then(|g| g.get("loci")) {
        for entry in loci {
            let id = int(entry.get("id")).ok_or_else(life_panic)?;
            names.insert(id, text(entry.get("name")).ok_or_else(life_panic)?);
        }
    }
    let loci = column(tables, "genes", "locus")?;
    let name_of = |gene: i64| -> Result<&str, Refused> {
        let locus = at(&loci, gene)?;
        names.get(&locus).copied().ok_or_else(life_panic)
    };
    let starts = column(tables, "genes", "edge_start")?;
    let edge_k = column(tables, "edges", "k")?;
    let edge_n = column(tables, "edges", "n")?;
    let mut tf_rows: BTreeMap<&str, (usize, i64)> = BTreeMap::new();
    for (row, gene) in column(tables, "tf", "gene")?.into_iter().enumerate() {
        tf_rows.insert(name_of(gene)?, (row, gene));
    }
    let prod = column(tables, "tf", "prod")?;
    let deg = column(tables, "tf", "deg")?;
    let mut alpha = [0_i64; 3];
    let mut beta = [0_i64; 3];
    let mut ks = [ONE; 3];
    let mut ns = [1_i64; 3];
    let slots = alpha.iter_mut().zip(beta.iter_mut()).zip(ks.iter_mut()).zip(ns.iter_mut());
    for (gene_name, (((a, b), k), n)) in life.clock.genes.iter().zip(slots) {
        let Some(&(row, gene)) = tf_rows.get(gene_name.as_str()) else {
            continue;
        };
        *a = prod.get(row).copied().ok_or_else(life_panic)?;
        *b = deg.get(row).copied().ok_or_else(life_panic)?;
        let first = at(&starts, gene)?;
        let next = at(&starts, gene.checked_add(1).ok_or_else(life_panic)?)?;
        if next > first {
            *k = at(&edge_k, first)?;
            *n = at(&edge_n, first)?;
        }
    }
    let mut light = 0_i64;
    let gains = column(tables, "rec", "gain")?;
    for (row, gene) in column(tables, "rec", "gene")?.into_iter().enumerate() {
        if name_of(gene)? == life.clock.light {
            light = gains.get(row).copied().ok_or_else(life_panic)?;
            break;
        }
    }
    let mut enz_rows: BTreeMap<&str, usize> = BTreeMap::new();
    for (row, gene) in column(tables, "enz", "gene")?.into_iter().enumerate() {
        enz_rows.insert(name_of(gene)?, row);
    }
    let kcat = column(tables, "enz", "kcat")?;
    let km = column(tables, "enz", "km")?;
    let mut rates = [0_i64; 4];
    let mut first_km = [ONE; 4];
    for ((names_of_role, scale), (rate, km_slot)) in life.chem.roles.iter().zip(rates.iter_mut().zip(first_km.iter_mut())) {
        let mut total: i128 = 0;
        let mut found = false;
        for locus in names_of_role {
            if let Some(&row) = enz_rows.get(locus.as_str()) {
                total = total.checked_add(i128::from(kcat.get(row).copied().ok_or_else(life_panic)?)).ok_or_else(life_panic)?;
                if !found {
                    *km_slot = km.get(row).copied().ok_or_else(life_panic)?;
                    found = true;
                }
            }
        }
        let scaled = total.checked_mul(i128::from(*scale)).and_then(|v| v.checked_shr(16)).ok_or_else(life_panic)?;
        *rate = i64::try_from(scaled).map_err(|_| life_panic())?;
    }
    // ROLES order: hydrolysis, photosynthesis, respiration, synthesis.
    let [hy, ps, r, sy] = rates;
    let [_, km_ps, km_r, _] = first_km;
    Ok((ChemK { hy, km_ps, km_r, ps, r, sy }, ClockK { alpha, beta, k: ks, light, n: ns }))
}
