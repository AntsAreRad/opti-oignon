//! Phonology: the sounds a being's genome allows, the forms it may say, and
//! how it coins them; the twin of `opti_oignon/allium/ref/organs/phon.py`.
//!
//! Everything here is a pure function of its arguments: a 63-byte language
//! block (read from the genome's seven LEX records), a seed, a concept and
//! its iconic signs, and the lists a caller passes. There is no store, no
//! journal and no clock.
//!
//! * The alphabet is 27 symbols, one ASCII character per phoneme, with
//!   integer features in a table file the law pins by digest. The classes
//!   (vowels, consonants, stops, nasals) are derived from the parsed
//!   features, never written out.
//! * `decode` turns a block into a phonology: the potential inventory with
//!   its floors, the template and word-length weights, the phonotactic genes.
//! * `licit` says whether a form is sayable, with the first failing reason,
//!   or its syllables and split points.
//! * `invent_case` coins a form: keyed draws, an integer iconic bias, at
//!   most sixteen attempts, then a gesture. A rejected candidate is reported
//!   only by its digest.
//! * The taboo filter holds SHA-256 digests keyed by `(length, digest)`,
//!   matched on the whole form and every substring of three to eight
//!   letters; the work it charges does not depend on what it holds.
//! * `sas_list` derives 2048 distinct sayable forms from a block, and
//!   `sas_indices` reads the first 66 bits of a digest as six of them.
//!
//! Every lookup is a `get` or a slice pattern and every operation on a
//! runtime value is checked. A check that fails where the reference cannot
//! fail refuses `engine_panic` with a detail naming its site, `phon <site>`;
//! no failed lookup ever becomes a default value.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::min;
use sha2::{Digest, Sha256};

use super::genome::CountingStream;
use crate::ocj::{self, obj, refused, Refused, Value};
use crate::rng;

pub const ALPHABET: &[u8; 27] = b"aeioupbtdkgq'cjfvszxhmnlrwy";
pub const WEIGHTS: usize = 27;
pub const LEX_BYTES: usize = 63;
/// `(onset length, coda length)` of V, CV, VC, CVC, CCV, CCVC, in LEX order.
pub const TEMPLATES: [(usize, usize); 6] = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)];
/// The sonority each manner must carry.
const SON: [u8; 8] = [1, 2, 3, 4, 5, 5, 6, 7];
/// The s of the s + stop exception.
pub const S_PHONEME: usize = 17;
pub const FORM_MAX: usize = 12;
pub const SHOWN_MAX: usize = 24;
pub const INVENT_TRIES: u64 = 16;
pub const FLOOR_CONSONANTS: usize = 6;

pub const SAS_SIZE: usize = 2048;
pub const SAS_CAP: u64 = 16384;
/// The syllables of a phase-1 word, then of a phase-2 word.
pub const SAS_SYLLABLES: [usize; 2] = [3, 4];
pub const SAS_PREFIX: &[u8] = b"oo-allium-sas-v1";

pub const DOMAIN_FALLBACK: &str = "lang.fallback";
pub const DOMAIN_FIRST: &str = "lang.first_sound";
pub const DOMAIN_INVENT: &str = "lang.invent";
pub const DOMAIN_SAS: &str = "lang.sas";
pub const DOMAINS: [&str; 4] = [DOMAIN_FALLBACK, DOMAIN_FIRST, DOMAIN_INVENT, DOMAIN_SAS];

/// The reasons `licit` gives, in check order.
pub const REASONS: [&str; 9] = ["length", "alphabet", "inventory", "ocp", "long_vowel", "no_nucleus", "onset", "cluster", "coda"];

const TABLE_KEYS: [&str; 21] = [
    "alphabet",
    "anchored_max",
    "anchored_total_max",
    "bias",
    "features",
    "first_sound_exclude",
    "floor_consonants",
    "floor_vowels",
    "fold",
    "form_max",
    "invent_tries",
    "lex_box",
    "name",
    "potential_min",
    "sas",
    "shown_max",
    "taboo_extra_max",
    "taboo_max",
    "taboo_window",
    "templates",
    "version",
];

// Feature columns.
const F_CLASS: usize = 0;
const F_PLACE: usize = 1;
const F_MANNER: usize = 2;
const F_SON: usize = 4;
const F_HEIGHT: usize = 5;
const F_BACK: usize = 6;
const F_ROUND: usize = 7;
const FEATURE_RANGES: [(i64, i64); 8] = [(0, 1), (0, 8), (0, 7), (0, 1), (1, 7), (0, 2), (0, 2), (0, 1)];

/// The genes a block carries after its 27 weights, each at its byte.
const GENE_BYTES: [(&str, usize); 21] = [
    ("iconicity", 36),
    ("playfulness", 37),
    ("conformity", 38),
    ("openness", 39),
    ("conservatism", 40),
    ("chattiness", 41),
    ("critical_len", 42),
    ("voice_pitch", 43),
    ("voice_tempo", 44),
    ("onset_max", 45),
    ("coda_max", 46),
    ("coda_classes", 47),
    ("s_exception", 48),
    ("stress", 49),
    ("harmony", 50),
    ("length", 51),
    ("reduction_vowel", 52),
    ("epenthetic_vowel", 53),
    ("repair", 54),
    ("head_order", 55),
    ("dominance", 56),
];
const TEMPLATE_BYTES: usize = 27;
const WORD_LEN_BYTES: usize = 33;

/// A check the reference has no refusal for failed: an engine defect, named by its site.
fn defect(site: &str) -> Refused {
    refused("engine_panic", format!("phon {}", site))
}

fn count_value(n: u64, site: &str) -> Result<Value, Refused> {
    i64::try_from(n).map(Value::Int).map_err(|_| defect(site))
}

fn index_value(n: usize, site: &str) -> Result<Value, Refused> {
    i64::try_from(n).map(Value::Int).map_err(|_| defect(site))
}

/// The phoneme a byte spells, if it is one of the 27 symbols.
pub fn index_of(byte: u8) -> Option<usize> {
    ALPHABET.iter().position(|symbol| *symbol == byte)
}

/// One to `max` characters, each in `[a-z']` (the whole alphabet, the apostrophe included).
pub fn is_form(text: &str, max: usize) -> bool {
    !text.is_empty() && text.len() <= max && text.bytes().all(|byte| index_of(byte).is_some())
}

fn text_of(letters: &[u8]) -> String {
    letters.iter().map(|byte| char::from(*byte)).collect()
}

fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

/// A validated phonology table, parsed into fixed arrays, with its derived classes.
#[derive(Clone, Debug)]
pub struct Table {
    pub features: [[u8; 8]; WEIGHTS],
    pub bias: [[i16; 4]; WEIGHTS],
    pub lex_box: [(u8, u8); LEX_BYTES],
    pub potential_min: u8,
    pub anchored_max: usize,
    pub anchored_total_max: usize,
    pub taboo_max: usize,
    pub taboo_extra_max: usize,
    pub fold: Vec<(i64, String)>,
    pub first_exclude: Vec<usize>,
    pub sas_exclude: Vec<usize>,
    pub vowels: Vec<usize>,
    pub cons: Vec<usize>,
    pub stops: Vec<usize>,
    pub nasals: Vec<usize>,
}

impl Table {
    fn feature(&self, p: usize, column: usize) -> Option<u8> {
        self.features.get(p)?.get(column).copied()
    }

    pub fn class(&self, p: usize) -> Option<u8> {
        self.feature(p, F_CLASS)
    }

    pub fn manner(&self, p: usize) -> Option<u8> {
        self.feature(p, F_MANNER)
    }

    pub fn son(&self, p: usize) -> Option<u8> {
        self.feature(p, F_SON)
    }
}

fn int_of(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Int(number)) => Some(*number),
        _ => None,
    }
}

/// A list of integers (a boolean is not one), of exactly `length` items when one is asked.
fn int_list(value: Option<&Value>, length: Option<usize>) -> Option<Vec<i64>> {
    let Some(Value::Arr(items)) = value else {
        return None;
    };
    if length.is_some_and(|length| items.len() != length) {
        return None;
    }
    items
        .iter()
        .map(|item| match item {
            Value::Int(number) => Some(*number),
            _ => None,
        })
        .collect()
}

/// A list of rows of integers (a boolean is not one) equal to the six templates.
fn templates_match(value: Option<&Value>) -> bool {
    let Some(Value::Arr(rows)) = value else {
        return false;
    };
    rows.len() == TEMPLATES.len()
        && rows.iter().zip(TEMPLATES.iter()).all(|(row, (onset, coda))| match row {
            Value::Arr(pair) => match pair.as_slice() {
                [Value::Int(a), Value::Int(b)] => {
                    usize::try_from(*a).ok() == Some(*onset) && usize::try_from(*b).ok() == Some(*coda)
                }
                _ => false,
            },
            _ => false,
        })
}

/// The `sas` object every table must carry.
pub fn sas_value() -> Value {
    obj(vec![
        (String::from("bits"), Value::Int(11)),
        (String::from("cap"), Value::Int(16384)),
        (String::from("exclude"), Value::Arr(vec![Value::Int(12)])),
        (String::from("size"), Value::Int(2048)),
        (String::from("syllables"), Value::Arr(vec![Value::Int(3), Value::Int(4)])),
        (String::from("words"), Value::Int(6)),
    ])
}

fn parse_features(value: Option<&Value>) -> Option<[[u8; 8]; WEIGHTS]> {
    let Some(Value::Arr(rows)) = value else {
        return None;
    };
    if rows.len() != WEIGHTS {
        return None;
    }
    let mut out = [[0_u8; 8]; WEIGHTS];
    for (p, (row, slot)) in rows.iter().zip(out.iter_mut()).enumerate() {
        let numbers = int_list(Some(row), Some(8))?;
        for ((number, (low, high)), cell) in numbers.iter().zip(FEATURE_RANGES.iter()).zip(slot.iter_mut()) {
            if number < low || number > high {
                return None;
            }
            *cell = u8::try_from(*number).ok()?;
        }
        let vowel = p <= 4;
        let column = |at: usize| slot.get(at).copied();
        let (class, place, manner, son) = (column(F_CLASS)?, column(F_PLACE)?, column(F_MANNER)?, column(F_SON)?);
        if (class == 0) != vowel || (manner == 7) != vowel || (place == 8) != vowel {
            return None;
        }
        // The range check above already refused a manner past 7; the sonority is read only within it.
        if Some(&son) != SON.get(usize::from(manner)) {
            return None;
        }
        if !vowel && (column(F_HEIGHT)? != 0 || column(F_BACK)? != 0 || column(F_ROUND)? != 0) {
            return None;
        }
    }
    let with_manner = |wanted: u8| -> Vec<usize> {
        (0..WEIGHTS).filter(|p| out.get(*p).and_then(|row| row.get(F_MANNER)) == Some(&wanted)).collect()
    };
    let stops: Vec<usize> = (5..=12).collect();
    if with_manner(0) != stops || with_manner(3) != vec![21, 22] || ALPHABET.get(S_PHONEME) != Some(&b's') {
        return None;
    }
    Some(out)
}

fn parse_bias(value: Option<&Value>) -> Option<[[i16; 4]; WEIGHTS]> {
    let Some(Value::Arr(rows)) = value else {
        return None;
    };
    if rows.len() != WEIGHTS {
        return None;
    }
    let mut out = [[0_i16; 4]; WEIGHTS];
    for (row, slot) in rows.iter().zip(out.iter_mut()) {
        let numbers = int_list(Some(row), Some(4))?;
        for (number, cell) in numbers.iter().zip(slot.iter_mut()) {
            if !(-64..=64).contains(number) {
                return None;
            }
            *cell = i16::try_from(*number).ok()?;
        }
    }
    Some(out)
}

fn parse_lex_box(value: Option<&Value>) -> Option<[(u8, u8); LEX_BYTES]> {
    let Some(Value::Arr(pairs)) = value else {
        return None;
    };
    if pairs.len() != LEX_BYTES {
        return None;
    }
    let mut out = [(0_u8, 0_u8); LEX_BYTES];
    for (pair, slot) in pairs.iter().zip(out.iter_mut()) {
        let numbers = int_list(Some(pair), Some(2))?;
        let [low, high] = numbers.as_slice() else {
            return None;
        };
        if !(0 <= *low && low <= high && *high <= 255) {
            return None;
        }
        *slot = (u8::try_from(*low).ok()?, u8::try_from(*high).ok()?);
    }
    // `licit` is exact only for onsets of at most two and codas of at most one.
    if out.get(45)?.1 > 2 || out.get(46)?.1 > 1 {
        return None;
    }
    // The reduction and epenthetic vowels are coded 0 to 4, the phoneme index of the five vowels.
    if out.get(52)?.1 > 4 || out.get(53)?.1 > 4 {
        return None;
    }
    Some(out)
}

fn parse_fold(value: Option<&Value>) -> Option<Vec<(i64, String)>> {
    let Some(Value::Arr(entries)) = value else {
        return None;
    };
    let mut out = Vec::with_capacity(entries.len());
    let mut previous: i64 = -1;
    for entry in entries {
        let Value::Arr(pair) = entry else {
            return None;
        };
        let [Value::Int(code), Value::Str(replacement)] = pair.as_slice() else {
            return None;
        };
        let replacement_ok = replacement.len() <= 2 && replacement.bytes().all(|byte| index_of(byte).is_some());
        if !(previous < *code && *code <= 0x0010_FFFF && replacement_ok) {
            return None;
        }
        previous = *code;
        out.push((*code, replacement.clone()));
    }
    Some(out)
}

fn scalar_in(value: &Value, key: &str, low: i64, high: i64) -> Option<i64> {
    int_of(value.get(key)).filter(|number| (low..=high).contains(number))
}

/// The table, parsed, when it is sound; `None` for any defect the reference names.
pub fn parse_table(value: &Value) -> Option<Table> {
    let Value::Obj(members) = value else {
        return None;
    };
    if members.len() != TABLE_KEYS.len() || !TABLE_KEYS.iter().all(|key| value.get(key).is_some()) {
        return None;
    }
    if !matches!(value.get("alphabet"), Some(Value::Str(text)) if text.as_bytes() == ALPHABET.as_slice()) {
        return None;
    }
    if !templates_match(value.get("templates")) {
        return None;
    }
    for (key, expected) in [("invent_tries", 16), ("form_max", 12), ("shown_max", 24), ("floor_consonants", 6)] {
        if int_of(value.get(key)) != Some(expected) {
            return None;
        }
    }
    for (key, expected) in [("taboo_window", vec![3, 8]), ("floor_vowels", vec![0, 2, 4]), ("first_sound_exclude", vec![12])] {
        if int_list(value.get(key), None) != Some(expected) {
            return None;
        }
    }
    if value.get("sas") != Some(&sas_value()) {
        return None;
    }
    let features = parse_features(value.get("features"))?;
    let bias = parse_bias(value.get("bias"))?;
    let lex_box = parse_lex_box(value.get("lex_box"))?;
    let potential_min = u8::try_from(scalar_in(value, "potential_min", 1, 255)?).ok()?;
    let mut caps = [0_usize; 4];
    for (key, slot) in ["anchored_max", "anchored_total_max", "taboo_max", "taboo_extra_max"].iter().zip(caps.iter_mut()) {
        *slot = usize::try_from(scalar_in(value, key, 1, 65536)?).ok()?;
    }
    let [anchored_max, anchored_total_max, taboo_max, taboo_extra_max] = caps;
    let fold = parse_fold(value.get("fold"))?;
    if !matches!(value.get("name"), Some(Value::Str(_))) || int_of(value.get("version")).is_none() {
        return None;
    }
    let first_exclude = int_list(value.get("first_sound_exclude"), None)?
        .iter()
        .map(|p| usize::try_from(*p).ok())
        .collect::<Option<Vec<usize>>>()?;
    let sas_exclude = int_list(value.get("sas").and_then(|sas| sas.get("exclude")), None)?
        .iter()
        .map(|p| usize::try_from(*p).ok())
        .collect::<Option<Vec<usize>>>()?;
    let with = |test: &dyn Fn(&[u8; 8]) -> bool| -> Vec<usize> {
        features.iter().enumerate().filter(|(_, row)| test(row)).map(|(p, _)| p).collect()
    };
    let vowels = with(&|row| row.first() == Some(&0));
    let cons = with(&|row| row.first() == Some(&1));
    let stops = with(&|row| row.get(F_MANNER) == Some(&0));
    let nasals = with(&|row| row.get(F_MANNER) == Some(&3));
    Some(Table {
        features,
        bias,
        lex_box,
        potential_min,
        anchored_max,
        anchored_total_max,
        taboo_max,
        taboo_extra_max,
        fold,
        first_exclude,
        sas_exclude,
        vowels,
        cons,
        stops,
        nasals,
    })
}

/// A taboo pair `[length 1..=12, 64 lowercase hex]`, read as `(length, digest)`.
pub fn taboo_pair(entry: &Value) -> Option<(usize, [u8; 32])> {
    let Value::Arr(pair) = entry else {
        return None;
    };
    let [Value::Int(length), Value::Str(hex)] = pair.as_slice() else {
        return None;
    };
    if !(1..=12).contains(length) || !ocj::is_hex(hex, 64) {
        return None;
    }
    let digest = <[u8; 32]>::try_from(ocj::from_hex(hex)?).ok()?;
    Some((usize::try_from(*length).ok()?, digest))
}

/// The entries of a sound taboo table, in file order; `None` for any defect the reference names.
pub fn parse_taboo(value: &Value, table: &Table) -> Option<Vec<(usize, [u8; 32])>> {
    if !matches!(value, Value::Obj(_)) || value.keys() != ["entries", "name", "version"] {
        return None;
    }
    if !matches!(value.get("name"), Some(Value::Str(_))) || int_of(value.get("version")).is_none() {
        return None;
    }
    let Some(Value::Arr(entries)) = value.get("entries") else {
        return None;
    };
    if entries.len() > table.taboo_max {
        return None;
    }
    let mut out: Vec<(usize, [u8; 32])> = Vec::with_capacity(entries.len());
    for entry in entries {
        let pair = taboo_pair(entry)?;
        // Lowercase hex orders as its bytes, so the reference's (length, hex) order is this one.
        if out.last().is_some_and(|previous| pair <= *previous) {
            return None;
        }
        out.push(pair);
    }
    Some(out)
}

/// The law-level block: every byte at the top of its box.
pub fn law_lex(table: &Table) -> [u8; LEX_BYTES] {
    let mut out = [0_u8; LEX_BYTES];
    for (slot, (_, high)) in out.iter_mut().zip(table.lex_box.iter()) {
        *slot = *high;
    }
    out
}

pub fn lex_ok(lex: &[u8], table: &Table) -> bool {
    lex.len() == LEX_BYTES && lex.iter().zip(table.lex_box.iter()).all(|(byte, (low, high))| low <= byte && byte <= high)
}

// ---------------------------------------------------------------------------
// The taboo filter
// ---------------------------------------------------------------------------

/// The taboo pairs, sorted and without repeats; only ever asked for membership.
pub struct Taboo {
    pairs: Vec<(usize, [u8; 32])>,
    lengths: Vec<usize>,
}

impl Taboo {
    /// The law's list and a request's additions; neither order nor repeats matter.
    pub fn new(entries: &[(usize, [u8; 32])], extra: &[(usize, [u8; 32])]) -> Taboo {
        let mut pairs: Vec<(usize, [u8; 32])> = entries.iter().chain(extra.iter()).copied().collect();
        pairs.sort_unstable();
        pairs.dedup();
        let mut lengths: Vec<usize> = pairs.iter().map(|(length, _)| *length).collect();
        lengths.dedup();
        Taboo { pairs, lengths }
    }

    fn listed(&self, data: &[u8]) -> bool {
        let length = data.len();
        // Hashing only the lengths present changes neither the answer nor the work charged.
        self.lengths.binary_search(&length).is_ok() && self.pairs.binary_search(&(length, sha256(data))).is_ok()
    }

    /// True when the whole form, or a substring of three to eight letters shorter than it, is listed.
    pub fn hit(&self, form: &[u8]) -> bool {
        if self.pairs.is_empty() {
            return false;
        }
        if self.listed(form) {
            return true;
        }
        let top = min(8, form.len().saturating_sub(1));
        (3..=top).any(|length| form.windows(length).any(|window| self.listed(window)))
    }
}

/// `1 + sum over L in 3..=min(8, n - 1) of (n - L + 1)`: charged whatever the list holds.
pub fn taboo_work(n: usize) -> Option<u64> {
    let mut work: u64 = 1;
    for length in 3..=min(8, n.saturating_sub(1)) {
        let windows = n.checked_sub(length)?.checked_add(1)?;
        work = work.checked_add(u64::try_from(windows).ok()?)?;
    }
    Some(work)
}

// ---------------------------------------------------------------------------
// Decoding a block
// ---------------------------------------------------------------------------

/// A block's phonology: the membership, its floors and weights, and the phonotactic genes.
pub struct Phonology<'t> {
    pub table: &'t Table,
    pub lex: [u8; LEX_BYTES],
    pub member: [bool; WEIGHTS],
    pub floors: Vec<usize>,
    pub weight: [u64; WEIGHTS],
    pub templates: [u64; 6],
    pub word_len: [u64; 3],
    pub onset_max: usize,
    pub coda_max: usize,
    pub coda_classes: u8,
    pub s_exception: u8,
    pub length: u8,
    pub iconicity: u8,
    pub c1set: [bool; WEIGHTS],
    pub codaset: [bool; WEIGHTS],
}

/// The highest-weight non-member, the lowest index winning ties; `None` when all are members.
fn best(candidates: &[usize], member: &[bool; WEIGHTS], lex: &[u8; LEX_BYTES]) -> Option<Option<usize>> {
    let mut kept: Option<usize> = None;
    for p in candidates {
        if *member.get(*p)? {
            continue;
        }
        kept = match kept {
            None => Some(*p),
            Some(k) if lex.get(*p)? > lex.get(k)? => Some(*p),
            Some(k) => Some(k),
        };
    }
    Some(kept)
}

/// A consonant pair a cluster may hold: a sonority rise of two, or s + stop when the gene allows it.
pub fn cluster_ok(table: &Table, s_exception: u8, c: usize, d: usize) -> Option<bool> {
    if c == d {
        return Some(false);
    }
    if i64::from(table.son(d)?).checked_sub(i64::from(table.son(c)?))? >= 2 {
        return Some(true);
    }
    if s_exception != 1 || c != S_PHONEME {
        return Some(false);
    }
    Some(table.manner(d)? == 0)
}

fn coda_ok(table: &Table, coda_classes: u8, c: usize) -> Option<bool> {
    Some(coda_classes.checked_shr(u32::from(table.manner(c)?))? & 1 == 1)
}

impl Phonology<'_> {
    fn cluster_ok(&self, c: usize, d: usize) -> Option<bool> {
        cluster_ok(self.table, self.s_exception, c, d)
    }

    fn coda_ok(&self, c: usize) -> Option<bool> {
        coda_ok(self.table, self.coda_classes, c)
    }
}

/// The phonology of a validated block; `inventory` (a mask) replaces the membership.
pub fn decode<'t>(lex: &[u8; LEX_BYTES], table: &'t Table, inventory: Option<u32>) -> Result<Phonology<'t>, Refused> {
    let site = || defect("decode");
    let byte = |at: usize| lex.get(at).copied().ok_or_else(site);
    let mut member = [false; WEIGHTS];
    for (p, slot) in member.iter_mut().enumerate() {
        *slot = byte(p)? >= table.potential_min;
    }
    let mut floors: Vec<usize> = Vec::new();
    for p in [0, 2, 4, usize::from(byte(52)?), usize::from(byte(53)?)] {
        let slot = member.get_mut(p).ok_or_else(site)?;
        if !*slot {
            *slot = true;
            floors.push(p);
        }
    }
    for group in [&table.stops, &table.nasals] {
        let mut any = false;
        for p in group.iter() {
            any = any || *member.get(*p).ok_or_else(site)?;
        }
        if !any {
            let p = best(group, &member, lex).ok_or_else(site)?.ok_or_else(site)?;
            *member.get_mut(p).ok_or_else(site)? = true;
            floors.push(p);
        }
    }
    loop {
        let mut members: usize = 0;
        for c in &table.cons {
            if *member.get(*c).ok_or_else(site)? {
                members = members.checked_add(1).ok_or_else(site)?;
            }
        }
        if members >= FLOOR_CONSONANTS {
            break;
        }
        let p = best(&table.cons, &member, lex).ok_or_else(site)?.ok_or_else(site)?;
        *member.get_mut(p).ok_or_else(site)? = true;
        floors.push(p);
    }
    if let Some(mask) = inventory {
        for (p, slot) in member.iter_mut().enumerate() {
            let shift = u32::try_from(p).map_err(|_| site())?;
            *slot = mask.checked_shr(shift).ok_or_else(site)? & 1 == 1;
        }
    }
    let mut weight = [0_u64; WEIGHTS];
    for (p, slot) in weight.iter_mut().enumerate() {
        if *member.get(p).ok_or_else(site)? {
            *slot = u64::from(byte(p)?).checked_add(1).ok_or_else(site)?;
        }
    }
    let onset_max = usize::from(byte(45)?);
    let coda_max = usize::from(byte(46)?);
    let coda_classes = byte(47)?;
    let s_exception = byte(48)?;
    let mut c1set = [false; WEIGHTS];
    let mut codaset = [false; WEIGHTS];
    for c in &table.cons {
        if !*member.get(*c).ok_or_else(site)? {
            continue;
        }
        let mut pairs = false;
        for d in &table.cons {
            if *member.get(*d).ok_or_else(site)? && cluster_ok(table, s_exception, *c, *d).ok_or_else(site)? {
                pairs = true;
                break;
            }
        }
        *c1set.get_mut(*c).ok_or_else(site)? = pairs;
        *codaset.get_mut(*c).ok_or_else(site)? = coda_ok(table, coda_classes, *c).ok_or_else(site)?;
    }
    let any_c1 = c1set.iter().any(|flag| *flag);
    let any_coda = codaset.iter().any(|flag| *flag);
    let mut templates = [0_u64; 6];
    for (t, ((onset, coda), slot)) in TEMPLATES.iter().zip(templates.iter_mut()).enumerate() {
        let allowed = *onset <= onset_max && *coda <= coda_max && (*onset < 2 || any_c1) && (*coda < 1 || any_coda);
        if allowed {
            let at = TEMPLATE_BYTES.checked_add(t).ok_or_else(site)?;
            *slot = u64::from(byte(at)?).checked_add(1).ok_or_else(site)?;
        }
    }
    let mut word_len = [0_u64; 3];
    for (n, slot) in word_len.iter_mut().enumerate() {
        let at = WORD_LEN_BYTES.checked_add(n).ok_or_else(site)?;
        *slot = u64::from(byte(at)?).checked_add(1).ok_or_else(site)?;
    }
    Ok(Phonology {
        table,
        lex: *lex,
        member,
        floors,
        weight,
        templates,
        word_len,
        onset_max,
        coda_max,
        coda_classes,
        s_exception,
        length: byte(51)?,
        iconicity: byte(36)?,
        c1set,
        codaset,
    })
}

/// The wire form of a phonology (`phon_inventory`).
pub fn phonology_value(ph: &Phonology<'_>) -> Result<Value, Refused> {
    let site = "inventory";
    let mut members: Vec<(String, Value)> = Vec::with_capacity(27);
    for (name, at) in GENE_BYTES {
        let gene = ph.lex.get(at).copied().ok_or_else(|| defect(site))?;
        members.push((String::from(name), Value::Int(i64::from(gene))));
    }
    let mut inventory: Vec<Value> = Vec::new();
    let mut mask: i64 = 0;
    for (p, is_member) in ph.member.iter().enumerate() {
        if *is_member {
            inventory.push(index_value(p, site)?);
            let bit = 1_i64.checked_shl(u32::try_from(p).map_err(|_| defect(site))?).ok_or_else(|| defect(site))?;
            mask |= bit;
        }
    }
    let list = |values: &[u64]| -> Result<Value, Refused> {
        values.iter().map(|n| count_value(*n, site)).collect::<Result<Vec<Value>, Refused>>().map(Value::Arr)
    };
    let floors = ph.floors.iter().map(|p| index_value(*p, site)).collect::<Result<Vec<Value>, Refused>>()?;
    members.push((String::from("floors"), Value::Arr(floors)));
    members.push((String::from("inventory"), Value::Arr(inventory)));
    members.push((String::from("mask"), Value::Int(mask)));
    members.push((String::from("templates"), list(&ph.templates)?));
    members.push((String::from("weights"), list(&ph.weight)?));
    members.push((String::from("word_len"), list(&ph.word_len)?));
    Ok(obj(members))
}

/// A block from the seed alone, each byte uniform in its box: `(block, draws)`.
pub fn fallback_lex(seed: &[u8; 32], table: &Table) -> Result<([u8; LEX_BYTES], u64), Refused> {
    let site = || defect("fallback");
    let mut stream = CountingStream::from_key(&rng::key(seed, DOMAIN_FALLBACK, &[]));
    let mut block = [0_u8; LEX_BYTES];
    for (slot, (low, high)) in block.iter_mut().zip(table.lex_box.iter()) {
        *slot = if low == high {
            *low
        } else {
            let span = u64::from(*high).checked_sub(u64::from(*low)).and_then(|n| n.checked_add(1)).ok_or_else(site)?;
            let draw = stream.below(span).ok_or_else(site)?;
            u8::try_from(u64::from(*low).checked_add(draw).ok_or_else(site)?).map_err(|_| site())?
        };
    }
    Ok((block, stream.draws))
}

/// The block the seven LEX rows of compiled tables carry, placed by their offset field.
pub fn lex_from_tables(tables: &Value) -> Result<[u8; LEX_BYTES], Refused> {
    let site = || refused("engine_panic", String::from("lex block"));
    let column = tables.get("reserved").and_then(|reserved| reserved.get("lex")).and_then(|lex| lex.get("fields")).ok_or_else(site)?;
    let (_, fields) = ocj::unpack_bulk(column).map_err(|_| site())?;
    if fields.len() != 70 {
        return Err(site());
    }
    let mut block = [0_u8; LEX_BYTES];
    let mut seen: Vec<usize> = Vec::with_capacity(7);
    for row in fields.chunks(10) {
        let [offset, rest @ ..] = row else {
            return Err(site());
        };
        let offset = usize::try_from(*offset).map_err(|_| site())?;
        if ![0, 9, 18, 27, 36, 45, 54].contains(&offset) || seen.contains(&offset) {
            return Err(site());
        }
        seen.push(offset);
        if rest.len() != 9 {
            return Err(site());
        }
        // The row's offset first, then each of its nine values in order, each within a byte.
        for (k, value) in rest.iter().enumerate() {
            let byte = u8::try_from(*value).map_err(|_| site())?;
            let at = offset.checked_add(k).ok_or_else(site)?;
            *block.get_mut(at).ok_or_else(site)? = byte;
        }
    }
    Ok(block)
}

// ---------------------------------------------------------------------------
// Licit forms
// ---------------------------------------------------------------------------

/// What `licit` says of a form: the first failing reason, or its syllables and split points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Reason(&'static str),
    Licit { syllables: usize, splits: Vec<usize> },
}

/// Whether a form is sayable under a phonology. The first failing check decides.
pub fn licit(form: &[u8], ph: &Phonology<'_>) -> Result<Verdict, Refused> {
    let site = || defect("licit");
    if form.is_empty() || form.len() > FORM_MAX {
        return Ok(Verdict::Reason("length"));
    }
    let mut ps: Vec<usize> = Vec::with_capacity(form.len());
    for byte in form {
        match index_of(*byte) {
            Some(p) => ps.push(p),
            None => return Ok(Verdict::Reason("alphabet")),
        }
    }
    for p in &ps {
        if !*ph.member.get(*p).ok_or_else(site)? {
            return Ok(Verdict::Reason("inventory"));
        }
    }
    let table = ph.table;
    let mut vowel: Vec<bool> = Vec::with_capacity(ps.len());
    for p in &ps {
        vowel.push(table.class(*p).ok_or_else(site)? == 0);
    }
    for (pair, kinds) in ps.windows(2).zip(vowel.windows(2)) {
        let ([a, b], [vowel_a, vowel_b]) = (pair, kinds) else {
            return Err(site());
        };
        if !*vowel_a && !*vowel_b && a == b {
            return Ok(Verdict::Reason("ocp"));
        }
    }
    // Nuclei: maximal runs of one vowel letter, as (start, length).
    let n = ps.len();
    let mut nuclei: Vec<(usize, usize)> = Vec::new();
    let mut i: usize = 0;
    while i < n {
        if !*vowel.get(i).ok_or_else(site)? {
            i = i.checked_add(1).ok_or_else(site)?;
            continue;
        }
        let letter = *ps.get(i).ok_or_else(site)?;
        let mut j = i;
        while j < n && *vowel.get(j).ok_or_else(site)? && *ps.get(j).ok_or_else(site)? == letter {
            j = j.checked_add(1).ok_or_else(site)?;
        }
        let run = j.checked_sub(i).ok_or_else(site)?;
        if (run == 2 && ph.length != 1) || run >= 3 {
            return Ok(Verdict::Reason("long_vowel"));
        }
        nuclei.push((i, run));
        i = j;
    }
    let Some((first, _)) = nuclei.first().copied() else {
        return Ok(Verdict::Reason("no_nucleus"));
    };
    let initial = ps.get(..first).ok_or_else(site)?;
    if initial.len() > ph.onset_max {
        return Ok(Verdict::Reason("onset"));
    }
    if let [c, d] = initial {
        if !ph.cluster_ok(*c, *d).ok_or_else(site)? {
            return Ok(Verdict::Reason("onset"));
        }
    }
    let mut splits: Vec<usize> = Vec::with_capacity(nuclei.len());
    for pair in nuclei.windows(2) {
        let [(start, run), (next_start, _)] = pair else {
            return Err(site());
        };
        let s0 = start.checked_add(*run).ok_or_else(site)?;
        let m = next_start.checked_sub(s0).ok_or_else(site)?;
        if m == 0 {
            splits.push(*next_start);
            continue;
        }
        let cluster = ps.get(s0..*next_start).ok_or_else(site)?;
        let mut chosen: Option<usize> = None;
        for o in (0..=min(m, ph.onset_max)).rev() {
            let cd = m.checked_sub(o).ok_or_else(site)?;
            if cd > ph.coda_max {
                continue;
            }
            if cd != 0 && !ph.coda_ok(*cluster.first().ok_or_else(site)?).ok_or_else(site)? {
                continue;
            }
            if o >= 2 {
                let c = *cluster.get(cd).ok_or_else(site)?;
                let d = *cluster.get(cd.checked_add(1).ok_or_else(site)?).ok_or_else(site)?;
                if !ph.cluster_ok(c, d).ok_or_else(site)? {
                    continue;
                }
            }
            chosen = Some(o);
            break;
        }
        let Some(onset) = chosen else {
            return Ok(Verdict::Reason("cluster"));
        };
        splits.push(s0.checked_add(m).and_then(|end| end.checked_sub(onset)).ok_or_else(site)?);
    }
    let (last_start, last_run) = nuclei.last().copied().ok_or_else(site)?;
    let tail = ps.get(last_start.checked_add(last_run).ok_or_else(site)?..).ok_or_else(site)?;
    if tail.len() > ph.coda_max {
        return Ok(Verdict::Reason("coda"));
    }
    if let [c] = tail {
        if !ph.coda_ok(*c).ok_or_else(site)? {
            return Ok(Verdict::Reason("coda"));
        }
    }
    Ok(Verdict::Licit { syllables: nuclei.len(), splits })
}

// ---------------------------------------------------------------------------
// Draws, invention
// ---------------------------------------------------------------------------

/// An index drawn with probability weight over sum: the first whose running sum exceeds `below(sum)`.
pub fn pick(stream: &mut CountingStream, weights: &[u64]) -> Result<usize, Refused> {
    let site = || defect("pick");
    let mut total: u64 = 0;
    for value in weights {
        total = total.checked_add(*value).ok_or_else(site)?;
    }
    if total < 1 {
        return Err(site());
    }
    let r = stream.below(total).ok_or_else(site)?;
    let mut running: u64 = 0;
    for (index, value) in weights.iter().enumerate() {
        running = running.checked_add(*value).ok_or_else(site)?;
        if running > r {
            return Ok(index);
        }
    }
    Err(site())
}

/// Exact Levenshtein distance 1.
pub fn near1(a: &[u8], b: &[u8]) -> Option<bool> {
    if a.len() == b.len() {
        return Some(a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() == 1);
    }
    let (longer, shorter) = if a.len() > b.len() { (a, b) } else { (b, a) };
    if longer.len().checked_sub(shorter.len())? != 1 {
        return Some(false);
    }
    let q = longer.iter().zip(shorter.iter()).take_while(|(x, y)| x == y).count();
    Some(longer.get(q.checked_add(1)?..)? == shorter.get(q..)?)
}

/// The draw weight of each member, bent by the concept's iconic signs; 0 for a non-member.
fn iconic_weights(ph: &Phonology<'_>, signs: &[i64; 4]) -> Result<[u64; WEIGHTS], Refused> {
    let site = || defect("iconic");
    let mut out = [0_u64; WEIGHTS];
    for (p, slot) in out.iter_mut().enumerate() {
        if !*ph.member.get(p).ok_or_else(site)? {
            continue;
        }
        let row = ph.table.bias.get(p).ok_or_else(site)?;
        let mut sum: i64 = 0;
        for (sign, bias) in signs.iter().zip(row.iter()) {
            sum = sum.checked_add(sign.checked_mul(i64::from(*bias)).ok_or_else(site)?).ok_or_else(site)?;
        }
        let adjust = i64::from(ph.iconicity).checked_mul(sum).and_then(|n| n.checked_div_euclid(256)).ok_or_else(site)?;
        let base = i64::try_from(*ph.weight.get(p).ok_or_else(site)?).map_err(|_| site())?;
        let bent = base.checked_add(adjust).ok_or_else(site)?.max(1);
        *slot = u64::try_from(bent).map_err(|_| site())?;
    }
    Ok(out)
}

fn restricted(weights: &[u64; WEIGHTS], allowed: &[bool; WEIGHTS]) -> [u64; WEIGHTS] {
    let mut out = [0_u64; WEIGHTS];
    for ((slot, weight), keep) in out.iter_mut().zip(weights.iter()).zip(allowed.iter()) {
        if *keep {
            *slot = *weight;
        }
    }
    out
}

fn class_mask(ph: &Phonology<'_>, class: u8) -> Result<[bool; WEIGHTS], Refused> {
    let mut out = [false; WEIGHTS];
    for (p, slot) in out.iter_mut().enumerate() {
        let is_member = *ph.member.get(p).ok_or_else(|| defect("draw"))?;
        *slot = is_member && ph.table.class(p).ok_or_else(|| defect("draw"))? == class;
    }
    Ok(out)
}

fn draw_form(stream: &mut CountingStream, ph: &Phonology<'_>, wi: &[u64; WEIGHTS], syllables: usize) -> Result<Vec<u8>, Refused> {
    let site = || defect("draw");
    let n = if syllables != 0 { syllables } else { pick(stream, &ph.word_len)?.checked_add(1).ok_or_else(site)? };
    let cons = class_mask(ph, 1)?;
    let vowels = class_mask(ph, 0)?;
    let mut letters: Vec<usize> = Vec::new();
    for _ in 0..n {
        let (onset, coda) = *TEMPLATES.get(pick(stream, &ph.templates)?).ok_or_else(site)?;
        if onset == 1 {
            letters.push(pick(stream, &restricted(wi, &cons))?);
        } else if onset == 2 {
            let first = pick(stream, &restricted(wi, &ph.c1set))?;
            // A cluster is two consonants: a vowel's sonority would pass the rise test, so it is excluded.
            let mut second = [false; WEIGHTS];
            for (d, slot) in second.iter_mut().enumerate() {
                *slot = *cons.get(d).ok_or_else(site)? && ph.cluster_ok(first, d).ok_or_else(site)?;
            }
            letters.push(first);
            letters.push(pick(stream, &restricted(wi, &second))?);
        }
        letters.push(pick(stream, &restricted(wi, &vowels))?);
        if coda != 0 {
            letters.push(pick(stream, &restricted(wi, &ph.codaset))?);
        }
    }
    letters.iter().map(|p| ALPHABET.get(*p).copied().ok_or_else(site)).collect()
}

/// One coinage to make: whose, for what, and the entrenched forms it must avoid.
pub struct Coinage<'r> {
    pub seed: [u8; 32],
    pub concept: u64,
    pub coin: u64,
    pub signs: [i64; 4],
    pub epoch: u64,
    pub syllables: usize,
    pub anchored: Vec<&'r str>,
    pub others: Vec<&'r str>,
}

/// `(case output, work)` for one coinage.
pub fn invent_case(ph: &Phonology<'_>, case: &Coinage<'_>, taboo: &Taboo) -> Result<(Value, u64), Refused> {
    let site = || defect("invent");
    let add = |total: u64, n: u64| total.checked_add(n).ok_or_else(site);
    let size = |n: usize| u64::try_from(n).map_err(|_| site());
    let wi = iconic_weights(ph, &case.signs)?;
    let known: BTreeSet<&[u8]> = case.anchored.iter().chain(case.others.iter()).map(|form| form.as_bytes()).collect();
    let mut work = add(add(63, size(case.anchored.len())?)?, size(case.others.len())?)?;
    let mut tries: Vec<Value> = Vec::new();
    for k in 0..INVENT_TRIES {
        let key = rng::key(&case.seed, DOMAIN_INVENT, &[case.concept, case.coin, k, case.epoch]);
        let mut stream = CountingStream::from_key(&key);
        let form = draw_form(&mut stream, ph, &wi, case.syllables)?;
        let letters = size(form.len())?;
        work = add(add(add(work, stream.draws)?, letters)?, 1)?;
        let verdict = licit(&form, ph)?;
        let mut reason: Option<&'static str> = None;
        let mut syllables: usize = 0;
        match verdict {
            Verdict::Reason(code) => reason = Some(code),
            Verdict::Licit { syllables: count, .. } => {
                syllables = count;
                work = add(work, 1)?;
                if known.contains(form.as_slice()) {
                    reason = Some("same");
                } else {
                    work = add(add(work, letters)?, 1)?;
                    let mut near = false;
                    for anchored in &case.anchored {
                        if near1(&form, anchored.as_bytes()).ok_or_else(site)? {
                            near = true;
                            break;
                        }
                    }
                    if near {
                        reason = Some("near");
                    } else {
                        work = add(work, taboo_work(form.len()).ok_or_else(site)?)?;
                        if taboo.hit(&form) {
                            reason = Some("taboo");
                        }
                    }
                }
            }
        }
        let Some(code) = reason else {
            let result = obj(vec![
                (String::from("form"), Value::Str(text_of(&form))),
                (String::from("outcome"), ocj::s("word")),
                (String::from("syllables"), index_value(syllables, "invent")?),
                (String::from("tries"), Value::Arr(tries)),
            ]);
            return Ok((result, work));
        };
        work = add(work, 1)?;
        tries.push(Value::Arr(vec![Value::Str(ocj::hex(&sha256(&form))), ocj::s(code)]));
    }
    let result = obj(vec![(String::from("outcome"), ocj::s("gesture")), (String::from("tries"), Value::Arr(tries))]);
    Ok((result, work))
}

/// `(phoneme, draws)`: one potential phoneme keyed on the seed and the block.
pub fn first_sound(seed: &[u8; 32], ph: &Phonology<'_>) -> Result<(usize, u64), Refused> {
    let site = || defect("first sound");
    let mut stream = CountingStream::from_key(&rng::key(seed, DOMAIN_FIRST, &[]));
    let mut weights = [0_u64; WEIGHTS];
    for (p, slot) in weights.iter_mut().enumerate() {
        if *ph.member.get(p).ok_or_else(site)? && !ph.table.first_exclude.contains(&p) {
            *slot = *ph.weight.get(p).ok_or_else(site)?;
        }
    }
    let p = pick(&mut stream, &weights)?;
    Ok((p, stream.draws))
}

// ---------------------------------------------------------------------------
// SAS words
// ---------------------------------------------------------------------------

/// `(list, candidates drawn, work)`: 2048 distinct sayable forms derived from the block alone.
pub fn sas_list(ph: &Phonology<'_>, taboo: &Taboo) -> Result<(Vec<Vec<u8>>, u64, u64), Refused> {
    let site = || defect("sas");
    let mut prefix: Vec<u8> = Vec::with_capacity(SAS_PREFIX.len().saturating_add(1).saturating_add(LEX_BYTES));
    prefix.extend_from_slice(SAS_PREFIX);
    prefix.push(0);
    prefix.extend_from_slice(&ph.lex);
    let seed = sha256(&prefix);
    let mut stream = CountingStream::from_key(&rng::key(&seed, DOMAIN_SAS, &[]));
    let mut cons = [0_u64; WEIGHTS];
    let mut vowels = [0_u64; WEIGHTS];
    for (p, (c, v)) in cons.iter_mut().zip(vowels.iter_mut()).enumerate() {
        let is_member = *ph.member.get(p).ok_or_else(site)?;
        let class = ph.table.class(p).ok_or_else(site)?;
        if is_member && class == 1 && !ph.table.sas_exclude.contains(&p) {
            *c = 1;
        }
        if is_member && class == 0 {
            *v = 1;
        }
    }
    let mut words: Vec<Vec<u8>> = Vec::with_capacity(SAS_SIZE);
    let mut seen: BTreeSet<Vec<u8>> = BTreeSet::new();
    let mut candidates: u64 = 0;
    let mut taboo_cost: u64 = 0;
    for syllables in SAS_SYLLABLES {
        let mut drawn: u64 = 0;
        while words.len() < SAS_SIZE && drawn < SAS_CAP {
            let mut form: Vec<u8> = Vec::with_capacity(syllables.saturating_mul(2));
            for _ in 0..syllables {
                form.push(*ALPHABET.get(pick(&mut stream, &cons)?).ok_or_else(site)?);
                form.push(*ALPHABET.get(pick(&mut stream, &vowels)?).ok_or_else(site)?);
            }
            drawn = drawn.checked_add(1).ok_or_else(site)?;
            candidates = candidates.checked_add(1).ok_or_else(site)?;
            if seen.contains(&form) {
                continue;
            }
            taboo_cost = taboo_cost.checked_add(taboo_work(form.len()).ok_or_else(site)?).ok_or_else(site)?;
            if taboo.hit(&form) {
                continue;
            }
            seen.insert(form.clone());
            words.push(form);
        }
        if words.len() >= SAS_SIZE {
            break;
        }
    }
    if words.len() < SAS_SIZE {
        return Err(refused("limit", String::from("sas list")));
    }
    let work = stream.draws.checked_add(taboo_cost).ok_or_else(site)?;
    Ok((words, candidates, work))
}

/// The six 11-bit indices a digest's first 66 bits carry, most significant first, and those bits.
pub fn sas_indices(digest: &[u8; 32]) -> Result<([usize; 6], u128), Refused> {
    let site = || defect("sas words");
    let mut x: u128 = 0;
    for byte in digest.iter().take(9) {
        x = x.checked_shl(8).ok_or_else(site)? | u128::from(*byte);
    }
    x = x.checked_shr(6).ok_or_else(site)?;
    let mut out = [0_usize; 6];
    for (j, slot) in out.iter_mut().enumerate() {
        let shift = 5_usize.checked_sub(j).and_then(|n| n.checked_mul(11)).and_then(|n| u32::try_from(n).ok()).ok_or_else(site)?;
        *slot = usize::try_from(x.checked_shr(shift).ok_or_else(site)? & 0x7FF).map_err(|_| site())?;
    }
    Ok((out, x))
}

/// Six typed words back to their indices (-1 for a word not in the list) and, when all are known, the value.
pub fn sas_parse(phrase: &[&str], positions: &BTreeMap<&[u8], usize>) -> Result<Value, Refused> {
    let site = || defect("sas parse");
    let found: Vec<Option<usize>> = phrase.iter().map(|word| positions.get(word.as_bytes()).copied()).collect();
    let mut indices: Vec<Value> = Vec::with_capacity(found.len());
    for index in &found {
        indices.push(match index {
            Some(i) => index_value(*i, "sas parse")?,
            None => Value::Int(-1),
        });
    }
    let mut known: Vec<usize> = Vec::with_capacity(found.len());
    for index in &found {
        match index {
            Some(i) => known.push(*i),
            None => {
                return Ok(obj(vec![(String::from("indices"), Value::Arr(indices)), (String::from("value"), Value::Null)]));
            }
        }
    }
    let mut x: u128 = 0;
    for index in known {
        x = x.checked_shl(11).ok_or_else(site)? | u128::try_from(index).map_err(|_| site())?;
    }
    let bytes = x.to_be_bytes();
    let low = bytes.get(7..).ok_or_else(site)?;
    Ok(obj(vec![(String::from("indices"), Value::Arr(indices)), (String::from("value"), Value::Str(ocj::hex(low)))]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laws;
    use crate::rng::Stream;

    const FUZZ_CASES: u64 = 10_000;
    const HOSTILE: &[u8] = b"aeioupbtdkgq'cjfvszxhmnlrwyaaeeiiuusstt'XZ09-_ ";

    fn fixture_table() -> Option<Table> {
        let law = laws::parse_file(laws::law("fixture")?).ok()?;
        let name = law.get("lang")?.get("phon")?.get("name").and_then(|name| match name {
            Value::Str(text) => Some(text.clone()),
            _ => None,
        })?;
        parse_table(&laws::parse_file(laws::table(&name)?).ok()?)
    }

    fn below(stream: &mut Stream, n: usize) -> Option<usize> {
        usize::try_from(stream.below(u64::try_from(n.max(1)).ok()?)).ok()
    }

    fn block(stream: &mut Stream, table: &Table) -> Option<String> {
        let mut out = Vec::with_capacity(LEX_BYTES);
        for (low, high) in table.lex_box.iter() {
            let span = usize::from(*high).checked_sub(usize::from(*low))?.checked_add(1)?;
            let value = match stream.below(4) {
                0 => *low,
                1 => *high,
                _ => u8::try_from(usize::from(*low).checked_add(below(stream, span)?)?).ok()?,
            };
            out.push(value);
        }
        Some(ocj::hex(&out))
    }

    fn word(stream: &mut Stream) -> Option<String> {
        let mut out = String::new();
        for _ in 0..below(stream, 15)? {
            out.push(char::from(*HOSTILE.get(below(stream, HOSTILE.len())?)?));
        }
        Some(out)
    }

    fn words(stream: &mut Stream, most: usize) -> Option<String> {
        let mut out: Vec<String> = Vec::new();
        for _ in 0..below(stream, most)? {
            out.push(format!("\"{}\"", word(stream)?));
        }
        Some(out.join(","))
    }

    fn seed(stream: &mut Stream) -> String {
        let mut out = String::new();
        for _ in 0..4 {
            out.push_str(&format!("{:016x}", stream.next_u64()));
        }
        out
    }

    fn licit_request(stream: &mut Stream, table: &Table, law: &str) -> Option<String> {
        let forms = words(stream, 6)?;
        let inventory = match stream.below(3) {
            0 => format!(",\"inventory\":{}", stream.below(1 << 27)),
            _ => String::new(),
        };
        let lex = block(stream, table)?;
        Some(format!("{{\"forms\":[{}]{},\"law\":\"{}\",\"lex\":\"{}\",\"op\":\"phon_licit\",\"v\":1}}", forms, inventory, law, lex))
    }

    fn invent_request(stream: &mut Stream, table: &Table, law: &str) -> Option<String> {
        let mut cases: Vec<String> = Vec::new();
        for _ in 0..below(stream, 4)?.checked_add(1)? {
            let anchored = words(stream, 3)?;
            let others = words(stream, 3)?;
            let mut signs: Vec<String> = Vec::with_capacity(4);
            for _ in 0..4 {
                signs.push(format!("{}", i64::try_from(stream.below(3)).ok()?.checked_sub(1)?));
            }
            let inventory = match stream.below(4) {
                0 => format!(",\"inventory\":{}", stream.below(1 << 27)),
                _ => String::new(),
            };
            let (coin, concept, epoch) = (stream.below(3), stream.below(1 << 40), stream.below(1 << 32));
            let lex = block(stream, table)?;
            let seed = seed(stream);
            cases.push(format!(
                "{{\"anchored\":[{}],\"coin\":{},\"concept\":{},\"epoch\":{}{},\"lex\":\"{}\",\"others\":[{}],\"seed\":\"{}\",\"signs\":[{}],\"syllables\":{}}}",
                anchored,
                coin,
                concept,
                epoch,
                inventory,
                lex,
                others,
                seed,
                signs.join(","),
                stream.below(5)
            ));
        }
        // Keys go in byte order, so a budget comes first.
        let budget = match stream.below(5) {
            0 => format!("\"budget\":{},", stream.below(4000)),
            _ => String::new(),
        };
        Some(format!("{{{}\"cases\":[{}],\"law\":\"{}\",\"op\":\"phon_invent\",\"v\":1}}", budget, cases.join(","), law))
    }

    #[test]
    fn ten_thousand_seeded_licit_and_invent_requests_are_each_answered_in_canonical_json() {
        let table = fixture_table();
        assert!(table.is_some(), "the fixture law pins a sound phonology table");
        let Some(table) = table else { return };
        let mut stream = Stream::new(&[0_u8; 32], "test.phon.fuzz", 0);
        let (mut answered, mut refused_count, mut words, mut gestures) = (0_u64, 0_u64, 0_u64, 0_u64);
        for case in 0..FUZZ_CASES {
            let law = if stream.below(2) == 0 { "fixture" } else { "v0_1" };
            let request = if case & 1 == 0 { licit_request(&mut stream, &table, law) } else { invent_request(&mut stream, &table, law) };
            assert!(request.is_some(), "case {} could not be built", case);
            let Some(request) = request else { return };
            let answer = crate::call(request.as_bytes());
            let value = ocj::parse(&answer, false);
            assert!(value.is_ok(), "case {} answered outside OCJ: {}", case, request);
            let Ok(value) = value else { return };
            assert!(matches!(value, Value::Obj(_)), "case {} answered a non-object", case);
            if let Some(code) = value.get("refused") {
                assert!(code != &ocj::s("engine_panic"), "case {} is an engine defect: {}", case, request);
                refused_count = refused_count.saturating_add(1);
                continue;
            }
            answered = answered.saturating_add(1);
            if let Some(Value::Arr(out)) = value.get("out") {
                for item in out {
                    match item.get("outcome") {
                        Some(Value::Str(outcome)) if outcome == "word" => words = words.saturating_add(1),
                        Some(Value::Str(_)) => gestures = gestures.saturating_add(1),
                        _ => {}
                    }
                }
            }
        }
        assert!(answered > 0 && refused_count > 0, "both answers are reached: {} answered, {} refused", answered, refused_count);
        assert!(words > 0, "inventions coin words: {} words, {} gestures", words, gestures);
    }

    fn table_value() -> Option<Value> {
        laws::parse_file(laws::table("phon_v1")?).ok()
    }

    /// The cell at `row`, `column` of the list of lists under `key`.
    fn cell_mut<'a>(value: &'a mut Value, key: &str, row: usize, column: usize) -> Option<&'a mut Value> {
        let Value::Obj(members) = value else {
            return None;
        };
        let (_, rows) = members.iter_mut().find(|(name, _)| name == key)?;
        let Value::Arr(rows) = rows else {
            return None;
        };
        let Value::Arr(cells) = rows.get_mut(row)? else {
            return None;
        };
        cells.get_mut(column)
    }

    fn refused_with(key: &str, row: usize, column: usize, replacement: Value) -> Option<bool> {
        let mut value = table_value()?;
        *cell_mut(&mut value, key, row, column)? = replacement;
        Some(parse_table(&value).is_none())
    }

    #[test]
    fn the_validator_refuses_a_manner_past_seven_a_vowel_code_box_past_four_and_a_boolean_template() {
        let sound = table_value().map(|value| parse_table(&value).is_some());
        assert_eq!(sound, Some(true), "the embedded table is sound");
        assert_eq!(refused_with("features", 5, 2, Value::Int(8)), Some(true), "a manner of 8");
        assert_eq!(refused_with("lex_box", 52, 1, Value::Int(30)), Some(true), "a reduction code box up to 30");
        assert_eq!(refused_with("lex_box", 53, 1, Value::Int(5)), Some(true), "an epenthetic code box up to 5");
        assert_eq!(refused_with("templates", 1, 0, Value::Bool(true)), Some(true), "a boolean template");
        assert_eq!(refused_with("templates", 0, 0, Value::Bool(false)), Some(true), "a false template");
        // A change that stays sound is accepted: the refusals above are the checks, not the mutation.
        assert_eq!(refused_with("lex_box", 52, 1, Value::Int(3)), Some(false), "a reduction code box up to 3");
    }

    #[test]
    fn near_one_is_exact_levenshtein_distance_one() {
        assert_eq!(near1(b"ab", b"ba"), Some(false));
        assert_eq!(near1(b"abd", b"abcd"), Some(true));
        assert_eq!(near1(b"abd", b"abce"), Some(false));
        assert_eq!(near1(b"abc", b"abcd"), Some(true));
    }

    #[test]
    fn the_taboo_work_is_the_published_count() {
        assert_eq!(taboo_work(6), Some(10));
        assert_eq!(taboo_work(8), Some(21));
        assert_eq!(taboo_work(12), Some(46));
        assert_eq!(taboo_work(3), Some(1));
    }
}
