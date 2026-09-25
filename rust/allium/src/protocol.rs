//! The byte protocol, wire v1: the twin of `opti_oignon/allium/ref/protocol.py`.
//!
//! `call` answers every request with the bytes the reference answers, and
//! the order of the checks is part of that: it decides which refusal a
//! request with several defects gets.
//!
//! The genome operations read the embedded law and pool files, parsed and
//! validated at every call (the reference remembers them per process; the
//! answers are the same). `decode` and `compile` never read the pool.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use crate::fx::{self, Work};
use crate::laws;
use crate::ocj::{self, obj, refused, s, Refused, Value};
use crate::organs::compile as organ_compile;
use crate::organs::genome::{self, View};
use crate::organs::phon::{self, Taboo};
use crate::rng;

pub const ENGINE_VERSION: &str = "0.1.0";
pub const WIRE_VERSION: i64 = 1;

const LIMIT_BODY: usize = 4096;
const LIMIT_INPUT: usize = 1 << 20;
const LIMIT_ITEMS: usize = 100_000;
const LIMIT_STATE: usize = 1 << 19;

const OPS: [&str; 19] = [
    "bulk",
    "echo",
    "engine",
    "fact_id",
    "fx",
    "genome_compile",
    "genome_corner",
    "genome_decode",
    "genome_found",
    "law",
    "phon_first_sound",
    "phon_invent",
    "phon_inventory",
    "phon_lex",
    "phon_licit",
    "phon_sas",
    "phon_table",
    "phon_taboo",
    "rng",
];
const RNG_KINDS: [&str; 5] = ["below", "key", "noise", "stream", "unit"];
const FACT_FIELDS: [&str; 7] = ["being", "body", "kind", "laws", "origin", "oseq", "t"];

type Answer = Result<Value, Refused>;

fn bad(code: &'static str, detail: &str) -> Refused {
    refused(code, String::from(detail))
}

fn int(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Int(n)) => Some(*n),
        _ => None,
    }
}

fn text(value: Option<&Value>) -> Option<&str> {
    match value {
        Some(Value::Str(t)) => Some(t.as_str()),
        _ => None,
    }
}

/// A key present with a null value reads as absent, as `dict.get` does in the reference.
fn present(value: Option<&Value>) -> Option<&Value> {
    match value {
        Some(Value::Null) | None => None,
        Some(other) => Some(other),
    }
}

fn fields(request: &Value, required: &[&str], optional: &[&str]) -> Result<(), Refused> {
    for name in request.keys() {
        if !required.contains(&name) && !optional.contains(&name) {
            return Err(bad("bad_request", "fields"));
        }
    }
    for name in required {
        if request.get(name).is_none() {
            return Err(bad("bad_request", "fields"));
        }
    }
    Ok(())
}

fn wide(n: u64) -> Value {
    Value::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

fn int_value(n: i128) -> Value {
    Value::Int(i64::try_from(n).unwrap_or(i64::MAX))
}

fn engine() -> Answer {
    let mut law_members = Vec::new();
    for (name, text) in laws::LAWS {
        let law = laws::parse_file(text)?;
        let digest = laws::digest(&law)?;
        law_members.push((
            String::from(name),
            obj(vec![
                (String::from("digest"), Value::Str(ocj::hex(&digest))),
                (String::from("provisional"), law.get("provisional").cloned().unwrap_or(Value::Null)),
                (String::from("version"), law.get("version").cloned().unwrap_or(Value::Null)),
            ]),
        ));
    }
    let mut table_members = Vec::new();
    for (name, text) in laws::TABLES {
        let table = laws::parse_file(text)?;
        table_members.push((String::from(name), Value::Str(ocj::hex(&laws::digest(&table)?))));
    }
    let mut founder_members = Vec::new();
    for (name, text) in laws::FOUNDERS {
        let pool = laws::parse_file(text)?;
        founder_members.push((String::from(name), Value::Str(ocj::hex(&laws::digest(&pool)?))));
    }
    let limit = |n: usize| Value::Int(i64::try_from(n).unwrap_or(i64::MAX));
    let mut domains: Vec<&str> = genome::DOMAINS.iter().chain(phon::DOMAINS.iter()).copied().collect();
    domains.sort_unstable();
    Ok(obj(vec![
        (String::from("domains"), Value::Arr(domains.iter().map(|domain| s(domain)).collect())),
        (String::from("engine"), s(ENGINE_VERSION)),
        (String::from("founders"), obj(founder_members)),
        (String::from("genome_schema"), Value::Int(genome::SCHEMA)),
        (String::from("laws"), obj(law_members)),
        (
            String::from("limits"),
            obj(vec![
                (String::from("body"), limit(LIMIT_BODY)),
                (String::from("depth"), limit(ocj::MAX_DEPTH)),
                (String::from("input"), limit(LIMIT_INPUT)),
                (String::from("items"), limit(LIMIT_ITEMS)),
                (String::from("state"), limit(LIMIT_STATE)),
                (String::from("steps"), int_value(fx::STEPS_MAX)),
            ]),
        ),
        (String::from("ops"), Value::Arr(OPS.iter().map(|op| s(op)).collect())),
        (String::from("refusals"), Value::Arr(ocj::REFUSALS.iter().map(|code| s(code)).collect())),
        (String::from("tables"), obj(table_members)),
        (String::from("wire"), Value::Int(WIRE_VERSION)),
    ]))
}

/// The engine's identity as OCJ bytes: what the handshake compares.
pub fn engine_info() -> Vec<u8> {
    match engine().and_then(|value| ocj::emit(&value)) {
        Ok(bytes) => bytes,
        Err(refusal) => refusal_bytes(&refusal),
    }
}

fn op_echo(request: &Value) -> Answer {
    fields(request, &["doc", "op", "v"], &[])?;
    let doc = request.get("doc").cloned().unwrap_or(Value::Null);
    if ocj::emit(&doc)?.len() > LIMIT_STATE {
        return Err(bad("limit", "state size"));
    }
    Ok(obj(vec![(String::from("doc"), doc)]))
}

fn op_fx(request: &Value) -> Answer {
    fields(request, &["args", "fn", "op", "v"], &["budget", "table"])?;
    let name = text(request.get("fn")).unwrap_or("");
    let arity = match fx::arity(name) {
        Some(arity) if matches!(request.get("fn"), Some(Value::Str(_))) => arity,
        _ => return Err(bad("unknown_op", "fx function")),
    };
    let mut table: Vec<i128> = Vec::new();
    if name == "lut" {
        let entries = match request.get("table") {
            Some(Value::Arr(entries)) if entries.len() == 257 => entries,
            _ => return Err(bad("bad_request", "table")),
        };
        for entry in entries {
            match entry {
                Value::Int(n) if i128::from(*n) >= fx::I32_MIN && i128::from(*n) <= fx::I32_MAX => {
                    table.push(i128::from(*n))
                }
                _ => return Err(bad("bad_request", "table")),
            }
        }
    } else if request.get("table").is_some() {
        return Err(bad("bad_request", "fields"));
    }
    let budget = match present(request.get("budget")) {
        None => None,
        Some(Value::Int(n)) if *n >= 0 => Some(*n),
        Some(_) => return Err(bad("bad_request", "budget")),
    };
    let args = match request.get("args") {
        Some(Value::Arr(args)) => args,
        _ => return Err(bad("bad_request", "args")),
    };
    if args.len() > LIMIT_ITEMS {
        return Err(bad("limit", "items"));
    }
    let mut rows: Vec<Vec<i128>> = Vec::with_capacity(args.len());
    let mut steps: i128 = 0;
    for item in args {
        let values = match item {
            Value::Arr(values) if values.len() == arity => values,
            _ => return Err(bad("bad_request", "args")),
        };
        let mut row = Vec::with_capacity(arity);
        for value in values {
            match value {
                Value::Int(n) => row.push(i128::from(*n)),
                _ => return Err(bad("bad_request", "args")),
            }
        }
        if name == "decay_iter" {
            let requested = row.get(2).copied().unwrap_or(0);
            steps = steps.saturating_add(requested.clamp(0, fx::STEPS_MAX));
        }
        rows.push(row);
    }
    if steps > fx::STEPS_MAX {
        return Err(bad("limit", "steps"));
    }
    let sine = if name == "sin_b" || name == "cos_b" { laws::sine()? } else { Vec::new() };
    let mut work = Work::default();
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let a = |i: usize| row.get(i).copied().unwrap_or(0);
        let result = match name {
            "mul" => fx::mul(a(0), a(1), &mut work),
            "div" => fx::div(a(0), a(1), &mut work),
            "pow" => fx::pow(a(0), a(1), &mut work),
            "hill_up" => fx::hill_up(a(0), a(1), a(2), &mut work),
            "hill_down" => fx::hill_down(a(0), a(1), a(2), &mut work),
            "mm" => fx::mm(a(0), a(1), &mut work),
            "sig" => fx::sig(a(0), &mut work),
            "sat" => fx::sat(a(0), a(1), a(2), &mut work),
            "isqrt" => fx::isqrt(a(0), &mut work),
            "sin_b" => fx::sin_b(a(0), &sine, &mut work),
            "cos_b" => fx::cos_b(a(0), &sine, &mut work),
            "lut" => fx::lut(a(0), &table, &mut work),
            "decay_iter" => fx::decay_iter(a(0), a(1), a(2), &mut work),
            _ => fx::decay_lazy(a(0), a(1), a(2), a(3), &mut work),
        };
        out.push(int_value(result));
    }
    if let Some(budget) = budget {
        if work.units > u64::try_from(budget).unwrap_or(0) {
            return Err(bad("budget", "fx"));
        }
    }
    Ok(obj(vec![
        (String::from("alarm"), wide(work.alarm)),
        (String::from("out"), Value::Arr(out)),
        (String::from("work"), wide(work.units)),
    ]))
}

fn op_rng(request: &Value) -> Answer {
    let kind_value = present(request.get("kind"));
    if text(kind_value) == Some("below") {
        fields(request, &["bound", "domain", "index", "kind", "n", "op", "seed", "v"], &[])?;
    } else {
        fields(request, &["domain", "index", "kind", "n", "op", "seed", "v"], &[])?;
    }
    let kind = match text(kind_value) {
        Some(kind) if RNG_KINDS.contains(&kind) => kind,
        _ => return Err(bad("bad_request", "kind")),
    };
    let seed_text = match text(request.get("seed")) {
        Some(seed) if ocj::is_hex(seed, 64) => seed,
        _ => return Err(bad("bad_request", "seed")),
    };
    let domain = match text(request.get("domain")) {
        Some(domain) if !domain.is_empty() => domain,
        _ => return Err(bad("bad_request", "domain")),
    };
    let index = match int(request.get("index")) {
        Some(index) if index >= 0 => u64::try_from(index).unwrap_or(0),
        _ => return Err(bad("bad_request", "index")),
    };
    let count = match int(request.get("n")) {
        Some(n) if n >= 1 => usize::try_from(n).unwrap_or(usize::MAX),
        _ => return Err(bad("bad_request", "n")),
    };
    if count > LIMIT_ITEMS {
        return Err(bad("limit", "items"));
    }
    if kind == "key" && count != 1 {
        return Err(bad("bad_request", "n"));
    }
    let mut seed = [0_u8; 32];
    let decoded = ocj::from_hex(seed_text).unwrap_or_default();
    if decoded.len() == 32 {
        seed.copy_from_slice(&decoded);
    }
    let units = Value::Int(i64::try_from(count).unwrap_or(i64::MAX));
    if kind == "key" {
        let key = rng::key(&seed, domain, &[index]);
        return Ok(obj(vec![
            (String::from("out"), Value::Arr(vec![Value::Str(ocj::hex(&key))])),
            (String::from("work"), Value::Int(1)),
        ]));
    }
    let mut out = Vec::with_capacity(count);
    if kind == "noise" {
        let k = rng::first_word(&rng::key(&seed, domain, &[index]));
        let mut counter: u64 = 0;
        while out.len() < count {
            out.push(Value::Str(format!("{:016x}", rng::noise(k, counter))));
            counter = counter.saturating_add(1);
        }
        return Ok(obj(vec![(String::from("out"), Value::Arr(out)), (String::from("work"), units)]));
    }
    let mut stream = rng::Stream::new(&seed, domain, index);
    if kind == "stream" {
        while out.len() < count {
            out.push(Value::Str(format!("{:016x}", stream.next_u64())));
        }
    } else if kind == "unit" {
        while out.len() < count {
            out.push(wide(stream.unit()));
        }
    } else {
        let bound = match int(request.get("bound")) {
            Some(bound) if bound >= 1 => u64::try_from(bound).unwrap_or(1),
            _ => return Err(bad("bad_request", "bound")),
        };
        while out.len() < count {
            out.push(wide(stream.below(bound)));
        }
    }
    Ok(obj(vec![(String::from("out"), Value::Arr(out)), (String::from("work"), units)]))
}

fn op_bulk(request: &Value) -> Answer {
    if request.get("pack").is_some() {
        fields(request, &["op", "pack", "v"], &[])?;
        let pack = match request.get("pack") {
            Some(pack @ Value::Obj(_)) if pack.keys() == ["type", "values"] => pack,
            _ => return Err(bad("bad_request", "pack")),
        };
        let values = match pack.get("values") {
            Some(Value::Arr(values)) if values.len() <= LIMIT_ITEMS => values,
            _ => return Err(bad("bad_request", "pack")),
        };
        let kind = match pack.get("type") {
            Some(Value::Str(kind)) => kind,
            _ => return Err(bad("bad_request", "pack")),
        };
        let text = ocj::pack_bulk(kind, values)?;
        return Ok(obj(vec![(String::from("text"), Value::Str(text))]));
    }
    fields(request, &["op", "unpack", "v"], &[])?;
    let (kind, values) = ocj::unpack_bulk(request.get("unpack").unwrap_or(&Value::Null))?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if value > i128::from(ocj::MAX_INT) || value < i128::from(ocj::MAX_INT).saturating_neg() {
            return Err(bad("limit", "integer range"));
        }
        out.push(int_value(value));
    }
    Ok(obj(vec![(String::from("type"), Value::Str(kind)), (String::from("values"), Value::Arr(out))]))
}

fn op_fact_id(request: &Value) -> Answer {
    fields(request, &["fact", "op", "v"], &[])?;
    let fact = match request.get("fact") {
        Some(fact @ Value::Obj(_)) => fact,
        _ => return Err(bad("bad_fact", "fact")),
    };
    for name in fact.keys() {
        if !FACT_FIELDS.contains(&name) {
            return Err(bad("bad_fact", "fields"));
        }
    }
    for name in FACT_FIELDS {
        if fact.get(name).is_none() {
            return Err(bad("bad_fact", "fields"));
        }
    }
    if !matches!(text(fact.get("being")), Some(being) if ocj::is_hex(being, 32)) {
        return Err(bad("bad_fact", "being"));
    }
    let kind_ok = match text(fact.get("kind")) {
        Some(kind) => {
            (1..=32).contains(&kind.len())
                && kind.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        }
        None => false,
    };
    if !kind_ok {
        return Err(bad("bad_fact", "kind"));
    }
    if !matches!(int(fact.get("laws")), Some(laws) if laws >= 0) {
        return Err(bad("bad_fact", "laws"));
    }
    if !matches!(text(fact.get("origin")), Some(origin) if ocj::is_hex(origin, 16)) {
        return Err(bad("bad_fact", "origin"));
    }
    if !matches!(int(fact.get("oseq")), Some(oseq) if oseq >= 0) {
        return Err(bad("bad_fact", "oseq"));
    }
    if int(fact.get("t")).is_none() {
        return Err(bad("bad_fact", "t"));
    }
    let body = match fact.get("body") {
        Some(body @ Value::Obj(_)) => body,
        _ => return Err(bad("bad_fact", "body")),
    };
    let body_bytes = ocj::emit(body)?;
    if body_bytes.len() > LIMIT_BODY {
        return Err(bad("limit", "body size"));
    }
    let body_digest = ocj::hex(&Sha256::digest(&body_bytes));
    let mut members = Vec::with_capacity(FACT_FIELDS.len());
    for name in FACT_FIELDS {
        let value = if name == "body" {
            Value::Str(body_digest.clone())
        } else {
            fact.get(name).cloned().unwrap_or(Value::Null)
        };
        members.push((String::from(name), value));
    }
    let envelope = ocj::emit(&obj(members))?;
    Ok(obj(vec![
        (String::from("body"), Value::Str(body_digest)),
        (String::from("eid"), Value::Str(ocj::hex(&Sha256::digest(&envelope)))),
    ]))
}

fn op_law(request: &Value) -> Answer {
    fields(request, &["name", "op", "v"], &[])?;
    let name = match text(request.get("name")) {
        Some(name) if laws::law(name).is_some() => name,
        _ => return Err(bad("unknown_law", "law")),
    };
    let law = laws::parse_file(laws::law(name).unwrap_or(""))?;
    Ok(obj(vec![
        (String::from("digest"), Value::Str(ocj::hex(&laws::digest(&law)?))),
        (String::from("name"), s(name)),
        (String::from("provisional"), law.get("provisional").cloned().unwrap_or(Value::Null)),
        (String::from("version"), law.get("version").cloned().unwrap_or(Value::Null)),
    ]))
}

/// The requested law and its codec view; refused by name when unsound.
fn genome_law(request: &Value) -> Result<(Value, View), Refused> {
    let file = match text(request.get("law")).and_then(laws::law) {
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

/// The law's founder pool, checked against the law's pin, then validated.
fn genome_pool(law: &Value, lawview: &View) -> Result<BTreeMap<i64, Vec<genome::Allele>>, Refused> {
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

fn genome_bytes(request: &Value, lawview: &View) -> Result<Vec<u8>, Refused> {
    let hex = match request.get("genome") {
        Some(Value::Str(hex)) => hex,
        _ => return Err(bad("bad_request", "genome hex")),
    };
    if let Some(cap) = lawview.max_bytes.checked_mul(2) {
        if hex.len() > cap {
            return Err(bad("limit", "genome size"));
        }
    }
    if !genome::is_genome_hex(hex) {
        return Err(bad("bad_request", "genome hex"));
    }
    ocj::from_hex(hex).ok_or_else(|| bad("bad_request", "genome hex"))
}

fn count(n: u64, detail: &str) -> Result<Value, Refused> {
    i64::try_from(n).map(Value::Int).map_err(|_| bad("engine_panic", detail))
}

fn op_genome_found(request: &Value) -> Answer {
    fields(request, &["law", "op", "seed", "v"], &[])?;
    let (law, lawview) = genome_law(request)?;
    let alleles = genome_pool(&law, &lawview)?;
    let seed_text = match text(request.get("seed")) {
        Some(seed) if ocj::is_hex(seed, 64) => seed,
        _ => return Err(bad("bad_request", "seed")),
    };
    let seed = ocj::from_hex(seed_text)
        .and_then(|bytes| <[u8; 32]>::try_from(bytes).ok())
        .ok_or_else(|| bad("bad_request", "seed"))?;
    let (data, chosen, work) = genome::found(&seed, &lawview, &alleles)?;
    let chosen: Vec<Value> = chosen.into_iter().map(Value::Int).collect();
    Ok(obj(vec![
        (String::from("alleles"), Value::Str(ocj::pack_bulk("u8", &chosen)?)),
        (String::from("genome"), Value::Str(ocj::hex(&data))),
        (String::from("sha256"), Value::Str(genome::sha256_hex(&data))),
        (String::from("work"), count(work, "genome found")?),
    ]))
}

fn op_genome_corner(request: &Value) -> Answer {
    fields(request, &["corner", "law", "op", "v"], &[])?;
    let (_, lawview) = genome_law(request)?;
    let k = match int(request.get("corner")) {
        Some(k) if (0..=genome::CORNER_MAX).contains(&k) => k,
        _ => return Err(bad("bad_request", "corner")),
    };
    let (data, work) = genome::corner(k, &lawview)?;
    Ok(obj(vec![
        (String::from("genome"), Value::Str(ocj::hex(&data))),
        (String::from("sha256"), Value::Str(genome::sha256_hex(&data))),
        (String::from("work"), count(work, "genome corner")?),
    ]))
}

fn op_genome_decode(request: &Value) -> Answer {
    fields(request, &["genome", "law", "op", "v"], &[])?;
    let (_, lawview) = genome_law(request)?;
    let data = genome_bytes(request, &lawview)?;
    let chromosomes = genome::decode(&data, &lawview)?;
    let mut records: u64 = 0;
    let mut counts = Vec::with_capacity(chromosomes.len());
    for chrom in &chromosomes {
        let n = u64::try_from(chrom.len()).map_err(|_| bad("engine_panic", "genome decode"))?;
        counts.push(count(n, "genome decode")?);
        records = records.checked_add(n).ok_or_else(|| bad("engine_panic", "genome decode"))?;
    }
    let work = genome::blocks(data.len())
        .and_then(|blocks| blocks.checked_add(records))
        .ok_or_else(|| bad("engine_panic", "genome decode"))?;
    Ok(obj(vec![
        (String::from("chromosomes"), Value::Arr(counts)),
        (String::from("sha256"), Value::Str(genome::sha256_hex(&data))),
        (String::from("work"), count(work, "genome decode")?),
    ]))
}

fn op_genome_compile(request: &Value) -> Answer {
    fields(request, &["genome", "law", "op", "v"], &[])?;
    let (law, lawview) = genome_law(request)?;
    let data = genome_bytes(request, &lawview)?;
    // Only the tables carry the law's digest; the other operations never compute it.
    let digest = ocj::hex(&laws::digest(&law)?);
    let (tables, work) = organ_compile::compile_genome(&data, &lawview, &digest)?;
    let tables_digest = genome::sha256_hex(&ocj::emit(&tables)?);
    Ok(obj(vec![
        (String::from("sha256"), Value::Str(tables_digest)),
        (String::from("tables"), tables),
        (String::from("work"), count(work, "genome compile")?),
    ]))
}

// ---------------------------------------------------------------------------
// Phonology
// ---------------------------------------------------------------------------

/// The phonology and taboo tables a law pins, each checked against its pin.
struct Lang {
    table: phon::Table,
    value: Value,
    taboo: Vec<(usize, [u8; 32])>,
    phon_digest: String,
    taboo_digest: String,
}

/// A pinned table: exactly `{name, sha256}`, a name the pin may carry, the file's canonical digest.
fn pin(lang: &Value, key: &str, names: &[&str], detail: &str) -> Result<(Value, String), Refused> {
    let refuse = || bad("unknown_law", detail);
    let pin = match lang.get(key) {
        Some(pin @ Value::Obj(_)) if pin.keys() == ["name", "sha256"] => pin,
        _ => return Err(refuse()),
    };
    let file = match text(pin.get("name")) {
        Some(name) if names.contains(&name) => laws::table(name).ok_or_else(refuse)?,
        _ => return Err(refuse()),
    };
    let value = laws::parse_file(file)?;
    let digest = ocj::hex(&laws::digest(&value)?);
    if text(pin.get("sha256")) != Some(digest.as_str()) {
        return Err(refuse());
    }
    Ok((value, digest))
}

/// The phonology and taboo tables the requested law pins; refused by name when unsound.
fn phon_law(request: &Value) -> Result<Lang, Refused> {
    let file = match text(request.get("law")).and_then(laws::law) {
        Some(file) => file,
        None => return Err(bad("unknown_law", "law")),
    };
    let law = laws::parse_file(file)?;
    let lang = match law.get("lang") {
        Some(lang @ Value::Obj(_)) if lang.keys() == ["phon", "taboo"] => lang,
        _ => return Err(bad("unknown_law", "phon digest")),
    };
    let (value, phon_digest) = pin(lang, "phon", &laws::PHON_TABLES, "phon digest")?;
    let table = phon::parse_table(&value).ok_or_else(|| bad("unknown_law", "phon table"))?;
    let (taboo_value, taboo_digest) = pin(lang, "taboo", &laws::TABOO_TABLES, "taboo digest")?;
    let taboo = phon::parse_taboo(&taboo_value, &table).ok_or_else(|| bad("unknown_law", "taboo table"))?;
    Ok(Lang { table, value, taboo, phon_digest, taboo_digest })
}

fn size(n: usize, detail: &str) -> Result<Value, Refused> {
    i64::try_from(n).map(Value::Int).map_err(|_| bad("engine_panic", detail))
}

fn lex_of(value: Option<&Value>, table: &phon::Table) -> Result<[u8; phon::LEX_BYTES], Refused> {
    let refuse = || bad("bad_request", "lex");
    let hex = match value {
        Some(Value::Str(hex)) if hex.len() == phon::LEX_BYTES.saturating_mul(2) && genome::is_genome_hex(hex) => hex,
        _ => return Err(refuse()),
    };
    let block = ocj::from_hex(hex).and_then(|bytes| <[u8; phon::LEX_BYTES]>::try_from(bytes).ok()).ok_or_else(refuse)?;
    if !phon::lex_ok(&block, table) {
        return Err(refuse());
    }
    Ok(block)
}

fn seed_of(value: Option<&Value>) -> Result<[u8; 32], Refused> {
    let refuse = || bad("bad_request", "seed");
    match value {
        Some(Value::Str(hex)) if ocj::is_hex(hex, 64) => {
            ocj::from_hex(hex).and_then(|bytes| <[u8; 32]>::try_from(bytes).ok()).ok_or_else(refuse)
        }
        _ => Err(refuse()),
    }
}

/// A membership mask: 27 bits, at least one vowel and one consonant.
fn inventory_of(value: Option<&Value>) -> Result<u32, Refused> {
    match int(value) {
        Some(mask)
            if (0..(1_i64 << phon::WEIGHTS)).contains(&mask)
                && mask & 0x1F != 0
                && mask.checked_shr(5).is_some_and(|consonants| consonants != 0) =>
        {
            u32::try_from(mask).map_err(|_| bad("bad_request", "inventory"))
        }
        _ => Err(bad("bad_request", "inventory")),
    }
}

fn taboo_extra(request: &Value, table: &phon::Table) -> Result<Vec<(usize, [u8; 32])>, Refused> {
    let Some(extra) = request.get("taboo_extra") else {
        return Ok(Vec::new());
    };
    let Value::Arr(entries) = extra else {
        return Err(bad("bad_request", "taboo"));
    };
    if entries.len() > table.taboo_extra_max {
        return Err(bad("limit", "taboo"));
    }
    entries.iter().map(|entry| phon::taboo_pair(entry).ok_or_else(|| bad("bad_request", "taboo"))).collect()
}

fn list<'a>(request: &'a Value, key: &str) -> Result<&'a Vec<Value>, Refused> {
    let Some(Value::Arr(items)) = request.get(key) else {
        return Err(bad("bad_request", key));
    };
    if items.len() > LIMIT_ITEMS {
        return Err(bad("limit", "items"));
    }
    Ok(items)
}

fn forms<'a>(value: Option<&'a Value>, name: &str, table: &phon::Table) -> Result<Vec<&'a str>, Refused> {
    let Some(Value::Arr(items)) = value else {
        return Err(bad("bad_request", name));
    };
    if items.len() > table.anchored_max {
        return Err(bad("limit", name));
    }
    items
        .iter()
        .map(|item| match item {
            Value::Str(form) if phon::is_form(form, phon::FORM_MAX) => Ok(form.as_str()),
            _ => Err(bad("bad_request", name)),
        })
        .collect()
}

fn op_phon_table(request: &Value) -> Answer {
    fields(request, &["law", "op", "v"], &[])?;
    let lang = phon_law(request)?;
    let table = &lang.table;
    let detail = "phon table";
    let echo = |key: &str| lang.value.get(key).cloned().ok_or_else(|| bad("engine_panic", detail));
    let mut classify: Vec<Value> = Vec::with_capacity(256);
    for byte in 0..=u8::MAX {
        classify.push(match phon::index_of(byte) {
            Some(p) => size(p, detail)?,
            None => Value::Int(-1),
        });
    }
    let mut lengths: Vec<usize> = lang.taboo.iter().map(|(length, _)| *length).collect();
    lengths.sort_unstable();
    lengths.dedup();
    let lengths = lengths.iter().map(|length| size(*length, detail)).collect::<Result<Vec<Value>, Refused>>()?;
    let features = table
        .features
        .iter()
        .map(|row| Value::Arr(row.iter().map(|cell| Value::Int(i64::from(*cell))).collect()))
        .collect();
    let bias = table.bias.iter().map(|row| Value::Arr(row.iter().map(|cell| Value::Int(i64::from(*cell))).collect())).collect();
    let lex_box = table
        .lex_box
        .iter()
        .map(|(low, high)| Value::Arr(vec![Value::Int(i64::from(*low)), Value::Int(i64::from(*high))]))
        .collect();
    let fold = table.fold.iter().map(|(code, replacement)| Value::Arr(vec![Value::Int(*code), Value::Str(replacement.clone())])).collect();
    let first_exclude = table.first_exclude.iter().map(|p| size(*p, detail)).collect::<Result<Vec<Value>, Refused>>()?;
    let work = phon::WEIGHTS.checked_add(256).ok_or_else(|| bad("engine_panic", detail))?;
    Ok(obj(vec![
        (String::from("alphabet"), echo("alphabet")?),
        (String::from("anchored_max"), size(table.anchored_max, detail)?),
        (String::from("anchored_total_max"), size(table.anchored_total_max, detail)?),
        (String::from("bias"), Value::Arr(bias)),
        (String::from("classify"), Value::Str(ocj::pack_bulk("i8", &classify)?)),
        (String::from("features"), Value::Arr(features)),
        (String::from("first_sound_exclude"), Value::Arr(first_exclude)),
        (String::from("floor_consonants"), echo("floor_consonants")?),
        (String::from("floor_vowels"), echo("floor_vowels")?),
        (String::from("fold"), Value::Arr(fold)),
        (String::from("form_max"), echo("form_max")?),
        (String::from("invent_tries"), echo("invent_tries")?),
        (String::from("lex_box"), Value::Arr(lex_box)),
        (String::from("phon"), Value::Str(lang.phon_digest.clone())),
        (String::from("potential_min"), Value::Int(i64::from(table.potential_min))),
        (String::from("sas"), echo("sas")?),
        (String::from("shown_max"), echo("shown_max")?),
        (
            String::from("taboo"),
            obj(vec![
                (String::from("entries"), size(lang.taboo.len(), detail)?),
                (String::from("lengths"), Value::Arr(lengths)),
                (String::from("sha256"), Value::Str(lang.taboo_digest.clone())),
            ]),
        ),
        (String::from("taboo_extra_max"), size(table.taboo_extra_max, detail)?),
        (String::from("taboo_max"), size(table.taboo_max, detail)?),
        (String::from("taboo_window"), echo("taboo_window")?),
        (String::from("templates"), echo("templates")?),
        (String::from("work"), size(work, detail)?),
    ]))
}

fn op_phon_lex(request: &Value) -> Answer {
    fields(request, &["law", "op", "v"], &["genome", "seed"])?;
    // Presence is the key's, whatever its value: a null genome is a genome given.
    let by_genome = request.get("genome").is_some();
    if by_genome == request.get("seed").is_some() {
        return Err(bad("bad_request", "fields"));
    }
    let lang = phon_law(request)?;
    let lex_width = u64::try_from(phon::LEX_BYTES).map_err(|_| bad("engine_panic", "lex block"))?;
    if by_genome {
        let (law, lawview) = genome_law(request)?;
        let data = genome_bytes(request, &lawview)?;
        let digest = ocj::hex(&laws::digest(&law)?);
        let (tables, work) = organ_compile::compile_genome(&data, &lawview, &digest)?;
        let block = phon::lex_from_tables(&tables)?;
        if !phon::lex_ok(&block, &lang.table) {
            return Err(bad("engine_panic", "lex block"));
        }
        let work = work.checked_add(lex_width).ok_or_else(|| bad("engine_panic", "lex block"))?;
        return Ok(obj(vec![
            (String::from("lex"), Value::Str(ocj::hex(&block))),
            (String::from("source"), s("genome")),
            (String::from("work"), count(work, "lex block")?),
        ]));
    }
    let seed = seed_of(request.get("seed"))?;
    let (block, draws) = phon::fallback_lex(&seed, &lang.table)?;
    Ok(obj(vec![
        (String::from("lex"), Value::Str(ocj::hex(&block))),
        (String::from("source"), s("seed")),
        (String::from("work"), count(draws, "phon fallback")?),
    ]))
}

fn op_phon_inventory(request: &Value) -> Answer {
    fields(request, &["law", "lex", "op", "v"], &[])?;
    let lang = phon_law(request)?;
    let blocks = list(request, "lex")?
        .iter()
        .map(|value| lex_of(Some(value), &lang.table))
        .collect::<Result<Vec<[u8; phon::LEX_BYTES]>, Refused>>()?;
    let mut out = Vec::with_capacity(blocks.len());
    for block in &blocks {
        out.push(phon::phonology_value(&phon::decode(block, &lang.table, None)?)?);
    }
    let work = blocks.len().checked_mul(phon::LEX_BYTES).ok_or_else(|| bad("engine_panic", "phon inventory"))?;
    Ok(obj(vec![(String::from("out"), Value::Arr(out)), (String::from("work"), size(work, "phon inventory")?)]))
}

fn op_phon_licit(request: &Value) -> Answer {
    fields(request, &["forms", "law", "lex", "op", "v"], &["inventory"])?;
    let lang = phon_law(request)?;
    let block = lex_of(request.get("lex"), &lang.table)?;
    let inventory = match request.get("inventory") {
        Some(value) => Some(inventory_of(Some(value))?),
        None => None,
    };
    let items = list(request, "forms")?;
    let mut forms: Vec<&str> = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Value::Str(form) => forms.push(form.as_str()),
            _ => return Err(bad("bad_request", "forms")),
        }
    }
    let ph = phon::decode(&block, &lang.table, inventory)?;
    let detail = "phon licit";
    let mut out = Vec::with_capacity(forms.len());
    let mut work = phon::LEX_BYTES;
    for form in forms {
        out.push(match phon::licit(form.as_bytes(), &ph)? {
            phon::Verdict::Reason(code) => obj(vec![(String::from("reason"), s(code))]),
            phon::Verdict::Licit { syllables, splits } => obj(vec![
                (
                    String::from("splits"),
                    Value::Arr(splits.iter().map(|at| size(*at, detail)).collect::<Result<Vec<Value>, Refused>>()?),
                ),
                (String::from("syllables"), size(syllables, detail)?),
            ]),
        });
        work = work.checked_add(form.len()).and_then(|n| n.checked_add(1)).ok_or_else(|| bad("engine_panic", detail))?;
    }
    Ok(obj(vec![(String::from("out"), Value::Arr(out)), (String::from("work"), size(work, detail)?)]))
}

const CASE_KEYS: [&str; 9] = ["anchored", "coin", "concept", "epoch", "lex", "others", "seed", "signs", "syllables"];

/// One validated invention case: its block, its override and its coinage.
struct Case<'r> {
    block: [u8; phon::LEX_BYTES],
    inventory: Option<u32>,
    coinage: phon::Coinage<'r>,
}

fn invent_case_of<'r>(case: &'r Value, table: &phon::Table) -> Result<Case<'r>, Refused> {
    let shaped = matches!(case, Value::Obj(_))
        && CASE_KEYS.iter().all(|key| case.get(key).is_some())
        && case.keys().iter().all(|key| CASE_KEYS.contains(key) || *key == "inventory");
    if !shaped {
        return Err(bad("bad_request", "case fields"));
    }
    let block = lex_of(case.get("lex"), table)?;
    let seed = seed_of(case.get("seed"))?;
    let mut ids = [0_u64; 2];
    for (key, slot) in ["concept", "coin"].iter().zip(ids.iter_mut()) {
        *slot = match int(case.get(key)) {
            Some(n) if (0..=ocj::MAX_INT).contains(&n) => u64::try_from(n).map_err(|_| bad("bad_request", key))?,
            _ => return Err(bad("bad_request", key)),
        };
    }
    let [concept, coin] = ids;
    let mut signs = [0_i64; 4];
    match case.get("signs") {
        Some(Value::Arr(items)) if items.len() == 4 => {
            for (item, slot) in items.iter().zip(signs.iter_mut()) {
                *slot = match item {
                    Value::Int(sign) if (-1..=1).contains(sign) => *sign,
                    _ => return Err(bad("bad_request", "signs")),
                };
            }
        }
        _ => return Err(bad("bad_request", "signs")),
    }
    let epoch = match int(case.get("epoch")) {
        Some(n) if (0..=0xFFFF_FFFF).contains(&n) => u64::try_from(n).map_err(|_| bad("bad_request", "epoch"))?,
        _ => return Err(bad("bad_request", "epoch")),
    };
    let syllables = match int(case.get("syllables")) {
        Some(n) if (0..=3).contains(&n) => usize::try_from(n).map_err(|_| bad("bad_request", "syllables"))?,
        _ => return Err(bad("bad_request", "syllables")),
    };
    let inventory = match case.get("inventory") {
        Some(value) => Some(inventory_of(Some(value))?),
        None => None,
    };
    let anchored = forms(case.get("anchored"), "anchored", table)?;
    let others = forms(case.get("others"), "others", table)?;
    Ok(Case { block, inventory, coinage: phon::Coinage { seed, concept, coin, signs, epoch, syllables, anchored, others } })
}

fn op_phon_invent(request: &Value) -> Answer {
    fields(request, &["cases", "law", "op", "v"], &["budget", "taboo_extra"])?;
    let lang = phon_law(request)?;
    let table = &lang.table;
    // A null budget reads as absent, as in `fx`.
    let budget = match present(request.get("budget")) {
        None => None,
        Some(Value::Int(n)) if *n >= 0 => Some(u64::try_from(*n).map_err(|_| bad("bad_request", "budget"))?),
        Some(_) => return Err(bad("bad_request", "budget")),
    };
    let extra = taboo_extra(request, table)?;
    let cases = list(request, "cases")?;
    let detail = "phon invent";
    let mut checked: Vec<Case<'_>> = Vec::with_capacity(cases.len());
    let mut lists: usize = 0;
    for case in cases {
        let item = invent_case_of(case, table)?;
        lists = lists
            .checked_add(item.coinage.anchored.len())
            .and_then(|n| n.checked_add(item.coinage.others.len()))
            .ok_or_else(|| bad("engine_panic", detail))?;
        if lists > table.anchored_total_max {
            return Err(bad("limit", "lists"));
        }
        checked.push(item);
    }
    let taboo = Taboo::new(&lang.taboo, &extra);
    let mut out = Vec::with_capacity(checked.len());
    let mut work: u64 = 0;
    for item in &checked {
        let ph = phon::decode(&item.block, table, item.inventory)?;
        let (result, cost) = phon::invent_case(&ph, &item.coinage, &taboo)?;
        work = work.checked_add(cost).ok_or_else(|| bad("engine_panic", detail))?;
        if budget.is_some_and(|budget| work > budget) {
            return Err(bad("budget", detail));
        }
        out.push(result);
    }
    Ok(obj(vec![(String::from("out"), Value::Arr(out)), (String::from("work"), count(work, detail)?)]))
}

fn op_phon_first_sound(request: &Value) -> Answer {
    fields(request, &["cases", "law", "op", "v"], &[])?;
    let lang = phon_law(request)?;
    let mut checked: Vec<([u8; phon::LEX_BYTES], [u8; 32])> = Vec::new();
    for case in list(request, "cases")? {
        if !matches!(case, Value::Obj(_)) || case.keys() != ["lex", "seed"] {
            return Err(bad("bad_request", "case fields"));
        }
        let block = lex_of(case.get("lex"), &lang.table)?;
        checked.push((block, seed_of(case.get("seed"))?));
    }
    let detail = "phon first sound";
    let lex_width = u64::try_from(phon::LEX_BYTES).map_err(|_| bad("engine_panic", detail))?;
    let mut out = Vec::with_capacity(checked.len());
    let mut symbols = String::with_capacity(checked.len());
    let mut work: u64 = 0;
    for (block, seed) in &checked {
        let (p, draws) = phon::first_sound(seed, &phon::decode(block, &lang.table, None)?)?;
        out.push(size(p, detail)?);
        symbols.push(char::from(*phon::ALPHABET.get(p).ok_or_else(|| bad("engine_panic", detail))?));
        work = work.checked_add(lex_width).and_then(|n| n.checked_add(draws)).ok_or_else(|| bad("engine_panic", detail))?;
    }
    Ok(obj(vec![
        (String::from("out"), Value::Arr(out)),
        (String::from("symbols"), Value::Str(symbols)),
        (String::from("work"), count(work, detail)?),
    ]))
}

fn op_phon_sas(request: &Value) -> Answer {
    fields(request, &["law", "lex", "op", "v"], &["digests", "phrases", "taboo_extra"])?;
    let lang = phon_law(request)?;
    let block = lex_of(request.get("lex"), &lang.table)?;
    let extra = taboo_extra(request, &lang.table)?;
    let digests = match request.get("digests") {
        None => None,
        Some(_) => {
            let mut out: Vec<[u8; 32]> = Vec::new();
            for digest in list(request, "digests")? {
                match digest {
                    Value::Str(hex) if ocj::is_hex(hex, 64) => out.push(
                        ocj::from_hex(hex)
                            .and_then(|bytes| <[u8; 32]>::try_from(bytes).ok())
                            .ok_or_else(|| bad("bad_request", "digests"))?,
                    ),
                    _ => return Err(bad("bad_request", "digests")),
                }
            }
            Some(out)
        }
    };
    let phrases = match request.get("phrases") {
        None => None,
        Some(_) => {
            let mut out: Vec<Vec<&str>> = Vec::new();
            for phrase in list(request, "phrases")? {
                let words: Option<Vec<&str>> = match phrase {
                    Value::Arr(words) if words.len() == 6 => words
                        .iter()
                        .map(|word| match word {
                            Value::Str(word) if phon::is_form(word, phon::FORM_MAX) => Some(word.as_str()),
                            _ => None,
                        })
                        .collect(),
                    _ => None,
                };
                out.push(words.ok_or_else(|| bad("bad_request", "phrases"))?);
            }
            Some(out)
        }
    };
    let taboo = Taboo::new(&lang.taboo, &extra);
    let (words, candidates, mut work) = phon::sas_list(&phon::decode(&block, &lang.table, None)?, &taboo)?;
    let detail = "phon sas";
    let panic_sas = || bad("engine_panic", detail);
    let listed: Vec<Value> = words.iter().map(|word| Value::Str(word.iter().map(|byte| char::from(*byte)).collect())).collect();
    let digest = genome::sha256_hex(&ocj::emit(&Value::Arr(listed.clone()))?);
    let mut members = vec![
        (String::from("candidates"), count(candidates, detail)?),
        (String::from("list"), Value::Arr(listed.clone())),
        (String::from("sha256"), Value::Str(digest)),
    ];
    if let Some(digests) = digests {
        let mut indices = Vec::with_capacity(digests.len());
        let mut rendered = Vec::with_capacity(digests.len());
        for digest in &digests {
            let (row, _) = phon::sas_indices(digest)?;
            indices.push(Value::Arr(row.iter().map(|index| size(*index, detail)).collect::<Result<Vec<Value>, Refused>>()?));
            rendered.push(Value::Arr(
                row.iter().map(|index| listed.get(*index).cloned().ok_or_else(panic_sas)).collect::<Result<Vec<Value>, Refused>>()?,
            ));
        }
        members.push((String::from("indices"), Value::Arr(indices)));
        members.push((String::from("words"), Value::Arr(rendered)));
        let cost = u64::try_from(digests.len()).ok().and_then(|n| n.checked_mul(6)).ok_or_else(panic_sas)?;
        work = work.checked_add(cost).ok_or_else(panic_sas)?;
    }
    if let Some(phrases) = phrases {
        let mut positions: BTreeMap<&[u8], usize> = BTreeMap::new();
        for (index, word) in words.iter().enumerate() {
            positions.insert(word.as_slice(), index);
        }
        let mut parsed = Vec::with_capacity(phrases.len());
        for phrase in &phrases {
            parsed.push(phon::sas_parse(phrase, &positions)?);
        }
        members.push((String::from("parsed"), Value::Arr(parsed)));
        let cost = u64::try_from(phrases.len()).ok().and_then(|n| n.checked_mul(6)).ok_or_else(panic_sas)?;
        work = work.checked_add(cost).ok_or_else(panic_sas)?;
    }
    members.push((String::from("work"), count(work, detail)?));
    Ok(obj(members))
}

fn op_phon_taboo(request: &Value) -> Answer {
    fields(request, &["law", "op", "strings", "v"], &["taboo_extra"])?;
    let lang = phon_law(request)?;
    let extra = taboo_extra(request, &lang.table)?;
    let mut strings: Vec<&str> = Vec::new();
    for item in list(request, "strings")? {
        match item {
            Value::Str(text) if phon::is_form(text, phon::SHOWN_MAX) => strings.push(text.as_str()),
            _ => return Err(bad("bad_request", "strings")),
        }
    }
    let taboo = Taboo::new(&lang.taboo, &extra);
    let detail = "phon taboo";
    let out: Vec<Value> = strings.iter().map(|text| Value::Bool(taboo.hit(text.as_bytes()))).collect();
    let mut work: u64 = 0;
    for text in &strings {
        let cost = phon::taboo_work(text.len()).ok_or_else(|| bad("engine_panic", detail))?;
        work = work.checked_add(cost).ok_or_else(|| bad("engine_panic", detail))?;
    }
    Ok(obj(vec![(String::from("out"), Value::Arr(out)), (String::from("work"), count(work, detail)?)]))
}

fn answer(data: &[u8]) -> Answer {
    if data.len() > LIMIT_INPUT {
        return Err(bad("limit", "input size"));
    }
    let request = ocj::parse(data, false)?;
    if !matches!(request, Value::Obj(_)) {
        return Err(bad("bad_request", "request"));
    }
    let op = match text(request.get("op")) {
        Some(op) => op,
        None => return Err(bad("bad_request", "op")),
    };
    if int(request.get("v")) != Some(WIRE_VERSION) {
        return Err(bad("unknown_op", "wire version"));
    }
    if !OPS.contains(&op) {
        return Err(bad("unknown_op", "op"));
    }
    match op {
        "bulk" => op_bulk(&request),
        "echo" => op_echo(&request),
        "engine" => {
            fields(&request, &["op", "v"], &[])?;
            engine()
        }
        "fact_id" => op_fact_id(&request),
        "fx" => op_fx(&request),
        "genome_compile" => op_genome_compile(&request),
        "genome_corner" => op_genome_corner(&request),
        "genome_decode" => op_genome_decode(&request),
        "genome_found" => op_genome_found(&request),
        "law" => op_law(&request),
        "phon_first_sound" => op_phon_first_sound(&request),
        "phon_invent" => op_phon_invent(&request),
        "phon_inventory" => op_phon_inventory(&request),
        "phon_lex" => op_phon_lex(&request),
        "phon_licit" => op_phon_licit(&request),
        "phon_sas" => op_phon_sas(&request),
        "phon_table" => op_phon_table(&request),
        "phon_taboo" => op_phon_taboo(&request),
        "rng" => op_rng(&request),
        _ => Err(bad("unknown_op", "op")),
    }
}

fn refusal_bytes(refusal: &Refused) -> Vec<u8> {
    let value = obj(vec![
        (String::from("detail"), Value::Str(refusal.detail.clone())),
        (String::from("refused"), s(refusal.code)),
    ]);
    ocj::emit(&value).unwrap_or_default()
}

/// Answer one request: OCJ bytes in, OCJ bytes out, never a panic by design.
pub fn call(data: &[u8]) -> Vec<u8> {
    match answer(data).and_then(|value| ocj::emit(&value)) {
        Ok(bytes) => bytes,
        Err(refusal) => refusal_bytes(&refusal),
    }
}
