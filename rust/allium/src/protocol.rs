//! The byte protocol, wire v1: the twin of `opti_oignon/allium/ref/protocol.py`.
//!
//! `call` answers every request with the bytes the reference answers, and
//! the order of the checks is part of that: it decides which refusal a
//! request with several defects gets.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use crate::fx::{self, Work};
use crate::laws;
use crate::ocj::{self, obj, refused, s, Refused, Value};
use crate::rng;

pub const ENGINE_VERSION: &str = "0.1.0";
pub const WIRE_VERSION: i64 = 1;

const LIMIT_BODY: usize = 4096;
const LIMIT_INPUT: usize = 1 << 20;
const LIMIT_ITEMS: usize = 100_000;
const LIMIT_STATE: usize = 1 << 19;

const OPS: [&str; 7] = ["bulk", "echo", "engine", "fact_id", "fx", "law", "rng"];
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
    let limit = |n: usize| Value::Int(i64::try_from(n).unwrap_or(i64::MAX));
    Ok(obj(vec![
        (String::from("engine"), s(ENGINE_VERSION)),
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
        "law" => op_law(&request),
        _ => op_rng(&request),
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
