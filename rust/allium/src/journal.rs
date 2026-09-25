#![deny(clippy::indexing_slicing, clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented)]
//! The identity of a fact, the twin of `opti_oignon/allium/ref/journal.py`.
//!
//! A fact is `{being, body, kind, laws, origin, oseq, t}` with its body an
//! object. Its envelope is the same object with the body replaced by the
//! SHA-256 hex of the body's canonical bytes, and its event id (eid) is the
//! SHA-256 hex of the envelope's canonical bytes. Nothing local enters the
//! envelope, so a fact has the same eid on every device that holds it.
//!
//! Every function here answers as its Python reference answers, refusal
//! codes and details included, and checks in the same order: the first
//! defect met decides the refusal. Nothing here indexes, unwraps or panics.
//!
//! `check_body` checks a body against its kind's schema in the embedded
//! kinds table. The table's own soundness is the reference's alone: the
//! handshake proves the embedded table by its digest, so a spec this module
//! cannot read is refused `unknown_law "journal"`, defensively.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use crate::ocj::{self, obj, refused, Refused, Value};
use crate::protocol::{bad, int, text};

pub const FACT_FIELDS: [&str; 7] = ["being", "body", "kind", "laws", "origin", "oseq", "t"];
pub const BODY_LIMIT: usize = 4096;

/// Refuse, by name, a fact that is not well formed; nothing else happens.
///
/// The key set comes first, then `being`, `kind`, `laws`, `origin`, `oseq`
/// and `t`, then the body: an object when `digest_body` is false, a
/// lowercase hex64 digest when it is true.
pub fn check_envelope(fact: &Value, digest_body: bool) -> Result<(), Refused> {
    if !matches!(fact, Value::Obj(_)) {
        return Err(bad("bad_fact", "fact"));
    }
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
    let body_ok = if digest_body {
        matches!(text(fact.get("body")), Some(body) if ocj::is_hex(body, 64))
    } else {
        matches!(fact.get("body"), Some(Value::Obj(_)))
    };
    if !body_ok {
        return Err(bad("bad_fact", "body"));
    }
    Ok(())
}

/// The SHA-256 hex of a body's canonical bytes; a body over `BODY_LIMIT` bytes is refused.
pub fn body_digest(body: &Value) -> Result<String, Refused> {
    let body_bytes = ocj::emit(body)?;
    if body_bytes.len() > BODY_LIMIT {
        return Err(bad("limit", "body size"));
    }
    Ok(ocj::hex(&Sha256::digest(&body_bytes)))
}

/// The eid of a checked fact: the SHA-256 hex of its envelope, its body replaced by `digest`.
pub fn eid(fact: &Value, digest: &str) -> Result<String, Refused> {
    let mut members = Vec::with_capacity(FACT_FIELDS.len());
    for name in FACT_FIELDS {
        let value = if name == "body" {
            Value::Str(String::from(digest))
        } else {
            fact.get(name).cloned().unwrap_or(Value::Null)
        };
        members.push((String::from(name), value));
    }
    let envelope = ocj::emit(&obj(members))?;
    Ok(ocj::hex(&Sha256::digest(&envelope)))
}

/// A fact's body digest and eid, `{body, eid}`; a malformed fact is refused by name.
pub fn fact_id(fact: &Value) -> Result<Value, Refused> {
    check_envelope(fact, false)?;
    let digest = body_digest(fact.get("body").unwrap_or(&Value::Null))?;
    let id = eid(fact, &digest)?;
    Ok(obj(vec![(String::from("body"), Value::Str(digest)), (String::from("eid"), Value::Str(id))]))
}

/// The eids of a list of envelopes, `{eids}` in their order.
///
/// Every envelope is checked, left to right, before any eid is computed, so
/// one malformed envelope refuses the whole list.
pub fn fact_envelope(envelopes: &[Value]) -> Result<Value, Refused> {
    for envelope in envelopes {
        check_envelope(envelope, true)?;
    }
    let mut eids = Vec::with_capacity(envelopes.len());
    for envelope in envelopes {
        eids.push(Value::Str(eid(envelope, text(envelope.get("body")).unwrap_or(""))?));
    }
    Ok(obj(vec![(String::from("eids"), Value::Arr(eids))]))
}

fn refuse_body(path: &str, problem: &str) -> Refused {
    if path.is_empty() {
        refused("bad_fact", format!("body {}", problem))
    } else {
        refused("bad_fact", format!("body {} {}", path, problem))
    }
}

fn unreadable_table() -> Refused {
    bad("unknown_law", "journal")
}

/// Printable ASCII, 1..=`most` characters, no space at either end.
fn is_text(value: &str, most: i64) -> bool {
    let length = i64::try_from(value.len()).unwrap_or(i64::MAX);
    (1..=most).contains(&length)
        && value.bytes().all(|b| (0x20..=0x7E).contains(&b))
        && !value.starts_with(' ')
        && !value.ends_with(' ')
}

fn check_value(spec: &Value, value: &Value, path: &str) -> Result<(), Refused> {
    match text(spec.get("type")) {
        Some("int") => {
            let (Some(lo), Some(hi)) = (int(spec.get("lo")), int(spec.get("hi"))) else {
                return Err(unreadable_table());
            };
            let Value::Int(number) = value else {
                return Err(refuse_body(path, "int"));
            };
            if !(lo..=hi).contains(number) {
                return Err(refuse_body(path, "range"));
            }
        }
        Some("bool") => {
            if !matches!(value, Value::Bool(_)) {
                return Err(refuse_body(path, "bool"));
            }
        }
        Some("symbol") => {
            let Some(Value::Arr(members)) = spec.get("of") else {
                return Err(unreadable_table());
            };
            let known = match value {
                Value::Str(given) => members.iter().any(|member| matches!(member, Value::Str(m) if m == given)),
                _ => false,
            };
            if !known {
                return Err(refuse_body(path, "symbol"));
            }
        }
        Some("hex") => {
            let Some(length) = int(spec.get("len")).and_then(|n| usize::try_from(n).ok()) else {
                return Err(unreadable_table());
            };
            if !matches!(value, Value::Str(given) if ocj::is_hex(given, length)) {
                return Err(refuse_body(path, "hex"));
            }
        }
        Some("text") => {
            let Some(most) = int(spec.get("max")) else {
                return Err(unreadable_table());
            };
            if !matches!(value, Value::Str(given) if is_text(given, most)) {
                return Err(refuse_body(path, "text"));
            }
        }
        Some("object") => {
            let Some(fields) = spec.get("fields") else {
                return Err(unreadable_table());
            };
            check_body(fields, value, path)?;
        }
        _ => return Err(unreadable_table()),
    }
    Ok(())
}

/// Refuse a body that is not an object with exactly the schema's keys and valid values.
///
/// The detail reads `body <path> <problem>`, for example
/// `body laws.sha256 hex`. Unknown keys, then missing ones, are `fields`;
/// then each field, in the order of its name, is checked against its spec.
pub fn check_body(schema: &Value, body: &Value, path: &str) -> Result<(), Refused> {
    let Value::Obj(specs) = schema else {
        return Err(unreadable_table());
    };
    let Value::Obj(given) = body else {
        return Err(refuse_body(path, "object"));
    };
    for (name, _) in given {
        if schema.get(name).is_none() {
            return Err(refuse_body(path, "fields"));
        }
    }
    for (name, _) in specs {
        if body.get(name).is_none() {
            return Err(refuse_body(path, "fields"));
        }
    }
    let mut ordered: Vec<&(String, Value)> = specs.iter().collect();
    ordered.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    for (name, spec) in ordered {
        let inner = if path.is_empty() { name.clone() } else { format!("{}.{}", path, name) };
        check_value(spec, body.get(name).unwrap_or(&Value::Null), &inner)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Stream;

    const FUZZ_CASES: usize = 5_000;
    const GOLDEN_BODY: &str = "2e23a73a0d01ba27f7d814553b7572b21b115624c5e682488d1b70f080e0192f";
    const GOLDEN_EID: &str = "b1e8fbdce5df4fbc2bce7261fb945d2a33ae0bd5d5c539fe1cff7edcf5585d3a";
    const KIND_ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789_";
    const DETAILS: [&str; 9] = ["being", "body", "fact", "fields", "kind", "laws", "origin", "oseq", "t"];

    fn member(name: &str, value: Value) -> (String, Value) {
        (String::from(name), value)
    }

    fn pick(stream: &mut Stream, n: usize) -> usize {
        let bound = u64::try_from(n.max(1)).unwrap_or(1);
        usize::try_from(stream.below(bound)).unwrap_or(0)
    }

    fn hex_of(stream: &mut Stream, bytes: usize) -> String {
        let mut raw = Vec::with_capacity(bytes);
        while raw.len() < bytes {
            raw.push(u8::try_from(stream.below(256)).unwrap_or(0));
        }
        ocj::hex(&raw)
    }

    fn natural(stream: &mut Stream) -> i64 {
        match stream.below(3) {
            0 => 0,
            1 => ocj::MAX_INT,
            _ => {
                let span = u64::try_from(ocj::MAX_INT).unwrap_or(0).saturating_add(1);
                i64::try_from(stream.below(span)).unwrap_or(0)
            }
        }
    }

    fn signed(stream: &mut Stream) -> i64 {
        let n = natural(stream);
        if stream.below(2) == 0 {
            n
        } else {
            n.saturating_neg()
        }
    }

    fn kind_of(stream: &mut Stream, length: usize) -> String {
        let mut out = String::with_capacity(length);
        while out.len() < length {
            out.push(char::from(*KIND_ALPHABET.get(pick(stream, KIND_ALPHABET.len())).unwrap_or(&b'_')));
        }
        out
    }

    /// A string whose one letter (or first digit) is replaced by an upper-case letter.
    fn upper_once(stream: &mut Stream, value: &str) -> String {
        let letters: Vec<usize> =
            value.bytes().enumerate().filter(|(_, b)| b.is_ascii_lowercase()).map(|(at, _)| at).collect();
        let target = letters.get(pick(stream, letters.len())).copied().unwrap_or(0);
        value
            .bytes()
            .enumerate()
            .map(|(at, b)| {
                if at != target {
                    char::from(b)
                } else if b.is_ascii_lowercase() {
                    char::from(b.to_ascii_uppercase())
                } else {
                    'A'
                }
            })
            .collect()
    }

    fn resized(stream: &mut Stream, value: &str) -> String {
        if stream.below(2) == 0 {
            value.chars().skip(1).collect()
        } else {
            format!("{}0", value)
        }
    }

    fn not_text(stream: &mut Stream) -> Value {
        match stream.below(4) {
            0 => Value::Int(1),
            1 => Value::Null,
            2 => Value::Bool(true),
            _ => Value::Arr(Vec::new()),
        }
    }

    fn not_int(stream: &mut Stream) -> Value {
        match stream.below(4) {
            0 => Value::Str(String::from("0")),
            1 => Value::Null,
            2 => Value::Bool(false),
            _ => Value::Obj(Vec::new()),
        }
    }

    /// A well-formed fact with a body object, drawn from the stream.
    fn fact(stream: &mut Stream) -> Vec<(String, Value)> {
        let length = pick(stream, 32).saturating_add(1);
        let body = obj(vec![member("n", Value::Int(signed(stream))), member("w", Value::Str(hex_of(stream, 4)))]);
        vec![
            member("being", Value::Str(hex_of(stream, 16))),
            member("body", body),
            member("kind", Value::Str(kind_of(stream, length))),
            member("laws", Value::Int(natural(stream))),
            member("origin", Value::Str(hex_of(stream, 8))),
            member("oseq", Value::Int(natural(stream))),
            member("t", Value::Int(signed(stream))),
        ]
    }

    fn with(members: &[(String, Value)], name: &str, value: Value) -> Value {
        obj(members.iter().map(|(key, old)| (key.clone(), if key == name { value.clone() } else { old.clone() })).collect())
    }

    fn str_at<'a>(value: &'a Value, key: &str) -> &'a str {
        text(value.get(key)).unwrap_or("")
    }

    /// An envelope, well formed or broken on purpose, and the refusal detail it must draw.
    fn envelope(stream: &mut Stream, members: &[(String, Value)], digest: &str) -> (Value, Option<&'static str>) {
        let base = with(members, "body", Value::Str(String::from(digest)));
        match stream.below(16) {
            0 => {
                let dropped = FACT_FIELDS.get(pick(stream, FACT_FIELDS.len())).copied().unwrap_or("t");
                let kept = match &base {
                    Value::Obj(all) => all.iter().filter(|(key, _)| key != dropped).cloned().collect(),
                    _ => Vec::new(),
                };
                (obj(kept), Some("fields"))
            }
            1 => {
                let mut all = match &base {
                    Value::Obj(all) => all.clone(),
                    _ => Vec::new(),
                };
                all.push(member("seq", Value::Int(1)));
                (obj(all), Some("fields"))
            }
            2 => (not_text(stream), Some("fact")),
            3 => {
                let broken = match stream.below(3) {
                    0 => Value::Str(upper_once(stream, str_at(&base, "being"))),
                    1 => Value::Str(resized(stream, str_at(&base, "being"))),
                    _ => not_text(stream),
                };
                (with(&rebuilt(&base), "being", broken), Some("being"))
            }
            4 => {
                let broken = match stream.below(5) {
                    0 => Value::Str(String::new()),
                    1 => Value::Str(kind_of(stream, 33)),
                    2 => Value::Str(format!("{}A", kind_of(stream, 3))),
                    3 => Value::Str(format!("{}.{}", kind_of(stream, 3), kind_of(stream, 2))),
                    _ => not_text(stream),
                };
                (with(&rebuilt(&base), "kind", broken), Some("kind"))
            }
            5 => {
                let broken = match stream.below(2) {
                    0 => Value::Int(natural(stream).saturating_add(1).min(ocj::MAX_INT).saturating_neg()),
                    _ => not_int(stream),
                };
                (with(&rebuilt(&base), "laws", broken), Some("laws"))
            }
            6 => {
                let broken = match stream.below(3) {
                    0 => Value::Str(upper_once(stream, str_at(&base, "origin"))),
                    1 => Value::Str(resized(stream, str_at(&base, "origin"))),
                    _ => not_text(stream),
                };
                (with(&rebuilt(&base), "origin", broken), Some("origin"))
            }
            7 => {
                let broken = match stream.below(2) {
                    0 => Value::Int(natural(stream).saturating_add(1).min(ocj::MAX_INT).saturating_neg()),
                    _ => not_int(stream),
                };
                (with(&rebuilt(&base), "oseq", broken), Some("oseq"))
            }
            8 => (with(&rebuilt(&base), "t", not_int(stream)), Some("t")),
            9 => {
                let broken = match stream.below(4) {
                    0 => Value::Str(upper_once(stream, digest)),
                    1 => Value::Str(resized(stream, digest)),
                    2 => members.iter().find(|(key, _)| key == "body").map(|(_, body)| body.clone()).unwrap_or(Value::Null),
                    _ => not_text(stream),
                };
                (with(&rebuilt(&base), "body", broken), Some("body"))
            }
            _ => (base, None),
        }
    }

    fn rebuilt(base: &Value) -> Vec<(String, Value)> {
        match base {
            Value::Obj(all) => all.clone(),
            _ => Vec::new(),
        }
    }

    fn ask(request: &Value) -> Option<Value> {
        let bytes = ocj::emit(request).ok()?;
        ocj::parse(&crate::call(&bytes), false).ok()
    }

    fn fact_id_of(members: &[(String, Value)]) -> Option<(String, String)> {
        let answer = ask(&obj(vec![
            member("fact", obj(members.to_vec())),
            member("op", Value::Str(String::from("fact_id"))),
            member("v", Value::Int(1)),
        ]))?;
        Some((String::from(text(answer.get("body"))?), String::from(text(answer.get("eid"))?)))
    }

    fn envelopes_request(envelopes: Vec<Value>) -> Value {
        obj(vec![
            member("envelopes", Value::Arr(envelopes)),
            member("op", Value::Str(String::from("fact_envelope"))),
            member("v", Value::Int(1)),
        ])
    }

    fn refusal(code: &str, detail: &str) -> Value {
        obj(vec![member("detail", Value::Str(String::from(detail))), member("refused", Value::Str(String::from(code)))])
    }

    #[test]
    fn the_golden_fact_and_its_redacted_envelope_have_one_eid() {
        let members = vec![
            member("being", Value::Str(String::from("0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f"))),
            member("body", obj(vec![member("note", Value::Str(String::from("a first word"))), member("turn", Value::Int(3))])),
            member("kind", Value::Str(String::from("sow"))),
            member("laws", Value::Int(0)),
            member("origin", Value::Str(String::from("a1a1a1a1a1a1a1a1"))),
            member("oseq", Value::Int(0)),
            member("t", Value::Int(0)),
        ];
        assert_eq!(fact_id_of(&members), Some((String::from(GOLDEN_BODY), String::from(GOLDEN_EID))));
        let redacted = with(&members, "body", Value::Str(String::from(GOLDEN_BODY)));
        let answer = ask(&envelopes_request(vec![redacted.clone(), redacted]));
        let expected = obj(vec![member("eids", Value::Arr(vec![Value::Str(String::from(GOLDEN_EID)); 2]))]);
        assert_eq!(answer, Some(expected));
        assert_eq!(ask(&envelopes_request(Vec::new())), Some(obj(vec![member("eids", Value::Arr(Vec::new()))])));
    }

    #[test]
    fn a_malformed_list_is_refused_before_any_envelope_is_read() {
        let op = || member("op", Value::Str(String::from("fact_envelope")));
        let not_a_list = obj(vec![member("envelopes", Value::Obj(Vec::new())), op(), member("v", Value::Int(1))]);
        assert_eq!(ask(&not_a_list), Some(refusal("bad_request", "envelopes")));
        let extra = obj(vec![member("envelopes", Value::Arr(Vec::new())), member("more", Value::Int(1)), op(), member("v", Value::Int(1))]);
        assert_eq!(ask(&extra), Some(refusal("bad_request", "fields")));
        let missing = obj(vec![op(), member("v", Value::Int(1))]);
        assert_eq!(ask(&missing), Some(refusal("bad_request", "fields")));
        let over = envelopes_request(vec![Value::Int(0); 100_001]);
        assert_eq!(ask(&over), Some(refusal("limit", "items")));
        let at = envelopes_request(vec![Value::Int(0); 100_000]);
        assert_eq!(ask(&at), Some(refusal("bad_fact", "fact")), "at the limit the envelopes are read");
    }

    #[test]
    fn five_thousand_mutated_envelopes_are_each_answered_as_the_reference_orders_its_checks() {
        let mut stream = Stream::new(&[0_u8; 32], "test.journal.fuzz", 0);
        let mut seen: Vec<&str> = Vec::new();
        let (mut accepted, mut refused_count, mut built) = (0_usize, 0_usize, 0_usize);
        while built < FUZZ_CASES {
            let count = pick(&mut stream, 4).saturating_add(1);
            let mut envelopes = Vec::with_capacity(count);
            let mut eids = Vec::with_capacity(count);
            let mut first_defect: Option<&str> = None;
            for _ in 0..count {
                let members = fact(&mut stream);
                let known = fact_id_of(&members);
                assert!(known.is_some(), "a drawn fact is well formed");
                let Some((digest, id)) = known else { return };
                let (value, defect) = envelope(&mut stream, &members, &digest);
                if first_defect.is_none() {
                    first_defect = defect;
                }
                envelopes.push(value);
                eids.push(Value::Str(id));
                built = built.saturating_add(1);
            }
            let answer = ask(&envelopes_request(envelopes));
            assert!(answer.is_some(), "case {} answered outside OCJ", built);
            let Some(answer) = answer else { return };
            match first_defect {
                Some(detail) => {
                    assert_eq!(answer, refusal("bad_fact", detail), "case {}", built);
                    if !seen.contains(&detail) {
                        seen.push(detail);
                    }
                    refused_count = refused_count.saturating_add(1);
                }
                None => {
                    assert_eq!(answer, obj(vec![member("eids", Value::Arr(eids))]), "case {}", built);
                    accepted = accepted.saturating_add(1);
                }
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, DETAILS.to_vec(), "every refusal detail is reached");
        assert!(accepted > 0 && refused_count > 0, "{} accepted, {} refused", accepted, refused_count);
    }

    /// The body schema of every kind the embedded table defines, by kind.
    fn defined_schemas() -> Vec<(String, Value)> {
        let text_of = crate::laws::table("journal_v1").unwrap_or("");
        let table = crate::laws::parse_file(text_of).unwrap_or(Value::Null);
        match table.get("kinds") {
            Some(Value::Obj(kinds)) => kinds
                .iter()
                .filter_map(|(kind, entry)| match entry.get("body") {
                    Some(body @ Value::Obj(_)) => Some((kind.clone(), body.clone())),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    fn word(stream: &mut Stream, length: usize) -> String {
        kind_of(stream, length.max(1))
    }

    /// A value the spec admits, drawn from the stream.
    fn valid(stream: &mut Stream, spec: &Value) -> Value {
        match text(spec.get("type")) {
            Some("int") => {
                let lo = int(spec.get("lo")).unwrap_or(0);
                let hi = int(spec.get("hi")).unwrap_or(0);
                match stream.below(3) {
                    0 => Value::Int(lo),
                    1 => Value::Int(hi),
                    _ => Value::Int(lo.saturating_add(hi.saturating_sub(lo).checked_div(2).unwrap_or(0))),
                }
            }
            Some("bool") => Value::Bool(stream.below(2) == 0),
            Some("symbol") => match spec.get("of") {
                Some(Value::Arr(members)) => members.get(pick(stream, members.len())).cloned().unwrap_or(Value::Null),
                _ => Value::Null,
            },
            Some("hex") => {
                let length = usize::try_from(int(spec.get("len")).unwrap_or(0)).unwrap_or(0);
                Value::Str(hex_of(stream, length.div_ceil(2)).chars().take(length).collect())
            }
            Some("text") => {
                let most = usize::try_from(int(spec.get("max")).unwrap_or(1)).unwrap_or(1).min(32);
                let length = pick(stream, most).saturating_add(1);
                Value::Str(word(stream, length))
            }
            Some("object") => valid_body(stream, spec.get("fields").unwrap_or(&Value::Null)),
            _ => Value::Null,
        }
    }

    fn valid_body(stream: &mut Stream, schema: &Value) -> Value {
        match schema {
            Value::Obj(specs) => obj(specs.iter().map(|(name, spec)| (name.clone(), valid(stream, spec))).collect()),
            _ => Value::Null,
        }
    }

    fn expected(path: &str, problem: &str) -> String {
        if path.is_empty() {
            format!("body {}", problem)
        } else {
            format!("body {} {}", path, problem)
        }
    }

    /// A value the spec refuses, and the problem the refusal names.
    fn wrong(stream: &mut Stream, spec: &Value) -> Option<(Value, &'static str)> {
        let kind = text(spec.get("type"))?;
        Some(match kind {
            "int" => {
                let lo = int(spec.get("lo"))?;
                let hi = int(spec.get("hi"))?;
                match stream.below(4) {
                    0 if hi < ocj::MAX_INT => (Value::Int(hi.saturating_add(1)), "range"),
                    1 if lo > ocj::MAX_INT.saturating_neg() => (Value::Int(lo.saturating_sub(1)), "range"),
                    2 => (Value::Bool(true), "int"),
                    _ => (Value::Str(String::from("0")), "int"),
                }
            }
            "bool" => (if stream.below(2) == 0 { Value::Int(1) } else { Value::Null }, "bool"),
            "symbol" => (if stream.below(2) == 0 { Value::Str(String::from("zz_none")) } else { Value::Int(0) }, "symbol"),
            "hex" => {
                let length = usize::try_from(int(spec.get("len"))?).ok()?;
                let good: String = hex_of(stream, length.div_ceil(2)).chars().take(length).collect();
                match stream.below(3) {
                    0 => (Value::Str(upper_once(stream, &good)), "hex"),
                    1 => (Value::Str(resized(stream, &good)), "hex"),
                    _ => (Value::Int(7), "hex"),
                }
            }
            "text" => {
                let most = usize::try_from(int(spec.get("max"))?).ok()?;
                match stream.below(4) {
                    0 => (Value::Str(String::new()), "text"),
                    1 => (Value::Str(format!(" {}", word(stream, 2))), "text"),
                    2 => (Value::Str(word(stream, most.saturating_add(1))), "text"),
                    _ => (Value::Arr(Vec::new()), "text"),
                }
            }
            "object" => (Value::Int(3), "object"),
            _ => return None,
        })
    }

    /// A body broken at exactly one place under `path`, and the detail it must draw.
    fn broken(stream: &mut Stream, schema: &Value, path: &str) -> Option<(Value, String)> {
        let body = valid_body(stream, schema);
        let members = match &body {
            Value::Obj(members) => members.clone(),
            _ => return None,
        };
        if members.is_empty() || stream.below(4) == 0 {
            return Some(match stream.below(3) {
                0 => (Value::Arr(Vec::new()), expected(path, "object")),
                1 => {
                    let mut more = members.clone();
                    more.push((String::from("zz_extra"), Value::Int(1)));
                    (obj(more), expected(path, "fields"))
                }
                _ if !members.is_empty() => {
                    let dropped = pick(stream, members.len());
                    let kept = members.iter().enumerate().filter(|(at, _)| *at != dropped).map(|(_, m)| m.clone());
                    (obj(kept.collect()), expected(path, "fields"))
                }
                _ => (obj(vec![member("a", Value::Int(0))]), expected(path, "fields")),
            });
        }
        let at = pick(stream, members.len());
        let (name, _) = members.get(at)?;
        let spec = schema.get(name)?;
        let inner = if path.is_empty() { name.clone() } else { format!("{}.{}", path, name) };
        let (value, detail) = match (text(spec.get("type")), spec.get("fields")) {
            (Some("object"), Some(fields)) if stream.below(2) == 0 => broken(stream, fields, &inner)?,
            _ => {
                let (value, problem) = wrong(stream, spec)?;
                (value, expected(&inner, problem))
            }
        };
        let replaced = members.iter().map(|(key, old)| (key.clone(), if key == name { value.clone() } else { old.clone() }));
        Some((obj(replaced.collect()), detail))
    }

    #[test]
    fn three_thousand_bodies_broken_at_one_place_are_refused_where_the_reference_refuses_them() {
        let schemas = defined_schemas();
        assert!(schemas.len() >= 19, "the table defines {} bodies", schemas.len());
        let mut stream = Stream::new(&[0_u8; 32], "test.journal.body", 0);
        for (kind, schema) in &schemas {
            for _ in 0..8 {
                let body = valid_body(&mut stream, schema);
                assert_eq!(check_body(schema, &body, ""), Ok(()), "a valid {} body", kind);
            }
        }
        let mut seen: Vec<String> = Vec::new();
        let mut nested = 0_usize;
        for case in 0..3000_usize {
            let (kind, schema) = schemas.get(pick(&mut stream, schemas.len())).cloned().unwrap_or((String::new(), Value::Null));
            let made = broken(&mut stream, &schema, "");
            assert!(made.is_some(), "case {} of {} built", case, kind);
            let Some((body, detail)) = made else { return };
            assert_eq!(check_body(&schema, &body, ""), Err(refused("bad_fact", detail.clone())), "case {} of {}", case, kind);
            if detail.matches('.').count() > 0 {
                nested = nested.saturating_add(1);
            }
            let problem = String::from(detail.rsplit(' ').next().unwrap_or(""));
            if !seen.contains(&problem) {
                seen.push(problem);
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, ["bool", "fields", "hex", "int", "object", "range", "symbol", "text"].map(String::from).to_vec());
        assert!(nested > 0, "a nested field was broken and named by its path");
    }

    #[test]
    fn a_spec_the_twin_cannot_read_is_refused_as_an_unsound_table() {
        let odd = obj(vec![member("x", obj(vec![member("type", Value::Str(String::from("float")))]))]);
        let body = obj(vec![member("x", Value::Int(1))]);
        assert_eq!(check_body(&odd, &body, ""), Err(bad("unknown_law", "journal")));
        assert_eq!(check_body(&Value::Null, &body, ""), Err(bad("unknown_law", "journal")));
        let genesis = defined_schemas().into_iter().find(|(kind, _)| kind == "genesis").map(|(_, s)| s).unwrap_or(Value::Null);
        let mut stream = Stream::new(&[0_u8; 32], "test.journal.genesis", 0);
        let mut body = valid_body(&mut stream, &genesis);
        assert_eq!(check_body(&genesis, &body, ""), Ok(()));
        if let Value::Obj(members) = &mut body {
            for (name, value) in members.iter_mut() {
                if name == "laws" {
                    if let Value::Obj(fields) = value {
                        for (field, inner) in fields.iter_mut() {
                            if field == "params" {
                                *inner = obj(vec![member("sun_max", Value::Int(1))]);
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(check_body(&genesis, &body, ""), Err(refused("bad_fact", String::from("body laws.params fields"))));
    }
}
