//! The law and table files, embedded at build time from the very files the
//! Python reference reads (`opti_oignon/allium/laws`, `.../tables`): one
//! source each. A file's digest is the SHA-256 of its canonical re-emission.

use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use crate::ocj::{self, Refused, Value};

pub const LAWS: [(&str, &str); 1] = [(
    "fixture",
    include_str!("../../../opti_oignon/allium/laws/fixture.json"),
)];

pub const TABLES: [(&str, &str); 1] = [(
    "sine_q15_v1",
    include_str!("../../../opti_oignon/allium/tables/sine_q15_v1.json"),
)];

pub fn parse_file(text: &str) -> Result<Value, Refused> {
    ocj::parse(text.as_bytes(), true)
}

pub fn digest(value: &Value) -> Result<[u8; 32], Refused> {
    let bytes = ocj::emit(value)?;
    Ok(Sha256::digest(&bytes).into())
}

pub fn law(name: &str) -> Option<&'static str> {
    LAWS.iter().find(|(n, _)| *n == name).map(|(_, text)| *text)
}

/// The frozen Q15 sine table: 1024 entries, one binary-angle turn.
pub fn sine() -> Result<Vec<i128>, Refused> {
    let text = TABLES.first().map(|(_, text)| *text).unwrap_or("");
    let table = parse_file(text)?;
    let mut out = Vec::with_capacity(1024);
    if let Some(Value::Arr(entries)) = table.get("entries") {
        for entry in entries {
            if let Value::Int(number) = entry {
                out.push(i128::from(*number));
            }
        }
    }
    Ok(out)
}
