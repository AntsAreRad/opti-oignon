//! The law, founder pool and table files, embedded at build time from the
//! very files the Python reference reads (`opti_oignon/allium/laws`,
//! `.../tables`): one source each. A file's digest is the SHA-256 of its
//! canonical re-emission. The files are parsed at every call that needs
//! them; nothing here is cached.

use alloc::vec::Vec;
use sha2::{Digest, Sha256};

use crate::ocj::{self, Refused, Value};

pub const LAWS: [(&str, &str); 2] = [
    ("fixture", include_str!("../../../opti_oignon/allium/laws/fixture.json")),
    ("v0_1", include_str!("../../../opti_oignon/allium/laws/v0_1.json")),
];

pub const FOUNDERS: [(&str, &str); 2] = [
    ("fixture", include_str!("../../../opti_oignon/allium/laws/founders_fixture.json")),
    ("v1", include_str!("../../../opti_oignon/allium/laws/founders_v1.json")),
];

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

fn named(files: &[(&'static str, &'static str)], name: &str) -> Option<&'static str> {
    files.iter().find(|(n, _)| *n == name).map(|(_, text)| *text)
}

pub fn law(name: &str) -> Option<&'static str> {
    named(&LAWS, name)
}

/// The founder pool file `name` (`laws/founders_<name>.json`).
pub fn founders(name: &str) -> Option<&'static str> {
    named(&FOUNDERS, name)
}

pub fn table(name: &str) -> Option<&'static str> {
    named(&TABLES, name)
}

/// The frozen Q15 sine table: 1024 entries, one binary-angle turn.
pub fn sine() -> Result<Vec<i128>, Refused> {
    let text = table("sine_q15_v1").unwrap_or("");
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
