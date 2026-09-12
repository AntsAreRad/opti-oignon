//! The onion memory's native core: integrity primitives and window assembly.
//!
//! Python stays the reference. Every function here must answer exactly as
//! the reference does -- byte for byte on the hashes, field for field on
//! the assembled window, word for word on the refusals -- and the contracts
//! on the Python side hold it to that. The canonical JSON below reproduces
//! `json.dumps(value, sort_keys=True, separators=(",", ":"),
//! ensure_ascii=False)` for the value shapes the memory stores: strings,
//! integers, booleans, null, lists and string-keyed objects. A float is
//! refused rather than formatted: Python's shortest-repr float printing is
//! not reproduced here, and a refusal sends the caller to the reference.
//!
//! Known divergence, recorded: the token estimate splits on Unicode
//! `White_Space`; Python's `str.split()` also treats the four information
//! separators U+001C..U+001F as whitespace. No memory text carries them.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use sha2::{Digest, Sha256};

const VERSION: &str = "0.1.0";

fn escape_into(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_value(out: &mut String, value: &Bound<'_, PyAny>, sort_keys: bool, compact: bool) -> PyResult<()> {
    let (item_sep, key_sep) = if compact { (",", ":") } else { (", ", ": ") };
    if value.is_none() {
        out.push_str("null");
    } else if let Ok(b) = value.downcast::<PyBool>() {
        out.push_str(if b.is_true() { "true" } else { "false" });
    } else if value.downcast::<PyFloat>().is_ok() {
        return Err(PyTypeError::new_err(
            "oo_core canonical JSON refuses floats: the reference formats them, this core does not",
        ));
    } else if value.downcast::<PyInt>().is_ok() {
        out.push_str(&value.str()?.to_string_lossy());
    } else if let Ok(s) = value.downcast::<PyString>() {
        escape_into(out, &s.to_cow()?);
    } else if let Ok(list) = value.downcast::<PyList>() {
        out.push('[');
        for (i, item) in list.iter().enumerate() {
            if i > 0 {
                out.push_str(item_sep);
            }
            write_value(out, &item, sort_keys, compact)?;
        }
        out.push(']');
    } else if let Ok(tuple) = value.downcast::<PyTuple>() {
        out.push('[');
        for (i, item) in tuple.iter().enumerate() {
            if i > 0 {
                out.push_str(item_sep);
            }
            write_value(out, &item, sort_keys, compact)?;
        }
        out.push(']');
    } else if let Ok(dict) = value.downcast::<PyDict>() {
        let mut entries: Vec<(String, Bound<'_, PyAny>)> = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key = k.downcast::<PyString>().map_err(|_| {
                PyTypeError::new_err("oo_core canonical JSON takes string keys only")
            })?;
            entries.push((key.to_cow()?.into_owned(), v));
        }
        if sort_keys {
            entries.sort_by(|a, b| a.0.cmp(&b.0));
        }
        out.push('{');
        for (i, (k, v)) in entries.iter().enumerate() {
            if i > 0 {
                out.push_str(item_sep);
            }
            escape_into(out, k);
            out.push_str(key_sep);
            write_value(out, v, sort_keys, compact)?;
        }
        out.push('}');
    } else {
        return Err(PyTypeError::new_err(format!(
            "oo_core canonical JSON cannot serialise {}",
            value.get_type().name()?
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

fn as_list<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    if let Ok(list) = value.downcast::<PyList>() {
        return Ok(list.clone());
    }
    let items: Vec<Bound<'py, PyAny>> = value.iter()?.collect::<PyResult<_>>()?;
    PyList::new_bound(value.py(), items).extract()
}

/// The id of a Core entry: SHA-256 of the UTF-8 bytes of its text.
#[pyfunction]
fn entry_hash(text: &str) -> String {
    sha256_hex(text.as_bytes())
}

/// The recall key of a span: SHA-256 of its canonical JSON (sorted, compact).
#[pyfunction]
fn span_key(span: &Bound<'_, PyAny>) -> PyResult<String> {
    let list = as_list(span)?;
    let mut out = String::new();
    write_value(&mut out, list.as_any(), true, true)?;
    Ok(sha256_hex(out.as_bytes()))
}

/// The digest of several spans: SHA-256 of their canonical JSON as a list of lists.
#[pyfunction]
fn digest_spans(spans: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = spans.py();
    let mut lists: Vec<Bound<'_, PyAny>> = Vec::new();
    for span in spans.iter()? {
        lists.push(as_list(&span?)?.into_any());
    }
    let outer = PyList::new_bound(py, lists);
    let mut out = String::new();
    write_value(&mut out, outer.as_any(), true, true)?;
    Ok(sha256_hex(out.as_bytes()))
}

/// The id of a peel: SHA-256 of `json.dumps([text, sources], ensure_ascii=False)`.
#[pyfunction]
fn peel_id(text: &str, sources: Vec<String>) -> String {
    let mut out = String::from("[");
    escape_into(&mut out, text);
    out.push_str(", [");
    for (i, s) in sources.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        escape_into(&mut out, s);
    }
    out.push_str("]]");
    sha256_hex(out.as_bytes())
}

fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let words = text.split_whitespace().count() as f64;
    let estimate = (words * 1.3) as i64;
    if estimate < 1 { 1 } else { estimate as u64 }
}

fn refuse_over(layer: &str, tokens: u64, cap: u64, why: &str) -> PyResult<()> {
    if tokens > cap {
        return Err(PyValueError::new_err(format!(
            "{} is {} tokens against a cap of {}: {}",
            layer, tokens, cap, why
        )));
    }
    Ok(())
}

type Segment = (String, String, String, u64, bool);

/// Assemble the window: Core, receipts digest, selected Peels, Flesh, turn.
///
/// Returns `(segments, total_tokens, dropped_peels)`; a segment is
/// `(layer, text, provenance, tokens, instruction_bearing)`. Refuses with
/// the reference's own sentence when a layer that may not be cut is over
/// its cap. `caps` is `(window, reserve, core, receipts, peels, flesh, turn)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn compose_segments(
    core_text: &str,
    core_root: &str,
    digest: &str,
    peels: Vec<(String, String)>,
    flesh: Vec<(String, String, String)>,
    turn: &str,
    caps: (u64, u64, u64, u64, u64, u64, u64),
) -> PyResult<(Vec<Segment>, u64, u64)> {
    let (window, reserve, cap_core, cap_receipts, cap_peels, cap_flesh, cap_turn) = caps;
    let mut segments: Vec<Segment> = Vec::new();

    let core_tokens = estimate_tokens(core_text);
    refuse_over("core", core_tokens, cap_core, "the Core is never cut, it is the registry's bytes")?;
    segments.push(("core".into(), core_text.into(), format!("core:{}", core_root), core_tokens, true));

    if !digest.is_empty() {
        let tokens = estimate_tokens(digest);
        refuse_over("receipts", tokens, cap_receipts, "resolve or archive receipts first")?;
        segments.push(("receipts".into(), digest.into(), "receipts".into(), tokens, false));
    }

    let mut used: u64 = 0;
    let mut dropped: u64 = 0;
    for (text, provenance) in peels {
        let tokens = estimate_tokens(&text);
        if used + tokens > cap_peels {
            dropped += 1;
            continue;
        }
        used += tokens;
        segments.push(("peels".into(), text, provenance, tokens, false));
    }

    let mut flesh_segments: Vec<Segment> = Vec::new();
    let mut flesh_total: u64 = 0;
    for (text, turn_id, role) in flesh {
        let tokens = estimate_tokens(&text);
        flesh_total += tokens;
        flesh_segments.push(("flesh".into(), text, format!("flesh:{}:{}", turn_id, role), tokens, false));
    }
    refuse_over(
        "flesh",
        flesh_total,
        cap_flesh,
        "a turn leaves the window only with a receipt; evict first, the composer drops nothing",
    )?;
    segments.extend(flesh_segments);

    let turn_tokens = estimate_tokens(turn);
    refuse_over("turn", turn_tokens, cap_turn, "the current turn is never truncated")?;
    segments.push(("turn".into(), turn.into(), "turn".into(), turn_tokens, true));

    let total: u64 = segments.iter().map(|s| s.3).sum();
    refuse_over(
        "window",
        total,
        window.saturating_sub(reserve),
        "the generation reserve is not assembled into",
    )?;
    Ok((segments, total, dropped))
}

#[pymodule]
fn oo_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("VERSION", VERSION)?;
    m.add_function(wrap_pyfunction!(entry_hash, m)?)?;
    m.add_function(wrap_pyfunction!(span_key, m)?)?;
    m.add_function(wrap_pyfunction!(digest_spans, m)?)?;
    m.add_function(wrap_pyfunction!(peel_id, m)?)?;
    m.add_function(wrap_pyfunction!(compose_segments, m)?)?;
    Ok(())
}
