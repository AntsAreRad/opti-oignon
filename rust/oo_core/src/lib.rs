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
//! The window is assembled from parts the caller counted: nothing here
//! counts a token, so the window is counted once, with the counter of the
//! turn it is sent in.
//!
//! The recall probes of the eviction gate live in `probes`: they answer
//! only for text they class exactly as Python does, and refuse the rest.
//! The trust kernel's twin lives in `trust`: the labels of a tool call's
//! arguments and the gate's decision, asked in the gate's shadow.

mod allium;
mod linear;
mod probes;
mod trust;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use sha2::{Digest, Sha256};

const VERSION: &str = "0.10.0";

/// The shape of `compose_counted`'s arguments; the composer asks a core whose format is its own, and no other.
const COMPOSE_FORMAT: i64 = 2;

/// The ABI of the interpreter this artefact was built for (build.rs), stamped in the file where the loader reads it
/// before it loads the artefact: a line `oo_core built for python: <SOABI>`.
#[used]
static PYTHON_STAMP: &[u8] = concat!("\noo_core built for python: ", env!("OO_CORE_PYTHON_TAG"), "\n").as_bytes();

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

/// One counted part of the window: `(text, provenance, tokens)`.
type Part = (String, String, u64);

/// Assemble the window from counted parts: system, Core, receipts digest, Peels, Flesh, tail, turn.
///
/// Returns `(segments, total_tokens, dropped_peels)`; a segment is
/// `(layer, text, provenance, tokens, instruction_bearing)`. A part the
/// caller does not lay is `None`. Refuses with the reference's own sentence
/// when a part that may not be cut is over its cap; the Peels are cut to
/// whole items and the cut is counted. `caps` is
/// `(window, reserve, system, core, receipts, peels, flesh, tail, turn)`.
/// Every argument is required and positional, the parts not laid included.
#[pyfunction]
#[pyo3(signature = (system, core, receipts, peels, flesh, tail, turn, caps))]
#[allow(clippy::too_many_arguments)]
fn compose_counted(
    system: Option<Part>,
    core: Part,
    receipts: Option<Part>,
    peels: Vec<Part>,
    flesh: Vec<Part>,
    tail: Option<Part>,
    turn: Part,
    caps: (u64, u64, u64, u64, u64, u64, u64, u64, u64),
) -> PyResult<(Vec<Segment>, u64, u64)> {
    let (window, reserve, cap_system, cap_core, cap_receipts, cap_peels, cap_flesh, cap_tail, cap_turn) = caps;
    let mut segments: Vec<Segment> = Vec::new();

    if let Some((text, provenance, tokens)) = system {
        refuse_over("system", tokens, cap_system, "the system prompt is never cut")?;
        segments.push(("system".into(), text, provenance, tokens, true));
    }

    let (text, provenance, tokens) = core;
    refuse_over("core", tokens, cap_core, "the Core is never cut, it is the registry's bytes")?;
    segments.push(("core".into(), text, provenance, tokens, true));

    if let Some((text, provenance, tokens)) = receipts {
        refuse_over("receipts", tokens, cap_receipts, "resolve or archive receipts first")?;
        segments.push(("receipts".into(), text, provenance, tokens, false));
    }

    let mut used: u64 = 0;
    let mut dropped: u64 = 0;
    for (text, provenance, tokens) in peels {
        if used.saturating_add(tokens) > cap_peels {
            dropped = dropped.saturating_add(1);
            continue;
        }
        used = used.saturating_add(tokens);
        segments.push(("peels".into(), text, provenance, tokens, false));
    }

    let flesh_total = flesh.iter().fold(0u64, |sum, part| sum.saturating_add(part.2));
    refuse_over(
        "flesh",
        flesh_total,
        cap_flesh,
        "a turn leaves the window only with a receipt; evict first, the composer drops nothing",
    )?;
    for (text, provenance, tokens) in flesh {
        segments.push(("flesh".into(), text, provenance, tokens, false));
    }

    if let Some((text, provenance, tokens)) = tail {
        refuse_over("tail", tokens, cap_tail, "the tail is never cut here; its caller fits its sources to the cap")?;
        segments.push(("tail".into(), text, provenance, tokens, false));
    }

    let (text, provenance, tokens) = turn;
    refuse_over("turn", tokens, cap_turn, "the current turn is never truncated")?;
    segments.push(("turn".into(), text, provenance, tokens, true));

    let total = segments.iter().fold(0u64, |sum, segment| sum.saturating_add(segment.3));
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
    m.add("PYTHON_TAG", env!("OO_CORE_PYTHON_TAG"))?;
    m.add_function(wrap_pyfunction!(entry_hash, m)?)?;
    m.add_function(wrap_pyfunction!(span_key, m)?)?;
    m.add_function(wrap_pyfunction!(digest_spans, m)?)?;
    m.add_function(wrap_pyfunction!(peel_id, m)?)?;
    m.add("COMPOSE_FORMAT", COMPOSE_FORMAT)?;
    m.add_function(wrap_pyfunction!(compose_counted, m)?)?;
    m.add_function(wrap_pyfunction!(probes::probe_generate, m)?)?;
    m.add_function(wrap_pyfunction!(probes::probe_score, m)?)?;
    m.add_function(wrap_pyfunction!(probes::probe_text_classes, m)?)?;
    m.add_function(wrap_pyfunction!(probes::probe_letter_folds, m)?)?;
    m.add_function(wrap_pyfunction!(allium::allium_call, m)?)?;
    m.add_function(wrap_pyfunction!(allium::allium_engine, m)?)?;
    m.add("LINEAR_FORMAT", linear::FORMAT)?;
    m.add("TRUST_FORMAT", trust::FORMAT)?;
    m.add_function(wrap_pyfunction!(trust::trust_labels, m)?)?;
    m.add_function(wrap_pyfunction!(trust::trust_decide, m)?)?;
    m.add_class::<linear::LinearProgram>()?;
    m.add_class::<linear::LinearRead>()?;
    Ok(())
}
