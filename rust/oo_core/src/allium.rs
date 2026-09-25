//! The componion's engine, reached from Python: two thin wrappers over the
//! `allium` crate's byte protocol.
//!
//! `allium_call` copies the request, then releases the GIL for the whole of
//! the engine's work, so a long catch-up or a night of dreaming never stalls
//! a chat stream running in the same process. The engine itself is pure:
//! bytes in, bytes out, no clock, no file, no network.
//!
//! The engine never panics by design; its arithmetic is checked, and this
//! artefact is built with overflow checks as a second net. Should a panic
//! happen all the same, it must not unwind into Python: both entry points
//! reach the engine only through `guarded`, which catches it and answers
//! `engine_panic`, the refusal the reference would leave the view frozen on.

use std::panic::AssertUnwindSafe;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// What a caught panic answers: a refusal like any other, never an exception.
const PANIC_ANSWER: &[u8] = br#"{"detail":"panic","refused":"engine_panic"}"#;

/// Runs the engine and turns a panic inside it into the `engine_panic` refusal.
fn guarded(run: impl FnOnce() -> Vec<u8>) -> Vec<u8> {
    std::panic::catch_unwind(AssertUnwindSafe(run)).unwrap_or_else(|_| PANIC_ANSWER.to_vec())
}

/// The engine's two entry points, each behind the guard. They carry the
/// engine crate's names, so the functions below read as the engine's own
/// calls; the crate itself is reached only from here, by its absolute path.
mod allium {
    pub fn call(request: &[u8]) -> Vec<u8> {
        super::guarded(|| ::allium::call(request))
    }

    pub fn engine_info() -> Vec<u8> {
        super::guarded(::allium::engine_info)
    }
}

#[pyfunction]
pub fn allium_call(py: Python<'_>, request: &[u8]) -> PyObject {
    let owned: Vec<u8> = request.to_vec();
    let answer = py.allow_threads(move || allium::call(&owned));
    PyBytes::new_bound(py, &answer).into()
}

#[pyfunction]
pub fn allium_engine(py: Python<'_>) -> PyObject {
    PyBytes::new_bound(py, &allium::engine_info()).into()
}
