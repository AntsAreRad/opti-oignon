//! The componion's engine, reached from Python: two thin wrappers over the
//! `allium` crate's byte protocol.
//!
//! `allium_call` copies the request, then releases the GIL for the whole of
//! the engine's work, so a long catch-up or a night of dreaming never stalls
//! a chat stream running in the same process. The engine itself is pure:
//! bytes in, bytes out, no clock, no file, no network.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

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
