#![deny(clippy::indexing_slicing, clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! The organs of the being, the twin of `opti_oignon/allium/ref/organs`.
//!
//! Every function here answers as its Python reference answers, refusal
//! codes and details included. Nothing here indexes, unwraps or panics: a
//! slice is read with `get` or a slice pattern, every arithmetic operation is
//! checked, and a check that fails where the reference has no refusal maps
//! to one named `engine_panic`.

pub mod compile;
pub mod genome;
