#![deny(clippy::indexing_slicing, clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::todo, clippy::unimplemented)]
//! The organs of the being, the twin of `opti_oignon/allium/ref/organs`.
//!
//! The genome's codec and compiler and the phonology read what a genome
//! holds; the clock, the reserve, the soil, the stage and the weather are
//! the life that runs on it, stepped by `world`.
//!
//! Every function here answers as its Python reference answers, refusal
//! codes and details included. Nothing here indexes, unwraps or panics: a
//! slice is read with `get` or a slice pattern, every arithmetic operation is
//! checked, and a check that fails where the reference has no refusal maps
//! to one named `engine_panic`.

pub mod chem;
pub mod clock;
pub mod compile;
pub mod genome;
pub mod phon;
pub mod soil;
pub mod stage;
pub mod weather;
