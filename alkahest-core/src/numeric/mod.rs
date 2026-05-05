//! V2-6 numeric helpers — currently limited to linear integer relation guesses.

pub mod pslq;

pub use pslq::{guess_integer_relation, PslqError};
