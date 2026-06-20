//! Real algebraic computations (cad, etc.).
//!
//! V2-9 — Cylindrical Algebraic Deconstruction / quantifier elimination (single-block,
//! purely polynomial formulas in one quantified real variable).

pub mod cad;
pub mod routh;

pub use cad::{cad_lift, cad_project, decide, decide_expr, CadError, QeResult};
pub use routh::{routh_hurwitz, RouthHurwitz, ROUTH_MAX_DEGREE};
