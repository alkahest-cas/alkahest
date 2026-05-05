//! Domain-specific algebras built on the expression kernel (V3-2+).

pub mod noncommutative;

pub use noncommutative::{
    clifford_orthogonal_rules, imag_unit_atom, pauli_product_rules, PauliSpinAlgebraRule,
};
