//! Domain-specific algebras built on the expression kernel (V3-2+).

pub mod noncommutative;
pub mod quaternion;
#[cfg(test)]
mod quaternion_tests;

pub use quaternion::{Quaternion, QuaternionError};

pub use noncommutative::{
    clifford_orthogonal_rules, imag_unit_atom, pauli_product_rules, PauliSpinAlgebraRule,
};
