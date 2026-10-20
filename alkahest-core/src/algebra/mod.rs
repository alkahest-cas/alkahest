//! Domain-specific algebras built on the expression kernel (V3-2+).

pub mod noncommutative;
pub mod quaternion;
#[cfg(test)]
mod quaternion_tests;

pub use quaternion::{Quaternion, QuaternionError};

pub use noncommutative::{
    clifford_orthogonal_rules, imag_unit_atom, pauli_product_rules, PauliSpinAlgebraRule,
};
pub mod gamma_dim;
pub mod lie;
pub mod nc_poly;

pub use nc_poly::{
    anticommutator, commutator, gamma_trace, nc_expand, nc_rewrite, nc_terms, normal_order,
    pauli_reduce, simplify_with_noncommutative, vacuum_expectation, GammaSlot, LadderMode, NcError,
};

pub use gamma_dim::{gamma_contract, gamma_trace_dim, lorentz_contract};
pub use lie::{
    su2_structure_constants, su3_d_symbols, su3_structure_constants, LieAlgebra, StructureEntry,
};

#[cfg(test)]
mod f20_tests;
