//! Non-commutative generator rules — Pauli spin algebra and a minimal Clifford snippet.
//!
//! Symbols should be created with [`ExprPool::symbol_commutative(..., false)`].
//! Use [`imag_unit_atom`] for a commuting Complex marker treated as \\(i\\).

use crate::kernel::domain::Domain;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::rules::{one_step, RewriteRule};

/// Re-exported for integrators that only need the predicate without Pauli rules.
pub use crate::kernel::expr_props::{
    expr_contains_noncommutative_symbol, mult_tree_is_commutative,
};

/// Commuting formal imaginary unit (\\(i\\)) for Pauli product tables.
pub fn imag_unit_atom(pool: &ExprPool) -> ExprId {
    pool.symbol_commutative("ImagUnit", Domain::Complex, true)
}

/// Name of a **non-commutative** symbol, or `None`.
///
/// The commutativity requirement is load-bearing, not decoration. These product
/// tables are order-sensitive — `σxσy = iσz` but `σyσx = −iσz` — while a
/// commutative `Mul` sorts its operands, so `sx*sy` and `sy*sx` are literally
/// the same expression. Firing the table on that shared form hands back one
/// answer for two different products, so one of the two readings is always
/// wrong, with nothing to distinguish it.
///
/// Matching by name alone did exactly that for a commutative symbol that
/// happened to be called `sx` — a plausible name for something unrelated.
fn symbol_name(expr: ExprId, pool: &ExprPool) -> Option<String> {
    pool.with(expr, |d| match d {
        ExprData::Symbol {
            name,
            commutative: false,
            ..
        } => Some(name.clone()),
        _ => None,
    })
}

/// Cyclic Pauli products \\(\\sigma_x\\sigma_y = i\\sigma_z\\), squares \\(\\sigma_k^2 = 1\\).
///
/// Recognises symbols by name: `sx`, `sy`, `sz` (ASCII); order matters.
pub struct PauliSpinAlgebraRule;

impl RewriteRule for PauliSpinAlgebraRule {
    fn name(&self) -> &'static str {
        "pauli_algebra"
    }

    fn apply(
        &self,
        expr: ExprId,
        pool: &ExprPool,
    ) -> Option<(ExprId, crate::deriv::log::DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Mul(v) if v.len() == 2 => v,
            _ => return None,
        };
        let (a, b) = (args[0], args[1]);
        let na = symbol_name(a, pool)?;
        let nb = symbol_name(b, pool)?;
        let ii = imag_unit_atom(pool);
        let neg_one = pool.integer(-1_i32);
        let sx_id = pool.symbol_commutative("sx", Domain::Complex, false);
        let sy_id = pool.symbol_commutative("sy", Domain::Complex, false);
        let sz_id = pool.symbol_commutative("sz", Domain::Complex, false);

        let after = match (na.as_str(), nb.as_str()) {
            ("sx", "sx") | ("sy", "sy") | ("sz", "sz") => pool.integer(1_i32),
            ("sx", "sy") => pool.mul(vec![ii, sz_id]),
            ("sy", "sx") => pool.mul(vec![neg_one, ii, sz_id]),
            ("sy", "sz") => pool.mul(vec![ii, sx_id]),
            ("sz", "sy") => pool.mul(vec![neg_one, ii, sx_id]),
            ("sz", "sx") => pool.mul(vec![ii, sy_id]),
            ("sx", "sz") => pool.mul(vec![neg_one, ii, sy_id]),
            _ => return None,
        };

        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// `sx * sy` → `ImagUnit * sz` once flattening has produced a binary `Mul`.
pub fn pauli_product_rules() -> Vec<Box<dyn RewriteRule>> {
    vec![Box::new(PauliSpinAlgebraRule)]
}

/// Clifford / Grassmann-style orthogonal basis: `cliff_e1 * cliff_e2 -> -(cliff_e2 * cliff_e1)`,
/// `cliff_e1^2 -> 1`, `cliff_e2^2 -> 1`.
pub struct CliffordOrthogonalRule;

impl RewriteRule for CliffordOrthogonalRule {
    fn name(&self) -> &'static str {
        "clifford_orthogonal"
    }

    fn apply(
        &self,
        expr: ExprId,
        pool: &ExprPool,
    ) -> Option<(ExprId, crate::deriv::log::DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Mul(v) if v.len() == 2 => v,
            _ => return None,
        };
        let (a, b) = (args[0], args[1]);
        let na = symbol_name(a, pool)?;
        let nb = symbol_name(b, pool)?;

        let after = match (na.as_str(), nb.as_str()) {
            ("cliff_e1", "cliff_e1") | ("cliff_e2", "cliff_e2") => pool.integer(1_i32),
            ("cliff_e1", "cliff_e2") => pool.mul(vec![pool.integer(-1_i32), b, a]),
            _ => return None,
        };
        Some((after, one_step(self.name(), expr, after)))
    }
}

pub fn clifford_orthogonal_rules() -> Vec<Box<dyn RewriteRule>> {
    vec![Box::new(CliffordOrthogonalRule)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ExprPool;
    use crate::simplify::engine::{rules_for_config, simplify_with, SimplifyConfig};

    #[test]
    fn pauli_sx_sy_is_i_sz() {
        let p = ExprPool::new();
        let sx = p.symbol_commutative("sx", Domain::Complex, false);
        let sy = p.symbol_commutative("sy", Domain::Complex, false);
        let sz = p.symbol_commutative("sz", Domain::Complex, false);
        let expr = p.mul(vec![sx, sy]);
        let mut rules = rules_for_config(&SimplifyConfig::default());
        rules.extend(pauli_product_rules());
        let r = simplify_with(expr, &p, &rules, SimplifyConfig::default());
        let i_sz = p.mul(vec![imag_unit_atom(&p), sz]);
        assert_eq!(r.value, i_sz);
    }

    #[test]
    fn pauli_sy_sx_is_neg_i_sz() {
        let p = ExprPool::new();
        let sx = p.symbol_commutative("sx", Domain::Complex, false);
        let sy = p.symbol_commutative("sy", Domain::Complex, false);
        let sz = p.symbol_commutative("sz", Domain::Complex, false);
        let expr = p.mul(vec![sy, sx]);
        let mut rules = rules_for_config(&SimplifyConfig::default());
        rules.extend(pauli_product_rules());
        let r = simplify_with(expr, &p, &rules, SimplifyConfig::default());
        let neg_i_sz = p.mul(vec![p.integer(-1_i32), imag_unit_atom(&p), sz]);
        assert_eq!(r.value, neg_i_sz);
    }

    #[test]
    fn clifford_antijoin() {
        let p = ExprPool::new();
        let e1 = p.symbol_commutative("cliff_e1", Domain::Real, false);
        let e2 = p.symbol_commutative("cliff_e2", Domain::Real, false);
        let expr = p.mul(vec![e1, e2]);
        let mut rules = rules_for_config(&SimplifyConfig::default());
        rules.extend(clifford_orthogonal_rules());
        let r = simplify_with(expr, &p, &rules, SimplifyConfig::default());
        let expect = p.mul(vec![p.integer(-1_i32), e2, e1]);
        assert_eq!(r.value, expect);
    }

    #[test]
    fn nc_mul_distinct_order() {
        let p = ExprPool::new();
        let a = p.symbol_commutative("A", Domain::Real, false);
        let b = p.symbol_commutative("B", Domain::Real, false);
        assert_ne!(p.mul(vec![a, b]), p.mul(vec![b, a]));
    }
}

#[cfg(test)]
mod commutativity_guard_tests {
    use super::*;
    use crate::simplify::engine::{rules_for_config, simplify_with, SimplifyConfig};

    fn run(expr: ExprId, pool: &ExprPool, extra: Vec<Box<dyn RewriteRule>>) -> ExprId {
        let mut rules = rules_for_config(&SimplifyConfig::default());
        rules.extend(extra);
        simplify_with(expr, pool, &rules, SimplifyConfig::default()).value
    }

    /// The product tables must not fire on *commutative* symbols.
    ///
    /// A commutative `Mul` sorts its operands, so `sx*sy` and `sy*sx` are the
    /// same expression. `σxσy = iσz` and `σyσx = −iσz` differ in sign, so
    /// applying the table to that shared form returns one answer for two
    /// different products — one of the readings is necessarily wrong, and
    /// nothing in the result says which.
    ///
    /// Matching by name alone did exactly that for any commutative symbol that
    /// happened to be named `sx`.
    #[test]
    fn pauli_table_ignores_commutative_symbols() {
        let pool = ExprPool::new();
        let sx = pool.symbol_commutative("sx", Domain::Complex, true);
        let sy = pool.symbol_commutative("sy", Domain::Complex, true);
        let product = pool.mul(vec![sx, sy]);

        let out = run(product, &pool, pauli_product_rules());
        assert_eq!(
            out, product,
            "Pauli table fired on commutative symbols, where operand order is not recoverable"
        );
    }

    #[test]
    fn clifford_rules_ignore_commutative_symbols() {
        let pool = ExprPool::new();
        let e1 = pool.symbol_commutative("cliff_e1", Domain::Real, true);
        let e2 = pool.symbol_commutative("cliff_e2", Domain::Real, true);
        let product = pool.mul(vec![e1, e2]);

        let out = run(product, &pool, clifford_orthogonal_rules());
        assert_eq!(out, product, "Clifford rules fired on commutative symbols");
    }

    /// The genuine non-commutative path is untouched by the guard.
    #[test]
    fn noncommutative_generators_still_reduce() {
        let pool = ExprPool::new();
        let sx = pool.symbol_commutative("sx", Domain::Complex, false);
        let sy = pool.symbol_commutative("sy", Domain::Complex, false);

        let xy = run(pool.mul(vec![sx, sy]), &pool, pauli_product_rules());
        let yx = run(pool.mul(vec![sy, sx]), &pool, pauli_product_rules());
        assert_ne!(
            xy, yx,
            "sigma_x sigma_y and sigma_y sigma_x must differ in sign"
        );
    }
}
