use super::engine::simplify;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::poly::UniPoly;
use proptest::prelude::*;

fn small_coeff() -> impl Strategy<Value = i64> {
    -5i64..=5i64
}

/// Build a random polynomial expression tree in `x` from coefficient slices.
/// Returns the ExprId of the expression in the pool.
fn poly_expr(pool: &ExprPool, x: ExprId, coeffs: &[i64]) -> ExprId {
    // p(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
    let mut terms: Vec<ExprId> = vec![];
    for (i, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let c_id = pool.integer(c);
        if i == 0 {
            terms.push(c_id);
        } else {
            let deg = pool.integer(i as i32);
            let xpow = pool.pow(x, deg);
            if c == 1 {
                terms.push(xpow);
            } else {
                terms.push(pool.mul(vec![c_id, xpow]));
            }
        }
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn simplify_idempotent(
        coeffs in proptest::collection::vec(small_coeff(), 1..=4),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = poly_expr(&pool, x, &coeffs);
        let first = simplify(expr, &pool);
        let second = simplify(first.value, &pool);
        prop_assert_eq!(
            first.value, second.value,
            "simplify(simplify(e)) != simplify(e) for coeffs={:?}", coeffs
        );
    }

    #[test]
    fn simplify_constant_zero_is_zero(
        coeffs in proptest::collection::vec(Just(0i64), 1..=4),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = poly_expr(&pool, x, &coeffs);
        let r = simplify(expr, &pool);
        // All-zero polynomial should simplify to Integer(0)
        prop_assert_eq!(r.value, pool.integer(0_i32));
    }

    #[test]
    fn simplify_integer_constant_folds(a in -100i64..=100i64, b in -100i64..=100i64) {
        let pool = ExprPool::new();
        let expr = pool.add(vec![pool.integer(a), pool.integer(b)]);
        let r = simplify(expr, &pool);
        prop_assert_eq!(r.value, pool.integer(a + b));
    }

    #[test]
    fn simplify_mul_constant_folds(a in -20i64..=20i64, b in -20i64..=20i64) {
        let pool = ExprPool::new();
        let expr = pool.mul(vec![pool.integer(a), pool.integer(b)]);
        let r = simplify(expr, &pool);
        prop_assert_eq!(r.value, pool.integer(a * b));
    }

    /// sqrt(n) → integer root when n is a perfect square (n > 0).
    #[test]
    fn simplify_sqrt_perfect_square(n in 1u32..=50u32) {
        let pool = ExprPool::new();
        let n_sq = (n as i64) * (n as i64);
        let expr = pool.func("sqrt", vec![pool.integer(n_sq)]);
        let r = simplify(expr, &pool);
        prop_assert_eq!(r.value, pool.integer(n as i64));
    }

    #[test]
    fn simplify_preserves_polynomial_value(
        coeffs in proptest::collection::vec(small_coeff(), 1..=4),
    ) {
        // Convert to UniPoly before and after simplification; coefficients must match.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = poly_expr(&pool, x, &coeffs);
        let r = simplify(expr, &pool);

        // Both should convert to polynomial in x (if they're polynomials)
        let poly_before = UniPoly::from_symbolic(expr, x, &pool);
        let poly_after = UniPoly::from_symbolic(r.value, x, &pool);
        if let (Ok(pb), Ok(pa)) = (poly_before, poly_after) {
            prop_assert_eq!(
                pb.coefficients_i64(), pa.coefficients_i64(),
                "polynomial changed under simplification"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Differential tests for the DAG / hot-rule performance work.
//
// Each compares a fast path against the algorithm it replaced, kept verbatim
// as a `#[cfg(test)]` oracle, on inputs generated to exercise sharing.
// ---------------------------------------------------------------------------

use super::assumptions::{collect_static_domain_facts, collect_static_domain_facts_reference};
use super::engine::{rules_for_config, simplify_with_carry, SimplifyConfig};
use super::rules::reference::{DivSelfReference, SubSelfReference};
use super::rules::{DivSelf, RewriteRule, SubSelf};
use crate::deriv::log::{DerivationLog, SideCondition};

/// One construction step of a random DAG: an operator and the (wrapped)
/// indices of earlier nodes it combines.  Later nodes reuse earlier ones, so
/// the result shares subexpressions the way real derivations do.
pub(super) type Op = (u8, usize, usize, i8);

pub(super) fn dag_ops(max: usize) -> impl Strategy<Value = Vec<Op>> {
    proptest::collection::vec(
        (any::<u8>(), any::<usize>(), any::<usize>(), -3i8..=3),
        1..=max,
    )
}

/// Build the DAG `ops` describes.  With `domains`, two of the leaf symbols
/// carry the `Positive` / `NonZero` domains that static fact collection keys
/// on; without, every symbol is `Real`, which keeps `simplify` off the
/// assumption-driven e-graph pass.
pub(super) fn build_dag(pool: &ExprPool, ops: &[Op], domains: bool) -> ExprId {
    let (dy, dz) = if domains {
        (Domain::Positive, Domain::NonZero)
    } else {
        (Domain::Real, Domain::Real)
    };
    let mut nodes = vec![
        pool.symbol("x", Domain::Real),
        pool.symbol("y", dy),
        pool.symbol("z", dz),
        pool.integer(0),
        pool.integer(1),
        pool.integer(-1),
        pool.integer(2),
        pool.rational(1, 2),
    ];
    for &(op, a, b, k) in ops {
        let a = nodes[a % nodes.len()];
        let b = nodes[b % nodes.len()];
        let n = match op % 10 {
            0 => pool.add(vec![a, b]),
            1 => pool.mul(vec![a, b]),
            2 => pool.pow(a, pool.integer(k)),
            3 => pool.add(vec![a, pool.mul(vec![pool.integer(-1), b])]),
            4 => pool.mul(vec![a, pool.pow(b, pool.integer(-1))]),
            5 => {
                let f = ["sin", "cos", "exp", "log", "sqrt"][usize::from(op / 10) % 5];
                pool.func(f, vec![a])
            }
            6 => pool.mul(vec![pool.integer(k), a]),
            7 => pool.add(vec![a, b, a]),
            8 => pool.mul(vec![a, b, a]),
            _ => pool.add(vec![pool.mul(vec![pool.integer(2), a, b]), a]),
        };
        nodes.push(n);
    }
    *nodes.last().unwrap()
}

type Step = (&'static str, ExprId, ExprId, Vec<SideCondition>);

fn steps(log: &DerivationLog) -> Vec<Step> {
    log.steps()
        .iter()
        .map(|s| (s.rule_name, s.before, s.after, s.side_conditions.clone()))
        .collect()
}

fn outcome(r: Option<(ExprId, DerivationLog)>) -> Option<(ExprId, Vec<Step>)> {
    r.map(|(e, log)| (e, steps(&log)))
}

/// Bases for the `DivSelf` / `SubSelf` differentials: symbols, a
/// non-commuting symbol, literal `0` and `2`, and compound bases.
fn factor_bases(pool: &ExprPool) -> Vec<ExprId> {
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    vec![
        x,
        y,
        pool.symbol_commutative("A", Domain::Real, false),
        pool.integer(0),
        pool.integer(2),
        pool.func("sin", vec![x]),
        pool.add(vec![x, pool.integer(1)]),
        pool.mul(vec![x, y]),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The visited-set walk collects exactly the facts, in exactly the order,
    /// of the one-visit-per-path walk it replaced.
    #[test]
    fn static_domain_facts_match_tree_walk(ops in dag_ops(10)) {
        let pool = ExprPool::new();
        let e = build_dag(&pool, &ops, true);
        let (mut fast, mut slow) = (Vec::new(), Vec::new());
        collect_static_domain_facts(e, &pool, &mut fast);
        collect_static_domain_facts_reference(e, &pool, &mut slow);
        prop_assert_eq!(fast, slow);
    }

    /// `collect_mul_factors`' allocation-free reject path declines exactly
    /// when the full exponent map would have, and fires identically otherwise.
    #[test]
    fn div_self_matches_reference(
        parts in proptest::collection::vec((0u8..8, 0u8..7), 1..=6),
    ) {
        let pool = ExprPool::new();
        let bases = factor_bases(&pool);
        let factors: Vec<ExprId> = parts
            .iter()
            .map(|&(b, e)| {
                let base = bases[usize::from(b)];
                match e {
                    // 5 and 6: the bare factor, implicit exponent 1.
                    5 | 6 => base,
                    e => pool.pow(base, pool.integer(i64::from(e) - 2)),
                }
            })
            .collect();
        let expr = pool.mul(factors);
        prop_assert_eq!(
            outcome(DivSelf.apply(expr, &pool)),
            outcome(DivSelfReference.apply(expr, &pool))
        );
    }

    /// Same for `collect_add_terms` on sums with repeated bases, zero and
    /// rational coefficients.
    #[test]
    fn sub_self_matches_reference(
        parts in proptest::collection::vec((0u8..6, 0u8..8), 1..=6),
    ) {
        let pool = ExprPool::new();
        let bases = factor_bases(&pool);
        let terms: Vec<ExprId> = parts
            .iter()
            .map(|&(c, b)| {
                let base = bases[usize::from(b)];
                match c {
                    0 => base,
                    1 => pool.mul(vec![pool.integer(-1), base]),
                    2 => pool.mul(vec![pool.integer(0), base]),
                    3 => pool.mul(vec![pool.rational(1, 2), base]),
                    4 => pool.mul(vec![pool.rational(-1, 2), base]),
                    _ => pool.mul(vec![pool.integer(3), base]),
                }
            })
            .collect();
        let expr = pool.add(terms);
        prop_assert_eq!(
            outcome(SubSelf.apply(expr, &pool)),
            outcome(SubSelfReference.apply(expr, &pool))
        );
    }

    /// Carrying settled memo entries between passes changes neither the
    /// result nor a single step of the log, compared with the historical
    /// fresh-memo-per-pass loop.
    #[test]
    fn settled_memo_carry_matches_fresh_memo(ops in dag_ops(14)) {
        let pool = ExprPool::new();
        let e = build_dag(&pool, &ops, false);
        let config = SimplifyConfig::default();
        let rules = rules_for_config(&config);
        let carried = simplify_with_carry(e, &pool, &rules, config.clone(), true);
        let fresh = simplify_with_carry(e, &pool, &rules, config, false);
        prop_assert_eq!(carried.value, fresh.value);
        prop_assert_eq!(steps(&carried.log), steps(&fresh.log));
    }
}
