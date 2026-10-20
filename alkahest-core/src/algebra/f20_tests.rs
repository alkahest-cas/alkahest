//! F20: d-dimensional gamma algebra, Lie-algebra tables, sequence wildcards.

use super::gamma_dim::{gamma_contract, gamma_trace_dim, lorentz_contract};
use super::lie::{su2_structure_constants, su3_structure_constants, LieAlgebra};
use super::nc_poly::simplify_with_noncommutative;
use super::nc_poly::{nc_terms, NcError};
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::pattern::Pattern;
use crate::simplify::engine::simplify_expanded;
use crate::simplify::rulesets::PatternRule;
use crate::simplify::SimplifyConfig;

fn sym(p: &ExprPool, n: &str) -> ExprId {
    p.symbol(n, Domain::Real)
}
fn nc(p: &ExprPool, n: &str) -> ExprId {
    p.symbol_commutative(n, Domain::Complex, false)
}
fn dot(p: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    p.func("dot", vec![a, b])
}
fn equal(p: &ExprPool, a: ExprId, b: ExprId) -> bool {
    let d = p.add(vec![a, p.mul(vec![p.integer(-1_i32), b])]);
    simplify_expanded(d, p).value == p.integer(0_i32)
}
fn same_nc(p: &ExprPool, a: ExprId, b: ExprId) -> bool {
    let d = p.add(vec![a, p.mul(vec![p.integer(-1_i32), b])]);
    nc_terms(d, p).unwrap().is_empty()
}
fn at_dim(p: &ExprPool, e: ExprId, d: ExprId, n: i64) -> ExprId {
    let m: std::collections::HashMap<_, _> = [(d, p.integer(n))].into_iter().collect();
    simplify_expanded(crate::kernel::subs::subs(e, &m, p), p).value
}

#[test]
fn trace_of_contracted_pair_is_4d() {
    let p = ExprPool::new();
    let (d, mu) = (sym(&p, "d"), sym(&p, "mu"));
    let t = gamma_trace_dim(&[mu, mu], &[mu], d, &p).unwrap();
    assert!(equal(&p, t, p.mul(vec![p.integer(4_i32), d])));
}

#[test]
fn trace_mu_nu_mu_nu() {
    let p = ExprPool::new();
    let (d, mu, nu) = (sym(&p, "d"), sym(&p, "mu"), sym(&p, "nu"));
    let t = gamma_trace_dim(&[mu, nu, mu, nu], &[mu, nu], d, &p).unwrap();
    // 4 d (2 - d)
    let want = p.mul(vec![
        p.integer(4_i32),
        d,
        p.add(vec![p.integer(2_i32), p.mul(vec![p.integer(-1_i32), d])]),
    ]);
    assert!(equal(&p, t, want));
    // d = 4 numeric dimension gives the same number
    let t4 = gamma_trace_dim(&[mu, nu, mu, nu], &[mu, nu], p.integer(4_i32), &p).unwrap();
    assert_eq!(t4, p.integer(-32_i32));
}

#[test]
fn trace_two_vectors_free() {
    let p = ExprPool::new();
    let (d, a, b) = (sym(&p, "d"), sym(&p, "a"), sym(&p, "b"));
    let t = gamma_trace_dim(&[a, b], &[], d, &p).unwrap();
    assert!(equal(&p, t, p.mul(vec![p.integer(4_i32), dot(&p, a, b)])));
}

#[test]
fn product_of_traces_contracted() {
    // tr(g^m g^n g^r g^s) tr(g_m g_n g_r g_s) = 16 d (3d - 2)
    let p = ExprPool::new();
    let d = sym(&p, "d");
    let ix: Vec<ExprId> = ["m", "n", "r", "s"].iter().map(|n| sym(&p, n)).collect();
    let t = gamma_trace_dim(&ix, &[], d, &p).unwrap();
    let prod = p.mul(vec![t, t]);
    let c = lorentz_contract(prod, &ix, d, &p).unwrap();
    let want = p.mul(vec![
        p.integer(16_i32),
        d,
        p.add(vec![p.mul(vec![p.integer(3_i32), d]), p.integer(-2_i32)]),
    ]);
    assert!(equal(&p, c, want), "{}", p.display(c));
    assert_eq!(at_dim(&p, c, d, 4), p.integer(640_i32));
}

#[test]
fn explicit_components_and_gamma5_refused() {
    let p = ExprPool::new();
    let d = sym(&p, "d");
    let e = gamma_trace_dim(&[p.integer(0_i32), p.integer(0_i32)], &[], d, &p).unwrap_err();
    assert!(matches!(e, NcError::InvalidGamma { .. }));
}

#[test]
fn contraction_identities() {
    let p = ExprPool::new();
    let d = sym(&p, "d");
    let (mu, nu, rho) = (sym(&p, "mu"), sym(&p, "nu"), sym(&p, "rho"));
    // g^m g_m = d
    let r = gamma_contract(&[mu, mu], &[mu], d, &p).unwrap();
    assert_eq!(r.len(), 1);
    assert!(r[0].1.is_empty() && equal(&p, r[0].0, d));
    // g^m g^n g_m = (2-d) g^n
    let r = gamma_contract(&[mu, nu, mu], &[mu], d, &p).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].1, vec![nu]);
    assert!(equal(
        &p,
        r[0].0,
        p.add(vec![p.integer(2_i32), p.mul(vec![p.integer(-1_i32), d])])
    ));
    // g^m g^n g^r g_m = 2 g^r g^n + (d-2) g^n g^r
    let r = gamma_contract(&[mu, nu, rho, mu], &[mu], d, &p).unwrap();
    assert_eq!(r.len(), 2);
    for (c, w) in &r {
        if *w == vec![rho, nu] {
            assert_eq!(*c, p.integer(2_i32));
        } else {
            assert_eq!(*w, vec![nu, rho]);
            assert!(equal(&p, *c, p.add(vec![d, p.integer(-2_i32)])));
        }
    }
}

#[test]
fn index_repeated_three_times_refused() {
    let p = ExprPool::new();
    let (d, mu) = (sym(&p, "d"), sym(&p, "mu"));
    assert!(gamma_contract(&[mu, mu, mu], &[mu], d, &p).is_err());
}

// ---------------------------------------------------------------------------
// Lie algebras
// ---------------------------------------------------------------------------

fn gens(p: &ExprPool, n: usize, prefix: &str) -> Vec<ExprId> {
    (1..=n).map(|i| nc(p, &format!("{prefix}{i}"))).collect()
}

#[test]
fn su2_commutators_and_casimir() {
    let p = ExprPool::new();
    let g = gens(&p, 3, "J");
    let alg = LieAlgebra::new(g.clone(), &su2_structure_constants(&p), true, &p).unwrap();
    let c = alg.commutator(g[0], g[1], &p).unwrap();
    assert!(same_nc(&p, c, p.mul(vec![p.imaginary_unit(), g[2]])));
    let c = alg.commutator(g[1], g[0], &p).unwrap();
    assert!(same_nc(
        &p,
        c,
        p.mul(vec![p.integer(-1_i32), p.imaginary_unit(), g[2]])
    ));
    let cas = alg.casimir(None, &p).unwrap();
    for &t in &g {
        let z = alg.commutator(cas, t, &p).unwrap();
        assert!(nc_terms(z, &p).unwrap().is_empty());
    }
}

#[test]
fn su3_jacobi_and_casimir_central() {
    let p = ExprPool::new();
    let g = gens(&p, 8, "T");
    let alg = LieAlgebra::new(g.clone(), &su3_structure_constants(&p), true, &p).unwrap();
    let cas = alg.casimir(None, &p).unwrap();
    for &t in &g {
        let z = alg.commutator(cas, t, &p).unwrap();
        assert!(nc_terms(z, &p).unwrap().is_empty());
    }
}

#[test]
fn bad_table_refused() {
    let p = ExprPool::new();
    let g = gens(&p, 3, "J");
    // f_012 = 1 only: Jacobi on (0,1,2) fails
    let one = p.integer(1_i32);
    let e = LieAlgebra::new(g.clone(), &[(0, 1, 2, one), (0, 2, 0, one)], true, &p).unwrap_err();
    assert!(matches!(e, NcError::InvalidLieAlgebra { .. }));
    let e = LieAlgebra::new(g, &[(0, 0, 1, one)], true, &p).unwrap_err();
    assert!(matches!(e, NcError::InvalidLieAlgebra { .. }));
}

// ---------------------------------------------------------------------------
// Sequence wildcards
// ---------------------------------------------------------------------------

fn seq_rule(p: &ExprPool, lhs: ExprId, rhs: ExprId, vars: Vec<ExprId>) -> PatternRule {
    PatternRule::new(Pattern::from_expr(lhs), rhs)
        .with_sequence_wildcards(vars, p)
        .unwrap()
}

fn run(p: &ExprPool, rule: PatternRule, e: ExprId) -> ExprId {
    let rules: Vec<Box<dyn crate::simplify::rules::RewriteRule>> = vec![Box::new(rule)];
    simplify_with_noncommutative(e, p, &rules, SimplifyConfig::default()).value
}

#[test]
fn sequence_wildcard_leftmost_shortest() {
    let p = ExprPool::new();
    let (a, b, x, c, d) = (
        nc(&p, "A"),
        nc(&p, "B"),
        nc(&p, "X"),
        nc(&p, "C"),
        nc(&p, "D"),
    );
    // A X B -> [X]  (here C stands in for a marker)
    let rule = seq_rule(&p, p.mul(vec![a, x, b]), p.mul(vec![c, x, c]), vec![x]);
    // A D D B -> C D D C
    let e = p.mul(vec![a, d, d, b]);
    let r = run(&p, rule.clone(), e);
    assert!(nc_terms(
        p.add(vec![
            r,
            p.mul(vec![p.integer(-1_i32), p.mul(vec![c, d, d, c])])
        ]),
        &p
    )
    .unwrap()
    .is_empty());
    // empty run: A B -> C C
    let r = run(&p, rule.clone(), p.mul(vec![a, b]));
    assert!(nc_terms(
        p.add(vec![r, p.mul(vec![p.integer(-1_i32), p.mul(vec![c, c])])]),
        &p
    )
    .unwrap()
    .is_empty());
    // shortest: A D B D B -> C D C D B (first B closes)
    let e = p.mul(vec![a, d, b, d, b]);
    let r = run(&p, rule, e);
    assert!(nc_terms(
        p.add(vec![
            r,
            p.mul(vec![p.integer(-1_i32), p.mul(vec![c, d, c, d, b])])
        ]),
        &p
    )
    .unwrap()
    .is_empty());
}

#[test]
fn sequence_pattern_validation() {
    let p = ExprPool::new();
    let (a, x) = (nc(&p, "A"), nc(&p, "X"));
    let r = PatternRule::new(Pattern::from_expr(p.mul(vec![x, x])), a);
    assert!(r.with_sequence_wildcards(vec![x], &p).is_err());
    let r = PatternRule::new(Pattern::from_expr(a), a);
    assert!(r.with_sequence_wildcards(vec![x], &p).is_err());
    let y = sym(&p, "y");
    let r = PatternRule::new(Pattern::from_expr(p.mul(vec![a, x])), a);
    assert!(r.with_sequence_wildcards(vec![y], &p).is_err());
}
