//! Tests for the ℚ(params) partial-fraction path.
//!
//! Every decomposition is checked *numerically*: the input and the output are
//! evaluated at several random parameter/variable points and must agree.  A
//! partial-fraction identity that only holds symbolically is exactly the class
//! of bug this module is here to avoid.

use super::*;
use crate::deriv::SideCondition;
use crate::kernel::{Domain, ExprData};
use std::collections::HashMap;

fn pool_with(names: &[&str]) -> (ExprPool, HashMap<String, ExprId>) {
    let p = ExprPool::new();
    let mut syms = HashMap::new();
    for n in names {
        syms.insert((*n).to_owned(), p.symbol(*n, Domain::Real));
    }
    (p, syms)
}

/// Numeric evaluator over `Integer/Rational/Add/Mul/Pow/Symbol`.
fn eval(expr: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> f64 {
    match pool.get(expr) {
        ExprData::Integer(n) => n.0.to_f64(),
        ExprData::Rational(r) => {
            let (n, d) = r.0.clone().into_numer_denom();
            n.to_f64() / d.to_f64()
        }
        ExprData::Symbol { .. } => *env
            .get(&expr)
            .unwrap_or_else(|| panic!("unbound symbol {}", pool.display(expr))),
        ExprData::Add(args) => args.iter().map(|&a| eval(a, env, pool)).sum(),
        ExprData::Mul(args) => args.iter().map(|&a| eval(a, env, pool)).product(),
        ExprData::Pow { base, exp } => {
            let b = eval(base, env, pool);
            let e = eval(exp, env, pool);
            b.powf(e)
        }
        other => panic!("eval: unsupported {other:?}"),
    }
}

/// Decompose, then assert the result agrees with the input at every sample.
fn check(
    src_num: ExprId,
    var: ExprId,
    samples: &[HashMap<ExprId, f64>],
    pool: &ExprPool,
) -> ParamApart {
    let d = apart_param(src_num, var, pool).expect("apart_param failed");
    for env in samples {
        let lhs = eval(src_num, env, pool);
        let rhs = eval(d.expr, env, pool);
        assert!(
            (lhs - rhs).abs() < 1e-9 * (1.0 + lhs.abs()),
            "apart_param ≠ input: {lhs} vs {rhs}\n  in  = {}\n  out = {}",
            pool.display(src_num),
            pool.display(d.expr)
        );
    }
    d
}

fn cond_strings(d: &ParamApart, pool: &ExprPool) -> Vec<String> {
    d.conditions
        .iter()
        .map(|c| c.display_with(pool).to_string())
        .collect()
}

#[test]
fn distinct_symbolic_linear_poles_report_the_coincidence() {
    // 1/((s+ka)(s+ke)) — the Bateman denominator.  Valid only for ka ≠ ke.
    let (p, sy) = pool_with(&["s", "ka", "ke"]);
    let (s, ka, ke) = (sy["s"], sy["ka"], sy["ke"]);
    let f = p.pow(
        p.mul(vec![p.add(vec![s, ka]), p.add(vec![s, ke])]),
        p.integer(-1_i32),
    );

    let samples = vec![
        HashMap::from([(s, 1.3), (ka, 0.7), (ke, 2.1)]),
        HashMap::from([(s, -0.4), (ka, 3.5), (ke, 0.25)]),
        HashMap::from([(s, 5.0), (ka, -1.25), (ke, 4.0)]),
    ];
    let d = check(f, s, &samples, &p);

    let conds = cond_strings(&d, &p);
    assert_eq!(conds.len(), 1, "expected exactly one hypothesis: {conds:?}");
    // ka − ke ≠ 0, up to the canonical sign.
    let c = &conds[0];
    assert!(
        c.contains("ka") && c.contains("ke") && c.ends_with("≠ 0"),
        "expected the ka/ke coincidence condition, got {c}"
    );
    assert!(matches!(d.conditions[0], SideCondition::NonZero(_)));
}

#[test]
fn irreducible_quadratic_over_the_parameter_field_is_kept_intact() {
    // 1/(s² + w²) is irreducible over ℚ(w): −w² is not a square there.
    let (p, sy) = pool_with(&["s", "w"]);
    let (s, w) = (sy["s"], sy["w"]);
    let den = p.add(vec![p.pow(s, p.integer(2_i32)), p.pow(w, p.integer(2_i32))]);
    let f = p.pow(den, p.integer(-1_i32));

    let samples = vec![
        HashMap::from([(s, 1.1), (w, 2.0)]),
        HashMap::from([(s, -3.0), (w, 0.5)]),
    ];
    let d = check(f, s, &samples, &p);
    // No division by a parameter polynomial happened, so no hypothesis.
    assert!(
        d.conditions.is_empty(),
        "unexpected conditions: {:?}",
        cond_strings(&d, &p)
    );
    let out = p.display(d.expr).to_string();
    assert!(!out.contains("I"), "quadratic must not be split: {out}");
}

#[test]
fn non_monic_denominator_reports_its_leading_coefficient() {
    // 1/(a·s + b): making the pole monic divides by `a`, and at a = 0 the input
    // is the constant 1/b, whose inverse transform is a δ rather than an
    // exponential.  That division has to be visible.
    let (p, sy) = pool_with(&["s", "a", "b"]);
    let (s, a, b) = (sy["s"], sy["a"], sy["b"]);
    let f = p.pow(p.add(vec![p.mul(vec![a, s]), b]), p.integer(-1_i32));

    let samples = vec![
        HashMap::from([(s, 1.0), (a, 2.0), (b, 3.0)]),
        HashMap::from([(s, -2.5), (a, 0.5), (b, -1.0)]),
    ];
    let d = check(f, s, &samples, &p);
    let conds = cond_strings(&d, &p);
    assert_eq!(conds, vec!["a ≠ 0".to_string()], "got {conds:?}");
}

#[test]
fn a_parameter_the_input_already_divides_by_is_not_a_new_hypothesis() {
    // 1/(w·(s+1)): the input is already undefined at w = 0, so the
    // decomposition is not making a new claim there.
    let (p, sy) = pool_with(&["s", "w"]);
    let (s, w) = (sy["s"], sy["w"]);
    let f = p.pow(
        p.mul(vec![w, p.add(vec![s, p.integer(1_i32)])]),
        p.integer(-1_i32),
    );
    let samples = vec![
        HashMap::from([(s, 2.0), (w, 3.0)]),
        HashMap::from([(s, -0.5), (w, -2.0)]),
    ];
    let d = check(f, s, &samples, &p);
    assert!(
        d.conditions.is_empty(),
        "w ≠ 0 is the input's own pole, not a new hypothesis: {:?}",
        cond_strings(&d, &p)
    );
}

#[test]
fn repeated_symbolic_pole_needs_no_hypothesis() {
    // 1/(s+a)² has one factor; nothing is divided by, so nothing is assumed.
    let (p, sy) = pool_with(&["s", "a"]);
    let (s, a) = (sy["s"], sy["a"]);
    let f = p.pow(p.add(vec![s, a]), p.integer(-2_i32));
    let samples = vec![
        HashMap::from([(s, 1.0), (a, 2.0)]),
        HashMap::from([(s, 4.0), (a, -1.5)]),
    ];
    let d = check(f, s, &samples, &p);
    assert!(d.conditions.is_empty(), "{:?}", cond_strings(&d, &p));
}

#[test]
fn three_distinct_symbolic_poles() {
    // D/((s+a)(s+b)(s+c)) — three pairwise coincidences to report.
    let (p, sy) = pool_with(&["s", "a", "b", "c", "D"]);
    let (s, a, b, c, dsym) = (sy["s"], sy["a"], sy["b"], sy["c"], sy["D"]);
    let den = p.mul(vec![
        p.add(vec![s, a]),
        p.add(vec![s, b]),
        p.add(vec![s, c]),
    ]);
    let f = p.mul(vec![dsym, p.pow(den, p.integer(-1_i32))]);
    let samples = vec![
        HashMap::from([(s, 1.0), (a, 0.5), (b, 2.0), (c, 3.5), (dsym, 7.0)]),
        HashMap::from([(s, -4.0), (a, 1.5), (b, -0.25), (c, 6.0), (dsym, -2.0)]),
    ];
    let d = check(f, s, &samples, &p);
    assert_eq!(
        d.conditions.len(),
        3,
        "expected a−b, a−c, b−c: {:?}",
        cond_strings(&d, &p)
    );
}

#[test]
fn improper_parametric_rational_keeps_its_polynomial_part() {
    // (s² + a)/(s + b) = s − b + (a + b²)/(s + b).
    let (p, sy) = pool_with(&["s", "a", "b"]);
    let (s, a, b) = (sy["s"], sy["a"], sy["b"]);
    let f = p.mul(vec![
        p.add(vec![p.pow(s, p.integer(2_i32)), a]),
        p.pow(p.add(vec![s, b]), p.integer(-1_i32)),
    ]);
    let samples = vec![
        HashMap::from([(s, 2.0), (a, 3.0), (b, 1.0)]),
        HashMap::from([(s, -1.5), (a, -2.0), (b, 4.0)]),
    ];
    let d = check(f, s, &samples, &p);
    assert!(d.conditions.is_empty(), "{:?}", cond_strings(&d, &p));
}

#[test]
fn transcendental_generators_are_opaque_constants() {
    // 1/((s + sin(y))(s + 1)) — `sin(y)` is a generator, not a rational
    // function of `s`, and the decomposition treats it as one more parameter.
    let (p, sy) = pool_with(&["s", "y"]);
    let (s, y) = (sy["s"], sy["y"]);
    let sy_ = p.func("sin", vec![y]);
    let den = p.mul(vec![p.add(vec![s, sy_]), p.add(vec![s, p.integer(1_i32)])]);
    let f = p.pow(den, p.integer(-1_i32));
    let d = apart_param(f, s, &p).expect("apart_param failed");
    // sin(y) − 1 ≠ 0 must be reported.
    assert_eq!(d.conditions.len(), 1, "{:?}", cond_strings(&d, &p));
    let c = cond_strings(&d, &p).remove(0);
    assert!(c.contains("sin"), "{c}");
}

#[test]
fn no_var_at_all_is_returned_unchanged_in_value() {
    let (p, sy) = pool_with(&["s", "a"]);
    let (s, a) = (sy["s"], sy["a"]);
    let f = p.pow(a, p.integer(-1_i32));
    let samples = vec![HashMap::from([(s, 1.0), (a, 2.0)])];
    check(f, s, &samples, &p);
}
