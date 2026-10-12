use super::*;
use crate::eval::{eval_complex_f64, ComplexF64};

fn int(p: &ExprPool, n: i32) -> ExprId {
    p.integer(n)
}

fn neg(p: &ExprPool, e: ExprId) -> ExprId {
    p.mul(vec![p.integer(-1_i32), e])
}

fn sq(p: &ExprPool, e: ExprId) -> ExprId {
    p.pow(e, p.integer(2_i32))
}

fn contains_i(e: ExprId, p: &ExprPool) -> bool {
    match p.get(e) {
        ExprData::Symbol { .. } => p.is_imaginary_unit(e),
        ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
            a.iter().any(|&x| contains_i(x, p))
        }
        ExprData::Pow { base, exp } => contains_i(base, p) || contains_i(exp, p),
        _ => false,
    }
}

/// Evaluate `a` and `b` in complex double precision at `point` and require
/// agreement, plus a real `b`.
fn agree(a: ExprId, b: ExprId, point: &[(ExprId, f64)], p: &ExprPool) {
    let bind: HashMap<ExprId, ComplexF64> = point
        .iter()
        .map(|&(s, v)| (s, ComplexF64::new(v, 0.0)))
        .collect();
    let va = eval_complex_f64(a, p, &bind).expect("original evaluates");
    let vb = eval_complex_f64(b, p, &bind).expect("rewrite evaluates");
    let scale = 1.0 + va.re.abs() + va.im.abs();
    assert!(
        (va.re - vb.re).abs() < 1e-9 * scale && (va.im - vb.im).abs() < 1e-9 * scale,
        "{} = {va:?} but {} = {vb:?}",
        p.display(a),
        p.display(b)
    );
    assert!(vb.im.abs() < 1e-12 * scale, "rewrite not real: {vb:?}");
}

#[test]
fn sqrt_of_square_is_abs_or_identity_under_sign() {
    let p = ExprPool::new();
    let a = p.symbol("a", Domain::Real);
    let e = p.func("sqrt", vec![p.mul(vec![int(&p, 4), sq(&p, a)])]);
    let r = to_real_form(e, &p, &AssumptionContext::new());
    assert_eq!(
        p.display(r).to_string(),
        p.display(p.mul(vec![int(&p, 2), p.func("abs", vec![a])]))
            .to_string()
    );

    let mut ctx = AssumptionContext::new();
    ctx.refine(p.pred_ge(a, int(&p, 0)), &p).unwrap();
    let r = to_real_form(e, &p, &ctx);
    assert_eq!(
        p.display(r).to_string(),
        p.display(p.mul(vec![int(&p, 2), a])).to_string()
    );
}

#[test]
fn two_dof_radicand_is_collected() {
    // √(9(mk)² − 4k²m²) = √5·k·m for positive k, m.
    let p = ExprPool::new();
    let k = p.symbol("k", Domain::Positive);
    let m = p.symbol("m", Domain::Positive);
    let rad = p.add(vec![
        p.mul(vec![int(&p, 9), sq(&p, p.mul(vec![m, k]))]),
        p.mul(vec![int(&p, -4), sq(&p, k), sq(&p, m)]),
    ]);
    let e = p.func("sqrt", vec![rad]);
    let r = to_real_form(e, &p, &AssumptionContext::new());
    assert!(!looks_complex(p.mul(vec![k, m]), &p));
    agree(e, r, &[(k, 1.3), (m, 0.7)], &p);
    let shown = p.display(r).to_string();
    assert!(shown.contains('5') && !shown.contains("abs"), "{shown}");
}

#[test]
fn two_compartment_discriminant_is_positive() {
    let p = ExprPool::new();
    let a = p.symbol("k10", Domain::Positive);
    let b = p.symbol("k12", Domain::Positive);
    let c = p.symbol("k21", Domain::Positive);
    // (a+b+c)² − 4ac
    let d = p.add(vec![
        sq(&p, p.add(vec![a, b, c])),
        p.mul(vec![int(&p, -4), a, c]),
    ]);
    assert_eq!(real_sign(d, &p, None), Some(Sign::Positive));
    assert_eq!(real_sign(neg(&p, d), &p, None), Some(Sign::Negative));
    // Without positivity nothing is claimed... except that it is real.
    let x = p.symbol("x", Domain::Real);
    let y = p.symbol("y", Domain::Real);
    let e = p.add(vec![x, neg(&p, y)]);
    assert_eq!(real_sign(e, &p, None), None);
}

#[test]
fn damped_oscillator_discriminant_under_assumptions() {
    let p = ExprPool::new();
    let z = p.symbol("z", Domain::Real);
    let w = p.symbol("w", Domain::Real);
    // 4w²(z² − 1), expanded.
    let d = p.add(vec![
        p.mul(vec![int(&p, 4), sq(&p, w), sq(&p, z)]),
        p.mul(vec![int(&p, -4), sq(&p, w)]),
    ]);
    assert_eq!(real_sign(d, &p, None), None);
    let mut ctx = AssumptionContext::new();
    ctx.refine(p.pred_gt(z, int(&p, 0)), &p).unwrap();
    ctx.refine(p.pred_lt(z, int(&p, 1)), &p).unwrap();
    ctx.refine(p.pred_gt(w, int(&p, 0)), &p).unwrap();
    assert_eq!(real_sign(d, &p, Some(&ctx)), Some(Sign::Negative));
}

#[test]
fn conjugate_exponentials_become_cosine() {
    let p = ExprPool::new();
    let w = p.symbol("w", Domain::Real);
    let t = p.symbol("t", Domain::Real);
    let i = p.imaginary_unit();
    let wt = p.mul(vec![w, t]);
    let e = p.mul(vec![
        p.rational(1, 2),
        p.add(vec![
            p.func("exp", vec![p.mul(vec![i, wt])]),
            p.func("exp", vec![neg(&p, p.mul(vec![i, wt]))]),
        ]),
    ]);
    let r = to_real_form(e, &p, &AssumptionContext::new());
    assert!(!contains_i(r, &p), "{}", p.display(r));
    agree(e, r, &[(w, 1.7), (t, 0.4)], &p);
}

#[test]
fn oscillator_matrix_exponential_entry() {
    // (e^{√(−4w²t²)/2} + e^{−√(−4w²t²)/2})/2 = cos(w t) for all real w, t.
    let p = ExprPool::new();
    let w = p.symbol("w", Domain::Real);
    let t = p.symbol("t", Domain::Real);
    let rad = p.mul(vec![int(&p, -4), sq(&p, w), sq(&p, t)]);
    let root = p.pow(rad, p.rational(1, 2));
    let half = p.rational(1, 2);
    let e = p.mul(vec![
        half,
        p.add(vec![
            p.func("exp", vec![p.mul(vec![half, root])]),
            p.func("exp", vec![p.mul(vec![p.rational(-1, 2), root])]),
        ]),
    ]);
    let r = to_real_form(e, &p, &AssumptionContext::new());
    let shown = p.display(r).to_string();
    assert!(shown.contains("cos") && !shown.contains("abs"), "{shown}");
    agree(e, r, &[(w, 2.0), (t, 0.7)], &p);
    agree(e, r, &[(w, -1.1), (t, 0.3)], &p);
}

#[test]
fn cosh_of_imaginary_is_cos() {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let e = p.func("cosh", vec![p.mul(vec![p.imaginary_unit(), x])]);
    let r = to_real_form(e, &p, &AssumptionContext::new());
    assert_eq!(
        p.display(r).to_string(),
        p.display(p.func("cos", vec![x])).to_string()
    );
}

#[test]
fn cos_of_root_of_negative_becomes_cosh() {
    // cos(√(−a²−1)·t) with a real a → cosh(√(a²+1)·t)
    let p = ExprPool::new();
    let a = p.symbol("a", Domain::Real);
    let t = p.symbol("t", Domain::Real);
    let rad = p.add(vec![neg(&p, sq(&p, a)), int(&p, -1)]);
    let e = p.func("cos", vec![p.mul(vec![p.pow(rad, p.rational(1, 2)), t])]);
    let r = to_real_form(e, &p, &AssumptionContext::new());
    let shown = p.display(r).to_string();
    assert!(shown.contains("cosh"), "{shown}");
    let bind: HashMap<ExprId, ComplexF64> = [(a, 0.8), (t, 1.3)]
        .iter()
        .map(|&(s, v)| (s, ComplexF64::new(v, 0.0)))
        .collect();
    let v = eval_complex_f64(r, &p, &bind).unwrap();
    let want = ((0.8f64 * 0.8 + 1.0).sqrt() * 1.3).cosh();
    assert!(
        (v.re - want).abs() < 1e-12 && v.im.abs() < 1e-12,
        "{v:?} vs {want}"
    );
    // And the original agrees in complex arithmetic (cos of an imaginary).
    agree(e, r, &[(a, 0.8), (t, 1.3)], &p);
}

#[test]
fn nonvanishing_imaginary_part_is_kept() {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let e = p.func("exp", vec![p.mul(vec![p.imaginary_unit(), x])]);
    assert_eq!(to_real_form(e, &p, &AssumptionContext::new()), e);
    // Unknown-sign radicand: nothing to prove, nothing changed.
    let y = p.symbol("y", Domain::Real);
    let s = p.func("cos", vec![p.func("sqrt", vec![y])]);
    assert_eq!(to_real_form(s, &p, &AssumptionContext::new()), s);
}

#[test]
fn complex_symbols_are_left_alone() {
    let p = ExprPool::new();
    let z = p.symbol("z", Domain::Complex);
    let e = p.func("sqrt", vec![p.mul(vec![int(&p, 4), sq(&p, z)])]);
    assert_eq!(to_real_form(e, &p, &AssumptionContext::new()), e);
}

#[test]
fn sign_set_algebra() {
    assert!(SignSet::POS.mul(SignSet::NEG).is_neg());
    assert!(SignSet::POS.add(SignSet::NONNEG).is_pos());
    assert_eq!(SignSet::POS.add(SignSet::NEG), SignSet::ANY);
    assert!(SignSet::ANY.pow_int(2).is_nonneg());
    assert!(SignSet::NONZERO.pow_int(2).is_pos());
    assert!(SignSet::NEG.pow_int(-1).is_neg());
}
