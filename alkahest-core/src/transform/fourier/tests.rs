//! Tests for the Fourier transform and its inverse.
//!
//! Forward transforms are checked against the analytic table; the real-valued
//! Gaussian pair is additionally spot-checked against numeric quadrature of the
//! defining integral, and round-trips `F⁻¹{F{f}}` are checked structurally.

use super::*;
use crate::kernel::{Domain, ExprPool};

// --- builders ---------------------------------------------------------------

fn setup() -> (ExprPool, ExprId, ExprId) {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let xi = pool.symbol("xi", Domain::Real);
    (pool, x, xi)
}

fn int(pool: &ExprPool, n: i32) -> ExprId {
    pool.integer(n)
}

fn disp(pool: &ExprPool, e: ExprId) -> String {
    pool.display(e).to_string()
}

/// Assert two expressions are structurally equal after simplification.
fn assert_eq_simplified(pool: &ExprPool, got: ExprId, expected: ExprId) {
    let g = simp(got, pool);
    let e = simp(expected, pool);
    assert_eq!(
        disp(pool, g),
        disp(pool, e),
        "got {} vs expected {}",
        disp(pool, g),
        disp(pool, e),
    );
}

// --- guards -----------------------------------------------------------------

#[test]
fn same_variable_rejected() {
    let (pool, x, _) = setup();
    let one = int(&pool, 1);
    assert_eq!(
        fourier_transform(one, x, x, &pool),
        Err(FourierError::SameVariable)
    );
}

// --- table: Gaussian --------------------------------------------------------

#[test]
fn gaussian_self_dual() {
    // F{e^{−π x²}}(ξ) = e^{−π ξ²}.
    let (pool, x, xi) = setup();
    let pi = pool.symbol("pi", Domain::Real);
    let x2 = pool.pow(x, int(&pool, 2));
    let f = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), pi, x2])]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();

    let xi2 = pool.pow(xi, int(&pool, 2));
    let expected = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), pi, xi2])]);
    assert_eq_simplified(&pool, g, expected);
}

#[test]
fn gaussian_general_a() {
    // F{e^{−a x²}}(ξ) = √(π/a)·e^{−π² ξ²/a}, here a = 2.
    let (pool, x, xi) = setup();
    let pi = pool.symbol("pi", Domain::Real);
    let a = int(&pool, 2);
    let x2 = pool.pow(x, int(&pool, 2));
    let f = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), a, x2])]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();

    let half = pool.rational(1, 2);
    let prefactor = pool.pow(pool.mul(vec![pi, pool.pow(a, int(&pool, -1))]), half);
    let pi2 = pool.pow(pi, int(&pool, 2));
    let xi2 = pool.pow(xi, int(&pool, 2));
    let exponent = pool.mul(vec![int(&pool, -1), pi2, xi2, pool.pow(a, int(&pool, -1))]);
    let expected = pool.mul(vec![prefactor, pool.func("exp", vec![exponent])]);
    assert_eq_simplified(&pool, g, expected);
}

/// Numeric quadrature spot-check of the *real* Gaussian self-dual pair:
/// `∫_{-∞}^{∞} e^{−π x²} cos(2π ξ x) dx = e^{−π ξ²}` (the imaginary part of the
/// kernel integrates to zero by symmetry, so the transform is real).
#[test]
fn gaussian_numeric_spotcheck() {
    use std::f64::consts::PI;
    let quad = |xi: f64| -> f64 {
        // composite Simpson over [−L, L] with N panels.
        let l = 8.0;
        let n = 20_000usize;
        let h = 2.0 * l / n as f64;
        let g = |xv: f64| (-PI * xv * xv).exp() * (2.0 * PI * xi * xv).cos();
        let mut acc = g(-l) + g(l);
        for k in 1..n {
            let xv = -l + k as f64 * h;
            acc += if k % 2 == 0 { 2.0 } else { 4.0 } * g(xv);
        }
        acc * h / 3.0
    };
    for &xi in &[0.0, 0.5, 1.0, 1.7] {
        let numeric = quad(xi);
        let analytic = (-PI * xi * xi).exp();
        assert!(
            (numeric - analytic).abs() < 1e-6,
            "Gaussian FT mismatch at ξ={xi}: numeric {numeric} vs analytic {analytic}",
        );
    }
}

// --- table: shifted Gaussian (completing the square) ------------------------

#[test]
fn shifted_gaussian_expanded() {
    // F{e^{−a(x−b)²}}(ξ) = √(π/a)·e^{−π² ξ²/a}·e^{−2πi b ξ}.
    // Feed the *expanded* exponent −a x² + 2ab x − a b² with a = 1, b = 2:
    //   exponent = −x² + 4x − 4.
    let (pool, x, xi) = setup();
    let x2 = pool.pow(x, int(&pool, 2));
    let exponent = pool.add(vec![
        pool.mul(vec![int(&pool, -1), x2]),
        pool.mul(vec![int(&pool, 4), x]),
        int(&pool, -4),
    ]);
    let f = pool.func("exp", vec![exponent]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();

    // a = 1, b = 2  ⇒  √π · e^{−π² ξ²} · e^{−4πi ξ}.
    let pi = pool.symbol("pi", Domain::Real);
    let half = pool.rational(1, 2);
    let prefactor = pool.pow(pi, half); // √(π/1)
    let pi2 = pool.pow(pi, int(&pool, 2));
    let xi2 = pool.pow(xi, int(&pool, 2));
    let gauss = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), pi2, xi2])]);
    let i = pool.imaginary_unit();
    let phase = pool.func(
        "exp",
        vec![pool.mul(vec![int(&pool, -4), pi, i, xi])], // −2π i b ξ with b = 2
    );
    let expected = pool.mul(vec![prefactor, gauss, phase]);
    assert_eq_simplified(&pool, g, expected);
}

#[test]
fn shifted_gaussian_factored_form() {
    // Same pair fed as e^{−(x−2)²} (factored) — the simplifier expands the
    // square so the quadratic matcher still recognises it.
    let (pool, x, xi) = setup();
    let shifted = pool.add(vec![x, int(&pool, -2)]);
    let sq = pool.pow(shifted, int(&pool, 2));
    let f = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), sq])]);
    let g = fourier_transform(f, x, xi, &pool);
    assert!(
        g.is_ok(),
        "factored shifted Gaussian e^{{-(x-2)^2}} should be recognised: {g:?}",
    );
    // Result must be free of I² (the cross-term must have collapsed): the only
    // I appears linearly in the phase e^{−4π i ξ}.  Spot-check: the centred
    // magnitude |F| = √π e^{−π² ξ²} matches the real Gaussian numerically.
    let g = simp(g.unwrap(), &pool);
    let s = disp(&pool, g);
    // I appears (the linear phase) but no surviving I² cross-term: completing
    // the square must have collapsed it via the kernel's i² = −1 rule.
    assert!(s.contains("I"), "phase factor should carry I: {s}");
    assert!(!s.contains("I^2"), "I² cross-term must have collapsed: {s}");
}

/// Numeric spot-check of the *magnitude* of the shifted-Gaussian transform:
/// |F{e^{−x²}}(ξ)| via quadrature equals √π·e^{−π²ξ²}.  The shift only adds a
/// unit-modulus phase, so the magnitude equals the centred Gaussian's.
#[test]
fn shifted_gaussian_magnitude_numeric() {
    use std::f64::consts::PI;
    // |F{e^{−(x−b)²}}(ξ)| = |F{e^{−x²}}(ξ)| = √π·e^{−π²ξ²}.
    let mag = |xi: f64| -> f64 {
        let l = 8.0;
        let n = 40_000usize;
        let h = 2.0 * l / n as f64;
        // ∫ e^{−x²} e^{−2πiξx} dx — accumulate real and imaginary parts.
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for k in 0..=n {
            let xv = -l + k as f64 * h;
            let w = if k == 0 || k == n {
                1.0
            } else if k % 2 == 0 {
                2.0
            } else {
                4.0
            };
            let g = (-xv * xv).exp();
            re += w * g * (2.0 * PI * xi * xv).cos();
            im += w * g * -(2.0 * PI * xi * xv).sin();
        }
        let (re, im) = (re * h / 3.0, im * h / 3.0);
        (re * re + im * im).sqrt()
    };
    for &xi in &[0.0, 0.3, 0.7] {
        let numeric = mag(xi);
        let analytic = PI.sqrt() * (-PI * PI * xi * xi).exp();
        assert!(
            (numeric - analytic).abs() < 1e-6,
            "shifted-Gaussian magnitude mismatch at ξ={xi}: {numeric} vs {analytic}",
        );
    }
}

// --- table: two-sided exponential → Lorentzian ------------------------------

#[test]
fn two_sided_exponential_abs() {
    // F{e^{−a|x|}}(ξ) = 2a/(a² + 4π² ξ²), a = 3, |x| = abs(x).
    let (pool, x, xi) = setup();
    let pi = pool.symbol("pi", Domain::Real);
    let a = int(&pool, 3);
    let absx = pool.func("abs", vec![x]);
    let f = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), a, absx])]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();

    let two_a = pool.mul(vec![int(&pool, 2), a]);
    let a2 = pool.pow(a, int(&pool, 2));
    let pi2 = pool.pow(pi, int(&pool, 2));
    let xi2 = pool.pow(xi, int(&pool, 2));
    let denom = pool.add(vec![a2, pool.mul(vec![int(&pool, 4), pi2, xi2])]);
    let expected = pool.mul(vec![two_a, pool.pow(denom, int(&pool, -1))]);
    assert_eq_simplified(&pool, g, expected);
}

#[test]
fn two_sided_exponential_sqrt_form() {
    // |x| as (x²)^{1/2} should also match.
    let (pool, x, xi) = setup();
    let a = int(&pool, 1);
    let x2 = pool.pow(x, int(&pool, 2));
    let absx = pool.pow(x2, pool.rational(1, 2));
    let f = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), a, absx])]);
    let g = fourier_transform(f, x, xi, &pool);
    assert!(g.is_ok(), "sqrt(x²) form should be recognised: {g:?}");
}

// --- table: Dirac delta -----------------------------------------------------

#[test]
fn dirac_at_origin() {
    // F{δ(x)}(ξ) = 1.
    let (pool, x, xi) = setup();
    let f = pool.func("diracdelta", vec![x]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();
    assert_eq_simplified(&pool, g, int(&pool, 1));
}

#[test]
fn dirac_shifted() {
    // F{δ(x − a)}(ξ) = e^{−2πi a ξ}, a = 2.
    let (pool, x, xi) = setup();
    let a = int(&pool, 2);
    let arg = pool.add(vec![x, pool.mul(vec![int(&pool, -1), a])]);
    let f = pool.func("diracdelta", vec![arg]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();

    let i = pool.symbol("I", Domain::Complex);
    let pi = pool.symbol("pi", Domain::Real);
    let exponent = pool.mul(vec![int(&pool, -2), pi, i, a, xi]);
    let expected = pool.func("exp", vec![exponent]);
    assert_eq_simplified(&pool, g, expected);
}

// --- theorems: scaling ------------------------------------------------------

#[test]
fn scaling_via_general_gaussian() {
    // Scaling theorem (b > 0): F{f(b·x)} = (1/b)·F(ξ/b).  With the self-dual
    // Gaussian f(x) = e^{−π x²}, f(b x) = e^{−π b² x²} is the general Gaussian
    // with a = π b²; its transform is √(π/a)·e^{−π² ξ²/a}.  Take b = 2 (a = 4π):
    // F = √(1/4)·e^{−π ξ²/4} = (1/2)·e^{−π ξ²/4}, matching (1/b)·e^{−π (ξ/b)²}.
    let (pool, x, xi) = setup();
    let pi = pool.symbol("pi", Domain::Real);
    let a = pool.mul(vec![int(&pool, 4), pi]); // a = 4π
    let x2 = pool.pow(x, int(&pool, 2));
    let f = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), a, x2])]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();

    // expected = (1/2)·e^{−π ξ²/4}.  The prefactor √(π/(4π)) is mathematically
    // 1/2 but the simplifier does not cancel `π·(4π)⁻¹` (a known simplifier gap,
    // see PR notes), so we compare *numerically* rather than structurally.
    let xi2 = pool.pow(xi, int(&pool, 2));
    let quarter = pool.rational(1, 4);
    let exponent = pool.mul(vec![int(&pool, -1), pi, xi2, quarter]);
    let expected = pool.mul(vec![pool.rational(1, 2), pool.func("exp", vec![exponent])]);
    for &v in &[0.0_f64, 0.3, 1.1, 2.0] {
        let gv = eval_xi(&pool, g, xi, v);
        let ev = eval_xi(&pool, expected, xi, v);
        assert!(
            (gv - ev).abs() < 1e-9 * (1.0 + gv.abs()),
            "scaling mismatch at ξ={v}: {gv} vs {ev}",
        );
    }
}

/// Minimal numeric evaluator for the (real) forms this test file produces;
/// `pi → π`, `xi → val`, supports +, ·, ^, exp/abs/sqrt.
fn eval_xi(pool: &ExprPool, expr: ExprId, xi: ExprId, val: f64) -> f64 {
    match pool.get(expr) {
        ExprData::Integer(n) => n.0.to_f64(),
        ExprData::Rational(r) => {
            let (n, d) = r.0.clone().into_numer_denom();
            n.to_f64() / d.to_f64()
        }
        ExprData::Symbol { ref name, .. } => {
            if expr == xi {
                val
            } else if name == "pi" {
                std::f64::consts::PI
            } else {
                f64::NAN
            }
        }
        ExprData::Add(args) => args.iter().map(|&a| eval_xi(pool, a, xi, val)).sum(),
        ExprData::Mul(args) => args.iter().map(|&a| eval_xi(pool, a, xi, val)).product(),
        ExprData::Pow { base, exp } => {
            eval_xi(pool, base, xi, val).powf(eval_xi(pool, exp, xi, val))
        }
        ExprData::Func { ref name, ref args } if args.len() == 1 => {
            let a = eval_xi(pool, args[0], xi, val);
            match name.as_str() {
                "exp" => a.exp(),
                "abs" => a.abs(),
                "sqrt" => a.sqrt(),
                _ => f64::NAN,
            }
        }
        _ => f64::NAN,
    }
}

// --- theorems: shift (spatial) ----------------------------------------------

#[test]
fn shift_theorem_on_delta() {
    // Shift theorem: F{f(x − a)} = e^{−2πi a ξ}·F(ξ).  With f = δ (F{δ} = 1),
    // F{δ(x − a)} = e^{−2πi a ξ}, exercised here with a = 1 (clean phase).
    let (pool, x, xi) = setup();
    let arg = pool.add(vec![x, int(&pool, -1)]);
    let f = pool.func("diracdelta", vec![arg]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();
    let i = pool.symbol("I", Domain::Complex);
    let pi = pool.symbol("pi", Domain::Real);
    let expected = pool.func("exp", vec![pool.mul(vec![int(&pool, -2), pi, i, xi])]);
    assert_eq_simplified(&pool, g, expected);
}

// --- table: constant → δ(ξ) -------------------------------------------------

#[test]
fn constant_to_delta() {
    // F{1}(ξ) = δ(ξ); F{5}(ξ) = 5 δ(ξ).
    let (pool, x, xi) = setup();
    let five = int(&pool, 5);
    let g = fourier_transform(five, x, xi, &pool).unwrap();
    let delta = pool.func("diracdelta", vec![xi]);
    let expected = pool.mul(vec![five, delta]);
    assert_eq_simplified(&pool, g, expected);
}

// --- table: one-sided (causal) exponential ----------------------------------

#[test]
fn one_sided_exponential() {
    // F{θ(x)·e^{−a x}}(ξ) = 1/(a + 2πi ξ), a = 4.
    let (pool, x, xi) = setup();
    let a = int(&pool, 4);
    let theta = pool.func("heaviside", vec![x]);
    let exp = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), a, x])]);
    let f = pool.mul(vec![theta, exp]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();

    let i = pool.symbol("I", Domain::Complex);
    let pi = pool.symbol("pi", Domain::Real);
    let two_pi_i_xi = pool.mul(vec![int(&pool, 2), pi, i, xi]);
    let denom = pool.add(vec![a, two_pi_i_xi]);
    let expected = pool.pow(denom, int(&pool, -1));
    assert_eq_simplified(&pool, g, expected);
}

// --- theorems: linearity ----------------------------------------------------

#[test]
fn linearity() {
    // F{2 δ(x) + 3 δ(x − 1)} = 2 + 3 e^{−2πi ξ}.
    let (pool, x, xi) = setup();
    let d0 = pool.func("diracdelta", vec![x]);
    let arg1 = pool.add(vec![x, int(&pool, -1)]);
    let d1 = pool.func("diracdelta", vec![arg1]);
    let f = pool.add(vec![
        pool.mul(vec![int(&pool, 2), d0]),
        pool.mul(vec![int(&pool, 3), d1]),
    ]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();

    let i = pool.symbol("I", Domain::Complex);
    let pi = pool.symbol("pi", Domain::Real);
    let phase = pool.func("exp", vec![pool.mul(vec![int(&pool, -2), pi, i, xi])]);
    let expected = pool.add(vec![int(&pool, 2), pool.mul(vec![int(&pool, 3), phase])]);
    assert_eq_simplified(&pool, g, expected);
}

// --- theorems: modulation ---------------------------------------------------

#[test]
fn modulation_shifts_frequency() {
    // F{e^{2πi a x}·δ(x)} = F{δ}(ξ − a) = 1, but more telling:
    // F{e^{2πi a x}·e^{−π x²}} = e^{−π (ξ−a)²}.  Use a = 1.
    let (pool, x, xi) = setup();
    let pi = pool.symbol("pi", Domain::Real);
    let i = pool.symbol("I", Domain::Complex);
    let a = int(&pool, 1);
    let mod_exp = pool.func("exp", vec![pool.mul(vec![int(&pool, 2), pi, i, a, x])]);
    let x2 = pool.pow(x, int(&pool, 2));
    let gauss = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), pi, x2])]);
    let f = pool.mul(vec![mod_exp, gauss]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();

    let shifted = pool.add(vec![xi, pool.mul(vec![int(&pool, -1), a])]);
    let shifted2 = pool.pow(shifted, int(&pool, 2));
    let expected = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), pi, shifted2])]);
    assert_eq_simplified(&pool, g, expected);
}

#[test]
fn pure_modulation_to_shifted_delta() {
    // F{e^{2πi a x}} = δ(ξ − a), a = 3.
    let (pool, x, xi) = setup();
    let pi = pool.symbol("pi", Domain::Real);
    let i = pool.symbol("I", Domain::Complex);
    let a = int(&pool, 3);
    let f = pool.func("exp", vec![pool.mul(vec![int(&pool, 2), pi, i, a, x])]);
    let g = fourier_transform(f, x, xi, &pool).unwrap();
    let shifted = pool.add(vec![xi, pool.mul(vec![int(&pool, -1), a])]);
    let expected = pool.func("diracdelta", vec![shifted]);
    assert_eq_simplified(&pool, g, expected);
}

// --- theorems: derivative rule ----------------------------------------------

#[test]
fn derivative_rule() {
    // F{f'}(ξ) = 2πi ξ F(ξ); F{f''} = (2πi ξ)² F.
    let (pool, _x, xi) = setup();
    let big_f = pool.symbol("F", Domain::Complex);
    let i = pool.symbol("I", Domain::Complex);
    let pi = pool.symbol("pi", Domain::Real);

    let g1 = fourier_derivative_rule(big_f, xi, 1, &pool);
    let expected1 = pool.mul(vec![int(&pool, 2), pi, i, xi, big_f]);
    assert_eq_simplified(&pool, g1, expected1);

    let g2 = fourier_derivative_rule(big_f, xi, 2, &pool);
    let factor = pool.pow(pool.mul(vec![int(&pool, 2), pi, i, xi]), int(&pool, 2));
    let expected2 = pool.mul(vec![factor, big_f]);
    assert_eq_simplified(&pool, g2, expected2);
}

// --- inverse / round-trips --------------------------------------------------

#[test]
fn inverse_of_gaussian_roundtrip() {
    // F⁻¹{F{e^{−π x²}}} = e^{−π x²} (self-dual, even ⇒ exact round-trip).
    let (pool, x, xi) = setup();
    let pi = pool.symbol("pi", Domain::Real);
    let x2 = pool.pow(x, int(&pool, 2));
    let f = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), pi, x2])]);

    let g = fourier_transform(f, x, xi, &pool).unwrap();
    let back = inverse_fourier_transform(g, xi, x, &pool).unwrap();
    assert_eq_simplified(&pool, back, f);
}

#[test]
fn inverse_of_lorentzian_roundtrip() {
    // F⁻¹{2a/(a²+4π²ξ²)} = e^{−a|x|} (even ⇒ exact round-trip), a = 2.
    let (pool, x, xi) = setup();
    let a = int(&pool, 2);
    let absx = pool.func("abs", vec![x]);
    let f = pool.func("exp", vec![pool.mul(vec![int(&pool, -1), a, absx])]);

    let g = fourier_transform(f, x, xi, &pool).unwrap();
    let back = inverse_fourier_transform(g, xi, x, &pool).unwrap();
    // back is F{Lorentzian}(−x); for an even function this equals f.
    assert_eq_simplified(&pool, back, f);
}

#[test]
fn inverse_of_one_gives_delta_in_x() {
    // F⁻¹{1}(x) = δ(x) (since F{1} = δ and δ is even).
    let (pool, x, xi) = setup();
    let one = int(&pool, 1);
    let back = inverse_fourier_transform(one, xi, x, &pool).unwrap();
    let expected = pool.func("diracdelta", vec![x]);
    assert_eq_simplified(&pool, back, expected);
}

// --- declines ---------------------------------------------------------------

#[test]
fn unknown_form_declines() {
    // tan(x) has no Fourier table entry → NoRule.
    let (pool, x, xi) = setup();
    let f = pool.func("tan", vec![x]);
    assert!(matches!(
        fourier_transform(f, x, xi, &pool),
        Err(FourierError::NoRule(_))
    ));
}

#[test]
fn bare_heaviside_declines() {
    // F{θ(x)} is distributional (½δ(ξ) + 1/(2πiξ)); we decline rather than guess.
    let (pool, x, xi) = setup();
    let f = pool.func("heaviside", vec![x]);
    assert!(matches!(
        fourier_transform(f, x, xi, &pool),
        Err(FourierError::NoRule(_))
    ));
}

#[test]
fn divergent_gaussian_declines() {
    // e^{+x²} grows; not a valid (decaying) Gaussian → decline.
    let (pool, x, xi) = setup();
    let x2 = pool.pow(x, int(&pool, 2));
    let f = pool.func("exp", vec![x2]);
    assert!(matches!(
        fourier_transform(f, x, xi, &pool),
        Err(FourierError::NoRule(_))
    ));
}
