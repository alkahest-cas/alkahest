//! Numeric evaluation of [`ExprData::RootSum`].
//!
//! # Why this exists
//!
//! A `RootSum(m(r), r . body(x, r))` is the Rothstein–Trager / Lazard–Rioboo–
//! Trager answer for the logarithmic part of `∫ A(x)/D(x) dx` whose residues are
//! algebraic numbers of degree ≥ 2.  It is a compact and (as far as the decision
//! procedure goes) correct answer — but until this module existed **no
//! verification tier could read it**:
//!
//! * `simplify` treats a `RootSum` as an opaque atom, so the symbolic arm of
//!   [`crate::integrate::verify_antiderivative_status`] never reduces a residual
//!   containing one to zero;
//! * [`crate::jit::eval_interp`] and [`crate::integrate::gate::eval_at`] had no
//!   `RootSum` rule, so the numeric arm returned "unevaluable".
//!
//! The consequence was not a decline. The rational-function route in
//! `integrate::engine` returned its answer *without a gate at all*, so a
//! `RootSum` reached the caller as an unchecked assertion. Adding the gate
//! without adding this module would have deleted the whole degree-≥3 residue
//! capability instead; adding a "this route is trusted" verdict would have
//! manufactured a ninth false certificate. So the answers are made **checkable**
//! instead.
//!
//! # What it computes
//!
//! `Σ_{c : m(c) = 0} body(x, c)`, literally:
//!
//! 1. `m` is read as a univariate polynomial over ℚ in the bound variable and
//!    cleared to ℤ (same roots), then to `f64` coefficients;
//! 2. its **complex** roots are found by Durand–Kerner, and each is accepted
//!    only after a relative backward-error check — a half-converged root set is
//!    refused rather than trusted;
//! 3. `body` is evaluated at each root in IEEE-754 complex arithmetic
//!    ([`crate::eval::eval_complex_f64`]), with the free variables taken from
//!    the caller's real environment;
//! 4. the terms are summed and the result is returned **only if it is real** to
//!    within a relative tolerance.
//!
//! Step 4 is the substantive one.  The sum over a full conjugate-closed root set
//! of a real polynomial is real, and the principal branch of `log` is
//! conjugate-symmetric on `ℂ∖(−∞,0]`, so conjugate root pairs contribute
//! conjugate terms whose imaginary parts cancel.  A residual imaginary part that
//! is *not* rounding noise means the branch choices did not pair up (or the root
//! set is not the one `m` describes), and that is reported as "cannot evaluate",
//! never rounded away.
//!
//! # Soundness
//!
//! Nothing here can make a wrong antiderivative look right. Every failure mode
//! — a non-convergent root set, a root set that fails the backward-error check,
//! a body the complex evaluator cannot read, a non-real sum — returns `None`,
//! which the callers treat as "this sample point is not evidence". A *wrong*
//! numeric value would have to coincide with the true derivative at every
//! sample point of the gate's grid to be admitted, which is the same standing
//! assumption the rest of the numeric tier already makes.
//!
//! The backward-error check is deliberately per-root and not a check on the
//! *multiset*: two Durand–Kerner iterates converging to the same root, with a
//! third missed, would pass it. That is a real possibility and it is left
//! unguarded on purpose — the consequence is a sum that is simply wrong, which
//! the gate then rejects, so the cost is a decline rather than a false accept.
//! A multiset check (reconstructing the coefficients from the roots) would
//! convert those declines into a different kind of decline while adding a way
//! to refuse a root set that was fine.
//!
//! What this module is **not** is a proof. It is an `f64` screen, and the
//! module docs of [`crate::integrate::gate`] describe exactly what such a screen
//! can and cannot see (catastrophic cancellation, chiefly). An exact
//! alternative exists — `Σ_{m(c)=0} body(x, c)` is a number-field trace and can
//! be computed by resultants — but it is not implemented here.
//!
//! # The precision ceiling, measured
//!
//! `Σ_{m(c)=0} body(x, c)` is **ill-conditioned in the roots**, and that, not
//! the root finder, is what bounds this module. Measured on
//! `∫dx/(xⁿ − x − 1)`, whose residues are a tight cluster of `n` algebraic
//! numbers of modulus `≈ n^{−1/(n−1)}`, against a 60-digit reference:
//!
//! | `n` | answer's own error (60-digit) | error from perturbing each root by 1 ulp |
//! |----:|------------------------------:|-----------------------------------------:|
//! |   9 |                       `3e−59` |                                   `3e−13` |
//! |  13 |                       `1e−55` |                                   `5e−10` |
//! |  15 |                       `8e−54` |                                   `2e−08` |
//! |  18 |                       `5e−50` |                                   `2e−04` |
//! |  21 |                       `9e−48` |                                   `1e−02` |
//!
//! Two things follow, and they should be read together.
//!
//! * **The answers are right.** Column two is the Rothstein–Trager answer
//!   evaluated at 60 digits; it is correct to fifty-odd places at every degree
//!   tried. Nothing here has ever found a wrong one.
//! * **`f64` stops being able to say so at about degree 15.** Column three is
//!   the error a *perfect* `f64` root set would still produce. Against the
//!   gate's `1e−7` tolerance it passes through zero somewhere between degree 14
//!   and 15, so from there up the honest verdict is "this implementation cannot
//!   check its own answer" — and the caller declines rather than shipping it.
//!
//! Raising that ceiling means high-precision *roots*, not merely a
//! high-precision body: about six extra digits buys degree 21, so a 128-bit
//! complex evaluator would cover everything the rational route can produce
//! before its own number-field GCD becomes the binding cost. That is a
//! self-contained follow-up; `crate::ball` has the MPFR machinery but no
//! complex layer.

use std::collections::HashMap;

use super::{eval_complex_f64, ComplexF64};
use crate::kernel::{ExprId, ExprPool};

/// Largest minimal-polynomial degree this evaluator will attempt.
///
/// Durand–Kerner is `O(deg²)` per iteration with a fixed iteration cap, so the
/// cost is bounded either way; the ceiling is here so that a pathological
/// `RootSum` cannot turn a per-sample gate check into a visible pause. Degree 24
/// is far above anything the rational route emits in practice (the residue
/// minimal polynomial divides `res_t(N − t·P′, P)`, whose degree is that of the
/// squarefree denominator).
const MAX_DEGREE: usize = 24;

/// Relative backward error a candidate root must achieve to be trusted.
const ROOT_RESIDUAL_TOL: f64 = 1e-6;

/// Durand–Kerner iteration cap. Cost is `O(deg²)` per sweep; at the degree
/// ceiling above that is a few hundred microseconds for the whole cap, and
/// convergence on a well-scaled polynomial takes tens of sweeps, not hundreds.
const MAX_ITERATIONS: usize = 500;

/// Relative step size below which the iteration is considered settled. It is
/// only an early exit — what decides whether the roots are usable is the
/// backward-error check, which runs either way.
const STEP_TOL: f64 = 1e-15;

/// Relative size the imaginary part of the sum may reach and still count as
/// rounding noise around a real value.
const REAL_TOL: f64 = 1e-6;

/// Evaluate `Σ_{c : poly(c) = 0} body[rvar := c]` in `f64`.
///
/// `env` binds the *free* variables of `body` (the integration variable, in the
/// integrator's case). `rvar` is the bound root placeholder and is always
/// rebound here, shadowing any entry `env` may have for it.
///
/// Returns `None` — "no numeric value", never a guess — when the minimal
/// polynomial cannot be read, its roots cannot be trusted, the body cannot be
/// evaluated in complex arithmetic at some root, or the sum is not real.
pub(crate) fn eval_root_sum_f64(
    poly: ExprId,
    rvar: ExprId,
    body: ExprId,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
) -> Option<f64> {
    let roots = minimal_polynomial_roots(poly, rvar, pool)?;

    let mut bindings: HashMap<ExprId, ComplexF64> = env
        .iter()
        .map(|(&k, &v)| (k, ComplexF64::new(v, 0.0)))
        .collect();

    let (mut sum_re, mut sum_im, mut scale) = (0.0f64, 0.0f64, 0.0f64);
    for root in roots {
        bindings.insert(rvar, root);
        let term = eval_complex_f64(body, pool, &bindings).ok()?;
        sum_re += term.re;
        sum_im += term.im;
        scale += term.re.abs() + term.im.abs();
    }

    if !sum_re.is_finite() || !sum_im.is_finite() {
        return None;
    }
    // The sum over a conjugate-closed root set of a real polynomial is real.
    // An imaginary part above rounding noise means this evaluation is not
    // describing that sum, so decline rather than silently take the real part.
    if sum_im.abs() > REAL_TOL * scale.max(1.0) {
        return None;
    }
    Some(sum_re)
}

/// The complex roots of `poly` (a univariate polynomial in `rvar` over ℚ),
/// or `None` when it is not such a polynomial, is too large, or the iteration
/// does not produce a root set that passes a relative backward-error check.
fn minimal_polynomial_roots(
    poly: ExprId,
    rvar: ExprId,
    pool: &ExprPool,
) -> Option<Vec<ComplexF64>> {
    // Rational coefficients are cleared to ℤ, which leaves the root set alone.
    let up = crate::poly::UniPoly::from_symbolic_clear_denoms(poly, rvar, pool).ok()?;
    let degree = usize::try_from(up.degree()).ok()?;
    if degree == 0 || degree > MAX_DEGREE {
        return None;
    }
    let coeffs: Vec<f64> = up.coefficients().iter().map(rug::Integer::to_f64).collect();
    if coeffs.iter().any(|c| !c.is_finite()) {
        return None;
    }

    let roots = polynomial_roots(&coeffs)?;
    if roots.len() != degree {
        return None;
    }

    // Whether the iteration settled is not the question; whether these numbers
    // are roots is. `|p(z)|` must be small *relative to* the size of the terms
    // that were added to produce it — the standard backward-error criterion,
    // and the only one that means anything when the coefficients span sixteen
    // orders of magnitude, which these routinely do.
    for &z in &roots {
        let (value, magnitude) = horner_with_magnitude(&coeffs, z);
        if value.re.hypot(value.im) > ROOT_RESIDUAL_TOL * magnitude.max(1.0) {
            return None;
        }
    }
    Some(roots)
}

/// All complex roots of the real polynomial with ascending coefficients
/// `coeffs`, by Durand–Kerner **after rescaling the variable**.
///
/// The rescaling is the load-bearing part and the reason
/// [`crate::calculus::asymptotic_common::complex_roots`] is not reused here.
/// That routine stops on an *absolute* step size of `1e-14`, which silently
/// assumes roots of order 1.  A Rothstein–Trager residue minimal polynomial has
/// no such roots: the residues of `∫dx/(x¹⁴−x−1)` all lie between `0.038` and
/// `0.32`, in seven conjugate pairs, and the coefficients span `10¹⁶`.  On that
/// input the absolute criterion is asking for twelve significant digits from a
/// tightly clustered set and never trips, so the iteration burns its cap and
/// reports failure on a polynomial it had in fact almost solved.
///
/// Substituting `z = s·w` with `s` the geometric mean root modulus
/// `|c₀/cₙ|^{1/n}` puts the whole root set on a circle of radius about 1, where
/// a relative step criterion and an `f64` Horner evaluation both behave.
fn polynomial_roots(coeffs: &[f64]) -> Option<Vec<ComplexF64>> {
    let mut c = coeffs.to_vec();
    while c.len() > 1 && c.last() == Some(&0.0) {
        c.pop();
    }
    if c.len() <= 1 {
        return Some(Vec::new());
    }
    // A zero constant term is a root at the origin; peel those off first so the
    // geometric-mean scale below is not `0`.
    let at_origin = c.iter().take_while(|v| **v == 0.0).count();
    let tail = &c[at_origin..];
    let degree = tail.len() - 1;
    let leading = *tail.last()?;
    if leading == 0.0 || !leading.is_finite() {
        return None;
    }
    let mut roots = vec![ComplexF64::new(0.0, 0.0); at_origin];
    if degree == 0 {
        return Some(roots);
    }

    let monic: Vec<f64> = tail.iter().map(|v| v / leading).collect();
    if monic.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let scale = {
        let s = monic[0].abs().powf(1.0 / degree as f64);
        if s.is_finite() && s > 0.0 {
            s
        } else {
            1.0
        }
    };
    // `q(w) = p(s·w)/sⁿ`, i.e. `qₖ = pₖ·s^{k−n}`. With `s = |p₀|^{1/n}` this
    // makes `|q₀| = 1` and leaves `q` monic, so the roots of `q` have geometric
    // mean modulus 1 whatever scale the roots of `p` live at.
    let scaled: Vec<f64> = monic
        .iter()
        .enumerate()
        .map(|(k, &v)| v / scale.powi((degree - k) as i32))
        .collect();
    if scaled.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Classic Durand–Kerner seed: powers of `0.4 + 0.9i`, which is off every
    // symmetry axis a real polynomial has and so cannot start two iterates on
    // top of each other.
    let seed = ComplexF64::new(0.4, 0.9);
    let mut z = Vec::with_capacity(degree);
    let mut point = ComplexF64::new(1.0, 0.0);
    for _ in 0..degree {
        z.push(point);
        point = cmul(point, seed);
    }

    for _ in 0..MAX_ITERATIONS {
        let mut worst = 0.0f64;
        for i in 0..degree {
            let mut denom = ComplexF64::new(1.0, 0.0);
            for j in 0..degree {
                if i != j {
                    denom = cmul(denom, ComplexF64::new(z[i].re - z[j].re, z[i].im - z[j].im));
                }
            }
            let denom_abs = denom.re.hypot(denom.im);
            if denom_abs == 0.0 || !denom_abs.is_finite() {
                return None;
            }
            let (value, _) = horner_with_magnitude(&scaled, z[i]);
            let step = cdiv(value, denom);
            z[i] = ComplexF64::new(z[i].re - step.re, z[i].im - step.im);
            if !z[i].re.is_finite() || !z[i].im.is_finite() {
                return None;
            }
            worst = worst.max(step.re.hypot(step.im) / (1.0 + z[i].re.hypot(z[i].im)));
        }
        if worst < STEP_TOL {
            break;
        }
    }

    roots.extend(
        z.into_iter()
            .map(|w| ComplexF64::new(w.re * scale, w.im * scale)),
    );
    Some(roots)
}

fn cmul(a: ComplexF64, b: ComplexF64) -> ComplexF64 {
    ComplexF64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

fn cdiv(a: ComplexF64, b: ComplexF64) -> ComplexF64 {
    let d = b.re * b.re + b.im * b.im;
    ComplexF64::new(
        (a.re * b.re + a.im * b.im) / d,
        (a.im * b.re - a.re * b.im) / d,
    )
}

/// `p(z)` by Horner, together with `Σ |cₖ| · |z|ᵏ` — the size of the largest
/// intermediate the evaluation passes through, which is the scale a residual
/// has to be compared against.
fn horner_with_magnitude(coeffs: &[f64], z: ComplexF64) -> (ComplexF64, f64) {
    let az = z.re.hypot(z.im);
    let mut acc = ComplexF64::new(0.0, 0.0);
    let mut magnitude = 0.0f64;
    for &c in coeffs.iter().rev() {
        // acc = acc·z + c
        acc = ComplexF64::new(
            acc.re * z.re - acc.im * z.im + c,
            acc.re * z.im + acc.im * z.re,
        );
        magnitude = magnitude * az + c.abs();
    }
    (acc, magnitude)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    /// `Σ_{c² = 2} c = 0` and `Σ_{c² = 2} c² = 4`: the two textbook power sums,
    /// on a polynomial whose roots are irrational.
    #[test]
    fn power_sums_of_a_quadratic() {
        let pool = ExprPool::new();
        let r = pool.symbol("r", Domain::Real);
        let m = pool.add(vec![pool.pow(r, pool.integer(2_i32)), pool.integer(-2_i32)]);
        let env = HashMap::new();

        let s1 = eval_root_sum_f64(m, r, r, &env, &pool).unwrap();
        assert!(s1.abs() < 1e-10, "Σc = {s1}");
        let s2 = eval_root_sum_f64(m, r, pool.pow(r, pool.integer(2_i32)), &env, &pool).unwrap();
        assert!((s2 - 4.0).abs() < 1e-10, "Σc² = {s2}");
    }

    /// The conjugate-pair case: `x² + 1` has no real root, and the sum of `c²`
    /// over `±i` is `−2` — real, and reached only because both terms are
    /// evaluated in complex arithmetic.
    #[test]
    fn a_complex_root_pair_still_sums_to_a_real() {
        let pool = ExprPool::new();
        let r = pool.symbol("r", Domain::Real);
        let m = pool.add(vec![pool.pow(r, pool.integer(2_i32)), pool.integer(1_i32)]);
        let got = eval_root_sum_f64(
            m,
            r,
            pool.pow(r, pool.integer(2_i32)),
            &HashMap::new(),
            &pool,
        )
        .unwrap();
        assert!((got + 2.0).abs() < 1e-10, "Σc² over ±i = {got}");
    }

    /// A body whose sum is genuinely non-real must be refused, not rounded.
    /// `Σ_{c² + 1 = 0}` of the *single-branch* body `c` is `0`, so pick one that
    /// is not conjugate-symmetric: `1/(c − 2i)` evaluated over `±i` gives
    /// `1/(−i) + 1/(−3i) = (4/3)i`.
    #[test]
    fn a_non_real_sum_is_declined() {
        let pool = ExprPool::new();
        let r = pool.symbol("r", Domain::Real);
        let m = pool.add(vec![pool.pow(r, pool.integer(2_i32)), pool.integer(1_i32)]);
        let two_i = pool.mul(vec![pool.integer(-2_i32), pool.imaginary_unit()]);
        let body = pool.pow(pool.add(vec![r, two_i]), pool.integer(-1_i32));
        assert_eq!(
            eval_root_sum_f64(m, r, body, &HashMap::new(), &pool),
            None,
            "a sum with a real imaginary part is not a real number"
        );
    }

    /// The minimal polynomial has to be a polynomial in the bound variable.
    #[test]
    fn a_non_polynomial_minimal_polynomial_is_declined() {
        let pool = ExprPool::new();
        let r = pool.symbol("r", Domain::Real);
        let m = pool.func("sin", vec![r]);
        assert_eq!(eval_root_sum_f64(m, r, r, &HashMap::new(), &pool), None);
    }

    /// The shape the rational integrator actually emits, end to end through
    /// the interpreter the antiderivative gate uses.
    ///
    /// `∫dx/(x³−2) = Σ_{a³=2} (a/6)·log(x − a)`, which Rothstein–Trager writes
    /// with the residue itself as the bound variable: `RootSum(r³ − 1/108,
    /// r . r·log(x − 6r))`.  Its derivative is `Σ r/(x − 6r)`, and the claim
    /// checked here is that [`crate::jit::eval_interp`] — which returned `None`
    /// for any `RootSum` before this module existed — now agrees with
    /// `1/(x³−2)` to `f64` accuracy.  All three roots are needed: two are a
    /// complex conjugate pair, and dropping them leaves a value that is not
    /// even close.
    #[test]
    fn the_rothstein_trager_shape_evaluates_through_eval_interp() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let r = pool.symbol("$root$", Domain::Real);

        // r³ − 1/108
        let m = pool.add(vec![
            pool.pow(r, pool.integer(3_i32)),
            pool.rational(-1, 108),
        ]);
        // body' = r · (x − 6r)⁻¹
        let arg = pool.add(vec![x, pool.mul(vec![pool.integer(-6_i32), r])]);
        let dbody = pool.mul(vec![r, pool.pow(arg, pool.integer(-1_i32))]);
        let sum = pool.root_sum(m, r, dbody);

        for &xv in &[0.3719_f64, 1.4231, -2.8123] {
            let env: HashMap<ExprId, f64> = std::iter::once((x, xv)).collect();
            let got = crate::jit::eval_interp(sum, &env, &pool)
                .unwrap_or_else(|| panic!("no value at x = {xv}"));
            let want = 1.0 / (xv * xv * xv - 2.0);
            assert!(
                (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
                "at x = {xv}: Σ r/(x−6r) = {got}, 1/(x³−2) = {want}"
            );
        }
    }

    /// Degree above the ceiling declines instead of spending the time.
    #[test]
    fn an_oversized_minimal_polynomial_is_declined() {
        let pool = ExprPool::new();
        let r = pool.symbol("r", Domain::Real);
        let m = pool.add(vec![
            pool.pow(r, pool.integer((MAX_DEGREE + 1) as i32)),
            pool.integer(-1_i32),
        ]);
        assert_eq!(eval_root_sum_f64(m, r, r, &HashMap::new(), &pool), None);
    }
}
