//! The Almkvist–Zeilberger search: continuous creative telescoping.
//!
//! Given `F(n,x)` hyperexponential in `x` and hypergeometric in `n`, find an
//! order `J`, polynomial coefficients `a_0(n) … a_J(n)` (not all zero) and a
//! rational certificate `R ∈ Q(n)(x)` with
//!
//! ```text
//! Σ_{i=0}^{J} a_i(n)·F(n+i, x) = D_x( R(n,x)·F(n,x) ).
//! ```
//!
//! # Method: undetermined coefficients, `J` ascending
//!
//! Exactly the shape `telescoping2d::search` uses for the
//! discrete multi-sum case, and for the same reason: there is no differential
//! analogue of Gosper's normal form that handles a *parametric* right-hand side,
//! so the module does not pretend to one. Divide through by `F`, and the
//! identity becomes the parametric Risch differential equation
//!
//! ```text
//! R′ + θ·R = Σ_i a_i·r_i,     θ = ∂_x F/F,  r_i = F(n+i,x)/F(n,x),
//! ```
//! which [`mod@super::rde`] solves by clearing a fixed denominator and equating
//! coefficients of `x` — one linear system over `Q(n)` in the unknowns
//! `(a_0 … a_{J−1}, coefficients of P)`, with `a_J` normalised to `1`.
//!
//! The grid is walked `J`-major, ascending from `J = 0`, and every `(κ, d)` in
//! bounds is tried before `J+1` is considered. So a returned order **is** the
//! least one reachable within `max_den_power` and `max_degree` — which is the
//! claim [`mod@super::super::zeilberger`]'s cost-ordered plan deliberately gives
//! up in exchange for speed. Here the systems are small enough (a certificate
//! numerator is `d+1` unknowns, not `(d+1)^(m+1)`) that the order-major sweep is
//! affordable, and minimality is worth more than the last factor of two.
//!
//! Normalising `a_J = 1` excludes solutions with `a_J = 0`, which is exactly
//! right under an ascending search: such a solution *is* a relation of lower
//! order, and every lower order has already been exhausted.
//!
//! # What is proved, and what is not
//!
//! The verified content is the identity above — a statement about the
//! **integrand**. Integrating it over `[a,b]` gives
//! `Σ_i a_i(n)·f(n+i) = [R·F]_a^b`, so the homogeneous recurrence for
//! `f(n) = ∫_a^b F dx` needs that boundary term to vanish, which is
//! [`mod@super::boundary`]'s question and not this one's. `∫_0^1 x^n dx` is the
//! one-line reminder: the certificate is `R = x/(n+1)` at order `0`, and
//! `[R·F]_0^1 = 1/(n+1) ≠ 0`, so "the certificate exists" and "the integral
//! satisfies the homogeneous relation" are genuinely different statements.

use super::hyperexp::HyperExpTerm;
use super::rde::{rescale, solve_probe, verify, RdeSetup};
use super::{AzOpts, DiffTelescopingError};
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::holonomic::hyperterm::{ratk_to_expr, ratuni_to_expr};
use crate::holonomic::qfield::{clear_denominators, rn_is_zero, rn_poly, RatK};
use crate::kernel::{ExprId, ExprPool};
use crate::matrix::normal_form::RatUniPoly;

/// A verified Almkvist–Zeilberger certificate:
/// `Σ_i coeffs[i](n)·F(n+i,x) = D_x(R·F)`.
///
/// The verified content is the **telescoping identity in `x`**. Turning it into
/// a recurrence for `f(n) = ∫_a^b F(n,x) dx` needs `[R·F]_a^b` to vanish; see
/// the module docs and [`mod@super::boundary`].
#[derive(Debug, Clone)]
pub struct AzResult {
    /// Recurrence order `J`; `coeffs.len() == order + 1`.
    ///
    /// It is the least order for which a certificate exists within the
    /// `max_degree` / `max_den_power` bounds of the call — the search is
    /// order-major ascending, so every lower order was exhausted first.
    pub order: usize,
    /// `a_0(n), …, a_J(n)` as expressions in `n`; integer-content-primitive,
    /// `a_J` not identically zero.
    pub coeffs: Vec<ExprId>,
    /// `R(n,x)` as an expression, with `G(n,x) = R(n,x)·F(n,x)` the quantity
    /// whose values at the integration limits decide the boundary question.
    pub certificate: ExprId,
    /// How many `(order, κ, degree)` probes the search made, the successful one
    /// included.
    pub probes: usize,
}

/// `G(n,x) = R(n,x)·F(n,x)`, the antiderivative whose boundary values decide
/// whether the certificate's recurrence holds for the *integral*.
///
/// `term` must be the same `F(n,x)` that was passed to
/// [`almkvist_zeilberger`]. The recurrence for `f(n) = ∫_a^b F(n,x) dx` is
/// `Σ_i a_i(n)·f(n+i) = G(n,b) − G(n,a)`, so this is exactly what a caller
/// needs in order to discharge (or refute) the boundary hypothesis for their
/// own limits by hand — [`mod@super::boundary`] does it mechanically.
pub fn integrand_antiderivative(result: &AzResult, term: ExprId, pool: &ExprPool) -> ExprId {
    crate::simplify::simplify(pool.mul(vec![result.certificate, term]), pool).value
}

/// Almkvist–Zeilberger: find a verified differential creative-telescoping
/// relation for a hyperexponential integrand `F(n, x)`.
///
/// Refuses with [`DiffTelescopingError`] rather than guessing when `term` is
/// outside the class (see [`mod@super::hyperexp`]) or when the bounded search
/// finds no certificate that passes exact verification.
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::holonomic::azeil::{almkvist_zeilberger, AzOpts};
///
/// let pool = ExprPool::new();
/// let n = pool.symbol("n", Domain::Real);
/// let x = pool.symbol("x", Domain::Real);
/// // F = x^n · e^(−x); ∫_0^∞ F dx = n!, so f(n+1) = (n+1)·f(n).
/// let f = pool.mul(vec![
///     pool.pow(x, n),
///     pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), x])]),
/// ]);
/// let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
/// assert_eq!(out.value.order, 1);
/// ```
pub fn almkvist_zeilberger(
    term: ExprId,
    n: ExprId,
    x: ExprId,
    pool: &ExprPool,
    opts: &AzOpts,
) -> Result<DerivedExpr<AzResult>, DiffTelescopingError> {
    if n == x {
        return Err(DiffTelescopingError::InvalidInput(
            "the outer index n and the integration variable x must be distinct symbols".into(),
        ));
    }

    let f = HyperExpTerm::parse(term, n, x, pool)?;
    let theta = f.theta()?;

    // The shift ratios are built one order at a time, and a failure to build
    // `r_J` ends the search rather than aborting it: `J = 0` needs no ratio at
    // all, so an integrand that is hyperexponential in `x` but *not*
    // hypergeometric in `n` (`exp(n·x)`) can still have an order-0 certificate,
    // and does.
    let mut ratios: Vec<RatK> = Vec::with_capacity(opts.max_order + 1);
    let mut ratio_failure: Option<DiffTelescopingError> = None;
    let mut probes = 0usize;

    for order in 0..=opts.max_order {
        while ratios.len() <= order {
            match f.ratio_n(ratios.len() as i64) {
                Ok(r) => ratios.push(r),
                Err(e) => {
                    ratio_failure = Some(e);
                    break;
                }
            }
        }
        if ratios.len() <= order {
            break;
        }

        let Some(setup) = RdeSetup::new(&theta, ratios.clone()) else {
            continue;
        };

        for kappa in 0..=opts.max_den_power {
            for d in 0..=opts.max_degree {
                probes += 1;
                let Some(cand) = solve_probe(&setup, kappa, d) else {
                    continue;
                };

                // `a_J = 1`, so the common scale that clears denominators is
                // just the integer-primitive image of `a_J`; applying it to `R`
                // as well is an `x`-free rescale and preserves the identity.
                let a_int: Vec<RatUniPoly> = clear_denominators(&cand.a);
                if a_int.iter().all(|p| p.is_zero()) {
                    continue;
                }
                let scale = rn_poly(a_int[order].clone());
                if rn_is_zero(&scale) {
                    continue;
                }
                let (a_scaled, r_final) = rescale(&cand.a, &cand.r, &scale);

                // The only acceptance test, and it is run on the pair that is
                // about to be returned — not on the pre-scaled candidate.
                if !verify(&setup, &a_scaled, &r_final) {
                    continue;
                }

                let coeffs: Vec<ExprId> =
                    a_int.iter().map(|p| ratuni_to_expr(pool, n, p)).collect();
                let certificate = ratk_to_expr(pool, n, x, &r_final);

                let mut log = DerivationLog::new();
                log.push(RewriteStep::simple(
                    "almkvist_zeilberger_certificate",
                    term,
                    certificate,
                ));
                return Ok(DerivedExpr::with_log(
                    AzResult {
                        order,
                        coeffs,
                        certificate,
                        probes,
                    },
                    log,
                ));
            }
        }
    }

    // An `n`-ratio failure is reported as itself: it closes the branch for every
    // algorithm in this family, where an exhausted grid merely invites larger
    // bounds. Reporting the second when the first is what happened would send a
    // caller off to raise bounds that cannot help.
    if let Some(e) = ratio_failure {
        return Err(e);
    }
    Err(DiffTelescopingError::SearchExhausted(format!(
        "no verified relation of order <= {} with certificate numerator degree <= {} and \
         denominator power <= {} was found for {} ({probes} probes)",
        opts.max_order,
        opts.max_degree,
        opts.max_den_power,
        pool.display(term)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use std::collections::HashMap;

    fn nx(pool: &ExprPool) -> (ExprId, ExprId) {
        (
            pool.symbol("n", Domain::Real),
            pool.symbol("x", Domain::Real),
        )
    }

    /// Check `a_i(n)/a_J(n)` against an expected closed form at several `n`.
    ///
    /// The certificate itself is already proved exactly by the search; what
    /// these assertions pin is that the *recurrence* is the known one, checked
    /// against an independent source (the classical identity), not against the
    /// machinery that produced it.
    fn assert_ratio(r: &AzResult, pool: &ExprPool, n: ExprId, i: usize, want: fn(f64) -> f64) {
        for ni in [2.0_f64, 5.0, 9.5] {
            let env = HashMap::from([(n, ni)]);
            let aj = crate::eval_f64(r.coeffs[r.order], pool, &env).expect("a_J evaluates");
            let ai = crate::eval_f64(r.coeffs[i], pool, &env).expect("a_i evaluates");
            assert!(aj.abs() > 1e-12, "leading coefficient must not vanish");
            let got = ai / aj;
            let expect = want(ni);
            assert!(
                (got - expect).abs() < 1e-9,
                "a_{i}/a_{J} at n={ni}: got {got}, want {expect}",
                J = r.order
            );
        }
    }

    /// `∫_0^∞ xⁿ·e^(−x) dx = n!` — the Gamma recurrence `f(n+1) = (n+1)·f(n)`.
    #[test]
    fn gamma_moments() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            pool.pow(x, n),
            pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), x])]),
        ]);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        let r = &out.value;
        assert_eq!(r.order, 1, "expected the order-1 Gamma recurrence");
        // (n+1)·F(n) − F(n+1) = D_x(x·F): a_0/a_1 = −(n+1).
        assert_ratio(r, &pool, n, 0, |v| -(v + 1.0));
    }

    /// `∫_0^1 xⁿ(1−x)^(1/2) dx = B(n+1, 3/2)`, whose recurrence is
    /// `f(n+1)/f(n) = (n+1)/(n+5/2)`.
    ///
    /// The second exponent is a concrete `1/2` because the coefficient field is
    /// `Q(n)`: a second *symbolic* parameter would need `Q(n,m)`, which this
    /// module does not have and does not pretend to.
    #[test]
    fn beta_integral() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let one_minus = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let f = pool.mul(vec![
            pool.pow(x, n),
            pool.pow(one_minus, pool.rational(1, 2)),
        ]);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        let r = &out.value;
        assert_eq!(r.order, 1);
        // (2n+2)F(n) − (2n+5)F(n+1) = D_x(2x(1−x)F): a_0/a_1 = −(n+1)/(n+5/2).
        assert_ratio(r, &pool, n, 0, |v| -(v + 1.0) / (v + 2.5));
    }

    /// The same integrand with an *integer* second exponent is order **0**, and
    /// that is the correct answer rather than a missed order-1 relation:
    /// `xⁿ(1−x)³` expands to a `Q(n)`-combination of `x^(n+j)`, each of which
    /// integrates to a rational multiple of `xⁿ`, so `F` is literally
    /// `D_x(R·F)` for a rational `R`.
    ///
    /// It is also a clean reminder that an order-0 certificate proves nothing
    /// about the integral by itself: here `[R·F]_0^1 = B(n+1,4) ≠ 0`, so the
    /// "recurrence" `1·f(n) = 0` is exactly as false as the boundary term is
    /// nonzero.
    #[test]
    fn beta_integral_with_an_integer_exponent_collapses_to_order_zero() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let one_minus = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let f = pool.mul(vec![
            pool.pow(x, n),
            pool.pow(one_minus, pool.integer(3_i32)),
        ]);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        assert_eq!(out.value.order, 0);
    }

    /// `∫_0^1 xⁿ(1−x)ⁿ dx` — both exponents moving, still order 1.
    #[test]
    fn central_beta_integral() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let one_minus = pool.add(vec![
            pool.integer(1_i32),
            pool.mul(vec![pool.integer(-1_i32), x]),
        ]);
        let f = pool.mul(vec![pool.pow(x, n), pool.pow(one_minus, n)]);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        let r = &out.value;
        assert_eq!(r.order, 1);
        // f(n) = n!²/(2n+1)!, so f(n+1)/f(n) = (n+1)/(2(2n+3)):
        // a_0/a_1 = −(n+1)/(2(2n+3)).
        assert_ratio(r, &pool, n, 0, |v| -(v + 1.0) / (2.0 * (2.0 * v + 3.0)));
    }

    /// `∫_{−∞}^{∞} x^(2n)·e^(−x²) dx` — the Gaussian even-moment recurrence
    /// `f(n+1) = (n + 1/2)·f(n)`.
    #[test]
    fn gaussian_even_moments() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            pool.pow(x, pool.mul(vec![pool.integer(2_i32), n])),
            pool.func(
                "exp",
                vec![pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))])],
            ),
        ]);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        let r = &out.value;
        assert_eq!(r.order, 1);
        // (2n+1)F(n) − 2F(n+1) = D_x(x·F): a_0/a_1 = −(2n+1)/2.
        assert_ratio(r, &pool, n, 0, |v| -(2.0 * v + 1.0) / 2.0);
    }

    /// `∫_0^∞ xⁿ·e^(−x²) dx = Γ((n+1)/2)/2` — genuinely order **2**, and the
    /// middle coefficient is zero. The ascending search must not settle for a
    /// spurious order-1 relation, and must not miss `a_1 = 0`.
    #[test]
    fn half_gaussian_moments_are_order_two() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            pool.pow(x, n),
            pool.func(
                "exp",
                vec![pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))])],
            ),
        ]);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        let r = &out.value;
        assert_eq!(r.order, 2, "f(n+2) = ((n+1)/2)·f(n) is an order-2 relation");
        assert_ratio(r, &pool, n, 1, |_| 0.0);
        assert_ratio(r, &pool, n, 0, |v| -(v + 1.0) / 2.0);
    }

    /// Bessel: `J_n(2) = (1/2πi)∮ e^(x − 1/x)·x^(−n−1) dx`, whose classical
    /// three-term recurrence at `z = 2` is `J_n + J_(n+2) = (n+1)·J_(n+1)`.
    ///
    /// The certificate here is the whole point — the contour representation is
    /// *outside* the interval boundary analysis, but the identity about the
    /// integrand is the same one either way.
    #[test]
    fn bessel_contour_representation() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let arg = pool.add(vec![
            x,
            pool.mul(vec![
                pool.integer(-1_i32),
                pool.pow(x, pool.integer(-1_i32)),
            ]),
        ]);
        let f = pool.mul(vec![
            pool.func("exp", vec![arg]),
            pool.pow(
                x,
                pool.add(vec![
                    pool.mul(vec![pool.integer(-1_i32), n]),
                    pool.integer(-1_i32),
                ]),
            ),
        ]);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        let r = &out.value;
        assert_eq!(r.order, 2, "Bessel's is a three-term recurrence");
        assert_ratio(r, &pool, n, 1, |v| -(v + 1.0));
        assert_ratio(r, &pool, n, 0, |_| 1.0);
    }

    /// Legendre via the Schläfli integral at `z = 3`:
    /// `P_n(3) = (1/2πi)∮ (x²−1)ⁿ / (2ⁿ·(x−3)^(n+1)) dx`, whose recurrence is
    /// `(n+2)P_(n+2) − 3(2n+3)P_(n+1) + (n+1)P_n = 0`.
    #[test]
    fn legendre_schlafli_representation() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let x2m1 = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
        let xm3 = pool.add(vec![x, pool.integer(-3_i32)]);
        let f = pool.mul(vec![
            pool.pow(x2m1, n),
            pool.pow(pool.integer(2_i32), pool.mul(vec![pool.integer(-1_i32), n])),
            pool.pow(
                xm3,
                pool.add(vec![
                    pool.mul(vec![pool.integer(-1_i32), n]),
                    pool.integer(-1_i32),
                ]),
            ),
        ]);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        let r = &out.value;
        assert_eq!(r.order, 2);
        assert_ratio(r, &pool, n, 1, |v| -3.0 * (2.0 * v + 3.0) / (v + 2.0));
        assert_ratio(r, &pool, n, 0, |v| (v + 1.0) / (v + 2.0));
    }

    /// The order-0 case is legitimate and must not be skipped: `F = x·e^(−x²)`
    /// carries no `n` at all and *is* `D_x(R·F)`.
    #[test]
    fn order_zero_is_reachable() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            x,
            pool.func(
                "exp",
                vec![pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))])],
            ),
        ]);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        assert_eq!(out.value.order, 0);
    }

    // --- negative tests ----------------------------------------------------

    #[test]
    fn out_of_class_integrand_is_refused() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![pool.pow(x, n), pool.func("log", vec![x])]);
        let err = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default())
            .expect_err("log is out of class");
        assert!(matches!(err, DiffTelescopingError::NotHyperexponential(_)));
    }

    /// `exp(n·x)/x` is hyperexponential in `x`, has no order-0 certificate, and
    /// is not hypergeometric in `n`. The refusal must say the *second* thing —
    /// raising the search bounds cannot help.
    #[test]
    fn non_hypergeometric_in_n_is_refused_specifically() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            pool.func("exp", vec![pool.mul(vec![n, x])]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        let err = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default())
            .expect_err("exp(n*x)/x has no relation in this family");
        assert!(
            matches!(err, DiffTelescopingError::NotHypergeometricInN(_)),
            "expected the n-side refusal, got {err:?}"
        );
    }

    #[test]
    fn identical_symbols_are_refused() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let err = almkvist_zeilberger(x, x, x, &pool, &AzOpts::default())
            .expect_err("n and x must differ");
        assert!(matches!(err, DiffTelescopingError::InvalidInput(_)));
    }

    #[test]
    fn bounds_too_small_refuse_rather_than_guess() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            pool.pow(x, n),
            pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), x])]),
        ]);
        let opts = AzOpts {
            max_order: 0,
            max_degree: 4,
            max_den_power: 1,
        };
        let err = almkvist_zeilberger(f, n, x, &pool, &opts).expect_err("bounds exhausted");
        assert!(matches!(err, DiffTelescopingError::SearchExhausted(_)));
    }

    /// The antiderivative helper must return `R·F`, the object whose boundary
    /// values the integral recurrence depends on.
    #[test]
    fn antiderivative_is_available() {
        let pool = ExprPool::new();
        let (n, x) = nx(&pool);
        let f = pool.mul(vec![
            pool.pow(x, n),
            pool.func("exp", vec![pool.mul(vec![pool.integer(-1_i32), x])]),
        ]);
        let out = almkvist_zeilberger(f, n, x, &pool, &AzOpts::default()).expect("certificate");
        let g = integrand_antiderivative(&out.value, f, &pool);
        let s = format!("{}", pool.display(g));
        assert!(s.contains("exp"), "G must still carry the integrand: {s}");
    }
}
