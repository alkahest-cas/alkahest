//! **Differential Gosper**: indefinite integration in the hyperexponential
//! class.
//!
//! Given a hyperexponential `F`, decide whether
//!
//! ```text
//! ∫ F dx = R·F     for some rational R,
//! ```
//!
//! and return `R` when it exists. This is the exact differential mirror of
//! Gosper's algorithm — "is the antiderivative of a hyperexponential term again
//! a *rational multiple* of that term" plays the same role as "is the
//! antidifference of a hypergeometric term again a rational multiple of it" —
//! and it is the inner loop `super::search` runs at every order.
//!
//! # It is one equation
//!
//! `D_x(R·F) = (R′ + θ·R)·F` with `θ = ∂_x F/F`, so the question is exactly
//! whether the Risch differential equation
//!
//! ```text
//! R′ + θ·R = 1
//! ```
//!
//! has a rational solution. That is [`super::rde`] with `J = 0` and `a_0 = 1`,
//! and it is solved there — including the reason
//! `integrate::risch::rational_rde` is not called instead, which is a
//! correctness reason and not a stylistic one.
//!
//! # What a returned `R` is and is not
//!
//! It is a proof: `R` is re-differentiated and `R′ + θ·R = 1` is checked as an
//! exact identity in `Q(n)(x)` before [`dgosper`] returns. It is not a claim
//! about *convergence* or about branches — `∫ x^(−1/2) dx = 2·x^(1/2)` comes
//! back as `R = 2x`, which is the correct formal antiderivative on either side
//! of `0` and says nothing about `∫_{−1}^{1}`.
//!
//! A refusal is a refusal of *this* method, not a proof that no elementary
//! antiderivative exists. `∫ eˣ/x dx` has none in the hyperexponential class
//! (it is `Ei(x)`), and `∫ e^(x²) dx` likewise, so both are genuine
//! Liouville-theoretic non-existence — but [`dgosper`] reports both the same
//! way it reports "the `κ`/`d` bounds ran out", as
//! [`super::DiffTelescopingError::SearchExhausted`]. Do not read it as a
//! non-existence theorem.

use super::hyperexp::{fresh_outer_index, HyperExpTerm};
use super::rde::{solve_probe, verify, RdeSetup};
use super::{AzOpts, DiffTelescopingError};
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::holonomic::hyperterm::ratk_to_expr;
use crate::holonomic::qfield::RatK;
use crate::kernel::{ExprId, ExprPool};

/// `∫ F dx = R·F` with `R` rational, or a typed refusal.
///
/// `f` must be hyperexponential in `x` (see [`super::hyperexp`]); it may not
/// contain any other free symbol, since `Q(x)` is the field the certificate
/// lives in. The returned expression is `R`, exactly, in lowest terms.
///
/// `opts.max_order` is ignored — the order is `0` by definition here. Only
/// `max_degree` and `max_den_power` bound the search.
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::holonomic::azeil::{dgosper, AzOpts};
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// // ∫ x·eˣ dx = (x−1)·eˣ, so R = (x−1)/x.
/// let f = pool.mul(vec![x, pool.func("exp", vec![x])]);
/// let r = dgosper(f, x, &pool, &AzOpts::default()).expect("hyperexponential antiderivative");
/// // The certificate is verified as an exact identity before it is returned.
/// assert!(format!("{}", pool.display(r.value)).contains('x'));
/// ```
pub fn dgosper(
    f: ExprId,
    x: ExprId,
    pool: &ExprPool,
    opts: &AzOpts,
) -> Result<DerivedExpr<ExprId>, DiffTelescopingError> {
    let n = fresh_outer_index(pool);
    if n == x {
        return Err(DiffTelescopingError::InvalidInput(
            "the integration variable collides with the internal outer index".into(),
        ));
    }
    let term = HyperExpTerm::parse(f, n, x, pool)?;
    let r = dgosper_term(&term, opts)?;
    let expr = ratk_to_expr(pool, n, x, &r);

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("dgosper_certificate", f, expr));
    Ok(DerivedExpr::with_log(expr, log))
}

/// [`dgosper`] on an already-parsed term, returning the certificate as an exact
/// `Q(n)(x)` element.
///
/// Split out because `super::search` and the boundary analysis both want the
/// exact object rather than an expression, and because a caller who already has
/// a [`HyperExpTerm`] should not have to round-trip it through the parser.
pub fn dgosper_term(term: &HyperExpTerm, opts: &AzOpts) -> Result<RatK, DiffTelescopingError> {
    let theta = term.theta()?;
    let setup = RdeSetup::new(&theta, vec![RatK::one()]).ok_or_else(|| {
        DiffTelescopingError::SearchExhausted(
            "the logarithmic derivative has a degenerate denominator".into(),
        )
    })?;

    let mut saw_candidate = false;
    for kappa in 0..=opts.max_den_power {
        for d in 0..=opts.max_degree {
            let Some(cand) = solve_probe(&setup, kappa, d) else {
                continue;
            };
            saw_candidate = true;
            // The only acceptance test: an exact identity in Q(n)(x).
            if verify(&setup, &cand.a, &cand.r) {
                return Ok(cand.r);
            }
        }
    }
    Err(if saw_candidate {
        DiffTelescopingError::CertificateVerificationFailed(
            "every candidate R found by the ansatz search failed the exact R' + theta*R = 1 \
             check in Q(n)(x)"
                .into(),
        )
    } else {
        DiffTelescopingError::SearchExhausted(format!(
            "no rational R with (R*F)' = F was found with deg R numerator <= {} and \
             certificate denominator power <= {}; this is a limit of the search, not a proof \
             that no elementary antiderivative exists",
            opts.max_degree, opts.max_den_power
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holonomic::azeil::hyperexp::hyperexp_log_derivative;
    use crate::holonomic::hyperterm::as_ratk;
    use crate::kernel::Domain;

    fn pool_x() -> (ExprPool, ExprId) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        (pool, x)
    }

    /// Independent check: parse `R` back and confirm `(R·F)' = F` by testing
    /// `R' + θ·R = 1` in exact `Q(x)` arithmetic, using nothing the search
    /// produced beyond `R` itself.
    fn assert_is_antiderivative(f: ExprId, r: ExprId, x: ExprId, pool: &ExprPool) {
        let n = fresh_outer_index(pool);
        let theta_expr = hyperexp_log_derivative(f, x, pool).expect("hyperexponential");
        let theta = as_ratk(theta_expr, n, x, pool, 0).expect("theta is rational");
        let rr = as_ratk(r, n, x, pool, 0).expect("R is rational");
        let lhs = crate::holonomic::qfield::ratk_deriv_k(&rr).add(&theta.mul(&rr));
        assert!(
            lhs.eq_ratk(&RatK::one()),
            "R' + theta*R must be 1; got {} for R = {}",
            pool.display(ratk_to_expr(pool, n, x, &lhs)),
            pool.display(r)
        );
    }

    #[test]
    fn exp_is_its_own_antiderivative() {
        let (pool, x) = pool_x();
        let f = pool.func("exp", vec![x]);
        let r = dgosper(f, x, &pool, &AzOpts::default()).expect("R = 1");
        assert_is_antiderivative(f, r.value, x, &pool);
    }

    #[test]
    fn x_times_exp_x() {
        // ∫ x·eˣ = (x−1)·eˣ ⇒ R = (x−1)/x, a simple pole at a simple pole of θ
        // with residue 1: the κ = 1 case.
        let (pool, x) = pool_x();
        let f = pool.mul(vec![x, pool.func("exp", vec![x])]);
        let r = dgosper(f, x, &pool, &AzOpts::default()).expect("certificate");
        assert_is_antiderivative(f, r.value, x, &pool);
    }

    #[test]
    fn x_squared_times_exp_x_needs_a_double_pole() {
        // ∫ x²·eˣ = (x²−2x+2)·eˣ ⇒ R = (x²−2x+2)/x². This is the case the
        // `E = gcd(D, D')` denominator bound in `integrate::risch::rational_rde`
        // misses, and the reason this module does not delegate to it.
        let (pool, x) = pool_x();
        let f = pool.mul(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.func("exp", vec![x]),
        ]);
        let r = dgosper(f, x, &pool, &AzOpts::default()).expect("certificate");
        assert_is_antiderivative(f, r.value, x, &pool);
    }

    #[test]
    fn gaussian_derivative() {
        // ∫ x·e^(−x²) dx = −e^(−x²)/2 ⇒ R = −1/(2x).
        let (pool, x) = pool_x();
        let f = pool.mul(vec![
            x,
            pool.func(
                "exp",
                vec![pool.mul(vec![pool.integer(-1_i32), pool.pow(x, pool.integer(2_i32))])],
            ),
        ]);
        let r = dgosper(f, x, &pool, &AzOpts::default()).expect("certificate");
        assert_is_antiderivative(f, r.value, x, &pool);
    }

    #[test]
    fn rational_power() {
        // ∫ x^(−1/2) dx = 2·x^(1/2) ⇒ R = 2x. Formal: says nothing about x < 0.
        let (pool, x) = pool_x();
        let f = pool.pow(x, pool.rational(-1, 2));
        let r = dgosper(f, x, &pool, &AzOpts::default()).expect("certificate");
        assert_is_antiderivative(f, r.value, x, &pool);
    }

    #[test]
    fn a_polynomial_is_hyperexponential_too() {
        // θ = 3/(x+1) for F = (x+1)³; ∫F = (x+1)⁴/4 ⇒ R = (x+1)/4.
        let (pool, x) = pool_x();
        let base = pool.add(vec![x, pool.integer(1_i32)]);
        let f = pool.pow(base, pool.integer(3_i32));
        let r = dgosper(f, x, &pool, &AzOpts::default()).expect("certificate");
        assert_is_antiderivative(f, r.value, x, &pool);
    }

    #[test]
    fn exp_over_x_has_no_hyperexponential_antiderivative() {
        // ∫ eˣ/x dx = Ei(x): genuinely not of the form R·F. Must refuse, not
        // return something wrong.
        let (pool, x) = pool_x();
        let f = pool.mul(vec![
            pool.func("exp", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        let err = dgosper(f, x, &pool, &AzOpts::default()).expect_err("Ei is not R*F");
        assert!(
            matches!(err, DiffTelescopingError::SearchExhausted(_)),
            "expected a search refusal, got {err:?}"
        );
    }

    #[test]
    fn erf_has_no_hyperexponential_antiderivative() {
        // ∫ e^(x²) dx is not elementary at all.
        let (pool, x) = pool_x();
        let f = pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]);
        let err = dgosper(f, x, &pool, &AzOpts::default()).expect_err("erfi is not R*F");
        assert!(matches!(err, DiffTelescopingError::SearchExhausted(_)));
    }

    #[test]
    fn non_hyperexponential_input_is_refused_cleanly() {
        let (pool, x) = pool_x();
        let f = pool.func("log", vec![x]);
        let err = dgosper(f, x, &pool, &AzOpts::default()).expect_err("log is out of class");
        assert!(matches!(err, DiffTelescopingError::NotHyperexponential(_)));
    }

    #[test]
    fn certificate_is_exact_for_a_rational_integrand() {
        // ∫ 1/x² dx = −1/x: F = x^(−2), R = −x.
        let (pool, x) = pool_x();
        let f = pool.pow(x, pool.integer(-2_i32));
        let r = dgosper(f, x, &pool, &AzOpts::default()).expect("certificate");
        assert_is_antiderivative(f, r.value, x, &pool);
    }
}
