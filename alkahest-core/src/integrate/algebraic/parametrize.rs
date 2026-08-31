//! Genus-0 reduction by rational parametrization — Risch milestones **M2 / MC0**.
//!
//! A single radical generator `y = r(x)^{1/n}` whose radicand `r` is a
//! **linear-fractional** function `r = (a₁x+a₀)/(b₁x+b₀)` (numerator and
//! denominator each of degree ≤ 1) defines a genus-0 curve `yⁿ = r(x)`.  Solving
//! `sⁿ = r(x)` for `x` gives the rational parametrization
//!
//! ```text
//!   x(s) = (a₀ − b₀·sⁿ) / (b₁·sⁿ − a₁),   y = s,
//! ```
//! and substituting `dx = x'(s) ds` turns `∫ R(x, y) dx` into an integrand that is
//! **rational in `s`** — hence always elementary and handled by the ordinary
//! rational/Risch engine.  Back-substituting `s = r(x)^{1/n}` recovers the
//! antiderivative.  The pure polynomial-linear case (`b₁ = 0`, M2) — `∛x/(x+1)`,
//! `x^{2/3}`, … — and the genuinely fractional case (MC0) — `√((1−x)/(1+x))`,
//! `∛((x+1)/(x−1))`, … — are the same formula.
//!
//! This covers the cubic-and-higher radical genus-0 cases the simple-radical
//! integral part (MA) cannot finish, **including their logarithmic part** (which
//! MA omitted — previously returning a *wrong* `NonElementary` for e.g.
//! `∫ ∛x/(x+1) dx`).
//!
//! Scope ([`try_parametrize_genus0`]): a single radical with linear-fractional
//! radicand (any index `n ≥ 2`).  Radicands `yⁿ = p(x)` of `deg ≥ 2` (non-Möbius)
//! are generally higher genus and out of scope **except** the genus-0
//! `√(quadratic)` case, which [`try_euler_quadratic`] handles for an arbitrary
//! rational `R(x, √(quadratic))` via an Euler substitution.  Both are sound by
//! construction: a result is accepted only after a numeric `d/dx F = integrand`
//! check.

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::integrate::risch::poly_rde::{
    degree, is_free_of_var, poly_mul, qpoly_to_expr, rational_to_expr, trim,
};
use crate::integrate::risch::rational_rde::expr_to_qrational;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use std::collections::HashMap;

type QPoly = Vec<rug::Rational>;

/// Try the genus-0 parametrization of a single linear-fractional-radicand
/// radical.  Returns `None` when the integrand is not of this shape (caller falls
/// through).
pub(super) fn try_parametrize_genus0(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let (n, radicand) = detect_single_radical(expr, var, pool)?;
    // Radicand as a reduced fraction num/den, each of degree ≤ 1.
    let (num, den) = expr_to_qrational(radicand, var, pool)?;
    let (num, den) = (trim(num), trim(den));
    if degree(&num) > 1 || degree(&den) > 1 || (degree(&num) < 1 && degree(&den) < 1) {
        return None;
    }
    let coeff = |p: &QPoly, i: usize| p.get(i).cloned().unwrap_or_else(|| rug::Rational::from(0));
    let (a0, a1) = (coeff(&num, 0), coeff(&num, 1));
    let (b0, b1) = (coeff(&den, 0), coeff(&den, 1));

    // s = r(x)^{1/n};  x(s) = (a₀ − b₀·sⁿ)/(b₁·sⁿ − a₁).
    let s = pool.symbol("$param_s$", Domain::Real);
    let s_n = pool.pow(s, pool.integer(n as i32));
    let lin = |c1: &rug::Rational, c0: &rug::Rational| {
        // c1·sⁿ + c0
        pool.add(vec![
            pool.mul(vec![rational_to_expr(c1, pool), s_n]),
            rational_to_expr(c0, pool),
        ])
    };
    let x_num = lin(&-b0, &a0); // −b₀·sⁿ + a₀
    let x_den = lin(&b1, &-a1.clone()); // b₁·sⁿ − a₁
    if degree(&num) < 1 && b1 == 0 {
        return None; // x would not depend on s
    }
    let x_of_s = pool.mul(vec![x_num, pool.pow(x_den, pool.integer(-1))]);

    // Rewrite the integrand directly in `s`: standalone `x → x(s)`, and every
    // power `r(x)^{c/d}` of the radicand → `s^{c·n/d}`, so no un-reducible
    // `(sⁿ)^{1/n}` is ever formed.
    let core = to_s(expr, var, &num, &den, n, s, x_of_s, pool)?;
    let dx_ds = simplify(crate::diff::diff(x_of_s, s, pool).ok()?.value, pool).value;
    let integrand_s = simplify(pool.mul(vec![core, dx_ds]), pool).value;

    // Integrate the rational-in-`s` integrand (always elementary), then
    // back-substitute s = r(x)^{1/n}.
    let f_s = {
        // This frame consumes a `RootSum` rather than passing one up: see
        // `expand_rootsums` on the next line.  Say so, or an enclosing
        // `RootSumSuppressed` (from `generator_subst`, `subst` or the engine's
        // Weierstrass / u-substitution routes) reaches down through this
        // sub-integration and suppresses the very `RootSum` this route exists
        // to expand.
        let _expanded =
            crate::integrate::risch::rational_integrate::RootSumExpandedByCaller::enter();
        match crate::integrate::engine::integrate(integrand_s, s, pool) {
            Ok(d) => d.value,
            Err(_) => return None,
        }
    };
    // Resolve an algebraic-residue `RootSum` into real `log`/`atan` before
    // back-substitution (`subs` cannot enter the binder), or decline.
    let f_s = super::rootsum_expand::expand_rootsums(f_s, pool)?;
    let radical_expr = pool.pow(radicand, pool.rational(1, n as i32));
    let mut back = HashMap::new();
    back.insert(s, radical_expr);
    let f_x = simplify(crate::kernel::subs(f_s, &back, pool), pool).value;

    // Soundness gate: d/dx F = integrand numerically (where the radicand > 0).
    if !verify_derivative(f_x, expr, radicand, var, pool) {
        return None;
    }

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "algebraic_genus0_parametrize",
        expr,
        f_x,
    ));
    Some(Ok(DerivedExpr::with_log(f_x, log)))
}

/// Genus-0 integration of `∫ R(x, √(a x²+b x+c)) dx` with **`R` an arbitrary
/// rational function** (not just a polynomial coefficient on the radical), via an
/// **Euler substitution**.  A nondegenerate quadratic radicand is a genus-0
/// conic, so a rational point gives a parameter `t` in which both `x` and
/// `√(quad)` are rational — turning the whole integrand rational in `t` (always
/// elementary).  Two substitutions cover the rational-point cases:
///
/// * `a = e²` a perfect square: `√(quad) = t − e·x`, so
///   `x = (t²−c)/(2e·t + b)`, and `t = √(quad) + e·x`;
/// * else `c = g²` a perfect square: `√(quad) = x·t + g`, so
///   `x = (2g·t − b)/(a − t²)`, and `t = (√(quad) − g)/x`.
///
/// Returns `None` when not a single `sqrt(quadratic-over-ℚ[x])` generator, or
/// when neither leading nor constant coefficient is a rational square (a rational
/// point at infinity / at `x=0` is then unavailable in this bounded form).
pub(super) fn try_euler_quadratic(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let (n, radicand) = detect_single_radical(expr, var, pool)?;
    if n != 2 {
        return None;
    }
    // Radicand must be a degree-2 polynomial in x over ℚ.
    let (num, den) = expr_to_qrational(radicand, var, pool)?;
    let (num, den) = (trim(num), trim(den));
    if degree(&den) != 0 || degree(&num) != 2 {
        return None;
    }
    let coeff = |p: &QPoly, i: usize| p.get(i).cloned().unwrap_or_else(|| rug::Rational::from(0));
    let (c, b, a) = (coeff(&num, 0), coeff(&num, 1), coeff(&num, 2));
    let quad = num.clone(); // a·x²+b·x+c (den is the constant 1 after normalization)

    let t = pool.symbol("$euler_t$", Domain::Real);
    let two = rug::Rational::from(2);
    let radical = pool.func("sqrt", vec![radicand]); // √(quad) in x, for back-sub
    let (x_of_t, sqrt_t, back_t) = if let Some(e) = sqrt_rational(&a) {
        // x = (t²−c)/(2e·t + b);  √quad = t − e·x;  t = √quad + e·x.
        let t2 = pool.pow(t, pool.integer(2));
        let x_num = pool.add(vec![t2, rational_to_expr(&-c.clone(), pool)]);
        let x_den = pool.add(vec![
            pool.mul(vec![rational_to_expr(&(two.clone() * &e), pool), t]),
            rational_to_expr(&b, pool),
        ]);
        let x_of_t = pool.mul(vec![x_num, pool.pow(x_den, pool.integer(-1))]);
        let sqrt_t = simplify(
            pool.add(vec![
                t,
                pool.mul(vec![rational_to_expr(&-e.clone(), pool), x_of_t]),
            ]),
            pool,
        )
        .value;
        let back_t = pool.add(vec![
            radical,
            pool.mul(vec![rational_to_expr(&e, pool), var]),
        ]);
        (x_of_t, sqrt_t, back_t)
    } else if let Some(g) = sqrt_rational(&c) {
        // x = (2g·t − b)/(a − t²);  √quad = x·t + g;  t = (√quad − g)/x.
        let t2 = pool.pow(t, pool.integer(2));
        let x_num = pool.add(vec![
            pool.mul(vec![rational_to_expr(&(two.clone() * &g), pool), t]),
            rational_to_expr(&-b.clone(), pool),
        ]);
        let x_den = pool.add(vec![
            rational_to_expr(&a, pool),
            pool.mul(vec![rational_to_expr(&rug::Rational::from(-1), pool), t2]),
        ]);
        let x_of_t = pool.mul(vec![x_num, pool.pow(x_den, pool.integer(-1))]);
        let sqrt_t = simplify(
            pool.add(vec![pool.mul(vec![x_of_t, t]), rational_to_expr(&g, pool)]),
            pool,
        )
        .value;
        let back_t = pool.mul(vec![
            pool.add(vec![radical, rational_to_expr(&-g.clone(), pool)]),
            pool.pow(var, pool.integer(-1)),
        ]);
        (x_of_t, sqrt_t, back_t)
    } else {
        return None;
    };

    // Rewrite the integrand rational in `t`, multiply by dx/dt, integrate, and
    // back-substitute `t`.
    let core = to_t(expr, var, &quad, sqrt_t, x_of_t, pool)?;
    let dx_dt = simplify(crate::diff::diff(x_of_t, t, pool).ok()?.value, pool).value;
    let integrand_t = simplify(pool.mul(vec![core, dx_dt]), pool).value;
    let f_t = {
        // As in `try_genus0_rational_radicand`: this frame expands `RootSum`s
        // instead of passing them up, so an enclosing suppression must not
        // reach into it.
        let _expanded =
            crate::integrate::risch::rational_integrate::RootSumExpandedByCaller::enter();
        match crate::integrate::engine::integrate(integrand_t, t, pool) {
            Ok(d) => d.value,
            Err(_) => return None,
        }
    };
    // An algebraic-residue logarithmic part comes back as a `RootSum` binder,
    // which `subs` cannot enter and no verification tier can evaluate.  Expand
    // it to explicit real `log`/`atan` first; declining when it will not expand
    // leaves exactly the decline this route already gave.
    let f_t = super::rootsum_expand::expand_rootsums(f_t, pool)?;
    let mut back = HashMap::new();
    back.insert(t, back_t);
    let f_x = simplify(crate::kernel::subs(f_t, &back, pool), pool).value;

    let f_x = collapse_euler_shape(f_x, expr, radicand, var, pool)?;
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("algebraic_genus0_euler", expr, f_x));
    Some(Ok(DerivedExpr::with_log(f_x, log)))
}

/// Put an Euler answer back into `A(x) + B(x)·√P` form, then gate it.
///
/// `t = x + √P` is a rational parameter, so what comes back from the `t`-integral
/// is a rational function *of `t`* — powers of `x + √P` with both signs, not the
/// radical.  `∫x/√(x²−1) dx` arrives as `½(x+√(x²−1)) − ½(x+√(x²−1))⁻¹` rather
/// than `√(x²−1)`.  That is merely ugly as a final answer, but fatal as the `v`
/// of an integration-by-parts step, which has to differentiate and re-integrate
/// the shape it is handed.
///
/// [`super::decompose::normalize_over_sqrt`] rationalizes those radical
/// denominators through the field inverse `1/(a+b·y) = (a−b·y)/(a²−b²P)`, which
/// collapses the pair above to `√(x²−1)`.
///
/// The collapse is *shape only* — but `simplify` is not a proof, so the
/// normalized form is gated first and the original is gated as a fallback.  A
/// normalization that lost something therefore costs nothing: the un-normalized
/// answer is still emitted.  Only if **both** fail does the route decline.
fn collapse_euler_shape(
    f_x: ExprId,
    integrand: ExprId,
    radicand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let collapsed = simplify(
        super::decompose::normalize_over_sqrt(f_x, radicand, var, pool),
        pool,
    )
    .value;
    if collapsed != f_x && verify_derivative(collapsed, integrand, radicand, var, pool) {
        return Some(collapsed);
    }
    if verify_derivative(f_x, integrand, radicand, var, pool) {
        return Some(f_x);
    }
    None
}

/// General genus-0 `∫ R(x, √(a·x²+b·x+c)) dx` for **any** nondegenerate
/// quadratic radicand with `a > 0` — including the cases where **neither** the
/// leading coefficient `a` **nor** the constant `c` is a rational square
/// (`√(2x²+3)`, `√(3x²+2x+2)`, …), which [`try_euler_quadratic`] declines because
/// no rational point on the conic is available in its bounded form.
///
/// Completing the square gives `a·x²+b·x+c = a·((x + b/2a)² + k)` with
/// `k = c/a − b²/4a²`, so with `u = x + b/2a` the radical factors as
/// `√(quad) = √a · √(u²+k)`, and the **monic** `u²+k` (leading coefficient
/// `1 = 1²`) has the rational point at infinity the first-kind Euler
/// substitution `t = u + √(u²+k)` needs.  That substitution makes `u`, `√(u²+k)`,
/// and hence the whole integrand rational in `t`; the irrational constant `√a`
/// rides along as an **opaque symbol** `k_a` (so the recursively-integrated
/// integrand is a genuine rational function of `t` — never a bare `sqrt` constant
/// that the engine would misroute as an algebraic generator), and is resolved to
/// `√a` only at the very end.
///
/// Only the `a > 0` branch (monic `u²+k`) is taken; the `a < 0` conic reduces to
/// `√(k−u²)` (an `arcsin`-type genus-0 form) and is left to decline.  As always,
/// a result is emitted only after the shared numeric `d/dx F = integrand` gate,
/// so an unsupported reduction (e.g. one needing an `arctan` the constant-coefficient
/// rational engine cannot form) simply declines — never a wrong integral.
pub(super) fn try_euler_quadratic_general(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let (n, radicand) = detect_single_radical(expr, var, pool)?;
    if n != 2 {
        return None;
    }
    let (num, den) = expr_to_qrational(radicand, var, pool)?;
    let (num, den) = (trim(num), trim(den));
    if degree(&den) != 0 || degree(&num) != 2 {
        return None;
    }
    let coeff = |p: &QPoly, i: usize| p.get(i).cloned().unwrap_or_else(|| rug::Rational::from(0));
    let (c, b, a) = (coeff(&num, 0), coeff(&num, 1), coeff(&num, 2));
    // Only the a>0 branch (monic `u²+k`); a<0 is the arcsin conic, left to decline.
    if a <= 0 {
        return None;
    }
    let quad = num.clone();

    // shift = b/(2a);  k = c/a − b²/(4a²).
    let shift = b.clone() / (rug::Rational::from(2) * a.clone());
    let k = c.clone() / a.clone()
        - (b.clone() * b.clone()) / (rug::Rational::from(4) * a.clone() * a.clone());
    let neg_k = -k.clone();
    let neg_shift = -shift.clone();

    let t = pool.symbol("$euler_t$", Domain::Real);
    let k_a = pool.symbol("$euler_sqrt_a$", Domain::Real); // opaque √a
    let t2 = pool.pow(t, pool.integer(2));
    let inv_two_t = pool.pow(pool.mul(vec![pool.integer(2), t]), pool.integer(-1));
    // u(t) = (t²−k)/(2t);  x(t) = u − shift.
    let u_of_t = pool.mul(vec![
        pool.add(vec![t2, rational_to_expr(&neg_k, pool)]),
        inv_two_t,
    ]);
    let x_of_t = simplify(
        pool.add(vec![u_of_t, rational_to_expr(&neg_shift, pool)]),
        pool,
    )
    .value;
    // √(u²+k) = (t²+k)/(2t) — the *monic* radical value (rational in `t`); the
    // actual `√(quad) = k_a · sqrt_u`, with the constant `k_a = √a` kept as a
    // separate factor so every radical power emits `k_a^M · sqrt_u^M` (a
    // *distributed* product), letting `simplify` collect all `k_a` powers into a
    // single leading constant the rational engine can factor out.
    let sqrt_u = pool.mul(vec![
        pool.add(vec![t2, rational_to_expr(&k, pool)]),
        inv_two_t,
    ]);

    // Rewrite the integrand rational in `t` (`to_t_scaled`: any power of the
    // radicand → `k_a^M · sqrt_u^M`), times dx/dt.
    let core = to_t_scaled(expr, var, &quad, k_a, sqrt_u, x_of_t, pool)?;
    let dx_dt = simplify(crate::diff::diff(x_of_t, t, pool).ok()?.value, pool).value;
    let integrand_t = simplify(pool.mul(vec![core, dx_dt]), pool).value;
    // Integrate term-by-term.  The opaque constant `k_a = √a` would defeat the
    // rational-function integrator (which normalizes over ℚ(t)), so from each
    // additive term we pull *every* `t`-free factor — including the `k_a` power —
    // out as a constant, collapse the remaining `t`-part into a single rational
    // function `N(t)/D(t)`, integrate that pure ℚ(t) integrand with the engine,
    // and multiply the constant back (linearity).
    let f_t = {
        let _expanded =
            crate::integrate::risch::rational_integrate::RootSumExpandedByCaller::enter();
        integrate_scaled_rational(integrand_t, t, pool)?
    };
    // As in `try_euler_quadratic`: resolve an algebraic-residue `RootSum` into
    // real `log`/`atan` before back-substitution, or decline.
    let f_t = super::rootsum_expand::expand_rootsums(f_t, pool)?;

    // Back-substitute t = u + √(u²+k) = (x+shift) + √(quad)/√a, then the opaque
    // symbol k_a → √a.
    let radical = pool.func("sqrt", vec![radicand]);
    let back_t = pool.add(vec![
        var,
        rational_to_expr(&shift, pool),
        pool.mul(vec![radical, pool.pow(k_a, pool.integer(-1))]),
    ]);
    let mut back = HashMap::new();
    back.insert(t, back_t);
    let f_bt = crate::kernel::subs(f_t, &back, pool);
    let sqrt_a_val = pool.func("sqrt", vec![rational_to_expr(&a, pool)]);
    let mut back_a = HashMap::new();
    back_a.insert(k_a, sqrt_a_val);
    let f_x = simplify(crate::kernel::subs(f_bt, &back_a, pool), pool).value;

    let f_x = collapse_euler_shape(f_x, expr, radicand, var, pool)?;
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "algebraic_genus0_euler_general",
        expr,
        f_x,
    ));
    Some(Ok(DerivedExpr::with_log(f_x, log)))
}

/// Genus-0 `∫ R(x, √(a x²+b x+c)) dx` through **any** rational point of the conic
/// `y² = a x² + b x + c`.
///
/// [`try_euler_quadratic`] uses the two *classical* points — the one at infinity
/// (needs `a` a rational square) and the one at `x = 0` (needs `c` a rational
/// square) — and [`try_euler_quadratic_general`] completes the square for `a > 0`
/// when neither is available.  What is left over is `a < 0` with an irrational
/// `√c`: `∫√(2−x²)/(1+x²) dx`, `∫dx/((x²+1)√(3−x²))`, and every other bounded
/// radicand whose value at `x = 0` is not a square.  That is not a hard case, it
/// is a *badly chosen basepoint*: `y² = 2−x²` has the perfectly good rational
/// point `(1, 1)`.
///
/// A conic with one rational point has infinitely many, and the pencil of lines
/// through a chosen point `(x₀, y₀)` parametrizes all of them.  Substituting
/// `y = y₀ + m·(x−x₀)` into `y² = Q(x)` and cancelling the known root `x = x₀`,
///
/// ```text
///   x(m) = (m²·x₀ + a·x₀ + b − 2·y₀·m)/(m² − a),   y(m) = y₀ + m·(x(m) − x₀),
/// ```
/// with the inverse `m = (√Q − y₀)/(x − x₀)`.  Both are rational with **rational**
/// coefficients — which is what the two earlier routes cannot arrange for `a < 0`
/// with irrational `√c` (their parameter drags `√|a|` or `√c` into the
/// coefficients of `x(t)`, and the resulting integrand is then no longer in
/// `ℚ(t)` for the rational integrator to take).
///
/// The point is found by [`find_conic_point`]: a rational root of `Q` when the
/// discriminant is a rational square (this is the classical *third* Euler
/// substitution, `y₀ = 0`), otherwise a bounded search for a small rational `x₀`
/// with `Q(x₀)` a rational square.  A conic can genuinely have no rational point
/// at all (`y² = 3x²+2` has none, by Hasse–Minkowski), so the search failing is a
/// decline and never a claim about the integral.
///
/// Tried **last**, after both earlier Euler routes have declined, so no integrand
/// that already had an answer changes shape.  As always the result is emitted
/// only through the shared `d/dx F = integrand` gate.
///
/// # Branch caveat
///
/// `m → ∞` as `x → x₀`, so the emitted antiderivative can jump by a constant
/// across the basepoint — the same removable artefact the existing `x = 0`
/// substitution already has (`∫dx/(x√(1−x²))` comes back as
/// `log((√(1−x²)−1)/x)`, singular at `0`).  The derivative, which is what the
/// gate checks, is unaffected away from `x₀`.
pub(super) fn try_euler_conic_point(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let (n, radicand) = detect_single_radical(expr, var, pool)?;
    if n != 2 {
        return None;
    }
    let (num, den) = expr_to_qrational(radicand, var, pool)?;
    let (num, den) = (trim(num), trim(den));
    if degree(&den) != 0 || degree(&num) != 2 {
        return None;
    }
    let coeff = |p: &QPoly, i: usize| p.get(i).cloned().unwrap_or_else(|| rug::Rational::from(0));
    let (c, b, a) = (coeff(&num, 0), coeff(&num, 1), coeff(&num, 2));
    // Nondegenerate: a double root is `√(a(x−r)²) = √a·|x−r|`, not a genus-0
    // conic parametrization problem, and the sign of `x−r` would be a branch
    // choice this route has no way to make.
    let disc = b.clone() * b.clone() - rug::Rational::from(4) * a.clone() * c.clone();
    if disc == 0 {
        return None;
    }
    let quad = num.clone();
    let (x0, y0) = find_conic_point(&a, &b, &c, &disc)?;

    // x(m) = (m²·x₀ + a·x₀ + b − 2·y₀·m)/(m² − a).
    let m = pool.symbol("$euler_t$", Domain::Real);
    let m2 = pool.pow(m, pool.integer(2));
    let k0 = a.clone() * x0.clone() + b.clone();
    let x_num = pool.add(vec![
        pool.mul(vec![rational_to_expr(&x0, pool), m2]),
        rational_to_expr(&k0, pool),
        pool.mul(vec![
            rational_to_expr(&(rug::Rational::from(-2) * y0.clone()), pool),
            m,
        ]),
    ]);
    let x_den = pool.add(vec![m2, rational_to_expr(&-a.clone(), pool)]);
    let x_of_m = simplify(
        pool.mul(vec![x_num, pool.pow(x_den, pool.integer(-1))]),
        pool,
    )
    .value;
    // y(m) = y₀ + m·(x(m) − x₀) — the value the radical takes on the parameter.
    let sqrt_m = simplify(
        pool.add(vec![
            rational_to_expr(&y0, pool),
            pool.mul(vec![
                m,
                pool.add(vec![x_of_m, rational_to_expr(&-x0.clone(), pool)]),
            ]),
        ]),
        pool,
    )
    .value;
    // m = (√Q − y₀)/(x − x₀).
    let radical = pool.func("sqrt", vec![radicand]);
    let back_m = pool.mul(vec![
        pool.add(vec![radical, rational_to_expr(&-y0.clone(), pool)]),
        pool.pow(
            pool.add(vec![var, rational_to_expr(&-x0.clone(), pool)]),
            pool.integer(-1),
        ),
    ]);

    let core = to_t(expr, var, &quad, sqrt_m, x_of_m, pool)?;
    let dx_dm = simplify(crate::diff::diff(x_of_m, m, pool).ok()?.value, pool).value;
    let integrand_m = simplify(pool.mul(vec![core, dx_dm]), pool).value;
    let f_m = {
        let _expanded =
            crate::integrate::risch::rational_integrate::RootSumExpandedByCaller::enter();
        match crate::integrate::engine::integrate(integrand_m, m, pool) {
            Ok(d) => d.value,
            Err(_) => return None,
        }
    };
    let f_m = super::rootsum_expand::expand_rootsums(f_m, pool)?;
    let mut back = HashMap::new();
    back.insert(m, back_m);
    let f_x = simplify(crate::kernel::subs(f_m, &back, pool), pool).value;

    let f_x = collapse_euler_shape(f_x, expr, radicand, var, pool)?;
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "algebraic_genus0_euler_conic_point",
        expr,
        f_x,
    ));
    Some(Ok(DerivedExpr::with_log(f_x, log)))
}

/// A rational point `(x₀, y₀)` on `y² = a·x² + b·x + c`, or `None`.
///
/// Prefers a root of `Q` (`y₀ = 0`, the classical third Euler substitution) and
/// otherwise scans small rationals.  `None` is "not found within the search",
/// which for a conic with **no** rational point at all (`y² = 3x²+2`) is also the
/// truth — either way the caller declines rather than concluding anything.
fn find_conic_point(
    a: &rug::Rational,
    b: &rug::Rational,
    c: &rug::Rational,
    disc: &rug::Rational,
) -> Option<(rug::Rational, rug::Rational)> {
    /// Numerator/denominator bound for the basepoint scan.  A conic with a
    /// rational point has one of small height in the cases that arise here;
    /// widening this only trades time for a longer tail.
    const SCAN: i32 = 12;

    // Rational root of Q: the third Euler substitution.
    if let Some(r) = sqrt_rational(disc) {
        if *a != 0 {
            let x0 = (-b.clone() + r) / (rug::Rational::from(2) * a.clone());
            return Some((x0, rug::Rational::from(0)));
        }
    }
    // Otherwise look for a small rational `x₀` with `Q(x₀)` a rational square.
    let q =
        |x: &rug::Rational| a.clone() * x.clone() * x.clone() + b.clone() * x.clone() + c.clone();
    for den in 1..=SCAN {
        for numer in -(SCAN * SCAN)..=(SCAN * SCAN) {
            let x0 = rug::Rational::from((numer, den));
            let v = q(&x0);
            if v <= 0 {
                continue;
            }
            if let Some(y0) = sqrt_rational(&v) {
                return Some((x0, y0));
            }
        }
    }
    None
}

/// Genus-0 `∫ R(x, √(a·x²+b·x+c)) dx` for a **negative** leading coefficient
/// `a < 0` — the `arcsin` family (`∫dx/√(2−3x²)`, `∫x/√(5−2x²)`,
/// `∫dx/√(−x²+2x+3)`, `∫√(2−3x²) dx`, …).
///
/// For `a < 0` the conic `y² = a x²+b x+c` has no real point at infinity, so the
/// Euler substitution used for `a > 0` ([`try_euler_quadratic_general`]) has no
/// real form; the natural normal form is `arcsin`.  Completing the square,
/// ```text
///   a·x²+b·x+c = |a|·(k² − (x−h)²),   h = −b/(2a),   k² = (c − b²/(4a))/|a|,
/// ```
/// (`k² > 0` iff the radicand is positive somewhere — else it is `√(negative)`
/// everywhere and we decline).  The shift `w = x−h` (rational, `dx = dw`) gives
/// `√(quad) = √|a|·√(k²−w²)`, turning the integrand into
/// `Σ_n c_n · wⁿ / √(k²−w²)` (a polynomial numerator over the radical) whenever
/// `R` is such that multiplying by the radical clears it — i.e. `R = poly(x)/√P`
/// or `poly(x)·√P`.  Those reduce by the standard table integrals
/// ```text
///   ∫ dw/√(k²−w²)      = asin(w/k),
///   ∫ w/√(k²−w²) dw    = −√(k²−w²),
///   ∫ wⁿ/√(k²−w²) dw   = −wⁿ⁻¹√(k²−w²)/n + (n−1)k²/n · ∫ wⁿ⁻²/√(k²−w²) dw,
/// ```
/// back-substituted `w = x−h`, `√(k²−w²) = √(quad)/√|a|`.
///
/// Returns `None` (declines, no regression) unless the integrand is exactly a
/// single `√(a x²+b x+c)` generator with `a < 0`, a positive real interval, and a
/// polynomial-numerator-over-radical shape.  As always the result is emitted only
/// after the shared numeric `d/dx F = integrand` gate — an unsupported shape (a
/// rational weight with a pole, a mixed rational+radical integrand, …) simply
/// declines rather than emitting a wrong answer.
pub(super) fn try_arcsin_quadratic(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let (n, radicand) = detect_single_radical(expr, var, pool)?;
    if n != 2 {
        return None;
    }
    let (num, den) = expr_to_qrational(radicand, var, pool)?;
    let (num, den) = (trim(num), trim(den));
    if degree(&den) != 0 || degree(&num) != 2 {
        return None;
    }
    let coeff = |p: &QPoly, i: usize| p.get(i).cloned().unwrap_or_else(|| rug::Rational::from(0));
    let (c, b, a) = (coeff(&num, 0), coeff(&num, 1), coeff(&num, 2));
    // Only the a<0 branch (the arcsin conic); a>0 is left to the earlier paths.
    if a >= 0 {
        return None;
    }
    let quad = num.clone();

    // h = −b/(2a);  |a| = −a;  k² = (c − b²/(4a))/|a|  (need k² > 0).
    let abs_a = -a.clone();
    let h = -b.clone() / (rug::Rational::from(2) * a.clone());
    let disc = c.clone() - (b.clone() * b.clone()) / (rug::Rational::from(4) * a.clone());
    let kk = disc / abs_a.clone();
    if kk <= 0 {
        return None; // radicand negative everywhere — √(negative), decline
    }

    // Opaque constants: amp = √|a|, s = √(k²−w²); w is the shifted variable.
    let w = pool.symbol("$arcsin_w$", Domain::Real);
    let s = pool.symbol("$arcsin_s$", Domain::Real);
    let amp = pool.symbol("$arcsin_amp$", Domain::Real);
    let x_of_w = pool.add(vec![w, rational_to_expr(&h, pool)]);

    // Rewrite R in `w`, `√(quad) = amp·s` (reusing the completed-square rewriter):
    // standalone x → w+h, every radical power `quad^{c/d}` → amp^M · s^M.
    let integrand_w = to_t_scaled(expr, var, &quad, amp, s, x_of_w, pool)?;
    // Multiply by the radical `s` so a `poly/√` term becomes a bare polynomial and
    // a `poly·√` term becomes `poly·s²`; then rewrite the even power `s² = k²−w²`.
    let prod = simplify(pool.mul(vec![integrand_w, s]), pool).value;
    let kk_minus_w2 = pool.add(vec![
        rational_to_expr(&kk, pool),
        pool.mul(vec![pool.integer(-1), pool.pow(w, pool.integer(2))]),
    ]);
    let reduced = reduce_even_s_powers(prod, s, kk_minus_w2, pool);
    // A leftover odd power of `s` means a genuine rational (non-radical) part or an
    // `s⁻³`-type shape this table does not cover: decline.
    if !is_free_of_var(reduced, s, pool) {
        return None;
    }
    // Collect `reduced` as a polynomial in `w` (coefficients free of `w`, possibly
    // carrying the opaque `amp`).  A pole in `w` (rational weight) → `None`.
    let coeffs = poly_coeffs_in(reduced, w, pool)?;

    // Integrate `Σ c_n wⁿ / √(k²−w²)` by the asin/√ table above.
    let f_w = integrate_poly_over_neg_quad(&coeffs, w, s, &kk, pool);

    // Back-substitute: w → x−h,  amp → √|a|,  s → √(quad)/√|a|.
    let amp_val = pool.func("sqrt", vec![rational_to_expr(&abs_a, pool)]);
    let sqrt_p = pool.func("sqrt", vec![radicand]);
    let s_val = pool.mul(vec![sqrt_p, pool.pow(amp_val, pool.integer(-1))]);
    let mut back = HashMap::new();
    back.insert(w, pool.add(vec![var, rational_to_expr(&-h.clone(), pool)]));
    back.insert(amp, amp_val);
    back.insert(s, s_val);
    let f_x = simplify(crate::kernel::subs(f_w, &back, pool), pool).value;

    if !verify_derivative(f_x, expr, radicand, var, pool) {
        return None;
    }
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("algebraic_arcsin_quadratic", expr, f_x));
    Some(Ok(DerivedExpr::with_log(f_x, log)))
}

/// Rewrite every **even** integer power `s^{2m}` of the opaque radical symbol `s`
/// to `(k²−w²)^m` (`kk_minus_w2`), leaving odd powers of `s` untouched (so a
/// caller can detect an uncovered shape by testing whether the result still
/// contains `s`).  Used by [`try_arcsin_quadratic`] after multiplying the
/// integrand by one factor of `s`.
fn reduce_even_s_powers(expr: ExprId, s: ExprId, kk_minus_w2: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Pow { base, exp } if base == s => {
            if let ExprData::Integer(e) = pool.get(exp) {
                if let Some(ei) = e.0.to_i64() {
                    if ei % 2 == 0 {
                        return pool.pow(kk_minus_w2, pool.integer((ei / 2) as i32));
                    }
                }
            }
            expr // odd/non-integer power of s: leave it (signals decline)
        }
        ExprData::Add(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| reduce_even_s_powers(a, s, kk_minus_w2, pool))
                .collect();
            pool.add(v)
        }
        ExprData::Mul(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| reduce_even_s_powers(a, s, kk_minus_w2, pool))
                .collect();
            pool.mul(v)
        }
        ExprData::Pow { base, exp } => {
            let nb = reduce_even_s_powers(base, s, kk_minus_w2, pool);
            pool.pow(nb, exp)
        }
        ExprData::Func { ref name, ref args } => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| reduce_even_s_powers(a, s, kk_minus_w2, pool))
                .collect();
            pool.func(name.clone(), v)
        }
        _ => expr,
    }
}

/// Collect `expr` as a polynomial in `w`: a map `degree → coefficient` where each
/// coefficient is an [`ExprId`] free of `w` (it may carry other symbols, e.g. the
/// opaque `amp = √|a|`).  Returns `None` if `expr` is not polynomial in `w` (a
/// `w`-dependent denominator / pole, a radical or transcendental of `w`, …).
fn poly_coeffs_in(expr: ExprId, w: ExprId, pool: &ExprPool) -> Option<HashMap<usize, ExprId>> {
    if expr == w {
        let mut m = HashMap::new();
        m.insert(1usize, pool.integer(1));
        return Some(m);
    }
    if is_free_of_var(expr, w, pool) {
        let mut m = HashMap::new();
        m.insert(0usize, expr);
        return Some(m);
    }
    match pool.get(expr) {
        ExprData::Add(args) => {
            let mut acc: HashMap<usize, ExprId> = HashMap::new();
            for &a in &args {
                let cm = poly_coeffs_in(a, w, pool)?;
                for (d, c) in cm {
                    acc.entry(d)
                        .and_modify(|e| *e = pool.add(vec![*e, c]))
                        .or_insert(c);
                }
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc: HashMap<usize, ExprId> = HashMap::new();
            acc.insert(0usize, pool.integer(1));
            for &a in &args {
                let cm = poly_coeffs_in(a, w, pool)?;
                acc = poly_convolve(&acc, &cm, pool);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let e = match pool.get(exp) {
                ExprData::Integer(e) => e.0.to_i64()?,
                _ => return None,
            };
            if e < 0 {
                return None; // w-dependent denominator: pole, decline
            }
            let bm = poly_coeffs_in(base, w, pool)?;
            let mut acc: HashMap<usize, ExprId> = HashMap::new();
            acc.insert(0usize, pool.integer(1));
            for _ in 0..e {
                acc = poly_convolve(&acc, &bm, pool);
            }
            Some(acc)
        }
        _ => None,
    }
}

/// Polynomial (Cauchy) product of two `degree → coefficient` maps.
fn poly_convolve(
    p: &HashMap<usize, ExprId>,
    q: &HashMap<usize, ExprId>,
    pool: &ExprPool,
) -> HashMap<usize, ExprId> {
    let mut out: HashMap<usize, ExprId> = HashMap::new();
    for (&dp, &cp) in p {
        for (&dq, &cq) in q {
            let term = pool.mul(vec![cp, cq]);
            out.entry(dp + dq)
                .and_modify(|e| *e = pool.add(vec![*e, term]))
                .or_insert(term);
        }
    }
    out
}

/// Integrate `∫ (Σ_n c_n wⁿ) / √(k²−w²) dw` by the `asin`/√ reduction, returning
/// the antiderivative in terms of `w`, the opaque radical `s = √(k²−w²)`, and
/// `asin(w/√(k²))` (`kk = k²`, rational).  The `c_n` are arbitrary `w`-free
/// coefficients (linearity), so this is exact for any polynomial numerator.
fn integrate_poly_over_neg_quad(
    coeffs: &HashMap<usize, ExprId>,
    w: ExprId,
    s: ExprId,
    kk: &rug::Rational,
    pool: &ExprPool,
) -> ExprId {
    let maxd = coeffs.keys().copied().max().unwrap_or(0);
    let k_expr = pool.func("sqrt", vec![rational_to_expr(kk, pool)]);
    let k_inv = pool.pow(k_expr, pool.integer(-1));
    // reductions[n] = ∫ wⁿ/√(k²−w²) dw.
    let mut red: Vec<ExprId> = Vec::with_capacity(maxd + 1);
    // I₀ = asin(w/k).
    red.push(pool.func("asin", vec![pool.mul(vec![w, k_inv])]));
    if maxd >= 1 {
        // I₁ = −√(k²−w²) = −s.
        red.push(pool.mul(vec![pool.integer(-1), s]));
    }
    for nn in 2..=maxd {
        let n = nn as i64;
        // Iₙ = (−1/n)·wⁿ⁻¹·s + ((n−1)/n)·k²·Iₙ₋₂.
        let c1 = rug::Rational::from((-1, n));
        let term1 = pool.mul(vec![
            rational_to_expr(&c1, pool),
            pool.pow(w, pool.integer((nn - 1) as i32)),
            s,
        ]);
        let c2 = rug::Rational::from((n - 1, n)) * kk.clone();
        let term2 = pool.mul(vec![rational_to_expr(&c2, pool), red[nn - 2]]);
        red.push(pool.add(vec![term1, term2]));
    }
    let mut terms = Vec::with_capacity(coeffs.len());
    for (&d, &c) in coeffs {
        terms.push(pool.mul(vec![c, red[d]]));
    }
    if terms.len() == 1 {
        terms.remove(0)
    } else {
        pool.add(terms)
    }
}

/// Rewrite `expr` (rational in `x` and `√(quad)`) as a rational function of the
/// Euler parameter `t`: `x → x_of_t`, `√(quad) → sqrt_t` (and any half-integer
/// power of the radicand → the matching power of `sqrt_t`).  `None` if a subterm
/// is not expressible this way.
fn to_t(
    expr: ExprId,
    var: ExprId,
    quad: &QPoly,
    sqrt_t: ExprId,
    x_of_t: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if expr == var {
        return Some(x_of_t);
    }
    if is_free_of_var(expr, var, pool) {
        return Some(expr);
    }
    let one = vec![rug::Rational::from(1)];
    // `quad^{c/d}` (base ≡ radicand) → `sqrt_t^{2c/d}` when `d | 2c`.
    let radical_power = |base: ExprId, c: i64, d: i64, pool: &ExprPool| -> Option<ExprId> {
        if same_fraction(base, quad, &one, var, pool) && (2 * c) % d == 0 {
            Some(pool.pow(sqrt_t, pool.integer(((2 * c) / d) as i32)))
        } else {
            None
        }
    };
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            radical_power(args[0], 1, 2, pool)
        }
        ExprData::Add(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| to_t(a, var, quad, sqrt_t, x_of_t, pool))
                .collect::<Option<_>>()?;
            Some(pool.add(v))
        }
        ExprData::Mul(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| to_t(a, var, quad, sqrt_t, x_of_t, pool))
                .collect::<Option<_>>()?;
            Some(pool.mul(v))
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(m) => {
                let inner = to_t(base, var, quad, sqrt_t, x_of_t, pool)?;
                Some(pool.pow(inner, pool.integer(m.0.to_i64()? as i32)))
            }
            ExprData::Rational(r) => {
                radical_power(base, r.0.numer().to_i64()?, r.0.denom().to_i64()?, pool)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Like [`to_t`], but for the completed-square general Euler reduction where the
/// radical value is `√(quad) = k_a · sqrt_u` (`k_a = √a` an opaque constant
/// symbol, `sqrt_u = √(u²+k)` rational in `t`).  Every radicand power
/// `quad^{c/d}` becomes the **distributed** product `k_a^M · sqrt_u^M`
/// (`M = 2c/d`) — keeping `k_a` a separate factor so that, after `simplify`
/// collects the `k_a` powers into one leading constant, the integrand is a
/// genuine rational function of `t` the engine can integrate (a nested
/// `(k_a·sqrt_u)^M` would instead read as an irreducible two-variable product).
/// `None` if a subterm is not expressible this way.
fn to_t_scaled(
    expr: ExprId,
    var: ExprId,
    quad: &QPoly,
    k_a: ExprId,
    sqrt_u: ExprId,
    x_of_t: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if expr == var {
        return Some(x_of_t);
    }
    if is_free_of_var(expr, var, pool) {
        return Some(expr);
    }
    let one = vec![rug::Rational::from(1)];
    // `quad^{c/d}` (base ≡ radicand) → `k_a^{2c/d} · sqrt_u^{2c/d}` when `d | 2c`.
    let radical_power = |base: ExprId, c: i64, d: i64, pool: &ExprPool| -> Option<ExprId> {
        if same_fraction(base, quad, &one, var, pool) && (2 * c) % d == 0 {
            let m = ((2 * c) / d) as i32;
            Some(pool.mul(vec![
                pool.pow(k_a, pool.integer(m)),
                pool.pow(sqrt_u, pool.integer(m)),
            ]))
        } else {
            None
        }
    };
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            radical_power(args[0], 1, 2, pool)
        }
        ExprData::Add(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| to_t_scaled(a, var, quad, k_a, sqrt_u, x_of_t, pool))
                .collect::<Option<_>>()?;
            Some(pool.add(v))
        }
        ExprData::Mul(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| to_t_scaled(a, var, quad, k_a, sqrt_u, x_of_t, pool))
                .collect::<Option<_>>()?;
            Some(pool.mul(v))
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(m) => {
                let inner = to_t_scaled(base, var, quad, k_a, sqrt_u, x_of_t, pool)?;
                // Distribute the outer integer power over any `k_a^M · sqrt_u^M`
                // product `radical_power` produced, so `√(quad)^{-1}` becomes
                // `k_a^{-1} · sqrt_u^{-1}` (separate factors) rather than
                // `(k_a·sqrt_u)^{-1}` (which reads as a two-variable inverse).
                Some(pow_int_distribute(inner, m.0.to_i64()? as i32, pool))
            }
            ExprData::Rational(r) => {
                radical_power(base, r.0.numer().to_i64()?, r.0.denom().to_i64()?, pool)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Integrate `∫ expr dt` where `expr` is a sum of terms, each a product of `t`-free
/// constants (notably powers of the opaque `k_a = √a`) times a rational function
/// of `t`.  Each term's constant part is pulled out, its `t`-part is collapsed to a
/// single `N(t)/D(t)` (so the engine's Rothstein–Trager rational integrator — which
/// normalizes over ℚ(t) and would otherwise be defeated by the extra `k_a`
/// indeterminate — sees a pure rational function), integrated, and the constant
/// multiplied back.  `None` if any term's `t`-part is not rational in `t` or the
/// engine cannot integrate it.
fn integrate_scaled_rational(expr: ExprId, t: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let terms: Vec<ExprId> = match pool.get(expr) {
        ExprData::Add(args) => args.clone(),
        _ => vec![expr],
    };
    let mut pieces = Vec::with_capacity(terms.len());
    for term in terms {
        let factors: Vec<ExprId> = match pool.get(term) {
            ExprData::Mul(args) => args.clone(),
            _ => vec![term],
        };
        let (consts, tdep): (Vec<ExprId>, Vec<ExprId>) = factors
            .into_iter()
            .partition(|&f| is_free_of_var(f, t, pool));
        let int_tpart = if tdep.is_empty() {
            // ∫ (constant) dt = constant · t.
            t
        } else {
            let tpart = if tdep.len() == 1 {
                tdep[0]
            } else {
                pool.mul(tdep)
            };
            let (num, den) = expr_to_qrational(tpart, t, pool)?;
            let frac = pool.mul(vec![
                qpoly_to_expr(&trim(num), t, pool),
                pool.pow(qpoly_to_expr(&trim(den), t, pool), pool.integer(-1)),
            ]);
            crate::integrate::engine::integrate(frac, t, pool)
                .ok()?
                .value
        };
        let mut all = consts;
        all.push(int_tpart);
        pieces.push(pool.mul(all));
    }
    Some(if pieces.len() == 1 {
        pieces.remove(0)
    } else {
        pool.add(pieces)
    })
}

/// Raise `base` to the integer power `m`, distributing over `Mul` factors and
/// folding into inner integer `Pow` exponents, so a product never ends up buried
/// inside a single `(…)^m` (which the rational engine treats as one opaque
/// var-dependent factor).  Used only by [`to_t_scaled`] on the small
/// `k_a^M · sqrt_u^M` shapes it builds.
fn pow_int_distribute(base: ExprId, m: i32, pool: &ExprPool) -> ExprId {
    if m == 1 {
        return base;
    }
    match pool.get(base) {
        ExprData::Mul(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&f| pow_int_distribute(f, m, pool))
                .collect();
            pool.mul(v)
        }
        ExprData::Pow { base: b, exp } => {
            if let ExprData::Integer(e) = pool.get(exp) {
                if let Some(ei) = e.0.to_i64() {
                    return pool.pow(b, pool.integer(ei as i32 * m));
                }
            }
            pool.pow(base, pool.integer(m))
        }
        _ => pool.pow(base, pool.integer(m)),
    }
}

/// A rational square root of `v ≥ 0` (numerator and denominator both perfect
/// squares), else `None`.
fn sqrt_rational(v: &rug::Rational) -> Option<rug::Rational> {
    if *v < 0 {
        return None;
    }
    if *v == 0 {
        return Some(rug::Rational::from(0));
    }
    let nn = v.numer().clone();
    let dd = v.denom().clone();
    let ns = nn.clone().sqrt();
    let ds = dd.clone().sqrt();
    if rug::Integer::from(&ns * &ns) == nn && rug::Integer::from(&ds * &ds) == dd {
        Some(rug::Rational::from((ns, ds)))
    } else {
        None
    }
}

/// Find the unique `x`-dependent radical generator and return `(n, radicand)`.
/// `None` if there is no such generator or more than one distinct one.
fn detect_single_radical(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(usize, ExprId)> {
    let mut found: Vec<(usize, ExprId)> = Vec::new();
    scan(expr, var, pool, &mut found);
    let mut distinct: Vec<(usize, ExprId)> = Vec::new();
    for (n, r) in found {
        if !distinct.iter().any(|&(m, q)| m == n && q == r) {
            distinct.push((n, r));
        }
    }
    if distinct.len() == 1 {
        Some(distinct.remove(0))
    } else {
        None
    }
}

fn scan(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<(usize, ExprId)>) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args }
            if name == "sqrt" && args.len() == 1 && !is_free_of_var(args[0], var, pool) =>
        {
            out.push((2, args[0]));
        }
        ExprData::Func { ref name, ref args }
            if name == "cbrt" && args.len() == 1 && !is_free_of_var(args[0], var, pool) =>
        {
            out.push((3, args[0]));
        }
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if let Some(den) = r.0.denom().to_i64() {
                    if den >= 2 && !is_free_of_var(base, var, pool) {
                        out.push((den as usize, base));
                        return;
                    }
                }
            }
            scan(base, var, pool, out);
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                scan(a, var, pool, out);
            }
        }
        _ => {}
    }
}

/// Rewrite `expr` (rational in `x` and the single radical `r(x)^{1/n}`, `r =
/// num/den`) as a rational function of `s`, where `s = r(x)^{1/n}`: standalone
/// `x → x(s)`, and any power `r(x)^{c/d}` of the radicand → `s^{c·n/d}`.  Returns
/// `None` if a subterm is not expressible this way (a different radical, a
/// transcendental of `x`, or a fractional power with `d ∤ c·n`).
#[allow(clippy::too_many_arguments)]
fn to_s(
    expr: ExprId,
    var: ExprId,
    num: &QPoly,
    den: &QPoly,
    n: usize,
    s: ExprId,
    x_of_s: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if expr == var {
        return Some(x_of_s);
    }
    if is_free_of_var(expr, var, pool) {
        return Some(expr); // constant in x (incl. other symbols / numbers)
    }
    // `r(x)^{c/d}` → `s^{c·n/d}` when `base = r` (as a fraction) and the exponent
    // is an integer.
    let radical_power = |base: ExprId, c: i64, d: i64, pool: &ExprPool| -> Option<ExprId> {
        if same_fraction(base, num, den, var, pool) && (c * n as i64) % d == 0 {
            Some(pool.pow(s, pool.integer(((c * n as i64) / d) as i32)))
        } else {
            None
        }
    };
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            radical_power(args[0], 1, 2, pool)
        }
        ExprData::Func { ref name, ref args } if name == "cbrt" && args.len() == 1 => {
            radical_power(args[0], 1, 3, pool)
        }
        ExprData::Add(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| to_s(a, var, num, den, n, s, x_of_s, pool))
                .collect::<Option<_>>()?;
            Some(pool.add(v))
        }
        ExprData::Mul(args) => {
            let v: Vec<ExprId> = args
                .iter()
                .map(|&a| to_s(a, var, num, den, n, s, x_of_s, pool))
                .collect::<Option<_>>()?;
            Some(pool.mul(v))
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(m) => {
                let inner = to_s(base, var, num, den, n, s, x_of_s, pool)?;
                Some(pool.pow(inner, pool.integer(m.0.to_i64()? as i32)))
            }
            ExprData::Rational(r) => {
                radical_power(base, r.0.numer().to_i64()?, r.0.denom().to_i64()?, pool)
            }
            _ => None,
        },
        _ => None,
    }
}

/// Is `base` equal, as a rational function of `x`, to the fraction `num/den`?
/// Tested by cross-multiplication so unequal scalings are *not* matched.
fn same_fraction(base: ExprId, num: &QPoly, den: &QPoly, var: ExprId, pool: &ExprPool) -> bool {
    let Some((bn, bd)) = expr_to_qrational(base, var, pool) else {
        return false;
    };
    // base nontrivial in x (so it really is the radicand, not a constant).
    if degree(&trim(bn.clone())) < 1 && degree(&trim(bd.clone())) < 1 {
        return false;
    }
    trim(poly_mul(&bn, den)) == trim(poly_mul(num, &bd))
}

// ---------------------------------------------------------------------------
// The gate: `d/dx F = integrand` on the region where the radical is real
// ---------------------------------------------------------------------------
//
// The old check screened a **fixed** grid `[0.3, 0.8, 1.6, 2.7, 3.9, 5.1]` and
// required two survivors.  That grid is not a property of the integrand: it
// presumes the radical is real somewhere in `(0, 5.1]`, which silently excludes
// every *bounded* radicand.  On `√(1−2x²)` (real only for `|x| < 0.707`) exactly
// one grid point survives, so the floor rejected **correct** antiderivatives —
// Charlwood #49's residual `x/((1−x²)√(1−2x²))` among them.  A domain-blind
// sampler is a soundness knob pointed the wrong way: it cannot accept a wrong
// answer, but it rejects right ones for a reason that has nothing to do with
// them.
//
// The grid is now derived from the integrand.  [`discover_domain`] scans
// `[−h, h]` (widening `h` when nothing is in range), keeps the abscissae where
// the radicand is safely positive *and* the integrand is finite and away from a
// pole, groups them into maximal runs, and spreads the samples across the widest
// runs — so a candidate that is right on only part of its domain is still
// caught.  The runs double as the closed boxes for the enclosure tier.

/// Half-widths tried, in order, until one yields enough in-domain probes.
///
/// The first covers everything ordinary.  The second exists because a radicand
/// can be real only *far* from the origin — `√(x²−100)` needs `|x| > 10` — and a
/// scan that stops at `6` would decline a correct answer for want of anywhere to
/// look, which is the same mistake as the fixed grid this replaced, one order of
/// magnitude out.
const SCAN_HALF_WIDTHS: [f64; 2] = [6.0, 60.0];
/// How many probes each scan uses across `[−half_width, half_width]`.
const SCAN_N: usize = 481;
/// Minimum radicand value for a probe to count as inside the real branch.
const RADICAND_FLOOR: f64 = 1e-3;
/// Reject a probe whose integrand magnitude exceeds this (a pole is near).
const POLE_CEILING: f64 = 1e6;

/// Sample points and enclosure boxes for the region where `√radicand` is real
/// and `integrand` is finite.
struct EulerDomain {
    samples: Vec<f64>,
    boxes: Vec<(f64, f64)>,
}

/// Rewrite `cbrt(u)` as `u^(1/3)` for the numeric tiers only.
///
/// [`crate::integrate::gate::eval_at`] dispatches function heads through the
/// [`crate::primitive::PrimitiveRegistry`], and `cbrt` is **not registered** —
/// so a candidate or integrand mentioning it evaluates to `None` at every point
/// and the gate declines for a reason that has nothing to do with the
/// mathematics.  (This route's radicands include `x^{1/3}`, so that is not
/// hypothetical: it silently declined the whole cube-root family.)
///
/// `u^(1/3)` evaluates through `powf`, which is exact for `u > 0` — and every
/// sample point this module offers already satisfies `radicand > 0`.  For `u < 0`
/// `powf` yields `NaN` where `cbrt` would not, which only makes the gate *skip*
/// the point.  The rewrite is applied to the gate's copies alone; the emitted
/// antiderivative keeps its `cbrt`.
fn for_gate(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "cbrt" && args.len() == 1 => {
            let inner = for_gate(args[0], pool);
            pool.pow(inner, pool.rational(1_i32, 3_i32))
        }
        ExprData::Func { ref name, ref args } => {
            let out: Vec<ExprId> = args.iter().map(|&a| for_gate(a, pool)).collect();
            pool.func(name, out)
        }
        ExprData::Add(args) => pool.add(args.iter().map(|&a| for_gate(a, pool)).collect()),
        ExprData::Mul(args) => pool.mul(args.iter().map(|&a| for_gate(a, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(for_gate(base, pool), for_gate(exp, pool)),
        _ => expr,
    }
}

fn discover_domain(
    integrand: ExprId,
    radicand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> EulerDomain {
    let admissible = |xv: f64| -> bool {
        match crate::integrate::gate::eval_at(radicand, var, xv, pool) {
            Some(r) if r.is_finite() && r > RADICAND_FLOOR => {}
            _ => return false,
        }
        match crate::integrate::gate::eval_at(integrand, var, xv, pool) {
            Some(v) => v.is_finite() && v.abs() < POLE_CEILING,
            None => false,
        }
    };

    let mut best = EulerDomain {
        samples: Vec::new(),
        boxes: Vec::new(),
    };
    for half_width in SCAN_HALF_WIDTHS {
        let step = 2.0 * half_width / (SCAN_N - 1) as f64;
        let mut runs: Vec<Vec<f64>> = Vec::new();
        let mut cur: Vec<f64> = Vec::new();
        for i in 0..SCAN_N {
            let xv = -half_width + step * i as f64;
            if admissible(xv) {
                cur.push(xv);
            } else if !cur.is_empty() {
                runs.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            runs.push(cur);
        }

        // Widest runs first, so a broad principal branch is preferred to a sliver.
        runs.sort_by_key(|r| std::cmp::Reverse(r.len()));
        let mut samples: Vec<f64> = Vec::new();
        let mut boxes = Vec::new();
        for run in runs.iter().take(3) {
            if run.len() < 3 {
                continue;
            }
            // Four points spread across the run, avoiding both endpoints (where
            // the radicand vanishes, or a pole sits just outside).
            for k in 1..=4 {
                let idx = (run.len() * k / 5).min(run.len() - 1);
                let xv = run[idx];
                if !samples.iter().any(|&s| (s - xv).abs() < 1e-9) {
                    samples.push(xv);
                }
            }
            // A closed box strictly inside the run, for the rigorous tier.
            let lo = run[run.len() / 6];
            let hi = run[run.len() - 1 - run.len() / 6];
            if hi - lo > 4.0 * step {
                boxes.push((lo, hi));
            }
        }
        if samples.len() > best.samples.len() {
            best = EulerDomain { samples, boxes };
        }
        if best.samples.len() >= 4 {
            break; // the near scan already found plenty; do not widen
        }
    }
    best
}

/// Gate a genus-0 candidate through [`crate::integrate::gate`]:
/// `d/dx F = integrand` on the region where the radical is real, graded
/// `Proven` / `EnclosureVerified` / `SampledOnly`.
///
/// Returns `false` for every non-passing verdict — a declining gate makes this
/// route decline, never emit and never certify non-elementarity.
fn verify_derivative(
    f: ExprId,
    integrand: ExprId,
    radicand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> bool {
    gate_verdict(f, integrand, radicand, var, pool).is_verified()
}

/// The graded verdict behind [`verify_derivative`], kept separate so tests can
/// assert *which* tier a route reached.
fn gate_verdict(
    f: ExprId,
    integrand: ExprId,
    radicand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> crate::integrate::gate::Verdict {
    use crate::integrate::gate::{self, EnclosureBudget, EnclosurePolicy, GateOptions, Target};

    let too_few = |found: usize| gate::Verdict::Declined {
        reason: gate::DeclineReason::NotEnoughPoints { found, required: 2 },
    };

    // A leftover substitution symbol (`$euler_t$`, `$euler_sqrt_a$`) means the
    // back-substitution did not reach every occurrence — `kernel::subs` does not
    // descend into a `RootSum` binder, so an unresolved algebraic-residue log
    // keeps the internal variable.  The gate would merely *skip* such points
    // (`eval_at` returns `None` on an unbound symbol); refuse outright instead,
    // because an expression still mentioning an internal symbol is never
    // emittable whatever the samples say.
    if mentions_internal_symbol(f, pool) {
        return too_few(0);
    }

    let (gf, gi, gr) = (
        for_gate(f, pool),
        for_gate(integrand, pool),
        for_gate(radicand, pool),
    );
    let dom = discover_domain(gi, gr, var, pool);
    if dom.samples.len() < 2 {
        return too_few(dom.samples.len());
    }
    let in_domain = move |xv: f64| {
        matches!(
            gate::eval_at(gr, var, xv, pool),
            Some(r) if r.is_finite() && r > RADICAND_FLOOR
        ) && matches!(
            gate::eval_at(gi, var, xv, pool),
            Some(v) if v.is_finite() && v.abs() < POLE_CEILING
        )
    };
    let domain = gate::Domain::from_samples(dom.samples)
        .with_predicate(in_domain)
        .with_boxes(dom.boxes);

    let opts = GateOptions {
        tolerance: 1e-7,
        min_points: 2,
        symbolic: true,
        egraph: false,
        // Additive: a failed enclosure keeps the sampled verdict, so the
        // rigorous tier can only ever *raise* the grade.
        enclosure: EnclosurePolicy::BestEffort(EnclosureBudget::cheap()),
        min_strength: gate::Strength::Sampled,
    };
    gate::verify(gf, &Target::symbolic(gi), var, &domain, &opts, pool)
}

/// Does `expr` still mention one of this module's internal substitution symbols?
///
/// Unlike [`super::poly_utils::is_free_of_subexpr`] this descends into the
/// `RootSum` binder, which is exactly where a missed back-substitution hides.
fn mentions_internal_symbol(expr: ExprId, pool: &ExprPool) -> bool {
    /// `Err(())` marks "found one" so the walk short-circuits.
    fn walk(e: ExprId, pool: &ExprPool, seen: &mut std::collections::HashSet<ExprId>) -> bool {
        if !seen.insert(e) {
            return false;
        }
        let mut hit = false;
        let kids: Vec<ExprId> = pool.with(e, |data| match data {
            ExprData::Symbol { name, .. } => {
                hit = name.starts_with("$euler") || name.starts_with("$param");
                vec![]
            }
            ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => args.clone(),
            ExprData::Pow { base, exp } => vec![*base, *exp],
            ExprData::RootSum { poly, var, body } => vec![*poly, *var, *body],
            _ => vec![],
        });
        hit || kids.iter().any(|&c| walk(c, pool, seen))
    }
    walk(expr, pool, &mut std::collections::HashSet::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(build: impl Fn(&ExprPool, ExprId) -> ExprId) {
        check_at(build, &[1.3, 2.4, 3.7]);
    }

    fn check_at(build: impl Fn(&ExprPool, ExprId) -> ExprId, samples: &[f64]) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = build(&pool, x);
        let r = crate::integrate::engine::integrate(f, x, &pool);
        assert!(r.is_ok(), "expected elementary; got {r:?}");
        // d/dx F = f at samples (chosen inside the radicand-positive domain).
        let g = r.unwrap().value;
        let ds = simplify(crate::diff::diff(g, x, &pool).unwrap().value, &pool).value;
        for &xv in samples {
            // `for_gate` so `cbrt` evaluates: it has no registry entry.
            let lhs = crate::integrate::gate::eval_at(for_gate(ds, &pool), x, xv, &pool).unwrap();
            let rhs = crate::integrate::gate::eval_at(for_gate(f, &pool), x, xv, &pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, f = {rhs}\n  F = {}",
                pool.display(g)
            );
        }
    }

    /// MC0 (Möbius radicand): `∫ √((1−x)/(1+x)) dx` — genus-0 via `x = (1−s²)/(1+s²)`.
    /// Radicand positive on `(−1, 1)`.
    #[test]
    fn sqrt_mobius_one_minus_x_over_one_plus_x() {
        check_at(
            |p, x| {
                let num = p.add(vec![p.integer(1), p.mul(vec![p.integer(-1), x])]);
                let den = p.add(vec![p.integer(1), x]);
                let ratio = p.mul(vec![num, p.pow(den, p.integer(-1))]);
                p.func("sqrt", vec![ratio])
            },
            &[0.2, 0.55, 0.85],
        );
    }

    /// MC0: `∫ ∛((x+1)/(x−1)) dx` — radicand positive for `x > 1`.
    #[test]
    fn cbrt_mobius_x_plus_1_over_x_minus_1() {
        check_at(
            |p, x| {
                let num = p.add(vec![x, p.integer(1)]);
                let den = p.add(vec![x, p.integer(-1)]);
                let ratio = p.mul(vec![num, p.pow(den, p.integer(-1))]);
                p.func("cbrt", vec![ratio])
            },
            &[1.7, 2.6, 4.3],
        );
    }

    /// MC0: `∫ 1/((1+x)·√((1−x)/(1+x))) dx` — a rational weight times the Möbius
    /// radical.  Radicand positive on `(−1, 1)`.
    #[test]
    fn weighted_sqrt_mobius() {
        check_at(
            |p, x| {
                let num = p.add(vec![p.integer(1), p.mul(vec![p.integer(-1), x])]);
                let den = p.add(vec![p.integer(1), x]);
                let ratio = p.mul(vec![num, p.pow(den, p.integer(-1))]);
                let rad = p.func("sqrt", vec![ratio]);
                let w = p.pow(p.add(vec![p.integer(1), x]), p.integer(-1));
                p.mul(vec![w, p.pow(rad, p.integer(-1))])
            },
            &[0.2, 0.55, 0.85],
        );
    }

    #[test]
    fn cbrt_x_over_x_plus_1() {
        // ∫ ∛x/(x+1) dx — elementary (was wrongly NonElementary before M2).
        check(|p, x| {
            let num = p.func("cbrt", vec![x]);
            let den = p.add(vec![x, p.integer(1)]);
            p.mul(vec![num, p.pow(den, p.integer(-1))])
        });
    }

    #[test]
    fn one_over_x_cbrt_x_plus_1() {
        // ∫ 1/(x·∛(x+1)) dx.
        check(|p, x| {
            let xp1 = p.add(vec![x, p.integer(1)]);
            let cb = p.func("cbrt", vec![xp1]);
            p.pow(p.mul(vec![x, cb]), p.integer(-1))
        });
    }

    #[test]
    fn cbrt_x_over_one_plus_cbrt_x() {
        // ∫ ∛x/(1+∛x) dx.
        check(|p, x| {
            let cb = p.func("cbrt", vec![x]);
            let den = p.add(vec![p.integer(1), cb]);
            p.mul(vec![cb, p.pow(den, p.integer(-1))])
        });
    }

    #[test]
    fn x_two_thirds() {
        // ∫ x^(2/3) dx = (3/5) x^(5/3).
        check(|p, x| p.pow(x, p.rational(2, 3)));
    }

    #[test]
    fn fifth_root_of_linear() {
        // ∫ (2x+1)^(1/5) dx = (5/12)(2x+1)^(6/5).
        check(|p, x| {
            let lin = p.add(vec![p.mul(vec![p.integer(2), x]), p.integer(1)]);
            p.pow(lin, p.rational(1, 5))
        });
    }

    /// Euler (a=1 square): `∫ dx/((x²−1)·√(x²+1))` — a *rational* coefficient on a
    /// quadratic radical (the deg-2 sqrt engine handles only polynomial weights).
    /// Elementary; radicand positive everywhere, avoid the poles `x=±1`.
    #[test]
    fn euler_rational_weight_quadratic() {
        check_at(
            |p, x| {
                let q = p.add(vec![p.pow(x, p.integer(2)), p.integer(1)]);
                let d = p.add(vec![p.pow(x, p.integer(2)), p.integer(-1)]);
                p.mul(vec![
                    p.pow(d, p.integer(-1)),
                    p.pow(p.func("sqrt", vec![q]), p.integer(-1)),
                ])
            },
            &[0.3, 1.7, 2.6],
        );
    }

    /// Euler (a=1 square): `∫ dx/(x·√(x²+1))` = `log((√(x²+1)−1)/x)`-type.
    #[test]
    fn euler_one_over_x_sqrt_quadratic() {
        check_at(
            |p, x| {
                let q = p.add(vec![p.pow(x, p.integer(2)), p.integer(1)]);
                p.mul(vec![
                    p.pow(x, p.integer(-1)),
                    p.pow(p.func("sqrt", vec![q]), p.integer(-1)),
                ])
            },
            &[0.6, 1.4, 3.1],
        );
    }

    /// Euler (a=1 square, c=−1 not a square): `∫ √(x²−1)/x dx`.  Radicand positive
    /// for `x > 1`.
    #[test]
    fn euler_sqrt_quadratic_over_x() {
        check_at(
            |p, x| {
                let q = p.add(vec![p.pow(x, p.integer(2)), p.integer(-1)]);
                p.mul(vec![p.func("sqrt", vec![q]), p.pow(x, p.integer(-1))])
            },
            &[1.4, 2.5, 3.8],
        );
    }

    /// General Euler (`a=2` not a square, `c=3` not a square): `∫ x/√(2x²+3) dx =
    /// √(2x²+3)/2`.  Completed-square reduction; radicand positive everywhere.
    #[test]
    fn euler_general_x_over_sqrt_2x2_plus_3() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.mul(vec![p.integer(2), p.pow(x, p.integer(2))]),
                    p.integer(3),
                ]);
                p.mul(vec![x, p.pow(p.func("sqrt", vec![q]), p.integer(-1))])
            },
            &[-1.5, 0.4, 1.7, 3.1],
        );
    }

    /// General Euler with a linear term (`a=3`, `b=2`, `c=2`, discriminant < 0):
    /// `∫ 1/√(3x²+2x+2) dx` — an `asinh`/`log` form.  Radicand positive everywhere.
    #[test]
    fn euler_general_one_over_sqrt_3x2_2x_2() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.mul(vec![p.integer(3), p.pow(x, p.integer(2))]),
                    p.mul(vec![p.integer(2), x]),
                    p.integer(2),
                ]);
                p.pow(p.func("sqrt", vec![q]), p.integer(-1))
            },
            &[-2.0, -0.3, 1.1, 2.6],
        );
    }

    /// General Euler with a rational weight: `∫ 1/((x−1)·√(2x²+3)) dx` — an
    /// elementary `log` form.  Radicand positive everywhere; avoid the pole `x=1`.
    #[test]
    fn euler_general_weighted_1_over_x_minus_1_sqrt_2x2_3() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.mul(vec![p.integer(2), p.pow(x, p.integer(2))]),
                    p.integer(3),
                ]);
                let w = p.pow(p.add(vec![x, p.integer(-1)]), p.integer(-1));
                p.mul(vec![w, p.pow(p.func("sqrt", vec![q]), p.integer(-1))])
            },
            &[-1.5, 0.2, 2.3, 3.7],
        );
    }

    /// Regression: `∫ √(2x²+3) dx` — already worked via the polynomial-`B`
    /// integral part; the new general fallback must not disturb it.
    #[test]
    fn regression_sqrt_2x2_plus_3() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.mul(vec![p.integer(2), p.pow(x, p.integer(2))]),
                    p.integer(3),
                ]);
                p.func("sqrt", vec![q])
            },
            &[-1.5, 0.4, 1.7, 3.1],
        );
    }

    /// Regression: `∫ 1/√(2x²+3) dx` — already worked; keep it working.
    #[test]
    fn regression_one_over_sqrt_2x2_plus_3() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.mul(vec![p.integer(2), p.pow(x, p.integer(2))]),
                    p.integer(3),
                ]);
                p.pow(p.func("sqrt", vec![q]), p.integer(-1))
            },
            &[-1.5, 0.4, 1.7, 3.1],
        );
    }

    /// Regression: `∫ x·√(x²+1) dx = (x²+1)^{3/2}/3` — the polynomial-`B` integral
    /// part's nicer closed form must be preserved (a=1 square, not routed here).
    #[test]
    fn regression_x_sqrt_x2_plus_1() {
        check_at(
            |p, x| {
                let q = p.add(vec![p.pow(x, p.integer(2)), p.integer(1)]);
                p.mul(vec![x, p.func("sqrt", vec![q])])
            },
            &[-1.3, 0.4, 1.7, 3.1],
        );
    }

    /// Regression: `∫ dx/((x²−1)·√(x²+1))` — the existing `a=1`-square Euler path
    /// (rational weight on a quadratic radical) must keep working.  Avoid `x=±1`.
    #[test]
    fn regression_dx_over_x2_minus_1_sqrt_x2_plus_1() {
        check_at(
            |p, x| {
                let q = p.add(vec![p.pow(x, p.integer(2)), p.integer(1)]);
                let d = p.add(vec![p.pow(x, p.integer(2)), p.integer(-1)]);
                p.mul(vec![
                    p.pow(d, p.integer(-1)),
                    p.pow(p.func("sqrt", vec![q]), p.integer(-1)),
                ])
            },
            &[0.3, 1.7, 2.6, -2.0],
        );
    }

    /// Arcsin family (a<0, non-square leading coeff): `∫ 1/√(2−3x²) dx =
    /// (1/√3)·asin(x·√(3/2))`.  Radicand positive on `|x| < √(2/3) ≈ 0.816`.
    #[test]
    fn arcsin_one_over_sqrt_2_minus_3x2() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.integer(2),
                    p.mul(vec![p.integer(-3), p.pow(x, p.integer(2))]),
                ]);
                p.pow(p.func("sqrt", vec![q]), p.integer(-1))
            },
            &[-0.6, -0.1, 0.3, 0.7],
        );
    }

    /// Arcsin family: `∫ 1/√(5−2x²) dx`.  Radicand positive on `|x| < √(5/2) ≈ 1.58`.
    #[test]
    fn arcsin_one_over_sqrt_5_minus_2x2() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.integer(5),
                    p.mul(vec![p.integer(-2), p.pow(x, p.integer(2))]),
                ]);
                p.pow(p.func("sqrt", vec![q]), p.integer(-1))
            },
            &[-1.2, -0.4, 0.5, 1.3],
        );
    }

    /// Arcsin family with a linear term: `∫ 1/√(−x²+2x+3) dx = asin((x−1)/2)`
    /// (complete square: `4 − (x−1)²`).  Radicand positive on `(−1, 3)`.
    #[test]
    fn arcsin_one_over_sqrt_neg_x2_2x_3() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.mul(vec![p.integer(-1), p.pow(x, p.integer(2))]),
                    p.mul(vec![p.integer(2), x]),
                    p.integer(3),
                ]);
                p.pow(p.func("sqrt", vec![q]), p.integer(-1))
            },
            &[-0.5, 0.7, 1.6, 2.5],
        );
    }

    /// Arcsin family, linear numerator: `∫ x/√(5−2x²) dx = −√(5−2x²)/2`.
    /// Radicand positive on `|x| < √(5/2)`.
    #[test]
    fn arcsin_x_over_sqrt_5_minus_2x2() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.integer(5),
                    p.mul(vec![p.integer(-2), p.pow(x, p.integer(2))]),
                ]);
                p.mul(vec![x, p.pow(p.func("sqrt", vec![q]), p.integer(-1))])
            },
            &[-1.2, -0.4, 0.5, 1.3],
        );
    }

    /// Arcsin family, `√P` numerator (the improved-form bonus): `∫ √(2−3x²) dx =
    /// x√(2−3x²)/2 + (1/√3)·asin(x√(3/2))` — a *real* asin form (no `sqrt(-3)`).
    #[test]
    fn arcsin_sqrt_2_minus_3x2() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.integer(2),
                    p.mul(vec![p.integer(-3), p.pow(x, p.integer(2))]),
                ]);
                p.func("sqrt", vec![q])
            },
            &[-0.6, -0.1, 0.3, 0.7],
        );
    }

    /// Regression: `∫ 1/√(1−x²) dx = asin(x)`.  Radicand positive on `(−1, 1)`.
    #[test]
    fn arcsin_one_over_sqrt_1_minus_x2() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.integer(1),
                    p.mul(vec![p.integer(-1), p.pow(x, p.integer(2))]),
                ]);
                p.pow(p.func("sqrt", vec![q]), p.integer(-1))
            },
            &[-0.7, -0.2, 0.3, 0.8],
        );
    }

    /// Regression: `∫ 1/√(4−x²) dx = asin(x/2)`.  Radicand positive on `(−2, 2)`.
    #[test]
    fn arcsin_one_over_sqrt_4_minus_x2() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.integer(4),
                    p.mul(vec![p.integer(-1), p.pow(x, p.integer(2))]),
                ]);
                p.pow(p.func("sqrt", vec![q]), p.integer(-1))
            },
            &[-1.5, -0.4, 0.6, 1.7],
        );
    }

    /// `√(−1−x²)` is negative everywhere (`k² < 0`): the arcsin path declines
    /// cleanly (no panic).
    #[test]
    fn arcsin_negative_everywhere_declines() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let q = pool.add(vec![
            pool.integer(-1),
            pool.mul(vec![pool.integer(-1), pool.pow(x, pool.integer(2))]),
        ]);
        let f = pool.pow(pool.func("sqrt", vec![q]), pool.integer(-1));
        // Must not panic; declining (Err) is acceptable — just never a wrong answer.
        let _ = crate::integrate::engine::integrate(f, x, &pool);
    }

    // -----------------------------------------------------------------------
    // Charlwood cluster C1: `∫B(x)·√(quadratic)` with a *rational* `B`
    //
    // The four integrands below are the by-parts residuals of Charlwood #29,
    // #30, #47 and #49 — the shape the integration-by-parts reduction hands to
    // the algebraic engine.  All four are genus-0 and elementary; each used to
    // decline.  They are checked by differentiation only: Alkahest's forms
    // legitimately differ from the published optimal antiderivatives.
    // -----------------------------------------------------------------------

    /// Charlwood #29 residual `∫ √(1−x²)/(x(1+x²)) dx` — a rational weight with
    /// a pole at `x = 0` *and* an algebraic-residue log part.
    #[test]
    fn charlwood29_residual() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.integer(1),
                    p.mul(vec![p.integer(-1), p.pow(x, p.integer(2))]),
                ]);
                let d = p.mul(vec![x, p.add(vec![p.integer(1), p.pow(x, p.integer(2))])]);
                p.mul(vec![p.func("sqrt", vec![q]), p.pow(d, p.integer(-1))])
            },
            &[0.3, 0.5, 0.77],
        );
    }

    /// Charlwood #8 residual `∫ √(t²+2t+2)/(1+t²) dt`, which the `sec/√(sec⁴−1)`
    /// generator substitution reduces to.  It declined with the same
    /// `"B = (1+t²)^-1 not handled"` message as #29/#30/#47/#49 — the same
    /// rational-`B` gap, reached from the transcendental side.
    ///
    /// Sampled on **both** sides of the vertex: `t²+2t+2 = (t+1)²+1` is positive
    /// everywhere, so the antiderivative has to hold on all of ℝ, not just where
    /// the old grid looked.
    #[test]
    fn charlwood8_residual() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.pow(x, p.integer(2)),
                    p.mul(vec![p.integer(2), x]),
                    p.integer(2),
                ]);
                let d = p.add(vec![p.integer(1), p.pow(x, p.integer(2))]);
                p.mul(vec![p.func("sqrt", vec![q]), p.pow(d, p.integer(-1))])
            },
            &[-2.5, -1.3, -0.4, 0.4, 1.2, 2.4],
        );
    }

    /// Charlwood #30 residual `∫ √(1−x²)/(1+x²) dx`.  The Euler reduction's
    /// `t`-integral has residues `±i/√2`, so this only closes once the `RootSum`
    /// becomes a real `atan` (`rootsum_expand`).
    #[test]
    fn charlwood30_residual() {
        check_at(
            |p, x| {
                let q = p.add(vec![
                    p.integer(1),
                    p.mul(vec![p.integer(-1), p.pow(x, p.integer(2))]),
                ]);
                let d = p.add(vec![p.integer(1), p.pow(x, p.integer(2))]);
                p.mul(vec![p.func("sqrt", vec![q]), p.pow(d, p.integer(-1))])
            },
            &[0.3, 0.5, 0.77],
        );
    }

    /// Charlwood #47 residual `∫ x(1+2x²)/(√(1+x²)(1+x²+x⁴)) dx`.  Its
    /// Rothstein–Trager resultant is the **biquadratic** `r⁴ − r²/4 + 1/16`,
    /// whose roots `±√3/4 ± i/4` give the `√3`-log plus `atan` answer.
    #[test]
    fn charlwood47_residual() {
        check_at(
            |p, x| {
                let x2 = p.pow(x, p.integer(2));
                let q = p.add(vec![p.integer(1), x2]);
                let num = p.mul(vec![
                    x,
                    p.add(vec![p.integer(1), p.mul(vec![p.integer(2), x2])]),
                ]);
                let den = p.add(vec![p.integer(1), x2, p.pow(x, p.integer(4))]);
                p.mul(vec![
                    num,
                    p.pow(p.func("sqrt", vec![q]), p.integer(-1)),
                    p.pow(den, p.integer(-1)),
                ])
            },
            &[0.3, 0.9, 1.7],
        );
    }

    /// Charlwood #49 residual `∫ x/((1−x²)√(1−2x²)) dx`.  The radical is real
    /// only on `|x| < 1/√2`, which the old fixed sample grid could not cover —
    /// the reduction was already right and the *gate* rejected it.
    #[test]
    fn charlwood49_residual() {
        check_at(
            |p, x| {
                let x2 = p.pow(x, p.integer(2));
                let q = p.add(vec![p.integer(1), p.mul(vec![p.integer(-2), x2])]);
                let d = p.add(vec![p.integer(1), p.mul(vec![p.integer(-1), x2])]);
                p.mul(vec![
                    x,
                    p.pow(d, p.integer(-1)),
                    p.pow(p.func("sqrt", vec![q]), p.integer(-1)),
                ])
            },
            &[0.2, 0.35, 0.6],
        );
    }

    /// A bounded radicand whose conic has **no** rational point at infinity and
    /// none at `x = 0` (`a = −1`, `c = 2`, neither a square) — but does have
    /// `(1, 1)`.  This is the [`try_euler_conic_point`] route.
    #[test]
    fn conic_point_route_closes_sqrt_2_minus_x2_over_1_plus_x2() {
        check_at(
            |p, x| {
                let x2 = p.pow(x, p.integer(2));
                let q = p.add(vec![p.integer(2), p.mul(vec![p.integer(-1), x2])]);
                let d = p.add(vec![p.integer(1), x2]);
                p.mul(vec![p.func("sqrt", vec![q]), p.pow(d, p.integer(-1))])
            },
            &[0.3, 0.8, 1.2],
        );
    }

    /// `∫ x/√(x²−1) dx` must come back as **`√(x²−1)`**, not as the Euler
    /// parameter's `½(x+√(x²−1)) − ½(x+√(x²−1))⁻¹`.
    ///
    /// This is an exact structural assertion, not a numeric one: the shape is
    /// the point.  A by-parts step that receives this as its `v` has to
    /// differentiate and re-integrate it, and the un-collapsed form is what
    /// makes that fail.
    #[test]
    fn euler_answer_collapses_to_the_radical() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let q = pool.add(vec![pool.pow(x, pool.integer(2)), pool.integer(-1)]);
        let sq = pool.func("sqrt", vec![q]);
        let f = pool.mul(vec![x, pool.pow(sq, pool.integer(-1))]);
        let got = crate::integrate::engine::integrate(f, x, &pool)
            .expect("∫x/√(x²−1) is elementary")
            .value;
        assert_eq!(
            simplify(got, &pool).value,
            simplify(sq, &pool).value,
            "expected √(x²−1), got {}",
            pool.display(got)
        );
    }

    /// `∫ dx/(x²√(1+x²)) = −√(1+x²)/x` — the collapse has to reach *inside* a
    /// power of the Euler parameter, not just cancel a `t ∓ t⁻¹` pair.
    #[test]
    fn euler_answer_collapses_inside_a_power() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let q = pool.add(vec![pool.integer(1), pool.pow(x, pool.integer(2))]);
        let sq = pool.func("sqrt", vec![q]);
        let f = pool.mul(vec![
            pool.pow(x, pool.integer(-2)),
            pool.pow(sq, pool.integer(-1)),
        ]);
        let got = crate::integrate::engine::integrate(f, x, &pool)
            .expect("∫dx/(x²√(1+x²)) is elementary")
            .value;
        // Equal up to the integration constant: the difference must be constant.
        let d = simplify(crate::diff::diff(got, x, &pool).unwrap().value, &pool).value;
        for &xv in &[0.4_f64, 1.3, 2.9] {
            let lhs = crate::integrate::gate::eval_at(d, x, xv, &pool).unwrap();
            let rhs = crate::integrate::gate::eval_at(f, x, xv, &pool).unwrap();
            assert!((lhs - rhs).abs() < 1e-9 * (1.0 + rhs.abs()));
        }
        // No power of the Euler parameter survives: the answer is short.
        let printed = pool.display(got).to_string();
        assert!(
            printed.len() < 60,
            "expected a collapsed A + B·√P form, got {printed}"
        );
    }

    /// A candidate that is right on only **half** its domain must be *refuted*,
    /// not accepted.
    ///
    /// This is the branch hazard of the whole route family in its sharpest form.
    /// `√(x²−1)` is real on two disjoint intervals, and
    /// `F = √(x⁴−x²)/x = |x|√(x²−1)/x` equals `√(x²−1)` on `x > 1` and
    /// `−√(x²−1)` on `x < −1` — so `d/dx F` is the integrand `x/√(x²−1)` on the
    /// right branch and its negative on the left.  The old fixed grid
    /// `[0.3 … 5.1]` never sampled a negative abscissa at all and would have
    /// waved this through; the domain-derived grid takes samples from *every*
    /// maximal run where the original integrand is a finite real, so the left
    /// branch is seen and the verdict is `Failed`, not `SampledOnly`.
    #[test]
    fn a_half_domain_candidate_is_refuted_not_accepted() {
        use crate::integrate::gate::Verdict;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let q = pool.add(vec![pool.pow(x, pool.integer(2)), pool.integer(-1)]);
        let integrand = pool.mul(vec![
            x,
            pool.pow(pool.func("sqrt", vec![q]), pool.integer(-1)),
        ]);

        // The honest antiderivative passes on both branches.
        let good = pool.func("sqrt", vec![q]);
        assert!(
            matches!(
                gate_verdict(good, integrand, q, x, &pool),
                Verdict::Proven | Verdict::EnclosureVerified { .. } | Verdict::SampledOnly { .. }
            ),
            "√(x²−1) is the antiderivative on both branches"
        );

        // `√(x⁴−x²)/x` agrees only for x > 1.
        let x4mx2 = pool.add(vec![
            pool.pow(x, pool.integer(4)),
            pool.mul(vec![pool.integer(-1), pool.pow(x, pool.integer(2))]),
        ]);
        let half = pool.mul(vec![
            pool.func("sqrt", vec![x4mx2]),
            pool.pow(x, pool.integer(-1)),
        ]);
        assert!(
            matches!(
                gate_verdict(half, integrand, q, x, &pool),
                Verdict::Failed { .. }
            ),
            "a candidate wrong on the left branch must be refuted, got {:?}",
            gate_verdict(half, integrand, q, x, &pool)
        );
    }

    /// The collapse must survive **parser output**, not just builder-constructed
    /// trees.
    ///
    /// `parse("x^2-1")` builds `x² + (−1)·1`, a different `ExprId` from the
    /// `x² + (−1)` the candidate carries after `simplify` — so keying the
    /// generator on node identity made the normalization silently do nothing and
    /// `∫x/√(x²−1)` came back in Euler form from `ak.parse` while collapsing to
    /// `√(x²−1)` from the equivalent builder tree.  That is the Charlwood
    /// form-sensitivity class again, on the generator instead of on `Mul`
    /// associativity.
    #[test]
    fn collapse_survives_parser_output() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = std::collections::HashMap::from([("x".to_owned(), x)]);
        let f = crate::parse::parse("x/sqrt(x^2-1)", &pool, &mut syms).unwrap();
        let got = crate::integrate::engine::integrate(f, x, &pool)
            .expect("∫x/√(x²−1) is elementary")
            .value;
        let want = pool.func(
            "sqrt",
            vec![pool.add(vec![pool.pow(x, pool.integer(2)), pool.integer(-1)])],
        );
        assert_eq!(
            simplify(got, &pool).value,
            simplify(want, &pool).value,
            "expected √(x²−1), got {}",
            pool.display(got)
        );
    }

    /// A radicand that is real only *far* from the origin: `√(x²−100)` needs
    /// `|x| > 10`, so a scan that stopped at `6` would find nowhere to look and
    /// decline a correct answer — the fixed-grid mistake, one order of magnitude
    /// out.  `∫x/√(x²−100) dx = √(x²−100)`.
    #[test]
    fn a_distant_real_branch_is_still_found() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let q = pool.add(vec![pool.pow(x, pool.integer(2)), pool.integer(-100)]);
        let sq = pool.func("sqrt", vec![q]);
        let f = pool.mul(vec![x, pool.pow(sq, pool.integer(-1))]);
        let got = crate::integrate::engine::integrate(f, x, &pool)
            .expect("∫x/√(x²−100) is elementary")
            .value;
        assert_eq!(
            simplify(got, &pool).value,
            simplify(sq, &pool).value,
            "expected √(x²−100), got {}",
            pool.display(got)
        );
    }

    /// A conic with **no** rational point at all (`y² = 3x²+2` is insoluble over
    /// ℚ by Hasse–Minkowski) must decline, not emit and not certify.
    #[test]
    fn conic_without_a_rational_point_declines() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let q = pool.add(vec![
            pool.mul(vec![pool.integer(3), pool.pow(x, pool.integer(2))]),
            pool.integer(2),
        ]);
        let f = pool.mul(vec![
            pool.pow(pool.add(vec![x, pool.integer(-5)]), pool.integer(-1)),
            pool.pow(pool.func("sqrt", vec![q]), pool.integer(-1)),
        ]);
        // Whatever the engine decides, it must never be a non-elementarity claim:
        // this integrand *is* elementary, so a `NonElementary` here would be a
        // false certificate.
        match crate::integrate::engine::integrate(f, x, &pool) {
            Ok(_) => {}
            Err(e) => assert!(
                !matches!(e, IntegrationError::NonElementary(_)),
                "a method failure must not become a non-elementarity certificate: {e:?}"
            ),
        }
    }
}
