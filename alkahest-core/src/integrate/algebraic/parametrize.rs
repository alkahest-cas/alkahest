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
    let f_s = match crate::integrate::engine::integrate(integrand_s, s, pool) {
        Ok(d) => d.value,
        Err(_) => return None,
    };
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
    let f_t = match crate::integrate::engine::integrate(integrand_t, t, pool) {
        Ok(d) => d.value,
        Err(_) => return None,
    };
    let mut back = HashMap::new();
    back.insert(t, back_t);
    let f_x = simplify(crate::kernel::subs(f_t, &back, pool), pool).value;

    if !verify_derivative(f_x, expr, radicand, var, pool) {
        return None;
    }
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("algebraic_genus0_euler", expr, f_x));
    Some(Ok(DerivedExpr::with_log(f_x, log)))
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
    let f_t = integrate_scaled_rational(integrand_t, t, pool)?;

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

    if !verify_derivative(f_x, expr, radicand, var, pool) {
        return None;
    }
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "algebraic_genus0_euler_general",
        expr,
        f_x,
    ));
    Some(Ok(DerivedExpr::with_log(f_x, log)))
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

/// Numeric `d/dx F = integrand` check at sample points where the radicand is
/// positive (so the principal real branch of the radical is well defined).
fn verify_derivative(
    f: ExprId,
    integrand: ExprId,
    radicand: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> bool {
    let Ok(df) = crate::diff::diff(f, var, pool) else {
        return false;
    };
    let ds = simplify(df.value, pool).value;
    let mut checked = 0;
    for &xv in &[0.3_f64, 0.8, 1.6, 2.7, 3.9, 5.1] {
        match eval(radicand, var, xv, pool) {
            Some(r) if r > 1e-6 && r.is_finite() => {}
            _ => continue, // radicand not safely positive: skip
        }
        let (Some(lhs), Some(rhs)) = (eval(ds, var, xv, pool), eval(integrand, var, xv, pool))
        else {
            return false;
        };
        if !lhs.is_finite() || !rhs.is_finite() || (lhs - rhs).abs() > 1e-6 * (1.0 + rhs.abs()) {
            return false;
        }
        checked += 1;
    }
    checked >= 2
}

fn eval(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> Option<f64> {
    if expr == x {
        return Some(xv);
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => Some(r.0.to_f64()),
        ExprData::Add(args) => args
            .iter()
            .try_fold(0.0, |s, &a| Some(s + eval(a, x, xv, pool)?)),
        ExprData::Mul(args) => args
            .iter()
            .try_fold(1.0, |s, &a| Some(s * eval(a, x, xv, pool)?)),
        ExprData::Pow { base, exp } => Some(eval(base, x, xv, pool)?.powf(eval(exp, x, xv, pool)?)),
        ExprData::Func { ref name, ref args } if args.len() == 1 => {
            let a = eval(args[0], x, xv, pool)?;
            match name.as_str() {
                "exp" => Some(a.exp()),
                "log" => Some(a.ln()),
                "sqrt" => Some(a.sqrt()),
                "cbrt" => Some(a.cbrt()),
                _ => None,
            }
        }
        _ => None,
    }
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
            let lhs = eval(ds, x, xv, &pool).unwrap();
            let rhs = eval(f, x, xv, &pool).unwrap();
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
}
