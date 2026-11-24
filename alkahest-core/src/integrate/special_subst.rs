//! Special-function antiderivatives with **non-trivial arguments**, as a
//! structured pass after the elementary pipeline.
//!
//! [`super::special`] is a table of shapes whose argument is linear (or a pure
//! quadratic): `eˣ/x → Ei(x)`, `sin(x)/x → Si(x)`.  The Risch tower already
//! *proves* that `∫ (2 − x²)·e^{x/(x²+2)}/(x³ + 2x) dx` and `∫ sin(eˣ) dx` have
//! no elementary antiderivative, but the answers — `Ei(x/(x²+2))` and
//! `Si(eˣ)` — have arguments the table cannot see.  This module finds them in
//! two ways.
//!
//! 1. **Derivative-divides substitution into the full integrator.**  For an
//!    inner function `g` of the integrand (the argument of an `exp`, `sin`,
//!    `log`, …, or an `exp`/`log`/inverse-trig node itself), form
//!    `h = f/g'` and replace `g` by a fresh `u`; when `x` survives and `g` is
//!    invertible on its range (`eˣ`, `log x`, `asin x`, `atan x`, …), replace
//!    `x` by `g⁻¹(u)` too.  If `h` is then free of `x`, `∫ h(u) du` goes through
//!    the **whole** integrator — Risch, the special-function table, everything
//!    — and is back-substituted.  This is what turns `sin(eˣ)` into
//!    `∫ sin(u)/u du = Si(u)`, `(eˣ + 1)·e^{x+eˣ}/(x + eˣ)` into
//!    `∫ eᵘ/u du = Ei(u)`, and `x³·e^{asin x}/√(1 − x²)` into the elementary
//!    `∫ sin³(u)·eᵘ du`.  (The rule engine's own derivative-divides search
//!    integrates `h` with the rule engine only, so it can find neither a
//!    special function nor anything that needs the Risch tower.)
//! 2. **The `Ei` part of a hyperexponential integrand.**  For `R(x)·e^{g(x)}`
//!    with `R, g ∈ ℚ(x)`, the Risch DE `v′ + g′v = R` has a rational solution
//!    exactly when `∫ R·e^g` is elementary.  When it is not, the obstruction at
//!    a simple zero of `g` is a simple pole of `R` that no `v·e^g` can produce
//!    (a pole of `v` of order `k` makes `v′` a pole of order `k + 1 ≥ 2`), and
//!    `Ei(g)′ = g′e^g/g` has precisely such a pole.  Its coefficient is the
//!    constant `c ≡ R·g/g′ (mod p)`, `p` the squarefree part of `g`'s
//!    numerator.  Subtracting `c·g′e^g/g` leaves an integrand that the full
//!    integrator is asked to close *elementarily*; the answer is
//!    `c·Ei(g) + F`.  This is Hebisch's `∫ (2x⁴ − x³ + 3x² + 2x + 2)·
//!    e^{x/(x²+2)}/(x³ + 2x) dx = (x² + 2)·e^{x/(x²+2)} + Ei(x/(x²+2))`.
//!
//! # Soundness
//!
//! Neither step is trusted: every candidate is returned to
//! [`super::special`], which accepts it only through the same
//! `verify_antiderivative_status_parametric` gate as the table entries —
//! symbolic `F′ − f ≡ 0` through the special functions' registered
//! derivatives, else the in-domain numeric screen.  A failed candidate is a
//! decline.  Nothing here produces or changes a non-elementarity verdict.

use std::cell::Cell;
use std::collections::HashMap;

use rug::Rational;

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::engine::{integrate, is_free_of};
use super::risch::poly_rde::{degree, poly_deriv, poly_mul, trim, QPoly};
use super::risch::rational_rde::{expr_to_qrational, poly_divrem, poly_gcd, poly_sub};

thread_local! {
    /// Set while a composite candidate is being built: the inner integrals it
    /// starts may reach the special-function post-pass again, which must then
    /// stick to the table.
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

struct ActiveGuard;

impl ActiveGuard {
    fn enter() -> Option<Self> {
        ACTIVE.with(|a| {
            if a.get() {
                None
            } else {
                a.set(true);
                Some(ActiveGuard)
            }
        })
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE.with(|a| a.set(false));
    }
}

/// Most inner functions tried per integrand; each success costs a full
/// recursive `integrate`.
const MAX_CANDIDATES: usize = 6;

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

fn recip(e: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(e) {
        ExprData::Mul(args) => pool.mul(args.iter().map(|&a| recip(a, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(base, neg(exp, pool)),
        _ => pool.pow(e, pool.integer(-1_i32)),
    }
}

/// Antiderivative candidates for `expr`, **unverified**.  The caller gates
/// every one of them.
pub(crate) fn composite_candidates(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let Some(_guard) = ActiveGuard::enter() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if crate::budget::check().is_err() {
        return out;
    }
    if let Some(f) = ei_part(expr, var, pool) {
        out.push(f);
    }
    if let Some(f) = li_part(expr, var, pool) {
        out.push(f);
    }
    out.extend(substitution_candidates(expr, var, pool));
    out
}

// ---------------------------------------------------------------------------
// 1. Derivative-divides substitution
// ---------------------------------------------------------------------------

fn inner_functions(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    fn walk(e: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
        if is_free_of(e, var, pool) {
            return;
        }
        match pool.get(e) {
            ExprData::Func { name, args } => {
                if args.len() == 1 {
                    let a = args[0];
                    if matches!(
                        name.as_str(),
                        "exp" | "log" | "asin" | "acos" | "atan" | "asinh"
                    ) && !out.contains(&e)
                    {
                        out.push(e);
                    }
                    if a != var
                        && matches!(
                            name.as_str(),
                            "exp" | "log" | "sin" | "cos" | "sinh" | "cosh" | "tan"
                        )
                        && !out.contains(&a)
                    {
                        out.push(a);
                    }
                }
                for a in args {
                    walk(a, var, pool, out);
                }
            }
            ExprData::Pow { base, exp } => {
                walk(base, var, pool, out);
                walk(exp, var, pool, out);
            }
            ExprData::Add(args) | ExprData::Mul(args) => {
                for a in args {
                    walk(a, var, pool, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(expr, var, pool, &mut out);
    // Composite arguments first (`x + eˣ` before `eˣ`).
    out.sort_by_key(|&g| std::cmp::Reverse(pool.depth(g)));
    out
}

/// `(a, b)` with `arg = a·var + b`, `a ≠ 0` numeric-or-symbolic constant.
fn linear(arg: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let a = simplify(crate::diff::diff(arg, var, pool).ok()?.value, pool).value;
    if !is_free_of(a, var, pool) || super::algebraic::poly_utils::is_zero_expr(a, pool) {
        return None;
    }
    let b = simplify(pool.add(vec![arg, neg(pool.mul(vec![a, var]), pool)]), pool).value;
    is_free_of(b, var, pool).then_some((a, b))
}

/// The radicand-and-replacement pairs that become polynomial in the
/// trigonometric form of the inverse function: on the range of `asin`,
/// `√(1 − x²) = cos u` exactly, and so on.
fn inverse_of(
    g: ExprId,
    var: ExprId,
    u: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, Vec<(ExprId, ExprId)>)> {
    let ExprData::Func { name, args } = pool.get(g) else {
        return None;
    };
    let arg = args.first().copied()?;
    let one = pool.integer(1_i32);
    let x2 = pool.pow(var, pool.integer(2_i32));
    match name.as_str() {
        "exp" => {
            let (a, b) = linear(arg, var, pool)?;
            let x = pool.mul(vec![
                pool.add(vec![pool.func("log", vec![u]), neg(b, pool)]),
                recip(a, pool),
            ]);
            Some((x, vec![]))
        }
        "log" => {
            let (a, b) = linear(arg, var, pool)?;
            let x = pool.mul(vec![
                pool.add(vec![pool.func("exp", vec![u]), neg(b, pool)]),
                recip(a, pool),
            ]);
            Some((x, vec![]))
        }
        "asin" if arg == var => Some((
            pool.func("sin", vec![u]),
            vec![(
                pool.add(vec![one, neg(x2, pool)]),
                pool.func("cos", vec![u]),
            )],
        )),
        "acos" if arg == var => Some((
            pool.func("cos", vec![u]),
            vec![(
                pool.add(vec![one, neg(x2, pool)]),
                pool.func("sin", vec![u]),
            )],
        )),
        "atan" if arg == var => Some((
            pool.func("tan", vec![u]),
            vec![(
                pool.add(vec![one, x2]),
                recip(pool.func("cos", vec![u]), pool),
            )],
        )),
        "asinh" if arg == var => Some((
            pool.func("sinh", vec![u]),
            vec![(pool.add(vec![one, x2]), pool.func("cosh", vec![u]))],
        )),
        _ => None,
    }
}

/// Replace every `radicand^(k/2)` (radicand equal to one of `roots`' keys up
/// to simplification) by `value^k`.
fn replace_radicals(expr: ExprId, roots: &[(ExprId, ExprId)], pool: &ExprPool) -> ExprId {
    if roots.is_empty() {
        return expr;
    }
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if *r.0.denom() == 2 {
                    for &(rad, val) in roots {
                        let diff = simplify(pool.add(vec![base, neg(rad, pool)]), pool).value;
                        if super::algebraic::poly_utils::is_zero_expr(diff, pool) {
                            return pool.pow(val, pool.integer(r.0.numer().clone()));
                        }
                    }
                }
            }
            pool.pow(
                replace_radicals(base, roots, pool),
                replace_radicals(exp, roots, pool),
            )
        }
        ExprData::Func { name, args } => {
            if name == "sqrt" && args.len() == 1 {
                for &(rad, val) in roots {
                    let diff = simplify(pool.add(vec![args[0], neg(rad, pool)]), pool).value;
                    if super::algebraic::poly_utils::is_zero_expr(diff, pool) {
                        return val;
                    }
                }
            }
            pool.func(
                name,
                args.iter()
                    .map(|&a| replace_radicals(a, roots, pool))
                    .collect(),
            )
        }
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| replace_radicals(a, roots, pool))
                .collect(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| replace_radicals(a, roots, pool))
                .collect(),
        ),
        _ => expr,
    }
}

fn substitution_candidates(expr: ExprId, var: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::new();
    let u = pool.symbol("__special_subst_u", Domain::Real);
    let mut tried = 0usize;
    for g in inner_functions(expr, var, pool) {
        if tried >= MAX_CANDIDATES || crate::budget::check().is_err() {
            break;
        }
        // A linear `g` is the table's business, and substituting it changes
        // nothing structural.
        if linear(g, var, pool).is_some() {
            continue;
        }
        let Ok(dg) = crate::diff::diff(g, var, pool) else {
            continue;
        };
        let dg = simplify(dg.value, pool).value;
        if super::algebraic::poly_utils::is_zero_expr(dg, pool) {
            continue;
        }
        let quotient = simplify(pool.mul(vec![expr, recip(dg, pool)]), pool).value;
        let mut fwd = HashMap::new();
        fwd.insert(g, u);
        let mut h = crate::kernel::subs(quotient, &fwd, pool);
        if !is_free_of(h, var, pool) {
            let Some((x_of_u, roots)) = inverse_of(g, var, u, pool) else {
                continue;
            };
            let h2 = replace_radicals(h, &roots, pool);
            let mut back = HashMap::new();
            back.insert(var, x_of_u);
            h = simplify(crate::kernel::subs(h2, &back, pool), pool).value;
            if !is_free_of(h, var, pool) {
                continue;
            }
        } else {
            h = simplify(h, pool).value;
        }
        if is_free_of(h, u, pool) {
            continue;
        }
        tried += 1;
        // The pre-normalising tier (`eᵘ·sin³u` by product-to-sum) runs only
        // at the outermost `integrate`; this inner integral is a fresh
        // problem in `u`, so give it that tier explicitly.
        let inner = match integrate(h, u, pool) {
            Ok(v) => v.value,
            Err(_) => match super::prenorm::try_prenormalised(h, u, pool) {
                Ok(Some(v)) => v.value,
                _ => continue,
            },
        };
        let mut back = HashMap::new();
        back.insert(u, g);
        out.push(simplify(crate::kernel::subs(inner, &back, pool), pool).value);
    }
    out
}

// ---------------------------------------------------------------------------
// 2. The Ei part of R(x)·e^{g(x)}
// ---------------------------------------------------------------------------

fn terms(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(expr) {
        ExprData::Add(args) => args,
        _ => vec![expr],
    }
}

/// Split a product term into `(exp argument, cofactor)` when it has exactly one
/// `exp(·)` factor with a `var`-dependent argument, to the first power.
fn split_exp(term: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let factors = match pool.get(term) {
        ExprData::Mul(args) => args,
        _ => vec![term],
    };
    let mut arg = None;
    let mut rest = Vec::new();
    for f in factors {
        match pool.get(f) {
            ExprData::Func { name, args }
                if name == "exp" && args.len() == 1 && !is_free_of(f, var, pool) =>
            {
                if arg.is_some() {
                    return None;
                }
                arg = Some(args[0]);
            }
            _ => rest.push(f),
        }
    }
    let cof = match rest.len() {
        0 => pool.integer(1_i32),
        1 => rest[0],
        _ => pool.mul(rest),
    };
    Some((arg?, cof))
}

fn qpoly_is_zero(p: &QPoly) -> bool {
    trim(p.clone()).is_empty()
}

/// The constant `c` with `num/den ≡ c (mod p)`, or `None` when the residue is
/// not a constant (or `den` is not invertible mod `p`).
fn constant_mod(num: &QPoly, den: &QPoly, p: &QPoly) -> Option<Rational> {
    let (_, n) = poly_divrem(num, p);
    let (_, m) = poly_divrem(den, p);
    let (n, m) = (trim(n), trim(m));
    if m.is_empty() {
        return None;
    }
    if n.is_empty() {
        return Some(Rational::from(0));
    }
    if degree(&n) != degree(&m) {
        return None;
    }
    let c = Rational::from(n.last()? / m.last()?);
    let scaled: QPoly = m.iter().map(|q| Rational::from(q * &c)).collect();
    qpoly_is_zero(&poly_sub(&n, &scaled)).then_some(c)
}

// ---------------------------------------------------------------------------
// 3. The li part of R(x, log x)
// ---------------------------------------------------------------------------

/// For `R(x, t)` rational in `t = log x`: a simple pole of `R` at `t = 0` with
/// residue `c·xᵏ` contributes `c·xᵏ/log x`, whose integral is `c·li(x)`
/// (`k = 0`) or `c·Ei((k+1)·log x)` — the hyperlogarithmic analogue of the
/// `Ei` part above.  Subtract it and ask the full integrator for the rest,
/// which must then be elementary: Bronstein's
/// `∫ (2 log²x − log x − x²)/(log³x − x² log x) dx
///   = li(x) + ½ log(x + log x) − ½ log(x − log x)`.
fn li_part(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let work = simplify(expr, pool).value;
    let log_x = pool.func("log", vec![var]);
    let t = pool.symbol("__special_li_t", Domain::Real);
    let mut fwd = HashMap::new();
    fwd.insert(log_x, t);
    let rt = crate::kernel::subs(work, &fwd, pool);
    if is_free_of(rt, t, pool) {
        return None;
    }
    let (n, d) = crate::poly::cancel::together_parts(rt, vec![t, var], pool).ok()?;
    let at_zero = |e: ExprId| {
        let mut m = HashMap::new();
        m.insert(t, pool.integer(0_i32));
        simplify(crate::kernel::subs(e, &m, pool), pool).value
    };
    if !super::algebraic::poly_utils::is_zero_expr(at_zero(d), pool) {
        return None;
    }
    let dt = crate::diff::diff(d, t, pool).ok()?.value;
    let dt0 = at_zero(dt);
    if super::algebraic::poly_utils::is_zero_expr(dt0, pool) {
        return None; // a pole of order ≥ 2 at t = 0
    }
    let r = simplify(pool.mul(vec![at_zero(n), recip(dt0, pool)]), pool).value;
    if super::algebraic::poly_utils::is_zero_expr(r, pool) {
        return None;
    }
    // r = c·xᵏ.
    let (rn, rd) = expr_to_qrational(r, var, pool)?;
    let (rn, rd) = (trim(rn), trim(rd));
    let lead = |p: &QPoly| -> Option<(usize, Rational)> {
        let nz: Vec<usize> = (0..p.len()).filter(|&i| p[i] != 0).collect();
        (nz.len() == 1).then(|| (nz[0], p[nz[0]].clone()))
    };
    let (i, cn) = lead(&rn)?;
    let (j, cd) = lead(&rd)?;
    let k = i as i64 - j as i64;
    if k == -1 {
        return None; // c/(x·log x) = (c·log log x)′, elementary
    }
    let c = Rational::from(&cn / &cd);
    let c_expr = pool.rational(c.numer().clone(), c.denom().clone());
    let special = if k == 0 {
        pool.func("li", vec![var])
    } else {
        pool.func("Ei", vec![pool.mul(vec![pool.integer(k + 1), log_x])])
    };
    let rest = simplify(
        pool.add(vec![
            work,
            neg(
                pool.mul(vec![
                    c_expr,
                    pool.pow(var, pool.integer(k)),
                    recip(log_x, pool),
                ]),
                pool,
            ),
        ]),
        pool,
    )
    .value;
    let mut parts = vec![pool.mul(vec![c_expr, special])];
    if !super::algebraic::poly_utils::is_zero_expr(rest, pool) {
        let f = integrate(rest, var, pool).ok()?.value;
        if !super::special::basis_functions_used(f, pool).is_empty() {
            return None;
        }
        parts.push(f);
    }
    Some(simplify(pool.add(parts), pool).value)
}

fn ei_part(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let work = simplify(expr, pool).value;
    // Group the exponential terms by argument.
    let mut groups: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
    for t in terms(work, pool) {
        if is_free_of(t, var, pool) {
            continue;
        }
        let Some((arg, cof)) = split_exp(t, var, pool) else {
            continue;
        };
        match groups.iter_mut().find(|(a, _)| *a == arg) {
            Some((_, v)) => v.push(cof),
            None => groups.push((arg, vec![cof])),
        }
    }
    let mut ei_terms = Vec::new();
    let mut subtract = Vec::new();
    for (g, cofs) in groups {
        let r = simplify(pool.add(cofs), pool).value;
        let Some((a, b)) = expr_to_qrational(r, var, pool) else {
            continue;
        };
        let Some((p, q)) = expr_to_qrational(g, var, pool) else {
            continue;
        };
        if qpoly_is_zero(&b) || qpoly_is_zero(&q) || degree(&p) < 1 {
            continue; // `g` has no zero: no `Ei` term from it.
        }
        // A linear `g` is the table's.
        if degree(&p) <= 1 && degree(&q) == 0 {
            continue;
        }
        // R·g/g′ = A·P·Q / (B·(P′Q − PQ′)).
        let dg_num = poly_sub(
            &poly_mul(&poly_deriv(&p), &q),
            &poly_mul(&p, &poly_deriv(&q)),
        );
        if qpoly_is_zero(&dg_num) {
            continue;
        }
        let mut num = poly_mul(&poly_mul(&a, &p), &q);
        let mut den = poly_mul(&b, &dg_num);
        let common = poly_gcd(&num, &den);
        if degree(&common) > 0 {
            num = poly_divrem(&num, &common).0;
            den = poly_divrem(&den, &common).0;
        }
        // Squarefree part of P.
        let gp = poly_gcd(&p, &poly_deriv(&p));
        let sqf = if degree(&gp) > 0 {
            poly_divrem(&p, &gp).0
        } else {
            p.clone()
        };
        let Some(c) = constant_mod(&num, &den, &sqf) else {
            continue;
        };
        if c == 0 {
            continue;
        }
        let c_expr = pool.rational(c.numer().clone(), c.denom().clone());
        ei_terms.push(pool.mul(vec![c_expr, pool.func("Ei", vec![g])]));
        let dg = simplify(crate::diff::diff(g, var, pool).ok()?.value, pool).value;
        subtract.push(pool.mul(vec![
            neg(c_expr, pool),
            dg,
            pool.func("exp", vec![g]),
            recip(g, pool),
        ]));
    }
    if ei_terms.is_empty() {
        return None;
    }
    let mut rest_terms = vec![work];
    rest_terms.extend(subtract);
    let rest = simplify(pool.add(rest_terms), pool).value;
    let mut parts = ei_terms;
    if !super::algebraic::poly_utils::is_zero_expr(rest, pool) {
        let f = integrate(rest, var, pool).ok()?.value;
        if !super::special::basis_functions_used(f, pool).is_empty() {
            return None; // the remainder is meant to be elementary
        }
        parts.push(f);
    }
    Some(simplify(pool.add(parts), pool).value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::engine::verify_antiderivative_status;

    fn solves_with(src: &str, name: &str) {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        let x = pool.symbol("x", Domain::Real);
        syms.insert("x".to_string(), x);
        let e = crate::parse::parse(src, &pool, &mut syms).expect("parses");
        let f = integrate(e, x, &pool)
            .unwrap_or_else(|err| panic!("∫ {src} dx failed: {err}"))
            .value;
        assert!(
            verify_antiderivative_status(f, e, x, &pool).is_some(),
            "∫ {src} dx = {} does not verify",
            pool.display(f)
        );
        let used = super::super::special::basis_functions_used(f, &pool);
        if name.is_empty() {
            assert!(
                used.is_empty(),
                "∫ {src} dx = {} should be elementary",
                pool.display(f)
            );
        } else {
            assert!(
                used.contains(&name),
                "∫ {src} dx = {} should use {name}",
                pool.display(f)
            );
        }
    }

    #[test]
    fn hebisch_ei_with_rational_argument() {
        solves_with("(2 - x^2)*exp(x/(x^2 + 2))/(x^3 + 2*x)", "Ei");
        solves_with(
            "(2*x^4 - x^3 + 3*x^2 + 2*x + 2)*exp(x/(x^2 + 2))/(x^3 + 2*x)",
            "Ei",
        );
    }

    #[test]
    fn ei_of_a_tower_argument() {
        solves_with("(exp(x) + 1)*exp(x + exp(x))/(x + exp(x))", "Ei");
    }

    #[test]
    fn ei_of_a_quadratic_argument() {
        solves_with("exp(x^2)/x", "Ei");
        solves_with("exp(-x^2)/x", "Ei");
    }

    #[test]
    fn si_of_an_exponential() {
        solves_with("sin(exp(x))", "Si");
        solves_with("cos(exp(x))", "Ci");
    }

    #[test]
    fn li_plus_logarithms() {
        solves_with(
            "(-x^2 + 2*log(x)^2 - log(x))/(-x^2*log(x) + log(x)^3)",
            "li",
        );
        solves_with("x^2/log(x) + 1/x", "Ei");
    }

    #[test]
    fn li_of_a_square() {
        solves_with("x/log(x^2)", "li");
    }

    #[test]
    fn substitution_through_an_inverse_function() {
        solves_with("x^3*exp(asin(x))/sqrt(1 - x^2)", "");
    }

    #[test]
    fn constant_residue_mod_p() {
        // num/den = (x + 2)/(2x + 4) ≡ 1/2 everywhere.
        let num = vec![Rational::from(2), Rational::from(1)];
        let den = vec![Rational::from(4), Rational::from(2)];
        let p = vec![Rational::from(0), Rational::from(1)];
        assert_eq!(constant_mod(&num, &den, &p), Some(Rational::from((1, 2))));
        // x/(1) mod x² + 1 is not constant.
        let p2 = vec![Rational::from(1), Rational::from(0), Rational::from(1)];
        assert_eq!(
            constant_mod(
                &vec![Rational::from(0), Rational::from(1)],
                &vec![Rational::from(1)],
                &p2
            ),
            None
        );
    }
}
