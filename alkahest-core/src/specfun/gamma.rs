//! `Γ`, `ψ` and `ψ⁽ⁿ⁾`: exact special values, the `polygamma(n, x)` node, and
//! the pole shift that lets `series` expand them at `0, −1, −2, …` (G11c).
//!
//! # Exact values
//!
//! Every value is built from exact atoms — rationals, `π`, `√3`, `log 2`,
//! `log 3`, `ζ(n)`, Catalan's `G` and the named constant `EulerGamma` — and
//! never from a float literal:
//!
//! | node | values |
//! |---|---|
//! | `gamma(x)` | `(n−1)!` at a positive integer; `Γ(m+½) = r·√π` |
//! | `digamma(x)` | `ψ(f+m)` for `f ∈ {1, ½, ⅓, ⅔, ¼, ¾, ⅙, ⅚}` in classical form (DLMF 5.4.13–5.4.19), every other `f = p/q` with `q ≤` [`MAX_GAUSS_DENOMINATOR`] by Gauss's digamma theorem (F26), then `ψ(x+1) = ψ(x) + 1/x` |
//! | `trigamma(x)` | `ψ₁(f+m)` for `f ∈ {1, ½, ¼, ¾}` (`ψ₁(¼) = π² + 8G`); any other rational as `ζ(2, f)` (`hurwitz_zeta`) |
//! | `polygamma(n, x)`, `n ≥ 2` | `(−1)ⁿ⁺¹ n! ζ(n+1, f+m)` at every rational: `ζ(s, ½) = (2ˢ − 1) ζ(s)`; for odd `s` at `f ∈ {⅓, ⅔, ¼, ¾, ⅙, ⅚}` in `ζ(s)`, `πˢ`, `√3` (multiplication + reflection, F26); otherwise the exact `hurwitz_zeta(s, f)` |
//!
//! `ζ(n+1)` is the [`crate::specfun`] node, so an even argument becomes its
//! power of `π` (`ψ₁(1) = π²/6`) and an odd one stays `zeta(3)`.
//!
//! # Poles
//!
//! `Γ`, `ψ` and every `ψ⁽ⁿ⁾` have poles at `0, −1, −2, …`. A node whose
//! argument *is* such an integer keeps its node (there is no value). For a
//! series expansion [`shift_off_poles`] rewrites a node whose argument
//! *tends* to one by the recurrences
//!
//! * `Γ(u) = Γ(u+k+1) / (u (u+1) ⋯ (u+k))`,
//! * `ψ⁽ⁿ⁾(u) = ψ⁽ⁿ⁾(u+k+1) − (−1)ⁿ n! Σ_{j=0}^{k} (u+j)^{−n−1}`,
//!
//! after which the function factor is analytic at the point and the pole is a
//! plain power — so `series(gamma(eps), eps, 0, 2)` is
//! `1/eps − EulerGamma + (EulerGamma²/2 + π²/12)·eps + O(eps²)`.

use rug::ops::Pow;
use rug::{Integer, Rational};

use crate::ball::ArbBall;
use crate::kernel::{subs, ExprData, ExprId, ExprPool};
use crate::orthopoly::{int_literal, mk_add, mk_mul, rat_node, rational_literal};
use crate::primitive::Primitive;
use crate::simplify::simplify;

use super::numeric::Kernel;
use super::SpecialFunctionError;

/// Largest `|m|` for which `ψ⁽ⁿ⁾(f + m)` is rewritten with an explicit
/// rational correction (`ψ(1001) = H₁₀₀₀ − γ` has a 433-digit denominator).
pub(crate) const MAX_SHIFT: i64 = 1000;
/// Largest positive integer `n` for which `Γ(n) = (n−1)!` is folded.
pub(crate) const MAX_GAMMA_FACTORIAL: i64 = 1000;
/// Largest polygamma order rewritten exactly.
pub(crate) const MAX_POLYGAMMA_ORDER: i64 = 200;
/// Deepest pole `−k` that [`shift_off_poles`] shifts away.
const MAX_POLE_SHIFT: i64 = 64;

// ---------------------------------------------------------------------------
// Named constants
// ---------------------------------------------------------------------------

/// The Euler–Mascheroni constant `γ = 0.5772…` as the exact symbol
/// `EulerGamma`.
///
/// It prints as `EulerGamma` (LaTeX `\gamma`), parses back, and every
/// numeric evaluator (`f64`, complex, balls at any precision, the JITs, the
/// C emitter) resolves it — but no symbolic path folds it into a float.
pub fn euler_gamma(pool: &ExprPool) -> ExprId {
    crate::eval::symbols::euler_gamma_symbol(pool)
}

/// Catalan's constant `G = Σ_{k≥0} (−1)ᵏ/(2k+1)² = 0.9159…` as the symbol
/// `Catalan` (LaTeX `G`).
pub fn catalan(pool: &ExprPool) -> ExprId {
    pool.symbol(
        crate::eval::symbols::CATALAN_NAME,
        crate::kernel::Domain::Real,
    )
}

/// The Glaisher–Kinkelin constant `A = 1.2824…` as the symbol `Glaisher`
/// (LaTeX `A`). Its ball enclosure needs Arb.
pub fn glaisher(pool: &ExprPool) -> ExprId {
    pool.symbol(
        crate::eval::symbols::GLAISHER_NAME,
        crate::kernel::Domain::Real,
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn log_of(pool: &ExprPool, n: i64) -> ExprId {
    pool.func("log", vec![pool.integer(n)])
}

fn sqrt3(pool: &ExprPool) -> ExprId {
    pool.func("sqrt", vec![pool.integer(3)])
}

fn pi(pool: &ExprPool) -> ExprId {
    crate::eval::symbols::pi_symbol(pool)
}

fn neg(pool: &ExprPool, e: ExprId) -> ExprId {
    mk_mul(pool, vec![pool.integer(-1), e])
}

/// `x = f + m` with `f ∈ (0, 1]` and `m` an integer.
fn split(x: &Rational) -> (Rational, i64) {
    let fl = x.clone().floor();
    let mut f = x.clone() - &fl;
    let mut m = fl;
    if f == 0 {
        f = Rational::from(1);
        m -= 1;
    }
    (f, m.numer().to_i64().unwrap_or(i64::MAX))
}

fn factorial(n: u32) -> Integer {
    Integer::from(Integer::factorial(n))
}

/// The integer `−k ≤ 0` that `x` is, if it is one.
fn nonpositive_integer(x: &Rational) -> Option<Integer> {
    (*x.denom() == 1 && *x <= 0).then(|| Integer::from(-x.numer()))
}

// ---------------------------------------------------------------------------
// Γ
// ---------------------------------------------------------------------------

/// The exact value of `Γ(x)` for a literal `x`: `(n−1)!` at a positive
/// integer `n ≤` [`MAX_GAMMA_FACTORIAL`], a rational multiple of `√π` at a
/// half-integer; `None` elsewhere (including the poles).
pub(crate) fn gamma_value(x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let q = rational_literal(pool, x)?;
    if *q.denom() == 1 {
        let n = q.numer().to_i64()?;
        if !(1..=MAX_GAMMA_FACTORIAL).contains(&n) {
            return None;
        }
        return Some(pool.integer(factorial((n - 1) as u32)));
    }
    crate::simplify::special_values::gamma_half_integer(&q, pool)
}

// ---------------------------------------------------------------------------
// ψ
// ---------------------------------------------------------------------------

/// `ψ(f)` for the tabulated `f ∈ (0, 1]` (DLMF 5.4.12–5.4.19).
fn digamma_base(f: &Rational, pool: &ExprPool) -> Option<ExprId> {
    let g = neg(pool, crate::eval::symbols::euler_gamma_symbol(pool));
    let (n, d) = (f.numer().to_i64()?, f.denom().to_i64()?);
    let r = |a: i64, b: i64| pool.rational(a, b);
    // π√3·c for a rational c.
    let pi_sqrt3 = |c: Rational| mk_mul(pool, vec![rat_node(pool, &c), pi(pool), sqrt3(pool)]);
    let terms = match (n, d) {
        (1, 1) => vec![g],
        // −γ − 2 log 2
        (1, 2) => vec![g, mk_mul(pool, vec![pool.integer(-2), log_of(pool, 2)])],
        // −γ ∓ π/(2√3) − (3/2) log 3
        (1, 3) | (2, 3) => {
            let s = if n == 1 { -1 } else { 1 };
            vec![
                g,
                pi_sqrt3(Rational::from((s, 6))),
                mk_mul(pool, vec![r(-3, 2), log_of(pool, 3)]),
            ]
        }
        // −γ ∓ π/2 − 3 log 2
        (1, 4) | (3, 4) => {
            let s = if n == 1 { -1 } else { 1 };
            vec![
                g,
                mk_mul(pool, vec![r(s, 2), pi(pool)]),
                mk_mul(pool, vec![pool.integer(-3), log_of(pool, 2)]),
            ]
        }
        // −γ ∓ (√3/2) π − 2 log 2 − (3/2) log 3
        (1, 6) | (5, 6) => {
            let s = if n == 1 { -1 } else { 1 };
            vec![
                g,
                pi_sqrt3(Rational::from((s, 2))),
                mk_mul(pool, vec![pool.integer(-2), log_of(pool, 2)]),
                mk_mul(pool, vec![r(-3, 2), log_of(pool, 3)]),
            ]
        }
        _ => return None,
    };
    Some(mk_add(pool, terms))
}

/// Largest denominator `q` for which `ψ(p/q)` is written out by Gauss's
/// digamma theorem (`⌊(q−1)/2⌋` cosine–log terms).
pub(crate) const MAX_GAUSS_DENOMINATOR: i64 = 100;

/// `ψ(p/q)` for `0 < p < q`, `gcd(p, q) = 1`, by Gauss's digamma theorem
/// (DLMF 5.4.19 with `ln(2 − 2cos 2θ) = 2 ln 2 + 2 ln sin θ` and
/// `Σ_{k=1}^{q−1} cos(2πkp/q) = −1`):
///
/// `ψ(p/q) = −γ − log(2q) − (π/2) cot(πp/q) + 2 Σ_{k=1}^{⌊(q−1)/2⌋} cos(2πkp/q) log sin(πk/q)`.
///
/// Every trigonometric value is an exact node at a rational multiple of `π`
/// (folded to radicals where `simplify` knows them); `None` past
/// [`MAX_GAUSS_DENOMINATOR`] or when the active budget runs out.
fn gauss_digamma(f: &Rational, pool: &ExprPool) -> Option<ExprId> {
    let (p, q) = (f.numer().to_i64()?, f.denom().to_i64()?);
    if !(2..=MAX_GAUSS_DENOMINATOR).contains(&q) || !(1..q).contains(&p) {
        return None;
    }
    let at = |num: i64, den: i64| mk_mul(pool, vec![pool.rational(num, den), pi(pool)]);
    let mut terms = vec![
        neg(pool, crate::eval::symbols::euler_gamma_symbol(pool)),
        neg(pool, log_of(pool, 2 * q)),
        // −(π/2)·cot(πp/q) = −(π/2)/tan(πp/q); q ≥ 3 here, so no pole.
        mk_mul(
            pool,
            vec![
                pool.rational(-1, 2),
                pi(pool),
                pool.pow(pool.func("tan", vec![at(p, q)]), pool.integer(-1)),
            ],
        ),
    ];
    for k in 1..=(q - 1) / 2 {
        crate::budget::check().ok()?;
        terms.push(mk_mul(
            pool,
            vec![
                pool.integer(2),
                pool.func("cos", vec![at(2 * k * p, q)]),
                pool.func("log", vec![pool.func("sin", vec![at(k, q)])]),
            ],
        ));
    }
    Some(simplify(mk_add(pool, terms), pool).value)
}

/// The exact value of `ψ(x)` for a rational literal `x`: the tabulated
/// fractional parts in their classical forms, every other denominator up to
/// [`MAX_GAUSS_DENOMINATOR`] by Gauss's digamma theorem; `None` at a pole or
/// elsewhere.
pub(crate) fn digamma_value(x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let q = rational_literal(pool, x)?;
    if nonpositive_integer(&q).is_some() {
        return None;
    }
    let (f, m) = split(&q);
    if m.abs() > MAX_SHIFT {
        return None;
    }
    let base = digamma_base(&f, pool).or_else(|| gauss_digamma(&f, pool))?;
    // ψ(f + m) = ψ(f) + Σ_{j=0}^{m−1} 1/(f+j);  ψ(f − m) = ψ(f) − Σ_{j=1}^{m} 1/(f−j).
    let mut corr = Rational::new();
    if m > 0 {
        for j in 0..m {
            corr += (f.clone() + Rational::from(j)).recip();
        }
    } else {
        for j in 1..=(-m) {
            corr -= (f.clone() - Rational::from(j)).recip();
        }
    }
    Some(if corr == 0 {
        base
    } else {
        mk_add(pool, vec![rat_node(pool, &corr), base])
    })
}

// ---------------------------------------------------------------------------
// ψ⁽ⁿ⁾, n ≥ 1
// ---------------------------------------------------------------------------

/// `ζ(s, f)` for the tabulated `f`, `s = n + 1 ≥ 2`.
fn hurwitz_base(s: u32, f: &Rational, pool: &ExprPool) -> Option<ExprId> {
    let zeta = || super::node(Kernel::Zeta, vec![pool.integer(s)], pool);
    let (n, d) = (f.numer().to_i64()?, f.denom().to_i64()?);
    Some(match (n, d) {
        (1, 1) => zeta(),
        // ζ(s, ½) = (2ˢ − 1) ζ(s)  (DLMF 25.11.11)
        (1, 2) => {
            let c = (Integer::from(1) << s) - 1u32;
            mk_mul(pool, vec![pool.integer(c), zeta()])
        }
        // ζ(2, ¼) = π² + 8G, ζ(2, ¾) = π² − 8G  (from ψ₁(¼) and reflection)
        (1, 4) | (3, 4) if s == 2 => {
            let sign = if n == 1 { 8 } else { -8 };
            mk_add(
                pool,
                vec![
                    pool.pow(pi(pool), pool.integer(2)),
                    mk_mul(
                        pool,
                        vec![
                            pool.integer(sign),
                            pool.symbol(
                                crate::eval::symbols::CATALAN_NAME,
                                crate::kernel::Domain::Real,
                            ),
                        ],
                    ),
                ],
            )
        }
        _ => return None,
    })
}

/// `a + b√3` with rational `a`, `b` — enough for `cot(πf)` at `f` with
/// denominator 3, 4 or 6.
#[derive(Clone)]
struct QSqrt3(Rational, Rational);

impl QSqrt3 {
    fn mul(&self, o: &QSqrt3) -> QSqrt3 {
        QSqrt3(
            Rational::from(&self.0 * &o.0) + Rational::from(&self.1 * &o.1) * 3,
            Rational::from(&self.0 * &o.1) + Rational::from(&self.1 * &o.0),
        )
    }
}

/// `Pₙ` with `dⁿ/dθⁿ cot θ = Pₙ(cot θ)`: `P₀ = c`, `Pₖ₊₁ = −(1 + c²) Pₖ′`
/// (coefficients, constant term first).
fn cot_derivative_poly(n: u32) -> Vec<Integer> {
    let mut p = vec![Integer::from(0), Integer::from(1)];
    for _ in 0..n {
        // p′
        let d: Vec<Integer> = p
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, c)| Integer::from(c * k as u32))
            .collect();
        // −(1 + c²)·p′
        let mut out = vec![Integer::from(0); d.len() + 2];
        for (k, c) in d.iter().enumerate() {
            out[k] -= c;
            out[k + 2] -= c;
        }
        p = out;
    }
    p
}

/// `ζ(s, f)` for odd `s ≥ 3` and `f ∈ {⅓, ⅔, ¼, ¾, ⅙, ⅚}` in terms of
/// `ζ(s)`, `πˢ` and `√3`:
///
/// * the multiplication theorem `Σ_{j=1}^{q} ζ(s, j/q) = qˢ ζ(s)` with the
///   values at ½ and 1 gives `ζ(s, f) + ζ(s, 1−f) = c_q ζ(s)` (`c₃ = 3ˢ − 1`,
///   `c₄ = 4ˢ − 2ˢ`, `c₆ = 6ˢ − 3ˢ − 2ˢ + 1`);
/// * the reflection formula for `ψ⁽ⁿ⁾`, `n = s − 1` even, gives
///   `ζ(s, f) − ζ(s, 1−f) = πˢ Pₙ(cot πf) / n!` (DLMF 25.11.12, 5.15.6).
///
/// Even `s` (odd `n`) would need a Dirichlet `L`-value (`β(4)`, …) that has no
/// classical form; those keep the Hurwitz node (`ζ(2, ¼)` is in
/// [`hurwitz_base`] through Catalan's constant).
fn hurwitz_odd_classical(s: u32, f: &Rational, pool: &ExprPool) -> Option<ExprId> {
    if s < 3 || s % 2 == 0 {
        return None;
    }
    let (p, q) = (f.numer().to_i64()?, f.denom().to_i64()?);
    let pw = |b: u32| Integer::from(b).pow(s);
    let c_q = match q {
        3 => pw(3) - 1u32,
        4 => pw(4) - pw(2),
        6 => pw(6) - pw(3) - pw(2) + 1u32,
        _ => return None,
    };
    let sign = |v: i64| if 2 * p < q { v } else { -v };
    // cot(πp/q)
    let cot = match q {
        4 => QSqrt3(Rational::from(sign(1)), Rational::new()),
        3 => QSqrt3(Rational::new(), Rational::from((sign(1), 3))),
        _ => QSqrt3(Rational::new(), Rational::from(sign(1))),
    };
    let n = s - 1;
    let poly = cot_derivative_poly(n);
    let mut acc = QSqrt3(Rational::new(), Rational::new());
    for c in poly.iter().rev() {
        acc = acc.mul(&cot);
        acc.0 += c;
    }
    let scale = Rational::from((Integer::from(1), factorial(n) * 2u32));
    let a = acc.0 * &scale;
    let b = acc.1 * &scale;
    let pis = pool.pow(pi(pool), pool.integer(s));
    let mut terms = vec![mk_mul(
        pool,
        vec![
            rat_node(pool, &Rational::from((c_q, Integer::from(2)))),
            super::node(Kernel::Zeta, vec![pool.integer(s)], pool),
        ],
    )];
    if a != 0 {
        terms.push(mk_mul(pool, vec![rat_node(pool, &a), pis]));
    }
    if b != 0 {
        terms.push(mk_mul(pool, vec![rat_node(pool, &b), sqrt3(pool), pis]));
    }
    Some(mk_add(pool, terms))
}

/// The exact value of `ψ⁽ⁿ⁾(x)` for `n ≥ 1` and a rational literal `x`:
/// `(−1)ⁿ⁺¹ n! ζ(n+1, x)` with `ζ` in its classical form (`πˢ`, `ζ(s)`,
/// Catalan, `√3`) where one exists and the exact Hurwitz node
/// `hurwitz_zeta(n+1, f)` otherwise; `None` at a pole.
pub(crate) fn polygamma_value(n: u32, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if n == 0 {
        return digamma_value(x, pool);
    }
    if i64::from(n) > MAX_POLYGAMMA_ORDER {
        return None;
    }
    let q = rational_literal(pool, x)?;
    if nonpositive_integer(&q).is_some() {
        return None;
    }
    let (f, m) = split(&q);
    if m.abs() > MAX_SHIFT {
        return None;
    }
    let s = n + 1;
    // ψ⁽ⁿ⁾ at any rational: the classical forms where they exist, else the
    // exact Hurwitz value ζ(n+1, f) itself (F26).
    let base = hurwitz_base(s, &f, pool)
        .or_else(|| hurwitz_odd_classical(s, &f, pool))
        .unwrap_or_else(|| {
            super::node(
                Kernel::HurwitzZeta,
                vec![pool.integer(s), rat_node(pool, &f)],
                pool,
            )
        });
    // ζ(s, f+m) = ζ(s, f) − Σ_{j=0}^{m−1} (f+j)^{−s};  ζ(s, f−m) = ζ(s, f) + Σ_{j=1}^{m} (f−j)^{−s}
    let pw = |r: Rational| -> Rational {
        let (a, b) = r.into_numer_denom();
        Rational::from((b.pow(s), a.pow(s)))
    };
    let mut corr = Rational::new();
    if m > 0 {
        for j in 0..m {
            corr -= pw(f.clone() + Rational::from(j));
        }
    } else {
        for j in 1..=(-m) {
            corr += pw(f.clone() - Rational::from(j));
        }
    }
    let hz = if corr == 0 {
        base
    } else {
        mk_add(pool, vec![base, rat_node(pool, &corr)])
    };
    // ψ⁽ⁿ⁾(a) = (−1)ⁿ⁺¹ n! ζ(n+1, a)  (DLMF 25.11.12)
    let mut c = factorial(n);
    if n % 2 == 0 {
        c = -c;
    }
    // Fold `6 · 15 · π⁴/90` to `π⁴`: the value has no polygamma node left, so
    // this cannot recurse back here.
    Some(simplify(mk_mul(pool, vec![pool.integer(c), hz]), pool).value)
}

/// `ψ⁽ⁿ⁾(x)` spelled canonically: `digamma(x)` for `n = 0`, `trigamma(x)` for
/// `n = 1`, `polygamma(n, x)` otherwise — exact where tabulated.
pub(crate) fn polygamma_node(n: u32, x: ExprId, pool: &ExprPool) -> ExprId {
    if let Some(v) = polygamma_value(n, x, pool) {
        return v;
    }
    match n {
        0 => pool.func("digamma", vec![x]),
        1 => pool.func("trigamma", vec![x]),
        _ => pool.func("polygamma", vec![pool.integer(n), x]),
    }
}

/// The polygamma function `ψ⁽ⁿ⁾(x) = dⁿ⁺¹/dxⁿ⁺¹ log Γ(x)` (DLMF 5.15).
///
/// `n = 0` is `digamma(x)` and `n = 1` is `trigamma(x)` (one spelling per
/// function); exact values at the tabulated rationals (module docs). A
/// symbolic `n` keeps the node `polygamma(n, x)`.
///
/// # Errors
///
/// `E-SPECFN-008` for a literal `n` that is not a non-negative integer
/// (`ψ⁽⁻¹⁾` would be `log Γ`, and fractional orders are not built here),
/// `E-SPECFN-006` for a literal `x` at a pole (`0, −1, −2, …`).
pub fn polygamma(n: ExprId, x: ExprId, pool: &ExprPool) -> Result<ExprId, SpecialFunctionError> {
    let bad_order = match pool.get(n) {
        ExprData::Float(_) => true,
        _ => rational_literal(pool, n).is_some_and(|q| *q.denom() != 1 || q < 0),
    };
    if bad_order {
        return Err(SpecialFunctionError::InvalidOrder {
            function: "polygamma",
            value: pool.display(n).to_string(),
        });
    }
    if let Some(q) = rational_literal(pool, x) {
        if nonpositive_integer(&q).is_some() {
            return Err(SpecialFunctionError::Pole {
                function: "polygamma",
                at: format!("x = {}", pool.display(x)),
            });
        }
    }
    crate::budget::check()?;
    Ok(match int_literal(pool, n).and_then(|k| k.to_u32()) {
        Some(k) => polygamma_node(k, x, pool),
        None => pool.func("polygamma", vec![n, x]),
    })
}

// ---------------------------------------------------------------------------
// The polygamma primitive
// ---------------------------------------------------------------------------

/// Registry entry for `polygamma(n, x)`.
///
/// * `simplify`: `n = 0, 1` respell as `digamma` / `trigamma`; exact values.
/// * `diff_forward`: `∂/∂x ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁺¹⁾(x)`; a derivative in `n` has no
///   closed form and declines (`E-DIFF-001`).
/// * `numeric_f64` / `numeric_ball`: Arb's `acb_polygamma` at a real point,
///   accepted only when the imaginary part is an exact zero.
pub struct PolygammaPrimitive;

impl Primitive for PolygammaPrimitive {
    fn name(&self) -> &'static str {
        "polygamma"
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        match args {
            [n, x] => format!("ψ^({})({})", pool.display(*n), pool.display(*x)),
            _ => "polygamma(?)".into(),
        }
    }

    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        let [n, x] = args else { return None };
        let k = int_literal(pool, *n)?.to_u32()?;
        let out = polygamma_node(k, *x, pool);
        (out != pool.func("polygamma", vec![*n, *x])).then_some(out)
    }

    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        let [n, x] = args else { return None };
        if crate::kernel::subs::mentions_var(*n, wrt, pool) {
            return None;
        }
        let dx = crate::diff::diff(*x, wrt, pool).ok()?.value;
        let next = match int_literal(pool, *n).and_then(|k| k.to_u32()) {
            Some(k) => polygamma_node(k + 1, *x, pool),
            None => pool.func("polygamma", vec![pool.add(vec![*n, pool.integer(1)]), *x]),
        };
        Some(mk_mul(pool, vec![next, dx]))
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        match args {
            [n, x] => polygamma_f64(*n, *x),
            _ => None,
        }
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        match args {
            [n, x] => polygamma_ball(n, x),
            _ => None,
        }
    }
}

/// An `f64` that is a non-negative integer order.
fn order_of(n: f64) -> Option<u32> {
    (n.is_finite() && n >= 0.0 && n == n.trunc() && n <= MAX_POLYGAMMA_ORDER as f64)
        .then_some(n as u32)
}

/// `ψ⁽ⁿ⁾(x)` in `f64`, certified to 53 bits by Arb; `None` at a pole or
/// without Arb.
pub(crate) fn polygamma_f64(n: f64, x: f64) -> Option<f64> {
    let k = order_of(n)?;
    if !x.is_finite() || (x <= 0.0 && x == x.trunc()) {
        return None;
    }
    #[cfg(flint_arb)]
    {
        arb_poly::f64_value(k, x)
    }
    #[cfg(not(flint_arb))]
    {
        let _ = k;
        None
    }
}

/// A rigorous enclosure of `ψ⁽ⁿ⁾(x)` for an exact integer order ball `n`.
pub(crate) fn polygamma_ball(n: &ArbBall, x: &ArbBall) -> Option<ArbBall> {
    if !n.rad.is_zero() {
        return None;
    }
    let k = order_of(n.mid.to_f64())?;
    #[cfg(flint_arb)]
    {
        arb_poly::ball_value(k, x)
    }
    #[cfg(not(flint_arb))]
    {
        let _ = (k, x);
        None
    }
}

/// Complex `f64` values of `gamma(z)`, `digamma(z)`, `trigamma(z)` and
/// `polygamma(n, z)` (integer `n ≥ 0`) from Arb's `acb_gamma` /
/// `acb_polygamma`, certified to 53 bits; `None` at a pole, for another name
/// or without Arb. Used by [`crate::eval::eval_complex_f64`] (F26: the Gamma
/// coefficients of the ₂F₁ connection formulas are evaluated there).
pub(crate) fn gamma_family_c64(name: &str, args: &[(f64, f64)]) -> Option<(f64, f64)> {
    let (k, z) = match (name, args) {
        ("gamma", [z]) => (None, *z),
        ("digamma", [z]) => (Some(0), *z),
        ("trigamma", [z]) => (Some(1), *z),
        ("polygamma", [(n, 0.0), z]) => (Some(order_of(*n)?), *z),
        _ => return None,
    };
    if !z.0.is_finite() || !z.1.is_finite() || (z.1 == 0.0 && z.0 <= 0.0 && z.0 == z.0.trunc()) {
        return None;
    }
    #[cfg(flint_arb)]
    {
        arb_poly::c64_value(k, z)
    }
    #[cfg(not(flint_arb))]
    {
        let _ = (k, z);
        None
    }
}

#[cfg(flint_arb)]
mod arb_poly {
    use super::super::numeric::arb_impl::{
        available, certified_c64, certified_f64, from_arb, to_arb,
    };
    use crate::ball::ArbBall;
    use crate::flint::acb::{Acb, Arb};
    use crate::flint::arb as ffi;

    fn call(k: u32, x: &Arb, prec: i64) -> Option<Arb> {
        let s = Acb::from_parts(&Arb::from_i64(i64::from(k)), &Arb::from_i64(0));
        let z = Acb::from_parts(x, &Arb::from_i64(0));
        let mut out = Acb::new();
        unsafe { ffi::acb_polygamma(out.as_mut_ptr(), s.as_ptr(), z.as_ptr(), prec) };
        if !out.is_finite() {
            return None;
        }
        // A real argument gives a real value: the imaginary part is an
        // enclosure of 0 (not always an exact one off the positive axis).
        let im = out.imag();
        if !im.contains_zero() {
            return None;
        }
        Some(out.real())
    }

    pub(super) fn f64_value(k: u32, x: f64) -> Option<f64> {
        if !available() {
            return None;
        }
        let xb = Arb::from_f64(x);
        for prec in [128_i64, 256, 512] {
            if let Some(v) = call(k, &xb, prec).as_ref().and_then(certified_f64) {
                return Some(v);
            }
        }
        None
    }

    pub(super) fn ball_value(k: u32, x: &ArbBall) -> Option<ArbBall> {
        if !available() {
            return None;
        }
        let xb = to_arb(x)?;
        let v = call(k, &xb, i64::from(x.prec) + 16)?;
        from_arb(&v, x.prec)
    }

    /// `Γ(z)` (`k = None`) or `ψ⁽ᵏ⁾(z)` at a complex point.
    pub(super) fn c64_value(k: Option<u32>, z: (f64, f64)) -> Option<(f64, f64)> {
        if !available() {
            return None;
        }
        let zb = Acb::from_f64(z.0, z.1);
        for prec in [128_i64, 256, 512] {
            let mut out = Acb::new();
            unsafe {
                match k {
                    None => ffi::acb_gamma(out.as_mut_ptr(), zb.as_ptr(), prec),
                    Some(k) => {
                        let s = Acb::from_f64(f64::from(k), 0.0);
                        ffi::acb_polygamma(out.as_mut_ptr(), s.as_ptr(), zb.as_ptr(), prec)
                    }
                }
            }
            if let Some(v) = certified_c64(&out) {
                return Some(v);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Laurent expansion at the poles
// ---------------------------------------------------------------------------

/// Does `e` contain a float literal anywhere? (The regression check that no
/// exact path here emits a float `γ`.)
#[cfg(test)]
pub(crate) fn contains_float(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Float(_) => true,
        ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
            a.iter().any(|&x| contains_float(x, pool))
        }
        ExprData::Pow { base, exp } => contains_float(base, pool) || contains_float(exp, pool),
        ExprData::BigO(x) => contains_float(x, pool),
        _ => false,
    }
}

/// The gamma-family function at the head of `e`: `(order, argument)` with
/// order `None` for `Γ` and `Some(n)` for `ψ⁽ⁿ⁾`.
fn gamma_family(e: ExprId, pool: &ExprPool) -> Option<(Option<u32>, ExprId)> {
    let (name, args) = pool.with(e, |d| match d {
        ExprData::Func { name, args } => Some((name.clone(), args.clone())),
        _ => None,
    })?;
    match (name.as_str(), args.as_slice()) {
        ("gamma", [x]) => Some((None, *x)),
        ("digamma", [x]) => Some((Some(0), *x)),
        ("trigamma", [x]) => Some((Some(1), *x)),
        ("polygamma", [n, x]) => {
            let k = int_literal(pool, *n)?.to_u32()?;
            (i64::from(k) <= MAX_POLYGAMMA_ORDER).then_some((Some(k), *x))
        }
        _ => None,
    }
}

/// `k` when `u` tends to the pole `−k` as `var → 0` (and depends on `var`).
fn pole_index(u: ExprId, var: ExprId, pool: &ExprPool) -> Option<i64> {
    if !crate::kernel::subs::mentions_var(u, var, pool) {
        return None;
    }
    let mut map = std::collections::HashMap::new();
    map.insert(var, pool.integer(0));
    let at = simplify(subs(u, &map, pool), pool).value;
    let q = rational_literal(pool, at)?;
    let k = nonpositive_integer(&q)?.to_i64()?;
    (k <= MAX_POLE_SHIFT).then_some(k)
}

/// One gamma-family node shifted off its pole `−k` (module docs).
fn shifted_node(order: Option<u32>, u: ExprId, k: i64, pool: &ExprPool) -> ExprId {
    let plus = |j: i64| {
        if j == 0 {
            u
        } else {
            pool.add(vec![u, pool.integer(j)])
        }
    };
    let up = plus(k + 1);
    match order {
        None => {
            // Γ(u) = Γ(u+k+1) / Π_{j=0}^{k} (u+j)
            let mut f = vec![pool.func("gamma", vec![up])];
            for j in 0..=k {
                f.push(pool.pow(plus(j), pool.integer(-1)));
            }
            pool.mul(f)
        }
        Some(n) => {
            // ψ⁽ⁿ⁾(u) = ψ⁽ⁿ⁾(u+k+1) − (−1)ⁿ n! Σ_{j=0}^{k} (u+j)^{−n−1}
            let mut c = factorial(n);
            if n % 2 == 0 {
                c = -c;
            }
            let e = pool.integer(-(i64::from(n)) - 1);
            let mut t = vec![polygamma_node(n, up, pool)];
            for j in 0..=k {
                t.push(pool.mul(vec![pool.integer(c.clone()), pool.pow(plus(j), e)]));
            }
            pool.add(t)
        }
    }
}

/// Rewrite every `Γ`, `ψ`, `ψ⁽ⁿ⁾` in `expr` whose argument tends to a pole
/// `0, −1, −2, …` as `var → 0` into the analytic function times / plus its
/// principal part (module docs), or `None` when there is none.
///
/// Both sides agree on a punctured neighbourhood of `var = 0`, which is all a
/// Laurent expansion there looks at.
pub(crate) fn shift_off_poles(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    fn rec(
        e: ExprId,
        var: ExprId,
        pool: &ExprPool,
        memo: &mut std::collections::HashMap<ExprId, ExprId>,
    ) -> ExprId {
        if let Some(&r) = memo.get(&e) {
            return r;
        }
        let out = match pool.get(e) {
            ExprData::Add(a) => {
                let b: Vec<ExprId> = a.iter().map(|&x| rec(x, var, pool, memo)).collect();
                if b == a {
                    e
                } else {
                    pool.add(b)
                }
            }
            ExprData::Mul(a) => {
                let b: Vec<ExprId> = a.iter().map(|&x| rec(x, var, pool, memo)).collect();
                if b == a {
                    e
                } else {
                    pool.mul(b)
                }
            }
            ExprData::Pow { base, exp } => {
                let (b, x) = (rec(base, var, pool, memo), rec(exp, var, pool, memo));
                if (b, x) == (base, exp) {
                    e
                } else {
                    pool.pow(b, x)
                }
            }
            ExprData::Func { name, args } => {
                let b: Vec<ExprId> = args.iter().map(|&x| rec(x, var, pool, memo)).collect();
                let f = if b == args { e } else { pool.func(name, b) };
                match gamma_family(f, pool) {
                    Some((order, u)) => match pole_index(u, var, pool) {
                        Some(k) => shifted_node(order, u, k, pool),
                        None => f,
                    },
                    None => f,
                }
            }
            _ => e,
        };
        memo.insert(e, out);
        out
    }
    let mut memo = std::collections::HashMap::new();
    let out = rec(expr, var, pool, &mut memo);
    (out != expr).then_some(out)
}
