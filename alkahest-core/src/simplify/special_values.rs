//! Constant folding and special values (G4).
//!
//! The default rule set folds arithmetic on literals, but before this module
//! it stopped at the first function or radical: `sqrt(2)^2`, `cos(pi)`,
//! `erf(-x) + erf(x)`, `gamma(1/2)`, `exp(-x)·exp(x)`, `(mu·I)^2` all came
//! back as written.  Every rewrite here is an identity that holds for every
//! complex value of the free symbols (or for every value the symbol's declared
//! domain admits), so none of them needs a side condition:
//!
//! * [`SpecialValue`] — trig / inverse-trig at rational multiples of `π` for
//!   the standard table (denominators 1, 2, 3, 4, 6), `sin(nπ) = 0` and
//!   `cos(nπ) = (−1)^n` for integer-valued `n`, `exp(iπq)`, `Γ` at
//!   half-integers, `erf(0)`, `erfc(0)`, and the hyperbolic zeros.
//! * [`RadicalPower`] — `(√a)^(2k) = a^k` (literal or provably non-negative
//!   `a`), `(a^(p/q))^n = a^(pn/q)` for integer `n`, and `√(k²m) = k√m` (and
//!   `q`-th roots) for positive literals.
//! * [`PowMulIntExp`] — `(a·i)^n = a^n·i^n`, so `(μ·i)^2` reaches `−μ^2`.
//! * [`PowMulCancel`] — `(a·b)^n` distributed inside a product holding `a`,
//!   so `(S/K)·(S/K)⁻¹` cancels.
//! * [`SqrtLiteralProduct`] — `√a·√b = √(ab)` for positive literals.
//! * [`ExpCancel`] — `exp(a)·exp(b) = exp(a+b)` when the exponents share a
//!   term, so `exp(−x)·exp(x)` reaches `1`.
//!
//! [`OddEvenArg`] (`f(−u) = ∓f(u)`, valid on all of ℂ for the listed
//! functions) is opt-in: it is *not* in the default set, because re-spelling
//! `sin(−x)` as `−sin(x)` everywhere changes the shape every downstream
//! pattern matcher sees.  The zero test applies the same symmetry internally,
//! so `erf(−x) + erf(x)` still reaches `0`.

use super::rules::{as_rational, one_step, NodeKinds, RewriteRule};
use crate::deriv::log::DerivationLog;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use rug::ops::Pow;
use rug::{Integer, Rational};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn intern_rational(r: Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.into_numer_denom().0)
    } else {
        pool.intern(ExprData::Rational(crate::kernel::expr::BigRat(r)))
    }
}

fn is_pi(e: ExprId, pool: &ExprPool) -> bool {
    crate::eval::symbols::is_pi(e, pool)
}

fn pi(pool: &ExprPool) -> ExprId {
    crate::eval::symbols::pi_symbol(pool)
}

fn sqrt_of(n: i64, pool: &ExprPool) -> ExprId {
    pool.func("sqrt", vec![pool.integer(n)])
}

/// `c · body`, folding `c = 1` and `c = 0`.
fn scaled(c: Rational, body: ExprId, pool: &ExprPool) -> ExprId {
    if c == 0 {
        return pool.integer(0_i32);
    }
    if c == 1 {
        return body;
    }
    pool.mul(vec![intern_rational(c, pool), body])
}

/// `q·π` as an expression.
fn pi_times(q: Rational, pool: &ExprPool) -> ExprId {
    scaled(q, pi(pool), pool)
}

/// Whether `e` takes only integer values: integer literals, `Domain::Integer`
/// symbols, and sums / products / non-negative integer powers of those.
fn is_integer_valued(e: ExprId, pool: &ExprPool, depth: u32) -> bool {
    if depth == 0 {
        return false;
    }
    enum K {
        Yes,
        No,
        All(Vec<ExprId>),
        Pow(ExprId, ExprId),
    }
    let k = pool.with(e, |d| match d {
        ExprData::Integer(_) => K::Yes,
        ExprData::Symbol {
            domain: Domain::Integer,
            ..
        } => K::Yes,
        ExprData::Add(a) | ExprData::Mul(a) => K::All(a.clone()),
        ExprData::Pow { base, exp } => K::Pow(*base, *exp),
        _ => K::No,
    });
    match k {
        K::Yes => true,
        K::No => false,
        K::All(a) => a.iter().all(|&x| is_integer_valued(x, pool, depth - 1)),
        K::Pow(b, x) => {
            pool.with(x, |d| matches!(d, ExprData::Integer(n) if n.0 >= 0))
                && is_integer_valued(b, pool, depth - 1)
        }
    }
}

/// An argument of the shape `q·π` or `q·n·π`.
enum PiMultiple {
    /// `q·π` for a rational literal `q`.
    Rational(Rational),
    /// `q·n·π` with `n` integer-valued (and not a literal).
    IntegerValued(Rational, ExprId),
}

/// Split `arg` as a multiple of `π`.
fn pi_multiple(arg: ExprId, pool: &ExprPool) -> Option<PiMultiple> {
    if is_pi(arg, pool) {
        return Some(PiMultiple::Rational(Rational::from(1)));
    }
    let factors = pool.with(arg, |d| match d {
        ExprData::Mul(a) => Some(a.clone()),
        _ => None,
    })?;
    let mut q = Rational::from(1);
    let mut saw_pi = false;
    let mut rest = Vec::new();
    for f in factors {
        if is_pi(f, pool) {
            if saw_pi {
                return None;
            }
            saw_pi = true;
        } else if let Some(r) = as_rational(f, pool) {
            q *= r;
        } else {
            rest.push(f);
        }
    }
    if !saw_pi {
        return None;
    }
    if rest.is_empty() {
        return Some(PiMultiple::Rational(q));
    }
    let n = if rest.len() == 1 {
        rest[0]
    } else {
        pool.mul(rest)
    };
    if is_integer_valued(n, pool, 6) {
        Some(PiMultiple::IntegerValued(q, n))
    } else {
        None
    }
}

/// `q mod m` in `[0, m)`.
fn rat_mod(q: &Rational, m: i64) -> Rational {
    let m = Rational::from(m);
    let k = Rational::from(q / &m).floor();
    q - k * m
}

/// Exact `sin(qπ)` for `q` with denominator 1, 2, 3, 4 or 6.
pub(super) fn sin_pi(q: &Rational, pool: &ExprPool) -> Option<ExprId> {
    let den = q.denom().to_u32()?;
    if !matches!(den, 1 | 2 | 3 | 4 | 6) {
        return None;
    }
    let r = rat_mod(q, 2);
    let (r, sign) = if r >= 1 {
        (Rational::from(&r - 1), -1)
    } else {
        (r, 1)
    };
    // r ∈ [0, 1): sin(rπ) ≥ 0, symmetric about 1/2.
    let r = if r > Rational::from((1, 2)) { 1 - r } else { r };
    let half = Rational::from((1, 2));
    let v: ExprId = if r == 0 {
        pool.integer(0_i32)
    } else if r == Rational::from((1, 6)) {
        pool.rational(1, 2)
    } else if r == Rational::from((1, 4)) {
        scaled(half, sqrt_of(2, pool), pool)
    } else if r == Rational::from((1, 3)) {
        scaled(half, sqrt_of(3, pool), pool)
    } else if r == half {
        pool.integer(1_i32)
    } else {
        return None;
    };
    Some(if sign < 0 { negate(v, pool) } else { v })
}

fn negate(v: ExprId, pool: &ExprPool) -> ExprId {
    if let Some(r) = as_rational(v, pool) {
        return intern_rational(-r, pool);
    }
    pool.mul(vec![pool.integer(-1_i32), v])
}

/// Exact `cos(qπ) = sin((q + 1/2)π)`.
pub(super) fn cos_pi(q: &Rational, pool: &ExprPool) -> Option<ExprId> {
    sin_pi(&(q + Rational::from((1, 2))), pool)
}

/// Exact `tan(qπ)`; `None` at the poles.
fn tan_pi(q: &Rational, pool: &ExprPool) -> Option<ExprId> {
    let den = q.denom().to_u32()?;
    if !matches!(den, 1 | 3 | 4 | 6) {
        return None;
    }
    let r = rat_mod(q, 1);
    let sqrt3 = sqrt_of(3, pool);
    let third = Rational::from((1, 3));
    Some(if r == 0 {
        pool.integer(0_i32)
    } else if r == Rational::from((1, 6)) {
        scaled(third, sqrt3, pool)
    } else if r == Rational::from((1, 4)) {
        pool.integer(1_i32)
    } else if r == third {
        sqrt3
    } else if r == Rational::from((2, 3)) {
        negate(sqrt3, pool)
    } else if r == Rational::from((3, 4)) {
        pool.integer(-1_i32)
    } else if r == Rational::from((5, 6)) {
        scaled(-third, sqrt3, pool)
    } else {
        return None;
    })
}

/// `c·√n` for a rational literal `c` and a positive integer `n`
/// (`sqrt(n)` or `n^(1/2)`); a bare literal is `(c, 1)`.
fn literal_times_sqrt(v: ExprId, pool: &ExprPool) -> Option<(Rational, u64)> {
    if let Some(r) = as_rational(v, pool) {
        return Some((r, 1));
    }
    let sqrt_arg = |e: ExprId| -> Option<u64> {
        pool.with(e, |d| match d {
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                pool.with(args[0], |a| match a {
                    ExprData::Integer(n) if n.0 > 0 => n.0.to_u64(),
                    _ => None,
                })
            }
            ExprData::Pow { base, exp } => {
                let half = pool.with(
                    *exp,
                    |x| matches!(x, ExprData::Rational(r) if *r.0.numer() == 1 && *r.0.denom() == 2),
                );
                if !half {
                    return None;
                }
                pool.with(*base, |a| match a {
                    ExprData::Integer(n) if n.0 > 0 => n.0.to_u64(),
                    _ => None,
                })
            }
            _ => None,
        })
    };
    if let Some(n) = sqrt_arg(v) {
        return Some((Rational::from(1), n));
    }
    let fs = pool.with(v, |d| match d {
        ExprData::Mul(a) if a.len() == 2 => Some(a.clone()),
        _ => None,
    })?;
    let (c, s) = if let Some(c) = as_rational(fs[0], pool) {
        (c, fs[1])
    } else {
        (as_rational(fs[1], pool)?, fs[0])
    };
    Some((c, sqrt_arg(s)?))
}

/// `|v|` and whether `v` was negative, for literal-times-radical values.
fn abs_value(v: ExprId, pool: &ExprPool) -> Option<(Rational, u64, bool)> {
    let (c, n) = literal_times_sqrt(v, pool)?;
    if c < 0 {
        Some((-c, n, true))
    } else {
        Some((c, n, false))
    }
}

/// `asin(v)/π` for `v ≥ 0` in the standard table.
fn asin_over_pi(c: &Rational, n: u64) -> Option<Rational> {
    let half = Rational::from((1, 2));
    match n {
        1 if *c == 0 => Some(Rational::from(0)),
        1 if *c == half => Some(Rational::from((1, 6))),
        1 if *c == 1 => Some(Rational::from((1, 2))),
        2 if *c == half => Some(Rational::from((1, 4))),
        3 if *c == half => Some(Rational::from((1, 3))),
        _ => None,
    }
}

/// `atan(v)/π` for `v ≥ 0` in the standard table.
fn atan_over_pi(c: &Rational, n: u64) -> Option<Rational> {
    match n {
        1 if *c == 0 => Some(Rational::from(0)),
        1 if *c == 1 => Some(Rational::from((1, 4))),
        3 if *c == 1 => Some(Rational::from((1, 3))),
        3 if *c == Rational::from((1, 3)) => Some(Rational::from((1, 6))),
        _ => None,
    }
}

/// `n!` for small `n`.
fn factorial(n: u32) -> Integer {
    Integer::from(Integer::factorial(n))
}

/// `Γ(q)` for a half-integer `q` with `|q| ≤ 60`: a rational multiple of `√π`.
pub(crate) fn gamma_half_integer(q: &Rational, pool: &ExprPool) -> Option<ExprId> {
    if *q.denom() != 2 {
        return None;
    }
    // q = 1/2 + n.
    let n = q - Rational::from((1, 2));
    let n = n.numer().to_i32()?;
    if n.abs() > 60 {
        return None;
    }
    let coeff = if n >= 0 {
        // Γ(n + 1/2) = (2n)! / (4^n n!) · √π
        let n = n as u32;
        Rational::from((factorial(2 * n), Integer::from(4).pow(n) * factorial(n)))
    } else {
        // Γ(1/2 − m) = (−4)^m m! / (2m)! · √π
        let m = (-n) as u32;
        Rational::from((Integer::from(-4).pow(m) * factorial(m), factorial(2 * m)))
    };
    let sqrt_pi = pool.func("sqrt", vec![pi(pool)]);
    Some(scaled(coeff, sqrt_pi, pool))
}

fn is_literal(e: ExprId, pool: &ExprPool, v: i32) -> bool {
    pool.with(e, |d| matches!(d, ExprData::Integer(n) if n.0 == v))
}

// ---------------------------------------------------------------------------
// SpecialValue
// ---------------------------------------------------------------------------

/// Exact values of elementary and special functions at special arguments.
/// See the module documentation for the table.
pub struct SpecialValue;

impl SpecialValue {
    fn value(name: &str, arg: ExprId, pool: &ExprPool) -> Option<ExprId> {
        match name {
            "sin" | "cos" | "tan" => match pi_multiple(arg, pool)? {
                PiMultiple::Rational(q) => match name {
                    "sin" => sin_pi(&q, pool),
                    "cos" => cos_pi(&q, pool),
                    _ => tan_pi(&q, pool),
                },
                // q·n·π with n an integer: only an integer q is periodic
                // enough to decide.
                PiMultiple::IntegerValued(q, n) => {
                    if *q.denom() != 1 {
                        return None;
                    }
                    match name {
                        "sin" | "tan" => Some(pool.integer(0_i32)),
                        _ => {
                            if q.numer().is_even() {
                                Some(pool.integer(1_i32))
                            } else {
                                Some(pool.pow(pool.integer(-1_i32), n))
                            }
                        }
                    }
                }
            },
            "exp" => {
                // exp(i·q·π) and exp(i·q·n·π).
                let factors = pool.with(arg, |d| match d {
                    ExprData::Mul(a) => Some(a.clone()),
                    _ => None,
                })?;
                let i_pos = factors.iter().position(|&f| pool.is_imaginary_unit(f))?;
                let rest: Vec<ExprId> = factors
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != i_pos)
                    .map(|(_, &f)| f)
                    .collect();
                let rest = if rest.len() == 1 {
                    rest[0]
                } else {
                    pool.mul(rest)
                };
                match pi_multiple(rest, pool)? {
                    PiMultiple::Rational(q) => {
                        let den = q.denom().to_u32()?;
                        if !matches!(den, 1 | 2) {
                            // cos + i·sin would be a sum of radicals; keep
                            // the exponential, which is the shorter form.
                            return None;
                        }
                        let c = cos_pi(&q, pool)?;
                        let s = sin_pi(&q, pool)?;
                        let i = pool.imaginary_unit();
                        Some(if is_literal(s, pool, 0) {
                            c
                        } else if is_literal(c, pool, 0) {
                            if is_literal(s, pool, 1) {
                                i
                            } else {
                                pool.mul(vec![pool.integer(-1_i32), i])
                            }
                        } else {
                            return None;
                        })
                    }
                    PiMultiple::IntegerValued(q, n) => {
                        if *q.denom() != 1 {
                            return None;
                        }
                        if q.numer().is_even() {
                            Some(pool.integer(1_i32))
                        } else {
                            Some(pool.pow(pool.integer(-1_i32), n))
                        }
                    }
                }
            }
            "asin" | "acos" | "atan" => {
                let (c, n, neg) = abs_value(arg, pool)?;
                let q = match name {
                    "atan" => atan_over_pi(&c, n)?,
                    _ => asin_over_pi(&c, n)?,
                };
                let q = match (name, neg) {
                    // asin, atan are odd.
                    ("asin" | "atan", false) => q,
                    ("asin" | "atan", true) => -q,
                    // acos(v) = π/2 − asin(v); acos(−v) = π − acos(v).
                    (_, false) => Rational::from((1, 2)) - q,
                    (_, true) => Rational::from((1, 2)) + q,
                };
                Some(pi_times(q, pool))
            }
            "erf" | "tanh" | "asinh" | "atanh" if is_literal(arg, pool, 0) => {
                Some(pool.integer(0_i32))
            }
            "erfc" if is_literal(arg, pool, 0) => Some(pool.integer(1_i32)),
            "acosh" if is_literal(arg, pool, 1) => Some(pool.integer(0_i32)),
            "gamma" => {
                let q = as_rational(arg, pool)?;
                gamma_half_integer(&q, pool)
            }
            _ => None,
        }
    }
}

impl RewriteRule for SpecialValue {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::FUNC
    }
    fn name(&self) -> &'static str {
        "special_value"
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (name, arg) = pool.with(expr, |d| match d {
            ExprData::Func { name, args } if args.len() == 1 => {
                let interesting = matches!(
                    name.as_str(),
                    "sin"
                        | "cos"
                        | "tan"
                        | "exp"
                        | "asin"
                        | "acos"
                        | "atan"
                        | "erf"
                        | "erfc"
                        | "tanh"
                        | "asinh"
                        | "atanh"
                        | "acosh"
                        | "gamma"
                );
                interesting.then(|| (name.clone(), args[0]))
            }
            _ => None,
        })?;
        let after = Self::value(&name, arg, pool)?;
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// OddEvenArg
// ---------------------------------------------------------------------------

/// `f(−u) → −f(u)` for odd `f`, `f(−u) → f(u)` for even `f`.
pub struct OddEvenArg;

pub(super) fn parity(name: &str) -> Option<bool> {
    // Some(true) = odd, Some(false) = even.
    match name {
        "sin" | "tan" | "sinh" | "tanh" | "asin" | "atan" | "asinh" | "atanh" | "erf" | "Si"
        | "Shi" | "fresnels" | "fresnelc" => Some(true),
        "cos" | "cosh" | "abs" => Some(false),
        _ => None,
    }
}

/// `u` when `arg = c·u` with a negative literal `c` (or `arg` is a negative
/// literal), with `c` replaced by `−c`.
pub(super) fn strip_negative(arg: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if let Some(r) = as_rational(arg, pool) {
        return (r < 0).then(|| intern_rational(-r, pool));
    }
    let factors = pool.with(arg, |d| match d {
        ExprData::Mul(a) => Some(a.clone()),
        _ => None,
    })?;
    let pos = factors
        .iter()
        .position(|&f| matches!(as_rational(f, pool), Some(r) if r < 0))?;
    let c = -as_rational(factors[pos], pool)?;
    let mut rest: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != pos)
        .map(|(_, &f)| f)
        .collect();
    if c != 1 {
        rest.push(intern_rational(c, pool));
    }
    Some(match rest.len() {
        0 => pool.integer(1_i32),
        1 => rest[0],
        _ => pool.mul(rest),
    })
}

impl RewriteRule for OddEvenArg {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::FUNC
    }
    fn name(&self) -> &'static str {
        "odd_even_symmetry"
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (name, odd, arg) = pool.with(expr, |d| match d {
            ExprData::Func { name, args } if args.len() == 1 => {
                parity(name).map(|odd| (name.clone(), odd, args[0]))
            }
            _ => None,
        })?;
        let inner = strip_negative(arg, pool)?;
        let f = pool.func(name, vec![inner]);
        let after = if odd {
            pool.mul(vec![pool.integer(-1_i32), f])
        } else {
            f
        };
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// RadicalPower
// ---------------------------------------------------------------------------

/// Powers of radicals and perfect-power extraction from literal radicands.
pub struct RadicalPower;

/// Largest `k` with `k^q | n`, by trial division over small primes; returns
/// `(k, n / k^q)`.  Primes past the trial bound stay in the cofactor, which
/// is always correct, only possibly not fully reduced.
fn extract_perfect_power(n: &Integer, q: u32) -> (Integer, Integer) {
    let mut k = Integer::from(1);
    let mut m = n.clone();
    if m.significant_bits() > 256 {
        return (k, m);
    }
    let mut p: u32 = 2;
    while p < 1000 {
        let pq = Integer::from(p).pow(q);
        if pq > m {
            break;
        }
        while m.is_divisible(&pq) {
            m /= &pq;
            k *= p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    // A cofactor that is itself a perfect q-th power.
    if m > 1 {
        let (root, rem) = m.clone().root_rem(Integer::new(), q);
        if rem == 0 {
            k *= &root;
            m = Integer::from(1);
        }
    }
    (k, m)
}

/// [`extract_perfect_power`] for a machine-word `n`, without allocating —
/// the same trial bound and the same final perfect-power test, so the same
/// `(k, n / k^q)`.
fn extract_perfect_power_u64(n: u64, q: u32) -> (u64, u64) {
    let mut k: u64 = 1;
    let mut m = n;
    let mut p: u64 = 2;
    while p < 1000 {
        let Some(pq) = p.checked_pow(q) else {
            break;
        };
        if pq > m {
            break;
        }
        while m % pq == 0 {
            m /= pq;
            k *= p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if m > 1 {
        // Exact integer q-th root, if any: the float estimate is within one
        // of it for every u64, and the candidates are checked exactly.
        let est = (m as f64).powf(1.0 / f64::from(q)).round() as u64;
        for r in est.saturating_sub(1)..=est.saturating_add(1) {
            if r > 1 && r.checked_pow(q) == Some(m) {
                k *= r;
                m = 1;
                break;
            }
        }
    }
    (k, m)
}

impl RadicalPower {
    /// `(base, p/q)` for `sqrt(base)` or `base^(p/q)` (non-integer `p/q`).
    fn radical(e: ExprId, pool: &ExprPool) -> Option<(ExprId, Rational, bool)> {
        pool.with(e, |d| match d {
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                Some((args[0], Rational::from((1, 2)), true))
            }
            ExprData::Pow { base, exp } => pool.with(*exp, |x| match x {
                ExprData::Rational(r) => Some((*base, r.0.clone(), false)),
                _ => None,
            }),
            _ => None,
        })
    }

    /// `sqrt(r)` / `r^(1/q)` for a positive literal `r` → `k · m^(1/q)`.
    fn literal_root(base: ExprId, q: u32, spelled_sqrt: bool, pool: &ExprPool) -> Option<ExprId> {
        // Allocation-free decline for a machine-word integer radicand with no
        // q-th power factor (`√5`, met on every visit of every pass): the
        // bignum path below would reach the same `k == 1 && den == 1`.
        if let Some(n) = pool.with(base, |d| match d {
            ExprData::Integer(n) => n.0.to_u64(),
            _ => None,
        }) {
            if n == 0 || q > 12 || extract_perfect_power_u64(n, q).0 == 1 {
                return None;
            }
        }
        let r = as_rational(base, pool)?;
        if r <= 0 || q > 12 {
            return None;
        }
        // r^(1/q) = (num · den^(q−1))^(1/q) / den.
        let (num, den) = r.into_numer_denom();
        let radicand = &num * den.clone().pow(q - 1);
        let (k, m) = extract_perfect_power(&radicand, q);
        if k == 1 && den == 1 {
            return None;
        }
        // A fractional radicand is folded only to an exact rational
        // (`√(9/4) = 3/2`); rationalising `√(1/2)` to `√2/2` is a spelling
        // choice that downstream pattern matchers (the integrator's
        // substitution search) are sensitive to.
        if den != 1 && m != 1 {
            return None;
        }
        let coeff = Rational::from((k, den));
        let unit = m == 1;
        let rest = if unit {
            pool.integer(1_i32)
        } else if spelled_sqrt && q == 2 {
            pool.func("sqrt", vec![pool.integer(m)])
        } else {
            pool.pow(pool.integer(m), pool.rational(1, q))
        };
        Some(if unit {
            intern_rational(coeff, pool)
        } else {
            scaled(coeff, rest, pool)
        })
    }
}

impl RewriteRule for RadicalPower {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::POW.or(NodeKinds::FUNC)
    }
    fn name(&self) -> &'static str {
        "radical_power"
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let after = self.rewrite(expr, pool)?;
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

impl RadicalPower {
    fn rewrite(&self, expr: ExprId, pool: &ExprPool) -> Option<ExprId> {
        enum Shape {
            Sqrt(ExprId),
            Pow(ExprId, ExprId),
        }
        let shape = pool.with(expr, |d| match d {
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                Some(Shape::Sqrt(args[0]))
            }
            ExprData::Pow { base, exp } => Some(Shape::Pow(*base, *exp)),
            _ => None,
        })?;
        match shape {
            Shape::Sqrt(a) => {
                if !pool.with(a, |d| {
                    matches!(d, ExprData::Integer(_) | ExprData::Rational(_))
                }) {
                    return None;
                }
                Self::literal_root(a, 2, true, pool)
            }
            Shape::Pow(base, exp) => {
                // Cheap screen (this is offered every `Pow`): a rational
                // exponent over a literal base, or an integer exponent over a
                // radical.
                let rational_exp = pool.with(exp, |d| matches!(d, ExprData::Rational(_)));
                let int_exp = pool.with(exp, |d| matches!(d, ExprData::Integer(_)));
                let literal_base = pool.with(base, |d| {
                    matches!(d, ExprData::Integer(_) | ExprData::Rational(_))
                });
                let radical_base = pool.with(base, |d| match d {
                    ExprData::Func { name, args } => name == "sqrt" && args.len() == 1,
                    ExprData::Pow { exp, .. } => {
                        pool.with(*exp, |x| matches!(x, ExprData::Rational(_)))
                    }
                    _ => false,
                });
                let screened = (rational_exp && literal_base) || (int_exp && radical_base);
                if !screened {
                    return None;
                }
                // Literal radicand: r^(1/q), and r^(p/q) when r is a
                // perfect q-th power.
                if let Some(e) = as_rational(exp, pool).filter(|_| rational_exp) {
                    if *e.denom() > 1 {
                        if let Some(q) = e.denom().to_u32() {
                            if let Some(v) = Self::literal_root(base, q, false, pool) {
                                if *e.numer() == 1 {
                                    return Some(v);
                                }
                                if as_rational(v, pool).is_some() {
                                    return Some(pool.pow(v, pool.integer(e.numer().clone())));
                                }
                            }
                        }
                    }
                }
                // (radical)^n for an integer n.
                let n = pool.with(exp, |d| match d {
                    ExprData::Integer(n) => Some(n.0.clone()),
                    _ => None,
                })?;
                if n == 0 || n == 1 {
                    return None;
                }
                let (a, r, spelled_sqrt) = Self::radical(base, pool)?;
                let numeric = as_rational(a, pool).is_some();
                let total = Rational::from(&r * &n);
                // `(√a)² = a` holds for every complex `a`, but for a real `a <
                // 0` the left side is not a real number while the right side
                // is — and real-valued callers (the antiderivative gates, the
                // ODE verifier) rely on that difference to refuse.  So a
                // symbolic radicand must be provably non-negative.
                if !numeric && !is_nonneg(a, pool, 6) {
                    return None;
                }
                if *total.denom() == 1 {
                    // (a^(p/q))^n = a^(pn/q), an integer power.
                    let m = total.numer().clone();
                    return Some(if m == 1 {
                        a
                    } else {
                        pool.pow(a, pool.integer(m))
                    });
                }
                if spelled_sqrt {
                    // sqrt(a)^(2k+1) = a^k · sqrt(a) for k ≥ 1.  A negative
                    // odd power is kept as written (`1/sqrt(2)` is not
                    // rationalised to `sqrt(2)/2`): rewriting it in some
                    // outputs and not in others splits one subexpression
                    // into two spellings that codegen CSE cannot share.
                    let k = Integer::from(&n - 1) / 2;
                    if k < 0 {
                        return None;
                    }
                    let ak = if k == 1 {
                        a
                    } else {
                        pool.pow(a, pool.integer(k))
                    };
                    return Some(pool.mul(vec![ak, base]));
                }
                // (a^(p/q))^n = a^(pn/q) for integer n.
                Some(pool.pow(a, intern_rational(total, pool)))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PowMulIntExp
// ---------------------------------------------------------------------------

/// `(a·b·…)^n → a^n·b^n·…` for a literal integer `n`; commutative only.
pub struct PowMulIntExp;

impl RewriteRule for PowMulIntExp {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::POW
    }
    fn name(&self) -> &'static str {
        "pow_of_product"
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (base, exp) = pool.with(expr, |d| match d {
            ExprData::Pow { base, exp } => Some((*base, *exp)),
            _ => None,
        })?;
        let is_int = pool.with(
            exp,
            |d| matches!(d, ExprData::Integer(n) if n.0 != 0 && n.0 != 1),
        );
        if !is_int {
            return None;
        }
        let factors = pool.with(base, |d| match d {
            ExprData::Mul(a) => Some(a.clone()),
            _ => None,
        })?;
        if !pool.is_mult_commutative(base) {
            return None;
        }
        // Only where it lets `i^n` fold: `(μ·i)² → −μ²`.  Distributing every
        // integer power of a product would re-spell `(x·y)^n` throughout the
        // library's outputs; [`PowMulCancel`] covers the cancelling case.
        if !factors.iter().any(|&f| pool.is_imaginary_unit(f)) {
            return None;
        }
        let after = pool.mul(factors.iter().map(|&f| pool.pow(f, exp)).collect());
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// PowMulCancel
// ---------------------------------------------------------------------------

/// Inside a product, `(a·b)^n · a^k · … → a^n·b^n · a^k · …` when the
/// distribution makes factors cancel (fewer distinct bases survive) — what
/// lets `(S/K)·(S/K)⁻¹` reach `1`.  Integer `n`, commutative only.
pub struct PowMulCancel;

impl RewriteRule for PowMulCancel {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::MUL
    }
    fn name(&self) -> &'static str {
        "pow_of_product_cancel"
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        // Cheap screen on the borrowed node: some factor must be an
        // integer power of a product.
        let candidate = pool.with(expr, |d| match d {
            ExprData::Mul(a) if a.len() >= 2 => a.iter().any(|&f| {
                pool.with(f, |fd| match fd {
                    ExprData::Pow { base, exp } => {
                        pool.with(*exp, |x| matches!(x, ExprData::Integer(_)))
                            && pool.with(*base, |b| matches!(b, ExprData::Mul(_)))
                    }
                    _ => false,
                })
            }),
            _ => false,
        });
        if !candidate || !pool.is_mult_commutative(expr) {
            return None;
        }
        let args = pool.with(expr, |d| match d {
            ExprData::Mul(a) => Some(a.clone()),
            _ => None,
        })?;
        for (i, &f) in args.iter().enumerate() {
            let Some((inner, exp)) = pool.with(f, |d| match d {
                ExprData::Pow { base, exp } => Some((*base, *exp)),
                _ => None,
            }) else {
                continue;
            };
            if !pool.with(exp, |d| matches!(d, ExprData::Integer(_))) {
                continue;
            }
            let Some(fs) = pool.with(inner, |d| match d {
                ExprData::Mul(a) => Some(a.clone()),
                _ => None,
            }) else {
                continue;
            };
            // Fire only when distributing makes factors *cancel*: the number
            // of distinct bases with a non-zero total integer exponent must
            // drop.  `y·(y/(y+K))⁻¹ → y+K` and `(S/K)·(S/K)⁻¹ → 1` qualify;
            // a distribution that merely re-spells the product (which the
            // integrator's substitution search is sensitive to) does not.
            let Some(n) = pool.with(exp, |d| match d {
                ExprData::Integer(n) => n.0.to_i64(),
                _ => None,
            }) else {
                continue;
            };
            let mut tally: Vec<(ExprId, i64)> = Vec::new();
            let mut add = |g: ExprId, k: i64| {
                let (b, e) = pool.with(g, |d| match d {
                    ExprData::Pow { base, exp } => match pool.with(*exp, |x| match x {
                        ExprData::Integer(m) => m.0.to_i64(),
                        _ => None,
                    }) {
                        Some(m) => (*base, m),
                        None => (g, 1),
                    },
                    _ => (g, 1),
                });
                let e = e.saturating_mul(k);
                match tally.iter_mut().find(|(x, _)| *x == b) {
                    Some((_, acc)) => *acc = acc.saturating_add(e),
                    None => tally.push((b, e)),
                }
            };
            let mut literals = 0usize;
            for (j, &g) in args.iter().enumerate() {
                if as_rational(g, pool).is_some() {
                    literals += 1;
                } else if j != i {
                    add(g, 1);
                }
            }
            for &h in &fs {
                if as_rational(h, pool).is_some() {
                    literals += 1;
                } else {
                    add(h, n);
                }
            }
            let after_count = tally.iter().filter(|(_, e)| *e != 0).count() + literals.min(1);
            if after_count >= args.len() {
                continue;
            }
            let mut out: Vec<ExprId> = Vec::with_capacity(args.len() + fs.len());
            for (j, &g) in args.iter().enumerate() {
                if j == i {
                    out.extend(fs.iter().map(|&h| pool.pow(h, exp)));
                } else {
                    out.push(g);
                }
            }
            let after = pool.mul(out);
            if after == expr {
                continue;
            }
            return Some((after, one_step(self.name(), expr, after)));
        }
        None
    }
}

// ---------------------------------------------------------------------------
// ExpCancel
// ---------------------------------------------------------------------------

/// `exp(a)·exp(b) → exp(a + b)` inside a product — but only when the
/// exponents share a term, so the merge cancels or combines something
/// (`exp(−x)·exp(x) → 1`, `exp(−rT)·exp(rT + a) → exp(a)`).  Unrelated
/// exponentials (`exp(x)·exp(y)`) keep their spelling.
pub struct ExpCancel;

/// `arg` as `Σ c_i · core_i`.
fn linear_parts(arg: ExprId, pool: &ExprPool) -> Vec<(Rational, ExprId)> {
    let terms = pool.with(arg, |d| match d {
        ExprData::Add(a) => a.clone(),
        _ => vec![arg],
    });
    terms
        .into_iter()
        .map(|t| {
            if let Some(r) = as_rational(t, pool) {
                return (r, pool.integer(1_i32));
            }
            match pool.with(t, |d| match d {
                ExprData::Mul(a) => Some(a.clone()),
                _ => None,
            }) {
                Some(fs) => {
                    let mut c = Rational::from(1);
                    let mut rest = Vec::new();
                    for f in fs {
                        match as_rational(f, pool) {
                            Some(r) => c *= r,
                            None => rest.push(f),
                        }
                    }
                    let core = match rest.len() {
                        0 => pool.integer(1_i32),
                        1 => rest[0],
                        _ => pool.mul(rest),
                    };
                    (c, core)
                }
                None => (Rational::from(1), t),
            }
        })
        .collect()
}

impl RewriteRule for ExpCancel {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::MUL
    }
    fn name(&self) -> &'static str {
        "exp_product_cancel"
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let n_exp = pool.with(expr, |d| match d {
            ExprData::Mul(a) => a
                .iter()
                .filter(|&&f| {
                    pool.with(f, |fd| {
                        matches!(fd, ExprData::Func { name, args } if name == "exp" && args.len() == 1)
                    })
                })
                .count(),
            _ => 0,
        });
        if n_exp < 2 {
            return None;
        }
        let args = pool.with(expr, |d| match d {
            ExprData::Mul(a) => Some(a.clone()),
            _ => None,
        })?;
        let mut exps = Vec::new();
        let mut other = Vec::new();
        for &a in &args {
            match pool.with(a, |d| match d {
                ExprData::Func { name, args } if name == "exp" && args.len() == 1 => Some(args[0]),
                _ => None,
            }) {
                Some(u) => exps.push(u),
                None => other.push(a),
            }
        }
        if exps.len() < 2 {
            return None;
        }
        let mut total = 0usize;
        let mut merged: Vec<(ExprId, Rational)> = Vec::new();
        for &u in &exps {
            for (c, core) in linear_parts(u, pool) {
                total += 1;
                match merged.iter_mut().find(|(k, _)| *k == core) {
                    Some((_, acc)) => *acc += c,
                    None => merged.push((core, c)),
                }
            }
        }
        if merged.len() == total {
            return None;
        }
        let one = pool.integer(1_i32);
        let terms: Vec<ExprId> = merged
            .into_iter()
            .filter(|(_, c)| *c != 0)
            .map(|(core, c)| {
                if core == one {
                    intern_rational(c, pool)
                } else {
                    scaled(c, core, pool)
                }
            })
            .collect();
        let sum = match terms.len() {
            0 => None,
            1 => Some(terms[0]),
            _ => Some(pool.add(terms)),
        };
        if let Some(sum) = sum {
            other.push(pool.func("exp", vec![sum]));
        }
        let after = match other.len() {
            0 => one,
            1 => other[0],
            _ => pool.mul(other),
        };
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// Whether `e` is provably a real number `≥ 0` (conservative).
pub(super) fn is_nonneg(e: ExprId, pool: &ExprPool, depth: u32) -> bool {
    if depth == 0 {
        return false;
    }
    enum K {
        Yes,
        No,
        All(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, ExprId),
    }
    let k = pool.with(e, |d| match d {
        ExprData::Integer(n) => {
            if n.0 >= 0 {
                K::Yes
            } else {
                K::No
            }
        }
        ExprData::Rational(r) => {
            if r.0 >= 0 {
                K::Yes
            } else {
                K::No
            }
        }
        ExprData::Symbol { name, domain, .. } => {
            if matches!(domain, Domain::Positive | Domain::NonNegative)
                || crate::eval::symbols::constant_value_f64(name).is_some()
            {
                K::Yes
            } else {
                K::No
            }
        }
        ExprData::Add(a) | ExprData::Mul(a) => K::All(a.clone()),
        ExprData::Pow { base, exp } => K::Pow(*base, *exp),
        ExprData::Func { name, args } if args.len() == 1 => K::Func(name.clone(), args[0]),
        _ => K::No,
    });
    match k {
        K::Yes => true,
        K::No => false,
        K::All(a) => a.iter().all(|&x| is_nonneg(x, pool, depth - 1)),
        K::Pow(b, x) => {
            let int = pool.with(x, |d| match d {
                ExprData::Integer(n) => Some(n.0.is_even()),
                ExprData::Rational(_) => Some(false),
                _ => None,
            });
            match int {
                Some(_) if is_nonneg(b, pool, depth - 1) => true,
                Some(true) => is_real(b, pool, depth - 1),
                _ => false,
            }
        }
        K::Func(name, a) => match name.as_str() {
            "abs" => true,
            "sqrt" => is_nonneg(a, pool, depth - 1),
            "exp" | "cosh" => is_real(a, pool, depth - 1),
            _ => false,
        },
    }
}

/// Whether `e` is provably real (conservative).
fn is_real(e: ExprId, pool: &ExprPool, depth: u32) -> bool {
    if depth == 0 {
        return false;
    }
    enum K {
        Yes,
        No,
        All(Vec<ExprId>),
        IntPow(ExprId),
        Func(String, ExprId),
    }
    let k = pool.with(e, |d| match d {
        ExprData::Integer(_) | ExprData::Rational(_) => K::Yes,
        ExprData::Symbol { domain, .. } => {
            if matches!(
                domain,
                Domain::Real | Domain::Integer | Domain::Positive | Domain::NonNegative
            ) {
                K::Yes
            } else {
                K::No
            }
        }
        ExprData::Add(a) | ExprData::Mul(a) => K::All(a.clone()),
        ExprData::Pow { base, exp } if pool.with(*exp, |x| matches!(x, ExprData::Integer(_))) => {
            K::IntPow(*base)
        }
        ExprData::Func { name, args } if args.len() == 1 => K::Func(name.clone(), args[0]),
        _ => K::No,
    });
    match k {
        K::Yes => true,
        K::No => false,
        K::All(a) => a.iter().all(|&x| is_real(x, pool, depth - 1)),
        K::IntPow(b) => is_real(b, pool, depth - 1),
        K::Func(name, a) => {
            matches!(
                name.as_str(),
                "sin" | "cos" | "exp" | "sinh" | "cosh" | "tanh" | "atan" | "erf"
            ) && is_real(a, pool, depth - 1)
        }
    }
}

// ---------------------------------------------------------------------------
// SqrtLiteralProduct
// ---------------------------------------------------------------------------

/// `√a · √b → √(ab)` for positive integer literals `a`, `b`.
pub struct SqrtLiteralProduct;

impl RewriteRule for SqrtLiteralProduct {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::MUL
    }
    fn name(&self) -> &'static str {
        "sqrt_literal_product"
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let n_sqrt = pool.with(expr, |d| match d {
            ExprData::Mul(a) => a
                .iter()
                .filter(|&&f| {
                    pool.with(f, |fd| match fd {
                        ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                            pool.with(args[0], |x| matches!(x, ExprData::Integer(_)))
                        }
                        _ => false,
                    })
                })
                .count(),
            _ => 0,
        });
        if n_sqrt < 2 {
            return None;
        }
        let args = pool.with(expr, |d| match d {
            ExprData::Mul(a) => Some(a.clone()),
            _ => None,
        })?;
        let radicand = |e: ExprId| -> Option<Integer> {
            pool.with(e, |d| match d {
                ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                    pool.with(args[0], |a| match a {
                        ExprData::Integer(n) if n.0 > 0 && n.0.significant_bits() < 128 => {
                            Some(n.0.clone())
                        }
                        _ => None,
                    })
                }
                _ => None,
            })
        };
        let mut prod = Integer::from(1);
        let mut count = 0;
        let mut rest = Vec::new();
        for &a in &args {
            match radicand(a) {
                Some(n) => {
                    prod *= n;
                    count += 1;
                }
                None => rest.push(a),
            }
        }
        if count < 2 {
            return None;
        }
        rest.push(pool.func("sqrt", vec![pool.integer(prod)]));
        let after = if rest.len() == 1 {
            rest[0]
        } else {
            pool.mul(rest)
        };
        Some((after, one_step(self.name(), expr, after)))
    }
}

/// The special-value and constant-folding rules, in the order the engine
/// should try them.
pub(crate) fn special_value_rules() -> Vec<Box<dyn RewriteRule>> {
    vec![
        Box::new(SpecialValue),
        Box::new(RadicalPower),
        Box::new(PowMulIntExp),
        Box::new(MulFolds),
    ]
}

/// [`PowMulCancel`], [`SqrtLiteralProduct`] and [`ExpCancel`] behind one
/// screen: the engine offers every rule every `Mul`, and one scan of the
/// factors decides which (if any) of the three can fire.
struct MulFolds;

impl RewriteRule for MulFolds {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::MUL
    }
    fn name(&self) -> &'static str {
        "mul_folds"
    }
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (mut sqrts, mut exps, mut pow_of_mul) = (0usize, 0usize, false);
        pool.with(expr, |d| {
            if let ExprData::Mul(a) = d {
                for &f in a {
                    pool.with(f, |fd| match fd {
                        ExprData::Func { name, args } if args.len() == 1 && name == "exp" => {
                            exps += 1;
                        }
                        ExprData::Func { name, args }
                            if args.len() == 1
                                && name == "sqrt"
                                && pool.with(args[0], |x| matches!(x, ExprData::Integer(_))) =>
                        {
                            sqrts += 1;
                        }
                        ExprData::Pow { base, exp }
                            if pool.with(*exp, |x| matches!(x, ExprData::Integer(_)))
                                && pool.with(*base, |b| matches!(b, ExprData::Mul(_))) =>
                        {
                            pow_of_mul = true;
                        }
                        _ => {}
                    });
                }
            }
        });
        if pow_of_mul {
            if let Some(r) = PowMulCancel.apply(expr, pool) {
                return Some(r);
            }
        }
        if sqrts >= 2 {
            if let Some(r) = SqrtLiteralProduct.apply(expr, pool) {
                return Some(r);
            }
        }
        if exps >= 2 {
            return ExpCancel.apply(expr, pool);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplify::engine::simplify;

    fn s(e: ExprId, pool: &ExprPool) -> ExprId {
        simplify(e, pool).value
    }

    /// The allocation-free word path agrees with the bignum one it screens
    /// for, on small radicands, perfect powers, and large words.
    #[test]
    fn perfect_power_word_path_matches_bignum() {
        let mut cases: Vec<u64> = (1..4000).collect();
        for b in [2u64, 3, 7, 997, 1009, 65_537, 4_294_967_291] {
            for q in 2..=5u32 {
                if let Some(v) = b.checked_pow(q) {
                    cases.extend([v, v.saturating_sub(1), v + 1, v.saturating_mul(6)]);
                }
            }
        }
        cases.extend([u64::MAX, u64::MAX - 1, 1 << 63, 999_983 * 999_983]);
        for &n in &cases {
            for q in 2..=12u32 {
                let (k, m) = extract_perfect_power_u64(n, q);
                let (kb, mb) = extract_perfect_power(&Integer::from(n), q);
                assert_eq!(
                    (Integer::from(k), Integer::from(m)),
                    (kb, mb),
                    "n={n} q={q}"
                );
            }
        }
    }

    #[test]
    fn trig_table_matches_f64() {
        let pool = ExprPool::new();
        let p = pi(&pool);
        for den in [1i64, 2, 3, 4, 6] {
            for num in -13i64..=13 {
                let q = Rational::from((num, den));
                let arg = pool.mul(vec![intern_rational(q.clone(), &pool), p]);
                let qf = q.to_f64() * std::f64::consts::PI;
                for (name, f) in [("sin", qf.sin()), ("cos", qf.cos())] {
                    let v = s(pool.func(name, vec![arg]), &pool);
                    let got = crate::eval::eval_f64(v, &pool, &Default::default())
                        .unwrap_or_else(|_| panic!("{name}({q}π) -> {}", pool.display(v)));
                    assert!((got - f).abs() < 1e-12, "{name}({q}π) = {got}, want {f}");
                }
                let t = s(pool.func("tan", vec![arg]), &pool);
                if let Ok(got) = crate::eval::eval_f64(t, &pool, &Default::default()) {
                    assert!((got - qf.tan()).abs() < 1e-9, "tan({q}π) = {got}");
                }
            }
        }
    }

    #[test]
    fn integer_multiples_of_pi() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Integer);
        let x = pool.symbol("x", Domain::Real);
        let p = pi(&pool);
        let npi = pool.mul(vec![n, p]);
        assert_eq!(s(pool.func("sin", vec![npi]), &pool), pool.integer(0_i32));
        assert_eq!(
            s(pool.func("cos", vec![npi]), &pool),
            pool.pow(pool.integer(-1_i32), n)
        );
        let two_npi = pool.mul(vec![pool.integer(2_i32), n, p]);
        assert_eq!(
            s(pool.func("cos", vec![two_npi]), &pool),
            pool.integer(1_i32)
        );
        // A real (not integer) symbol decides nothing.
        let xpi = pool.mul(vec![x, p]);
        let e = pool.func("sin", vec![xpi]);
        assert_eq!(s(e, &pool), e);
    }

    #[test]
    fn gamma_half_integers_match_f64() {
        let pool = ExprPool::new();
        for k in -9i64..=9 {
            let q = Rational::from((2 * k + 1, 2));
            let v = s(
                pool.func("gamma", vec![intern_rational(q.clone(), &pool)]),
                &pool,
            );
            let got = crate::eval::eval_f64(v, &pool, &Default::default()).unwrap();
            // Γ(1/2) = √π and Γ(z+1) = z·Γ(z), in f64.
            let mut want = std::f64::consts::PI.sqrt();
            let mut z = 0.5f64;
            if k >= 0 {
                for _ in 0..k {
                    want *= z;
                    z += 1.0;
                }
            } else {
                for _ in 0..(-k) {
                    z -= 1.0;
                    want /= z;
                }
            }
            assert!(
                ((got - want) / want).abs() < 1e-12,
                "Γ({q}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn radicals() {
        let pool = ExprPool::new();
        let two = pool.integer(2_i32);
        let sq2 = pool.func("sqrt", vec![two]);
        assert_eq!(s(pool.pow(sq2, two), &pool), two);
        let sq8 = pool.func("sqrt", vec![pool.integer(8_i32)]);
        assert_eq!(s(sq8, &pool), pool.mul(vec![two, sq2]));
        let x = pool.symbol("x", Domain::NonNegative);
        let sqx = pool.func("sqrt", vec![x]);
        assert_eq!(s(pool.pow(sqx, two), &pool), x);
        // 1/sqrt(x) keeps its spelling.
        let inv = pool.pow(sqx, pool.integer(-1_i32));
        assert_eq!(s(inv, &pool), inv);
        // A real radicand of unknown sign is left alone: (√y)² is not a real
        // number for y < 0.
        let y = pool.symbol("y", Domain::Real);
        let sqy2 = pool.pow(pool.func("sqrt", vec![y]), two);
        assert_eq!(s(sqy2, &pool), sqy2);
    }

    #[test]
    fn exp_cancel_merges_only_cancelling_exponentials() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let negx = pool.mul(vec![pool.integer(-1_i32), x]);
        let e = pool.mul(vec![
            pool.func("exp", vec![negx]),
            pool.func("exp", vec![x]),
        ]);
        assert_eq!(s(e, &pool), pool.integer(1_i32));
        let unrelated = pool.mul(vec![pool.func("exp", vec![x]), pool.func("exp", vec![y])]);
        assert_eq!(s(unrelated, &pool), unrelated);
    }

    #[test]
    fn odd_functions_cancel() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let negx = pool.mul(vec![pool.integer(-1_i32), x]);
        let e = pool.add(vec![
            pool.func("erf", vec![negx]),
            pool.func("erf", vec![x]),
        ]);
        assert_eq!(s(e, &pool), pool.integer(0_i32));
    }
}
