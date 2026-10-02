//! Truncated power series over ℚ — the fast path behind
//! [`crate::calculus::series::local_expansion`].
//!
//! The general route forms the `k`-th Taylor coefficient as `f⁽ᵏ⁾(0)/k!`, by
//! differentiating the expression `k` times and simplifying a substitution at
//! every step. That is exact and handles anything `diff` handles, but the
//! derivative trees grow with `k` (geometrically for compositions such as
//! `sin(tan x)`), and every coefficient pays for a `simplify`.
//!
//! When every coefficient is a **rational number** — which is the case for
//! any composition of `+ − × ÷`, integer and rational powers, and the
//! elementary heads below, applied to `ξ` and rational constants, at a point
//! where nothing irrational appears — the series can instead be computed
//! *compositionally*: expand each subterm once as a vector of exact rationals
//! and combine them with the standard truncated-series algorithms. Every
//! operation is a `O(n²)` loop over exact rationals, no expression is built
//! until the end, and the result is the same series (the coefficients are the
//! same rational numbers, and [`local_expansion_tps`] emits them as the same
//! canonical literals `simplify` produces).
//!
//! | operation | method |
//! |---|---|
//! | `a ± b`, `a · b` | coefficient-wise / truncated convolution |
//! | `1/a` | the recurrence `d_k = −(Σ_{j≥1} a_j d_{k−j})/a₀` |
//! | `a^α`, `α ∈ ℚ` | J. C. P. Miller's recurrence `a₀ k w_k = Σ_{j=1}^{k} ((α+1)j − k) a_j w_{k−j}` |
//! | `exp g` | `E' = g'E` |
//! | `log g` | `g L' = g'` |
//! | `sin g`, `cos g` (and `sinh`, `cosh`) | `S' = g'C`, `C' = ∓g'S` |
//! | `tan g`, `tanh g` | `S/C` |
//! | `atan`, `atanh`, `asin`, `asinh` | `∫ g' · (1 ± g²)^{−1}` or `^{−1/2}` |
//!
//! Anything else — a free parameter, a float, an irrational constant term
//! (`sin(1 + ξ)`, `log(2 + ξ)`, `√(2 + ξ)`), a head with no rule here, a
//! branch point (`√ξ`, `log ξ`), an essential singularity (`exp(1/ξ)`) — is
//! declined with `None`, and the caller runs the general route exactly as
//! before. The fast path never *refuses*: every refusal and every
//! indeterminate-form diagnosis still comes from the general route.
//!
//! # Precision
//!
//! A series is held as `Σ cᵢ ξ^{v+i} + O(ξ^{prec})` with `cᵢ` exact, so every
//! coefficient it stores is *correct*, and each operation computes the
//! precision of its result from the precisions of its operands (a product
//! loses precision by the valuation of the other factor; a reciprocal by twice
//! its own). Everything is truncated at a working precision `P`; when the
//! result comes back short of what was asked — a pole multiplying a
//! cancellation, a divisor whose leading terms cancel — `P` is raised and the
//! evaluation repeated, up to a bound, after which the general route takes
//! over. A coefficient is therefore never *guessed*: a series that is zero to
//! its precision is reported as such, and a division by one asks for more
//! precision rather than dividing by a zero that might not be one.

use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::ops::Pow;
use rug::{Integer, Rational};
use std::collections::HashMap;

/// `Σ c[i] ξ^{v+i} + O(ξ^{prec})`.
///
/// Invariants: `c.len() == prec − v`, and `c[0] ≠ 0` when `c` is non-empty.
/// An empty `c` is a series known only to be `O(ξ^{prec})`, and then
/// `v == prec` (its valuation is *at least* `prec`, which is exactly what the
/// precision rules below need from it).
#[derive(Clone, Debug)]
struct Tps {
    v: i64,
    c: Vec<Rational>,
    prec: i64,
}

/// Why an evaluation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fail {
    /// The expression is outside what this module expands over ℚ. Final.
    Unsupported,
    /// A leading coefficient that is needed is not known at this working
    /// precision (a divisor that is `O(ξ^k)` so far). Retry with more.
    Precision,
}

type R<T> = Result<T, Fail>;

impl Tps {
    /// Build from coefficients starting at exponent `v`, truncate to `prec`,
    /// and strip leading zeros.
    fn new(v: i64, mut c: Vec<Rational>, prec: i64) -> Tps {
        let len = (prec - v).max(0) as usize;
        c.truncate(len);
        c.resize(len, Rational::new());
        let lead = c.iter().position(|x| *x != 0);
        match lead {
            None => Tps {
                v: prec,
                c: Vec::new(),
                prec,
            },
            Some(i) => {
                c.drain(..i);
                Tps {
                    v: v + i as i64,
                    c,
                    prec,
                }
            }
        }
    }

    fn constant(r: Rational, p: i64) -> Tps {
        Tps::new(0, vec![r], p)
    }

    /// `ξ^k`, known to the working precision (it is exact).
    fn monomial(k: i64, p: i64) -> Tps {
        Tps::new(k, vec![Rational::from(1)], p)
    }

    fn is_zero(&self) -> bool {
        self.c.is_empty()
    }

    /// Coefficient of `ξ^e`; `e` must be below `prec`.
    fn coef(&self, e: i64) -> Rational {
        debug_assert!(e < self.prec);
        if e < self.v {
            Rational::new()
        } else {
            self.c[(e - self.v) as usize].clone()
        }
    }

    /// Coefficients of `ξ⁰ … ξ^{n−1}` for a series with no negative powers.
    fn dense_from_zero(&self, n: usize) -> Vec<Rational> {
        (0..n as i64).map(|e| self.coef(e)).collect()
    }
}

/// Budget checkpoint for the coefficient loops, every 16th row.
///
/// Each row is `O(n)` rational operations, so this bounds the work between
/// checks. A trip declines the fast path; the general route that runs next
/// trips the same budget at once and reports it as before.
fn checkpoint(k: usize) -> R<()> {
    if k % 16 == 15 && crate::budget::check().is_err() {
        return Err(Fail::Unsupported);
    }
    Ok(())
}

/// Dense truncated product `a · b mod ξⁿ` (both indexed from `ξ⁰`).
fn dmul(a: &[Rational], b: &[Rational], n: usize) -> R<Vec<Rational>> {
    let mut out = vec![Rational::new(); n];
    for (i, ai) in a.iter().enumerate().take(n) {
        checkpoint(i)?;
        if *ai == 0 {
            continue;
        }
        for (j, bj) in b.iter().enumerate().take(n - i) {
            if *bj == 0 {
                continue;
            }
            out[i + j] += Rational::from(ai * bj);
        }
    }
    Ok(out)
}

/// `1/a mod ξⁿ`, `a[0] ≠ 0`.
fn dinv(a: &[Rational], n: usize) -> R<Vec<Rational>> {
    let inv0 = Rational::from(1) / &a[0];
    let mut d: Vec<Rational> = Vec::with_capacity(n);
    for k in 0..n {
        checkpoint(k)?;
        if k == 0 {
            d.push(inv0.clone());
            continue;
        }
        let mut s = Rational::new();
        for j in 1..=k.min(a.len().saturating_sub(1)) {
            if a[j] != 0 {
                s += Rational::from(&a[j] * &d[k - j]);
            }
        }
        d.push(-(s * &inv0));
    }
    Ok(d)
}

/// `a^α mod ξⁿ` by Miller's recurrence, given `w0 = a[0]^α` (`a[0] ≠ 0`).
fn dpow(a: &[Rational], alpha: &Rational, w0: Rational, n: usize) -> R<Vec<Rational>> {
    let mut w: Vec<Rational> = Vec::with_capacity(n);
    if n == 0 {
        return Ok(w);
    }
    w.push(w0);
    let a0 = &a[0];
    let alpha1 = Rational::from(alpha + 1u32);
    for k in 1..n {
        checkpoint(k)?;
        let mut s = Rational::new();
        for j in 1..=k.min(a.len().saturating_sub(1)) {
            if a[j] == 0 {
                continue;
            }
            // ((α+1)·j − k) · a_j · w_{k−j}
            let f = Rational::from(&alpha1 * j as u64) - k as u64;
            if f == 0 {
                continue;
            }
            s += f * Rational::from(&a[j] * &w[k - j]);
        }
        w.push(s / Rational::from(a0 * k as u64));
    }
    Ok(w)
}

/// `exp(g) mod ξⁿ`, `g[0] = 0`.
fn dexp(g: &[Rational], n: usize) -> R<Vec<Rational>> {
    let mut e: Vec<Rational> = Vec::with_capacity(n);
    if n == 0 {
        return Ok(e);
    }
    e.push(Rational::from(1));
    for k in 1..n {
        checkpoint(k)?;
        let mut s = Rational::new();
        for j in 1..=k.min(g.len().saturating_sub(1)) {
            if g[j] != 0 {
                s += Rational::from(&g[j] * &e[k - j]) * j as u64;
            }
        }
        e.push(s / k as u64);
    }
    Ok(e)
}

/// `log(g) mod ξⁿ`, `g[0] = 1`.
fn dlog(g: &[Rational], n: usize) -> R<Vec<Rational>> {
    let mut l: Vec<Rational> = Vec::with_capacity(n);
    if n == 0 {
        return Ok(l);
    }
    l.push(Rational::new());
    for k in 1..n {
        checkpoint(k)?;
        // k L_k = k g_k − Σ_{j=1}^{k−1} j L_j g_{k−j}
        let mut s = if k < g.len() {
            Rational::from(&g[k] * k as u64)
        } else {
            Rational::new()
        };
        for j in 1..k {
            if k - j < g.len() && g[k - j] != 0 && l[j] != 0 {
                s -= Rational::from(&l[j] * &g[k - j]) * j as u64;
            }
        }
        l.push(s / k as u64);
    }
    Ok(l)
}

/// `(sin g, cos g)` — or `(sinh g, cosh g)` when `hyperbolic` — `mod ξⁿ`,
/// `g[0] = 0`.
fn dsincos(g: &[Rational], n: usize, hyperbolic: bool) -> R<(Vec<Rational>, Vec<Rational>)> {
    let mut s: Vec<Rational> = Vec::with_capacity(n);
    let mut c: Vec<Rational> = Vec::with_capacity(n);
    if n == 0 {
        return Ok((s, c));
    }
    s.push(Rational::new());
    c.push(Rational::from(1));
    for k in 1..n {
        checkpoint(k)?;
        let mut ss = Rational::new();
        let mut cs = Rational::new();
        for j in 1..=k.min(g.len().saturating_sub(1)) {
            if g[j] == 0 {
                continue;
            }
            let jg = Rational::from(&g[j] * j as u64);
            ss += Rational::from(&jg * &c[k - j]);
            cs += jg * &s[k - j];
        }
        s.push(ss / k as u64);
        let ck = cs / k as u64;
        c.push(if hyperbolic { ck } else { -ck });
    }
    Ok((s, c))
}

/// `g' mod ξ^{n−1}`.
fn dderiv(g: &[Rational], n: usize) -> Vec<Rational> {
    (1..n)
        .map(|k| {
            if k < g.len() {
                Rational::from(&g[k] * k as u64)
            } else {
                Rational::new()
            }
        })
        .collect()
}

/// `∫₀ h mod ξⁿ` (constant of integration `0`).
fn dinteg(h: &[Rational], n: usize) -> Vec<Rational> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    out.push(Rational::new());
    for k in 1..n {
        out.push(if k - 1 < h.len() {
            Rational::from(&h[k - 1] / k as u64)
        } else {
            Rational::new()
        });
    }
    out
}

/// `r^{p/q}` when it is rational and real-positive, else `None`.
fn rational_power(r: &Rational, alpha: &Rational) -> Option<Rational> {
    let (p, q) = (alpha.numer(), alpha.denom());
    let q = q.to_u32()?;
    let p = p.to_i32()?;
    let base = if q == 1 {
        r.clone()
    } else {
        // Principal real root of a positive rational only: a negative base
        // under an even root is complex, and under an odd one depends on the
        // branch convention of the rest of the system — leave both alone.
        if *r <= 0 {
            return None;
        }
        let root = |z: &Integer| -> Option<Integer> {
            let t = Integer::from(z.root_ref(q));
            if t.clone().pow(q) == *z {
                Some(t)
            } else {
                None
            }
        };
        Rational::from((root(r.numer())?, root(r.denom())?))
    };
    if p < 0 && base == 0 {
        return None;
    }
    Some(base.pow(p))
}

/// Longest coefficient vector the fast path builds. A pole deep enough to
/// need more is left to the general route rather than allocated.
const MAX_COEFFICIENTS: i64 = 1 << 15;

/// `prec − v` as a vector length, or `Unsupported` past [`MAX_COEFFICIENTS`].
fn coefficient_count(v: i64, prec: i64) -> R<usize> {
    match prec.checked_sub(v) {
        Some(n) if (0..=MAX_COEFFICIENTS).contains(&n) => Ok(n as usize),
        _ => Err(Fail::Unsupported),
    }
}

/// An exact rational literal.
fn literal(id: ExprId, pool: &ExprPool) -> Option<Rational> {
    pool.with(id, |d| match d {
        ExprData::Integer(n) => Some(Rational::from(&n.0)),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    })
}

struct Ctx<'a> {
    xi: ExprId,
    /// Working precision: nothing at or above `ξ^p` is computed.
    p: i64,
    memo: HashMap<ExprId, Tps>,
    pool: &'a ExprPool,
}

impl Ctx<'_> {
    fn add(&self, a: &Tps, b: &Tps) -> Tps {
        let prec = a.prec.min(b.prec);
        let v = a.v.min(b.v).min(prec);
        let c = (v..prec).map(|e| a.coef(e) + b.coef(e)).collect();
        Tps::new(v, c, prec)
    }

    fn mul(&self, a: &Tps, b: &Tps) -> R<Tps> {
        // a = ξ^{va}(…) + O(ξ^{pa}), b likewise: the product is known to
        // O(ξ^{min(pa + vb, pb + va)}). For an `O(ξ^k)` operand `v = prec`,
        // so the same formula covers it.
        let prec = (a.prec + b.v).min(b.prec + a.v).min(self.p);
        let v = a.v + b.v;
        if a.is_zero() || b.is_zero() || prec <= v {
            return Ok(Tps::new(prec, Vec::new(), prec));
        }
        let n = coefficient_count(v, prec)?;
        Ok(Tps::new(v, dmul(&a.c, &b.c, n)?, prec))
    }

    fn pow(&self, a: &Tps, alpha: &Rational) -> R<Tps> {
        if *alpha == 0 {
            // `f⁰` — leave the meaning of `0⁰` to the general route.
            return Err(Fail::Unsupported);
        }
        if a.is_zero() {
            // `O(ξ^k)^n = O(ξ^{nk})` for a positive integer `n` and `k ≥ 1`;
            // anything else needs the leading term.
            if alpha.denom() == &1 && *alpha > 0 && a.prec >= 1 {
                let n = alpha.numer().to_i64().ok_or(Fail::Unsupported)?;
                let prec = a.prec.saturating_mul(n).min(self.p);
                return Ok(Tps::new(prec, Vec::new(), prec));
            }
            return Err(Fail::Precision);
        }
        let integral = alpha.denom() == &1;
        if !integral && a.v != 0 {
            // `(ξ²)^{1/2}` is `|ξ|` on the reals, `ξ^{1/2}` is a branch point:
            // not a Laurent series this module should be deciding about.
            return Err(Fail::Unsupported);
        }
        let v = if integral {
            let n = alpha.numer().to_i64().ok_or(Fail::Unsupported)?;
            a.v.checked_mul(n).ok_or(Fail::Unsupported)?
        } else {
            0
        };
        let w0 = rational_power(&a.c[0], alpha).ok_or(Fail::Unsupported)?;
        let prec = v
            .checked_add(a.c.len() as i64)
            .ok_or(Fail::Unsupported)?
            .min(self.p);
        if prec <= v {
            return Ok(Tps::new(prec, Vec::new(), prec));
        }
        let n = coefficient_count(v, prec)?;
        Ok(Tps::new(v, dpow(&a.c, alpha, w0, n)?, prec))
    }

    /// Dense coefficients `ξ⁰ … ξ^{n−1}` of an argument whose constant term
    /// is `0`, and `n` — the precision the composite is known to.
    fn zero_constant_arg(&self, g: &Tps) -> R<(Vec<Rational>, usize)> {
        if g.prec <= 0 {
            return Err(Fail::Precision);
        }
        if !g.is_zero() && g.v < 1 {
            // A nonzero constant term (irrational image) or a pole
            // (essential singularity): not ours.
            return Err(Fail::Unsupported);
        }
        let n = g.prec.min(self.p) as usize;
        Ok((g.dense_from_zero(n), n))
    }

    /// Dense coefficients of an argument with constant term `1`.
    fn unit_constant_arg(&self, g: &Tps) -> R<(Vec<Rational>, usize)> {
        if g.prec <= 0 {
            return Err(Fail::Precision);
        }
        if g.is_zero() || g.v != 0 || g.c[0] != 1 {
            return Err(Fail::Unsupported);
        }
        let n = g.prec.min(self.p) as usize;
        Ok((g.dense_from_zero(n), n))
    }

    fn func(&self, name: &str, g: &Tps) -> R<Tps> {
        let dense = |c: Vec<Rational>, n: usize| Tps::new(0, c, n as i64);
        match name {
            "exp" => {
                let (gd, n) = self.zero_constant_arg(g)?;
                Ok(dense(dexp(&gd, n)?, n))
            }
            "log" => {
                let (gd, n) = self.unit_constant_arg(g)?;
                Ok(dense(dlog(&gd, n)?, n))
            }
            "sin" | "cos" | "sinh" | "cosh" | "tan" | "tanh" => {
                let (gd, n) = self.zero_constant_arg(g)?;
                let hyp = name.ends_with('h');
                let (s, c) = dsincos(&gd, n, hyp)?;
                Ok(match name {
                    "sin" | "sinh" => dense(s, n),
                    "cos" | "cosh" => dense(c, n),
                    _ => dense(dmul(&s, &dinv(&c, n)?, n)?, n),
                })
            }
            "atan" | "atanh" | "asin" | "asinh" => {
                let (gd, n) = self.zero_constant_arg(g)?;
                // ∫ g' · (1 ± g²)^α with α = −1 (atan/atanh) or −1/2 (asin/asinh).
                let m = n.saturating_sub(1);
                let g2 = dmul(&gd, &gd, m)?;
                let sign_plus = matches!(name, "atan" | "asinh");
                let one_pm: Vec<Rational> = g2
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let x = if sign_plus { x } else { -x };
                        if i == 0 {
                            x + 1u32
                        } else {
                            x
                        }
                    })
                    .collect();
                let factor = if m == 0 {
                    Vec::new()
                } else if name.starts_with("at") {
                    dinv(&one_pm, m)?
                } else {
                    let alpha = Rational::from((-1, 2));
                    dpow(&one_pm, &alpha, Rational::from(1), m)?
                };
                let h = dmul(&dderiv(&gd, n), &factor, m)?;
                Ok(dense(dinteg(&h, n), n))
            }
            "sqrt" => self.pow(g, &Rational::from((1, 2))),
            _ => Err(Fail::Unsupported),
        }
    }

    fn eval(&mut self, e: ExprId) -> R<Tps> {
        if let Some(t) = self.memo.get(&e) {
            return Ok(t.clone());
        }
        // One checkpoint per node: every operation below is a bounded loop
        // over at most `p` coefficients, so this bounds the work between
        // checks by `O(p²)` rational operations.
        if crate::budget::check().is_err() {
            return Err(Fail::Unsupported);
        }
        let pool = self.pool;
        let out = match pool.get(e) {
            ExprData::Integer(n) => Tps::constant(Rational::from(&n.0), self.p),
            ExprData::Rational(r) => Tps::constant(r.0, self.p),
            ExprData::Symbol { .. } if e == self.xi => Tps::monomial(1, self.p),
            ExprData::Add(xs) => {
                let mut acc: Option<Tps> = None;
                for x in xs {
                    let t = self.eval(x)?;
                    acc = Some(match acc {
                        None => t,
                        Some(a) => self.add(&a, &t),
                    });
                }
                acc.unwrap_or_else(|| Tps::constant(Rational::new(), self.p))
            }
            ExprData::Mul(xs) => {
                let mut acc: Option<Tps> = None;
                for x in xs {
                    let t = self.eval(x)?;
                    acc = Some(match acc {
                        None => t,
                        Some(a) => self.mul(&a, &t)?,
                    });
                }
                acc.unwrap_or_else(|| Tps::constant(Rational::from(1), self.p))
            }
            ExprData::Pow { base, exp } => {
                let alpha = literal(exp, pool).ok_or(Fail::Unsupported)?;
                if base == self.xi && alpha.denom() == &1 {
                    // An exact monomial, including a pole: no precision lost.
                    let k = alpha.numer().to_i64().ok_or(Fail::Unsupported)?;
                    if k == 0 {
                        return Err(Fail::Unsupported);
                    }
                    if k < 0 {
                        coefficient_count(k, self.p)?;
                    }
                    Tps::monomial(k, self.p)
                } else {
                    let b = self.eval(base)?;
                    self.pow(&b, &alpha)?
                }
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                let g = self.eval(args[0])?;
                self.func(&name, &g)?
            }
            _ => return Err(Fail::Unsupported),
        };
        self.memo.insert(e, out.clone());
        Ok(out)
    }
}

/// Largest `order` the fast path takes on. Beyond it the general route (and
/// its work ceiling) decides, exactly as before.
const MAX_TPS_ORDER: u32 = 4096;

/// How far past the needed precision the working precision may be raised
/// while hunting for a leading coefficient, before handing over.
const MAX_EXTRA_PRECISION: i64 = 128;

/// `shifted` expanded about `ξ = 0`, as `(valuation, coefficients)` in the
/// shape [`crate::calculus::series::local_expansion`] returns: for a
/// valuation `v < 0`, the `order` coefficients of `ξ^v … ξ^{v+order−1}`;
/// otherwise valuation `0` and the coefficients of `ξ⁰ … ξ^{order−1}` (leading
/// zeros included, as the direct Taylor route reports them).
///
/// `None` when the expression is not one this module expands exactly over ℚ,
/// or the needed precision is out of reach — the caller then runs the general
/// route, unchanged.
pub(crate) fn tps_expansion(
    shifted: ExprId,
    xi: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Option<(i32, Vec<Rational>)> {
    if order == 0 || order > MAX_TPS_ORDER {
        return None;
    }
    let order = i64::from(order);
    let p_max = 2 * order + MAX_EXTRA_PRECISION;
    let mut p = order;
    loop {
        let mut ctx = Ctx {
            xi,
            p,
            memo: HashMap::new(),
            pool,
        };
        let deficit = match ctx.eval(shifted) {
            Err(Fail::Unsupported) => return None,
            Err(Fail::Precision) => (p / 2).max(8),
            Ok(r) => {
                let target = if r.is_zero() || r.v >= 0 {
                    order
                } else {
                    r.v + order
                };
                if r.prec >= target {
                    return shape(&r, order);
                }
                target - r.prec
            }
        };
        p += deficit;
        if p > p_max {
            return None;
        }
    }
}

fn shape(r: &Tps, order: i64) -> Option<(i32, Vec<Rational>)> {
    let start = if !r.is_zero() && r.v < 0 { r.v } else { 0 };
    // `LocalExpansion::valuation` is an `i32`.
    let valuation = i32::try_from(start).ok()?;
    let coeffs = (start..start + order).map(|e| r.coef(e)).collect();
    Some((valuation, coeffs))
}

/// The canonical literal for `r`: an `Integer` when the denominator is `1`,
/// else a `Rational` — what `simplify` folds a rational coefficient to.
pub(crate) fn rational_literal(r: &Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        pool.rational(r.numer().clone(), r.denom().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn q(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }

    fn expand(e: ExprId, x: ExprId, order: u32, p: &ExprPool) -> Option<(i32, Vec<Rational>)> {
        tps_expansion(e, x, order, p)
    }

    #[test]
    fn sin_and_exp_coefficients() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let (v, c) = expand(p.func("sin", vec![x]), x, 8, &p).unwrap();
        assert_eq!(v, 0);
        assert_eq!(
            c,
            vec![
                q(0, 1),
                q(1, 1),
                q(0, 1),
                q(-1, 6),
                q(0, 1),
                q(1, 120),
                q(0, 1),
                q(-1, 5040)
            ]
        );
        let (_, c) = expand(p.func("exp", vec![x]), x, 5, &p).unwrap();
        assert_eq!(c, vec![q(1, 1), q(1, 1), q(1, 2), q(1, 6), q(1, 24)]);
    }

    #[test]
    fn a_deep_cancellation_raises_the_working_precision() {
        // (sin(tan x) − tan(sin x)) / x⁷ = −1/30 − 29/756 x² + …
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let st = p.func("sin", vec![p.func("tan", vec![x])]);
        let ts = p.func("tan", vec![p.func("sin", vec![x])]);
        let num = p.add(vec![st, p.mul(vec![p.integer(-1), ts])]);
        let e = p.mul(vec![num, p.pow(x, p.integer(-7))]);
        let (v, c) = expand(e, x, 3, &p).unwrap();
        assert_eq!(v, 0);
        assert_eq!(c, vec![q(-1, 30), q(0, 1), q(-29, 756)]);
    }

    #[test]
    fn a_pole_reports_its_valuation() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = p.pow(p.func("sin", vec![x]), p.integer(-2));
        let (v, c) = expand(e, x, 3, &p).unwrap();
        assert_eq!(v, -2);
        assert_eq!(c, vec![q(1, 1), q(0, 1), q(1, 3)]);
    }

    #[test]
    fn irrational_or_singular_inputs_are_declined() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let a = p.symbol("a", Domain::Real);
        let one_plus_x = p.add(vec![p.integer(1), x]);
        let two_plus_x = p.add(vec![p.integer(2), x]);
        for e in [
            p.func("sin", vec![one_plus_x]),
            p.func("log", vec![two_plus_x]),
            p.func("sqrt", vec![two_plus_x]),
            p.func("sqrt", vec![x]),
            p.func("log", vec![x]),
            p.func("exp", vec![p.pow(x, p.integer(-1))]),
            p.mul(vec![a, x]),
            p.func("acos", vec![x]),
            p.pow(p.pow(x, p.integer(2)), p.rational(1, 2)),
        ] {
            assert!(expand(e, x, 4, &p).is_none(), "{}", p.display(e));
        }
        // …but a perfect-power constant term is fine.
        let four_plus_x = p.add(vec![p.integer(4), x]);
        let (_, c) = expand(p.func("sqrt", vec![four_plus_x]), x, 3, &p).unwrap();
        assert_eq!(c, vec![q(2, 1), q(1, 4), q(-1, 64)]);
    }

    #[test]
    fn a_huge_pole_is_declined_not_allocated() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = p.pow(x, p.integer(-1_000_000_000_i64));
        assert!(expand(e, x, 4, &p).is_none());
        // A deep pole of a *series* is cheap — the coefficient vector is
        // relative to the valuation — and correct: sin(x)^-n = x^-n (1 + n x²/6 + …).
        let e = p.pow(p.func("sin", vec![x]), p.integer(-1_000_000_000_i64));
        let (v, c) = expand(e, x, 3, &p).unwrap();
        assert_eq!(v, -1_000_000_000);
        assert_eq!(c, vec![q(1, 1), q(0, 1), q(500_000_000, 3)]);
        // …until the valuation no longer fits `LocalExpansion`.
        let e = p.pow(p.func("sin", vec![x]), p.integer(-3_000_000_000_i64));
        assert!(expand(e, x, 3, &p).is_none());
    }

    #[test]
    fn an_exhausted_budget_declines() {
        use crate::budget::{self, Budget};
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = p.func("exp", vec![p.func("sin", vec![x])]);
        let _guard = budget::enter(Budget::new().with_max_steps(1));
        assert!(expand(e, x, 400, &p).is_none());
    }

    #[test]
    fn a_divisor_that_is_zero_to_every_precision_is_declined() {
        // sin² + cos² − 1 is identically 0: its reciprocal has no leading
        // term at any precision, so the fast path must hand over rather than
        // divide by a zero it cannot see the end of.
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let s2 = p.pow(p.func("sin", vec![x]), p.integer(2));
        let c2 = p.pow(p.func("cos", vec![x]), p.integer(2));
        let z = p.add(vec![s2, c2, p.integer(-1)]);
        let (v, c) = expand(z, x, 4, &p).unwrap();
        assert_eq!(v, 0);
        assert!(c.iter().all(|r| *r == 0));
        assert!(expand(p.pow(z, p.integer(-1)), x, 4, &p).is_none());
    }
}
