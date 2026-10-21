//! Asymptotic expansions of special functions at `+∞` with exact
//! coefficients (F25).
//!
//! The power-scale core of [`super::asymptotic_expand`] substitutes `x = 1/t`
//! and Taylor-expands; that cannot see Stirling's series (`x log x` and
//! `x^x` scales), Hankel's expansions (oscillatory, `x^{−1/2}`) or the
//! `e^{−x²}` of `erfc`. Those are classical theorems, so they are produced
//! here from their closed-form coefficients:
//!
//! | input (`u = a·x + b`, `a > 0` rational) | expansion | DLMF |
//! |---|---|---|
//! | `loggamma(u)` | `(u−½)log u − u + ½log 2π + Σ (−1)^{k+1} B_{k+1}(b)/(k(k+1)(ax)^k)` | 5.11.8 |
//! | `gamma(u)^e`, `factorial`, `binomial` | `exp(e·(…))` of the above: `√(2π) x^{x−½} e^{−x} (1 + 1/(12x) + …)` | 5.11.3 |
//! | `digamma(u)`, `polygamma(n, u)` | `log(ax) + Σ (−1)^{m+1} B_m(b)/(m (ax)^m)`, differentiated | 5.15.8 |
//! | `bessel_j/y(ν, ax)` | `√(2/(πz)) (cos ω Σ (−1)^k a_{2k}/z^{2k} ∓ sin ω Σ …)` | 10.17.3–4 |
//! | `bessel_i/k(ν, ax)` | `e^{±z}/√(2πz)·Σ (∓1)^k a_k(ν)/z^k`, `K` with `√(π/(2z))` | 10.40.1–2 |
//! | `erfc(ax)` | `e^{−z²}/(z√π) Σ (−1)^k (½)_k / z^{2k}` | 7.12.1 |
//! | `Ei(ax)` | `e^z/z Σ k!/z^k` | 6.12.2 |
//! | `airy_ai/bi(ax)` and primes | `e^{∓ζ}/(2√π z^{1/4}) Σ (∓1)^k u_k/ζ^k`, `ζ = (2/3)z^{3/2}` | 9.7.5–8 |
//!
//! Products of these with each other, with constants, with `log(x)^k` and
//! with any power-scale cofactor, and sums of such products that share one
//! exponential scale, are combined term by term. Every coefficient is
//! exact (rationals, `π`, `log` of rationals); a non-rational order, a
//! non-linear argument or a negative slope is not handled here (the caller
//! then refuses with `E-ASYMPT-005`).
//!
//! # The gate
//!
//! The formulas are theorems, but the code that assembles them is not, so
//! every expansion still passes a numeric check against the function itself:
//! at the points `x₀, 3x₀, 10x₀, 100x₀` past the regime where the series is
//! asymptotic (`x₀ = 100` scaled by the parameters), with 640-bit balls, the
//! normalised residual `f/scale − Σ_{k<n} termₖ` must be bounded by
//! `8 ×` the size of the first omitted term. The longest prefix that passes
//! is returned; none passing is `E-ASYMPT-004`.

use std::cmp::Ordering;
use std::collections::HashMap;

use rug::{Integer, Rational};

use super::{power_scale_coeffs, AsymptoticError, AsymptoticExpansion, AsymptoticTerm};
use crate::ball::{ArbBall, IntervalEval};
use crate::jit::eval_interp;
use crate::kernel::subs::mentions_var;
use crate::kernel::{subs, ExprData, ExprId, ExprPool};
use crate::orthopoly::{mk_add, mk_mul, rat_node, rational_literal};
use crate::simplify::simplify;

/// Largest number of terms this module produces (more are not returned).
const MAX_TERMS: usize = 60;
/// Ball precision of the gate, in bits.
const GATE_PREC: u32 = 640;
/// Allowed ratio between the residual and the first omitted term.
const GATE_SLACK: f64 = 8.0;

/// One term `coeff · x^pow · log(x)^logp`. `coeff` is free of `x` except for
/// the bounded oscillatory factor of the Bessel `J`/`Y` expansions; `amp` is
/// an upper bound of `|coeff|` (the gate's envelope).
#[derive(Clone, Debug)]
struct Term {
    pow: Rational,
    logp: u32,
    coeff: ExprId,
    amp: f64,
}

/// `scale · Σ terms + O(x^cut)` (`cut = None`: the series is exact up to
/// terms beyond all orders), most significant term first.
#[derive(Clone, Debug)]
struct Series {
    scale: Option<ExprId>,
    terms: Vec<Term>,
    cut: Option<Rational>,
    /// Where the expansion starts being asymptotic (scales the gate points).
    x_min: f64,
}

fn key_cmp(a: &Term, b: &Term) -> Ordering {
    b.pow.cmp(&a.pow).then(b.logp.cmp(&a.logp))
}

fn is_zero_lit(e: ExprId, pool: &ExprPool) -> bool {
    rational_literal(pool, e).is_some_and(|q| q == 0)
}

fn const_f64(e: ExprId, pool: &ExprPool) -> f64 {
    eval_interp(e, &HashMap::new(), pool).map_or(f64::NAN, f64::abs)
}

fn q_f64(q: &Rational) -> f64 {
    q.to_f64()
}

impl Series {
    fn exact(terms: Vec<Term>) -> Self {
        Series {
            scale: None,
            terms,
            cut: None,
            x_min: 1.0,
        }
        .normalize_pool_free()
    }

    /// Sort; no merging (the builders never emit equal keys).
    fn normalize_pool_free(mut self) -> Self {
        self.terms.sort_by(key_cmp);
        self
    }

    /// Sort, merge equal keys, drop zero coefficients and every term at or
    /// below the cut.
    fn normalize(mut self, pool: &ExprPool) -> Self {
        self.terms.sort_by(key_cmp);
        let mut out: Vec<Term> = Vec::with_capacity(self.terms.len());
        for t in self.terms {
            if let Some(c) = &self.cut {
                if t.pow <= *c {
                    continue;
                }
            }
            match out.last_mut() {
                Some(l) if l.pow == t.pow && l.logp == t.logp => {
                    l.coeff = simplify(pool.add(vec![l.coeff, t.coeff]), pool).value;
                    l.amp += t.amp;
                }
                _ => out.push(t),
            }
        }
        out.retain(|t| !is_zero_lit(t.coeff, pool));
        self.terms = out;
        self
    }

    fn lead(&self) -> Option<&Rational> {
        self.terms.first().map(|t| &t.pow)
    }

    fn mul(&self, other: &Series, pool: &ExprPool) -> Series {
        let scale = match (self.scale, other.scale) {
            (None, s) | (s, None) => s,
            (Some(a), Some(b)) => Some(mk_mul(pool, vec![a, b])),
        };
        // Error of A·B: A.err·B.lead and B.err·A.lead.
        let mut cut: Option<Rational> = None;
        for (s, o) in [(self, other), (other, self)] {
            if let (Some(c), Some(l)) = (&s.cut, o.lead()) {
                let v = Rational::from(c + l);
                cut = Some(match cut {
                    Some(old) if old > v => old,
                    _ => v,
                });
            }
        }
        let mut terms = Vec::with_capacity(self.terms.len() * other.terms.len());
        for a in &self.terms {
            for b in &other.terms {
                let pow = Rational::from(&a.pow + &b.pow);
                if cut.as_ref().is_some_and(|c| pow <= *c) {
                    continue;
                }
                terms.push(Term {
                    pow,
                    logp: a.logp + b.logp,
                    coeff: simplify(mk_mul(pool, vec![a.coeff, b.coeff]), pool).value,
                    amp: a.amp * b.amp,
                });
            }
        }
        Series {
            scale,
            terms,
            cut,
            x_min: self.x_min.max(other.x_min),
        }
        .normalize(pool)
    }

    /// `self + other`; `None` when the two have different exponential scales
    /// (neither ordering is decided here).
    fn add(&self, other: &Series, pool: &ExprPool) -> Option<Series> {
        let scale = match (self.scale, other.scale) {
            (None, None) => None,
            (Some(a), Some(b)) => {
                if simplify(a, pool).value != simplify(b, pool).value {
                    return None;
                }
                Some(a)
            }
            _ => return None,
        };
        let cut = match (&self.cut, &other.cut) {
            (None, c) | (c, None) => c.clone(),
            (Some(a), Some(b)) => Some(if a > b { a.clone() } else { b.clone() }),
        };
        let mut terms = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        Some(
            Series {
                scale,
                terms,
                cut,
                x_min: self.x_min.max(other.x_min),
            }
            .normalize(pool),
        )
    }

    fn scale_by(&self, c: ExprId, pool: &ExprPool) -> Series {
        let amp = const_f64(c, pool);
        let mut s = self.clone();
        for t in &mut s.terms {
            t.coeff = simplify(mk_mul(pool, vec![c, t.coeff]), pool).value;
            t.amp *= amp;
        }
        s.normalize(pool)
    }

    /// `d/dx` term by term (valid for the ψ family: the expansions hold in a
    /// sector and may be differentiated, DLMF 2.1(ii)).
    fn derivative(&self, pool: &ExprPool) -> Series {
        let mut terms = Vec::new();
        for t in &self.terms {
            let p1 = Rational::from(&t.pow - 1);
            if t.pow != 0 {
                terms.push(Term {
                    pow: p1.clone(),
                    logp: t.logp,
                    coeff: simplify(mk_mul(pool, vec![rat_node(pool, &t.pow), t.coeff]), pool)
                        .value,
                    amp: t.amp * q_f64(&t.pow).abs(),
                });
            }
            if t.logp > 0 {
                terms.push(Term {
                    pow: p1,
                    logp: t.logp - 1,
                    coeff: simplify(
                        mk_mul(pool, vec![pool.integer(i64::from(t.logp)), t.coeff]),
                        pool,
                    )
                    .value,
                    amp: t.amp * f64::from(t.logp),
                });
            }
        }
        Series {
            scale: self.scale,
            terms,
            cut: self.cut.as_ref().map(|c| Rational::from(c - 1)),
            x_min: self.x_min,
        }
        .normalize(pool)
    }
}

fn term(pow: Rational, logp: u32, coeff: ExprId, pool: &ExprPool) -> Term {
    let coeff = simplify(coeff, pool).value;
    Term {
        pow,
        logp,
        amp: const_f64(coeff, pool),
        coeff,
    }
}

fn rterm(pow: Rational, q: &Rational, pool: &ExprPool) -> Term {
    Term {
        pow,
        logp: 0,
        coeff: rat_node(pool, q),
        amp: q_f64(q).abs(),
    }
}

fn rq(n: i64, d: i64) -> Rational {
    Rational::from((n, d))
}

fn rpow(a: &Rational, k: i64) -> Rational {
    let mut out = Rational::from(1);
    for _ in 0..k.unsigned_abs() {
        out *= a;
    }
    if k < 0 {
        out.recip_mut();
    }
    out
}

fn pi(pool: &ExprPool) -> ExprId {
    crate::eval::symbols::pi_symbol(pool)
}

fn log_of(q: &Rational, pool: &ExprPool) -> ExprId {
    pool.func("log", vec![rat_node(pool, q)])
}

/// `B_m(b)` for `m = 0..=n` (DLMF 24.2.5).
fn bernoulli_values(b: &Rational, n: usize) -> Option<Vec<Rational>> {
    let bn = crate::specfun::closed::bernoulli(n).ok()?;
    let mut out = Vec::with_capacity(n + 1);
    for m in 0..=n {
        let mut s = Rational::new();
        let mut binom = Integer::from(1); // C(m, k)
        for (k, bk) in bn.iter().enumerate().take(m + 1) {
            if *bk != 0 {
                s += Rational::from(&binom) * bk * rpow(b, (m - k) as i64);
            }
            binom *= (m - k) as u64;
            binom /= (k + 1) as u64;
        }
        out.push(s);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Γ family
// ---------------------------------------------------------------------------

/// The rational coefficients `r_k`, `k = 1..=K`, of the `1/x` tail of
/// `log Γ(ax + b)` (DLMF 5.11.8): `(−1)^{k+1} B_{k+1}(b) / (k(k+1) a^k)`.
fn stirling_tail(a: &Rational, b: &Rational, k_max: usize) -> Option<Vec<Rational>> {
    let bv = bernoulli_values(b, k_max + 1)?;
    Some(
        (1..=k_max)
            .map(|k| {
                let sign = if k % 2 == 1 { 1 } else { -1 };
                let den = Rational::from(k as i64 * (k as i64 + 1)) * rpow(a, k as i64);
                Rational::from(&bv[k + 1] * sign) / den
            })
            .collect(),
    )
}

fn gamma_x_min(a: &Rational, b: &Rational) -> f64 {
    ((q_f64(b).abs() + 1.0) / q_f64(a)).max(1.0)
}

/// `log Γ(ax + b)`.
fn loggamma_series(a: &Rational, b: &Rational, k_max: usize, pool: &ExprPool) -> Option<Series> {
    let half = rq(1, 2);
    let bh = Rational::from(b - &half);
    let mut terms = vec![rterm(Rational::from(1), a, pool)];
    terms[0].logp = 1;
    // a(log a − 1)·x
    let c1 = if *a == 1 {
        pool.integer(-1)
    } else {
        mk_add(
            pool,
            vec![
                mk_mul(pool, vec![rat_node(pool, a), log_of(a, pool)]),
                rat_node(pool, &-a.clone()),
            ],
        )
    };
    terms.push(term(Rational::from(1), 0, c1, pool));
    if bh != 0 {
        terms.push(Term {
            pow: Rational::new(),
            logp: 1,
            coeff: rat_node(pool, &bh),
            amp: q_f64(&bh).abs(),
        });
    }
    // (b − ½) log a + ½ log 2π
    let two_pi = mk_mul(pool, vec![pool.integer(2), pi(pool)]);
    let mut c0 = vec![mk_mul(
        pool,
        vec![pool.rational(1, 2), pool.func("log", vec![two_pi])],
    )];
    if *a != 1 && bh != 0 {
        c0.push(mk_mul(pool, vec![rat_node(pool, &bh), log_of(a, pool)]));
    }
    terms.push(term(Rational::new(), 0, mk_add(pool, c0), pool));
    for (k, r) in stirling_tail(a, b, k_max)?.into_iter().enumerate() {
        if r != 0 {
            terms.push(rterm(Rational::from(-(k as i64) - 1), &r, pool));
        }
    }
    Some(Series {
        scale: None,
        terms,
        cut: Some(Rational::from(-(k_max as i64) - 1)),
        x_min: gamma_x_min(a, b),
    })
}

/// `ψ(ax + b)` (DLMF 5.15.8 after `z = ax`).
fn digamma_series(a: &Rational, b: &Rational, k_max: usize, pool: &ExprPool) -> Option<Series> {
    let mut terms = vec![Term {
        pow: Rational::new(),
        logp: 1,
        coeff: pool.integer(1),
        amp: 1.0,
    }];
    if *a != 1 {
        terms.push(term(Rational::new(), 0, log_of(a, pool), pool));
    }
    let bv = bernoulli_values(b, k_max)?;
    for (m, bm) in bv.iter().enumerate().skip(1) {
        if *bm == 0 {
            continue;
        }
        let sign = if m % 2 == 1 { 1 } else { -1 };
        let c = Rational::from(bm * sign) / (Rational::from(m as i64) * rpow(a, m as i64));
        terms.push(rterm(Rational::from(-(m as i64)), &c, pool));
    }
    Some(Series {
        scale: None,
        terms,
        cut: Some(Rational::from(-(k_max as i64) - 1)),
        x_min: gamma_x_min(a, b),
    })
}

/// `ψ⁽ⁿ⁾(ax + b) = a^{−n} dⁿ/dxⁿ ψ(ax + b)`.
fn polygamma_series(
    n: u32,
    a: &Rational,
    b: &Rational,
    k_max: usize,
    pool: &ExprPool,
) -> Option<Series> {
    let mut s = digamma_series(a, b, k_max + n as usize, pool)?.normalize(pool);
    for _ in 0..n {
        s = s.derivative(pool);
    }
    let inv = rpow(a, -i64::from(n));
    Some(if inv == 1 {
        s
    } else {
        s.scale_by(rat_node(pool, &inv), pool)
    })
}

/// `Π Γ(aᵢx + bᵢ)^{eᵢ}` = `scale · exp(Σ eᵢ Rᵢ(1/x))` (DLMF 5.11.3, 5.11.8):
/// `scale = (2π)^{Σeᵢ/2} Π aᵢ^{eᵢ(aᵢx + bᵢ − ½)} x^{αx + β} e^{−αx}` with
/// `α = Σ eᵢaᵢ`, `β = Σ eᵢ(bᵢ − ½)`; the exponential of the rational
/// power series is formed exactly.
fn gamma_product(
    factors: &[(Rational, Rational, Rational)],
    var: ExprId,
    k_max: usize,
    pool: &ExprPool,
) -> Option<Series> {
    let half = rq(1, 2);
    let mut alpha = Rational::new();
    let mut beta = Rational::new();
    let mut esum = Rational::new();
    let mut tail = vec![Rational::new(); k_max + 1]; // index k: coefficient of x^{-k}
    let mut scale: Vec<ExprId> = Vec::new();
    let mut x_min: f64 = 1.0;
    for (a, b, e) in factors {
        alpha += Rational::from(e * a);
        beta += e * Rational::from(b - &half);
        esum += e;
        for (k, r) in stirling_tail(a, b, k_max)?.into_iter().enumerate() {
            tail[k + 1] += r * e;
        }
        if *a != 1 {
            // a^{e(a x + b − ½)}
            let ex = mk_add(
                pool,
                vec![
                    mk_mul(pool, vec![rat_node(pool, &Rational::from(e * a)), var]),
                    rat_node(pool, &(e * Rational::from(b - &half))),
                ],
            );
            scale.push(pool.pow(rat_node(pool, a), ex));
        }
        x_min = x_min.max(gamma_x_min(a, b));
    }
    if esum != 0 {
        let two_pi = mk_mul(pool, vec![pool.integer(2), pi(pool)]);
        scale.push(pool.pow(two_pi, rat_node(pool, &(esum / 2))));
    }
    if alpha != 0 || beta != 0 {
        let ex = mk_add(
            pool,
            vec![
                mk_mul(pool, vec![rat_node(pool, &alpha), var]),
                rat_node(pool, &beta),
            ],
        );
        scale.push(pool.pow(var, ex));
    }
    if alpha != 0 {
        scale.push(pool.func(
            "exp",
            vec![mk_mul(pool, vec![rat_node(pool, &-alpha.clone()), var])],
        ));
    }
    // E = exp(T): E_0 = 1, E_n = (1/n) Σ_{k=1}^{n} k T_k E_{n−k}.
    let mut ecoef = vec![Rational::from(1)];
    for n in 1..=k_max {
        let mut s = Rational::new();
        for k in 1..=n {
            if tail[k] != 0 {
                s += Rational::from(k as i64) * &tail[k] * &ecoef[n - k];
            }
        }
        ecoef.push(s / Rational::from(n as i64));
    }
    let terms = ecoef
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .map(|(j, c)| rterm(Rational::from(-(j as i64)), c, pool))
        .collect();
    let scale = if scale.is_empty() {
        None
    } else {
        Some(simplify(mk_mul(pool, scale), pool).value)
    };
    Some(Series {
        scale,
        terms,
        cut: Some(Rational::from(-(k_max as i64) - 1)),
        x_min,
    })
}

// ---------------------------------------------------------------------------
// Bessel, erfc, Ei, Airy
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cyl {
    J,
    Y,
    I,
    K,
}

/// Hankel's `a_m(ν) = Π_{j=1}^{m} (4ν² − (2j−1)²) / (m! 8^m)` (DLMF 10.17.1).
fn hankel_a(nu: &Rational, m_max: usize) -> Vec<Rational> {
    let mu = Rational::from(nu * nu) * 4;
    let mut out = vec![Rational::from(1)];
    for m in 1..=m_max {
        let odd = Rational::from((2 * m as i64 - 1) * (2 * m as i64 - 1));
        let next = (&out[m - 1] * (&mu - odd)) / (8 * m as i64);
        out.push(next);
    }
    out
}

fn bessel_series(
    kind: Cyl,
    nu: &Rational,
    a: &Rational,
    var: ExprId,
    k_max: usize,
    pool: &ExprPool,
) -> Series {
    let am = hankel_a(nu, k_max);
    let af = q_f64(a);
    let z = mk_mul(pool, vec![rat_node(pool, a), var]);
    let p = pi(pool);
    let sqrt = |e: ExprId| pool.pow(e, pool.rational(1, 2));
    // Prefactor constant (x^{-1/2} goes into the powers).
    let (c0, c0f, scale) = match kind {
        // √(2/(π a))
        Cyl::J | Cyl::Y => (
            sqrt(mk_mul(
                pool,
                vec![
                    pool.integer(2),
                    pool.pow(mk_mul(pool, vec![p, rat_node(pool, a)]), pool.integer(-1)),
                ],
            )),
            (2.0 / (std::f64::consts::PI * af)).sqrt(),
            None,
        ),
        // 1/√(2π a)
        Cyl::I => (
            pool.pow(
                mk_mul(pool, vec![pool.integer(2), p, rat_node(pool, a)]),
                pool.rational(-1, 2),
            ),
            1.0 / (2.0 * std::f64::consts::PI * af).sqrt(),
            Some(pool.func("exp", vec![z])),
        ),
        // √(π/(2a))
        Cyl::K => (
            sqrt(mk_mul(
                pool,
                vec![
                    p,
                    pool.pow(
                        mk_mul(pool, vec![pool.integer(2), rat_node(pool, a)]),
                        pool.integer(-1),
                    ),
                ],
            )),
            (std::f64::consts::PI / (2.0 * af)).sqrt(),
            Some(pool.func("exp", vec![mk_mul(pool, vec![pool.integer(-1), z])])),
        ),
    };
    // ω = a x − (ν/2 + 1/4) π
    let shift = Rational::from(nu / 2) + rq(1, 4);
    let omega = mk_add(
        pool,
        vec![z, mk_mul(pool, vec![rat_node(pool, &-shift), p])],
    );
    let (cos_w, sin_w) = (pool.func("cos", vec![omega]), pool.func("sin", vec![omega]));
    let mut terms = Vec::new();
    let mut exact = false;
    for (m, amv) in am.iter().enumerate().take(k_max) {
        if *amv == 0 {
            exact = true;
            break;
        }
        let q = amv / rpow(a, m as i64);
        let (q, trig) = match kind {
            Cyl::J | Cyl::Y => {
                let sign = if (m / 2) % 2 == 0 { 1 } else { -1 };
                let q = Rational::from(&q * sign);
                match (kind, m % 2) {
                    (Cyl::J, 0) => (q, Some(cos_w)),
                    (Cyl::J, _) => (-q, Some(sin_w)),
                    (_, 0) => (q, Some(sin_w)),
                    _ => (q, Some(cos_w)),
                }
            }
            Cyl::I => (if m % 2 == 0 { q } else { -q }, None),
            Cyl::K => (q, None),
        };
        let mut parts = vec![rat_node(pool, &q), c0];
        parts.extend(trig);
        terms.push(Term {
            pow: rq(-1, 2) - Rational::from(m as i64),
            logp: 0,
            coeff: simplify(mk_mul(pool, parts), pool).value,
            amp: q_f64(&q).abs() * c0f,
        });
    }
    let x_min = q_f64(nu).powi(2).max(1.0) / af;
    Series {
        scale,
        terms,
        cut: (!exact).then(|| rq(-1, 2) - Rational::from(k_max as i64)),
        x_min: x_min.max(1.0),
    }
}

/// `erfc(ax) ~ e^{−a²x²}/(√π) Σ (−1)^k (2k−1)!!/2^k (ax)^{−2k−1}`.
fn erfc_series(a: &Rational, var: ExprId, k_max: usize, pool: &ExprPool) -> Series {
    let inv_sqrt_pi = pool.pow(pool.func("sqrt", vec![pi(pool)]), pool.integer(-1));
    let isp = 1.0 / std::f64::consts::PI.sqrt();
    let mut terms = Vec::new();
    let mut c = Rational::from(1); // (−1)^k (2k−1)!!/2^k
    for k in 0..k_max {
        if k > 0 {
            c *= Rational::from(-(2 * k as i64 - 1)) / 2;
        }
        let q = &c / rpow(a, 2 * k as i64 + 1);
        terms.push(Term {
            pow: Rational::from(-(2 * k as i64) - 1),
            logp: 0,
            coeff: simplify(mk_mul(pool, vec![rat_node(pool, &q), inv_sqrt_pi]), pool).value,
            amp: q_f64(&q).abs() * isp,
        });
    }
    let a2 = Rational::from(a * a);
    let scale = pool.func(
        "exp",
        vec![mk_mul(
            pool,
            vec![rat_node(pool, &-a2), pool.pow(var, pool.integer(2))],
        )],
    );
    Series {
        scale: Some(scale),
        terms,
        cut: Some(Rational::from(-(2 * k_max as i64) - 1)),
        x_min: (1.0 / q_f64(a)).max(1.0),
    }
}

/// `Ei(ax) ~ e^{ax} Σ k!/(ax)^{k+1}` (DLMF 6.12.2).
fn ei_series(a: &Rational, var: ExprId, k_max: usize, pool: &ExprPool) -> Series {
    let mut terms = Vec::new();
    let mut fact = Integer::from(1);
    for k in 0..k_max {
        if k > 0 {
            fact *= k as u64;
        }
        let q = Rational::from(&fact) / rpow(a, k as i64 + 1);
        terms.push(rterm(Rational::from(-(k as i64) - 1), &q, pool));
    }
    Series {
        scale: Some(pool.func("exp", vec![mk_mul(pool, vec![rat_node(pool, a), var])])),
        terms,
        cut: Some(Rational::from(-(k_max as i64) - 1)),
        x_min: (1.0 / q_f64(a)).max(1.0),
    }
}

/// `Γ(s, ax) ~ e^{−ax} (ax)^{s−1} Σ_k (s−1)(s−2)⋯(s−k) / (ax)^k` (DLMF 8.11.2)
/// for a rational `s`; finite (and exact) at a positive integer `s`.
fn uppergamma_series(
    s: &Rational,
    a: &Rational,
    var: ExprId,
    k_max: usize,
    pool: &ExprPool,
) -> Series {
    let af = q_f64(a);
    let mut terms = Vec::new();
    let mut c = Rational::from(1); // (s−1)(s−2)⋯(s−k)
    let mut exact = false;
    for k in 0..k_max {
        if k > 0 {
            c *= Rational::from(s - k as i64);
        }
        if c == 0 {
            exact = true;
            break;
        }
        // coefficient of x^{s−1−k}: c · a^{s−1−k}
        let p = Rational::from(s - 1) - Rational::from(k as i64);
        let a_pow = if *a == 1 {
            pool.integer(1)
        } else {
            pool.pow(rat_node(pool, a), rat_node(pool, &p))
        };
        terms.push(Term {
            pow: p.clone(),
            logp: 0,
            coeff: simplify(mk_mul(pool, vec![rat_node(pool, &c), a_pow]), pool).value,
            amp: q_f64(&c).abs() * af.powf(q_f64(&p)),
        });
    }
    Series {
        scale: Some(pool.func(
            "exp",
            vec![mk_mul(pool, vec![rat_node(pool, &-a.clone()), var])],
        )),
        terms,
        cut: (!exact).then(|| Rational::from(s - 1) - Rational::from(k_max as i64)),
        x_min: ((q_f64(s).abs() + 1.0) / af).max(1.0),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Airy {
    Ai,
    Bi,
    AiPrime,
    BiPrime,
}

/// `u_k = (2k+1)(2k+3)⋯(6k−1) / (216^k k!)` (DLMF 9.7.2).
fn airy_u(k_max: usize) -> Vec<Rational> {
    (0..k_max)
        .map(|k| {
            let mut num = Integer::from(1);
            let mut j = 2 * k as u64 + 1;
            while j < 6 * k as u64 {
                num *= j;
                j += 2;
            }
            let mut den = Integer::from(Integer::factorial(k as u32));
            for _ in 0..k {
                den *= 216u32;
            }
            Rational::from((num, den))
        })
        .collect()
}

fn airy_series(kind: Airy, a: &Rational, var: ExprId, k_max: usize, pool: &ExprPool) -> Series {
    let u = airy_u(k_max);
    let af = q_f64(a);
    let sp = pool.pow(pool.func("sqrt", vec![pi(pool)]), pool.integer(-1));
    let (base_pow, half, decaying, prime) = match kind {
        Airy::Ai => (rq(-1, 4), true, true, false),
        Airy::Bi => (rq(-1, 4), false, false, false),
        Airy::AiPrime => (rq(1, 4), true, true, true),
        Airy::BiPrime => (rq(1, 4), false, false, true),
    };
    let mut terms = Vec::new();
    for (k, uk) in u.iter().enumerate() {
        // v_k = −(6k+1)/(6k−1) u_k  (v_0 = 1)
        let ck = if prime && k > 0 {
            -(uk.clone() * Rational::from((6 * k as i64 + 1, 6 * k as i64 - 1)))
        } else {
            uk.clone()
        };
        // (∓1)^k from e^{∓ζ}, (3/2)^k from ζ^{−k}, 1/2 for Ai, −1 for Ai′.
        let mut q = &ck * rpow(&rq(3, 2), k as i64);
        if decaying && k % 2 == 1 {
            q = -q;
        }
        if half {
            q /= 2;
        }
        if kind == Airy::AiPrime {
            q = -q;
        }
        let ap = &base_pow - Rational::from((3 * k as i64, 2));
        let a_pow = if *a == 1 {
            pool.integer(1)
        } else {
            pool.pow(rat_node(pool, a), rat_node(pool, &ap))
        };
        terms.push(Term {
            pow: ap.clone(),
            logp: 0,
            coeff: simplify(mk_mul(pool, vec![rat_node(pool, &q), sp, a_pow]), pool).value,
            amp: q_f64(&q).abs() * af.powf(q_f64(&ap)) / std::f64::consts::PI.sqrt(),
        });
    }
    // ζ = (2/3) a^{3/2} x^{3/2}
    let mut zeta = vec![
        pool.rational(if decaying { -2 } else { 2 }, 3),
        pool.pow(var, pool.rational(3, 2)),
    ];
    if *a != 1 {
        zeta.push(pool.pow(rat_node(pool, a), pool.rational(3, 2)));
    }
    Series {
        scale: Some(pool.func("exp", vec![mk_mul(pool, zeta)])),
        terms,
        cut: Some(&base_pow - Rational::from((3 * k_max as i64, 2))),
        x_min: (1.0 / af).max(1.0),
    }
}

// ---------------------------------------------------------------------------
// Recognition
// ---------------------------------------------------------------------------

const SPECIAL_HEADS: &[&str] = &[
    "gamma",
    "loggamma",
    "digamma",
    "trigamma",
    "polygamma",
    "factorial",
    "binomial",
    "bessel_j",
    "bessel_y",
    "bessel_i",
    "bessel_k",
    "bessel_j0",
    "bessel_j1",
    "erfc",
    "Ei",
    "uppergamma",
    "airy_ai",
    "airy_bi",
    "airy_ai_prime",
    "airy_bi_prime",
];

fn mentions_special(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Func { name, args } => {
            (SPECIAL_HEADS.contains(&name.as_str()) && mentions_var(e, var, pool))
                || args.iter().any(|&a| mentions_special(a, var, pool))
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|&x| mentions_special(x, var, pool)),
        ExprData::Pow { base, exp } => {
            mentions_special(base, var, pool) || mentions_special(exp, var, pool)
        }
        _ => false,
    }
}

/// `factorial(u) = Γ(u + 1)`, `binomial(p, q) = Γ(p+1)/(Γ(q+1)Γ(p−q+1))`.
fn rewrite_combinatorial(e: ExprId, pool: &ExprPool) -> ExprId {
    let one = pool.integer(1);
    let g = |u: ExprId| pool.func("gamma", vec![pool.add(vec![u, one])]);
    match pool.get(e) {
        ExprData::Func { name, args } => {
            let args: Vec<ExprId> = args
                .iter()
                .map(|&a| rewrite_combinatorial(a, pool))
                .collect();
            match (name.as_str(), args.as_slice()) {
                ("factorial", [u]) => g(*u),
                ("binomial", [p, q]) => {
                    let diff = pool.add(vec![*p, mk_mul(pool, vec![pool.integer(-1), *q])]);
                    mk_mul(
                        pool,
                        vec![
                            g(*p),
                            pool.pow(g(*q), pool.integer(-1)),
                            pool.pow(g(diff), pool.integer(-1)),
                        ],
                    )
                }
                _ => pool.func(name, args),
            }
        }
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| rewrite_combinatorial(x, pool)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| rewrite_combinatorial(x, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(
            rewrite_combinatorial(base, pool),
            rewrite_combinatorial(exp, pool),
        ),
        _ => e,
    }
}

/// `arg = a·x + b` with rational `a`, `b`.
fn linear(arg: ExprId, var: ExprId, pool: &ExprPool) -> Option<(Rational, Rational)> {
    let at = |v: i64| {
        let mut m = HashMap::new();
        m.insert(var, pool.integer(v));
        rational_literal(pool, simplify(subs(arg, &m, pool), pool).value)
    };
    let b = at(0)?;
    let a = at(1)? - &b;
    let rebuilt = mk_add(
        pool,
        vec![
            mk_mul(pool, vec![rat_node(pool, &a), var]),
            rat_node(pool, &b),
        ],
    );
    let diff = simplify(
        mk_add(
            pool,
            vec![arg, mk_mul(pool, vec![pool.integer(-1), rebuilt])],
        ),
        pool,
    )
    .value;
    is_zero_lit(diff, pool).then_some((a, b))
}

/// A positive slope `a` of `arg = a·x` (no offset).
fn slope(arg: ExprId, var: ExprId, pool: &ExprPool) -> Option<Rational> {
    let (a, b) = linear(arg, var, pool)?;
    (a > 0 && b == 0).then_some(a)
}

/// `arg = a·x^r` with rational `a > 0`, `r > 0`.
fn monomial(arg: ExprId, var: ExprId, pool: &ExprPool) -> Option<(Rational, Rational)> {
    let power = |e: ExprId| -> Option<Rational> {
        if e == var {
            return Some(Rational::from(1));
        }
        match pool.get(e) {
            ExprData::Pow { base, exp } if base == var => rational_literal(pool, exp),
            ExprData::Func { name, args }
                if name == "sqrt" && args.len() == 1 && args[0] == var =>
            {
                Some(rq(1, 2))
            }
            _ => None,
        }
    };
    // √(c·x^r) = √c·x^{r/2} when √c is rational.
    if let ExprData::Func { name, args } = pool.get(arg) {
        if name == "sqrt" && args.len() == 1 && args[0] != var {
            let (c, r) = monomial(args[0], var, pool)?;
            let (n, d) = (c.numer().clone(), c.denom().clone());
            if !n.is_perfect_square() || !d.is_perfect_square() {
                return None;
            }
            return Some((Rational::from((n.sqrt(), d.sqrt())), r / 2));
        }
    }
    let (a, r) = match pool.get(arg) {
        ExprData::Mul(xs) if xs.len() == 2 => {
            let (c, m) = if rational_literal(pool, xs[0]).is_some() {
                (xs[0], xs[1])
            } else {
                (xs[1], xs[0])
            };
            (rational_literal(pool, c)?, power(m)?)
        }
        _ => match power(arg) {
            Some(r) => (Rational::from(1), r),
            None => (slope(arg, var, pool)?, Rational::from(1)),
        },
    };
    (a > 0 && r > 0).then_some((a, r))
}

/// `base^k` for `base = a·x^r` (or `√(a·x^r)`): `(a^k, r·k)`, the constant as
/// an exact node. `None` for anything else.
fn monomial_power(
    base: ExprId,
    k: &Rational,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, Rational)> {
    let (inner, k) = match pool.get(base) {
        ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
            (args[0], Rational::from(k / 2))
        }
        _ => (base, k.clone()),
    };
    if inner == var {
        return None;
    }
    let (a, r) = monomial(inner, var, pool)?;
    let c = if a == 1 {
        pool.integer(1)
    } else {
        pool.pow(rat_node(pool, &a), rat_node(pool, &k))
    };
    Some((simplify(c, pool).value, r * k))
}

/// Build a series in `z = a·t` with the builder, then put `t = x^r`: powers
/// and the truncation scale by `r`, and `t` inside the scale and the
/// oscillatory coefficients becomes `x^r`. `erfc(√x)` — what `uppergamma(½, x)`
/// simplifies to — is the case that needs it.
fn via_monomial(
    u: ExprId,
    var: ExprId,
    pool: &ExprPool,
    build: impl FnOnce(&Rational, ExprId) -> Series,
) -> Option<Series> {
    let (a, r) = monomial(u, var, pool)?;
    if r == 1 {
        return Some(build(&a, var));
    }
    let t = pool.symbol("__asy_mono", crate::kernel::Domain::Positive);
    let s = build(&a, t);
    let mut m = HashMap::new();
    m.insert(t, pool.pow(var, rat_node(pool, &r)));
    let put = |e: ExprId| simplify(subs(e, &m, pool), pool).value;
    let terms = s
        .terms
        .into_iter()
        .map(|tm| Term {
            pow: Rational::from(&tm.pow * &r),
            logp: tm.logp,
            coeff: put(tm.coeff),
            amp: tm.amp,
        })
        .collect();
    Some(Series {
        scale: s.scale.map(put),
        terms,
        cut: s.cut.map(|c| c * &r),
        x_min: s.x_min.powf(1.0 / q_f64(&r)).max(1.0),
    })
}

/// The series of one special-function factor `head(args)` (power 1), or
/// `None` when it is not one this module knows.
fn atom_series(
    name: &str,
    args: &[ExprId],
    var: ExprId,
    k_max: usize,
    pool: &ExprPool,
) -> Option<Series> {
    let order = |e: ExprId| rational_literal(pool, e);
    match (name, args) {
        ("loggamma", [u]) => {
            let (a, b) = linear(*u, var, pool)?;
            (a > 0).then(|| loggamma_series(&a, &b, k_max, pool))?
        }
        ("digamma", [u]) => {
            let (a, b) = linear(*u, var, pool)?;
            (a > 0).then(|| digamma_series(&a, &b, k_max, pool))?
        }
        ("trigamma", [u]) => {
            let (a, b) = linear(*u, var, pool)?;
            (a > 0).then(|| polygamma_series(1, &a, &b, k_max, pool))?
        }
        ("polygamma", [n, u]) => {
            let n = order(*n)?;
            if *n.denom() != 1 || !(0..=50).contains(&n) {
                return None;
            }
            let (a, b) = linear(*u, var, pool)?;
            let n = n.numer().to_u32()?;
            (a > 0).then(|| polygamma_series(n, &a, &b, k_max, pool))?
        }
        ("bessel_j0" | "bessel_j1", [u]) => {
            let nu = Rational::from(i64::from(name == "bessel_j1"));
            via_monomial(*u, var, pool, |a, t| {
                bessel_series(Cyl::J, &nu, a, t, k_max, pool)
            })
        }
        ("bessel_j" | "bessel_y" | "bessel_i" | "bessel_k", [nu, u]) => {
            let kind = match name {
                "bessel_j" => Cyl::J,
                "bessel_y" => Cyl::Y,
                "bessel_i" => Cyl::I,
                _ => Cyl::K,
            };
            let nu = order(*nu)?;
            if q_f64(&nu).abs() > 1000.0 {
                return None;
            }
            via_monomial(*u, var, pool, |a, t| {
                bessel_series(kind, &nu, a, t, k_max, pool)
            })
        }
        ("erfc", [u]) => via_monomial(*u, var, pool, |a, t| erfc_series(a, t, k_max, pool)),
        ("Ei", [u]) => via_monomial(*u, var, pool, |a, t| ei_series(a, t, k_max, pool)),
        ("uppergamma", [s, u]) => {
            let s = order(*s)?;
            if q_f64(&s).abs() > 1000.0 {
                return None;
            }
            via_monomial(*u, var, pool, |a, t| {
                uppergamma_series(&s, a, t, k_max, pool)
            })
        }
        ("airy_ai" | "airy_bi" | "airy_ai_prime" | "airy_bi_prime", [u]) => {
            let kind = match name {
                "airy_ai" => Airy::Ai,
                "airy_bi" => Airy::Bi,
                "airy_ai_prime" => Airy::AiPrime,
                _ => Airy::BiPrime,
            };
            via_monomial(*u, var, pool, |a, t| airy_series(kind, a, t, k_max, pool))
        }
        _ => None,
    }
}

/// `g(x) → ±∞` polynomially (its leading power-scale exponent is positive).
fn grows(g: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    power_scale_coeffs(g, var, 4, pool)
        .ok()
        .and_then(|(c, _)| c.iter().map(|(p, _)| *p).max())
        .is_some_and(|p| p > 0)
}

/// The series of one summand (a product), or `None`.
fn product_series(e: ExprId, var: ExprId, k_max: usize, pool: &ExprPool) -> Option<Series> {
    let factors = match pool.get(e) {
        ExprData::Mul(xs) => xs,
        _ => vec![e],
    };
    let mut consts: Vec<ExprId> = Vec::new();
    let mut gammas: Vec<(Rational, Rational, Rational)> = Vec::new();
    let mut series: Vec<Series> = Vec::new();
    let mut log_power: u32 = 0;
    let mut rest: Vec<ExprId> = Vec::new();
    for f in factors {
        if !mentions_var(f, var, pool) {
            consts.push(f);
            continue;
        }
        // base^k with a literal exponent.
        let (base, k) = match pool.get(f) {
            ExprData::Pow { base, exp } => match rational_literal(pool, exp) {
                Some(q) => (base, q),
                None => (f, Rational::from(1)),
            },
            _ => (f, Rational::from(1)),
        };
        match pool.get(base) {
            // x^q: an exact monomial.
            _ if base == var => {
                series.push(Series::exact(vec![Term {
                    pow: k,
                    logp: 0,
                    coeff: pool.integer(1),
                    amp: 1.0,
                }]));
            }
            // (a·x^r)^q and √(a·x^r): the monomial a^q·x^{rq}.
            _ if base != var && monomial_power(base, &k, var, pool).is_some() => {
                let (c, p) = monomial_power(base, &k, var, pool)?;
                series.push(Series::exact(vec![term(p, 0, c, pool)]));
            }
            // √x^q = x^{q/2}.
            ExprData::Func { name, args }
                if name == "sqrt" && args.len() == 1 && args[0] == var =>
            {
                series.push(Series::exact(vec![Term {
                    pow: k / 2,
                    logp: 0,
                    coeff: pool.integer(1),
                    amp: 1.0,
                }]));
            }
            // exp(g(x)): a pure exponential scale (closed forms such as
            // uppergamma(5/2, x) = e^{−x}(…) + √π erfc(√x)·(…) mix one with
            // the special-function terms of the same scale).
            ExprData::Func { name, args }
                if name == "exp" && args.len() == 1 && grows(args[0], var, pool) =>
            {
                let mut sc = Series::exact(vec![Term {
                    pow: Rational::new(),
                    logp: 0,
                    coeff: pool.integer(1),
                    amp: 1.0,
                }]);
                let arg = if k == 1 {
                    args[0]
                } else {
                    simplify(mk_mul(pool, vec![rat_node(pool, &k), args[0]]), pool).value
                };
                sc.scale = Some(simplify(pool.func("exp", vec![arg]), pool).value);
                series.push(sc);
            }
            ExprData::Func { name, args } if name == "gamma" && args.len() == 1 => {
                let (a, b) = linear(args[0], var, pool)?;
                if a <= 0 {
                    return None;
                }
                gammas.push((a, b, k));
            }
            ExprData::Func { name, args } if name == "log" && args.len() == 1 && args[0] == var => {
                if *k.denom() != 1 || k <= 0 {
                    rest.push(f);
                    continue;
                }
                log_power += k.numer().to_u32()?;
            }
            ExprData::Func { name, args } if SPECIAL_HEADS.contains(&name.as_str()) => {
                // Positive integer powers only (a reciprocal needs a series
                // inverse, which is not formed here).
                if *k.denom() != 1 || k <= 0 || k > 8 {
                    return None;
                }
                let s = atom_series(&name, &args, var, k_max, pool)?;
                for _ in 0..k.numer().to_u32()? {
                    series.push(s.clone());
                }
            }
            _ => {
                if mentions_special(f, var, pool) {
                    return None;
                }
                rest.push(f);
            }
        }
    }
    if !gammas.is_empty() {
        series.push(gamma_product(&gammas, var, k_max, pool)?);
    }
    if log_power > 0 {
        series.push(Series::exact(vec![Term {
            pow: Rational::new(),
            logp: log_power,
            coeff: pool.integer(1),
            amp: 1.0,
        }]));
    }
    if !rest.is_empty() {
        let h = simplify(pool.mul(rest), pool).value;
        let (coeffs, cut) = power_scale_coeffs(h, var, (k_max as u32 + 4).min(60), pool).ok()?;
        let terms = coeffs
            .into_iter()
            .map(|(p, c)| term(Rational::from(p), 0, c, pool))
            .collect();
        series.push(Series {
            scale: None,
            terms,
            cut: Some(Rational::from(cut)),
            x_min: 1.0,
        });
    }
    let mut it = series.into_iter();
    let mut acc = it.next()?.normalize(pool);
    for s in it {
        acc = acc.mul(&s.normalize(pool), pool);
    }
    if !consts.is_empty() {
        acc = acc.scale_by(simplify(mk_mul(pool, consts), pool).value, pool);
    }
    Some(acc)
}

/// The summands of `e` after distributing every product over a sum factor
/// that mentions a special function (closed forms such as
/// `uppergamma(5/2, x) = e^{−x}x^{3/2} + 3/2·(e^{−x}x^{1/2} + √π/2·erfc(√x))`
/// nest them). Polynomial cofactors are left alone.
fn summands(e: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>, budget: &mut usize) {
    if *budget == 0 {
        out.push(e);
        return;
    }
    *budget -= 1;
    match pool.get(e) {
        ExprData::Add(xs) => {
            for x in xs {
                summands(x, var, pool, out, budget);
            }
        }
        ExprData::Mul(xs) => {
            let pos = xs.iter().position(|&f| {
                matches!(pool.get(f), ExprData::Add(_)) && mentions_special(f, var, pool)
            });
            match pos {
                Some(i) => {
                    let ExprData::Add(parts) = pool.get(xs[i]) else {
                        unreachable!()
                    };
                    for part in parts {
                        let mut fs = xs.clone();
                        fs[i] = part;
                        summands(simplify(pool.mul(fs), pool).value, var, pool, out, budget);
                    }
                }
                None => out.push(e),
            }
        }
        _ => out.push(e),
    }
}

fn total_series(f: ExprId, var: ExprId, k_max: usize, pool: &ExprPool) -> Option<Series> {
    let mut parts = Vec::new();
    let mut budget = 64usize;
    summands(f, var, pool, &mut parts, &mut budget);
    let summands = parts;
    let mut acc: Option<Series> = None;
    for s in summands {
        crate::budget::check().ok()?;
        let si = product_series(s, var, k_max, pool)?;
        acc = Some(match acc {
            None => si,
            Some(a) => a.add(&si, pool)?,
        });
    }
    acc
}

// ---------------------------------------------------------------------------
// Entry point and gate
// ---------------------------------------------------------------------------

/// `Some(result)` when `f` contains a special function this module expands
/// (the result is then final: an expansion, or `E-ASYMPT-004` when the gate
/// confirmed nothing); `None` when it does not apply.
pub(super) fn special_expand(
    f: ExprId,
    var: ExprId,
    n_terms: usize,
    pool: &ExprPool,
) -> Option<Result<AsymptoticExpansion, AsymptoticError>> {
    let special = mentions_special(f, var, pool);
    if !special && !has_growing_exp(f, var, pool) {
        return None;
    }
    let f = simplify(rewrite_combinatorial(f, pool), pool).value;
    let n = n_terms.min(MAX_TERMS);
    let mut k_max = 2 * n + 6;
    let series = loop {
        let s = total_series(f, var, k_max, pool)?;
        // n terms plus the first omitted one (the gate's envelope), unless
        // the series is exact.
        if s.terms.len() > n || s.cut.is_none() || k_max >= 4 * MAX_TERMS {
            break s;
        }
        k_max *= 2;
    };
    if series.terms.is_empty() {
        return None;
    }
    let take = series.terms.len().min(n);
    let unscaled: Vec<ExprId> = series.terms[..take]
        .iter()
        .map(|t| unscaled_expr(t, var, pool))
        .collect();
    let next = series.terms.get(take);
    let accepted = gate(gate_form(f, pool), var, &series, &unscaled, next, pool);
    if accepted == 0 {
        // Without a special function the older exp/log peel gets its turn.
        return special.then_some(Err(AsymptoticError::GateFailed));
    }
    let terms = unscaled[..accepted]
        .iter()
        .map(|&u| {
            let expr = match series.scale {
                Some(s) => simplify(mk_mul(pool, vec![s, u]), pool).value,
                None => u,
            };
            AsymptoticTerm { expr }
        })
        .collect();
    Some(Ok(AsymptoticExpansion { terms }))
}

/// `f` with every `trigamma(u)` spelled `polygamma(1, u)` (built without
/// simplification, which would respell it back): the gate needs a ball at
/// 640 bits, and Arb's polygamma gives one where the unary kernel does not.
fn gate_form(e: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(e) {
        ExprData::Func { name, args } => {
            let args: Vec<ExprId> = args.iter().map(|&a| gate_form(a, pool)).collect();
            if name == "trigamma" && args.len() == 1 {
                pool.func("polygamma", vec![pool.integer(1), args[0]])
            } else {
                pool.func(name, args)
            }
        }
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| gate_form(x, pool)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| gate_form(x, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(gate_form(base, pool), gate_form(exp, pool)),
        _ => e,
    }
}

/// Some factor of some summand is `exp(g)` with `g → ±∞` polynomially
/// (`Γ(3, x) = 2e^{−x}(x²/2 + x + 1)` once simplified). Such products are
/// expanded here too, with the ball gate; the f64 gate of the older peel
/// underflows at `e^{−10⁴}`.
fn has_growing_exp(f: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let is_exp = |e: ExprId| match pool.get(e) {
        ExprData::Func { name, args } => {
            name == "exp" && args.len() == 1 && grows(args[0], var, pool)
        }
        _ => false,
    };
    let in_product = |e: ExprId| match pool.get(e) {
        ExprData::Mul(xs) => xs.iter().any(|&x| is_exp(x)),
        _ => is_exp(e),
    };
    match pool.get(f) {
        ExprData::Add(xs) => xs.iter().any(|&x| in_product(x)),
        _ => in_product(f),
    }
}

fn unscaled_expr(t: &Term, var: ExprId, pool: &ExprPool) -> ExprId {
    let mut parts = vec![t.coeff];
    if t.pow != 0 {
        parts.push(pool.pow(var, rat_node(pool, &t.pow)));
    }
    if t.logp > 0 {
        let l = pool.func("log", vec![var]);
        parts.push(if t.logp == 1 {
            l
        } else {
            pool.pow(l, pool.integer(i64::from(t.logp)))
        });
    }
    simplify(mk_mul(pool, parts), pool).value
}

/// `amp · x^pow · log(x)^logp` at `x`.
fn envelope(t: &Term, x: f64) -> f64 {
    t.amp * x.powf(q_f64(&t.pow)) * x.ln().powi(t.logp as i32)
}

/// A ball whose midpoint is a nonzero number well inside MPFR's exponent
/// range. Far out, `erfc(3x)` and its scale `e^{−9x²}` underflow; their
/// quotient would then be `0/tiny`, not the function, so such a point is
/// skipped rather than trusted.
fn representable(b: &ArbBall) -> bool {
    const LIMIT: i32 = 1 << 28;
    b.mid.get_exp().is_some_and(|e| e.abs() < LIMIT) && b.rad.is_finite()
}

/// Upper bound of `|b|`, as an `f64` (`inf` when not finite).
fn upper(b: &ArbBall) -> f64 {
    b.mid_f64().abs() + b.rad_f64()
}

/// The number of leading terms confirmed at every resolving gate point
/// (module docs).
fn gate(
    f: ExprId,
    var: ExprId,
    series: &Series,
    unscaled: &[ExprId],
    next: Option<&Term>,
    pool: &ExprPool,
) -> usize {
    let base = 100.0 * series.x_min.max(1.0);
    let points = [base, 3.0 * base, 10.0 * base, 100.0 * base];
    let n = unscaled.len();
    // ok[k] for k in 1..=n: the prefix of k terms passed at a resolving point
    // and failed nowhere.
    let mut failed = vec![false; n + 1];
    let mut resolved = vec![false; n + 1];
    for &xv in &points {
        let mut ev = IntervalEval::new(GATE_PREC);
        ev.bind(var, ArbBall::from_f64(xv, GATE_PREC));
        // A point where the function (or its scale) has no ball value — Ei's
        // power series gives up far out — is skipped; every accepted term
        // must still be confirmed at one point at least.
        let Some(fv) = ev.eval(f, pool).filter(representable) else {
            continue;
        };
        let g = match series.scale {
            Some(s) => match ev.eval(s, pool).filter(representable) {
                Some(sv) if !sv.contains(0.0) => match fv / sv {
                    Some(q) => q,
                    None => continue,
                },
                _ => continue,
            },
            None => fv,
        };
        let gmag = upper(&g);
        let mut residual = g;
        for k in 1..=n {
            let Some(tv) = ev.eval(unscaled[k - 1], pool) else {
                failed[k] = true;
                break;
            };
            residual = residual - tv;
            let env = if k < n {
                envelope(&series.terms[k], xv)
            } else {
                match next {
                    Some(t) => envelope(t, xv),
                    // Exact: what remains is beyond all orders.
                    None if series.cut.is_none() => 1e-30 * gmag,
                    // Every term the series has was asked for: the rest
                    // is O(x^cut), sized by the leading amplitude.
                    None => match &series.cut {
                        Some(c) => series.terms[0].amp * xv.powf(q_f64(c)),
                        None => 0.0,
                    },
                }
            };
            let rad = residual.rad_f64();
            if !(env.is_finite() && env > 0.0 && rad.is_finite() && rad <= env) {
                continue; // not resolvable here
            }
            resolved[k] = true;
            if upper(&residual) > GATE_SLACK * env {
                failed[k] = true;
            }
        }
    }
    let mut accepted = 0;
    for k in 1..=n {
        if failed[k] || !resolved[k] {
            break;
        }
        accepted = k;
    }
    accepted
}

#[cfg(test)]
#[path = "special_tests.rs"]
mod tests;
