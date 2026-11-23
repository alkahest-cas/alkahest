//! Exact roots of a characteristic (or indicial) polynomial over ℚ, and the
//! exact undetermined-coefficients particular solution that goes with them.
//!
//! The polynomial is factored over ℤ by FLINT. Each irreducible factor gives
//! its roots in closed form:
//!
//! | degree | form |
//! |---|---|
//! | 1 | rational |
//! | 2 | `(−b ± √D)/2a`, or `α ± iβ` with `β = √(−D)/2a` |
//! | 3 | Cardano with **real** cube roots (one real root), or the trigonometric form (three real roots) |
//! | 4 | Ferrari: the quartic is split into two real quadratics through a positive root of its resolvent cubic |
//! | ≥ 5 | `RootOf(p, k)` when every root of the factor is real; otherwise declined |
//!
//! A complex pair is always reported as real `α`, `β > 0`, so the caller's
//! basis is the real `e^{αx}cos βx, e^{αx}sin βx`. Nothing here is a float:
//! a sign that has to be decided for a constant radical expression (which of
//! two real quadratics has real roots, which resolvent root is positive) is
//! read off a numeric evaluation of the exact expression, and the expression
//! itself is what is returned — the caller verifies the final solution by
//! substitution in any case.

use std::collections::HashMap;

use rug::{Integer, Rational};

use super::DsolveError;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

/// One distinct root (or conjugate pair) with its multiplicity.
#[derive(Clone, Debug)]
pub(crate) enum ExactRoot {
    Real {
        r: ExprId,
        m: usize,
    },
    /// `α ± iβ` with `β > 0`.
    Complex {
        alpha: ExprId,
        beta: ExprId,
        m: usize,
    },
}

fn simp(e: ExprId, pool: &ExprPool) -> ExprId {
    simplify(e, pool).value
}

pub(crate) fn rat(q: &Rational, pool: &ExprPool) -> ExprId {
    if *q.denom() == 1 {
        pool.integer(q.numer().clone())
    } else {
        pool.rational(q.numer().clone(), q.denom().clone())
    }
}

fn sqrt(e: ExprId, pool: &ExprPool) -> ExprId {
    simp(pool.func("sqrt", vec![e]), pool)
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

/// Numeric value of a constant expression (for sign decisions only).
fn value(e: ExprId, pool: &ExprPool) -> Option<f64> {
    crate::jit::eval_interp(e, &HashMap::new(), pool).filter(|v| v.is_finite())
}

/// A rational literal.
pub(crate) fn literal(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

/// Roots of `Σ a[k] λ^k` (rational coefficients, `a.last() ≠ 0`).
pub(crate) fn exact_roots(a: &[Rational], pool: &ExprPool) -> Result<Vec<ExactRoot>, DsolveError> {
    let mut a: Vec<Rational> = a.to_vec();
    while a.len() > 1 && a.last().is_some_and(|c| *c == 0) {
        a.pop();
    }
    if a.len() < 2 {
        return Ok(Vec::new());
    }
    let mut den = Integer::from(1);
    for c in &a {
        den = den.lcm(c.denom());
    }
    let ints: Vec<Integer> = a
        .iter()
        .map(|c| (c.clone() * Rational::from(den.clone())).numer().clone())
        .collect();
    let tmp = ExprPool::new();
    let lam = tmp.symbol("lam", Domain::Real);
    let poly_expr = int_poly_expr(&ints, lam, &tmp);
    let up = crate::poly::UniPoly::from_symbolic(poly_expr, lam, &tmp)
        .map_err(|e| DsolveError::Unsupported(format!("characteristic polynomial: {e}")))?;
    let fac = up
        .factor_z()
        .map_err(|e| DsolveError::Unsupported(format!("characteristic polynomial: {e}")))?;
    let mut out: Vec<ExactRoot> = Vec::new();
    for (f, mult) in &fac.factors {
        let c = f.coefficients();
        let q: Vec<Rational> = c.iter().map(|v| Rational::from(v.clone())).collect();
        let m = *mult as usize;
        let roots = irreducible_roots(&q, &c, pool)?;
        for r in roots {
            out.push(match r {
                ExactRoot::Real { r, .. } => ExactRoot::Real { r, m },
                ExactRoot::Complex { alpha, beta, .. } => ExactRoot::Complex { alpha, beta, m },
            });
        }
    }
    Ok(out)
}

fn int_poly_expr(c: &[Integer], var: ExprId, pool: &ExprPool) -> ExprId {
    let terms: Vec<ExprId> = c
        .iter()
        .enumerate()
        .map(|(i, v)| {
            pool.mul(vec![
                pool.integer(v.clone()),
                pool.pow(var, pool.integer(i as i64)),
            ])
        })
        .collect();
    pool.add(terms)
}

/// Roots of one irreducible factor with coefficients `q` (low → high).
fn irreducible_roots(
    q: &[Rational],
    ints: &[Integer],
    pool: &ExprPool,
) -> Result<Vec<ExactRoot>, DsolveError> {
    let n = q.len() - 1;
    let lc = q[n].clone();
    let monic: Vec<Rational> = q.iter().map(|c| c.clone() / lc.clone()).collect();
    match n {
        1 => Ok(vec![ExactRoot::Real {
            r: rat(&(-monic[0].clone()), pool),
            m: 1,
        }]),
        2 => Ok(quadratic_rational(&monic[1], &monic[0], pool)),
        3 => Ok(cubic(&monic[2], &monic[1], &monic[0], pool)),
        4 => quartic(&monic, pool),
        _ => root_of_roots(ints, pool),
    }
}

/// Roots of `λ² + p λ + s` with rational `p`, `s`.
fn quadratic_rational(p: &Rational, s: &Rational, pool: &ExprPool) -> Vec<ExactRoot> {
    let disc = p.clone() * p.clone() - Rational::from(4) * s.clone();
    let half_p = p.clone() / Rational::from(2);
    let alpha = rat(&(-half_p), pool);
    if disc > 0 {
        let sq = sqrt(rat(&disc, pool), pool);
        let half = pool.rational(1, 2);
        vec![
            ExactRoot::Real {
                r: simp(pool.add(vec![alpha, pool.mul(vec![half, sq])]), pool),
                m: 1,
            },
            ExactRoot::Real {
                r: simp(
                    pool.add(vec![alpha, neg(pool.mul(vec![half, sq]), pool)]),
                    pool,
                ),
                m: 1,
            },
        ]
    } else if disc == 0 {
        vec![ExactRoot::Real { r: alpha, m: 2 }]
    } else {
        let beta = simp(
            pool.mul(vec![pool.rational(1, 2), sqrt(rat(&(-disc), pool), pool)]),
            pool,
        );
        vec![ExactRoot::Complex { alpha, beta, m: 1 }]
    }
}

/// Roots of `λ² + u λ + v` with constant (radical) `u`, `v`, the sign of the
/// discriminant decided numerically. `None` on a numerically zero
/// discriminant (cannot happen for a factor of an irreducible quartic).
fn quadratic_expr(u: ExprId, v: ExprId, pool: &ExprPool) -> Option<Vec<ExactRoot>> {
    let disc = simp(
        pool.add(vec![
            pool.pow(u, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(-4_i32), v]),
        ]),
        pool,
    );
    let d = value(disc, pool)?;
    let scale = value(pool.pow(u, pool.integer(2_i32)), pool)?.abs() + value(v, pool)?.abs();
    if d.abs() <= 1e-12 * scale.max(1.0) {
        return None;
    }
    let half = pool.rational(1, 2);
    let alpha = simp(pool.mul(vec![pool.rational(-1, 2), u]), pool);
    if d > 0.0 {
        let sq = sqrt(disc, pool);
        Some(vec![
            ExactRoot::Real {
                r: simp(pool.add(vec![alpha, pool.mul(vec![half, sq])]), pool),
                m: 1,
            },
            ExactRoot::Real {
                r: simp(
                    pool.add(vec![alpha, neg(pool.mul(vec![half, sq]), pool)]),
                    pool,
                ),
                m: 1,
            },
        ])
    } else {
        let beta = simp(pool.mul(vec![half, sqrt(neg(disc, pool), pool)]), pool);
        Some(vec![ExactRoot::Complex { alpha, beta, m: 1 }])
    }
}

/// `√q` as a rational when `q ≥ 0` is a square of one.
fn rational_sqrt(q: &Rational) -> Option<Rational> {
    if *q < 0 {
        return None;
    }
    let (n, d) = (q.numer(), q.denom());
    (n.is_perfect_square() && d.is_perfect_square())
        .then(|| Rational::from((n.clone().sqrt(), d.clone().sqrt())))
}

/// Real cube root of `r ± √d` (`d ≥ 0` rational), written so that the base of
/// the radical is non-negative; see `matrix::eigen`.
fn real_cbrt(r: &Rational, d: &Rational, minus: bool, pool: &ExprPool) -> ExprId {
    let third = pool.rational(1, 3);
    // A square `d`: the radicand is rational, possibly zero (`λ³ = 2` has
    // `−q/2 = √Δ`, so one Cardano radical vanishes).
    if let Some(sd) = rational_sqrt(d) {
        let w = if minus {
            r.clone() - sd
        } else {
            r.clone() + sd
        };
        if w == 0 {
            return pool.integer(0_i32);
        }
        let mag = simp(
            pool.pow(rat(&Rational::from(w.abs_ref()), pool), third),
            pool,
        );
        return if w < 0 {
            simp(neg(mag, pool), pool)
        } else {
            mag
        };
    }
    let sqrt_d = sqrt(rat(d, pool), pool);
    let r_sq = r.clone() * r.clone();
    let nonneg = if minus {
        *r >= 0 && *d <= r_sq
    } else {
        *r >= 0 || *d >= r_sq
    };
    let (rp, sp) = if nonneg {
        (r.clone(), if minus { neg(sqrt_d, pool) } else { sqrt_d })
    } else {
        (-r.clone(), if minus { sqrt_d } else { neg(sqrt_d, pool) })
    };
    let mag = simp(pool.pow(pool.add(vec![rat(&rp, pool), sp]), third), pool);
    if nonneg {
        mag
    } else {
        simp(neg(mag, pool), pool)
    }
}

/// Roots of the monic cubic `λ³ + a λ² + b λ + c` (irreducible over ℚ).
fn cubic(a: &Rational, b: &Rational, c: &Rational, pool: &ExprPool) -> Vec<ExactRoot> {
    let a2 = a.clone() * a.clone();
    let p = b.clone() - a2.clone() / Rational::from(3);
    let q = c.clone()
        + (Rational::from(2) * a2 * a.clone() - Rational::from(9) * a.clone() * b.clone())
            / Rational::from(27);
    let shift = rat(&(-(a.clone() / Rational::from(3))), pool);
    let half_q = q.clone() / Rational::from(2);
    let third_p = p.clone() / Rational::from(3);
    let delta =
        half_q.clone() * half_q.clone() + third_p.clone() * third_p.clone() * third_p.clone();
    if delta < 0 {
        // Three real roots: t_k = 2√(−p/3) cos((θ + 2πk)/3), cos θ = (−q/2)/(−p/3)^{3/2}.
        let m = -third_p;
        let sq = sqrt(rat(&m, pool), pool);
        let cos_theta = simp(
            pool.mul(vec![
                rat(&(-half_q), pool),
                pool.pow(sq, pool.integer(-3_i32)),
            ]),
            pool,
        );
        let theta = pool.func("acos", vec![cos_theta]);
        let pi = pool.symbol("pi", Domain::Real);
        (0..3)
            .map(|k| {
                let angle = pool.mul(vec![
                    pool.rational(1, 3),
                    pool.add(vec![theta, pool.mul(vec![pool.integer(2 * k), pi])]),
                ]);
                ExactRoot::Real {
                    r: simp(
                        pool.add(vec![
                            pool.mul(vec![pool.integer(2_i32), sq, pool.func("cos", vec![angle])]),
                            shift,
                        ]),
                        pool,
                    ),
                    m: 1,
                }
            })
            .collect()
    } else {
        // One real root t₀ = A + B; pair −(A+B)/2 ± i(√3/2)(A − B), A > B.
        let nhq = -half_q;
        let big_a = real_cbrt(&nhq, &delta, false, pool);
        let big_b = real_cbrt(&nhq, &delta, true, pool);
        let sum = pool.add(vec![big_a, big_b]);
        let diff = pool.add(vec![big_a, neg(big_b, pool)]);
        vec![
            ExactRoot::Real {
                r: simp(pool.add(vec![sum, shift]), pool),
                m: 1,
            },
            ExactRoot::Complex {
                alpha: simp(
                    pool.add(vec![pool.mul(vec![pool.rational(-1, 2), sum]), shift]),
                    pool,
                ),
                beta: simp(
                    pool.mul(vec![
                        pool.rational(1, 2),
                        sqrt(pool.integer(3_i32), pool),
                        diff,
                    ]),
                    pool,
                ),
                m: 1,
            },
        ]
    }
}

/// Roots of the monic quartic `λ⁴ + a₃λ³ + a₂λ² + a₁λ + a₀` (irreducible),
/// by splitting the depressed quartic into two real quadratics.
fn quartic(a: &[Rational], pool: &ExprPool) -> Result<Vec<ExactRoot>, DsolveError> {
    let (a0, a1, a2, a3) = (a[0].clone(), a[1].clone(), a[2].clone(), a[3].clone());
    let a3s = a3.clone() * a3.clone();
    let p = a2.clone() - Rational::from(3) * a3s.clone() / Rational::from(8);
    let q = a1.clone() - a3.clone() * a2.clone() / Rational::from(2)
        + a3s.clone() * a3.clone() / Rational::from(8);
    let r = a0.clone() - a3.clone() * a1.clone() / Rational::from(4)
        + a3s.clone() * a2.clone() / Rational::from(16)
        - Rational::from(3) * a3s.clone() * a3s.clone() / Rational::from(256);
    let shift = rat(&(-(a3 / Rational::from(4))), pool);
    let fail = || {
        DsolveError::Unsupported(
            "could not split the irreducible quartic characteristic factor into real quadratics"
                .to_string(),
        )
    };
    // (t² + u t + v)(t² − u t + w)
    let (u, v, w) = if q == 0 {
        let disc = p.clone() * p.clone() - Rational::from(4) * r.clone();
        if disc >= 0 {
            // t² = μ±: two quadratics t² − μ.
            let sq = sqrt(rat(&disc, pool), pool);
            let mut roots = Vec::new();
            for sign in [1, -1] {
                let mu = simp(
                    pool.mul(vec![
                        pool.rational(1, 2),
                        pool.add(vec![
                            rat(&(-p.clone()), pool),
                            pool.mul(vec![pool.integer(sign), sq]),
                        ]),
                    ]),
                    pool,
                );
                let zero = pool.integer(0_i32);
                roots.extend(quadratic_expr(zero, neg(mu, pool), pool).ok_or_else(fail)?);
            }
            return Ok(shift_roots(roots, shift, pool));
        }
        // P² < 4R: t⁴ + P t² + R = (t² + u t + √R)(t² − u t + √R), u² = 2√R − P.
        let s = sqrt(rat(&r, pool), pool);
        let u = sqrt(
            simp(
                pool.add(vec![
                    pool.mul(vec![pool.integer(2_i32), s]),
                    rat(&(-p), pool),
                ]),
                pool,
            ),
            pool,
        );
        (u, s, s)
    } else {
        // Resolvent: U³ + 2P U² + (P² − 4R) U − Q² = 0, take a positive root.
        let res = [
            -(q.clone() * q.clone()),
            p.clone() * p.clone() - Rational::from(4) * r.clone(),
            Rational::from(2) * p.clone(),
            Rational::from(1),
        ];
        let mut best: Option<(f64, ExprId)> = None;
        for root in exact_roots(&res, pool)? {
            if let ExactRoot::Real { r: big_u, .. } = root {
                if let Some(val) = value(big_u, pool) {
                    if val > 1e-12 && !matches!(best, Some((b, _)) if val <= b) {
                        best = Some((val, big_u));
                    }
                }
            }
        }
        let (_, big_u) = best.ok_or_else(fail)?;
        let u = sqrt(big_u, pool);
        let q_over_u = pool.mul(vec![rat(&q, pool), pool.pow(u, pool.integer(-1_i32))]);
        let base = pool.add(vec![rat(&p, pool), big_u]);
        let v = simp(
            pool.mul(vec![
                pool.rational(1, 2),
                pool.add(vec![base, neg(q_over_u, pool)]),
            ]),
            pool,
        );
        let w = simp(
            pool.mul(vec![pool.rational(1, 2), pool.add(vec![base, q_over_u])]),
            pool,
        );
        (u, v, w)
    };
    let mut roots = quadratic_expr(u, v, pool).ok_or_else(fail)?;
    roots.extend(quadratic_expr(simp(neg(u, pool), pool), w, pool).ok_or_else(fail)?);
    Ok(shift_roots(roots, shift, pool))
}

fn shift_roots(roots: Vec<ExactRoot>, shift: ExprId, pool: &ExprPool) -> Vec<ExactRoot> {
    roots
        .into_iter()
        .map(|r| match r {
            ExactRoot::Real { r, m } => ExactRoot::Real {
                r: simp(pool.add(vec![r, shift]), pool),
                m,
            },
            ExactRoot::Complex { alpha, beta, m } => ExactRoot::Complex {
                alpha: simp(pool.add(vec![alpha, shift]), pool),
                beta,
                m,
            },
        })
        .collect()
}

/// Degree ≥ 5: `RootOf(p, k)` for every root, when all are real.
fn root_of_roots(ints: &[Integer], pool: &ExprPool) -> Result<Vec<ExactRoot>, DsolveError> {
    let lam = pool.symbol("__dsolve_lambda", Domain::Real);
    let p = int_poly_expr(ints, lam, pool);
    let n = ints.len() - 1;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let r = crate::poly::root_of::root_of(p, lam, k, pool)
            .map_err(|e| DsolveError::Unsupported(format!("RootOf: {e}")))?;
        let real = crate::poly::root_of::root_enclosure(r, pool, 128).is_some_and(|e| e.is_real);
        if !real {
            return Err(DsolveError::Unsupported(format!(
                "the characteristic polynomial has an irreducible factor of degree {n} with \
                 non-real roots; no radical form exists in general and its real and imaginary \
                 parts are not represented exactly here"
            )));
        }
        out.push(ExactRoot::Real { r, m: 1 });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Exact undetermined coefficients over ℚ(i)
// ---------------------------------------------------------------------------

/// Gaussian rational `re + i·im`.
#[derive(Clone, Debug, PartialEq)]
struct Gq {
    re: Rational,
    im: Rational,
}

impl Gq {
    fn new(re: Rational, im: Rational) -> Self {
        Gq { re, im }
    }
    fn zero() -> Self {
        Gq::new(Rational::new(), Rational::new())
    }
    fn is_zero(&self) -> bool {
        self.re == 0 && self.im == 0
    }
    fn add(&self, o: &Gq) -> Gq {
        Gq::new(self.re.clone() + &o.re, self.im.clone() + &o.im)
    }
    fn sub(&self, o: &Gq) -> Gq {
        Gq::new(self.re.clone() - &o.re, self.im.clone() - &o.im)
    }
    fn mul(&self, o: &Gq) -> Gq {
        Gq::new(
            self.re.clone() * &o.re - self.im.clone() * &o.im,
            self.re.clone() * &o.im + self.im.clone() * &o.re,
        )
    }
    fn scale(&self, k: &Rational) -> Gq {
        Gq::new(self.re.clone() * k, self.im.clone() * k)
    }
    fn div(&self, o: &Gq) -> Gq {
        let n = o.re.clone() * &o.re + o.im.clone() * &o.im;
        let conj = Gq::new(o.re.clone(), -o.im.clone());
        let t = self.mul(&conj);
        Gq::new(t.re / n.clone(), t.im / n)
    }
}

/// One `e^{ax}(Pc(x) cos bx + Ps(x) sin bx)` family of a forcing term.
#[derive(Clone, Debug)]
pub(crate) struct Family {
    a: Rational,
    b: Rational,
    pc: Vec<Rational>,
    ps: Vec<Rational>,
}

fn add_coeff(v: &mut Vec<Rational>, k: usize, c: Rational) {
    if v.len() <= k {
        v.resize(k + 1, Rational::new());
    }
    v[k] += c;
}

/// `e = c·x` with rational `c` (no constant term).
fn linear_rate(e: ExprId, x: ExprId, pool: &ExprPool) -> Option<Rational> {
    if e == x {
        return Some(Rational::from(1));
    }
    let ExprData::Mul(xs) = pool.get(e) else {
        return None;
    };
    let mut c = Rational::from(1);
    let mut seen = false;
    for f in xs.iter() {
        if let Some(v) = literal(*f, pool) {
            c *= v;
        } else if *f == x && !seen {
            seen = true;
        } else {
            return None;
        }
    }
    seen.then_some(c)
}

/// Split an (expanded) forcing term into families; `None` when some term is
/// not `c·x^k·e^{ax}·{1, cos bx, sin bx}` with rational `c`, `a`, `b`.
pub(crate) fn forcing_families(r: ExprId, x: ExprId, pool: &ExprPool) -> Option<Vec<Family>> {
    let terms: Vec<ExprId> = match pool.get(r) {
        ExprData::Add(xs) => xs.to_vec(),
        _ => vec![r],
    };
    let mut fams: Vec<Family> = Vec::new();
    for t in terms {
        let factors: Vec<ExprId> = match pool.get(t) {
            ExprData::Mul(xs) => xs.to_vec(),
            _ => vec![t],
        };
        let mut c = Rational::from(1);
        let mut k = 0usize;
        let mut a = Rational::new();
        let mut trig: Option<(bool, Rational)> = None; // (is_sin, b)
        for f in factors {
            if let Some(v) = literal(f, pool) {
                c *= v;
                continue;
            }
            if f == x {
                k += 1;
                continue;
            }
            match pool.get(f) {
                ExprData::Pow { base, exp } if base == x => {
                    let e = literal(exp, pool)?;
                    if *e.denom() != 1 || e < 0 {
                        return None;
                    }
                    k += e.numer().to_usize()?;
                }
                ExprData::Pow { base, exp } => {
                    // exp(u)^n
                    let ExprData::Func { name, args } = pool.get(base) else {
                        return None;
                    };
                    if name != "exp" || args.len() != 1 {
                        return None;
                    }
                    let n = literal(exp, pool)?;
                    a += linear_rate(args[0], x, pool)? * n;
                }
                ExprData::Func { name, args } if args.len() == 1 => match name.as_str() {
                    "exp" => a += linear_rate(args[0], x, pool)?,
                    "cos" | "sin" if trig.is_none() => {
                        let mut b = linear_rate(args[0], x, pool)?;
                        let mut is_sin = name == "sin";
                        if b < 0 {
                            b = -b;
                            if is_sin {
                                c = -c;
                            }
                        }
                        if b == 0 {
                            if is_sin {
                                c = Rational::new();
                            }
                            is_sin = false;
                        }
                        trig = Some((is_sin, b));
                    }
                    _ => return None,
                },
                _ => return None,
            }
        }
        let (is_sin, b) = trig.unwrap_or((false, Rational::new()));
        let fam = match fams.iter_mut().position(|f| f.a == a && f.b == b) {
            Some(i) => &mut fams[i],
            None => {
                fams.push(Family {
                    a: a.clone(),
                    b: b.clone(),
                    pc: Vec::new(),
                    ps: Vec::new(),
                });
                fams.last_mut().unwrap()
            }
        };
        if is_sin {
            add_coeff(&mut fam.ps, k, c);
        } else {
            add_coeff(&mut fam.pc, k, c);
        }
    }
    Some(fams)
}

/// `P^{(m)}(μ)/m!` for `m = 0..=deg`.
fn taylor_at(p: &[Rational], mu: &Gq) -> Vec<Gq> {
    let n = p.len();
    let mut out = Vec::with_capacity(n);
    // Repeated synthetic division: coefficients of P(μ + h) in h.
    let mut work: Vec<Gq> = p
        .iter()
        .map(|c| Gq::new(c.clone(), Rational::new()))
        .collect();
    for _ in 0..n {
        // Horner: evaluate and deflate.
        let mut acc = Gq::zero();
        let mut quot = vec![Gq::zero(); work.len().saturating_sub(1)];
        for i in (0..work.len()).rev() {
            acc = acc.mul(mu).add(&work[i]);
            if i > 0 {
                quot[i - 1] = acc.clone();
            }
        }
        out.push(acc);
        work = quot;
        if work.is_empty() {
            break;
        }
    }
    out
}

/// Exact particular solution of `Σ p[k] y^(k) = Σ families` (constant
/// rational coefficients), as an expression in `x`.
pub(crate) fn particular_exact(
    p: &[Rational],
    fams: &[Family],
    x: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let mut parts = Vec::new();
    for fam in fams {
        let mu = Gq::new(fam.a.clone(), fam.b.clone());
        let d = fam.pc.len().max(fam.ps.len());
        if d == 0 {
            continue;
        }
        // F = Pc − i Ps (b > 0) or Pc (b = 0).
        let f: Vec<Gq> = (0..d)
            .map(|k| {
                let pc = fam.pc.get(k).cloned().unwrap_or_default();
                let ps = fam.ps.get(k).cloned().unwrap_or_default();
                if fam.b == 0 {
                    Gq::new(pc, Rational::new())
                } else {
                    Gq::new(pc, -ps)
                }
            })
            .collect();
        let c = taylor_at(p, &mu);
        let s = c.iter().position(|v| !v.is_zero())?;
        // g = Σ_{j=s}^{s+d−1} g_j x^j with Σ_m c_m g^{(m)} = F.
        let top = s + d;
        let mut g = vec![Gq::zero(); top];
        for k in (0..d).rev() {
            let mut rhs = f[k].clone();
            for (m, cm) in c.iter().enumerate().skip(s + 1) {
                let j = k + m;
                if j >= top {
                    continue;
                }
                rhs = rhs.sub(&cm.mul(&g[j]).scale(&falling(j, m)));
            }
            g[k + s] = rhs.div(&c[s].scale(&falling(k + s, s)));
        }
        let poly = |coef: &dyn Fn(&Gq) -> Rational| -> ExprId {
            let terms: Vec<ExprId> = g
                .iter()
                .enumerate()
                .filter(|(_, v)| coef(v) != 0)
                .map(|(j, v)| {
                    pool.mul(vec![
                        rat(&coef(v), pool),
                        pool.pow(x, pool.integer(j as i64)),
                    ])
                })
                .collect();
            pool.add(terms)
        };
        let ex = if fam.a == 0 {
            pool.integer(1_i32)
        } else {
            pool.func("exp", vec![pool.mul(vec![rat(&fam.a, pool), x])])
        };
        if fam.b == 0 {
            parts.push(pool.mul(vec![ex, poly(&|v: &Gq| v.re.clone())]));
        } else {
            let bx = pool.mul(vec![rat(&fam.b, pool), x]);
            let re = poly(&|v: &Gq| v.re.clone());
            let im = poly(&|v: &Gq| v.im.clone());
            parts.push(pool.mul(vec![
                ex,
                pool.add(vec![
                    pool.mul(vec![re, pool.func("cos", vec![bx])]),
                    neg(pool.mul(vec![im, pool.func("sin", vec![bx])]), pool),
                ]),
            ]));
        }
    }
    Some(pool.add(parts))
}

/// `j!/(j − m)!`.
fn falling(j: usize, m: usize) -> Rational {
    let mut v = Integer::from(1);
    for t in (j + 1 - m)..=j {
        v *= t as u64;
    }
    Rational::from(v)
}

// ---------------------------------------------------------------------------
// Polynomial solutions of linear equations with polynomial coefficients
// ---------------------------------------------------------------------------

/// Largest degree searched by [`polynomial_solutions`].
const MAX_POLY_SOLUTION_DEGREE: usize = 12;

/// A basis of the polynomial solutions (degree ≤ [`MAX_POLY_SOLUTION_DEGREE`])
/// of `Σ coeffs[k]·y^(k) = 0`, every coefficient a rational function of `x`
/// over ℚ. The equation is multiplied through by the product of the
/// denominators, the coefficients of `L[Σ cⱼ xʲ]` are set to zero, and the
/// null space is computed exactly over ℚ. This is how the Legendre, Hermite,
/// Laguerre and Chebyshev equations get their polynomial solution at integer
/// parameter. Empty when there is none (or the coefficients are not rational
/// functions).
pub(crate) fn polynomial_solutions(coeffs: &[ExprId], x: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    use crate::calculus::asymptotic_common::{as_rational_function, qp_mul, qp_trim};
    let mut fns = Vec::with_capacity(coeffs.len());
    for &c in coeffs {
        let Some(f) = as_rational_function(simp(c, pool), x, pool) else {
            return Vec::new();
        };
        fns.push(f);
    }
    // a_k = num_k · Π_{j≠k} den_j
    let polys: Vec<Vec<Rational>> = (0..fns.len())
        .map(|k| {
            let mut acc = fns[k].num.clone();
            for (j, f) in fns.iter().enumerate() {
                if j != k {
                    acc = qp_mul(&acc, &f.den);
                }
            }
            qp_trim(acc)
        })
        .collect();
    let dmax = MAX_POLY_SOLUTION_DEGREE;
    let maxdeg = polys.iter().map(|p| p.len()).max().unwrap_or(0) + dmax + 1;
    // Column j: coefficients of L[x^j].
    let mut cols: Vec<Vec<Rational>> = Vec::with_capacity(dmax + 1);
    for j in 0..=dmax {
        let mut col = vec![Rational::new(); maxdeg];
        for (k, a) in polys.iter().enumerate() {
            if k > j {
                continue;
            }
            let fall = falling(j, k);
            for (i, ai) in a.iter().enumerate() {
                col[i + j - k] += ai.clone() * fall.clone();
            }
        }
        cols.push(col);
    }
    // Null space of the maxdeg × (dmax+1) matrix by RREF.
    let ncols = dmax + 1;
    let mut m: Vec<Vec<Rational>> = (0..maxdeg)
        .map(|i| (0..ncols).map(|j| cols[j][i].clone()).collect())
        .collect();
    let mut pivots = Vec::new();
    let mut row = 0;
    for col in 0..ncols {
        let Some(pr) = (row..m.len()).find(|&r| m[r][col] != 0) else {
            continue;
        };
        m.swap(row, pr);
        let inv = Rational::from(1) / m[row][col].clone();
        for v in m[row].iter_mut() {
            *v *= inv.clone();
        }
        let pivot_row = m[row].clone();
        for (r, line) in m.iter_mut().enumerate() {
            if r != row && line[col] != 0 {
                let f = line[col].clone();
                for (v, pv) in line.iter_mut().zip(pivot_row.iter()) {
                    *v -= pv.clone() * f.clone();
                }
            }
        }
        pivots.push(col);
        row += 1;
        if row == m.len() {
            break;
        }
    }
    let mut out = Vec::new();
    for free in (0..ncols).filter(|c| !pivots.contains(c)) {
        let mut v = vec![Rational::new(); ncols];
        v[free] = Rational::from(1);
        for (r, &pc) in pivots.iter().enumerate() {
            v[pc] = -m[r][free].clone();
        }
        // Clear denominators for a tidy polynomial.
        let mut den = Integer::from(1);
        for c in &v {
            den = den.lcm(c.denom());
        }
        let terms: Vec<ExprId> = v
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != 0)
            .map(|(j, c)| {
                let cj = c.clone() * Rational::from(den.clone());
                pool.mul(vec![rat(&cj, pool), pool.pow(x, pool.integer(j as i64))])
            })
            .collect();
        out.push(simp(pool.add(terms), pool));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(e: ExprId, pool: &ExprPool) -> f64 {
        value(e, pool).unwrap()
    }

    #[test]
    fn cubic_x3_minus_1() {
        let pool = ExprPool::new();
        let r = exact_roots(&[Rational::from(-1), 0.into(), 0.into(), 1.into()], &pool).unwrap();
        assert_eq!(r.len(), 2);
        for root in r {
            match root {
                ExactRoot::Real { r, .. } => assert!((n(r, &pool) - 1.0).abs() < 1e-14),
                ExactRoot::Complex { alpha, beta, .. } => {
                    assert!((n(alpha, &pool) + 0.5).abs() < 1e-14);
                    assert!((n(beta, &pool) - 3f64.sqrt() / 2.0).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn quartic_x4_plus_1() {
        let pool = ExprPool::new();
        let r = exact_roots(&[1.into(), 0.into(), 0.into(), 0.into(), 1.into()], &pool).unwrap();
        assert_eq!(r.len(), 2);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        for root in r {
            let ExactRoot::Complex { alpha, beta, .. } = root else {
                panic!("expected complex pairs");
            };
            assert!((n(alpha, &pool).abs() - h).abs() < 1e-14);
            assert!((n(beta, &pool) - h).abs() < 1e-14);
        }
    }

    #[test]
    fn quartic_general_and_three_real_cubic() {
        let pool = ExprPool::new();
        // λ⁴ − λ − 1 (irreducible, Q ≠ 0), λ³ − 3λ + 1 (casus irreducibilis).
        for p in [
            vec![
                Rational::from(-1),
                (-1).into(),
                0.into(),
                0.into(),
                1.into(),
            ],
            vec![Rational::from(1), (-3).into(), 0.into(), 1.into()],
        ] {
            let roots = exact_roots(&p, &pool).unwrap();
            let eval = |re: f64, im: f64| {
                let (mut ar, mut ai) = (0.0, 0.0);
                for c in p.iter().rev() {
                    let (nr, ni) = (ar * re - ai * im, ar * im + ai * re);
                    ar = nr + c.to_f64();
                    ai = ni;
                }
                (ar * ar + ai * ai).sqrt()
            };
            let mut count = 0;
            for root in roots {
                match root {
                    ExactRoot::Real { r, .. } => {
                        assert!(eval(n(r, &pool), 0.0) < 1e-10);
                        count += 1;
                    }
                    ExactRoot::Complex { alpha, beta, .. } => {
                        assert!(eval(n(alpha, &pool), n(beta, &pool)) < 1e-10);
                        count += 2;
                    }
                }
            }
            assert_eq!(count, p.len() - 1);
        }
    }

    #[test]
    fn exact_particular_resonant_trig() {
        // y'' + y = cos x  →  y_p = x sin(x)/2
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let r = pool.func("cos", vec![x]);
        let fams = forcing_families(r, x, &pool).unwrap();
        let yp = particular_exact(&[1.into(), 0.into(), 1.into()], &fams, x, &pool).unwrap();
        let yp = simp(yp, &pool);
        let want = simp(
            pool.mul(vec![pool.rational(1, 2), x, pool.func("sin", vec![x])]),
            &pool,
        );
        assert_eq!(yp, want, "{}", pool.display(yp));
    }
}
