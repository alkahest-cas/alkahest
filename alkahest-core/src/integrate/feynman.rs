//! Definite integrals of `log Q(x)` and `Q(x)^{−n}` for a quadratic `Q` with
//! symbolic coefficients — the Feynman-parameter integrals of one-loop
//! amplitudes, `∫₀¹ log(m² − x(1−x)p²) dx` and `∫₀¹ dx/(m² − x(1−x)p²)ⁿ`
//! (F11).
//!
//! # Method
//!
//! Complete the square: `Q = A((x − v)² + κ)` with the vertex
//! `v = −B/(2A)` and `κ = (4AC − B²)/(4A²)`, the discriminant scaled so that
//! for the Feynman integrand `κ = m²/p² − ¼`. With `u = x − v` and
//! `G = u·log Q − 2x`, the antiderivative of `log Q` is, where `Q > 0`,
//!
//! * `κ > 0`: `G + 2√κ·atan(u/√κ)` (here `A > 0`);
//! * `κ = 0`: `G`;
//! * `κ = −λ < 0`: `G + √λ·log R(u)`, `R = (u+√λ)/(u−√λ)` for `A > 0`
//!   (`|u| > √λ` where `Q > 0`) and `(√λ+u)/(√λ−u)` for `A < 0`;
//!
//! and `∫dx/Q` is `atan(u/√κ)/(A√κ)`, `−1/(Au)`, `−log R(u)/(2A√λ)`
//! respectively, raised to `Q^{−n}` by the reduction
//! `Jₙ = u/(2(n−1)Aκ·Q^{n−1}) + (2n−3)/(2(n−1)Aκ)·J_{n−1}` (and for `κ = 0`,
//! `Q^{−n} = A^{−n}u^{−2n}` directly). Every one of these is odd in `u`, so
//! on an interval symmetric about the vertex (`[0, 1]` for Feynman
//! parameters) the endpoint difference is written as twice one value.
//!
//! # Branches
//!
//! The route needs `Q > 0` at both endpoints and the sign of `A`
//! established — or `A = c·sᵏ` for one parameter `s`, when the result also
//! branches on `A > 0`, `A < 0` and `A = 0` (where `Q|_{s=0}` must be a
//! constant) — and decides where the vertex lies. With the vertex outside
//! the interval (or `A < 0`) `Q` stays positive whatever the sign of `κ`;
//! with it inside (the Feynman case, `v = ½`), `κ < 0` puts both roots of
//! `Q` in the interval. Then:
//!
//! * `log Q` integrates to the principal-branch value, `log|Q| + iπ` where
//!   `Q < 0`: the real part above plus `2πi√λ` (the length of `{Q < 0}` is
//!   `2√λ`). The Feynman `−i0` prescription (`m² → m² − i0`) gives the
//!   opposite sign of the imaginary part; take the conjugate for it.
//! * `Q^{−n}` diverges; the closed form is returned for `κ > 0` with that
//!   stated side condition.
//!
//! When the sign of `κ` is established one form is returned; otherwise the
//! result is `Piecewise((form, κ > 0), (form, κ = 0), default)`. For the
//! Feynman integrand with `m, p > 0` the three branches are exactly
//! `p² < 4m²`, `p² = 4m²` and `p² > 4m²`.
//!
//! Every returned value is checked against Gauss–Legendre quadrature at
//! sampled parameter values in each real branch; a disagreement is a
//! decline, never an answer.

use super::beta_family::power_factors;
use super::engine::{collect_gate_parameters, IntegrationError};
use super::prenorm::{poly_coeffs, sign_of};
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use crate::simplify::Sign;
use std::collections::HashMap;

type Outcome = Option<Result<DerivedExpr<ExprId>, IntegrationError>>;

/// Largest `n` in `Q^{−n}` handled by the reduction formula.
const MAX_POWER: i64 = 8;

fn simp(e: ExprId, pool: &ExprPool) -> ExprId {
    simplify(e, pool).value
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, neg(b, pool)])
}

fn inv(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(e, pool.integer(-1_i32))
}

fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![a, inv(b, pool)])
}

fn log(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("log", vec![e])
}

fn sqrt(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("sqrt", vec![e])
}

/// What is integrated.
#[derive(Clone, Copy)]
enum Kind {
    Log,
    /// `Q^{−n}`.
    Recip(i64),
}

/// Which closed form.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Branch {
    /// `κ > 0`.
    Atan,
    /// `κ = 0`.
    Double,
    /// `κ < 0`, `A > 0`, `Q > 0` on the interval (roots outside it).
    Log,
    /// `κ < 0`, `A < 0`, `Q > 0` on the interval (between the roots).
    LogConcave,
    /// `κ < 0`, `A > 0`, both roots inside the interval.
    Straddle,
}

struct Quad {
    a: ExprId,
    b: ExprId,
    c: ExprId,
    var: ExprId,
    /// The vertex `v = −B/(2A)`.
    v: ExprId,
    /// `κ = (4AC − B²)/(4A²)`.
    kappa: ExprId,
}

impl Quad {
    fn q_at(&self, x: ExprId, pool: &ExprPool) -> ExprId {
        pool.add(vec![
            pool.mul(vec![self.a, pool.pow(x, pool.integer(2_i32))]),
            pool.mul(vec![self.b, x]),
            self.c,
        ])
    }

    fn sqrt_kappa(&self, pool: &ExprPool) -> ExprId {
        sqrt(self.kappa, pool)
    }

    /// `√λ = √(−κ)`.
    fn sqrt_lambda(&self, pool: &ExprPool) -> ExprId {
        sqrt(simp(neg(self.kappa, pool), pool), pool)
    }

    /// `R(u)`, the positive ratio in the `κ < 0` forms.
    fn ratio(&self, branch: Branch, u: ExprId, pool: &ExprPool) -> ExprId {
        let sl = self.sqrt_lambda(pool);
        if branch != Branch::LogConcave {
            div(pool.add(vec![u, sl]), sub(u, sl, pool), pool)
        } else {
            div(pool.add(vec![sl, u]), sub(sl, u, pool), pool)
        }
    }

    /// The odd part of the antiderivative (everything but `G`) at `u`.
    fn odd_part(&self, kind: Kind, branch: Branch, u: ExprId, pool: &ExprPool) -> ExprId {
        match kind {
            Kind::Log => match branch {
                Branch::Double => pool.integer(0_i32),
                Branch::Atan => {
                    let sk = self.sqrt_kappa(pool);
                    pool.mul(vec![
                        pool.integer(2_i32),
                        sk,
                        pool.func("atan", vec![div(u, sk, pool)]),
                    ])
                }
                Branch::Log | Branch::LogConcave | Branch::Straddle => pool.mul(vec![
                    self.sqrt_lambda(pool),
                    log(self.ratio(branch, u, pool), pool),
                ]),
            },
            Kind::Recip(n) => self.recip(n, branch, u, pool),
        }
    }

    /// `∫ Q^{−n}` at `u` (odd in `u`).
    fn recip(&self, n: i64, branch: Branch, u: ExprId, pool: &ExprPool) -> ExprId {
        let a = self.a;
        if branch == Branch::Double {
            return pool.mul(vec![
                pool.pow(a, pool.integer(-n)),
                pool.pow(u, pool.integer(1 - 2 * n)),
                pool.rational(1, 1 - 2 * n),
            ]);
        }
        let mut j = match branch {
            Branch::Atan => {
                let sk = self.sqrt_kappa(pool);
                pool.mul(vec![
                    inv(pool.mul(vec![a, sk]), pool),
                    pool.func("atan", vec![div(u, sk, pool)]),
                ])
            }
            _ => {
                let sl = self.sqrt_lambda(pool);
                pool.mul(vec![
                    pool.rational(-1, 2),
                    inv(pool.mul(vec![a, sl]), pool),
                    log(self.ratio(branch, u, pool), pool),
                ])
            }
        };
        // Q = A(u² + κ)
        let q = pool.mul(vec![
            a,
            pool.add(vec![pool.pow(u, pool.integer(2_i32)), self.kappa]),
        ]);
        for k in 2..=n {
            let den = pool.mul(vec![pool.integer(2 * (k - 1)), a, self.kappa]);
            let first = div(
                u,
                pool.mul(vec![den, pool.pow(q, pool.integer(k - 1))]),
                pool,
            );
            let factor = div(pool.integer(2 * k - 3), den, pool);
            j = pool.add(vec![first, pool.mul(vec![factor, j])]);
        }
        j
    }

    /// `f(u_hi) − f(u_lo)` for an odd `f`, written as `2f(u_hi)` when
    /// `u_lo = −u_hi`, and with `f(u) = −f(−u)` for a negative `u` so that
    /// arguments read positive.
    fn odd_difference(
        &self,
        f: &dyn Fn(ExprId) -> ExprId,
        u_hi: ExprId,
        u_lo: ExprId,
        pool: &ExprPool,
    ) -> ExprId {
        let at = |u: ExprId| {
            if sign_of(u, pool) == Some(Sign::Negative) {
                neg(f(simp(neg(u, pool), pool)), pool)
            } else {
                f(u)
            }
        };
        if simp(pool.add(vec![u_hi, u_lo]), pool) == pool.integer(0_i32) {
            pool.mul(vec![pool.integer(2_i32), at(u_hi)])
        } else {
            sub(at(u_hi), at(u_lo), pool)
        }
    }

    fn definite(
        &self,
        kind: Kind,
        branch: Branch,
        lo: ExprId,
        hi: ExprId,
        pool: &ExprPool,
    ) -> ExprId {
        let u_hi = simp(sub(hi, self.v, pool), pool);
        let u_lo = simp(sub(lo, self.v, pool), pool);
        let odd = self.odd_difference(&|u| self.odd_part(kind, branch, u, pool), u_hi, u_lo, pool);
        let mut parts = vec![odd];
        if let Kind::Log = kind {
            // G = u·log Q − 2x
            let g = |x: ExprId, u: ExprId| {
                pool.add(vec![
                    pool.mul(vec![u, log(simp(self.q_at(x, pool), pool), pool)]),
                    pool.mul(vec![pool.integer(-2_i32), x]),
                ])
            };
            parts.push(sub(g(hi, u_hi), g(lo, u_lo), pool));
            if branch == Branch::Straddle {
                parts.push(pool.mul(vec![
                    pool.integer(2_i32),
                    pool.imaginary_unit(),
                    super::beta_family::pi(pool),
                    self.sqrt_lambda(pool),
                ]));
            }
        }
        simp(pool.add(parts), pool)
    }
}

/// When the sign of `A` is unknown: the parameter `s` with `A = c·sᵏ`
/// (so `A = 0 ⇔ s = 0`), if `A` has that shape.
fn monomial_parameter(a: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let mut params = Vec::new();
    collect_gate_parameters(a, var, pool, &mut params);
    let [s] = params.as_slice() else {
        return None;
    };
    let c = poly_coeffs(a, *s, pool)?;
    let zero = pool.integer(0_i32);
    let nonzero: Vec<&ExprId> = c.iter().filter(|&&k| k != zero).collect();
    let [k] = nonzero.as_slice() else {
        return None;
    };
    matches!(pool.get(**k), ExprData::Integer(_) | ExprData::Rational(_)).then_some(*s)
}

/// `∫_lo^hi c·log Q dx` or `∫_lo^hi c·Q^{−n} dx`; see the module docs.
pub(crate) fn try_definite(
    expr: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Outcome {
    if lo == pool.pos_infinity() || hi == pool.pos_infinity() {
        return None;
    }
    let h = simp(sub(hi, lo, pool), pool);
    if sign_of(h, pool) != Some(Sign::Positive) {
        return None;
    }
    let (coeff, powers) = power_factors(expr, var, pool)?;
    let [(base, e)] = powers.as_slice() else {
        return None;
    };
    let (qexpr, kind) = match pool.get(*base) {
        ExprData::Func { name, args } if name == "log" && args.len() == 1 => {
            if *e != pool.integer(1_i32) {
                return None;
            }
            (args[0], Kind::Log)
        }
        _ => match pool.get(*e) {
            ExprData::Integer(n) => {
                let n = -n.0.to_i64()?;
                if !(1..=MAX_POWER).contains(&n) {
                    return None;
                }
                (*base, Kind::Recip(n))
            }
            _ => return None,
        },
    };
    let c = poly_coeffs(qexpr, var, pool)?;
    if c.len() != 3 {
        return None;
    }
    let (cc, bb, aa) = (c[0], c[1], c[2]);
    let a_sign = sign_of(aa, pool);
    if a_sign == Some(Sign::Zero) {
        return None;
    }
    let two_a = pool.mul(vec![pool.integer(2_i32), aa]);
    let v = simp(neg(div(bb, two_a, pool), pool), pool);
    let kappa = simp(
        sub(
            div(cc, aa, pool),
            div(
                pool.pow(bb, pool.integer(2_i32)),
                pool.mul(vec![pool.integer(4_i32), pool.pow(aa, pool.integer(2_i32))]),
                pool,
            ),
            pool,
        ),
        pool,
    );
    // Q at the vertex, `Aκ = C − B²/(4A)`: the readable form of the condition.
    let q_vertex = simp(
        sub(
            cc,
            div(
                pool.pow(bb, pool.integer(2_i32)),
                pool.mul(vec![pool.integer(4_i32), aa]),
                pool,
            ),
            pool,
        ),
        pool,
    );
    let quad = Quad {
        a: aa,
        b: bb,
        c: cc,
        var,
        v,
        kappa,
    };
    for x in [lo, hi] {
        if sign_of(simp(quad.q_at(x, pool), pool), pool) != Some(Sign::Positive) {
            return None;
        }
    }
    // Is the vertex inside the interval (decides where `Q` can vanish when A > 0)?
    let after_lo = sign_of(simp(sub(v, lo, pool), pool), pool);
    let before_hi = sign_of(simp(sub(hi, v, pool), pool), pool);
    let inside = match (after_lo, before_hi) {
        (Some(Sign::Positive), Some(Sign::Positive)) => Some(true),
        (Some(Sign::Negative | Sign::Zero), _) | (_, Some(Sign::Negative | Sign::Zero)) => {
            Some(false)
        }
        _ => None,
    };
    let zero = pool.integer(0_i32);
    let def = |b: Branch| quad.definite(kind, b, lo, hi, pool);
    let divergent = || {
        Some(Err(IntegrationError::NotImplemented(format!(
            "divergent integral: {} vanishes inside [{}, {}]",
            pool.display(qexpr),
            pool.display(lo),
            pool.display(hi)
        ))))
    };
    let mut conds = Vec::new();
    let (value, branches): (ExprId, Vec<Branch>) = match a_sign {
        // Concave with Q > 0 at both ends: positive throughout, κ < 0.
        Some(Sign::Negative) => (def(Branch::LogConcave), vec![Branch::LogConcave]),
        Some(_) => {
            let inside = inside?;
            let neg_branch = if inside {
                Branch::Straddle
            } else {
                Branch::Log
            };
            match (kind, sign_of(q_vertex, pool)) {
                (_, Some(Sign::Positive)) => (def(Branch::Atan), vec![Branch::Atan]),
                (Kind::Recip(_), Some(_)) if inside => return divergent(),
                (_, Some(Sign::Zero)) => (def(Branch::Double), vec![Branch::Double]),
                (_, Some(_)) => (def(neg_branch), vec![neg_branch]),
                (Kind::Recip(_), None) if inside => {
                    conds.push(SideCondition::Positive(q_vertex));
                    (def(Branch::Atan), vec![Branch::Atan])
                }
                (_, None) => (
                    pool.piecewise(
                        vec![
                            (pool.pred_gt(q_vertex, zero), def(Branch::Atan)),
                            (pool.pred_eq(q_vertex, zero), def(Branch::Double)),
                        ],
                        def(neg_branch),
                    ),
                    vec![Branch::Atan, Branch::Double, neg_branch],
                ),
            }
        }
        None => {
            // Unknown sign of A = c·sᵏ: branch on it too. At A = 0, Q must be
            // the constant Q|_{s=0}.
            let s = monomial_parameter(aa, var, pool)?;
            let mut m = HashMap::new();
            m.insert(s, zero);
            let q0 = simp(crate::kernel::subs::subs(qexpr, &m, pool), pool);
            if !super::engine::is_free_of(q0, var, pool) {
                return None;
            }
            let degenerate = simp(
                pool.mul(vec![
                    h,
                    match kind {
                        Kind::Log => log(q0, pool),
                        Kind::Recip(n) => pool.pow(q0, pool.integer(-n)),
                    },
                ]),
                pool,
            );
            let inside = inside?;
            let a_zero = (pool.pred_eq(aa, zero), degenerate);
            let atan = (pool.pred_gt(kappa, zero), def(Branch::Atan));
            match kind {
                Kind::Recip(_) if inside => {
                    conds.push(SideCondition::Positive(q_vertex));
                    (
                        pool.piecewise(vec![a_zero, atan], def(Branch::LogConcave)),
                        vec![Branch::Atan, Branch::LogConcave],
                    )
                }
                _ => {
                    let up = if inside {
                        Branch::Straddle
                    } else {
                        Branch::Log
                    };
                    (
                        pool.piecewise(
                            vec![
                                a_zero,
                                atan,
                                (pool.pred_eq(kappa, zero), def(Branch::Double)),
                                (pool.pred_gt(aa, zero), def(up)),
                            ],
                            def(Branch::LogConcave),
                        ),
                        vec![Branch::Atan, Branch::Double, up, Branch::LogConcave],
                    )
                }
            }
        }
    };
    let mut factors = coeff;
    factors.push(value);
    let value = simp(pool.mul(factors), pool);
    if !numerically_consistent(expr, value, &quad, lo, hi, &branches, pool) {
        return None;
    }
    let mut log_ = DerivationLog::new();
    log_.push(RewriteStep::with_conditions(
        "quadratic_feynman_parameter_integral",
        expr,
        value,
        conds,
    ));
    Some(Ok(DerivedExpr::with_log(value, log_)))
}

// ---------------------------------------------------------------------------
// Numeric cross-check
// ---------------------------------------------------------------------------

/// Gauss–Legendre nodes and weights on `[−1, 1]` (10 points).
const GL_X: [f64; 5] = [
    0.148_874_338_981_631_2,
    0.433_395_394_129_247_2,
    0.679_409_568_299_024_4,
    0.865_063_366_688_984_5,
    0.973_906_528_517_171_7,
];
const GL_W: [f64; 5] = [
    0.295_524_224_714_752_9,
    0.269_266_719_309_996_4,
    0.219_086_362_515_982_04,
    0.149_451_349_150_580_6,
    0.066_671_344_308_688_14,
];

fn quadrature(f: &dyn Fn(f64) -> Option<f64>, a: f64, b: f64, panels: usize) -> Option<f64> {
    let w = (b - a) / panels as f64;
    let mut total = 0.0;
    for i in 0..panels {
        let mid = a + (i as f64 + 0.5) * w;
        let half = w / 2.0;
        for k in 0..5 {
            for sgn in [-1.0, 1.0] {
                total += GL_W[k] * half * f(mid + sgn * GL_X[k] * half)?;
            }
        }
    }
    Some(total)
}

fn samples_for(param: ExprId, pool: &ExprPool) -> &'static [f64] {
    match pool.get(param) {
        ExprData::Symbol {
            domain: Domain::Positive | Domain::NonNegative,
            ..
        } => &[0.6131, 1.3719, 2.8123, 0.3127],
        _ => &[0.6131, -1.7231, 1.3719, 2.8123],
    }
}

/// Check `value` against quadrature at up to 64 parameter points. A point is
/// used when the integrand is real and finite on the interval (every real
/// branch); the straddling branch's imaginary part is not sampled here.
/// `false` on any disagreement.
fn numerically_consistent(
    integrand: ExprId,
    value: ExprId,
    quad: &Quad,
    lo: ExprId,
    hi: ExprId,
    branches: &[Branch],
    pool: &ExprPool,
) -> bool {
    let var = quad.var;
    let mut params = Vec::new();
    collect_gate_parameters(integrand, var, pool, &mut params);
    collect_gate_parameters(value, var, pool, &mut params);
    collect_gate_parameters(lo, var, pool, &mut params);
    collect_gate_parameters(hi, var, pool, &mut params);
    if params.len() > 4 {
        return true; // too many to sweep; the formulas are identities
    }
    let real_only = !branches.contains(&Branch::Straddle);
    let radix = 4_usize;
    let total = radix.pow(params.len() as u32).min(64);
    let mut checked = 0;
    for idx in 0..total {
        let mut env: HashMap<ExprId, f64> = HashMap::new();
        let mut rem = idx;
        for &p in &params {
            env.insert(p, samples_for(p, pool)[rem % radix]);
            rem /= radix;
        }
        let ev = |e: ExprId, env: &HashMap<ExprId, f64>| crate::jit::eval_interp(e, env, pool);
        let (Some(a), Some(b)) = (ev(lo, &env), ev(hi, &env)) else {
            continue;
        };
        if !(a.is_finite() && b.is_finite() && b > a) {
            continue;
        }
        let f = |x: f64| {
            let mut e2 = env.clone();
            e2.insert(var, x);
            ev(integrand, &e2).filter(|v| v.is_finite())
        };
        let Some(want) = quadrature(&f, a, b, 64) else {
            continue; // integrand not real on the whole interval here
        };
        if !real_only {
            // Skip points where Q dips below zero (the complex branch).
            let dips = (0..=64).any(|i| {
                let x = a + (b - a) * i as f64 / 64.0;
                let mut e2 = env.clone();
                e2.insert(var, x);
                ev(quad.q_at(var, pool), &e2).is_none_or(|q| q <= 0.0)
            });
            if dips {
                continue;
            }
        }
        let Some(got) = ev(value, &env) else {
            continue;
        };
        if !got.is_finite() {
            continue;
        }
        if (got - want).abs() > 1e-7 * (1.0 + want.abs()) {
            return false;
        }
        checked += 1;
        if checked >= 6 {
            break;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feynman_q(p: &ExprPool, x: ExprId, m: ExprId, k: ExprId) -> ExprId {
        // m² − x(1−x)k²
        p.add(vec![
            p.pow(m, p.integer(2_i32)),
            neg(
                p.mul(vec![
                    x,
                    p.add(vec![p.integer(1_i32), neg(x, p)]),
                    p.pow(k, p.integer(2_i32)),
                ]),
                p,
            ),
        ])
    }

    #[test]
    fn feynman_log_is_piecewise_in_threshold() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let m = p.symbol("m", Domain::Positive);
        let k = p.symbol("p", Domain::Positive);
        let f = log(feynman_q(&p, x, m, k), &p);
        let r = try_definite(f, x, p.integer(0_i32), p.integer(1_i32), &p)
            .expect("route applies")
            .expect("value");
        assert!(matches!(p.get(r.value), ExprData::Piecewise { .. }));
        // Below threshold, m = 1, p = 1: log 1 − 2 + 2√3·atan(1/√3) = −2 + π/√3.
        let env: HashMap<ExprId, f64> = [(m, 1.0), (k, 1.0)].into_iter().collect();
        let got = crate::jit::eval_interp(r.value, &env, &p).unwrap();
        let want = -2.0 + std::f64::consts::PI / 3f64.sqrt();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    /// Report 10-9 Q2 verbatim: `p2` a plain real symbol of unknown sign.
    #[test]
    fn feynman_log_unknown_sign_momentum() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let m = p.symbol("m", Domain::Positive);
        let p2 = p.symbol("p2", Domain::Real);
        let q = p.add(vec![
            p.pow(m, p.integer(2_i32)),
            neg(
                p.mul(vec![x, p.add(vec![p.integer(1_i32), neg(x, &p)]), p2]),
                &p,
            ),
        ]);
        let f = log(q, &p);
        let r = try_definite(f, x, p.integer(0_i32), p.integer(1_i32), &p)
            .expect("route applies")
            .expect("value");
        for pv in [-3.0, 0.0, 2.0] {
            let env: HashMap<ExprId, f64> = [(m, 1.0), (p2, pv)].into_iter().collect();
            let got = crate::jit::eval_interp(r.value, &env, &p).unwrap();
            let want = quadrature(&|u| Some((1.0 - u * (1.0 - u) * pv).ln()), 0.0, 1.0, 64)
                .unwrap();
            assert!((got - want).abs() < 1e-10, "p2 = {pv}: {got} vs {want}");
        }
    }

    #[test]
    fn feynman_reciprocal_states_threshold_condition() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let m = p.symbol("m", Domain::Positive);
        let k = p.symbol("p", Domain::Positive);
        let f = p.pow(feynman_q(&p, x, m, k), p.integer(-2_i32));
        let r = try_definite(f, x, p.integer(0_i32), p.integer(1_i32), &p)
            .unwrap()
            .unwrap();
        let conds: Vec<_> = r
            .log
            .0
            .iter()
            .flat_map(|s| s.side_conditions.clone())
            .collect();
        assert_eq!(conds.len(), 1);
    }
}
