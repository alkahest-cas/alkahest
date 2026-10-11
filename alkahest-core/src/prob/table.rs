//! Closed-form expectations of the common functions of a continuous variate.
//!
//! The general route in [`super::expect`] pushes `f(x)·p(x)` through a change
//! of variables and hands the result to the integrator. That is the right
//! route for a payoff, and the wrong one for `E[X²]` under a `Gamma` with a
//! symbolic shape — `∫ z^{k+1} e^{-z} dz / Γ(k)` — or `E[log X]`, where the
//! integrator has no rule and the answer is one line in every reference. This
//! table is those lines. It recognises `f` term by term after expansion, and
//! answers only when **every** term is in the table; otherwise it steps aside
//! and the general route runs.
//!
//! | law | term | `E[term]` |
//! |---|---|---|
//! | any | `Xᵏ`, `k ∈ ℕ` | the raw-moment table |
//! | `Normal(μ, σ)` | `Xᵏ e^{aX}` | `e^{aμ + a²σ²/2} · E[Yᵏ]`, `Y ~ Normal(μ + aσ², σ)` (tilting) |
//! | `Normal(μ, σ)` | `e^{aX} cos(ωX + φ)`, `sin` | `e^{aμ + (a² - ω²)σ²/2} cos(ω(μ + aσ²) + φ)` |
//! | `Gamma(k, θ)`, `Exponential` | `X^s e^{aX} (log X)^{0,1}` | `θ^s Γ(k+s)/Γ(k) (1-aθ)^{-(k+s)}`, times `ψ(k+s) + log θ - log(1-aθ)` for the log |
//! | `Beta(α, β)` | `X^s (log X)^{0,1}` | `Γ(α+s)Γ(α+β)/(Γ(α)Γ(α+β+s))`, times `ψ(α+s) - ψ(α+β+s)` |
//! | `LogNormal(μ, σ)` | `X^s (log X)^{0,1}` | `e^{sμ + s²σ²/2}`, times `μ + sσ²` |
//!
//! Every entry has a hypothesis — `k + s > 0` for a `Gamma` power, `1 - aθ > 0`
//! for its tilt, `α + s > 0` for a `Beta` power. A hypothesis that is decidably
//! false is [`ProbError::Divergent`]: `E[1/X]` under `Gamma(1, θ)` is `+∞`, and
//! the closed form `1/(θ(k-1))` there is a division by zero, not a value. One
//! that cannot be decided is carried as a [`SideCondition`].
//!
//! Nothing here is trusted: the assembled value goes through the same numeric
//! gate as every other expectation, against quadrature of `f(x)·p(x)`.

use crate::deriv::{DerivationLog, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::{simplify, simplify_expanded};

use super::dists::{self, Sign};
use super::{div, sub, DistKind, Distribution, ProbError};

fn exp(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("exp", vec![arg])
}

fn log(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("log", vec![arg])
}

fn gamma_fn(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("gamma", vec![arg])
}

fn digamma(arg: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("digamma", vec![arg])
}

fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0)
}

fn literal_int(e: ExprId, pool: &ExprPool) -> Option<i64> {
    match pool.get(e) {
        ExprData::Integer(n) => n.0.to_i64(),
        _ => None,
    }
}

fn sum(parts: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    if parts.is_empty() {
        pool.integer(0)
    } else {
        simplify(pool.add(parts), pool).value
    }
}

/// `(slope, intercept)` when `e` is affine in `x` with both free of `x`.
fn affine(e: ExprId, x: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let e = simplify_expanded(e, pool).value;
    let terms: Vec<ExprId> = match pool.get(e) {
        ExprData::Add(args) => args.to_vec(),
        _ => vec![e],
    };
    let mut slope = Vec::new();
    let mut intercept = Vec::new();
    for t in terms {
        if !dists::contains(t, x, pool) {
            intercept.push(t);
            continue;
        }
        if t == x {
            slope.push(pool.integer(1));
            continue;
        }
        let ExprData::Mul(args) = pool.get(t) else {
            return None;
        };
        let mut seen = false;
        let mut rest = Vec::new();
        for a in args {
            if a == x && !seen {
                seen = true;
            } else if dists::contains(a, x, pool) {
                return None;
            } else {
                rest.push(a);
            }
        }
        if !seen {
            return None;
        }
        slope.push(if rest.is_empty() {
            pool.integer(1)
        } else {
            pool.mul(rest)
        });
    }
    Some((sum(slope, pool), sum(intercept, pool)))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Trig {
    Cos,
    Sin,
}

/// One recognised term: `const · X^s · e^{aX + c} · (log X)^{logs} · trig`.
struct Term {
    constant: ExprId,
    s: ExprId,
    a: ExprId,
    c: ExprId,
    logs: u32,
    trig: Option<(Trig, ExprId, ExprId)>,
}

fn recognise(term: ExprId, x: ExprId, pool: &ExprPool) -> Option<Term> {
    let factors: Vec<ExprId> = match pool.get(term) {
        ExprData::Mul(args) => args.to_vec(),
        _ => vec![term],
    };
    let mut constants = Vec::new();
    let mut powers = Vec::new();
    let mut slopes = Vec::new();
    let mut intercepts = Vec::new();
    let mut logs = 0u32;
    let mut trig = None;
    for f in factors {
        if !dists::contains(f, x, pool) {
            constants.push(f);
            continue;
        }
        if f == x {
            powers.push(pool.integer(1));
            continue;
        }
        match pool.get(f) {
            ExprData::Pow { base, exp: e } if base == x && !dists::contains(e, x, pool) => {
                powers.push(e)
            }
            ExprData::Pow { base, exp: e } => {
                // (log x)ᵏ is past the table; so is anything else raised.
                let _ = (base, e);
                return None;
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                let arg = args[0];
                match name.as_str() {
                    "sqrt" if arg == x => powers.push(pool.rational(1, 2)),
                    "log" if arg == x => logs += 1,
                    "exp" => {
                        let (a, c) = affine(arg, x, pool)?;
                        slopes.push(a);
                        intercepts.push(c);
                    }
                    "cos" | "sin" if trig.is_none() => {
                        let (w, phi) = affine(arg, x, pool)?;
                        let kind = if name == "cos" { Trig::Cos } else { Trig::Sin };
                        trig = Some((kind, w, phi));
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
    if logs > 1 {
        return None;
    }
    Some(Term {
        constant: if constants.is_empty() {
            pool.integer(1)
        } else {
            pool.mul(constants)
        },
        s: sum(powers, pool),
        a: sum(slopes, pool),
        c: sum(intercepts, pool),
        logs,
        trig,
    })
}

/// Record `gap > 0` or refuse when it is decidably false.
fn require(
    gap: ExprId,
    what: &str,
    conditions: &mut Vec<SideCondition>,
    pool: &ExprPool,
) -> Result<(), ProbError> {
    let gap = simplify(gap, pool).value;
    match dists::sign_of(gap, pool) {
        Sign::Positive => Ok(()),
        Sign::Negative | Sign::Zero => Err(ProbError::Divergent(format!(
            "{what}: the defining integral diverges ({} ≤ 0)",
            pool.display(gap)
        ))),
        Sign::Unknown => {
            let c = SideCondition::Positive(gap);
            if !conditions.contains(&c) {
                conditions.push(c);
            }
            Ok(())
        }
    }
}

/// `E[X^s e^{aX}(log X)^logs·trig]` for one recognised term, or `None` when
/// the term is outside the table for this law.
fn term_value(
    t: &Term,
    dist: &Distribution,
    conditions: &mut Vec<SideCondition>,
    pool: &ExprPool,
) -> Option<Result<ExprId, ProbError>> {
    let p = dist.params();
    let one = pool.integer(1);
    let two = pool.integer(2);
    let half = pool.rational(1, 2);
    let s_int = literal_int(t.s, pool);
    let plain = is_zero(t.a, pool) && t.logs == 0 && t.trig.is_none();
    // Polynomial terms: the raw-moment table, for every law.
    if plain {
        if let Some(k) = s_int.filter(|&k| (0..=i64::from(super::MAX_MOMENT_ORDER)).contains(&k)) {
            return Some(super::moments::raw_moment_unchecked(dist, k as u32, pool));
        }
    }
    match dist.kind() {
        DistKind::Normal => {
            let (mu, sigma) = (p[0], p[1]);
            let s2 = pool.pow(sigma, two);
            if t.logs > 0 {
                return None;
            }
            let k = s_int.filter(|&k| (0..=i64::from(super::MAX_MOMENT_ORDER)).contains(&k))?;
            let shifted = pool.add(vec![mu, pool.mul(vec![t.a, s2])]);
            match t.trig {
                None => {
                    // E[Xᵏ e^{aX}] = M(a) · E[Yᵏ], Y ~ Normal(μ + aσ², σ).
                    let tilt = exp(
                        pool.add(vec![
                            pool.mul(vec![t.a, mu]),
                            pool.mul(vec![half, pool.pow(t.a, two), s2]),
                        ]),
                        pool,
                    );
                    let y = Distribution {
                        kind: DistKind::Normal,
                        params: vec![shifted, sigma],
                    };
                    Some(
                        super::moments::raw_moment_unchecked(&y, k as u32, pool)
                            .map(|m| pool.mul(vec![tilt, m])),
                    )
                }
                Some((kind, w, phi)) => {
                    if k != 0 {
                        return None;
                    }
                    // E[e^{(a+iω)X}] = e^{aμ + (a²-ω²)σ²/2} e^{i(ω(μ + aσ²))}.
                    let modulus = exp(
                        pool.add(vec![
                            pool.mul(vec![t.a, mu]),
                            pool.mul(vec![
                                half,
                                sub(pool.pow(t.a, two), pool.pow(w, two), pool),
                                s2,
                            ]),
                        ]),
                        pool,
                    );
                    let phase = pool.add(vec![pool.mul(vec![w, shifted]), phi]);
                    let f = match kind {
                        Trig::Cos => "cos",
                        Trig::Sin => "sin",
                    };
                    Some(Ok(pool.mul(vec![modulus, pool.func(f, vec![phase])])))
                }
            }
        }
        DistKind::Gamma | DistKind::Exponential => {
            if t.trig.is_some() {
                return None;
            }
            let (k, theta) = if dist.kind() == DistKind::Gamma {
                (p[0], p[1])
            } else {
                (one, pool.pow(p[0], pool.integer(-1)))
            };
            let nu = simplify(pool.add(vec![k, t.s]), pool).value;
            if let Err(e) = require(
                nu,
                "E[X^s] under a Gamma law needs k + s > 0",
                conditions,
                pool,
            ) {
                return Some(Err(e));
            }
            let tilted = !is_zero(t.a, pool);
            let gap = sub(one, pool.mul(vec![t.a, theta]), pool);
            if tilted {
                if let Err(e) = require(
                    gap,
                    "E[e^{aX}·…] under a Gamma law needs 1 - aθ > 0",
                    conditions,
                    pool,
                ) {
                    return Some(Err(e));
                }
            }
            // θ^s Γ(k+s)/Γ(k), as a finite product when s is an integer.
            let ratio = match s_int {
                Some(m) if (0..=i64::from(super::MAX_MOMENT_ORDER)).contains(&m) => {
                    let mut f = vec![one];
                    f.extend((0..m).map(|i| pool.add(vec![k, pool.integer(i)])));
                    pool.mul(f)
                }
                Some(m) if (-i64::from(super::MAX_MOMENT_ORDER)..0).contains(&m) => {
                    let den: Vec<ExprId> =
                        (1..=-m).map(|i| sub(k, pool.integer(i), pool)).collect();
                    pool.pow(pool.mul(den), pool.integer(-1))
                }
                _ => div(gamma_fn(nu, pool), gamma_fn(k, pool), pool),
            };
            let mut value = vec![pool.pow(theta, t.s), ratio];
            if tilted {
                value.push(pool.pow(gap, pool.mul(vec![pool.integer(-1), nu])));
            }
            if t.logs == 1 {
                let mut l = vec![digamma(nu, pool), log(theta, pool)];
                if tilted {
                    l.push(pool.mul(vec![pool.integer(-1), log(gap, pool)]));
                }
                value.push(pool.add(l));
            }
            Some(Ok(pool.mul(value)))
        }
        DistKind::Beta => {
            if t.trig.is_some() || !is_zero(t.a, pool) {
                return None;
            }
            let (al, be) = (p[0], p[1]);
            let nu = simplify(pool.add(vec![al, t.s]), pool).value;
            if let Err(e) = require(
                nu,
                "E[X^s] under a Beta law needs α + s > 0",
                conditions,
                pool,
            ) {
                return Some(Err(e));
            }
            let total = pool.add(vec![al, be]);
            let top = pool.add(vec![total, t.s]);
            let ratio = if is_zero(t.s, pool) {
                one
            } else {
                pool.mul(vec![
                    gamma_fn(nu, pool),
                    gamma_fn(total, pool),
                    pool.pow(
                        pool.mul(vec![gamma_fn(al, pool), gamma_fn(top, pool)]),
                        pool.integer(-1),
                    ),
                ])
            };
            let mut value = vec![ratio];
            if t.logs == 1 {
                value.push(sub(digamma(nu, pool), digamma(top, pool), pool));
            }
            Some(Ok(pool.mul(value)))
        }
        DistKind::LogNormal => {
            if t.trig.is_some() || !is_zero(t.a, pool) {
                return None;
            }
            let (mu, sigma) = (p[0], p[1]);
            let s2 = pool.pow(sigma, two);
            let mut value = vec![exp(
                pool.add(vec![
                    pool.mul(vec![t.s, mu]),
                    pool.mul(vec![half, pool.pow(t.s, two), s2]),
                ]),
                pool,
            )];
            if t.logs == 1 {
                value.push(pool.add(vec![mu, pool.mul(vec![t.s, s2])]));
            }
            Some(Ok(pool.mul(value)))
        }
        _ => None,
    }
}

/// `E[f(X)]` from the table, or `None` when some term of `f` is not in it.
///
/// `Some(Err(..))` is a decided refusal — a hypothesis of the table entry is
/// decidably false and the expectation diverges.
pub(crate) fn closed_form(
    f: ExprId,
    x: ExprId,
    dist: &Distribution,
    pool: &ExprPool,
) -> Option<Result<(ExprId, DerivationLog), ProbError>> {
    if dist.support(pool).is_discrete() {
        return None;
    }
    let expanded = simplify_expanded(f, pool).value;
    let terms: Vec<ExprId> = match pool.get(expanded) {
        ExprData::Add(args) => args.to_vec(),
        _ => vec![expanded],
    };
    if terms.len() > 64 {
        return None;
    }
    let recognised: Vec<Term> = terms
        .iter()
        .map(|&t| recognise(t, x, pool))
        .collect::<Option<Vec<Term>>>()?;
    let mut conditions = Vec::new();
    let mut parts = Vec::with_capacity(recognised.len());
    for t in &recognised {
        let v = match term_value(t, dist, &mut conditions, pool)? {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let mut factors = vec![t.constant, v];
        if !is_zero(t.c, pool) {
            factors.push(exp(t.c, pool));
        }
        parts.push(pool.mul(factors));
    }
    let value = simplify(pool.add(parts), pool).value;
    let mut log = DerivationLog::new();
    log.push(RewriteStep::with_conditions(
        "prob_expectation_from_closed_form_table",
        f,
        value,
        conditions,
    ));
    Some(Ok((value, log)))
}
