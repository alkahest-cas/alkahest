//! Autonomous equations `y'' = f(y)` whose energy integral is elliptic, and
//! Liouville's `y'' = a·e^{b·y}`.
//!
//! Multiplying by `y'` gives `y'² = 2∫f dy + C₁`, so `x + C₂ = ∫ dy/√(2F(y) + C₁)`.
//! For `f` a quadratic or cubic polynomial, or `sin y`, that quadrature is an
//! elliptic integral; the integrator rightly declines it, and the `p(y) = y'`
//! route in [`super::special_linear`] ends without an explicit answer.  The
//! *inversion* of the quadrature is classical, though, and is what this module
//! returns:
//!
//! | `f(y)` | general solution |
//! |---|---|
//! | `a y² + b y + c` | `y = (6/a)·℘(x + C₂; (b² − 4ac)/12, C₁) − b/(2a)` |
//! | `a w³ + p w` (`w = y + b/3a`, no constant term after the shift), `p ≠ 0` | `w = A·cn(λ(x + C₂) \| m)`, `λ² = p/(2m − 1)`, `A² = −2mp/(a(2m − 1))` |
//! | `a w³` | `w = C₁·nc(√a·C₁(x + C₂) \| ½)` (`a > 0`), `C₁·cn(√(−a)·C₁(x + C₂) \| ½)` (`a < 0`) |
//! | `−k sin y` | `y = 2·am(√(k/C₁)·(x + C₂) \| C₁)` (`k < 0`: shifted by `π`) |
//! | `k y⁻³` (Ermakov–Pinney) | `y = √(C₁(x + C₂)² + k/C₁)` |
//! | `a e^{b y}` | `y = (1/b)·log(C₁²/(2ab·cos²(C₁(x + C₂)/2)))` (`ab < 0`: `cosh`) |
//!
//! In the `cn` row the parameter `m` is a function of `C₁` chosen per sign
//! pattern of `(a, p)` so that the family is real for the constants the gate
//! samples (`m = C₁` for `a < 0 < p`, `m = −C₁` for `p < 0 < a`,
//! `m = C₁/(2C₁ + 1) ∈ (0, ½)` for `a, p < 0`); `a, p > 0` has no real `cn`
//! family and is declined.
//!
//! Each row is derived from the Jacobi/Weierstrass differential identities
//! (`cn'' = (2m − 1)cn − 2m·cn³`, `℘'' = 6℘² − g₂/2`, `am' = dn`,
//! `sin 2am = 2 sn cn`) — and **none of that derivation is trusted**: every
//! candidate goes through the ordinary substitution gate,
//! [`residual_is_zero`], exactly like the output of any other class.
//!
//! Recognition requires the equation to be `A·y'' + B(y) = 0` with `A` a
//! constant and `B` free of `x` and `y'`.

use super::{
    contains, ddx, div, is_zero, residual_is_zero, simp, ConstGen, DsolveBranch, DsolveError,
    OdeInput,
};
use crate::kernel::eval_const::try_expr_f64;
use crate::kernel::{Domain, ExprId, ExprPool};

/// Try the closed forms above; `Ok(None)` when the equation is not of the
/// shape, or no candidate verifies.
pub(super) fn solve_autonomous_special(
    input: &OdeInput,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<Vec<DsolveBranch>>, DsolveError> {
    if input.order() != 2 || contains(input.equation, input.x, pool) {
        return Ok(None);
    }
    let (yp, ypp, y) = (input.derivs[0], input.derivs[1], input.y);
    if contains(input.equation, yp, pool) || !contains(input.equation, y, pool) {
        return Ok(None);
    }
    let a_coef = simp(ddx(input.equation, ypp, pool)?, pool);
    if contains(a_coef, y, pool) || contains(a_coef, ypp, pool) || is_zero(a_coef, pool) {
        return Ok(None);
    }
    let rest = simp(
        pool.add(vec![
            input.equation,
            pool.mul(vec![pool.integer(-1_i32), a_coef, ypp]),
        ]),
        pool,
    );
    if contains(rest, ypp, pool) {
        return Ok(None);
    }
    // y'' = f(y)
    let f = div(pool.mul(vec![pool.integer(-1_i32), rest]), a_coef, pool);

    let mark = gen.checkpoint();
    let c1 = gen.fresh(pool);
    let c2 = gen.fresh(pool);
    let constants = vec![c1, c2];
    let t = pool.add(vec![input.x, c2]);

    let candidates: Vec<(ExprId, &'static str)> = if let Some((a, b)) = exponential(f, y, pool) {
        liouville(a, b, c1, t, pool)
    } else if let Some(k) = sine(f, y, pool) {
        pendulum(k, c1, t, pool)
    } else if let Some(k) = inverse_cube(f, y, pool) {
        // Ermakov–Pinney `y'' = k/y³`: y² = C₁(x + C₂)² + k/C₁ (elementary).
        let y2 = pool.add(vec![
            pool.mul(vec![c1, pool.pow(t, pool.integer(2_i32))]),
            pool.mul(vec![k, pool.pow(c1, pool.integer(-1_i32))]),
        ]);
        vec![(sqrt(y2, pool), "autonomous_ermakov")]
    } else if let Some(cs) = poly_coeffs(f, y, pool) {
        match cs.len() {
            3 => weierstrass(&cs, c1, t, pool),
            4 => duffing(&cs, c1, t, pool),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };
    for (cand, method) in candidates {
        let cand = simp(cand, pool);
        if residual_is_zero(input, cand, &constants, pool).is_ok() {
            return Ok(Some(vec![DsolveBranch::explicit(
                cand,
                constants.clone(),
                method,
            )]));
        }
        #[cfg(test)]
        if std::env::var_os("ALKAHEST_DSOLVE_TRACE").is_some() {
            eprintln!(
                "AUTONOMOUS candidate refused: {} ({:?})",
                pool.display(cand),
                residual_is_zero(input, cand, &constants, pool).err()
            );
        }
    }
    gen.rollback(mark, pool);
    Ok(None)
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

fn sqrt(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(e, pool.rational(1, 2))
}

fn sign_of(e: ExprId, pool: &ExprPool) -> Option<f64> {
    try_expr_f64(e, pool).filter(|v| v.is_finite() && *v != 0.0)
}

/// `f = a·e^{b·y}`: `f'/f` a non-zero constant `b`, then `a = f·e^{−b·y}`.
fn exponential(f: ExprId, y: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let df = ddx(f, y, pool).ok()?;
    let b = div(df, f, pool);
    if contains(b, y, pool) || is_zero(b, pool) {
        return None;
    }
    let a = simp(
        pool.mul(vec![
            f,
            pool.func("exp", vec![neg(pool.mul(vec![b, y]), pool)]),
        ]),
        pool,
    );
    if contains(a, y, pool) || is_zero(a, pool) {
        return None;
    }
    Some((a, b))
}

/// `f = −k·sin y`: returns `k`.
fn sine(f: ExprId, y: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let k = div(neg(f, pool), pool.func("sin", vec![y]), pool);
    if contains(k, y, pool) || is_zero(k, pool) {
        return None;
    }
    Some(k)
}

/// `f = k·y⁻³`: returns `k`.
fn inverse_cube(f: ExprId, y: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let k = simp(pool.mul(vec![f, pool.pow(y, pool.integer(3_i32))]), pool);
    if contains(k, y, pool) || is_zero(k, pool) {
        return None;
    }
    Some(k)
}

/// Coefficients `[c₀, c₁, …, c_d]` of `f` as a polynomial in `y` of degree
/// `d ∈ {2, 3}` with `y`-free coefficients, by Taylor coefficients at `y = 0`
/// and an identically vanishing `(d + 1)`-th derivative.
fn poly_coeffs(f: ExprId, y: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
    let zero = pool.integer(0_i32);
    let mut derivs = vec![f];
    for _ in 0..4 {
        let d = simp(ddx(*derivs.last()?, y, pool).ok()?, pool);
        if is_zero(d, pool) {
            break;
        }
        derivs.push(d);
    }
    let deg = derivs.len() - 1;
    if !(2..=3).contains(&deg) {
        return None;
    }
    // The top derivative must be a constant, or the "polynomial" is not one.
    if contains(derivs[deg], y, pool) {
        return None;
    }
    let mut fact = 1i64;
    let mut out = Vec::with_capacity(deg + 1);
    for (k, &d) in derivs.iter().enumerate() {
        if k > 0 {
            fact *= k as i64;
        }
        let at0 = super::subs1(d, y, zero, pool);
        if contains(at0, y, pool) {
            return None;
        }
        out.push(simp(pool.mul(vec![at0, pool.rational(1, fact)]), pool));
    }
    Some(out)
}

/// `y'' = a y² + b y + c`.
fn weierstrass(
    cs: &[ExprId],
    c1: ExprId,
    t: ExprId,
    pool: &ExprPool,
) -> Vec<(ExprId, &'static str)> {
    let (c, b, a) = (cs[0], cs[1], cs[2]);
    // g₂ = (b² − 4ac)/12
    let g2 = simp(
        pool.mul(vec![
            pool.rational(1, 12),
            pool.add(vec![
                pool.pow(b, pool.integer(2_i32)),
                pool.mul(vec![pool.integer(-4_i32), a, c]),
            ]),
        ]),
        pool,
    );
    let wp = pool.func("WeierstrassP", vec![t, g2, c1]);
    let y = pool.add(vec![
        pool.mul(vec![
            pool.integer(6_i32),
            pool.pow(a, pool.integer(-1_i32)),
            wp,
        ]),
        pool.mul(vec![
            pool.rational(-1, 2),
            b,
            pool.pow(a, pool.integer(-1_i32)),
        ]),
    ]);
    vec![(y, "autonomous_weierstrass")]
}

/// `y'' = a y³ + b y² + c y + d` with no constant term after `y = w − b/(3a)`.
fn duffing(cs: &[ExprId], c1: ExprId, t: ExprId, pool: &ExprPool) -> Vec<(ExprId, &'static str)> {
    let (d, c, b, a) = (cs[0], cs[1], cs[2], cs[3]);
    let inv_a = pool.pow(a, pool.integer(-1_i32));
    // shift s = b/(3a): y = w − s
    let s = simp(pool.mul(vec![pool.rational(1, 3), b, inv_a]), pool);
    // p = c − b²/(3a), q = d − bc/(3a) + 2b³/(27a²)
    let p = simp(
        pool.add(vec![
            c,
            pool.mul(vec![
                pool.rational(-1, 3),
                pool.pow(b, pool.integer(2_i32)),
                inv_a,
            ]),
        ]),
        pool,
    );
    let q = pool.add(vec![
        d,
        pool.mul(vec![pool.rational(-1, 3), b, c, inv_a]),
        pool.mul(vec![
            pool.rational(2, 27),
            pool.pow(b, pool.integer(3_i32)),
            pool.pow(a, pool.integer(-2_i32)),
        ]),
    ]);
    if !is_zero(q, pool) {
        return Vec::new();
    }
    let shift = |w: ExprId| pool.add(vec![w, neg(s, pool)]);
    let sa = sign_of(a, pool);
    if is_zero(p, pool) {
        // w'' = a w³
        let half = pool.rational(1, 2);
        let (k, recip) = match sa {
            Some(v) if v < 0.0 => (sqrt(neg(a, pool), pool), false),
            _ => (sqrt(a, pool), true),
        };
        let cn = pool.func("JacobiCN", vec![pool.mul(vec![k, c1, t]), half]);
        let w = if recip {
            pool.mul(vec![c1, pool.pow(cn, pool.integer(-1_i32))])
        } else {
            pool.mul(vec![c1, cn])
        };
        return vec![(shift(w), "autonomous_jacobi")];
    }
    // w'' = a w³ + p w:  w = A cn(λ t | m)
    let sp = sign_of(p, pool);
    let m = match (sa, sp) {
        (Some(va), Some(vp)) if va > 0.0 && vp > 0.0 => return Vec::new(),
        (Some(va), Some(vp)) if va > 0.0 && vp < 0.0 => neg(c1, pool),
        (Some(va), Some(vp)) if va < 0.0 && vp < 0.0 => pool.mul(vec![
            c1,
            pool.pow(
                pool.add(vec![
                    pool.mul(vec![pool.integer(2_i32), c1]),
                    pool.integer(1_i32),
                ]),
                pool.integer(-1_i32),
            ),
        ]),
        _ => c1,
    };
    let two_m_1 = pool.add(vec![
        pool.mul(vec![pool.integer(2_i32), m]),
        pool.integer(-1_i32),
    ]);
    let inv = pool.pow(two_m_1, pool.integer(-1_i32));
    let lam = sqrt(pool.mul(vec![p, inv]), pool);
    let amp = sqrt(pool.mul(vec![pool.integer(-2_i32), m, p, inv_a, inv]), pool);
    let w = pool.mul(vec![
        amp,
        pool.func("JacobiCN", vec![pool.mul(vec![lam, t]), m]),
    ]);
    vec![(shift(w), "autonomous_jacobi")]
}

/// `y'' = −k sin y`.
fn pendulum(k: ExprId, c1: ExprId, t: ExprId, pool: &ExprPool) -> Vec<(ExprId, &'static str)> {
    let negative = matches!(sign_of(k, pool), Some(v) if v < 0.0);
    let kk = if negative { neg(k, pool) } else { k };
    let lam = sqrt(pool.mul(vec![kk, pool.pow(c1, pool.integer(-1_i32))]), pool);
    let am = pool.func("JacobiAmplitude", vec![pool.mul(vec![lam, t]), c1]);
    let mut y = pool.mul(vec![pool.integer(2_i32), am]);
    if negative {
        y = pool.add(vec![y, pool.symbol("pi", Domain::Real)]);
    }
    vec![(y, "autonomous_pendulum")]
}

/// `y'' = a e^{b y}` (Liouville).
fn liouville(
    a: ExprId,
    b: ExprId,
    c1: ExprId,
    t: ExprId,
    pool: &ExprPool,
) -> Vec<(ExprId, &'static str)> {
    let ab = simp(pool.mul(vec![a, b]), pool);
    let theta = pool.mul(vec![pool.rational(1, 2), c1, t]);
    let c1sq = pool.pow(c1, pool.integer(2_i32));
    let build = |trig: &str, sign: i64| {
        let den = pool.mul(vec![
            pool.integer(2 * sign),
            ab,
            pool.pow(pool.func(trig, vec![theta]), pool.integer(2_i32)),
        ]);
        let arg = pool.mul(vec![c1sq, pool.pow(den, pool.integer(-1_i32))]);
        pool.mul(vec![
            pool.pow(b, pool.integer(-1_i32)),
            pool.func("log", vec![arg]),
        ])
    };
    match sign_of(ab, pool) {
        Some(v) if v < 0.0 => vec![(build("cosh", -1), "autonomous_liouville")],
        Some(_) => vec![(build("cos", 1), "autonomous_liouville")],
        None => vec![
            (build("cos", 1), "autonomous_liouville"),
            (build("cosh", -1), "autonomous_liouville"),
        ],
    }
}
