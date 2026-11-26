//! Constant-coefficient operators: factor the full symbol
//! `P(ξ, η) = Aξ² + Bξη + Cη² + Dξ + Eη + F` into two first-order factors
//! (over ℚ with FLINT when the coefficients are rational numbers, otherwise
//! over an algebraic / complex extension by the quadratic formula), solve
//! each factor by characteristics, and fall back to the exponential ansatz
//! `e^{kx + ω(k) y}` with the dispersion relation `P(k, ω) = 0` when the
//! symbol is irreducible.

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

use super::super::error::PdeError;
use super::first_order::{note_nonzero, Factor1};
use super::linear::Linear2;
use super::util::{d, div, exact_sqrt, is_zero, neg, simp, sqrt, sub, subs1};

/// `P = scale · (p₁ξ + q₁η + c₁)(p₂ξ + q₂η + c₂)`.
#[derive(Clone, Debug)]
pub(crate) struct Factored {
    pub scale: ExprId,
    pub f1: Factor1,
    pub f2: Factor1,
    /// The two factors are the same operator.
    pub repeated: bool,
    /// `"rational"` (FLINT over ℚ) or `"algebraic"` (quadratic formula).
    pub how: &'static str,
}

fn rational_value(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    })
}

/// Factor over ℚ with FLINT when all six coefficients are rational numbers.
fn factor_rational(l: &Linear2, pool: &ExprPool) -> Option<Factored> {
    let vals: Vec<rug::Rational> = l
        .coef
        .iter()
        .map(|&c| rational_value(simp(c, pool), pool))
        .collect::<Option<_>>()?;
    // Clear denominators.
    let mut lcm = rug::Integer::from(1);
    for v in &vals {
        lcm.lcm_mut(v.denom());
    }
    let ints: Vec<rug::Integer> = vals
        .iter()
        .map(|v| {
            (v.clone() * rug::Rational::from(lcm.clone()))
                .numer()
                .clone()
        })
        .collect();
    let xi = pool.symbol("__alkahest_pde2_xi", Domain::Real);
    let eta = pool.symbol("__alkahest_pde2_eta", Domain::Real);
    let monos = [
        pool.pow(xi, pool.integer(2)),
        pool.mul(vec![xi, eta]),
        pool.pow(eta, pool.integer(2)),
        xi,
        eta,
        pool.integer(1),
    ];
    let p = pool.add(
        ints.iter()
            .zip(monos)
            .map(|(c, m)| pool.mul(vec![pool.integer(c.clone()), m]))
            .collect(),
    );
    let mp = crate::poly::multipoly::MultiPoly::from_symbolic(p, vec![xi, eta], pool).ok()?;
    let fac = crate::poly::factor::factor_multivariate_z(&mp).ok()?;
    let mut linear: Vec<Factor1> = Vec::new();
    let mut scale = rug::Rational::from(fac.unit.clone()) / rug::Rational::from(lcm);
    for (f, m) in &fac.factors {
        let e = f.to_expr(pool);
        let pc = d(e, xi, pool).ok()?;
        let qc = d(e, eta, pool).ok()?;
        let cc = simp(
            subs1(
                subs1(e, xi, pool.integer(0), pool),
                eta,
                pool.integer(0),
                pool,
            ),
            pool,
        );
        let deg1 = rational_value(pc, pool).is_some() && rational_value(qc, pool).is_some();
        if !deg1 {
            return None;
        }
        if is_zero(pc, pool) && is_zero(qc, pool) {
            // a constant factor
            let v = rational_value(cc, pool)?;
            for _ in 0..*m {
                scale *= v.clone();
            }
            continue;
        }
        for _ in 0..*m {
            linear.push(Factor1 {
                p: pc,
                q: qc,
                c: cc,
            });
        }
    }
    if linear.len() != 2 {
        return None;
    }
    let repeated =
        linear[0].p == linear[1].p && linear[0].q == linear[1].q && linear[0].c == linear[1].c;
    Some(Factored {
        scale: pool.rational(scale.numer().clone(), scale.denom().clone()),
        f1: linear[0],
        f2: linear[1],
        repeated,
        how: "rational",
    })
}

/// Factor over an extension: the principal part by the quadratic formula,
/// then match the lower-order terms.
fn factor_algebraic(l: &Linear2, conds: &mut Vec<ExprId>, pool: &ExprPool) -> Option<Factored> {
    let [a, b, c, dd, e, f] = l.coef;
    let zero = pool.integer(0);
    let one = pool.integer(1);
    let disc = simp(
        sub(
            pool.pow(b, pool.integer(2)),
            pool.mul(vec![pool.integer(4), a, c]),
            pool,
        ),
        pool,
    );
    let mut local = Vec::new();
    // principal factors (p₁ξ + q₁η)(p₂ξ + q₂η) · scale
    let (scale, p1, q1, p2, q2) = if !is_zero(a, pool) {
        note_nonzero(a, &mut local, pool);
        let sq = sqrt(disc, pool);
        let two_a = pool.mul(vec![pool.integer(2), a]);
        let m1 = div(pool.add(vec![neg(b, pool), sq]), two_a, pool);
        let m2 = div(sub(neg(b, pool), sq, pool), two_a, pool);
        (
            a,
            one,
            simp(neg(m1, pool), pool),
            one,
            simp(neg(m2, pool), pool),
        )
    } else if !is_zero(c, pool) {
        note_nonzero(c, &mut local, pool);
        let sq = sqrt(disc, pool);
        let two_c = pool.mul(vec![pool.integer(2), c]);
        let n1 = div(pool.add(vec![neg(b, pool), sq]), two_c, pool);
        let n2 = div(sub(neg(b, pool), sq, pool), two_c, pool);
        (
            c,
            simp(neg(n1, pool), pool),
            one,
            simp(neg(n2, pool), pool),
            one,
        )
    } else if !is_zero(b, pool) {
        (b, one, zero, zero, one)
    } else {
        return None;
    };
    let dp = div(dd, scale, pool);
    let ep = div(e, scale, pool);
    let fp = div(f, scale, pool);
    let det = simp(
        sub(pool.mul(vec![p1, q2]), pool.mul(vec![p2, q1]), pool),
        pool,
    );
    let (c1, c2, repeated) = if !is_zero(det, pool) {
        // p₁c₂ + p₂c₁ = D', q₁c₂ + q₂c₁ = E'
        let det2 = neg(det, pool);
        let c1 = div(
            sub(pool.mul(vec![dp, q1]), pool.mul(vec![p1, ep]), pool),
            det2,
            pool,
        );
        let c2 = div(
            sub(pool.mul(vec![p2, ep]), pool.mul(vec![q2, dp]), pool),
            det2,
            pool,
        );
        if !is_zero(sub(pool.mul(vec![c1, c2]), fp, pool), pool) {
            return None;
        }
        (c1, c2, false)
    } else {
        // proportional principal parts: (p ξ + q η)² with p = p₁, q = q₁
        let (p, q) = (p1, q1);
        let sigma = if !is_zero(p, pool) {
            div(dp, p, pool)
        } else {
            div(ep, q, pool)
        };
        if !is_zero(sub(dp, pool.mul(vec![sigma, p]), pool), pool)
            || !is_zero(sub(ep, pool.mul(vec![sigma, q]), pool), pool)
        {
            return None;
        }
        let rad = simp(
            sub(
                pool.pow(sigma, pool.integer(2)),
                pool.mul(vec![pool.integer(4), fp]),
                pool,
            ),
            pool,
        );
        let half = pool.rational(1, 2);
        if is_zero(rad, pool) {
            let c0 = simp(pool.mul(vec![half, sigma]), pool);
            (c0, c0, true)
        } else {
            let sq = exact_sqrt(rad, pool).unwrap_or_else(|| sqrt(rad, pool));
            let c1 = simp(pool.mul(vec![half, pool.add(vec![sigma, sq])]), pool);
            let c2 = simp(pool.mul(vec![half, sub(sigma, sq, pool)]), pool);
            (c1, c2, false)
        }
    };
    conds.extend(local);
    Some(Factored {
        scale,
        f1: Factor1 {
            p: p1,
            q: q1,
            c: c1,
        },
        f2: Factor1 {
            p: p2,
            q: q2,
            c: c2,
        },
        repeated,
        how: "algebraic",
    })
}

/// Factor the full symbol of a constant-coefficient operator.
pub(crate) fn factor_symbol(
    l: &Linear2,
    conds: &mut Vec<ExprId>,
    pool: &ExprPool,
) -> Option<Factored> {
    factor_rational(l, pool).or_else(|| factor_algebraic(l, conds, pool))
}

/// What the constant-coefficient route produced.
pub(crate) struct ConstantSolution {
    pub solution: ExprId,
    pub method: &'static str,
    pub general: bool,
    pub parameters: Vec<ExprId>,
    pub steps: Vec<String>,
}

/// The general solution `scale·L₁L₂ u = G` from a factorisation.
pub(crate) fn solve_factored(
    l: &Linear2,
    fac: &Factored,
    fresh: &mut dyn FnMut(ExprId) -> ExprId,
    conds: &mut Vec<ExprId>,
    pool: &ExprPool,
) -> Result<Option<ConstantSolution>, PdeError> {
    let (x, y) = (l.x, l.y);
    let mut steps = vec![format!(
        "symbol factors ({}) as {} · ({}·ξ + {}·η + {}) · ({}·ξ + {}·η + {})",
        fac.how,
        pool.display(fac.scale),
        pool.display(fac.f1.p),
        pool.display(fac.f1.q),
        pool.display(fac.f1.c),
        pool.display(fac.f2.p),
        pool.display(fac.f2.q),
        pool.display(fac.f2.c),
    )];
    let homogeneous = if fac.repeated {
        let (inv, mult) = fac.f1.kernel(x, y, conds, pool);
        let v = fac.f1.repeat_var(x, y, pool);
        let f1 = fresh(inv);
        let f2 = fresh(inv);
        steps.push("repeated factor: u = m·(F₁(φ) + v·F₂(φ))".into());
        simp(
            pool.mul(vec![mult, pool.add(vec![f1, pool.mul(vec![v, f2])])]),
            pool,
        )
    } else {
        let (inv1, m1) = fac.f1.kernel(x, y, conds, pool);
        let (inv2, m2) = fac.f2.kernel(x, y, conds, pool);
        let f1 = fresh(inv1);
        let f2 = fresh(inv2);
        steps.push("distinct factors: u = m₁·F₁(φ₁) + m₂·F₂(φ₂)".into());
        simp(
            pool.add(vec![pool.mul(vec![m1, f1]), pool.mul(vec![m2, f2])]),
            pool,
        )
    };
    let particular = if l.homogeneous(pool) {
        pool.integer(0)
    } else {
        let g = div(l.rhs, fac.scale, pool);
        let Some(w1) = fac.f1.particular(g, x, y, pool) else {
            return Ok(None);
        };
        let Some(up) = fac.f2.particular(w1, x, y, pool) else {
            return Ok(None);
        };
        steps.push(format!(
            "particular solution by characteristics: {}",
            pool.display(up)
        ));
        up
    };
    Ok(Some(ConstantSolution {
        solution: simp(pool.add(vec![homogeneous, particular]), pool),
        method: "factored_symbol",
        general: true,
        parameters: Vec::new(),
        steps,
    }))
}

/// The exponential ansatz `u = e^{k x + ω y}` with `P(k, ω) = 0` solved for
/// the variable it is linear in (or by the quadratic formula): a
/// one-parameter family of solutions of a homogeneous equation.
pub(crate) fn dispersion_family(
    l: &Linear2,
    k: ExprId,
    pool: &ExprPool,
) -> Option<ConstantSolution> {
    if !l.homogeneous(pool) {
        return None;
    }
    let [a, b, c, dd, e, f] = l.coef;
    // P(k, ω) = C ω² + (B k + E) ω + (A k² + D k + F)
    let lin = simp(pool.add(vec![pool.mul(vec![b, k]), e]), pool);
    let cst = simp(
        pool.add(vec![
            pool.mul(vec![a, pool.pow(k, pool.integer(2))]),
            pool.mul(vec![dd, k]),
            f,
        ]),
        pool,
    );
    let (omega, swapped) = if is_zero(c, pool) && !is_zero(lin, pool) {
        (div(neg(cst, pool), lin, pool), false)
    } else if !is_zero(c, pool) {
        // Prefer a variable in which P is linear: try k ↔ ω by symmetry.
        let lin2 = simp(pool.add(vec![pool.mul(vec![b, k]), dd]), pool);
        if is_zero(a, pool) && !is_zero(lin2, pool) {
            let cst2 = simp(
                pool.add(vec![
                    pool.mul(vec![c, pool.pow(k, pool.integer(2))]),
                    pool.mul(vec![e, k]),
                    f,
                ]),
                pool,
            );
            (div(neg(cst2, pool), lin2, pool), true)
        } else {
            let disc = simp(
                sub(
                    pool.pow(lin, pool.integer(2)),
                    pool.mul(vec![pool.integer(4), c, cst]),
                    pool,
                ),
                pool,
            );
            let sq = sqrt(disc, pool);
            (
                div(
                    pool.add(vec![neg(lin, pool), sq]),
                    pool.mul(vec![pool.integer(2), c]),
                    pool,
                ),
                false,
            )
        }
    } else {
        return None;
    };
    let (x, y) = (l.x, l.y);
    let expo = if swapped {
        pool.add(vec![pool.mul(vec![omega, x]), pool.mul(vec![k, y])])
    } else {
        pool.add(vec![pool.mul(vec![k, x]), pool.mul(vec![omega, y])])
    };
    let sol = pool.func("exp", vec![simp(expo, pool)]);
    Some(ConstantSolution {
        solution: sol,
        method: "dispersion_relation",
        general: false,
        parameters: vec![k],
        steps: vec![format!(
            "exponential ansatz e^(k x + ω y): dispersion relation gives ω = {}",
            pool.display(omega)
        )],
    })
}

/// `u = e^{αx + βy} v` with `(α, β)` removing both first-order terms of a
/// non-parabolic constant-coefficient operator: returns `(α, β, F')` where
/// `F' = P(α, β)` is the zeroth-order coefficient of the equation for `v`.
pub(crate) fn exponential_shift(l: &Linear2, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId)> {
    let [a, b, c, dd, e, f] = l.coef;
    // [2A B; B 2C] [α; β] = −[D; E]
    let det = simp(
        sub(
            pool.mul(vec![pool.integer(4), a, c]),
            pool.pow(b, pool.integer(2)),
            pool,
        ),
        pool,
    );
    if is_zero(det, pool) {
        return None;
    }
    let alpha = div(
        sub(
            pool.mul(vec![b, e]),
            pool.mul(vec![pool.integer(2), c, dd]),
            pool,
        ),
        det,
        pool,
    );
    let beta = div(
        sub(
            pool.mul(vec![b, dd]),
            pool.mul(vec![pool.integer(2), a, e]),
            pool,
        ),
        det,
        pool,
    );
    let fp = simp(
        pool.add(vec![
            pool.mul(vec![a, pool.pow(alpha, pool.integer(2))]),
            pool.mul(vec![b, alpha, beta]),
            pool.mul(vec![c, pool.pow(beta, pool.integer(2))]),
            pool.mul(vec![dd, alpha]),
            pool.mul(vec![e, beta]),
            f,
        ]),
        pool,
    );
    Some((alpha, beta, fp))
}
