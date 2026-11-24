//! `∫ R(x)·e^{g(x)} dx` in terms of `Ei`, `erf` and `erfi`, for `R` rational
//! and `g` a polynomial of degree one or two with rational coefficients.
//!
//! This is the quadrature reduction of order needs for the second solution of
//! Hermite's and Laguerre's equations (and of `y'' + x·y' + y = 0`): with the
//! polynomial first solution `y₁`, `y₂ = y₁·∫ e^{−∫p}/y₁² dx`, an integrand
//! `e^{x²}/H_n(x)²` or `eˣ/(x·L_n(x)²)`.  The integrator correctly reports it
//! non-elementary; its value is elementary **plus one special-function term**.
//!
//! # Method (Hermite reduction against `e^g`)
//!
//! Write `R = N/Q` and let `D = gcd(Q, Q′)`, `Q = D·Q*` (`Q*` square-free).
//! Look for a rational `S = P/D` with
//!
//! `R = S′ + g′·S + rem`,
//!
//! where `rem` is what no `S` can absorb: a constant `c` when `deg g = 2`
//! (`∫c·e^g` is an `erf`/`erfi`), and `Σⱼ cⱼ/(x − rⱼ)` over the rational roots
//! `rⱼ` of `Q*` when `deg g = 1` (`∫e^{ax+b}/(x − r) = e^{ar+b}·Ei(a(x − r))`).
//! Multiplying through by `D·Q` makes this a linear system over ℚ in the
//! coefficients of `P` and the `cⱼ`; a solution gives
//!
//! `∫ R e^g dx = S·e^g + (special term)`.
//!
//! An inconsistent system — a pole the special basis cannot carry, a quadratic
//! `g` with a pole at all — is a decline, never a guess.  The answer is checked
//! here by differentiation where the evaluator allows it, and in any case the
//! caller's ODE gate verifies the solution it is assembled into.
//!
//! `erfi` is the real primitive `erfi(z) = −i·erf(i·z)` (real on ℝ, with
//! `erfi′(z) = 2/√π·e^{z²}`), so the answers carry no imaginary unit.

use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::{Integer, Rational};

type P = Vec<Rational>;

fn trim(mut p: P) -> P {
    while p.last().is_some_and(|c| *c == 0) {
        p.pop();
    }
    p
}

fn deg(p: &P) -> isize {
    p.len() as isize - 1
}

fn padd(a: &P, b: &P) -> P {
    let n = a.len().max(b.len());
    let mut out = vec![Rational::new(); n];
    for (i, c) in a.iter().enumerate() {
        out[i] += c;
    }
    for (i, c) in b.iter().enumerate() {
        out[i] += c;
    }
    trim(out)
}

fn pscale(a: &P, s: &Rational) -> P {
    trim(a.iter().map(|c| Rational::from(c * s)).collect())
}

fn psub(a: &P, b: &P) -> P {
    padd(a, &pscale(b, &Rational::from(-1)))
}

fn pmul(a: &P, b: &P) -> P {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Rational::new(); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += Rational::from(x * y);
        }
    }
    trim(out)
}

fn pderiv(a: &P) -> P {
    trim(
        a.iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| Rational::from(c * i as u32))
            .collect(),
    )
}

/// `(quotient, remainder)` of `a / b`, `b ≠ 0`.
fn pdivrem(a: &P, b: &P) -> (P, P) {
    let mut r = a.clone();
    let db = b.len() - 1;
    let lead = b[db].clone();
    if r.len() < b.len() {
        return (Vec::new(), r);
    }
    let mut q = vec![Rational::new(); r.len() - db];
    while r.len() > db && !r.is_empty() {
        let k = r.len() - 1 - db;
        let c = Rational::from(&r[r.len() - 1] / &lead);
        for (i, bc) in b.iter().enumerate() {
            r[k + i] -= Rational::from(&c * bc);
        }
        q[k] = c;
        r.pop();
        r = trim(r);
    }
    (trim(q), r)
}

fn pgcd(a: &P, b: &P) -> P {
    let (mut a, mut b) = (a.clone(), b.clone());
    while !b.is_empty() {
        let (_, r) = pdivrem(&a, &b);
        a = b;
        b = r;
    }
    if let Some(l) = a.last().cloned() {
        a = pscale(&a, &(1 / l));
    }
    a
}

fn peval(a: &P, x: &Rational) -> Rational {
    let mut acc = Rational::new();
    for c in a.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

fn monomial(k: usize) -> P {
    let mut v = vec![Rational::new(); k + 1];
    v[k] = Rational::from(1);
    v
}

/// `e` as a rational function `(num, den)` of `x` with rational coefficients.
fn ratfun(e: ExprId, x: ExprId, pool: &ExprPool) -> Option<(P, P)> {
    let one = vec![Rational::from(1)];
    match pool.get(e) {
        ExprData::Integer(n) => Some((trim(vec![Rational::from(&n.0)]), one)),
        ExprData::Rational(r) => Some((trim(vec![r.0.clone()]), one)),
        ExprData::Symbol { .. } if e == x => Some((monomial(1), one)),
        ExprData::Add(args) => {
            let (mut n, mut d) = (Vec::new(), one);
            for a in args {
                let (an, ad) = ratfun(a, x, pool)?;
                n = padd(&pmul(&n, &ad), &pmul(&an, &d));
                d = pmul(&d, &ad);
            }
            Some((n, d))
        }
        ExprData::Mul(args) => {
            let (mut n, mut d) = (one.clone(), one);
            for a in args {
                let (an, ad) = ratfun(a, x, pool)?;
                n = pmul(&n, &an);
                d = pmul(&d, &ad);
            }
            Some((n, d))
        }
        ExprData::Pow { base, exp } => {
            let k = match pool.get(exp) {
                ExprData::Integer(k) => k.0.to_i32()?,
                _ => return None,
            };
            if k.unsigned_abs() > 24 {
                return None;
            }
            let (bn, bd) = ratfun(base, x, pool)?;
            let (mut n, mut d) = (one.clone(), one);
            for _ in 0..k.unsigned_abs() {
                n = pmul(&n, &bn);
                d = pmul(&d, &bd);
            }
            if k < 0 {
                std::mem::swap(&mut n, &mut d);
            }
            if d.is_empty() {
                return None;
            }
            Some((n, d))
        }
        _ => None,
    }
}

fn poly_to_expr(p: &P, x: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    for (k, c) in p.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        let cexpr = rat_expr(c, pool);
        terms.push(match k {
            0 => cexpr,
            1 => pool.mul(vec![cexpr, x]),
            _ => pool.mul(vec![cexpr, pool.pow(x, pool.integer(k as i64))]),
        });
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

fn rat_expr(c: &Rational, pool: &ExprPool) -> ExprId {
    if *c.denom() == 1 {
        pool.integer(c.numer().clone())
    } else {
        pool.rational(c.numer().clone(), c.denom().clone())
    }
}

/// Split `f` into `(g, R)` with `f = R·e^{g}`: every `exp(…)` factor's
/// argument goes to `g`, except summands `k·log u`, which become `u^k` in `R`.
fn split_exp(f: ExprId, pool: &ExprPool) -> (Vec<ExprId>, Vec<ExprId>) {
    let factors = match pool.get(f) {
        ExprData::Mul(xs) => xs,
        _ => vec![f],
    };
    let mut g = Vec::new();
    let mut r = Vec::new();
    for fac in factors {
        // exp(a)^k, k an integer, as well as exp(a).
        let (base, k) = match pool.get(fac) {
            ExprData::Pow { base, exp } if matches!(pool.get(exp), ExprData::Integer(_)) => {
                (base, Some(exp))
            }
            _ => (fac, None),
        };
        match pool.get(base) {
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                let arg = match k {
                    Some(k) => pool.mul(vec![k, args[0]]),
                    None => args[0],
                };
                let arg = super::simp(arg, pool);
                let summands = match pool.get(arg) {
                    ExprData::Add(xs) => xs,
                    _ => vec![arg],
                };
                for s in summands {
                    match log_term(s, pool) {
                        Some((u, c)) => r.push(pool.pow(u, c)),
                        None => g.push(s),
                    }
                }
            }
            _ => r.push(fac),
        }
    }
    (g, r)
}

/// `c·log(u)` with `c` a numeric literal.
fn log_term(t: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let is_log = |e: ExprId| match pool.get(e) {
        ExprData::Func { name, args } if name == "log" && args.len() == 1 => Some(args[0]),
        _ => None,
    };
    if let Some(u) = is_log(t) {
        return Some((u, pool.integer(1_i32)));
    }
    if let ExprData::Mul(xs) = pool.get(t) {
        if xs.len() == 2 {
            for (a, b) in [(xs[0], xs[1]), (xs[1], xs[0])] {
                if matches!(pool.get(a), ExprData::Integer(_) | ExprData::Rational(_)) {
                    if let Some(u) = is_log(b) {
                        return Some((u, a));
                    }
                }
            }
        }
    }
    None
}

/// Rational roots of `p` (rational-root theorem on the cleared polynomial).
fn rational_roots(p: &P) -> Vec<Rational> {
    if p.len() < 2 {
        return Vec::new();
    }
    let mut lcm = Integer::from(1);
    for c in p {
        lcm.lcm_mut(c.denom());
    }
    let ints: Vec<Integer> = p
        .iter()
        .map(|c| c.numer() * Integer::from(&lcm / c.denom()))
        .collect();
    let mut out = Vec::new();
    // x = 0
    let mut lo = 0;
    while lo < ints.len() && ints[lo] == 0 {
        lo += 1;
    }
    if lo > 0 {
        out.push(Rational::new());
    }
    let a0 = ints[lo].clone().abs();
    let an = ints[ints.len() - 1].clone().abs();
    let divisors = |n: &Integer| -> Option<Vec<Integer>> {
        let n = n.to_u64()?;
        if n > 1_000_000 {
            return None;
        }
        Some((1..=n).filter(|d| n % d == 0).map(Integer::from).collect())
    };
    let (Some(ps), Some(qs)) = (divisors(&a0), divisors(&an)) else {
        return out;
    };
    for pn in &ps {
        for qd in &qs {
            for sgn in [1, -1] {
                let r = Rational::from((Integer::from(pn * sgn), qd.clone()));
                if !out.contains(&r) && peval(p, &r) == 0 {
                    out.push(r);
                }
            }
        }
    }
    out
}

/// Rational zeros of `e`, a rational function of `x` with rational
/// coefficients (roots of the numerator that are not poles), as expressions.
pub(crate) fn rational_zeros(e: ExprId, x: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let Some((n, d)) = ratfun(e, x, pool) else {
        return Vec::new();
    };
    rational_roots(&n)
        .into_iter()
        .filter(|r| peval(&d, r) != 0)
        .map(|r| rat_expr(&r, pool))
        .collect()
}

/// Solve `Σ uᵢ·colsᵢ = target` over ℚ (polynomial coefficient identity).
fn solve_linear(cols: &[P], target: &P) -> Option<Vec<Rational>> {
    let rows = cols
        .iter()
        .map(|c| c.len())
        .chain(std::iter::once(target.len()))
        .max()
        .unwrap_or(0);
    let n = cols.len();
    let mut m: Vec<Vec<Rational>> = (0..rows)
        .map(|r| {
            let mut row: Vec<Rational> = cols
                .iter()
                .map(|c| c.get(r).cloned().unwrap_or_default())
                .collect();
            row.push(target.get(r).cloned().unwrap_or_default());
            row
        })
        .collect();
    let mut pivots = Vec::new();
    let mut r = 0;
    for c in 0..n {
        let Some(pr) = (r..rows).find(|&i| m[i][c] != 0) else {
            continue;
        };
        m.swap(r, pr);
        let inv = 1 / m[r][c].clone();
        for v in m[r].iter_mut() {
            *v *= &inv;
        }
        for i in 0..rows {
            if i != r && m[i][c] != 0 {
                let f = m[i][c].clone();
                let pivot_row = m[r].clone();
                for (v, pr) in m[i].iter_mut().zip(&pivot_row) {
                    *v -= Rational::from(&f * pr);
                }
            }
        }
        pivots.push(c);
        r += 1;
    }
    // Consistency: no row 0 = non-zero.
    if m[r..].iter().any(|row| row[n] != 0) {
        return None;
    }
    let mut sol = vec![Rational::new(); n];
    for (i, &c) in pivots.iter().enumerate() {
        sol[c] = m[i][n].clone();
    }
    Some(sol)
}

/// `∫ f dx` for `f = R(x)·e^{g(x)}` as described in the module docs, or
/// `None`.
pub(crate) fn exp_rational_integral(f: ExprId, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let f = super::simp(f, pool);
    let (gs, rs) = split_exp(f, pool);
    if gs.is_empty() {
        return None;
    }
    let g_expr = super::simp(pool.add(gs), pool);
    let (gn, gd) = ratfun(g_expr, x, pool)?;
    if gd.len() != 1 {
        return None;
    }
    let g = pscale(&gn, &(1 / gd[0].clone()));
    let dg = deg(&g);
    if !(1..=2).contains(&dg) {
        return None;
    }
    let r_expr = if rs.is_empty() {
        pool.integer(1_i32)
    } else {
        super::simp(pool.mul(rs), pool)
    };
    let (n0, q0) = ratfun(r_expr, x, pool)?;
    if n0.is_empty() || q0.is_empty() {
        return None;
    }
    // Reduce N/Q.
    let h = pgcd(&n0, &q0);
    let (n, _) = pdivrem(&n0, &h);
    let (q, _) = pdivrem(&q0, &h);
    let d = pgcd(&q, &pderiv(&q));
    let (qstar, _) = pdivrem(&q, &d);
    let gp = pderiv(&g);

    // Degree of P: enough to match N·D against Q*·g′·P·D at the top.
    let dp = (deg(&n) + deg(&d) + 2).max(0) as usize;
    let mut cols: Vec<P> = Vec::new();
    for k in 0..=dp {
        let xk = monomial(k);
        // Q*·(P′D − P D′ + g′ P D) for P = x^k
        let inner = padd(
            &psub(&pmul(&pderiv(&xk), &d), &pmul(&xk, &pderiv(&d))),
            &pmul(&pmul(&gp, &xk), &d),
        );
        cols.push(pmul(&qstar, &inner));
    }
    let qd = pmul(&q, &d);
    let roots = if dg == 1 {
        rational_roots(&qstar)
    } else {
        Vec::new()
    };
    if dg == 1 {
        for r in &roots {
            let lin = vec![Rational::from(-r), Rational::from(1)];
            let (quo, rem) = pdivrem(&qd, &lin);
            if !rem.is_empty() {
                return None;
            }
            cols.push(quo);
        }
    } else {
        cols.push(qd.clone());
    }
    let target = pmul(&n, &d);
    let sol = solve_linear(&cols, &target)?;

    let pcoef: P = trim(sol[..=dp].to_vec());
    let rem = &sol[dp + 1..];
    let exp_g = pool.func("exp", vec![g_expr]);
    let mut terms = Vec::new();
    if !pcoef.is_empty() {
        let s = pool.mul(vec![
            poly_to_expr(&pcoef, x, pool),
            pool.pow(poly_to_expr(&d, x, pool), pool.integer(-1_i32)),
        ]);
        terms.push(pool.mul(vec![s, exp_g]));
    }
    if dg == 1 {
        let a = g[1].clone();
        let b = g.first().cloned().unwrap_or_default();
        for (r, c) in roots.iter().zip(rem) {
            if *c == 0 {
                continue;
            }
            // c·e^{a r + b}·Ei(a(x − r))
            let shift = Rational::from(&a * r) + &b;
            let arg = pool.mul(vec![
                rat_expr(&a, pool),
                pool.add(vec![x, rat_expr(&Rational::from(-r), pool)]),
            ]);
            terms.push(pool.mul(vec![
                rat_expr(c, pool),
                pool.func("exp", vec![rat_expr(&shift, pool)]),
                pool.func("Ei", vec![arg]),
            ]));
        }
    } else if let Some(c) = rem.first().filter(|c| **c != 0) {
        // ∫ e^{αx² + βx + γ} = e^{γ − β²/(4α)}·√π/(2√|α|)·{erfi, erf}(√|α|·(x + β/(2α)))
        let alpha = g[2].clone();
        let beta = g.get(1).cloned().unwrap_or_default();
        let gamma = g.first().cloned().unwrap_or_default();
        let shift = &beta / (Rational::from(2) * &alpha);
        let konst = gamma - Rational::from(&beta * &beta) / (Rational::from(4) * &alpha);
        let abs_alpha = alpha.clone().abs();
        let sqrt_a = pool.pow(rat_expr(&abs_alpha, pool), pool.rational(1, 2));
        let z = pool.mul(vec![sqrt_a, pool.add(vec![x, rat_expr(&shift, pool)])]);
        let pref = pool.mul(vec![
            rat_expr(c, pool),
            pool.func("exp", vec![rat_expr(&konst, pool)]),
            pool.pow(
                pool.symbol("pi", crate::kernel::Domain::Real),
                pool.rational(1, 2),
            ),
            pool.rational(1, 2),
            pool.pow(sqrt_a, pool.integer(-1_i32)),
        ]);
        let special = if alpha > 0 {
            pool.func("erfi", vec![z])
        } else {
            pool.func("erf", vec![z])
        };
        terms.push(pool.mul(vec![pref, special]));
    }
    if terms.is_empty() {
        return None;
    }
    Some(super::simp(pool.add(terms), pool))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn check(f: ExprId, x: ExprId, pool: &ExprPool, head: &str) {
        let big_f = exp_rational_integral(f, x, pool).expect("closes");
        let shown = pool.display(big_f).to_string();
        assert!(shown.contains(head), "{shown}");
        // F′ − f must vanish: symbolically, or at sample points over ℂ.
        let d = super::super::ddx(big_f, x, pool).unwrap();
        let diff = super::super::simp(
            pool.add(vec![d, pool.mul(vec![pool.integer(-1_i32), f])]),
            pool,
        );
        if super::super::is_zero(diff, pool) {
            return;
        }
        for xv in [0.3, 0.55, 0.9, 1.7] {
            let mut env = std::collections::HashMap::new();
            env.insert(x, super::super::verify::C64::real(xv));
            let v = super::super::verify::eval_complex(diff, &env, pool)
                .unwrap_or_else(|| panic!("unevaluable residual {}", pool.display(diff)));
            assert!(
                v.abs() < 1e-9,
                "F' − f = {} at {xv}: {}",
                v.abs(),
                pool.display(diff)
            );
        }
    }

    #[test]
    fn hermite_second_solution_integrand() {
        // e^{x²}/(2x² − 1)²  (H₂ ∝ 2x² − 1)
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let h2 = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), pool.pow(x, pool.integer(2_i32))]),
            pool.integer(-1_i32),
        ]);
        let f = pool.mul(vec![
            pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]),
            pool.pow(h2, pool.integer(-2_i32)),
        ]);
        check(f, x, &pool, "erfi(");
    }

    #[test]
    fn laguerre_second_solution_integrand() {
        // eˣ/(x·(x − 1)²)
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let l1 = pool.add(vec![x, pool.integer(-1_i32)]);
        let f = pool.mul(vec![
            pool.func("exp", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
            pool.pow(l1, pool.integer(-2_i32)),
        ]);
        check(f, x, &pool, "Ei");
    }

    #[test]
    fn gaussian_and_shifted_quadratic() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // e^{−x²/2 + x}
        let g = pool.add(vec![
            pool.mul(vec![pool.rational(-1, 2), pool.pow(x, pool.integer(2_i32))]),
            x,
        ]);
        check(pool.func("exp", vec![g]), x, &pool, "erf");
    }

    #[test]
    fn a_pole_the_basis_cannot_carry_is_declined() {
        // e^{x²}/x has no erf/erfi + elementary antiderivative.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![
            pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        assert!(exp_rational_integral(f, x, &pool).is_none());
        // eˣ/(x² + 1): poles at ±i, outside the rational-root Ei basis.
        let f2 = pool.mul(vec![
            pool.func("exp", vec![x]),
            pool.pow(
                pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]),
                pool.integer(-1_i32),
            ),
        ]);
        assert!(exp_rational_integral(f2, x, &pool).is_none());
    }
}
