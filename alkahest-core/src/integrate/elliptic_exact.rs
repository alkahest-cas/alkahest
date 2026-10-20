//! Exact definite elliptic integrals `∫_a^b R(x)/√P(x) dx` for radicands that
//! split over `ℚ`.
//!
//! The indefinite elliptic reduction (`algebraic::elliptic_output`) finds the
//! roots of `P` numerically and lifts its reduction constants from `f64`; a
//! definite integral built on it is refused (W3), because `F(b) − F(a)` would be
//! an exact-looking expression in float-derived rationals.  When `P` is a cubic
//! or quartic with **rational roots**, every constant of the Legendre reduction
//! is rational (or the square root of one), so the definite integral can be
//! computed exactly instead.  That is what this module does.
//!
//! # The reduction
//!
//! Put the roots of `P` (with `∞` added for a cubic) on the projective line:
//! four real points.  The interval `[a, b]` lies on one arc `(u, v)` between two
//! cyclically adjacent points, on which `P > 0`; call `z` the point beyond `u`
//! and `w` the point beyond `v`.  The Möbius map with `h(u) = 0`, `h(v) = 1`,
//! `h(z) = ∞` is the cross-ratio
//!
//! ```text
//!   h(x) = [x,u]·[v,z] / ([x,z]·[v,u]),        [p,q] = p₁q₂ − p₂q₁,
//! ```
//!
//! and with `sin²φ = h(x)` and `m = 1/h(w) ∈ (0, 1)`,
//!
//! ```text
//!   φ′(x)/√(1 − m sin²φ) = c/√P(x),     c² = P·h′² / (4h(1−h)(1−mh))  ∈ ℚ,
//! ```
//!
//! a constant (computed exactly at an interior point and confirmed at a second
//! one), with the sign of `h′`.  So `R(x) dx/√P = (1/c)·R(x(y)) dφ/Δ` where
//! `y = sin²φ` and `x(y)` is the inverse Möbius map.
//!
//! The `φ`-integral of a rational function `ρ(y)` of `y = sin²φ` against
//! `dφ/Δ` is found by an exact ansatz.  With `W(y) = y(1−y)(1−my)`, so that
//! `√W = sin φ cos φ Δ`, one has `d/dφ[√W·T(y)] = (W′T + 2WT′)/Δ`, and
//!
//! ```text
//!   ρ(y) = α + β(1 − my) + Σᵢ γᵢ/(1 − nᵢy) + W′T + 2WT′
//!   ⇒  ∫ ρ dφ/Δ = α F(φ|m) + β E(φ|m) + Σᵢ γᵢ Π(nᵢ; φ|m) + √W·T(y).
//! ```
//!
//! `T` is a rational function whose poles are those of `ρ` (one order lower at
//! a pole that is not a root of `W`, where a `Π` term with `nᵢ = 1/yᵢ` carries
//! the rest), and the unknowns `α, β, γᵢ` and `T`'s numerator are the solution
//! of a linear system over `ℚ`.  An inconsistent system declines.
//!
//! The endpoint values are exact: `y = h(a)` is rational, `φ = asin √y`, and at
//! `y = 1` the complete integrals `K(m)`, `E(m)` appear.  A pole of `T` at an
//! endpoint (a divergent integral) or a `Π` whose path from `φ = 0` would cross
//! its own pole declines.
//!
//! # Soundness
//!
//! Everything above is exact rational arithmetic; nothing is lifted from a
//! float.  As an independent check the assembled value is compared with a
//! tanh–sinh quadrature of the integrand on finite intervals, with every
//! polynomial Taylor-shifted to the nearer endpoint so the `1/√(x − root)`
//! singularity is resolved to full precision; a disagreement beyond `1e-10`
//! (relative) declines.

use rug::{Integer, Rational};

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::risch::poly_rde::{
    degree, expr_to_qpoly_expanded, poly_add, poly_deriv, poly_mul, poly_one, poly_scale,
    qpoly_to_expr, rational_to_expr, trim, QPoly,
};
use super::risch::rational_rde::{expr_to_qrational, poly_div_exact, poly_gcd, poly_sub};

/// A point of the projective line over `ℚ`.
#[derive(Clone, Debug, PartialEq)]
enum Pt {
    Fin(Rational),
    Inf,
}

impl Pt {
    fn pq(&self) -> (Rational, Rational) {
        match self {
            Pt::Fin(r) => (r.clone(), Rational::from(1)),
            Pt::Inf => (Rational::from(1), Rational::from(0)),
        }
    }
}

/// `[p, q] = p₁q₂ − p₂q₁`.
fn bracket(a: &Pt, b: &Pt) -> Rational {
    let (ap, aq) = a.pq();
    let (bp, bq) = b.pq();
    ap * bq - aq * bp
}

/// An integration bound.
#[derive(Clone, Debug, PartialEq)]
enum Ext {
    NegInf,
    Fin(Rational),
    PosInf,
}

impl Ext {
    fn le_q(&self, r: &Rational) -> bool {
        match self {
            Ext::NegInf => true,
            Ext::Fin(a) => a <= r,
            Ext::PosInf => false,
        }
    }
    fn ge_q(&self, r: &Rational) -> bool {
        match self {
            Ext::NegInf => false,
            Ext::Fin(a) => a >= r,
            Ext::PosInf => true,
        }
    }
    fn pt(&self) -> Pt {
        match self {
            Ext::Fin(r) => Pt::Fin(r.clone()),
            _ => Pt::Inf,
        }
    }
    fn lt(&self, other: &Ext) -> bool {
        match (self, other) {
            (Ext::NegInf, Ext::NegInf) | (Ext::PosInf, _) => false,
            (Ext::NegInf, _) | (_, Ext::PosInf) => true,
            (Ext::Fin(a), Ext::Fin(b)) => a < b,
            (Ext::Fin(_), Ext::NegInf) => false,
        }
    }
}

fn bound(e: ExprId, pool: &ExprPool) -> Option<Ext> {
    let pos = pool.pos_infinity();
    if e == pos {
        return Some(Ext::PosInf);
    }
    match pool.get(e) {
        ExprData::Integer(n) => Some(Ext::Fin(Rational::from(n.0.clone()))),
        ExprData::Rational(r) => Some(Ext::Fin(r.0.clone())),
        ExprData::Mul(args) if args.len() == 2 => {
            let m1 = pool.integer(-1_i32);
            ((args[0] == m1 && args[1] == pos) || (args[1] == m1 && args[0] == pos))
                .then_some(Ext::NegInf)
        }
        _ => None,
    }
}

fn mentions(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    crate::kernel::subs::mentions_var(expr, var, pool)
}

fn eval_q(p: &QPoly, x: &Rational) -> Rational {
    let mut acc = Rational::from(0);
    for c in p.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

fn qpow(p: &QPoly, n: usize) -> QPoly {
    let mut acc = poly_one();
    for _ in 0..n {
        acc = poly_mul(&acc, p);
    }
    acc
}

/// The integrand as `consts · (num/den)(x) / √P(x)`, with the odd-power
/// radicands whose product is `P` (each must be positive on the interval).
struct Split {
    consts: Vec<ExprId>,
    num: QPoly,
    den: QPoly,
    p: QPoly,
    odd_bases: Vec<QPoly>,
}

/// The factors of `expr` with integer exponents, looking through products and
/// integer powers of products (`((x+1)·√P)^{−1}`).
fn flatten(expr: ExprId, k: i64, pool: &ExprPool, out: &mut Vec<(ExprId, i64)>) -> Option<()> {
    match pool.get(expr) {
        ExprData::Mul(xs) => {
            for x in xs {
                flatten(x, k, pool, out)?;
            }
        }
        ExprData::Pow { base, exp } if matches!(pool.get(base), ExprData::Mul(_)) => {
            match pool.get(exp) {
                ExprData::Integer(n) => flatten(base, k.checked_mul(n.0.to_i64()?)?, pool, out)?,
                _ => out.push((expr, k)),
            }
        }
        _ => out.push((expr, k)),
    }
    Some(())
}

fn split(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<Split> {
    let mut factors = Vec::new();
    flatten(expr, 1, pool, &mut factors)?;
    let mut s = Split {
        consts: Vec::new(),
        num: poly_one(),
        den: poly_one(),
        p: poly_one(),
        odd_bases: Vec::new(),
    };
    let push_pow = |s: &mut Split, base: &QPoly, k: i64| {
        if k >= 0 {
            s.num = poly_mul(&s.num, &qpow(base, k as usize));
        } else {
            s.den = poly_mul(&s.den, &qpow(base, (-k) as usize));
        }
    };
    for (f, mult) in factors {
        if !mentions(f, var, pool) {
            s.consts.push(if mult == 1 {
                f
            } else {
                pool.pow(f, pool.integer(mult))
            });
            continue;
        }
        // Half-integer powers: `sqrt(B)`, `B^(k/2)`, `sqrt(B)^n`.
        let radical: Option<(ExprId, i64)> = match pool.get(f) {
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                Some((args[0], 1))
            }
            ExprData::Pow { base, exp } => match (pool.get(base), pool.get(exp)) {
                (ExprData::Func { name, args }, ExprData::Integer(n))
                    if name == "sqrt" && args.len() == 1 =>
                {
                    Some((args[0], n.0.to_i64()?))
                }
                (_, ExprData::Rational(r)) if *r.0.denom() == 2 => {
                    Some((base, r.0.numer().to_i64()?))
                }
                _ => None,
            },
            _ => None,
        };
        match radical {
            Some((b, k)) => {
                let k = k.checked_mul(mult)?;
                let bq = trim(expr_to_qpoly_expanded(b, var, pool)?);
                if degree(&bq) < 1 {
                    return None;
                }
                if k.rem_euclid(2) == 1 {
                    // B^{k/2} = B^{(k+1)/2} / √B
                    s.p = poly_mul(&s.p, &bq);
                    s.odd_bases.push(bq.clone());
                    push_pow(&mut s, &bq, (k + 1) / 2);
                } else {
                    push_pow(&mut s, &bq, k / 2);
                }
            }
            None => {
                let (n, d) = expr_to_qrational(f, var, pool)?;
                let (n, d) = if mult >= 0 { (n, d) } else { (d, n) };
                let e = usize::try_from(mult.unsigned_abs()).ok()?;
                s.num = poly_mul(&s.num, &qpow(&n, e));
                s.den = poly_mul(&s.den, &qpow(&d, e));
            }
        }
    }
    s.num = trim(s.num);
    s.den = trim(s.den);
    s.p = trim(s.p);
    if s.num.is_empty() || s.den.is_empty() || s.odd_bases.is_empty() {
        return None;
    }
    Some(s)
}

/// The rational roots of `p` with multiplicities, if `p` splits into linear
/// factors over `ℚ`; `None` otherwise.
fn split_roots(p: &QPoly, pool: &ExprPool) -> Option<Vec<(Rational, usize)>> {
    let p = trim(p.clone());
    let d = degree(&p);
    if d < 1 {
        return Some(Vec::new());
    }
    let mut l = Integer::from(1);
    for c in &p {
        l = l.lcm(c.denom());
    }
    let pi = poly_scale(&p, &Rational::from(l));
    let y = pool.symbol("$ell_exact_y$", Domain::Real);
    let up = crate::poly::UniPoly::from_symbolic(qpoly_to_expr(&pi, y, pool), y, pool).ok()?;
    let fac = up.factor_z().ok()?;
    let mut out = Vec::new();
    let mut total = 0usize;
    for (f, mult) in &fac.factors {
        let mult: usize = (*mult).try_into().ok()?;
        match f.degree() {
            0 => {}
            1 => {
                let c = f.coefficients();
                let r = -Rational::from((c[0].clone(), c[1].clone()));
                out.push((r, mult));
                total += mult;
            }
            _ => return None,
        }
    }
    (total as i64 == d).then_some(out)
}

/// Gaussian elimination over `ℚ`: `cols` unknowns, rows `[a | rhs]`.  Free
/// unknowns are set to zero; `None` if the system is inconsistent.
fn solve_q(mut rows: Vec<Vec<Rational>>, cols: usize) -> Option<Vec<Rational>> {
    let mut pivots: Vec<(usize, usize)> = Vec::new();
    let mut r = 0;
    for c in 0..cols {
        let Some(p) = (r..rows.len()).find(|&i| rows[i][c] != 0) else {
            continue;
        };
        rows.swap(r, p);
        let inv = Rational::from(1) / rows[r][c].clone();
        for v in rows[r].iter_mut() {
            *v *= &inv;
        }
        let pivot_row = rows[r].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i != r && row[c] != 0 {
                let f = row[c].clone();
                for (v, pv) in row.iter_mut().zip(pivot_row.iter()) {
                    *v -= Rational::from(&f * pv);
                }
            }
        }
        pivots.push((r, c));
        r += 1;
        if r == rows.len() {
            break;
        }
    }
    // Consistency: a zero row must have a zero right-hand side.
    for row in &rows[r..] {
        if row[cols] != 0 {
            return None;
        }
    }
    let mut x = vec![Rational::from(0); cols];
    for (r, c) in pivots {
        x[c] = rows[r][cols].clone();
    }
    Some(x)
}

/// What the `φ`-integral of `ρ(y) dφ/Δ` reduces to.
struct Reduction {
    alpha: Rational,
    beta: Rational,
    pis: Vec<(Rational, Rational)>, // (γᵢ, nᵢ)
    t_num: QPoly,
    t_den: QPoly,
}

/// Solve `ρ = α + β(1−my) + Σ γᵢ/(1−nᵢy) + W′T + 2WT′` for `ρ = ny/dy`.
fn reduce_phi(ny: &QPoly, dy: &QPoly, m: &Rational, pool: &ExprPool) -> Option<Reduction> {
    let one = Rational::from(1);
    let w_roots = [
        Rational::from(0),
        one.clone(),
        Rational::from(1) / m.clone(),
    ];
    let poles = split_roots(dy, pool)?;
    // T's denominator L and the Π characteristics.
    let mut l = poly_one();
    let mut ns: Vec<Rational> = Vec::new();
    for (y0, k) in &poles {
        let lin: QPoly = vec![-y0.clone(), one.clone()];
        let e = if w_roots.contains(y0) {
            *k
        } else {
            ns.push(one.clone() / y0.clone());
            k - 1
        };
        l = poly_mul(&l, &qpow(&lin, e));
    }
    let p_excess = (degree(ny) - degree(dy)).max(0);
    let nt = (degree(&l) + (p_excess - 2).max(0)) as usize; // deg N_T ≤ nt
                                                            // W = y(1−y)(1−my), W′.
    let w = poly_mul(
        &poly_mul(
            &vec![Rational::from(0), one.clone()],
            &vec![one.clone(), -one.clone()],
        ),
        &vec![one.clone(), -m.clone()],
    );
    let wp = poly_deriv(&w);
    let lp = poly_deriv(&l);
    let pi_lin: Vec<QPoly> = ns.iter().map(|n| vec![one.clone(), -n.clone()]).collect();
    let pi_all = pi_lin.iter().fold(poly_one(), |acc, f| poly_mul(&acc, f));
    let l2 = poly_mul(&l, &l);
    let common = poly_mul(&poly_mul(dy, &l2), &pi_all); // M
    let target = poly_mul(&poly_mul(ny, &l2), &pi_all);

    let mut contribs: Vec<QPoly> = Vec::new();
    contribs.push(common.clone()); // α
    contribs.push(poly_mul(&common, &vec![one.clone(), -m.clone()])); // β
    for f in &pi_lin {
        contribs.push(poly_div_exact(&common, f)); // γᵢ
    }
    let dy_pi = poly_mul(dy, &pi_all);
    for j in 0..=nt {
        let mut yj = vec![Rational::from(0); j + 1];
        yj[j] = one.clone();
        let yjp = poly_deriv(&yj);
        // W′·yʲ·L + 2W(j·yʲ⁻¹·L − yʲ·L′)
        let inner = poly_add(
            &poly_mul(&poly_mul(&wp, &yj), &l),
            &poly_scale(
                &poly_mul(&w, &poly_sub(&poly_mul(&yjp, &l), &poly_mul(&yj, &lp))),
                &Rational::from(2),
            ),
        );
        contribs.push(poly_mul(&dy_pi, &inner));
    }
    let cols = contribs.len();
    let height = contribs
        .iter()
        .map(|c| c.len())
        .chain(std::iter::once(target.len()))
        .max()?;
    let at = |p: &QPoly, i: usize| p.get(i).cloned().unwrap_or_else(|| Rational::from(0));
    let rows: Vec<Vec<Rational>> = (0..height)
        .map(|i| {
            let mut row: Vec<Rational> = contribs.iter().map(|c| at(c, i)).collect();
            row.push(at(&target, i));
            row
        })
        .collect();
    let x = solve_q(rows, cols)?;
    let g = ns.len();
    let t_num = trim(x[2 + g..].to_vec());
    // Cancel common factors of T so an endpoint test sees its true poles.
    let (t_num, t_den) = if t_num.is_empty() {
        (t_num, poly_one())
    } else {
        let gcd = poly_gcd(&t_num, &l);
        (poly_div_exact(&t_num, &gcd), poly_div_exact(&l, &gcd))
    };
    Some(Reduction {
        alpha: x[0].clone(),
        beta: x[1].clone(),
        pis: ns
            .into_iter()
            .zip(x[2..2 + g].iter().cloned())
            .map(|(n, gm)| (gm, n))
            .collect(),
        t_num,
        t_den,
    })
}

/// `R(x(y))` as `ny/dy` for the Möbius map `x = (Ay+B)/(Cy+D)`.
fn substitute(
    num: &QPoly,
    den: &QPoly,
    a: &Rational,
    b: &Rational,
    c: &Rational,
    d: &Rational,
) -> (QPoly, QPoly) {
    let lin_n: QPoly = trim(vec![b.clone(), a.clone()]);
    let lin_d: QPoly = trim(vec![d.clone(), c.clone()]);
    let homog = |p: &QPoly| -> QPoly {
        let k = degree(p).max(0) as usize;
        let mut acc: QPoly = Vec::new();
        for (i, ci) in p.iter().enumerate() {
            if *ci == 0 {
                continue;
            }
            let term = poly_scale(&poly_mul(&qpow(&lin_n, i), &qpow(&lin_d, k - i)), ci);
            acc = poly_add(&acc, &term);
        }
        trim(acc)
    };
    let dn = degree(num).max(0) as usize;
    let dd = degree(den).max(0) as usize;
    let mut ny = homog(num);
    let mut dy = homog(den);
    if dd >= dn {
        ny = poly_mul(&ny, &qpow(&lin_d, dd - dn));
    } else {
        dy = poly_mul(&dy, &qpow(&lin_d, dn - dd));
    }
    let g = poly_gcd(&ny, &dy);
    (trim(poly_div_exact(&ny, &g)), trim(poly_div_exact(&dy, &g)))
}

fn half_pi(pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.rational(1, 2), pool.symbol("pi", Domain::Real)])
}

fn sqrt_q(r: &Rational, pool: &ExprPool) -> ExprId {
    pool.func("sqrt", vec![rational_to_expr(r, pool)])
}

/// `G(y) = α F + β E + Σ γ Π + √W·T` at `y = sin²φ`, or `None` where it is
/// not finite (a pole of `T`) or a `Π` would integrate through its own pole.
fn g_at(red: &Reduction, y: &Rational, m: &Rational, pool: &ExprPool) -> Option<ExprId> {
    let one = Rational::from(1);
    if red.t_num.iter().any(|c| *c != 0) && eval_q(&red.t_den, y) == 0 {
        return None;
    }
    for (_, n) in &red.pis {
        if Rational::from(n * y) >= one {
            return None;
        }
    }
    if *y == 0 {
        return Some(pool.integer(0_i32));
    }
    let m_e = rational_to_expr(m, pool);
    let mut terms: Vec<ExprId> = Vec::new();
    let complete = *y == one;
    let phi = if complete {
        half_pi(pool)
    } else {
        pool.func("asin", vec![sqrt_q(y, pool)])
    };
    if red.alpha != 0 {
        let f = if complete {
            pool.func("EllipticK", vec![m_e])
        } else {
            pool.func("EllipticF", vec![phi, m_e])
        };
        terms.push(pool.mul(vec![rational_to_expr(&red.alpha, pool), f]));
    }
    if red.beta != 0 {
        let e = if complete {
            pool.func("EllipticE", vec![m_e])
        } else {
            pool.func("EllipticE", vec![phi, m_e])
        };
        terms.push(pool.mul(vec![rational_to_expr(&red.beta, pool), e]));
    }
    for (gm, n) in &red.pis {
        if *gm != 0 {
            let p = pool.func("EllipticPi", vec![rational_to_expr(n, pool), phi, m_e]);
            terms.push(pool.mul(vec![rational_to_expr(gm, pool), p]));
        }
    }
    if !complete && red.t_num.iter().any(|c| *c != 0) {
        let wv = (y * (one.clone() - y)) * (one.clone() - Rational::from(m * y));
        let tv = eval_q(&red.t_num, y) / eval_q(&red.t_den, y);
        if tv != 0 {
            terms.push(pool.mul(vec![rational_to_expr(&tv, pool), sqrt_q(&wv, pool)]));
        }
    }
    Some(match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    })
}

/// `p(c + σ·s)` as a polynomial in `s` (exact Taylor shift), for `σ = ±1`.
fn shifted(p: &QPoly, c: &Rational, sigma: i32) -> Vec<f64> {
    let lin: QPoly = vec![c.clone(), Rational::from(sigma)];
    let mut acc: QPoly = Vec::new();
    for coef in p.iter().rev() {
        acc = poly_mul(&acc, &lin);
        acc = poly_add(&acc, &vec![coef.clone()]);
    }
    trim(acc).iter().map(|r| r.to_f64()).collect()
}

fn horner(c: &[f64], s: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &k| acc * s + k)
}

/// `∫_a^b consts·(num/den)(x)/√P(x) dx` by tanh–sinh quadrature, evaluating
/// every polynomial in the distance `s` to the nearer endpoint (an exact Taylor
/// shift), so the `1/√s` singularity at a root of `P` loses no digits to the
/// cancellation in `x − root`.  Independent of the reduction above.
fn quadrature(sp: &Split, a: &Rational, b: &Rational, pool: &ExprPool) -> Option<f64> {
    let k = if sp.consts.is_empty() {
        1.0
    } else {
        crate::jit::eval_interp(
            pool.mul(sp.consts.clone()),
            &std::collections::HashMap::new(),
            pool,
        )
        .filter(|v| v.is_finite())?
    };
    let ends = [(a, 1), (b, -1)];
    let polys: Vec<[Vec<f64>; 3]> = ends
        .iter()
        .map(|(c, sg)| {
            [
                shifted(&sp.num, c, *sg),
                shifted(&sp.den, c, *sg),
                shifted(&sp.p, c, *sg),
            ]
        })
        .collect();
    let half = Rational::from(b - a).to_f64() / 2.0;
    let f = |end: usize, s: f64| -> Option<f64> {
        let [n, d, p] = &polys[end];
        let pv = horner(p, s);
        let dv = horner(d, s);
        if pv <= 0.0 || dv == 0.0 {
            return None;
        }
        Some(horner(n, s) / dv / pv.sqrt())
    };
    const STEP: f64 = 1.0 / 128.0;
    const T_MAX: f64 = 4.5;
    let hp = std::f64::consts::FRAC_PI_2;
    let mut total = 0.0;
    let n = (T_MAX / STEP) as i64;
    for j in -n..=n {
        let t = j as f64 * STEP;
        let u = hp * t.sinh();
        let cu = u.cosh();
        let gap = half * (-u.abs()).exp() / cu; // distance to the nearer end
        let w = STEP * half * hp * t.cosh() / (cu * cu);
        if gap <= 0.0 || w == 0.0 || !w.is_finite() {
            continue;
        }
        let end = if t >= 0.0 { 1 } else { 0 };
        let y = f(end, gap)?;
        if !y.is_finite() {
            return None;
        }
        total += w * y;
    }
    Some(k * total)
}

/// `∫_lower^upper expr d var` exactly, for `R(x)/√P(x)` with `P` a cubic or
/// quartic splitting over `ℚ`.  `None` when the shape does not apply or any
/// step declines (see the module docs); the caller falls through.
pub(super) fn try_definite(
    expr: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    definite(expr, var, lower, upper, pool).ok()
}

#[cfg(test)]
std::thread_local! {
    static SKIP_QUADRATURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`try_definite`] with the reason for a decline (for tests and debugging).
fn definite(
    expr: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, &'static str> {
    let (mut a, mut b) = (
        bound(lower, pool).ok_or("bound")?,
        bound(upper, pool).ok_or("bound")?,
    );
    if a == b {
        return Err("empty interval");
    }
    let mut orient = 1;
    if b.lt(&a) {
        std::mem::swap(&mut a, &mut b);
        orient = -1;
    }
    let sp = split(expr, var, pool).ok_or("not R(x)/sqrt(P)")?;
    let deg = degree(&sp.p);
    if deg != 3 && deg != 4 {
        return Err("deg P not 3 or 4");
    }
    let roots = split_roots(&sp.p, pool).ok_or("P does not split over Q")?;
    if roots.len() as i64 != deg || roots.iter().any(|(_, k)| *k != 1) {
        return Err("P not squarefree");
    }
    let mut finite: Vec<Rational> = roots.into_iter().map(|(r, _)| r).collect();
    finite.sort();
    let mut pts: Vec<Pt> = finite.iter().cloned().map(Pt::Fin).collect();
    if deg == 3 {
        pts.push(Pt::Inf);
    }

    // The arc (u, v) containing [a, b].
    let inside = |u: &Pt, v: &Pt| -> bool {
        match (u, v) {
            (Pt::Fin(u), Pt::Fin(v)) if u < v => a.ge_q(u) && b.le_q(v),
            // Wraps through ∞ (a quartic's outer arc): [a, b] lies in one of
            // its two real pieces, (u, ∞) or (−∞, v).
            (Pt::Fin(u), Pt::Fin(v)) => a.ge_q(u) || b.le_q(v),
            (Pt::Fin(u), Pt::Inf) => a.ge_q(u),
            (Pt::Inf, Pt::Fin(v)) => b.le_q(v),
            (Pt::Inf, Pt::Inf) => false,
        }
    };
    let i = (0..4)
        .find(|&i| inside(&pts[i], &pts[(i + 1) % 4]))
        .ok_or("interval crosses a root")?;
    let (u, v, w, z) = (
        pts[i].clone(),
        pts[(i + 1) % 4].clone(),
        pts[(i + 2) % 4].clone(),
        pts[(i + 3) % 4].clone(),
    );

    // An interior sample point.
    let x0 = match (&a, &b) {
        (Ext::Fin(p), Ext::Fin(q)) => Rational::from(p + q) / 2,
        (Ext::Fin(p), Ext::PosInf) => Rational::from(p + 1),
        (Ext::NegInf, Ext::Fin(q)) => Rational::from(q - 1),
        _ => return Err("both bounds infinite"),
    };
    if eval_q(&sp.p, &x0) <= 0 || sp.odd_bases.iter().any(|bq| eval_q(bq, &x0) <= 0) {
        return Err("integrand not real on the interval");
    }
    // No pole of the rational part on the closed interval; every pole rational.
    let den_roots = split_roots(&sp.den, pool).ok_or("irrational pole")?;
    if den_roots.iter().any(|(r, _)| a.le_q(r) && b.ge_q(r)) {
        return Err("pole on the interval");
    }

    // The Möbius map and the parameter.
    let kn = bracket(&v, &z);
    let kd = bracket(&v, &u);
    let m = bracket(&w, &z) * kd.clone() / (bracket(&w, &u) * kn.clone());
    if !(m > 0 && m < 1) {
        return Err("parameter outside (0, 1)");
    }
    let lin = |p: &Pt| -> (Rational, Rational) {
        match p {
            Pt::Fin(r) => (Rational::from(1), -r.clone()),
            Pt::Inf => (Rational::from(0), Rational::from(-1)),
        }
    };
    let (au, bu) = lin(&u);
    let (az, bz) = lin(&z);
    let h = |x: &Pt| -> Option<Rational> {
        let den = bracket(x, &z) * kd.clone();
        (den != 0).then(|| bracket(x, &u) * kn.clone() / den)
    };
    // h′(x) = (kn/kd)·(α_u β_z − β_u α_z)/(α_z x + β_z)².
    let hp = |x: &Rational| -> Rational {
        let s = Rational::from(&az * x) + &bz;
        kn.clone() / kd.clone() * (au.clone() * &bz - bu.clone() * &az) / (s.clone() * s)
    };
    let c2_at = |x: &Rational| -> Option<Rational> {
        let hv = h(&Pt::Fin(x.clone()))?;
        let one = Rational::from(1);
        let den = Rational::from(4)
            * hv.clone()
            * (one.clone() - hv.clone())
            * (one - Rational::from(&m * &hv));
        if den == 0 {
            return None;
        }
        let d = hp(x);
        Some(eval_q(&sp.p, x) * d.clone() * d / den)
    };
    let c2 = c2_at(&x0).ok_or("c2")?;
    let x1 = &x0 + Rational::from((1, 7));
    if c2 <= 0 || c2_at(&x1).is_some_and(|c| c != c2) {
        return Err("reduction constant not constant");
    }
    let sign = if hp(&x0) > 0 { 1 } else { -1 };

    // x(y) = (Ay + B)/(Cy + D).
    let ca = -(kd.clone() * &bz);
    let cb = kn.clone() * &bu;
    let cc = kd.clone() * &az;
    let cd = -(kn.clone() * &au);
    let (ny, dy) = substitute(&sp.num, &sp.den, &ca, &cb, &cc, &cd);
    let red = reduce_phi(&ny, &dy, &m, pool).ok_or("no reduction")?;

    let ya = h(&a.pt()).ok_or("h(a)")?;
    let yb = h(&b.pt()).ok_or("h(b)")?;
    let ga = g_at(&red, &ya, &m, pool).ok_or("divergent at a")?;
    let gb = g_at(&red, &yb, &m, pool).ok_or("divergent at b")?;
    let diff = pool.add(vec![gb, pool.mul(vec![pool.integer(-1_i32), ga])]);
    // 1/c = sign · (c²)^{−1/2}
    let inv_c = pool.mul(vec![
        pool.integer(sign * orient),
        sqrt_q(&(Rational::from(1) / c2.clone()), pool),
    ]);
    let mut factors = sp.consts.clone();
    factors.push(inv_c);
    factors.push(diff);
    let value = simplify(pool.mul(factors), pool).value;
    #[cfg(test)]
    if SKIP_QUADRATURE.with(|c| c.get()) {
        return Ok(value);
    }

    // Independent numeric check on finite intervals, in `x`.
    if let (Ext::Fin(p), Ext::Fin(q)) = (&a, &b) {
        let want = quadrature(&sp, p, q, pool).ok_or("quadrature failed")?;
        let got = crate::jit::eval_interp(value, &std::collections::HashMap::new(), pool)
            .filter(|v| v.is_finite())
            .ok_or("value does not evaluate")?
            * f64::from(orient);
        if (got - want).abs() > 1e-10 * want.abs().max(1.0) {
            return Err("quadrature disagrees");
        }
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn value(e: ExprId, pool: &ExprPool) -> f64 {
        crate::jit::eval_interp(e, &HashMap::new(), pool).expect("closed value evaluates")
    }

    /// `(x − r)` as an expression.
    fn lin(x: ExprId, r: i64, pool: &ExprPool) -> ExprId {
        pool.add(vec![x, pool.integer(-r)])
    }

    fn inv_sqrt(p: ExprId, pool: &ExprPool) -> ExprId {
        pool.pow(p, pool.rational(-1, 2))
    }

    fn check(f: ExprId, x: ExprId, a: ExprId, b: ExprId, want: f64, pool: &ExprPool) {
        SKIP_QUADRATURE.with(|c| c.set(true));
        let raw = definite(f, x, a, b, pool);
        SKIP_QUADRATURE.with(|c| c.set(false));
        if let Ok(r) = raw {
            let v = value(r, pool);
            assert!(
                (v - want).abs() <= 1e-13 * want.abs().max(1.0),
                "unchecked value {v}"
            );
        }
        let got = definite(f, x, a, b, pool)
            .unwrap_or_else(|why| panic!("{} declined: {why}", pool.display(f)));
        let v = value(got, pool);
        assert!(
            (v - want).abs() <= 1e-13 * want.abs().max(1.0),
            "{} = {} ≈ {v}, want {want}",
            pool.display(f),
            pool.display(got)
        );
    }

    #[test]
    fn quartic_four_rational_roots() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.mul(vec![
            lin(x, 1, &pool),
            lin(x, 2, &pool),
            lin(x, 3, &pool),
            lin(x, 4, &pool),
        ]);
        let f = inv_sqrt(p, &pool);
        check(
            f,
            x,
            pool.integer(2),
            pool.integer(3),
            2.156_515_647_499_643_2,
            &pool,
        );
        check(
            f,
            x,
            pool.integer(4),
            pool.pos_infinity(),
            1.078_257_823_749_821_6,
            &pool,
        );
        // (1 − x²)(4 − x²) on [0, 1/2] and x²/… on [0, 1].
        let q = pool.mul(vec![
            pool.add(vec![
                pool.integer(1),
                pool.mul(vec![pool.integer(-1), pool.pow(x, pool.integer(2))]),
            ]),
            pool.add(vec![
                pool.integer(4),
                pool.mul(vec![pool.integer(-1), pool.pow(x, pool.integer(2))]),
            ]),
        ]);
        check(
            inv_sqrt(q, &pool),
            x,
            pool.integer(0),
            pool.rational(1, 2),
            0.264_714_313_525_952_9,
            &pool,
        );
        let g = pool.mul(vec![pool.pow(x, pool.integer(2)), inv_sqrt(q, &pool)]);
        check(
            g,
            x,
            pool.integer(0),
            pool.integer(1),
            0.436_576_290_946_337_8,
            &pool,
        );
    }

    #[test]
    fn cubic_first_second_and_third_kind() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // x(1 − x)(2 − x) = x(x − 1)(x − 2)
        let p = pool.mul(vec![x, lin(x, 1, &pool), lin(x, 2, &pool)]);
        let f = inv_sqrt(p, &pool);
        check(
            f,
            x,
            pool.integer(0),
            pool.integer(1),
            2.622_057_554_292_119_8,
            &pool,
        );
        let g = pool.mul(vec![x, f]);
        check(
            g,
            x,
            pool.integer(0),
            pool.integer(1),
            1.423_917_319_556_527_6,
            &pool,
        );
        let h = pool.mul(vec![pool.pow(lin(x, -1, &pool), pool.integer(-1)), f]);
        check(
            h,
            x,
            pool.integer(0),
            pool.integer(1),
            1.800_473_988_685_746_3,
            &pool,
        );
        // Reversed bounds flip the sign.
        check(
            f,
            x,
            pool.integer(1),
            pool.integer(0),
            -2.622_057_554_292_119_8,
            &pool,
        );
    }

    #[test]
    fn declines_are_reasoned() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.mul(vec![x, lin(x, 1, &pool), lin(x, 2, &pool)]);
        let f = inv_sqrt(p, &pool);
        // Crossing a root of P.
        assert!(definite(f, x, pool.integer(0), pool.integer(3), &pool).is_err());
        // P < 0 on (1, 2).
        assert!(definite(f, x, pool.integer(1), pool.integer(2), &pool).is_err());
        // A pole of the rational part inside the interval.
        let g = pool.mul(vec![
            pool.pow(pool.add(vec![x, pool.rational(-1, 2)]), pool.integer(-1)),
            f,
        ]);
        assert!(definite(g, x, pool.integer(0), pool.integer(1), &pool).is_err());
        // Divergent at ∞: ∫_2^∞ x/√(x(x−1)(x−2)).
        let h = pool.mul(vec![x, f]);
        assert!(definite(h, x, pool.integer(2), pool.pos_infinity(), &pool).is_err());
        // x³ + 1 does not split over ℚ.
        let c = pool.add(vec![pool.pow(x, pool.integer(3)), pool.integer(1)]);
        assert!(definite(
            inv_sqrt(c, &pool),
            x,
            pool.integer(0),
            pool.integer(1),
            &pool
        )
        .is_err());
    }
}
