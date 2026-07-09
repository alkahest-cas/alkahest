//! Coates' algorithm (hyperelliptic case): construct a function `u` on the curve
//! `y² = a(x)` with a prescribed **principal** divisor `D`.
//!
//! For `∫ B(x)·√a dx` on a genus ≥ 2 hyperelliptic curve whose residue divisor
//! `δ` is torsion of order `N` ([`super::jacobian_torsion`] decides this), the
//! integral is elementary and equals `g + (1/N)·log(u)` where `u ∈ ℚ(x)(y)` has
//! `div(u) = N·δ` (a principal divisor).  Emitting the log part therefore needs
//! a **constructive** step — the decision procedure only certifies principality.
//!
//! Davenport's presentation of Coates' algorithm (COATES + INTEGRAL_BASIS_ and
//! NORMAL_BASIS_REDUCTION) builds a basis for the space of functions with poles
//! bounded by a divisor and then imposes the zeros.  For the hyperelliptic case
//! `n = 2` we use the equivalent — and much more direct — **Cantor/Mumford**
//! realisation: a principal divisor on `y² = F(X)` (monic, odd degree) is the
//! divisor of a product of the elementary functions `(Y − v(X))` and `(X − α)`
//! that appear as Cantor's reduction and the hyperelliptic involution act on the
//! Mumford representation.  Concretely we fold the divisor into the Jacobian one
//! place at a time, keeping the running class **reduced** (Cantor) and tracking
//! the realising function; when the total divisor is principal the running class
//! collapses to the identity and the accumulated function is `u`.
//!
//! # Invariant
//!
//! Throughout, `(u, v)` is a semi-reduced Mumford pair (`v² ≡ F (mod u)`) and
//! `ans` a function on the curve with
//!
//! ```text
//!     T_partial  =  div(u, v)  +  div(ans)
//! ```
//!
//! where `T_partial` is the sum of the divisor units processed so far and
//! `div(u, v) = Σ_{roots α of u} (α, v(α)) − deg(u)·∞`.  When every unit of the
//! (principal) target `T` has been folded in, `[T] = 0`, so the reduced running
//! class is the identity `(1, 0)` and `div(ans) = T`.  The result is returned as
//! a symbolic function of `x` and `√a`; **its correctness is additionally gated
//! numerically** at the call site (`d/dx F = integrand`), so a construction that
//! ever produced a wrong `u` would be declined rather than emitted.
//!
//! Scope: `n = 2`, `a` squarefree of **odd** degree `2g+1` (imaginary model, a
//! single place at infinity).  Even-degree (real) models and `n > 2` return
//! `None` (the caller then declines soundly).

use rug::{Integer, Rational};

use super::super::risch::poly_rde::{
    degree, poly_add, poly_mul, poly_scale, qpoly_to_expr, trim, QPoly,
};
use super::super::risch::rational_rde::{poly_divrem, poly_monic, poly_sub};
use crate::kernel::{ExprId, ExprPool};

/// A finite place `(x, y)` of `y² = a(x)` with an integer multiplicity.
#[derive(Clone, Debug)]
pub(crate) struct CoatesPlace {
    pub x: Rational,
    /// Sheet value: `y² = a(x)`.  `y = 0` marks a branch (ramified) place.
    pub y: Rational,
    pub coeff: Integer,
}

// ---------------------------------------------------------------------------
// Small ℚ[x] helpers
// ---------------------------------------------------------------------------

fn q_is_zero(p: &QPoly) -> bool {
    degree(p) < 0
}

fn poly_rem(a: &QPoly, b: &QPoly) -> QPoly {
    trim(poly_divrem(a, b).1)
}

fn poly_eval(p: &QPoly, x: &Rational) -> Rational {
    let mut acc = Rational::from(0);
    for c in p.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// `r^e` for a rational base and (possibly negative) integer exponent.
fn pow_rat(r: &Rational, e: i64) -> Rational {
    let mut acc = Rational::from(1);
    if e >= 0 {
        for _ in 0..e {
            acc *= r;
        }
        acc
    } else {
        for _ in 0..(-e) {
            acc *= r;
        }
        Rational::from(1) / acc
    }
}

/// Substitute `X = lc·x` into `p(X)`, returning `p(lc·x)` as a polynomial in `x`.
fn subst_lc(p: &QPoly, lc: &Rational) -> QPoly {
    let mut out = p.clone();
    let mut s = Rational::from(1);
    for c in out.iter_mut() {
        *c *= &s;
        s *= lc;
    }
    trim(out)
}

/// Linear factor `X − α`.
fn linear(alpha: &Rational) -> QPoly {
    trim(vec![-alpha.clone(), Rational::from(1)])
}

// ---------------------------------------------------------------------------
// The running state: reduced Mumford class + tracked realising function.
// ---------------------------------------------------------------------------

/// `ans = (num_x(X) / den_x(X)) · Π_j (Y − yfac_j(X))`.  Only the numerator ever
/// carries the algebraic generator `Y`, so `ans ∈ ℚ(X)[Y]`.
struct Coates<'a> {
    f: &'a QPoly, // monic curve, deg 2g+1
    g: usize,
    u: QPoly,
    v: QPoly,
    num_x: QPoly,
    den_x: QPoly,
    yfac: Vec<QPoly>,
}

impl<'a> Coates<'a> {
    fn new(f: &'a QPoly, g: usize) -> Self {
        Coates {
            f,
            g,
            u: vec![Rational::from(1)],
            v: vec![],
            num_x: vec![Rational::from(1)],
            den_x: vec![Rational::from(1)],
            yfac: Vec::new(),
        }
    }

    /// Cantor reduction with function tracking: while `deg u > g`, replace
    /// `(u, v)` by the reduced `(w, −v mod w)` and multiply `ans` by
    /// `(Y − v)/w`, which keeps `T_partial = div(u,v) + div(ans)` invariant
    /// because `div(u_old, v_old) = div(w, −v mod w) + div((Y − v_old)/w)`.
    fn reduce(&mut self) -> Option<()> {
        while degree(&self.u) > self.g as i64 {
            let v2 = poly_mul(&self.v, &self.v);
            let num = poly_sub(self.f, &v2);
            let (w, r) = poly_divrem(&num, &self.u);
            if !q_is_zero(&r) {
                return None; // not semi-reduced — should not happen
            }
            let w = poly_monic(&w);
            // ans *= (Y − v)/w
            self.yfac.push(self.v.clone());
            self.den_x = poly_mul(&self.den_x, &w);
            // step
            let nv = poly_scale(&self.v, &Rational::from(-1));
            let vp = if degree(&w) <= 0 {
                vec![]
            } else {
                poly_rem(&nv, &w)
            };
            self.u = w;
            self.v = vp;
        }
        Some(())
    }

    /// Append a point `(α, β)` to the Mumford pair, increasing its multiplicity
    /// there by one: `u ← u·(X−α)`, `v ← v + u_old·t` with `t` chosen so the
    /// pair stays semi-reduced.  Requires `β² = F(α)`.  (α may already be a root
    /// of `u` on the *same* sheet — a multiplicity bump.)
    fn mumford_append(&mut self, alpha: &Rational, beta: &Rational) -> Option<()> {
        let u_old = self.u.clone();
        let ua = poly_eval(&u_old, alpha);
        let t = if ua != 0 {
            // v(α) + u(α)·t = β
            (beta.clone() - poly_eval(&self.v, alpha)) / ua
        } else {
            // α is a root of u with v(α)=β (same sheet).  Bump multiplicity:
            // q = (v²−F)/u, t = −q(α)/(2β).  (β ≠ 0 here — branch points never
            // reach this path.)
            if *beta == 0 {
                return None;
            }
            let v2 = poly_mul(&self.v, &self.v);
            let num = poly_sub(&v2, self.f);
            let (q, r) = poly_divrem(&num, &u_old);
            if !q_is_zero(&r) {
                return None;
            }
            -poly_eval(&q, alpha) / (Rational::from(2) * beta)
        };
        self.u = poly_mul(&u_old, &linear(alpha));
        self.v = trim(poly_add(&self.v, &poly_scale(&u_old, &t)));
        Some(())
    }

    /// Remove one copy of the point `(α, v(α))` from the Mumford pair.
    fn mumford_remove(&mut self, alpha: &Rational) {
        let (q, _r) = poly_divrem(&self.u, &linear(alpha));
        self.u = poly_monic(&trim(q));
        self.v = if degree(&self.u) <= 0 {
            vec![]
        } else {
            poly_rem(&self.v, &self.u)
        };
    }

    /// Fold in `+[(α, β) − ∞]` (one unit).  Handles the three configurations:
    /// append (α not a root), same-sheet multiplicity bump, and opposite-sheet
    /// (or branch) cancellation `(α,β)+(α,−β) = div(X−α)`.
    fn add_point(&mut self, alpha: &Rational, beta: &Rational) -> Option<()> {
        let on_u = degree(&self.u) >= 0 && poly_eval(&self.u, alpha) == 0;
        if on_u {
            let vat = poly_eval(&self.v, alpha);
            if *beta != 0 && vat == *beta {
                // same sheet → bump multiplicity
                self.mumford_append(alpha, beta)?;
                self.reduce()?;
            } else {
                // opposite sheet, or branch pair → cancel, ans *= (X−α)
                self.mumford_remove(alpha);
                self.num_x = poly_mul(&self.num_x, &linear(alpha));
            }
        } else {
            self.mumford_append(alpha, beta)?;
            self.reduce()?;
        }
        Some(())
    }

    /// Fold in `sign·[(α, β) − ∞]` (one unit, `sign = ±1`).
    fn apply_unit(&mut self, sign: i32, alpha: &Rational, beta: &Rational) -> Option<()> {
        if sign > 0 {
            self.add_point(alpha, beta)
        } else {
            // −[(α,β)−∞]: if (α,β) is present, just remove it; otherwise use
            // −[(α,β)−∞] = +[(α,−β)−∞] − div(X−α).
            let on_u = degree(&self.u) >= 0 && poly_eval(&self.u, alpha) == 0;
            if on_u && poly_eval(&self.v, alpha) == *beta {
                self.mumford_remove(alpha);
            } else {
                self.add_point(alpha, &(-beta.clone()))?;
                self.den_x = poly_mul(&self.den_x, &linear(alpha));
            }
            Some(())
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

/// Construct `u ∈ ℚ(x)(y)` with `div(u) = Σ places` on `y² = a(x)`, or `None` if
/// the divisor is not principal in the handled scope.
///
/// `places` is a **degree-0** divisor of finite places; the (single, ramified)
/// place at infinity is implied by degree balance `Σ coeff·(−∞)`.  `a` must be
/// squarefree of **odd** degree `≥ 3`.  The returned expression is a symbolic
/// function of `var` and `sqrt(a)`.
pub(crate) fn coates_hyperelliptic(
    a: &QPoly,
    places: &[CoatesPlace],
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let a = trim(a.clone());
    let d = degree(&a);
    if d < 3 || d % 2 == 0 {
        return None; // odd model only
    }
    let dd = d as usize;
    let g = (dd - 1) / 2;

    // Monic curve F(X): X = lc·x, Y = lc^g·y, F_k = a_k·lc^{d-1-k}.
    let lc = a[dd].clone();
    let mut f = vec![Rational::from(0); dd + 1];
    for (k, slot) in f.iter_mut().enumerate() {
        let e = dd as i64 - 1 - k as i64;
        *slot = a[k].clone() * pow_rat(&lc, e);
    }
    let f = trim(f);
    let lc_g = pow_rat(&lc, g as i64);

    let mut st = Coates::new(&f, g);
    for pl in places {
        if pl.coeff == 0 {
            continue;
        }
        let big_x = lc.clone() * &pl.x;
        let big_y = lc_g.clone() * &pl.y;
        // On-curve sanity: Y² = F(X).
        if big_y.clone() * &big_y != poly_eval(&f, &big_x) {
            return None;
        }
        let sign = if pl.coeff < 0 { -1 } else { 1 };
        let reps = pl.coeff.clone().abs().to_u64()?;
        for _ in 0..reps {
            st.apply_unit(sign, &big_x, &big_y)?;
        }
    }

    // Principal ⇒ the running class must have collapsed to the identity.
    if degree(&st.u) != 0 {
        return None;
    }

    // Assemble ans = (num_x/den_x)·Π(Y − yfac_j), substituting X = lc·x,
    // Y = lc^g·y.
    let y_expr = pool.func("sqrt", vec![qpoly_to_expr(&a, var, pool)]);
    let big_y_expr = if lc_g == 1 {
        y_expr
    } else {
        pool.mul(vec![rat_expr(&lc_g, pool), y_expr])
    };

    let mut num_terms: Vec<ExprId> = Vec::new();
    let num_poly_x = subst_lc(&st.num_x, &lc);
    if !(degree(&num_poly_x) == 0 && num_poly_x[0] == 1) {
        num_terms.push(qpoly_to_expr(&num_poly_x, var, pool));
    }
    for vj in &st.yfac {
        let vpoly_x = subst_lc(vj, &lc);
        let factor = pool.add(vec![
            big_y_expr,
            pool.mul(vec![
                pool.integer(-1_i32),
                qpoly_to_expr(&vpoly_x, var, pool),
            ]),
        ]);
        num_terms.push(factor);
    }
    let num_expr = match num_terms.len() {
        0 => pool.integer(1_i32),
        1 => num_terms.remove(0),
        _ => pool.mul(num_terms),
    };

    let den_poly_x = subst_lc(&st.den_x, &lc);
    let u_expr = if degree(&den_poly_x) == 0 && den_poly_x[0] == 1 {
        num_expr
    } else {
        let den_expr = qpoly_to_expr(&den_poly_x, var, pool);
        pool.mul(vec![num_expr, pool.pow(den_expr, pool.integer(-1_i32))])
    };
    Some(u_expr)
}

fn rat_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    super::super::risch::poly_rde::rational_to_expr(r, pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use crate::simplify::engine::simplify;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }

    fn place(x: i64, y: i64, c: i64) -> CoatesPlace {
        CoatesPlace {
            x: Rational::from(x),
            y: Rational::from(y),
            coeff: Integer::from(c),
        }
    }

    /// Numeric eval of an ExprId at `x = xv`, `sqrt(a) = +√a`.
    fn eval(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> Option<f64> {
        use crate::kernel::ExprData;
        if expr == x {
            return Some(xv);
        }
        match pool.get(expr) {
            ExprData::Integer(n) => Some(n.0.to_f64()),
            ExprData::Rational(r) => Some(r.0.to_f64()),
            ExprData::Add(args) => args
                .iter()
                .try_fold(0.0, |s, &a| Some(s + eval(a, x, xv, pool)?)),
            ExprData::Mul(args) => args
                .iter()
                .try_fold(1.0, |s, &a| Some(s * eval(a, x, xv, pool)?)),
            ExprData::Pow { base, exp } => {
                Some(eval(base, x, xv, pool)?.powf(eval(exp, x, xv, pool)?))
            }
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let v = eval(args[0], x, xv, pool)?;
                match name.as_str() {
                    "sqrt" => Some(v.sqrt()),
                    "log" => Some(v.ln()),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn eval_qp(p: &QPoly, xv: f64) -> f64 {
        p.iter().rev().fold(0.0, |acc, c| acc * xv + c.to_f64())
    }

    /// Check `div(u_coates) = div(ref_fn)` by confirming `u/ref` is constant (a
    /// function is determined by its divisor up to a nonzero scalar).
    fn assert_constant_ratio(u: ExprId, ref_fn: ExprId, a: &QPoly, x: ExprId, pool: &ExprPool) {
        let u = simplify(u, pool).value;
        let mut ratios = Vec::new();
        for &xv in &[0.31_f64, 1.27, 2.73, 3.61, 4.19, 5.53] {
            let av = eval_qp(a, xv);
            if av <= 1e-6 {
                continue;
            }
            let (Some(uu), Some(rr)) = (eval(u, x, xv, pool), eval(ref_fn, x, xv, pool)) else {
                continue;
            };
            if !uu.is_finite() || !rr.is_finite() || rr.abs() < 1e-9 {
                continue;
            }
            ratios.push(uu / rr);
        }
        assert!(ratios.len() >= 3, "too few sample points: {ratios:?}");
        let r0 = ratios[0];
        assert!(r0.abs() > 1e-9, "reference/u vanished");
        for r in &ratios {
            assert!(
                (r - r0).abs() < 1e-6 * (1.0 + r0.abs()),
                "u/ref not constant: {ratios:?}"
            );
        }
    }

    /// Asymmetric principal divisor `div(y − v)` on a genus-2 quintic.  Pick five
    /// rational roots and a linear `v = x + 1`, set `a = v² + Π(x − rᵢ)`.  Then
    /// `div(y − v) = Σ (rᵢ, v(rᵢ)) − 5∞`, and Coates must recover `u ∝ (y − v)`.
    #[test]
    fn recover_y_minus_v() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let roots = [-2_i64, -1, 3, 4, 5];
        let mut prod = qp(&[1]);
        for &r in &roots {
            prod = poly_mul(&prod, &qp(&[-r, 1]));
        }
        let v = qp(&[1, 1]); // x + 1
        let a = trim(poly_add(&poly_mul(&v, &v), &prod));
        assert_eq!(degree(&a), 5);
        let places: Vec<CoatesPlace> = roots
            .iter()
            .map(|&r| place(r, eval_qp(&v, r as f64) as i64, 1))
            .collect();
        let u = coates_hyperelliptic(&a, &places, x, &pool).expect("principal");
        let y = pool.func("sqrt", vec![qpoly_to_expr(&a, x, &pool)]);
        let ref_fn = pool.add(vec![
            y,
            pool.mul(vec![pool.integer(-1_i32), qpoly_to_expr(&v, x, &pool)]),
        ]);
        assert_constant_ratio(u, ref_fn, &a, x, &pool);
    }

    /// Anti-symmetric principal divisor `div((y − v)/(y + v))` (the pure
    /// logarithmic-derivative / pseudo-elliptic shape): zeros `(rᵢ, v(rᵢ))`,
    /// poles `(rᵢ, −v(rᵢ))`.  Coates must recover `u ∝ (y − v)/(y + v)`.
    #[test]
    fn recover_antisymmetric_unit() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let roots = [-2_i64, -1, 3, 4, 5];
        let mut prod = qp(&[1]);
        for &r in &roots {
            prod = poly_mul(&prod, &qp(&[-r, 1]));
        }
        let v = qp(&[1, 1]);
        let a = trim(poly_add(&poly_mul(&v, &v), &prod));
        let mut places: Vec<CoatesPlace> = Vec::new();
        for &r in &roots {
            let vr = eval_qp(&v, r as f64) as i64;
            places.push(place(r, vr, 1)); // zero
            places.push(place(r, -vr, -1)); // pole (opposite sheet)
        }
        let u = coates_hyperelliptic(&a, &places, x, &pool).expect("principal");
        let y = pool.func("sqrt", vec![qpoly_to_expr(&a, x, &pool)]);
        let vexpr = qpoly_to_expr(&v, x, &pool);
        let num = pool.add(vec![y, pool.mul(vec![pool.integer(-1_i32), vexpr])]);
        let den = pool.add(vec![y, vexpr]);
        let ref_fn = pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))]);
        assert_constant_ratio(u, ref_fn, &a, x, &pool);
    }

    /// Symmetric principal divisor `div((x − 1)/(x − 2))` (a rational function of
    /// `x`): both sheets over `x=1` are zeros, both over `x=2` poles.  On
    /// `y² = x⁵ + x + 2` (`a(1)=4`, `a(2)=36`).  Coates recovers `(x−1)/(x−2)`.
    #[test]
    fn recover_rational_x_function() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = qp(&[2, 1, 0, 0, 0, 1]); // x⁵ + x + 2
        assert_eq!(eval_qp(&a, 1.0), 4.0);
        assert_eq!(eval_qp(&a, 2.0), 36.0);
        let places = vec![
            place(1, 2, 1),
            place(1, -2, 1),
            place(2, 6, -1),
            place(2, -6, -1),
        ];
        let u = coates_hyperelliptic(&a, &places, x, &pool).expect("principal");
        let ref_fn = pool.mul(vec![
            pool.add(vec![x, pool.integer(-1_i32)]),
            pool.pow(
                pool.add(vec![x, pool.integer(-2_i32)]),
                pool.integer(-1_i32),
            ),
        ]);
        assert_constant_ratio(u, ref_fn, &a, x, &pool);
    }

    /// A non-principal divisor must return `None`: `(0,1) − ∞` alone on
    /// `y² = x⁵ + x + 1` is non-torsion, so no function has that divisor.
    #[test]
    fn non_principal_returns_none() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = qp(&[1, 1, 0, 0, 0, 1]); // x⁵ + x + 1
        let places = vec![place(0, 1, 1)];
        assert!(coates_hyperelliptic(&a, &places, x, &pool).is_none());
    }
}
