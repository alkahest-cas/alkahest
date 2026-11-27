//! A certified count of the eigenvalues below the threshold of a regular
//! constant-coefficient Sturm–Liouville problem with Robin (or Dirichlet /
//! Neumann) conditions.
//!
//! For `(pX')' + qX + λwX = 0` on `(a, b)` with `p, q, w` constant, `p, w > 0`,
//! and `α_a X(a) + β_a X'(a) = 0`, `α_b X(b) + β_b X'(b) = 0`, every eigenvalue
//! below `λ₀ = −q/w` is `λ = λ₀ − (p/w)κ²` with `κ > 0` and eigenfunction
//! (up to a factor) `X = β_a cosh κy − α_a sinh(κy)/κ`, `y = x − a` — the
//! solutions of `X'' = κ²X` meeting the left condition form a line, spanned
//! by this `X` (which is never zero: `X(a) = β_a`, `X'(a) = −α_a`).  So the
//! eigenvalues below `λ₀` are in bijection with the positive roots of the
//! characteristic function
//!
//! `G(κ) = α_b X(b) + β_b X'(b) = A cosh κL + (Bκ² + C) sinh(κL)/κ`,
//! `A = α_bβ_a − α_aβ_b`, `B = β_aβ_b`, `C = −α_aα_b`, `L = b − a`,
//!
//! an even entire function with `G(0) = A + CL` (zero exactly when `λ₀` is an
//! eigenvalue).  The identity between `G` and the boundary functional, the ODE
//! and the left condition are checked symbolically, identically in `κ`.
//!
//! The roots are then counted rigorously:
//!
//! * **Rayleigh quotient**: with `α_aβ_a ≤ 0 ≤ α_bβ_b` (exact signs) there is
//!   no eigenvalue below `λ₀` — no arithmetic needed.
//! * **Tail bound**: for `κ > K` the sign of `G` is that of `B` (of `A` when
//!   `B = 0`): with `κ coth κL ≤ κ + 1/L`,
//!   `κG/sinh κL = Aκ coth κL + Bκ² + C` has the sign of `B` once
//!   `|B|κ² − |A|κ − |A|/L − |C| > 0`; for `B = 0`,
//!   `|C| sinh(κL)/κ < |A| cosh κL` once `κ ≥ |C|/|A|`.
//! * **Interval cover of `[κ_start, K]`** in outward-rounded MPFR arithmetic:
//!   `cosh`, `sinh`, `S(z) = sinh z / z` and `S'(z)` are increasing on
//!   `z ≥ 0` (power series with positive coefficients), so their ranges over
//!   a cell are the values at its ends.  A cell is discarded when the range of
//!   `G` excludes 0; when the range of `G'` excludes 0, `G` is strictly
//!   monotone there and has exactly one root if and only if the signs at the
//!   ends (certified) differ; otherwise the cell is bisected.  A cell that
//!   stays undecided down to the width limit makes the count refuse.
//!
//! At most two eigenvalues lie at or below `λ₀` (min–max: the quadratic form
//! minus `λ₀∫wX²` is `∫pX'² > 0` on `H¹₀`, of codimension 2), which the count
//! is checked against.

use std::cmp::Ordering;
use std::collections::HashMap;

use rug::float::Round;
use rug::Float;

use crate::ball::IntervalEval;
use crate::kernel::subs::{mentions_var, subs};
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::super::error::PdeError;
use super::super::verify::VerificationStatus;
use super::{
    d, div, fresh_symbol, neg, sign, simp, sub, subs1, EigenEnclosure, EndCondition, Sgn,
    SlBoundary, SturmLiouvilleProblem,
};

/// The certified count of [`count_lower_eigenvalues`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct LowerCount {
    /// The number of eigenvalues strictly below the threshold `λ₀`.
    pub count: usize,
    /// `λ₀ = −q/w`.
    pub threshold: ExprId,
    /// Whether `λ₀` itself is an eigenvalue (decided exactly: `G(0) = 0`).
    pub threshold_is_eigenvalue: bool,
    /// The number of negative eigenvalues, when it follows: `count` for
    /// `λ₀ = 0`, the roots with `κ > √(λ₀w/p)` for `λ₀ > 0`; `None` for
    /// `λ₀ < 0` (the oscillatory eigenvalues in `[λ₀, 0)` are not counted
    /// here).
    pub negative: Option<usize>,
    /// Enclosures of the eigenvalues below `λ₀`, in increasing order (index
    /// `0` is the lowest eigenvalue).
    pub enclosures: Vec<EigenEnclosure>,
    /// The symbol `κ` of [`Self::characteristic`] and [`Self::eigenfunction`].
    pub kappa: ExprId,
    /// `G(κ)`, whose positive roots are the eigenvalues `λ₀ − (p/w)κ²`.
    pub characteristic: ExprId,
    /// `X(x; κ)`, satisfying the ODE with `λ = λ₀ − (p/w)κ²` and the left
    /// condition identically in `κ`.
    pub eigenfunction: ExprId,
    /// `"rayleigh"` (the sign hypothesis holds: no arithmetic) or
    /// `"interval_cover"`.
    pub method: &'static str,
    /// The tail bound `K` (no root for `κ > K`), for `"interval_cover"`.
    pub tail_bound: Option<Float>,
    /// The number of cells the cover used.
    pub cells: usize,
    /// The exact symbolic checks: `(what, status, detail)`.
    pub checks: Vec<(&'static str, ZeroStatus, String)>,
    /// `ExactlyVerified` when every symbolic check is a proof (the interval
    /// part is a proof by construction, or the function refuses).
    pub status: VerificationStatus,
    /// The certificate, step by step.
    pub certificate: Vec<String>,
}

// ---------------------------------------------------------------------------
// Outward-rounded interval arithmetic
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Iv {
    lo: Float,
    hi: Float,
}

impl Iv {
    fn point(v: &Float) -> Iv {
        Iv {
            lo: v.clone(),
            hi: v.clone(),
        }
    }

    fn excludes_zero(&self) -> bool {
        self.lo > 0 || self.hi < 0
    }

    fn sign(&self) -> Option<i32> {
        if self.lo > 0 {
            Some(1)
        } else if self.hi < 0 {
            Some(-1)
        } else {
            None
        }
    }

    fn add(&self, o: &Iv, prec: u32) -> Iv {
        Iv {
            lo: Float::with_val_round(prec, &self.lo + &o.lo, Round::Down).0,
            hi: Float::with_val_round(prec, &self.hi + &o.hi, Round::Up).0,
        }
    }

    fn mul(&self, o: &Iv, prec: u32) -> Iv {
        let ends = [
            (&self.lo, &o.lo),
            (&self.lo, &o.hi),
            (&self.hi, &o.lo),
            (&self.hi, &o.hi),
        ];
        let mut lo: Option<Float> = None;
        let mut hi: Option<Float> = None;
        for (a, b) in ends {
            let d = Float::with_val_round(prec, a * b, Round::Down).0;
            let u = Float::with_val_round(prec, a * b, Round::Up).0;
            lo = Some(match lo {
                Some(l) if l <= d => l,
                _ => d,
            });
            hi = Some(match hi {
                Some(h) if h >= u => h,
                _ => u,
            });
        }
        Iv {
            lo: lo.expect("four products"),
            hi: hi.expect("four products"),
        }
    }

    fn abs_hi(&self) -> Float {
        let a = Float::with_val(self.lo.prec(), self.lo.abs_ref());
        let b = Float::with_val(self.hi.prec(), self.hi.abs_ref());
        if a > b {
            a
        } else {
            b
        }
    }

    /// A lower bound of `|v|` over the interval (0 when it contains 0).
    fn abs_lo(&self) -> Float {
        if self.lo > 0 {
            self.lo.clone()
        } else if self.hi < 0 {
            Float::with_val(self.hi.prec(), -&self.hi)
        } else {
            Float::new(self.lo.prec())
        }
    }
}

/// `f(z)` rounded down and up for `z ≥ 0` (exact input).
fn cosh_bounds(z: &Float, prec: u32) -> (Float, Float) {
    (
        Float::with_val_round(prec, z.cosh_ref(), Round::Down).0,
        Float::with_val_round(prec, z.cosh_ref(), Round::Up).0,
    )
}

fn sinh_bounds(z: &Float, prec: u32) -> (Float, Float) {
    (
        Float::with_val_round(prec, z.sinh_ref(), Round::Down).0,
        Float::with_val_round(prec, z.sinh_ref(), Round::Up).0,
    )
}

/// `S(z) = sinh z / z` (`S(0) = 1`), bounds at an exact `z ≥ 0`.
fn s_bounds(z: &Float, prec: u32) -> (Float, Float) {
    if *z == 0 {
        let one = Float::with_val(prec, 1);
        return (one.clone(), one);
    }
    let (sd, su) = sinh_bounds(z, prec);
    (
        Float::with_val_round(prec, &sd / z, Round::Down).0,
        Float::with_val_round(prec, &su / z, Round::Up).0,
    )
}

/// `S'(z) = (z cosh z − sinh z)/z²` (`S'(0) = 0`, `S' ≥ 0`), bounds at an
/// exact `z ≥ 0`.
fn sp_bounds(z: &Float, prec: u32) -> (Float, Float) {
    let zero = Float::new(prec);
    if *z == 0 {
        return (zero.clone(), zero);
    }
    let (cd, cu) = cosh_bounds(z, prec);
    let (sd, su) = sinh_bounds(z, prec);
    let z2d = Float::with_val_round(prec, z * z, Round::Down).0;
    let z2u = Float::with_val_round(prec, z * z, Round::Up).0;
    let zc_d = Float::with_val_round(prec, z * &cd, Round::Down).0;
    let zc_u = Float::with_val_round(prec, z * &cu, Round::Up).0;
    let num_d = Float::with_val_round(prec, &zc_d - &su, Round::Down).0;
    let num_u = Float::with_val_round(prec, &zc_u - &sd, Round::Up).0;
    let lo = if num_d > 0 {
        Float::with_val_round(prec, &num_d / &z2u, Round::Down).0
    } else {
        zero
    };
    let hi = if num_u > 0 {
        Float::with_val_round(prec, &num_u / &z2d, Round::Up).0
    } else {
        Float::new(prec)
    };
    (lo, hi)
}

/// The data of `G(κ) = A cosh κL + (Bκ² + C) L S(κL)`.
struct Char {
    a: Iv,
    b: Iv,
    c: Iv,
    l: Iv,
    prec: u32,
}

impl Char {
    /// `[zlo, zhi] ⊇ κL` for `κ ∈ k` (`k.lo ≥ 0`).
    fn z(&self, k: &Iv) -> (Float, Float) {
        (
            Float::with_val_round(self.prec, &k.lo * &self.l.lo, Round::Down).0,
            Float::with_val_round(self.prec, &k.hi * &self.l.hi, Round::Up).0,
        )
    }

    fn mono(&self, f: fn(&Float, u32) -> (Float, Float), k: &Iv) -> Iv {
        let (zlo, zhi) = self.z(k);
        Iv {
            lo: f(&zlo, self.prec).0,
            hi: f(&zhi, self.prec).1,
        }
    }

    fn k2(&self, k: &Iv) -> Iv {
        Iv {
            lo: Float::with_val_round(self.prec, &k.lo * &k.lo, Round::Down).0,
            hi: Float::with_val_round(self.prec, &k.hi * &k.hi, Round::Up).0,
        }
    }

    /// A range enclosure of `G` over `k`.
    fn g(&self, k: &Iv) -> Iv {
        let p = self.prec;
        let ch = self.mono(cosh_bounds, k);
        let s = self.mono(s_bounds, k);
        let poly = self.b.mul(&self.k2(k), p).add(&self.c, p);
        self.a.mul(&ch, p).add(&poly.mul(&self.l, p).mul(&s, p), p)
    }

    /// A range enclosure of `G'` over `k`:
    /// `A L sinh κL + 2Bκ L S(κL) + (Bκ² + C) L² S'(κL)`.
    fn gp(&self, k: &Iv) -> Iv {
        let p = self.prec;
        let sh = self.mono(sinh_bounds, k);
        let s = self.mono(s_bounds, k);
        let sp = self.mono(sp_bounds, k);
        let two = Iv::point(&Float::with_val(p, 2));
        let t1 = self.a.mul(&self.l, p).mul(&sh, p);
        let t2 = two.mul(&self.b, p).mul(k, p).mul(&self.l, p).mul(&s, p);
        let poly = self.b.mul(&self.k2(k), p).add(&self.c, p);
        let t3 = poly.mul(&self.l, p).mul(&self.l, p).mul(&sp, p);
        t1.add(&t2, p).add(&t3, p)
    }
}

fn mid(a: &Float, b: &Float, prec: u32) -> Float {
    Float::with_val(prec, a + b) / 2u32
}

/// Count the roots of `G` in `(start.hi, ∞)` — `G` must not vanish on
/// `start` — returning root enclosures in increasing `κ`.
fn count_roots(
    ch: &Char,
    start: &Iv,
    tail: &Float,
    max_cells: usize,
) -> Result<(Vec<(Float, Float)>, usize), String> {
    let p = ch.prec;
    if !ch.g(start).excludes_zero() {
        return Err(
            "G does not have a certified sign at the start of the range \
                    (a root at the threshold or at κ*)"
                .into(),
        );
    }
    if *tail <= start.hi {
        return Ok((Vec::new(), 1));
    }
    let width_min = Float::with_val(p, tail) >> 60u32;
    let mut stack = vec![Iv {
        lo: start.hi.clone(),
        hi: tail.clone(),
    }];
    let mut roots = Vec::new();
    let mut cells = 0usize;
    while let Some(cell) = stack.pop() {
        cells += 1;
        if cells > max_cells {
            return Err(format!("more than {max_cells} cells"));
        }
        if ch.g(&cell).excludes_zero() {
            continue;
        }
        if ch.gp(&cell).excludes_zero() {
            let sl = ch.g(&Iv::point(&cell.lo)).sign();
            let sh = ch.g(&Iv::point(&cell.hi)).sign();
            match (sl, sh) {
                (Some(a), Some(b)) => {
                    if a != b {
                        // Refine the single root by bisection.
                        let (mut lo, mut hi) = (cell.lo.clone(), cell.hi.clone());
                        for _ in 0..200 {
                            let w = Float::with_val(p, &hi - &lo);
                            if w <= Float::with_val(p, &hi) >> 50u32 {
                                break;
                            }
                            let m = mid(&lo, &hi, p);
                            match ch.g(&Iv::point(&m)).sign() {
                                Some(s) if s == a => lo = m,
                                Some(_) => hi = m,
                                None => break,
                            }
                        }
                        roots.push((lo, hi));
                    }
                    continue;
                }
                _ => {
                    return Err("a root sits on a cell boundary to working precision".into());
                }
            }
        }
        let w = Float::with_val(p, &cell.hi - &cell.lo);
        if w < width_min {
            return Err(format!(
                "a cell near κ ≈ {:.6} stays undecided (a double root or a tangency)",
                cell.lo.to_f64()
            ));
        }
        let m = mid(&cell.lo, &cell.hi, p);
        // Push the right half first so the left is processed first.
        stack.push(Iv {
            lo: m.clone(),
            hi: cell.hi.clone(),
        });
        stack.push(Iv { lo: cell.lo, hi: m });
    }
    roots.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    Ok((roots, cells))
}

fn ball(e: ExprId, pool: &ExprPool, what: &str, prec: u32) -> Result<Iv, PdeError> {
    let b = IntervalEval::new(prec).eval(e, pool).ok_or_else(|| {
        PdeError::SlCountNotCertified(format!(
            "{what} = {} has no numeric enclosure (substitute numbers for the parameters)",
            pool.display(e)
        ))
    })?;
    let (lo, hi) = (b.lo(), b.hi());
    if !(lo.is_finite() && hi.is_finite()) {
        return Err(PdeError::SlCountNotCertified(format!(
            "{what}: non-finite enclosure"
        )));
    }
    Ok(Iv { lo, hi })
}

fn coefficients(c: EndCondition, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    match c {
        EndCondition::Dirichlet => Some((pool.integer(1), pool.integer(0))),
        EndCondition::Neumann => Some((pool.integer(0), pool.integer(1))),
        EndCondition::Robin { alpha, beta } => Some((alpha, beta)),
        _ => None,
    }
}

/// A certified count of the eigenvalues below the threshold `λ₀ = −q/w` (and
/// of the negative eigenvalues, when that follows) of a regular
/// Sturm–Liouville problem with constant `p`, `q`, `w` and separated
/// Dirichlet / Neumann / Robin conditions, after substituting `params`
/// (parameter → number).  The certificate: the eigenvalues below `λ₀` are
/// the positive roots of `G(κ) = A cosh κL + (Bκ² + C) sinh(κL)/κ` (checked
/// symbolically), counted by the Rayleigh test or an outward-rounded interval
/// cover of `[0, K]` with a proved tail bound `K`.
///
/// The count is exact or the function refuses: a numeric check never
/// certifies on its own, every bound is outward rounded, and the symbolic
/// identities that tie the counted function to the eigenproblem are proved.
///
/// # Errors
///
/// `E-PDE-259` — a symbolic parameter remains, the problem is not regular
/// with constant coefficients, `p` or `w` is not positive, a root cannot be
/// separated (a double root, a root at the threshold), or a symbolic check is
/// not a proof.
pub fn count_lower_eigenvalues(
    problem: &SturmLiouvilleProblem,
    params: &HashMap<ExprId, ExprId>,
    pool: &ExprPool,
) -> Result<LowerCount, PdeError> {
    let prec = 192u32;
    let x = problem.var;
    let sb = |e: ExprId| simp(subs(e, params, pool), pool);
    let (p, q, w, a, b) = (
        sb(problem.p),
        sb(problem.q),
        sb(problem.w),
        sb(problem.a),
        sb(problem.b),
    );
    for (what, e) in [("p", p), ("q", q), ("w", w)] {
        if mentions_var(e, x, pool) {
            return Err(PdeError::SlCountNotCertified(format!(
                "{what} = {} depends on {}: the certified count is for constant coefficients",
                pool.display(e),
                pool.display(x)
            )));
        }
    }
    let SlBoundary::Separated { left, right } = problem.boundary else {
        return Err(PdeError::SlCountNotCertified(
            "periodic conditions: the certified count is for separated conditions".into(),
        ));
    };
    let ((aa, ba), (ab, bb)) = match (coefficients(left, pool), coefficients(right, pool)) {
        (Some(l), Some(r)) => ((sb(l.0), sb(l.1)), (sb(r.0), sb(r.1))),
        _ => {
            return Err(PdeError::SlCountNotCertified(
                "a singular end condition: the certified count is for regular ends".into(),
            ))
        }
    };
    let pi_v = ball(p, pool, "p", prec)?;
    let wi_v = ball(w, pool, "w", prec)?;
    if !(pi_v.lo > 0 && wi_v.lo > 0) {
        return Err(PdeError::SlCountNotCertified(
            "p and w must be certified positive".into(),
        ));
    }
    let len = simp(sub(b, a, pool), pool);
    let l_iv = ball(len, pool, "b − a", prec)?;
    if l_iv.lo <= 0 {
        return Err(PdeError::SlCountNotCertified(
            "the interval length must be certified positive".into(),
        ));
    }
    let lam0 = simp(neg(div(q, w, pool), pool), pool);
    let a_c = simp(
        sub(pool.mul(vec![ab, ba]), pool.mul(vec![aa, bb]), pool),
        pool,
    );
    let b_c = simp(pool.mul(vec![ba, bb]), pool);
    let c_c = simp(neg(pool.mul(vec![aa, ab]), pool), pool);
    // Rayleigh: α_aβ_a ≤ 0 ≤ α_bβ_b gives none below λ₀.
    let sa = sign(simp(pool.mul(vec![aa, ba]), pool), pool);
    let sbb = sign(simp(pool.mul(vec![ab, bb]), pool), pool);
    let rayleigh = matches!(sa, Sgn::Neg | Sgn::Zero) && matches!(sbb, Sgn::Pos | Sgn::Zero);
    // Numeric data first (cheap): a symbolic parameter refuses here.
    let abc_iv = if rayleigh {
        None
    } else {
        Some((
            ball(a_c, pool, "A", prec)?,
            ball(b_c, pool, "B", prec)?,
            ball(c_c, pool, "C", prec)?,
        ))
    };
    // Symbolic part: X, G, and their identities in κ.
    let avoid = [p, q, w, a, b, x, aa, ba, ab, bb];
    let kappa = fresh_symbol("kappa", Domain::Positive, &avoid, pool);
    let yv = sub(x, a, pool);
    let ky = pool.mul(vec![kappa, yv]);
    let xf = simp(
        sub(
            pool.mul(vec![ba, pool.func("cosh", vec![ky])]),
            div(pool.mul(vec![aa, pool.func("sinh", vec![ky])]), kappa, pool),
            pool,
        ),
        pool,
    );
    let lam = sub(
        lam0,
        pool.mul(vec![div(p, w, pool), pool.pow(kappa, pool.integer(2))]),
        pool,
    );
    let dx = |e: ExprId| d(e, x, pool);
    let ode = pool.add(vec![
        dx(pool.mul(vec![p, dx(xf)?]))?,
        pool.mul(vec![q, xf]),
        pool.mul(vec![lam, w, xf]),
    ]);
    let zs = |e: ExprId| crate::pde::green::zstatus(e, pool);
    let mut checks = Vec::new();
    checks.push((
        "ode",
        zs(ode),
        "(pX')' + qX + λwX = 0 with λ = λ₀ − (p/w)κ², identically in κ".to_string(),
    ));
    let left_res = pool.add(vec![
        pool.mul(vec![aa, subs1(xf, x, a, pool)]),
        pool.mul(vec![ba, subs1(dx(xf)?, x, a, pool)]),
    ]);
    checks.push((
        "left",
        zs(left_res),
        "α_a X(a) + β_a X'(a) = 0 identically in κ".to_string(),
    ));
    let kl = pool.mul(vec![kappa, len]);
    let g_sym = pool.add(vec![
        pool.mul(vec![a_c, pool.func("cosh", vec![kl])]),
        div(
            pool.mul(vec![
                pool.add(vec![
                    pool.mul(vec![b_c, pool.pow(kappa, pool.integer(2))]),
                    c_c,
                ]),
                pool.func("sinh", vec![kl]),
            ]),
            kappa,
            pool,
        ),
    ]);
    let right_res = sub(
        pool.add(vec![
            pool.mul(vec![ab, subs1(xf, x, b, pool)]),
            pool.mul(vec![bb, subs1(dx(xf)?, x, b, pool)]),
        ]),
        g_sym,
        pool,
    );
    let mut rs = zs(right_res);
    if rs != ZeroStatus::ProvedZero {
        rs = zs(crate::pde::green::expand_func_args(right_res, pool));
    }
    checks.push((
        "characteristic",
        rs,
        "α_b X(b) + β_b X'(b) = A cosh κL + (Bκ² + C) sinh(κL)/κ identically in κ".to_string(),
    ));
    if let Some((what, st, _)) = checks.iter().find(|c| c.1 != ZeroStatus::ProvedZero) {
        return Err(PdeError::SlCountNotCertified(format!(
            "the symbolic check `{what}` is {}",
            st.as_str()
        )));
    }
    // The threshold: λ₀ is an eigenvalue iff G(0) = A + CL = 0.
    let g0 = simp(pool.add(vec![a_c, pool.mul(vec![c_c, len])]), pool);
    let g0_status = zs(g0);
    let threshold_is_eigenvalue = match g0_status {
        ZeroStatus::ProvedZero => true,
        ZeroStatus::ProvedNonzero => false,
        _ => {
            return Err(PdeError::SlCountNotCertified(format!(
                "whether λ₀ is an eigenvalue (G(0) = {} = 0) was not decided",
                pool.display(g0)
            )))
        }
    };
    let mut certificate = vec![
        format!(
            "below λ₀ = {}: λ = λ₀ − (p/w)κ², X = {} spans the solutions meeting the left \
             condition (checked: ODE and left condition identically in κ)",
            pool.display(lam0),
            pool.display(xf)
        ),
        format!(
            "eigenvalues below λ₀ ↔ positive roots of G(κ) = {} (checked identically in κ)",
            pool.display(simp(g_sym, pool))
        ),
    ];
    let lam0_iv = ball(lam0, pool, "λ₀", prec)?;
    let ratio = div(p, w, pool);
    let ratio_iv = ball(ratio, pool, "p/w", prec)?;
    let (roots, cells, tail_bound, method) = if rayleigh {
        certificate.push(
            "Rayleigh quotient: λ∫wX² = ∫pX'² − ∫qX² + p(b)(α_b/β_b)X(b)² − \
             p(a)(α_a/β_a)X(a)², and α_aβ_a ≤ 0 ≤ α_bβ_b (exact signs), so λ ≥ λ₀"
                .to_string(),
        );
        (Vec::new(), 0usize, None, "rayleigh")
    } else {
        let (a_iv, b_iv, c_iv) = abc_iv.expect("numeric when the Rayleigh test fails");
        let one = Float::with_val(prec, 1);
        let inv_l_up = Float::with_val_round(prec, &one / &l_iv.lo, Round::Up).0;
        let tail = if b_iv.excludes_zero() {
            // K = (|A| + √(A² + 4|B|(|C| + |A|/L)))/(2|B|), rounded up.
            let abs_a = a_iv.abs_hi();
            let abs_b = b_iv.abs_lo();
            let abs_c = c_iv.abs_hi();
            let t = Float::with_val_round(prec, &abs_a * &inv_l_up, Round::Up).0;
            let t = Float::with_val_round(prec, &t + &abs_c, Round::Up).0;
            let t = Float::with_val_round(prec, &t * &abs_b, Round::Up).0;
            let t = Float::with_val_round(prec, &t * 4u32, Round::Up).0;
            let a2 = Float::with_val_round(prec, &abs_a * &abs_a, Round::Up).0;
            let disc = Float::with_val_round(prec, &a2 + &t, Round::Up).0;
            let root = Float::with_val_round(prec, disc.sqrt_ref(), Round::Up).0;
            let num = Float::with_val_round(prec, &abs_a + &root, Round::Up).0;
            let den = Float::with_val_round(prec, &abs_b * 2u32, Round::Down).0;
            Float::with_val_round(prec, &num / &den, Round::Up).0
        } else if zs(b_c) == ZeroStatus::ProvedZero {
            if a_iv.excludes_zero() {
                let (num, den) = (c_iv.abs_hi(), a_iv.abs_lo());
                Float::with_val_round(prec, &num / &den, Round::Up).0
            } else if zs(a_c) == ZeroStatus::ProvedZero {
                // G = C sinh(κL)/κ with C ≠ 0: no root.
                Float::new(prec)
            } else {
                return Err(PdeError::SlCountNotCertified(
                    "the leading coefficient A of G could not be separated from 0".into(),
                ));
            }
        } else {
            return Err(PdeError::SlCountNotCertified(
                "the leading coefficient B = β_aβ_b of G could not be separated from 0".into(),
            ));
        };
        // Slightly enlarge the tail bound (the bound is strict beyond K).
        let tail = Float::with_val_round(prec, &tail * 1.000_001f64, Round::Up).0
            + Float::with_val(prec, 1e-30);
        let tail = Float::with_val(prec, tail);
        let ch = Char {
            a: a_iv,
            b: b_iv,
            c: c_iv,
            l: l_iv.clone(),
            prec,
        };
        if threshold_is_eigenvalue {
            return Err(PdeError::SlCountNotCertified(
                "λ₀ is an eigenvalue and the Rayleigh hypothesis fails: G has a double root at \
                 κ = 0 that the interval cover does not separate"
                    .into(),
            ));
        }
        let zero = Iv::point(&Float::new(prec));
        let (roots, cells) =
            count_roots(&ch, &zero, &tail, 400_000).map_err(PdeError::SlCountNotCertified)?;
        certificate.push(format!(
            "tail bound: G has the sign of {} for κ > K = {:.6e}",
            if ch.b.excludes_zero() { "B" } else { "A" },
            tail.to_f64()
        ));
        certificate.push(format!(
            "interval cover of [0, K] with {cells} cells (outward-rounded MPFR, {prec} bits): \
             {} sign change(s), each in a cell where G' ≠ 0",
            roots.len()
        ));
        if roots.len() > 2 {
            return Err(PdeError::SlVerificationFailed(format!(
                "{} roots below the threshold contradict the min–max bound of 2",
                roots.len()
            )));
        }
        (roots, cells, Some(tail), "interval_cover")
    };
    let count = roots.len();
    if count + usize::from(threshold_is_eigenvalue) > 2 {
        return Err(PdeError::SlVerificationFailed(
            "more than two eigenvalues at or below the threshold".into(),
        ));
    }
    // Enclosures λ = λ₀ − (p/w)κ², increasing λ ↔ decreasing κ.
    let mut enclosures = Vec::new();
    for (i, (klo, khi)) in roots.iter().rev().enumerate() {
        let k2 = Iv {
            lo: Float::with_val_round(prec, klo * klo, Round::Down).0,
            hi: Float::with_val_round(prec, khi * khi, Round::Up).0,
        };
        let t = ratio_iv.mul(&k2, prec);
        let neg_t = Iv {
            lo: Float::with_val(prec, -&t.hi),
            hi: Float::with_val(prec, -&t.lo),
        };
        let l = lam0_iv.add(&neg_t, prec);
        enclosures.push(EigenEnclosure {
            index: i as i64,
            lo: l.lo,
            hi: l.hi,
        });
    }
    // Negative eigenvalues.
    let negative = if zs(lam0) == ZeroStatus::ProvedZero {
        Some(count)
    } else if lam0_iv.lo > 0 {
        // κ > κ* = √(λ₀ w/p): every eigenvalue in [λ₀ ...) is positive; count
        // the enclosures certified below 0.
        let below: Vec<&EigenEnclosure> = enclosures.iter().filter(|e| e.hi < 0).collect();
        let above = enclosures.iter().filter(|e| e.lo > 0).count();
        if below.len() + above == enclosures.len() {
            Some(below.len())
        } else {
            None
        }
    } else {
        None
    };
    Ok(LowerCount {
        count,
        threshold: lam0,
        threshold_is_eigenvalue,
        negative,
        enclosures,
        kappa,
        characteristic: simp(g_sym, pool),
        eigenfunction: xf,
        method,
        tail_bound,
        cells,
        checks,
        status: VerificationStatus::ExactlyVerified,
        certificate,
    })
}
