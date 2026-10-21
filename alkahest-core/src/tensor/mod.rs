//! Tensor calculus on a coordinate chart: metric → Christoffel symbols →
//! Riemann, Ricci, Einstein, Weyl and the Kretschmann scalar (F27).
//!
//! A [`Metric`] is built from an `n × n` symmetric matrix of expressions and
//! `n` coordinate symbols.  Entries may contain parameters, trigonometric and
//! hyperbolic functions, `exp`, `log`, radicals, and **undefined functions of
//! the coordinates** such as the FLRW scale factor `a(t)` (their derivatives
//! become `Derivative` nodes, see [`crate::diff::undefined`]).
//!
//! # Exactness
//!
//! Every component is computed in an exact differential field (`field.rs`):
//! a quotient of FLINT integer polynomials over the coordinates, the
//! parameters and one generator per transcendental atom, with `sin² = 1 − cos²`
//! (and `sinh² = cosh² − 1`, `i² = −1`, `(√b)² = b`) applied after every
//! operation and every denominator made free of `sin`/`√`/`i`.  That form is
//! canonical, so a component that is zero *as a function* comes out as the
//! literal `0` — this is how [`Metric::ricci`] proves the Kerr metric
//! Ricci-flat — and a non-zero component is never reported as zero.
//! Nothing is sampled and nothing is a float.
//!
//! # Conventions
//!
//! * `Γ^a_{bc} = ½ g^{ad} (∂_b g_{dc} + ∂_c g_{db} − ∂_d g_{bc})`;
//!   `Γ_{abc} = g_{ad} Γ^d_{bc}` (first kind, first index lowered).
//! * `R^a_{bcd} = ∂_c Γ^a_{db} − ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} − Γ^a_{de} Γ^e_{cb}`
//!   (MTW / Carroll / Wald; SymPy's `einsteinpy` agrees).
//! * `R_{bd} = R^a_{bad}`, `R = g^{bd} R_{bd}`, `G_{ab} = R_{ab} − ½ R g_{ab}`.
//! * `C_{abcd}` is the trace-free part of `R_{abcd}` (`n ≥ 3`).
//! * `K = R_{abcd} R^{abcd}`.
//!
//! With these signs the 2-sphere of radius `ρ` has `R = 2/ρ²` and
//! Schwarzschild has `K = 48 M²/r⁶`.
//!
//! # Work done
//!
//! Only independent components are computed: `Γ_{abc}` for `b ≤ c`,
//! `R_{abcd}` for `a < b`, `c < d`, `(a, b) ≤ (c, d)`, `R_{ab}` for `a ≤ b`;
//! the rest are filled in by symmetry.  Each stage is cached on the
//! [`Metric`], and [`Metric::timings`] reports the wall time of each.

// Slot arithmetic reads clearest as explicit index loops.
#![allow(clippy::needless_range_loop)]

mod field;
#[cfg(test)]
mod tests;

use crate::budget::BudgetTrip;
use crate::kernel::{ExprData, ExprId, ExprPool};
use field::{DiffField, FieldBuilder, Rf};
use std::fmt;
use std::time::Instant;

/// How many derivatives of an undefined function are prepared up front.
/// Curvature needs two; the third leaves room for a Bianchi check.
const ATOM_DERIVATIVE_DEPTH: usize = 3;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a tensor computation was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorError {
    /// The metric is not square, does not match the number of coordinates,
    /// or a coordinate is not a distinct symbol (`E-TENSOR-001`).
    Shape(String),
    /// `g_{ij} ≠ g_{ji}` (`E-TENSOR-002`).
    NotSymmetric {
        /// Row index.
        i: usize,
        /// Column index.
        j: usize,
    },
    /// The metric determinant is identically zero (`E-TENSOR-003`).
    Degenerate,
    /// A floating-point number appeared in an entry (`E-TENSOR-004`): the
    /// layer is exact, and a float would make every "exact" zero a guess.
    InexactInput {
        /// Rendered form of the offending node.
        expr: String,
    },
    /// An entry contains a node the field cannot represent (`Piecewise`,
    /// a predicate, …) (`E-TENSOR-005`).
    UnsupportedExpression {
        /// Rendered form of the offending node.
        expr: String,
    },
    /// A partial derivative needed by the computation is not available
    /// (`E-TENSOR-006`).
    Differentiation {
        /// What could not be differentiated, and why.
        expr: String,
    },
    /// A denominator vanished identically (`E-TENSOR-007`).
    DivisionByZero,
    /// The request needs more dimensions than the chart has (the Weyl
    /// tensor is defined for `n ≥ 3`) (`E-TENSOR-008`).
    Dimension {
        /// What was asked for.
        what: &'static str,
        /// Minimum dimension.
        min: usize,
        /// The chart's dimension.
        got: usize,
    },
    /// The active budget tripped; the code is the budget's own `E-BUDGET-*`.
    Budget(BudgetTrip),
    /// An internal invariant failed; refused rather than answered
    /// (`E-TENSOR-010`).
    Internal(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::Shape(msg) => write!(f, "{msg}"),
            TensorError::NotSymmetric { i, j } => write!(
                f,
                "the metric is not symmetric: g[{i}][{j}] - g[{j}][{i}] is not identically zero"
            ),
            TensorError::Degenerate => {
                write!(f, "the metric determinant is identically zero")
            }
            TensorError::InexactInput { expr } => write!(
                f,
                "`{expr}` is a floating-point number; tensor components are computed exactly"
            ),
            TensorError::UnsupportedExpression { expr } => {
                write!(f, "`{expr}` cannot be represented in the component field")
            }
            TensorError::Differentiation { expr } => {
                write!(f, "a needed partial derivative is unavailable: {expr}")
            }
            TensorError::DivisionByZero => {
                write!(f, "a denominator vanished identically")
            }
            TensorError::Dimension { what, min, got } => write!(
                f,
                "the {what} needs dimension at least {min}; the chart has {got}"
            ),
            TensorError::Budget(trip) => write!(f, "{trip}"),
            TensorError::Internal(msg) => write!(f, "internal error in the tensor layer: {msg}"),
        }
    }
}

impl std::error::Error for TensorError {}

impl crate::errors::AlkahestError for TensorError {
    fn code(&self) -> &'static str {
        match self {
            TensorError::Shape(_) => "E-TENSOR-001",
            TensorError::NotSymmetric { .. } => "E-TENSOR-002",
            TensorError::Degenerate => "E-TENSOR-003",
            TensorError::InexactInput { .. } => "E-TENSOR-004",
            TensorError::UnsupportedExpression { .. } => "E-TENSOR-005",
            TensorError::Differentiation { .. } => "E-TENSOR-006",
            TensorError::DivisionByZero => "E-TENSOR-007",
            TensorError::Dimension { .. } => "E-TENSOR-008",
            TensorError::Budget(trip) => crate::errors::AlkahestError::code(trip),
            TensorError::Internal(_) => "E-TENSOR-010",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        if let TensorError::Budget(trip) = self {
            return crate::errors::AlkahestError::remediation(trip);
        }
        crate::errors::codes::REGISTRY
            .iter()
            .find(|spec| spec.code == crate::errors::AlkahestError::code(self))
            .and_then(|spec| spec.remediation)
    }
}

impl TensorError {
    /// `true` when this is a budget or cancellation trip.
    pub fn is_budget(&self) -> bool {
        matches!(self, TensorError::Budget(_))
    }

    /// The budget trip, if this is one.
    pub fn budget_trip(&self) -> Option<BudgetTrip> {
        match self {
            TensorError::Budget(t) => Some(*t),
            _ => None,
        }
    }
}

fn budget() -> Result<(), TensorError> {
    crate::budget::check_all().map_err(TensorError::Budget)
}

// ---------------------------------------------------------------------------
// Metric
// ---------------------------------------------------------------------------

type T2 = Vec<Vec<Rf>>;
type T3 = Vec<Vec<Vec<Rf>>>;
type T4 = Vec<Vec<Vec<Vec<Rf>>>>;

/// A metric on a coordinate chart, with its curvature computed on demand.
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::tensor::Metric;
/// let pool = ExprPool::new();
/// let th = pool.symbol("theta", Domain::Real);
/// let ph = pool.symbol("phi", Domain::Real);
/// let r = pool.symbol("rho", Domain::Positive);
/// let z = pool.integer(0);
/// // the round 2-sphere of radius ρ
/// let r2 = pool.pow(r, pool.integer(2));
/// let s2 = pool.pow(pool.func("sin", vec![th]), pool.integer(2));
/// let g = vec![vec![r2, z], vec![z, pool.mul(vec![r2, s2])]];
/// let mut m = Metric::new(&g, &[th, ph], &pool).unwrap();
/// let scalar = m.ricci_scalar(&pool).unwrap();
/// assert_eq!(pool.display(scalar).to_string(), "(2 * rho^-2)");
/// ```
pub struct Metric {
    field: DiffField,
    coords: Vec<ExprId>,
    n: usize,
    g: T2,
    det: Option<Rf>,
    ginv: Option<T2>,
    dg: Option<T3>,
    gamma1: Option<T3>,
    gamma2: Option<T3>,
    riemann_lower: Option<T4>,
    riemann_mixed: Option<T4>,
    ricci: Option<T2>,
    scalar: Option<Rf>,
    timings: Vec<(&'static str, f64)>,
}

impl fmt::Debug for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Metric").field("dim", &self.n).finish()
    }
}

fn check_coords(coords: &[ExprId], pool: &ExprPool) -> Result<(), TensorError> {
    for (i, &c) in coords.iter().enumerate() {
        if !matches!(pool.get(c), ExprData::Symbol { .. }) {
            return Err(TensorError::Shape(format!(
                "coordinate {i} (`{}`) is not a symbol",
                pool.display(c)
            )));
        }
        if coords[..i].contains(&c) {
            return Err(TensorError::Shape(format!(
                "coordinate `{}` appears twice",
                pool.display(c)
            )));
        }
    }
    Ok(())
}

impl Metric {
    /// Build a metric from its components `g[i][j]` on the chart `coords`.
    ///
    /// # Errors
    ///
    /// `E-TENSOR-001` (shape / coordinates), `E-TENSOR-002` (not symmetric),
    /// `E-TENSOR-003` (zero determinant), `E-TENSOR-004` (a float),
    /// `E-TENSOR-005` (an unrepresentable node).
    pub fn new(g: &[Vec<ExprId>], coords: &[ExprId], pool: &ExprPool) -> Result<Self, TensorError> {
        let t0 = Instant::now();
        let n = coords.len();
        if n == 0 {
            return Err(TensorError::Shape(
                "the chart has no coordinates".to_string(),
            ));
        }
        if g.len() != n || g.iter().any(|row| row.len() != n) {
            return Err(TensorError::Shape(format!(
                "the metric must be {n} x {n} to match {n} coordinates"
            )));
        }
        check_coords(coords, pool)?;
        let mut b = FieldBuilder::new(coords, pool);
        for row in g {
            for &e in row {
                b.scan(e, ATOM_DERIVATIVE_DEPTH)?;
            }
        }
        let mut field = b.build()?;
        let mut gm: T2 = Vec::with_capacity(n);
        for row in g {
            let mut r = Vec::with_capacity(n);
            for &e in row {
                r.push(field.convert(e, pool)?);
            }
            gm.push(r);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if !field.sub(&gm[i][j], &gm[j][i])?.is_zero() {
                    return Err(TensorError::NotSymmetric { i, j });
                }
            }
        }
        let mut m = Metric {
            field,
            coords: coords.to_vec(),
            n,
            g: gm,
            det: None,
            ginv: None,
            dg: None,
            gamma1: None,
            gamma2: None,
            riemann_lower: None,
            riemann_mixed: None,
            ricci: None,
            scalar: None,
            timings: Vec::new(),
        };
        m.timings.push(("convert", t0.elapsed().as_secs_f64()));
        m.ensure_inverse()?;
        Ok(m)
    }

    /// The chart dimension `n`.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// The coordinates.
    pub fn coords(&self) -> &[ExprId] {
        &self.coords
    }

    /// Wall time (seconds) of every stage computed so far, in order.
    pub fn timings(&self) -> Vec<(&'static str, f64)> {
        self.timings.clone()
    }

    // ----- stages -----------------------------------------------------------

    fn ensure_inverse(&mut self) -> Result<(), TensorError> {
        if self.ginv.is_some() {
            return Ok(());
        }
        let t0 = Instant::now();
        let f = &self.field;
        let n = self.n;
        // Gauss–Jordan on [g | I] over the field.
        let mut a: T2 = self.g.clone();
        let mut inv: T2 = (0..n)
            .map(|i| (0..n).map(|j| f.int(i64::from(i == j))).collect())
            .collect();
        let mut det = f.int(1);
        for col in 0..n {
            budget()?;
            let piv = (col..n).find(|&r| !a[r][col].is_zero());
            let Some(p) = piv else {
                return Err(TensorError::Degenerate);
            };
            if p != col {
                a.swap(p, col);
                inv.swap(p, col);
                det = f.neg(&det);
            }
            let pv = a[col][col].clone();
            det = f.mul(&det, &pv)?;
            let pinv = f.inv(&pv)?;
            for j in 0..n {
                a[col][j] = f.mul(&a[col][j], &pinv)?;
                inv[col][j] = f.mul(&inv[col][j], &pinv)?;
            }
            for r in 0..n {
                if r == col || a[r][col].is_zero() {
                    continue;
                }
                let factor = a[r][col].clone();
                for j in 0..n {
                    let t = f.mul(&factor, &a[col][j])?;
                    a[r][j] = f.sub(&a[r][j], &t)?;
                    let t = f.mul(&factor, &inv[col][j])?;
                    inv[r][j] = f.sub(&inv[r][j], &t)?;
                }
            }
        }
        self.det = Some(det);
        self.ginv = Some(inv);
        self.timings.push(("inverse", t0.elapsed().as_secs_f64()));
        Ok(())
    }

    fn ensure_christoffel(&mut self) -> Result<(), TensorError> {
        if self.gamma2.is_some() {
            return Ok(());
        }
        self.ensure_inverse()?;
        let t0 = Instant::now();
        let f = &self.field;
        let n = self.n;
        // dg[k][i][j] = ∂_k g_ij
        let mut dg: T3 = Vec::with_capacity(n);
        for k in 0..n {
            let mut m: T2 = vec![vec![f.zero(); n]; n];
            for i in 0..n {
                for j in i..n {
                    budget()?;
                    let d = f.derivative(&self.g[i][j], k)?;
                    m[j][i] = d.clone();
                    m[i][j] = d;
                }
            }
            dg.push(m);
        }
        // Γ_{abc} = ½(∂_b g_ac + ∂_c g_ab − ∂_a g_bc)
        let mut g1: T3 = vec![vec![vec![f.zero(); n]; n]; n];
        for a in 0..n {
            for b in 0..n {
                for c in b..n {
                    budget()?;
                    let s = f.add(&dg[b][a][c], &dg[c][a][b])?;
                    let s = f.sub(&s, &dg[a][b][c])?;
                    let s = f.scale(&s, 1, 2)?;
                    g1[a][c][b] = s.clone();
                    g1[a][b][c] = s;
                }
            }
        }
        let ginv = self.ginv.as_ref().expect("inverse computed above");
        let mut g2: T3 = vec![vec![vec![f.zero(); n]; n]; n];
        for a in 0..n {
            for b in 0..n {
                for c in b..n {
                    budget()?;
                    let mut s = f.zero();
                    for d in 0..n {
                        if ginv[a][d].is_zero() || g1[d][b][c].is_zero() {
                            continue;
                        }
                        s = f.add(&s, &f.mul(&ginv[a][d], &g1[d][b][c])?)?;
                    }
                    g2[a][c][b] = s.clone();
                    g2[a][b][c] = s;
                }
            }
        }
        self.dg = Some(dg);
        self.gamma1 = Some(g1);
        self.gamma2 = Some(g2);
        self.timings
            .push(("christoffel", t0.elapsed().as_secs_f64()));
        Ok(())
    }

    fn ensure_riemann(&mut self) -> Result<(), TensorError> {
        if self.riemann_lower.is_some() {
            return Ok(());
        }
        self.ensure_christoffel()?;
        let t0 = Instant::now();
        let f = &self.field;
        let n = self.n;
        let g1 = self.gamma1.as_ref().expect("christoffel computed above");
        let g2 = self.gamma2.as_ref().expect("christoffel computed above");
        // ∂_c Γ_{a,(d b)}, computed on demand.
        let mut dg1: std::collections::HashMap<(usize, usize, usize, usize), Rf> =
            std::collections::HashMap::new();
        let mut dgam = |c: usize, a: usize, d: usize, b: usize| -> Result<Rf, TensorError> {
            let (lo, hi) = if d <= b { (d, b) } else { (b, d) };
            if let Some(v) = dg1.get(&(c, a, lo, hi)) {
                return Ok(v.clone());
            }
            let v = f.derivative(&g1[a][lo][hi], c)?;
            dg1.insert((c, a, lo, hi), v.clone());
            Ok(v)
        };
        let mut r: T4 = vec![vec![vec![vec![f.zero(); n]; n]; n]; n];
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|a| ((a + 1)..n).map(move |b| (a, b)))
            .collect();
        for (pi, &(a, b)) in pairs.iter().enumerate() {
            for &(c, d) in &pairs[pi..] {
                budget()?;
                // R_abcd = ∂_c Γ_{adb} − ∂_d Γ_{acb} + Γ_{ead} Γ^e_{bc} − Γ_{eac} Γ^e_{bd}
                let mut s = f.sub(&dgam(c, a, d, b)?, &dgam(d, a, c, b)?)?;
                for e in 0..n {
                    if !g1[e][a][d].is_zero() && !g2[e][b][c].is_zero() {
                        s = f.add(&s, &f.mul(&g1[e][a][d], &g2[e][b][c])?)?;
                    }
                    if !g1[e][a][c].is_zero() && !g2[e][b][d].is_zero() {
                        s = f.sub(&s, &f.mul(&g1[e][a][c], &g2[e][b][d])?)?;
                    }
                }
                let neg = f.neg(&s);
                for (p, q, x, y, val) in [
                    (a, b, c, d, &s),
                    (b, a, c, d, &neg),
                    (a, b, d, c, &neg),
                    (b, a, d, c, &s),
                    (c, d, a, b, &s),
                    (d, c, a, b, &neg),
                    (c, d, b, a, &neg),
                    (d, c, b, a, &s),
                ] {
                    r[p][q][x][y] = val.clone();
                }
            }
        }
        // R^a_{bcd} = g^{ae} R_{ebcd}
        let ginv = self.ginv.as_ref().expect("inverse computed above");
        let mut mixed: T4 = vec![vec![vec![vec![f.zero(); n]; n]; n]; n];
        for a in 0..n {
            for b in 0..n {
                for &(c, d) in &pairs {
                    budget()?;
                    let mut s = f.zero();
                    for e in 0..n {
                        if ginv[a][e].is_zero() || r[e][b][c][d].is_zero() {
                            continue;
                        }
                        s = f.add(&s, &f.mul(&ginv[a][e], &r[e][b][c][d])?)?;
                    }
                    mixed[a][b][d][c] = f.neg(&s);
                    mixed[a][b][c][d] = s;
                }
            }
        }
        self.riemann_lower = Some(r);
        self.riemann_mixed = Some(mixed);
        self.timings.push(("riemann", t0.elapsed().as_secs_f64()));
        Ok(())
    }

    fn ensure_ricci(&mut self) -> Result<(), TensorError> {
        if self.scalar.is_some() {
            return Ok(());
        }
        self.ensure_riemann()?;
        let t0 = Instant::now();
        let f = &self.field;
        let n = self.n;
        let mixed = self.riemann_mixed.as_ref().expect("riemann computed above");
        let ginv = self.ginv.as_ref().expect("inverse computed above");
        let mut ric: T2 = vec![vec![f.zero(); n]; n];
        for b in 0..n {
            for d in b..n {
                budget()?;
                let mut s = f.zero();
                for a in 0..n {
                    s = f.add(&s, &mixed[a][b][a][d])?;
                }
                ric[d][b] = s.clone();
                ric[b][d] = s;
            }
        }
        let mut sc = f.zero();
        for b in 0..n {
            for d in 0..n {
                if ginv[b][d].is_zero() || ric[b][d].is_zero() {
                    continue;
                }
                sc = f.add(&sc, &f.mul(&ginv[b][d], &ric[b][d])?)?;
            }
        }
        self.ricci = Some(ric);
        self.scalar = Some(sc);
        self.timings.push(("ricci", t0.elapsed().as_secs_f64()));
        Ok(())
    }

    // ----- public results ---------------------------------------------------

    fn out2(&self, t: &T2, pool: &ExprPool) -> Result<Vec<Vec<ExprId>>, TensorError> {
        t.iter()
            .map(|row| row.iter().map(|x| self.field.to_expr(x, pool)).collect())
            .collect()
    }

    fn out3(&self, t: &T3, pool: &ExprPool) -> Result<Vec<Vec<Vec<ExprId>>>, TensorError> {
        t.iter().map(|m| self.out2(m, pool)).collect()
    }

    fn out4(&self, t: &T4, pool: &ExprPool) -> Result<Vec<Vec<Vec<Vec<ExprId>>>>, TensorError> {
        t.iter().map(|m| self.out3(m, pool)).collect()
    }

    /// The metric components, in canonical form.
    pub fn components(&self, pool: &ExprPool) -> Result<Vec<Vec<ExprId>>, TensorError> {
        self.out2(&self.g, pool)
    }

    /// `det g`.
    pub fn determinant(&mut self, pool: &ExprPool) -> Result<ExprId, TensorError> {
        self.ensure_inverse()?;
        let d = self.det.as_ref().expect("computed");
        self.field.to_expr(d, pool)
    }

    /// The inverse metric `g^{ab}`.
    pub fn inverse(&mut self, pool: &ExprPool) -> Result<Vec<Vec<ExprId>>, TensorError> {
        self.ensure_inverse()?;
        self.out2(self.ginv.as_ref().expect("computed"), pool)
    }

    /// Christoffel symbols of the second kind, `out[a][b][c] = Γ^a_{bc}`.
    pub fn christoffel(&mut self, pool: &ExprPool) -> Result<Vec<Vec<Vec<ExprId>>>, TensorError> {
        self.ensure_christoffel()?;
        self.out3(self.gamma2.as_ref().expect("computed"), pool)
    }

    /// Christoffel symbols of the first kind, `out[a][b][c] = Γ_{abc}`.
    pub fn christoffel_first_kind(
        &mut self,
        pool: &ExprPool,
    ) -> Result<Vec<Vec<Vec<ExprId>>>, TensorError> {
        self.ensure_christoffel()?;
        self.out3(self.gamma1.as_ref().expect("computed"), pool)
    }

    /// The Riemann tensor `out[a][b][c][d] = R^a_{bcd}`.
    pub fn riemann(&mut self, pool: &ExprPool) -> Result<Vec<Vec<Vec<Vec<ExprId>>>>, TensorError> {
        self.ensure_riemann()?;
        self.out4(self.riemann_mixed.as_ref().expect("computed"), pool)
    }

    /// The all-lower Riemann tensor `out[a][b][c][d] = R_{abcd}`.
    pub fn riemann_lower(
        &mut self,
        pool: &ExprPool,
    ) -> Result<Vec<Vec<Vec<Vec<ExprId>>>>, TensorError> {
        self.ensure_riemann()?;
        self.out4(self.riemann_lower.as_ref().expect("computed"), pool)
    }

    /// The Ricci tensor `R_{ab}`.
    pub fn ricci(&mut self, pool: &ExprPool) -> Result<Vec<Vec<ExprId>>, TensorError> {
        self.ensure_ricci()?;
        self.out2(self.ricci.as_ref().expect("computed"), pool)
    }

    /// The Ricci scalar `R`.
    pub fn ricci_scalar(&mut self, pool: &ExprPool) -> Result<ExprId, TensorError> {
        self.ensure_ricci()?;
        self.field
            .to_expr(self.scalar.as_ref().expect("computed"), pool)
    }

    /// `true` when every Ricci component is exactly zero (a proof, not a sample).
    pub fn is_ricci_flat(&mut self) -> Result<bool, TensorError> {
        self.ensure_ricci()?;
        Ok(self
            .ricci
            .as_ref()
            .expect("computed")
            .iter()
            .all(|row| row.iter().all(Rf::is_zero)))
    }

    /// The Einstein tensor `G_{ab} = R_{ab} − ½ R g_{ab}`.
    pub fn einstein(&mut self, pool: &ExprPool) -> Result<Vec<Vec<ExprId>>, TensorError> {
        self.ensure_ricci()?;
        let t0 = Instant::now();
        let f = &self.field;
        let n = self.n;
        let ric = self.ricci.as_ref().expect("computed");
        let half_r = f.scale(self.scalar.as_ref().expect("computed"), 1, 2)?;
        let mut out: T2 = vec![vec![f.zero(); n]; n];
        for a in 0..n {
            for b in a..n {
                budget()?;
                let v = f.sub(&ric[a][b], &f.mul(&half_r, &self.g[a][b])?)?;
                out[b][a] = v.clone();
                out[a][b] = v;
            }
        }
        let res = self.out2(&out, pool);
        self.timings.push(("einstein", t0.elapsed().as_secs_f64()));
        res
    }

    /// The Weyl tensor `C_{abcd}` (all indices down; `n ≥ 3`).
    pub fn weyl(&mut self, pool: &ExprPool) -> Result<Vec<Vec<Vec<Vec<ExprId>>>>, TensorError> {
        if self.n < 3 {
            return Err(TensorError::Dimension {
                what: "Weyl tensor",
                min: 3,
                got: self.n,
            });
        }
        self.ensure_ricci()?;
        let t0 = Instant::now();
        let f = &self.field;
        let n = self.n;
        let g = &self.g;
        let ric = self.ricci.as_ref().expect("computed");
        let r = self.riemann_lower.as_ref().expect("computed");
        let sc = self.scalar.as_ref().expect("computed");
        let nn = n as i64;
        let k1 = f.rational(&1.into(), &(nn - 2).into())?;
        let k2 = f.mul(sc, &f.rational(&1.into(), &((nn - 1) * (nn - 2)).into())?)?;
        let mut out: T4 = vec![vec![vec![vec![f.zero(); n]; n]; n]; n];
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|a| ((a + 1)..n).map(move |b| (a, b)))
            .collect();
        let m = |x: &Rf, y: &Rf| f.mul(x, y);
        for (pi, &(a, b)) in pairs.iter().enumerate() {
            for &(c, d) in &pairs[pi..] {
                budget()?;
                // g_ac R_bd − g_ad R_bc − g_bc R_ad + g_bd R_ac
                let mut t = m(&g[a][c], &ric[b][d])?;
                t = f.sub(&t, &m(&g[a][d], &ric[b][c])?)?;
                t = f.sub(&t, &m(&g[b][c], &ric[a][d])?)?;
                t = f.add(&t, &m(&g[b][d], &ric[a][c])?)?;
                let gg = f.sub(&m(&g[a][c], &g[b][d])?, &m(&g[a][d], &g[b][c])?)?;
                let mut v = f.sub(&r[a][b][c][d], &m(&k1, &t)?)?;
                v = f.add(&v, &m(&k2, &gg)?)?;
                let neg = f.neg(&v);
                for (p, q, x, y, val) in [
                    (a, b, c, d, &v),
                    (b, a, c, d, &neg),
                    (a, b, d, c, &neg),
                    (b, a, d, c, &v),
                    (c, d, a, b, &v),
                    (d, c, a, b, &neg),
                    (c, d, b, a, &neg),
                    (d, c, b, a, &v),
                ] {
                    out[p][q][x][y] = val.clone();
                }
            }
        }
        let res = self.out4(&out, pool);
        self.timings.push(("weyl", t0.elapsed().as_secs_f64()));
        res
    }

    /// The Kretschmann scalar `K = R_{abcd} R^{abcd}`.
    pub fn kretschmann(&mut self, pool: &ExprPool) -> Result<ExprId, TensorError> {
        self.ensure_riemann()?;
        let t0 = Instant::now();
        let f = &self.field;
        let n = self.n;
        let r = self.riemann_lower.as_ref().expect("computed");
        let ginv = self.ginv.as_ref().expect("computed");
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|a| ((a + 1)..n).map(move |b| (a, b)))
            .collect();
        // M^{ab}_{cd} = g^{ae} g^{bf} R_{efcd}, for a < b and c < d.
        let mut up1: T4 = vec![vec![vec![vec![f.zero(); n]; n]; n]; n];
        for a in 0..n {
            for b in 0..n {
                for &(c, d) in &pairs {
                    budget()?;
                    let mut s = f.zero();
                    for e in 0..n {
                        if ginv[a][e].is_zero() || r[e][b][c][d].is_zero() {
                            continue;
                        }
                        s = f.add(&s, &f.mul(&ginv[a][e], &r[e][b][c][d])?)?;
                    }
                    up1[a][b][c][d] = s;
                }
            }
        }
        let mut m: T4 = vec![vec![vec![vec![f.zero(); n]; n]; n]; n];
        for &(a, b) in &pairs {
            for &(c, d) in &pairs {
                budget()?;
                let mut s = f.zero();
                for e in 0..n {
                    if ginv[b][e].is_zero() || up1[a][e][c][d].is_zero() {
                        continue;
                    }
                    s = f.add(&s, &f.mul(&ginv[b][e], &up1[a][e][c][d])?)?;
                }
                m[a][b][c][d] = s;
            }
        }
        // K = 4 Σ_{a<b, c<d} M^{ab}_{cd} M^{cd}_{ab}
        let mut k = f.zero();
        for &(a, b) in &pairs {
            for &(c, d) in &pairs {
                if m[a][b][c][d].is_zero() || m[c][d][a][b].is_zero() {
                    continue;
                }
                k = f.add(&k, &f.mul(&m[a][b][c][d], &m[c][d][a][b])?)?;
            }
        }
        let k = f.scale(&k, 4, 1)?;
        let res = self.field.to_expr(&k, pool);
        self.timings
            .push(("kretschmann", t0.elapsed().as_secs_f64()));
        res
    }

    /// Geodesic equations: `out[a] = −Γ^a_{bc} v^b v^c`, the coordinate
    /// accelerations `d²x^a/dτ²` given the velocities `v` (symbols or any
    /// expressions, e.g. `Derivative(x(τ), τ)`).
    pub fn geodesic_accelerations(
        &mut self,
        velocities: &[ExprId],
        pool: &ExprPool,
    ) -> Result<Vec<ExprId>, TensorError> {
        if velocities.len() != self.n {
            return Err(TensorError::Shape(format!(
                "expected {} velocities, got {}",
                self.n,
                velocities.len()
            )));
        }
        self.ensure_christoffel()?;
        let gam = self.christoffel(pool)?;
        let n = self.n;
        let mut out = Vec::with_capacity(n);
        for row in gam.iter() {
            let mut terms = Vec::new();
            for b in 0..n {
                for c in b..n {
                    let gb = row[b][c];
                    if matches!(pool.get(gb), ExprData::Integer(ref z) if z.0 == 0) {
                        continue;
                    }
                    let coeff = if b == c { -1 } else { -2 };
                    terms.push(pool.mul(vec![
                        pool.integer(coeff),
                        gb,
                        velocities[b],
                        velocities[c],
                    ]));
                }
            }
            out.push(match terms.len() {
                0 => pool.integer(0),
                1 => terms[0],
                _ => pool.add(terms),
            });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Free-standing canonical form
// ---------------------------------------------------------------------------

/// Canonical form of each expression in the same exact field the [`Metric`]
/// computes in (`sin² = 1 − cos²`, rationalised denominators, gcd-cancelled,
/// factored output).  `coords` name the chart; they matter only for the
/// derivative rules, so any symbols may be passed.
///
/// An expression is identically zero exactly when its canonical form is `0`
/// (see the module docs for the one caveat, a radicand that is a perfect
/// square).
pub fn canonical_form(
    exprs: &[ExprId],
    coords: &[ExprId],
    pool: &ExprPool,
) -> Result<Vec<ExprId>, TensorError> {
    check_coords(coords, pool)?;
    let mut b = FieldBuilder::new(coords, pool);
    for &e in exprs {
        b.scan(e, 0)?;
    }
    let mut field = b.build()?;
    let mut out = Vec::with_capacity(exprs.len());
    for &e in exprs {
        budget()?;
        let r = field.convert(e, pool)?;
        out.push(field.to_expr(&r, pool)?);
    }
    Ok(out)
}

/// `∂e/∂x` for each expression, returned in canonical form.
pub fn canonical_derivative(
    exprs: &[ExprId],
    coords: &[ExprId],
    var: usize,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, TensorError> {
    check_coords(coords, pool)?;
    if var >= coords.len() {
        return Err(TensorError::Shape(format!(
            "coordinate index {var} out of range for {} coordinates",
            coords.len()
        )));
    }
    let mut b = FieldBuilder::new(coords, pool);
    for &e in exprs {
        b.scan(e, 1)?;
    }
    let mut field = b.build()?;
    let mut out = Vec::with_capacity(exprs.len());
    for &e in exprs {
        budget()?;
        let r = field.convert(e, pool)?;
        let d = field.derivative(&r, var)?;
        out.push(field.to_expr(&d, pool)?);
    }
    Ok(out)
}
