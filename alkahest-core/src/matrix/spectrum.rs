//! The canonical "are these actually the eigenvalues?" check.
//!
//! `Π_i (z − λ_i)` and `det(zI − A)` are the same polynomial in `z` when the
//! `λ_i` really are the spectrum of `A`, so evaluating both at a few sampled
//! `z` (and, for symbolic entries, a few sampled parameter values) either
//! confirms the list or refutes it.
//!
//! # Why a closed-form spectrum needs checking at all
//!
//! Because a radical is not a number until a branch is chosen, and the formula
//! that produced it may need a *coordinated* choice that the expression does
//! not record. Cardano's solution of `t³ + p·t + q` is
//!
//! ```text
//!     t = A + B,   A³ = −q/2 + √Δ,   B³ = −q/2 − √Δ,   Δ = (q/2)² + (p/3)³
//! ```
//!
//! but only for the pair `(A, B)` satisfying `A·B = −p/3`; there are nine
//! `(A, B)` pairs and only three are roots. Writing the two cube roots as
//! independent principal-branch powers `(…)^{1/3}` throws that constraint
//! away, and when both radicands are negative — `Δ > 0` with `q > 0 > p` —
//! each picks up its own `e^{iπ/3}`, `A·B` acquires a factor `e^{2iπ/3}`, and
//! the three expressions are roots of nothing. That is not hypothetical: it is
//! what `matrix::eigen::cubic_roots` emitted for `λ³ + 3λ² + 2λ + 1` until the
//! real-cube-root form replaced it, and `p(λ)` at the returned values was
//! `2.29`, not `0`.
//!
//! The generator is now written so its own convention is the principal one
//! (see [`crate::matrix::eigen`]), which is what makes those values usable;
//! this module is the independent check that says so, kept separate from the
//! formula it checks on the principle that a gate is not allowed to be its own
//! witness.
//!
//! # What each verdict means
//!
//! | outcome | meaning |
//! |---|---|
//! | `SpectrumCheck::Confirmed` | the identity held at every sample that could be evaluated |
//! | `SpectrumCheck::Unevaluated` | nothing could be evaluated — **no information** |
//! | `Err(`[`SpectrumRefusal`]`)` | the identity was evaluated and **failed** |
//!
//! `Unevaluated` is deliberately not a refusal *here*: an expression this
//! module's evaluator does not model is a property of the expression, not
//! evidence against the spectrum, and the same classification is what
//! `integrate`'s antiderivative gates and `limit`'s value gate use. Callers
//! that need a positive confirmation before building on the list — `dsolve`'s
//! Putzer expansion, which turns a bad spectrum into a page-long candidate —
//! ask for `SpectrumCheck::Confirmed` explicitly.

use crate::eval::symbols::{collect_free_symbols, collect_named_constants, is_pi};
use crate::eval::{eval_complex_f64, ComplexF64};
use crate::kernel::{ExprId, ExprPool};
use crate::matrix::Matrix;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Complex helpers
// ---------------------------------------------------------------------------
//
// `ComplexF64` exposes its parts but not its arithmetic, and the determinant
// below needs four operations on it. They are three lines each.

fn cadd(a: ComplexF64, b: ComplexF64) -> ComplexF64 {
    ComplexF64::new(a.re + b.re, a.im + b.im)
}

fn csub(a: ComplexF64, b: ComplexF64) -> ComplexF64 {
    ComplexF64::new(a.re - b.re, a.im - b.im)
}

fn cmul(a: ComplexF64, b: ComplexF64) -> ComplexF64 {
    ComplexF64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

fn cdiv(a: ComplexF64, b: ComplexF64) -> ComplexF64 {
    let d = b.re * b.re + b.im * b.im;
    ComplexF64::new(
        (a.re * b.re + a.im * b.im) / d,
        (a.im * b.re - a.re * b.im) / d,
    )
}

fn cabs(a: ComplexF64) -> f64 {
    a.re.hypot(a.im)
}

fn cfinite(a: ComplexF64) -> bool {
    a.re.is_finite() && a.im.is_finite()
}

/// `det` by Gaussian elimination with partial pivoting; `None` when the matrix
/// is singular to working precision, which is no information about the
/// identity being tested rather than a value of zero.
fn complex_det(m: &mut [Vec<ComplexF64>]) -> Option<ComplexF64> {
    let n = m.len();
    let mut det = ComplexF64::ONE;
    for k in 0..n {
        let mut piv = k;
        for r in (k + 1)..n {
            if cabs(m[r][k]) > cabs(m[piv][k]) {
                piv = r;
            }
        }
        if cabs(m[piv][k]) < 1e-13 {
            return None;
        }
        if piv != k {
            m.swap(piv, k);
            det = cmul(det, ComplexF64::new(-1.0, 0.0));
        }
        det = cmul(det, m[k][k]);
        for r in (k + 1)..n {
            let f = cdiv(m[r][k], m[k][k]);
            // `r > k`, so the pivot row and the row being cleared are in
            // different halves and can be borrowed at once.
            let (above, from_r) = m.split_at_mut(r);
            let pivot_row = &above[k];
            for (dst, &src) in from_r[0].iter_mut().zip(pivot_row.iter()).skip(k) {
                *dst = csub(*dst, cmul(f, src));
            }
        }
    }
    if cfinite(det) {
        Some(det)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// The refusal
// ---------------------------------------------------------------------------

/// A closed-form spectrum that is not the spectrum it claims to be.
///
/// # Why this is not an error variant
///
/// [`EigenError`](crate::matrix::EigenError) and
/// [`LinearAlgebraError`](crate::matrix::LinearAlgebraError) are public
/// *exhaustive* enums, so growing either a `SpectrumUnverified` variant is a
/// major semver break — and so is marking them `#[non_exhaustive]` to allow one
/// later. A correctness fix inside a patch release cannot spend a major
/// version, so the refusal travels out of band instead: the refusing routine
/// returns `UnsupportedIrreducibleDegree { degree }`, whose reworded text
/// states exactly the disjunction that is known ("no *usable* closed form for a
/// factor of this degree"), and the real cause is recorded here for
/// [`take_spectrum_refusal`] to hand to the bindings, which raise its own
/// `E-EIGEN-008`.
///
/// This is the pattern [`crate::matrix::take_zero_test_refusal`] uses for
/// undecided zero tests inside `MatrixError::SingularMatrix`, and
/// `crate::calculus::series::take_series_refusal` for truncated expansions
/// inside `SeriesError::InvalidOrder`.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumRefusal {
    lambdas: Vec<String>,
    probe: (f64, f64),
    product: (f64, f64),
    determinant: (f64, f64),
}

impl SpectrumRefusal {
    /// The candidate eigenvalues, rendered, in the order they were checked.
    pub fn lambdas(&self) -> &[String] {
        &self.lambdas
    }

    /// The `z` at which `Π(z − λ)` and `det(zI − A)` were found to differ.
    pub fn probe(&self) -> (f64, f64) {
        self.probe
    }
}

impl fmt::Display for SpectrumRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the closed-form eigenvalues [{}] are not roots of the characteristic \
             polynomial when their radicals are read on principal branches: at \
             z = {} + {}i, Π(z − λ) = {} + {}i but det(zI − A) = {} + {}i. \
             Refusing rather than returning a spectrum whose correctness depends \
             on a branch convention nothing in the expression records",
            self.lambdas.join(", "),
            self.probe.0,
            self.probe.1,
            self.product.0,
            self.product.1,
            self.determinant.0,
            self.determinant.1,
        )
    }
}

impl std::error::Error for SpectrumRefusal {}

impl crate::errors::AlkahestError for SpectrumRefusal {
    fn code(&self) -> &'static str {
        "E-EIGEN-008"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some("substitute concrete numbers for any symbolic entries, or use real_roots / numeric eigenvalues for this matrix")
    }
}

thread_local! {
    /// The refusal behind the `UnsupportedIrreducibleDegree` the current thread
    /// is about to return, when that error is a failed spectrum check rather
    /// than a degree this module never had a formula for.
    static LAST_REFUSAL: RefCell<Option<SpectrumRefusal>> = const { RefCell::new(None) };
}

/// Take the spectrum refusal recorded by the most recent eigen call on this
/// thread, if it made one. Clears it.
pub fn take_spectrum_refusal() -> Option<SpectrumRefusal> {
    LAST_REFUSAL.with(|c| c.borrow_mut().take())
}

pub(crate) fn record_refusal(r: SpectrumRefusal) {
    LAST_REFUSAL.with(|c| *c.borrow_mut() = Some(r));
}

/// Drop any refusal left on this thread, so a later, unrelated
/// `UnsupportedIrreducibleDegree` cannot inherit `E-EIGEN-008`.
pub(crate) fn forget_refusal() {
    LAST_REFUSAL.with(|c| *c.borrow_mut() = None);
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

/// What [`confirm_spectrum`] was able to establish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpectrumCheck {
    /// `Π(z − λ) = det(zI − A)` held at every one of `samples` evaluated probes.
    Confirmed { samples: usize },
    /// Neither the entries nor the eigenvalues could be turned into numbers at
    /// any sample, so nothing was tested. No information either way.
    Unevaluated,
}

impl SpectrumCheck {
    pub(crate) fn is_confirmed(self) -> bool {
        matches!(self, SpectrumCheck::Confirmed { .. })
    }
}

/// Parameter values to substitute for free symbols, chosen to be unrelated and
/// away from the small integers that make accidental agreement likely.
const PARAM_SETS: [&[f64]; 2] = [&[1.7, 0.6, 2.3, 1.1, 0.4], &[0.37, 1.9, 0.83, 2.7, 1.3]];
/// `z` values at which the two polynomials are compared. Off the real axis and
/// away from any small-integer eigenvalue.
const PROBES: [(f64, f64); 3] = [(0.53, 0.29), (-1.17, 0.71), (2.31, -0.43)];

/// Check that `lambdas` (listed with multiplicity, so `lambdas.len() == a.rows`)
/// is the spectrum of `a`.
///
/// The comparison is `Π_i (z − λ_i)` against `det(zI − A)`, a polynomial
/// identity of degree `n` in `z`; agreement at three probes is not a proof, but
/// disagreement at one is a refutation, and refutation is the direction that
/// matters here.
pub(crate) fn confirm_spectrum(
    a: &Matrix,
    lambdas: &[ExprId],
    pool: &ExprPool,
) -> Result<SpectrumCheck, SpectrumRefusal> {
    let n = a.rows;
    if n != a.cols || lambdas.len() != n {
        // Not a shape this check is about; the caller's own arity errors say so.
        return Ok(SpectrumCheck::Unevaluated);
    }

    let mut params: Vec<ExprId> = Vec::new();
    for &e in a.entries() {
        collect_free_symbols(e, pool, &mut params);
    }
    for &l in lambdas {
        collect_free_symbols(l, pool, &mut params);
    }
    params.sort_by_key(|&s| pool.display(s).to_string());

    let mut checked = 0usize;
    for ps in PARAM_SETS {
        let mut env: HashMap<ExprId, ComplexF64> = HashMap::new();
        bind_constants(pool, &mut env, a, lambdas);
        for (i, &p) in params.iter().enumerate() {
            env.insert(p, ComplexF64::new(ps[i % ps.len()], 0.0));
        }
        let Some(entries) = a
            .entries()
            .iter()
            .map(|&e| eval_complex_f64(e, pool, &env).ok().filter(|&v| cfinite(v)))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let Some(lam) = lambdas
            .iter()
            .map(|&l| eval_complex_f64(l, pool, &env).ok().filter(|&v| cfinite(v)))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        for (zr, zi) in PROBES {
            let z = ComplexF64::new(zr, zi);
            let mut m: Vec<Vec<ComplexF64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let e = entries[i * n + j];
                            let neg = ComplexF64::new(-e.re, -e.im);
                            if i == j {
                                cadd(z, neg)
                            } else {
                                neg
                            }
                        })
                        .collect()
                })
                .collect();
            let Some(det) = complex_det(&mut m) else {
                continue;
            };
            let prod = lam
                .iter()
                .fold(ComplexF64::ONE, |acc, &l| cmul(acc, csub(z, l)));
            let scale = cabs(det).max(cabs(prod)).max(1.0);
            if cabs(csub(det, prod)) > 1e-7 * scale {
                return Err(SpectrumRefusal {
                    lambdas: lambdas
                        .iter()
                        .map(|&l| pool.display(l).to_string())
                        .collect(),
                    probe: (zr, zi),
                    product: (prod.re, prod.im),
                    determinant: (det.re, det.im),
                });
            }
            checked += 1;
        }
    }

    if checked == 0 {
        Ok(SpectrumCheck::Unevaluated)
    } else {
        Ok(SpectrumCheck::Confirmed { samples: checked })
    }
}

/// Bind the named constants the eigen formulas emit but that
/// [`eval_complex_f64`] has no value for.
///
/// `pi` is a plain symbol in this crate (see [`crate::eval::symbols`]), so the
/// casus irreducibilis form `2√(−p/3)·cos((acos c + 2πk)/3)` would otherwise be
/// unevaluable — and, worse, `pi` would be collected as a free *parameter* and
/// given a sample value, turning a correct spectrum into a spurious refusal.
/// Binding it here is what lets the three-real-roots branch be confirmed rather
/// than merely tolerated.
///
/// The imaginary unit is the other symbol
/// [`collect_named_constants`](crate::eval::symbols::collect_named_constants)
/// reports, and it is deliberately *not* bound: `eval_complex_f64` knows it
/// natively, and binding it here would overwrite `i` with `π`.
fn bind_constants(
    pool: &ExprPool,
    env: &mut HashMap<ExprId, ComplexF64>,
    a: &Matrix,
    lambdas: &[ExprId],
) {
    let mut all: Vec<ExprId> = Vec::new();
    for &e in a.entries() {
        collect_named_constants(e, pool, &mut all);
    }
    for &l in lambdas {
        collect_named_constants(l, pool, &mut all);
    }
    for s in all {
        if is_pi(s, pool) {
            env.insert(s, ComplexF64::new(std::f64::consts::PI, 0.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn mat(rows: usize, entries: Vec<ExprId>, pool: &ExprPool) -> Matrix {
        let mut m = Matrix::zeros(rows, rows, pool);
        for i in 0..rows {
            for j in 0..rows {
                m.set(i, j, entries[i * rows + j]);
            }
        }
        m
    }

    #[test]
    fn a_correct_rational_spectrum_is_confirmed() {
        let p = ExprPool::new();
        // diag(2, 5): the spectrum is {2, 5}.
        let z = p.integer(0_i32);
        let m = mat(2, vec![p.integer(2_i32), z, z, p.integer(5_i32)], &p);
        let out = confirm_spectrum(&m, &[p.integer(2_i32), p.integer(5_i32)], &p).unwrap();
        assert!(out.is_confirmed(), "{out:?}");
    }

    #[test]
    fn a_wrong_spectrum_is_refuted() {
        let p = ExprPool::new();
        let z = p.integer(0_i32);
        let m = mat(2, vec![p.integer(2_i32), z, z, p.integer(5_i32)], &p);
        let err = confirm_spectrum(&m, &[p.integer(2_i32), p.integer(6_i32)], &p)
            .expect_err("{2, 6} is not the spectrum of diag(2, 5)");
        assert_eq!(
            <SpectrumRefusal as crate::errors::AlkahestError>::code(&err),
            "E-EIGEN-008"
        );
        assert!(err.lambdas().len() == 2, "{err}");
    }

    #[test]
    fn an_unevaluable_spectrum_is_no_information_not_a_refusal() {
        let p = ExprPool::new();
        let z = p.integer(0_i32);
        let m = mat(2, vec![p.integer(2_i32), z, z, p.integer(5_i32)], &p);
        // `zeta` has no entry in the complex evaluator, so neither λ is a number.
        let opaque = p.func("zeta", vec![p.integer(3_i32)]);
        let out = confirm_spectrum(&m, &[opaque, opaque], &p).unwrap();
        assert_eq!(out, SpectrumCheck::Unevaluated);
    }

    #[test]
    fn pi_is_a_constant_not_a_sampled_parameter() {
        let p = ExprPool::new();
        // A matrix whose spectrum is {π, 0}: `[[π, 0], [0, 0]]`.
        let pi = p.symbol("pi", Domain::Real);
        let z = p.integer(0_i32);
        let m = mat(2, vec![pi, z, z, z], &p);
        let out = confirm_spectrum(&m, &[pi, z], &p).unwrap();
        assert!(
            out.is_confirmed(),
            "π must be bound to 3.14159…, not sampled: {out:?}"
        );
    }
}
