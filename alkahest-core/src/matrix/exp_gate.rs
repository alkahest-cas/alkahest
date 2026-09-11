//! The standing "is this really `e^A`?" check.
//!
//! `matrix_exponential` returns a closed-form matrix. This module recomputes
//! `e^A` by an entirely different route — scaling-and-squaring on a truncated
//! Taylor series, in `f64` complex arithmetic — and compares. Agreement at a
//! few sampled parameter values is not a proof; **disagreement at one is a
//! refutation**, and refutation is the direction that matters.
//!
//! # Why this exists
//!
//! Until 3.10.1 the defective branch of `matrix_exponential` built `e^{J}` for
//! a Jordan block as `e^λ·λ^k/k!` rather than `e^λ/k!` — the nilpotent power
//! `N^k` was replaced by `λ^k`. Nothing downstream could tell:
//!
//! ```text
//! exp([[0,1],[0,0]])  →  [[1,0],[0,1]]        truth [[1,1],[0,1]]
//! exp([[2,1],[0,2]])  →  [[e², 2e²],[0, e²]]  truth [[e², e²],[0, e²]]
//! ```
//!
//! Both are clean, plausible matrices returned with no exception and no flag,
//! which is the exact shape of failure `tests/silent_errors/` exists to catch.
//! The formula is now Putzer's (see `crate::matrix::putzer`) and this is the
//! independent witness that says so — kept in its own file on the principle
//! that a gate may not be its own witness.
//!
//! # What each verdict means
//!
//! | outcome | meaning |
//! |---|---|
//! | `ExpCheck::Confirmed` | `‖candidate − e^A‖` was within tolerance at every sample that could be evaluated |
//! | `ExpCheck::Unevaluated` | nothing could be turned into numbers — **no information** |
//! | `Err(`[`MatrixExpRefusal`]`)` | the two were evaluated and **differ** |
//!
//! `Unevaluated` is not a refusal: an expression this module's evaluator does
//! not model is a property of the expression, not evidence against the answer,
//! which is the same classification [`crate::matrix::spectrum`] and
//! `integrate`'s antiderivative gates use.

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
// `ComplexF64` exposes its parts but not its arithmetic, and a matrix product
// needs two operations on it.

fn cadd(a: ComplexF64, b: ComplexF64) -> ComplexF64 {
    ComplexF64::new(a.re + b.re, a.im + b.im)
}

fn cmul(a: ComplexF64, b: ComplexF64) -> ComplexF64 {
    ComplexF64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

fn cscale(a: ComplexF64, s: f64) -> ComplexF64 {
    ComplexF64::new(a.re * s, a.im * s)
}

fn cabs(a: ComplexF64) -> f64 {
    a.re.hypot(a.im)
}

fn cfinite(a: ComplexF64) -> bool {
    a.re.is_finite() && a.im.is_finite()
}

type Cm = Vec<Vec<ComplexF64>>;

fn cmat_mul(a: &Cm, b: &Cm) -> Cm {
    let n = a.len();
    let mut out = vec![vec![ComplexF64::new(0.0, 0.0); n]; n];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut s = ComplexF64::new(0.0, 0.0);
            for k in 0..n {
                s = cadd(s, cmul(a[i][k], b[k][j]));
            }
            *cell = s;
        }
    }
    out
}

fn cmat_norm(a: &Cm) -> f64 {
    a.iter()
        .map(|r| r.iter().map(|&c| cabs(c)).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

/// `e^A` by scaling and squaring with a truncated Taylor series.
///
/// `A/2^s` is scaled to norm ≤ ½, where a 20-term Taylor series is accurate to
/// well past `f64` resolution, and the result is squared back `s` times.
/// Deliberately not the algorithm under test, and deliberately not symbolic:
/// its independence is the whole point.
fn cmat_exp(a: &Cm) -> Option<Cm> {
    let n = a.len();
    let norm = cmat_norm(a);
    if !norm.is_finite() {
        return None;
    }
    let s = if norm > 0.5 {
        (norm / 0.5).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    // `2^s` past this means the answer has lost every digit it had.
    if s > 60 {
        return None;
    }
    let scale = 1.0 / (2.0_f64).powi(s as i32);
    let b: Cm = a
        .iter()
        .map(|r| r.iter().map(|&c| cscale(c, scale)).collect())
        .collect();

    let mut term: Cm = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| ComplexF64::new(f64::from(u8::from(i == j)), 0.0))
                .collect()
        })
        .collect();
    let mut acc = term.clone();
    for k in 1..=20u32 {
        term = cmat_mul(&term, &b);
        let inv = 1.0 / f64::from(k);
        for row in term.iter_mut() {
            for c in row.iter_mut() {
                *c = cscale(*c, inv);
            }
        }
        for (ar, tr) in acc.iter_mut().zip(term.iter()) {
            for (a, &t) in ar.iter_mut().zip(tr.iter()) {
                *a = cadd(*a, t);
            }
        }
    }
    for _ in 0..s {
        acc = cmat_mul(&acc, &acc);
    }
    if acc.iter().flatten().all(|&c| cfinite(c)) {
        Some(acc)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// The refusal
// ---------------------------------------------------------------------------

/// A closed-form `e^A` that is not `e^A`.
///
/// # Why this is not an error variant
///
/// [`LinearAlgebraError`](crate::matrix::LinearAlgebraError) is a public
/// *exhaustive* enum, so growing it a variant is a major semver break — and so
/// is marking it `#[non_exhaustive]` to allow one later. A correctness fix
/// inside a patch release cannot spend a major version, so the refusal travels
/// out of band: the refusing routine returns
/// [`UnsupportedField`](crate::matrix::LinearAlgebraError::UnsupportedField),
/// whose text states the disjunction that is known, and the real cause is
/// recorded here for [`take_matrix_exp_refusal`] to hand to the bindings, which
/// raise its own `E-LINALG-011`.
///
/// This is the pattern [`crate::matrix::take_spectrum_refusal`] uses for a
/// spectrum that fails `Π(z−λ) = det(zI−A)`, and
/// [`crate::matrix::take_zero_test_refusal`] for undecided zero tests.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixExpRefusal {
    reason: ExpRefusalReason,
    detail: String,
}

/// Which of the two ways `matrix_exponential` can decline it was.
///
/// An undecided eigenvalue gap is deliberately absent: it is *reported*, not
/// refused. See
/// [`take_matrix_exp_side_conditions`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpRefusalReason {
    /// The eigenvalue list could not be positively confirmed as the spectrum,
    /// and Putzer's expansion is built entirely on it.
    SpectrumUnconfirmed,
    /// A candidate was produced and then disagreed with an independently
    /// computed `e^A`.
    FailedVerification,
}

impl MatrixExpRefusal {
    /// Which of the two refusals this is.
    pub fn reason(&self) -> ExpRefusalReason {
        self.reason
    }
}

impl fmt::Display for MatrixExpRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.detail)
    }
}

impl std::error::Error for MatrixExpRefusal {}

impl crate::errors::AlkahestError for MatrixExpRefusal {
    fn code(&self) -> &'static str {
        "E-LINALG-011"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "substitute concrete numbers for any symbolic entries, or state the assumption \
             that makes the eigenvalue gap non-zero",
        )
    }
}

thread_local! {
    /// The refusal behind the `UnsupportedField` the current thread is about to
    /// return, when that error is a matrix-exponential refusal rather than the
    /// thing its variant usually means.
    static LAST_REFUSAL: RefCell<Option<MatrixExpRefusal>> = const { RefCell::new(None) };
}

/// Take the matrix-exponential refusal recorded by the most recent
/// [`matrix_exponential`](crate::matrix::matrix_exponential) call on this
/// thread, if it made one. Clears it.
pub fn take_matrix_exp_refusal() -> Option<MatrixExpRefusal> {
    LAST_REFUSAL.with(|c| c.borrow_mut().take())
}

pub(crate) fn record_refusal(reason: ExpRefusalReason, detail: String) {
    // A refusal recorded here is the *whole* explanation of the error about to
    // be returned; a zero-test refusal left over from an inner elimination
    // would otherwise be picked up by the bindings for the same variant.
    crate::matrix::zero_test::forget_refusal();
    LAST_REFUSAL.with(|c| *c.borrow_mut() = Some(MatrixExpRefusal { reason, detail }));
}

/// Drop any refusal left on this thread, so a later, unrelated
/// `UnsupportedField` cannot inherit `E-LINALG-011`.
pub(crate) fn forget_refusal() {
    LAST_REFUSAL.with(|c| *c.borrow_mut() = None);
    LAST_CONDITIONS.with(|c| c.borrow_mut().clear());
}

thread_local! {
    /// Eigenvalue gaps the most recent successful `matrix_exponential` divided
    /// by without being able to settle them.
    static LAST_CONDITIONS: RefCell<Vec<ExprId>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn record_side_conditions(gaps: Vec<ExprId>) {
    LAST_CONDITIONS.with(|c| *c.borrow_mut() = gaps);
}

/// The hypotheses the matrix returned by the most recent
/// [`matrix_exponential`](crate::matrix::matrix_exponential) call on this
/// thread rests on, as [`crate::deriv::SideCondition::NonZero`].
///
/// `exp([[a,1],[0,b]])` has the off-diagonal `(e^a − e^b)/(a − b)`, which is
/// the answer **for `a ≠ b`**: at `a = b` the matrix is defective and the
/// entry is `e^a`, the limit, not a quotient. The generic-parameter reading is
/// the useful one and is what every CAS returns — but a caller cannot audit an
/// assumption that is never stated, and a symbolic `e^A` is returned with no
/// verification the caller can read (the standing gate samples *away* from the
/// confluence by construction), so this is the only honest signal available on
/// that path.
///
/// # Why out of band
///
/// [`matrix_exponential`](crate::matrix::matrix_exponential) returns a bare
/// [`Matrix`], a public type; it cannot grow a conditions field without a major
/// semver break. The hypotheses therefore travel beside the result, the same
/// out-of-band treatment [`crate::solver::take_solve_side_conditions`] gives
/// the solver's `a ≠ 0` and [`crate::matrix::take_zero_test_refusal`] gives an
/// undecided zero test.
///
/// Consuming, so one call's hypotheses cannot be read as a later call's. Empty
/// means every gap divided by was *proven* non-zero — not that none was looked
/// at.
pub fn take_matrix_exp_side_conditions() -> Vec<crate::deriv::SideCondition> {
    LAST_CONDITIONS.with(|c| {
        std::mem::take(&mut *c.borrow_mut())
            .into_iter()
            .map(crate::deriv::SideCondition::NonZero)
            .collect()
    })
}

pub(crate) fn spectrum_unconfirmed(n: usize) -> String {
    format!(
        "the eigenvalues of this {n}×{n} matrix could not be evaluated at any sample, so \
         nothing confirms they are its spectrum — and Putzer's expansion of e^A is built \
         entirely on them. Refusing rather than returning a matrix whose correctness rests \
         on an unchecked eigenvalue list"
    )
}

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

/// What [`confirm_matrix_exponential`] was able to establish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpCheck {
    /// The candidate agreed with an independently computed `e^A` at every one
    /// of `samples` evaluated parameter bindings.
    Confirmed { samples: usize },
    /// Neither the matrix nor the candidate could be turned into numbers at any
    /// sample, so nothing was tested. No information either way.
    Unevaluated,
}

impl ExpCheck {
    /// Only the *negative* verdict changes what a caller does, so this is the
    /// gate's own tests asking whether it can still say yes.
    #[cfg(test)]
    pub(crate) fn is_confirmed(self) -> bool {
        matches!(self, ExpCheck::Confirmed { .. })
    }
}

/// Parameter values to substitute for free symbols. Unrelated, away from the
/// small integers that make accidental agreement likely, and small enough in
/// magnitude that `e^A` stays inside `f64`.
const PARAM_SETS: [&[f64]; 3] = [
    &[1.7, 0.6, -2.3, 1.1, 0.4],
    &[0.37, -1.9, 0.83, 2.7, 1.3],
    &[-0.85, 1.45, 0.21, -0.62, 2.05],
];

/// Relative band on the entrywise difference.
///
/// Loose on purpose: the candidate is an exact expression evaluated in `f64`
/// through radicals and complex exponentials, so a few ulps of cancellation are
/// expected. The failures this gate is for are off by factors of two and whole
/// missing terms, not by `1e-9`.
const TOL: f64 = 1e-7;

/// Check that `candidate` is `e^a`.
pub(crate) fn confirm_matrix_exponential(
    a: &Matrix,
    candidate: &Matrix,
    pool: &ExprPool,
) -> Result<ExpCheck, MatrixExpRefusal> {
    let n = a.rows;
    if n != a.cols || candidate.rows != n || candidate.cols != n {
        // Not a shape this check is about; the caller's own arity errors say so.
        return Ok(ExpCheck::Unevaluated);
    }

    let mut params: Vec<ExprId> = Vec::new();
    for &e in a.entries().iter().chain(candidate.entries().iter()) {
        collect_free_symbols(e, pool, &mut params);
    }
    params.sort_by_key(|&s| pool.display(s).to_string());
    params.dedup();

    let mut checked = 0usize;
    for ps in PARAM_SETS {
        let mut env: HashMap<ExprId, ComplexF64> = HashMap::new();
        bind_constants(pool, &mut env, a, candidate);
        for (i, &p) in params.iter().enumerate() {
            env.insert(p, ComplexF64::new(ps[i % ps.len()], 0.0));
        }
        let Some(av) = eval_grid(a, pool, &env) else {
            continue;
        };
        let Some(cv) = eval_grid(candidate, pool, &env) else {
            continue;
        };
        let Some(truth) = cmat_exp(&av) else {
            continue;
        };
        let scale = cmat_norm(&truth).max(cmat_norm(&cv)).max(1.0);
        for i in 0..n {
            for j in 0..n {
                let diff = cabs(ComplexF64::new(
                    cv[i][j].re - truth[i][j].re,
                    cv[i][j].im - truth[i][j].im,
                ));
                if diff > TOL * scale {
                    return Err(MatrixExpRefusal {
                        reason: ExpRefusalReason::FailedVerification,
                        detail: format!(
                            "the closed-form e^A disagrees with e^A computed by scaling and \
                             squaring: at entry ({i}, {j}) the closed form is {} + {}i and the \
                             independent value is {} + {}i (matrix norm {scale:.6}). Refusing \
                             rather than returning a matrix that is not the exponential it \
                             claims to be",
                            cv[i][j].re, cv[i][j].im, truth[i][j].re, truth[i][j].im,
                        ),
                    });
                }
            }
        }
        checked += 1;
    }

    if checked == 0 {
        Ok(ExpCheck::Unevaluated)
    } else {
        Ok(ExpCheck::Confirmed { samples: checked })
    }
}

fn eval_grid(m: &Matrix, pool: &ExprPool, env: &HashMap<ExprId, ComplexF64>) -> Option<Cm> {
    let n = m.rows;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            let v = eval_complex_f64(m.get(i, j), pool, env).ok()?;
            if !cfinite(v) {
                return None;
            }
            row.push(v);
        }
        out.push(row);
    }
    Some(out)
}

/// Bind the named constants the eigen formulas emit but that
/// [`eval_complex_f64`] has no value for.
///
/// `pi` is a plain symbol in this crate (see [`crate::eval::symbols`]), so
/// binding it here is what stops it being collected as a free *parameter* and
/// given a sample value — which would turn a correct answer into a spurious
/// refusal. The imaginary unit is deliberately *not* bound: `eval_complex_f64`
/// knows it natively.
fn bind_constants(
    pool: &ExprPool,
    env: &mut HashMap<ExprId, ComplexF64>,
    a: &Matrix,
    candidate: &Matrix,
) {
    let mut all: Vec<ExprId> = Vec::new();
    for &e in a.entries().iter().chain(candidate.entries().iter()) {
        collect_named_constants(e, pool, &mut all);
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

    fn ints(rows: &[&[i32]], pool: &ExprPool) -> Matrix {
        Matrix::new(
            rows.iter()
                .map(|r| r.iter().map(|&v| pool.integer(v)).collect())
                .collect(),
        )
        .expect("square")
    }

    #[test]
    fn the_identity_of_the_zero_matrix_is_confirmed() {
        let p = ExprPool::new();
        let z = ints(&[&[0, 0], &[0, 0]], &p);
        let out = confirm_matrix_exponential(&z, &Matrix::identity(2, &p), &p).unwrap();
        assert!(out.is_confirmed(), "{out:?}");
    }

    #[test]
    fn the_old_wrong_nilpotent_answer_is_refuted() {
        // The 3.10.0 defect: exp([[0,1],[0,0]]) returned I. The truth is
        // [[1,1],[0,1]], so the gate must refuse the identity here.
        let p = ExprPool::new();
        let n = ints(&[&[0, 1], &[0, 0]], &p);
        let err = confirm_matrix_exponential(&n, &Matrix::identity(2, &p), &p)
            .expect_err("I is not exp([[0,1],[0,0]])");
        assert_eq!(err.reason(), ExpRefusalReason::FailedVerification);
        assert_eq!(
            <MatrixExpRefusal as crate::errors::AlkahestError>::code(&err),
            "E-LINALG-011"
        );
    }

    #[test]
    fn the_old_off_by_two_defective_answer_is_refuted() {
        // exp([[2,1],[0,2]]) returned [[e², 2e²],[0, e²]]; the truth is
        // [[e², e²],[0, e²]].
        let p = ExprPool::new();
        let a = ints(&[&[2, 1], &[0, 2]], &p);
        let e2 = p.func("exp", vec![p.integer(2_i32)]);
        let two_e2 = p.mul(vec![p.integer(2_i32), e2]);
        let z = p.integer(0_i32);
        let wrong = Matrix::new(vec![vec![e2, two_e2], vec![z, e2]]).unwrap();
        let err = confirm_matrix_exponential(&a, &wrong, &p)
            .expect_err("the off-diagonal is e², not 2e²");
        assert_eq!(err.reason(), ExpRefusalReason::FailedVerification);
    }

    #[test]
    fn an_unevaluable_candidate_is_no_information_not_a_refusal() {
        let p = ExprPool::new();
        let a = ints(&[&[1, 0], &[0, 2]], &p);
        // `zeta` has no entry in the complex evaluator.
        let opaque = p.func("zeta", vec![p.integer(3_i32)]);
        let cand = Matrix::new(vec![vec![opaque, opaque], vec![opaque, opaque]]).unwrap();
        let out = confirm_matrix_exponential(&a, &cand, &p).unwrap();
        assert_eq!(out, ExpCheck::Unevaluated);
    }

    #[test]
    fn pi_is_a_constant_not_a_sampled_parameter() {
        let p = ExprPool::new();
        let pi = p.symbol("pi", Domain::Real);
        let z = p.integer(0_i32);
        let one = p.integer(1_i32);
        // A = diag(π, 0); e^A = diag(e^π, 1).
        let a = Matrix::new(vec![vec![pi, z], vec![z, z]]).unwrap();
        let epi = p.func("exp", vec![pi]);
        let cand = Matrix::new(vec![vec![epi, z], vec![z, one]]).unwrap();
        let out = confirm_matrix_exponential(&a, &cand, &p).unwrap();
        assert!(out.is_confirmed(), "π must be bound to 3.14159…: {out:?}");
    }
}
