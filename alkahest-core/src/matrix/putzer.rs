//! `e^{At}` by **Putzer's algorithm** — the one matrix-exponential kernel.
//!
//! ```text
//! e^{At} = Σ_{k=1}^{n} r_k(t)·P_{k−1},
//!     P_0 = I,  P_k = (A − λ_k I)·P_{k−1},
//!     r_1' = λ_1 r_1,          r_1(0) = 1,
//!     r_{k+1}' = λ_{k+1} r_{k+1} + r_k,  r_{k+1}(0) = 0.
//! ```
//!
//! # Why Putzer rather than Jordan
//!
//! It needs the eigenvalues **and nothing else** — no eigenvectors, no
//! nullspaces, no similarity transform. That matters twice over:
//!
//! * A **defective** `A` costs nothing extra. For `[[2,1],[0,2]]` the
//!   eigenvalue list is `2, 2`, the recurrence gives `r_1 = e^{2t}`,
//!   `r_2 = t·e^{2t}`, and `e^{At} = e^{2t}I + t·e^{2t}(A − 2I)` is
//!   `[[e^{2t}, t·e^{2t}], [0, e^{2t}]]` — the Jordan answer, with no Jordan
//!   machinery and no `jordan_form` refusal to route around.
//! * Every eigenvector routine in the crate has to decide whether a candidate
//!   pivot vanishes, and over symbolic entries that question is undecidable
//!   (see [`crate::matrix::zero_test`]). Putzer moves the only such question to
//!   one place — `λ_i − λ_j`, in the recurrence for `r_{k+1}` — where it can be
//!   reported instead of guessed.
//!
//! `r_k` is kept as an explicit `Σ c·t^j·e^{λt}` list rather than being handed
//! to the integration engine, so the recurrence is exact term by term and the
//! `λ_i = λ_j` confluence is a visible branch rather than a failed integral.
//!
//! # Why the gap question is the caller's
//!
//! Both callers divide by an undecided `λ_i − λ_j` — the generic expansion is a
//! correct answer on `λ_i ≠ λ_j`, and an open condition is reported rather than
//! refused — but they have different places to *put* the hypothesis, and a
//! hypothesis with nowhere to go is the thing that must not happen:
//!
//! * [`crate::ode::dsolve`] has a `side_conditions` field on its solution type
//!   and puts `ka − ke ≠ 0` there, with a note naming the confluent limit.
//! * [`crate::matrix::matrix_exponential`] returns a bare `Matrix`, so its
//!   hypotheses travel out of band through
//!   [`crate::matrix::take_matrix_exp_side_conditions`].
//!
//! So the algorithm asks, through [`GapPolicy`], and does not decide. Both
//! policies treat a *settled* gap — a literal, or a constant the zero test
//! proves non-zero — as needing no hypothesis at all, which is why a rational
//! spectrum reports nothing.
//!
//! # History
//!
//! This module is the former `ode::dsolve::system` Putzer expansion, lifted
//! here after `matrix_exponential`'s Jordan route was found to be wrong for
//! every defective matrix — `e^{J}` was built as `e^λ·λ^k/k!` instead of
//! `e^λ/k!`, so `exp([[0,1],[0,0]])` was the identity. Two implementations of
//! `e^{At}` in one crate, one of them wrong, is the situation this file exists
//! to end.

use crate::kernel::{ExprId, ExprPool};
use crate::matrix::zero_test::{settled_nonzero, zero_status, ZeroStatus};
use crate::matrix::Matrix;
use crate::simplify::engine::{expand_powers, simplify_expanded};

fn simp(e: ExprId, pool: &ExprPool) -> ExprId {
    simplify_expanded(e, pool).value
}

fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let inv_b = pool.pow(b, pool.integer(-1_i32));
    simp(pool.mul(vec![a, inv_b]), pool)
}

// ---------------------------------------------------------------------------
// The gap policy
// ---------------------------------------------------------------------------

/// What the caller decided about the eigenvalue gap `λ − μ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gap {
    /// `λ = μ`. The confluent branch,
    /// `∫₀ᵗ e^{μ(t−s)} s^j e^{λs} ds = t^{j+1}/(j+1)·e^{μt}`, which needs no
    /// division at all.
    Confluent,
    /// `λ ≠ μ`, and this is the difference the generic branch divides by.
    ///
    /// The caller returns the exact expression it wants used as the divisor,
    /// because the spelling matters: a gap normalised one way cancels later and
    /// the same gap normalised another way does not.
    Distinct(ExprId),
}

/// Decides the one undecidable question Putzer asks.
pub(crate) trait GapPolicy {
    /// Classify `λ − μ` for the `r_{k+1}` recurrence.
    fn classify(&mut self, lambda: ExprId, mu: ExprId, pool: &ExprPool) -> Gap;
}

/// The policy for [`crate::matrix::matrix_exponential`]: proceed on an
/// undecided gap, but **record** it, and hand the caller the list.
///
/// An open condition is recorded and returned; a condition settled the wrong
/// way is a refusal. `λ_i − λ_j` for symbolic eigenvalues is open — the
/// oscillator `[[0,1],[−ω²,0]]` has distinct eigenvalues for every `ω ≠ 0` and
/// is defective at `ω = 0` — so the generic expansion is a correct answer on a
/// domain, and the domain is what has to travel with it.
///
/// "Settled non-zero" is deliberately stronger than [`ZeroStatus::NonZero`],
/// which for an expression mentioning a free symbol means only "not identically
/// zero as a function of it": `ζ² − 1` qualifies and still vanishes at `ζ = 1`.
/// That is exactly the difference between a gap that needs no condition and one
/// that does.
#[derive(Default)]
pub(crate) struct RecordedGaps {
    /// Gaps divided by without being settled, de-duplicated, in order.
    pub(crate) assumed_nonzero: Vec<ExprId>,
}

impl GapPolicy for RecordedGaps {
    fn classify(&mut self, lambda: ExprId, mu: ExprId, pool: &ExprPool) -> Gap {
        let d = normalised_gap(lambda, mu, pool);
        if lambda == mu || matches!(zero_status(pool, d), ZeroStatus::Zero) {
            return Gap::Confluent;
        }
        if !settled_nonzero(d, pool) && !self.assumed_nonzero.contains(&d) {
            self.assumed_nonzero.push(d);
        }
        Gap::Distinct(d)
    }
}

/// `λ − μ`, in the spelling the zero test is most likely to settle.
///
/// Every quantity whose *vanishing* is going to be tested goes through
/// [`expand_powers`] first, so the zero test and the assumption matcher see one
/// spelling rather than two.
pub(crate) fn normalised_gap(lambda: ExprId, mu: ExprId, pool: &ExprPool) -> ExprId {
    let neg_mu = pool.mul(vec![pool.integer(-1_i32), mu]);
    expand_powers(simp(pool.add(vec![lambda, neg_mu]), pool), pool)
}

// ---------------------------------------------------------------------------
// The recurrence
// ---------------------------------------------------------------------------

/// One `coeff · t^power · e^{lambda·t}` summand.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpTerm {
    coeff: ExprId,
    power: usize,
    lambda: ExprId,
}

/// `r_1 … r_n` of Putzer's recurrence, as explicit exponential-polynomial sums.
///
/// `r_{k+1}(t) = ∫₀ᵗ e^{μ(t−s)} r_k(s) ds` with `μ = λ_{k+1}`, evaluated in
/// closed form per summand: with `d = λ − μ`,
///
/// ```text
/// ∫₀ᵗ e^{μ(t−s)} s^j e^{λs} ds
///   = t^{j+1}/(j+1) · e^{μt}                                      (d = 0)
///   = Σ_{i=0}^{j} (−1)^i j!/(j−i)! · t^{j−i}/d^{i+1} · e^{λt}
///     − (−1)^j j!/d^{j+1} · e^{μt}                                (d ≠ 0)
/// ```
///
/// `None` for an empty `lambdas`: `r_1` has no `λ_1` to be built from, and an
/// `n = 0` matrix is the caller's own arity error to report.
pub(crate) fn putzer_r(
    lambdas: &[ExprId],
    policy: &mut dyn GapPolicy,
    pool: &ExprPool,
) -> Option<Vec<Vec<ExpTerm>>> {
    let mut rs: Vec<Vec<ExpTerm>> = Vec::with_capacity(lambdas.len());
    rs.push(vec![ExpTerm {
        coeff: pool.integer(1_i32),
        power: 0,
        lambda: *lambdas.first()?,
    }]);
    for &mu in &lambdas[1..] {
        let prev = rs.last().expect("r_1 was pushed before the loop").clone();
        let mut next: Vec<ExpTerm> = Vec::new();
        for term in prev {
            let d = match policy.classify(term.lambda, mu, pool) {
                Gap::Confluent => {
                    next.push(ExpTerm {
                        coeff: div(term.coeff, pool.integer((term.power + 1) as i64), pool),
                        power: term.power + 1,
                        lambda: mu,
                    });
                    continue;
                }
                Gap::Distinct(d) => d,
            };
            let j = term.power;
            let mut falling = 1_i64; // j!/(j−i)!
            for i in 0..=j {
                if i > 0 {
                    falling *= (j - i + 1) as i64;
                }
                let sign = if i % 2 == 0 { 1_i64 } else { -1_i64 };
                let denom = pool.pow(d, pool.integer((i + 1) as i64));
                let coeff = div(
                    pool.mul(vec![term.coeff, pool.integer(sign * falling)]),
                    denom,
                    pool,
                );
                next.push(ExpTerm {
                    coeff,
                    power: j - i,
                    lambda: term.lambda,
                });
            }
            // The `s = 0` boundary term, carrying `e^{μt}`.
            let jfact: i64 = (1..=(j as i64)).product::<i64>().max(1);
            let sign = if j % 2 == 0 { -1_i64 } else { 1_i64 };
            let denom = pool.pow(d, pool.integer((j + 1) as i64));
            next.push(ExpTerm {
                coeff: div(
                    pool.mul(vec![term.coeff, pool.integer(sign * jfact)]),
                    denom,
                    pool,
                ),
                power: 0,
                lambda: mu,
            });
        }
        rs.push(merge(next, pool));
    }
    Some(rs)
}

/// Add together summands with the same `(power, lambda)`.
fn merge(terms: Vec<ExpTerm>, pool: &ExprPool) -> Vec<ExpTerm> {
    let mut out: Vec<ExpTerm> = Vec::new();
    for t in terms {
        if let Some(slot) = out
            .iter_mut()
            .find(|o| o.power == t.power && o.lambda == t.lambda)
        {
            slot.coeff = simp(pool.add(vec![slot.coeff, t.coeff]), pool);
        } else {
            out.push(t);
        }
    }
    out.retain(|t| !matches!(zero_status(pool, t.coeff), ZeroStatus::Zero));
    out
}

pub(crate) fn exp_poly_to_expr(terms: &[ExpTerm], t: ExprId, pool: &ExprPool) -> ExprId {
    let mut sum = Vec::with_capacity(terms.len());
    for term in terms {
        let mut factors = vec![term.coeff];
        if term.power > 0 {
            factors.push(pool.pow(t, pool.integer(term.power as i64)));
        }
        if !matches!(zero_status(pool, term.lambda), ZeroStatus::Zero) {
            let lt = simp(pool.mul(vec![term.lambda, t]), pool);
            factors.push(pool.func("exp", vec![lt]));
        }
        sum.push(pool.mul(factors));
    }
    simp(pool.add(sum), pool)
}

// ---------------------------------------------------------------------------
// The expansion
// ---------------------------------------------------------------------------

/// `e^{At}` for `A` with eigenvalues `lambdas` **listed with multiplicity**.
///
/// `None` only for an empty `lambdas`. Does *not* check that
/// `lambdas` really is the spectrum of `a`: that is
/// [`crate::matrix::spectrum::confirm_spectrum`]'s job, and both callers run it
/// first — a wrong eigenvalue list turns into a page-long candidate here rather
/// than into an error, which is the slowest possible way to find out.
pub(crate) fn matrix_exponential(
    a: &Matrix,
    lambdas: &[ExprId],
    t: ExprId,
    policy: &mut dyn GapPolicy,
    pool: &ExprPool,
) -> Option<Matrix> {
    let n = a.rows;
    let rs = putzer_r(lambdas, policy, pool)?;
    // P_0 = I, P_k = (A − λ_k I) P_{k−1}.
    let mut p = Matrix::identity(n, pool);
    let mut acc = Matrix::zeros(n, n, pool);
    for (k, r) in rs.iter().enumerate() {
        let r_expr = exp_poly_to_expr(r, t, pool);
        for i in 0..n {
            for j in 0..n {
                acc.set(
                    i,
                    j,
                    pool.add(vec![acc.get(i, j), pool.mul(vec![r_expr, p.get(i, j)])]),
                );
            }
        }
        if k + 1 < n {
            p = mat_mul_shift(a, &p, lambdas[k], pool);
        }
    }
    for i in 0..n {
        for j in 0..n {
            acc.set(i, j, simp(acc.get(i, j), pool));
        }
    }
    Some(acc)
}

/// `(A − λI)·P`.
fn mat_mul_shift(a: &Matrix, p: &Matrix, lam: ExprId, pool: &ExprPool) -> Matrix {
    let n = a.rows;
    let mut out = Matrix::zeros(n, n, pool);
    for i in 0..n {
        for j in 0..n {
            let mut terms = Vec::with_capacity(n + 1);
            for k in 0..n {
                let mut aik = a.get(i, k);
                if i == k {
                    aik = pool.add(vec![aik, pool.mul(vec![pool.integer(-1_i32), lam])]);
                }
                terms.push(pool.mul(vec![aik, p.get(k, j)]));
            }
            out.set(i, j, simp(pool.add(terms), pool));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{eval_complex_f64, ComplexF64};
    use std::collections::HashMap;

    fn num(m: &Matrix, i: usize, j: usize, pool: &ExprPool) -> ComplexF64 {
        eval_complex_f64(m.get(i, j), pool, &HashMap::new()).expect("numeric entry")
    }

    fn ints(rows: &[&[i32]], pool: &ExprPool) -> Matrix {
        Matrix::new(
            rows.iter()
                .map(|r| r.iter().map(|&v| pool.integer(v)).collect())
                .collect(),
        )
        .expect("square")
    }

    #[test]
    fn defective_block_gets_the_t_e_lambda_t_entry() {
        // e^{At} for A = [[2,1],[0,2]] is [[e^{2t}, t e^{2t}],[0, e^{2t}]];
        // at t = 1 the off-diagonal is e², not 2e² and not 0.
        let p = ExprPool::new();
        let a = ints(&[&[2, 1], &[0, 2]], &p);
        let two = p.integer(2_i32);
        let mut policy = RecordedGaps::default();
        let e = matrix_exponential(&a, &[two, two], p.integer(1_i32), &mut policy, &p)
            .expect("rational spectrum");
        let e2 = std::f64::consts::E * std::f64::consts::E;
        assert!((num(&e, 0, 0, &p).re - e2).abs() < 1e-12);
        assert!(
            (num(&e, 0, 1, &p).re - e2).abs() < 1e-12,
            "{:?}",
            num(&e, 0, 1, &p)
        );
        assert!(num(&e, 1, 0, &p).re.abs() < 1e-12);
        assert!((num(&e, 1, 1, &p).re - e2).abs() < 1e-12);
        assert!(
            policy.assumed_nonzero.is_empty(),
            "a repeated rational eigenvalue is the confluent branch; nothing is divided by"
        );
    }

    #[test]
    fn a_settled_gap_records_no_hypothesis() {
        // λ = {1, 2}: the gap is the literal −1, so there is nothing to assume.
        let p = ExprPool::new();
        let a = ints(&[&[1, 0], &[0, 2]], &p);
        let mut policy = RecordedGaps::default();
        matrix_exponential(
            &a,
            &[p.integer(1_i32), p.integer(2_i32)],
            p.integer(1_i32),
            &mut policy,
            &p,
        )
        .expect("distinct rational spectrum");
        assert!(policy.assumed_nonzero.is_empty());
    }

    #[test]
    fn an_undecided_gap_is_recorded_rather_than_assumed_silently() {
        // λ = {a, b} for free symbols a, b: `a − b` is not identically zero, so
        // the generic branch is a correct answer — but only for `a ≠ b`, and
        // that is the part a caller cannot see unless it is stated.
        let p = ExprPool::new();
        let a = p.symbol("a", crate::kernel::Domain::Real);
        let b = p.symbol("b", crate::kernel::Domain::Real);
        let z = p.integer(0_i32);
        let m = Matrix::new(vec![vec![a, p.integer(1_i32)], vec![z, b]]).unwrap();
        let mut policy = RecordedGaps::default();
        matrix_exponential(&m, &[a, b], p.integer(1_i32), &mut policy, &p)
            .expect("the generic branch");
        assert_eq!(
            policy.assumed_nonzero.len(),
            1,
            "a − b was divided by and is not settled non-zero"
        );
    }
}
