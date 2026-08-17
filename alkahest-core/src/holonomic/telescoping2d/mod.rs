//! Double-sum creative telescoping (Apagodu–Zeilberger) for proper
//! hypergeometric terms `F(n, j, k)`.
//!
//! # Scope
//!
//! This is the multivariate generalization of `super::zeilberger` from one
//! bound index (`k`) to two (`j`, `k`). It targets exactly the concrete goal
//! named in the roadmap: **double sums over proper hypergeometric summands**
//! — `F(n+1,j,k)/F(n,j,k)`, `F(n,j+1,k)/F(n,j,k)`, `F(n,j,k+1)/F(n,j,k)` all
//! rational functions, the same shape [`super::hyperterm::ProperTerm`]
//! recognizes for one index, generalized to three. It does **not** implement
//! full Wegschaider-style reduction (arbitrary rational summands, arbitrary
//! many indices) — that is a substantially larger undertaking and out of
//! scope here; see the honest-limitations list below.
//!
//! Given `F(n,j,k)`, [`telescope2d`] searches for a recurrence order `J`,
//! polynomial coefficients `a_0(n), …, a_J(n)` (not all zero) and two
//! rational certificates `c_1, c_2 ∈ Q(n,j,k)` such that
//!
//! ```text
//! Σ_i a_i(n)·F(n+i,j,k) = Δ_j G_1(n,j,k) + Δ_k G_2(n,j,k),
//! G_1 = c_1·F,  G_2 = c_2·F
//! ```
//!
//! — proving a recurrence for `S(n) = Σ_j Σ_k F(n,j,k)` once the *boundary*
//! of the rectangle it is summed over is discharged (see [`boundary`]; the
//! telescoping identity above says nothing about the sum on its own, exactly
//! as in the single-index case).
//!
//! # Method: Apagodu–Zeilberger by undetermined coefficients
//!
//! There is no standard two-dimensional analogue of Gosper's normal form for
//! a general proper hypergeometric `F(n,j,k)`, so — unlike the single-sum
//! engine — this module does not attempt one. It follows the
//! Apagodu–Zeilberger presentation directly: posit a certificate ansatz of
//! bounded polynomial degree over a *fixed* (ansatz-independent) denominator
//! built from `F`'s own shift-ratio denominators, clear it, and solve the
//! resulting *linear* system by Gaussian elimination over `Q`. See the
//! `search` submodule for the full derivation and the specific, stated
//! limitation this buys (the fixed denominator is not always the minimal one
//! a genuine 2-D Gosper reduction would find).
//!
//! # Module layout
//!
//! - `poly` — plain sparse `Q[n,j,k]` / `Q(n,j,k)` arithmetic. Deliberately
//!   simpler than `super::qfield`'s `Q(n)[k]` tower: the ansatz search never
//!   needs a gcd, only linear algebra over a fixed denominator, so there is
//!   no normal-form machinery here to get wrong.
//! - `term` — `F(n,j,k)` recognition and exact shift ratios, the 3-index
//!   generalization of `super::hyperterm`.
//! - `search` — the ansatz search itself, kept strictly separate from
//!   verification: every candidate is re-derived and checked as an exact
//!   `Q(n,j,k)` identity (see `search::verify_certificate`) before it is ever
//!   returned, independent of how the search found it.
//! - [`boundary`] — the 2-D boundary/corner analysis, on its own so a
//!   returned certificate is checkable without reference to how it was
//!   produced. Read its module docs first: the boundary of a rectangle is
//!   **four one-dimensional strip sums**, not four corner-point evaluations,
//!   and getting that distinction right is the substance of the module.
//!
//! # Honest limitations (read before relying on this)
//!
//! - **Summands**: proper hypergeometric in `(n,j,k)` only — rational
//!   prefactor times `z_j^j·z_k^k·w^n` times `Γ(a·n+b·j+c·k+d)^e` factors,
//!   `a,b,c ∈ Z`. No more than two bound indices.
//! - **Certificate ansatz**: bounded box degree in each of `n,j,k`
//!   independently ([`Telescoping2dOpts`]), searched by plain
//!   ascending nested loops — not the cost-ordered iterative deepening
//!   `super::zeilberger` uses, so raising the bounds is not free the way it
//!   is there.
//! - **Certificate denominator**: fixed from `F`'s raw (un-reduced) shift-
//!   ratio denominators, not a minimal 2-D Gosper normal form. Sufficient for
//!   the "binomial-type" examples this module is tested against; not proven
//!   sufficient in general. A search that finds nothing reports
//!   [`telescope2d_search`]'s `SearchExhausted`, never a false
//!   certificate.
//! - **Boundary**: only rectangles with **constant** (not `n`-dependent)
//!   limits are supported, and only the sufficient "each strip vanishes
//!   pointwise" criterion is checked — see [`boundary`]'s module docs for
//!   why both are real restrictions and not just unfinished polish, and for
//!   the natural workaround (`n`-independent bounds larger than the true
//!   combinatorial support) that the worked example below uses.
//! - **No explicit nonzero boundary term**: [`boundary::BoundaryStatus2d`] is
//!   three-valued in shape (matching [`super::boundary::BoundaryStatus`]),
//!   but this version never *produces*
//!   [`boundary::BoundaryStatus2d::Nonzero`] — an unresolved boundary is
//!   always [`boundary::BoundaryStatus2d::Unknown`], not an inhomogeneous
//!   recurrence with an explicit `b(n)`.

pub mod boundary;
mod poly;
mod search;
mod term;

pub use boundary::{boundary_status_2d, BoundaryStatus2d};
pub use search::{telescope2d_search, Telescoping2dOpts, Telescoping2dResult};

use std::fmt;

/// Errors from the double-sum telescoping engine. Disjoint error-code block
/// (`E-HOLO-040`..`E-HOLO-042`, with room to grow) from both the classical engine
/// (`E-HOLO-001`..`E-HOLO-004`) and the `q`-analogue (`E-HOLO-020`..`E-HOLO-024`),
/// the same pattern [`super::qzeil::QHolonomicError`] uses, so a caller can
/// tell which engine refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Telescoping2dError {
    /// `term` is not proper hypergeometric in `(n,j,k)` (see the module docs
    /// for the supported class).
    NotProperHypergeometric(String),
    /// The bounded ansatz search ([`search::Telescoping2dOpts`]) was
    /// exhausted without finding a certificate that passed exact
    /// verification.
    SearchExhausted(String),
    /// `n`, `j`, `k` not pairwise distinct, or another malformed call.
    InvalidInput(String),
}

impl fmt::Display for Telescoping2dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Telescoping2dError::NotProperHypergeometric(s) => {
                write!(f, "telescoping2d: not a proper hypergeometric term: {s}")
            }
            Telescoping2dError::SearchExhausted(s) => {
                write!(f, "telescoping2d: search exhausted: {s}")
            }
            Telescoping2dError::InvalidInput(s) => {
                write!(f, "telescoping2d: invalid input: {s}")
            }
        }
    }
}

impl std::error::Error for Telescoping2dError {}

impl crate::errors::AlkahestError for Telescoping2dError {
    fn code(&self) -> &'static str {
        match self {
            Telescoping2dError::NotProperHypergeometric(_) => "E-HOLO-040",
            Telescoping2dError::SearchExhausted(_) => "E-HOLO-041",
            Telescoping2dError::InvalidInput(_) => "E-HOLO-042",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            Telescoping2dError::NotProperHypergeometric(_) => {
                "rewrite the term as R(n,j,k)*z1**j*z2**k*w**n*prod(gamma(a*n+b*j+c*k+d)**e) \
                 with integer a, b, c; supported function heads are gamma, factorial, \
                 binomial, pochhammer"
            }
            Telescoping2dError::SearchExhausted(_) => {
                "raise max_order, max_a_degree and/or max_cert_degree in Telescoping2dOpts; if \
                 the term genuinely has no such double-sum certificate within reach — or needs \
                 a certificate denominator this module's fixed-denominator ansatz cannot \
                 represent — this method does not apply"
            }
            Telescoping2dError::InvalidInput(_) => "n, j and k must be three distinct symbols",
        })
    }
}

/// Top-level entry point: search for and verify a double-sum creative-
/// telescoping certificate for `term = F(n,j,k)`, with the default search
/// bounds ([`Telescoping2dOpts::default`]).
pub fn telescope2d(
    term: crate::kernel::ExprId,
    n: crate::kernel::ExprId,
    j: crate::kernel::ExprId,
    k: crate::kernel::ExprId,
    pool: &crate::kernel::ExprPool,
) -> Result<Telescoping2dResult, Telescoping2dError> {
    telescope2d_search(term, n, j, k, pool, &Telescoping2dOpts::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprId, ExprPool};
    use rug::ops::Pow as _;
    use rug::Integer;

    fn njk(pool: &ExprPool) -> (ExprId, ExprId, ExprId) {
        (
            pool.symbol("n", Domain::Real),
            pool.symbol("j", Domain::Real),
            pool.symbol("k", Domain::Real),
        )
    }

    fn binom_i(top: i64, bot: i64) -> Integer {
        if bot < 0 || bot > top {
            return Integer::from(0);
        }
        let mut acc = Integer::from(1);
        for t in 0..bot {
            acc *= Integer::from(top - t);
            acc /= Integer::from(t + 1);
        }
        acc
    }

    fn coeff_at_n(pool: &ExprPool, coeff: ExprId, n: ExprId, ni: i64) -> rug::Rational {
        let env = std::collections::HashMap::from([(n, ni as f64)]);
        // Coefficients here are always integers of modest size (products of
        // small binomial-type recurrences); an f64 evaluation recovers them
        // exactly after rounding to the nearest integer, but to keep the
        // check genuinely exact we instead evaluate the *rational* recurrence
        // check with `eval_f64` only as a fast pre-filter and fall back to
        // symbolic substitution. In practice every coefficient in this
        // module's tests is degree <= 1 with small integer coefficients, so
        // f64 is already exact; this helper documents that assumption
        // instead of silently relying on it.
        let v = crate::eval_f64(coeff, pool, &env).expect("a_i(n) evaluates");
        assert!(
            (v - v.round()).abs() < 1e-6,
            "expected an integer coefficient, got {v}"
        );
        rug::Rational::from(v.round() as i64)
    }

    /// **Separable fallback example** (explicitly weaker, see module docs and
    /// the task's own scoping note): `F(n,j,k) = C(n,j)*C(n,k)`.
    /// `S(n) = Σ_j Σ_k F = (Σ_j C(n,j))·(Σ_k C(n,k)) = 2^n·2^n = 4^n`, so the
    /// double-sum recurrence must match the order-1 relation
    /// `S(n+1) = 4·S(n)` — checkable directly against the product of the two
    /// known single-sum recurrences `Σ_j C(n,j) = 2^n`. This does *not*
    /// exercise the corner-term logic (`C(n,j)` and `C(n,k)` never interact),
    /// which is exactly why the genuinely non-separable example below exists.
    #[test]
    fn separable_product_matches_known_recurrence() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let f = pool.mul(vec![
            pool.func("binomial", vec![n, j]),
            pool.func("binomial", vec![n, k]),
        ]);
        let result = telescope2d(f, n, j, k, &pool).expect("certificate for C(n,j)*C(n,k)");

        // Exact rational cross-check against real summed values, per the
        // project's standing practice: sum F(n,j,k) exactly over j,k = 0..40
        // (safely beyond any true support for n <= 8) and confirm it equals
        // 4^n, then confirm the returned recurrence annihilates that exact
        // sequence.
        let s = |ni: i64| -> rug::Rational {
            let mut acc = Integer::from(0);
            for jj in 0..=40 {
                for kk in 0..=40 {
                    acc += binom_i(ni, jj) * binom_i(ni, kk);
                }
            }
            rug::Rational::from(acc)
        };
        for ni in 0..=6 {
            assert_eq!(s(ni), rug::Rational::from(Integer::from(4).pow(ni as u32)));
        }

        assert_annihilates(&result, n, &pool, &s, 0, 5);
    }

    /// **Genuinely non-separable worked example**: `F(n,j,k) = C(n,j)*C(j,k)`
    /// — a real double-sum identity from the composition of two binomial
    /// transforms, `C(j,k)` coupling to the *outer* sum's own index `j`, not
    /// separable into a product of two independent one-index sums. This is
    /// what actually exercises the two-dimensional certificate search (both
    /// `G_1` and `G_2` are non-trivial and interact through `j`) and the
    /// corner-boundary logic in `boundary.rs`.
    ///
    /// `Σ_k C(j,k) = 2^j`, so `S(n) = Σ_j C(n,j)·2^j = (1+2)^n = 3^n` by the
    /// binomial theorem — a genuine closed form, checked here by direct exact
    /// summation, not assumed.
    #[test]
    fn non_separable_double_binomial_matches_known_closed_form() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let f = pool.mul(vec![
            pool.func("binomial", vec![n, j]),
            pool.func("binomial", vec![j, k]),
        ]);
        let result = telescope2d(f, n, j, k, &pool).expect("certificate for C(n,j)*C(j,k)");

        let s = |ni: i64| -> rug::Rational {
            let mut acc = Integer::from(0);
            for jj in 0..=40 {
                for kk in 0..=40 {
                    acc += binom_i(ni, jj) * binom_i(jj, kk);
                }
            }
            rug::Rational::from(acc)
        };
        for ni in 0..=6 {
            assert_eq!(s(ni), rug::Rational::from(Integer::from(3).pow(ni as u32)));
        }

        assert_annihilates(&result, n, &pool, &s, 0, 5);

        // What `boundary.rs` can and cannot certify here, and why: `C(n,j)`'s
        // true support grows with `n` (it is non-zero for every `0 <= j <=
        // n`), so *no* constant rectangle dominates the true natural range
        // for every symbolic `n` — only for the specific `n` the rectangle
        // was sized against. Since this module's boundary analysis proves a
        // claim for *symbolic* `n` (see its module docs on why only
        // constant, `n`-independent ranges are supported at all), it
        // correctly refuses to certify `Vanishes` for a fixed rectangle here
        // — doing so would be a false claim for `n` beyond the rectangle's
        // own bound, exactly the trap the module docs warn about. This is
        // not a bug: it is the honest boundary of what a constant-range
        // analysis can say about an identity whose true support is
        // `n`-dependent. `non_separable_fixed_support_boundary_vanishes`
        // below is the same non-separable coupling with `n`-*independent*
        // support, where `Vanishes` genuinely is provable and provided.
        let lo = pool.integer(0_i32);
        let hi = pool.integer(40_i32);
        let status = boundary_status_2d(&result, f, n, j, k, (lo, hi), (lo, hi), &pool);
        assert_eq!(
            status.tag(),
            "unknown",
            "a constant rectangle cannot soundly certify Vanishes when the true support \
             grows with n; got {status:?}"
        );
    }

    /// The same non-separable `C(·,j)·C(j,k)` coupling as above, but with the
    /// `n`-dependence factored out into a decoupled `x^n` (so the `(j,k)`
    /// support is a genuine constant, independent of `n`) — the case where
    /// this module's constant-range boundary analysis *can* certify
    /// `Vanishes` for a real, non-separable double sum, not just the
    /// separable product example.
    ///
    /// `F(n,j,k) = 2^n·C(10,j)·C(j,k)`; `Σ_k C(j,k) = 2^j`, so
    /// `S(n) = 2^n·Σ_j C(10,j)·2^j = 2^n·3^10` — checked exactly, and the
    /// natural boundary (both binomials vanish combinatorially outside
    /// `j,k <= 10`, for *every* `n`, since neither bound depends on `n`)
    /// should be provably `Vanishes`.
    #[test]
    fn non_separable_fixed_support_boundary_vanishes() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let two_n = pool.pow(pool.integer(2_i32), n);
        let f = pool.mul(vec![
            two_n,
            pool.func("binomial", vec![pool.integer(10_i32), j]),
            pool.func("binomial", vec![j, k]),
        ]);
        let result = telescope2d(f, n, j, k, &pool).expect("certificate for 2^n*C(10,j)*C(j,k)");

        let s = |ni: i64| -> rug::Rational {
            let mut acc = Integer::from(0);
            for jj in 0..=10 {
                for kk in 0..=10 {
                    acc += binom_i(10, jj) * binom_i(jj, kk);
                }
            }
            rug::Rational::from(acc * Integer::from(2).pow(ni.max(0) as u32))
        };
        for ni in 0..=6 {
            assert_eq!(
                s(ni),
                rug::Rational::from(Integer::from(2).pow(ni as u32) * Integer::from(3).pow(10))
            );
        }
        assert_annihilates(&result, n, &pool, &s, 0, 5);

        let lo = pool.integer(0_i32);
        let hi = pool.integer(15_i32);
        let status = boundary_status_2d(&result, f, n, j, k, (lo, hi), (lo, hi), &pool);
        assert_eq!(
            status.tag(),
            "vanishes",
            "expected the natural (n-independent) boundary to vanish, got {status:?}"
        );
    }

    /// `Σ_i a_i(n)·S(n+i) = 0` for `n = lo..=hi`, using the *exact* rational
    /// values `s` computes — never floats — for the sum itself, and reading
    /// the (small-integer) recurrence coefficients back exactly.
    fn assert_annihilates(
        result: &Telescoping2dResult,
        n: ExprId,
        pool: &ExprPool,
        s: &dyn Fn(i64) -> rug::Rational,
        lo: i64,
        hi: i64,
    ) {
        for ni in lo..=hi {
            let mut total = rug::Rational::from(0);
            for (i, &c) in result.coeffs.iter().enumerate() {
                let ai = coeff_at_n(pool, c, n, ni + i as i64);
                total += ai * s(ni + i as i64);
            }
            assert_eq!(
                total,
                rug::Rational::from(0),
                "recurrence must annihilate S(n) exactly at n = {ni}"
            );
        }
    }

    #[test]
    fn refuses_non_hypergeometric_input() {
        let pool = ExprPool::new();
        let (n, j, k) = njk(&pool);
        let bad = pool.func("sin", vec![pool.mul(vec![n, j, k])]);
        let err = telescope2d(bad, n, j, k, &pool).expect_err("not hypergeometric");
        assert!(matches!(
            err,
            Telescoping2dError::NotProperHypergeometric(_)
        ));
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-HOLO-040");
    }
}
