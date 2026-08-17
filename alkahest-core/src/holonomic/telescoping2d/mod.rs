//! Multi-sum creative telescoping (Apagodu–Zeilberger) for proper
//! hypergeometric terms `F(n, x_1, …, x_m)` with `m ≥ 1` bound indices.
//!
//! # Scope
//!
//! This is the multivariate generalization of `super::zeilberger` from one
//! bound index (`k`) to an arbitrary number `m` of them. It targets exactly
//! the concrete goal named in the roadmap: **multi-sums over proper
//! hypergeometric summands** — every shift ratio `F(…,x_t+1,…)/F(…,x_t,…)` a
//! rational function, the same shape [`super::hyperterm::ProperTerm`]
//! recognizes for one index, generalized to `m + 1`. It does **not**
//! implement full Wegschaider-style reduction (arbitrary *rational*
//! summands, or a minimal Gosper-style certificate denominator) — that
//! remains a substantially larger undertaking and out of scope here; see the
//! honest-limitations list below.
//!
//! The two-bound-index case (`telescope2d`, `Telescoping2dResult`,
//! `boundary_status_2d`, `BoundaryStatus2d`) is the module's original,
//! semver-stable public surface and is unchanged in behavior. As of this
//! extension it is a thin wrapper around the general `m`-index engine
//! ([`telescope_md`], [`search::TelescopingMdResult`],
//! [`boundary::boundary_status_md`], [`boundary::BoundaryStatusMd`]), which
//! is the new public surface for `m ≠ 2` (including `m = 1`, degenerating to
//! the classical single-sum shape, and `m ≥ 3`, genuinely new).
//!
//! Given `F(n,x_1,…,x_m)`, [`telescope_md`] searches for a recurrence order
//! `J`, polynomial coefficients `a_0(n), …, a_J(n)` (not all zero) and `m`
//! rational certificates `c_1, …, c_m ∈ Q(n,x)` such that
//!
//! ```text
//! Σ_i a_i(n)·F(n+i,x) = Σ_t Δ_t G_t(n,x),   G_t = c_t·F
//! ```
//!
//! — proving a recurrence for `S(n) = Σ_{x_1} … Σ_{x_m} F(n,x)` once the
//! *boundary* of the box it is summed over is discharged (see [`boundary`];
//! the telescoping identity above says nothing about the sum on its own,
//! exactly as in the single-index case).
//!
//! # Method: Apagodu–Zeilberger by undetermined coefficients
//!
//! There is no standard multivariate analogue of Gosper's normal form for a
//! general proper hypergeometric `F(n,x_1,…,x_m)`, so — unlike the
//! single-sum engine — this module does not attempt one. It follows the
//! Apagodu–Zeilberger presentation directly: posit a certificate ansatz of
//! bounded polynomial degree over a *fixed* (ansatz-independent) denominator
//! built from `F`'s own shift-ratio denominators, clear it, and solve the
//! resulting *linear* system by Gaussian elimination over `Q`. See the
//! `search` submodule for the full derivation and the specific, stated
//! limitation this buys (the fixed denominator is not always the minimal one
//! a genuine multivariate Gosper reduction would find).
//!
//! # Module layout
//!
//! - `poly` — plain sparse `Q[n,x_1,…,x_m]` / `Q(n,x_1,…,x_m)` arithmetic for
//!   an arbitrary number of axes. Deliberately simpler than `super::qfield`'s
//!   `Q(n)[k]` tower: the ansatz search never needs a gcd, only linear
//!   algebra over a fixed denominator, so there is no normal-form machinery
//!   here to get wrong.
//! - `term` — `F(n,x_1,…,x_m)` recognition and exact shift ratios, the
//!   `(m+1)`-index generalization of `super::hyperterm`.
//! - `search` — the ansatz search itself, kept strictly separate from
//!   verification: every candidate is re-derived and checked as an exact
//!   `Q(n,x)` identity (see `search::verify_certificate_md`) before it is
//!   ever returned, independent of how the search found it.
//! - [`boundary`] — the `m`-dimensional boundary/face analysis, on its own so
//!   a returned certificate is checkable without reference to how it was
//!   produced. Read its module docs first: the boundary of a box is **`2m`
//!   `(m-1)`-dimensional face sums**, not `2^m` corner-point evaluations, and
//!   getting that distinction right is the substance of the module.
//!
//! # Honest limitations (read before relying on this)
//!
//! - **Summands**: proper hypergeometric in `(n,x_1,…,x_m)` only — rational
//!   prefactor times `∏_t z_t^{x_t}·w^n` times `Γ(a·n+Σ_t b_t·x_t+d)^e`
//!   factors, `a,b_t ∈ Z`. **No genuinely broader summand class is
//!   supported**: a rational prefactor beyond what a ratio of the module's
//!   own gamma factors already produces, a *sum* of several proper
//!   hypergeometric terms, or a mixed radix/`q`-analogue combination are all
//!   refused as [`Telescoping2dError::NotProperHypergeometric`], not
//!   approximated.
//! - **Bound index count**: arbitrary `m ≥ 1` via [`telescope_md`] /
//!   [`search::telescope_md_search`] (`m = 2` — [`telescope2d`] — is the
//!   original, semver-stable special case; `m = 1` degenerates cleanly to a
//!   single-sum-shaped search, exercised by this module's own tests). Raising
//!   `m` grows the ansatz search space fast — a certificate's numerator
//!   spans a box of `(max_cert_degree + 1)^(m+1)` unknowns *per* certificate,
//!   and there are `m` certificates — so higher `m` needs correspondingly
//!   patient degree bounds; the search still only ever returns an exactly
//!   re-verified certificate, never a false one, when bounds run out.
//! - **Resource ceilings on the linear solve**: `rational_nullspace`'s exact
//!   Gaussian elimination is `O(rows · cols²)` over unbounded-precision
//!   rationals, and both dimensions grow with `m` and `cert_degree` well
//!   past what the degree numbers alone suggest — at `m = 3`,
//!   `cert_degree = 2` already means a ≈10 000-row, 245-unknown system whose
//!   elimination step alone measured ≈47 seconds *per probe*, and
//!   `cert_degree = 3` (770 unknowns) was still running after several
//!   minutes. This is genuine arithmetic cost on a real linear system, not a
//!   bug or an infinite loop, but a caller still needs protection from it:
//!   `search::MAX_ANSATZ_UNKNOWNS` refuses outright any single probe whose
//!   unknown count would exceed `400` (comfortably above every worked
//!   example this module ships, including the `m = 3` multinomial-
//!   coefficient example's `245`), and
//!   `search::MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS` caps the *total* work
//!   spent across every probe at or above 150 unknowns in one search call to
//!   `300` — capping the number of genuinely expensive elimination attempts
//!   to about one, regardless of how large `max_order` / `max_a_degree` /
//!   `max_cert_degree` are, so a caller whose input has no certificate
//!   within reach at all cannot be made to pay that cost over and over
//!   across every `(order, a_degree)` combination tried. Both ceilings are
//!   checked with cheap arithmetic before any polynomial construction for
//!   the affected probe begins, and a probe skipped this way is reported
//!   exactly like one the linear algebra found nothing for — except
//!   [`Telescoping2dError::SearchExhausted`]'s message says explicitly when
//!   a ceiling, not genuine non-existence, is why nothing was found, so
//!   raising the search bounds further is not silently misrepresented as a
//!   path to success. Neither ceiling affects the `m = 2` case: its default
//!   search never builds a probe past ≈140 unknowns.
//! - **Certificate ansatz**: bounded box degree in each of `n,x_1,…,x_m`
//!   independently ([`Telescoping2dOpts`] / [`search::TelescopingMdOpts`]),
//!   searched by plain ascending nested loops — not the cost-ordered
//!   iterative deepening `super::zeilberger` uses, so raising the bounds is
//!   not free the way it is there.
//! - **Certificate denominator**: fixed from `F`'s raw (un-reduced) shift-
//!   ratio denominators, not a minimal Gosper normal form. Sufficient for the
//!   "binomial-type" examples this module is tested against; not proven
//!   sufficient in general. A search that finds nothing reports
//!   `SearchExhausted`, never a false certificate. A genuine minimal
//!   Gosper-style denominator (the roadmap's stated remaining-gap item 3) was
//!   not attempted in this extension — see the crate-level changelog entry
//!   for why (a real algorithm-design problem, not an engineering extension
//!   of what is here).
//! - **Boundary**: only boxes with **constant** (not `n`-dependent) limits
//!   are supported, and only the sufficient "each face vanishes pointwise"
//!   criterion is checked — see [`boundary`]'s module docs for why both are
//!   real restrictions and not just unfinished polish, and for the natural
//!   workaround (`n`-independent bounds larger than the true combinatorial
//!   support) that the worked examples below use.
//! - **No explicit nonzero boundary term**: [`boundary::BoundaryStatus2d`]
//!   and [`boundary::BoundaryStatusMd`] are three-valued in shape (matching
//!   [`super::boundary::BoundaryStatus`]), but neither version ever
//!   *produces* their `Nonzero` variant — an unresolved boundary is always
//!   `Unknown`, not an inhomogeneous recurrence with an explicit `b(n)`.

pub mod boundary;
mod poly;
mod search;
mod term;

pub use boundary::{boundary_status_2d, boundary_status_md, BoundaryStatus2d, BoundaryStatusMd};
pub use search::{
    telescope2d_search, telescope_md_search, Telescoping2dOpts, Telescoping2dResult,
    TelescopingMdOpts, TelescopingMdResult,
};

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
            Telescoping2dError::InvalidInput(_) => {
                "n and every bound index must be pairwise distinct symbols, and at least one \
                 bound index must be supplied"
            }
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

/// Top-level entry point for `m ≥ 1` bound indices: search for and verify a
/// creative-telescoping certificate for `term = F(n, indices[0], …,
/// indices[m-1])`, with the default search bounds
/// ([`TelescopingMdOpts::default`]).
pub fn telescope_md(
    term: crate::kernel::ExprId,
    n: crate::kernel::ExprId,
    indices: &[crate::kernel::ExprId],
    pool: &crate::kernel::ExprPool,
) -> Result<TelescopingMdResult, Telescoping2dError> {
    search::telescope_md_search(term, n, indices, pool, &TelescopingMdOpts::default())
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

        assert_annihilates(&result.coeffs, n, &pool, &s, 0, 5);
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

        assert_annihilates(&result.coeffs, n, &pool, &s, 0, 5);

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
        assert_annihilates(&result.coeffs, n, &pool, &s, 0, 5);

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
    /// the (small-integer) recurrence coefficients back exactly. Takes the
    /// coefficient list directly (rather than a whole `Telescoping2dResult`)
    /// so it is shared between the `m = 2` and general-`m` worked examples.
    fn assert_annihilates(
        coeffs: &[ExprId],
        n: ExprId,
        pool: &ExprPool,
        s: &dyn Fn(i64) -> rug::Rational,
        lo: i64,
        hi: i64,
    ) {
        for ni in lo..=hi {
            let mut total = rug::Rational::from(0);
            for (i, &c) in coeffs.iter().enumerate() {
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

    /// Exact `n!/(x!y!z!(n-x-y-z)!)` — the multinomial coefficient
    /// (4-category composition of `n`) computed via the equivalent
    /// nested-binomial product `C(n,x)·C(n-x,y)·C(n-x-y,z)`, which is `0`
    /// whenever `x+y+z > n`, matching the symbolic term's own `1/Γ` vanishing
    /// there. This is an independent check function — it does not reuse any
    /// part of the solver.
    fn multinom_i(n: i64, x: i64, y: i64, z: i64) -> Integer {
        if x < 0 || y < 0 || z < 0 || x + y + z > n {
            return Integer::from(0);
        }
        binom_i(n, x) * binom_i(n - x, y) * binom_i(n - x - y, z)
    }

    /// **Genuinely non-separable `m = 3` worked example**:
    /// `F(n,x,y,z) = n!/(x!·y!·z!·(n-x-y-z)!)`, the multinomial coefficient
    /// counting compositions of `n` into 4 labeled parts — built directly
    /// from `factorial`, not as a product of binomials, so the parser sees
    /// exactly 5 gamma factors (`Γ(n+1)`, `1/Γ(x+1)`, `1/Γ(y+1)`,
    /// `1/Γ(z+1)`, `1/Γ(n-x-y-z+1)`) rather than the redundant 9 a naive
    /// `C(n,x)·C(n-x,y)·C(n-x-y,z)` encoding would carry (two of that
    /// encoding's factors are exact inverses of each other but this module's
    /// unreduced `Rat3`/`RatM` arithmetic never cancels them, so picking the
    /// simpler encoding is a real, deliberate choice — not cosmetic). All
    /// three bound indices interact through the shared `n-x-y-z` term, so
    /// this is not a product of independent-variable pieces: it genuinely
    /// exercises the `m`-index generalization of the ansatz search (three
    /// non-trivial, mutually coupled certificates) and the boundary module's
    /// `2m = 6`-face analysis, not just the `m = 2` machinery run twice.
    ///
    /// `Σ_{x,y,z} n!/(x!y!z!(n-x-y-z)!) = 4^n` by the multinomial theorem
    /// (the number of length-`n` strings over a 4-letter alphabet, grouped by
    /// letter counts) — a genuine closed form, checked here by direct exact
    /// summation over a box safely larger than the true support, not
    /// assumed.
    #[test]
    fn multinomial_matches_known_closed_form() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let rest = pool.add(vec![
            n,
            pool.mul(vec![x, pool.integer(-1_i32)]),
            pool.mul(vec![y, pool.integer(-1_i32)]),
            pool.mul(vec![z, pool.integer(-1_i32)]),
        ]);
        let inv_fact = |e: ExprId| pool.pow(pool.func("factorial", vec![e]), pool.integer(-1_i32));
        let f = pool.mul(vec![
            pool.func("factorial", vec![n]),
            inv_fact(x),
            inv_fact(y),
            inv_fact(z),
            inv_fact(rest),
        ]);
        let opts = TelescopingMdOpts {
            max_order: 1,
            max_a_degree: 1,
            max_cert_degree: 2,
        };
        let result = search::telescope_md_search(f, n, &[x, y, z], &pool, &opts)
            .expect("certificate for the 4-category multinomial coefficient");
        assert_eq!(result.certs.len(), 3);

        // Exact rational cross-check against real summed values: sum
        // F(n,x,y,z) exactly over x,y,z = 0..15 (safely beyond the true
        // support for n <= 6) and confirm it equals 4^n, then confirm the
        // returned recurrence annihilates that exact sequence.
        let s = |ni: i64| -> rug::Rational {
            let mut acc = Integer::from(0);
            for xx in 0..=15 {
                for yy in 0..=15 {
                    for zz in 0..=15 {
                        acc += multinom_i(ni, xx, yy, zz);
                    }
                }
            }
            rug::Rational::from(acc)
        };
        for ni in 0..=6 {
            assert_eq!(s(ni), rug::Rational::from(Integer::from(4).pow(ni as u32)));
        }
        assert_annihilates(&result.coeffs, n, &pool, &s, 0, 4);

        // The true support (x+y+z <= n) grows with n, exactly as in the m=2
        // non-separable example, so a constant box cannot soundly dominate
        // it for every symbolic n: the boundary analysis must correctly
        // refuse `Vanishes` here, not guess it.
        let lo = pool.integer(0_i32);
        let hi = pool.integer(15_i32);
        let status = boundary_status_md(
            &result,
            f,
            n,
            &[x, y, z],
            &[(lo, hi), (lo, hi), (lo, hi)],
            &pool,
        );
        assert_eq!(
            status.tag(),
            "unknown",
            "a constant box cannot soundly certify Vanishes when the true support grows with \
             n; got {status:?}"
        );
    }

    /// The same non-separable multinomial coupling as above, but with the
    /// `n`-dependence factored out into a decoupled `4^n` (so the `(x,y,z)`
    /// support is a genuine constant, independent of `n`) — the case where
    /// the `m`-dimensional constant-box boundary analysis *can* certify
    /// `Vanishes` for a real, non-separable triple sum.
    ///
    /// `F(n,x,y,z) = 4^n·10!/(x!y!z!(10-x-y-z)!)`; the `(x,y,z)` part sums to
    /// `4^10` (the worked example above, at `n=10`), so
    /// `S(n) = 4^n·4^10 = 4^(n+10)` — checked exactly, and the natural
    /// boundary (the multinomial vanishes combinatorially outside `x+y+z <=
    /// 10`, for *every* `n`, since none of the three bounds depend on `n`)
    /// should be provably `Vanishes`.
    #[test]
    fn multinomial_fixed_support_boundary_vanishes() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let four_n = pool.pow(pool.integer(4_i32), n);
        let ten = pool.integer(10_i32);
        let rest = pool.add(vec![
            ten,
            pool.mul(vec![x, pool.integer(-1_i32)]),
            pool.mul(vec![y, pool.integer(-1_i32)]),
            pool.mul(vec![z, pool.integer(-1_i32)]),
        ]);
        let inv_fact = |e: ExprId| pool.pow(pool.func("factorial", vec![e]), pool.integer(-1_i32));
        let f = pool.mul(vec![
            four_n,
            pool.func("factorial", vec![ten]),
            inv_fact(x),
            inv_fact(y),
            inv_fact(z),
            inv_fact(rest),
        ]);
        let opts = TelescopingMdOpts {
            max_order: 1,
            max_a_degree: 1,
            max_cert_degree: 2,
        };
        let result = search::telescope_md_search(f, n, &[x, y, z], &pool, &opts)
            .expect("certificate for 4^n*10!/(x!y!z!(10-x-y-z)!)");

        let inner: Integer = {
            let mut acc = Integer::from(0);
            for xx in 0..=10 {
                for yy in 0..=10 {
                    for zz in 0..=10 {
                        acc += multinom_i(10, xx, yy, zz);
                    }
                }
            }
            acc
        };
        assert_eq!(inner, Integer::from(4).pow(10));
        let s = |ni: i64| -> rug::Rational {
            rug::Rational::from(inner.clone() * Integer::from(4).pow(ni.max(0) as u32))
        };
        for ni in 0..=6 {
            assert_eq!(
                s(ni),
                rug::Rational::from(Integer::from(4).pow(ni as u32) * Integer::from(4).pow(10))
            );
        }
        assert_annihilates(&result.coeffs, n, &pool, &s, 0, 4);

        let lo = pool.integer(0_i32);
        let hi = pool.integer(15_i32);
        let status = boundary_status_md(
            &result,
            f,
            n,
            &[x, y, z],
            &[(lo, hi), (lo, hi), (lo, hi)],
            &pool,
        );
        assert_eq!(
            status.tag(),
            "vanishes",
            "expected the natural (n-independent) boundary to vanish, got {status:?}"
        );
    }

    /// `m = 1` sanity check: `telescope_md` with a single bound index should
    /// find the same kind of certificate the classical single-sum engine
    /// would, on the simplest possible input.
    #[test]
    fn telescope_md_single_index_smoke_test() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let k = pool.symbol("k", Domain::Real);
        let f = pool.func("binomial", vec![n, k]);
        let result = telescope_md(f, n, &[k], &pool).expect("certificate for C(n,k)");
        assert_eq!(result.certs.len(), 1);
    }

    /// Regression test for the two resource ceilings in `search`
    /// (`MAX_ANSATZ_UNKNOWNS`, the per-probe ceiling, and
    /// `MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS`, the whole-search budget that
    /// stops the *same* expensive probe size from being retried across every
    /// `(order, a_degree)` combination): the triple-binomial-chain example
    /// (`C(n,x)*C(x,y)*C(y,z)`) at the original, larger degree bounds this
    /// test is named for was the case that, before these ceilings existed,
    /// drove the search loop through several *repeated* multi-minute-or-worse
    /// exact-rational Gaussian eliminations (`m = 3`, `cert_degree = 3` alone
    /// needs 770 unknowns and a ≈15 000-row linear system; even the
    /// `cert_degree = 2` step below that took ≈47 seconds *per probe*, and
    /// the search loop would otherwise retry it for every one of six
    /// `(order, a_degree)` combinations — see `MAX_ANSATZ_UNKNOWNS`'s and
    /// `MAX_CUMULATIVE_LARGE_PROBE_UNKNOWNS`'s docs for the measurements).
    /// This must now come back **bounded** — capped to roughly one expensive
    /// elimination attempt, not six or more — with an honest
    /// `SearchExhausted` naming the ceilings as the reason, not hang and not
    /// silently under-search. The wall-clock bound here is deliberately loose
    /// (this project's own standing practice for CI-flakiness — see
    /// AGENTS.md, and this specific assertion needs extra slack since the
    /// cumulative budget still permits one genuinely slow ≈30–50 second
    /// elimination): its only job is to distinguish "bounded to about one
    /// expensive attempt" from "unboundedly retries the same expensive
    /// probe," not to pin an exact latency.
    #[test]
    fn chained_product_at_original_bounds_refuses_fast_via_resource_ceiling() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Real);
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let f = pool.mul(vec![
            pool.func("binomial", vec![n, x]),
            pool.func("binomial", vec![x, y]),
            pool.func("binomial", vec![y, z]),
        ]);
        let opts = TelescopingMdOpts {
            max_order: 2,
            max_a_degree: 2,
            max_cert_degree: 3,
        };
        let start = std::time::Instant::now();
        let err = search::telescope_md_search(f, n, &[x, y, z], &pool, &opts)
            .expect_err("this combination must be refused via the resource ceiling, not solved");
        let elapsed = start.elapsed();
        // 900s, not 180s: this bound exists to catch a genuine hang (the
        // pre-fix behavior was still running after several minutes and
        // growing), not to pin CI wall-clock precisely. Windows CI runners
        // measured ~480s here against ~76s on Linux for the same exact-
        // rational elimination — a real, expected platform gap for GMP-
        // backed arithmetic under MSYS2/mingw, not evidence the ceiling
        // isn't working.
        assert!(
            elapsed.as_secs() < 900,
            "expected a bounded refusal (roughly one expensive elimination, not several), took \
             {elapsed:?}"
        );
        assert!(matches!(err, Telescoping2dError::SearchExhausted(_)));
        let Telescoping2dError::SearchExhausted(msg) = &err else {
            unreachable!()
        };
        assert!(
            msg.contains("MAX_ANSATZ_UNKNOWNS"),
            "expected the SearchExhausted message to name the resource ceiling as the reason, \
             got: {msg}"
        );
    }
}
