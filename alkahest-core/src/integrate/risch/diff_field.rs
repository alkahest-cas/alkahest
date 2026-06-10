//! The [`DifferentialField`] abstraction — Risch **M4** foundation.
//!
//! M4 makes the Risch differential-equation machinery polymorphic over the
//! *tower level* so the elementary-integration recursion can descend through
//! nested mixed algebraic/transcendental towers.  This module introduces the
//! foundational abstraction only:
//!
//! - [`DifferentialField`]: a differential field `K` with derivation `D`,
//!   exposing the Risch sub-algorithms the recursion needs at each level.
//! - Concrete implementations that **wrap the existing solvers** — no new math:
//!   * [`RationalDiffField`] for `ℚ(x)` (wraps
//!     [`solve_rational_rde_generalized`]),
//!   * [`DifferentialField`] for [`ExpTowerField`] (wraps `solve_tower_rde`),
//!   * [`DifferentialField`] for [`LogTowerField`] (wraps the field-generic
//!     `solve_tower_rde_generic`).
//!
//! **Additive / non-rewiring.** Nothing here changes any production integration
//! path; `integrate_risch` dispatch is untouched.  The trait is exercised by the
//! equivalence unit tests below (which assert it faithfully reproduces the
//! existing solvers) and will be consumed by the M4 PR2 multi-generator
//! recursive integrator.
//!
//! ## Relationship to [`CoeffField`]
//!
//! [`CoeffField`] is the *scalar* field abstraction (arithmetic + an optional
//! `derivation`) the polynomial-quotient core is generic over.  Every field that
//! implements `DifferentialField` here either *is* a `CoeffField` (the tower
//! fields) or is backed by one ([`RationalDiffField`] is backed by
//! [`RationalFunctionField`]).  `DifferentialField` layers the Risch *DE solvers*
//! on top of that scalar structure; it is deliberately a separate trait so that
//! a level may carry solvers a bare `CoeffField` does not.

use super::alg_field::{AlgElem, AlgExtension, RatFn, RationalFunctionField};
use super::alg_rde::solve_alg_rde_general;
use super::number_field::CoeffField;
use super::poly_rde::{degree, poly_deriv, trim};
use super::rational_rde::{
    numerator_degree_bound, poly_div_exact, poly_gcd, solve_rational_rde_generalized,
};
use super::tower_field::{
    solve_tower_coupled_radical_rde_bounded, solve_tower_rde_generic_bounded, tower_x_degree_bound,
    ExpTowerField, LogTowerField, TExpr,
};

// ===========================================================================
// The trait
// ===========================================================================

/// A differential field `K` with derivation `D`, supporting the Risch
/// sub-algorithms the elementary-integration recursion needs at each tower
/// level.
///
/// The field-structure methods mirror [`CoeffField`] (and, for the
/// implementations in this module, simply delegate to one); the value
/// `DifferentialField` adds over `CoeffField` is the **Risch DE machinery**
/// ([`rational_rde`](DifferentialField::rational_rde) and the PR2+ stubs
/// [`limited_integrate`](DifferentialField::limited_integrate) /
/// [`param_log_deriv`](DifferentialField::param_log_deriv)) made polymorphic
/// over the level.
pub trait DifferentialField {
    /// Element type of the field.
    type Elem: Clone + std::fmt::Debug;

    // --- field structure (mirrors CoeffField) ---

    /// The additive identity `0`.
    fn zero(&self) -> Self::Elem;
    /// The multiplicative identity `1`.
    fn one(&self) -> Self::Elem;
    /// `a + b`.
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    /// `a − b`.
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    /// `a · b`.
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    /// `−a`.
    fn neg(&self, a: &Self::Elem) -> Self::Elem;
    /// `a⁻¹`, or `None` if `a` is zero.
    fn inv(&self, a: &Self::Elem) -> Option<Self::Elem>;
    /// Is `a` the zero element?
    fn is_zero(&self, a: &Self::Elem) -> bool;
    /// Are `a` and `b` equal?
    fn eq(&self, a: &Self::Elem, b: &Self::Elem) -> bool;

    /// The derivation `D` of the field.
    fn derivation(&self, a: &Self::Elem) -> Self::Elem;

    /// Is `a` a constant (`D(a) = 0`)?  Default: `is_zero(derivation(a))`.
    fn is_constant(&self, a: &Self::Elem) -> bool {
        self.is_zero(&self.derivation(a))
    }

    // --- the Risch sub-algorithms, polymorphic over the level ---

    /// Solve the Risch differential equation `D(y) + f·y = g` for `y ∈ K`.
    ///
    /// Returns `Some(y)` when an elementary solution **in this field** exists,
    /// `None` otherwise.  Every implementation in this module *verifies* its
    /// candidate in-field before returning it, so a `Some` is always correct.
    ///
    /// The precise meaning of `None` depends on the level: for `ℚ(x)` the
    /// underlying solver is a decision procedure (so `None` certifies "no
    /// rational solution exists"); for the tower fields the underlying solver is
    /// verification-guarded over a bounded ansatz, so `None` means "no solution
    /// found within the search bounds", **not** a proof of non-existence.  See
    /// each implementation's docs.
    fn rational_rde(&self, f: &Self::Elem, g: &Self::Elem) -> Option<Self::Elem>;

    /// Upper bound on the degree (in the base variable `x`) of the numerator
    /// ansatz for a solution of `D(y) + f·y = g`, per Bronstein §6.5, or `None`
    /// if this level exposes no analytic bound (the caller then falls back to its
    /// heuristic cap).
    ///
    /// MUST be a **sound upper bound**: the true solution's numerator
    /// `x`-degree is `≤` this value.  It is used only as a *search ceiling* — the
    /// ansatz solvers take `max(this_bound, their_fixed_cap)`, so the ceiling is
    /// never lowered below today's heuristic (zero regression), and every
    /// candidate is still exact-verified in-field before being returned
    /// (soundness unchanged).  Over-estimating only widens the search;
    /// under-estimating would *miss* solutions, so when in doubt return a
    /// generous bound (or `None`).
    fn rde_degree_bound(&self, f: &Self::Elem, g: &Self::Elem) -> Option<usize> {
        let _ = (f, g);
        None
    }

    /// Solve the *coupled* radical Risch DE `D(u) + f·u = g` over the radical
    /// extension `yⁿ = a` of THIS field, for a possibly NON-BASE `f`.  `a` is the
    /// radicand (an element of THIS field); `f`, `g`, and the returned `u` are
    /// given as coefficient vectors `[c₀,…,c_{n−1}]` over the power basis
    /// `1,y,…,y^{n−1}` (each cᵢ an element of THIS field).
    ///
    /// Default: `None` — "this base field has no coupled-radical solver"
    /// (the caller then declines).  Implemented where the coupled system is
    /// tractable (e.g. `ℚ(x)` via `AlgExtension` / `solve_alg_rde_general`).
    /// Returns `None` for unsolvable/out-of-scope inputs; a `Some` need not be
    /// pre-verified (the caller re-verifies in-field).
    fn coupled_radical_rde(
        &self,
        n: usize,
        a: &Self::Elem,
        f: &[Self::Elem],
        g: &[Self::Elem],
    ) -> Option<Vec<Self::Elem>> {
        let _ = (n, a, f, g);
        None
    }

    /// LimitedIntegrate (Bronstein §7.1): given `g` and `D`-generators
    /// `w₁…wₙ ∈ K`, find `v ∈ K` and constants `c₁…cₙ` with
    /// `g = D(v) + Σᵢ cᵢ · D(wᵢ)/wᵢ`.
    ///
    /// **Not yet implemented at any level.**  The default returns `None`,
    /// meaning "declined / not yet implemented" — it is **not** a proof that no
    /// such decomposition exists.  Declared here so the abstraction is complete
    /// for PR2+ to fill in.
    fn limited_integrate(
        &self,
        g: &Self::Elem,
        ws: &[Self::Elem],
    ) -> Option<(Self::Elem, Vec<Self::Elem>)> {
        let _ = (g, ws);
        None
    }

    /// ParametricLogarithmicDerivative (Bronstein §7.3): decide whether
    /// `f = (n/m)·D(w)/w + D(v)/v` for some integers `n, m` and `v, w ∈ K`.
    /// On success returns `(n, m, v)`.
    ///
    /// **Not yet implemented at any level.**  The default returns `None`,
    /// meaning "declined / not yet implemented" — **not** a proof of
    /// non-existence.  Declared here so the abstraction is complete for PR2+.
    fn param_log_deriv(&self, f: &Self::Elem, w: &Self::Elem) -> Option<(i64, i64, Self::Elem)> {
        let _ = (f, w);
        None
    }
}

// ===========================================================================
// ℚ(x) — wraps solve_rational_rde_generalized
// ===========================================================================

/// The base differential field `ℚ(x)` with derivation `d/dx`.
///
/// An element is a [`RatFn`] (a reduced rational function over `ℚ`).  Field
/// arithmetic and the derivation delegate to the existing
/// [`RationalFunctionField`] [`CoeffField`];
/// [`rational_rde`](DifferentialField::rational_rde) wraps
/// [`solve_rational_rde_generalized`].
#[derive(Clone, Debug, Default)]
pub struct RationalDiffField {
    inner: RationalFunctionField,
}

impl RationalDiffField {
    /// Build the `ℚ(x)` differential field.
    pub fn new() -> Self {
        Self::default()
    }
}

impl DifferentialField for RationalDiffField {
    type Elem = RatFn;

    fn zero(&self) -> RatFn {
        self.inner.zero()
    }
    fn one(&self) -> RatFn {
        self.inner.one()
    }
    fn add(&self, a: &RatFn, b: &RatFn) -> RatFn {
        self.inner.add(a, b)
    }
    fn sub(&self, a: &RatFn, b: &RatFn) -> RatFn {
        self.inner.sub(a, b)
    }
    fn mul(&self, a: &RatFn, b: &RatFn) -> RatFn {
        self.inner.mul(a, b)
    }
    fn neg(&self, a: &RatFn) -> RatFn {
        self.inner.neg(a)
    }
    fn inv(&self, a: &RatFn) -> Option<RatFn> {
        self.inner.inv(a)
    }
    fn is_zero(&self, a: &RatFn) -> bool {
        self.inner.is_zero(a)
    }
    fn eq(&self, a: &RatFn, b: &RatFn) -> bool {
        self.inner.eq(a, b)
    }
    fn derivation(&self, a: &RatFn) -> RatFn {
        self.inner.derivation(a)
    }

    /// Solve `D(y) + f·y = g` over `ℚ(x)` by wrapping
    /// [`solve_rational_rde_generalized`].
    ///
    /// The underlying routine is a decision procedure for a *rational* solution:
    /// `None` means no `y ∈ ℚ(x)` satisfies the equation (e.g. `f = 0, g = 1/x`
    /// ⇒ `∫1/x = log x ∉ ℚ(x)`).
    fn rational_rde(&self, f: &RatFn, g: &RatFn) -> Option<RatFn> {
        let (num, den) =
            solve_rational_rde_generalized(f.numer(), f.denom(), g.numer(), g.denom())?;
        Some(RatFn::new(num, den))
    }

    /// The canonical Bronstein §6.5 base-case numerator-degree bound, mirroring
    /// `numerator_degree_bound` (the source of truth used inside
    /// [`solve_rational_rde`](super::rational_rde::solve_rational_rde)).
    ///
    /// Writing the solution as `y = N/E` with `E = gcd(B, B')` for the reduced
    /// denominator `B` of `g = C/B`, the bound is
    /// `deg E + max(deg C − deg B, deg f) + 2`.  When `f` is a genuine rational
    /// function (denominator `Bf`) we add `deg Bf` for the matching generalized
    /// path.  Always a sound upper bound on the numerator's `x`-degree.
    fn rde_degree_bound(&self, f: &RatFn, g: &RatFn) -> Option<usize> {
        let c_num = trim(g.numer().clone());
        let c_den = trim(g.denom().clone());
        // g = 0 ⇒ y = 0 ⇒ numerator degree 0.
        if c_num.is_empty() {
            return Some(0);
        }
        if c_den.is_empty() {
            return None;
        }
        // Reduce g = C/B (B from the denominator), then E = gcd(B, B').
        let gc = poly_gcd(&c_num, &c_den);
        let big_c = poly_div_exact(&c_num, &gc);
        let big_b = poly_div_exact(&c_den, &gc);
        let bprime = poly_deriv(&big_b);
        let e_poly = poly_gcd(&big_b, &bprime);

        let deg_b = degree(&big_b);
        let deg_c = degree(&big_c);
        let deg_e = degree(&e_poly);

        // f's effective polynomial degree, plus its denominator degree for the
        // rational-f (generalized) path.
        let f_den = trim(f.denom().clone());
        let deg_f = degree(f.numer());
        let deg_bf = if degree(&f_den) > 0 {
            degree(&f_den).max(0)
        } else {
            0
        };

        Some(numerator_degree_bound(deg_b, deg_c, deg_e, deg_f) + deg_bf as usize)
    }

    /// Coupled radical RDE over `ℚ(x)`: solve `D(u) + f·u = g` in the radical
    /// extension `yⁿ = a` for a possibly non-base `f`, by bridging to the proven
    /// coupled solver `solve_alg_rde_general` over an `AlgExtension`.
    ///
    /// Requires the radicand `a` to be a polynomial (its canonical denominator is
    /// the monic constant `1`); otherwise returns `None`.  Builds
    /// `AlgExtension::radical(n, a.numer())` (so `α = a^{1/n}` with the same
    /// diagonal derivation twist `D(α) = (1/n)(a'/a)α` as `RadicalExt`), maps the
    /// power-basis coefficient vectors `f`, `g` (each `&[RatFn]`, padded/truncated
    /// to length `n`) to `AlgElem`, runs the coupled solver, and maps the result
    /// back to a length-`n` `Vec<RatFn>`.  `n < 2` returns `None`
    /// (`AlgExtension::radical` needs degree ≥ 2; the degenerate `n = 1` case has
    /// no `y`-coupling anyway).  The caller re-verifies in-field.
    fn coupled_radical_rde(
        &self,
        n: usize,
        a: &RatFn,
        f: &[RatFn],
        g: &[RatFn],
    ) -> Option<Vec<RatFn>> {
        if n < 2 {
            return None;
        }
        // Radicand must be a polynomial: canonical denominator is monic, so a
        // constant denominator is exactly `[1]` (poly_one).
        if degree(a.denom()) != 0 {
            return None;
        }
        let ext = AlgExtension::radical(n, a.numer());

        // Pad/truncate a power-basis coefficient vector to length `n`, then reduce
        // mod `yⁿ − a` into an `AlgElem`.
        let to_alg = |v: &[RatFn]| -> AlgElem {
            let mut comps = vec![RatFn::int(0); n];
            for (i, c) in v.iter().take(n).enumerate() {
                comps[i] = c.clone();
            }
            ext.reduce(&comps)
        };

        let f_alg = to_alg(f);
        let g_alg = to_alg(g);
        let u_alg = solve_alg_rde_general(&ext, &f_alg, &g_alg)?;

        // Map the AlgElem (length ≤ n) back to a length-n power-basis vector.
        let mut u = vec![RatFn::int(0); n];
        for (i, c) in u_alg.into_iter().take(n).enumerate() {
            u[i] = c;
        }
        Some(u)
    }
}

// ===========================================================================
// Exponential tower ℚ(x)(t), t = exp(η) — wraps solve_tower_rde
// ===========================================================================

impl DifferentialField for ExpTowerField {
    type Elem = TExpr;

    fn zero(&self) -> TExpr {
        <Self as CoeffField>::zero(self)
    }
    fn one(&self) -> TExpr {
        <Self as CoeffField>::one(self)
    }
    fn add(&self, a: &TExpr, b: &TExpr) -> TExpr {
        <Self as CoeffField>::add(self, a, b)
    }
    fn sub(&self, a: &TExpr, b: &TExpr) -> TExpr {
        <Self as CoeffField>::sub(self, a, b)
    }
    fn mul(&self, a: &TExpr, b: &TExpr) -> TExpr {
        <Self as CoeffField>::mul(self, a, b)
    }
    fn neg(&self, a: &TExpr) -> TExpr {
        <Self as CoeffField>::neg(self, a)
    }
    fn inv(&self, a: &TExpr) -> Option<TExpr> {
        <Self as CoeffField>::inv(self, a)
    }
    fn is_zero(&self, a: &TExpr) -> bool {
        <Self as CoeffField>::is_zero(self, a)
    }
    fn eq(&self, a: &TExpr, b: &TExpr) -> bool {
        <Self as CoeffField>::eq(self, a, b)
    }
    fn derivation(&self, a: &TExpr) -> TExpr {
        <Self as CoeffField>::derivation(self, a)
    }

    /// Solve `D(v) + f·v = g` over the exponential tower `ℚ(x)(eᵑ)` by wrapping
    /// the bounded solver (with `omega = f`, `c = g`), passing the analytic
    /// `x`-degree ceiling from [`rde_degree_bound`](Self::rde_degree_bound).
    ///
    /// The underlying solver is verification-guarded over a bounded ansatz, so a
    /// `Some` is always a correct solution but `None` only means "no solution
    /// found within the search bounds" — not a non-elementarity certificate.
    fn rational_rde(&self, f: &TExpr, g: &TExpr) -> Option<TExpr> {
        solve_tower_rde_generic_bounded(self, f, g, self.rde_degree_bound(f, g))
    }

    /// Sound `x`-degree ceiling for the exp-tower ansatz (Bronstein §6.5): the
    /// drift coefficient is `η' = deta`, so the bound is driven by the `x`-degrees
    /// of `f`, `g`, and `η'`.  See `tower_x_degree_bound`.
    fn rde_degree_bound(&self, f: &TExpr, g: &TExpr) -> Option<usize> {
        let drift = degree(self.deta.numer()).max(degree(self.deta.denom()));
        Some(tower_x_degree_bound(f, g, drift))
    }

    /// Coupled radical RDE over the exponential tower `ℚ(x)(eᵑ)`: solve
    /// `D(u) + f·u = g` in the radical extension `yⁿ = a` for a possibly
    /// non-diagonal `f`, via the tower-base coupled solver
    /// `solve_tower_coupled_radical_rde_bounded` (undetermined-coefficient ansatz
    /// `uᵢ = (Σ cᵢⱼₖ xᵏ tʲ)/D` per `y`-component, exact ℚ-linear system, Gauss
    /// solve, exact in-field verification of `D(u)+f·u=g`).  The `x`-degree search
    /// ceiling is raised to the analytic bound (drift `η'`) but never below the
    /// heuristic floor.  `n < 2` or `a = 0` returns `None`; the caller re-verifies.
    fn coupled_radical_rde(
        &self,
        n: usize,
        a: &TExpr,
        f: &[TExpr],
        g: &[TExpr],
    ) -> Option<Vec<TExpr>> {
        let drift = degree(self.deta.numer()).max(degree(self.deta.denom()));
        let x_bound = tower_coupled_x_bound(a, f, g, drift);
        solve_tower_coupled_radical_rde_bounded(self, n, a, f, g, Some(x_bound))
    }
}

// ===========================================================================
// Logarithmic tower ℚ(x)(t), t = log(h) — wraps solve_tower_rde_generic
// ===========================================================================

impl DifferentialField for LogTowerField {
    type Elem = TExpr;

    fn zero(&self) -> TExpr {
        <Self as CoeffField>::zero(self)
    }
    fn one(&self) -> TExpr {
        <Self as CoeffField>::one(self)
    }
    fn add(&self, a: &TExpr, b: &TExpr) -> TExpr {
        <Self as CoeffField>::add(self, a, b)
    }
    fn sub(&self, a: &TExpr, b: &TExpr) -> TExpr {
        <Self as CoeffField>::sub(self, a, b)
    }
    fn mul(&self, a: &TExpr, b: &TExpr) -> TExpr {
        <Self as CoeffField>::mul(self, a, b)
    }
    fn neg(&self, a: &TExpr) -> TExpr {
        <Self as CoeffField>::neg(self, a)
    }
    fn inv(&self, a: &TExpr) -> Option<TExpr> {
        <Self as CoeffField>::inv(self, a)
    }
    fn is_zero(&self, a: &TExpr) -> bool {
        <Self as CoeffField>::is_zero(self, a)
    }
    fn eq(&self, a: &TExpr, b: &TExpr) -> bool {
        <Self as CoeffField>::eq(self, a, b)
    }
    fn derivation(&self, a: &TExpr) -> TExpr {
        <Self as CoeffField>::derivation(self, a)
    }

    /// Solve `D(v) + f·v = g` over the logarithmic tower `ℚ(x)(log h)` by
    /// wrapping the field-generic bounded solver (with `omega = f`, `c = g`) and
    /// passing the analytic `x`-degree ceiling from
    /// [`rde_degree_bound`](Self::rde_degree_bound).  Same verification-guarded
    /// semantics as the exp-tower impl: a `Some` is correct; `None` means "not
    /// found within bounds".
    fn rational_rde(&self, f: &TExpr, g: &TExpr) -> Option<TExpr> {
        solve_tower_rde_generic_bounded(self, f, g, self.rde_degree_bound(f, g))
    }

    /// Sound `x`-degree ceiling for the log-tower ansatz (Bronstein §6.5): the
    /// drift coefficient is `h'/h = dh_over_h`, so the bound is driven by the
    /// `x`-degrees of `f`, `g`, and the drift.  See `tower_x_degree_bound`.
    fn rde_degree_bound(&self, f: &TExpr, g: &TExpr) -> Option<usize> {
        let drift = degree(self.dh_over_h.numer()).max(degree(self.dh_over_h.denom()));
        Some(tower_x_degree_bound(f, g, drift))
    }

    /// Coupled radical RDE over the logarithmic tower `ℚ(x)(log h)`: solve
    /// `D(u) + f·u = g` in the radical extension `yⁿ = a` for a possibly
    /// non-diagonal `f`, via the tower-base coupled solver
    /// `solve_tower_coupled_radical_rde_bounded` (same ansatz / exact ℚ-linear
    /// system / Gauss solve / exact in-field verification as the exp-tower impl;
    /// only the tower derivation differs).  The `x`-degree ceiling uses the
    /// log-tower drift `h'/h` and is never lowered below the heuristic floor.
    /// `n < 2` or `a = 0` returns `None`; the caller re-verifies.
    fn coupled_radical_rde(
        &self,
        n: usize,
        a: &TExpr,
        f: &[TExpr],
        g: &[TExpr],
    ) -> Option<Vec<TExpr>> {
        let drift = degree(self.dh_over_h.numer()).max(degree(self.dh_over_h.denom()));
        let x_bound = tower_coupled_x_bound(a, f, g, drift);
        solve_tower_coupled_radical_rde_bounded(self, n, a, f, g, Some(x_bound))
    }
}

/// A sound `x`-degree search ceiling for the coupled tower radical ansatz: the
/// max of the scalar [`tower_x_degree_bound`] over the radicand `a` and every
/// component of `f`, `g` (with the given derivation drift degree).  Over-
/// estimating only widens the search — verification gates correctness.
fn tower_coupled_x_bound(a: &TExpr, f: &[TExpr], g: &[TExpr], drift: i64) -> usize {
    let mut bound = tower_x_degree_bound(a, a, drift);
    for c in f.iter().chain(g.iter()) {
        bound = bound.max(tower_x_degree_bound(c, c, drift));
    }
    bound
}

// ===========================================================================
// Tests — equivalence with the wrapped solvers + correctness in-field
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::super::tower_field::{solve_tower_rde, solve_tower_rde_generic};
    use super::*;
    use crate::integrate::risch::poly_rde::QPoly;
    use rug::Rational;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    /// A polynomial `RatFn` from ascending ℚ-coefficients.
    fn rf_poly(c: &[i64]) -> RatFn {
        let p: QPoly = c.iter().map(|&n| rat(n)).collect();
        RatFn::from_poly(&p)
    }

    // ---- ℚ(x): trait reproduces solve_rational_rde_generalized exactly ----

    /// Assert the trait result agrees with calling the underlying solver
    /// directly, and (when solvable) that `D(y) + f·y = g` holds in-field.
    fn check_qx(f: &RatFn, g: &RatFn) {
        let field = RationalDiffField::new();
        let trait_res = field.rational_rde(f, g);
        let direct = solve_rational_rde_generalized(f.numer(), f.denom(), g.numer(), g.denom())
            .map(|(n, d)| RatFn::new(n, d));
        assert_eq!(
            trait_res, direct,
            "trait rational_rde must match the wrapped solver exactly"
        );
        if let Some(y) = trait_res {
            // D(y) + f·y = g, verified in ℚ(x).
            let lhs = field.add(&field.derivation(&y), &field.mul(f, &y));
            assert!(
                field.eq(&lhs, g),
                "ℚ(x) RDE solution must satisfy D(y)+f·y=g; got y={y:?}"
            );
        }
    }

    #[test]
    fn qx_f0_g_2x_gives_x_squared() {
        // D(y) = 2x  ⇒  y = x².
        let field = RationalDiffField::new();
        let f = field.zero();
        let g = rf_poly(&[0, 2]); // 2x
        check_qx(&f, &g);
        let y = field.rational_rde(&f, &g).expect("solvable");
        assert!(
            field.eq(&y, &rf_poly(&[0, 0, 1])),
            "y should be x²; got {y:?}"
        );
    }

    #[test]
    fn qx_f1_known_solution() {
        // D(y) + y = x + 1  ⇒  y = x   (D(x) + x = 1 + x). ✓
        let field = RationalDiffField::new();
        let f = field.one();
        let g = rf_poly(&[1, 1]); // x + 1
        check_qx(&f, &g);
        let y = field.rational_rde(&f, &g).expect("solvable");
        assert!(field.eq(&y, &rf_poly(&[0, 1])), "y should be x; got {y:?}");
    }

    #[test]
    fn qx_f_const_g_const() {
        // D(y) + 2·y = 4  ⇒  constant y = 2.
        let field = RationalDiffField::new();
        let f = rf_poly(&[2]);
        let g = rf_poly(&[4]);
        check_qx(&f, &g);
        let y = field.rational_rde(&f, &g).expect("solvable");
        assert!(field.eq(&y, &rf_poly(&[2])), "y should be 2; got {y:?}");
    }

    #[test]
    fn qx_rational_f_solvable() {
        // f = 1/x, g = D(x) + (1/x)·x = 1 + 1 = 2  ⇒  y = x.
        // Exercises the generalized (rational-f) path.
        let field = RationalDiffField::new();
        let f = RatFn::new(vec![rat(1)], vec![rat(0), rat(1)]); // 1/x
        let y_expected = rf_poly(&[0, 1]); // x
        let g = field.add(&field.derivation(&y_expected), &field.mul(&f, &y_expected));
        check_qx(&f, &g);
        let y = field.rational_rde(&f, &g).expect("solvable");
        assert!(field.eq(&y, &y_expected), "y should be x; got {y:?}");
    }

    #[test]
    fn qx_ei_type_is_none() {
        // D(y) = 1/x  ⇒  y = log x ∉ ℚ(x): no rational solution.
        let field = RationalDiffField::new();
        let f = field.zero();
        let g = RatFn::new(vec![rat(1)], vec![rat(0), rat(1)]); // 1/x
        check_qx(&f, &g); // also asserts trait == direct (both None)
        assert!(field.rational_rde(&f, &g).is_none(), "1/x ⇒ None (Ei/Li)");
    }

    #[test]
    fn qx_derivation_and_is_constant() {
        let field = RationalDiffField::new();
        // D(x) = 1.
        let x = rf_poly(&[0, 1]);
        assert!(field.eq(&field.derivation(&x), &field.one()));
        assert!(!field.is_constant(&x));
        // D(c) = 0 for a constant.
        let c = rf_poly(&[7]);
        assert!(field.is_zero(&field.derivation(&c)));
        assert!(field.is_constant(&c));
        // D(x²) = 2x.
        let x2 = rf_poly(&[0, 0, 1]);
        assert!(field.eq(&field.derivation(&x2), &rf_poly(&[0, 2])));
    }

    // ---- exp tower: trait reproduces solve_tower_rde ----

    /// `x` as a `ℚ(x)(t)` constant-in-t element.
    fn x_elem() -> TExpr {
        TExpr::from_ratfn(rf_poly(&[0, 1]))
    }

    #[test]
    fn exp_tower_derivation_basics() {
        // η = x ⇒ η' = 1, t = eˣ.
        let field = ExpTowerField::new(RatFn::int(1));
        // D(t) = t  (i.e. D(exp x) = exp x).
        let t = TExpr::t();
        assert!(<ExpTowerField as DifferentialField>::eq(
            &field,
            &DifferentialField::derivation(&field, &t),
            &t
        ));
        assert!(!field.is_constant(&t));
        // D(x) = 1.
        let x = x_elem();
        assert!(<ExpTowerField as DifferentialField>::eq(
            &field,
            &DifferentialField::derivation(&field, &x),
            &DifferentialField::one(&field)
        ));
        // A pure ℚ constant is constant.
        let c = TExpr::int(5);
        assert!(field.is_constant(&c));
    }

    #[test]
    fn exp_tower_v_equals_t() {
        // D(v) + 0·v = t  ⇒  v = t.  Trait must match solve_tower_rde.
        let field = ExpTowerField::new(RatFn::int(1));
        let f = DifferentialField::zero(&field);
        let g = TExpr::t();
        let trait_res = DifferentialField::rational_rde(&field, &f, &g);
        let direct = solve_tower_rde(&field, &f, &g);
        assert_eq!(trait_res, direct, "trait must match solve_tower_rde");
        let v = trait_res.expect("v = t");
        assert_eq!(v, TExpr::t());
    }

    #[test]
    fn exp_tower_example15_component() {
        // Example-15 i=2 component in ℚ(x)(eˣ):
        //   ω₂ = (2/3)(1+t)/(x+t),  c₂ = [(2x+3)t + 5x]/(x+t)  ⇒  v₂ = 3x.
        let field = ExpTowerField::new(RatFn::int(1));
        let a = DifferentialField::add(&field, &x_elem(), &TExpr::t()); // x + t
        let a_prime = DifferentialField::derivation(&field, &a); // 1 + t
        let two_thirds = TExpr::from_ratfn(RatFn::new(vec![rat(2)], vec![rat(3)]));
        let inv_a = DifferentialField::inv(&field, &a).unwrap();
        let omega2 = DifferentialField::mul(
            &field,
            &two_thirds,
            &DifferentialField::mul(&field, &a_prime, &inv_a),
        );

        let num = vec![
            RatFn::from_poly(&vec![rat(0), rat(5)]), // 5x   · t⁰
            RatFn::from_poly(&vec![rat(3), rat(2)]), // 2x+3 · t¹
        ];
        let den = vec![
            RatFn::from_poly(&vec![rat(0), rat(1)]), // x · t⁰
            RatFn::int(1),                           // 1 · t¹
        ];
        let c2 = TExpr::new(num, den);

        let trait_res = DifferentialField::rational_rde(&field, &omega2, &c2);
        let direct = solve_tower_rde(&field, &omega2, &c2);
        assert_eq!(trait_res, direct, "trait must match solve_tower_rde");

        let v = trait_res.expect("Example-15 component is solvable");
        let expected = TExpr::from_ratfn(RatFn::from_poly(&vec![rat(0), rat(3)])); // 3x
        assert!(
            <ExpTowerField as DifferentialField>::eq(&field, &v, &expected),
            "v₂ should be 3x; got {v:?}"
        );
        // D(v) + ω·v = c, in-field.
        let lhs = DifferentialField::add(
            &field,
            &DifferentialField::derivation(&field, &v),
            &DifferentialField::mul(&field, &omega2, &v),
        );
        assert!(
            <ExpTowerField as DifferentialField>::eq(&field, &lhs, &c2),
            "D(v)+ω·v=c must hold in-field"
        );
    }

    #[test]
    fn exp_tower_nonelementary_is_none() {
        // D(v) = t/x  (∫eˣ/x = Ei): no solution within bounds ⇒ None.
        let field = ExpTowerField::new(RatFn::int(1));
        let f = DifferentialField::zero(&field);
        let inv_x = DifferentialField::inv(&field, &x_elem()).unwrap();
        let g = DifferentialField::mul(&field, &TExpr::t(), &inv_x); // t/x
        let trait_res = DifferentialField::rational_rde(&field, &f, &g);
        let direct = solve_tower_rde(&field, &f, &g);
        assert_eq!(trait_res, direct);
        assert!(trait_res.is_none(), "Ei-type ⇒ None");
    }

    // ---- log tower: trait reproduces solve_tower_rde_generic ----

    #[test]
    fn log_tower_derivation_basics() {
        // h = x ⇒ h'/h = 1/x, t = log(x).  D(t) = 1/x.
        let dh_over_h = RatFn::new(vec![rat(1)], vec![rat(0), rat(1)]); // 1/x
        let field = LogTowerField::new(dh_over_h.clone());
        let t = TExpr::t();
        let dt = DifferentialField::derivation(&field, &t);
        let expected = TExpr::from_ratfn(dh_over_h);
        assert!(
            <LogTowerField as DifferentialField>::eq(&field, &dt, &expected),
            "D(log x) = 1/x; got {dt:?}"
        );
        assert!(!field.is_constant(&t));
        // D(x) = 1.
        let x = x_elem();
        assert!(<LogTowerField as DifferentialField>::eq(
            &field,
            &DifferentialField::derivation(&field, &x),
            &DifferentialField::one(&field)
        ));
    }

    #[test]
    fn log_tower_v_equals_t() {
        // h = x, t = log x, D(t) = 1/x.  D(v) + 0·v = 1/x  ⇒  v = t = log x.
        let dh_over_h = RatFn::new(vec![rat(1)], vec![rat(0), rat(1)]); // 1/x
        let field = LogTowerField::new(dh_over_h.clone());
        let f = DifferentialField::zero(&field);
        let g = TExpr::from_ratfn(dh_over_h); // 1/x
        let trait_res = DifferentialField::rational_rde(&field, &f, &g);
        let direct = solve_tower_rde_generic(&field, &f, &g);
        assert_eq!(
            trait_res, direct,
            "trait must match solve_tower_rde_generic"
        );
        let v = trait_res.expect("v = log x");
        assert_eq!(v, TExpr::t());
        // Verify D(v) = g in-field.
        let lhs = DifferentialField::derivation(&field, &v);
        assert!(<LogTowerField as DifferentialField>::eq(&field, &lhs, &g));
    }

    #[test]
    fn log_tower_polynomial_in_t() {
        // h = x, t = log x.  D(t²) = 2t·(1/x).  So with f = 0,
        // g = 2t/x  ⇒  v = t² = (log x)².
        let dh_over_h = RatFn::new(vec![rat(1)], vec![rat(0), rat(1)]); // 1/x
        let field = LogTowerField::new(dh_over_h);
        let t = TExpr::t();
        let t2 = DifferentialField::mul(&field, &t, &t);
        let g = DifferentialField::derivation(&field, &t2); // 2t/x
        let f = DifferentialField::zero(&field);
        let trait_res = DifferentialField::rational_rde(&field, &f, &g);
        let direct = solve_tower_rde_generic(&field, &f, &g);
        assert_eq!(trait_res, direct);
        let v = trait_res.expect("v = (log x)²");
        assert!(
            <LogTowerField as DifferentialField>::eq(&field, &v, &t2),
            "v should be t²; got {v:?}"
        );
    }

    // ---- the PR2+ stubs decline (None), at every level ----

    #[test]
    fn stubs_decline() {
        let qx = RationalDiffField::new();
        assert!(qx.limited_integrate(&qx.one(), &[qx.one()]).is_none());
        assert!(qx.param_log_deriv(&qx.one(), &qx.one()).is_none());

        let exp = ExpTowerField::new(RatFn::int(1));
        let one_e = DifferentialField::one(&exp);
        assert!(exp
            .limited_integrate(&one_e, std::slice::from_ref(&one_e))
            .is_none());
        assert!(exp.param_log_deriv(&one_e, &one_e).is_none());
    }
}
