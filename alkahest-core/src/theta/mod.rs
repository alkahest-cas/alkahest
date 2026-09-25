//! Riemann theta functions, classical modular functions and Weierstrass
//! elliptic functions — **as rigorous enclosures**, backed by FLINT's Arb layer.
//!
//! Every value returned by this module is a [`ComplexBall`]: a midpoint and a
//! radius, with the guarantee that the true value lies inside. Nothing here
//! returns a bare `f64`, and the one accessor that produces one
//! ([`ComplexBall::value_if_accurate`]) refuses unless the enclosure meets a
//! stated bit requirement. That is the whole point of the module: a theta value
//! is the kind of quantity that is easy to compute wrongly and impossible to
//! check afterwards, so the error bound travels with it.
//!
//! ```
//! use alkahest_cas::experimental::{j_invariant, ComplexBall, Precision};
//!
//! // j(i) = 1728, exactly — the classical normalisation check.
//! let tau = ComplexBall::exact_i64(0, 1, 256);
//! let j = j_invariant(&tau, Precision::Bits(256)).unwrap();
//! assert!(j.contains_f64(1728.0, 0.0));
//! assert!(j.accuracy_bits() > 200);
//! ```
//!
//! # What is here
//!
//! * **Genus 1** ([`acb_modular`], [`acb_elliptic`]): [`dedekind_eta`],
//!   [`j_invariant`], [`modular_lambda`], [`modular_discriminant`],
//!   [`eisenstein_series`], the Jacobi theta functions [`jacobi_theta`] /
//!   [`jacobi_theta_null`], and the Weierstrass functions
//!   [`weierstrass_p`], [`weierstrass_p_prime`], [`weierstrass_zeta`],
//!   [`weierstrass_sigma`], [`weierstrass_invariants`], [`weierstrass_roots`].
//! * **Genus `g`** (`acb_theta`): [`riemann_theta`], [`riemann_theta_squared`],
//!   [`riemann_theta_characteristic`], the characteristic helpers
//!   [`theta_characteristic_index`] / [`theta_characteristic_is_even`], and the
//!   Siegel helpers [`SiegelMatrix::is_certainly_in_siegel_upper_half_space`],
//!   [`siegel_is_reduced`] and [`siegel_reduce`].
//! * **Ball arithmetic** on [`ComplexBall`] and [`RealBall`], enough to build
//!   inputs that are not rational — `rho = exp(2 pi i / 3)`, for instance — and
//!   to check identities between computed results inside their own radii.
//!
//! [`acb_modular`]: https://flintlib.org/doc/acb_modular.html
//! [`acb_elliptic`]: https://flintlib.org/doc/acb_elliptic.html
//!
//! # Normalisation (read this before comparing against another system)
//!
//! Theta functions have more inconsistent conventions in the literature than
//! almost anything else in analysis. These are FLINT's, and they are the ones
//! used here without reinterpretation:
//!
//! * `theta_{a,b}(z, tau) = sum_{n in Z^g + a/2} exp(pi i n^T tau n + 2 pi i n^T (z + b/2))`
//!   for `a, b` in `{0,1}^g`. The `4^g` values are indexed by the `2g`-bit
//!   integer `(a << g) | b`, with `a` in the more significant half.
//! * The classical Jacobi functions use `q = exp(pi i tau)` and
//!   `w = exp(pi i z)`, so `theta_3(z, tau) = 1 + 2 sum_{n>=1} q^{n^2} cos(2 n pi z)`.
//! * Consequently, in genus 1,
//!   `(theta_1, theta_2, theta_3, theta_4) = (-theta_{1,1}, theta_{1,0}, theta_{0,0}, theta_{0,1})`.
//!   The sign on `theta_1` is real and is checked by
//!   `theta::tests::genus_one_agrees_with_classical_jacobi`, which evaluates
//!   both FLINT code paths and asserts the balls overlap.
//! * `eta`, `j`, `lambda` and `Delta` are the standard ones: `j(i) = 1728`,
//!   `j(rho) = 0` for `rho = exp(2 pi i / 3)`, `eta(i) = Gamma(1/4) / (2 pi^{3/4})`.
//! * Weierstrass `℘(z, tau)` is for the lattice `Z + tau Z`, and
//!   [`weierstrass_invariants`] returns `(g2, g3)` for that same lattice.
//!
//! # Precision, accuracy and refusal
//!
//! [`Precision`] has two modes and they mean different things:
//!
//! * [`Precision::Bits`] spends a stated working precision once and hands back
//!   whatever enclosure that produced — possibly a wide one, possibly an
//!   infinite one. Nothing is hidden; interrogate
//!   [`ComplexBall::accuracy_bits`].
//! * [`Precision::AccurateTo`] raises the working precision until the result
//!   has at least the requested **relative** accuracy, and refuses with
//!   [`ThetaError::InsufficientAccuracy`] if it runs out of room.
//!
//! One consequence deserves stating plainly: **`AccurateTo` can never be
//! satisfied by a value that is exactly zero.** Relative accuracy of zero is
//! not a thing. `j(rho) = 0` is checked with [`Precision::Bits`] and
//! [`ComplexBall::contains_zero`], which is the correct question to ask of a
//! vanishing quantity; asking `AccurateTo` there will correctly refuse forever.
//!
//! # Scope limits
//!
//! * **Genus ceiling.** [`MAX_GENUS`] is 6, and that is a guard rather than a
//!   recommendation. `acb_theta_all` returns `4^g` values, so the output alone
//!   is exponential in `g` before any work is counted. Measured on one x86-64
//!   machine, release build, `z = 0`, 128-bit working precision, one
//!   [`riemann_theta`] call on a near-diagonal period matrix:
//!
//!   | genus | values | time    |
//!   |-------|--------|---------|
//!   | 1     | 4      | 0.9 ms  |
//!   | 2     | 16     | 5.7 ms  |
//!   | 3     | 64     | 62 ms   |
//!   | 4     | 256    | 0.62 s  |
//!   | 5     | 1024   | 11 s    |
//!   | 6     | 4096   | 244 s   |
//!
//!   Genus 1 and 2 are what this module is built and tested for. Genus 3 is
//!   exercised by a test. **Genus 4 to 6 are accepted and covered by no test
//!   here**: treat a result there as untested code, and read its
//!   [`ThetaValues::worst_accuracy_bits`] before believing any of it. The cost
//!   grows by roughly an order of magnitude per genus and the last step is
//!   worse than that, so a genus-7 request would be minutes to hours — hence
//!   the cap rather than an open-ended refusal at run time.
//! * **No derivatives.** FLINT's `acb_theta_jet` computes partial derivatives
//!   of theta with respect to `z` to any order; only order 0 is wrapped. The
//!   ordering conventions for jets are intricate enough that exposing them
//!   without tests would be worse than not exposing them.
//! * **No genus-2 Siegel modular forms.** `acb_theta_g2_*` (`chi_5`, `chi_35`,
//!   the Igusa–Clebsch invariants) is not wrapped.
//! * **Inputs are balls, not exact symbols.** A period matrix is given as
//!   enclosures. If the input balls are wide, the output balls are wider, and
//!   this module will not pretend otherwise.
//! * **Working precision is not output accuracy.** FLINT's own precision
//!   management means a `Precision::Bits(p)` request typically returns rather
//!   fewer than `p` accurate bits: genus 1 loses a handful, and at genus 2 a
//!   300-bit request came back with enclosures around `1e-46` on `O(1)` values,
//!   i.e. roughly half the bits asked for. That is why
//!   [`Precision::AccurateTo`] exists — it is the mode to use when a specific
//!   accuracy is actually required, rather than guessing at a working
//!   precision.
//! * **`Im(tau)` must be *provably* positive definite** at the working
//!   precision, or the call refuses with
//!   [`ThetaError::NotInSiegelUpperHalfSpace`]. A refusal means "not proved",
//!   never "proved false" — raising the precision may settle it.
//!
//! # Relationship to the rest of the crate
//!
//! [`crate::ball::ArbBall`] is a *separate*, MPFR-backed ball type used by
//! `IntervalEval`. It is untouched by this module and continues to work exactly
//! as before. This module is the first genuine `arb_t`/`acb_t` binding in the
//! crate; migrating `ball` onto it is deliberately left as separate work.
//!
//! [`crate::funcfield`] models hyperelliptic curves, divisors and Riemann–Roch
//! symbolically over `Q`. There is no wiring between the two: the analytic
//! Jacobian of such a curve is exactly what a period matrix here describes, but
//! computing the period matrix *from* a curve is not implemented.
//!
//! # FLINT versions
//!
//! Arb became part of FLINT in 3.0 and `acb_theta` was rewritten in 3.2.
//! `build.rs` probes `libflint`'s symbol table and sets `flint_arb` /
//! `flint_acb_theta`. On a FLINT that lacks either, this module still compiles
//! and every entry point refuses with `E-THETA-001`, so the Python surface is
//! the same shape on every build.

pub mod ball;
pub mod error;
pub mod modular;
pub mod riemann;

#[cfg_attr(flint_arb, path = "backend_flint.rs")]
#[cfg_attr(not(flint_arb), path = "backend_stub.rs")]
mod backend;

mod arith;

#[cfg(test)]
mod tests;

pub use ball::{ComplexBall, RealBall};
pub use error::ThetaError;
pub use modular::{
    dedekind_eta, eisenstein_series, j_invariant, jacobi_theta, jacobi_theta_null,
    modular_discriminant, modular_lambda, weierstrass_invariants, weierstrass_p,
    weierstrass_p_prime, weierstrass_roots, weierstrass_sigma, weierstrass_zeta,
};
pub use riemann::{
    riemann_theta, riemann_theta_characteristic, riemann_theta_squared, siegel_is_reduced,
    siegel_reduce, theta_characteristic_bits, theta_characteristic_index,
    theta_characteristic_is_even, SiegelMatrix, SiegelReduction, ThetaValues,
};

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/// Largest working precision, in bits, this module will accept or climb to.
///
/// A million bits is well past the point where a theta evaluation stops being
/// interactive; the cap exists so that an unsatisfiable
/// [`Precision::AccurateTo`] refuses in bounded time rather than running until
/// the machine gives up.
pub const MAX_PRECISION_BITS: u32 = 1 << 20;

/// Smallest working precision this module will accept.
pub const MIN_PRECISION_BITS: u32 = 2;

/// Working precision used by the convenience paths that do not take one.
pub const DEFAULT_PRECISION_BITS: u32 = 256;

/// Largest genus accepted by [`riemann_theta`] and friends.
///
/// See the module-level "Scope limits": genus 1 and 2 are what this is built
/// for, genus 3 is tested, and 4 to 6 are accepted but untested.
pub const MAX_GENUS: usize = 6;

// ---------------------------------------------------------------------------
// Precision
// ---------------------------------------------------------------------------

/// How much precision to spend, and what to do if the result is not accurate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    /// Evaluate once at this working precision and return the enclosure that
    /// results, however wide. Nothing is hidden and nothing is refused on
    /// accuracy grounds — read [`ComplexBall::accuracy_bits`].
    Bits(u32),
    /// Raise the working precision until the result carries at least this many
    /// bits of **relative** accuracy, or refuse with
    /// [`ThetaError::InsufficientAccuracy`].
    ///
    /// Cannot be satisfied by a value that is exactly zero — see the module
    /// docs.
    AccurateTo(u32),
}

impl Precision {
    /// The working precision an [`Precision::AccurateTo`] request starts from.
    fn initial_working_bits(self) -> u32 {
        match self {
            Precision::Bits(b) => b,
            Precision::AccurateTo(t) => t.saturating_add(32).clamp(64, MAX_PRECISION_BITS),
        }
    }

    /// The highest working precision this request will climb to.
    ///
    /// Bounded relative to the ask: if `16 * target + 256` bits of work has not
    /// produced `target` bits of accuracy, the quantity is structurally
    /// resistant — most often because it is zero — and more precision will not
    /// help.
    fn ceiling_bits(self) -> u32 {
        match self {
            Precision::Bits(b) => b,
            Precision::AccurateTo(t) => t
                .saturating_mul(16)
                .saturating_add(256)
                .clamp(1024, MAX_PRECISION_BITS),
        }
    }

    fn validate(self) -> Result<(), ThetaError> {
        let bits = match self {
            Precision::Bits(b) | Precision::AccurateTo(b) => b,
        };
        if !(MIN_PRECISION_BITS..=MAX_PRECISION_BITS).contains(&bits) {
            return Err(ThetaError::PrecisionOutOfRange {
                bits: u64::from(bits),
                max: MAX_PRECISION_BITS,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Enclosure summary + the refinement driver
// ---------------------------------------------------------------------------

/// Anything a theta entry point can return, summarised by how good its worst
/// enclosure is. Implemented for the shapes the public functions use.
pub(crate) trait Enclosure {
    /// Relative accuracy of the *worst* ball in the result.
    fn worst_accuracy_bits(&self) -> i64;
    /// Does any ball in the result have an infinite radius?
    fn any_indeterminate(&self) -> bool;
}

impl Enclosure for ComplexBall {
    fn worst_accuracy_bits(&self) -> i64 {
        self.accuracy_bits()
    }
    fn any_indeterminate(&self) -> bool {
        self.is_indeterminate()
    }
}

impl Enclosure for RealBall {
    fn worst_accuracy_bits(&self) -> i64 {
        self.accuracy_bits()
    }
    fn any_indeterminate(&self) -> bool {
        self.is_indeterminate()
    }
}

impl<T: Enclosure> Enclosure for Vec<T> {
    fn worst_accuracy_bits(&self) -> i64 {
        self.iter()
            .map(Enclosure::worst_accuracy_bits)
            .min()
            .unwrap_or(i64::MAX)
    }
    fn any_indeterminate(&self) -> bool {
        self.iter().any(Enclosure::any_indeterminate)
    }
}

impl<T: Enclosure, const N: usize> Enclosure for [T; N] {
    fn worst_accuracy_bits(&self) -> i64 {
        self.iter()
            .map(Enclosure::worst_accuracy_bits)
            .min()
            .unwrap_or(i64::MAX)
    }
    fn any_indeterminate(&self) -> bool {
        self.iter().any(Enclosure::any_indeterminate)
    }
}

impl<A: Enclosure, B: Enclosure> Enclosure for (A, B) {
    fn worst_accuracy_bits(&self) -> i64 {
        self.0
            .worst_accuracy_bits()
            .min(self.1.worst_accuracy_bits())
    }
    fn any_indeterminate(&self) -> bool {
        self.0.any_indeterminate() || self.1.any_indeterminate()
    }
}

/// Run `f` at the precision [`Precision`] asks for, refining when it asks for
/// accuracy rather than for a precision.
///
/// The refusal carries the accuracy that *was* reached and the precision it was
/// reached at, so a caller can tell "the value is zero" (accuracy stays at
/// `i64::MIN` however much precision is spent) from "this needs more room".
pub(crate) fn evaluate_at<T, F>(
    function: &'static str,
    precision: Precision,
    f: F,
) -> Result<T, ThetaError>
where
    T: Enclosure,
    F: Fn(u32) -> Result<T, ThetaError>,
{
    precision.validate()?;
    let target = match precision {
        Precision::Bits(b) => return f(b),
        Precision::AccurateTo(t) => t,
    };
    let ceiling = precision.ceiling_bits();
    let mut working = precision.initial_working_bits();
    loop {
        let value = f(working)?;
        if !value.any_indeterminate() && value.worst_accuracy_bits() >= i64::from(target) {
            return Ok(value);
        }
        if working >= ceiling {
            let achieved = if value.any_indeterminate() {
                None
            } else {
                Some(value.worst_accuracy_bits())
            };
            return Err(ThetaError::InsufficientAccuracy {
                function,
                requested_bits: target,
                achieved_bits: achieved,
                at_precision: working,
            });
        }
        working = working.saturating_mul(2).min(ceiling);
    }
}

// ---------------------------------------------------------------------------
// Backend operation selectors
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub(crate) enum CBinop {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CUnop {
    Neg,
    Conj,
    Sqrt,
    Exp,
    Log,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RBinop {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RUnop {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Gamma,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ModularFn {
    Eta,
    JInvariant,
    Lambda,
    Delta,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum EllipticFn {
    P,
    PPrime,
    Zeta,
    Sigma,
}

/// Is the genus-`g` Riemann theta path (FLINT's `acb_theta`) compiled in?
///
/// `false` on a build whose FLINT predates 3.2; every `riemann_*` entry point
/// then refuses with `E-THETA-001`. The genus-1 modular functions may still be
/// available — check [`arb_backend_available`].
pub fn riemann_theta_available() -> bool {
    backend::riemann_theta_supported()
}

/// Is the Arb ball layer (FLINT >= 3.1) available in this build?
pub fn arb_backend_available() -> bool {
    backend::ensure_available().is_ok()
}
