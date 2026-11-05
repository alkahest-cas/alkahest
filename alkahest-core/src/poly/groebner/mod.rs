//! Gröbner basis computation over ℚ — a multimodular engine (F4 modulo
//! word-size primes, lifted and verified over ℚ; the default under grevlex),
//! Buchberger's algorithm ([`buchberger::compute_buchberger_basis`]), a
//! signature-based F5-family algorithm ([`compute_groebner_basis_f5`]), and the
//! grevlex-then-FGLM strategy for lex ([`GroebnerBasis::compute_lex`]).
//!
//! # Quick start
//!
//! ```rust,ignore
//! use alkahest_cas::poly::groebner::{GroebnerBasis, MonomialOrder};
//! use alkahest_cas::poly::groebner::ideal::GbPoly;
//!
//! // Ideal: (x^2 - 1, x - 1) in 1 variable
//! let f = GbPoly {
//!     terms: [(vec![2], rug::Rational::from(1)), (vec![0], rug::Rational::from(-1))].into_iter().collect(),
//!     n_vars: 1,
//! };
//! let g = GbPoly {
//!     terms: [(vec![1], rug::Rational::from(1)), (vec![0], rug::Rational::from(-1))].into_iter().collect(),
//!     n_vars: 1,
//! };
//! let gb = GroebnerBasis::compute(vec![f, g], MonomialOrder::Lex);
//! assert_eq!(gb.generators().len(), 1); // {x - 1}
//! ```

pub mod buchberger;
#[cfg(feature = "groebner-cuda")]
pub mod cuda;
pub mod f5;
pub mod fglm;
pub mod ideal;
mod modp;
pub mod monomial_order;
mod multimodular;
mod pairs;
pub mod parametric;
pub mod paramfield;
pub mod reduce;

// Keep f4 as a thin re-export so any external crate that depended on the old
// module path still compiles (semver-minor compat shim).
pub mod f4 {
    pub use super::buchberger::compute_buchberger_basis as compute_groebner_basis;
}

pub use buchberger::compute_buchberger_basis;
#[cfg(feature = "groebner-cuda")]
pub use cuda::{compute_groebner_basis_gpu, GpuBackendReport, GpuGroebnerError, MacaulayMatrix};
pub use f5::compute_groebner_basis_f5;
pub use fglm::{fglm, grevlex_staircase, is_zero_dimensional};
pub use ideal::GbPoly;
pub use monomial_order::MonomialOrder;
pub use parametric::{ParamGbPoly, ParamGroebnerBasis, ParamGroebnerError};
pub use paramfield::{ParamPoly, QParam};
pub use reduce::reduce;

/// Run `f` with budget interrupt points armed, and turn a recorded trip into
/// an error so the partial value `f` returned is never seen.
fn armed<T>(f: impl FnOnce() -> T) -> Result<T, crate::budget::BudgetTrip> {
    let guard = crate::budget::arm();
    let value = f();
    match guard.trip() {
        Some(trip) => {
            if guard.is_outermost() {
                crate::budget::clear_trip();
            }
            Err(trip)
        }
        None => Ok(value),
    }
}

/// How sure a [`GroebnerBasis`] is of being the basis of the ideal it was asked
/// about.  See [`GroebnerBasis::proof_status`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GroebnerProofStatus {
    /// The generators are the reduced Gröbner basis of exactly `⟨input⟩`.
    Proved,
    /// The generators are, verified exactly, the reduced Gröbner basis of an
    /// ideal **containing** `⟨input⟩`; that the two ideals are equal rests on
    /// random primes having been lucky.
    Probabilistic,
}

impl GroebnerProofStatus {
    /// `"proved"` or `"probabilistic"`.
    pub fn as_str(self) -> &'static str {
        match self {
            GroebnerProofStatus::Proved => "proved",
            GroebnerProofStatus::Probabilistic => "probabilistic",
        }
    }
}

/// A computed Gröbner basis.
#[derive(Clone, Debug)]
pub struct GroebnerBasis {
    generators: Vec<GbPoly>,
    order: MonomialOrder,
    proof_status: GroebnerProofStatus,
}

/// Does every generator have a single total degree?  Exactly the inputs for
/// which a verified multimodular lift is proved to be the basis of `⟨gens⟩`.
fn all_homogeneous(gens: &[GbPoly]) -> bool {
    gens.iter().all(|g| {
        let mut degrees = g.terms.keys().map(|e| e.iter().sum::<u32>());
        match degrees.next() {
            Some(d) => degrees.all(|x| x == d),
            None => true,
        }
    })
}

/// Inputs so small that the exact algorithm is as fast as any other, so there
/// is no reason to settle for less than a proof: one non-zero generator,
/// a non-zero constant among them, or every generator of degree at most one.
fn trivially_exact(gens: &[GbPoly]) -> bool {
    let is_constant = |g: &GbPoly| g.terms.keys().all(|e| e.iter().all(|&x| x == 0));
    gens.iter().filter(|g| !g.is_zero()).count() <= 1
        || gens.iter().any(|g| !g.is_zero() && is_constant(g))
        || gens
            .iter()
            .all(|g| g.terms.keys().all(|e| e.iter().sum::<u32>() <= 1))
}

/// Is this basis `{1}`?
fn is_unit_basis(gens: &[GbPoly]) -> bool {
    matches!(gens, [g] if g.terms.len() == 1
        && g.terms.keys().all(|e| e.iter().all(|&x| x == 0)))
}

/// Decide `1 ∈ ⟨gens⟩` with a proof, for a `{1}` from the multimodular engine,
/// which passes the verification over ℚ vacuously.
///
/// Homogenise with a new smallest variable `h` and run the same engine: on
/// homogeneous input its result `B` is proved to be the basis of `⟨gens^h⟩`.
/// Setting `h = 1` in `Σ cᵢ·gᵢ^h = hᵏ` gives `Σ cᵢ'·gᵢ = 1`, and conversely
/// `1 ∈ ⟨gens⟩` puts a power of `h` in `⟨gens^h⟩`, whose leading monomial only a
/// pure power of `h` in `B` can divide under grevlex.  So `1 ∈ ⟨gens⟩` exactly
/// when `B` contains one.
///
/// `Some(false)` means the unit basis was **wrong**: every prime was unlucky.
/// `None` means the run was inconclusive or an armed budget fired.
///
/// This costs a homogenised multimodular run: 0.02 s to 2.4 s on cyclic-5,6 and
/// katsura-5..7 plus a generic quadric, where Buchberger over ℚ took over 180 s.
fn prove_unit_ideal(gens: &[GbPoly]) -> Option<bool> {
    let n = gens.iter().find(|g| !g.is_zero())?.n_vars;
    let homogenised: Vec<GbPoly> = gens
        .iter()
        .filter(|g| !g.is_zero())
        .map(|g| {
            let degree = g.terms.keys().map(|e| e.iter().sum::<u32>()).max()?;
            let terms = g
                .terms
                .iter()
                .map(|(e, c)| {
                    let mut with_h = e.clone();
                    with_h.push(degree - e.iter().sum::<u32>());
                    (with_h, c.clone())
                })
                .collect();
            Some(GbPoly {
                terms,
                n_vars: n + 1,
            })
        })
        .collect::<Option<_>>()?;
    let basis = multimodular::groebner_basis(&homogenised, MonomialOrder::GRevLex).ok()?;
    Some(
        basis
            .iter()
            .any(|g| g.terms.keys().all(|e| e[..n].iter().all(|&x| x == 0))),
    )
}

impl GroebnerBasis {
    /// Compute a Gröbner basis for the given generators under the given order.
    ///
    /// Under [`MonomialOrder::GRevLex`] the basis is computed multimodularly
    /// (F4 modulo word-size primes, Chinese remaindering, rational
    /// reconstruction) and the lift is verified over ℚ before it is returned —
    /// see the `multimodular` module; when that path is not applicable or not
    /// conclusive, and under the other orders, it is Buchberger's algorithm
    /// over ℚ ([`compute_buchberger_basis`]).  Either way the result is the
    /// reduced (monic, interreduced) basis.
    ///
    /// A multimodular result for an inhomogeneous input is only
    /// [`GroebnerProofStatus::Probabilistic`]; check [`Self::proof_status`], or
    /// use [`Self::compute_exact`] when the answer has to be a proof.
    pub fn compute(gens: Vec<GbPoly>, order: MonomialOrder) -> Self {
        if order == MonomialOrder::GRevLex && !trivially_exact(&gens) {
            match multimodular::groebner_basis(&gens, order) {
                Ok(generators) => {
                    let proof_status = if all_homogeneous(&gens) {
                        GroebnerProofStatus::Proved
                    } else if is_unit_basis(&generators) {
                        match prove_unit_ideal(&gens) {
                            Some(true) => GroebnerProofStatus::Proved,
                            Some(false) => return Self::compute_exact(gens, order),
                            None => GroebnerProofStatus::Probabilistic,
                        }
                    } else {
                        GroebnerProofStatus::Probabilistic
                    };
                    return GroebnerBasis {
                        generators,
                        order,
                        proof_status,
                    };
                }
                // Stopped by an armed budget: the caller discards this.
                Err(multimodular::Unavailable::Interrupted) => {
                    return GroebnerBasis {
                        generators: Vec::new(),
                        order,
                        proof_status: GroebnerProofStatus::Proved,
                    }
                }
                Err(multimodular::Unavailable::Fallback) => {}
            }
        }
        Self::compute_exact(gens, order)
    }

    /// Compute the basis by Buchberger's algorithm over ℚ, whatever the order.
    /// The result is always [`GroebnerProofStatus::Proved`], and much slower
    /// than [`Self::compute`] on hard grevlex inputs.
    pub fn compute_exact(gens: Vec<GbPoly>, order: MonomialOrder) -> Self {
        let generators = compute_buchberger_basis(gens, order);
        GroebnerBasis {
            generators,
            order,
            proof_status: GroebnerProofStatus::Proved,
        }
    }

    /// How sure this basis is of being the reduced Gröbner basis of the input.
    ///
    /// [`GroebnerProofStatus::Proved`] comes from every exact route (Buchberger,
    /// F5, [`Self::compute_exact`]) and from a multimodular run on homogeneous
    /// input, where the verification over ℚ and the Hilbert function force
    /// `⟨B⟩ = I` (Arnold, JSC 35 (2003), Thm 7.1).
    ///
    /// [`GroebnerProofStatus::Probabilistic`] comes from a multimodular run on
    /// inhomogeneous input and from a lex basis converted from one by FGLM.
    /// Checked exactly: `B` is a reduced Gröbner basis and `I ⊆ ⟨B⟩`.  Not
    /// checked: `⟨B⟩ ⊆ I`.  Since `I ⊆ ⟨B⟩`, equal leading terms would give
    /// equal ideals, so `⟨B⟩ ⊋ I` needs a majority of the primes used, at least
    /// two independent ones drawn from the ≈5·10⁷ in `[2³⁰, 2³¹)`, to have
    /// leading terms that `I` does not have.  Those primes divide a fixed
    /// non-zero integer depending only on the input, so if `k` of them exist the
    /// chance is of order `(k / 5·10⁷)²`.
    ///
    /// Consequences for a probabilistic `B`: `p ∉ ⟨B⟩` proves `p ∉ I`, while
    /// `p ∈ ⟨B⟩` is only as sure as `B` is; every zero of `B` is a zero of `I`,
    /// but `I` could have more.  `{1}` gets no evidence from the verification
    /// at all, so it is proved separately, by a homogenised run; it stays
    /// probabilistic only if that run is inconclusive, and then
    /// [`Self::compute_lex`] (and so `solve`) falls back to Buchberger.
    pub fn proof_status(&self) -> GroebnerProofStatus {
        self.proof_status
    }

    /// Wrap an already-computed set of generators, without recomputing.
    ///
    /// The caller asserts that `generators` really is a Gröbner basis under
    /// `order`; nothing here checks it.  This exists so that the specialisation
    /// of a [`parametric::ParamGroebnerBasis`] — which is a Gröbner basis by
    /// [`parametric::ParamGroebnerBasis::specialize`]'s contract — can be
    /// handed back as an ordinary [`GroebnerBasis`] instead of being recomputed
    /// from scratch.
    pub fn from_generators(generators: Vec<GbPoly>, order: MonomialOrder) -> Self {
        GroebnerBasis {
            generators,
            order,
            proof_status: GroebnerProofStatus::Proved,
        }
    }

    /// Compute a lex Gröbner basis using the grevlex-then-FGLM strategy.
    ///
    /// For 0-dimensional ideals this is typically orders of magnitude faster
    /// than direct lex Buchberger. Falls back to direct lex Buchberger when
    /// the ideal is positive-dimensional or FGLM fails.
    pub fn compute_lex(gens: Vec<GbPoly>) -> Self {
        let n_vars = gens.first().map(|g| g.n_vars).unwrap_or(0);

        if n_vars <= 1 {
            return Self::compute(gens, MonomialOrder::Lex);
        }

        // Step 1: GRevLex basis (fast).
        let mut grevlex = Self::compute(gens.clone(), MonomialOrder::GRevLex);
        // A `{1}` that could not be proved (see `prove_unit_ideal`): "this system
        // has no solutions" is the worst thing to get wrong, so settle it exactly.
        if grevlex.proof_status == GroebnerProofStatus::Probabilistic
            && is_unit_basis(&grevlex.generators)
            && !crate::budget::interrupt_pending()
        {
            grevlex = Self::compute_exact(gens.clone(), MonomialOrder::GRevLex);
        }
        if crate::budget::interrupt_pending() {
            // Stopped by an armed budget: the caller discards this.
            return grevlex;
        }
        if is_unit_basis(&grevlex.generators) {
            return GroebnerBasis {
                order: MonomialOrder::Lex,
                ..grevlex
            };
        }

        // Step 2: Try FGLM.  What it adds is verified exactly against the
        // grevlex basis, so the result is exactly as sure as that basis is.
        if is_zero_dimensional(&grevlex.generators, n_vars) {
            if let Some(generators) = fglm(&grevlex.generators, n_vars) {
                return GroebnerBasis {
                    generators,
                    order: MonomialOrder::Lex,
                    proof_status: grevlex.proof_status,
                };
            }
        }

        // Fallback: direct lex Buchberger.
        Self::compute(gens, MonomialOrder::Lex)
    }

    /// Compute a Gröbner basis with the signature-based (F5-family) engine in
    /// [`f5`]: incremental, position-over-term signatures under `order`, with
    /// the syzygy and rewrite criteria.  Exact rational arithmetic throughout;
    /// [`Self::compute`] (multimodular under grevlex) is the faster default.
    ///
    /// Under [`MonomialOrder::Lex`] a zero-dimensional ideal is computed under
    /// grevlex and converted by FGLM, as [`Self::compute_lex`] does: direct
    /// lex is the hardest order for any completion algorithm.  The result is
    /// the same reduced basis either way.
    pub fn compute_f5(gens: Vec<GbPoly>, order: MonomialOrder) -> Self {
        let n_vars = gens.first().map(|g| g.n_vars).unwrap_or(0);
        if order == MonomialOrder::Lex && n_vars > 1 {
            let grb = compute_groebner_basis_f5(gens.clone(), MonomialOrder::GRevLex);
            if crate::budget::interrupt_pending() {
                return GroebnerBasis {
                    generators: grb,
                    order: MonomialOrder::GRevLex,
                    proof_status: GroebnerProofStatus::Proved,
                };
            }
            if is_zero_dimensional(&grb, n_vars) {
                if let Some(generators) = fglm(&grb, n_vars) {
                    return GroebnerBasis {
                        generators,
                        order,
                        proof_status: GroebnerProofStatus::Proved,
                    };
                }
            }
        }
        let generators = compute_groebner_basis_f5(gens, order);
        GroebnerBasis {
            generators,
            order,
            proof_status: GroebnerProofStatus::Proved,
        }
    }

    /// [`Self::compute`], honouring the active [`crate::budget`].
    ///
    /// The infallible constructors above have no way to report a stop, so
    /// they run to completion whatever budget is active. This one polls the
    /// budget (wall clock, `max_steps`, `max_bytes`, cancellation) once per
    /// S-pair and every [`crate::budget::INTERRUPT_POLL_INTERVAL`] division
    /// steps, and returns the trip instead of a partial basis — a basis that
    /// stopped part-way is not a Gröbner basis of anything.
    ///
    /// # Errors
    ///
    /// The [`crate::budget::BudgetTrip`] that stopped the computation
    /// (`E-BUDGET-001` wall clock, `-002` steps, `-003` cancelled, `-004`
    /// memory).
    pub fn try_compute(
        gens: Vec<GbPoly>,
        order: MonomialOrder,
    ) -> Result<Self, crate::budget::BudgetTrip> {
        armed(|| Self::compute(gens, order))
    }

    /// [`Self::compute_lex`] (grevlex, then FGLM), honouring the active budget
    /// as [`Self::try_compute`] does.
    ///
    /// # Errors
    ///
    /// The [`crate::budget::BudgetTrip`] that stopped the computation.
    pub fn try_compute_lex(gens: Vec<GbPoly>) -> Result<Self, crate::budget::BudgetTrip> {
        armed(|| Self::compute_lex(gens))
    }

    /// [`Self::compute_f5`], honouring the active budget as
    /// [`Self::try_compute`] does.
    ///
    /// # Errors
    ///
    /// The [`crate::budget::BudgetTrip`] that stopped the computation.
    pub fn try_compute_f5(
        gens: Vec<GbPoly>,
        order: MonomialOrder,
    ) -> Result<Self, crate::budget::BudgetTrip> {
        armed(|| Self::compute_f5(gens, order))
    }

    /// Return the basis generators (interreduced, monic).
    pub fn generators(&self) -> &[GbPoly] {
        &self.generators
    }

    /// The monomial order the generators were reduced under.
    ///
    /// Needed to interpret leading terms of [`Self::generators`] and to build a
    /// compatible polynomial before calling [`Self::reduce`].
    pub fn order(&self) -> MonomialOrder {
        self.order
    }

    /// Reduce a polynomial by this basis. Returns the remainder.
    pub fn reduce(&self, p: &GbPoly) -> GbPoly {
        reduce(p, &self.generators, self.order)
    }

    /// Test ideal membership: p is in the ideal iff `reduce(p) == 0`.
    pub fn contains(&self, p: &GbPoly) -> bool {
        self.reduce(p).is_zero()
    }

    /// Return the number of generators.
    pub fn len(&self) -> usize {
        self.generators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.generators.is_empty()
    }

    /// Eliminate a set of variables (given by their indices into the
    /// exponent vector).  Under a `Lex` basis with eliminated variables
    /// placed first, the elimination theorem says the generators whose
    /// leading monomial mentions none of the eliminated variables form a
    /// Gröbner basis for the elimination ideal `I ∩ k[remaining vars]`.
    ///
    /// This implementation is more permissive: it drops any generator
    /// whose support touches an eliminated variable in *any* monomial.
    /// That is strictly correct for the common Lex layout used by the
    /// V1-4 solver (eliminated variables appear with *only* positive
    /// exponents in the generators that are being dropped).
    pub fn eliminate(&self, vars: &[usize]) -> GroebnerBasis {
        let generators: Vec<GbPoly> = self
            .generators
            .iter()
            .filter(|g| {
                !g.terms
                    .keys()
                    .any(|e| vars.iter().any(|&i| e.get(i).copied().unwrap_or(0) > 0))
            })
            .cloned()
            .collect();
        GroebnerBasis {
            generators,
            order: self.order,
            proof_status: self.proof_status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod proof_status {
        use super::*;
        use GroebnerProofStatus::{Probabilistic, Proved};

        fn poly(terms: &[(&[u32], i64)]) -> GbPoly {
            GbPoly {
                terms: terms
                    .iter()
                    .map(|(e, c)| (e.to_vec(), rug::Rational::from(*c)))
                    .collect(),
                n_vars: terms[0].0.len(),
            }
        }

        /// `x² + y² − 1`, `xy − 1`: zero-dimensional and inhomogeneous.
        fn inhomogeneous() -> Vec<GbPoly> {
            vec![
                poly(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -1)]),
                poly(&[(&[1, 1], 1), (&[0, 0], -1)]),
            ]
        }

        /// `x² + y² − 1`, `x² + y² − 2`: inconsistent.
        fn inconsistent() -> Vec<GbPoly> {
            vec![
                poly(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -1)]),
                poly(&[(&[2, 0], 1), (&[0, 2], 1), (&[0, 0], -2)]),
            ]
        }

        #[test]
        fn homogeneous_multimodular_is_proved() {
            let gens = vec![
                poly(&[(&[2, 0], 1), (&[0, 2], -1)]),
                poly(&[(&[1, 1], 1), (&[0, 2], 1)]),
            ];
            let gb = GroebnerBasis::compute(gens, MonomialOrder::GRevLex);
            assert_eq!(gb.proof_status(), Proved);
        }

        #[test]
        fn inhomogeneous_multimodular_is_probabilistic_and_so_is_what_derives_from_it() {
            let gb = GroebnerBasis::compute(inhomogeneous(), MonomialOrder::GRevLex);
            assert_eq!(gb.proof_status(), Probabilistic);
            assert_eq!(gb.eliminate(&[0]).proof_status(), Probabilistic);
            let lex = GroebnerBasis::compute_lex(inhomogeneous());
            assert_eq!(lex.order(), MonomialOrder::Lex);
            assert_eq!(lex.proof_status(), Probabilistic);
            // The status is about the ideal, not the generators: same basis.
            let exact = GroebnerBasis::compute_exact(inhomogeneous(), MonomialOrder::Lex);
            assert_eq!(exact.proof_status(), Proved);
            let terms = |g: &GroebnerBasis| {
                let mut t: Vec<_> = g.generators().iter().map(|p| p.terms.clone()).collect();
                t.sort();
                t
            };
            assert_eq!(terms(&lex), terms(&exact));
        }

        #[test]
        fn exact_routes_are_proved() {
            for gb in [
                GroebnerBasis::compute(inhomogeneous(), MonomialOrder::GrLex),
                GroebnerBasis::compute(inhomogeneous(), MonomialOrder::Lex),
                GroebnerBasis::compute_f5(inhomogeneous(), MonomialOrder::GRevLex),
                GroebnerBasis::compute_f5(inhomogeneous(), MonomialOrder::Lex),
                GroebnerBasis::compute_exact(inhomogeneous(), MonomialOrder::GRevLex),
            ] {
                assert_eq!(gb.proof_status(), Proved, "{:?}", gb.order());
            }
            // Trivial inputs are exact under grevlex too.
            let one = vec![poly(&[(&[2, 0], 1), (&[0, 0], -1)])];
            let linear = vec![
                poly(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], -1)]),
                poly(&[(&[1, 0], 1), (&[0, 1], -1)]),
            ];
            for gens in [one, linear] {
                let gb = GroebnerBasis::compute(gens, MonomialOrder::GRevLex);
                assert_eq!(gb.proof_status(), Proved);
            }
        }

        /// An exponent the modular engine cannot represent falls back to
        /// Buchberger, and the status follows the path actually taken.
        #[test]
        fn the_buchberger_fallback_is_proved() {
            let gens = vec![
                poly(&[(&[70_000, 0], 1), (&[0, 1], 1), (&[0, 0], -1)]),
                poly(&[(&[0, 2], 1), (&[0, 0], -1)]),
            ];
            let gb = GroebnerBasis::compute(gens, MonomialOrder::GRevLex);
            assert_eq!(gb.proof_status(), Proved);
        }

        /// `{1}` passes the verification vacuously, so it needs its own proof.
        #[test]
        fn unit_ideal_is_proved_by_the_homogenised_run() {
            let gb = GroebnerBasis::compute(inconsistent(), MonomialOrder::GRevLex);
            assert!(is_unit_basis(gb.generators()));
            assert_eq!(gb.proof_status(), Proved);
            let lex = GroebnerBasis::compute_lex(inconsistent());
            assert!(is_unit_basis(lex.generators()));
            assert_eq!(lex.order(), MonomialOrder::Lex);
            assert_eq!(lex.proof_status(), Proved);
        }

        /// The check decides both ways: a consistent system that looks like the
        /// unit ideal modulo `p | N` (`⟨y², y² + N·x − 1⟩` has the zero `x = 1/N`)
        /// is not one.  This is what catches a run of unlucky primes.
        #[test]
        fn prove_unit_ideal_decides_both_ways() {
            assert_eq!(prove_unit_ideal(&inconsistent()), Some(true));
            let n = 1_073_741_827_i64; // a prime in the engine's range
            let gens = vec![
                poly(&[(&[0, 2], 1)]),
                poly(&[(&[0, 2], 1), (&[1, 0], n), (&[0, 0], -1)]),
            ];
            assert_eq!(prove_unit_ideal(&gens), Some(false));
            assert_eq!(prove_unit_ideal(&inhomogeneous()), Some(false));
        }
    }

    mod budget {
        use super::super::f5::tests::cyclic_system;
        use super::*;
        use crate::budget::{self, Budget, BudgetError, BudgetTrip};
        use std::time::{Duration, Instant};

        fn terms(gb: &GroebnerBasis) -> Vec<&std::collections::BTreeMap<Vec<u32>, rug::Rational>> {
            gb.generators().iter().map(|g| &g.terms).collect()
        }

        /// A step budget of zero trips at the first interrupt point, on every
        /// `try_*` entry point, and the trip is returned — not a basis.
        #[test]
        fn try_entry_points_report_a_step_trip() {
            let gens = cyclic_system(4);
            let _g = budget::enter(Budget::new().with_max_steps(0));
            let steps = |r: Result<GroebnerBasis, BudgetTrip>| {
                matches!(r, Err(BudgetTrip::Budget(BudgetError::Steps { .. })))
            };
            assert!(steps(GroebnerBasis::try_compute(
                gens.clone(),
                MonomialOrder::GRevLex
            )));
            assert!(steps(GroebnerBasis::try_compute_lex(gens.clone())));
            assert!(steps(GroebnerBasis::try_compute_f5(
                gens,
                MonomialOrder::GRevLex
            )));
            // The trip was handed back, not left behind for a later call.
            assert!(budget::take_trip().is_none());
        }

        /// Without arming, the infallible constructors ignore the budget
        /// exactly as before, and agree with the `try_*` result under a
        /// generous one.
        #[test]
        fn infallible_constructors_are_unchanged() {
            let gens = cyclic_system(4);
            let reference = GroebnerBasis::compute(gens.clone(), MonomialOrder::GRevLex);
            {
                let _g = budget::enter(Budget::new().with_max_steps(0));
                let unarmed = GroebnerBasis::compute(gens.clone(), MonomialOrder::GRevLex);
                assert_eq!(terms(&unarmed), terms(&reference));
            }
            let _g = budget::enter(Budget::new().with_wall(Duration::from_secs(600)));
            let budgeted = GroebnerBasis::try_compute(gens, MonomialOrder::GRevLex)
                .expect("a generous budget must not trip");
            assert_eq!(terms(&budgeted), terms(&reference));
        }

        /// The report's repro: cyclic-6 under grevlex ran for minutes with any
        /// budget. Under a 200 ms wall budget every engine now stops within a
        /// small multiple of the limit (generous tolerance for loaded CI
        /// machines).  Grevlex is multimodular and finishes cyclic-6 in well
        /// under the budget, so it gets cyclic-7; the ℚ Buchberger engine
        /// (grlex) still gets cyclic-6.
        #[test]
        fn cyclic6_stops_within_the_wall_budget() {
            let _g = budget::enter(Budget::new().with_wall(Duration::from_millis(200)));
            let t = Instant::now();
            let wall = |r: Result<GroebnerBasis, BudgetTrip>| {
                matches!(r, Err(BudgetTrip::Budget(BudgetError::WallClock { .. })))
            };
            assert!(wall(GroebnerBasis::try_compute(
                cyclic_system(6),
                MonomialOrder::GrLex
            )));
            assert!(wall(GroebnerBasis::try_compute(
                cyclic_system(7),
                MonomialOrder::GRevLex
            )));
            let r5 = GroebnerBasis::try_compute_f5(cyclic_system(7), MonomialOrder::GRevLex);
            assert!(r5.is_err(), "F5 on cyclic-7 cannot finish in 200 ms");
            assert!(
                t.elapsed() < crate::test_time::limit_secs(10),
                "took {:?}",
                t.elapsed()
            );
        }
    }

    #[test]
    fn eliminate_drops_generators() {
        let xm_y = GbPoly {
            terms: [
                (vec![1u32, 0], rug::Rational::from(1)),
                (vec![0, 1], rug::Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let two_y2_m1 = GbPoly {
            terms: [
                (vec![0, 2], rug::Rational::from(2)),
                (vec![0, 0], rug::Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let gb = GroebnerBasis::compute(vec![xm_y, two_y2_m1], MonomialOrder::Lex);
        let elim = gb.eliminate(&[0]);
        assert_eq!(elim.generators().len(), 1);
        for term in elim.generators()[0].terms.keys() {
            assert_eq!(term[0], 0, "eliminated variable x must not appear");
        }
    }

    #[test]
    fn compute_lex_circle_parabola() {
        // Same system as fglm.rs test — verify the public API path.
        let f = GbPoly {
            terms: [
                (vec![2u32, 0], rug::Rational::from(1)),
                (vec![0, 2], rug::Rational::from(1)),
                (vec![0, 0], rug::Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let g = GbPoly {
            terms: [
                (vec![0u32, 1], rug::Rational::from(1)),
                (vec![2, 0], rug::Rational::from(-1)),
            ]
            .into_iter()
            .collect(),
            n_vars: 2,
        };
        let gb_lex = GroebnerBasis::compute_lex(vec![f.clone(), g.clone()]);
        let gb_direct = GroebnerBasis::compute(vec![f, g], MonomialOrder::Lex);
        for p in gb_direct.generators() {
            assert!(gb_lex.contains(p), "FGLM basis missing generator");
        }
        for p in gb_lex.generators() {
            assert!(gb_direct.contains(p), "direct basis missing FGLM generator");
        }
    }
}
