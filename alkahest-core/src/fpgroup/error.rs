//! [`FpGroupError`] — every way the finitely-presented-group subsystem refuses.
//!
//! # Why this is its own enum
//!
//! `GroupError` (permutation groups, `E-GRP-*`) is a public **exhaustive** enum
//! re-exported from `alkahest_cas::stable`'s neighbourhood, so adding a variant
//! to it is a semver-major change that `cargo semver-checks` reports as
//! `enum_variant_added`. This enum is therefore separate and
//! **`#[non_exhaustive]` from birth**: every future refusal of this subsystem
//! can be added without a break. Refusals that come from a module this one
//! *calls* — `group` for the permutation representation, `matrix::normal_form`
//! for Smith and Hermite forms — arrive through the delegating variants
//! [`FpGroupError::Permutation`] and [`FpGroupError::NormalForm`], which forward
//! `code()` and `remediation()` to the error they wrap so that one condition
//! keeps one code.
//!
//! # The distinction this enum exists to keep
//!
//! [`FpGroupError::EnumerationIncomplete`] and
//! [`FpGroupError::ProvablyInfinite`] are **different codes with different
//! meanings** and must never be conflated:
//!
//! * `E-FPGRP-004` — coset enumeration hit its cap. The word problem for
//!   finitely presented groups is undecidable, so this says only *"I did not
//!   finish"*. It is **not** a claim that the group is infinite, and it is not
//!   a claim that it is finite either. Raising the cap may answer it; nothing
//!   guarantees that.
//! * `E-FPGRP-005` — the group is **proved** infinite, because its
//!   abelianisation `G/[G,G]` has an infinite cyclic factor, which is a Smith
//!   normal form computation that always terminates. This is a verdict about
//!   the group, not a report about this implementation.

use crate::errors::AlkahestError;
use crate::group::GroupError;
use crate::matrix::normal_form::NormalFormError;
use std::fmt;

/// Refusals from finitely-presented groups, Todd–Coxeter enumeration,
/// Reidemeister–Schreier and low-degree group cohomology.
///
/// `#[non_exhaustive]`: match with a `_` arm. New variants will be added.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FpGroupError {
    /// A word letter names a generator this presentation does not have.
    ///
    /// Letters are **signed and 1-based**: `+3` is the third generator and `-3`
    /// its inverse. `0` is not a letter — the identity is the empty word.
    InvalidGenerator {
        /// The offending letter, exactly as supplied.
        letter: i32,
        /// The number of generators the presentation has.
        rank: usize,
    },
    /// The presentation itself is malformed: repeated or empty generator names,
    /// or a rank above [`super::MAX_FREE_RANK`].
    InvalidPresentation {
        /// What is wrong, concretely.
        detail: String,
    },
    /// A word given as text could not be parsed.
    WordSyntax {
        /// Byte offset into the input where parsing stopped.
        position: usize,
        /// What was expected there.
        detail: String,
    },
    /// Coset enumeration did not complete within its cap.
    ///
    /// **This is not a claim that the group is infinite.** It is not a claim
    /// that it is finite either. The word problem for finitely presented groups
    /// is undecidable, and no coset enumeration can distinguish "needs a bigger
    /// cap" from "will never terminate": `⟨a, b | a², b³, (ab)⁷⟩` is infinite
    /// and `⟨a, b | a², b³, (ab)⁵⟩` is `A₅`, and the enumerator sees the same
    /// thing in both until one of them stops.
    ///
    /// When a group *is* provably infinite for an abelian reason,
    /// [`FpGroupError::ProvablyInfinite`] (`E-FPGRP-005`) says so instead.
    EnumerationIncomplete {
        /// Live cosets when the enumerator gave up.
        live_cosets: usize,
        /// Cosets ever defined, dead ones included.
        cosets_defined: usize,
        /// The cap that was in force.
        max_cosets: usize,
    },
    /// The group is **infinite**, proved rather than suspected.
    ///
    /// The abelianisation `G/[G,G]` is the Smith normal form of the relation
    /// matrix, which always terminates; an infinite cyclic factor in it forces
    /// `G` itself to be infinite. Unlike
    /// [`FpGroupError::EnumerationIncomplete`] this is a statement about the
    /// group: raising the coset cap will not change it.
    ProvablyInfinite {
        /// The abelianisation that witnesses it, e.g. `"Z^2"`.
        abelian_invariants: String,
    },
    /// The requested coset cap is above what this build will allocate.
    ///
    /// The table costs `cap · 2 · rank` machine words, so the ceiling is on the
    /// product, not on the cap alone.
    CosetCapTooLarge {
        /// Cosets requested.
        requested: usize,
        /// Largest cap admissible at this rank.
        max: usize,
    },
    /// A coset outside `0 .. index` was named. Cosets are **0-based**, and
    /// coset `0` is the subgroup `H` itself.
    CosetOutOfRange {
        /// The offending coset.
        coset: usize,
        /// The index `[G:H]`, one past the largest valid coset.
        index: usize,
    },
    /// Cohomology was asked for in a degree this implementation does not have.
    ///
    /// Only `H⁰`, `H¹` and `H²` are implemented; see the module docs for why
    /// the bar resolution stops there.
    UnsupportedCohomologyDegree {
        /// The degree asked for.
        degree: usize,
        /// The largest degree available.
        max: usize,
    },
    /// The cohomology computation is above the size this implementation admits.
    ///
    /// The bar resolution's cochain group in degree `d + 1` has
    /// `rank(M) · |G|^(d+1)` generators, so `H²` of a group of order 20 already
    /// asks for an 8000-column integer matrix. Refusing is deliberate: a
    /// general-looking answer that took an hour and cannot be checked is worse
    /// than a stated limit.
    CohomologyTooLarge {
        /// Which limit was exceeded, and by how much.
        detail: String,
    },
    /// The module's shape is wrong: bad invariants, a non-square action matrix,
    /// the wrong number of action matrices, or a rank above
    /// [`super::MAX_MODULE_RANK`].
    ModuleShape {
        /// What is wrong, concretely.
        detail: String,
    },
    /// The supplied matrices do not define an action of this group on this
    /// module.
    ///
    /// Two things are checked, and both must hold or the "cohomology" of the
    /// thing is meaningless: each matrix must map the module's relation lattice
    /// into itself (so that it is an endomorphism of `M` at all), and the
    /// assignment must be a homomorphism — `A(gx) ≡ A(g)·A(x)` for every group
    /// element `g` and every generator `x`, which is exactly the statement that
    /// every relator acts as the identity.
    ActionNotWellDefined {
        /// Which of the two failed, and where.
        detail: String,
    },
    /// A permutation-group operation underneath this one refused. Carries
    /// `E-GRP-*` unchanged.
    Permutation(GroupError),
    /// A Smith or Hermite normal form underneath this one refused. Carries
    /// `E-NFM-*` unchanged.
    NormalForm(NormalFormError),
    /// An internal invariant did not hold.
    ///
    /// Present so that a "cannot happen" branch is a **typed refusal rather
    /// than a panic**: these functions run under a PyO3 boundary, where a panic
    /// arrives as `pyo3_runtime.PanicException` — a `BaseException` that a
    /// caller's `except Exception` does not catch. Every one of these is a bug
    /// in this module.
    Internal {
        /// What was violated.
        detail: String,
    },
}

impl fmt::Display for FpGroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FpGroupError::InvalidGenerator { letter, rank } => write!(
                f,
                "word letter {letter} names no generator of a presentation with {rank} of them; \
                 letters are signed and 1-based (+1 is the first generator, -1 its inverse) and \
                 0 is not a letter"
            ),
            FpGroupError::InvalidPresentation { detail } => {
                write!(f, "malformed presentation: {detail}")
            }
            FpGroupError::WordSyntax { position, detail } => {
                write!(f, "cannot parse word at byte {position}: {detail}")
            }
            FpGroupError::EnumerationIncomplete {
                live_cosets,
                cosets_defined,
                max_cosets,
            } => write!(
                f,
                "coset enumeration did not complete within {max_cosets} cosets ({live_cosets} \
                 live of {cosets_defined} defined); this is NOT a claim that the group is \
                 infinite, nor that it is finite"
            ),
            FpGroupError::ProvablyInfinite { abelian_invariants } => write!(
                f,
                "the group is infinite: its abelianisation is {abelian_invariants}, which has an \
                 infinite cyclic factor, so no finite order exists to report"
            ),
            FpGroupError::CosetCapTooLarge { requested, max } => write!(
                f,
                "a coset cap of {requested} is above the ceiling of {max} for this rank; the \
                 table costs 2 * rank machine words per coset"
            ),
            FpGroupError::CosetOutOfRange { coset, index } => write!(
                f,
                "coset {coset} is outside 0..{index}; cosets are 0-based and coset 0 is the \
                 subgroup H itself"
            ),
            FpGroupError::UnsupportedCohomologyDegree { degree, max } => write!(
                f,
                "group cohomology is implemented for degrees 0..={max}; H^{degree} is not \
                 available"
            ),
            FpGroupError::CohomologyTooLarge { detail } => {
                write!(f, "cohomology computation too large: {detail}")
            }
            FpGroupError::ModuleShape { detail } => write!(f, "malformed module: {detail}"),
            FpGroupError::ActionNotWellDefined { detail } => write!(
                f,
                "the supplied matrices do not define an action of this group on this module: \
                 {detail}"
            ),
            FpGroupError::Permutation(e) => write!(f, "{e}"),
            FpGroupError::NormalForm(e) => write!(f, "{e}"),
            FpGroupError::Internal { detail } => {
                write!(f, "internal fp-group invariant violated: {detail}")
            }
        }
    }
}

impl std::error::Error for FpGroupError {}

impl From<GroupError> for FpGroupError {
    fn from(e: GroupError) -> Self {
        FpGroupError::Permutation(e)
    }
}

impl From<NormalFormError> for FpGroupError {
    fn from(e: NormalFormError) -> Self {
        FpGroupError::NormalForm(e)
    }
}

impl AlkahestError for FpGroupError {
    fn code(&self) -> &'static str {
        match self {
            FpGroupError::InvalidGenerator { .. } => "E-FPGRP-001",
            FpGroupError::InvalidPresentation { .. } => "E-FPGRP-002",
            FpGroupError::WordSyntax { .. } => "E-FPGRP-003",
            FpGroupError::EnumerationIncomplete { .. } => "E-FPGRP-004",
            FpGroupError::ProvablyInfinite { .. } => "E-FPGRP-005",
            FpGroupError::CosetCapTooLarge { .. } => "E-FPGRP-006",
            FpGroupError::CosetOutOfRange { .. } => "E-FPGRP-007",
            FpGroupError::UnsupportedCohomologyDegree { .. } => "E-FPGRP-008",
            FpGroupError::CohomologyTooLarge { .. } => "E-FPGRP-009",
            FpGroupError::ModuleShape { .. } => "E-FPGRP-010",
            FpGroupError::ActionNotWellDefined { .. } => "E-FPGRP-011",
            FpGroupError::Internal { .. } => "E-FPGRP-012",
            FpGroupError::Permutation(e) => e.code(),
            FpGroupError::NormalForm(e) => e.code(),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            FpGroupError::InvalidGenerator { .. } => Some(
                "build words with `FreeGroup::word`, `FreeGroup::parse` or `Word::generator`, \
                 whose letters are checked against the rank",
            ),
            FpGroupError::InvalidPresentation { .. } => Some(
                "generator names must be non-empty and pairwise distinct, and the rank must be \
                 at most MAX_FREE_RANK",
            ),
            FpGroupError::WordSyntax { .. } => Some(
                "the syntax is a product of powers of generator names and parenthesised \
                 sub-words, e.g. `a^2`, `(a*b)^5`, `a*b^-1*a^-1*b`; `1` is the identity",
            ),
            FpGroupError::EnumerationIncomplete { .. } => Some(
                "raise `max_cosets` if the group may simply be large, or use \
                 `abelian_invariants()` — which always terminates — to look for an infinite \
                 cyclic factor; do NOT read this as evidence either way about finiteness",
            ),
            FpGroupError::ProvablyInfinite { .. } => Some(
                "ask for `abelian_invariants()` rather than `order()`, or pass to a \
                 finite-index subgroup or a finite quotient",
            ),
            FpGroupError::CosetCapTooLarge { .. } => Some(
                "lower the cap, or reduce the number of generators — the table is 2 * rank \
                 words wide per coset",
            ),
            FpGroupError::CosetOutOfRange { .. } => {
                Some("cosets are 0-based: the valid range is 0..index()")
            }
            FpGroupError::UnsupportedCohomologyDegree { .. } => Some(
                "only H^0, H^1 and H^2 are implemented; H^2 is the one that classifies central \
                 extensions, and higher degrees need a resolution this module does not build",
            ),
            FpGroupError::CohomologyTooLarge { .. } => Some(
                "use a smaller group or a smaller module: the degree-(d+1) cochain group has \
                 rank(M) * |G|^(d+1) generators, so the cost is cubic in |G| for H^2",
            ),
            FpGroupError::ModuleShape { .. } => Some(
                "invariants are non-negative (0 means a Z summand, d >= 1 means Z/d), and there \
                 must be one square rank-by-rank action matrix per group generator",
            ),
            FpGroupError::ActionNotWellDefined { .. } => Some(
                "check that every relator's matrix product is the identity on M and that each \
                 generator's matrix maps the relation lattice into itself; for a trivial action \
                 use `GModule::trivial`, which cannot fail either check",
            ),
            FpGroupError::Internal { .. } => {
                Some("this is a bug: please report it with the presentation that produced it")
            }
            FpGroupError::Permutation(e) => e.remediation(),
            FpGroupError::NormalForm(e) => e.remediation(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_own_variant_has_a_distinct_code_in_the_expected_range() {
        let all = [
            FpGroupError::InvalidGenerator { letter: 0, rank: 2 },
            FpGroupError::InvalidPresentation {
                detail: String::new(),
            },
            FpGroupError::WordSyntax {
                position: 0,
                detail: String::new(),
            },
            FpGroupError::EnumerationIncomplete {
                live_cosets: 1,
                cosets_defined: 2,
                max_cosets: 3,
            },
            FpGroupError::ProvablyInfinite {
                abelian_invariants: "Z".into(),
            },
            FpGroupError::CosetCapTooLarge {
                requested: 2,
                max: 1,
            },
            FpGroupError::CosetOutOfRange { coset: 5, index: 3 },
            FpGroupError::UnsupportedCohomologyDegree { degree: 3, max: 2 },
            FpGroupError::CohomologyTooLarge {
                detail: String::new(),
            },
            FpGroupError::ModuleShape {
                detail: String::new(),
            },
            FpGroupError::ActionNotWellDefined {
                detail: String::new(),
            },
            FpGroupError::Internal {
                detail: String::new(),
            },
        ];
        let codes: Vec<&str> = all.iter().map(|e| e.code()).collect();
        assert_eq!(
            codes,
            [
                "E-FPGRP-001",
                "E-FPGRP-002",
                "E-FPGRP-003",
                "E-FPGRP-004",
                "E-FPGRP-005",
                "E-FPGRP-006",
                "E-FPGRP-007",
                "E-FPGRP-008",
                "E-FPGRP-009",
                "E-FPGRP-010",
                "E-FPGRP-011",
                "E-FPGRP-012",
            ]
        );
        for e in &all {
            assert!(e.remediation().is_some(), "{e} has no remediation");
        }
    }

    /// The two refusals a caller must not confuse have to *stay* unconfusable:
    /// different codes, and prose that says which is which.
    #[test]
    fn incomplete_enumeration_is_not_a_claim_of_infiniteness() {
        let incomplete = FpGroupError::EnumerationIncomplete {
            live_cosets: 1000,
            cosets_defined: 1000,
            max_cosets: 1000,
        };
        let infinite = FpGroupError::ProvablyInfinite {
            abelian_invariants: "Z^2".into(),
        };
        assert_ne!(incomplete.code(), infinite.code());
        assert!(incomplete.to_string().contains("NOT a claim"));
        assert!(infinite.to_string().contains("is infinite"));
    }

    /// Wrapped errors keep their own identity — otherwise one condition would
    /// have two codes depending on which module a caller entered through.
    #[test]
    fn delegating_variants_forward_code_and_remediation() {
        let inner = GroupError::DegreeMismatch { left: 3, right: 4 };
        let wrapped: FpGroupError = inner.clone().into();
        assert_eq!(wrapped.code(), inner.code());
        assert_eq!(wrapped.remediation(), inner.remediation());
        assert_eq!(wrapped.to_string(), inner.to_string());

        let inner = NormalFormError::IncompatibleMultiply {
            left_cols: 2,
            right_rows: 3,
        };
        let wrapped: FpGroupError = inner.clone().into();
        assert_eq!(wrapped.code(), inner.code());
        assert_eq!(wrapped.remediation(), inner.remediation());
    }
}
