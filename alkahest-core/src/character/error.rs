//! [`CharacterError`] — every refusal of the conjugacy-class and
//! character-table surface.
//!
//! # Why this is its own enum, and `#[non_exhaustive]` from birth
//!
//! [`GroupError`] is a public **exhaustive** enum re-exported from
//! `alkahest_cas::stable`, so adding a variant to it is a semver-major change
//! (`cargo semver-checks` reports `enum_variant_added`). Conjugacy classes are
//! a new experimental subsystem and are not worth a major bump, so their
//! refusals live here, on an enum that is `#[non_exhaustive]` from the start —
//! which is the point: every future refusal of this subsystem can be added
//! without another break.
//!
//! # Codes
//!
//! `E-CHAR-001` … `E-CHAR-009` are this enum's own. The three delegating
//! variants — [`CharacterError::Group`], [`CharacterError::Field`] and
//! [`CharacterError::LinearAlgebra`] — carry the wrapped error's `E-GRP-NNN`,
//! `E-NUMF-NNN` or `E-GFQ-NNN` code, message and remediation through
//! unchanged, exactly as `LatticeGeometryError::Reduction` does. A failure
//! raised inside the permutation-group layer, the cyclotomic field or the
//! GF(p) matrix layer therefore keeps its own code rather than being
//! relabelled, and a caller can tell "this group is too big to enumerate"
//! (`E-GRP-004`) from "this group is too big for a character table"
//! (`E-CHAR-001`).

use crate::errors::AlkahestError;
use crate::ffield::FiniteFieldError;
use crate::group::GroupError;
use crate::numfield::NumberFieldError;
use std::fmt;

/// Why a conjugacy-class or character-table computation refused.
///
/// Every variant is a refusal, never a degraded answer. In particular there is
/// no partial character table: a table that failed one of its own
/// orthogonality checks arrives as [`CharacterError::SelfCheckFailed`] and is
/// withheld, because a caller cannot tell a checked table from an unchecked
/// one once it is in their hands.
///
/// `#[non_exhaustive]`: match with a `_` arm. New variants will be added.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CharacterError {
    /// A permutation-group operation underneath failed. Carries its
    /// `E-GRP-NNN` code unchanged.
    Group(GroupError),
    /// An operation in the cyclotomic field `ℚ(ζ_e)` failed. Carries its
    /// `E-NUMF-NNN` code unchanged.
    Field(NumberFieldError),
    /// An operation in the GF(p) matrix layer failed. Carries its `E-GFQ-NNN`
    /// code unchanged.
    LinearAlgebra(FiniteFieldError),
    /// The group is too large for this subsystem's element-by-element
    /// enumeration of conjugacy classes.
    ///
    /// The order is still exact and still available from
    /// [`crate::group::PermutationGroup::order`]; it is the class partition
    /// that is refused. Nothing partial is returned: half the classes of a
    /// group is not a coarser answer, it is a wrong one.
    OrderTooLarge {
        /// Decimal rendering of `|G|` (it may not fit a `u64`).
        order: String,
        /// The cap that was in force.
        cap: u64,
    },
    /// A caller-supplied cap exceeds this implementation's hard ceiling.
    CapTooLarge {
        /// The cap that was asked for.
        cap: u64,
        /// The largest cap this implementation accepts.
        max: u64,
    },
    /// `ℚ(ζ_e)` for `e = exp G` has degree `φ(e)` beyond the ceiling this
    /// subsystem works in.
    ///
    /// Every character value is an element of that one field, and every
    /// orthogonality check multiplies in it, so the field degree — not the
    /// group order — is what bounds the arithmetic here.
    ExponentFieldTooLarge {
        /// `exp G`, the lcm of the element orders.
        exponent: u64,
        /// `φ(exp G)`, the degree of `ℚ(ζ_e)`.
        degree: usize,
        /// The largest degree this implementation accepts.
        max: usize,
    },
    /// No prime `p ≡ 1 (mod exp G)` with `p > |G|` was found inside the search
    /// bound, so Dixon's reduction has no modulus to work over.
    NoSuitablePrime {
        /// `exp G`, which `p − 1` has to be a multiple of.
        exponent: u64,
        /// Decimal rendering of `|G|`, which `p` has to exceed.
        order: String,
        /// The largest `p` that was tried.
        searched_to: u64,
    },
    /// An element whose class was asked for is not in the group.
    ///
    /// Membership is decided exactly, by sifting through the stabilizer chain;
    /// this is never a lookup miss on a hash table that happened to be
    /// incomplete.
    NotAMember {
        /// The element, in cycle notation.
        element: String,
    },
    /// A class index is out of range.
    ClassIndexOutOfRange {
        /// The index that was asked for.
        index: usize,
        /// The number of classes there are.
        classes: usize,
    },
    /// A result was computed, failed one of this module's own invariants, and
    /// was **withheld**.
    ///
    /// This is the variant that exists so that a wrong character table is never
    /// returned. `check` names the identity that failed — `"row
    /// orthogonality"`, `"column orthogonality"`, `"degree sum"`, `"class
    /// sizes"` — and `detail` says where.
    SelfCheckFailed {
        /// Which invariant failed.
        check: &'static str,
        /// Where and how, concretely.
        detail: String,
    },
    /// Dixon's simultaneous diagonalisation of the class multiplication
    /// matrices did not split the space into one line per class.
    ///
    /// With `p ∤ |G|` and `p ≡ 1 (mod exp G)` this cannot happen for a correct
    /// implementation, so reaching it means a bug here rather than a limitation
    /// — it is a typed refusal rather than a panic because these functions are
    /// called across a PyO3 boundary.
    SplittingIncomplete {
        /// How many common eigenspaces were isolated.
        found: usize,
        /// How many there had to be.
        classes: usize,
    },
    /// An internal invariant did not hold.
    ///
    /// As with [`CharacterError::SplittingIncomplete`], this is a typed refusal
    /// rather than a panic because a panic crosses the PyO3 boundary as
    /// `pyo3_runtime.PanicException` — a `BaseException` that a caller's
    /// `except Exception` does not catch.
    Internal {
        /// What went wrong, concretely.
        detail: String,
    },
}

impl From<GroupError> for CharacterError {
    fn from(e: GroupError) -> Self {
        CharacterError::Group(e)
    }
}

impl From<NumberFieldError> for CharacterError {
    fn from(e: NumberFieldError) -> Self {
        CharacterError::Field(e)
    }
}

impl From<FiniteFieldError> for CharacterError {
    fn from(e: FiniteFieldError) -> Self {
        CharacterError::LinearAlgebra(e)
    }
}

impl fmt::Display for CharacterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CharacterError::Group(e) => write!(f, "{e}"),
            CharacterError::Field(e) => write!(f, "{e}"),
            CharacterError::LinearAlgebra(e) => write!(f, "{e}"),
            CharacterError::OrderTooLarge { order, cap } => write!(
                f,
                "the group has order {order}, above the conjugacy-class enumeration cap of \
                 {cap}; classes here are found by conjugating every element, so its order is \
                 known exactly but its classes will not be listed"
            ),
            CharacterError::CapTooLarge { cap, max } => write!(
                f,
                "a conjugacy-class cap of {cap} is above the hard ceiling of {max}"
            ),
            CharacterError::ExponentFieldTooLarge {
                exponent,
                degree,
                max,
            } => write!(
                f,
                "exp G = {exponent}, so character values live in Q(zeta_{exponent}) of degree \
                 phi({exponent}) = {degree}, above this module's ceiling of {max}"
            ),
            CharacterError::NoSuitablePrime {
                exponent,
                order,
                searched_to,
            } => write!(
                f,
                "no prime p = 1 (mod {exponent}) with p > {order} was found at or below \
                 {searched_to}"
            ),
            CharacterError::NotAMember { element } => {
                write!(f, "{element} is not an element of this group")
            }
            CharacterError::ClassIndexOutOfRange { index, classes } => write!(
                f,
                "class index {index} is out of range: this group has {classes} conjugacy classes"
            ),
            CharacterError::SelfCheckFailed { check, detail } => write!(
                f,
                "the computed result failed its own {check} check and was withheld: {detail}"
            ),
            CharacterError::SplittingIncomplete { found, classes } => write!(
                f,
                "the class multiplication matrices split the space into {found} common \
                 eigenspaces, not the {classes} required"
            ),
            CharacterError::Internal { detail } => {
                write!(f, "internal character-table invariant violated: {detail}")
            }
        }
    }
}

impl std::error::Error for CharacterError {}

impl AlkahestError for CharacterError {
    fn code(&self) -> &'static str {
        match self {
            CharacterError::Group(e) => e.code(),
            CharacterError::Field(e) => e.code(),
            CharacterError::LinearAlgebra(e) => e.code(),
            CharacterError::OrderTooLarge { .. } => "E-CHAR-001",
            CharacterError::CapTooLarge { .. } => "E-CHAR-002",
            CharacterError::ExponentFieldTooLarge { .. } => "E-CHAR-003",
            CharacterError::NoSuitablePrime { .. } => "E-CHAR-004",
            CharacterError::NotAMember { .. } => "E-CHAR-005",
            CharacterError::ClassIndexOutOfRange { .. } => "E-CHAR-006",
            CharacterError::SelfCheckFailed { .. } => "E-CHAR-007",
            CharacterError::SplittingIncomplete { .. } => "E-CHAR-008",
            CharacterError::Internal { .. } => "E-CHAR-009",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            CharacterError::Group(e) => e.remediation(),
            CharacterError::Field(e) => e.remediation(),
            CharacterError::LinearAlgebra(e) => e.remediation(),
            CharacterError::OrderTooLarge { .. } => Some(
                "use `order()` and the stabilizer chain instead, or raise the cap with \
                 `ConjugacyClasses::of_with_cap` if |G| elements really fit in memory; there is \
                 no partial class partition on offer",
            ),
            CharacterError::CapTooLarge { .. } => Some(
                "the ceiling bounds memory, not patience: the class lookup holds one degree-n \
                 images array per group element. Work with a smaller group, or with a quotient \
                 or subgroup whose table you actually need",
            ),
            CharacterError::ExponentFieldTooLarge { .. } => Some(
                "exp G drives the field, so a large cyclic factor is the usual cause; the cap \
                 is on phi(exp G) and not on |G| at all, so a group of small exponent passes \
                 this ceiling at any order and meets the |G| cap instead, while a cyclic \
                 group of order 1000 is refused here despite being small",
            ),
            CharacterError::NoSuitablePrime { .. } => Some(
                "this is a search-bound miss rather than a mathematical obstruction — \
                 Dirichlet guarantees such a prime exists; report the group as a bug",
            ),
            CharacterError::NotAMember { .. } => Some(
                "test with `PermutationGroup::contains` first; note that points are 0-based \
                 here and composition is left-to-right, so a generator transcribed from GAP \
                 needs `Permutation::from_cycles_one_based`",
            ),
            CharacterError::ClassIndexOutOfRange { .. } => {
                Some("class indices run over 0..len(), with class 0 always the identity class")
            }
            CharacterError::SelfCheckFailed { .. } => Some(
                "this is a defect in alkahest, not in the input: the orthogonality relations \
                 are implied by the group axioms, so a violation means the table is wrong. \
                 Report the generators as a minimal failing example",
            ),
            CharacterError::SplittingIncomplete { .. } => Some(
                "report the generators as a minimal failing example; with p prime to |G| and \
                 p = 1 mod exp G the class algebra is split semisimple, so this cannot happen \
                 for a correct implementation",
            ),
            CharacterError::Internal { .. } => {
                Some("report the generators as a minimal failing example")
            }
        }
    }
}
