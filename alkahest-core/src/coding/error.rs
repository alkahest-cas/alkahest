//! Refusals for classical linear codes, weight enumerators and the Delsarte
//! linear programme.
//!
//! Every variant here is a *refusal*, never a fallback: the alternative to
//! `EnumerationTooLarge` is a search that runs until the machine dies, and the
//! alternative to `LpFailure` is an "upper bound" that was never certified and
//! may be smaller than a code that actually exists. A bound that is quietly too
//! small is the worst outcome this module can produce, so the paths that could
//! produce one end in an error instead.

use std::fmt;

/// Why a coding-theory computation refused.
///
/// Codes are `E-CODE-001` … `E-CODE-008`; see [`crate::errors::codes::REGISTRY`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodingError {
    /// A code of length zero, or a generator / parity-check matrix with no
    /// columns.
    ///
    /// Length is the ambient dimension; with `n = 0` there is nothing for a
    /// weight to count and the Krawtchouk system is empty. Refused rather than
    /// returned as a degenerate "code" whose every parameter is zero.
    InvalidLength {
        /// The length that was offered.
        n: usize,
    },
    /// The minimum distance argument is outside `1 ..= n`.
    ///
    /// `d = 0` makes `A_q(n, d)` meaningless (every pair of words is at
    /// distance ≥ 0) and `d > n` makes the code empty; neither is a bound
    /// anyone can use.
    InvalidDistance {
        /// The distance that was offered.
        d: usize,
        /// The length it was offered against.
        n: usize,
    },
    /// The GF(q) backend refused a step.
    ///
    /// The underlying `E-GFQ-NNN` code and message are carried in `detail`
    /// rather than re-emitted as this error's own code: one code means one
    /// thing, and a caller branching on `E-GFQ-009` must be able to trust that
    /// it came from the finite-field subsystem.
    LinearAlgebra {
        /// The operation that was attempted, e.g. `"nullspace"`.
        op: &'static str,
        /// The underlying refusal, rendered with its own stable code.
        detail: String,
    },
    /// Exhaustive codeword enumeration would exceed its hard cap.
    ///
    /// The number of codewords is `q^k`, which grows faster than anything else
    /// in this module. There is no partial answer: a minimum distance from a
    /// truncated search is an *upper* bound on the distance presented as the
    /// distance, which is exactly the confident wrong answer this crate refuses
    /// by policy.
    EnumerationTooLarge {
        /// `q^k`, the number of codewords, in decimal.
        codewords: String,
        /// The length of the code.
        n: usize,
        /// Which cap was exceeded, in words.
        limit: String,
    },
    /// The code length exceeds the Delsarte linear programme's cap.
    ///
    /// The programme has `n` constraints and `n - d + 1` variables and is
    /// solved in exact rational arithmetic; the entries are Krawtchouk values,
    /// which reach `C(n, n/2) (q-1)^{n/2}`. The cap is empirical, not
    /// mathematical — see [`MAX_LP_LENGTH`](super::MAX_LP_LENGTH).
    LengthTooLarge {
        /// The length that was offered.
        n: usize,
        /// The cap.
        cap: usize,
    },
    /// A weight distribution cannot come from a linear code over GF(q).
    ///
    /// Reported for a vector with `A_0 ≠ 1`, a negative multiplicity, a length
    /// that is not `n + 1`, or a MacWilliams transform whose coefficients do
    /// not come out as non-negative integers after dividing by `|C|`. The last
    /// case is the useful one: it means the input was not the weight
    /// enumerator of a linear code, and returning the fractional "dual" would
    /// have hidden that.
    MalformedDistribution {
        /// What is wrong with it.
        reason: String,
    },
    /// The exact simplex did not return an optimal vertex, or the two sides of
    /// the linear programme disagreed.
    ///
    /// The Delsarte programme is provably feasible (`A = 0` satisfies every
    /// constraint) and provably bounded (see [`super::delsarte_lp_bound`]), and
    /// its primal and dual optima must coincide. Any other outcome is a bug in
    /// this module or in the simplex, and is reported rather than rounded into
    /// a number that would look like a bound.
    LpFailure {
        /// What went wrong, naming the side and the simplex status.
        detail: String,
    },
    /// The alphabet size `q` is not usable.
    ///
    /// `q < 2` is not an alphabet. For the exhaustive-enumeration paths `q` is
    /// additionally capped, because the scalar-multiple table those paths build
    /// is linear in `q`.
    InvalidAlphabet {
        /// The alphabet size that was offered, in decimal.
        q: String,
        /// Why it is unusable.
        reason: String,
    },
}

impl fmt::Display for CodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodingError::InvalidLength { n } => write!(
                f,
                "a linear code needs a positive length; got n = {n}, which leaves \
                 no coordinate for a weight to count"
            ),
            CodingError::InvalidDistance { d, n } => {
                write!(f, "the minimum distance must lie in 1..={n}; got d = {d}")
            }
            CodingError::LinearAlgebra { op, detail } => {
                write!(f, "the GF(q) backend refused `{op}`: {detail}")
            }
            CodingError::EnumerationTooLarge {
                codewords,
                n,
                limit,
            } => write!(
                f,
                "exhaustive enumeration of {codewords} codewords of length {n} is \
                 past the cap ({limit}); there is no partial answer, because a \
                 minimum distance from a truncated search is an upper bound on \
                 the distance wearing the distance's name"
            ),
            CodingError::LengthTooLarge { n, cap } => write!(
                f,
                "the Delsarte programme is capped at length {cap}; got n = {n}"
            ),
            CodingError::MalformedDistribution { reason } => write!(
                f,
                "this is not the weight distribution of a linear code: {reason}"
            ),
            CodingError::LpFailure { detail } => write!(
                f,
                "the exact simplex did not certify a bound: {detail}; the Delsarte \
                 programme is feasible and bounded by construction, so this is a bug \
                 and the (uncertified) number is withheld rather than returned"
            ),
            CodingError::InvalidAlphabet { q, reason } => {
                write!(f, "unusable alphabet size q = {q}: {reason}")
            }
        }
    }
}

impl std::error::Error for CodingError {}

impl crate::errors::AlkahestError for CodingError {
    fn code(&self) -> &'static str {
        match self {
            CodingError::InvalidLength { .. } => "E-CODE-001",
            CodingError::InvalidDistance { .. } => "E-CODE-002",
            CodingError::LinearAlgebra { .. } => "E-CODE-003",
            CodingError::EnumerationTooLarge { .. } => "E-CODE-004",
            CodingError::LengthTooLarge { .. } => "E-CODE-005",
            CodingError::MalformedDistribution { .. } => "E-CODE-006",
            CodingError::LpFailure { .. } => "E-CODE-007",
            CodingError::InvalidAlphabet { .. } => "E-CODE-008",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        crate::errors::codes::REGISTRY
            .iter()
            .find(|spec| spec.code == crate::errors::AlkahestError::code(self))
            .and_then(|spec| spec.remediation)
    }
}

/// Wrap a finite-field refusal, keeping its own code visible in the message.
pub(crate) fn wrap_ff(op: &'static str, e: crate::ffield::FiniteFieldError) -> CodingError {
    use crate::errors::AlkahestError;
    let code = AlkahestError::code(&e);
    CodingError::LinearAlgebra {
        op,
        detail: format!("{code}: {e}"),
    }
}
