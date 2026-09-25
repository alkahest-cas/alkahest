//! Weight distributions, weight enumerator polynomials and the MacWilliams
//! transform.
//!
//! The weight distribution of a code of length `n` is `A_0 … A_n`, where `A_i`
//! counts the codewords of Hamming weight `i`. For a linear code `A_0 = 1`
//! always (the zero word, and only it) and `Σ A_i = |C| = q^k`. Both are
//! asserted by property tests.
//!
//! The **weight enumerator** packages the same data as the homogeneous
//! polynomial
//!
//! ```text
//!   W_C(x, y) = Σ_i A_i x^(n-i) y^i
//! ```
//!
//! and the **MacWilliams identity** relates it to the dual code's:
//!
//! ```text
//!   W_{C⊥}(x, y) = (1/|C|) · W_C(x + (q-1)y, x - y)
//! ```
//!
//! which in coefficients is `B_j = (1/|C|) Σ_i A_i K_j(i; n, q)` with `K_j` the
//! Krawtchouk polynomial. That is the form implemented here, because it is the
//! one that stays in exact integer arithmetic: the division by `|C|` is exact
//! for a genuine linear code, and [`WeightEnumerator::macwilliams`] refuses
//! (`E-CODE-006`) when it is not, rather than returning a fractional "weight
//! distribution" that would quietly announce that the input was never a code.
//!
//! # Scope
//!
//! [`WeightEnumerator`] holds exact [`rug::Integer`] multiplicities and renders
//! itself; it is **not** an [`crate::kernel::ExprId`] and does not go through
//! the simplifier. Turning one into a symbolic expression is a caller's job and
//! is a one-line fold over [`WeightEnumerator::coefficients`].
//!
//! Only the *complete* weight distribution is modelled. Split and complete
//! weight enumerators (which track the value in each coordinate, not just
//! whether it is non-zero) are not here.

use rug::ops::Pow;
use rug::Integer;
use std::fmt;

use super::error::CodingError;
use super::krawtchouk::krawtchouk;

/// The weight enumerator of a code of length `n` over an alphabet of size `q`.
///
/// Constructed from a weight distribution `A_0 … A_n`, which is validated:
/// a linear code has `A_0 = 1` and no negative multiplicity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightEnumerator {
    n: usize,
    q: u64,
    coeffs: Vec<Integer>,
}

impl WeightEnumerator {
    /// Build from a weight distribution.
    ///
    /// # Errors
    ///
    /// `E-CODE-001` for an empty distribution, `E-CODE-008` for `q < 2`, and
    /// `E-CODE-006` when `A_0 ≠ 1` or some `A_i < 0` — neither can come from a
    /// linear code, and a "dual" computed from one is meaningless.
    pub fn new(q: u64, coeffs: Vec<Integer>) -> Result<Self, CodingError> {
        if coeffs.is_empty() {
            return Err(CodingError::InvalidLength { n: 0 });
        }
        if q < 2 {
            return Err(CodingError::InvalidAlphabet {
                q: q.to_string(),
                reason: "an alphabet needs at least two symbols".to_string(),
            });
        }
        if coeffs[0] != 1 {
            return Err(CodingError::MalformedDistribution {
                reason: format!(
                    "A_0 = {} but a linear code contains the zero word exactly once",
                    coeffs[0]
                ),
            });
        }
        if let Some(i) = coeffs.iter().position(|c| *c < 0) {
            return Err(CodingError::MalformedDistribution {
                reason: format!("A_{i} = {} is negative", coeffs[i]),
            });
        }
        let n = coeffs.len() - 1;
        Ok(Self { n, q, coeffs })
    }

    /// The length of the code.
    pub fn length(&self) -> usize {
        self.n
    }

    /// The alphabet size.
    pub fn alphabet_size(&self) -> u64 {
        self.q
    }

    /// The weight distribution `A_0 … A_n`.
    pub fn coefficients(&self) -> &[Integer] {
        &self.coeffs
    }

    /// `A_i`, or zero outside `0..=n`.
    pub fn coefficient(&self, i: usize) -> Integer {
        self.coeffs
            .get(i)
            .cloned()
            .unwrap_or_else(|| Integer::from(0))
    }

    /// `|C| = Σ_i A_i`.
    pub fn size(&self) -> Integer {
        self.coeffs.iter().fold(Integer::from(0), |a, c| a + c)
    }

    /// The minimum distance: the least `i ≥ 1` with `A_i > 0`.
    ///
    /// `None` for the zero code, whose only word is the zero word and which has
    /// no minimum distance (it is conventionally `n`, or `∞`, or undefined, and
    /// picking one silently would be a guess).
    pub fn minimum_distance(&self) -> Option<usize> {
        self.coeffs
            .iter()
            .skip(1)
            .position(|c| *c > 0)
            .map(|i| i + 1)
    }

    /// `W_C(x, y) = Σ_i A_i x^(n-i) y^i` at integer arguments.
    pub fn evaluate(&self, x: &Integer, y: &Integer) -> Integer {
        let mut total = Integer::from(0);
        for (i, ai) in self.coeffs.iter().enumerate() {
            if *ai == 0 {
                continue;
            }
            let mut term = ai.clone();
            term *= x.clone().pow((self.n - i) as u32);
            term *= y.clone().pow(i as u32);
            total += term;
        }
        total
    }

    /// The MacWilliams transform: the weight enumerator of the dual code.
    ///
    /// `B_j = (1/|C|) Σ_i A_i K_j(i; n, q)`.
    ///
    /// For a linear `[n, k]` code the result is the weight enumerator of `C⊥`,
    /// an `[n, n-k]` code, and applying the transform twice is the identity.
    ///
    /// # Errors
    ///
    /// `E-CODE-006` when some `Σ_i A_i K_j(i)` is not divisible by `|C|`, or
    /// when the quotient is negative or has `B_0 ≠ 1`. Each of those says the
    /// input was not the weight enumerator of a linear code over GF(q); the
    /// division is exact whenever it was.
    pub fn macwilliams(&self) -> Result<WeightEnumerator, CodingError> {
        let size = self.size();
        if size == 0 {
            return Err(CodingError::MalformedDistribution {
                reason: "the code is empty, so there is nothing to divide by".to_string(),
            });
        }
        let mut out = Vec::with_capacity(self.n + 1);
        for j in 0..=self.n {
            let mut acc = Integer::from(0);
            for (i, ai) in self.coeffs.iter().enumerate() {
                if *ai == 0 {
                    continue;
                }
                acc += ai * krawtchouk(j, i as i64, self.n, self.q);
            }
            let (quot, rem) = acc.clone().div_rem(size.clone());
            if rem != 0 {
                return Err(CodingError::MalformedDistribution {
                    reason: format!(
                        "the MacWilliams sum at j = {j} is {acc}, which is not divisible \
                         by |C| = {size}; for a linear code over GF({}) it always is, so \
                         this input is not one",
                        self.q
                    ),
                });
            }
            if quot < 0 {
                return Err(CodingError::MalformedDistribution {
                    reason: format!("the transform gives B_{j} = {quot} < 0"),
                });
            }
            out.push(quot);
        }
        WeightEnumerator::new(self.q, out)
    }

    /// Whether every Delsarte dual-feasibility constraint holds:
    /// `Σ_i A_i K_k(i; n, q) ≥ 0` for `k = 0 … n`.
    ///
    /// True for the distance distribution of *any* code over an alphabet of
    /// size `q`, linear or not — it is exactly the hypothesis the linear
    /// programme in [`super::delsarte_lp_bound`] optimises under. Cheap, and a
    /// good sanity check on a hand-entered distribution.
    pub fn is_dual_feasible(&self) -> bool {
        (0..=self.n).all(|k| {
            let mut acc = Integer::from(0);
            for (i, ai) in self.coeffs.iter().enumerate() {
                if *ai == 0 {
                    continue;
                }
                acc += ai * krawtchouk(k, i as i64, self.n, self.q);
            }
            acc >= 0
        })
    }
}

impl fmt::Display for WeightEnumerator {
    /// Renders `W_C(x, y)`, highest power of `x` first, omitting zero terms.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, ai) in self.coeffs.iter().enumerate() {
            if *ai == 0 {
                continue;
            }
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            let xdeg = self.n - i;
            let mut wrote = false;
            if *ai != 1 || (xdeg == 0 && i == 0) {
                write!(f, "{ai}")?;
                wrote = true;
            }
            if xdeg > 0 {
                if wrote {
                    write!(f, "*")?;
                }
                write!(f, "x")?;
                if xdeg > 1 {
                    write!(f, "^{xdeg}")?;
                }
                wrote = true;
            }
            if i > 0 {
                if wrote {
                    write!(f, "*")?;
                }
                write!(f, "y")?;
                if i > 1 {
                    write!(f, "^{i}")?;
                }
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}
