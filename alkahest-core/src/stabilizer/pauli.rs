//! The `n`-qubit Pauli group in `(x | z)` symplectic form, and abelian
//! subgroups of it.
//!
//! # How an operator is stored, exactly
//!
//! A [`PauliOperator`] is
//!
//! ```text
//!     P  =  i^e · X^{x_1}Z^{z_1} ⊗ X^{x_2}Z^{z_2} ⊗ … ⊗ X^{x_n}Z^{z_n}
//! ```
//!
//! with `x, z ∈ F_2^n` and the phase exponent `e ∈ Z_4`. The bits are the
//! `(x | z)` symplectic vector; the phase is carried alongside and plays no part
//! in the commutation test.
//!
//! # The `Display` / parse convention — read this before typing a generator
//!
//! Strings use the **Hermitian** Pauli letters: `I`, `X`, `Y`, `Z`, where `Y`
//! is the usual Hermitian `Y = iXZ`, **not** the raw symplectic product `XZ`.
//! A leading `+`, `-`, `i`, `+i` or `-i` gives the overall sign.
//!
//! The two conventions differ by one factor of `i` per `Y`, so the conversion
//! is explicit in both directions:
//!
//! ```text
//!     letters  =  i^y · X^x Z^z          y = #{ j : x_j = z_j = 1 }
//!     P        =  i^e · X^x Z^z  =  i^(e − y) · letters
//! ```
//!
//! and [`PauliOperator::to_string`] prints `i^(e − y)` as its prefix. That is
//! why `"+Y"` round-trips to an operator whose stored `phase()` is `1` rather
//! than `0`: `Y = i·XZ`.
//!
//! The payoff is that `"+Y"` is Hermitian, which is what a stabilizer generator
//! has to be, so generators transcribed from a paper are Hermitian without the
//! caller doing anything. [`StabilizerGroup`] enforces Hermiticity
//! ([`PauliOperator::is_hermitian`]) rather than assuming it.

use std::fmt;
use std::str::FromStr;

use super::{check_qubits, matrix_from_rows, StabilizerError};
use crate::ffield::GfMatrix;

/// An element of the `n`-qubit Pauli group, as `(x | z)` bits and a phase in
/// `Z₄`.
///
/// See the [module docs](self) for the storage and string conventions; they are
/// the part of this type most likely to produce a confident wrong answer if
/// skipped.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PauliOperator {
    n: usize,
    x: Vec<u8>,
    z: Vec<u8>,
    /// `e` in `P = i^e · X^x Z^z`, reduced mod 4.
    phase: u8,
}

impl PauliOperator {
    /// The identity on `n` qubits.
    ///
    /// # Errors
    ///
    /// `E-STAB-009` above [`MAX_QUBITS`](super::MAX_QUBITS).
    pub fn identity(n: usize) -> Result<Self, StabilizerError> {
        check_qubits(n)?;
        Ok(Self {
            n,
            x: vec![0; n],
            z: vec![0; n],
            phase: 0,
        })
    }

    /// From separate `x` and `z` bit vectors and a phase exponent `e ∈ Z₄`.
    ///
    /// The operator built is `i^e · X^x Z^z` *literally*: with `x_j = z_j = 1`
    /// and `e = 0` this is `XZ = −iY`, not `Y`. Use
    /// [`PauliOperator::hermitian`] or [`str::parse`] if what you meant was the
    /// Hermitian `Y`.
    ///
    /// # Errors
    ///
    /// `E-STAB-002` when `x` and `z` have different lengths; `E-STAB-010` for a
    /// bit outside `{0, 1}`; `E-STAB-009` above
    /// [`MAX_QUBITS`](super::MAX_QUBITS).
    pub fn from_xz(x: &[u8], z: &[u8], phase: u8) -> Result<Self, StabilizerError> {
        if x.len() != z.len() {
            return Err(StabilizerError::ShapeMismatch {
                op: "build a Pauli operator",
                expected: "x and z of the same length".to_string(),
                got: format!("lengths {} and {}", x.len(), z.len()),
            });
        }
        check_qubits(x.len())?;
        for (&a, &b) in x.iter().zip(z) {
            if a > 1 || b > 1 {
                return Err(StabilizerError::MalformedPauli {
                    reason: format!("bits must be 0 or 1; got {}", a.max(b)),
                });
            }
        }
        Ok(Self {
            n: x.len(),
            x: x.to_vec(),
            z: z.to_vec(),
            phase: phase % 4,
        })
    }

    /// From a `(x | z)` symplectic vector of length `2n` and a phase exponent.
    ///
    /// # Errors
    ///
    /// `E-STAB-002` when the length is odd; `E-STAB-010` for a bit outside
    /// `{0, 1}`; `E-STAB-009` above [`MAX_QUBITS`](super::MAX_QUBITS).
    pub fn from_symplectic(bits: &[u8], phase: u8) -> Result<Self, StabilizerError> {
        if bits.len() % 2 != 0 {
            return Err(StabilizerError::ShapeMismatch {
                op: "build a Pauli operator",
                expected: "a symplectic vector of even length 2n".to_string(),
                got: format!("length {}", bits.len()),
            });
        }
        let n = bits.len() / 2;
        Self::from_xz(&bits[..n], &bits[n..], phase)
    }

    /// The **Hermitian** Pauli with the given symplectic type, `+` or `−`.
    ///
    /// This picks the unique phase making the operator Hermitian:
    /// `e = (z·x) + 2·[negative]`, so a qubit with `x_j = z_j = 1` contributes
    /// the `Y` of the textbooks rather than `XZ`.
    ///
    /// # Errors
    ///
    /// As [`PauliOperator::from_xz`].
    pub fn hermitian(x: &[u8], z: &[u8], negative: bool) -> Result<Self, StabilizerError> {
        let p = Self::from_xz(x, z, 0)?;
        let y = p.y_count();
        Self::from_xz(x, z, ((y % 4) as u8 + if negative { 2 } else { 0 }) % 4)
    }

    /// Number of qubits.
    pub fn qubits(&self) -> usize {
        self.n
    }

    /// The `X` exponents.
    pub fn x_bits(&self) -> &[u8] {
        &self.x
    }

    /// The `Z` exponents.
    pub fn z_bits(&self) -> &[u8] {
        &self.z
    }

    /// The phase exponent `e` in `P = i^e · X^x Z^z`.
    pub fn phase(&self) -> u8 {
        self.phase
    }

    /// The `(x | z)` symplectic vector, length `2n`.
    pub fn symplectic(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(2 * self.n);
        v.extend_from_slice(&self.x);
        v.extend_from_slice(&self.z);
        v
    }

    /// `#{ j : x_j = z_j = 1 }` — the number of `Y` positions.
    fn y_count(&self) -> usize {
        self.x
            .iter()
            .zip(&self.z)
            .filter(|(&a, &b)| a & b == 1)
            .count()
    }

    /// The number of qubits the operator acts on non-trivially.
    ///
    /// The phase is irrelevant: `−I` has weight `0`.
    pub fn weight(&self) -> usize {
        self.x
            .iter()
            .zip(&self.z)
            .filter(|(&a, &b)| (a | b) == 1)
            .count()
    }

    /// The symplectic-type letter at a qubit: `'I'`, `'X'`, `'Y'` or `'Z'`.
    ///
    /// Returns `None` for a qubit index out of range.
    pub fn letter(&self, qubit: usize) -> Option<char> {
        match (self.x.get(qubit)?, self.z.get(qubit)?) {
            (0, 0) => Some('I'),
            (1, 0) => Some('X'),
            (0, 1) => Some('Z'),
            _ => Some('Y'),
        }
    }

    /// Is `P = P†`?
    ///
    /// `i^e X^x Z^z` is Hermitian exactly when `e ≡ z·x (mod 2)`. Every
    /// stabilizer generator must be, which is why [`StabilizerGroup::new`]
    /// checks it.
    pub fn is_hermitian(&self) -> bool {
        (self.phase % 2) as usize == self.y_count() % 2
    }

    /// Is this the identity, phase included?
    pub fn is_identity(&self) -> bool {
        self.phase == 0 && self.x.iter().all(|&b| b == 0) && self.z.iter().all(|&b| b == 0)
    }

    /// Is this `±I` or `±iI` — the identity up to a phase?
    pub fn is_scalar(&self) -> bool {
        self.x.iter().all(|&b| b == 0) && self.z.iter().all(|&b| b == 0)
    }

    fn same_size(&self, other: &Self) -> Result<(), StabilizerError> {
        if self.n != other.n {
            return Err(StabilizerError::QubitCountMismatch {
                left: self.n,
                right: other.n,
            });
        }
        Ok(())
    }

    /// The group product `self · other`, phase included.
    ///
    /// The phase arithmetic is the whole content of this function. Writing
    /// `self = i^a X^{x₁}Z^{z₁}` and `other = i^b X^{x₂}Z^{z₂}`, moving
    /// `X^{x₂}` left past `Z^{z₁}` costs `(−1)^{z₁·x₂}`, so
    ///
    /// ```text
    ///     self · other  =  i^(a + b + 2·(z₁·x₂)) · X^{x₁+x₂} Z^{z₁+z₂}
    /// ```
    ///
    /// with the exponent sums taken mod 2 (because `X² = Z² = I`, with no
    /// phase) and the phase mod 4.
    ///
    /// # Errors
    ///
    /// `E-STAB-003` when the two operators are on different numbers of qubits.
    pub fn mul(&self, other: &Self) -> Result<Self, StabilizerError> {
        self.same_size(other)?;
        let mut cross = 0usize;
        for i in 0..self.n {
            cross += usize::from(self.z[i] & other.x[i]);
        }
        let phase = (self.phase + other.phase + 2 * ((cross % 2) as u8)) % 4;
        let x: Vec<u8> = self.x.iter().zip(&other.x).map(|(a, b)| a ^ b).collect();
        let z: Vec<u8> = self.z.iter().zip(&other.z).map(|(a, b)| a ^ b).collect();
        Ok(Self {
            n: self.n,
            x,
            z,
            phase,
        })
    }

    /// The inverse `P⁻¹ = P†`.
    ///
    /// `(i^e X^x Z^z)† = i^{−e} Z^z X^x = i^{−e + 2(z·x)} X^x Z^z`.
    pub fn inverse(&self) -> Self {
        let cross = self.y_count() % 2;
        let phase = ((4 - self.phase % 4) + 2 * (cross as u8)) % 4;
        Self {
            n: self.n,
            x: self.x.clone(),
            z: self.z.clone(),
            phase,
        }
    }

    /// `−P`.
    pub fn negate(&self) -> Self {
        Self {
            n: self.n,
            x: self.x.clone(),
            z: self.z.clone(),
            phase: (self.phase + 2) % 4,
        }
    }

    /// The symplectic form `⟨self, other⟩`: `0` if they commute, `1` if they
    /// anticommute.
    ///
    /// # Errors
    ///
    /// `E-STAB-003` for different qubit counts.
    pub fn symplectic_product(&self, other: &Self) -> Result<u8, StabilizerError> {
        self.same_size(other)?;
        let mut acc = 0u8;
        for i in 0..self.n {
            acc ^= (self.x[i] & other.z[i]) ^ (self.z[i] & other.x[i]);
        }
        Ok(acc & 1)
    }

    /// Do the two operators commute?
    ///
    /// Two Paulis either commute or anticommute; there is no third case, so
    /// this is exactly `symplectic_product == 0`.
    ///
    /// # Errors
    ///
    /// `E-STAB-003` for different qubit counts.
    pub fn commutes_with(&self, other: &Self) -> Result<bool, StabilizerError> {
        Ok(self.symplectic_product(other)? == 0)
    }
}

/// `+`, `+i`, `-`, `-i` for the phase exponents `0, 1, 2, 3`.
fn phase_prefix(e: u8) -> &'static str {
    match e % 4 {
        0 => "+",
        1 => "+i",
        2 => "-",
        _ => "-i",
    }
}

impl fmt::Display for PauliOperator {
    /// Hermitian letters with a sign prefix, e.g. `+XIZY` or `-iXZZXI`.
    ///
    /// The prefix is `i^(e − y)`, where `y` is the number of `Y` positions —
    /// see the [module docs](self). `+Y` therefore prints for the operator with
    /// `phase() == 1`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let y = (self.y_count() % 4) as u8;
        let shown = (self.phase + 4 - y) % 4;
        f.write_str(phase_prefix(shown))?;
        for i in 0..self.n {
            f.write_str(match self.letter(i).unwrap_or('I') {
                'X' => "X",
                'Y' => "Y",
                'Z' => "Z",
                _ => "I",
            })?;
        }
        Ok(())
    }
}

impl FromStr for PauliOperator {
    type Err = StabilizerError;

    /// Parse `"+XIZY"`, `"-iXZZXI"`, `"XZZXI"` — Hermitian letters with an
    /// optional sign.
    ///
    /// Accepted prefixes: none, `+`, `-`, `i`, `+i`, `-i`. Letters are
    /// case-insensitive.
    ///
    /// # Errors
    ///
    /// `E-STAB-010` for an unrecognised prefix or letter; `E-STAB-009` above
    /// [`MAX_QUBITS`](super::MAX_QUBITS).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let t = s.trim();
        let (sign, rest) = if let Some(r) = t.strip_prefix("-i") {
            (3u8, r)
        } else if let Some(r) = t.strip_prefix("+i") {
            (1u8, r)
        } else if let Some(r) = t.strip_prefix('-') {
            (2u8, r)
        } else if let Some(r) = t.strip_prefix('+') {
            (0u8, r)
        } else if let Some(r) = t.strip_prefix('i') {
            (1u8, r)
        } else {
            (0u8, t)
        };
        let rest = rest.trim();
        let mut x = Vec::with_capacity(rest.len());
        let mut z = Vec::with_capacity(rest.len());
        for c in rest.chars() {
            let (a, b) = match c.to_ascii_uppercase() {
                'I' => (0u8, 0u8),
                'X' => (1, 0),
                'Y' => (1, 1),
                'Z' => (0, 1),
                other => {
                    return Err(StabilizerError::MalformedPauli {
                        reason: format!(
                            "{other:?} is not a Pauli letter; expected one of I, X, Y, Z"
                        ),
                    })
                }
            };
            x.push(a);
            z.push(b);
        }
        check_qubits(x.len())?;
        let y = x.iter().zip(&z).filter(|(&a, &b)| a & b == 1).count();
        Self::from_xz(&x, &z, (sign + (y % 4) as u8) % 4)
    }
}

// ---------------------------------------------------------------------------
// StabilizerGroup
// ---------------------------------------------------------------------------

/// An abelian subgroup of the `n`-qubit Pauli group, given by generators.
///
/// Construction **checks** the two facts a stabilizer group must satisfy and
/// refuses if either fails:
///
/// 1. the generators pairwise commute ([`StabilizerError::NotCommuting`]), and
/// 2. no product of them is `−I` ([`StabilizerError::ContainsMinusIdentity`]).
///
/// The second is only reachable from *dependent* generators: if the symplectic
/// vectors are linearly independent then no non-trivial product is scalar at
/// all. It is checked anyway, because a dependent generating list is a
/// perfectly ordinary thing to hand in and `−I ∈ S` means the stabilized
/// subspace is `{0}` — a "code" with no states.
///
/// Generators are also required to be **Hermitian**
/// ([`PauliOperator::is_hermitian`]): `i·X` generates a group containing `−I`
/// after four steps and is not an observable.
#[derive(Clone, Debug)]
pub struct StabilizerGroup {
    n: usize,
    generators: Vec<PauliOperator>,
    rank: usize,
}

impl StabilizerGroup {
    /// Build from generators, checking commutation, Hermiticity and `−I`.
    ///
    /// The qubit count is read off the first generator. An **empty** list is
    /// refused (`E-STAB-002`) rather than silently taken to mean `n = 0`: the
    /// trivial group on `n` qubits is a perfectly ordinary object — it is the
    /// stabilizer of an `[[n, n]]` code — and which `n` was meant is not
    /// recoverable from an empty list. Say so with
    /// [`StabilizerGroup::with_qubits`].
    ///
    /// # Errors
    ///
    /// `E-STAB-002` for an empty generator list; `E-STAB-003` for mixed qubit
    /// counts; `E-STAB-004` when two generators anticommute; `E-STAB-006` when
    /// a product of them is `−I`; `E-STAB-010` for a non-Hermitian generator;
    /// `E-STAB-013` if the `−I` check finds a scalar product with an odd
    /// phase, which cannot happen for commuting Hermitian operators.
    pub fn new(generators: Vec<PauliOperator>) -> Result<Self, StabilizerError> {
        let n = match generators.first() {
            Some(g) => g.qubits(),
            None => {
                return Err(StabilizerError::ShapeMismatch {
                    op: "build a stabilizer group",
                    expected: "at least one generator, so that the qubit count is determined; \
                               use `StabilizerGroup::with_qubits(n, gens)` for the trivial group"
                        .to_string(),
                    got: "an empty generator list".to_string(),
                })
            }
        };
        Self::with_qubits(n, generators)
    }

    /// Build from generators on an explicitly stated number of qubits.
    ///
    /// This is [`StabilizerGroup::new`] with the qubit count given rather than
    /// inferred, which is the only way to name the trivial group on `n > 0`
    /// qubits.
    ///
    /// # Errors
    ///
    /// As [`StabilizerGroup::new`], minus the empty-list refusal.
    pub fn with_qubits(n: usize, generators: Vec<PauliOperator>) -> Result<Self, StabilizerError> {
        check_qubits(n)?;
        for g in &generators {
            if g.qubits() != n {
                return Err(StabilizerError::QubitCountMismatch {
                    left: n,
                    right: g.qubits(),
                });
            }
            if !g.is_hermitian() {
                return Err(StabilizerError::MalformedPauli {
                    reason: format!(
                        "{g} is not Hermitian (phase exponent {}); a stabilizer generator is an \
                         observable, so its phase must be ±1 on the Hermitian letters",
                        g.phase()
                    ),
                });
            }
        }
        for i in 0..generators.len() {
            for j in (i + 1)..generators.len() {
                if generators[i].symplectic_product(&generators[j])? == 1 {
                    return Err(StabilizerError::NotCommuting { i, j });
                }
            }
        }

        let rows: Vec<Vec<u8>> = generators.iter().map(|g| g.symplectic()).collect();
        let rank = if rows.is_empty() || n == 0 {
            0
        } else {
            matrix_from_rows(&rows, 2 * n)?.rank()
        };

        // -I check. Every linear dependency among the symplectic vectors gives
        // a product with trivial symplectic part, i.e. a scalar; its phase says
        // which scalar.
        if rank < generators.len() && n > 0 {
            let m = matrix_from_rows(&rows, 2 * n)?.transpose();
            let deps = m.nullspace()?;
            let combos = super::matrix_cols(&deps)?;
            for combo in &combos {
                let idx: Vec<usize> = combo
                    .iter()
                    .enumerate()
                    .filter(|(_, &b)| b == 1)
                    .map(|(i, _)| i)
                    .collect();
                if idx.is_empty() {
                    continue;
                }
                let mut prod = PauliOperator::identity(n)?;
                for &i in &idx {
                    prod = prod.mul(&generators[i])?;
                }
                match prod.phase() {
                    0 => {}
                    2 => return Err(StabilizerError::ContainsMinusIdentity { combination: idx }),
                    other => {
                        return Err(StabilizerError::InconsistentResult {
                            reason: format!(
                                "the product of generators {idx:?} is scalar with phase \
                                 exponent {other}, which commuting Hermitian Paulis cannot be"
                            ),
                        })
                    }
                }
            }
        }

        Ok(Self {
            n,
            generators,
            rank,
        })
    }

    /// Number of qubits.
    pub fn qubits(&self) -> usize {
        self.n
    }

    /// The generators, as supplied.
    pub fn generators(&self) -> &[PauliOperator] {
        &self.generators
    }

    /// `rank(S)` — the dimension of the span of the generators' symplectic
    /// vectors, i.e. `log₂|S / {±I}|`, and the number of independent
    /// stabilizer conditions.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Are the generators independent — is `rank() == generators().len()`?
    pub fn is_independent(&self) -> bool {
        self.rank == self.generators.len()
    }

    /// The `m × 2n` check matrix over GF(2): row `i` is generator `i`'s
    /// `(x | z)` vector.
    ///
    /// # Errors
    ///
    /// `E-GFQ-012` for a shape FLINT cannot allocate.
    pub fn check_matrix(&self) -> Result<GfMatrix, StabilizerError> {
        let rows: Vec<Vec<u8>> = self.generators.iter().map(|g| g.symplectic()).collect();
        matrix_from_rows(&rows, 2 * self.n)
    }

    /// The syndrome of `error`: one bit per generator, `1` where the error
    /// anticommutes with it.
    ///
    /// # Errors
    ///
    /// `E-STAB-003` when `error` is on a different number of qubits.
    pub fn syndrome(&self, error: &PauliOperator) -> Result<Vec<u8>, StabilizerError> {
        if error.qubits() != self.n {
            return Err(StabilizerError::QubitCountMismatch {
                left: self.n,
                right: error.qubits(),
            });
        }
        self.generators
            .iter()
            .map(|g| g.symplectic_product(error))
            .collect()
    }

    /// Does `p` commute with every generator — is it in the centralizer
    /// `N(S)`?
    ///
    /// # Errors
    ///
    /// `E-STAB-003` for a different qubit count.
    pub fn centralizes(&self, p: &PauliOperator) -> Result<bool, StabilizerError> {
        Ok(self.syndrome(p)?.iter().all(|&b| b == 0))
    }

    /// Is `p` an element of `S` — phase included?
    ///
    /// A `p` whose symplectic vector is in the span but whose phase is wrong
    /// (`−g` for a generator `g`) is **not** in `S`, and this returns `false`
    /// rather than quietly ignoring the sign.
    ///
    /// # Errors
    ///
    /// `E-STAB-003` for a different qubit count; `E-GFQ-*` from the underlying
    /// solve.
    pub fn contains(&self, p: &PauliOperator) -> Result<bool, StabilizerError> {
        if p.qubits() != self.n {
            return Err(StabilizerError::QubitCountMismatch {
                left: self.n,
                right: p.qubits(),
            });
        }
        if self.generators.is_empty() || self.n == 0 {
            return Ok(p.is_identity());
        }
        let gt = self.check_matrix()?.transpose();
        let target = matrix_from_rows(&[p.symplectic()], 2 * self.n)?.transpose();
        let coeffs = match gt.solve(&target) {
            Ok(c) => c,
            Err(crate::ffield::FiniteFieldError::Inconsistent) => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        let bits = super::matrix_cols(&coeffs)?;
        let sel = &bits[0];
        let mut prod = PauliOperator::identity(self.n)?;
        for (i, &b) in sel.iter().enumerate() {
            if b == 1 {
                prod = prod.mul(&self.generators[i])?;
            }
        }
        Ok(prod == *p)
    }
}
