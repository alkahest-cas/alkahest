//! Genus-`g` Riemann theta functions `theta_{a,b}(z, tau)` on the Siegel upper
//! half-space, and the Siegel reduction helpers that go with them.
//!
//! # Conventions
//!
//! ```text
//! theta_{a,b}(z, tau) = sum_{n in Z^g + a/2} exp(pi i n^T tau n + 2 pi i n^T (z + b/2))
//! ```
//!
//! for `a, b` in `{0,1}^g`, with `tau` a symmetric `g x g` complex matrix whose
//! imaginary part is positive definite. The `4^g` values are indexed by the
//! `2g`-bit integer `(a << g) | b`: `a` occupies the more significant `g` bits,
//! with `a_0` the most significant of those. [`theta_characteristic_index`]
//! builds that index from bit vectors so the ordering never has to be
//! rediscovered at a call site.
//!
//! In genus 1 this agrees with the classical Jacobi functions as
//! `(theta_1, theta_2, theta_3, theta_4) = (-theta_{1,1}, theta_{1,0},
//! theta_{0,0}, theta_{0,1})` — a relation this module's tests assert by
//! evaluating both FLINT code paths independently.

use super::ball::ComplexBall;
use super::error::ThetaError;
use super::{backend, evaluate_at, Precision, MAX_GENUS};

// ---------------------------------------------------------------------------
// Characteristics
// ---------------------------------------------------------------------------

/// Pack a characteristic `(a, b)` given as two `g`-long bit vectors into the
/// index FLINT uses to order theta values.
///
/// Bits are most-significant-first within each half, and `a` occupies the high
/// half: in genus 2, `a = (1, 0)`, `b = (0, 0)` is index 8.
///
/// Any non-zero entry counts as a set bit; the vectors are characteristics in
/// `{0,1}^g`, and a `2` would be a caller error that has no meaning here.
pub fn theta_characteristic_index(a: &[u8], b: &[u8]) -> Result<u64, ThetaError> {
    if a.len() != b.len() {
        return Err(ThetaError::DimensionMismatch {
            what: "theta characteristic halves",
            expected: a.len(),
            got: b.len(),
        });
    }
    let g = a.len();
    check_genus(g)?;
    let mut ab: u64 = 0;
    for &bit in a {
        ab = (ab << 1) | u64::from(bit != 0);
    }
    for &bit in b {
        ab = (ab << 1) | u64::from(bit != 0);
    }
    Ok(ab)
}

/// The inverse of [`theta_characteristic_index`]: split an index into its
/// `(a, b)` bit vectors.
pub fn theta_characteristic_bits(ab: u64, genus: usize) -> Result<(Vec<u8>, Vec<u8>), ThetaError> {
    check_genus(genus)?;
    check_characteristic(ab, genus)?;
    let mut a = vec![0u8; genus];
    let mut b = vec![0u8; genus];
    for k in 0..genus {
        a[k] = u8::from((ab >> (2 * genus - 1 - k)) & 1 == 1);
        b[k] = u8::from((ab >> (genus - 1 - k)) & 1 == 1);
    }
    Ok((a, b))
}

/// Is the characteristic `(a, b)` **even**?
///
/// `theta_{a,b}(-z, tau) = (-1)^{a . b} theta_{a,b}(z, tau)`, so the parity of
/// the dot product `a . b` decides whether the function is even or odd in `z`;
/// the odd ones vanish at `z = 0`. There are `2^{g-1}(2^g + 1)` even
/// characteristics and `2^{g-1}(2^g - 1)` odd ones.
///
/// Computed here as the parity of `popcount(a & b)`, in pure Rust, so that it
/// works on a build with no `acb_theta`. `tests::char_parity_matches_flint`
/// checks it against FLINT's own `acb_theta_char_dot` for every characteristic
/// up to genus 3.
pub fn theta_characteristic_is_even(ab: u64, genus: usize) -> Result<bool, ThetaError> {
    check_genus(genus)?;
    check_characteristic(ab, genus)?;
    let mask = (1u64 << genus) - 1;
    let a = (ab >> genus) & mask;
    let b = ab & mask;
    Ok((a & b).count_ones() % 2 == 0)
}

fn check_genus(genus: usize) -> Result<(), ThetaError> {
    if genus == 0 || genus > MAX_GENUS {
        return Err(ThetaError::GenusOutOfRange {
            genus,
            max: MAX_GENUS,
        });
    }
    Ok(())
}

fn check_characteristic(ab: u64, genus: usize) -> Result<(), ThetaError> {
    // `2 * genus <= 2 * MAX_GENUS = 12`, so the shift cannot overflow.
    if ab >= 1u64 << (2 * genus) {
        return Err(ThetaError::CharacteristicOutOfRange { ab, genus });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// SiegelMatrix
// ---------------------------------------------------------------------------

/// A symmetric `g x g` complex matrix — a candidate point of the Siegel upper
/// half-space `H_g`.
///
/// Construction checks the shape and the symmetry. It does **not** check that
/// `Im(tau)` is positive definite, because that question has a precision in it:
/// ask [`SiegelMatrix::is_certainly_in_siegel_upper_half_space`], or let
/// [`riemann_theta`] ask on your behalf (it refuses with
/// [`ThetaError::NotInSiegelUpperHalfSpace`] when the answer is no).
#[derive(Clone, Debug)]
pub struct SiegelMatrix {
    genus: usize,
    entries: Vec<ComplexBall>,
}

impl SiegelMatrix {
    /// Build from `g * g` entries in row-major order.
    ///
    /// Symmetry is checked as **ball identity**: `tau[i][j]` and `tau[j][i]`
    /// must be the same midpoint and the same radius. Two overlapping but
    /// different enclosures of one number are two different inputs, and picking
    /// one of them — or intersecting them — would be this module guessing what
    /// the caller meant. Use [`SiegelMatrix::from_upper_triangle`] when the
    /// entries were computed independently.
    pub fn new(genus: usize, entries: Vec<ComplexBall>) -> Result<Self, ThetaError> {
        check_genus(genus)?;
        if entries.len() != genus * genus {
            return Err(ThetaError::DimensionMismatch {
                what: "period matrix",
                expected: genus * genus,
                got: entries.len(),
            });
        }
        for i in 0..genus {
            for j in (i + 1)..genus {
                if !same_ball(&entries[i * genus + j], &entries[j * genus + i]) {
                    return Err(ThetaError::NotSymmetric { i, j });
                }
            }
        }
        Ok(SiegelMatrix { genus, entries })
    }

    /// Build from the upper triangle, row by row and including the diagonal:
    /// `g (g + 1) / 2` entries, ordered `(0,0), (0,1), ..., (0,g-1), (1,1), ...`.
    ///
    /// The lower triangle is filled by copying, so symmetry holds by
    /// construction and cannot be tripped by two separately-computed balls.
    pub fn from_upper_triangle(genus: usize, upper: Vec<ComplexBall>) -> Result<Self, ThetaError> {
        check_genus(genus)?;
        let want = genus * (genus + 1) / 2;
        if upper.len() != want {
            return Err(ThetaError::DimensionMismatch {
                what: "period matrix upper triangle",
                expected: want,
                got: upper.len(),
            });
        }
        let mut entries: Vec<Option<ComplexBall>> = vec![None; genus * genus];
        let mut k = 0;
        for i in 0..genus {
            for j in i..genus {
                entries[i * genus + j] = Some(upper[k].clone());
                entries[j * genus + i] = Some(upper[k].clone());
                k += 1;
            }
        }
        Ok(SiegelMatrix {
            genus,
            entries: entries
                .into_iter()
                .map(|e| e.expect("filled above"))
                .collect(),
        })
    }

    /// The genus-1 case: a single `tau` in the upper half-plane.
    pub fn genus_one(tau: ComplexBall) -> Self {
        SiegelMatrix {
            genus: 1,
            entries: vec![tau],
        }
    }

    /// The genus of this matrix (its side length).
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// Entry `(i, j)`, or `None` if either index is out of range.
    pub fn entry(&self, i: usize, j: usize) -> Option<&ComplexBall> {
        if i >= self.genus || j >= self.genus {
            return None;
        }
        Some(&self.entries[i * self.genus + j])
    }

    /// All entries, row-major.
    pub fn entries(&self) -> &[ComplexBall] {
        &self.entries
    }

    /// Is `Im(tau)` **certainly** positive definite at `prec` bits — i.e. is
    /// this matrix certainly a point of `H_g`?
    ///
    /// `false` means "not proved at this precision", never "proved false".
    /// Answered by asking FLINT for a Cholesky factor of `Im(tau)`.
    pub fn is_certainly_in_siegel_upper_half_space(&self, prec: u32) -> Result<bool, ThetaError> {
        super::arith::check_precision(prec)?;
        backend::siegel_positive_definite(&self.entries, self.genus, prec)
    }
}

fn same_ball(x: &ComplexBall, y: &ComplexBall) -> bool {
    x.midpoint_re() == y.midpoint_re()
        && x.radius_re() == y.radius_re()
        && x.midpoint_im() == y.midpoint_im()
        && x.radius_im() == y.radius_im()
}

// ---------------------------------------------------------------------------
// ThetaValues
// ---------------------------------------------------------------------------

/// The `4^g` theta values for one `(z, tau)`, indexed by characteristic.
#[derive(Clone, Debug)]
pub struct ThetaValues {
    genus: usize,
    squared: bool,
    values: Vec<ComplexBall>,
}

impl ThetaValues {
    /// The genus these values belong to.
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// Are these `theta^2` rather than `theta`?
    ///
    /// [`riemann_theta_squared`] uses a faster FLINT algorithm that computes
    /// the squares directly. Squares determine theta only up to sign, so the
    /// two are kept apart by a flag rather than silently interchanged.
    pub fn is_squared(&self) -> bool {
        self.squared
    }

    /// How many values there are: `4^g`.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Never true for a well-formed result; present because clippy asks for it
    /// alongside [`ThetaValues::len`].
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The value at characteristic index `ab`, or `None` when `ab >= 4^g`.
    pub fn get(&self, ab: u64) -> Option<&ComplexBall> {
        usize::try_from(ab).ok().and_then(|i| self.values.get(i))
    }

    /// The value at the characteristic `(a, b)` given as bit vectors.
    ///
    /// Both must have exactly `genus` entries. A shorter pair would pack into a
    /// perfectly valid index for a *smaller* genus and silently return the
    /// wrong value, so the length is checked rather than inferred.
    pub fn get_characteristic(&self, a: &[u8], b: &[u8]) -> Result<&ComplexBall, ThetaError> {
        if a.len() != self.genus {
            return Err(ThetaError::DimensionMismatch {
                what: "theta characteristic",
                expected: self.genus,
                got: a.len(),
            });
        }
        let ab = theta_characteristic_index(a, b)?;
        check_characteristic(ab, self.genus)?;
        self.get(ab).ok_or(ThetaError::CharacteristicOutOfRange {
            ab,
            genus: self.genus,
        })
    }

    /// All values, in characteristic order.
    pub fn values(&self) -> &[ComplexBall] {
        &self.values
    }

    /// Consume and return the values.
    pub fn into_values(self) -> Vec<ComplexBall> {
        self.values
    }

    /// The relative accuracy of the *worst* value in the vector — the number to
    /// look at before believing any of them.
    pub fn worst_accuracy_bits(&self) -> i64 {
        self.values
            .iter()
            .map(ComplexBall::accuracy_bits)
            .min()
            .unwrap_or(i64::MAX)
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

fn prepare(
    function: &'static str,
    z: &[ComplexBall],
    tau: &SiegelMatrix,
    bits: u32,
) -> Result<(), ThetaError> {
    if z.len() != tau.genus() {
        return Err(ThetaError::DimensionMismatch {
            what: "theta argument z",
            expected: tau.genus(),
            got: z.len(),
        });
    }
    let _ = function;
    if !tau.is_certainly_in_siegel_upper_half_space(bits)? {
        return Err(ThetaError::NotInSiegelUpperHalfSpace {
            genus: tau.genus(),
            prec: bits,
        });
    }
    Ok(())
}

fn theta_vector(
    function: &'static str,
    z: &[ComplexBall],
    tau: &SiegelMatrix,
    sqr: bool,
    precision: Precision,
) -> Result<ThetaValues, ThetaError> {
    check_genus(tau.genus())?;
    let values = evaluate_at(function, precision, |bits| {
        prepare(function, z, tau, bits)?;
        backend::theta_all(z, tau.entries(), tau.genus(), sqr, bits)
    })?;
    Ok(ThetaValues {
        genus: tau.genus(),
        squared: sqr,
        values,
    })
}

/// All `4^g` values `theta_{a,b}(z, tau)`.
///
/// `z` must have exactly `g` entries. `tau` must be provably in `H_g` at the
/// working precision, or the call refuses with
/// [`ThetaError::NotInSiegelUpperHalfSpace`].
///
/// # Cost
///
/// The output alone is `4^g` balls. Genus 1 and 2 are fast; genus 3 is
/// noticeably slower; above that see the module-level scope limits in
/// [`crate::theta`].
pub fn riemann_theta(
    z: &[ComplexBall],
    tau: &SiegelMatrix,
    precision: Precision,
) -> Result<ThetaValues, ThetaError> {
    theta_vector("riemann_theta", z, tau, false, precision)
}

/// All `4^g` values `theta_{a,b}(z, tau)^2`, by FLINT's faster squared
/// algorithm.
///
/// The result is flagged [`ThetaValues::is_squared`]. Taking a square root to
/// recover `theta` is *not* done here: the sign is not determined by the square
/// and choosing one would be a guess.
pub fn riemann_theta_squared(
    z: &[ComplexBall],
    tau: &SiegelMatrix,
    precision: Precision,
) -> Result<ThetaValues, ThetaError> {
    theta_vector("riemann_theta_squared", z, tau, true, precision)
}

/// A single value `theta_{a,b}(z, tau)` for the characteristic index `ab`.
///
/// Marginally cheaper than [`riemann_theta`] — FLINT documents `acb_theta_all`
/// as "only slightly slower" than the single-characteristic route — so prefer
/// [`riemann_theta`] when more than one characteristic is wanted.
pub fn riemann_theta_characteristic(
    z: &[ComplexBall],
    tau: &SiegelMatrix,
    ab: u64,
    precision: Precision,
) -> Result<ComplexBall, ThetaError> {
    let g = tau.genus();
    check_genus(g)?;
    check_characteristic(ab, g)?;
    evaluate_at("riemann_theta_characteristic", precision, |bits| {
        prepare("riemann_theta_characteristic", z, tau, bits)?;
        backend::theta_one(z, tau.entries(), g, ab, bits)
    })
}

// ---------------------------------------------------------------------------
// Siegel reduction
// ---------------------------------------------------------------------------

/// The result of [`siegel_reduce`]: a symplectic matrix and the point it moves
/// `tau` to.
#[derive(Clone, Debug)]
pub struct SiegelReduction {
    genus: usize,
    symplectic: Vec<rug::Integer>,
    reduced: SiegelMatrix,
}

impl SiegelReduction {
    /// The genus.
    pub fn genus(&self) -> usize {
        self.genus
    }

    /// The symplectic matrix `M` in `Sp_{2g}(Z)`, row-major, `2g x 2g`.
    pub fn symplectic(&self) -> &[rug::Integer] {
        &self.symplectic
    }

    /// Entry `(i, j)` of the symplectic matrix.
    pub fn symplectic_entry(&self, i: usize, j: usize) -> Option<&rug::Integer> {
        let n = 2 * self.genus;
        if i >= n || j >= n {
            return None;
        }
        Some(&self.symplectic[i * n + j])
    }

    /// `M . tau = (alpha tau + beta)(gamma tau + delta)^{-1}`, as enclosures.
    pub fn reduced(&self) -> &SiegelMatrix {
        &self.reduced
    }
}

/// Move `tau` towards the fundamental domain of `Sp_{2g}(Z)` acting on `H_g`.
///
/// Returns the symplectic matrix FLINT chose *and* the transformed period
/// matrix. FLINT falls back to the identity when the entries of `tau` are
/// unreasonable or `det Im(tau)` is vanishingly small, so an identity result is
/// a "no reduction found", not a claim that `tau` was already reduced — ask
/// [`siegel_is_reduced`] for that.
///
/// Theta values are *not* invariant under this action; they transform by the
/// theta transformation formula, with an eighth root of unity and a square root
/// of `det(gamma tau + delta)` in it. That formula is not wrapped, so the
/// reduced matrix here is for conditioning an evaluation, not for rewriting one
/// result into another.
pub fn siegel_reduce(tau: &SiegelMatrix, prec: u32) -> Result<SiegelReduction, ThetaError> {
    super::arith::check_precision(prec)?;
    let g = tau.genus();
    check_genus(g)?;
    let (symplectic, reduced) = backend::siegel_reduce(tau.entries(), g, prec)?;
    Ok(SiegelReduction {
        genus: g,
        symplectic,
        reduced: SiegelMatrix {
            genus: g,
            entries: reduced,
        },
    })
}

/// Is `tau` **certainly** in the reduced domain with tolerance `2^tol_exp`?
///
/// `tol_exp` is negative in practice (`-10` is a reasonable default). As
/// everywhere else here, `false` means "not proved".
pub fn siegel_is_reduced(tau: &SiegelMatrix, tol_exp: i64, prec: u32) -> Result<bool, ThetaError> {
    super::arith::check_precision(prec)?;
    check_genus(tau.genus())?;
    backend::siegel_is_reduced(tau.entries(), tau.genus(), tol_exp, prec)
}
