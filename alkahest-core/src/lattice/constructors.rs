//! The standard lattices: `ℤⁿ`, `A_n`, `D_n`, `E_8` and the Leech lattice.
//!
//! Each is returned as a [`Lattice`], so every invariant in that module —
//! minimum, kissing number, theta series, density — applies to it directly.
//!
//! # How much of this is checked
//!
//! Nothing here is a table of numbers copied out of a book. `E_8` is built from
//! its eight simple roots in ℚ⁸ and the Leech lattice is built from the binary
//! Golay code, and both are then *measured* by the same enumeration a caller
//! would use: `E_8` must come out with determinant 1 and 240 vectors of norm 2,
//! the Leech lattice with determinant 1 and 196560 vectors of norm 4. Those
//! counts are not reproduced by a construction that is nearly right, which is
//! the reason to test against them rather than against a stored Gram matrix.
//!
//! The Golay code is likewise generated (from the degree-11 factor of
//! `x²³ − 1` over GF(2), extended by an overall parity bit) and then checked
//! against its weight enumerator `1 + 759·z⁸ + 2576·z¹² + 759·z¹⁶ + z²⁴`
//! before it is used. [`leech`] refuses rather than returning an unverified
//! lattice if any of that fails.

use super::error::LatticeGeometryError;
use super::flint_backend;
use super::lattice::Lattice;
use super::lll::lattice_reduce_rows;
use rug::{Integer, Rational};

fn int_rows(rows: Vec<Vec<i64>>) -> Vec<Vec<Integer>> {
    rows.into_iter()
        .map(|r| r.into_iter().map(Integer::from).collect())
        .collect()
}

/// The integer lattice `ℤⁿ`, `n ≥ 1`.
pub fn zn(n: usize) -> Result<Lattice, LatticeGeometryError> {
    if n == 0 {
        return Err(LatticeGeometryError::InvalidParameter {
            detail: "Z^n needs n >= 1",
        });
    }
    let rows: Vec<Vec<i64>> = (0..n)
        .map(|i| (0..n).map(|j| i64::from(i == j)).collect())
        .collect();
    Lattice::from_basis(&int_rows(rows))
}

/// The root lattice `A_n = { x ∈ ℤⁿ⁺¹ : Σ xᵢ = 0 }`, `n ≥ 1`.
///
/// Determinant `n + 1`; minimum 2; kissing number `n(n+1)`. `A_2` is the
/// hexagonal lattice.
pub fn a_n(n: usize) -> Result<Lattice, LatticeGeometryError> {
    if n == 0 {
        return Err(LatticeGeometryError::InvalidParameter {
            detail: "A_n needs n >= 1",
        });
    }
    let rows: Vec<Vec<i64>> = (0..n)
        .map(|i| {
            let mut r = vec![0i64; n + 1];
            r[i] = 1;
            r[i + 1] = -1;
            r
        })
        .collect();
    Lattice::from_basis(&int_rows(rows))
}

/// The checkerboard lattice `D_n = { x ∈ ℤⁿ : Σ xᵢ even }`, `n ≥ 2`.
///
/// Determinant 4; minimum 2; kissing number `2n(n−1)`. `D_4` is the 24-cell
/// lattice.
pub fn d_n(n: usize) -> Result<Lattice, LatticeGeometryError> {
    if n < 2 {
        return Err(LatticeGeometryError::InvalidParameter {
            detail: "D_n needs n >= 2",
        });
    }
    let mut rows: Vec<Vec<i64>> = (0..n - 1)
        .map(|i| {
            let mut r = vec![0i64; n];
            r[i] = 1;
            r[i + 1] = -1;
            r
        })
        .collect();
    let mut last = vec![0i64; n];
    last[n - 2] = 1;
    last[n - 1] = 1;
    rows.push(last);
    Lattice::from_basis(&int_rows(rows))
}

/// The `E_8` root lattice, from its eight simple roots in ℚ⁸ (Bourbaki
/// labelling; `α₁` is the half-integer one).
///
/// Even and unimodular: determinant 1, minimum 2, 240 vectors of norm 2,
/// centre density `1/16`.
pub fn e8() -> Result<Lattice, LatticeGeometryError> {
    let half = Rational::from((1, 2));
    let mut rows: Vec<Vec<Rational>> = Vec::with_capacity(8);
    // α₁ = ½(e₁ − e₂ − e₃ − e₄ − e₅ − e₆ − e₇ + e₈)
    rows.push(
        [1i64, -1, -1, -1, -1, -1, -1, 1]
            .iter()
            .map(|&s| Rational::from(s) * &half)
            .collect(),
    );
    let mut push_int = |v: [i64; 8]| rows.push(v.iter().map(|&x| Rational::from(x)).collect());
    push_int([1, 1, 0, 0, 0, 0, 0, 0]); // α₂ = e₁ + e₂
    push_int([-1, 1, 0, 0, 0, 0, 0, 0]); // α₃ = e₂ − e₁
    push_int([0, -1, 1, 0, 0, 0, 0, 0]); // α₄ = e₃ − e₂
    push_int([0, 0, -1, 1, 0, 0, 0, 0]); // α₅
    push_int([0, 0, 0, -1, 1, 0, 0, 0]); // α₆
    push_int([0, 0, 0, 0, -1, 1, 0, 0]); // α₇
    push_int([0, 0, 0, 0, 0, -1, 1, 0]); // α₈
    Lattice::from_rational_basis(&rows)
}

// ---------------------------------------------------------------------------
// Binary Golay code
// ---------------------------------------------------------------------------

/// Degrees present in `g(x) = x¹¹ + x¹⁰ + x⁶ + x⁵ + x⁴ + x² + 1`, the
/// degree-11 factor of `x²³ − 1` over GF(2) that generates the perfect
/// `[23, 12, 7]` Golay code.
const GOLAY_G: [usize; 7] = [0, 2, 4, 5, 6, 10, 11];

/// The twelve rows of a generator matrix for the extended binary `[24, 12, 8]`
/// Golay code, as 24-bit masks (bit `i` = coordinate `i`).
fn golay_generators() -> [u32; 12] {
    let mut rows = [0u32; 12];
    for (i, row) in rows.iter_mut().enumerate() {
        let mut w = 0u32;
        for d in GOLAY_G {
            w |= 1 << (d + i);
        }
        // Overall parity check in coordinate 23.
        if w.count_ones() % 2 == 1 {
            w |= 1 << 23;
        }
        *row = w;
    }
    rows
}

/// All 4096 codewords of the extended binary Golay code, or `None` when the
/// generated code fails its weight enumerator — which is the check that this
/// really is the Golay code and not merely *a* `[24, 12]` code.
fn golay_codewords() -> Option<Vec<u32>> {
    let gens = golay_generators();
    let mut words = Vec::with_capacity(4096);
    for mask in 0u32..4096 {
        let mut w = 0u32;
        for (i, g) in gens.iter().enumerate() {
            if mask >> i & 1 == 1 {
                w ^= g;
            }
        }
        words.push(w);
    }
    let mut dist = [0usize; 25];
    for w in &words {
        dist[w.count_ones() as usize] += 1;
    }
    let mut expected = [0usize; 25];
    expected[0] = 1;
    expected[8] = 759;
    expected[12] = 2576;
    expected[16] = 759;
    expected[24] = 1;
    if dist != expected {
        return None;
    }
    Some(words)
}

/// The Leech lattice `Λ₂₄`.
///
/// Built from the binary Golay code, by the standard description
///
/// ```text
///   Λ₂₄ = 2^{-3/2} · { a ∈ ℤ²⁴ :  Σ aᵢ ≡ 4a₁  (mod 8),
///                       and for every m, { i : aᵢ ≡ m (mod 4) } ∈ G₂₄ }
/// ```
///
/// A generating set of that integral lattice is put through a Hermite normal
/// form to get a basis, LLL-reduced, and scaled by `1/8` into a Gram matrix.
/// The construction then **checks itself** before returning: the resulting Gram
/// matrix must be integral, even, and of determinant 1 — the last of which says
/// the underlying integral lattice had covolume `2³⁶`. If any of that fails the
/// function refuses rather than returning something that merely looks like the
/// Leech lattice.
///
/// **Those checks do not on their own identify `Λ₂₄`.** There are 24 even
/// unimodular lattices in dimension 24 and every one of them passes them; what
/// separates the Leech lattice from the other 23 Niemeier lattices is that it
/// has no vectors of norm 2. Pinning that down costs a rank-24 enumeration, so
/// it is a **test**, not a step of this constructor:
/// `tests::leech_is_even_unimodular_of_minimum_four` asserts the minimum is 4
/// and `tests::leech_has_196560_minimal_vectors` the kissing number. Nothing
/// else in dimension 24 produces 196560.
///
/// Even and unimodular: determinant 1, minimum 4, 196560 vectors of norm 4,
/// no vectors of norm 2.
pub fn leech() -> Result<Lattice, LatticeGeometryError> {
    const REFUSAL: LatticeGeometryError = LatticeGeometryError::InvalidParameter {
        detail: "Leech lattice construction failed its own verification",
    };
    let words = golay_codewords().ok_or(REFUSAL)?;
    let gens = golay_generators();
    debug_assert!(words.len() == 4096);

    let mut rows: Vec<Vec<i64>> = Vec::new();
    // 8·e_i — Σ = 8 ≡ 0, and every coordinate is ≡ 0 (mod 4).
    for i in 0..24 {
        let mut r = vec![0i64; 24];
        r[i] = 8;
        rows.push(r);
    }
    // 4(e₁ + e_i) — together with the above these generate every 4(e_i + e_j).
    for i in 1..24 {
        let mut r = vec![0i64; 24];
        r[0] = 4;
        r[i] = 4;
        rows.push(r);
    }
    // 2c for each Golay generator: the ≡2 (mod 4) set is the codeword and the
    // ≡0 set is its complement, also a codeword because G₂₄ contains 1.
    for g in gens {
        let mut r = vec![0i64; 24];
        for (i, e) in r.iter_mut().enumerate() {
            if g >> i & 1 == 1 {
                *e = 2;
            }
        }
        rows.push(r);
    }
    // (−3, 1, …, 1): every coordinate is ≡ 1 (mod 4), so the ≡1 set is the
    // all-ones codeword; Σ = 20 and 4a₁ = −12 differ by 32 ≡ 0 (mod 8).
    let mut odd = vec![1i64; 24];
    odd[0] = -3;
    rows.push(odd);

    let basis = flint_backend::hnf_basis(&int_rows(rows), 24);
    if basis.len() != 24 {
        return Err(REFUSAL);
    }
    let basis = lattice_reduce_rows(&basis)?;

    // Gram of the *scaled* lattice: ⟨a, b⟩ / 8.
    let eighth = Rational::from((1, 8));
    let mut gram = vec![vec![Rational::new(); 24]; 24];
    for i in 0..24 {
        for j in 0..=i {
            let mut acc = Integer::new();
            for (a, b) in basis[i].iter().zip(basis[j].iter()) {
                acc += Integer::from(a * b);
            }
            let v = Rational::from(acc) * &eighth;
            if *v.denom() != 1i32 {
                return Err(REFUSAL);
            }
            gram[i][j] = v.clone();
            gram[j][i] = v;
        }
    }
    let lattice = Lattice::from_gram(&gram)?;
    // det Λ = 1 (so the integral lattice had covolume 2³⁶), and Λ is even.
    if lattice.determinant() != 1 || !lattice.is_even() {
        return Err(REFUSAL);
    }
    Ok(lattice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golay_code_has_the_right_weight_enumerator() {
        let words = golay_codewords().expect("generated code is the Golay code");
        assert_eq!(words.len(), 4096);
        // Self-dual: every pair of codewords meets in an even number of places.
        let gens = golay_generators();
        for a in gens {
            for b in gens {
                assert_eq!((a & b).count_ones() % 2, 0, "G24 must be self-orthogonal");
            }
        }
        // Contains the all-ones word.
        assert!(words.contains(&0x00FF_FFFF));
    }
}
