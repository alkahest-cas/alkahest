//! FLINT's `fmpz_lll` behind a safe, allocation-checked Rust wrapper.
//!
//! This is the *engine* under [`super::lll`]; it is deliberately not public.
//! Everything it returns is re-checked against the exact rational LLL
//! predicates before a caller sees it — see [`super::lll`] for the two-stage
//! arrangement and why it exists.
//!
//! # FFI hazard: `fmpz_mat_struct` changed layout
//!
//! FLINT replaced the `fmpz ** rows` row-pointer array with a `slong stride`
//! in 3.1. Both fields are pointer-sized, so picking the wrong one is neither
//! a compile error nor a `size_of` mismatch — it is an integer dereferenced as
//! a pointer. Two defences, the same two `crate::ffield` uses:
//!
//! 1. `build.rs` selects the declaration from `flint/fmpz_types.h`
//!    (`flint3_stride`), never from a guess.
//! 2. **No pointer arithmetic on `entries` happens here.** Every entry read and
//!    write goes through FLINT's own `fmpz_mat_entry`, which computes the
//!    offset inside the installed library. Only `r` and `c` are read from the
//!    struct, and those sit at offsets 8 and 16 in *both* layouts.
//!
//! `tests::entry_round_trip_non_square` writes a distinct value into every cell
//! of several non-square shapes and reads them all back, which is the check a
//! wrong stride fails (a wrong stride frequently still passes on square or
//! single-column matrices).

use crate::flint::ffi;
use crate::flint::integer::FlintInteger;
use rug::Integer;

/// Owned `r × c` FLINT integer matrix.
struct FmpzMat {
    inner: ffi::FmpzMatStruct,
}

impl FmpzMat {
    fn new(rows: usize, cols: usize) -> Self {
        assert!(rows <= i64::MAX as usize && cols <= i64::MAX as usize);
        #[cfg(not(flint3_stride))]
        let mut inner = ffi::FmpzMatStruct {
            entries: std::ptr::null_mut(),
            r: 0,
            c: 0,
            rows: std::ptr::null_mut(),
        };
        #[cfg(flint3_stride)]
        let mut inner = ffi::FmpzMatStruct {
            entries: std::ptr::null_mut(),
            r: 0,
            c: 0,
            stride: 0,
        };
        unsafe { ffi::fmpz_mat_init(&mut inner, rows as ffi::slong, cols as ffi::slong) };
        Self { inner }
    }

    fn rows(&self) -> usize {
        self.inner.r as usize
    }

    fn cols(&self) -> usize {
        self.inner.c as usize
    }

    fn set(&mut self, i: usize, j: usize, v: &Integer) {
        assert!(i < self.rows() && j < self.cols());
        let fi = FlintInteger::from_rug(v);
        unsafe {
            let e = ffi::fmpz_mat_entry(&self.inner, i as ffi::slong, j as ffi::slong);
            ffi::fmpz_set(e, fi.inner_ptr());
        }
    }

    fn get(&self, i: usize, j: usize) -> Integer {
        assert!(i < self.rows() && j < self.cols());
        let mut out = FlintInteger::new();
        unsafe {
            let e = ffi::fmpz_mat_entry(&self.inner, i as ffi::slong, j as ffi::slong);
            ffi::fmpz_set(out.inner_mut_ptr(), e);
        }
        out.to_rug()
    }

    fn rank(&self) -> usize {
        unsafe { ffi::fmpz_mat_rank(&self.inner) as usize }
    }
}

impl Drop for FmpzMat {
    fn drop(&mut self) {
        unsafe { ffi::fmpz_mat_clear(&mut self.inner) };
    }
}

/// FLINT's Lovász parameter. Higher than the classical `3/4`: the exact polish
/// pass in [`super::lll`] only ever *weakens* a reduction when it re-reduces a
/// `|μ|` that FLINT left in `(1/2, η]`, so starting strictly stronger than the
/// caller asked for is what makes the polish converge in a sweep or two.
const FLINT_DELTA_FLOOR: f64 = 0.99;
/// Upper clamp — FLINT wants `δ < 1` strictly and `η < sqrt(δ)`.
const FLINT_DELTA_CEIL: f64 = 0.9999;
/// FLINT's size-reduction parameter. Must be `> 1/2` (floating-point LLL cannot
/// promise the exact `1/2`), which is precisely the gap the exact polish closes.
const FLINT_ETA: f64 = 0.51;

/// LLL-reduce the rows of `basis` with FLINT, targeting Lovász parameter
/// `delta_hint` (clamped into FLINT's comfortable range).
///
/// Returns `None` when the input is rank-deficient. FLINT's `fmpz_lll` is an
/// MLLL-style routine on such input and its row handling there is not something
/// this crate pins down by test, so the rank-deficient path stays on the exact
/// rational implementation, which has regression tests for it. The rank probe
/// is one `fmpz_mat_rank` — negligible next to a reduction.
pub(crate) fn flint_lll_rows(basis: &[Vec<Integer>], delta_hint: f64) -> Option<Vec<Vec<Integer>>> {
    let m = basis.len();
    if m == 0 {
        return None;
    }
    let n = basis[0].len();
    if n == 0 {
        return None;
    }

    let mut mat = FmpzMat::new(m, n);
    for (i, row) in basis.iter().enumerate() {
        debug_assert_eq!(row.len(), n);
        for (j, v) in row.iter().enumerate() {
            mat.set(i, j, v);
        }
    }

    // A rank-deficient basis has a zero Gram–Schmidt norm, which is where a
    // floating-point LLL divides by zero. Hand those to the exact path.
    if mat.rank() != m {
        return None;
    }

    let delta = delta_hint.clamp(FLINT_DELTA_FLOOR, FLINT_DELTA_CEIL);
    let fl = ffi::FmpzLllStruct {
        delta,
        eta: FLINT_ETA,
        rt: ffi::FMPZ_LLL_Z_BASIS,
        gt: 0,
    };
    unsafe {
        // `u = NULL`: we do not need the transform, only the reduced rows.
        ffi::fmpz_lll(&mut mat.inner, std::ptr::null_mut(), &fl);
    }

    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            row.push(mat.get(i, j));
        }
        out.push(row);
    }
    Some(out)
}

/// Hermite normal form of a row-generating set, with the zero rows dropped.
///
/// Used to turn a redundant list of lattice generators into an honest basis.
/// FLINT's `fmpz_mat_hnf` leaves the non-zero rows at the top.
pub(crate) fn hnf_basis(rows: &[Vec<Integer>], cols: usize) -> Vec<Vec<Integer>> {
    let m = rows.len();
    if m == 0 || cols == 0 {
        return Vec::new();
    }
    let mut a = FmpzMat::new(m, cols);
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            a.set(i, j, v);
        }
    }
    let mut h = FmpzMat::new(m, cols);
    unsafe { ffi::fmpz_mat_hnf(&mut h.inner, &a.inner) };
    let mut out = Vec::new();
    for i in 0..m {
        let row: Vec<Integer> = (0..cols).map(|j| h.get(i, j)).collect();
        if row.iter().any(|v| *v != 0) {
            out.push(row);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stride check. A wrong `fmpz_mat_struct` layout is silent memory
    /// corruption, not a compile error, and it frequently survives square
    /// shapes; every entry of several rectangular shapes is written and read
    /// back here, in both orientations.
    #[test]
    fn entry_round_trip_non_square() {
        for &(r, c) in &[
            (1usize, 7usize),
            (7, 1),
            (2, 5),
            (5, 2),
            (3, 3),
            (4, 9),
            (9, 4),
        ] {
            let mut m = FmpzMat::new(r, c);
            assert_eq!(m.rows(), r);
            assert_eq!(m.cols(), c);
            for i in 0..r {
                for j in 0..c {
                    // Distinct per cell, and large enough to be a heap `fmpz`
                    // for some entries so both fmpz representations are covered.
                    let v = Integer::from(1_000_000_007u64) * (i as u64 * c as u64 + j as u64 + 1)
                        + Integer::from(i as i64 - j as i64);
                    m.set(i, j, &v);
                }
            }
            for i in 0..r {
                for j in 0..c {
                    let expect = Integer::from(1_000_000_007u64)
                        * (i as u64 * c as u64 + j as u64 + 1)
                        + Integer::from(i as i64 - j as i64);
                    assert_eq!(m.get(i, j), expect, "entry ({i},{j}) of a {r}x{c} matrix");
                }
            }
        }
    }

    #[test]
    fn rank_probe_matches_expectation() {
        let full = FmpzMat::new(2, 2);
        assert_eq!(full.rank(), 0, "a freshly initialised matrix is zero");

        let mut m = FmpzMat::new(3, 3);
        for i in 0..3 {
            m.set(i, i, &Integer::from(1));
        }
        assert_eq!(m.rank(), 3);
        m.set(2, 2, &Integer::from(0));
        assert_eq!(m.rank(), 2);
    }

    #[test]
    fn flint_reduces_a_known_bad_basis() {
        let rows = vec![
            vec![Integer::from(1), Integer::from(0), Integer::from(0)],
            vec![Integer::from(0), Integer::from(1), Integer::from(0)],
            vec![Integer::from(97), Integer::from(89), Integer::from(1)],
        ];
        let out = flint_lll_rows(&rows, 0.75).expect("full rank");
        assert_eq!(out.len(), 3);
        let norm = |r: &Vec<Integer>| {
            r.iter()
                .fold(Integer::from(0), |a, x| a + Integer::from(x * x))
        };
        assert!(out.iter().all(|r| norm(r) <= 4));
    }

    #[test]
    fn rank_deficient_input_is_declined() {
        let rows = vec![
            vec![Integer::from(1), Integer::from(2)],
            vec![Integer::from(2), Integer::from(4)],
        ];
        assert!(flint_lll_rows(&rows, 0.75).is_none());
    }
}
