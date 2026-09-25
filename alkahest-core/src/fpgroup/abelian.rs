//! Finitely generated abelian groups as an answer type, and the one lattice
//! quotient every question in this module reduces to.
//!
//! Both the abelianisation `G/[G, G]` of a presentation and the cohomology
//! groups `H^d(G, M)` are computed as `K/S` for two lattices `S ⊆ K ⊆ ℤ^N`. The
//! machinery is entirely borrowed: [`hermite_form`] (FLINT's
//! `fmpz_mat_hnf_transform`) puts `K` in echelon form and supplies integer
//! kernels, and [`smith_invariants`] turns the coordinate matrix of `S` in that
//! basis into invariant factors. There is no second normal-form implementation
//! here.

// Index arithmetic over parallel arrays is the subject here: a row index, a
// column index and a modulus index all range over the same bounds and are used
// together, so `needless_range_loop`'s enumerate() rewrites would name one of
// the three and index the others anyway. `matrix::smith` allows it for the same
// reason.
#![allow(clippy::needless_range_loop)]

use super::error::FpGroupError;
use crate::matrix::normal_form::{hermite_basis, hermite_form, smith_invariants, IntegerMatrix};
use rug::Integer;
use std::fmt;

/// A finitely generated abelian group, as a free rank plus invariant factors.
///
/// `torsion` is ascending under divisibility, exactly as Smith normal form
/// leaves it, and never contains `1`: the group is
/// `ℤ^free_rank ⊕ ℤ/torsion[0] ⊕ … ⊕ ℤ/torsion[k-1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbelianInvariants {
    free_rank: usize,
    torsion: Vec<Integer>,
}

impl AbelianInvariants {
    /// The trivial group.
    pub fn trivial() -> AbelianInvariants {
        AbelianInvariants {
            free_rank: 0,
            torsion: Vec::new(),
        }
    }

    /// Build directly. Invariant factors equal to `0` or `±1` are dropped —
    /// `0` is not a torsion factor and `ℤ/1` is trivial — and the rest are
    /// taken in absolute value.
    pub fn new(free_rank: usize, torsion: Vec<Integer>) -> AbelianInvariants {
        let torsion = torsion
            .into_iter()
            .map(|d| d.abs())
            .filter(|d| *d > 1)
            .collect();
        AbelianInvariants { free_rank, torsion }
    }

    /// The rank of the free part: the number of `ℤ` summands.
    pub fn free_rank(&self) -> usize {
        self.free_rank
    }

    /// The invariant factors, ascending under divisibility, each at least `2`.
    pub fn torsion(&self) -> &[Integer] {
        &self.torsion
    }

    /// Is this the trivial group?
    pub fn is_trivial(&self) -> bool {
        self.free_rank == 0 && self.torsion.is_empty()
    }

    /// Is the group finite?
    pub fn is_finite(&self) -> bool {
        self.free_rank == 0
    }

    /// The order, or `None` when the group is infinite.
    ///
    /// `None` means *infinite*, proved — not "could not tell".
    pub fn order(&self) -> Option<Integer> {
        if self.free_rank > 0 {
            return None;
        }
        let mut n = Integer::from(1);
        for d in &self.torsion {
            n *= d;
        }
        Some(n)
    }
}

impl fmt::Display for AbelianInvariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_trivial() {
            return write!(f, "0");
        }
        let mut parts: Vec<String> = Vec::new();
        match self.free_rank {
            0 => {}
            1 => parts.push("Z".to_string()),
            r => parts.push(format!("Z^{r}")),
        }
        for d in &self.torsion {
            parts.push(format!("Z/{d}"));
        }
        write!(f, "{}", parts.join(" + "))
    }
}

// ---------------------------------------------------------------------------
// Lattice helpers
// ---------------------------------------------------------------------------

fn matrix_from_rows(rows: &[Vec<Integer>], cols: usize) -> Result<IntegerMatrix, FpGroupError> {
    if rows.is_empty() {
        return IntegerMatrix::from_rug_rows(Vec::new()).map_err(FpGroupError::from);
    }
    for (i, r) in rows.iter().enumerate() {
        if r.len() != cols {
            return Err(FpGroupError::Internal {
                detail: format!("row {i} has {} entries, expected {cols}", r.len()),
            });
        }
    }
    IntegerMatrix::from_rug_rows(rows.to_vec()).map_err(FpGroupError::from)
}

/// An echelon basis of the lattice spanned by `rows`, with its pivot columns.
///
/// FLINT's Hermite normal form is row echelon with strictly increasing pivots
/// and the zero rows last, which is exactly what the back-substitution in
/// [`coordinates_in`] needs.
fn echelon_basis(
    rows: &[Vec<Integer>],
    cols: usize,
) -> Result<(Vec<Vec<Integer>>, Vec<usize>), FpGroupError> {
    if rows.is_empty() || cols == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let m = matrix_from_rows(rows, cols)?;
    // `hermite_basis`, not `hermite_form`: the transform is not wanted here, and
    // computing it is the expensive half on a generating set of this shape.
    let h = hermite_basis(&m);
    let mut basis: Vec<Vec<Integer>> = Vec::new();
    let mut pivots: Vec<usize> = Vec::new();
    for i in 0..h.rows {
        let row: Vec<Integer> = (0..h.cols).map(|j| h.get(i, j).clone()).collect();
        if let Some(p) = row.iter().position(|e| *e != 0) {
            if pivots.last().is_some_and(|&q| q >= p) {
                return Err(FpGroupError::Internal {
                    detail: "Hermite normal form is not in increasing-pivot echelon form".into(),
                });
            }
            pivots.push(p);
            basis.push(row);
        }
    }
    Ok((basis, pivots))
}

/// A basis for the integer left kernel `{x : x·M = 0}` of `rows`.
///
/// From `U·M = H`, the rows of `U` opposite the zero rows of `H` are a basis of
/// the left kernel: `U` is unimodular, so its rows are a basis of `ℤ^m`, and a
/// row maps to zero exactly when the corresponding row of `H` is zero.
fn left_kernel(rows: &[Vec<Integer>], cols: usize) -> Result<Vec<Vec<Integer>>, FpGroupError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    if cols == 0 {
        // Everything is in the kernel.
        return Ok((0..rows.len())
            .map(|i| {
                (0..rows.len())
                    .map(|j| Integer::from(usize::from(i == j)))
                    .collect()
            })
            .collect());
    }
    let m = matrix_from_rows(rows, cols)?;
    let (h, u) = hermite_form(&m);
    let mut kernel: Vec<Vec<Integer>> = Vec::new();
    for i in 0..h.rows {
        if (0..h.cols).all(|j| *h.get(i, j) == 0) {
            kernel.push((0..u.cols).map(|j| u.get(i, j).clone()).collect());
        }
    }
    Ok(kernel)
}

/// Coordinates of `v` in the echelon basis `basis`, or `None` if `v` is not in
/// the lattice.
fn coordinates_in(basis: &[Vec<Integer>], pivots: &[usize], v: &[Integer]) -> Option<Vec<Integer>> {
    let mut residue: Vec<Integer> = v.to_vec();
    let mut coords = vec![Integer::from(0); basis.len()];
    for (i, &p) in pivots.iter().enumerate() {
        if residue[p] == 0 {
            continue;
        }
        let piv = &basis[i][p];
        let (q, r) = residue[p].clone().div_rem(piv.clone());
        if r != 0 {
            return None;
        }
        if q != 0 {
            for (c, entry) in basis[i].iter().enumerate().skip(p) {
                let delta = Integer::from(&q * entry);
                residue[c] -= delta;
            }
        }
        coords[i] = q;
    }
    if residue.iter().any(|e| *e != 0) {
        return None;
    }
    Some(coords)
}

/// The invariants of `K/S`, for lattices `S ⊆ K ⊆ ℤ^dim` given by generating
/// sets.
///
/// Every generator of `S` is checked to lie in `K` on the way through — that is
/// the natural place to catch a wrong coboundary matrix, since `S ⊆ K` is
/// exactly the statement `d ∘ d = 0`, and a violation raises
/// [`FpGroupError::Internal`] rather than producing a quotient of the wrong
/// thing.
pub(super) fn quotient_invariants(
    dim: usize,
    k_gens: &[Vec<Integer>],
    s_gens: &[Vec<Integer>],
) -> Result<AbelianInvariants, FpGroupError> {
    if dim == 0 {
        return Ok(AbelianInvariants::trivial());
    }
    let (basis, pivots) = echelon_basis(k_gens, dim)?;
    let r = basis.len();
    if r == 0 {
        for s in s_gens {
            if s.iter().any(|e| *e != 0) {
                return Err(FpGroupError::Internal {
                    detail: "a subgroup generator lies outside the kernel lattice".into(),
                });
            }
        }
        return Ok(AbelianInvariants::trivial());
    }
    let mut coord_rows: Vec<Vec<Integer>> = Vec::with_capacity(s_gens.len());
    for s in s_gens {
        if s.len() != dim {
            return Err(FpGroupError::Internal {
                detail: format!("subgroup generator has {} entries, expected {dim}", s.len()),
            });
        }
        if s.iter().all(|e| *e == 0) {
            continue;
        }
        match coordinates_in(&basis, &pivots, s) {
            Some(c) => coord_rows.push(c),
            None => {
                return Err(FpGroupError::Internal {
                    detail: "a subgroup generator lies outside the kernel lattice, so the \
                             complex does not satisfy d o d = 0"
                        .into(),
                })
            }
        }
    }
    if coord_rows.is_empty() {
        return Ok(AbelianInvariants::new(r, Vec::new()));
    }
    let c = matrix_from_rows(&coord_rows, r)?;
    // `smith_invariants`, not `smith_form`: only the diagonal is wanted, and the
    // transforms would be built and thrown away.
    let mut torsion: Vec<Integer> = Vec::new();
    let mut rank = 0usize;
    for d in smith_invariants(&c) {
        let d = d.abs();
        if d != 0 {
            rank += 1;
            if d > 1 {
                torsion.push(d);
            }
        }
    }
    if rank > r {
        return Err(FpGroupError::Internal {
            detail: "coordinate matrix has rank above the lattice rank".into(),
        });
    }
    Ok(AbelianInvariants::new(r - rank, torsion))
}

/// `K = {x ∈ ℤ^dim : x·M ∈ L}` where `L` is the **diagonal** lattice spanned by
/// `moduli[j]·e_j` for the `j` with `moduli[j] > 0`, and a `moduli[j] == 0`
/// coordinate must vanish outright.
///
/// Computed as the projection of the left kernel of `M` stacked on `-diag(L)`:
/// `x·M = t·diag(L)` has a solution `t` exactly when `x·M ∈ L`.
pub(super) fn preimage_lattice(
    m_rows: &[Vec<Integer>],
    dim: usize,
    cols: usize,
    moduli: &[Integer],
) -> Result<Vec<Vec<Integer>>, FpGroupError> {
    if m_rows.len() != dim {
        return Err(FpGroupError::Internal {
            detail: format!("expected {dim} rows, got {}", m_rows.len()),
        });
    }
    if moduli.len() != cols {
        return Err(FpGroupError::Internal {
            detail: format!("expected {cols} moduli, got {}", moduli.len()),
        });
    }
    let mut stacked: Vec<Vec<Integer>> = m_rows.to_vec();
    for (j, mj) in moduli.iter().enumerate() {
        if *mj > 0 {
            let mut row = vec![Integer::from(0); cols];
            row[j] = -mj.clone();
            stacked.push(row);
        }
    }
    let kernel = left_kernel(&stacked, cols)?;
    let mut out: Vec<Vec<Integer>> = Vec::with_capacity(kernel.len());
    for k in &kernel {
        if k.len() < dim {
            return Err(FpGroupError::Internal {
                detail: "kernel row is shorter than the domain".into(),
            });
        }
        let projected: Vec<Integer> = k[..dim].to_vec();
        if projected.iter().any(|e| *e != 0) {
            out.push(projected);
        }
    }
    Ok(out)
}

/// How many columns of the coboundary are intersected at a time by
/// [`preimage_lattice_blocked`].
///
/// The unblocked computation stacks **one extra row per column**, so its Hermite
/// form is on a `(dim + cols) × cols` matrix — and that form has to produce the
/// unimodular transform, whose entries grow with the rank deficiency rather than
/// with the answer. Intersecting a block at a time keeps every Hermite form at
/// `(dim + 64) × 64`, and reducing the intermediate basis modulo the module's
/// exponent keeps its entries from growing across blocks.
const PREIMAGE_BLOCK: usize = 64;

fn unit_row(i: usize, dim: usize) -> Vec<Integer> {
    let mut row = vec![Integer::from(0); dim];
    row[i] = Integer::from(1);
    row
}

/// `K = {x ∈ ℤ^dim : x·M ∈ L}`, computed by intersecting the columns of `M` a
/// block at a time.
///
/// `{x : x·M ∈ L} = ⋂_blocks {x : x·M_block ∈ L_block}`, and each intersection is
/// the same [`preimage_lattice`] call on a narrow matrix, so this is the
/// unblocked computation rearranged rather than a second algorithm. A unit test
/// asserts the two agree.
///
/// Blocking is skipped when every modulus is zero: with no relation lattice
/// nothing is stacked, the unblocked Hermite form is already on a `dim × cols`
/// matrix, and splitting it would only cost entry growth in the intermediate
/// bases.
pub(super) fn preimage_lattice_blocked(
    m_rows: &[Vec<Integer>],
    dim: usize,
    cols: usize,
    moduli: &[Integer],
) -> Result<Vec<Vec<Integer>>, FpGroupError> {
    if cols <= PREIMAGE_BLOCK || moduli.iter().all(|m| *m == 0) {
        return preimage_lattice(m_rows, dim, cols, moduli);
    }
    if m_rows.len() != dim || moduli.len() != cols {
        return Err(FpGroupError::Internal {
            detail: format!(
                "blocked preimage got {} rows and {} moduli for dim {dim} and {cols} columns",
                m_rows.len(),
                moduli.len()
            ),
        });
    }
    // The exponent of the relation lattice, when there is one: every modulus
    // divides it, so `L·ℤ^dim` sits inside every lattice this loop produces. A
    // single zero modulus is an *exact* condition rather than a congruence and
    // breaks that, so the reduction below is switched off for a mixed module.
    let lcm_modulus = if moduli.iter().all(|m| *m > 0) {
        moduli.iter().fold(Integer::from(1), |acc, m| acc.lcm(m))
    } else {
        Integer::from(0)
    };
    // Invariant: the lattice computed so far is `⟨basis⟩ + L·ℤ^dim`, with the
    // second summand carried *implicitly* while the loop runs. It is legitimate
    // to drop it from the generators because every element of `L·ℤ^dim` satisfies
    // every remaining congruence (`m_j | L`), so intersecting the next block
    // commutes with adding it back.
    let mut basis: Vec<Vec<Integer>> = (0..dim).map(|i| unit_row(i, dim)).collect();
    let mut start = 0;
    while start < cols && !basis.is_empty() {
        let end = (start + PREIMAGE_BLOCK).min(cols);
        let width = end - start;
        // The block's columns, read in the coordinates of the current basis.
        let mut block_rows: Vec<Vec<Integer>> = Vec::with_capacity(basis.len());
        for b in &basis {
            let mut row = Vec::with_capacity(width);
            for j in start..end {
                let mut acc = Integer::from(0);
                for (i, bi) in b.iter().enumerate() {
                    if *bi != 0 && m_rows[i][j] != 0 {
                        acc += Integer::from(bi * &m_rows[i][j]);
                    }
                }
                row.push(acc);
            }
            block_rows.push(row);
        }
        let gens = preimage_lattice(&block_rows, basis.len(), width, &moduli[start..end])?;
        // Back to ℤ^dim, then re-normalise so the entries stay bounded.
        let mut mapped: Vec<Vec<Integer>> = Vec::with_capacity(gens.len());
        for g in &gens {
            let mut row = vec![Integer::from(0); dim];
            for (i, gi) in g.iter().enumerate() {
                if *gi == 0 {
                    continue;
                }
                for c in 0..dim {
                    if basis[i][c] != 0 {
                        let term = Integer::from(gi * &basis[i][c]);
                        row[c] += term;
                    }
                }
            }
            mapped.push(row);
        }
        // Size reduction, and the reason this stays linear in the number of
        // blocks. The generators coming back inherit the entries of a Hermite
        // *transform* matrix, which grow block over block until one later Hermite
        // form is the entire cost: measured on `H²(ℤ/8, ℤ/2)`, block 2 of 8 took
        // 12.7 s of a 16 s run while every other block took 70 ms. Taking the
        // entries modulo `L` costs nothing and bounds them, because the `L·ℤ^dim`
        // that the convention above carries separately is exactly what such a
        // reduction discards.
        if lcm_modulus > 0 {
            for row in &mut mapped {
                for e in row.iter_mut() {
                    *e = e.clone().modulo(&lcm_modulus);
                }
            }
        }
        let (next, _pivots) = echelon_basis(&mapped, dim)?;
        basis = next;
        start = end;
    }
    if lcm_modulus > 0 {
        // Make the carried `L·ℤ^dim` explicit again now that the answer is
        // leaving.
        for i in 0..dim {
            let mut row = vec![Integer::from(0); dim];
            row[i] = lcm_modulus.clone();
            basis.push(row);
        }
    }
    Ok(basis)
}
