//! Trager's ℚ-basis criterion for the logarithmic part (Risch **MC**, the
//! algebraic-residue case).
//!
//! After Hermite reduction the third-kind part `h dx` has simple poles with
//! residues `res_P` (possibly **algebraic**).  By Liouville `∫ h dx` is
//! elementary iff it is `Σ_j c_j log(u_j)`; Trager's criterion turns this into a
//! finite set of **torsion** tests:
//!
//! 1. The residues span a finite-dimensional `ℚ`-vector space `V` (here:
//!    elements of a number field `ℚ(θ)`).  Pick a `ℚ`-basis `ω_1,…,ω_r` of the
//!    span and write `res_P = Σ_k λ_{P,k} ω_k` with `λ_{P,k} ∈ ℚ`.
//! 2. For each component `k`, form the **rational-coefficient** divisor
//!    `δ_k = Σ_P λ_{P,k}·P`.
//! 3. `∫ h dx` is elementary **iff every `δ_k` is torsion** (a principal
//!    multiple) in the Jacobian — decided per component by [`super::find_order`].
//!
//! This module implements the `ℚ`-basis decomposition ([`qbasis_decompose`]) and
//! the per-component dispatch ([`trager_log_criterion`]) for residues given as
//! elements of a **common** `ℚ(θ)` at **rational** places.  A non-torsion
//! component certifies `NonElementary` (sound); all-torsion means the log part
//! exists (`Principal`).  Getting the residues from several distinct places into
//! a common `ℚ(θ)` (a compositum) and torsion of **non-branch algebraic** places
//! are the remaining glue toward the fully general criterion.

use rug::{Integer, Rational};

use super::super::risch::number_field::KElem;
use super::super::risch::poly_rde::QPoly;
use super::find_order::{find_order_placed, FindOrder};
use super::jacobian_torsion::{find_order_genus_ge2_alg, AlgPlace};
use super::residues::{PlacedResidue, Residue};

/// `ℚ`-basis decomposition of residues `r_i ∈ ℚ(θ)` (each a `KElem`, i.e. a
/// `QPoly` of degree `< dim`).  Returns `(r, coords)` where `r` is the dimension
/// of the `ℚ`-span and `coords[i]` is the length-`r` coordinate vector of `r_i`
/// in the chosen basis.  The basis is the set of pivot columns of the
/// row-reduced residue matrix, so `coords[i][k] = r_i[pivot_k]` exactly.
pub(crate) fn qbasis_decompose(residues: &[KElem], dim: usize) -> (usize, Vec<Vec<Rational>>) {
    if dim == 0 || residues.is_empty() {
        return (0, residues.iter().map(|_| Vec::new()).collect());
    }
    // Residues as ℚ-vectors of length `dim` (pad with zeros).
    let vecs: Vec<Vec<Rational>> = residues
        .iter()
        .map(|r| {
            (0..dim)
                .map(|j| r.get(j).cloned().unwrap_or_else(|| Rational::from(0)))
                .collect()
        })
        .collect();

    // Row-reduce a copy to find the pivot columns (a basis of the row span).
    let mut m = vecs.clone();
    let nrows = m.len();
    let mut pivots: Vec<usize> = Vec::new();
    let mut row = 0usize;
    for col in 0..dim {
        if row >= nrows {
            break;
        }
        let Some(sel) = (row..nrows).find(|&r| m[r][col] != 0) else {
            continue;
        };
        m.swap(row, sel);
        let piv = m[row][col].clone();
        for v in m[row].iter_mut() {
            *v /= &piv;
        }
        let pr = m[row].clone();
        for r in 0..nrows {
            if r != row && m[r][col] != 0 {
                let f = m[r][col].clone();
                for (dst, pv) in m[r].iter_mut().zip(pr.iter()) {
                    *dst -= f.clone() * pv;
                }
            }
        }
        pivots.push(col);
        row += 1;
    }

    // With the basis = pivot columns of the RREF, the coordinate of `r_i` along
    // basis vector `k` is exactly `r_i[pivot_k]`.
    let coords = vecs
        .iter()
        .map(|v| pivots.iter().map(|&p| v[p].clone()).collect())
        .collect();
    (pivots.len(), coords)
}

/// Trager's criterion for residues in a **common** `ℚ(θ)` at **rational** places.
///
/// `places` and `residues` are parallel (same length); each residue is a `KElem`
/// in `ℚ(θ)` (`dim = deg θ`).  Decomposes the residues over `ℚ`, and for each
/// component `k` runs [`find_order_placed`] on the rational-coefficient divisor
/// `δ_k`.  Returns:
/// * [`FindOrder::NonElementary`] if **any** component is non-torsion (sound);
/// * [`FindOrder::Principal`] with the lcm of the component orders if **all** are
///   torsion (the log part exists);
/// * [`FindOrder::NotDecided`] if any component is undecided.
pub(crate) fn trager_log_criterion(
    n: usize,
    a: &QPoly,
    places: &[PlacedResidue],
    residues: &[KElem],
    dim: usize,
) -> FindOrder {
    if places.len() != residues.len() || places.is_empty() {
        return FindOrder::NotDecided;
    }
    let (ncomp, coords) = qbasis_decompose(residues, dim);
    if ncomp == 0 {
        // All residues zero ⇒ no logarithmic part.
        return FindOrder::Principal { order: 1 };
    }

    let mut lcm_order: u32 = 1;
    for k in 0..ncomp {
        // Component divisor δ_k: same places, residue value = coords[i][k].
        let div_k: Vec<PlacedResidue> = places
            .iter()
            .zip(&coords)
            .map(|(pl, c)| PlacedResidue {
                residue: Residue {
                    point: pl.residue.point.clone(),
                    at_infinity: pl.residue.at_infinity,
                    sheet: pl.residue.sheet,
                    ramification: pl.residue.ramification,
                    value: c[k].clone(),
                },
                y_coord: pl.y_coord.clone(),
            })
            .collect();
        match find_order_placed(n, a, &div_k) {
            FindOrder::NonElementary => return FindOrder::NonElementary,
            FindOrder::Principal { order } => lcm_order = lcm_u32(lcm_order, order),
            FindOrder::NotDecided => return FindOrder::NotDecided,
        }
    }
    FindOrder::Principal { order: lcm_order }
}

/// Trager's criterion with **algebraic** places: residues live in a common
/// `ℚ(θ)` (`dim = deg θ`), attached to rational places (`rat_places`) and
/// algebraic-place orbits (`alg_places`, see [`AlgPlace`]).  Decomposes all
/// residues over `ℚ`; for each component the rational-coefficient divisor `δ_k`
/// (over both place kinds) is tested by [`find_order_genus_ge2_alg`].  Returns
/// `NonElementary` if any component is non-torsion, `Principal` (lcm of the
/// component orders) if all are torsion, else `NotDecided`.
///
/// Each component's coordinates are cleared to integers (a constant rescaling
/// that preserves torsion-ness; `find_order_genus_ge2_alg` re-primitivizes, so
/// the reported order is that of the primitive class).
#[allow(clippy::too_many_arguments)]
pub(crate) fn trager_log_criterion_alg(
    n: usize,
    a: &QPoly,
    rat_places: &[PlacedResidue],
    rat_residues: &[KElem],
    alg_places: &[AlgPlace],
    alg_residues: &[KElem],
    dim: usize,
) -> FindOrder {
    if rat_places.len() != rat_residues.len() || alg_places.len() != alg_residues.len() {
        return FindOrder::NotDecided;
    }
    let nr = rat_residues.len();
    let all_res: Vec<KElem> = rat_residues.iter().chain(alg_residues).cloned().collect();
    if all_res.is_empty() {
        return FindOrder::NotDecided;
    }
    let (ncomp, coords) = qbasis_decompose(&all_res, dim);
    if ncomp == 0 {
        return FindOrder::Principal { order: 1 };
    }

    let mut lcm_order: u32 = 1;
    for k in 0..ncomp {
        // Clear denominators of this component's coordinates to integers.
        let mut l = Integer::from(1);
        for c in &coords {
            l = l.lcm(c[k].denom());
        }
        let rat_div: Vec<PlacedResidue> = rat_places
            .iter()
            .zip(&coords[..nr])
            .map(|(pl, c)| PlacedResidue {
                residue: Residue {
                    point: pl.residue.point.clone(),
                    at_infinity: pl.residue.at_infinity,
                    sheet: pl.residue.sheet,
                    ramification: pl.residue.ramification,
                    value: c[k].clone() * Rational::from(l.clone()),
                },
                y_coord: pl.y_coord.clone(),
            })
            .collect();
        let alg_div: Vec<AlgPlace> = alg_places
            .iter()
            .zip(&coords[nr..])
            .map(|(ap, c)| AlgPlace {
                minpoly: ap.minpoly.clone(),
                x_coord: ap.x_coord.clone(),
                y_coord: ap.y_coord.clone(),
                coeff: (c[k].clone() * Rational::from(l.clone())).numer().clone(),
                orbit: ap.orbit,
            })
            .collect();
        match find_order_genus_ge2_alg(n, a, &rat_div, &alg_div) {
            FindOrder::NonElementary => return FindOrder::NonElementary,
            FindOrder::Principal { order } => lcm_order = lcm_u32(lcm_order, order),
            FindOrder::NotDecided => return FindOrder::NotDecided,
        }
    }
    FindOrder::Principal { order: lcm_order }
}

fn lcm_u32(a: u32, b: u32) -> u32 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd_u32(a, b) * b
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(cs: &[i64]) -> QPoly {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }
    fn ke(cs: &[i64]) -> KElem {
        cs.iter().map(|&c| Rational::from(c)).collect()
    }
    fn place(point: i64, y: i64, inf: bool) -> PlacedResidue {
        PlacedResidue {
            residue: Residue {
                point: Rational::from(point),
                at_infinity: inf,
                sheet: 0,
                ramification: 1,
                value: Rational::from(0),
            },
            y_coord: Rational::from(y),
        }
    }

    /// ℚ-basis decomposition of `{√2, −√2, 3}` in `ℚ(√2)` (dim 2): the span is
    /// 2-dimensional (`{1, √2}` via pivots), with the expected coordinates.
    #[test]
    fn decompose_quadratic_field() {
        // √2 = [0,1], −√2 = [0,−1], 3 = [3,0].
        let res = [ke(&[0, 1]), ke(&[0, -1]), ke(&[3, 0])];
        let (ncomp, coords) = qbasis_decompose(&res, 2);
        assert_eq!(ncomp, 2); // span = ℚ·1 ⊕ ℚ·√2
                              // Pivot columns are 0 (the `1` part) then 1 (the `√2` part):
        assert_eq!(coords[0], vec![Rational::from(0), Rational::from(1)]); // √2
        assert_eq!(coords[1], vec![Rational::from(0), Rational::from(-1)]); // −√2
        assert_eq!(coords[2], vec![Rational::from(3), Rational::from(0)]); // 3
    }

    /// 1-dimensional span: `{√2, −√2}` ⇒ one component, coords `+1, −1`.
    #[test]
    fn decompose_one_dimensional() {
        let res = [ke(&[0, 1]), ke(&[0, -1])];
        let (ncomp, coords) = qbasis_decompose(&res, 2);
        assert_eq!(ncomp, 1);
        assert_eq!(coords[0], vec![Rational::from(1)]);
        assert_eq!(coords[1], vec![Rational::from(-1)]);
    }

    /// Trager criterion, **NonElementary**: residues `√2` at `(3,5)` and `−√2`
    /// at `∞` on the Mordell curve `y²=x³−2` (rank 1).  The single `√2`-component
    /// divisor `(3,5) − ∞` maps to the infinite-order point `(3,5)` ⇒
    /// non-torsion ⇒ `NonElementary`.
    #[test]
    fn criterion_nonelementary_via_component() {
        let a = qp(&[-2, 0, 0, 1]); // x³ − 2
        let places = [place(3, 5, false), place(0, 0, true)];
        let residues = [ke(&[0, 1]), ke(&[0, -1])]; // √2, −√2
        assert_eq!(
            trager_log_criterion(2, &a, &places, &residues, 2),
            FindOrder::NonElementary
        );
    }

    /// Trager criterion, **all torsion**: residues `√2` at `(−1,0)` and `−√2` at
    /// `∞` on `y²=x³+1` (ℤ/6).  The `√2`-component `(−1,0) − ∞` is the 2-torsion
    /// point `(−1,0)` ⇒ `Principal{2}` (the log part exists).
    #[test]
    fn criterion_principal_via_component() {
        let a = qp(&[1, 0, 0, 1]); // x³ + 1
        let places = [place(-1, 0, false), place(0, 0, true)];
        let residues = [ke(&[0, 1]), ke(&[0, -1])]; // √2, −√2
        assert_eq!(
            trager_log_criterion(2, &a, &places, &residues, 2),
            FindOrder::Principal { order: 2 }
        );
    }

    /// Criterion with an **algebraic place**: on the genus-2 curve
    /// `y²=(x²−2)(x³+1)`, residues in `ℚ(√2)` — `√2` at the algebraic branch
    /// orbit over `x²−2`, `−√2` at the rational branch point `(−1,0)`.  The
    /// single `√2`-component divisor is branch-only ⇒ 2-torsion ⇒ `Principal`.
    #[test]
    fn criterion_alg_place_principal() {
        let a = qp(&[-2, 0, 1, -2, 0, 1]); // (x²−2)(x³+1) = x⁵ − 2x³ + x² − 2
        let rat_places = [place(-1, 0, false)]; // branch point (−1,0)
        let rat_residues = [ke(&[0, -1])]; // −√2
        let alg_places = [AlgPlace {
            minpoly: qp(&[-2, 0, 1]), // x² − 2
            x_coord: qp(&[0, 1]),     // θ
            y_coord: Vec::new(),      // branch
            coeff: Integer::from(0),  // (coeff is set per-component by the criterion)
            orbit: true,              // Galois orbit of the two branch points
        }];
        let alg_residues = [ke(&[0, 1])]; // √2
        let res = trager_log_criterion_alg(
            2,
            &a,
            &rat_places,
            &rat_residues,
            &alg_places,
            &alg_residues,
            2,
        );
        assert!(matches!(res, FindOrder::Principal { .. }), "got {res:?}");
    }

    /// Genuine **two-component** all-torsion case on `y²=x³+1` (ℤ/6), residues
    /// summing to zero (residue theorem): `(2,3)↦1`, `(−1,0)↦√2`, `∞↦−1−√2`.
    /// The `1`-component is `(2,3)−∞` (order 6), the `√2`-component is `(−1,0)−∞`
    /// (order 2) ⇒ both torsion ⇒ `Principal{lcm(6,2)=6}`.
    #[test]
    fn criterion_two_components_principal() {
        let a = qp(&[1, 0, 0, 1]); // x³ + 1
        let places = [place(2, 3, false), place(-1, 0, false), place(0, 0, true)];
        let residues = [ke(&[1, 0]), ke(&[0, 1]), ke(&[-1, -1])]; // 1, √2, −1−√2
        assert_eq!(
            trager_log_criterion(2, &a, &places, &residues, 2),
            FindOrder::Principal { order: 6 }
        );
    }
}
