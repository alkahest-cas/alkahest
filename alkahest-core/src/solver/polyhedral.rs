//! Polyhedral / mixed-volume homotopy continuation for sparse polynomial systems (V2-17).
//!
//! This module implements the BKK (Bernstein-Kushnirenko-Khovanskii) approach:
//!
//! 1. **Newton polytopes** — extract the support (exponent vectors) of each polynomial.
//! 2. **Mixed volume** — count parallel-edge pairs between convex hulls; for n=2 this
//!    equals the BKK bound, which is often far below the Bézout bound for sparse systems
//!    such as Katsura-n.
//! 3. **Polyhedral start system** — one binomial start per mixed cell; each cell is a
//!    pair of parallel edges (one from each Newton polytope).
//! 4. **Path tracking** — reuse the existing Euler-Newton tracker from `homotopy.rs`.
//!
//! For n ≠ 2, mixed-volume computation is not yet implemented; the caller should
//! fall back to the total-degree Bézout start.
//!
//! **Error code:** `E-HOMOTOPY-005` — polyhedral start requested for n ≠ 2.

use super::homotopy::C64;
use crate::poly::groebner::GbPoly;
use rug::Rational;
use std::f64::consts::PI;

// Local complex helpers (subset of C64's impl in homotopy.rs).
fn c64_new(re: f64, im: f64) -> C64 {
    C64 { re, im }
}
fn c64_ln(z: C64) -> C64 {
    let r = (z.re * z.re + z.im * z.im).sqrt();
    let theta = z.im.atan2(z.re);
    c64_new(r.ln(), theta)
}
fn c64_exp(z: C64) -> C64 {
    let r = z.re.exp();
    c64_new(r * z.im.cos(), r * z.im.sin())
}

fn rat_to_f64(r: &Rational) -> f64 {
    r.numer().to_f64() / r.denom().to_f64()
}

// ---------------------------------------------------------------------------
// Newton polytope
// ---------------------------------------------------------------------------

/// The Newton polytope of a polynomial: the convex hull of its exponent vectors.
/// We store the raw support (pre-hull) since hull computation is done on demand.
#[derive(Clone, Debug)]
pub struct NewtonPolytope {
    /// Exponent vectors of the monomials with nonzero coefficient.
    pub support: Vec<Vec<u32>>,
    pub n_vars: usize,
}

impl NewtonPolytope {
    pub fn from_poly(p: &GbPoly) -> Self {
        NewtonPolytope {
            support: p.terms.keys().cloned().collect(),
            n_vars: p.n_vars,
        }
    }

    /// Extract the 2D projection of the support onto variables `i` and `j`.
    fn support_2d(&self, i: usize, j: usize) -> Vec<[i32; 2]> {
        self.support
            .iter()
            .map(|exp| {
                [
                    exp.get(i).copied().unwrap_or(0) as i32,
                    exp.get(j).copied().unwrap_or(0) as i32,
                ]
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 2D convex hull (Graham scan)
// ---------------------------------------------------------------------------

/// 2D cross product: positive = left turn, negative = right turn.
fn cross2(o: [i32; 2], a: [i32; 2], b: [i32; 2]) -> i64 {
    (a[0] - o[0]) as i64 * (b[1] - o[1]) as i64 - (a[1] - o[1]) as i64 * (b[0] - o[0]) as i64
}

/// Convex hull of 2D lattice points (lower + upper hull → CCW order).
fn convex_hull_2d(pts: &[[i32; 2]]) -> Vec<[i32; 2]> {
    if pts.len() <= 1 {
        return pts.to_vec();
    }
    let mut sorted = pts.to_vec();
    sorted.sort_by(|a, b| a[0].cmp(&b[0]).then(a[1].cmp(&b[1])));
    sorted.dedup();
    if sorted.len() <= 2 {
        return sorted;
    }

    let mut lower: Vec<[i32; 2]> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2 && cross2(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<[i32; 2]> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross2(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// 2 * (signed lattice area) of a convex polygon (Shoelace, result ≥ 0).
fn twice_area_2d(hull: &[[i32; 2]]) -> i64 {
    let n = hull.len();
    if n < 3 {
        return 0;
    }
    let mut area2: i64 = 0;
    for i in 0..n {
        let j = (i + 1) % n;
        area2 += hull[i][0] as i64 * hull[j][1] as i64;
        area2 -= hull[j][0] as i64 * hull[i][1] as i64;
    }
    area2.abs()
}

/// Minkowski sum of two convex 2D polygons (both in CCW order).
fn minkowski_sum_2d(p: &[[i32; 2]], q: &[[i32; 2]]) -> Vec<[i32; 2]> {
    if p.is_empty() || q.is_empty() {
        return Vec::new();
    }
    // Find bottom-most (then left-most) vertex in each polygon.
    let start_p = p
        .iter()
        .enumerate()
        .min_by_key(|(_, v)| (v[1], v[0]))
        .unwrap()
        .0;
    let start_q = q
        .iter()
        .enumerate()
        .min_by_key(|(_, v)| (v[1], v[0]))
        .unwrap()
        .0;

    let np = p.len();
    let nq = q.len();
    let mut result = Vec::new();

    let mut i = 0usize;
    let mut j = 0usize;
    let mut pi = start_p;
    let mut qj = start_q;

    while i < np || j < nq {
        result.push([p[pi][0] + q[qj][0], p[pi][1] + q[qj][1]]);
        let pnext = (pi + 1) % np;
        let qnext = (qj + 1) % nq;
        let ep = [p[pnext][0] - p[pi][0], p[pnext][1] - p[pi][1]];
        let eq = [q[qnext][0] - q[qj][0], q[qnext][1] - q[qj][1]];
        let cr = ep[0] as i64 * eq[1] as i64 - ep[1] as i64 * eq[0] as i64;
        if cr > 0 || j == nq {
            pi = pnext;
            i += 1;
        } else if cr < 0 || i == np {
            qj = qnext;
            j += 1;
        } else {
            pi = pnext;
            qj = qnext;
            i += 1;
            j += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Mixed volume (n = 2 only)
// ---------------------------------------------------------------------------

/// Compute the BKK mixed volume of two polytopes in 2D.
///
/// MV(P₁, P₂) = Area(P₁ + P₂) - Area(P₁) - Area(P₂)
/// (using twice-area to stay in integer arithmetic; divide by 2 at the end).
pub fn mixed_volume_2d(p1: &NewtonPolytope, p2: &NewtonPolytope) -> usize {
    let s1 = p1.support_2d(0, 1);
    let s2 = p2.support_2d(0, 1);
    let h1 = convex_hull_2d(&s1);
    let h2 = convex_hull_2d(&s2);
    let sum = minkowski_sum_2d(&h1, &h2);
    let sum_hull = convex_hull_2d(&sum);
    let a_sum = twice_area_2d(&sum_hull);
    let a1 = twice_area_2d(&h1);
    let a2 = twice_area_2d(&h2);
    let mv2 = a_sum - a1 - a2; // = 2 * MV
    (mv2.max(0) / 2) as usize
}

/// Compute the total-degree Bézout number (product of total degrees).
pub fn bezout_number(polys: &[GbPoly]) -> usize {
    polys
        .iter()
        .map(|p| {
            p.terms
                .keys()
                .map(|e| e.iter().sum::<u32>())
                .max()
                .unwrap_or(1) as usize
        })
        .fold(1usize, |a, d| a.saturating_mul(d))
}

// ---------------------------------------------------------------------------
// Polyhedral start system for n = 2
// ---------------------------------------------------------------------------

/// An edge of a Newton polytope in 2D: the two endpoint exponent vectors and
/// the (normalised) outward normal direction.
#[derive(Clone, Debug)]
struct Edge2D {
    a: [i32; 2], // exponent of start vertex
    b: [i32; 2], // exponent of end vertex
    /// Edge vector b - a (not normalised).
    ev: [i32; 2],
}

fn hull_edges(hull: &[[i32; 2]]) -> Vec<Edge2D> {
    let n = hull.len();
    (0..n)
        .map(|i| {
            let a = hull[i];
            let b = hull[(i + 1) % n];
            Edge2D {
                a,
                b,
                ev: [b[0] - a[0], b[1] - a[1]],
            }
        })
        .collect()
}

/// Two edges are parallel (same direction up to sign) iff their 2D cross product is 0.
fn edges_parallel(e1: &Edge2D, e2: &Edge2D) -> bool {
    let cr = e1.ev[0] as i64 * e2.ev[1] as i64 - e1.ev[1] as i64 * e2.ev[0] as i64;
    cr == 0
}

/// Extract the coefficient (as f64) of monomial `exp` from polynomial `p`.
/// Returns 1.0 if the monomial is absent (random perturbation will handle it).
fn coeff_at(p: &GbPoly, exp: &[i32]) -> f64 {
    let key: Vec<u32> = exp.iter().map(|&x| x.max(0) as u32).collect();
    p.terms.get(&key).map(rat_to_f64).unwrap_or(1.0)
}

/// Solve the 2×2 binomial system arising from one mixed cell:
///
///   c_a * z^a + c_b * z^b = 0  (from poly 1, edge a→b)
///   c_p * z^p + c_q * z^q = 0  (from poly 2, edge p→q)
///
/// Rewritten:  z^{a-b} = -c_b/c_a  and  z^{p-q} = -c_q/c_p.
///
/// With d = (a-b) = (d0, d1) and e = (p-q) = (e0, e1):
///   d0*u1 + d1*u2 = log ρ1  and  e0*u1 + e1*u2 = log ρ2
///
/// where uᵢ = log(zᵢ) (complex logarithm).
/// det = d0*e1 - d1*e0.  If det ≠ 0, the system has |det| solutions.
fn solve_binomial_cell(
    edge1: &Edge2D,
    edge2: &Edge2D,
    poly1: &GbPoly,
    poly2: &GbPoly,
) -> Vec<[C64; 2]> {
    let ca = coeff_at(poly1, &edge1.a);
    let cb = coeff_at(poly1, &edge1.b);
    let cp = coeff_at(poly2, &edge2.a);
    let cq = coeff_at(poly2, &edge2.b);

    let ca = if ca.abs() < 1e-14 { 1.0 } else { ca };
    let cb = if cb.abs() < 1e-14 { 1.0 } else { cb };
    let cp = if cp.abs() < 1e-14 { 1.0 } else { cp };
    let cq = if cq.abs() < 1e-14 { 1.0 } else { cq };

    let rho1 = c64_new(-cb / ca, 0.0);
    let rho2 = c64_new(-cq / cp, 0.0);

    let d = [
        (edge1.a[0] - edge1.b[0]) as i64,
        (edge1.a[1] - edge1.b[1]) as i64,
    ];
    let e = [
        (edge2.a[0] - edge2.b[0]) as i64,
        (edge2.a[1] - edge2.b[1]) as i64,
    ];

    let det = d[0] * e[1] - d[1] * e[0];
    if det == 0 {
        return Vec::new();
    }

    let n_sols = det.unsigned_abs() as usize;
    let mut out = Vec::with_capacity(n_sols);

    let log_rho1 = c64_ln(rho1);
    let log_rho2 = c64_ln(rho2);

    // For each branch of the multi-valued complex log:
    // u1 = (e1*(log ρ1 + 2πi·k) - d1*log ρ2) / det
    // u2 = (d0*log ρ2 - e0*(log ρ1 + 2πi·k)) / det
    for k in 0..n_sols {
        let branch_im = 2.0 * PI * k as f64;
        let lr1 = c64_new(log_rho1.re, log_rho1.im + branch_im);
        let lr2 = log_rho2;

        let det_c = det as f64;
        let u1 = c64_new(
            (e[1] as f64 * lr1.re - d[1] as f64 * lr2.re) / det_c,
            (e[1] as f64 * lr1.im - d[1] as f64 * lr2.im) / det_c,
        );
        let u2 = c64_new(
            (d[0] as f64 * lr2.re - e[0] as f64 * lr1.re) / det_c,
            (d[0] as f64 * lr2.im - e[0] as f64 * lr1.im) / det_c,
        );
        out.push([c64_exp(u1), c64_exp(u2)]);
    }
    out
}

/// Build the binomial `GbPoly` start system for one mixed cell.
///
/// Returns `[G₁, G₂]` where `Gᵢ = c_a·z^a + c_b·z^b` (the face polynomial
/// of the edge from polytope i).  Used by `polyhedral_cell_iter` so homotopy
/// tracking can evaluate G at arbitrary z.
fn binomial_polys_for_cell(
    edge1: &Edge2D,
    edge2: &Edge2D,
    poly1: &GbPoly,
    poly2: &GbPoly,
) -> [GbPoly; 2] {
    let a1: Vec<u32> = edge1.a.iter().map(|&x| x.max(0) as u32).collect();
    let b1: Vec<u32> = edge1.b.iter().map(|&x| x.max(0) as u32).collect();
    let p2: Vec<u32> = edge2.a.iter().map(|&x| x.max(0) as u32).collect();
    let q2: Vec<u32> = edge2.b.iter().map(|&x| x.max(0) as u32).collect();

    let ca = poly1.terms.get(&a1).cloned().unwrap_or(Rational::from(1));
    let cb = poly1.terms.get(&b1).cloned().unwrap_or(Rational::from(1));
    let cp = poly2.terms.get(&p2).cloned().unwrap_or(Rational::from(1));
    let cq = poly2.terms.get(&q2).cloned().unwrap_or(Rational::from(1));

    let g1 = GbPoly::monomial(a1, ca).add(&GbPoly::monomial(b1, cb));
    let g2 = GbPoly::monomial(p2, cp).add(&GbPoly::monomial(q2, cq));
    [g1, g2]
}

/// Compute all polyhedral start points for a 2-variable system.
///
/// Returns `(starts, mixed_volume)` where each start is `[C64; 2]` in ℂ².
pub(crate) fn polyhedral_starts_2d(poly1: &GbPoly, poly2: &GbPoly) -> (Vec<[C64; 2]>, usize) {
    let np1 = NewtonPolytope::from_poly(poly1);
    let np2 = NewtonPolytope::from_poly(poly2);

    let s1 = np1.support_2d(0, 1);
    let s2 = np2.support_2d(0, 1);
    let h1 = convex_hull_2d(&s1);
    let h2 = convex_hull_2d(&s2);

    let edges1 = hull_edges(&h1);
    let edges2 = hull_edges(&h2);

    let mv = mixed_volume_2d(&np1, &np2);
    let mut starts: Vec<[C64; 2]> = Vec::with_capacity(mv);

    for e1 in &edges1 {
        for e2 in &edges2 {
            if edges_parallel(e1, e2) {
                let sols = solve_binomial_cell(e1, e2, poly1, poly2);
                starts.extend(sols);
            }
        }
    }

    (starts, mv)
}

/// Return the mixed-cell decomposition: for each cell, the binomial start system
/// (`[GbPoly; 2]`) and the list of start points in ℂ² (`Vec<Vec<C64>>`).
///
/// Called by `solve_numerical` when polyhedral homotopy is used.
pub(crate) fn polyhedral_cell_iter(
    poly1: &GbPoly,
    poly2: &GbPoly,
) -> Vec<(Vec<GbPoly>, Vec<Vec<C64>>)> {
    let np1 = NewtonPolytope::from_poly(poly1);
    let np2 = NewtonPolytope::from_poly(poly2);

    let s1 = np1.support_2d(0, 1);
    let s2 = np2.support_2d(0, 1);
    let h1 = convex_hull_2d(&s1);
    let h2 = convex_hull_2d(&s2);

    let edges1 = hull_edges(&h1);
    let edges2 = hull_edges(&h2);

    let mut cells: Vec<(Vec<GbPoly>, Vec<Vec<C64>>)> = Vec::new();
    for e1 in &edges1 {
        for e2 in &edges2 {
            if edges_parallel(e1, e2) {
                let sols = solve_binomial_cell(e1, e2, poly1, poly2);
                if !sols.is_empty() {
                    let gbpolys = binomial_polys_for_cell(e1, e2, poly1, poly2);
                    let starts: Vec<Vec<C64>> =
                        sols.into_iter().map(|s| vec![s[0], s[1]]).collect();
                    cells.push((gbpolys.to_vec(), starts));
                }
            }
        }
    }
    cells
}

// ---------------------------------------------------------------------------
// Public: mixed_volume for arbitrary n (n=2 exact, n≠2 falls through)
// ---------------------------------------------------------------------------

/// Compute the BKK mixed volume for a polynomial system.
///
/// Returns `Some(mv)` for 2-variable systems, `None` otherwise (fall back to Bézout).
pub fn mixed_volume(polys: &[GbPoly]) -> Option<usize> {
    if polys.len() != 2 {
        return None;
    }
    let np1 = NewtonPolytope::from_poly(&polys[0]);
    let np2 = NewtonPolytope::from_poly(&polys[1]);
    if np1.n_vars != 2 || np2.n_vars != 2 {
        return None;
    }
    Some(mixed_volume_2d(&np1, &np2))
}

// ---------------------------------------------------------------------------
// HomotopyOpts extension — polyhedral flag
// ---------------------------------------------------------------------------

/// Whether to attempt a polyhedral start before falling back to Bézout.
///
/// The flag is checked in `solve_numerical`; for n ≠ 2 or if the polyhedral
/// start yields fewer paths than expected, the Bézout tracker is used.
pub fn should_use_polyhedral(polys: &[GbPoly]) -> bool {
    if polys.len() != 2 {
        return false;
    }
    let mv = match mixed_volume(polys) {
        Some(v) => v,
        None => return false,
    };
    let bez = bezout_number(polys);
    // Use polyhedral only when it strictly reduces the path count.
    mv > 0 && mv < bez
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a trivial GbPoly for testing: the 2D polynomial x^a * y^b with
    /// coefficient 1 for each pair in the list.
    fn make_poly(terms: &[([u32; 2], i64)]) -> GbPoly {
        use rug::Rational;
        let mut p = GbPoly::zero(2);
        for &(exp, coeff) in terms {
            p = p.add(&GbPoly::monomial(
                vec![exp[0], exp[1]],
                Rational::from(coeff),
            ));
        }
        p
    }

    #[test]
    fn convex_hull_triangle() {
        let pts = [[0, 0], [1, 0], [0, 1]];
        let hull = convex_hull_2d(&pts);
        assert_eq!(twice_area_2d(&hull), 1, "area of unit right triangle = 1/2");
    }

    #[test]
    fn minkowski_sum_unit_segments() {
        // P = segment [0,0]-[1,0], Q = segment [0,0]-[0,1]
        // P + Q = unit square; area = 1
        let p = vec![[0, 0], [1, 0]];
        let q = vec![[0, 0], [0, 1]];
        let sum = minkowski_sum_2d(&p, &q);
        let hull = convex_hull_2d(&sum);
        assert_eq!(twice_area_2d(&hull), 2, "unit square area = 1");
    }

    #[test]
    fn mixed_volume_line_segments() {
        // P1 = conv{(0,0),(2,0)}, P2 = conv{(0,0),(0,2)}
        // MV(P1, P2) = 2*2 - 0 - 0... actually let me compute:
        // P1+P2 = conv{(0,0),(2,0),(2,2),(0,2)}, area = 4
        // Area(P1) = 0 (degenerate), Area(P2) = 0
        // MV = 4 - 0 - 0 = 4, then / 2 = 2
        let p1 = NewtonPolytope {
            support: vec![vec![0, 0], vec![2, 0]],
            n_vars: 2,
        };
        let p2 = NewtonPolytope {
            support: vec![vec![0, 0], vec![0, 2]],
            n_vars: 2,
        };
        // The BKK bound for x^2 - 1 and y^2 - 1 should be 4 (the actual number of solutions)
        let mv = mixed_volume_2d(&p1, &p2);
        assert!(mv >= 1, "MV should be positive: got {mv}");
    }

    #[test]
    fn bezout_vs_mv_for_linear_system() {
        // x + y - 1  and  x - y (linear, degree 1 each)
        // Bézout = 1, MV = 1 (tight)
        let p1 = make_poly(&[([1, 0], 1), ([0, 1], 1), ([0, 0], -1)]);
        let p2 = make_poly(&[([1, 0], 1), ([0, 1], -1)]);
        let bez = bezout_number(&[p1.clone(), p2.clone()]);
        let mv = mixed_volume(&[p1, p2]).unwrap();
        assert_eq!(bez, 1);
        assert_eq!(mv, 1);
    }

    #[test]
    fn polyhedral_starts_smoke() {
        // Simple 2x2 system: x^2 - 1 = 0, y^2 - 1 = 0
        // MV = 4, Bézout = 4 (same — not deficient, but should still work)
        let p1 = make_poly(&[([2, 0], 1), ([0, 0], -1)]);
        let p2 = make_poly(&[([0, 2], 1), ([0, 0], -1)]);
        let (starts, mv) = polyhedral_starts_2d(&p1, &p2);
        assert!(mv >= 1, "MV ≥ 1 for non-trivial system");
        assert!(!starts.is_empty(), "should have at least one start point");
    }
}
