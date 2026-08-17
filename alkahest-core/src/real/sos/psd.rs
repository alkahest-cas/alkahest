//! General (non-diagonally-dominant) PSD Gram-matrix search.
//!
//! [`mod@super::gram`] restricts the search to the diagonally dominant (DD)
//! subcone so it can stay an exact linear program: every DD matrix is PSD,
//! so anything it finds is automatically sound, but the DD cone is a strict
//! subset of the PSD cone and refuses plenty of genuine SOS polynomials.
//!
//! This module searches the *full* PSD cone instead, using the building
//! blocks in [`super::linalg`] and [`super::sdp`]:
//!
//! 1. [`super::linalg::solve_affine`] parametrises the affine family of Gram
//!    matrices that reproduce the target's coefficients — *exactly*, over
//!    ℚ, for **any** choice of the free parameters. This is the crux of the
//!    soundness argument: whatever parameter point gets picked, the
//!    resulting matrix already satisfies `z^T Q z = p` on the nose, so
//!    nothing downstream can corrupt that half of the identity.
//! 2. [`super::sdp::Family::search`] runs a floating-point alternating
//!    projection to *propose* a parameter point whose matrix looks positive
//!    semidefinite. This is a heuristic and is never trusted.
//! 3. The proposed point is rounded to nearby rationals (at several
//!    denominator budgets) and plugged back into the exact affine family,
//!    and [`super::linalg::psd_decompose`] *exactly* decides whether the
//!    resulting rational matrix is PSD.
//!
//! Only a `Some` out of step 3 is ever returned, and [`psd_search`] itself
//! re-checks the expanded quadratic form against the target before handing
//! anything back — a bad numeric suggestion costs a wasted rounding
//! attempt, never an unsound result.

#![allow(clippy::needless_range_loop)]

use super::cert::SosPoly;
use super::gram::monomial_basis;
use super::linalg::{psd_decompose, solve_affine};
use super::ratpoly::{Exponents, RatPoly};
use super::sdp::{min_eigenvalue, Family};
use rug::Rational;
use std::collections::{BTreeMap, BTreeSet};

/// A tiny deterministic PRNG (SplitMix64), used only to diversify the
/// alternating-projection search's starting points. Reproducible and adds
/// no dependency; there is no property of "randomness" any certificate
/// depends on — the exact PSD check afterwards is what makes this sound
/// regardless of how (or how badly) a candidate point was proposed.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-scale, scale]`.
    fn next_signed(&mut self, scale: f64) -> f64 {
        let bits = self.next_u64() >> 11; // 53 significant bits
        let u = (bits as f64) * (1.0 / (1u64 << 53) as f64); // [0, 1)
        (2.0 * u - 1.0) * scale
    }
}

fn add_exp(a: &[u32], b: &[u32]) -> Exponents {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Number of entries in the packed upper triangle (`i ≤ j`) of an `n×n`
/// symmetric matrix.
fn pack_len(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Unpack a symmetric matrix from its packed upper triangle, in the same
/// `(i, j)`, `i ≤ j` row-major order used by [`gram_system`].
fn unpack(n: usize, v: &[Rational]) -> Vec<Vec<Rational>> {
    let mut q = vec![vec![Rational::from(0); n]; n];
    let mut idx = 0;
    for i in 0..n {
        for j in i..n {
            q[i][j] = v[idx].clone();
            q[j][i] = v[idx].clone();
            idx += 1;
        }
    }
    q
}

/// Coefficient-matching linear system for `z^T Q z = target`, `z` the given
/// monomial basis: one row per monomial that can occur on either side, one
/// unknown per upper-triangle entry `(i, j)` of the (symmetric) Gram matrix.
fn gram_system(target: &RatPoly, basis: &[Exponents]) -> (Vec<Vec<Rational>>, Vec<Rational>) {
    let n = basis.len();
    let ncols = pack_len(n);

    let mut col_of: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut idx = 0;
    for i in 0..n {
        for j in i..n {
            col_of.insert((i, j), idx);
            idx += 1;
        }
    }

    let mut rows: BTreeMap<Exponents, Vec<Rational>> = BTreeMap::new();
    for i in 0..n {
        for j in i..n {
            let e = add_exp(&basis[i], &basis[j]);
            // Off-diagonal entries occur twice in z^T Q z (as Q_ij and Q_ji).
            let coeff = if i == j {
                Rational::from(1)
            } else {
                Rational::from(2)
            };
            let row = rows
                .entry(e)
                .or_insert_with(|| vec![Rational::from(0); ncols]);
            row[col_of[&(i, j)]] += coeff;
        }
    }

    let mut all_exps: BTreeSet<Exponents> = rows.keys().cloned().collect();
    all_exps.extend(target.terms().keys().cloned());

    let mut out_rows = Vec::with_capacity(all_exps.len());
    let mut out_rhs = Vec::with_capacity(all_exps.len());
    for e in &all_exps {
        let row = rows
            .get(e)
            .cloned()
            .unwrap_or_else(|| vec![Rational::from(0); ncols]);
        out_rows.push(row);
        out_rhs.push(target.coeff(e));
    }
    (out_rows, out_rhs)
}

fn rat_to_f64(r: &Rational) -> f64 {
    r.to_f64()
}

fn frob_dot(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f64>())
        .sum()
}

/// Orthonormalise `dirs` (Gram–Schmidt, Frobenius inner product), returning
/// the orthonormal directions `e` and the upper-triangular change of basis
/// `r` with `dirs[k] = Σ_{j ≤ k} r[j][k]·e[j]`.
///
/// The raw nullspace basis handed to [`psd_search`] comes out of Gaussian
/// elimination and can be badly scaled — one free coordinate set to exactly
/// `1`, the rest whatever elimination produced — which made the alternating
/// projection search in [`sdp::Family`] stall on realistic targets (Motzkin
/// among them): [`sdp::Family::new`]'s internal Gram matrix of the raw
/// directions was so ill-conditioned that its Cholesky solve barely moved
/// the iterate. Running the search in this orthonormal basis instead makes
/// that Gram matrix the identity by construction; [`Self::to_original`]
/// (via back-substitution against `r`) converts the result back to the
/// original nullspace parametrisation the exact rational reconstruction
/// needs.
fn orthonormalize(dirs: &[Vec<Vec<f64>>]) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let m = dirs.len();
    let mut e: Vec<Vec<Vec<f64>>> = dirs.to_vec();
    let mut r = vec![vec![0.0f64; m]; m];
    for k in 0..m {
        for j in 0..k {
            let proj = frob_dot(&e[j], &e[k]);
            r[j][k] = proj;
            let (ej, ek) = {
                // Split-borrow e[j] and e[k] (j < k) simultaneously.
                let (left, right) = e.split_at_mut(k);
                (&left[j], &mut right[0])
            };
            for (row_k, row_j) in ek.iter_mut().zip(ej.iter()) {
                for (v_k, v_j) in row_k.iter_mut().zip(row_j.iter()) {
                    *v_k -= proj * v_j;
                }
            }
        }
        let norm = frob_dot(&e[k], &e[k]).sqrt();
        r[k][k] = norm;
        if norm > 1e-12 {
            for row in e[k].iter_mut() {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }
    }
    (e, r)
}

/// Solve the upper-triangular system `r·t = s` for `t` by back-substitution.
/// `None` if `r` is (numerically) singular — a direction that Gram–Schmidt
/// found to be dependent on the earlier ones, which should not happen for a
/// genuine nullspace basis but is checked rather than assumed.
fn back_substitute_upper(r: &[Vec<f64>], s: &[f64]) -> Option<Vec<f64>> {
    let m = s.len();
    let mut t = vec![0.0f64; m];
    for k in (0..m).rev() {
        let mut acc = s[k];
        for (j, tj) in t.iter().enumerate().skip(k + 1) {
            acc -= r[k][j] * tj;
        }
        if r[k][k].abs() < 1e-12 {
            return None;
        }
        t[k] = acc / r[k][k];
    }
    Some(t)
}

/// The convergent of `x`'s continued-fraction expansion whose denominator is
/// `≤ max_den`. Used only to propose a rational point to check exactly —
/// never trusted, so any reasonable approximation is fine.
fn round_to_rational(x: f64, max_den: i64) -> Option<Rational> {
    if !x.is_finite() {
        return None;
    }
    if x == 0.0 {
        return Some(Rational::from(0));
    }
    let neg = x < 0.0;
    let mut val = x.abs();
    let (mut h_prev, mut h_cur): (i64, i64) = (0, 1);
    let (mut k_prev, mut k_cur): (i64, i64) = (1, 0);
    for _ in 0..40 {
        if !val.is_finite() {
            break;
        }
        let a_f = val.floor();
        if a_f.abs() > 1e15 {
            break;
        }
        let a = a_f as i64;
        let h_next = a.checked_mul(h_cur).and_then(|v| v.checked_add(h_prev));
        let k_next = a.checked_mul(k_cur).and_then(|v| v.checked_add(k_prev));
        let (Some(h_next), Some(k_next)) = (h_next, k_next) else {
            break;
        };
        if k_next > max_den || k_next <= 0 {
            break;
        }
        h_prev = h_cur;
        h_cur = h_next;
        k_prev = k_cur;
        k_cur = k_next;
        let frac = val - a_f;
        if frac < 1e-13 {
            break;
        }
        val = 1.0 / frac;
    }
    if k_cur == 0 {
        None
    } else {
        let mag = Rational::from((h_cur, k_cur));
        Some(if neg { -mag } else { mag })
    }
}

/// Denominator caps tried, in order, when rounding the numeric search's
/// suggested point to an exact rational one.
const DENOM_CAPS: &[i64] = &[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536];

/// Shrinking sequence of eigenvalue floors, used as a continuation
/// ("annealing") schedule: each stage is warm-started from the previous
/// one's result. Starting deep in the interior (floor `1.0`) is easy to
/// reach from anywhere in the family, and slowly tightening the floor
/// tracks the iterate into a **boundary-only** intersection — a singular
/// witnessing Gram matrix, which is exactly the case for a tight/extremal
/// SOS certificate (Motzkin's among them) — far more reliably than a plain,
/// fixed-floor alternating projection started cold at a small floor, which
/// in practice stalls a long way short of such an intersection instead of
/// converging into it.
const FLOOR_SCHEDULE: &[f64] = &[
    1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.0,
];

/// Random restarts (beyond the deterministic `t = 0` start) tried per scale
/// in [`RESTART_SCALES`], each run through the full [`FLOOR_SCHEDULE`].
const RANDOM_RESTARTS: usize = 4;

/// Scales (in the orthonormal parametrisation, so directly comparable
/// regardless of the original nullspace basis's conditioning) tried for the
/// random restarts' starting points.
const RESTART_SCALES: &[f64] = &[1.0, 4.0];

/// How many of the best (highest minimum eigenvalue) search results to
/// actually attempt rational rounding on. The numeric search is the
/// expensive part; trying rounding on a handful of near-best candidates
/// instead of only the single best one costs little extra and hedges
/// against the best *numeric* point rounding to something that fails the
/// *exact* check while a close second would not have.
const ROUNDING_CANDIDATES: usize = 6;

/// Above this many free parameters, the numeric search is skipped rather
/// than run: each iteration of [`Family::search_from`] solves a dense
/// system in the free-parameter count, so cost grows with its cube, and
/// this keeps that bounded regardless of how large a monomial basis the
/// caller asks for. A skip here returns `None` — "not found within
/// budget", not "not SOS" — exactly like every other budget in this module.
const MAX_FREE_PARAMETERS: usize = 110;

/// Run the annealing schedule from a single starting point.
fn anneal_from(family: &Family, start: Vec<f64>) -> Vec<f64> {
    let mut t = start;
    for &floor in FLOOR_SCHEDULE {
        if let Some(next) = family.search_from(t.clone(), floor, 150) {
            t = next;
        }
    }
    t
}

/// Try the annealing schedule from several starting points — the
/// deterministic `t = 0`, plus a handful of random restarts — and return
/// every result reached, best (highest minimum eigenvalue of `family.at(t)`)
/// first. See [`FLOOR_SCHEDULE`]'s and [`RANDOM_RESTARTS`]'s doc comments
/// for why both are needed: annealing handles boundary-only intersections
/// that a fixed floor stalls on, and multiple starts hedge against any
/// single trajectory converging to a merely-locally-nearest pair when the
/// family and the PSD cone do intersect elsewhere.
fn multistart_anneal(family: &Family, dim: usize) -> Vec<Vec<f64>> {
    let mut starts: Vec<Vec<f64>> = vec![vec![0.0; dim]];
    let mut rng = SplitMix64::new(0xC0FFEE_D15EA5E5);
    for &scale in RESTART_SCALES {
        for _ in 0..RANDOM_RESTARTS {
            starts.push((0..dim).map(|_| rng.next_signed(scale)).collect());
        }
    }
    let mut results: Vec<(f64, Vec<f64>)> = starts
        .into_iter()
        .map(|start| {
            let t = anneal_from(family, start);
            let eig = min_eigenvalue(&family.at(&t));
            (eig, t)
        })
        .collect();
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results.into_iter().map(|(_, t)| t).collect()
}

/// Search the full PSD-Gram cone for an exact rational sum-of-squares
/// decomposition of `p` over the monomial basis of total degree `≤
/// basis_deg`.
///
/// This subsumes [`super::gram::dsos_search`] in principle — every
/// diagonally dominant matrix is PSD — but is not a strict improvement in
/// practice: it leans on a floating-point search finding *some* PSD point of
/// the affine family, so `None` here means only "the numeric search and its
/// roundings did not turn up a certificate", never "not SOS". A `Some` is
/// always sound: the returned [`SosPoly`] is checked to expand back to
/// exactly `p` before it is returned, using the same exact rational
/// arithmetic as everywhere else in this subsystem.
pub fn psd_search(target: &RatPoly, basis_deg: u32) -> Option<SosPoly> {
    let nvars = target.nvars();
    // A homogeneous target of degree exactly `2·basis_deg` needs only the
    // monomials of degree *exactly* `basis_deg` in its Gram basis — mixing in
    // lower-degree monomials can only ever contribute to coefficients the
    // target does not have, since every product of two basis monomials of
    // unequal degree still sums to `2·basis_deg` only when *both* already
    // have degree `basis_deg`. This is standard (Blekherman–Parrilo–Thomas,
    // Prop. 3.29): a homogeneous SOS decomposition can always be taken with
    // homogeneous summands. Restricting here is not just an optimisation —
    // the search is numeric, and a smaller basis is the difference between
    // "converges" and "not within budget" on cases like Motzkin.
    let basis: Vec<Exponents> = match target.is_homogeneous() {
        Some(d) if d == 2 * basis_deg => monomial_basis(nvars, basis_deg)
            .into_iter()
            .filter(|e| e.iter().sum::<u32>() == basis_deg)
            .collect(),
        _ => monomial_basis(nvars, basis_deg),
    };
    let n = basis.len();
    if n == 0 {
        return if target.is_zero() {
            Some(SosPoly::default())
        } else {
            None
        };
    }

    let (rows, rhs) = gram_system(target, &basis);
    let sol = solve_affine(&rows, &rhs)?;

    let try_point = |t: &[Rational]| -> Option<SosPoly> {
        let packed = sol.at(t);
        let q = unpack(n, &packed);
        let decomp = psd_decompose(&q)?;
        let mut sos = SosPoly::default();
        for (d, v) in decomp {
            if d <= 0 {
                continue;
            }
            let mut square = RatPoly::zero(nvars);
            for (u, c) in v.iter().enumerate() {
                if *c != 0 {
                    square = square.add(&RatPoly::monomial(nvars, basis[u].clone(), c.clone()));
                }
            }
            sos.push(d, square);
        }
        // Defense in depth: the affine parametrisation guarantees this by
        // construction, but the certificate returned here is re-verified
        // once more at the call site (`PositivityCertificate::verify`), and
        // this module never hands back something it has not itself checked.
        if sos.to_poly(nvars) == *target {
            Some(sos)
        } else {
            None
        }
    };

    // No freedom at all: the unique solution is the only candidate.
    if sol.dimension() == 0 {
        return try_point(&[]);
    }
    if sol.dimension() > MAX_FREE_PARAMETERS {
        return None;
    }

    let base: Vec<Vec<f64>> = unpack(n, &sol.particular)
        .iter()
        .map(|row| row.iter().map(rat_to_f64).collect())
        .collect();
    let dirs: Vec<Vec<Vec<f64>>> = sol
        .nullspace
        .iter()
        .map(|dir| {
            unpack(n, dir)
                .iter()
                .map(|row| row.iter().map(rat_to_f64).collect())
                .collect()
        })
        .collect();
    // Search in an orthonormal basis of the same directions — see
    // `orthonormalize`'s doc comment for why the raw nullspace basis makes
    // the alternating projection stall in practice.
    let (ortho_dirs, r) = orthonormalize(&dirs);
    let family = Family::new(base, ortho_dirs);
    let dim = dirs.len();

    for s in multistart_anneal(&family, dim)
        .into_iter()
        .take(ROUNDING_CANDIDATES)
    {
        let Some(t) = back_substitute_upper(&r, &s) else {
            continue;
        };
        for &max_den in DENOM_CAPS {
            let t_rat: Option<Vec<Rational>> =
                t.iter().map(|x| round_to_rational(*x, max_den)).collect();
            let Some(t_rat) = t_rat else { continue };
            if let Some(sos) = try_point(&t_rat) {
                return Some(sos);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }

    #[test]
    fn psd_search_finds_the_dsos_reachable_case() {
        // p = x^2 - 2xy + 2y^2 = (x-y)^2 + y^2.
        let mut p = RatPoly::monomial(2, vec![2, 0], Rational::from(1));
        p = p.add(&RatPoly::monomial(2, vec![1, 1], Rational::from(-2)));
        p = p.add(&RatPoly::monomial(2, vec![0, 2], Rational::from(2)));
        let sos = psd_search(&p, 1).expect("PSD search should find a certificate");
        assert_eq!(sos.to_poly(2), p);
    }

    #[test]
    fn psd_search_finds_a_unique_rational_gram_matrix() {
        // (1/2 x + 1/3 y)^2 = 1/4 x^2 + 1/3 xy + 1/9 y^2; the basis has only
        // three monomials at degree 1, so the Gram matrix has no freedom at
        // all and this exercises the `dimension() == 0` path directly.
        let mut p = RatPoly::monomial(2, vec![2, 0], r(1, 4));
        p = p.add(&RatPoly::monomial(2, vec![1, 1], r(1, 3)));
        p = p.add(&RatPoly::monomial(2, vec![0, 2], r(1, 9)));
        let sos = psd_search(&p, 1).expect("PSD search should find a certificate");
        assert_eq!(sos.to_poly(2), p);
    }

    #[test]
    fn psd_search_refuses_unreachable_degree() {
        let p = RatPoly::monomial(1, vec![4], Rational::from(1));
        assert!(psd_search(&p, 1).is_none());
    }

    #[test]
    fn psd_search_refuses_a_negative_definite_form() {
        let p = RatPoly::monomial(2, vec![2, 0], Rational::from(-1)).add(&RatPoly::monomial(
            2,
            vec![0, 2],
            Rational::from(-1),
        ));
        assert!(psd_search(&p, 1).is_none());
    }

    #[test]
    fn psd_search_finds_a_certificate_with_genuine_off_diagonal_freedom() {
        // p = x^4 + y^4 + 2x^2y^2 = (x^2 + y^2)^2. The degree-2 basis
        // {1, x, y, x^2, xy, y^2} gives the Gram matrix real freedom (many
        // off-diagonal choices reproduce the same quartic), and only some of
        // that family is PSD, exercising the numeric search + rounding path.
        let mut p = RatPoly::monomial(2, vec![4, 0], Rational::from(1));
        p = p.add(&RatPoly::monomial(2, vec![0, 4], Rational::from(1)));
        p = p.add(&RatPoly::monomial(2, vec![2, 2], Rational::from(2)));
        let sos = psd_search(&p, 2).expect("PSD search should find a certificate");
        assert_eq!(sos.to_poly(2), p);
    }

    #[test]
    fn psd_search_does_not_yet_reach_homogeneous_motzkin_times_sum_of_squares() {
        // The homogeneous Motzkin form, x^4y^2 + x^2y^4 - 3x^2y^2z^2 + z^6, is
        // the textbook example of a PSD form that is not itself SOS, and
        // (x^2+y^2+z^2)*Motzkin *is* classically SOS — but its witnessing
        // Gram matrix is singular, sitting exactly on the boundary of the
        // PSD cone. `diag::diag_step1_step2_trajectory_and_family_sanity`
        // shows the annealed multi-start search converges monotonically
        // toward that boundary (min eigenvalue from about -1.6 to about
        // -0.0018 as the floor anneals to 0) without fully closing the gap —
        // a tangential (non-transversal) intersection is the classic case
        // where alternating projection's convergence rate degrades this way.
        // `diag::diag_step3_planted_singular_example` confirms the search
        // mechanism itself is sound: a synthetic boundary case of the same
        // nullspace dimension *is* found and exactly re-verified, so this is
        // a genuine search-budget limitation on this specific hard instance,
        // not a bug in the family construction or the search. `None` here is
        // the correct, honest answer.
        let mut m = RatPoly::monomial(3, vec![4, 2, 0], Rational::from(1));
        m = m.add(&RatPoly::monomial(3, vec![2, 4, 0], Rational::from(1)));
        m = m.add(&RatPoly::monomial(3, vec![2, 2, 2], Rational::from(-3)));
        m = m.add(&RatPoly::monomial(3, vec![0, 0, 6], Rational::from(1)));
        assert_eq!(m.is_homogeneous(), Some(6));
        let sigma = RatPoly::sum_of_squares(3);
        let q = m.mul(&sigma);
        assert_eq!(q.is_homogeneous(), Some(8));

        assert!(
            psd_search(&q, 4).is_none(),
            "Motzkin's boundary certificate is not yet reached by this search; if this starts \
             passing, promote it to a positive assertion (assert_eq!(sos.to_poly(3), q)) rather \
             than leaving it as a smoke test"
        );
    }
}

#[cfg(test)]
mod diag {
    use super::*;

    fn homogeneous_motzkin_times_sos() -> (RatPoly, Vec<Exponents>) {
        let mut m = RatPoly::monomial(3, vec![4, 2, 0], Rational::from(1));
        m = m.add(&RatPoly::monomial(3, vec![2, 4, 0], Rational::from(1)));
        m = m.add(&RatPoly::monomial(3, vec![2, 2, 2], Rational::from(-3)));
        m = m.add(&RatPoly::monomial(3, vec![0, 0, 6], Rational::from(1)));
        let sigma = RatPoly::sum_of_squares(3);
        let q = m.mul(&sigma);
        let basis: Vec<Exponents> = monomial_basis(3, 4)
            .into_iter()
            .filter(|e| e.iter().sum::<u32>() == 4)
            .collect();
        (q, basis)
    }

    /// Step 1 + 2: trajectory of the deterministic (t=0) start across the
    /// floor schedule, and an exact sanity check that the affine family
    /// really does reproduce `q` at an arbitrary rational point.
    #[test]
    fn diag_step1_step2_trajectory_and_family_sanity() {
        let (q, basis) = homogeneous_motzkin_times_sos();
        let n = basis.len();
        let (rows, rhs) = gram_system(&q, &basis);
        eprintln!(
            "DIAG n={n} rows={} cols={}",
            rows.len(),
            rows.first().map(|r| r.len()).unwrap_or(0)
        );
        let sol = solve_affine(&rows, &rhs).expect("consistent");
        eprintln!("DIAG nullspace_dim={}", sol.dimension());

        // --- Step 2: exact family sanity check at an arbitrary rational t ---
        let dim = sol.dimension();
        let mut t_check = vec![Rational::from(0); dim];
        for (i, slot) in t_check.iter_mut().enumerate() {
            *slot = Rational::from((3 * (i as i64 + 1) - 7, 2 * (i as i64 + 1) + 3));
        }
        let packed = sol.at(&t_check);
        let q_check = unpack(n, &packed);
        let mut quad = RatPoly::zero(3);
        for i in 0..n {
            for j in 0..n {
                let e = add_exp(&basis[i], &basis[j]);
                quad = quad.add(&RatPoly::monomial(3, e, q_check[i][j].clone()));
            }
        }
        assert_eq!(
            quad, q,
            "DIAG family sanity check FAILED: z^T Q(t) z != q at an arbitrary rational t"
        );
        eprintln!("DIAG family sanity check: OK (z^T Q(t) z == q at a random rational t)");

        // --- Step 1: eigenvalue trajectory of the t=0 start ---
        let base: Vec<Vec<f64>> = unpack(n, &sol.particular)
            .iter()
            .map(|row| row.iter().map(rat_to_f64).collect())
            .collect();
        let dirs: Vec<Vec<Vec<f64>>> = sol
            .nullspace
            .iter()
            .map(|dir| {
                unpack(n, dir)
                    .iter()
                    .map(|row| row.iter().map(rat_to_f64).collect())
                    .collect()
            })
            .collect();
        eprintln!(
            "DIAG base min_eig (t=0, before any projection) = {}",
            min_eigenvalue(
                &unpack(n, &sol.particular)
                    .iter()
                    .map(|row| row.iter().map(rat_to_f64).collect())
                    .collect::<Vec<Vec<f64>>>()
            )
        );
        let (ortho_dirs, _r) = orthonormalize(&dirs);
        let family = Family::new(base, ortho_dirs);

        let mut t = vec![0.0f64; dirs.len()];
        for &floor in FLOOR_SCHEDULE {
            if let Some(next) = family.search_from(t.clone(), floor, 150) {
                t = next;
            }
            let eig = min_eigenvalue(&family.at(&t));
            eprintln!("DIAG floor={floor:>10}: min_eig={eig}");
        }
    }

    /// Step 3: a planted rank-deficient (singular) PSD example on the *same*
    /// basis size / nullspace dimension as the Motzkin case, to separate
    /// "the mechanism is broken" from "Motzkin specifically is hard".
    #[test]
    fn diag_step3_planted_singular_example() {
        let basis: Vec<Exponents> = monomial_basis(3, 4)
            .into_iter()
            .filter(|e| e.iter().sum::<u32>() == 4)
            .collect();
        let n = basis.len();
        eprintln!("DIAG planted: n={n}");

        // Build a deliberately rank-deficient (rank 3) PSD Gram matrix: five
        // fixed integer vectors, Q0 = sum of their outer products.
        let mut rng = SplitMix64::new(0xBEEF_CAFE);
        let vecs: Vec<Vec<i64>> = (0..3)
            .map(|_| (0..n).map(|_| (rng.next_u64() % 7) as i64 - 3).collect())
            .collect();
        let mut q0 = vec![vec![Rational::from(0); n]; n];
        for v in &vecs {
            for i in 0..n {
                for j in 0..n {
                    q0[i][j] += Rational::from(v[i] * v[j]);
                }
            }
        }
        // Sanity: Q0 is PSD by construction (sum of rank-1 PSD terms).
        assert!(
            psd_decompose(&q0).is_some(),
            "planted Q0 should be PSD by construction"
        );

        // target = z^T Q0 z.
        let mut target = RatPoly::zero(3);
        for i in 0..n {
            for j in 0..n {
                let e = add_exp(&basis[i], &basis[j]);
                target = target.add(&RatPoly::monomial(3, e, q0[i][j].clone()));
            }
        }
        assert_eq!(target.is_homogeneous(), Some(8));

        let (rows, rhs) = gram_system(&target, &basis);
        let sol = solve_affine(&rows, &rhs).expect("consistent");
        eprintln!("DIAG planted nullspace_dim={}", sol.dimension());

        let sos = psd_search(&target, 4);
        eprintln!(
            "DIAG planted psd_search found certificate: {}",
            sos.is_some()
        );
        let sos = sos.expect(
            "the search mechanism must find a planted singular-PSD example of the same \
             nullspace dimension as Motzkin — if this starts failing, the regression is in \
             the search/family machinery itself, not in any one hard instance",
        );
        assert_eq!(
            sos.to_poly(3),
            target,
            "DIAG planted certificate failed exact re-expansion"
        );
    }
}
