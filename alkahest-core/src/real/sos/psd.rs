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
//!
//! When the direct search fails, [`psd_search`] tries two further, cheaper
//! fallbacks in order, both restricted to a *smaller* affine family
//! (recomputing which is still bounded by `search_rational_family`'s own
//! cost, so this stays a small multiple of one full search rather than
//! unbounded escalation): `facial_reduction_search` guesses the true
//! certificate's near-null directions numerically, and — since this round —
//! `symmetry_reduced_search` restricts to the subspace fixed by `target`'s
//! own signed-permutation symmetry (variable permutations and independent
//! sign flips that leave it exactly invariant), when that subspace is
//! actually smaller. Both are gated to fire only once the cheaper search
//! above has already given up, so neither adds cost to a case the direct
//! search already closes.

#![allow(clippy::needless_range_loop)]

use super::cert::SosPoly;
use super::gram::monomial_basis;
use super::linalg::{psd_decompose, solve_affine};
use super::lp::{Lp, LpStatus, Rel};
use super::ratpoly::{Exponents, RatPoly};
use super::sdp::{min_eigenvalue, smallest_magnitude_eigenvectors, Family};
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

/// Above this much work — candidate basis monomials × support monomials of
/// the target — the half-Newton-polytope reduction below is skipped rather
/// than run: each candidate costs one exact-rational LP feasibility solve
/// whose column count is the support size, so this keeps the reduction's own
/// cost bounded no matter how large a basis the caller asks for. Skipping
/// only ever leaves the basis *wider* than necessary, so it can cost search
/// time but can never cost soundness or a certificate that would otherwise
/// have been found.
const MAX_NEWTON_REDUCTION_WORK: usize = 400_000;

/// Is `point` in the convex hull of `support`? Decided exactly, over ℚ, by
/// the same rational simplex the DSOS search uses: the hull membership
/// `point = Σ λ_i·support[i]`, `λ ≥ 0`, `Σ λ_i = 1` is a linear feasibility
/// programme verbatim.
///
/// A pivot-budget exhaustion (defensive; Bland's rule makes it unreachable)
/// is reported as `true` — "keep this monomial" — so an unexpected LP
/// outcome can only ever widen the basis, never narrow it wrongly.
fn in_convex_hull(support: &[Exponents], point: &[u32]) -> bool {
    let mut lp = Lp::new(support.len());
    for k in 0..point.len() {
        let row: Vec<Rational> = support.iter().map(|s| Rational::from(s[k])).collect();
        lp.constrain(row, Rel::Eq, Rational::from(point[k]));
    }
    lp.constrain(
        vec![Rational::from(1); support.len()],
        Rel::Eq,
        Rational::from(1),
    );
    !matches!(lp.solve(), LpStatus::Infeasible)
}

/// Restrict `basis` to the lattice points of `½·Newton(target)`.
///
/// Reznick's theorem (1978): if `p = Σ_i q_i²` then `Newton(q_i) ⊆
/// ½·Newton(p)` for **every** `i`. So every SOS decomposition of `p` is
/// already expressible over the monomials of `½·Newton(p)`, and dropping the
/// rest cannot lose a certificate — this is an exact, complete reduction,
/// not a heuristic narrowing, and that is what makes it safe to apply
/// unconditionally rather than as a fallback.
///
/// It matters because the numeric search's cost and its *accuracy* both
/// scale with the free-parameter count, and the free-parameter count scales
/// quadratically with the basis size. On `(x²+y²+z²)·Motzkin_hom` this cuts
/// the degree-4 ternary basis from 15 monomials to 9, which is the
/// difference between a 75-parameter affine family whose numeric solution
/// lands ~0.96 away from the true certificate in parameter space (and so
/// never rounds onto it) and an 18-parameter family that lands on it
/// exactly. Note that this is a *dimension* reduction and nothing else: the
/// certificate is the unique PSD point of the affine family on both bases,
/// with the same rank and the same zero minimum eigenvalue, so `λ_min` alone
/// does not reveal the difference — distance in parameter space does.
///
/// Forms whose Newton polytope already fills the simplex — Robinson's form
/// times `σ`, for instance, where 15 monomials reduce to 15 — come back
/// unchanged, which is exactly right: there is nothing to remove.
fn half_newton_reduce(target: &RatPoly, basis: Vec<Exponents>) -> Vec<Exponents> {
    let support: Vec<Exponents> = target.terms().keys().cloned().collect();
    if support.is_empty() || basis.is_empty() {
        return basis;
    }
    if basis.len().saturating_mul(support.len()) > MAX_NEWTON_REDUCTION_WORK {
        return basis;
    }
    let nvars = support[0].len();
    // Cheap coordinate-wise bounding box of Newton(target): a point outside
    // it cannot be in the hull, and this rejects most of the discarded
    // monomials without an LP solve at all.
    let mut hi = vec![0u32; nvars];
    for s in &support {
        for k in 0..nvars {
            hi[k] = hi[k].max(s[k]);
        }
    }
    basis
        .into_iter()
        .filter(|e| {
            let doubled: Vec<u32> = e.iter().map(|c| 2 * c).collect();
            if (0..nvars).any(|k| doubled[k] > hi[k]) {
                return false;
            }
            in_convex_hull(&support, &doubled)
        })
        .collect()
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
const MAX_FREE_PARAMETERS: usize = 200;

/// How [`MAX_FREE_PARAMETERS`] reports itself. Never silently: this ceiling
/// returns `None` *without searching at all*, which is a categorically
/// different thing from a search that ran and came up empty, and a caller
/// that cannot tell them apart will read "we did not look" as "we looked and
/// found nothing". `at_least` distinguishes the pre-solve estimate (a lower
/// bound on the dimension) from the exact post-solve count.
fn ceiling_note(basis_len: usize, free_params: usize, at_least: bool) -> String {
    let qualifier = if at_least { "at least " } else { "" };
    format!(
        "PSD Gram search: NOT SEARCHED — the affine Gram family over this {basis_len}-monomial \
         basis has {qualifier}{free_params} free parameters, above the numeric-search ceiling of \
         {MAX_FREE_PARAMETERS}; no search was attempted at this basis, so this is a budget that \
         fired, not a search that came up empty"
    )
}

/// Number of rows [`gram_system`] would build for `target` over `basis` —
/// one per monomial occurring on either side — without building the (large,
/// exact-rational) system itself.
fn row_count(target: &RatPoly, basis: &[Exponents]) -> usize {
    let mut exps: BTreeSet<Exponents> = target.terms().keys().cloned().collect();
    for i in 0..basis.len() {
        for j in i..basis.len() {
            exps.insert(add_exp(&basis[i], &basis[j]));
        }
    }
    exps.len()
}

/// Over-relaxation parameters tried for the Douglas–Rachford polish, in
/// increasing order of overshoot. `1.0` is plain (non-relaxed)
/// Douglas–Rachford; the larger value is the standard mitigation for a
/// stall at a shallow tangential approach to the intersection (see
/// `Family::douglas_rachford_from`'s doc comment). Kept to two values
/// deliberately: [`DR_ITERS`] below is what actually closes the gap on hard
/// (boundary-only) instances — see its doc comment — and that only stays
/// affordable across [`DR_POLISH_CANDIDATES`] starting points *and* several
/// facial-reduction attempts ([`FACIAL_SEARCH_BUDGET`]) if the spread here
/// is kept small.
const DR_LAMBDAS: &[f64] = &[1.0];

/// Iterations run per Douglas–Rachford attempt. Reflections make *some*
/// progress every iteration on a boundary-only (tangential) intersection,
/// but the rate is still only sublinear there — a diagnostic run on the
/// homogeneous Motzkin family (165 free parameters) needed on the order of
/// `10^4` iterations to bring the minimum eigenvalue from `~-2·10⁻³` (where
/// the alternating-projection annealing schedule alone stalls) down to
/// `~-5·10⁻⁶`, which is close enough that rational rounding at
/// [`DENOM_CAPS`]'s larger denominators reliably lands exactly on the true
/// (small-denominator) certificate. This is run only on the best few
/// annealed candidates (see [`DR_POLISH_CANDIDATES`]), not every start, to
/// keep that cost affordable.
const DR_ITERS: usize = 15_000;

/// How many of the (cheap, alternating-projection-only) annealed candidates
/// get the expensive Douglas–Rachford polish. Bounded well below the total
/// number of starts multistart annealing tries — see [`DR_ITERS`] for why
/// the polish itself is not cheap — since a candidate's annealed minimum
/// eigenvalue is already a good proxy for whether it is worth polishing:
/// the polish improves a near-feasible point, it does not rescue a
/// genuinely bad one.
const DR_POLISH_CANDIDATES: usize = 2;

/// Run the alternating-projection annealing schedule from a single starting
/// point. This alone reaches most cases outright, and gets *close* even on
/// the hard boundary-only ones (on the homogeneous Motzkin case
/// specifically, `diag::diag_step1_step2_trajectory_and_family_sanity`
/// shows it running the minimum eigenvalue from about −1.6 to about
/// −0.0018 and no further) — closing that last "close but stalled" stretch
/// is what the Douglas–Rachford polish in [`multistart_anneal`] is for.
fn anneal_from(family: &Family, start: Vec<f64>) -> Vec<f64> {
    let mut t = start;
    for &floor in FLOOR_SCHEDULE {
        // Returning the best point so far is always safe: these are only
        // *candidates*, and every one of them is re-verified exactly before it
        // can become a certificate.
        if !super::budget_ok() {
            return t;
        }
        if let Some(next) = family.search_from(t.clone(), floor, 150) {
            t = next;
        }
    }
    t
}

/// Try the annealing schedule from several starting points — the
/// deterministic `t = 0`, plus a handful of random restarts — then hand the
/// best few results to a deep Douglas–Rachford polish (see
/// `Family::douglas_rachford_from` and [`DR_ITERS`]'s doc comment), and
/// return every result reached, best (highest minimum eigenvalue of
/// `family.at(t)`) first.
///
/// See [`FLOOR_SCHEDULE`]'s and [`RANDOM_RESTARTS`]'s doc comments for why
/// both annealing and multiple starts are needed: annealing handles
/// boundary-only intersections that a fixed floor stalls on, and multiple
/// starts hedge against any single trajectory converging to a merely
/// locally-nearest pair when the family and the PSD cone do intersect
/// elsewhere. The Douglas–Rachford polish is applied only to the best few
/// annealed candidates, not every start — it is the standard reflection-based
/// upgrade for exactly the "close but stalled" signature annealing alone
/// leaves on a *tangential* (non-transversal) intersection (a singular
/// witnessing Gram matrix — the case for tight certificates like Motzkin's),
/// but running it to the depth that actually closes such a gap ([`DR_ITERS`])
/// is too expensive to spend on every start indiscriminately; a candidate's
/// annealed eigenvalue is already a good proxy for which ones are worth it.
/// Polishing can never make a candidate worse — its own annealed point is
/// always kept as a floor.
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

    for (eig, t) in results.iter_mut().take(DR_POLISH_CANDIDATES) {
        if !super::budget_ok() {
            break;
        }
        for &lambda in DR_LAMBDAS {
            if let Some(cand) = family.douglas_rachford_from(t.clone(), 0.0, lambda, DR_ITERS) {
                let cand_eig = min_eigenvalue(&family.at(&cand));
                if cand_eig > *eig {
                    *eig = cand_eig;
                    *t = cand;
                }
            }
        }
    }
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
/// Build `base + Σ t_k·dirs[k]` exactly, for a rational affine family whose
/// members are already unpacked `n×n` matrices (as opposed to
/// `linalg::AffineSolution::at`, which works in the packed upper-triangle
/// representation `gram_system` uses).
fn rat_family_at(
    base: &[Vec<Rational>],
    dirs: &[Vec<Vec<Rational>>],
    t: &[Rational],
) -> Vec<Vec<Rational>> {
    let n = base.len();
    let mut q = base.to_vec();
    for (tk, dir) in t.iter().zip(dirs.iter()) {
        if *tk == 0 {
            continue;
        }
        for i in 0..n {
            for j in 0..n {
                if dir[i][j] != 0 {
                    q[i][j] += Rational::from(tk * &dir[i][j]);
                }
            }
        }
    }
    q
}

/// Search the exact rational affine family `base_rat + Σ t_k·dirs_rat[k]`
/// (already unpacked `n×n` matrices) for a point that is both PSD and
/// reproduces `target` over `basis` — the shared core of [`psd_search`],
/// factored out so [`facial_reduction_search`] can run it again on a
/// *smaller* family without duplicating the numeric-search/rounding logic.
///
/// Returns the certificate if found, together with the best numeric
/// candidate matrices tried (nearest-to-feasible first) regardless of
/// whether a certificate was found — [`facial_reduction_search`] reads
/// near-null eigenvectors off those candidates, so they are worth handing
/// back even from a `None` result.
fn search_rational_family(
    nvars: usize,
    basis: &[Exponents],
    target: &RatPoly,
    base_rat: &[Vec<Rational>],
    dirs_rat: &[Vec<Vec<Rational>>],
    log: &mut Vec<String>,
) -> (Option<SosPoly>, Vec<Vec<Vec<f64>>>) {
    let try_point = |t: &[Rational]| -> Option<SosPoly> {
        let q = rat_family_at(base_rat, dirs_rat, t);
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
    if dirs_rat.is_empty() {
        return (try_point(&[]), Vec::new());
    }
    if dirs_rat.len() > MAX_FREE_PARAMETERS {
        log.push(ceiling_note(basis.len(), dirs_rat.len(), false));
        return (None, Vec::new());
    }
    log.push(format!(
        "PSD Gram search: searching a {}-parameter affine family over a {}-monomial basis",
        dirs_rat.len(),
        basis.len()
    ));

    let base: Vec<Vec<f64>> = base_rat
        .iter()
        .map(|row| row.iter().map(rat_to_f64).collect())
        .collect();
    let dirs: Vec<Vec<Vec<f64>>> = dirs_rat
        .iter()
        .map(|dir| {
            dir.iter()
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

    let candidates = multistart_anneal(&family, dim);
    let candidate_matrices: Vec<Vec<Vec<f64>>> = candidates.iter().map(|s| family.at(s)).collect();

    for s in candidates.iter().take(ROUNDING_CANDIDATES) {
        if !super::budget_ok() {
            return (None, candidate_matrices);
        }
        let Some(t) = back_substitute_upper(&r, s) else {
            continue;
        };
        for &max_den in DENOM_CAPS {
            let t_rat: Option<Vec<Rational>> =
                t.iter().map(|x| round_to_rational(*x, max_den)).collect();
            let Some(t_rat) = t_rat else { continue };
            if let Some(sos) = try_point(&t_rat) {
                return (Some(sos), candidate_matrices);
            }
        }
    }
    (None, candidate_matrices)
}

/// Number of near-null eigenvector directions imposed at once, tried in
/// increasing order — the guessed *corank* of the true certificate's
/// witnessing Gram matrix.
const FACIAL_CORANK_GUESSES: &[usize] = &[1, 2, 3];

/// Denominator caps tried when rounding a near-null eigenvector to an exact
/// rational direction. Coarser than [`DENOM_CAPS`]: facial reduction is a
/// *guess* at the true certificate's nullspace, and a wrong guess is cheap
/// to detect (`solve_affine`, or the final exact re-expansion check, simply
/// finds nothing) — so there is no benefit in trying many denominators here,
/// and [`DENOM_CAPS`]'s full spread is tried on the *reduced* family once a
/// guess produces one.
const FACIAL_DENOM_CAPS: &[i64] = &[64, 4096];

/// How many of the best numeric candidates (matrices) to read near-null
/// directions off of.
const FACIAL_CANDIDATES: usize = 2;

/// Total number of reduced-family searches [`facial_reduction_search`] will
/// actually run (across every candidate/corank/denominator combination
/// tried) — each one reruns the full numeric search machinery on a smaller
/// family, so this bounds the added cost to a small multiple of one
/// [`search_rational_family`] call, regardless of how many guesses are
/// considered.
const FACIAL_SEARCH_BUDGET: usize = 3;

/// Given a rational vector `v` — a *guessed* near-null direction of the
/// affine family's true PSD certificate — impose `Q(t)·v = 0` on the family
/// `base_rat + Σ t_k·dirs_rat[k]`, and return the resulting (generally
/// smaller) affine family in the same unpacked `n×n`-matrix representation.
/// `None` means the guess is inconsistent with the family — a routine "try
/// a different guess" outcome, not a bug: `dirs_rat`'s directions already
/// satisfy `z^T Q z = target` for any `t` by construction, and imposing
/// `Q(t)·v = 0` only ever cuts that same solution set down (to the sub-family
/// where a specific vector actually is a nullspace direction), never
/// enlarges or corrupts it — so any family this returns still reproduces
/// `target` exactly, for the same reason the original one does.
#[allow(clippy::type_complexity)]
fn restrict_by_near_null_vectors(
    base_rat: &[Vec<Rational>],
    dirs_rat: &[Vec<Vec<Rational>>],
    vs: &[Vec<Rational>],
) -> Option<(Vec<Vec<Rational>>, Vec<Vec<Vec<Rational>>>)> {
    let n = base_rat.len();
    let dim = dirs_rat.len();
    let mut rows = Vec::with_capacity(vs.len() * n);
    let mut rhs = Vec::with_capacity(vs.len() * n);
    for v in vs {
        for i in 0..n {
            let mut row = vec![Rational::from(0); dim];
            for (k, dir) in dirs_rat.iter().enumerate() {
                let mut acc = Rational::from(0);
                for j in 0..n {
                    if dir[i][j] != 0 && v[j] != 0 {
                        acc += Rational::from(&dir[i][j] * &v[j]);
                    }
                }
                row[k] = acc;
            }
            let mut b = Rational::from(0);
            for j in 0..n {
                if base_rat[i][j] != 0 && v[j] != 0 {
                    b += Rational::from(&base_rat[i][j] * &v[j]);
                }
            }
            rows.push(row);
            rhs.push(-b);
        }
    }
    let sol = solve_affine(&rows, &rhs)?;

    let mut new_base = base_rat.to_vec();
    for (k, tk) in sol.particular.iter().enumerate() {
        if *tk == 0 {
            continue;
        }
        for i in 0..n {
            for j in 0..n {
                if dirs_rat[k][i][j] != 0 {
                    new_base[i][j] += Rational::from(tk * &dirs_rat[k][i][j]);
                }
            }
        }
    }
    let mut new_dirs = Vec::with_capacity(sol.nullspace.len());
    for coeffs in &sol.nullspace {
        let mut nd = vec![vec![Rational::from(0); n]; n];
        for (k, ck) in coeffs.iter().enumerate() {
            if *ck == 0 {
                continue;
            }
            for i in 0..n {
                for j in 0..n {
                    if dirs_rat[k][i][j] != 0 {
                        nd[i][j] += Rational::from(ck * &dirs_rat[k][i][j]);
                    }
                }
            }
        }
        new_dirs.push(nd);
    }
    Some((new_base, new_dirs))
}

/// Facial-reduction fallback for when [`search_rational_family`]'s direct
/// numeric search gets *close* to a PSD point but never lands one that
/// survives exact rational rounding — the textbook signature of a
/// boundary-only (singular) witnessing Gram matrix, which is exactly what
/// makes both plain alternating projection and Douglas–Rachford converge
/// only asymptotically rather than in finitely many steps (see the module
/// doc and `Family::douglas_rachford_from`'s doc comment).
///
/// The fix, rather than searching harder in the same (degenerate) family, is
/// to search a *smaller* one: read the near-null eigenvector directions off
/// the best numeric candidates found so far (`smallest_magnitude_eigenvectors`),
/// round each to an exact rational, and impose it as an exact `Q(t)·v = 0`
/// constraint. If the guess is right, this restricts the family to (an
/// affine reparametrisation of) the face of the PSD cone the true
/// certificate actually lives on, where the intersection with the reduced
/// family is far less degenerate — and the same search-then-round-then-check
/// machinery in [`search_rational_family`] is run again on that smaller
/// family, this time with a real chance of landing exactly on it. If the
/// guess is wrong, `restrict_by_near_null_vectors` (or the reduced family's
/// own exact check) simply comes back empty and the next guess is tried —
/// every path through this function is either an exact rational identity or
/// nothing, never a fabricated certificate.
#[allow(clippy::too_many_arguments)]
fn facial_reduction_search(
    nvars: usize,
    basis: &[Exponents],
    target: &RatPoly,
    base_rat: &[Vec<Rational>],
    dirs_rat: &[Vec<Vec<Rational>>],
    candidates: &[Vec<Vec<f64>>],
    log: &mut Vec<String>,
) -> Option<SosPoly> {
    let max_corank = *FACIAL_CORANK_GUESSES.iter().max().unwrap_or(&0);
    let mut budget = FACIAL_SEARCH_BUDGET;
    for q_best in candidates.iter().take(FACIAL_CANDIDATES) {
        let near_null = smallest_magnitude_eigenvectors(q_best, max_corank);
        for &corank in FACIAL_CORANK_GUESSES {
            if budget == 0 {
                return None;
            }
            if corank > near_null.len() {
                continue;
            }
            for &max_den in FACIAL_DENOM_CAPS {
                let vs: Option<Vec<Vec<Rational>>> = near_null[..corank]
                    .iter()
                    .map(|v| {
                        v.iter()
                            .map(|&x| round_to_rational(x, max_den))
                            .collect::<Option<Vec<_>>>()
                    })
                    .collect();
                let Some(vs) = vs else { continue };
                let Some((new_base, new_dirs)) =
                    restrict_by_near_null_vectors(base_rat, dirs_rat, &vs)
                else {
                    continue;
                };
                if new_dirs.len() >= dirs_rat.len() {
                    // The guess added no real constraint (rare, but possible
                    // if the rounded vector happens to be in the common
                    // nullspace of every direction) — searching the
                    // "reduced" family again would just repeat work.
                    continue;
                }
                if budget == 0 {
                    return None;
                }
                budget -= 1;
                let (found, _deeper) =
                    search_rational_family(nvars, basis, target, &new_base, &new_dirs, log);
                if found.is_some() {
                    return found;
                }
            }
        }
    }
    None
}

/// Inverse of [`unpack`]: the packed upper triangle (`i ≤ j`, row-major) of
/// an `n×n` symmetric matrix, in the same order [`gram_system`] uses.
fn pack(n: usize, m: &[Vec<Rational>]) -> Vec<Rational> {
    let mut v = Vec::with_capacity(pack_len(n));
    for i in 0..n {
        for j in i..n {
            v.push(m[i][j].clone());
        }
    }
    v
}

/// Above this many variables, [`detect_polynomial_symmetry_group`] is
/// skipped rather than run: it enumerates `nvars! · 2^nvars` candidate
/// signed permutations, and this keeps that bounded (`6! · 2^6 = 46080` at
/// the cap, still fast) regardless of how many variables a caller passes.
/// Skipping just means [`symmetry_reduced_search`] finds no symmetry to
/// exploit and falls through to `None`, exactly like every other budget in
/// this module.
const MAX_SYMMETRY_NVARS: usize = 6;

/// All permutations of `0..nvars` (Heap's algorithm via simple recursive
/// swaps). Only ever called with `nvars ≤ `[`MAX_SYMMETRY_NVARS`], so the
/// `nvars!` output size stays small.
fn permutations(nvars: usize) -> Vec<Vec<usize>> {
    fn helper(items: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
        if k == items.len() {
            out.push(items.clone());
            return;
        }
        for i in k..items.len() {
            items.swap(k, i);
            helper(items, k + 1, out);
            items.swap(k, i);
        }
    }
    let mut items: Vec<usize> = (0..nvars).collect();
    let mut out = Vec::new();
    helper(&mut items, 0, &mut out);
    out
}

/// The image of monomial `e` under the substitution `x_i ↦ signs[i]·x_{perm[i]}`
/// — the new exponent vector, and the overall sign picked up (`-1` exactly
/// when an odd number of `signs[i] = -1` land on an odd `e[i]`).
fn transform_exponents(e: &Exponents, perm: &[usize], signs: &[i8]) -> (Exponents, i8) {
    let mut e2 = vec![0u32; e.len()];
    let mut sign = 1i8;
    for i in 0..e.len() {
        e2[perm[i]] = e[i];
        if signs[i] < 0 && e[i] % 2 == 1 {
            sign = -sign;
        }
    }
    (e2, sign)
}

/// Whether the substitution `x_i ↦ signs[i]·x_{perm[i]}` leaves `target`
/// exactly fixed (every term maps to a term with the transformed coefficient,
/// checked exactly over ℚ).
fn symmetry_fixes_target(target: &RatPoly, perm: &[usize], signs: &[i8]) -> bool {
    for (e, c) in target.terms() {
        let (e2, sign) = transform_exponents(e, perm, signs);
        let expected = if sign < 0 { -c.clone() } else { c.clone() };
        if target.coeff(&e2) != expected {
            return false;
        }
    }
    true
}

/// The (signed-permutation) symmetry group of `target`: every substitution
/// `x_i ↦ ±x_{perm(i)}` (a variable permutation composed with independent
/// sign flips — the full hyperoctahedral group on `nvars` variables) that
/// leaves `target` exactly fixed. Always contains at least the identity.
/// Exhaustive rather than clever, because `nvars` is small in every case this
/// is used for (see [`MAX_SYMMETRY_NVARS`]) — above the cap, only the
/// identity is returned, which downstream treats as "no usable symmetry"
/// rather than a bug.
fn detect_polynomial_symmetry_group(target: &RatPoly, nvars: usize) -> Vec<(Vec<usize>, Vec<i8>)> {
    if nvars == 0 || nvars > MAX_SYMMETRY_NVARS {
        return vec![((0..nvars).collect(), vec![1i8; nvars])];
    }
    let mut group = Vec::new();
    for perm in permutations(nvars) {
        for mask in 0..(1u32 << nvars) {
            let signs: Vec<i8> = (0..nvars)
                .map(|i| if (mask >> i) & 1 == 1 { -1 } else { 1 })
                .collect();
            if symmetry_fixes_target(target, &perm, &signs) {
                group.push((perm.clone(), signs));
            }
        }
    }
    group
}

/// How a single symmetry-group element permutes the *basis* (rather than
/// the original variables): for each basis index `i`, `(j, sign)` such that
/// the substitution sends basis monomial `i` to `sign · basis[j]`. The
/// monomial basis built by [`monomial_basis`] (optionally filtered to a
/// single homogeneous degree) is always closed under any variable
/// permutation or sign flip — it is cut out purely by total degree — so the
/// lookup below always succeeds.
fn signed_permutation_action(
    basis: &[Exponents],
    perm: &[usize],
    signs: &[i8],
) -> Vec<(usize, i8)> {
    basis
        .iter()
        .map(|e| {
            let (e2, sign) = transform_exponents(e, perm, signs);
            let j = basis
                .iter()
                .position(|b| *b == e2)
                .expect("basis is closed under variable permutation/sign flip");
            (j, sign)
        })
        .collect()
}

/// Conjugate `m` by the signed permutation `action` (`action[i] = (j, s)`
/// means basis index `i` maps to `j` with sign `s`): the matrix `m'` with
/// `m'[j1][j2] = s1·s2·m[i1][i2]` for every `i1, i2`. Since `action` is a
/// bijection, every `(j1, j2)` is hit exactly once, so this can assign
/// directly rather than accumulate.
fn conjugate_by_action(m: &[Vec<Rational>], action: &[(usize, i8)]) -> Vec<Vec<Rational>> {
    let n = m.len();
    let mut out = vec![vec![Rational::from(0); n]; n];
    for (i1, &(j1, s1)) in action.iter().enumerate() {
        for (i2, &(j2, s2)) in action.iter().enumerate() {
            if m[i1][i2] == 0 {
                continue;
            }
            out[j1][j2] = if s1 * s2 < 0 {
                -m[i1][i2].clone()
            } else {
                m[i1][i2].clone()
            };
        }
    }
    out
}

/// The `G`-average `(1/|G|)·Σ_g P_g·m·P_gᵀ` of `m` over the group actions in
/// `actions` — exact rational arithmetic throughout, since `|actions|` is a
/// plain integer divisor. The result is fixed by every action in the group
/// (a matrix in the *symmetric* subspace) and, when `m` is itself a member
/// of a family whose every point reproduces `target` (any `search_rational_family`
/// input is), so is this average: `target`'s own coefficients are invariant
/// under the same substitutions (that is exactly what
/// [`detect_polynomial_symmetry_group`] checked), so averaging conjugates of
/// valid solutions is still a valid solution — never a fabricated one.
fn symmetrize_matrix(m: &[Vec<Rational>], actions: &[Vec<(usize, i8)>]) -> Vec<Vec<Rational>> {
    let n = m.len();
    let mut acc = vec![vec![Rational::from(0); n]; n];
    for action in actions {
        let c = conjugate_by_action(m, action);
        for i in 0..n {
            for j in 0..n {
                if c[i][j] != 0 {
                    acc[i][j] += &c[i][j];
                }
            }
        }
    }
    let g = Rational::from(actions.len() as i64);
    for row in acc.iter_mut() {
        for v in row.iter_mut() {
            if *v != 0 {
                *v /= &g;
            }
        }
    }
    acc
}

/// A linearly independent basis of the span of `rows`, by Gauss–Jordan
/// elimination (same style as [`solve_affine`]'s core loop, without the
/// right-hand side): the nonzero rows of the row-reduced echelon form.
/// [`symmetry_reduced_search`] uses this to collapse the (typically highly
/// redundant) `Σ_g P_g·dir_k·P_gᵀ` images down to the actual dimension of the
/// symmetric subspace, which is what makes the reduced family cheaper to
/// search rather than merely a relabelling of the original one.
fn row_reduce_basis(mut rows: Vec<Vec<Rational>>) -> Vec<Vec<Rational>> {
    let Some(ncols) = rows.first().map(|r| r.len()) else {
        return Vec::new();
    };
    let m = rows.len();
    let mut r = 0usize;
    for c in 0..ncols {
        let Some(p) = (r..m).find(|&i| rows[i][c] != 0) else {
            continue;
        };
        rows.swap(r, p);
        let inv = rows[r][c].clone();
        for v in rows[r].iter_mut() {
            *v /= &inv;
        }
        let prow = rows[r].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i == r || row[c] == 0 {
                continue;
            }
            let f = row[c].clone();
            for (t, pv) in row.iter_mut().zip(prow.iter()) {
                *t -= Rational::from(&f * pv);
            }
        }
        r += 1;
        if r == m {
            break;
        }
    }
    rows.truncate(r);
    rows
}

/// Above this many free parameters in the *original* (unreduced) affine
/// family, the plain search and facial reduction have already had a real
/// chance at [`psd_search`]'s call site and failed; below it there is no
/// reason to pay the extra symmetry-detection cost. Chosen well above the
/// affine (2-variable) Motzkin and Robinson families (75 and 84 free
/// parameters respectively) that the earlier steps already close on their
/// own, so this fallback only ever runs on families those steps already gave
/// up on — never adding search cost to a case that already succeeds.
const MIN_PARAMS_FOR_SYMMETRY_SEARCH: usize = 100;

/// Symmetry-reduction fallback for when both the plain numeric search
/// ([`search_rational_family`]) and facial reduction give up: restrict the
/// affine family to the subspace of Gram matrices fixed by `target`'s own
/// signed-permutation symmetry group (variable permutations composed with
/// independent sign flips — [`detect_polynomial_symmetry_group`]), and search
/// *that* (typically much smaller) family instead.
///
/// This is sound for the same reason facial reduction is: every matrix in
/// the reduced family is an exact ℚ-affine combination of matrices in the
/// original family (a `G`-average, via [`symmetrize_matrix`]), so it still
/// reproduces `target` exactly by construction, and the same
/// search-then-round-then-check machinery in [`search_rational_family`] (and
/// [`facial_reduction_search`] on top of it) does the rest. If `target` has
/// no usable symmetry, or the reduction does not actually shrink the
/// parameter count, this simply returns `None` — a routine "nothing to
/// exploit here", not a bug.
///
/// The motivating case was the *homogeneous* ternary Motzkin form at
/// multiplier power `N = 2`: 165 free parameters, all-even exponents in
/// every monomial (invariant under any sign flip) and symmetric under
/// swapping `x, y` — an order-16 group — which collapses the family to 26
/// parameters (see `tests::symmetry_group_and_zero_vector_shrink_n2_family`,
/// which asserts that shrink exactly). **That motivation was based on a
/// false premise** and is recorded here only so it is not re-derived: the
/// homogeneous ternary Motzkin form is SOS at `N = 1`, not `N = 2`, and it
/// is `half_newton_reduce` — a plain dimension reduction — that closes it,
/// not any amount of extra Douglas–Rachford. This fallback is *not* known to
/// close any case on its own; it is retained because it is cheap, sound, and
/// only ever runs after everything else has already given up.
fn symmetry_reduced_search(
    nvars: usize,
    basis: &[Exponents],
    target: &RatPoly,
    base_rat: &[Vec<Rational>],
    dirs_rat: &[Vec<Vec<Rational>>],
    log: &mut Vec<String>,
) -> Option<SosPoly> {
    if nvars == 0 || nvars > MAX_SYMMETRY_NVARS || dirs_rat.len() < MIN_PARAMS_FOR_SYMMETRY_SEARCH {
        return None;
    }
    let group = detect_polynomial_symmetry_group(target, nvars);
    if group.len() <= 1 {
        return None;
    }
    let actions: Vec<Vec<(usize, i8)>> = group
        .iter()
        .map(|(perm, signs)| signed_permutation_action(basis, perm, signs))
        .collect();

    let n = basis.len();
    let sym_base = symmetrize_matrix(base_rat, &actions);
    let sym_dirs_packed: Vec<Vec<Rational>> = dirs_rat
        .iter()
        .map(|d| pack(n, &symmetrize_matrix(d, &actions)))
        .collect();
    let reduced_packed = row_reduce_basis(sym_dirs_packed);
    if reduced_packed.len() >= dirs_rat.len() {
        // The symmetry did not actually cut the parameter count down (rare,
        // but possible for a group too small or too trivially-acting to
        // help) — nothing gained by searching the "reduced" family again.
        return None;
    }
    let sym_dirs: Vec<Vec<Vec<Rational>>> = reduced_packed.iter().map(|v| unpack(n, v)).collect();

    let (found, sym_candidates) =
        search_rational_family(nvars, basis, target, &sym_base, &sym_dirs, log);
    if found.is_some() {
        return found;
    }
    if sym_candidates.is_empty() {
        return None;
    }
    facial_reduction_search(
        nvars,
        basis,
        target,
        &sym_base,
        &sym_dirs,
        &sym_candidates,
        log,
    )
}

/// [`psd_search`], but also appending a human-readable trace of what the
/// search actually did to `log`.
///
/// The trace exists because `None` out of this module covers three
/// materially different situations — the search ran and was exhausted, the
/// search ran and hit an iteration/rounding budget, and *the search never
/// ran at all* because the family was over `MAX_FREE_PARAMETERS` — and a
/// bare `None` (or the `E-SOS-002` it turns into) cannot distinguish them.
/// [`super::sos_decompose`] folds this trace into the error message so that
/// distinction reaches the caller.
pub fn psd_search_logged(
    target: &RatPoly,
    basis_deg: u32,
    log: &mut Vec<String>,
) -> Option<SosPoly> {
    let nvars = target.nvars();
    let (basis, homogeneous_len) = restricted_basis(target, basis_deg);
    if basis.len() < homogeneous_len {
        log.push(format!(
            "half-Newton-polytope reduction: {homogeneous_len} → {} monomials in the Gram basis",
            basis.len()
        ));
    }
    let n = basis.len();
    if n == 0 {
        return if target.is_zero() {
            Some(SosPoly::default())
        } else {
            None
        };
    }

    // Apply the free-parameter ceiling *before* the exact nullspace solve,
    // not after it: `solve_affine` is a rational Gauss–Jordan on a
    // `rows × pack_len(n)` matrix and on a family this large costs far more
    // than the search that is then skipped anyway (C₇'s N=1 multiplier is a
    // 924 × 3570 exact solve, thrown away immediately). The rank of the
    // coefficient-matching system is at most its row count, so
    // `pack_len(n) − rows` is a *lower* bound on the family's dimension —
    // when even that exceeds the ceiling the search provably cannot run.
    let min_free = pack_len(n).saturating_sub(row_count(target, &basis));
    if min_free > MAX_FREE_PARAMETERS {
        log.push(ceiling_note(n, min_free, true));
        return None;
    }

    let (rows, rhs) = gram_system(target, &basis);
    let sol = solve_affine(&rows, &rhs)?;

    let base_rat = unpack(n, &sol.particular);
    let dirs_rat: Vec<Vec<Vec<Rational>>> = sol.nullspace.iter().map(|d| unpack(n, d)).collect();

    let (found, candidates) =
        search_rational_family(nvars, &basis, target, &base_rat, &dirs_rat, log);
    if found.is_some() {
        return found;
    }
    if candidates.is_empty() {
        return None;
    }
    if let Some(found) = facial_reduction_search(
        nvars,
        &basis,
        target,
        &base_rat,
        &dirs_rat,
        &candidates,
        log,
    ) {
        return Some(found);
    }
    symmetry_reduced_search(nvars, &basis, target, &base_rat, &dirs_rat, log)
}

/// Search the full PSD-Gram cone for an exact rational sum-of-squares
/// decomposition of `target`, discarding the diagnostic trace. See
/// [`psd_search_logged`] for the variant that keeps it.
pub fn psd_search(target: &RatPoly, basis_deg: u32) -> Option<SosPoly> {
    let mut log = Vec::new();
    psd_search_logged(target, basis_deg, &mut log)
}

/// The monomial basis [`psd_search_logged`] will actually search for
/// `target` at `basis_deg`, plus the size it had before the
/// half-Newton-polytope reduction.
///
/// A homogeneous target of degree exactly `2·basis_deg` needs only the
/// monomials of degree *exactly* `basis_deg` in its Gram basis — mixing in
/// lower-degree monomials can only ever contribute to coefficients the
/// target does not have, since every product of two basis monomials of
/// unequal degree still sums to `2·basis_deg` only when *both* already have
/// degree `basis_deg`. This is standard (Blekherman–Parrilo–Thomas,
/// Prop. 3.29): a homogeneous SOS decomposition can always be taken with
/// homogeneous summands. Then [`half_newton_reduce`] drops whatever survives
/// that but still lies outside `½·Newton(target)`. Neither step is only an
/// optimisation — the search is numeric, and the parameter count (quadratic
/// in the basis size) is the difference between "rounds onto the exact
/// certificate" and "lands 0.96 away from it".
fn restricted_basis(target: &RatPoly, basis_deg: u32) -> (Vec<Exponents>, usize) {
    let nvars = target.nvars();
    let basis: Vec<Exponents> = match target.is_homogeneous() {
        Some(d) if d == 2 * basis_deg => monomial_basis(nvars, basis_deg)
            .into_iter()
            .filter(|e| e.iter().sum::<u32>() == basis_deg)
            .collect(),
        _ => monomial_basis(nvars, basis_deg),
    };
    let homogeneous_len = basis.len();
    (half_newton_reduce(target, basis), homogeneous_len)
}

/// Size of the monomial basis [`psd_search_logged`] will actually search for
/// `target` at `basis_deg`.
///
/// Exposed so `multiplier_search` can budget against the size it is
/// really going to search rather than the raw `monomial_basis` count: those
/// two differ by nearly 4× on realistic targets (a degree-4 ternary basis is
/// 35 monomials raw, 15 after the homogeneity restriction, 9 after the
/// Newton reduction on `σ·Motzkin`), and budgeting against the raw number
/// rejects multiplier powers whose real basis is comfortably inside the
/// budget.
pub fn searched_basis_len(target: &RatPoly, basis_deg: u32) -> usize {
    restricted_basis(target, basis_deg).0.len()
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
    fn psd_search_certifies_homogeneous_motzkin_times_sum_of_squares_at_n1() {
        // The homogeneous Motzkin form, x^4y^2 + x^2y^4 - 3x^2y^2z^2 + z^6, is
        // the textbook example of a PSD form that is not itself SOS — and it
        // is the textbook example precisely *because* multiplying it by
        // σ = x^2+y^2+z^2 makes it SOS. The classical identity, exact over ℚ:
        //
        //   σ·Motzkin = (½x³y + xy³ − 3⁄2xyz²)² + ¾(x³y − xyz²)²
        //             + (xy²z − xz³)² + (x²yz − yz³)² + (x²y² − z⁴)²
        //
        // so `N = 1` suffices. This test used to be its own negation —
        // `psd_search_does_not_yet_reach_...`, asserting `is_none()` and
        // *passing*, with a comment claiming the classical fact needs N=2 and
        // that "this N=1 refusal is expected to stay `None` permanently". That
        // claim was simply false (the identity above expands to zero
        // difference; re-check it in any CAS), and the green test pinned the
        // search bug that produced the refusal. It is stated in the positive
        // direction now so that a regression is a failure rather than a
        // confirmation.
        //
        // What was actually missing was the half-Newton-polytope reduction
        // (`half_newton_reduce`): σ·Motzkin's degree-4 ternary basis is 15
        // monomials, only 9 of which lie in ½·Newton(σ·Motzkin), and the
        // 75-parameter family over the unreduced basis puts the numeric
        // search ~0.96 away from the true point in parameter space — far too
        // far to round onto it — while the 18-parameter family over the
        // reduced basis lands on it exactly. Note this is a *dimension*
        // effect, not a conditioning one: on both bases the certificate is
        // the unique PSD point of the affine family, rank 5, with minimum
        // eigenvalue exactly 0, so λ_min does not distinguish them and more
        // Douglas–Rachford iterations do not help (4× the budget on the
        // unreduced family still fails to round).
        let mut m = RatPoly::monomial(3, vec![4, 2, 0], Rational::from(1));
        m = m.add(&RatPoly::monomial(3, vec![2, 4, 0], Rational::from(1)));
        m = m.add(&RatPoly::monomial(3, vec![2, 2, 2], Rational::from(-3)));
        m = m.add(&RatPoly::monomial(3, vec![0, 0, 6], Rational::from(1)));
        assert_eq!(m.is_homogeneous(), Some(6));
        let sigma = RatPoly::sum_of_squares(3);
        let q = m.mul(&sigma);
        assert_eq!(q.is_homogeneous(), Some(8));

        // The reduction itself, asserted rather than assumed: 15 → 9.
        let (reduced, homogeneous_len) = restricted_basis(&q, 4);
        assert_eq!(homogeneous_len, 15);
        assert_eq!(reduced.len(), 9);

        let sos = psd_search(&q, 4).expect(
            "(x^2+y^2+z^2)·Motzkin_hom is a sum of squares at N=1 — the classical fact that \
             makes Motzkin the standard PSD-not-SOS example — so the search must find a \
             certificate here",
        );
        // Exact re-expansion over ℚ: the soundness argument, independent of
        // whatever the numeric search converged to.
        assert_eq!(sos.to_poly(3), q);
    }

    #[test]
    fn psd_search_certifies_choi_lam_times_sum_of_squares_at_n1() {
        // The Choi–Lam form, x²y² + y²z² + z²x² + w⁴ − 4xyzw: a quaternary
        // quartic that is PSD but not SOS, and (like Motzkin) becomes SOS
        // after one multiplication by σ = Σxᵢ². Four variables, so the
        // degree-3 basis is 20 monomials before the Newton reduction and 16
        // after; the corresponding drop in free parameters is what makes the
        // search land on the certificate.
        let mono = |ex: Vec<u32>, c: i64| RatPoly::monomial(4, ex, Rational::from(c));
        let mut cl = mono(vec![2, 2, 0, 0], 1);
        cl = cl.add(&mono(vec![0, 2, 2, 0], 1));
        cl = cl.add(&mono(vec![2, 0, 2, 0], 1));
        cl = cl.add(&mono(vec![0, 0, 0, 4], 1));
        cl = cl.add(&mono(vec![1, 1, 1, 1], -4));
        assert_eq!(cl.is_homogeneous(), Some(4));

        let q = cl.mul(&RatPoly::sum_of_squares(4));
        assert_eq!(q.is_homogeneous(), Some(6));
        let sos = psd_search(&q, 3)
            .expect("(Σxᵢ²)·Choi–Lam is SOS at N=1; the search should find a certificate");
        assert_eq!(sos.to_poly(4), q);
    }

    #[test]
    fn half_newton_reduction_is_a_no_op_when_the_polytope_is_already_full() {
        // Robinson's form times σ is the guard case for `half_newton_reduce`:
        // its Newton polytope fills the degree-8 simplex, so *no* monomial of
        // the degree-4 basis is outside ½·Newton, and the reduction must
        // return all 15 — a reduction that trimmed anything here would be
        // dropping monomials a real certificate needs.
        let mono = |ex: Vec<u32>, c: i64| RatPoly::monomial(3, ex, Rational::from(c));
        let mut r = mono(vec![6, 0, 0], 1);
        r = r.add(&mono(vec![0, 6, 0], 1));
        r = r.add(&mono(vec![0, 0, 6], 1));
        r = r.add(&mono(vec![4, 2, 0], -1));
        r = r.add(&mono(vec![2, 4, 0], -1));
        r = r.add(&mono(vec![0, 4, 2], -1));
        r = r.add(&mono(vec![0, 2, 4], -1));
        r = r.add(&mono(vec![4, 0, 2], -1));
        r = r.add(&mono(vec![2, 0, 4], -1));
        r = r.add(&mono(vec![2, 2, 2], 3));
        let q = r.mul(&RatPoly::sum_of_squares(3));
        let (reduced, homogeneous_len) = restricted_basis(&q, 4);
        assert_eq!(homogeneous_len, 15);
        assert_eq!(reduced.len(), 15, "Robinson admits no Newton reduction");

        // Same check on the direct (unmultiplied) sextic, and on a target
        // where the reduction genuinely bites, so this test pins both
        // directions rather than only the no-op one.
        let (reduced_direct, direct_len) = restricted_basis(&r, 3);
        assert_eq!((direct_len, reduced_direct.len()), (10, 10));

        // The cyclic AM-GM sextic x⁴y²+y⁴z²+z⁴x²−3x²y²z², times σ: the same
        // 15 → 9 profile as Motzkin.
        let mut c = mono(vec![4, 2, 0], 1);
        c = c.add(&mono(vec![0, 4, 2], 1));
        c = c.add(&mono(vec![2, 0, 4], 1));
        c = c.add(&mono(vec![2, 2, 2], -3));
        let (reduced_cyclic, cyclic_len) = restricted_basis(&c.mul(&RatPoly::sum_of_squares(3)), 4);
        assert_eq!((cyclic_len, reduced_cyclic.len()), (15, 9));
    }

    #[test]
    fn the_free_parameter_ceiling_is_reported_not_silently_dropped() {
        // x⁸ + y⁸ + z⁸ + 1 is a sum of squares by inspection. Searched over
        // the degree-4 basis its affine Gram family still has 465 free
        // parameters — above `MAX_FREE_PARAMETERS` — so `psd_search` gives up
        // *without running any search at all*. That is a legitimate budget,
        // but it used to be indistinguishable from an exhausted search: the
        // same instant `None`, and nothing recorded. The point of this test
        // is not the `None` (which is unchanged behaviour) but that the
        // ceiling is now *reported*.
        let mono = |ex: Vec<u32>, c: i64| RatPoly::monomial(3, ex, Rational::from(c));
        let mut p = mono(vec![8, 0, 0], 1);
        p = p.add(&mono(vec![0, 8, 0], 1));
        p = p.add(&mono(vec![0, 0, 8], 1));
        p = p.add(&mono(vec![0, 0, 0], 1));

        let mut log = Vec::new();
        assert!(psd_search_logged(&p, 4, &mut log).is_none());
        let ceiling = log
            .iter()
            .find(|l| l.contains("NOT SEARCHED"))
            .unwrap_or_else(|| panic!("the free-parameter ceiling must be logged; got {log:?}"));
        assert!(
            ceiling.contains("465") && ceiling.contains(&MAX_FREE_PARAMETERS.to_string()),
            "the ceiling report must name both the family size and the ceiling: {ceiling}"
        );
        assert!(
            !log.iter().any(|l| l.contains("searching a")),
            "nothing was searched, so nothing may claim to have searched: {log:?}"
        );
    }

    #[test]
    fn a_search_that_really_runs_says_so() {
        // The counterpart to the test above: when a search does run, the log
        // says *searched*, so the two outcomes are distinguishable in the
        // trace and not only in the (identical) `None`/`Some`.
        let mut p = RatPoly::monomial(2, vec![4, 0], Rational::from(1));
        p = p.add(&RatPoly::monomial(2, vec![0, 4], Rational::from(1)));
        p = p.add(&RatPoly::monomial(2, vec![2, 2], Rational::from(2)));
        let mut log = Vec::new();
        assert!(psd_search_logged(&p, 2, &mut log).is_some());
        assert!(
            log.iter().any(|l| l.contains("searching a")),
            "a search that runs must record that it ran: {log:?}"
        );
        assert!(!log.iter().any(|l| l.contains("NOT SEARCHED")));
    }

    /// Fast, exact-arithmetic-only regression test for the
    /// `symmetry_reduced_search` machinery: documents the 165 → 26 → 16
    /// free-parameter shrink referenced in this module's doc comments, with
    /// real assertions rather than `eprintln!` probes. Deliberately does
    /// *not* call the full `psd_search` (which at `N = 2` spends several
    /// minutes on Douglas–Rachford before giving up) so this stays fast
    /// enough to run on every `cargo test`.
    ///
    /// The `N = 2` family is exercised here purely because it is a large,
    /// highly symmetric family that makes the reduction machinery easy to
    /// assert on. It is **not** the multiplier power this target needs — the
    /// homogeneous ternary Motzkin form is SOS at `N = 1`, and
    /// `psd_search_certifies_homogeneous_motzkin_times_sum_of_squares_at_n1`
    /// is where that is checked.
    #[test]
    fn symmetry_group_and_zero_vector_shrink_n2_family() {
        let mut m = RatPoly::monomial(3, vec![4, 2, 0], Rational::from(1));
        m = m.add(&RatPoly::monomial(3, vec![2, 4, 0], Rational::from(1)));
        m = m.add(&RatPoly::monomial(3, vec![2, 2, 2], Rational::from(-3)));
        m = m.add(&RatPoly::monomial(3, vec![0, 0, 6], Rational::from(1)));
        let sigma2 = RatPoly::sum_of_squares(3).pow(2);
        let q = m.mul(&sigma2);
        assert_eq!(q.is_homogeneous(), Some(10));
        assert_eq!(m.eval(&[r(1, 1), r(1, 1), r(1, 1)]), Rational::from(0));
        assert_eq!(q.eval(&[r(1, 1), r(1, 1), r(1, 1)]), Rational::from(0));

        let basis: Vec<Exponents> = monomial_basis(3, 5)
            .into_iter()
            .filter(|e| e.iter().sum::<u32>() == 5)
            .collect();
        let n = basis.len();
        let (rows, rhs) = gram_system(&q, &basis);
        let sol = solve_affine(&rows, &rhs).expect("consistent");
        assert_eq!(sol.dimension(), 165);

        let group = detect_polynomial_symmetry_group(&q, 3);
        assert_eq!(
            group.len(),
            16,
            "expected the order-16 group: swap x,y composed with independent sign flips \
             of x, y, z (every exponent in q is even)"
        );

        let base_rat = unpack(n, &sol.particular);
        let dirs_rat: Vec<Vec<Vec<Rational>>> =
            sol.nullspace.iter().map(|d| unpack(n, d)).collect();
        let actions: Vec<Vec<(usize, i8)>> = group
            .iter()
            .map(|(perm, signs)| signed_permutation_action(&basis, perm, signs))
            .collect();
        let sym_base = symmetrize_matrix(&base_rat, &actions);
        let sym_dirs_packed: Vec<Vec<Rational>> = dirs_rat
            .iter()
            .map(|d| pack(n, &symmetrize_matrix(d, &actions)))
            .collect();
        let sym_dirs: Vec<Vec<Vec<Rational>>> = row_reduce_basis(sym_dirs_packed)
            .iter()
            .map(|v| unpack(n, v))
            .collect();
        assert_eq!(sym_dirs.len(), 26);

        // The symmetrized base point must still reproduce q exactly — the
        // actual soundness argument for symmetry reduction, not just a claim.
        let mut quad = RatPoly::zero(3);
        for i in 0..n {
            for j in 0..n {
                let e = add_exp(&basis[i], &basis[j]);
                quad = quad.add(&RatPoly::monomial(3, e, sym_base[i][j].clone()));
            }
        }
        assert_eq!(quad, q);

        // z(1,1,1) is the literal all-ones vector (every degree-5 monomial
        // evaluates to 1 there): q(1,1,1)=0 with Q PSD forces Q·z(1,1,1)=0
        // exactly, no numerics involved.
        let v_ones: Vec<Rational> = vec![Rational::from(1); n];
        let (restricted_base, restricted_dirs) =
            restrict_by_near_null_vectors(&sym_base, &sym_dirs, &[v_ones])
                .expect("z(1,1,1)=0 must be consistent with the symmetric family");
        assert_eq!(restricted_dirs.len(), 16);
        let mut quad2 = RatPoly::zero(3);
        for i in 0..n {
            for j in 0..n {
                let e = add_exp(&basis[i], &basis[j]);
                quad2 = quad2.add(&RatPoly::monomial(3, e, restricted_base[i][j].clone()));
            }
        }
        assert_eq!(quad2, q);

        // The extra tangent-direction candidate (the exact gradient of the
        // basis vector at (1,1,1) in the x direction) is genuinely
        // *inconsistent* with the family — confirming the corank
        // contributed by this zero really is 1, not a missed higher-corank
        // guess (a wrong-but-silently-accepted guess would be a bug; this
        // asserts the guess is correctly rejected instead).
        let v_grad_x: Vec<Rational> = basis.iter().map(|e| Rational::from(e[0])).collect();
        assert!(
            restrict_by_near_null_vectors(
                &sym_base,
                &sym_dirs,
                &[vec![Rational::from(1); n], v_grad_x]
            )
            .is_none(),
            "grad_x should NOT be a valid null direction of the true certificate"
        );
    }

    #[test]
    fn psd_search_certifies_robinsons_form_with_a_reznick_multiplier() {
        // Robinson's form: x^6+y^6+z^6 - (x^4y^2+x^2y^4+y^4z^2+y^2z^4+x^4z^2+x^2z^4) + 3x^2y^2z^2.
        // A second textbook PSD-not-SOS example (distinct from Motzkin) whose
        // multiplier certificate the Douglas-Rachford / facial-reduction
        // search below now reaches. Direct search (no multiplier) still
        // fails, as it must: Robinson's form is genuinely not SOS itself.
        let mono = |ex: Vec<u32>, c: i64| RatPoly::monomial(3, ex, Rational::from(c));
        let mut r = mono(vec![6, 0, 0], 1);
        r = r.add(&mono(vec![0, 6, 0], 1));
        r = r.add(&mono(vec![0, 0, 6], 1));
        r = r.add(&mono(vec![4, 2, 0], -1));
        r = r.add(&mono(vec![2, 4, 0], -1));
        r = r.add(&mono(vec![0, 4, 2], -1));
        r = r.add(&mono(vec![0, 2, 4], -1));
        r = r.add(&mono(vec![4, 0, 2], -1));
        r = r.add(&mono(vec![2, 0, 4], -1));
        r = r.add(&mono(vec![2, 2, 2], 3));
        assert_eq!(r.is_homogeneous(), Some(6));

        assert!(
            psd_search(&r, 3).is_none(),
            "Robinson's form is not itself SOS, so a direct (unmultiplied) search must refuse"
        );

        let sigma = RatPoly::sum_of_squares(3);
        let q = r.mul(&sigma);
        assert_eq!(q.is_homogeneous(), Some(8));
        let sos = psd_search(&q, 4)
            .expect("(x^2+y^2+z^2)*Robinson is a classical SOS example; the search should find it");
        // Exact re-expansion, over the rationals — the actual soundness
        // argument, not the numeric search that proposed it.
        assert_eq!(sos.to_poly(3), q);
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
