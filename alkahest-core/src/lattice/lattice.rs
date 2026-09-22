//! The [`Lattice`] type: a rank-`m` lattice given by a basis or by a Gram
//! matrix, with the invariants a packing/coding calculation asks for.
//!
//! # What a `Lattice` is here
//!
//! A `Lattice` is carried by its **Gram matrix** `G[i][j] = ⟨b_i, b_j⟩`, an
//! exact rational symmetric positive-definite matrix. A basis, when one was
//! supplied, is kept alongside it so that answers can be reported in ambient
//! coordinates; it is not required, because every quantity below — determinant,
//! minimum, kissing number, theta series, density, dual — is an invariant of
//! the Gram matrix alone. That is what lets `E_8` and the Leech lattice be
//! first-class here: neither has a basis with rational coordinates in its
//! natural embedding, and both have perfectly ordinary rational Gram matrices.
//!
//! # Scope limits — read these before trusting a result
//!
//! * **SVP, CVP, minimal vectors and theta series are exact enumeration.**
//!   Their cost is exponential in the rank. The rank ceiling is
//!   [`MAX_ENUM_RANK`] = 24 (chosen so the Leech lattice is inside it) and
//!   every enumerating method also takes a node budget. Above the ceiling, or
//!   past the budget, you get [`LatticeGeometryError::RankTooLarge`] or
//!   [`LatticeGeometryError::EnumerationBudget`] — never a heuristic answer. There is
//!   deliberately no "approximate SVP" in this module: a non-shortest vector
//!   returned from a function called `shortest_vector` is the failure mode this
//!   crate exists to avoid. Use [`super::lattice_reduce_rows`] if what you want
//!   is a *short* vector cheaply.
//! * **Rank 24 is a ceiling, not a promise.** A rank-24 lattice with a large
//!   determinant will exhaust a sane budget long before it finishes. The Leech
//!   lattice is enumerable at norm ≤ 4 because it is extremal, not because 24
//!   is easy.
//! * **Theta series need an integral Gram matrix** — otherwise "the number of
//!   vectors of norm `n`" indexes over the wrong set. Non-integral forms are
//!   refused with [`LatticeGeometryError::NonIntegralGram`].
//! * **One enumeration is memoised per `Lattice`, and only one.**
//!   [`Lattice::minimum`], [`Lattice::shortest_vector`],
//!   [`Lattice::kissing_number`], [`Lattice::hermite_invariant`],
//!   [`Lattice::center_density`] and [`Lattice::center_density_exact`] share a
//!   single shortest-vector pass, so asking all six costs what asking one
//!   costs. [`Lattice::minimal_vectors`] and [`Lattice::theta_series`] do not
//!   share it — they need the vectors themselves, not just the count — and
//!   every `*_with_budget` entry point deliberately bypasses the memo, because
//!   passing a budget is a request to do the work under that budget. At rank
//!   24 a pass is tens of seconds, so hold on to the `Lattice` rather than
//!   rebuilding it.
//! * **Densities are `f64`.** `center_density` and `packing_density` return
//!   floating-point numbers because `Δ` involves `π^{m/2}`.
//!   [`Lattice::center_density_exact`] gives the exact rational when the
//!   lattice has one (even rank, square determinant), and returns `None`
//!   rather than a rounded stand-in when it does not.

use super::enumerate::Enumerator;
use super::error::LatticeGeometryError;
use super::lll::lattice_reduce_rows;
use super::lll::LatticeError;
use super::quadform;
use rug::{Integer, Rational};
use std::fmt;
use std::sync::OnceLock;

/// Hard ceiling on the rank of a lattice this module will enumerate.
///
/// 24 so that the Leech lattice is inside it. Enumeration is exponential in the
/// rank; this bound is a refusal threshold, not a performance promise.
pub const MAX_ENUM_RANK: usize = 24;

/// Default Fincke–Pohst node budget. Exceeding it is
/// [`LatticeGeometryError::EnumerationBudget`], never a truncated answer.
pub const DEFAULT_ENUM_NODE_BUDGET: u64 = 50_000_000;

/// Largest norm a theta series will be tabulated up to, to keep the returned
/// vector from being the thing that exhausts memory.
pub const MAX_THETA_NORM: u64 = 1_000_000;

/// A Fincke–Pohst enumerator over an LLL-reduced form, the unimodular `U`
/// taking the lattice's own basis to that form, and the reduced Gram matrix.
type ReducedEnumerator = (Enumerator, Vec<Vec<Integer>>, Vec<Vec<Rational>>);

/// `(λ₁², kissing number, representatives in reduced coordinates, U)`.
type MinimalPass = (Rational, u64, Vec<Vec<Integer>>, Vec<Vec<Integer>>);

/// A lattice vector found by enumeration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeVector {
    /// Coefficients against the lattice's own basis (or Gram rows).
    pub coefficients: Vec<Integer>,
    /// Ambient coordinates, when the lattice was built from a basis.
    pub coordinates: Option<Vec<Rational>>,
    /// The **squared** norm `⟨v, v⟩` (or `⟨v − t, v − t⟩` for a CVP answer).
    pub norm: Rational,
}

/// A lattice of rank `m`, carried by its Gram matrix.
///
/// Holds a memo of one successful default-budget shortest-vector pass, so that
/// [`Lattice::minimum`], [`Lattice::shortest_vector`],
/// [`Lattice::kissing_number`], [`Lattice::hermite_invariant`],
/// [`Lattice::center_density`] and [`Lattice::center_density_exact`] together
/// cost **one** enumeration rather than six. The memo is part of no observable
/// behaviour: it is skipped by every `*_with_budget` entry point, it is not
/// compared by `PartialEq`, it is not printed by `Debug`, and a clone starts
/// cold.
pub struct Lattice {
    basis: Option<Vec<Vec<Rational>>>,
    gram: Vec<Vec<Rational>>,
    /// `None` until a default-budget [`Lattice::minimal_pass`] has *succeeded*.
    /// Failures are never cached, so a refusal stays reproducible.
    minimal: OnceLock<MinimalPass>,
}

impl Clone for Lattice {
    /// The clone starts with a cold memo: copying tens of megabytes of
    /// enumeration results is not what a caller asks for by cloning a lattice.
    fn clone(&self) -> Self {
        Self {
            basis: self.basis.clone(),
            gram: self.gram.clone(),
            minimal: OnceLock::new(),
        }
    }
}

/// Equality of *representation*, not of lattice.
///
/// Two different Gram matrices can present the same lattice in different bases;
/// this does not attempt to decide that (lattice isomorphism is a hard problem
/// and a wrong answer would be worse than no answer). Compare
/// [`Lattice::determinant`], [`Lattice::minimum`] and [`Lattice::theta_series`]
/// if what you want is evidence of isometry.
impl PartialEq for Lattice {
    fn eq(&self, other: &Self) -> bool {
        self.basis == other.basis && self.gram == other.gram
    }
}

impl Eq for Lattice {}

impl fmt::Debug for Lattice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lattice")
            .field("rank", &self.rank())
            .field("ambient_dimension", &self.ambient_dimension())
            .field("gram", &self.gram)
            .field("basis", &self.basis)
            .finish()
    }
}

impl Lattice {
    /// From an integer row basis. Rows must be linearly independent.
    pub fn from_basis(rows: &[Vec<Integer>]) -> Result<Self, LatticeGeometryError> {
        let rational: Vec<Vec<Rational>> = rows
            .iter()
            .map(|r| r.iter().map(Rational::from).collect())
            .collect();
        Self::from_rational_basis(&rational)
    }

    /// From a rational row basis. Rows must be linearly independent.
    pub fn from_rational_basis(rows: &[Vec<Rational>]) -> Result<Self, LatticeGeometryError> {
        if rows.is_empty() {
            return Err(LatticeError::EmptyBasis.into());
        }
        let cols = rows[0].len();
        if cols == 0 {
            return Err(LatticeError::EmptyBasis.into());
        }
        for (i, r) in rows.iter().enumerate() {
            if r.len() != cols {
                return Err(LatticeError::RaggedBasis {
                    row: i,
                    expected_cols: cols,
                    got_cols: r.len(),
                }
                .into());
            }
        }
        let gram = quadform::gram_of_rows(rows);
        // Rejects dependent rows: a dependency makes some ‖b*_i‖² zero.
        quadform::gram_schmidt(&gram)?;
        Ok(Self {
            basis: Some(rows.to_vec()),
            gram,
            minimal: OnceLock::new(),
        })
    }

    /// From a symmetric positive-definite rational Gram matrix.
    pub fn from_gram(g: &[Vec<Rational>]) -> Result<Self, LatticeGeometryError> {
        quadform::validate_gram(g)?;
        quadform::gram_schmidt(g)?;
        Ok(Self {
            basis: None,
            gram: g.to_vec(),
            minimal: OnceLock::new(),
        })
    }

    /// Number of basis vectors.
    pub fn rank(&self) -> usize {
        self.gram.len()
    }

    /// Dimension of the ambient space, when there is one. Equals the rank for a
    /// lattice given only by a Gram matrix.
    pub fn ambient_dimension(&self) -> usize {
        match &self.basis {
            Some(b) => b[0].len(),
            None => self.gram.len(),
        }
    }

    /// `G[i][j] = ⟨b_i, b_j⟩`.
    pub fn gram_matrix(&self) -> &[Vec<Rational>] {
        &self.gram
    }

    /// The basis rows, when the lattice was built from one.
    pub fn basis(&self) -> Option<&[Vec<Rational>]> {
        self.basis.as_deref()
    }

    /// `det G` — the **squared** covolume, exactly. (Also called the lattice
    /// discriminant.)
    pub fn determinant(&self) -> Rational {
        quadform::determinant(&self.gram).expect("Gram matrix validated at construction")
    }

    /// `sqrt(det G)` — the covolume of a fundamental domain, as an `f64`.
    pub fn covolume(&self) -> f64 {
        self.determinant().to_f64().sqrt()
    }

    /// All Gram entries are integers.
    pub fn is_integral(&self) -> bool {
        self.gram
            .iter()
            .all(|r| r.iter().all(|x| *x.denom() == 1i32))
    }

    /// Integral, and every vector has even norm.
    pub fn is_even(&self) -> bool {
        self.is_integral()
            && (0..self.rank()).all(|i| {
                let n = self.gram[i][i].numer().clone();
                n.is_divisible_u(2)
            })
    }

    /// The dual lattice `L* = { y : ⟨y, x⟩ ∈ ℤ for all x ∈ L }`.
    ///
    /// Its Gram matrix is `G⁻¹`; when this lattice has a basis, the dual basis
    /// `G⁻¹ B` is carried along, so `dual().dual()` returns the original
    /// lattice and not merely an isometric copy.
    pub fn dual(&self) -> Result<Self, LatticeGeometryError> {
        let inv = quadform::inverse(&self.gram)?;
        let basis = self.basis.as_ref().map(|b| {
            let m = b.len();
            let n = b[0].len();
            (0..m)
                .map(|i| {
                    (0..n)
                        .map(|c| {
                            let mut acc = Rational::new();
                            for t in 0..m {
                                acc += Rational::from(&inv[i][t] * &b[t][c]);
                            }
                            acc
                        })
                        .collect()
                })
                .collect()
        });
        Ok(Self {
            basis,
            gram: inv,
            minimal: OnceLock::new(),
        })
    }

    // ---------------------------------------------------------------------
    // Enumeration
    // ---------------------------------------------------------------------

    /// Gram–Schmidt data of an LLL-reduced form of this lattice, the
    /// unimodular `U` taking the original basis to the reduced one, and the
    /// reduced Gram matrix.
    fn reduced_enumerator(&self, budget: u64) -> Result<ReducedEnumerator, LatticeGeometryError> {
        if self.rank() > MAX_ENUM_RANK {
            return Err(LatticeGeometryError::RankTooLarge {
                rank: self.rank(),
                max: MAX_ENUM_RANK,
            });
        }
        let (g, u) = quadform::gram_lll(&self.gram, &Rational::from((99, 100)))?;
        let (mu, b) = quadform::gram_schmidt(&g)?;
        Ok((Enumerator::new(mu, b, budget), u, g))
    }

    /// Coefficients against the reduced basis, back to coefficients against the
    /// lattice's own basis: `x_orig = x_reduced · U`.
    fn lift(&self, x: &[Integer], u: &[Vec<Integer>]) -> Vec<Integer> {
        let m = u.len();
        (0..m)
            .map(|t| {
                let mut acc = Integer::new();
                for (i, xi) in x.iter().enumerate() {
                    acc += Integer::from(xi * &u[i][t]);
                }
                acc
            })
            .collect()
    }

    fn coordinates_of(&self, coeffs: &[Integer]) -> Option<Vec<Rational>> {
        let b = self.basis.as_ref()?;
        let n = b[0].len();
        Some(
            (0..n)
                .map(|c| {
                    let mut acc = Rational::new();
                    for (t, co) in coeffs.iter().enumerate() {
                        acc += Rational::from(&b[t][c] * &Rational::from(co));
                    }
                    acc
                })
                .collect(),
        )
    }

    fn vector_from(&self, x: &[Integer], u: &[Vec<Integer>], norm: Rational) -> LatticeVector {
        let coefficients = self.lift(x, u);
        let coordinates = self.coordinates_of(&coefficients);
        LatticeVector {
            coefficients,
            coordinates,
            norm,
        }
    }

    /// One Fincke–Pohst pass that finds the minimum **and** everything attaining
    /// it, rather than two passes with the minimum known in between.
    ///
    /// Returns `(λ₁², representatives)` where the representatives are one from
    /// each `±v` pair, in reduced coordinates. The starting radius is the
    /// smallest diagonal entry of the *reduced* Gram matrix — a genuine lattice
    /// vector's norm, so the ball provably contains a minimal vector.
    fn minimal_pass(
        &self,
        budget: u64,
        collect: bool,
    ) -> Result<MinimalPass, LatticeGeometryError> {
        let (mut e, u, g) = self.reduced_enumerator(budget)?;
        let bound = (0..g.len())
            .map(|i| g[i][i].clone())
            .fold(None::<Rational>, |acc, x| {
                Some(match acc {
                    Some(a) if a < x => a,
                    _ => x,
                })
            });
        let bound = bound.ok_or(LatticeError::EmptyBasis)?;

        let mut best: Option<Rational> = None;
        let mut count: u64 = 0;
        let mut reps: Vec<Vec<Integer>> = Vec::new();
        let mut witness: Vec<Integer> = Vec::new();
        e.enumerate(&bound, None, true, &mut |found| {
            if found.norm == 0 {
                return;
            }
            match &best {
                Some(b) if found.norm > *b => (),
                Some(b) if found.norm == *b => {
                    count += 1;
                    if collect {
                        reps.push(found.x.to_vec());
                    }
                }
                _ => {
                    best = Some(found.norm.clone());
                    count = 1;
                    witness = found.x.to_vec();
                    if collect {
                        reps.clear();
                        reps.push(found.x.to_vec());
                    }
                }
            }
        })?;
        let best = best.ok_or(LatticeGeometryError::NotPositiveDefinite { pivot: 1 })?;
        if !collect {
            reps.push(witness);
        }
        Ok((best, count * 2, reps, u))
    }

    /// The memoised default-budget shortest-vector pass.
    ///
    /// Only the no-budget entry points come through here; every
    /// `*_with_budget` method enumerates afresh, so passing a budget always
    /// means what it says. Only a *successful* pass is stored — a refusal is
    /// reproducible on the next call.
    fn minimal_cached(&self) -> Result<&MinimalPass, LatticeGeometryError> {
        if let Some(v) = self.minimal.get() {
            return Ok(v);
        }
        let computed = self.minimal_pass(DEFAULT_ENUM_NODE_BUDGET, false)?;
        // A concurrent caller may have won the race; either value is correct,
        // because both are the same lattice's minimum.
        let _ = self.minimal.set(computed);
        self.minimal.get().ok_or(LatticeGeometryError::Internal {
            detail: "the shortest-vector memo was empty immediately after being set",
        })
    }

    /// Squared norm of the shortest non-zero vector, exactly.
    pub fn minimum(&self) -> Result<Rational, LatticeGeometryError> {
        Ok(self.minimal_cached()?.0.clone())
    }

    /// [`Lattice::minimum`] with an explicit Fincke–Pohst node budget.
    ///
    /// Bypasses the memo — a budget is a request to do the work under that
    /// budget.
    pub fn minimum_with_budget(&self, budget: u64) -> Result<Rational, LatticeGeometryError> {
        Ok(self.minimal_pass(budget, false)?.0)
    }

    /// A shortest non-zero lattice vector.
    ///
    /// Exact: the returned vector attains the minimum. Which of the (at least
    /// two, `±v`) minimal vectors comes back is unspecified.
    pub fn shortest_vector(&self) -> Result<LatticeVector, LatticeGeometryError> {
        let (norm, _, reps, u) = self.minimal_cached()?;
        let x = reps.first().ok_or(LatticeError::EmptyBasis)?;
        Ok(self.vector_from(x, u, norm.clone()))
    }

    /// [`Lattice::shortest_vector`] with an explicit node budget.
    pub fn shortest_vector_with_budget(
        &self,
        budget: u64,
    ) -> Result<LatticeVector, LatticeGeometryError> {
        let (norm, _, reps, u) = self.minimal_pass(budget, false)?;
        let x = reps.first().ok_or(LatticeError::EmptyBasis)?;
        Ok(self.vector_from(x, &u, norm))
    }

    /// Every vector attaining the minimum — the shell at radius `λ₁`, both
    /// signs included. Its length is the kissing number.
    ///
    /// Runs its own enumeration: the memo that [`Lattice::minimum`] and
    /// [`Lattice::kissing_number`] share keeps one representative vector, not
    /// the whole shell, because for the Leech lattice the whole shell is
    /// 196560 vectors and nobody asking for a *number* wants to pay for it.
    pub fn minimal_vectors(&self) -> Result<Vec<LatticeVector>, LatticeGeometryError> {
        self.minimal_vectors_with_budget(DEFAULT_ENUM_NODE_BUDGET)
    }

    /// [`Lattice::minimal_vectors`] with an explicit node budget.
    pub fn minimal_vectors_with_budget(
        &self,
        budget: u64,
    ) -> Result<Vec<LatticeVector>, LatticeGeometryError> {
        let (norm, _, reps, u) = self.minimal_pass(budget, true)?;
        let mut out = Vec::with_capacity(reps.len() * 2);
        for x in reps {
            let v = self.vector_from(&x, &u, norm.clone());
            let neg = LatticeVector {
                coefficients: v.coefficients.iter().map(|c| Integer::from(-c)).collect(),
                coordinates: v
                    .coordinates
                    .as_ref()
                    .map(|c| c.iter().map(|x| -x.clone()).collect()),
                norm: norm.clone(),
            };
            out.push(v);
            out.push(neg);
        }
        Ok(out)
    }

    /// The kissing number: how many lattice vectors attain the minimum.
    pub fn kissing_number(&self) -> Result<u64, LatticeGeometryError> {
        Ok(self.minimal_cached()?.1)
    }

    /// [`Lattice::kissing_number`] with an explicit node budget.
    pub fn kissing_number_with_budget(&self, budget: u64) -> Result<u64, LatticeGeometryError> {
        Ok(self.minimal_pass(budget, false)?.1)
    }

    /// `theta[n]` = the number of lattice vectors of squared norm exactly `n`,
    /// for `n = 0 ..= max_norm`. `theta[0] = 1`.
    ///
    /// Requires an integral Gram matrix.
    pub fn theta_series(&self, max_norm: u64) -> Result<Vec<Integer>, LatticeGeometryError> {
        self.theta_series_with_budget(max_norm, DEFAULT_ENUM_NODE_BUDGET)
    }

    /// [`Lattice::theta_series`] with an explicit node budget.
    pub fn theta_series_with_budget(
        &self,
        max_norm: u64,
        budget: u64,
    ) -> Result<Vec<Integer>, LatticeGeometryError> {
        if max_norm > MAX_THETA_NORM {
            return Err(LatticeGeometryError::InvalidParameter {
                detail: "theta series bound exceeds MAX_THETA_NORM",
            });
        }
        for (i, row) in self.gram.iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                if *x.denom() != 1i32 {
                    return Err(LatticeGeometryError::NonIntegralGram { row: i, col: j });
                }
            }
        }
        let (mut e, _, _) = self.reduced_enumerator(budget)?;
        let mut counts = vec![Integer::new(); (max_norm as usize) + 1];
        // An integral Gram matrix makes every `Q(x)` a non-negative integer, and
        // the enumeration bound makes it `≤ max_norm`, so `slot` is always
        // `Some` — but "always" here is a property of two other functions, and
        // this runs under a PyO3 boundary where a panic arrives as a
        // `BaseException` that `except Exception` does not catch. So it is a
        // refusal the caller can handle, not an `expect`.
        let mut broken: Option<&'static str> = None;
        e.enumerate(&Rational::from(max_norm), None, true, &mut |found| {
            let slot = if *found.norm.denom() != 1i32 {
                None
            } else {
                found
                    .norm
                    .numer()
                    .to_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|n| *n < counts.len())
            };
            match slot {
                Some(n) => counts[n] += 1u32,
                None => {
                    broken.get_or_insert(
                        "an enumerated norm was not an integer in 0..=max_norm, \
                         despite an integral Gram matrix",
                    );
                }
            }
        })?;
        if let Some(detail) = broken {
            return Err(LatticeGeometryError::Internal { detail });
        }
        // The half-space enumeration saw one of each `±v` pair, and the origin
        // exactly once.
        for c in counts.iter_mut().skip(1) {
            *c *= 2u32;
        }
        counts[0] = Integer::from(1);
        Ok(counts)
    }

    /// A lattice vector closest to `target` (given in **ambient** coordinates).
    ///
    /// Exact. Requires a basis; a Gram-only lattice has no ambient space to
    /// name a target in and is refused with [`LatticeGeometryError::NoBasis`].
    pub fn closest_vector(
        &self,
        target: &[Rational],
    ) -> Result<LatticeVector, LatticeGeometryError> {
        self.closest_vector_with_budget(target, DEFAULT_ENUM_NODE_BUDGET)
    }

    /// [`Lattice::closest_vector`] with an explicit node budget.
    pub fn closest_vector_with_budget(
        &self,
        target: &[Rational],
        budget: u64,
    ) -> Result<LatticeVector, LatticeGeometryError> {
        let basis = self.basis.as_ref().ok_or(LatticeGeometryError::NoBasis)?;
        let n = basis[0].len();
        if target.len() != n {
            return Err(LatticeGeometryError::DimensionMismatch {
                expected: n,
                got: target.len(),
            });
        }
        let m = self.rank();
        // Coordinates of the orthogonal projection of `target` onto the span:
        // c = G⁻¹ · (⟨b_i, target⟩). Minimising over the span is the same
        // problem — the component orthogonal to the span is a constant offset.
        let rhs: Vec<Rational> = (0..m)
            .map(|i| {
                let mut acc = Rational::new();
                for (c, t) in basis[i].iter().zip(target.iter()) {
                    acc += Rational::from(c * t);
                }
                acc
            })
            .collect();
        let ginv = quadform::inverse(&self.gram)?;
        let coords: Vec<Rational> = (0..m)
            .map(|i| {
                let mut acc = Rational::new();
                for t in 0..m {
                    acc += Rational::from(&ginv[i][t] * &rhs[t]);
                }
                acc
            })
            .collect();

        let (mut e, u, _) = self.reduced_enumerator(budget)?;
        // Re-express the target in the *reduced* coordinates: reduced row i is
        // Σ_t U[i][t] b_t, so c = c' U  ⟹  c' = c U⁻¹.
        let u_rat: Vec<Vec<Rational>> = u
            .iter()
            .map(|r| r.iter().map(Rational::from).collect())
            .collect();
        let u_inv = quadform::inverse(&u_rat)?;
        let center: Vec<Rational> = (0..m)
            .map(|i| {
                let mut acc = Rational::new();
                for t in 0..m {
                    acc += Rational::from(&coords[t] * &u_inv[t][i]);
                }
                acc
            })
            .collect();

        let bound = e.babai_bound(&center);
        let mut best: Option<(Vec<Integer>, Rational)> = None;
        e.enumerate(&bound, Some(&center), false, &mut |found| {
            let better = match &best {
                None => true,
                Some((_, n)) => found.norm < *n,
            };
            if better {
                best = Some((found.x.to_vec(), found.norm.clone()));
            }
        })?;
        let (x, norm) = best.ok_or(LatticeGeometryError::NotPositiveDefinite { pivot: 1 })?;
        let coefficients = self.lift(&x, &u);
        let coordinates = self.coordinates_of(&coefficients);
        Ok(LatticeVector {
            coefficients,
            coordinates,
            norm,
        })
    }

    // ---------------------------------------------------------------------
    // Packing invariants
    // ---------------------------------------------------------------------

    /// Hermite invariant `γ = λ₁² / (det G)^{1/m}` (with `λ₁²` the minimum).
    ///
    /// `2.0` for `E_8`, `4.0` for the Leech lattice.
    pub fn hermite_invariant(&self) -> Result<f64, LatticeGeometryError> {
        let min = self.minimum()?.to_f64();
        let det = self.determinant().to_f64();
        Ok(min / det.powf(1.0 / self.rank() as f64))
    }

    /// Centre density `δ = (λ₁²/4)^{m/2} / sqrt(det G)`.
    ///
    /// `1/16` for `E_8`.
    pub fn center_density(&self) -> Result<f64, LatticeGeometryError> {
        let min = self.minimum()?.to_f64();
        let det = self.determinant().to_f64();
        Ok((min / 4.0).powf(self.rank() as f64 / 2.0) / det.sqrt())
    }

    /// The centre density as an exact rational, or `None` when it is irrational.
    ///
    /// Returns `None` rather than a rounded stand-in: `δ` is rational exactly
    /// when `(λ₁²/4)^{m/2}` and `sqrt(det G)` both are, which needs an even
    /// rank and a square determinant.
    pub fn center_density_exact(&self) -> Result<Option<Rational>, LatticeGeometryError> {
        let m = self.rank();
        if m % 2 != 0 {
            return Ok(None);
        }
        let Some(root) = exact_sqrt(&self.determinant()) else {
            return Ok(None);
        };
        let quarter_min = self.minimum()? / Rational::from(4);
        let mut num = Rational::from(1);
        for _ in 0..(m / 2) {
            num *= &quarter_min;
        }
        Ok(Some(num / root))
    }

    /// Sphere-packing density `Δ = δ · V_m`, `V_m` the volume of the unit
    /// `m`-ball. `≈ 0.2537` for `E_8`.
    pub fn packing_density(&self) -> Result<f64, LatticeGeometryError> {
        Ok(self.center_density()? * unit_ball_volume(self.rank()))
    }

    /// An LLL-reduced basis of the same lattice, as a new [`Lattice`].
    ///
    /// When this lattice has an **integer** basis the reduction runs through
    /// FLINT (see [`super::lattice_reduce_rows`]); otherwise it is the exact
    /// rational Gram-driven reduction, and only the Gram matrix is reduced.
    pub fn lll_reduced(&self) -> Result<Self, LatticeGeometryError> {
        if let Some(b) = &self.basis {
            if b.iter().all(|r| r.iter().all(|x| *x.denom() == 1i32)) {
                let int_rows: Vec<Vec<Integer>> = b
                    .iter()
                    .map(|r| r.iter().map(|x| Integer::from(x.numer())).collect())
                    .collect();
                let reduced = lattice_reduce_rows(&int_rows)?;
                return Self::from_basis(&reduced);
            }
        }
        let (g, u) = quadform::gram_lll(&self.gram, &Rational::from((99, 100)))?;
        let basis = self.basis.as_ref().map(|b| {
            let m = b.len();
            let n = b[0].len();
            (0..m)
                .map(|i| {
                    (0..n)
                        .map(|c| {
                            let mut acc = Rational::new();
                            for t in 0..m {
                                acc += Rational::from(&b[t][c] * &Rational::from(&u[i][t]));
                            }
                            acc
                        })
                        .collect()
                })
                .collect()
        });
        Ok(Self {
            basis,
            gram: g,
            minimal: OnceLock::new(),
        })
    }
}

/// Exact square root of a non-negative rational, or `None` when irrational.
fn exact_sqrt(x: &Rational) -> Option<Rational> {
    if *x < 0 {
        return None;
    }
    let n = Integer::from(x.numer());
    let d = Integer::from(x.denom());
    let (rn, remn) = n.sqrt_rem(Integer::new());
    if remn != 0 {
        return None;
    }
    let (rd, remd) = d.sqrt_rem(Integer::new());
    if remd != 0 {
        return None;
    }
    Some(Rational::from((rn, rd)))
}

/// Volume of the unit ball in `m` dimensions, `π^{m/2} / Γ(m/2 + 1)`.
///
/// Built up by the recursion `V_m = V_{m−2} · 2π/m` from `V_0 = 1`, `V_1 = 2`,
/// which avoids needing a `Γ` at half-integers.
fn unit_ball_volume(m: usize) -> f64 {
    let pi = std::f64::consts::PI;
    let even = m % 2 == 0;
    let mut v = if even { 1.0 } else { 2.0 };
    let mut k = if even { 2 } else { 3 };
    while k <= m {
        v *= 2.0 * pi / k as f64;
        k += 2;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_ball_volumes_match_the_closed_forms() {
        assert!((unit_ball_volume(1) - 2.0).abs() < 1e-12);
        assert!((unit_ball_volume(2) - std::f64::consts::PI).abs() < 1e-12);
        assert!((unit_ball_volume(3) - 4.0 / 3.0 * std::f64::consts::PI).abs() < 1e-12);
        assert!((unit_ball_volume(4) - std::f64::consts::PI.powi(2) / 2.0).abs() < 1e-12);
        // V_8 = π⁴/24
        assert!((unit_ball_volume(8) - std::f64::consts::PI.powi(4) / 24.0).abs() < 1e-12);
    }

    #[test]
    fn exact_sqrt_refuses_irrationals() {
        assert_eq!(exact_sqrt(&Rational::from(4)), Some(Rational::from(2)));
        assert_eq!(
            exact_sqrt(&Rational::from((9, 16))),
            Some(Rational::from((3, 4)))
        );
        assert_eq!(exact_sqrt(&Rational::from(2)), None);
        assert_eq!(exact_sqrt(&Rational::from((1, 3))), None);
    }
}
