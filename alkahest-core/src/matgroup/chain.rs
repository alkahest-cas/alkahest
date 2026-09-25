//! A base and strong generating set for a matrix group over GF(q), from
//! Schreier–Sims on the action on **vectors**.
//!
//! # Why this is not [`crate::group`]'s Schreier–Sims
//!
//! [`crate::group::PermutationGroup`] already has a deterministic Schreier–Sims
//! and this module reuses it wherever the induced action is small enough — see
//! [`super::MatGroup::permutation_action_on_vectors`]. What it cannot do is
//! carry this module's main case. A permutation representation of a subgroup of
//! `GL(d, q)` acting faithfully needs `q^d − 1` points, and
//! [`crate::group::MAX_BSGS_DEGREE`] is `256`, so every group past `GL(2, 16)`,
//! `GL(3, 5)` or `GL(8, 2)` would be refused before any mathematics happened.
//! Worse, a permutation of degree `q^d − 1` costs `q^d − 1` words where the
//! matrix it came from costs `d²`: for `GL(6, 4)` that is 4 095 words against
//! 36, and the transversals dominate the memory.
//!
//! So the chain here is on matrices. The structure is the same as the
//! permutation one — base, basic orbits with Schreier vectors, explicit
//! transversals, membership by sifting — and the algorithm is the same
//! deterministic Schreier-generator algorithm rather than a randomised
//! (Monte-Carlo) variant. Only the objects differ.
//!
//! # Base points are standard basis vectors
//!
//! A matrix that fixes every `eᵢ` (with the row-vector action `v ↦ v·M`, so
//! `eᵢ·M` is row `i` of `M`) has `M = I`. So a base can always be chosen inside
//! `{e₀, …, e_{d−1}}`, the chain has **at most `d` levels**, and the first step
//! of each sifting step — computing `β·h` — is a row read rather than a
//! matrix–vector product. Every base point recorded here is therefore a basis
//! *index*.
//!
//! # Correctness condition
//!
//! `(B, S)` is a BSGS exactly when, for every level `i`, every Schreier
//! generator `u_γ · x · u_{γ·x}⁻¹` of level `i` sifts to the identity through
//! levels `i+1 …`. The builder reaches that fixed point and stops; the deepest
//! level's Schreier generators sifting to the identity through *no* levels is
//! precisely the statement that the pointwise stabilizer of the base is
//! trivial, so the condition covers the end of the chain too.
//!
//! Schreier's lemma is used with a generating set that is **not** closed under
//! inversion. That is sound here because the group is finite: every element is
//! a positive word `s₁s₂⋯s_k` in the generators, and
//!
//! ```text
//!     h = (u_β s₁ u_{β·s₁}⁻¹)(u_{β·s₁} s₂ u_{β·s₁s₂}⁻¹) ⋯ (u_{β·s₁⋯s_{k−1}} s_k u_β⁻¹)
//! ```
//!
//! telescopes for any `h` in the stabilizer of `β`, so the Schreier generators
//! for positive words alone generate it.

use std::collections::HashMap;

use rug::Integer;

use super::element::{
    basis_vector, entry_key, first_moved_basis_vector, is_identity, row_vector, EntryKey,
};
use super::error::MatGroupError;
use super::{MAX_MATGROUP_ORBIT, MAX_MATGROUP_SCHREIER_WORK};
use crate::ffield::{FiniteField, GfMatrix};

/// One edge of a Schreier tree: the orbit point was reached from
/// `predecessor` by applying the level generator at `generator`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixSchreierEntry {
    /// Index into the level's generator list.
    pub generator: usize,
    /// Index into the orbit's point list.
    pub predecessor: usize,
}

/// The orbit of a basis vector under one level's generators, with its Schreier
/// vector and explicit transversal.
#[derive(Clone, Debug)]
pub struct MatrixOrbit {
    base_index: usize,
    points: Vec<GfMatrix>,
    index: HashMap<EntryKey, usize>,
    schreier: Vec<Option<MatrixSchreierEntry>>,
    transversal: Vec<GfMatrix>,
    transversal_inverse: Vec<GfMatrix>,
}

impl MatrixOrbit {
    /// The one-point orbit `{e_base_index}`, before any generator is known.
    fn trivial(
        field: &FiniteField,
        degree: usize,
        base_index: usize,
    ) -> Result<Self, MatGroupError> {
        let base = basis_vector(field, degree, base_index)?;
        let identity = GfMatrix::identity(field, degree)?;
        let mut index = HashMap::new();
        index.insert(entry_key(field, &base), 0);
        Ok(MatrixOrbit {
            base_index,
            points: vec![base],
            index,
            schreier: vec![None],
            transversal: vec![identity.clone()],
            transversal_inverse: vec![identity],
        })
    }

    /// Which standard basis vector this orbit starts from.
    pub fn base_index(&self) -> usize {
        self.base_index
    }

    /// The orbit points, as `1 × d` row vectors, in discovery order.
    /// `points()[0]` is the base point.
    pub fn points(&self) -> &[GfMatrix] {
        &self.points
    }

    /// How many points the orbit has.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// An orbit always contains its own base point, so this is never `true`.
    /// Present because clippy asks for it next to `len`.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The Schreier vector, parallel to [`MatrixOrbit::points`]. The base
    /// point's entry is `None`.
    pub fn schreier_vector(&self) -> &[Option<MatrixSchreierEntry>] {
        &self.schreier
    }

    /// The transversal element `u` with `β·u = points()[i]`.
    pub fn transversal_element(&self, i: usize) -> Option<&GfMatrix> {
        self.transversal.get(i)
    }

    /// The position of `v` in [`MatrixOrbit::points`], or `None` if `v` is not
    /// in the orbit.
    pub fn position(&self, field: &FiniteField, v: &GfMatrix) -> Option<usize> {
        self.index.get(&entry_key(field, v)).copied()
    }

    /// Is `v` in this orbit?
    pub fn contains(&self, field: &FiniteField, v: &GfMatrix) -> bool {
        self.position(field, v).is_some()
    }
}

/// One level of the chain: the base point, the generators of the stabilizer of
/// the earlier base points, and this level's basic orbit.
#[derive(Clone, Debug)]
pub struct MatrixStabilizerLevel {
    base_index: usize,
    generators: Vec<GfMatrix>,
    orbit: MatrixOrbit,
}

impl MatrixStabilizerLevel {
    /// Which standard basis vector this level stabilizes next.
    pub fn base_index(&self) -> usize {
        self.base_index
    }

    /// Strong generators at this level: they generate the pointwise stabilizer
    /// of all earlier base points.
    pub fn generators(&self) -> &[GfMatrix] {
        &self.generators
    }

    /// This level's basic orbit.
    pub fn orbit(&self) -> &MatrixOrbit {
        &self.orbit
    }
}

/// The outcome of sifting a matrix through the chain.
#[derive(Clone, Debug)]
pub struct MatrixSiftResult {
    residue: GfMatrix,
    level: usize,
    is_member: bool,
}

impl MatrixSiftResult {
    /// What was left after stripping off transversal elements. The identity
    /// exactly when the matrix is in the group.
    pub fn residue(&self) -> &GfMatrix {
        &self.residue
    }

    /// The level at which sifting stopped — `chain.levels().len()` when it ran
    /// all the way through.
    pub fn level(&self) -> usize {
        self.level
    }

    /// Is the sifted matrix in the group?
    pub fn is_member(&self) -> bool {
        self.is_member
    }
}

/// A base and strong generating set for a subgroup of `GL(d, q)`.
#[derive(Clone, Debug)]
pub struct MatrixStabilizerChain {
    field: FiniteField,
    degree: usize,
    levels: Vec<MatrixStabilizerLevel>,
    work: u64,
}

impl MatrixStabilizerChain {
    /// The field the matrices are over.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// The size of the matrices.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The base, as standard-basis indices.
    pub fn base(&self) -> Vec<usize> {
        self.levels.iter().map(|l| l.base_index).collect()
    }

    /// The base, as `1 × d` row vectors.
    pub fn base_vectors(&self) -> Result<Vec<GfMatrix>, MatGroupError> {
        self.levels
            .iter()
            .map(|l| basis_vector(&self.field, self.degree, l.base_index))
            .collect()
    }

    /// The levels, outermost first.
    pub fn levels(&self) -> &[MatrixStabilizerLevel] {
        &self.levels
    }

    /// How many elementary matrix operations building this chain cost. Exposed
    /// so that a caller who hits [`MatGroupError::WorkBudgetExhausted`] can see
    /// how far a successful run was from the budget.
    pub fn work(&self) -> u64 {
        self.work
    }

    /// Every strong generator, deduplicated, level 0 first.
    pub fn strong_generators(&self) -> Vec<GfMatrix> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for level in &self.levels {
            for g in &level.generators {
                if seen.insert(entry_key(&self.field, g)) {
                    out.push(g.clone());
                }
            }
        }
        out
    }

    /// `|G|`, as the product of the basic orbit lengths.
    ///
    /// Exact and arbitrary precision. An empty chain is the trivial group and
    /// gives `1`.
    pub fn order(&self) -> Integer {
        let mut acc = Integer::from(1);
        for level in &self.levels {
            acc *= level.orbit.len();
        }
        acc
    }

    /// Sift `m` through the chain.
    ///
    /// # Errors
    ///
    /// `E-MATGRP-001` / `-002` / `-003` when `m` is not a `d × d` matrix over
    /// this field; `E-GFQ-*` from the underlying linear algebra.
    pub fn sift(&self, m: &GfMatrix) -> Result<MatrixSiftResult, MatGroupError> {
        self.check_shape(m)?;
        let (residue, level) = self.strip_from(m, 0)?;
        let is_member = is_identity(&self.field, &residue);
        Ok(MatrixSiftResult {
            residue,
            level,
            is_member,
        })
    }

    fn check_shape(&self, m: &GfMatrix) -> Result<(), MatGroupError> {
        let (rows, cols) = m.shape();
        if rows != cols {
            return Err(MatGroupError::NotSquare { rows, cols });
        }
        if rows != self.degree {
            return Err(MatGroupError::DegreeMismatch {
                expected: self.degree,
                got: rows,
            });
        }
        if m.field() != &self.field {
            return Err(MatGroupError::FieldMismatch {
                expected: format!("{}", self.field),
                got: format!("{}", m.field()),
            });
        }
        Ok(())
    }

    /// Strip `m` through levels `start …`, returning the residue and the level
    /// at which it stopped.
    fn strip_from(&self, m: &GfMatrix, start: usize) -> Result<(GfMatrix, usize), MatGroupError> {
        let mut h = m.clone();
        let mut level = start;
        while level < self.levels.len() {
            let lev = &self.levels[level];
            // β·h with β = e_{base_index} is row `base_index` of h.
            let image = row_vector(&self.field, &h, lev.base_index)?;
            match lev.orbit.position(&self.field, &image) {
                None => return Ok((h, level)),
                Some(i) => {
                    h = h.mul(&lev.orbit.transversal_inverse[i])?;
                }
            }
            level += 1;
        }
        Ok((h, level))
    }

    /// Every element of the group, as products of transversal elements.
    ///
    /// The factorisation is `g = u^{(k−1)} ⋯ u^{(1)} u^{(0)}`: deepest level
    /// leftmost, because `β₀·g = points[i]` gives `g·u_i⁻¹ ∈ G^{(1)}`, i.e.
    /// `g = h·u_i` with `h` in the next stabilizer.
    ///
    /// # Errors
    ///
    /// `E-MATGRP-008` when `|G|` is above `cap`.
    pub fn elements_with_cap(&self, cap: u64) -> Result<Vec<GfMatrix>, MatGroupError> {
        let order = self.order();
        if order > cap {
            return Err(MatGroupError::EnumerationTooLarge {
                order: order.to_string(),
                cap,
            });
        }
        let mut acc = vec![GfMatrix::identity(&self.field, self.degree)?];
        for level in &self.levels {
            let mut next = Vec::with_capacity(acc.len() * level.orbit.len());
            for u in &level.orbit.transversal {
                for a in &acc {
                    next.push(u.mul(a)?);
                }
            }
            acc = next;
        }
        Ok(acc)
    }
}

// ---------------------------------------------------------------------------
// The builder
// ---------------------------------------------------------------------------

struct ChainBuilder {
    field: FiniteField,
    degree: usize,
    levels: Vec<MatrixStabilizerLevel>,
    work: u64,
    budget: u64,
}

impl ChainBuilder {
    fn spend(&mut self, n: u64) -> Result<(), MatGroupError> {
        self.work = self.work.saturating_add(n);
        if self.work > self.budget {
            return Err(MatGroupError::WorkBudgetExhausted {
                budget: self.budget,
            });
        }
        Ok(())
    }

    /// Append a level whose base point is a basis vector that `g` moves.
    ///
    /// `g` stabilizes every existing base point (it is a sifted residue), so
    /// the basis vector it first moves is necessarily new — asserted rather
    /// than assumed, because a repeated base point would make the order a
    /// product with a duplicated factor.
    fn push_level(&mut self, g: &GfMatrix) -> Result<(), MatGroupError> {
        let base_index =
            first_moved_basis_vector(&self.field, g).ok_or(MatGroupError::Internal {
                detail: "a non-identity matrix that fixes every standard basis vector",
            })?;
        if self.levels.iter().any(|l| l.base_index == base_index) {
            return Err(MatGroupError::Internal {
                detail: "a sifted residue moved a basis vector that is already a base point",
            });
        }
        let orbit = MatrixOrbit::trivial(&self.field, self.degree, base_index)?;
        self.levels.push(MatrixStabilizerLevel {
            base_index,
            generators: Vec::new(),
            orbit,
        });
        Ok(())
    }

    /// Recompute level `l`'s basic orbit, Schreier vector and transversal from
    /// its current generator list.
    fn recompute_orbit(&mut self, l: usize) -> Result<(), MatGroupError> {
        let base_index = self.levels[l].base_index;
        let generators = self.levels[l].generators.clone();
        let base = basis_vector(&self.field, self.degree, base_index)?;
        let identity = GfMatrix::identity(&self.field, self.degree)?;

        let mut points = vec![base.clone()];
        let mut index = HashMap::new();
        index.insert(entry_key(&self.field, &base), 0usize);
        let mut schreier: Vec<Option<MatrixSchreierEntry>> = vec![None];
        let mut transversal = vec![identity.clone()];
        let mut transversal_inverse = vec![identity];

        let mut i = 0;
        while i < points.len() {
            for (gi, g) in generators.iter().enumerate() {
                let image = points[i].mul(g)?;
                let key = entry_key(&self.field, &image);
                if let std::collections::hash_map::Entry::Vacant(slot) = index.entry(key) {
                    if points.len() >= MAX_MATGROUP_ORBIT {
                        return Err(MatGroupError::OrbitTooLarge {
                            points: points.len() + 1,
                            max: MAX_MATGROUP_ORBIT,
                        });
                    }
                    let u = transversal[i].mul(g)?;
                    let u_inverse = u.inverse()?;
                    slot.insert(points.len());
                    points.push(image);
                    schreier.push(Some(MatrixSchreierEntry {
                        generator: gi,
                        predecessor: i,
                    }));
                    transversal.push(u);
                    transversal_inverse.push(u_inverse);
                }
                self.spend(2)?;
            }
            i += 1;
        }

        self.levels[l].orbit = MatrixOrbit {
            base_index,
            points,
            index,
            schreier,
            transversal,
            transversal_inverse,
        };
        Ok(())
    }

    fn strip_from(&self, m: &GfMatrix, start: usize) -> Result<(GfMatrix, usize), MatGroupError> {
        let mut h = m.clone();
        let mut level = start;
        while level < self.levels.len() {
            let lev = &self.levels[level];
            let image = row_vector(&self.field, &h, lev.base_index)?;
            match lev.orbit.position(&self.field, &image) {
                None => return Ok((h, level)),
                Some(i) => h = h.mul(&lev.orbit.transversal_inverse[i])?,
            }
            level += 1;
        }
        Ok((h, level))
    }

    /// Make `levels[l …]` a complete BSGS for `⟨levels[l].generators⟩`.
    ///
    /// Recurses into the deeper levels it changes, deepest first, so that when
    /// a Schreier generator of level `l` sifts to the identity the chain below
    /// `l` is already a *correct* BSGS for its group and the sift is an exact
    /// membership test. That ordering is what makes it safe not to rescan the
    /// Schreier generators of level `l` already checked: adding generators to
    /// deeper levels only enlarges the groups they generate, and level `l`'s own
    /// generator list — hence its orbit and transversal — is never touched,
    /// because a residue is always inserted at a level `> l`.
    fn schreier_sims(&mut self, l: usize) -> Result<(), MatGroupError> {
        self.recompute_orbit(l)?;
        let mut i = 0;
        while i < self.levels[l].orbit.points.len() {
            let generator_count = self.levels[l].generators.len();
            for gi in 0..generator_count {
                let schreier_generator = {
                    let lev = &self.levels[l];
                    let x = &lev.generators[gi];
                    let image = lev.orbit.points[i].mul(x)?;
                    let j =
                        lev.orbit
                            .position(&self.field, &image)
                            .ok_or(MatGroupError::Internal {
                                detail: "a generator carried an orbit point outside its own orbit",
                            })?;
                    // A Schreier-tree edge gives u_i·x·u_j⁻¹ = identity by
                    // construction (u_j was *defined* as u_i·x); skipping it
                    // avoids the two multiplications and the comparison.
                    if lev.orbit.schreier[j]
                        == Some(MatrixSchreierEntry {
                            generator: gi,
                            predecessor: i,
                        })
                    {
                        None
                    } else {
                        Some(
                            lev.orbit.transversal[i]
                                .mul(x)?
                                .mul(&lev.orbit.transversal_inverse[j])?,
                        )
                    }
                };
                let Some(s) = schreier_generator else {
                    continue;
                };
                self.spend(4)?;
                if is_identity(&self.field, &s) {
                    continue;
                }
                let (residue, j) = self.strip_from(&s, l + 1)?;
                self.spend(u64::try_from(self.levels.len()).unwrap_or(u64::MAX))?;
                if is_identity(&self.field, &residue) {
                    continue;
                }
                if j == self.levels.len() {
                    self.push_level(&residue)?;
                }
                for k in (l + 1)..=j {
                    self.levels[k].generators.push(residue.clone());
                }
                for k in ((l + 1)..=j).rev() {
                    self.schreier_sims(k)?;
                }
            }
            i += 1;
        }
        Ok(())
    }
}

/// Build a BSGS for `⟨generators⟩ ≤ GL(degree, q)`.
///
/// The generators are assumed already validated: square, of size `degree`, over
/// `field`, and invertible. [`super::MatGroup::new`] is what enforces that.
///
/// # Errors
///
/// `E-MATGRP-006` when a basic orbit outgrows [`MAX_MATGROUP_ORBIT`];
/// `E-MATGRP-007` when the work budget is exhausted — a refusal, never a
/// partial chain; `E-MATGRP-013` on a broken invariant; `E-GFQ-*` from the
/// linear algebra.
pub(crate) fn build_chain(
    field: &FiniteField,
    degree: usize,
    generators: &[GfMatrix],
    budget: u64,
) -> Result<MatrixStabilizerChain, MatGroupError> {
    let mut builder = ChainBuilder {
        field: field.clone(),
        degree,
        levels: Vec::new(),
        work: 0,
        budget: budget.max(1),
    };
    let nontrivial: Vec<GfMatrix> = generators
        .iter()
        .filter(|g| !is_identity(field, g))
        .cloned()
        .collect();
    if !nontrivial.is_empty() {
        builder.push_level(&nontrivial[0])?;
        builder.levels[0].generators = nontrivial;
        builder.schreier_sims(0)?;
    }
    Ok(MatrixStabilizerChain {
        field: field.clone(),
        degree,
        levels: builder.levels,
        work: builder.work,
    })
}

/// The default Schreier–Sims work budget, in elementary matrix operations.
pub(crate) const DEFAULT_BUDGET: u64 = MAX_MATGROUP_SCHREIER_WORK;
