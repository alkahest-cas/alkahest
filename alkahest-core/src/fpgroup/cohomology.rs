//! Low-degree group cohomology `H^d(G, M)` from the inhomogeneous bar
//! resolution, for `d ∈ {0, 1, 2}`.
//!
//! # What is computed
//!
//! For a **finite** group `G` and a finitely generated abelian group `M` with a
//! `G`-action, the inhomogeneous ("bar") cochain complex is
//!
//! ```text
//!     C⁰ = M  →  C¹ = {f: G → M}  →  C² = {f: G² → M}  →  C³ = {f: G³ → M}
//! ```
//!
//! with
//!
//! ```text
//!     (d⁰m)(g)      = g·m − m
//!     (d¹f)(g,h)    = g·f(h) − f(gh) + f(g)
//!     (d²f)(g,h,k)  = g·f(h,k) − f(gh,k) + f(g,hk) − f(g,h)
//! ```
//!
//! and `H^d = ker d^d / im d^(d−1)`. `H⁰` is `M^G`, `H¹` classifies the
//! complements of `M` in `M ⋊ G` up to conjugacy, and `H²` classifies the
//! extensions of `M` by `G` with the given action — central extensions when the
//! action is trivial. That last one is the case this module exists for.
//!
//! # `M` may have torsion, and that is why this is not a kernel-of-a-matrix
//!
//! `M = ℤ^k / L` with `L` the diagonal lattice `⟨m_i e_i⟩` (`m_i = 0` meaning a
//! `ℤ` summand). The cochain groups are then `ℤ^N / L^{G^d}`, **not** free, so
//! `ker` and `im` cannot be read off a Smith form of the coboundary alone. What
//! is computed instead is
//!
//! ```text
//!     H^d  =  K / S,     K = { c ∈ ℤ^{N_d} : c·D_d ∈ L_{d+1} },
//!                        S = im D_{d−1} + L_d
//! ```
//!
//! `K` is the projection of an integer kernel — the left kernel of `D_d` stacked
//! on `−diag(L_{d+1})`, which is a Hermite normal form, taken a block of columns
//! at a time — and `K/S` is the invariant-factor list of the coordinate matrix of
//! `S` in an echelon basis of `K`. Both normal forms come from
//! [`crate::matrix::normal_form`]; the blocking that makes the Hermite form
//! affordable, and why it is needed, are documented on `fpgroup::abelian`'s
//! `preimage_lattice_blocked`.
//!
//! # Conventions
//!
//! * The action is a **left** action written with matrices on the left of column
//!   vectors: `(g·v)_i = Σ_j A_g[i][j] v_j`, and `A_{gh} = A_g · A_h`. Only the
//!   generators' matrices are supplied; the rest are built by walking the Cayley
//!   graph.
//! * Group elements are numbered by the cosets of the trivial subgroup, so
//!   element `0` is the identity.
//!
//! # What is checked before an answer is returned
//!
//! * Each generator matrix must map `L` into itself — otherwise it is not an
//!   endomorphism of `M` at all.
//! * The assignment must be a homomorphism: `A_{g·x} ≡ A_g·A_x (mod L)` for
//!   every element `g` and generator `x`. Checking every edge of the Cayley
//!   graph is equivalent to checking every relator, and avoids having to invert
//!   a matrix over `M`. A failure is [`FpGroupError::ActionNotWellDefined`], not
//!   a number.
//! * `S ⊆ K` is verified while solving for coordinates, which is exactly
//!   `d ∘ d = 0`. A violation is an internal error, and has never fired.
//!
//! # Scope limits, stated rather than approximated
//!
//! * Degrees `0`, `1`, `2` only. `H³` needs `C⁴`, of dimension `rank(M)·|G|⁴`.
//! * `|G| ≤ `[`MAX_COHOMOLOGY_GROUP_ORDER`], `rank(M) ≤ `[`MAX_MODULE_RANK`],
//!   and the degree-`(d+1)` cochain dimension `rank(M)·|G|^(d+1)` at most
//!   [`MAX_COCHAIN_DIMENSION`]. `H²` is therefore available for `|G| ≤ 12` at
//!   rank 1 — `A₄` yes, `S₄` no — and for smaller groups at higher rank. This is
//!   a deliberately small window: the cost is driven by a Hermite form over a
//!   matrix with `|G|³` columns, and a general-looking answer that takes an hour
//!   and cannot be checked by hand is worse than a stated limit.
//! * `G` must be finite *and* its order must be computable by coset
//!   enumeration. An infinite `G` has cohomology; this module will not compute
//!   it.

// The coboundary matrices are built by coordinate arithmetic — a group element
// index, two module-coordinate indices and a flattened tuple index, all in the
// same expression — so `needless_range_loop`'s enumerate() rewrites would name
// one collection and index the rest. `matrix::smith` allows it for the same
// reason.
#![allow(clippy::needless_range_loop)]

use super::abelian::{preimage_lattice_blocked, quotient_invariants, AbelianInvariants};
use super::error::FpGroupError;
use super::presentation::FpGroup;
use rug::Integer;

/// Largest module rank accepted.
pub const MAX_MODULE_RANK: usize = 8;

/// Largest group order accepted for a cohomology computation.
pub const MAX_COHOMOLOGY_GROUP_ORDER: usize = 32;

/// Largest cochain dimension `rank(M)·|G|^(d+1)` accepted.
///
/// The binding constraint for `H²`, where it is cubic in `|G|`: at rank 1 it
/// admits `|G| ≤ 12` and refuses `S₄`, whose degree-3 cochain group already has
/// 13824 generators. Measured against it in an unoptimised build, `H²` of a
/// group of order 12 takes between one and fifteen seconds depending on the
/// coefficients — torsion coefficients cost the most, because the relation
/// lattice has to be intersected as well as the kernel taken. `H¹` and `H⁰` are
/// bounded by [`MAX_COHOMOLOGY_GROUP_ORDER`] long before they reach this.
pub const MAX_COCHAIN_DIMENSION: usize = 2000;

/// Highest cohomological degree implemented.
pub const MAX_COHOMOLOGY_DEGREE: usize = 2;

/// Magnitude ceiling on a supplied action-matrix entry, in bits.
const MAX_ACTION_ENTRY_BITS: u32 = 20;

/// Magnitude ceiling on an entry of a *derived* action matrix. Exceeding it
/// means the assignment is not a finite-order action; the homomorphism check
/// would reject it anyway, but not before the products got large.
const MAX_DERIVED_ENTRY_BITS: u32 = 128;

type Mat = Vec<Vec<Integer>>;

// ---------------------------------------------------------------------------
// The module
// ---------------------------------------------------------------------------

/// A finitely generated abelian group `M = ℤ^k / ⟨m_i e_i⟩` with a `G`-action.
///
/// `invariants[i] == 0` makes the `i`-th summand `ℤ`; `invariants[i] == d ≥ 1`
/// makes it `ℤ/d`. The action is given by one integer matrix per **generator**
/// of the presentation, acting on the left of column vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GModule {
    invariants: Vec<Integer>,
    action: Vec<Mat>,
}

impl GModule {
    /// A module with the **trivial** action: every generator acts as the
    /// identity.
    ///
    /// Cannot fail either well-definedness check, which is why the textbook
    /// values (`H²(ℤ/n, ℤ) = ℤ/n`, `H¹(G, ℤ) = 0`) are all reachable through it.
    pub fn trivial(invariants: Vec<Integer>, group_rank: usize) -> Result<GModule, FpGroupError> {
        let rank = invariants.len();
        let identity: Mat = (0..rank)
            .map(|i| {
                (0..rank)
                    .map(|j| Integer::from(usize::from(i == j)))
                    .collect()
            })
            .collect();
        GModule::new(invariants, vec![identity; group_rank])
    }

    /// A module with an explicit action: `action[k]` is the matrix of the `k`-th
    /// generator of the presentation.
    ///
    /// Validated on construction for shape and for mapping the relation lattice
    /// into itself. Whether the matrices satisfy the *relators* cannot be
    /// checked without the group, so it is checked when the cohomology is
    /// computed.
    pub fn new(invariants: Vec<Integer>, action: Vec<Mat>) -> Result<GModule, FpGroupError> {
        let rank = invariants.len();
        if rank == 0 {
            return Err(FpGroupError::ModuleShape {
                detail: "a module needs at least one generator".into(),
            });
        }
        if rank > MAX_MODULE_RANK {
            return Err(FpGroupError::ModuleShape {
                detail: format!("rank {rank} exceeds MAX_MODULE_RANK ({MAX_MODULE_RANK})"),
            });
        }
        for (i, d) in invariants.iter().enumerate() {
            if *d < 0 {
                return Err(FpGroupError::ModuleShape {
                    detail: format!(
                        "invariant {i} is {d}; use 0 for a Z summand and d >= 1 for Z/d"
                    ),
                });
            }
        }
        for (k, m) in action.iter().enumerate() {
            if m.len() != rank {
                return Err(FpGroupError::ModuleShape {
                    detail: format!("action matrix {k} has {} rows, expected {rank}", m.len()),
                });
            }
            for (i, row) in m.iter().enumerate() {
                if row.len() != rank {
                    return Err(FpGroupError::ModuleShape {
                        detail: format!(
                            "row {i} of action matrix {k} has {} entries, expected {rank}",
                            row.len()
                        ),
                    });
                }
                for e in row {
                    if e.significant_bits() > MAX_ACTION_ENTRY_BITS {
                        return Err(FpGroupError::ModuleShape {
                            detail: format!(
                                "action matrix {k} has an entry of {} bits, above the {MAX_ACTION_ENTRY_BITS}-bit limit",
                                e.significant_bits()
                            ),
                        });
                    }
                }
            }
            check_preserves_lattice(m, &invariants, k)?;
        }
        Ok(GModule { invariants, action })
    }

    /// The rank of `M` as a quotient of `ℤ^k` — the number of invariants.
    pub fn rank(&self) -> usize {
        self.invariants.len()
    }

    /// The invariants: `0` for a `ℤ` summand, `d` for `ℤ/d`.
    pub fn invariants(&self) -> &[Integer] {
        &self.invariants
    }

    /// The number of action matrices, which must equal the presentation's rank.
    pub fn group_rank(&self) -> usize {
        self.action.len()
    }

    /// The action matrix of generator `k`.
    pub fn generator_action(&self, k: usize) -> Option<&Mat> {
        self.action.get(k)
    }

    /// Does every generator act as the identity?
    pub fn is_trivial_action(&self) -> bool {
        let identity = identity_matrix(self.rank());
        self.action.iter().all(|m| *m == identity)
    }
}

/// `A(L) ⊆ L`: an endomorphism of `ℤ^k` descends to `M` exactly when it maps the
/// relation lattice into itself.
fn check_preserves_lattice(
    m: &Mat,
    invariants: &[Integer],
    which: usize,
) -> Result<(), FpGroupError> {
    let rank = invariants.len();
    for j in 0..rank {
        if invariants[j] == 0 {
            continue;
        }
        for i in 0..rank {
            let product = Integer::from(&invariants[j] * &m[i][j]);
            let ok = if invariants[i] == 0 {
                product == 0
            } else {
                product.is_divisible(&invariants[i])
            };
            if !ok {
                return Err(FpGroupError::ActionNotWellDefined {
                    detail: format!(
                        "generator {which}'s matrix does not respect the module's relations: \
                         entry ({i},{j}) is {} and {} * that is not 0 in Z/{}",
                        m[i][j], invariants[j], invariants[i]
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Reduce each row `i` modulo `invariants[i]` when that is positive. Leaves free
/// rows alone; the result represents the same endomorphism of `M`.
fn reduce(m: &Mat, invariants: &[Integer]) -> Mat {
    m.iter()
        .enumerate()
        .map(|(i, row)| {
            if invariants[i] > 0 {
                row.iter()
                    .map(|e| e.clone().modulo(&invariants[i]))
                    .collect()
            } else {
                row.clone()
            }
        })
        .collect()
}

fn mat_mul(a: &Mat, b: &Mat) -> Mat {
    let n = a.len();
    let mut out = vec![vec![Integer::from(0); n]; n];
    for i in 0..n {
        for k in 0..n {
            if a[i][k] == 0 {
                continue;
            }
            for j in 0..n {
                if b[k][j] == 0 {
                    continue;
                }
                let term = Integer::from(&a[i][k] * &b[k][j]);
                out[i][j] += term;
            }
        }
    }
    out
}

fn identity_matrix(rank: usize) -> Mat {
    (0..rank)
        .map(|i| {
            (0..rank)
                .map(|j| Integer::from(usize::from(i == j)))
                .collect()
        })
        .collect()
}

/// The action matrix of every group element, by breadth-first search over the
/// Cayley graph, checking every edge.
///
/// Only *positive* generator edges are used. In a finite group the monoid
/// generated by the generators is the whole group, so the search reaches
/// everything, and no matrix has to be inverted over `M`.
fn element_actions(right_mult: &[Vec<usize>], module: &GModule) -> Result<Vec<Mat>, FpGroupError> {
    let n = right_mult.len();
    let rank = module.rank();
    let rk_gens = module.group_rank();
    let mut actions: Vec<Option<Mat>> = vec![None; n];
    actions[0] = Some(identity_matrix(rank));
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(0usize);
    let mut visited = 1usize;
    while let Some(c) = queue.pop_front() {
        let current = actions[c].clone().ok_or_else(|| FpGroupError::Internal {
            detail: format!("element {c} dequeued without an action matrix"),
        })?;
        for k in 0..rk_gens {
            let target = right_mult[c][k];
            let product = reduce(&mat_mul(&current, &module.action[k]), &module.invariants);
            for row in &product {
                for e in row {
                    if e.significant_bits() > MAX_DERIVED_ENTRY_BITS {
                        return Err(FpGroupError::ActionNotWellDefined {
                            detail: "the matrix products grow without bound, so the generators \
                                     cannot have finite order on M"
                                .into(),
                        });
                    }
                }
            }
            match &actions[target] {
                Some(existing) => {
                    if *existing != product {
                        return Err(FpGroupError::ActionNotWellDefined {
                            detail: format!(
                                "A(g*x) != A(g)*A(x) for element {c} and generator {k}, so some \
                                 relator does not act as the identity on M"
                            ),
                        });
                    }
                }
                None => {
                    actions[target] = Some(product);
                    visited += 1;
                    queue.push_back(target);
                }
            }
        }
    }
    if visited != n {
        return Err(FpGroupError::Internal {
            detail: format!("Cayley graph walk reached {visited} of {n} elements"),
        });
    }
    actions
        .into_iter()
        .map(|a| {
            a.ok_or_else(|| FpGroupError::Internal {
                detail: "element without an action matrix after the walk".into(),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The complex
// ---------------------------------------------------------------------------

fn cochain_dimension(rank: usize, n: usize, degree: usize) -> Result<usize, FpGroupError> {
    let tuples = n
        .checked_pow(degree as u32)
        .ok_or_else(|| FpGroupError::CohomologyTooLarge {
            detail: format!("|G|^{degree} overflows for |G| = {n}"),
        })?;
    tuples
        .checked_mul(rank)
        .ok_or_else(|| FpGroupError::CohomologyTooLarge {
            detail: format!("rank(M)*|G|^{degree} overflows for |G| = {n}"),
        })
}

/// The modulus of each coordinate of `C^degree`: coordinate `(tuple, i)` carries
/// the module's `i`-th invariant.
fn moduli(rank: usize, n: usize, degree: usize, invariants: &[Integer]) -> Vec<Integer> {
    let tuples = n.pow(degree as u32);
    let mut out = Vec::with_capacity(tuples * rank);
    for _ in 0..tuples {
        for i in 0..rank {
            out.push(invariants[i].clone());
        }
    }
    out
}

/// Generators of the relation lattice of `C^degree`: one row `m_p·e_p` per
/// coordinate with a positive modulus.
fn lattice_generators(moduli: &[Integer]) -> Vec<Vec<Integer>> {
    let dim = moduli.len();
    moduli
        .iter()
        .enumerate()
        .filter(|(_, m)| **m > 0)
        .map(|(p, m)| {
            let mut row = vec![Integer::from(0); dim];
            row[p] = m.clone();
            row
        })
        .collect()
}

/// The coboundary `d^degree` as a `dim_degree × dim_(degree+1)` matrix whose row
/// `p` is the image of the `p`-th basis cochain.
fn coboundary(
    degree: usize,
    n: usize,
    rank: usize,
    mult: &[Vec<usize>],
    actions: &[Mat],
) -> Result<Vec<Vec<Integer>>, FpGroupError> {
    let in_dim = cochain_dimension(rank, n, degree)?;
    let out_dim = cochain_dimension(rank, n, degree + 1)?;
    let mut m = vec![vec![Integer::from(0); out_dim]; in_dim];
    match degree {
        // (d0 m)(g) = g.m - m
        0 => {
            for g in 0..n {
                for i in 0..rank {
                    let out = g * rank + i;
                    for j in 0..rank {
                        let a = actions[g][i][j].clone();
                        m[j][out] += a;
                    }
                    m[i][out] -= 1;
                }
            }
        }
        // (d1 f)(x, y) = x.f(y) - f(xy) + f(x)
        1 => {
            for x in 0..n {
                for y in 0..n {
                    let out_tuple = x * n + y;
                    for i in 0..rank {
                        let out = out_tuple * rank + i;
                        for j in 0..rank {
                            let a = actions[x][i][j].clone();
                            m[y * rank + j][out] += a;
                        }
                        m[mult[x][y] * rank + i][out] -= 1;
                        m[x * rank + i][out] += 1;
                    }
                }
            }
        }
        // (d2 f)(x, y, z) = x.f(y, z) - f(xy, z) + f(x, yz) - f(x, y)
        2 => {
            for x in 0..n {
                for y in 0..n {
                    for z in 0..n {
                        let out_tuple = (x * n + y) * n + z;
                        for i in 0..rank {
                            let out = out_tuple * rank + i;
                            for j in 0..rank {
                                let a = actions[x][i][j].clone();
                                m[(y * n + z) * rank + j][out] += a;
                            }
                            m[(mult[x][y] * n + z) * rank + i][out] -= 1;
                            m[(x * n + mult[y][z]) * rank + i][out] += 1;
                            m[(x * n + y) * rank + i][out] -= 1;
                        }
                    }
                }
            }
        }
        d => {
            return Err(FpGroupError::UnsupportedCohomologyDegree {
                degree: d,
                max: MAX_COHOMOLOGY_DEGREE,
            })
        }
    }
    Ok(m)
}

// ---------------------------------------------------------------------------
// The entry point
// ---------------------------------------------------------------------------

/// `H^degree(G, M)`.
///
/// See the module docs for the conventions, the checks, and the size limits.
pub(super) fn cohomology(
    group: &FpGroup,
    degree: usize,
    module: &GModule,
) -> Result<AbelianInvariants, FpGroupError> {
    if degree > MAX_COHOMOLOGY_DEGREE {
        return Err(FpGroupError::UnsupportedCohomologyDegree {
            degree,
            max: MAX_COHOMOLOGY_DEGREE,
        });
    }
    if module.group_rank() != group.rank() {
        return Err(FpGroupError::ModuleShape {
            detail: format!(
                "the module carries {} action matrices but the presentation has {} generators",
                module.group_rank(),
                group.rank()
            ),
        });
    }
    let abelian = group.abelian_invariants()?;
    if abelian.free_rank() > 0 {
        return Err(FpGroupError::ProvablyInfinite {
            abelian_invariants: abelian.to_string(),
        });
    }

    let table = group.coset_table(&[])?;
    let n = table.index();
    if n > MAX_COHOMOLOGY_GROUP_ORDER {
        return Err(FpGroupError::CohomologyTooLarge {
            detail: format!(
                "|G| = {n} is above MAX_COHOMOLOGY_GROUP_ORDER ({MAX_COHOMOLOGY_GROUP_ORDER})"
            ),
        });
    }
    let rank = module.rank();
    let top = cochain_dimension(rank, n, degree + 1)?;
    if top > MAX_COCHAIN_DIMENSION {
        return Err(FpGroupError::CohomologyTooLarge {
            detail: format!(
                "the degree-{} cochain group has rank(M)*|G|^{} = {top} generators, above \
                 MAX_COCHAIN_DIMENSION ({MAX_COCHAIN_DIMENSION})",
                degree + 1,
                degree + 1
            ),
        });
    }

    // Right multiplication by each generator is exactly the coset table, and the
    // full multiplication table comes from tracing transversal words.
    let transversal = table.transversal()?;
    let mut right_mult = vec![vec![0usize; group.rank()]; n];
    for c in 0..n {
        for k in 0..group.rank() {
            right_mult[c][k] = table.image(c, k as i32 + 1)?;
        }
    }
    let mut mult = vec![vec![0usize; n]; n];
    for a in 0..n {
        for b in 0..n {
            mult[a][b] = table.trace(a, &transversal[b])?;
        }
    }
    let actions = element_actions(&right_mult, module)?;

    let dim = cochain_dimension(rank, n, degree)?;
    let d_here = coboundary(degree, n, rank, &mult, &actions)?;
    let moduli_here = moduli(rank, n, degree, module.invariants());
    let moduli_above = moduli(rank, n, degree + 1, module.invariants());

    // K = { c : c.D in L_{degree+1} }. For a free module every modulus is zero,
    // the stacked block is empty, and this degenerates to the plain integer
    // kernel of the coboundary.
    let kernel = preimage_lattice_blocked(&d_here, dim, top, &moduli_above)?;

    // S = im d^(degree-1) + L_degree.
    let mut image: Vec<Vec<Integer>> = if degree == 0 {
        Vec::new()
    } else {
        coboundary(degree - 1, n, rank, &mult, &actions)?
    };
    image.extend(lattice_generators(&moduli_here));

    quotient_invariants(dim, &kernel, &image)
}
