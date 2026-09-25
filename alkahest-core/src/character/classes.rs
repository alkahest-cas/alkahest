//! Conjugacy classes of a permutation group, by orbit enumeration under
//! conjugation.
//!
//! The classes are the orbits of `G` acting on itself by `x ↦ g⁻¹ x g`, and
//! this module computes them the direct way: list the elements, then breadth-
//! first search each orbit using the generators alone (conjugating by a
//! generating set suffices, because conjugation is an action). That costs
//! `O(|G| · |gens| · n)` permutation compositions and `O(|G| · n)` words of
//! memory, which is why the group order is **capped** rather than the method
//! being made cleverer — see [`MAX_CLASS_ENUMERATION_CAP`].
//!
//! Nothing is randomised and nothing is approximate. The class ordering is
//! deterministic (see [`ConjugacyClasses::of`]) so that a character table's
//! columns mean the same thing on every run.

use std::cmp::Ordering;

use rug::Integer;

use super::error::CharacterError;
use crate::group::{Permutation, PermutationGroup};

/// Default ceiling on `|G|` for conjugacy-class enumeration.
///
/// Chosen so that the element list and the parallel class map stay comfortably
/// inside a few tens of megabytes at the degrees [`crate::group`] admits. Raise
/// it deliberately with [`ConjugacyClasses::of_with_cap`].
pub const DEFAULT_CLASS_ENUMERATION_CAP: u64 = 20_000;

/// Hard ceiling on the cap [`ConjugacyClasses::of_with_cap`] will accept.
///
/// Above this the element list alone is the problem: this module holds every
/// element of `G` as a degree-`n` images array, so the memory is
/// `O(|G| · n)` words no matter how patient the caller is.
pub const MAX_CLASS_ENUMERATION_CAP: u64 = 200_000;

/// One conjugacy class: a representative, its size, and the invariants that
/// follow from them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConjugacyClass {
    representative: Permutation,
    size: u64,
    centraliser_order: Integer,
    element_order: u64,
}

impl ConjugacyClass {
    /// The class representative.
    ///
    /// This is the **lexicographically smallest** images array in the class, so
    /// it does not depend on the order the generators were given in.
    pub fn representative(&self) -> &Permutation {
        &self.representative
    }

    /// `|K|`, the number of elements in the class.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// `|C_G(g)| = |G| / |K|` for any `g` in the class.
    ///
    /// The orbit–stabilizer theorem for the conjugation action: the stabilizer
    /// of `g` is exactly its centraliser, so this is exact rather than an
    /// estimate, and it is arbitrary precision because `|G|` is.
    pub fn centraliser_order(&self) -> &Integer {
        &self.centraliser_order
    }

    /// The order of the representative — a class invariant, since conjugate
    /// elements have equal order.
    pub fn element_order(&self) -> u64 {
        self.element_order
    }
}

/// The conjugacy classes of a [`PermutationGroup`], with element-to-class
/// lookup and the class multiplication coefficients.
#[derive(Clone, Debug)]
pub struct ConjugacyClasses {
    degree: usize,
    group_order: Integer,
    exponent: u64,
    classes: Vec<ConjugacyClass>,
    /// Every element of `G`, sorted by images array so that lookup is a binary
    /// search. Sorted order also makes the whole structure independent of the
    /// order `elements()` happened to produce.
    elements: Vec<Permutation>,
    /// Parallel to `elements`: which class each one is in.
    class_of: Vec<usize>,
    /// Per class, the indices into `elements` of its members.
    members: Vec<Vec<usize>>,
    /// `inverse_class[i]` is the class of `g⁻¹` for `g` in class `i`.
    inverse_class: Vec<usize>,
}

impl ConjugacyClasses {
    /// The conjugacy classes of `group`, refusing above
    /// [`DEFAULT_CLASS_ENUMERATION_CAP`].
    ///
    /// # Ordering
    ///
    /// Classes are sorted by `(order of the representative, class size,
    /// representative)`. Class `0` is therefore always `{1}`: the identity is
    /// the only element of order 1. The ordering depends on the group as a
    /// permutation group and not on how it was presented.
    ///
    /// # Checks
    ///
    /// Every class size divides `|G|` and the sizes sum to `|G|`; a violation
    /// is [`CharacterError::SelfCheckFailed`] rather than a returned partition.
    pub fn of(group: &PermutationGroup) -> Result<Self, CharacterError> {
        Self::of_with_cap(group, DEFAULT_CLASS_ENUMERATION_CAP)
    }

    /// As [`ConjugacyClasses::of`], with an explicit ceiling on `|G|`.
    ///
    /// Refuses with [`CharacterError::CapTooLarge`] above
    /// [`MAX_CLASS_ENUMERATION_CAP`].
    pub fn of_with_cap(group: &PermutationGroup, cap: u64) -> Result<Self, CharacterError> {
        if cap > MAX_CLASS_ENUMERATION_CAP {
            return Err(CharacterError::CapTooLarge {
                cap,
                max: MAX_CLASS_ENUMERATION_CAP,
            });
        }
        let group_order = group.order()?;
        if group_order > cap {
            return Err(CharacterError::OrderTooLarge {
                order: group_order.to_string(),
                cap,
            });
        }

        let degree = group.degree();
        let mut elements = group.elements_with_cap(cap)?;
        elements.sort_unstable();
        elements.dedup();
        if group_order != elements.len() {
            return Err(CharacterError::Internal {
                detail: format!(
                    "element enumeration produced {} distinct elements for a group of order {}",
                    elements.len(),
                    group_order
                ),
            });
        }

        // Conjugation orbits. `g^x = x⁻¹ g x`; conjugating by the generators
        // alone reaches the whole orbit because conjugation is a group action.
        let inverse_generators: Vec<Permutation> =
            group.generators().iter().map(|g| g.inverse()).collect();

        let unassigned = usize::MAX;
        let mut class_of = vec![unassigned; elements.len()];
        let mut raw_members: Vec<Vec<usize>> = Vec::new();

        for seed in 0..elements.len() {
            if class_of[seed] != unassigned {
                continue;
            }
            let label = raw_members.len();
            let mut orbit = vec![seed];
            class_of[seed] = label;
            let mut frontier = 0usize;
            while frontier < orbit.len() {
                let current = elements[orbit[frontier]].clone();
                frontier += 1;
                for (g, g_inv) in group.generators().iter().zip(&inverse_generators) {
                    let conjugate = g_inv.compose(&current)?.compose(g)?;
                    let position = Self::locate(&elements, &conjugate).ok_or_else(|| {
                        CharacterError::Internal {
                            detail: format!(
                                "the conjugate {conjugate} of a group element is not in the \
                                 element list"
                            ),
                        }
                    })?;
                    if class_of[position] == unassigned {
                        class_of[position] = label;
                        orbit.push(position);
                    }
                }
            }
            raw_members.push(orbit);
        }

        // Deterministic ordering: (element order, class size, representative).
        let mut summaries: Vec<(u64, u64, usize, usize)> = Vec::with_capacity(raw_members.len());
        for (label, orbit) in raw_members.iter().enumerate() {
            let rep_index = *orbit.iter().min().expect("an orbit is never empty");
            let order = to_u64(&elements[rep_index].order(), "an element order")?;
            summaries.push((order, orbit.len() as u64, rep_index, label));
        }
        summaries.sort_by(|a, b| match a.0.cmp(&b.0) {
            Ordering::Equal => match a.1.cmp(&b.1) {
                Ordering::Equal => elements[a.2].cmp(&elements[b.2]),
                other => other,
            },
            other => other,
        });

        let mut classes = Vec::with_capacity(summaries.len());
        let mut members: Vec<Vec<usize>> = Vec::with_capacity(summaries.len());
        let mut relabel = vec![0usize; raw_members.len()];
        let mut exponent = Integer::from(1);
        for (new_label, &(order, size, rep_index, old_label)) in summaries.iter().enumerate() {
            relabel[old_label] = new_label;
            if !group_order.is_divisible(&Integer::from(size)) {
                return Err(CharacterError::SelfCheckFailed {
                    check: "class sizes",
                    detail: format!(
                        "class {new_label} has size {size}, which does not divide |G| = \
                         {group_order}"
                    ),
                });
            }
            classes.push(ConjugacyClass {
                representative: elements[rep_index].clone(),
                size,
                centraliser_order: Integer::from(&group_order) / Integer::from(size),
                element_order: order,
            });
            members.push(Vec::new());
            exponent.lcm_mut(&Integer::from(order));
        }
        for label in class_of.iter_mut() {
            *label = relabel[*label];
        }
        for (new_label, &(_, _, _, old_label)) in summaries.iter().enumerate() {
            members[new_label] = std::mem::take(&mut raw_members[old_label]);
        }

        let total = classes
            .iter()
            .fold(Integer::new(), |acc, c| acc + Integer::from(c.size));
        if total != group_order {
            return Err(CharacterError::SelfCheckFailed {
                check: "class sizes",
                detail: format!("the class sizes sum to {total}, not to |G| = {group_order}"),
            });
        }

        let exponent = to_u64(&exponent, "exp G")?;

        let mut inverse_class = vec![0usize; classes.len()];
        for (i, class) in classes.iter().enumerate() {
            let inverse = class.representative.inverse();
            let position =
                Self::locate(&elements, &inverse).ok_or_else(|| CharacterError::Internal {
                    detail: format!("the inverse {inverse} of a group element is not in the list"),
                })?;
            inverse_class[i] = class_of[position];
        }

        Ok(ConjugacyClasses {
            degree,
            group_order,
            exponent,
            classes,
            elements,
            class_of,
            members,
            inverse_class,
        })
    }

    fn locate(elements: &[Permutation], target: &Permutation) -> Option<usize> {
        elements.binary_search(target).ok()
    }

    /// The degree of the underlying permutation group.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// `|G|`, exactly.
    pub fn group_order(&self) -> &Integer {
        &self.group_order
    }

    /// `exp G`, the lcm of the element orders. Always divides `|G|`.
    pub fn exponent(&self) -> u64 {
        self.exponent
    }

    /// The number of conjugacy classes — equivalently, the number of
    /// irreducible characters.
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// A group always has at least the identity class, so this is never true;
    /// it exists because clippy asks for it next to `len`.
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// The classes, in the deterministic order described on
    /// [`ConjugacyClasses::of`]. Class `0` is `{1}`.
    pub fn classes(&self) -> &[ConjugacyClass] {
        &self.classes
    }

    /// The class at `index`.
    pub fn class(&self, index: usize) -> Result<&ConjugacyClass, CharacterError> {
        self.classes
            .get(index)
            .ok_or(CharacterError::ClassIndexOutOfRange {
                index,
                classes: self.classes.len(),
            })
    }

    /// The index of the class containing `element`.
    ///
    /// Refuses with [`CharacterError::NotAMember`] for an element outside the
    /// group, and with `E-GRP-002` (through [`CharacterError::Group`]) for one
    /// of the wrong degree.
    pub fn class_of(&self, element: &Permutation) -> Result<usize, CharacterError> {
        if element.degree() != self.degree {
            return Err(CharacterError::Group(
                crate::group::GroupError::DegreeMismatch {
                    left: self.degree,
                    right: element.degree(),
                },
            ));
        }
        match Self::locate(&self.elements, element) {
            Some(position) => Ok(self.class_of[position]),
            None => Err(CharacterError::NotAMember {
                element: element.to_string(),
            }),
        }
    }

    /// The index of the class of `g⁻¹`, for `g` in class `index`.
    ///
    /// This is how complex conjugation reaches this module: `χ(g⁻¹) =
    /// conj(χ(g))` for every character, so a conjugate value is a table lookup
    /// rather than a field automorphism.
    pub fn inverse_class(&self, index: usize) -> Result<usize, CharacterError> {
        self.inverse_class
            .get(index)
            .copied()
            .ok_or(CharacterError::ClassIndexOutOfRange {
                index,
                classes: self.classes.len(),
            })
    }

    /// Every element of class `index`, in sorted order.
    ///
    /// The full list; this is `|K|` permutations, and the whole group is
    /// already in memory by the time a `ConjugacyClasses` exists.
    pub fn class_elements(&self, index: usize) -> Result<Vec<Permutation>, CharacterError> {
        let members = self
            .members
            .get(index)
            .ok_or(CharacterError::ClassIndexOutOfRange {
                index,
                classes: self.classes.len(),
            })?;
        let mut out: Vec<Permutation> = members.iter().map(|&i| self.elements[i].clone()).collect();
        out.sort_unstable();
        Ok(out)
    }

    /// The class multiplication coefficient
    /// `a_{ijk} = #{ (x, y) ∈ K_i × K_j : x·y = g_k }`, where `g_k` is the
    /// representative of class `k`.
    ///
    /// The count does not depend on which `g_k` is chosen, which is what makes
    /// the class sums `K̂_i` a basis of the centre of the group algebra with
    /// `K̂_i · K̂_j = Σ_k a_{ijk} K̂_k`. Products are formed with
    /// [`Permutation::compose`], i.e. **left to right**: `x·y` applies `x`
    /// first.
    pub fn multiplication_coefficient(
        &self,
        i: usize,
        j: usize,
        k: usize,
    ) -> Result<u64, CharacterError> {
        let n = self.classes.len();
        for &index in &[i, j, k] {
            if index >= n {
                return Err(CharacterError::ClassIndexOutOfRange { index, classes: n });
            }
        }
        let target = &self.classes[k].representative;
        let mut count = 0u64;
        for &member in &self.members[i] {
            let x = &self.elements[member];
            let y = x.inverse().compose(target)?;
            let position =
                Self::locate(&self.elements, &y).ok_or_else(|| CharacterError::Internal {
                    detail: format!("the product {y} of two group elements left the group"),
                })?;
            if self.class_of[position] == j {
                count += 1;
            }
        }
        Ok(count)
    }

    /// The class multiplication matrix `M_k`, with `M_k[i][j] = a_{kij}`.
    ///
    /// This is the matrix whose common eigenvectors Dixon's algorithm splits:
    /// `M_k · ω = ω_k · ω` for `ω_i = |K_i| χ(g_i) / χ(1)`. One pass over
    /// `K_k` per column, so `O(r · |K_k|)` products rather than the `O(r³)` of
    /// calling [`ConjugacyClasses::multiplication_coefficient`] entrywise.
    pub fn multiplication_matrix(&self, k: usize) -> Result<Vec<Vec<u64>>, CharacterError> {
        let n = self.classes.len();
        if k >= n {
            return Err(CharacterError::ClassIndexOutOfRange {
                index: k,
                classes: n,
            });
        }
        let mut matrix = vec![vec![0u64; n]; n];
        for (j, class) in self.classes.iter().enumerate() {
            let target = &class.representative;
            for &member in &self.members[k] {
                let u = &self.elements[member];
                let v = u.inverse().compose(target)?;
                let position =
                    Self::locate(&self.elements, &v).ok_or_else(|| CharacterError::Internal {
                        detail: format!("the product {v} of two group elements left the group"),
                    })?;
                matrix[self.class_of[position]][j] += 1;
            }
        }
        Ok(matrix)
    }
}

/// A `rug::Integer` that is known to fit a `u64`, or a typed refusal.
pub(super) fn to_u64(value: &Integer, what: &str) -> Result<u64, CharacterError> {
    value.to_u64().ok_or_else(|| CharacterError::Internal {
        detail: format!("{what} = {value} does not fit a 64-bit word"),
    })
}
