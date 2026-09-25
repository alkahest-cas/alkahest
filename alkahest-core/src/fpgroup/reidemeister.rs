//! Reidemeister–Schreier: a presentation for a finite-index subgroup.
//!
//! Given a complete coset table for `H ≤ G`, the **Schreier generators** of `H`
//! are the elements `u_c · x · u_{c·x}⁻¹` for each coset `c` and each generator
//! `x` of `G`, where `u_c` is the transversal word reaching `c`. The `|G:H| − 1`
//! pairs `(c, x)` that are the edges of the transversal's spanning tree give the
//! identity and are dropped, leaving `|G:H|·|X| − |G:H| + 1` generators — the
//! Nielsen–Schreier count, which for a free `G` is exactly the rank of the free
//! subgroup.
//!
//! The relators are obtained by **rewriting**: trace each relator of `G` from
//! each coset and record the Schreier generator crossed at every step. Because
//! the relator closes (`c·r = c`), the recorded word equals `u_c r u_c⁻¹`, which
//! is trivial in `G` and lies in `H`, so it is a relator of `H`; and the
//! `|G:H|·|R|` words so obtained, together with the generators above, present
//! `H`.
//!
//! # No simplification
//!
//! The presentation returned is the raw one. Duplicate relators are removed and
//! empty ones dropped, but there is **no Tietze transformation pass**: no
//! generator elimination, no relator shortening. A subgroup of index 30 in a
//! 2-generator group therefore arrives with 31 generators and up to 90 relators
//! even when it is cyclic of order 2. That is a real limitation — the usual way
//! to check one of these presentations is to run Todd–Coxeter on it and compare
//! `|H|` with `|G| / [G:H]`, which is what this module's tests do.

use super::error::FpGroupError;
use super::presentation::FpGroup;
use super::word::{FreeGroup, Word};

/// A presentation for a finite-index subgroup, together with each of its
/// generators written as a word in the parent group's generators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubgroupPresentation {
    presentation: FpGroup,
    generator_words: Vec<Word>,
    index: usize,
}

impl SubgroupPresentation {
    /// The presentation of `H`, on generators named `y1, y2, …`.
    pub fn presentation(&self) -> &FpGroup {
        &self.presentation
    }

    /// Generator `k` of the subgroup presentation, as a word in the **parent**
    /// group's generators. Same order as the presentation's generators.
    pub fn generator_words(&self) -> &[Word] {
        &self.generator_words
    }

    /// `[G:H]`.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The number of Schreier generators, `[G:H]·|X| − [G:H] + 1`.
    pub fn rank(&self) -> usize {
        self.generator_words.len()
    }
}

/// Build a presentation for `H = ⟨subgroup_generators⟩ ≤ G`.
///
/// Refuses with [`FpGroupError::EnumerationIncomplete`] if the coset
/// enumeration does not complete — a subgroup of infinite index has no
/// Reidemeister–Schreier presentation *of this kind*, and one whose index is
/// merely large is indistinguishable from it at the cap.
pub fn reidemeister_schreier(
    group: &FpGroup,
    subgroup_generators: &[Word],
    max_cosets: usize,
) -> Result<SubgroupPresentation, FpGroupError> {
    let table = group.coset_table_with_cap(subgroup_generators, max_cosets)?;
    let index = table.index();
    let rank = group.rank();
    let tree = table.spanning_tree();
    let transversal = table.transversal()?;

    // Number the non-tree pairs (coset, generator); those are the generators.
    let mut generator_of: Vec<Option<usize>> = vec![None; index * rank.max(1)];
    let mut generator_words: Vec<Word> = Vec::new();
    for c in 0..index {
        for i in 0..rank {
            let letter = i as i32 + 1;
            let d = table.image(c, letter)?;
            let forward_tree_edge = tree[d] == Some((c, 2 * i));
            let backward_tree_edge = tree[c] == Some((d, 2 * i + 1));
            if forward_tree_edge || backward_tree_edge {
                continue;
            }
            let w = transversal[c]
                .times(&Word::generator(i)?)
                .times(&transversal[d].inverse());
            generator_of[c * rank + i] = Some(generator_words.len());
            generator_words.push(w);
        }
    }

    // Exactly one pair is killed per spanning-tree edge, and there are
    // `index - 1` of those; a mismatch means the tree and the table disagree.
    if generator_words.len() + (index - 1) != index * rank {
        return Err(FpGroupError::Internal {
            detail: format!(
                "expected {} Schreier generators, built {}",
                index * rank - (index - 1),
                generator_words.len()
            ),
        });
    }

    // Rewrite each relator from each coset.
    let mut relators: Vec<Word> = Vec::new();
    for c in 0..index {
        for r in group.relators() {
            let mut letters: Vec<i32> = Vec::new();
            let mut a = c;
            for &l in r.letters() {
                let b = table.image(a, l)?;
                let i = l.unsigned_abs() as usize - 1;
                if l > 0 {
                    if let Some(id) = generator_of[a * rank + i] {
                        letters.push(id as i32 + 1);
                    }
                } else if let Some(id) = generator_of[b * rank + i] {
                    letters.push(-(id as i32 + 1));
                }
                a = b;
            }
            if a != c {
                return Err(FpGroupError::Internal {
                    detail: format!("relator {r} did not close at coset {c} while rewriting"),
                });
            }
            let w = Word::from_letters(&letters)?;
            if !w.is_empty() && !relators.contains(&w) {
                relators.push(w);
            }
        }
    }

    let names: Vec<String> = (1..=generator_words.len())
        .map(|k| format!("y{k}"))
        .collect();
    let free = FreeGroup::with_names(&names)?;
    let presentation = FpGroup::new(free, relators)?;
    Ok(SubgroupPresentation {
        presentation,
        generator_words,
        index,
    })
}
