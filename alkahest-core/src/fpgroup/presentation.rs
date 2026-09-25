//! [`FpGroup`] — a presentation `⟨X | R⟩`, and everything read off it.

use super::abelian::{quotient_invariants, AbelianInvariants};
use super::cohomology::GModule;
use super::error::FpGroupError;
use super::reidemeister::SubgroupPresentation;
use super::todd_coxeter::{self, default_max_cosets, CosetTable};
use super::word::{FreeGroup, Word};
use crate::group::PermutationGroup;
use rug::Integer;
use std::fmt;

/// A finitely presented group `⟨X | R⟩`: a free group and a list of relators.
///
/// A *relator* is a word that is trivial in the quotient. Relators are stored
/// freely reduced; an empty relator says nothing and is kept out of the
/// enumeration.
///
/// Almost every question about an `FpGroup` is undecidable in general, so the
/// methods here divide sharply into two kinds, and the return types say which
/// is which:
///
/// * **Always terminates.** [`abelian_invariants`](FpGroup::abelian_invariants)
///   and everything derived from it — it is a Smith normal form of an
///   `|R| × |X|` integer matrix.
/// * **May refuse.** [`order`](FpGroup::order), [`index`](FpGroup::index),
///   [`coset_table`](FpGroup::coset_table),
///   [`permutation_group`](FpGroup::permutation_group),
///   [`subgroup_presentation`](FpGroup::subgroup_presentation) and the
///   cohomology — all of these run Todd–Coxeter enumeration, which has no
///   termination guarantee, and refuse with
///   [`FpGroupError::EnumerationIncomplete`] at the cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FpGroup {
    free: FreeGroup,
    relators: Vec<Word>,
}

impl FpGroup {
    /// A presentation on the given free group with the given relators.
    pub fn new(free: FreeGroup, relators: Vec<Word>) -> Result<FpGroup, FpGroupError> {
        let rank = free.rank();
        for r in &relators {
            if let Some(g) = r.max_generator() {
                if g >= rank {
                    return Err(FpGroupError::InvalidGenerator {
                        letter: g as i32 + 1,
                        rank,
                    });
                }
            }
        }
        Ok(FpGroup { free, relators })
    }

    /// A free group of the given rank, presented with no relators.
    pub fn free(rank: usize) -> Result<FpGroup, FpGroupError> {
        Ok(FpGroup {
            free: FreeGroup::new(rank)?,
            relators: Vec::new(),
        })
    }

    /// `⟨generators | relators⟩` from text, e.g.
    ///
    /// ```
    /// use alkahest_cas::experimental::FpGroup;
    /// let a5 = FpGroup::from_strings(&["a", "b"], &["a^2", "b^3", "(a*b)^5"]).unwrap();
    /// assert_eq!(a5.order().unwrap(), 60);
    /// ```
    pub fn from_strings<S: AsRef<str>, T: AsRef<str>>(
        generators: &[S],
        relators: &[T],
    ) -> Result<FpGroup, FpGroupError> {
        let free = FreeGroup::with_names(generators)?;
        let mut rels = Vec::with_capacity(relators.len());
        for r in relators {
            rels.push(free.parse(r.as_ref())?);
        }
        FpGroup::new(free, rels)
    }

    /// The underlying free group.
    pub fn free_group(&self) -> &FreeGroup {
        &self.free
    }

    /// The number of generators.
    pub fn rank(&self) -> usize {
        self.free.rank()
    }

    /// The generator names.
    pub fn generator_names(&self) -> &[String] {
        self.free.names()
    }

    /// The relators, freely reduced.
    pub fn relators(&self) -> &[Word] {
        &self.relators
    }

    /// Parse a word in this presentation's generators.
    pub fn parse_word(&self, text: &str) -> Result<Word, FpGroupError> {
        self.free.parse(text)
    }

    /// A word from signed 1-based letters, checked against the rank.
    pub fn word(&self, letters: &[i32]) -> Result<Word, FpGroupError> {
        self.free.word(letters)
    }

    /// Parse several words at once — a subgroup's generators, usually.
    pub fn parse_words<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Word>, FpGroupError> {
        texts.iter().map(|t| self.free.parse(t.as_ref())).collect()
    }

    // -----------------------------------------------------------------------
    // Coset enumeration
    // -----------------------------------------------------------------------

    /// The coset table of `H = ⟨subgroup_generators⟩`, with the default cap —
    /// [`super::DEFAULT_MAX_COSETS`], lowered at large rank so that the table
    /// itself stays inside [`super::MAX_COSET_TABLE_CELLS`] words.
    pub fn coset_table(&self, subgroup_generators: &[Word]) -> Result<CosetTable, FpGroupError> {
        self.coset_table_with_cap(subgroup_generators, default_max_cosets(self.rank()))
    }

    /// The coset table of `H = ⟨subgroup_generators⟩`, refusing above
    /// `max_cosets` cosets.
    ///
    /// The refusal ([`FpGroupError::EnumerationIncomplete`]) means *"did not
    /// complete"*, never "the group is infinite".
    pub fn coset_table_with_cap(
        &self,
        subgroup_generators: &[Word],
        max_cosets: usize,
    ) -> Result<CosetTable, FpGroupError> {
        todd_coxeter::enumerate(self.rank(), &self.relators, subgroup_generators, max_cosets)
    }

    /// `[G:H]` for `H = ⟨subgroup_generators⟩`.
    pub fn index(&self, subgroup_generators: &[Word]) -> Result<usize, FpGroupError> {
        Ok(self.coset_table(subgroup_generators)?.index())
    }

    /// `[G:H]`, refusing above `max_cosets` cosets.
    pub fn index_with_cap(
        &self,
        subgroup_generators: &[Word],
        max_cosets: usize,
    ) -> Result<usize, FpGroupError> {
        Ok(self
            .coset_table_with_cap(subgroup_generators, max_cosets)?
            .index())
    }

    /// `|G|` — the index of the trivial subgroup.
    ///
    /// Two different failures are possible and they are **not** the same fact:
    ///
    /// * [`FpGroupError::ProvablyInfinite`] (`E-FPGRP-005`) — the abelianisation
    ///   has an infinite cyclic factor, so `G` is infinite. This is checked
    ///   first, because it is cheap and it always terminates.
    /// * [`FpGroupError::EnumerationIncomplete`] (`E-FPGRP-004`) — the
    ///   enumeration ran out of cosets. This says nothing about whether `G` is
    ///   finite. `⟨a, b | a², b³, (ab)⁷⟩` is infinite with a trivial
    ///   abelianisation, and lands here; so would a finite group of order `10¹⁰`.
    pub fn order(&self) -> Result<Integer, FpGroupError> {
        self.order_with_cap(default_max_cosets(self.rank()))
    }

    /// `|G|`, refusing above `max_cosets` cosets.
    pub fn order_with_cap(&self, max_cosets: usize) -> Result<Integer, FpGroupError> {
        let abelian = self.abelian_invariants()?;
        if abelian.free_rank() > 0 {
            return Err(FpGroupError::ProvablyInfinite {
                abelian_invariants: abelian.to_string(),
            });
        }
        let table = self.coset_table_with_cap(&[], max_cosets)?;
        Ok(Integer::from(table.index()))
    }

    /// The permutation representation of `G` on the cosets of
    /// `⟨subgroup_generators⟩`, as a [`PermutationGroup`].
    ///
    /// Degree `[G:H]`. The kernel is the core of `H`, so this is faithful
    /// exactly when that core is trivial — in particular for `H = 1`, where it
    /// is the regular representation and
    /// [`PermutationGroup::order`] independently recomputes `|G|` by
    /// Schreier–Sims.
    pub fn permutation_group(
        &self,
        subgroup_generators: &[Word],
    ) -> Result<PermutationGroup, FpGroupError> {
        self.coset_table(subgroup_generators)?.permutation_group()
    }

    /// The regular representation: `G` acting on itself by right
    /// multiplication, of degree `|G|`.
    pub fn regular_representation(&self) -> Result<PermutationGroup, FpGroupError> {
        self.permutation_group(&[])
    }

    /// The multiplication table of a finite `G`, indexed by the cosets of the
    /// trivial subgroup — so element `0` is the identity and element `c` is the
    /// one whose transversal word reaches coset `c`.
    ///
    /// `table[a][b]` is the index of `g_a · g_b`. Needed by the cohomology, and
    /// exposed because a Latin-square check on it is a cheap independent test of
    /// the enumeration.
    pub fn multiplication_table(&self) -> Result<Vec<Vec<usize>>, FpGroupError> {
        let table = self.coset_table(&[])?;
        let n = table.index();
        let transversal = table.transversal()?;
        let mut mult = vec![vec![0usize; n]; n];
        for (a, row) in mult.iter_mut().enumerate() {
            for (b, cell) in row.iter_mut().enumerate() {
                *cell = table.trace(a, &transversal[b])?;
            }
        }
        Ok(mult)
    }

    // -----------------------------------------------------------------------
    // Reidemeister-Schreier
    // -----------------------------------------------------------------------

    /// A presentation for the subgroup `H = ⟨subgroup_generators⟩`, by
    /// Reidemeister–Schreier rewriting.
    ///
    /// Needs `H` to have finite index, and refuses with
    /// [`FpGroupError::EnumerationIncomplete`] when the enumeration does not
    /// complete — which, as everywhere here, is not a claim that the index is
    /// infinite.
    ///
    /// The presentation is **not simplified**; see [`SubgroupPresentation`].
    pub fn subgroup_presentation(
        &self,
        subgroup_generators: &[Word],
    ) -> Result<SubgroupPresentation, FpGroupError> {
        self.subgroup_presentation_with_cap(subgroup_generators, default_max_cosets(self.rank()))
    }

    /// [`subgroup_presentation`](FpGroup::subgroup_presentation) with an explicit
    /// coset cap.
    pub fn subgroup_presentation_with_cap(
        &self,
        subgroup_generators: &[Word],
        max_cosets: usize,
    ) -> Result<SubgroupPresentation, FpGroupError> {
        super::reidemeister::reidemeister_schreier(self, subgroup_generators, max_cosets)
    }

    // -----------------------------------------------------------------------
    // Abelianisation
    // -----------------------------------------------------------------------

    /// The relation matrix: one row per relator, one column per generator, entry
    /// the exponent sum.
    ///
    /// This is the presentation of `G/[G, G]` as `ℤ^X / ⟨rows⟩`; abelianising
    /// discards the order of the letters and keeps only how many times each
    /// generator occurs.
    pub fn relation_matrix(&self) -> Result<Vec<Vec<i64>>, FpGroupError> {
        self.relators
            .iter()
            .map(|r| r.exponent_sums(self.rank()))
            .collect()
    }

    /// `G/[G, G]` as a direct sum of cyclic groups, from the Smith normal form
    /// of the relation matrix.
    ///
    /// **This always terminates** — it is linear algebra over `ℤ`, not coset
    /// enumeration. An infinite cyclic factor here is a proof that `G` itself is
    /// infinite, which is what [`order`](FpGroup::order) uses to refuse with
    /// [`FpGroupError::ProvablyInfinite`] instead of burning the coset cap.
    ///
    /// The converse does not hold: `A₅` is perfect, so a trivial abelianisation
    /// is no evidence of finiteness.
    pub fn abelian_invariants(&self) -> Result<AbelianInvariants, FpGroupError> {
        let rank = self.rank();
        if rank == 0 {
            return Ok(AbelianInvariants::trivial());
        }
        let rows = self.relation_matrix()?;
        let gens: Vec<Vec<Integer>> = (0..rank)
            .map(|j| {
                (0..rank)
                    .map(|i| Integer::from(usize::from(i == j)))
                    .collect()
            })
            .collect();
        let relations: Vec<Vec<Integer>> = rows
            .iter()
            .map(|r| r.iter().map(|&e| Integer::from(e)).collect())
            .collect();
        quotient_invariants(rank, &gens, &relations)
    }

    // -----------------------------------------------------------------------
    // Cohomology
    // -----------------------------------------------------------------------

    /// `H^degree(G, M)` for a finite `G` and a finitely generated abelian module
    /// `M`, from the inhomogeneous bar resolution.
    ///
    /// Only degrees `0`, `1` and `2` exist here; `H²` is the one that classifies
    /// extensions of `M` by `G`. See [`super::cohomology`] for the size limits,
    /// the conventions, and what is checked before an answer is returned.
    pub fn cohomology(
        &self,
        degree: usize,
        module: &GModule,
    ) -> Result<AbelianInvariants, FpGroupError> {
        super::cohomology::cohomology(self, degree, module)
    }
}

impl fmt::Display for FpGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let gens = self.free.names().join(", ");
        if self.relators.is_empty() {
            return write!(f, "⟨{gens} | ⟩");
        }
        let rels: Vec<String> = self.relators.iter().map(|r| self.free.format(r)).collect();
        write!(f, "⟨{gens} | {}⟩", rels.join(", "))
    }
}
