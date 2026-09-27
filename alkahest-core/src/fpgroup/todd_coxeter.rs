//! Todd–Coxeter coset enumeration, and the [`CosetTable`] it produces.
//!
//! # The strategy
//!
//! This is **HLT** (Haselgrove–Leech–Trotter) with **lookahead**: the main loop
//! walks the cosets in order, scans every relator at each one filling in blanks
//! as it goes, and then completes that coset's row with new definitions. When
//! the table fills up, a lookahead pass scans every relator at every live coset
//! *without* filling anything, which can only produce coincidences, and the dead
//! cosets are then compacted away. If the pass frees enough space the
//! enumeration resumes; if it does not, the enumeration refuses.
//!
//! HLT is chosen over Felsch because its failure mode is the honest one: it
//! defines cosets eagerly, so it fills the table quickly on an infinite group
//! and reaches the refusal instead of grinding. Felsch's deduction stack is
//! more economical in table space on many presentations and is the better
//! choice for a production enumerator; it is not implemented here.
//!
//! # Coincidence handling
//!
//! Coincidences — the discovery that two coset numbers name the same coset —
//! are where these implementations go wrong, because a coincidence can cascade:
//! merging `α` into `β` can force further merges through every entry of `α`'s
//! row, and those merges can force more. The handling here is the standard
//! queue-driven one (Holt, *Handbook of Computational Group Theory*, §5.2):
//!
//! * A union-find array `p` with the invariant `p[α] ≤ α` maps every dead coset
//!   to its live representative, so the smallest number in a class always
//!   survives and coset `0` (the subgroup `H`) is never killed.
//! * Merging pushes the dead coset onto a queue rather than recursing, and the
//!   queue is drained to fixpoint, so a cascade of any depth is handled
//!   iteratively.
//! * Draining a dead coset `α` relinks each of its edges onto the
//!   representatives, deleting the reverse edge first so that no live row is
//!   ever left pointing at a dead coset.
//!
//! # Termination, and what a refusal means
//!
//! The word problem for finitely presented groups is undecidable, so **this
//! function can refuse**. `⟨a, b | a², b³, (ab)⁷⟩` is infinite and no cap will
//! complete it. A [`FpGroupError::EnumerationIncomplete`] (`E-FPGRP-004`) means
//! only *"not finished within this cap"* — it is **not** a claim that the group
//! is infinite. That distinction is the whole point of the error being typed:
//! see the module docs of [`FpGroupError`].
//!
//! # The answer is checked before it is returned
//!
//! When the loop finishes, the table is verified: every row complete, every
//! edge's reverse edge present, every relator scanning closed at every coset,
//! and every subgroup generator closed at coset `0`. Those are the defining
//! properties of *the* coset table of `H` in `G`, so a table that passes them
//! is right and a table that does not raises
//! [`FpGroupError::Internal`] rather than being returned. The check costs one
//! pass and removes the possibility of a confidently wrong index.

use super::error::FpGroupError;
use super::word::Word;
use crate::group::{Permutation, PermutationGroup};

/// Default ceiling on the number of cosets an enumeration may define.
///
/// Reached in well under a second on a two-generator presentation, which is
/// what makes it a usable default for "is this group small?".
pub const DEFAULT_MAX_COSETS: usize = 200_000;

/// Ceiling on `max_cosets · 2 · rank` — the number of machine words the table
/// may occupy. 64 million words is 512 MB.
pub const MAX_COSET_TABLE_CELLS: usize = 64_000_000;

/// How many lookahead-and-compact rounds are attempted before refusing.
const MAX_LOOKAHEAD_ROUNDS: usize = 8;

/// The default coset cap at a given rank.
///
/// [`DEFAULT_MAX_COSETS`], except where that many rows of `2·rank` words would
/// exceed [`MAX_COSET_TABLE_CELLS`]. Without this, the default cap would itself
/// be refused as too large on a presentation of four-figure rank — which is
/// exactly what Reidemeister–Schreier produces from a subgroup of large index,
/// so it is reachable without anyone asking for it.
pub fn default_max_cosets(rank: usize) -> usize {
    let ceiling = MAX_COSET_TABLE_CELLS / (2 * rank.max(1));
    DEFAULT_MAX_COSETS.min(ceiling).max(1)
}

/// The enumeration gave up on space; the driver decides what to do next.
struct Full;

/// A letter (`±(k+1)`) as a table column: `2k` for the generator, `2k+1` for its
/// inverse.
fn column_of(letter: i32) -> usize {
    let g = letter.unsigned_abs() as usize - 1;
    if letter > 0 {
        2 * g
    } else {
        2 * g + 1
    }
}

/// The signed letter a column stands for.
fn letter_of(column: usize) -> i32 {
    let g = (column / 2) as i32 + 1;
    if column % 2 == 0 {
        g
    } else {
        -g
    }
}

fn columns_of(w: &Word) -> Vec<usize> {
    w.letters().iter().map(|&l| column_of(l)).collect()
}

// ---------------------------------------------------------------------------
// The enumerator
// ---------------------------------------------------------------------------

struct Enumerator {
    ncols: usize,
    /// Flat `(n + 1) × ncols`; row 0 is unused so that `0` can mean undefined.
    table: Vec<usize>,
    /// Union-find over coset numbers with the invariant `p[α] ≤ α`.
    p: Vec<usize>,
    n: usize,
    cap: usize,
    coincidences: usize,
    /// Bumped by every table write and every merge; the closure test uses it to
    /// tell "nothing changed" from "something did".
    changes: u64,
}

impl Enumerator {
    fn new(rank: usize, cap: usize) -> Enumerator {
        let ncols = 2 * rank;
        Enumerator {
            ncols,
            table: vec![0; 2 * ncols],
            p: vec![0, 1],
            n: 1,
            cap,
            coincidences: 0,
            changes: 0,
        }
    }

    #[inline]
    fn get(&self, a: usize, x: usize) -> usize {
        self.table[a * self.ncols + x]
    }

    #[inline]
    fn set(&mut self, a: usize, x: usize, v: usize) {
        let slot = a * self.ncols + x;
        if self.table[slot] != v {
            self.table[slot] = v;
            self.changes += 1;
        }
    }

    /// The live representative of `k`, with path compression.
    fn rep(&mut self, k: usize) -> usize {
        let mut l = k;
        while self.p[l] != l {
            l = self.p[l];
        }
        let mut m = k;
        while self.p[m] != l {
            let next = self.p[m];
            self.p[m] = l;
            m = next;
        }
        l
    }

    #[inline]
    fn is_alive(&self, a: usize) -> bool {
        self.p[a] == a
    }

    fn live_count(&self) -> usize {
        (1..=self.n).filter(|&a| self.is_alive(a)).count()
    }

    fn define(&mut self, a: usize, x: usize) -> Result<usize, Full> {
        if self.n >= self.cap {
            return Err(Full);
        }
        self.n += 1;
        let b = self.n;
        self.table.resize(self.table.len() + self.ncols, 0);
        self.p.push(b);
        self.set(a, x, b);
        self.set(b, x ^ 1, a);
        Ok(b)
    }

    /// Record that `a` and `b` are the same coset; the *larger* number dies.
    fn merge(&mut self, a: usize, b: usize, queue: &mut Vec<usize>) {
        let mut a = self.rep(a);
        let mut b = self.rep(b);
        if a == b {
            return;
        }
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        self.p[b] = a;
        self.coincidences += 1;
        self.changes += 1;
        queue.push(b);
    }

    /// Process the coincidence `a = b` to fixpoint.
    fn coincidence(&mut self, a: usize, b: usize) {
        let mut queue: Vec<usize> = Vec::new();
        self.merge(a, b, &mut queue);
        let mut head = 0;
        while head < queue.len() {
            let dead = queue[head];
            head += 1;
            for x in 0..self.ncols {
                let d = self.get(dead, x);
                if d == 0 {
                    continue;
                }
                // Drop the reverse edge before relinking: a live row must never
                // be left pointing at a dead coset.
                self.set(d, x ^ 1, 0);
                let mu = self.rep(dead);
                let nu = self.rep(d);
                let forward = self.get(mu, x);
                if forward != 0 {
                    self.merge(nu, forward, &mut queue);
                } else {
                    let backward = self.get(nu, x ^ 1);
                    if backward != 0 {
                        self.merge(mu, backward, &mut queue);
                    } else {
                        self.set(mu, x, nu);
                        self.set(nu, x ^ 1, mu);
                    }
                }
            }
        }
    }

    /// Scan `w` at coset `a`, filling in undefined entries (defining new cosets
    /// where the two ends of the scan cannot yet meet).
    fn scan_and_fill(&mut self, a: usize, w: &[usize]) -> Result<(), Full> {
        if w.is_empty() {
            return Ok(());
        }
        let mut f = a;
        let mut b = a;
        let mut i: isize = 0;
        let mut j: isize = w.len() as isize - 1;
        loop {
            while i <= j {
                let t = self.get(f, w[i as usize]);
                if t == 0 {
                    break;
                }
                f = t;
                i += 1;
            }
            if i > j {
                if f != b {
                    self.coincidence(f, b);
                }
                return Ok(());
            }
            while j >= i {
                let t = self.get(b, w[j as usize] ^ 1);
                if t == 0 {
                    break;
                }
                b = t;
                j -= 1;
            }
            if j < i {
                self.coincidence(f, b);
                return Ok(());
            }
            if j == i {
                let x = w[i as usize];
                self.set(f, x, b);
                self.set(b, x ^ 1, f);
                return Ok(());
            }
            self.define(f, w[i as usize])?;
        }
    }

    /// Scan `w` at coset `a` **without** defining anything.
    ///
    /// Returns `true` if the scan closed (either trivially or through a
    /// coincidence it then applied). A `false` means the scan was incomplete
    /// and carried no information.
    fn scan(&mut self, a: usize, w: &[usize]) -> bool {
        if w.is_empty() {
            return true;
        }
        let mut f = a;
        let mut b = a;
        let mut i: isize = 0;
        let mut j: isize = w.len() as isize - 1;
        while i <= j {
            let t = self.get(f, w[i as usize]);
            if t == 0 {
                break;
            }
            f = t;
            i += 1;
        }
        if i > j {
            if f != b {
                self.coincidence(f, b);
            }
            return true;
        }
        while j >= i {
            let t = self.get(b, w[j as usize] ^ 1);
            if t == 0 {
                break;
            }
            b = t;
            j -= 1;
        }
        if j < i {
            self.coincidence(f, b);
            return true;
        }
        if j == i {
            let x = w[i as usize];
            self.set(f, x, b);
            self.set(b, x ^ 1, f);
            return true;
        }
        false
    }

    /// One HLT pass: scan the relators at every coset and complete its row.
    fn hlt_pass(&mut self, relators: &[Vec<usize>], subgens: &[Vec<usize>]) -> Result<(), Full> {
        for w in subgens {
            self.scan_and_fill(1, w)?;
        }
        let mut a = 1;
        while a <= self.n {
            if !self.is_alive(a) {
                a += 1;
                continue;
            }
            for r in relators {
                self.scan_and_fill(a, r)?;
                if !self.is_alive(a) {
                    break;
                }
            }
            if self.is_alive(a) {
                for x in 0..self.ncols {
                    if !self.is_alive(a) {
                        break;
                    }
                    if self.get(a, x) == 0 {
                        self.define(a, x)?;
                    }
                }
            }
            a += 1;
        }
        Ok(())
    }

    /// Lookahead: scan every relator at every live coset without defining.
    /// Cannot grow the table, and may kill a great many cosets.
    fn lookahead(&mut self, relators: &[Vec<usize>]) {
        for a in 1..=self.n {
            if !self.is_alive(a) {
                continue;
            }
            for r in relators {
                self.scan(a, r);
                if !self.is_alive(a) {
                    break;
                }
            }
        }
    }

    /// Renumber the live cosets `1 … m`, discarding the dead rows. Preserves
    /// order, so coset 1 stays coset 1.
    fn compact(&mut self) {
        let mut renumber = vec![0usize; self.n + 1];
        let mut m = 0;
        for (a, slot) in renumber.iter_mut().enumerate().skip(1) {
            if self.p[a] == a {
                m += 1;
                *slot = m;
            }
        }
        let mut table = vec![0usize; (m + 1) * self.ncols];
        for a in 1..=self.n {
            if !self.is_alive(a) {
                continue;
            }
            let na = renumber[a];
            for x in 0..self.ncols {
                let d = self.get(a, x);
                if d != 0 {
                    let rd = self.rep(d);
                    table[na * self.ncols + x] = renumber[rd];
                }
            }
        }
        self.table = table;
        self.n = m;
        self.p = (0..=m).collect();
    }

    /// Is the table closed? Runs one filling pass over everything and reports
    /// whether it changed anything.
    fn closed(&mut self, relators: &[Vec<usize>], subgens: &[Vec<usize>]) -> Result<bool, Full> {
        let before = self.changes;
        for w in subgens {
            self.scan_and_fill(1, w)?;
        }
        let mut a = 1;
        while a <= self.n {
            if self.is_alive(a) {
                for r in relators {
                    self.scan_and_fill(a, r)?;
                    if !self.is_alive(a) {
                        break;
                    }
                }
            }
            a += 1;
        }
        if self.changes != before {
            return Ok(false);
        }
        // Rows must also be complete.
        for a in 1..=self.n {
            if !self.is_alive(a) {
                continue;
            }
            for x in 0..self.ncols {
                if self.get(a, x) == 0 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// The public table
// ---------------------------------------------------------------------------

/// A complete coset table for a subgroup `H ≤ G`, and the permutation
/// representation of `G` on the cosets that it *is*.
///
/// **Cosets are 0-based, and coset `0` is `H`.** Entry `(c, x)` of the table is
/// the coset `Hw·x` where `c = Hw`; the table is complete, so every entry is a
/// coset and every generator acts as a permutation of `0 .. index`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CosetTable {
    rank: usize,
    index: usize,
    /// `index × 2·rank`, 0-based cosets.
    table: Vec<usize>,
    subgroup_generators: Vec<Word>,
    cosets_defined: usize,
    coincidences: usize,
    max_cosets: usize,
}

impl CosetTable {
    /// `[G:H]` — the number of cosets.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The number of generators of the group this is a table for.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The subgroup generators this table was built from.
    pub fn subgroup_generators(&self) -> &[Word] {
        &self.subgroup_generators
    }

    /// Total cosets ever defined, dead ones included — a measure of how much
    /// work the enumeration did relative to its answer.
    pub fn cosets_defined(&self) -> usize {
        self.cosets_defined
    }

    /// How many coincidences were processed.
    pub fn coincidences(&self) -> usize {
        self.coincidences
    }

    /// The coset cap that was in force.
    pub fn max_cosets(&self) -> usize {
        self.max_cosets
    }

    /// The enumeration strategy that produced this table.
    pub fn strategy(&self) -> &'static str {
        "HLT with lookahead"
    }

    /// Row `coset` of the table, indexed by column: `2k` is generator `k` and
    /// `2k+1` its inverse.
    pub fn row(&self, coset: usize) -> Result<&[usize], FpGroupError> {
        if coset >= self.index {
            return Err(FpGroupError::CosetOutOfRange {
                coset,
                index: self.index,
            });
        }
        let w = 2 * self.rank;
        Ok(&self.table[coset * w..coset * w + w])
    }

    /// The whole table as nested rows.
    pub fn rows(&self) -> Vec<Vec<usize>> {
        let w = 2 * self.rank;
        (0..self.index)
            .map(|c| self.table[c * w..c * w + w].to_vec())
            .collect()
    }

    /// The image of `coset` under the signed letter `letter` (`+k` for the
    /// `k`-th generator, `-k` for its inverse).
    pub fn image(&self, coset: usize, letter: i32) -> Result<usize, FpGroupError> {
        if coset >= self.index {
            return Err(FpGroupError::CosetOutOfRange {
                coset,
                index: self.index,
            });
        }
        if letter == 0 || letter.unsigned_abs() as usize > self.rank {
            return Err(FpGroupError::InvalidGenerator {
                letter,
                rank: self.rank,
            });
        }
        Ok(self.table[coset * 2 * self.rank + column_of(letter)])
    }

    /// Trace a whole word from `coset`: the coset `coset · w`.
    pub fn trace(&self, coset: usize, w: &Word) -> Result<usize, FpGroupError> {
        let mut c = coset;
        if c >= self.index {
            return Err(FpGroupError::CosetOutOfRange {
                coset,
                index: self.index,
            });
        }
        for &l in w.letters() {
            c = self.image(c, l)?;
        }
        Ok(c)
    }

    /// The permutation of the cosets induced by generator `index` (0-based).
    pub fn generator_permutation(&self, index: usize) -> Result<Permutation, FpGroupError> {
        if index >= self.rank {
            return Err(FpGroupError::InvalidGenerator {
                letter: index as i32 + 1,
                rank: self.rank,
            });
        }
        let w = 2 * self.rank;
        let images: Vec<usize> = (0..self.index)
            .map(|c| self.table[c * w + 2 * index])
            .collect();
        Permutation::from_images(images).map_err(FpGroupError::from)
    }

    /// One permutation per generator, in order.
    pub fn permutations(&self) -> Result<Vec<Permutation>, FpGroupError> {
        (0..self.rank)
            .map(|i| self.generator_permutation(i))
            .collect()
    }

    /// The permutation representation of `G` on the cosets of `H`, as a
    /// [`PermutationGroup`] — so orbits, Schreier–Sims, the exact order and
    /// membership testing all come from the permutation-group module rather
    /// than being reimplemented here.
    ///
    /// The degree is `[G:H]`. For `H = 1` this is the **regular**
    /// representation, so the group's order is the index; for a larger `H` the
    /// kernel is the core of `H` in `G` and the image can be smaller than `G`.
    pub fn permutation_group(&self) -> Result<PermutationGroup, FpGroupError> {
        let perms = self.permutations()?;
        PermutationGroup::new(self.index, perms).map_err(FpGroupError::from)
    }

    /// A breadth-first spanning tree of the table, rooted at coset 0.
    ///
    /// `tree[c] = Some((parent, column))` means `parent · letter_of(column) = c`
    /// and that this is the edge by which `c` was first reached; `tree[0]` is
    /// `None`. This is the Schreier transversal Reidemeister–Schreier rewrites
    /// against.
    pub(super) fn spanning_tree(&self) -> Vec<Option<(usize, usize)>> {
        let mut tree: Vec<Option<(usize, usize)>> = vec![None; self.index];
        let mut seen = vec![false; self.index];
        let mut queue = std::collections::VecDeque::new();
        seen[0] = true;
        queue.push_back(0usize);
        let w = 2 * self.rank;
        while let Some(c) = queue.pop_front() {
            for x in 0..w {
                let d = self.table[c * w + x];
                if !seen[d] {
                    seen[d] = true;
                    tree[d] = Some((c, x));
                    queue.push_back(d);
                }
            }
        }
        tree
    }

    /// A Schreier transversal: for each coset `c`, a word `u_c` in the group's
    /// generators with `0 · u_c = c`, and `u_0` the identity.
    ///
    /// The words come from a breadth-first spanning tree, so `|u_c|` is the
    /// distance from `H` in the Cayley graph and the set is prefix-closed.
    pub fn transversal(&self) -> Result<Vec<Word>, FpGroupError> {
        let tree = self.spanning_tree();
        let mut words: Vec<Option<Word>> = vec![None; self.index];
        words[0] = Some(Word::identity());
        // A tree parent always has a smaller BFS depth, but not necessarily a
        // smaller coset *number* — the table is numbered in enumeration order —
        // so resolve by repeated passes until nothing more can be filled.
        let mut remaining = self.index - 1;
        while remaining > 0 {
            let mut progressed = false;
            for c in 1..self.index {
                if words[c].is_some() {
                    continue;
                }
                if let Some((parent, column)) = tree[c] {
                    if let Some(pw) = words[parent].clone() {
                        let mut letters = pw.letters().to_vec();
                        letters.push(letter_of(column));
                        words[c] = Some(Word::from_letters(&letters)?);
                        remaining -= 1;
                        progressed = true;
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        // Unreachable in a complete table — every coset was created as the image
        // of an existing one — but a silent identity word here would make a
        // multiplication table, a Reidemeister–Schreier rewriting and a
        // cohomology group quietly wrong, so it is a refusal rather than a
        // fallback.
        words
            .into_iter()
            .enumerate()
            .map(|(c, w)| {
                w.ok_or_else(|| FpGroupError::Internal {
                    detail: format!("coset {c} is not reachable from H in a complete table"),
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// Enumerate the cosets of `⟨subgroup_generators⟩` in `⟨rank generators | relators⟩`.
///
/// Returns the complete [`CosetTable`], whose [`index`](CosetTable::index) is
/// `[G:H]`.
///
/// # Refusal
///
/// [`FpGroupError::EnumerationIncomplete`] when the cap is reached. That is a
/// statement about this run, **not** about the group: see the module docs.
pub fn enumerate(
    rank: usize,
    relators: &[Word],
    subgroup_generators: &[Word],
    max_cosets: usize,
) -> Result<CosetTable, FpGroupError> {
    if rank == 0 {
        // The trivial group has one coset and an empty table; there is nothing
        // to enumerate and no relator can say otherwise.
        return Ok(CosetTable {
            rank: 0,
            index: 1,
            table: Vec::new(),
            subgroup_generators: subgroup_generators.to_vec(),
            cosets_defined: 1,
            coincidences: 0,
            max_cosets,
        });
    }
    if max_cosets == 0 {
        return Err(FpGroupError::CosetCapTooLarge {
            requested: 0,
            max: MAX_COSET_TABLE_CELLS / (2 * rank),
        });
    }
    let ceiling = MAX_COSET_TABLE_CELLS / (2 * rank);
    if max_cosets > ceiling {
        return Err(FpGroupError::CosetCapTooLarge {
            requested: max_cosets,
            max: ceiling,
        });
    }
    for w in relators.iter().chain(subgroup_generators.iter()) {
        if let Some(g) = w.max_generator() {
            if g >= rank {
                return Err(FpGroupError::InvalidGenerator {
                    letter: g as i32 + 1,
                    rank,
                });
            }
        }
    }

    let rels: Vec<Vec<usize>> = relators
        .iter()
        .filter(|w| !w.is_empty())
        .map(columns_of)
        .collect();
    let subs: Vec<Vec<usize>> = subgroup_generators
        .iter()
        .filter(|w| !w.is_empty())
        .map(columns_of)
        .collect();

    let mut e = Enumerator::new(rank, max_cosets);
    let mut total_defined = 1usize;
    let mut rounds = 0usize;
    loop {
        let outcome = (|| -> Result<(), Full> {
            loop {
                e.hlt_pass(&rels, &subs)?;
                if e.closed(&rels, &subs)? {
                    return Ok(());
                }
            }
        })();
        total_defined = total_defined.max(e.n);
        match outcome {
            Ok(()) => break,
            Err(Full) => {
                rounds += 1;
                let before = e.live_count();
                e.lookahead(&rels);
                e.compact();
                let after = e.live_count();
                // Give up when a lookahead round cannot free at least a quarter
                // of the table: further rounds would cost the same and free
                // less.
                if rounds >= MAX_LOOKAHEAD_ROUNDS || after * 4 >= before * 3 {
                    return Err(FpGroupError::EnumerationIncomplete {
                        live_cosets: after,
                        cosets_defined: total_defined,
                        max_cosets,
                    });
                }
            }
        }
    }

    e.compact();
    let index = e.n;
    let ncols = e.ncols;
    let mut table = vec![0usize; index * ncols];
    for c in 0..index {
        for x in 0..ncols {
            let v = e.get(c + 1, x);
            if v == 0 {
                return Err(FpGroupError::Internal {
                    detail: format!("coset table entry ({c}, {x}) undefined after completion"),
                });
            }
            table[c * ncols + x] = v - 1;
        }
    }

    let out = CosetTable {
        rank,
        index,
        table,
        subgroup_generators: subgroup_generators.to_vec(),
        cosets_defined: total_defined,
        coincidences: e.coincidences,
        max_cosets,
    };
    verify(&out, relators, subgroup_generators)?;
    Ok(out)
}

/// Check the defining properties of a coset table. Anything that fails here is
/// a bug in the enumerator, not a property of the input.
fn verify(
    t: &CosetTable,
    relators: &[Word],
    subgroup_generators: &[Word],
) -> Result<(), FpGroupError> {
    let ncols = 2 * t.rank;
    for c in 0..t.index {
        for x in 0..ncols {
            let d = t.table[c * ncols + x];
            if d >= t.index {
                return Err(FpGroupError::Internal {
                    detail: format!("coset table entry ({c}, {x}) = {d} is out of range"),
                });
            }
            if t.table[d * ncols + (x ^ 1)] != c {
                return Err(FpGroupError::Internal {
                    detail: format!(
                        "coset table is not consistent: ({c}, {x}) -> {d} but ({d}, {}) does not \
                         come back",
                        x ^ 1
                    ),
                });
            }
        }
    }
    for r in relators {
        for c in 0..t.index {
            if t.trace(c, r)? != c {
                return Err(FpGroupError::Internal {
                    detail: format!("relator {r} does not close at coset {c}"),
                });
            }
        }
    }
    for s in subgroup_generators {
        if t.trace(0, s)? != 0 {
            return Err(FpGroupError::Internal {
                detail: format!("subgroup generator {s} does not fix the coset of H"),
            });
        }
    }
    Ok(())
}
