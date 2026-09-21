//! Permutation groups: orbits with Schreier vectors, Schreier–Sims, and the
//! stabilizer chain everything else is read off.

use super::{GroupError, Permutation};
use rug::Integer;
use std::sync::OnceLock;

/// Largest degree for which a stabilizer chain will be built.
///
/// The chain stores an explicit transversal *permutation* for every point of
/// every basic orbit, so its memory is `Σ_i |Δ_i| · n` words. For the worst
/// case `S_n` that is about `n³/2` words — roughly 67 MB at degree 256, and
/// eight times that for each doubling. Orbits alone are unaffected; only
/// [`PermutationGroup::stabilizer_chain`] and its dependents
/// ([`order`](PermutationGroup::order), [`contains`](PermutationGroup::contains),
/// [`elements`](PermutationGroup::elements)) refuse above this.
pub const MAX_BSGS_DEGREE: usize = 256;

/// Default ceiling on [`PermutationGroup::elements`].
pub const DEFAULT_ELEMENT_CAP: u64 = 100_000;

/// The largest cap [`PermutationGroup::elements_with_cap`] will accept.
///
/// A list of a million permutations is already tens of megabytes; beyond that
/// the answer to "give me every element" is that there is a better question.
///
/// Note that the cap counts *elements*, not bytes, and an element costs
/// `8·degree` bytes: a million permutations of degree 256 is two gigabytes.
/// The default cap keeps that under a few hundred megabytes at any admissible
/// degree, but raising it explicitly is a decision about memory as well as
/// about patience.
pub const MAX_ELEMENT_CAP: u64 = 1_000_000;

// ---------------------------------------------------------------------------
// Orbits and Schreier vectors
// ---------------------------------------------------------------------------

/// One entry of a Schreier vector: how a point was first reached.
///
/// `β = α^(g)` where `α` is [`from`](SchreierEntry::from) and `g` is the
/// generator at index [`generator`](SchreierEntry::generator) in the generator
/// list the orbit was built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchreierEntry {
    /// Index into the generator list.
    pub generator: usize,
    /// The point this one was reached from.
    pub from: usize,
}

/// The orbit of a point under a list of generators, with its Schreier vector
/// and the transversal derived from it.
///
/// A *transversal element* `u_β` satisfies `base_point^(u_β) = β`. The
/// transversal stored here and the one
/// [`transversal_from_schreier_vector`](Orbit::transversal_from_schreier_vector)
/// recomputes are built by the same breadth-first walk and are equal point by
/// point; a unit test asserts exactly that, so the Schreier vector is the
/// definition and the stored permutations are a cache of it.
#[derive(Clone, Debug)]
pub struct Orbit {
    base_point: usize,
    degree: usize,
    /// Orbit points in breadth-first discovery order; `points[0]` is the base
    /// point.
    points: Vec<usize>,
    schreier: Vec<Option<SchreierEntry>>,
    transversal: Vec<Option<Permutation>>,
}

impl Orbit {
    /// The point whose orbit this is.
    pub fn base_point(&self) -> usize {
        self.base_point
    }

    /// The degree of the ambient symmetric group.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The orbit points, in breadth-first discovery order.
    pub fn points(&self) -> &[usize] {
        &self.points
    }

    /// The orbit points, ascending.
    pub fn sorted_points(&self) -> Vec<usize> {
        let mut p = self.points.clone();
        p.sort_unstable();
        p
    }

    /// The orbit length. Always at least 1.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Never true — an orbit always contains its own base point. Present
    /// because `len` without `is_empty` is a lint, and because saying so is
    /// cheaper than leaving a reader to wonder.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Is `point` in this orbit?
    pub fn contains(&self, point: usize) -> bool {
        point < self.degree && self.transversal[point].is_some()
    }

    /// The Schreier vector, indexed by point.
    ///
    /// `None` at the base point (the root of the walk) and at every point
    /// outside the orbit. Use [`contains`](Orbit::contains) to tell those two
    /// apart.
    pub fn schreier_vector(&self) -> &[Option<SchreierEntry>] {
        &self.schreier
    }

    /// The stored transversal element `u_β` with `base_point^(u_β) = β`, or
    /// `None` if `β` is outside the orbit.
    pub fn transversal_element(&self, point: usize) -> Option<&Permutation> {
        self.transversal.get(point).and_then(|u| u.as_ref())
    }

    /// Recompute `u_β` by walking the Schreier vector back to the base point.
    ///
    /// `generators` must be the same list, in the same order, that the orbit
    /// was built from — the Schreier vector stores generator *indices*.
    pub fn transversal_from_schreier_vector(
        &self,
        point: usize,
        generators: &[Permutation],
    ) -> Option<Permutation> {
        if !self.contains(point) {
            return None;
        }
        let mut path = Vec::new();
        let mut current = point;
        while let Some(entry) = self.schreier[current] {
            path.push(entry.generator);
            current = entry.from;
        }
        debug_assert_eq!(current, self.base_point);
        let mut u = Permutation::identity(self.degree);
        // `path` runs from `point` back to the base point, so the generators
        // are applied in reverse: the first one applied is the last pushed.
        for &g in path.iter().rev() {
            u = u.compose_unchecked(generators.get(g)?);
        }
        Some(u)
    }
}

/// Breadth-first orbit of `base` under `generators`, building the Schreier
/// vector and the transversal in the same pass.
fn build_orbit(degree: usize, generators: &[Permutation], base: usize) -> Orbit {
    let mut schreier: Vec<Option<SchreierEntry>> = vec![None; degree];
    let mut transversal: Vec<Option<Permutation>> = vec![None; degree];
    let mut points = Vec::new();
    if base < degree {
        points.push(base);
        transversal[base] = Some(Permutation::identity(degree));
    }
    let mut head = 0;
    while head < points.len() {
        let alpha = points[head];
        head += 1;
        let u_alpha = transversal[alpha]
            .clone()
            .expect("a discovered point has a transversal element");
        for (index, g) in generators.iter().enumerate() {
            let beta = g.img(alpha);
            if transversal[beta].is_none() {
                // u_β = u_α · g, left to right: base^(u_α·g) = (base^u_α)^g = α^g = β.
                transversal[beta] = Some(u_alpha.compose_unchecked(g));
                schreier[beta] = Some(SchreierEntry {
                    generator: index,
                    from: alpha,
                });
                points.push(beta);
            }
        }
    }
    Orbit {
        base_point: base,
        degree,
        points,
        schreier,
        transversal,
    }
}

// ---------------------------------------------------------------------------
// The stabilizer chain
// ---------------------------------------------------------------------------

/// One level of a stabilizer chain.
///
/// Level `i` carries the base point `b_i`, the strong generators `S_i` of the
/// pointwise stabilizer `G^{(i)} = G_{b_0,…,b_{i−1}}`, and the basic orbit
/// `Δ_i = b_i^{S_i}` with its transversal.
#[derive(Clone, Debug)]
pub struct StabilizerLevel {
    base_point: usize,
    generators: Vec<Permutation>,
    orbit: Orbit,
}

impl StabilizerLevel {
    /// The base point `b_i` of this level.
    pub fn base_point(&self) -> usize {
        self.base_point
    }

    /// The strong generators `S_i`, which generate `G^{(i)}`.
    pub fn generators(&self) -> &[Permutation] {
        &self.generators
    }

    /// The basic orbit `Δ_i = b_i^{S_i}`, with Schreier vector and transversal.
    pub fn orbit(&self) -> &Orbit {
        &self.orbit
    }
}

/// A base and strong generating set, as a chain of stabilizers.
///
/// `|G| = Π_i |Δ_i|`, and an element is in `G` exactly when it sifts to the
/// identity through the chain. Both facts are theorems about a *correct* BSGS;
/// the correctness here rests on [`schreier_sims`], the deterministic
/// Schreier-generator algorithm, which is tested against `|S_n| = n!`,
/// `|A_n| = n!/2`, `|M₁₁| = 7920` and `|M₁₂| = 95040`.
#[derive(Clone, Debug)]
pub struct StabilizerChain {
    degree: usize,
    levels: Vec<StabilizerLevel>,
}

impl StabilizerChain {
    /// The degree of the ambient symmetric group.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The base `[b_0, …, b_{k−1}]`.
    pub fn base(&self) -> Vec<usize> {
        self.levels.iter().map(|l| l.base_point).collect()
    }

    /// The levels, outermost (`G` itself) first.
    pub fn levels(&self) -> &[StabilizerLevel] {
        &self.levels
    }

    /// The union of every level's strong generators, deduplicated, ascending.
    ///
    /// This is the "strong generating set" of the name; the per-level split is
    /// on [`levels`](StabilizerChain::levels).
    pub fn strong_generators(&self) -> Vec<Permutation> {
        let mut all: Vec<Permutation> = self
            .levels
            .iter()
            .flat_map(|l| l.generators.iter().cloned())
            .collect();
        all.sort();
        all.dedup();
        all
    }

    /// `|G| = Π_i |Δ_i|`, exactly.
    pub fn order(&self) -> Integer {
        let mut order = Integer::from(1);
        for level in &self.levels {
            order *= level.orbit.len() as u64;
        }
        order
    }

    /// Sift `g` through the chain.
    ///
    /// At each level the current residue is multiplied by the inverse of the
    /// transversal element carrying `b_i` to `b_i^g`, which makes the residue
    /// fix `b_i`. If that image is outside the basic orbit the sift stops, and
    /// `g` is not in `G`.
    pub fn sift(&self, g: &Permutation) -> Result<SiftResult, GroupError> {
        if g.degree() != self.degree {
            return Err(GroupError::DegreeMismatch {
                left: self.degree,
                right: g.degree(),
            });
        }
        let (residue, level) = self.strip(g, 0);
        let is_member = residue.is_identity();
        Ok(SiftResult {
            residue,
            level,
            is_member,
        })
    }

    /// The stripping loop, starting at `from_level`.
    ///
    /// Returns the residue and the level at which it stopped — `levels.len()`
    /// if it stripped all the way through.
    fn strip(&self, g: &Permutation, from_level: usize) -> (Permutation, usize) {
        let mut h = g.clone();
        for (index, level) in self.levels.iter().enumerate().skip(from_level) {
            let image = h.img(level.base_point);
            match level.orbit.transversal_element(image) {
                None => return (h, index),
                Some(u) => h = h.compose_unchecked(&u.inverse()),
            }
        }
        (h, self.levels.len())
    }
}

/// What [`StabilizerChain::sift`] found.
#[derive(Clone, Debug)]
pub struct SiftResult {
    residue: Permutation,
    level: usize,
    is_member: bool,
}

impl SiftResult {
    /// The residue after stripping. The identity exactly when the element is
    /// in the group.
    pub fn residue(&self) -> &Permutation {
        &self.residue
    }

    /// The level at which stripping stopped — the number of levels if it went
    /// all the way through.
    pub fn level(&self) -> usize {
        self.level
    }

    /// Is the sifted element a member of the group?
    pub fn is_member(&self) -> bool {
        self.is_member
    }
}

// ---------------------------------------------------------------------------
// Schreier–Sims
// ---------------------------------------------------------------------------

/// Mutable state of the incremental Schreier–Sims run.
struct ChainBuilder {
    degree: usize,
    base: Vec<usize>,
    generators: Vec<Vec<Permutation>>,
    orbits: Vec<Orbit>,
}

impl ChainBuilder {
    fn strip(&self, g: &Permutation, from_level: usize) -> (Permutation, usize) {
        let mut h = g.clone();
        for index in from_level..self.base.len() {
            let image = h.img(self.base[index]);
            match self.orbits[index].transversal_element(image) {
                None => return (h, index),
                Some(u) => h = h.compose_unchecked(&u.inverse()),
            }
        }
        (h, self.base.len())
    }

    /// Make level `i` satisfy the BSGS condition
    /// `⟨S_{i+1}⟩ = Stab_{⟨S_i⟩}(b_i)`, assuming every deeper level already
    /// does.
    ///
    /// Schreier's lemma: the stabilizer of `b_i` in `⟨S_i⟩` is generated by
    /// `u_β · x · u_{β^x}^{-1}` over `β ∈ Δ_i`, `x ∈ S_i`. Each such generator
    /// is stripped through the deeper levels; a non-identity residue `h` is a
    /// witness that the chain below is too small, so `h` is adjoined to every
    /// level from `i+1` down to the one where stripping stopped, and those
    /// levels are re-verified from the deepest upwards.
    ///
    /// `S_i` and `Δ_i` are untouched by this call and by everything it
    /// recurses into — the recursion only ever adds generators at levels
    /// *deeper* than `i` — so the doubly-nested loop below iterates over sets
    /// that do not change under it. That is why no restart is needed at level
    /// `i` itself: a Schreier generator already checked stripped to the
    /// identity through a *smaller* `⟨S_{i+1}⟩`, and it still does through the
    /// larger one.
    fn complete_level(&mut self, i: usize) {
        self.orbits[i] = build_orbit(self.degree, &self.generators[i], self.base[i]);
        let points = self.orbits[i].points().to_vec();
        let generator_count = self.generators[i].len();

        for beta in points {
            let u_beta = self.orbits[i]
                .transversal_element(beta)
                .expect("orbit point has a transversal element")
                .clone();
            for xi in 0..generator_count {
                let x = self.generators[i][xi].clone();
                let image = x.img(beta);
                let u_image = self.orbits[i]
                    .transversal_element(image)
                    .expect("the image of an orbit point is in the orbit")
                    .clone();
                // Schreier generator u_β · x · u_{β^x}^{-1}; it fixes b_i.
                let schreier_generator = u_beta
                    .compose_unchecked(&x)
                    .compose_unchecked(&u_image.inverse());
                debug_assert_eq!(schreier_generator.img(self.base[i]), self.base[i]);
                if schreier_generator.is_identity() {
                    continue;
                }
                let (residue, mut stop) = self.strip(&schreier_generator, i + 1);
                if residue.is_identity() {
                    continue;
                }
                if stop == self.base.len() {
                    // The residue fixes every base point but is not the
                    // identity, so the base does not yet separate it from 1.
                    let new_point = residue
                        .first_moved_point()
                        .expect("a non-identity permutation moves a point");
                    self.base.push(new_point);
                    self.generators.push(Vec::new());
                    self.orbits.push(build_orbit(self.degree, &[], new_point));
                    stop = self.base.len() - 1;
                }
                for level in (i + 1)..=stop {
                    self.generators[level].push(residue.clone());
                }
                for level in ((i + 1)..=stop).rev() {
                    self.complete_level(level);
                }
            }
        }
    }
}

/// Deterministic Schreier–Sims: a base and strong generating set for the group
/// generated by `generators`.
///
/// Refuses degrees above [`MAX_BSGS_DEGREE`] — see that constant for the memory
/// argument.
pub(super) fn schreier_sims(
    degree: usize,
    generators: &[Permutation],
) -> Result<StabilizerChain, GroupError> {
    if degree > MAX_BSGS_DEGREE {
        return Err(GroupError::DegreeTooLargeForBsgs {
            degree,
            max: MAX_BSGS_DEGREE,
        });
    }
    for g in generators {
        if g.degree() != degree {
            return Err(GroupError::DegreeMismatch {
                left: degree,
                right: g.degree(),
            });
        }
    }

    // Initial base: enough points that no generator fixes all of them.
    let mut base: Vec<usize> = Vec::new();
    for g in generators {
        if g.is_identity() {
            continue;
        }
        if base.iter().all(|&b| g.img(b) == b) {
            base.push(
                g.first_moved_point()
                    .expect("a non-identity permutation moves a point"),
            );
        }
    }

    // S_i = the generators fixing b_0 … b_{i−1}.  The condition is monotone in
    // i, so each generator lands in a prefix of the levels.
    let mut level_generators: Vec<Vec<Permutation>> = vec![Vec::new(); base.len()];
    for g in generators {
        if g.is_identity() {
            continue;
        }
        for (level, &b) in base.iter().enumerate() {
            level_generators[level].push(g.clone());
            if g.img(b) != b {
                break;
            }
        }
    }

    let orbits: Vec<Orbit> = base
        .iter()
        .enumerate()
        .map(|(i, &b)| build_orbit(degree, &level_generators[i], b))
        .collect();

    let mut builder = ChainBuilder {
        degree,
        base,
        generators: level_generators,
        orbits,
    };

    // Deepest level first: complete_level(i) assumes levels > i are already
    // correct.
    let mut i = builder.base.len();
    while i > 0 {
        i -= 1;
        builder.complete_level(i);
    }

    let levels = builder
        .base
        .iter()
        .enumerate()
        .map(|(index, &base_point)| StabilizerLevel {
            base_point,
            generators: builder.generators[index].clone(),
            orbit: builder.orbits[index].clone(),
        })
        .collect();

    Ok(StabilizerChain { degree, levels })
}

// ---------------------------------------------------------------------------
// The group
// ---------------------------------------------------------------------------

/// A subgroup of `S_n` given by generators.
///
/// The stabilizer chain is computed on first demand and cached; construction
/// itself is cheap and cannot fail for reasons of size, so orbits are available
/// at any degree even where [`MAX_BSGS_DEGREE`] forbids a chain.
#[derive(Clone, Debug)]
pub struct PermutationGroup {
    degree: usize,
    generators: Vec<Permutation>,
    chain: OnceLock<Result<StabilizerChain, GroupError>>,
}

impl PermutationGroup {
    /// The subgroup of `S_degree` generated by `generators`.
    ///
    /// Every generator must have exactly this degree
    /// ([`GroupError::DegreeMismatch`] otherwise); nothing is padded. An empty
    /// generator list is the trivial group, which is a group.
    pub fn new(degree: usize, generators: Vec<Permutation>) -> Result<Self, GroupError> {
        for g in &generators {
            if g.degree() != degree {
                return Err(GroupError::DegreeMismatch {
                    left: degree,
                    right: g.degree(),
                });
            }
        }
        Ok(PermutationGroup {
            degree,
            generators,
            chain: OnceLock::new(),
        })
    }

    /// Like [`new`](PermutationGroup::new), taking the degree from the first
    /// generator.
    ///
    /// Refuses an empty list: the degree of the trivial group is not
    /// recoverable from no generators, and guessing 0 would silently produce a
    /// group acting on no points.
    pub fn from_generators(generators: Vec<Permutation>) -> Result<Self, GroupError> {
        let degree =
            generators
                .first()
                .map(|g| g.degree())
                .ok_or(GroupError::UnsupportedDegree {
                    family: "from_generators",
                    requested: 0,
                    reason: "an empty generator list does not determine a degree; use \
                         PermutationGroup::new(degree, vec![]) for the trivial group",
                })?;
        PermutationGroup::new(degree, generators)
    }

    /// The degree of the ambient symmetric group.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The generators, as supplied.
    pub fn generators(&self) -> &[Permutation] {
        &self.generators
    }

    /// Is every generator the identity? Then the group is trivial.
    ///
    /// Decided from the generators alone, with no stabilizer chain, so it works
    /// at any degree.
    pub fn is_trivial(&self) -> bool {
        self.generators.iter().all(|g| g.is_identity())
    }

    /// The orbit of `point`, with its Schreier vector and transversal.
    ///
    /// The generator indices in the Schreier vector index
    /// [`generators`](PermutationGroup::generators).
    pub fn orbit(&self, point: usize) -> Result<Orbit, GroupError> {
        if point >= self.degree {
            return Err(GroupError::PointOutOfRange {
                point,
                degree: self.degree,
            });
        }
        Ok(build_orbit(self.degree, &self.generators, point))
    }

    /// The orbits of the group on `{0, …, n−1}`, each ascending, ordered by
    /// least element. Together they partition the point set.
    pub fn orbits(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.degree];
        let mut out = Vec::new();
        for point in 0..self.degree {
            if seen[point] {
                continue;
            }
            let points = build_orbit(self.degree, &self.generators, point).sorted_points();
            for &p in &points {
                seen[p] = true;
            }
            out.push(points);
        }
        out
    }

    /// Is the action transitive — exactly one orbit?
    ///
    /// Degree 0 has no orbits at all and is reported as **not** transitive;
    /// there is no point for the action to be transitive on.
    pub fn is_transitive(&self) -> bool {
        self.degree > 0 && build_orbit(self.degree, &self.generators, 0).len() == self.degree
    }

    /// The stabilizer chain, computed on first call and cached.
    pub fn stabilizer_chain(&self) -> Result<&StabilizerChain, GroupError> {
        self.chain
            .get_or_init(|| schreier_sims(self.degree, &self.generators))
            .as_ref()
            .map_err(|e| e.clone())
    }

    /// `|G|`, exactly, as an arbitrary-precision integer.
    pub fn order(&self) -> Result<Integer, GroupError> {
        Ok(self.stabilizer_chain()?.order())
    }

    /// Is `g` an element of this group?
    ///
    /// By sifting through the stabilizer chain — no enumeration, so this is
    /// usable in groups far too large to list.
    pub fn contains(&self, g: &Permutation) -> Result<bool, GroupError> {
        Ok(self.sift(g)?.is_member())
    }

    /// Sift `g` through the chain, reporting the residue and where it stopped.
    pub fn sift(&self, g: &Permutation) -> Result<SiftResult, GroupError> {
        if g.degree() != self.degree {
            return Err(GroupError::DegreeMismatch {
                left: self.degree,
                right: g.degree(),
            });
        }
        self.stabilizer_chain()?.sift(g)
    }

    /// A uniformly random element, from a deterministic PRNG seeded by `seed`.
    ///
    /// Every element of `G` factors **uniquely** as `u_{k−1} ⋯ u_1 · u_0` with
    /// `u_i` in the level-`i` transversal, so choosing each `u_i` uniformly and
    /// independently gives a uniform element. (Uniform up to the modulo bias of
    /// reducing a `u64` modulo the orbit length, which is below `2^-50` for any
    /// degree this module accepts.)
    pub fn random_element(&self, seed: u64) -> Result<Permutation, GroupError> {
        let chain = self.stabilizer_chain()?;
        let mut rng = SplitMix64::new(seed);
        let mut element = Permutation::identity(self.degree);
        for level in chain.levels().iter().rev() {
            let points = level.orbit().points();
            let beta = points[rng.below(points.len())];
            let u = level
                .orbit()
                .transversal_element(beta)
                .expect("orbit point has a transversal element");
            element = element.compose_unchecked(u);
        }
        Ok(element)
    }

    /// Every element of the group, if there are at most
    /// [`DEFAULT_ELEMENT_CAP`] of them.
    ///
    /// Refuses with [`GroupError::EnumerationTooLarge`] otherwise. The order is
    /// still exact and still available from [`order`](PermutationGroup::order):
    /// this refusal is about the list, not about the arithmetic.
    pub fn elements(&self) -> Result<Vec<Permutation>, GroupError> {
        self.elements_with_cap(DEFAULT_ELEMENT_CAP)
    }

    /// [`elements`](PermutationGroup::elements) with an explicit cap, itself
    /// capped at [`MAX_ELEMENT_CAP`].
    pub fn elements_with_cap(&self, cap: u64) -> Result<Vec<Permutation>, GroupError> {
        let cap = cap.min(MAX_ELEMENT_CAP);
        let chain = self.stabilizer_chain()?;
        let order = chain.order();
        if order > cap {
            return Err(GroupError::EnumerationTooLarge {
                order: order.to_string(),
                cap,
            });
        }
        // G^{(i)} is the disjoint union of the cosets G^{(i+1)}·u for u in the
        // level-i transversal, so an element of G^{(i)} is uniquely h·u.
        let mut elements = vec![Permutation::identity(self.degree)];
        for level in chain.levels().iter().rev() {
            let mut next = Vec::with_capacity(elements.len() * level.orbit().len());
            for h in &elements {
                for &beta in level.orbit().points() {
                    let u = level
                        .orbit()
                        .transversal_element(beta)
                        .expect("orbit point has a transversal element");
                    next.push(h.compose_unchecked(u));
                }
            }
            elements = next;
        }
        Ok(elements)
    }
}

// ---------------------------------------------------------------------------
// A small deterministic PRNG, so `random_element` is reproducible in tests
// without adding a dependency.
// ---------------------------------------------------------------------------

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> SplitMix64 {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }
}
