//! The ordinary character table, by Dixon–Schneider, with exact values in
//! `ℚ(ζ_e)`.
//!
//! # The algorithm, and why this one
//!
//! Burnside's method — build the regular representation and decompose it — is
//! `|G| × |G|` linear algebra over a field nobody knows in advance, and it
//! needs the irrationalities before it can start. Dixon's method needs none of
//! them. The class sums `K̂_1, …, K̂_r` are a basis of the centre of the group
//! algebra, the class multiplication matrices `M_k` (`M_k[i][j] = a_{kij}`)
//! commute, and each irreducible character `χ` gives a common eigenvector
//!
//! ```text
//!     ω(χ)_i = |K_i| · χ(g_i) / χ(1),          M_k · ω(χ) = ω(χ)_k · ω(χ).
//! ```
//!
//! Those `ω_i` are algebraic integers in `ℚ(ζ_e)`, `e = exp G`. Choose a prime
//! `p ≡ 1 (mod e)` with `p > |G|`: then `ℤ[ζ_e]` has a residue map onto GF(p)
//! (`p` splits completely), so **every eigenvalue lies in GF(p)** and the whole
//! eigen-decomposition is `r × r` linear algebra over a word-sized prime field —
//! [`crate::ffield`]'s `nmod_mat`. `p ∤ |G|` (which `p > |G|` gives for free)
//! makes the class algebra split semisimple mod `p`, so the `r` common
//! eigenvectors exist, are one-dimensional and are distinct.
//!
//! # How the lift back to `ℤ[ζ_e]` is made exact
//!
//! The usual worry about a modular algorithm is the lift. Here it is not a
//! worry, because what gets lifted is a **multiplicity**. For `g` of order `m`,
//! the eigenvalues of `ρ(g)` are `m`-th roots of unity, so
//!
//! ```text
//!     χ(g) = Σ_{k<m} n_k ζ_m^k,      n_k = (1/m) Σ_{l<m} χ(g^l) ζ_m^{-kl}
//! ```
//!
//! where `n_k ≥ 0` is the multiplicity of the eigenvalue `ζ_m^k` and
//! `Σ_k n_k = χ(1)`. Each `n_k` is therefore a non-negative rational integer
//! bounded by `χ(1) ≤ √|G| < p`, so **its residue in `0..p` is its value** — no
//! symmetric lift, no rational reconstruction, no bound to get wrong. And the
//! two facts `0 ≤ n_k ≤ χ(1)` and `Σ_k n_k = χ(1)` are checked for every entry,
//! which makes each value carry its own checksum.
//!
//! The degree comes out the same way: `Σ_i ω_i ω_{i*} / |K_i| = |G| / χ(1)²`
//! with `i*` the class of inverses, so `χ(1)²` is read mod `p`, and since
//! `χ(1)² ≤ |G| < p` its residue is the integer itself.
//!
//! # Exactness
//!
//! Values are [`NumberFieldElement`]s of one cyclotomic field `ℚ(ζ_e)`,
//! `e = exp G`. Nothing here is floating point and nothing falls back to
//! rationals: `A_4`'s two non-real linear characters come out as `ζ_3` and
//! `ζ_3²` in `ℚ(ζ_{6})`, and `A_5`'s two degree-3 characters as the golden-ratio
//! pair `(1 ± √5)/2` written in `ℚ(ζ_{30})`. A group whose characters happen to
//! be rational simply gets rational elements of that same field.
//!
//! Complex conjugation is **not** taken as a field automorphism — `numfield`
//! has none. It is taken as `conj(χ(g)) = χ(g⁻¹)`, a lookup through
//! [`ConjugacyClasses::inverse_class`], which is exact and needs no embedding.
//!
//! # Nothing unchecked is returned
//!
//! Before a [`CharacterTable`] exists, all four of the following have passed as
//! exact identities in `ℚ(ζ_e)`; any failure is
//! [`CharacterError::SelfCheckFailed`] and the table is withheld:
//!
//! * row orthogonality `⟨χ_i, χ_j⟩ = δ_ij`,
//! * column orthogonality `Σ_i χ_i(g) χ_i(h⁻¹) = |C_G(g)|·[g ~ h]`,
//! * `Σ_i χ_i(1)² = |G|`, and every `χ_i(1)` divides `|G|`,
//! * one irreducible character per conjugacy class.
//!
//! # Scope
//!
//! Two ceilings are enforced. The first is on `|G|`, because classes are found
//! by conjugating every element and so the whole group sits in memory (see
//! [`DEFAULT_CHARACTER_TABLE_CAP`] and
//! [`super::MAX_CLASS_ENUMERATION_CAP`]). The second is on `φ(exp G)`, the
//! degree of the field every value lives in
//! ([`MAX_EXPONENT_FIELD_DEGREE`]). Neither is negotiable at run time beyond
//! the caps named here, and a group past either gets a typed refusal rather
//! than a partial table.
//!
//! Neither is usually what bites first. The `O(r³)` field multiplications of
//! the orthogonality checks dominate, so the **number of classes** is the
//! practical limit: see [`DEFAULT_CHARACTER_TABLE_CAP`] for measured figures.
//! A group with few classes is cheap at any admitted order; an abelian group,
//! where `r = |G|`, is not.
//!
//! Not here: Brauer characters and modular representation theory, the power
//! map and Galois/Frobenius action beyond what `inverse_class` gives, symmetric
//! and exterior powers, induced and restricted characters, tensor
//! decomposition, the character table of a *finitely presented* group, and any
//! table read out of a library rather than computed.

use rug::{Integer, Rational};

use super::classes::{to_u64, ConjugacyClasses};
use super::dixon;
use super::error::CharacterError;
use crate::group::PermutationGroup;
use crate::numfield::{NumberField, NumberFieldElement};

/// Default ceiling on `|G|` for [`CharacterTable::of`].
///
/// Lower than [`super::DEFAULT_CLASS_ENUMERATION_CAP`] on purpose: the class
/// partition costs `O(|G|)`, but the table costs a further `O(r · |G|)` group
/// products for the class multiplication matrices and `O(r³)` multiplications
/// in `ℚ(ζ_e)` for the orthogonality checks, and `r` grows with `|G|`.
///
/// # This cap is not the binding constraint, and it is worth knowing which is
///
/// The `O(r³)` field multiplications dominate, so the practical limit is the
/// number of **classes**. Measured on this implementation: `S_6` (`|G| = 720`,
/// `r = 11`) and `S_7` (`|G| = 5040`, `r = 15`) are both well under a tenth of a
/// second, while `ℤ/60` (`r = 60`) takes about a second and `ℤ/100` (`r = 100`)
/// about fifteen. A group with few classes is cheap at any order this cap
/// allows; an abelian group, where `r = |G|`, gets slow long before it.
pub const DEFAULT_CHARACTER_TABLE_CAP: u64 = 5_000;

/// Ceiling on `φ(exp G)`, the degree of the cyclotomic field every character
/// value lives in.
///
/// This is a bound on the *arithmetic*, not on the group. The two ceilings are
/// independent: a group of small exponent passes this one at any order (and
/// meets [`DEFAULT_CHARACTER_TABLE_CAP`] instead), while the cyclic group of
/// order 1000 has `φ(1000) = 400` and is refused here despite being small.
/// `ℚ(ζ_n)` itself admits degree up to 1024 (`E-NUMF-008`); the tighter limit
/// here is because the orthogonality checks multiply in that field `O(r³)`
/// times.
pub const MAX_EXPONENT_FIELD_DEGREE: usize = 256;

/// The ordinary (complex) character table of a finite permutation group, with
/// exact values in `ℚ(ζ_{exp G})`.
///
/// Rows are irreducible characters, columns are conjugacy classes in
/// [`ConjugacyClasses`]'s deterministic order (class 0 is `{1}`). Row 0 is the
/// trivial character; the rest are ordered by degree and then canonically by
/// their coordinates, so the table is reproducible run to run.
#[derive(Clone, Debug)]
pub struct CharacterTable {
    field: NumberField,
    exponent: u64,
    prime: u64,
    classes: ConjugacyClasses,
    degrees: Vec<u64>,
    values: Vec<Vec<NumberFieldElement>>,
}

impl CharacterTable {
    /// The character table of `group`, refusing above
    /// [`DEFAULT_CHARACTER_TABLE_CAP`].
    pub fn of(group: &PermutationGroup) -> Result<Self, CharacterError> {
        Self::of_with_cap(group, DEFAULT_CHARACTER_TABLE_CAP)
    }

    /// As [`CharacterTable::of`], with an explicit ceiling on `|G|`.
    ///
    /// Refuses with [`CharacterError::CapTooLarge`] above
    /// [`super::MAX_CLASS_ENUMERATION_CAP`]. Raising the cap raises the run time
    /// roughly as `|G| · r`; it does not change the answer.
    pub fn of_with_cap(group: &PermutationGroup, cap: u64) -> Result<Self, CharacterError> {
        let classes = ConjugacyClasses::of_with_cap(group, cap)?;
        Self::from_classes(classes)
    }

    /// The character table from an already-computed class partition.
    ///
    /// Useful when the classes are wanted for their own sake as well; the
    /// partition is the expensive half.
    pub fn from_classes(classes: ConjugacyClasses) -> Result<Self, CharacterError> {
        let r = classes.len();
        let exponent = classes.exponent();
        let phi = euler_phi(exponent);
        if phi > MAX_EXPONENT_FIELD_DEGREE {
            return Err(CharacterError::ExponentFieldTooLarge {
                exponent,
                degree: phi,
                max: MAX_EXPONENT_FIELD_DEGREE,
            });
        }
        let field = NumberField::cyclotomic(exponent)?;
        let zeta = field.generator();

        let order = to_u64(classes.group_order(), "|G|")?;
        let p = dixon::working_prime(exponent, order)?;
        let w = dixon::root_of_unity(p, exponent)?;

        // Class multiplication matrices, reduced mod p.
        let mut matrices = Vec::with_capacity(r);
        for k in 0..r {
            matrices.push(classes.multiplication_matrix(k)?);
        }

        let eigenvectors = dixon::common_eigenvectors(p, r, &matrices)?;

        // Per class: |K_i|, its inverse mod p, the class of each power of the
        // representative, and the order of the representative.
        let mut size_inverses = Vec::with_capacity(r);
        let mut inverse_class = Vec::with_capacity(r);
        let mut power_classes: Vec<Vec<usize>> = Vec::with_capacity(r);
        for i in 0..r {
            let class = classes.class(i)?;
            size_inverses.push(dixon::inv_mod(class.size() % p, p)?);
            inverse_class.push(classes.inverse_class(i)?);
            let m = class.element_order();
            let mut powers = Vec::with_capacity(m as usize);
            for l in 0..m {
                let g_l = class.representative().pow(l as i64);
                powers.push(classes.class_of(&g_l)?);
            }
            power_classes.push(powers);
        }

        let order_mod_p = order % p;
        let mut rows: Vec<(u64, Vec<NumberFieldElement>)> = Vec::with_capacity(r);

        for vector in &eigenvectors {
            if vector.len() != r {
                return Err(CharacterError::Internal {
                    detail: format!("an eigenvector has length {}, not {r}", vector.len()),
                });
            }
            // Normalise so that the identity class's coordinate is 1: ω_0 = 1
            // because |K_0| = 1 and χ(g_0) = χ(1).
            let scale = dixon::inv_mod(vector[0], p)?;
            let omega: Vec<u64> = vector
                .iter()
                .map(|&v| dixon::mul_mod(v, scale, p))
                .collect();

            // |G| / χ(1)² = Σ_i ω_i ω_{i*} / |K_i|.
            let mut accumulator = 0u64;
            for i in 0..r {
                let term = dixon::mul_mod(
                    dixon::mul_mod(omega[i], omega[inverse_class[i]], p),
                    size_inverses[i],
                    p,
                );
                accumulator = (accumulator + term) % p;
            }
            let square = dixon::mul_mod(order_mod_p, dixon::inv_mod(accumulator, p)?, p);
            let degree = isqrt(square);
            if degree == 0 || degree * degree != square || order % degree != 0 {
                return Err(CharacterError::SelfCheckFailed {
                    check: "degree",
                    detail: format!(
                        "chi(1)^2 lifted to {square}, which is not the square of a divisor of \
                         |G| = {order}"
                    ),
                });
            }

            // χ(g_i) mod p = χ(1) · ω_i / |K_i|.
            let residues: Vec<u64> = (0..r)
                .map(|i| {
                    dixon::mul_mod(dixon::mul_mod(degree % p, omega[i], p), size_inverses[i], p)
                })
                .collect();

            let mut row = Vec::with_capacity(r);
            for powers in &power_classes {
                row.push(lift_value(
                    &field, &zeta, p, w, exponent, degree, &residues, powers,
                )?);
            }
            rows.push((degree, row));
        }

        // Deterministic row order: the trivial character first, then by degree,
        // then by the coordinates of the values. The coordinates are exact
        // rationals, so this ordering does not depend on the working prime.
        let one = field.one();
        let width = field.degree();
        rows.sort_by_cached_key(|(degree, row)| {
            let trivial = if row.iter().all(|v| *v == one) {
                0u8
            } else {
                1u8
            };
            let coords: Vec<Vec<Rational>> = row.iter().map(|v| padded(v, width)).collect();
            (trivial, *degree, coords)
        });

        let degrees: Vec<u64> = rows.iter().map(|(d, _)| *d).collect();
        let values: Vec<Vec<NumberFieldElement>> = rows.into_iter().map(|(_, row)| row).collect();

        let table = CharacterTable {
            field,
            exponent,
            prime: p,
            classes,
            degrees,
            values,
        };
        table.verify()?;
        Ok(table)
    }

    /// The cyclotomic field `ℚ(ζ_{exp G})` every value lives in.
    pub fn field(&self) -> &NumberField {
        &self.field
    }

    /// `exp G` — so the field is `ℚ(ζ_n)` for this `n`.
    pub fn exponent(&self) -> u64 {
        self.exponent
    }

    /// The prime Dixon's reduction ran over. Exposed for reproducibility; the
    /// returned values do not depend on it.
    pub fn working_prime(&self) -> u64 {
        self.prime
    }

    /// The conjugacy classes indexing the columns.
    pub fn classes(&self) -> &ConjugacyClasses {
        &self.classes
    }

    /// The number of irreducible characters — equal, always, to the number of
    /// conjugacy classes.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Never true: every group has at least the trivial character.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The degrees `χ_i(1)`, in row order: ascending, with the trivial
    /// character's `1` first.
    pub fn degrees(&self) -> &[u64] {
        &self.degrees
    }

    /// Row `i` of the table: `χ_i` evaluated on every class.
    pub fn character(&self, i: usize) -> Result<&[NumberFieldElement], CharacterError> {
        self.values
            .get(i)
            .map(|row| row.as_slice())
            .ok_or(CharacterError::ClassIndexOutOfRange {
                index: i,
                classes: self.values.len(),
            })
    }

    /// `χ_i(g_c)` for the representative of class `c`.
    pub fn value(&self, i: usize, c: usize) -> Result<&NumberFieldElement, CharacterError> {
        let row = self
            .values
            .get(i)
            .ok_or(CharacterError::ClassIndexOutOfRange {
                index: i,
                classes: self.values.len(),
            })?;
        row.get(c).ok_or(CharacterError::ClassIndexOutOfRange {
            index: c,
            classes: row.len(),
        })
    }

    /// `⟨χ_i, χ_j⟩ = (1/|G|) Σ_c |K_c| χ_i(g_c) χ_j(g_c⁻¹)`.
    ///
    /// `δ_ij` for the irreducible characters of this table — which is checked
    /// before the table is returned, so this accessor exists for callers who
    /// want to re-check it, or to test a class function of their own against
    /// the rows.
    pub fn inner_product(&self, i: usize, j: usize) -> Result<Rational, CharacterError> {
        let left = self.character(i)?;
        let right = self.character(j)?;
        let mut total = self.field.zero();
        for (c, value) in left.iter().enumerate() {
            let inverse = self.classes.inverse_class(c)?;
            let weight = Rational::from(self.classes.class(c)?.size());
            let term = value
                .mul(&right[inverse])?
                .mul(&self.field.rational(&weight))?;
            total = total.add(&term)?;
        }
        let order = Rational::from(self.classes.group_order().clone());
        let scaled = total.mul(&self.field.rational(&order.recip()))?;
        as_rational(&scaled).ok_or_else(|| CharacterError::SelfCheckFailed {
            check: "row orthogonality",
            detail: format!(
                "<chi_{i}, chi_{j}> came out as {scaled}, which is not a rational number"
            ),
        })
    }

    /// Re-run every invariant this table was checked against before it was
    /// returned.
    ///
    /// Calling it again is redundant by construction and cheap enough to be
    /// worth having: it is how a test asserts that the checks are real rather
    /// than that they were skipped.
    pub fn verify(&self) -> Result<(), CharacterError> {
        let r = self.classes.len();
        if self.values.len() != r {
            return Err(CharacterError::SelfCheckFailed {
                check: "character count",
                detail: format!(
                    "{} irreducible characters for {r} conjugacy classes",
                    self.values.len()
                ),
            });
        }

        // Degrees: each divides |G|, and the squares sum to |G|.
        let order = self.classes.group_order().clone();
        let mut square_sum = Integer::new();
        for (i, &d) in self.degrees.iter().enumerate() {
            if d == 0 || !order.is_divisible(&Integer::from(d)) {
                return Err(CharacterError::SelfCheckFailed {
                    check: "degree",
                    detail: format!("chi_{i}(1) = {d} does not divide |G| = {order}"),
                });
            }
            let value = self.value(i, 0)?;
            if *value != self.field.rational(&Rational::from(d)) {
                return Err(CharacterError::SelfCheckFailed {
                    check: "degree",
                    detail: format!("chi_{i}(1) is recorded as {d} but the table holds {value}"),
                });
            }
            square_sum += Integer::from(d) * Integer::from(d);
        }
        if square_sum != order {
            return Err(CharacterError::SelfCheckFailed {
                check: "degree sum",
                detail: format!("the squared degrees sum to {square_sum}, not to |G| = {order}"),
            });
        }

        // Row orthogonality. Both relations are `O(r³)` multiplications in
        // `ℚ(ζ_e)`, and that is what dominates the cost of a table: `r = 60`
        // takes about a second and `r = 100` about fifteen. The number of
        // *classes*, not the group order, is the practical ceiling.
        for i in 0..r {
            for j in 0..r {
                let product = self.inner_product(i, j)?;
                let expected = if i == j { 1 } else { 0 };
                if product != expected {
                    return Err(CharacterError::SelfCheckFailed {
                        check: "row orthogonality",
                        detail: format!("<chi_{i}, chi_{j}> = {product}, expected {expected}"),
                    });
                }
            }
        }

        // Column orthogonality: `Σ_i χ_i(g) χ_i(h⁻¹)` is `|C_G(g)|` when
        // `g ~ h` and zero otherwise.
        for c in 0..r {
            for d in 0..r {
                let inverse = self.classes.inverse_class(d)?;
                let mut total = self.field.zero();
                for i in 0..r {
                    total = total.add(&self.value(i, c)?.mul(self.value(i, inverse)?)?)?;
                }
                let expected = if c == d {
                    Rational::from(self.classes.class(c)?.centraliser_order().clone())
                } else {
                    Rational::new()
                };
                if total != self.field.rational(&expected) {
                    return Err(CharacterError::SelfCheckFailed {
                        check: "column orthogonality",
                        detail: format!(
                            "sum_i chi_i(g_{c}) conj(chi_i(g_{d})) = {total}, expected {expected}"
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// The table as text: one row per character, one column per class, with
    /// each value written in terms of the generator of `ℚ(ζ_{exp G})`.
    ///
    /// The header names the field, then labels each column `order^size` — the
    /// order of the class representative and the size of the class. Those two
    /// numbers are what distinguish groups that share a character table (`D_4`
    /// and `Q_8`), so they are in the header rather than left to
    /// [`CharacterTable::classes`].
    pub fn render(&self) -> String {
        let mut cells: Vec<Vec<String>> = Vec::with_capacity(self.values.len() + 1);
        let mut header = vec![format!("Q(zeta_{})", self.exponent)];
        for class in self.classes.classes() {
            header.push(format!("{}^{}", class.element_order(), class.size()));
        }
        cells.push(header);
        for (i, row) in self.values.iter().enumerate() {
            let mut line = vec![format!("chi_{i}")];
            for value in row {
                line.push(value.to_string());
            }
            cells.push(line);
        }
        let columns = cells[0].len();
        let widths: Vec<usize> = (0..columns)
            .map(|j| {
                cells
                    .iter()
                    .map(|row| row[j].chars().count())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let mut out = String::new();
        for row in &cells {
            for (j, cell) in row.iter().enumerate() {
                if j > 0 {
                    out.push_str("  ");
                }
                out.push_str(cell);
                for _ in cell.chars().count()..widths[j] {
                    out.push(' ');
                }
            }
            while out.ends_with(' ') {
                out.pop();
            }
            out.push('\n');
        }
        out
    }
}

impl std::fmt::Display for CharacterTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render())
    }
}

/// One character value, lifted from its residues mod `p` to an exact element of
/// `ℚ(ζ_e)`.
///
/// `powers[l]` is the class of `g^l`, so `residues[powers[l]]` is `χ(g^l) mod p`
/// and the inverse DFT recovers the eigenvalue multiplicities `n_k` of `ρ(g)`.
/// Both `0 ≤ n_k ≤ χ(1)` and `Σ_k n_k = χ(1)` are checked: they are what makes
/// the lift a fact rather than a hope.
#[allow(clippy::too_many_arguments)]
fn lift_value(
    field: &NumberField,
    zeta: &NumberFieldElement,
    p: u64,
    w: u64,
    exponent: u64,
    degree: u64,
    residues: &[u64],
    powers: &[usize],
) -> Result<NumberFieldElement, CharacterError> {
    let m = powers.len() as u64;
    if m == 0 || exponent % m != 0 {
        return Err(CharacterError::Internal {
            detail: format!("an element order of {m} does not divide exp G = {exponent}"),
        });
    }
    // ζ_m ↦ z under the same ring homomorphism ζ_e ↦ w.
    let z = dixon::pow_mod(w, exponent / m, p);
    let m_inverse = dixon::inv_mod(m % p, p)?;
    let mut total = field.zero();
    let mut multiplicity_sum = 0u64;
    for k in 0..m {
        let mut accumulator = 0u64;
        for (l, &class) in powers.iter().enumerate() {
            let shift = (m - (k * l as u64) % m) % m;
            let term = dixon::mul_mod(residues[class], dixon::pow_mod(z, shift, p), p);
            accumulator = (accumulator + term) % p;
        }
        let multiplicity = dixon::mul_mod(accumulator, m_inverse, p);
        if multiplicity > degree {
            return Err(CharacterError::SelfCheckFailed {
                check: "eigenvalue multiplicity",
                detail: format!(
                    "the multiplicity of zeta_{m}^{k} lifted to {multiplicity}, above the \
                     character degree {degree}"
                ),
            });
        }
        multiplicity_sum += multiplicity;
        if multiplicity != 0 {
            let power = zeta.pow(k * (exponent / m));
            let scaled = power.mul(&field.rational(&Rational::from(multiplicity)))?;
            total = total.add(&scaled)?;
        }
    }
    if multiplicity_sum != degree {
        return Err(CharacterError::SelfCheckFailed {
            check: "eigenvalue multiplicity",
            detail: format!(
                "the eigenvalue multiplicities of an element of order {m} sum to \
                 {multiplicity_sum}, not to the character degree {degree}"
            ),
        });
    }
    Ok(total)
}

/// Coefficients padded to the field degree, so two elements always compare on
/// vectors of the same length.
fn padded(value: &NumberFieldElement, width: usize) -> Vec<Rational> {
    let mut coefficients = value.coefficients();
    coefficients.resize(width.max(coefficients.len()), Rational::new());
    coefficients
}

/// A field element that happens to be rational, as a [`Rational`].
fn as_rational(value: &NumberFieldElement) -> Option<Rational> {
    let coefficients = value.coefficients();
    let constant = coefficients.first().cloned().unwrap_or_else(Rational::new);
    if coefficients.iter().skip(1).all(|c| *c == 0) {
        Some(constant)
    } else {
        None
    }
}

/// `⌊√n⌋`.
fn isqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = (n as f64).sqrt() as u64;
    while x > 0 && x.saturating_mul(x) > n {
        x -= 1;
    }
    while (x + 1).saturating_mul(x + 1) <= n {
        x += 1;
    }
    x
}

/// Euler's totient `φ(n)`, for `n ≥ 1`.
fn euler_phi(n: u64) -> usize {
    if n == 0 {
        return 0;
    }
    let mut result = n;
    for q in dixon::distinct_prime_factors(n) {
        result = result / q * (q - 1);
    }
    result as usize
}
