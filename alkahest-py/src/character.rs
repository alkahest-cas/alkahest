//! PyO3 bindings for `alkahest_core::character` — conjugacy classes and exact
//! character tables of permutation groups.
//!
//! The Python surface is `alkahest.experimental.ConjugacyClasses`,
//! `ConjugacyClass` and `CharacterTable`; refusals arrive as
//! `alkahest.experimental.CharacterError` with a stable `E-CHAR-NNN` `.code`.
//! See `alkahest_cas::character` for the mathematics, for the two ceilings this
//! binding carries over unchanged, and for the list of what is out of scope.
//!
//! # How a character value crosses the boundary
//!
//! As a `NumberFieldElement` of one cyclotomic field `ℚ(ζ_{exp G})` — the same
//! type `alkahest.experimental.NumberField` produces, with the same exact
//! arithmetic on it. Not a float, and not a string: `table.value(i, c) ** 3`
//! and `u * v` are meant to work, which is how a caller checks that `A_4`'s
//! irrational entries really are cube roots of unity rather than taking it on
//! trust. `CharacterTable.field()` hands back the field itself, so the power
//! basis those values are written in is available rather than implied.
//!
//! Group orders and centraliser orders go out as Python `int` — arbitrary
//! precision, built from decimal text, never truncated. Inner products go out
//! as `fractions.Fraction`, which is exact and is what keeps
//! `table.inner_product(i, i) == 1` a true statement rather than a near miss.

use pyo3::prelude::*;
use pyo3::types::{PyInt, PyModule};

use alkahest_core::character::{
    CharacterError, CharacterTable, ConjugacyClass, ConjugacyClasses, DEFAULT_CHARACTER_TABLE_CAP,
    DEFAULT_CLASS_ENUMERATION_CAP, MAX_CLASS_ENUMERATION_CAP, MAX_EXPONENT_FIELD_DEGREE,
};

use crate::group::PyPermutation;
use crate::numfield::{PyNumberField, PyNumberFieldElement};

pyo3::create_exception!(alkahest, PyCharacterError, crate::PyAlkahestError);

fn char_err(e: CharacterError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyCharacterError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// A `rug::Integer` as an exact Python `int`, at any size.
fn big_int(py: Python<'_>, value: &rug::Integer) -> PyResult<PyObject> {
    let int_cls = py.get_type_bound::<PyInt>();
    Ok(int_cls.call1((value.to_string(),))?.into_py(py))
}

/// A `rug::Rational` as a `fractions.Fraction`, via its `"p/q"` text.
fn py_fraction(py: Python<'_>, r: &rug::Rational) -> PyResult<PyObject> {
    Ok(py
        .import_bound("fractions")?
        .getattr("Fraction")?
        .call1((r.to_string(),))?
        .into())
}

// ---------------------------------------------------------------------------
// ConjugacyClass
// ---------------------------------------------------------------------------

/// One conjugacy class of a permutation group.
///
/// ``representative`` is the **lexicographically smallest** element of the
/// class, so it does not depend on the order the generators were given in and
/// the class ordering is reproducible run to run.
#[pyclass(name = "ConjugacyClass", module = "alkahest")]
#[derive(Clone)]
pub struct PyConjugacyClass {
    inner: ConjugacyClass,
}

#[pymethods]
impl PyConjugacyClass {
    /// The class representative: the lexicographically smallest member.
    #[getter]
    fn representative(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.representative().clone(),
        }
    }

    /// ``|K|``, the number of elements in the class.
    #[getter]
    fn size(&self) -> u64 {
        self.inner.size()
    }

    /// ``|C_G(g)| = |G| / |K|``, exactly, at arbitrary precision.
    ///
    /// Orbit–stabilizer for the conjugation action: the stabilizer of ``g`` is
    /// its centraliser. Only the *order* is available here, not the centraliser
    /// as a subgroup.
    #[getter]
    fn centraliser_order(&self, py: Python<'_>) -> PyResult<PyObject> {
        big_int(py, self.inner.centraliser_order())
    }

    /// The order of the representative — a class invariant, since conjugate
    /// elements have equal order.
    #[getter]
    fn element_order(&self) -> u64 {
        self.inner.element_order()
    }

    fn __repr__(&self) -> String {
        format!(
            "ConjugacyClass(representative={}, size={}, element_order={})",
            self.inner.representative(),
            self.inner.size(),
            self.inner.element_order()
        )
    }
}

// ---------------------------------------------------------------------------
// ConjugacyClasses
// ---------------------------------------------------------------------------

/// The conjugacy classes of a :class:`PermutationGroup`.
///
/// ``ConjugacyClasses(group)`` refuses above
/// ``CHARACTER_DEFAULT_CLASS_CAP`` (``E-CHAR-001``); pass ``cap`` to raise it,
/// up to ``CHARACTER_MAX_CLASS_CAP`` (``E-CHAR-002`` past that). The ceiling is
/// on ``|G|`` and it bounds memory: classes are the orbits of ``G`` acting on
/// itself, so every element is held as a degree-``n`` images array. The group's
/// *order* is still exact at any size — it is the class partition that is
/// refused, and nothing partial is returned.
///
/// Classes are ordered by ``(element_order, size, representative)``, so class
/// ``0`` is always ``{1}``.
#[pyclass(name = "ConjugacyClasses", module = "alkahest")]
#[derive(Clone)]
pub struct PyConjugacyClasses {
    inner: ConjugacyClasses,
}

#[pymethods]
impl PyConjugacyClasses {
    /// ``ConjugacyClasses(group, cap=None)``.
    #[new]
    #[pyo3(signature = (group, cap = None))]
    fn __new__(
        group: &crate::group::PyPermutationGroup,
        cap: Option<u64>,
    ) -> PyResult<PyConjugacyClasses> {
        let cap = cap.unwrap_or(DEFAULT_CLASS_ENUMERATION_CAP);
        ConjugacyClasses::of_with_cap(&group.inner, cap)
            .map(|inner| PyConjugacyClasses { inner })
            .map_err(char_err)
    }

    /// The number of classes — equivalently, of irreducible characters.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// ``classes[i]``, the class at ``i``.
    fn __getitem__(&self, index: usize) -> PyResult<PyConjugacyClass> {
        self.inner
            .class(index)
            .map(|c| PyConjugacyClass { inner: c.clone() })
            .map_err(char_err)
    }

    /// The degree of the underlying permutation group.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// ``|G|``, exactly.
    #[getter]
    fn group_order(&self, py: Python<'_>) -> PyResult<PyObject> {
        big_int(py, self.inner.group_order())
    }

    /// ``exp G``, the lcm of the element orders. Always divides ``|G|``.
    #[getter]
    fn exponent(&self) -> u64 {
        self.inner.exponent()
    }

    /// All the classes, in order.
    fn classes(&self) -> Vec<PyConjugacyClass> {
        self.inner
            .classes()
            .iter()
            .map(|c| PyConjugacyClass { inner: c.clone() })
            .collect()
    }

    /// The class sizes, in order. They divide ``|G|`` and sum to it.
    fn sizes(&self) -> Vec<u64> {
        self.inner.classes().iter().map(|c| c.size()).collect()
    }

    /// The order of each class representative, in order.
    fn element_orders(&self) -> Vec<u64> {
        self.inner
            .classes()
            .iter()
            .map(|c| c.element_order())
            .collect()
    }

    /// The index of the class containing ``element``.
    ///
    /// ``E-CHAR-005`` for an element outside the group — membership is decided
    /// exactly, by sifting — and ``E-GRP-002`` for one of the wrong degree,
    /// which is never repaired by padding with fixed points.
    fn class_of(&self, element: &PyPermutation) -> PyResult<usize> {
        self.inner.class_of(&element.inner).map_err(char_err)
    }

    /// The index of the class of ``g**-1``, for ``g`` in class ``index``.
    ///
    /// This is how complex conjugation reaches a character table:
    /// ``conj(chi(g)) == chi(g**-1)``, so a conjugate value is a column lookup
    /// and needs no embedding into ℂ.
    fn inverse_class(&self, index: usize) -> PyResult<usize> {
        self.inner.inverse_class(index).map_err(char_err)
    }

    /// Every element of class ``index``, sorted.
    fn class_elements(&self, index: usize) -> PyResult<Vec<PyPermutation>> {
        self.inner
            .class_elements(index)
            .map(|members| {
                members
                    .into_iter()
                    .map(|inner| PyPermutation { inner })
                    .collect()
            })
            .map_err(char_err)
    }

    /// ``a_ijk``, the number of pairs ``(x, y)`` in ``K_i x K_j`` whose product
    /// is the representative of class ``k``.
    ///
    /// Independent of which representative is chosen, which is what makes the
    /// class sums a basis of the centre of the group algebra. Products are
    /// formed left-to-right, matching :meth:`Permutation.compose`.
    fn multiplication_coefficient(&self, i: usize, j: usize, k: usize) -> PyResult<u64> {
        self.inner
            .multiplication_coefficient(i, j, k)
            .map_err(char_err)
    }

    /// The class multiplication matrix ``M_k``, with ``M_k[i][j] == a_kij``.
    ///
    /// These are the matrices Dixon's algorithm diagonalises: they commute, and
    /// their common eigenvectors are ``|K_i| chi(g_i) / chi(1)``.
    fn multiplication_matrix(&self, k: usize) -> PyResult<Vec<Vec<u64>>> {
        self.inner.multiplication_matrix(k).map_err(char_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "ConjugacyClasses(degree={}, order={}, classes={})",
            self.inner.degree(),
            self.inner.group_order(),
            self.inner.len()
        )
    }
}

// ---------------------------------------------------------------------------
// CharacterTable
// ---------------------------------------------------------------------------

/// The ordinary character table of a permutation group, with **exact** values.
///
/// Computed by Dixon–Schneider: the class multiplication matrices are
/// diagonalised over GF(p) for a prime ``p = 1 (mod exp G)`` with ``p > |G|``,
/// then each value is lifted back to ``Q(zeta_exp G)`` as a sum of roots of
/// unity with integer multiplicities. Nothing is floating point and nothing
/// degrades to rationals when an irrationality appears: ``A_4`` comes out with
/// cube roots of unity and ``A_5`` with the golden-ratio pair, from the same
/// code path that gives ``S_4`` its integers.
///
/// Rows are irreducible characters — row ``0`` is the trivial one, and the rest
/// ascend by degree — and columns are the classes of :meth:`classes`.
///
/// Row and column orthogonality, ``sum chi_i(1)**2 == |G|``, ``chi_i(1)``
/// dividing ``|G|``, and one character per class are all checked as exact
/// identities **before** the table is returned. A violation is ``E-CHAR-007``
/// and nothing comes back: a caller cannot tell a checked table from an
/// unchecked one once it is in their hands.
///
/// Two ceilings. ``|G|`` is capped at ``CHARACTER_DEFAULT_TABLE_CAP`` by
/// default (``E-CHAR-001``; pass ``cap``), because the class partition holds
/// every element. ``phi(exp G)`` is capped at
/// ``CHARACTER_MAX_EXPONENT_FIELD_DEGREE`` (``E-CHAR-003``) because every value
/// lives in that one cyclotomic field — a limit on the exponent rather than on
/// the order. The two are independent: a group of small exponent passes the
/// second at any order and meets the first instead, while the cyclic group of
/// order 1000 is refused by the second despite being small.
#[pyclass(name = "CharacterTable", module = "alkahest")]
pub struct PyCharacterTable {
    inner: CharacterTable,
}

#[pymethods]
impl PyCharacterTable {
    /// ``CharacterTable(group, cap=None)``.
    #[new]
    #[pyo3(signature = (group, cap = None))]
    fn __new__(
        group: &crate::group::PyPermutationGroup,
        cap: Option<u64>,
    ) -> PyResult<PyCharacterTable> {
        let cap = cap.unwrap_or(DEFAULT_CHARACTER_TABLE_CAP);
        CharacterTable::of_with_cap(&group.inner, cap)
            .map(|inner| PyCharacterTable { inner })
            .map_err(char_err)
    }

    /// The table of an already-computed :class:`ConjugacyClasses`.
    ///
    /// The class partition is the expensive half, so this avoids repeating it
    /// when the classes are wanted for their own sake as well.
    #[staticmethod]
    fn from_classes(classes: &PyConjugacyClasses) -> PyResult<PyCharacterTable> {
        CharacterTable::from_classes(classes.inner.clone())
            .map(|inner| PyCharacterTable { inner })
            .map_err(char_err)
    }

    /// The number of irreducible characters — always the number of classes.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The conjugacy classes indexing the columns.
    fn classes(&self) -> PyConjugacyClasses {
        PyConjugacyClasses {
            inner: self.inner.classes().clone(),
        }
    }

    /// The cyclotomic field ``Q(zeta_exp G)`` every value lives in.
    fn field(&self) -> PyNumberField {
        PyNumberField {
            inner: self.inner.field().clone(),
        }
    }

    /// ``exp G`` — so the field is ``Q(zeta_n)`` for this ``n``.
    #[getter]
    fn exponent(&self) -> u64 {
        self.inner.exponent()
    }

    /// The prime Dixon's reduction ran over.
    ///
    /// Exposed for reproducibility only: the returned values are exact and do
    /// not depend on it.
    #[getter]
    fn working_prime(&self) -> u64 {
        self.inner.working_prime()
    }

    /// The degrees ``chi_i(1)``, ascending, with the trivial character's ``1``
    /// first. Each divides ``|G|`` and the squares sum to ``|G|``.
    fn degrees(&self) -> Vec<u64> {
        self.inner.degrees().to_vec()
    }

    /// Row ``i``: ``chi_i`` on every class, as exact field elements.
    fn character(&self, i: usize) -> PyResult<Vec<PyNumberFieldElement>> {
        self.inner
            .character(i)
            .map(|row| {
                row.iter()
                    .map(|v| PyNumberFieldElement { inner: v.clone() })
                    .collect()
            })
            .map_err(char_err)
    }

    /// ``chi_i(g_c)`` for the representative of class ``c``.
    fn value(&self, i: usize, c: usize) -> PyResult<PyNumberFieldElement> {
        self.inner
            .value(i, c)
            .map(|v| PyNumberFieldElement { inner: v.clone() })
            .map_err(char_err)
    }

    /// The whole table as a list of rows.
    fn table(&self) -> PyResult<Vec<Vec<PyNumberFieldElement>>> {
        (0..self.inner.len()).map(|i| self.character(i)).collect()
    }

    /// ``<chi_i, chi_j> = (1/|G|) sum_c |K_c| chi_i(g_c) chi_j(g_c**-1)``, as a
    /// ``Fraction``.
    ///
    /// ``1`` when ``i == j`` and ``0`` otherwise for the rows of this table —
    /// already checked before the table was returned, so this is here for
    /// callers who want to re-check it, or to test a class function of their own
    /// against the irreducibles.
    fn inner_product(&self, py: Python<'_>, i: usize, j: usize) -> PyResult<PyObject> {
        let value = self.inner.inner_product(i, j).map_err(char_err)?;
        py_fraction(py, &value)
    }

    /// Re-run every invariant the table was checked against before it was
    /// returned. Raises ``E-CHAR-007`` on a violation.
    fn verify(&self) -> PyResult<()> {
        self.inner.verify().map_err(char_err)
    }

    fn __str__(&self) -> String {
        self.inner.render()
    }

    fn __repr__(&self) -> String {
        format!(
            "CharacterTable(order={}, classes={}, degrees={:?})",
            self.inner.classes().group_order(),
            self.inner.classes().len(),
            self.inner.degrees()
        )
    }
}

/// Register the character-theory surface on the `alkahest` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConjugacyClass>()?;
    m.add_class::<PyConjugacyClasses>()?;
    m.add_class::<PyCharacterTable>()?;
    m.add(
        "CharacterError",
        m.py().get_type_bound::<PyCharacterError>(),
    )?;
    m.add("CHARACTER_DEFAULT_CLASS_CAP", DEFAULT_CLASS_ENUMERATION_CAP)?;
    m.add("CHARACTER_MAX_CLASS_CAP", MAX_CLASS_ENUMERATION_CAP)?;
    m.add("CHARACTER_DEFAULT_TABLE_CAP", DEFAULT_CHARACTER_TABLE_CAP)?;
    m.add(
        "CHARACTER_MAX_EXPONENT_FIELD_DEGREE",
        MAX_EXPONENT_FIELD_DEGREE,
    )?;
    Ok(())
}
