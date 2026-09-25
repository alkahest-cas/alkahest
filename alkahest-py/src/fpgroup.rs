//! PyO3 bindings for `alkahest_core::fpgroup` — finitely presented groups,
//! Todd–Coxeter coset enumeration, Reidemeister–Schreier, and low-degree group
//! cohomology.
//!
//! The Python surface lives under `alkahest.experimental`. See
//! `alkahest_cas::fpgroup` for the mathematics and the scope limits; this module
//! adds nothing and hides nothing.
//!
//! # The refusal to read before the first call
//!
//! Almost every question about a presentation is undecidable, so `order()`,
//! `index()` and the rest **can refuse**, and the two refusals are different
//! facts:
//!
//! * ``E-FPGRP-004`` — coset enumeration hit its cap. This is **not** a claim
//!   that the group is infinite, nor that it is finite. Raise ``max_cosets`` and
//!   try again, or ask ``abelian_invariants()``, which always terminates.
//! * ``E-FPGRP-005`` — the group **is** infinite, proved, because its
//!   abelianisation has an infinite cyclic factor.
//!
//! ``FpGroup("a b", "a^2 b^3 (ab)^7")`` — the `(2,3,7)` triangle group — is
//! infinite with a *trivial* abelianisation, so it lands on ``E-FPGRP-004``.
//! Branch on ``.code``; never on the prose.
//!
//! # Conventions carried over unchanged
//!
//! Word letters are signed and **1-based**; cosets are **0-based** and coset 0
//! is the subgroup `H`; the coset action is on the right, matching the
//! left-to-right composition of `alkahest.Permutation`.

use alkahest_core::fpgroup::{
    default_max_cosets, reidemeister_schreier, AbelianInvariants, CosetTable, FpGroup,
    FpGroupError, GModule, SubgroupPresentation, Word, DEFAULT_MAX_COSETS, MAX_COCHAIN_DIMENSION,
    MAX_COHOMOLOGY_DEGREE, MAX_COHOMOLOGY_GROUP_ORDER, MAX_FREE_RANK, MAX_MODULE_RANK,
};
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyModule};
use rug::Integer;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pyo3::create_exception!(alkahest, PyFpGroupError, crate::PyAlkahestError);

fn fp_err(e: FpGroupError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyFpGroupError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// A `rug::Integer` as an exact Python `int`, at any size.
fn big_int(py: Python<'_>, value: &Integer) -> PyResult<PyObject> {
    let int_cls = py.get_type_bound::<PyInt>();
    Ok(int_cls.call1((value.to_string(),))?.into_py(py))
}

// ---------------------------------------------------------------------------
// Word
// ---------------------------------------------------------------------------

/// A word in a free group: a freely reduced sequence of **signed, 1-based**
/// generator indices.
///
/// ``+k`` is the ``k``-th generator and ``-k`` its inverse; ``0`` is not a
/// letter and the identity is the empty word. Every ``Word`` is freely reduced
/// on construction, so ``==`` is equality in the *free* group — it says nothing
/// about equality in a quotient, which is undecidable.
#[pyclass(name = "Word", module = "alkahest")]
#[derive(Clone)]
pub struct PyWord {
    inner: Word,
}

#[pymethods]
impl PyWord {
    /// ``Word(letters)`` — e.g. ``Word([1, 2, -1, -2])`` is ``[a, b]``.
    #[new]
    #[pyo3(signature = (letters = None))]
    fn __new__(letters: Option<Vec<i32>>) -> PyResult<PyWord> {
        let letters = letters.unwrap_or_default();
        Word::from_letters(&letters)
            .map(|inner| PyWord { inner })
            .map_err(fp_err)
    }

    /// The letters, freely reduced.
    #[getter]
    fn letters(&self) -> Vec<i32> {
        self.inner.letters().to_vec()
    }

    /// The formal inverse.
    fn inverse(&self) -> PyWord {
        PyWord {
            inner: self.inner.inverse(),
        }
    }

    /// Is this the identity?
    fn is_identity(&self) -> bool {
        self.inner.is_empty()
    }

    /// Exponent sum of each of ``rank`` generators — the image in ``Z^rank``.
    fn exponent_sums(&self, rank: usize) -> PyResult<Vec<i64>> {
        self.inner.exponent_sums(rank).map_err(fp_err)
    }

    fn __mul__(&self, other: &PyWord) -> PyWord {
        PyWord {
            inner: self.inner.times(&other.inner),
        }
    }

    fn __pow__(&self, exponent: i32, modulo: Option<i32>) -> PyResult<PyWord> {
        if modulo.is_some() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "pow() with a modulus is not defined for free-group words",
            ));
        }
        Ok(PyWord {
            inner: self.inner.pow(exponent),
        })
    }

    fn __invert__(&self) -> PyWord {
        self.inverse()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __eq__(&self, other: &PyWord) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.letters().hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("Word({:?})", self.inner.letters())
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A subgroup argument: a sequence of `Word`s or of strings to parse.
fn words_from(group: &FpGroup, items: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Word>> {
    let Some(items) = items else {
        return Ok(Vec::new());
    };
    if items.is_none() {
        return Ok(Vec::new());
    }
    // A bare string is iterable, and iterating it would read `"ab"` as the two
    // generators `a` and `b` rather than the one word `ab` — a wrong answer with
    // no error. Ask for the list.
    if items.is_instance_of::<pyo3::types::PyString>() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "pass the subgroup generators as a list, e.g. [\"a*b\"], not as a bare string: \
             iterating a string would read \"ab\" as two generators rather than one word",
        ));
    }
    let mut out = Vec::new();
    for item in items.iter()? {
        let item = item?;
        if let Ok(w) = item.extract::<PyWord>() {
            out.push(w.inner);
        } else if let Ok(text) = item.extract::<String>() {
            out.push(group.parse_word(&text).map_err(fp_err)?);
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "subgroup generators must be Word objects or strings",
            ));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// AbelianInvariants
// ---------------------------------------------------------------------------

/// A finitely generated abelian group, as a free rank plus invariant factors.
///
/// Returned by :meth:`FpGroup.abelian_invariants` and
/// :meth:`FpGroup.cohomology`. ``order()`` is ``None`` exactly when the group is
/// **infinite** — that is a proof, not a failure to decide.
#[pyclass(name = "AbelianInvariants", module = "alkahest")]
#[derive(Clone)]
pub struct PyAbelianInvariants {
    inner: AbelianInvariants,
}

#[pymethods]
impl PyAbelianInvariants {
    /// The number of ``Z`` summands.
    #[getter]
    fn free_rank(&self) -> usize {
        self.inner.free_rank()
    }

    /// The invariant factors, ascending under divisibility and each at least 2.
    #[getter]
    fn torsion(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .torsion()
            .iter()
            .map(|d| big_int(py, d))
            .collect()
    }

    /// Is this the trivial group?
    fn is_trivial(&self) -> bool {
        self.inner.is_trivial()
    }

    /// Is the group finite?
    fn is_finite(&self) -> bool {
        self.inner.is_finite()
    }

    /// The order, or ``None`` when the group is infinite.
    fn order(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match self.inner.order() {
            Some(n) => Ok(Some(big_int(py, &n)?)),
            None => Ok(None),
        }
    }

    fn __eq__(&self, other: &PyAbelianInvariants) -> bool {
        self.inner == other.inner
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("AbelianInvariants({})", self.inner)
    }
}

// ---------------------------------------------------------------------------
// CosetTable
// ---------------------------------------------------------------------------

/// A complete coset table for ``H <= G``, and the permutation representation it
/// is.
///
/// **Cosets are 0-based and coset 0 is** ``H``. ``rows()[c][2*k]`` is the coset
/// ``c`` maps to under generator ``k``, and ``[2*k+1]`` under its inverse.
#[pyclass(name = "CosetTable", module = "alkahest")]
#[derive(Clone)]
pub struct PyCosetTable {
    inner: CosetTable,
}

#[pymethods]
impl PyCosetTable {
    /// ``[G:H]``.
    #[getter]
    fn index(&self) -> usize {
        self.inner.index()
    }

    /// The number of generators of the group.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Cosets ever defined, dead ones included — how much work the enumeration
    /// did relative to its answer.
    #[getter]
    fn cosets_defined(&self) -> usize {
        self.inner.cosets_defined()
    }

    /// How many coincidences were processed.
    #[getter]
    fn coincidences(&self) -> usize {
        self.inner.coincidences()
    }

    /// The coset cap that was in force.
    #[getter]
    fn max_cosets(&self) -> usize {
        self.inner.max_cosets()
    }

    /// The enumeration strategy, as a string.
    #[getter]
    fn strategy(&self) -> &'static str {
        self.inner.strategy()
    }

    /// The whole table as a list of rows.
    fn rows(&self) -> Vec<Vec<usize>> {
        self.inner.rows()
    }

    /// The image of ``coset`` under the signed letter ``letter``.
    fn image(&self, coset: usize, letter: i32) -> PyResult<usize> {
        self.inner.image(coset, letter).map_err(fp_err)
    }

    /// Trace a word from ``coset``: the coset ``coset * word``.
    fn trace(&self, coset: usize, word: &PyWord) -> PyResult<usize> {
        self.inner.trace(coset, &word.inner).map_err(fp_err)
    }

    /// A Schreier transversal: one word per coset, reaching it from ``H``.
    fn transversal(&self) -> PyResult<Vec<PyWord>> {
        Ok(self
            .inner
            .transversal()
            .map_err(fp_err)?
            .into_iter()
            .map(|inner| PyWord { inner })
            .collect())
    }

    /// One :class:`alkahest.Permutation` per generator, of degree ``index``.
    fn permutations(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let cls = py.get_type_bound::<crate::group::PyPermutation>();
        let mut out = Vec::new();
        for p in self.inner.permutations().map_err(fp_err)? {
            out.push(cls.call1((p.images().to_vec(),))?.into_py(py));
        }
        Ok(out)
    }

    /// The action of ``G`` on the cosets, as a
    /// :class:`alkahest.PermutationGroup` of degree ``index``.
    ///
    /// For ``H = 1`` this is the regular representation, and its ``order()`` —
    /// Schreier–Sims, in the permutation-group module — independently recomputes
    /// ``|G|``.
    fn permutation_group(&self) -> PyResult<crate::group::PyPermutationGroup> {
        self.inner
            .permutation_group()
            .map(crate::group::PyPermutationGroup::wrap)
            .map_err(fp_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "CosetTable(index={}, rank={}, cosets_defined={}, coincidences={})",
            self.inner.index(),
            self.inner.rank(),
            self.inner.cosets_defined(),
            self.inner.coincidences()
        )
    }
}

// ---------------------------------------------------------------------------
// SubgroupPresentation
// ---------------------------------------------------------------------------

/// A Reidemeister–Schreier presentation of a finite-index subgroup, with each
/// of its generators written as a word in the **parent** group's generators.
///
/// Not simplified: there is no Tietze pass, so a cyclic subgroup of index 30 in
/// a two-generator group arrives with 31 generators.
#[pyclass(name = "SubgroupPresentation", module = "alkahest")]
#[derive(Clone)]
pub struct PySubgroupPresentation {
    inner: SubgroupPresentation,
}

#[pymethods]
impl PySubgroupPresentation {
    /// ``[G:H]``.
    #[getter]
    fn index(&self) -> usize {
        self.inner.index()
    }

    /// The number of Schreier generators, ``[G:H]*|X| - [G:H] + 1``.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// The presentation of ``H``, on generators named ``y1, y2, …``.
    fn presentation(&self) -> PyFpGroup {
        PyFpGroup {
            inner: self.inner.presentation().clone(),
        }
    }

    /// Each generator of that presentation as a word in the parent's
    /// generators.
    fn generator_words(&self) -> Vec<PyWord> {
        self.inner
            .generator_words()
            .iter()
            .map(|w| PyWord { inner: w.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SubgroupPresentation(index={}, rank={}, relators={})",
            self.inner.index(),
            self.inner.rank(),
            self.inner.presentation().relators().len()
        )
    }
}

// ---------------------------------------------------------------------------
// FpGroup
// ---------------------------------------------------------------------------

/// A finitely presented group ``<X | R>``.
///
/// ``FpGroup(["a", "b"], ["a^2", "b^3", "(a*b)^5"])`` is ``A5``. Relators are
/// products of powers of generator names and parenthesised sub-words; ``1`` is
/// the identity, and juxtaposition means multiplication, so ``"ab"`` and
/// ``"a*b"`` are the same word.
#[pyclass(name = "FpGroup", module = "alkahest")]
#[derive(Clone)]
pub struct PyFpGroup {
    inner: FpGroup,
}

#[pymethods]
impl PyFpGroup {
    /// ``FpGroup(generators, relators)``.
    #[new]
    #[pyo3(signature = (generators, relators = None))]
    fn __new__(generators: Vec<String>, relators: Option<Vec<String>>) -> PyResult<PyFpGroup> {
        let relators = relators.unwrap_or_default();
        FpGroup::from_strings(&generators, &relators)
            .map(|inner| PyFpGroup { inner })
            .map_err(fp_err)
    }

    /// The free group of the given rank, with generators named ``x1 … xn``.
    #[staticmethod]
    fn free(rank: usize) -> PyResult<PyFpGroup> {
        FpGroup::free(rank)
            .map(|inner| PyFpGroup { inner })
            .map_err(fp_err)
    }

    /// The generator names.
    #[getter]
    fn generators(&self) -> Vec<String> {
        self.inner.generator_names().to_vec()
    }

    /// The number of generators.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// The relators, as :class:`Word` objects.
    #[getter]
    fn relators(&self) -> Vec<PyWord> {
        self.inner
            .relators()
            .iter()
            .map(|w| PyWord { inner: w.clone() })
            .collect()
    }

    /// Parse a word in this presentation's generators.
    fn parse(&self, text: &str) -> PyResult<PyWord> {
        self.inner
            .parse_word(text)
            .map(|inner| PyWord { inner })
            .map_err(fp_err)
    }

    /// The ``index``-th generator (0-based) as a word.
    fn generator(&self, index: usize) -> PyResult<PyWord> {
        self.inner
            .free_group()
            .generator(index)
            .map(|inner| PyWord { inner })
            .map_err(fp_err)
    }

    /// ``|G|`` — the index of the trivial subgroup.
    ///
    /// Raises ``E-FPGRP-005`` when the group is **proved** infinite, and
    /// ``E-FPGRP-004`` when the enumeration did not finish within
    /// ``max_cosets`` — which is not a claim either way about finiteness.
    #[pyo3(signature = (max_cosets = None))]
    fn order(&self, py: Python<'_>, max_cosets: Option<usize>) -> PyResult<PyObject> {
        let cap = max_cosets.unwrap_or_else(|| default_max_cosets(self.inner.rank()));
        let n = self.inner.order_with_cap(cap).map_err(fp_err)?;
        big_int(py, &n)
    }

    /// ``[G:H]`` for the subgroup generated by ``subgroup``.
    #[pyo3(signature = (subgroup = None, max_cosets = None))]
    fn index(
        &self,
        subgroup: Option<&Bound<'_, PyAny>>,
        max_cosets: Option<usize>,
    ) -> PyResult<usize> {
        let words = words_from(&self.inner, subgroup)?;
        let cap = max_cosets.unwrap_or_else(|| default_max_cosets(self.inner.rank()));
        self.inner.index_with_cap(&words, cap).map_err(fp_err)
    }

    /// The coset table of the subgroup generated by ``subgroup`` (the trivial
    /// subgroup by default).
    #[pyo3(signature = (subgroup = None, max_cosets = None))]
    fn coset_table(
        &self,
        subgroup: Option<&Bound<'_, PyAny>>,
        max_cosets: Option<usize>,
    ) -> PyResult<PyCosetTable> {
        let words = words_from(&self.inner, subgroup)?;
        let cap = max_cosets.unwrap_or_else(|| default_max_cosets(self.inner.rank()));
        self.inner
            .coset_table_with_cap(&words, cap)
            .map(|inner| PyCosetTable { inner })
            .map_err(fp_err)
    }

    /// The permutation representation on the cosets of ``subgroup``.
    #[pyo3(signature = (subgroup = None, max_cosets = None))]
    fn permutation_group(
        &self,
        subgroup: Option<&Bound<'_, PyAny>>,
        max_cosets: Option<usize>,
    ) -> PyResult<crate::group::PyPermutationGroup> {
        let words = words_from(&self.inner, subgroup)?;
        let cap = max_cosets.unwrap_or_else(|| default_max_cosets(self.inner.rank()));
        self.inner
            .coset_table_with_cap(&words, cap)
            .and_then(|t| t.permutation_group())
            .map(crate::group::PyPermutationGroup::wrap)
            .map_err(fp_err)
    }

    /// The multiplication table of a finite ``G``, indexed so that element 0 is
    /// the identity.
    fn multiplication_table(&self) -> PyResult<Vec<Vec<usize>>> {
        self.inner.multiplication_table().map_err(fp_err)
    }

    /// ``G/[G, G]`` from the Smith normal form of the relation matrix.
    ///
    /// **Always terminates** — this is linear algebra over ``Z``, not coset
    /// enumeration. An infinite cyclic factor here proves ``G`` is infinite.
    fn abelian_invariants(&self) -> PyResult<PyAbelianInvariants> {
        self.inner
            .abelian_invariants()
            .map(|inner| PyAbelianInvariants { inner })
            .map_err(fp_err)
    }

    /// The relation matrix: one row per relator, one column per generator,
    /// entry the exponent sum.
    fn relation_matrix(&self) -> PyResult<Vec<Vec<i64>>> {
        self.inner.relation_matrix().map_err(fp_err)
    }

    /// A Reidemeister–Schreier presentation for the subgroup generated by
    /// ``subgroup``.
    #[pyo3(signature = (subgroup = None, max_cosets = None))]
    fn subgroup_presentation(
        &self,
        subgroup: Option<&Bound<'_, PyAny>>,
        max_cosets: Option<usize>,
    ) -> PyResult<PySubgroupPresentation> {
        let words = words_from(&self.inner, subgroup)?;
        let cap = max_cosets.unwrap_or_else(|| default_max_cosets(self.inner.rank()));
        reidemeister_schreier(&self.inner, &words, cap)
            .map(|inner| PySubgroupPresentation { inner })
            .map_err(fp_err)
    }

    /// ``H^degree(G, M)`` for ``degree`` in ``0, 1, 2``.
    ///
    /// ``invariants`` describes ``M = Z/d1 + … + Z/dk``, with ``0`` meaning a
    /// ``Z`` summand. ``action`` is one ``k x k`` integer matrix per generator
    /// of ``G``, acting on the left of column vectors; omitting it means the
    /// **trivial** action.
    ///
    /// ``H^2`` classifies extensions of ``M`` by ``G`` — central extensions when
    /// the action is trivial. Size limits are ``|G| <=
    /// FPGROUP_MAX_COHOMOLOGY_GROUP_ORDER``, ``k <= FPGROUP_MAX_MODULE_RANK``
    /// and ``k * |G|**(degree+1) <= FPGROUP_MAX_COCHAIN_DIMENSION``; past those
    /// it raises ``E-FPGRP-009`` rather than running for an hour.
    ///
    /// An ``action`` that is not one — some relator not acting as the identity
    /// on ``M`` — raises ``E-FPGRP-011``. It is never silently used.
    #[pyo3(signature = (degree, invariants, action = None))]
    fn cohomology(
        &self,
        degree: usize,
        invariants: Vec<i64>,
        action: Option<Vec<Vec<Vec<i64>>>>,
    ) -> PyResult<PyAbelianInvariants> {
        let invariants: Vec<Integer> = invariants.into_iter().map(Integer::from).collect();
        let module = match action {
            None => GModule::trivial(invariants, self.inner.rank()).map_err(fp_err)?,
            Some(matrices) => {
                let matrices: Vec<Vec<Vec<Integer>>> = matrices
                    .into_iter()
                    .map(|m| {
                        m.into_iter()
                            .map(|row| row.into_iter().map(Integer::from).collect())
                            .collect()
                    })
                    .collect();
                GModule::new(invariants, matrices).map_err(fp_err)?
            }
        };
        self.inner
            .cohomology(degree, &module)
            .map(|inner| PyAbelianInvariants { inner })
            .map_err(fp_err)
    }

    fn __eq__(&self, other: &PyFpGroup) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("FpGroup({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Register the finitely-presented-group surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWord>()?;
    m.add_class::<PyFpGroup>()?;
    m.add_class::<PyCosetTable>()?;
    m.add_class::<PySubgroupPresentation>()?;
    m.add_class::<PyAbelianInvariants>()?;
    m.add("FpGroupError", m.py().get_type_bound::<PyFpGroupError>())?;
    m.add("FPGROUP_DEFAULT_MAX_COSETS", DEFAULT_MAX_COSETS)?;
    m.add("FPGROUP_MAX_FREE_RANK", MAX_FREE_RANK)?;
    m.add("FPGROUP_MAX_MODULE_RANK", MAX_MODULE_RANK)?;
    m.add(
        "FPGROUP_MAX_COHOMOLOGY_GROUP_ORDER",
        MAX_COHOMOLOGY_GROUP_ORDER,
    )?;
    m.add("FPGROUP_MAX_COCHAIN_DIMENSION", MAX_COCHAIN_DIMENSION)?;
    m.add("FPGROUP_MAX_COHOMOLOGY_DEGREE", MAX_COHOMOLOGY_DEGREE)?;
    Ok(())
}

// `FreeGroup` is exported from the core crate but deliberately not bound here:
// its only role in Python is as the alphabet inside `FpGroup`, and
// `FpGroup.parse` already exposes the parser. One fewer type to explain.
