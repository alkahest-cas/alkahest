//! PyO3 bindings for `alkahest_core::group` — permutation groups.
//!
//! The Python surface lives under `alkahest.experimental`; see
//! `alkahest_cas::group` for the mathematics, the scope limits, and the two
//! conventions (**0-based points**, **left-to-right composition**) that this
//! binding carries over unchanged.

use alkahest_core::group::{
    alternating as core_alternating, cyclic as core_cyclic, dihedral as core_dihedral,
    symmetric as core_symmetric, trivial as core_trivial, GroupError, Permutation,
    PermutationGroup, DEFAULT_ELEMENT_CAP, MAX_BSGS_DEGREE, MAX_ELEMENT_CAP,
};
use pyo3::prelude::*;
use pyo3::types::PyInt;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pyo3::create_exception!(alkahest, PyGroupError, crate::PyAlkahestError);

fn group_error_to_py(e: GroupError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyGroupError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// A `rug::Integer` as an exact Python `int`, at any size.
fn big_int(py: Python<'_>, value: &rug::Integer) -> PyResult<PyObject> {
    let int_cls = py.get_type_bound::<PyInt>();
    Ok(int_cls.call1((value.to_string(),))?.into_py(py))
}

// ---------------------------------------------------------------------------
// Permutation
// ---------------------------------------------------------------------------

/// A permutation of the points ``0, 1, …, degree-1``.
///
/// **Points are 0-based.** GAP, the ATLAS and essentially all of the
/// literature number points from 1, so use
/// :meth:`Permutation.from_cycles_one_based` when transcribing a generator
/// rather than subtracting one by hand.
///
/// **Composition is left-to-right.** ``p.compose(q)`` — also written
/// ``p * q`` — applies ``p`` first and ``q`` second, so
/// ``(p * q).apply(i) == q.apply(p.apply(i))``. This is GAP's convention and
/// the opposite of the ``f ∘ g = f(g(x))`` convention used for functions in
/// analysis. Mixing the two yields a group of the right *order* whose elements
/// are the inverses of the ones you wanted.
#[pyclass(name = "Permutation", module = "alkahest")]
#[derive(Clone)]
pub struct PyPermutation {
    pub(crate) inner: Permutation,
}

impl PyPermutation {
    fn wrap(inner: Permutation) -> PyPermutation {
        PyPermutation { inner }
    }
}

#[pymethods]
impl PyPermutation {
    /// ``Permutation(images)`` — ``images[i]`` is the image of point ``i``.
    ///
    /// Raises :class:`GroupError` (``E-GRP-001``) unless ``images`` is a
    /// bijection of ``range(len(images))``.
    #[new]
    fn __new__(images: Vec<usize>) -> PyResult<PyPermutation> {
        Permutation::from_images(images)
            .map(PyPermutation::wrap)
            .map_err(group_error_to_py)
    }

    /// The identity permutation on ``degree`` points.
    #[staticmethod]
    fn identity(degree: usize) -> PyPermutation {
        PyPermutation::wrap(Permutation::identity(degree))
    }

    /// Build from **0-based** disjoint cycles; unnamed points are fixed.
    #[staticmethod]
    fn from_cycles(degree: usize, cycles: Vec<Vec<usize>>) -> PyResult<PyPermutation> {
        Permutation::from_cycles(degree, &cycles)
            .map(PyPermutation::wrap)
            .map_err(group_error_to_py)
    }

    /// Build from **1-based** disjoint cycles on the points ``1..=degree``.
    ///
    /// For transcribing generators out of GAP or the ATLAS unchanged. The
    /// result is still indexed from 0 — only the input is shifted.
    #[staticmethod]
    fn from_cycles_one_based(degree: usize, cycles: Vec<Vec<usize>>) -> PyResult<PyPermutation> {
        Permutation::from_cycles_one_based(degree, &cycles)
            .map(PyPermutation::wrap)
            .map_err(group_error_to_py)
    }

    /// Number of points acted on. A property: it is a length read.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// The images list, ``images()[i]`` being the image of ``i``.
    fn images(&self) -> Vec<usize> {
        self.inner.images().to_vec()
    }

    /// The image of ``point``; raises ``E-GRP-003`` if it is out of range.
    fn apply(&self, point: usize) -> PyResult<usize> {
        self.inner.apply(point).map_err(group_error_to_py)
    }

    /// Apply ``self`` first, then ``other``. See the class docstring.
    fn compose(&self, other: &PyPermutation) -> PyResult<PyPermutation> {
        self.inner
            .compose(&other.inner)
            .map(PyPermutation::wrap)
            .map_err(group_error_to_py)
    }

    /// The inverse permutation.
    fn inverse(&self) -> PyPermutation {
        PyPermutation::wrap(self.inner.inverse())
    }

    /// Re-read this permutation on ``degree`` points, fixing the added ones.
    ///
    /// Never restricts: a smaller degree raises ``E-GRP-002``.
    fn extend_degree(&self, degree: usize) -> PyResult<PyPermutation> {
        self.inner
            .extend_degree(degree)
            .map(PyPermutation::wrap)
            .map_err(group_error_to_py)
    }

    /// ``self`` raised to an integer power, positive, zero or negative.
    fn pow(&self, exponent: i64) -> PyPermutation {
        PyPermutation::wrap(self.inner.pow(exponent))
    }

    /// Is this the identity? A method rather than a property: it is a scan of
    /// the whole images array, not a field read.
    fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    /// The moved points, ascending.
    fn support(&self) -> Vec<usize> {
        self.inner.support()
    }

    /// The non-trivial cycles, each from its least point. Fixed points are
    /// omitted — use :meth:`cycle_type` for the partition of the degree.
    fn cycles(&self) -> Vec<Vec<usize>> {
        self.inner.cycles()
    }

    /// The cycle lengths **including fixed points**, descending. Sums to
    /// :attr:`degree`.
    fn cycle_type(&self) -> Vec<usize> {
        self.inner.cycle_type()
    }

    /// The multiplicative order, as an exact Python ``int``.
    ///
    /// Arbitrary precision on purpose: the largest order in ``S_n`` is
    /// Landau's function, which passes 64 bits a little after degree 180.
    fn order(&self, py: Python<'_>) -> PyResult<PyObject> {
        big_int(py, &self.inner.order())
    }

    /// ``+1`` for an even permutation, ``-1`` for an odd one.
    fn sign(&self) -> i32 {
        self.inner.sign()
    }

    /// Is this permutation even?
    fn is_even(&self) -> bool {
        self.inner.is_even()
    }

    /// ``p * q`` applies ``p`` first, then ``q``.
    fn __mul__(&self, other: &PyPermutation) -> PyResult<PyPermutation> {
        self.compose(other)
    }

    /// ``~p`` is the inverse.
    fn __invert__(&self) -> PyPermutation {
        self.inverse()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyRef<'_, PyPermutation>>() {
            Ok(o) => self.inner == o.inner,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    /// Cycle notation with 0-based points; ``()`` for the identity.
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "Permutation(degree={}, cycles={})",
            self.inner.degree(),
            self.inner
        )
    }
}

// ---------------------------------------------------------------------------
// SiftResult
// ---------------------------------------------------------------------------

/// What sifting an element through a stabilizer chain found.
#[pyclass(name = "SiftResult", module = "alkahest")]
pub struct PySiftResult {
    residue: Permutation,
    level: usize,
    is_member: bool,
}

#[pymethods]
impl PySiftResult {
    /// Is the sifted element a member of the group? A property: a stored flag.
    #[getter]
    fn is_member(&self) -> bool {
        self.is_member
    }

    /// The chain level at which stripping stopped — the number of levels when
    /// it went all the way through. A property: a stored integer.
    #[getter]
    fn level(&self) -> usize {
        self.level
    }

    /// The residue after stripping; the identity exactly when the element is a
    /// member.
    fn residue(&self) -> PyPermutation {
        PyPermutation::wrap(self.residue.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "SiftResult(is_member={}, level={}, residue={})",
            if self.is_member { "True" } else { "False" },
            self.level,
            self.residue
        )
    }
}

// ---------------------------------------------------------------------------
// PermutationGroup
// ---------------------------------------------------------------------------

/// A subgroup of ``S_n`` given by generators.
///
/// The stabilizer chain (Schreier–Sims) is computed on first demand and
/// cached, and everything exact — :meth:`order`, :meth:`contains`,
/// :meth:`elements` — is read off it. Orbits do not need the chain and work at
/// any degree; the chain itself refuses above ``MAX_BSGS_DEGREE``
/// (``E-GRP-005``).
#[pyclass(name = "PermutationGroup", module = "alkahest")]
pub struct PyPermutationGroup {
    pub(crate) inner: PermutationGroup,
}

impl PyPermutationGroup {
    pub(crate) fn wrap(inner: PermutationGroup) -> PyPermutationGroup {
        PyPermutationGroup { inner }
    }
}

#[pymethods]
impl PyPermutationGroup {
    /// ``PermutationGroup(degree, generators)``.
    ///
    /// Every generator must already have this degree; nothing is padded
    /// (``E-GRP-002``). An empty generator list is the trivial group.
    #[new]
    fn __new__(degree: usize, generators: Vec<PyPermutation>) -> PyResult<PyPermutationGroup> {
        let generators = generators.into_iter().map(|g| g.inner).collect();
        PermutationGroup::new(degree, generators)
            .map(PyPermutationGroup::wrap)
            .map_err(group_error_to_py)
    }

    /// The symmetric group ``S_n`` on ``n`` points. ``|S_n| = n!``.
    #[staticmethod]
    fn symmetric(n: usize) -> PyResult<PyPermutationGroup> {
        core_symmetric(n)
            .map(PyPermutationGroup::wrap)
            .map_err(group_error_to_py)
    }

    /// The alternating group ``A_n``. ``|A_n| = n!/2`` for ``n >= 2``.
    #[staticmethod]
    fn alternating(n: usize) -> PyResult<PyPermutationGroup> {
        core_alternating(n)
            .map(PyPermutationGroup::wrap)
            .map_err(group_error_to_py)
    }

    /// The cyclic group ``C_n`` in its regular action. ``|C_n| = n``.
    #[staticmethod]
    fn cyclic(n: usize) -> PyResult<PyPermutationGroup> {
        core_cyclic(n)
            .map(PyPermutationGroup::wrap)
            .map_err(group_error_to_py)
    }

    /// The dihedral group ``D_n`` on the ``n`` vertices of a regular
    /// ``n``-gon. **``|D_n| = 2n``.**
    ///
    /// Raises ``E-GRP-006`` for ``n < 3``, where the degree-``n`` action is not
    /// faithful and the group returned would not have order ``2n``.
    #[staticmethod]
    fn dihedral(n: usize) -> PyResult<PyPermutationGroup> {
        core_dihedral(n)
            .map(PyPermutationGroup::wrap)
            .map_err(group_error_to_py)
    }

    /// The trivial group on ``degree`` points.
    #[staticmethod]
    fn trivial(degree: usize) -> PyPermutationGroup {
        PyPermutationGroup::wrap(core_trivial(degree))
    }

    /// The degree of the ambient symmetric group. A property: a field read.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// The generators, as supplied.
    fn generators(&self) -> Vec<PyPermutation> {
        self.inner
            .generators()
            .iter()
            .cloned()
            .map(PyPermutation::wrap)
            .collect()
    }

    /// ``|G|``, exactly, as a Python ``int`` of any size.
    fn order(&self, py: Python<'_>) -> PyResult<PyObject> {
        let order = self.inner.order().map_err(group_error_to_py)?;
        big_int(py, &order)
    }

    /// The orbit of ``point``, ascending.
    fn orbit(&self, point: usize) -> PyResult<Vec<usize>> {
        Ok(self
            .inner
            .orbit(point)
            .map_err(group_error_to_py)?
            .sorted_points())
    }

    /// The **Schreier vector** of the orbit of ``point``, indexed by point.
    ///
    /// Entry ``beta`` is ``(generator_index, from_point)`` — the generator that
    /// first carried ``from_point`` to ``beta`` — or ``None`` at the base point
    /// and at every point outside the orbit. ``generator_index`` indexes
    /// :meth:`generators`.
    fn schreier_vector(&self, point: usize) -> PyResult<Vec<Option<(usize, usize)>>> {
        let orbit = self.inner.orbit(point).map_err(group_error_to_py)?;
        Ok(orbit
            .schreier_vector()
            .iter()
            .map(|e| e.map(|e| (e.generator, e.from)))
            .collect())
    }

    /// The transversal element ``u`` with ``point^u == target``, or ``None``
    /// when ``target`` is outside the orbit of ``point``.
    fn transversal_element(&self, point: usize, target: usize) -> PyResult<Option<PyPermutation>> {
        let orbit = self.inner.orbit(point).map_err(group_error_to_py)?;
        Ok(orbit
            .transversal_element(target)
            .cloned()
            .map(PyPermutation::wrap))
    }

    /// The orbits of the group on ``0..degree``; together they partition it.
    fn orbits(&self) -> Vec<Vec<usize>> {
        self.inner.orbits()
    }

    /// Is the action transitive — exactly one orbit?
    fn is_transitive(&self) -> bool {
        self.inner.is_transitive()
    }

    /// Is every generator the identity? Decided without a stabilizer chain, so
    /// it works at any degree.
    fn is_trivial(&self) -> bool {
        self.inner.is_trivial()
    }

    /// Is ``p`` an element of this group? By sifting, so no enumeration.
    fn contains(&self, p: &PyPermutation) -> PyResult<bool> {
        self.inner.contains(&p.inner).map_err(group_error_to_py)
    }

    /// ``p in G``, the same test as :meth:`contains`.
    fn __contains__(&self, p: &PyPermutation) -> PyResult<bool> {
        self.contains(p)
    }

    /// Sift ``p`` through the stabilizer chain.
    fn sift(&self, p: &PyPermutation) -> PyResult<PySiftResult> {
        let result = self.inner.sift(&p.inner).map_err(group_error_to_py)?;
        Ok(PySiftResult {
            residue: result.residue().clone(),
            level: result.level(),
            is_member: result.is_member(),
        })
    }

    /// The base ``[b_0, …, b_{k-1}]`` of the stabilizer chain.
    fn base(&self) -> PyResult<Vec<usize>> {
        Ok(self
            .inner
            .stabilizer_chain()
            .map_err(group_error_to_py)?
            .base())
    }

    /// The strong generating set: the union over levels, deduplicated.
    fn strong_generators(&self) -> PyResult<Vec<PyPermutation>> {
        Ok(self
            .inner
            .stabilizer_chain()
            .map_err(group_error_to_py)?
            .strong_generators()
            .into_iter()
            .map(PyPermutation::wrap)
            .collect())
    }

    /// The basic orbit ``Δ_i = b_i^{S_i}`` at each level, ascending within a
    /// level. Their lengths multiply to :meth:`order`.
    fn basic_orbits(&self) -> PyResult<Vec<Vec<usize>>> {
        Ok(self
            .inner
            .stabilizer_chain()
            .map_err(group_error_to_py)?
            .levels()
            .iter()
            .map(|l| l.orbit().sorted_points())
            .collect())
    }

    /// The strong generators ``S_i`` of the ``level``-th stabilizer
    /// ``G_{b_0,…,b_{level-1}}``.
    ///
    /// Raises ``IndexError`` for a level past the end of the chain.
    fn stabilizer_generators(&self, level: usize) -> PyResult<Vec<PyPermutation>> {
        let chain = self.inner.stabilizer_chain().map_err(group_error_to_py)?;
        let levels = chain.levels();
        let entry = levels.get(level).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "stabilizer level {level} is past the end of a chain with {} levels",
                levels.len()
            ))
        })?;
        Ok(entry
            .generators()
            .iter()
            .cloned()
            .map(PyPermutation::wrap)
            .collect())
    }

    /// A uniformly random element, from a deterministic PRNG seeded by
    /// ``seed`` — the same seed always gives the same element.
    fn random_element(&self, seed: u64) -> PyResult<PyPermutation> {
        self.inner
            .random_element(seed)
            .map(PyPermutation::wrap)
            .map_err(group_error_to_py)
    }

    /// Every element of the group, if there are at most ``cap`` of them
    /// (default ``DEFAULT_ELEMENT_CAP``, itself capped at
    /// ``MAX_ELEMENT_CAP``).
    ///
    /// Raises ``E-GRP-004`` above the cap. The order is still exact and still
    /// available from :meth:`order`: it is the *list* that is refused, never
    /// the arithmetic.
    #[pyo3(signature = (cap = None))]
    fn elements(&self, cap: Option<u64>) -> PyResult<Vec<PyPermutation>> {
        let cap = cap.unwrap_or(DEFAULT_ELEMENT_CAP);
        Ok(self
            .inner
            .elements_with_cap(cap)
            .map_err(group_error_to_py)?
            .into_iter()
            .map(PyPermutation::wrap)
            .collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "PermutationGroup(degree={}, generators={})",
            self.inner.degree(),
            self.inner.generators().len()
        )
    }
}

/// Register the permutation-group surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPermutation>()?;
    m.add_class::<PyPermutationGroup>()?;
    m.add_class::<PySiftResult>()?;
    m.add("GroupError", m.py().get_type_bound::<PyGroupError>())?;
    m.add("GROUP_DEFAULT_ELEMENT_CAP", DEFAULT_ELEMENT_CAP)?;
    m.add("GROUP_MAX_ELEMENT_CAP", MAX_ELEMENT_CAP)?;
    m.add("GROUP_MAX_BSGS_DEGREE", MAX_BSGS_DEGREE)?;
    Ok(())
}
