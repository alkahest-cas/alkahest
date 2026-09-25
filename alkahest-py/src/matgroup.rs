//! PyO3 bindings for `alkahest_core::matgroup` — matrix groups over GF(q) from
//! arbitrary generators.
//!
//! The Python surface is `alkahest.experimental.MatGroup`; refusals arrive as
//! `alkahest.experimental.MatGroupError` with a stable `E-MATGRP-NNN` `.code`,
//! except for those raised inside the GF(q) or permutation layers, which keep
//! their own `E-GFQ-NNN` / `E-GRP-NNN`.
//!
//! # The convention that crosses the boundary unchanged
//!
//! **The action is on row vectors, `v -> v @ M`.** So vectors are `1 x d`
//! `GfMatrix` objects, `M @ N` means "apply `M`, then `N`", and the permutation
//! groups returned by `permutation_action_on_*` compose in the same left-to-right
//! order as `alkahest.experimental.Permutation`. The column-vector convention
//! used by most linear-algebra texts is the *other* one.
//!
//! # `order()` is a method, not a property
//!
//! It builds a stabilizer chain, which is real work and can refuse. Only
//! `degree` and `field` are properties here — see `CONTRIBUTING.md` on the
//! accessor convention.

use pyo3::prelude::*;
use pyo3::types::{PyInt, PyType};

use alkahest_core::experimental::{
    gl_order as core_gl_order, sl_order as core_sl_order, sp_order as core_sp_order, MatGroup,
    MatGroupError, DEFAULT_MATGROUP_ELEMENT_CAP, MAX_MATGROUP_COMMUTANT_ELEMENTS,
    MAX_MATGROUP_DEGREE, MAX_MATGROUP_FIELD_ORDER, MAX_MATGROUP_ORBIT, MAX_MATGROUP_SCHREIER_WORK,
};

use crate::ffield::{PyFiniteField, PyGfMatrix};
use crate::group::PyPermutationGroup;

pyo3::create_exception!(alkahest, PyMatGroupError, crate::PyAlkahestError);

fn mg_err(e: MatGroupError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyMatGroupError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// A `rug::Integer` as an exact Python `int`, at any size.
fn big_int(py: Python<'_>, value: &rug::Integer) -> PyResult<PyObject> {
    let int_cls = py.get_type_bound::<PyInt>();
    Ok(int_cls.call1((value.to_string(),))?.into_py(py))
}

fn matrices(items: &[PyRef<'_, PyGfMatrix>]) -> Vec<alkahest_core::ffield::GfMatrix> {
    items.iter().map(|m| m.inner.clone()).collect()
}

fn wrap_all(items: Vec<alkahest_core::ffield::GfMatrix>) -> Vec<PyGfMatrix> {
    items.into_iter().map(PyGfMatrix::wrap).collect()
}

/// The result of sifting a matrix through a stabilizer chain.
#[pyclass(name = "MatSiftResult", module = "alkahest")]
pub struct PyMatSiftResult {
    residue: alkahest_core::ffield::GfMatrix,
    level: usize,
    is_member: bool,
}

#[pymethods]
impl PyMatSiftResult {
    /// What was left after stripping transversal elements — the identity
    /// exactly when the matrix is in the group.
    #[getter]
    fn residue(&self) -> PyGfMatrix {
        PyGfMatrix::wrap(self.residue.clone())
    }

    /// The level at which sifting stopped.
    #[getter]
    fn level(&self) -> usize {
        self.level
    }

    /// Is the sifted matrix in the group?
    #[getter]
    fn is_member(&self) -> bool {
        self.is_member
    }

    fn __repr__(&self) -> String {
        format!(
            "MatSiftResult(level={}, is_member={})",
            self.level,
            if self.is_member { "True" } else { "False" }
        )
    }
}

/// A subgroup of ``GL(d, q)`` given by a list of invertible ``d x d`` matrices.
///
/// ``MatGroup(field, degree, generators)``. An empty generator list is the
/// trivial group, which is why the degree is explicit.
///
/// The action is on **row** vectors, ``v -> v @ M``.
#[pyclass(name = "MatGroup", module = "alkahest")]
pub struct PyMatGroup {
    inner: MatGroup,
}

#[pymethods]
impl PyMatGroup {
    #[new]
    fn new(
        field: &PyFiniteField,
        degree: usize,
        generators: Vec<PyRef<'_, PyGfMatrix>>,
    ) -> PyResult<Self> {
        MatGroup::new(&field.inner, degree, matrices(&generators))
            .map(|inner| Self { inner })
            .map_err(mg_err)
    }

    /// ``GL(n, q)``, from elementary transvections and a scaling by a primitive
    /// element.
    #[classmethod]
    fn general_linear(_cls: &Bound<'_, PyType>, field: &PyFiniteField, n: usize) -> PyResult<Self> {
        MatGroup::general_linear(&field.inner, n)
            .map(|inner| Self { inner })
            .map_err(mg_err)
    }

    /// ``SL(n, q)``, from elementary transvections.
    #[classmethod]
    fn special_linear(_cls: &Bound<'_, PyType>, field: &PyFiniteField, n: usize) -> PyResult<Self> {
        MatGroup::special_linear(&field.inner, n)
            .map(|inner| Self { inner })
            .map_err(mg_err)
    }

    /// ``Sp(2n, q)``, from symplectic transvections. The argument is ``n``, so
    /// the matrices are ``2n x 2n``.
    #[classmethod]
    fn symplectic(_cls: &Bound<'_, PyType>, field: &PyFiniteField, n: usize) -> PyResult<Self> {
        MatGroup::symplectic(&field.inner, n)
            .map(|inner| Self { inner })
            .map_err(mg_err)
    }

    /// A Singer cycle in ``GL(n, q)``: cyclic of order ``q**n - 1``.
    ///
    /// Available over prime fields only (``E-MATGRP-010`` otherwise).
    #[classmethod]
    fn singer_cycle(_cls: &Bound<'_, PyType>, field: &PyFiniteField, n: usize) -> PyResult<Self> {
        MatGroup::singer_cycle(&field.inner, n)
            .map(|inner| Self { inner })
            .map_err(mg_err)
    }

    /// The trivial subgroup of ``GL(degree, q)``.
    #[classmethod]
    fn trivial(_cls: &Bound<'_, PyType>, field: &PyFiniteField, degree: usize) -> PyResult<Self> {
        MatGroup::trivial(&field.inner, degree)
            .map(|inner| Self { inner })
            .map_err(mg_err)
    }

    /// The size of the matrices.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// The field.
    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    /// The Schreier–Sims work budget in force.
    #[getter]
    fn budget(&self) -> u64 {
        self.inner.budget()
    }

    /// Is every generator the identity? Decided without a stabilizer chain.
    #[getter]
    fn is_trivial(&self) -> bool {
        self.inner.is_trivial()
    }

    /// The generators, as supplied.
    fn generators(&self) -> Vec<PyGfMatrix> {
        wrap_all(self.inner.generators().to_vec())
    }

    /// The identity matrix.
    fn identity(&self) -> PyResult<PyGfMatrix> {
        self.inner.identity().map(PyGfMatrix::wrap).map_err(mg_err)
    }

    /// The same group with a different Schreier–Sims work budget.
    fn with_budget(&self, budget: u64) -> Self {
        Self {
            inner: self.inner.with_budget(budget),
        }
    }

    /// ``|G|``, exactly, from a Schreier–Sims base and strong generating set —
    /// an arbitrary-precision Python ``int``.
    ///
    /// Not a formula: this is computed from the generators.
    fn order(&self, py: Python<'_>) -> PyResult<PyObject> {
        let order = self.inner.order().map_err(mg_err)?;
        big_int(py, &order)
    }

    /// Is ``m`` an element? A singular matrix is ``False``, not an error; a
    /// matrix of the wrong shape or field *is* an error.
    fn contains(&self, m: &PyGfMatrix) -> PyResult<bool> {
        self.inner.contains(&m.inner).map_err(mg_err)
    }

    /// Sift ``m`` through the stabilizer chain.
    fn sift(&self, m: &PyGfMatrix) -> PyResult<PyMatSiftResult> {
        let sift = self.inner.sift(&m.inner).map_err(mg_err)?;
        Ok(PyMatSiftResult {
            residue: sift.residue().clone(),
            level: sift.level(),
            is_member: sift.is_member(),
        })
    }

    /// The base, as standard-basis indices. Always inside ``range(degree)``,
    /// and at most ``degree`` long.
    fn base(&self) -> PyResult<Vec<usize>> {
        self.inner.base().map_err(mg_err)
    }

    /// The base, as ``1 x d`` row vectors.
    fn base_vectors(&self) -> PyResult<Vec<PyGfMatrix>> {
        let chain = self.inner.stabilizer_chain().map_err(mg_err)?;
        chain.base_vectors().map(wrap_all).map_err(mg_err)
    }

    /// The lengths of the basic orbits; their product is ``order()``.
    fn basic_orbit_lengths(&self) -> PyResult<Vec<usize>> {
        let chain = self.inner.stabilizer_chain().map_err(mg_err)?;
        Ok(chain.levels().iter().map(|l| l.orbit().len()).collect())
    }

    /// Every strong generator, deduplicated.
    fn strong_generators(&self) -> PyResult<Vec<PyGfMatrix>> {
        self.inner.strong_generators().map(wrap_all).map_err(mg_err)
    }

    /// Every element, refusing above ``MATGROUP_DEFAULT_ELEMENT_CAP``
    /// (``E-MATGRP-008``). ``order()`` stays exact above the cap.
    #[pyo3(signature = (cap = None))]
    fn elements(&self, cap: Option<u64>) -> PyResult<Vec<PyGfMatrix>> {
        match cap {
            None => self.inner.elements(),
            Some(cap) => self.inner.elements_with_cap(cap),
        }
        .map(wrap_all)
        .map_err(mg_err)
    }

    /// The orbit of the ``1 x d`` row vector ``v`` under ``v -> v @ M``.
    fn vector_orbit(&self, v: &PyGfMatrix) -> PyResult<Vec<PyGfMatrix>> {
        self.inner
            .vector_orbit(&v.inner)
            .map(wrap_all)
            .map_err(mg_err)
    }

    /// The orbits on the non-zero vectors of ``GF(q)**d``.
    fn vector_orbits(&self) -> PyResult<Vec<Vec<PyGfMatrix>>> {
        self.inner
            .vector_orbits()
            .map(|orbits| orbits.into_iter().map(wrap_all).collect())
            .map_err(mg_err)
    }

    /// The orbit of the projective point ``[v]``, as canonical representatives
    /// (first non-zero coordinate scaled to ``1``).
    fn projective_orbit(&self, v: &PyGfMatrix) -> PyResult<Vec<PyGfMatrix>> {
        self.inner
            .projective_orbit(&v.inner)
            .map(wrap_all)
            .map_err(mg_err)
    }

    /// The orbits on the ``(q**d - 1)/(q - 1)`` projective points.
    fn projective_orbits(&self) -> PyResult<Vec<Vec<PyGfMatrix>>> {
        self.inner
            .projective_orbits()
            .map(|orbits| orbits.into_iter().map(wrap_all).collect())
            .map_err(mg_err)
    }

    /// The points of ``PG(d-1, q)``, as canonical representatives.
    fn projective_points(&self) -> PyResult<Vec<PyGfMatrix>> {
        self.inner.projective_points().map(wrap_all).map_err(mg_err)
    }

    /// Every non-zero row vector of ``GF(q)**d``, in the order the permutation
    /// actions number them.
    fn nonzero_vectors(&self) -> PyResult<Vec<PyGfMatrix>> {
        self.inner.nonzero_vectors().map(wrap_all).map_err(mg_err)
    }

    /// The induced action on the non-zero vectors, as a ``PermutationGroup``.
    ///
    /// Faithful, so the permutation group has exactly ``order()`` elements.
    fn permutation_action_on_vectors(&self) -> PyResult<PyPermutationGroup> {
        self.inner
            .permutation_action_on_vectors()
            .map(PyPermutationGroup::wrap)
            .map_err(mg_err)
    }

    /// The induced action on the projective points, as a ``PermutationGroup``
    /// — this is ``PGL``/``PSL``/``PSp``.
    ///
    /// **Not** faithful: its order is ``|G|`` divided by the number of scalars
    /// in ``G``. ``PSL(2, 7)`` is this action of ``SL(2, 7)`` on the 8 points of
    /// ``PG(1, 7)``, of order 168.
    fn permutation_action_on_projective_points(&self) -> PyResult<PyPermutationGroup> {
        self.inner
            .permutation_action_on_projective_points()
            .map(PyPermutationGroup::wrap)
            .map_err(mg_err)
    }

    /// A pseudo-random element by product replacement.
    ///
    /// Always an element of the group by construction; ``seed`` makes it
    /// reproducible. The distribution is the standard heuristic and is not
    /// proved uniform.
    fn random_element(&self, seed: u64) -> PyResult<PyGfMatrix> {
        self.inner
            .random_element(seed)
            .map(PyGfMatrix::wrap)
            .map_err(mg_err)
    }

    /// ``count`` pseudo-random elements from one product-replacement run.
    fn random_elements(&self, seed: u64, count: usize) -> PyResult<Vec<PyGfMatrix>> {
        self.inner
            .random_elements(seed, count)
            .map(wrap_all)
            .map_err(mg_err)
    }

    /// The derived subgroup ``[G, G]``.
    fn derived_subgroup(&self) -> PyResult<Self> {
        self.inner
            .derived_subgroup()
            .map(|inner| Self { inner })
            .map_err(mg_err)
    }

    /// Is ``G' == G``?
    fn is_perfect(&self) -> PyResult<bool> {
        self.inner.is_perfect().map_err(mg_err)
    }

    /// The centre ``Z(G)``.
    fn centre(&self) -> PyResult<Self> {
        self.inner
            .centre()
            .map(|inner| Self { inner })
            .map_err(mg_err)
    }

    /// The elements of the centre.
    fn centre_elements(&self) -> PyResult<Vec<PyGfMatrix>> {
        self.inner.centre_elements().map(wrap_all).map_err(mg_err)
    }

    /// A basis for the algebra of matrices commuting with every generator.
    ///
    /// One element for an absolutely irreducible module, in which case the
    /// centre is the scalars of ``G``.
    fn commutant_basis(&self) -> PyResult<Vec<PyGfMatrix>> {
        self.inner.commutant_basis().map(wrap_all).map_err(mg_err)
    }

    /// The normal closure of ``subset`` in this group.
    fn normal_closure(&self, subset: Vec<PyRef<'_, PyGfMatrix>>) -> PyResult<Self> {
        self.inner
            .normal_closure(&matrices(&subset))
            .map(|inner| Self { inner })
            .map_err(mg_err)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// ``|GL(n, q)|`` from the product formula ``prod(q**n - q**i)``.
///
/// Closed form, with no enumeration and no cap — provided so that a caller can
/// check `MatGroup.order()` against it, which is what this module's Rust tests
/// do.
#[pyfunction]
fn matgroup_gl_order(py: Python<'_>, q: u64, n: usize) -> PyResult<PyObject> {
    big_int(py, &core_gl_order(&rug::Integer::from(q), n))
}

/// ``|SL(n, q)| = |GL(n, q)| / (q - 1)``.
#[pyfunction]
fn matgroup_sl_order(py: Python<'_>, q: u64, n: usize) -> PyResult<PyObject> {
    big_int(py, &core_sl_order(&rug::Integer::from(q), n))
}

/// ``|Sp(2n, q)| = q**(n*n) * prod(q**(2i) - 1)``. The argument is ``n``.
#[pyfunction]
fn matgroup_sp_order(py: Python<'_>, q: u64, n: usize) -> PyResult<PyObject> {
    big_int(py, &core_sp_order(&rug::Integer::from(q), n))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMatGroup>()?;
    m.add_class::<PyMatSiftResult>()?;
    m.add("MatGroupError", m.py().get_type_bound::<PyMatGroupError>())?;
    m.add_function(wrap_pyfunction!(matgroup_gl_order, m)?)?;
    m.add_function(wrap_pyfunction!(matgroup_sl_order, m)?)?;
    m.add_function(wrap_pyfunction!(matgroup_sp_order, m)?)?;
    m.add("MATGROUP_MAX_DEGREE", MAX_MATGROUP_DEGREE)?;
    m.add("MATGROUP_MAX_ORBIT", MAX_MATGROUP_ORBIT)?;
    m.add("MATGROUP_MAX_SCHREIER_WORK", MAX_MATGROUP_SCHREIER_WORK)?;
    m.add("MATGROUP_DEFAULT_ELEMENT_CAP", DEFAULT_MATGROUP_ELEMENT_CAP)?;
    m.add("MATGROUP_MAX_FIELD_ORDER", MAX_MATGROUP_FIELD_ORDER)?;
    m.add(
        "MATGROUP_MAX_COMMUTANT_ELEMENTS",
        MAX_MATGROUP_COMMUTANT_ELEMENTS,
    )?;
    Ok(())
}
