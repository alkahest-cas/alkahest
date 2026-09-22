//! PyO3 bindings for `alkahest_core::stabilizer` — the binary symplectic /
//! stabilizer-code layer.
//!
//! The Python surface is `alkahest.experimental`: `PauliOperator`,
//! `StabilizerGroup`, `StabilizerCode`, `CssCode`, `MatrixGroup`, `Distance`
//! and the four free functions on the symplectic form. Refusals arrive as
//! `alkahest.experimental.StabilizerError` with a stable `E-STAB-NNN` `.code`
//! — except for those that came out of the GF(q) layer, which keep their own
//! `E-GFQ-NNN`.
//!
//! # Two conventions that cross the boundary unchanged
//!
//! * Symplectic vectors are `list[int]` of length `2n` in **`(x | z)`
//!   layout** — the first `n` entries are the `X` exponents.
//! * Pauli strings use the **Hermitian** letters `I`, `X`, `Y`, `Z` with an
//!   optional `+`, `-`, `i`, `+i`, `-i` prefix, so `"Y"` is the Hermitian `Y`
//!   and not the raw symplectic product `XZ`. The stored phase exponent
//!   (`.phase`) differs from the printed sign by one factor of `i` per `Y`;
//!   see `alkahest_cas::stabilizer::pauli` for the arithmetic.
//!
//! # Distance is not an int
//!
//! `minimum_distance()` and `distance_upper_bound()` both return a
//! [`PyDistance`], which carries `.value` **and** `.exact`. They deliberately
//! do not return a bare `int`: the entire point of the upper-bound path is
//! that it must not be stored in a field the reader takes for a distance.

use pyo3::prelude::*;
use pyo3::types::{PyInt, PyType};

use alkahest_core::experimental::{
    is_symplectic as core_is_symplectic, symplectic_complement as core_complement,
    symplectic_form as core_form, symplectic_gram_matrix as core_gram,
    symplectic_gram_schmidt as core_gram_schmidt, CssCode, Distance, MatrixGroup, MatrixGroupKind,
    PauliOperator, StabilizerCode, StabilizerError, StabilizerGroup, MAX_DISTANCE_SEARCH_DIM,
    MAX_MATRIX_ENUMERATION, MAX_QUBITS,
};

use crate::ffield::{PyFiniteField, PyGfMatrix};
use crate::group::PyPermutationGroup;

pyo3::create_exception!(alkahest, PyStabilizerError, crate::PyAlkahestError);

fn stab_err(e: StabilizerError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyStabilizerError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// A `rug::Integer` as an exact Python `int`, at any size.
fn big_int(py: Python<'_>, value: &rug::Integer) -> PyResult<PyObject> {
    let int_cls = py.get_type_bound::<PyInt>();
    Ok(int_cls.call1((value.to_string(),))?.into_py(py))
}

fn bits_from_py(v: &[i64]) -> PyResult<Vec<u8>> {
    v.iter()
        .map(|&b| match b {
            0 => Ok(0u8),
            1 => Ok(1u8),
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "symplectic vectors are over GF(2): every entry must be 0 or 1, got {other}"
            ))),
        })
        .collect()
}

fn bits_to_py(v: &[u8]) -> Vec<u8> {
    v.to_vec()
}

// ---------------------------------------------------------------------------
// Distance
// ---------------------------------------------------------------------------

/// A minimum distance, or a bound on one.
///
/// ``exact`` is ``True`` only when the whole centralizer was searched. An
/// upper bound says a logical operator of that weight *exists*; it says
/// nothing at all about whether a lighter one does.
#[pyclass(name = "Distance", module = "alkahest")]
#[derive(Clone)]
pub struct PyDistance {
    inner: Distance,
}

#[pymethods]
impl PyDistance {
    /// The number, whatever its status.
    #[getter]
    fn value(&self) -> usize {
        self.inner.value()
    }

    /// ``True`` when the value was verified exhaustively.
    #[getter]
    fn exact(&self) -> bool {
        self.inner.is_exact()
    }

    fn __repr__(&self) -> String {
        match self.inner {
            Distance::Exact(d) => format!("Distance(exact={d})"),
            Distance::UpperBound(d) => format!("Distance(at_most={d})"),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __eq__(&self, other: &PyDistance) -> bool {
        self.inner == other.inner
    }
}

// ---------------------------------------------------------------------------
// PauliOperator
// ---------------------------------------------------------------------------

/// An element of the ``n``-qubit Pauli group, as ``(x | z)`` bits and a phase
/// in ``Z4``.
///
/// ``PauliOperator("XZZXI")`` parses the Hermitian letters with an optional
/// sign prefix (``+``, ``-``, ``i``, ``+i``, ``-i``). ``str(p)`` prints the
/// same form back.
///
/// ``Y`` is the Hermitian ``Y = i·XZ``. The stored ``phase`` is the exponent
/// ``e`` in ``P = i**e · X**x · Z**z``, which for ``"Y"`` is ``1`` — the
/// printed sign and the stored phase differ by one factor of ``i`` per ``Y``
/// position.
#[pyclass(name = "PauliOperator", module = "alkahest")]
#[derive(Clone)]
pub struct PyPauliOperator {
    inner: PauliOperator,
}

impl PyPauliOperator {
    fn wrap(inner: PauliOperator) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPauliOperator {
    #[new]
    fn new(s: &str) -> PyResult<Self> {
        s.parse::<PauliOperator>().map(Self::wrap).map_err(stab_err)
    }

    /// The identity on ``n`` qubits.
    #[classmethod]
    fn identity(_cls: &Bound<'_, PyType>, n: usize) -> PyResult<Self> {
        PauliOperator::identity(n).map(Self::wrap).map_err(stab_err)
    }

    /// ``i**phase · X**x · Z**z``, taken literally — with ``x[j] == z[j] == 1``
    /// and ``phase == 0`` this is ``XZ = -iY``, not ``Y``.
    #[classmethod]
    #[pyo3(signature = (x, z, phase = 0))]
    fn from_xz(_cls: &Bound<'_, PyType>, x: Vec<i64>, z: Vec<i64>, phase: u8) -> PyResult<Self> {
        PauliOperator::from_xz(&bits_from_py(&x)?, &bits_from_py(&z)?, phase)
            .map(Self::wrap)
            .map_err(stab_err)
    }

    /// From a ``(x | z)`` vector of length ``2n``.
    #[classmethod]
    #[pyo3(signature = (bits, phase = 0))]
    fn from_symplectic(_cls: &Bound<'_, PyType>, bits: Vec<i64>, phase: u8) -> PyResult<Self> {
        PauliOperator::from_symplectic(&bits_from_py(&bits)?, phase)
            .map(Self::wrap)
            .map_err(stab_err)
    }

    /// The **Hermitian** Pauli of the given symplectic type, with a ``+`` or
    /// ``-`` sign. This is the constructor that picks the phase for you.
    #[classmethod]
    #[pyo3(signature = (x, z, negative = false))]
    fn hermitian(
        _cls: &Bound<'_, PyType>,
        x: Vec<i64>,
        z: Vec<i64>,
        negative: bool,
    ) -> PyResult<Self> {
        PauliOperator::hermitian(&bits_from_py(&x)?, &bits_from_py(&z)?, negative)
            .map(Self::wrap)
            .map_err(stab_err)
    }

    /// Number of qubits.
    #[getter]
    fn qubits(&self) -> usize {
        self.inner.qubits()
    }

    /// The ``X`` exponents.
    #[getter]
    fn x_bits(&self) -> Vec<u8> {
        bits_to_py(self.inner.x_bits())
    }

    /// The ``Z`` exponents.
    #[getter]
    fn z_bits(&self) -> Vec<u8> {
        bits_to_py(self.inner.z_bits())
    }

    /// The phase exponent ``e`` in ``P = i**e · X**x · Z**z``.
    #[getter]
    fn phase(&self) -> u8 {
        self.inner.phase()
    }

    /// The ``(x | z)`` vector, length ``2n``.
    fn symplectic(&self) -> Vec<u8> {
        self.inner.symplectic()
    }

    /// The number of qubits acted on non-trivially. The phase is irrelevant.
    fn weight(&self) -> usize {
        self.inner.weight()
    }

    /// ``0`` if the two operators commute, ``1`` if they anticommute.
    fn symplectic_product(&self, other: &PyPauliOperator) -> PyResult<u8> {
        self.inner
            .symplectic_product(&other.inner)
            .map_err(stab_err)
    }

    /// Do the two operators commute? Two Paulis always either commute or
    /// anticommute; there is no third case.
    fn commutes_with(&self, other: &PyPauliOperator) -> PyResult<bool> {
        self.inner.commutes_with(&other.inner).map_err(stab_err)
    }

    /// Is ``P == P.conj().T``? A stabilizer generator must be.
    fn is_hermitian(&self) -> bool {
        self.inner.is_hermitian()
    }

    /// Is this the identity, phase included?
    fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    /// The inverse ``P**-1``.
    fn inverse(&self) -> Self {
        Self::wrap(self.inner.inverse())
    }

    /// ``-P``.
    fn negate(&self) -> Self {
        Self::wrap(self.inner.negate())
    }

    fn __mul__(&self, other: &PyPauliOperator) -> PyResult<Self> {
        self.inner
            .mul(&other.inner)
            .map(Self::wrap)
            .map_err(stab_err)
    }

    fn __len__(&self) -> usize {
        self.inner.qubits()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("PauliOperator({:?})", self.inner.to_string())
    }

    fn __eq__(&self, other: &PyPauliOperator) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
    }
}

// ---------------------------------------------------------------------------
// StabilizerGroup
// ---------------------------------------------------------------------------

/// An abelian subgroup of the Pauli group, given by generators.
///
/// Construction checks that the generators pairwise **commute**
/// (``E-STAB-004``), are **Hermitian** (``E-STAB-010``), and that no product
/// of them is ``-I`` (``E-STAB-006``). None of the three is silently repaired.
///
/// ``qubits`` may be given explicitly; it is required when the generator list
/// is empty, because the trivial group on ``n`` qubits is a real object and
/// ``n`` is not recoverable from an empty list.
#[pyclass(name = "StabilizerGroup", module = "alkahest")]
#[derive(Clone)]
pub struct PyStabilizerGroup {
    inner: StabilizerGroup,
}

#[pymethods]
impl PyStabilizerGroup {
    #[new]
    #[pyo3(signature = (generators, qubits = None))]
    fn new(generators: Vec<PyPauliOperator>, qubits: Option<usize>) -> PyResult<Self> {
        let gens: Vec<PauliOperator> = generators.into_iter().map(|g| g.inner).collect();
        let inner = match qubits {
            Some(n) => StabilizerGroup::with_qubits(n, gens),
            None => StabilizerGroup::new(gens),
        }
        .map_err(stab_err)?;
        Ok(Self { inner })
    }

    /// Number of qubits.
    #[getter]
    fn qubits(&self) -> usize {
        self.inner.qubits()
    }

    /// ``rank(S)`` — the number of independent stabilizer conditions.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// The generators, as supplied.
    #[getter]
    fn generators(&self) -> Vec<PyPauliOperator> {
        self.inner
            .generators()
            .iter()
            .cloned()
            .map(PyPauliOperator::wrap)
            .collect()
    }

    /// Is ``rank == len(generators)``?
    fn is_independent(&self) -> bool {
        self.inner.is_independent()
    }

    /// The ``m x 2n`` check matrix over GF(2), one ``(x | z)`` row per
    /// generator.
    fn check_matrix(&self) -> PyResult<PyGfMatrix> {
        self.inner
            .check_matrix()
            .map(PyGfMatrix::wrap)
            .map_err(stab_err)
    }

    /// One bit per generator: ``1`` where ``error`` anticommutes with it.
    fn syndrome(&self, error: &PyPauliOperator) -> PyResult<Vec<u8>> {
        self.inner.syndrome(&error.inner).map_err(stab_err)
    }

    /// Does ``p`` commute with every generator — is it in ``N(S)``?
    fn centralizes(&self, p: &PyPauliOperator) -> PyResult<bool> {
        self.inner.centralizes(&p.inner).map_err(stab_err)
    }

    /// Is ``p`` in ``S``, **sign included**? ``-g`` for a generator ``g`` is
    /// not in ``S`` and returns ``False``.
    fn contains(&self, p: &PyPauliOperator) -> PyResult<bool> {
        self.inner.contains(&p.inner).map_err(stab_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "StabilizerGroup(qubits={}, generators={}, rank={})",
            self.inner.qubits(),
            self.inner.generators().len(),
            self.inner.rank()
        )
    }
}

// ---------------------------------------------------------------------------
// StabilizerCode
// ---------------------------------------------------------------------------

/// An ``[[n, k]]`` stabilizer code.
#[pyclass(name = "StabilizerCode", module = "alkahest")]
#[derive(Clone)]
pub struct PyStabilizerCode {
    inner: StabilizerCode,
}

impl PyStabilizerCode {
    fn wrap(inner: StabilizerCode) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyStabilizerCode {
    #[new]
    #[pyo3(signature = (generators, qubits = None))]
    fn new(generators: Vec<PyPauliOperator>, qubits: Option<usize>) -> PyResult<Self> {
        let gens: Vec<PauliOperator> = generators.into_iter().map(|g| g.inner).collect();
        let inner = match qubits {
            Some(n) => StabilizerCode::from_generators_on(n, gens),
            None => StabilizerCode::from_generators(gens),
        }
        .map_err(stab_err)?;
        Ok(Self { inner })
    }

    /// Block length.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Number of logical qubits, ``n - rank(S)``.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// The stabilizer group.
    #[getter]
    fn stabilizer(&self) -> PyStabilizerGroup {
        PyStabilizerGroup {
            inner: self.inner.stabilizer().clone(),
        }
    }

    /// The ``k`` logical ``X`` operators.
    #[getter]
    fn logical_x(&self) -> Vec<PyPauliOperator> {
        self.inner
            .logical_x()
            .iter()
            .cloned()
            .map(PyPauliOperator::wrap)
            .collect()
    }

    /// The ``k`` logical ``Z`` operators.
    #[getter]
    fn logical_z(&self) -> Vec<PyPauliOperator> {
        self.inner
            .logical_z()
            .iter()
            .cloned()
            .map(PyPauliOperator::wrap)
            .collect()
    }

    /// One bit per stabilizer generator.
    fn syndrome(&self, error: &PyPauliOperator) -> PyResult<Vec<u8>> {
        self.inner.syndrome(&error.inner).map_err(stab_err)
    }

    /// Is ``error`` in ``N(S) \\ S`` — an error the syndrome cannot see?
    fn is_logical_error(&self, error: &PyPauliOperator) -> PyResult<bool> {
        self.inner.is_logical_error(&error.inner).map_err(stab_err)
    }

    /// The **exact** minimum distance, by exhaustive search over ``N(S)``.
    ///
    /// Raises ``E-STAB-007`` when ``k == 0`` and ``E-STAB-008`` when ``n + k``
    /// exceeds ``cap`` (default ``STABILIZER_MAX_DISTANCE_SEARCH_DIM``).
    #[pyo3(signature = (cap = None))]
    fn minimum_distance(&self, cap: Option<usize>) -> PyResult<PyDistance> {
        let d = match cap {
            Some(c) => self.inner.minimum_distance_with_cap(c),
            None => self.inner.minimum_distance(),
        }
        .map_err(stab_err)?;
        Ok(PyDistance { inner: d })
    }

    /// An upper bound on the distance, available at any size.
    fn distance_upper_bound(&self) -> PyResult<PyDistance> {
        self.inner
            .distance_upper_bound()
            .map(|d| PyDistance { inner: d })
            .map_err(stab_err)
    }

    fn __repr__(&self) -> String {
        format!("StabilizerCode([[{}, {}]])", self.inner.n(), self.inner.k())
    }
}

// ---------------------------------------------------------------------------
// CssCode
// ---------------------------------------------------------------------------

/// A CSS code from a pair of GF(2) check matrices.
///
/// ``CssCode(hx, hz)`` refuses with ``E-STAB-005`` unless
/// ``hx @ hz.T == 0`` — the CSS condition, checked rather than assumed. The
/// error names the row of ``H_X`` and the row of ``H_Z`` whose stabilizers
/// anticommute.
#[pyclass(name = "CssCode", module = "alkahest")]
#[derive(Clone)]
pub struct PyCssCode {
    inner: CssCode,
}

#[pymethods]
impl PyCssCode {
    #[new]
    fn new(hx: &PyGfMatrix, hz: &PyGfMatrix) -> PyResult<Self> {
        CssCode::new(&hx.inner, &hz.inner)
            .map(|inner| Self { inner })
            .map_err(stab_err)
    }

    /// Block length.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// ``n - rank(H_X) - rank(H_Z)``.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// ``rank(H_X)``.
    #[getter]
    fn x_rank(&self) -> usize {
        self.inner.x_rank()
    }

    /// ``rank(H_Z)``.
    #[getter]
    fn z_rank(&self) -> usize {
        self.inner.z_rank()
    }

    /// The ``X``-type check matrix, as supplied.
    #[getter]
    fn hx(&self) -> PyGfMatrix {
        PyGfMatrix::wrap(self.inner.hx().clone())
    }

    /// The ``Z``-type check matrix, as supplied.
    #[getter]
    fn hz(&self) -> PyGfMatrix {
        PyGfMatrix::wrap(self.inner.hz().clone())
    }

    /// The ``X``-type logical operators, as ``n``-bit supports.
    #[getter]
    fn logical_x_bits(&self) -> Vec<Vec<u8>> {
        self.inner.logical_x_bits().to_vec()
    }

    /// The ``Z``-type logical operators, as ``n``-bit supports.
    #[getter]
    fn logical_z_bits(&self) -> Vec<Vec<u8>> {
        self.inner.logical_z_bits().to_vec()
    }

    /// The ``X``-type logical operators as Pauli operators.
    #[getter]
    fn logical_x(&self) -> PyResult<Vec<PyPauliOperator>> {
        Ok(self
            .inner
            .logical_x()
            .map_err(stab_err)?
            .into_iter()
            .map(PyPauliOperator::wrap)
            .collect())
    }

    /// The ``Z``-type logical operators as Pauli operators.
    #[getter]
    fn logical_z(&self) -> PyResult<Vec<PyPauliOperator>> {
        Ok(self
            .inner
            .logical_z()
            .map_err(stab_err)?
            .into_iter()
            .map(PyPauliOperator::wrap)
            .collect())
    }

    /// The stabilizer generators, ``X``-type first.
    fn stabilizer_generators(&self) -> PyResult<Vec<PyPauliOperator>> {
        Ok(self
            .inner
            .stabilizer_generators()
            .map_err(stab_err)?
            .into_iter()
            .map(PyPauliOperator::wrap)
            .collect())
    }

    /// The same code through the general :class:`StabilizerCode` machinery.
    fn to_stabilizer_code(&self) -> PyResult<PyStabilizerCode> {
        self.inner
            .to_stabilizer_code()
            .map(PyStabilizerCode::wrap)
            .map_err(stab_err)
    }

    /// The **exact** minimum distance, as ``min(d_X, d_Z)``.
    #[pyo3(signature = (cap = None))]
    fn minimum_distance(&self, cap: Option<usize>) -> PyResult<PyDistance> {
        let d = match cap {
            Some(c) => self.inner.minimum_distance_with_cap(c),
            None => self.inner.minimum_distance(),
        }
        .map_err(stab_err)?;
        Ok(PyDistance { inner: d })
    }

    /// An upper bound on the distance: the lightest logical generator.
    fn distance_upper_bound(&self) -> PyResult<PyDistance> {
        self.inner
            .distance_upper_bound()
            .map(|d| PyDistance { inner: d })
            .map_err(stab_err)
    }

    fn __repr__(&self) -> String {
        format!("CssCode([[{}, {}]])", self.inner.n(), self.inner.k())
    }
}

// ---------------------------------------------------------------------------
// MatrixGroup
// ---------------------------------------------------------------------------

/// A classical matrix group over GF(q): ``GL(n, q)``, ``SL(n, q)`` or
/// ``Sp(2n, q)``.
///
/// ``order`` is closed form and has no size limit. ``elements()`` brute-forces
/// over all ``q**(d*d)`` matrices and is capped at
/// ``STABILIZER_MAX_MATRIX_ENUMERATION`` candidates; the order is still exact
/// above the cap, because it never came from the list.
#[pyclass(name = "MatrixGroup", module = "alkahest")]
#[derive(Clone)]
pub struct PyMatrixGroup {
    inner: MatrixGroup,
}

#[pymethods]
impl PyMatrixGroup {
    /// ``GL(n, q)``.
    #[classmethod]
    fn general_linear(_cls: &Bound<'_, PyType>, field: &PyFiniteField, n: usize) -> PyResult<Self> {
        MatrixGroup::general_linear(&field.inner, n)
            .map(|inner| Self { inner })
            .map_err(stab_err)
    }

    /// ``SL(n, q)``.
    #[classmethod]
    fn special_linear(_cls: &Bound<'_, PyType>, field: &PyFiniteField, n: usize) -> PyResult<Self> {
        MatrixGroup::special_linear(&field.inner, n)
            .map(|inner| Self { inner })
            .map_err(stab_err)
    }

    /// ``Sp(2n, q)`` — the argument is ``n``, so the matrices are ``2n x 2n``.
    #[classmethod]
    fn symplectic(_cls: &Bound<'_, PyType>, field: &PyFiniteField, n: usize) -> PyResult<Self> {
        MatrixGroup::symplectic(&field.inner, n)
            .map(|inner| Self { inner })
            .map_err(stab_err)
    }

    /// ``"GL"``, ``"SL"`` or ``"Sp"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind() {
            MatrixGroupKind::GeneralLinear => "GL",
            MatrixGroupKind::SpecialLinear => "SL",
            MatrixGroupKind::Symplectic => "Sp",
        }
    }

    /// The size of the matrices — ``n`` for ``GL``/``SL``, ``2n`` for ``Sp``.
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

    /// The exact group order, from the product formula — an arbitrary-precision
    /// Python ``int``.
    #[getter]
    fn order(&self, py: Python<'_>) -> PyResult<PyObject> {
        big_int(py, &self.inner.order())
    }

    /// Is ``m`` an element?
    fn contains(&self, m: &PyGfMatrix) -> PyResult<bool> {
        self.inner.contains(&m.inner).map_err(stab_err)
    }

    /// Every element, by brute force. Raises ``E-STAB-011`` above the cap.
    #[pyo3(signature = (cap = None))]
    fn elements(&self, cap: Option<u64>) -> PyResult<Vec<PyGfMatrix>> {
        let els = match cap {
            Some(c) => self.inner.elements_with_cap(c),
            None => self.inner.elements(),
        }
        .map_err(stab_err)?;
        Ok(els.into_iter().map(PyGfMatrix::wrap).collect())
    }

    /// The induced action on the ``q**d - 1`` non-zero vectors of ``F_q**d``,
    /// as a :class:`PermutationGroup`.
    ///
    /// The action is on **row** vectors (``v -> v @ M``) so that it composes
    /// the same way round as this library's left-to-right permutation product.
    /// It is faithful, so the returned group has exactly ``order`` elements.
    fn permutation_action(&self) -> PyResult<PyPermutationGroup> {
        self.inner
            .permutation_action()
            .map(PyPermutationGroup::wrap)
            .map_err(stab_err)
    }

    fn __repr__(&self) -> String {
        format!("MatrixGroup({})", self.inner)
    }
}

// ---------------------------------------------------------------------------
// Free functions on the symplectic form
// ---------------------------------------------------------------------------

/// The symplectic form ``<u, v> = u_x . v_z + u_z . v_x`` over GF(2).
///
/// Both arguments are ``(x | z)`` vectors of the same even length. ``0`` means
/// the two Paulis commute, ``1`` that they anticommute.
#[pyfunction]
fn symplectic_form(u: Vec<i64>, v: Vec<i64>) -> PyResult<u8> {
    core_form(&bits_from_py(&u)?, &bits_from_py(&v)?).map_err(stab_err)
}

/// The Gram matrix ``Omega = [[0, I], [I, 0]]`` over GF(2), ``2n x 2n``.
#[pyfunction]
fn symplectic_gram_matrix(n: usize) -> PyResult<PyGfMatrix> {
    core_gram(n).map(PyGfMatrix::wrap).map_err(stab_err)
}

/// Is ``m`` in ``Sp(2n, q)``? Tests ``m.T @ Omega @ m == Omega``.
#[pyfunction]
fn is_symplectic(m: &PyGfMatrix) -> PyResult<bool> {
    core_is_symplectic(&m.inner).map_err(stab_err)
}

/// A basis of the symplectic complement of the span of ``vectors``.
#[pyfunction]
fn symplectic_complement(vectors: Vec<Vec<i64>>, n: usize) -> PyResult<Vec<Vec<u8>>> {
    let vs: Vec<Vec<u8>> = vectors
        .iter()
        .map(|v| bits_from_py(v))
        .collect::<PyResult<_>>()?;
    core_complement(&vs, n).map_err(stab_err)
}

/// Symplectic Gram–Schmidt: ``(pairs, radical)``.
///
/// ``pairs`` is a list of ``(u, v)`` with ``<u, v> == 1`` and every other
/// pairing zero; ``radical`` spans the part of the subspace on which the form
/// vanishes identically.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn symplectic_gram_schmidt(
    vectors: Vec<Vec<i64>>,
    n: usize,
) -> PyResult<(Vec<(Vec<u8>, Vec<u8>)>, Vec<Vec<u8>>)> {
    let vs: Vec<Vec<u8>> = vectors
        .iter()
        .map(|v| bits_from_py(v))
        .collect::<PyResult<_>>()?;
    let hb = core_gram_schmidt(&vs, n).map_err(stab_err)?;
    Ok((hb.pairs().to_vec(), hb.radical().to_vec()))
}

/// Register the stabilizer surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistance>()?;
    m.add_class::<PyPauliOperator>()?;
    m.add_class::<PyStabilizerGroup>()?;
    m.add_class::<PyStabilizerCode>()?;
    m.add_class::<PyCssCode>()?;
    m.add_class::<PyMatrixGroup>()?;
    m.add(
        "StabilizerError",
        m.py().get_type_bound::<PyStabilizerError>(),
    )?;
    m.add_function(wrap_pyfunction!(symplectic_form, m)?)?;
    m.add_function(wrap_pyfunction!(symplectic_gram_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(is_symplectic, m)?)?;
    m.add_function(wrap_pyfunction!(symplectic_complement, m)?)?;
    m.add_function(wrap_pyfunction!(symplectic_gram_schmidt, m)?)?;
    m.add("STABILIZER_MAX_QUBITS", MAX_QUBITS)?;
    m.add(
        "STABILIZER_MAX_DISTANCE_SEARCH_DIM",
        MAX_DISTANCE_SEARCH_DIM,
    )?;
    m.add("STABILIZER_MAX_MATRIX_ENUMERATION", MAX_MATRIX_ENUMERATION)?;
    Ok(())
}
