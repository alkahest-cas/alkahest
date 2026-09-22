//! PyO3 bindings for `alkahest_core::coding` — classical linear codes, weight
//! enumerators, Krawtchouk polynomials and the Delsarte LP bound.
//!
//! The Python surface is `alkahest.experimental.LinearCode`,
//! `alkahest.experimental.WeightEnumerator`,
//! `alkahest.experimental.DelsarteBound` and the free functions
//! `krawtchouk`, `krawtchouk_poly`, `delsarte_lp_bound`, `singleton_bound` and
//! `hamming_bound`. Refusals arrive as `alkahest.experimental.CodingError` with
//! a stable `E-CODE-NNN` `.code`.
//!
//! # Exactness across the boundary
//!
//! Weight distributions and code sizes are arbitrary-precision Python `int`s,
//! never floats: `|C|` for a `[128, 64]` code does not fit a machine word and
//! silently returning `float(2**64)` would be a rounding error wearing a count's
//! name. The Delsarte optimum is generally not an integer — `A_2(11, 3)`
//! optimises at `512/3` — so it crosses as a `fractions.Fraction`, and the
//! certified bound `⌊optimum⌋` crosses as an `int`. Nothing in this binding
//! produces a `float`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyType};
use rug::{Integer, Rational};

use alkahest_core::experimental::{
    delsarte_lp_bound as core_delsarte, hamming_bound as core_hamming_bound,
    krawtchouk as core_krawtchouk, krawtchouk_poly as core_krawtchouk_poly,
    singleton_bound as core_singleton_bound, CodingError, DelsarteBound, LinearCode,
    WeightEnumerator,
};

use crate::ffield::{PyFiniteField, PyGfMatrix};

pyo3::create_exception!(alkahest, PyCodingError, crate::PyAlkahestError);

fn coding_err(e: CodingError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyCodingError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// A `rug::Integer` as an exact Python `int`, at any size.
fn big_int(py: Python<'_>, value: &Integer) -> PyResult<PyObject> {
    let int_cls = py.get_type_bound::<PyInt>();
    Ok(int_cls.call1((value.to_string(),))?.into_py(py))
}

/// A `rug::Rational` as an `int` when integral, else a `fractions.Fraction`.
fn big_rat(py: Python<'_>, value: &Rational) -> PyResult<PyObject> {
    if value.is_integer() {
        return big_int(py, value.numer());
    }
    let fractions = PyModule::import_bound(py, "fractions")?;
    let frac = fractions.getattr("Fraction")?;
    Ok(frac.call1((value.to_string(),))?.into_py(py))
}

fn int_list(py: Python<'_>, values: &[Integer]) -> PyResult<Vec<PyObject>> {
    values.iter().map(|v| big_int(py, v)).collect()
}

// ---------------------------------------------------------------------------
// LinearCode
// ---------------------------------------------------------------------------

/// A linear ``[n, k]`` code over GF(q).
///
/// Build one from a generator matrix (rows span the code) or a parity-check
/// matrix (rows span the dual); each is derived from the other, and both are
/// stored in reduced row echelon form, so
/// ``LinearCode.from_generator(c.generator())`` reproduces ``c`` exactly rather
/// than merely spanning the same subspace.
///
/// ``minimum_distance()`` and ``weight_distribution()`` enumerate all ``q**k``
/// codewords. That is capped, and past the cap they raise ``E-CODE-004`` rather
/// than truncating the search: the minimum weight of *some* of the codewords is
/// an upper bound on ``d`` wearing ``d``'s name.
#[pyclass(name = "LinearCode", module = "alkahest")]
#[derive(Clone)]
pub struct PyLinearCode {
    inner: LinearCode,
}

#[pymethods]
impl PyLinearCode {
    /// The code spanned by the rows of ``generator``.
    ///
    /// The rows need not be independent; ``k`` is their rank.
    #[new]
    fn new(generator: &PyGfMatrix) -> PyResult<Self> {
        LinearCode::from_generator(&generator.inner)
            .map(|inner| Self { inner })
            .map_err(coding_err)
    }

    /// The code spanned by the rows of ``generator`` — the same as the
    /// constructor, named for symmetry with :meth:`from_parity_check`.
    #[classmethod]
    fn from_generator(_cls: &Bound<'_, PyType>, generator: &PyGfMatrix) -> PyResult<Self> {
        Self::new(generator)
    }

    /// The code ``{x : H @ x.T == 0}``.
    #[classmethod]
    fn from_parity_check(_cls: &Bound<'_, PyType>, parity: &PyGfMatrix) -> PyResult<Self> {
        LinearCode::from_parity_check(&parity.inner)
            .map(|inner| Self { inner })
            .map_err(coding_err)
    }

    /// The Hamming code over GF(q) with ``r`` parity checks:
    /// ``[(q**r - 1)/(q - 1), n - r, 3]``. Over GF(2) with ``r = 3`` this is
    /// the ``[7, 4, 3]`` code.
    #[classmethod]
    fn hamming(_cls: &Bound<'_, PyType>, field: &PyFiniteField, r: u32) -> PyResult<Self> {
        LinearCode::hamming(&field.inner, r)
            .map(|inner| Self { inner })
            .map_err(coding_err)
    }

    /// The binary Golay code ``[23, 12, 7]``.
    ///
    /// ``LinearCode.golay_binary().extend()`` is the self-dual extended Golay
    /// code ``[24, 12, 8]``.
    #[classmethod]
    fn golay_binary(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        LinearCode::golay_binary()
            .map(|inner| Self { inner })
            .map_err(coding_err)
    }

    /// The repetition code ``[n, 1, n]`` over GF(q).
    #[classmethod]
    fn repetition(_cls: &Bound<'_, PyType>, field: &PyFiniteField, n: usize) -> PyResult<Self> {
        LinearCode::repetition(&field.inner, n)
            .map(|inner| Self { inner })
            .map_err(coding_err)
    }

    /// The length ``n``.
    #[getter]
    fn length(&self) -> usize {
        self.inner.length()
    }

    /// The dimension ``k``.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// The redundancy ``n - k``.
    #[getter]
    fn redundancy(&self) -> usize {
        self.inner.redundancy()
    }

    /// The field GF(q) the code is defined over.
    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    /// ``q**k``, the number of codewords, as an exact ``int``.
    fn size(&self, py: Python<'_>) -> PyResult<PyObject> {
        big_int(py, &self.inner.size())
    }

    /// The generator matrix: ``k x n``, in reduced row echelon form.
    fn generator(&self) -> PyGfMatrix {
        PyGfMatrix::wrap(self.inner.generator().clone())
    }

    /// The parity-check matrix: ``(n-k) x n``, in reduced row echelon form.
    fn parity_check(&self) -> PyGfMatrix {
        PyGfMatrix::wrap(self.inner.parity_check().clone())
    }

    /// The dual code, an ``[n, n-k]`` code. Free: the two matrices swap roles.
    fn dual(&self) -> Self {
        Self {
            inner: self.inner.dual(),
        }
    }

    /// The extended code ``[n+1, k]``: one overall parity symbol appended.
    fn extend(&self) -> PyResult<Self> {
        self.inner
            .extend()
            .map(|inner| Self { inner })
            .map_err(coding_err)
    }

    /// Whether ``C`` is contained in its dual, i.e. ``G @ G.T == 0``.
    fn is_self_orthogonal(&self) -> PyResult<bool> {
        self.inner.is_self_orthogonal().map_err(coding_err)
    }

    /// Whether ``C == C.dual()``: self-orthogonal and ``2k == n``.
    fn is_self_dual(&self) -> PyResult<bool> {
        self.inner.is_self_dual().map_err(coding_err)
    }

    /// Whether the ``1 x n`` row vector ``word`` is a codeword.
    fn contains(&self, word: &PyGfMatrix) -> PyResult<bool> {
        self.inner.contains(&word.inner).map_err(coding_err)
    }

    /// The weight distribution ``[A_0, …, A_n]`` as exact ``int``s.
    ///
    /// Raises ``E-CODE-004`` when ``q**k`` is past the enumeration cap.
    fn weight_distribution(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let a = self.inner.weight_distribution().map_err(coding_err)?;
        int_list(py, &a)
    }

    /// The weight enumerator polynomial.
    fn weight_enumerator(&self) -> PyResult<PyWeightEnumerator> {
        self.inner
            .weight_enumerator()
            .map(|inner| PyWeightEnumerator { inner })
            .map_err(coding_err)
    }

    /// The minimum distance, or ``None`` for the zero code — which has no
    /// non-zero codeword and so no minimum distance. Conventions differ on
    /// whether to call that ``n`` or infinity, and picking one silently would
    /// be a guess.
    fn minimum_distance(&self) -> PyResult<Option<usize>> {
        self.inner.minimum_distance().map_err(coding_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearCode(n={}, k={}, q={})",
            self.inner.length(),
            self.inner.dimension(),
            self.inner
                .field()
                .order()
                .map(|q| q.to_string())
                .unwrap_or_else(|| "?".to_string())
        )
    }
}

// ---------------------------------------------------------------------------
// WeightEnumerator
// ---------------------------------------------------------------------------

/// The weight enumerator ``W(x, y) = sum_i A_i x**(n-i) y**i`` of a code.
///
/// ``WeightEnumerator(q, coeffs)`` builds one from a weight distribution, which
/// is validated: a linear code has ``A_0 == 1`` and no negative multiplicity.
#[pyclass(name = "WeightEnumerator", module = "alkahest")]
#[derive(Clone)]
pub struct PyWeightEnumerator {
    inner: WeightEnumerator,
}

#[pymethods]
impl PyWeightEnumerator {
    #[new]
    fn new(q: u64, coeffs: Vec<i128>) -> PyResult<Self> {
        let coeffs: Vec<Integer> = coeffs.into_iter().map(Integer::from).collect();
        WeightEnumerator::new(q, coeffs)
            .map(|inner| Self { inner })
            .map_err(coding_err)
    }

    /// The length ``n``.
    #[getter]
    fn length(&self) -> usize {
        self.inner.length()
    }

    /// The alphabet size ``q``.
    #[getter]
    fn alphabet_size(&self) -> u64 {
        self.inner.alphabet_size()
    }

    /// The weight distribution ``[A_0, …, A_n]`` as exact ``int``s.
    fn coefficients(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        int_list(py, self.inner.coefficients())
    }

    /// ``|C| = sum_i A_i``.
    fn size(&self, py: Python<'_>) -> PyResult<PyObject> {
        big_int(py, &self.inner.size())
    }

    /// The least ``i >= 1`` with ``A_i > 0``, or ``None`` for the zero code.
    fn minimum_distance(&self) -> Option<usize> {
        self.inner.minimum_distance()
    }

    /// ``W(x, y)`` at integer arguments.
    fn evaluate(&self, py: Python<'_>, x: i128, y: i128) -> PyResult<PyObject> {
        let v = self.inner.evaluate(&Integer::from(x), &Integer::from(y));
        big_int(py, &v)
    }

    /// The MacWilliams transform: the weight enumerator of the dual code.
    ///
    /// Raises ``E-CODE-006`` when the transform does not come out as a
    /// non-negative integer vector after dividing by ``|C|`` — which says the
    /// input was not the weight enumerator of a linear code over GF(q). The
    /// division is exact whenever it was.
    fn macwilliams(&self) -> PyResult<Self> {
        self.inner
            .macwilliams()
            .map(|inner| Self { inner })
            .map_err(coding_err)
    }

    /// Whether every Delsarte constraint ``sum_i A_i K_k(i) >= 0`` holds.
    ///
    /// True for the distance distribution of *any* code over an alphabet of
    /// size ``q``, linear or not.
    fn is_dual_feasible(&self) -> bool {
        self.inner.is_dual_feasible()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("WeightEnumerator({})", self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

// ---------------------------------------------------------------------------
// DelsarteBound
// ---------------------------------------------------------------------------

/// A certified Delsarte upper bound on ``A_q(n, d)``.
///
/// ``bound`` is an upper bound on the size of **every** code of length ``n``
/// over an alphabet of size ``q`` with minimum distance at least ``d`` — linear
/// or not. It is certified, not estimated: it is read off the dual linear
/// programme, and ``certificate()`` is the vector of non-negative multipliers
/// that proves it. ``verify_certificate()`` re-checks them from scratch in
/// exact rational arithmetic; :func:`delsarte_lp_bound` already refuses rather
/// than return a bound whose certificate did not pass.
#[pyclass(name = "DelsarteBound", module = "alkahest")]
#[derive(Clone)]
pub struct PyDelsarteBound {
    inner: DelsarteBound,
}

#[pymethods]
impl PyDelsarteBound {
    /// The length ``n``.
    #[getter]
    fn length(&self) -> usize {
        self.inner.length()
    }

    /// The minimum distance ``d``.
    #[getter]
    fn distance(&self) -> usize {
        self.inner.distance()
    }

    /// The alphabet size ``q``.
    #[getter]
    fn alphabet_size(&self) -> u64 {
        self.inner.alphabet_size()
    }

    /// ``floor(optimum)``: the certified upper bound on ``A_q(n, d)``, an
    /// exact ``int``. A code has an integer number of words, so flooring is
    /// sound.
    #[getter]
    fn bound(&self, py: Python<'_>) -> PyResult<PyObject> {
        big_int(py, self.inner.bound())
    }

    /// The exact LP optimum: an ``int`` when integral, else a
    /// ``fractions.Fraction``. Never a ``float``.
    fn optimum(&self, py: Python<'_>) -> PyResult<PyObject> {
        big_rat(py, self.inner.optimum())
    }

    /// The optimal distance distribution ``[A_0, …, A_n]`` the programme found.
    ///
    /// A vertex of the relaxation, **not** the distance distribution of any
    /// code: entries are generally fractional and no code need attain them.
    fn distribution(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .distribution()
            .iter()
            .map(|v| big_rat(py, v))
            .collect()
    }

    /// The dual certificate ``y_1 … y_n >= 0`` with
    /// ``sum_k y_k K_k(i) <= -1`` for ``d <= i <= n``, indexed from ``k = 1``.
    ///
    /// With that and a Krawtchouk evaluator, the bound can be re-derived
    /// without trusting this library at all.
    fn certificate(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .certificate()
            .iter()
            .map(|v| big_rat(py, v))
            .collect()
    }

    /// Re-check the dual certificate from scratch, exactly.
    fn verify_certificate(&self) -> bool {
        self.inner.verify_certificate()
    }

    /// Re-check that the reported distribution is feasible for the primal.
    fn verify_distribution(&self) -> bool {
        self.inner.verify_distribution()
    }

    fn __repr__(&self) -> String {
        format!(
            "DelsarteBound(n={}, d={}, q={}, bound={})",
            self.inner.length(),
            self.inner.distance(),
            self.inner.alphabet_size(),
            self.inner.bound()
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// The Krawtchouk polynomial ``K_k(x; n, q)`` at an integer ``x``, exactly.
///
/// These are the eigenvalues of the Hamming association scheme; ``x`` may lie
/// outside ``0..n``.
#[pyfunction]
#[pyo3(name = "krawtchouk")]
fn py_krawtchouk(py: Python<'_>, k: usize, x: i64, n: usize, q: u64) -> PyResult<PyObject> {
    if q < 2 {
        return Err(PyValueError::new_err(
            "an alphabet needs at least two symbols",
        ));
    }
    big_int(py, &core_krawtchouk(k, x, n, q))
}

/// ``K_k(., n, q)`` as a polynomial in ``x``: coefficients ascending, each an
/// ``int`` or a ``fractions.Fraction``.
#[pyfunction]
#[pyo3(name = "krawtchouk_poly")]
fn py_krawtchouk_poly(py: Python<'_>, k: usize, n: usize, q: u64) -> PyResult<Vec<PyObject>> {
    if q < 2 {
        return Err(PyValueError::new_err(
            "an alphabet needs at least two symbols",
        ));
    }
    core_krawtchouk_poly(k, n, q)
        .iter()
        .map(|c| big_rat(py, c))
        .collect()
}

/// The Delsarte linear-programming upper bound on ``A_q(n, d)``.
///
/// Solved in exact rational arithmetic and returned with the dual certificate
/// that proves it; see :class:`DelsarteBound`.
///
/// >>> from alkahest.experimental import delsarte_lp_bound
/// >>> delsarte_lp_bound(24, 8, 2).bound      # the extended Golay code
/// 4096
#[pyfunction]
#[pyo3(name = "delsarte_lp_bound")]
fn py_delsarte_lp_bound(n: usize, d: usize, q: u64) -> PyResult<PyDelsarteBound> {
    core_delsarte(n, d, q)
        .map(|inner| PyDelsarteBound { inner })
        .map_err(coding_err)
}

/// The Singleton bound ``A_q(n, d) <= q**(n - d + 1)``.
#[pyfunction]
#[pyo3(name = "singleton_bound")]
fn py_singleton_bound(py: Python<'_>, n: usize, d: usize, q: u64) -> PyResult<PyObject> {
    let v = core_singleton_bound(n, d, q).map_err(coding_err)?;
    big_int(py, &v)
}

/// The Hamming (sphere-packing) bound, floored.
#[pyfunction]
#[pyo3(name = "hamming_bound")]
fn py_hamming_bound(py: Python<'_>, n: usize, d: usize, q: u64) -> PyResult<PyObject> {
    let v = core_hamming_bound(n, d, q).map_err(coding_err)?;
    big_int(py, &v)
}

/// Register the coding-theory surface on the `alkahest` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinearCode>()?;
    m.add_class::<PyWeightEnumerator>()?;
    m.add_class::<PyDelsarteBound>()?;
    m.add("CodingError", m.py().get_type_bound::<PyCodingError>())?;
    m.add_function(wrap_pyfunction!(py_krawtchouk, m)?)?;
    m.add_function(wrap_pyfunction!(py_krawtchouk_poly, m)?)?;
    m.add_function(wrap_pyfunction!(py_delsarte_lp_bound, m)?)?;
    m.add_function(wrap_pyfunction!(py_singleton_bound, m)?)?;
    m.add_function(wrap_pyfunction!(py_hamming_bound, m)?)?;
    Ok(())
}
