//! PyO3 bindings for `alkahest_core::ffield` — dense linear algebra over GF(q).
//!
//! The Python surface is `alkahest.experimental.FiniteField` /
//! `alkahest.experimental.GfMatrix`, and refusals arrive as
//! `alkahest.experimental.FiniteFieldError` with a stable `E-GFQ-NNN` `.code`.
//!
//! # How an element crosses the boundary
//!
//! Over a **prime** field GF(p) an element is a plain `int`. Over an
//! **extension** GF(p^k) it is a `list[int]` of exactly `k` coordinates in the
//! polynomial basis `1, a, a², …` — ascending, so `[1, 1, 0]` is `a + 1` in
//! GF(2³). Inputs accept either form (an `int` is read as an element of the
//! prime subfield); outputs always use the form that matches the field, so a
//! caller never has to branch on what came back.

use pyo3::prelude::*;
use pyo3::types::{PyInt, PyList, PyType};

use alkahest_core::experimental::{FieldElement, FiniteField, FiniteFieldError, GfMatrix};

pyo3::create_exception!(alkahest, PyFiniteFieldError, crate::PyAlkahestError);

fn ff_err(e: FiniteFieldError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyFiniteFieldError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// Read one field element from Python: an `int`, or a sequence of `int`.
fn elem_from_py(field: &FiniteField, obj: &Bound<'_, PyAny>) -> PyResult<FieldElement> {
    let p = i128::from(field.characteristic());
    if let Ok(v) = obj.extract::<i128>() {
        return field.element(&[v.rem_euclid(p) as u64]).map_err(ff_err);
    }
    if obj.is_instance_of::<PyInt>() {
        // An int that did not fit `i128`. Saying so beats "must be an int or a
        // sequence of ints", which is what the branch below would have said.
        return Err(PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
            "this integer is too large to reduce into GF(q) here; reduce it \
             modulo the characteristic before passing it",
        ));
    }
    let coeffs: Vec<i128> = obj.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "a GF(q) element must be an int, or a sequence of ints giving its \
             coordinates over the prime subfield (ascending)",
        )
    })?;
    let reduced: Vec<u64> = coeffs.iter().map(|v| v.rem_euclid(p) as u64).collect();
    field.element(&reduced).map_err(ff_err)
}

/// Write one field element back to Python — `int` over GF(p), `list[int]`
/// of exactly `k` coordinates over GF(p^k).
fn elem_to_py(py: Python<'_>, field: &FiniteField, e: &FieldElement) -> PyObject {
    if field.is_prime_field() {
        e.as_u64().unwrap_or(0).into_py(py)
    } else {
        PyList::new_bound(py, e.coefficients_padded(field.degree())).into()
    }
}

// ---------------------------------------------------------------------------
// FiniteField
// ---------------------------------------------------------------------------

/// A finite field GF(q) with `q = p^k`.
///
/// ``FiniteField(p)`` is the prime field GF(p). ``FiniteField(p, k)`` is the
/// degree-`k` extension, built from FLINT's Conway polynomial when one is
/// tabulated for `(p, k)` and from its deterministic minimal-weight irreducible
/// otherwise. ``FiniteField.with_defining_polynomial(p, coeffs)`` pins the
/// defining polynomial explicitly; it is checked for irreducibility first.
///
/// Two fields of the same order but different defining polynomials are
/// isomorphic and are **not** interchangeable here: an element's coordinates
/// mean different things in each, so mixing them raises ``E-GFQ-006``.
#[pyclass(name = "FiniteField", module = "alkahest")]
#[derive(Clone)]
pub struct PyFiniteField {
    pub(crate) inner: FiniteField,
}

#[pymethods]
impl PyFiniteField {
    #[new]
    #[pyo3(signature = (p, k = 1))]
    fn new(p: &Bound<'_, PyAny>, k: u32) -> PyResult<Self> {
        // `p` is taken as a `PyAny` rather than a `u64` on purpose: a Python
        // int has no width, and letting PyO3's conversion overflow would turn
        // "this characteristic does not fit in a machine word" — which has a
        // stable code, `E-GFQ-002`, and a documented remediation — into a bare
        // `OverflowError` naming nothing.
        let decimal = p.str()?.to_string_lossy().into_owned();
        let base = FiniteField::prime_from_decimal(&decimal).map_err(ff_err)?;
        if k == 1 {
            return Ok(Self { inner: base });
        }
        FiniteField::extension(base.characteristic(), k)
            .map(|inner| Self { inner })
            .map_err(ff_err)
    }

    /// GF(p^k) from an explicit defining polynomial.
    ///
    /// ``coeffs`` ascend in degree, so ``x**3 + x + 1`` over GF(2) is
    /// ``[1, 1, 0, 1]``. Raises ``E-GFQ-004`` if it is reducible over GF(p) —
    /// a reducible modulus would make the quotient a ring with zero divisors,
    /// in which "rank" and "nullspace" stop being well defined.
    #[classmethod]
    fn with_defining_polynomial(
        _cls: &Bound<'_, PyType>,
        p: u64,
        coeffs: Vec<u64>,
    ) -> PyResult<Self> {
        FiniteField::with_defining_polynomial(p, &coeffs)
            .map(|inner| Self { inner })
            .map_err(ff_err)
    }

    /// The characteristic `p`.
    #[getter]
    fn characteristic(&self) -> u64 {
        self.inner.characteristic()
    }

    /// The extension degree `k`; `1` for a prime field.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// The order `q = p**k`.
    #[getter]
    fn order(&self) -> Option<u128> {
        self.inner.order()
    }

    /// ``True`` when this is GF(p) rather than a proper extension.
    #[getter]
    fn is_prime_field(&self) -> bool {
        self.inner.is_prime_field()
    }

    /// Coefficients of the defining polynomial, ascending. Empty for GF(p).
    fn defining_polynomial(&self) -> Vec<u64> {
        self.inner.defining_polynomial().to_vec()
    }

    /// The generator `a` of the extension, or ``None`` for a prime field.
    fn generator(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner
            .generator()
            .map(|g| elem_to_py(py, &self.inner, &g))
    }

    /// Render an element in this field's generator, e.g. ``"a^2 + 1"``.
    fn render(&self, value: &Bound<'_, PyAny>) -> PyResult<String> {
        let e = elem_from_py(&self.inner, value)?;
        Ok(self.inner.render(&e))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        // A comparison against something that is not a field is `False`, not a
        // `TypeError`: `field in [...]` and `==` on mixed containers are normal.
        other
            .extract::<PyRef<'_, Self>>()
            .map(|o| self.inner == o.inner)
            .unwrap_or(false)
    }

    fn __hash__(&self) -> u64 {
        let mut h = u64::from(self.inner.degree() as u32);
        h = h
            .wrapping_mul(1_000_003)
            .wrapping_add(self.inner.characteristic());
        for &c in self.inner.defining_polynomial() {
            h = h.wrapping_mul(1_000_003).wrapping_add(c);
        }
        h
    }
}

// ---------------------------------------------------------------------------
// GfMatrix
// ---------------------------------------------------------------------------

/// A dense matrix over GF(q).
///
/// ``GfMatrix(field, [[1, 0, 1], [0, 1, 1]])`` builds from nested rows.
/// Rectangular shapes are first-class — parity-check matrices are never square.
#[pyclass(name = "GfMatrix", module = "alkahest")]
pub struct PyGfMatrix {
    inner: GfMatrix,
}

impl PyGfMatrix {
    fn wrap(inner: GfMatrix) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyGfMatrix {
    #[new]
    fn new(field: &PyFiniteField, rows: &Bound<'_, PyAny>) -> PyResult<Self> {
        let f = &field.inner;
        let row_objs: Vec<Bound<'_, PyAny>> = rows.extract().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "GfMatrix(field, rows) expects a sequence of rows",
            )
        })?;
        let nrows = row_objs.len();
        let mut entries: Vec<FieldElement> = Vec::new();
        let mut ncols = 0usize;
        for (i, row) in row_objs.iter().enumerate() {
            let cells: Vec<Bound<'_, PyAny>> = row.extract().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "row {i} is not a sequence of entries"
                ))
            })?;
            if i == 0 {
                ncols = cells.len();
            } else if cells.len() != ncols {
                return Err(ff_err(FiniteFieldError::DimensionMismatch {
                    op: "fill",
                    lhs: (nrows, ncols),
                    rhs: (i + 1, cells.len()),
                }));
            }
            for cell in &cells {
                entries.push(elem_from_py(f, cell)?);
            }
        }
        GfMatrix::from_elements(f, nrows, ncols, &entries)
            .map(Self::wrap)
            .map_err(ff_err)
    }

    /// The all-zero ``rows x cols`` matrix.
    #[classmethod]
    fn zeros(
        _cls: &Bound<'_, PyType>,
        field: &PyFiniteField,
        rows: usize,
        cols: usize,
    ) -> PyResult<Self> {
        GfMatrix::zeros(&field.inner, rows, cols)
            .map(Self::wrap)
            .map_err(ff_err)
    }

    /// The ``n x n`` identity.
    #[classmethod]
    fn identity(_cls: &Bound<'_, PyType>, field: &PyFiniteField, n: usize) -> PyResult<Self> {
        GfMatrix::identity(&field.inner, n)
            .map(Self::wrap)
            .map_err(ff_err)
    }

    /// Build from a flat row-major sequence of entries.
    #[classmethod]
    fn from_flat(
        _cls: &Bound<'_, PyType>,
        field: &PyFiniteField,
        rows: usize,
        cols: usize,
        entries: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let cells: Vec<Bound<'_, PyAny>> = entries.extract().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>("entries must be a sequence")
        })?;
        let f = &field.inner;
        let elems: Vec<FieldElement> = cells
            .iter()
            .map(|c| elem_from_py(f, c))
            .collect::<PyResult<_>>()?;
        GfMatrix::from_elements(f, rows, cols, &elems)
            .map(Self::wrap)
            .map_err(ff_err)
    }

    /// Number of rows.
    #[getter]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    /// Number of columns.
    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// The field this matrix lives over.
    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    /// ``(rows, cols)``.
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Entry ``(i, j)``.
    fn entry(&self, py: Python<'_>, i: usize, j: usize) -> PyResult<PyObject> {
        let e = self.inner.entry(i, j).map_err(ff_err)?;
        Ok(elem_to_py(py, self.inner.field(), &e))
    }

    /// Overwrite entry ``(i, j)``.
    fn set_entry(&mut self, i: usize, j: usize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let e = elem_from_py(self.inner.field(), value)?;
        self.inner.set_entry(i, j, &e).map_err(ff_err)
    }

    /// All entries as nested rows.
    fn to_list(&self, py: Python<'_>) -> PyObject {
        let (r, c) = self.inner.shape();
        let f = self.inner.field();
        let rows: Vec<PyObject> = (0..r)
            .map(|i| {
                let cells: Vec<PyObject> = (0..c)
                    .map(|j| {
                        elem_to_py(
                            py,
                            f,
                            &self.inner.entry(i, j).expect("index built from the shape"),
                        )
                    })
                    .collect();
                PyList::new_bound(py, cells).into()
            })
            .collect();
        PyList::new_bound(py, rows).into()
    }

    /// ``True`` when every entry is zero. O(rows * cols), hence a method.
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Entrywise sum.
    fn add(&self, other: &Self) -> PyResult<Self> {
        self.inner.add(&other.inner).map(Self::wrap).map_err(ff_err)
    }

    /// Entrywise difference.
    fn sub(&self, other: &Self) -> PyResult<Self> {
        self.inner.sub(&other.inner).map(Self::wrap).map_err(ff_err)
    }

    /// Additive inverse.
    fn neg(&self) -> Self {
        Self::wrap(self.inner.neg())
    }

    /// Matrix product ``self @ other``.
    fn mul(&self, other: &Self) -> PyResult<Self> {
        self.inner.mul(&other.inner).map(Self::wrap).map_err(ff_err)
    }

    /// Multiply every entry by a field element.
    fn scalar_mul(&self, c: &Bound<'_, PyAny>) -> PyResult<Self> {
        let e = elem_from_py(self.inner.field(), c)?;
        self.inner.scalar_mul(&e).map(Self::wrap).map_err(ff_err)
    }

    /// Transpose.
    fn transpose(&self) -> Self {
        Self::wrap(self.inner.transpose())
    }

    /// Rank over GF(q). Defined for any shape.
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Reduced row echelon form, with the invertible transform ``U`` such that
    /// ``U @ A == R``.
    fn rref(&self, py: Python<'_>) -> PyResult<PyGfRref> {
        let r = self.inner.rref().map_err(ff_err)?;
        let shape = r.matrix.shape();
        Ok(PyGfRref {
            matrix: Py::new(py, Self::wrap(r.matrix))?,
            transform: Py::new(py, Self::wrap(r.transform))?,
            rank: r.rank,
            pivots: r.pivots,
            shape,
        })
    }

    /// A basis for the right nullspace ``{x : A @ x == 0}``, as the **columns**
    /// of an ``ncols x nullity`` matrix.
    ///
    /// For coding work the next step is usually ``.transpose()``, which turns
    /// the kernel basis of a parity-check matrix ``H`` into a generator matrix
    /// ``G`` with ``H @ G.transpose() == 0``.
    fn nullspace(&self) -> PyResult<Self> {
        self.inner.nullspace().map(Self::wrap).map_err(ff_err)
    }

    /// One solution ``X`` of ``A @ X == B``, for any shape of ``A``.
    ///
    /// The full solution set is that ``X`` plus the span of
    /// :meth:`nullspace`. Raises ``E-GFQ-010`` when no solution exists — an
    /// inconsistent system is refused rather than least-squares'd.
    fn solve(&self, rhs: &Self) -> PyResult<Self> {
        self.inner.solve(&rhs.inner).map(Self::wrap).map_err(ff_err)
    }

    /// The multiplicative inverse. Raises ``E-GFQ-009`` when singular.
    fn inverse(&self) -> PyResult<Self> {
        self.inner.inverse().map(Self::wrap).map_err(ff_err)
    }

    /// The determinant.
    fn determinant(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = self.inner.determinant().map_err(ff_err)?;
        Ok(elem_to_py(py, self.inner.field(), &d))
    }

    /// Coefficients of the characteristic polynomial ``det(x*I - A)``,
    /// ascending, so the last entry is the leading ``1``.
    fn charpoly(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cp = self.inner.charpoly().map_err(ff_err)?;
        let f = self.inner.field();
        let cells: Vec<PyObject> = cp.iter().map(|c| elem_to_py(py, f, c)).collect();
        Ok(PyList::new_bound(py, cells).into())
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.add(other)
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.sub(other)
    }

    fn __neg__(&self) -> Self {
        self.neg()
    }

    fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        self.mul(other)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .map(|o| self.inner.equals(&o.inner))
            .unwrap_or(false)
    }

    fn __repr__(&self) -> String {
        let (r, c) = self.inner.shape();
        format!("GfMatrix({r}x{c} over {})", self.inner.field())
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __copy__(&self) -> Self {
        Self::wrap(self.inner.clone())
    }
}

/// The result of :meth:`GfMatrix.rref`.
///
/// ``matrix`` and ``transform`` are stored, not recomputed: reading either is a
/// reference bump, which is what makes them properties rather than methods
/// under CONTRIBUTING's accessor rule.
#[pyclass(name = "GfRref", module = "alkahest")]
pub struct PyGfRref {
    matrix: Py<PyGfMatrix>,
    transform: Py<PyGfMatrix>,
    rank: usize,
    pivots: Vec<usize>,
    shape: (usize, usize),
}

#[pymethods]
impl PyGfRref {
    /// ``R``, the reduced row echelon form — same shape as the input.
    #[getter]
    fn matrix(&self, py: Python<'_>) -> Py<PyGfMatrix> {
        self.matrix.clone_ref(py)
    }

    /// ``U``, invertible and ``rows x rows``, with ``U @ A == R``.
    #[getter]
    fn transform(&self, py: Python<'_>) -> Py<PyGfMatrix> {
        self.transform.clone_ref(py)
    }

    /// The rank of the input.
    #[getter]
    fn rank(&self) -> usize {
        self.rank
    }

    /// Pivot column index of each of the first ``rank`` rows, ascending.
    fn pivots(&self) -> Vec<usize> {
        self.pivots.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "GfRref(rank={}, pivots={:?}, shape={:?})",
            self.rank, self.pivots, self.shape
        )
    }
}

/// Register the GF(q) surface on the `alkahest` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFiniteField>()?;
    m.add_class::<PyGfMatrix>()?;
    m.add_class::<PyGfRref>()?;
    m.add(
        "FiniteFieldError",
        m.py().get_type_bound::<PyFiniteFieldError>(),
    )?;
    Ok(())
}
