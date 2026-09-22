//! PyO3 bindings for `alkahest_core::theta` — Riemann theta functions,
//! classical modular functions and the Weierstrass family, as rigorous
//! enclosures.
//!
//! The Python surface is `alkahest.experimental.j_invariant`,
//! `alkahest.experimental.riemann_theta`, … and refusals arrive as
//! `alkahest.experimental.ThetaError` with a stable `E-THETA-NNN` `.code`.
//!
//! # What crosses the boundary
//!
//! Values come back as [`ComplexBall`], never as a Python `complex`. That is
//! the whole point: a `complex` would silently drop the radius, and the radius
//! is the part a caller cannot reconstruct. `ComplexBall.value()` produces a
//! `complex`, and **refuses** unless the enclosure meets a stated accuracy —
//! so the conversion is always a deliberate act with a threshold attached.
//!
//! Arguments are accepted as `ComplexBall`, `complex`, `float` or `int`; the
//! plain-Python forms are read as *exact* inputs, which they are (a Python
//! float is a binary double).
//!
//! # Precision
//!
//! Every evaluator takes `prec` (working precision in bits, default 256) and
//! an optional `accurate_to`. Passing `accurate_to=N` switches to the refining
//! mode: the working precision is raised until the result carries `N` bits of
//! **relative** accuracy, or `ThetaError` (`E-THETA-010`) is raised. Note that
//! a value which is exactly zero — `j(rho)`, `theta_1(0, tau)` — can never
//! satisfy `accurate_to` and must be checked with `contains_zero()` instead.

use pyo3::prelude::*;
use pyo3::types::PyComplex;

use alkahest_core::experimental::{
    arb_backend_available, dedekind_eta, eisenstein_series, j_invariant, jacobi_theta,
    jacobi_theta_null, modular_discriminant, modular_lambda, riemann_theta,
    riemann_theta_available, riemann_theta_characteristic, riemann_theta_squared,
    siegel_is_reduced, siegel_reduce, theta_characteristic_bits, theta_characteristic_index,
    theta_characteristic_is_even, weierstrass_invariants, weierstrass_p, weierstrass_p_prime,
    weierstrass_roots, weierstrass_sigma, weierstrass_zeta, ComplexBall, Precision, SiegelMatrix,
    SiegelReduction, ThetaError, ThetaValues,
};

pyo3::create_exception!(alkahest, PyThetaError, crate::PyAlkahestError);

fn th_err(e: ThetaError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyThetaError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

const DEFAULT_PREC: u32 = 256;

fn precision(prec: u32, accurate_to: Option<u32>) -> Precision {
    match accurate_to {
        Some(bits) => Precision::AccurateTo(bits),
        None => Precision::Bits(prec),
    }
}

// ---------------------------------------------------------------------------
// ComplexBall
// ---------------------------------------------------------------------------

/// A complex number as a ball: a midpoint and a radius on each component, with
/// the guarantee that the true value lies inside.
///
/// ``ComplexBall(re, im, prec=256)`` is the **exact** ball at ``re + im*1j``;
/// a Python float is a binary double and crosses without rounding.
///
/// ``midpoint_real`` and friends are named *midpoint*, not *value*, on purpose.
/// The midpoint of ``[0 +/- 1e9]`` is ``0`` and means nothing. Use ``value()``,
/// which refuses unless the enclosure is good enough, or read
/// ``accuracy_bits``.
#[pyclass(name = "ComplexBall", module = "alkahest")]
#[derive(Clone)]
pub struct PyComplexBall {
    inner: ComplexBall,
}

/// Read one argument: a `ComplexBall`, a `complex`, a `float` or an `int`.
///
/// The plain-Python forms are read as **exact** balls, which is what they are:
/// a Python float is a binary double and crosses without rounding. Nothing here
/// invents a radius for an input the caller stated exactly.
fn ball_from_any(obj: &Bound<'_, PyAny>, prec: u32) -> PyResult<ComplexBall> {
    if let Ok(b) = obj.extract::<PyComplexBall>() {
        return Ok(b.inner);
    }
    if let Ok(c) = obj.downcast::<PyComplex>() {
        return Ok(ComplexBall::exact_f64(c.real(), c.imag(), prec));
    }
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(ComplexBall::exact_f64(v, 0.0, prec));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "expected a ComplexBall, a complex, a float or an int",
    ))
}

/// Hand Python an exact integer of any size.
fn integer_to_py(py: Python<'_>, i: &rug::Integer) -> PyResult<PyObject> {
    if let Some(v) = i.to_i64() {
        return Ok(v.into_py(py));
    }
    Ok(py
        .eval_bound(&format!("int('{i}')"), None, None)?
        .into_py(py))
}

#[pymethods]
impl PyComplexBall {
    #[new]
    #[pyo3(signature = (re=0.0, im=0.0, prec=DEFAULT_PREC))]
    fn new(re: f64, im: f64, prec: u32) -> Self {
        PyComplexBall {
            inner: ComplexBall::exact_f64(re, im, prec),
        }
    }

    /// The ball ``[mid_re +/- rad_re] + i [mid_im +/- rad_im]``.
    ///
    /// Use this to carry a *measured* or otherwise uncertain input through the
    /// computation: the output radius will reflect it rather than pretend it
    /// away.
    #[staticmethod]
    #[pyo3(signature = (mid_re, rad_re, mid_im, rad_im, prec=DEFAULT_PREC))]
    fn from_midpoint_radius(mid_re: f64, rad_re: f64, mid_im: f64, rad_im: f64, prec: u32) -> Self {
        use alkahest_core::experimental::RealBall;
        let p = prec.max(53);
        let mk = |m: f64, r: f64| {
            RealBall::from_mid_rad(
                rug::Float::with_val(p, m),
                rug::Float::with_val(p, r.abs()),
                prec,
            )
        };
        PyComplexBall {
            inner: ComplexBall::from_parts(mk(mid_re, rad_re), mk(mid_im, rad_im)),
        }
    }

    /// ``exp(2 pi i / n)``, as an enclosure.
    ///
    /// Present because the classical checkpoints for this module sit at ``i``
    /// and at ``rho = exp(2 pi i / 3)``, and writing ``rho`` as a decimal
    /// literal would turn "j vanishes at rho" into "j is small near something
    /// close to rho".
    #[staticmethod]
    #[pyo3(signature = (n, prec=DEFAULT_PREC))]
    fn root_of_unity(n: u32, prec: u32) -> PyResult<Self> {
        Ok(PyComplexBall {
            inner: ComplexBall::root_of_unity(n, prec).map_err(th_err)?,
        })
    }

    /// An enclosure of ``pi`` on the real axis.
    #[staticmethod]
    #[pyo3(signature = (prec=DEFAULT_PREC))]
    fn pi(prec: u32) -> PyResult<Self> {
        Ok(PyComplexBall {
            inner: ComplexBall::pi(prec).map_err(th_err)?,
        })
    }

    /// Midpoint of the real part, as a float. **Not** the value.
    #[getter]
    fn midpoint_real(&self) -> f64 {
        self.inner.real().midpoint_f64()
    }

    /// Midpoint of the imaginary part, as a float. **Not** the value.
    #[getter]
    fn midpoint_imag(&self) -> f64 {
        self.inner.imag().midpoint_f64()
    }

    /// Radius of the real part, rounded **up**.
    #[getter]
    fn radius_real(&self) -> f64 {
        self.inner.real().radius_f64()
    }

    /// Radius of the imaginary part, rounded **up**.
    #[getter]
    fn radius_imag(&self) -> f64 {
        self.inner.imag().radius_f64()
    }

    /// ``floor(log2(|mid| / rad))`` — how many leading bits of the midpoint the
    /// radius justifies. Very large for an exact ball, very negative for one
    /// whose radius swamps its midpoint.
    #[getter]
    fn accuracy_bits(&self) -> i64 {
        self.inner.accuracy_bits()
    }

    /// The working precision, in bits, that produced this ball.
    #[getter]
    fn precision(&self) -> u32 {
        self.inner.precision()
    }

    /// Is the radius infinite? Then the midpoint means nothing.
    #[getter]
    fn is_indeterminate(&self) -> bool {
        self.inner.is_indeterminate()
    }

    /// Is the radius exactly zero?
    #[getter]
    fn is_exact(&self) -> bool {
        self.inner.is_exact()
    }

    /// ``(lo, hi)`` for the real part, rounded outward.
    fn real_interval(&self) -> (f64, f64) {
        self.inner.real().interval_f64()
    }

    /// ``(lo, hi)`` for the imaginary part, rounded outward.
    fn imag_interval(&self) -> (f64, f64) {
        self.inner.imag().interval_f64()
    }

    /// Does this ball **certainly** contain the given point?
    #[pyo3(signature = (re, im=0.0))]
    fn contains(&self, re: f64, im: f64) -> bool {
        self.inner.contains_f64(re, im)
    }

    /// Does this ball certainly contain zero?
    ///
    /// This — not ``value()`` — is the right question for a quantity that is
    /// expected to vanish, such as ``j(rho)`` or ``theta_1(0, tau)``.
    fn contains_zero(&self) -> bool {
        self.inner.contains_zero()
    }

    /// Do the two balls share a point?
    ///
    /// The test to use when checking an identity between two computed results:
    /// it is true exactly when the identity is consistent with both error
    /// bounds.
    fn overlaps(&self, other: &PyComplexBall) -> bool {
        self.inner.overlaps(&other.inner)
    }

    /// The midpoint as a Python ``complex`` — **only if** the enclosure carries
    /// at least ``min_accuracy_bits`` bits of relative accuracy.
    ///
    /// Raises ``ThetaError`` (``E-THETA-010`` / ``E-THETA-011``) otherwise. A
    /// value that is exactly zero can never satisfy this and should be checked
    /// with ``contains_zero()``.
    #[pyo3(signature = (min_accuracy_bits=53))]
    fn value(&self, py: Python<'_>, min_accuracy_bits: u32) -> PyResult<PyObject> {
        let (re, im) = self
            .inner
            .value_if_accurate("value", min_accuracy_bits)
            .map_err(th_err)?;
        Ok(PyComplex::from_doubles_bound(py, re, im).into())
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let p = self.inner.precision();
        let rhs = ball_from_any(other, p)?;
        Ok(PyComplexBall {
            inner: self.inner.add(&rhs, p).map_err(th_err)?,
        })
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let p = self.inner.precision();
        let rhs = ball_from_any(other, p)?;
        Ok(PyComplexBall {
            inner: self.inner.sub(&rhs, p).map_err(th_err)?,
        })
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let p = self.inner.precision();
        let rhs = ball_from_any(other, p)?;
        Ok(PyComplexBall {
            inner: self.inner.mul(&rhs, p).map_err(th_err)?,
        })
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let p = self.inner.precision();
        let rhs = ball_from_any(other, p)?;
        Ok(PyComplexBall {
            inner: self.inner.div(&rhs, p).map_err(th_err)?,
        })
    }

    fn __neg__(&self) -> PyResult<Self> {
        let p = self.inner.precision();
        Ok(PyComplexBall {
            inner: self.inner.neg(p).map_err(th_err)?,
        })
    }

    /// The principal square root.
    fn sqrt(&self) -> PyResult<Self> {
        let p = self.inner.precision();
        Ok(PyComplexBall {
            inner: self.inner.sqrt(p).map_err(th_err)?,
        })
    }

    /// ``exp(self)``.
    fn exp(&self) -> PyResult<Self> {
        let p = self.inner.precision();
        Ok(PyComplexBall {
            inner: self.inner.exp(p).map_err(th_err)?,
        })
    }

    /// The principal logarithm.
    fn log(&self) -> PyResult<Self> {
        let p = self.inner.precision();
        Ok(PyComplexBall {
            inner: self.inner.log(p).map_err(th_err)?,
        })
    }

    /// The complex conjugate.
    fn conjugate(&self) -> PyResult<Self> {
        let p = self.inner.precision();
        Ok(PyComplexBall {
            inner: self.inner.conj(p).map_err(th_err)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("ComplexBall({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// SiegelMatrix
// ---------------------------------------------------------------------------

/// A symmetric ``g x g`` complex matrix — a candidate point of the Siegel upper
/// half-space ``H_g``.
///
/// ``SiegelMatrix(genus, entries)`` takes ``g*g`` entries in row-major order
/// and checks symmetry as **ball identity**: two separately-computed enclosures
/// of one number are two different inputs and are refused (``E-THETA-006``).
/// ``SiegelMatrix.from_upper_triangle(genus, entries)`` fills the lower half by
/// copying and cannot trip that check.
#[pyclass(name = "SiegelMatrix", module = "alkahest")]
#[derive(Clone)]
pub struct PySiegelMatrix {
    inner: SiegelMatrix,
}

fn ball_list(objs: &Bound<'_, PyAny>, prec: u32) -> PyResult<Vec<ComplexBall>> {
    let mut out = Vec::new();
    for item in objs.iter()? {
        out.push(ball_from_any(&item?, prec)?);
    }
    Ok(out)
}

#[pymethods]
impl PySiegelMatrix {
    #[new]
    #[pyo3(signature = (genus, entries, prec=DEFAULT_PREC))]
    fn new(genus: usize, entries: &Bound<'_, PyAny>, prec: u32) -> PyResult<Self> {
        let e = ball_list(entries, prec)?;
        Ok(PySiegelMatrix {
            inner: SiegelMatrix::new(genus, e).map_err(th_err)?,
        })
    }

    /// Build from the upper triangle, row by row and including the diagonal:
    /// ``g (g + 1) / 2`` entries ordered ``(0,0), (0,1), …, (1,1), …``.
    #[staticmethod]
    #[pyo3(signature = (genus, entries, prec=DEFAULT_PREC))]
    fn from_upper_triangle(genus: usize, entries: &Bound<'_, PyAny>, prec: u32) -> PyResult<Self> {
        let e = ball_list(entries, prec)?;
        Ok(PySiegelMatrix {
            inner: SiegelMatrix::from_upper_triangle(genus, e).map_err(th_err)?,
        })
    }

    /// The genus-1 case: a single ``tau`` in the upper half-plane.
    #[staticmethod]
    #[pyo3(signature = (tau, prec=DEFAULT_PREC))]
    fn genus_one(tau: &Bound<'_, PyAny>, prec: u32) -> PyResult<Self> {
        Ok(PySiegelMatrix {
            inner: SiegelMatrix::genus_one(ball_from_any(tau, prec)?),
        })
    }

    /// The genus (the side length of the matrix).
    #[getter]
    fn genus(&self) -> usize {
        self.inner.genus()
    }

    /// Entry ``(i, j)``.
    fn entry(&self, i: usize, j: usize) -> PyResult<PyComplexBall> {
        self.inner
            .entry(i, j)
            .map(|b| PyComplexBall { inner: b.clone() })
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "({i}, {j}) is outside a {0}x{0} period matrix",
                    self.inner.genus()
                ))
            })
    }

    /// All entries, row-major.
    fn entries(&self) -> Vec<PyComplexBall> {
        self.inner
            .entries()
            .iter()
            .map(|b| PyComplexBall { inner: b.clone() })
            .collect()
    }

    /// Is ``Im(tau)`` **certainly** positive definite at ``prec`` bits?
    ///
    /// ``False`` means "not proved at this precision", never "proved false".
    #[pyo3(signature = (prec=DEFAULT_PREC))]
    fn in_siegel_upper_half_space(&self, prec: u32) -> PyResult<bool> {
        self.inner
            .is_certainly_in_siegel_upper_half_space(prec)
            .map_err(th_err)
    }

    fn __repr__(&self) -> String {
        format!("SiegelMatrix(genus={})", self.inner.genus())
    }
}

// ---------------------------------------------------------------------------
// ThetaValues
// ---------------------------------------------------------------------------

/// The ``4^g`` values ``theta[a;b](z, tau)`` for one ``(z, tau)``, indexed by
/// characteristic.
///
/// The index is the ``2g``-bit integer ``(a << g) | b``, most significant bit
/// first within each half; ``theta_characteristic_index`` builds it from bit
/// lists so the ordering never has to be rediscovered at a call site.
#[pyclass(name = "ThetaValues", module = "alkahest")]
#[derive(Clone)]
pub struct PyThetaValues {
    inner: ThetaValues,
}

#[pymethods]
impl PyThetaValues {
    /// The genus.
    #[getter]
    fn genus(&self) -> usize {
        self.inner.genus()
    }

    /// Are these ``theta^2`` rather than ``theta``?
    ///
    /// Squares determine theta only up to sign, so the two are kept apart by
    /// this flag rather than silently interchanged.
    #[getter]
    fn is_squared(&self) -> bool {
        self.inner.is_squared()
    }

    /// The relative accuracy of the **worst** value in the vector — the number
    /// to look at before believing any of them.
    #[getter]
    fn worst_accuracy_bits(&self) -> i64 {
        self.inner.worst_accuracy_bits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, ab: u64) -> PyResult<PyComplexBall> {
        self.inner
            .get(ab)
            .map(|b| PyComplexBall { inner: b.clone() })
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "characteristic {ab} is outside 0..4^{}",
                    self.inner.genus()
                ))
            })
    }

    /// The value at the characteristic ``(a, b)`` given as two bit lists.
    fn get_characteristic(&self, a: Vec<u8>, b: Vec<u8>) -> PyResult<PyComplexBall> {
        Ok(PyComplexBall {
            inner: self
                .inner
                .get_characteristic(&a, &b)
                .map_err(th_err)?
                .clone(),
        })
    }

    /// All values, in characteristic order.
    fn values(&self) -> Vec<PyComplexBall> {
        self.inner
            .values()
            .iter()
            .map(|b| PyComplexBall { inner: b.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ThetaValues(genus={}, squared={}, n={}, worst_accuracy_bits={})",
            self.inner.genus(),
            self.inner.is_squared(),
            self.inner.len(),
            self.inner.worst_accuracy_bits()
        )
    }
}

// ---------------------------------------------------------------------------
// SiegelReduction
// ---------------------------------------------------------------------------

/// The result of ``siegel_reduce``: a symplectic matrix and the point it moves
/// ``tau`` to.
#[pyclass(name = "SiegelReduction", module = "alkahest")]
#[derive(Clone)]
pub struct PySiegelReduction {
    inner: SiegelReduction,
}

#[pymethods]
impl PySiegelReduction {
    /// The genus.
    #[getter]
    fn genus(&self) -> usize {
        self.inner.genus()
    }

    /// The symplectic matrix, as a list of ``2g`` rows of ``2g`` integers.
    fn symplectic(&self, py: Python<'_>) -> PyResult<Vec<Vec<PyObject>>> {
        let n = 2 * self.inner.genus();
        let flat = self.inner.symplectic();
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                row.push(integer_to_py(py, &flat[i * n + j])?);
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// ``M . tau``, as enclosures.
    #[getter]
    fn reduced(&self) -> PySiegelMatrix {
        PySiegelMatrix {
            inner: self.inner.reduced().clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("SiegelReduction(genus={})", self.inner.genus())
    }
}

// ---------------------------------------------------------------------------
// Genus-1 functions
// ---------------------------------------------------------------------------

/// The Dedekind eta function ``eta(tau)``.
///
/// ``eta(i) = Gamma(1/4) / (2 pi^(3/4)) = 0.76822542...``.
#[pyfunction]
#[pyo3(name = "dedekind_eta", signature = (tau, prec=DEFAULT_PREC, accurate_to=None))]
fn py_dedekind_eta(
    tau: &Bound<'_, PyAny>,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<PyComplexBall> {
    let t = ball_from_any(tau, prec)?;
    Ok(PyComplexBall {
        inner: dedekind_eta(&t, precision(prec, accurate_to)).map_err(th_err)?,
    })
}

/// Klein's ``j``-invariant, normalised so that ``j(i) = 1728`` and ``j(rho) = 0``
/// for ``rho = exp(2 pi i / 3)``.
///
/// ``j(rho) = 0`` must be checked with ``contains_zero()``: an exactly zero
/// value has no relative accuracy, so ``accurate_to`` will correctly refuse.
#[pyfunction]
#[pyo3(name = "j_invariant", signature = (tau, prec=DEFAULT_PREC, accurate_to=None))]
fn py_j_invariant(
    tau: &Bound<'_, PyAny>,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<PyComplexBall> {
    let t = ball_from_any(tau, prec)?;
    Ok(PyComplexBall {
        inner: j_invariant(&t, precision(prec, accurate_to)).map_err(th_err)?,
    })
}

/// The modular lambda function.
#[pyfunction]
#[pyo3(name = "modular_lambda", signature = (tau, prec=DEFAULT_PREC, accurate_to=None))]
fn py_modular_lambda(
    tau: &Bound<'_, PyAny>,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<PyComplexBall> {
    let t = ball_from_any(tau, prec)?;
    Ok(PyComplexBall {
        inner: modular_lambda(&t, precision(prec, accurate_to)).map_err(th_err)?,
    })
}

/// The modular discriminant ``Delta(tau) = eta(tau)^24``.
#[pyfunction]
#[pyo3(name = "modular_discriminant", signature = (tau, prec=DEFAULT_PREC, accurate_to=None))]
fn py_modular_discriminant(
    tau: &Bound<'_, PyAny>,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<PyComplexBall> {
    let t = ball_from_any(tau, prec)?;
    Ok(PyComplexBall {
        inner: modular_discriminant(&t, precision(prec, accurate_to)).map_err(th_err)?,
    })
}

/// The normalised Eisenstein series ``E_4, E_6, ..., E_{2*count+2}``.
#[pyfunction]
#[pyo3(name = "eisenstein_series", signature = (tau, count, prec=DEFAULT_PREC, accurate_to=None))]
fn py_eisenstein_series(
    tau: &Bound<'_, PyAny>,
    count: usize,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<Vec<PyComplexBall>> {
    let t = ball_from_any(tau, prec)?;
    Ok(eisenstein_series(&t, count, precision(prec, accurate_to))
        .map_err(th_err)?
        .into_iter()
        .map(|inner| PyComplexBall { inner })
        .collect())
}

/// The four classical Jacobi theta functions ``[theta_1, theta_2, theta_3,
/// theta_4]`` at ``(z, tau)``, with ``q = exp(pi i tau)`` and ``w = exp(pi i z)``.
#[pyfunction]
#[pyo3(name = "jacobi_theta", signature = (z, tau, prec=DEFAULT_PREC, accurate_to=None))]
fn py_jacobi_theta(
    z: &Bound<'_, PyAny>,
    tau: &Bound<'_, PyAny>,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<Vec<PyComplexBall>> {
    let zz = ball_from_any(z, prec)?;
    let t = ball_from_any(tau, prec)?;
    Ok(jacobi_theta(&zz, &t, precision(prec, accurate_to))
        .map_err(th_err)?
        .into_iter()
        .map(|inner| PyComplexBall { inner })
        .collect())
}

/// The Jacobi theta constants ``[theta_1(0), theta_2(0), theta_3(0),
/// theta_4(0)]``.
///
/// ``theta_1(0, tau)`` is identically zero, so ``accurate_to`` can never be
/// satisfied by this function; check the first entry with ``contains_zero()``.
#[pyfunction]
#[pyo3(name = "jacobi_theta_null", signature = (tau, prec=DEFAULT_PREC, accurate_to=None))]
fn py_jacobi_theta_null(
    tau: &Bound<'_, PyAny>,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<Vec<PyComplexBall>> {
    let t = ball_from_any(tau, prec)?;
    Ok(jacobi_theta_null(&t, precision(prec, accurate_to))
        .map_err(th_err)?
        .into_iter()
        .map(|inner| PyComplexBall { inner })
        .collect())
}

macro_rules! elliptic_binding {
    ($py_fn:ident, $core_fn:ident, $name:literal, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        #[pyo3(name = $name, signature = (z, tau, prec=DEFAULT_PREC, accurate_to=None))]
        fn $py_fn(
            z: &Bound<'_, PyAny>,
            tau: &Bound<'_, PyAny>,
            prec: u32,
            accurate_to: Option<u32>,
        ) -> PyResult<PyComplexBall> {
            let zz = ball_from_any(z, prec)?;
            let t = ball_from_any(tau, prec)?;
            Ok(PyComplexBall {
                inner: $core_fn(&zz, &t, precision(prec, accurate_to)).map_err(th_err)?,
            })
        }
    };
}

elliptic_binding!(
    py_weierstrass_p,
    weierstrass_p,
    "weierstrass_p",
    "The Weierstrass elliptic function ``p(z, tau)`` for the lattice ``Z + tau Z``.\n\nAt a lattice point the result is indeterminate (an infinite radius), which is the honest answer for a double pole."
);
elliptic_binding!(
    py_weierstrass_p_prime,
    weierstrass_p_prime,
    "weierstrass_p_prime",
    "``p'(z, tau)``, the derivative of ``weierstrass_p`` with respect to ``z``."
);
elliptic_binding!(
    py_weierstrass_zeta,
    weierstrass_zeta,
    "weierstrass_zeta",
    "The Weierstrass zeta function ``zeta(z, tau)``."
);
elliptic_binding!(
    py_weierstrass_sigma,
    weierstrass_sigma,
    "weierstrass_sigma",
    "The Weierstrass sigma function ``sigma(z, tau)``."
);

/// The lattice invariants ``(g2, g3)`` of ``Z + tau Z``, so that
/// ``p'^2 = 4 p^3 - g2 p - g3``.
#[pyfunction]
#[pyo3(name = "weierstrass_invariants", signature = (tau, prec=DEFAULT_PREC, accurate_to=None))]
fn py_weierstrass_invariants(
    tau: &Bound<'_, PyAny>,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<(PyComplexBall, PyComplexBall)> {
    let t = ball_from_any(tau, prec)?;
    let (g2, g3) = weierstrass_invariants(&t, precision(prec, accurate_to)).map_err(th_err)?;
    Ok((PyComplexBall { inner: g2 }, PyComplexBall { inner: g3 }))
}

/// The three roots ``(e1, e2, e3)`` of ``4 x^3 - g2 x - g3``.
#[pyfunction]
#[pyo3(name = "weierstrass_roots", signature = (tau, prec=DEFAULT_PREC, accurate_to=None))]
fn py_weierstrass_roots(
    tau: &Bound<'_, PyAny>,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<Vec<PyComplexBall>> {
    let t = ball_from_any(tau, prec)?;
    Ok(weierstrass_roots(&t, precision(prec, accurate_to))
        .map_err(th_err)?
        .into_iter()
        .map(|inner| PyComplexBall { inner })
        .collect())
}

// ---------------------------------------------------------------------------
// Genus-g functions
// ---------------------------------------------------------------------------

/// All ``4^g`` values ``theta[a;b](z, tau)``.
///
/// ``z`` is a sequence of ``g`` points; ``tau`` a ``SiegelMatrix`` whose
/// imaginary part is provably positive definite (``E-THETA-007`` otherwise).
#[pyfunction]
#[pyo3(name = "riemann_theta", signature = (z, tau, prec=DEFAULT_PREC, accurate_to=None))]
fn py_riemann_theta(
    z: &Bound<'_, PyAny>,
    tau: &PySiegelMatrix,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<PyThetaValues> {
    let zv = ball_list(z, prec)?;
    Ok(PyThetaValues {
        inner: riemann_theta(&zv, &tau.inner, precision(prec, accurate_to)).map_err(th_err)?,
    })
}

/// All ``4^g`` values ``theta[a;b](z, tau)^2``, by FLINT's faster squared
/// algorithm. The result is flagged ``is_squared``.
#[pyfunction]
#[pyo3(name = "riemann_theta_squared", signature = (z, tau, prec=DEFAULT_PREC, accurate_to=None))]
fn py_riemann_theta_squared(
    z: &Bound<'_, PyAny>,
    tau: &PySiegelMatrix,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<PyThetaValues> {
    let zv = ball_list(z, prec)?;
    Ok(PyThetaValues {
        inner: riemann_theta_squared(&zv, &tau.inner, precision(prec, accurate_to))
            .map_err(th_err)?,
    })
}

/// A single value ``theta[a;b](z, tau)`` for the characteristic index ``ab``.
#[pyfunction]
#[pyo3(name = "riemann_theta_characteristic", signature = (z, tau, ab, prec=DEFAULT_PREC, accurate_to=None))]
fn py_riemann_theta_characteristic(
    z: &Bound<'_, PyAny>,
    tau: &PySiegelMatrix,
    ab: u64,
    prec: u32,
    accurate_to: Option<u32>,
) -> PyResult<PyComplexBall> {
    let zv = ball_list(z, prec)?;
    Ok(PyComplexBall {
        inner: riemann_theta_characteristic(&zv, &tau.inner, ab, precision(prec, accurate_to))
            .map_err(th_err)?,
    })
}

/// Pack a characteristic ``(a, b)`` given as two ``g``-long bit lists into the
/// index FLINT uses to order theta values.
///
/// ``a`` occupies the more significant half: in genus 2, ``a = (1, 0)``,
/// ``b = (0, 0)`` is index 8.
#[pyfunction]
#[pyo3(name = "theta_characteristic_index")]
fn py_theta_characteristic_index(a: Vec<u8>, b: Vec<u8>) -> PyResult<u64> {
    theta_characteristic_index(&a, &b).map_err(th_err)
}

/// Split a characteristic index into its ``(a, b)`` bit lists.
#[pyfunction]
#[pyo3(name = "theta_characteristic_bits")]
fn py_theta_characteristic_bits(ab: u64, genus: usize) -> PyResult<(Vec<u8>, Vec<u8>)> {
    theta_characteristic_bits(ab, genus).map_err(th_err)
}

/// Is the characteristic ``(a, b)`` even — i.e. is ``theta[a;b]`` an even
/// function of ``z``? The odd ones vanish at ``z = 0``.
#[pyfunction]
#[pyo3(name = "theta_characteristic_is_even")]
fn py_theta_characteristic_is_even(ab: u64, genus: usize) -> PyResult<bool> {
    theta_characteristic_is_even(ab, genus).map_err(th_err)
}

/// Move ``tau`` towards the fundamental domain of ``Sp_2g(Z)`` acting on
/// ``H_g``, returning the symplectic matrix and the transformed point.
///
/// FLINT falls back to the identity when ``tau`` is unreasonable, so an
/// identity result is "no reduction found", not "already reduced" — ask
/// ``siegel_is_reduced`` for that.
#[pyfunction]
#[pyo3(name = "siegel_reduce", signature = (tau, prec=DEFAULT_PREC))]
fn py_siegel_reduce(tau: &PySiegelMatrix, prec: u32) -> PyResult<PySiegelReduction> {
    Ok(PySiegelReduction {
        inner: siegel_reduce(&tau.inner, prec).map_err(th_err)?,
    })
}

/// Is ``tau`` **certainly** in the reduced domain with tolerance ``2^tol_exp``?
#[pyfunction]
#[pyo3(name = "siegel_is_reduced", signature = (tau, tol_exp=-10, prec=DEFAULT_PREC))]
fn py_siegel_is_reduced(tau: &PySiegelMatrix, tol_exp: i64, prec: u32) -> PyResult<bool> {
    siegel_is_reduced(&tau.inner, tol_exp, prec).map_err(th_err)
}

/// Is the genus-``g`` Riemann theta path (FLINT's ``acb_theta``) compiled in?
#[pyfunction]
#[pyo3(name = "riemann_theta_available")]
fn py_riemann_theta_available() -> bool {
    riemann_theta_available()
}

/// Is the Arb ball layer (FLINT >= 3.1) available in this build?
#[pyfunction]
#[pyo3(name = "arb_backend_available")]
fn py_arb_backend_available() -> bool {
    arb_backend_available()
}

/// Register the theta / modular surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyComplexBall>()?;
    m.add_class::<PySiegelMatrix>()?;
    m.add_class::<PyThetaValues>()?;
    m.add_class::<PySiegelReduction>()?;
    m.add_function(wrap_pyfunction!(py_dedekind_eta, m)?)?;
    m.add_function(wrap_pyfunction!(py_j_invariant, m)?)?;
    m.add_function(wrap_pyfunction!(py_modular_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(py_modular_discriminant, m)?)?;
    m.add_function(wrap_pyfunction!(py_eisenstein_series, m)?)?;
    m.add_function(wrap_pyfunction!(py_jacobi_theta, m)?)?;
    m.add_function(wrap_pyfunction!(py_jacobi_theta_null, m)?)?;
    m.add_function(wrap_pyfunction!(py_weierstrass_p, m)?)?;
    m.add_function(wrap_pyfunction!(py_weierstrass_p_prime, m)?)?;
    m.add_function(wrap_pyfunction!(py_weierstrass_zeta, m)?)?;
    m.add_function(wrap_pyfunction!(py_weierstrass_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(py_weierstrass_invariants, m)?)?;
    m.add_function(wrap_pyfunction!(py_weierstrass_roots, m)?)?;
    m.add_function(wrap_pyfunction!(py_riemann_theta, m)?)?;
    m.add_function(wrap_pyfunction!(py_riemann_theta_squared, m)?)?;
    m.add_function(wrap_pyfunction!(py_riemann_theta_characteristic, m)?)?;
    m.add_function(wrap_pyfunction!(py_theta_characteristic_index, m)?)?;
    m.add_function(wrap_pyfunction!(py_theta_characteristic_bits, m)?)?;
    m.add_function(wrap_pyfunction!(py_theta_characteristic_is_even, m)?)?;
    m.add_function(wrap_pyfunction!(py_siegel_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(py_siegel_is_reduced, m)?)?;
    m.add_function(wrap_pyfunction!(py_riemann_theta_available, m)?)?;
    m.add_function(wrap_pyfunction!(py_arb_backend_available, m)?)?;
    m.add("ThetaError", m.py().get_type_bound::<PyThetaError>())?;
    Ok(())
}
