//! PyO3 bindings for `alkahest_core::funcfield` — function fields of algebraic
//! curves: divisors, the divisor class group `Pic⁰`, and Riemann–Roch.
//!
//! The Python surface lives under `alkahest.experimental`, not
//! `alkahest.__all__`.  Everything it can refuse raises `FunctionFieldError`
//! with a stable `E-FFLD-*` code; see the module docs on the Rust side for the
//! exact boundary (imaginary hyperelliptic model, ℚ-rational places).
//!
//! # Exactness
//!
//! Coordinates and coefficients are **exact rationals**.  `int`, `str`
//! (`"3/2"`) and `fractions.Fraction` are accepted; `float` is **rejected**
//! rather than rounded, because a place that is silently off the curve is
//! precisely the failure mode this crate refuses by policy.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};
use rug::{Integer, Rational};

use alkahest_core::errors::AlkahestError as AlkahestErrorTrait;
use alkahest_core::funcfield::{
    riemann_roch, Divisor, DivisorClass, FunctionField, FunctionFieldElement, FunctionFieldError,
    Place, RiemannRochSpace,
};

pyo3::create_exception!(alkahest, PyFunctionFieldError, crate::PyAlkahestError);

/// Build a structured exception carrying `.code`, `.remediation` and `.span`.
///
/// Mirrors `lib.rs`'s `make_structured_err`; kept local so that adding this
/// module touches `lib.rs` in exactly two lines.
fn structured(e: &FunctionFieldError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyFunctionFieldError>();
        let msg = e.to_string();
        let code = e.code();
        let remediation = e.remediation().unwrap_or("");
        let full = if remediation.is_empty() {
            format!("[{code}] {msg}")
        } else {
            format!("[{code}] {msg}\nRemediation: {remediation}")
        };
        match exc_type.call1((full,)) {
            Ok(exc) => {
                exc.setattr("code", code).ok();
                exc.setattr("remediation", e.remediation()).ok();
                exc.setattr("span", e.span()).ok();
                PyErr::from_value_bound(exc)
            }
            Err(err) => err,
        }
    })
}

fn ff(e: FunctionFieldError) -> PyErr {
    structured(&e)
}

// ---------------------------------------------------------------------------
// Exact rational conversion
// ---------------------------------------------------------------------------

/// `int`, `str` or `fractions.Fraction` to an exact `Rational`.
///
/// A `float` is refused: `0.1` is not `1/10`, and a place that misses the
/// curve by a rounding error would be reported as "not on the curve" — or,
/// worse, land on it.
fn to_rational(obj: &Bound<'_, PyAny>) -> PyResult<Rational> {
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        return Err(PyTypeError::new_err(
            "function-field coordinates are exact: pass an int, a string like '3/2', or a \
             fractions.Fraction, not a float",
        ));
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(Rational::from(v));
    }
    let s = obj.str()?.to_string_lossy().into_owned();
    Rational::from_str(s.trim()).map_err(|_| {
        PyValueError::new_err(format!("could not read `{s}` as an exact rational number"))
    })
}

fn to_integer(obj: &Bound<'_, PyAny>) -> PyResult<Integer> {
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(Integer::from(v));
    }
    let s = obj.str()?.to_string_lossy().into_owned();
    Integer::from_str(s.trim())
        .map_err(|_| PyValueError::new_err(format!("could not read `{s}` as an integer")))
}

/// Exact rational back to Python: an `int` when integral, else a
/// `fractions.Fraction`.
fn rational_to_py(py: Python<'_>, r: &Rational) -> PyResult<PyObject> {
    if r.is_integer() {
        return integer_to_py(py, r.numer());
    }
    // `Fraction("9/4")` — the single-string form, because `Fraction(str, str)`
    // rejects its arguments and the integers may not fit a machine word.
    let fractions = PyModule::import_bound(py, "fractions")?;
    let frac = fractions.getattr("Fraction")?;
    Ok(frac.call1((r.to_string(),))?.into_py(py))
}

fn integer_to_py(py: Python<'_>, i: &Integer) -> PyResult<PyObject> {
    if let Some(v) = i.to_i64() {
        return Ok(v.into_py(py));
    }
    // Arbitrary precision: hand Python the decimal digits.
    Ok(py
        .eval_bound(&format!("int('{i}')"), None, None)?
        .into_py(py))
}

fn coeffs_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Rational>> {
    let mut out = Vec::new();
    for item in obj.iter()? {
        out.push(to_rational(&item?)?);
    }
    Ok(out)
}

fn coeffs_to_py(py: Python<'_>, cs: &[Rational]) -> PyResult<Vec<PyObject>> {
    cs.iter().map(|c| rational_to_py(py, c)).collect()
}

// ---------------------------------------------------------------------------
// FunctionField
// ---------------------------------------------------------------------------

/// An algebraic function field `ℚ(x)[y]/(f(x, y))`, normalised to `y² = a(x)`.
///
/// Construct from the `y`-coefficients of `f` (lowest power of `y` first, each
/// a list of `x`-coefficients lowest degree first), or with
/// :meth:`FunctionField.hyperelliptic` from `a(x)` alone.
#[pyclass(name = "FunctionField")]
#[derive(Clone)]
pub struct PyFunctionField {
    pub(crate) inner: FunctionField,
}

#[pymethods]
impl PyFunctionField {
    #[new]
    fn new(coeffs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut cs = Vec::new();
        for item in coeffs.iter()? {
            cs.push(coeffs_from_py(&item?)?);
        }
        Ok(PyFunctionField {
            inner: FunctionField::new(&cs).map_err(ff)?,
        })
    }

    /// `y² = a(x)` from the coefficients of `a`, lowest degree first.
    #[classmethod]
    fn hyperelliptic(_cls: &Bound<'_, PyType>, a: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyFunctionField {
            inner: FunctionField::hyperelliptic(&coeffs_from_py(a)?).map_err(ff)?,
        })
    }

    /// The geometric genus.  Available for every accepted model, odd or even
    /// degree — it does not depend on the model.
    #[getter]
    fn genus(&self) -> usize {
        self.inner.genus()
    }

    /// `True` for the imaginary (odd-degree) model — the one with divisors.
    #[getter]
    fn is_imaginary(&self) -> bool {
        self.inner.is_imaginary()
    }

    /// `deg a`: `2g+1` on the imaginary model, `2g+2` on the real one.
    #[getter]
    fn curve_degree(&self) -> usize {
        self.inner.curve_degree()
    }

    /// The coefficients of the normalised `a(x)`, lowest degree first.
    fn curve(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        coeffs_to_py(py, self.inner.curve())
    }

    /// How the input `y` was rewritten to reach the normalised model.
    ///
    /// **Places are always expressed in the normalised model.**  When this
    /// reports the identity, they mean what you wrote.
    fn normalisation(&self) -> String {
        self.inner.normalisation().to_string()
    }

    /// `True` when the normalisation is the identity.
    #[getter]
    fn is_normalised(&self) -> bool {
        self.inner.normalisation().is_identity()
    }

    /// `True` when `(x, y)` satisfies `y² = a(x)` in the normalised model.
    fn contains_point(&self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self
            .inner
            .contains_point(&to_rational(x)?, &to_rational(y)?))
    }

    /// `True` when the place above `x` is ramified, i.e. `a(x) = 0`.
    fn is_branch_point(&self, x: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.inner.is_branch_point(&to_rational(x)?))
    }

    /// A canonical divisor `K = (2g − 2)·∞`.  Imaginary model only.
    fn canonical_divisor(&self) -> PyResult<PyDivisor> {
        Ok(PyDivisor {
            inner: self.inner.canonical_divisor().map_err(ff)?,
        })
    }

    /// `True` when `divisor` is the divisor of a function.
    ///
    /// Exact — Cantor arithmetic over ℚ compared against the identity class.
    /// A divisor of non-zero degree raises `E-FFLD-005` rather than returning
    /// `False`: it is a malformed question, not a negative answer.
    fn is_principal(&self, divisor: &PyDivisor) -> PyResult<bool> {
        self.inner.is_principal(&divisor.inner).map_err(ff)
    }

    fn __eq__(&self, other: &PyFunctionField) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("FunctionField({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// Place
// ---------------------------------------------------------------------------

/// A degree-one place: a rational point `(x, y)`, or the place at infinity.
#[pyclass(name = "Place")]
#[derive(Clone)]
pub struct PyPlace {
    pub(crate) inner: Place,
}

#[pymethods]
impl PyPlace {
    /// A finite rational place `(x, y)`.  Validated against a curve only when
    /// it is put into a :class:`Divisor`.
    #[classmethod]
    fn finite(
        _cls: &Bound<'_, PyType>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(PyPlace {
            inner: Place::finite(to_rational(x)?, to_rational(y)?),
        })
    }

    /// The place at infinity of the imaginary model.
    #[classmethod]
    fn infinity(_cls: &Bound<'_, PyType>) -> Self {
        PyPlace {
            inner: Place::infinity(),
        }
    }

    /// `True` for the place at infinity.
    #[getter]
    fn is_infinite(&self) -> bool {
        self.inner.is_infinite()
    }

    /// `True` at a branch point, and at the place at infinity.
    #[getter]
    fn is_ramified(&self) -> bool {
        self.inner.is_ramified()
    }

    /// The residue degree.  Always 1 — nothing else is representable.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// The `x`-coordinate, or `None` at infinity.
    #[getter]
    fn x(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.inner.x().map(|v| rational_to_py(py, v)).transpose()
    }

    /// The `y`-coordinate, or `None` at infinity.
    #[getter]
    fn y(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.inner.y().map(|v| rational_to_py(py, v)).transpose()
    }

    /// The image under the hyperelliptic involution `(α, β) ↦ (α, −β)`.
    fn involution(&self) -> PyPlace {
        PyPlace {
            inner: self.inner.involution(),
        }
    }

    fn __eq__(&self, other: &PyPlace) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.inner.to_string().hash(&mut h);
        h.finish()
    }

    fn __repr__(&self) -> String {
        format!("Place({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// Divisor
// ---------------------------------------------------------------------------

/// A formal ℤ-combination of degree-one places.
///
/// Comparison (`<=`, `<`, …) is the **divisor partial order**: `D <= E` iff
/// `E - D` is effective.  Two divisors at different places are incomparable,
/// and then every one of `<`, `<=`, `>`, `>=` is `False` — which is what a
/// partial order means, not a bug.
#[pyclass(name = "Divisor")]
#[derive(Clone)]
pub struct PyDivisor {
    pub(crate) inner: Divisor,
}

#[pymethods]
impl PyDivisor {
    /// `Divisor(field, [(place, multiplicity), ...])`.
    #[new]
    #[pyo3(signature = (field, terms = None))]
    fn new(field: &PyFunctionField, terms: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut parsed: Vec<(Place, Integer)> = Vec::new();
        if let Some(terms) = terms {
            for item in terms.iter()? {
                let item = item?;
                let place: PyRef<PyPlace> = item.get_item(0)?.extract()?;
                let mult = to_integer(&item.get_item(1)?)?;
                parsed.push((place.inner.clone(), mult));
            }
        }
        Ok(PyDivisor {
            inner: Divisor::from_terms(field.inner.clone(), parsed).map_err(ff)?,
        })
    }

    /// The zero divisor.
    #[classmethod]
    fn zero(_cls: &Bound<'_, PyType>, field: &PyFunctionField) -> PyResult<Self> {
        Ok(PyDivisor {
            inner: Divisor::zero(field.inner.clone()).map_err(ff)?,
        })
    }

    /// The function field this divisor lives on.
    #[getter]
    fn field(&self) -> PyFunctionField {
        PyFunctionField {
            inner: self.inner.field().clone(),
        }
    }

    /// `deg D` — the sum of the multiplicities, every place having degree 1.
    #[getter]
    fn degree(&self, py: Python<'_>) -> PyResult<PyObject> {
        integer_to_py(py, &self.inner.degree())
    }

    /// `True` for the zero divisor.
    #[getter]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// `True` when every multiplicity is non-negative.
    #[getter]
    fn is_effective(&self) -> bool {
        self.inner.is_effective()
    }

    /// The number of places in the support.
    fn __len__(&self) -> usize {
        self.inner.support_len()
    }

    /// The places with non-zero multiplicity, in canonical order.
    fn support(&self) -> Vec<PyPlace> {
        self.inner
            .support()
            .into_iter()
            .map(|inner| PyPlace { inner })
            .collect()
    }

    /// The `(place, multiplicity)` pairs, in canonical order.
    fn terms(&self, py: Python<'_>) -> PyResult<Vec<(PyPlace, PyObject)>> {
        self.inner
            .terms()
            .into_iter()
            .map(|(p, c)| Ok((PyPlace { inner: p }, integer_to_py(py, &c)?)))
            .collect()
    }

    /// The multiplicity at `place`; zero when it is not in the support.
    fn coefficient(&self, py: Python<'_>, place: &PyPlace) -> PyResult<PyObject> {
        integer_to_py(py, &self.inner.coefficient(&place.inner))
    }

    /// The image under the hyperelliptic involution.
    fn involution(&self) -> PyDivisor {
        PyDivisor {
            inner: self.inner.involution(),
        }
    }

    /// The class of this divisor in `Pic⁰`.  Needs `deg D = 0`.
    fn divisor_class(&self) -> PyResult<PyDivisorClass> {
        Ok(PyDivisorClass {
            inner: DivisorClass::of(&self.inner).map_err(ff)?,
        })
    }

    /// `L(D)` — its dimension and an explicit basis.
    fn riemann_roch(&self) -> PyResult<PyRiemannRochSpace> {
        Ok(PyRiemannRochSpace {
            inner: riemann_roch(&self.inner).map_err(ff)?,
        })
    }

    fn __add__(&self, other: &PyDivisor) -> PyResult<PyDivisor> {
        Ok(PyDivisor {
            inner: self.inner.add(&other.inner).map_err(ff)?,
        })
    }

    fn __sub__(&self, other: &PyDivisor) -> PyResult<PyDivisor> {
        Ok(PyDivisor {
            inner: self.inner.sub(&other.inner).map_err(ff)?,
        })
    }

    fn __neg__(&self) -> PyDivisor {
        PyDivisor {
            inner: self.inner.neg(),
        }
    }

    fn __mul__(&self, k: &Bound<'_, PyAny>) -> PyResult<PyDivisor> {
        Ok(PyDivisor {
            inner: self.inner.scale(&to_integer(k)?),
        })
    }

    fn __rmul__(&self, k: &Bound<'_, PyAny>) -> PyResult<PyDivisor> {
        self.__mul__(k)
    }

    fn __eq__(&self, other: &PyDivisor) -> bool {
        self.inner == other.inner
    }

    fn __le__(&self, other: &PyDivisor) -> bool {
        matches!(
            self.inner.partial_cmp(&other.inner),
            Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
        )
    }

    fn __lt__(&self, other: &PyDivisor) -> bool {
        matches!(
            self.inner.partial_cmp(&other.inner),
            Some(std::cmp::Ordering::Less)
        )
    }

    fn __ge__(&self, other: &PyDivisor) -> bool {
        matches!(
            self.inner.partial_cmp(&other.inner),
            Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
        )
    }

    fn __gt__(&self, other: &PyDivisor) -> bool {
        matches!(
            self.inner.partial_cmp(&other.inner),
            Some(std::cmp::Ordering::Greater)
        )
    }

    fn __repr__(&self) -> String {
        format!("Divisor({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// DivisorClass
// ---------------------------------------------------------------------------

/// A class in `Pic⁰`, held as a reduced Mumford pair.
#[pyclass(name = "DivisorClass")]
#[derive(Clone)]
pub struct PyDivisorClass {
    pub(crate) inner: DivisorClass,
}

#[pymethods]
impl PyDivisorClass {
    /// The class of a degree-zero divisor.
    #[classmethod]
    fn of(_cls: &Bound<'_, PyType>, divisor: &PyDivisor) -> PyResult<Self> {
        Ok(PyDivisorClass {
            inner: DivisorClass::of(&divisor.inner).map_err(ff)?,
        })
    }

    /// The identity class.
    #[classmethod]
    fn identity(_cls: &Bound<'_, PyType>, field: &PyFunctionField) -> PyResult<Self> {
        Ok(PyDivisorClass {
            inner: DivisorClass::identity(field.inner.clone()).map_err(ff)?,
        })
    }

    /// The function field this class lives on.
    #[getter]
    fn field(&self) -> PyFunctionField {
        PyFunctionField {
            inner: self.inner.field().clone(),
        }
    }

    /// `True` for the identity — equivalently, the divisor is principal.
    #[getter]
    fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    /// `deg u`, the weight of the reduced representative.  At most `g`.
    #[getter]
    fn weight(&self) -> usize {
        self.inner.weight()
    }

    /// Mumford `u(X)` on the monicised model, lowest degree first.
    fn mumford_u(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        coeffs_to_py(py, self.inner.mumford_u())
    }

    /// Mumford `v(X)` on the monicised model, lowest degree first.
    fn mumford_v(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        coeffs_to_py(py, self.inner.mumford_v())
    }

    /// The reduced representative as an explicit divisor.
    ///
    /// Raises `E-FFLD-003` when the representative is supported at a place of
    /// degree ≥ 2; the class itself is still fine, and `mumford_u` always works.
    fn reduced_divisor(&self) -> PyResult<PyDivisor> {
        Ok(PyDivisor {
            inner: self.inner.reduced_divisor().map_err(ff)?,
        })
    }

    /// The order of the class.
    ///
    /// Raises `FunctionFieldError` with `E-FFLD-007` when the class has
    /// **infinite** order — a verdict — and `E-FFLD-006` when the order could
    /// not be decided, which is neither.  Branch on `.code`.
    fn order(&self) -> PyResult<u64> {
        self.inner.order().map_err(ff)
    }

    /// `True` when the class is torsion, `False` on the non-torsion verdict.
    /// An undecided order still raises.
    fn is_torsion(&self) -> PyResult<bool> {
        self.inner.is_torsion().map_err(ff)
    }

    fn __add__(&self, other: &PyDivisorClass) -> PyResult<PyDivisorClass> {
        Ok(PyDivisorClass {
            inner: self.inner.add(&other.inner).map_err(ff)?,
        })
    }

    fn __sub__(&self, other: &PyDivisorClass) -> PyResult<PyDivisorClass> {
        Ok(PyDivisorClass {
            inner: self.inner.add(&other.inner.neg()).map_err(ff)?,
        })
    }

    fn __neg__(&self) -> PyDivisorClass {
        PyDivisorClass {
            inner: self.inner.neg(),
        }
    }

    fn __mul__(&self, k: &Bound<'_, PyAny>) -> PyResult<PyDivisorClass> {
        Ok(PyDivisorClass {
            inner: self.inner.scalar_mul(&to_integer(k)?).map_err(ff)?,
        })
    }

    fn __rmul__(&self, k: &Bound<'_, PyAny>) -> PyResult<PyDivisorClass> {
        self.__mul__(k)
    }

    fn __eq__(&self, other: &PyDivisorClass) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("DivisorClass({})", self.inner)
    }
}

// ---------------------------------------------------------------------------
// FunctionFieldElement
// ---------------------------------------------------------------------------

/// `(p(x) + q(x)·y) / d(x)` — an element of the function field.
#[pyclass(name = "FunctionFieldElement")]
#[derive(Clone)]
pub struct PyFunctionFieldElement {
    pub(crate) inner: FunctionFieldElement,
}

#[pymethods]
impl PyFunctionFieldElement {
    /// `FunctionFieldElement(field, p, q=None, d=None)`, coefficient lists
    /// lowest degree first.
    #[new]
    #[pyo3(signature = (field, p, q = None, d = None))]
    fn new(
        field: &PyFunctionField,
        p: &Bound<'_, PyAny>,
        q: Option<&Bound<'_, PyAny>>,
        d: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let p = coeffs_from_py(p)?;
        let q = match q {
            Some(q) => coeffs_from_py(q)?,
            None => Vec::new(),
        };
        let d = match d {
            Some(d) => coeffs_from_py(d)?,
            None => vec![Rational::from(1)],
        };
        Ok(PyFunctionFieldElement {
            inner: FunctionFieldElement::new(field.inner.clone(), p, q, d).map_err(ff)?,
        })
    }

    /// The coordinate function `x`.
    #[classmethod]
    fn x(_cls: &Bound<'_, PyType>, field: &PyFunctionField) -> Self {
        PyFunctionFieldElement {
            inner: FunctionFieldElement::x(field.inner.clone()),
        }
    }

    /// The coordinate function `y`.
    #[classmethod]
    fn y(_cls: &Bound<'_, PyType>, field: &PyFunctionField) -> Self {
        PyFunctionFieldElement {
            inner: FunctionFieldElement::y(field.inner.clone()),
        }
    }

    /// The function field this element lives in.
    #[getter]
    fn field(&self) -> PyFunctionField {
        PyFunctionField {
            inner: self.inner.field().clone(),
        }
    }

    /// `True` for the zero function.
    #[getter]
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// The `y⁰` part of the numerator, lowest degree first.
    fn numerator_rational_part(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        coeffs_to_py(py, self.inner.numerator_rational_part())
    }

    /// The `y¹` part of the numerator, lowest degree first.
    fn numerator_algebraic_part(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        coeffs_to_py(py, self.inner.numerator_algebraic_part())
    }

    /// The denominator, lowest degree first.
    fn denominator(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        coeffs_to_py(py, self.inner.denominator())
    }

    /// The norm `p² − q²·a` of the numerator, down to `ℚ(x)`.
    fn numerator_norm(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        coeffs_to_py(py, &self.inner.numerator_norm())
    }

    /// `div(u)`.
    ///
    /// Raises `E-FFLD-003` when the divisor has a place of degree ≥ 2 — which
    /// is common, and is a refusal to misreport rather than a failure.
    fn divisor(&self) -> PyResult<PyDivisor> {
        Ok(PyDivisor {
            inner: self.inner.divisor().map_err(ff)?,
        })
    }

    fn __mul__(&self, other: &PyFunctionFieldElement) -> PyResult<PyFunctionFieldElement> {
        Ok(PyFunctionFieldElement {
            inner: self.inner.mul(&other.inner).map_err(ff)?,
        })
    }

    fn __eq__(&self, other: &PyFunctionFieldElement) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "FunctionFieldElement(p={:?}, q={:?}, d={:?})",
            self.inner
                .numerator_rational_part()
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>(),
            self.inner
                .numerator_algebraic_part()
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>(),
            self.inner
                .denominator()
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>(),
        )
    }
}

// ---------------------------------------------------------------------------
// RiemannRochSpace
// ---------------------------------------------------------------------------

/// The space `L(D)`: its dimension and an explicit basis.
#[pyclass(name = "RiemannRochSpace")]
#[derive(Clone)]
pub struct PyRiemannRochSpace {
    pub(crate) inner: RiemannRochSpace,
}

#[pymethods]
impl PyRiemannRochSpace {
    /// `dim L(D)`.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// A ℚ-basis of `L(D)`.  Every element satisfies `div(u) + D >= 0`.
    fn basis(&self) -> Vec<PyFunctionFieldElement> {
        self.inner
            .basis()
            .iter()
            .map(|u| PyFunctionFieldElement { inner: u.clone() })
            .collect()
    }

    /// The divisor this space was computed for.
    fn divisor(&self) -> PyDivisor {
        PyDivisor {
            inner: self.inner.divisor().clone(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.dimension()
    }

    fn __repr__(&self) -> String {
        format!(
            "RiemannRochSpace(dim={}, D={})",
            self.inner.dimension(),
            self.inner.divisor()
        )
    }
}

/// `L(D)` — dimension and basis.
#[pyfunction]
#[pyo3(name = "riemann_roch")]
fn py_riemann_roch(divisor: &PyDivisor) -> PyResult<PyRiemannRochSpace> {
    Ok(PyRiemannRochSpace {
        inner: riemann_roch(&divisor.inner).map_err(ff)?,
    })
}

/// Register the function-field surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFunctionField>()?;
    m.add_class::<PyPlace>()?;
    m.add_class::<PyDivisor>()?;
    m.add_class::<PyDivisorClass>()?;
    m.add_class::<PyFunctionFieldElement>()?;
    m.add_class::<PyRiemannRochSpace>()?;
    m.add_function(wrap_pyfunction!(py_riemann_roch, m)?)?;
    m.add(
        "FunctionFieldError",
        m.py().get_type_bound::<PyFunctionFieldError>(),
    )?;
    Ok(())
}
