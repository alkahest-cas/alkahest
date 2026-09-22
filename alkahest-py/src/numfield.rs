//! PyO3 bindings for `alkahest_core::numfield` and for the classical
//! arithmetic functions added to `alkahest_core::number_theory`.
//!
//! The Python surface is `alkahest.experimental.NumberField` /
//! `NumberFieldElement`, plus the free functions `partition_number`,
//! `bernoulli_number`, `euler_number`, `harmonic_number`, the Stirling
//! numbers, `moebius_mu`, `divisor_sigma` and `sum_of_squares`. Refusals
//! arrive as `alkahest.experimental.NumberFieldError` (stable `E-NUMF-NNN`
//! `.code`) or `alkahest.number_theory.NumberTheoryError` (`E-NT-NNN`).
//!
//! # How a number crosses the boundary
//!
//! Integers go out as Python `int` — arbitrary precision, never truncated.
//! Rationals go out as `fractions.Fraction`, which is the exact type for them
//! and the one that keeps `norm(a) * norm(b) == norm(a*b)` true in Python.
//! Coming *in*, a coefficient may be an `int`, a `str` such as `"-1/2"`, or a
//! `Fraction`: each is read through its decimal text, so a 300-digit
//! coefficient survives the crossing where any fixed-width conversion would
//! not. A `float` is refused (`E-NUMF-004`) rather than rounded into a
//! rational nobody asked for.

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};

use alkahest_core::experimental::{
    cyclotomic_polynomial, NumberField, NumberFieldElement, NumberFieldError,
};
use alkahest_core::number_theory::{
    bernoulli_number, divisor_sigma, euler_number, harmonic_number, moebius_mu, partition_number,
    stirling_first, stirling_first_unsigned, stirling_second, sum_of_squares, NumberTheoryError,
};

pyo3::create_exception!(alkahest, PyNumberFieldError, crate::PyAlkahestError);

fn nf_err(e: NumberFieldError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyNumberFieldError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

fn nt_err(e: NumberTheoryError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<crate::PyNumberTheoryError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// A decimal string into a Python `int`, with no width limit anywhere on the
/// path.
fn py_int(py: Python<'_>, decimal: &str) -> PyResult<PyObject> {
    Ok(py
        .import_bound("builtins")?
        .getattr("int")?
        .call1((decimal,))?
        .into())
}

/// A `rug::Rational` into a `fractions.Fraction`, via its `"p/q"` text.
fn py_fraction(py: Python<'_>, r: &rug::Rational) -> PyResult<PyObject> {
    Ok(py
        .import_bound("fractions")?
        .getattr("Fraction")?
        .call1((r.to_string(),))?
        .into())
}

/// Read a sequence of coefficients as decimal text.
///
/// `int`, `str` and `Fraction` all render to something
/// `Rational::from_str_radix` accepts; a `float` renders to `"0.5"`, which does
/// not, and so becomes `E-NUMF-004` naming the value rather than a silent
/// binary-to-rational conversion.
fn coeff_strings(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for item in obj.iter()? {
        out.push(item?.str()?.to_string_lossy().into_owned());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// NumberField
// ---------------------------------------------------------------------------

/// An algebraic number field ``Q[x]/(f)``.
///
/// ``NumberField(coeffs)`` takes the coefficients of ``f`` **ascending** in
/// degree, so ``x**2 - 2`` is ``[-2, 0, 1]``. Each coefficient may be an
/// ``int``, a ``str`` like ``"-1/2"``, or a ``fractions.Fraction``.
///
/// The polynomial is **checked for irreducibility over Q**, not assumed to be
/// irreducible: ``NumberField([-1, 0, 1])`` raises ``E-NUMF-003`` naming a
/// proper factor, because ``Q[x]/(x**2 - 1)`` is a ring with zero divisors in
/// which ``inverse`` would have no answer for some non-zero elements.
///
/// ``NumberField.cyclotomic(n)`` is ``Q(zeta_n)``, of degree ``phi(n)``.
#[pyclass(name = "NumberField", module = "alkahest")]
#[derive(Clone)]
pub struct PyNumberField {
    inner: NumberField,
}

#[pymethods]
impl PyNumberField {
    #[new]
    fn new(coeffs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let strings = coeff_strings(coeffs)?;
        NumberField::from_strings(&strings)
            .map(|inner| Self { inner })
            .map_err(nf_err)
    }

    /// The cyclotomic field ``Q(zeta_n)``, defined by ``Phi_n`` and of degree
    /// ``phi(n)``.
    ///
    /// ``Phi_n`` is irreducible over ``Q`` as a theorem, so this constructor
    /// skips the factorisation the general one runs — which is what makes
    /// ``NumberField.cyclotomic(2048)`` (degree 1024, the ring-LWE shape)
    /// cheap to build.
    #[classmethod]
    fn cyclotomic(_cls: &Bound<'_, PyType>, n: u64) -> PyResult<Self> {
        NumberField::cyclotomic(n)
            .map(|inner| Self { inner })
            .map_err(nf_err)
    }

    /// The degree ``[K : Q]``.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// The canonical defining polynomial: primitive, integral, ascending in
    /// degree, positive leading coefficient.
    #[getter]
    fn defining_polynomial(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .defining_polynomial()
            .iter()
            .map(|c| py_int(py, &c.to_string()))
            .collect()
    }

    /// The discriminant of the **defining polynomial**.
    ///
    /// This is *not* the discriminant of the field. For a monic integral ``f``
    /// with root ``a``, ``disc(f) = [O_K : Z[a]]**2 * d_K``, so they agree only
    /// when ``Z[a]`` is the full ring of integers: ``Q[x]/(x**2 - 5)`` has
    /// ``disc(f) = 20`` and ``d_K = 5``. The ring of integers is not computed
    /// here, and there is deliberately no ``field_discriminant`` to confuse
    /// this with.
    #[getter]
    fn polynomial_discriminant(&self, py: Python<'_>) -> PyResult<PyObject> {
        py_int(py, &self.inner.polynomial_discriminant().to_string())
    }

    /// ``n`` if this field was built as ``Q(zeta_n)``, else ``None``.
    #[getter]
    fn cyclotomic_order(&self) -> Option<u64> {
        self.inner.cyclotomic_order()
    }

    /// The name of the generator used when rendering elements.
    #[getter]
    fn variable(&self) -> String {
        self.inner.variable().to_string()
    }

    /// A copy of this field that renders elements with a different generator
    /// name. Purely cosmetic; the two compare equal.
    fn with_variable(&self, var: &str) -> Self {
        Self {
            inner: self.inner.with_variable(var),
        }
    }

    /// The generator ``a``, a root of the defining polynomial.
    fn generator(&self) -> PyNumberFieldElement {
        PyNumberFieldElement {
            inner: self.inner.generator(),
        }
    }

    /// The additive identity.
    fn zero(&self) -> PyNumberFieldElement {
        PyNumberFieldElement {
            inner: self.inner.zero(),
        }
    }

    /// The multiplicative identity.
    fn one(&self) -> PyNumberFieldElement {
        PyNumberFieldElement {
            inner: self.inner.one(),
        }
    }

    /// An element from its coordinates in the power basis ``1, a, a**2, ...``,
    /// ascending, with at most ``degree`` of them.
    ///
    /// More than ``degree`` coordinates raises ``E-NUMF-007``: they are not
    /// silently reduced modulo the defining polynomial, because a caller who
    /// passes four coordinates to a cubic field has made a mistake.
    fn element(&self, coeffs: &Bound<'_, PyAny>) -> PyResult<PyNumberFieldElement> {
        let strings = coeff_strings(coeffs)?;
        self.inner
            .element_from_strings(&strings)
            .map(|inner| PyNumberFieldElement { inner })
            .map_err(nf_err)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "NumberField({}, degree={})",
            self.inner.defining_polynomial_string("x"),
            self.inner.degree()
        )
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// NumberFieldElement
// ---------------------------------------------------------------------------

/// An element of a :class:`NumberField`, in the power basis ``1, a, a**2, …``.
///
/// ``+``, ``-``, ``*``, ``/`` and ``**`` are defined. Operands from two
/// different fields raise ``E-NUMF-006`` rather than being coerced: two fields
/// with different canonical defining polynomials are not interchangeable even
/// when they are isomorphic, because an element's coordinates mean different
/// things in each.
#[pyclass(name = "NumberFieldElement", module = "alkahest")]
#[derive(Clone)]
pub struct PyNumberFieldElement {
    inner: NumberFieldElement,
}

#[pymethods]
impl PyNumberFieldElement {
    /// The field this element lives in.
    #[getter]
    fn field(&self) -> PyNumberField {
        PyNumberField {
            inner: self.inner.field().clone(),
        }
    }

    /// Coordinates in the power basis, ascending, always exactly ``degree`` of
    /// them, as ``fractions.Fraction``.
    fn coefficients(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .coefficients()
            .iter()
            .map(|c| py_fraction(py, c))
            .collect()
    }

    /// The field norm ``N(self)`` — the product of the conjugates.
    fn norm(&self, py: Python<'_>) -> PyResult<PyObject> {
        py_fraction(py, &self.inner.norm())
    }

    /// The field trace ``Tr(self)`` — the sum of the conjugates.
    fn trace(&self, py: Python<'_>) -> PyResult<PyObject> {
        py_fraction(py, &self.inner.trace())
    }

    /// The monic minimal polynomial over ``Q``, ascending in degree, with a
    /// trailing ``1``. Its degree divides ``[K : Q]``.
    fn minimal_polynomial(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .minimal_polynomial()
            .iter()
            .map(|c| py_fraction(py, c))
            .collect()
    }

    /// ``self ** -1``. Raises ``E-NUMF-005`` for zero.
    fn inverse(&self) -> PyResult<Self> {
        self.inner
            .inverse()
            .map(|inner| Self { inner })
            .map_err(nf_err)
    }

    /// True for the additive identity.
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// True for the multiplicative identity.
    fn is_one(&self) -> bool {
        self.inner.is_one()
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .add(&other.inner)
            .map(|inner| Self { inner })
            .map_err(nf_err)
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .sub(&other.inner)
            .map(|inner| Self { inner })
            .map_err(nf_err)
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .mul(&other.inner)
            .map(|inner| Self { inner })
            .map_err(nf_err)
    }

    fn __truediv__(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .div(&other.inner)
            .map(|inner| Self { inner })
            .map_err(nf_err)
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: self.inner.neg(),
        }
    }

    fn __pow__(&self, exp: i64, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "pow() with a modulus is not defined for number-field elements",
            ));
        }
        if exp < 0 {
            // A negative power is the inverse of a positive one, and the
            // inverse of zero is the one thing a field cannot give.
            let inv = self.inner.inverse().map_err(nf_err)?;
            return Ok(Self {
                inner: inv.pow(exp.unsigned_abs()),
            });
        }
        Ok(Self {
            inner: self.inner.pow(exp as u64),
        })
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("NumberFieldElement({})", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// ---------------------------------------------------------------------------
// Arithmetic functions
// ---------------------------------------------------------------------------

/// The cyclotomic polynomial ``Phi_n``, ascending in degree, as ``int``
/// coefficients. Its degree is ``phi(n)``.
///
/// ``Phi_1 = x - 1``, ``Phi_2 = x + 1``, ``Phi_6 = x**2 - x + 1``.
#[pyfunction]
#[pyo3(name = "cyclotomic_polynomial_coeffs")]
fn py_cyclotomic_polynomial_coeffs(py: Python<'_>, n: u64) -> PyResult<Vec<PyObject>> {
    cyclotomic_polynomial(n)
        .iter()
        .map(|c| py_int(py, &c.to_string()))
        .collect()
}

/// The partition function ``p(n)`` — exact, arbitrary precision.
///
/// ``p(100) == 190569292``. Raises ``E-NT-006`` past the work cap.
#[pyfunction]
#[pyo3(name = "partition_number")]
fn py_partition_number(py: Python<'_>, n: u64) -> PyResult<PyObject> {
    py_int(py, &partition_number(n).map_err(nt_err)?)
}

/// The Bernoulli number ``B_n`` as an exact ``fractions.Fraction``.
///
/// **Convention: ``B_1 == Fraction(-1, 2)``** — the "first Bernoulli numbers"
/// of DLMF, Mathematica's ``BernoulliB`` and SymPy's ``bernoulli``, from the
/// generating function ``t / (exp(t) - 1)``. The other convention, ``B_1 =
/// +1/2``, differs in *exactly this one value* — every other Bernoulli number
/// is identical — which is what makes picking the wrong one so quiet a bug.
/// Negate ``B_1`` if you want ``B_n^+``.
#[pyfunction]
#[pyo3(name = "bernoulli_number")]
fn py_bernoulli_number(py: Python<'_>, n: u64) -> PyResult<PyObject> {
    py_fraction(py, &bernoulli_number(n).map_err(nt_err)?)
}

/// The Euler number ``E_n`` (the secant numbers ``1, 0, -1, 0, 5, 0, -61, …``).
///
/// These are *not* the Eulerian numbers.
#[pyfunction]
#[pyo3(name = "euler_number")]
fn py_euler_number(py: Python<'_>, n: u64) -> PyResult<PyObject> {
    py_int(py, &euler_number(n).map_err(nt_err)?)
}

/// The harmonic number ``H_n = 1 + 1/2 + … + 1/n`` as an exact ``Fraction``.
#[pyfunction]
#[pyo3(name = "harmonic_number")]
fn py_harmonic_number(py: Python<'_>, n: u64) -> PyResult<PyObject> {
    py_fraction(py, &harmonic_number(n).map_err(nt_err)?)
}

/// The **signed** Stirling number of the first kind ``s(n, k)``.
///
/// The coefficients of the falling factorial: ``s(4, 2) == 11``,
/// ``s(3, 2) == -3``. See :func:`stirling_first_unsigned` for the cycle counts
/// ``|s(n, k)|`` — the two differ by ``(-1)**(n-k)``, so confusing them is a
/// sign error that hides whenever ``n - k`` is even.
#[pyfunction]
#[pyo3(name = "stirling_first")]
fn py_stirling_first(py: Python<'_>, n: u64, k: u64) -> PyResult<PyObject> {
    py_int(py, &stirling_first(n, k).map_err(nt_err)?)
}

/// The **unsigned** Stirling number of the first kind — the number of
/// permutations of ``n`` elements with exactly ``k`` cycles.
#[pyfunction]
#[pyo3(name = "stirling_first_unsigned")]
fn py_stirling_first_unsigned(py: Python<'_>, n: u64, k: u64) -> PyResult<PyObject> {
    py_int(py, &stirling_first_unsigned(n, k).map_err(nt_err)?)
}

/// The Stirling number of the second kind ``S(n, k)`` — partitions of ``n``
/// labelled elements into ``k`` non-empty blocks. ``S(4, 2) == 7``.
#[pyfunction]
#[pyo3(name = "stirling_second")]
fn py_stirling_second(py: Python<'_>, n: u64, k: u64) -> PyResult<PyObject> {
    py_int(py, &stirling_second(n, k).map_err(nt_err)?)
}

/// The Moebius function ``mu(n)`` for ``n >= 1``, as ``-1``, ``0`` or ``1``.
#[pyfunction]
#[pyo3(name = "moebius_mu")]
fn py_moebius_mu(n: &Bound<'_, PyAny>) -> PyResult<i32> {
    let decimal = n.str()?.to_string_lossy().into_owned();
    moebius_mu(&decimal).map_err(nt_err)
}

/// The divisor sum ``sigma_k(n) = sum of d**k over the divisors d of n``.
///
/// ``divisor_sigma(0, 12) == 6`` counts them; ``divisor_sigma(1, 12) == 28``
/// sums them.
#[pyfunction]
#[pyo3(name = "divisor_sigma")]
fn py_divisor_sigma(py: Python<'_>, k: u64, n: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let decimal = n.str()?.to_string_lossy().into_owned();
    py_int(py, &divisor_sigma(k, &decimal).map_err(nt_err)?)
}

/// ``r_k(n)``, the number of representations of ``n`` as an **ordered** sum of
/// ``k`` integer squares, counting signs. ``r_2(5) == 8``.
#[pyfunction]
#[pyo3(name = "sum_of_squares")]
fn py_sum_of_squares(py: Python<'_>, k: u64, n: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let decimal = n.str()?.to_string_lossy().into_owned();
    py_int(py, &sum_of_squares(k, &decimal).map_err(nt_err)?)
}

/// Register the number-field and arithmetic-function surface on the
/// `alkahest` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNumberField>()?;
    m.add_class::<PyNumberFieldElement>()?;
    m.add(
        "NumberFieldError",
        m.py().get_type_bound::<PyNumberFieldError>(),
    )?;
    m.add_function(wrap_pyfunction!(py_cyclotomic_polynomial_coeffs, m)?)?;
    m.add_function(wrap_pyfunction!(py_partition_number, m)?)?;
    m.add_function(wrap_pyfunction!(py_bernoulli_number, m)?)?;
    m.add_function(wrap_pyfunction!(py_euler_number, m)?)?;
    m.add_function(wrap_pyfunction!(py_harmonic_number, m)?)?;
    m.add_function(wrap_pyfunction!(py_stirling_first, m)?)?;
    m.add_function(wrap_pyfunction!(py_stirling_first_unsigned, m)?)?;
    m.add_function(wrap_pyfunction!(py_stirling_second, m)?)?;
    m.add_function(wrap_pyfunction!(py_moebius_mu, m)?)?;
    m.add_function(wrap_pyfunction!(py_divisor_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(py_sum_of_squares, m)?)?;
    Ok(())
}
