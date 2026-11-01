//! PyO3 bindings for `alkahest_core::units` — physical units and dimensional
//! analysis (G10).
//!
//! The Python surface is `alkahest.Unit`, `alkahest.parse_unit`,
//! `alkahest.dimension_of` and the `alkahest.units` module (which builds the
//! `Quantity` class on top of these). Refusals arrive as `alkahest.UnitError`
//! with a stable `E-UNIT-NNN` `.code`. Every scale crosses the boundary as a
//! `fractions.Fraction` (times a power of π), never as a float.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

use alkahest_core::units::{self as core_units, Unit, UnitError};
use rug::Rational;

use crate::bigint::{rational_from_py, rational_to_py};

pyo3::create_exception!(alkahest, PyUnitError, crate::PyAlkahestError);

fn unit_err(e: UnitError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyUnitError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// A physical unit: a product of named units with rational exponents, an
/// exact scale relative to SI, and a dimension over the seven SI base
/// dimensions.
///
/// Build one from a string — ``Unit("mg*h/L")``, ``Unit("m/s^2")``,
/// ``Unit("kg m s^-2")``, ``Unit("m^(1/2)")``, ``Unit("µg/mL")`` — or by
/// combining units with ``*``, ``/`` and ``**`` (rational exponent).
///
/// Equality is *semantic*: ``Unit("N") == Unit("kg*m/s^2")`` is ``True``;
/// ``str(u)`` keeps the spelling. ``degC`` / ``degF`` are affine scales and
/// may only stand alone and be converted (``E-UNIT-003`` otherwise).
///
/// Raises ``UnitError`` (``E-UNIT-002``) on an unknown or malformed unit.
#[pyclass(name = "Unit", module = "alkahest", frozen)]
#[derive(Clone)]
pub(crate) struct PyUnit {
    pub(crate) inner: Unit,
}

/// Accept a `Unit` or a unit string.
pub(crate) fn unit_from_py(ob: &Bound<'_, PyAny>) -> PyResult<Option<Unit>> {
    if let Ok(u) = ob.downcast::<PyUnit>() {
        return Ok(Some(u.get().inner.clone()));
    }
    if let Ok(s) = ob.downcast::<PyString>() {
        // `to_cow`, not `to_str`: PyO3 0.21 has no `to_str` under the 3.9 limited API.
        return core_units::parse_unit(&s.to_cow()?)
            .map(Some)
            .map_err(unit_err);
    }
    Ok(None)
}

fn require_unit(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<Unit> {
    unit_from_py(ob)?.ok_or_else(|| {
        PyTypeError::new_err(format!(
            "{what} must be a Unit or a unit string, got {}",
            ob.get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_default()
        ))
    })
}

fn wrap(u: Unit) -> PyUnit {
    PyUnit { inner: u }
}

fn scale_tuple(py: Python<'_>, s: &core_units::Scale) -> PyResult<PyObject> {
    Ok(PyTuple::new_bound(
        py,
        [
            rational_to_py(py, &s.ratio)?,
            rational_to_py(py, &s.pi_power)?,
        ],
    )
    .into_py(py))
}

#[pymethods]
impl PyUnit {
    #[new]
    fn new(spec: &str) -> PyResult<Self> {
        core_units::parse_unit(spec).map(wrap).map_err(unit_err)
    }

    /// The unit of a pure number, written ``1``.
    #[staticmethod]
    fn dimensionless() -> Self {
        wrap(Unit::dimensionless())
    }

    /// The unit's spelling as a product, e.g. ``"mg*h/L"``.
    #[getter]
    fn name(&self) -> String {
        self.inner.name()
    }

    /// The dimension, e.g. ``"mass/length^3"`` or ``"dimensionless"``.
    #[getter]
    fn dimension(&self) -> String {
        self.inner.dimension().to_string()
    }

    /// ``{base_dimension: Fraction}`` for the non-zero exponents.
    #[getter]
    fn dimension_exponents(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        for (i, name) in core_units::BASE_DIMENSIONS.iter().enumerate() {
            let e = self.inner.dimension().exponent(i);
            if *e != 0 {
                d.set_item(*name, rational_to_py(py, e)?)?;
            }
        }
        Ok(d.into_py(py))
    }

    /// ``[(name, Fraction exponent), …]`` as written.
    #[getter]
    fn factors(&self, py: Python<'_>) -> PyResult<Vec<(String, PyObject)>> {
        self.inner
            .factors()
            .iter()
            .map(|(n, e)| Ok((n.clone(), rational_to_py(py, e)?)))
            .collect()
    }

    /// ``(ratio, pi_power)``: one of this unit is ``ratio · π**pi_power`` of the
    /// coherent SI unit of the same dimension (both ``Fraction``).
    #[getter]
    fn scale(&self, py: Python<'_>) -> PyResult<PyObject> {
        scale_tuple(py, self.inner.scale())
    }

    /// ``True`` for ``degC`` / ``degF``.
    #[getter]
    fn is_affine(&self) -> bool {
        self.inner.is_affine()
    }

    /// ``True`` when dimensionless with scale 1 (``1``, ``rad``, ``m/m``).
    #[getter]
    fn is_pure_number(&self) -> bool {
        self.inner.is_pure_number()
    }

    /// ``True`` when ``other`` has the same dimension.
    fn same_dimension(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let o = require_unit(other, "other")?;
        Ok(self.inner.dimension() == o.dimension())
    }

    /// Raise ``UnitError`` ``E-UNIT-001`` (naming both dimensions) unless
    /// ``other`` has the same dimension.
    #[pyo3(signature = (other, operation = "add"))]
    fn check_same_dimension(&self, other: &Bound<'_, PyAny>, operation: &str) -> PyResult<()> {
        let o = require_unit(other, "other")?;
        self.inner
            .check_same_dimension(&o, operation)
            .map_err(unit_err)
    }

    /// Raise ``UnitError`` ``E-UNIT-004`` unless this is a pure number;
    /// ``context`` names where the value is used (``"argument of exp"``).
    fn require_pure_number(&self, context: &str) -> PyResult<()> {
        if self.inner.is_pure_number() {
            return Ok(());
        }
        if self.inner.is_affine() {
            return Err(unit_err(UnitError::AffineUnit {
                unit: self.inner.name(),
                context: context.to_string(),
            }));
        }
        Err(unit_err(UnitError::NotDimensionless {
            context: context.to_string(),
            unit: self.inner.name(),
            dimension: self.inner.dimension().to_string(),
        }))
    }

    /// The exact map ``x [self] = ratio·π**pi_power·x + shift [target]`` as
    /// ``(ratio, pi_power, shift)`` (``Fraction``\ s; ``shift`` is ``0`` except
    /// between affine temperature scales). ``E-UNIT-001`` on a dimension
    /// mismatch.
    fn conversion(&self, py: Python<'_>, target: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let t = require_unit(target, "target")?;
        let (f, shift) = self.inner.conversion(&t).map_err(unit_err)?;
        Ok(PyTuple::new_bound(
            py,
            [
                rational_to_py(py, &f.ratio)?,
                rational_to_py(py, &f.pi_power)?,
                rational_to_py(py, &shift)?,
            ],
        )
        .into_py(py))
    }

    /// The coherent SI unit of the same dimension in base units
    /// (``Unit("J").to_base()`` is ``kg*m^2/s^2``).
    fn to_base(&self) -> Self {
        wrap(self.inner.to_base())
    }

    /// Merge factors of equal dimension (``mg/kg*g → mg``); returns
    /// ``(ratio, pi_power, unit)`` with ``x [self] = ratio·π**pi_power·x [unit]``.
    ///
    /// With ``derived=True`` the merged unit is also renamed to the named SI
    /// derived unit it exactly equals (``kg*m/s^2 → N``), see
    /// :meth:`to_derived`; ``allowed`` restricts the candidate names.
    #[pyo3(signature = (derived = false, allowed = None))]
    fn simplify(
        &self,
        py: Python<'_>,
        derived: bool,
        allowed: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let (f, u) = if derived {
            let a: Option<Vec<&str>> = allowed
                .as_ref()
                .map(|v| v.iter().map(String::as_str).collect());
            self.inner.simplify_derived(a.as_deref())
        } else {
            self.inner.simplify()
        };
        Ok(PyTuple::new_bound(
            py,
            [
                rational_to_py(py, &f.ratio)?,
                rational_to_py(py, &f.pi_power)?,
                Py::new(py, wrap(u))?.into_py(py),
            ],
        )
        .into_py(py))
    }

    /// The named SI derived unit this unit exactly equals, or ``None``.
    ///
    /// ``Unit("kg*m/s^2").to_derived()`` is ``N``; ``g*m/s^2`` is ``mN``
    /// (prefixes in steps of 1000 only); ``kg*cm/s^2`` is not renamed (its
    /// scale is not a power of 1000 times ``N``). The candidates are ``N, Pa,
    /// J, W, C, V, ohm, F, H, T, Wb, S, Hz`` (``allowed`` restricts them). A
    /// unit that is already one named factor, is dimensionless or affine, or
    /// contains a dimensionless factor (``rad/s`` is not ``Hz``) is left
    /// alone, and ``Gy``/``Sv``/``Bq`` are never produced (they share a
    /// dimension with ``m^2/s^2`` and ``Hz``). ``N*m`` is a ``J``.
    #[pyo3(signature = (allowed = None))]
    fn to_derived(&self, allowed: Option<Vec<String>>) -> Option<PyUnit> {
        let a: Option<Vec<&str>> = allowed
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        self.inner.to_derived(a.as_deref()).map(wrap)
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        match unit_from_py(other)? {
            Some(o) => Ok(Py::new(py, wrap(self.inner.mul(&o).map_err(unit_err)?))?.into_py(py)),
            None => Ok(py.NotImplemented()),
        }
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        match unit_from_py(other)? {
            Some(o) => Ok(Py::new(py, wrap(o.mul(&self.inner).map_err(unit_err)?))?.into_py(py)),
            None => Ok(py.NotImplemented()),
        }
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        match unit_from_py(other)? {
            Some(o) => Ok(Py::new(py, wrap(self.inner.div(&o).map_err(unit_err)?))?.into_py(py)),
            None => Ok(py.NotImplemented()),
        }
    }

    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let lhs = if other.extract::<i64>().ok() == Some(1) {
            Some(Unit::dimensionless())
        } else {
            unit_from_py(other)?
        };
        match lhs {
            Some(o) => Ok(Py::new(py, wrap(o.div(&self.inner).map_err(unit_err)?))?.into_py(py)),
            None => Ok(py.NotImplemented()),
        }
    }

    fn __pow__(
        &self,
        py: Python<'_>,
        exp: &Bound<'_, PyAny>,
        modulo: Option<PyObject>,
    ) -> PyResult<PyObject> {
        if modulo.is_some() {
            return Ok(py.NotImplemented());
        }
        match rational_from_py(exp)? {
            Some(q) => Ok(Py::new(py, wrap(self.inner.pow(&q).map_err(unit_err)?))?.into_py(py)),
            None => Err(PyTypeError::new_err(
                "a unit exponent must be an int or a fractions.Fraction (exact rational)",
            )),
        }
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<PyObject> {
        let Some(o) = (match other.downcast::<PyUnit>() {
            Ok(u) => Some(u.get().inner.clone()),
            Err(_) => None,
        }) else {
            return Ok(py.NotImplemented());
        };
        match op {
            CompareOp::Eq => Ok((self.inner == o).into_py(py)),
            CompareOp::Ne => Ok((self.inner != o).into_py(py)),
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __hash__(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.inner.dimension().hash(&mut h);
        self.inner.scale().hash(&mut h);
        self.inner.offset().hash(&mut h);
        h.finish()
    }

    fn __repr__(&self) -> String {
        format!("Unit('{}')", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.inner.name()
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.get_type_bound::<PyUnit>();
        Ok((cls, (self.inner.name(),)).into_py(py))
    }
}

/// Parse a unit string into a :class:`Unit`.
///
/// Accepts products and quotients (``"mg*h/L"``, ``"kg m s^-2"``,
/// ``"kg*m^2/(s^2*A)"``), exponents with ``^`` or ``**`` (fractional ones in
/// parentheses, ``"m^(1/2)"``) or Unicode superscripts (``"m s⁻²"``), SI
/// prefixes from yocto to yotta (``µ``, ``μ`` and ``u`` all mean micro), and
/// exact numeric factors (``"1000*m"``). ``/`` divides by the single factor that
/// follows it.
///
/// Raises ``UnitError``: ``E-UNIT-002`` for a malformed string or an unknown
/// name (``"m2"`` is refused, not read as ``m^2``), ``E-UNIT-003`` for an
/// affine temperature unit inside a compound unit (``"degC/s"``).
///
/// Example
/// -------
/// >>> parse_unit("mg*h/L").dimension
/// 'mass*time/length^3'
#[pyfunction]
#[pyo3(name = "parse_unit")]
fn py_parse_unit(spec: &str) -> PyResult<PyUnit> {
    core_units::parse_unit(spec).map(wrap).map_err(unit_err)
}

/// Every unit name the parser knows (without SI prefixes), sorted.
#[pyfunction]
#[pyo3(name = "known_units")]
fn py_known_units() -> Vec<&'static str> {
    core_units::known_units()
}

/// The unit of a symbolic formula, given the unit of each of its symbols.
///
/// Checks the **dimensional consistency** of ``expr`` and returns the unit
/// its value is in when every symbol is given in its listed unit.
///
/// Parameters
/// ----------
/// expr : Expr
///     The formula.
/// units : dict[Expr, Unit | str]
///     The unit of each symbol. Keys may be any sub-expression, e.g. an
///     applied undefined function ``C(t)``.
///
/// Returns
/// -------
/// Unit
///
/// Raises
/// ------
/// UnitError
///     ``E-UNIT-001`` the terms of a sum have different dimensions (both are
///     named); ``E-UNIT-004`` the argument of ``exp``/``log``/``sin``/… or a
///     non-literal exponent is not a pure number (an angle must be in ``rad``);
///     ``E-UNIT-005`` a symbol or function with no known unit; ``E-UNIT-006``
///     the terms of a sum share a dimension but not a unit (``h`` and ``min``);
///     ``E-UNIT-003`` an affine temperature inside arithmetic.
///
/// Example
/// -------
/// >>> p = ExprPool(); D, V = p.symbol("D"), p.symbol("V")
/// >>> dimension_of(D / V, {D: "mg", V: "L"})
/// Unit('mg/L')
#[pyfunction]
#[pyo3(name = "dimension_of")]
fn py_dimension_of(
    py: Python<'_>,
    expr: PyRef<'_, crate::PyExpr>,
    units: &Bound<'_, PyDict>,
) -> PyResult<PyUnit> {
    let mut map: HashMap<alkahest_core::ExprId, Unit> = HashMap::new();
    for (k, v) in units.iter() {
        let key = k.extract::<PyRef<'_, crate::PyExpr>>().map_err(|_| {
            PyTypeError::new_err("the keys of `units` must be Expr (symbols or sub-expressions)")
        })?;
        if !key.pool.is(&expr.pool) {
            return Err(PyTypeError::new_err(
                "a key of `units` belongs to a different ExprPool than `expr`",
            ));
        }
        map.insert(key.id, require_unit(&v, "each value of `units`")?);
    }
    let pool = expr.pool.borrow(py);
    alkahest_core::units::dimension_of(expr.id, &map, &pool.inner)
        .map(wrap)
        .map_err(unit_err)
}

/// Like :func:`dimension_of`, but a sum whose terms share a dimension and
/// differ in unit (``a`` in ``h`` plus ``b`` in ``min``) is **converted**
/// instead of refused with ``E-UNIT-006``.
///
/// Each such term is multiplied by the exact ratio of its unit to the target
/// unit chosen by ``prefer``: ``"finest"`` (default; the smallest unit among
/// the terms, so factors are integers when possible), ``"coarsest"`` or
/// ``"first"`` (the first term in canonical order). The same applies to
/// ``min``/``max``/``atan2`` arguments and ``Piecewise`` values.
///
/// Returns ``(expr, unit)``: the rewritten formula and the unit its value is in
/// when every symbol is in its listed unit. A formula without mixed sums comes
/// back unchanged. Every other refusal of :func:`dimension_of` stands
/// (``E-UNIT-001`` different dimensions, ``E-UNIT-003`` affine units in a sum,
/// ``E-UNIT-004``, ``E-UNIT-005``, ``E-UNIT-007``).
///
/// Example
/// -------
/// >>> p = ExprPool(); a, b = p.symbol("a"), p.symbol("b")
/// >>> unify_units(a + b, {a: "h", b: "min"})  # 60*a + b, in min
#[pyfunction]
#[pyo3(name = "unify_units", signature = (expr, units, prefer = "finest"))]
fn py_unify_units(
    py: Python<'_>,
    expr: PyRef<'_, crate::PyExpr>,
    units: &Bound<'_, PyDict>,
    prefer: &str,
) -> PyResult<(crate::PyExpr, PyUnit)> {
    let policy = match prefer {
        "finest" => core_units::UnifyPolicy::Finest,
        "coarsest" => core_units::UnifyPolicy::Coarsest,
        "first" => core_units::UnifyPolicy::First,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "prefer must be 'finest', 'coarsest' or 'first', got {other:?}"
            )))
        }
    };
    let mut map: HashMap<alkahest_core::ExprId, Unit> = HashMap::new();
    for (k, v) in units.iter() {
        let key = k.extract::<PyRef<'_, crate::PyExpr>>().map_err(|_| {
            PyTypeError::new_err("the keys of `units` must be Expr (symbols or sub-expressions)")
        })?;
        if !key.pool.is(&expr.pool) {
            return Err(PyTypeError::new_err(
                "a key of `units` belongs to a different ExprPool than `expr`",
            ));
        }
        map.insert(key.id, require_unit(&v, "each value of `units`")?);
    }
    let (id, unit) = {
        let pool = expr.pool.borrow(py);
        core_units::unify_units(expr.id, &map, &pool.inner, policy).map_err(unit_err)?
    };
    Ok((
        crate::PyExpr {
            id,
            pool: expr.pool.clone_ref(py),
        },
        wrap(unit),
    ))
}

/// ``ratio · π**pi_power · value + shift`` as an ``Expr`` (all three
/// coefficients exact rationals). Used by ``Quantity.convert``.
///
/// ``value`` is an ``Expr`` (its pool is used) or a number, which is placed in
/// the pool of ``like``.
#[pyfunction]
#[pyo3(name = "_unit_affine_map", signature = (value, ratio, pi_power, shift, like = None))]
fn py_unit_affine_map(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    ratio: &Bound<'_, PyAny>,
    pi_power: &Bound<'_, PyAny>,
    shift: &Bound<'_, PyAny>,
    like: Option<PyRef<'_, crate::PyExpr>>,
) -> PyResult<crate::PyExpr> {
    let pool_py = if let Ok(e) = value.extract::<PyRef<'_, crate::PyExpr>>() {
        e.pool.clone_ref(py)
    } else if let Some(l) = like.as_ref() {
        l.pool.clone_ref(py)
    } else {
        return Err(PyTypeError::new_err(
            "a numeric value needs `like` (an Expr whose pool to use)",
        ));
    };
    let value_id = crate::coerce_substituent(&pool_py, value, py)?;
    let as_rat = |ob: &Bound<'_, PyAny>| -> PyResult<Rational> {
        rational_from_py(ob)?.ok_or_else(|| PyTypeError::new_err("expected an exact rational"))
    };
    let s = core_units::Scale {
        ratio: as_rat(ratio)?,
        pi_power: as_rat(pi_power)?,
    };
    let shift = as_rat(shift)?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        let mut id = if s.is_one() {
            value_id
        } else {
            pool.mul(vec![s.to_expr(pool), value_id])
        };
        if shift != 0 {
            let sh = if *shift.denom() == 1 {
                pool.integer(shift.numer().clone())
            } else {
                pool.rational(shift.numer().clone(), shift.denom().clone())
            };
            id = pool.add(vec![id, sh]);
        }
        id
    };
    Ok(crate::PyExpr { id, pool: pool_py })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUnit>()?;
    m.add("UnitError", m.py().get_type_bound::<PyUnitError>())?;
    m.add_function(wrap_pyfunction!(py_parse_unit, m)?)?;
    m.add_function(wrap_pyfunction!(py_known_units, m)?)?;
    m.add_function(wrap_pyfunction!(py_dimension_of, m)?)?;
    m.add_function(wrap_pyfunction!(py_unify_units, m)?)?;
    m.add_function(wrap_pyfunction!(py_unit_affine_map, m)?)?;
    Ok(())
}
