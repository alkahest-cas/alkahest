//! Python `int` ↔ `rug::Integer` conversion by bytes, never by decimal text.
//!
//! Every big integer that crosses the PyO3 boundary goes through the helpers
//! in this module.  They used to go through `str(n)` → `Integer::parse` one
//! way and `int(z.to_string())` the other, which had two problems:
//!
//! * **It was broken above 4300 digits.**  CPython ≥ 3.11 (and 3.10.7+) caps
//!   int ↔ decimal-string conversion at `sys.get_int_max_str_digits()` digits
//!   (default 4300), so `ExprPool().integer(10**5000 + 7)` raised
//!   `ValueError: Exceeds the limit (4300) for integer string conversion`, and
//!   so did every result that came back as a big `int` or `Fraction`.
//! * **It was quadratic.**  CPython's int ↔ str conversion is `O(n²)`; at one
//!   million bits `pool.integer(n)` spent ~1 s formatting a string that GMP
//!   then parsed again.
//!
//! `int.to_bytes` / `int.from_bytes` are linear, are not subject to the digit
//! limit, and map directly onto `rug::Integer::{from_digits, to_digits}` with
//! least-significant-first byte order.  Values that fit an `i64` keep PyO3's
//! native fast path.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};
use rug::integer::Order;
use rug::{Integer, Rational};

/// A Python `int` (or anything implementing `__index__`) as an exact
/// `rug::Integer`, at any size.
///
/// Raises `TypeError` for objects that are not integers by protocol.
pub(crate) fn int_from_py(ob: &Bound<'_, PyAny>) -> PyResult<Integer> {
    if let Ok(v) = ob.extract::<i64>() {
        return Ok(Integer::from(v));
    }
    let owned;
    let n = if ob.is_instance_of::<PyInt>() {
        ob
    } else {
        owned = ob.call_method0("__index__").map_err(|_| {
            PyTypeError::new_err(format!(
                "expected an int, got {}",
                ob.get_type()
                    .name()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|_| "<unknown>".into())
            ))
        })?;
        if let Ok(v) = owned.extract::<i64>() {
            return Ok(Integer::from(v));
        }
        &owned
    };
    if let Ok(v) = n.extract::<u64>() {
        return Ok(Integer::from(v));
    }
    let negative = n.lt(0i64)?;
    let magnitude = if negative {
        n.call_method0("__neg__")?
    } else {
        n.clone()
    };
    let bits: u64 = magnitude.call_method0("bit_length")?.extract()?;
    let nbytes = bits.div_ceil(8);
    let raw = magnitude.call_method1("to_bytes", (nbytes, "little"))?;
    let bytes = raw.downcast::<PyBytes>()?.as_bytes();
    let z = Integer::from_digits(bytes, Order::Lsf);
    Ok(if negative { -z } else { z })
}

/// An exact `rug::Integer` as a Python `int`, at any size.
pub(crate) fn int_to_py(py: Python<'_>, z: &Integer) -> PyResult<PyObject> {
    if let Some(v) = z.to_i64() {
        return Ok(v.into_py(py));
    }
    if let Some(v) = z.to_u64() {
        return Ok(v.into_py(py));
    }
    // `to_digits` ignores the sign: these are the magnitude's bytes.
    let digits = z.to_digits::<u8>(Order::Lsf);
    let bytes = PyBytes::new_bound(py, &digits);
    let magnitude = py
        .get_type_bound::<PyInt>()
        .call_method1("from_bytes", (bytes, "little"))?;
    Ok(if z.is_negative() {
        magnitude.call_method0("__neg__")?.unbind()
    } else {
        magnitude.unbind()
    })
}

/// An exact rational as a `fractions.Fraction` (always, even when integral).
pub(crate) fn fraction_to_py(
    py: Python<'_>,
    numer: &Integer,
    denom: &Integer,
) -> PyResult<PyObject> {
    let n = int_to_py(py, numer)?;
    let d = int_to_py(py, denom)?;
    Ok(py
        .import_bound("fractions")?
        .getattr("Fraction")?
        .call1((n, d))?
        .unbind())
}

/// An exact rational as a Python `int` when integral, else a
/// `fractions.Fraction`.
pub(crate) fn rational_to_py(py: Python<'_>, r: &Rational) -> PyResult<PyObject> {
    if *r.denom() == 1 {
        return int_to_py(py, r.numer());
    }
    fraction_to_py(py, r.numer(), r.denom())
}

/// The decimal text of a Python integer, for core entry points that take
/// decimal strings.  A Python `int` is converted by bytes and formatted by GMP
/// (no digit limit, sub-quadratic); anything else falls back to `str(ob)`, so
/// callers that also accept `"123"` or `Fraction` text keep doing so.
pub(crate) fn decimal_of(ob: &Bound<'_, PyAny>) -> PyResult<String> {
    if ob.is_instance_of::<PyInt>() {
        return Ok(int_from_py(ob)?.to_string());
    }
    Ok(ob.str()?.to_string_lossy().into_owned())
}

/// A Python `int` or `numbers.Rational` (e.g. `fractions.Fraction`) as an
/// exact `rug::Rational`, or `None` when *ob* is neither (callers then fall
/// back to their own parsing, e.g. of `"3/2"` text).  A zero denominator is
/// also `None`, leaving the caller's error path to name it.
pub(crate) fn rational_from_py(ob: &Bound<'_, PyAny>) -> PyResult<Option<Rational>> {
    if ob.is_instance_of::<PyInt>() {
        return Ok(Some(Rational::from(int_from_py(ob)?)));
    }
    let (Ok(n), Ok(d)) = (ob.getattr("numerator"), ob.getattr("denominator")) else {
        return Ok(None);
    };
    if !n.is_instance_of::<PyInt>() || !d.is_instance_of::<PyInt>() {
        return Ok(None);
    }
    let (n, d) = (int_from_py(&n)?, int_from_py(&d)?);
    if d == 0 {
        return Ok(None);
    }
    Ok(Some(Rational::from((n, d))))
}

/// The exact text of a Python number for core entry points that parse
/// `"p"` / `"p/q"` strings: ints and `Fraction`s are rendered by GMP (no digit
/// limit); anything else is `str(ob)`, so floats etc. still reach the core's
/// own "not an exact rational" error with their text.
pub(crate) fn rational_text_of(ob: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Some(r) = rational_from_py(ob)? {
        return Ok(r.to_string());
    }
    Ok(ob.str()?.to_string_lossy().into_owned())
}

/// A decimal integer string (as returned by core entry points) as a Python
/// `int`, parsed by GMP rather than by `int(str)` — so it is not subject to
/// `sys.get_int_max_str_digits()`.
pub(crate) fn int_from_decimal(py: Python<'_>, decimal: &str) -> PyResult<PyObject> {
    let z = Integer::from(Integer::parse(decimal.trim()).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid decimal integer: {decimal:.64}"))
    })?);
    int_to_py(py, &z)
}

/// `_decimal_to_int(s)` — parse a decimal integer string into an `int` with
/// no `sys.get_int_max_str_digits()` limit.  Internal: the Python wrappers use
/// it on the decimal strings some binding functions return.
#[pyfunction]
#[pyo3(name = "_decimal_to_int")]
fn py_decimal_to_int(py: Python<'_>, s: &str) -> PyResult<PyObject> {
    int_from_decimal(py, s)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_decimal_to_int, m)?)?;
    Ok(())
}
