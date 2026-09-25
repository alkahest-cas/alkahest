//! PyO3 bindings for `alkahest_core::lattice` — the lattice toolkit: standard
//! lattices, determinants and duals, exact shortest/closest vectors, theta
//! series and sphere-packing densities.
//!
//! The Python surface lives under `alkahest.experimental`, not
//! `alkahest.__all__`. Everything it can refuse raises `LatticeError` with a
//! stable `E-LAT-*` code.
//!
//! # Exactness
//!
//! Gram entries, basis entries and CVP targets are **exact rationals**. `int`,
//! `str` (`"3/2"`) and `fractions.Fraction` are accepted; `float` is
//! **rejected** rather than rounded. A lattice is a discrete object and a
//! rounded basis entry is a different lattice, silently — which is exactly the
//! failure this module refuses to produce.
//!
//! Determinants, minima and norms come back as `int` when integral and
//! `fractions.Fraction` otherwise, never as `float`. The densities are `float`
//! because `Δ` carries a `π^{m/2}`; `center_density_exact()` gives the rational
//! when there is one and `None` when there is not.
//!
//! # Cost
//!
//! `minimum`, `shortest_vector`, `minimal_vectors`, `kissing_number`,
//! `theta_series` and `closest_vector` are **exact enumeration**, exponential
//! in the rank. Rank is capped at `LATTICE_MAX_ENUM_RANK` (24, so the Leech
//! lattice is inside it) and each takes an optional node `budget`. Above the
//! cap or past the budget they raise `LatticeError` (`E-LAT-008` /
//! `E-LAT-009`) rather than returning an approximation.

use std::str::FromStr;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rug::{Integer, Rational};

use alkahest_core::errors::AlkahestError as AlkahestErrorTrait;
use alkahest_core::lattice::{
    a_n, d_n, e8, leech, zn, Lattice, LatticeGeometryError, LatticeVector,
    DEFAULT_ENUM_NODE_BUDGET, MAX_ENUM_RANK, MAX_THETA_NORM,
};

// The toolkit's exception, and deliberately **a subclass of `LatticeError`**.
//
// `LatticeGeometryError` exists on the Rust side only because `LatticeError` is
// an exhaustive enum in the stable surface, so extending it would force a major
// version bump — see `alkahest_core::lattice::LatticeGeometryError`. Making it
// a subclass here keeps that a Rust-side detail: one `except LatticeError`
// still catches every refusal the lattice subsystem raises, and `.code` still
// reads `E-LAT-NNN` across the whole range.
pyo3::create_exception!(alkahest, PyLatticeGeometryError, crate::PyLatticeError);

/// Build a structured exception carrying `.code`, `.remediation` and `.span`.
///
/// Mirrors `lib.rs`'s `make_structured_err`; kept local so that adding this
/// module touches `lib.rs` in exactly two lines.
fn lat(e: LatticeGeometryError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyLatticeGeometryError>();
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

// ---------------------------------------------------------------------------
// Exact conversion
// ---------------------------------------------------------------------------

/// `int`, `str` or `fractions.Fraction` to an exact `Rational`.
fn to_rational(obj: &Bound<'_, PyAny>) -> PyResult<Rational> {
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        return Err(PyTypeError::new_err(
            "lattice entries are exact: pass an int, a string like '3/2', or a \
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

fn rows_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Rational>>> {
    let mut out = Vec::new();
    for row in obj.iter()? {
        let row = row?;
        let mut r = Vec::new();
        for item in row.iter()? {
            r.push(to_rational(&item?)?);
        }
        out.push(r);
    }
    Ok(out)
}

fn integer_to_py(py: Python<'_>, i: &Integer) -> PyResult<PyObject> {
    if let Some(v) = i.to_i64() {
        return Ok(v.into_py(py));
    }
    Ok(py
        .eval_bound(&format!("int('{i}')"), None, None)?
        .into_py(py))
}

/// Exact rational back to Python: an `int` when integral, else a
/// `fractions.Fraction`. Never a `float`.
fn rational_to_py(py: Python<'_>, r: &Rational) -> PyResult<PyObject> {
    if r.is_integer() {
        return integer_to_py(py, r.numer());
    }
    let fractions = PyModule::import_bound(py, "fractions")?;
    let frac = fractions.getattr("Fraction")?;
    Ok(frac.call1((r.to_string(),))?.into_py(py))
}

fn matrix_to_py(py: Python<'_>, m: &[Vec<Rational>]) -> PyResult<Vec<Vec<PyObject>>> {
    m.iter()
        .map(|row| row.iter().map(|x| rational_to_py(py, x)).collect())
        .collect()
}

fn budget_or_default(budget: Option<u64>) -> u64 {
    budget.unwrap_or(DEFAULT_ENUM_NODE_BUDGET)
}

// ---------------------------------------------------------------------------
// LatticeVector
// ---------------------------------------------------------------------------

/// A lattice vector produced by an exact enumeration.
#[pyclass(name = "LatticeVector", module = "alkahest", frozen)]
pub struct PyLatticeVector {
    inner: LatticeVector,
}

#[pymethods]
impl PyLatticeVector {
    /// Coefficients against the lattice's own basis (or Gram rows).
    fn coefficients(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner
            .coefficients
            .iter()
            .map(|c| integer_to_py(py, c))
            .collect()
    }

    /// Ambient coordinates, or `None` for a lattice built from a Gram matrix.
    fn coordinates(&self, py: Python<'_>) -> PyResult<Option<Vec<PyObject>>> {
        match &self.inner.coordinates {
            None => Ok(None),
            Some(c) => Ok(Some(
                c.iter()
                    .map(|x| rational_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )),
        }
    }

    /// The **squared** norm — `⟨v, v⟩`, or `⟨v − t, v − t⟩` for a CVP answer.
    fn norm(&self, py: Python<'_>) -> PyResult<PyObject> {
        rational_to_py(py, &self.inner.norm)
    }

    fn __repr__(&self) -> String {
        format!(
            "LatticeVector(coefficients={:?}, norm={})",
            self.inner
                .coefficients
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>(),
            self.inner.norm
        )
    }
}

fn vec_to_py(v: LatticeVector) -> PyLatticeVector {
    PyLatticeVector { inner: v }
}

// ---------------------------------------------------------------------------
// Lattice
// ---------------------------------------------------------------------------

/// A rank-`m` lattice, carried by its Gram matrix.
///
/// Build one from a basis (`Lattice.from_basis`), from a Gram matrix
/// (`Lattice.from_gram`), or by name (`Lattice.zn`, `Lattice.a_n`,
/// `Lattice.d_n`, `Lattice.e8`, `Lattice.leech`).
#[pyclass(name = "Lattice", module = "alkahest", frozen)]
pub struct PyLattice {
    inner: Lattice,
}

#[pymethods]
impl PyLattice {
    /// From a row basis of exact rationals. Rows must be linearly independent.
    #[staticmethod]
    fn from_basis(rows: &Bound<'_, PyAny>) -> PyResult<Self> {
        let r = rows_from_py(rows)?;
        Ok(Self {
            inner: Lattice::from_rational_basis(&r).map_err(lat)?,
        })
    }

    /// From a symmetric positive-definite Gram matrix `G[i][j] = ⟨b_i, b_j⟩`.
    ///
    /// This is how `E_8` and the Leech lattice are representable at all:
    /// neither has a rational basis in its natural embedding.
    #[staticmethod]
    fn from_gram(rows: &Bound<'_, PyAny>) -> PyResult<Self> {
        let r = rows_from_py(rows)?;
        Ok(Self {
            inner: Lattice::from_gram(&r).map_err(lat)?,
        })
    }

    /// The integer lattice `Z^n`.
    #[staticmethod]
    fn zn(n: usize) -> PyResult<Self> {
        Ok(Self {
            inner: zn(n).map_err(lat)?,
        })
    }

    /// The root lattice `A_n` (`A_2` is the hexagonal lattice).
    #[staticmethod]
    fn a_n(n: usize) -> PyResult<Self> {
        Ok(Self {
            inner: a_n(n).map_err(lat)?,
        })
    }

    /// The checkerboard lattice `D_n`.
    #[staticmethod]
    fn d_n(n: usize) -> PyResult<Self> {
        Ok(Self {
            inner: d_n(n).map_err(lat)?,
        })
    }

    /// The `E_8` root lattice: even, unimodular, 240 vectors of norm 2.
    #[staticmethod]
    fn e8() -> PyResult<Self> {
        Ok(Self {
            inner: e8().map_err(lat)?,
        })
    }

    /// The Leech lattice `Λ₂₄`: even, unimodular, minimum 4, 196560 vectors of
    /// norm 4. Constructed from the binary Golay code and verified before it is
    /// returned.
    #[staticmethod]
    fn leech() -> PyResult<Self> {
        Ok(Self {
            inner: leech().map_err(lat)?,
        })
    }

    /// Number of basis vectors.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Dimension of the ambient space (equal to the rank for a Gram-only
    /// lattice).
    #[getter]
    fn ambient_dimension(&self) -> usize {
        self.inner.ambient_dimension()
    }

    /// Every inner product is an integer.
    #[getter]
    fn is_integral(&self) -> bool {
        self.inner.is_integral()
    }

    /// Integral, and every vector has even norm.
    #[getter]
    fn is_even(&self) -> bool {
        self.inner.is_even()
    }

    /// `det G` — the **squared** covolume, exactly.
    fn determinant(&self, py: Python<'_>) -> PyResult<PyObject> {
        rational_to_py(py, &self.inner.determinant())
    }

    /// `sqrt(det G)`, as a `float`.
    fn covolume(&self) -> f64 {
        self.inner.covolume()
    }

    /// The Gram matrix, as exact rationals.
    fn gram_matrix(&self, py: Python<'_>) -> PyResult<Vec<Vec<PyObject>>> {
        matrix_to_py(py, self.inner.gram_matrix())
    }

    /// The basis rows, or `None` when the lattice was built from a Gram matrix.
    fn basis(&self, py: Python<'_>) -> PyResult<Option<Vec<Vec<PyObject>>>> {
        match self.inner.basis() {
            None => Ok(None),
            Some(b) => Ok(Some(matrix_to_py(py, b)?)),
        }
    }

    /// The dual lattice `L* = { y : ⟨y, x⟩ ∈ Z for all x in L }`.
    fn dual(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.dual().map_err(lat)?,
        })
    }

    /// An LLL-reduced basis of the same lattice (FLINT-backed for an integer
    /// basis).
    fn lll_reduced(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.lll_reduced().map_err(lat)?,
        })
    }

    /// Squared norm of the shortest non-zero vector, exactly.
    ///
    /// With no `budget`, shares one enumeration with `shortest_vector`,
    /// `kissing_number`, `hermite_invariant` and the densities. Passing a
    /// `budget` always does the work under that budget.
    #[pyo3(signature = (budget=None))]
    fn minimum(&self, py: Python<'_>, budget: Option<u64>) -> PyResult<PyObject> {
        let m = match budget {
            None => self.inner.minimum().map_err(lat)?,
            Some(b) => self.inner.minimum_with_budget(b).map_err(lat)?,
        };
        rational_to_py(py, &m)
    }

    /// A shortest non-zero lattice vector. Exact — it attains the minimum.
    #[pyo3(signature = (budget=None))]
    fn shortest_vector(&self, budget: Option<u64>) -> PyResult<PyLatticeVector> {
        let v = match budget {
            None => self.inner.shortest_vector().map_err(lat)?,
            Some(b) => self.inner.shortest_vector_with_budget(b).map_err(lat)?,
        };
        Ok(vec_to_py(v))
    }

    /// Every vector attaining the minimum, both signs included.
    #[pyo3(signature = (budget=None))]
    fn minimal_vectors(&self, budget: Option<u64>) -> PyResult<Vec<PyLatticeVector>> {
        Ok(self
            .inner
            .minimal_vectors_with_budget(budget_or_default(budget))
            .map_err(lat)?
            .into_iter()
            .map(vec_to_py)
            .collect())
    }

    /// How many lattice vectors attain the minimum.
    #[pyo3(signature = (budget=None))]
    fn kissing_number(&self, budget: Option<u64>) -> PyResult<u64> {
        match budget {
            None => self.inner.kissing_number().map_err(lat),
            Some(b) => self.inner.kissing_number_with_budget(b).map_err(lat),
        }
    }

    /// `theta[n]` = the number of vectors of squared norm exactly `n`, for
    /// `n = 0 ..= max_norm`. Needs an integral Gram matrix.
    #[pyo3(signature = (max_norm, budget=None))]
    fn theta_series(
        &self,
        py: Python<'_>,
        max_norm: u64,
        budget: Option<u64>,
    ) -> PyResult<Vec<PyObject>> {
        self.inner
            .theta_series_with_budget(max_norm, budget_or_default(budget))
            .map_err(lat)?
            .iter()
            .map(|c| integer_to_py(py, c))
            .collect()
    }

    /// A lattice vector closest to `target`, given in ambient coordinates.
    /// Exact. Needs a basis.
    #[pyo3(signature = (target, budget=None))]
    fn closest_vector(
        &self,
        target: &Bound<'_, PyAny>,
        budget: Option<u64>,
    ) -> PyResult<PyLatticeVector> {
        let mut t = Vec::new();
        for item in target.iter()? {
            t.push(to_rational(&item?)?);
        }
        Ok(vec_to_py(
            self.inner
                .closest_vector_with_budget(&t, budget_or_default(budget))
                .map_err(lat)?,
        ))
    }

    /// Centre density `δ = (λ₁²/4)^{m/2} / sqrt(det G)`, as a `float`.
    fn center_density(&self) -> PyResult<f64> {
        self.inner.center_density().map_err(lat)
    }

    /// The centre density as an exact rational, or `None` when it is
    /// irrational — never a rounded stand-in.
    fn center_density_exact(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match self.inner.center_density_exact().map_err(lat)? {
            None => Ok(None),
            Some(r) => Ok(Some(rational_to_py(py, &r)?)),
        }
    }

    /// Sphere-packing density `Δ = δ · V_m`, as a `float`.
    fn packing_density(&self) -> PyResult<f64> {
        self.inner.packing_density().map_err(lat)
    }

    /// Hermite invariant `γ = λ₁² / (det G)^{1/m}`, as a `float`.
    fn hermite_invariant(&self) -> PyResult<f64> {
        self.inner.hermite_invariant().map_err(lat)
    }

    fn __repr__(&self) -> String {
        format!(
            "Lattice(rank={}, ambient_dimension={}, determinant={})",
            self.inner.rank(),
            self.inner.ambient_dimension(),
            self.inner.determinant()
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLattice>()?;
    m.add_class::<PyLatticeVector>()?;
    m.add(
        "LatticeGeometryError",
        m.py().get_type_bound::<PyLatticeGeometryError>(),
    )?;
    m.add("LATTICE_MAX_ENUM_RANK", MAX_ENUM_RANK)?;
    m.add("LATTICE_DEFAULT_ENUM_NODE_BUDGET", DEFAULT_ENUM_NODE_BUDGET)?;
    m.add("LATTICE_MAX_THETA_NORM", MAX_THETA_NORM)?;
    Ok(())
}
