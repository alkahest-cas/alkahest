//! PyO3 bindings for `alkahest_core::specfun` — Bessel, Airy, Riemann and
//! Hurwitz zeta, and polylogarithm functions (G11b).
//!
//! Refusals arrive as `alkahest.SpecialFunctionError` with a stable
//! `E-SPECFN-NNN` `.code` (a budget trip as `BudgetExceededError`).

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use alkahest_core::specfun::{self as core, AiryKind, BesselKind, SphericalBesselKind};

use crate::orthopoly::{pool_of, specfn_err};
use crate::{PyExpr, PyExprPool};

/// Coerce every argument into the pool of the first `Expr` among them.
fn ids(
    py: Python<'_>,
    what: &str,
    obs: &[&Bound<'_, PyAny>],
) -> PyResult<(Py<PyExprPool>, Vec<alkahest_core::kernel::ExprId>)> {
    let pool_py = pool_of(py, obs).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "{what}: at least one argument (normally x) must be an Expr, so the result has a \
             pool; use pool.symbol(\"x\") or pool.integer(...)"
        ))
    })?;
    let mut out = Vec::with_capacity(obs.len());
    for ob in obs {
        out.push(crate::coerce_substituent(&pool_py, ob, py)?);
    }
    Ok((pool_py, out))
}

fn cylinder(
    py: Python<'_>,
    kind: BesselKind,
    nu: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    let (pool_py, a) = ids(py, kind.name(), &[nu, x])?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::bessel(kind, a[0], a[1], expand, pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Bessel function of the first kind ``J_nu(x)`` (DLMF 10.2; SymPy's
/// ``besselj``).
///
/// A half-integer order (``Fraction(1, 2)``, ``pool.rational(3, 2)``) with
/// ``expand=True`` (the default) gives the elementary form —
/// ``bessel_j(1/2, x) = sqrt(2)*sin(x)/(sqrt(pi)*sqrt(x))``.  Otherwise the
/// node ``bessel_j(nu, x)``, canonicalised: ``bessel_j(0, x)`` is
/// ``bessel_j0(x)``, ``bessel_j(1, x)`` is ``bessel_j1(x)``,
/// ``J_{-n} = (-1)^n J_n``, ``J_n(-x) = (-1)^n J_n(x)``, ``J_0(0) = 1`` and
/// ``J_nu(0) = 0`` for ``nu > 0``.
///
/// The node differentiates in ``x`` (``d/dx J_nu = (J_{nu-1} - J_{nu+1})/2``;
/// a derivative in ``nu`` is refused with ``E-DIFF-001``), evaluates in
/// ``f64``, in rigorous balls (``interval_eval``) and at complex points
/// (``evaluate(e, {x: 1+2j})``), through Arb.
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-004``) for an expansion past
/// order 200 — keep it symbolic with ``expand=False``.
///
/// >>> import alkahest as ak
/// >>> from fractions import Fraction
/// >>> p = ak.ExprPool(); x = p.symbol("x")
/// >>> ak.bessel_j(Fraction(1, 2), x)     # sqrt(2)*pi^(-1/2)*x^(-1/2)*sin(x)
#[pyfunction]
#[pyo3(signature = (nu, x, *, expand = true))]
fn bessel_j(
    py: Python<'_>,
    nu: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    cylinder(py, BesselKind::J, nu, x, expand)
}

/// Bessel function of the second kind ``Y_nu(x)`` (DLMF 10.2; SymPy's
/// ``bessely``).  Half-integer orders expand
/// (``Y_{1/2}(x) = -sqrt(2)*cos(x)/(sqrt(pi)*sqrt(x))``); ``Y_{-n} = (-1)^n
/// Y_n``.  Real only for ``x > 0``: ``f64`` evaluation at ``x <= 0`` declines.
#[pyfunction]
#[pyo3(signature = (nu, x, *, expand = true))]
fn bessel_y(
    py: Python<'_>,
    nu: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    cylinder(py, BesselKind::Y, nu, x, expand)
}

/// Modified Bessel function of the first kind ``I_nu(x)`` (DLMF 10.25;
/// SymPy's ``besseli``).  Half-integer orders expand into ``sinh``/``cosh``;
/// ``I_{-n} = I_n``; ``d/dx I_nu = (I_{nu-1} + I_{nu+1})/2``.
#[pyfunction]
#[pyo3(signature = (nu, x, *, expand = true))]
fn bessel_i(
    py: Python<'_>,
    nu: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    cylinder(py, BesselKind::I, nu, x, expand)
}

/// Modified Bessel function of the second kind ``K_nu(x)`` (DLMF 10.25;
/// SymPy's ``besselk``).  Half-integer orders expand
/// (``K_{1/2}(x) = sqrt(pi/(2x))*exp(-x)``); ``K_{-nu} = K_nu``;
/// ``d/dx K_nu = -(K_{nu-1} + K_{nu+1})/2``.
#[pyfunction]
#[pyo3(signature = (nu, x, *, expand = true))]
fn bessel_k(
    py: Python<'_>,
    nu: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    cylinder(py, BesselKind::K, nu, x, expand)
}

fn spherical(
    py: Python<'_>,
    kind: SphericalBesselKind,
    n: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    let (pool_py, a) = ids(py, kind.name(), &[n, x])?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::spherical_bessel(kind, a[0], a[1], expand, pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Spherical Bessel function of the first kind
/// ``j_n(x) = sqrt(pi/(2x)) J_{n+1/2}(x)`` (DLMF 10.47; SymPy's ``jn``).
///
/// An integer ``n`` expands (``j_1(x) = sin(x)/x**2 - cos(x)/x``); a
/// symbolic ``n`` keeps the node, with ``d/dx j_n = -j_{n+1} + (n/x) j_n``.
/// A non-integer literal order raises ``E-SPECFN-007``.
#[pyfunction]
#[pyo3(signature = (n, x, *, expand = true))]
fn spherical_jn(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    spherical(py, SphericalBesselKind::J, n, x, expand)
}

/// Spherical Bessel function of the second kind
/// ``y_n(x) = sqrt(pi/(2x)) Y_{n+1/2}(x)`` (DLMF 10.47; SymPy's ``yn``).
/// ``y_0(x) = -cos(x)/x``.
#[pyfunction]
#[pyo3(signature = (n, x, *, expand = true))]
fn spherical_yn(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    spherical(py, SphericalBesselKind::Y, n, x, expand)
}

fn airy_fn(py: Python<'_>, kind: AiryKind, x: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let (pool_py, a) = ids(py, kind.name(), &[x])?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::airy(kind, a[0], pool)
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Airy function ``Ai(x)`` (DLMF 9.2; SymPy's ``airyai``).  ``d/dx Ai =
/// airy_ai_prime``; ``Ai(0) = 3**(-2/3)/gamma(2/3)``.
#[pyfunction]
fn airy_ai(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    airy_fn(py, AiryKind::Ai, x)
}

/// Airy function ``Bi(x)`` (DLMF 9.2; SymPy's ``airybi``).  ``d/dx Bi =
/// airy_bi_prime``; ``Bi(0) = 3**(-1/6)/gamma(2/3)``.
#[pyfunction]
fn airy_bi(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    airy_fn(py, AiryKind::Bi, x)
}

/// ``Ai'(x)`` (SymPy's ``airyaiprime``).  ``d/dx Ai'(x) = x*Ai(x)``;
/// ``Ai'(0) = -3**(-1/3)/gamma(1/3)``.
#[pyfunction]
fn airy_ai_prime(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    airy_fn(py, AiryKind::AiPrime, x)
}

/// ``Bi'(x)`` (SymPy's ``airybiprime``).  ``d/dx Bi'(x) = x*Bi(x)``;
/// ``Bi'(0) = 3**(1/6)/gamma(1/3)``.
#[pyfunction]
fn airy_bi_prime(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    airy_fn(py, AiryKind::BiPrime, x)
}

/// Riemann zeta function ``zeta(s)`` (DLMF 25.2).
///
/// Exact at even positive integers (``zeta(2) = pi**2/6``,
/// ``zeta(4) = pi**4/90``, via Bernoulli numbers) and at non-positive
/// integers (``zeta(0) = -1/2``, ``zeta(-1) = -1/12``, ``zeta(-2n) = 0``);
/// ``zeta(3)`` stays ``zeta(3)`` (no closed form is known).  Evaluates in
/// ``f64``, balls and at complex ``s`` through Arb.  There is no derivative
/// rule: ``diff(zeta(s), s)`` raises ``DiffError`` (``E-DIFF-001``).
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-006``) at the pole ``s = 1``.
#[pyfunction]
fn zeta(py: Python<'_>, s: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let (pool_py, a) = ids(py, "zeta", &[s])?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::zeta(a[0], pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Hurwitz zeta function ``zeta(s, a) = sum_{k>=0} (k + a)**(-s)``
/// (DLMF 25.11; SymPy's two-argument ``zeta``).
///
/// ``hurwitz_zeta(s, 1) = zeta(s)``, ``hurwitz_zeta(s, 1/2) = (2**s - 1)*zeta(s)``,
/// ``hurwitz_zeta(-n, a) = -B_{n+1}(a)/(n+1)`` (a Bernoulli polynomial in any
/// ``a``); ``d/da zeta(s, a) = -s*zeta(s+1, a)``.  ``E-SPECFN-006`` at ``s = 1``.
#[pyfunction]
fn hurwitz_zeta(py: Python<'_>, s: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let (pool_py, ids_) = ids(py, "hurwitz_zeta", &[s, a])?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::hurwitz_zeta(ids_[0], ids_[1], pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Polylogarithm ``Li_s(z)`` (DLMF 25.12; SymPy's ``polylog``), principal
/// branch with the cut ``[1, oo)``.
///
/// ``polylog(2, z)`` is ``dilog(z)``; ``polylog(1, z) = -log(1 - z)``;
/// ``polylog(0, z) = z/(1 - z)``; ``polylog(-n, z)`` is the Eulerian rational
/// function (``Li_{-1}(z) = z/(1 - z)**2``); ``polylog(s, 1) = zeta(s)`` for
/// ``s > 1`` and ``polylog(s, -1) = -(1 - 2**(1-s))*zeta(s)``.
/// ``d/dz Li_s(z) = Li_{s-1}(z)/z``; a derivative in ``s`` is refused.
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-006``) at ``z = 1`` for
/// ``s <= 1`` (a pole).
#[pyfunction]
fn polylog(py: Python<'_>, s: &Bound<'_, PyAny>, z: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let (pool_py, a) = ids(py, "polylog", &[s, z])?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::polylog(a[0], a[1], pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Rewrite every Bessel node whose order is now a half-integer literal (and
/// every spherical Bessel node with an integer order) into its elementary
/// form, and re-canonicalise ``zeta`` / ``hurwitz_zeta`` / ``polylog`` nodes —
/// e.g. after ``subs(bessel_j(nu, x), {nu: Fraction(1, 2)})``.  SymPy's
/// ``expand_func`` for these families.
#[pyfunction]
fn expand_special(py: Python<'_>, expr: PyRef<'_, PyExpr>) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::expand_special(expr.id, pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bessel_j, m)?)?;
    m.add_function(wrap_pyfunction!(bessel_y, m)?)?;
    m.add_function(wrap_pyfunction!(bessel_i, m)?)?;
    m.add_function(wrap_pyfunction!(bessel_k, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_jn, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_yn, m)?)?;
    m.add_function(wrap_pyfunction!(airy_ai, m)?)?;
    m.add_function(wrap_pyfunction!(airy_bi, m)?)?;
    m.add_function(wrap_pyfunction!(airy_ai_prime, m)?)?;
    m.add_function(wrap_pyfunction!(airy_bi_prime, m)?)?;
    m.add_function(wrap_pyfunction!(zeta, m)?)?;
    m.add_function(wrap_pyfunction!(hurwitz_zeta, m)?)?;
    m.add_function(wrap_pyfunction!(polylog, m)?)?;
    m.add_function(wrap_pyfunction!(expand_special, m)?)?;
    Ok(())
}
