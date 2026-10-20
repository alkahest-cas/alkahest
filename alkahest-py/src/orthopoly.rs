//! PyO3 bindings for `alkahest_core::orthopoly` — classical orthogonal
//! polynomials, spherical harmonics, Wigner 3j/6j symbols and Clebsch–Gordan
//! coefficients (G11a).
//!
//! Refusals arrive as `alkahest.SpecialFunctionError` with a stable
//! `E-SPECFN-NNN` `.code` (a budget trip as `BudgetExceededError`).

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyFloat;

use alkahest_core::kernel::{ExprData, ExprPool};
use alkahest_core::orthopoly::{self as core, HalfInt, OrthoFamily, SpecialFunctionError};
use rug::Rational;

use crate::bigint::rational_from_py;
use crate::{PyDerivedResult, PyExpr, PyExprPool};

pyo3::create_exception!(alkahest, PySpecialFunctionError, crate::PyAlkahestError);

pub(crate) fn specfn_err(e: SpecialFunctionError) -> PyErr {
    Python::with_gil(|py| {
        if let SpecialFunctionError::Budget(b) = &e {
            let exc_type = py.get_type_bound::<crate::PyBudgetExceededError>();
            return crate::make_structured_err(py, &exc_type, b);
        }
        let exc_type = py.get_type_bound::<PySpecialFunctionError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

/// The pool of the first `Expr` (or `DerivedResult`) among `obs`.
pub(crate) fn pool_of(py: Python<'_>, obs: &[&Bound<'_, PyAny>]) -> Option<Py<PyExprPool>> {
    for ob in obs {
        if let Ok(e) = ob.extract::<PyRef<'_, PyExpr>>() {
            return Some(e.pool.clone_ref(py));
        }
        if let Ok(dr) = ob.downcast::<PyDerivedResult>() {
            return Some(dr.borrow().value.pool.clone_ref(py));
        }
    }
    None
}

fn ortho(
    py: Python<'_>,
    family: OrthoFamily,
    n: &Bound<'_, PyAny>,
    params: &[&Bound<'_, PyAny>],
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    let mut all: Vec<&Bound<'_, PyAny>> = vec![x, n];
    all.extend_from_slice(params);
    let pool_py = pool_of(py, &all).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "{}: at least one argument (normally x) must be an Expr, so the result has a \
             pool; use pool.symbol(\"x\") or pool.integer(...)",
            family.name()
        ))
    })?;
    let n_id = crate::coerce_substituent(&pool_py, n, py)?;
    let x_id = crate::coerce_substituent(&pool_py, x, py)?;
    let mut p_ids = Vec::with_capacity(params.len());
    for p in params {
        p_ids.push(crate::coerce_substituent(&pool_py, p, py)?);
    }
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::orthopoly(family, n_id, &p_ids, x_id, expand, pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Legendre polynomial ``P_n(x)`` (DLMF 18.3; ``P_n(1) = 1``).
///
/// With an integer degree ``n`` (a Python ``int`` or an integer ``Expr``)
/// and ``expand=True`` (the default) the explicit polynomial is returned, with
/// exact rational coefficients; ``P_{-n-1} = P_n``.  A symbolic or
/// non-integer degree, or ``expand=False``, gives the unevaluated node
/// ``legendre_p(n, x)``, which differentiates in ``x``
/// (``d/dx P_n = C_{n-1}^{(3/2)}``), evaluates numerically (``f64`` and
/// rigorous balls, any real degree) and simplifies at special values
/// (``P_n(1) = 1``, parity for integer ``n``).
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-004``) for an explicit
/// expansion past degree 1000 — keep it symbolic with ``expand=False``.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); x = p.symbol("x")
/// >>> ak.legendre_p(2, x)            # 3/2*x^2 - 1/2
#[pyfunction]
#[pyo3(signature = (n, x, *, expand = true))]
fn legendre_p(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    ortho(py, OrthoFamily::Legendre, n, &[], x, expand)
}

/// Associated Legendre (Ferrers) function ``P_n^m(x)``, **Condon–Shortley
/// phase included**: ``P_n^m(x) = (-1)^m (1-x^2)^{m/2} d^m P_n/dx^m`` for
/// ``m >= 0`` and ``P_n^{-m} = (-1)^m (n-m)!/(n+m)! P_n^m`` (DLMF 14.6.1,
/// 14.9.3; SymPy's ``assoc_legendre``).  ``|m| > n`` gives ``0``.
///
/// Integer ``n`` and ``m`` expand (``(1-x^2)^{m/2}`` times a polynomial);
/// otherwise the node ``assoc_legendre_p(n, m, x)``.  For odd ``m`` the
/// function is real only on ``[-1, 1]``; numeric evaluation outside it
/// declines rather than return a real part.
#[pyfunction]
#[pyo3(signature = (n, m, x, *, expand = true))]
fn assoc_legendre_p(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    ortho(py, OrthoFamily::AssocLegendre, n, &[m], x, expand)
}

/// Physicists' Hermite polynomial ``H_n(x)`` (weight ``exp(-x^2)``,
/// ``H_n = 2^n x^n + ...``; SymPy's ``hermite``).  Integer ``n >= 0`` expands;
/// ``d/dx H_n = 2n H_{n-1}``.
#[pyfunction]
#[pyo3(signature = (n, x, *, expand = true))]
fn hermite_h(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    ortho(py, OrthoFamily::HermiteH, n, &[], x, expand)
}

/// Probabilists' Hermite polynomial ``He_n(x)`` (monic, weight
/// ``exp(-x^2/2)``; SymPy's ``hermite_prob``).  ``He_n(x) = 2^{-n/2}
/// H_n(x/sqrt(2))``; ``d/dx He_n = n He_{n-1}``.
#[pyfunction]
#[pyo3(signature = (n, x, *, expand = true))]
fn hermite_he(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    ortho(py, OrthoFamily::HermiteHe, n, &[], x, expand)
}

/// Laguerre polynomial ``L_n(x) = L_n^{(0)}(x)`` (weight ``exp(-x)`` on
/// ``[0, oo)``; SymPy's ``laguerre``).  ``d/dx L_n = -L_{n-1}^{(1)}``.
#[pyfunction]
#[pyo3(signature = (n, x, *, expand = true))]
fn laguerre_l(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    ortho(py, OrthoFamily::Laguerre, n, &[], x, expand)
}

/// Generalised (associated) Laguerre polynomial ``L_n^{(alpha)}(x)``
/// (SymPy's ``assoc_laguerre``).  ``alpha`` may be symbolic: the expansion is
/// then polynomial in it.  ``d/dx L_n^{(a)} = -L_{n-1}^{(a+1)}``.  Hydrogen
/// radial functions use ``L_{n-l-1}^{(2l+1)}``.
#[pyfunction]
#[pyo3(signature = (n, alpha, x, *, expand = true))]
fn assoc_laguerre_l(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    alpha: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    ortho(py, OrthoFamily::AssocLaguerre, n, &[alpha], x, expand)
}

/// Chebyshev polynomial of the first kind ``T_n(x)`` (``T_n(cos t) =
/// cos(n t)``; ``T_{-n} = T_n``).  ``d/dx T_n = n U_{n-1}``.
#[pyfunction]
#[pyo3(signature = (n, x, *, expand = true))]
fn chebyshev_t(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    ortho(py, OrthoFamily::ChebyshevT, n, &[], x, expand)
}

/// Chebyshev polynomial of the second kind ``U_n(x)`` (``U_n(cos t) =
/// sin((n+1)t)/sin t``; ``U_{-1} = 0``, ``U_{-n} = -U_{n-2}``).
#[pyfunction]
#[pyo3(signature = (n, x, *, expand = true))]
fn chebyshev_u(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    ortho(py, OrthoFamily::ChebyshevU, n, &[], x, expand)
}

/// Gegenbauer (ultraspherical) polynomial ``C_n^{(a)}(x)`` (SymPy's
/// ``gegenbauer``; ``C_n^{(0)} = 0`` for ``n >= 1``).
/// ``d/dx C_n^{(a)} = 2a C_{n-1}^{(a+1)}``.
#[pyfunction]
#[pyo3(signature = (n, a, x, *, expand = true))]
fn gegenbauer_c(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    ortho(py, OrthoFamily::Gegenbauer, n, &[a], x, expand)
}

/// Jacobi polynomial ``P_n^{(a,b)}(x)`` (SymPy's ``jacobi``).
/// ``d/dx P_n^{(a,b)} = (n+a+b+1)/2 P_{n-1}^{(a+1,b+1)}``; parity swaps
/// ``a`` and ``b``.
#[pyfunction]
#[pyo3(signature = (n, a, b, x, *, expand = true))]
fn jacobi_p(
    py: Python<'_>,
    n: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    ortho(py, OrthoFamily::Jacobi, n, &[a, b], x, expand)
}

/// Rewrite every orthogonal-polynomial node in ``expr`` whose degree (and
/// Legendre order) is now an integer literal into its explicit polynomial —
/// e.g. after ``subs(legendre_p(n, x), {n: 5})``.  Nodes with a symbolic
/// degree are left alone.
#[pyfunction]
fn expand_orthopoly(py: Python<'_>, expr: PyRef<'_, PyExpr>) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::expand_orthopoly(expr.id, pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

// ---------------------------------------------------------------------------
// Angular momentum
// ---------------------------------------------------------------------------

/// An exact rational from an int, ``Fraction``, a float that is exactly
/// representable (half-integers are), or a numeric ``Expr`` literal.
fn exact_number(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<Rational> {
    let not_number = || {
        specfn_err(SpecialFunctionError::NotHalfInteger {
            what: what.to_string(),
            value: ob
                .str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "?".into()),
        })
    };
    if let Ok(e) = ob.extract::<PyRef<'_, PyExpr>>() {
        let py = ob.py();
        let pool = &e.pool.borrow(py).inner;
        return match pool.get(e.id) {
            ExprData::Integer(n) => Ok(Rational::from(n.0.clone())),
            ExprData::Rational(q) => Ok(q.0.clone()),
            ExprData::Float(f) => Rational::from_f64(f.inner.to_f64()).ok_or_else(not_number),
            _ => Err(not_number()),
        };
    }
    if ob.is_instance_of::<PyFloat>() {
        let v: f64 = ob.extract()?;
        return Rational::from_f64(v).ok_or_else(not_number);
    }
    if let Some(q) = rational_from_py(ob)? {
        return Ok(q);
    }
    Err(not_number())
}

fn half_int(ob: &Bound<'_, PyAny>, what: &str) -> PyResult<HalfInt> {
    let q = exact_number(ob, what)?;
    HalfInt::from_rational(&q, what).map_err(specfn_err)
}

static DEFAULT_POOL: pyo3::sync::GILOnceCell<Py<PyExprPool>> = pyo3::sync::GILOnceCell::new();

fn result_pool(
    py: Python<'_>,
    args: &[&Bound<'_, PyAny>],
    pool: Option<Py<PyExprPool>>,
) -> PyResult<Py<PyExprPool>> {
    if let Some(p) = pool_of(py, args) {
        return Ok(p);
    }
    if let Some(p) = pool {
        return Ok(p);
    }
    Ok(DEFAULT_POOL
        .get_or_try_init(py, || {
            Py::new(
                py,
                PyExprPool {
                    inner: ExprPool::new(),
                },
            )
        })?
        .clone_ref(py))
}

type CouplingFn = fn(
    HalfInt,
    HalfInt,
    HalfInt,
    HalfInt,
    HalfInt,
    HalfInt,
) -> Result<core::Surd, SpecialFunctionError>;

fn coupling(
    py: Python<'_>,
    f: CouplingFn,
    args: [&Bound<'_, PyAny>; 6],
    names: [&str; 6],
    pool: Option<Py<PyExprPool>>,
) -> PyResult<PyExpr> {
    let mut h = [HalfInt::from_int(0); 6];
    for i in 0..6 {
        h[i] = half_int(args[i], names[i])?;
    }
    let pool_py = result_pool(py, &args, pool)?;
    let s = py
        .allow_threads(|| f(h[0], h[1], h[2], h[3], h[4], h[5]))
        .map_err(specfn_err)?;
    let id = s.to_expr(&pool_py.borrow(py).inner);
    Ok(PyExpr { id, pool: pool_py })
}

/// Wigner 3j symbol ``(j1 j2 j3; m1 m2 m3)``, exactly.
///
/// Arguments are integers or half-integers: ``int``, ``Fraction``, a float
/// that is exactly ``k/2`` (``0.5``), or a numeric ``Expr``.  The result is
/// an ``Expr`` of the form ``q`` or ``q*r**(1/2)`` (``q`` rational, ``r``
/// square-free), computed by Racah's formula over big integers — the same
/// normalisation and phase as SymPy's ``wigner_3j`` and Mathematica's
/// ``ThreeJSymbol``.  Selection-rule violations (triangle, ``|m| > j``,
/// ``m1+m2+m3 != 0``, ``j-m`` not an integer) give ``0``.
///
/// The result lives in the pool of any ``Expr`` argument, else ``pool``,
/// else a module-wide default pool.
///
/// Raises ``SpecialFunctionError``: ``E-SPECFN-001`` for an argument that is
/// not an integer or half-integer, ``E-SPECFN-002`` for a negative ``j``,
/// ``E-SPECFN-004`` past ``2j = 20000``.
///
/// >>> import alkahest as ak
/// >>> ak.wigner_3j(1, 1, 0, 0, 0, 0)      # -1/3*3^(1/2)
#[pyfunction]
#[pyo3(signature = (j1, j2, j3, m1, m2, m3, *, pool = None))]
#[allow(clippy::too_many_arguments)]
fn wigner_3j(
    py: Python<'_>,
    j1: &Bound<'_, PyAny>,
    j2: &Bound<'_, PyAny>,
    j3: &Bound<'_, PyAny>,
    m1: &Bound<'_, PyAny>,
    m2: &Bound<'_, PyAny>,
    m3: &Bound<'_, PyAny>,
    pool: Option<Py<PyExprPool>>,
) -> PyResult<PyExpr> {
    coupling(
        py,
        core::wigner_3j,
        [j1, j2, j3, m1, m2, m3],
        ["j1", "j2", "j3", "m1", "m2", "m3"],
        pool,
    )
}

/// Clebsch–Gordan coefficient ``<j1 m1; j2 m2 | j3 m3>`` (Condon–Shortley
/// phase), exactly.  **Argument order is SymPy's**:
/// ``clebsch_gordan(j1, j2, j3, m1, m2, m3)``.
///
/// ``<j1 m1 j2 m2|J M> = (-1)^(j1-j2+M) sqrt(2J+1) (j1 j2 J; m1 m2 -M)``.
/// Inputs, pool and refusals as :func:`wigner_3j`.
///
/// >>> import alkahest as ak
/// >>> from fractions import Fraction as F
/// >>> ak.clebsch_gordan(F(1, 2), F(1, 2), 1, F(1, 2), F(-1, 2), 0)   # 1/2*2^(1/2)
#[pyfunction]
#[pyo3(signature = (j1, j2, j3, m1, m2, m3, *, pool = None))]
#[allow(clippy::too_many_arguments)]
fn clebsch_gordan(
    py: Python<'_>,
    j1: &Bound<'_, PyAny>,
    j2: &Bound<'_, PyAny>,
    j3: &Bound<'_, PyAny>,
    m1: &Bound<'_, PyAny>,
    m2: &Bound<'_, PyAny>,
    m3: &Bound<'_, PyAny>,
    pool: Option<Py<PyExprPool>>,
) -> PyResult<PyExpr> {
    coupling(
        py,
        core::clebsch_gordan,
        [j1, j2, j3, m1, m2, m3],
        ["j1", "j2", "j3", "m1", "m2", "m3"],
        pool,
    )
}

/// Wigner 6j symbol ``{j1 j2 j3; j4 j5 j6}``, exactly (Racah's formula; SymPy's
/// ``wigner_6j``).  A triad ``(j1 j2 j3)``, ``(j1 j5 j6)``, ``(j4 j2 j6)``,
/// ``(j4 j5 j3)`` that fails the triangle condition gives ``0``.  Inputs, pool
/// and refusals as :func:`wigner_3j`.
#[pyfunction]
#[pyo3(signature = (j1, j2, j3, j4, j5, j6, *, pool = None))]
#[allow(clippy::too_many_arguments)]
fn wigner_6j(
    py: Python<'_>,
    j1: &Bound<'_, PyAny>,
    j2: &Bound<'_, PyAny>,
    j3: &Bound<'_, PyAny>,
    j4: &Bound<'_, PyAny>,
    j5: &Bound<'_, PyAny>,
    j6: &Bound<'_, PyAny>,
    pool: Option<Py<PyExprPool>>,
) -> PyResult<PyExpr> {
    coupling(
        py,
        core::wigner_6j,
        [j1, j2, j3, j4, j5, j6],
        ["j1", "j2", "j3", "j4", "j5", "j6"],
        pool,
    )
}

/// Complex spherical harmonic ``Y_l^m(theta, phi)`` as an explicit expression.
///
/// ``Y_l^m = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!) P_l^m(cos theta) exp(i m phi)``
/// with the Condon–Shortley phase inside ``P_l^m`` — SymPy's ``Ynm`` and
/// Mathematica's ``SphericalHarmonicY``; orthonormal on the unit sphere,
/// ``Y_l^{-m} = (-1)^m conj(Y_l^m)``.  ``(1 - cos^2 theta)^{|m|/2}`` is
/// written ``sin(theta)^|m|``, exact for the polar range ``0 <= theta <= pi``.
/// ``|m| > l`` gives ``0``.  The imaginary unit is the pool's ``I``.
///
/// Raises ``SpecialFunctionError``: ``E-SPECFN-003`` for non-integer ``l`` or
/// ``m``, ``E-SPECFN-002`` for ``l < 0``, ``E-SPECFN-004`` for ``l > 1000``.
#[pyfunction]
fn spherical_harmonic_y(
    py: Python<'_>,
    l: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
    theta: &Bound<'_, PyAny>,
    phi: &Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    let pool_py = pool_of(py, &[theta, phi, l, m]).ok_or_else(|| {
        PyTypeError::new_err("spherical_harmonic_y: theta or phi must be an Expr")
    })?;
    let lq = exact_number(l, "l").map_err(|_| {
        specfn_err(SpecialFunctionError::NotAnInteger {
            what: "l".into(),
            value: l.str().map(|s| s.to_string()).unwrap_or_default(),
        })
    })?;
    let mq = exact_number(m, "m").map_err(|_| {
        specfn_err(SpecialFunctionError::NotAnInteger {
            what: "m".into(),
            value: m.str().map(|s| s.to_string()).unwrap_or_default(),
        })
    })?;
    let t = crate::coerce_substituent(&pool_py, theta, py)?;
    let p = crate::coerce_substituent(&pool_py, phi, py)?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::spherical_harmonic_y(&lq, &mq, t, p, pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Real spherical harmonic ``Y_{l,m}(theta, phi)`` as an explicit expression.
///
/// The tesseral (real) basis, Wikipedia's convention: for ``m > 0``
/// ``sqrt(2)*N*P_l^m(cos theta)*cos(m phi)``, for ``m < 0``
/// ``sqrt(2)*N*P_l^{|m|}(cos theta)*sin(|m| phi)``, and ``Y_l^0`` for
/// ``m = 0``, where ``N = sqrt((2l+1)/(4 pi) (l-|m|)!/(l+|m|)!)`` and ``P_l^m``
/// is the Legendre function *without* the Condon-Shortley phase (the phase of
/// the complex ``Y_l^m`` cancels the ``(-1)^m`` in the definition
/// ``sqrt(2) (-1)^m Re/Im Y_l^{|m|}``).  So ``Y_{1,1} = sqrt(3/(4 pi)) x/r``,
/// ``Y_{1,-1} = sqrt(3/(4 pi)) y/r``, ``Y_{1,0} = sqrt(3/(4 pi)) z/r``.  The
/// functions are real and orthonormal on the unit sphere.
/// ``sin(theta)**|m|`` stands for ``(1 - cos(theta)**2)**(|m|/2)`` (exact for
/// ``0 <= theta <= pi``).  ``|m| > l`` gives ``0``.
///
/// Raises ``SpecialFunctionError``: ``E-SPECFN-003`` for non-integer ``l`` or
/// ``m``, ``E-SPECFN-002`` for ``l < 0``, ``E-SPECFN-004`` for ``l > 1000``.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); t = p.symbol("t"); f = p.symbol("f")
/// >>> ak.real_spherical_harmonic_y(1, 1, t, f)   # sqrt(3/(4 pi)) sin(t) cos(f)
#[pyfunction]
fn real_spherical_harmonic_y(
    py: Python<'_>,
    l: &Bound<'_, PyAny>,
    m: &Bound<'_, PyAny>,
    theta: &Bound<'_, PyAny>,
    phi: &Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    let pool_py = pool_of(py, &[theta, phi, l, m]).ok_or_else(|| {
        PyTypeError::new_err("real_spherical_harmonic_y: theta or phi must be an Expr")
    })?;
    let lq = exact_number(l, "l").map_err(|_| {
        specfn_err(SpecialFunctionError::NotAnInteger {
            what: "l".into(),
            value: l.str().map(|s| s.to_string()).unwrap_or_default(),
        })
    })?;
    let mq = exact_number(m, "m").map_err(|_| {
        specfn_err(SpecialFunctionError::NotAnInteger {
            what: "m".into(),
            value: m.str().map(|s| s.to_string()).unwrap_or_default(),
        })
    })?;
    let t = crate::coerce_substituent(&pool_py, theta, py)?;
    let p = crate::coerce_substituent(&pool_py, phi, py)?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::real_spherical_harmonic_y(&lq, &mq, t, p, pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "SpecialFunctionError",
        m.py().get_type_bound::<PySpecialFunctionError>(),
    )?;
    m.add_function(wrap_pyfunction!(legendre_p, m)?)?;
    m.add_function(wrap_pyfunction!(assoc_legendre_p, m)?)?;
    m.add_function(wrap_pyfunction!(hermite_h, m)?)?;
    m.add_function(wrap_pyfunction!(hermite_he, m)?)?;
    m.add_function(wrap_pyfunction!(laguerre_l, m)?)?;
    m.add_function(wrap_pyfunction!(assoc_laguerre_l, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_t, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_u, m)?)?;
    m.add_function(wrap_pyfunction!(gegenbauer_c, m)?)?;
    m.add_function(wrap_pyfunction!(jacobi_p, m)?)?;
    m.add_function(wrap_pyfunction!(expand_orthopoly, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_3j, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_6j, m)?)?;
    m.add_function(wrap_pyfunction!(clebsch_gordan, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_harmonic_y, m)?)?;
    m.add_function(wrap_pyfunction!(real_spherical_harmonic_y, m)?)?;
    Ok(())
}
