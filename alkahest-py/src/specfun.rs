//! PyO3 bindings for `alkahest_core::specfun` — Bessel, Airy, Riemann and
//! Hurwitz zeta, and polylogarithm functions (G11b).
//!
//! Refusals arrive as `alkahest.SpecialFunctionError` with a stable
//! `E-SPECFN-NNN` `.code` (a budget trip as `BudgetExceededError`).

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use alkahest_core::specfun::{self as core, AiryKind, BesselKind, HankelKind, SphericalBesselKind};

use crate::orthopoly::{pool_of, specfn_err};
use crate::{PyExpr, PyExprPool};
use alkahest_core::kernel::{ExprId, ExprPool};

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
/// ``f64``, balls and at complex ``s`` through Arb.  ``diff(zeta(s), s)`` is
/// ``zeta_prime(s)``; higher derivatives raise ``DiffError`` (``E-DIFF-001``).
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

/// Derivative of the Riemann zeta function, ``zeta'(s)``.
///
/// Exact at ``s = 0`` (``-log(2*pi)/2``), ``s = -1`` (``1/12 - log(Glaisher)``),
/// ``s = 2`` (``pi**2/6 (EulerGamma + log(2 pi) - 12 log(Glaisher))``) and the
/// negative even integers (``zeta'(-2) = -zeta(3)/(4 pi**2)``); the node
/// ``zeta_prime(s)`` elsewhere.  This is what ``diff(zeta(s), s)`` returns;
/// the second derivative has no node and is refused (``E-DIFF-001``).
/// Evaluates in ``f64``, balls (``interval_eval``) and at complex ``s``
/// through Arb (``acb_dirichlet_zeta_jet``).
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-006``) at the pole ``s = 1``.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); s = p.symbol("s")
/// >>> ak.diff(ak.zeta(s), s).value           # zeta_prime(s)
#[pyfunction]
fn zeta_prime(py: Python<'_>, s: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let (pool_py, a) = ids(py, "zeta_prime", &[s])?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::zeta_prime(a[0], pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

fn hankel_fn(
    py: Python<'_>,
    kind: HankelKind,
    nu: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    let (pool_py, a) = ids(py, kind.name(), &[nu, x])?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::hankel(kind, a[0], a[1], expand, pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Hankel function of the first kind ``H1_nu(x) = J_nu(x) + i*Y_nu(x)``
/// (DLMF 10.4; SymPy's ``hankel1``, mpmath's ``hankel1``).
///
/// Complex-valued.  A half-integer order with ``expand=True`` (the default)
/// gives the elementary form —
/// ``hankel1(1/2, x) = -I*sqrt(2)*exp(I*x)/(sqrt(pi)*sqrt(x))`` — and the
/// reflection ``H1_{-nu} = exp(I*pi*nu)*H1_nu`` is applied exactly;
/// ``H1_{-n} = (-1)**n H1_n`` for an integer order.  The node differentiates
/// in ``x`` (``(H1_{nu-1} - H1_{nu+1})/2``); a derivative in ``nu`` is refused
/// (``E-DIFF-001``).  There is no real ``f64`` / ball value (the value is
/// complex): ``evaluate(e, {x: 2.0})`` returns the complex number, through
/// Arb's principal branch.
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-004``) for an expansion past
/// order 200 (use ``expand=False``).
///
/// >>> import alkahest as ak
/// >>> from fractions import Fraction
/// >>> p = ak.ExprPool(); x = p.symbol("x")
/// >>> ak.hankel1(Fraction(1, 2), x)      # -I*sqrt(2)*pi^(-1/2)*x^(-1/2)*exp(I*x)
#[pyfunction]
#[pyo3(signature = (nu, x, *, expand = true))]
fn hankel1(
    py: Python<'_>,
    nu: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    hankel_fn(py, HankelKind::H1, nu, x, expand)
}

/// Hankel function of the second kind ``H2_nu(x) = J_nu(x) - i*Y_nu(x)``
/// (DLMF 10.4); see ``hankel1``.  For real ``x`` it is the complex conjugate
/// of ``hankel1`` at the same real order.
#[pyfunction]
#[pyo3(signature = (nu, x, *, expand = true))]
fn hankel2(
    py: Python<'_>,
    nu: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    expand: bool,
) -> PyResult<PyExpr> {
    hankel_fn(py, HankelKind::H2, nu, x, expand)
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

fn constant(pool: &Bound<'_, PyExprPool>, f: fn(&ExprPool) -> ExprId) -> PyExpr {
    let id = f(&pool.borrow().inner);
    PyExpr {
        id,
        pool: pool.clone().unbind(),
    }
}

/// The Euler–Mascheroni constant ``gamma = 0.5772...`` as the exact symbol
/// ``EulerGamma`` in ``pool`` (SymPy's ``EulerGamma``).
///
/// Prints and parses as ``EulerGamma`` (LaTeX ``\gamma``); every numeric
/// evaluator resolves it (``evaluate``, ``interval_eval`` at any precision,
/// ``compile_expr``, ``emit_c``), and no symbolic operation folds it into a
/// float: ``simplify(digamma(1))`` is ``-EulerGamma``.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool()
/// >>> ak.euler_gamma(p)                      # EulerGamma
#[pyfunction]
fn euler_gamma(pool: &Bound<'_, PyExprPool>) -> PyExpr {
    constant(pool, core::euler_gamma)
}

/// Catalan's constant ``G = sum (-1)**k/(2k+1)**2 = 0.9159...`` as the exact
/// symbol ``Catalan`` (LaTeX ``G``); ``trigamma(1/4) = pi**2 + 8*Catalan``.
#[pyfunction]
fn catalan(pool: &Bound<'_, PyExprPool>) -> PyExpr {
    constant(pool, core::catalan)
}

/// The Glaisher–Kinkelin constant ``A = 1.2824...`` as the exact symbol
/// ``Glaisher`` (LaTeX ``A``).
#[pyfunction]
fn glaisher(pool: &Bound<'_, PyExprPool>) -> PyExpr {
    constant(pool, core::glaisher)
}

/// The polygamma function ``polygamma(n, x)`` = ``psi^(n)(x)``, the
/// ``(n+1)``-th derivative of ``log(gamma(x))`` (DLMF 5.15; SymPy's
/// ``polygamma``).
///
/// ``polygamma(0, x)`` is ``digamma(x)`` and ``polygamma(1, x)`` is
/// ``trigamma(x)``.  Exact values use ``EulerGamma``, ``pi``, ``log``,
/// ``zeta`` and ``Catalan`` — never floats: ``polygamma(0, 1) = -EulerGamma``,
/// ``polygamma(1, 1) = pi**2/6``, ``polygamma(2, 1) = -2*zeta(3)``,
/// ``polygamma(0, 1/2) = -EulerGamma - 2*log(2)``, at any integer shift.
/// ``d/dx polygamma(n, x) = polygamma(n+1, x)``; a derivative in ``n`` is
/// refused (``E-DIFF-001``).  Evaluates in ``f64`` and balls through Arb.
///
/// Raises ``SpecialFunctionError``: ``E-SPECFN-007`` for a non-integer order,
/// ``E-SPECFN-003`` for a negative one, ``E-SPECFN-006`` at a pole
/// (``x = 0, -1, -2, ...``).
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool()
/// >>> ak.polygamma(2, p.integer(1))          # -2*zeta(3)
#[pyfunction]
fn polygamma(py: Python<'_>, n: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let (pool_py, a) = ids(py, "polygamma", &[x, n])?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::polygamma(a[1], a[0], pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

fn binary(
    py: Python<'_>,
    what: &str,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    f: fn(ExprId, ExprId, &ExprPool) -> Result<ExprId, core::SpecialFunctionError>,
) -> PyResult<PyExpr> {
    let (pool_py, ids) = ids(py, what, &[a, b])?;
    let id = {
        let pool = &pool_py.borrow(py).inner;
        f(ids[0], ids[1], pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// The Euler Beta function ``beta(a, b) = gamma(a)*gamma(b)/gamma(a + b)``
/// (DLMF 5.12.1; SymPy's ``beta``, mpmath's ``beta``).
///
/// Args: ``a``, ``b`` (Expr, int or Fraction; at least one an Expr).
/// Returns the exact value when the gamma ratio is exact (``beta(2, 3) =
/// 1/12``, ``beta(1/2, 1/2) = pi``), the rational function
/// ``(n-1)!/(a*(a+1)*...*(a+n-1))`` when one argument is a positive integer
/// ``n`` (``beta(a, 1) = 1/a``), and the node ``beta(a, b)`` otherwise.
/// Differentiates through ``digamma``; evaluates in ``f64`` and balls via Arb.
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-006``) when ``a`` or ``b`` is
/// a non-positive integer (a pole).
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); a = p.symbol("a")
/// >>> ak.beta(a, p.integer(2))      # 1/(a*(a + 1))
#[pyfunction]
fn beta(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    binary(py, "beta", a, b, core::beta)
}

/// The upper incomplete gamma function ``uppergamma(s, z)`` =
/// ``integral(t**(s-1)*exp(-t), (t, z, oo))`` (DLMF 8.2.2; SymPy's
/// ``uppergamma``, mpmath's ``gammainc(s, z)``).
///
/// Args: ``s`` (order), ``z`` (lower limit of the integral).  Returns the
/// elementary form for a positive integer ``s``
/// (``uppergamma(1, z) = exp(-z)``), the ``erfc`` form for a half-integer
/// ``s > 0`` (``uppergamma(1/2, z) = sqrt(pi)*erfc(sqrt(z))``),
/// ``gamma(s)`` at ``z = 0`` for a literal ``s > 0``, else the node.
/// ``d/dz uppergamma(s, z) = -z**(s-1)*exp(-z)``; a derivative in ``s`` is
/// refused (``E-DIFF-001``).  Real values via Arb.
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-006``) for ``z = 0`` with a
/// literal ``s <= 0``.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); z = p.symbol("z")
/// >>> ak.uppergamma(p.integer(2), z)     # (z + 1)*exp(-z)
#[pyfunction]
fn uppergamma(py: Python<'_>, s: &Bound<'_, PyAny>, z: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    binary(py, "uppergamma", s, z, core::uppergamma)
}

/// The lower incomplete gamma function ``lowergamma(s, z)`` =
/// ``integral(t**(s-1)*exp(-t), (t, 0, z))`` (DLMF 8.2.1; SymPy's
/// ``lowergamma``, mpmath's ``gammainc(s, 0, z)``).
///
/// Args: ``s`` (order), ``z`` (upper limit).  Elementary for a positive
/// integer ``s``, ``erf`` form for a half-integer ``s > 0``
/// (``lowergamma(1/2, z) = sqrt(pi)*erf(sqrt(z))``), ``0`` at ``z = 0`` for a
/// literal ``s > 0``; else the node.  ``uppergamma(s, z) + lowergamma(s, z) =
/// gamma(s)``.  ``d/dz lowergamma(s, z) = z**(s-1)*exp(-z)``.
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-006``) for a non-positive
/// integer ``s``.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); z = p.symbol("z")
/// >>> ak.lowergamma(p.integer(1), z)     # 1 - exp(-z)
#[pyfunction]
fn lowergamma(py: Python<'_>, s: &Bound<'_, PyAny>, z: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    binary(py, "lowergamma", s, z, core::lowergamma)
}

/// The generalized hypergeometric function ``pFq(a; b; z)`` (DLMF 16.2;
/// SymPy's ``hyper(a, b, z)``).
///
/// ``a`` and ``b`` are sequences of parameters (Expr, int, Fraction).  The
/// node prints as ``hyper(p, q, a1, ..., ap, b1, ..., bq, z)`` and parses
/// back; parameters common to both lists cancel, and an upper ``0`` or
/// ``z = 0`` gives ``1``.  ``d/dz`` shifts every parameter by one
/// (``prod(a)/prod(b) * hyper(a+1, b+1, z)``); a derivative in a parameter is
/// refused.  ``f64`` and ball values come from Arb (real values, the
/// principal branch for ``p = q + 1``); ``evaluate(..., mode="complex")``
/// gives complex values, with ``2F1`` continued analytically past
/// ``|z| = 1`` (on the cut ``[1, oo)`` the limit from below, as in mpmath;
/// other ``p = q + 1`` series decline for ``|z| >= 1``).  Use
/// ``expand_hyper`` for closed forms and ``hyper_transform`` for the
/// Euler/Pfaff/connection/quadratic/Kummer transformations.
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-006``) for a lower parameter
/// that is a non-positive integer (unless the series terminates first).
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); z = p.symbol("z")
/// >>> ak.expand_hyper(ak.hyper([1, 1], [2], z))   # -log(1 - z)/z
#[pyfunction]
fn hyper(
    py: Python<'_>,
    a: Vec<Bound<'_, PyAny>>,
    b: Vec<Bound<'_, PyAny>>,
    z: &Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    let mut all: Vec<&Bound<'_, PyAny>> = vec![z];
    all.extend(a.iter());
    all.extend(b.iter());
    let (pool_py, ids_) = ids(py, "hyper", &all)?;
    let p = a.len();
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::hyper(&ids_[1..1 + p], &ids_[1 + p..], ids_[0], pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Rewrite every ``hyper`` node into a polynomial (a terminating series) or a
/// closed form where one is known — SymPy's ``hyperexpand``:
/// ``0F0(;;z) = exp(z)``, ``1F0(a;;z) = (1-z)**(-a)``,
/// ``2F1(1,1;2;z) = -log(1-z)/z``, ``2F1(1/2,1;3/2;z) = atanh(sqrt(z))/sqrt(z)``,
/// ``2F1(1/2,1/2;3/2;z) = asin(sqrt(z))/sqrt(z)``, Gauss's
/// ``2F1(a,b;c;1) = gamma(c)gamma(c-a-b)/(gamma(c-a)gamma(c-b))`` when
/// ``c-a-b`` is a positive literal, ``1F1(1;2;z) = (exp(z)-1)/z``,
/// ``1F1(1/2;3/2;-w) = sqrt(pi)*erf(sqrt(w))/(2 sqrt(w))``, and ``0F1`` as
/// ``cosh``/``sinh`` or Bessel ``J``/``I``; the ``2F1(a, a+1/2; c; z)``
/// families for ``c = 1/2, 3/2, 2a, 2a+1``, ``1F1(a; a+1; -w) = a*w**(-a)*lowergamma(a, w)``, ``1F1(a; 2a; z)`` as Bessel
/// ``I``, and any node one Euler, Pfaff or Kummer transformation away from
/// one of these (``1F1(1; 3/2; z)`` becomes an ``erf``).  Other nodes are
/// kept.
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-004``) for a terminating
/// series of degree above 200.
#[pyfunction]
fn expand_hyper(py: Python<'_>, expr: PyRef<'_, PyExpr>) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::expand_hyper(expr.id, pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Rewrite one hypergeometric node by a classical transformation and return
/// the exact right-hand side (Gamma prefactors as ``gamma`` nodes, the
/// transformed series as new ``hyper`` nodes).
///
/// ``kind`` is one of (non-regularized A&S 15.3 / DLMF 15.8, principal
/// branches; each identity holds on the region given, which is part of the
/// result's contract):
///
/// * ``"euler"`` — ``(1-z)**(c-a-b) * 2F1(c-a, c-b; c; z)``, ``z`` off ``[1, oo)``;
/// * ``"pfaff"`` (alias ``"z/(z-1)"``) — ``(1-z)**(-a) * 2F1(a, c-b; c; z/(z-1))``;
///   ``"pfaff_b"`` the same in ``b``;
/// * ``"1-z"``, ``"1/z"``, ``"1/(1-z)"``, ``"1-1/z"`` — the two-term connection
///   formulas with Gamma coefficients (A&S 15.3.6-15.3.9); ``1-z`` and
///   ``1-1/z`` hold for ``z`` off ``(-oo, 0] U [1, oo)``, ``1/z`` and
///   ``1/(1-z)`` for ``z`` off ``[0, oo)``; a symbolic ``c-a-b`` / ``a-b`` is
///   taken to be a non-integer;
/// * ``"quadratic"`` — the quadratic transformation of the family the
///   parameters are in: ``c = 2b`` (DLMF 15.8.13, ``z`` off ``[1, oo)``),
///   ``c = a-b+1`` (15.8.15, ``|z| < 1``), ``2c = a+b+1`` (15.8.18,
///   ``Re z < 1/2``);
/// * ``"kummer"`` (``1F1``) — ``exp(z) * 1F1(b-a; b; -z)``, every ``z``.
///
/// Raises ``SpecialFunctionError`` (``E-SPECFN-009``) when ``expr`` is not a
/// ``2F1`` (``1F1`` for ``"kummer"``) node, for the logarithmic case of a
/// two-term formula (``c-a-b`` or ``a-b`` an integer), for ``c`` a
/// non-positive integer, and for parameters in no quadratic family;
/// ``ValueError`` for an unknown ``kind``.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); z = p.symbol("z")
/// >>> ak.hyper_transform(ak.hyper([1, 1], [3], z), "euler")   # (1-z) * 2F1(2, 2; 3; z)
#[pyfunction]
fn hyper_transform(py: Python<'_>, expr: PyRef<'_, PyExpr>, kind: &str) -> PyResult<PyExpr> {
    let Some(t) = core::HyperTransformation::from_name(kind) else {
        let names: Vec<&str> = core::HyperTransformation::ALL
            .iter()
            .map(|t| t.name())
            .collect();
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hyper_transform: unknown kind {kind:?}; expected one of {}",
            names.join(", ")
        )));
    };
    let pool_py = expr.pool.clone_ref(py);
    let id = {
        let pool = &pool_py.borrow(py).inner;
        core::hyper_transform(expr.id, t, pool).map_err(specfn_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hyper_transform, m)?)?;
    m.add_function(wrap_pyfunction!(euler_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(catalan, m)?)?;
    m.add_function(wrap_pyfunction!(glaisher, m)?)?;
    m.add_function(wrap_pyfunction!(polygamma, m)?)?;
    m.add_function(wrap_pyfunction!(hyper, m)?)?;
    m.add_function(wrap_pyfunction!(beta, m)?)?;
    m.add_function(wrap_pyfunction!(uppergamma, m)?)?;
    m.add_function(wrap_pyfunction!(lowergamma, m)?)?;
    m.add_function(wrap_pyfunction!(expand_hyper, m)?)?;
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
    m.add_function(wrap_pyfunction!(zeta_prime, m)?)?;
    m.add_function(wrap_pyfunction!(hankel1, m)?)?;
    m.add_function(wrap_pyfunction!(hankel2, m)?)?;
    m.add_function(wrap_pyfunction!(hurwitz_zeta, m)?)?;
    m.add_function(wrap_pyfunction!(polylog, m)?)?;
    m.add_function(wrap_pyfunction!(expand_special, m)?)?;
    Ok(())
}
