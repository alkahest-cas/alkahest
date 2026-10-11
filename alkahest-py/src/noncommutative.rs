//! Python bindings for noncommutative algebra (G10c): expansion, word
//! rewriting, commutators, the Pauli algebra, ladder operators and Dirac
//! traces. Refusals arrive as `alkahest.NcError` (`E-NC-001` … `E-NC-007`).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use alkahest_core::algebra::nc_poly::{
    self as core_nc, GammaSlot, LadderMode, NcError, DEFAULT_MAX_STEPS,
};
use alkahest_core::ExprId;

use crate::{guard_depth, PyExpr, PyExprPool, PyRewriteRule};

pyo3::create_exception!(alkahest, PyNcError, crate::PyAlkahestError);

fn nc_err(e: NcError) -> PyErr {
    Python::with_gil(|py| {
        let exc_type = py.get_type_bound::<PyNcError>();
        crate::make_structured_err(py, &exc_type, &e)
    })
}

fn wrap(py: Python<'_>, id: ExprId, pool: &Py<PyExprPool>) -> PyExpr {
    PyExpr {
        id,
        pool: pool.clone_ref(py),
    }
}

fn same_pool_or_err(a: &Py<PyExprPool>, b: &Py<PyExprPool>, what: &str) -> PyResult<()> {
    if a.is(b) {
        Ok(())
    } else {
        Err(crate::cross_pool_err(format!(
            "[E-POOL-001] {what} belongs to a different ExprPool than the expression"
        )))
    }
}

/// ``nc_expand(expr) -> Expr``
///
/// Expand ``expr`` as a polynomial in noncommutative symbols: products
/// distribute *keeping factor order*, non-negative integer powers of
/// noncommutative factors unroll, equal words collect and their commutative
/// coefficients are simplified.
///
/// Example
/// -------
/// >>> sx, sy = (p.symbol(n, commutative=False) for n in ("sx", "sy"))
/// >>> nc_expand((sx + sy)**2)     # sx^2 + sx*sy + sy*sx + sy^2
///
/// Raises ``NcError`` ``E-NC-002`` past the size caps.
#[pyfunction]
#[pyo3(name = "nc_expand")]
fn py_nc_expand(py: Python<'_>, expr: PyRef<'_, PyExpr>) -> PyResult<PyExpr> {
    let pool = expr.pool.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
    let id = core_nc::nc_expand(expr.id, &pool.inner).map_err(nc_err)?;
    Ok(wrap(py, id, &expr.pool))
}

/// ``nc_terms(expr) -> list[tuple[Expr, list[Expr]]]``
///
/// The terms ``(coefficient, word)`` of ``expr`` read as a noncommutative
/// polynomial (see :func:`nc_expand`): ``coefficient`` is commutative and
/// ``word`` is the ordered list of noncommutative factors (empty for the
/// scalar part).
#[pyfunction]
#[pyo3(name = "nc_terms")]
fn py_nc_terms(py: Python<'_>, expr: PyRef<'_, PyExpr>) -> PyResult<PyObject> {
    let terms = {
        let pool = expr.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        core_nc::nc_terms(expr.id, &pool.inner).map_err(nc_err)?
    };
    let out = PyList::empty_bound(py);
    for (c, w) in terms {
        let word = PyList::empty_bound(py);
        for a in w {
            word.append(wrap(py, a, &expr.pool).into_py(py))?;
        }
        let t = PyTuple::new_bound(py, [wrap(py, c, &expr.pool).into_py(py), word.into_py(py)]);
        out.append(t)?;
    }
    Ok(out.into_py(py))
}

/// ``nc_rewrite(expr, rules, *, max_steps=200000) -> Expr``
///
/// Rewrite ``expr`` to a normal form with literal word rules made by
/// :func:`make_rule`: every occurrence of a rule's left-hand word *inside* a
/// longer word is replaced (``make_rule(X*P, P*X + I*hbar)`` rewrites
/// ``X*P*P``), the result is re-expanded, and the process repeats until no
/// rule applies.
///
/// Each left-hand side must be a single product of noncommutative factors
/// with coefficient 1; symbols in these rules are literal (no wildcards).
///
/// Raises ``NcError``: ``E-NC-003`` for a malformed left-hand side,
/// ``E-NC-001`` when no normal form is reached within ``max_steps`` rewrites
/// (non-terminating rules), ``E-NC-002`` past the size caps.
#[pyfunction]
#[pyo3(name = "nc_rewrite", signature = (expr, rules, *, max_steps = DEFAULT_MAX_STEPS))]
fn py_nc_rewrite(
    py: Python<'_>,
    expr: PyRef<'_, PyExpr>,
    rules: Vec<PyRef<'_, PyRewriteRule>>,
    max_steps: usize,
) -> PyResult<PyExpr> {
    let mut pairs = Vec::with_capacity(rules.len());
    for r in &rules {
        same_pool_or_err(&r.pool, &expr.pool, "a rule")?;
        pairs.push((r.inner.lhs.root, r.inner.rhs));
    }
    let pool = expr.pool.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
    for &(l, r) in &pairs {
        guard_depth(&pool.inner, l)?;
        guard_depth(&pool.inner, r)?;
    }
    let id = core_nc::nc_rewrite(expr.id, &pairs, &pool.inner, max_steps).map_err(nc_err)?;
    Ok(wrap(py, id, &expr.pool))
}

fn commutator_like(
    py: Python<'_>,
    a: &PyExpr,
    b: &PyExpr,
    rules: Option<Vec<PyRef<'_, PyRewriteRule>>>,
    anti: bool,
) -> PyResult<PyExpr> {
    same_pool_or_err(&b.pool, &a.pool, "`b`")?;
    let mut pairs = Vec::new();
    for r in rules.iter().flatten() {
        same_pool_or_err(&r.pool, &a.pool, "a rule")?;
        pairs.push((r.inner.lhs.root, r.inner.rhs));
    }
    let pool = a.pool.borrow(py);
    guard_depth(&pool.inner, a.id)?;
    guard_depth(&pool.inner, b.id)?;
    let p = &pool.inner;
    let c = if anti {
        core_nc::anticommutator(a.id, b.id, p)
    } else {
        core_nc::commutator(a.id, b.id, p)
    }
    .map_err(nc_err)?;
    let id = if pairs.is_empty() {
        c
    } else {
        core_nc::nc_rewrite(c, &pairs, p, DEFAULT_MAX_STEPS).map_err(nc_err)?
    };
    Ok(wrap(py, id, &a.pool))
}

/// ``commutator(a, b, rules=None) -> Expr``
///
/// ``[a, b] = a*b - b*a``, expanded as a noncommutative polynomial. With
/// ``rules`` (from :func:`make_rule`) the result is rewritten to normal form
/// with :func:`nc_rewrite`, e.g. ``commutator(X, P**2, [make_rule(X*P, P*X +
/// I*hbar)])`` gives ``2*I*hbar*P``.
#[pyfunction]
#[pyo3(name = "commutator", signature = (a, b, rules = None))]
fn py_commutator(
    py: Python<'_>,
    a: PyRef<'_, PyExpr>,
    b: PyRef<'_, PyExpr>,
    rules: Option<Vec<PyRef<'_, PyRewriteRule>>>,
) -> PyResult<PyExpr> {
    commutator_like(py, &a, &b, rules, false)
}

/// ``anticommutator(a, b, rules=None) -> Expr``
///
/// ``{a, b} = a*b + b*a``, expanded; ``rules`` as for :func:`commutator`.
#[pyfunction]
#[pyo3(name = "anticommutator", signature = (a, b, rules = None))]
fn py_anticommutator(
    py: Python<'_>,
    a: PyRef<'_, PyExpr>,
    b: PyRef<'_, PyExpr>,
    rules: Option<Vec<PyRef<'_, PyRewriteRule>>>,
) -> PyResult<PyExpr> {
    commutator_like(py, &a, &b, rules, true)
}

fn parse_modes(
    py: Python<'_>,
    pool: &Py<PyExprPool>,
    bosons: Option<&Bound<'_, PyAny>>,
    fermions: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<LadderMode>> {
    let mut modes = Vec::new();
    for (spec, fermionic) in [(bosons, false), (fermions, true)] {
        let Some(spec) = spec else { continue };
        // A single pair, or a list of pairs.
        let pairs: Vec<Bound<'_, PyAny>> = match spec.extract::<(PyRef<PyExpr>, PyRef<PyExpr>)>() {
            Ok(_) => vec![spec.clone()],
            Err(_) => spec
                .iter()
                .map_err(|_| {
                    PyTypeError::new_err(
                        "modes must be an (annihilator, creator) pair or a list of pairs",
                    )
                })?
                .collect::<PyResult<Vec<_>>>()?,
        };
        for pair in pairs {
            let (a, ad): (PyRef<PyExpr>, PyRef<PyExpr>) = pair.extract().map_err(|_| {
                PyTypeError::new_err("each mode must be an (annihilator, creator) pair of Expr")
            })?;
            same_pool_or_err(&a.pool, pool, "a ladder operator")?;
            same_pool_or_err(&ad.pool, pool, "a ladder operator")?;
            modes.push(LadderMode {
                annihilator: a.id,
                creator: ad.id,
                fermionic,
            });
        }
    }
    let _ = py;
    if modes.is_empty() {
        return Err(PyValueError::new_err(
            "give at least one mode: bosons=(a, adag) and/or fermions=(c, cdag)",
        ));
    }
    Ok(modes)
}

/// ``normal_order(expr, bosons=None, fermions=None) -> Expr``
///
/// Normal-order ``expr`` with the canonical commutation relations: every
/// creation operator moves left of every annihilation operator using
/// ``[a, a†] = 1`` for bosonic modes and ``{c, c†} = 1``, ``c**2 = 0`` for
/// fermionic ones (Wick's theorem: the contractions appear as the extra
/// terms). Operators of different modes are ordered by their position in the
/// mode lists (bosons first), with a sign for each swap of two fermionic
/// operators.
///
/// Args:
///     expr: an expression in noncommutative symbols.
///     bosons: an ``(a, adag)`` pair or a list of them.
///     fermions: a ``(c, cdag)`` pair or a list of them.
///
/// Each operator must be a noncommutative symbol (``commutative=False``).
/// Other noncommutative factors stay in place and block reordering.
///
/// Example
/// -------
/// >>> a, ad = p.symbol("a", commutative=False), p.symbol("ad", commutative=False)
/// >>> normal_order(a * ad**3, bosons=(a, ad))     # ad^3*a + 3*ad^2
///
/// Raises ``NcError`` ``E-NC-004`` for a malformed mode list.
#[pyfunction]
#[pyo3(name = "normal_order", signature = (expr, bosons = None, fermions = None))]
fn py_normal_order(
    py: Python<'_>,
    expr: PyRef<'_, PyExpr>,
    bosons: Option<&Bound<'_, PyAny>>,
    fermions: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyExpr> {
    let modes = parse_modes(py, &expr.pool, bosons, fermions)?;
    let pool = expr.pool.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
    let id = core_nc::normal_order(expr.id, &modes, &pool.inner).map_err(nc_err)?;
    Ok(wrap(py, id, &expr.pool))
}

/// ``vacuum_expectation(expr, bosons=None, fermions=None) -> Expr``
///
/// ``<0| expr |0>``: the scalar part of :func:`normal_order` ``(expr)``.
/// E.g. ``vacuum_expectation(a*a*ad*ad, bosons=(a, ad)) == 2``.
///
/// Raises ``NcError`` ``E-NC-007`` when ``expr`` contains a noncommutative
/// factor that is not one of the listed ladder operators.
#[pyfunction]
#[pyo3(name = "vacuum_expectation", signature = (expr, bosons = None, fermions = None))]
fn py_vacuum_expectation(
    py: Python<'_>,
    expr: PyRef<'_, PyExpr>,
    bosons: Option<&Bound<'_, PyAny>>,
    fermions: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyExpr> {
    let modes = parse_modes(py, &expr.pool, bosons, fermions)?;
    let pool = expr.pool.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
    let id = core_nc::vacuum_expectation(expr.id, &modes, &pool.inner).map_err(nc_err)?;
    Ok(wrap(py, id, &expr.pool))
}

/// ``gamma_trace(slots, pool=None) -> Expr``
///
/// Trace of a product of Dirac matrices in four dimensions, metric
/// ``diag(+1, -1, -1, -1)``, ``{γ^μ, γ^ν} = 2 g^{μν}``, ``tr 1 = 4``,
/// ``γ^5 = i γ^0 γ^1 γ^2 γ^3``.
///
/// Args:
///     slots: a list whose entries are ``0``–``3`` (``γ^0`` … ``γ^3``),
///         ``5`` (``γ^5``), or a commutative ``Expr`` ``p`` standing for the
///         slashed vector ``p̸ = p_μ γ^μ`` (equally, a free Lorentz index).
///     pool: the pool for the result when ``slots`` holds no ``Expr``.
///
/// Returns:
///     A polynomial in ``dot(p, q)`` (``p·q``; ``dot(k, p) = p^k`` for an
///     explicit index ``k``) and ``levi_civita(a, b, c, d)`` ``= ε^{μνρσ}
///     a_μ b_ν c_ρ d_σ`` with ``ε^{0123} = +1``, so ``tr(γ^5 a̸ b̸ c̸ d̸) =
///     -4i·levi_civita(a, b, c, d)``. With ``γ^5`` and more than four other
///     slots, ``γ^5`` is expanded as ``iγ^0γ^1γ^2γ^3`` and the answer uses
///     explicit components ``dot(k, p)``.
///
/// Example
/// -------
/// >>> gamma_trace([p, q, k, l])   # 4*dot(p,q)*dot(k,l) - 4*dot(k,p)*dot(l,q) + ...
/// >>> gamma_trace([5, 0, 1, 2, 3], pool=pool)   # -4*I
///
/// Raises ``NcError``: ``E-NC-005`` for an invalid slot, ``E-NC-006`` for
/// more than 14 slots to expand.
#[pyfunction]
#[pyo3(name = "gamma_trace", signature = (slots, pool = None))]
fn py_gamma_trace(
    py: Python<'_>,
    slots: &Bound<'_, PyList>,
    pool: Option<Py<PyExprPool>>,
) -> PyResult<PyExpr> {
    let mut pool_py: Option<Py<PyExprPool>> = pool;
    let mut parsed = Vec::with_capacity(slots.len());
    for item in slots.iter() {
        if let Ok(e) = item.extract::<PyRef<PyExpr>>() {
            match &pool_py {
                Some(p) => same_pool_or_err(&e.pool, p, "a gamma_trace slot")?,
                None => pool_py = Some(e.pool.clone_ref(py)),
            }
            parsed.push(GammaSlot::Vector(e.id));
        } else if let Ok(k) = item.extract::<i64>() {
            parsed.push(match k {
                0..=3 => GammaSlot::Index(k as u8),
                5 => GammaSlot::Gamma5,
                _ => {
                    return Err(nc_err(NcError::InvalidGamma {
                        reason: format!("index {k} is not 0, 1, 2, 3 or 5"),
                    }))
                }
            });
        } else {
            return Err(PyTypeError::new_err(
                "gamma_trace slots must be ints (0-3, 5) or Expr vector labels",
            ));
        }
    }
    let pool_py = pool_py.ok_or_else(|| {
        PyValueError::new_err("gamma_trace: pass pool=... when no slot is an Expr")
    })?;
    let id = {
        let pool = pool_py.borrow(py);
        core_nc::gamma_trace(&parsed, &pool.inner).map_err(nc_err)?
    };
    Ok(PyExpr { id, pool: pool_py })
}

/// Pauli reduction used by ``simplify_pauli`` for noncommutative inputs.
pub(crate) fn pauli_reduce(pool: &alkahest_core::kernel::ExprPool, id: ExprId) -> PyResult<ExprId> {
    core_nc::pauli_reduce(id, pool).map_err(nc_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NcError", m.py().get_type_bound::<PyNcError>())?;
    m.add_function(wrap_pyfunction!(py_nc_expand, m)?)?;
    m.add_function(wrap_pyfunction!(py_nc_terms, m)?)?;
    m.add_function(wrap_pyfunction!(py_nc_rewrite, m)?)?;
    m.add_function(wrap_pyfunction!(py_commutator, m)?)?;
    m.add_function(wrap_pyfunction!(py_anticommutator, m)?)?;
    m.add_function(wrap_pyfunction!(py_normal_order, m)?)?;
    m.add_function(wrap_pyfunction!(py_vacuum_expectation, m)?)?;
    m.add_function(wrap_pyfunction!(py_gamma_trace, m)?)?;
    Ok(())
}
