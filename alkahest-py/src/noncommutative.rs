//! Python bindings for noncommutative algebra (G10c): expansion, word
//! rewriting, commutators, the Pauli algebra, ladder operators and Dirac
//! traces. Refusals arrive as `alkahest.NcError` (`E-NC-001` … `E-NC-007`).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use alkahest_core::algebra::nc_poly::{
    self as core_nc, GammaSlot, LadderMode, NcError, DEFAULT_MAX_STEPS,
};
use alkahest_core::algebra::{gamma_dim as core_gd, lie as core_lie};
use alkahest_core::pattern::Pattern;
use alkahest_core::{ExprId, PatternRule};

use crate::{guard_depth, PyExpr, PyExprPool, PyRewriteRule};

pyo3::create_exception!(alkahest, PyNcError, crate::PyAlkahestError);

pub(crate) fn nc_err(e: NcError) -> PyErr {
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

/// `nc_rewrite` reads rule sides as literal words; a sequence-wildcard rule
/// would silently treat its wildcard as a literal symbol, so refuse it.
fn refuse_sequence_rule(r: &PyRewriteRule) -> PyResult<()> {
    if r.inner.sequence_wildcards().is_empty() {
        Ok(())
    } else {
        Err(nc_err(NcError::InvalidRule {
            reason: "rules with sequence wildcards (make_rule(..., sequence=[...])) apply through simplify_with, not as literal word rules".into(),
        }))
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
        refuse_sequence_rule(r)?;
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
        refuse_sequence_rule(r)?;
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
/// With ``dim`` (an integer or a symbolic ``d``) the trace is taken in
/// ``dim`` dimensions (``tr 1 = 4``, as in FORM / FeynCalc): every slot is a
/// commutative ``Expr`` label, ``indices`` lists the labels that are Lorentz
/// indices, and an index occurring twice is summed with ``g^μ_μ = dim``::
///
///     gamma_trace([mu, mu], dim=d, indices=[mu])             # 4*d
///     gamma_trace([mu, nu, mu, nu], dim=d, indices=[mu, nu]) # 4*d*(2 - d)
///
/// An index occurring once stays free (``dot(μ, ν) = g^{μν}``); contract
/// products of traces with :func:`lorentz_contract`. ``γ^5`` and explicit
/// components ``0``-``3`` are four-dimensional and are refused here.
///
/// Raises ``NcError``: ``E-NC-005`` for an invalid slot (or ``γ^5`` /
/// an explicit component with ``dim``), ``E-NC-006`` for more than 14 slots
/// to expand.
#[pyfunction]
#[pyo3(name = "gamma_trace", signature = (slots, pool = None, *, dim = None, indices = None))]
fn py_gamma_trace(
    py: Python<'_>,
    slots: &Bound<'_, PyList>,
    pool: Option<Py<PyExprPool>>,
    dim: Option<PyRef<'_, PyExpr>>,
    indices: Option<Vec<PyRef<'_, PyExpr>>>,
) -> PyResult<PyExpr> {
    if let Some(dim) = dim {
        let (labels, idx) = dim_labels(py, slots, indices, &dim)?;
        let id = {
            let pool = dim.pool.borrow(py);
            core_gd::gamma_trace_dim(&labels, &idx, dim.id, &pool.inner).map_err(nc_err)?
        };
        return Ok(wrap(py, id, &dim.pool));
    }
    if indices.is_some() {
        return Err(PyValueError::new_err(
            "gamma_trace: `indices` needs `dim` (contraction of dummy indices is the d-dimensional mode)",
        ));
    }
    let mut pool_py: Option<Py<PyExprPool>> = pool;
    let mut parsed = Vec::with_capacity(slots.len());
    for item in slots.iter() {
        if let Ok(e) = item.extract::<PyRef<PyExpr>>() {
            match &pool_py {
                Some(p) => same_pool_or_err(&e.pool, p, "a gamma_trace slot")?,
                None => pool_py = Some(e.pool.clone_ref(py)),
            }
            guard_depth(&e.pool.borrow(py).inner, e.id)?;
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

/// Labels and indices of a d-dimensional gamma call, checked against the
/// pool of `dim`. Integers (explicit components, γ⁵) are refused.
fn dim_labels(
    py: Python<'_>,
    slots: &Bound<'_, PyList>,
    indices: Option<Vec<PyRef<'_, PyExpr>>>,
    dim: &PyRef<'_, PyExpr>,
) -> PyResult<(Vec<ExprId>, Vec<ExprId>)> {
    let mut labels = Vec::with_capacity(slots.len());
    for item in slots.iter() {
        if let Ok(e) = item.extract::<PyRef<PyExpr>>() {
            same_pool_or_err(&e.pool, &dim.pool, "a gamma slot")?;
            guard_depth(&e.pool.borrow(py).inner, e.id)?;
            labels.push(e.id);
        } else if item.extract::<i64>().is_ok() {
            return Err(nc_err(NcError::InvalidGamma {
                reason: "explicit components 0-3 and γ⁵ exist only in four dimensions: omit `dim` for the 4D trace (γ⁵ has no anticommuting d-dimensional extension; use a symbolic label for γ^μ)".into(),
            }));
        } else {
            return Err(PyTypeError::new_err(
                "d-dimensional gamma slots must be Expr labels",
            ));
        }
    }
    let mut idx = Vec::new();
    for e in indices.iter().flatten() {
        same_pool_or_err(&e.pool, &dim.pool, "a Lorentz index")?;
        idx.push(e.id);
    }
    guard_depth(&dim.pool.borrow(py).inner, dim.id)?;
    Ok((labels, idx))
}

/// ``gamma_contract(slots, indices, dim) -> list[tuple[Expr, list[Expr]]]``
///
/// Contract the dummy Lorentz indices of a *product* of gamma matrices in
/// ``dim`` dimensions using only ``{γ^μ, γ^ν} = 2 g^{μν}``. ``slots`` are
/// commutative labels; a label listed in ``indices`` that occurs twice is a
/// summed index. Returns terms ``(coefficient, labels)`` meaning
/// ``coefficient * γ(labels[0]) γ(labels[1]) ...``, with no index repeated:
///
/// * ``γ^μ γ_μ = d`` -> ``[(d, [])]``
/// * ``γ^μ γ^ν γ_μ = (2 - d) γ^ν``
/// * ``γ^μ γ^ν γ^ρ γ_μ = 2 γ^ρ γ^ν + (d - 2) γ^ν γ^ρ``  (``= 4 g^{νρ} - (4-d) γ^ν γ^ρ``)
///
/// Raises ``NcError`` ``E-NC-005`` for non-commutative labels or an index
/// repeated more than twice, ``E-NC-006`` past 14 slots.
#[pyfunction]
#[pyo3(name = "gamma_contract")]
fn py_gamma_contract(
    py: Python<'_>,
    slots: &Bound<'_, PyList>,
    indices: Vec<PyRef<'_, PyExpr>>,
    dim: PyRef<'_, PyExpr>,
) -> PyResult<PyObject> {
    let (labels, idx) = dim_labels(py, slots, Some(indices), &dim)?;
    let terms = {
        let pool = dim.pool.borrow(py);
        core_gd::gamma_contract(&labels, &idx, dim.id, &pool.inner).map_err(nc_err)?
    };
    let out = PyList::empty_bound(py);
    for (c, w) in terms {
        let word = PyList::empty_bound(py);
        for a in w {
            word.append(wrap(py, a, &dim.pool).into_py(py))?;
        }
        out.append(PyTuple::new_bound(
            py,
            [wrap(py, c, &dim.pool).into_py(py), word.into_py(py)],
        ))?;
    }
    Ok(out.into_py(py))
}

/// ``lorentz_contract(expr, indices, dim) -> Expr``
///
/// Contract the dummy Lorentz indices ``indices`` of a polynomial in
/// ``dot(a, b)`` factors (``dot(μ, μ) -> d``; ``dot(μ, a)*dot(μ, b) ->
/// dot(a, b)``). An index occurring once is free and kept. Use it on products
/// of traces, e.g. ``tr(γ^μγ^νγ^ργ^σ) tr(γ_μγ_νγ_ργ_σ) = 16 d (3d - 2)``::
///
///     t = gamma_trace([m, n, r, s], dim=d)          # free indices
///     lorentz_contract(t * t, [m, n, r, s], d)       # 48*d**2 - 32*d  (up to form)
///
/// Raises ``NcError`` ``E-NC-005`` for an index occurring more than twice or
/// inside a factor that is not a ``dot`` (e.g. ``levi_civita``).
#[pyfunction]
#[pyo3(name = "lorentz_contract")]
fn py_lorentz_contract(
    py: Python<'_>,
    expr: PyRef<'_, PyExpr>,
    indices: Vec<PyRef<'_, PyExpr>>,
    dim: PyRef<'_, PyExpr>,
) -> PyResult<PyExpr> {
    same_pool_or_err(&dim.pool, &expr.pool, "`dim`")?;
    let mut idx = Vec::new();
    for e in &indices {
        same_pool_or_err(&e.pool, &expr.pool, "a Lorentz index")?;
        idx.push(e.id);
    }
    let pool = expr.pool.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
    guard_depth(&pool.inner, dim.id)?;
    let id = core_gd::lorentz_contract(expr.id, &idx, dim.id, &pool.inner).map_err(nc_err)?;
    Ok(wrap(py, id, &expr.pool))
}

// ---------------------------------------------------------------------------
// Lie algebras
// ---------------------------------------------------------------------------

/// A Lie algebra given by ``[T_a, T_b] = i Σ_c f_{abc} T_c`` on noncommutative
/// generators (see :func:`lie_algebra`).
#[pyclass(name = "LieAlgebra", module = "alkahest")]
pub(crate) struct PyLieAlgebra {
    inner: core_lie::LieAlgebra,
    pool: Py<PyExprPool>,
}

impl PyLieAlgebra {
    fn check(&self, e: &PyExpr, what: &str) -> PyResult<()> {
        same_pool_or_err(&e.pool, &self.pool, what)
    }
}

#[pymethods]
impl PyLieAlgebra {
    /// The generators, in table order.
    #[getter]
    fn generators(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.inner
            .generators()
            .iter()
            .map(|&g| wrap(py, g, &self.pool))
            .collect()
    }

    /// The ordering rules ``T_a T_b -> T_b T_a + [T_a, T_b]`` (``a > b``) as
    /// ``RewriteRule`` objects, usable with :func:`nc_rewrite`.
    #[getter]
    fn rules(&self, py: Python<'_>) -> Vec<PyRewriteRule> {
        self.inner
            .rules()
            .iter()
            .map(|&(l, r)| PyRewriteRule {
                inner: PatternRule::new(Pattern::from_expr(l), r),
                pool: self.pool.clone_ref(py),
            })
            .collect()
    }

    /// ``[a, b]`` reduced with the table: ``commutator(T_a, T_b)`` is
    /// ``i Σ f_abc T_c``; for polynomials the result is in normal order
    /// (generators in increasing index order).
    fn commutator(
        &self,
        py: Python<'_>,
        a: PyRef<'_, PyExpr>,
        b: PyRef<'_, PyExpr>,
    ) -> PyResult<PyExpr> {
        self.check(&a, "`a`")?;
        self.check(&b, "`b`")?;
        let pool = self.pool.borrow(py);
        guard_depth(&pool.inner, a.id)?;
        guard_depth(&pool.inner, b.id)?;
        let id = self
            .inner
            .commutator(a.id, b.id, &pool.inner)
            .map_err(nc_err)?;
        Ok(wrap(py, id, &self.pool))
    }

    /// ``{a, b} = a b + b a`` reduced with the table.
    fn anticommutator(
        &self,
        py: Python<'_>,
        a: PyRef<'_, PyExpr>,
        b: PyRef<'_, PyExpr>,
    ) -> PyResult<PyExpr> {
        self.check(&a, "`a`")?;
        self.check(&b, "`b`")?;
        let pool = self.pool.borrow(py);
        guard_depth(&pool.inner, a.id)?;
        guard_depth(&pool.inner, b.id)?;
        let id = self
            .inner
            .anticommutator(a.id, b.id, &pool.inner)
            .map_err(nc_err)?;
        Ok(wrap(py, id, &self.pool))
    }

    /// Rewrite ``expr`` to normal order: generators in increasing table order
    /// (Poincaré–Birkhoff–Witt basis), commutator terms added.
    fn normal_order(&self, py: Python<'_>, expr: PyRef<'_, PyExpr>) -> PyResult<PyExpr> {
        self.check(&expr, "`expr`")?;
        let pool = self.pool.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        let id = self
            .inner
            .normal_order(expr.id, &pool.inner)
            .map_err(nc_err)?;
        Ok(wrap(py, id, &self.pool))
    }

    /// The quadratic Casimir ``Σ g^{ab} T_a T_b``. ``inverse_metric`` is an
    /// ``n × n`` nested list ``g^{ab}`` (ints or ``Expr``); the default is the
    /// identity, i.e. ``Σ T_a²``, correct for the su(n) tables of this module
    /// (basis orthonormal under the Killing form).
    #[pyo3(signature = (inverse_metric = None))]
    fn casimir(
        &self,
        py: Python<'_>,
        inverse_metric: Option<Vec<Vec<Bound<'_, PyAny>>>>,
    ) -> PyResult<PyExpr> {
        let pool = self.pool.borrow(py);
        let metric = match inverse_metric {
            None => None,
            Some(rows) => {
                let mut m = Vec::with_capacity(rows.len());
                for row in rows {
                    let mut r = Vec::with_capacity(row.len());
                    for v in row {
                        r.push(coerce_value(&v, &self.pool, &pool.inner)?);
                    }
                    m.push(r);
                }
                Some(m)
            }
        };
        let id = self
            .inner
            .casimir(metric.as_deref(), &pool.inner)
            .map_err(nc_err)?;
        Ok(wrap(py, id, &self.pool))
    }

    /// Check the Jacobi identity for every triple; raises ``NcError``
    /// ``E-NC-008`` if it fails or cannot be verified symbolically.
    fn verify_jacobi(&self, py: Python<'_>) -> PyResult<()> {
        let pool = self.pool.borrow(py);
        self.inner.verify_jacobi(&pool.inner).map_err(nc_err)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        let g: Vec<String> = self
            .inner
            .generators()
            .iter()
            .map(|&g| pool.inner.display(g).to_string())
            .collect();
        format!("LieAlgebra(generators=[{}])", g.join(", "))
    }
}

/// An `Expr` of `pool`, or an integer, as an expression id.
fn coerce_value(
    v: &Bound<'_, PyAny>,
    pool_py: &Py<PyExprPool>,
    pool: &alkahest_core::kernel::ExprPool,
) -> PyResult<ExprId> {
    if let Ok(e) = v.extract::<PyRef<PyExpr>>() {
        same_pool_or_err(&e.pool, pool_py, "a structure constant")?;
        guard_depth(pool, e.id)?;
        return Ok(e.id);
    }
    if let Ok(n) = v.extract::<i64>() {
        return Ok(pool.integer(n));
    }
    Err(PyTypeError::new_err(
        "structure constants must be Expr or int",
    ))
}

fn build_lie(
    py: Python<'_>,
    pool_py: Py<PyExprPool>,
    generators: Vec<ExprId>,
    entries: Vec<core_lie::StructureEntry>,
    check: bool,
) -> PyResult<PyLieAlgebra> {
    let inner = {
        let pool = pool_py.borrow(py);
        core_lie::LieAlgebra::new(generators, &entries, check, &pool.inner).map_err(nc_err)?
    };
    Ok(PyLieAlgebra {
        inner,
        pool: pool_py,
    })
}

fn extract_generators(
    py: Python<'_>,
    generators: &[PyRef<'_, PyExpr>],
) -> PyResult<(Py<PyExprPool>, Vec<ExprId>)> {
    let first = generators
        .first()
        .ok_or_else(|| PyValueError::new_err("give at least one generator"))?;
    let pool = first.pool.clone_ref(py);
    let mut ids = Vec::with_capacity(generators.len());
    for g in generators {
        same_pool_or_err(&g.pool, &pool, "a generator")?;
        ids.push(g.id);
    }
    Ok((pool, ids))
}

/// ``lie_algebra(generators, structure_constants, *, check_jacobi=True) -> LieAlgebra``
///
/// A Lie algebra on noncommutative ``generators`` with
/// ``[T_a, T_b] = i Σ_c f_{abc} T_c``. ``structure_constants`` maps 1-based
/// triples ``(a, b, c)`` to ``f_{abc}`` (``Expr`` or ``int``); an entry with
/// ``a > b`` means ``f_{abc} = -f_{bac}`` and both orders may be given if
/// they agree; absent entries are zero. The returned :class:`LieAlgebra`
/// reduces commutators of arbitrary noncommutative polynomials in the
/// generators (``commutator``), normal-orders words (``normal_order``) and
/// builds the Casimir.
///
/// The table is checked against the Jacobi identity (``check_jacobi=False``
/// skips it). Raises ``NcError`` ``E-NC-008`` for generators that are not
/// distinct noncommutative symbols, bad or inconsistent entries, or a failed
/// (or symbolically unverifiable) Jacobi identity.
///
/// Example::
///
///     J = [p.symbol(f"J{i}", commutative=False) for i in (1, 2, 3)]
///     su2 = lie_algebra(J, {(1, 2, 3): 1, (2, 3, 1): 1, (3, 1, 2): 1})
///     su2.commutator(J[0], J[1])      # I*J3
#[pyfunction]
#[pyo3(name = "lie_algebra", signature = (generators, structure_constants, *, check_jacobi = true))]
fn py_lie_algebra(
    py: Python<'_>,
    generators: Vec<PyRef<'_, PyExpr>>,
    structure_constants: &Bound<'_, PyDict>,
    check_jacobi: bool,
) -> PyResult<PyLieAlgebra> {
    let (pool_py, ids) = extract_generators(py, &generators)?;
    let mut entries = Vec::new();
    {
        let pool = pool_py.borrow(py);
        for (k, v) in structure_constants.iter() {
            let (a, b, c): (usize, usize, usize) = k.extract().map_err(|_| {
                PyTypeError::new_err("structure_constants keys must be (a, b, c) int triples")
            })?;
            if a == 0 || b == 0 || c == 0 {
                return Err(PyValueError::new_err(
                    "structure_constants indices are 1-based (f_{123} is the key (1, 2, 3))",
                ));
            }
            entries.push((
                a - 1,
                b - 1,
                c - 1,
                coerce_value(&v, &pool_py, &pool.inner)?,
            ));
        }
    }
    entries.sort_unstable_by_key(|&(a, b, c, _)| (a, b, c));
    build_lie(py, pool_py, ids, entries, check_jacobi)
}

fn make_generators(
    py: Python<'_>,
    pool: &Bound<'_, PyExprPool>,
    generators: Option<Vec<PyRef<'_, PyExpr>>>,
    prefix: &str,
    n: usize,
) -> PyResult<Vec<ExprId>> {
    use alkahest_core::kernel::Domain;
    let pool_ref = pool.borrow();
    match generators {
        Some(g) => {
            if g.len() != n {
                return Err(PyValueError::new_err(format!(
                    "expected {n} generators, got {}",
                    g.len()
                )));
            }
            for e in &g {
                same_pool_or_err(&e.pool, &pool.clone().unbind(), "a generator")?;
            }
            Ok(g.iter().map(|e| e.id).collect())
        }
        None => {
            let _ = py;
            Ok((1..=n)
                .map(|i| {
                    pool_ref.inner.symbol_commutative(
                        format!("{prefix}{i}"),
                        Domain::Complex,
                        false,
                    )
                })
                .collect())
        }
    }
}

fn table_dict(
    py: Python<'_>,
    pool_py: &Py<PyExprPool>,
    entries: Vec<((usize, usize, usize), ExprId)>,
) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    for ((a, b, c), f) in entries {
        d.set_item((a, b, c), wrap(py, f, pool_py).into_py(py))?;
    }
    Ok(d.into_py(py))
}

/// ``lie_su2(pool, generators=None, *, prefix="J") -> LieAlgebra``
///
/// su(2): ``[J_a, J_b] = i ε_{abc} J_c`` (``J = σ/2``, spin matrices). With
/// no ``generators`` it creates noncommutative symbols ``J1, J2, J3``. The
/// Casimir ``J1² + J2² + J3²`` commutes with every generator.
#[pyfunction]
#[pyo3(name = "lie_su2", signature = (pool, generators = None, *, prefix = "J"))]
fn py_lie_su2(
    py: Python<'_>,
    pool: &Bound<'_, PyExprPool>,
    generators: Option<Vec<PyRef<'_, PyExpr>>>,
    prefix: &str,
) -> PyResult<PyLieAlgebra> {
    let ids = make_generators(py, pool, generators, prefix, 3)?;
    let entries = core_lie::su2_structure_constants(&pool.borrow().inner);
    build_lie(py, pool.clone().unbind(), ids, entries, false)
}

/// ``lie_su3(pool, generators=None, *, prefix="T") -> LieAlgebra``
///
/// su(3) in the Gell-Mann basis: ``[T_a, T_b] = i f_{abc} T_c`` with
/// ``T = λ/2`` (``f_{123} = 1``, ``f_{458} = f_{678} = √3/2``, …; see
/// :func:`su3_f`). With no ``generators`` it creates ``T1 … T8``. The
/// Casimir ``Σ T_a²`` equals ``4/3`` in the fundamental representation.
#[pyfunction]
#[pyo3(name = "lie_su3", signature = (pool, generators = None, *, prefix = "T"))]
fn py_lie_su3(
    py: Python<'_>,
    pool: &Bound<'_, PyExprPool>,
    generators: Option<Vec<PyRef<'_, PyExpr>>>,
    prefix: &str,
) -> PyResult<PyLieAlgebra> {
    let ids = make_generators(py, pool, generators, prefix, 8)?;
    let entries = core_lie::su3_structure_constants(&pool.borrow().inner);
    build_lie(py, pool.clone().unbind(), ids, entries, false)
}

/// ``su3_f(pool) -> dict[tuple[int, int, int], Expr]``
///
/// The Gell-Mann structure constants ``f_{abc}`` (1-based, all nonzero
/// entries in all six index orders): ``[λ_a/2, λ_b/2] = i f_{abc} λ_c/2``.
#[pyfunction]
#[pyo3(name = "su3_f")]
fn py_su3_f(py: Python<'_>, pool: &Bound<'_, PyExprPool>) -> PyResult<PyObject> {
    let pool_py = pool.clone().unbind();
    let mut map: std::collections::BTreeMap<(usize, usize, usize), ExprId> = Default::default();
    {
        let p = pool.borrow();
        let neg = |e: ExprId| p.inner.mul(vec![p.inner.integer(-1_i32), e]);
        for (a, b, c, f) in core_lie::su3_structure_constants(&p.inner) {
            map.insert((a + 1, b + 1, c + 1), f);
            map.insert((b + 1, a + 1, c + 1), neg(f));
        }
    }
    table_dict(py, &pool_py, map.into_iter().collect())
}

/// ``su3_d(pool) -> dict[tuple[int, int, int], Expr]``
///
/// The Gell-Mann symmetric constants ``d_{abc}`` (1-based, all nonzero
/// entries in every index order): ``{λ_a, λ_b} = (4/3) δ_{ab} + 2 d_{abc} λ_c``.
#[pyfunction]
#[pyo3(name = "su3_d")]
fn py_su3_d(py: Python<'_>, pool: &Bound<'_, PyExprPool>) -> PyResult<PyObject> {
    let pool_py = pool.clone().unbind();
    let mut map: std::collections::BTreeMap<(usize, usize, usize), ExprId> = Default::default();
    {
        let p = pool.borrow();
        for ((a, b, c), d) in core_lie::su3_d_symbols(&p.inner) {
            for t in [
                (a, b, c),
                (a, c, b),
                (b, a, c),
                (b, c, a),
                (c, a, b),
                (c, b, a),
            ] {
                map.insert(t, d);
            }
        }
    }
    table_dict(py, &pool_py, map.into_iter().collect())
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
    m.add_function(wrap_pyfunction!(py_gamma_contract, m)?)?;
    m.add_function(wrap_pyfunction!(py_lorentz_contract, m)?)?;
    m.add_class::<PyLieAlgebra>()?;
    m.add_function(wrap_pyfunction!(py_lie_algebra, m)?)?;
    m.add_function(wrap_pyfunction!(py_lie_su2, m)?)?;
    m.add_function(wrap_pyfunction!(py_lie_su3, m)?)?;
    m.add_function(wrap_pyfunction!(py_su3_f, m)?)?;
    m.add_function(wrap_pyfunction!(py_su3_d, m)?)?;
    Ok(())
}
