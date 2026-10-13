//! PyO3 bindings for the propositional layer (`alkahest_core::logic::boolean`
//! and `alkahest_core::logic::bdd`): n-ary connectives, normal forms, exact
//! two-level minimisation, ROBDDs and equivalence checking.
//!
//! Refusals arrive as `alkahest.LogicError` (`E-LOGIC-002` … `E-LOGIC-006`);
//! a trip of the active `alkahest.Budget` arrives as `BudgetExceededError`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use alkahest_core::logic::bdd::{BddManager, BddRef, ReorderOptions, DEFAULT_MAX_BDD_NODES};
use alkahest_core::logic::boolean::{
    self as lb, BooleanError, EquivMethod, Equivalence, LogicForm, MinimizedLogic,
    MultiMinimizedLogic,
};
use alkahest_core::{ExprData, ExprId, ExprPool};

use crate::{
    cross_pool_err, guard_depth, make_structured_err, PyBudgetExceededError, PyExpr, PyExprPool,
    PyLogicError,
};

fn bool_err(e: BooleanError) -> PyErr {
    Python::with_gil(|py| match &e {
        BooleanError::Budget(b) => {
            let t = py.get_type_bound::<PyBudgetExceededError>();
            make_structured_err(py, &t, b)
        }
        _ => {
            let t = py.get_type_bound::<PyLogicError>();
            make_structured_err(py, &t, &e)
        }
    })
}

/// Arguments of a variadic connective: `Expr`s (one pool) and Python bools.
fn collect_args(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
) -> PyResult<(Py<PyExprPool>, Vec<ExprId>)> {
    let mut pool_py: Option<Py<PyExprPool>> = None;
    for a in args.iter() {
        if let Ok(e) = a.extract::<PyRef<PyExpr>>() {
            match &pool_py {
                None => pool_py = Some(e.pool.clone_ref(py)),
                Some(p) => {
                    if !e.pool.is(p) {
                        return Err(cross_pool_err(format!(
                            "[E-POOL-001] {op}(): arguments belong to different ExprPools"
                        )));
                    }
                }
            }
        }
    }
    let Some(pool_py) = pool_py else {
        return Err(PyTypeError::new_err(format!(
            "{op}() needs at least one Expr argument (Python bools alone carry no ExprPool)"
        )));
    };
    let mut ids = Vec::with_capacity(args.len());
    {
        let pool = pool_py.borrow(py);
        for a in args.iter() {
            if let Ok(e) = a.extract::<PyRef<PyExpr>>() {
                guard_depth(&pool.inner, e.id)?;
                ids.push(e.id);
            } else if a.is_instance_of::<pyo3::types::PyBool>() {
                let b: bool = a.extract()?;
                ids.push(if b {
                    pool.inner.pred_true()
                } else {
                    pool.inner.pred_false()
                });
            } else {
                return Err(PyTypeError::new_err(format!(
                    "{op}() arguments must be Expr predicates or bools, not {}",
                    a.get_type().name()?
                )));
            }
        }
    }
    Ok((pool_py, ids))
}

fn wrap(pool: Py<PyExprPool>, id: ExprId) -> PyExpr {
    PyExpr { id, pool }
}

/// Logical conjunction of any number of predicates.
///
/// ``And(a, b, c)`` is flattened (nested ``And`` arguments are spliced in),
/// ``True`` arguments are dropped, a ``False`` argument or a complementary
/// pair ``x, Not(x)`` gives ``False``, duplicates are removed and the
/// arguments are put in a canonical order (atoms by name, then compound
/// terms), so ``And(b, a) is`` the same expression as ``And(a, b)``.
/// Python ``True``/``False`` are accepted alongside expressions.  A single
/// argument is returned unchanged.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); a, b, c = p.symbol("a"), p.symbol("b"), p.symbol("c")
/// >>> ak.And(c, ak.And(b, a)) == ak.And(a, b, c)
/// True
#[pyfunction(name = "And", signature = (*args))]
fn py_and(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<PyExpr> {
    let (pool_py, ids) = collect_args(py, "And", args)?;
    let id = lb::logic_and(&pool_py.borrow(py).inner, &ids);
    Ok(wrap(pool_py, id))
}

/// Logical disjunction of any number of predicates (dual of :func:`And`:
/// flattened, ``False`` dropped, ``True`` or ``x, Not(x)`` gives ``True``,
/// deduplicated and canonically ordered).
#[pyfunction(name = "Or", signature = (*args))]
fn py_or(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<PyExpr> {
    let (pool_py, ids) = collect_args(py, "Or", args)?;
    let id = lb::logic_or(&pool_py.borrow(py).inner, &ids);
    Ok(wrap(pool_py, id))
}

/// Logical negation; ``Not(Not(x))`` is ``x`` and ``Not(True)`` is ``False``.
#[pyfunction(name = "Not")]
fn py_not(py: Python<'_>, a: PyRef<PyExpr>) -> PyResult<PyExpr> {
    let pool_py = a.pool.clone_ref(py);
    let id = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, a.id)?;
        lb::logic_not(&pool.inner, a.id)
    };
    Ok(wrap(pool_py, id))
}

/// Exclusive or (odd parity) of any number of predicates.
///
/// The kernel has no ``Xor`` node, so the result is the parity chain written
/// with ``And``/``Or``/``Not``; consecutive stages share their sub-formula, so
/// the expression grows linearly with the number of arguments.  ``Xor(x, x)``
/// is ``False``; ``Xor()`` of bools only is refused (no pool).
#[pyfunction(name = "Xor", signature = (*args))]
fn py_xor(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<PyExpr> {
    let (pool_py, ids) = collect_args(py, "Xor", args)?;
    let id = lb::logic_xor(&pool_py.borrow(py).inner, &ids);
    Ok(wrap(pool_py, id))
}

/// Material implication ``a → b``, built as ``Or(Not(a), b)``.
#[pyfunction(name = "Implies")]
fn py_implies(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let t = PyTuple::new_bound(py, [a, b]);
    let (pool_py, ids) = collect_args(py, "Implies", &t)?;
    let id = lb::logic_implies(&pool_py.borrow(py).inner, ids[0], ids[1]);
    Ok(wrap(pool_py, id))
}

/// Equivalence of any number of predicates: true when all have the same
/// value, built as ``Or(And(a₁…aₙ), And(¬a₁…¬aₙ))``.
#[pyfunction(name = "Equivalent", signature = (*args))]
fn py_equivalent_ctor(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<PyExpr> {
    let (pool_py, ids) = collect_args(py, "Equivalent", args)?;
    let id = lb::logic_equivalent(&pool_py.borrow(py).inner, &ids);
    Ok(wrap(pool_py, id))
}

/// If-then-else ``(c ∧ t) ∨ (¬c ∧ e)`` (the multiplexer).
#[pyfunction(name = "ITE")]
fn py_ite(
    py: Python<'_>,
    c: &Bound<'_, PyAny>,
    t: &Bound<'_, PyAny>,
    e: &Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    let tup = PyTuple::new_bound(py, [c, t, e]);
    let (pool_py, ids) = collect_args(py, "ITE", &tup)?;
    let id = lb::logic_ite(&pool_py.borrow(py).inner, ids[0], ids[1], ids[2]);
    Ok(wrap(pool_py, id))
}

fn unary_transform(
    py: Python<'_>,
    expr: &PyExpr,
    f: impl FnOnce(&ExprPool, ExprId) -> Result<ExprId, BooleanError>,
) -> PyResult<PyExpr> {
    let pool_py = expr.pool.clone_ref(py);
    let id = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        f(&pool.inner, expr.id).map_err(bool_err)?
    };
    Ok(wrap(pool_py, id))
}

/// Negation normal form: negations pushed onto atoms, only ``And``/``Or``
/// above them.  Arithmetic relations are opaque atoms (``Not(x < 0)`` is kept,
/// not rewritten to ``x >= 0``).
///
/// Raises ``LogicError`` (``E-LOGIC-002``) for quantified formulas or
/// non-Boolean terms.
#[pyfunction(name = "to_nnf")]
fn py_to_nnf(py: Python<'_>, expr: PyRef<PyExpr>) -> PyResult<PyExpr> {
    unary_transform(py, &expr, lb::to_nnf)
}

/// Conjunctive normal form (product of sums).
///
/// With ``simplify=False`` (default) the form is obtained by distribution, with
/// tautological, duplicate and subsumed clauses dropped; it can be
/// exponentially larger than the input and is refused past 100 000 clauses
/// (``LogicError``, ``E-LOGIC-003``).  ``simplify=True`` returns the minimum
/// CNF of :func:`simplify_logic` instead.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); a, b, c = p.symbol("a"), p.symbol("b"), p.symbol("c")
/// >>> print(ak.to_cnf(ak.Or(a, ak.And(b, c))))
/// (a ∨ b) ∧ (a ∨ c)
#[pyfunction(name = "to_cnf", signature = (expr, simplify=false))]
fn py_to_cnf(py: Python<'_>, expr: PyRef<PyExpr>, simplify: bool) -> PyResult<PyExpr> {
    if simplify {
        return unary_transform(py, &expr, |p, e| {
            lb::simplify_logic(p, e, LogicForm::Cnf, None, None).map(|m| m.expr)
        });
    }
    unary_transform(py, &expr, lb::to_cnf)
}

/// Disjunctive normal form (sum of products); the dual of :func:`to_cnf`,
/// with the same ``simplify`` switch and size ceiling.
#[pyfunction(name = "to_dnf", signature = (expr, simplify=false))]
fn py_to_dnf(py: Python<'_>, expr: PyRef<PyExpr>, simplify: bool) -> PyResult<PyExpr> {
    if simplify {
        return unary_transform(py, &expr, |p, e| {
            lb::simplify_logic(p, e, LogicForm::Dnf, None, None).map(|m| m.expr)
        });
    }
    unary_transform(py, &expr, lb::to_dnf)
}

/// Result of :func:`minimize_logic`: the two-level formula and how it was
/// obtained.
#[pyclass(name = "MinimizedLogic", module = "alkahest", frozen)]
struct PyMinimizedLogic {
    #[pyo3(get)]
    expr: PyExpr,
    /// ``"dnf"`` or ``"cnf"``.
    #[pyo3(get)]
    form: &'static str,
    /// ``True`` when the cover is proven minimum (fewest terms, then fewest
    /// literals); ``False`` for the heuristic cover.
    #[pyo3(get)]
    exact: bool,
    /// Product terms (DNF) or clauses (CNF).
    #[pyo3(get)]
    terms: usize,
    /// Literal occurrences.
    #[pyo3(get)]
    literals: usize,
}

#[pymethods]
impl PyMinimizedLogic {
    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.expr.pool.borrow(py);
        format!(
            "MinimizedLogic({}, form='{}', exact={}, terms={}, literals={})",
            pool.inner.display(self.expr.id),
            self.form,
            if self.exact { "True" } else { "False" },
            self.terms,
            self.literals
        )
    }
}

fn run_minimize(
    py: Python<'_>,
    expr: &PyExpr,
    form: &str,
    dontcare: Option<PyRef<PyExpr>>,
    exact: Option<bool>,
) -> PyResult<(Py<PyExprPool>, MinimizedLogic)> {
    let form = LogicForm::parse(form).map_err(bool_err)?;
    let pool_py = expr.pool.clone_ref(py);
    if let Some(d) = &dontcare {
        if !d.pool.is(&pool_py) {
            return Err(cross_pool_err(
                "[E-POOL-001] simplify_logic(): dontcare belongs to a different ExprPool".into(),
            ));
        }
    }
    let m = {
        let pool = pool_py.borrow(py);
        guard_depth(&pool.inner, expr.id)?;
        if let Some(d) = &dontcare {
            guard_depth(&pool.inner, d.id)?;
        }
        lb::simplify_logic(&pool.inner, expr.id, form, dontcare.map(|d| d.id), exact)
            .map_err(bool_err)?
    };
    Ok((pool_py, m))
}

/// Minimise a propositional formula to a two-level form.
///
/// Args:
///     expr: formula over Boolean symbols (relations such as ``x > 0`` are
///         accepted as opaque atoms).
///     form: ``"dnf"`` (sum of products), ``"cnf"`` (product of sums) or
///         ``"auto"`` (whichever has fewer literals, DNF on a tie).
///     dontcare: optional formula marking assignments whose value is free;
///         the result agrees with ``expr`` wherever ``dontcare`` is false.
///     exact: ``None`` (default) — exact up to 12 variables, heuristic beyond;
///         ``True`` — exact or raise ``LogicError`` (``E-LOGIC-004``, more
///         than 16 variables or a covering search past its bound);
///         ``False`` — heuristic.
///
/// Returns:
///     ``Expr`` — the minimised formula.  :func:`minimize_logic` returns the
///     same with ``exact``/``terms``/``literals`` metadata.
///
/// The exact method is Quine–McCluskey prime implicants followed by a
/// minimum cover (Petrick's method, run as branch-and-bound): fewest terms,
/// then fewest literals.  The heuristic is an espresso-style expand /
/// irredundant pass on the ROBDD of the function; its cover is always
/// equivalent but not necessarily minimum.  Every result is checked against
/// the input before it is returned.
///
/// Raises ``LogicError`` (``E-LOGIC-002`` not propositional, ``E-LOGIC-003``
/// size ceiling, ``E-LOGIC-004`` exactness unavailable, ``E-LOGIC-005`` bad
/// ``form``) and ``BudgetExceededError`` under an active budget.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); a, b = p.symbol("a"), p.symbol("b")
/// >>> print(ak.simplify_logic(ak.Or(ak.And(a, b), ak.And(a, ak.Not(b)))))
/// a
#[pyfunction(name = "simplify_logic", signature = (expr, form="auto", dontcare=None, exact=None))]
fn py_simplify_logic(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    form: &str,
    dontcare: Option<PyRef<PyExpr>>,
    exact: Option<bool>,
) -> PyResult<PyExpr> {
    let (pool_py, m) = run_minimize(py, &expr, form, dontcare, exact)?;
    Ok(wrap(pool_py, m.expr))
}

/// Result of :func:`minimize_logic` on a list of formulas: one two-level
/// formula per output, built from product terms (or clauses) shared between
/// outputs.
#[pyclass(name = "MultiMinimizedLogic", module = "alkahest", frozen)]
struct PyMultiMinimizedLogic {
    /// One formula per input, in order.
    #[pyo3(get)]
    exprs: Vec<PyExpr>,
    /// ``"dnf"`` or ``"cnf"``.
    #[pyo3(get)]
    form: &'static str,
    /// ``True`` when the shared cover is proven minimum (fewest distinct
    /// terms, then fewest literals); ``False`` for the heuristic cover.
    #[pyo3(get)]
    exact: bool,
    /// Distinct product terms (DNF) or clauses (CNF) over all outputs; a term
    /// feeding several outputs counts once.
    #[pyo3(get)]
    terms: usize,
    /// Literal occurrences over the distinct terms.
    #[pyo3(get)]
    literals: usize,
    /// Terms in each output's own formula.
    #[pyo3(get)]
    per_output_terms: Vec<usize>,
    /// Distinct terms used by more than one output.
    #[pyo3(get)]
    shared_terms: usize,
}

#[pymethods]
impl PyMultiMinimizedLogic {
    fn __len__(&self) -> usize {
        self.exprs.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiMinimizedLogic(outputs={}, form='{}', exact={}, terms={}, literals={}, shared_terms={})",
            self.exprs.len(),
            self.form,
            if self.exact { "True" } else { "False" },
            self.terms,
            self.literals,
            self.shared_terms
        )
    }
}

fn form_name(f: LogicForm) -> &'static str {
    if f == LogicForm::Cnf {
        "cnf"
    } else {
        "dnf"
    }
}

fn run_minimize_multi(
    py: Python<'_>,
    exprs: &[PyRef<PyExpr>],
    form: &str,
    dontcare: Option<PyRef<PyExpr>>,
    exact: Option<bool>,
) -> PyResult<PyMultiMinimizedLogic> {
    let form = LogicForm::parse(form).map_err(bool_err)?;
    if exprs.is_empty() {
        return Err(bool_err(BooleanError::InvalidArgument(
            "minimize_logic needs at least one formula".into(),
        )));
    }
    let pool_py = exprs[0].pool.clone_ref(py);
    for e in exprs.iter().skip(1) {
        if !e.pool.is(&pool_py) {
            return Err(cross_pool_err(
                "[E-POOL-001] minimize_logic(): formulas belong to different ExprPools".into(),
            ));
        }
    }
    if let Some(d) = &dontcare {
        if !d.pool.is(&pool_py) {
            return Err(cross_pool_err(
                "[E-POOL-001] minimize_logic(): dontcare belongs to a different ExprPool".into(),
            ));
        }
    }
    let m: MultiMinimizedLogic = {
        let pool = pool_py.borrow(py);
        for e in exprs {
            guard_depth(&pool.inner, e.id)?;
        }
        if let Some(d) = &dontcare {
            guard_depth(&pool.inner, d.id)?;
        }
        let ids: Vec<ExprId> = exprs.iter().map(|e| e.id).collect();
        lb::simplify_logic_multi(&pool.inner, &ids, form, dontcare.map(|d| d.id), exact)
            .map_err(bool_err)?
    };
    Ok(PyMultiMinimizedLogic {
        exprs: m
            .exprs
            .iter()
            .map(|&id| PyExpr {
                pool: pool_py.clone_ref(py),
                id,
            })
            .collect(),
        form: form_name(m.form),
        exact: m.exact,
        terms: m.terms,
        literals: m.literals,
        per_output_terms: m.per_output_terms,
        shared_terms: m.shared_terms,
    })
}

/// :func:`simplify_logic` returning a :class:`MinimizedLogic` record
/// (``expr``, ``form``, ``exact``, ``terms``, ``literals``) so a caller can see
/// whether minimality was proven.  Default ``form`` is ``"dnf"``.
///
/// **Multi-output.** Pass a list or tuple of formulas to minimise them
/// jointly: the result is a :class:`MultiMinimizedLogic` whose ``exprs`` are
/// two-level forms built from product terms (clauses for ``form="cnf"``)
/// *shared* between outputs, so ``terms`` counts each distinct term once —
/// never more than the sum of the separately minimised outputs.  ``dontcare``
/// applies to every output.  Up to 12 variables (16 with ``exact=True``) the
/// shared cover is exact (multi-output Quine–McCluskey primes, then a minimum
/// cover of the (output, minterm) pairs: fewest distinct terms, then fewest
/// literals); beyond that the espresso-style pass runs per output and the
/// pooled cubes are cut down to an irredundant shared set, flagged
/// ``exact=False``.  Every output is verified against its input before the
/// result is returned.  At most 64 outputs (``LogicError``, ``E-LOGIC-003``).
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); a, b = p.symbol("a"), p.symbol("b")
/// >>> r = ak.minimize_logic([ak.Xor(a, b), ak.And(a, b), ak.Or(a, b)])
/// >>> r.terms, r.exact
/// (3, True)
#[pyfunction(name = "minimize_logic", signature = (expr, form="dnf", dontcare=None, exact=None))]
fn py_minimize_logic(
    py: Python<'_>,
    expr: &Bound<'_, PyAny>,
    form: &str,
    dontcare: Option<PyRef<PyExpr>>,
    exact: Option<bool>,
) -> PyResult<PyObject> {
    if expr.downcast::<PyList>().is_ok() || expr.downcast::<PyTuple>().is_ok() {
        let mut es: Vec<PyRef<PyExpr>> = Vec::new();
        for item in expr.iter()? {
            es.push(
                item?
                    .extract::<PyRef<PyExpr>>()
                    .map_err(|_| PyTypeError::new_err("minimize_logic: every output is an Expr"))?,
            );
        }
        let r = run_minimize_multi(py, &es, form, dontcare, exact)?;
        return Ok(Py::new(py, r)?.into_py(py));
    }
    let expr = expr
        .extract::<PyRef<PyExpr>>()
        .map_err(|_| PyTypeError::new_err("minimize_logic expects an Expr or a list of Exprs"))?;
    let (pool_py, m) = run_minimize(py, &expr, form, dontcare, exact)?;
    let r = PyMinimizedLogic {
        expr: wrap(pool_py, m.expr),
        form: form_name(m.form),
        exact: m.exact,
        terms: m.terms,
        literals: m.literals,
    };
    Ok(Py::new(py, r)?.into_py(py))
}

/// Decide whether two propositional formulas are equivalent.
///
/// Args:
///     f, g: formulas (same pool).
///     method: ``"auto"`` (truth table up to 16 variables, then a BDD capped
///         at 200 000 nodes, then SAT), ``"truth_table"``, ``"bdd"`` or
///         ``"sat"`` (DPLL on the miter ``f ⊕ g``).
///     counterexample: return ``(verdict, assignment_or_None)`` instead of a
///         bare bool; the assignment maps symbol names to bools and is checked
///         to separate ``f`` and ``g``.
///
/// Relations (``x > 0``) are accepted as opaque atoms: ``True`` is then still
/// a proof, but a propositional difference is refused with ``LogicError``
/// (``E-LOGIC-002``) rather than reported as ``False``, because ``x > 0`` and
/// ``Not(x <= 0)`` are propositionally independent yet equal over ℝ.  A search
/// that exhausts its bound raises ``LogicError`` (``E-LOGIC-003``) — never a
/// guessed verdict.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); a, b = p.symbol("a"), p.symbol("b")
/// >>> ak.equivalent(ak.Not(ak.And(a, b)), ak.Or(ak.Not(a), ak.Not(b)))
/// True
#[pyfunction(name = "equivalent", signature = (f, g, method="auto", counterexample=false))]
fn py_equivalent(
    py: Python<'_>,
    f: PyRef<PyExpr>,
    g: PyRef<PyExpr>,
    method: &str,
    counterexample: bool,
) -> PyResult<PyObject> {
    if !f.pool.is(&g.pool) {
        return Err(cross_pool_err(
            "[E-POOL-001] equivalent(): f and g belong to different ExprPools".into(),
        ));
    }
    let method = EquivMethod::parse(method).map_err(bool_err)?;
    let verdict = {
        let pool = f.pool.borrow(py);
        guard_depth(&pool.inner, f.id)?;
        guard_depth(&pool.inner, g.id)?;
        lb::equivalent(&pool.inner, f.id, g.id, method).map_err(bool_err)?
    };
    let (ok, cex) = match verdict {
        Equivalence::Equivalent => (true, py.None()),
        Equivalence::Counterexample(m) => {
            let d = PyDict::new_bound(py);
            for (k, v) in m {
                d.set_item(k, v)?;
            }
            (false, d.into_py(py))
        }
        _ => unreachable!("Equivalence has two variants"),
    };
    if counterexample {
        Ok((ok, cex).into_py(py))
    } else {
        Ok(ok.into_py(py))
    }
}

/// Truth table of a formula over Boolean symbols.
///
/// Args:
///     expr: formula.
///     variables: the symbols to tabulate, in column order; must include
///         every symbol of ``expr``.  ``None`` uses the formula's symbols
///         sorted by name.
///
/// Returns:
///     ``list[tuple[tuple[bool, ...], bool]]`` — one ``(assignment, value)``
///     row per assignment, from all-``False`` to all-``True`` with the last
///     variable varying fastest.  At most 22 variables (``E-LOGIC-003``).
#[pyfunction(name = "truth_table", signature = (expr, variables=None))]
fn py_truth_table(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    variables: Option<Vec<PyRef<PyExpr>>>,
) -> PyResult<Vec<(Py<PyTuple>, bool)>> {
    let pool = expr.pool.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
    let vars: Option<Vec<ExprId>> = match &variables {
        Some(vs) => {
            for v in vs {
                if !v.pool.is(&expr.pool) {
                    return Err(cross_pool_err(
                        "[E-POOL-001] truth_table(): a variable belongs to a different ExprPool"
                            .into(),
                    ));
                }
            }
            Some(vs.iter().map(|v| v.id).collect())
        }
        None => None,
    };
    let (names, rows) = lb::truth_table(&pool.inner, expr.id, vars.as_deref()).map_err(bool_err)?;
    let n = names.len();
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(r, v)| {
            let bits: Vec<bool> = (0..n).map(|i| (r >> (n - 1 - i)) & 1 == 1).collect();
            (PyTuple::new_bound(py, bits).unbind(), v)
        })
        .collect())
}

// ---------------------------------------------------------------------------
// BDD
// ---------------------------------------------------------------------------

struct Shared {
    mgr: BddManager,
    atoms: HashMap<String, ExprId>,
}

/// A reduced ordered binary decision diagram, returned by :func:`bdd`.
///
/// Canonical for its variable order: two BDDs of the same manager are
/// equivalent exactly when :meth:`equivalent` says so, in time independent of
/// the formulas.  ``&``, ``|``, ``^`` and ``~`` combine BDDs (or a BDD and an
/// ``Expr``); operands built by separate :func:`bdd` calls are merged by
/// variable name.
#[pyclass(name = "BDD", module = "alkahest")]
struct PyBdd {
    shared: Arc<Mutex<Shared>>,
    root: BddRef,
    pool: Py<PyExprPool>,
}

fn lock(s: &Arc<Mutex<Shared>>) -> std::sync::MutexGuard<'_, Shared> {
    s.lock().unwrap_or_else(|p| p.into_inner())
}

/// Resolve a variable given as a symbol `Expr` or a name to its level;
/// `None` for a variable the manager has never seen (no function in it
/// depends on that variable, so restricting or quantifying it is the
/// identity).
fn level_of(py: Python<'_>, sh: &Shared, ob: &Bound<'_, PyAny>) -> PyResult<Option<u32>> {
    let name: String = if let Ok(e) = ob.extract::<PyRef<PyExpr>>() {
        let pool = e.pool.borrow(py);
        match pool.inner.get(e.id) {
            ExprData::Symbol { name, .. } => name,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "'{}' is not a symbol",
                    pool.inner.display(e.id)
                )))
            }
        }
    } else if let Ok(s) = ob.extract::<String>() {
        s
    } else {
        return Err(PyTypeError::new_err(
            "a BDD variable is a symbol Expr or its name",
        ));
    };
    Ok(sh.mgr.level_of(&name))
}

impl Drop for PyBdd {
    fn drop(&mut self) {
        lock(&self.shared).mgr.unpin(self.root);
    }
}

/// Node count past which a ``bdd()`` call starts sifting by default.
const DEFAULT_AUTO_REORDER: usize = 50_000;

/// Parse an ``auto_reorder`` argument: ``False`` = off, ``True`` = the
/// default threshold, an ``int`` = that many live nodes.
fn parse_auto_reorder(ob: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    if ob.is_none() {
        return Ok(None);
    }
    if let Ok(b) = ob.downcast::<pyo3::types::PyBool>() {
        return Ok(b.is_true().then_some(DEFAULT_AUTO_REORDER));
    }
    let n: usize = ob.extract().map_err(|_| {
        PyTypeError::new_err("auto_reorder is False, True or a node-count threshold")
    })?;
    if n == 0 {
        return Err(PyValueError::new_err(
            "auto_reorder threshold must be at least 1 node",
        ));
    }
    Ok(Some(n))
}

impl PyBdd {
    /// A new handle on `root` of this manager.  The root is pinned for the
    /// handle's lifetime (reordering keeps pinned functions alive), and
    /// automatic reordering gets its chance to run.
    fn with_root(&self, py: Python<'_>, root: BddRef) -> PyBdd {
        let out = PyBdd {
            shared: self.shared.clone(),
            root,
            pool: self.pool.clone_ref(py),
        };
        {
            let mut sh = lock(&self.shared);
            sh.mgr.pin(root);
            // Reordering is an optimisation: a refusal (budget) is left for
            // the next operation to report.
            let _ = sh.mgr.maybe_reorder(&[]);
        }
        out
    }

    /// Bring `other` (a BDD or an `Expr`) into this manager.
    fn operand(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<BddRef> {
        if let Ok(o) = other.extract::<PyRef<PyBdd>>() {
            if !o.pool.is(&self.pool) {
                return Err(cross_pool_err(
                    "[E-POOL-001] BDD operands belong to different ExprPools".into(),
                ));
            }
            if Arc::ptr_eq(&o.shared, &self.shared) {
                return Ok(o.root);
            }
            let src = lock(&o.shared);
            let mut dst = lock(&self.shared);
            let r = dst.mgr.transfer_from(&src.mgr, o.root).map_err(bool_err)?;
            for (k, v) in &src.atoms {
                dst.atoms.entry(k.clone()).or_insert(*v);
            }
            return Ok(r);
        }
        if let Ok(e) = other.extract::<PyRef<PyExpr>>() {
            if !e.pool.is(&self.pool) {
                return Err(cross_pool_err(
                    "[E-POOL-001] BDD and Expr belong to different ExprPools".into(),
                ));
            }
            let pool = self.pool.borrow(py);
            guard_depth(&pool.inner, e.id)?;
            let mut sh = lock(&self.shared);
            let (r, atoms) = lb::bdd_from_expr(&pool.inner, e.id, &mut sh.mgr).map_err(bool_err)?;
            for (k, v) in atoms {
                sh.atoms.entry(k).or_insert(v);
            }
            return Ok(r);
        }
        if let Ok(b) = other.extract::<bool>() {
            return Ok(if b { BddRef::TRUE } else { BddRef::FALSE });
        }
        Err(PyTypeError::new_err(
            "BDD operand must be a BDD, an Expr or a bool",
        ))
    }

    fn binop(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: fn(&mut BddManager, BddRef, BddRef) -> Result<BddRef, BooleanError>,
    ) -> PyResult<PyBdd> {
        let g = self.operand(py, other)?;
        let r = {
            let mut sh = lock(&self.shared);
            op(&mut sh.mgr, self.root, g).map_err(bool_err)?
        };
        Ok(self.with_root(py, r))
    }

    fn levels(&self, py: Python<'_>, vars: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
        let sh = lock(&self.shared);
        let mut out = Vec::new();
        for v in vars.iter()? {
            if let Some(l) = level_of(py, &sh, &v?)? {
                out.push(l);
            }
        }
        Ok(out)
    }
}

#[pymethods]
impl PyBdd {
    fn __and__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        self.binop(py, other, BddManager::and)
    }
    fn __rand__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        self.binop(py, other, BddManager::and)
    }
    fn __or__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        self.binop(py, other, BddManager::or)
    }
    fn __ror__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        self.binop(py, other, BddManager::or)
    }
    fn __xor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        self.binop(py, other, BddManager::xor)
    }
    fn __rxor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        self.binop(py, other, BddManager::xor)
    }
    fn __invert__(&self, py: Python<'_>) -> PyResult<PyBdd> {
        let r = {
            let mut sh = lock(&self.shared);
            sh.mgr.not(self.root).map_err(bool_err)?
        };
        Ok(self.with_root(py, r))
    }

    /// ``self → other``.
    fn implies(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        self.binop(py, other, BddManager::implies)
    }

    /// ``self ↔ other``.
    fn iff(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        self.binop(py, other, BddManager::iff)
    }

    /// ``if self then g else h`` (the ITE operator every BDD operation reduces to).
    fn ite(&self, py: Python<'_>, g: &Bound<'_, PyAny>, h: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        let gr = self.operand(py, g)?;
        lock(&self.shared).mgr.pin(gr);
        let hr = self.operand(py, h);
        let r = {
            let mut sh = lock(&self.shared);
            sh.mgr.unpin(gr);
            match hr {
                Ok(hr) => sh.mgr.ite(self.root, gr, hr).map_err(bool_err),
                Err(e) => Err(e),
            }
        }?;
        Ok(self.with_root(py, r))
    }

    /// Cofactor: fix variables to constants.  ``assignment`` maps symbols (or
    /// names) to bools.
    fn restrict(&self, py: Python<'_>, assignment: &Bound<'_, PyDict>) -> PyResult<PyBdd> {
        let mut pairs = Vec::new();
        {
            let sh = lock(&self.shared);
            for (k, v) in assignment.iter() {
                let val = v.extract::<bool>()?;
                if let Some(l) = level_of(py, &sh, &k)? {
                    pairs.push((l, val));
                }
            }
        }
        let mut sh = lock(&self.shared);
        let mut r = self.root;
        for (l, v) in pairs {
            r = sh.mgr.restrict(r, l, v).map_err(bool_err)?;
        }
        drop(sh);
        Ok(self.with_root(py, r))
    }

    /// Existential quantification ``∃ vars . self``.
    fn exists(&self, py: Python<'_>, variables: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        let ls = self.levels(py, variables)?;
        let r = {
            let mut sh = lock(&self.shared);
            sh.mgr.quantify(self.root, &ls, false).map_err(bool_err)?
        };
        Ok(self.with_root(py, r))
    }

    /// Universal quantification ``∀ vars . self``.
    fn forall(&self, py: Python<'_>, variables: &Bound<'_, PyAny>) -> PyResult<PyBdd> {
        let ls = self.levels(py, variables)?;
        let r = {
            let mut sh = lock(&self.shared);
            sh.mgr.quantify(self.root, &ls, true).map_err(bool_err)?
        };
        Ok(self.with_root(py, r))
    }

    /// Number of satisfying assignments (a Python ``int``, exact).
    ///
    /// With ``variables=None`` the count is over every variable of the BDD's
    /// manager — the formula's symbols plus any brought in by combining it
    /// with other BDDs.  Pass ``variables`` (symbols or names, a superset of
    /// :meth:`support`) to count over exactly that set.
    #[pyo3(signature = (variables=None))]
    fn sat_count(
        &self,
        py: Python<'_>,
        variables: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let sh = lock(&self.shared);
        let nm = sh.mgr.num_vars() as u32;
        let total = sh.mgr.sat_count(self.root, nm);
        let count = match variables {
            None => total,
            Some(vs) => {
                let mut names: Vec<String> = Vec::new();
                for v in vs.iter()? {
                    let v = v?;
                    let name = if let Ok(e) = v.extract::<PyRef<PyExpr>>() {
                        let pool = e.pool.borrow(py);
                        match pool.inner.get(e.id) {
                            ExprData::Symbol { name, .. } => name,
                            _ => return Err(PyValueError::new_err("variables must be symbols")),
                        }
                    } else {
                        v.extract::<String>()?
                    };
                    if !names.contains(&name) {
                        names.push(name);
                    }
                }
                for l in sh.mgr.support(self.root) {
                    let nmv = &sh.mgr.var_names()[l as usize];
                    if !names.contains(nmv) {
                        return Err(bool_err(BooleanError::InvalidArgument(format!(
                            "the function depends on '{nmv}', which is not in variables"
                        ))));
                    }
                }
                let k_in = names
                    .iter()
                    .filter(|n| sh.mgr.level_of(n).is_some())
                    .count() as u32;
                let k_out = names.len() as u32 - k_in;
                (total >> (nm - k_in)) << k_out
            }
        };
        let s = count.to_string();
        let int = py.get_type_bound::<pyo3::types::PyLong>();
        Ok(int.call1((s,))?.into_py(py))
    }

    /// Re-sort the manager's variable order by Rudell sifting to shrink the
    /// diagrams.
    ///
    /// Every function of the manager (this BDD and any combined with it)
    /// keeps its meaning: ``sat_count``, :meth:`equivalent` and
    /// :meth:`to_expr` are unchanged; :attr:`order` and :meth:`node_count`
    /// change.  Unreferenced intermediate nodes are collected.
    ///
    /// Args:
    ///     max_growth: abandon a variable's excursion in one direction once
    ///         the manager exceeds this multiple of the best size seen for it
    ///         (default 1.2; at least 1.0).
    ///     max_swaps: stop after this many adjacent-variable swaps (default
    ///         200 000); ``complete`` is then ``False``.
    ///
    /// Returns:
    ///     ``dict`` with ``nodes_before`` / ``nodes_after`` (live nodes of the
    ///     whole manager), ``swaps`` and ``complete``.  An active
    ///     ``alkahest.Budget`` is honoured between swaps
    ///     (``BudgetExceededError``; the order reached stays valid).
    ///
    /// >>> import alkahest as ak
    /// >>> p = ak.ExprPool()
    /// >>> xs = [p.symbol(f"x{i}") for i in range(6)]
    /// >>> ys = [p.symbol(f"y{i}") for i in range(6)]
    /// >>> f = ak.bdd(ak.Or(*[ak.And(x, y) for x, y in zip(xs, ys)]),
    /// ...            order=xs + ys, auto_reorder=False)
    /// >>> before = f.node_count()
    /// >>> _ = f.reorder()
    /// >>> f.node_count() < before
    /// True
    #[pyo3(signature = (max_growth=1.2, max_swaps=200_000))]
    fn reorder(&self, py: Python<'_>, max_growth: f64, max_swaps: usize) -> PyResult<PyObject> {
        let st = {
            let mut sh = lock(&self.shared);
            sh.mgr
                .reorder(
                    &[],
                    &ReorderOptions {
                        max_growth,
                        max_swaps,
                    },
                )
                .map_err(bool_err)?
        };
        let d = PyDict::new_bound(py);
        d.set_item("nodes_before", st.nodes_before)?;
        d.set_item("nodes_after", st.nodes_after)?;
        d.set_item("swaps", st.swaps)?;
        d.set_item("complete", st.complete)?;
        Ok(d.into_py(py))
    }

    /// Turn automatic reordering of this BDD's manager on or off: ``False``
    /// (off), ``True`` (sift once the manager holds 50 000 live nodes) or an
    /// ``int`` threshold.  Checked after every operation that produces a BDD.
    fn set_auto_reorder(&self, threshold: &Bound<'_, PyAny>) -> PyResult<()> {
        let t = parse_auto_reorder(threshold)?;
        lock(&self.shared).mgr.set_auto_reorder(t);
        Ok(())
    }

    /// The automatic-reordering threshold in nodes, or ``None`` when off.
    #[getter]
    fn auto_reorder(&self) -> Option<usize> {
        lock(&self.shared).mgr.auto_reorder_threshold()
    }

    /// Non-terminal nodes reachable from this function.
    fn node_count(&self) -> usize {
        lock(&self.shared).mgr.node_count(self.root)
    }

    /// Variable names in the manager's order (root first).
    #[getter]
    fn order(&self) -> Vec<String> {
        lock(&self.shared).mgr.var_names().to_vec()
    }

    /// Names of the variables the function depends on, in BDD order.
    fn support(&self) -> Vec<String> {
        let sh = lock(&self.shared);
        sh.mgr
            .support(self.root)
            .into_iter()
            .map(|l| sh.mgr.var_names()[l as usize].clone())
            .collect()
    }

    /// One satisfying assignment ``{name: bool}`` (variables left out are
    /// free), or ``None`` if unsatisfiable.
    fn satisfy_one(&self, py: Python<'_>) -> PyResult<PyObject> {
        let sh = lock(&self.shared);
        match sh.mgr.any_sat(self.root) {
            None => Ok(py.None()),
            Some(lits) => {
                let d = PyDict::new_bound(py);
                for (l, v) in lits {
                    d.set_item(&sh.mgr.var_names()[l as usize], v)?;
                }
                Ok(d.into_py(py))
            }
        }
    }

    /// ``True`` when the function is the constant true.
    fn is_true(&self) -> bool {
        self.root.is_true()
    }

    /// ``True`` when the function is the constant false.
    fn is_false(&self) -> bool {
        self.root.is_false()
    }

    /// Canonical equivalence: same node after merging the operand into this
    /// manager.
    fn equivalent(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let g = self.operand(py, other)?;
        Ok(g == self.root)
    }

    /// The function as a formula (Shannon expansion at every node).
    fn to_expr(&self, py: Python<'_>) -> PyResult<PyExpr> {
        let id = {
            let pool = self.pool.borrow(py);
            let sh = lock(&self.shared);
            let atoms = sh.atoms.clone();
            lb::bdd_to_expr(&pool.inner, &sh.mgr, self.root, &|n| atoms.get(n).copied())
                .map_err(bool_err)?
        };
        Ok(wrap(self.pool.clone_ref(py), id))
    }

    fn __repr__(&self) -> String {
        let sh = lock(&self.shared);
        format!(
            "BDD(nodes={}, support={:?})",
            sh.mgr.node_count(self.root),
            sh.mgr
                .support(self.root)
                .into_iter()
                .map(|l| sh.mgr.var_names()[l as usize].clone())
                .collect::<Vec<_>>()
        )
    }
}

/// Build the reduced ordered BDD of a formula over Boolean symbols.
///
/// Args:
///     expr: formula (``And``/``Or``/``Not``/``Xor``/… over symbols).
///     order: optional list of symbols (or names) fixing the variable order,
///         root first; symbols of ``expr`` not listed are appended.  ``None``
///         uses depth-first first-visit order over the canonically sorted
///         formula, which interleaves the operand bits of bit-sliced
///         arithmetic (the order adders and comparators need).
///     max_nodes: node ceiling (default 2 000 000).  A BDD that would exceed
///         it is refused with ``LogicError`` (``E-LOGIC-003``); an active
///         ``alkahest.Budget`` is honoured too (``BudgetExceededError``).
///     auto_reorder: dynamic variable reordering (Rudell sifting) once the
///         manager holds this many live nodes: ``False`` off, ``True`` or an
///         ``int`` threshold on.  ``None`` (default) means on at 50 000 nodes
///         unless an explicit ``order`` was given.  It never changes what a
///         BDD means, only its order and size; see :meth:`BDD.reorder` to
///         sift on demand.
///
/// Returns:
///     :class:`BDD`.
///
/// >>> import alkahest as ak
/// >>> p = ak.ExprPool(); a, b, c = p.symbol("a"), p.symbol("b"), p.symbol("c")
/// >>> f = ak.bdd(ak.Or(ak.And(a, b), ak.And(b, c), ak.And(a, c)))
/// >>> f.sat_count(), f.node_count()
/// (4, 4)
#[pyfunction(name = "bdd", signature = (expr, order=None, max_nodes=None, auto_reorder=None))]
fn py_bdd(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    order: Option<&Bound<'_, PyAny>>,
    max_nodes: Option<usize>,
    auto_reorder: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyBdd> {
    let mut mgr = BddManager::with_max_nodes(max_nodes.unwrap_or(DEFAULT_MAX_BDD_NODES));
    mgr.set_auto_reorder(match auto_reorder {
        Some(a) => parse_auto_reorder(a)?,
        None => order.is_none().then_some(DEFAULT_AUTO_REORDER),
    });
    let pool_py = expr.pool.clone_ref(py);
    let pool = pool_py.borrow(py);
    guard_depth(&pool.inner, expr.id)?;
    if let Some(order) = order {
        for v in order.iter()? {
            let v = v?;
            let name = if let Ok(e) = v.extract::<PyRef<PyExpr>>() {
                if !e.pool.is(&pool_py) {
                    return Err(cross_pool_err(
                        "[E-POOL-001] bdd(): an order variable belongs to a different ExprPool"
                            .into(),
                    ));
                }
                match pool.inner.get(e.id) {
                    ExprData::Symbol { name, .. } => name,
                    _ => {
                        return Err(bool_err(BooleanError::InvalidArgument(format!(
                            "order entry '{}' is not a symbol",
                            pool.inner.display(e.id)
                        ))))
                    }
                }
            } else if let Ok(s) = v.extract::<String>() {
                s
            } else {
                return Err(PyTypeError::new_err("order entries are symbols or names"));
            };
            mgr.declare(&name);
        }
    }
    let (root, atoms) = lb::bdd_from_expr(&pool.inner, expr.id, &mut mgr).map_err(bool_err)?;
    drop(pool);
    mgr.pin(root);
    Ok(PyBdd {
        shared: Arc::new(Mutex::new(Shared {
            mgr,
            atoms: atoms.into_iter().collect(),
        })),
        root,
        pool: pool_py,
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBdd>()?;
    m.add_class::<PyMinimizedLogic>()?;
    m.add_class::<PyMultiMinimizedLogic>()?;
    m.add_function(wrap_pyfunction!(py_and, m)?)?;
    m.add_function(wrap_pyfunction!(py_or, m)?)?;
    m.add_function(wrap_pyfunction!(py_not, m)?)?;
    m.add_function(wrap_pyfunction!(py_xor, m)?)?;
    m.add_function(wrap_pyfunction!(py_implies, m)?)?;
    m.add_function(wrap_pyfunction!(py_equivalent_ctor, m)?)?;
    m.add_function(wrap_pyfunction!(py_ite, m)?)?;
    m.add_function(wrap_pyfunction!(py_to_nnf, m)?)?;
    m.add_function(wrap_pyfunction!(py_to_cnf, m)?)?;
    m.add_function(wrap_pyfunction!(py_to_dnf, m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_logic, m)?)?;
    m.add_function(wrap_pyfunction!(py_minimize_logic, m)?)?;
    m.add_function(wrap_pyfunction!(py_equivalent, m)?)?;
    m.add_function(wrap_pyfunction!(py_truth_table, m)?)?;
    m.add_function(wrap_pyfunction!(py_bdd, m)?)?;
    Ok(())
}
