//! PyO3 bindings for the joint-law layer of `alkahest_core::prob` (G10f):
//! `MultivariateNormal`, `joint_expectation`, `covariance`, `joint_variance`,
//! `sum_independent`, `order_statistic` / `OrderStatistic`, and the
//! `NegativeBinomial` constructor. Everything lands in
//! `alkahest.experimental` next to the univariate `Distribution` surface, and
//! refuses with the same `ProbabilityError` / `E-PROB-NNN` codes.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use alkahest_core::prob::{
    covariance as core_covariance, entropy_asymptotic as core_entropy_asymptotic,
    joint_expectation as core_joint_expectation, joint_variance as core_joint_variance,
    kth_order_statistic as core_kth_order_statistic, order_statistic as core_order_statistic,
    sum_independent as core_sum_independent, Distribution as CoreDistribution, JointBlock,
    MultivariateNormal as CoreMvn, OrderKind, OrderStatistic as CoreOrderStatistic,
};
use alkahest_core::{ExprId, ExprPool};

use crate::{
    build_dist, capture_prob_side_conditions, coerce_substituent, guard_depth, guard_depth_all,
    pool_mismatch_err, prob_error_to_py, PyDistribution, PyExpr, PyExprPool, PyMatrix,
};

type CoreResult = Result<alkahest_core::deriv::DerivedExpr<ExprId>, alkahest_core::prob::ProbError>;

fn wrap(py: Python<'_>, pool: &Py<PyExprPool>, id: ExprId) -> PyExpr {
    PyExpr {
        id,
        pool: pool.clone_ref(py),
    }
}

/// Run a core call, drain the side-condition channel on both paths, and wrap
/// the value.
fn derived<F>(py: Python<'_>, pool_py: &Py<PyExprPool>, f: F) -> PyResult<PyExpr>
where
    F: FnOnce(&ExprPool) -> CoreResult,
{
    let pool = pool_py.borrow(py);
    let out = f(&pool.inner);
    capture_prob_side_conditions(&pool.inner);
    let out = out.map_err(prob_error_to_py)?;
    Ok(wrap(py, pool_py, out.value))
}

/// The pool of the first `Expr` found anywhere in `obs` (recursing into
/// lists/tuples and `Matrix` objects).
fn find_pool(py: Python<'_>, obs: &[&Bound<'_, PyAny>]) -> Option<Py<PyExprPool>> {
    fn go(py: Python<'_>, ob: &Bound<'_, PyAny>) -> Option<Py<PyExprPool>> {
        if let Ok(e) = ob.extract::<PyRef<PyExpr>>() {
            return Some(e.pool.clone_ref(py));
        }
        if let Ok(m) = ob.extract::<PyRef<PyMatrix>>() {
            return Some(m.pool.clone_ref(py));
        }
        if let Ok(l) = ob.downcast::<PyList>() {
            return l.iter().find_map(|x| go(py, &x));
        }
        if let Ok(t) = ob.downcast::<PyTuple>() {
            return t.iter().find_map(|x| go(py, &x));
        }
        None
    }
    obs.iter().find_map(|o| go(py, o))
}

fn vector(py: Python<'_>, pool: &Py<PyExprPool>, ob: &Bound<'_, PyAny>) -> PyResult<Vec<ExprId>> {
    let items: Vec<Bound<'_, PyAny>> = ob.extract().map_err(|_| {
        PyTypeError::new_err("expected a sequence of expressions (Expr, int or Fraction)")
    })?;
    let ids = items
        .iter()
        .map(|x| coerce_substituent(pool, x, py))
        .collect::<PyResult<Vec<ExprId>>>()?;
    guard_depth_all(&pool.borrow(py).inner, &ids)?;
    Ok(ids)
}

fn matrix(
    py: Python<'_>,
    pool: &Py<PyExprPool>,
    ob: &Bound<'_, PyAny>,
) -> PyResult<Vec<Vec<ExprId>>> {
    if let Ok(m) = ob.extract::<PyRef<PyMatrix>>() {
        if !m.pool.is(pool) {
            return Err(pool_mismatch_err());
        }
        return Ok(m.inner.to_nested());
    }
    let rows: Vec<Bound<'_, PyAny>> = ob
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a Matrix or a list of rows of expressions"))?;
    rows.iter().map(|r| vector(py, pool, r)).collect()
}

fn base_id(pool: &Py<PyExprPool>, base: Option<PyRef<PyExpr>>) -> PyResult<Option<ExprId>> {
    match base {
        None => Ok(None),
        Some(b) if b.pool.is(pool) => Ok(Some(b.id)),
        Some(_) => Err(pool_mismatch_err()),
    }
}

fn to_indices(ob: Vec<i64>, what: &str) -> PyResult<Vec<usize>> {
    ob.into_iter()
        .map(|i| {
            usize::try_from(i)
                .map_err(|_| PyValueError::new_err(format!("{what}: index {i} is negative")))
        })
        .collect()
}

/// ``MultivariateNormal(mean, cov)`` — ``N(mu, Sigma)`` on ``R^d`` with
/// symbolic entries allowed.
///
/// ``mean`` is a sequence of ``d`` expressions; ``cov`` is a ``d x d``
/// :class:`~alkahest.Matrix` or list of rows. ``Sigma`` must be symmetric
/// (entries ``(i, j)`` and ``(j, i)`` the same expression) and is checked for
/// positive definiteness by Sylvester's criterion: a leading principal minor
/// that evaluates to a number ``<= 0`` raises ``E-PROB-001``; a symbolic minor
/// is accepted and carried on :meth:`constraints` and on
/// :func:`prob_side_conditions` after every derived quantity. ``d <= 8``.
///
/// Example::
///
///     mvn = MultivariateNormal([mu1, mu2], [[s11, s12], [s12, s22]])
///     mvn.conditional([1], [a]).mean()  # [mu1 + s12*(a - mu2)/s22]
#[pyclass(name = "MultivariateNormal", module = "alkahest.experimental")]
pub(crate) struct PyMultivariateNormal {
    pub(crate) inner: CoreMvn,
    pub(crate) pool: Py<PyExprPool>,
}

#[pymethods]
impl PyMultivariateNormal {
    #[new]
    fn __new__(
        py: Python<'_>,
        mean: Bound<'_, PyAny>,
        cov: Bound<'_, PyAny>,
    ) -> PyResult<PyMultivariateNormal> {
        let pool = find_pool(py, &[&mean, &cov]).ok_or_else(|| {
            PyTypeError::new_err(
                "MultivariateNormal could not determine an ExprPool: at least one entry of mean \
                 or cov must be an Expr",
            )
        })?;
        let mu = vector(py, &pool, &mean)?;
        let sigma = matrix(py, &pool, &cov)?;
        let inner = {
            let p = pool.borrow(py);
            CoreMvn::new(mu, sigma, &p.inner).map_err(prob_error_to_py)?
        };
        Ok(PyMultivariateNormal { inner, pool })
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        let mean: Vec<String> = self
            .inner
            .mean_vector()
            .iter()
            .map(|&e| pool.inner.display(e).to_string())
            .collect();
        let cov: Vec<String> = self
            .inner
            .covariance_matrix()
            .iter()
            .map(|r| {
                let row: Vec<String> = r
                    .iter()
                    .map(|&e| pool.inner.display(e).to_string())
                    .collect();
                format!("[{}]", row.join(", "))
            })
            .collect();
        format!(
            "MultivariateNormal([{}], [{}])",
            mean.join(", "),
            cov.join(", ")
        )
    }

    /// The dimension ``d``.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// The mean vector, as a list of expressions.
    fn mean(&self, py: Python<'_>) -> Vec<PyExpr> {
        self.inner
            .mean_vector()
            .iter()
            .map(|&e| wrap(py, &self.pool, e))
            .collect()
    }

    /// The covariance matrix, as a list of rows of expressions.
    fn covariance(&self, py: Python<'_>) -> Vec<Vec<PyExpr>> {
        self.inner
            .covariance_matrix()
            .iter()
            .map(|r| r.iter().map(|&e| wrap(py, &self.pool, e)).collect())
            .collect()
    }

    /// ``Delta_k > 0`` for every leading principal minor, as predicates. Numeric
    /// ones were checked at construction; symbolic ones are the caller's to
    /// discharge.
    fn constraints(&self, py: Python<'_>) -> Vec<PyExpr> {
        let pool = self.pool.borrow(py);
        self.inner
            .constraints(&pool.inner)
            .into_iter()
            .map(|e| wrap(py, &self.pool, e))
            .collect()
    }

    /// The density ``(2 pi)^{-d/2} det(Sigma)^{-1/2} exp(-(x-mu)^T Sigma^{-1} (x-mu)/2)``
    /// at the point ``xs`` (a sequence of ``d`` expressions).
    ///
    /// Raises ``E-PROB-002`` when ``Sigma`` cannot be inverted symbolically.
    fn pdf(&self, py: Python<'_>, xs: Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let v = vector(py, &self.pool, &xs)?;
        derived(py, &self.pool, |p| self.inner.pdf(&v, p))
    }

    /// ``M(t) = E[exp(t^T X)] = exp(mu^T t + t^T Sigma t / 2)`` — entire.
    fn moment_generating_function(&self, py: Python<'_>, ts: Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let v = vector(py, &self.pool, &ts)?;
        derived(py, &self.pool, |p| {
            self.inner.moment_generating_function(&v, p)
        })
    }

    /// ``phi(t) = E[exp(i t^T X)] = exp(i mu^T t - t^T Sigma t / 2)``.
    fn characteristic_function(&self, py: Python<'_>, ts: Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let v = vector(py, &self.pool, &ts)?;
        derived(py, &self.pool, |p| {
            self.inner.characteristic_function(&v, p)
        })
    }

    /// The law of ``A X + b``: ``N(A mu + b, A Sigma A^T)``. ``A`` is ``m x d``
    /// (a :class:`~alkahest.Matrix` or list of rows), ``b`` an optional
    /// length-``m`` sequence.
    ///
    /// Raises ``E-PROB-001`` when ``A Sigma A^T`` is decidably singular (``A``
    /// without full row rank gives a degenerate normal, which has no density).
    #[pyo3(signature = (a, b=None))]
    fn linear_transform(
        &self,
        py: Python<'_>,
        a: Bound<'_, PyAny>,
        b: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyMultivariateNormal> {
        let a = matrix(py, &self.pool, &a)?;
        let b = b.map(|b| vector(py, &self.pool, &b)).transpose()?;
        let inner = {
            let pool = self.pool.borrow(py);
            self.inner
                .linear_transform(&a, b.as_deref(), &pool.inner)
                .map_err(prob_error_to_py)?
        };
        Ok(PyMultivariateNormal {
            inner,
            pool: self.pool.clone_ref(py),
        })
    }

    /// The marginal law of the components ``indices`` (0-based, in the order
    /// given).
    fn marginal(&self, py: Python<'_>, indices: Vec<i64>) -> PyResult<PyMultivariateNormal> {
        let idx = to_indices(indices, "marginal")?;
        let inner = self.inner.marginal(&idx).map_err(prob_error_to_py)?;
        Ok(PyMultivariateNormal {
            inner,
            pool: self.pool.clone_ref(py),
        })
    }

    /// The conditional law of the remaining components (in increasing index
    /// order) given ``X[indices] = values``:
    /// ``N(mu_1 + S12 S22^{-1}(a - mu_2), S11 - S12 S22^{-1} S21)``.
    fn conditional(
        &self,
        py: Python<'_>,
        indices: Vec<i64>,
        values: Bound<'_, PyAny>,
    ) -> PyResult<PyMultivariateNormal> {
        let idx = to_indices(indices, "conditional")?;
        let vals = vector(py, &self.pool, &values)?;
        let inner = {
            let pool = self.pool.borrow(py);
            self.inner
                .conditional(&idx, &vals, &pool.inner)
                .map_err(prob_error_to_py)?
        };
        Ok(PyMultivariateNormal {
            inner,
            pool: self.pool.clone_ref(py),
        })
    }

    /// The differential entropy ``h = (1/2) log((2 pi e)^d det Sigma)``
    /// (nats; ``base=2`` for bits). Independent of the mean; can be negative.
    #[pyo3(signature = (base=None))]
    fn entropy(&self, py: Python<'_>, base: Option<PyRef<PyExpr>>) -> PyResult<PyExpr> {
        let b = base_id(&self.pool, base)?;
        derived(py, &self.pool, |p| self.inner.entropy(b, p))
    }

    /// ``KL(self || other) = (1/2)[tr(S1^-1 S0) + (m1-m0)^T S1^-1 (m1-m0) - d +
    /// log(det S1 / det S0)]`` between two multivariate normals of the same
    /// dimension. Not symmetric. Raises ``E-PROB-002`` on a dimension mismatch.
    #[pyo3(signature = (other, base=None))]
    fn kl_divergence(
        &self,
        py: Python<'_>,
        other: PyRef<PyMultivariateNormal>,
        base: Option<PyRef<PyExpr>>,
    ) -> PyResult<PyExpr> {
        if !other.pool.is(&self.pool) {
            return Err(pool_mismatch_err());
        }
        let b = base_id(&self.pool, base)?;
        derived(py, &self.pool, |p| {
            self.inner.kl_divergence(&other.inner, b, p)
        })
    }

    /// The mutual information ``I(X_A; X_B)`` between the components
    /// ``a`` and ``b`` (disjoint lists of 0-based indices):
    /// ``-(1/2) log det(I - S_AA^-1 S_AB S_BB^-1 S_BA)``. For two scalar
    /// components with correlation ``rho`` this is ``-(1/2) log(1 - rho^2)``.
    #[pyo3(signature = (a, b, base=None))]
    fn mutual_information(
        &self,
        py: Python<'_>,
        a: Vec<i64>,
        b: Vec<i64>,
        base: Option<PyRef<PyExpr>>,
    ) -> PyResult<PyExpr> {
        let ia = to_indices(a, "mutual_information")?;
        let ib = to_indices(b, "mutual_information")?;
        let bs = base_id(&self.pool, base)?;
        derived(py, &self.pool, |p| {
            self.inner.mutual_information(&ia, &ib, bs, p)
        })
    }

    /// ``E[X^T A X] = tr(A Sigma) + mu^T A mu`` for a square ``A``.
    fn quadratic_form_expectation(&self, py: Python<'_>, a: Bound<'_, PyAny>) -> PyResult<PyExpr> {
        let a = matrix(py, &self.pool, &a)?;
        derived(py, &self.pool, |p| {
            self.inner.quadratic_form_expectation(&a, p)
        })
    }

    /// ``E[expr]`` with ``vars`` naming the components — shorthand for
    /// ``joint_expectation(expr, [(vars, self)])``.
    fn expectation(
        &self,
        py: Python<'_>,
        expr: PyRef<PyExpr>,
        vars: Vec<PyRef<PyExpr>>,
    ) -> PyResult<PyExpr> {
        if !expr.pool.is(&self.pool) || vars.iter().any(|v| !v.pool.is(&self.pool)) {
            return Err(pool_mismatch_err());
        }
        guard_depth(&self.pool.borrow(py).inner, expr.id)?;
        let blocks = vec![JointBlock::Gaussian(
            vars.iter().map(|v| v.id).collect(),
            self.inner.clone(),
        )];
        let e = expr.id;
        derived(py, &self.pool, |p| core_joint_expectation(e, &blocks, p))
    }
}

/// Parse ``[(var, Distribution) | ([vars], MultivariateNormal), ...]``.
fn blocks(
    py: Python<'_>,
    pool: &Py<PyExprPool>,
    ob: &Bound<'_, PyAny>,
) -> PyResult<Vec<JointBlock>> {
    let items: Vec<Bound<'_, PyAny>> = ob.extract().map_err(|_| {
        PyTypeError::new_err(
            "blocks must be a list of (var, Distribution) or ([vars], MultivariateNormal) pairs",
        )
    })?;
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        let (first, law): (Bound<'_, PyAny>, Bound<'_, PyAny>) = it.extract().map_err(|_| {
            PyTypeError::new_err(
                "each block must be a pair (var, Distribution) or ([vars], MultivariateNormal)",
            )
        })?;
        if let Ok(d) = law.extract::<PyRef<PyDistribution>>() {
            if !d.pool.is(pool) {
                return Err(pool_mismatch_err());
            }
            let v: PyRef<PyExpr> = first.extract()?;
            if !v.pool.is(pool) {
                return Err(pool_mismatch_err());
            }
            out.push(JointBlock::Univariate(v.id, d.inner.clone()));
        } else if let Ok(m) = law.extract::<PyRef<PyMultivariateNormal>>() {
            if !m.pool.is(pool) {
                return Err(pool_mismatch_err());
            }
            let vs: Vec<PyRef<PyExpr>> = first.extract()?;
            if vs.iter().any(|v| !v.pool.is(pool)) {
                return Err(pool_mismatch_err());
            }
            out.push(JointBlock::Gaussian(
                vs.iter().map(|v| v.id).collect(),
                m.inner.clone(),
            ));
        } else {
            let _ = py;
            return Err(PyTypeError::new_err(
                "the second element of a block must be a Distribution or a MultivariateNormal",
            ));
        }
    }
    Ok(out)
}

/// ``joint_expectation(expr, blocks)`` — ``E[expr]`` under a joint law of
/// **mutually independent** blocks.
///
/// ``blocks`` is a list whose entries are ``(var, Distribution)`` or
/// ``([vars], MultivariateNormal)``. Dependence exists only inside a
/// multivariate-normal block. ``expr`` is expanded and each term split into
/// per-block factors (independence: ``E[g(X) h(Y)] = E[g(X)] E[h(Y)]``);
/// a univariate factor goes through :func:`expectation` (quadrature-checked),
/// and a Gaussian block's factor must be a polynomial in its variates times
/// ``exp`` of an affine form, computed exactly by Stein's identity and
/// exponential tilting.
///
/// Raises ``E-PROB-002`` for a factor that couples two independent blocks
/// non-separably (``sin(X + Y)``), a non-polynomial Gaussian factor, or a
/// Gaussian moment of total degree above 12.
///
/// Example::
///
///     X, Y = pool.symbol("X"), pool.symbol("Y")
///     joint_expectation(X * Y, [([X, Y], mvn)])   # s12 + mu1*mu2
#[pyfunction]
#[pyo3(name = "joint_expectation")]
fn py_joint_expectation(
    py: Python<'_>,
    expr: PyRef<PyExpr>,
    blocks_ob: Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    let pool = expr.pool.clone_ref(py);
    guard_depth(&pool.borrow(py).inner, expr.id)?;
    let bl = blocks(py, &pool, &blocks_ob)?;
    let e = expr.id;
    derived(py, &pool, |p| core_joint_expectation(e, &bl, p))
}

/// ``covariance(e1, e2, blocks)`` — ``Cov[e1, e2] = E[e1 e2] - E[e1] E[e2]``
/// under a joint law of independent blocks (see :func:`joint_expectation`).
///
/// Example::
///
///     covariance(X + Y, X - Y, [([X, Y], mvn)])   # s11 - s22
#[pyfunction]
#[pyo3(name = "covariance")]
fn py_covariance(
    py: Python<'_>,
    e1: PyRef<PyExpr>,
    e2: PyRef<PyExpr>,
    blocks_ob: Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    if !e1.pool.is(&e2.pool) {
        return Err(pool_mismatch_err());
    }
    let pool = e1.pool.clone_ref(py);
    guard_depth_all(&pool.borrow(py).inner, &[e1.id, e2.id])?;
    let bl = blocks(py, &pool, &blocks_ob)?;
    let (a, b) = (e1.id, e2.id);
    derived(py, &pool, |p| core_covariance(a, b, &bl, p))
}

/// ``joint_variance(e, blocks)`` — ``Var[e] = Cov[e, e]`` under a joint law of
/// independent blocks: the correlated counterpart of
/// :func:`variance_affine_independent`, and not restricted to affine ``e``.
#[pyfunction]
#[pyo3(name = "joint_variance")]
fn py_joint_variance(
    py: Python<'_>,
    e: PyRef<PyExpr>,
    blocks_ob: Bound<'_, PyAny>,
) -> PyResult<PyExpr> {
    let pool = e.pool.clone_ref(py);
    guard_depth(&pool.borrow(py).inner, e.id)?;
    let bl = blocks(py, &pool, &blocks_ob)?;
    let id = e.id;
    derived(py, &pool, |p| core_joint_variance(id, &bl, p))
}

/// ``sum_independent([d1, d2, ...])`` — the law of ``X1 + X2 + ...`` for
/// **independent** summands, where it stays in a closed family:
/// ``Normal`` (means and variances add), ``Gamma``/``Exponential`` with a
/// common scale, ``Poisson``, ``Binomial``/``Bernoulli`` with a common ``p``
/// (symbolic ``n`` allowed), ``NegativeBinomial`` with a common ``p``.
///
/// "Common" means the same expression: two different symbols are different
/// parameters. Anything else raises ``E-PROB-002``.
#[pyfunction]
#[pyo3(name = "sum_independent")]
fn py_sum_independent(
    py: Python<'_>,
    dists: Vec<PyRef<PyDistribution>>,
) -> PyResult<PyDistribution> {
    let Some(first) = dists.first() else {
        return Err(PyValueError::new_err(
            "sum_independent needs at least one summand",
        ));
    };
    let pool = first.pool.clone_ref(py);
    if dists.iter().any(|d| !d.pool.is(&pool)) {
        return Err(pool_mismatch_err());
    }
    let inner: Vec<CoreDistribution> = dists.iter().map(|d| d.inner.clone()).collect();
    let out = {
        let p = pool.borrow(py);
        core_sum_independent(&inner, &p.inner).map_err(prob_error_to_py)?
    };
    Ok(PyDistribution { inner: out, pool })
}

/// The law of ``min`` or ``max`` of ``n`` iid copies of a base law —
/// returned by :func:`order_statistic`.
#[pyclass(name = "OrderStatistic", module = "alkahest.experimental")]
struct PyOrderStatistic {
    inner: CoreOrderStatistic,
    pool: Py<PyExprPool>,
}

#[pymethods]
impl PyOrderStatistic {
    fn __repr__(&self, py: Python<'_>) -> String {
        let pool = self.pool.borrow(py);
        let base = self.inner.base();
        let args: Vec<String> = base
            .params()
            .iter()
            .map(|&p| pool.inner.display(p).to_string())
            .collect();
        let kind = if self.inner.is_extreme() {
            self.kind().to_string()
        } else {
            format!("k={}", pool.inner.display(self.inner.rank(&pool.inner)))
        };
        format!(
            "OrderStatistic({}, {}({}), n={})",
            kind,
            base.kind().name(),
            args.join(", "),
            pool.inner.display(self.inner.n())
        )
    }

    /// ``"min"``, ``"max"`` or ``"kth"`` (the ``k``-th smallest; a literal
    /// ``k = 1`` or ``k = n`` is normalised to ``"min"`` / ``"max"``).
    #[getter]
    fn kind(&self) -> &'static str {
        if !self.inner.is_extreme() {
            return "kth";
        }
        match self.inner.kind() {
            OrderKind::Min => "min",
            _ => "max",
        }
    }

    /// The rank ``k`` (1-based, counted from the smallest): ``1`` for
    /// ``"min"``, ``n`` for ``"max"``.
    #[getter]
    fn k(&self, py: Python<'_>) -> PyExpr {
        let id = {
            let p = self.pool.borrow(py);
            self.inner.rank(&p.inner)
        };
        wrap(py, &self.pool, id)
    }

    /// The sample size ``n``.
    #[getter]
    fn n(&self, py: Python<'_>) -> PyExpr {
        wrap(py, &self.pool, self.inner.n())
    }

    /// ``P(M <= x)``: ``F(x)**n`` (max), ``1 - (1 - F(x))**n`` (min) or, for
    /// the ``k``-th smallest, the binomial tail
    /// ``sum_{j=k}^{n} C(n, j) F**j (1 - F)**(n - j)`` (a literal ``n`` only;
    /// a symbolic ``n`` or ``k`` raises ``E-PROB-004``: it is an incomplete
    /// beta). The base CDF is verified and placed exactly as
    /// :meth:`Distribution.cdf` does.
    fn cdf(&self, py: Python<'_>, x: PyRef<PyExpr>) -> PyResult<PyExpr> {
        if !x.pool.is(&self.pool) {
            return Err(pool_mismatch_err());
        }
        let id = x.id;
        derived(py, &self.pool, |p| self.inner.cdf(id, p))
    }

    /// The density on the support: ``n F**(n-1) f`` (max),
    /// ``n (1 - F)**(n-1) f`` (min), and for the ``k``-th smallest
    /// ``n!/((k-1)!(n-k)!) F**(k-1) (1 - F)**(n-k) f`` (a Gamma ratio when
    /// ``n`` or ``k`` is symbolic).
    fn pdf(&self, py: Python<'_>, x: PyRef<PyExpr>) -> PyResult<PyExpr> {
        if !x.pool.is(&self.pool) {
            return Err(pool_mismatch_err());
        }
        let id = x.id;
        derived(py, &self.pool, |p| self.inner.pdf(id, p))
    }

    /// ``E[M]``, verified against quadrature. ``Uniform(a, b)``:
    /// ``a + (b-a) n/(n+1)`` (max), ``a + (b-a)/(n+1)`` (min);
    /// ``Exponential(lam)``: ``H_n/lam = (digamma(n+1) - digamma(1))/lam``
    /// (max), ``1/(n lam)`` (min) -- for a symbolic ``n``. For every other
    /// case (the ``k``-th smallest, any other base law) ``n`` must be a
    /// literal (at most 24) and the mean is the expectation of
    /// ``x * f_(k)(x) / f(x)`` under the base law, computed by the integrator
    /// and gated: it closes for ``Uniform``, ``Exponential``, and ``Beta`` /
    /// ``Gamma`` with integer parameters, and raises ``E-PROB-003`` where the
    /// integrator declines (the ``erf`` CDF of a ``Normal``), ``E-PROB-002``
    /// for a symbolic ``n``.
    fn mean(&self, py: Python<'_>) -> PyResult<PyExpr> {
        derived(py, &self.pool, |p| self.inner.mean(p))
    }

    /// ``Var[M]``, verified against quadrature.
    fn variance(&self, py: Python<'_>) -> PyResult<PyExpr> {
        derived(py, &self.pool, |p| self.inner.variance(p))
    }

    /// The order statistic as a named :class:`Distribution` where it is one
    /// (``min`` of ``Exponential(lam)`` is ``Exponential(n lam)``; ``max``/``min``
    /// of ``Uniform(0, 1)`` are ``Beta(n, 1)``/``Beta(1, n)``), else ``None``.
    fn distribution(&self, py: Python<'_>) -> Option<PyDistribution> {
        let d = {
            let p = self.pool.borrow(py);
            self.inner.distribution(&p.inner)
        }?;
        Some(PyDistribution {
            inner: d,
            pool: self.pool.clone_ref(py),
        })
    }
}

/// ``order_statistic(dist, n, kind="max")`` — the law of the maximum (or, with
/// ``kind="min"``, the minimum) of ``n`` iid copies of ``dist``, for a literal
/// or **symbolic** ``n``.
///
/// With ``k=`` given, the ``k``-th smallest of ``n`` (1-based; ``k=1`` is the
/// minimum, ``k=n`` the maximum): density
/// ``n!/((k-1)!(n-k)!) F**(k-1) (1-F)**(n-k) f`` for **any continuous law with
/// a closed-form CDF** (``Normal``, ``LogNormal``, ``Uniform``,
/// ``Exponential``, ``Gamma`` with integer shape, ``Beta`` with integer
/// parameters), and the CDF as a binomial tail for a literal ``n``. Moments:
/// see :meth:`OrderStatistic.mean`.
///
/// A discrete base law raises ``E-PROB-002`` (ties change the density formula).
/// A numeric ``n`` or ``k`` that is not a positive integer, or ``k > n``,
/// raises ``E-PROB-001``; a symbolic ``n`` or ``k`` carries ``n > 0``,
/// integrality and ``n - k + 1 > 0`` on :func:`prob_side_conditions`.
///
/// Example::
///
///     order_statistic(Uniform(0, 1), n).mean()          # n/(n + 1)
///     order_statistic(Exponential(lam), n, "min").distribution()  # Exponential(n*lam)
///     order_statistic(Normal(0, 1), 5, k=2).pdf(x)      # 20 F (1-F)**3 f
#[pyfunction]
#[pyo3(name = "order_statistic", signature = (dist, n, kind=None, k=None))]
fn py_order_statistic(
    py: Python<'_>,
    dist: PyRef<PyDistribution>,
    n: Bound<'_, PyAny>,
    kind: Option<&str>,
    k: Option<Bound<'_, PyAny>>,
) -> PyResult<PyOrderStatistic> {
    let pool = dist.pool.clone_ref(py);
    let n_id = coerce_substituent(&pool, &n, py)?;
    let k = match (kind, k) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "order_statistic: pass either kind=\"min\"/\"max\" or k=<rank>, not both",
            ))
        }
        (_, Some(rank)) => {
            let rank_id = coerce_substituent(&pool, &rank, py)?;
            let inner = {
                let p = pool.borrow(py);
                core_kth_order_statistic(&dist.inner, n_id, rank_id, &p.inner)
                    .map_err(prob_error_to_py)?
            };
            return Ok(PyOrderStatistic { inner, pool });
        }
        (None | Some("max"), None) => OrderKind::Max,
        (Some("min"), None) => OrderKind::Min,
        (Some(other), None) => {
            return Err(PyValueError::new_err(format!(
                "order_statistic: kind must be \"min\" or \"max\", not {other:?}"
            )))
        }
    };
    let inner = {
        let p = pool.borrow(py);
        core_order_statistic(&dist.inner, n_id, k, &p.inner).map_err(prob_error_to_py)?
    };
    Ok(PyOrderStatistic { inner, pool })
}

/// ``NegativeBinomial(r, p)`` — the number of failures before the ``r``-th
/// success, mass ``Gamma(k+r)/(Gamma(r) k!) p**r (1-p)**k`` on ``{0, 1, 2, ...}``
/// (SciPy's ``nbinom(r, p)``), mean ``r(1-p)/p``. Requires ``r > 0`` (real
/// ``r`` allowed) and ``0 < p <= 1``.
#[pyfunction]
#[pyo3(name = "NegativeBinomial")]
fn py_dist_negative_binomial(
    py: Python<'_>,
    r: PyRef<PyExpr>,
    prob: PyRef<PyExpr>,
) -> PyResult<PyDistribution> {
    if !r.pool.is(&prob.pool) {
        return Err(pool_mismatch_err());
    }
    let (a, b) = (r.id, prob.id);
    build_dist(py, r.pool.clone_ref(py), &[a, b], |p| {
        CoreDistribution::negative_binomial(a, b, p)
    })
}

/// ``entropy_asymptotic(dist, base=None)`` -- the **large-parameter
/// asymptotic** Shannon entropy of a ``Binomial`` (symbolic ``n`` allowed),
/// ``NegativeBinomial`` or ``Poisson``, whose exact entropy has no closed form
/// (:meth:`Distribution.entropy` refuses them):
/// ``H ~ (1/2) log(2 pi e sigma^2) - gamma1^2/12`` with error ``O(sigma^-4)``,
/// ``sigma^2`` the variance and ``gamma1`` the skewness.
///
/// **This is an approximation**, returned by its own function and recorded in
/// the derivation under the rule ``prob_entropy_asymptotic_edgeworth`` so it
/// can never be mistaken for the exact entropy. Use it only where ``n p (1-p)``
/// (resp. ``r (1-p)/p^2``, ``lam``) is large.
///
/// Raises ``E-PROB-002`` for any other law, ``E-PROB-001`` for a numeric
/// ``base`` that is ``<= 0`` or ``1``.
///
/// Example::
///
///     entropy_asymptotic(Binomial(n, p))   # 1/2 + log(2 pi n p (1-p))/2 - (1-2p)^2/(12 n p (1-p))
#[pyfunction]
#[pyo3(name = "entropy_asymptotic", signature = (dist, base=None))]
fn py_entropy_asymptotic(
    py: Python<'_>,
    dist: PyRef<PyDistribution>,
    base: Option<PyRef<PyExpr>>,
) -> PyResult<PyExpr> {
    let pool = dist.pool.clone_ref(py);
    let b = base_id(&pool, base)?;
    derived(py, &pool, |p| core_entropy_asymptotic(&dist.inner, b, p))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_entropy_asymptotic, m)?)?;
    m.add_class::<PyMultivariateNormal>()?;
    m.add_class::<PyOrderStatistic>()?;
    m.add_function(wrap_pyfunction!(py_joint_expectation, m)?)?;
    m.add_function(wrap_pyfunction!(py_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(py_joint_variance, m)?)?;
    m.add_function(wrap_pyfunction!(py_sum_independent, m)?)?;
    m.add_function(wrap_pyfunction!(py_order_statistic, m)?)?;
    m.add_function(wrap_pyfunction!(py_dist_negative_binomial, m)?)?;
    Ok(())
}
