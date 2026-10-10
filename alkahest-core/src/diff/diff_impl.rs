use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprData, ExprId, ExprPool, IdMap};
use crate::poly::UniPoly;
use crate::simplify::engine::simplify;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

/// Build a canonical constant node for a rational `r`: an `Integer` when `r` is
/// integer-valued, otherwise a `Rational`.  Shared by all three diff modes for
/// the constant-exponent power rule.
pub(crate) fn const_node(pool: &ExprPool, r: rug::Rational) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.numer().clone())
    } else {
        let (n, d) = r.into_numer_denom();
        pool.rational(n, d)
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffError {
    /// An unknown function was encountered; differentiation is not defined.
    UnknownFunction(String),
    /// A `Pow` node whose exponent is not a constant integer.
    NonIntegerExponent,
    /// Forward-mode: unknown function (folded from the former `ForwardDiffError`).
    ForwardUnknownFunction(String),
    /// Forward-mode: non-integer exponent (folded from the former `ForwardDiffError`).
    ForwardNonIntegerExponent,
}

impl fmt::Display for DiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffError::UnknownFunction(name) => {
                write!(f, "cannot differentiate unknown function '{name}'")
            }
            DiffError::NonIntegerExponent => {
                write!(f, "cannot differentiate power with non-integer exponent")
            }
            DiffError::ForwardUnknownFunction(name) => {
                write!(f, "diff_forward: unknown function '{name}'")
            }
            DiffError::ForwardNonIntegerExponent => {
                write!(f, "diff_forward: non-integer exponent")
            }
        }
    }
}

impl std::error::Error for DiffError {}

impl crate::errors::AlkahestError for DiffError {
    fn code(&self) -> &'static str {
        match self {
            DiffError::UnknownFunction(_) => "E-DIFF-001",
            DiffError::NonIntegerExponent => "E-DIFF-002",
            DiffError::ForwardUnknownFunction(_) => "E-DIFF-003",
            DiffError::ForwardNonIntegerExponent => "E-DIFF-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            DiffError::UnknownFunction(_) => Some(
                "register the function in PrimitiveRegistry, or use diff_forward with a custom rule",
            ),
            DiffError::NonIntegerExponent => Some(
                "symbolic exponents require the chain rule; use diff_forward for non-integer powers",
            ),
            DiffError::ForwardUnknownFunction(_) => Some(
                "register the function in PrimitiveRegistry with diff_forward implemented",
            ),
            DiffError::ForwardNonIntegerExponent => Some(
                "substitute concrete values first; diff_forward requires integer exponents",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Symbolically differentiate `expr` with respect to `var`.
///
/// The returned log records every rule applied, including post-differentiation
/// simplification steps appended at the end.
///
/// # Cost on shared sub-expressions
///
/// Expressions are hash-consed DAGs, and the walk is memoised per call: each
/// distinct sub-expression is differentiated once and contributes its rule
/// step to the log once, so both the time and the log length are linear in
/// the number of distinct nodes (not in the number of root-to-leaf paths,
/// which is exponential for e.g. `e ← e·e + 1` iterated).
///
/// # The dense-polynomial fast path
///
/// When `expr` *as a whole* is a polynomial in `var` with integer
/// coefficients and its expanded form is not much larger than the input (see
/// `root_univariate_fastpath`), the derivative is taken on the dense FLINT
/// polynomial and returned expanded (rule `diff_univariate_poly`).  Otherwise
/// the ordinary rules apply everywhere, so a compact factored input keeps its
/// shape: `d/dx (x+1)^800 = 800·(x+1)^799`, not an 800-term expansion.
pub fn diff(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<DerivedExpr<ExprId>, DiffError> {
    if let Some(hit) = root_univariate_fastpath(expr, var, pool) {
        return Ok(hit.and_then(|v| simplify(v, pool)));
    }
    let memo = DiffMemo::for_call(var, pool);
    let _scope = memo.enter_scope();
    let mut log = DerivationLog::new();
    let value = diff_raw(expr, var, pool, &memo, &mut log)?;
    Ok(DerivedExpr::with_log(value, log).and_then(|v| simplify(v, pool)))
}

// ---------------------------------------------------------------------------
// Memo shared with re-entrant calls
// ---------------------------------------------------------------------------

type MemoTable = Rc<RefCell<IdMap<ExprId>>>;

/// One active top-level `diff` call: the pool and variable it is for and its
/// raw-derivative memo.
struct ActiveDiff {
    pool: usize,
    var: ExprId,
    memo: MemoTable,
}

thread_local! {
    /// Stack of the `diff` calls running on this thread.  A `PrimitiveRegistry`
    /// rule (`tan`, `atan`, `erf`, …) computes its argument's derivative by
    /// calling [`diff`] again; without sharing, `tan(tan(…tan(x)))` re-derived
    /// every inner argument from scratch at every level — exponential in the
    /// nesting depth.
    static ACTIVE_DIFFS: RefCell<Vec<ActiveDiff>> = const { RefCell::new(Vec::new()) };
}

/// Memo for one `diff` call.
///
/// Writes go only to `own`.  Reads fall back to `outer` — the memo of an
/// enclosing `diff` call on this thread for the same pool and variable, if
/// any.  An enclosing call reaches a registry primitive only after it has
/// differentiated that primitive's arguments itself (logging the steps), so
/// the nested call finds them there; keeping the outer table read-only means
/// nodes a nested call derives never pre-empt the outer call's own log steps.
/// Raw derivatives are a deterministic function of the node, so a value read
/// from either table is exactly what a fresh computation would produce.
struct DiffMemo {
    own: MemoTable,
    outer: Option<MemoTable>,
    pool: usize,
    var: ExprId,
}

struct ScopeGuard;

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        ACTIVE_DIFFS.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

impl DiffMemo {
    fn for_call(var: ExprId, pool: &ExprPool) -> Self {
        let pool_addr = pool as *const ExprPool as usize;
        let outer = ACTIVE_DIFFS.with(|s| {
            s.borrow()
                .iter()
                .rev()
                .find(|a| a.pool == pool_addr && a.var == var)
                .map(|a| Rc::clone(&a.memo))
        });
        DiffMemo {
            own: Rc::new(RefCell::new(IdMap::default())),
            outer,
            pool: pool_addr,
            var,
        }
    }

    /// Register this call as active until the returned guard drops.
    fn enter_scope(&self) -> ScopeGuard {
        ACTIVE_DIFFS.with(|s| {
            s.borrow_mut().push(ActiveDiff {
                pool: self.pool,
                var: self.var,
                memo: Rc::clone(&self.own),
            })
        });
        ScopeGuard
    }

    fn get(&self, e: ExprId) -> Option<ExprId> {
        if let Some(&v) = self.own.borrow().get(&e) {
            return Some(v);
        }
        self.outer
            .as_ref()
            .and_then(|o| o.borrow().get(&e).copied())
    }

    fn insert(&self, e: ExprId, v: ExprId) {
        self.own.borrow_mut().insert(e, v);
    }
}

// ---------------------------------------------------------------------------
// Core recursive differentiation (no simplification)
// ---------------------------------------------------------------------------

/// The primitive registry used for derivative dispatch.
///
/// A singleton built with `dispatch_registry`, for two reasons. It was
/// previously `default_registry()` called *inside* the recursive walk, so every
/// `Func` node rebuilt all 41 primitives — and `default_registry` additionally
/// probes each one's capability bundle, work this path never reads: it only
/// calls `diff_forward` and treats `None` as "unknown function". Adding a ball
/// kernel for `gamma` in 3.9.0 made that probe measurably more expensive and
/// surfaced as a ~14% regression on `test_series_sin_order12`, which
/// differentiates.
fn diff_registry() -> &'static crate::primitive::PrimitiveRegistry {
    static REGISTRY: std::sync::OnceLock<crate::primitive::PrimitiveRegistry> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(crate::primitive::PrimitiveRegistry::dispatch_registry)
}

/// `true` for `var`, `var^n`, and `c·var^n` with numeric `c` — inputs the power
/// rule already answers in closed form, where building a dense polynomial of
/// degree `n` would only cost time and memory.
fn is_monomial_in(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let is_var_power = |e: ExprId| {
        e == var
            || pool.with(e, |d| match d {
                ExprData::Pow { base, exp } => {
                    *base == var && pool.with(*exp, |x| matches!(x, ExprData::Integer(_)))
                }
                _ => false,
            })
    };
    if is_var_power(expr) {
        return true;
    }
    let args = match pool.get(expr) {
        ExprData::Mul(args) => args,
        _ => return false,
    };
    let mut powers = 0;
    for a in args {
        if is_var_power(a) {
            powers += 1;
        } else if !pool.with(a, |d| {
            matches!(
                d,
                ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_)
            )
        }) {
            return false;
        }
    }
    powers == 1
}

/// Upper bound on the degree in `var` of `expr` viewed as a polynomial with
/// integer coefficients, or `None` when it is not one (another symbol, a
/// function, a negative / symbolic / oversized exponent, a non-integer
/// coefficient).  Mirrors what [`UniPoly::from_symbolic`] accepts.
///
/// Memoised over the DAG; `memo.len()` is afterwards the number of distinct
/// nodes visited.  Saturating, so a huge `(…)^n` tower cannot overflow.
fn zpoly_degree_bound(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    memo: &mut IdMap<u64>,
) -> Option<u64> {
    if let Some(&d) = memo.get(&expr) {
        return Some(d);
    }
    enum Shape {
        Leaf(Option<u64>),
        Sum(Vec<ExprId>),
        Prod(Vec<ExprId>),
        Pow(ExprId, u32),
    }
    let shape = pool.with(expr, |data| match data {
        ExprData::Symbol { .. } if expr == var => Shape::Leaf(Some(1)),
        ExprData::Integer(_) => Shape::Leaf(Some(0)),
        ExprData::Rational(r) if *r.0.denom() == 1 => Shape::Leaf(Some(0)),
        ExprData::Add(args) => Shape::Sum(args.clone()),
        ExprData::Mul(args) => Shape::Prod(args.clone()),
        ExprData::Pow { base, exp } => match pool.with(*exp, |x| match x {
            ExprData::Integer(n) if n.0 >= 0 => n.0.to_u32(),
            _ => None,
        }) {
            Some(n) => Shape::Pow(*base, n),
            None => Shape::Leaf(None),
        },
        _ => Shape::Leaf(None),
    });
    let d = match shape {
        Shape::Leaf(d) => d?,
        Shape::Sum(args) => {
            let mut m = 0u64;
            for a in args {
                m = m.max(zpoly_degree_bound(a, var, pool, memo)?);
            }
            m
        }
        Shape::Prod(args) => {
            let mut s = 0u64;
            for a in args {
                s = s.saturating_add(zpoly_degree_bound(a, var, pool, memo)?);
            }
            s
        }
        Shape::Pow(base, n) => {
            zpoly_degree_bound(base, var, pool, memo)?.saturating_mul(u64::from(n))
        }
    };
    memo.insert(expr, d);
    Some(d)
}

/// Build the dense [`UniPoly`] of `expr` (already vetted by
/// [`zpoly_degree_bound`]) bottom-up with a per-node memo, so a shared
/// sub-expression is converted once.
fn build_zpoly(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    memo: &mut IdMap<UniPoly>,
) -> Option<UniPoly> {
    if let Some(p) = memo.get(&expr) {
        return Some(p.clone());
    }
    let p = match pool.get(expr) {
        ExprData::Add(args) => {
            let mut acc = UniPoly::zero(var);
            for a in args {
                let t = build_zpoly(a, var, pool, memo)?;
                acc = &acc + &t;
            }
            acc
        }
        ExprData::Mul(args) => {
            let mut acc = UniPoly::constant(var, 1);
            for a in args {
                let t = build_zpoly(a, var, pool, memo)?;
                acc = &acc * &t;
            }
            acc
        }
        ExprData::Pow { base, exp } => {
            let n = pool.with(exp, |x| match x {
                ExprData::Integer(n) => n.0.to_u32(),
                _ => None,
            })?;
            build_zpoly(base, var, pool, memo)?.checked_pow(n).ok()?
        }
        // `var` and integer constants: atoms, so the tree conversion is O(1).
        _ => UniPoly::from_symbolic(expr, var, pool).ok()?,
    };
    memo.insert(expr, p.clone());
    Some(p)
}

/// Dense ℤ-polynomial derivative, tried **once, at the root** of a [`diff`]
/// call — never at inner nodes.
///
/// Taken only when all of the following hold:
/// * `expr` is not an atom or a monomial `c·var^n` (the dedicated rules and
///   the power rule already answer those in closed form);
/// * `expr` is a polynomial in `var` with integer coefficients;
/// * the expanded result is not much larger than the input: the degree bound
///   `d` satisfies `d + 1 ≤ 2·(distinct DAG nodes)`.  This is what keeps a
///   compact factored input compact — `(x+1)^800` has 5 nodes and degree 800,
///   and `e ← e·e + 1` iterated 14 times has 29 nodes and degree 16384 — while
///   an already-expanded sum (the case the fast path is for) passes easily.
///
/// Both passes are memoised over the DAG, so the check is linear in the
/// number of distinct nodes.  This used to run at *every* non-atom node and
/// convert each by a tree walk, which made `Σ aᵢxⁱ` quadratic and nested
/// shared polynomials exponential.
fn root_univariate_fastpath(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<DerivedExpr<ExprId>> {
    if matches!(
        pool.get(expr),
        ExprData::Symbol { .. } | ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_)
    ) || is_monomial_in(expr, var, pool)
    {
        return None;
    }
    let mut deg_memo: IdMap<u64> = IdMap::default();
    let deg = zpoly_degree_bound(expr, var, pool, &mut deg_memo)?;
    let nodes = deg_memo.len() as u64;
    if deg.saturating_add(1) > nodes.saturating_mul(2) {
        return None;
    }
    let poly = build_zpoly(expr, var, pool, &mut IdMap::default())?;
    let der = poly.derivative();
    let result = der.to_symbolic_expr(pool);
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("diff_univariate_poly", expr, result));
    Some(DerivedExpr::with_log(result, log))
}

/// Memoised differentiation worker.
///
/// `memo` maps a node to its (unsimplified) derivative.  Each distinct node's
/// rule step is appended to `log` the first time it is differentiated, in
/// post-order (children before parent); a later occurrence of the same node
/// is a memo hit and adds nothing.
fn diff_raw(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    memo: &DiffMemo,
    log: &mut DerivationLog,
) -> Result<ExprId, DiffError> {
    // Return cached derivative for shared subexpressions.
    if let Some(cached) = memo.get(expr) {
        return Ok(cached);
    }

    // Extract only what we need from the pool in a single lock acquisition,
    // then release the lock before any recursive diff_raw calls.
    enum Node {
        IdentVar,
        Const,
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow {
            base: ExprId,
            exp: ExprId,
        },
        Func {
            name: String,
            args: Vec<ExprId>,
        },
        Piecewise {
            branches: Vec<(ExprId, ExprId)>,
            default: ExprId,
        },
        RootSum {
            poly: ExprId,
            rvar: ExprId,
            body: ExprId,
        },
    }

    let node = pool.with(expr, |data| match data {
        ExprData::Symbol { .. } if expr == var => Node::IdentVar,
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => Node::Const,
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => Node::Pow {
            base: *base,
            exp: *exp,
        },
        ExprData::Func { name, args } => Node::Func {
            name: name.clone(),
            args: args.clone(),
        },
        ExprData::Piecewise { branches, default } => Node::Piecewise {
            branches: branches.clone(),
            default: *default,
        },
        // Predicates have no algebraic derivative.
        ExprData::Predicate { .. } => Node::Const,
        ExprData::Forall { .. } | ExprData::Exists { .. } => Node::Const,
        ExprData::BigO(_) => Node::Const,
        ExprData::RootSum { poly, var, body } => Node::RootSum {
            poly: *poly,
            rvar: *var,
            body: *body,
        },
    });

    let result = match node {
        // d/dx x = 1
        Node::IdentVar => {
            let one = pool.integer(1_i32);
            log.push(RewriteStep::simple("diff_identity", expr, one));
            one
        }
        // d/dx c = 0  (any atom that is not the target variable)
        Node::Const => {
            let zero = pool.integer(0_i32);
            log.push(RewriteStep::simple("diff_const", expr, zero));
            zero
        }
        // Sum rule: d/dx (f₁ + f₂ + …) = f₁' + f₂' + …
        Node::Add(args) => {
            let mut dargs: Vec<ExprId> = Vec::with_capacity(args.len());
            for a in args {
                dargs.push(diff_raw(a, var, pool, memo, log)?);
            }
            let sum = pool.add(dargs);
            log.push(RewriteStep::simple("sum_rule", expr, sum));
            sum
        }
        // Product rule (n-ary Leibniz): d/dx (∏ᵢ fᵢ) = Σᵢ (fᵢ' · ∏_{j≠i} fⱼ)
        Node::Mul(args) => {
            let dargs: Vec<ExprId> = args
                .iter()
                .map(|&a| diff_raw(a, var, pool, memo, log))
                .collect::<Result<_, _>>()?;
            let mut terms: Vec<ExprId> = Vec::with_capacity(args.len());
            for (i, &di) in dargs.iter().enumerate() {
                let rest: Vec<ExprId> = args
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &a)| a)
                    .collect();
                let term = if rest.is_empty() {
                    di
                } else if rest.len() == 1 {
                    pool.mul(vec![di, rest[0]])
                } else {
                    let prod = pool.mul(rest);
                    pool.mul(vec![di, prod])
                };
                terms.push(term);
            }
            let result_id = match terms.len() {
                0 => pool.integer(0_i32),
                1 => terms[0],
                _ => pool.add(terms),
            };
            log.push(RewriteStep::simple("product_rule", expr, result_id));
            result_id
        }
        // Power rule, constant exponent (integer or rational):
        //   d/dx f^r = r · f^(r-1) · f'.
        // A var-dependent / non-constant exponent (e.g. x^y, x^x) is a different
        // rule (logarithmic differentiation) and remains unsupported.
        Node::Pow { base, exp } => {
            // Read the exponent without holding the pool lock during recursion.
            let r = pool
                .with(exp, |data| match data {
                    ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
                    ExprData::Rational(q) => Some(q.0.clone()),
                    _ => None,
                })
                .ok_or(DiffError::NonIntegerExponent)?;

            if r == 0 {
                // Special case r=0: d/dx f^0 = 0
                let zero = pool.integer(0_i32);
                log.push(RewriteStep::simple("power_rule_n0", expr, zero));
                zero
            } else if r == 1 {
                // Special case r=1: d/dx f^1 = f'
                let df = diff_raw(base, var, pool, memo, log)?;
                log.push(RewriteStep::simple("power_rule_n1", expr, df));
                df
            } else {
                let df = diff_raw(base, var, pool, memo, log)?;
                let r_id = const_node(pool, r.clone());
                let r_minus_1 = r - 1;
                // Emit `r·f` rather than `r·f^1·1` for the common `x²` case:
                // the monomial no longer takes the dense-polynomial fast path,
                // and leaving `f^1` / `·1` for `simplify` to fold costs more
                // than the whole derivative (CodSpeed `test_diff_sin_x_squared`).
                let base_pow = if r_minus_1 == 1 {
                    base
                } else {
                    pool.pow(base, const_node(pool, r_minus_1))
                };
                let one = pool.integer(1_i32);
                let result_id = if df == one {
                    pool.mul(vec![r_id, base_pow])
                } else {
                    pool.mul(vec![r_id, base_pow, df])
                };
                log.push(RewriteStep::simple("power_rule", expr, result_id));
                result_id
            }
        }
        // Chain rules for single-argument named functions
        Node::Func { name, args } if args.len() == 1 => {
            let f = args[0];
            let df = diff_raw(f, var, pool, memo, log)?;
            match name.as_str() {
                "sin" => {
                    let cos_f = pool.func("cos", vec![f]);
                    let r = pool.mul(vec![cos_f, df]);
                    log.push(RewriteStep::simple("diff_sin", expr, r));
                    r
                }
                "cos" => {
                    let sin_f = pool.func("sin", vec![f]);
                    let neg_one = pool.integer(-1_i32);
                    let r = pool.mul(vec![neg_one, sin_f, df]);
                    log.push(RewriteStep::simple("diff_cos", expr, r));
                    r
                }
                "exp" => {
                    let exp_f = pool.func("exp", vec![f]);
                    let r = pool.mul(vec![exp_f, df]);
                    log.push(RewriteStep::simple("diff_exp", expr, r));
                    r
                }
                "log" => {
                    let f_inv = pool.pow(f, pool.integer(-1_i32));
                    let r = pool.mul(vec![df, f_inv]);
                    log.push(RewriteStep::simple("diff_log", expr, r));
                    r
                }
                "sqrt" => {
                    let sqrt_f = pool.func("sqrt", vec![f]);
                    let two_sqrt = pool.mul(vec![pool.integer(2_i32), sqrt_f]);
                    let denom_inv = pool.pow(two_sqrt, pool.integer(-1_i32));
                    let r = pool.mul(vec![df, denom_inv]);
                    log.push(RewriteStep::simple("diff_sqrt", expr, r));
                    r
                }
                other => {
                    // Fall back to PrimitiveRegistry for V1-12 primitives.  The
                    // primitive differentiates `f` itself via a nested `diff`,
                    // which reads this call's memo (see `DiffMemo`).
                    let reg = diff_registry();
                    if let Some(d) = reg.diff_forward(other, &[f], var, pool) {
                        log.push(RewriteStep::simple("diff_primitive_registry", expr, d));
                        d
                    } else {
                        return Err(DiffError::UnknownFunction(other.to_string()));
                    }
                }
            }
        }
        // Multi-argument named functions: route through the PrimitiveRegistry.
        // The primitive's `diff_forward` computes each argument's derivative
        // internally (via `crate::diff::diff`) and returns the total chain-rule
        // derivative, so we only need to dispatch — no per-argument recursion
        // here, which avoids double-counting.  We still touch each argument via
        // `diff_raw` so the shared-subexpression memo stays consistent (and so
        // the nested `diff` calls find them memoised).
        Node::Func { name, args } => {
            for &a in &args {
                diff_raw(a, var, pool, memo, log)?;
            }
            let reg = diff_registry();
            if let Some(d) = reg.diff_forward(&name, &args, var, pool) {
                log.push(RewriteStep::simple("diff_primitive_registry", expr, d));
                d
            } else {
                return Err(DiffError::UnknownFunction(name));
            }
        }
        // PA-9: Piecewise diff distributes into branches.
        // d/dx Piecewise([(c₁,v₁), …], d) = Piecewise([(c₁, d/dx v₁), …], d/dx d)
        Node::Piecewise { branches, default } => {
            let mut new_branches = Vec::with_capacity(branches.len());
            for (cond, val) in branches {
                let dval = diff_raw(val, var, pool, memo, log)?;
                new_branches.push((cond, dval));
            }
            let ddefault = diff_raw(default, var, pool, memo, log)?;
            let result = pool.piecewise(new_branches, ddefault);
            log.push(RewriteStep::simple("diff_piecewise", expr, result));
            result
        }
        // d/dx Σ_{c:P(c)=0} body(c,x) = Σ_{c:P(c)=0} ∂body/∂x.
        // The root `c` (rvar) is constant in `x`; `poly` is free of `x`.
        Node::RootSum { poly, rvar, body } => {
            let dbody = diff_raw(body, var, pool, memo, log)?;
            let result = pool.root_sum(poly, rvar, dbody);
            log.push(RewriteStep::simple("diff_root_sum", expr, result));
            result
        }
    };
    memo.insert(expr, result);
    Ok(result)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use crate::poly::UniPoly;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn diff_constant() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.integer(5_i32), x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(0_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_const"));
    }

    #[test]
    fn diff_identity() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(x, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(1_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_identity"));
    }

    #[test]
    fn diff_other_variable() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let r = diff(y, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(0_i32));
    }

    #[test]
    fn diff_linear() {
        // d/dx (3x) = 3
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.integer(3_i32), x]);
        let r = diff(expr, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(3_i32));
    }

    #[test]
    fn diff_quadratic() {
        // d/dx x² = 2x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.pow(x, pool.integer(2_i32)), x, &pool).unwrap();
        let poly = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![0, 2]);
    }

    #[test]
    fn diff_cubic() {
        // d/dx x³ = 3x²
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.pow(x, pool.integer(3_i32)), x, &pool).unwrap();
        let poly = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![0, 0, 3]);
    }

    #[test]
    fn diff_polynomial() {
        // d/dx (x³ + 2x² + x + 1) = 3x² + 4x + 1
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(2_i32), pool.pow(x, pool.integer(2_i32))]),
            x,
            pool.integer(1_i32),
        ]);
        let r = diff(expr, x, &pool).unwrap();
        let poly = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        assert_eq!(poly.coefficients_i64(), vec![1, 4, 3]);
    }

    #[test]
    fn diff_sum_rule_logged() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let r = diff(pool.add(vec![x, y]), x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(1_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "sum_rule"));
    }

    #[test]
    fn diff_product_rule_logged() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let r = diff(pool.mul(vec![x, y]), x, &pool).unwrap();
        assert_eq!(r.value, y);
        assert!(r.log.steps().iter().any(|s| s.rule_name == "product_rule"));
    }

    #[test]
    fn diff_sin() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.func("sin", vec![x]), x, &pool).unwrap();
        assert_eq!(r.value, pool.func("cos", vec![x]));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_sin"));
    }

    #[test]
    fn diff_cos() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.func("cos", vec![x]), x, &pool).unwrap();
        // d/dx cos(x) = -sin(x) = Mul([-1, sin(x)]) in canonical arg order
        let sin_x = pool.func("sin", vec![x]);
        let neg_one = pool.integer(-1_i32);
        match pool.get(r.value) {
            ExprData::Mul(ref args) => {
                assert_eq!(args.len(), 2);
                assert!(args.contains(&neg_one) && args.contains(&sin_x));
            }
            _ => panic!("expected Mul, got {:?}", pool.display(r.value)),
        }
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_cos"));
    }

    #[test]
    fn diff_exp() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let r = diff(exp_x, x, &pool).unwrap();
        assert_eq!(r.value, exp_x);
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_exp"));
    }

    #[test]
    fn diff_log() {
        // d/dx log(x) = x^(-1)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(pool.func("log", vec![x]), x, &pool).unwrap();
        assert_eq!(r.value, pool.pow(x, pool.integer(-1_i32)));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_log"));
    }

    #[test]
    fn diff_chain_rule_sin() {
        // d/dx sin(x²): the inner x² is a monomial, so it takes the power rule —
        // the dense ℤ-polynomial fast path is tried only at the root of a call.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let r = diff(
            pool.func("sin", vec![pool.pow(x, pool.integer(2_i32))]),
            x,
            &pool,
        )
        .unwrap();
        assert_ne!(r.value, pool.integer(0_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "diff_sin"));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "power_rule"));
        assert!(!r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == "diff_univariate_poly"));
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let expected = pool.mul(vec![
            pool.func("cos", vec![pool.pow(x, pool.integer(2_i32))]),
            two_x,
        ]);
        assert_eq!(r.value, simplify(expected, &pool).value);
    }

    #[test]
    fn diff_pow_n0() {
        // d/dx f^0 = 0 — a monomial, so the power rule answers it directly.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(0_i32));
        let r = diff(expr, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(0_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "power_rule_n0"));
    }

    #[test]
    fn diff_pow_n1() {
        // d/dx x^1 — a monomial, so the power rule answers it directly.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(1_i32));
        let r = diff(expr, x, &pool).unwrap();
        assert_eq!(r.value, pool.integer(1_i32));
        assert!(r.log.steps().iter().any(|s| s.rule_name == "power_rule_n1"));
    }

    #[test]
    fn diff_unknown_function_error() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let err = diff(pool.func("zeta", vec![x]), x, &pool);
        assert!(matches!(err, Err(DiffError::UnknownFunction(_))));
    }

    #[test]
    fn diff_non_integer_exponent_error() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // A *var-dependent* exponent still needs logarithmic differentiation and
        // remains unsupported.
        let err = diff(pool.pow(x, y), x, &pool);
        assert!(matches!(err, Err(DiffError::NonIntegerExponent)));
    }

    #[test]
    fn diff_fractional_power() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // d/dx x^{1/2} = (1/2) x^{-1/2}.
        let half = pool.pow(x, pool.rational(1_i32, 2_i32));
        let d = diff(half, x, &pool).unwrap();
        let expected = pool.mul(vec![
            pool.rational(1_i32, 2_i32),
            pool.pow(x, pool.rational(-1_i32, 2_i32)),
        ]);
        assert_eq!(
            simplify(d.value, &pool).value,
            simplify(expected, &pool).value
        );

        // d/dx x^{2/3} = (2/3) x^{-1/3}.
        let two_thirds = pool.pow(x, pool.rational(2_i32, 3_i32));
        let d = diff(two_thirds, x, &pool).unwrap();
        let expected = pool.mul(vec![
            pool.rational(2_i32, 3_i32),
            pool.pow(x, pool.rational(-1_i32, 3_i32)),
        ]);
        assert_eq!(
            simplify(d.value, &pool).value,
            simplify(expected, &pool).value
        );
    }

    #[test]
    fn diff_fractional_power_chain_rule() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // d/dx (x²+1)^{3/2} = (3/2)(x²+1)^{1/2}·2x = 3x·(x²+1)^{1/2}.
        let base = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let expr = pool.pow(base, pool.rational(3_i32, 2_i32));
        let d = diff(expr, x, &pool).unwrap();
        let expected = pool.mul(vec![
            pool.integer(3_i32),
            x,
            pool.pow(base, pool.rational(1_i32, 2_i32)),
        ]);
        assert_eq!(
            simplify(d.value, &pool).value,
            simplify(expected, &pool).value
        );
    }

    #[test]
    fn diff_balanced_geom_series_univariate_fastpath() {
        fn balanced_sum(pool: &ExprPool, terms: &[ExprId]) -> ExprId {
            match terms.len() {
                0 => pool.integer(0_i32),
                1 => terms[0],
                _ => {
                    let mid = terms.len() / 2;
                    pool.add(vec![
                        balanced_sum(pool, &terms[..mid]),
                        balanced_sum(pool, &terms[mid..]),
                    ])
                }
            }
        }
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let n = 80i32;
        let mut terms = vec![pool.integer(1_i32)];
        for k in 1..=n {
            terms.push(pool.pow(x, pool.integer(k)));
        }
        let expr = balanced_sum(&pool, &terms);
        let r = diff(expr, x, &pool).unwrap();
        assert!(
            r.log
                .steps()
                .iter()
                .any(|s| s.rule_name == "diff_univariate_poly"),
            "expected dense ℤ-poly fast-path for balanced sum"
        );
        let poly = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        assert_eq!(poly.degree(), i64::from(n) - 1);
        let coeffs = poly.coefficients_i64();
        assert_eq!(coeffs.first().copied(), Some(1));
        assert_eq!(coeffs.last().copied(), Some(n as i64));
    }

    #[test]
    fn diff_log_has_both_diff_and_simplify_steps() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            y,
            pool.integer(0_i32),
        ]);
        let r = diff(expr, x, &pool).unwrap();
        let rules: Vec<&str> = r.log.steps().iter().map(|s| s.rule_name).collect();
        assert!(
            rules.contains(&"sum_rule"),
            "should have sum_rule: {rules:?}"
        );
        // `y` makes the whole sum non-univariate, and the fast path is tried
        // only at the root, so x² takes the power rule.
        assert!(
            rules.contains(&"power_rule"),
            "x² term differentiates via the power rule: {rules:?}"
        );
        assert!(rules.len() > 1, "log should have multiple steps: {rules:?}");
    }

    // -----------------------------------------------------------------------
    // Shared DAGs and the root-only fast path
    // -----------------------------------------------------------------------

    fn elapsed_under(t0: std::time::Instant, secs: u64, what: &str) {
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(secs),
            "{what} took {:?}",
            t0.elapsed()
        );
    }

    /// `e ← e·e + 1` iterated: 2k+1 distinct nodes, degree 2^k.  The whole-
    /// input fast path used to expand it densely (1.4 s at k = 12, and the
    /// Python `diff` of k = 14 took 36 s with an 18.7 MB derivation); the rules
    /// on the DAG are linear, with one log step per distinct node.
    #[test]
    fn diff_nested_square_dag_is_linear() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let build = |k: usize| {
            let mut e = x;
            for _ in 0..k {
                e = pool.add(vec![pool.mul(vec![e, e]), one]);
            }
            e
        };
        let t0 = std::time::Instant::now();
        let r = diff(build(40), x, &pool).unwrap();
        elapsed_under(t0, 2, "diff of e ← e·e+1, k = 40");
        let raw_steps = r
            .log
            .steps()
            .iter()
            .filter(|s| s.rule_name.starts_with("diff_") || s.rule_name.ends_with("_rule"))
            .count();
        assert!(
            raw_steps <= 5 * 40 + 2,
            "{raw_steps} diff steps for 81 nodes"
        );

        // Correctness where the dense form is still checkable: k = 5 (degree
        // 32, 11 nodes — past the fast-path size cap, so the rules run).
        let e = build(5);
        let r = diff(e, x, &pool).unwrap();
        assert!(!r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == "diff_univariate_poly"));
        let got = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        let want = UniPoly::from_symbolic(e, x, &pool).unwrap().derivative();
        assert_eq!(got, want);
    }

    /// `tan(tan(…tan(x)))`: `tan` is a registry primitive whose rule calls
    /// `diff` on its argument.  Each nested call used to start a fresh memo
    /// and re-derive the whole argument, 2^k work; it now reads the enclosing
    /// call's memo.
    #[test]
    fn diff_nested_registry_primitive_is_not_exponential() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let mut e = x;
        for _ in 0..40 {
            e = pool.func("tan", vec![e]);
        }
        let t0 = std::time::Instant::now();
        let r = diff(e, x, &pool).unwrap();
        elapsed_under(t0, 2, "diff of tan^40(x)");
        let registry_steps = r
            .log
            .steps()
            .iter()
            .filter(|s| s.rule_name == "diff_primitive_registry")
            .count();
        assert_eq!(registry_steps, 40, "one step per distinct tan node");

        // The value is right: d/dx tan(tan x) = sec²(tan x)·sec²(x).  `eval_f64`
        // has no `tan`, so rewrite tan(u) = sin(u)/cos(u) before evaluating.
        let t1 = pool.func("tan", vec![x]);
        let t2 = pool.func("tan", vec![t1]);
        let d = diff(t2, x, &pool).unwrap().value;
        let q = |u: ExprId| {
            pool.mul(vec![
                pool.func("sin", vec![u]),
                pool.pow(pool.func("cos", vec![u]), pool.integer(-1_i32)),
            ])
        };
        let mut m = std::collections::HashMap::new();
        m.insert(t1, q(x));
        m.insert(t2, q(q(x)));
        let d_sc = crate::kernel::subs::subs(d, &m, &pool);
        let mut env = std::collections::HashMap::new();
        env.insert(x, 0.3_f64);
        let got = crate::eval_f64(d_sc, &pool, &env).unwrap();
        let sec2 = |v: f64| 1.0 + v.tan() * v.tan();
        let want = sec2(0.3_f64.tan()) * sec2(0.3);
        assert!((got - want).abs() < 1e-12 * want.abs(), "{got} vs {want}");
    }

    /// A compact power keeps its shape: `d/dx (x+1)^800 = 800·(x+1)^799`, not
    /// an 800-term expansion.
    #[test]
    fn diff_power_of_sum_stays_factored() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let base = pool.add(vec![x, pool.integer(1_i32)]);
        let r = diff(pool.pow(base, pool.integer(800_i32)), x, &pool).unwrap();
        let expected = pool.mul(vec![
            pool.integer(800_i32),
            pool.pow(base, pool.integer(799_i32)),
        ]);
        assert_eq!(r.value, simplify(expected, &pool).value);
        // Mathematically the same polynomial as the dense derivative.
        let e20 = pool.pow(base, pool.integer(20_i32));
        let d20 = diff(e20, x, &pool).unwrap().value;
        assert_eq!(
            UniPoly::from_symbolic(d20, x, &pool).unwrap(),
            UniPoly::from_symbolic(e20, x, &pool).unwrap().derivative()
        );
    }

    /// The Chebyshev recurrence: ~3n nodes, degree n, ~fib(n) paths.  The dense
    /// fast path still applies at the root (the expansion is small), but the
    /// conversion is memoised over the DAG (550 ms at n = 24 before).
    #[test]
    fn diff_chebyshev_dag_is_polynomial() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let (two, m1) = (pool.integer(2_i32), pool.integer(-1_i32));
        let (mut a, mut b) = (pool.integer(1_i32), x);
        for _ in 1..40 {
            let c = pool.add(vec![pool.mul(vec![two, x, b]), pool.mul(vec![m1, a])]);
            a = b;
            b = c;
        }
        let t0 = std::time::Instant::now();
        let r = diff(b, x, &pool).unwrap();
        elapsed_under(t0, 2, "diff of Chebyshev T_40");
        assert!(r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == "diff_univariate_poly"));
        let d = UniPoly::from_symbolic(r.value, x, &pool).unwrap();
        assert_eq!(d.degree(), 39);
        // T_n'(1) = n².
        assert_eq!(
            d.eval_rational(&rug::Rational::from(1)),
            rug::Rational::from(1600)
        );
    }

    /// `Σ aᵢ·x^i`: the fast path used to run at every node, densely expanding
    /// each `x^i` (quadratic: 2.1 s at n = 2500, 12 s at n = 5000).
    #[test]
    fn diff_wide_sum_is_not_quadratic() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let n = 3000;
        let terms: Vec<ExprId> = (0..n)
            .map(|i| {
                let a = pool.symbol(format!("a{i}"), Domain::Real);
                pool.mul(vec![a, pool.pow(x, pool.integer(i as i64 + 1))])
            })
            .collect();
        let t0 = std::time::Instant::now();
        let r = diff(pool.add(terms), x, &pool).unwrap();
        elapsed_under(t0, 10, "diff of Σ aᵢ·x^i, n = 3000");
        let n_terms = pool.with(r.value, |d| match d {
            ExprData::Add(a) => a.len(),
            _ => 1,
        });
        assert_eq!(n_terms, n);
    }

    /// W5: the conditions of a Piecewise are carried over untouched and each
    /// branch value is differentiated.
    #[test]
    fn diff_piecewise_keeps_conditions_and_differentiates_branches() {
        use crate::kernel::expr::PredicateKind;
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let zero = pool.integer(0_i32);
        let c1 = pool.predicate(PredicateKind::Gt, vec![x, zero]);
        let c2 = pool.predicate(PredicateKind::Lt, vec![x, a]);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let sin_x = pool.func("sin", vec![x]);
        let inner = pool.piecewise(vec![(c2, sin_x)], x2);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let pw = pool.piecewise(vec![(c1, inner)], neg_x);
        let d = diff(pw, x, &pool).unwrap().value;
        let ExprData::Piecewise { branches, default } = pool.get(d) else {
            panic!("expected a Piecewise, got {}", pool.display(d));
        };
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].0, c1, "outer condition changed");
        assert_eq!(default, pool.integer(-1_i32));
        let ExprData::Piecewise {
            branches: ib,
            default: idef,
        } = pool.get(branches[0].1)
        else {
            panic!(
                "expected nested Piecewise, got {}",
                pool.display(branches[0].1)
            );
        };
        assert_eq!(ib[0].0, c2, "inner condition changed");
        assert_eq!(ib[0].1, pool.func("cos", vec![x]));
        let env: std::collections::HashMap<ExprId, f64> = [(x, 3.0)].into_iter().collect();
        assert_eq!(crate::jit::eval_interp(idef, &env, &pool), Some(6.0));
    }
}
