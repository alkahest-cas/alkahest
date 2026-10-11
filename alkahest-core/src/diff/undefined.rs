//! Undefined (user) functions under differentiation: `f(t)`, `q(t)`, `a(t)`.
//!
//! A function name the kernel attaches no meaning to — anything outside the
//! built-in table ([`crate::kernel::known_func_arity`]), the primitive
//! registry and a short list of reserved names — is an *undefined function*:
//! a smooth function about which nothing else is known.  `diff` used to refuse
//! such a node (`E-DIFF-001`), which left Euler–Lagrange, any metric with an
//! `a(t)` in it and every ODE written as `y(t)` to be hand-rolled with symbols.
//!
//! # Representation
//!
//! The derivative of an applied undefined function is a reserved function
//! node (the kernel's expression enum is frozen, so it is not a new variant):
//!
//! ```text
//! Derivative(f(u₁, …, uₙ), k₁, …, kₙ)        kᵢ ≥ 0 integers, Σkᵢ ≥ 1
//! ```
//!
//! meaning `∂^{k₁+…+kₙ} f / ∂x₁^{k₁}…∂xₙ^{kₙ}`, evaluated at `(u₁, …, uₙ)`.
//! The orders are counted **per argument slot**, not per variable, so:
//!
//! * it is canonical and hash-consed: mixed partials commute by construction
//!   (`∂x∂y f` and `∂y∂x f` are the same `[1, 1]` node);
//! * the chain rule is exact: `d/dt f(g(t)) = Derivative(f(g(t)), 1)·g'(t)` is
//!   `f'(g(t))·g'(t)`, never the meaningless "derivative of `f(g(t))` with
//!   respect to `g(t)`";
//! * substituting for a variable is ordinary structural substitution:
//!   `Derivative(f(t), 1)` at `t = 0` is `Derivative(f(0), 1)`, which is `f'(0)`.
//!
//! It prints the way SymPy does, so the printed form is valid input both here
//! and in SymPy: `Derivative(f(t), t)`, `Derivative(f(x, t), x, t)`, and, for a
//! slot whose argument is not a plain variable,
//! `Subs(Derivative(f(_xi_1), _xi_1), _xi_1, g(t))`.  See
//! `crate::kernel::display` and [`crate::parse()`].
//!
//! # What uses it
//!
//! [`crate::diff::diff`], [`crate::diff::diff_forward`] and
//! [`crate::diff::try_grad`] all differentiate undefined functions and their
//! derivatives; [`crate::kernel::subs::subs`] with a key `f(x)` (distinct
//! variables) differentiates its value into every `Derivative(f(x), …)`;
//! [`euler_lagrange`] and [`functions_to_symbols`] build the usual mechanics
//! workflow on top.  Numeric evaluation refuses an undefined function — it has
//! no value — and so refuses its derivatives too.

use std::collections::HashMap;
use std::fmt;

use crate::diff::diff_impl::{diff, DiffError};
use crate::errors::AlkahestError;
use crate::kernel::subs::mentions_var;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

/// Name of the reserved function node that carries a derivative of an
/// undefined function.  See the module documentation.
pub const DERIVATIVE_FUNC: &str = "Derivative";

/// Names that mean something even though no kernel table lists them: SymPy /
/// Mathematica spellings a user or a converter may produce, the reserved
/// derivative node itself, and internal markers.  None of them is a smooth
/// function nobody knows anything about, so none is differentiated as one.
const RESERVED: &[&str] = &[
    "Derivative",
    "Subs",
    "Integral",
    "Sum",
    "Product",
    "Limit",
    "Piecewise",
    "binomial",
    "factorial",
    "factorial2",
    "zeta",
    "qbinomial",
    "qpochhammer",
    "pochhammer",
    "cbrt",
    "sec",
    "csc",
    "cot",
    "sech",
    "csch",
    "coth",
    "ln",
    "log10",
    "log2",
    "exp2",
    "rand",
    "beta",
    "tgamma",
    "lgamma",
    "loggamma",
    "polygamma",
    "signum",
    "frac",
    "mod",
    "gcd",
    "lcm",
    "hyper",
    "erfi",
    "erfinv",
    "sinc",
    "harmonic",
    "fibonacci",
    "bernoulli",
    "catalan",
    "Abs",
    "Heaviside",
    "DiracDelta",
    "KroneckerDelta",
    "Max",
    "Min",
];

/// Whether `name` is an *undefined function*: a name the kernel attaches no
/// meaning to, which `diff` therefore treats as an arbitrary smooth function.
///
/// `false` for every built-in (`sin`, `floor`, `EllipticPi`, …), every
/// registered primitive, the reserved names (`Derivative`, `Subs`,
/// `binomial`, `zeta`, …), internal `__alkahest…` placeholders, and anything
/// that is not an identifier.
pub fn is_undefined_function_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_alphabetic() || first == '_') || !chars.all(|c| c.is_alphanumeric() || c == '_') {
        return false;
    }
    if name.starts_with("__alkahest") || RESERVED.contains(&name) {
        return false;
    }
    crate::kernel::known_func_arity(name).is_none() && !registry().is_registered(name)
}

fn registry() -> &'static crate::primitive::PrimitiveRegistry {
    static REGISTRY: std::sync::OnceLock<crate::primitive::PrimitiveRegistry> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(crate::primitive::PrimitiveRegistry::dispatch_registry)
}

/// A decoded applied undefined function or derivative of one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct UndefView {
    /// The applied function `f(u₁, …, uₙ)`.
    pub inner: ExprId,
    /// Its name `f`.
    pub name: String,
    /// The arguments `u₁, …, uₙ`.
    pub args: Vec<ExprId>,
    /// Per-slot derivative orders (all zero for `inner` itself).
    pub orders: Vec<u32>,
}

/// Decode `expr` as `f(u…)` (orders all zero) or a well-formed
/// `Derivative(f(u…), k…)`; `None` for anything else, including a malformed
/// `Derivative` node, which every consumer then treats as an unknown function.
pub(crate) fn view(expr: ExprId, pool: &ExprPool) -> Option<UndefView> {
    let (name, args) = pool.with(expr, |d| match d {
        ExprData::Func { name, args } => Some((name.clone(), args.clone())),
        _ => None,
    })?;
    if name == DERIVATIVE_FUNC {
        let (&inner, orders) = args.split_first()?;
        let (fname, fargs) = pool.with(inner, |d| match d {
            ExprData::Func { name, args } => Some((name.clone(), args.clone())),
            _ => None,
        })?;
        if !is_undefined_function_name(&fname) || fargs.is_empty() || fargs.len() != orders.len() {
            return None;
        }
        let orders: Vec<u32> = orders
            .iter()
            .map(|&k| {
                pool.with(k, |d| match d {
                    ExprData::Integer(n) => n.0.to_u32(),
                    _ => None,
                })
            })
            .collect::<Option<_>>()?;
        if orders.iter().all(|&k| k == 0) {
            return None;
        }
        return Some(UndefView {
            inner,
            name: fname,
            args: fargs,
            orders,
        });
    }
    if !is_undefined_function_name(&name) || args.is_empty() {
        return None;
    }
    let n = args.len();
    Some(UndefView {
        inner: expr,
        name,
        args,
        orders: vec![0; n],
    })
}

/// `Derivative(inner, orders…)`, or `inner` itself when every order is zero.
/// The caller guarantees `inner` is an applied undefined function with
/// `orders.len()` arguments.
pub(crate) fn build(inner: ExprId, orders: &[u32], pool: &ExprPool) -> ExprId {
    if orders.iter().all(|&k| k == 0) {
        return inner;
    }
    let mut args = Vec::with_capacity(orders.len() + 1);
    args.push(inner);
    args.extend(orders.iter().map(|&k| pool.integer(k)));
    pool.func(DERIVATIVE_FUNC, args)
}

/// The applied function and per-argument orders of a derivative node
/// `Derivative(f(u…), k…)`; `None` when `expr` is not one.
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::diff::{diff, undefined::derivative_parts};
///
/// let pool = ExprPool::new();
/// let t = pool.symbol("t", Domain::Real);
/// let f = pool.func("f", vec![t]);
/// let d2 = diff(diff(f, t, &pool).unwrap().value, t, &pool).unwrap().value;
/// assert_eq!(derivative_parts(d2, &pool), Some((f, vec![2])));
/// ```
pub fn derivative_parts(expr: ExprId, pool: &ExprPool) -> Option<(ExprId, Vec<u32>)> {
    let v = view(expr, pool)?;
    (v.inner != expr).then_some((v.inner, v.orders))
}

/// Build the derivative of the applied undefined function `func` of the given
/// per-argument `orders` — `Derivative(f(u…), k…)`, or `func` itself when every
/// order is zero.
///
/// # Errors
///
/// [`UndefinedFunctionError::NotAnAppliedFunction`] (`E-DIFF-005`) when `func`
/// is not an undefined function applied to at least one argument, or
/// [`UndefinedFunctionError::OrderMismatch`] (`E-DIFF-006`) when `orders` does
/// not have one entry per argument.
pub fn derivative_of(
    func: ExprId,
    orders: &[u32],
    pool: &ExprPool,
) -> Result<ExprId, UndefinedFunctionError> {
    let v = applied(func, pool)?;
    if orders.len() != v.args.len() {
        return Err(UndefinedFunctionError::OrderMismatch {
            expected: v.args.len(),
            got: orders.len(),
        });
    }
    Ok(build(func, orders, pool))
}

/// `func` decoded as a plain applied undefined function (orders zero).
fn applied(func: ExprId, pool: &ExprPool) -> Result<UndefView, UndefinedFunctionError> {
    match view(func, pool) {
        Some(v) if v.inner == func => Ok(v),
        _ => Err(UndefinedFunctionError::NotAnAppliedFunction(
            pool.display(func).to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failure of the undefined-function helpers ([`derivative_of`],
/// [`euler_lagrange`], [`functions_to_symbols`]).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum UndefinedFunctionError {
    /// The expression given as a coordinate / function is not an undefined
    /// function applied to its arguments (`E-DIFF-005`).
    NotAnAppliedFunction(String),
    /// A derivative order list whose length is not the function's arity
    /// (`E-DIFF-006`).
    OrderMismatch { expected: usize, got: usize },
    /// A coordinate `q(…)` that is not a function of the independent variable
    /// alone (`E-DIFF-007`).
    NotAFunctionOf { func: String, var: String },
    /// Differentiation of the Lagrangian failed (carries the `E-DIFF-00x` code
    /// of the underlying [`DiffError`]).
    Diff(DiffError),
}

impl fmt::Display for UndefinedFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UndefinedFunctionError::NotAnAppliedFunction(s) => write!(
                f,
                "'{s}' is not an undefined function applied to arguments, such as q(t)"
            ),
            UndefinedFunctionError::OrderMismatch { expected, got } => write!(
                f,
                "derivative orders: expected one per argument ({expected}), got {got}"
            ),
            UndefinedFunctionError::NotAFunctionOf { func, var } => {
                write!(f, "'{func}' is not a function of '{var}' alone")
            }
            UndefinedFunctionError::Diff(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for UndefinedFunctionError {}

impl AlkahestError for UndefinedFunctionError {
    fn code(&self) -> &'static str {
        match self {
            UndefinedFunctionError::NotAnAppliedFunction(_) => "E-DIFF-005",
            UndefinedFunctionError::OrderMismatch { .. } => "E-DIFF-006",
            UndefinedFunctionError::NotAFunctionOf { .. } => "E-DIFF-007",
            UndefinedFunctionError::Diff(e) => e.code(),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            UndefinedFunctionError::NotAnAppliedFunction(_) => Some(
                "pass the applied function, e.g. pool.func(\"q\", [t]); a built-in name (sin, gamma, …) or a reserved one (Derivative, binomial, …) is not an undefined function",
            ),
            UndefinedFunctionError::OrderMismatch { .. } => {
                Some("give one derivative order per argument of the function")
            }
            UndefinedFunctionError::NotAFunctionOf { .. } => Some(
                "each coordinate must be applied to the independent variable alone, e.g. q(t) for t",
            ),
            UndefinedFunctionError::Diff(e) => e.remediation(),
        }
    }
}

impl From<DiffError> for UndefinedFunctionError {
    fn from(e: DiffError) -> Self {
        UndefinedFunctionError::Diff(e)
    }
}

// ---------------------------------------------------------------------------
// Substitution into derivative nodes
// ---------------------------------------------------------------------------

/// The image of the derivative node `v` (with `v.orders` not all zero) when
/// `v.inner` is replaced by `value`: `∂^k value / ∂x^k` when the key's
/// arguments are distinct variables (so `value` defines the function), else
/// `None` (the function is only known at that one argument, which says nothing
/// about its derivative there).
pub(crate) fn differentiate_definition(
    v: &UndefView,
    value: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let mut seen = Vec::with_capacity(v.args.len());
    for &a in &v.args {
        let is_sym = pool.with(a, |d| matches!(d, ExprData::Symbol { .. }));
        if !is_sym || seen.contains(&a) {
            return None;
        }
        seen.push(a);
    }
    let mut cur = value;
    for (i, &k) in v.orders.iter().enumerate() {
        for _ in 0..k {
            cur = diff(cur, v.args[i], pool).ok()?.value;
        }
    }
    Some(cur)
}

/// Replace the undefined function `f` everywhere in `expr` by a definition:
/// `pattern = f(x₁, …, xₙ)` (distinct variables) and `body` in terms of the
/// `xᵢ`.  Every application `f(u…)` becomes `body[xᵢ := uᵢ]` and every
/// derivative `Derivative(f(u…), k…)` becomes `(∂^k body/∂x^k)[xᵢ := uᵢ]`, at
/// any argument — unlike [`crate::kernel::subs::subs`], which replaces only the
/// exact node `f(x…)`.  Nested applications (`f(f(t))`) are replaced inside
/// out.  This is SymPy's `subs(f, Lambda(x, body)).doit()`.
///
/// # Errors
///
/// `E-DIFF-005` when `pattern` is not an undefined function applied to
/// distinct variables; the [`DiffError`] code when `body` cannot be
/// differentiated as often as a derivative of `f` in `expr` requires.
///
/// # Example
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::diff::{diff, undefined::replace_function};
///
/// let pool = ExprPool::new();
/// let t = pool.symbol("t", Domain::Real);
/// let x = pool.symbol("x", Domain::Real);
/// let t2 = pool.pow(t, pool.integer(2));
/// // d/dt f(t²) = 2t·f'(t²); with f = sin it is 2t·cos(t²).
/// let e = diff(pool.func("f", vec![t2]), t, &pool).unwrap().value;
/// let r = replace_function(e, pool.func("f", vec![x]), pool.func("sin", vec![x]), &pool).unwrap();
/// let want = diff(pool.func("sin", vec![t2]), t, &pool).unwrap().value;
/// assert_eq!(alkahest_cas::simplify(r, &pool).value, want);
/// ```
pub fn replace_function(
    expr: ExprId,
    pattern: ExprId,
    body: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, UndefinedFunctionError> {
    let pv = applied(pattern, pool)?;
    let distinct = pv.args.iter().enumerate().all(|(i, &a)| {
        pool.with(a, |d| matches!(d, ExprData::Symbol { .. })) && !pv.args[..i].contains(&a)
    });
    if !distinct {
        return Err(UndefinedFunctionError::NotAnAppliedFunction(
            pool.display(pattern).to_string(),
        ));
    }
    // Every node that is `f(…)` or a derivative of it, children first.
    let mut order = Vec::new();
    let mut seen = std::collections::HashSet::new();
    collect_post_order(expr, pool, &mut seen, &mut order);
    let mut derivs: HashMap<Vec<u32>, ExprId> = HashMap::new();
    let mut map: HashMap<ExprId, ExprId> = HashMap::new();
    for n in order {
        let Some(v) = view(n, pool) else { continue };
        if v.name != pv.name || v.args.len() != pv.args.len() {
            continue;
        }
        let d = match derivs.get(&v.orders) {
            Some(&d) => d,
            None => {
                let mut cur = body;
                for (i, &k) in v.orders.iter().enumerate() {
                    for _ in 0..k {
                        cur = diff(cur, pv.args[i], pool)?.value;
                    }
                }
                derivs.insert(v.orders.clone(), cur);
                cur
            }
        };
        // The arguments with any inner application already replaced; then
        // all slots at once, so a body mentioning x₂ is not re-substituted
        // by the image of x₁.
        let slot: HashMap<ExprId, ExprId> = pv
            .args
            .iter()
            .zip(&v.args)
            .map(|(&x, &u)| (x, crate::kernel::subs::subs(u, &map, pool)))
            .collect();
        let image = crate::kernel::subs::subs(d, &slot, pool);
        map.insert(n, image);
    }
    Ok(crate::kernel::subs::subs(expr, &map, pool))
}

fn collect_post_order(
    n: ExprId,
    pool: &ExprPool,
    seen: &mut std::collections::HashSet<ExprId>,
    out: &mut Vec<ExprId>,
) {
    if !seen.insert(n) {
        return;
    }
    let children: Vec<ExprId> = pool.with(n, |d| match d {
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.clone(),
        ExprData::Func { args, .. } | ExprData::Predicate { args, .. } => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        ExprData::Piecewise { branches, default } => branches
            .iter()
            .flat_map(|&(c, v)| [c, v])
            .chain(std::iter::once(*default))
            .collect(),
        ExprData::BigO(a) => vec![*a],
        ExprData::RootSum { poly, body, .. } => vec![*poly, *body],
        ExprData::Forall { body, .. } | ExprData::Exists { body, .. } => vec![*body],
        _ => vec![],
    });
    for c in children {
        collect_post_order(c, pool, seen, out);
    }
    out.push(n);
}

// ---------------------------------------------------------------------------
// Coordinates ↔ symbols
// ---------------------------------------------------------------------------

/// One replaced node from [`functions_to_symbols`]: `node` (`q(t)` or a
/// `Derivative(q(t), k)`) stands for `symbol`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionSymbol {
    /// The coordinate `q(t)` this entry belongs to.
    pub func: ExprId,
    /// Derivative order: `0` for `q(t)` itself, `1` for `q'(t)`, ….
    pub order: u32,
    /// The node that was replaced.
    pub node: ExprId,
    /// The symbol that replaced it.
    pub symbol: ExprId,
}

/// Validate a coordinate `q(t)`: an undefined function of `t` alone.
fn coordinate(q: ExprId, t: ExprId, pool: &ExprPool) -> Result<String, UndefinedFunctionError> {
    let v = applied(q, pool)?;
    if v.args.as_slice() != [t] {
        return Err(UndefinedFunctionError::NotAFunctionOf {
            func: pool.display(q).to_string(),
            var: pool.display(t).to_string(),
        });
    }
    Ok(v.name)
}

/// Every derivative order of `q` that occurs in `expr` (always including 0).
fn orders_present(expr: ExprId, q: ExprId, pool: &ExprPool) -> Vec<u32> {
    let mut found = vec![0u32];
    let mut stack = vec![expr];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some((inner, ks)) = derivative_parts(n, pool) {
            if inner == q && !found.contains(&ks[0]) {
                found.push(ks[0]);
            }
        }
        pool.with(n, |d| match d {
            ExprData::Add(xs) | ExprData::Mul(xs) => stack.extend_from_slice(xs),
            ExprData::Func { args, .. } | ExprData::Predicate { args, .. } => {
                stack.extend_from_slice(args)
            }
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                stack.push(*exp);
            }
            ExprData::Piecewise { branches, default } => {
                for &(c, v) in branches {
                    stack.push(c);
                    stack.push(v);
                }
                stack.push(*default);
            }
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                stack.push(*var);
                stack.push(*body);
            }
            ExprData::BigO(a) => stack.push(*a),
            ExprData::RootSum { poly, var, body } => {
                stack.push(*poly);
                stack.push(*var);
                stack.push(*body);
            }
            _ => {}
        });
    }
    found.sort_unstable();
    found
}

/// A symbol named `base` (or `base_`, `base__`, … when that name is already
/// taken by a symbol of `expr`), in `domain`.
fn fresh_symbol(base: &str, taken: &[String], domain: Domain, pool: &ExprPool) -> ExprId {
    let mut name = base.to_string();
    while taken.iter().any(|n| n == &name) {
        name.push('_');
    }
    pool.symbol(name, domain)
}

fn symbol_names(exprs: &[ExprId], pool: &ExprPool) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack: Vec<ExprId> = exprs.to_vec();
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        pool.with(n, |d| match d {
            ExprData::Symbol { name, .. } => out.push(name.clone()),
            ExprData::Add(xs) | ExprData::Mul(xs) => stack.extend_from_slice(xs),
            ExprData::Func { args, .. } | ExprData::Predicate { args, .. } => {
                stack.extend_from_slice(args)
            }
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                stack.push(*exp);
            }
            ExprData::Piecewise { branches, default } => {
                for &(c, v) in branches {
                    stack.push(c);
                    stack.push(v);
                }
                stack.push(*default);
            }
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                stack.push(*var);
                stack.push(*body);
            }
            ExprData::BigO(a) => stack.push(*a),
            ExprData::RootSum { poly, var, body } => {
                stack.push(*poly);
                stack.push(*var);
                stack.push(*body);
            }
            _ => {}
        });
    }
    out
}

/// Conventional symbol name for the `order`-th derivative of `name`:
/// `q`, `q_dot`, `q_ddot`, then `q_d3`, `q_d4`, ….
fn derivative_symbol_name(name: &str, order: u32) -> String {
    match order {
        0 => name.to_string(),
        1 => format!("{name}_dot"),
        2 => format!("{name}_ddot"),
        k => format!("{name}_d{k}"),
    }
}

/// Replace each coordinate `q(t)` in `funcs` and every derivative
/// `Derivative(q(t), k)` of it that occurs in `exprs` by a fresh symbol
/// (`q`, `q_dot`, `q_ddot`, `q_d3`, … in `domain`; a trailing `_` is added
/// while the name is taken by a symbol already in `exprs`), so the result can
/// be handed to `solve`, `jacobian`, `lambdify` and friends.
///
/// Returns the rewritten expressions and the replacement table; substituting
/// each `symbol → node` back undoes the rewrite.  Derivative nodes are
/// replaced before `q(t)` itself, so `q'(t)` becomes `q_dot`, not
/// `Derivative(q, 1)`.
///
/// # Errors
///
/// `E-DIFF-005` / `E-DIFF-007` when an entry of `funcs` is not an undefined
/// function of `t` alone.
pub fn functions_to_symbols(
    exprs: &[ExprId],
    funcs: &[ExprId],
    t: ExprId,
    domain: Domain,
    pool: &ExprPool,
) -> Result<(Vec<ExprId>, Vec<FunctionSymbol>), UndefinedFunctionError> {
    let mut taken = symbol_names(exprs, pool);
    let mut table = Vec::new();
    for &q in funcs {
        let name = coordinate(q, t, pool)?;
        let mut orders = Vec::new();
        for &e in exprs {
            for k in orders_present(e, q, pool) {
                if !orders.contains(&k) {
                    orders.push(k);
                }
            }
        }
        orders.sort_unstable();
        for k in orders {
            let sym = fresh_symbol(&derivative_symbol_name(&name, k), &taken, domain, pool);
            if let ExprData::Symbol { name, .. } = pool.get(sym) {
                taken.push(name);
            }
            table.push(FunctionSymbol {
                func: q,
                order: k,
                node: build(q, &[k], pool),
                symbol: sym,
            });
        }
    }
    let map: HashMap<ExprId, ExprId> = table.iter().map(|e| (e.node, e.symbol)).collect();
    let out = exprs
        .iter()
        .map(|&e| crate::kernel::subs::subs(e, &map, pool))
        .collect();
    Ok((out, table))
}

// ---------------------------------------------------------------------------
// Euler–Lagrange
// ---------------------------------------------------------------------------

/// Euler–Lagrange equations of the Lagrangian `lagrangian` in the
/// coordinates `coords = [q₁(t), …, qₙ(t)]`.
///
/// Returns one expression per coordinate, each to be read as `= 0`:
///
/// ```text
/// EOMᵢ = d/dt (∂L/∂q̇ᵢ) − ∂L/∂qᵢ
/// ```
///
/// (Lagrange's sign convention; add generalized forces on the right).  The
/// partial derivatives treat `qᵢ`, `q̇ᵢ`, … as independent, and when `L`
/// contains higher derivatives `qᵢ^(k)` the Euler–Poisson terms
/// `(−1)^(k+1) dᵏ/dtᵏ ∂L/∂qᵢ^(k)` are included, so the result is the full
/// variational derivative.  The equations are written with `qᵢ(t)` and
/// `Derivative(qᵢ(t), t, …)` nodes; pass them through
/// [`functions_to_symbols`] to solve for the accelerations.
///
/// # Errors
///
/// `E-DIFF-005` / `E-DIFF-007` for a coordinate that is not an undefined
/// function of `t` alone; the [`DiffError`] code when `L` contains a function
/// that cannot be differentiated.
///
/// # Example
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::diff::{diff, undefined::euler_lagrange};
///
/// // Harmonic oscillator: L = q̇²/2 − q²/2  ⇒  q̈ + q = 0.
/// let pool = ExprPool::new();
/// let t = pool.symbol("t", Domain::Real);
/// let q = pool.func("q", vec![t]);
/// let qd = diff(q, t, &pool).unwrap().value;
/// let half = pool.rational(1, 2);
/// let l = pool.add(vec![
///     pool.mul(vec![half, pool.pow(qd, pool.integer(2))]),
///     pool.mul(vec![pool.rational(-1, 2), pool.pow(q, pool.integer(2))]),
/// ]);
/// let eom = euler_lagrange(l, &[q], t, &pool).unwrap();
/// let qdd = diff(qd, t, &pool).unwrap().value;
/// assert_eq!(eom[0], alkahest_cas::simplify(pool.add(vec![qdd, q]), &pool).value);
/// ```
pub fn euler_lagrange(
    lagrangian: ExprId,
    coords: &[ExprId],
    t: ExprId,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, UndefinedFunctionError> {
    for &q in coords {
        coordinate(q, t, pool)?;
    }
    // Complex placeholders: no real/positivity assumption may leak into the
    // partial derivatives of an expression that stands for q(t), q̇(t), ….
    let (ls, table) = functions_to_symbols(&[lagrangian], coords, t, Domain::Complex, pool)?;
    let l_sym = ls[0];
    let back: HashMap<ExprId, ExprId> = table.iter().map(|e| (e.symbol, e.node)).collect();
    let mut eoms = Vec::with_capacity(coords.len());
    for &q in coords {
        let mut terms = Vec::new();
        for entry in table.iter().filter(|e| e.func == q) {
            // ∂L/∂q^(k), as a function of t again.
            let partial = diff(l_sym, entry.symbol, pool)?.value;
            let mut term = crate::kernel::subs::subs(partial, &back, pool);
            for _ in 0..entry.order {
                term = diff(term, t, pool)?.value;
            }
            // (−1)^(k+1)
            if entry.order % 2 == 0 {
                term = pool.mul(vec![pool.integer(-1), term]);
            }
            terms.push(term);
        }
        let eom = match terms.len() {
            0 => pool.integer(0),
            1 => terms[0],
            _ => pool.add(terms),
        };
        eoms.push(crate::simplify::engine::simplify(eom, pool).value);
    }
    Ok(eoms)
}

/// `true` when `expr` mentions an undefined function or a derivative of one.
pub fn has_undefined_function(expr: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![expr];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if view(n, pool).is_some() {
            return true;
        }
        pool.with(n, |d| match d {
            ExprData::Add(xs) | ExprData::Mul(xs) => stack.extend_from_slice(xs),
            ExprData::Func { args, .. } | ExprData::Predicate { args, .. } => {
                stack.extend_from_slice(args)
            }
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                stack.push(*exp);
            }
            ExprData::Piecewise { branches, default } => {
                for &(c, v) in branches {
                    stack.push(c);
                    stack.push(v);
                }
                stack.push(*default);
            }
            ExprData::BigO(a) => stack.push(*a),
            ExprData::RootSum { poly, body, .. } => {
                stack.push(*poly);
                stack.push(*body);
            }
            _ => {}
        });
    }
    false
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

/// For each slot, whether its argument is a variable that occurs in no other
/// slot — the slots a derivative can be written "with respect to" directly.
pub(crate) fn clean_slots(v: &UndefView, pool: &ExprPool) -> Vec<bool> {
    (0..v.args.len())
        .map(|i| {
            let a = v.args[i];
            pool.with(a, |d| matches!(d, ExprData::Symbol { .. }))
                && v.args
                    .iter()
                    .enumerate()
                    .all(|(j, &b)| j == i || !mentions_var(b, a, pool))
        })
        .collect()
}

/// The SymPy spelling of a derivative node, which both parsers read back:
/// `Derivative(f(t), t, t)`, and for a slot whose argument is not a free-standing
/// variable a dummy `_xi_<slot>` bound by `Subs(…, _xi_<slot>, argument)`.
pub(crate) fn plain_form(v: &UndefView, pool: &ExprPool) -> String {
    let clean = clean_slots(v, pool);
    let mut slot_text = Vec::with_capacity(v.args.len());
    let mut dummies = Vec::new();
    for (i, &a) in v.args.iter().enumerate() {
        if v.orders[i] > 0 && !clean[i] {
            let name = format!("_xi_{}", i + 1);
            dummies.push((name.clone(), pool.display(a).to_string()));
            slot_text.push(name);
        } else {
            slot_text.push(pool.display(a).to_string());
        }
    }
    let mut wrt = Vec::new();
    for (i, &k) in v.orders.iter().enumerate() {
        for _ in 0..k {
            wrt.push(slot_text[i].clone());
        }
    }
    let mut out = format!(
        "{DERIVATIVE_FUNC}({}({}), {})",
        v.name,
        slot_text.join(", "),
        wrt.join(", ")
    );
    for (name, arg) in dummies {
        out = format!("Subs({out}, {name}, {arg})");
    }
    out
}

/// `Σᵢ Derivative(f(u), k + eᵢ) · duᵢ` over the slots whose argument mentions
/// `var`, with `du(i)` supplying `duᵢ`.  Shared by the three diff modes.
pub(crate) fn chain_rule(
    v: &UndefView,
    var: ExprId,
    pool: &ExprPool,
    mut du: impl FnMut(usize, ExprId) -> Result<ExprId, DiffError>,
) -> Result<ExprId, DiffError> {
    let mut terms = Vec::new();
    for (i, &u) in v.args.iter().enumerate() {
        if !mentions_var(u, var, pool) {
            continue;
        }
        let d = du(i, u)?;
        let mut k = v.orders.clone();
        k[i] = k[i].saturating_add(1);
        let node = build(v.inner, &k, pool);
        terms.push(if d == pool.integer(1) {
            node
        } else {
            pool.mul(vec![node, d])
        });
    }
    Ok(match terms.len() {
        0 => pool.integer(0),
        1 => terms[0],
        _ => pool.add(terms),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::subs::subs;
    use crate::simplify::engine::simplify;

    fn p() -> (ExprPool, ExprId) {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        (pool, t)
    }

    fn d(e: ExprId, v: ExprId, pool: &ExprPool) -> ExprId {
        diff(e, v, pool).unwrap().value
    }

    #[test]
    fn names() {
        assert!(is_undefined_function_name("f"));
        assert!(is_undefined_function_name("q1"));
        assert!(is_undefined_function_name("my_func"));
        for n in [
            "sin",
            "floor",
            "gamma",
            "EllipticPi",
            "Derivative",
            "binomial",
            "zeta",
            "",
            "1f",
            "f(x)",
            "__alkahest_grad_arg0",
            "Ei",
            "min",
        ] {
            assert!(!is_undefined_function_name(n), "{n}");
        }
    }

    #[test]
    fn first_and_second_derivative() {
        let (pool, t) = p();
        let f = pool.func("f", vec![t]);
        let d1 = d(f, t, &pool);
        assert_eq!(derivative_parts(d1, &pool), Some((f, vec![1])));
        assert_eq!(pool.display(d1).to_string(), "Derivative(f(t), t)");
        let d2 = d(d1, t, &pool);
        assert_eq!(derivative_parts(d2, &pool), Some((f, vec![2])));
        assert_eq!(pool.display(d2).to_string(), "Derivative(f(t), t, t)");
        // Independent of another variable.
        let x = pool.symbol("x", Domain::Real);
        assert_eq!(d(f, x, &pool), pool.integer(0));
    }

    #[test]
    fn mixed_partials_commute() {
        let (pool, t) = p();
        let x = pool.symbol("x", Domain::Real);
        let u = pool.func("u", vec![x, t]);
        let a = d(d(u, x, &pool), t, &pool);
        let b = d(d(u, t, &pool), x, &pool);
        assert_eq!(a, b);
        assert_eq!(derivative_parts(a, &pool), Some((u, vec![1, 1])));
        assert_eq!(pool.display(a).to_string(), "Derivative(u(x, t), x, t)");
    }

    #[test]
    fn chain_rule_through_inner_function() {
        // d/dt f(t²) = f'(t²)·2t.
        let (pool, t) = p();
        let t2 = pool.pow(t, pool.integer(2));
        let f = pool.func("f", vec![t2]);
        let r = d(f, t, &pool);
        let fp = build(f, &[1], &pool);
        let want = simplify(pool.mul(vec![pool.integer(2), t, fp]), &pool).value;
        assert_eq!(r, want);
        assert_eq!(
            pool.display(fp).to_string(),
            "Subs(Derivative(f(_xi_1), _xi_1), _xi_1, t^2)"
        );
    }

    #[test]
    fn subs_definition_differentiates_into_derivatives() {
        let (pool, t) = p();
        let f = pool.func("f", vec![t]);
        let e = pool.add(vec![d(d(f, t, &pool), t, &pool), f]);
        let sin_t = pool.func("sin", vec![t]);
        let map = HashMap::from([(f, sin_t)]);
        let r = simplify(subs(e, &map, &pool), &pool).value;
        assert_eq!(r, pool.integer(0)); // −sin t + sin t
    }

    #[test]
    fn subs_point_is_structural() {
        // f'(t) at t = 0 is f'(0) — not a derivative with respect to 0.
        let (pool, t) = p();
        let f = pool.func("f", vec![t]);
        let fp = d(f, t, &pool);
        let map = HashMap::from([(t, pool.integer(0))]);
        let r = subs(fp, &map, &pool);
        let f0 = pool.func("f", vec![pool.integer(0)]);
        assert_eq!(r, build(f0, &[1], &pool));
        // f(2t) := sin(...)? The key fixes f only at one argument, so its
        // derivative there is unknown and left alone.
        let f2t = pool.func("f", vec![pool.mul(vec![pool.integer(2), t])]);
        let e = d(f2t, t, &pool);
        let map = HashMap::from([(f2t, pool.func("sin", vec![t]))]);
        let r = subs(e, &map, &pool);
        assert!(has_undefined_function(r, &pool));
    }

    #[test]
    fn forward_and_reverse_agree() {
        let (pool, t) = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("f", vec![pool.mul(vec![x, t]), t]);
        let g = pool.mul(vec![pool.func("sin", vec![f]), d(f, x, &pool)]);
        let sym = d(g, t, &pool);
        let fwd = crate::diff::diff_forward(g, t, &pool).unwrap().value;
        let rev = crate::diff::try_grad(g, &[x, t], &pool).unwrap();
        let symx = d(g, x, &pool);
        // Compare numerically with f := X·sin(T) + exp(X·T²) instantiated.
        let xx = pool.symbol("X", Domain::Real);
        let tt = pool.symbol("T", Domain::Real);
        let body = pool.add(vec![
            pool.mul(vec![xx, pool.func("sin", vec![tt])]),
            pool.func(
                "exp",
                vec![pool.mul(vec![xx, pool.pow(tt, pool.integer(2))])],
            ),
        ]);
        let pat = pool.func("f", vec![xx, tt]);
        let at = |e: ExprId| {
            let r = replace_function(e, pat, body, &pool).unwrap();
            let m = HashMap::from([(x, 0.7), (t, -0.4)]);
            crate::jit::eval_interp(r, &m, &pool).unwrap()
        };
        for (a, b) in [(sym, fwd), (sym, rev[1]), (symx, rev[0])] {
            let (va, vb) = (at(a), at(b));
            assert!((va - vb).abs() < 1e-12 * va.abs().max(1.0), "{va} vs {vb}");
        }
        // And against a finite difference of the instantiated g.
        let gi = replace_function(g, pat, body, &pool).unwrap();
        let h = 1e-6;
        let ev = |tv: f64| {
            let m = HashMap::from([(x, 0.7), (t, tv)]);
            crate::jit::eval_interp(gi, &m, &pool).unwrap()
        };
        let fd = (ev(-0.4 + h) - ev(-0.4 - h)) / (2.0 * h);
        assert!((fd - at(sym)).abs() < 1e-6, "{fd} vs {}", at(sym));
    }

    #[test]
    fn printed_form_parses_back() {
        let (pool, t) = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("f", vec![t]);
        let u = pool.func("u", vec![x, t]);
        let w = pool.func("w", vec![t, t]);
        let ft2 = pool.func("f", vec![pool.pow(t, pool.integer(2))]);
        let cases = [
            build(f, &[1], &pool),
            build(f, &[3], &pool),
            build(u, &[2, 1], &pool),
            build(u, &[0, 1], &pool),
            build(w, &[1, 0], &pool),
            build(w, &[1, 1], &pool),
            build(ft2, &[2], &pool),
            pool.add(vec![
                pool.mul(vec![build(ft2, &[1], &pool), t]),
                pool.pow(build(f, &[1], &pool), pool.integer(2)),
            ]),
        ];
        for e in cases {
            let text = pool.display(e).to_string();
            let mut syms = HashMap::from([
                ("t".to_string(), t),
                ("x".to_string(), x),
                ("f".to_string(), f),
                ("u".to_string(), u),
                ("w".to_string(), w),
            ]);
            let back = crate::parse::parse(&text, &pool, &mut syms)
                .unwrap_or_else(|err| panic!("{text}: {err}"));
            assert_eq!(back, e, "{text} re-read as {}", pool.display(back));
        }
        // An undeclared unknown name is still refused.
        let mut syms = HashMap::new();
        assert!(crate::parse::parse("g(t)", &pool, &mut syms).is_err());
    }

    #[test]
    fn latex_and_unicode() {
        use crate::kernel::display::{render_latex, render_unicode};
        let (pool, t) = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("f", vec![t]);
        let u = pool.func("u", vec![x, t]);
        let ft2 = pool.func("f", vec![pool.pow(t, pool.integer(2))]);
        assert_eq!(render_unicode(build(f, &[1], &pool), &pool), "f′(t)");
        assert_eq!(render_unicode(build(f, &[4], &pool), &pool), "f⁽⁴⁾(t)");
        assert_eq!(
            render_unicode(build(u, &[1, 1], &pool), &pool),
            "∂²u(x, t)/∂x∂t"
        );
        assert_eq!(
            render_latex(build(f, &[2], &pool), &pool),
            r"\frac{d^{2}}{d t^{2}} \operatorname{f}\!\left(t\right)"
        );
        assert_eq!(
            render_latex(build(u, &[2, 1], &pool), &pool),
            r"\frac{\partial^{3}}{\partial x^{2} \partial t} \operatorname{u}\!\left(x, t\right)"
        );
        assert_eq!(
            render_latex(build(ft2, &[1], &pool), &pool),
            r"\operatorname{f}'\!\left(t^2\right)"
        );
        // Raised to a power, the Leibniz form is bracketed.
        let sq = pool.pow(build(f, &[1], &pool), pool.integer(2));
        assert!(render_latex(sq, &pool).starts_with(r"\left("));
    }

    #[test]
    fn replace_function_handles_nesting_and_chain_rule() {
        // d/dt f(f(t)) = f'(f(t))·f'(t); with f = exp: exp(exp t)·exp(t)·…
        let (pool, t) = p();
        let x = pool.symbol("x", Domain::Real);
        let ff = pool.func("f", vec![pool.func("f", vec![t])]);
        let e = d(ff, t, &pool);
        let body = pool.func("sin", vec![x]);
        let r = replace_function(e, pool.func("f", vec![x]), body, &pool).unwrap();
        let want = d(pool.func("sin", vec![pool.func("sin", vec![t])]), t, &pool);
        let diffr = simplify(
            pool.add(vec![r, pool.mul(vec![pool.integer(-1), want])]),
            &pool,
        );
        assert_eq!(diffr.value, pool.integer(0), "{}", pool.display(r));
        // Not a definition pattern.
        let bad = pool.func("f", vec![pool.integer(2)]);
        assert_eq!(
            replace_function(e, bad, body, &pool).unwrap_err().code(),
            "E-DIFF-005"
        );
    }

    #[test]
    fn euler_lagrange_pendulum() {
        // L = ½ m l² θ̇² + m g l cos θ  ⇒  m l² θ̈ + m g l sin θ = 0.
        let (pool, t) = p();
        let m = pool.symbol("m", Domain::Positive);
        let l = pool.symbol("l", Domain::Positive);
        let g = pool.symbol("g", Domain::Positive);
        let th = pool.func("theta", vec![t]);
        let thd = d(th, t, &pool);
        let lag = pool.add(vec![
            pool.mul(vec![
                pool.rational(1, 2),
                m,
                pool.pow(l, pool.integer(2)),
                pool.pow(thd, pool.integer(2)),
            ]),
            pool.mul(vec![m, g, l, pool.func("cos", vec![th])]),
        ]);
        let eom = euler_lagrange(lag, &[th], t, &pool).unwrap();
        let thdd = d(thd, t, &pool);
        let want = pool.add(vec![
            pool.mul(vec![m, pool.pow(l, pool.integer(2)), thdd]),
            pool.mul(vec![m, g, l, pool.func("sin", vec![th])]),
        ]);
        let diffr = simplify(
            pool.add(vec![eom[0], pool.mul(vec![pool.integer(-1), want])]),
            &pool,
        );
        assert_eq!(diffr.value, pool.integer(0), "{}", pool.display(eom[0]));
    }

    #[test]
    fn functions_to_symbols_round_trip() {
        let (pool, t) = p();
        let q = pool.func("q", vec![t]);
        let qd = d(q, t, &pool);
        let qdd = d(qd, t, &pool);
        let taken = pool.symbol("q_dot", Domain::Real);
        let e = pool.add(vec![qdd, pool.mul(vec![qd, q]), taken]);
        let (out, table) = functions_to_symbols(&[e], &[q], t, Domain::Real, &pool).unwrap();
        let names: Vec<String> = table
            .iter()
            .map(|e| pool.display(e.symbol).to_string())
            .collect();
        assert_eq!(names, ["q", "q_dot_", "q_ddot"]);
        assert!(!has_undefined_function(out[0], &pool));
        let back: HashMap<_, _> = table.iter().map(|e| (e.symbol, e.node)).collect();
        assert_eq!(subs(out[0], &back, &pool), e);
        // Not a coordinate.
        let err = functions_to_symbols(&[e], &[pool.func("sin", vec![t])], t, Domain::Real, &pool);
        assert_eq!(err.unwrap_err().code(), "E-DIFF-005");
        let x = pool.symbol("x", Domain::Real);
        let err = functions_to_symbols(&[e], &[pool.func("q", vec![x])], t, Domain::Real, &pool);
        assert_eq!(err.unwrap_err().code(), "E-DIFF-007");
    }
}
