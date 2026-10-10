//! StableHLO / XLA bridge.
//!
//! Converts a Alkahest symbolic expression to a StableHLO MLIR text module
//! that can be parsed by `jaxlib.mlir.dialects.stablehlo` or `mlir-opt`.
//! Every operation emitted is in the `stablehlo` dialect (plus `func.func` /
//! `return`); no `chlo` or `mhlo` op is used.
//!
//! # What lowers
//!
//! | Alkahest | StableHLO |
//! |---|---|
//! | `+`, `*`, `x^n`, `x^y` | `add`, `multiply`, `divide` / `power` (with a parity fix for odd exponents past `2^53`) |
//! | `sin`, `cos`, `exp`, `log` (= `ln`), `sqrt`, `tanh`, `abs`, `sign`, `floor`, `ceil` | the op of the same meaning |
//! | `round` | `round_nearest_afz` (half away from zero, as `f64::round`) |
//! | `atan2(y, x)` | `atan2` |
//! | `max(a, b, …)`, `min(a, b, …)` | a left fold of `maximum` / `minimum` |
//! | `tan` | `sine / cosine` |
//! | `sinh`, `cosh` | `(expm1(x) − expm1(−x))/2`, `(eˣ + e⁻ˣ)/2` |
//! | `atan` | `atan2(x, 1)` |
//! | `asin`, `acos` | `atan2(x, √((1−x)(1+x)))`, `atan2(√((1−x)(1+x)), x)` |
//! | `atanh` | `sign(x) · ½·log1p(2|x|/(1−|x|))` |
//! | `heaviside` | `(sign(x) + 1)/2` (so `θ(0) = ½`, as `eval_expr`) |
//! | `Piecewise` | a chain of `select` over `compare` / `and` / `or` / `not` |
//! | a predicate used as a number | `select(p, 1, 0)` |
//! | the symbol `pi` (when not an input) | its `f64` value |
//!
//! Every composition above is an exact identity of the real function, so the
//! only error it adds is floating-point rounding in the ops it uses.
//!
//! The NaN conventions are StableHLO's, not `eval_expr`'s: `maximum` /
//! `minimum` and `sign` propagate a NaN operand, where the evaluator's
//! `f64::max` drops it and its `sign` maps it to `0`.  Finite and infinite
//! inputs agree.
//!
//! # What is refused
//!
//! Everything else — `erf`, `gamma`, `lambert_w`, `asinh`, `acosh`, the
//! complex parts, user functions, unbound symbols, non-finite constants.
//! StableHLO has no op for these special functions and a polynomial
//! approximation would silently change the function exported, so
//! [`try_emit_stablehlo`] returns a [`StableHloError`] naming the offender
//! instead of a module.
//!
//! # Example
//! ```
//! use alkahest_cas::kernel::{Domain, ExprPool};
//! use alkahest_cas::stablehlo::{emit_stablehlo, try_emit_stablehlo};
//!
//! let pool = ExprPool::new();
//! let x = pool.symbol("x", Domain::Real);
//! let expr = pool.func("sin", vec![x]);
//! let mlir = emit_stablehlo(expr, &[x], "my_fn", &pool);
//! assert!(mlir.contains("stablehlo.sine"));
//!
//! let err = try_emit_stablehlo(pool.func("erf", vec![x]), &[x], "my_fn", &pool).unwrap_err();
//! assert_eq!(alkahest_cas::errors::AlkahestError::code(&err), "E-STABLEHLO-001");
//! ```

use crate::errors::AlkahestError;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{
    integer_is_exact_f64, integer_to_f64, rational_to_f64, try_predicate_bool_from_expr, ExprData,
    ExprId, ExprPool,
};
use std::collections::HashMap;
use std::fmt;

/// Why [`try_emit_stablehlo`] produced no module.
///
/// | Variant | Code |
/// |---|---|
/// | [`UnsupportedFunction`](Self::UnsupportedFunction) | `E-STABLEHLO-001` |
/// | [`UnsupportedNode`](Self::UnsupportedNode) | `E-STABLEHLO-002` |
/// | [`UnboundSymbol`](Self::UnboundSymbol) | `E-STABLEHLO-003` |
/// | [`NonFiniteConstant`](Self::NonFiniteConstant) | `E-STABLEHLO-004` |
/// | [`InvalidFunctionName`](Self::InvalidFunctionName) | `E-STABLEHLO-005` |
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StableHloError {
    /// A function with no exact StableHLO lowering (`erf`, `gamma`, a user
    /// function …), or a known function at an arity it does not take.
    UnsupportedFunction {
        /// The function's name, as stored in the expression.
        name: String,
        /// The number of arguments it was applied to.
        arity: usize,
    },
    /// An expression node that has no numeric value to compute (`O(x)`, a
    /// quantifier, a root sum …). The string names the node kind.
    UnsupportedNode(String),
    /// A symbol that is neither an input nor `pi`: the exported function would
    /// have no value for it.
    UnboundSymbol(String),
    /// An infinite or NaN constant, which has no MLIR float literal.
    NonFiniteConstant(f64),
    /// A function name that is not a valid MLIR bare identifier
    /// (`[A-Za-z_][A-Za-z0-9_$.]*`), so the module would not parse.
    InvalidFunctionName(String),
}

impl fmt::Display for StableHloError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StableHloError::UnsupportedFunction { name, arity } => write!(
                f,
                "to_stablehlo: no exact StableHLO lowering for {name}/{arity}; \
                 no module was emitted"
            ),
            StableHloError::UnsupportedNode(kind) => write!(
                f,
                "to_stablehlo: a {kind} node has no StableHLO lowering; no module was emitted"
            ),
            StableHloError::UnboundSymbol(name) => write!(
                f,
                "to_stablehlo: the expression depends on symbol {name}, which is not in the \
                 input list; no module was emitted"
            ),
            StableHloError::NonFiniteConstant(v) => write!(
                f,
                "to_stablehlo: the non-finite constant {v} has no MLIR float literal; \
                 no module was emitted"
            ),
            StableHloError::InvalidFunctionName(name) => write!(
                f,
                "to_stablehlo: {name:?} is not a valid MLIR function name \
                 ([A-Za-z_][A-Za-z0-9_$.]*)"
            ),
        }
    }
}

impl std::error::Error for StableHloError {}

impl AlkahestError for StableHloError {
    fn code(&self) -> &'static str {
        match self {
            StableHloError::UnsupportedFunction { .. } => "E-STABLEHLO-001",
            StableHloError::UnsupportedNode(_) => "E-STABLEHLO-002",
            StableHloError::UnboundSymbol(_) => "E-STABLEHLO-003",
            StableHloError::NonFiniteConstant(_) => "E-STABLEHLO-004",
            StableHloError::InvalidFunctionName(_) => "E-STABLEHLO-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            StableHloError::UnsupportedFunction { .. } => {
                "rewrite the function in terms of supported ones (see the stablehlo module \
                 docs), or substitute it before exporting"
            }
            StableHloError::UnsupportedNode(_) => {
                "export a plain numeric expression: remove O() terms, quantifiers and root sums"
            }
            StableHloError::UnboundSymbol(_) => {
                "add the symbol to the input list, or substitute a value for it"
            }
            StableHloError::NonFiniteConstant(_) => "replace the infinite or NaN constant",
            StableHloError::InvalidFunctionName(_) => {
                "use a name made of letters, digits, '_', '$' and '.', not starting with a digit"
            }
        })
    }
}

/// Emit a StableHLO MLIR text module for `expr` as a function named `fn_name`.
///
/// `inputs` gives the list of symbolic variables (in order) that become the
/// function arguments.  Returns the complete MLIR text, or an **empty string**
/// when [`try_emit_stablehlo`] refuses — use that function to learn why.
///
/// Returning nothing rather than a partial module is deliberate, and follows
/// the same rule as the Lean exporter: emit no artifact rather than an
/// incorrect one.
pub fn emit_stablehlo(expr: ExprId, inputs: &[ExprId], fn_name: &str, pool: &ExprPool) -> String {
    try_emit_stablehlo(expr, inputs, fn_name, pool).unwrap_or_default()
}

/// Emit a StableHLO MLIR text module for `expr`, or say why there is none.
///
/// The result is either a complete module whose function computes `expr` (up
/// to floating-point rounding — see the [module docs](self) for each lowering)
/// or a coded [`StableHloError`]; never a partial module and never one with a
/// stand-in for something it could not lower.  An exported program is run by
/// another toolchain that will never compare it back against the source
/// expression, so a wrong program is worse here than anywhere else.
pub fn try_emit_stablehlo(
    expr: ExprId,
    inputs: &[ExprId],
    fn_name: &str,
    pool: &ExprPool,
) -> Result<String, StableHloError> {
    if !is_mlir_bare_id(fn_name) {
        return Err(StableHloError::InvalidFunctionName(fn_name.to_string()));
    }
    let mut emitter = Emitter::new(inputs);
    let result_var = emitter.emit_expr(expr, pool)?;

    let args: Vec<String> = inputs
        .iter()
        .enumerate()
        .map(|(i, _)| format!("%arg{i}: tensor<f64>"))
        .collect();
    let args_str = args.join(", ");
    let mut body = emitter.body.join("\n    ");
    if !body.is_empty() {
        body.push_str("\n    ");
    }

    Ok(format!(
        r#"module {{
  func.func @{fn_name}({args_str}) -> tensor<f64> {{
    {body}return {result_var} : tensor<f64>
  }}
}}"#
    ))
}

/// MLIR `bare-id`: `(letter | '_') (letter | digit | '_' | '$' | '.')*`.
fn is_mlir_bare_id(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.'))
}

struct Emitter {
    arg_map: HashMap<ExprId, String>,
    /// SSA value already computed for a node, so a shared subexpression is
    /// emitted once (the pool is a DAG; re-emitting it is exponential).
    cache: HashMap<ExprId, String>,
    body: Vec<String>,
    counter: usize,
}

/// Render a finite `f64` as an MLIR float literal.
///
/// MLIR's grammar is `[-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?` — the decimal
/// point is **required**. Rust's `{:?}` gives the shortest round-tripping form,
/// which drops it whenever the mantissa is a single digit: `1e30`, `1e16` and
/// `5e-324` all come out without one. `mlir-opt` rejects each of those with
/// `error: expected '>'`, so `to_stablehlo` returned a module that does not
/// parse while reporting success — on constants a large integer or a small
/// rational reaches easily.
fn mlir_f64_literal(val: f64) -> String {
    let rendered = format!("{val:?}");
    if rendered.contains('.') {
        return rendered;
    }
    match rendered.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => format!("{mantissa}.0e{exponent}"),
        None => format!("{rendered}.0"),
    }
}

const F64: &str = "tensor<f64>";

impl Emitter {
    fn new(inputs: &[ExprId]) -> Self {
        let mut arg_map = HashMap::new();
        for (i, &id) in inputs.iter().enumerate() {
            arg_map.insert(id, format!("%arg{i}"));
        }
        Emitter {
            arg_map,
            cache: HashMap::new(),
            body: Vec::new(),
            counter: 0,
        }
    }

    fn fresh(&mut self) -> String {
        let v = format!("%v{}", self.counter);
        self.counter += 1;
        v
    }

    /// Emit `%v = stablehlo.<op> <operands> : tensor<f64>` and return `%v`.
    fn op(&mut self, op: &str, operands: &[&str]) -> String {
        let v = self.fresh();
        self.body.push(format!(
            "{v} = stablehlo.{op} {} : {F64}",
            operands.join(", ")
        ));
        v
    }

    /// `stablehlo.compare <dir>` of two `f64` scalars, yielding `tensor<i1>`.
    fn compare(&mut self, dir: &str, a: &str, b: &str) -> String {
        let v = self.fresh();
        self.body.push(format!(
            "{v} = stablehlo.compare {dir}, {a}, {b}, FLOAT : ({F64}, {F64}) -> tensor<i1>"
        ));
        v
    }

    /// `stablehlo.select` between two `f64` values.
    fn select(&mut self, pred: &str, on_true: &str, on_false: &str) -> String {
        let v = self.fresh();
        self.body.push(format!(
            "{v} = stablehlo.select {pred}, {on_true}, {on_false} : tensor<i1>, {F64}"
        ));
        v
    }

    /// A logical op on `tensor<i1>` values.
    fn bool_op(&mut self, op: &str, operands: &[&str]) -> String {
        let v = self.fresh();
        self.body.push(format!(
            "{v} = stablehlo.{op} {} : tensor<i1>",
            operands.join(", ")
        ));
        v
    }

    fn bool_const(&mut self, value: bool) -> String {
        let v = self.fresh();
        self.body.push(format!(
            "{v} = stablehlo.constant dense<{value}> : tensor<i1>"
        ));
        v
    }

    /// Emit an `f64` constant as a *float* literal.
    ///
    /// `{val}` renders `1.0_f64` as `1`, and MLIR rejects a decimal integer
    /// literal for an `f64` tensor ("unexpected decimal integer"). Every
    /// expression containing an integer constant — `x**2 + 1` — therefore
    /// emitted a module that would not parse. [`mlir_f64_literal`] fixes that
    /// and the second half of the same problem: `{val:?}` does *not* always
    /// render a decimal point.
    fn emit_const_f64(&mut self, val: f64) -> Result<String, StableHloError> {
        if !val.is_finite() {
            return Err(StableHloError::NonFiniteConstant(val));
        }
        let v = self.fresh();
        let lit = mlir_f64_literal(val);
        self.body
            .push(format!("{v} = stablehlo.constant dense<{lit}> : {F64}"));
        Ok(v)
    }

    fn emit_expr(&mut self, expr: ExprId, pool: &ExprPool) -> Result<String, StableHloError> {
        if let Some(s) = self.arg_map.get(&expr) {
            return Ok(s.clone());
        }
        if let Some(s) = self.cache.get(&expr) {
            return Ok(s.clone());
        }
        let v = self.emit_uncached(expr, pool)?;
        self.cache.insert(expr, v.clone());
        Ok(v)
    }

    fn emit_uncached(&mut self, expr: ExprId, pool: &ExprPool) -> Result<String, StableHloError> {
        match pool.get(expr) {
            // `to_i64().unwrap_or(0)` used to sit here, so an integer past
            // `i64` became the *constant zero* and `to_stablehlo(2^70·x + 1)`
            // emitted valid MLIR for `x·0 + 1`. Rounding to `f64` loses one
            // ulp on a large coefficient instead of losing the whole term, and
            // it is what every other backend does with the same literal.
            ExprData::Integer(n) => self.emit_const_f64(integer_to_f64(&n.0)),
            ExprData::Float(f) => self.emit_const_f64(f.inner.to_f64()),
            ExprData::Rational(r) => self.emit_const_f64(rational_to_f64(&r.0)),
            // Inputs were handled by the caller; an unbound `pi` is π, as in
            // every evaluator in the crate.
            ExprData::Symbol { name, .. } => {
                if name == crate::eval::symbols::PI_NAME {
                    self.emit_const_f64(std::f64::consts::PI)
                } else {
                    Err(StableHloError::UnboundSymbol(name))
                }
            }
            ExprData::Add(args) => self.fold(&args, "add", pool),
            ExprData::Mul(args) => self.fold(&args, "multiply", pool),
            ExprData::Pow { base, exp } => self.emit_pow(base, exp, pool),
            ExprData::Func { name, args } => self.emit_func(&name, &args, pool),
            ExprData::Piecewise { branches, default } => {
                // Built back to front so the *first* true condition wins.
                let mut acc = self.emit_expr(default, pool)?;
                for &(cond, value) in branches.iter().rev() {
                    let c = self.emit_pred(cond, pool)?;
                    let val = self.emit_expr(value, pool)?;
                    acc = self.select(&c, &val, &acc);
                }
                Ok(acc)
            }
            ExprData::Predicate { .. } => {
                let p = self.emit_pred(expr, pool)?;
                let one = self.emit_const_f64(1.0)?;
                let zero = self.emit_const_f64(0.0)?;
                Ok(self.select(&p, &one, &zero))
            }
            other => Err(StableHloError::UnsupportedNode(node_kind(&other).into())),
        }
    }

    /// Left fold of a binary op over `args` (an `Add`/`Mul` or `max`/`min`).
    fn fold(
        &mut self,
        args: &[ExprId],
        op: &str,
        pool: &ExprPool,
    ) -> Result<String, StableHloError> {
        let Some((&first, rest)) = args.split_first() else {
            return Err(StableHloError::UnsupportedNode(format!("empty {op}")));
        };
        let mut acc = self.emit_expr(first, pool)?;
        for &a in rest {
            let operand = self.emit_expr(a, pool)?;
            acc = self.op(op, &[&acc, &operand]);
        }
        Ok(acc)
    }

    /// A predicate node as a `tensor<i1>`.
    fn emit_pred(&mut self, pred: ExprId, pool: &ExprPool) -> Result<String, StableHloError> {
        // A comparison of constants is decided exactly, as `eval_expr` decides
        // it: rounding both sides to `f64` would make `10^30 + 1 = 10^30` true.
        if let Some(b) = try_predicate_bool_from_expr(pred, pool) {
            return Ok(self.bool_const(b));
        }
        let ExprData::Predicate { kind, args } = pool.get(pred) else {
            return Err(StableHloError::UnsupportedNode(
                "non-predicate Piecewise condition".into(),
            ));
        };
        let arity_ok = match kind {
            PredicateKind::True | PredicateKind::False => args.is_empty(),
            PredicateKind::Not => args.len() == 1,
            PredicateKind::And | PredicateKind::Or => !args.is_empty(),
            _ => args.len() == 2,
        };
        if !arity_ok {
            return Err(StableHloError::UnsupportedNode(format!(
                "{kind} predicate with {} arguments",
                args.len()
            )));
        }
        match kind {
            PredicateKind::True => Ok(self.bool_const(true)),
            PredicateKind::False => Ok(self.bool_const(false)),
            PredicateKind::Not => {
                let a = self.emit_pred(args[0], pool)?;
                Ok(self.bool_op("not", &[&a]))
            }
            PredicateKind::And | PredicateKind::Or => {
                let op = if kind == PredicateKind::And {
                    "and"
                } else {
                    "or"
                };
                let mut acc = self.emit_pred(args[0], pool)?;
                for &a in &args[1..] {
                    let b = self.emit_pred(a, pool)?;
                    acc = self.bool_op(op, &[&acc, &b]);
                }
                Ok(acc)
            }
            // IEEE comparisons, as `eval_expr`'s `<`, `==`, `!=` on `f64`:
            // every ordered comparison with a NaN is false and `NE` is true.
            PredicateKind::Lt
            | PredicateKind::Le
            | PredicateKind::Gt
            | PredicateKind::Ge
            | PredicateKind::Eq
            | PredicateKind::Ne => {
                let dir = match kind {
                    PredicateKind::Lt => "LT",
                    PredicateKind::Le => "LE",
                    PredicateKind::Gt => "GT",
                    PredicateKind::Ge => "GE",
                    PredicateKind::Eq => "EQ",
                    _ => "NE",
                };
                let a = self.emit_expr(args[0], pool)?;
                let b = self.emit_expr(args[1], pool)?;
                Ok(self.compare(dir, &a, &b))
            }
        }
    }

    fn emit_pow(
        &mut self,
        base: ExprId,
        exp: ExprId,
        pool: &ExprPool,
    ) -> Result<String, StableHloError> {
        let exp_int = pool.with(exp, |d| match d {
            ExprData::Integer(n) => n.0.to_i64(),
            _ => None,
        });
        let base_v = self.emit_expr(base, pool)?;
        match exp_int {
            Some(-1) => {
                let one = self.emit_const_f64(1.0)?;
                return Ok(self.op("divide", &[&one, &base_v]));
            }
            Some(2) => return Ok(self.op("multiply", &[&base_v, &base_v])),
            Some(0) => return self.emit_const_f64(1.0),
            _ => {}
        }
        // An odd integer exponent that `f64` cannot hold rounds to an
        // *even* double (every double past 2^53 is even), so a plain
        // `power` loses the sign: `x^(2^63+1)` at `x = -1` gave `+1`.
        // Emit `|x|^n` and negate it where `x < 0` — the same parity
        // correction `eval_const::pow_f64_integer_exponent` makes.
        let wide_odd = pool.with(exp, |d| match d {
            ExprData::Integer(n) => n.0.is_odd() && !integer_is_exact_f64(&n.0),
            _ => false,
        });
        let exp_v = self.emit_expr(exp, pool)?;
        if wide_odd {
            let abs_v = self.op("abs", &[&base_v]);
            let mag = self.op("power", &[&abs_v, &exp_v]);
            let neg = self.op("negate", &[&mag]);
            let zero = self.emit_const_f64(0.0)?;
            let is_neg = self.compare("LT", &base_v, &zero);
            return Ok(self.select(&is_neg, &neg, &mag));
        }
        Ok(self.op("power", &[&base_v, &exp_v]))
    }

    fn emit_func(
        &mut self,
        name: &str,
        args: &[ExprId],
        pool: &ExprPool,
    ) -> Result<String, StableHloError> {
        let unsupported = || StableHloError::UnsupportedFunction {
            name: name.to_string(),
            arity: args.len(),
        };
        // Arity first: a node built with the unchecked constructor can carry
        // the wrong number of arguments, and indexing `args[0]` of `sin()`
        // would panic.
        let (lo, hi) = match name {
            "atan2" => (2, 2),
            "max" | "min" => (2, usize::MAX),
            "sin" | "cos" | "tan" | "exp" | "log" | "ln" | "sqrt" | "tanh" | "sinh" | "cosh"
            | "abs" | "sign" | "floor" | "ceil" | "round" | "heaviside" | "atan" | "asin"
            | "acos" | "atanh" => (1, 1),
            _ => return Err(unsupported()),
        };
        if !(lo..=hi).contains(&args.len()) {
            return Err(unsupported());
        }
        if name == "max" || name == "min" {
            let op = if name == "max" { "maximum" } else { "minimum" };
            return self.fold(args, op, pool);
        }
        if name == "atan2" {
            let y = self.emit_expr(args[0], pool)?;
            let x = self.emit_expr(args[1], pool)?;
            return Ok(self.op("atan2", &[&y, &x]));
        }

        let x = self.emit_expr(args[0], pool)?;
        let direct = match name {
            "sin" => Some("sine"),
            "cos" => Some("cosine"),
            "exp" => Some("exponential"),
            "log" | "ln" => Some("log"),
            "sqrt" => Some("sqrt"),
            "tanh" => Some("tanh"),
            "abs" => Some("abs"),
            "sign" => Some("sign"),
            "floor" => Some("floor"),
            "ceil" => Some("ceil"),
            // `f64::round` rounds half away from zero.
            "round" => Some("round_nearest_afz"),
            _ => None,
        };
        if let Some(op) = direct {
            return Ok(self.op(op, &[&x]));
        }

        match name {
            "tan" => {
                let s = self.op("sine", &[&x]);
                let c = self.op("cosine", &[&x]);
                Ok(self.op("divide", &[&s, &c]))
            }
            "sinh" => {
                // (expm1(x) − expm1(−x))/2: no cancellation near 0, where
                // (eˣ − e⁻ˣ)/2 loses every digit of a tiny x.
                let neg = self.op("negate", &[&x]);
                let ep = self.op("exponential_minus_one", &[&x]);
                let en = self.op("exponential_minus_one", &[&neg]);
                let d = self.op("subtract", &[&ep, &en]);
                let half = self.emit_const_f64(0.5)?;
                Ok(self.op("multiply", &[&d, &half]))
            }
            "cosh" => {
                let neg = self.op("negate", &[&x]);
                let ep = self.op("exponential", &[&x]);
                let en = self.op("exponential", &[&neg]);
                let s = self.op("add", &[&ep, &en]);
                let half = self.emit_const_f64(0.5)?;
                Ok(self.op("multiply", &[&s, &half]))
            }
            "atan" => {
                let one = self.emit_const_f64(1.0)?;
                Ok(self.op("atan2", &[&x, &one]))
            }
            "asin" | "acos" => {
                // √(1 − x²) as √((1−x)(1+x)), which keeps its digits near
                // |x| = 1; for |x| > 1 it is NaN, as the real function is.
                let one = self.emit_const_f64(1.0)?;
                let a = self.op("subtract", &[&one, &x]);
                let b = self.op("add", &[&one, &x]);
                let p = self.op("multiply", &[&a, &b]);
                let r = self.op("sqrt", &[&p]);
                Ok(if name == "asin" {
                    self.op("atan2", &[&x, &r])
                } else {
                    self.op("atan2", &[&r, &x])
                })
            }
            "atanh" => {
                // atanh|x| = ½·log1p(2|x|/(1−|x|)), then the sign of x. Both
                // pieces are relatively accurate for every |x| < 1; |x| = 1
                // gives ±∞ and |x| > 1 gives NaN, as the real function does.
                let t = self.op("abs", &[&x]);
                let one = self.emit_const_f64(1.0)?;
                let den = self.op("subtract", &[&one, &t]);
                let two_t = self.op("add", &[&t, &t]);
                let q = self.op("divide", &[&two_t, &den]);
                let l = self.op("log_plus_one", &[&q]);
                let half = self.emit_const_f64(0.5)?;
                let m = self.op("multiply", &[&l, &half]);
                let s = self.op("sign", &[&x]);
                Ok(self.op("multiply", &[&m, &s]))
            }
            "heaviside" => {
                // (sign(x) + 1)/2 is 0, ½, 1 for x <, =, > 0 — the evaluator's
                // θ(0) = ½ convention.
                let s = self.op("sign", &[&x]);
                let one = self.emit_const_f64(1.0)?;
                let p = self.op("add", &[&s, &one]);
                let half = self.emit_const_f64(0.5)?;
                Ok(self.op("multiply", &[&p, &half]))
            }
            _ => Err(unsupported()),
        }
    }
}

/// A short name for an expression node kind, for error messages.
fn node_kind(data: &ExprData) -> &'static str {
    match data {
        ExprData::BigO(_) => "O()",
        ExprData::Forall { .. } => "Forall",
        ExprData::Exists { .. } => "Exists",
        ExprData::RootSum { .. } => "RootSum",
        _ => "non-numeric",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn emit_sin() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let sin_x = p.func("sin", vec![x]);
        let mlir = emit_stablehlo(sin_x, &[x], "test_fn", &p);
        assert!(mlir.contains("stablehlo.sine"), "missing sin: {mlir}");
        assert!(mlir.contains("func.func @test_fn"), "missing func: {mlir}");
    }

    #[test]
    fn emit_add() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let expr = p.add(vec![x, y]);
        let mlir = emit_stablehlo(expr, &[x, y], "add_fn", &p);
        assert!(mlir.contains("stablehlo.add"), "missing add: {mlir}");
    }

    #[test]
    fn emit_mul() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let expr = p.mul(vec![x, x]);
        let mlir = emit_stablehlo(expr, &[x], "mul_fn", &p);
        assert!(mlir.contains("stablehlo.multiply"), "missing mul: {mlir}");
    }

    #[test]
    fn emit_constant() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let three = p.integer(3_i32);
        let expr = p.mul(vec![three, x]);
        let mlir = emit_stablehlo(expr, &[x], "const_fn", &p);
        assert!(mlir.contains("stablehlo.constant"), "missing const: {mlir}");
    }

    #[test]
    fn emit_module_structure() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let expr = p.func("exp", vec![x]);
        let mlir = emit_stablehlo(expr, &[x], "exp_fn", &p);
        assert!(
            mlir.starts_with("module {"),
            "should start with module: {mlir}"
        );
        assert!(mlir.contains("return"), "should have return: {mlir}");
    }
}

#[cfg(test)]
mod emitter_soundness_tests {
    use super::*;
    use crate::kernel::Domain;

    /// Float tensors need a float literal.
    ///
    /// `{val}` renders `1.0_f64` as `1`, and MLIR rejects a decimal integer for
    /// an `f64` tensor. Every expression with an integer constant — `x**2 + 1`
    /// — emitted a module that would not parse, which IREE reports as
    /// "unexpected decimal integer".
    #[test]
    fn integer_constants_emit_float_literals() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![pool.mul(vec![x, x]), pool.integer(1_i32)]);

        let src = emit_stablehlo(expr, &[x], "f", &pool);
        assert!(
            src.contains("dense<1.0>"),
            "constant must carry a decimal point, got:\n{src}"
        );
        assert!(
            !src.contains("dense<1>"),
            "bare integer literal in f64 tensor"
        );
    }

    /// An unsupported function must yield *nothing*, not a stand-in.
    ///
    /// The emitter used to push a comment and return the constant 0, so
    /// `to_stablehlo(erf(x))` produced a module evaluating to zero everywhere.
    /// An exported program is executed by another toolchain that never compares
    /// it back against the source expression, so a wrong program is worse here
    /// than anywhere else in the library.
    #[test]
    fn unsupported_functions_emit_nothing() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for name in ["erf", "lambert_w", "asinh", "digamma"] {
            let expr = pool.func(name, vec![x]);
            let src = emit_stablehlo(expr, &[x], "f", &pool);
            assert!(
                src.is_empty(),
                "{name} emitted a module instead of refusing:\n{src}"
            );
        }
    }

    /// No emitted module may contain a stand-in constant where an operation
    /// was meant to be — the shape of the old bug, stated directly.
    #[test]
    fn no_module_contains_an_unsupported_marker() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for name in [
            "sin", "cos", "exp", "log", "sqrt", "tanh", "abs", "tan", "sinh", "cosh",
        ] {
            let expr = pool.func(name, vec![x]);
            let src = emit_stablehlo(expr, &[x], "f", &pool);
            assert!(!src.is_empty(), "{name} should be emittable");
            assert!(
                !src.contains("unsupported"),
                "{name} emitted an unsupported marker:\n{src}"
            );
        }
    }

    /// Functions with a direct StableHLO primitive use it.
    #[test]
    fn direct_primitives_are_used() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for (name, op) in [("tanh", "stablehlo.tanh"), ("abs", "stablehlo.abs")] {
            let src = emit_stablehlo(pool.func(name, vec![x]), &[x], "f", &pool);
            assert!(src.contains(op), "{name} should emit {op}:\n{src}");
        }
    }

    /// Non-finite constants have no valid MLIR spelling here; refuse them.
    #[test]
    fn non_finite_constants_are_refused() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.float(f64::INFINITY, 53)]);
        assert!(emit_stablehlo(expr, &[x], "f", &pool).is_empty());
    }
}

#[cfg(test)]
mod large_constant_tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use rug::ops::Pow;

    /// `to_i64().unwrap_or(0)` turned any coefficient past `i64` into the
    /// constant zero, so the emitted module computed a different function —
    /// valid MLIR, no diagnostic, wrong answer.
    #[test]
    fn a_coefficient_past_i64_is_not_emitted_as_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![
            pool.mul(vec![pool.integer(rug::Integer::from(2u32).pow(70)), x]),
            pool.integer(1_i32),
        ]);
        let mlir = emit_stablehlo(expr, &[x], "f", &pool);
        assert!(
            mlir.contains("1.1805916207174113e21"),
            "coefficient lost: {mlir}"
        );
        assert!(
            !mlir.contains("dense<0.0>"),
            "coefficient became the constant zero: {mlir}"
        );
    }

    #[test]
    fn a_rational_coefficient_rounds_to_nearest() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![pool.rational(2, 5), x]);
        let mlir = emit_stablehlo(expr, &[x], "f", &pool);
        assert!(mlir.contains("dense<0.4>"), "expected 0.4: {mlir}");
    }

    /// MLIR's float literal needs a decimal point. Rust's `{:?}` drops it for
    /// a single-digit mantissa, so `dense<1e30>` came out of the emitter and
    /// `mlir-opt` answered `error: expected '>'` — a module reported as
    /// emitted that does not parse.
    #[test]
    fn every_emitted_constant_carries_a_decimal_point() {
        assert_eq!(mlir_f64_literal(1e30), "1.0e30");
        assert_eq!(mlir_f64_literal(1e16), "1.0e16");
        assert_eq!(mlir_f64_literal(5e-324), "5.0e-324");
        assert_eq!(mlir_f64_literal(-1e30), "-1.0e30");
        // Forms that already had one are untouched.
        assert_eq!(mlir_f64_literal(1.0), "1.0");
        assert_eq!(mlir_f64_literal(0.4), "0.4");
        assert_eq!(
            mlir_f64_literal(1.1805916207174113e21),
            "1.1805916207174113e21"
        );
        assert_eq!(mlir_f64_literal(1e15), "1000000000000000.0");

        // …and end to end, over the constants that reach the emitter.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for coefficient in [
            pool.integer(rug::Integer::from(10).pow(30)),
            pool.integer(rug::Integer::from(10).pow(16)),
            pool.integer(1_i32),
            pool.rational(1, rug::Integer::from(10).pow(30)),
            pool.rational(2, 5),
        ] {
            let expr = pool.add(vec![pool.mul(vec![coefficient, x]), pool.integer(1_i32)]);
            let mlir = emit_stablehlo(expr, &[x], "f", &pool);
            for chunk in mlir.split("dense<").skip(1) {
                let literal = chunk.split('>').next().unwrap();
                assert!(
                    literal.contains('.'),
                    "MLIR rejects the literal `{literal}` in: {mlir}"
                );
            }
        }
    }

    /// Audit A11: `to_stablehlo(x^(2^63+1))` emitted the exponent as the `f64`
    /// `2^63`, which is even, so `power(-1, 2^63)` is `+1` where the answer is
    /// `-1`.  An odd exponent past `2^53` now carries its sign explicitly.
    #[test]
    fn an_odd_exponent_past_f64_keeps_its_sign() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let odd = rug::Integer::from(2u32).pow(63) + 1u32;
        let mlir = emit_stablehlo(pool.pow(x, pool.integer(odd.clone())), &[x], "f", &pool);
        assert!(mlir.contains("stablehlo.compare LT"), "{mlir}");
        assert!(mlir.contains("stablehlo.select"), "{mlir}");
        assert!(mlir.contains("stablehlo.negate"), "{mlir}");
        // An even exponent has nothing to lose and stays a plain `power`.
        let even = pool.pow(x, pool.integer(odd - 1u32));
        let mlir = emit_stablehlo(even, &[x], "f", &pool);
        assert!(!mlir.contains("stablehlo.select"), "{mlir}");
        assert!(mlir.contains("stablehlo.power"), "{mlir}");
        // So does an odd exponent `f64` holds exactly: `powf` keeps its parity.
        let small = pool.pow(x, pool.integer(9_007_199_254_740_991_i64));
        let mlir = emit_stablehlo(small, &[x], "f", &pool);
        assert!(!mlir.contains("stablehlo.select"), "{mlir}");
    }
}

/// W10: `to_stablehlo` used to return `''` — no error — for every function the
/// emitter did not lower (`erf`, `lambert_w`, `max` …).  These tests pin both
/// halves of the fix: what lowers is checked *numerically*, by running the
/// emitted module through a small reference interpreter of the StableHLO ops
/// it uses and comparing with `eval_interp`; what does not lower is refused
/// with a coded error naming the function.
#[cfg(test)]
mod lowering_tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::Domain;

    #[derive(Clone, Copy, Debug)]
    enum Val {
        F(f64),
        B(bool),
    }

    impl Val {
        fn f(self) -> f64 {
            match self {
                Val::F(v) => v,
                Val::B(_) => panic!("expected f64, got i1"),
            }
        }
        fn b(self) -> bool {
            match self {
                Val::B(v) => v,
                Val::F(_) => panic!("expected i1, got f64"),
            }
        }
    }

    fn lookup(env: &HashMap<String, Val>, k: &str, src: &str) -> Val {
        *env.get(k.trim())
            .unwrap_or_else(|| panic!("use of undefined {k:?} in\n{src}"))
    }

    /// Execute a module emitted by this file at one point. Panics on anything
    /// it does not recognise, so it also checks the module is well formed:
    /// one `module`/`func.func`, SSA values defined before use, a `return`.
    fn run(src: &str, args: &[f64]) -> f64 {
        let mut env: HashMap<String, Val> = HashMap::new();
        for (i, &a) in args.iter().enumerate() {
            env.insert(format!("%arg{i}"), Val::F(a));
        }
        let lines: Vec<&str> = src.lines().map(str::trim).collect();
        assert_eq!(lines[0], "module {", "{src}");
        assert!(lines[1].starts_with("func.func @"), "{src}");
        assert_eq!(&lines[lines.len() - 2..], ["}", "}"], "{src}");
        for line in &lines[2..lines.len() - 2] {
            if let Some(rest) = line.strip_prefix("return ") {
                let (v, ty) = rest.split_once(" : ").unwrap();
                assert_eq!(ty, "tensor<f64>");
                return lookup(&env, v, src).f();
            }
            let (lhs, rhs) = line
                .split_once(" = stablehlo.")
                .unwrap_or_else(|| panic!("not a stablehlo op: {line}"));
            assert!(!env.contains_key(lhs), "{lhs} redefined");
            let (op, rest) = rhs.split_once(' ').unwrap();
            let (operands, _ty) = rest.split_once(" : ").unwrap();
            let val = if op == "constant" {
                let lit = operands
                    .strip_prefix("dense<")
                    .and_then(|s| s.strip_suffix('>'))
                    .unwrap();
                match lit {
                    "true" => Val::B(true),
                    "false" => Val::B(false),
                    _ => {
                        assert!(lit.contains('.'), "bad float literal {lit}");
                        Val::F(lit.parse().unwrap())
                    }
                }
            } else if op == "compare" {
                let parts: Vec<&str> = operands.split(", ").collect();
                assert_eq!(parts.len(), 4);
                assert_eq!(parts[3], "FLOAT");
                let a = lookup(&env, parts[1], src).f();
                let b = lookup(&env, parts[2], src).f();
                Val::B(match parts[0] {
                    "LT" => a < b,
                    "LE" => a <= b,
                    "GT" => a > b,
                    "GE" => a >= b,
                    "EQ" => a == b,
                    "NE" => a != b,
                    d => panic!("compare {d}"),
                })
            } else {
                let o: Vec<Val> = operands.split(", ").map(|k| lookup(&env, k, src)).collect();
                let f = |i: usize| o[i].f();
                match (op, o.len()) {
                    ("select", 3) => {
                        if o[0].b() {
                            o[1]
                        } else {
                            o[2]
                        }
                    }
                    ("and", 2) => Val::B(o[0].b() && o[1].b()),
                    ("or", 2) => Val::B(o[0].b() || o[1].b()),
                    ("not", 1) => Val::B(!o[0].b()),
                    (op, 2) => Val::F(match op {
                        "add" => f(0) + f(1),
                        "subtract" => f(0) - f(1),
                        "multiply" => f(0) * f(1),
                        "divide" => f(0) / f(1),
                        "power" => f(0).powf(f(1)),
                        "atan2" => f(0).atan2(f(1)),
                        // StableHLO propagates NaN; `f64::max` does not.
                        "maximum" | "minimum" if f(0).is_nan() || f(1).is_nan() => f64::NAN,
                        "maximum" => f(0).max(f(1)),
                        "minimum" => f(0).min(f(1)),
                        _ => panic!("unknown binary op {op}"),
                    }),
                    (op, 1) => Val::F(match op {
                        "negate" => -f(0),
                        "abs" => f(0).abs(),
                        "sign" if f(0).is_nan() || f(0) == 0.0 => f(0),
                        "sign" => f(0).signum(),
                        "floor" => f(0).floor(),
                        "ceil" => f(0).ceil(),
                        "round_nearest_afz" => f(0).round(),
                        "sine" => f(0).sin(),
                        "cosine" => f(0).cos(),
                        "tanh" => f(0).tanh(),
                        "exponential" => f(0).exp(),
                        "exponential_minus_one" => f(0).exp_m1(),
                        "log" => f(0).ln(),
                        "log_plus_one" => f(0).ln_1p(),
                        "sqrt" => f(0).sqrt(),
                        _ => panic!("unknown unary op {op}"),
                    }),
                    _ => panic!("bad op {op}/{}", o.len()),
                }
            };
            env.insert(lhs.to_string(), val);
        }
        panic!("no return in\n{src}");
    }

    fn close(got: f64, want: f64) -> bool {
        if want.is_nan() {
            return got.is_nan();
        }
        if want.is_infinite() {
            return got == want;
        }
        (got - want).abs() <= 1e-13 * want.abs().max(1.0)
    }

    const POINTS: [f64; 13] = [
        -3.7, -1.0, -0.999, -0.5, -1e-9, -0.0, 0.0, 1e-9, 0.25, 0.5, 0.999, 1.0, 2.5,
    ];

    #[test]
    fn every_supported_unary_function_matches_eval_interp() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for name in [
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "ln",
            "sqrt",
            "tanh",
            "sinh",
            "cosh",
            "abs",
            "sign",
            "floor",
            "ceil",
            "round",
            "heaviside",
            "atan",
            "asin",
            "acos",
            "atanh",
        ] {
            let expr = pool.func(name, vec![x]);
            let src = try_emit_stablehlo(expr, &[x], "f", &pool)
                .unwrap_or_else(|e| panic!("{name} refused: {e}"));
            // `ln` is the natural log everywhere it is recognised, but it has
            // no registered evaluator of its own; check it against `log`.
            let reference = if name == "ln" {
                pool.func("log", vec![x])
            } else {
                expr
            };
            let mut checked = 0;
            for &p in &POINTS {
                let env: HashMap<ExprId, f64> = [(x, p)].into_iter().collect();
                let Some(want) = eval_interp(reference, &env, &pool) else {
                    continue;
                };
                let got = run(&src, &[p]);
                assert!(
                    close(got, want),
                    "{name}({p}): module {got}, eval {want}\n{src}"
                );
                checked += 1;
            }
            assert!(checked >= 5, "{name}: only {checked} points evaluated");
        }
    }

    /// `sinh` near 0: `(eˣ − e⁻ˣ)/2` returned ~1.00000008e-9 for 1e-9.
    #[test]
    fn sinh_keeps_its_digits_near_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let src = emit_stablehlo(pool.func("sinh", vec![x]), &[x], "f", &pool);
        for p in [1e-9, -3e-12, 1e-300] {
            let got = run(&src, &[p]);
            assert!(
                (got - p.sinh()).abs() <= 1e-15 * p.abs(),
                "sinh({p}) = {got}"
            );
        }
    }

    #[test]
    fn binary_and_variadic_functions_match_eval_interp() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let pts = [
            (-2.0, 1.5),
            (0.0, -0.0),
            (3.0, 3.0),
            (-1.0, -4.0),
            (0.0, 0.0),
        ];
        for name in ["atan2", "max", "min"] {
            let expr = pool.func(name, vec![x, y]);
            let src = try_emit_stablehlo(expr, &[x, y], "f", &pool).unwrap();
            for &(a, b) in &pts {
                let env: HashMap<ExprId, f64> = [(x, a), (y, b)].into_iter().collect();
                let want = eval_interp(expr, &env, &pool).unwrap();
                let got = run(&src, &[a, b]);
                assert!(close(got, want), "{name}({a},{b}): {got} vs {want}");
            }
        }
        // `max` of three folds left; `eval_interp` takes only two arguments.
        let z = pool.symbol("z", Domain::Real);
        let src = emit_stablehlo(pool.func("max", vec![x, y, z]), &[x, y, z], "f", &pool);
        assert_eq!(run(&src, &[1.0, 5.0, -2.0]), 5.0);
        assert_eq!(run(&src, &[1.0, 5.0, 7.0]), 7.0);
    }

    #[test]
    fn piecewise_takes_the_first_true_branch() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let one = pool.integer(1_i32);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let in_range = pool.pred_and(vec![
            pool.pred_ge(x, zero),
            pool.pred_le(x, pool.integer(2_i32)),
            pool.pred_not(pool.pred_eq(x, pool.rational(3, 2))),
        ]);
        let beyond = pool.pred_or(vec![
            pool.pred_gt(x, pool.integer(10_i32)),
            pool.pred_false(),
        ]);
        let expr = pool.piecewise(
            vec![
                (
                    pool.pred_lt(x, zero),
                    pool.mul(vec![pool.integer(-1_i32), x]),
                ),
                (pool.pred_lt(x, one), x2),
                (in_range, pool.integer(7_i32)),
                (beyond, pool.func("cos", vec![x])),
            ],
            pool.func("sin", vec![x]),
        );
        let src = try_emit_stablehlo(expr, &[x], "f", &pool).unwrap();
        assert!(src.contains("stablehlo.select"), "{src}");
        for p in [-2.0, -0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 11.0] {
            let env: HashMap<ExprId, f64> = [(x, p)].into_iter().collect();
            let want = eval_interp(expr, &env, &pool).unwrap();
            assert!(close(run(&src, &[p]), want), "x = {p}: want {want}\n{src}");
        }
        // A predicate used as a number is 1 or 0.
        let ind = pool.add(vec![pool.pred_ne(x, zero), x]);
        let src = emit_stablehlo(ind, &[x], "f", &pool);
        assert_eq!(run(&src, &[0.0]), 0.0);
        assert_eq!(run(&src, &[2.0]), 3.0);
    }

    #[test]
    fn unbound_pi_is_its_value() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let pi = pool.symbol("pi", Domain::Real);
        let src = emit_stablehlo(pool.mul(vec![pi, x]), &[x], "f", &pool);
        assert!(
            close(run(&src, &[2.0]), 2.0 * std::f64::consts::PI),
            "{src}"
        );
    }

    #[test]
    fn a_shared_subexpression_is_emitted_once() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut e = x;
        for _ in 0..40 {
            e = pool.add(vec![pool.func("sin", vec![e]), pool.func("cos", vec![e])]);
        }
        let src = emit_stablehlo(e, &[x], "f", &pool);
        assert!(src.lines().count() < 200, "{} lines", src.lines().count());
    }

    fn code_of(r: Result<String, StableHloError>) -> &'static str {
        match r {
            Ok(src) => panic!("expected a refusal, got a module:\n{src}"),
            Err(e) => {
                assert!(!e.to_string().is_empty());
                AlkahestError::code(&e)
            }
        }
    }

    #[test]
    fn unsupported_functions_are_refused_with_their_name() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for name in [
            "erf",
            "erfc",
            "gamma",
            "lambert_w",
            "digamma",
            "asinh",
            "acosh",
            "re",
            "im",
            "conjugate",
            "diracdelta",
            "f",
        ] {
            let expr = pool.func(name, vec![x]);
            let err = try_emit_stablehlo(expr, &[x], "f", &pool).unwrap_err();
            assert_eq!(AlkahestError::code(&err), "E-STABLEHLO-001", "{name}");
            assert!(err.to_string().contains(name), "{err}");
            assert_eq!(emit_stablehlo(expr, &[x], "f", &pool), "");
        }
        // Nested inside a supported expression: still refused, not partial.
        let nested = pool.add(vec![pool.func("sin", vec![x]), pool.func("erf", vec![x])]);
        assert_eq!(
            code_of(try_emit_stablehlo(nested, &[x], "f", &pool)),
            "E-STABLEHLO-001"
        );
        // A built-in at the wrong arity (the unchecked constructor allows it)
        // is refused rather than indexed out of bounds.
        assert_eq!(
            code_of(try_emit_stablehlo(
                pool.func("sin", vec![]),
                &[x],
                "f",
                &pool
            )),
            "E-STABLEHLO-001"
        );
        assert_eq!(
            code_of(try_emit_stablehlo(
                pool.func("max", vec![x]),
                &[x],
                "f",
                &pool
            )),
            "E-STABLEHLO-001"
        );
    }

    #[test]
    fn other_refusals_carry_their_codes() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        assert_eq!(
            code_of(try_emit_stablehlo(pool.big_o(x), &[x], "f", &pool)),
            "E-STABLEHLO-002"
        );
        let err = try_emit_stablehlo(pool.add(vec![x, y]), &[x], "f", &pool).unwrap_err();
        assert_eq!(AlkahestError::code(&err), "E-STABLEHLO-003");
        assert!(err.to_string().contains('y'), "{err}");
        let inf = pool.add(vec![x, pool.float(f64::INFINITY, 53)]);
        assert_eq!(
            code_of(try_emit_stablehlo(inf, &[x], "f", &pool)),
            "E-STABLEHLO-004"
        );
        for bad in ["", "my fn", "1f", "f-g", "f@"] {
            assert_eq!(
                code_of(try_emit_stablehlo(x, &[x], bad, &pool)),
                "E-STABLEHLO-005",
                "{bad:?}"
            );
        }
        for good in ["f", "_f", "alkahest_fn", "a.b$c1"] {
            assert!(try_emit_stablehlo(x, &[x], good, &pool).is_ok(), "{good}");
        }
        // An input returned as is still produces a well-formed module.
        assert_eq!(run(&emit_stablehlo(x, &[x], "f", &pool), &[4.5]), 4.5);
    }

    #[test]
    fn every_code_is_registered() {
        for code in [
            "E-STABLEHLO-001",
            "E-STABLEHLO-002",
            "E-STABLEHLO-003",
            "E-STABLEHLO-004",
            "E-STABLEHLO-005",
        ] {
            assert!(
                crate::errors::codes::REGISTRY
                    .iter()
                    .any(|s| s.code == code),
                "{code} missing from REGISTRY"
            );
        }
    }
}
