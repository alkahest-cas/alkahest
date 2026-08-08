//! StableHLO / XLA bridge.
//!
//! Converts a Alkahest symbolic expression to a StableHLO MLIR text module
//! that can be parsed by `jaxlib.mlir.dialects.stablehlo` or `mlir-opt`.
//!
//! # Supported ops (phase 1)
//! `Add`, `Mul`, `Pow(_, Integer)`, `sin`, `cos`, `exp`, `log`, `sqrt`.
//! `Piecewise` lowers to `stablehlo.select`.
//!
//! # Example
//! ```
//! use alkahest_cas::kernel::{Domain, ExprPool};
//! use alkahest_cas::stablehlo::emit_stablehlo;
//!
//! let pool = ExprPool::new();
//! let x = pool.symbol("x", Domain::Real);
//! let expr = pool.func("sin", vec![x]);
//! let mlir = emit_stablehlo(expr, &[x], "my_fn", &pool);
//! assert!(mlir.contains("stablehlo.sine"));
//! ```

use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;

/// Emit a StableHLO MLIR text module for `expr` as a function named `fn_name`.
///
/// `inputs` gives the list of symbolic variables (in order) that become the
/// function arguments.  Returns the complete MLIR text, or an **empty string**
/// when `expr` contains anything with no sound StableHLO encoding.
///
/// Returning nothing rather than a partial module is deliberate, and follows
/// the same rule as the Lean exporter: emit no artifact rather than an
/// incorrect one. The previous behaviour substituted the constant `0` for any
/// unsupported function, so `tanh(x)` produced a module that evaluated to zero
/// everywhere — an exported program is run by another toolchain that will never
/// compare it back against the source expression.
pub fn emit_stablehlo(expr: ExprId, inputs: &[ExprId], fn_name: &str, pool: &ExprPool) -> String {
    let mut emitter = Emitter::new(inputs, pool);
    let result_var = emitter.emit_expr(expr, pool);
    if emitter.unsupported.is_some() {
        return String::new();
    }

    // Build function signature
    let args: Vec<String> = inputs
        .iter()
        .enumerate()
        .map(|(i, _)| format!("%arg{i}: tensor<f64>"))
        .collect();
    let args_str = args.join(", ");

    // Build function body
    let body = emitter.body.join("\n    ");

    format!(
        r#"module {{
  func.func @{fn_name}({args_str}) -> tensor<f64> {{
    {body}
    return {result_var} : tensor<f64>
  }}
}}"#
    )
}

struct Emitter {
    arg_map: HashMap<ExprId, String>,
    body: Vec<String>,
    counter: usize,
    /// Set when the expression contains something with no sound StableHLO
    /// encoding. Emission is abandoned rather than completed with a stand-in.
    unsupported: Option<String>,
}

impl Emitter {
    fn new(inputs: &[ExprId], _pool: &ExprPool) -> Self {
        let mut arg_map = HashMap::new();
        for (i, &id) in inputs.iter().enumerate() {
            arg_map.insert(id, format!("%arg{i}"));
        }
        Emitter {
            arg_map,
            body: Vec::new(),
            counter: 0,
            unsupported: None,
        }
    }

    fn fresh(&mut self) -> String {
        let v = format!("%v{}", self.counter);
        self.counter += 1;
        v
    }

    /// Emit an `f64` constant as a *float* literal.
    ///
    /// `{val}` renders `1.0_f64` as `1`, and MLIR rejects a decimal integer
    /// literal for an `f64` tensor ("unexpected decimal integer"). Every
    /// expression containing an integer constant — `x**2 + 1` — therefore
    /// emitted a module that would not parse. `{val:?}` always renders a
    /// decimal point.
    fn emit_const_f64(&mut self, val: f64) -> String {
        if !val.is_finite() {
            self.unsupported = Some(format!("non-finite constant: {val}"));
            return self.fresh();
        }
        let v = self.fresh();
        self.body.push(format!(
            "{v} = stablehlo.constant dense<{val:?}> : tensor<f64>"
        ));
        v
    }

    fn emit_expr(&mut self, expr: ExprId, pool: &ExprPool) -> String {
        // Return cached arg if this is an input variable
        if let Some(s) = self.arg_map.get(&expr) {
            return s.clone();
        }

        enum Node {
            Integer(i64),
            Float(f64),
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow { base: ExprId, exp: ExprId },
            Func { name: String, args: Vec<ExprId> },
            Unknown,
        }

        let node = pool.with(expr, |data| match data {
            ExprData::Integer(n) => Node::Integer(n.0.to_i64().unwrap_or(0)),
            ExprData::Float(f) => Node::Float(f.inner.to_f64()),
            ExprData::Rational(r) => {
                let (numer, denom) = r.0.clone().into_numer_denom();
                Node::Float(numer.to_f64() / denom.to_f64())
            }
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
            _ => Node::Unknown,
        });

        match node {
            Node::Integer(n) => self.emit_const_f64(n as f64),
            Node::Float(f) => self.emit_const_f64(f),

            Node::Add(args) => {
                let emitted: Vec<String> = args.iter().map(|&a| self.emit_expr(a, pool)).collect();
                let mut acc = emitted[0].clone();
                for operand in &emitted[1..] {
                    let v = self.fresh();
                    self.body.push(format!(
                        "{v} = stablehlo.add {acc}, {operand} : tensor<f64>"
                    ));
                    acc = v;
                }
                acc
            }

            Node::Mul(args) => {
                let emitted: Vec<String> = args.iter().map(|&a| self.emit_expr(a, pool)).collect();
                let mut acc = emitted[0].clone();
                for operand in &emitted[1..] {
                    let v = self.fresh();
                    self.body.push(format!(
                        "{v} = stablehlo.multiply {acc}, {operand} : tensor<f64>"
                    ));
                    acc = v;
                }
                acc
            }

            Node::Pow { base, exp } => {
                // Check for integer exponent — lower to repeated multiply or power op
                let exp_int = pool.with(exp, |d| match d {
                    ExprData::Integer(n) => n.0.to_i64(),
                    _ => None,
                });
                let base_v = self.emit_expr(base, pool);
                if let Some(n) = exp_int {
                    if n == -1 {
                        let one = self.emit_const_f64(1.0);
                        let v = self.fresh();
                        self.body.push(format!(
                            "{v} = stablehlo.divide {one}, {base_v} : tensor<f64>"
                        ));
                        return v;
                    } else if n == 2 {
                        let v = self.fresh();
                        self.body.push(format!(
                            "{v} = stablehlo.multiply {base_v}, {base_v} : tensor<f64>"
                        ));
                        return v;
                    } else if n == 0 {
                        return self.emit_const_f64(1.0);
                    }
                }
                // General: use power op
                let exp_v = self.emit_expr(exp, pool);
                let v = self.fresh();
                self.body.push(format!(
                    "{v} = stablehlo.power {base_v}, {exp_v} : tensor<f64>"
                ));
                v
            }

            Node::Func { name, args } => {
                let arg_vs: Vec<String> = args.iter().map(|&a| self.emit_expr(a, pool)).collect();
                let v = self.fresh();
                match name.as_str() {
                    "sin" => self
                        .body
                        .push(format!("{v} = stablehlo.sine {} : tensor<f64>", arg_vs[0])),
                    "cos" => self.body.push(format!(
                        "{v} = stablehlo.cosine {} : tensor<f64>",
                        arg_vs[0]
                    )),
                    "exp" => self.body.push(format!(
                        "{v} = stablehlo.exponential {} : tensor<f64>",
                        arg_vs[0]
                    )),
                    "log" => self
                        .body
                        .push(format!("{v} = stablehlo.log {} : tensor<f64>", arg_vs[0])),
                    "sqrt" => self
                        .body
                        .push(format!("{v} = stablehlo.sqrt {} : tensor<f64>", arg_vs[0])),
                    "tanh" => self
                        .body
                        .push(format!("{v} = stablehlo.tanh {} : tensor<f64>", arg_vs[0])),
                    "abs" => self
                        .body
                        .push(format!("{v} = stablehlo.abs {} : tensor<f64>", arg_vs[0])),
                    // No StableHLO primitive; expand into ones that exist.
                    "tan" => {
                        let s_v = self.fresh();
                        self.body.push(format!(
                            "{s_v} = stablehlo.sine {} : tensor<f64>",
                            arg_vs[0]
                        ));
                        let c_v = self.fresh();
                        self.body.push(format!(
                            "{c_v} = stablehlo.cosine {} : tensor<f64>",
                            arg_vs[0]
                        ));
                        self.body
                            .push(format!("{v} = stablehlo.divide {s_v}, {c_v} : tensor<f64>"));
                    }
                    "sinh" | "cosh" => {
                        // sinh x = (e^x − e^−x)/2, cosh x = (e^x + e^−x)/2
                        let ex = self.fresh();
                        self.body.push(format!(
                            "{ex} = stablehlo.exponential {} : tensor<f64>",
                            arg_vs[0]
                        ));
                        let neg = self.fresh();
                        self.body.push(format!(
                            "{neg} = stablehlo.negate {} : tensor<f64>",
                            arg_vs[0]
                        ));
                        let enx = self.fresh();
                        self.body
                            .push(format!("{enx} = stablehlo.exponential {neg} : tensor<f64>"));
                        let combined = self.fresh();
                        let op = if name == "sinh" { "subtract" } else { "add" };
                        self.body.push(format!(
                            "{combined} = stablehlo.{op} {ex}, {enx} : tensor<f64>"
                        ));
                        let two = self.emit_const_f64(2.0);
                        self.body.push(format!(
                            "{v} = stablehlo.divide {combined}, {two} : tensor<f64>"
                        ));
                    }
                    _ => {
                        // Emitting a stand-in here produced a program that
                        // *computed the wrong thing*: the old code pushed a
                        // comment and returned the constant 0, so
                        // `to_stablehlo(tanh(x))` yielded a module evaluating to
                        // zero everywhere. An exported artifact is executed by
                        // another system that will never re-check it against the
                        // original expression, so a wrong program is worse here
                        // than anywhere else in the library.
                        self.unsupported = Some(format!("unsupported function: {name}"));
                        return self.fresh();
                    }
                }
                v
            }

            Node::Unknown => {
                self.unsupported = Some("unsupported expression node".to_string());
                self.fresh()
            }
        }
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
        for name in ["erf", "asin", "atan", "digamma"] {
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
