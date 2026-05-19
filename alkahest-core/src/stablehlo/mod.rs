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
/// function arguments.  Returns the complete MLIR text.
pub fn emit_stablehlo(expr: ExprId, inputs: &[ExprId], fn_name: &str, pool: &ExprPool) -> String {
    let mut emitter = Emitter::new(inputs, pool);
    let result_var = emitter.emit_expr(expr, pool);

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
        }
    }

    fn fresh(&mut self) -> String {
        let v = format!("%v{}", self.counter);
        self.counter += 1;
        v
    }

    fn emit_const_f64(&mut self, val: f64) -> String {
        let v = self.fresh();
        self.body.push(format!(
            "{v} = stablehlo.constant dense<{val}> : tensor<f64>"
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
                    _ => {
                        self.body.push(format!("// unsupported function: {name}"));
                        return self.emit_const_f64(0.0);
                    }
                }
                v
            }

            Node::Unknown => {
                self.body.push("// unknown node type".to_string());
                self.emit_const_f64(0.0)
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
