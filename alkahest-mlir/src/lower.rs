//! Progressive lowering passes.
//!
//! Three targets are supported as textual MLIR:
//!
//! * [`LoweringStage::ArithMath`] — `arith + math` dialects.  Polynomial
//!   subtrees become `math.fma` chains (the canonical Horner form).
//! * [`LoweringStage::StableHlo`] — delegates to
//!   [`alkahest_core::emit_stablehlo`] (feeds the JAX bridge).
//! * [`LoweringStage::Llvm`] — `llvm.*` dialect suitable for the host JIT /
//!   NVPTX pipelines.

use alkahest_core::kernel::{ExprData, ExprId, ExprPool};
use alkahest_core::poly::UniPoly;
use alkahest_core::stablehlo::emit_stablehlo;
use std::collections::HashMap;

use crate::LoweringStage;

/// Emit MLIR at the requested lowering stage.
pub fn lower(
    expr: ExprId,
    inputs: &[ExprId],
    fn_name: &str,
    stage: LoweringStage,
    pool: &ExprPool,
) -> String {
    match stage {
        LoweringStage::ArithMath => lower_arith_math(expr, inputs, fn_name, pool),
        LoweringStage::StableHlo => emit_stablehlo(expr, inputs, fn_name, pool),
        LoweringStage::Llvm => lower_llvm(expr, inputs, fn_name, pool),
    }
}

// ---------------------------------------------------------------------------
// Shared walker core
// ---------------------------------------------------------------------------

struct Backend {
    // op name conventions for the target dialect
    add: &'static str,
    mul: &'static str,
    div: &'static str,
    pow: &'static str,
    fma: &'static str,
    /// `llvm` dialect uses `llvm.call @llvm.sin.f64(...)`; `math` has intrinsics.
    math_intrinsic: fn(&str) -> Option<String>,
    const_op: fn(f64) -> String,
}

fn math_intrinsic_mlir(name: &str) -> Option<String> {
    Some(match name {
        "sin" => "math.sin".to_string(),
        "cos" => "math.cos".to_string(),
        "tan" => "math.tan".to_string(),
        "exp" => "math.exp".to_string(),
        "log" => "math.log".to_string(),
        "sqrt" => "math.sqrt".to_string(),
        "abs" => "math.absf".to_string(),
        _ => return None,
    })
}

fn llvm_intrinsic_mlir(name: &str) -> Option<String> {
    // Emitted as `llvm.intr.<name>` in the LLVM dialect.
    Some(match name {
        "sin" => "llvm.intr.sin".to_string(),
        "cos" => "llvm.intr.cos".to_string(),
        "exp" => "llvm.intr.exp".to_string(),
        "log" => "llvm.intr.log".to_string(),
        "sqrt" => "llvm.intr.sqrt".to_string(),
        "abs" => "llvm.intr.fabs".to_string(),
        _ => return None,
    })
}

fn const_arith(val: f64) -> String {
    format!("arith.constant {val:?} : f64")
}

fn const_llvm(val: f64) -> String {
    format!("llvm.mlir.constant({val:?} : f64) : f64")
}

// ---------------------------------------------------------------------------
// arith + math lowering
// ---------------------------------------------------------------------------

fn lower_arith_math(expr: ExprId, inputs: &[ExprId], fn_name: &str, pool: &ExprPool) -> String {
    let backend = Backend {
        add: "arith.addf",
        mul: "arith.mulf",
        div: "arith.divf",
        pow: "math.powf",
        fma: "math.fma",
        math_intrinsic: math_intrinsic_mlir,
        const_op: const_arith,
    };
    emit_func(expr, inputs, fn_name, &backend, pool)
}

// ---------------------------------------------------------------------------
// llvm lowering
// ---------------------------------------------------------------------------

fn lower_llvm(expr: ExprId, inputs: &[ExprId], fn_name: &str, pool: &ExprPool) -> String {
    let backend = Backend {
        add: "llvm.fadd",
        mul: "llvm.fmul",
        div: "llvm.fdiv",
        pow: "llvm.intr.pow",
        fma: "llvm.intr.fmuladd",
        math_intrinsic: llvm_intrinsic_mlir,
        const_op: const_llvm,
    };
    // LLVM dialect uses `llvm.func` + `llvm.return`
    emit_func_llvm(expr, inputs, fn_name, &backend, pool)
}

// ---------------------------------------------------------------------------
// Walker
// ---------------------------------------------------------------------------

struct Walker<'a> {
    backend: &'a Backend,
    arg_map: HashMap<ExprId, String>,
    input_set: Vec<ExprId>,
    cache: HashMap<ExprId, String>,
    body: Vec<String>,
    counter: usize,
}

impl<'a> Walker<'a> {
    fn new(backend: &'a Backend, inputs: &[ExprId]) -> Self {
        let mut arg_map = HashMap::new();
        for (i, &id) in inputs.iter().enumerate() {
            arg_map.insert(id, format!("%arg{i}"));
        }
        Walker {
            backend,
            arg_map,
            input_set: inputs.to_vec(),
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

    fn emit(&mut self, expr: ExprId, pool: &ExprPool) -> String {
        if let Some(s) = self.arg_map.get(&expr) {
            return s.clone();
        }
        if let Some(s) = self.cache.get(&expr) {
            return s.clone();
        }

        // Polynomial lift: if this node is a univariate polynomial of degree ≥ 2
        // in a known input, emit the Horner fma sequence directly.
        if let Some(ssa) = self.try_emit_horner(expr, pool) {
            self.cache.insert(expr, ssa.clone());
            return ssa;
        }

        let ssa = self.emit_raw(expr, pool);
        self.cache.insert(expr, ssa.clone());
        ssa
    }

    fn try_emit_horner(&mut self, expr: ExprId, pool: &ExprPool) -> Option<String> {
        let is_compound = pool.with(expr, |d| {
            matches!(
                d,
                ExprData::Add(_) | ExprData::Mul(_) | ExprData::Pow { .. }
            )
        });
        if !is_compound {
            return None;
        }
        for &var in &self.input_set.clone() {
            if let Ok(poly) = UniPoly::from_symbolic(expr, var, pool) {
                if poly.degree() >= 2 {
                    let Some(coeffs) = poly.coefficients_i64_checked() else {
                        continue;
                    };
                    let x = self.arg_map.get(&var).cloned()?;
                    return Some(self.emit_horner_fma(&coeffs, &x));
                }
            }
        }
        None
    }

    /// Build `a0 + x*(a1 + x*(a2 + …))` using fma:
    ///   `r = a_n`
    ///   `r = a_{k}` followed by `r = fma(x, r_prev, a_k)` for each lower k.
    fn emit_horner_fma(&mut self, coeffs: &[i64], x: &str) -> String {
        assert!(!coeffs.is_empty());
        let n = coeffs.len();
        let mut acc = self.emit_const(coeffs[n - 1] as f64);
        for k in (0..n - 1).rev() {
            let ck = self.emit_const(coeffs[k] as f64);
            // fma(x, acc, ck) = x * acc + ck
            let out = self.fresh();
            self.body.push(format!(
                "{out} = {fma} {x}, {acc}, {ck} : f64",
                fma = self.backend.fma,
            ));
            acc = out;
            let _ = ck; // already consumed as operand
        }
        acc
    }

    fn emit_const(&mut self, val: f64) -> String {
        let out = self.fresh();
        let rhs = (self.backend.const_op)(val);
        self.body.push(format!("{out} = {rhs}"));
        out
    }

    fn emit_binop(&mut self, op: &str, a: &str, b: &str) -> String {
        let out = self.fresh();
        self.body.push(format!("{out} = {op} {a}, {b} : f64"));
        out
    }

    fn emit_unary_call(&mut self, op: &str, a: &str) -> String {
        let out = self.fresh();
        self.body.push(format!("{out} = {op}({a}) : (f64) -> f64"));
        out
    }

    fn emit_raw(&mut self, expr: ExprId, pool: &ExprPool) -> String {
        enum Node {
            Const(f64),
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow { base: ExprId, exp: ExprId },
            Call { name: String, args: Vec<ExprId> },
            FreeSym,
        }
        let node = pool.with(expr, |d| match d {
            ExprData::Integer(n) => Node::Const(n.0.to_f64()),
            ExprData::Float(f) => Node::Const(f.inner.to_f64()),
            ExprData::Rational(r) => {
                let (n, d) = r.0.clone().into_numer_denom();
                Node::Const(n.to_f64() / d.to_f64())
            }
            ExprData::Add(args) => Node::Add(args.clone()),
            ExprData::Mul(args) => Node::Mul(args.clone()),
            ExprData::Pow { base, exp } => Node::Pow {
                base: *base,
                exp: *exp,
            },
            ExprData::Func { name, args } => Node::Call {
                name: name.clone(),
                args: args.clone(),
            },
            _ => Node::FreeSym,
        });

        match node {
            Node::Const(v) => self.emit_const(v),
            Node::Add(args) => {
                let vs: Vec<String> = args.iter().map(|&a| self.emit(a, pool)).collect();
                let mut acc = vs[0].clone();
                for rhs in &vs[1..] {
                    acc = self.emit_binop(self.backend.add, &acc, rhs);
                }
                acc
            }
            Node::Mul(args) => {
                let vs: Vec<String> = args.iter().map(|&a| self.emit(a, pool)).collect();
                let mut acc = vs[0].clone();
                for rhs in &vs[1..] {
                    acc = self.emit_binop(self.backend.mul, &acc, rhs);
                }
                acc
            }
            Node::Pow { base, exp } => {
                // Integer exponent → unrolled mul (stability) up to a small limit.
                let exp_int = pool.with(exp, |d| match d {
                    ExprData::Integer(n) => n.0.to_i64(),
                    _ => None,
                });
                let b = self.emit(base, pool);
                match exp_int {
                    Some(0) => self.emit_const(1.0),
                    Some(1) => b,
                    Some(n) if (2..=8).contains(&n) => {
                        let mut acc = b.clone();
                        for _ in 1..n {
                            acc = self.emit_binop(self.backend.mul, &acc, &b);
                        }
                        acc
                    }
                    Some(-1) => {
                        let one = self.emit_const(1.0);
                        self.emit_binop(self.backend.div, &one, &b)
                    }
                    _ => {
                        let e = self.emit(exp, pool);
                        self.emit_binop(self.backend.pow, &b, &e)
                    }
                }
            }
            Node::Call { name, args } => {
                let vs: Vec<String> = args.iter().map(|&a| self.emit(a, pool)).collect();
                let intr = (self.backend.math_intrinsic)(&name);
                match intr {
                    Some(op) if vs.len() == 1 => self.emit_unary_call(&op, &vs[0]),
                    _ => {
                        // Generic func.call
                        let out = self.fresh();
                        let operands = vs.join(", ");
                        let sig = vec!["f64"; vs.len()].join(", ");
                        self.body.push(format!(
                            "{out} = func.call @{name}({operands}) : ({sig}) -> f64"
                        ));
                        out
                    }
                }
            }
            Node::FreeSym => {
                // Represent as an unbound argument placeholder.
                let out = self.fresh();
                self.body
                    .push(format!("{out} = arith.constant 0.0 : f64  // unresolved"));
                out
            }
        }
    }
}

fn emit_func(
    expr: ExprId,
    inputs: &[ExprId],
    fn_name: &str,
    backend: &Backend,
    pool: &ExprPool,
) -> String {
    let mut w = Walker::new(backend, inputs);
    let result = w.emit(expr, pool);
    let args: Vec<String> = (0..inputs.len()).map(|i| format!("%arg{i}: f64")).collect();
    let body = w.body.join("\n    ");
    format!(
        "module {{\n  func.func @{fn}({args}) -> f64 {{\n    {body}\n    func.return {result} : f64\n  }}\n}}\n",
        fn = fn_name,
        args = args.join(", "),
        body = body,
        result = result,
    )
}

fn emit_func_llvm(
    expr: ExprId,
    inputs: &[ExprId],
    fn_name: &str,
    backend: &Backend,
    pool: &ExprPool,
) -> String {
    let mut w = Walker::new(backend, inputs);
    let result = w.emit(expr, pool);
    let args: Vec<String> = (0..inputs.len()).map(|i| format!("%arg{i}: f64")).collect();
    let body = w.body.join("\n    ");
    format!(
        "module {{\n  llvm.func @{fn}({args}) -> f64 {{\n    {body}\n    llvm.return {result} : f64\n  }}\n}}\n",
        fn = fn_name,
        args = args.join(", "),
        body = body,
        result = result,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use alkahest_core::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn arith_math_add() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let out = lower(
            pool.add(vec![x, y]),
            &[x, y],
            "f",
            LoweringStage::ArithMath,
            &pool,
        );
        assert!(out.contains("arith.addf %arg0, %arg1 : f64"), "{out}");
        assert!(out.contains("func.return"));
    }

    #[test]
    fn arith_math_horner_emits_fma() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let expr = pool.add(vec![x2, two_x, pool.integer(1_i32)]);
        let out = lower(expr, &[x], "f", LoweringStage::ArithMath, &pool);
        assert!(
            out.contains("math.fma"),
            "Horner should lower to math.fma:\n{out}"
        );
    }

    #[test]
    fn llvm_lowering_uses_llvm_dialect() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let out = lower(
            pool.mul(vec![x, y]),
            &[x, y],
            "f",
            LoweringStage::Llvm,
            &pool,
        );
        assert!(out.contains("llvm.fmul"), "{out}");
        assert!(out.contains("llvm.func @f"), "{out}");
        assert!(out.contains("llvm.return"), "{out}");
    }

    #[test]
    fn llvm_lowering_horner_uses_fmuladd() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let expr = pool.add(vec![x2, pool.integer(1_i32)]);
        let out = lower(expr, &[x], "f", LoweringStage::Llvm, &pool);
        assert!(
            out.contains("llvm.intr.fmuladd"),
            "Horner in LLVM dialect should use llvm.intr.fmuladd:\n{out}"
        );
    }

    #[test]
    fn stablehlo_delegates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let out = lower(
            pool.func("sin", vec![x]),
            &[x],
            "f",
            LoweringStage::StableHlo,
            &pool,
        );
        assert!(out.contains("stablehlo.sine"), "{out}");
    }

    #[test]
    fn math_sin_lowers_to_math_intrinsic() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let out = lower(
            pool.func("sin", vec![x]),
            &[x],
            "f",
            LoweringStage::ArithMath,
            &pool,
        );
        assert!(out.contains("math.sin"), "{out}");
    }
}
