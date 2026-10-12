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

use alkahest_core::kernel::expr::PredicateKind;
use alkahest_core::kernel::{ExprData, ExprId, ExprPool};
use alkahest_core::poly::UniPoly;
use alkahest_core::stablehlo::try_emit_stablehlo;
use std::collections::HashMap;

use crate::{LoweringError, LoweringStage};

/// Emit MLIR at the requested lowering stage.
///
/// # Errors
///
/// Never returns a module that computes something other than `expr`: a
/// symbol that is not in `inputs` ([`LoweringError::UnboundSymbol`], except
/// the named constant `pi`), a function the stage has no exact lowering for
/// ([`LoweringError::UnsupportedFunction`]), a node with no numeric value
/// (`O(x)`, quantifiers, a bare predicate — [`LoweringError::UnsupportedNode`])
/// and a non-finite constant are all refused.  The StableHLO stage forwards
/// the `E-STABLEHLO-NNN` error of [`try_emit_stablehlo`] unchanged.
pub fn lower(
    expr: ExprId,
    inputs: &[ExprId],
    fn_name: &str,
    stage: LoweringStage,
    pool: &ExprPool,
) -> Result<String, LoweringError> {
    match stage {
        LoweringStage::ArithMath => lower_arith_math(expr, inputs, fn_name, pool),
        LoweringStage::StableHlo => Ok(try_emit_stablehlo(expr, inputs, fn_name, pool)?),
        LoweringStage::Llvm => lower_llvm(expr, inputs, fn_name, pool),
    }
}

// ---------------------------------------------------------------------------
// Shared walker core
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dialect {
    Arith,
    Llvm,
}

struct Backend {
    dialect: Dialect,
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

fn lower_arith_math(
    expr: ExprId,
    inputs: &[ExprId],
    fn_name: &str,
    pool: &ExprPool,
) -> Result<String, LoweringError> {
    let backend = Backend {
        dialect: Dialect::Arith,
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

fn lower_llvm(
    expr: ExprId,
    inputs: &[ExprId],
    fn_name: &str,
    pool: &ExprPool,
) -> Result<String, LoweringError> {
    let backend = Backend {
        dialect: Dialect::Llvm,
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

    fn emit(&mut self, expr: ExprId, pool: &ExprPool) -> Result<String, LoweringError> {
        if let Some(s) = self.arg_map.get(&expr) {
            return Ok(s.clone());
        }
        if let Some(s) = self.cache.get(&expr) {
            return Ok(s.clone());
        }

        // Polynomial lift: if this node is a univariate polynomial of degree ≥ 2
        // in a known input, emit the Horner fma sequence directly.
        if let Some(ssa) = self.try_emit_horner(expr, pool) {
            self.cache.insert(expr, ssa.clone());
            return Ok(ssa);
        }

        let ssa = self.emit_raw(expr, pool)?;
        self.cache.insert(expr, ssa.clone());
        Ok(ssa)
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
        let mut acc = self.emit_const_unchecked(coeffs[n - 1] as f64);
        for k in (0..n - 1).rev() {
            let ck = self.emit_const_unchecked(coeffs[k] as f64);
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

    fn emit_const(&mut self, val: f64) -> Result<String, LoweringError> {
        if !val.is_finite() {
            return Err(LoweringError::NonFiniteConstant(val));
        }
        Ok(self.emit_const_unchecked(val))
    }

    fn emit_const_unchecked(&mut self, val: f64) -> String {
        let out = self.fresh();
        let rhs = (self.backend.const_op)(val);
        self.body.push(format!("{out} = {rhs}"));
        out
    }

    // -- predicates (i1) -----------------------------------------------------

    fn emit_bool_const(&mut self, val: bool) -> String {
        let out = self.fresh();
        let rhs = match self.backend.dialect {
            Dialect::Arith => format!("arith.constant {val}"),
            Dialect::Llvm => format!("llvm.mlir.constant({val}) : i1"),
        };
        self.body.push(format!("{out} = {rhs}"));
        out
    }

    fn emit_cmp(&mut self, pred: &str, a: &str, b: &str) -> String {
        let out = self.fresh();
        let rhs = match self.backend.dialect {
            Dialect::Arith => format!("arith.cmpf {pred}, {a}, {b} : f64"),
            Dialect::Llvm => format!("llvm.fcmp \"{pred}\" {a}, {b} : f64"),
        };
        self.body.push(format!("{out} = {rhs}"));
        out
    }

    fn emit_bool_op(&mut self, op: &str, a: &str, b: &str) -> String {
        let out = self.fresh();
        let name = match (self.backend.dialect, op) {
            (Dialect::Arith, "and") => "arith.andi",
            (Dialect::Arith, "or") => "arith.ori",
            (Dialect::Arith, _) => "arith.xori",
            (Dialect::Llvm, "and") => "llvm.and",
            (Dialect::Llvm, "or") => "llvm.or",
            (Dialect::Llvm, _) => "llvm.xor",
        };
        self.body.push(format!("{out} = {name} {a}, {b} : i1"));
        out
    }

    fn emit_select(&mut self, cond: &str, t: &str, f: &str) -> String {
        let out = self.fresh();
        let rhs = match self.backend.dialect {
            Dialect::Arith => format!("arith.select {cond}, {t}, {f} : f64"),
            Dialect::Llvm => format!("llvm.select {cond}, {t}, {f} : i1, f64"),
        };
        self.body.push(format!("{out} = {rhs}"));
        out
    }

    /// Lower a predicate node to an `i1` SSA value.
    fn emit_pred(&mut self, expr: ExprId, pool: &ExprPool) -> Result<String, LoweringError> {
        let (kind, args) = pool
            .with(expr, |d| match d {
                ExprData::Predicate { kind, args } => Some((kind.clone(), args.clone())),
                _ => None,
            })
            .ok_or_else(|| {
                LoweringError::UnsupportedNode(
                    "a Piecewise condition that is not a predicate".to_string(),
                )
            })?;
        let arity_ok = match kind {
            PredicateKind::Lt
            | PredicateKind::Le
            | PredicateKind::Gt
            | PredicateKind::Ge
            | PredicateKind::Eq
            | PredicateKind::Ne => args.len() == 2,
            PredicateKind::Not => args.len() == 1,
            PredicateKind::And | PredicateKind::Or => true,
            PredicateKind::True | PredicateKind::False => args.is_empty(),
        };
        if !arity_ok {
            return Err(LoweringError::UnsupportedNode(format!(
                "a malformed predicate ({} argument(s))",
                args.len()
            )));
        }
        match kind {
            PredicateKind::True => Ok(self.emit_bool_const(true)),
            PredicateKind::False => Ok(self.emit_bool_const(false)),
            PredicateKind::Not => {
                let a = self.emit_pred(args[0], pool)?;
                let t = self.emit_bool_const(true);
                Ok(self.emit_bool_op("xor", &a, &t))
            }
            PredicateKind::And | PredicateKind::Or => {
                let (op, unit) = if kind == PredicateKind::And {
                    ("and", true)
                } else {
                    ("or", false)
                };
                if args.is_empty() {
                    return Ok(self.emit_bool_const(unit));
                }
                let mut acc = self.emit_pred(args[0], pool)?;
                for &a in &args[1..] {
                    let rhs = self.emit_pred(a, pool)?;
                    acc = self.emit_bool_op(op, &acc, &rhs);
                }
                Ok(acc)
            }
            cmp => {
                // Ordered comparisons are false when an operand is NaN;
                // `!=` is true then (`une`).
                let p = match cmp {
                    PredicateKind::Lt => "olt",
                    PredicateKind::Le => "ole",
                    PredicateKind::Gt => "ogt",
                    PredicateKind::Ge => "oge",
                    PredicateKind::Eq => "oeq",
                    _ => "une",
                };
                let a = self.emit(args[0], pool)?;
                let b = self.emit(args[1], pool)?;
                Ok(self.emit_cmp(p, &a, &b))
            }
        }
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

    fn emit_raw(&mut self, expr: ExprId, pool: &ExprPool) -> Result<String, LoweringError> {
        enum Node {
            Const(f64),
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow {
                base: ExprId,
                exp: ExprId,
            },
            Call {
                name: String,
                args: Vec<ExprId>,
            },
            Piecewise {
                branches: Vec<(ExprId, ExprId)>,
                default: ExprId,
            },
            Sym(String),
            Unsupported(&'static str),
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
            ExprData::Piecewise { branches, default } => Node::Piecewise {
                branches: branches.clone(),
                default: *default,
            },
            ExprData::Symbol { name, .. } => Node::Sym(name.clone()),
            ExprData::Predicate { .. } => Node::Unsupported("a bare predicate"),
            ExprData::Forall { .. } | ExprData::Exists { .. } => Node::Unsupported("a quantifier"),
            ExprData::BigO(_) => Node::Unsupported("a big-O remainder"),
            _ => Node::Unsupported("this expression node"),
        });

        match node {
            Node::Const(v) => self.emit_const(v),
            Node::Sym(name) => {
                // Same convention as the evaluators: an unbound `pi` is pi.
                if name == "pi" {
                    self.emit_const(std::f64::consts::PI)
                } else {
                    Err(LoweringError::UnboundSymbol(name))
                }
            }
            Node::Unsupported(what) => Err(LoweringError::UnsupportedNode(what.to_string())),
            Node::Piecewise { branches, default } => {
                // Every branch is a pure expression, so all values are computed
                // and chained with selects, last branch innermost: the first
                // matching condition wins, `default` otherwise.
                let mut conds = Vec::with_capacity(branches.len());
                let mut vals = Vec::with_capacity(branches.len());
                for &(c, v) in &branches {
                    conds.push(self.emit_pred(c, pool)?);
                    vals.push(self.emit(v, pool)?);
                }
                let mut acc = self.emit(default, pool)?;
                for (c, v) in conds.iter().zip(vals.iter()).rev() {
                    acc = self.emit_select(c, v, &acc);
                }
                Ok(acc)
            }
            Node::Add(args) => {
                let mut vs = Vec::with_capacity(args.len());
                for &a in &args {
                    vs.push(self.emit(a, pool)?);
                }
                if vs.is_empty() {
                    return self.emit_const(0.0);
                }
                let mut acc = vs[0].clone();
                for rhs in &vs[1..] {
                    acc = self.emit_binop(self.backend.add, &acc, rhs);
                }
                Ok(acc)
            }
            Node::Mul(args) => {
                let mut vs = Vec::with_capacity(args.len());
                for &a in &args {
                    vs.push(self.emit(a, pool)?);
                }
                if vs.is_empty() {
                    return self.emit_const(1.0);
                }
                let mut acc = vs[0].clone();
                for rhs in &vs[1..] {
                    acc = self.emit_binop(self.backend.mul, &acc, rhs);
                }
                Ok(acc)
            }
            Node::Pow { base, exp } => {
                // Integer exponent -> unrolled mul (stability) up to a small limit.
                let exp_int = pool.with(exp, |d| match d {
                    ExprData::Integer(n) => n.0.to_i64(),
                    _ => None,
                });
                let b = self.emit(base, pool)?;
                match exp_int {
                    Some(0) => self.emit_const(1.0),
                    Some(1) => Ok(b),
                    Some(n) if (2..=8).contains(&n) => {
                        let mut acc = b.clone();
                        for _ in 1..n {
                            acc = self.emit_binop(self.backend.mul, &acc, &b);
                        }
                        Ok(acc)
                    }
                    Some(-1) => {
                        let one = self.emit_const(1.0)?;
                        Ok(self.emit_binop(self.backend.div, &one, &b))
                    }
                    _ => {
                        let e = self.emit(exp, pool)?;
                        Ok(self.emit_binop(self.backend.pow, &b, &e))
                    }
                }
            }
            Node::Call { name, args } => {
                let mut vs = Vec::with_capacity(args.len());
                for &a in &args {
                    vs.push(self.emit(a, pool)?);
                }
                // There is no linked callee for anything else, so a generic
                // `func.call @name` would be a module that does not verify
                // (or, worse, binds to an unrelated symbol).
                match (self.backend.math_intrinsic)(&name) {
                    Some(op) if vs.len() == 1 => Ok(self.emit_unary_call(&op, &vs[0])),
                    _ => Err(LoweringError::UnsupportedFunction {
                        name,
                        arity: vs.len(),
                    }),
                }
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
) -> Result<String, LoweringError> {
    let mut w = Walker::new(backend, inputs);
    let result = w.emit(expr, pool)?;
    let args: Vec<String> = (0..inputs.len()).map(|i| format!("%arg{i}: f64")).collect();
    let body = w.body.join("\n    ");
    Ok(format!(
        "module {{\n  func.func @{fn}({args}) -> f64 {{\n    {body}\n    func.return {result} : f64\n  }}\n}}\n",
        fn = fn_name,
        args = args.join(", "),
        body = body,
        result = result,
    ))
}

fn emit_func_llvm(
    expr: ExprId,
    inputs: &[ExprId],
    fn_name: &str,
    backend: &Backend,
    pool: &ExprPool,
) -> Result<String, LoweringError> {
    let mut w = Walker::new(backend, inputs);
    let result = w.emit(expr, pool)?;
    let args: Vec<String> = (0..inputs.len()).map(|i| format!("%arg{i}: f64")).collect();
    let body = w.body.join("\n    ");
    Ok(format!(
        "module {{\n  llvm.func @{fn}({args}) -> f64 {{\n    {body}\n    llvm.return {result} : f64\n  }}\n}}\n",
        fn = fn_name,
        args = args.join(", "),
        body = body,
        result = result,
    ))
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
        )
        .unwrap();
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
        let out = lower(expr, &[x], "f", LoweringStage::ArithMath, &pool).unwrap();
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
        )
        .unwrap();
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
        let out = lower(expr, &[x], "f", LoweringStage::Llvm, &pool).unwrap();
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
        )
        .unwrap();
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
        )
        .unwrap();
        assert!(out.contains("math.sin"), "{out}");
    }

    // -- refusals instead of placeholders -----------------------------------

    use alkahest_core::AlkahestError;

    const STAGES: [LoweringStage; 3] = [
        LoweringStage::ArithMath,
        LoweringStage::StableHlo,
        LoweringStage::Llvm,
    ];

    #[test]
    fn unbound_symbol_is_an_error_at_every_stage() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = pool.add(vec![x, y]);
        for stage in STAGES {
            let err = lower(e, &[x], "f", stage, &pool).unwrap_err();
            let want = if stage == LoweringStage::StableHlo {
                "E-STABLEHLO-003"
            } else {
                "E-MLIR-001"
            };
            assert_eq!(err.code(), want, "{stage:?}: {err}");
            assert!(err.to_string().contains('y'), "{err}");
        }
    }

    #[test]
    fn unknown_function_is_an_error_at_every_stage() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        for name in ["erf", "my_user_fn"] {
            let e = pool.func(name, vec![x]);
            for stage in STAGES {
                let err = lower(e, &[x], "f", stage, &pool).unwrap_err();
                let want = if stage == LoweringStage::StableHlo {
                    "E-STABLEHLO-001"
                } else {
                    "E-MLIR-002"
                };
                assert_eq!(err.code(), want, "{name} {stage:?}: {err}");
            }
        }
        // A known intrinsic at the wrong arity is refused too.
        let e = pool.func("sin", vec![x, x]);
        assert_eq!(
            lower(e, &[x], "f", LoweringStage::ArithMath, &pool)
                .unwrap_err()
                .code(),
            "E-MLIR-002"
        );
    }

    #[test]
    fn unbound_pi_is_its_value() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let pi = pool.symbol("pi", Domain::Real);
        let out = lower(
            pool.mul(vec![pi, x]),
            &[x],
            "f",
            LoweringStage::ArithMath,
            &pool,
        )
        .unwrap();
        assert!(out.contains("3.141592653589793"), "{out}");
    }

    #[test]
    fn non_finite_constant_is_refused() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let inf = pool.float(f64::INFINITY, 53);
        let e = pool.add(vec![x, inf]);
        assert_eq!(
            lower(e, &[x], "f", LoweringStage::Llvm, &pool)
                .unwrap_err()
                .code(),
            "E-MLIR-004"
        );
    }

    #[test]
    fn big_o_and_bare_predicate_are_refused() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let lt = pool.pred_lt(x, pool.integer(0_i32));
        for e in [lt, pool.big_o(x)] {
            assert_eq!(
                lower(e, &[x], "f", LoweringStage::ArithMath, &pool)
                    .unwrap_err()
                    .code(),
                "E-MLIR-003"
            );
        }
    }

    #[test]
    fn stablehlo_stage_forwards_its_error_instead_of_empty_text() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let err = lower(
            pool.func("gamma", vec![x]),
            &[x],
            "f",
            LoweringStage::StableHlo,
            &pool,
        )
        .unwrap_err();
        assert!(matches!(err, LoweringError::StableHlo(_)), "{err:?}");
        assert_eq!(err.code(), "E-STABLEHLO-001");
        let err = lower(x, &[x], "bad name", LoweringStage::StableHlo, &pool).unwrap_err();
        assert_eq!(err.code(), "E-STABLEHLO-005");
    }

    // -- Piecewise ------------------------------------------------------------

    fn branchy(pool: &ExprPool, x: ExprId) -> ExprId {
        // x < 0 -> -x ; x == 3 -> 7 ; otherwise x
        let neg = pool.mul(vec![pool.integer(-1_i32), x]);
        pool.piecewise(
            vec![
                (pool.pred_lt(x, pool.integer(0_i32)), neg),
                (pool.pred_eq(x, pool.integer(3_i32)), pool.integer(7_i32)),
            ],
            x,
        )
    }

    #[test]
    fn piecewise_lowers_to_cmp_and_select_in_arith() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let out = lower(
            branchy(&pool, x),
            &[x],
            "f",
            LoweringStage::ArithMath,
            &pool,
        )
        .unwrap();
        assert!(out.contains("arith.cmpf olt, %arg0"), "{out}");
        assert!(out.contains("arith.cmpf oeq, %arg0"), "{out}");
        assert_eq!(out.matches("arith.select").count(), 2, "{out}");
        assert!(!out.contains("unresolved"), "{out}");
        // First matching branch wins: the last branch's select is innermost
        // (its false operand is the default %arg0) and feeds the first one.
        let sels: Vec<&str> = out.lines().filter(|l| l.contains("arith.select")).collect();
        assert!(sels[0].ends_with("%arg0 : f64"), "{out}");
        let inner = sels[0].split_whitespace().next().unwrap();
        assert!(sels[1].contains(inner), "{out}");
    }

    #[test]
    fn piecewise_lowers_to_fcmp_and_select_in_llvm() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let out = lower(branchy(&pool, x), &[x], "f", LoweringStage::Llvm, &pool).unwrap();
        assert!(out.contains("llvm.fcmp \"olt\" %arg0"), "{out}");
        assert_eq!(out.matches("llvm.select").count(), 2, "{out}");
        assert!(out.contains(": i1, f64"), "{out}");
    }

    #[test]
    fn compound_predicates_lower() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let one = pool.integer(1_i32);
        let cond = pool.pred_and(vec![
            pool.pred_ge(x, zero),
            pool.pred_not(pool.pred_ge(x, one)),
        ]);
        let e = pool.piecewise(vec![(cond, pool.integer(5_i32))], pool.integer(6_i32));
        let out = lower(e, &[x], "f", LoweringStage::ArithMath, &pool).unwrap();
        assert!(out.contains("arith.andi"), "{out}");
        assert!(out.contains("arith.xori"), "{out}");
        let out = lower(e, &[x], "f", LoweringStage::Llvm, &pool).unwrap();
        assert!(out.contains("llvm.and"), "{out}");
        assert!(out.contains("llvm.xor"), "{out}");
    }

    #[test]
    fn piecewise_with_unbound_symbol_in_branch_is_refused() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = pool.piecewise(vec![(pool.pred_lt(x, pool.integer(0_i32)), y)], x);
        let err = lower(e, &[x], "f", LoweringStage::ArithMath, &pool).unwrap_err();
        assert_eq!(err.code(), "E-MLIR-001");
    }
}
