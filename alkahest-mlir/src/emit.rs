//! Textual emission of the high-level `alkahest` MLIR dialect.
//!
//! The emitter walks an [`ExprPool`] rooted at an [`ExprId`] and produces a
//! self-contained `func.func` whose body is populated with `alkahest.*`
//! ops.  Polynomial subtrees are lifted to [`AlkahestOp::PolyEval`] (naive
//! form) — the canonicalize pass in `crate::canon` rewrites them into
//! [`AlkahestOp::Horner`].

use alkahest_core::kernel::{ExprData, ExprId, ExprPool};
use alkahest_core::poly::UniPoly;
use std::collections::HashMap;

use crate::ops::AlkahestOp;

/// Options for high-level dialect emission.
#[derive(Debug, Clone)]
pub struct EmitOptions {
    /// Name for the generated `func.func`.
    pub fn_name: String,
    /// If true, walk the tree and lift univariate-polynomial subtrees into
    /// `alkahest.poly_eval` ops.  Default: true.
    pub detect_polynomials: bool,
    /// If true, apply the canonicalize pass to rewrite `alkahest.poly_eval`
    /// ops into `alkahest.horner`.  Default: true.
    pub canonicalize: bool,
}

impl Default for EmitOptions {
    fn default() -> Self {
        EmitOptions {
            fn_name: "alkahest_fn".to_string(),
            detect_polynomials: true,
            canonicalize: true,
        }
    }
}

/// Emit the high-level `alkahest` dialect representation of `expr`.
///
/// `inputs` gives the SSA order of the function arguments: each symbol in
/// `inputs` becomes `%arg0 … %argN` inside the body.
pub fn emit_dialect(
    expr: ExprId,
    inputs: &[ExprId],
    opts: &EmitOptions,
    pool: &ExprPool,
) -> String {
    let mut emitter = DialectEmitter::new(inputs, opts);
    let result = emitter.emit(expr, pool);

    let args: Vec<String> = (0..inputs.len()).map(|i| format!("%arg{i}: f64")).collect();
    let body = emitter.body.join("\n    ");
    format!(
        "module {{\n  func.func @{fn}({args}) -> f64 {{\n    {body}\n    func.return {result} : f64\n  }}\n}}\n",
        fn = opts.fn_name,
        args = args.join(", "),
        body = body,
        result = result,
    )
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

pub(crate) struct DialectEmitter<'a> {
    pub(crate) arg_map: HashMap<ExprId, String>,
    /// Cache of already-emitted nodes → SSA name, so each subtree is lowered once.
    pub(crate) cache: HashMap<ExprId, String>,
    pub(crate) body: Vec<String>,
    pub(crate) counter: usize,
    pub(crate) opts: &'a EmitOptions,
    pub(crate) input_set: Vec<ExprId>,
}

impl<'a> DialectEmitter<'a> {
    fn new(inputs: &[ExprId], opts: &'a EmitOptions) -> Self {
        let mut arg_map = HashMap::new();
        for (i, &id) in inputs.iter().enumerate() {
            arg_map.insert(id, format!("%arg{i}"));
        }
        DialectEmitter {
            arg_map,
            cache: HashMap::new(),
            body: Vec::new(),
            counter: 0,
            opts,
            input_set: inputs.to_vec(),
        }
    }

    pub(crate) fn fresh(&mut self) -> String {
        let v = format!("%v{}", self.counter);
        self.counter += 1;
        v
    }

    fn emit(&mut self, expr: ExprId, pool: &ExprPool) -> String {
        // Arg?
        if let Some(s) = self.arg_map.get(&expr) {
            return s.clone();
        }
        if let Some(s) = self.cache.get(&expr) {
            return s.clone();
        }

        // Try polynomial lift.  Only univariate in a single input symbol; skip
        // the top-level if it's trivially a symbol/constant.
        if self.opts.detect_polynomials {
            if let Some(ssa) = self.try_emit_polynomial(expr, pool) {
                self.cache.insert(expr, ssa.clone());
                return ssa;
            }
        }

        let ssa = self.emit_raw(expr, pool);
        self.cache.insert(expr, ssa.clone());
        ssa
    }

    fn try_emit_polynomial(&mut self, expr: ExprId, pool: &ExprPool) -> Option<String> {
        // Skip leaves and Pow nodes.  Pow nodes preserve their factored
        // structure via emit_raw (base and exponent each recurse through emit),
        // so lifting Pow(poly, n) into a flat high-degree Horner form here
        // causes catastrophic cancellation when the expanded polynomial has
        // large coefficients that nearly cancel at the evaluation point.
        let is_compound = pool.with(expr, |d| matches!(d, ExprData::Add(_) | ExprData::Mul(_)));
        if !is_compound {
            return None;
        }
        // Try each input as the polynomial variable; pick the first that succeeds.
        for &var in &self.input_set.clone() {
            if let Ok(poly) = UniPoly::from_symbolic(expr, var, pool) {
                // Only lift non-trivial polys (degree ≥ 2) — otherwise plain
                // arith ops are already optimal.
                if poly.degree() >= 2 {
                    let Some(coeffs) = poly.coefficients_i64_checked() else {
                        continue;
                    };
                    let x = self.arg_map.get(&var).cloned().unwrap_or_else(|| {
                        // fallback (shouldn't trigger for inputs)
                        let v = self.fresh();
                        self.body.push(format!("{v} = alkahest.sym : f64"));
                        v
                    });
                    let coeff_attr = format_coeffs_attr(&coeffs);
                    let form = if self.opts.canonicalize {
                        "horner"
                    } else {
                        "naive"
                    };
                    let op = if self.opts.canonicalize {
                        AlkahestOp::Horner
                    } else {
                        AlkahestOp::PolyEval
                    };
                    let out = self.fresh();
                    self.body.push(format!(
                        "{out} = {op} {x} {{coeffs = {coeffs}, form = \"{form}\"}} : (f64) -> f64",
                        op = op.mnemonic(),
                        coeffs = coeff_attr,
                        form = form,
                    ));
                    return Some(out);
                }
            }
        }
        None
    }

    fn emit_raw(&mut self, expr: ExprId, pool: &ExprPool) -> String {
        enum Node {
            Const(f64),
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow { base: ExprId, exp: ExprId },
            Call { name: String, args: Vec<ExprId> },
            Unknown,
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
            ExprData::Symbol { .. } => Node::Unknown, // a free symbol not in inputs
            _ => Node::Unknown,
        });

        match node {
            Node::Const(v) => self.emit_const(v),
            Node::Add(args) => {
                let vs: Vec<String> = args.iter().map(|&a| self.emit(a, pool)).collect();
                self.emit_nary(AlkahestOp::Add, &vs)
            }
            Node::Mul(args) => {
                let vs: Vec<String> = args.iter().map(|&a| self.emit(a, pool)).collect();
                self.emit_nary(AlkahestOp::Mul, &vs)
            }
            Node::Pow { base, exp } => {
                let b = self.emit(base, pool);
                let e = self.emit(exp, pool);
                let out = self.fresh();
                self.body.push(format!(
                    "{out} = {op} {b}, {e} : (f64, f64) -> f64",
                    op = AlkahestOp::Pow.mnemonic(),
                ));
                out
            }
            Node::Call { name, args } => {
                let vs: Vec<String> = args.iter().map(|&a| self.emit(a, pool)).collect();
                let operands = vs.join(", ");
                let sig = vec!["f64"; vs.len()].join(", ");
                let out = self.fresh();
                self.body.push(format!(
                    "{out} = {op} @{name}({operands}) : ({sig}) -> f64",
                    op = AlkahestOp::Call.mnemonic(),
                ));
                out
            }
            Node::Unknown => {
                // Emit as a free symbolic value so the text is still valid.
                let out = self.fresh();
                self.body.push(format!(
                    "{out} = {op} : f64  // unresolved",
                    op = AlkahestOp::Sym.mnemonic()
                ));
                out
            }
        }
    }

    fn emit_const(&mut self, val: f64) -> String {
        let out = self.fresh();
        self.body.push(format!(
            "{out} = {op} {{value = {val:?} : f64}} : f64",
            op = AlkahestOp::Const.mnemonic()
        ));
        out
    }

    fn emit_nary(&mut self, op: AlkahestOp, vs: &[String]) -> String {
        assert!(!vs.is_empty());
        if vs.len() == 1 {
            return vs[0].clone();
        }
        let operands = vs.join(", ");
        let sig = vec!["f64"; vs.len()].join(", ");
        let out = self.fresh();
        self.body.push(format!(
            "{out} = {op} {operands} : ({sig}) -> f64",
            op = op.mnemonic()
        ));
        out
    }
}

fn format_coeffs_attr(coeffs: &[i64]) -> String {
    // MLIR dense<[a, b, c]> : tensor<Nxf64>
    let n = coeffs.len();
    let inner = coeffs
        .iter()
        .map(|c| format!("{c:.1}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("dense<[{inner}]> : tensor<{n}xf64>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use alkahest_core::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn emits_module_and_func() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let out = emit_dialect(x, &[x], &EmitOptions::default(), &pool);
        assert!(out.contains("module"));
        assert!(out.contains("func.func @alkahest_fn"));
        assert!(out.contains("func.return %arg0"));
    }

    #[test]
    fn sum_lowers_to_alkahest_add() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let sum = pool.add(vec![x, y]);
        let out = emit_dialect(sum, &[x, y], &EmitOptions::default(), &pool);
        assert!(out.contains("alkahest.add %arg0, %arg1"), "got:\n{out}");
    }

    #[test]
    fn polynomial_lifts_to_horner_when_canonicalized() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // x^2 + 2x + 1
        let x2 = pool.pow(x, pool.integer(2_i32));
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let one = pool.integer(1_i32);
        let expr = pool.add(vec![x2, two_x, one]);
        let out = emit_dialect(expr, &[x], &EmitOptions::default(), &pool);
        assert!(
            out.contains("alkahest.horner"),
            "expected alkahest.horner in:\n{out}"
        );
        assert!(out.contains("dense<["), "coeffs attr missing:\n{out}");
    }

    #[test]
    fn polynomial_stays_poly_eval_when_not_canonicalized() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let expr = pool.add(vec![x2, x, pool.integer(1_i32)]);
        let opts = EmitOptions {
            canonicalize: false,
            ..EmitOptions::default()
        };
        let out = emit_dialect(expr, &[x], &opts, &pool);
        assert!(out.contains("alkahest.poly_eval"), "got:\n{out}");
    }

    #[test]
    fn func_call_lowers_to_alkahest_call() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let out = emit_dialect(sin_x, &[x], &EmitOptions::default(), &pool);
        assert!(out.contains("alkahest.call @sin"), "got:\n{out}");
    }
}
