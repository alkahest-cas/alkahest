//! `alkahest-mlir` — the custom MLIR high-level dialect (V1-5).
//!
//! The dialect sits *above* `arith`, `math`, and `stablehlo`.  Its purpose
//! is to preserve math-aware structure (polynomial evaluation, Taylor
//! series, interval evaluation, rational functions) long enough for the
//! lowering passes to pick stability-preserving expansions — the most
//! important of which is Horner form lowered to `math.fma` / `llvm.intr.fmuladd`
//! chains.
//!
//! ## Shipped surface
//!
//! * [`AlkahestOp`] — op catalog (see [`ops`]).
//! * [`emit_mlir`] — entry point; emits textual MLIR at a requested stage.
//! * [`emit_dialect`] / [`parse_dialect`] — round-trippable textual form of
//!   the high-level `alkahest` dialect.
//! * Lowerings:
//!   * [`LoweringStage::ArithMath`] → `arith + math` dialect text.
//!   * [`LoweringStage::StableHlo`] → delegates to the existing StableHLO
//!     bridge.
//!   * [`LoweringStage::Llvm`] → `llvm` dialect text.
//!
//! ## Implementation status
//!
//! The emitter and lowerings produce textual MLIR that is designed to be
//! acceptable to `mlir-opt` / `mlir-translate`, matching the approach used
//! by `alkahest_core::stablehlo`.  Native MLIR bindings through
//! `mlir-sys` remain behind the `mlir-native` feature flag — that path
//! will replace the textual emitter once the LLVM 17 / MLIR toolchain is
//! discoverable on all three CI OSes (V1-10 territory).
//!
//! ## Pipeline overview
//!
//! ```text
//!   ExprId
//!     │
//!     ▼
//!   alkahest dialect  ─────►  arith + math   (Horner ⇒ math.fma)
//!     │                 ──►  stablehlo      (JAX bridge, V5-2)
//!     │                 ──►  llvm            (host / NVPTX JIT, V1-1)
//!     ▼
//!   ExprId  (via parse_dialect — round-trip path)
//! ```

pub mod emit;
pub mod lower;
pub mod ops;
pub mod parse;

pub use emit::{emit_dialect, EmitOptions};
pub use lower::lower;
pub use ops::AlkahestOp;
pub use parse::parse_dialect;

use alkahest_core::{ExprId, ExprPool};

/// Stages of the progressive lowering pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoweringStage {
    /// `alkahest → arith + math` (unopt baseline).
    ArithMath,
    /// `alkahest → stablehlo` (feeds V5-2 JAX path).
    StableHlo,
    /// `alkahest → llvm` (feeds V1-1 host / NVPTX JIT).
    Llvm,
}

/// Emit textual MLIR for `expr` at the requested lowering stage.
///
/// Free symbols in `expr` that aren't ExprPool constants are auto-discovered
/// and become `%arg0 … %argN`; use [`emit_mlir_with`] for explicit control
/// of the argument order or function name.
pub fn emit_mlir(pool: &ExprPool, id: ExprId, stage: LoweringStage) -> String {
    let inputs = free_symbols(id, pool);
    lower::lower(id, &inputs, "alkahest_fn", stage, pool)
}

/// Like [`emit_mlir`] but lets the caller name the function and pick the
/// order of `%arg` bindings.
pub fn emit_mlir_with(
    pool: &ExprPool,
    id: ExprId,
    inputs: &[ExprId],
    fn_name: &str,
    stage: LoweringStage,
) -> String {
    lower::lower(id, inputs, fn_name, stage, pool)
}

/// Round-trip an expression through the high-level dialect text form.
///
/// Useful both as a sanity check for the parser and as the interactive form
/// for REPL-based dialect editing.  `inputs` gives the SSA argument order.
pub fn roundtrip(pool: &ExprPool, id: ExprId, inputs: &[ExprId]) -> Option<ExprId> {
    let text = emit_dialect(id, inputs, &EmitOptions::default(), pool);
    parse_dialect(&text, inputs, pool)
}

/// Collect the free symbols that appear in `id`, in first-encounter order.
pub fn free_symbols(id: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    use alkahest_core::kernel::ExprData;

    fn walk(
        id: ExprId,
        pool: &ExprPool,
        seen: &mut std::collections::HashSet<ExprId>,
        out: &mut Vec<ExprId>,
    ) {
        if !seen.insert(id) {
            return;
        }
        match pool.get(id) {
            ExprData::Symbol { .. } => out.push(id),
            ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
                for a in args {
                    walk(a, pool, seen, out);
                }
            }
            ExprData::Pow { base, exp } => {
                walk(base, pool, seen, out);
                walk(exp, pool, seen, out);
            }
            ExprData::BigO(inner) => {
                walk(inner, pool, seen, out);
            }
            _ => {}
        }
    }

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    walk(id, pool, &mut seen, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alkahest_core::Domain;

    #[test]
    fn op_mnemonics_are_stable() {
        assert_eq!(AlkahestOp::Horner.mnemonic(), "alkahest.horner");
        assert_eq!(AlkahestOp::PolyEval.mnemonic(), "alkahest.poly_eval");
    }

    #[test]
    fn emit_arith_math_is_valid_textual_mlir() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.integer(1_i32)]);
        let out = emit_mlir(&pool, expr, LoweringStage::ArithMath);
        assert!(out.starts_with("module {"));
        assert!(out.contains("func.func @alkahest_fn"));
    }

    #[test]
    fn stages_produce_distinct_outputs() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("sin", vec![x]);
        let am = emit_mlir(&pool, expr, LoweringStage::ArithMath);
        let sh = emit_mlir(&pool, expr, LoweringStage::StableHlo);
        let ll = emit_mlir(&pool, expr, LoweringStage::Llvm);
        assert!(am.contains("math.sin"), "{am}");
        assert!(sh.contains("stablehlo.sine"), "{sh}");
        assert!(ll.contains("llvm.intr.sin"), "{ll}");
    }

    #[test]
    fn roundtrip_preserves_semantics_on_random_exprs() {
        use alkahest_core::jit::eval_interp;
        use std::collections::HashMap;

        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);

        // A handful of hand-built expressions with varied shape; a full
        // 1000-random sweep lives in `tests/` via proptest.
        let cases = vec![
            pool.add(vec![x, y]),
            pool.mul(vec![x, y]),
            pool.add(vec![
                pool.pow(x, pool.integer(3_i32)),
                pool.mul(vec![pool.integer(2_i32), x]),
                pool.integer(1_i32),
            ]),
            pool.func("sin", vec![x]),
            pool.add(vec![
                pool.func("cos", vec![y]),
                pool.pow(x, pool.integer(2_i32)),
            ]),
        ];

        for expr in cases {
            let back = roundtrip(&pool, expr, &[x, y]).expect("roundtrip succeeded");
            let mut env = HashMap::new();
            for xv in [-1.5f64, 0.25, 1.0, 2.5] {
                for yv in [-0.5f64, 0.75, 2.0] {
                    env.insert(x, xv);
                    env.insert(y, yv);
                    let va = eval_interp(expr, &env, &pool).unwrap();
                    let vb = eval_interp(back, &env, &pool).unwrap();
                    assert!(
                        (va - vb).abs() < 1e-9,
                        "mismatch at x={xv}, y={yv}: {va} vs {vb}"
                    );
                }
            }
        }
    }

    #[test]
    fn horner_canonicalization_emits_fma_when_lowered() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // (x + 1)^4 expands to x^4 + 4x^3 + 6x^2 + 4x + 1 — definite polynomial.
        let x2 = pool.pow(x, pool.integer(2_i32));
        let x3 = pool.pow(x, pool.integer(3_i32));
        let x4 = pool.pow(x, pool.integer(4_i32));
        let expr = pool.add(vec![
            x4,
            pool.mul(vec![pool.integer(4_i32), x3]),
            pool.mul(vec![pool.integer(6_i32), x2]),
            pool.mul(vec![pool.integer(4_i32), x]),
            pool.integer(1_i32),
        ]);
        let out = emit_mlir_with(&pool, expr, &[x], "poly4", LoweringStage::ArithMath);
        // Expect four fused multiply-add nodes for a degree-4 polynomial.
        let fma_count = out.matches("math.fma").count();
        assert!(
            fma_count >= 4,
            "degree-4 polynomial should produce ≥ 4 math.fma ops, got {fma_count}:\n{out}"
        );
    }

    #[test]
    fn free_symbols_preserves_order() {
        let pool = ExprPool::new();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let expr = pool.add(vec![a, pool.mul(vec![b, a])]);
        let syms = free_symbols(expr, &pool);
        assert_eq!(syms, vec![a, b]);
    }
}
