//! Cranelift Tier-1 JIT backend for symbolic expression evaluation.
//!
//! Cranelift (the Wasmtime code generator) is written in pure Rust and requires
//! no system LLVM installation, making it suitable for inclusion in the default
//! PyPI wheel.  It compiles ~10× faster than LLVM at the cost of ~10–20% slower
//! generated code — the right trade-off for interactive notebook use.
//!
//! # Architecture
//!
//! ```text
//! ExprId ──► topo_sort ──► codegen_node ──► Cranelift IR ──► JITModule
//!                                                          ──► fn(*const f64, u64) -> f64
//! ```
//!
//! Math functions (sin, cos, exp, log, sqrt, tan, abs, pow) are implemented as
//! `extern "C"` trampolines and registered with the `JITBuilder` so Cranelift
//! can call them as imported symbols.

use super::{topo_sort, CompiledFn, CompiledFnInner, JitError};
use crate::kernel::{ExprData, ExprId, ExprPool};
use cranelift_codegen::{
    ir::{types, AbiParam, InstBuilder, MemFlags},
    settings,
    settings::Configurable,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Math trampolines — extern "C" wrappers called by JIT-compiled code
// ---------------------------------------------------------------------------

extern "C" fn tramp_sin(x: f64) -> f64 {
    x.sin()
}
extern "C" fn tramp_cos(x: f64) -> f64 {
    x.cos()
}
extern "C" fn tramp_exp(x: f64) -> f64 {
    x.exp()
}
extern "C" fn tramp_log(x: f64) -> f64 {
    x.ln()
}
extern "C" fn tramp_sqrt(x: f64) -> f64 {
    x.sqrt()
}
extern "C" fn tramp_tan(x: f64) -> f64 {
    x.tan()
}
extern "C" fn tramp_abs(x: f64) -> f64 {
    x.abs()
}
extern "C" fn tramp_pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

// ---------------------------------------------------------------------------
// Helper: declared function IDs for math imports
// ---------------------------------------------------------------------------

struct MathFuncIds {
    sin_id: FuncId,
    cos_id: FuncId,
    exp_id: FuncId,
    log_id: FuncId,
    sqrt_id: FuncId,
    tan_id: FuncId,
    abs_id: FuncId,
    pow_id: FuncId,
}

// ---------------------------------------------------------------------------
// codegen_node — emit Cranelift IR for a single DAG node
// ---------------------------------------------------------------------------

fn codegen_node(
    node: ExprId,
    pool: &ExprPool,
    values: &HashMap<ExprId, cranelift_codegen::ir::Value>,
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    math: &MathFuncIds,
) -> Result<cranelift_codegen::ir::Value, JitError> {
    match pool.get(node) {
        ExprData::Integer(n) => Ok(builder.ins().f64const(n.0.to_f64())),
        ExprData::Rational(r) => {
            let (num, den) = r.0.clone().into_numer_denom();
            Ok(builder.ins().f64const(num.to_f64() / den.to_f64()))
        }
        ExprData::Float(f) => Ok(builder.ins().f64const(f.inner.to_f64())),
        ExprData::Symbol { name, .. } => Err(JitError::UnsupportedNode(format!(
            "unbound symbol '{name}'"
        ))),
        ExprData::Add(args) => {
            let mut acc = builder.ins().f64const(0.0);
            for &a in &args {
                let v = *values
                    .get(&a)
                    .ok_or_else(|| JitError::CompilationFailed("missing Add child".to_string()))?;
                acc = builder.ins().fadd(acc, v);
            }
            Ok(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = builder.ins().f64const(1.0);
            for &a in &args {
                let v = *values
                    .get(&a)
                    .ok_or_else(|| JitError::CompilationFailed("missing Mul child".to_string()))?;
                acc = builder.ins().fmul(acc, v);
            }
            Ok(acc)
        }
        ExprData::Pow { base, exp } => {
            let b = *values
                .get(&base)
                .ok_or_else(|| JitError::CompilationFailed("missing Pow base".to_string()))?;
            let e = *values
                .get(&exp)
                .ok_or_else(|| JitError::CompilationFailed("missing Pow exp".to_string()))?;
            let pow_ref = module.declare_func_in_func(math.pow_id, builder.func);
            let call = builder.ins().call(pow_ref, &[b, e]);
            Ok(builder.inst_results(call)[0])
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let a = *values
                .get(&args[0])
                .ok_or_else(|| JitError::CompilationFailed("missing Func arg".to_string()))?;
            let func_id = match name.as_str() {
                "sin" => math.sin_id,
                "cos" => math.cos_id,
                "exp" => math.exp_id,
                "log" => math.log_id,
                "sqrt" => math.sqrt_id,
                "tan" => math.tan_id,
                "abs" => math.abs_id,
                other => {
                    return Err(JitError::UnsupportedNode(format!("function '{other}'")));
                }
            };
            let func_ref = module.declare_func_in_func(func_id, builder.func);
            let call = builder.ins().call(func_ref, &[a]);
            Ok(builder.inst_results(call)[0])
        }
        other => Err(JitError::UnsupportedNode(format!("{other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// compile_cranelift — public entry point
// ---------------------------------------------------------------------------

/// Compile `expr` to a native function using the Cranelift JIT backend.
///
/// Returns a [`CompiledFn`] wrapping the generated code and keeping the
/// [`JITModule`] alive so the code pages remain valid.
pub fn compile_cranelift(
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CompiledFn, JitError> {
    // ------------------------------------------------------------------
    // 1. ISA — native host architecture, speed optimised
    // ------------------------------------------------------------------
    let mut flag_builder = settings::builder();
    // Disable PIC; we're doing in-process JIT so we don't need it.
    flag_builder.set("is_pic", "false").unwrap();
    // Use colocated call stubs for imports.
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("opt_level", "speed").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let isa_builder = cranelift_native::builder()
        .map_err(|e| JitError::LlvmInitError(format!("Cranelift ISA builder: {e}")))?;
    let isa = isa_builder
        .finish(flags)
        .map_err(|e| JitError::CompilationFailed(format!("Cranelift ISA finish: {e}")))?;

    // ------------------------------------------------------------------
    // 2. JIT module — register math trampolines as known symbols
    // ------------------------------------------------------------------
    let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());
    jit_builder.symbol("alkahest_sin", tramp_sin as *const u8);
    jit_builder.symbol("alkahest_cos", tramp_cos as *const u8);
    jit_builder.symbol("alkahest_exp", tramp_exp as *const u8);
    jit_builder.symbol("alkahest_log", tramp_log as *const u8);
    jit_builder.symbol("alkahest_sqrt", tramp_sqrt as *const u8);
    jit_builder.symbol("alkahest_tan", tramp_tan as *const u8);
    jit_builder.symbol("alkahest_abs", tramp_abs as *const u8);
    jit_builder.symbol("alkahest_pow", tramp_pow as *const u8);

    let mut module = JITModule::new(jit_builder);

    // ------------------------------------------------------------------
    // 3. Declare imported math functions
    // ------------------------------------------------------------------
    let mut f1_sig = module.make_signature();
    f1_sig.params.push(AbiParam::new(types::F64));
    f1_sig.returns.push(AbiParam::new(types::F64));

    let mut f2_sig = module.make_signature();
    f2_sig.params.push(AbiParam::new(types::F64));
    f2_sig.params.push(AbiParam::new(types::F64));
    f2_sig.returns.push(AbiParam::new(types::F64));

    let decl = |module: &mut JITModule, name: &str, sig| {
        module
            .declare_function(name, Linkage::Import, sig)
            .map_err(|e| JitError::CompilationFailed(e.to_string()))
    };

    let math = MathFuncIds {
        sin_id: decl(&mut module, "alkahest_sin", &f1_sig)?,
        cos_id: decl(&mut module, "alkahest_cos", &f1_sig)?,
        exp_id: decl(&mut module, "alkahest_exp", &f1_sig)?,
        log_id: decl(&mut module, "alkahest_log", &f1_sig)?,
        sqrt_id: decl(&mut module, "alkahest_sqrt", &f1_sig)?,
        tan_id: decl(&mut module, "alkahest_tan", &f1_sig)?,
        abs_id: decl(&mut module, "alkahest_abs", &f1_sig)?,
        pow_id: decl(&mut module, "alkahest_pow", &f2_sig)?,
    };

    // ------------------------------------------------------------------
    // 4. Declare the exported eval function: fn(*const f64, i64) -> f64
    // ------------------------------------------------------------------
    let ptr_type = module.target_config().pointer_type();

    let mut eval_sig = module.make_signature();
    eval_sig.params.push(AbiParam::new(ptr_type)); // inputs pointer
    eval_sig.params.push(AbiParam::new(types::I64)); // count (for ABI compat with LLVM path)
    eval_sig.returns.push(AbiParam::new(types::F64));

    let func_id = module
        .declare_function("alkahest_eval", Linkage::Export, &eval_sig)
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

    // ------------------------------------------------------------------
    // 5. Build the function body
    // ------------------------------------------------------------------
    let mut ctx = module.make_context();
    ctx.func.signature = eval_sig;

    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);

        // params[0] = inputs ptr; params[1] = count (unused in codegen)
        let inputs_ptr = builder.block_params(block)[0];

        let mut values: HashMap<ExprId, cranelift_codegen::ir::Value> = HashMap::new();

        // Load each input variable from inputs[i]
        for (i, &var) in inputs.iter().enumerate() {
            let byte_offset = (i * std::mem::size_of::<f64>()) as i32;
            let val = builder
                .ins()
                .load(types::F64, MemFlags::trusted(), inputs_ptr, byte_offset);
            values.insert(var, val);
        }

        // Topological order so every child is ready before its parent
        let topo = topo_sort(expr, pool);
        for &node in &topo {
            if values.contains_key(&node) {
                continue; // already computed (input var or shared subexpr)
            }
            let val = codegen_node(node, pool, &values, &mut builder, &mut module, &math)?;
            values.insert(node, val);
        }

        let result = values
            .get(&expr)
            .copied()
            .ok_or_else(|| JitError::CompilationFailed("root node not emitted".to_string()))?;
        builder.ins().return_(&[result]);
        builder.finalize();
    }

    // ------------------------------------------------------------------
    // 6. Compile and finalise
    // ------------------------------------------------------------------
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

    let code_ptr = module.get_finalized_function(func_id);
    // SAFETY: `code_ptr` is a valid executable code page produced by
    // Cranelift. The `JITModule` stored in `_module` owns the allocation
    // and must outlive this function pointer.
    let fn_ptr: unsafe extern "C" fn(*const f64, u64) -> f64 =
        unsafe { std::mem::transmute(code_ptr) };

    Ok(CompiledFn {
        inner: CompiledFnInner::Cranelift {
            fn_ptr,
            _module: Box::new(module),
        },
        n_inputs: inputs.len(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn cranelift_constant() {
        let pool = p();
        let five = pool.integer(5_i32);
        let f = compile_cranelift(five, &[], &pool).unwrap();
        assert!((f.call(&[]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn cranelift_identity() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = compile_cranelift(x, &[x], &pool).unwrap();
        assert!((f.call(&[3.7]) - 3.7).abs() < 1e-10);
    }

    #[test]
    fn cranelift_add() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![x, y]);
        let f = compile_cranelift(expr, &[x, y], &pool).unwrap();
        assert!((f.call(&[2.0, 3.0]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn cranelift_polynomial() {
        // f(x) = x² + 2x + 1 = (x+1)²
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let one = pool.integer(1_i32);
        let expr = pool.add(vec![x2, two_x, one]);
        let f = compile_cranelift(expr, &[x], &pool).unwrap();
        // f(3) = 9 + 6 + 1 = 16
        assert!((f.call(&[3.0]) - 16.0).abs() < 1e-10);
    }

    #[test]
    fn cranelift_sin() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let f = compile_cranelift(sin_x, &[x], &pool).unwrap();
        let pi_2 = std::f64::consts::PI / 2.0;
        assert!((f.call(&[pi_2]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cranelift_exp() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let exp_x = pool.func("exp", vec![x]);
        let f = compile_cranelift(exp_x, &[x], &pool).unwrap();
        assert!((f.call(&[0.0]) - 1.0).abs() < 1e-10);
        assert!((f.call(&[1.0]) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn cranelift_pow() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(3_i32));
        let f = compile_cranelift(expr, &[x], &pool).unwrap();
        assert!((f.call(&[2.0]) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn cranelift_multivariate() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let y2 = pool.pow(y, pool.integer(2_i32));
        let expr = pool.add(vec![x2, y2]);
        let f = compile_cranelift(expr, &[x, y], &pool).unwrap();
        // Pythagorean triple: f(3,4) = 25
        assert!((f.call(&[3.0, 4.0]) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn cranelift_shared_dag_node() {
        // expr = (x+1) * (x+1)  — shares the same ExprId for (x+1)
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let xp1 = pool.add(vec![x, pool.integer(1_i32)]);
        let expr = pool.mul(vec![xp1, xp1]); // xp1 appears twice, same ExprId
        let f = compile_cranelift(expr, &[x], &pool).unwrap();
        // f(4) = 5 * 5 = 25
        assert!((f.call(&[4.0]) - 25.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "expected 1 inputs")]
    fn cranelift_wrong_n_inputs_panics() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = compile_cranelift(x, &[x], &pool).unwrap();
        f.call(&[]);
    }
}
