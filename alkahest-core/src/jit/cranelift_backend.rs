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

use super::{
    topo_sort, CompileTier, CompiledFn, CompiledFnInner, JitBulkFn, JitError, JitScalarFn,
};
use crate::kernel::{ExprData, ExprId, ExprPool};
use cranelift_codegen::ir::{condcodes::IntCC, types, AbiParam, BlockArg, InstBuilder, MemFlags};
use cranelift_codegen::{settings, settings::Configurable};
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
extern "C" fn tramp_sinh(x: f64) -> f64 {
    x.sinh()
}
extern "C" fn tramp_cosh(x: f64) -> f64 {
    x.cosh()
}
extern "C" fn tramp_tanh(x: f64) -> f64 {
    x.tanh()
}
extern "C" fn tramp_asin(x: f64) -> f64 {
    x.asin()
}
extern "C" fn tramp_acos(x: f64) -> f64 {
    x.acos()
}
extern "C" fn tramp_atan(x: f64) -> f64 {
    x.atan()
}
extern "C" fn tramp_asinh(x: f64) -> f64 {
    x.asinh()
}
extern "C" fn tramp_acosh(x: f64) -> f64 {
    x.acosh()
}
extern "C" fn tramp_atanh(x: f64) -> f64 {
    x.atanh()
}
extern "C" fn tramp_floor(x: f64) -> f64 {
    x.floor()
}
extern "C" fn tramp_ceil(x: f64) -> f64 {
    x.ceil()
}
extern "C" fn tramp_round(x: f64) -> f64 {
    x.round()
}
/// Matches `SignPrimitive::numeric_f64`: `0` for `x == 0`, `±1` otherwise.
extern "C" fn tramp_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
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
    sinh_id: FuncId,
    cosh_id: FuncId,
    tanh_id: FuncId,
    asin_id: FuncId,
    acos_id: FuncId,
    atan_id: FuncId,
    asinh_id: FuncId,
    acosh_id: FuncId,
    atanh_id: FuncId,
    floor_id: FuncId,
    ceil_id: FuncId,
    round_id: FuncId,
    sign_id: FuncId,
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
                "sinh" => math.sinh_id,
                "cosh" => math.cosh_id,
                "tanh" => math.tanh_id,
                "asin" => math.asin_id,
                "acos" => math.acos_id,
                "atan" => math.atan_id,
                "asinh" => math.asinh_id,
                "acosh" => math.acosh_id,
                "atanh" => math.atanh_id,
                "floor" => math.floor_id,
                "ceil" => math.ceil_id,
                "round" => math.round_id,
                "sign" => math.sign_id,
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
// Load input variables into the values map (scalar or batch layout)
// ---------------------------------------------------------------------------

fn load_input_vars(
    builder: &mut FunctionBuilder,
    inputs_ptr: cranelift_codegen::ir::Value,
    inputs: &[ExprId],
    values: &mut HashMap<ExprId, cranelift_codegen::ir::Value>,
    point_idx: Option<cranelift_codegen::ir::Value>,
    n_points: Option<cranelift_codegen::ir::Value>,
) {
    for (i, &var) in inputs.iter().enumerate() {
        let val = if let (Some(idx), Some(n_pts)) = (point_idx, n_points) {
            let var_i = builder.ins().iconst(types::I64, i as i64);
            let stride = builder.ins().imul(var_i, n_pts);
            let elem = builder.ins().iadd(stride, idx);
            let byte_off = builder.ins().imul_imm(elem, 8);
            let addr = builder.ins().iadd(inputs_ptr, byte_off);
            builder.ins().load(types::F64, MemFlags::trusted(), addr, 0)
        } else {
            let byte_offset = (i * std::mem::size_of::<f64>()) as i32;
            builder
                .ins()
                .load(types::F64, MemFlags::trusted(), inputs_ptr, byte_offset)
        };
        values.insert(var, val);
    }
}

fn emit_eval_body(
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    math: &MathFuncIds,
    inputs_ptr: cranelift_codegen::ir::Value,
    point_idx: Option<cranelift_codegen::ir::Value>,
    n_points: Option<cranelift_codegen::ir::Value>,
) -> Result<cranelift_codegen::ir::Value, JitError> {
    let mut values: HashMap<ExprId, cranelift_codegen::ir::Value> = HashMap::new();
    load_input_vars(
        builder,
        inputs_ptr,
        inputs,
        &mut values,
        point_idx,
        n_points,
    );
    let topo = topo_sort(expr, pool);
    for &node in &topo {
        if values.contains_key(&node) {
            continue;
        }
        let val = codegen_node(node, pool, &values, builder, module, math)?;
        values.insert(node, val);
    }
    values
        .get(&expr)
        .copied()
        .ok_or_else(|| JitError::CompilationFailed("root node not emitted".to_string()))
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
    jit_builder.symbol("alkahest_sinh", tramp_sinh as *const u8);
    jit_builder.symbol("alkahest_cosh", tramp_cosh as *const u8);
    jit_builder.symbol("alkahest_tanh", tramp_tanh as *const u8);
    jit_builder.symbol("alkahest_asin", tramp_asin as *const u8);
    jit_builder.symbol("alkahest_acos", tramp_acos as *const u8);
    jit_builder.symbol("alkahest_atan", tramp_atan as *const u8);
    jit_builder.symbol("alkahest_asinh", tramp_asinh as *const u8);
    jit_builder.symbol("alkahest_acosh", tramp_acosh as *const u8);
    jit_builder.symbol("alkahest_atanh", tramp_atanh as *const u8);
    jit_builder.symbol("alkahest_floor", tramp_floor as *const u8);
    jit_builder.symbol("alkahest_ceil", tramp_ceil as *const u8);
    jit_builder.symbol("alkahest_round", tramp_round as *const u8);
    jit_builder.symbol("alkahest_sign", tramp_sign as *const u8);

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
        sinh_id: decl(&mut module, "alkahest_sinh", &f1_sig)?,
        cosh_id: decl(&mut module, "alkahest_cosh", &f1_sig)?,
        tanh_id: decl(&mut module, "alkahest_tanh", &f1_sig)?,
        asin_id: decl(&mut module, "alkahest_asin", &f1_sig)?,
        acos_id: decl(&mut module, "alkahest_acos", &f1_sig)?,
        atan_id: decl(&mut module, "alkahest_atan", &f1_sig)?,
        asinh_id: decl(&mut module, "alkahest_asinh", &f1_sig)?,
        acosh_id: decl(&mut module, "alkahest_acosh", &f1_sig)?,
        atanh_id: decl(&mut module, "alkahest_atanh", &f1_sig)?,
        floor_id: decl(&mut module, "alkahest_floor", &f1_sig)?,
        ceil_id: decl(&mut module, "alkahest_ceil", &f1_sig)?,
        round_id: decl(&mut module, "alkahest_round", &f1_sig)?,
        sign_id: decl(&mut module, "alkahest_sign", &f1_sig)?,
    };

    // ------------------------------------------------------------------
    // 4–5. Scalar eval: fn(*const f64, i64) -> f64
    // ------------------------------------------------------------------
    let ptr_type = module.target_config().pointer_type();

    let mut eval_sig = module.make_signature();
    eval_sig.params.push(AbiParam::new(ptr_type));
    eval_sig.params.push(AbiParam::new(types::I64));
    eval_sig.returns.push(AbiParam::new(types::F64));

    let scalar_id = module
        .declare_function("alkahest_eval", Linkage::Export, &eval_sig)
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

    let mut scalar_ctx = module.make_context();
    scalar_ctx.func.signature = eval_sig;
    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut scalar_ctx.func, &mut func_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);
        let inputs_ptr = builder.block_params(block)[0];
        let result = emit_eval_body(
            expr,
            inputs,
            pool,
            &mut builder,
            &mut module,
            &math,
            inputs_ptr,
            None,
            None,
        )?;
        builder.ins().return_(&[result]);
        builder.finalize();
    }
    module
        .define_function(scalar_id, &mut scalar_ctx)
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
    module.clear_context(&mut scalar_ctx);

    // ------------------------------------------------------------------
    // 6. Bulk eval: fn(*const f64, n_vars, *mut f64, n_points) -> ()
    // ------------------------------------------------------------------
    let mut bulk_sig = module.make_signature();
    bulk_sig.params.push(AbiParam::new(ptr_type));
    bulk_sig.params.push(AbiParam::new(types::I64));
    bulk_sig.params.push(AbiParam::new(ptr_type));
    bulk_sig.params.push(AbiParam::new(types::I64));

    let bulk_id = module
        .declare_function("alkahest_eval_bulk", Linkage::Export, &bulk_sig)
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

    let mut bulk_ctx = module.make_context();
    bulk_ctx.func.signature = bulk_sig;
    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut bulk_ctx.func, &mut func_ctx);
        let entry = builder.create_block();
        let loop_hdr = builder.create_block();
        let loop_body = builder.create_block();
        let exit = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.append_block_param(loop_hdr, types::I64);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let bulk_inputs_ptr = builder.block_params(entry)[0];
        let bulk_outputs_ptr = builder.block_params(entry)[2];
        let bulk_n_points = builder.block_params(entry)[3];
        let zero = builder.ins().iconst(types::I64, 0);
        let zero_arg = BlockArg::from(zero);
        builder
            .ins()
            .jump(loop_hdr, std::slice::from_ref(&zero_arg));

        builder.switch_to_block(loop_hdr);
        let loop_idx = builder.block_params(loop_hdr)[0];
        let done = builder
            .ins()
            .icmp(IntCC::SignedGreaterThanOrEqual, loop_idx, bulk_n_points);
        builder.ins().brif(done, exit, &[], loop_body, &[]);

        builder.switch_to_block(loop_body);
        let result = emit_eval_body(
            expr,
            inputs,
            pool,
            &mut builder,
            &mut module,
            &math,
            bulk_inputs_ptr,
            Some(loop_idx),
            Some(bulk_n_points),
        )?;
        let out_byte_off = builder.ins().imul_imm(loop_idx, 8);
        let out_addr = builder.ins().iadd(bulk_outputs_ptr, out_byte_off);
        builder
            .ins()
            .store(MemFlags::trusted(), result, out_addr, 0);
        let next = builder.ins().iadd_imm(loop_idx, 1);
        let next_arg = BlockArg::from(next);
        builder
            .ins()
            .jump(loop_hdr, std::slice::from_ref(&next_arg));
        builder.seal_block(loop_body);

        builder.switch_to_block(exit);
        builder.seal_block(loop_hdr);
        builder.seal_block(exit);
        builder.ins().return_(&[]);
        builder.finalize();
    }
    module
        .define_function(bulk_id, &mut bulk_ctx)
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
    module.clear_context(&mut bulk_ctx);

    module
        .finalize_definitions()
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

    let scalar_ptr = module.get_finalized_function(scalar_id);
    let bulk_ptr = module.get_finalized_function(bulk_id);
    let fn_ptr: JitScalarFn = unsafe { std::mem::transmute(scalar_ptr) };
    let bulk_fn: JitBulkFn = unsafe { std::mem::transmute(bulk_ptr) };

    Ok(CompiledFn {
        inner: CompiledFnInner::Cranelift {
            fn_ptr,
            bulk_fn: Some(bulk_fn),
            _module: Box::new(module),
        },
        n_inputs: inputs.len(),
        tier: CompileTier::Cranelift,
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

    /// The newly-registered unary math trampolines (sinh, cosh, tanh, asin,
    /// acos, atan, asinh, acosh, atanh, floor, ceil, round, sign) all compile
    /// and evaluate correctly via the Cranelift backend.
    #[test]
    fn cranelift_unary_math_trampolines() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);

        let cases: &[(&str, f64, f64)] = &[
            ("sinh", 0.5, 0.5_f64.sinh()),
            ("cosh", 0.5, 0.5_f64.cosh()),
            ("tanh", 0.5, 0.5_f64.tanh()),
            ("asin", 0.5, 0.5_f64.asin()),
            ("acos", 0.5, 0.5_f64.acos()),
            ("atan", 0.5, 0.5_f64.atan()),
            ("asinh", 0.5, 0.5_f64.asinh()),
            ("acosh", 1.5, 1.5_f64.acosh()),
            ("atanh", 0.5, 0.5_f64.atanh()),
            ("floor", 1.7, 1.0),
            ("ceil", 1.2, 2.0),
            ("round", 1.5, 2.0),
            ("sign", -3.0, -1.0),
        ];
        for &(name, input, expected) in cases {
            let expr = pool.func(name, vec![x]);
            let f = compile_cranelift(expr, &[x], &pool).unwrap();
            let got = f.call(&[input]);
            assert!(
                (got - expected).abs() < 1e-12,
                "{name}({input}): got {got}, expected {expected}"
            );
        }
    }
}
