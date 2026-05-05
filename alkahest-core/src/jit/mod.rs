//! Phase 21 — LLVM JIT for compiled evaluation of symbolic expressions.
//!
//! Feature-gated behind `--features jit`.  Without the feature this module
//! still compiles but provides only the interpreter-based fallback.
//!
//! # Architecture
//!
//! ```text
//! ExprId  ──► codegen ──► LLVM IR ──► MCJIT ──► fn(*const f64, usize) -> f64
//! ```
//!
//! Supported primitives
//! ────────────────────
//! | Expr node      | LLVM lowering                              |
//! |----------------|--------------------------------------------|
//! | Integer(n)     | `arith.constant f64 n`                     |
//! | Rational(p/q)  | `arith.constant f64 p/q`                   |
//! | Float(x)       | `arith.constant f64 x`                     |
//! | Symbol         | load from input array by position          |
//! | Add([…])       | chain of `fadd`                            |
//! | Mul([…])       | chain of `fmul`                            |
//! | Pow(b, n)      | unrolled `fmul` for integer n, else `pow`  |
//! | sin/cos/…      | `llvm.sin`, `llvm.cos`, `llvm.exp`, …      |
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "jit")]
//! # {
//! use alkahest_core::kernel::{Domain, ExprPool};
//! use alkahest_core::jit::compile;
//!
//! let pool = ExprPool::new();
//! let x = pool.symbol("x", Domain::Real);
//! let y = pool.symbol("y", Domain::Real);
//! let expr = pool.add(vec![
//!     pool.mul(vec![x, x]),       // x²
//!     pool.mul(vec![y, y]),       // y²
//! ]);
//! let f = compile(expr, &[x, y], &pool).unwrap();
//! let result = f.call(&[3.0, 4.0]);   // 9 + 16 = 25
//! assert!((result - 25.0).abs() < 1e-10);
//! # }
//! ```

use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;
use std::fmt;

#[cfg(feature = "cuda")]
pub mod nvptx;
#[cfg(feature = "cuda")]
pub use nvptx::{compile_cuda, CudaCompiledFn, CudaError};

// ---------------------------------------------------------------------------
// Error type (always compiled)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum JitError {
    UnsupportedNode(String),
    CompilationFailed(String),
    LlvmInitError(String),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::UnsupportedNode(s) => write!(f, "unsupported expression node: {s}"),
            JitError::CompilationFailed(s) => write!(f, "JIT compilation failed: {s}"),
            JitError::LlvmInitError(s) => write!(f, "LLVM init error: {s}"),
        }
    }
}

impl std::error::Error for JitError {}

impl crate::errors::AlkahestError for JitError {
    fn code(&self) -> &'static str {
        match self {
            JitError::UnsupportedNode(_) => "E-JIT-001",
            JitError::CompilationFailed(_) => "E-JIT-002",
            JitError::LlvmInitError(_) => "E-JIT-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            JitError::UnsupportedNode(_) => Some(
                "use eval_expr (interpreted) or simplify the expression to remove unsupported nodes",
            ),
            JitError::CompilationFailed(_) => Some(
                "check LLVM installation; run with RUST_LOG=debug for details",
            ),
            JitError::LlvmInitError(_) => Some(
                "ensure LLVM 15 is installed and LLVM_SYS_150_PREFIX is set correctly",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// CompiledFn — wraps a callable function pointer
// ---------------------------------------------------------------------------

/// A JIT-compiled function that evaluates a symbolic expression numerically.
///
/// The function accepts a slice of `f64` inputs corresponding to the variables
/// given to `compile`.
pub struct CompiledFn {
    #[cfg(feature = "jit")]
    fn_ptr: unsafe extern "C" fn(*const f64, u64) -> f64,
    // execution_engine must be declared before _context so it drops first;
    // the context must outlive the execution engine.
    #[cfg(feature = "jit")]
    #[allow(dead_code)]
    execution_engine: inkwell::execution_engine::ExecutionEngine<'static>,
    #[cfg(feature = "jit")]
    _context: Box<inkwell::context::Context>,

    /// Fallback interpreter for when the `jit` feature is disabled.
    #[cfg(not(feature = "jit"))]
    #[allow(clippy::type_complexity)]
    interpreter: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,

    /// Number of inputs expected.
    pub n_inputs: usize,
}

impl CompiledFn {
    /// Evaluate the compiled function with the given inputs.
    ///
    /// `inputs.len()` must equal `n_inputs`.
    pub fn call(&self, inputs: &[f64]) -> f64 {
        assert_eq!(
            inputs.len(),
            self.n_inputs,
            "expected {} inputs, got {}",
            self.n_inputs,
            inputs.len()
        );

        #[cfg(feature = "jit")]
        {
            unsafe { (self.fn_ptr)(inputs.as_ptr(), inputs.len() as u64) }
        }

        #[cfg(not(feature = "jit"))]
        {
            (self.interpreter)(inputs)
        }
    }

    /// Batch-evaluate over N points.
    ///
    /// `inputs` is a slice of per-variable slices: `inputs[i]` contains the
    /// values of variable `i` for all N points.  All slices must have the same
    /// length N.  `output` must also have length N.
    ///
    /// This is the hot path for NumPy/JAX array evaluation (Phase 25).
    pub fn call_batch(&self, inputs: &[&[f64]], output: &mut [f64]) {
        let n = output.len();
        assert_eq!(
            inputs.len(),
            self.n_inputs,
            "expected {} input arrays, got {}",
            self.n_inputs,
            inputs.len()
        );
        for col in inputs {
            assert_eq!(col.len(), n, "all input arrays must have the same length");
        }
        for i in 0..n {
            let point: Vec<f64> = inputs.iter().map(|col| col[i]).collect();
            output[i] = self.call(&point);
        }
    }
}

// ---------------------------------------------------------------------------
// compile — main entry point
// ---------------------------------------------------------------------------

/// Compile `expr` to a native function.
///
/// `inputs` defines the ordered list of symbolic variables; their values must
/// be supplied in the same order when calling the returned `CompiledFn`.
pub fn compile(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> Result<CompiledFn, JitError> {
    #[cfg(feature = "jit")]
    {
        compile_llvm(expr, inputs, pool)
    }

    #[cfg(not(feature = "jit"))]
    {
        compile_interpreter(expr, inputs, pool)
    }
}

// ---------------------------------------------------------------------------
// Interpreter fallback (always available)
// ---------------------------------------------------------------------------

/// Tree-walking interpreter for evaluating symbolic expressions numerically.
///
/// This is always compiled (no `jit` feature needed) and serves as the
/// fallback when LLVM is unavailable.  For production workloads, prefer the
/// LLVM-JIT path via `compile` with `--features jit`.
pub fn eval_interp(expr: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<f64> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => {
            let (n, d) = r.0.clone().into_numer_denom();
            Some(n.to_f64() / d.to_f64())
        }
        ExprData::Float(f) => Some(f.inner.to_f64()),
        ExprData::Symbol { .. } => env.get(&expr).copied(),
        ExprData::Add(args) => {
            let mut sum = 0.0f64;
            for &a in &args {
                sum += eval_interp(a, env, pool)?;
            }
            Some(sum)
        }
        ExprData::Mul(args) => {
            let mut prod = 1.0f64;
            for &a in &args {
                prod *= eval_interp(a, env, pool)?;
            }
            Some(prod)
        }
        ExprData::Pow { base, exp } => {
            let b = eval_interp(base, env, pool)?;
            let e = eval_interp(exp, env, pool)?;
            Some(b.powf(e))
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let x = eval_interp(args[0], env, pool)?;
            Some(match name.as_str() {
                "sin" => x.sin(),
                "cos" => x.cos(),
                "tan" => x.tan(),
                "exp" => x.exp(),
                "log" => x.ln(),
                "sqrt" => x.sqrt(),
                "gamma" => rug::Float::with_val(53, x).gamma().to_f64(),
                "abs" => x.abs(),
                _ => return None,
            })
        }
        _ => None,
    }
}

#[cfg(not(feature = "jit"))]
fn compile_interpreter(
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CompiledFn, JitError> {
    let inputs_vec = inputs.to_vec();
    let n = inputs_vec.len();
    // We need to capture the pool data — snapshot the relevant nodes
    let snapshot = snapshot_expr(expr, pool);

    let interp = move |vals: &[f64]| -> f64 {
        let mut env: HashMap<ExprId, f64> = HashMap::new();
        for (&var, &val) in inputs_vec.iter().zip(vals.iter()) {
            env.insert(var, val);
        }
        eval_interp_snap(expr, &env, &snapshot).unwrap_or(f64::NAN)
    };

    Ok(CompiledFn {
        interpreter: Box::new(interp),
        n_inputs: n,
    })
}

// ---------------------------------------------------------------------------
// Snapshot-based interpreter (captures expression tree without pool reference)
// ---------------------------------------------------------------------------

/// A self-contained snapshot of an expression subgraph.
#[cfg(not(feature = "jit"))]
#[derive(Clone)]
pub struct ExprSnapshot {
    nodes: HashMap<ExprId, ExprData>,
}

#[cfg(not(feature = "jit"))]
fn snapshot_expr(root: ExprId, pool: &ExprPool) -> ExprSnapshot {
    let mut visited: std::collections::HashSet<ExprId> = std::collections::HashSet::new();
    let mut stack = vec![root];
    let mut nodes: HashMap<ExprId, ExprData> = HashMap::new();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let data = pool.get(id);
        match &data {
            ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
                stack.extend_from_slice(args);
            }
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                stack.push(*exp);
            }
            _ => {}
        }
        nodes.insert(id, data);
    }
    ExprSnapshot { nodes }
}

#[cfg(not(feature = "jit"))]
fn eval_interp_snap(expr: ExprId, env: &HashMap<ExprId, f64>, snap: &ExprSnapshot) -> Option<f64> {
    match snap.nodes.get(&expr)? {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => {
            let (n, d) = r.0.clone().into_numer_denom();
            Some(n.to_f64() / d.to_f64())
        }
        ExprData::Float(f) => Some(f.inner.to_f64()),
        ExprData::Symbol { .. } => env.get(&expr).copied(),
        ExprData::Add(args) => {
            let mut s = 0.0f64;
            for &a in args {
                s += eval_interp_snap(a, env, snap)?;
            }
            Some(s)
        }
        ExprData::Mul(args) => {
            let mut p = 1.0f64;
            for &a in args {
                p *= eval_interp_snap(a, env, snap)?;
            }
            Some(p)
        }
        ExprData::Pow { base, exp } => {
            Some(eval_interp_snap(*base, env, snap)?.powf(eval_interp_snap(*exp, env, snap)?))
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let x = eval_interp_snap(args[0], env, snap)?;
            Some(match name.as_str() {
                "sin" => x.sin(),
                "cos" => x.cos(),
                "tan" => x.tan(),
                "exp" => x.exp(),
                "log" => x.ln(),
                "sqrt" => x.sqrt(),
                "gamma" => rug::Float::with_val(53, x).gamma().to_f64(),
                "abs" => x.abs(),
                _ => return None,
            })
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// LLVM JIT path (only when `--features jit`)
// ---------------------------------------------------------------------------

#[cfg(feature = "jit")]
mod llvm_backend {
    use super::*;
    use inkwell::{
        builder::Builder,
        context::Context,
        module::Module,
        targets::{InitializationConfig, Target},
        types::BasicMetadataTypeEnum,
        values::{FloatValue, FunctionValue},
        AddressSpace, OptimizationLevel,
    };

    type AlkahestJitFn = unsafe extern "C" fn(*const f64, u64) -> f64;

    pub fn compile_llvm_inner(
        expr: ExprId,
        inputs: &[ExprId],
        pool: &ExprPool,
    ) -> Result<CompiledFn, JitError> {
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| JitError::LlvmInitError(e.to_string()))?;

        // Leak the context to obtain a 'static reference for the execution engine.
        // The Box is reconstructed below and stored in CompiledFn._context so it is
        // freed only after the execution engine drops (field drop order: fn_ptr →
        // execution_engine → _context).
        let context = Box::new(Context::create());
        let ctx: &'static Context = Box::leak(context);

        let module = ctx.create_module("alkahest_jit");
        let builder = ctx.create_builder();

        // Function signature: f64 alkahest_eval(f64* inputs, u64 n)
        let f64_type = ctx.f64_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default()); // opaque pointer (LLVM 15+)
        let i64_type = ctx.i64_type();
        let fn_type = f64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let function = module.add_function("alkahest_eval", fn_type, None);
        let entry = ctx.append_basic_block(function, "entry");
        builder.position_at_end(entry);

        // Map from ExprId to computed LLVM values
        let mut values: HashMap<ExprId, FloatValue<'_>> = HashMap::new();

        // Load input values from array
        let inputs_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        for (i, &var) in inputs.iter().enumerate() {
            let idx = i64_type.const_int(i as u64, false);
            let gep = unsafe {
                builder
                    .build_gep(f64_type, inputs_ptr, &[idx], &format!("in_{i}"))
                    .map_err(|e| JitError::CompilationFailed(e.to_string()))?
            };
            let val = builder
                .build_load(f64_type, gep, &format!("x_{i}"))
                .map_err(|e| JitError::CompilationFailed(e.to_string()))?
                .into_float_value();
            values.insert(var, val);
        }

        // Topological sort and codegen
        let topo = topo_sort_jit(expr, pool);
        for &node in &topo {
            if values.contains_key(&node) {
                continue;
            }
            let val = codegen_node(node, pool, &values, &builder, &module, ctx, function)?;
            values.insert(node, val);
        }

        let result = *values
            .get(&expr)
            .ok_or_else(|| JitError::CompilationFailed("root node not computed".to_string()))?;
        builder
            .build_return(Some(&result))
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

        // Verify
        if module.verify().is_err() {
            return Err(JitError::CompilationFailed(
                "LLVM module verification failed".to_string(),
            ));
        }

        // Create execution engine
        let ee = module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

        let fn_ptr: AlkahestJitFn = unsafe {
            ee.get_function("alkahest_eval")
                .map_err(|e| JitError::CompilationFailed(e.to_string()))?
                .as_raw()
        };

        // SAFETY: fn_ptr is valid as long as execution_engine (and the context it
        // references) are alive.  Both are stored in CompiledFn and drop in the
        // order fn_ptr → execution_engine → _context, satisfying the constraint.
        Ok(CompiledFn {
            fn_ptr,
            execution_engine: ee,
            _context: unsafe { Box::from_raw(ctx as *const Context as *mut Context) },
            n_inputs: inputs.len(),
        })
    }

    fn topo_sort_jit(root: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();
        dfs_jit(root, pool, &mut visited, &mut order);
        order
    }

    fn dfs_jit(
        node: ExprId,
        pool: &ExprPool,
        visited: &mut std::collections::HashSet<ExprId>,
        order: &mut Vec<ExprId>,
    ) {
        if !visited.insert(node) {
            return;
        }
        let children = pool.with(node, |d| match d {
            ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => a.clone(),
            ExprData::Pow { base, exp } => vec![*base, *exp],
            ExprData::BigO(inner) => vec![inner],
            _ => vec![],
        });
        for c in children {
            dfs_jit(c, pool, visited, order);
        }
        order.push(node);
    }

    fn codegen_node<'ctx>(
        node: ExprId,
        pool: &ExprPool,
        values: &HashMap<ExprId, FloatValue<'ctx>>,
        builder: &Builder<'ctx>,
        module: &Module<'ctx>,
        ctx: &'ctx Context,
        _function: FunctionValue<'ctx>,
    ) -> Result<FloatValue<'ctx>, JitError> {
        let f64_type = ctx.f64_type();
        match pool.get(node) {
            ExprData::Integer(n) => Ok(f64_type.const_float(n.0.to_f64())),
            ExprData::Rational(r) => {
                let (n, d) = r.0.clone().into_numer_denom();
                Ok(f64_type.const_float(n.to_f64() / d.to_f64()))
            }
            ExprData::Float(f) => Ok(f64_type.const_float(f.inner.to_f64())),
            ExprData::Symbol { name, .. } => Err(JitError::UnsupportedNode(format!(
                "unbound symbol '{name}'"
            ))),
            ExprData::Add(args) => {
                let mut acc = f64_type.const_float(0.0);
                for &a in &args {
                    let v = *values
                        .get(&a)
                        .ok_or_else(|| JitError::CompilationFailed("missing child".to_string()))?;
                    acc = builder
                        .build_float_add(acc, v, "fadd")
                        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
                }
                Ok(acc)
            }
            ExprData::Mul(args) => {
                let mut acc = f64_type.const_float(1.0);
                for &a in &args {
                    let v = *values
                        .get(&a)
                        .ok_or_else(|| JitError::CompilationFailed("missing child".to_string()))?;
                    acc = builder
                        .build_float_mul(acc, v, "fmul")
                        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
                }
                Ok(acc)
            }
            ExprData::Pow { base, exp } => {
                let b = *values
                    .get(&base)
                    .ok_or_else(|| JitError::CompilationFailed("missing base".to_string()))?;
                let e = *values
                    .get(&exp)
                    .ok_or_else(|| JitError::CompilationFailed("missing exp".to_string()))?;
                let pow_fn = get_intrinsic(
                    module,
                    ctx,
                    "llvm.pow.f64",
                    &[f64_type.into(), f64_type.into()],
                    f64_type,
                );
                let result = builder
                    .build_call(pow_fn, &[b.into(), e.into()], "fpow")
                    .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
                Ok(result
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_float_value())
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                let a = *values
                    .get(&args[0])
                    .ok_or_else(|| JitError::CompilationFailed("missing arg".to_string()))?;
                let intrinsic_name = match name.as_str() {
                    "sin" => "llvm.sin.f64",
                    "cos" => "llvm.cos.f64",
                    "exp" => "llvm.exp.f64",
                    "log" => "llvm.log.f64",
                    "sqrt" => "llvm.sqrt.f64",
                    "abs" => "llvm.fabs.f64",
                    other => return Err(JitError::UnsupportedNode(format!("function '{other}'"))),
                };
                let f = get_intrinsic(module, ctx, intrinsic_name, &[f64_type.into()], f64_type);
                let result = builder
                    .build_call(f, &[a.into()], "fcall")
                    .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
                Ok(result
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_float_value())
            }
            other => Err(JitError::UnsupportedNode(format!("{other:?}"))),
        }
    }

    fn get_intrinsic<'ctx>(
        module: &Module<'ctx>,
        _ctx: &'ctx Context,
        name: &str,
        param_types: &[BasicMetadataTypeEnum<'ctx>],
        return_type: inkwell::types::FloatType<'ctx>,
    ) -> FunctionValue<'ctx> {
        if let Some(f) = module.get_function(name) {
            return f;
        }
        let fn_type = return_type.fn_type(param_types, false);
        module.add_function(name, fn_type, None)
    }
}

#[cfg(feature = "jit")]
fn compile_llvm(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> Result<CompiledFn, JitError> {
    llvm_backend::compile_llvm_inner(expr, inputs, pool)
}

// ---------------------------------------------------------------------------
// Tests (interpreter path — always run)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn interp_constant() {
        let pool = p();
        let five = pool.integer(5_i32);
        let f = compile(five, &[], &pool).unwrap();
        assert!((f.call(&[]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn interp_identity() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = compile(x, &[x], &pool).unwrap();
        assert!((f.call(&[2.5_f64]) - 2.5_f64).abs() < 1e-10);
    }

    #[test]
    fn interp_add() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![x, y]);
        let f = compile(expr, &[x, y], &pool).unwrap();
        assert!((f.call(&[2.0, 3.0]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn interp_polynomial() {
        // f(x) = x² + 2x + 1  = (x+1)²
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let one = pool.integer(1_i32);
        let expr = pool.add(vec![x2, two_x, one]);
        let f = compile(expr, &[x], &pool).unwrap();
        // f(3) = 9 + 6 + 1 = 16
        assert!((f.call(&[3.0]) - 16.0).abs() < 1e-10);
    }

    #[test]
    fn interp_rational() {
        let pool = p();
        let half = pool.rational(1, 2);
        let f = compile(half, &[], &pool).unwrap();
        assert!((f.call(&[]) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn interp_sin() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let f = compile(sin_x, &[x], &pool).unwrap();
        let pi_2 = std::f64::consts::PI / 2.0;
        assert!((f.call(&[pi_2]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn interp_pow_non_integer() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let half = pool.float(0.5, 53);
        let expr = pool.pow(x, half);
        let f = compile(expr, &[x], &pool).unwrap();
        assert!((f.call(&[4.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn interp_multivariate() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let y2 = pool.pow(y, pool.integer(2_i32));
        let expr = pool.add(vec![x2, y2]);
        let f = compile(expr, &[x, y], &pool).unwrap();
        // Pythagorean triple: f(3,4) = 25
        assert!((f.call(&[3.0, 4.0]) - 25.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "expected 1 inputs")]
    fn interp_wrong_n_inputs_panics() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = compile(x, &[x], &pool).unwrap();
        f.call(&[]);
    }
}
