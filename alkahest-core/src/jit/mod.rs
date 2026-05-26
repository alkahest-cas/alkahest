//! Tiered JIT compilation for symbolic expressions.
//!
//! Compilation tier is chosen from expression size and an optional
//! [`CompileConfig::expected_evals`] hint (interp → Cranelift → LLVM):
//!
//! ```text
//! ExprId + CompileConfig
//!   │
//!   ├─ interpreter — small DAG + few planned evals (zero compile latency)
//!   │
//!   ├─ cranelift (--features cranelift) — pure Rust, fast compile
//!   │
//!   ├─ LLVM (--features jit) — best throughput for large batches
//!   │
//!   └─ interpreter fallback — when native JIT is unavailable or fails
//! ```
//!
//! Native backends also emit a **bulk** entry point
//! `fn(*const f64, n_vars, *mut f64, n_points)` for column-major batch evaluation
//! (see [`CompiledFn::call_bulk`]).
//!
//! The public API (`compile`, `eval_interp`, `CompiledFn`) is the same
//! regardless of which features are compiled in.
//!
//! # Example
//!
//! ```no_run
//! use alkahest_cas::kernel::{Domain, ExprPool};
//! use alkahest_cas::jit::compile;
//!
//! let pool = ExprPool::new();
//! let x = pool.symbol("x", Domain::Real);
//! let y = pool.symbol("y", Domain::Real);
//! let expr = pool.add(vec![
//!     pool.mul(vec![x, x]),
//!     pool.mul(vec![y, y]),
//! ]);
//! let f = compile(expr, &[x, y], &pool).unwrap();
//! let result = f.call(&[3.0, 4.0]); // 9 + 16 = 25
//! assert!((result - 25.0).abs() < 1e-10);
//! ```

use crate::kernel::{ExprData, ExprId, ExprPool};
use std::collections::HashMap;
use std::fmt;

#[cfg(feature = "cuda")]
pub mod nvptx;
#[cfg(feature = "cuda")]
pub use nvptx::{compile_cuda, CudaCompiledFn, CudaError};

#[cfg(feature = "cranelift")]
mod cranelift_backend;

pub mod cache;
pub use cache::CompileCache;

// ---------------------------------------------------------------------------
// JIT function signatures and compile configuration
// ---------------------------------------------------------------------------

/// Scalar JIT entry: `inputs` points to `n_inputs` consecutive `f64` values.
pub(super) type JitScalarFn = unsafe extern "C" fn(*const f64, u64) -> f64;

/// Bulk JIT entry: column-major `inputs` (`n_vars * n_points` values), `outputs` length `n_points`.
pub(super) type JitBulkFn = unsafe extern "C" fn(*const f64, u64, *mut f64, u64);

/// Which backend to use for a compilation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompileTier {
    Interpreter,
    #[cfg(feature = "cranelift")]
    Cranelift,
    #[cfg(feature = "jit")]
    Llvm,
}

/// Hints for [`compile_with`] tier selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CompileConfig {
    /// Planned number of evaluations (e.g. batch length). When `None`, only
    /// expression size is used and LLVM is not preferred unless forced.
    pub expected_evals: Option<u64>,
    /// Override automatic tier selection.
    pub force_tier: Option<CompileTier>,
}

impl CompileConfig {
    /// Config for a batch of `n_points` evaluations (enables LLVM on large N).
    pub const fn for_batch(n_points: u64) -> Self {
        Self {
            expected_evals: Some(n_points),
            force_tier: None,
        }
    }
}

/// Max DAG nodes for the interpreter fast path (with few planned evals).
pub const INTERP_MAX_NODES: usize = 64;
/// Max planned evaluations for the interpreter fast path (with a small DAG).
pub const INTERP_MAX_EXPECTED_EVALS: u64 = 16;
/// At or above this many planned evaluations, prefer LLVM over Cranelift.
pub const LLVM_MIN_EXPECTED_EVALS: u64 = 4096;

// ---------------------------------------------------------------------------
// Error type (always compiled)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum JitError {
    UnsupportedNode(String),
    CompilationFailed(String),
    LlvmInitError(String),
    /// The JIT backend is not compiled into this build.
    ///
    /// Returned when `compile_jit_only` is called on a build that was not
    /// compiled with `--features jit`.  Use `eval_expr` for interpreted
    /// evaluation or rebuild with `--features jit` and LLVM 15 installed.
    NotAvailable(String),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::UnsupportedNode(s) => write!(f, "unsupported expression node: {s}"),
            JitError::CompilationFailed(s) => write!(f, "JIT compilation failed: {s}"),
            JitError::LlvmInitError(s) => write!(f, "LLVM/Cranelift init error: {s}"),
            JitError::NotAvailable(s) => write!(f, "JIT not available: {s}"),
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
            JitError::NotAvailable(_) => "E-JIT-004",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            JitError::UnsupportedNode(_) => Some(
                "use eval_expr (interpreted) or simplify the expression to remove unsupported nodes",
            ),
            JitError::CompilationFailed(_) => Some(
                "check LLVM/Cranelift installation; run with RUST_LOG=debug for details",
            ),
            JitError::LlvmInitError(_) => Some(
                "rebuild with --features cranelift (pure Rust) or ensure LLVM 15 is installed and \
                 LLVM_SYS_150_PREFIX is set correctly",
            ),
            JitError::NotAvailable(_) => Some(
                "rebuild with --features cranelift (no system deps) or --features jit (LLVM 15), \
                 or use eval_expr() for the interpreter path",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// CompiledFn — wraps a callable function from any backend
// ---------------------------------------------------------------------------

/// Inner representation covering all three compilation tiers.
enum CompiledFnInner {
    #[cfg(feature = "jit")]
    Llvm {
        fn_ptr: JitScalarFn,
        bulk_fn: Option<JitBulkFn>,
        // execution_engine must be declared before _context so it drops first;
        // the context must outlive the execution engine.
        #[allow(dead_code)]
        execution_engine: inkwell::execution_engine::ExecutionEngine<'static>,
        _context: Box<inkwell::context::Context>,
    },

    #[cfg(feature = "cranelift")]
    Cranelift {
        fn_ptr: JitScalarFn,
        bulk_fn: Option<JitBulkFn>,
        /// JITModule owns the code pages; must outlive `fn_ptr`.
        _module: Box<cranelift_jit::JITModule>,
    },

    Interpreter(Box<dyn Fn(&[f64]) -> f64 + Send + Sync>),
}

/// A compiled function that evaluates a symbolic expression numerically.
///
/// The function accepts a slice of `f64` inputs corresponding to the variables
/// given to `compile`.  It may be backed by Cranelift JIT, LLVM JIT, or the
/// tree-walking interpreter, depending on which features are compiled in.
pub struct CompiledFn {
    inner: CompiledFnInner,
    /// Number of inputs expected by [`call`](CompiledFn::call).
    pub n_inputs: usize,
    /// Backend used for this function (for diagnostics).
    pub tier: CompileTier,
}

impl CompiledFn {
    /// Evaluate with the given inputs.  Panics if `inputs.len() != n_inputs`.
    pub fn call(&self, inputs: &[f64]) -> f64 {
        assert_eq!(
            inputs.len(),
            self.n_inputs,
            "expected {} inputs, got {}",
            self.n_inputs,
            inputs.len()
        );
        match &self.inner {
            #[cfg(feature = "jit")]
            CompiledFnInner::Llvm { fn_ptr, .. } => unsafe {
                fn_ptr(inputs.as_ptr(), inputs.len() as u64)
            },
            #[cfg(feature = "cranelift")]
            CompiledFnInner::Cranelift { fn_ptr, .. } => unsafe {
                fn_ptr(inputs.as_ptr(), inputs.len() as u64)
            },
            CompiledFnInner::Interpreter(f) => f(inputs),
        }
    }

    /// Bulk-evaluate over N points using the native bulk JIT entry when available.
    ///
    /// `inputs_flat` is **column-major** (var-major), length `n_inputs * n_points`:
    /// variable `i` occupies `inputs_flat[i * n_points .. (i + 1) * n_points]`.
    /// `output` has length `n_points`.
    pub fn call_bulk(&self, inputs_flat: &[f64], output: &mut [f64]) {
        let n_points = output.len();
        assert_eq!(
            inputs_flat.len(),
            self.n_inputs * n_points,
            "inputs_flat length {} != n_inputs({}) * n_points({})",
            inputs_flat.len(),
            self.n_inputs,
            n_points
        );
        #[cfg(feature = "jit")]
        if let CompiledFnInner::Llvm {
            bulk_fn: Some(bulk_fn),
            ..
        } = &self.inner
        {
            return unsafe {
                bulk_fn(
                    inputs_flat.as_ptr(),
                    self.n_inputs as u64,
                    output.as_mut_ptr(),
                    n_points as u64,
                )
            };
        }
        #[cfg(feature = "cranelift")]
        if let CompiledFnInner::Cranelift {
            bulk_fn: Some(bulk_fn),
            ..
        } = &self.inner
        {
            return unsafe {
                bulk_fn(
                    inputs_flat.as_ptr(),
                    self.n_inputs as u64,
                    output.as_mut_ptr(),
                    n_points as u64,
                )
            };
        }
        let mut point = vec![0.0f64; self.n_inputs];
        for j in 0..n_points {
            for (i, slot) in point.iter_mut().enumerate() {
                *slot = inputs_flat[i * n_points + j];
            }
            output[j] = self.call(&point);
        }
    }

    /// Batch-evaluate over N points (sequential).
    ///
    /// `inputs[i]` contains the values of variable `i` for all N points.
    /// All slices must have the same length N.  `output` must also have length N.
    ///
    /// This is the hot path for NumPy/JAX array evaluation.
    /// For multi-core throughput see [`call_batch_par`](Self::call_batch_par).
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
        if self.n_inputs == 0 {
            return;
        }
        // Column-major flat layout matches `call_bulk` / Python `call_batch_raw`.
        let mut flat = Vec::with_capacity(self.n_inputs * n);
        for col in inputs {
            flat.extend_from_slice(col);
        }
        self.call_bulk(&flat, output);
    }

    /// Batch-evaluate over N points **in parallel** using Rayon.
    ///
    /// Identical semantics to [`call_batch`](Self::call_batch) but distributes
    /// points across all available CPU cores.  Each point is independent — no
    /// synchronisation required — so the speedup scales linearly with core count.
    ///
    /// Available only when compiled with `--features parallel`.
    ///
    /// # Performance notes
    ///
    /// - For very small N (< ~1 000 points) the thread-scheduling overhead may
    ///   exceed the computation time; `call_batch` is faster in that regime.
    /// - The function pointer is shared (`&self`) across threads safely via
    ///   `CompiledFn: Sync`.
    #[cfg(feature = "parallel")]
    pub fn call_batch_par(&self, inputs: &[&[f64]], output: &mut [f64]) {
        use rayon::prelude::*;

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

        // Each point `j` is independent — reuse scalar `call` (bulk JIT is sequential).
        output.par_iter_mut().enumerate().for_each(|(j, out)| {
            let point: Vec<f64> = inputs.iter().map(|col| col[j]).collect();
            *out = self.call(&point);
        });
    }

    /// Backend tier used to implement this function.
    pub fn compile_tier(&self) -> CompileTier {
        self.tier
    }
}

// SAFETY: `CompiledFn` holds one of:
// - An interpreter closure that is already `Send + Sync`.
// - A Cranelift `JITModule` (its code pages are read-only after
//   `finalize_definitions`) plus an immutable function pointer.
// - An LLVM `ExecutionEngine` and `Context` whose code pages are similarly
//   read-only after JIT finalization.
//
// All three variants are safe to move across threads (`Send`) and to call
// concurrently from multiple threads (`Sync`) because:
//   1. The function pointer targets immutable executable pages.
//   2. `call()` / `call_batch()` take `&self` (no mutation) and only read
//      through the pointer.
//   3. The backing objects (JITModule, ExecutionEngine) are used solely to
//      keep the code pages alive and are never accessed after construction.
unsafe impl Send for CompiledFn {}
unsafe impl Sync for CompiledFn {}

// ---------------------------------------------------------------------------
// compile — main entry point (tiered dispatch)
// ---------------------------------------------------------------------------

/// Count distinct nodes in the subgraph rooted at `root`.
pub fn expr_subgraph_size(root: ExprId, pool: &ExprPool) -> usize {
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![root];
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
            ExprData::BigO(inner) => stack.push(*inner),
            _ => {}
        }
    }
    visited.len()
}

/// Select compilation tier from expression size and [`CompileConfig`].
pub fn select_compile_tier(expr: ExprId, pool: &ExprPool, config: &CompileConfig) -> CompileTier {
    if let Some(tier) = config.force_tier {
        return tier;
    }
    let nodes = expr_subgraph_size(expr, pool);
    let evals = config.expected_evals.unwrap_or(0);

    if nodes <= INTERP_MAX_NODES && evals <= INTERP_MAX_EXPECTED_EVALS {
        return CompileTier::Interpreter;
    }

    #[cfg(feature = "jit")]
    if evals >= LLVM_MIN_EXPECTED_EVALS {
        return CompileTier::Llvm;
    }

    #[cfg(feature = "cranelift")]
    {
        return CompileTier::Cranelift;
    }

    #[cfg(feature = "jit")]
    {
        return CompileTier::Llvm;
    }

    #[allow(unreachable_code)]
    CompileTier::Interpreter
}

fn compile_for_tier(
    tier: CompileTier,
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CompiledFn, JitError> {
    match tier {
        CompileTier::Interpreter => compile_interpreter(expr, inputs, pool),
        #[cfg(feature = "cranelift")]
        CompileTier::Cranelift => cranelift_backend::compile_cranelift(expr, inputs, pool),
        #[cfg(feature = "jit")]
        CompileTier::Llvm => compile_llvm(expr, inputs, pool),
    }
}

fn compile_with_fallbacks(
    tier: CompileTier,
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CompiledFn, JitError> {
    match compile_for_tier(tier, expr, inputs, pool) {
        Ok(f) => return Ok(f),
        Err(e) => match tier {
            CompileTier::Interpreter => return Err(e),
            #[cfg(feature = "jit")]
            CompileTier::Llvm => {
                #[cfg(feature = "cranelift")]
                if let Ok(f) = cranelift_backend::compile_cranelift(expr, inputs, pool) {
                    return Ok(f);
                }
                return compile_interpreter(expr, inputs, pool);
            }
            #[cfg(feature = "cranelift")]
            CompileTier::Cranelift => {
                #[cfg(feature = "jit")]
                if let Ok(f) = compile_llvm(expr, inputs, pool) {
                    return Ok(f);
                }
                return compile_interpreter(expr, inputs, pool);
            }
        },
    }
}

/// Compile `expr` with tier selection driven by [`CompileConfig`].
pub fn compile_with(
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
    config: CompileConfig,
) -> Result<CompiledFn, JitError> {
    let tier = select_compile_tier(expr, pool, &config);
    compile_with_fallbacks(tier, expr, inputs, pool)
}

/// Compile `expr` to a native or interpreted function (default [`CompileConfig`]).
///
/// Small expressions with no large batch hint use the interpreter even when JIT
/// features are enabled.  Use [`compile_with`] with [`CompileConfig::for_batch`]
/// before large numerical sweeps.
pub fn compile(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> Result<CompiledFn, JitError> {
    compile_with(expr, inputs, pool, CompileConfig::default())
}

/// Returns `true` if any native JIT backend (Cranelift or LLVM) is available.
pub const fn jit_available() -> bool {
    cfg!(feature = "cranelift") || cfg!(feature = "jit")
}

/// Returns `true` if the LLVM JIT backend is available.
pub const fn llvm_jit_available() -> bool {
    cfg!(feature = "jit")
}

/// Returns `true` if the Cranelift JIT backend is available.
pub const fn cranelift_jit_available() -> bool {
    cfg!(feature = "cranelift")
}

/// Compile `expr` refusing to fall back to the interpreter.
///
/// Returns `Err(JitError::NotAvailable)` when no native JIT backend is
/// compiled in.  Use `compile` for the version that silently falls back.
pub fn compile_jit_only(
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CompiledFn, JitError> {
    #[cfg(feature = "cranelift")]
    match cranelift_backend::compile_cranelift(expr, inputs, pool) {
        Ok(f) => return Ok(f),
        Err(e) => return Err(e),
    }

    #[cfg(all(feature = "jit", not(feature = "cranelift")))]
    return compile_llvm(expr, inputs, pool);

    #[cfg(not(any(feature = "jit", feature = "cranelift")))]
    {
        let _ = (expr, inputs, pool);
        Err(JitError::NotAvailable(
            "this build was compiled without --features cranelift or --features jit; \
             use eval_expr() for interpreted evaluation, or rebuild with a JIT feature."
                .to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Interpreter fast path — always available
// ---------------------------------------------------------------------------

/// Tree-walking interpreter for evaluating symbolic expressions numerically.
///
/// `env` maps symbolic variable `ExprId`s to their `f64` values.
/// Shared subexpressions (same `ExprId`) are evaluated once per call via an
/// internal memo table.
pub fn eval_interp(expr: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<f64> {
    let mut memo: HashMap<ExprId, f64> = HashMap::new();
    eval_interp_inner(expr, env, pool, &mut memo)
}

fn eval_interp_inner(
    expr: ExprId,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, f64>,
) -> Option<f64> {
    if let Some(&cached) = memo.get(&expr) {
        return Some(cached);
    }
    let val = match pool.get(expr) {
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
                sum += eval_interp_inner(a, env, pool, memo)?;
            }
            Some(sum)
        }
        ExprData::Mul(args) => {
            let mut prod = 1.0f64;
            for &a in &args {
                prod *= eval_interp_inner(a, env, pool, memo)?;
            }
            Some(prod)
        }
        ExprData::Pow { base, exp } => {
            let b = eval_interp_inner(base, env, pool, memo)?;
            let e = eval_interp_inner(exp, env, pool, memo)?;
            Some(b.powf(e))
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let x = eval_interp_inner(args[0], env, pool, memo)?;
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
    };
    if let Some(v) = val {
        memo.insert(expr, v);
    }
    val
}

// ---------------------------------------------------------------------------
// Interpreter compile path — always available (used as fallback by `compile`)
// ---------------------------------------------------------------------------

fn compile_interpreter(
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CompiledFn, JitError> {
    let inputs_vec = inputs.to_vec();
    let n = inputs_vec.len();
    let snapshot = snapshot_expr(expr, pool);

    let interp = move |vals: &[f64]| -> f64 {
        let mut env: HashMap<ExprId, f64> = HashMap::new();
        for (&var, &val) in inputs_vec.iter().zip(vals.iter()) {
            env.insert(var, val);
        }
        let mut memo: HashMap<ExprId, f64> = HashMap::new();
        eval_interp_snap(expr, &env, &snapshot, &mut memo).unwrap_or(f64::NAN)
    };

    Ok(CompiledFn {
        inner: CompiledFnInner::Interpreter(Box::new(interp)),
        n_inputs: n,
        tier: CompileTier::Interpreter,
    })
}

// ---------------------------------------------------------------------------
// Snapshot-based interpreter (captures expression DAG without pool reference)
// ---------------------------------------------------------------------------

/// A self-contained snapshot of an expression subgraph.
#[derive(Clone)]
pub struct ExprSnapshot {
    nodes: HashMap<ExprId, ExprData>,
}

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

fn eval_interp_snap(
    expr: ExprId,
    env: &HashMap<ExprId, f64>,
    snap: &ExprSnapshot,
    memo: &mut HashMap<ExprId, f64>,
) -> Option<f64> {
    if let Some(&cached) = memo.get(&expr) {
        return Some(cached);
    }
    let val = match snap.nodes.get(&expr)? {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => {
            let (n, d) = r.0.clone().into_numer_denom();
            Some(n.to_f64() / d.to_f64())
        }
        ExprData::Float(f) => Some(f.inner.to_f64()),
        ExprData::Symbol { .. } => env.get(&expr).copied(),
        ExprData::Add(args) => {
            let args = args.clone();
            let mut s = 0.0f64;
            for a in args {
                s += eval_interp_snap(a, env, snap, memo)?;
            }
            Some(s)
        }
        ExprData::Mul(args) => {
            let args = args.clone();
            let mut p = 1.0f64;
            for a in args {
                p *= eval_interp_snap(a, env, snap, memo)?;
            }
            Some(p)
        }
        ExprData::Pow { base, exp } => {
            let (b, e) = (*base, *exp);
            Some(eval_interp_snap(b, env, snap, memo)?.powf(eval_interp_snap(e, env, snap, memo)?))
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let a = args[0];
            let x = eval_interp_snap(a, env, snap, memo)?;
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
    };
    if let Some(v) = val {
        memo.insert(expr, v);
    }
    val
}

// ---------------------------------------------------------------------------
// Shared topological sort — used by both Cranelift and LLVM backends
// ---------------------------------------------------------------------------

/// Returns nodes of the subgraph rooted at `root` in topological (post) order:
/// every node appears after all of its children.
// Used by cranelift_backend and llvm_backend; appears "unused" when both
// optional features are disabled.
#[cfg_attr(not(any(feature = "jit", feature = "cranelift")), allow(dead_code))]
pub(super) fn topo_sort(root: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut visited = std::collections::HashSet::new();
    let mut order = Vec::new();
    topo_dfs(root, pool, &mut visited, &mut order);
    order
}

#[cfg_attr(not(any(feature = "jit", feature = "cranelift")), allow(dead_code))]
fn topo_dfs(
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
        ExprData::BigO(inner) => vec![*inner],
        _ => vec![],
    });
    for c in children {
        topo_dfs(c, pool, visited, order);
    }
    order.push(node);
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

    use inkwell::values::IntValue;
    use inkwell::IntPredicate;

    fn load_scalar_inputs<'ctx>(
        builder: &Builder<'ctx>,
        f64_type: inkwell::types::FloatType<'ctx>,
        i64_type: inkwell::types::IntType<'ctx>,
        inputs_ptr: inkwell::values::PointerValue<'ctx>,
        inputs: &[ExprId],
        values: &mut HashMap<ExprId, FloatValue<'ctx>>,
    ) -> Result<(), JitError> {
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
        Ok(())
    }

    fn load_batch_inputs<'ctx>(
        builder: &Builder<'ctx>,
        f64_type: inkwell::types::FloatType<'ctx>,
        i64_type: inkwell::types::IntType<'ctx>,
        inputs_ptr: inkwell::values::PointerValue<'ctx>,
        inputs: &[ExprId],
        point_idx: IntValue<'ctx>,
        n_points: IntValue<'ctx>,
        values: &mut HashMap<ExprId, FloatValue<'ctx>>,
    ) -> Result<(), JitError> {
        let n_vars = i64_type.const_int(inputs.len() as u64, false);
        for (i, &var) in inputs.iter().enumerate() {
            let var_i = i64_type.const_int(i as u64, false);
            let elem_idx = builder
                .build_int_add(
                    builder
                        .build_int_mul(var_i, n_points, "var_stride")
                        .map_err(|e| JitError::CompilationFailed(e.to_string()))?,
                    point_idx,
                    "elem_idx",
                )
                .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
            let gep = unsafe {
                builder
                    .build_gep(f64_type, inputs_ptr, &[elem_idx], &format!("bulk_in_{i}"))
                    .map_err(|e| JitError::CompilationFailed(e.to_string()))?
            };
            let val = builder
                .build_load(f64_type, gep, &format!("bulk_x_{i}"))
                .map_err(|e| JitError::CompilationFailed(e.to_string()))?
                .into_float_value();
            values.insert(var, val);
        }
        Ok(())
    }

    fn emit_expr_values<'ctx>(
        expr: ExprId,
        pool: &ExprPool,
        inputs: &[ExprId],
        builder: &Builder<'ctx>,
        module: &Module<'ctx>,
        ctx: &'ctx Context,
        function: FunctionValue<'ctx>,
        f64_type: inkwell::types::FloatType<'ctx>,
        i64_type: inkwell::types::IntType<'ctx>,
        inputs_ptr: inkwell::values::PointerValue<'ctx>,
        batch_point: Option<IntValue<'ctx>>,
        n_points: Option<IntValue<'ctx>>,
    ) -> Result<FloatValue<'ctx>, JitError> {
        let mut values: HashMap<ExprId, FloatValue<'ctx>> = HashMap::new();
        match batch_point {
            None => {
                load_scalar_inputs(builder, f64_type, i64_type, inputs_ptr, inputs, &mut values)?
            }
            Some(idx) => load_batch_inputs(
                builder,
                f64_type,
                i64_type,
                inputs_ptr,
                inputs,
                idx,
                n_points.expect("n_points required for batch load"),
                &mut values,
            )?,
        }
        let topo = topo_sort(expr, pool);
        for &node in &topo {
            if values.contains_key(&node) {
                continue;
            }
            let val = codegen_node(node, pool, &values, builder, module, ctx, function)?;
            values.insert(node, val);
        }
        values
            .get(&expr)
            .copied()
            .ok_or_else(|| JitError::CompilationFailed("root node not computed".to_string()))
    }

    pub fn compile_llvm_inner(
        expr: ExprId,
        inputs: &[ExprId],
        pool: &ExprPool,
    ) -> Result<CompiledFn, JitError> {
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| JitError::LlvmInitError(e.to_string()))?;

        let context = Box::new(Context::create());
        let ctx: &'static Context = Box::leak(context);

        let module = ctx.create_module("alkahest_jit");
        let builder = ctx.create_builder();

        let f64_type = ctx.f64_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i64_type = ctx.i64_type();

        // Scalar: f64 alkahest_eval(f64* inputs, u64 n)
        let scalar_fn_type = f64_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
        let scalar_fn = module.add_function("alkahest_eval", scalar_fn_type, None);
        let scalar_entry = ctx.append_basic_block(scalar_fn, "entry");
        builder.position_at_end(scalar_entry);
        let scalar_inputs_ptr = scalar_fn
            .get_nth_param(0)
            .ok_or_else(|| {
                JitError::CompilationFailed("failed to get JIT inputs parameter".to_string())
            })?
            .into_pointer_value();
        let scalar_result = emit_expr_values(
            expr,
            pool,
            inputs,
            &builder,
            &module,
            ctx,
            scalar_fn,
            f64_type,
            i64_type,
            scalar_inputs_ptr,
            None,
            None,
        )?;
        builder
            .build_return(Some(&scalar_result))
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

        // Bulk: void alkahest_eval_bulk(f64* inputs, u64 n_vars, f64* outputs, u64 n_points)
        let void_type = ctx.void_type();
        let bulk_fn_type = void_type.fn_type(
            &[
                ptr_type.into(),
                i64_type.into(),
                ptr_type.into(),
                i64_type.into(),
            ],
            false,
        );
        let bulk_fn = module.add_function("alkahest_eval_bulk", bulk_fn_type, None);
        let bulk_entry = ctx.append_basic_block(bulk_fn, "entry");
        let bulk_loop_hdr = ctx.append_basic_block(bulk_fn, "loop_hdr");
        let bulk_loop_body = ctx.append_basic_block(bulk_fn, "loop_body");
        let bulk_exit = ctx.append_basic_block(bulk_fn, "loop_exit");

        builder.position_at_end(bulk_entry);
        let bulk_inputs_ptr = bulk_fn.get_nth_param(0).unwrap().into_pointer_value();
        let bulk_n_points = bulk_fn.get_nth_param(3).unwrap().into_int_value();
        let bulk_outputs_ptr = bulk_fn.get_nth_param(2).unwrap().into_pointer_value();
        let zero = i64_type.const_int(0, false);
        builder
            .build_unconditional_branch(bulk_loop_hdr)
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

        builder.position_at_end(bulk_loop_hdr);
        let loop_idx = builder
            .build_phi(i64_type, "i")
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
        loop_idx.add_incoming(&[(&zero, bulk_entry)]);
        let cur_idx = loop_idx.as_basic_value().into_int_value();
        let done = builder
            .build_int_compare(IntPredicate::UGE, cur_idx, bulk_n_points, "done")
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
        builder
            .build_conditional_branch(done, bulk_exit, bulk_loop_body)
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

        builder.position_at_end(bulk_loop_body);
        let body_result = emit_expr_values(
            expr,
            pool,
            inputs,
            &builder,
            &module,
            ctx,
            bulk_fn,
            f64_type,
            i64_type,
            bulk_inputs_ptr,
            Some(cur_idx),
            Some(bulk_n_points),
        )?;
        let out_gep = unsafe {
            builder
                .build_gep(f64_type, bulk_outputs_ptr, &[cur_idx], "out_gep")
                .map_err(|e| JitError::CompilationFailed(e.to_string()))?
        };
        builder
            .build_store(out_gep, body_result)
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
        let one = i64_type.const_int(1, false);
        let next_idx = builder
            .build_int_add(cur_idx, one, "next_i")
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
        loop_idx.add_incoming(&[(&next_idx, bulk_loop_body)]);
        builder
            .build_unconditional_branch(bulk_loop_hdr)
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

        builder.position_at_end(bulk_exit);
        builder
            .build_return(None)
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

        if module.verify().is_err() {
            return Err(JitError::CompilationFailed(
                "LLVM module verification failed".to_string(),
            ));
        }

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

        let fn_ptr: super::JitScalarFn = unsafe {
            ee.get_function("alkahest_eval")
                .map_err(|e| JitError::CompilationFailed(e.to_string()))?
                .as_raw()
        };
        let bulk_fn_ptr: super::JitBulkFn = unsafe {
            ee.get_function("alkahest_eval_bulk")
                .map_err(|e| JitError::CompilationFailed(e.to_string()))?
                .as_raw()
        };

        Ok(CompiledFn {
            inner: CompiledFnInner::Llvm {
                fn_ptr,
                bulk_fn: Some(bulk_fn_ptr),
                execution_engine: ee,
                _context: unsafe { Box::from_raw(ctx as *const Context as *mut Context) },
            },
            n_inputs: inputs.len(),
            tier: super::CompileTier::Llvm,
        })
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
                    other => {
                        return Err(JitError::UnsupportedNode(format!("function '{other}'")));
                    }
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

    /// Regression: shared DAG node (same ExprId used twice) should evaluate
    /// correctly via the interpreter and not double-count.
    #[test]
    fn eval_interp_dag_shared_subexpr_correct() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // node = x + 1
        let node = pool.add(vec![x, pool.integer(1_i32)]);
        // expr = node * node  (same ExprId, shared)
        let expr = pool.mul(vec![node, node]);

        let mut env = HashMap::new();
        env.insert(x, 4.0);
        // (4+1) * (4+1) = 25
        let val = eval_interp(expr, &env, &pool).unwrap();
        assert!((val - 25.0).abs() < 1e-10);
    }

    /// Deeply shared DAG: 20 levels of squaring produces 21 nodes but
    /// 2^20 tree references — must terminate in O(n) time.
    #[test]
    fn eval_interp_deep_dag_terminates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let mut cur = pool.add(vec![x, pool.integer(1_i32)]);
        for _ in 0..20 {
            cur = pool.mul(vec![cur, cur]); // each step shares `cur`
        }
        let mut env = HashMap::new();
        env.insert(x, 0.0); // (0+1)^(2^20) = 1
        let val = eval_interp(cur, &env, &pool).unwrap();
        assert!((val - 1.0).abs() < 1e-10);
    }

    /// `call_batch_par` produces the same results as `call_batch` on N points.
    #[cfg(feature = "parallel")]
    #[test]
    fn par_batch_matches_sequential() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // f(x, y) = x² + y²
        let expr = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.pow(y, pool.integer(2_i32)),
        ]);
        let f = compile(expr, &[x, y], &pool).unwrap();

        const N: usize = 10_000;
        let xs: Vec<f64> = (0..N).map(|i| i as f64 * 0.01).collect();
        let ys: Vec<f64> = (0..N).map(|i| i as f64 * 0.02).collect();
        let cols: Vec<&[f64]> = vec![&xs, &ys];

        let mut out_seq = vec![0.0f64; N];
        let mut out_par = vec![0.0f64; N];

        f.call_batch(&cols, &mut out_seq);
        f.call_batch_par(&cols, &mut out_par);

        for (a, b) in out_seq.iter().zip(out_par.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "sequential {a} != parallel {b} at some point"
            );
        }
    }

    #[test]
    fn call_bulk_matches_call_batch() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.pow(y, pool.integer(2_i32)),
        ]);
        let f = compile(expr, &[x, y], &pool).unwrap();
        const N: usize = 128;
        let xs: Vec<f64> = (0..N).map(|i| i as f64 * 0.01).collect();
        let ys: Vec<f64> = (0..N).map(|i| i as f64 * 0.02).collect();
        let cols: Vec<&[f64]> = vec![&xs, &ys];
        let mut out_batch = vec![0.0f64; N];
        let mut out_bulk = vec![0.0f64; N];
        f.call_batch(&cols, &mut out_batch);
        let mut flat = Vec::with_capacity(2 * N);
        for col in &cols {
            flat.extend_from_slice(col);
        }
        f.call_bulk(&flat, &mut out_bulk);
        for (a, b) in out_batch.iter().zip(out_bulk.iter()) {
            assert!((a - b).abs() < 1e-12, "call_batch {a} != call_bulk {b}");
        }
    }

    #[test]
    fn small_expr_defaults_to_interpreter() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![x, pool.integer(1_i32)]);
        let f = compile(expr, &[x], &pool).unwrap();
        assert_eq!(f.compile_tier(), CompileTier::Interpreter);
    }

    #[test]
    fn select_tier_respects_batch_hint() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(2_i32));
        assert_eq!(
            select_compile_tier(expr, &pool, &CompileConfig::default()),
            CompileTier::Interpreter
        );
        #[cfg(feature = "jit")]
        assert_eq!(
            select_compile_tier(expr, &pool, &CompileConfig::for_batch(10_000)),
            CompileTier::Llvm
        );
        #[cfg(all(feature = "cranelift", not(feature = "jit")))]
        assert_eq!(
            select_compile_tier(expr, &pool, &CompileConfig::for_batch(10_000)),
            CompileTier::Cranelift
        );
    }

    /// `call_batch_par` on a single-variable polynomial: f(x) = x³ − 2x + 1.
    #[cfg(feature = "parallel")]
    #[test]
    fn par_batch_polynomial() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // x³ − 2x + 1
        let x3 = pool.pow(x, pool.integer(3_i32));
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let neg_two_x = pool.mul(vec![pool.integer(-1_i32), two_x]);
        let expr = pool.add(vec![x3, neg_two_x, pool.integer(1_i32)]);
        let f = compile(expr, &[x], &pool).unwrap();

        let xs: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let cols: Vec<&[f64]> = vec![&xs];

        let mut out = vec![0.0f64; xs.len()];
        f.call_batch_par(&cols, &mut out);

        // f(x) = x³ − 2x + 1
        let expected: Vec<f64> = xs.iter().map(|&x| x * x * x - 2.0 * x + 1.0).collect();
        for (got, exp) in out.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-10, "got {got}, expected {exp}");
        }
    }
}
