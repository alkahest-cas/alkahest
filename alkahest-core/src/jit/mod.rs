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

use crate::eval::program::NumericProgram;
use crate::eval::IdMap;
use crate::kernel::eval_const::try_predicate_bool_from_expr;
use crate::kernel::expr::PredicateKind;
use crate::kernel::IdSet;
use crate::kernel::{
    integer_is_exact_f64, integer_to_f64, pow_f64, rational_to_f64, ExprData, ExprId, ExprPool,
};
use crate::primitive::PrimitiveRegistry;
use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Shared primitive registry — used to evaluate `ExprData::Func` nodes in the
// tree-walking interpreters below.
// ---------------------------------------------------------------------------

/// Returns a lazily-initialised, process-wide [`PrimitiveRegistry`].
///
/// `eval_interp` and the snapshot interpreter dispatch every `Func { name,
/// args }` node through `registry().numeric_f64(name, &vals)` so that *every*
/// primitive with a `numeric_f64` kernel is automatically evaluable here —
/// new primitives don't need a hand-written match arm in this module.
pub(crate) fn registry() -> &'static PrimitiveRegistry {
    static REGISTRY: OnceLock<PrimitiveRegistry> = OnceLock::new();
    REGISTRY.get_or_init(PrimitiveRegistry::default_registry)
}

#[cfg(feature = "cuda")]
pub mod nvptx;
#[cfg(feature = "cuda")]
pub use nvptx::{compile_cuda, cuda_device_count, CudaCompiledFn, CudaError};

#[cfg(feature = "cranelift")]
mod cranelift_backend;

pub mod cache;
pub(crate) mod complex_fallback;
use complex_fallback::ComplexFallback;

#[cfg(test)]
mod agreement_tests;
#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod block_tests;
pub use cache::CompileCache;

// ---------------------------------------------------------------------------
// JIT function signatures and compile configuration
// ---------------------------------------------------------------------------

/// Scalar JIT entry: `inputs` points to `n_inputs` consecutive `f64` values.
#[cfg(any(feature = "jit", feature = "cranelift"))]
pub(super) type JitScalarFn = unsafe extern "C" fn(*const f64, u64) -> f64;

/// Column JIT entry: `cols[i]` points at input column `i`, `outs[k]` at output
/// column `k`; evaluates points `start..end` (indices into every column).
///
/// One entry point serves a whole batch, a Rayon chunk of one (a sub-range),
/// a single point of a multi-output function (`start = 0, end = 1` over
/// pointers to scalars), and any number of outputs sharing subexpressions.
/// Both native backends emit it (LLVM for one output).
#[cfg(any(feature = "jit", feature = "cranelift"))]
pub(super) type JitColsFn = unsafe extern "C" fn(*const *const f64, *const *mut f64, u64, u64);

/// Interpreter closure type for the Rust fallback backend.
pub(super) type InterpreterFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

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
    /// evaluation or rebuild with `--features jit` and LLVM 21 installed.
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
                "rebuild with --features cranelift (pure Rust) or ensure LLVM 21 is installed and \
                 LLVM_SYS_211_PREFIX is set correctly",
            ),
            JitError::NotAvailable(_) => Some(
                "rebuild with --features cranelift (no system deps) or --features jit (LLVM 21), \
                 or use eval_expr() for the interpreter path",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Unbound symbols — refused at compile time (E-JIT-005)
// ---------------------------------------------------------------------------

/// A compile request whose expression depends on a symbol the input list does
/// not bind (`E-JIT-005`).
///
/// A compiled function takes its values *only* from its inputs. A symbol that
/// is neither an input nor `π` has no value at any point, so the function it
/// would produce is undefined everywhere: the interpreter tier used to return
/// `NaN` for every call, silently, while `eval_expr` on the same expression
/// raised. [`compile`], [`compile_with`] and [`compile_jit_only`] refuse such a
/// request instead.
///
/// # Why this is not a [`JitError`] variant
///
/// `JitError` is a public exhaustive enum, and a new variant is a major semver
/// break. The refusal is therefore returned in band as
/// [`JitError::UnsupportedNode`] naming the symbols, and recorded out of band
/// for [`take_unbound_symbols`] — the pattern
/// [`crate::calculus::take_series_refusal`] uses. Callers that want the check
/// without compiling call [`check_inputs_bind`] directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnboundSymbolsError {
    /// Display names of the unbound symbols, sorted and de-duplicated.
    names: Vec<String>,
    /// Names among `names` that *do* appear in the input list, but as a
    /// different symbol (same name, another domain or commutativity).
    shadowed: Vec<String>,
}

impl UnboundSymbolsError {
    /// Names of the symbols the input list does not bind, sorted.
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

impl fmt::Display for UnboundSymbolsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plural = if self.names.len() == 1 { "" } else { "s" };
        write!(
            f,
            "cannot compile: the expression depends on symbol{plural} {} that the input \
             list does not bind, so the compiled function would have no value at any point",
            self.names.join(", ")
        )?;
        if !self.shadowed.is_empty() {
            write!(
                f,
                " ({} appears in the input list, but as a different symbol: same name, \
                 different domain)",
                self.shadowed.join(", ")
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for UnboundSymbolsError {}

impl crate::errors::AlkahestError for UnboundSymbolsError {
    fn code(&self) -> &'static str {
        "E-JIT-005"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "add every free symbol of the expression to the input list, or substitute a value \
             for each parameter before compiling",
        )
    }
}

thread_local! {
    static LAST_UNBOUND: std::cell::RefCell<Option<UnboundSymbolsError>> =
        const { std::cell::RefCell::new(None) };
}

/// Take the unbound-symbols refusal behind the [`JitError`] that just came
/// back from a compile on this thread, if that is what it was.
///
/// `Some` means the error was an `E-JIT-005` refusal; `None` means the error
/// means what its variant says. Consuming, and cleared at the start of every
/// compile, so a refusal is never attributed to a later, unrelated failure.
pub fn take_unbound_symbols() -> Option<UnboundSymbolsError> {
    LAST_UNBOUND.with(|c| c.borrow_mut().take())
}

/// Check that every free symbol of `expr` is bound by `inputs`.
///
/// `π` needs no binding: every evaluator resolves an unbound `pi` to its
/// value (an input named `pi` still takes precedence). Bound variables of
/// binders (a `RootSum`'s root placeholder, a quantifier's variable) are not
/// free. The walk is iterative, so a deep expression cannot overflow the
/// stack, and it runs once per compile, never per call.
pub fn check_inputs_bind(
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<(), UnboundSymbolsError> {
    // Only a symbol binds anything: the evaluators consult their environment at
    // `Symbol` nodes alone, so an input like `x + 1` leaves `x` unbound.
    let input_set: IdSet = inputs
        .iter()
        .copied()
        .filter(|&i| pool.with(i, |d| matches!(d, ExprData::Symbol { .. })))
        .collect();
    // Binder scopes as `(parent scope, bound variable)`; scope 0 is the root
    // and binds nothing (its entry is a placeholder).
    let mut scopes: Vec<(usize, ExprId)> = vec![(0, expr)];
    let is_bound = |scopes: &[(usize, ExprId)], mut scope: usize, id: ExprId| {
        while scope != 0 {
            let (parent, var) = scopes[scope];
            if var == id {
                return true;
            }
            scope = parent;
        }
        false
    };
    let mut visited: std::collections::HashSet<(ExprId, usize)> = Default::default();
    let mut unbound: std::collections::BTreeSet<String> = Default::default();
    let mut stack: Vec<(ExprId, usize)> = vec![(expr, 0)];
    while let Some((id, scope)) = stack.pop() {
        if input_set.contains(&id) || !visited.insert((id, scope)) {
            continue;
        }
        pool.with(id, |data| match data {
            ExprData::Symbol { name, .. } => {
                if crate::eval::symbols::constant_value_f64(name).is_none()
                    && !is_bound(&scopes, scope, id)
                {
                    unbound.insert(name.clone());
                }
            }
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => {}
            ExprData::Add(args)
            | ExprData::Mul(args)
            | ExprData::Func { args, .. }
            | ExprData::Predicate { args, .. } => {
                stack.extend(args.iter().map(|&a| (a, scope)));
            }
            ExprData::Pow { base, exp } => {
                stack.push((*base, scope));
                stack.push((*exp, scope));
            }
            ExprData::Piecewise { branches, default } => {
                for &(c, v) in branches {
                    stack.push((c, scope));
                    stack.push((v, scope));
                }
                stack.push((*default, scope));
            }
            ExprData::BigO(inner) => stack.push((*inner, scope)),
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                scopes.push((scope, *var));
                stack.push((*body, scopes.len() - 1));
            }
            ExprData::RootSum { poly, var, body } => {
                scopes.push((scope, *var));
                let inner = scopes.len() - 1;
                stack.push((*poly, inner));
                stack.push((*body, inner));
            }
        });
    }
    if unbound.is_empty() {
        return Ok(());
    }
    let input_names: std::collections::HashSet<String> = inputs
        .iter()
        .filter_map(|&i| {
            pool.with(i, |d| match d {
                ExprData::Symbol { name, .. } => Some(name.clone()),
                _ => None,
            })
        })
        .collect();
    let names: Vec<String> = unbound.into_iter().collect();
    let shadowed = names
        .iter()
        .filter(|n| input_names.contains(*n))
        .cloned()
        .collect();
    Err(UnboundSymbolsError { names, shadowed })
}

/// Refuse a compile whose expression has an unbound symbol, recording the
/// refusal for [`take_unbound_symbols`].
///
/// Also refuses an expression that applies a function with no numeric
/// kernel (`E-JIT-006`, see [`UnevaluableFunctionError`]).
fn refuse_unbound(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> Result<(), JitError> {
    LAST_UNBOUND.with(|c| *c.borrow_mut() = None);
    LAST_UNEVALUABLE.with(|c| *c.borrow_mut() = None);
    check_inputs_bind(expr, inputs, pool).map_err(|e| {
        let err = JitError::UnsupportedNode(e.to_string());
        LAST_UNBOUND.with(|c| *c.borrow_mut() = Some(e));
        err
    })?;
    check_functions_evaluable(expr, pool).map_err(|e| {
        let err = JitError::UnsupportedNode(e.to_string());
        LAST_UNEVALUABLE.with(|c| *c.borrow_mut() = Some(e));
        err
    })
}

// ---------------------------------------------------------------------------
// Functions with no numeric kernel — refused at compile time (E-JIT-006)
// ---------------------------------------------------------------------------

/// A compile request whose expression applies a function that has no numeric
/// kernel (`E-JIT-006`): an undefined function `f(x)`, a `Derivative` of one,
/// or any other head the primitive registry does not know.
///
/// Every evaluator reaches such a node on every evaluation and can only give
/// up there, so the compiled function would be `NaN` at every point. It used
/// to be exactly that, silently: `compile_expr(f(x), [x])` compiled, and every
/// call and every batch returned `NaN`. [`compile`], [`compile_with`],
/// [`compile_jit_only`] and [`compile_many`] refuse it instead.
///
/// Like [`UnboundSymbolsError`], it travels in band as
/// [`JitError::UnsupportedNode`] (an exhaustive public enum cannot grow a
/// variant) and out of band through [`take_unevaluable_functions`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnevaluableFunctionError {
    /// The function names, sorted and de-duplicated.
    names: Vec<String>,
}

impl UnevaluableFunctionError {
    /// Names of the functions that have no numeric kernel, sorted.
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

impl fmt::Display for UnevaluableFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plural = if self.names.len() == 1 { "" } else { "s" };
        write!(
            f,
            "cannot compile: the expression applies function{plural} {} with no numeric \
             implementation (an undefined function, or a derivative of one), so the compiled \
             function would be NaN at every point",
            self.names.join(", ")
        )
    }
}

impl std::error::Error for UnevaluableFunctionError {}

impl crate::errors::AlkahestError for UnevaluableFunctionError {
    fn code(&self) -> &'static str {
        "E-JIT-006"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "substitute a concrete expression for each undefined function (and evaluate its \
             derivatives) before compiling",
        )
    }
}

thread_local! {
    static LAST_UNEVALUABLE: std::cell::RefCell<Option<UnevaluableFunctionError>> =
        const { std::cell::RefCell::new(None) };
}

/// Take the `E-JIT-006` refusal behind the [`JitError`] that just came back
/// from a compile on this thread, if that is what it was. Consuming, and
/// cleared at the start of every compile — see [`take_unbound_symbols`].
pub fn take_unevaluable_functions() -> Option<UnevaluableFunctionError> {
    LAST_UNEVALUABLE.with(|c| c.borrow_mut().take())
}

/// Check that every function `expr` applies has a numeric kernel in the
/// primitive registry — the table every evaluator dispatches `Func` nodes to.
pub fn check_functions_evaluable(
    expr: ExprId,
    pool: &ExprPool,
) -> Result<(), UnevaluableFunctionError> {
    let mut visited = IdSet::default();
    let mut stack = vec![expr];
    let mut missing: std::collections::BTreeSet<String> = Default::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        pool.with(id, |data| match data {
            ExprData::Func { name, args } => {
                // `RootOf` is a constant every evaluator computes itself (from
                // its integer arguments), not a registry primitive.
                if name != crate::poly::root_of::ROOT_OF && registry().get(name).is_none() {
                    missing.insert(name.clone());
                }
                stack.extend_from_slice(args);
            }
            ExprData::Add(args) | ExprData::Mul(args) | ExprData::Predicate { args, .. } => {
                stack.extend_from_slice(args);
            }
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                stack.push(*exp);
            }
            ExprData::Piecewise { branches, default } => {
                for &(c, v) in branches {
                    stack.push(c);
                    stack.push(v);
                }
                stack.push(*default);
            }
            ExprData::BigO(inner) => stack.push(*inner),
            ExprData::Forall { body, .. } | ExprData::Exists { body, .. } => stack.push(*body),
            ExprData::RootSum { poly, body, .. } => {
                stack.push(*poly);
                stack.push(*body);
            }
            ExprData::Symbol { .. }
            | ExprData::Integer(_)
            | ExprData::Rational(_)
            | ExprData::Float(_) => {}
        });
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(UnevaluableFunctionError {
            names: missing.into_iter().collect(),
        })
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
        /// The column entry point, as Cranelift's: a batch, a Rayon chunk
        /// (a sub-range) or a flat bulk call are all one call of it.
        cols_fn: JitColsFn,
        // execution_engine must be declared before _context so it drops first;
        // the context must outlive the execution engine.
        #[allow(dead_code)]
        execution_engine: inkwell::execution_engine::ExecutionEngine<'static>,
        _context: Box<inkwell::context::Context>,
    },

    #[cfg(feature = "cranelift")]
    Cranelift {
        fn_ptr: JitScalarFn,
        cols_fn: JitColsFn,
        /// The registry primitives the code calls through its generic
        /// trampolines; the code embeds the address of each element.
        _prims: Box<[&'static dyn crate::primitive::Primitive]>,
        /// Owns the code pages `fn_ptr` and `cols_fn` point into, and frees
        /// them when this variant is dropped (see
        /// [`cranelift_backend::OwnedJitModule`]).
        _module: Box<cranelift_backend::OwnedJitModule>,
    },

    Interpreter(InterpreterFn),

    /// The interpreter tier for an expression [`NumericProgram`] can model
    /// (`Piecewise` included): the DAG flattened once at compile time, so a
    /// call is a loop over operations with no hashing, and a batch runs each
    /// operation over a block of points at a time.
    Program(Box<NumericProgram>),
}

thread_local! {
    /// Scratch slots for [`CompiledFn::call`] on the program tier, so a scalar
    /// call does not allocate. Taken out and put back rather than borrowed, so
    /// a re-entrant call (a primitive that evaluates something itself) simply
    /// finds it empty and allocates its own.
    static PROGRAM_SCRATCH: std::cell::Cell<Vec<f64>> = const { std::cell::Cell::new(Vec::new()) };
}

/// Points per Rayon task in [`CompiledFn::call_batch_par`].
#[cfg(feature = "parallel")]
const PAR_CHUNK: usize = 8192;

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
    /// The expression over complex `f64`, for a point whose real evaluation
    /// is NaN; `None` when it cannot have complex intermediates. See
    /// [`complex_fallback`].
    complex: Option<Box<ComplexFallback>>,
}

impl CompiledFn {
    /// Evaluate with the given inputs.  Panics if `inputs.len() != n_inputs`.
    ///
    /// A point where the real arithmetic produces NaN although the value is a
    /// real number reached through complex intermediates (`cos(t·√ω²)` with
    /// `ω² < 0`, Cardano's formula with three real roots) is re-evaluated in
    /// complex arithmetic, and the real part is returned when the imaginary
    /// part is zero within the evaluation's rounding-error bound. NaN is
    /// returned only where the value is not a real number (`log(−1)`, `√−2`)
    /// or cannot be decided in double precision.
    pub fn call(&self, inputs: &[f64]) -> f64 {
        let v = self.call_real(inputs);
        if v.is_nan() {
            if let Some(c) = &self.complex {
                return c.eval_real(inputs).unwrap_or(v);
            }
        }
        v
    }

    /// Replace each NaN in `output` (point `offset + j`) by its complex
    /// fallback value where there is one. `point(j, buf)` fills the inputs
    /// of point `j` of `output`.
    fn repair(&self, output: &mut [f64], mut point: impl FnMut(usize, &mut [f64])) {
        let Some(c) = &self.complex else { return };
        let mut buf = vec![0.0f64; self.n_inputs];
        for (j, out) in output.iter_mut().enumerate() {
            if out.is_nan() {
                point(j, &mut buf);
                if let Some(v) = c.eval_real(&buf) {
                    *out = v;
                }
            }
        }
    }

    fn repair_slices(&self, inputs: &[&[f64]], output: &mut [f64], offset: usize) {
        self.repair(output, |j, buf| {
            for (slot, col) in buf.iter_mut().zip(inputs) {
                *slot = col[offset + j];
            }
        });
    }

    /// The real evaluation alone (no complex fallback).
    fn call_real(&self, inputs: &[f64]) -> f64 {
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
            CompiledFnInner::Program(p) => {
                let mut scratch = PROGRAM_SCRATCH.with(|c| c.take());
                if scratch.len() != p.scratch_len() {
                    scratch = p.scratch();
                } else {
                    p.reset_scratch(&mut scratch);
                }
                let v = p.eval_in(inputs, &mut scratch).unwrap_or(f64::NAN);
                PROGRAM_SCRATCH.with(|c| c.set(scratch));
                v
            }
        }
    }

    /// Bulk-evaluate over N points using the native bulk JIT entry when available.
    ///
    /// `inputs_flat` is **column-major** (var-major), length `n_inputs * n_points`:
    /// variable `i` occupies `inputs_flat[i * n_points .. (i + 1) * n_points]`.
    /// `output` has length `n_points`.
    pub fn call_bulk(&self, inputs_flat: &[f64], output: &mut [f64]) {
        self.call_bulk_real(inputs_flat, output);
        let n_points = output.len();
        self.repair(output, |j, buf| {
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = inputs_flat[i * n_points + j];
            }
        });
    }

    fn call_bulk_real(&self, inputs_flat: &[f64], output: &mut [f64]) {
        let n_points = output.len();
        assert_eq!(
            inputs_flat.len(),
            self.n_inputs * n_points,
            "inputs_flat length {} != n_inputs({}) * n_points({})",
            inputs_flat.len(),
            self.n_inputs,
            n_points
        );
        #[cfg(any(feature = "jit", feature = "cranelift"))]
        if let Some(cols_fn) = self.native_cols_fn() {
            let cols: Vec<*const f64> = (0..self.n_inputs)
                .map(|i| inputs_flat[i * n_points..].as_ptr())
                .collect();
            let outs = [output.as_mut_ptr()];
            // SAFETY: every column holds `n_points` values (asserted above)
            // and the output slice is exclusively borrowed.
            unsafe { cols_fn(cols.as_ptr(), outs.as_ptr(), 0, n_points as u64) };
            return;
        }
        if let CompiledFnInner::Program(p) = &self.inner {
            return p.eval_columns(inputs_flat, output);
        }
        let mut point = vec![0.0f64; self.n_inputs];
        for j in 0..n_points {
            for (i, slot) in point.iter_mut().enumerate() {
                *slot = inputs_flat[i * n_points + j];
            }
            output[j] = self.call_real(&point);
        }
    }

    /// Batch-evaluate over N points (sequential).
    ///
    /// `inputs[i]` contains the values of variable `i` for all N points.
    /// All slices must have the same length N.  `output` must also have length N.
    ///
    /// This is the hot path for NumPy/JAX array evaluation.
    /// For multi-core throughput see `call_batch_par` (requires the `parallel` feature).
    pub fn call_batch(&self, inputs: &[&[f64]], output: &mut [f64]) {
        self.call_batch_real(inputs, output);
        self.repair_slices(inputs, output, 0);
    }

    fn call_batch_real(&self, inputs: &[&[f64]], output: &mut [f64]) {
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
        // No early return for `n_inputs == 0`: a function of no inputs is a
        // constant, and a batch of N points is that constant N times. An
        // early return here used to leave `output` as the caller allocated
        // it — zeros — so `compile(5, [])` batched to `[0, 0, 0]` while
        // `call(&[])` answered 5. Every path below handles zero columns.
        //
        // The program tier reads the columns in place; the copy below exists
        // only to give the native bulk entry points their flat layout.
        if let CompiledFnInner::Program(p) = &self.inner {
            return p.eval_slices(inputs, output, 0);
        }
        // A native backend reads the columns where they are: no copy, no
        // allocation beyond one pointer per column.
        #[cfg(any(feature = "jit", feature = "cranelift"))]
        if let Some(cols_fn) = self.native_cols_fn() {
            let cols: Vec<*const f64> = inputs.iter().map(|c| c.as_ptr()).collect();
            let outs = [output.as_mut_ptr()];
            // SAFETY: every column has `n` values (asserted above) and the
            // output slice is exclusively borrowed.
            unsafe { cols_fn(cols.as_ptr(), outs.as_ptr(), 0, n as u64) };
            return;
        }
        // Column-major flat layout matches `call_bulk` / Python `call_batch_raw`.
        let mut flat = Vec::with_capacity(self.n_inputs * n);
        for col in inputs {
            flat.extend_from_slice(col);
        }
        self.call_bulk_real(&flat, output);
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
        self.call_batch_par_real(inputs, output);
        if self.complex.is_some() {
            output
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .for_each(|(c, chunk)| self.repair_slices(inputs, chunk, c * PAR_CHUNK));
        }
    }

    #[cfg(feature = "parallel")]
    fn call_batch_par_real(&self, inputs: &[&[f64]], output: &mut [f64]) {
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

        // Below a couple of chunks, threads cost more than they save.
        if n < 2 * PAR_CHUNK {
            return self.call_batch_real(inputs, output);
        }

        // A native backend (Cranelift or LLVM): each chunk is one call of the
        // column entry over its own index range — nothing is copied or
        // allocated per chunk.
        #[cfg(any(feature = "jit", feature = "cranelift"))]
        if let Some(cols_fn) = self.native_cols_fn() {
            let cols: Vec<usize> = inputs.iter().map(|c| c.as_ptr() as usize).collect();
            let out_base = output.as_mut_ptr() as usize;
            (0..n.div_ceil(PAR_CHUNK)).into_par_iter().for_each(|c| {
                let start = c * PAR_CHUNK;
                let end = (start + PAR_CHUNK).min(n);
                let outs = [out_base as *mut f64];
                // SAFETY: the columns hold `n` values each; the chunks are
                // disjoint index ranges of `output`, which this call borrows
                // exclusively, so no two tasks write the same element.
                unsafe {
                    cols_fn(
                        cols.as_ptr() as *const *const f64,
                        outs.as_ptr(),
                        start as u64,
                        end as u64,
                    )
                };
            });
            return;
        }

        // Each point is independent. Work is split into chunks so the
        // per-task state (a point buffer, or the program's block storage) is
        // allocated once per chunk rather than once per point; the program
        // tier then runs each chunk a block of points at a time.
        output
            .par_chunks_mut(PAR_CHUNK)
            .enumerate()
            .for_each(|(c, chunk)| {
                let offset = c * PAR_CHUNK;
                if let CompiledFnInner::Program(p) = &self.inner {
                    return p.eval_slices(inputs, chunk, offset);
                }
                let mut point = vec![0.0f64; self.n_inputs];
                for (j, out) in chunk.iter_mut().enumerate() {
                    for (slot, col) in point.iter_mut().zip(inputs) {
                        *slot = col[offset + j];
                    }
                    *out = self.call_real(&point);
                }
            });
    }

    /// Backend tier used to implement this function.
    pub fn compile_tier(&self) -> CompileTier {
        self.tier
    }

    /// The native column entry point ([`JitColsFn`]), when a JIT backend
    /// compiled this function.
    #[cfg(any(feature = "jit", feature = "cranelift"))]
    fn native_cols_fn(&self) -> Option<JitColsFn> {
        match &self.inner {
            #[cfg(feature = "jit")]
            CompiledFnInner::Llvm { cols_fn, .. } => Some(*cols_fn),
            #[cfg(feature = "cranelift")]
            CompiledFnInner::Cranelift { cols_fn, .. } => Some(*cols_fn),
            _ => None,
        }
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
// CompiledVecFn — several outputs, one pass
// ---------------------------------------------------------------------------

enum CompiledVecInner {
    /// One native loop computing every output per point; subexpressions the
    /// outputs share (the common factors of a gradient or Jacobian) are
    /// evaluated once.
    #[cfg(feature = "cranelift")]
    Cranelift {
        cols_fn: JitColsFn,
        _prims: Box<[&'static dyn crate::primitive::Primitive]>,
        _module: Box<cranelift_backend::OwnedJitModule>,
    },
    /// No native backend (or one that declined): each output compiled on its
    /// own, through the same tiers as [`compile_with`].
    PerOutput(Vec<CompiledFn>),
}

/// A compiled vector-valued function: `n_outputs` expressions of the same
/// `n_inputs` variables, evaluated together. Built by [`compile_many`].
///
/// Every output equals, bit for bit, what [`compile`] of that expression
/// alone computes at the same point — sharing changes how often a common
/// subexpression is evaluated, never its value.
pub struct CompiledVecFn {
    inner: CompiledVecInner,
    n_inputs: usize,
    n_outputs: usize,
    /// Per-output complex fallbacks for the shared native loop (the
    /// per-output tier carries them in each [`CompiledFn`]); empty when no
    /// output needs one.
    #[cfg_attr(not(feature = "cranelift"), allow(dead_code))]
    complex: Vec<Option<ComplexFallback>>,
}

// SAFETY: as for `CompiledFn` — immutable code pages, function pointers only
// read, and owners kept solely to keep the pages alive.
unsafe impl Send for CompiledVecFn {}
unsafe impl Sync for CompiledVecFn {}

impl CompiledVecFn {
    /// Number of input variables.
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// Number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// Backend evaluating the outputs (for diagnostics).
    pub fn compile_tier(&self) -> CompileTier {
        match &self.inner {
            #[cfg(feature = "cranelift")]
            CompiledVecInner::Cranelift { .. } => CompileTier::Cranelift,
            CompiledVecInner::PerOutput(fs) => {
                fs.first().map_or(CompileTier::Interpreter, |f| f.tier)
            }
        }
    }

    /// Evaluate every output at one point. Panics unless
    /// `inputs.len() == n_inputs` and `out.len() == n_outputs`.
    pub fn call(&self, inputs: &[f64], out: &mut [f64]) {
        assert_eq!(
            inputs.len(),
            self.n_inputs,
            "expected {} inputs",
            self.n_inputs
        );
        assert_eq!(
            out.len(),
            self.n_outputs,
            "expected {} outputs",
            self.n_outputs
        );
        match &self.inner {
            #[cfg(feature = "cranelift")]
            CompiledVecInner::Cranelift { cols_fn, .. } => {
                // One point: each "column" is a single value.
                let cols: Vec<*const f64> = inputs.iter().map(|v| v as *const f64).collect();
                let outs: Vec<*mut f64> = out.iter_mut().map(|v| v as *mut f64).collect();
                // SAFETY: every pointer addresses one live element; index 0
                // is the only one read or written.
                unsafe { cols_fn(cols.as_ptr(), outs.as_ptr(), 0, 1) };
                for (o, c) in out.iter_mut().zip(&self.complex) {
                    if let (true, Some(c)) = (o.is_nan(), c) {
                        if let Some(v) = c.eval_real(inputs) {
                            *o = v;
                        }
                    }
                }
            }
            CompiledVecInner::PerOutput(fs) => {
                for (o, f) in out.iter_mut().zip(fs) {
                    *o = f.call(inputs);
                }
            }
        }
    }

    /// The complex fallback (see [`complex_fallback`]) over every NaN of a
    /// native batch.
    #[cfg(feature = "cranelift")]
    fn repair_outputs(&self, inputs: &[&[f64]], outputs: &mut [&mut [f64]]) {
        let mut buf = vec![0.0f64; self.n_inputs];
        for (o, c) in outputs.iter_mut().zip(&self.complex) {
            let Some(c) = c else { continue };
            for (j, v) in o.iter_mut().enumerate() {
                if v.is_nan() {
                    for (slot, col) in buf.iter_mut().zip(inputs) {
                        *slot = col[j];
                    }
                    if let Some(r) = c.eval_real(&buf) {
                        *v = r;
                    }
                }
            }
        }
    }

    fn check_batch(&self, inputs: &[&[f64]], outputs: &[&mut [f64]]) -> usize {
        assert_eq!(
            inputs.len(),
            self.n_inputs,
            "expected {} input arrays, got {}",
            self.n_inputs,
            inputs.len()
        );
        assert_eq!(
            outputs.len(),
            self.n_outputs,
            "expected {} output arrays, got {}",
            self.n_outputs,
            outputs.len()
        );
        let n = outputs
            .first()
            .map(|o| o.len())
            .or_else(|| inputs.first().map(|c| c.len()))
            .unwrap_or(0);
        for col in inputs {
            assert_eq!(col.len(), n, "all input arrays must have the same length");
        }
        for col in outputs {
            assert_eq!(col.len(), n, "all output arrays must have the same length");
        }
        n
    }

    /// Evaluate over `N` points: `inputs[i]` holds variable `i` at every
    /// point, `outputs[k]` receives output `k`. All slices have length `N`.
    pub fn call_batch(&self, inputs: &[&[f64]], outputs: &mut [&mut [f64]]) {
        let n = self.check_batch(inputs, outputs);
        match &self.inner {
            #[cfg(feature = "cranelift")]
            CompiledVecInner::Cranelift { cols_fn, .. } => {
                let cols: Vec<*const f64> = inputs.iter().map(|c| c.as_ptr()).collect();
                let outs: Vec<*mut f64> = outputs.iter_mut().map(|o| o.as_mut_ptr()).collect();
                // SAFETY: every column has `n` values (checked) and the output
                // slices are exclusively borrowed (and so pairwise disjoint).
                unsafe { cols_fn(cols.as_ptr(), outs.as_ptr(), 0, n as u64) };
                self.repair_outputs(inputs, outputs);
            }
            CompiledVecInner::PerOutput(fs) => {
                let _ = n;
                for (o, f) in outputs.iter_mut().zip(fs) {
                    f.call_batch(inputs, o);
                }
            }
        }
    }

    /// [`call_batch`](Self::call_batch) across all cores (Rayon).
    #[cfg(feature = "parallel")]
    pub fn call_batch_par(&self, inputs: &[&[f64]], outputs: &mut [&mut [f64]]) {
        let n = self.check_batch(inputs, outputs);
        if n < 2 * PAR_CHUNK {
            return self.call_batch(inputs, outputs);
        }
        match &self.inner {
            #[cfg(feature = "cranelift")]
            CompiledVecInner::Cranelift { cols_fn, .. } => {
                use rayon::prelude::*;
                let cols: Vec<usize> = inputs.iter().map(|c| c.as_ptr() as usize).collect();
                let outs: Vec<usize> = outputs
                    .iter_mut()
                    .map(|o| o.as_mut_ptr() as usize)
                    .collect();
                let cols_fn = *cols_fn;
                (0..n.div_ceil(PAR_CHUNK)).into_par_iter().for_each(|c| {
                    let start = c * PAR_CHUNK;
                    let end = (start + PAR_CHUNK).min(n);
                    // SAFETY: columns hold `n` values; chunks are disjoint
                    // index ranges of the exclusively borrowed outputs.
                    unsafe {
                        cols_fn(
                            cols.as_ptr() as *const *const f64,
                            outs.as_ptr() as *const *mut f64,
                            start as u64,
                            end as u64,
                        )
                    };
                });
                self.repair_outputs(inputs, outputs);
            }
            CompiledVecInner::PerOutput(fs) => {
                for (o, f) in outputs.iter_mut().zip(fs) {
                    f.call_batch_par(inputs, o);
                }
            }
        }
    }
}

/// Compile several expressions of the same inputs into one
/// [`CompiledVecFn`], evaluated in a single pass per point.
///
/// The main consumer is a gradient or Jacobian: its entries share most of
/// their subexpressions, and evaluating them together computes each shared
/// node once. Refusals are those of [`compile`] (`E-JIT-005` for an unbound
/// symbol, `E-JIT-006` for a function with no numeric kernel), checked for
/// every output. Without a native backend, or for a form the native backend
/// does not compile, each output is compiled on its own.
pub fn compile_many(
    exprs: &[ExprId],
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CompiledVecFn, JitError> {
    compile_many_with(exprs, inputs, pool, CompileConfig::default())
}

/// [`compile_many`] with an explicit [`CompileConfig`]. Only
/// `force_tier` is consulted: a vector of outputs is compiled natively
/// whenever a native backend can take it.
pub fn compile_many_with(
    exprs: &[ExprId],
    inputs: &[ExprId],
    pool: &ExprPool,
    config: CompileConfig,
) -> Result<CompiledVecFn, JitError> {
    for &e in exprs {
        refuse_unbound(e, inputs, pool)?;
    }
    let per_output = |tier: Option<CompileTier>| -> Result<CompiledVecFn, JitError> {
        let fs = exprs
            .iter()
            .map(|&e| {
                let t = tier.unwrap_or_else(|| select_compile_tier(e, pool, &config));
                compile_with_fallbacks(t, e, inputs, pool)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CompiledVecFn {
            inner: CompiledVecInner::PerOutput(fs),
            n_inputs: inputs.len(),
            n_outputs: exprs.len(),
            complex: Vec::new(),
        })
    };
    #[cfg(feature = "cranelift")]
    {
        let native_ok = !matches!(config.force_tier, Some(CompileTier::Interpreter))
            && !exprs.is_empty()
            && !exprs.iter().any(|&e| needs_exact_integer_exponent(e, pool));
        if native_ok {
            if let Ok(code) = cranelift_backend::compile_cranelift_many(exprs, inputs, pool) {
                let complex: Vec<Option<ComplexFallback>> = exprs
                    .iter()
                    .map(|&e| ComplexFallback::build(e, inputs, pool))
                    .collect();
                let any = complex.iter().any(Option::is_some);
                return Ok(CompiledVecFn {
                    inner: CompiledVecInner::Cranelift {
                        cols_fn: code.cols,
                        _prims: code.prims,
                        _module: code.module,
                    },
                    n_inputs: inputs.len(),
                    n_outputs: exprs.len(),
                    complex: if any { complex } else { Vec::new() },
                });
            }
        }
    }
    per_output(config.force_tier)
}

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

/// True when `expr` contains a power whose exponent is an exact integer that
/// `f64` cannot hold.
///
/// Both native backends materialise the exponent as an `f64` constant and call
/// `pow` — Cranelift through `alkahest_pow`, LLVM through `llvm.pow.f64` —
/// which is exactly the reduction [`crate::kernel::pow_f64`] exists to undo.
/// Neither can see the exact node from inside the emitted code, so the fix
/// belongs where the tier is chosen: an expression like `x^(10^30 + 1)` is
/// evaluated by the interpreter, which still has the node and gets the sign
/// right.  It is one DAG walk per *compile*, never per call.
fn needs_exact_integer_exponent(expr: ExprId, pool: &ExprPool) -> bool {
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let data = pool.get(id);
        if let ExprData::Pow { exp, .. } = &data {
            let n = match pool.get(*exp) {
                ExprData::Integer(n) => Some(n.0.clone()),
                ExprData::Rational(r) if *r.0.denom() == 1 => Some(r.0.numer().clone()),
                _ => None,
            };
            if n.is_some_and(|n| !integer_is_exact_f64(&n)) {
                return true;
            }
        }
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
    false
}

/// Attach the complex fallback (see [`complex_fallback`]) to a compiled
/// function, when the expression can need one.
fn with_complex_fallback(
    mut f: CompiledFn,
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> CompiledFn {
    f.complex = ComplexFallback::build(expr, inputs, pool).map(Box::new);
    f
}

fn compile_with_fallbacks(
    tier: CompileTier,
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CompiledFn, JitError> {
    compile_with_fallbacks_real(tier, expr, inputs, pool)
        .map(|f| with_complex_fallback(f, expr, inputs, pool))
}

fn compile_with_fallbacks_real(
    tier: CompileTier,
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CompiledFn, JitError> {
    // Correctness outranks the tier, including a forced one: `compile_tier()`
    // reports `Interpreter` afterwards, so the downgrade is visible rather
    // than silent.
    let tier =
        if matches!(tier, CompileTier::Interpreter) || !needs_exact_integer_exponent(expr, pool) {
            tier
        } else {
            CompileTier::Interpreter
        };
    match compile_for_tier(tier, expr, inputs, pool) {
        Ok(f) => Ok(f),
        Err(e) => match tier {
            CompileTier::Interpreter => Err(e),
            #[cfg(feature = "jit")]
            CompileTier::Llvm => {
                #[cfg(feature = "cranelift")]
                if let Ok(f) = cranelift_backend::compile_cranelift(expr, inputs, pool) {
                    return Ok(f);
                }
                compile_interpreter(expr, inputs, pool)
            }
            #[cfg(feature = "cranelift")]
            CompileTier::Cranelift => {
                #[cfg(feature = "jit")]
                if let Ok(f) = compile_llvm(expr, inputs, pool) {
                    return Ok(f);
                }
                compile_interpreter(expr, inputs, pool)
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
    refuse_unbound(expr, inputs, pool)?;
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
    refuse_unbound(expr, inputs, pool)?;
    #[cfg(feature = "cranelift")]
    return cranelift_backend::compile_cranelift(expr, inputs, pool)
        .map(|f| with_complex_fallback(f, expr, inputs, pool));

    #[cfg(all(feature = "jit", not(feature = "cranelift")))]
    return compile_llvm(expr, inputs, pool).map(|f| with_complex_fallback(f, expr, inputs, pool));

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
    let mut memo: IdMap<f64> = IdMap::default();
    eval_interp_inner(expr, env, pool, &mut memo)
}

/// [`eval_interp_checked`], with a NaN result re-evaluated through the
/// complex fallback (see [`complex_fallback`]) as [`CompiledFn::call`] does:
/// a real value reached through complex intermediates — `cos(t·√ω²)` at
/// `ω² < 0`, Cardano's formula with three real roots — is returned rather
/// than refused. [`InterpEvalError::NonFinite`] then means the value is not a
/// real number at this point (`log(−1)`, `√−2`), not that real arithmetic
/// could not reach it. This is what Python's `eval_expr` calls.
///
/// [`eval_interp`] and [`eval_interp_checked`] keep plain real `f64`
/// semantics (a NaN is a NaN); the many internal callers that sample an
/// expression rely on that.
pub fn eval_interp_real(
    expr: ExprId,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
) -> Result<f64, InterpEvalError> {
    match eval_interp(expr, env, pool) {
        Some(v) if v.is_nan() => {
            let (vars, vals): (Vec<ExprId>, Vec<f64>) = env.iter().map(|(&k, &v)| (k, v)).unzip();
            ComplexFallback::build(expr, &vars, pool)
                .and_then(|c| c.eval_real(&vals))
                .filter(|v| v.is_finite())
                .ok_or(InterpEvalError::NonFinite)
        }
        _ => eval_interp_checked(expr, env, pool),
    }
}

/// One expression sampled at many values of one variable, every other binding
/// held fixed — the shape of the integrator's pole and jump scans.
///
/// [`Sampler::eval`]`(t)` returns exactly what
/// `eval_interp(expr, fixed ∪ {var: t}, pool)` returns, but the expression is
/// flattened into a [`NumericProgram`] once, so each sample is a pass over a
/// slot array rather than a hashed walk of the pool. Forms the program does
/// not model (`Piecewise`, `RootSum`, …) fall back to [`eval_interp`].
pub(crate) struct Sampler<'a> {
    expr: ExprId,
    var: ExprId,
    pool: &'a ExprPool,
    state: std::cell::RefCell<SamplerState>,
}

enum SamplerState {
    /// `inputs[0]` is `var`; the rest are the fixed bindings.
    Program {
        program: NumericProgram,
        inputs: Vec<f64>,
        scratch: Vec<f64>,
    },
    Walk(HashMap<ExprId, f64>),
}

impl<'a> Sampler<'a> {
    pub(crate) fn new(
        expr: ExprId,
        var: ExprId,
        fixed: &HashMap<ExprId, f64>,
        pool: &'a ExprPool,
    ) -> Self {
        let mut vars = vec![var];
        let mut inputs = vec![0.0];
        for (&k, &v) in fixed {
            if k != var {
                vars.push(k);
                inputs.push(v);
            }
        }
        let state = match NumericProgram::compile(expr, &vars, pool) {
            Some(program) => SamplerState::Program {
                scratch: program.scratch(),
                program,
                inputs,
            },
            None => SamplerState::Walk(fixed.clone()),
        };
        Self {
            expr,
            var,
            pool,
            state: std::cell::RefCell::new(state),
        }
    }

    /// The expression's value with `var = t`; see the type docs.
    pub(crate) fn eval(&self, t: f64) -> Option<f64> {
        match &mut *self.state.borrow_mut() {
            SamplerState::Program {
                program,
                inputs,
                scratch,
            } => {
                inputs[0] = t;
                program.eval_in(inputs, scratch)
            }
            SamplerState::Walk(env) => {
                env.insert(self.var, t);
                eval_interp(self.expr, env, self.pool)
            }
        }
    }
}

/// Why [`eval_interp_checked`] could not produce a usable number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpEvalError {
    /// An unbound symbol or an expression form the interpreter doesn't
    /// support — the same condition under which [`eval_interp`] returns
    /// `None`.
    Unevaluable,
    /// The interpreter produced `NaN` or `±inf`. Under plain IEEE-754
    /// arithmetic this happens for e.g. `0^-1` or `0 * inf`, which is what
    /// `(x²-1)/(x-1)` reduces to at `x = 1`: the numerator vanishes, the
    /// denominator's reciprocal blows up, and `0 * inf = NaN`. That is not a
    /// number the caller should treat as the value of the expression at that
    /// point — the expression is undefined there (the value `2` is the limit
    /// as `x → 1`, obtained only after an explicit `cancel()`).
    NonFinite,
}

impl std::fmt::Display for InterpEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpEvalError::Unevaluable => {
                write!(
                    f,
                    "expression could not be evaluated (unbound variable or unsupported node)"
                )
            }
            InterpEvalError::NonFinite => write!(
                f,
                "expression is undefined at this point (substitution produced a non-finite \
                 result, e.g. division by zero); this is not the same as a removable-\
                 singularity limit — call cancel() first if that's what you want"
            ),
        }
    }
}

impl std::error::Error for InterpEvalError {}

/// Like [`eval_interp`], but treats a non-finite result (`NaN`/`±inf`) as an
/// evaluation failure instead of a number.
///
/// `eval_interp` itself is unchanged (and stays available for callers that
/// already guard for finiteness themselves, e.g. the SVG plotter, which
/// skips non-finite sample points) — this wrapper exists for entry points
/// that hand the numeric result directly to a caller who could otherwise
/// mistake a silent `NaN` for a valid answer, most importantly the Python
/// `eval_expr` binding.
pub fn eval_interp_checked(
    expr: ExprId,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
) -> Result<f64, InterpEvalError> {
    match eval_interp(expr, env, pool) {
        Some(v) if v.is_finite() => Ok(v),
        Some(_) => Err(InterpEvalError::NonFinite),
        None => Err(InterpEvalError::Unevaluable),
    }
}

fn eval_interp_predicate(
    pred: ExprId,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
    memo: &mut IdMap<f64>,
) -> Option<bool> {
    if let Some(b) = try_predicate_bool_from_expr(pred, pool) {
        return Some(b);
    }
    let ExprData::Predicate { kind, args } = pool.get(pred) else {
        return None;
    };
    match kind {
        PredicateKind::True => Some(true),
        PredicateKind::False => Some(false),
        PredicateKind::Not => Some(!eval_interp_predicate(args[0], env, pool, memo)?),
        PredicateKind::And => {
            for &a in &args {
                if !eval_interp_predicate(a, env, pool, memo)? {
                    return Some(false);
                }
            }
            Some(true)
        }
        PredicateKind::Or => {
            for &a in &args {
                if eval_interp_predicate(a, env, pool, memo)? {
                    return Some(true);
                }
            }
            Some(false)
        }
        PredicateKind::Lt => Some(
            eval_interp_inner(args[0], env, pool, memo)?
                < eval_interp_inner(args[1], env, pool, memo)?,
        ),
        PredicateKind::Le => Some(
            eval_interp_inner(args[0], env, pool, memo)?
                <= eval_interp_inner(args[1], env, pool, memo)?,
        ),
        PredicateKind::Gt => Some(
            eval_interp_inner(args[0], env, pool, memo)?
                > eval_interp_inner(args[1], env, pool, memo)?,
        ),
        PredicateKind::Ge => Some(
            eval_interp_inner(args[0], env, pool, memo)?
                >= eval_interp_inner(args[1], env, pool, memo)?,
        ),
        PredicateKind::Eq => Some(
            eval_interp_inner(args[0], env, pool, memo)?
                == eval_interp_inner(args[1], env, pool, memo)?,
        ),
        PredicateKind::Ne => Some(
            eval_interp_inner(args[0], env, pool, memo)?
                != eval_interp_inner(args[1], env, pool, memo)?,
        ),
    }
}

fn eval_interp_inner(
    expr: ExprId,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
    memo: &mut IdMap<f64>,
) -> Option<f64> {
    if let Some(&cached) = memo.get(&expr) {
        return Some(cached);
    }
    // Borrow the node rather than `pool.get` it: a clone copies the argument
    // vector, and a `Func`'s name, on every visit.
    let val = pool.with(expr, |data| eval_interp_data(expr, data, env, pool, memo));
    if let Some(v) = val {
        memo.insert(expr, v);
    }
    val
}

fn eval_interp_data(
    expr: ExprId,
    data: &ExprData,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
    memo: &mut IdMap<f64>,
) -> Option<f64> {
    match data {
        ExprData::Integer(n) => Some(integer_to_f64(&n.0)),
        ExprData::Rational(r) => Some(rational_to_f64(&r.0)),
        ExprData::Float(f) => Some(f.inner.to_f64()),
        // `π` is an ordinary symbol in this crate, so it resolves to its own
        // value rather than being reported unevaluable; an explicit `env` entry
        // still wins.  See [`crate::eval::symbols`].
        ExprData::Symbol { name, .. } => env
            .get(&expr)
            .copied()
            .or_else(|| crate::eval::symbols::constant_value_f64(name)),
        ExprData::Add(args) => {
            let mut sum = 0.0f64;
            for &a in args {
                sum += eval_interp_inner(a, env, pool, memo)?;
            }
            Some(sum)
        }
        ExprData::Mul(args) => {
            let mut prod = 1.0f64;
            for &a in args {
                prod *= eval_interp_inner(a, env, pool, memo)?;
            }
            Some(prod)
        }
        ExprData::Pow { base, exp } => {
            let b = eval_interp_inner(*base, env, pool, memo)?;
            let e = eval_interp_inner(*exp, env, pool, memo)?;
            Some(pow_f64(b, e, || Some(pool.get(*exp))))
        }
        // A real algebraic number; a non-real root has no `f64` value.
        ExprData::Func { name, .. } if name == crate::poly::root_of::ROOT_OF => {
            crate::poly::root_of::root_of_real_f64(expr, pool)
        }
        ExprData::Func { name, args } => {
            // Unary heads (the common case) need no heap buffer.
            if let [a] = args.as_slice() {
                let v = eval_interp_inner(*a, env, pool, memo)?;
                return registry().numeric_f64(name.as_str(), &[v]);
            }
            let mut vals = Vec::with_capacity(args.len());
            for &a in args {
                vals.push(eval_interp_inner(a, env, pool, memo)?);
            }
            registry().numeric_f64(name.as_str(), &vals)
        }
        ExprData::Piecewise { branches, default } => {
            for &(c, v) in branches {
                match eval_interp_predicate(c, env, pool, memo) {
                    Some(true) => return eval_interp_inner(v, env, pool, memo),
                    Some(false) => {}
                    None => return None,
                }
            }
            eval_interp_inner(*default, env, pool, memo)
        }
        ExprData::Predicate { .. } => Some(if eval_interp_predicate(expr, env, pool, memo)? {
            1.0
        } else {
            0.0
        }),
        // `Σ_{c : m(c)=0} body(x, c)` — the roots are found numerically and the
        // body is summed over them in complex arithmetic. See
        // `crate::eval::root_sum` for what is and is not established by that.
        // Without this arm every Rothstein–Trager answer with an algebraic
        // residue is unevaluable, and therefore unverifiable.
        ExprData::RootSum {
            poly,
            var: rvar,
            body,
        } => crate::eval::eval_root_sum_f64(*poly, *rvar, *body, env, pool),
        _ => None,
    }
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
    // `Piecewise` compiles to a lazy program: every branch computed over a
    // block of points, then a masked select, with failures tracked per point
    // so that a branch a point does not select cannot fail it.
    if let Some(program) = NumericProgram::compile(expr, inputs, pool)
        .or_else(|| NumericProgram::compile_lazy(expr, inputs, pool))
    {
        return Ok(CompiledFn {
            inner: CompiledFnInner::Program(Box::new(program)),
            n_inputs: n,
            tier: CompileTier::Interpreter,
            complex: None,
        });
    }
    // The rest (`RootSum`, a `Predicate` used as a value, a condition that is
    // not a predicate, …) keep the snapshot walker, which evaluates only the
    // branch a point selects.
    let snapshot = snapshot_expr(expr, pool);

    let interp = move |vals: &[f64]| -> f64 {
        let mut env: HashMap<ExprId, f64> = HashMap::new();
        for (&var, &val) in inputs_vec.iter().zip(vals.iter()) {
            env.insert(var, val);
        }
        let mut memo: IdMap<f64> = IdMap::default();
        eval_interp_snap(expr, &env, &snapshot, &mut memo).unwrap_or(f64::NAN)
    };

    Ok(CompiledFn {
        inner: CompiledFnInner::Interpreter(Box::new(interp)),
        n_inputs: n,
        tier: CompileTier::Interpreter,
        complex: None,
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
    let mut visited: IdSet = IdSet::default();
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
            ExprData::Piecewise { branches, default } => {
                for (c, v) in branches {
                    stack.push(*c);
                    stack.push(*v);
                }
                stack.push(*default);
            }
            ExprData::Predicate { args, .. } => {
                stack.extend_from_slice(args);
            }
            _ => {}
        }
        nodes.insert(id, data);
    }
    ExprSnapshot { nodes }
}

fn snap_data(snap: &ExprSnapshot, id: ExprId) -> Option<&ExprData> {
    snap.nodes.get(&id)
}

fn try_expr_f64_snap(
    expr: ExprId,
    snap: &ExprSnapshot,
    env: &HashMap<ExprId, f64>,
    memo: &mut IdMap<f64>,
) -> Option<f64> {
    if let Some(&cached) = memo.get(&expr) {
        return Some(cached);
    }
    let val = match snap_data(snap, expr)? {
        ExprData::Integer(n) => Some(integer_to_f64(&n.0)),
        ExprData::Rational(r) => Some(rational_to_f64(&r.0)),
        ExprData::Float(f) => Some(f.inner.to_f64()),
        // `π` resolves without a binding, as in every other evaluator here.
        ExprData::Symbol { name, .. } => env
            .get(&expr)
            .copied()
            .or_else(|| crate::eval::symbols::constant_value_f64(name)),
        ExprData::Add(args) => {
            let mut s = 0.0f64;
            for &a in args {
                s += try_expr_f64_snap(a, snap, env, memo)?;
            }
            Some(s)
        }
        ExprData::Mul(args) => {
            let mut p = 1.0f64;
            for &a in args {
                p *= try_expr_f64_snap(a, snap, env, memo)?;
            }
            Some(p)
        }
        ExprData::Pow { base, exp } => {
            let (base, exp) = (*base, *exp);
            let b = try_expr_f64_snap(base, snap, env, memo)?;
            let e = try_expr_f64_snap(exp, snap, env, memo)?;
            Some(pow_f64(b, e, || snap_data(snap, exp).cloned()))
        }
        ExprData::Func { name, args } => {
            let mut vals = Vec::with_capacity(args.len());
            for &a in args {
                vals.push(try_expr_f64_snap(a, snap, env, memo)?);
            }
            registry().numeric_f64(name.as_str(), &vals)
        }
        _ => None,
    };
    if let Some(v) = val {
        memo.insert(expr, v);
    }
    val
}

fn try_predicate_bool_snap(
    kind: &PredicateKind,
    args: &[ExprId],
    snap: &ExprSnapshot,
    env: &HashMap<ExprId, f64>,
    memo: &mut IdMap<f64>,
) -> Option<bool> {
    match kind {
        PredicateKind::True => Some(true),
        PredicateKind::False => Some(false),
        PredicateKind::Not => Some(!try_predicate_bool_snap_expr(args[0], snap, env, memo)?),
        PredicateKind::And => {
            for &a in args {
                if !try_predicate_bool_snap_expr(a, snap, env, memo)? {
                    return Some(false);
                }
            }
            Some(true)
        }
        PredicateKind::Or => {
            for &a in args {
                if try_predicate_bool_snap_expr(a, snap, env, memo)? {
                    return Some(true);
                }
            }
            Some(false)
        }
        PredicateKind::Lt => Some(
            try_expr_f64_snap(args[0], snap, env, memo)?
                < try_expr_f64_snap(args[1], snap, env, memo)?,
        ),
        PredicateKind::Le => Some(
            try_expr_f64_snap(args[0], snap, env, memo)?
                <= try_expr_f64_snap(args[1], snap, env, memo)?,
        ),
        PredicateKind::Gt => Some(
            try_expr_f64_snap(args[0], snap, env, memo)?
                > try_expr_f64_snap(args[1], snap, env, memo)?,
        ),
        PredicateKind::Ge => Some(
            try_expr_f64_snap(args[0], snap, env, memo)?
                >= try_expr_f64_snap(args[1], snap, env, memo)?,
        ),
        PredicateKind::Eq => Some(
            try_expr_f64_snap(args[0], snap, env, memo)?
                == try_expr_f64_snap(args[1], snap, env, memo)?,
        ),
        PredicateKind::Ne => Some(
            try_expr_f64_snap(args[0], snap, env, memo)?
                != try_expr_f64_snap(args[1], snap, env, memo)?,
        ),
    }
}

fn try_predicate_bool_snap_expr(
    expr: ExprId,
    snap: &ExprSnapshot,
    env: &HashMap<ExprId, f64>,
    memo: &mut IdMap<f64>,
) -> Option<bool> {
    match snap_data(snap, expr)? {
        ExprData::Predicate { kind, args } => try_predicate_bool_snap(kind, args, snap, env, memo),
        _ => None,
    }
}

fn eval_interp_snap(
    expr: ExprId,
    env: &HashMap<ExprId, f64>,
    snap: &ExprSnapshot,
    memo: &mut IdMap<f64>,
) -> Option<f64> {
    if let Some(&cached) = memo.get(&expr) {
        return Some(cached);
    }
    let val = match snap.nodes.get(&expr)? {
        ExprData::Integer(n) => Some(integer_to_f64(&n.0)),
        ExprData::Rational(r) => Some(rational_to_f64(&r.0)),
        ExprData::Float(f) => Some(f.inner.to_f64()),
        // Same rule as [`eval_interp_inner`].  A snapshot has no pool to ask,
        // so the name is read off the node directly — the two must agree or the
        // snapshot path would disagree with the interpreter it mirrors.
        ExprData::Symbol { name, .. } => env
            .get(&expr)
            .copied()
            .or_else(|| crate::eval::symbols::constant_value_f64(name)),
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
            let (base, exp) = (*base, *exp);
            let b = eval_interp_snap(base, env, snap, memo)?;
            let e = eval_interp_snap(exp, env, snap, memo)?;
            Some(pow_f64(b, e, || snap.nodes.get(&exp).cloned()))
        }
        ExprData::Func { name, args } => {
            let args = args.clone();
            let mut vals = Vec::with_capacity(args.len());
            for a in args {
                vals.push(eval_interp_snap(a, env, snap, memo)?);
            }
            registry().numeric_f64(name.as_str(), &vals)
        }
        ExprData::Piecewise { branches, default } => {
            for (c, v) in branches {
                match try_predicate_bool_snap_expr(*c, snap, env, memo) {
                    Some(true) => return eval_interp_snap(*v, env, snap, memo),
                    Some(false) => {}
                    None => return None,
                }
            }
            eval_interp_snap(*default, env, snap, memo)
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
    let mut visited = IdSet::default();
    let mut order = Vec::new();
    topo_dfs(root, pool, &mut visited, &mut order);
    order
}

#[cfg_attr(not(any(feature = "jit", feature = "cranelift")), allow(dead_code))]
fn topo_dfs(node: ExprId, pool: &ExprPool, visited: &mut IdSet, order: &mut Vec<ExprId>) {
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

    /// External symbol compiled powers call; mapped to [`llvm_pow`].
    const POW_SYMBOL: &str = "alkahest_pow_narrow";

    extern "C" fn llvm_pow(base: f64, exp: f64) -> f64 {
        crate::kernel::eval_const::pow_f64_narrow(base, exp)
    }

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

    /// Load input `k` of point `idx` from column base pointer `cols[k]`.
    fn load_col_inputs<'ctx>(
        builder: &Builder<'ctx>,
        f64_type: inkwell::types::FloatType<'ctx>,
        col_ptrs: &[inkwell::values::PointerValue<'ctx>],
        inputs: &[ExprId],
        idx: IntValue<'ctx>,
        values: &mut HashMap<ExprId, FloatValue<'ctx>>,
    ) -> Result<(), JitError> {
        for (i, (&var, &col)) in inputs.iter().zip(col_ptrs).enumerate() {
            let gep = unsafe {
                builder
                    .build_gep(f64_type, col, &[idx], &format!("col_in_{i}"))
                    .map_err(|e| JitError::CompilationFailed(e.to_string()))?
            };
            let val = builder
                .build_load(f64_type, gep, &format!("col_x_{i}"))
                .map_err(|e| JitError::CompilationFailed(e.to_string()))?
                .into_float_value();
            values.insert(var, val);
        }
        Ok(())
    }

    /// Emit the nodes of `expr` (inputs already in `values`) and return the
    /// root's value.
    fn emit_expr_values<'ctx>(
        expr: ExprId,
        pool: &ExprPool,
        builder: &Builder<'ctx>,
        module: &Module<'ctx>,
        ctx: &'ctx Context,
        function: FunctionValue<'ctx>,
        mut values: HashMap<ExprId, FloatValue<'ctx>>,
    ) -> Result<FloatValue<'ctx>, JitError> {
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
        let mut scalar_values = HashMap::new();
        load_scalar_inputs(
            &builder,
            f64_type,
            i64_type,
            scalar_inputs_ptr,
            inputs,
            &mut scalar_values,
        )?;
        let scalar_result =
            emit_expr_values(expr, pool, &builder, &module, ctx, scalar_fn, scalar_values)?;
        builder
            .build_return(Some(&scalar_result))
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

        // Columns: void alkahest_eval_cols(f64** cols, f64** outs, u64 start, u64 end)
        // — the same entry Cranelift emits ([`super::JitColsFn`]): the column
        // base pointers are loaded once, then points `start..end` run in one
        // loop, so a Rayon chunk is just a sub-range.
        let err = |e: inkwell::builder::BuilderError| JitError::CompilationFailed(e.to_string());
        let void_type = ctx.void_type();
        let cols_fn_type = void_type.fn_type(
            &[
                ptr_type.into(),
                ptr_type.into(),
                i64_type.into(),
                i64_type.into(),
            ],
            false,
        );
        let cols_fn = module.add_function("alkahest_eval_cols", cols_fn_type, None);
        let cols_entry = ctx.append_basic_block(cols_fn, "entry");
        let loop_hdr = ctx.append_basic_block(cols_fn, "loop_hdr");
        let loop_body = ctx.append_basic_block(cols_fn, "loop_body");
        let loop_exit = ctx.append_basic_block(cols_fn, "loop_exit");

        builder.position_at_end(cols_entry);
        let param = |i: u32| {
            cols_fn.get_nth_param(i).ok_or_else(|| {
                JitError::CompilationFailed("missing column entry parameter".to_string())
            })
        };
        let cols_arg = param(0)?.into_pointer_value();
        let outs_arg = param(1)?.into_pointer_value();
        let start_idx = param(2)?.into_int_value();
        let end_idx = param(3)?.into_int_value();
        let load_ptr = |base: inkwell::values::PointerValue<'static>, k: usize, name: &str| {
            let slot = unsafe {
                builder
                    .build_gep(ptr_type, base, &[i64_type.const_int(k as u64, false)], name)
                    .map_err(err)?
            };
            Ok::<_, JitError>(
                builder
                    .build_load(ptr_type, slot, name)
                    .map_err(err)?
                    .into_pointer_value(),
            )
        };
        let col_ptrs = (0..inputs.len())
            .map(|k| load_ptr(cols_arg, k, &format!("col_{k}")))
            .collect::<Result<Vec<_>, _>>()?;
        let out_ptr = load_ptr(outs_arg, 0, "out_0")?;
        builder.build_unconditional_branch(loop_hdr).map_err(err)?;

        builder.position_at_end(loop_hdr);
        let loop_idx = builder.build_phi(i64_type, "i").map_err(err)?;
        loop_idx.add_incoming(&[(&start_idx, cols_entry)]);
        let cur_idx = loop_idx.as_basic_value().into_int_value();
        let done = builder
            .build_int_compare(IntPredicate::UGE, cur_idx, end_idx, "done")
            .map_err(err)?;
        builder
            .build_conditional_branch(done, loop_exit, loop_body)
            .map_err(err)?;

        builder.position_at_end(loop_body);
        let mut point_values = HashMap::new();
        load_col_inputs(
            &builder,
            f64_type,
            &col_ptrs,
            inputs,
            cur_idx,
            &mut point_values,
        )?;
        let body_result =
            emit_expr_values(expr, pool, &builder, &module, ctx, cols_fn, point_values)?;
        let out_gep = unsafe {
            builder
                .build_gep(f64_type, out_ptr, &[cur_idx], "out_gep")
                .map_err(err)?
        };
        builder.build_store(out_gep, body_result).map_err(err)?;
        // The body may have added blocks (none today); the back edge comes
        // from wherever the builder ended up.
        let body_end = builder
            .get_insert_block()
            .ok_or_else(|| JitError::CompilationFailed("no insert block".to_string()))?;
        let next_idx = builder
            .build_int_add(cur_idx, i64_type.const_int(1, false), "next_i")
            .map_err(err)?;
        loop_idx.add_incoming(&[(&next_idx, body_end)]);
        builder.build_unconditional_branch(loop_hdr).map_err(err)?;

        builder.position_at_end(loop_exit);
        builder.build_return(None).map_err(err)?;

        if module.verify().is_err() {
            return Err(JitError::CompilationFailed(
                "LLVM module verification failed".to_string(),
            ));
        }

        let ee = module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
        if let Some(f) = module.get_function(POW_SYMBOL) {
            ee.add_global_mapping(&f, llvm_pow as *const () as usize);
        }

        let fn_ptr: super::JitScalarFn = unsafe {
            ee.get_function("alkahest_eval")
                .map_err(|e| JitError::CompilationFailed(e.to_string()))?
                .as_raw()
        };
        let cols_fn_ptr: super::JitColsFn = unsafe {
            ee.get_function("alkahest_eval_cols")
                .map_err(|e| JitError::CompilationFailed(e.to_string()))?
                .as_raw()
        };

        Ok(CompiledFn {
            inner: CompiledFnInner::Llvm {
                fn_ptr,
                cols_fn: cols_fn_ptr,
                execution_engine: ee,
                _context: unsafe { Box::from_raw(ctx as *const Context as *mut Context) },
            },
            n_inputs: inputs.len(),
            tier: super::CompileTier::Llvm,
            complex: None,
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
            ExprData::Integer(n) => Ok(f64_type.const_float(integer_to_f64(&n.0))),
            ExprData::Rational(r) => Ok(f64_type.const_float(rational_to_f64(&r.0))),
            ExprData::Float(f) => Ok(f64_type.const_float(f.inner.to_f64())),
            // `π` is a constant, not an input — as in the Cranelift backend
            // and the interpreter. Declared inputs were loaded before codegen,
            // so an input *named* `pi` still wins. The same holds for the
            // other named constants (`EulerGamma`, `Catalan`, `Glaisher`).
            ExprData::Symbol { name, .. } => {
                match crate::eval::symbols::constant_value_f64(&name) {
                    Some(v) => Ok(f64_type.const_float(v)),
                    None => Err(JitError::UnsupportedNode(format!(
                        "unbound symbol '{name}'"
                    ))),
                }
            }
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
                // Not `llvm.pow.f64`: every evaluator computes a power with
                // the shared kernel (`pow_f64_narrow` — compensated squaring
                // for small integer exponents, `sqrt` for 1/2), and the LLVM
                // tier must give the same bits. The symbol is mapped to that
                // Rust function once the execution engine exists.
                let pow_fn = get_intrinsic(
                    module,
                    ctx,
                    POW_SYMBOL,
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
                    "floor" => "llvm.floor.f64",
                    "ceil" => "llvm.ceil.f64",
                    "round" => "llvm.round.f64",
                    // No dedicated LLVM intrinsic; declared as external libm
                    // calls resolved by the JIT execution engine.
                    "sinh" => "sinh",
                    "cosh" => "cosh",
                    "tanh" => "tanh",
                    "asin" => "asin",
                    "acos" => "acos",
                    "atan" => "atan",
                    "asinh" => "asinh",
                    "acosh" => "acosh",
                    "atanh" => "atanh",
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

    #[test]
    fn interp_piecewise_positive_branch() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let pw = pool.piecewise(
            vec![(pool.pred_gt(x, pool.integer(0_i32)), x)],
            pool.integer(-1_i32),
        );
        let mut env = HashMap::new();
        env.insert(x, 2.0);
        let v = eval_interp(pw, &env, &pool).unwrap();
        assert!((v - 2.0).abs() < 1e-12);
    }

    #[test]
    fn compiled_piecewise_matches_interp() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let pw = pool.piecewise(
            vec![(pool.pred_gt(x, pool.integer(0_i32)), x)],
            pool.integer(-1_i32),
        );
        let f = compile(pw, &[x], &pool).unwrap();
        for xv in [-2.0, -0.5, 0.5, 3.0] {
            let mut env = HashMap::new();
            env.insert(x, xv);
            let direct = eval_interp(pw, &env, &pool).unwrap();
            let compiled = f.call(&[xv]);
            assert!(
                (direct - compiled).abs() < 1e-12,
                "piecewise mismatch at x={xv}: interp={direct} compiled={compiled}"
            );
            assert!(!compiled.is_nan());
        }
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

    /// `(x²-1)/(x-1)` evaluated AS WRITTEN at `x = 1` (no `cancel()` first)
    /// must not silently produce the limit `2`. Under raw IEEE-754 semantics
    /// this is `0 * (1/0) = NaN`; `eval_interp` (unchecked) still returns
    /// that `NaN` — mirroring plain float arithmetic — but
    /// `eval_interp_checked` must refuse it as [`InterpEvalError::NonFinite`]
    /// rather than let a caller mistake `NaN` for a number.
    #[test]
    fn eval_interp_checked_rejects_removable_singularity_evaluated_as_written() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let numerator = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-1_i32)]);
        let denominator = pool.add(vec![x, pool.integer(-1_i32)]);
        let inv_denominator = pool.pow(denominator, pool.integer(-1_i32));
        let f = pool.mul(vec![numerator, inv_denominator]);

        let mut env = HashMap::new();
        env.insert(x, 1.0);

        assert!(eval_interp(f, &env, &pool).unwrap().is_nan());
        assert_eq!(
            eval_interp_checked(f, &env, &pool),
            Err(InterpEvalError::NonFinite)
        );

        // Away from the pole, both agree with the algebraic simplification.
        env.insert(x, 3.0);
        assert_eq!(eval_interp_checked(f, &env, &pool), Ok(4.0));
    }

    /// After an explicit `cancel()`-style rewrite to `x + 1`, evaluating at
    /// `x = 1` legitimately gives `2` — `eval_interp_checked` must not
    /// refuse ordinary, well-defined evaluation.
    #[test]
    fn eval_interp_checked_accepts_cancelled_form_at_former_pole() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let cancelled = pool.add(vec![x, pool.integer(1_i32)]);

        let mut env = HashMap::new();
        env.insert(x, 1.0);
        assert_eq!(eval_interp_checked(cancelled, &env, &pool), Ok(2.0));
    }

    #[test]
    fn eval_interp_checked_reports_unbound_symbol() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let env = HashMap::new();
        assert_eq!(
            eval_interp_checked(x, &env, &pool),
            Err(InterpEvalError::Unevaluable)
        );
    }

    /// Deeply shared DAG: 20 levels of squaring produces 21 nodes but
    /// 2^20 tree references — must terminate in O(n) time.
    #[test]
    fn eval_interp_keeps_the_parity_of_an_exponent_wider_than_f64() {
        use rug::{ops::Pow, Integer};
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let n = Integer::from(10).pow(30) + 1u32;
        let expr = p.pow(x, p.integer(n.clone()));
        let env = HashMap::from([(x, -1.0)]);

        // The exponent is odd; its `f64` image `1e30` is even, and `eval_expr`
        // returned `1.0` for a value that is `-1`.
        assert_eq!(integer_to_f64(&n), 1e30);
        assert_eq!(eval_interp(expr, &env, &p), Some(-1.0));
        assert_eq!(eval_interp_checked(expr, &env, &p), Ok(-1.0));

        // Controls: the even neighbour, a small odd exponent, and an exponent
        // past 2^53 that is still exactly representable.
        let even = p.pow(x, p.integer(Integer::from(10).pow(30)));
        assert_eq!(eval_interp(even, &env, &p), Some(1.0));
        assert_eq!(
            eval_interp(p.pow(x, p.integer(3_i32)), &env, &p),
            Some(-1.0)
        );
        let wide_even = p.pow(x, p.integer(Integer::from(1u64 << 60)));
        assert_eq!(eval_interp(wide_even, &env, &p), Some(1.0));

        // A magnitude `f64` cannot hold stays a refusal rather than becoming a
        // signed guess.
        assert_eq!(
            eval_interp_checked(expr, &HashMap::from([(x, 2.0)]), &p),
            Err(InterpEvalError::NonFinite)
        );
    }

    #[test]
    fn a_wide_integer_exponent_is_evaluated_by_the_interpreter_not_a_backend() {
        use rug::{ops::Pow, Integer};
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let wide = p.pow(x, p.integer(Integer::from(10).pow(30) + 1u32));
        // A batch hint big enough that tier selection reaches for a native
        // backend wherever one is compiled in. Both lower the exponent to an
        // `f64` constant, so neither can be trusted with this node.
        let expr = p.add(vec![wide, p.mul(vec![x, x])]);
        let f = compile_with(expr, &[x], &p, CompileConfig::for_batch(1 << 20)).expect("compile");
        assert_eq!(f.compile_tier(), CompileTier::Interpreter);
        assert_eq!(f.call(&[-1.0]), 0.0, "-1 + 1");
        assert!(needs_exact_integer_exponent(expr, &p));

        // Control: nothing else is downgraded, including a wide exponent that
        // `f64` *can* hold exactly.
        let plain = p.add(vec![p.pow(x, p.integer(19_i32)), p.mul(vec![x, x])]);
        assert!(!needs_exact_integer_exponent(plain, &p));
        assert!(!needs_exact_integer_exponent(
            p.pow(x, p.integer(Integer::from(1u64 << 60))),
            &p
        ));
    }

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

    /// The interpreter tier's flat program and the snapshot walker it replaced
    /// for straight-line expressions agree bit for bit through every entry
    /// point (scalar, bulk, batch, parallel batch).
    #[test]
    fn interpreter_tier_entry_points_agree_with_eval_interp() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![
            pool.mul(vec![
                pool.func("sin", vec![x]),
                pool.func(
                    "exp",
                    vec![pool.mul(vec![pool.rational(-1, 2), pool.pow(x, pool.integer(2_i32))])],
                ),
            ]),
            pool.pow(x, pool.integer(3_i32)),
            pool.mul(vec![pool.integer(-2_i32), x, y]),
            pool.pow(y, pool.integer(2_i32)),
            pool.func("log", vec![y]), // NaN for y < 0
        ]);
        let f = compile(expr, &[x, y], &pool).unwrap();
        assert_eq!(f.compile_tier(), CompileTier::Interpreter);
        assert!(matches!(f.inner, CompiledFnInner::Program(_)));

        const N: usize = 5_000;
        let xs: Vec<f64> = (0..N).map(|i| (i as f64 - 2500.0) / 700.0).collect();
        let ys: Vec<f64> = (0..N).map(|i| (i as f64 - 1000.0) / 900.0).collect();
        let bits = |v: f64| if v.is_nan() { u64::MAX } else { v.to_bits() };
        let want: Vec<u64> = (0..N)
            .map(|j| {
                let env = HashMap::from([(x, xs[j]), (y, ys[j])]);
                bits(eval_interp(expr, &env, &pool).unwrap_or(f64::NAN))
            })
            .collect();

        let scalar: Vec<u64> = (0..N).map(|j| bits(f.call(&[xs[j], ys[j]]))).collect();
        assert_eq!(scalar, want);

        let mut out = vec![0.0; N];
        f.call_batch(&[&xs, &ys], &mut out);
        assert_eq!(out.iter().map(|&v| bits(v)).collect::<Vec<_>>(), want);

        let mut flat = xs.clone();
        flat.extend_from_slice(&ys);
        let mut out = vec![0.0; N];
        f.call_bulk(&flat, &mut out);
        assert_eq!(out.iter().map(|&v| bits(v)).collect::<Vec<_>>(), want);

        #[cfg(feature = "parallel")]
        {
            let mut out = vec![0.0; N];
            f.call_batch_par(&[&xs, &ys], &mut out);
            assert_eq!(out.iter().map(|&v| bits(v)).collect::<Vec<_>>(), want);
        }
    }

    /// Two programs of the same slot count share the thread-local scratch;
    /// each must see its own constants.
    #[test]
    fn interpreter_tier_scalar_calls_do_not_leak_constants_between_functions() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = compile(pool.add(vec![x, pool.integer(1_i32)]), &[x], &pool).unwrap();
        let g = compile(pool.add(vec![x, pool.integer(5_i32)]), &[x], &pool).unwrap();
        for _ in 0..3 {
            assert_eq!(f.call(&[1.0]), 2.0);
            assert_eq!(g.call(&[1.0]), 6.0);
        }
    }

    #[test]
    fn sampler_matches_eval_interp_including_fallback_forms() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let straight = pool.add(vec![
            pool.func("tan", vec![pool.mul(vec![a, x])]),
            pool.pow(x, pool.rational(-1, 3)),
        ]);
        let piecewise = pool.piecewise(
            vec![(
                pool.pred_lt(x, pool.integer(0_i32)),
                pool.func("sin", vec![x]),
            )],
            pool.mul(vec![a, x]),
        );
        let fixed = HashMap::from([(a, 1.7), (x, 99.0)]); // `x` is overridden
        for expr in [straight, piecewise] {
            let s = Sampler::new(expr, x, &fixed, &pool);
            for t in [-2.0, -0.5, 0.0, 0.25, 1.0, 3.0] {
                let env = HashMap::from([(a, 1.7), (x, t)]);
                let want = eval_interp(expr, &env, &pool);
                let got = s.eval(t);
                assert!(
                    want.map(f64::to_bits) == got.map(f64::to_bits)
                        || (want.is_some_and(f64::is_nan) && got.is_some_and(f64::is_nan)),
                    "{} at {t}: {want:?} vs {got:?}",
                    pool.display(expr)
                );
            }
        }
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

    // -----------------------------------------------------------------
    // Registry / eval_interp sync test
    // -----------------------------------------------------------------

    /// Probe argument lists, ordered to prefer well-behaved (in-domain)
    /// inputs for every registered primitive — mirrors the probe sets used
    /// by `primitive::probe_caps`.
    const PROBE_ARG_SETS: &[&[f64]] = &[
        &[0.5],
        &[0.5, 0.3],
        &[0.2, 0.3, 0.4],
        &[1.0],
        &[1.0, 2.0],
        &[1.0, 2.0, 3.0],
        // Four arguments: `jacobi_p(n, a, b, x)`.
        &[2.0, 0.5, 0.5, 0.3],
        // `hyper(0, 1, b, z)`: ₀F₁(; 1; 0.5).
        &[0.0, 1.0, 1.0, 0.5],
        // `lambert_wm1` is real only on [−1/e, 0).
        &[-0.2],
    ];

    /// Every primitive in [`PrimitiveRegistry::default_registry`] that has a
    /// `numeric_f64` kernel must be evaluable through [`eval_interp`] when
    /// expressed as `Func { name, args }`.
    ///
    /// `diracdelta` is the documented exception: it is a distribution with no
    /// pointwise `f64` value (see the `primitive::mod` coverage doctest and
    /// `tests/test_v10.py::test_primitive_registry_all_have_numeric_f64`).
    #[test]
    fn eval_interp_covers_all_numeric_f64_primitives() {
        use crate::primitive::{Capabilities, PrimitiveRegistry};

        let reg = PrimitiveRegistry::default_registry();
        let pool = p();

        let mut missing = Vec::new();

        for (name, caps) in reg.iter() {
            if !caps.contains(Capabilities::NUMERIC_F64) {
                continue;
            }
            if name == "diracdelta" {
                continue;
            }

            // Find an arg-count / value combination this primitive accepts.
            let Some(probe) = PROBE_ARG_SETS
                .iter()
                .find(|args| reg.numeric_f64(name, args).is_some())
            else {
                // Registered as NUMERIC_F64 but none of our probe sets work —
                // treat as a coverage gap to investigate rather than silently
                // skipping.
                missing.push(format!("{name} (no working probe args)"));
                continue;
            };

            let expected = reg.numeric_f64(name, probe).unwrap();

            let arg_ids: Vec<ExprId> = probe.iter().map(|&v| pool.float(v, 53)).collect();
            let expr = pool.func(name, arg_ids);

            let env = HashMap::new();
            match eval_interp(expr, &env, &pool) {
                Some(got) => {
                    if (got - expected).abs() > 1e-9 {
                        missing.push(format!(
                            "{name}: eval_interp({probe:?}) = {got}, \
                             registry.numeric_f64 = {expected}"
                        ));
                    }
                }
                None => missing.push(format!(
                    "{name}: eval_interp returned None for {probe:?} \
                     (registry.numeric_f64 = {expected})"
                )),
            }
        }

        assert!(
            missing.is_empty(),
            "eval_interp is missing coverage for registered NUMERIC_F64 \
             primitives:\n{}",
            missing.join("\n")
        );
    }

    #[test]
    fn eval_interp_unary_heads() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let mut env = HashMap::new();
        env.insert(x, 0.5_f64);

        let cases: &[(&str, f64)] = &[
            ("sinh", 0.5_f64.sinh()),
            ("cosh", 0.5_f64.cosh()),
            ("tanh", 0.5_f64.tanh()),
            ("asin", 0.5_f64.asin()),
            ("acos", 0.5_f64.acos()),
            ("atan", 0.5_f64.atan()),
            ("sign", 1.0),
            ("floor", 0.0),
            ("ceil", 1.0),
            ("round", 1.0),
        ];
        for &(name, expected) in cases {
            let expr = pool.func(name, vec![x]);
            let got = eval_interp(expr, &env, &pool)
                .unwrap_or_else(|| panic!("eval_interp returned None for {name}"));
            assert!(
                (got - expected).abs() < 1e-12,
                "{name}(0.5): got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn eval_interp_erf_erfc_gamma() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let mut env = HashMap::new();
        env.insert(x, 1.0_f64);

        let erf = eval_interp(pool.func("erf", vec![x]), &env, &pool).unwrap();
        let erfc = eval_interp(pool.func("erfc", vec![x]), &env, &pool).unwrap();
        assert!((erf + erfc - 1.0).abs() < 1e-9, "erf+erfc should be 1");

        let gamma = eval_interp(pool.func("gamma", vec![x]), &env, &pool).unwrap();
        assert!((gamma - 1.0).abs() < 1e-9, "Γ(1) = 1, got {gamma}");
    }

    #[test]
    fn eval_interp_heaviside() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);

        // θ(−1) = 0, θ(0) = 1/2, θ(1) = 1 — matches `HeavisidePrimitive`.
        for (xv, expected) in [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)] {
            let mut env = HashMap::new();
            env.insert(x, xv);
            let expr = pool.func("heaviside", vec![x]);
            let got = eval_interp(expr, &env, &pool).unwrap();
            assert_eq!(got, expected, "heaviside({xv})");
        }
    }

    #[test]
    fn eval_interp_binary_and_ternary_heads() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let mut env = HashMap::new();
        env.insert(x, 1.0_f64);
        env.insert(y, 2.0_f64);

        let atan2 = eval_interp(pool.func("atan2", vec![x, y]), &env, &pool).unwrap();
        assert!((atan2 - 1.0_f64.atan2(2.0)).abs() < 1e-12);

        let min = eval_interp(pool.func("min", vec![x, y]), &env, &pool).unwrap();
        assert_eq!(min, 1.0);

        let max = eval_interp(pool.func("max", vec![x, y]), &env, &pool).unwrap();
        assert_eq!(max, 2.0);

        // EllipticK(0) = EllipticE(0) = π/2.
        let zero = pool.integer(0_i32);
        let env = HashMap::new();
        let k0 = eval_interp(pool.func("EllipticK", vec![zero]), &env, &pool).unwrap();
        let e0 = eval_interp(pool.func("EllipticE", vec![zero]), &env, &pool).unwrap();
        let half_pi = std::f64::consts::PI / 2.0;
        assert!((k0 - half_pi).abs() < 1e-9);
        assert!((e0 - half_pi).abs() < 1e-9);
    }
}

/// Constant functions in batch, and unbound symbols at compile time (audit A11).
#[cfg(test)]
mod constant_and_unbound_tests {
    use super::*;
    use crate::errors::AlkahestError;
    use crate::kernel::Domain;

    /// Every tier this build has, forced, for `expr` over `inputs`.
    fn every_tier(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> Vec<CompiledFn> {
        #[allow(unused_mut)]
        let mut tiers = vec![CompileTier::Interpreter];
        #[cfg(feature = "cranelift")]
        tiers.push(CompileTier::Cranelift);
        #[cfg(feature = "jit")]
        tiers.push(CompileTier::Llvm);
        tiers
            .into_iter()
            .map(|t| {
                let config = CompileConfig {
                    expected_evals: None,
                    force_tier: Some(t),
                };
                let f = compile_with(expr, inputs, pool, config).expect("compiles");
                assert_eq!(f.compile_tier(), t, "the forced tier was used");
                f
            })
            .collect()
    }

    fn assert_batches_to(f: &CompiledFn, want: f64) {
        let tier = f.compile_tier();
        assert_eq!(f.call(&[]), want, "{tier:?}: scalar call");

        let mut out = vec![0.0; 3];
        f.call_batch(&[], &mut out);
        assert_eq!(out, [want; 3], "{tier:?}: call_batch of a constant");

        let mut out = vec![0.0; 3];
        f.call_bulk(&[], &mut out);
        assert_eq!(out, [want; 3], "{tier:?}: call_bulk of a constant");

        #[cfg(feature = "parallel")]
        {
            // More than one Rayon chunk, so every chunk is checked.
            let mut out = vec![0.0; 2 * PAR_CHUNK + 3];
            f.call_batch_par(&[], &mut out);
            assert!(
                out.iter().all(|&v| v == want),
                "{tier:?}: call_batch_par of a constant"
            );
        }
    }

    /// `call_batch` returned early for a function of no inputs, leaving the
    /// caller's zero-filled buffer as the "answer": `[0, 0, 0]` for `5`.
    #[test]
    fn a_function_of_no_inputs_batches_to_its_constant() {
        let pool = ExprPool::new();
        let five = pool.integer(5_i32);
        for f in every_tier(five, &[], &pool) {
            assert_batches_to(&f, 5.0);
        }
        // `π` resolves to its value.
        let pi = pool.symbol(crate::eval::symbols::PI_NAME, Domain::Real);
        for f in every_tier(pi, &[], &pool) {
            assert_batches_to(&f, std::f64::consts::PI);
        }
        // A Piecewise takes the lazy block program ...
        let constant_piecewise = pool.piecewise(
            vec![(pool.pred_gt(pool.integer(2_i32), pool.integer(1_i32)), five)],
            pool.integer(-1_i32),
        );
        let f = compile(constant_piecewise, &[], &pool).unwrap();
        assert!(matches!(f.inner, CompiledFnInner::Program(_)));
        assert_batches_to(&f, 5.0);
        // ... including one compared inside a condition, which the closure
        // path's walk had no value for (`NaN`) and `eval_interp` does.
        let nested = pool.piecewise(
            vec![(pool.pred_lt(constant_piecewise, pool.integer(7_i32)), five)],
            pool.integer(-1_i32),
        );
        let f = compile(nested, &[], &pool).unwrap();
        assert!(matches!(f.inner, CompiledFnInner::Program(_)));
        assert_batches_to(&f, 5.0);
        // A condition that is not a predicate keeps the closure path, where
        // it has no value — at a point and in a batch alike.
        let bare = pool.piecewise(vec![(five, five)], pool.integer(-1_i32));
        let f = compile(bare, &[], &pool).unwrap();
        assert!(matches!(f.inner, CompiledFnInner::Interpreter(_)));
        assert!(f.call(&[]).is_nan());
        let mut out = vec![0.0; 3];
        f.call_batch(&[], &mut out);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    /// `compile(x + y, [x])` used to succeed and return `NaN` at every point.
    #[test]
    fn compile_refuses_a_symbol_the_inputs_do_not_bind() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let expr = pool.add(vec![x, pool.mul(vec![y, z]), y]);

        let err = compile(expr, &[x], &pool).err().expect("refused");
        assert!(matches!(err, JitError::UnsupportedNode(_)), "{err:?}");
        assert!(err.to_string().contains("y, z"), "{err}");
        let refusal = take_unbound_symbols().expect("recorded out of band");
        assert_eq!(refusal.code(), "E-JIT-005");
        assert_eq!(refusal.names(), ["y", "z"]);
        assert!(take_unbound_symbols().is_none(), "consumed");

        // Every entry point and every forced tier refuses, not only the default.
        for tier in [Some(CompileTier::Interpreter), None] {
            let config = CompileConfig {
                expected_evals: Some(1 << 20),
                force_tier: tier,
            };
            assert!(compile_with(expr, &[x], &pool, config).is_err());
            assert!(take_unbound_symbols().is_some());
        }
        let mut cache = CompileCache::new();
        assert!(cache.compile(expr, &[x], &pool).is_err());
        assert!(take_unbound_symbols().is_some());
        #[cfg(any(feature = "cranelift", feature = "jit"))]
        {
            assert!(compile_jit_only(expr, &[x], &pool).is_err());
            assert!(take_unbound_symbols().is_some());
        }

        // Bound, it compiles, and the success leaves no refusal behind.
        let f = compile(expr, &[x, y, z], &pool).unwrap();
        assert_eq!(f.call(&[1.0, 2.0, 3.0]), 9.0);
        assert!(take_unbound_symbols().is_none());
    }

    #[test]
    fn what_does_not_need_binding() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // π is a constant.
        let pi = pool.symbol(crate::eval::symbols::PI_NAME, Domain::Real);
        let sin_pi_x = pool.func("sin", vec![pool.mul(vec![pi, x])]);
        assert_eq!(check_inputs_bind(sin_pi_x, &[x], &pool), Ok(()));
        assert!(compile(sin_pi_x, &[x], &pool).is_ok());
        // A RootSum's placeholder is bound by the RootSum, not by the inputs...
        let c = pool.symbol("c", Domain::Complex);
        let poly = pool.add(vec![pool.pow(c, pool.integer(2_i32)), pool.integer(1_i32)]);
        let body = pool.func(
            "log",
            vec![pool.add(vec![x, pool.mul(vec![pool.integer(-1_i32), c])])],
        );
        let rs = pool.root_sum(poly, c, body);
        assert_eq!(check_inputs_bind(rs, &[x], &pool), Ok(()));
        // ...but the same symbol outside the binder is free.
        let both = pool.add(vec![rs, c]);
        assert_eq!(
            check_inputs_bind(both, &[x], &pool).unwrap_err().names(),
            ["c"]
        );
    }

    #[test]
    fn a_same_named_symbol_of_another_domain_is_named_as_such() {
        let pool = ExprPool::new();
        let x_real = pool.symbol("x", Domain::Real);
        let x_complex = pool.symbol("x", Domain::Complex);
        let expr = pool.add(vec![x_complex, pool.integer(1_i32)]);
        let err = check_inputs_bind(expr, &[x_real], &pool).unwrap_err();
        assert_eq!(err.names(), ["x"]);
        assert!(err.to_string().contains("different domain"), "{err}");
    }

    /// Only a symbol binds: the evaluators look values up at `Symbol` nodes,
    /// so an input `x + 1` leaves the `x` inside it unbound.
    #[test]
    fn a_non_symbol_input_binds_nothing() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let x1 = pool.add(vec![x, pool.integer(1_i32)]);
        assert_eq!(
            check_inputs_bind(x1, &[x1], &pool).unwrap_err().names(),
            ["x"]
        );
    }
}
