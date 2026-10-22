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
    registry, topo_sort, CompileTier, CompiledFn, CompiledFnInner, JitColsFn, JitError, JitScalarFn,
};
use crate::kernel::eval_const::{
    has_hw_fma, pow_f64_narrow, powi_with, small_integer_exponent, sqrt_pow_with, FloatOps,
};
use crate::kernel::{integer_to_f64, rational_to_f64, ExprData, ExprId, ExprPool};
use crate::primitive::Primitive;
use cranelift_codegen::ir::{
    condcodes::{FloatCC, IntCC},
    types, AbiParam, BlockArg, InstBuilder, MemFlags, StackSlot, StackSlotData, StackSlotKind,
    Value,
};
use cranelift_codegen::{settings, settings::Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

// ---------------------------------------------------------------------------
// Owned JIT module — frees its code pages on drop
// ---------------------------------------------------------------------------

/// A [`JITModule`] that returns its memory to the OS when it is dropped.
///
/// `cranelift_jit` cannot know whether a function pointer it handed out is
/// still in use, so its own `Drop` deliberately *leaks* every mapping
/// (`mem::forget` on each allocation) and leaves releasing them to the
/// `unsafe` [`JITModule::free_memory`]. Nothing called that, so every
/// compile-and-drop kept its code, read-only and writable pages mapped for
/// the life of the process: about 4 kB resident per compile, so 20 000
/// compiles grew the process by 80 MB.
///
/// Freeing is sound here because of where the pointers live. The scalar and
/// bulk entry points of a compiled function are stored only inside the
/// [`CompiledFnInner::Cranelift`] variant that also owns this module; they
/// are never handed out, and every call goes through `&CompiledFn`. The module
/// is therefore dropped only once no call can be in flight and none can start.
///
/// The wrapper is taken on as soon as the module exists, so a compile that
/// fails half-way (after some functions were already defined) frees its pages
/// too.
pub(super) struct OwnedJitModule(Option<JITModule>);

impl OwnedJitModule {
    fn new(module: JITModule) -> Self {
        Self(Some(module))
    }
}

impl Deref for OwnedJitModule {
    type Target = JITModule;
    fn deref(&self) -> &JITModule {
        self.0.as_ref().expect("JIT module used after it was freed")
    }
}

impl DerefMut for OwnedJitModule {
    fn deref_mut(&mut self) -> &mut JITModule {
        self.0.as_mut().expect("JIT module used after it was freed")
    }
}

impl Drop for OwnedJitModule {
    fn drop(&mut self) {
        if let Some(module) = self.0.take() {
            // SAFETY: see the type's documentation — every pointer into this
            // module's memory is owned by the value being dropped.
            unsafe { module.free_memory() };
            #[cfg(test)]
            FREED_MODULES.with(|n| n.set(n.get() + 1));
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Modules freed on this thread — lets a test observe the release path
    /// without depending on RSS, which parallel tests make noisy.
    static FREED_MODULES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

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
/// `pow_f64_narrow`: the interpreter's kernel for an exponent known only at
/// run time (a constant exponent is specialised at compile time instead).
extern "C" fn tramp_pow(base: f64, exp: f64) -> f64 {
    pow_f64_narrow(base, exp)
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
extern "C" fn tramp_fma(a: f64, b: f64, c: f64) -> f64 {
    crate::kernel::eval_const::fma_exact(a, b, c)
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
// Registry primitives with no dedicated trampoline
// ---------------------------------------------------------------------------

/// Evaluate a one-argument registry primitive for compiled code.
///
/// `prim` points at a `&'static dyn Primitive` owned by the compiled function
/// (see [`CraneliftCode::prims`]); `fail` is set to 1 when the kernel has no
/// value at this point — the interpreter's `None`, which turns the whole
/// evaluation into `NaN` — and to 0 otherwise. A panic in a kernel is caught
/// here (unwinding out of an `extern "C"` function would abort the process)
/// and reported as a failure.
extern "C" fn tramp_prim1(prim: *const &'static dyn Primitive, a: f64, fail: *mut u8) -> f64 {
    // SAFETY: `prim` is the address of an element of the boxed slice that the
    // compiled function owns for as long as its code can run; `fail` is a
    // stack slot of the calling frame.
    let p: &'static dyn Primitive = unsafe { *prim };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.numeric_f64(&[a])))
        .ok()
        .flatten();
    unsafe { *fail = u8::from(r.is_none()) };
    r.unwrap_or(f64::NAN)
}

/// [`tramp_prim1`] for any other arity: `args` points at `n` values.
extern "C" fn tramp_primn(
    prim: *const &'static dyn Primitive,
    args: *const f64,
    n: u64,
    fail: *mut u8,
) -> f64 {
    // SAFETY: as in `tramp_prim1`; `args` is a stack slot of the calling frame
    // holding `n` initialised values.
    let p: &'static dyn Primitive = unsafe { *prim };
    let vals = unsafe { std::slice::from_raw_parts(args, n as usize) };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.numeric_f64(vals)))
        .ok()
        .flatten();
    unsafe { *fail = u8::from(r.is_none()) };
    r.unwrap_or(f64::NAN)
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
    prim1_id: FuncId,
    primn_id: FuncId,
}

impl MathFuncIds {
    /// The dedicated trampoline for a one-argument head, if it has one.
    fn unary(&self, name: &str) -> Option<FuncId> {
        Some(match name {
            "sin" => self.sin_id,
            "cos" => self.cos_id,
            "exp" => self.exp_id,
            "log" => self.log_id,
            "sqrt" => self.sqrt_id,
            "tan" => self.tan_id,
            "abs" => self.abs_id,
            "sinh" => self.sinh_id,
            "cosh" => self.cosh_id,
            "tanh" => self.tanh_id,
            "asin" => self.asin_id,
            "acos" => self.acos_id,
            "atan" => self.atan_id,
            "asinh" => self.asinh_id,
            "acosh" => self.acosh_id,
            "atanh" => self.atanh_id,
            "floor" => self.floor_id,
            "ceil" => self.ceil_id,
            "round" => self.round_id,
            "sign" => self.sign_id,
            _ => return None,
        })
    }
}

/// True when `node` is a `Func` that is evaluated through its registry
/// primitive (no dedicated trampoline), and so can fail at a point.
fn is_generic_call(node: ExprId, pool: &ExprPool, math: &MathFuncIds) -> bool {
    pool.with(node, |d| match d {
        ExprData::Func { name, .. } if name == crate::poly::root_of::ROOT_OF => false,
        ExprData::Func { name, args } => !(args.len() == 1 && math.unary(name).is_some()),
        _ => false,
    })
}

// ---------------------------------------------------------------------------
// FloatOps over Cranelift IR — the shared power kernels, emitted inline
// ---------------------------------------------------------------------------

struct ClifOps<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
}

impl FloatOps for ClifOps<'_, '_> {
    type V = Value;
    fn konst(&mut self, v: f64) -> Value {
        self.builder.ins().f64const(v)
    }
    fn add(&mut self, a: Value, b: Value) -> Value {
        self.builder.ins().fadd(a, b)
    }
    fn mul(&mut self, a: Value, b: Value) -> Value {
        self.builder.ins().fmul(a, b)
    }
    fn div(&mut self, a: Value, b: Value) -> Value {
        self.builder.ins().fdiv(a, b)
    }
    fn neg(&mut self, a: Value) -> Value {
        self.builder.ins().fneg(a)
    }
    fn fma(&mut self, a: Value, b: Value, c: Value) -> Value {
        self.builder.ins().fma(a, b, c)
    }
    fn sqrt(&mut self, a: Value) -> Value {
        self.builder.ins().sqrt(a)
    }
    fn select_finite(&mut self, t: Value, a: Value, b: Value) -> Value {
        let abs = self.builder.ins().fabs(t);
        let inf = self.builder.ins().f64const(f64::INFINITY);
        let c = self.builder.ins().fcmp(FloatCC::LessThan, abs, inf);
        self.builder.ins().select(c, a, b)
    }
    fn select_finite_nonzero(&mut self, t: Value, a: Value, b: Value) -> Value {
        let abs = self.builder.ins().fabs(t);
        let inf = self.builder.ins().f64const(f64::INFINITY);
        let zero = self.builder.ins().f64const(0.0);
        let finite = self.builder.ins().fcmp(FloatCC::LessThan, abs, inf);
        // `NotEqual` is true for NaN, which `finite` excludes anyway (and
        // matches Rust's `!=`); `OrderedNotEqual` has no aarch64 lowering.
        let nonzero = self.builder.ins().fcmp(FloatCC::NotEqual, t, zero);
        let c = self.builder.ins().band(finite, nonzero);
        self.builder.ins().select(c, a, b)
    }
    fn select_neg_inf(&mut self, t: Value, a: Value, b: Value) -> Value {
        let ninf = self.builder.ins().f64const(f64::NEG_INFINITY);
        let c = self.builder.ins().fcmp(FloatCC::Equal, t, ninf);
        self.builder.ins().select(c, a, b)
    }
}

// ---------------------------------------------------------------------------
// codegen_node — emit Cranelift IR for a single DAG node
// ---------------------------------------------------------------------------

/// Everything node emission needs besides the builder.
struct Codegen<'a> {
    pool: &'a ExprPool,
    math: &'a MathFuncIds,
    /// Address of each generic primitive's slot in [`CraneliftCode::prims`].
    prim_addr: &'a HashMap<String, i64>,
    /// One 8-byte slot the generic trampolines report failure through, and
    /// one for their arguments (sized for the widest call), if any.
    fail_slot: Option<StackSlot>,
    args_slot: Option<StackSlot>,
}

/// The emitted value of a node, and — for a generic call — its failure flag.
struct Emitted {
    value: Value,
    fail: Option<Value>,
}

fn codegen_node(
    node: ExprId,
    cg: &Codegen<'_>,
    values: &HashMap<ExprId, Value>,
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
) -> Result<Emitted, JitError> {
    let pool = cg.pool;
    let math = cg.math;
    let get = |id: &ExprId, what: &str| -> Result<Value, JitError> {
        values
            .get(id)
            .copied()
            .ok_or_else(|| JitError::CompilationFailed(format!("missing {what}")))
    };
    let plain = |value| Ok(Emitted { value, fail: None });
    match pool.get(node) {
        ExprData::Integer(n) => plain(builder.ins().f64const(integer_to_f64(&n.0))),
        ExprData::Rational(r) => plain(builder.ins().f64const(rational_to_f64(&r.0))),
        ExprData::Float(f) => plain(builder.ins().f64const(f.inner.to_f64())),
        // `π` is a constant, not an input.  `load_inputs` has already
        // populated `values` for every declared input and the caller skips
        // those nodes, so an input *named* `pi` still wins; this arm is only
        // reached for a `pi` nobody declared.  Without it the compiled function
        // and [`crate::jit::eval_interp`] disagree about `sin(π·x)` — one
        // answers, the other refuses — and the agreement tests exist precisely
        // to stop that. `EulerGamma`, `Catalan` and `Glaisher` likewise.
        ExprData::Symbol { name, .. } => match crate::eval::symbols::constant_value_f64(&name) {
            Some(v) => plain(builder.ins().f64const(v)),
            None => Err(JitError::UnsupportedNode(format!(
                "unbound symbol '{name}'"
            ))),
        },
        ExprData::Add(args) => {
            // Seeded with +0.0 like the interpreter, so an all-(-0.0) sum is
            // +0.0 in both.
            let mut acc = builder.ins().f64const(0.0);
            for a in &args {
                let v = get(a, "Add child")?;
                acc = builder.ins().fadd(acc, v);
            }
            plain(acc)
        }
        ExprData::Mul(args) => {
            // The interpreter's `1.0 · a · …` seed is the identity on every
            // value (`1·x` is `x`, signed zeros and NaN included), so it is
            // not emitted.
            let mut acc: Option<Value> = None;
            for a in &args {
                let v = get(a, "Mul child")?;
                acc = Some(match acc {
                    None => v,
                    Some(p) => builder.ins().fmul(p, v),
                });
            }
            plain(acc.unwrap_or_else(|| builder.ins().f64const(1.0)))
        }
        ExprData::Pow { base, exp } => {
            let b = get(&base, "Pow base")?;
            // A constant exponent picks its kernel now, exactly as
            // `pow_f64_narrow` would at run time.
            let konst = match pool.get(exp) {
                ExprData::Integer(n) => Some(integer_to_f64(&n.0)),
                ExprData::Rational(r) => Some(rational_to_f64(&r.0)),
                ExprData::Float(f) => Some(f.inner.to_f64()),
                _ => None,
            };
            if let Some(v) = konst {
                if let Some(n) = small_integer_exponent(v) {
                    // Exactly the choice `powi_f64` makes at run time.
                    if (-1..=2).contains(&n) || has_hw_fma() {
                        return plain(powi_with(&mut ClifOps { builder }, b, n));
                    }
                }
                if v == 0.5 {
                    return plain(sqrt_pow_with(&mut ClifOps { builder }, b));
                }
            }
            let e = get(&exp, "Pow exp")?;
            let pow_ref = module.declare_func_in_func(math.pow_id, builder.func);
            let call = builder.ins().call(pow_ref, &[b, e]);
            plain(builder.inst_results(call)[0])
        }
        // A real algebraic number is a constant, computed now exactly as the
        // interpreter does; a non-real one has no `f64` value, which — like
        // the interpreter's `None` — makes every output reaching it `NaN`.
        ExprData::Func { name, .. } if name == crate::poly::root_of::ROOT_OF => {
            match crate::poly::root_of::root_of_real_f64(node, pool) {
                Some(v) => plain(builder.ins().f64const(v)),
                None => Ok(Emitted {
                    value: builder.ins().f64const(f64::NAN),
                    fail: Some(builder.ins().iconst(types::I8, 1)),
                }),
            }
        }
        ExprData::Func { name, args } => {
            if args.len() == 1 {
                if let Some(func_id) = math.unary(&name) {
                    let a = get(&args[0], "Func arg")?;
                    let func_ref = module.declare_func_in_func(func_id, builder.func);
                    let call = builder.ins().call(func_ref, &[a]);
                    return plain(builder.inst_results(call)[0]);
                }
            }
            // Any other registered head runs its own `numeric_f64` kernel —
            // the one the interpreter calls — through a generic trampoline.
            let &addr = cg
                .prim_addr
                .get(name.as_str())
                .ok_or_else(|| JitError::UnsupportedNode(format!("function '{name}'")))?;
            let (fail_slot, ptr_ty) = (
                cg.fail_slot
                    .ok_or_else(|| JitError::CompilationFailed("no fail slot".into()))?,
                types::I64,
            );
            let prim = builder.ins().iconst(ptr_ty, addr);
            let fail_addr = builder.ins().stack_addr(ptr_ty, fail_slot, 0);
            let call = if args.len() == 1 {
                let a = get(&args[0], "Func arg")?;
                let r = module.declare_func_in_func(math.prim1_id, builder.func);
                builder.ins().call(r, &[prim, a, fail_addr])
            } else {
                let args_slot = cg
                    .args_slot
                    .ok_or_else(|| JitError::CompilationFailed("no args slot".into()))?;
                for (k, a) in args.iter().enumerate() {
                    let v = get(a, "Func arg")?;
                    builder.ins().stack_store(v, args_slot, (k * 8) as i32);
                }
                let args_addr = builder.ins().stack_addr(ptr_ty, args_slot, 0);
                let n = builder.ins().iconst(types::I64, args.len() as i64);
                let r = module.declare_func_in_func(math.primn_id, builder.func);
                builder.ins().call(r, &[prim, args_addr, n, fail_addr])
            };
            let value = builder.inst_results(call)[0];
            let fail = builder.ins().stack_load(types::I8, fail_slot, 0);
            Ok(Emitted {
                value,
                fail: Some(fail),
            })
        }
        other => Err(JitError::UnsupportedNode(format!("{other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// Emitting the outputs of one evaluation point
// ---------------------------------------------------------------------------

/// What is being compiled: the output roots, their input variables, and the
/// pool they all live in.
#[derive(Clone, Copy)]
struct EvalTarget<'a> {
    outputs: &'a [ExprId],
    inputs: &'a [ExprId],
    pool: &'a ExprPool,
}

/// Emit every output of one point, given the input values. Shared nodes are
/// emitted once and reused by every output that reaches them. An output that
/// reaches a generic call whose kernel failed is `NaN`, as the interpreter's
/// `None` is; outputs that do not reach it are unaffected.
fn emit_point(
    target: EvalTarget<'_>,
    cg: &Codegen<'_>,
    input_values: &[Value],
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
) -> Result<Vec<Value>, JitError> {
    let EvalTarget {
        outputs,
        inputs,
        pool,
    } = target;
    let mut values: HashMap<ExprId, Value> = HashMap::new();
    // A symbol listed twice takes its value from the *last* position, as the
    // interpreter's environment does.
    for (&var, &v) in inputs.iter().zip(input_values) {
        values.insert(var, v);
    }
    let mut fails: HashMap<ExprId, Value> = HashMap::new();
    for &root in outputs {
        for node in topo_sort(root, pool) {
            if values.contains_key(&node) {
                continue;
            }
            let e = codegen_node(node, cg, &values, builder, module)?;
            values.insert(node, e.value);
            if let Some(f) = e.fail {
                fails.insert(node, f);
            }
        }
    }
    let mut results = Vec::with_capacity(outputs.len());
    for &root in outputs {
        let r = *values
            .get(&root)
            .ok_or_else(|| JitError::CompilationFailed("root node not emitted".to_string()))?;
        if fails.is_empty() {
            results.push(r);
            continue;
        }
        let mut mask: Option<Value> = None;
        for node in topo_sort(root, pool) {
            if let Some(&f) = fails.get(&node) {
                mask = Some(match mask {
                    None => f,
                    Some(m) => builder.ins().bor(m, f),
                });
            }
        }
        results.push(match mask {
            None => r,
            Some(m) => {
                let nan = builder.ins().f64const(f64::NAN);
                builder.ins().select(m, nan, r)
            }
        })
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// compile — public entry points
// ---------------------------------------------------------------------------

/// The code Cranelift produced for a set of outputs, and what keeps it valid.
pub(super) struct CraneliftCode {
    /// `fn(*const f64, n) -> f64` for the first output (single-output
    /// compiles only).
    pub(super) scalar: Option<JitScalarFn>,
    /// The column entry point, see [`JitColsFn`].
    pub(super) cols: JitColsFn,
    /// Owns the code pages both pointers point into.
    pub(super) module: Box<OwnedJitModule>,
    /// The primitives the code calls through [`tramp_prim1`] /
    /// [`tramp_primn`]; the code holds the address of each element.
    pub(super) prims: Box<[&'static dyn Primitive]>,
}

/// Compile `expr` to a native function using the Cranelift JIT backend.
///
/// Returns a [`CompiledFn`] wrapping the generated code and keeping the
/// [`JITModule`] alive so the code pages remain valid.
pub fn compile_cranelift(
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CompiledFn, JitError> {
    let code = compile_cranelift_code(&[expr], inputs, pool, true)?;
    Ok(CompiledFn {
        inner: CompiledFnInner::Cranelift {
            fn_ptr: code
                .scalar
                .ok_or_else(|| JitError::CompilationFailed("no scalar entry".into()))?,
            cols_fn: code.cols,
            _prims: code.prims,
            _module: code.module,
        },
        n_inputs: inputs.len(),
        tier: CompileTier::Cranelift,
        complex: None,
    })
}

/// Compile several outputs into one column entry point (shared
/// subexpressions are evaluated once per point).
pub(super) fn compile_cranelift_many(
    outputs: &[ExprId],
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CraneliftCode, JitError> {
    compile_cranelift_code(outputs, inputs, pool, false)
}

fn compile_cranelift_code(
    outputs: &[ExprId],
    inputs: &[ExprId],
    pool: &ExprPool,
    want_scalar: bool,
) -> Result<CraneliftCode, JitError> {
    let target = EvalTarget {
        outputs,
        inputs,
        pool,
    };

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
    jit_builder.symbol("alkahest_prim1", tramp_prim1 as *const u8);
    // Where the ISA has no FMA instruction Cranelift lowers `fma` to a call to
    // the libcall `fma`; route it to the crate's exactly rounded one, not to
    // whatever the C library provides (MinGW's is not fused).
    jit_builder.symbol("fma", tramp_fma as *const u8);
    jit_builder.symbol("alkahest_primn", tramp_primn as *const u8);

    let mut module = OwnedJitModule::new(JITModule::new(jit_builder));

    // ------------------------------------------------------------------
    // 3. Declare imported math functions
    // ------------------------------------------------------------------
    let ptr_type = module.target_config().pointer_type();
    let mut f1_sig = module.make_signature();
    f1_sig.params.push(AbiParam::new(types::F64));
    f1_sig.returns.push(AbiParam::new(types::F64));

    let mut f2_sig = module.make_signature();
    f2_sig.params.push(AbiParam::new(types::F64));
    f2_sig.params.push(AbiParam::new(types::F64));
    f2_sig.returns.push(AbiParam::new(types::F64));

    let mut prim1_sig = module.make_signature();
    prim1_sig.params.push(AbiParam::new(ptr_type));
    prim1_sig.params.push(AbiParam::new(types::F64));
    prim1_sig.params.push(AbiParam::new(ptr_type));
    prim1_sig.returns.push(AbiParam::new(types::F64));

    let mut primn_sig = module.make_signature();
    primn_sig.params.push(AbiParam::new(ptr_type));
    primn_sig.params.push(AbiParam::new(ptr_type));
    primn_sig.params.push(AbiParam::new(types::I64));
    primn_sig.params.push(AbiParam::new(ptr_type));
    primn_sig.returns.push(AbiParam::new(types::F64));

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
        prim1_id: decl(&mut module, "alkahest_prim1", &prim1_sig)?,
        primn_id: decl(&mut module, "alkahest_primn", &primn_sig)?,
    };

    // ------------------------------------------------------------------
    // 4. Resolve the generic heads to their primitives, up front: the code
    //    embeds the address of each one's slot.
    // ------------------------------------------------------------------
    let mut prim_names: Vec<String> = Vec::new();
    let mut max_arity = 0usize;
    let mut seen = crate::kernel::IdSet::default();
    for &root in outputs {
        for node in topo_sort(root, pool) {
            if !seen.insert(node) || !is_generic_call(node, pool, &math) {
                continue;
            }
            pool.with(node, |d| {
                if let ExprData::Func { name, args } = d {
                    if !prim_names.iter().any(|n| n == name) {
                        prim_names.push(name.clone());
                    }
                    if args.len() > 1 {
                        max_arity = max_arity.max(args.len());
                    }
                }
            });
        }
    }
    let prims: Box<[&'static dyn Primitive]> = prim_names
        .iter()
        .map(|name| {
            registry()
                .get(name)
                .ok_or_else(|| JitError::UnsupportedNode(format!("function '{name}'")))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_boxed_slice();
    let prim_addr: HashMap<String, i64> = prim_names
        .iter()
        .zip(prims.iter())
        .map(|(n, p)| (n.clone(), p as *const &'static dyn Primitive as i64))
        .collect();
    let has_generic = !prims.is_empty();
    let make_codegen = |builder: &mut FunctionBuilder| {
        let mut slot = |bytes: usize| {
            builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                bytes as u32,
                3,
            ))
        };
        let fail_slot = has_generic.then(|| slot(8));
        let args_slot = (max_arity > 0).then(|| slot(max_arity * 8));
        (fail_slot, args_slot)
    };

    // ------------------------------------------------------------------
    // 5. Scalar eval: fn(*const f64, i64) -> f64 (single output only)
    // ------------------------------------------------------------------
    let mut scalar_id = None;
    if want_scalar {
        let mut eval_sig = module.make_signature();
        eval_sig.params.push(AbiParam::new(ptr_type));
        eval_sig.params.push(AbiParam::new(types::I64));
        eval_sig.returns.push(AbiParam::new(types::F64));

        let id = module
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
            let input_values: Vec<Value> = (0..inputs.len())
                .map(|i| {
                    builder.ins().load(
                        types::F64,
                        MemFlags::trusted(),
                        inputs_ptr,
                        (i * std::mem::size_of::<f64>()) as i32,
                    )
                })
                .collect();
            let (fail_slot, args_slot) = make_codegen(&mut builder);
            let cg = Codegen {
                pool,
                math: &math,
                prim_addr: &prim_addr,
                fail_slot,
                args_slot,
            };
            let result = emit_point(
                EvalTarget {
                    outputs: &outputs[..1],
                    ..target
                },
                &cg,
                &input_values,
                &mut builder,
                &mut module,
            )?;
            builder.ins().return_(&result[..1]);
            builder.finalize();
        }
        module
            .define_function(id, &mut scalar_ctx)
            .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
        module.clear_context(&mut scalar_ctx);
        scalar_id = Some(id);
    }

    // ------------------------------------------------------------------
    // 6. Column eval: fn(cols, outs, start, end) -> ()
    // ------------------------------------------------------------------
    let mut cols_sig = module.make_signature();
    cols_sig.params.push(AbiParam::new(ptr_type));
    cols_sig.params.push(AbiParam::new(ptr_type));
    cols_sig.params.push(AbiParam::new(types::I64));
    cols_sig.params.push(AbiParam::new(types::I64));

    let cols_id = module
        .declare_function("alkahest_eval_cols", Linkage::Export, &cols_sig)
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

    let mut cols_ctx = module.make_context();
    cols_ctx.func.signature = cols_sig;
    {
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut cols_ctx.func, &mut func_ctx);
        let entry = builder.create_block();
        let loop_hdr = builder.create_block();
        let loop_body = builder.create_block();
        let exit = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.append_block_param(loop_hdr, types::I64);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let cols_ptr = builder.block_params(entry)[0];
        let outs_ptr = builder.block_params(entry)[1];
        let start = builder.block_params(entry)[2];
        let end = builder.block_params(entry)[3];
        // Column and output base pointers are loop-invariant.
        let col_bases: Vec<Value> = (0..inputs.len())
            .map(|i| {
                builder
                    .ins()
                    .load(ptr_type, MemFlags::trusted(), cols_ptr, (i * 8) as i32)
            })
            .collect();
        let out_bases: Vec<Value> = (0..outputs.len())
            .map(|k| {
                builder
                    .ins()
                    .load(ptr_type, MemFlags::trusted(), outs_ptr, (k * 8) as i32)
            })
            .collect();
        let (fail_slot, args_slot) = make_codegen(&mut builder);
        let cg = Codegen {
            pool,
            math: &math,
            prim_addr: &prim_addr,
            fail_slot,
            args_slot,
        };
        let start_arg = BlockArg::from(start);
        builder
            .ins()
            .jump(loop_hdr, std::slice::from_ref(&start_arg));

        builder.switch_to_block(loop_hdr);
        let idx = builder.block_params(loop_hdr)[0];
        let done = builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, idx, end);
        builder.ins().brif(done, exit, &[], loop_body, &[]);

        builder.switch_to_block(loop_body);
        let byte_off = builder.ins().ishl_imm(idx, 3);
        let input_values: Vec<Value> = col_bases
            .iter()
            .map(|&base| {
                let addr = builder.ins().iadd(base, byte_off);
                builder.ins().load(types::F64, MemFlags::trusted(), addr, 0)
            })
            .collect();
        let results = emit_point(target, &cg, &input_values, &mut builder, &mut module)?;
        for (&base, &r) in out_bases.iter().zip(&results) {
            let addr = builder.ins().iadd(base, byte_off);
            builder.ins().store(MemFlags::trusted(), r, addr, 0);
        }
        let next = builder.ins().iadd_imm(idx, 1);
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
        .define_function(cols_id, &mut cols_ctx)
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;
    module.clear_context(&mut cols_ctx);

    module
        .finalize_definitions()
        .map_err(|e| JitError::CompilationFailed(e.to_string()))?;

    let scalar = scalar_id.map(|id| {
        let p = module.get_finalized_function(id);
        // SAFETY: defined above with exactly this signature.
        unsafe { std::mem::transmute::<*const u8, JitScalarFn>(p) }
    });
    let cols_raw = module.get_finalized_function(cols_id);
    // SAFETY: defined above with exactly this signature.
    let cols: JitColsFn = unsafe { std::mem::transmute::<*const u8, JitColsFn>(cols_raw) };

    Ok(CraneliftCode {
        scalar,
        cols,
        module: Box::new(module),
        prims,
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

    /// `cranelift_jit`'s `Drop` leaks a module's code pages by design; every
    /// compile-and-drop used to keep them mapped for the life of the process
    /// (about 4 kB resident per compile). Dropping a compiled
    /// function — or a compile that fails part-way — must release them.
    #[test]
    fn dropping_a_compiled_function_frees_its_code_pages() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("sin", vec![pool.add(vec![x, pool.integer(1_i32)])]);
        let freed = || FREED_MODULES.with(|n| n.get());

        let before = freed();
        for i in 0..4 {
            let f = compile_cranelift(expr, &[x], &pool).unwrap();
            assert_eq!(f.call(&[0.5]), 1.5_f64.sin());
            assert_eq!(freed(), before + i, "still alive: not freed yet");
            drop(f);
            assert_eq!(freed(), before + i + 1, "freed on drop");
        }

        // An unbound symbol is refused by codegen after the module exists.
        let y = pool.symbol("y", Domain::Real);
        let before = freed();
        assert!(compile_cranelift(pool.add(vec![x, y]), &[x], &pool).is_err());
        assert_eq!(freed(), before + 1, "a failed compile frees its module too");
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
