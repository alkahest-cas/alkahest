//! NVPTX JIT backend for Alkahest compiled functions.
//!
//! Compiles symbolic expressions to PTX for execution on NVIDIA GPUs. V1-1
//! replaces the V5-3 scaffold with a full inkwell-driven lowering through the
//! LLVM NVPTX backend plus a cudarc-based runtime.
//!
//! Kernel ABI: `alkahest_eval(double* out, const double* in, u64 n_pts)`.
//! Input layout is structure-of-arrays — variable `v` for point `t` is at
//! `in[v * n_pts + t]` — which maps directly onto the NumPy per-variable
//! columns already used by [`crate::jit::CompiledFn::call_batch`].
//!
//! Transcendentals lower to the `__nv_*` functions supplied by NVIDIA's
//! libdevice bitcode; the bitcode is linked into the LLVM module before
//! codegen so the resulting PTX is self-contained.

use crate::kernel::{ExprData, ExprId, ExprPool};

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

/// Error type for CUDA compilation and runtime failures.
#[derive(Debug, Clone)]
pub enum CudaError {
    NvptxTargetUnavailable,
    PtxGenerationFailed(String),
    DriverError(String),
    NotImplemented(&'static str),
    LibdeviceNotFound(String),
    LaunchError(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::NvptxTargetUnavailable => write!(
                f,
                "NVPTX target not available in this LLVM build; \
                 rebuild LLVM with nvptx64 in LLVM_TARGETS_TO_BUILD"
            ),
            CudaError::PtxGenerationFailed(msg) => write!(f, "PTX generation failed: {msg}"),
            CudaError::DriverError(msg) => write!(f, "CUDA driver error: {msg}"),
            CudaError::NotImplemented(what) => write!(f, "not implemented: {what}"),
            CudaError::LibdeviceNotFound(msg) => write!(f, "libdevice bitcode not found: {msg}"),
            CudaError::LaunchError(msg) => write!(f, "kernel launch failed: {msg}"),
        }
    }
}

impl std::error::Error for CudaError {}

impl crate::errors::AlkahestError for CudaError {
    fn code(&self) -> &'static str {
        match self {
            CudaError::NvptxTargetUnavailable => "E-CUDA-001",
            CudaError::PtxGenerationFailed(_) => "E-CUDA-002",
            CudaError::DriverError(_) => "E-CUDA-003",
            CudaError::NotImplemented(_) => "E-CUDA-004",
            CudaError::LibdeviceNotFound(_) => "E-CUDA-005",
            CudaError::LaunchError(_) => "E-CUDA-006",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            CudaError::NvptxTargetUnavailable => Some(
                "Rebuild LLVM with NVPTX in LLVM_TARGETS_TO_BUILD, or omit --features cuda.",
            ),
            CudaError::PtxGenerationFailed(_) => {
                Some("Inspect the emitted PTX; verify every primitive has a CUDA lowering.")
            }
            CudaError::DriverError(_) => Some(
                "Run `nvidia-smi`; ensure a CUDA context is current; retry with CUDA_LAUNCH_BLOCKING=1.",
            ),
            CudaError::NotImplemented(_) => Some(
                "This path is a v1.0 stub; track its completion under V1-1 (production NVPTX codegen).",
            ),
            CudaError::LibdeviceNotFound(_) => Some(
                "Install nvidia-cuda-toolkit (for /usr/lib/nvidia-cuda-toolkit/libdevice), \
                 or set ALKAHEST_LIBDEVICE_PATH to your libdevice.10.bc file.",
            ),
            CudaError::LaunchError(_) => Some(
                "Check block/grid dimensions and device memory; rerun with compute-sanitizer for details.",
            ),
        }
    }
}

/// Number of CUDA devices visible to this process, or `0` when none are.
///
/// Answers "which ordinals may I pass to [`CudaCompiledFn::call_batch_on`]?"
/// without the caller having to probe by launching and catching
/// `E-CUDA-003`, which is what `docs/mdbook/src/gpu.md` had to recommend
/// while no verified implementation existed.
///
/// Returns `0` rather than an error for every "no GPU here" shape — no
/// driver, no device, driver too old. The distinction a caller acts on is
/// "can I use a GPU", and each of those answers it identically; an ordinal
/// is valid iff it is `< cuda_device_count()`.
///
/// `catch_unwind` is load-bearing for the same reason it is in
/// `groebner_cuda.rs::gpu_available`: `cudarc` *panics* rather than
/// returning `Err` when `libcuda.so` cannot be dlopen'd at all, which is the
/// state of any machine with no driver installed. A capability probe that
/// aborts the process on the exact configuration it exists to report on
/// would be worse than useless.
#[cfg(feature = "cuda")]
pub fn cuda_device_count() -> usize {
    std::panic::catch_unwind(|| {
        cudarc::driver::CudaContext::device_count()
            .map(|n| n.max(0) as usize)
            .unwrap_or(0)
    })
    .unwrap_or(0)
}

/// A compiled CUDA kernel for evaluating an Alkahest expression on GPU.
///
/// The generated PTX is self-contained (libdevice has been linked in during
/// codegen) and can be reused across multiple launches and multiple devices.
pub struct CudaCompiledFn {
    /// The PTX source code (inspectable, cacheable).
    pub ptx: String,
    /// Number of input variables.
    pub n_inputs: usize,

    /// Lazy per-device runtime cache. Only populated on the first launch for a
    /// given device ordinal; subsequent launches reuse the loaded module.
    #[cfg(feature = "cuda")]
    runtime: Mutex<HashMap<usize, DeviceRuntime>>,
}

#[cfg(feature = "cuda")]
struct DeviceRuntime {
    #[allow(dead_code)] // kept alive so module/function stay valid
    ctx: Arc<cudarc::driver::CudaContext>,
    #[allow(dead_code)]
    module: Arc<cudarc::driver::CudaModule>,
    function: cudarc::driver::CudaFunction,
}

impl CudaCompiledFn {
    /// Return the PTX assembly as a string.
    pub fn ptx_source(&self) -> &str {
        &self.ptx
    }

    /// Batch-evaluate the compiled kernel on the default CUDA device (0).
    ///
    /// `inputs[i]` must carry the values of variable `i` across all N points;
    /// every slice must have the same length N matching `output.len()`.
    #[cfg(feature = "cuda")]
    pub fn call_batch(&self, inputs: &[&[f64]], output: &mut [f64]) -> Result<(), CudaError> {
        self.call_batch_on(0, inputs, output)
    }

    /// Batch-evaluate on a specific CUDA device ordinal.
    #[cfg(feature = "cuda")]
    pub fn call_batch_on(
        &self,
        device_ordinal: usize,
        inputs: &[&[f64]],
        output: &mut [f64],
    ) -> Result<(), CudaError> {
        if inputs.len() != self.n_inputs {
            return Err(CudaError::LaunchError(format!(
                "expected {} input arrays, got {}",
                self.n_inputs,
                inputs.len()
            )));
        }
        let n_pts = output.len();
        for (i, col) in inputs.iter().enumerate() {
            if col.len() != n_pts {
                return Err(CudaError::LaunchError(format!(
                    "input {i} length {} != output length {n_pts}",
                    col.len(),
                )));
            }
        }
        if n_pts == 0 {
            return Ok(());
        }

        let mut cache = self
            .runtime
            .lock()
            .map_err(|_| CudaError::LaunchError("runtime cache poisoned".to_string()))?;
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(device_ordinal) {
            let rt = runtime::load_ptx(device_ordinal, &self.ptx)?;
            e.insert(rt);
        }
        let rt = cache.get(&device_ordinal).unwrap();

        runtime::launch(rt, inputs, output, self.n_inputs, n_pts)
    }

    /// Batch-evaluate using the pre-allocated device pointers (DLPack-friendly).
    ///
    /// `device_in` points to an `n_inputs * n_pts` contiguous SoA f64 buffer on
    /// the given device; `device_out` points to an `n_pts` f64 buffer on the
    /// same device. Both must be 8-byte aligned.
    #[cfg(feature = "cuda")]
    pub fn call_device_ptrs(
        &self,
        device_ordinal: usize,
        device_in: u64,
        device_out: u64,
        n_pts: usize,
    ) -> Result<(), CudaError> {
        if n_pts == 0 {
            return Ok(());
        }
        let mut cache = self
            .runtime
            .lock()
            .map_err(|_| CudaError::LaunchError("runtime cache poisoned".to_string()))?;
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(device_ordinal) {
            let rt = runtime::load_ptx(device_ordinal, &self.ptx)?;
            e.insert(rt);
        }
        let rt = cache.get(&device_ordinal).unwrap();
        runtime::launch_raw(rt, self.n_inputs, device_in, device_out, n_pts)
    }
}

/// Compile a symbolic expression to a CUDA kernel targeting `sm_86`.
pub fn compile_cuda(
    expr: ExprId,
    inputs: &[ExprId],
    pool: &ExprPool,
) -> Result<CudaCompiledFn, CudaError> {
    let n_inputs = inputs.len();
    #[cfg(all(feature = "cuda", feature = "jit"))]
    let ptx = codegen::emit_ptx(expr, inputs, pool)?;

    // Without `jit`, we still return a stub PTX so the error codes and PTX
    // header are available for testing. `cuda` implies `jit` at the Cargo
    // feature level, so this branch is only taken during unit tests that build
    // without either feature.
    #[cfg(not(all(feature = "cuda", feature = "jit")))]
    let ptx = emit_ptx_stub(expr, inputs, n_inputs, pool);

    Ok(CudaCompiledFn {
        ptx,
        n_inputs,
        #[cfg(feature = "cuda")]
        runtime: Mutex::new(HashMap::new()),
    })
}

// Stub PTX kept only so `compile_cuda` has a build-only fallback when the
// caller has deliberately disabled `jit`. The production path goes through
// `codegen::emit_ptx`.
#[cfg(not(all(feature = "cuda", feature = "jit")))]
fn emit_ptx_stub(_e: ExprId, _ins: &[ExprId], _n: usize, _pool: &ExprPool) -> String {
    let mut s = String::new();
    s.push_str(".version 7.5\n.target sm_86\n.address_size 64\n\n");
    s.push_str(".visible .entry alkahest_eval(\n");
    s.push_str("    .param .u64 param_out,\n");
    s.push_str("    .param .u64 param_in,\n");
    s.push_str("    .param .u64 param_n_pts\n");
    s.push_str(") { ret; }\n");
    s
}

// ---------------------------------------------------------------------------
// Codegen — inkwell → LLVM IR → NVPTX PTX
// ---------------------------------------------------------------------------

#[cfg(all(feature = "cuda", feature = "jit"))]
mod codegen {
    use super::{CudaError, ExprData, ExprId, ExprPool};
    use inkwell::{
        attributes::{Attribute, AttributeLoc},
        builder::Builder,
        context::{AsContextRef, Context},
        module::{Linkage, Module},
        passes::PassBuilderOptions,
        targets::{
            CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
            TargetTriple,
        },
        values::{
            AsValueRef, BasicMetadataValueEnum, FloatValue, FunctionValue, IntValue, PointerValue,
        },
        AddressSpace, IntPredicate, OptimizationLevel,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;

    // NVPTX kernel calling convention in LLVM.
    const PTX_KERNEL_CC: u32 = 71;

    pub fn emit_ptx(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> Result<String, CudaError> {
        // Initialize the NVPTX target.
        Target::initialize_nvptx(&InitializationConfig::default());

        let triple = TargetTriple::create("nvptx64-nvidia-cuda");
        let target = Target::from_triple(&triple)
            .map_err(|e| CudaError::PtxGenerationFailed(format!("no NVPTX target in LLVM: {e}")))?;

        // ptx75 corresponds to CUDA 11.0+; supports sm_80/sm_86 Ampere features.
        let machine = target
            .create_target_machine(
                &triple,
                "sm_86",
                "+ptx75",
                OptimizationLevel::Default,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or_else(|| {
                CudaError::PtxGenerationFailed("failed to create NVPTX TargetMachine".to_string())
            })?;

        // Build a context that owns the module + (later) libdevice bitcode.
        let ctx = Context::create();
        let module = ctx.create_module("alkahest_nvptx");
        module.set_triple(&triple);
        // Set the datalayout to match NVPTX64 so the verifier is happy once
        // libdevice is linked in.
        module.set_data_layout(&machine.get_target_data().get_data_layout());

        let builder = ctx.create_builder();
        build_kernel(&ctx, &module, &builder, expr, inputs, pool)?;

        // Link libdevice bitcode so calls to __nv_sin etc. resolve.
        link_libdevice(&ctx, &module)?;

        // Internalize everything except our kernel entry, then DCE + inline.
        // Without this, libdevice brings in __nv_tanh (and friends) which LLVM 21
        // lowers to llvm.nvvm.tanh.approx.f32 — an intrinsic the NVPTX backend
        // cannot emit as PTX, and that surfaces as ptxas parse failures.
        internalize_non_kernel(&module, "alkahest_eval");
        run_libdevice_cleanup_passes(&module, &machine)?;

        module.verify().map_err(|e| {
            CudaError::PtxGenerationFailed(format!("module verify: {}", e.to_string()))
        })?;

        // Emit PTX.
        let buf = machine
            .write_to_memory_buffer(&module, FileType::Assembly)
            .map_err(|e| CudaError::PtxGenerationFailed(format!("write PTX: {e}")))?;

        // LLVM's memory buffer for an assembly write is nul-terminated; strip
        // any trailing nuls so the string is safe to pass to CString::new.
        let mut ptx_bytes = buf.as_slice();
        while ptx_bytes.last() == Some(&0) {
            ptx_bytes = &ptx_bytes[..ptx_bytes.len() - 1];
        }
        let ptx = std::str::from_utf8(ptx_bytes)
            .map_err(|e| CudaError::PtxGenerationFailed(format!("PTX utf8: {e}")))?
            .to_string();

        Ok(ptx)
    }

    fn build_kernel<'ctx>(
        ctx: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
        expr: ExprId,
        inputs: &[ExprId],
        pool: &ExprPool,
    ) -> Result<(), CudaError> {
        let f64_t = ctx.f64_type();
        let i32_t = ctx.i32_type();
        let i64_t = ctx.i64_type();
        let ptr_t = ctx.ptr_type(AddressSpace::default());
        let void_t = ctx.void_type();

        let fn_type = void_t.fn_type(&[ptr_t.into(), ptr_t.into(), i64_t.into()], false);
        let kernel = module.add_function("alkahest_eval", fn_type, Some(Linkage::External));
        kernel.set_call_conventions(PTX_KERNEL_CC);

        // NVPTX requires both the calling convention AND the !nvvm.annotations
        // metadata entry to recognize a function as a kernel. Emit the metadata.
        emit_nvvm_kernel_annotation(ctx, module, kernel);

        // noinline hint: we want the kernel function itself to stay a kernel
        // entry, not get inlined away when libdevice is linked.
        let noinline_kind = Attribute::get_named_enum_kind_id("noinline");
        kernel.add_attribute(
            AttributeLoc::Function,
            ctx.create_enum_attribute(noinline_kind, 0),
        );

        let entry_bb = ctx.append_basic_block(kernel, "entry");
        let body_bb = ctx.append_basic_block(kernel, "body");
        let exit_bb = ctx.append_basic_block(kernel, "exit");
        builder.position_at_end(entry_bb);

        // tid = blockIdx.x * blockDim.x + threadIdx.x
        let tid_x = declare_nvvm_read(module, ctx, "llvm.nvvm.read.ptx.sreg.tid.x");
        let ctaid_x = declare_nvvm_read(module, ctx, "llvm.nvvm.read.ptx.sreg.ctaid.x");
        let ntid_x = declare_nvvm_read(module, ctx, "llvm.nvvm.read.ptx.sreg.ntid.x");

        let tid = builder
            .build_call(tid_x, &[], "tid")
            .map_err(codegen_err)?
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        let bid = builder
            .build_call(ctaid_x, &[], "bid")
            .map_err(codegen_err)?
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        let bdim = builder
            .build_call(ntid_x, &[], "bdim")
            .map_err(codegen_err)?
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();

        let bid_bdim = builder
            .build_int_mul(bid, bdim, "bid_bdim")
            .map_err(codegen_err)?;
        let gid_i32 = builder
            .build_int_add(bid_bdim, tid, "gid_i32")
            .map_err(codegen_err)?;
        let gid = builder
            .build_int_z_extend(gid_i32, i64_t, "gid")
            .map_err(codegen_err)?;

        // Bounds check: if gid >= n_pts, return.
        let n_pts = kernel.get_nth_param(2).unwrap().into_int_value();
        let in_bounds = builder
            .build_int_compare(IntPredicate::ULT, gid, n_pts, "in_bounds")
            .map_err(codegen_err)?;
        builder
            .build_conditional_branch(in_bounds, body_bb, exit_bb)
            .map_err(codegen_err)?;

        // Body.
        builder.position_at_end(body_bb);
        let out_ptr = kernel.get_nth_param(0).unwrap().into_pointer_value();
        let in_ptr = kernel.get_nth_param(1).unwrap().into_pointer_value();

        // Load inputs[v] for this thread: in[v * n_pts + gid].
        let mut values: HashMap<ExprId, FloatValue<'ctx>> = HashMap::new();
        for (v_idx, &var) in inputs.iter().enumerate() {
            let v_idx_v = i64_t.const_int(v_idx as u64, false);
            let offset_v = builder
                .build_int_mul(v_idx_v, n_pts, &format!("row_off_{v_idx}"))
                .map_err(codegen_err)?;
            let idx = builder
                .build_int_add(offset_v, gid, &format!("in_idx_{v_idx}"))
                .map_err(codegen_err)?;
            let gep = unsafe {
                builder
                    .build_gep(f64_t, in_ptr, &[idx], &format!("in_gep_{v_idx}"))
                    .map_err(codegen_err)?
            };
            let val = builder
                .build_load(f64_t, gep, &format!("x{v_idx}"))
                .map_err(codegen_err)?
                .into_float_value();
            values.insert(var, val);
        }

        // Topological sort, codegen each node in order.
        let topo = topo_sort(expr, pool);
        for &node in &topo {
            if values.contains_key(&node) {
                continue;
            }
            let v = codegen_node(node, pool, &values, ctx, module, builder)?;
            values.insert(node, v);
        }

        let result = *values
            .get(&expr)
            .ok_or_else(|| CudaError::PtxGenerationFailed("root node not computed".to_string()))?;

        // Store to out[gid].
        let out_gep = unsafe {
            builder
                .build_gep(f64_t, out_ptr, &[gid], "out_gep")
                .map_err(codegen_err)?
        };
        builder.build_store(out_gep, result).map_err(codegen_err)?;
        builder
            .build_unconditional_branch(exit_bb)
            .map_err(codegen_err)?;

        builder.position_at_end(exit_bb);
        builder.build_return(None).map_err(codegen_err)?;

        // Silence a "declared but not used" warning on i32_t.
        let _ = i32_t;
        Ok(())
    }

    fn codegen_node<'ctx>(
        node: ExprId,
        pool: &ExprPool,
        values: &HashMap<ExprId, FloatValue<'ctx>>,
        ctx: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> Result<FloatValue<'ctx>, CudaError> {
        let f64_t = ctx.f64_type();
        match pool.get(node) {
            ExprData::Integer(n) => Ok(f64_t.const_float(n.0.to_f64())),
            ExprData::Rational(r) => {
                let (n, d) = r.0.clone().into_numer_denom();
                Ok(f64_t.const_float(n.to_f64() / d.to_f64()))
            }
            ExprData::Float(f) => Ok(f64_t.const_float(f.inner.to_f64())),
            ExprData::Symbol { name, .. } => Err(CudaError::PtxGenerationFailed(format!(
                "unbound symbol '{name}' (not provided in inputs)"
            ))),
            ExprData::Add(args) => {
                let mut acc = f64_t.const_float(0.0);
                for &a in &args {
                    let v = *values.get(&a).ok_or_else(|| {
                        CudaError::PtxGenerationFailed("missing Add child".to_string())
                    })?;
                    acc = builder
                        .build_float_add(acc, v, "fadd")
                        .map_err(codegen_err)?;
                }
                Ok(acc)
            }
            ExprData::Mul(args) => {
                let mut acc = f64_t.const_float(1.0);
                for &a in &args {
                    let v = *values.get(&a).ok_or_else(|| {
                        CudaError::PtxGenerationFailed("missing Mul child".to_string())
                    })?;
                    acc = builder
                        .build_float_mul(acc, v, "fmul")
                        .map_err(codegen_err)?;
                }
                Ok(acc)
            }
            ExprData::Pow { base, exp } => {
                let b = *values.get(&base).ok_or_else(|| {
                    CudaError::PtxGenerationFailed("missing Pow base".to_string())
                })?;
                // Fast path: small positive integer exponent → unrolled multiplies.
                if let ExprData::Integer(n) = pool.get(exp) {
                    let n_i64 = n.0.to_f64() as i64;
                    if (0..=16).contains(&n_i64) && (n_i64 as f64) == n.0.to_f64() {
                        let mut acc = f64_t.const_float(1.0);
                        for _ in 0..n_i64 {
                            acc = builder
                                .build_float_mul(acc, b, "pow_unroll")
                                .map_err(codegen_err)?;
                        }
                        return Ok(acc);
                    }
                }
                let e = *values.get(&exp).ok_or_else(|| {
                    CudaError::PtxGenerationFailed("missing Pow exponent".to_string())
                })?;
                let pow_fn = declare_libdevice(module, ctx, "__nv_pow", 2);
                let r = builder
                    .build_call(pow_fn, &[b.into(), e.into()], "pow")
                    .map_err(codegen_err)?;
                Ok(r.try_as_basic_value().unwrap_basic().into_float_value())
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                let a = *values.get(&args[0]).ok_or_else(|| {
                    CudaError::PtxGenerationFailed("missing Func arg".to_string())
                })?;
                let nv = match name.as_str() {
                    "sin" => "__nv_sin",
                    "cos" => "__nv_cos",
                    "tan" => "__nv_tan",
                    "exp" => "__nv_exp",
                    "log" => "__nv_log",
                    "sqrt" => "__nv_sqrt",
                    "abs" => "__nv_fabs",
                    other => {
                        return Err(CudaError::PtxGenerationFailed(format!(
                            "unsupported function '{other}' in NVPTX codegen"
                        )))
                    }
                };
                let f = declare_libdevice(module, ctx, nv, 1);
                let r = builder
                    .build_call(f, &[a.into()], "nvcall")
                    .map_err(codegen_err)?;
                Ok(r.try_as_basic_value().unwrap_basic().into_float_value())
            }
            other => Err(CudaError::PtxGenerationFailed(format!(
                "unsupported node {other:?}"
            ))),
        }
    }

    fn topo_sort(root: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();
        dfs(root, pool, &mut visited, &mut order);
        order
    }

    fn dfs(
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
            _ => vec![],
        });
        for c in children {
            dfs(c, pool, visited, order);
        }
        order.push(node);
    }

    fn declare_nvvm_read<'ctx>(
        module: &Module<'ctx>,
        ctx: &'ctx Context,
        name: &str,
    ) -> FunctionValue<'ctx> {
        if let Some(f) = module.get_function(name) {
            return f;
        }
        let i32_t = ctx.i32_type();
        let ty = i32_t.fn_type(&[], false);
        let f = module.add_function(name, ty, None);
        f
    }

    fn declare_libdevice<'ctx>(
        module: &Module<'ctx>,
        ctx: &'ctx Context,
        name: &str,
        arity: usize,
    ) -> FunctionValue<'ctx> {
        if let Some(f) = module.get_function(name) {
            return f;
        }
        let f64_t = ctx.f64_type();
        let params: Vec<inkwell::types::BasicMetadataTypeEnum<'ctx>> =
            (0..arity).map(|_| f64_t.into()).collect();
        let ty = f64_t.fn_type(&params, false);
        module.add_function(name, ty, Some(Linkage::External))
    }

    fn codegen_err<E: std::fmt::Display>(e: E) -> CudaError {
        CudaError::PtxGenerationFailed(e.to_string())
    }

    /// Emit the `!nvvm.annotations` metadata entry that flags our function
    /// as a PTX kernel. Inkwell 0.9 doesn't expose named-metadata APIs, so
    /// we drop to llvm-sys directly.
    /// NVVM requires the `nvvm.annotations` metadata node to mark a function as
    /// a kernel, and inkwell exposes no safe wrapper for it. The `*InContext`
    /// builders are deprecated upstream in favour of `*InContext2`, but those
    /// take an explicit length and are not surfaced by inkwell's re-export at
    /// the LLVM 21 version this crate pins. Migrating is a change to codegen,
    /// not a lint fix, so the deprecation is acknowledged rather than papered
    /// over by widening the lint level for the whole module.
    #[allow(deprecated)]
    fn emit_nvvm_kernel_annotation<'ctx>(
        ctx: &'ctx Context,
        module: &Module<'ctx>,
        kernel: FunctionValue<'ctx>,
    ) {
        use inkwell::llvm_sys::core::{
            LLVMAddNamedMetadataOperand, LLVMMDNodeInContext, LLVMMDStringInContext,
        };
        use std::ffi::CString;

        unsafe {
            let ctx_ref = ctx.as_ctx_ref();
            let module_ref = module.as_mut_ptr();

            let kernel_md_val = kernel.as_global_value().as_value_ref();

            let kernel_str = CString::new("kernel").unwrap();
            let kernel_md_name =
                LLVMMDStringInContext(ctx_ref, kernel_str.as_ptr(), "kernel".len() as u32);

            let one_i32 = ctx.i32_type().const_int(1, false).as_value_ref();
            // Build the metadata tuple { kernel_val, "kernel", i32 1 }.
            let mut tuple = [kernel_md_val, kernel_md_name, one_i32];
            let node = LLVMMDNodeInContext(ctx_ref, tuple.as_mut_ptr(), tuple.len() as u32);

            let anno = CString::new("nvvm.annotations").unwrap();
            LLVMAddNamedMetadataOperand(module_ref, anno.as_ptr(), node);
        }
    }

    fn link_libdevice<'ctx>(_ctx: &'ctx Context, module: &Module<'ctx>) -> Result<(), CudaError> {
        let path = locate_libdevice()?;
        let ld = Module::parse_bitcode_from_path(&path, module.get_context())
            .map_err(|e| CudaError::LibdeviceNotFound(format!("parse {}: {e}", path.display())))?;
        // The libdevice bitcode carries its own datalayout/triple (nvptx64-
        // nvidia-gpulibs); override them before linking to match our target.
        ld.set_triple(&module.get_triple());
        ld.set_data_layout(&module.get_data_layout());
        module.link_in_module(ld).map_err(|e| {
            CudaError::PtxGenerationFailed(format!("link libdevice: {}", e.to_string()))
        })?;
        Ok(())
    }

    fn locate_libdevice() -> Result<PathBuf, CudaError> {
        if let Ok(p) = std::env::var("ALKAHEST_LIBDEVICE_PATH") {
            let pb = PathBuf::from(p);
            if pb.is_file() {
                return Ok(pb);
            }
        }
        let candidates = [
            "/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc",
            "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
            "/opt/cuda/nvvm/libdevice/libdevice.10.bc",
        ];
        for c in &candidates {
            let pb = PathBuf::from(c);
            if pb.is_file() {
                return Ok(pb);
            }
        }
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            let pb = PathBuf::from(cuda_path).join("nvvm/libdevice/libdevice.10.bc");
            if pb.is_file() {
                return Ok(pb);
            }
        }
        Err(CudaError::LibdeviceNotFound(
            "searched common CUDA install paths; \
             set ALKAHEST_LIBDEVICE_PATH to override"
                .to_string(),
        ))
    }

    /// Mark every function except the named kernel as internal linkage so the
    /// subsequent `global-dce` pass can drop all unreferenced libdevice helpers.
    fn internalize_non_kernel<'ctx>(module: &Module<'ctx>, kernel_name: &str) {
        let mut func = module.get_first_function();
        while let Some(f) = func {
            let name = f.get_name().to_str().unwrap_or("");
            if name != kernel_name && !f.is_null() {
                // Preserve declarations (bodyless) — they're usually LLVM
                // intrinsics and changing their linkage confuses the verifier.
                if f.count_basic_blocks() > 0 {
                    f.as_global_value().set_linkage(Linkage::Internal);
                }
            }
            func = f.get_next_function();
        }
    }

    /// Run a minimal whole-module cleanup: inline libdevice helpers into the
    /// kernel, then DCE everything the kernel doesn't reach.
    ///
    /// Uses the new pass manager (`PassBuilderOptions` + `Module::run_passes`).
    /// The legacy `PassManager` this used to call was removed from LLVM in 17,
    /// so it does not merely warn on LLVM 21 — it does not compile.
    ///
    /// The pipeline is the same four passes in the same order as the legacy
    /// version, spelled in the new pass manager's textual syntax. Two of the
    /// names differ because the new PM does not accept the old ones:
    /// `function-inlining` is `inline`, and `strip-dead-prototypes` keeps its
    /// name but is a module pass, so the whole string is a module pipeline.
    ///
    /// **The emitted PTX is not guaranteed to be identical to the legacy
    /// pipeline's.** The new PM schedules differently, and this function exists
    /// because libdevice otherwise drags in `__nv_tanh` and friends, which LLVM
    /// lowers to `llvm.nvvm.tanh.approx.f32` — an intrinsic the NVPTX backend
    /// cannot emit, surfacing as a ptxas parse failure. That failure mode is
    /// what `scripts/verify_cuda_on_gpu.sh` is for; this change has not been run
    /// against a GPU.
    fn run_libdevice_cleanup_passes<'ctx>(
        module: &Module<'ctx>,
        machine: &TargetMachine,
    ) -> Result<(), CudaError> {
        // Run twice, belt-and-braces. The legacy pipeline did too, on the
        // theory that DCE after inlining catches helpers that only became dead
        // once the calls were folded in — but that is *not* observable: a GPU-box
        // run of a 14-kernel battery (poly, the six transcendentals, three `pow`
        // forms, all_funcs, deep_nest, wide) emitted byte-identical PTX at one
        // iteration and two, under both LLVM 15 and LLVM 21. Kept because a second
        // fixed-point pass is cheap next to the ptxas failure it guards against,
        // not because any case is known to need it.
        for _ in 0..2 {
            module
                .run_passes(
                    "always-inline,inline,globaldce,strip-dead-prototypes",
                    machine,
                    PassBuilderOptions::create(),
                )
                .map_err(|e| {
                    CudaError::PtxGenerationFailed(format!(
                        "libdevice cleanup passes: {}",
                        e.to_string()
                    ))
                })?;
        }
        Ok(())
    }

    // Silence unused-type warnings in paths not yet exercised.
    #[allow(dead_code)]
    fn _unused_types() -> (
        IntValue<'static>,
        PointerValue<'static>,
        BasicMetadataValueEnum<'static>,
    ) {
        unreachable!()
    }
}

// ---------------------------------------------------------------------------
// Runtime — cudarc driver API, lazy per-device module loading, launch.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod runtime {
    use super::{CudaError, DeviceRuntime};
    use cudarc::driver::{sys as cu_sys, CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::Ptx;

    pub fn load_ptx(device: usize, ptx: &str) -> Result<DeviceRuntime, CudaError> {
        let ctx = CudaContext::new(device)
            .map_err(|e| CudaError::DriverError(format!("context {device}: {e:?}")))?;

        let module = ctx
            .load_module(Ptx::from_src(ptx))
            .map_err(|e| CudaError::DriverError(format!("module load: {e:?}")))?;
        let function = module
            .load_function("alkahest_eval")
            .map_err(|e| CudaError::DriverError(format!("load_function: {e:?}")))?;

        Ok(DeviceRuntime {
            ctx,
            module,
            function,
        })
    }

    #[allow(deprecated)]
    pub fn launch(
        rt: &DeviceRuntime,
        inputs: &[&[f64]],
        output: &mut [f64],
        n_inputs: usize,
        n_pts: usize,
    ) -> Result<(), CudaError> {
        let stream = rt.ctx.default_stream();

        // SoA: concatenate each input column contiguously.
        let mut flat_in = Vec::with_capacity(n_inputs * n_pts);
        for col in inputs.iter().take(n_inputs) {
            flat_in.extend_from_slice(col);
        }

        let in_dev = stream
            .memcpy_stod(&flat_in)
            .map_err(|e| CudaError::DriverError(format!("H2D in: {e:?}")))?;
        let mut out_dev = stream
            .alloc_zeros::<f64>(n_pts)
            .map_err(|e| CudaError::DriverError(format!("alloc out: {e:?}")))?;

        let cfg = kernel_config(n_pts);
        let n_pts_u64 = n_pts as u64;
        unsafe {
            stream
                .launch_builder(&rt.function)
                .arg(&mut out_dev)
                .arg(&in_dev)
                .arg(&n_pts_u64)
                .launch(cfg)
                .map_err(|e| CudaError::LaunchError(format!("{e:?}")))?;
        }

        let host_out = stream
            .memcpy_dtov(&out_dev)
            .map_err(|e| CudaError::DriverError(format!("D2H out: {e:?}")))?;
        output.copy_from_slice(&host_out);
        Ok(())
    }

    pub fn launch_raw(
        rt: &DeviceRuntime,
        n_inputs: usize,
        in_dev_ptr: u64,
        out_dev_ptr: u64,
        n_pts: usize,
    ) -> Result<(), CudaError> {
        let stream = rt.ctx.default_stream();
        let cfg = kernel_config(n_pts);
        let n_pts_u64 = n_pts as u64;

        // Wrap the caller's device pointers as CudaSlice without taking ownership.
        // We leak both slices back to raw pointers after the launch so their Drop
        // doesn't free memory the caller still owns (DLPack contract).
        unsafe {
            let in_slice = stream
                .upgrade_device_ptr::<f64>(in_dev_ptr as cu_sys::CUdeviceptr, n_inputs * n_pts);
            let mut out_slice =
                stream.upgrade_device_ptr::<f64>(out_dev_ptr as cu_sys::CUdeviceptr, n_pts);

            let launch_res = stream
                .launch_builder(&rt.function)
                .arg(&mut out_slice)
                .arg(&in_slice)
                .arg(&n_pts_u64)
                .launch(cfg);

            // Release ownership back to caller regardless of launch outcome.
            let _ = in_slice.leak();
            let _ = out_slice.leak();

            launch_res.map_err(|e| CudaError::LaunchError(format!("{e:?}")))?;
            stream
                .synchronize()
                .map_err(|e| CudaError::DriverError(format!("sync: {e:?}")))?;
        }
        Ok(())
    }

    fn kernel_config(n_pts: usize) -> LaunchConfig {
        let block: u32 = 256;
        let n = n_pts as u32;
        let grid = n.div_ceil(block);
        LaunchConfig {
            grid_dim: (grid.max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn compile_cuda_produces_ptx() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.mul(vec![x, x]);
        let result = compile_cuda(expr, &[x], &pool);
        assert!(result.is_ok(), "compile_cuda failed: {:?}", result.err());
        let compiled = result.unwrap();
        assert!(!compiled.ptx.is_empty(), "PTX should not be empty");
        assert_eq!(compiled.n_inputs, 1);
    }

    #[test]
    fn cuda_compiled_fn_ptx_source() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let sin_x = pool.func("sin", vec![x]);
        let compiled = compile_cuda(sin_x, &[x], &pool).unwrap();
        let ptx = compiled.ptx_source();
        assert!(
            ptx.contains(".version"),
            "PTX should have .version directive"
        );
        assert!(ptx.contains("sm_86"), "PTX should target sm_86 (Ampere)");
    }

    #[test]
    fn cuda_error_display() {
        let e = CudaError::NvptxTargetUnavailable;
        assert!(e.to_string().contains("NVPTX"));
        let e2 = CudaError::NotImplemented("batch launch");
        assert!(e2.to_string().contains("batch launch"));
        let e3 = CudaError::LibdeviceNotFound("missing".to_string());
        assert!(e3.to_string().contains("libdevice"));
    }

    // Running-on-device tests live in alkahest-core/tests/nvptx_gpu.rs and
    // only run with `--features cuda` on a machine with CUDA + libdevice.
}
