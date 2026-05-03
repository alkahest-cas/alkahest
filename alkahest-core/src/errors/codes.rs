//! Central registry of all stable diagnostic codes for alkahest-core.
//!
//! Every code returned by an `AlkahestError::code()` implementation must appear
//! in `REGISTRY`.  Tests below assert no duplicates and ascending order within
//! each prefix.

/// Root cause of an error — informs remediation style, not type dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cause {
    UserInput,
    Domain,
    Unsupported,
    Resource,
    Internal,
}

pub struct ErrorSpec {
    pub code: &'static str,
    pub class: &'static str,
    pub cause: Cause,
    pub remediation: Option<&'static str>,
}

pub const REGISTRY: &[ErrorSpec] = &[
    // E-POLY — ConversionError
    ErrorSpec { code: "E-POLY-001", class: "ConversionError", cause: Cause::UserInput,   remediation: Some("remove the unsupported symbol, or declare it as a parameter") },
    ErrorSpec { code: "E-POLY-002", class: "ConversionError", cause: Cause::UserInput,   remediation: Some("all coefficients must be rational integers; rationalize or substitute") },
    ErrorSpec { code: "E-POLY-003", class: "ConversionError", cause: Cause::UserInput,   remediation: Some("only non-negative integer exponents are supported in poly_normal") },
    ErrorSpec { code: "E-POLY-004", class: "ConversionError", cause: Cause::UserInput,   remediation: Some("reduce the degree or switch to a sparse representation") },
    ErrorSpec { code: "E-POLY-005", class: "ConversionError", cause: Cause::UserInput,   remediation: Some("substitute a concrete integer for the exponent before calling poly_normal") },
    ErrorSpec { code: "E-POLY-006", class: "ConversionError", cause: Cause::Unsupported, remediation: Some("only polynomial expressions are supported; remove transcendental functions") },
    ErrorSpec { code: "E-POLY-007", class: "ConversionError", cause: Cause::Domain,      remediation: Some("ensure the denominator is non-zero before converting") },
    ErrorSpec { code: "E-POLY-008", class: "FactorError", cause: Cause::UserInput,   remediation: Some("factorization is only defined for non-zero polynomials") },
    ErrorSpec { code: "E-POLY-009", class: "FactorError", cause: Cause::UserInput,   remediation: Some("use a modulus ≥ 2 that fits in a machine word (FLINT nmod)") },
    ErrorSpec { code: "E-POLY-010", class: "FactorError", cause: Cause::Internal,    remediation: Some("report the polynomial as a minimal failing example") },
    // E-DIFF — DiffError (symbolic + forward-mode)
    ErrorSpec { code: "E-DIFF-001", class: "DiffError", cause: Cause::Unsupported, remediation: Some("register the function in PrimitiveRegistry, or use diff_forward with a custom rule") },
    ErrorSpec { code: "E-DIFF-002", class: "DiffError", cause: Cause::UserInput,   remediation: Some("symbolic exponents require the chain rule; use diff_forward for non-integer powers") },
    ErrorSpec { code: "E-DIFF-003", class: "DiffError", cause: Cause::Unsupported, remediation: Some("register the function in PrimitiveRegistry with diff_forward implemented") },
    ErrorSpec { code: "E-DIFF-004", class: "DiffError", cause: Cause::UserInput,   remediation: Some("substitute concrete values first; diff_forward requires integer exponents") },
    // E-INT — IntegrationError
    ErrorSpec { code: "E-INT-001", class: "IntegrationError", cause: Cause::Unsupported, remediation: Some("use a numeric integrator for arbitrary functions") },
    ErrorSpec { code: "E-INT-002", class: "IntegrationError", cause: Cause::Domain,      remediation: None },
    ErrorSpec { code: "E-INT-003", class: "IntegrationError", cause: Cause::Unsupported, remediation: Some("v1.1 supports sqrt(P(x)) only; higher-degree radicals planned for v2.0") },
    ErrorSpec { code: "E-INT-004", class: "IntegrationError", cause: Cause::Domain,      remediation: Some("no elementary antiderivative exists; use a numeric integrator or elliptic-integral library") },
    // E-MAT — MatrixError
    ErrorSpec { code: "E-MAT-001", class: "MatrixError", cause: Cause::UserInput, remediation: Some("check that row/column counts agree") },
    ErrorSpec { code: "E-MAT-002", class: "MatrixError", cause: Cause::UserInput, remediation: Some("use pseudo-inverse for rectangular matrices") },
    ErrorSpec { code: "E-MAT-003", class: "MatrixError", cause: Cause::Domain,    remediation: Some("check for linear dependence in the rows/columns") },
    // E-ODE — OdeError
    ErrorSpec { code: "E-ODE-001", class: "OdeError", cause: Cause::UserInput,   remediation: Some("number of state variables must equal number of RHS expressions") },
    ErrorSpec { code: "E-ODE-002", class: "OdeError", cause: Cause::UserInput,   remediation: Some("use lower_to_first_order() before passing to a solver") },
    ErrorSpec { code: "E-ODE-003", class: "OdeError", cause: Cause::Unsupported, remediation: Some("check differentiability of all functions in the system") },
    // E-DAE — DaeError
    ErrorSpec { code: "E-DAE-001", class: "DaeError", cause: Cause::Unsupported, remediation: Some("ensure all functions are differentiable before calling pantelides()") },
    ErrorSpec { code: "E-DAE-002", class: "DaeError", cause: Cause::UserInput,   remediation: Some("DAE index exceeds depth-10 limit; reformulate the model") },
    ErrorSpec { code: "E-DAE-003", class: "DaeError", cause: Cause::UserInput,   remediation: Some("check constraint count against variable count") },
    // E-SOLVE — SolverError (polynomial system solver + GPU Gröbner folded in)
    ErrorSpec { code: "E-SOLVE-001", class: "SolverError", cause: Cause::UserInput,   remediation: Some("ensure all equations are polynomial in the declared variables") },
    ErrorSpec { code: "E-SOLVE-002", class: "SolverError", cause: Cause::Unsupported, remediation: Some("only degree ≤ 2 univariate solving is implemented; Gröbner basis is still returned") },
    ErrorSpec { code: "E-SOLVE-003", class: "SolverError", cause: Cause::UserInput,   remediation: Some("provide one equation per variable") },
    ErrorSpec { code: "E-SOLVE-010", class: "SolverError", cause: Cause::Resource,    remediation: Some("check GPU availability; pass device_id=None to fall back to CPU") },
    ErrorSpec { code: "E-SOLVE-011", class: "SolverError", cause: Cause::Resource,    remediation: Some("CRT reconstruction failed; try adding more equations or use CPU path") },
    // E-JIT — JitError
    ErrorSpec { code: "E-JIT-001", class: "JitError", cause: Cause::Unsupported, remediation: Some("use eval_expr or simplify the expression before JIT") },
    ErrorSpec { code: "E-JIT-002", class: "JitError", cause: Cause::Resource,    remediation: Some("check LLVM 15 installation; run with RUST_LOG=debug for details") },
    ErrorSpec { code: "E-JIT-003", class: "JitError", cause: Cause::Resource,    remediation: Some("ensure LLVM_SYS_150_PREFIX is set correctly") },
    // E-CUDA — CudaError
    ErrorSpec { code: "E-CUDA-001", class: "CudaError", cause: Cause::Resource,    remediation: Some("rebuild LLVM with nvptx64 in LLVM_TARGETS_TO_BUILD") },
    ErrorSpec { code: "E-CUDA-002", class: "CudaError", cause: Cause::Unsupported, remediation: Some("inspect PTX; verify every primitive has CUDA lowering") },
    ErrorSpec { code: "E-CUDA-003", class: "CudaError", cause: Cause::Resource,    remediation: Some("run nvidia-smi; retry with CUDA_LAUNCH_BLOCKING=1") },
    ErrorSpec { code: "E-CUDA-004", class: "CudaError", cause: Cause::Unsupported, remediation: Some("V1.0 stub; track feature request") },
    ErrorSpec { code: "E-CUDA-005", class: "CudaError", cause: Cause::Resource,    remediation: Some("install nvidia-cuda-toolkit or set ALKAHEST_LIBDEVICE_PATH") },
    ErrorSpec { code: "E-CUDA-006", class: "CudaError", cause: Cause::Resource,    remediation: Some("check grid/block dimensions; rerun with compute-sanitizer") },
    // E-IO — IoError (formerly PoolPersistError with E-POOL-* codes)
    ErrorSpec { code: "E-IO-001", class: "IoError", cause: Cause::Resource,  remediation: None },
    ErrorSpec { code: "E-IO-002", class: "IoError", cause: Cause::UserInput, remediation: Some("file is not an alkahest pool; check the path or regenerate with ExprPool::checkpoint()") },
    ErrorSpec { code: "E-IO-003", class: "IoError", cause: Cause::UserInput, remediation: Some("run `alkahest migrate-pool` to upgrade the file, or regenerate from source") },
    ErrorSpec { code: "E-IO-004", class: "IoError", cause: Cause::Resource,  remediation: Some("file was truncated (likely a crash during checkpoint); rerun from source and checkpoint again") },
    ErrorSpec { code: "E-IO-005", class: "IoError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-IO-006", class: "IoError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-IO-007", class: "IoError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-IO-008", class: "IoError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-IO-009", class: "IoError", cause: Cause::UserInput, remediation: None },
    // E-MOD — ModularError (V2-1 Modular/CRT framework)
    ErrorSpec { code: "E-MOD-001", class: "ModularError", cause: Cause::UserInput,   remediation: Some("use a prime modulus p ≥ 2, e.g. 101, 1009, 32749") },
    ErrorSpec { code: "E-MOD-002", class: "ModularError", cause: Cause::UserInput,   remediation: Some("ensure all images share the same variable ordering and modulus") },
    ErrorSpec { code: "E-MOD-003", class: "ModularError", cause: Cause::UserInput,   remediation: Some("provide at least one (MultiPolyFp, prime) pair") },
    ErrorSpec { code: "E-MOD-004", class: "ModularError", cause: Cause::Unsupported, remediation: Some("provide more modular images so the prime product M exceeds 2 * max_coeff²") },
    // E-LAT — LatticeError (V2-6 LLL)
    ErrorSpec { code: "E-LAT-001", class: "LatticeError", cause: Cause::UserInput,   remediation: Some("pass a non-empty matrix of integer rows, all of equal length") },
    ErrorSpec { code: "E-LAT-002", class: "LatticeError", cause: Cause::UserInput,   remediation: Some("every row must lie in ℤ^m for fixed ambient dimension m") },
    ErrorSpec { code: "E-LAT-003", class: "LatticeError", cause: Cause::UserInput,   remediation: Some("pick δ strictly between ¼ and 1; the default δ = ¾ is standard") },
    ErrorSpec { code: "E-LAT-004", class: "LatticeError", cause: Cause::Unsupported, remediation: Some("check for rank deficiency; try a smaller basis or report a minimal reproducer") },
    // E-PSLQ — PslqError (V2-6 augmented-lattice relation heuristic)
    ErrorSpec { code: "E-PSLQ-001", class: "PslqError", cause: Cause::UserInput,   remediation: Some("pass at least two constants that might admit a linear dependence") },
    ErrorSpec { code: "E-PSLQ-002", class: "PslqError", cause: Cause::UserInput,   remediation: Some("literals must not truncate to zero — use higher precision or decimal strings") },
    ErrorSpec { code: "E-PSLQ-003", class: "PslqError", cause: Cause::UserInput,   remediation: Some("allocate at least 64 MPFR bits; ≈664 bits ≈ 200 decimal digits") },
    // E-PARSE — reserved for parser integration
    // E-DOMAIN — reserved; DomainError is Python-only pending Rust implementation
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn no_duplicate_codes() {
        let mut seen = HashSet::new();
        for spec in REGISTRY {
            assert!(
                seen.insert(spec.code),
                "duplicate error code in REGISTRY: {}",
                spec.code
            );
        }
    }

    #[test]
    fn codes_ascending_within_prefix() {
        let mut by_prefix: std::collections::BTreeMap<&str, Vec<u32>> =
            std::collections::BTreeMap::new();
        for spec in REGISTRY {
            if let Some(pos) = spec.code.rfind('-') {
                let prefix = &spec.code[..pos];
                if let Ok(num) = spec.code[pos + 1..].parse::<u32>() {
                    by_prefix.entry(prefix).or_default().push(num);
                }
            }
        }
        for (prefix, nums) in &by_prefix {
            let mut sorted = nums.clone();
            sorted.sort_unstable();
            assert_eq!(
                nums, &sorted,
                "codes under prefix {prefix} are not in ascending order"
            );
        }
    }
}
