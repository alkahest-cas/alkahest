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
    // E-SIMPLIFY — AssumptionError
    ErrorSpec { code: "E-SIMPLIFY-001", class: "AssumptionError", cause: Cause::Domain, remediation: Some("remove the conflicting refinement or create a separate AssumptionContext") },
    // E-SERIES — SeriesError (V2-15 truncated expansions)
    ErrorSpec { code: "E-SERIES-001", class: "SeriesError", cause: Cause::Unsupported, remediation: Some("ensure all functions are registered primitives with differentiation rules") },
    ErrorSpec { code: "E-SERIES-002", class: "SeriesError", cause: Cause::UserInput,   remediation: Some("pass order >= 1 (exclusive truncation degree in x)") },
    // A `series` call that ran past its work ceiling (or an active budget) before
    // reaching the requested order. Refusing is the point: coefficients are formed by
    // repeated differentiation without re-simplifying, so a nested radical grows by a
    // constant factor per coefficient, and returning the prefix under the requested
    // `O(h^order)` label would understate the remainder rather than admit the miss.
    // Carried out of band on `SeriesError::InvalidOrder` (exhaustive public enum) —
    // see `calculus::series::take_series_refusal`.
    ErrorSpec { code: "E-SERIES-003", class: "SeriesRefusal", cause: Cause::Resource,  remediation: Some("ask for a lower order, raise the budget, or rewrite the expression so its repeated derivatives close") },
    // E-INT — IntegrationError
    ErrorSpec { code: "E-INT-001", class: "IntegrationError", cause: Cause::Unsupported, remediation: Some("use a numeric integrator for arbitrary functions") },
    ErrorSpec { code: "E-INT-002", class: "IntegrationError", cause: Cause::Domain,      remediation: None },
    ErrorSpec { code: "E-INT-003", class: "IntegrationError", cause: Cause::Unsupported, remediation: Some("v1.1 supports sqrt(P(x)) only; higher-degree radicals planned for v2.0") },
    ErrorSpec { code: "E-INT-004", class: "IntegrationError", cause: Cause::Domain,      remediation: Some("no elementary antiderivative exists; use a numeric integrator or elliptic-integral library") },
    // E-MAT — MatrixError
    ErrorSpec { code: "E-MAT-001", class: "MatrixError", cause: Cause::UserInput, remediation: Some("check that row/column counts agree") },
    ErrorSpec { code: "E-MAT-002", class: "MatrixError", cause: Cause::UserInput, remediation: Some("use pseudo-inverse for rectangular matrices") },
    ErrorSpec { code: "E-MAT-003", class: "MatrixError", cause: Cause::Domain,    remediation: Some("check for linear dependence in the rows/columns") },
    ErrorSpec { code: "E-MAT-004", class: "MatrixError", cause: Cause::Unsupported, remediation: Some("rewrite the entries into a form whose determinant's vanishing is decidable, or substitute concrete values") },
    // V2-17 — EigenError
    ErrorSpec { code: "E-EIGEN-001", class: "EigenError", cause: Cause::UserInput,   remediation: Some("pass a square n×n matrix") },
    ErrorSpec { code: "E-EIGEN-002", class: "EigenError", cause: Cause::UserInput,   remediation: Some("ensure det(λI−M) is a ℤ-polynomial in the fresh λ variable") },
    ErrorSpec { code: "E-EIGEN-003", class: "EigenError", cause: Cause::Internal,    remediation: Some("report the polynomial as a minimal failing example") },
    ErrorSpec { code: "E-EIGEN-004", class: "EigenError", cause: Cause::Unsupported, remediation: Some("irreducible characteristic factors of degree > 2 require a future algebraic-number extension") },
    ErrorSpec { code: "E-EIGEN-005", class: "EigenError", cause: Cause::Domain,    remediation: Some("the matrix is defective or the eigenbasis is incomplete") },
    ErrorSpec { code: "E-EIGEN-006", class: "EigenError", cause: Cause::Unsupported, remediation: Some("nullspace elimination failed; try a purely rational or ℚ(i) spectrum") },
    ErrorSpec { code: "E-EIGEN-007", class: "EigenError", cause: Cause::Domain,    remediation: Some("eigenvector matrix is singular; check multiplicities") },
    // E-LINALG — LinearAlgebraError (symbolic LA coverage)
    ErrorSpec { code: "E-LINALG-001", class: "LinearAlgebraError", cause: Cause::UserInput,   remediation: Some("pass a square n×n matrix") },
    ErrorSpec { code: "E-LINALG-002", class: "LinearAlgebraError", cause: Cause::Unsupported, remediation: Some("nullspace elimination failed; try rational entries") },
    ErrorSpec { code: "E-LINALG-003", class: "LinearAlgebraError", cause: Cause::Domain,      remediation: Some("Cholesky requires symmetric positive definite input") },
    ErrorSpec { code: "E-LINALG-004", class: "LinearAlgebraError", cause: Cause::UserInput,   remediation: Some("ensure det(λI−M) is a polynomial in λ") },
    ErrorSpec { code: "E-LINALG-005", class: "LinearAlgebraError", cause: Cause::Internal,    remediation: None },
    ErrorSpec { code: "E-LINALG-006", class: "LinearAlgebraError", cause: Cause::Unsupported, remediation: Some("irreducible factor of degree > 2 in minimal polynomial") },
    ErrorSpec { code: "E-LINALG-007", class: "LinearAlgebraError", cause: Cause::Unsupported, remediation: Some("use rational entries for Smith-based decompositions") },
    ErrorSpec { code: "E-LINALG-008", class: "LinearAlgebraError", cause: Cause::Domain,      remediation: Some("similarity transform matrix is singular") },
    ErrorSpec { code: "E-LINALG-009", class: "LinearAlgebraError", cause: Cause::UserInput,   remediation: Some("matrix entries must be rational constants") },
    // Zero-testing over a transcendental extension is not decidable in general, so
    // elimination can reach an entry it can neither prove zero nor prove non-zero.
    // Refusing is the point: treating "unknown" as "non-zero" is what produced a
    // confident wrong rank (and a false inconsistency signature) before this code existed.
    // Raised through `EigenError` too (`eigenvects`), which shares the elimination it
    // refuses in; the class below names where the code is defined, not every route.
    ErrorSpec { code: "E-LINALG-010", class: "LinearAlgebraError", cause: Cause::Unsupported, remediation: Some("rewrite the entry into a form whose vanishing is decidable, or substitute concrete values for the parameters") },
    // E-ODE — OdeError
    ErrorSpec { code: "E-ODE-001", class: "OdeError", cause: Cause::UserInput,   remediation: Some("number of state variables must equal number of RHS expressions") },
    ErrorSpec { code: "E-ODE-002", class: "OdeError", cause: Cause::UserInput,   remediation: Some("use lower_to_first_order() before passing to a solver") },
    ErrorSpec { code: "E-ODE-003", class: "OdeError", cause: Cause::Unsupported, remediation: Some("check differentiability of all functions in the system") },
    ErrorSpec { code: "E-ODE-010", class: "DsolveError", cause: Cause::Unsupported, remediation: None },
    ErrorSpec { code: "E-ODE-011", class: "DsolveError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-ODE-012", class: "DsolveError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-ODE-020", class: "NumericOdeError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-ODE-021", class: "NumericOdeError", cause: Cause::Resource, remediation: None },
    ErrorSpec { code: "E-ODE-022", class: "NumericOdeError", cause: Cause::Resource, remediation: None },
    ErrorSpec { code: "E-ODE-023", class: "NumericOdeError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-ODE-024", class: "NumericOdeError", cause: Cause::Unsupported, remediation: None },
    ErrorSpec { code: "E-ODE-025", class: "NumericOdeError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-ODE-026", class: "NumericOdeError", cause: Cause::UserInput, remediation: None },
    // E-DAE — DaeError
    ErrorSpec { code: "E-DAE-001", class: "DaeError", cause: Cause::Unsupported, remediation: Some("ensure all functions are differentiable before calling pantelides()") },
    ErrorSpec { code: "E-DAE-002", class: "DaeError", cause: Cause::UserInput,   remediation: Some("DAE index exceeds depth-10 limit; reformulate the model") },
    ErrorSpec { code: "E-DAE-003", class: "DaeError", cause: Cause::UserInput,   remediation: Some("check constraint count against variable count") },
    // E-DIFFALG — DiffAlgError (V2-13 differential algebra / Rosenfeld–Gröbner)
    ErrorSpec { code: "E-DIFFALG-001", class: "DiffAlgError", cause: Cause::Unsupported, remediation: Some("ensure the DAE is polynomial in its state and derivative symbols") },
    ErrorSpec { code: "E-DIFFALG-002", class: "DiffAlgError", cause: Cause::UserInput,   remediation: Some("declare all jet variables; remove transcendental functions") },
    ErrorSpec { code: "E-DIFFALG-003", class: "DiffAlgError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-SOLVE-001", class: "SolverError", cause: Cause::UserInput,   remediation: Some("ensure all equations are polynomial in the declared variables") },
    ErrorSpec { code: "E-SOLVE-002", class: "SolverError", cause: Cause::Unsupported, remediation: Some("only degree ≤ 2 univariate solving is implemented; Gröbner basis is still returned") },
    ErrorSpec { code: "E-SOLVE-003", class: "SolverError", cause: Cause::UserInput,   remediation: Some("provide one equation per variable") },
    // `triangularize` extracts one polynomial per main variable, so two basis
    // generators sharing a main variable lose one of them and the chain describes
    // a larger variety than the input system.  Splitting on the initials
    // (Lazard–Kalkbrener) is what would decompose those ideals; refusing is the
    // point until it exists.  Travels inside `SolverError::NotPolynomial` — see
    // `solver::regular_chains::TriangularizeRefusal`.
    ErrorSpec { code: "E-SOLVE-004", class: "TriangularizeRefusal", cause: Cause::Unsupported, remediation: Some("this ideal needs a splitting triangular decomposition (Lazard–Kalkbrener on the initials), which is not implemented; use GroebnerBasis::compute or primary_decomposition instead") },
    ErrorSpec { code: "E-SOLVE-010", class: "SolverError", cause: Cause::Resource,    remediation: Some("check GPU availability; pass device_id=None to fall back to CPU") },
    ErrorSpec { code: "E-SOLVE-011", class: "SolverError", cause: Cause::Resource,    remediation: Some("CRT reconstruction failed; try adding more equations or use CPU path") },
    // E-IDEAL — PrimaryDecompositionError (ideal/primary.rs) and IdealRefusal
    ErrorSpec { code: "E-IDEAL-001", class: "PrimaryDecompositionError", cause: Cause::UserInput,   remediation: Some("pass at least one generator") },
    ErrorSpec { code: "E-IDEAL-002", class: "PrimaryDecompositionError", cause: Cause::UserInput,   remediation: Some("all generators must be polynomials in the same variable list") },
    ErrorSpec { code: "E-IDEAL-003", class: "PrimaryDecompositionError", cause: Cause::Resource,    remediation: Some("the saturation split recursed past its depth limit; simplify the generating set") },
    ErrorSpec { code: "E-IDEAL-004", class: "PrimaryDecompositionError", cause: Cause::Internal,    remediation: Some("report the generating set as a minimal failing example") },
    // √I over an arbitrary ideal needs Gianni–Trager–Zacharias (or a
    // characteristic-set method).  Only monomial, principal and zero-dimensional
    // ideals are certified; outside those, returning the input unchanged would be
    // asserting √I = I with no justification, so the routine refuses instead.
    ErrorSpec { code: "E-IDEAL-005", class: "IdealRefusal", cause: Cause::Unsupported, remediation: Some("radical is certified for monomial, principal and zero-dimensional ideals; intersect the associated primes of a primary decomposition if one is available") },
    ErrorSpec { code: "E-IDEAL-006", class: "IdealRefusal", cause: Cause::Unsupported, remediation: Some("primary decomposition is certified for monomial and principal ideals, for saturation/CRT splits of them, and for shape-position zero-dimensional ideals; no general algorithm is implemented") },
    // E-HOMOTOPY — HomotopyError (V2-14 numerical algebraic geometry)
    ErrorSpec { code: "E-HOMOTOPY-002", class: "HomotopyError", cause: Cause::Unsupported, remediation: Some("raise HomotopyOpts.max_bezout_paths or use mixed-volume continuation for deficient systems") },
    ErrorSpec { code: "E-HOMOTOPY-003", class: "HomotopyError", cause: Cause::Resource,    remediation: Some("try HomotopyOpts.gamma_angle_seed or rescale equations") },
    ErrorSpec { code: "E-HOMOTOPY-004", class: "HomotopyError", cause: Cause::Resource,    remediation: Some("adjust predictor step or increase max_tracker_steps") },
    // E-JIT — JitError
    ErrorSpec { code: "E-JIT-001", class: "JitError", cause: Cause::Unsupported, remediation: Some("use eval_expr or simplify the expression before JIT") },
    ErrorSpec { code: "E-JIT-002", class: "JitError", cause: Cause::Resource,    remediation: Some("check LLVM 15 installation; run with RUST_LOG=debug for details") },
    ErrorSpec { code: "E-JIT-003", class: "JitError", cause: Cause::Resource,    remediation: Some("ensure LLVM_SYS_150_PREFIX is set correctly") },
    ErrorSpec { code: "E-JIT-004", class: "JitError", cause: Cause::UserInput, remediation: None },
    // E-CAD — CadError (V2-9 QE / cylindrical decomposition)
    ErrorSpec {
        code: "E-CAD-001",
        class: "CadError",
        cause: Cause::Unsupported,
        // Two distinct causes now share this code, and the remediation has to
        // cover both: outside the supported fragment (≤ 2 variables, ≤ 2
        // quantifiers, polynomial atoms), or inside it but undecidable by the
        // sample points available — a non-strict atom whose only solutions sit
        // at an irrational boundary. The second is a *refusal to guess*: before
        // 3.8 that case silently answered, which produced false universals.
        remediation: Some(
            "keep to polynomial atoms in at most two real variables with at most two quantifiers; if the sentence is already in that fragment, its solutions may lie only at an irrational boundary point, which cannot be tested exactly — substitute concrete values, use a strict inequality, or hand it to an SMT solver via alkahest.smt",
        ),
    },
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
    // E-LOGIC — first-order formulas (V3-3)
    ErrorSpec { code: "E-LOGIC-001", class: "LogicError", cause: Cause::UserInput, remediation: Some("pass a predicate or quantified Expr; use pool.gt/… or And/Or/Not") },
    // E-PSLQ — PslqError (V2-6 augmented-lattice relation heuristic)
    ErrorSpec { code: "E-PSLQ-001", class: "PslqError", cause: Cause::UserInput,   remediation: Some("pass at least two constants that might admit a linear dependence") },
    ErrorSpec { code: "E-PSLQ-002", class: "PslqError", cause: Cause::UserInput,   remediation: Some("literals must not truncate to zero — use higher precision or decimal strings") },
    ErrorSpec { code: "E-PSLQ-003", class: "PslqError", cause: Cause::UserInput,   remediation: Some("allocate at least 64 MPFR bits; ≈664 bits ≈ 200 decimal digits") },
    // E-RSOLVE — RsolveError (V2-18 difference equations / rsolve)
    ErrorSpec { code: "E-RSOLVE-001", class: "RsolveError", cause: Cause::UserInput,   remediation: Some("write the recurrence as a sum of pool.func(seq, [n ± integer]) shifts plus a polynomial in n, then call rsolve(equation, n, seq_name)") },
    ErrorSpec { code: "E-RSOLVE-002", class: "RsolveError", cause: Cause::UserInput,   remediation: Some("each addend may contain at most one sequence application; avoid products like n*f(n)") },
    ErrorSpec { code: "E-RSOLVE-003", class: "RsolveError", cause: Cause::UserInput,   remediation: Some("clear denominators; the rhs must be a polynomial in the recurrence index") },
    ErrorSpec { code: "E-RSOLVE-004", class: "RsolveError", cause: Cause::Unsupported, remediation: Some("order > 2 non-homogeneous systems and some characteristic factorizations are not implemented yet") },
    ErrorSpec { code: "E-RSOLVE-005", class: "RsolveError", cause: Cause::UserInput,   remediation: Some("pass exactly order-many initial samples as a dict n → Expr value") },
    // E-DIOPH — DiophantineError (V2-19 linear / Pell / sum-of-squares)
    ErrorSpec { code: "E-DIOPH-001", class: "DiophantineError", cause: Cause::UserInput,   remediation: Some("pass one polynomial equation = 0 in two integer symbols") },
    ErrorSpec { code: "E-DIOPH-002", class: "DiophantineError", cause: Cause::UserInput,   remediation: Some("clear denominators; coefficients must be rational integers") },
    ErrorSpec { code: "E-DIOPH-003", class: "DiophantineError", cause: Cause::Unsupported, remediation: Some("supported: linear 2-variable, x²+y²=n, x²−D·y²=N including unit Pell (no xy term); very large integers may be unsupported") },
    ErrorSpec { code: "E-DIOPH-004", class: "DiophantineError", cause: Cause::Domain,      remediation: Some("check gcd divisibility (linear) or solvability over ℤ (quadratic)") },
    // E-PROD — ProductError (V2-22 symbolic discrete products)
    ErrorSpec { code: "E-PROD-001", class: "ProductError", cause: Cause::Unsupported, remediation: Some("reduce the term to ℚ(k) with only ℤ-linear factors in k (no extra symbols in k)") },
    ErrorSpec { code: "E-PROD-002", class: "ProductError", cause: Cause::Internal,    remediation: Some("report the polynomial as a minimal failing example") },
    ErrorSpec { code: "E-PROD-003", class: "ProductError", cause: Cause::Unsupported, remediation: Some("ℤ-irreducible factors of degree ≥ 2 in k require a Gamma extension beyond this path") },
    ErrorSpec { code: "E-PROD-004", class: "ProductError", cause: Cause::UserInput,   remediation: Some("check that lo/hi share the ExprPool with the summation index symbol") },
    // E-NT — NumberTheoryError (V3-1)
    ErrorSpec { code: "E-NT-001", class: "NumberTheoryError", cause: Cause::UserInput, remediation: Some("pass decimal strings parsable into fmpz") },
    ErrorSpec { code: "E-NT-002", class: "NumberTheoryError", cause: Cause::Domain, remediation: Some("check positivity / parity constraints on arguments") },
    ErrorSpec { code: "E-NT-003", class: "NumberTheoryError", cause: Cause::Domain, remediation: Some("adjust residue, base, or root degree until a modular solution exists") },
    ErrorSpec { code: "E-NT-004", class: "NumberTheoryError", cause: Cause::Domain, remediation: Some("use prime moduli for discrete_log/nthroot_mod as documented") },
    ErrorSpec { code: "E-NT-005", class: "NumberTheoryError", cause: Cause::Unsupported, remediation: Some("use quadratic roots or gcd(k,p−1)=1; general radicals require more machinery") },
    // E-PARSE — expression parser (V2-21)
    ErrorSpec { code: "E-PARSE-001", class: "ParseError", cause: Cause::UserInput,   remediation: Some("only ASCII arithmetic expressions are supported") },
    ErrorSpec { code: "E-PARSE-002", class: "ParseError", cause: Cause::UserInput,   remediation: Some("check parentheses and operator placement") },
    ErrorSpec { code: "E-PARSE-003", class: "ParseError", cause: Cause::UserInput,   remediation: Some("use a known function: sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, atan2, exp, log, sqrt, abs, sign, floor, ceil, round, erf, erfc, gamma, lambert_w, digamma, bessel_j0, bessel_j1") },
    ErrorSpec { code: "E-PARSE-004", class: "ParseError", cause: Cause::Resource,    remediation: Some("flatten the expression — deeply nested parentheses, prefix signs or function calls exceed the parser's recursion budget") },
    ErrorSpec { code: "E-EVAL-001", class: "EvalError", cause: Cause::UserInput,   remediation: Some("bind every free symbol before evaluation") },
    ErrorSpec { code: "E-EVAL-002", class: "EvalError", cause: Cause::UserInput,   remediation: Some("use mode='f64' or 'complex' for float literals") },
    ErrorSpec { code: "E-EVAL-003", class: "EvalError", cause: Cause::UserInput,   remediation: Some("only integer exponents are supported in exact mode") },
    ErrorSpec { code: "E-EVAL-004", class: "EvalError", cause: Cause::Domain,      remediation: Some("0 to a negative power is undefined") },
    ErrorSpec { code: "E-EVAL-005", class: "EvalError", cause: Cause::Unsupported, remediation: Some("register the function or use a supported evaluation mode") },
    ErrorSpec { code: "E-EVAL-006", class: "EvalError", cause: Cause::Unsupported, remediation: Some("this expression form is not evaluable in the requested mode") },
    ErrorSpec { code: "E-EVAL-007", class: "EvalError", cause: Cause::UserInput,   remediation: Some("check predicate arity") },
    ErrorSpec { code: "E-EVAL-008", class: "EvalError", cause: Cause::Domain,      remediation: Some("predicate truth value is not uniform over the input domain") },
    ErrorSpec { code: "E-EVAL-009", class: "EvalError", cause: Cause::Domain,      remediation: Some("result is not finite") },
    ErrorSpec { code: "E-EVAL-010", class: "EvalError", cause: Cause::Domain,      remediation: Some("interval evaluation failed or branch is indeterminate") },
    ErrorSpec { code: "E-EVAL-011", class: "EvalError", cause: Cause::Domain,      remediation: Some("principal Arg/log branch cut — expression stays unevaluated at this point") },
    ErrorSpec { code: "E-RESIDUE-001", class: "ResidueError", cause: Cause::UserInput,   remediation: Some("input must be a rational function of the variable over ℚ") },
    ErrorSpec { code: "E-RESIDUE-002", class: "ResidueError", cause: Cause::Domain,      remediation: Some("denominator must be non-zero") },
    ErrorSpec { code: "E-RESIDUE-003", class: "ResidueError", cause: Cause::Unsupported, remediation: Some("pole order exceeds supported bound; essential singularities are out of scope") },
    ErrorSpec { code: "E-RESIDUE-004", class: "ResidueError", cause: Cause::Domain,      remediation: Some("division by zero during Laurent coefficient extraction") },
    // E-BUDGET — BudgetError (P1 search plumbing item 4: budgets/cancellation/determinism)
    ErrorSpec { code: "E-BUDGET-001", class: "BudgetError", cause: Cause::Resource, remediation: Some("raise Budget(wall_ms=...), or accept a heuristic/numeric result for this candidate instead of an exact one") },
    ErrorSpec { code: "E-BUDGET-002", class: "BudgetError", cause: Cause::Resource, remediation: Some("raise Budget(max_steps=...), or accept a partial/heuristic result for this candidate instead of an exact one") },
    ErrorSpec { code: "E-BUDGET-003", class: "BudgetError", cause: Cause::Resource, remediation: Some("call alkahest.clear_cancel() (Python) or budget::clear_cancel() (Rust) before starting the next candidate") },
    // E-DEPTH — DepthLimitError (expression nesting ceiling; see kernel::depth).
    // Resource, not UserInput: the expression is well-formed, we decline to
    // recurse over it because a native stack overflow would kill the process.
    ErrorSpec { code: "E-DEPTH-001", class: "DepthLimitError", cause: Cause::Resource, remediation: Some("rebuild the expression with less nesting (a balanced n-ary Add is shallow where a chain of binary ones is not), or process it in smaller pieces") },
    // E-DOMAIN — reserved; DomainError is Python-only pending Rust implementation
    // E-SOS — SosError (P1 item 8: positivity certificates / Positivstellensatz)
    ErrorSpec { code: "E-SOS-001", class: "SosError", cause: Cause::UserInput,   remediation: Some("positivity certificates are for polynomials in the listed variables; expand or clear denominators first, and pass every symbol that occurs as a variable") },
    ErrorSpec { code: "E-SOS-002", class: "SosError", cause: Cause::Unsupported, remediation: Some("record this as unknown, not as a closed branch: raise basis_degree (unconstrained) or level (constrained); the search covers the diagonally dominant subcone, so this is not a proof that no SOS decomposition exists, and still less that the inequality is false — alkahest.decide is the complete (and far more expensive) fallback") },
    ErrorSpec { code: "E-SOS-003", class: "SosError", cause: Cause::UserInput,   remediation: Some("the witness point in the message satisfies the constraints and makes the target negative; the claim is false as stated") },
    ErrorSpec { code: "E-SOS-004", class: "SosError", cause: Cause::UserInput,   remediation: Some("pass at least one variable, and keep basis_degree/level within the supported range") },
    ErrorSpec { code: "E-SOS-005", class: "SosError", cause: Cause::Internal,    remediation: Some("internal: report the target and constraints as a minimal failing example") },
    // E-HOLO — HolonomicError (P1 item 7: creative telescoping / Zeilberger's algorithm)
    ErrorSpec { code: "E-HOLO-001", class: "HolonomicError", cause: Cause::UserInput,   remediation: Some("rewrite the term as R(n,k)*z**k*w**n*prod(gamma(a*n + b*k + c)**e) with integer a, b and rational c; supported function heads are gamma, factorial, binomial, pochhammer") },
    ErrorSpec { code: "E-HOLO-002", class: "HolonomicError", cause: Cause::Resource,    remediation: Some("raise max_order and/or max_degree in ZeilbergerOpts; if the term genuinely has no such recurrence within reach, Zeilberger's algorithm does not apply") },
    ErrorSpec { code: "E-HOLO-003", class: "HolonomicError", cause: Cause::Internal,    remediation: Some("internal: report the term as a minimal failing example") },
    ErrorSpec { code: "E-HOLO-004", class: "HolonomicError", cause: Cause::UserInput,   remediation: Some("n and k must be distinct symbols; max_order and max_degree must be positive") },
    // E-HOLO-005 is Python-only (`python/alkahest/_guess_holonomic.py`): a fit
    // the supplied terms cannot support. Registering a code no Rust
    // `AlkahestError` impl returns would fail `scripts/check_error_codes.py`.
    ErrorSpec { code: "E-HOLO-006", class: "HolonomicError", cause: Cause::UserInput,   remediation: Some("the modulus must be p**k with p prime, k >= 1 and p**k < 2**62; for a composite modulus, evaluate at each prime power and recombine by CRT") },
    ErrorSpec { code: "E-HOLO-007", class: "HolonomicError", cause: Cause::UserInput,   remediation: Some("no modulus repairs this: the recurrence itself leaves Z_p at that index. Supply more initial terms so the evaluation starts past it, use a recurrence whose leading coefficient does not vanish there, or accept that the sequence is not p-integral and rescale it") },
    ErrorSpec { code: "E-HOLO-008", class: "HolonomicError", cause: Cause::Resource,    remediation: Some("lower k, use a smaller prime, or ask for an index the recurrence reaches without crossing so many singular steps") },
    // E-HOLO-02x — QHolonomicError (M4b: q-analogue creative telescoping). A
    // separate block so a caller can tell which of the two engines refused.
    ErrorSpec { code: "E-HOLO-020", class: "HolonomicError", cause: Cause::UserInput,   remediation: Some("write the summand with qbinomial(N, K), qpochhammer(u, d, v), powers of q with a degree-2 exponent in n and k, and rational functions of q, q**n and q**k; a bare n or k outside an exponent is not q-hypergeometric") },
    ErrorSpec { code: "E-HOLO-021", class: "HolonomicError", cause: Cause::Resource,    remediation: Some("raise max_order and/or max_degree; if the sum genuinely satisfies no such q-recurrence, q-Zeilberger does not apply") },
    ErrorSpec { code: "E-HOLO-022", class: "HolonomicError", cause: Cause::Internal,    remediation: Some("internal: report the term as a minimal failing example") },
    ErrorSpec { code: "E-HOLO-023", class: "HolonomicError", cause: Cause::UserInput,   remediation: Some("q, n and k must be three distinct symbols; max_order and max_degree must be at least 1; a q-Pochhammer base step must be at least 1") },
    ErrorSpec { code: "E-HOLO-024", class: "HolonomicError", cause: Cause::Unsupported, remediation: Some("the term is q-hypergeometric in shape but its shift quotient is not a rational function of q**n and q**k — e.g. (q; q**2)_k shifted in k. No algorithm in this family applies; close the branch") },
    // E-HOLO-04x — Telescoping2dError (M4: double-sum / Apagodu-Zeilberger
    // creative telescoping). A separate block so a caller can tell which of
    // the three engines refused.
    ErrorSpec { code: "E-HOLO-040", class: "HolonomicError", cause: Cause::UserInput,   remediation: Some("rewrite the term as R(n,j,k)*z1**j*z2**k*w**n*prod(gamma(a*n+b*j+c*k+d)**e) with integer a, b, c; supported function heads are gamma, factorial, binomial, pochhammer") },
    ErrorSpec { code: "E-HOLO-041", class: "HolonomicError", cause: Cause::Resource,    remediation: Some("raise max_order, max_a_degree and/or max_cert_degree in Telescoping2dOpts; if the term genuinely has no such double-sum certificate within reach — or needs a certificate denominator this module's fixed-denominator ansatz cannot represent — this method does not apply") },
    ErrorSpec { code: "E-HOLO-042", class: "HolonomicError", cause: Cause::UserInput,   remediation: Some("n, j and k must be three distinct symbols") },
    // E-SMT — SmtError (P2 item 3: SMT/SAT bridge).
    //
    // Only the code Rust actually raises is registered here.  The rest of the
    // family lives entirely in `python/alkahest/smt.py`, which drives the solver
    // process: E-SMT-001 (no solver binary on PATH), E-SMT-003 (a model value —
    // a `root-obj` algebraic number — that cannot be lifted exactly), and
    // E-SMT-004 (a model that failed exact back-substitution).  Registering a
    // code no Rust `AlkahestError` impl returns would fail
    // `scripts/check_error_codes.py`; `E-BATCH-001` in `python/alkahest/_batch.py`
    // is the same precedent.
    ErrorSpec { code: "E-SMT-002", class: "SmtError", cause: Cause::Unsupported, remediation: Some("check alkahest.smt.supported(formula) before exporting; the fragment is polynomial (in)equalities over Int/Real symbols plus boolean structure and quantifiers, and float literals are refused because they are not the exact question they look like") },
    // Codes raised by alkahest-core that were never registered here. The registry
    // is the contract `scripts/check_error_codes.py` enforces, so an unregistered
    // code is an undocumented one. Remediation text is taken from each type's own
    // `AlkahestError::remediation` impl rather than re-worded, so the two cannot
    // disagree about what a caller should do.
    // E-ASYMPT — AsymptoticError
    ErrorSpec { code: "E-ASYMPT-001", class: "AsymptoticError", cause: Cause::UserInput, remediation: Some("pass n_terms >= 1") },
    ErrorSpec { code: "E-ASYMPT-002", class: "AsymptoticError", cause: Cause::UserInput, remediation: Some("the function may not be analytic/Laurent-expandable at infinity; try a simpler form or fewer terms") },
    ErrorSpec { code: "E-ASYMPT-003", class: "AsymptoticError", cause: Cause::UserInput, remediation: Some("ensure all functions are registered primitives with differentiation rules") },
    ErrorSpec { code: "E-ASYMPT-004", class: "AsymptoticError", cause: Cause::UserInput, remediation: Some("the expansion could not be numerically verified at large x; the function may have an oscillatory or non-power-scale tail") },
    ErrorSpec { code: "E-ASYMPT-005", class: "AsymptoticError", cause: Cause::Unsupported, remediation: Some("exp/log scale hierarchies and Gamma/Stirling asymptotics are out of scope for asymptotic_expand; power-scale (rational/algebraic) and single log/exp peels are supported. For the asymptotics of a *sum* use experimental.euler_maclaurin, which also reaches Stirling via the sum of log k") },
    // E-FPS — FpsError
    ErrorSpec { code: "E-FPS-001", class: "FpsError", cause: Cause::Domain, remediation: None },
    ErrorSpec { code: "E-FPS-002", class: "FpsError", cause: Cause::Domain, remediation: None },
    ErrorSpec { code: "E-FPS-003", class: "FpsError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-FPS-004", class: "FpsError", cause: Cause::Domain, remediation: None },
    ErrorSpec { code: "E-FPS-005", class: "FpsError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-FPS-006", class: "FpsError", cause: Cause::Domain, remediation: None },
    ErrorSpec { code: "E-FPS-007", class: "FpsError", cause: Cause::UserInput, remediation: None },
    // E-INTERP — SparseInterpError
    ErrorSpec { code: "E-INTERP-001", class: "SparseInterpError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-INTERP-002", class: "SparseInterpError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-INTERP-003", class: "SparseInterpError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-INTERP-004", class: "SparseInterpError", cause: Cause::Domain, remediation: None },
    ErrorSpec { code: "E-INTERP-010", class: "SparseGcdError", cause: Cause::Unsupported, remediation: None },
    ErrorSpec { code: "E-INTERP-011", class: "SparseGcdError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-INTERP-012", class: "SparseGcdError", cause: Cause::Internal, remediation: None },
    // E-LIMIT — LimitError
    ErrorSpec { code: "E-LIMIT-001", class: "LimitError", cause: Cause::UserInput, remediation: Some("increase truncation order indirectly by simplifying the expression, or rewrite using standard limits") },
    ErrorSpec { code: "E-LIMIT-002", class: "LimitError", cause: Cause::UserInput, remediation: Some("ensure primitives have differentiation rules, or simplify before taking the limit") },
    ErrorSpec { code: "E-LIMIT-003", class: "LimitError", cause: Cause::UserInput, remediation: Some("use LimitDirection::Plus or Minus matching the desired one-sided approach") },
    ErrorSpec { code: "E-LIMIT-004", class: "LimitError", cause: Cause::Resource, remediation: Some("try manual algebra (quotient form, cancellations) or split into simpler sub-expressions") },
    ErrorSpec { code: "E-LIMIT-005", class: "LimitError", cause: Cause::Unsupported, remediation: Some("limit could not be computed — try manual algebra, or the expression may involve oscillation or non-comparable growth not yet handled") },
    // E-NFM — NormalFormError
    ErrorSpec { code: "E-NFM-001", class: "NormalFormError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-NFM-002", class: "NormalFormError", cause: Cause::Unsupported, remediation: None },
    // E-REC — LinearRecurrenceError
    ErrorSpec { code: "E-REC-001", class: "LinearRecurrenceError", cause: Cause::Unsupported, remediation: None },
    ErrorSpec { code: "E-REC-002", class: "LinearRecurrenceError", cause: Cause::UserInput, remediation: None },
    // E-RES — ResultantError
    ErrorSpec { code: "E-RES-001", class: "ResultantError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-RES-003", class: "ResultantError", cause: Cause::UserInput, remediation: None },
    // E-ROOT — RealRootError
    ErrorSpec { code: "E-ROOT-001", class: "RealRootError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-ROOT-002", class: "RealRootError", cause: Cause::Domain, remediation: None },
    // E-SUM — SumError
    ErrorSpec { code: "E-SUM-001", class: "SumError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-SUM-002", class: "SumError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-SUM-003", class: "SumError", cause: Cause::UserInput, remediation: None },
    // E-PARAMGB — ParamGroebnerError (M9, Gröbner bases over Q(params))
    ErrorSpec { code: "E-PARAMGB-001", class: "ParamGroebnerError", cause: Cause::UserInput,   remediation: Some("pass at least one polynomial") },
    ErrorSpec { code: "E-PARAMGB-002", class: "ParamGroebnerError", cause: Cause::UserInput,   remediation: Some("build every generator against the same variable and parameter lists") },
    ErrorSpec { code: "E-PARAMGB-003", class: "ParamGroebnerError", cause: Cause::UserInput,   remediation: Some("supply exactly one value per parameter, in the parameter list's order") },
    // Not a malfunction: the basis is a *generic* one, and this is it saying so
    // at a point where it does not apply, rather than specialising anyway.
    ErrorSpec { code: "E-PARAMGB-004", class: "ParamGroebnerError", cause: Cause::Domain,      remediation: Some("compute the basis directly over ℚ at that parameter point, or move the vanishing factors into the generators and recompute") },
    // E-VALIDATED — ValidatedError
    ErrorSpec { code: "E-VALIDATED-001", class: "ValidatedError", cause: Cause::Unsupported, remediation: None },
    ErrorSpec { code: "E-VALIDATED-002", class: "ValidatedError", cause: Cause::UserInput, remediation: None },
    ErrorSpec { code: "E-VALIDATED-003", class: "ValidatedError", cause: Cause::Domain, remediation: None },
    ErrorSpec { code: "E-VALIDATED-004", class: "ValidatedError", cause: Cause::Domain, remediation: None },
    ErrorSpec { code: "E-VALIDATED-005", class: "ValidatedError", cause: Cause::UserInput, remediation: None },
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
