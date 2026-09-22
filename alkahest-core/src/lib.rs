// clippy 1.97 promoted `clippy::question_mark` to fire on `else if let … else {
// return None }` chains used idiomatically across the crate; under CI's
// `-D warnings` this newly fails the build.  Allow it crate-wide (toolchain
// adaptation; idiomatic `?` rewrites can follow as a separate cleanup).
#![allow(clippy::question_mark)]

pub mod acausal;
pub mod algebra;
pub mod ball;
// P1 search plumbing item 4 — budgets, cancellation, determinism
pub mod budget;
pub mod calculus;
pub mod coding;
pub mod dae;
pub mod deriv;
pub mod diff;
pub mod errors;
pub mod eval;
pub mod ffield;
pub mod flint;
pub mod funcfield;
pub mod group;
pub mod horner;
// P1 item 7 — creative telescoping / holonomic (D-finite) machinery
pub mod holonomic;
pub mod hybrid;
pub mod integrate;
pub mod jit;
pub mod kernel;
// V3-3 — First-order logic / FOFormula
pub mod logic;
// V2-6 — LLL + PSLQ (PSLQ in `numeric`)
pub mod lattice;
pub mod lean;
pub mod matrix;
// V2-1 — Modular / CRT framework
pub mod modular;
// V3-1 — Integer number theory (`fmpz` helpers)
pub mod number_theory;
pub mod numeric;
pub mod numfield;
pub mod ode;
pub mod parse;
pub mod pattern;
pub mod poly;
pub mod prob;
// V2-9 — CAD / real QE
#[cfg(feature = "groebner")]
pub mod ideal;
pub mod primitive;
pub mod real;
pub mod simplify;
#[cfg(feature = "groebner")]
pub mod solver;
pub mod special;
// V2-13 — Differential algebra / Rosenfeld–Gröbner
#[cfg(feature = "groebner")]
pub mod diffalg;
// Binary symplectic / stabilizer codes, and classical matrix groups over GF(q)
pub mod stabilizer;
// V2-10 — Gosper / creative telescoping (WZ certificates)
pub mod stablehlo;
pub mod sum;
pub mod theta;
// §3.3 — symbolic integral transforms (Laplace and inverse Laplace)
pub mod transform;
// P1 item 9 — rigorous global bounds (Taylor models / validated numerics)
pub mod validated;
// Vector calculus over orthogonal curvilinear coordinates (grad/div/curl/laplacian)
pub mod vector;
// Plot — dependency-free SVG / DOT renderers
pub mod plot;

pub use acausal::{capacitor, resistor, voltage_source, Component, Port, System};
pub use calculus::{limit, series, LimitDirection, LimitError, Series, SeriesError};
pub use dae::{pantelides, DaeError, PantelidesResult, DAE};
pub use deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
#[allow(deprecated)]
pub use diff::{diff, diff_forward, grad, DiffError, DualValue, ForwardDiffError};
pub use eval::{
    eval_complex_f64, eval_exact_rational, eval_f64, eval_interval, evaluate, ComplexF64,
    EvalError, EvalMode, EvalValue, UnsupportedReason,
};
pub use flint::{FlintInteger, FlintPoly};
pub use hybrid::{Event, GuardStructure, HybridODE};
pub use integrate::{
    integrate, integrate_definite, verify_antiderivative_exact, verify_antiderivative_status,
    verify_antiderivative_status_parametric, AntiderivativeVerification, IntegrationError,
};
#[allow(deprecated)]
pub use kernel::{
    check_expr_depth, check_expr_depths, expr_contains_noncommutative_symbol, load_from,
    mult_tree_is_commutative, open_persistent, render_latex, render_unicode, save_to, subs,
    DepthLimitError, Domain, ExprData, ExprDisplay, ExprId, ExprPool, IoError, PoolPersistError,
    MAX_EXPR_DEPTH,
};
pub use logic::{
    dpll_sat, formula_from_expr, satisfiable, BoolClause, BoolLit, Formula, LogicError,
    Satisfiability,
};
pub use real::{
    cad_lift, cad_project, decide, decide_expr, routh_hurwitz, CadError, QeResult, RouthHurwitz,
};
pub use simplify::{simplify_with_assumptions, AssumptionContext, AssumptionError};
// V2-6 — LLL + integer relations (augmented lattice heuristic)
pub use lattice::{
    lattice_reduce_rows, lattice_reduce_rows_exact, lattice_reduce_rows_with_delta,
    validate_lll_rows, LatticeError,
};
pub use matrix::{
    characteristic_polynomial_lambda_minus_m, cholesky, column_space_basis, diagonalize,
    eigenvalues, eigenvectors, hermite_form, hermite_form_poly, jacobian, jordan_form,
    lu_decomposition, matrix_exponential, matrix_inverse, minimal_polynomial, nullspace_basis,
    qr_decomposition, rank, rational_canonical_form, row_space_basis, rref, smith_form,
    smith_form_poly, EigenError, IntegerMatrix, LinearAlgebraError, LuDecomposition, Matrix,
    MatrixError, NormalFormError, PolyMatrixQ, QrDecomposition, RatUniPoly,
};
pub use numeric::{guess_integer_relation, PslqError};
pub use ode::{
    dsolve::{
        dsolve, dsolve_with,
        system::{dsolve_system, dsolve_system_with, DsolveSystemError, SystemSolution},
        DsolveBranch, DsolveError, DsolveReport, DsolveResult, DsolveSolution, OdeInput,
        SolutionForm,
    },
    lower_to_first_order,
    sensitivity::{adjoint_system, sensitivity_system, AdjointSystem, SensitivitySystem},
    OdeError, ScalarODE, ODE,
};
pub use parse::{parse, ParseError};
pub use pattern::{match_pattern, Pattern, Substitution};
pub use poly::{
    apart, cancel, collect_free_vars, factor_multivariate_z, factor_univariate_mod_p,
    factor_univariate_z, gcd_sparse_modular, poly_normal, real_roots, real_roots_symbolic,
    refine_root, residue, resultant, sparse_interpolate, sparse_interpolate_univariate,
    subresultant_prs, together, together_parts, ApartError, ConversionError, FactorError, GaussRat,
    MultiPoly, MultiPolyFactorization, RationalFunction, RealRootError, ResidueError,
    ResultantError, RootInterval, SparseGcdError, SparseInterpError, UniPoly, UniPolyFactorModP,
    UniPolyFactorization,
};

// §3.3 — Laplace transform and inverse (experimental surface; see `experimental`)
pub use transform::{inverse_laplace_transform, laplace_transform, LaplaceError};

// §3.4 — Fourier transform and inverse (experimental surface; see `experimental`)
pub use transform::{fourier_transform, inverse_fourier_transform, FourierError};

// §3.5 — Z-transform and inverse (experimental surface; see `experimental`)
pub use transform::{
    inverse_z_transform, z_shift_advance, z_shift_delay, z_transform, ZTransformError,
};

// Phase 24 — Horner form
pub use horner::{
    emit_expr_c, emit_expr_c_vec, emit_horner_c, eval_horner_f64, eval_horner_f64_batch, horner,
    EmitCError,
};
pub use simplify::rulesets::{
    log_exp_rules, log_exp_rules_safe, trig_normal_form_rules, trig_rules,
};
pub use simplify::{
    assumptions_satisfy, check_simplify_depth, rules_for_config, simplify, simplify_batch,
    simplify_colored, simplify_egraph, simplify_egraph_with, simplify_expanded, simplify_log_exp,
    simplify_trig_normal_form, simplify_with, ColorId, ColoredEgraph, DepthCost, EgraphConfig,
    EgraphCost, NoncommutativeCost, OpCost, PatternRule, RewriteRule, SimplifyConfig, SizeCost,
    StabilityCost, CONTEXT_COLOR, ROOT_COLOR,
};
pub use sum::{
    gosper_certificate, gosper_normal_form, hypergeom_ratio, product_definite, product_indefinite,
    rsolve, solve_linear_recurrence_homogeneous, sum_definite, sum_indefinite, verify_wz_pair,
    LinearRecurrenceError, ProductError, RatFunc, RecurrenceSolution, RsolveError, SumError,
    WzPair,
};

// Phase 21 — JIT
pub use jit::{
    compile, compile_jit_only, compile_with, eval_interp, eval_interp_checked, expr_subgraph_size,
    jit_available, select_compile_tier, CompileCache, CompileConfig, CompileTier, CompiledFn,
    InterpEvalError, JitError, INTERP_MAX_EXPECTED_EVALS, INTERP_MAX_NODES,
    LLVM_MIN_EXPECTED_EVALS,
};

// Plot — SVG polyline and Graphviz DOT renderers (dependency-free)
pub use plot::{render_dot, render_svg, render_svg_opts};

// V5-2 — StableHLO/XLA bridge
pub use stablehlo::emit_stablehlo;

// V5-3 — NVPTX JIT backend
#[cfg(feature = "cuda")]
pub use jit::{compile_cuda, cuda_device_count, CudaCompiledFn, CudaError};

// Phase 22 — Ball arithmetic
pub use ball::{AcbBall, ArbBall, IntervalEval, DEFAULT_PREC};

// Phase 23 — Parallel simplification
#[cfg(feature = "parallel")]
pub use simplify::dispatch::{choose_strategy, simplify_auto, simplify_auto_with_config, Strategy};
#[cfg(feature = "parallel")]
pub use simplify::parallel::{simplify_par, simplify_par_with_config};
#[cfg(feature = "parallel")]
pub use simplify::redex::{simplify_redex, simplify_redex_with_config};

// V5-11 — Gröbner basis
#[cfg(feature = "groebner")]
pub use poly::groebner::{
    compute_groebner_basis_f5, fglm, grevlex_staircase, is_zero_dimensional, GbPoly, GroebnerBasis,
    MonomialOrder,
};
// M9 — coefficient fields for elimination
#[cfg(feature = "groebner")]
pub use poly::groebner::{ParamGbPoly, ParamGroebnerBasis, ParamGroebnerError, ParamPoly, QParam};

// P1 search plumbing item 4 — budgets, cancellation, determinism
pub use budget::{
    check as budget_check, check_growth as budget_check_growth, clear_cancel,
    enter as budget_enter, is_active as budget_is_active, is_cancelled, request_cancel,
    seed as budget_seed, Budget, BudgetError, BudgetGuard, DEFAULT_MAX_GROWTH_UNITS,
};
pub use errors::AlkahestError;
pub use lean::{
    emit_definite_integration_cert, emit_gosper_cert, emit_integration_cert,
    emit_lean_expr as emit_lean, emit_lean_expr_wrt, emit_product_cert, emit_tendsto_cert,
    step_is_certifiable,
};
// V2-1 — Modular / CRT framework
#[cfg(feature = "groebner")]
pub use diffalg::{
    dae_index_reduce, dae_index_reduce_ranked, rosenfeld_groebner, rosenfeld_groebner_algebraic,
    rosenfeld_groebner_parametric, rosenfeld_groebner_ranked, rosenfeld_groebner_with_options,
    DaeIndexReduction, DiffAlgError, DifferentialIdeal, DifferentialRanking, DifferentialRing,
    ParametricProlongOpts, ParametricRosenfeldResult, RegularDifferentialChain,
    RosenfeldGroebnerResult,
};
#[cfg(feature = "groebner")]
pub use ideal::{
    primary_decomposition, radical, take_ideal_refusal, IdealRefusal, PrimaryComponent,
    PrimaryDecompositionError,
};
pub use modular::{
    is_prime, lift_crt, mignotte_bound, rational_reconstruction, reduce_mod, select_lucky_prime,
    ModularError, ModularValue, MultiPolyFp,
};
pub use number_theory::{
    discrete_log, factorint, isprime, jacobi_symbol, nextprime, nthroot_mod, totient,
    NumberTheoryError, QuadraticDirichlet,
};
pub use primitive::{
    taylor_model_blockers, taylor_model_refusal, taylor_model_supports, taylor_model_supports_call,
    Capabilities, CoverageReport, CoverageRow, Primitive, PrimitiveRegistry,
};
#[cfg(feature = "groebner")]
pub use solver::{
    diophantine, expr_to_gbpoly, expr_to_param_gbpoly, extract_regular_chain_from_basis,
    gbpoly_to_expr, main_variable_recursive, solve_numerical, solve_polynomial_system,
    solve_transcendental, triangularize, CertifiedPoint, DiophantineError, DiophantineSolution,
    HomotopyError, HomotopyOpts, RegularChain, Solution, SolutionSet, SolverError,
    TranscendentalOutcome,
};

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Stable 1.0 API surface (V1-8).
///
/// Items in this module are covered by the Alkahest semver promise: any
/// backwards-incompatible change to a symbol re-exported here requires a
/// major-version bump (`2.0`).  Everything else in `alkahest_core::*` —
/// including the top-level re-exports kept for backwards compatibility —
/// is considered experimental unless it also appears below.
///
/// See `README.md` ("Stability") for the full policy.
pub mod stable {
    pub use crate::algebra::{
        clifford_orthogonal_rules, imag_unit_atom, pauli_product_rules, PauliSpinAlgebraRule,
    };
    pub use crate::calculus::{limit, series, LimitDirection, LimitError, Series, SeriesError};
    pub use crate::dae::{pantelides, DaeError, DAE};
    pub use crate::diff::{diff, diff_forward, grad, DiffError};
    #[cfg(feature = "groebner")]
    pub use crate::diffalg::{
        dae_index_reduce, dae_index_reduce_ranked, rosenfeld_groebner,
        rosenfeld_groebner_algebraic, rosenfeld_groebner_ranked, rosenfeld_groebner_with_options,
        DaeIndexReduction, DiffAlgError, DifferentialIdeal, DifferentialRanking, DifferentialRing,
        RegularDifferentialChain, RosenfeldGroebnerResult,
    };
    pub use crate::errors::AlkahestError;
    pub use crate::eval::{
        eval_complex_f64, eval_exact_rational, eval_f64, eval_interval, evaluate, ComplexF64,
        EvalError, EvalMode, EvalValue, UnsupportedReason,
    };
    #[cfg(feature = "groebner")]
    pub use crate::ideal::{
        primary_decomposition, radical, take_ideal_refusal, IdealRefusal, PrimaryComponent,
        PrimaryDecompositionError,
    };
    pub use crate::integrate::{integrate, integrate_definite, IntegrationError};
    pub use crate::jit::{compile, CompileCache, CompiledFn, JitError};
    #[cfg(feature = "cuda")]
    pub use crate::jit::{compile_cuda, cuda_device_count, CudaCompiledFn, CudaError};
    #[allow(deprecated)]
    pub use crate::kernel::pool_persist::PoolPersistError;
    pub use crate::kernel::pool_persist::{load_from, open_persistent, save_to, IoError};
    pub use crate::kernel::{
        check_expr_depth, check_expr_depths, expr_contains_noncommutative_symbol,
        mult_tree_is_commutative, render_latex, render_unicode, subs, DepthLimitError, Domain,
        ExprData, ExprDisplay, ExprId, ExprPool, MAX_EXPR_DEPTH,
    };
    pub use crate::lattice::{
        lattice_reduce_rows, lattice_reduce_rows_with_delta, validate_lll_rows, LatticeError,
    };
    pub use crate::lean::emit_lean_expr as emit_lean;
    pub use crate::logic::{
        dpll_sat, formula_from_expr, satisfiable, BoolClause, BoolLit, Formula, LogicError,
        Satisfiability,
    };
    pub use crate::matrix::{
        characteristic_polynomial_lambda_minus_m, diagonalize, eigenvalues, eigenvectors,
        hermite_form, hermite_form_poly, jacobian, smith_form, smith_form_poly, EigenError,
        IntegerMatrix, Matrix, MatrixError, NormalFormError, PolyMatrixQ, RatUniPoly,
    };
    pub use crate::number_theory::{
        discrete_log, factorint, isprime, jacobi_symbol, nextprime, nthroot_mod, totient,
        NumberTheoryError, QuadraticDirichlet,
    };
    pub use crate::numeric::{guess_integer_relation, PslqError};
    pub use crate::ode::{lower_to_first_order, OdeError, ScalarODE, ODE};
    pub use crate::parse::{parse, ParseError};
    pub use crate::pattern::{match_pattern, Pattern, Substitution};
    pub use crate::poly::{
        apart, cancel, collect_free_vars, factor_multivariate_z, factor_univariate_mod_p,
        factor_univariate_z, gcd_sparse_modular, poly_normal, real_roots, real_roots_symbolic,
        refine_root, residue, resultant, sparse_interpolate, sparse_interpolate_univariate,
        subresultant_prs, together, together_parts, ApartError, ConversionError, FactorError,
        GaussRat, MultiPoly, MultiPolyFactorization, RationalFunction, RealRootError, ResidueError,
        ResultantError, RootInterval, SparseGcdError, SparseInterpError, UniPoly,
        UniPolyFactorModP, UniPolyFactorization,
    };
    pub use crate::primitive::{
        taylor_model_blockers, taylor_model_refusal, taylor_model_supports,
        taylor_model_supports_call, Primitive, PrimitiveRegistry,
    };
    pub use crate::real::{
        cad_lift, cad_project, decide, decide_expr, routh_hurwitz, CadError, QeResult, RouthHurwitz,
    };
    pub use crate::simplify::{
        check_simplify_depth, simplify, simplify_egraph, simplify_egraph_with,
        simplify_trig_normal_form, simplify_with, simplify_with_assumptions, AssumptionContext,
        DepthCost, EgraphConfig, EgraphCost, NoncommutativeCost, OpCost, SimplifyConfig, SizeCost,
        StabilityCost,
    };
    #[cfg(feature = "groebner")]
    pub use crate::solver::{
        diophantine, expr_to_gbpoly, extract_regular_chain_from_basis, gbpoly_to_expr,
        main_variable_recursive, solve_numerical, solve_polynomial_system, triangularize,
        CertifiedPoint, DiophantineError, DiophantineSolution, HomotopyError, HomotopyOpts,
        RegularChain, Solution, SolutionSet, SolverError,
    };
    pub use crate::stablehlo::emit_stablehlo;
    pub use crate::sum::{
        gosper_certificate, gosper_normal_form, hypergeom_ratio, product_definite,
        product_indefinite, rsolve, solve_linear_recurrence_homogeneous, sum_definite,
        sum_indefinite, verify_wz_pair, LinearRecurrenceError, ProductError, RatFunc,
        RecurrenceSolution, RsolveError, SumError, WzPair,
    };
    pub use crate::version;
}

/// Experimental surface — may change without a major-version bump (V1-8).
///
/// Anything here is subject to redesign.  Pin a point-release if you rely
/// on it.
pub mod experimental {
    pub use crate::acausal::{capacitor, resistor, voltage_source, Component, Port, System};
    /// Hamilton quaternions and the active rotation operator `q v q⁻¹`, with
    /// conversions to and from a rotation matrix and axis–angle form.
    pub use crate::algebra::quaternion::{Quaternion, QuaternionError};
    pub use crate::ball::{AcbBall, ArbBall, IntervalEval};
    pub use crate::calculus::asymptotic::{
        asymptotic_expand, AsymptoticError, AsymptoticExpansion, AsymptoticTerm,
    };
    pub use crate::calculus::fps::{Fps, FpsError};
    pub use crate::calculus::multilimit::{multilimit, MultiLimit, PathWitness};
    /// Puiseux (fractional-exponent) expansion — the sibling of `series` for
    /// the half-integer valuations `Series` has no representation for. Every
    /// returned expansion has been checked before it left; see the module docs.
    pub use crate::calculus::puiseux::{
        puiseux_series, Evidence, NotPuiseuxReason, PuiseuxError, PuiseuxExpansion,
        UnverifiedReason, MAX_RAMIFICATION,
    };
    /// Classical linear codes over GF(q): weight enumerators, MacWilliams,
    /// Krawtchouk polynomials, and the Delsarte linear-programming bound on
    /// `A_q(n, d)` — solved in exact rational arithmetic and returned with the
    /// dual certificate that proves it. See [`crate::coding`] for scope.
    pub use crate::coding::{
        binomial, binomial_generalised, delsarte_lp_bound, hamming_bound, krawtchouk,
        krawtchouk_pairing, krawtchouk_poly, singleton_bound, CodingError, DelsarteBound,
        LinearCode, WeightEnumerator, MAX_ENUMERATED_CODEWORDS, MAX_ENUMERATION_CELLS,
        MAX_LP_LENGTH,
    };
    pub use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
    pub use crate::eval::{
        eval_complex_f64, eval_exact_rational, eval_f64, eval_interval, evaluate, ComplexF64,
        EvalError, EvalMode, EvalValue, UnsupportedReason,
    };
    /// Dense linear algebra over the finite fields GF(q), q = p^k, backed by
    /// FLINT's `nmod_mat` (prime fields) and `fq_nmod_mat` (extensions).
    /// Built for linear codes: rectangular shapes are first-class and
    /// `nullspace` over GF(2) is the path everything else is arranged around.
    /// See [`crate::ffield`] for what is deliberately out of scope.
    pub use crate::ffield::{
        FieldElement, FiniteField, FiniteFieldError, GfMatrix, Rref, MAX_EXTENSION_DEGREE,
    };
    /// Function fields of algebraic curves: divisors, the divisor class group
    /// and Riemann–Roch, for the imaginary hyperelliptic model with rational
    /// places.  See `funcfield`'s module docs for exactly what is refused.
    pub use crate::funcfield::{
        riemann_roch, Divisor, DivisorClass, FunctionField, FunctionFieldElement,
        FunctionFieldError, Normalisation, Place, RiemannRochSpace,
    };
    /// Computational group theory: permutation groups, orbits with Schreier
    /// vectors, a base and strong generating set from Schreier–Sims, exact
    /// arbitrary-precision group order, and membership by sifting. Points are
    /// 0-based and composition is **left-to-right** (`p.compose(&q)` applies
    /// `p` first) — see [`crate::group`] for the conventions and for the
    /// explicit list of what is out of scope.
    pub use crate::group::{
        alternating, cyclic, dihedral, symmetric, trivial as trivial_group, GroupError, Orbit,
        Permutation, PermutationGroup, SchreierEntry, SiftResult, StabilizerChain, StabilizerLevel,
        DEFAULT_ELEMENT_CAP, MAX_BSGS_DEGREE, MAX_ELEMENT_CAP,
    };
    /// Continuous (differential) creative telescoping — Almkvist–Zeilberger,
    /// the twin of `q_zeilberger`/`telescope2d` on the `D_x` side. Rust-only
    /// for now: there is no PyO3 binding yet.
    pub use crate::holonomic::azeil::{
        almkvist_zeilberger, dgosper, dgosper_term, hyperexp_log_derivative,
        integral_boundary_status, AzOpts, AzResult, DiffTelescopingError, HyperExpTerm,
        IntegralBoundaryStatus, IntegrationLimit, PowerFactor,
    };
    pub use crate::holonomic::{
        boundary_status_2d, telescope2d, telescope2d_search, BoundaryStatus2d, Telescoping2dError,
        Telescoping2dOpts, Telescoping2dResult,
    };
    pub use crate::horner::{emit_expr_c, emit_expr_c_vec, emit_horner_c, horner, EmitCError};
    pub use crate::hybrid::{Event, GuardStructure, HybridODE};
    /// Lattices over ℤ: the standard families (`zn`, `a_n`, `d_n`, `e8`,
    /// `leech`), determinants and duals, **exact** shortest and closest
    /// vectors, theta series, kissing numbers and sphere-packing densities.
    ///
    /// `Lattice::from_gram` is the general constructor — `E_8` and the Leech
    /// lattice have no rational basis in their natural embedding, while every
    /// invariant here depends on the Gram matrix alone. SVP, CVP, minimal
    /// vectors and theta series are exact Fincke–Pohst enumeration: no
    /// heuristic, exponential in the rank, capped at [`MAX_ENUM_RANK`] with a
    /// node budget, and a typed refusal above either. See [`crate::lattice`]
    /// for the full list of scope limits.
    pub use crate::lattice::{
        a_n, d_n, e8, lattice_reduce_rows_exact, leech, zn, Lattice, LatticeVector,
        DEFAULT_ENUM_NODE_BUDGET, MAX_ENUM_RANK, MAX_THETA_NORM,
    };
    pub use crate::lean::emit_lean_expr as emit_lean;
    pub use crate::matrix::{
        cholesky, column_space_basis, jordan_form, lu_decomposition, matrix_exponential,
        matrix_inverse, minimal_polynomial, nullspace_basis, qr_decomposition, rank,
        rational_canonical_form, row_space_basis, rref, LinearAlgebraError, LuDecomposition,
        QrDecomposition,
    };
    pub use crate::modular::{
        is_prime, lift_crt, mignotte_bound, rational_reconstruction, reduce_mod,
        select_lucky_prime, ModularError, ModularValue, MultiPolyFp,
    };
    /// The classical arithmetic functions, on FLINT's `arith`, `bernoulli` and
    /// `partitions`. **FLINT's Bernoulli convention is `B_1 = -1/2`** (DLMF /
    /// Mathematica / SymPy); the other convention differs in exactly that one
    /// value and nowhere else, which is what makes picking the wrong one so
    /// quiet a bug. See [`crate::number_theory`].
    pub use crate::number_theory::{
        bernoulli_number, divisor_sigma, euler_number, harmonic_number, moebius_mu,
        partition_number, stirling_first, stirling_first_unsigned, stirling_second, sum_of_squares,
        MAX_BERNOULLI_N, MAX_EULER_N, MAX_HARMONIC_N, MAX_PARTITION_N, MAX_STIRLING_N,
    };
    pub use crate::numeric::{guess_integer_relation, PslqError};
    /// Algebraic number fields Q[x]/(f) on FLINT's `nf`/`nf_elem`, including
    /// the cyclotomic fields Q(zeta_n). The defining polynomial is **checked**
    /// for irreducibility, and `polynomial_discriminant` is the discriminant
    /// of that polynomial — *not* the field discriminant, which this module
    /// does not compute. See [`crate::numfield`].
    pub use crate::numfield::{
        cyclotomic_polynomial, NumberField, NumberFieldElement, NumberFieldError, MAX_FIELD_DEGREE,
    };
    pub use crate::ode::dsolve::system::{
        dsolve_system, dsolve_system_with, DsolveSystemError, SystemSolution,
    };
    pub use crate::ode::dsolve::{
        dsolve, dsolve_with, DsolveBranch, DsolveError, DsolveReport, DsolveResult, DsolveSolution,
        OdeInput, SolutionForm,
    };
    pub use crate::ode::sensitivity::{
        adjoint_system, sensitivity_system, AdjointSystem, SensitivitySystem,
    };
    pub use crate::ode::series_solve::{
        series_solve, PointKind, SeriesError as SeriesSolveError, SeriesOde, SeriesResult,
        SeriesSolution,
    };
    pub use crate::plot::{render_dot, render_svg, render_svg_opts};
    /// M9 — Gröbner bases over the coefficient field `Q(params)`, with the
    /// specialisation hypotheses reported rather than assumed.
    #[cfg(feature = "groebner")]
    pub use crate::poly::groebner::{
        ParamGbPoly, ParamGroebnerBasis, ParamGroebnerError, ParamPoly, QParam,
    };
    pub use crate::poly::{
        gcd_sparse_modular, residue, sparse_interpolate, sparse_interpolate_univariate, GaussRat,
        ResidueError, SparseGcdError, SparseInterpError,
    };
    /// Probability distributions as symbolic objects, expectations over them,
    /// and the numeric gate every closed form they produce has to pass. See
    /// [`crate::prob`] — in particular, what refuses and why.
    pub use crate::prob::{
        characteristic_function, expectation, expectation_affine, take_prob_side_conditions,
        variance_affine_independent, DistKind, Distribution, Evidence as ProbEvidence, ProbError,
        Support, UnverifiedReason as ProbUnverifiedReason, MAX_MOMENT_ORDER,
    };
    pub use crate::simplify::{
        simplify_colored, simplify_egraph, simplify_expanded, ColorId, ColoredEgraph,
        CONTEXT_COLOR, ROOT_COLOR,
    };
    /// The binary symplectic / stabilizer layer: the `(x | z)` symplectic form
    /// over GF(2), Pauli operators with a `Z₄` phase, stabilizer and CSS codes
    /// with their logical operators and syndrome map, and the classical matrix
    /// groups `GL`, `SL`, `Sp` over GF(q). The `(x | z)` convention and the
    /// scope limits — no Clifford simulation, no decoding, distance only by
    /// capped exhaustive search — are in [`crate::stabilizer`].
    pub use crate::stabilizer::{
        is_symplectic, symplectic_complement, symplectic_form, symplectic_gram_matrix,
        symplectic_gram_schmidt, CssCode, Distance, HyperbolicBasis, MatrixGroup, MatrixGroupKind,
        PauliOperator, StabilizerCode, StabilizerError, StabilizerGroup, MAX_DISTANCE_SEARCH_DIM,
        MAX_MATRIX_ENUMERATION, MAX_QUBITS,
    };
    pub use crate::stablehlo::emit_stablehlo;
    /// Riemann theta functions, classical modular functions (`eta`, `j`,
    /// `lambda`, `Delta`, Eisenstein) and the Weierstrass family, as
    /// **rigorous enclosures** backed by FLINT's Arb layer. Every value is a
    /// [`crate::theta::ComplexBall`] carrying its own error bound, and
    /// [`crate::theta::Precision::AccurateTo`] refuses rather than returning a
    /// midpoint with nothing behind it. Genus 1 and 2 are what this is built
    /// and tested for; see [`crate::theta`] for the genus ceiling and the rest
    /// of the scope limits.
    pub use crate::theta::{
        arb_backend_available, dedekind_eta, eisenstein_series, j_invariant, jacobi_theta,
        jacobi_theta_null, modular_discriminant, modular_lambda, riemann_theta,
        riemann_theta_available, riemann_theta_characteristic, riemann_theta_squared,
        siegel_is_reduced, siegel_reduce, theta_characteristic_bits, theta_characteristic_index,
        theta_characteristic_is_even, weierstrass_invariants, weierstrass_p, weierstrass_p_prime,
        weierstrass_roots, weierstrass_sigma, weierstrass_zeta, ComplexBall, Precision, RealBall,
        SiegelMatrix, SiegelReduction, ThetaError, ThetaValues, DEFAULT_PRECISION_BITS, MAX_GENUS,
        MAX_PRECISION_BITS, MIN_PRECISION_BITS,
    };
    pub use crate::transform::fourier::fourier_derivative_rule;
    pub use crate::transform::laplace::laplace_derivative_rule;
    pub use crate::transform::{
        fourier_transform, fourier_transform_with_conditions, inverse_fourier_transform,
        inverse_fourier_transform_with_conditions, inverse_laplace_transform,
        inverse_laplace_transform_with_assumptions, inverse_laplace_transform_with_conditions,
        laplace_transform, laplace_transform_with_conditions, FourierError, LaplaceError,
    };
    pub use crate::transform::{
        inverse_z_transform, inverse_z_transform_with_assumptions,
        inverse_z_transform_with_conditions, take_transform_side_conditions, z_shift_advance,
        z_shift_delay, z_transform, ZTransformError,
    };
    /// Vector calculus over orthogonal curvilinear coordinates: `grad`, `div`,
    /// `curl`, the scalar and vector Laplacian, and the vector algebra they
    /// are used with. Cartesian, cylindrical and spherical charts are built
    /// in; `Coordinates::from_embedding` derives one from its Cartesian
    /// parametrisation and checks orthogonality before returning it.
    pub use crate::vector::{
        cross, curl, divergence, dot, gradient, laplacian, norm, norm_squared, vector_laplacian,
        Coordinates, VectorError,
    };
    // ℚ(params) partial fractions: the in-band and out-of-band forms of the
    // hypotheses a parametric decomposition rests on.
    pub use crate::poly::{apart_with_conditions, take_apart_side_conditions};

    #[cfg(feature = "parallel")]
    pub use crate::simplify::dispatch::{
        choose_strategy, simplify_auto, simplify_auto_with_config, Strategy,
    };
    #[cfg(feature = "parallel")]
    pub use crate::simplify::parallel::{simplify_par, simplify_par_with_config};
    #[cfg(feature = "parallel")]
    pub use crate::simplify::redex::{simplify_redex, simplify_redex_with_config};

    #[cfg(feature = "groebner")]
    pub use crate::poly::groebner::{
        compute_groebner_basis_f5, fglm, grevlex_staircase, is_zero_dimensional, GbPoly,
        GroebnerBasis, MonomialOrder,
    };
    #[cfg(feature = "groebner-cuda")]
    pub use crate::poly::groebner::{GpuBackendReport, GpuGroebnerError};

    /// Bounded content-addressed expression pool (RFC 0001). Not used by default
    /// [`crate::ExprPool`]; unit-tested prototype only.
    pub mod merkle_pool;
}
