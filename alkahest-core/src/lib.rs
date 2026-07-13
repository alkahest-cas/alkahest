// clippy 1.97 promoted `clippy::question_mark` to fire on `else if let … else {
// return None }` chains used idiomatically across the crate; under CI's
// `-D warnings` this newly fails the build.  Allow it crate-wide (toolchain
// adaptation; idiomatic `?` rewrites can follow as a separate cleanup).
#![allow(clippy::question_mark)]

pub mod acausal;
pub mod algebra;
pub mod ball;
pub mod calculus;
pub mod dae;
pub mod deriv;
pub mod diff;
pub mod errors;
pub mod flint;
pub mod horner;
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
pub mod ode;
pub mod parse;
pub mod pattern;
pub mod poly;
// V2-9 — CAD / real QE
#[cfg(feature = "groebner")]
pub mod ideal;
pub mod primitive;
pub mod real;
pub mod simplify;
#[cfg(feature = "groebner")]
pub mod solver;
// V2-13 — Differential algebra / Rosenfeld–Gröbner
#[cfg(feature = "groebner")]
pub mod diffalg;
// V2-10 — Gosper / creative telescoping (WZ certificates)
pub mod stablehlo;
pub mod sum;
// §3.3 — symbolic integral transforms (Laplace and inverse Laplace)
pub mod transform;
// Plot — dependency-free SVG / DOT renderers
pub mod plot;

pub use acausal::{capacitor, resistor, voltage_source, Component, Port, System};
pub use calculus::{limit, series, LimitDirection, LimitError, Series, SeriesError};
pub use dae::{pantelides, DaeError, PantelidesResult, DAE};
pub use deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
#[allow(deprecated)]
pub use diff::{diff, diff_forward, grad, DiffError, DualValue, ForwardDiffError};
pub use flint::{FlintInteger, FlintPoly};
pub use hybrid::{Event, GuardStructure, HybridODE};
pub use integrate::{
    integrate, integrate_definite, verify_antiderivative_exact, verify_antiderivative_status,
    AntiderivativeVerification, IntegrationError,
};
#[allow(deprecated)]
pub use kernel::{
    expr_contains_noncommutative_symbol, load_from, mult_tree_is_commutative, open_persistent,
    render_latex, render_unicode, save_to, subs, Domain, ExprData, ExprDisplay, ExprId, ExprPool,
    IoError, PoolPersistError,
};
pub use logic::{
    dpll_sat, formula_from_expr, satisfiable, BoolClause, BoolLit, Formula, LogicError,
    Satisfiability,
};
pub use real::{
    cad_lift, cad_project, decide, decide_expr, routh_hurwitz, CadError, QeResult, RouthHurwitz,
};
// V2-6 — LLL + integer relations (augmented lattice heuristic)
pub use lattice::{
    lattice_reduce_rows, lattice_reduce_rows_with_delta, validate_lll_rows, LatticeError,
};
pub use matrix::{
    characteristic_polynomial_lambda_minus_m, cholesky, column_space_basis, diagonalize,
    eigenvalues, eigenvectors, hermite_form, hermite_form_poly, jacobian, jordan_form,
    lu_decomposition, matrix_exponential, matrix_inverse, minimal_polynomial, nullspace_basis,
    qr_decomposition, rank, rational_canonical_form, row_space_basis, smith_form, smith_form_poly,
    EigenError, IntegerMatrix, LinearAlgebraError, LuDecomposition, Matrix, MatrixError,
    NormalFormError, PolyMatrixQ, QrDecomposition, RatUniPoly,
};
pub use numeric::{guess_integer_relation, PslqError};
pub use ode::{
    dsolve::{dsolve, DsolveError, DsolveResult, DsolveSolution, OdeInput},
    lower_to_first_order,
    sensitivity::{adjoint_system, sensitivity_system, AdjointSystem, SensitivitySystem},
    OdeError, ScalarODE, ODE,
};
pub use parse::{parse, ParseError};
pub use pattern::{match_pattern, Pattern, Substitution};
pub use poly::{
    apart, cancel, factor_multivariate_z, factor_univariate_mod_p, factor_univariate_z,
    gcd_sparse_modular, poly_normal, real_roots, real_roots_symbolic, refine_root, resultant,
    sparse_interpolate, sparse_interpolate_univariate, subresultant_prs, together, together_parts,
    ApartError, ConversionError, FactorError, MultiPoly, MultiPolyFactorization, RationalFunction,
    RealRootError, ResultantError, RootInterval, SparseGcdError, SparseInterpError, UniPoly,
    UniPolyFactorModP, UniPolyFactorization,
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
    assumptions_satisfy, rules_for_config, simplify, simplify_batch, simplify_colored,
    simplify_egraph, simplify_egraph_with, simplify_expanded, simplify_trig_normal_form,
    simplify_with, ColorId, ColoredEgraph, DepthCost, EgraphConfig, EgraphCost, NoncommutativeCost,
    OpCost, PatternRule, RewriteRule, SimplifyConfig, SizeCost, StabilityCost, CONTEXT_COLOR,
    ROOT_COLOR,
};
pub use sum::{
    gosper_certificate, gosper_normal_form, hypergeom_ratio, product_definite, product_indefinite,
    rsolve, solve_linear_recurrence_homogeneous, sum_definite, sum_indefinite, verify_wz_pair,
    LinearRecurrenceError, ProductError, RatFunc, RecurrenceSolution, RsolveError, SumError,
    WzPair,
};

// Phase 21 — JIT
pub use jit::{
    compile, compile_jit_only, compile_with, eval_interp, expr_subgraph_size, jit_available,
    select_compile_tier, CompileCache, CompileConfig, CompileTier, CompiledFn, JitError,
    INTERP_MAX_EXPECTED_EVALS, INTERP_MAX_NODES, LLVM_MIN_EXPECTED_EVALS,
};

// Plot — SVG polyline and Graphviz DOT renderers (dependency-free)
pub use plot::{render_dot, render_svg, render_svg_opts};

// V5-2 — StableHLO/XLA bridge
pub use stablehlo::emit_stablehlo;

// V5-3 — NVPTX JIT backend
#[cfg(feature = "cuda")]
pub use jit::{compile_cuda, CudaCompiledFn, CudaError};

// Phase 22 — Ball arithmetic
pub use ball::{AcbBall, ArbBall, IntervalEval, DEFAULT_PREC};

// Phase 23 — Parallel simplification
#[cfg(feature = "parallel")]
pub use simplify::parallel::{simplify_par, simplify_par_with_config};

// V5-11 — Gröbner basis
#[cfg(feature = "groebner")]
pub use poly::groebner::{
    compute_groebner_basis_f5, fglm, grevlex_staircase, is_zero_dimensional, GbPoly, GroebnerBasis,
    MonomialOrder,
};

pub use errors::AlkahestError;
pub use lean::{emit_lean_expr as emit_lean, emit_lean_expr_wrt};
// V2-1 — Modular / CRT framework
#[cfg(feature = "groebner")]
pub use diffalg::{
    dae_index_reduce, rosenfeld_groebner, rosenfeld_groebner_algebraic,
    rosenfeld_groebner_with_options, DaeIndexReduction, DiffAlgError, DifferentialIdeal,
    DifferentialRanking, DifferentialRing, RegularDifferentialChain, RosenfeldGroebnerResult,
};
#[cfg(feature = "groebner")]
pub use ideal::{primary_decomposition, radical, PrimaryComponent, PrimaryDecompositionError};
pub use modular::{
    is_prime, lift_crt, mignotte_bound, rational_reconstruction, reduce_mod, select_lucky_prime,
    ModularError, ModularValue, MultiPolyFp,
};
pub use number_theory::{
    discrete_log, factorint, isprime, jacobi_symbol, nextprime, nthroot_mod, totient,
    NumberTheoryError, QuadraticDirichlet,
};
pub use primitive::{Capabilities, CoverageReport, CoverageRow, Primitive, PrimitiveRegistry};
#[cfg(feature = "groebner")]
pub use solver::{
    diophantine, expr_to_gbpoly, extract_regular_chain_from_basis, main_variable_recursive,
    solve_numerical, solve_polynomial_system, solve_transcendental, triangularize, CertifiedPoint,
    DiophantineError, DiophantineSolution, HomotopyError, HomotopyOpts, RegularChain, Solution,
    SolutionSet, SolverError, TranscendentalOutcome,
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
        dae_index_reduce, rosenfeld_groebner, rosenfeld_groebner_algebraic,
        rosenfeld_groebner_with_options, DaeIndexReduction, DiffAlgError, DifferentialIdeal,
        DifferentialRanking, DifferentialRing, RegularDifferentialChain, RosenfeldGroebnerResult,
    };
    pub use crate::errors::AlkahestError;
    #[cfg(feature = "groebner")]
    pub use crate::ideal::{
        primary_decomposition, radical, PrimaryComponent, PrimaryDecompositionError,
    };
    pub use crate::integrate::{integrate, integrate_definite, IntegrationError};
    pub use crate::jit::{compile, CompileCache, CompiledFn, JitError};
    #[cfg(feature = "cuda")]
    pub use crate::jit::{compile_cuda, CudaCompiledFn, CudaError};
    #[allow(deprecated)]
    pub use crate::kernel::pool_persist::PoolPersistError;
    pub use crate::kernel::pool_persist::{load_from, open_persistent, save_to, IoError};
    pub use crate::kernel::{
        expr_contains_noncommutative_symbol, mult_tree_is_commutative, render_latex,
        render_unicode, subs, Domain, ExprData, ExprDisplay, ExprId, ExprPool,
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
        apart, cancel, factor_multivariate_z, factor_univariate_mod_p, factor_univariate_z,
        gcd_sparse_modular, poly_normal, real_roots, real_roots_symbolic, refine_root, resultant,
        sparse_interpolate, sparse_interpolate_univariate, subresultant_prs, together,
        together_parts, ApartError, ConversionError, FactorError, MultiPoly,
        MultiPolyFactorization, RationalFunction, RealRootError, ResultantError, RootInterval,
        SparseGcdError, SparseInterpError, UniPoly, UniPolyFactorModP, UniPolyFactorization,
    };
    pub use crate::primitive::{Primitive, PrimitiveRegistry};
    pub use crate::real::{
        cad_lift, cad_project, decide, decide_expr, routh_hurwitz, CadError, QeResult, RouthHurwitz,
    };
    pub use crate::simplify::{
        simplify, simplify_egraph, simplify_egraph_with, simplify_trig_normal_form, simplify_with,
        DepthCost, EgraphConfig, EgraphCost, NoncommutativeCost, OpCost, SimplifyConfig, SizeCost,
        StabilityCost,
    };
    #[cfg(feature = "groebner")]
    pub use crate::solver::{
        diophantine, expr_to_gbpoly, extract_regular_chain_from_basis, main_variable_recursive,
        solve_numerical, solve_polynomial_system, triangularize, CertifiedPoint, DiophantineError,
        DiophantineSolution, HomotopyError, HomotopyOpts, RegularChain, Solution, SolutionSet,
        SolverError,
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
    pub use crate::ball::{AcbBall, ArbBall, IntervalEval};
    pub use crate::calculus::asymptotic::{
        asymptotic_expand, AsymptoticError, AsymptoticExpansion, AsymptoticTerm,
    };
    pub use crate::calculus::fps::{Fps, FpsError};
    pub use crate::calculus::multilimit::{multilimit, MultiLimit, PathWitness};
    pub use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
    pub use crate::horner::{emit_expr_c, emit_expr_c_vec, emit_horner_c, horner, EmitCError};
    pub use crate::hybrid::{Event, GuardStructure, HybridODE};
    pub use crate::lean::emit_lean_expr as emit_lean;
    pub use crate::matrix::{
        cholesky, column_space_basis, jordan_form, lu_decomposition, matrix_exponential,
        matrix_inverse, minimal_polynomial, nullspace_basis, qr_decomposition, rank,
        rational_canonical_form, row_space_basis, LinearAlgebraError, LuDecomposition,
        QrDecomposition,
    };
    pub use crate::modular::{
        is_prime, lift_crt, mignotte_bound, rational_reconstruction, reduce_mod,
        select_lucky_prime, ModularError, ModularValue, MultiPolyFp,
    };
    pub use crate::numeric::{guess_integer_relation, PslqError};
    pub use crate::ode::dsolve::{dsolve, DsolveError, DsolveResult, DsolveSolution, OdeInput};
    pub use crate::ode::sensitivity::{
        adjoint_system, sensitivity_system, AdjointSystem, SensitivitySystem,
    };
    pub use crate::ode::series_solve::{
        series_solve, PointKind, SeriesError as SeriesSolveError, SeriesOde, SeriesResult,
        SeriesSolution,
    };
    pub use crate::plot::{render_dot, render_svg, render_svg_opts};
    pub use crate::poly::{
        gcd_sparse_modular, sparse_interpolate, sparse_interpolate_univariate, SparseGcdError,
        SparseInterpError,
    };
    pub use crate::simplify::{
        simplify_colored, simplify_egraph, simplify_expanded, ColorId, ColoredEgraph,
        CONTEXT_COLOR, ROOT_COLOR,
    };
    pub use crate::stablehlo::emit_stablehlo;
    pub use crate::transform::fourier::fourier_derivative_rule;
    pub use crate::transform::laplace::laplace_derivative_rule;
    pub use crate::transform::{
        fourier_transform, inverse_fourier_transform, inverse_laplace_transform, laplace_transform,
        FourierError, LaplaceError,
    };
    pub use crate::transform::{
        inverse_z_transform, z_shift_advance, z_shift_delay, z_transform, ZTransformError,
    };

    #[cfg(feature = "parallel")]
    pub use crate::simplify::parallel::{simplify_par, simplify_par_with_config};

    #[cfg(feature = "groebner-cuda")]
    pub use crate::poly::groebner::GpuGroebnerError;
    #[cfg(feature = "groebner")]
    pub use crate::poly::groebner::{
        compute_groebner_basis_f5, fglm, grevlex_staircase, is_zero_dimensional, GbPoly,
        GroebnerBasis, MonomialOrder,
    };
}
