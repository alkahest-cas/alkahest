pub mod acausal;
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
pub mod numeric;
pub mod ode;
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

pub use acausal::{capacitor, resistor, voltage_source, Component, Port, System};
pub use dae::{pantelides, DaeError, PantelidesResult, DAE};
pub use deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
#[allow(deprecated)]
pub use diff::{diff, diff_forward, grad, DiffError, DualValue, ForwardDiffError};
pub use flint::{FlintInteger, FlintPoly};
pub use hybrid::{Event, GuardStructure, HybridODE};
pub use integrate::{integrate, IntegrationError};
pub use calculus::{series, Series, SeriesError};
#[allow(deprecated)]
pub use kernel::{
    load_from, open_persistent, save_to, subs, Domain, ExprData, ExprDisplay, ExprId, ExprPool,
    IoError, PoolPersistError,
};
pub use logic::{
    dpll_sat, formula_from_expr, satisfiable, BoolClause, BoolLit, Formula, LogicError,
    Satisfiability,
};
pub use real::{cad_lift, cad_project, decide, decide_expr, CadError, QeResult};
// V2-6 — LLL + integer relations (augmented lattice heuristic)
pub use lattice::{
    lattice_reduce_rows, lattice_reduce_rows_with_delta, validate_lll_rows, LatticeError,
};
pub use matrix::{jacobian, Matrix, MatrixError};
pub use numeric::{guess_integer_relation, PslqError};
pub use ode::{
    lower_to_first_order,
    sensitivity::{adjoint_system, sensitivity_system, AdjointSystem, SensitivitySystem},
    OdeError, ScalarODE, ODE,
};
pub use pattern::{match_pattern, Pattern, Substitution};
pub use poly::{
    factor_multivariate_z, factor_univariate_mod_p, factor_univariate_z, poly_normal, real_roots,
    real_roots_symbolic, refine_root, resultant, sparse_interpolate, sparse_interpolate_univariate,
    subresultant_prs, ConversionError, FactorError, MultiPoly, MultiPolyFactorization,
    RationalFunction, RealRootError, ResultantError, RootInterval, SparseInterpError, UniPoly,
    UniPolyFactorModP, UniPolyFactorization,
};

// Phase 24 — Horner form
pub use horner::{emit_horner_c, horner};
pub use simplify::rulesets::{log_exp_rules, log_exp_rules_safe, trig_rules};
pub use simplify::{
    rules_for_config, simplify, simplify_egraph, simplify_egraph_with, simplify_expanded,
    simplify_with, DepthCost, EgraphConfig, EgraphCost, OpCost, PatternRule, RewriteRule,
    SimplifyConfig, SizeCost, StabilityCost,
};
pub use sum::{
    gosper_certificate, gosper_normal_form, hypergeom_ratio, solve_linear_recurrence_homogeneous,
    sum_definite, sum_indefinite, verify_wz_pair, LinearRecurrenceError, RatFunc,
    RecurrenceSolution, SumError, WzPair,
};

// Phase 21 — JIT
pub use jit::{compile, eval_interp, CompiledFn, JitError};

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
pub use poly::groebner::{compute_groebner_basis_f5, GbPoly, GroebnerBasis, MonomialOrder};

pub use errors::AlkahestError;
pub use lean::emit_lean_expr as emit_lean;
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
pub use primitive::{Capabilities, CoverageReport, CoverageRow, Primitive, PrimitiveRegistry};
#[cfg(feature = "groebner")]
pub use solver::{
    expr_to_gbpoly, extract_regular_chain_from_basis, main_variable_recursive, solve_numerical,
    solve_polynomial_system, triangularize, CertifiedPoint, HomotopyError, HomotopyOpts,
    RegularChain, Solution, SolutionSet, SolverError,
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
    pub use crate::calculus::{series, Series, SeriesError};
    pub use crate::dae::{pantelides, DaeError, DAE};
    pub use crate::diff::{diff, diff_forward, DiffError};
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
    pub use crate::integrate::{integrate, IntegrationError};
    pub use crate::jit::{compile, CompiledFn, JitError};
    #[cfg(feature = "cuda")]
    pub use crate::jit::{compile_cuda, CudaCompiledFn, CudaError};
    #[allow(deprecated)]
    pub use crate::kernel::pool_persist::PoolPersistError;
    pub use crate::kernel::pool_persist::{load_from, open_persistent, save_to, IoError};
    pub use crate::kernel::{subs, Domain, ExprData, ExprDisplay, ExprId, ExprPool};
    pub use crate::lattice::{
        lattice_reduce_rows, lattice_reduce_rows_with_delta, validate_lll_rows, LatticeError,
    };
    pub use crate::logic::{
        dpll_sat, formula_from_expr, satisfiable, BoolClause, BoolLit, Formula, LogicError,
        Satisfiability,
    };
    pub use crate::matrix::{
        hermite_form, hermite_form_poly, jacobian, smith_form, smith_form_poly, IntegerMatrix,
        Matrix, MatrixError, NormalFormError, PolyMatrixQ, RatUniPoly,
    };
    pub use crate::numeric::{guess_integer_relation, PslqError};
    pub use crate::ode::{lower_to_first_order, OdeError, ScalarODE, ODE};
    pub use crate::pattern::{match_pattern, Pattern, Substitution};
    pub use crate::poly::{
        factor_multivariate_z, factor_univariate_mod_p, factor_univariate_z, poly_normal,
        real_roots, real_roots_symbolic, refine_root, resultant, sparse_interpolate,
        sparse_interpolate_univariate, subresultant_prs, ConversionError, FactorError, MultiPoly,
        MultiPolyFactorization, RationalFunction, RealRootError, ResultantError, RootInterval,
        SparseInterpError, UniPoly, UniPolyFactorModP, UniPolyFactorization,
    };
    pub use crate::primitive::{Primitive, PrimitiveRegistry};
    pub use crate::real::{cad_lift, cad_project, decide, decide_expr, CadError, QeResult};
    pub use crate::simplify::{simplify, simplify_with, SimplifyConfig};
    #[cfg(feature = "groebner")]
    pub use crate::solver::{
        expr_to_gbpoly, extract_regular_chain_from_basis, main_variable_recursive, solve_numerical,
        solve_polynomial_system, triangularize, CertifiedPoint, HomotopyError, HomotopyOpts,
        RegularChain, Solution, SolutionSet, SolverError,
    };
    pub use crate::sum::{
        gosper_certificate, gosper_normal_form, hypergeom_ratio,
        solve_linear_recurrence_homogeneous, sum_definite, sum_indefinite, verify_wz_pair,
        LinearRecurrenceError, RatFunc, RecurrenceSolution, SumError, WzPair,
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
    pub use crate::deriv::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
    pub use crate::horner::{emit_horner_c, horner};
    pub use crate::hybrid::{Event, GuardStructure, HybridODE};
    pub use crate::lean::emit_lean_expr as emit_lean;
    pub use crate::modular::{
        is_prime, lift_crt, mignotte_bound, rational_reconstruction, reduce_mod,
        select_lucky_prime, ModularError, ModularValue, MultiPolyFp,
    };
    pub use crate::numeric::{guess_integer_relation, PslqError};
    pub use crate::ode::sensitivity::{
        adjoint_system, sensitivity_system, AdjointSystem, SensitivitySystem,
    };
    pub use crate::poly::{sparse_interpolate, sparse_interpolate_univariate, SparseInterpError};
    pub use crate::simplify::{simplify_egraph, simplify_expanded};
    pub use crate::stablehlo::emit_stablehlo;

    #[cfg(feature = "parallel")]
    pub use crate::simplify::parallel::{simplify_par, simplify_par_with_config};

    #[cfg(feature = "groebner-cuda")]
    pub use crate::poly::groebner::GpuGroebnerError;
    #[cfg(feature = "groebner")]
    pub use crate::poly::groebner::{
        compute_groebner_basis_f5, GbPoly, GroebnerBasis, MonomialOrder,
    };
}
