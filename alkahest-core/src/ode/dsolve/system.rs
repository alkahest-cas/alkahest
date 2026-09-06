//! Linear constant-coefficient **systems** `y' = A·y + f(t)`.
//!
//! [`super::dsolve`] is scalar by construction — its input is one equation in
//! one unknown — so a two-compartment pharmacokinetic model
//! (`x' = −ka·x`, `y' = ka·x − ke·y`) or any state-space plant had no route
//! through it at all.  This module takes the [`ODE`] type the rest of the crate
//! already uses to model `d(state)/dt = rhs` and solves it in closed form when
//! the right-hand side is linear in the state with coefficients free of `t`.
//!
//! # Why a separate entry point rather than an overload of `dsolve`
//!
//! [`super::dsolve`]'s [`OdeInput`](super::OdeInput) names *one* unknown `y`
//! and a list of derivative symbols `y', y'', …` of that one unknown.  A system
//! has `n` unknowns and one derivative each; there is no reading of `OdeInput`
//! under which the two coincide, and widening it would mean a second
//! interpretation of `derivs` selected by a flag.  [`ODE`] already models
//! exactly the shape needed (`state_vars`, `rhs`, `time_var`), is already
//! public, and is already what the numeric integrators consume — so a system
//! solved here and a system integrated by `ode_integrate_rk45` are the same
//! object, which they would not be under an overload.
//!
//! # Method: Putzer, not Jordan
//!
//! The fundamental matrix is `Φ(t) = e^{At}`, computed by **Putzer's
//! algorithm**:
//!
//! ```text
//! e^{At} = Σ_{k=1}^{n} r_k(t)·P_{k−1},
//!     P_0 = I,  P_k = (A − λ_k I)·P_{k−1},
//!     r_1' = λ_1 r_1,  r_1(0) = 1,
//!     r_{k+1}' = λ_{k+1} r_{k+1} + r_k,  r_{k+1}(0) = 0.
//! ```
//!
//! It needs the eigenvalues **and nothing else** — no eigenvectors, no
//! nullspaces, no similarity transform.  That matters twice over:
//!
//! * A **defective** `A` costs nothing extra.  For `[[2,1],[0,2]]` the
//!   eigenvalue list is `2, 2`, the recurrence gives `r_1 = e^{2t}`,
//!   `r_2 = t·e^{2t}`, and `e^{At} = e^{2t}I + t·e^{2t}(A − 2I)` is
//!   `[[e^{2t}, t·e^{2t}], [0, e^{2t}]]` — the Jordan answer, with no Jordan
//!   machinery and no `jordan_form` refusal to route around.
//! * Every eigenvector routine in the crate has to decide whether a candidate
//!   pivot vanishes, and over symbolic entries that question is undecidable
//!   (see [`crate::matrix::zero_test`]).  Putzer moves the only such question
//!   to one place — `λ_i − λ_j`, in the recurrence for `r_{k+1}` — where it can
//!   be reported instead of guessed.
//!
//! `r_k` is kept as an explicit `Σ c·t^j·e^{λt}` list rather than being handed
//! to the integration engine, so the recurrence is exact term by term and the
//! `λ_i = λ_j` confluence is a visible branch rather than a failed integral.
//!
//! # The branch problem
//!
//! When `λ_i − λ_j` is neither provably zero nor a non-zero literal — `ka − ke`
//! in the compartment model — the generic formula divides by it.  The result is
//! then **undefined**, not merely non-general, at the confluence, so the
//! condition travels with the answer in
//! [`SystemSolution::side_conditions`] and the confluent limit is named in
//! [`SystemSolution::notes`].  This is a stronger caveat than the scalar
//! module's: there the two-exponential form still solves the equation at the
//! repeated root and only stops spanning, here the expression has a removable
//! singularity that this solver does not remove.
//!
//! # Verification
//!
//! Every component of the returned solution is substituted back into *its own*
//! equation and required to reduce to zero — symbolically, or numerically over
//! ℂ at sampled `t`, integration constants and free parameters, with the
//! finite / non-finite / unevaluable classification the scalar gate uses.  A
//! candidate that does not verify is withheld.

// Matrix and per-equation loops index several parallel structures at once
// (`A`, `P`, the candidate, its derivative and the system's own right-hand
// side), so an index is the clearer spelling.  `matrix::eigen` carries the same
// allow for the same reason.
#![allow(clippy::needless_range_loop)]

use super::verify::{eval_complex, C64};
use super::{contains, ddx, integrate_or_decline, simp, simp_plain, sub, AssumedSign, SolveCtx};
use crate::deriv::SideCondition;
use crate::kernel::eval_const::try_expr_f64;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::matrix::zero_test::{zero_status, ZeroStatus};
use crate::matrix::Matrix;
use crate::ode::ODE;
use crate::simplify::assumptions::AssumptionContext;
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// The general solution of a linear constant-coefficient system.
#[derive(Clone, Debug)]
pub struct SystemSolution {
    /// The state variables, in the order the [`ODE`] listed them.
    pub state_vars: Vec<ExprId>,
    /// `y_of_t[i]` is the closed form of `state_vars[i]` as a function of `t`.
    pub y_of_t: Vec<ExprId>,
    /// The fresh constants `C1 … Cn` appearing in [`Self::y_of_t`].
    pub constants: Vec<ExprId>,
    /// The fundamental matrix `e^{At}`, row-major, `n × n`.
    pub fundamental_matrix: Vec<Vec<ExprId>>,
    /// Short label of the route taken.
    pub method: &'static str,
    /// Conditions under which [`Self::y_of_t`] is defined and general.
    pub side_conditions: Vec<SideCondition>,
    /// Prose for the cases [`Self::side_conditions`] excludes.
    pub notes: Vec<String>,
}

/// Errors / declines from [`dsolve_system`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DsolveSystemError {
    /// A right-hand side is not affine in the state variables.
    NotLinear(String),
    /// A coefficient of the state depends on the independent variable, so the
    /// system is `y' = A(t)·y` and `e^{At}` is not its fundamental matrix.
    NonConstantCoefficient(String),
    /// The eigenvalues of `A` are not available in closed form.
    UnsupportedSpectrum(String),
    /// The forcing term could not be closed (a required integral is not
    /// elementary), or some other step declined.
    Unsupported(String),
    /// A candidate was produced and failed the substitution gate, so it is
    /// withheld rather than returned.
    VerificationFailed(String),
}

impl fmt::Display for DsolveSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DsolveSystemError::NotLinear(m) => {
                write!(f, "dsolve_system: system is not linear in the state: {m}")
            }
            DsolveSystemError::NonConstantCoefficient(m) => write!(
                f,
                "dsolve_system: coefficient depends on the independent variable: {m}"
            ),
            DsolveSystemError::UnsupportedSpectrum(m) => {
                write!(f, "dsolve_system: no closed-form spectrum: {m}")
            }
            DsolveSystemError::Unsupported(m) => write!(f, "dsolve_system: unsupported: {m}"),
            DsolveSystemError::VerificationFailed(m) => {
                write!(f, "dsolve_system: candidate failed verification: {m}")
            }
        }
    }
}

impl std::error::Error for DsolveSystemError {}

impl crate::errors::AlkahestError for DsolveSystemError {
    fn code(&self) -> &'static str {
        match self {
            DsolveSystemError::NotLinear(_) => "E-ODE-030",
            DsolveSystemError::NonConstantCoefficient(_) => "E-ODE-031",
            DsolveSystemError::UnsupportedSpectrum(_) => "E-ODE-032",
            DsolveSystemError::Unsupported(_) => "E-ODE-033",
            DsolveSystemError::VerificationFailed(_) => "E-ODE-034",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            DsolveSystemError::NotLinear(_) => Some(
                "each right-hand side must be affine in the state variables; \
                 nonlinear systems have no closed-form matrix exponential",
            ),
            DsolveSystemError::NonConstantCoefficient(_) => Some(
                "y' = A(t)·y needs a Magnus/Peano series, not e^{At}; freeze the \
                 coefficients or use a numeric integrator",
            ),
            DsolveSystemError::UnsupportedSpectrum(_) => Some(
                "the characteristic polynomial of A has no closed-form roots here; \
                 substitute concrete values for the symbolic entries, or reduce the \
                 system to 2×2/3×3 or a triangular shape",
            ),
            DsolveSystemError::Unsupported(_) => Some(
                "a required integral of the forcing term did not close in elementary \
                 form; no unevaluated-integral output is returned",
            ),
            DsolveSystemError::VerificationFailed(_) => Some(
                "the solver found a candidate that did not verify by substitution; \
                 this is reported rather than returned as a (possibly wrong) answer",
            ),
        }
    }
}

/// Solve `y' = A·y + f(t)` in closed form.
///
/// See the [module docs](self) for the method and for how a parameter-dependent
/// eigenvalue confluence is reported.
///
/// # Errors
///
/// See [`DsolveSystemError`].  Every returned solution has been verified by
/// substitution into every equation of the system.
pub fn dsolve_system(ode: &ODE, pool: &ExprPool) -> Result<SystemSolution, DsolveSystemError> {
    dsolve_system_with(ode, &AssumptionContext::new(), pool)
}

/// [`dsolve_system`], consulting caller-stated facts before assuming a branch.
///
/// A stated `λ_i − λ_j ≠ 0` (or any fact pinning it up to a non-zero constant
/// multiple) removes the corresponding side condition.
///
/// # Errors
///
/// See [`DsolveSystemError`].
pub fn dsolve_system_with(
    ode: &ODE,
    assumptions: &AssumptionContext,
    pool: &ExprPool,
) -> Result<SystemSolution, DsolveSystemError> {
    let mut ctx = SolveCtx {
        assumptions,
        conds: super::Conditions::default(),
    };
    solve_inner(ode, &mut ctx, pool)
}

// ---------------------------------------------------------------------------
// Extraction: y' = A·y + f(t)
// ---------------------------------------------------------------------------

/// Split each `rhs_i` into `Σ_j A[i][j]·y_j + f_i(t)`.
fn extract_linear_system(
    ode: &ODE,
    pool: &ExprPool,
) -> Result<(Matrix, Vec<ExprId>), DsolveSystemError> {
    let n = ode.state_vars.len();
    let t = ode.time_var;
    let mut rows: Vec<Vec<ExprId>> = Vec::with_capacity(n);
    let mut forcing: Vec<ExprId> = Vec::with_capacity(n);

    for &r in &ode.rhs {
        let mut row = Vec::with_capacity(n);
        for &v in &ode.state_vars {
            let c = ddx(r, v, pool).map_err(|e| DsolveSystemError::NotLinear(e.to_string()))?;
            // A coefficient still mentioning a state variable means the row is
            // not affine in the state.
            for &w in &ode.state_vars {
                if contains(c, w, pool) {
                    return Err(DsolveSystemError::NotLinear(format!(
                        "∂({})/∂({}) still depends on {}",
                        pool.display(r),
                        pool.display(v),
                        pool.display(w)
                    )));
                }
            }
            if contains(c, t, pool) {
                return Err(DsolveSystemError::NonConstantCoefficient(format!(
                    "∂({})/∂({}) = {} depends on {}",
                    pool.display(r),
                    pool.display(v),
                    pool.display(c),
                    pool.display(t)
                )));
            }
            row.push(simp(c, pool));
        }
        // f = rhs − Σ A[i][j]·y_j
        let mut acc = r;
        for (c, &v) in row.iter().zip(ode.state_vars.iter()) {
            acc = sub(acc, simp(pool.mul(vec![*c, v]), pool), pool);
        }
        for &w in &ode.state_vars {
            if contains(acc, w, pool) {
                return Err(DsolveSystemError::NotLinear(format!(
                    "the state-independent remainder {} still depends on {}",
                    pool.display(acc),
                    pool.display(w)
                )));
            }
        }
        rows.push(row);
        forcing.push(simp(acc, pool));
    }
    let a = Matrix::new(rows).map_err(|e| DsolveSystemError::NotLinear(e.to_string()))?;
    Ok((a, forcing))
}

// ---------------------------------------------------------------------------
// Spectrum
// ---------------------------------------------------------------------------

/// The `n` eigenvalues of `A`, listed with multiplicity.
///
/// A triangular `A` is answered from its diagonal directly.  That is not only a
/// shortcut: for a compartment model — lower triangular by construction, and
/// the case this module exists for — the general routine would build
/// `det(λI − A)`, fail to clear it to ℤ\[λ\] because the entries are symbolic,
/// and fall through to the Cardano formula, returning three nested-radical
/// expressions for what the diagonal states in three symbols.
fn spectrum(a: &Matrix, pool: &ExprPool) -> Result<Vec<ExprId>, DsolveSystemError> {
    let n = a.rows;
    if a.cols != n {
        return Err(DsolveSystemError::NotLinear(
            "coefficient matrix is not square".to_string(),
        ));
    }
    if is_triangular(a, pool) {
        return Ok((0..n).map(|i| simp(a.get(i, i), pool)).collect());
    }
    // A symbolic, non-triangular 3×3 has a cubic characteristic polynomial with
    // free-symbol coefficients, and the only closed form for its roots is
    // Cardano's.  Those radicals are correct only on a coordinated choice of
    // three cube-root branches, they are not what a caller wants to read, and
    // `simplify` grows rather than shrinks them: the resulting `e^{At}` is large
    // enough that building it, let alone verifying it, does not terminate in any
    // useful time.  Refusing here is cheaper and more honest than timing out and
    // then declining anyway.
    if n >= 3 && a.entries().iter().any(|&e| try_expr_f64(e, pool).is_none()) {
        return Err(DsolveSystemError::UnsupportedSpectrum(format!(
            "a {n}×{n} coefficient matrix with symbolic entries that is not triangular \
             needs Cardano (or higher) radicals for its spectrum; those do not simplify, \
             do not verify, and are not returned.  Substitute concrete values for the \
             symbolic entries, or write the system in triangular (compartment) form"
        )));
    }
    let eigs = crate::matrix::eigenvalues(a, pool)
        .map_err(|e| DsolveSystemError::UnsupportedSpectrum(e.to_string()))?;
    let mut out = Vec::with_capacity(n);
    for (lam, mult) in eigs {
        for _ in 0..mult {
            out.push(simp(lam, pool));
        }
    }
    if out.len() != n {
        return Err(DsolveSystemError::UnsupportedSpectrum(format!(
            "eigenvalue multiplicities sum to {} for an {n}×{n} matrix; the \
             characteristic polynomial did not split",
            out.len()
        )));
    }
    Ok(out)
}

/// Cross-check the eigenvalue list before anything is built on it.
///
/// `Π_i (z − λ_i)` must equal `det(zI − A)` — a polynomial identity, so testing
/// it at a few sampled `z` (and a few sampled parameter values) either confirms
/// it or refutes it.
///
/// This is not paranoia about arithmetic.  `matrix::eigenvalues` answers an
/// irreducible cubic with Cardano radicals, and Cardano's formula is only
/// correct on a *coordinated* choice of cube-root branches: written as two
/// independent radicals `(−q/2 ± √Δ)^{1/3}`, each evaluated on its own
/// principal branch, the constraint `AB = −p/3` fails and the three expressions
/// are not roots of the polynomial they came from.  Building `e^{At}` on such a
/// list produces a page-long candidate that the substitution gate then refuses
/// — correctly, but slowly and under an error that says "verification failed"
/// when the truth is "these are not the eigenvalues".
fn confirm_spectrum(
    a: &Matrix,
    lambdas: &[ExprId],
    pool: &ExprPool,
) -> Result<(), DsolveSystemError> {
    let n = a.rows;
    let mut params: Vec<ExprId> = Vec::new();
    for &e in a.entries() {
        collect_symbols(e, pool, &mut params);
    }
    for &l in lambdas {
        collect_symbols(l, pool, &mut params);
    }
    params.sort_by_key(|&s| pool.display(s).to_string());
    const PARAM_SETS: [&[f64]; 2] = [&[1.7, 0.6, 2.3, 1.1, 0.4], &[0.37, 1.9, 0.83, 2.7, 1.3]];
    const PROBES: [(f64, f64); 3] = [(0.53, 0.29), (-1.17, 0.71), (2.31, -0.43)];

    let mut checked = 0usize;
    for ps in PARAM_SETS {
        let mut env: HashMap<ExprId, C64> = HashMap::new();
        for (i, &p) in params.iter().enumerate() {
            env.insert(p, C64::real(ps[i % ps.len()]));
        }
        let Some(entries) = a
            .entries()
            .iter()
            .map(|&e| eval_complex(e, &env, pool).filter(|v| v.is_finite()))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let Some(lam) = lambdas
            .iter()
            .map(|&l| eval_complex(l, &env, pool).filter(|v| v.is_finite()))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        for (zr, zi) in PROBES {
            let z = C64::new(zr, zi);
            // det(zI − A)
            let mut m: Vec<Vec<C64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let e = entries[i * n + j].mul(C64::real(-1.0));
                            if i == j {
                                z.add(e)
                            } else {
                                e
                            }
                        })
                        .collect()
                })
                .collect();
            let Some(det) = complex_det(&mut m) else {
                continue;
            };
            let prod = lam.iter().fold(C64::real(1.0), |acc, &l| acc.mul(z.sub(l)));
            let scale = det.abs().max(prod.abs()).max(1.0);
            if det.sub(prod).abs() > 1e-7 * scale {
                return Err(DsolveSystemError::UnsupportedSpectrum(format!(
                    "the closed-form eigenvalues of the coefficient matrix are not roots \
                     of its characteristic polynomial when evaluated on principal \
                     branches (Π(z−λ) and det(zI−A) differ at z = {zr} + {zi}i).  This is \
                     the Cardano branch-coordination problem for an irreducible cubic; no \
                     answer is returned rather than one built on a spectrum that does not \
                     check out"
                )));
            }
            checked += 1;
        }
    }
    if checked == 0 {
        return Err(DsolveSystemError::UnsupportedSpectrum(
            "the eigenvalues of the coefficient matrix could not be evaluated at any \
             sample, so nothing confirms they are its spectrum"
                .to_string(),
        ));
    }
    Ok(())
}

/// Determinant by Gaussian elimination with partial pivoting, in place.
fn complex_det(m: &mut [Vec<C64>]) -> Option<C64> {
    let n = m.len();
    let mut det = C64::real(1.0);
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-14 {
            return Some(C64::real(0.0));
        }
        if piv != col {
            m.swap(col, piv);
            det = det.mul(C64::real(-1.0));
        }
        det = det.mul(m[col][col]);
        for r in (col + 1)..n {
            let f = m[r][col].div(m[col][col]);
            for c in col..n {
                let sub = f.mul(m[col][c]);
                m[r][c] = m[r][c].sub(sub);
            }
        }
    }
    if det.is_finite() {
        Some(det)
    } else {
        None
    }
}

fn is_triangular(a: &Matrix, pool: &ExprPool) -> bool {
    let n = a.rows;
    let upper =
        (0..n).all(|i| (0..i).all(|j| matches!(zero_status(pool, a.get(i, j)), ZeroStatus::Zero)));
    let lower = (0..n)
        .all(|i| ((i + 1)..n).all(|j| matches!(zero_status(pool, a.get(i, j)), ZeroStatus::Zero)));
    upper || lower
}

// ---------------------------------------------------------------------------
// Putzer
// ---------------------------------------------------------------------------

/// One `coeff · t^power · e^{lambda·t}` summand.
#[derive(Clone, Copy, Debug)]
struct ExpTerm {
    coeff: ExprId,
    power: usize,
    lambda: ExprId,
}

/// `r_1 … r_n` of Putzer's recurrence, as explicit exponential-polynomial sums.
///
/// `r_{k+1}(t) = ∫₀ᵗ e^{μ(t−s)} r_k(s) ds` with `μ = λ_{k+1}`, evaluated in
/// closed form per summand: with `d = λ − μ`,
///
/// ```text
/// ∫₀ᵗ e^{μ(t−s)} s^j e^{λs} ds
///   = t^{j+1}/(j+1) · e^{μt}                                      (d = 0)
///   = Σ_{i=0}^{j} (−1)^i j!/(j−i)! · t^{j−i}/d^{i+1} · e^{λt}
///     − (−1)^j j!/d^{j+1} · e^{μt}                                (d ≠ 0)
/// ```
fn putzer_r(lambdas: &[ExprId], ctx: &mut SolveCtx<'_>, pool: &ExprPool) -> Vec<Vec<ExpTerm>> {
    let mut rs: Vec<Vec<ExpTerm>> = Vec::with_capacity(lambdas.len());
    rs.push(vec![ExpTerm {
        coeff: pool.integer(1_i32),
        power: 0,
        lambda: lambdas[0],
    }]);
    for &mu in &lambdas[1..] {
        let prev = rs.last().expect("r_1 was pushed before the loop").clone();
        let mut next: Vec<ExpTerm> = Vec::new();
        for term in prev {
            let d = super::expand_powers(sub(term.lambda, mu, pool), pool);
            if term.lambda == mu || matches!(zero_status(pool, d), ZeroStatus::Zero) {
                next.push(ExpTerm {
                    coeff: super::div(term.coeff, pool.integer((term.power + 1) as i64), pool),
                    power: term.power + 1,
                    lambda: mu,
                });
                continue;
            }
            require_nonzero(d, ctx, pool);
            let j = term.power;
            let mut falling = 1_i64; // j!/(j−i)!
            for i in 0..=j {
                if i > 0 {
                    falling *= (j - i + 1) as i64;
                }
                let sign = if i % 2 == 0 { 1_i64 } else { -1_i64 };
                let denom = pool.pow(d, pool.integer((i + 1) as i64));
                let coeff = super::div(
                    pool.mul(vec![term.coeff, pool.integer(sign * falling)]),
                    denom,
                    pool,
                );
                next.push(ExpTerm {
                    coeff,
                    power: j - i,
                    lambda: term.lambda,
                });
            }
            // The `s = 0` boundary term, carrying `e^{μt}`.
            let jfact: i64 = (1..=(j as i64)).product::<i64>().max(1);
            let sign = if j % 2 == 0 { -1_i64 } else { 1_i64 };
            let denom = pool.pow(d, pool.integer((j + 1) as i64));
            next.push(ExpTerm {
                coeff: super::div(
                    pool.mul(vec![term.coeff, pool.integer(sign * jfact)]),
                    denom,
                    pool,
                ),
                power: 0,
                lambda: mu,
            });
        }
        rs.push(merge(next, pool));
    }
    rs
}

/// Add together summands with the same `(power, lambda)`.
fn merge(terms: Vec<ExpTerm>, pool: &ExprPool) -> Vec<ExpTerm> {
    let mut out: Vec<ExpTerm> = Vec::new();
    for t in terms {
        if let Some(slot) = out
            .iter_mut()
            .find(|o| o.power == t.power && o.lambda == t.lambda)
        {
            slot.coeff = simp(pool.add(vec![slot.coeff, t.coeff]), pool);
        } else {
            out.push(t);
        }
    }
    out.retain(|t| !matches!(zero_status(pool, t.coeff), ZeroStatus::Zero));
    out
}

fn exp_poly_to_expr(terms: &[ExpTerm], t: ExprId, pool: &ExprPool) -> ExprId {
    let mut sum = Vec::with_capacity(terms.len());
    for term in terms {
        let mut factors = vec![term.coeff];
        if term.power > 0 {
            factors.push(pool.pow(t, pool.integer(term.power as i64)));
        }
        if !matches!(zero_status(pool, term.lambda), ZeroStatus::Zero) {
            let lt = simp(pool.mul(vec![term.lambda, t]), pool);
            factors.push(pool.func("exp", vec![lt]));
        }
        sum.push(pool.mul(factors));
    }
    simp(pool.add(sum), pool)
}

/// `e^{At}` by Putzer's algorithm.
fn matrix_exponential(
    a: &Matrix,
    lambdas: &[ExprId],
    t: ExprId,
    ctx: &mut SolveCtx<'_>,
    pool: &ExprPool,
) -> Vec<Vec<ExprId>> {
    let n = a.rows;
    let rs = putzer_r(lambdas, ctx, pool);
    // P_0 = I, P_k = (A − λ_k I) P_{k−1}.
    let mut p: Vec<Vec<ExprId>> = (0..n)
        .map(|i| (0..n).map(|j| pool.integer(i32::from(i == j))).collect())
        .collect();
    let mut acc: Vec<Vec<ExprId>> = vec![vec![pool.integer(0_i32); n]; n];
    for (k, r) in rs.iter().enumerate() {
        let r_expr = exp_poly_to_expr(r, t, pool);
        for i in 0..n {
            for j in 0..n {
                acc[i][j] = pool.add(vec![acc[i][j], pool.mul(vec![r_expr, p[i][j]])]);
            }
        }
        if k + 1 < n {
            p = mat_mul_shift(a, &p, lambdas[k], pool);
        }
    }
    acc.iter()
        .map(|row| row.iter().map(|&e| simp(e, pool)).collect())
        .collect()
}

/// `(A − λI)·P`.
fn mat_mul_shift(a: &Matrix, p: &[Vec<ExprId>], lam: ExprId, pool: &ExprPool) -> Vec<Vec<ExprId>> {
    let n = a.rows;
    let mut out = vec![vec![pool.integer(0_i32); n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut terms = Vec::with_capacity(n + 1);
            for (k, prow) in p.iter().enumerate().take(n) {
                let mut aik = a.get(i, k);
                if i == k {
                    aik = pool.add(vec![aik, pool.mul(vec![pool.integer(-1_i32), lam])]);
                }
                terms.push(pool.mul(vec![aik, prow[j]]));
            }
            out[i][j] = simp(pool.add(terms), pool);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

fn solve_inner(
    ode: &ODE,
    ctx: &mut SolveCtx<'_>,
    pool: &ExprPool,
) -> Result<SystemSolution, DsolveSystemError> {
    let n = ode.state_vars.len();
    if n == 0 {
        return Err(DsolveSystemError::NotLinear(
            "the system has no state variables".to_string(),
        ));
    }
    if ode.rhs.len() != n {
        return Err(DsolveSystemError::NotLinear(
            "state variable and right-hand-side counts differ".to_string(),
        ));
    }
    let t = ode.time_var;
    let (a, forcing) = extract_linear_system(ode, pool)?;
    let lambdas = spectrum(&a, pool)?;
    confirm_spectrum(&a, &lambdas, pool)?;
    let phi = matrix_exponential(&a, &lambdas, t, ctx, pool);

    // Homogeneous part: y = Φ(t)·C.
    let mut used: HashSet<String> = HashSet::new();
    for &e in ode.rhs.iter().chain(ode.state_vars.iter()) {
        collect_symbol_names(e, pool, &mut used);
    }
    let mut gen = super::ConstGen::with_used(used);
    let constants: Vec<ExprId> = (0..n).map(|_| gen.fresh(pool)).collect();
    let mut y: Vec<ExprId> = (0..n)
        .map(|i| {
            let terms: Vec<ExprId> = (0..n)
                .map(|j| pool.mul(vec![phi[i][j], constants[j]]))
                .collect();
            simp(pool.add(terms), pool)
        })
        .collect();

    // Forcing: y_p = Φ(t)·∫ Φ(t)⁻¹ f(t) dt, using Φ(t)⁻¹ = Φ(−t).
    let forced = forcing.iter().any(|&f| !is_zero(f, pool));
    let method = if forced {
        let phi_inv = negate_time(&phi, t, pool);
        let mut integrated = Vec::with_capacity(n);
        for row in &phi_inv {
            let integrand = simp(
                pool.add(
                    row.iter()
                        .zip(forcing.iter())
                        .map(|(&p, &f)| pool.mul(vec![p, f]))
                        .collect(),
                ),
                pool,
            );
            let g = integrate_or_decline(integrand, t, pool)
                .map_err(|e| DsolveSystemError::Unsupported(e.to_string()))?;
            integrated.push(g);
        }
        for (i, yi) in y.iter_mut().enumerate() {
            let terms: Vec<ExprId> = (0..n)
                .map(|j| pool.mul(vec![phi[i][j], integrated[j]]))
                .collect();
            *yi = simp(pool.add(vec![*yi, pool.add(terms)]), pool);
        }
        "linear_system_putzer_variation_of_parameters"
    } else {
        "linear_system_putzer"
    };

    verify_system(ode, &y, &constants, pool)?;

    Ok(SystemSolution {
        state_vars: ode.state_vars.clone(),
        y_of_t: y,
        constants,
        fundamental_matrix: phi,
        method,
        side_conditions: ctx.conds.side.clone(),
        notes: ctx.conds.notes.clone(),
    })
}

/// `Φ(−t)`: substitute `t → −t` in every entry.
fn negate_time(phi: &[Vec<ExprId>], t: ExprId, pool: &ExprPool) -> Vec<Vec<ExprId>> {
    let neg_t = pool.mul(vec![pool.integer(-1_i32), t]);
    let mut m = HashMap::new();
    m.insert(t, neg_t);
    phi.iter()
        .map(|row| {
            row.iter()
                .map(|&e| simp(crate::kernel::subs::subs(e, &m, pool), pool))
                .collect()
        })
        .collect()
}

fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    matches!(zero_status(pool, e), ZeroStatus::Zero)
}

fn collect_symbol_names(expr: ExprId, pool: &ExprPool, out: &mut HashSet<String>) {
    pool.with(expr, |d| match d {
        ExprData::Symbol { name, .. } => {
            out.insert(name.clone());
        }
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            for &a in args {
                collect_symbol_names(a, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_symbol_names(*base, pool, out);
            collect_symbol_names(*exp, pool, out);
        }
        _ => {}
    });
}

/// Record `e ≠ 0` and the confluence it excludes, unless already settled.
fn require_nonzero(e: ExprId, ctx: &mut SolveCtx<'_>, pool: &ExprPool) {
    if super::is_settled_nonzero(e, pool) || ctx.assumed_sign(e, pool) != AssumedSign::Unknown {
        return;
    }
    ctx.conds.require_nonzero(e);
    ctx.conds.note(format!(
        "two eigenvalues of the coefficient matrix differ by {}; where that vanishes \
         the returned expression divides by zero.  Its limit there exists and is the \
         confluent (t·e^{{λt}}) form, but this solver does not take it — pass concrete \
         values, or assert the difference is non-zero, to get an answer valid at that \
         point",
        pool.display(e)
    ));
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Absolute floor of the "this residual is zero" band.
const ZERO_TOL: f64 = 1e-9;
/// Relative part of the band; see [`super::verify`] on why it is relative.
const REL_TOL: f64 = 1e-7;
/// Resolved, agreeing `(sample, equation)` pairs required for a numeric
/// certificate.
const MIN_AGREEING: usize = 12;

/// What one `(parameters, constants, t)` sample of the candidate produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleOutcome {
    /// Every component and every derivative is a finite complex number.
    Resolved,
    /// Every component evaluated, and at least one is not finite.
    CandidateNonFinite,
    /// Some component contains a construct the evaluator does not know.
    Unevaluable,
}

/// Is the system's own right-hand side finite at this `t` and these parameters,
/// candidate aside?  Probes several finite states, as the scalar gate's
/// `ode_is_regular_at` does; one finite probe is enough, since the question is
/// whether the *equations* have a singularity here.
fn system_is_regular_at(ode: &ODE, env: &HashMap<ExprId, C64>, pool: &ExprPool) -> bool {
    const PROBES: [f64; 3] = [1.0, 2.5, -1.5];
    PROBES.iter().any(|&p| {
        let mut e = env.clone();
        for (k, &v) in ode.state_vars.iter().enumerate() {
            e.insert(v, C64::real(p + 0.25 * k as f64));
        }
        ode.rhs
            .iter()
            .all(|&r| matches!(eval_complex(r, &e, pool), Some(v) if v.is_finite()))
    })
}

/// Substitute the candidate into **every** equation and require zero.
fn verify_system(
    ode: &ODE,
    y: &[ExprId],
    constants: &[ExprId],
    pool: &ExprPool,
) -> Result<(), DsolveSystemError> {
    let t = ode.time_var;
    let mut subst: HashMap<ExprId, ExprId> = HashMap::new();
    for (&v, &sol) in ode.state_vars.iter().zip(y.iter()) {
        subst.insert(v, sol);
    }
    let mut residuals = Vec::with_capacity(y.len());
    let mut derivs = Vec::with_capacity(y.len());
    for (i, &sol) in y.iter().enumerate() {
        let dy = ddx(sol, t, pool).map_err(|e| DsolveSystemError::Unsupported(e.to_string()))?;
        derivs.push(dy);
        let rhs = simp(crate::kernel::subs::subs(ode.rhs[i], &subst, pool), pool);
        residuals.push(simp(sub(dy, rhs, pool), pool));
    }
    if residuals
        .iter()
        .all(|&r| is_literal_zero(r, pool) || is_literal_zero(simp_plain(r, pool), pool))
    {
        return Ok(());
    }

    // Numeric fallback over ℂ: `t`, the constants, and any free parameters.
    let mut bound: Vec<ExprId> = vec![t];
    bound.extend_from_slice(&ode.state_vars);
    bound.extend_from_slice(constants);
    let mut params: Vec<ExprId> = Vec::new();
    for &r in &residuals {
        collect_symbols(r, pool, &mut params);
    }
    params.retain(|s| !bound.contains(s));
    params.sort_by_key(|&s| pool.display(s).to_string());

    let param_sets: [&[f64]; 3] = [
        &[1.7, 0.6, 2.3, 1.1, 0.4],
        &[0.37, 1.9, 0.83, 2.7, 1.3],
        &[2.9, 0.45, 1.15, 0.71, 3.3],
    ];
    let const_sets: [&[f64]; 2] = [&[5.7, 4.3, 6.4, 5.1, 4.9], &[8.5, 7.8, 6.6, 9.2, 7.1]];
    let t_samples = [0.11, 0.27, 0.43, 0.61, 0.79];

    let (mut agree, mut disagree, mut skipped) = (0usize, 0usize, 0usize);
    let mut sets_agreeing = 0usize;
    for ps in param_sets {
        let mut env: HashMap<ExprId, C64> = HashMap::new();
        for (i, &p) in params.iter().enumerate() {
            env.insert(p, C64::real(ps[i % ps.len()]));
        }
        let before = agree;
        for cs in const_sets {
            for (i, &c) in constants.iter().enumerate() {
                env.insert(c, C64::real(cs[i % cs.len()]));
            }
            for &tv in &t_samples {
                env.insert(t, C64::real(tv));
                // Split evaluation: the candidate on one side, the system's own
                // right-hand side on the other, so a non-finite sample can be
                // attributed rather than guessed at.
                //
                // The two failures are *not* symmetric, and treating them alike
                // is the mistake `integrate::gate` documents.  A construct the
                // evaluator does not know is a property of the expression, not
                // of the point, and carries no information.  A value it does
                // know and that comes back non-finite where the system itself is
                // perfectly regular is evidence the candidate is wrong — a
                // blanket skip there is how a domain hole clears a grid built to
                // catch one.
                let mut yv = Vec::with_capacity(y.len());
                let mut dv = Vec::with_capacity(y.len());
                let mut outcome = SampleOutcome::Resolved;
                for i in 0..y.len() {
                    match (
                        eval_complex(y[i], &env, pool),
                        eval_complex(derivs[i], &env, pool),
                    ) {
                        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => {
                            yv.push(a);
                            dv.push(b);
                        }
                        (Some(_), Some(_)) => {
                            outcome = SampleOutcome::CandidateNonFinite;
                            break;
                        }
                        _ => {
                            outcome = SampleOutcome::Unevaluable;
                            break;
                        }
                    }
                }
                match outcome {
                    SampleOutcome::Resolved => {}
                    SampleOutcome::Unevaluable => {
                        skipped += 1;
                        continue;
                    }
                    SampleOutcome::CandidateNonFinite => {
                        // Is the *system* regular here, candidate aside?  A
                        // constant-coefficient right-hand side is finite at any
                        // finite state unless the forcing term has a pole at
                        // this `t`, which is the one case that carries no
                        // information.
                        if system_is_regular_at(ode, &env, pool) {
                            disagree += 1;
                        } else {
                            skipped += 1;
                        }
                        continue;
                    }
                }
                let mut rhs_env = env.clone();
                for (k, &v) in ode.state_vars.iter().enumerate() {
                    rhs_env.insert(v, yv[k]);
                }
                let scale = yv
                    .iter()
                    .chain(dv.iter())
                    .fold(1.0_f64, |m, v| m.max(v.abs()));
                for i in 0..y.len() {
                    match eval_complex(ode.rhs[i], &rhs_env, pool) {
                        Some(v) if v.is_finite() => {
                            let resid = dv[i].sub(v).abs();
                            if resid <= ZERO_TOL.max(REL_TOL * scale) {
                                agree += 1;
                            } else {
                                disagree += 1;
                            }
                        }
                        // The system itself is not finite at this state — no
                        // information about the candidate.
                        _ => skipped += 1,
                    }
                }
            }
        }
        if agree > before {
            sets_agreeing += 1;
        }
    }
    if disagree == 0 && agree >= MIN_AGREEING && sets_agreeing >= 2 {
        return Ok(());
    }
    let mut shown = pool.display(residuals[0]).to_string();
    // A refused candidate can be pages long; the message is a diagnostic, not
    // the expression itself.
    if shown.len() > 400 {
        shown.truncate(400);
        shown.push_str(" …");
    }
    Err(DsolveSystemError::VerificationFailed(format!(
        "samples: {agree} agreeing, {disagree} disagreeing, {skipped} unresolved \
         ({sets_agreeing} parameter sets agreeing); first residual: {shown}"
    )))
}

fn is_literal_zero(e: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0)
}

fn collect_symbols(expr: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    pool.with(expr, |d| match d {
        ExprData::Symbol { .. } => {
            if !out.contains(&expr) {
                out.push(expr);
            }
        }
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            for &a in args {
                collect_symbols(a, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_symbols(*base, pool, out);
            collect_symbols(*exp, pool, out);
        }
        _ => {}
    });
}

#[cfg(test)]
mod tests;
