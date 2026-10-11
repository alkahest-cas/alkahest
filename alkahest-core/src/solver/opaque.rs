//! Sub-expressions free of the unknowns are coefficients, whatever they are.
//!
//! The Gröbner machinery only reads polynomials over ℚ in the unknowns *and*
//! the parameters, so `sin(q₁)` used to stop it cold even when it occurs only as
//! a coefficient — the equations of motion of a double pendulum,
//!
//! ```text
//! (m₁+m₂)·L₁²·q̈₁ + m₂·L₁·L₂·cos(q₁−q₂)·q̈₂ + … + (m₁+m₂)·g·L₁·sin(q₁) = 0
//! ```
//!
//! are **linear** in `q̈₁, q̈₂` and were refused with "sin is not polynomial".
//!
//! # The reduction
//!
//! Every maximal sub-expression that does not mention an unknown and is not
//! itself polynomial (a function application, a non-integer power) is replaced
//! by a fresh opaque parameter. The frozen system is solved by the ordinary
//! path, which treats those parameters as independent indeterminates, and the
//! original sub-expressions are substituted back into the solutions and the
//! hypotheses.
//!
//! That is sound in one direction only. A solution valid for *independent*
//! parameters stays valid under any specialisation at which it is defined, so
//! nothing it returns can be wrong where its hypotheses hold. What the
//! independence assumption can do is lose an identity between the frozen
//! atoms — `sin²q + cos²q = 1` — so that a determinant which is non-zero as a
//! polynomial in the atoms vanishes identically once they are substituted
//! back. The answer would then rest on a hypothesis that is never true.
//!
//! So nothing leaves this module unchecked:
//!
//! * every hypothesis the frozen solve assumed is mapped back and zero-tested;
//!   one that is identically zero refuses the call;
//! * every solution is substituted into the equations **as written** and
//!   evaluated at several generic points of the remaining parameters; a residual
//!   that is not zero there, or a solution that has no value at any of them,
//!   refuses the call. A parameter-free solution is checked in rigorous ball
//!   arithmetic instead.
//!
//! A positive-dimensional result (a Gröbner basis) is refused rather than
//! returned: its generators would be written in the opaque symbols.

use super::{
    assume_nonzero, solve_polynomial_system, substitute_solution, take_solve_side_conditions,
    verify, Solution, SolutionSet, SolverError,
};
use crate::eval::{eval_complex_f64, ComplexF64};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::poly::collect_free_vars;
use std::collections::HashMap;

/// Prefix of the fresh parameters. Not producible by the parser.
const OPAQUE_PREFIX: &str = "__opaque_coeff_";

/// Generic points tried by [`spot_check`].
const SPOT_POINTS: usize = 8;

/// Points that must evaluate for a parametric solution to be accepted.
const SPOT_REQUIRED: usize = 3;

/// Recursion guard for the tree walks.
const MAX_DEPTH: usize = 512;

/// The outcome of evaluating a solution at generic parameter values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpotCheck {
    /// The residual vanished (to rounding) at every point that evaluated, and
    /// enough of them did.
    Passed,
    /// Some point evaluated to a residual that is plainly not zero.
    Failed,
    /// Too few points evaluated to say anything.
    Inconclusive,
}

struct Freezer<'a> {
    pool: &'a ExprPool,
    vars: &'a [ExprId],
    memo: HashMap<ExprId, ExprId>,
    /// `(fresh symbol, original sub-expression)`, in creation order.
    frozen: Vec<(ExprId, ExprId)>,
    by_orig: HashMap<ExprId, ExprId>,
}

impl Freezer<'_> {
    fn mentions_var(&self, e: ExprId) -> bool {
        mentions_any(e, self.vars, self.pool, 0)
    }

    fn freeze_atom(&mut self, e: ExprId) -> ExprId {
        if let Some(&s) = self.by_orig.get(&e) {
            return s;
        }
        let name = format!("{OPAQUE_PREFIX}{}", self.frozen.len());
        let s = self.pool.symbol(name, Domain::Complex);
        self.frozen.push((s, e));
        self.by_orig.insert(e, s);
        s
    }

    fn walk(&mut self, e: ExprId, depth: usize) -> ExprId {
        if let Some(&hit) = self.memo.get(&e) {
            return hit;
        }
        if depth > MAX_DEPTH {
            return e;
        }
        let free = !self.mentions_var(e);
        let out = match self.pool.get(e) {
            ExprData::Add(args) => {
                let new: Vec<ExprId> = args.iter().map(|&a| self.walk(a, depth + 1)).collect();
                self.pool.add(new)
            }
            ExprData::Mul(args) => {
                let new: Vec<ExprId> = args.iter().map(|&a| self.walk(a, depth + 1)).collect();
                self.pool.mul(new)
            }
            ExprData::Pow { base, exp } => {
                let integer_exp = self.pool.with(exp, |d| matches!(d, ExprData::Integer(_)));
                if integer_exp {
                    let b = self.walk(base, depth + 1);
                    self.pool.pow(b, exp)
                } else if free {
                    self.freeze_atom(e)
                } else {
                    // A non-integer power of an unknown: not ours to rescue.
                    e
                }
            }
            ExprData::Func { .. } => {
                if free {
                    self.freeze_atom(e)
                } else {
                    e
                }
            }
            _ => e,
        };
        self.memo.insert(e, out);
        out
    }
}

/// Does `e` mention any of `vars`?
fn mentions_any(e: ExprId, vars: &[ExprId], pool: &ExprPool, depth: usize) -> bool {
    if vars.contains(&e) {
        return true;
    }
    if depth > MAX_DEPTH {
        return true;
    }
    match pool.get(e) {
        ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
            a.iter().any(|&x| mentions_any(x, vars, pool, depth + 1))
        }
        ExprData::Pow { base, exp } => {
            mentions_any(base, vars, pool, depth + 1) || mentions_any(exp, vars, pool, depth + 1)
        }
        _ => false,
    }
}

/// Is there still a non-polynomial node that mentions an unknown?
fn has_var_bearing_nonpoly(e: ExprId, vars: &[ExprId], pool: &ExprPool, depth: usize) -> bool {
    if depth > MAX_DEPTH {
        return true;
    }
    match pool.get(e) {
        ExprData::Add(a) | ExprData::Mul(a) => a
            .iter()
            .any(|&x| has_var_bearing_nonpoly(x, vars, pool, depth + 1)),
        ExprData::Pow { base, exp } => {
            let integer_exp = pool.with(exp, |d| matches!(d, ExprData::Integer(_)));
            if integer_exp {
                has_var_bearing_nonpoly(base, vars, pool, depth + 1)
            } else {
                mentions_any(e, vars, pool, 0)
            }
        }
        ExprData::Func { .. } => mentions_any(e, vars, pool, 0),
        _ => false,
    }
}

/// Retry a system the polynomial path refused, with the unknown-free
/// non-polynomial sub-expressions frozen into parameters.
///
/// `None` means the reduction does not apply (nothing to freeze, or an unknown
/// still sits inside a function or a fractional power): the caller keeps its
/// own refusal.
pub(super) fn solve_with_opaque_coefficients(
    equations: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
) -> Option<Result<SolutionSet, SolverError>> {
    let mut fz = Freezer {
        pool,
        vars,
        memo: HashMap::new(),
        frozen: Vec::new(),
        by_orig: HashMap::new(),
    };
    let frozen_eqs: Vec<ExprId> = equations.iter().map(|&e| fz.walk(e, 0)).collect();
    if fz.frozen.is_empty() {
        return None;
    }
    if frozen_eqs
        .iter()
        .any(|&e| has_var_bearing_nonpoly(e, vars, pool, 0))
    {
        return None;
    }
    let syms: Vec<ExprId> = fz.frozen.iter().map(|&(s, _)| s).collect();
    let origs: Vec<ExprId> = fz.frozen.iter().map(|&(_, o)| o).collect();
    let thaw = |e: ExprId| -> ExprId {
        let mut memo = HashMap::new();
        substitute_solution(e, &syms, &origs, pool, &mut memo)
    };

    // A square system affine in the unknowns — every equation of motion solved
    // for the accelerations — is `A·x = b`, solved by the adjugate. The
    // general path treats each opaque coefficient as an extra Lex variable,
    // which is exact but scales badly (a triple pendulum did not finish).
    let (inner, frozen_hyps) = match solve_affine_system(&frozen_eqs, vars, pool) {
        Some(out) => out,
        None => {
            let r = solve_polynomial_system(frozen_eqs, vars.to_vec(), pool);
            let h: Vec<ExprId> = take_solve_side_conditions()
                .into_iter()
                .filter_map(|c| match c {
                    crate::deriv::log::SideCondition::NonZero(e) => Some(e),
                    _ => None,
                })
                .collect();
            (r, h)
        }
    };
    // The frozen solve's hypotheses, thawed; re-recorded below once checked.
    let hypotheses: Vec<ExprId> = frozen_hyps.into_iter().map(thaw).collect();

    let refuse = |why: &str| -> Option<Result<SolutionSet, SolverError>> {
        Some(Err(SolverError::NotPolynomial(format!(
            "the system is polynomial in the unknowns once the sub-expressions free of them \
             are treated as coefficients, but {why}"
        ))))
    };

    let set = match inner {
        Err(e) => return Some(Err(e)),
        Ok(SolutionSet::NoSolution) => return Some(Ok(SolutionSet::NoSolution)),
        Ok(SolutionSet::Parametric(_)) => {
            return refuse("its solution set is not finite (a basis in the opaque coefficients would not be readable)");
        }
        Ok(SolutionSet::Finite(s)) => s,
    };

    for &h in &hypotheses {
        if crate::matrix::zero_test::zero_status(pool, h).is_proven_zero() {
            return refuse(&format!(
                "the solution divides by {}, which is identically zero once the coefficients \
                 are substituted back (an identity between them that the solve could not see)",
                pool.display(h)
            ));
        }
    }

    let thawed: Vec<Solution> = set
        .iter()
        .map(|sol| sol.iter().map(|&v| thaw(v)).collect())
        .collect();
    for sol in &thawed {
        let params_free = sol.iter().all(|&v| collect_free_vars(v, pool).is_empty())
            && equations
                .iter()
                .all(|&e| collect_free_vars(e, pool).iter().all(|s| vars.contains(s)));
        if params_free {
            let mut memo = HashMap::new();
            let mut ev = verify::CBallEval::default();
            for &eq in equations {
                let at = substitute_solution(eq, vars, sol, pool, &mut memo);
                match ev.eval(at, pool) {
                    Ok(ball) if ball.excludes_zero() => {
                        return refuse("a returned root does not satisfy the original equations")
                    }
                    Ok(_) => {}
                    Err(verify::VerifyGap::Undefined) => {
                        return refuse(
                            "a returned root is not in the domain of the original equations",
                        )
                    }
                    Err(verify::VerifyGap::Unsupported) => {}
                }
            }
        } else {
            match spot_check(equations, vars, sol, &[], pool) {
                SpotCheck::Passed => {}
                SpotCheck::Failed => {
                    return refuse(
                        "a returned solution does not satisfy the original equations at a generic \
                         point of the parameters",
                    )
                }
                SpotCheck::Inconclusive => {
                    return refuse(
                        "a returned solution has no value at generic points of the parameters \
                         (its denominator vanishes identically)",
                    )
                }
            }
        }
    }

    for h in hypotheses {
        if !crate::matrix::zero_test::settled_nonzero(h, pool) {
            assume_nonzero(h);
        }
    }
    Some(Ok(SolutionSet::Finite(thawed)))
}

/// Split `e` into `Σ coeffs[i]·vars[i] + constant`, with every coefficient and
/// the constant free of `vars`; `None` when `e` is not affine in them.
fn affine_parts(
    e: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
    depth: usize,
) -> Option<(Vec<ExprId>, ExprId)> {
    let n = vars.len();
    let zero = pool.integer(0_i32);
    if depth > MAX_DEPTH {
        return None;
    }
    if let Some(i) = vars.iter().position(|&v| v == e) {
        let mut c = vec![zero; n];
        c[i] = pool.integer(1_i32);
        return Some((c, zero));
    }
    if !mentions_any(e, vars, pool, 0) {
        return Some((vec![zero; n], e));
    }
    match pool.get(e) {
        ExprData::Add(args) => {
            let mut coeffs: Vec<Vec<ExprId>> = vec![Vec::new(); n];
            let mut consts = Vec::new();
            for a in args {
                let (c, k) = affine_parts(a, vars, pool, depth + 1)?;
                for (i, ci) in c.into_iter().enumerate() {
                    coeffs[i].push(ci);
                }
                consts.push(k);
            }
            Some((
                coeffs.into_iter().map(|v| pool.add(v)).collect(),
                pool.add(consts),
            ))
        }
        ExprData::Mul(args) => {
            let mut scale = Vec::new();
            let mut inner = None;
            for a in args {
                if mentions_any(a, vars, pool, 0) {
                    if inner.is_some() {
                        return None; // a product of two unknown-bearing factors
                    }
                    inner = Some(a);
                } else {
                    scale.push(a);
                }
            }
            let (c, k) = affine_parts(inner?, vars, pool, depth + 1)?;
            let scaled = |x: ExprId| {
                let mut f = scale.clone();
                f.push(x);
                pool.mul(f)
            };
            Some((c.into_iter().map(scaled).collect(), scaled(k)))
        }
        ExprData::Pow { base, exp } if exp == pool.integer(1_i32) => {
            affine_parts(base, vars, pool, depth + 1)
        }
        _ => None,
    }
}

/// `A·x = b` for a square system affine in `vars`, by the symbolic inverse.
///
/// Returns the solution set and the hypothesis it rests on (`det A ≠ 0`, when
/// the determinant is not a non-zero constant). `None` hands the system back
/// to the general path: not square, not affine, or an inverse that could not be
/// formed (singular as a polynomial in the opaque atoms, or undecided).
fn solve_affine_system(
    equations: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
) -> Option<(Result<SolutionSet, SolverError>, Vec<ExprId>)> {
    let n = vars.len();
    if equations.len() != n || n == 0 {
        return None;
    }
    let mut rows = Vec::with_capacity(n);
    let mut rhs = Vec::with_capacity(n);
    let neg_one = pool.integer(-1_i32);
    for &eq in equations {
        let (c, k) = affine_parts(eq, vars, pool, 0)?;
        let c: Vec<ExprId> = c
            .into_iter()
            .map(|x| crate::simplify::engine::simplify(x, pool).value)
            .collect();
        rows.push(c);
        rhs.push(pool.mul(vec![neg_one, k]));
    }
    let a = crate::matrix::Matrix::new(rows).ok()?;
    let inv = match crate::matrix::matrix_inverse(&a, pool) {
        Ok(inv) => inv,
        Err(_) => {
            let _ = crate::matrix::take_zero_test_refusal();
            let _ = crate::matrix::take_matrix_inverse_side_conditions();
            return None;
        }
    };
    let hyps: Vec<ExprId> = crate::matrix::take_matrix_inverse_side_conditions()
        .into_iter()
        .filter_map(|c| match c {
            crate::deriv::log::SideCondition::NonZero(e) => Some(e),
            _ => None,
        })
        .collect();
    let sol: Solution = (0..n)
        .map(|i| {
            let terms: Vec<ExprId> = (0..n)
                .map(|j| pool.mul(vec![inv.get(i, j), rhs[j]]))
                .collect();
            pool.add(terms)
        })
        .collect();
    Some((Ok(SolutionSet::Finite(vec![sol])), hyps))
}

/// A small deterministic generator for the sample points (no global RNG: the
/// verdict must be reproducible).
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Evaluate `solution` (parallel to `vars`) in `equations` at generic values of
/// every other free symbol.
///
/// Integer-domain symbols get small integers (they index a periodic family);
/// every other symbol gets a value in `[0.45, 2.05]`. Positive samples keep the
/// real-mode transcendental solutions (`log(kₑ/kₐ)`) on their principal branch;
/// a polynomial identity that holds on an open set holds everywhere, so nothing
/// is lost for the opaque-coefficient path.
///
/// `extra_integer` lists symbols to treat as integers even when their domain
/// says otherwise.
pub(crate) fn spot_check(
    equations: &[ExprId],
    vars: &[ExprId],
    solution: &[ExprId],
    extra_integer: &[ExprId],
    pool: &ExprPool,
) -> SpotCheck {
    let mut symbols: Vec<ExprId> = Vec::new();
    for &e in equations.iter().chain(solution.iter()) {
        for s in collect_free_vars(e, pool) {
            if !vars.contains(&s)
                && !symbols.contains(&s)
                && !crate::eval::symbols::is_named_constant(s, pool)
            {
                symbols.push(s);
            }
        }
    }
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15 ^ (symbols.len() as u64));
    let mut ok_points = 0usize;
    for _ in 0..SPOT_POINTS {
        let mut bind: HashMap<ExprId, ComplexF64> = HashMap::new();
        for &s in &symbols {
            let integer = extra_integer.contains(&s)
                || pool.with(s, |d| {
                    matches!(
                        d,
                        ExprData::Symbol {
                            domain: Domain::Integer,
                            ..
                        }
                    )
                });
            let v = if integer {
                (rng.next_unit() * 7.0).floor() - 3.0
            } else {
                0.45 + 1.6 * rng.next_unit()
            };
            bind.insert(s, ComplexF64::new(v, 0.0));
        }
        let mut values = Vec::with_capacity(vars.len());
        let mut defined = true;
        for &v in solution {
            match eval_complex_f64(v, pool, &bind) {
                Ok(c) => values.push(c),
                Err(_) => {
                    defined = false;
                    break;
                }
            }
        }
        if !defined {
            continue;
        }
        for (i, &v) in vars.iter().enumerate() {
            bind.insert(v, values[i]);
        }
        let mut point_ok = true;
        for &eq in equations {
            let Ok(r) = eval_complex_f64(eq, pool, &bind) else {
                point_ok = false;
                break;
            };
            let scale = term_scale(eq, pool, &bind).max(1.0);
            if r.re.hypot(r.im) > 1e-7 * scale {
                return SpotCheck::Failed;
            }
        }
        if point_ok {
            ok_points += 1;
        }
    }
    if ok_points >= SPOT_REQUIRED {
        SpotCheck::Passed
    } else {
        SpotCheck::Inconclusive
    }
}

/// The size of the largest top-level term of `eq` at the point — the scale the
/// cancellation in its residual happened at.
fn term_scale(eq: ExprId, pool: &ExprPool, bind: &HashMap<ExprId, ComplexF64>) -> f64 {
    let terms: Vec<ExprId> = match pool.get(eq) {
        ExprData::Add(a) => a,
        _ => vec![eq],
    };
    terms
        .iter()
        .filter_map(|&t| eval_complex_f64(t, pool, bind).ok())
        .map(|c| c.re.hypot(c.im))
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;

    /// A rotation by `q`: `[sin q, cos q; −cos q, sin q]·[x; y] = [a; b]`,
    /// linear in `x, y` with transcendental coefficients.
    #[test]
    fn rotation_system_is_solved_and_checked() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let q = pool.symbol("q", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let m1 = pool.integer(-1_i32);
        let s = pool.func("sin", vec![q]);
        let c = pool.func("cos", vec![q]);
        let e1 = pool.add(vec![
            pool.mul(vec![s, x]),
            pool.mul(vec![c, y]),
            pool.mul(vec![m1, a]),
        ]);
        let e2 = pool.add(vec![
            pool.mul(vec![m1, c, x]),
            pool.mul(vec![s, y]),
            pool.mul(vec![m1, b]),
        ]);
        let set = solve_polynomial_system(vec![e1, e2], vec![x, y], &pool).expect("solved");
        let SolutionSet::Finite(sols) = set else {
            panic!("expected a finite set")
        };
        assert_eq!(sols.len(), 1);
        let env: HashMap<ExprId, f64> = [(q, 0.3), (a, 1.2), (b, -0.4)].into_iter().collect();
        let xv = eval_interp(sols[0][0], &env, &pool).unwrap();
        let yv = eval_interp(sols[0][1], &env, &pool).unwrap();
        // The inverse of a rotation is its transpose (det = sin² + cos² = 1).
        let (sq, cq) = (0.3f64.sin(), 0.3f64.cos());
        assert!((xv - (sq * 1.2 - cq * -0.4)).abs() < 1e-12);
        assert!((yv - (cq * 1.2 + sq * -0.4)).abs() < 1e-12);
    }

    /// `(sin²a + cos²a − 1)·x = 1`: the coefficient is identically zero, which
    /// the frozen atoms cannot see. Nothing may be returned that divides by it.
    #[test]
    fn hidden_identity_is_refused() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let two = pool.integer(2_i32);
        let coeff = pool.add(vec![
            pool.pow(pool.func("sin", vec![a]), two),
            pool.pow(pool.func("cos", vec![a]), two),
            pool.integer(-1_i32),
        ]);
        let eq = pool.add(vec![pool.mul(vec![coeff, x]), pool.integer(-1_i32)]);
        match solve_polynomial_system(vec![eq], vec![x], &pool) {
            Err(SolverError::NotPolynomial(_)) => {}
            Ok(SolutionSet::Finite(s)) if s.is_empty() => {}
            Ok(SolutionSet::NoSolution) => {}
            Ok(SolutionSet::Finite(s)) => {
                panic!("returned {} solution(s) through a zero divisor", s.len())
            }
            Ok(SolutionSet::Parametric(_)) => panic!("returned a basis"),
            Err(e) => panic!("unexpected error {e}"),
        }
        let _ = take_solve_side_conditions();
    }

    #[test]
    fn unknown_under_a_function_is_not_frozen() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let eq = pool.add(vec![pool.func("sin", vec![x]), pool.integer(-1_i32)]);
        assert!(matches!(
            solve_polynomial_system(vec![eq], vec![x], &pool),
            Err(SolverError::NotPolynomial(_))
        ));
    }
}
