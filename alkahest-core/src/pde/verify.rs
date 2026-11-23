//! Verification of candidate PDE solutions (plan decision D9).
//!
//! A candidate is substituted into every equation and every boundary /
//! initial condition — derivatives included, at any argument, through
//! [`crate::diff::undefined::replace_function`] — and each residual goes to
//! the same decision procedure as `integrate`'s verification and `is_zero`
//! ([`crate::simplify::zero::zero_test`]):
//!
//! * `proved_zero` is exact symbolic reasoning, never sampling;
//! * numeric evidence (rigorous Arb balls at admissible points) can only
//!   produce `proved_nonzero` — a **refutation** — or `probably_zero`, which
//!   is reported but never counts as verified.
//!
//! Arbitrary functions in a general solution (`F1(x − c t)`) stay symbolic:
//! `F1(…)` and its derivatives are independent atoms of the normal form, which
//! is exactly "for every smooth `F1`".  When that is inconclusive the
//! arbitrary functions are additionally instantiated with fixed concrete test
//! functions; a nonzero residual for an instance refutes the general claim,
//! a zero one proves nothing and is not used.

use crate::diff::undefined::{replace_function, view};
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::error::PdeError;
use super::jet::{decode_unknown, push_children};
use super::problem::{ConditionKind, PdeCondition, PdeProblem};

/// Overall verdict of [`verify_pde_solution`].  The strings match
/// `integrate`'s `verification["status"]` vocabulary, plus `refuted`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VerificationStatus {
    /// Every residual is `proved_zero`.
    ExactlyVerified,
    /// No residual was refuted and every one is `proved_zero` or
    /// `probably_zero`.  Evidence, not proof.
    NumericallyChecked,
    /// Some residual is `proved_nonzero`: the candidate is not a solution.
    Refuted,
    /// Nothing conclusive for at least one residual.
    Unverified,
}

impl VerificationStatus {
    /// `"exactly_verified"`, `"numerically_checked"`, `"refuted"` or
    /// `"unverified"`.
    pub fn as_str(self) -> &'static str {
        match self {
            VerificationStatus::ExactlyVerified => "exactly_verified",
            VerificationStatus::NumericallyChecked => "numerically_checked",
            VerificationStatus::Refuted => "refuted",
            VerificationStatus::Unverified => "unverified",
        }
    }

    /// Whether this is a proof (only [`VerificationStatus::ExactlyVerified`]).
    pub fn is_verified(self) -> bool {
        self == VerificationStatus::ExactlyVerified
    }
}

/// One residual: an equation or a condition with the candidate substituted.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResidualCheck {
    /// `"pde"`, `"bc"` or `"ic"`.
    pub kind: &'static str,
    /// Position among the equations / conditions of that kind.
    pub index: usize,
    /// The residual after substitution and simplification.
    pub residual: ExprId,
    /// The zero-test verdict.
    pub status: ZeroStatus,
    /// How it was reached (`"simplify"`, `"rational_normal_form"`,
    /// `"arb_ball"`, …, or `"instantiated_arbitrary_functions"` for a
    /// refutation found by instantiating arbitrary functions).
    pub method: &'static str,
    /// Sample points and enclosures, when numeric.
    pub evidence: Vec<String>,
}

/// Result of [`verify_pde_solution`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PdeVerification {
    /// The overall verdict.
    pub status: VerificationStatus,
    /// One entry per equation, then per condition.
    pub checks: Vec<ResidualCheck>,
    /// Names of the arbitrary functions the candidate contains (undefined
    /// functions other than the unknowns), kept symbolic.
    pub arbitrary_functions: Vec<String>,
}

/// Verify that `solution[i]` (an expression in the independent variables)
/// solves `problem` for `problem.unknowns()[i]`: every equation, boundary and
/// initial condition.
///
/// # Errors
///
/// `E-PDE-013` when the number of solution components differs from the
/// number of unknowns.
pub fn verify_pde_solution(
    problem: &PdeProblem,
    solution: &[ExprId],
    pool: &ExprPool,
) -> Result<PdeVerification, PdeError> {
    verify_solution(
        problem.equations(),
        problem.unknowns(),
        solution,
        problem.conditions(),
        pool,
    )
}

/// [`verify_pde_solution`] on loose parts: `equations` (each `= 0`) in
/// `unknowns`, candidate `solution`, and `conditions`.
///
/// # Errors
///
/// `E-PDE-001` for a malformed unknown; `E-PDE-013` for a component-count
/// mismatch.
pub fn verify_solution(
    equations: &[ExprId],
    unknowns: &[ExprId],
    solution: &[ExprId],
    conditions: &[PdeCondition],
    pool: &ExprPool,
) -> Result<PdeVerification, PdeError> {
    if solution.len() != unknowns.len() {
        return Err(PdeError::InvalidCandidate(format!(
            "{} solution components for {} unknowns",
            solution.len(),
            unknowns.len()
        )));
    }
    let mut names = Vec::with_capacity(unknowns.len());
    for &u in unknowns {
        names.push(decode_unknown(u, pool)?.name);
    }
    let substitute = |e: ExprId| -> Result<ExprId, PdeError> {
        let mut cur = e;
        for (&u, &s) in unknowns.iter().zip(solution) {
            cur = replace_function(cur, u, s, pool).map_err(|err| {
                PdeError::InvalidCandidate(format!(
                    "cannot substitute {} for {}: {err}",
                    pool.display(s),
                    pool.display(u)
                ))
            })?;
        }
        Ok(cur)
    };
    // Arbitrary functions: undefined functions in the candidate that are not
    // unknowns (and every application of them, with its arity).
    let mut arbitrary: Vec<(String, usize)> = Vec::new();
    for &s in solution {
        collect_undefined(s, pool, &mut arbitrary);
    }
    arbitrary.retain(|(n, _)| !names.contains(n));
    arbitrary.sort();
    arbitrary.dedup();

    let mut items: Vec<(&'static str, usize, ExprId)> = Vec::new();
    for (i, &e) in equations.iter().enumerate() {
        items.push(("pde", i, e));
    }
    let (mut nb, mut ni) = (0, 0);
    for c in conditions {
        let (kind, idx) = match c.kind {
            ConditionKind::Initial => {
                ni += 1;
                ("ic", ni - 1)
            }
            _ => {
                nb += 1;
                ("bc", nb - 1)
            }
        };
        items.push((kind, idx, c.residual(pool)));
    }

    let mut checks = Vec::with_capacity(items.len());
    for (kind, index, e) in items {
        let r = substitute(e)?;
        // Alpha-equivalent sums must be one atom for the zero test, and
        // linearity / constant folding should have its chance first.
        let r = super::sum::simplify_sums(r, &[], pool);
        let r = super::sum::alpha_normalize(r, pool);
        let r = super::integral::alpha_normalize_integrals(r, pool);
        let r = crate::simplify::engine::simplify(r, pool).value;
        let r = if super::integral::has_integral(r, pool) {
            let c = super::integral::combine_integrals(r, pool);
            crate::simplify::engine::simplify(c, pool).value
        } else {
            r
        };
        let zt = guarded(zero_test(r, pool, None), r, pool);
        let mut check = ResidualCheck {
            kind,
            index,
            residual: r,
            status: zt.status,
            method: zt.method,
            evidence: zt.evidence,
        };
        if check.status != ZeroStatus::ProvedZero
            && check.status != ZeroStatus::ProvedNonzero
            && !arbitrary.is_empty()
        {
            if let Some(ev) = refute_by_instances(r, &arbitrary, pool) {
                check.status = ZeroStatus::ProvedNonzero;
                check.method = "instantiated_arbitrary_functions";
                check.evidence = ev;
            }
        }
        checks.push(check);
    }
    let status = if checks.iter().all(|c| c.status == ZeroStatus::ProvedZero) {
        VerificationStatus::ExactlyVerified
    } else if checks.iter().any(|c| c.status == ZeroStatus::ProvedNonzero) {
        VerificationStatus::Refuted
    } else if checks
        .iter()
        .all(|c| matches!(c.status, ZeroStatus::ProvedZero | ZeroStatus::ProbablyZero))
    {
        VerificationStatus::NumericallyChecked
    } else {
        VerificationStatus::Unverified
    };
    Ok(PdeVerification {
        status,
        checks,
        arbitrary_functions: arbitrary.into_iter().map(|(n, _)| n).collect(),
    })
}

/// A `proved_nonzero` from the rational normal form is a statement about
/// *independent atoms*.  A `Sum` or `RootOfFamily` is such an atom to the
/// normal form, but two different sums can still be equal (`Σ(a+b)` and
/// `Σa + Σb`), so with one present that verdict is downgraded to `unknown`:
/// verification may fail to prove, never wrongly refute.
fn guarded(
    zt: crate::simplify::zero::ZeroTest,
    r: ExprId,
    pool: &ExprPool,
) -> crate::simplify::zero::ZeroTest {
    if zt.status == ZeroStatus::ProvedNonzero
        && zt.method == "rational_normal_form"
        && has_opaque_head(r, pool)
    {
        let mut z = zt;
        z.status = ZeroStatus::Unknown;
        z.method = "none";
        z.evidence =
            vec!["a Sum or RootOfFamily is opaque to the normal form; nonzero not proved".into()];
        return z;
    }
    zt
}

fn has_opaque_head(e: ExprId, pool: &ExprPool) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let opaque = pool.with(n, |d| {
            matches!(d, crate::kernel::ExprData::Func { name, .. }
                if name == super::sum::SUM
                    || name == super::root_family::ROOT_OF_FAMILY
                    || name == super::integral::INTEGRAL)
        });
        if opaque {
            return true;
        }
        push_children(n, pool, &mut stack);
    }
    false
}

fn collect_undefined(e: ExprId, pool: &ExprPool, out: &mut Vec<(String, usize)>) {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(v) = view(n, pool) {
            out.push((v.name.clone(), v.args.len()));
            stack.extend(v.args);
            continue;
        }
        push_children(n, pool, &mut stack);
    }
}

/// Instantiate the arbitrary functions with fixed, generic test functions and
/// look for a proved-nonzero residual.  Returns the evidence of the
/// refutation, or `None`.
fn refute_by_instances(
    r: ExprId,
    arbitrary: &[(String, usize)],
    pool: &ExprPool,
) -> Option<Vec<String>> {
    // Two families: a transcendental one and a polynomial one, with distinct
    // coefficients per function so no accidental cancellation between them.
    for family in 0..2 {
        let mut cur = r;
        let mut shown = Vec::new();
        for (k, (name, arity)) in arbitrary.iter().enumerate() {
            let xi: Vec<ExprId> = (0..*arity)
                .map(|j| pool.symbol(format!("__alkahest_pde_xi{j}"), crate::kernel::Domain::Real))
                .collect();
            let lin = pool.add(
                xi.iter()
                    .enumerate()
                    .map(|(j, &x)| pool.mul(vec![pool.rational(j as i64 + 2 + k as i64, 3), x]))
                    .collect(),
            );
            let body = if family == 0 {
                pool.add(vec![
                    pool.func("sin", vec![lin]),
                    pool.func(
                        "exp",
                        vec![pool.mul(vec![pool.rational(1, 5 + k as i64), lin])],
                    ),
                ])
            } else {
                pool.add(vec![
                    pool.pow(lin, pool.integer(3 + k as i64)),
                    pool.mul(vec![pool.rational(1, 7), pool.pow(lin, pool.integer(2))]),
                    lin,
                ])
            };
            let pattern = pool.func(name.clone(), xi.clone());
            cur = replace_function(cur, pattern, body, pool).ok()?;
            shown.push(format!(
                "{} := {}",
                pool.display(pattern),
                pool.display(body)
            ));
        }
        let zt = guarded(zero_test(cur, pool, None), cur, pool);
        if zt.status == ZeroStatus::ProvedNonzero {
            let mut ev = vec![format!(
                "residual is nonzero for the instance {}",
                shown.join(", ")
            )];
            ev.extend(zt.evidence);
            return Some(ev);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    #[test]
    fn dalembert_general_solution_verifies() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let c = pool.symbol("c", Domain::Real);
        let u = pool.func("u", vec![x, t]);
        let d = |e, v| crate::diff::diff(e, v, &pool).unwrap().value;
        let wave = pool.add(vec![
            d(d(u, t), t),
            pool.mul(vec![
                pool.integer(-1),
                pool.pow(c, pool.integer(2)),
                d(d(u, x), x),
            ]),
        ]);
        let ct = pool.mul(vec![c, t]);
        let f1 = pool.func(
            "F1",
            vec![pool.add(vec![x, pool.mul(vec![pool.integer(-1), ct])])],
        );
        let f2 = pool.func("F2", vec![pool.add(vec![x, ct])]);
        let sol = pool.add(vec![f1, f2]);
        let p = PdeProblem::new(&[wave], &[u], &pool).unwrap();
        let v = verify_pde_solution(&p, &[sol], &pool).unwrap();
        assert_eq!(
            v.status,
            VerificationStatus::ExactlyVerified,
            "{:?}",
            v.checks
        );
        assert_eq!(v.arbitrary_functions, ["F1", "F2"]);
        // a wrong general solution is refuted
        let bad = pool.func(
            "F1",
            vec![pool.add(vec![x, pool.mul(vec![pool.integer(2), ct])])],
        );
        let v = verify_pde_solution(&p, &[bad], &pool).unwrap();
        assert_eq!(v.status, VerificationStatus::Refuted);
    }

    #[test]
    fn series_candidate_verifies_term_wise() {
        // u = Σ_{k=1}^{n} e^{−k² t} sin(k x) solves u_t = u_xx for every n.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let k = pool.symbol("k", Domain::Integer);
        let n = pool.symbol("n", Domain::Integer);
        let u = pool.func("u", vec![x, t]);
        let d = |e, v| crate::diff::diff(e, v, &pool).unwrap().value;
        let heat = pool.add(vec![
            d(u, t),
            pool.mul(vec![pool.integer(-1), d(d(u, x), x)]),
        ]);
        let term = pool.mul(vec![
            pool.func(
                "exp",
                vec![pool.mul(vec![pool.integer(-1), pool.pow(k, pool.integer(2)), t])],
            ),
            pool.func("sin", vec![pool.mul(vec![k, x])]),
        ]);
        let sol = crate::pde::sum::sum(term, k, pool.integer(1), n, &pool).unwrap();
        let p = PdeProblem::new(&[heat], &[u], &pool).unwrap();
        let v = verify_pde_solution(&p, &[sol], &pool).unwrap();
        assert_eq!(
            v.status,
            VerificationStatus::ExactlyVerified,
            "{:?}",
            v.checks
        );
        // A wrong series is never *refuted* through the opaque Sum atom.
        let wrong = crate::pde::sum::sum(
            pool.func("sin", vec![pool.mul(vec![k, x])]),
            k,
            pool.integer(1),
            n,
            &pool,
        )
        .unwrap();
        let v = verify_pde_solution(&p, &[wrong], &pool).unwrap();
        assert_ne!(v.status, VerificationStatus::ExactlyVerified);
    }

    #[test]
    fn heat_ibvp_with_conditions() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let pi = pool.symbol("pi", Domain::Real);
        let u = pool.func("u", vec![x, t]);
        let d = |e, v| crate::diff::diff(e, v, &pool).unwrap().value;
        let heat = pool.add(vec![
            d(u, t),
            pool.mul(vec![pool.integer(-1), d(d(u, x), x)]),
        ]);
        let zero = pool.integer(0);
        let sinpx = pool.func("sin", vec![pool.mul(vec![pi, x])]);
        let p = PdeProblem::new(&[heat], &[u], &pool)
            .unwrap()
            .bc(pool.func("u", vec![zero, t]), zero, &pool)
            .unwrap()
            .bc(pool.func("u", vec![pool.integer(1), t]), zero, &pool)
            .unwrap()
            .ic(pool.func("u", vec![x, zero]), sinpx, &pool)
            .unwrap();
        let decay = pool.func(
            "exp",
            vec![pool.mul(vec![pool.integer(-1), pool.pow(pi, pool.integer(2)), t])],
        );
        let sol = pool.mul(vec![sinpx, decay]);
        let v = verify_pde_solution(&p, &[sol], &pool).unwrap();
        assert_eq!(
            v.status,
            VerificationStatus::ExactlyVerified,
            "{:?}",
            v.checks
        );
        assert_eq!(v.checks.len(), 4);
        // wrong decay rate: PDE refuted, conditions still fine
        let wrong = pool.mul(vec![
            sinpx,
            pool.func("exp", vec![pool.mul(vec![pool.integer(-1), t])]),
        ]);
        let v = verify_pde_solution(&p, &[wrong], &pool).unwrap();
        assert_eq!(v.status, VerificationStatus::Refuted);
        assert_eq!(v.checks[0].status, ZeroStatus::ProvedNonzero);
        assert_eq!(v.checks[1].status, ZeroStatus::ProvedZero);
    }
}
