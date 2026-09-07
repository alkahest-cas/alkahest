//! First-order ODE classes for [`super::dsolve`].
//!
//! Strategy: most classes need the equation solved for `y'`.  We extract the
//! coefficient of `y'` (`A = ∂F/∂y'`) and the remainder (`B = F − A·y'`); when
//! `A` is free of `y'` the equation is linear in `y'` and `y' = −B/A`.  Clairaut
//! (nonlinear in `y'`) is handled directly on the equation.
//!
//! # Classification order
//!
//! An equation belongs to several classes at once far more often than not —
//! `y' = y` is separable, linear, Bernoulli-degenerate, exact after an
//! integrating factor and homogeneous — so the order is part of the contract,
//! not an implementation detail.  It is:
//!
//! | # | class | why here |
//! |---|---|---|
//! | 1 | Clairaut | the only class that is *nonlinear in `y'`*; it is tried on the raw equation, before the `y' = −B/A` reduction that the rest require |
//! | 2 | separable | the cheapest test, and it produces the most direct answer: two quadratures and no integrating factor |
//! | 3 | linear | exact, always closed-form when its two integrals close, and the answer is explicit by construction |
//! | 4 | Bernoulli | linear after `v = y^{1−n}`; strictly narrower than the exact route below and much cheaper |
//! | 5 | exact (+ the two integrating-factor rescues) | very wide — with the rescues it subsumes every linear and most separable equations — but it answers with a *potential*, so it is tried after the classes that answer with `y(x)` |
//! | 6 | homogeneous | needs the `v = y/x` substitution and a third quadrature |
//! | 7 | Riccati | needs a particular solution, which is guessed; last because a wrong guess costs the most |
//!
//! Within that order there is a second rule: **an explicit `y(x)` is always
//! preferred over an implicit relation.**  The cascade does not stop at the
//! first class that answers; it stops at the first class that answers
//! *explicitly*, keeping the first implicit answer as a fallback.  Without that
//! rule, adding implicit output to the separable class would have silently
//! demoted equations that the linear or Bernoulli class solves for `y` —
//! a regression in the answer, with no error to notice it by.
//!
//! # Declining with a reason
//!
//! Every class reports [`Outcome::NoMatch`] (the form does not fit) separately
//! from [`Outcome::Declined`] (the form fits, but something the method needs
//! did not work out — nearly always an integral).  The cascade keeps those
//! reasons and puts them in the final error, so a caller whose separable ODE
//! failed on `∫ dy/h(y)` is told exactly that instead of "no class matched",
//! which would be a false statement about the equation.

use super::{
    contains, ddx, div, implicit_relation_is_zero, integrate_or_decline, is_zero, residual_is_zero,
    simp, sub, subs1, ConstGen, DsolveBranch, DsolveError, OdeInput,
};
use crate::kernel::eval_const::try_expr_f64;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

/// What one class made of the equation.
enum Outcome {
    /// A verified general solution.
    Solved(DsolveBranch),
    /// The equation is not of this class.  Carries no information.
    NoMatch,
    /// The equation *is* of this class, but the method did not close: an
    /// integral was not elementary, a candidate failed verification, a Riccati
    /// particular solution was not found.
    Declined(DsolveError),
}

pub(crate) fn solve(
    input: &OdeInput,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Vec<DsolveBranch>, DsolveError> {
    let yp = input.derivs[0];
    let mut cascade = Cascade::default();

    // Clairaut is nonlinear in y'; try it before the linear-in-y' reduction.
    let mark = gen.checkpoint();
    let clairaut = try_clairaut(input, gen, pool);
    if let Some(res) = cascade.offer("clairaut", clairaut, pool) {
        return Ok(res);
    }
    gen.rollback(mark, pool);

    // Solve for y': require equation linear in y'.
    let a = ddx(input.equation, yp, pool)?; // ∂F/∂y'
    if contains(a, yp, pool) || is_zero(a, pool) {
        return Err(DsolveError::Unsupported(
            "equation is not linear in y' (or independent of y')".to_string(),
        ));
    }
    let ayp = simp(pool.mul(vec![a, yp]), pool);
    let b = sub(input.equation, ayp, pool); // F − A·y'
    if contains(b, yp, pool) {
        return Err(DsolveError::Unsupported(
            "equation is not affine in y'".to_string(),
        ));
    }
    // y' = rhs(x, y) = −B/A
    let rhs = simp(div(pool.mul(vec![pool.integer(-1_i32), b]), a, pool), pool);

    type Attempt = fn(&OdeInput, ExprId, &mut ConstGen, &ExprPool) -> Outcome;
    let attempts: [(&'static str, Attempt); 6] = [
        ("separable", try_separable),
        ("linear", try_linear),
        ("bernoulli", try_bernoulli),
        ("exact", try_exact),
        ("homogeneous", try_homogeneous),
        ("riccati", try_riccati),
    ];
    for (class, attempt_fn) in attempts {
        // Release the constants of every class that did not end the cascade,
        // so the answer's constants are `C1, C2, …` whoever answers rather
        // than a record of how many classes were tried first.  Safe for the
        // implicit fallback the cascade may be holding: at most one solution is
        // ever returned, so two classes cannot both spend `C1` in one answer.
        let mark = gen.checkpoint();
        let outcome = attempt_fn(input, rhs, gen, pool);
        if let Some(res) = cascade.offer(class, outcome, pool) {
            return Ok(res);
        }
        gen.rollback(mark, pool);
    }
    cascade.finish()
}

// ---------------------------------------------------------------------------
// The cascade: prefer explicit, remember why the others declined
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Cascade {
    /// The first implicit answer, kept in case no class answers explicitly.
    implicit: Option<DsolveBranch>,
    /// `class: reason` for every class that recognised the equation and then
    /// could not finish.
    notes: Vec<(String, DsolveError)>,
}

/// Print one line per class under `ALKAHEST_DSOLVE_TRACE` in a test build.
///
/// The final error carries only the *declines*, so a class that reported
/// `NoMatch` — the usual reason an ODE that ought to solve does not — leaves no
/// trace in it at all.  This is how that is seen.
#[cfg(test)]
fn trace_class(class: &str, outcome: &Outcome, pool: &ExprPool) {
    if std::env::var_os("ALKAHEST_DSOLVE_TRACE").is_none() {
        return;
    }
    let what = match outcome {
        Outcome::Solved(s) if s.is_explicit() => format!("SOLVED {}", s.render(pool)),
        Outcome::Solved(s) => format!("SOLVED(implicit) {}", s.render(pool)),
        Outcome::NoMatch => "NO_MATCH".to_string(),
        Outcome::Declined(e) => format!("DECLINED {e}"),
    };
    eprintln!("CLASS\t{class}\t{what}");
}

#[cfg(not(test))]
fn trace_class(_class: &str, _outcome: &Outcome, _pool: &ExprPool) {}

impl Cascade {
    /// Record one class's outcome.  `Some` means the cascade is finished.
    fn offer(
        &mut self,
        class: &str,
        outcome: Outcome,
        pool: &ExprPool,
    ) -> Option<Vec<DsolveBranch>> {
        trace_class(class, &outcome, pool);
        match outcome {
            Outcome::Solved(sol) if sol.is_explicit() => Some(vec![sol]),
            Outcome::Solved(sol) => {
                if self.implicit.is_none() {
                    self.implicit = Some(sol);
                }
                None
            }
            Outcome::NoMatch => None,
            Outcome::Declined(e) => {
                self.notes.push((class.to_string(), e));
                None
            }
        }
    }

    /// No class answered explicitly: return the implicit answer if there is
    /// one, else the most informative decline the cascade collected.
    fn finish(self) -> Result<Vec<DsolveBranch>, DsolveError> {
        if let Some(sol) = self.implicit {
            return Ok(vec![sol]);
        }
        if self.notes.is_empty() {
            return Err(DsolveError::Unsupported(
                "no implemented first-order class matched".to_string(),
            ));
        }
        let detail = self
            .notes
            .iter()
            .map(|(class, e)| format!("{class}: {e}"))
            .collect::<Vec<_>>()
            .join("; ");
        // Report the kind of the most actionable decline.  A failed quadrature
        // is the most actionable (it names an integral the engine could grow to
        // handle); a missing particular solution is next; a candidate that
        // failed verification is a statement about the candidate, not the
        // equation, and comes last.
        let kind = |e: &DsolveError| {
            if e.is_quadrature_failure() {
                3
            } else if e.is_missing_particular_solution() {
                2
            } else if matches!(e, DsolveError::VerificationFailed(_)) {
                1
            } else {
                0
            }
        };
        let best = self.notes.iter().map(|(_, e)| kind(e)).max().unwrap_or(0);
        let msg = format!("no first-order class produced a verified solution ({detail})");
        Err(match best {
            3 => DsolveError::quadrature_failed(msg),
            2 => DsolveError::no_particular_solution(msg),
            _ => DsolveError::Unsupported(msg),
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Verify an explicit candidate and package it, or decline with the gate's
/// reason.  A verification failure is *not* fatal: the cascade goes on to the
/// next class, and only the collected reasons reach the caller.
fn finalize_explicit(
    input: &OdeInput,
    y_of_x: ExprId,
    constants: Vec<ExprId>,
    method: &'static str,
    pool: &ExprPool,
) -> Outcome {
    let y_of_x = simp(y_of_x, pool);
    match residual_is_zero(input, y_of_x, &constants, pool) {
        Ok(()) => Outcome::Solved(DsolveBranch::explicit(y_of_x, constants, method)),
        Err(e) => Outcome::Declined(e),
    }
}

/// Verify an implicit relation `G(x, y) = 0` and package it.
fn finalize_implicit(
    input: &OdeInput,
    relation: ExprId,
    constants: Vec<ExprId>,
    method: &'static str,
    pool: &ExprPool,
) -> Outcome {
    let relation = simp(relation, pool);
    match implicit_relation_is_zero(input, relation, &constants, pool) {
        Ok(()) => Outcome::Solved(DsolveBranch::implicit(relation, constants, method)),
        Err(e) => Outcome::Declined(e),
    }
}

/// Solve `relation(x, y) = 0` for `y` explicitly, then verify; fall back to the
/// implicit relation when no inversion is available or the explicit form does
/// not verify.
///
/// Both halves go through the gate, so "explicit when we can, implicit when we
/// cannot" never costs correctness: a Lambert-W inversion that the sampler
/// cannot confirm is dropped in favour of the relation it came from, which is
/// checked independently.
fn explicit_or_implicit(
    input: &OdeInput,
    relation: ExprId,
    constants: Vec<ExprId>,
    method: &'static str,
    pool: &ExprPool,
) -> Outcome {
    let c = *constants
        .first()
        .expect("a general solution carries a constant");
    if let Some(y_expr) = solve_relation_for_y(relation, input.x, input.y, c, pool) {
        if !contains(y_expr, input.y, pool) {
            match finalize_explicit(input, y_expr, constants.clone(), method, pool) {
                Outcome::Solved(sol) => return Outcome::Solved(sol),
                // The explicit form did not verify; fall through to the
                // relation, which is checked on its own terms.  Under
                // `ALKAHEST_DSOLVE_TRACE` say so — an inversion the gate
                // refuses is otherwise invisible.
                #[cfg(test)]
                Outcome::Declined(e) if std::env::var_os("ALKAHEST_DSOLVE_TRACE").is_some() => {
                    eprintln!(
                        "INVERSION_REFUSED\t{method}\ty = {}\t{e}",
                        pool.display(y_expr)
                    );
                }
                _ => {}
            }
        }
    }
    finalize_implicit(input, relation, constants, method, pool)
}

/// Map a failed quadrature to a decline that names the class *and* the
/// integrand that did not close — the difference between "your ODE is not
/// supported" and "your ODE is separable and `∫ dy/h(y)` is the missing piece".
fn quadrature_decline(
    class: &str,
    what: &str,
    integrand: ExprId,
    e: DsolveError,
    pool: &ExprPool,
) -> Outcome {
    Outcome::Declined(DsolveError::quadrature_failed(format!(
        "{class} needs {what} = ∫ `{}`: {e}",
        pool.display(integrand)
    )))
}

/// Strip one-element `Mul`/`Add` wrappers.
///
/// `simplify` leaves a `Mul([y])` standing (the `dsolve` module docs call this
/// out for exponents), and such a node is a *different* `ExprId` from the bare
/// `y` it wraps.  Structural pattern matching that compares against `y` — "is
/// this `log(y)`?" — silently fails on it, which is how `y' + kₑ·y = 0` came
/// back as the implicit `log y + kₑ·x = C` instead of `y = C·e^{−kₑx}`.
fn unwrap1(e: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(e) {
        ExprData::Mul(args) | ExprData::Add(args) if args.len() == 1 => unwrap1(args[0], pool),
        _ => e,
    }
}

/// Try to write `expr` as `g(x) · h(y)` (multiplicative split).  Returns
/// `(g_of_x, h_of_y)` when the factorisation is clean.
fn separable_split(
    expr: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    // Gather multiplicative factors.
    let factors: Vec<ExprId> = match pool.get(expr) {
        ExprData::Mul(args) => args,
        _ => vec![expr],
    };
    let mut gx: Vec<ExprId> = Vec::new();
    let mut hy: Vec<ExprId> = Vec::new();
    for f in factors {
        let has_x = contains(f, x, pool);
        let has_y = contains(f, y, pool);
        match (has_x, has_y) {
            (true, false) => gx.push(f),
            (false, true) => hy.push(f),
            (false, false) => gx.push(f), // constant → lump into g(x)
            (true, true) => return None,  // mixed factor → not separable this way
        }
    }
    let g = if gx.is_empty() {
        pool.integer(1_i32)
    } else {
        pool.mul(gx)
    };
    let h = if hy.is_empty() {
        pool.integer(1_i32)
    } else {
        pool.mul(hy)
    };
    Some((unwrap1(simp(g, pool), pool), unwrap1(simp(h, pool), pool)))
}

/// Split `expr` after putting it over a common denominator: `N/D` is separable
/// when `N` and `D` are separately separable, and the quotient of the parts is
/// the answer.
///
/// This is the spelling that reaches `y' = (1+y²)/(1+x²)`.  Neither the
/// expanded nor the plain normal form of that right-hand side is a product —
/// both are sums whose every term mentions `x` and `y` — but its numerator and
/// denominator are each a pure one-variable polynomial.
fn separable_split_quotient(
    expr: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let (numer, denom) = crate::poly::cancel::together_parts(expr, vec![x, y], pool).ok()?;
    let (gn, hn) = separable_split(numer, x, y, pool)?;
    let (gd, hd) = separable_split(denom, x, y, pool)?;
    if is_zero(gd, pool) || is_zero(hd, pool) {
        return None;
    }
    Some((div(gn, gd, pool), div(hn, hd, pool)))
}

// ---------------------------------------------------------------------------
// Separable: y' = g(x)·h(y)  →  ∫ dy/h(y) = ∫ g(x) dx + C
// ---------------------------------------------------------------------------

fn try_separable(input: &OdeInput, rhs: ExprId, gen: &mut ConstGen, pool: &ExprPool) -> Outcome {
    let (x, y) = (input.x, input.y);
    // Three spellings, because the split is structural and `simp` expands:
    // `(1 + y²)/(1 + x²)` is a clean product of `(1+y²)` and `(1+x²)⁻¹` until
    // distribution turns it into a sum whose every term mentions both
    // variables, at which point no multiplicative split can be seen.  The
    // non-expanding normal form sometimes keeps the product intact, and
    // failing that the numerator and denominator are split separately.
    let Some((g, h)) = separable_split(rhs, x, y, pool)
        .or_else(|| separable_split(super::simp_plain(rhs, pool), x, y, pool))
        .or_else(|| separable_split_quotient(rhs, x, y, pool))
    else {
        return Outcome::NoMatch;
    };
    // Need genuine y-dependence in h for "separable" to be meaningful & invertible.
    if !contains(h, y, pool) {
        return Outcome::NoMatch; // pure y' = g(x): handled by linear path (q only)
    }
    // ∫ 1/h(y) dy = ∫ g(x) dx + C.
    let inv_h = simp(pool.pow(h, pool.integer(-1_i32)), pool);
    let lhs_int = match integrate_or_decline(inv_h, y, pool) {
        Ok(v) => v, // H(y)
        Err(e) => return quadrature_decline("separable", "∫ dy/h(y)", inv_h, e, pool),
    };
    let rhs_int = match integrate_or_decline(g, x, pool) {
        Ok(v) => v, // G(x)
        Err(e) => return quadrature_decline("separable", "∫ g(x) dx", g, e, pool),
    };
    let c = gen.fresh(pool);

    // H(y) − G(x) − C = 0.
    let relation = sub(sub(lhs_int, rhs_int, pool), c, pool);
    explicit_or_implicit(input, relation, vec![c], "separable", pool)
}

/// Solve `relation(x, y) = 0` for `y`, returning `y = …` when one of the
/// recognised shapes applies.
fn solve_relation_for_y(
    relation: ExprId,
    x: ExprId,
    y: ExprId,
    c: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    // Polynomial of degree ≤ 2 in y, coefficients free in x: covers `y = G+C`
    // and `y² − x² − C = 0`.
    let zero = pool.integer(0_i32);
    if let Some(e) = solve_implicit_for_y(relation, x, y, zero, pool) {
        return Some(e);
    }
    // Otherwise split off the y-part and invert it: H(y) = target(x).
    let (h_of_y, target) = split_y_part(relation, x, y, pool)?;
    invert_simple(h_of_y, y, target, c, pool)
}

/// Split `relation` into `H(y) − target(x)`, i.e. return `(H, target)` with `H`
/// free of `x` and `target` free of `y`.  `None` when a term mentions both.
fn split_y_part(
    relation: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let terms: Vec<ExprId> = match pool.get(relation) {
        ExprData::Add(args) => args,
        _ => vec![relation],
    };
    let mut h: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for t in terms {
        match (contains(t, y, pool), contains(t, x, pool)) {
            (true, false) => h.push(t),
            (true, true) => return None,
            (false, _) => rest.push(t),
        }
    }
    if h.is_empty() {
        return None;
    }
    let h = unwrap1(simp(pool.add(h), pool), pool);
    // relation = H(y) + rest = 0  →  H(y) = −rest.
    let neg_rest = if rest.is_empty() {
        pool.integer(0_i32)
    } else {
        unwrap1(
            simp(pool.mul(vec![pool.integer(-1_i32), pool.add(rest)]), pool),
            pool,
        )
    };
    Some((h, neg_rest))
}

/// `target = G + c` with `G` free of `c`: return `G`.
///
/// Which half of the target is the arbitrary constant matters when the
/// inversion *exponentiates*.  `log y = G(x) + C` inverted literally gives
/// `y = e^{G+C}`, a family that is positive for every real `C` — it cannot
/// express `y(0) < 0`, and `y' + kₑy = 0` would come back unable to describe a
/// decay from a negative initial value.  Pulling the constant out first and
/// writing `y = C·e^{G}` restores the whole family (and is the textbook form).
fn split_const(target: ExprId, c: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let g = sub(target, c, pool);
    (!contains(g, c, pool)).then_some(g)
}

/// `C·e^{arg}` when the constant can be pulled out of `target`, else `e^{arg}`
/// with the constant left inside the exponent.
fn exp_with_constant(target: ExprId, k: ExprId, c: ExprId, pool: &ExprPool) -> ExprId {
    match split_const(target, c, pool) {
        Some(g) => simp(
            pool.mul(vec![c, pool.func("exp", vec![div(g, k, pool)])]),
            pool,
        ),
        None => simp(pool.func("exp", vec![div(target, k, pool)]), pool),
    }
}

/// Recognise `k·f(y)` (or a bare `f(y)`) for a single unary `f`, returning
/// `(k, f)`.
fn unary_of_y(lhs: ExprId, y: ExprId, pool: &ExprPool) -> Option<(ExprId, String)> {
    let is_f_of_y = |e: ExprId| match pool.get(e) {
        ExprData::Func { name, args } if args.len() == 1 && unwrap1(args[0], pool) == y => {
            Some(name)
        }
        _ => None,
    };
    if let Some(name) = is_f_of_y(lhs) {
        return Some((pool.integer(1_i32), name));
    }
    let ExprData::Mul(args) = pool.get(lhs) else {
        return None;
    };
    let mut found: Option<String> = None;
    let mut k_factors = Vec::new();
    for a in &args {
        match is_f_of_y(*a) {
            Some(name) if found.is_none() => found = Some(name),
            Some(_) => return None, // f(y)·g(y): not this shape
            None => k_factors.push(*a),
        }
    }
    let name = found?;
    let k = if k_factors.is_empty() {
        pool.integer(1_i32)
    } else {
        unwrap1(simp(pool.mul(k_factors), pool), pool)
    };
    Some((k, name))
}

/// The inverse of the unary functions a separable quadrature actually produces.
///
/// `∫ dy/(1+y²) = atan y`, `∫ dy/(1−y²) = atanh y`, `∫ dy/√(1−y²) = asin y` —
/// each of which then has to be undone to state `y` explicitly.  `y' = 1 + y²`
/// is the standard example, and without `atan` it came back as the relation
/// `atan y − x = C` where `y = tan(x + C)` was available.
///
/// The inverses are principal-branch, which is why the result still has to
/// clear the verification gate; nothing here is trusted on the strength of the
/// table alone.
fn unary_inverse(name: &str) -> Option<&'static str> {
    Some(match name {
        "atan" => "tan",
        "atanh" => "tanh",
        "asin" => "sin",
        "acos" => "cos",
        _ => return None,
    })
}

/// Invert `lhs(y) = target` for a few common closed forms, returning `y = …`.
///
/// `c` is the integration constant carried inside `target`; the exponentiating
/// cases move it out in front (see [`split_const`]).
fn invert_simple(
    lhs: ExprId,
    y: ExprId,
    target: ExprId,
    c: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let lhs = unwrap1(simp(lhs, pool), pool);
    // Case H(y) = y  → y = target
    if lhs == y {
        return Some(target);
    }
    // Case H(y) = c0 * y  (linear in y) → y = target / c0
    let dl = super::ddx(lhs, y, pool).ok()?;
    if !contains(dl, y, pool) && !is_zero(dl, pool) {
        // lhs = dl*y + const; recover constant
        let lin = simp(pool.mul(vec![dl, y]), pool);
        let cst = sub(lhs, lin, pool);
        if !contains(cst, y, pool) {
            // dl*y + cst = target → y = (target - cst)/dl
            return Some(div(sub(target, cst, pool), dl, pool));
        }
    }
    // Case H(y) = k·f(y) for an invertible unary `f` → y = f⁻¹(target/k).
    if let Some((k, name)) = unary_of_y(lhs, y, pool) {
        if !contains(k, y, pool) && !is_zero(k, pool) {
            if name == "log" {
                // The one case whose inverse is an exponential, so the
                // constant comes out in front rather than staying additive.
                return Some(exp_with_constant(target, k, c, pool));
            }
            if let Some(inverse) = unary_inverse(&name) {
                let arg = div(target, k, pool);
                return Some(simp(pool.func(inverse, vec![arg]), pool));
            }
        }
    }
    // Case H(y) = Σ kᵢ·log(aᵢy + bᵢ): exponentiate to Π (aᵢy+bᵢ)^{kᵢ} = e^T and
    // solve when the result is linear in y (covers the logistic and similar).
    if let Some(y_expr) = invert_log_sum(lhs, y, target, c, pool) {
        return Some(y_expr);
    }
    // Case H(y) = k·log(y) + m·y: Lambert W (Michaelis–Menten elimination).
    if let Some(y_expr) = invert_log_plus_linear(lhs, y, target, pool) {
        return Some(y_expr);
    }
    None
}

/// One additive term of a candidate `Σ kᵢ·log(argᵢ)`: the coefficient and the
/// logarithm's argument.  `None` when the term is not of that shape.
fn log_term(term: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    match pool.get(term) {
        ExprData::Func { name, args } if name == "log" && args.len() == 1 => {
            Some((pool.integer(1_i32), args[0]))
        }
        ExprData::Mul(factors) => {
            let mut logarg = None;
            let mut coeff: Vec<ExprId> = Vec::new();
            for f in &factors {
                match pool.get(*f) {
                    ExprData::Func { name, args } if name == "log" && args.len() == 1 => {
                        if logarg.is_some() {
                            return None; // a product of two logs
                        }
                        logarg = Some(args[0]);
                    }
                    _ => coeff.push(*f),
                }
            }
            let arg = logarg?;
            // `simp` leaves a one-element `Mul` standing and `Mul([−1])` is not
            // the integer `−1`, which defeats the `kᵢ/k₀ = ±1` test below.
            let k = if coeff.is_empty() {
                pool.integer(1_i32)
            } else {
                unwrap1(simp(pool.mul(coeff), pool), pool)
            };
            Some((k, arg))
        }
        _ => None,
    }
}

/// Invert `Σ kᵢ·log(aᵢ·y + bᵢ) = target` when every `kᵢ/k₀ = ±1`, by
/// exponentiating to `Π (aᵢy+bᵢ)^{±1} = e^{target/k₀}` and solving the result
/// when it is linear in `y`.
///
/// The common factor `k₀` is not required to be numeric.  A logistic written
/// with symbolic parameters, `y' = r·y·(1 − y/K)`, integrates to
/// `(log y − log(y−K))/r`, whose coefficients are `±1/r`: insisting on literal
/// `±1` — as this did — solved the textbook `r = K = 1` case and nothing else.
fn invert_log_sum(
    lhs: ExprId,
    y: ExprId,
    target: ExprId,
    c: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let terms: Vec<ExprId> = match pool.get(lhs) {
        ExprData::Add(args) => args,
        _ => vec![lhs],
    };
    let parsed: Vec<(ExprId, ExprId)> = terms
        .iter()
        .map(|&t| log_term(t, pool))
        .collect::<Option<_>>()?;
    let (k0, _) = *parsed.first()?;
    if contains(k0, y, pool) || is_zero(k0, pool) {
        return None;
    }

    let mut numer_args: Vec<ExprId> = Vec::new(); // exponent +1
    let mut denom_args: Vec<ExprId> = Vec::new(); // exponent −1
    for (k, arg) in parsed {
        if contains(k, y, pool) {
            return None;
        }
        // arg must be linear (degree ≤ 1) in y.
        let darg = super::ddx(arg, y, pool).ok()?;
        if contains(darg, y, pool) || is_zero(darg, pool) {
            return None;
        }
        match try_expr_f64(unwrap1(div(k, k0, pool), pool), pool) {
            Some(r) if (r - 1.0).abs() < 1e-12 => numer_args.push(arg),
            Some(r) if (r + 1.0).abs() < 1e-12 => denom_args.push(arg),
            _ => return None,
        }
    }
    // Π numer / Π denom = C·e^{G/k₀}  →  Π numer − C·e^{…}·Π denom = 0.
    let et = exp_with_constant(target, k0, c, pool);
    // `unwrap1`: a single `Mul([y])` factor here is what made the textbook
    // logistic fall through to the implicit branch — `sub` would not cancel it
    // against the bare `y` the linear solve produces.
    let num = if numer_args.is_empty() {
        pool.integer(1_i32)
    } else {
        unwrap1(simp(pool.mul(numer_args), pool), pool)
    };
    let den = if denom_args.is_empty() {
        pool.integer(1_i32)
    } else {
        unwrap1(simp(pool.mul(denom_args), pool), pool)
    };
    // equation E(y) = num − e^T·den = 0
    let e_y = sub(num, simp(pool.mul(vec![et, den]), pool), pool);
    solve_linear_in_y(e_y, y, pool)
}

/// Solve `expr = 0` for `y` when `expr` is affine in `y`.
fn solve_linear_in_y(expr: ExprId, y: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let b = super::ddx(expr, y, pool).ok()?;
    if contains(b, y, pool) || is_zero(b, pool) {
        return None;
    }
    let by = simp(pool.mul(vec![b, y]), pool);
    let a = sub(expr, by, pool);
    if contains(a, y, pool) {
        return None;
    }
    Some(div(
        simp(pool.mul(vec![pool.integer(-1_i32), a]), pool),
        b,
        pool,
    ))
}

/// Invert `k·log(y) + m·y = target` through the Lambert W function:
///
/// ```text
///   k·log y + m·y = T   ⟺   (m/k)·y · e^{(m/k)·y} = (m/k)·e^{T/k}
///                       ⟺   y = (k/m)·W((m/k)·e^{T/k}).
/// ```
///
/// This is the Michaelis–Menten shape: `(Kₘ + y)·y' + Vₘ·y = 0` separates to
/// `Kₘ·log y + y = C − Vₘ·x`, giving `y = Kₘ·W(e^{(C − Vₘ·x)/Kₘ}/Kₘ)`.
///
/// The explicit form is offered, not forced: `W` is a principal-branch
/// function and the identity above needs `(m/k)·e^{T/k} ≥ −1/e`, so the answer
/// still has to clear the verification gate — and when it does not, the caller
/// falls back to the implicit relation, which is checked separately and is
/// just as valid an answer.
fn invert_log_plus_linear(
    lhs: ExprId,
    y: ExprId,
    target: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let terms: Vec<ExprId> = match pool.get(lhs) {
        ExprData::Add(args) => args,
        _ => return None,
    };
    let mut k: Option<ExprId> = None; // coefficient of log(y)
    let mut m: Option<ExprId> = None; // coefficient of y
    for t in terms {
        if let Some((coeff, arg)) = log_term(t, pool) {
            if unwrap1(arg, pool) != y || k.is_some() || contains(coeff, y, pool) {
                return None;
            }
            k = Some(coeff);
            continue;
        }
        // m·y: linear with a y-free coefficient.
        let d = super::ddx(t, y, pool).ok()?;
        if contains(d, y, pool) || is_zero(d, pool) || m.is_some() {
            return None;
        }
        if !is_zero(sub(t, simp(pool.mul(vec![d, y]), pool), pool), pool) {
            return None; // t is not exactly d·y
        }
        m = Some(d);
    }
    let (k, m) = (k?, m?);
    if is_zero(k, pool) || is_zero(m, pool) {
        return None;
    }
    let m_over_k = div(m, k, pool);
    let arg = simp(
        pool.mul(vec![
            m_over_k,
            simp(pool.func("exp", vec![div(target, k, pool)]), pool),
        ]),
        pool,
    );
    let w = pool.func("lambert_w", vec![arg]);
    Some(simp(pool.mul(vec![div(k, m, pool), w]), pool))
}

// ---------------------------------------------------------------------------
// Linear: y' = q(x) − p(x)·y  →  y = e^{−∫p}(∫ e^{∫p} q dx + C)
// ---------------------------------------------------------------------------

/// Decompose `rhs` (= y') as `q(x) + r(x)·y` when affine in `y`.  Returns
/// `(p, q)` for the standard form `y' + p·y = q`, i.e. `p = −r`.
fn linear_split(rhs: ExprId, _x: ExprId, y: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let dr = super::ddx(rhs, y, pool).ok()?; // ∂rhs/∂y = r(x)
    if contains(dr, y, pool) {
        return None; // not affine in y
    }
    let ry = simp(pool.mul(vec![dr, y]), pool);
    let q = sub(rhs, ry, pool); // rhs − r·y = q(x)
    if contains(q, y, pool) {
        return None;
    }
    // p = −r (standard form y' + p·y = q)
    let p = simp(pool.mul(vec![pool.integer(-1_i32), dr]), pool);
    Some((p, q))
}

fn try_linear(input: &OdeInput, rhs: ExprId, gen: &mut ConstGen, pool: &ExprPool) -> Outcome {
    let (x, y) = (input.x, input.y);
    let Some((p, q)) = linear_split(rhs, x, y, pool) else {
        return Outcome::NoMatch;
    };
    // Integrating factor μ = e^{∫p dx}.  `p` and `q` may be any x-expressions,
    // symbolic parameters included: nothing here assumes a numeric coefficient.
    let int_p = match integrate_or_decline(p, x, pool) {
        Ok(v) => v,
        Err(e) => return quadrature_decline("linear", "∫ p(x) dx", p, e, pool),
    };
    let mu = super::exp_of(int_p, pool);
    // y = (∫ μ q dx + C) / μ
    let muq = simp(pool.mul(vec![mu, q]), pool);
    let int_muq = match integrate_or_decline(muq, x, pool) {
        Ok(v) => v,
        Err(e) => return quadrature_decline("linear", "∫ μ(x)·q(x) dx", muq, e, pool),
    };
    let c = gen.fresh(pool);
    let numer = simp(pool.add(vec![int_muq, c]), pool);
    let y_expr = div(numer, mu, pool);
    finalize_explicit(input, y_expr, vec![c], "linear", pool)
}

// ---------------------------------------------------------------------------
// Bernoulli: y' + p y = q y^n  (n ≠ 0,1)  →  v = y^{1−n} linearises
// ---------------------------------------------------------------------------

fn try_bernoulli(input: &OdeInput, rhs: ExprId, gen: &mut ConstGen, pool: &ExprPool) -> Outcome {
    let (x, y) = (input.x, input.y);
    // rhs = −p·y + q·y^n.  Try to detect: split additive terms by power of y.
    let terms: Vec<ExprId> = match pool.get(rhs) {
        ExprData::Add(args) => args,
        _ => vec![rhs],
    };
    // For each additive term, find its power of y (must be a clean monomial in y).
    let mut linear_coeff: Vec<ExprId> = Vec::new(); // coeff of y^1
    let mut bern_coeff: Vec<ExprId> = Vec::new(); // coeff of y^n
    let mut n_exp: Option<i64> = None;
    for t in terms {
        let Some((coeff, pw)) = monomial_in_y(t, y, pool) else {
            return Outcome::NoMatch;
        };
        if contains(coeff, y, pool) {
            return Outcome::NoMatch;
        }
        match pw {
            1 => linear_coeff.push(coeff),
            other => {
                if other == 0 {
                    // constant term → would be Bernoulli with q·y^0; treat n=0 not supported here
                    return Outcome::NoMatch;
                }
                match n_exp {
                    None => n_exp = Some(other),
                    Some(e) if e == other => {}
                    Some(_) => return Outcome::NoMatch, // two different nonlinear powers
                }
                bern_coeff.push(coeff);
            }
        }
    }
    let Some(n) = n_exp else {
        return Outcome::NoMatch;
    };
    if n == 1 {
        return Outcome::NoMatch;
    }
    let p = simp(
        pool.mul(vec![pool.integer(-1_i32), pool.add(linear_coeff)]),
        pool,
    ); // y' + p y, p = −(coeff of y)
    let q = simp(pool.add(bern_coeff), pool); // q·y^n
    if contains(q, y, pool) || contains(p, y, pool) {
        return Outcome::NoMatch;
    }

    // v = y^{1−n}.  v' + (1−n) p v = (1−n) q.  Solve linear in v then y = v^{1/(1−n)}.
    let one_minus_n = pool.integer((1 - n) as i32);
    let pv = simp(pool.mul(vec![one_minus_n, p]), pool);
    let qv = simp(pool.mul(vec![one_minus_n, q]), pool);
    // integrating factor μ = e^{∫ pv}
    let int_pv = match integrate_or_decline(pv, x, pool) {
        Ok(v) => v,
        Err(e) => return quadrature_decline("bernoulli", "∫ (1−n)·p(x) dx", pv, e, pool),
    };
    let mu = super::exp_of(int_pv, pool);
    let muq = simp(pool.mul(vec![mu, qv]), pool);
    let int_muq = match integrate_or_decline(muq, x, pool) {
        Ok(v) => v,
        Err(e) => return quadrature_decline("bernoulli", "∫ μ(x)·(1−n)·q(x) dx", muq, e, pool),
    };
    let c = gen.fresh(pool);
    let v = div(simp(pool.add(vec![int_muq, c]), pool), mu, pool);
    // y = v^{1/(1−n)}
    let exp = pool.rational(1_i32, (1 - n) as i32);
    let y_expr = simp(pool.pow(v, exp), pool);
    finalize_explicit(input, y_expr, vec![c], "bernoulli", pool)
}

/// If `term` is `coeff · y^k` for integer `k ≥ 0` (with `coeff` free of `y`),
/// return `(coeff, k)`.  Returns `None` when `term` is not a clean integer-power
/// monomial in `y` (the caller treats that as "not this class").
fn monomial_in_y(term: ExprId, y: ExprId, pool: &ExprPool) -> Option<(ExprId, i64)> {
    // Decompose into factors, find the single y-power factor.
    let factors: Vec<ExprId> = match pool.get(term) {
        ExprData::Mul(args) => args,
        _ => vec![term],
    };
    let mut coeff: Vec<ExprId> = Vec::new();
    let mut power: i64 = 0;
    let mut found_y = false;
    for f in factors {
        if f == y {
            power += 1;
            found_y = true;
            continue;
        }
        if let ExprData::Pow { base, exp } = pool.get(f) {
            if base == y {
                if let ExprData::Integer(k) = pool.get(exp) {
                    power += k.0.to_i64()?;
                    found_y = true;
                    continue;
                }
                // non-integer power of y → not a clean monomial
                return None;
            }
        }
        if contains(f, y, pool) {
            // y appears inside a function/sub-expr → not a monomial
            return None;
        }
        coeff.push(f);
    }
    let c = if coeff.is_empty() {
        pool.integer(1_i32)
    } else {
        pool.mul(coeff)
    };
    Some((simp(c, pool), if found_y { power } else { 0 }))
}

// ---------------------------------------------------------------------------
// Homogeneous of degree zero: y' = G(y/x).  Substitute v = y/x.
// ---------------------------------------------------------------------------

/// Does `rhs` satisfy `rhs(t·x, t·y) = rhs(x, y)` at sample points?
///
/// The *decision* this feeds is only a classification: a wrong "yes" costs a
/// few integrals and produces a candidate that the verification gate then
/// refuses, so sampling is an acceptable instrument here in a way it would not
/// be for the answer itself.  A wrong "no" costs an ODE, which is what the
/// symbolic test was already doing on every quotient with a sum in the
/// denominator.
///
/// Inconclusive (nothing evaluated — symbolic parameters, an unknown head) is
/// reported as "no", keeping the previous behaviour for those.
fn samples_degree_zero(rhs: ExprId, x: ExprId, y: ExprId, pool: &ExprPool) -> bool {
    use std::collections::HashMap;
    const POINTS: [(f64, f64); 4] = [(1.3, 0.7), (0.9, 2.1), (2.4, -1.1), (0.6, 1.7)];
    const SCALES: [f64; 2] = [2.0, 0.45];
    let mut agreed = 0usize;
    for (xv, yv) in POINTS {
        let mut env: HashMap<ExprId, f64> = HashMap::new();
        env.insert(x, xv);
        env.insert(y, yv);
        let Some(base) = super::verify::eval(rhs, &env, pool).filter(|v| v.is_finite()) else {
            continue;
        };
        for t in SCALES {
            let mut scaled: HashMap<ExprId, f64> = HashMap::new();
            scaled.insert(x, t * xv);
            scaled.insert(y, t * yv);
            match super::verify::eval(rhs, &scaled, pool) {
                Some(v) if v.is_finite() => {
                    if (v - base).abs() > 1e-9 * (1.0 + base.abs()) {
                        return false;
                    }
                    agreed += 1;
                }
                _ => continue,
            }
        }
    }
    agreed >= 4
}

fn try_homogeneous(input: &OdeInput, rhs: ExprId, gen: &mut ConstGen, pool: &ExprPool) -> Outcome {
    let (x, y) = (input.x, input.y);
    if !contains(rhs, x, pool) || !contains(rhs, y, pool) {
        return Outcome::NoMatch;
    }
    // Replace y → v·x and check the result is free of x (degree-0 homogeneous).
    let v = pool.symbol("__v_hom", Domain::Real);
    let vx = simp(pool.mul(vec![v, x]), pool);
    let mut g_vx = subs1(rhs, y, vx, pool);
    if contains(g_vx, x, pool) {
        // `simp` will not cancel `x²·(x²·v)⁻¹`: the `Pow` wraps a whole `Mul`,
        // so the `x²` inside never meets the one outside (the same defect
        // `super::normalized` exists for).  Without this second look, every
        // homogeneous equation written as a single quotient — `(x²+y²)/(xy)` —
        // failed its own degree-zero test.
        g_vx = super::normalized(g_vx, pool);
    }
    if contains(g_vx, x, pool) {
        // Still not visibly free of `x`.  `(x+y)/(x−y)` is the shape that
        // defeats both attempts: the reciprocal wraps a *sum*, so no amount of
        // distributing cancels the `x`, and the class used to decline an
        // equation from its own textbook chapter.  Ask the defining property
        // numerically instead, and read `G(v)` off `rhs(1, v)`.
        if !samples_degree_zero(rhs, x, y, pool) {
            return Outcome::NoMatch;
        }
        g_vx = simp(subs1(g_vx, x, pool.integer(1_i32), pool), pool);
        if contains(g_vx, x, pool) {
            return Outcome::NoMatch;
        }
    }
    // Now y = v x, y' = v + x v'.  v + x v' = G(v)  →  x v' = G(v) − v.
    // Separable in v: ∫ dv/(G(v) − v) = ∫ dx/x = log|x| + C.
    let gmv = sub(g_vx, v, pool);
    if is_zero(gmv, pool) {
        return Outcome::NoMatch; // y' = y/x → degenerate (linear), let linear handle it
    }
    let inv = simp(pool.pow(gmv, pool.integer(-1_i32)), pool);
    let lhs_int = match integrate_or_decline(inv, v, pool) {
        Ok(val) => val, // Φ(v)
        Err(e) => return quadrature_decline("homogeneous", "∫ dv/(G(v)−v)", inv, e, pool),
    };
    let c = gen.fresh(pool);
    let logx = pool.func("log", vec![x]);
    let target = simp(pool.add(vec![logx, c]), pool);
    // Solve Φ(v) − log x − C = 0 for v, then y = v x.  The relation is solved
    // by the same routine the separable and exact classes use, so a `Φ`
    // quadratic in `v` (`y' = (x²+y²)/(xy)`, `Φ = v²/2`) comes back explicit
    // rather than as a relation in `y/x`.
    let relation_v = sub(lhs_int, target, pool);
    if let Some(v_expr) = solve_relation_for_y(relation_v, x, v, c, pool) {
        let y_expr = simp(pool.mul(vec![v_expr, x]), pool);
        // back-substitute v occurrences (none should remain)
        if !contains(y_expr, v, pool) {
            if let Outcome::Solved(sol) =
                finalize_explicit(input, y_expr, vec![c], "homogeneous", pool)
            {
                return Outcome::Solved(sol);
            }
        }
    }
    // Implicit form: Φ(y/x) − log x − C = 0.
    let relation = sub(subs1(lhs_int, v, div(y, x, pool), pool), target, pool);
    if contains(relation, v, pool) {
        return Outcome::NoMatch;
    }
    finalize_implicit(input, relation, vec![c], "homogeneous", pool)
}

// ---------------------------------------------------------------------------
// Exact: M dx + N dy = 0 with ∂M/∂y = ∂N/∂x.  y' = −M/N.
// Solution F(x,y)=C with ∂F/∂x = M, ∂F/∂y = N.
// ---------------------------------------------------------------------------

fn try_exact(input: &OdeInput, _rhs: ExprId, gen: &mut ConstGen, pool: &ExprPool) -> Outcome {
    let (x, y, yp) = (input.x, input.y, input.derivs[0]);
    // Recover the natural M dx + N dy = 0 split from the *original* equation
    // (`equation = M + N·y'`):  N = ∂equation/∂y', M = equation − N·y'.
    let n = match ddx(input.equation, yp, pool) {
        Ok(v) => v,
        Err(e) => return Outcome::Declined(e),
    };
    if contains(n, yp, pool) || is_zero(n, pool) {
        return Outcome::NoMatch;
    }
    let m = sub(input.equation, simp(pool.mul(vec![n, yp]), pool), pool);
    if contains(m, yp, pool) {
        return Outcome::NoMatch;
    }

    let mut declines: Vec<DsolveError> = Vec::new();
    for (m, n, method) in integrating_factor_candidates(m, n, x, y, pool) {
        match exact_potential(m, n, x, y, pool) {
            Ok(Some(f)) => {
                // One constant per candidate pair, released again when the pair
                // does not answer, so a rescue that fails does not push the
                // answer's constant to `C2`.
                let mark = gen.checkpoint();
                let c = gen.fresh(pool);
                let relation = sub(f, c, pool);
                match explicit_or_implicit(input, relation, vec![c], method, pool) {
                    Outcome::Solved(sol) => return Outcome::Solved(sol),
                    Outcome::Declined(e) => {
                        gen.rollback(mark, pool);
                        declines.push(e);
                    }
                    Outcome::NoMatch => gen.rollback(mark, pool),
                }
            }
            Ok(None) => {}
            Err(e) => declines.push(e),
        }
    }
    match declines.into_iter().next() {
        Some(e) => Outcome::Declined(e),
        None => Outcome::NoMatch,
    }
}

/// Put a quotient over a common denominator and divide out the polynomial GCD.
///
/// The two integrating-factor tests are *"is this ratio free of `y`"* / *"free
/// of `x`"*, and the ratio arrives unreduced: `(y − x)/(x² − xy)` is `−1/x`,
/// but only after a GCD that needs polynomial arithmetic rather than a rewrite
/// rule.  Falls back to the argument unchanged when the expression is not a
/// rational function of `(x, y)` — nothing here depends on the reduction
/// succeeding, and the pair it produces is re-tested for exactness anyway.
fn ratio_normal(e: ExprId, x: ExprId, y: ExprId, pool: &ExprPool) -> ExprId {
    let normal = super::normalized(e, pool);
    match crate::poly::cancel::cancel(normal, vec![x, y], pool) {
        Ok(v) => simp(v, pool),
        Err(_) => normal,
    }
}

/// The `(M, N)` pairs worth testing for exactness, in order: the equation as
/// given, then the two textbook integrating-factor rescues.
///
/// A first-order equation is *near*-exact far more often than exact — writing
/// `M dx + N dy = 0` from `y' = f(x, y)` fixes `M` and `N` only up to a common
/// factor, and the arbitrary choice is almost never the exact one.  The two
/// rescues cover the cases where the correcting factor depends on one variable:
///
/// ```text
///   (M_y − N_x)/N  free of y  →  μ(x) = exp(∫ (M_y − N_x)/N dx)
///   (N_x − M_y)/M  free of x  →  μ(y) = exp(∫ (N_x − M_y)/M dy)
/// ```
///
/// Both are only *offered*: the pair is re-tested for exactness after
/// multiplication, and the resulting potential still has to pass the gate.
fn integrating_factor_candidates(
    m: ExprId,
    n: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Vec<(ExprId, ExprId, &'static str)> {
    let (Ok(dmy), Ok(dnx)) = (ddx(m, y, pool), ddx(n, x, pool)) else {
        return Vec::new();
    };
    let diff = sub(dmy, dnx, pool);
    if is_zero(diff, pool) {
        return vec![(m, n, "exact")];
    }
    let mut out = Vec::new();
    // μ(x) = exp(∫ (M_y − N_x)/N dx).
    //
    // `super::normalized` is what makes the test usable: the ratio arrives as
    // `y·(x·y)⁻¹`, whose reciprocal wraps a whole product, so `simp` leaves the
    // `y` standing on both sides and the "free of y" question answers itself
    // wrongly.  `(x² + y² + x) + x·y·y' = 0` — μ = x, a first-week exercise —
    // was declined for exactly that reason.
    let fx = ratio_normal(div(diff, n, pool), x, y, pool);
    if !contains(fx, y, pool) {
        if let Ok(i) = integrate_or_decline(fx, x, pool) {
            let mu = super::exp_of(i, pool);
            out.push((
                simp(pool.mul(vec![mu, m]), pool),
                simp(pool.mul(vec![mu, n]), pool),
                "exact_integrating_factor_x",
            ));
        }
    }
    // μ(y) = exp(∫ (N_x − M_y)/M dy)
    let gy = ratio_normal(
        div(
            simp(pool.mul(vec![pool.integer(-1_i32), diff]), pool),
            m,
            pool,
        ),
        x,
        y,
        pool,
    );
    if !contains(gy, x, pool) {
        if let Ok(i) = integrate_or_decline(gy, y, pool) {
            let mu = super::exp_of(i, pool);
            out.push((
                simp(pool.mul(vec![mu, m]), pool),
                simp(pool.mul(vec![mu, n]), pool),
                "exact_integrating_factor_y",
            ));
        }
    }
    out
}

/// Build the potential `F` with `∂F/∂x = M`, `∂F/∂y = N`.
///
/// `Ok(None)` means the pair is not exact (or the split did not close), which
/// is not an error: the caller simply moves on to the next candidate pair.
fn exact_potential(
    m: ExprId,
    n: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, DsolveError> {
    let dmy = ddx(m, y, pool)?;
    let dnx = ddx(n, x, pool)?;
    if !is_zero(sub(dmy, dnx, pool), pool) {
        return Ok(None);
    }
    // F = ∫ M dx + g(y), with ∂F/∂y = N → g'(y) = N − ∂/∂y ∫M dx.
    let int_m = integrate_or_decline(m, x, pool).map_err(|e| {
        DsolveError::quadrature_failed(format!("exact needs ∫ M dx of `{}`: {e}", pool.display(m)))
    })?;
    let dint_m_dy = ddx(int_m, y, pool)?;
    let gy_prime = sub(n, dint_m_dy, pool); // should be free of x
    if contains(gy_prime, x, pool) {
        return Ok(None);
    }
    let g_of_y = integrate_or_decline(gy_prime, y, pool).map_err(|e| {
        DsolveError::quadrature_failed(format!(
            "exact needs ∫ (N − ∂ₓ⁻¹) dy of `{}`: {e}",
            pool.display(gy_prime)
        ))
    })?;
    Ok(Some(simp(pool.add(vec![int_m, g_of_y]), pool)))
}

/// Solve `F(x,y) = C` for `y` when `F` is polynomial of degree ≤ 2 in `y`.
/// Affine `F = b·y + a` gives `y = (C − a)/b`; quadratic `F = A·y² + B·y + D`
/// gives one branch of the quadratic formula with `D` shifted by `−C`.
fn solve_implicit_for_y(
    f: ExprId,
    _x: ExprId,
    y: ExprId,
    c: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let b = super::ddx(f, y, pool).ok()?; // ∂F/∂y
    if !contains(b, y, pool) {
        // Affine case: F = b·y + a.
        if is_zero(b, pool) {
            return None;
        }
        let by = simp(pool.mul(vec![b, y]), pool);
        let a = sub(f, by, pool);
        if contains(a, y, pool) {
            return None;
        }
        return Some(div(sub(c, a, pool), b, pool));
    }
    // Quadratic case: ∂F/∂y = 2A·y + B  ⇒  A = ½·∂²F/∂y², B = ∂F/∂y|_{coeff}.
    let d2 = super::ddx(b, y, pool).ok()?; // ∂²F/∂y² = 2A
    if contains(d2, y, pool) || is_zero(d2, pool) {
        return None; // degree > 2 or not quadratic
    }
    let a_coeff = simp(pool.mul(vec![pool.rational(1_i32, 2_i32), d2]), pool); // A
                                                                               // B = (∂F/∂y) − 2A·y
    let b_coeff = sub(b, simp(pool.mul(vec![d2, y]), pool), pool);
    if contains(b_coeff, y, pool) {
        return None;
    }
    // D = F − A·y² − B·y
    let ay2 = simp(
        pool.mul(vec![a_coeff, pool.pow(y, pool.integer(2_i32))]),
        pool,
    );
    let by = simp(pool.mul(vec![b_coeff, y]), pool);
    let d_coeff = sub(sub(f, ay2, pool), by, pool); // free of y
    if contains(d_coeff, y, pool) {
        return None;
    }
    // A y² + B y + (D − C) = 0 → y = (−B + sqrt(B² − 4A(D−C)))/(2A)
    let dc = sub(d_coeff, c, pool);
    let disc = sub(
        simp(pool.pow(b_coeff, pool.integer(2_i32)), pool),
        simp(pool.mul(vec![pool.integer(4_i32), a_coeff, dc]), pool),
        pool,
    );
    let sqrt_disc = simp(pool.pow(disc, pool.rational(1_i32, 2_i32)), pool);
    let numer = simp(
        pool.add(vec![
            pool.mul(vec![pool.integer(-1_i32), b_coeff]),
            sqrt_disc,
        ]),
        pool,
    );
    let denom = simp(pool.mul(vec![pool.integer(2_i32), a_coeff]), pool);
    Some(div(numer, denom, pool))
}

// ---------------------------------------------------------------------------
// Riccati: y' = q0 + q1 y + q2 y², with a polynomial particular solution y_p.
// Then y = y_p + 1/v reduces to a linear ODE in v.  We decline if no
// low-degree polynomial particular solution exists.
// ---------------------------------------------------------------------------

fn try_riccati(input: &OdeInput, rhs: ExprId, gen: &mut ConstGen, pool: &ExprPool) -> Outcome {
    let (x, y) = (input.x, input.y);
    // rhs must be quadratic in y: rhs = q0 + q1 y + q2 y², q2 ≠ 0, qi free of y.
    let Some((_q0, q1, q2)) = quadratic_in_y(rhs, y, pool) else {
        return Outcome::NoMatch;
    };
    if is_zero(q2, pool) {
        return Outcome::NoMatch; // linear, not Riccati
    }
    // Find a polynomial particular solution y_p = a0 + a1 x + a2 x² (degree ≤ 2).
    let Some(yp_part) = find_poly_particular(rhs, x, y, pool) else {
        // Recognised as Riccati, and *not* solvable from here.  The general
        // Riccati reduces to a second-order linear ODE (`y' = y² + x` to the
        // Airy equation), whose solutions are outside the vocabulary this
        // solver emits — so the honest answer is a named refusal, not a
        // "no class matched" that misdescribes the equation, and not an
        // attempt that could only produce something unverifiable.
        return Outcome::Declined(DsolveError::no_particular_solution(format!(
            "riccati `y' = {}` has no polynomial particular solution of degree ≤ 2; \
             the general solution is in terms of solutions of the associated \
             second-order linear equation, which dsolve does not emit",
            pool.display(rhs)
        )));
    };
    // y = y_p + 1/v.  v satisfies v' = −(q1 + 2 q2 y_p) v − q2  (linear).
    let two_q2_yp = simp(pool.mul(vec![pool.integer(2_i32), q2, yp_part]), pool);
    let p = simp(pool.add(vec![q1, two_q2_yp]), pool); // v' + p v = −q2
    let qrhs = simp(pool.mul(vec![pool.integer(-1_i32), q2]), pool);
    // linear in v: v' + p v = qrhs (here standard form v' + P v = Q with P=p, Q=qrhs)
    let int_p = match integrate_or_decline(p, x, pool) {
        Ok(v) => v,
        Err(e) => return quadrature_decline("riccati", "∫ (q₁ + 2q₂y_p) dx", p, e, pool),
    };
    let mu = super::exp_of(int_p, pool);
    let muq = simp(pool.mul(vec![mu, qrhs]), pool);
    let int_muq = match integrate_or_decline(muq, x, pool) {
        Ok(v) => v,
        Err(e) => return quadrature_decline("riccati", "∫ μ(x)·q₂ dx", muq, e, pool),
    };
    let c = gen.fresh(pool);
    let v = div(simp(pool.add(vec![int_muq, c]), pool), mu, pool);
    let inv_v = simp(pool.pow(v, pool.integer(-1_i32)), pool);
    let y_expr = simp(pool.add(vec![yp_part, inv_v]), pool);
    finalize_explicit(input, y_expr, vec![c], "riccati", pool)
}

/// Decompose `expr` as `q0 + q1·y + q2·y²` (each qi free of y).  Returns None if
/// it is not a polynomial of degree ≤ 2 in `y`.
fn quadratic_in_y(expr: ExprId, y: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId)> {
    let terms: Vec<ExprId> = match pool.get(expr) {
        ExprData::Add(args) => args,
        _ => vec![expr],
    };
    let mut q = [Vec::new(), Vec::new(), Vec::new()];
    for t in terms {
        let (coeff, pw) = monomial_in_y(t, y, pool)?;
        if contains(coeff, y, pool) || !(0..=2).contains(&pw) {
            return None;
        }
        q[pw as usize].push(coeff);
    }
    let mk = |v: Vec<ExprId>| {
        if v.is_empty() {
            pool.integer(0_i32)
        } else {
            simp(pool.add(v), pool)
        }
    };
    Some((mk(q[0].clone()), mk(q[1].clone()), mk(q[2].clone())))
}

/// Search for a polynomial particular solution `y_p = Σ aₖ xᵏ` (degree ≤ 2) of
/// `y' = rhs(x,y)`.  Uses a tiny rational-coefficient grid ansatz and requires
/// the residual to vanish symbolically — only constant/linear/quadratic are
/// attempted, which covers the standard textbook Riccati cases.
fn find_poly_particular(rhs: ExprId, x: ExprId, y: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let candidates = candidate_constants();
    for degree in 0..=2usize {
        for combo in coefficient_combinations(degree + 1, &candidates) {
            let yp = build_poly(&combo, x, pool);
            let ypp = super::ddx(yp, x, pool).ok()?;
            let r = subs1(rhs, y, yp, pool);
            let resid = sub(ypp, r, pool);
            if is_zero(resid, pool) {
                return Some(yp);
            }
        }
    }
    None
}

fn candidate_constants() -> Vec<(i64, i64)> {
    // (numerator, denominator) rationals to try for each coefficient.
    vec![
        (0, 1),
        (1, 1),
        (-1, 1),
        (2, 1),
        (-2, 1),
        (1, 2),
        (-1, 2),
        (3, 1),
        (-3, 1),
    ]
}

fn coefficient_combinations(n: usize, candidates: &[(i64, i64)]) -> Vec<Vec<(i64, i64)>> {
    if n == 0 {
        return vec![vec![]];
    }
    let rest = coefficient_combinations(n - 1, candidates);
    let mut out = Vec::new();
    for &c in candidates {
        for r in &rest {
            let mut v = vec![c];
            v.extend_from_slice(r);
            out.push(v);
        }
    }
    out
}

fn build_poly(coeffs: &[(i64, i64)], x: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    for (k, &(num, den)) in coeffs.iter().enumerate() {
        if num == 0 {
            continue;
        }
        let c = pool.rational(num, den);
        let term = if k == 0 {
            c
        } else {
            let xk = pool.pow(x, pool.integer(k as i32));
            pool.mul(vec![c, xk])
        };
        terms.push(term);
    }
    if terms.is_empty() {
        pool.integer(0_i32)
    } else {
        simp(pool.add(terms), pool)
    }
}

// ---------------------------------------------------------------------------
// Clairaut: y = x y' + f(y').  General solution y = C x + f(C).
// ---------------------------------------------------------------------------

fn try_clairaut(input: &OdeInput, gen: &mut ConstGen, pool: &ExprPool) -> Outcome {
    let (x, y, yp) = (input.x, input.y, input.derivs[0]);
    // Equation in the form y − x y' − f(y') = 0, i.e. equation = y − x·y' − f(y').
    // Detect: equation linear in y with coefficient 1 (or −1), and the
    // remaining part is −x·y' − f(y') (free of y).
    let coeff_y = match ddx(input.equation, y, pool) {
        Ok(v) => v,
        Err(e) => return Outcome::Declined(e),
    };
    // Require ∂/∂y = constant ±1
    let cy = try_expr_f64(simp(coeff_y, pool), pool);
    if cy != Some(1.0) && cy != Some(-1.0) {
        return Outcome::NoMatch;
    }
    let sign = cy.unwrap();
    // rest = equation − coeff_y·y  (should be free of y)
    let rest = sub(input.equation, simp(pool.mul(vec![coeff_y, y]), pool), pool);
    if contains(rest, y, pool) {
        return Outcome::NoMatch;
    }
    // Normalise to y = x y' + f(y'):  equation = sign·y + rest = 0 → y = −rest/sign.
    let y_solved = div(
        simp(pool.mul(vec![pool.integer(-1_i32), rest]), pool),
        simp(pool.integer(sign as i32), pool),
        pool,
    ); // should equal x·y' + f(y')
       // Check the Clairaut shape: y_solved − x·y' = f(y') must be free of x and y
       // (i.e. depend only on y'); the x·y' term carries all the x-dependence.
    let f_of_yp = sub(y_solved, simp(pool.mul(vec![x, yp]), pool), pool);
    if contains(f_of_yp, x, pool) || contains(f_of_yp, y, pool) {
        return Outcome::NoMatch;
    }
    // Ensure y' genuinely appears as the bare x·y' term (reject e.g. y = x·(y')²
    // which is not Clairaut): y_solved must still contain y' linearly via x·y'.
    if !contains(y_solved, yp, pool) {
        return Outcome::NoMatch;
    }
    // General solution: y = C x + f(C).
    let c = gen.fresh(pool);
    let f_of_c = subs1(f_of_yp, yp, c, pool);
    let y_expr = simp(pool.add(vec![pool.mul(vec![c, x]), f_of_c]), pool);
    finalize_explicit(input, y_expr, vec![c], "clairaut", pool)
}
