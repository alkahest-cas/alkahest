//! Transcendental-aware solving for single-variable `exp`/`log` equations.
//!
//! The polynomial system solver ([`super::solve_polynomial_system`]) rejects any
//! equation containing a transcendental function.  This module adds a *scoped*
//! pre-processing layer that handles the common closed-form cases:
//!
//! * **Isolation** — `exp(f) = g  ⟹  f = ln(g)` and `log(f) = g  ⟹  f = exp(g)`
//!   where `f` is linear in the unknown.  Covers `exp(x) = a → x = ln a`,
//!   `log(x) = a → x = exp a`, and the half-life shape `C·exp(-k·t) = C/2`.
//! * **Single-kernel substitution** — when exactly one transcendental kernel
//!   `K = exp(f)` (resp. `log(f)`) occurs and the equation is a polynomial in
//!   `K`, substitute `u = K`, solve the polynomial in `u` (degree ≤ 2), and
//!   back-substitute through `ln`/`exp`.  Covers
//!   `exp(x)² − 3·exp(x) + 2 = 0  →  x ∈ {0, ln 2}`.
//!
//! Scope boundary (documented honestly):
//! * exactly **one** equation in exactly **one** unknown;
//! * exactly **one** distinct transcendental kernel involving the unknown, and
//!   the kernel's argument `f` must be **affine** (degree ≤ 1) in the unknown so
//!   that back-substitution `f = ln r` / `f = exp r` is itself linearly
//!   solvable;
//! * the polynomial in the kernel has degree ≤ 2 (reuses the quadratic path).
//!
//! Anything outside this slice returns
//! [`TranscendentalOutcome::Unsupported`] — the caller surfaces a clean error
//! rather than a fabricated answer.  We only emit solutions we can justify:
//! roots of `exp(·)` that are non-positive are discarded (no real `ln`).

use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::Rational;
use std::collections::BTreeMap;

/// Maximum recursion depth while walking an expression tree (cycle / runaway
/// guard — the pool is a DAG but defence-in-depth is cheap).
const MAX_DEPTH: usize = 256;

/// Maximum number of solutions returned (the back-substituted set is small for
/// the supported degree ≤ 2 cases, but cap to be safe).
const MAX_SOLUTIONS: usize = 8;

/// Result of [`solve_transcendental`].
pub enum TranscendentalOutcome {
    /// Solved: a list of symbolic values for the single unknown.  May be empty
    /// when every algebraic root is rejected by the real-domain constraint
    /// (e.g. `exp(x) = −1` has no real solution).
    Solved(Vec<ExprId>),
    /// The equation is not in the supported transcendental slice; the caller
    /// should fall back to its normal (polynomial) error path.
    Unsupported,
}

/// A transcendental kernel `exp(arg)` or `log(arg)` occurring in the equation.
#[derive(Clone, Copy, PartialEq, Eq)]
enum KernelKind {
    Exp,
    Log,
}

#[derive(Clone, Copy)]
struct Kernel {
    kind: KernelKind,
    /// The `Func` node id (used for identity-based substitution `u = kernel`).
    node: ExprId,
    /// The argument expression `f`.
    arg: ExprId,
}

/// Attempt to solve `equation = 0` for the single variable `var`, handling
/// `exp`/`log` equations.  Returns [`TranscendentalOutcome::Unsupported`] when
/// the equation is outside the supported slice so the caller can fall back.
pub fn solve_transcendental(
    equation: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> TranscendentalOutcome {
    // Collect distinct transcendental kernels whose argument mentions `var`.
    let mut kernels: Vec<Kernel> = Vec::new();
    if !collect_kernels(equation, var, pool, 0, &mut kernels) {
        return TranscendentalOutcome::Unsupported;
    }
    // Deduplicate kernels by node identity (the pool hash-conses, so equal
    // kernels share an id).
    let mut seen = std::collections::HashSet::new();
    kernels.retain(|k| seen.insert(k.node));

    if kernels.is_empty() {
        // No var-bearing transcendental kernel — not our job.
        return TranscendentalOutcome::Unsupported;
    }
    if kernels.len() != 1 {
        // Multiple distinct kernels (e.g. exp(x) and log(x)) — out of scope.
        return TranscendentalOutcome::Unsupported;
    }
    let kernel = kernels[0];

    // The kernel argument must be affine in `var` for back-substitution to be
    // linearly solvable.
    let arg_affine = match affine_in_var(kernel.arg, var, pool) {
        Some(c) => c,
        None => return TranscendentalOutcome::Unsupported,
    };

    // Build a fresh placeholder symbol `u` and substitute the kernel node for
    // it, then check the equation is a polynomial in `u` alone.
    let u = pool.symbol("__u_kernel__", crate::kernel::Domain::Real);
    let in_u = match substitute_node(equation, kernel.node, u, pool, 0) {
        Some(e) => e,
        None => return TranscendentalOutcome::Unsupported,
    };

    // After substitution `in_u` must be a polynomial in `u` *only* (no residual
    // `var`).  Extract univariate coefficients in `u`.
    let coeffs = match poly_coeffs_in(in_u, u, var, kernel.node, pool) {
        Some(c) => c,
        None => return TranscendentalOutcome::Unsupported,
    };

    // Solve the polynomial in u (degree ≤ 2) for rational roots.
    let u_roots = match solve_poly_rational(&coeffs) {
        Some(r) => r,
        None => return TranscendentalOutcome::Unsupported,
    };

    // Back-substitute: kernel = u_root.
    //   exp(arg) = r  ⟹  arg = ln(r)   (only for r > 0)
    //   log(arg) = r  ⟹  arg = exp(r)
    // then solve arg = rhs for var using the affine coefficients.
    let (a, b) = arg_affine; // arg = a*var + b
    let mut out: Vec<ExprId> = Vec::new();
    for r in u_roots {
        let rhs = match kernel.kind {
            KernelKind::Exp => {
                // exp(arg) = r requires r > 0 for a real solution.
                if r <= 0 {
                    continue; // no real solution from this root
                }
                pool.func("log", vec![rational_to_expr(&r, pool)])
            }
            KernelKind::Log => {
                // log(arg) = r  ⟹  arg = exp(r), always real.
                pool.func("exp", vec![rational_to_expr(&r, pool)])
            }
        };
        // Solve a*var + b = rhs  ⟹  var = (rhs - b)/a.
        let var_val = solve_affine(&a, &b, rhs, pool);
        out.push(var_val);
        if out.len() >= MAX_SOLUTIONS {
            break;
        }
    }

    TranscendentalOutcome::Solved(out)
}

/// Walk the expression collecting `exp`/`log` kernels whose argument mentions
/// `var`.  Returns `false` if a *non-supported* transcendental occurs (e.g.
/// `sin`, `tan`, or a var-bearing kernel argument we cannot handle) — in that
/// case the whole equation is unsupported.
fn collect_kernels(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    depth: usize,
    out: &mut Vec<Kernel>,
) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    enum Node {
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
        Leaf,
    }
    let node = pool.with(expr, |d| match d {
        ExprData::Add(a) => Node::Add(a.clone()),
        ExprData::Mul(a) => Node::Mul(a.clone()),
        ExprData::Pow { base, exp } => Node::Pow(*base, *exp),
        ExprData::Func { name, args } => Node::Func(name.clone(), args.clone()),
        _ => Node::Leaf,
    });
    match node {
        Node::Leaf => true,
        Node::Add(args) | Node::Mul(args) => {
            for a in args {
                if !collect_kernels(a, var, pool, depth + 1, out) {
                    return false;
                }
            }
            true
        }
        Node::Pow(base, exp) => {
            collect_kernels(base, var, pool, depth + 1, out)
                && collect_kernels(exp, var, pool, depth + 1, out)
        }
        Node::Func(name, args) => {
            let mentions = args.iter().any(|&a| contains_var(a, var, pool, 0));
            let kind = match name.as_str() {
                "exp" => Some(KernelKind::Exp),
                "log" | "ln" => Some(KernelKind::Log),
                _ => None,
            };
            match (kind, mentions) {
                (Some(kind), true) if args.len() == 1 => {
                    // A nested transcendental inside the argument (e.g.
                    // exp(log(x))) would break the single-kernel assumption.
                    if has_transcendental(args[0], var, pool, 0) {
                        return false;
                    }
                    out.push(Kernel {
                        kind,
                        node: expr,
                        arg: args[0],
                    });
                    true
                }
                (Some(_), true) => false, // exp/log with arity != 1: bail
                (None, true) => false,    // unsupported var-bearing transcendental (sin, …)
                _ => {
                    // Function not mentioning var — recurse anyway to be safe.
                    for a in args {
                        if !collect_kernels(a, var, pool, depth + 1, out) {
                            return false;
                        }
                    }
                    true
                }
            }
        }
    }
}

/// Does `expr` contain a transcendental function mentioning `var`?
fn has_transcendental(expr: ExprId, var: ExprId, pool: &ExprPool, depth: usize) -> bool {
    if depth > MAX_DEPTH {
        return true; // conservative
    }
    pool.with(expr, |d| match d {
        ExprData::Func { name, args } => {
            let is_trans = matches!(name.as_str(), "exp" | "log" | "ln");
            (is_trans && args.iter().any(|&a| contains_var(a, var, pool, 0)))
                || args
                    .iter()
                    .any(|&a| has_transcendental(a, var, pool, depth + 1))
        }
        ExprData::Add(a) | ExprData::Mul(a) => a
            .iter()
            .any(|&x| has_transcendental(x, var, pool, depth + 1)),
        ExprData::Pow { base, exp } => {
            has_transcendental(*base, var, pool, depth + 1)
                || has_transcendental(*exp, var, pool, depth + 1)
        }
        _ => false,
    })
}

/// Does `expr` syntactically contain `var`?
fn contains_var(expr: ExprId, var: ExprId, pool: &ExprPool, depth: usize) -> bool {
    if expr == var {
        return true;
    }
    if depth > MAX_DEPTH {
        return true; // conservative
    }
    pool.with(expr, |d| match d {
        ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
            a.iter().any(|&x| contains_var(x, var, pool, depth + 1))
        }
        ExprData::Pow { base, exp } => {
            contains_var(*base, var, pool, depth + 1) || contains_var(*exp, var, pool, depth + 1)
        }
        _ => false,
    })
}

/// If `expr` is affine in `var` (i.e. `a·var + b` where `a`, `b` are free of
/// `var`), return `(a, b)` as `ExprId`s.  Returns `None` otherwise (including
/// nonlinear or var inside a function).
fn affine_in_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let zero = pool.integer(0_i32);
    let one = pool.integer(1_i32);
    if expr == var {
        return Some((one, zero));
    }
    if !contains_var(expr, var, pool, 0) {
        return Some((zero, expr)); // constant
    }
    enum Node {
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Other,
    }
    let node = pool.with(expr, |d| match d {
        ExprData::Add(a) => Node::Add(a.clone()),
        ExprData::Mul(a) => Node::Mul(a.clone()),
        _ => Node::Other,
    });
    match node {
        Node::Add(args) => {
            let mut a_terms: Vec<ExprId> = Vec::new();
            let mut b_terms: Vec<ExprId> = Vec::new();
            for t in args {
                let (ai, bi) = affine_in_var(t, var, pool)?;
                a_terms.push(ai);
                b_terms.push(bi);
            }
            let a = sum(a_terms, pool);
            let b = sum(b_terms, pool);
            Some((a, b))
        }
        Node::Mul(args) => {
            // Exactly one factor may contain var, and it must be affine; the
            // rest are coefficients (must be var-free).
            let mut coeff: Vec<ExprId> = Vec::new();
            let mut var_factor: Option<ExprId> = None;
            for f in &args {
                if contains_var(*f, var, pool, 0) {
                    if var_factor.is_some() {
                        return None; // var·var → nonlinear
                    }
                    var_factor = Some(*f);
                } else {
                    coeff.push(*f);
                }
            }
            let c = if coeff.is_empty() {
                pool.integer(1_i32)
            } else {
                pool.mul(coeff)
            };
            match var_factor {
                None => Some((zero, expr)),
                Some(vf) => {
                    let (a, b) = affine_in_var(vf, var, pool)?;
                    // coefficient * (a*var + b) = (c*a)*var + (c*b)
                    let ca = pool.mul(vec![c, a]);
                    let cb = pool.mul(vec![c, b]);
                    Some((ca, cb))
                }
            }
        }
        Node::Other => None, // pow / func containing var → not affine
    }
}

fn sum(mut terms: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    terms.retain(|&t| !pool.with(t, |d| matches!(d, ExprData::Integer(n) if n.0 == 0)));
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

/// Replace every occurrence of node `target` (by identity) in `expr` with
/// `replacement`.  Returns `None` if depth is exceeded.
fn substitute_node(
    expr: ExprId,
    target: ExprId,
    replacement: ExprId,
    pool: &ExprPool,
    depth: usize,
) -> Option<ExprId> {
    if expr == target {
        return Some(replacement);
    }
    if depth > MAX_DEPTH {
        return None;
    }
    enum Node {
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String, Vec<ExprId>),
        Leaf,
    }
    let node = pool.with(expr, |d| match d {
        ExprData::Add(a) => Node::Add(a.clone()),
        ExprData::Mul(a) => Node::Mul(a.clone()),
        ExprData::Pow { base, exp } => Node::Pow(*base, *exp),
        ExprData::Func { name, args } => Node::Func(name.clone(), args.clone()),
        _ => Node::Leaf,
    });
    match node {
        Node::Leaf => Some(expr),
        Node::Add(args) => {
            let mut new = Vec::with_capacity(args.len());
            for a in args {
                new.push(substitute_node(a, target, replacement, pool, depth + 1)?);
            }
            Some(pool.add(new))
        }
        Node::Mul(args) => {
            let mut new = Vec::with_capacity(args.len());
            for a in args {
                new.push(substitute_node(a, target, replacement, pool, depth + 1)?);
            }
            Some(pool.mul(new))
        }
        Node::Pow(base, exp) => {
            let b = substitute_node(base, target, replacement, pool, depth + 1)?;
            let e = substitute_node(exp, target, replacement, pool, depth + 1)?;
            Some(pool.pow(b, e))
        }
        Node::Func(name, args) => {
            let mut new = Vec::with_capacity(args.len());
            for a in args {
                new.push(substitute_node(a, target, replacement, pool, depth + 1)?);
            }
            Some(pool.func(name, new))
        }
    }
}

/// Extract univariate rational coefficients of `poly` in variable `u`.  Returns
/// `coeffs[k]` = coefficient of `u^k`.  Returns `None` if `poly` is not a
/// polynomial in `u` with rational coefficients, or if it still contains
/// `original_var` or the kernel node (meaning the substitution did not fully
/// linearise the equation).
fn poly_coeffs_in(
    poly: ExprId,
    u: ExprId,
    original_var: ExprId,
    kernel_node: ExprId,
    pool: &ExprPool,
) -> Option<Vec<Rational>> {
    // Reject residual var / kernel — substitution must be total.
    if contains_var(poly, original_var, pool, 0) {
        return None;
    }
    if contains_var(poly, kernel_node, pool, 0) {
        return None;
    }
    let mut map: BTreeMap<u32, Rational> = BTreeMap::new();
    accumulate_poly(poly, u, pool, Rational::from(1), 0, 0, &mut map)?;
    map.retain(|_, v| *v != 0);
    let degree = map.keys().max().copied().unwrap_or(0);
    if degree > 2 {
        return None; // reuse quadratic path only
    }
    let mut coeffs = vec![Rational::from(0); (degree + 1) as usize];
    for (k, v) in map {
        coeffs[k as usize] = v;
    }
    Some(coeffs)
}

/// Recursively accumulate `scale * expr` into `out[deg_in_u] += coeff`.
fn accumulate_poly(
    expr: ExprId,
    u: ExprId,
    pool: &ExprPool,
    scale: Rational,
    deg_so_far: u32,
    depth: usize,
    out: &mut BTreeMap<u32, Rational>,
) -> Option<()> {
    if depth > MAX_DEPTH {
        return None;
    }
    if expr == u {
        *out.entry(deg_so_far + 1)
            .or_insert_with(|| Rational::from(0)) += scale;
        return Some(());
    }
    enum Node {
        Int(rug::Integer),
        Rat(Rational),
        Float(f64),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Other,
    }
    let node = pool.with(expr, |d| match d {
        ExprData::Integer(n) => Node::Int(n.0.clone()),
        ExprData::Rational(r) => Node::Rat(r.0.clone()),
        ExprData::Float(f) => Node::Float(f.inner.to_f64()),
        ExprData::Add(a) => Node::Add(a.clone()),
        ExprData::Mul(a) => Node::Mul(a.clone()),
        ExprData::Pow { base, exp } => Node::Pow(*base, *exp),
        _ => Node::Other,
    });
    match node {
        Node::Int(n) => {
            *out.entry(deg_so_far).or_insert_with(|| Rational::from(0)) +=
                scale * Rational::from(n);
            Some(())
        }
        Node::Rat(r) => {
            *out.entry(deg_so_far).or_insert_with(|| Rational::from(0)) += scale * r;
            Some(())
        }
        Node::Float(f) => {
            let r = Rational::from_f64(f)?;
            *out.entry(deg_so_far).or_insert_with(|| Rational::from(0)) += scale * r;
            Some(())
        }
        Node::Add(args) => {
            for a in args {
                accumulate_poly(a, u, pool, scale.clone(), deg_so_far, depth + 1, out)?;
            }
            Some(())
        }
        Node::Mul(args) => {
            // Multiply factors by computing each factor's univariate map and
            // convolving (polynomial multiplication).
            let mut prod: BTreeMap<u32, Rational> = BTreeMap::new();
            prod.insert(0, Rational::from(1));
            for a in &args {
                let mut factor: BTreeMap<u32, Rational> = BTreeMap::new();
                accumulate_poly(*a, u, pool, Rational::from(1), 0, depth + 1, &mut factor)?;
                prod = convolve(&prod, &factor)?;
            }
            for (k, v) in prod {
                *out.entry(deg_so_far + k)
                    .or_insert_with(|| Rational::from(0)) += scale.clone() * v;
            }
            Some(())
        }
        Node::Pow(base, exp) => {
            // Only integer exponent ≥ 0 supported.
            let n = pool.with(exp, |d| match d {
                ExprData::Integer(n) => n.0.to_u32(),
                _ => None,
            })?;
            let mut base_map: BTreeMap<u32, Rational> = BTreeMap::new();
            accumulate_poly(
                base,
                u,
                pool,
                Rational::from(1),
                0,
                depth + 1,
                &mut base_map,
            )?;
            let mut acc: BTreeMap<u32, Rational> = BTreeMap::new();
            acc.insert(0, Rational::from(1));
            for _ in 0..n {
                acc = convolve(&acc, &base_map)?;
            }
            for (k, v) in acc {
                *out.entry(deg_so_far + k)
                    .or_insert_with(|| Rational::from(0)) += scale.clone() * v;
            }
            Some(())
        }
        Node::Other => None, // non-polynomial leftover
    }
}

/// Convolve two univariate coefficient maps (polynomial multiplication).
fn convolve(
    a: &BTreeMap<u32, Rational>,
    b: &BTreeMap<u32, Rational>,
) -> Option<BTreeMap<u32, Rational>> {
    let mut out: BTreeMap<u32, Rational> = BTreeMap::new();
    for (&ka, va) in a {
        for (&kb, vb) in b {
            let deg = ka.checked_add(kb)?;
            if deg > 64 {
                return None; // runaway degree guard
            }
            *out.entry(deg).or_insert_with(|| Rational::from(0)) += Rational::from(va * vb);
        }
    }
    Some(out)
}

/// Solve `c[0] + c[1]·u + c[2]·u² = 0` for *rational* roots only.  Returns
/// `None` if any root is irrational (we then refuse rather than fabricate a
/// non-closed-form `u`).
fn solve_poly_rational(coeffs: &[Rational]) -> Option<Vec<Rational>> {
    let mut degree = 0usize;
    for (i, c) in coeffs.iter().enumerate() {
        if *c != 0 {
            degree = i;
        }
    }
    match degree {
        0 => Some(vec![]), // constant (no var) — no kernel solution
        1 => {
            let a = &coeffs[1];
            let b = coeffs.first().cloned().unwrap_or_else(|| Rational::from(0));
            let neg_b = -b;
            Some(vec![neg_b / a.clone()])
        }
        2 => {
            let a = coeffs[2].clone();
            let b = coeffs.get(1).cloned().unwrap_or_else(|| Rational::from(0));
            let c = coeffs.first().cloned().unwrap_or_else(|| Rational::from(0));
            let disc = Rational::from(&b * &b) - Rational::from(4) * &a * &c;
            if disc < 0 {
                return Some(vec![]); // no real root
            }
            let dn = disc.numer().clone();
            let dd = disc.denom().clone();
            let (sn, rn) = dn.sqrt_rem(rug::Integer::new());
            let (sd, rd) = dd.sqrt_rem(rug::Integer::new());
            if rn != 0 || rd != 0 {
                return None; // irrational root → refuse
            }
            let sqrt_disc = Rational::from((sn, sd));
            let two_a = Rational::from(2) * &a;
            let r1 = Rational::from(&(-b.clone()) + &sqrt_disc) / two_a.clone();
            let r2 = (-b - sqrt_disc) / two_a;
            if r1 == r2 {
                Some(vec![r1])
            } else {
                Some(vec![r1, r2])
            }
        }
        _ => None,
    }
}

/// Solve `a·var + b = rhs` for `var`, returning `(rhs - b)/a` as an `ExprId`.
fn solve_affine(a: &ExprId, b: &ExprId, rhs: ExprId, pool: &ExprPool) -> ExprId {
    let neg_one = pool.integer(-1_i32);
    let neg_b = pool.mul(vec![neg_one, *b]);
    let num = pool.add(vec![rhs, neg_b]);
    // num / a = num * a^(-1)
    let inv_a = pool.pow(*a, neg_one);
    pool.mul(vec![num, inv_a])
}

fn rational_to_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    let (num, den) = r.clone().into_numer_denom();
    if den == 1 {
        pool.integer(num)
    } else {
        pool.rational(num, den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::{Domain, ExprPool};
    use std::collections::HashMap;

    fn evalf(e: ExprId, pool: &ExprPool) -> f64 {
        eval_interp(e, &HashMap::new(), pool).expect("numeric eval")
    }

    fn solved(eq: ExprId, var: ExprId, pool: &ExprPool) -> Vec<ExprId> {
        match solve_transcendental(eq, var, pool) {
            TranscendentalOutcome::Solved(v) => v,
            TranscendentalOutcome::Unsupported => panic!("expected Solved, got Unsupported"),
        }
    }

    #[test]
    fn exp_x_eq_a() {
        // exp(x) - 3 = 0  →  x = ln 3
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let neg3 = pool.integer(-3_i32);
        let eq = pool.add(vec![pool.func("exp", vec![x]), neg3]);
        let sols = solved(eq, x, &pool);
        assert_eq!(sols.len(), 1);
        assert!((evalf(sols[0], &pool) - 3f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn log_x_eq_a() {
        // log(x) - 2 = 0  →  x = exp 2
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let neg2 = pool.integer(-2_i32);
        let eq = pool.add(vec![pool.func("log", vec![x]), neg2]);
        let sols = solved(eq, x, &pool);
        assert_eq!(sols.len(), 1);
        assert!((evalf(sols[0], &pool) - 2f64.exp()).abs() < 1e-9);
    }

    #[test]
    fn half_life() {
        // exp(-k*t) - 1/2 = 0  →  t = ln(2)/k.  k is a free symbol; bind k = 3.
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let k = pool.symbol("k", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let neg_kt = pool.mul(vec![neg_one, k, t]);
        let half = pool.rational(rug::Integer::from(-1), rug::Integer::from(2));
        let eq = pool.add(vec![pool.func("exp", vec![neg_kt]), half]);
        let sols = solved(eq, t, &pool);
        assert_eq!(sols.len(), 1);
        // t = ln(1/2)/(-k) = ln(2)/k.  Bind k = 3 and check.
        let mut env = HashMap::new();
        env.insert(k, 3.0_f64);
        let v = eval_interp(sols[0], &env, &pool).expect("eval");
        assert!((v - 2f64.ln() / 3.0).abs() < 1e-10);
    }

    #[test]
    fn exp_polynomial() {
        // exp(x)² - 3*exp(x) + 2 = 0  →  u² - 3u + 2 = 0 → u ∈ {1, 2}
        //   exp(x) = 1 → x = ln 1 = 0;  exp(x) = 2 → x = ln 2
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let exp_x = pool.func("exp", vec![x]);
        let exp_x_sq = pool.pow(exp_x, two);
        let neg3 = pool.integer(-3_i32);
        let term2 = pool.mul(vec![neg3, exp_x]);
        let plus2 = pool.integer(2_i32);
        let eq = pool.add(vec![exp_x_sq, term2, plus2]);
        let sols = solved(eq, x, &pool);
        assert_eq!(sols.len(), 2);
        let vals: Vec<f64> = sols.iter().map(|&s| evalf(s, &pool)).collect();
        assert!(vals.iter().any(|v| v.abs() < 1e-10)); // ln 1 = 0
        assert!(vals.iter().any(|v| (v - 2f64.ln()).abs() < 1e-10)); // ln 2
    }

    #[test]
    fn exp_negative_no_real() {
        // exp(x) + 1 = 0  →  exp(x) = -1, no real solution → empty Solved.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let eq = pool.add(vec![pool.func("exp", vec![x]), one]);
        let sols = solved(eq, x, &pool);
        assert!(sols.is_empty());
    }

    #[test]
    fn unsupported_sin() {
        // sin(x) = 0 → unsupported (not exp/log).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let eq = pool.func("sin", vec![x]);
        assert!(matches!(
            solve_transcendental(eq, x, &pool),
            TranscendentalOutcome::Unsupported
        ));
    }

    #[test]
    fn unsupported_two_kernels() {
        // exp(x) + log(x) = 0 → two distinct kernels → unsupported.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let eq = pool.add(vec![pool.func("exp", vec![x]), pool.func("log", vec![x])]);
        assert!(matches!(
            solve_transcendental(eq, x, &pool),
            TranscendentalOutcome::Unsupported
        ));
    }

    #[test]
    fn unsupported_pure_polynomial() {
        // x² - 1 = 0 → no transcendental kernel → unsupported (caller handles).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let eq = pool.add(vec![pool.pow(x, pool.integer(2_i32)), neg_one]);
        assert!(matches!(
            solve_transcendental(eq, x, &pool),
            TranscendentalOutcome::Unsupported
        ));
    }
}
