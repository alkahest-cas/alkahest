//! Symbolic limits towards finite points or ±∞ via local expansions (`Series`),
//! L'Hôpital iterations, and algebraic transforms (V2-16).

use crate::calculus::series::{local_expansion, LocalExpansion};
use crate::diff::{diff, DiffError};
use crate::kernel::pool::POS_INFINITY_SYMBOL;
use crate::kernel::{subs, ExprData, ExprId, ExprPool};
use crate::poly::{poly_normal, RationalFunction};
use crate::simplify::{simplify, simplify_expanded};
use crate::SeriesError;
use std::collections::HashMap;
use std::fmt;

/// Approach direction toward `point` (real-axis ordering).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LimitDirection {
    /// Ordinary two-sided limit.
    Bidirectional,
    /// Limits with `var > point` (approach from the right on the usual number line picture).
    Plus,
    /// Limits with `var < point`.
    Minus,
}

#[derive(Debug)]
pub enum LimitError {
    /// Sub-problem rejected by [`crate::calculus::series`].
    Series(SeriesError),
    /// Derivative unavailable for L'Hôpital.
    Diff(DiffError),
    /// Odd-order pole requires a one-sided direction.
    NeedsOneSided,
    /// Maximal L'Hôpital / recursion depth exceeded.
    DepthExceeded,
    /// No implemented rule applies (non-comparable growth, oscillation, …).
    Unsupported,
}

impl fmt::Display for LimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LimitError::Series(e) => write!(f, "{e}"),
            LimitError::Diff(e) => write!(f, "{e}"),
            LimitError::NeedsOneSided => {
                write!(
                    f,
                    "two-sided limit undefined at this pole; pass direction Plus or Minus"
                )
            }
            LimitError::DepthExceeded => write!(f, "limit refinement depth exceeded"),
            LimitError::Unsupported => write!(f, "limit could not be computed with current rules"),
        }
    }
}

impl std::error::Error for LimitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LimitError::Series(e) => Some(e),
            LimitError::Diff(e) => Some(e),
            _ => None,
        }
    }
}

impl crate::errors::AlkahestError for LimitError {
    fn code(&self) -> &'static str {
        match self {
            LimitError::Series(_) => "E-LIMIT-001",
            LimitError::Diff(_) => "E-LIMIT-002",
            LimitError::NeedsOneSided => "E-LIMIT-003",
            LimitError::DepthExceeded => "E-LIMIT-004",
            LimitError::Unsupported => "E-LIMIT-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            LimitError::Series(_) => {
                "increase truncation order indirectly by simplifying the expression, or rewrite using standard limits"
            }
            LimitError::Diff(_) => {
                "ensure primitives have differentiation rules, or simplify before taking the limit"
            }
            LimitError::NeedsOneSided => "use LimitDirection::Plus or Minus matching the desired one-sided approach",
            LimitError::DepthExceeded => {
                "try manual algebra (quotient form, cancellations) or split into simpler sub-expressions"
            }
            LimitError::Unsupported => {
                "unsupported indeterminate — Gruntz-style comparability beyond this prototype is future work"
            }
        })
    }
}

impl From<SeriesError> for LimitError {
    fn from(e: SeriesError) -> Self {
        LimitError::Series(e)
    }
}

impl From<DiffError> for LimitError {
    fn from(e: DiffError) -> Self {
        LimitError::Diff(e)
    }
}

// ---------------------------------------------------------------------------

/// `limit(expr, var, point, dir)` — see [`LimitDirection`].
///
/// `point` may be finite or [`ExprPool::pos_infinity`]. Limits at `-∞` use
/// `pool.mul(pool.integer(-1), pool.pos_infinity())`.
pub fn limit(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
) -> Result<ExprId, LimitError> {
    let r = limit_inner(expr, var, point, direction, pool, 0)?;
    Ok(simplify(fold_known_reals(simplify(r, pool).value, pool), pool).value)
}

/// `(g^m)^n ↦ g^{m n}` when `m,n ∈ ℤ`, so substitutions like `(1/t)^k` become `t^{-k}` Laurent heads.
fn flatten_nested_integer_pow(expr: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            let base = flatten_nested_integer_pow(base, pool);
            let exp_fl = flatten_nested_integer_pow(exp, pool);
            if let (
                ExprData::Pow {
                    base: b2,
                    exp: inner_exp,
                },
                ExprData::Integer(outer_e),
            ) = (pool.get(base), pool.get(exp_fl))
            {
                if let ExprData::Integer(inner_e) = pool.get(inner_exp) {
                    let prod = (&inner_e.0).clone() * (&outer_e.0).clone();
                    return pool.pow(flatten_nested_integer_pow(b2, pool), pool.integer(prod));
                }
            }
            pool.pow(base, exp_fl)
        }
        ExprData::Mul(xs) => pool.mul(
            xs.iter()
                .map(|x| flatten_nested_integer_pow(*x, pool))
                .collect(),
        ),
        ExprData::Add(xs) => pool.add(
            xs.iter()
                .map(|x| flatten_nested_integer_pow(*x, pool))
                .collect(),
        ),
        ExprData::Func { name, args } => {
            let na: Vec<ExprId> = args
                .iter()
                .map(|a| flatten_nested_integer_pow(*a, pool))
                .collect();
            pool.func(name.clone(), na)
        }
        _ => expr,
    }
}

/// After ``x ↦ 1/t``, common forms are ``Mul(numer, denom^{-1})`` with ``Pow(t,-1)``
/// sprinkled through both.  Clear those poles by multiplying by ``t^k`` until
/// numerator and denominator describe an honest polynomial quotient in ``t``.
fn canonical_polynomial_quotient_in_var(expr: ExprId, t: ExprId, pool: &ExprPool) -> ExprId {
    let (n_raw, d_raw) = numerator_denominator(expr, pool);
    if d_raw == pool.integer(1_i32) {
        return expr;
    }
    for k in 0_i64..=40 {
        let tk = pool.pow(t, pool.integer(k));
        let n = simplify_expanded(pool.mul(vec![tk, n_raw]), pool).value;
        let d = simplify_expanded(pool.mul(vec![tk, d_raw]), pool).value;
        let (n, d) = match (poly_normal(n, vec![t], pool), poly_normal(d, vec![t], pool)) {
            (Ok(nn), Ok(dd)) => (nn, dd),
            _ => continue,
        };
        if let Ok(rf) = RationalFunction::from_symbolic(n, d, vec![t], pool) {
            let nx = rf.numer.to_expr(pool);
            let dx = rf.denom.to_expr(pool);
            return simplify(pool.mul(vec![nx, pool.pow(dx, pool.integer(-1_i32))]), pool).value;
        }
    }
    expr
}

fn limit_inner(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Result<ExprId, LimitError> {
    const MAX_DEPTH: u32 = 48;
    const SERIES_ORDER: u32 = 32;
    if depth > MAX_DEPTH {
        return Err(LimitError::DepthExceeded);
    }

    if !depends_on(expr, var, pool) {
        if substitution_is_singular(expr, pool) {
            return Err(LimitError::Unsupported);
        }
        return Ok(fold_known_reals(simplify(expr, pool).value, pool));
    }

    if let Some(r) = try_special_function_limits(expr, var, point, direction, pool)? {
        return Ok(r);
    }

    if is_pos_infinity(point, pool) {
        let t = pool.symbol("__lt_inf", crate::kernel::Domain::Real);
        let inv_t = pool.pow(t, pool.integer(-1_i32));
        let mut m = HashMap::new();
        m.insert(var, inv_t);
        let e2 = simplify(
            canonical_polynomial_quotient_in_var(
                flatten_nested_integer_pow(subs(expr, &m, pool), pool),
                t,
                pool,
            ),
            pool,
        )
        .value;
        return limit_inner(
            e2,
            t,
            pool.integer(0_i32),
            LimitDirection::Plus,
            pool,
            depth + 1,
        );
    }

    if is_neg_infinity(point, pool) {
        let t = pool.symbol("__lt_ninf", crate::kernel::Domain::Real);
        let rep = pool.mul(vec![
            pool.integer(-1_i32),
            pool.pow(t, pool.integer(-1_i32)),
        ]);
        let mut m = HashMap::new();
        m.insert(var, rep);
        let e2 = simplify(
            canonical_polynomial_quotient_in_var(
                flatten_nested_integer_pow(subs(expr, &m, pool), pool),
                t,
                pool,
            ),
            pool,
        )
        .value;
        return limit_inner(
            e2,
            t,
            pool.integer(0_i32),
            LimitDirection::Plus,
            pool,
            depth + 1,
        );
    }

    if let Some(r) = try_direct_substitution(expr, var, point, pool) {
        return Ok(r);
    }

    if let Some(r) = try_x_log_x_at_zero(expr, var, point, direction, pool, depth)? {
        return Ok(r);
    }

    if let Some(r) = try_lhopital(expr, var, point, direction, pool, depth)? {
        return Ok(r);
    }

    if let Some(r) = try_expansion_limit(expr, var, point, direction, pool, SERIES_ORDER)? {
        return Ok(r);
    }

    Err(LimitError::Unsupported)
}

fn try_x_log_x_at_zero(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Result<Option<ExprId>, LimitError> {
    if direction == LimitDirection::Minus {
        return Ok(None);
    }
    if !matches!(pool.get(point), ExprData::Integer(n) if n.0 == 0) {
        return Ok(None);
    }
    let ExprData::Mul(args) = pool.get(expr) else {
        return Ok(None);
    };
    if args.len() != 2 {
        return Ok(None);
    }
    let (a, b) = (args[0], args[1]);
    let log_of_var = |u: ExprId| {
        matches!(
            pool.get(u),
            ExprData::Func { name, args: av } if name == "log" && av.len() == 1 && av[0] == var
        )
    };
    let is_var = |u: ExprId| u == var;
    let ok = (is_var(a) && log_of_var(b)) || (is_var(b) && log_of_var(a));
    if !ok {
        return Ok(None);
    }
    // L'Hôpital on log(x) / x^{-1}: (1/x) / (-1/x^2) = -x  → 0 as x→0+.
    let f = pool.func("log", vec![var]);
    let g = pool.pow(var, pool.integer(-1_i32));
    let fp = diff(f, var, pool)?.value;
    let gp = diff(g, var, pool)?.value;
    let ratio = rational_quotient(fp, gp, pool);
    Ok(Some(limit_inner(
        ratio,
        var,
        point,
        LimitDirection::Plus,
        pool,
        depth + 1,
    )?))
}

fn try_special_function_limits(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
) -> Result<Option<ExprId>, LimitError> {
    let ExprData::Func { name, args } = pool.get(expr) else {
        return Ok(None);
    };
    if args.len() != 1 || args[0] != var {
        return Ok(None);
    }
    match name.as_str() {
        "exp" => {
            if is_pos_infinity(point, pool) {
                return Ok(Some(pool.pos_infinity()));
            }
            if is_neg_infinity(point, pool) {
                return Ok(Some(pool.integer(0_i32)));
            }
            if matches!(pool.get(point), ExprData::Integer(n) if n.0 == 0) {
                return Ok(Some(pool.integer(1_i32)));
            }
        }
        "log" => {
            if is_pos_infinity(point, pool) {
                return Ok(Some(pool.pos_infinity()));
            }
            if matches!(pool.get(point), ExprData::Integer(n) if n.0 == 0) {
                if direction == LimitDirection::Plus {
                    return Ok(Some(neg_infinity(pool)));
                }
                return Err(LimitError::NeedsOneSided);
            }
        }
        _ => {}
    }
    Ok(None)
}

fn neg_infinity(pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), pool.pos_infinity()])
}

fn is_pos_infinity(e: ExprId, pool: &ExprPool) -> bool {
    matches!(
        pool.get(e),
        ExprData::Symbol { name, domain: crate::kernel::Domain::Positive }
            if name == POS_INFINITY_SYMBOL
    ) || matches!(
        pool.get(e),
        ExprData::Symbol { name, domain: crate::kernel::Domain::Real }
            if name == POS_INFINITY_SYMBOL
    )
}

fn is_neg_infinity(e: ExprId, pool: &ExprPool) -> bool {
    let ExprData::Mul(args) = pool.get(e) else {
        return false;
    };
    if args.len() != 2 {
        return false;
    }
    let (a, b) = (args[0], args[1]);
    let m_one = pool.integer(-1_i32);
    (a == m_one && is_pos_infinity(b, pool)) || (b == m_one && is_pos_infinity(a, pool))
}

fn depends_on(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|a| depends_on(*a, var, pool)),
        ExprData::Pow { base, exp } => depends_on(base, var, pool) || depends_on(exp, var, pool),
        ExprData::Func { args, .. } => args.iter().any(|a| depends_on(*a, var, pool)),
        ExprData::Piecewise { branches, default } => {
            branches
                .iter()
                .any(|(c, v)| depends_on(*c, var, pool) || depends_on(*v, var, pool))
                || depends_on(default, var, pool)
        }
        ExprData::Predicate { args, .. } => args.iter().any(|a| depends_on(*a, var, pool)),
        ExprData::Forall { var: bv, body } | ExprData::Exists { var: bv, body } => {
            bv != var && depends_on(body, var, pool)
        }
        ExprData::BigO(a) => depends_on(a, var, pool),
        ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_)
        | ExprData::Symbol { .. } => false,
    }
}

fn try_direct_substitution(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if quotient_is_zero_over_zero(expr, var, point, pool) {
        return None;
    }
    let mut m = HashMap::new();
    m.insert(var, point);
    let raw = subs(expr, &m, pool);
    if is_zero_times_pole_indeterminate(raw, pool) {
        return None;
    }
    let sub = fold_known_reals(simplify(raw, pool).value, pool);
    if depends_on(sub, var, pool) {
        None
    } else if substitution_is_singular(sub, pool) {
        None
    } else {
        Some(sub)
    }
}

/// True when ``expr`` is a product quotient `n/d` with `n,d → 0` at substitution (classic `0/0`).
fn quotient_is_zero_over_zero(expr: ExprId, var: ExprId, point: ExprId, pool: &ExprPool) -> bool {
    let (n, d) = numerator_denominator(expr, pool);
    if d == pool.integer(1_i32) {
        return false;
    }
    let n0 = substitute_fully(n, var, point, pool);
    let d0 = substitute_fully(d, var, point, pool);
    is_zero_like(n0, pool) && is_zero_like(d0, pool)
}

/// `0 · (pole at 0)` style indeterminate — must not simplify to misleading `0`.
fn is_zero_times_pole_indeterminate(expr: ExprId, pool: &ExprPool) -> bool {
    let factors: Vec<ExprId> = if matches!(pool.get(expr), ExprData::Mul(_)) {
        flatten_mul(expr, pool)
    } else {
        vec![expr]
    };
    let mut any_zero_factor = false;
    let mut any_pole = false;
    for f in factors {
        if substitution_is_singular(f, pool) {
            any_pole = true;
        }
        if matches!(pool.get(f), ExprData::Integer(z) if z.0 == 0) {
            any_zero_factor = true;
        }
        if let ExprData::Func { name, args } = pool.get(f) {
            if args.len() == 1 && matches!(name.as_str(), "sin" | "sinh" | "tan") {
                if matches!(pool.get(args[0]), ExprData::Integer(z) if z.0 == 0) {
                    any_zero_factor = true;
                }
            }
        }
    }
    any_zero_factor && any_pole
}

/// `true` after substitution if some sub-expression is ``0^{-n}`` (possibly nested via ``(0^{-1})^e``).
fn substitution_is_singular(expr: ExprId, pool: &ExprPool) -> bool {
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            if let ExprData::Integer(nn) = pool.get(exp) {
                if nn.0 < 0 {
                    let b = simplify(base, pool).value;
                    if matches!(pool.get(b), ExprData::Integer(z) if z.0 == 0) {
                        return true;
                    }
                }
            }
            substitution_is_singular(base, pool) || substitution_is_singular(exp, pool)
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            xs.iter().any(|a| substitution_is_singular(*a, pool))
        }
        ExprData::Func { args, .. } => args.iter().any(|a| substitution_is_singular(*a, pool)),
        _ => false,
    }
}

fn try_lhopital(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    depth: u32,
) -> Result<Option<ExprId>, LimitError> {
    let (nume, deno) = numerator_denominator(expr, pool);
    if simplify(nume, pool).value == simplify(deno, pool).value {
        return Ok(None);
    }
    let n0 = substitute_fully(nume, var, point, pool);
    let d0 = substitute_fully(deno, var, point, pool);

    if !is_zero_like(n0, pool) || !is_zero_like(d0, pool) {
        return Ok(None);
    }

    let dn = diff(nume, var, pool)?.value;
    let dd = diff(deno, var, pool)?.value;
    if dn == nume && dd == deno {
        return Ok(None);
    }
    let quot = rational_quotient(dn, dd, pool);
    Ok(Some(limit_inner(
        quot,
        var,
        point,
        direction,
        pool,
        depth + 1,
    )?))
}

fn substitute_fully(expr: ExprId, var: ExprId, point: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(var, point);
    let s = simplify(subs(expr, &m, pool), pool).value;
    fold_known_reals(s, pool)
}

fn rational_quotient(n: ExprId, d: ExprId, pool: &ExprPool) -> ExprId {
    simplify(pool.mul(vec![n, pool.pow(d, pool.integer(-1_i32))]), pool).value
}

fn is_zero_like(e: ExprId, pool: &ExprPool) -> bool {
    let e = simplify(e, pool).value;
    if matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0) {
        return true;
    }
    if let ExprData::Rational(r) = pool.get(e) {
        if r.0 == 0 {
            return true;
        }
    }
    if let ExprData::Func { name, args } = pool.get(e) {
        if args.len() == 1 && matches!(name.as_str(), "sin" | "tan" | "sinh") {
            return is_zero_like(args[0], pool);
        }
    }
    false
}

fn is_one_like(e: ExprId, pool: &ExprPool) -> bool {
    let e = simplify(e, pool).value;
    if matches!(pool.get(e), ExprData::Integer(n) if n.0 == 1) {
        return true;
    }
    if let ExprData::Rational(r) = pool.get(e) {
        return r.0 == 1;
    }
    false
}

/// Constant-fold `sin`, `cos`, `exp`, … after limits (`sin(0) → 0`, `cos(0) → 1`).
fn fold_known_reals(expr: ExprId, pool: &ExprPool) -> ExprId {
    let e = simplify(expr, pool).value;
    match pool.get(e) {
        ExprData::Add(xs) => {
            let ys: Vec<ExprId> = xs.iter().map(|x| fold_known_reals(*x, pool)).collect();
            simplify(pool.add(ys), pool).value
        }
        ExprData::Mul(xs) => {
            let ys: Vec<ExprId> = xs.iter().map(|x| fold_known_reals(*x, pool)).collect();
            simplify(pool.mul(ys), pool).value
        }
        ExprData::Pow { base, exp } => {
            let b = fold_known_reals(base, pool);
            let xp = fold_known_reals(exp, pool);
            if is_one_like(b, pool) {
                return pool.integer(1_i32);
            }
            simplify(pool.pow(b, xp), pool).value
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let inner = fold_known_reals(args[0], pool);
            if is_zero_like(inner, pool) {
                match name.as_str() {
                    "sin" | "tan" | "sinh" => return pool.integer(0_i32),
                    "cos" | "cosh" => return pool.integer(1_i32),
                    "exp" => return pool.integer(1_i32),
                    _ => {}
                }
            }
            simplify(pool.func(name, vec![inner]), pool).value
        }
        ExprData::Func { name, args } => {
            let ys: Vec<ExprId> = args.iter().map(|x| fold_known_reals(*x, pool)).collect();
            simplify(pool.func(name, ys), pool).value
        }
        _ => e,
    }
}

fn flatten_mul(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(expr) {
        ExprData::Mul(xs) => xs.iter().flat_map(|a| flatten_mul(*a, pool)).collect(),
        _ => vec![expr],
    }
}

fn numerator_denominator(expr: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let fac = flatten_mul(expr, pool);
    let mut nums = Vec::new();
    let mut dens = Vec::new();
    for f in fac {
        match pool.get(f) {
            ExprData::Pow { base, exp } => {
                if let ExprData::Integer(n) = pool.get(exp) {
                    let nn = &n.0;
                    if *nn == 0 {
                        nums.push(pool.integer(1_i32));
                    } else if *nn > 0 {
                        nums.push(f);
                    } else {
                        let m = nn
                            .clone()
                            .abs()
                            .to_u64()
                            .and_then(|u| u32::try_from(u).ok())
                            .map(|mag| pool.pow(base, pool.integer(mag as i64)));
                        if let Some(p) = m {
                            dens.push(p);
                        } else {
                            nums.push(f);
                        }
                    }
                } else {
                    nums.push(f);
                }
            }
            _ => nums.push(f),
        }
    }
    let n = if nums.is_empty() {
        pool.integer(1_i32)
    } else if nums.len() == 1 {
        nums[0]
    } else {
        pool.mul(nums)
    };
    let d = if dens.is_empty() {
        pool.integer(1_i32)
    } else if dens.len() == 1 {
        dens[0]
    } else {
        pool.mul(dens)
    };
    (n, d)
}

fn try_expansion_limit(
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    direction: LimitDirection,
    pool: &ExprPool,
    order: u32,
) -> Result<Option<ExprId>, LimitError> {
    let exp = match local_expansion(expr, var, point, order, pool) {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };
    expansion_to_limit(exp, pool, direction)
}

fn expansion_to_limit(
    exp: LocalExpansion,
    pool: &ExprPool,
    direction: LimitDirection,
) -> Result<Option<ExprId>, LimitError> {
    let LocalExpansion {
        valuation,
        coeffs,
        h_expr: _,
    } = exp;

    let mut idx = 0usize;
    while idx < coeffs.len() && is_zero_like(coeffs[idx], pool) {
        idx += 1;
    }
    if idx >= coeffs.len() {
        // Truncation hit all zeros — indeterminate within this order.
        return Ok(None);
    }
    let power = valuation + idx as i32;
    let coeff = coeffs[idx];

    if power > 0 {
        return Ok(Some(pool.integer(0_i32)));
    }
    if power == 0 {
        return Ok(Some(coeff));
    }

    // Polar — power < 0
    let pole_order = (-power) as u32;
    let sgn_c = structural_sign(coeff, pool).unwrap_or(1);
    if pole_order % 2 == 0 {
        return Ok(Some(signed_infinity(pool, sgn_c)));
    }
    let Some(hdir) = sign_from_h(direction, power) else {
        return Err(LimitError::NeedsOneSided);
    };
    Ok(Some(signed_infinity(pool, sgn_c * hdir)))
}

/// For odd pole: sign of `h^power` with `power < 0` as `h → 0` from one side.
fn sign_from_h(direction: LimitDirection, power: i32) -> Option<i8> {
    if power >= 0 {
        return Some(1);
    }
    let odd = (-power) % 2 != 0;
    if !odd {
        return Some(1);
    }
    match direction {
        LimitDirection::Plus => Some(1),
        LimitDirection::Minus => Some(-1),
        LimitDirection::Bidirectional => None,
    }
}

fn signed_infinity(pool: &ExprPool, sign: i8) -> ExprId {
    if sign < 0 {
        neg_infinity(pool)
    } else {
        pool.pos_infinity()
    }
}

fn structural_sign(e: ExprId, pool: &ExprPool) -> Option<i8> {
    match pool.get(e) {
        ExprData::Integer(n) => {
            if n.0 > 0 {
                Some(1)
            } else if n.0 < 0 {
                Some(-1)
            } else {
                None
            }
        }
        ExprData::Rational(r) => {
            if r.0 == 0 {
                None
            } else if r.0 > 0 {
                Some(1)
            } else {
                Some(-1)
            }
        }
        ExprData::Mul(xs) => {
            let mut s = 1i8;
            for a in xs {
                let sa = structural_sign(a, pool)?;
                s *= sa;
            }
            Some(s)
        }
        ExprData::Pow { base: _, exp } if matches!(pool.get(exp), ExprData::Integer(n) if n.0.clone() % 2 == 0) => {
            Some(1)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    #[test]
    fn limit_sin_over_x_zero() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = simplify(
            p.mul(vec![p.func("sin", vec![x]), p.pow(x, p.integer(-1_i32))]),
            &p,
        )
        .value;
        let r = limit(ex, x, p.integer(0_i32), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.integer(1_i32));
    }

    #[test]
    fn limit_x_log_x_zero_plus() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = simplify(p.mul(vec![x, p.func("log", vec![x])]), &p).value;
        let r = limit(ex, x, p.integer(0_i32), LimitDirection::Plus, &p).unwrap();
        assert_eq!(r, p.integer(0_i32));
    }

    #[test]
    fn limit_exp_inf() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = p.func("exp", vec![x]);
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.pos_infinity());
    }

    #[test]
    fn limit_x_squared_at_positive_infinity() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let ex = simplify(p.pow(x, p.integer(2_i32)), &p).value;
        let r = limit(ex, x, p.pos_infinity(), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.pos_infinity(), "{}", p.display(r));
    }

    #[test]
    fn rational_x_over_x_plus_one_after_inf_subst() {
        let p = ExprPool::new();
        let t = p.symbol("__lt_inf", Domain::Real);
        let inv = p.pow(t, p.integer(-1));
        let ex = p.mul(vec![
            inv,
            p.pow(p.add(vec![p.integer(1), inv]), p.integer(-1)),
        ]);
        let folded = flatten_nested_integer_pow(ex.clone(), &p);
        let canon = canonical_polynomial_quotient_in_var(folded, t, &p);
        let r = simplify(canon, &p).value;
        let mut m = HashMap::new();
        m.insert(t, p.integer(0));
        let sub = fold_known_reals(simplify(subs(r, &m, &p), &p).value, &p);
        assert_eq!(sub, p.integer(1), "canonical={}", p.display(canon));
    }
}
