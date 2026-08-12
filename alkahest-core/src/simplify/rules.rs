use crate::deriv::log::{DerivationLog, RewriteStep, SideCondition};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use rug::ops::Pow;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Internal helper — extract numeric value (Integer or Rational) as rug::Rational
// ---------------------------------------------------------------------------

pub(super) fn as_rational(expr: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// RewriteRule trait
// ---------------------------------------------------------------------------

/// The set of node kinds a rewrite rule can possibly fire on, as a bitmask.
///
/// The rule engine tries every rule on every node, and each `apply` re-inspects
/// the node before deciding it does not match.  A rule that only rewrites
/// `Mul` nodes can declare so, and the engine skips it with a bit test.
///
/// Kinds correspond to [`ExprData`] variants; the leaf kinds that no built-in
/// rule dispatches on individually share [`NodeKinds::OTHER`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeKinds(u16);

impl NodeKinds {
    pub const ADD: NodeKinds = NodeKinds(1 << 0);
    pub const MUL: NodeKinds = NodeKinds(1 << 1);
    pub const POW: NodeKinds = NodeKinds(1 << 2);
    pub const FUNC: NodeKinds = NodeKinds(1 << 3);
    pub const INTEGER: NodeKinds = NodeKinds(1 << 4);
    pub const RATIONAL: NodeKinds = NodeKinds(1 << 5);
    pub const FLOAT: NodeKinds = NodeKinds(1 << 6);
    pub const SYMBOL: NodeKinds = NodeKinds(1 << 7);
    /// Piecewise, Predicate, Forall, Exists, BigO, RootSum.
    pub const OTHER: NodeKinds = NodeKinds(1 << 8);
    /// Every kind — the default, meaning "no useful prefilter".
    pub const ALL: NodeKinds = NodeKinds(u16::MAX);

    /// Union of two masks, usable in `const` position.
    pub const fn or(self, other: NodeKinds) -> NodeKinds {
        NodeKinds(self.0 | other.0)
    }

    /// Whether this mask admits `kind`.
    pub fn contains(self, kind: NodeKinds) -> bool {
        self.0 & kind.0 != 0
    }
}

/// The [`NodeKinds`] bit for the node `expr` currently holds.
///
/// Allocation-free, unlike [`super::discrimination_net::expr_head`], which
/// clones the node and builds a `String` for `Func`/`Symbol`.
pub fn node_kind(expr: ExprId, pool: &ExprPool) -> NodeKinds {
    pool.with(expr, |data| match data {
        ExprData::Add(_) => NodeKinds::ADD,
        ExprData::Mul(_) => NodeKinds::MUL,
        ExprData::Pow { .. } => NodeKinds::POW,
        ExprData::Func { .. } => NodeKinds::FUNC,
        ExprData::Integer(_) => NodeKinds::INTEGER,
        ExprData::Rational(_) => NodeKinds::RATIONAL,
        ExprData::Float(_) => NodeKinds::FLOAT,
        ExprData::Symbol { .. } => NodeKinds::SYMBOL,
        _ => NodeKinds::OTHER,
    })
}

pub trait RewriteRule: Send + Sync {
    fn name(&self) -> &'static str;
    /// Try to apply the rule to `expr`. Returns `None` if the rule does not match.
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)>;

    /// Node kinds this rule can rewrite.
    ///
    /// The engine skips `apply` entirely for nodes outside this mask, so an
    /// over-narrow answer silently disables the rule; debug builds assert that
    /// a skipped rule really would not have fired.  The default admits every
    /// kind, so implementations outside this crate keep working unchanged.
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::ALL
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn as_integer(expr: ExprId, pool: &ExprPool) -> Option<rug::Integer> {
    // `pool.get` would clone the whole node — for an `Integer` that is a
    // bignum allocation — only for `n.0` to be cloned again below.
    pool.with(expr, |data| match data {
        ExprData::Integer(n) => Some(n.0.clone()),
        _ => None,
    })
}

/// Test an integer literal without materialising it.
///
/// `is_zero`/`is_one` run on nearly every node visited by `MulZero`, `AddZero`
/// and `MulOne`, so going through `as_integer` cost two bignum clones per test.
fn integer_is(expr: ExprId, pool: &ExprPool, value: i32) -> bool {
    pool.with(expr, |data| match data {
        ExprData::Integer(n) => n.0 == value,
        _ => false,
    })
}

fn is_neg_imaginary_unit(expr: ExprId, pool: &ExprPool) -> bool {
    let args = pool.with(expr, |data| match data {
        ExprData::Mul(args) if args.len() == 2 => Some([args[0], args[1]]),
        _ => None,
    });
    let Some([a0, a1]) = args else {
        return false;
    };
    (as_integer(a0, pool).is_some_and(|n| n == -1) && pool.is_imaginary_unit(a1))
        || (as_integer(a1, pool).is_some_and(|n| n == -1) && pool.is_imaginary_unit(a0))
}

fn is_strictly_positive_literal(expr: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |data| match data {
        ExprData::Integer(n) => n.0 > 0,
        ExprData::Rational(r) => r.0 > 0,
        _ => false,
    })
}

fn is_positive_domain_symbol(expr: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |data| {
        matches!(
            data,
            ExprData::Symbol {
                domain: Domain::Positive,
                ..
            }
        )
    })
}

fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    integer_is(expr, pool, 0)
}

fn is_one(expr: ExprId, pool: &ExprPool) -> bool {
    integer_is(expr, pool, 1)
}

/// Whether `expr` is a literal `0` raised to a literal **negative** power.
///
/// `0^(-1)` — and every `0^(-n)`, `0^(-p/q)` — is division by zero, so it has
/// no value under any convention. A product containing such a factor is
/// therefore undefined too, and must not be folded to `0` (by `mul_zero` /
/// `const_fold`) or to `1` (by `collect_mul_factors` cancelling the negative
/// exponent against a positive one). `simplify(0^-1)` already leaves the power
/// alone; these guards make the surrounding product agree with it.
///
/// Only *literal* zero bases are recognised. Deciding whether an arbitrary
/// symbolic base vanishes is what [`crate::matrix::zero_test::zero_status`]
/// is for, and it costs several `ArbBall` evaluations at 128 bits — far too
/// much for a predicate on the hot `Mul` rewrite path. Because the engine
/// simplifies strictly bottom-up (`simplify_children` before the node itself),
/// any base the simplifier *can* reduce to zero — `x - x`, `sin(0)`,
/// `0 * y` — is already the literal `0` node by the time these rules see the
/// product, so the literal test covers those too. For a base that is not
/// provably zero, cancelling `b · b⁻¹ → 1` asserts `b ≠ 0`, which is the
/// library's documented convention (`simplify_control_cancel_x_over_x`).
fn is_zero_to_negative_power(expr: ExprId, pool: &ExprPool) -> bool {
    let parts = pool.with(expr, |data| match data {
        ExprData::Pow { base, exp } => Some((*base, *exp)),
        _ => None,
    });
    match parts {
        Some((base, exp)) => is_zero(base, pool) && is_negative_literal(exp, pool),
        None => false,
    }
}

/// Whether `expr` is, or is a product containing, a literal `0` to a negative
/// literal power — i.e. whether `expr` is undefined for that reason.
///
/// Only the top level and its immediate `Mul` factors are inspected: the
/// simplifier normalises bottom-up and flattens `Mul` nodes, so an undefined
/// factor of a product is a direct child by the time a `Mul`/`Add` rule sees
/// it. Callers use this to decline a rewrite, so a missed deeper occurrence
/// costs nothing beyond the existing behaviour.
fn has_zero_to_negative_power_factor(expr: ExprId, pool: &ExprPool) -> bool {
    if is_zero_to_negative_power(expr, pool) {
        return true;
    }
    pool.with(expr, |data| match data {
        ExprData::Mul(args) => args.clone(),
        _ => Vec::new(),
    })
    .into_iter()
    .any(|a| is_zero_to_negative_power(a, pool))
}

/// Whether `expr` is a negative `Integer` or `Rational` literal.
fn is_negative_literal(expr: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |data| match data {
        ExprData::Integer(n) => n.0 < 0,
        ExprData::Rational(r) => r.0 < 0,
        _ => false,
    })
}

pub(crate) fn one_step(name: &'static str, before: ExprId, after: ExprId) -> DerivationLog {
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(name, before, after));
    log
}

fn one_step_with(
    name: &'static str,
    before: ExprId,
    after: ExprId,
    conds: Vec<SideCondition>,
) -> DerivationLog {
    let mut log = DerivationLog::new();
    log.push(RewriteStep::with_conditions(name, before, after, conds));
    log
}

/// Extract (integer_coeff, base) from a Mul where some factors are integers.
/// Returns (1, expr) if no integer factor is found.
pub(super) fn extract_int_coeff(expr: ExprId, pool: &ExprPool) -> (rug::Integer, ExprId) {
    match pool.get(expr) {
        ExprData::Integer(n) => (n.0.clone(), pool.integer(1_i32)),
        ExprData::Mul(args) => {
            let mut int_product = rug::Integer::from(1);
            let mut non_ints: Vec<ExprId> = vec![];
            for &a in &args {
                match pool.with(a, |d| match d {
                    ExprData::Integer(n) => Some(n.0.clone()),
                    _ => None,
                }) {
                    Some(n) => int_product *= n,
                    None => non_ints.push(a),
                }
            }
            if non_ints.len() == args.len() {
                // No integer factors found
                return (rug::Integer::from(1), expr);
            }
            let base = match non_ints.len() {
                0 => pool.integer(1_i32),
                1 => non_ints[0],
                _ => pool.mul(non_ints),
            };
            (int_product, base)
        }
        _ => (rug::Integer::from(1), expr),
    }
}

// ---------------------------------------------------------------------------
// Imaginary unit `i = √(−1)` — pure algebraic power rules.
//
// `i^n` for a literal integer `n` cycles with period 4:
//   i^(4k+0) = 1,  i^(4k+1) = i,  i^(4k+2) = −1,  i^(4k+3) = −i.
// These rules are purely algebraic — no branch cuts, no `√(−1) → i`, no
// `log`/`exp` complex identities. They ride [`ConstFold`]'s existing `Pow`
// and `Mul` dispatch arms (and the cheap egraph post-pass), keeping every
// check literal-only so CodSpeed-sensitive paths stay fast.
// ---------------------------------------------------------------------------

/// If `expr` is an imaginary-unit power factor — the bare unit `i` (exponent
/// `1`) or `i^k` for a literal integer `k` — return its integer exponent `k`.
/// Returns `None` otherwise (including `i^(1/2)` and other non-integer powers,
/// which carry branch-cut ambiguity we deliberately do not touch).
fn imaginary_unit_exp(expr: ExprId, pool: &ExprPool) -> Option<rug::Integer> {
    if pool.is_imaginary_unit(expr) {
        return Some(rug::Integer::from(1));
    }
    match pool.get(expr) {
        ExprData::Pow { base, exp } if pool.is_imaginary_unit(base) => as_integer(exp, pool),
        _ => None,
    }
}

/// Build `i^r` in fully reduced form for `r = n mod 4 ∈ {0,1,2,3}`:
/// `1`, `i`, `−1`, `−i` respectively.
fn imaginary_unit_pow_residue(residue: u32, pool: &ExprPool) -> ExprId {
    let i = pool.imaginary_unit();
    match residue {
        0 => pool.integer(1_i32),
        1 => i,
        2 => pool.integer(-1_i32),
        // 3 → −i
        _ => pool.mul(vec![pool.integer(-1_i32), i]),
    }
}

/// Non-negative residue of `n mod 4` (rug's `Integer` remainder is truncating,
/// so adjust for negative `n` — exponents may be negative, e.g. `i⁻¹ = −i`).
fn mod4_nonneg(n: &rug::Integer) -> u32 {
    let mut r = rug::Integer::from(n % 4);
    if r < 0 {
        r += 4;
    }
    // r ∈ {0,1,2,3}, always fits in u32.
    r.to_u32().unwrap_or(0)
}

/// Extract (integer_exponent, base) for use in DivSelf.
/// Returns `Some((1, expr))` for all terms including integer constants so
/// that `n * n^(-1) → 1` is handled correctly.
/// Returns (n, base) for `Pow(base, Integer(n))`.
fn extract_int_exp(expr: ExprId, pool: &ExprPool) -> Option<(rug::Integer, ExprId)> {
    match pool.get(expr) {
        // Integer n is treated as n^1 so that n * n^(-1) can cancel.
        ExprData::Integer(_) => Some((rug::Integer::from(1), expr)),
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(n) => Some((n.0.clone(), base)),
            _ => Some((rug::Integer::from(1), expr)),
        },
        _ => Some((rug::Integer::from(1), expr)),
    }
}

fn rebuild_coeff_term(coeff: &rug::Integer, base: ExprId, pool: &ExprPool) -> ExprId {
    if is_one(base, pool) {
        // base is Integer(1)
        pool.integer(coeff.clone())
    } else if *coeff == 1 {
        base
    } else {
        pool.mul(vec![pool.integer(coeff.clone()), base])
    }
}

fn rebuild_exp_term(exp: &rug::Integer, base: ExprId, pool: &ExprPool) -> ExprId {
    if *exp == 1 {
        base
    } else {
        pool.pow(base, pool.integer(exp.clone()))
    }
}

// ---------------------------------------------------------------------------
// AddZero: remove Integer(0) from Add args
// ---------------------------------------------------------------------------

pub struct AddZero;

impl RewriteRule for AddZero {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::ADD
    }
    fn name(&self) -> &'static str {
        "add_zero"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };
        if !args.iter().any(|&a| is_zero(a, pool)) {
            return None;
        }
        let filtered: Vec<ExprId> = args.into_iter().filter(|&a| !is_zero(a, pool)).collect();
        let after = match filtered.len() {
            0 => pool.integer(0_i32),
            1 => filtered[0],
            _ => pool.add(filtered),
        };
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// MulOne: remove Integer(1) from Mul args
// ---------------------------------------------------------------------------

pub struct MulOne;

impl RewriteRule for MulOne {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::MUL
    }
    fn name(&self) -> &'static str {
        "mul_one"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Mul(v) => v,
            _ => return None,
        };
        if !args.iter().any(|&a| is_one(a, pool)) {
            return None;
        }
        let filtered: Vec<ExprId> = args.into_iter().filter(|&a| !is_one(a, pool)).collect();
        let after = match filtered.len() {
            0 => pool.integer(1_i32),
            1 => filtered[0],
            _ => pool.mul(filtered),
        };
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// MulZero: x * 0 → 0
// ---------------------------------------------------------------------------

pub struct MulZero;

impl RewriteRule for MulZero {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::MUL
    }
    fn name(&self) -> &'static str {
        "mul_zero"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Mul(v) => v,
            _ => return None,
        };
        if !args.iter().any(|&a| is_zero(a, pool)) {
            return None;
        }
        // Do not fold `0 * 0^(-1) * ...` (or `0 * 0^(-2)`, etc.) to `0`: a
        // literal `0^(negative)` factor is itself undefined (division by
        // zero), so the product is indeterminate, not `0`. This is the
        // n=0 boundary of `0 * x^(-1)` being indeterminate at x=0.
        if args.iter().any(|&a| is_zero_to_negative_power(a, pool)) {
            return None;
        }
        let after = pool.integer(0_i32);
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// PowOne: x^1 → x
// ---------------------------------------------------------------------------

pub struct PowOne;

impl RewriteRule for PowOne {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::POW
    }
    fn name(&self) -> &'static str {
        "pow_one"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (base, exp) = match pool.get(expr) {
            ExprData::Pow { base, exp } => (base, exp),
            _ => return None,
        };
        if !is_one(exp, pool) {
            return None;
        }
        Some((base, one_step(self.name(), expr, base)))
    }
}

// ---------------------------------------------------------------------------
// SqrtInteger: sqrt(n) → m when n is a perfect square (n, m > 0)
// ---------------------------------------------------------------------------

pub struct SqrtInteger;

impl RewriteRule for SqrtInteger {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::FUNC
    }
    fn name(&self) -> &'static str {
        "sqrt_integer"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let arg = match pool.get(expr) {
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => args[0],
            _ => return None,
        };
        let n = as_integer(arg, pool)?;
        if n <= 0 {
            return None;
        }
        let n_u = n.to_u64()?;
        let root = integer_sqrt_u64(n_u)?;
        if root * root != n_u {
            return None;
        }
        let after = pool.integer(i64::try_from(root).ok()?);
        Some((after, one_step(self.name(), expr, after)))
    }
}

fn integer_sqrt_u64(n: u64) -> Option<u64> {
    if n == 0 {
        return Some(0);
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// PowZero: x^0 → 1  (side condition: x ≠ 0 logged)
// ---------------------------------------------------------------------------

pub struct PowZero;

impl RewriteRule for PowZero {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::POW
    }
    fn name(&self) -> &'static str {
        "pow_zero"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (base, exp) = match pool.get(expr) {
            ExprData::Pow { base, exp } => (base, exp),
            _ => return None,
        };
        if !is_zero(exp, pool) {
            return None;
        }
        // 0^0 is undefined — do not rewrite.
        if is_zero(base, pool) {
            return None;
        }
        let after = pool.integer(1_i32);
        Some((
            after,
            one_step_with(self.name(), expr, after, vec![SideCondition::NonZero(base)]),
        ))
    }
}

// ---------------------------------------------------------------------------
// ConstFold: numeric folding for Add/Mul (partial), Pow (integer exponents,
// power-of-power, and distribution over a literal coefficient), Func
// (elementary functions at exact literal arguments), and Rational
// (denominator-1 canonicalization).
//
// All of the `Func`/`Pow` sub-cases below were originally separate rules
// (`ElementaryAtConst`, `PowOfPow`, `DistributePowOverLiteralCoeff`,
// `EvenPowerSignFold`, `RationalCanon`); they are folded into ConstFold's
// existing per-node dispatch so the rule-iteration loop in `simplify_node`
// does not pay a separate per-node check for each of them. Each sub-case
// retains its original soundness argument (see git history / PR description
// for the detailed proofs):
//
//   Func, single arg, elementary functions at 0/1:
//     exp(0)→1  sin(0)→0  cos(0)→1  sinh(0)→0  cosh(0)→1  tan(0)→0
//     atan(0)→0  asin(0)→0  log(1)→0  ln(1)→0
//     (exact values strictly inside the domain of analyticity — no branch
//     cuts, no poles — sound regardless of sign/positivity of other symbols)
//
//   Pow:
//     1^r → 1                                  (any literal rational r)
//     (-1·x)^n → x^n                           (literal even integer n)
//     (x^a)^b → x^(a·b)                        (literal integer a, b)
//     (c·rest)^n → c^n · rest^n                (literal integer c ≠ 0,±1, n)
//     b^e → integer/rational fold              (literal integer base/exp)
//
//   Rational(n/1) → Integer(n)
// ---------------------------------------------------------------------------

fn intern_rational(r: rug::Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.into_numer_denom().0)
    } else {
        pool.intern(ExprData::Rational(crate::kernel::expr::BigRat(r)))
    }
}

pub struct ConstFold;

impl RewriteRule for ConstFold {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::ADD
            .or(NodeKinds::MUL)
            .or(NodeKinds::POW)
            .or(NodeKinds::FUNC)
            .or(NodeKinds::RATIONAL)
    }
    fn name(&self) -> &'static str {
        "const_fold"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        match pool.get(expr) {
            ExprData::Add(args) => {
                let numeric_count = args
                    .iter()
                    .filter(|&&a| as_rational(a, pool).is_some())
                    .count();
                if numeric_count < 2 {
                    return None;
                }
                let mut sum = rug::Rational::from(0);
                let mut non_numeric: Vec<ExprId> = vec![];
                for &a in &args {
                    match as_rational(a, pool) {
                        Some(r) => sum += r,
                        None => non_numeric.push(a),
                    }
                }
                let after = if non_numeric.is_empty() {
                    intern_rational(sum, pool)
                } else if sum == 0 {
                    match non_numeric.len() {
                        1 => non_numeric[0],
                        _ => pool.add(non_numeric),
                    }
                } else {
                    let mut new_args = vec![intern_rational(sum, pool)];
                    new_args.extend(non_numeric);
                    pool.add(new_args)
                };
                if after == expr {
                    return None;
                }
                Some((after, one_step(self.name(), expr, after)))
            }
            ExprData::Mul(args) => {
                // Count how many factors are imaginary-unit powers (the bare
                // unit `i`, or `i^k` for literal integer `k`). When ≥2 such
                // factors are present they collapse via i² = −1 (e.g. i·i → −1,
                // (2i)·(3i) → −6); a single one is left untouched. The check is
                // O(args) of O(1) probes — cheap and fails fast.
                let imag_factor_count = args
                    .iter()
                    .filter(|&&a| imaginary_unit_exp(a, pool).is_some())
                    .count();
                let numeric_count = args
                    .iter()
                    .filter(|&&a| as_rational(a, pool).is_some())
                    .count();
                if numeric_count < 2 && imag_factor_count < 2 {
                    return None;
                }
                let mut prod = rug::Rational::from(1);
                // Total imaginary-unit exponent collected from i / i^k factors.
                let mut imag_exp = rug::Integer::from(0);
                let mut non_numeric: Vec<ExprId> = vec![];
                for &a in &args {
                    if let Some(r) = as_rational(a, pool) {
                        prod *= r;
                    } else if let Some(k) = imaginary_unit_exp(a, pool) {
                        imag_exp += k;
                    } else {
                        non_numeric.push(a);
                    }
                }
                // Collapse the accumulated i^(imag_exp) into {1, i, −1, −i}.
                // The sign (−1 for residues 2,3) folds into the rational
                // product; the residual `i` (residues 1,3) becomes a factor.
                match mod4_nonneg(&imag_exp) {
                    0 => {}
                    1 => non_numeric.push(pool.imaginary_unit()),
                    2 => prod *= -1,
                    _ => {
                        // residue 3 → −i
                        prod *= -1;
                        non_numeric.push(pool.imaginary_unit());
                    }
                }
                let after = if prod == 0 {
                    // Same guard as `mul_zero`: `0 * 0^(-1) * 5` is
                    // undefined, not `0`. The undefined factor is never
                    // numeric, so it lands in `non_numeric`.
                    if non_numeric
                        .iter()
                        .any(|&a| is_zero_to_negative_power(a, pool))
                    {
                        return None;
                    }
                    pool.integer(0_i32)
                } else if non_numeric.is_empty() {
                    intern_rational(prod, pool)
                } else if prod == 1 {
                    match non_numeric.len() {
                        1 => non_numeric[0],
                        _ => pool.mul(non_numeric),
                    }
                } else {
                    let mut new_args = vec![intern_rational(prod, pool)];
                    new_args.extend(non_numeric);
                    pool.mul(new_args)
                };
                if after == expr {
                    return None;
                }
                Some((after, one_step(self.name(), expr, after)))
            }
            ExprData::Pow { base, exp } => {
                // i^n → {1, i, −1, −i} for a literal integer exponent n,
                // cycling with period 4 (ImaginaryUnitPow). Pure algebra; no
                // branch cuts. Cheap discriminant: the base must be the
                // canonical imaginary unit and the exponent a literal integer,
                // both O(1) checks that fail fast for ordinary powers.
                if pool.is_imaginary_unit(base) {
                    if let Some(n) = as_integer(exp, pool) {
                        let after = imaginary_unit_pow_residue(mod4_nonneg(&n), pool);
                        if after != expr {
                            return Some((after, one_step(self.name(), expr, after)));
                        }
                    }
                }

                // 1^r = 1 for any literal rational (or integer) exponent `r`,
                // including non-integer exponents like 1/2. This is sound
                // unconditionally: 1^r = exp(r * log(1)) = exp(r * 0) = 1
                // under the principal branch, with no branch-cut ambiguity.
                if is_one(base, pool) && as_rational(exp, pool).is_some() {
                    let after = pool.integer(1_i32);
                    if after == expr {
                        return None;
                    }
                    return Some((after, one_step(self.name(), expr, after)));
                }

                // (-1·x)^n → x^n for literal even integer n (EvenPowerSignFold).
                // Cheap discriminant: exp must be an integer literal and base
                // must be a Mul — both checks are O(1) and fail fast for the
                // overwhelmingly common case of a plain symbol/atom base.
                if let Some(after) = even_power_sign_fold(base, exp, pool) {
                    if after != expr {
                        return Some((after, one_step(self.name(), expr, after)));
                    }
                }

                // (x^a)^b → x^(a·b) for literal integer a, b (PowOfPow).
                // Cheap discriminant: base must itself be a Pow node.
                if let ExprData::Pow {
                    base: inner_base,
                    exp: inner_exp,
                } = pool.get(base)
                {
                    if let (Some(a), Some(b)) = (as_integer(inner_exp, pool), as_integer(exp, pool))
                    {
                        let new_exp = pool.integer(a * b);
                        let after = pool.pow(inner_base, new_exp);
                        if after != expr {
                            return Some((after, one_step(self.name(), expr, after)));
                        }
                    }
                }

                // (c·rest)^n → c^n · rest^n for a literal integer coefficient
                // c (!= 0, ±1) and literal integer exponent n
                // (DistributePowOverLiteralCoeff). This is the key step that
                // lets `π · (4π)^(-1)` reduce to `π · 4^(-1) · π^(-1)`, which
                // `DivSelf` and the b^e fold below then collapse to `1/4`.
                if let Some(n) = as_integer(exp, pool) {
                    if pool.with(base, |d| matches!(d, ExprData::Mul(_))) {
                        let (coeff, rest) = extract_int_coeff(base, pool);
                        if coeff != 1 && coeff != -1 && coeff != 0 && rest != pool.integer(1_i32) {
                            let coeff_pow = pool.pow(pool.integer(coeff), pool.integer(n.clone()));
                            let rest_pow = pool.pow(rest, pool.integer(n));
                            let after = pool.mul(vec![coeff_pow, rest_pow]);
                            if after != expr {
                                return Some((after, one_step(self.name(), expr, after)));
                            }
                        }
                    }
                }

                let b = as_integer(base, pool)?;
                let e = as_integer(exp, pool)?;
                // 1^e = 1 and (-1)^e = ±1 for any integer e (including negative)
                if b == 1 {
                    let after = pool.integer(1_i32);
                    if after == expr {
                        return None;
                    }
                    return Some((after, one_step(self.name(), expr, after)));
                }
                if b == -1 {
                    let sign: i64 = if e.is_even() { 1 } else { -1 };
                    let after = pool.integer(sign);
                    if after == expr {
                        return None;
                    }
                    return Some((after, one_step(self.name(), expr, after)));
                }
                if e < 0 {
                    // b^e for nonzero integer base `b` and negative integer
                    // exponent `e` is the rational `1 / b^|e|`. Sound for any
                    // nonzero b (b == 0, ±1 handled above / 0 excluded since
                    // 0^(negative) is undefined and `as_integer` would give 0
                    // only for base literal 0, which we reject here).
                    if b == 0 {
                        return None; // 0^(negative) undefined
                    }
                    let e_u32 = (-e.clone()).to_u32()?;
                    let denom: rug::Integer = b.pow(e_u32);
                    let result = rug::Rational::from((rug::Integer::from(1), denom));
                    let after = intern_rational(result, pool);
                    if after == expr {
                        return None;
                    }
                    return Some((after, one_step(self.name(), expr, after)));
                }
                let e_u32 = e.to_u32()?;
                let result: rug::Integer = b.pow(e_u32);
                let after = pool.integer(result);
                if after == expr {
                    return None;
                }
                Some((after, one_step(self.name(), expr, after)))
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                // Elementary functions at exact literal arguments
                // (ElementaryAtConst).
                let arg = args[0];
                let after = match name.as_str() {
                    "conjugate" if pool.is_imaginary_unit(arg) => {
                        pool.mul(vec![pool.integer(-1_i32), arg])
                    }
                    "conjugate" => match pool.get(arg) {
                        ExprData::Func {
                            name: inner,
                            args: inner_args,
                        } if inner == "conjugate" && inner_args.len() == 1 => inner_args[0],
                        ExprData::Integer(_) | ExprData::Rational(_) => arg,
                        _ => return None,
                    },
                    "re" if matches!(
                        pool.get(arg),
                        ExprData::Integer(_) | ExprData::Rational(_)
                    ) =>
                    {
                        arg
                    }
                    "im" if matches!(
                        pool.get(arg),
                        ExprData::Integer(_) | ExprData::Rational(_)
                    ) =>
                    {
                        pool.integer(0_i32)
                    }
                    // Principal Arg ∈ (−π, π]: only literal/domain-safe cases.
                    // Leave arg(0), negative reals, and generic complex inputs
                    // unevaluated — no atan2/log/sqrt rewrites.
                    "arg" => {
                        let pi = pool.symbol("pi", Domain::Positive);
                        let half_pi = pool.mul(vec![pool.rational(1, 2), pi]);
                        if pool.is_imaginary_unit(arg) {
                            half_pi
                        } else if is_neg_imaginary_unit(arg, pool) {
                            pool.mul(vec![pool.integer(-1_i32), half_pi])
                        } else if is_strictly_positive_literal(arg, pool)
                            || is_positive_domain_symbol(arg, pool)
                        {
                            pool.integer(0_i32)
                        } else {
                            return None;
                        }
                    }
                    "exp" if is_zero(arg, pool) => pool.integer(1_i32),
                    "cos" if is_zero(arg, pool) => pool.integer(1_i32),
                    "cosh" if is_zero(arg, pool) => pool.integer(1_i32),
                    "sin" | "sinh" | "tan" | "atan" | "asin" if is_zero(arg, pool) => {
                        pool.integer(0_i32)
                    }
                    "log" | "ln" if is_one(arg, pool) => pool.integer(0_i32),
                    _ => return None,
                };
                if after == expr {
                    return None;
                }
                Some((after, one_step(self.name(), expr, after)))
            }
            // Rational(n/1) → Integer(n) (RationalCanon).
            //
            // `ExprPool::rational` reduces to lowest terms but does not
            // collapse a denominator of 1 to an `Integer` node — such nodes
            // can also arise from un-collapsed arithmetic (see PR #147).
            // Canonicalizing here ensures `Rational` nodes always have
            // denominator > 1, simplifying downstream pattern matches (e.g.
            // `as_integer`, polynomial coefficient extraction). Always sound:
            // the value is unchanged, only the representation changes.
            ExprData::Rational(r) if *r.0.denom() == 1 => {
                let after = pool.integer(r.0.numer().clone());
                Some((after, one_step(self.name(), expr, after)))
            }
            _ => None,
        }
    }
}

/// Helper for `ConstFold`'s `(-1·x)^n → x^n` fold (literal even integer `n`).
/// Returns `None` if the pattern doesn't match (cheap discriminant checks
/// fail fast for the common case).
fn even_power_sign_fold(base: ExprId, exp: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let n = as_integer(exp, pool)?;
    if !n.is_even() || n == 0 {
        return None;
    }
    let args = match pool.get(base) {
        ExprData::Mul(v) => v,
        _ => return None,
    };
    // Find a literal -1 factor.
    let neg_pos = args
        .iter()
        .position(|&a| as_integer(a, pool).is_some_and(|i| i == -1))?;
    let rest: Vec<ExprId> = args
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != neg_pos)
        .map(|(_, &a)| a)
        .collect();
    let new_base = match rest.len() {
        0 => pool.integer(1_i32),
        1 => rest[0],
        _ => pool.mul(rest),
    };
    Some(pool.pow(new_base, exp))
}

// ---------------------------------------------------------------------------
// SubSelf: collect like terms in Add; handles x - x → 0
// ---------------------------------------------------------------------------

pub struct SubSelf;

impl RewriteRule for SubSelf {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::ADD
    }
    fn name(&self) -> &'static str {
        "collect_add_terms"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };
        if args.len() < 2 {
            return None;
        }

        // Extract (coeff, base) for each arg
        let pairs: Vec<(rug::Integer, ExprId)> =
            args.iter().map(|&a| extract_int_coeff(a, pool)).collect();

        // Sum coefficients by base, preserving first-occurrence order
        let mut coeff_map: HashMap<ExprId, rug::Integer> = HashMap::new();
        let mut base_order: Vec<ExprId> = vec![];
        for (coeff, base) in &pairs {
            if !coeff_map.contains_key(base) {
                base_order.push(*base);
                coeff_map.insert(*base, rug::Integer::from(0));
            }
            *coeff_map.get_mut(base).unwrap() += coeff.clone();
        }

        // Check: any cancellation (coeff → 0) or merging (two args same base)?
        let any_zero = coeff_map.values().any(|c| *c == 0);
        let any_merged = coeff_map.len() < pairs.len();
        if !any_zero && !any_merged {
            return None;
        }

        // Dropping a term whose integer coefficient sums to `0` asserts that
        // the term's remaining factor is a *number* — `0 · u = 0` is false
        // when `u` is undefined. `diff(2/(x - x), x)` lands here as
        // `(0 · 0⁻¹) + (2 · −1 · 0 · 0⁻²)`, where both coefficients are the
        // literal `0` that came out of the numerator, and dropping both
        // reported a derivative of `0` for an expression that has none.
        // Only checked when something actually cancels, so ordinary
        // `x - x → 0` collection is untouched.
        if any_zero
            && coeff_map
                .iter()
                .any(|(base, c)| *c == 0 && has_zero_to_negative_power_factor(*base, pool))
        {
            return None;
        }

        // Build new args
        let mut new_args: Vec<ExprId> = vec![];
        let mut seen: HashSet<ExprId> = HashSet::new();
        for base in &base_order {
            if seen.contains(base) {
                continue;
            }
            seen.insert(*base);
            let coeff = &coeff_map[base];
            if *coeff == 0 {
                continue;
            }
            new_args.push(rebuild_coeff_term(coeff, *base, pool));
        }

        let after = match new_args.len() {
            0 => pool.integer(0_i32),
            1 => new_args[0],
            _ => pool.add(new_args),
        };
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// DivSelf: collect like factors in Mul; handles x / x → 1
// ---------------------------------------------------------------------------

pub struct DivSelf;

impl RewriteRule for DivSelf {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::MUL
    }
    fn name(&self) -> &'static str {
        "collect_mul_factors"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Mul(v) => v,
            _ => return None,
        };
        if args.len() < 2 {
            return None;
        }

        let globally_comm = args
            .iter()
            .all(|&a| crate::kernel::expr_props::mult_tree_is_commutative(pool, a));

        // Collect (integer exponent, base) for each factor.
        let mut exp_pairs: Vec<(rug::Integer, ExprId)> = vec![];
        for &a in &args {
            if let Some(pair) = extract_int_exp(a, pool) {
                exp_pairs.push(pair);
            }
        }
        if exp_pairs.len() < 2 {
            return None;
        }

        // Summing exponents of a common base is `b^k · b^m = b^(k+m)`, an
        // identity that fails for `b = 0` as soon as one exponent is
        // negative: `0^1 · 0^(-1)` is `0 · (1/0)`, undefined, while the
        // merged `0^0` would be `1`. `simplify(0^-1)` already declines to
        // give the undefined power a value; decline here too rather than
        // invent one for the product. The literal check is one sign test per
        // factor plus an `O(1)` node probe on the (rare) negative ones — see
        // `is_zero_to_negative_power` for why a full three-valued zero test
        // is not affordable on this path.
        if exp_pairs.iter().any(|(e, b)| *e < 0 && is_zero(*b, pool)) {
            return None;
        }

        let new_args: Vec<ExprId> = if globally_comm {
            // Commutative: sum exponents for each base anywhere in the product.
            let mut exp_map: HashMap<ExprId, rug::Integer> = HashMap::new();
            let mut base_order: Vec<ExprId> = vec![];
            for (exp, base) in &exp_pairs {
                if !exp_map.contains_key(base) {
                    base_order.push(*base);
                    exp_map.insert(*base, rug::Integer::from(0));
                }
                *exp_map.get_mut(base).unwrap() += exp.clone();
            }

            let any_zero = exp_map.values().any(|e| *e == 0);
            let any_merged = exp_map.len() < exp_pairs.len();
            if !any_zero && !any_merged {
                return None;
            }

            let mut seen: HashSet<ExprId> = HashSet::new();
            let mut new_args: Vec<ExprId> = vec![];
            for base in &base_order {
                if seen.contains(base) {
                    continue;
                }
                seen.insert(*base);
                let exp = &exp_map[base];
                if *exp == 0 {
                    continue;
                }
                new_args.push(rebuild_exp_term(exp, *base, pool));
            }
            new_args
        } else {
            // Non-commutative: only merge **consecutive** identical bases (V3-2).
            let mut merged: Vec<(rug::Integer, ExprId)> = vec![];
            let mut changed = false;
            for (e, b) in exp_pairs {
                if let Some((last_e, last_b)) = merged.last_mut() {
                    if *last_b == b {
                        *last_e += e;
                        changed = true;
                        continue;
                    }
                }
                merged.push((e, b));
            }
            let any_zero = merged.iter().any(|(e, _)| *e == 0);
            if !changed && !any_zero {
                return None;
            }
            merged
                .into_iter()
                .filter(|(e, _)| *e != 0)
                .map(|(e, b)| rebuild_exp_term(&e, b, pool))
                .collect()
        };

        let after = match new_args.len() {
            0 => pool.integer(1_i32),
            1 => new_args[0],
            _ => pool.mul(new_args),
        };
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// FlattenMul/FlattenAdd: flatten nested n-ary nodes
//   Mul([a, Mul([b, c]), d]) → Mul([a, b, c, d])
//   Add([a, Add([b, c]), d]) → Add([a, b, c, d])
// ---------------------------------------------------------------------------

pub struct FlattenMul;

impl RewriteRule for FlattenMul {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::MUL
    }
    fn name(&self) -> &'static str {
        "flatten_mul"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Mul(v) => v,
            _ => return None,
        };
        let mut flat = Vec::new();
        let mut changed = false;
        for &a in &args {
            // Borrow the child: cloning it here allocated a `Vec` per child on
            // every visit, and these two rules are tried on every node.
            let nested = pool.with(a, |d| match d {
                ExprData::Mul(inner) => Some(inner.clone()),
                _ => None,
            });
            match nested {
                Some(inner) => {
                    flat.extend_from_slice(&inner);
                    changed = true;
                }
                None => flat.push(a),
            }
        }
        if !changed {
            return None;
        }
        let after = pool.mul(flat);
        Some((after, one_step(self.name(), expr, after)))
    }
}

pub struct FlattenAdd;

impl RewriteRule for FlattenAdd {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::ADD
    }
    fn name(&self) -> &'static str {
        "flatten_add"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Add(v) => v,
            _ => return None,
        };
        let mut flat = Vec::new();
        let mut changed = false;
        for &a in &args {
            // Borrow the child: cloning it here allocated a `Vec` per child on
            // every visit, and these two rules are tried on every node.
            let nested = pool.with(a, |d| match d {
                ExprData::Add(inner) => Some(inner.clone()),
                _ => None,
            });
            match nested {
                Some(inner) => {
                    flat.extend_from_slice(&inner);
                    changed = true;
                }
                None => flat.push(a),
            }
        }
        if !changed {
            return None;
        }
        let after = pool.add(flat);
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// CanonicalOrder: sort Add/Mul args by ExprId for determinism
// ---------------------------------------------------------------------------

pub struct CanonicalOrder;

impl RewriteRule for CanonicalOrder {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::ADD.or(NodeKinds::MUL)
    }
    fn name(&self) -> &'static str {
        "canonical_order"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        match pool.get(expr) {
            ExprData::Add(args) => {
                let mut sorted = args.clone();
                sorted.sort_unstable();
                if sorted == args {
                    return None;
                }
                let after = pool.add(sorted);
                Some((after, one_step(self.name(), expr, after)))
            }
            ExprData::Mul(args) => {
                if !args
                    .iter()
                    .all(|&a| crate::kernel::expr_props::mult_tree_is_commutative(pool, a))
                {
                    return None;
                }
                let mut sorted = args.clone();
                sorted.sort_unstable();
                if sorted == args {
                    return None;
                }
                let after = pool.mul(sorted);
                Some((after, one_step(self.name(), expr, after)))
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ExpandMul: distribute multiplication over addition
//   (a + b) * c → a*c + b*c
//
// Only fires when at least one Mul argument is an Add. Gate behind
// SimplifyConfig::expand (default off) to avoid interfering with a future
// factor/collect rule.
// ---------------------------------------------------------------------------

pub struct ExpandMul;

impl RewriteRule for ExpandMul {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::MUL
    }
    fn name(&self) -> &'static str {
        "expand_mul"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Mul(v) => v,
            _ => return None,
        };

        // Find the first Add factor
        let add_pos = args
            .iter()
            .position(|&a| pool.with(a, |d| matches!(d, ExprData::Add(_))))?;

        let add_args = match pool.get(args[add_pos]) {
            ExprData::Add(v) => v,
            _ => return None,
        };

        // The remaining (non-add) factors become the common multiplier
        let other: Vec<ExprId> = args
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != add_pos)
            .map(|(_, &a)| a)
            .collect();

        // Distribute: each summand gets multiplied by `other`
        let new_summands: Vec<ExprId> = add_args
            .into_iter()
            .map(|summand| {
                let mut factors = vec![summand];
                factors.extend_from_slice(&other);
                match factors.len() {
                    1 => factors[0],
                    _ => pool.mul(factors),
                }
            })
            .collect();

        let after = match new_summands.len() {
            0 => pool.integer(0_i32),
            1 => new_summands[0],
            _ => pool.add(new_summands),
        };

        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// ExpandPow: (a + b + …)^n → fully distributed Σ of monomials
//
// for a literal positive integer exponent `n` within a small cap. This is the
// companion to `ExpandMul`: it unfolds a power of a sum directly into the flat
// polynomial expansion. Gated behind `SimplifyConfig::expand` (default off).
//
// IMPORTANT — termination: the result is an `Add` of products, never a `Mul`
// of repeated identical `Add` factors. Emitting `(a+b)·(a+b)` here would
// immediately be re-collapsed to `(a+b)^2` by `DivSelf`
// (`collect_mul_factors`), and the two rules would oscillate forever. By
// distributing all the way to monomials in one step, each summand is a distinct
// product whose factors `collect_mul_factors` may merge (`a·a → a²`) without
// ever reconstructing the original `(Add)^n`, so the fixed point is reached.
//
// The work is capped so a stray large literal exponent cannot trigger
// combinatorial blow-up — but a cap that is silently a no-op is its own defect,
// so declining is *recorded* (`take_expand_limits`) and surfaces as a step in
// the derivation log.
// ---------------------------------------------------------------------------

/// Exponent below which [`ExpandPow`] always unfolds, whatever the width of the
/// base.
///
/// Historically the *whole* bound: exponent ≤ 4, any number of summands. Kept as
/// a floor so nothing that expanded before stops expanding, but it is a poor
/// bound on its own — it permits `(a₁+…+a₂₀)⁴` (160 000 products) while refusing
/// `(x+y)⁵` (32). [`MAX_EXPAND_POW_PRODUCTS`] is the bound that actually
/// describes the work.
const MAX_EXPAND_POW_EXP: u32 = 4;

/// Maximum number of distributed products [`ExpandPow`] will form: a base of
/// `m` summands raised to `n` produces `mⁿ` of them before like terms are
/// collected.
///
/// Raising the old exponent-only cap was cheap for the shapes that matter — a
/// binomial now expands to the 12th power (4096 products) where it used to stop
/// at the 4th, and `(x+y+1)⁷` (2187) where it used to stop at `(x+y+1)⁴`.
/// Measured on `simplify_expanded` (release build): the whole pass — distribute,
/// collect, constant-fold — costs 1.2 ms at 64 products, 24 ms at 1024 and
/// ~100 ms at this budget, i.e. roughly linear in the product count with the
/// *collection* cost (the `n^5.7` term the 3.8 performance audit measured for
/// this route) taking over at the top end. That is the reason not to go higher.
///
/// Beyond it the honest answer is `poly_normal`, which is `n^2.1` for the same
/// result; the recorded log step says so rather than leaving the caller with an
/// unexpanded expression and no explanation.
const MAX_EXPAND_POW_PRODUCTS: u64 = 4096;

/// How many declined powers one pass will report.  A pass that hits the bound
/// on hundreds of distinct nodes has said everything useful in the first few,
/// and the cap bounds the recorder on engines that run rules without draining
/// it (`simplify_par`'s rayon workers each have their own copy of this
/// thread-local, and no one collects theirs).
const MAX_RECORDED_EXPAND_LIMITS: usize = 64;

thread_local! {
    /// Powers [`ExpandPow`] declined to unfold on this thread, with the size it
    /// would have taken, newest last and de-duplicated by node.
    ///
    /// The engine's contract is `Option<(ExprId, DerivationLog)>` — a rule that
    /// changes nothing returns `None` and contributes no step, and it cannot
    /// return a step *with* `before == after` because `apply_rules` would then
    /// spin on it forever. So the note travels out of band and
    /// [`crate::simplify::engine::simplify_with`] appends it to the log of the
    /// pass that declined, which is where `.steps` can show it.
    static DECLINED_EXPANSIONS: std::cell::RefCell<Vec<(ExprId, u32, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Forget any recorded declines (start of an expanding simplify pass).
pub(crate) fn clear_expand_limits() {
    DECLINED_EXPANSIONS.with(|c| c.borrow_mut().clear());
}

/// Take the powers this pass declined to expand, as `(node, exponent, summands)`.
pub(crate) fn take_expand_limits() -> Vec<(ExprId, u32, usize)> {
    DECLINED_EXPANSIONS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// The rule name carried by the derivation step that reports a declined
/// expansion. Named for what happened, not for a rewrite that did not.
pub(crate) const EXPAND_POW_LIMIT_RULE: &str = "expand_pow_limit_reached";

/// Number of distributed products `(m summands)^n` would form, saturating.
fn expansion_products(summands: usize, exp: u32) -> u64 {
    (summands as u64).checked_pow(exp).unwrap_or(u64::MAX)
}

pub struct ExpandPow;

impl ExpandPow {
    /// Distribute `acc · (summands)` into a flat list of product-terms.
    /// `acc` holds the partial monomials accumulated so far.
    fn distribute_once(acc: &[ExprId], summands: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
        let mut out = Vec::with_capacity(acc.len() * summands.len());
        for &term in acc {
            for &s in summands {
                out.push(pool.mul(vec![term, s]));
            }
        }
        out
    }
}

impl RewriteRule for ExpandPow {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::POW
    }
    fn name(&self) -> &'static str {
        "expand_pow"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (base, exp) = match pool.get(expr) {
            ExprData::Pow { base, exp } => (base, exp),
            _ => return None,
        };
        // Base must be a sum; otherwise there is nothing to distribute.
        let summands = match pool.get(base) {
            ExprData::Add(v) => v,
            _ => return None,
        };
        let n = as_integer(exp, pool)?;
        if n <= 1 {
            return None;
        }
        let n_u32 = n.to_u32()?;
        if n_u32 > MAX_EXPAND_POW_EXP
            && expansion_products(summands.len(), n_u32) > MAX_EXPAND_POW_PRODUCTS
        {
            // Declining is a decision about *this* expression, and a caller who
            // asked for expansion and got their input back deserves to be told
            // which bound stopped it — otherwise the rule is a silent no-op and
            // `.steps` records a derivation that never mentions the step it
            // refused to take.
            DECLINED_EXPANSIONS.with(|c| {
                let mut v = c.borrow_mut();
                if v.len() < MAX_RECORDED_EXPAND_LIMITS && !v.iter().any(|&(e, _, _)| e == expr) {
                    v.push((expr, n_u32, summands.len()));
                }
            });
            return None;
        }

        // Fully distribute: start from the summands themselves (n = 1) and
        // multiply by `summands` n−1 more times, producing a flat Σ of
        // products. Never emit `(Add)·(Add)`, which `collect_mul_factors`
        // would fold straight back into `(Add)^n`.
        let mut terms: Vec<ExprId> = summands.clone();
        for _ in 1..n_u32 {
            terms = Self::distribute_once(&terms, &summands, pool);
        }
        let after = pool.add(terms);
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// ExpPow: exp(h)^n → exp(n·h)  for integer n
// ---------------------------------------------------------------------------

pub struct ExpPow;

impl RewriteRule for ExpPow {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::POW
    }
    fn name(&self) -> &'static str {
        "exp_pow"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (base, exp) = match pool.get(expr) {
            ExprData::Pow { base, exp } => (base, exp),
            _ => return None,
        };
        // base must be exp(h)
        let h = match pool.get(base) {
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => args[0],
            _ => return None,
        };
        // exp must be an integer
        let n = as_integer(exp, pool)?;
        let n_id = pool.integer(n.clone());
        let new_arg = pool.mul(vec![n_id, h]);
        let after = pool.func("exp".to_string(), vec![new_arg]);
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// CollectExp: exp(a) · exp(b) · … → exp(a+b+…)  inside a Mul
// ---------------------------------------------------------------------------

pub struct CollectExp;

impl RewriteRule for CollectExp {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::MUL
    }
    fn name(&self) -> &'static str {
        "collect_exp"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let args = match pool.get(expr) {
            ExprData::Mul(v) => v,
            _ => return None,
        };

        let mut exp_args: Vec<ExprId> = Vec::new();
        let mut other: Vec<ExprId> = Vec::new();
        for &a in &args {
            match pool.with(a, |d| match d {
                ExprData::Func { name, args: fargs } if name == "exp" && fargs.len() == 1 => {
                    Some(fargs[0])
                }
                _ => None,
            }) {
                Some(inner) => exp_args.push(inner),
                None => other.push(a),
            }
        }

        if exp_args.len() < 2 {
            return None;
        }

        let sum = pool.add(exp_args);
        let merged_exp = pool.func("exp".to_string(), vec![sum]);

        let after = if other.is_empty() {
            merged_exp
        } else {
            other.push(merged_exp);
            pool.mul(other)
        };

        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// PrimitiveFold: call Primitive::simplify for registered Func nodes
// (e.g. gamma(1) → 1, digamma(n) → H_{n-1} − γ).
// ---------------------------------------------------------------------------

pub struct PrimitiveFold;

impl RewriteRule for PrimitiveFold {
    fn node_kinds(&self) -> NodeKinds {
        NodeKinds::FUNC
    }
    fn name(&self) -> &'static str {
        "primitive_simplify"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (name, args) = match pool.get(expr) {
            ExprData::Func { name, args } => (name, args),
            _ => return None,
        };
        use std::sync::OnceLock;
        static REG: OnceLock<crate::primitive::PrimitiveRegistry> = OnceLock::new();
        let reg = REG.get_or_init(crate::primitive::PrimitiveRegistry::default_registry);
        let after = reg.get(&name)?.simplify(&args, pool)?;
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// Unit tests for rules
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    // --- ExpandPow bound ---

    /// `(x+y)⁶` is 64 products — well inside the budget, and outside the old
    /// exponent-only cap of 4, which refused it while happily expanding a
    /// twenty-term sum to the fourth power (160 000 products).
    #[test]
    fn expand_pow_unfolds_past_the_old_exponent_cap_when_the_work_is_small() {
        let pool = p();
        clear_expand_limits();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let base = pool.add(vec![x, y]);
        let expr = pool.pow(base, pool.integer(6_i32));
        let (after, log) = ExpandPow.apply(expr, &pool).expect("within the budget");
        assert_ne!(after, expr);
        assert_eq!(log.steps()[0].rule_name, "expand_pow");
        assert!(take_expand_limits().is_empty(), "nothing was declined");
    }

    /// Above the budget the rule still declines — but it says so. The silent
    /// no-op was the defect: the caller got their input back with no indication
    /// that a bound, rather than the mathematics, stopped the expansion.
    #[test]
    fn expand_pow_records_the_bound_it_declined() {
        let pool = p();
        clear_expand_limits();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let base = pool.add(vec![x, y, z]);
        let expr = pool.pow(base, pool.integer(9_i32)); // 3^9 = 19 683 products
        assert!(ExpandPow.apply(expr, &pool).is_none());
        assert_eq!(take_expand_limits(), vec![(expr, 9, 3)]);
        // Consuming: the same decline is not reported twice.
        assert!(take_expand_limits().is_empty());
    }

    // --- AddZero ---

    #[test]
    fn add_zero_removes_zero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let expr = pool.add(vec![x, zero]);
        let (result, log) = AddZero.apply(expr, &pool).unwrap();
        assert_eq!(result, x);
        assert_eq!(log.len(), 1);
        assert_eq!(log.steps()[0].rule_name, "add_zero");
    }

    #[test]
    fn add_zero_no_match() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let expr = pool.add(vec![x, one]);
        assert!(AddZero.apply(expr, &pool).is_none());
    }

    // --- MulOne ---

    #[test]
    fn mul_one_removes_one() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let expr = pool.mul(vec![x, one]);
        let (result, _) = MulOne.apply(expr, &pool).unwrap();
        assert_eq!(result, x);
    }

    // --- MulZero ---

    #[test]
    fn mul_zero_returns_zero() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let expr = pool.mul(vec![x, zero]);
        let (result, _) = MulZero.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    // --- division by a literal zero: `0 · 0^(-1)` has no value ---
    //
    // `0^(-1)` is division by zero, so every product containing it is
    // undefined. `simplify(0^-1)` already leaves the power alone and
    // `eval_expr(0^-1)` raises `E-EVAL-009`; these check that the surrounding
    // product agrees instead of collapsing to `1` (exponent collection) or to
    // `0` (absorption / constant folding).

    /// `0 · 0^(-1)` — the exponents sum to `0`, but `0^0 = 1` is not the value
    /// of `0 · (1/0)`.
    #[test]
    fn div_self_does_not_cancel_a_literal_zero_base() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let inv_zero = pool.pow(zero, pool.integer(-1_i32));
        let expr = pool.mul(vec![zero, inv_zero]);
        assert!(DivSelf.apply(expr, &pool).is_none());
        assert_eq!(super::super::engine::simplify(expr, &pool).value, expr);
    }

    /// The same product with a spectator factor takes the constant-folding
    /// route (`prod == 0`) instead of the exponent-collecting one.
    #[test]
    fn const_fold_does_not_absorb_a_literal_zero_reciprocal() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let inv_zero = pool.pow(zero, pool.integer(-1_i32));
        let expr = pool.mul(vec![pool.integer(5_i32), inv_zero, zero]);
        assert!(ConstFold.apply(expr, &pool).is_none());
        assert!(MulZero.apply(expr, &pool).is_none());
    }

    /// `(0 · 0^-1) + (0 · 0^-2)` — both integer coefficients are `0`, but a
    /// term is only droppable when its remaining factor is a number. This is
    /// the shape `diff(2/(x - x), x)` produces.
    #[test]
    fn sub_self_does_not_drop_an_undefined_term_with_zero_coefficient() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let inv_zero = pool.pow(zero, pool.integer(-1_i32));
        let inv_zero_sq = pool.pow(zero, pool.integer(-2_i32));
        let expr = pool.add(vec![
            pool.mul(vec![zero, inv_zero]),
            pool.mul(vec![zero, inv_zero_sq]),
        ]);
        assert!(SubSelf.apply(expr, &pool).is_none());
    }

    /// A rational negative exponent is division by zero just the same.
    #[test]
    fn mul_zero_does_not_absorb_a_rational_negative_zero_power() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let root = pool.pow(zero, pool.rational(-1_i32, 2_u32));
        let expr = pool.mul(vec![zero, root]);
        assert!(MulZero.apply(expr, &pool).is_none());
    }

    /// The guards are keyed on a *literal* zero base only: a symbolic base
    /// still cancels, which is the library's documented convention.
    #[test]
    fn div_self_still_cancels_a_symbolic_base() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let inv_x = pool.pow(x, pool.integer(-1_i32));
        let expr = pool.mul(vec![x, inv_x]);
        let (result, _) = DivSelf.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1_i32));
    }

    /// …and `0 · x` still absorbs: the guard must not switch absorption off.
    #[test]
    fn mul_zero_still_absorbs_a_symbolic_factor() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let expr = pool.mul(vec![zero, x]);
        let (result, _) = MulZero.apply(expr, &pool).unwrap();
        assert_eq!(result, zero);
    }

    // --- PowOne ---

    #[test]
    fn pow_one_simplifies() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let expr = pool.pow(x, one);
        let (result, _) = PowOne.apply(expr, &pool).unwrap();
        assert_eq!(result, x);
    }

    // --- SqrtInteger ---

    #[test]
    fn sqrt_integer_perfect_square() {
        let pool = p();
        let four = pool.integer(4_i32);
        let expr = pool.func("sqrt", vec![four]);
        let (result, _) = SqrtInteger.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(2_i32));
    }

    #[test]
    fn sqrt_integer_non_square_unchanged() {
        let pool = p();
        let five = pool.integer(5_i32);
        let expr = pool.func("sqrt", vec![five]);
        assert!(SqrtInteger.apply(expr, &pool).is_none());
    }

    // --- PowZero ---

    #[test]
    fn pow_zero_gives_one_with_condition() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        let expr = pool.pow(x, zero);
        let (result, log) = PowZero.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1_i32));
        let step = &log.steps()[0];
        assert_eq!(step.side_conditions.len(), 1);
        assert!(matches!(step.side_conditions[0], SideCondition::NonZero(_)));
    }

    // --- ConstFold ---

    #[test]
    fn const_fold_add_integers() {
        let pool = p();
        let two = pool.integer(2_i32);
        let three = pool.integer(3_i32);
        let expr = pool.add(vec![two, three]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(5_i32));
    }

    #[test]
    fn const_fold_mul_integers() {
        let pool = p();
        let two = pool.integer(2_i32);
        let three = pool.integer(3_i32);
        let expr = pool.mul(vec![two, three]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(6_i32));
    }

    #[test]
    fn const_fold_pow() {
        let pool = p();
        let two = pool.integer(2_i32);
        let ten = pool.integer(10_i32);
        let expr = pool.pow(two, ten);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1024_i32));
    }

    #[test]
    fn const_fold_partial_add() {
        // Add([2, 3, x]) → Add([5, x])
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let three = pool.integer(3_i32);
        let expr = pool.add(vec![two, three, x]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.add(vec![pool.integer(5_i32), x]));
    }

    // --- SubSelf ---

    #[test]
    fn sub_self_cancels_terms() {
        // x + (-1)*x = 0
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let neg_x = pool.mul(vec![neg_one, x]);
        let expr = pool.add(vec![x, neg_x]);
        let (result, _) = SubSelf.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn sub_self_collects_coefficients() {
        // 2x + 3x = 5x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let three_x = pool.mul(vec![pool.integer(3_i32), x]);
        let expr = pool.add(vec![two_x, three_x]);
        let (result, _) = SubSelf.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.mul(vec![pool.integer(5_i32), x]));
    }

    // --- DivSelf ---

    #[test]
    fn div_self_cancels_factors() {
        // x * x^(-1) = 1
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x_inv = pool.pow(x, pool.integer(-1_i32));
        let expr = pool.mul(vec![x, x_inv]);
        let (result, _) = DivSelf.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1_i32));
    }

    #[test]
    fn div_self_combines_powers() {
        // x^2 * x^(-1) = x
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let x_inv = pool.pow(x, pool.integer(-1_i32));
        let expr = pool.mul(vec![x2, x_inv]);
        let (result, _) = DivSelf.apply(expr, &pool).unwrap();
        assert_eq!(result, x);
    }

    // --- CanonicalOrder ---

    #[test]
    fn canonical_order_sorts() {
        // PA-3: children are sorted at construction so CanonicalOrder is a no-op
        // (both orderings intern to the same ExprId).  The rule should return
        // None for any already-canonicalised expression.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![x, y]);
        // Since both orderings are identical after PA-3, CanonicalOrder should
        // find nothing to rewrite.
        let result = CanonicalOrder.apply(expr, &pool);
        assert!(
            result.is_none(),
            "CanonicalOrder should be a no-op when children are already sorted at construction"
        );
    }

    // -------------------------------------------------------------------
    // ElementaryAtConst
    // -------------------------------------------------------------------

    #[test]
    fn exp_zero_is_one() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("exp", vec![zero]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1_i32));
    }

    #[test]
    fn sin_zero_is_zero() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("sin", vec![zero]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn cos_zero_is_one() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("cos", vec![zero]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1_i32));
    }

    #[test]
    fn sinh_zero_is_zero() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("sinh", vec![zero]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn cosh_zero_is_one() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("cosh", vec![zero]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1_i32));
    }

    #[test]
    fn tan_zero_is_zero() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("tan", vec![zero]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn atan_zero_is_zero() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("atan", vec![zero]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn asin_zero_is_zero() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("asin", vec![zero]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn log_one_is_zero() {
        let pool = p();
        let one = pool.integer(1_i32);
        let expr = pool.func("log", vec![one]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn ln_one_is_zero() {
        let pool = p();
        let one = pool.integer(1_i32);
        let expr = pool.func("ln", vec![one]);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn elementary_at_const_no_match_for_nonzero_arg() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![x]);
        assert!(ConstFold.apply(expr, &pool).is_none());
    }

    #[test]
    fn exp_zero_fires_via_full_simplify() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("exp", vec![zero]);
        let r = crate::simplify::simplify(expr, &pool);
        assert_eq!(r.value, pool.integer(1_i32));
    }

    // -------------------------------------------------------------------
    // PowOne (x^1 → x) — already implemented; exercised via full simplify
    // -------------------------------------------------------------------

    #[test]
    fn pow_one_via_full_simplify() {
        let pool = p();
        let s = pool.symbol("s", Domain::Real);
        let expr = pool.pow(s, pool.integer(1_i32));
        let r = crate::simplify::simplify(expr, &pool);
        assert_eq!(r.value, s);
    }

    // -------------------------------------------------------------------
    // 1^r → 1 for literal rational exponents
    // -------------------------------------------------------------------

    #[test]
    fn one_pow_half_is_one() {
        let pool = p();
        let one = pool.integer(1_i32);
        let half = pool.rational(1_i32, 2_i32);
        let expr = pool.pow(one, half);
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1_i32));
    }

    #[test]
    fn one_pow_half_via_full_simplify() {
        let pool = p();
        let one = pool.integer(1_i32);
        let half = pool.rational(1_i32, 2_i32);
        let expr = pool.pow(one, half);
        let r = crate::simplify::simplify(expr, &pool);
        assert_eq!(r.value, pool.integer(1_i32));
    }

    // -------------------------------------------------------------------
    // PowOfPow: (x^a)^b → x^(a*b) for literal integer a, b
    // -------------------------------------------------------------------

    #[test]
    fn pow_of_pow_combines_integer_exponents() {
        // (s^4)^(-1) → s^(-4)
        let pool = p();
        let s = pool.symbol("s", Domain::Real);
        let s4 = pool.pow(s, pool.integer(4_i32));
        let expr = pool.pow(s4, pool.integer(-1_i32));
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.pow(s, pool.integer(-4_i32)));
    }

    #[test]
    fn pow_of_pow_via_full_simplify() {
        let pool = p();
        let s = pool.symbol("s", Domain::Real);
        let s4 = pool.pow(s, pool.integer(4_i32));
        let expr = pool.pow(s4, pool.integer(-1_i32));
        let r = crate::simplify::simplify(expr, &pool);
        assert_eq!(r.value, pool.pow(s, pool.integer(-4_i32)));
    }

    #[test]
    fn pow_of_pow_does_not_fire_for_fractional_inner_exponent() {
        // (x^(1/2))^2 is NOT rewritten by PowOfPow (left for other rules /
        // domain-aware identities) — branch-cut conservatism.
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let half = pool.rational(1_i32, 2_i32);
        let x_half = pool.pow(x, half);
        let expr = pool.pow(x_half, pool.integer(2_i32));
        assert!(ConstFold.apply(expr, &pool).is_none());
    }

    // -------------------------------------------------------------------
    // EvenPowerSignFold: (-1 * x)^n → x^n for literal even integer n
    // -------------------------------------------------------------------

    #[test]
    fn even_power_sign_fold_squares() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.pow(neg_x, pool.integer(2_i32));
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.pow(x, pool.integer(2_i32)));
    }

    #[test]
    fn even_power_sign_fold_via_full_simplify() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.pow(neg_x, pool.integer(2_i32));
        let r = crate::simplify::simplify(expr, &pool);
        assert_eq!(r.value, pool.pow(x, pool.integer(2_i32)));
    }

    #[test]
    fn odd_power_sign_fold_does_not_fire() {
        // (-1 * x)^3 should NOT drop the sign (it's -x^3).
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let neg_x = pool.mul(vec![pool.integer(-1_i32), x]);
        let expr = pool.pow(neg_x, pool.integer(3_i32));
        assert!(ConstFold.apply(expr, &pool).is_none());
    }

    // -------------------------------------------------------------------
    // RationalCanon: Rational(n/1) → Integer(n)
    // -------------------------------------------------------------------

    #[test]
    fn rational_with_denom_one_canonicalizes_to_integer() {
        let pool = p();
        // Build a Rational(3/1) node directly (bypassing ExprPool::rational's
        // own reduction, which still leaves a Rational node for denom == 1).
        let r = rug::Rational::from((rug::Integer::from(3), rug::Integer::from(1)));
        let expr = pool.intern(ExprData::Rational(crate::kernel::expr::BigRat(r)));
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(3_i32));
    }

    #[test]
    fn rational_with_denom_gt_one_unchanged() {
        let pool = p();
        let half = pool.rational(1_i32, 2_i32);
        assert!(ConstFold.apply(half, &pool).is_none());
    }

    // -------------------------------------------------------------------
    // Numeric cancellation across a product:
    //   π · (4π)^(-1) → 1/4   (DistributePowOverLiteralCoeff + DivSelf +
    //   ConstFold's new negative-integer-exponent fold)
    // -------------------------------------------------------------------

    #[test]
    fn distribute_pow_over_literal_coeff() {
        // (4*pi)^(-1) → 4^(-1) * pi^(-1)
        let pool = p();
        let pi = pool.symbol("pi", Domain::NonZero);
        let four_pi = pool.mul(vec![pool.integer(4_i32), pi]);
        let expr = pool.pow(four_pi, pool.integer(-1_i32));
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        let expected = pool.mul(vec![
            pool.pow(pool.integer(4_i32), pool.integer(-1_i32)),
            pool.pow(pi, pool.integer(-1_i32)),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn pi_times_inverse_four_pi_is_one_quarter() {
        // pi * (4*pi)^(-1) → 1/4
        let pool = p();
        let pi = pool.symbol("pi", Domain::NonZero);
        let four_pi = pool.mul(vec![pool.integer(4_i32), pi]);
        let inv = pool.pow(four_pi, pool.integer(-1_i32));
        let expr = pool.mul(vec![pi, inv]);
        let r = crate::simplify::simplify(expr, &pool);
        assert_eq!(r.value, pool.rational(1_i32, 4_i32));
    }

    #[test]
    fn integer_to_negative_one_is_reciprocal_rational() {
        // 4^(-1) → 1/4
        let pool = p();
        let expr = pool.pow(pool.integer(4_i32), pool.integer(-1_i32));
        let (result, _) = ConstFold.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.rational(1_i32, 4_i32));
    }

    // -------------------------------------------------------------------
    // Idempotency spot-checks on larger expressions
    // -------------------------------------------------------------------

    #[test]
    fn idempotent_on_combined_expression() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let pi = pool.symbol("pi", Domain::Real);

        // Build: exp(0) + sin(0) + (1)^(1/2) + (s^4)^(-1) + (-1*x)^2
        //        + pi * (4*pi)^(-1)
        let s = pool.symbol("s", Domain::Real);
        let exp0 = pool.func("exp", vec![pool.integer(0_i32)]);
        let sin0 = pool.func("sin", vec![pool.integer(0_i32)]);
        let one_pow_half = pool.pow(pool.integer(1_i32), pool.rational(1_i32, 2_i32));
        let s_pow_pow = pool.pow(pool.pow(s, pool.integer(4_i32)), pool.integer(-1_i32));
        let neg_x_sq = pool.pow(pool.mul(vec![pool.integer(-1_i32), x]), pool.integer(2_i32));
        let four_pi = pool.mul(vec![pool.integer(4_i32), pi]);
        let pi_cancel = pool.mul(vec![pi, pool.pow(four_pi, pool.integer(-1_i32))]);

        let expr = pool.add(vec![
            exp0,
            sin0,
            one_pow_half,
            s_pow_pow,
            neg_x_sq,
            pi_cancel,
        ]);

        let r1 = crate::simplify::simplify(expr, &pool);
        let r2 = crate::simplify::simplify(r1.value, &pool);
        assert_eq!(r1.value, r2.value, "simplify should be idempotent");
    }

    #[test]
    fn idempotent_on_rational_canon_node() {
        let pool = p();
        let r = rug::Rational::from((rug::Integer::from(5), rug::Integer::from(1)));
        let rat_five = pool.intern(ExprData::Rational(crate::kernel::expr::BigRat(r)));
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.add(vec![rat_five, x]);

        let r1 = crate::simplify::simplify(expr, &pool);
        let r2 = crate::simplify::simplify(r1.value, &pool);
        assert_eq!(r1.value, r2.value);
        assert_eq!(r1.value, pool.add(vec![pool.integer(5_i32), x]));
    }

    // -------------------------------------------------------------------
    // Imaginary unit — algebraic power rules (i² = −1, i^(4k+r) → i^r)
    // -------------------------------------------------------------------

    #[test]
    fn imaginary_unit_pow_cycle() {
        let pool = p();
        let i = pool.imaginary_unit();
        let neg_i = pool.mul(vec![pool.integer(-1_i32), i]);
        // i² = −1
        let i2 = pool.pow(i, pool.integer(2_i32));
        assert_eq!(
            crate::simplify::simplify(i2, &pool).value,
            pool.integer(-1_i32)
        );
        // i³ = −i
        let i3 = pool.pow(i, pool.integer(3_i32));
        assert_eq!(crate::simplify::simplify(i3, &pool).value, neg_i);
        // i⁴ = 1
        let i4 = pool.pow(i, pool.integer(4_i32));
        assert_eq!(
            crate::simplify::simplify(i4, &pool).value,
            pool.integer(1_i32)
        );
        // i⁵ = i
        let i5 = pool.pow(i, pool.integer(5_i32));
        assert_eq!(crate::simplify::simplify(i5, &pool).value, i);
        // i^(-1) = −i
        let im1 = pool.pow(i, pool.integer(-1_i32));
        assert_eq!(crate::simplify::simplify(im1, &pool).value, neg_i);
        // i^7 = −i  (4·1 + 3)
        let i7 = pool.pow(i, pool.integer(7_i32));
        assert_eq!(crate::simplify::simplify(i7, &pool).value, neg_i);
    }

    #[test]
    fn imaginary_unit_mul_collapses() {
        let pool = p();
        let i = pool.imaginary_unit();
        // i · i → −1
        let ii = pool.mul(vec![i, i]);
        assert_eq!(
            crate::simplify::simplify(ii, &pool).value,
            pool.integer(-1_i32)
        );
        // (2i)·(3i) → −6
        let two_i = pool.mul(vec![pool.integer(2_i32), i]);
        let three_i = pool.mul(vec![pool.integer(3_i32), i]);
        let prod = pool.mul(vec![two_i, three_i]);
        assert_eq!(
            crate::simplify::simplify(prod, &pool).value,
            pool.integer(-6_i32)
        );
        // i · i · i → −i
        let neg_i = pool.mul(vec![pool.integer(-1_i32), i]);
        let iii = pool.mul(vec![i, i, i]);
        assert_eq!(crate::simplify::simplify(iii, &pool).value, neg_i);
        // i² · i² → 1  (mix of i^k factors)
        let i2 = pool.pow(i, pool.integer(2_i32));
        let quad = pool.mul(vec![i2, i2]);
        assert_eq!(
            crate::simplify::simplify(quad, &pool).value,
            pool.integer(1_i32)
        );
    }

    #[test]
    fn imaginary_unit_single_factor_untouched() {
        // A lone `i` (or `c·i`) must not be folded away — only collapses
        // happen when ≥2 imaginary factors meet.
        let pool = p();
        let i = pool.imaginary_unit();
        let two_i = pool.mul(vec![pool.integer(2_i32), i]);
        assert_eq!(crate::simplify::simplify(two_i, &pool).value, two_i);
    }

    #[test]
    fn imaginary_unit_is_constant_under_diff() {
        // d/dx i = 0 — the imaginary unit is a constant atom (like π/e), so
        // differentiating w.r.t. an unrelated variable yields 0.
        let pool = p();
        let i = pool.imaginary_unit();
        let x = pool.symbol("x", crate::kernel::Domain::Real);
        let d = crate::diff::diff(i, x, &pool).unwrap().value;
        assert_eq!(
            crate::simplify::simplify(d, &pool).value,
            pool.integer(0_i32)
        );
    }

    #[test]
    fn principal_arg_safe_cases_and_branch_cut_refusal() {
        let pool = p();
        let i = pool.imaginary_unit();
        let neg_i = pool.mul(vec![pool.integer(-1_i32), i]);
        let pos = pool.symbol("x", Domain::Positive);
        let z = pool.symbol("z", Domain::Complex);
        let pi = pool.symbol("pi", Domain::Positive);
        let half_pi = pool.mul(vec![pool.rational(1, 2), pi]);
        let neg_half_pi = pool.mul(vec![pool.integer(-1_i32), half_pi]);

        assert_eq!(
            crate::simplify::simplify(pool.func("arg", vec![pool.integer(3_i32)]), &pool).value,
            pool.integer(0_i32)
        );
        assert_eq!(
            crate::simplify::simplify(pool.func("arg", vec![pos]), &pool).value,
            pool.integer(0_i32)
        );
        assert_eq!(
            crate::simplify::simplify(pool.func("arg", vec![i]), &pool).value,
            crate::simplify::simplify(half_pi, &pool).value
        );
        assert_eq!(
            crate::simplify::simplify(pool.func("arg", vec![neg_i]), &pool).value,
            crate::simplify::simplify(neg_half_pi, &pool).value
        );
        // Zero, negatives, and generic complex stay unevaluated.
        let arg0 = pool.func("arg", vec![pool.integer(0_i32)]);
        assert_eq!(crate::simplify::simplify(arg0, &pool).value, arg0);
        let arg_neg = pool.func("arg", vec![pool.integer(-1_i32)]);
        assert_eq!(crate::simplify::simplify(arg_neg, &pool).value, arg_neg);
        let arg_z = pool.func("arg", vec![z]);
        assert_eq!(crate::simplify::simplify(arg_z, &pool).value, arg_z);
    }
}
