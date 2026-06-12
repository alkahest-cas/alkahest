use crate::deriv::log::{DerivationLog, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};
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

pub trait RewriteRule: Send + Sync {
    fn name(&self) -> &'static str;
    /// Try to apply the rule to `expr`. Returns `None` if the rule does not match.
    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)>;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn as_integer(expr: ExprId, pool: &ExprPool) -> Option<rug::Integer> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.clone()),
        _ => None,
    }
}

fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    as_integer(expr, pool).is_some_and(|n| n == 0)
}

fn is_one(expr: ExprId, pool: &ExprPool) -> bool {
    as_integer(expr, pool).is_some_and(|n| n == 1)
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
                match pool.get(a) {
                    ExprData::Integer(n) => int_product *= n.0.clone(),
                    _ => non_ints.push(a),
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
        let has_zero_to_neg_pow = args.iter().any(|&a| match pool.get(a) {
            ExprData::Pow { base, exp } => {
                is_zero(base, pool) && as_integer(exp, pool).is_some_and(|e| e < 0)
            }
            _ => false,
        });
        if has_zero_to_neg_pow {
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
// ElementaryAtConst: fold single-argument elementary functions at known
// literal arguments where the value is *exact and branch-free*.
//
//   exp(0)  → 1        sin(0)  → 0        cos(0)  → 1
//   sinh(0) → 0        cosh(0) → 1        tan(0)  → 0
//   atan(0) → 0        asin(0) → 0
//   log(1)  → 0        ln(1)   → 0   (alternate head for log, see fps.rs)
//
// All of these are exact values at points strictly inside the domain of
// analyticity of each function (no branch cuts, no poles), so the fold is
// universally sound — it does not depend on the sign/positivity/realness of
// any other symbol.
// ---------------------------------------------------------------------------

pub struct ElementaryAtConst;

impl RewriteRule for ElementaryAtConst {
    fn name(&self) -> &'static str {
        "elementary_at_const"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (name, args) = match pool.get(expr) {
            ExprData::Func { name, args } if args.len() == 1 => (name, args),
            _ => return None,
        };
        let arg = args[0];
        let after = match name.as_str() {
            "exp" if is_zero(arg, pool) => pool.integer(1_i32),
            "cos" if is_zero(arg, pool) => pool.integer(1_i32),
            "cosh" if is_zero(arg, pool) => pool.integer(1_i32),
            "sin" | "sinh" | "tan" | "atan" | "asin" if is_zero(arg, pool) => pool.integer(0_i32),
            "log" | "ln" if is_one(arg, pool) => pool.integer(0_i32),
            _ => return None,
        };
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// ConstFold: numeric folding for Add/Mul (partial) and Pow (integer exponents)
// Handles Integer, Rational (with promotion), and Float atoms.
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
                let numeric_count = args
                    .iter()
                    .filter(|&&a| as_rational(a, pool).is_some())
                    .count();
                if numeric_count < 2 {
                    return None;
                }
                let mut prod = rug::Rational::from(1);
                let mut non_numeric: Vec<ExprId> = vec![];
                for &a in &args {
                    match as_rational(a, pool) {
                        Some(r) => prod *= r,
                        None => non_numeric.push(a),
                    }
                }
                let after = if prod == 0 {
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
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// PowOfPow: (x^a)^b → x^(a·b) for literal integer exponents a, b.
//
// Soundness: for *integer* a and b, integer exponentiation obeys the group
// law `(x^a)^b = x^(ab)` for all x ≠ 0, and the b == 0 / a == 0 boundary
// cases agree on both sides as well (both reduce via PowZero/PowOne). At
// x == 0 the two sides are simultaneously defined or simultaneously
// undefined (0^n is 0 for n > 0 and undefined for n ≤ 0), so no domain
// violation is introduced.
//
// Fractional `a` is deliberately **not** handled: `(x^(p/q))^b = x^(pb/q)`
// can fail for complex x because raising to a fractional power selects a
// branch, and re-exponentiating that branch value need not coincide with
// the principal branch of `x^(a·b)` (e.g. `((-1)^(1/3))^2 ≠ (-1)^(2/3)`
// under some branch conventions). Restricting to integer `a` and `b` avoids
// all such branch-cut ambiguity.
// ---------------------------------------------------------------------------

pub struct PowOfPow;

impl RewriteRule for PowOfPow {
    fn name(&self) -> &'static str {
        "pow_of_pow"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (outer_base, outer_exp) = match pool.get(expr) {
            ExprData::Pow { base, exp } => (base, exp),
            _ => return None,
        };
        let (inner_base, inner_exp) = match pool.get(outer_base) {
            ExprData::Pow { base, exp } => (base, exp),
            _ => return None,
        };
        let a = as_integer(inner_exp, pool)?;
        let b = as_integer(outer_exp, pool)?;
        let new_exp = a * b;
        let new_exp_id = pool.integer(new_exp);
        let after = pool.pow(inner_base, new_exp_id);
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// DistributePowOverLiteralCoeff: (c · rest)^n → c^n · rest^n for a literal
// integer coefficient `c` (extracted from a Mul) and a literal integer
// exponent `n`.
//
// Soundness: integer-power exponentiation distributes over multiplication
// for *integer* exponents — `(c·rest)^n = c^n · rest^n` for all complex
// `c, rest` and integer `n`, including negative `n` (where both sides are
// simultaneously undefined iff `c·rest == 0`). This is the key step that
// lets `π · (4π)^(-1)` reduce to `π · 4^(-1) · π^(-1)`, which `DivSelf` and
// `ConstFold` then collapse to `1/4`.
//
// Only fires when `c` is a literal integer `!= ±1` (those cases are
// already handled / no-ops) and `rest` is non-trivial, to avoid loops with
// `PowOne`/`ConstFold`.
// ---------------------------------------------------------------------------

pub struct DistributePowOverLiteralCoeff;

impl RewriteRule for DistributePowOverLiteralCoeff {
    fn name(&self) -> &'static str {
        "distribute_pow_literal_coeff"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (base, exp) = match pool.get(expr) {
            ExprData::Pow { base, exp } => (base, exp),
            _ => return None,
        };
        let n = as_integer(exp, pool)?;
        let (coeff, rest) = extract_int_coeff(base, pool);
        if coeff == 1 || coeff == -1 || coeff == 0 {
            return None;
        }
        if rest == pool.integer(1_i32) {
            // base was purely numeric — leave to ConstFold.
            return None;
        }
        let coeff_pow = pool.pow(pool.integer(coeff), pool.integer(n.clone()));
        let rest_pow = pool.pow(rest, pool.integer(n));
        let after = pool.mul(vec![coeff_pow, rest_pow]);
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// EvenPowerSignFold: (-1 · x)^n → x^n for literal even integer n.
//
// Soundness: for even integer n, (-y)^n = y^n for all y (real or complex),
// since (-1)^n = 1. This holds unconditionally — no domain restriction on
// x is needed.
// ---------------------------------------------------------------------------

pub struct EvenPowerSignFold;

impl RewriteRule for EvenPowerSignFold {
    fn name(&self) -> &'static str {
        "even_power_sign_fold"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let (base, exp) = match pool.get(expr) {
            ExprData::Pow { base, exp } => (base, exp),
            _ => return None,
        };
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
        let after = pool.pow(new_base, exp);
        if after == expr {
            return None;
        }
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// RationalCanon: Rational(n/1) → Integer(n)
//
// `ExprPool::rational` reduces to lowest terms but does not collapse a
// denominator of 1 to an `Integer` node — such nodes can also arise from
// un-collapsed arithmetic (see PR #147). Canonicalizing here ensures
// `Rational` nodes always have denominator > 1, simplifying downstream
// pattern matches (e.g. `as_integer`, polynomial coefficient extraction).
// Always sound: the value is unchanged, only the representation changes.
// ---------------------------------------------------------------------------

pub struct RationalCanon;

impl RewriteRule for RationalCanon {
    fn name(&self) -> &'static str {
        "rational_canon"
    }

    fn apply(&self, expr: ExprId, pool: &ExprPool) -> Option<(ExprId, DerivationLog)> {
        let r = match pool.get(expr) {
            ExprData::Rational(r) => r.0,
            _ => return None,
        };
        if *r.denom() != 1 {
            return None;
        }
        let after = pool.integer(r.numer().clone());
        Some((after, one_step(self.name(), expr, after)))
    }
}

// ---------------------------------------------------------------------------
// SubSelf: collect like terms in Add; handles x - x → 0
// ---------------------------------------------------------------------------

pub struct SubSelf;

impl RewriteRule for SubSelf {
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
            match pool.get(a) {
                ExprData::Mul(inner) => {
                    flat.extend_from_slice(&inner);
                    changed = true;
                }
                _ => flat.push(a),
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
            match pool.get(a) {
                ExprData::Add(inner) => {
                    flat.extend_from_slice(&inner);
                    changed = true;
                }
                _ => flat.push(a),
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
            .position(|&a| matches!(pool.get(a), ExprData::Add(_)))?;

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
// ExpPow: exp(h)^n → exp(n·h)  for integer n
// ---------------------------------------------------------------------------

pub struct ExpPow;

impl RewriteRule for ExpPow {
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
            match pool.get(a) {
                ExprData::Func { name, args: fargs } if name == "exp" && fargs.len() == 1 => {
                    exp_args.push(fargs[0]);
                }
                _ => other.push(a),
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
// Unit tests for rules
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
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
        let (result, _) = ElementaryAtConst.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1_i32));
    }

    #[test]
    fn sin_zero_is_zero() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("sin", vec![zero]);
        let (result, _) = ElementaryAtConst.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn cos_zero_is_one() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("cos", vec![zero]);
        let (result, _) = ElementaryAtConst.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1_i32));
    }

    #[test]
    fn sinh_zero_is_zero() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("sinh", vec![zero]);
        let (result, _) = ElementaryAtConst.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn cosh_zero_is_one() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("cosh", vec![zero]);
        let (result, _) = ElementaryAtConst.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(1_i32));
    }

    #[test]
    fn tan_zero_is_zero() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("tan", vec![zero]);
        let (result, _) = ElementaryAtConst.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn atan_zero_is_zero() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("atan", vec![zero]);
        let (result, _) = ElementaryAtConst.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn asin_zero_is_zero() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let expr = pool.func("asin", vec![zero]);
        let (result, _) = ElementaryAtConst.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn log_one_is_zero() {
        let pool = p();
        let one = pool.integer(1_i32);
        let expr = pool.func("log", vec![one]);
        let (result, _) = ElementaryAtConst.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn ln_one_is_zero() {
        let pool = p();
        let one = pool.integer(1_i32);
        let expr = pool.func("ln", vec![one]);
        let (result, _) = ElementaryAtConst.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(0_i32));
    }

    #[test]
    fn elementary_at_const_no_match_for_nonzero_arg() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("exp", vec![x]);
        assert!(ElementaryAtConst.apply(expr, &pool).is_none());
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
        let (result, _) = PowOfPow.apply(expr, &pool).unwrap();
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
        assert!(PowOfPow.apply(expr, &pool).is_none());
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
        let (result, _) = EvenPowerSignFold.apply(expr, &pool).unwrap();
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
        assert!(EvenPowerSignFold.apply(expr, &pool).is_none());
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
        let (result, _) = RationalCanon.apply(expr, &pool).unwrap();
        assert_eq!(result, pool.integer(3_i32));
    }

    #[test]
    fn rational_with_denom_gt_one_unchanged() {
        let pool = p();
        let half = pool.rational(1_i32, 2_i32);
        assert!(RationalCanon.apply(half, &pool).is_none());
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
        let pi = pool.symbol("pi", Domain::Real);
        let four_pi = pool.mul(vec![pool.integer(4_i32), pi]);
        let expr = pool.pow(four_pi, pool.integer(-1_i32));
        let (result, _) = DistributePowOverLiteralCoeff.apply(expr, &pool).unwrap();
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
        let pi = pool.symbol("pi", Domain::Real);
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
}
