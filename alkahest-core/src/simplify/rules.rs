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
                let b = as_integer(base, pool)?;
                let e = as_integer(exp, pool)?;
                if e < 0 {
                    return None; // negative integer pow → rational; skip
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
}
