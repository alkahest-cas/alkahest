//! The pre-DAG-audit bodies of `collect_add_terms` and `collect_mul_factors`,
//! kept verbatim as test oracles.
//!
//! Both rules gained an early exit that declines without building the
//! per-base coefficient/exponent map, and both stopped cloning the node they
//! inspect.  The differential tests in `simplify::proptests` check that the
//! rules still answer exactly as these do — same `None`, same rewritten node,
//! same log — on random products and sums built to hit repeated bases, zero
//! and negative exponents, literal zeros and non-commuting factors.

use super::*;

/// `SubSelf` as it was.
pub(crate) struct SubSelfReference;

/// `DivSelf` as it was.
pub(crate) struct DivSelfReference;

impl RewriteRule for SubSelfReference {
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

        // Extract (coeff, base) for each arg.  Coefficients admit a
        // *rational*: restricting them to integers made `¾·u + (−¾)·u` two
        // unrelated bases, so a term-wise cancellation that is pure arithmetic
        // never happened.  All of it is exact `rug` arithmetic — nothing here
        // is a numerical approximation.
        let pairs: Vec<(Option<Coeff>, ExprId)> = args
            .iter()
            .map(|&a| extract_numeric_coeff_reference(a, pool, true))
            .collect();

        // Sum coefficients by base, preserving first-occurrence order.
        // A `None` coefficient is the implicit `1`.
        let one = Coeff::one();
        let mut coeff_map: HashMap<ExprId, Coeff> = HashMap::new();
        let mut base_order: Vec<ExprId> = vec![];
        for (coeff, base) in &pairs {
            let entry = coeff_map.entry(*base).or_insert_with(|| {
                base_order.push(*base);
                Coeff::zero()
            });
            entry.add_assign(coeff.as_ref().unwrap_or(&one));
        }

        // Check: any cancellation (coeff → 0) or merging (two args same base)?
        let any_zero = coeff_map.values().any(Coeff::is_zero);
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
                .any(|(base, c)| c.is_zero() && has_zero_to_negative_power_factor(*base, pool))
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
            let coeff = coeff_map[base].clone();
            if coeff.is_zero() {
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

impl RewriteRule for DivSelfReference {
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
            if let Some(pair) = extract_int_exp_reference(a, pool) {
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

fn extract_int_exp_reference(expr: ExprId, pool: &ExprPool) -> Option<(rug::Integer, ExprId)> {
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

fn extract_numeric_coeff_reference(
    expr: ExprId,
    pool: &ExprPool,
    rationals_too: bool,
) -> (Option<Coeff>, ExprId) {
    let literal = |a: ExprId| -> Option<Coeff> {
        pool.with(a, |d| match d {
            ExprData::Integer(n) => Some(Coeff::Int(n.0.clone())),
            ExprData::Rational(r) if rationals_too => Some(Coeff::Rat(r.0.clone())),
            _ => None,
        })
    };
    match pool.get(expr) {
        ExprData::Integer(n) => (Some(Coeff::Int(n.0.clone())), pool.integer(1_i32)),
        ExprData::Rational(r) if rationals_too => {
            (Some(Coeff::Rat(r.0.clone())), pool.integer(1_i32))
        }
        ExprData::Mul(args) => {
            let mut product: Option<Coeff> = None;
            let mut rest: Vec<ExprId> = vec![];
            for &a in &args {
                match literal(a) {
                    Some(n) => match &mut product {
                        Some(p) => p.mul_assign(n),
                        None => product = Some(n),
                    },
                    None => rest.push(a),
                }
            }
            if product.is_none() {
                // No literal factors found — leave the term intact.
                return (None, expr);
            }
            let base = match rest.len() {
                0 => pool.integer(1_i32),
                1 => rest[0],
                _ => pool.mul(rest),
            };
            (product, base)
        }
        _ => (None, expr),
    }
}
