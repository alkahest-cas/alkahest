//! Multivariate Horner schemes.
//!
//! A sum of monomials `Σ cₖ · v₁^e₁ₖ ⋯ vₙ^eₙₖ` is rewritten by repeatedly
//! choosing the variable that occurs in the most terms (the greedy order of
//! Ceberio–Kreinovich and of Symbolica's evaluator), grouping the terms by
//! its exponent, and folding the groups into a Horner chain in that variable
//! whose coefficients are, recursively, schemes in the remaining variables:
//!
//! ```text
//! 1 + 2x + 3y + 4xy + 5x²   →   (1 + 3y) + x·((2 + 4y) + x·5)
//! ```
//!
//! Each level partitions its terms, so building a scheme costs
//! `O(n log n)` per level and the recursion is as deep as the number of
//! distinct variables in one monomial — not as deep as the degree.  A chain
//! longer than [`CHAIN_CHUNK`] links is cut into blocks joined by an explicit
//! power, which bounds the nesting depth of the result for the recursive
//! walkers downstream (topological sort, codegen).
//!
//! Two users share the builder: the public [`horner_multivariate`] rewrite,
//! and the compile pipeline's polynomial pass (`jit::optimize`), which also
//! treats non-polynomial subexpressions (`sin(x)`, `(1+x)^(1/2)`, …) as
//! extra variables so that polynomial coefficients of transcendental terms
//! are shared too.

use crate::kernel::{ExprData, ExprId, ExprPool, IdMap};
use crate::poly::{ConversionError, MultiPoly};

/// Longest Horner chain built in one block (see the module docs).
pub(crate) const CHAIN_CHUNK: usize = 128;

/// Largest exponent a monomial factor may carry and still be treated as a
/// power of a variable; a larger one is left as an opaque factor.
pub(crate) const MAX_TERM_EXPONENT: u32 = 1 << 16;

/// One monomial: constant factors times a product of atom powers.
#[derive(Clone, Debug)]
pub(crate) struct Term {
    /// Constant factors (no variable in them); empty means `1`.
    pub(crate) coef: Vec<ExprId>,
    /// `(atom index, exponent ≥ 1)`, sorted by atom index, no repeats.
    pub(crate) exps: Vec<(u32, u32)>,
}

impl Term {
    /// Multiply in `atom^e`, merging with an existing power of the same atom.
    /// `false` if the merged exponent would exceed [`MAX_TERM_EXPONENT`].
    pub(crate) fn push_power(&mut self, atom: u32, e: u32) -> bool {
        match self.exps.binary_search_by_key(&atom, |&(a, _)| a) {
            Ok(i) => {
                let merged = self.exps[i].1.saturating_add(e);
                if merged > MAX_TERM_EXPONENT {
                    return false;
                }
                self.exps[i].1 = merged;
            }
            Err(i) => self.exps.insert(i, (atom, e)),
        }
        true
    }

    fn take_exponent(&mut self, atom: u32) -> u32 {
        match self.exps.binary_search_by_key(&atom, |&(a, _)| a) {
            Ok(i) => self.exps.remove(i).1,
            Err(_) => 0,
        }
    }
}

/// Builds Horner schemes into a pool.
pub(crate) struct SchemeBuilder<'p> {
    pool: &'p ExprPool,
    /// Atom index → expression.  Ties in the greedy choice go to the lower
    /// index, so callers list preferred variables first.
    atoms: Vec<ExprId>,
    /// The `c + v·acc` links built, which a backend may evaluate with a fused
    /// multiply-add (they were reassociated already, so the rounding of the
    /// original form is not being preserved anyway).
    pub(crate) chain_links: Vec<ExprId>,
    zero: ExprId,
    one: ExprId,
}

impl<'p> SchemeBuilder<'p> {
    pub(crate) fn new(pool: &'p ExprPool, atoms: Vec<ExprId>) -> Self {
        SchemeBuilder {
            pool,
            atoms,
            chain_links: Vec::new(),
            zero: pool.integer(0_i32),
            one: pool.integer(1_i32),
        }
    }

    fn coef_expr(&self, coef: &[ExprId]) -> ExprId {
        match coef {
            [] => self.one,
            [c] => *c,
            _ => self.pool.mul(coef.to_vec()),
        }
    }

    fn power(&self, var: ExprId, e: u32) -> ExprId {
        if e == 1 {
            var
        } else {
            self.pool.pow(var, self.pool.integer(e))
        }
    }

    fn mul2(&self, a: ExprId, b: ExprId) -> ExprId {
        if b == self.one {
            a
        } else if a == self.one {
            b
        } else {
            self.pool.mul(vec![a, b])
        }
    }

    fn sum(&self, parts: Vec<(ExprId, u32)>) -> (ExprId, u32) {
        match parts.len() {
            0 => (self.zero, 0),
            1 => parts[0],
            _ => {
                let depth = parts.iter().map(|p| p.1).max().unwrap_or(0) + 1;
                (
                    self.pool.add(parts.into_iter().map(|p| p.0).collect()),
                    depth,
                )
            }
        }
    }

    /// Build the scheme for `terms`. Returns the expression and its nesting
    /// depth (number of Add/Mul levels).
    pub(crate) fn build(&mut self, terms: Vec<Term>) -> (ExprId, u32) {
        if terms.is_empty() {
            return (self.zero, 0);
        }
        // Occurrence counts of each atom.
        let mut counts: IdxCounts = IdxCounts::default();
        for t in &terms {
            for &(a, _) in &t.exps {
                counts.bump(a);
            }
        }
        let Some(v) = counts.best() else {
            // Constants only.
            let parts = terms
                .iter()
                .map(|t| (self.coef_expr(&t.coef), u32::from(t.coef.len() > 1)))
                .collect();
            return self.sum(parts);
        };

        let mut keyed: Vec<(u32, Term)> = terms
            .into_iter()
            .map(|mut t| (t.take_exponent(v), t))
            .collect();
        keyed.sort_by_key(|(e, _)| *e);

        let mut groups: Vec<(u32, ExprId, u32)> = Vec::new();
        let mut it = keyed.into_iter().peekable();
        while let Some((e, t)) = it.next() {
            let mut g = vec![t];
            while it.peek().is_some_and(|(e2, _)| *e2 == e) {
                g.push(it.next().map(|p| p.1).unwrap_or_else(|| unreachable!()));
            }
            let (c, d) = self.build(g);
            groups.push((e, c, d));
        }

        let var = self.atoms[v as usize];
        let mut parts: Vec<(ExprId, u32)> = Vec::new();
        for chunk in groups.chunks(CHAIN_CHUNK) {
            let base = chunk[0].0;
            let last = chunk[chunk.len() - 1];
            let (mut acc, mut depth) = (last.1, last.2);
            for i in (0..chunk.len() - 1).rev() {
                let gap = chunk[i + 1].0 - chunk[i].0;
                let m = self.mul2(self.power(var, gap), acc);
                acc = self.pool.add(vec![chunk[i].1, m]);
                depth = depth.max(chunk[i].2) + 2;
                self.chain_links.push(acc);
            }
            if base > 0 {
                acc = self.mul2(self.power(var, base), acc);
                depth += 1;
            }
            parts.push((acc, depth));
        }
        self.sum(parts)
    }
}

/// Occurrence counts keyed by atom index, with the deterministic greedy
/// choice: most occurrences, then lowest index.
#[derive(Default)]
struct IdxCounts(Vec<usize>);

impl IdxCounts {
    fn bump(&mut self, a: u32) {
        let a = a as usize;
        if self.0.len() <= a {
            self.0.resize(a + 1, 0);
        }
        self.0[a] += 1;
    }

    fn best(&self) -> Option<u32> {
        let mut best: Option<(usize, usize)> = None;
        for (i, &c) in self.0.iter().enumerate() {
            let better = match best {
                None => true,
                Some((_, bc)) => c > bc,
            };
            if c > 0 && better {
                best = Some((i, c));
            }
        }
        best.map(|(i, _)| i as u32)
    }
}

// ---------------------------------------------------------------------------
// Public rewrite
// ---------------------------------------------------------------------------

/// Rewrite a multivariate polynomial into a Horner scheme.
///
/// `expr` must be a polynomial in `vars`: a sum of monomials whose
/// coefficients do not mention any of `vars` (they may be numbers, constants
/// such as `π`, or other symbols acting as parameters). An expression that is
/// not written as such a sum — `(1 + x)·(1 + y)`, say — is expanded first,
/// which is supported for integer coefficients. When `vars` is empty, every
/// free symbol of `expr` is a variable (named constants such as `pi` are not).
///
/// The nesting order is chosen greedily: at each level the variable occurring
/// in the most remaining terms is factored out, ties going to the variable
/// listed first in `vars`. The result is the same polynomial; evaluating it
/// in floating point takes about one multiply-add per term, instead of the
/// powers and products of the expanded form, and rounds differently.
///
/// The univariate [`horner`](super::horner) keeps its own contract (integer
/// coefficients, one variable); this is the general entry point.
///
/// # Errors
///
/// The [`ConversionError`] of [`MultiPoly::from_symbolic`] when `expr` is not
/// a polynomial in `vars` (for instance `sin(x) + y`, or `x^(1/2)`).
///
/// # Example
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::horner::horner_multivariate;
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// let y = pool.symbol("y", Domain::Real);
/// // 1 + 2x + 3y + 4xy + 5x²
/// let e = pool.add(vec![
///     pool.integer(1),
///     pool.mul(vec![pool.integer(2), x]),
///     pool.mul(vec![pool.integer(3), y]),
///     pool.mul(vec![pool.integer(4), x, y]),
///     pool.mul(vec![pool.integer(5), pool.pow(x, pool.integer(2))]),
/// ]);
/// let h = horner_multivariate(e, &[x, y], &pool).unwrap();
/// assert_ne!(h, e);
/// ```
pub fn horner_multivariate(
    expr: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<ExprId, ConversionError> {
    let vars: Vec<ExprId> = if vars.is_empty() {
        free_variables(expr, pool)
    } else {
        let mut seen = crate::kernel::IdSet::default();
        vars.iter().copied().filter(|v| seen.insert(*v)).collect()
    };
    if vars.is_empty() {
        return Ok(expr);
    }
    let index: IdMap<u32> = vars
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    let mut const_memo: IdMap<bool> = IdMap::default();
    let terms = match structural_terms(expr, &index, pool, &mut const_memo) {
        Some(t) => t,
        None => expanded_terms(expr, &vars, pool)?,
    };
    let mut b = SchemeBuilder::new(pool, vars);
    Ok(b.build(terms).0)
}

/// The free symbols of `expr` that are variables (not named constants), in
/// order of first appearance.
fn free_variables(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut seen = crate::kernel::IdSet::default();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        pool.with(id, |d| match d {
            ExprData::Symbol { name, .. } => {
                if crate::eval::symbols::constant_value_f64(name).is_none() {
                    out.push(id);
                }
            }
            ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
                stack.extend(a.iter().rev())
            }
            ExprData::Pow { base, exp } => {
                stack.push(*exp);
                stack.push(*base);
            }
            _ => {}
        });
    }
    out
}

/// True when `id` mentions none of the variables in `index`.
fn mentions_none(id: ExprId, index: &IdMap<u32>, pool: &ExprPool, memo: &mut IdMap<bool>) -> bool {
    if let Some(&b) = memo.get(&id) {
        return b;
    }
    let r = if index.contains_key(&id) {
        false
    } else {
        let children: Vec<ExprId> = pool.with(id, |d| match d {
            ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => a.clone(),
            ExprData::Pow { base, exp } => vec![*base, *exp],
            ExprData::BigO(i) => vec![*i],
            ExprData::Piecewise { branches, default } => branches
                .iter()
                .flat_map(|(c, v)| [*c, *v])
                .chain(std::iter::once(*default))
                .collect(),
            ExprData::Predicate { args, .. } => args.clone(),
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                vec![*var, *body]
            }
            ExprData::RootSum { poly, var, body } => vec![*poly, *var, *body],
            _ => vec![],
        });
        children
            .into_iter()
            .all(|c| mentions_none(c, index, pool, memo))
    };
    memo.insert(id, r);
    r
}

/// The monomials of `expr` read off its structure, when it is literally a
/// sum of monomials in the variables of `index`; `None` otherwise.
fn structural_terms(
    expr: ExprId,
    index: &IdMap<u32>,
    pool: &ExprPool,
    memo: &mut IdMap<bool>,
) -> Option<Vec<Term>> {
    let summands = pool.with(expr, |d| match d {
        ExprData::Add(a) => a.clone(),
        _ => vec![expr],
    });
    let mut terms = Vec::with_capacity(summands.len());
    for s in summands {
        let factors = pool.with(s, |d| match d {
            ExprData::Mul(a) => a.clone(),
            _ => vec![s],
        });
        let mut t = Term {
            coef: Vec::new(),
            exps: Vec::new(),
        };
        for f in factors {
            if let Some(&i) = index.get(&f) {
                if !t.push_power(i, 1) {
                    return None;
                }
                continue;
            }
            if mentions_none(f, index, pool, memo) {
                t.coef.push(f);
                continue;
            }
            let (base, e) = pool.with(f, |d| match d {
                ExprData::Pow { base, exp } => (Some(*base), small_positive_exponent(*exp, pool)),
                _ => (None, None),
            });
            match (base.and_then(|b| index.get(&b).copied()), e) {
                (Some(i), Some(e)) => {
                    if !t.push_power(i, e) {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        terms.push(t);
    }
    Some(terms)
}

/// Monomials of `expr` through the polynomial converter (which expands).
fn expanded_terms(
    expr: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<Vec<Term>, ConversionError> {
    let p = MultiPoly::from_symbolic(expr, vars.to_vec(), pool)?;
    let mut terms = Vec::with_capacity(p.terms.len());
    for (exps, c) in p.terms.iter() {
        let mut t = Term {
            coef: if *c == 1 {
                Vec::new()
            } else {
                vec![pool.integer(c.clone())]
            },
            exps: Vec::new(),
        };
        for (i, &e) in exps.iter().enumerate() {
            if e > 0 {
                if e > MAX_TERM_EXPONENT {
                    return Err(ConversionError::ExponentTooLarge);
                }
                t.exps.push((i as u32, e));
            }
        }
        terms.push(t);
    }
    Ok(terms)
}

/// `e` when the node is an exact integer in `1..=MAX_TERM_EXPONENT`.
pub(crate) fn small_positive_exponent(exp: ExprId, pool: &ExprPool) -> Option<u32> {
    pool.with(exp, |d| match d {
        ExprData::Integer(n) => {
            n.0.to_u32()
                .filter(|&e| (1..=MAX_TERM_EXPONENT).contains(&e))
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::Domain;
    use std::collections::HashMap;

    fn eval(e: ExprId, env: &[(ExprId, f64)], pool: &ExprPool) -> f64 {
        let env: HashMap<ExprId, f64> = env.iter().copied().collect();
        eval_interp(e, &env, pool).unwrap()
    }

    fn depth(e: ExprId, pool: &ExprPool) -> usize {
        let ch: Vec<ExprId> = pool.with(e, |d| match d {
            ExprData::Add(a) | ExprData::Mul(a) => a.clone(),
            ExprData::Pow { base, .. } => vec![*base],
            _ => vec![],
        });
        1 + ch.into_iter().map(|c| depth(c, pool)).max().unwrap_or(0)
    }

    #[test]
    fn bivariate_matches_and_is_nested() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let i = |n: i32| pool.integer(n);
        let e = pool.add(vec![
            i(1),
            pool.mul(vec![i(2), x]),
            pool.mul(vec![i(3), y]),
            pool.mul(vec![i(4), x, y]),
            pool.mul(vec![i(5), pool.pow(x, i(2))]),
        ]);
        let h = horner_multivariate(e, &[x, y], &pool).unwrap();
        for &(a, b) in &[(0.5, 0.25), (-1.5, 2.0), (3.0, -0.125)] {
            let want = eval(e, &[(x, a), (y, b)], &pool);
            let got = eval(h, &[(x, a), (y, b)], &pool);
            assert!((want - got).abs() <= 1e-12 * want.abs().max(1.0));
        }
        // x² no longer appears as a power: the scheme only multiplies by x.
        let mut stack = vec![h];
        while let Some(n) = stack.pop() {
            pool.with(n, |d| match d {
                ExprData::Pow { .. } => panic!("unexpected power in {}", pool.display(h)),
                ExprData::Add(a) | ExprData::Mul(a) => stack.extend(a),
                _ => {}
            });
        }
    }

    #[test]
    fn parameters_and_rationals_are_coefficients() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let e = pool.add(vec![
            pool.mul(vec![a, pool.pow(x, pool.integer(3))]),
            pool.mul(vec![pool.rational(1, 3), x]),
            pool.func("sin", vec![a]),
        ]);
        let h = horner_multivariate(e, &[x], &pool).unwrap();
        let env = [(x, 0.7), (a, -1.3)];
        let (want, got) = (eval(e, &env, &pool), eval(h, &env, &pool));
        assert!((want - got).abs() <= 1e-14);
    }

    #[test]
    fn empty_vars_means_all_free_symbols() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let pi = pool.symbol("pi", Domain::Real);
        let e = pool.add(vec![
            pool.mul(vec![pi, x, y]),
            pool.pow(x, pool.integer(2)),
            pool.pow(y, pool.integer(2)),
        ]);
        let h = horner_multivariate(e, &[], &pool).unwrap();
        let env = [(x, 0.3), (y, 1.7)];
        assert!((eval(e, &env, &pool) - eval(h, &env, &pool)).abs() < 1e-14);
    }

    #[test]
    fn product_form_is_expanded() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let one = pool.integer(1);
        let e = pool.mul(vec![pool.add(vec![one, x]), pool.add(vec![one, y])]);
        let h = horner_multivariate(e, &[x, y], &pool).unwrap();
        let env = [(x, 0.3), (y, -2.5)];
        assert!((eval(e, &env, &pool) - eval(h, &env, &pool)).abs() < 1e-14);
    }

    #[test]
    fn non_polynomial_is_refused() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = pool.add(vec![pool.func("sin", vec![x]), y]);
        assert!(horner_multivariate(e, &[x, y], &pool).is_err());
        let e = pool.add(vec![pool.pow(x, pool.rational(1, 2)), y]);
        assert!(horner_multivariate(e, &[x, y], &pool).is_err());
    }

    #[test]
    fn long_chain_is_blocked() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let terms: Vec<ExprId> = (0..1000).map(|k| pool.pow(x, pool.integer(k))).collect();
        let e = pool.add(terms);
        let h = horner_multivariate(e, &[x], &pool).unwrap();
        assert!(
            depth(h, &pool) < 2 * CHAIN_CHUNK + 16,
            "{}",
            depth(h, &pool)
        );
        let v = eval(h, &[(x, 0.5)], &pool);
        assert!((v - 2.0).abs() < 1e-12, "{v}");
    }
}
