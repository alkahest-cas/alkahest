//! Polynomial fast path for [`simplify_expanded`](super::engine::simplify_expanded).
//!
//! Expanding `∏ (x + i·y + z + i)` by term rewriting distributes one `Add`
//! factor at a time (`ExpandMul`), interns every intermediate product, and
//! collects like terms over many fixed-point passes.  The same answer is one
//! sparse polynomial product, which FLINT (`fmpz_mpoly`) computes in a small
//! fraction of the time — the term-rewriting route was 40–300× slower than
//! Mathematica on such inputs.
//!
//! This module computes that product *before* the rule engine runs, for the
//! subexpressions where doing so provably lands on the expression the engine
//! itself would have produced, and hands the result to the engine, which then
//! has nothing (or very little) left to do.
//!
//! # Which subexpressions
//!
//! A subexpression is taken only when the rule engine is guaranteed to expand
//! it *completely* — every product distributed, every power of a sum unfolded,
//! every like term collected — because a complete expansion has exactly one
//! spelling: the `Add` of the monomials `c·x₁^e₁·…`, each built the way
//! `collect_add_terms` / `collect_mul_factors` rebuild a term ([`build`]).
//! Wherever the engine could *stop short* — a power past `ExpandPow`'s bound,
//! `(x·y)^n` (which no rule distributes), `(x/2)^n`, `(−x)^odd`, a constant
//! power past `ConstFold`'s bit budget, a power of zero, a negative exponent —
//! the subexpression is left to the engine untouched. The rules for that are in
//! [`Expander::nf`]; each one is the precondition of the rewrite that would
//! otherwise have to fire, and the differential tests at the bottom of this
//! file pin the result against the engine alone.
//!
//! The fast path also never takes a subexpression whose engine result depends
//! on the *order* in which the engine visits things.  `collect_mul_factors`
//! merges identical factors of a product into a power, and whether two factors
//! are identical at the moment it looks depends on how far each has been
//! simplified.  Two consequences:
//!
//! * a product holding five or more factors that could become the same sum is
//!   declined (`(A)·(A)·(A)·(A)·(A)` becomes `A⁵`, which `ExpandPow` may
//!   refuse);
//! * a power above `ExpandPow`'s always-expand exponent is taken only when its
//!   base is *already* in expanded form, so the engine sees the final base,
//!   and decides on its final length, in the first pass.
//!
//! And it is only applied at the root, under `Add`s and inside function
//! arguments: those contexts never merge a polynomial with something else
//! before the polynomial is fully simplified.  Beneath a non-polynomial `Mul`
//! or `Pow` the moment a factor is merged matters (`(x²+2x+1)·((x+1)²)⁻¹`
//! cancels only if both sides are spelled alike when `collect_mul_factors`
//! looks), so those are left to the engine.
//!
//! # Coefficients and size
//!
//! Coefficients are rational: a polynomial is an integer [`TermMap`] over a
//! positive common denominator, so products and powers go through the
//! FLINT-backed [`termmap_mul`] / [`termmap_pow`].  Every power is sized by
//! `termmap_pow`'s pre-flight (`poly::size`) and every product by the same
//! bound here, before anything is allocated; an exponent past `u32` is refused
//! by the checked exponent arithmetic.  A refusal or a tripped budget simply
//! means the fast path declines and the engine runs as it always did.

use super::idmap::IdMap;
use super::rules::{
    const_pow_within_budget, expansion_products, MAX_EXPAND_POW_EXP, MAX_EXPAND_POW_PRODUCTS,
};
use crate::deriv::log::{DerivationLog, RewriteStep};
use crate::kernel::expr::BigRat;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::multipoly::{termmap_add, termmap_mul, termmap_pow, TermMap};
use rug::{Integer, Rational};
use std::collections::HashMap;
use std::rc::Rc;

/// Rule name of the single derivation step recorded for each subexpression
/// the fast path expands.
pub(crate) const EXPAND_POLYNOMIAL_RULE: &str = "expand_polynomial";

/// Expressions deeper than this are left to the engine, whose traversal runs
/// on growable stack segments; this module recurses on the native stack.
const MAX_DEPTH: u32 = 256;

/// Subexpressions whose estimated distribution work is below this are left to
/// the engine.  `1` means "anything there is to expand": measured on the
/// release wheel, the conversion beats the rules even on `(x+1)·(x+2)`
/// (5.9 µs against 9.5 µs) and ties on `x·(y+1)` (4.4 µs both), and an
/// expression with nothing to distribute costs one structural walk.
const MIN_WORK: u64 = 1;

/// A polynomial with rational coefficients: `num / den`, `den > 0`, and
/// `gcd(content(num), den) = 1` so that equal polynomials compare equal.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct QPoly {
    num: TermMap,
    den: Integer,
}

impl QPoly {
    fn constant(r: &Rational) -> Self {
        let mut num = TermMap::new();
        if *r.numer() != 0 {
            num.insert(vec![], r.numer().clone());
        }
        QPoly {
            num,
            den: r.denom().clone(),
        }
    }

    fn var(idx: usize) -> Self {
        let mut e = vec![0u32; idx + 1];
        e[idx] = 1;
        let mut num = TermMap::new();
        num.insert(e, Integer::from(1));
        QPoly {
            num,
            den: Integer::from(1),
        }
    }

    fn len(&self) -> usize {
        self.num.len()
    }

    /// The value, when this is a constant (`0` for the zero polynomial).
    fn as_constant(&self) -> Option<Rational> {
        match self.num.len() {
            0 => Some(Rational::new()),
            1 => {
                let (e, c) = self.num.iter().next()?;
                e.is_empty()
                    .then(|| Rational::from((c.clone(), self.den.clone())))
            }
            _ => None,
        }
    }

    /// Divide out `gcd(content, den)`.
    fn normalized(mut self) -> Self {
        if self.den == 1 {
            return self;
        }
        if self.num.is_empty() {
            self.den = Integer::from(1);
            return self;
        }
        let mut g = self.den.clone();
        for c in self.num.values() {
            if g == 1 {
                break;
            }
            g.gcd_mut(c);
        }
        if g != 1 {
            for c in self.num.values_mut() {
                c.div_exact_mut(&g);
            }
            self.den.div_exact_mut(&g);
        }
        self
    }

    fn scaled(num: &TermMap, k: &Integer) -> TermMap {
        if *k == 1 {
            return num.clone();
        }
        num.iter()
            .map(|(e, c)| (e.clone(), Integer::from(c * k)))
            .collect()
    }

    fn add(&self, o: &QPoly) -> QPoly {
        if self.den == o.den {
            let num = termmap_add(self.num.clone(), o.num.clone());
            return QPoly {
                num,
                den: self.den.clone(),
            }
            .normalized();
        }
        let l = Integer::from(self.den.lcm_ref(&o.den));
        let a = Self::scaled(&self.num, &Integer::from(&l / &self.den));
        let b = Self::scaled(&o.num, &Integer::from(&l / &o.den));
        QPoly {
            num: termmap_add(a, b),
            den: l,
        }
        .normalized()
    }

    fn mul(&self, o: &QPoly) -> Option<QPoly> {
        check_product_size(&self.num, &o.num)?;
        let num = termmap_mul(&self.num, &o.num).ok()?;
        let den = Integer::from(&self.den * &o.den);
        Some(QPoly { num, den }.normalized())
    }

    fn pow(&self, n: u32) -> Option<QPoly> {
        // `termmap_pow` pre-flights the numerator; the denominator's power
        // is one integer of `n·bits(den)` bits.
        if self.den != 1 {
            let bits = u64::from(self.den.significant_bits()) * u64::from(n);
            if bits > crate::budget::MAX_INTEGER_BITS {
                return None;
            }
        }
        let num = termmap_pow(&self.num, n).ok()?;
        // Coprime stays coprime under powers: no normalisation needed.
        let den = Integer::from(rug::ops::Pow::pow(&self.den, n));
        Some(QPoly { num, den })
    }
}

/// Refuse `a·b` when its estimated size — at most `|a|·|b|` terms of at most
/// `log₂‖a‖₁ + log₂‖b‖₁` bits — would not fit the machine or the active memory
/// budget (the product counterpart of `termmap_pow`'s pre-flight).
fn check_product_size(a: &TermMap, b: &TermMap) -> Option<()> {
    let pairs = (a.len() as f64) * (b.len() as f64);
    // Small products cannot reach the probe threshold; skip the norms.
    if pairs < 4096.0 {
        return Some(());
    }
    let bits =
        crate::poly::size::log2_l1_norm(a.values()) + crate::poly::size::log2_l1_norm(b.values());
    let nvars = a.keys().chain(b.keys()).map(Vec::len).max().unwrap_or(0) as u64;
    // Per term: the exponent key's heap slot, the `Integer` header and the
    // B-tree's share; doubled because the FLINT operands and result coexist
    // with the maps they are converted from and to.
    crate::poly::size::check_power_size(2.0 * pairs, bits, 4 * nvars + 128).ok()
}

/// Per-call state: the variable numbering and the memoised normal forms.
struct Expander<'p> {
    pool: &'p ExprPool,
    vars: IdMap<usize>,
    /// The variables, indexed by their slot.
    var_ids: Vec<ExprId>,
    /// `None` for a node outside the fragment (or one the fast path declined).
    memo: IdMap<Option<Rc<QPoly>>>,
    /// Estimated distributed products the engine would form for each
    /// polynomial node — the work the fast path saves.
    work: IdMap<u64>,
    /// Set when the budget trips: every later query answers `None`.
    aborted: bool,
    /// Subexpressions whose estimated work is below this are left to the
    /// engine.
    min_work: u64,
}

impl<'p> Expander<'p> {
    fn new(pool: &'p ExprPool) -> Self {
        Expander {
            pool,
            vars: IdMap::default(),
            var_ids: Vec::new(),
            memo: IdMap::default(),
            work: IdMap::default(),
            aborted: false,
            min_work: MIN_WORK,
        }
    }

    fn budget_ok(&mut self) -> bool {
        if !self.aborted && crate::budget::check().is_err() {
            self.aborted = true;
        }
        !self.aborted
    }

    /// The fully expanded polynomial of `e`, when the rule engine is certain
    /// to expand `e` completely; `None` otherwise.
    fn nf(&mut self, e: ExprId) -> Option<Rc<QPoly>> {
        if self.aborted {
            return None;
        }
        if let Some(r) = self.memo.get(&e) {
            return r.clone();
        }
        let r = self.compute(e).map(Rc::new);
        if self.aborted {
            return None;
        }
        self.memo.insert(e, r.clone());
        r
    }

    fn work_of(&self, e: ExprId) -> u64 {
        self.work.get(&e).copied().unwrap_or(0)
    }

    fn compute(&mut self, e: ExprId) -> Option<QPoly> {
        enum Node {
            Var,
            Const(Rational),
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow(ExprId, Integer),
            Other,
        }
        let pool = self.pool;
        let node = pool.with(e, |d| match d {
            ExprData::Symbol { .. } => Node::Var,
            ExprData::Integer(n) => Node::Const(Rational::from(&n.0)),
            ExprData::Rational(r) => Node::Const(r.0.clone()),
            ExprData::Add(a) => Node::Add(a.clone()),
            ExprData::Mul(a) => Node::Mul(a.clone()),
            ExprData::Pow { base, exp } => match pool.with(*exp, |x| match x {
                ExprData::Integer(n) => Some(n.0.clone()),
                _ => None,
            }) {
                Some(n) => Node::Pow(*base, n),
                None => Node::Other,
            },
            _ => Node::Other,
        });
        match node {
            Node::Var => {
                // `i·i → −1`, a non-commuting product and `∞ − ∞` all have
                // rules of their own; only an ordinary scalar unknown is a
                // polynomial variable.
                if !pool.is_mult_commutative(e)
                    || pool.is_imaginary_unit(e)
                    || pool.has_non_finite(e)
                {
                    return None;
                }
                let idx = match self.vars.get(&e) {
                    Some(&i) => i,
                    None => {
                        self.vars.insert(e, self.var_ids.len());
                        self.var_ids.push(e);
                        self.var_ids.len() - 1
                    }
                };
                Some(QPoly::var(idx))
            }
            Node::Const(r) => Some(QPoly::constant(&r)),
            Node::Add(args) => {
                let mut acc: Option<QPoly> = None;
                let mut work = 0u64;
                for a in args {
                    let p = self.nf(a)?;
                    work = work.saturating_add(self.work_of(a));
                    acc = Some(match acc {
                        None => (*p).clone(),
                        Some(s) => s.add(&p),
                    });
                }
                self.work.insert(e, work);
                acc
            }
            Node::Mul(args) => self.compute_mul(e, &args),
            Node::Pow(base, n) => self.compute_pow(e, base, &n),
            Node::Other => None,
        }
    }

    fn compute_mul(&mut self, e: ExprId, args: &[ExprId]) -> Option<QPoly> {
        let mut factors: Vec<Rc<QPoly>> = Vec::with_capacity(args.len());
        let mut work = 0u64;
        for &a in args {
            factors.push(self.nf(a)?);
            work = work.saturating_add(self.work_of(a));
        }
        if !mergeable_factors_ok(&factors) {
            return None;
        }
        // The engine distributes one sum at a time: about ∏ len products.
        let products = factors
            .iter()
            .fold(1u64, |acc, f| acc.saturating_mul(f.len().max(1) as u64));
        self.work.insert(e, work.saturating_add(products));
        // Smallest first, so the running product grows as slowly as possible.
        factors.sort_by_key(|f| f.len());
        let mut acc = (*factors[0]).clone();
        for f in &factors[1..] {
            if !self.budget_ok() {
                return None;
            }
            acc = acc.mul(f)?;
        }
        Some(acc)
    }

    fn compute_pow(&mut self, e: ExprId, base: ExprId, n: &Integer) -> Option<QPoly> {
        // `x^1 → x` (`PowOne`).  `x^0` and negative powers are not polynomial
        // expansions; leave them to the engine (`PowZero` has a side
        // condition, and `0^0` stands).
        if *n == 1 {
            let p = self.nf(base)?;
            self.work.insert(e, self.work_of(base));
            return Some((*p).clone());
        }
        let n = n.to_u32().filter(|&n| n >= 2)?;
        let p = self.nf(base)?;
        if n > MAX_EXPAND_POW_EXP {
            // Past the always-expand exponent `ExpandPow` decides on the
            // length of the base it sees.  Taken only when that is final: the
            // base is already the expansion, and short enough.
            if p.len() >= 2 && expansion_products(p.len(), n) > MAX_EXPAND_POW_PRODUCTS {
                return None;
            }
            if build(&p, &self.var_ids, self.pool) != base {
                return None;
            }
        }
        if p.len() < 2 {
            // A constant or a single term: no `ExpandPow`, but `ConstFold`.
            if let Some(c) = p.as_constant() {
                // `b^n` folds only inside `MAX_CONST_POW_BITS`.
                if !const_pow_within_budget(&c, n) {
                    return None;
                }
            } else if !monomial_power_folds(&p, n) {
                return None;
            }
        }
        if !self.budget_ok() {
            return None;
        }
        let products = expansion_products(p.len().max(1), n);
        self.work
            .insert(e, self.work_of(base).saturating_add(products));
        p.pow(n)
    }
}

/// Whether `(c·m)^n` for the single term `p = c·m` reaches `cⁿ·mⁿ` in the
/// engine: `m` must be one variable power (`(x·y)^n` is not distributed), and
/// the coefficient must be `1`, an integer `|c| ≥ 2` whose power `ConstFold`
/// will fold, or `−1` under an even exponent (`(−x)^odd` and `(x/2)^n`
/// stand).
fn monomial_power_folds(p: &QPoly, n: u32) -> bool {
    let Some((exps, c)) = p.num.iter().next() else {
        return true;
    };
    if exps.iter().filter(|&&x| x != 0).count() != 1 || p.den != 1 {
        return false;
    }
    if *c == 1 {
        return true;
    }
    if *c == -1 {
        return n % 2 == 0;
    }
    const_pow_within_budget(&Rational::from(c), n)
}

/// `collect_mul_factors` merges factors that are identical when it looks
/// into a power, and `ExpandPow` may refuse that power.  Up to four copies
/// always expand; past that, decline unless the merged power is one the
/// engine certainly resolves — a bare variable power, or a constant inside
/// `ConstFold`'s budget.
fn mergeable_factors_ok(factors: &[Rc<QPoly>]) -> bool {
    if factors.len() <= MAX_EXPAND_POW_EXP as usize {
        return true;
    }
    let mut counts: HashMap<&QPoly, u32> = HashMap::new();
    for f in factors {
        *counts.entry(&**f).or_insert(0) += 1;
    }
    counts.into_iter().all(|(p, k)| {
        if k <= MAX_EXPAND_POW_EXP {
            return true;
        }
        if let Some(c) = p.as_constant() {
            return const_pow_within_budget(&c, k);
        }
        p.len() == 1 && p.den == 1 && {
            let (exps, c) = p.num.iter().next().expect("one term");
            *c == 1 && exps.iter().filter(|&&x| x != 0).count() == 1
        }
    })
}

fn intern_coeff(r: Rational, pool: &ExprPool) -> ExprId {
    if *r.denom() == 1 {
        pool.integer(r.into_numer_denom().0)
    } else {
        pool.intern(ExprData::Rational(BigRat(r)))
    }
}

/// The expression the rule engine settles on for the fully expanded `p`: an
/// `Add` of terms `c·x₁^e₁·…`, a coefficient of `1` omitted, `x^1` spelled `x`.
fn build(p: &QPoly, vars: &[ExprId], pool: &ExprPool) -> ExprId {
    let mut terms: Vec<ExprId> = Vec::with_capacity(p.num.len());
    for (exps, c) in &p.num {
        let coeff = Rational::from((c.clone(), p.den.clone()));
        let mut factors: Vec<ExprId> = Vec::with_capacity(exps.len() + 1);
        if coeff != 1 {
            factors.push(intern_coeff(coeff, pool));
        }
        for (i, &x) in exps.iter().enumerate() {
            match x {
                0 => {}
                1 => factors.push(vars[i]),
                _ => factors.push(pool.pow(vars[i], pool.integer(x))),
            }
        }
        terms.push(match factors.len() {
            0 => pool.integer(1_i32),
            1 => factors[0],
            _ => pool.mul(factors),
        });
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

/// Rewrite every polynomial subexpression of `expr` that the fast path may
/// take (see the module docs) into its expansion, with one
/// [`EXPAND_POLYNOMIAL_RULE`] step per rewrite.
pub(crate) fn expand_polynomials(expr: ExprId, pool: &ExprPool) -> (ExprId, DerivationLog) {
    expand_polynomials_with(expr, pool, MIN_WORK)
}

/// [`expand_polynomials`] with the work threshold explicit (`0` takes every
/// eligible subexpression; the tests use it to exercise the fast path on
/// small inputs).
fn expand_polynomials_with(
    expr: ExprId,
    pool: &ExprPool,
    min_work: u64,
) -> (ExprId, DerivationLog) {
    let mut log = DerivationLog::new();
    if pool.depth(expr) > MAX_DEPTH {
        return (expr, log);
    }
    let mut shape: IdMap<u64> = IdMap::default();
    if shape_work(expr, pool, &mut shape) < min_work {
        // Nothing here is worth converting: the common case for the small
        // expressions most callers pass, which then pay one walk and no
        // polynomial arithmetic.
        return (expr, log);
    }
    let mut ex = Expander::new(pool);
    ex.min_work = min_work;
    let mut done: IdMap<ExprId> = IdMap::default();
    let out = walk(expr, &mut ex, &mut done, &mut log, &mut shape);
    (out, log)
}

/// A structural upper estimate of the products the engine would distribute
/// in `e` — the sum over its products of `∏ (summands of each factor)` and
/// over its powers of sums of `summandsⁿ` — computed from the shape alone,
/// with no polynomial arithmetic.  Gates [`walk`] before anything is
/// converted.
fn shape_work(e: ExprId, pool: &ExprPool, memo: &mut IdMap<u64>) -> u64 {
    if let Some(&w) = memo.get(&e) {
        return w;
    }
    enum Shape {
        Sum(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, Option<u32>),
        Leaf,
    }
    let shape = pool.with(e, |d| match d {
        ExprData::Add(a) => Shape::Sum(a.clone()),
        ExprData::Func { args, .. } => Shape::Sum(args.clone()),
        ExprData::Mul(a) => Shape::Mul(a.clone()),
        ExprData::Pow { base, exp } => Shape::Pow(
            *base,
            pool.with(*exp, |x| match x {
                ExprData::Integer(n) => n.0.to_u32(),
                _ => None,
            }),
        ),
        _ => Shape::Leaf,
    });
    // Summands a factor contributes when distributed.
    let width = |f: ExprId| -> u64 {
        pool.with(f, |d| match d {
            ExprData::Add(a) => a.len() as u64,
            ExprData::Pow { base, exp } => pool.with(*base, |b| match b {
                ExprData::Add(a) => pool.with(*exp, |x| match x {
                    ExprData::Integer(n) => {
                        n.0.to_u32()
                            .map_or(1, |n| expansion_products(a.len(), n.max(1)))
                    }
                    _ => 1,
                }),
                _ => 1,
            }),
            _ => 1,
        })
    };
    let w = match shape {
        Shape::Sum(args) => args.iter().fold(0u64, |acc, &a| {
            acc.saturating_add(shape_work(a, pool, memo))
        }),
        Shape::Mul(args) => {
            let own = args
                .iter()
                .fold(1u64, |acc, &a| acc.saturating_mul(width(a)));
            let own = if own > 1 { own } else { 0 };
            args.iter()
                .fold(own, |acc, &a| acc.saturating_add(shape_work(a, pool, memo)))
        }
        Shape::Pow(base, n) => {
            let own = match n {
                Some(n) if n >= 2 => width(e),
                _ => 0,
            };
            let own = if own > 1 { own } else { 0 };
            own.saturating_add(shape_work(base, pool, memo))
        }
        Shape::Leaf => 0,
    };
    memo.insert(e, w);
    w
}

fn walk(
    e: ExprId,
    ex: &mut Expander,
    done: &mut IdMap<ExprId>,
    log: &mut DerivationLog,
    shape: &mut IdMap<u64>,
) -> ExprId {
    if let Some(&r) = done.get(&e) {
        return r;
    }
    let pool = ex.pool;
    if shape_work(e, pool, shape) < ex.min_work {
        done.insert(e, e);
        return e;
    }
    let r = if let Some(p) = ex.nf(e) {
        if ex.work_of(e) >= ex.min_work {
            let after = build(&p, &ex.var_ids, pool);
            if after != e {
                log.push(RewriteStep::simple(EXPAND_POLYNOMIAL_RULE, e, after));
            }
            after
        } else {
            e
        }
    } else {
        enum Shape {
            Add(Vec<ExprId>),
            Func(String, Vec<ExprId>),
            Leaf,
        }
        let node = pool.with(e, |d| match d {
            ExprData::Add(a) => Shape::Add(a.clone()),
            ExprData::Func { name, args } => Shape::Func(name.clone(), args.clone()),
            _ => Shape::Leaf,
        });
        match node {
            Shape::Add(args) => {
                let new: Vec<ExprId> = args
                    .iter()
                    .map(|&a| walk(a, ex, done, log, shape))
                    .collect();
                if new == args {
                    e
                } else {
                    pool.add(new)
                }
            }
            Shape::Func(name, args) => {
                let new: Vec<ExprId> = args
                    .iter()
                    .map(|&a| walk(a, ex, done, log, shape))
                    .collect();
                if new == args {
                    e
                } else {
                    pool.func(name, new)
                }
            }
            Shape::Leaf => e,
        }
    };
    done.insert(e, r);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use crate::simplify::engine::{simplify_expanded, simplify_expanded_by_rules};
    use proptest::prelude::*;

    fn xyz(pool: &ExprPool) -> (ExprId, ExprId, ExprId) {
        (
            pool.symbol("x", Domain::Real),
            pool.symbol("y", Domain::Real),
            pool.symbol("z", Domain::Real),
        )
    }

    /// `simplify_expanded` with every eligible subexpression taken by the
    /// fast path.
    fn fast(e: ExprId, pool: &ExprPool) -> ExprId {
        let (pre, _) = expand_polynomials_with(e, pool, 0);
        simplify_expanded_by_rules(pre, pool).value
    }

    /// The fast path against the rule engine alone, in the same pool (so the
    /// comparison is `ExprId` identity), in both orders: the engine's result
    /// may depend on what the pool has already interned.
    fn check_same(build: &dyn Fn(&ExprPool) -> ExprId) -> Result<(), String> {
        for fast_first in [false, true] {
            let pool = ExprPool::new();
            let e = build(&pool);
            let (f, s) = if fast_first {
                let f = fast(e, &pool);
                (f, simplify_expanded_by_rules(e, &pool).value)
            } else {
                let s = simplify_expanded_by_rules(e, &pool).value;
                (fast(e, &pool), s)
            };
            if f != s {
                return Err(format!(
                    "fast path {} != rule engine {} for {}",
                    pool.display(f),
                    pool.display(s),
                    pool.display(e)
                ));
            }
        }
        Ok(())
    }

    fn assert_same(build: impl Fn(&ExprPool) -> ExprId) {
        if let Err(m) = check_same(&build) {
            panic!("{m}");
        }
    }

    fn linprod(pool: &ExprPool, n: i64) -> ExprId {
        let (x, y, z) = xyz(pool);
        let fs: Vec<ExprId> = (0..n)
            .map(|i| {
                pool.add(vec![
                    x,
                    pool.mul(vec![pool.integer(i), y]),
                    z,
                    pool.integer(i),
                ])
            })
            .collect();
        pool.mul(fs)
    }

    #[test]
    fn linear_product_matches_rule_engine() {
        assert_same(|p| linprod(p, 8));
    }

    #[test]
    fn power_of_sum_matches_rule_engine() {
        assert_same(|p| {
            let (x, y, z) = xyz(p);
            p.pow(p.add(vec![x, y, z, p.integer(1)]), p.integer(5))
        });
    }

    #[test]
    fn rational_coefficients_match_rule_engine() {
        assert_same(|p| {
            let (x, y, _) = xyz(p);
            let a = p.add(vec![p.mul(vec![p.rational(1, 2), x]), p.rational(-2, 3)]);
            let b = p.add(vec![p.mul(vec![p.rational(3, 4), y]), x, p.integer(2)]);
            p.mul(vec![a, b, p.pow(b, p.integer(2))])
        });
    }

    #[test]
    fn polynomial_under_function_and_sum_matches_rule_engine() {
        assert_same(|p| {
            let (x, y, _) = xyz(p);
            let poly = p.pow(p.add(vec![x, y, p.integer(1)]), p.integer(4));
            let s = p.func("sin", vec![poly]);
            p.add(vec![s, poly, p.pow(x, p.integer(-1))])
        });
    }

    /// The fast path does the expansion: one `expand_polynomial` step, and
    /// the engine has nothing left to expand.
    #[test]
    fn records_one_expand_polynomial_step() {
        let pool = ExprPool::new();
        let e = linprod(&pool, 8);
        let r = simplify_expanded(e, &pool);
        assert_eq!(r.value, simplify_expanded_by_rules(e, &pool).value);
        let names: Vec<&str> = r.log.steps().iter().map(|s| s.rule_name).collect();
        assert_eq!(names, vec![EXPAND_POLYNOMIAL_RULE], "{names:?}");
    }

    /// Shapes the engine leaves (partly) unexpanded.
    fn stuck_shapes(pool: &ExprPool) -> Vec<ExprId> {
        let (x, y, _) = xyz(pool);
        let s = pool.add(vec![x, y, pool.integer(1)]);
        vec![
            // Past ExpandPow's bound: 3^9 > 4096.
            pool.pow(s, pool.integer(9)),
            // (x·y)^n is not distributed.
            pool.pow(pool.mul(vec![x, y]), pool.integer(3)),
            // (x/2)^n is not distributed.
            pool.pow(pool.mul(vec![pool.rational(1, 2), x]), pool.integer(3)),
            // (−x)^odd stands.
            pool.pow(pool.mul(vec![pool.integer(-1), x]), pool.integer(3)),
            // Copies of one sum merge into a power ExpandPow refuses.
            pool.mul(vec![s; 9]),
            // A constant power past ConstFold's bit budget.
            pool.pow(pool.integer(3), pool.integer(100_000)),
            // x^0 and negative powers are not expansions.
            pool.pow(s, pool.integer(0)),
            pool.pow(s, pool.integer(-2)),
        ]
    }

    /// Where the engine stops short the fast path must not go further.
    #[test]
    fn declines_where_the_engine_stops_short() {
        let pool = ExprPool::new();
        for e in stuck_shapes(&pool) {
            assert!(
                Expander::new(&pool).nf(e).is_none(),
                "took {}",
                pool.display(e)
            );
        }
        for i in 0..stuck_shapes(&pool).len() {
            assert_same(|p| {
                let (x, y, _) = xyz(p);
                let t = pool_sum(p, x, y);
                p.add(vec![
                    p.mul(vec![stuck_shapes(p)[i], t]),
                    p.pow(t, p.integer(3)),
                ])
            });
        }
    }

    fn pool_sum(p: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
        p.add(vec![x, p.mul(vec![p.integer(2), y]), p.integer(3)])
    }

    /// An exponent past `u32` is refused, never wrapped, and an expansion
    /// the pre-flight cannot fit is refused rather than allocated.
    #[test]
    fn size_and_exponent_refusals() {
        let pool = ExprPool::new();
        let (x, y, z) = xyz(&pool);
        let huge = pool.pow(x, pool.integer(1u64 << 33));
        assert!(Expander::new(&pool).nf(huge).is_none());
        let m = pool.mul(vec![pool.pow(x, pool.integer(u32::MAX)), x]);
        assert!(Expander::new(&pool).nf(m).is_none());

        // 4^31 terms: refused by the product pre-flight under a memory
        // budget, not attempted.
        let _guard =
            crate::budget::enter_with_memory(crate::budget::Budget::new(), Some(256 << 20));
        let sum = |k: u32| {
            pool.add(vec![
                pool.pow(x, pool.integer(k)),
                pool.pow(y, pool.integer(k)),
                pool.pow(z, pool.integer(k)),
                pool.integer(1),
            ])
        };
        let a: Vec<ExprId> = (0..31).map(|i| sum(1 << i)).collect();
        assert!(Expander::new(&pool).nf(pool.mul(a)).is_none());
    }

    // ------------------------------------------------------------------
    // Differential proptest: random polynomial(-ish) expressions.
    // ------------------------------------------------------------------

    #[derive(Clone, Debug)]
    enum T {
        X,
        Y,
        Z,
        Int(i64),
        Rat(i64, i64),
        Add(Vec<T>),
        Mul(Vec<T>),
        Pow(Box<T>, i64),
        SymPow(Box<T>),
        Sin(Box<T>),
        Repeat(Box<T>, usize),
    }

    fn arb_t() -> impl Strategy<Value = T> {
        let leaf = prop_oneof![
            Just(T::X),
            Just(T::Y),
            Just(T::Z),
            (-3i64..=3).prop_map(T::Int),
            ((-3i64..=3), (2i64..=4)).prop_map(|(n, d)| T::Rat(n, d)),
        ];
        leaf.prop_recursive(4, 40, 4, |inner| {
            prop_oneof![
                4 => proptest::collection::vec(inner.clone(), 2..=4).prop_map(T::Add),
                4 => proptest::collection::vec(inner.clone(), 2..=3).prop_map(T::Mul),
                3 => (
                    inner.clone(),
                    prop_oneof![
                        Just(-1i64),
                        Just(0),
                        Just(1),
                        Just(2),
                        Just(3),
                        Just(4),
                        Just(5),
                        Just(7)
                    ]
                )
                    .prop_map(|(b, n)| T::Pow(Box::new(b), n)),
                1 => inner.clone().prop_map(|b| T::SymPow(Box::new(b))),
                1 => inner.clone().prop_map(|b| T::Sin(Box::new(b))),
                1 => (inner, 2usize..=6).prop_map(|(b, k)| T::Repeat(Box::new(b), k)),
            ]
        })
    }

    fn depth(t: &T) -> usize {
        match t {
            T::Add(v) | T::Mul(v) => 1 + v.iter().map(depth).max().unwrap_or(0),
            T::Pow(b, _) | T::SymPow(b) | T::Sin(b) | T::Repeat(b, _) => 1 + depth(b),
            _ => 0,
        }
    }

    fn to_expr(t: &T, pool: &ExprPool) -> ExprId {
        let (x, y, z) = xyz(pool);
        match t {
            T::X => x,
            T::Y => y,
            T::Z => z,
            T::Int(n) => pool.integer(*n),
            T::Rat(n, d) => pool.rational(*n, *d),
            T::Add(v) => pool.add(v.iter().map(|a| to_expr(a, pool)).collect()),
            T::Mul(v) => pool.mul(v.iter().map(|a| to_expr(a, pool)).collect()),
            T::Pow(b, n) => {
                // Keep the reference run cheap: a high power only of a
                // shallow base.
                let n = if *n > 4 && depth(b) > 1 { 2 } else { *n };
                pool.pow(to_expr(b, pool), pool.integer(n))
            }
            T::SymPow(b) => pool.pow(to_expr(b, pool), y),
            T::Sin(b) => pool.func("sin", vec![to_expr(b, pool)]),
            T::Repeat(b, k) => {
                let e = to_expr(b, pool);
                pool.mul(vec![e; *k])
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn fast_path_matches_rule_engine(t in arb_t()) {
            let r = check_same(&|p: &ExprPool| to_expr(&t, p));
            prop_assert!(r.is_ok(), "{}", r.unwrap_err());
        }
    }
}
