//! Polynomial pass of the compile pipeline: multivariate Horner schemes.
//!
//! Before an expression is lowered, every sum in it that is a polynomial in
//! "atoms" — the compiled function's inputs, and any non-polynomial
//! subexpression such as `sin(k·x)`, `exp(-(x²+y²))` or `(1+x)^(1/2)` — is
//! rebuilt as a multivariate Horner scheme (see `crate::horner::multi`).
//! Atoms are themselves rewritten first, so a polynomial argument of `sin`
//! benefits too. The rebuilt expression is hash-consed into the same pool, so
//! the coefficient polynomials that several outputs (a Jacobian's entries)
//! share stay shared, and the backends' per-node lowering evaluates each once.
//!
//! On an expanded dense polynomial the effect is large: `(1+x+y)^20` written
//! out has 231 terms and 40 distinct powers, each power a double-double
//! square-and-multiply; the scheme is 230 multiply-adds.
//!
//! # Blocked chains
//!
//! A Horner chain is a serial dependency: in `(1+x+y)^20` the longest chain
//! is twenty multiply-adds, each waiting for the last, and on an
//! out-of-order core the scheme's 230 multiply-adds then took twice the time
//! their throughput allows. So the pass cuts every chain longer than
//! eight links into blocks of [`JIT_BLOCK`] links, combined by a second
//! chain in `v⁴` (two shared squarings per variable): a few more operations,
//! a quarter of the dependency depth, and on `poly20` about 51 → 34 ns per
//! point on a Zen 2 core. The public [`horner_multivariate`] keeps one chain
//! per variable.
//!
//! [`horner_multivariate`]: crate::horner_multivariate
//!
//! # When a sum is rewritten
//!
//! Only when the scheme is clearly cheaper by an operation count
//! ([`MIN_SAVING_PERCENT`], [`MIN_SAVING_OPS`]) and the sum has at least
//! [`MIN_TERMS`] terms; otherwise it is left exactly as written. A sum of
//! unrelated transcendental terms (`Σ sin(kx)·cos(ky)/k`) has nothing to
//! factor and is untouched.
//!
//! # Rounding
//!
//! A rewritten sum is evaluated in a different order, and the backends may
//! fuse its `c + v·acc` links into one multiply-add, so its value can differ
//! from [`eval_interp`](super::eval_interp) of the original in the last few
//! bits. Horner evaluation is backward stable — its error is bounded by the
//! condition number of the polynomial times a few ulps, the same bound as the
//! expanded form, without the powers' own rounding — but where a polynomial is
//! ill-conditioned (heavy cancellation near a root) the two forms can disagree
//! at the level of that condition number. Sums the pass leaves alone keep the
//! backends' bit-exact agreement with the interpreter. The rewrite is
//! switched off with [`CodegenOptions`](super::CodegenOptions) (`horner:
//! false`) or the environment variable `ALKAHEST_JIT_HORNER=0`.
//!
//! # Common-pair elimination
//!
//! After the Horner pass, [`super::cpe`] hoists operand pairs that recur in
//! several n-ary sums and products (across all outputs) into shared nodes.
//! It runs on every Horner candidate — the greedy scheme, and for sums of up
//! to [`ALT_MAX_TERMS`] terms a second scheme with the opposite tie-break —
//! and the cheaper result by operation count (no FMA credit, as for the
//! gate above) is kept; then once more over the whole DAG, kept only when it
//! lowers the FMA-aware operation count. The sums it rewrites are
//! reassociated like the Horner links and round accordingly. It is switched
//! off with [`CodegenOptions`](super::CodegenOptions) (`cpe: false`) or the
//! environment variable `ALKAHEST_JIT_CPE=0`.

use super::cpe;
use crate::horner::multi::{small_positive_exponent, SchemeBuilder, Term, JIT_BLOCK};
use crate::kernel::{ExprData, ExprId, ExprPool, IdMap, IdSet};

/// Fewest terms a sum needs before it is considered for rewriting.
pub(crate) const MIN_TERMS: usize = 6;
/// The scheme must save at least this share of the operations…
pub(crate) const MIN_SAVING_PERCENT: u64 = 20;
/// … and at least this many operations.
pub(crate) const MIN_SAVING_OPS: u64 = 4;
/// Largest nesting depth a rewritten sum may have.
const MAX_SCHEME_DEPTH: u32 = 4096;
/// Sums with more terms than this are left alone (compile-time guard: the
/// scheme itself is `O(n log n)`, but the backend then compiles every term).
pub(crate) const MAX_TERMS: usize = 200_000;
/// Largest sum for which a second Horner candidate is built and costed.
pub(crate) const ALT_MAX_TERMS: usize = 4096;

/// The rewritten outputs, and the scheme links a backend may fuse.
pub(crate) struct Lowered {
    pub(crate) outputs: Vec<ExprId>,
    /// `Add` nodes built as Horner links (`c + v·acc`): a backend with a
    /// hardware FMA may evaluate each `Mul` argument with a fused
    /// multiply-add into the running sum.
    pub(crate) fma_adds: IdSet,
}

impl Lowered {
    /// The identity lowering (pass disabled).
    pub(crate) fn identity(outputs: &[ExprId]) -> Self {
        Lowered {
            outputs: outputs.to_vec(),
            fma_adds: IdSet::default(),
        }
    }
}

/// Run the passes over `outputs`, compiled as functions of `inputs`: the
/// Horner rewrite when `horner`, common-pair elimination when `cpe`.
pub(crate) fn lower(
    outputs: &[ExprId],
    inputs: &[ExprId],
    pool: &ExprPool,
    horner: bool,
    cpe: bool,
) -> Lowered {
    let (mut outputs, mut fma_adds) = if horner {
        let mut r = Rewriter {
            pool,
            inputs: inputs.iter().copied().collect(),
            is_const: IdMap::default(),
            memo: IdMap::default(),
            fma_adds: IdSet::default(),
            cpe,
        };
        let outs: Vec<ExprId> = outputs.iter().map(|&o| r.rewrite(o)).collect();
        (outs, r.fma_adds)
    } else {
        (outputs.to_vec(), IdSet::default())
    };
    if cpe {
        let r = cpe::eliminate(&outputs, pool, &IdSet::default());
        if r.hoisted > 0 {
            let mut fused = r.remap_set(&fma_adds);
            fused.extend(r.reassociated.iter().copied());
            // The final choice credits fused multiply-adds: what matters
            // here is what the backend will execute.
            let none = IdSet::default();
            let before = op_cost_with(&outputs, &none, &fma_adds, true, pool);
            let after = op_cost_with(&r.outputs, &none, &fused, true, pool);
            // As for the Horner gate: rewriting a small expression to save
            // an operation or two is not worth giving up its bit-exact
            // agreement with the interpreter (a reassociated product can
            // overflow where the original did not, or the reverse). The
            // gate counts the shared operations themselves, not the `+0.0`
            // seeds a rewritten sum also drops.
            if r.saved >= MIN_SAVING_OPS && after < before {
                outputs = r.outputs;
                fma_adds = fused;
            }
        }
    }
    Lowered { outputs, fma_adds }
}

struct Rewriter<'p> {
    pool: &'p ExprPool,
    inputs: IdSet,
    is_const: IdMap<bool>,
    memo: IdMap<ExprId>,
    fma_adds: IdSet,
    /// Run common-pair elimination on each Horner candidate.
    cpe: bool,
}

fn children(id: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
    pool.with(id, |d| match d {
        ExprData::Symbol { .. }
        | ExprData::Integer(_)
        | ExprData::Rational(_)
        | ExprData::Float(_) => Some(Vec::new()),
        ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => Some(a.clone()),
        ExprData::Pow { base, exp } => Some(vec![*base, *exp]),
        _ => None,
    })
}

impl Rewriter<'_> {
    /// No input in the subtree (named constants such as `pi` included). A
    /// node kind the pass does not look into counts as not constant, which
    /// only means it is left alone.
    fn is_const(&mut self, id: ExprId) -> bool {
        if let Some(&b) = self.is_const.get(&id) {
            return b;
        }
        let r = if self.inputs.contains(&id) {
            false
        } else {
            match children(id, self.pool) {
                Some(ch) => ch.into_iter().all(|c| self.is_const(c)),
                None => false,
            }
        };
        self.is_const.insert(id, r);
        r
    }

    fn rewrite(&mut self, id: ExprId) -> ExprId {
        if let Some(&r) = self.memo.get(&id) {
            return r;
        }
        let r = if self.is_const(id) {
            id
        } else {
            let data = self.pool.get(id);
            match data {
                ExprData::Add(args) => self.rewrite_sum(id, &args),
                ExprData::Mul(args) => {
                    let new: Vec<ExprId> = args.iter().map(|&a| self.rewrite(a)).collect();
                    if new == args {
                        id
                    } else {
                        self.pool.mul(new)
                    }
                }
                ExprData::Pow { base, exp } => {
                    let (b, e) = (self.rewrite(base), self.rewrite(exp));
                    if (b, e) == (base, exp) {
                        id
                    } else {
                        self.pool.pow(b, e)
                    }
                }
                ExprData::Func { name, args } => {
                    let new: Vec<ExprId> = args.iter().map(|&a| self.rewrite(a)).collect();
                    if new == args {
                        id
                    } else {
                        self.pool.func(name, new)
                    }
                }
                _ => id,
            }
        };
        self.memo.insert(id, r);
        r
    }

    /// The sum with its terms rewritten but not restructured.
    fn plain_sum(&mut self, id: ExprId, args: &[ExprId]) -> ExprId {
        let new: Vec<ExprId> = args.iter().map(|&a| self.rewrite(a)).collect();
        if new == args {
            id
        } else {
            self.pool.add(new)
        }
    }

    fn rewrite_sum(&mut self, id: ExprId, args: &[ExprId]) -> ExprId {
        if args.len() < MIN_TERMS || args.len() > MAX_TERMS {
            return self.plain_sum(id, args);
        }
        // Read the sum as a polynomial in atoms.
        let mut atom_index: IdMap<u32> = IdMap::default();
        let mut atoms: Vec<ExprId> = Vec::new();
        // Original subexpressions standing for an atom or a coefficient: the
        // cost of the original form is counted down to them.
        let mut old_stop = IdSet::default();
        let mut new_stop = IdSet::default();
        let mut terms: Vec<Term> = Vec::with_capacity(args.len());
        for &s in args {
            let factors = self.pool.with(s, |d| match d {
                ExprData::Mul(a) => a.clone(),
                _ => vec![s],
            });
            let mut t = Term {
                coef: Vec::new(),
                exps: Vec::new(),
            };
            for f in factors {
                if self.is_const(f) {
                    t.coef.push(f);
                    old_stop.insert(f);
                    new_stop.insert(f);
                    continue;
                }
                let (orig, e) = self.pool.with(f, |d| match d {
                    ExprData::Pow { base, exp } => match small_positive_exponent(*exp, self.pool) {
                        Some(e) => (*base, e),
                        None => (f, 1),
                    },
                    _ => (f, 1),
                });
                old_stop.insert(orig);
                let atom = self.rewrite(orig);
                new_stop.insert(atom);
                let next = atoms.len() as u32;
                let idx = *atom_index.entry(atom).or_insert_with(|| {
                    atoms.push(atom);
                    next
                });
                if !t.push_power(idx, e) {
                    return self.plain_sum(id, args);
                }
            }
            terms.push(t);
        }
        if atoms.is_empty() {
            return self.plain_sum(id, args);
        }
        let n_candidates = if self.cpe && atoms.len() >= 2 && terms.len() <= ALT_MAX_TERMS {
            2
        } else {
            1
        };
        // (cost, scheme, nodes the backend may fuse)
        let mut best: Option<(u64, ExprId, IdSet)> = None;
        for k in 0..n_candidates {
            let mut b = SchemeBuilder::new(self.pool, atoms.clone()).with_blocks(JIT_BLOCK);
            b.prefer_last = k == 1;
            let t = if k + 1 == n_candidates {
                std::mem::take(&mut terms)
            } else {
                terms.clone()
            };
            let (mut scheme, depth) = b.build(t);
            if depth > MAX_SCHEME_DEPTH {
                continue;
            }
            let mut links: IdSet = b.chain_links.into_iter().collect();
            // Sums CPE reassociated are evaluated without the `+0.0` seed,
            // which the cost reflects. The Horner links are not credited the
            // same way, so with nothing to hoist the decision is exactly the
            // one made without CPE.
            let mut reassociated = IdSet::default();
            if self.cpe {
                let r = cpe::eliminate(&[scheme], self.pool, &new_stop);
                if r.hoisted > 0 {
                    links = r.remap_set(&links);
                    scheme = r.outputs[0];
                    reassociated = r.reassociated;
                }
            }
            // Fused multiply-adds are not credited here: they would make any
            // nesting of unrelated terms look cheaper (each link saves an
            // operation) while only serialising the sum. The scheme has to
            // win on structure — shared factors, no powers — alone.
            let cost = op_cost_with(&[scheme], &new_stop, &reassociated, false, self.pool);
            if best.as_ref().map_or(true, |b| cost < b.0) {
                links.extend(reassociated);
                best = Some((cost, scheme, links));
            }
        }
        let Some((new_cost, scheme, links)) = best else {
            return self.plain_sum(id, args);
        };
        let old_cost = op_cost(id, &old_stop, &IdSet::default(), self.pool);
        let saving = old_cost.saturating_sub(new_cost);
        if saving < MIN_SAVING_OPS || saving * 100 < old_cost * MIN_SAVING_PERCENT {
            return self.plain_sum(id, args);
        }
        self.fma_adds.extend(links);
        scheme
    }
}

/// Floating-point operation estimate for the DAG under `root`, not looking
/// below the nodes in `stop`, crediting fused multiply-adds for the `Add`
/// nodes in `fused` (see [`op_cost_with`]).
pub(crate) fn op_cost(root: ExprId, stop: &IdSet, fused: &IdSet, pool: &ExprPool) -> u64 {
    op_cost_with(&[root], stop, fused, true, pool)
}

/// Floating-point operation estimate for the DAG under `roots`, not looking
/// below the nodes in `stop` (other than the roots themselves).
///
/// An `Add` of `n` operands costs `n` (the backends seed a sum with `+0.0`).
/// A reassociated one — in `fused` — is evaluated without the seed, `n - 1`;
/// with `credit_fma` each `Mul` operand is folded into the running sum by a
/// multiply-add as well, which saves one more operation per such operand.
pub(crate) fn op_cost_with(
    roots: &[ExprId],
    stop: &IdSet,
    fused: &IdSet,
    credit_fma: bool,
    pool: &ExprPool,
) -> u64 {
    let mut seen = IdSet::default();
    let mut stack = roots.to_vec();
    let mut cost = 0u64;
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || (stop.contains(&id) && !roots.contains(&id)) {
            continue;
        }
        pool.with(id, |d| match d {
            ExprData::Add(a) => {
                let mut c = a.len() as u64;
                if fused.contains(&id) {
                    let muls = if credit_fma {
                        a.iter()
                            .filter(|&&x| pool.with(x, |d| matches!(d, ExprData::Mul(_))))
                            .count() as u64
                    } else {
                        0
                    };
                    c = c.saturating_sub(muls).saturating_sub(1);
                }
                cost += c;
                stack.extend_from_slice(a);
            }
            ExprData::Mul(a) => {
                cost += a.len().saturating_sub(1) as u64;
                stack.extend_from_slice(a);
            }
            ExprData::Pow { base, exp } => {
                cost += match small_positive_exponent(*exp, pool) {
                    Some(1) => 0,
                    Some(2) => 1,
                    // Double-double binary powering: about six operations
                    // per squaring and per multiplication.
                    Some(n) => 6 * u64::from(n.ilog2() + n.count_ones()),
                    None => 20,
                };
                stack.push(*base);
            }
            ExprData::Func { args, .. } => {
                cost += 20;
                stack.extend_from_slice(args);
            }
            _ => {}
        });
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::Domain;
    use std::collections::HashMap;

    /// `(1 + x + y)^n`, written out: Σ n!/(i!·j!·(n-i-j)!) · x^i · y^j.
    pub(crate) fn dense(pool: &ExprPool, x: ExprId, y: ExprId, n: u32) -> ExprId {
        let fact = |k: u32| (1..=k).fold(rug::Integer::from(1), |a, b| a * b);
        let mut terms = Vec::new();
        for i in 0..=n {
            for j in 0..=(n - i) {
                let c = fact(n) / (fact(i) * fact(j) * fact(n - i - j));
                let pw = |v: ExprId, e: u32| {
                    if e == 1 {
                        v
                    } else {
                        pool.pow(v, pool.integer(e))
                    }
                };
                let mut f = Vec::new();
                if c != 1 {
                    f.push(pool.integer(c));
                }
                if i > 0 {
                    f.push(pw(x, i));
                }
                if j > 0 {
                    f.push(pw(y, j));
                }
                terms.push(match f.len() {
                    0 => pool.integer(1),
                    1 => f[0],
                    _ => pool.mul(f),
                });
            }
        }
        pool.add(terms)
    }

    #[test]
    fn dense_bivariate_is_rewritten_and_agrees() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = dense(&pool, x, y, 12);
        let l = lower(&[e], &[x, y], &pool, true, true);
        assert_ne!(l.outputs[0], e);
        assert!(!l.fma_adds.is_empty());
        let stop = IdSet::default();
        assert!(
            op_cost(l.outputs[0], &stop, &l.fma_adds, &pool) * 3
                < op_cost(e, &stop, &IdSet::default(), &pool)
        );
        for &(a, b) in &[(0.05, 0.95), (0.5, 0.5), (-0.7, 0.2), (1.3, -2.1)] {
            let env: HashMap<ExprId, f64> = [(x, a), (y, b)].into_iter().collect();
            let want: f64 = (1.0 + a + b).powi(12);
            let got = eval_interp(l.outputs[0], &env, &pool).unwrap();
            // Σ|cₖ xⁱ yʲ| = (1 + |x| + |y|)^12 bounds the rounding error of
            // any evaluation order; (1.3, -2.1) cancels heavily.
            let scale = (1.0 + a.abs() + b.abs()).powi(12);
            assert!(
                (got - want).abs() <= 64.0 * f64::EPSILON * scale,
                "{got} vs {want}"
            );
        }
    }

    #[test]
    fn unrelated_transcendental_terms_are_untouched() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let terms: Vec<ExprId> = (1..=10)
            .map(|k| {
                let kx = pool.mul(vec![pool.integer(k), x]);
                let ky = pool.mul(vec![pool.integer(k), y]);
                pool.mul(vec![
                    pool.rational(1, k),
                    pool.func("sin", vec![kx]),
                    pool.func("cos", vec![ky]),
                ])
            })
            .collect();
        let e = pool.add(terms);
        let l = lower(&[e], &[x, y], &pool, true, true);
        assert_eq!(l.outputs[0], e);
        assert!(l.fma_adds.is_empty());
    }

    #[test]
    fn small_sums_are_untouched() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let i = |n: i32| pool.integer(n);
        let e = pool.add(vec![
            pool.mul(vec![pool.rational(2, 5), pool.pow(x, i(3))]),
            pool.mul(vec![pool.rational(-7, 3), pool.pow(x, i(2))]),
            pool.mul(vec![pool.rational(355, 113), x]),
            pool.rational(1, 7),
        ]);
        assert_eq!(lower(&[e], &[x], &pool, true, true).outputs[0], e);
    }

    #[test]
    fn polynomial_inside_function_and_as_coefficient() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let p = dense(&pool, x, y, 6);
        let s = pool.func("sin", vec![p]);
        // Σ_k x^k·y^(6-k)·cos(x) + sin(p): the cos(x) coefficients are a
        // polynomial too.
        let c = pool.func("cos", vec![x]);
        let mut terms: Vec<ExprId> = (0..=6)
            .map(|k| {
                pool.mul(vec![
                    pool.integer(k + 1),
                    pool.pow(x, pool.integer(k)),
                    pool.pow(y, pool.integer(6 - k)),
                    c,
                ])
            })
            .collect();
        terms.push(s);
        let e = pool.add(terms);
        let l = lower(&[e], &[x, y], &pool, true, true);
        assert_ne!(l.outputs[0], e);
        let env: HashMap<ExprId, f64> = [(x, 0.3), (y, 0.6)].into_iter().collect();
        let want = eval_interp(e, &env, &pool).unwrap();
        let got = eval_interp(l.outputs[0], &env, &pool).unwrap();
        assert!((got - want).abs() <= 1e-13 * want.abs().max(1.0));
    }

    #[test]
    fn shared_outputs_stay_shared() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let p = dense(&pool, x, y, 8);
        let a = pool.func("exp", vec![p]);
        let b = pool.func("sin", vec![p]);
        let l = lower(&[a, b], &[x, y], &pool, true, true);
        let arg = |e: ExprId| {
            pool.with(e, |d| match d {
                ExprData::Func { args, .. } => args[0],
                _ => unreachable!(),
            })
        };
        assert_eq!(arg(l.outputs[0]), arg(l.outputs[1]));
        assert_ne!(arg(l.outputs[0]), p);
    }

    /// `Σ_{k=1}^{30} sin(kx)·cos(ky)/k + exp(-(x²+y²))·log(1+xy)`, the
    /// comparison harness's `trig30`.
    fn trig30(pool: &ExprPool, x: ExprId, y: ExprId) -> ExprId {
        let mut terms: Vec<ExprId> = (1..=30)
            .map(|k| {
                let kx = pool.mul(vec![pool.integer(k), x]);
                let ky = pool.mul(vec![pool.integer(k), y]);
                let mut f = vec![pool.func("sin", vec![kx]), pool.func("cos", vec![ky])];
                if k > 1 {
                    f.push(pool.rational(1, k));
                }
                pool.mul(f)
            })
            .collect();
        let r2 = pool.add(vec![
            pool.pow(x, pool.integer(2)),
            pool.pow(y, pool.integer(2)),
        ]);
        let g = pool.func("exp", vec![pool.mul(vec![pool.integer(-1), r2])]);
        let l = pool.func(
            "log",
            vec![pool.add(vec![pool.integer(1), pool.mul(vec![x, y])])],
        );
        terms.push(pool.mul(vec![g, l]));
        pool.add(terms)
    }

    fn cost_of(l: &Lowered, pool: &ExprPool) -> u64 {
        op_cost_with(&l.outputs, &IdSet::default(), &l.fma_adds, true, pool)
    }

    /// The operation counts of `poly20` through each pass. Hash-consing
    /// already shares every repeated two-operand node of the scheme, and a
    /// dense polynomial with 231 distinct coefficients leaves no other pair
    /// to share, so CPE must not make it worse (Symbolica's evaluator ends
    /// at the same count: 457 instructions, about 228 multiply-add pairs).
    #[test]
    fn poly20_operation_count() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = dense(&pool, x, y, 20);
        let none = IdSet::default();
        let written = op_cost_with(&[e], &none, &none, true, &pool);
        let horner = lower(&[e], &[x, y], &pool, true, false);
        let both = lower(&[e], &[x, y], &pool, true, true);
        let (ch, cb) = (cost_of(&horner, &pool), cost_of(&both, &pool));
        eprintln!("poly20 operations: as written {written}, Horner {ch}, Horner + CPE {cb}");
        assert!(cb <= ch, "CPE made poly20 dearer: {cb} > {ch}");
        assert!(cb * 3 < written, "{cb} vs {written}");
        // 231 coefficients: about 230 multiply-adds, plus about one more
        // per four links for the blocked chains and their shared powers.
        assert!(cb <= 300, "{cb}");
    }

    #[test]
    fn trig30_is_untouched_by_both_passes() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = trig30(&pool, x, y);
        let l = lower(&[e], &[x, y], &pool, true, true);
        assert_eq!(l.outputs[0], e);
        assert!(l.fma_adds.is_empty());
    }

    /// The entries of a Jacobian-shaped vector share the pairs `a·b` and
    /// `a·b·c` with different neighbours; CPE computes them once for all
    /// outputs.
    #[test]
    fn pairs_are_shared_across_outputs() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let s = pool.func("sin", vec![x]);
        let c = pool.func("cos", vec![y]);
        let ex = pool.func("exp", vec![pool.mul(vec![x, y])]);
        let x2 = pool.pow(x, pool.integer(2));
        let outs: Vec<ExprId> = [s, c, ex, x2]
            .iter()
            .map(|&f| pool.mul(vec![x, y, s, c, f, pool.integer(3)]))
            .collect();
        let none = IdSet::default();
        let plain = op_cost_with(&outs, &none, &none, true, &pool);
        let l = lower(&outs, &[x, y], &pool, true, true);
        let after = cost_of(&l, &pool);
        assert!(after + 8 <= plain, "{after} vs {plain}");
        for &(a, b) in &[(0.3, 0.7), (-1.2, 0.4), (2.5, -0.9)] {
            let env: HashMap<ExprId, f64> = [(x, a), (y, b)].into_iter().collect();
            for (o, n) in outs.iter().zip(&l.outputs) {
                let want = eval_interp(*o, &env, &pool).unwrap();
                let got = eval_interp(*n, &env, &pool).unwrap();
                assert!(
                    (got - want).abs() <= 8.0 * f64::EPSILON * want.abs(),
                    "{got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn cpe_alone_and_neither_pass() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let fs = [
            pool.func("sin", vec![x]),
            pool.func("cos", vec![x]),
            pool.func("exp", vec![y]),
            pool.func("tan", vec![y]),
            pool.func("sin", vec![y]),
        ];
        let e = pool.add(
            fs.iter()
                .map(|&f| pool.mul(vec![pool.integer(3), x, y, f]))
                .collect(),
        );
        // CPE without the Horner pass still shares 3·x·y.
        let l = lower(&[e], &[x, y], &pool, false, true);
        assert_ne!(l.outputs[0], e);
        // Horner alone has nothing to do on a five-term sum.
        assert_eq!(lower(&[e], &[x, y], &pool, true, false).outputs[0], e);
    }

    /// Hoisting one pair saves one operation: not worth reassociating a
    /// small expression (and giving up its bit-exact agreement with the
    /// interpreter) for.
    #[test]
    fn a_saving_below_the_gate_is_not_taken() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = pool.add(vec![
            pool.mul(vec![x, y, pool.func("sin", vec![x])]),
            pool.mul(vec![x, y, pool.func("cos", vec![x])]),
        ]);
        assert_eq!(lower(&[e], &[x, y], &pool, true, true).outputs[0], e);
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(128))]

        /// Random sums of products over a small operand set (so pairs
        /// recur), several outputs at once: the rewritten outputs agree with
        /// the originals within the rounding bound of a reassociated sum.
        #[test]
        fn cpe_agrees_with_the_expression_as_written(
            picks in proptest::collection::vec(
                proptest::collection::vec(proptest::collection::vec(0usize..8, 1..5), 1..9),
                1..4,
            ),
            a in -1.5f64..1.5,
            b in -1.5f64..1.5,
        ) {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let y = pool.symbol("y", Domain::Real);
            let ops = [
                x,
                y,
                pool.func("sin", vec![x]),
                pool.func("cos", vec![y]),
                pool.pow(x, pool.integer(2)),
                pool.integer(3),
                pool.rational(-2, 7),
                pool.func("exp", vec![pool.mul(vec![x, y])]),
            ];
            let env: HashMap<ExprId, f64> = [(x, a), (y, b)].into_iter().collect();
            let mut outs = Vec::new();
            let mut scales = Vec::new();
            for out in &picks {
                let mut terms = Vec::new();
                let mut scale = 0.0f64;
                for t in out {
                    let f: Vec<ExprId> = t.iter().map(|&i| ops[i]).collect();
                    let m = pool.mul(f);
                    scale += eval_interp(m, &env, &pool).unwrap().abs();
                    terms.push(m);
                }
                outs.push(pool.add(terms));
                scales.push(scale);
            }
            for (horner, cpe) in [(false, true), (true, true)] {
                let l = lower(&outs, &[x, y], &pool, horner, cpe);
                for ((o, n), scale) in outs.iter().zip(&l.outputs).zip(&scales) {
                    let want = eval_interp(*o, &env, &pool).unwrap();
                    let got = eval_interp(*n, &env, &pool).unwrap();
                    proptest::prop_assert!(
                        (got - want).abs() <= 64.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE),
                        "{got} vs {want} (scale {scale})"
                    );
                }
            }
        }
    }

    /// Rounding of the compiled Horner scheme against a 256-bit reference,
    /// next to the same function compiled as written.
    #[cfg(feature = "cranelift")]
    mod compiled {
        use super::*;
        use crate::jit::{compile_with_options, CodegenOptions, CompileConfig, CompileTier};
        use proptest::prelude::*;
        use rug::ops::Pow;
        use rug::Float;

        const PREC: u32 = 256;

        fn native(
            e: ExprId,
            inputs: &[ExprId],
            pool: &ExprPool,
            horner: bool,
        ) -> crate::jit::CompiledFn {
            let config = CompileConfig {
                expected_evals: Some(1 << 20),
                force_tier: Some(CompileTier::Cranelift),
            };
            let opts = CodegenOptions::exact().with_horner(horner).with_cpe(horner);
            let f = compile_with_options(e, inputs, pool, config, opts).unwrap();
            assert_eq!(f.compile_tier(), CompileTier::Cranelift);
            f
        }

        /// `(x - y)^n` written out, alternating signs: evaluating it near
        /// `x = y` cancels catastrophically in any order.
        fn binomial_difference(pool: &ExprPool, x: ExprId, y: ExprId, n: u32) -> ExprId {
            let mut c = rug::Integer::from(1);
            let mut terms = Vec::new();
            for k in 0..=n {
                let sign = if k % 2 == 0 { 1 } else { -1 };
                let mut f = vec![pool.integer(c.clone() * sign)];
                if n - k > 0 {
                    f.push(pool.pow(x, pool.integer(n - k)));
                }
                if k > 0 {
                    f.push(pool.pow(y, pool.integer(k)));
                }
                terms.push(pool.mul(f));
                c = c * (n - k) / (k + 1);
            }
            pool.add(terms)
        }

        fn rel(got: f64, want: &Float) -> f64 {
            let w = want.to_f64();
            ((Float::with_val(PREC, got) - want).abs().to_f64()) / w.abs().max(1.0)
        }

        #[test]
        fn poly20_relative_error_stays_at_rounding_level() {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let y = pool.symbol("y", Domain::Real);
            let e = dense(&pool, x, y, 20);
            let (h, plain) = (
                native(e, &[x, y], &pool, true),
                native(e, &[x, y], &pool, false),
            );
            let (mut worst_h, mut worst_p) = (0.0f64, 0.0f64);
            let mut state = 0x2545_F491_4F6C_DD1Du64;
            let mut unit = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            for _ in 0..20_000 {
                let (a, b) = (0.05 + 0.9 * unit(), 0.05 + 0.9 * unit());
                let want = (Float::with_val(PREC, a) + b + 1u32).pow(20u32);
                worst_h = worst_h.max(rel(h.call(&[a, b]), &want));
                worst_p = worst_p.max(rel(plain.call(&[a, b]), &want));
            }
            assert!(worst_h < 1e-14, "horner {worst_h:e} (plain {worst_p:e})");
        }

        /// A vector compile with both passes on (pairs shared across the
        /// outputs, a Horner polynomial among them) against the
        /// interpreter on the expressions as written.
        #[test]
        fn vector_compile_with_cpe_matches_the_interpreter() {
            use crate::jit::compile_many_with_options;
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let y = pool.symbol("y", Domain::Real);
            let s = pool.func("sin", vec![x]);
            let c = pool.func("cos", vec![y]);
            let p = dense(&pool, x, y, 8);
            let mut outs: Vec<ExprId> = [s, c, p]
                .iter()
                .map(|&f| pool.mul(vec![x, y, s, c, f]))
                .collect();
            outs.push(p);
            outs.push(pool.add(vec![pool.mul(vec![x, y, s]), pool.mul(vec![x, y, c]), x]));
            let config = CompileConfig {
                expected_evals: Some(1 << 20),
                force_tier: Some(CompileTier::Cranelift),
            };
            for opts in [
                CodegenOptions::exact().with_horner(true).with_cpe(true),
                CodegenOptions::exact().with_cpe(true),
            ] {
                let f = compile_many_with_options(&outs, &[x, y], &pool, config, opts).unwrap();
                let mut got = vec![0.0; outs.len()];
                for &(a, b) in &[(0.3, 0.7), (-1.2, 0.4), (0.9, -0.95), (0.05, 0.05)] {
                    f.call(&[a, b], &mut got);
                    let env: HashMap<ExprId, f64> = [(x, a), (y, b)].into_iter().collect();
                    for (k, &o) in outs.iter().enumerate() {
                        let want = eval_interp(o, &env, &pool).unwrap();
                        assert!(
                            (got[k] - want).abs() <= 1e-13 * want.abs().max(1.0),
                            "output {k} at ({a}, {b}): {} vs {want}",
                            got[k]
                        );
                    }
                }
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// Near `x = y` both forms lose the digits the condition number
            /// says they must; the scheme must stay within the classical
            /// bound `γ·Σ|cₖ xⁱ yʲ|`, and within a small factor of the error
            /// of the form as written.
            #[test]
            fn cancellation_stays_within_the_condition_bound(
                a in -2.0f64..2.0,
                d in prop_oneof![-1e-3f64..1e-3, -1e-9f64..1e-9, -1.0f64..1.0],
            ) {
                let pool = ExprPool::new();
                let x = pool.symbol("x", Domain::Real);
                let y = pool.symbol("y", Domain::Real);
                let n = 9u32;
                let e = binomial_difference(&pool, x, y, n);
                let b = a + d;
                let want = (Float::with_val(PREC, a) - b).pow(n);
                // Σ |c_k| |a|^(n-k) |b|^k = (|a| + |b|)^n
                let scale = (a.abs() + b.abs()).powi(n as i32);
                let gamma = 4.0 * f64::from(2 * n + 2) * f64::EPSILON;
                let err = |v: f64| (Float::with_val(PREC, v) - &want).abs().to_f64();
                let eh = err(native(e, &[x, y], &pool, true).call(&[a, b]));
                let ep = err(native(e, &[x, y], &pool, false).call(&[a, b]));
                prop_assert!(eh <= gamma * scale + 1e-300, "horner err {eh:e}, bound {:e}", gamma * scale);
                prop_assert!(eh <= 8.0 * ep + gamma * scale * 0.25, "horner {eh:e} vs plain {ep:e}");
            }

            /// Random dense bivariate polynomials with signed coefficients:
            /// compiled scheme against the 256-bit value of the same sum.
            #[test]
            fn random_polynomials_agree(
                coeffs in proptest::collection::vec(-1000i32..1000, 10..40),
                a in -1.5f64..1.5,
                b in -1.5f64..1.5,
            ) {
                let pool = ExprPool::new();
                let x = pool.symbol("x", Domain::Real);
                let y = pool.symbol("y", Domain::Real);
                let mut terms = Vec::new();
                let mut want = Float::with_val(PREC, 0);
                let mut scale = 0.0f64;
                for (k, &c) in coeffs.iter().enumerate() {
                    let (i, j) = ((k % 7) as u32, (k / 7) as u32);
                    let mut f = vec![pool.integer(c)];
                    if i > 0 { f.push(pool.pow(x, pool.integer(i))); }
                    if j > 0 { f.push(pool.pow(y, pool.integer(j))); }
                    terms.push(pool.mul(f));
                    let m = Float::with_val(PREC, a).pow(i) * Float::with_val(PREC, b).pow(j);
                    want += m.clone() * c;
                    scale += (f64::from(c) * m.to_f64()).abs();
                }
                let e = pool.add(terms);
                let got = native(e, &[x, y], &pool, true).call(&[a, b]);
                let err = (Float::with_val(PREC, got) - &want).abs().to_f64();
                prop_assert!(err <= 64.0 * f64::EPSILON * scale + 1e-300, "{err:e} vs scale {scale:e}");
            }
        }
    }
}
