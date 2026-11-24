//! Common-pair elimination: the second half of the compile pipeline's
//! optimiser (after the Horner pass of [`super::optimize`]).
//!
//! Hash-consing already evaluates every *identical* subexpression once. What
//! it cannot share is a pair of operands that occurs inside several n-ary
//! sums or products next to different neighbours:
//!
//! ```text
//! a·b·sin(x) + a·b·cos(x)        Mul[a, b, sin x], Mul[a, b, cos x]
//!   →  t = a·b;  t·sin(x) + t·cos(x)
//! ```
//!
//! The pass counts, over every commutative `Add` and `Mul` node of the DAG
//! (all outputs at once, so the entries of a Jacobian share their pairs), how
//! many nodes contain each unordered pair of operands, hoists the most
//! frequent pairs into a shared two-operand node, and repeats until no pair
//! occurs twice or a round or work cap is reached. This is the greedy scheme
//! of Symbolica's evaluator optimiser. A hoisted node is itself an operand in
//! the next round, so the pass builds nested temporaries for longer shared
//! runs. Pairs are counted in hashed tables keyed by the two operand ids;
//! only operands that occur in at least two nodes of the same kind are
//! considered, at most [`CANDIDATES_PER_NODE`] per node, under a per-round
//! [`PAIR_BUDGET`], so a sum of 10⁵ terms costs one linear scan per round.
//!
//! # What changes in the evaluation
//!
//! A rewritten sum or product is evaluated in a different order, so the pass
//! is part of the same opt-in as the Horner pass and the rewritten nodes
//! join the set of reassociated `Add` nodes ([`Cpe::reassociated`]) that a
//! backend may evaluate without the `+0.0` seed and with fused multiply-adds.
//! Powers are left as they are: their double-double evaluation is more
//! accurate than any product of rounded shared powers, and the backends'
//! value numbering already shares the squaring chains of `x³` and `x⁵`.

use crate::kernel::{ExprData, ExprId, ExprPool, IdBuildHasher, IdMap, IdSet};
use std::collections::HashMap;

/// Most rounds of counting and hoisting.
const MAX_ROUNDS: usize = 64;
/// Most pairs hoisted in one run of the pass.
const MAX_HOISTS: usize = 100_000;
/// Operands of one node considered for pairs (the most frequent ones).
pub(crate) const CANDIDATES_PER_NODE: usize = 24;
/// Pair-count increments allowed per round.
pub(crate) const PAIR_BUDGET: usize = 4_000_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Add,
    Mul,
}

impl Op {
    fn idx(self) -> usize {
        match self {
            Op::Add => 0,
            Op::Mul => 1,
        }
    }

    fn intern(self, mut args: Vec<ExprId>, pool: &ExprPool) -> ExprId {
        // `intern`, not `add`/`mul`: those would splice a nested node back
        // into its parent and undo the hoisting.
        args.sort_unstable();
        pool.intern(match self {
            Op::Add => ExprData::Add(args),
            Op::Mul => ExprData::Mul(args),
        })
    }
}

/// One commutative n-ary node of the DAG, or a hoisted temporary.
struct Node {
    op: Op,
    orig: ExprId,
    args: Vec<ExprId>,
    changed: bool,
}

/// The result of the pass.
pub(crate) struct Cpe {
    /// The rewritten outputs (the inputs themselves when nothing was hoisted).
    pub(crate) outputs: Vec<ExprId>,
    /// Old node → new node for every node reached (unchanged ones map to
    /// themselves), so callers can carry other node sets across.
    pub(crate) remap: IdMap<ExprId>,
    /// `Add` nodes the pass rewrote or built: reassociated sums.
    pub(crate) reassociated: IdSet,
    /// Pairs hoisted.
    pub(crate) hoisted: usize,
    /// Binary operations saved, by the pass's own count (one per occurrence
    /// of a hoisted pair, less one for the temporary unless it existed).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) saved: u64,
}

impl Cpe {
    /// `set` carried across the rewrite.
    pub(crate) fn remap_set(&self, set: &IdSet) -> IdSet {
        set.iter()
            .map(|id| self.remap.get(id).copied().unwrap_or(*id))
            .collect()
    }
}

fn pair_key(a: ExprId, b: ExprId) -> u64 {
    debug_assert!(a < b);
    (u64::from(a.0) << 32) | u64::from(b.0)
}

fn unpack(k: u64) -> (ExprId, ExprId) {
    (ExprId((k >> 32) as u32), ExprId(k as u32))
}

/// Children the pass walks through; `None` for a node it treats as a leaf.
fn children(id: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    pool.with(id, |d| match d {
        ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => a.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        _ => Vec::new(),
    })
}

/// Remove one occurrence of `x` from `v`.
fn remove_one(v: &mut Vec<ExprId>, x: ExprId) {
    if let Some(i) = v.iter().position(|&y| y == x) {
        v.swap_remove(i);
    }
}

/// Run common-pair elimination over `outputs`, not looking below the nodes
/// in `stop` (they are opaque operands).
pub(crate) fn eliminate(outputs: &[ExprId], pool: &ExprPool, stop: &IdSet) -> Cpe {
    // ── Collect the DAG and its commutative n-ary nodes ─────────────────
    let mut in_dag = IdSet::default();
    let mut nodes: Vec<Node> = Vec::new();
    let mut node_of: IdMap<usize> = IdMap::default();
    let mut stack: Vec<ExprId> = outputs.to_vec();
    while let Some(id) = stack.pop() {
        if !in_dag.insert(id) || stop.contains(&id) {
            continue;
        }
        let entry = pool.with(id, |d| match d {
            ExprData::Add(a) if a.len() >= 2 => Some((Op::Add, a.clone())),
            ExprData::Mul(a) if a.len() >= 2 && pool.is_mult_commutative(id) => {
                Some((Op::Mul, a.clone()))
            }
            _ => None,
        });
        if let Some((op, args)) = entry {
            node_of.insert(id, nodes.len());
            nodes.push(Node {
                op,
                orig: id,
                args,
                changed: false,
            });
        }
        stack.extend(children(id, pool));
    }

    let mut hoisted = 0usize;
    let mut saved = 0u64;
    let mut temps = IdSet::default();
    let mut scratch: Vec<ExprId> = Vec::new();

    for _round in 0..MAX_ROUNDS {
        if hoisted >= MAX_HOISTS {
            break;
        }
        // How many nodes of each kind contain each operand.
        let mut freq: [IdMap<u32>; 2] = [IdMap::default(), IdMap::default()];
        for n in &nodes {
            scratch.clear();
            scratch.extend_from_slice(&n.args);
            scratch.sort_unstable();
            scratch.dedup();
            let f = &mut freq[n.op.idx()];
            for &a in &scratch {
                *f.entry(a).or_insert(0) += 1;
            }
        }
        // Pair occurrence counts, and which nodes hold each candidate.
        let mut pairs: [HashMap<u64, u32, IdBuildHasher>; 2] =
            [HashMap::default(), HashMap::default()];
        let mut holders: [IdMap<Vec<usize>>; 2] = [IdMap::default(), IdMap::default()];
        let mut budget = PAIR_BUDGET;
        'count: for (ni, n) in nodes.iter().enumerate() {
            let f = &freq[n.op.idx()];
            scratch.clear();
            scratch.extend(n.args.iter().copied().filter(|a| f[a] >= 2));
            scratch.sort_unstable();
            scratch.dedup();
            if scratch.len() < 2 {
                continue;
            }
            if scratch.len() > CANDIDATES_PER_NODE {
                scratch.sort_unstable_by(|a, b| f[b].cmp(&f[a]).then(a.cmp(b)));
                scratch.truncate(CANDIDATES_PER_NODE);
                scratch.sort_unstable();
            }
            let h = &mut holders[n.op.idx()];
            for &a in &scratch {
                h.entry(a).or_default().push(ni);
            }
            let p = &mut pairs[n.op.idx()];
            for i in 0..scratch.len() {
                for j in i + 1..scratch.len() {
                    *p.entry(pair_key(scratch[i], scratch[j])).or_insert(0) += 1;
                    budget -= 1;
                    if budget == 0 {
                        break 'count;
                    }
                }
            }
        }
        let mut ranked: Vec<(u32, usize, u64)> = Vec::new();
        for (k, p) in pairs.iter().enumerate() {
            ranked.extend(p.iter().filter(|e| *e.1 >= 2).map(|(&key, &c)| (c, k, key)));
        }
        if ranked.is_empty() {
            break;
        }
        // Most frequent first; ties in a fixed order, so the pass is
        // deterministic.
        ranked.sort_unstable_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));

        let mut progressed = false;
        for &(_, k, key) in &ranked {
            if hoisted >= MAX_HOISTS {
                break;
            }
            let op = if k == 0 { Op::Add } else { Op::Mul };
            let (a, b) = unpack(key);
            let (ha, hb) = (&holders[k][&a], &holders[k][&b]);
            let list = if ha.len() <= hb.len() { ha } else { hb };
            // Earlier hoists this round may have taken `a` or `b` out of a
            // node, so the count is re-checked on the nodes themselves.
            let targets: Vec<usize> = list
                .iter()
                .copied()
                .filter(|&ni| {
                    let n = &nodes[ni];
                    n.args.len() >= 3 && n.args.contains(&a) && n.args.contains(&b)
                })
                .collect();
            if targets.is_empty() {
                continue;
            }
            let temp = op.intern(vec![a, b], pool);
            let exists = in_dag.contains(&temp);
            let gain = targets.len() as u64 - u64::from(!exists);
            if gain == 0 {
                continue;
            }
            for &ni in &targets {
                let n = &mut nodes[ni];
                remove_one(&mut n.args, a);
                remove_one(&mut n.args, b);
                n.args.push(temp);
                n.changed = true;
            }
            if !exists {
                in_dag.insert(temp);
                node_of.insert(temp, nodes.len());
                nodes.push(Node {
                    op,
                    orig: temp,
                    args: vec![a, b],
                    changed: false,
                });
                temps.insert(temp);
            }
            hoisted += 1;
            saved += gain;
            progressed = true;
        }
        if !progressed {
            break;
        }
    }

    if hoisted == 0 {
        return Cpe {
            outputs: outputs.to_vec(),
            remap: IdMap::default(),
            reassociated: IdSet::default(),
            hoisted,
            saved,
        };
    }

    // ── Rebuild bottom-up (iterative: Horner chains can be deep) ────────
    let kids = |id: ExprId| -> Vec<ExprId> {
        if stop.contains(&id) {
            return Vec::new();
        }
        match node_of.get(&id) {
            Some(&i) if nodes[i].changed => nodes[i].args.clone(),
            _ => children(id, pool),
        }
    };
    let mut remap: IdMap<ExprId> = IdMap::default();
    let mut work: Vec<(ExprId, bool)> = outputs.iter().rev().map(|&o| (o, false)).collect();
    while let Some((id, expanded)) = work.pop() {
        if remap.contains_key(&id) {
            continue;
        }
        if !expanded {
            work.push((id, true));
            for c in kids(id) {
                if !remap.contains_key(&c) {
                    work.push((c, false));
                }
            }
            continue;
        }
        let new = if stop.contains(&id) {
            id
        } else if let Some(&i) = node_of.get(&id).filter(|&&i| nodes[i].changed) {
            let n = &nodes[i];
            let args = n.args.iter().map(|c| remap[c]).collect();
            n.op.intern(args, pool)
        } else {
            rebuild(id, &remap, pool)
        };
        remap.insert(id, new);
    }

    let mut reassociated = IdSet::default();
    for n in &nodes {
        if n.op == Op::Add && (n.changed || temps.contains(&n.orig)) {
            if let Some(&new) = remap.get(&n.orig) {
                reassociated.insert(new);
            }
        }
    }
    Cpe {
        outputs: outputs.iter().map(|o| remap[o]).collect(),
        remap,
        reassociated,
        hoisted,
        saved,
    }
}

/// `id` with its children replaced through `remap`; the node itself when
/// none changed.
fn rebuild(id: ExprId, remap: &IdMap<ExprId>, pool: &ExprPool) -> ExprId {
    let m = |c: &ExprId| remap.get(c).copied().unwrap_or(*c);
    match pool.get(id) {
        ExprData::Add(a) => {
            let new: Vec<ExprId> = a.iter().map(m).collect();
            if new == a {
                id
            } else {
                Op::Add.intern(new, pool)
            }
        }
        ExprData::Mul(a) => {
            let new: Vec<ExprId> = a.iter().map(m).collect();
            if new == a {
                id
            } else if pool.is_mult_commutative(id) {
                Op::Mul.intern(new, pool)
            } else {
                pool.intern(ExprData::Mul(new))
            }
        }
        ExprData::Pow { base, exp } => {
            let (b, e) = (m(&base), m(&exp));
            if (b, e) == (base, exp) {
                id
            } else {
                pool.pow(b, e)
            }
        }
        ExprData::Func { name, args } => {
            let new: Vec<ExprId> = args.iter().map(m).collect();
            if new == args {
                id
            } else {
                pool.func(name, new)
            }
        }
        _ => id,
    }
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

    #[test]
    fn shared_product_pair_is_hoisted_once() {
        let pool = ExprPool::new();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let x = pool.symbol("x", Domain::Real);
        let s = pool.func("sin", vec![x]);
        let c = pool.func("cos", vec![x]);
        let e = pool.add(vec![pool.mul(vec![a, b, s]), pool.mul(vec![a, b, c])]);
        let r = eliminate(&[e], &pool, &IdSet::default());
        assert_eq!(r.hoisted, 1);
        assert_eq!(r.saved, 1);
        let ab = pool.intern(ExprData::Mul({
            let mut v = vec![a, b];
            v.sort_unstable();
            v
        }));
        // Both products now use the one `a·b` node.
        let out = r.outputs[0];
        let terms = match pool.get(out) {
            ExprData::Add(t) => t,
            other => panic!("{other:?}"),
        };
        for t in terms {
            match pool.get(t) {
                ExprData::Mul(f) => assert!(f.contains(&ab), "{}", pool.display(t)),
                other => panic!("{other:?}"),
            }
        }
        let env = [(a, 0.3), (b, -1.7), (x, 0.9)];
        let (want, got) = (eval(e, &env, &pool), eval(out, &env, &pool));
        assert!((want - got).abs() <= 1e-15 * want.abs().max(1.0));
    }

    #[test]
    fn nothing_shared_is_the_identity() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let terms: Vec<ExprId> = (1..=10)
            .map(|k| {
                pool.mul(vec![
                    pool.rational(1, k),
                    pool.func("sin", vec![pool.mul(vec![pool.integer(k), x])]),
                    pool.func("cos", vec![pool.mul(vec![pool.integer(k), y])]),
                ])
            })
            .collect();
        let e = pool.add(terms);
        let r = eliminate(&[e], &pool, &IdSet::default());
        assert_eq!(r.hoisted, 0);
        assert_eq!(r.outputs, vec![e]);
    }

    #[test]
    fn pairs_are_shared_across_outputs_and_nest() {
        // Three outputs each containing a·b·c next to a different factor:
        // a·b is hoisted, then (a·b)·c.
        let pool = ExprPool::new();
        let v: Vec<ExprId> = ["a", "b", "c", "p", "q", "r"]
            .iter()
            .map(|n| pool.symbol(*n, Domain::Real))
            .collect();
        let (a, b, c) = (v[0], v[1], v[2]);
        let outs: Vec<ExprId> = v[3..].iter().map(|&f| pool.mul(vec![a, b, c, f])).collect();
        let r = eliminate(&outs, &pool, &IdSet::default());
        assert!(r.hoisted >= 2, "{}", r.hoisted);
        // 3 outputs × 3 multiplications = 9 before; after: 2 temporaries +
        // 3 × 1 = 5.
        assert_eq!(r.saved, 4);
        let env: Vec<(ExprId, f64)> = v
            .iter()
            .zip([0.5, -1.25, 2.0, 3.0, -0.75, 1.5])
            .map(|(&s, x)| (s, x))
            .collect();
        for (o, n) in outs.iter().zip(&r.outputs) {
            let (want, got) = (eval(*o, &env, &pool), eval(*n, &env, &pool));
            assert!((want - got).abs() <= 1e-15 * want.abs().max(1.0));
        }
    }

    #[test]
    fn existing_two_operand_node_is_reused() {
        // x·y already exists on its own, so hoisting it out of x·y·z is free.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let xy = pool.mul(vec![x, y]);
        let e = pool.add(vec![pool.func("sin", vec![xy]), pool.mul(vec![x, y, z])]);
        let r = eliminate(&[e], &pool, &IdSet::default());
        assert_eq!(r.hoisted, 1);
        assert_eq!(r.saved, 1);
    }

    #[test]
    fn sums_are_marked_reassociated() {
        let pool = ExprPool::new();
        let v: Vec<ExprId> = ["a", "b", "p", "q"]
            .iter()
            .map(|n| pool.symbol(*n, Domain::Real))
            .collect();
        let s1 = pool.add(vec![v[0], v[1], v[2]]);
        let s2 = pool.add(vec![v[0], v[1], v[3]]);
        let e = pool.mul(vec![pool.func("sin", vec![s1]), pool.func("cos", vec![s2])]);
        let r = eliminate(&[e], &pool, &IdSet::default());
        assert_eq!(r.hoisted, 1);
        // The two rewritten sums and the temporary a + b.
        assert_eq!(r.reassociated.len(), 3);
    }

    #[test]
    fn stop_nodes_are_opaque() {
        let pool = ExprPool::new();
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let m1 = pool.mul(vec![a, b, x]);
        let m2 = pool.mul(vec![a, b, y]);
        let e = pool.add(vec![m1, m2]);
        let stop: IdSet = [m1, m2].into_iter().collect();
        let r = eliminate(&[e], &pool, &stop);
        assert_eq!(r.hoisted, 0);
    }
}
