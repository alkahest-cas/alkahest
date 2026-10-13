//! Reduced ordered binary decision diagrams (ROBDDs).
//!
//! A [`BddManager`] owns a node table in which every Boolean function over
//! its variables has exactly one node: the diagram is *reduced* (no node with
//! equal children, no two nodes with the same `(level, lo, hi)`) and *ordered*
//! (variables appear along every path in the manager's fixed level order).
//! Canonicity is what makes the structure useful: two functions are equivalent
//! exactly when they are the same [`BddRef`], so equivalence is an integer
//! comparison once both are built.
//!
//! Every operation is reduced to [`BddManager::ite`] (if-then-else), memoised
//! on its three operands.  Restriction, existential and universal
//! quantification and model counting are the usual recursive passes.
//!
//! # Size is bounded, never assumed
//!
//! A BDD can be exponentially larger than the formula it represents — the
//! middle output bit of an `n`-bit multiplier needs `Ω(2^{n/8})` nodes under
//! every variable order (Bryant 1991).  The manager therefore carries a node
//! ceiling ([`BddManager::with_max_nodes`]) and consults the active
//! [`crate::budget::Budget`] as it allocates.  Exceeding either is reported as
//! [`BooleanError::TooLarge`] / [`BooleanError::Budget`]; nothing is truncated.

use super::boolean::BooleanError;
use std::collections::{HashMap, HashSet};

/// Handle to a node of a [`BddManager`].  Only meaningful with the manager
/// that issued it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BddRef(u32);

impl BddRef {
    /// The constant-false function.
    pub const FALSE: BddRef = BddRef(0);
    /// The constant-true function.
    pub const TRUE: BddRef = BddRef(1);

    /// `true` for the constant-true terminal.
    pub fn is_true(self) -> bool {
        self == BddRef::TRUE
    }

    /// `true` for the constant-false terminal.
    pub fn is_false(self) -> bool {
        self == BddRef::FALSE
    }

    /// `true` for either terminal.
    pub fn is_const(self) -> bool {
        self.0 < 2
    }

    /// Raw node index (stable for the manager's lifetime).
    pub fn index(self) -> u32 {
        self.0
    }
}

/// Level of the two terminals: below every variable.
const TERMINAL_LEVEL: u32 = u32::MAX;
/// Level marker of a node slot on the free list.
const DEAD_LEVEL: u32 = u32::MAX - 1;

/// Default node ceiling of [`BddManager::new`].
pub const DEFAULT_MAX_BDD_NODES: usize = 2_000_000;

#[derive(Debug, Clone, Copy)]
struct Node {
    level: u32,
    lo: BddRef,
    hi: BddRef,
}

/// Node table, unique table and operation caches of a family of ROBDDs that
/// share one variable order.
#[derive(Debug, Clone)]
pub struct BddManager {
    nodes: Vec<Node>,
    unique: HashMap<(u32, BddRef, BddRef), BddRef>,
    ite_cache: HashMap<(BddRef, BddRef, BddRef), BddRef>,
    /// Variable name at each level (level 0 is the root-most variable).
    names: Vec<String>,
    by_name: HashMap<String, u32>,
    max_nodes: usize,
    /// Slots of collected nodes, reused by [`BddManager::mk`].
    free: Vec<u32>,
    /// Functions the caller holds across a reordering (see [`BddManager::pin`]).
    pinned: HashMap<BddRef, usize>,
    /// Live-node count past which [`BddManager::maybe_reorder`] sifts.
    auto_reorder: Option<usize>,
    /// Current trigger: `auto_reorder`, raised to twice the size left by the
    /// last reordering so a stubbornly large diagram is not re-sifted on every
    /// operation.
    next_trigger: usize,
}

impl Default for BddManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BddManager {
    /// Empty manager with the default node ceiling
    /// ([`DEFAULT_MAX_BDD_NODES`]).
    pub fn new() -> Self {
        Self::with_max_nodes(DEFAULT_MAX_BDD_NODES)
    }

    /// Empty manager that refuses to hold more than `max_nodes` nodes.
    pub fn with_max_nodes(max_nodes: usize) -> Self {
        let term = |_| Node {
            level: TERMINAL_LEVEL,
            lo: BddRef::FALSE,
            hi: BddRef::FALSE,
        };
        BddManager {
            nodes: vec![term(0), term(1)],
            unique: HashMap::new(),
            ite_cache: HashMap::new(),
            names: Vec::new(),
            by_name: HashMap::new(),
            max_nodes: max_nodes.max(2),
            free: Vec::new(),
            pinned: HashMap::new(),
            auto_reorder: None,
            next_trigger: usize::MAX,
        }
    }

    /// Node ceiling.
    pub fn max_nodes(&self) -> usize {
        self.max_nodes
    }

    /// Total nodes allocated in the manager (including the two terminals and
    /// nodes no longer reachable from any live function).
    pub fn allocated_nodes(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    /// Variable names in level order.
    pub fn var_names(&self) -> &[String] {
        &self.names
    }

    /// Number of variables declared.
    pub fn num_vars(&self) -> usize {
        self.names.len()
    }

    /// Level of `name`, if declared.
    pub fn level_of(&self, name: &str) -> Option<u32> {
        self.by_name.get(name).copied()
    }

    /// Declare `name` (appended below every existing variable) and return its
    /// level; returns the existing level when already declared.
    pub fn declare(&mut self, name: &str) -> u32 {
        if let Some(&l) = self.by_name.get(name) {
            return l;
        }
        let l = self.names.len() as u32;
        self.names.push(name.to_string());
        self.by_name.insert(name.to_string(), l);
        l
    }

    /// The projection function of `name` (declaring it if needed).
    pub fn var(&mut self, name: &str) -> Result<BddRef, BooleanError> {
        let l = self.declare(name);
        self.mk(l, BddRef::FALSE, BddRef::TRUE)
    }

    /// The projection function of the variable at `level`.
    pub fn var_at(&mut self, level: u32) -> Result<BddRef, BooleanError> {
        if level as usize >= self.names.len() {
            return Err(BooleanError::InvalidArgument(format!(
                "BDD level {level} is not declared"
            )));
        }
        self.mk(level, BddRef::FALSE, BddRef::TRUE)
    }

    /// Level of the top variable of `f` (`None` for a terminal).
    pub fn top_level(&self, f: BddRef) -> Option<u32> {
        let l = self.nodes[f.0 as usize].level;
        (l != TERMINAL_LEVEL).then_some(l)
    }

    /// `(lo, hi)` children of a non-terminal node.
    pub fn children(&self, f: BddRef) -> Option<(BddRef, BddRef)> {
        if f.is_const() {
            return None;
        }
        let n = self.nodes[f.0 as usize];
        Some((n.lo, n.hi))
    }

    fn level(&self, f: BddRef) -> u32 {
        self.nodes[f.0 as usize].level
    }

    fn mk(&mut self, level: u32, lo: BddRef, hi: BddRef) -> Result<BddRef, BooleanError> {
        if lo == hi {
            return Ok(lo);
        }
        if let Some(&r) = self.unique.get(&(level, lo, hi)) {
            return Ok(r);
        }
        let live = self.nodes.len() - self.free.len();
        if live >= self.max_nodes {
            return Err(BooleanError::TooLarge {
                what: "BDD nodes",
                limit: self.max_nodes as u64,
            });
        }
        if live % 1024 == 0 {
            crate::budget::check().map_err(BooleanError::Budget)?;
        }
        let r = self.alloc(Node { level, lo, hi });
        self.unique.insert((level, lo, hi), r);
        Ok(r)
    }

    fn alloc(&mut self, n: Node) -> BddRef {
        if let Some(i) = self.free.pop() {
            self.nodes[i as usize] = n;
            BddRef(i)
        } else {
            let r = BddRef(self.nodes.len() as u32);
            self.nodes.push(n);
            r
        }
    }

    fn cofactors(&self, f: BddRef, level: u32) -> (BddRef, BddRef) {
        let n = self.nodes[f.0 as usize];
        if n.level == level {
            (n.lo, n.hi)
        } else {
            (f, f)
        }
    }

    /// `if f then g else h`.
    pub fn ite(&mut self, f: BddRef, g: BddRef, h: BddRef) -> Result<BddRef, BooleanError> {
        if f.is_true() {
            return Ok(g);
        }
        if f.is_false() {
            return Ok(h);
        }
        if g == h {
            return Ok(g);
        }
        if g.is_true() && h.is_false() {
            return Ok(f);
        }
        if let Some(&r) = self.ite_cache.get(&(f, g, h)) {
            return Ok(r);
        }
        let top = self.level(f).min(self.level(g)).min(self.level(h));
        let (f0, f1) = self.cofactors(f, top);
        let (g0, g1) = self.cofactors(g, top);
        let (h0, h1) = self.cofactors(h, top);
        let lo = self.ite(f0, g0, h0)?;
        let hi = self.ite(f1, g1, h1)?;
        let r = self.mk(top, lo, hi)?;
        self.ite_cache.insert((f, g, h), r);
        // The operation cache is bounded too: an apply can visit up to
        // |f|·|g| operand pairs while reusing existing nodes, so the node
        // ceiling alone does not bound its memory or time.
        let cached = self.ite_cache.len();
        if cached % 65_536 == 0 {
            crate::budget::check().map_err(BooleanError::Budget)?;
            let cap = self.max_nodes.saturating_mul(8).max(1 << 20);
            if cached > cap {
                return Err(BooleanError::TooLarge {
                    what: "BDD operation cache",
                    limit: cap as u64,
                });
            }
        }
        Ok(r)
    }

    /// `¬f`.
    pub fn not(&mut self, f: BddRef) -> Result<BddRef, BooleanError> {
        self.ite(f, BddRef::FALSE, BddRef::TRUE)
    }

    /// `f ∧ g`.
    pub fn and(&mut self, f: BddRef, g: BddRef) -> Result<BddRef, BooleanError> {
        self.ite(f, g, BddRef::FALSE)
    }

    /// `f ∨ g`.
    pub fn or(&mut self, f: BddRef, g: BddRef) -> Result<BddRef, BooleanError> {
        self.ite(f, BddRef::TRUE, g)
    }

    /// `f ⊕ g`.
    pub fn xor(&mut self, f: BddRef, g: BddRef) -> Result<BddRef, BooleanError> {
        let ng = self.not(g)?;
        self.ite(f, ng, g)
    }

    /// `f ↔ g`.
    pub fn iff(&mut self, f: BddRef, g: BddRef) -> Result<BddRef, BooleanError> {
        let ng = self.not(g)?;
        self.ite(f, g, ng)
    }

    /// `f → g`.
    pub fn implies(&mut self, f: BddRef, g: BddRef) -> Result<BddRef, BooleanError> {
        self.ite(f, g, BddRef::TRUE)
    }

    /// Cofactor `f|_{x_level = value}`.
    pub fn restrict(&mut self, f: BddRef, level: u32, value: bool) -> Result<BddRef, BooleanError> {
        let mut memo = HashMap::new();
        self.restrict_rec(f, level, value, &mut memo)
    }

    fn restrict_rec(
        &mut self,
        f: BddRef,
        level: u32,
        value: bool,
        memo: &mut HashMap<BddRef, BddRef>,
    ) -> Result<BddRef, BooleanError> {
        let n = self.nodes[f.0 as usize];
        if n.level == TERMINAL_LEVEL || n.level > level {
            return Ok(f);
        }
        if n.level == level {
            return Ok(if value { n.hi } else { n.lo });
        }
        if let Some(&r) = memo.get(&f) {
            return Ok(r);
        }
        let lo = self.restrict_rec(n.lo, level, value, memo)?;
        let hi = self.restrict_rec(n.hi, level, value, memo)?;
        let r = self.mk(n.level, lo, hi)?;
        memo.insert(f, r);
        Ok(r)
    }

    /// `∃ levels . f` (`forall = false`) or `∀ levels . f` (`forall = true`).
    pub fn quantify(
        &mut self,
        f: BddRef,
        levels: &[u32],
        forall: bool,
    ) -> Result<BddRef, BooleanError> {
        let mut set = vec![false; self.names.len()];
        for &l in levels {
            if let Some(s) = set.get_mut(l as usize) {
                *s = true;
            }
        }
        let mut memo = HashMap::new();
        self.quantify_rec(f, &set, forall, &mut memo)
    }

    fn quantify_rec(
        &mut self,
        f: BddRef,
        set: &[bool],
        forall: bool,
        memo: &mut HashMap<BddRef, BddRef>,
    ) -> Result<BddRef, BooleanError> {
        if f.is_const() {
            return Ok(f);
        }
        if let Some(&r) = memo.get(&f) {
            return Ok(r);
        }
        let n = self.nodes[f.0 as usize];
        let lo = self.quantify_rec(n.lo, set, forall, memo)?;
        let hi = self.quantify_rec(n.hi, set, forall, memo)?;
        let r = if set.get(n.level as usize).copied().unwrap_or(false) {
            if forall {
                self.and(lo, hi)?
            } else {
                self.or(lo, hi)?
            }
        } else {
            self.mk(n.level, lo, hi)?
        };
        memo.insert(f, r);
        Ok(r)
    }

    /// Number of non-terminal nodes reachable from `f`.
    pub fn node_count(&self, f: BddRef) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![f];
        while let Some(x) = stack.pop() {
            if x.is_const() || !seen.insert(x) {
                continue;
            }
            let n = self.nodes[x.0 as usize];
            stack.push(n.lo);
            stack.push(n.hi);
        }
        seen.len()
    }

    /// Levels of the variables `f` depends on, ascending.
    pub fn support(&self, f: BddRef) -> Vec<u32> {
        let mut seen = std::collections::HashSet::new();
        let mut levels = std::collections::BTreeSet::new();
        let mut stack = vec![f];
        while let Some(x) = stack.pop() {
            if x.is_const() || !seen.insert(x) {
                continue;
            }
            let n = self.nodes[x.0 as usize];
            levels.insert(n.level);
            stack.push(n.lo);
            stack.push(n.hi);
        }
        levels.into_iter().collect()
    }

    /// Number of assignments to **all** `num_vars` declared variables of the
    /// manager that satisfy `f`.  `num_vars` must be at least the number of
    /// declared variables; extra variables each double the count.
    pub fn sat_count(&self, f: BddRef, num_vars: u32) -> rug::Integer {
        let declared = self.names.len() as u32;
        let n = num_vars.max(declared);
        let mut memo: HashMap<BddRef, rug::Integer> = HashMap::new();
        // count(x) = models of x over the variables at levels ≥ level(x).
        fn lvl(m: &BddManager, x: BddRef, n: u32) -> u32 {
            if x.is_const() {
                n
            } else {
                m.nodes[x.0 as usize].level
            }
        }
        fn go(
            m: &BddManager,
            x: BddRef,
            n: u32,
            memo: &mut HashMap<BddRef, rug::Integer>,
        ) -> rug::Integer {
            if x.is_false() {
                return rug::Integer::new();
            }
            if x.is_true() {
                return rug::Integer::from(1);
            }
            if let Some(c) = memo.get(&x) {
                return c.clone();
            }
            let node = m.nodes[x.0 as usize];
            let l = node.level;
            let clo = go(m, node.lo, n, memo) << (lvl(m, node.lo, n) - l - 1);
            let chi = go(m, node.hi, n, memo) << (lvl(m, node.hi, n) - l - 1);
            let c = clo + chi;
            memo.insert(x, c.clone());
            c
        }
        let top = lvl(self, f, n);
        go(self, f, n, &mut memo) << top
    }

    /// One satisfying partial assignment `(level, value)` (variables not
    /// listed are free), or `None` when `f` is unsatisfiable.
    pub fn any_sat(&self, f: BddRef) -> Option<Vec<(u32, bool)>> {
        if f.is_false() {
            return None;
        }
        let mut out = Vec::new();
        let mut x = f;
        while !x.is_const() {
            let n = self.nodes[x.0 as usize];
            if !n.lo.is_false() {
                out.push((n.level, false));
                x = n.lo;
            } else {
                out.push((n.level, true));
                x = n.hi;
            }
        }
        Some(out)
    }

    /// Every path from `f` to the true terminal, as cubes of `(level, value)`
    /// literals.  The cubes are pairwise disjoint and their union is `f`.
    /// Refuses with [`BooleanError::TooLarge`] past `max_cubes` cubes.
    pub fn cubes(
        &self,
        f: BddRef,
        max_cubes: usize,
    ) -> Result<Vec<Vec<(u32, bool)>>, BooleanError> {
        let mut out = Vec::new();
        let mut path = Vec::new();
        self.cubes_rec(f, &mut path, &mut out, max_cubes)?;
        Ok(out)
    }

    fn cubes_rec(
        &self,
        f: BddRef,
        path: &mut Vec<(u32, bool)>,
        out: &mut Vec<Vec<(u32, bool)>>,
        max_cubes: usize,
    ) -> Result<(), BooleanError> {
        if f.is_false() {
            return Ok(());
        }
        if f.is_true() {
            if out.len() >= max_cubes {
                return Err(BooleanError::TooLarge {
                    what: "BDD path cubes",
                    limit: max_cubes as u64,
                });
            }
            out.push(path.clone());
            return Ok(());
        }
        let n = self.nodes[f.0 as usize];
        path.push((n.level, false));
        self.cubes_rec(n.lo, path, out, max_cubes)?;
        path.pop();
        path.push((n.level, true));
        self.cubes_rec(n.hi, path, out, max_cubes)?;
        path.pop();
        Ok(())
    }

    /// `true` when `f` is true at every point of the cube (`cube` maps level →
    /// value; levels absent from it range over both values).
    pub fn holds_on_cube(&self, f: BddRef, cube: &[Option<bool>]) -> bool {
        let mut memo: HashMap<BddRef, bool> = HashMap::new();
        self.holds_rec(f, cube, &mut memo)
    }

    fn holds_rec(
        &self,
        f: BddRef,
        cube: &[Option<bool>],
        memo: &mut HashMap<BddRef, bool>,
    ) -> bool {
        if f.is_true() {
            return true;
        }
        if f.is_false() {
            return false;
        }
        if let Some(&b) = memo.get(&f) {
            return b;
        }
        let n = self.nodes[f.0 as usize];
        let r = match cube.get(n.level as usize).copied().flatten() {
            Some(true) => self.holds_rec(n.hi, cube, memo),
            Some(false) => self.holds_rec(n.lo, cube, memo),
            None => self.holds_rec(n.lo, cube, memo) && self.holds_rec(n.hi, cube, memo),
        };
        memo.insert(f, r);
        r
    }

    /// Evaluate `f` at a total assignment indexed by level.
    pub fn eval(&self, f: BddRef, assignment: &[bool]) -> bool {
        let mut x = f;
        while !x.is_const() {
            let n = self.nodes[x.0 as usize];
            x = if assignment.get(n.level as usize).copied().unwrap_or(false) {
                n.hi
            } else {
                n.lo
            };
        }
        x.is_true()
    }

    /// Copy `f` from `src` into `self`, matching variables **by name**
    /// (declaring any that `self` lacks, below its existing order).
    pub fn transfer_from(&mut self, src: &BddManager, f: BddRef) -> Result<BddRef, BooleanError> {
        let mut memo: HashMap<BddRef, BddRef> = HashMap::new();
        self.transfer_rec(src, f, &mut memo)
    }

    fn transfer_rec(
        &mut self,
        src: &BddManager,
        f: BddRef,
        memo: &mut HashMap<BddRef, BddRef>,
    ) -> Result<BddRef, BooleanError> {
        if f.is_const() {
            return Ok(f);
        }
        if let Some(&r) = memo.get(&f) {
            return Ok(r);
        }
        let n = src.nodes[f.0 as usize];
        let lo = self.transfer_rec(src, n.lo, memo)?;
        let hi = self.transfer_rec(src, n.hi, memo)?;
        let v = self.var(&src.names[n.level as usize])?;
        let r = self.ite(v, hi, lo)?;
        memo.insert(f, r);
        Ok(r)
    }


    // -----------------------------------------------------------------
    // Dynamic variable reordering
    // -----------------------------------------------------------------

    /// Keep `f` alive (and its node index valid) across reorderings.  Calls
    /// nest: each `pin` needs a matching [`BddManager::unpin`].  Terminals
    /// need no pinning.
    pub fn pin(&mut self, f: BddRef) {
        if !f.is_const() {
            *self.pinned.entry(f).or_insert(0) += 1;
        }
    }

    /// Release one [`BddManager::pin`] of `f`.
    pub fn unpin(&mut self, f: BddRef) {
        if let Some(c) = self.pinned.get_mut(&f) {
            *c -= 1;
            if *c == 0 {
                self.pinned.remove(&f);
            }
        }
    }

    /// Enable (`Some(n)`) or disable (`None`) automatic reordering: once the
    /// manager holds more than `n` live nodes, [`BddManager::maybe_reorder`]
    /// runs a sifting pass.  Off by default.
    pub fn set_auto_reorder(&mut self, threshold: Option<usize>) {
        self.auto_reorder = threshold;
        self.next_trigger = threshold.unwrap_or(usize::MAX);
    }

    /// The automatic-reordering threshold, if enabled.
    pub fn auto_reorder_threshold(&self) -> Option<usize> {
        self.auto_reorder
    }

    /// Sift when automatic reordering is enabled and the node count passed
    /// the trigger; otherwise do nothing.  `roots` (plus every pinned
    /// function) are the functions that must survive: **every function the
    /// caller still intends to use must be pinned or listed**, because nodes
    /// reachable from neither are collected.
    pub fn maybe_reorder(
        &mut self,
        roots: &[BddRef],
    ) -> Result<Option<ReorderStats>, BooleanError> {
        if self.auto_reorder.is_none() || self.allocated_nodes() <= self.next_trigger {
            return Ok(None);
        }
        self.reorder(roots, &ReorderOptions::default()).map(Some)
    }

    /// Rudell sifting: move each variable (largest level first) through the
    /// whole order by adjacent swaps and leave it where the diagram of the
    /// functions in `roots` and the pinned ones was smallest.  Every function
    /// keeps its node index and its meaning, so `sat_count`, equivalence and
    /// `to_expr` are unchanged; only [`BddManager::var_names`] (the order)
    /// and the node counts change.  Nodes reachable from no root are
    /// collected.
    ///
    /// Bounded: a variable's excursion is abandoned in a direction once the
    /// diagram exceeds `max_growth ×` the best size seen for it, the pass
    /// stops after [`ReorderOptions::max_swaps`] swaps
    /// ([`ReorderStats::complete`] is then `false`), and the active
    /// [`crate::budget::Budget`] is consulted between swaps
    /// ([`BooleanError::Budget`]; the order reached so far stays valid).
    pub fn reorder(
        &mut self,
        roots: &[BddRef],
        opts: &ReorderOptions,
    ) -> Result<ReorderStats, BooleanError> {
        if !(opts.max_growth >= 1.0) {
            return Err(BooleanError::InvalidArgument(format!(
                "max_growth must be at least 1.0, not {}",
                opts.max_growth
            )));
        }
        self.ite_cache.clear();
        let mut all: Vec<BddRef> = self.pinned.keys().copied().collect();
        all.extend(roots.iter().copied().filter(|r| !r.is_const()));
        all.sort();
        all.dedup();
        let mut s = Sifter::new(self, &all);
        let before = s.live;
        let result = s.run(opts);
        let after = s.live;
        let (swaps, complete) = (s.swaps, s.complete);
        drop(s);
        let floor = self.auto_reorder.unwrap_or(usize::MAX);
        self.next_trigger = floor.max(after.saturating_mul(2));
        result?;
        Ok(ReorderStats {
            nodes_before: before,
            nodes_after: after,
            swaps,
            complete,
        })
    }

    /// Drop the operation cache (the node table is kept).
    pub fn clear_cache(&mut self) {
        self.ite_cache.clear();
    }
}


/// Knobs of [`BddManager::reorder`].
#[derive(Debug, Clone)]
pub struct ReorderOptions {
    /// Abandon a sifting direction once the diagram is larger than this
    /// multiple of the best size seen so far for the variable (Rudell's
    /// growth bound).  Default 1.2.
    pub max_growth: f64,
    /// Stop after this many adjacent-variable swaps.  Default 200 000.
    pub max_swaps: usize,
}

impl Default for ReorderOptions {
    fn default() -> Self {
        ReorderOptions {
            max_growth: 1.2,
            max_swaps: 200_000,
        }
    }
}

/// What a [`BddManager::reorder`] pass did.  Sizes count the live non-terminal
/// nodes of the whole manager (the union of the live functions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReorderStats {
    /// Live nodes before the pass.
    pub nodes_before: usize,
    /// Live nodes after the pass.
    pub nodes_after: usize,
    /// Adjacent swaps performed.
    pub swaps: usize,
    /// `false` when the pass stopped at the swap limit before sifting every
    /// variable.
    pub complete: bool,
}

/// One sifting session: reference counts and per-level node sets are built
/// from the live roots, kept exact across swaps, and dropped at the end.
struct Sifter<'a> {
    m: &'a mut BddManager,
    rc: Vec<u32>,
    lv: Vec<HashSet<u32>>,
    live: usize,
    swaps: usize,
    complete: bool,
}

impl<'a> Sifter<'a> {
    fn new(m: &'a mut BddManager, roots: &[BddRef]) -> Self {
        // Mark.
        let mut reach = vec![false; m.nodes.len()];
        let mut stack: Vec<BddRef> = roots.to_vec();
        while let Some(x) = stack.pop() {
            if x.is_const() || reach[x.0 as usize] {
                continue;
            }
            reach[x.0 as usize] = true;
            let n = m.nodes[x.0 as usize];
            stack.push(n.lo);
            stack.push(n.hi);
        }
        // Sweep.
        m.free.clear();
        for i in 2..m.nodes.len() {
            if reach[i] {
                continue;
            }
            let n = m.nodes[i];
            if n.level != DEAD_LEVEL {
                m.unique.remove(&(n.level, n.lo, n.hi));
                m.nodes[i].level = DEAD_LEVEL;
            }
            m.free.push(i as u32);
        }
        let nlev = m.names.len();
        let mut rc = vec![0u32; m.nodes.len()];
        let mut lv: Vec<HashSet<u32>> = vec![HashSet::new(); nlev];
        let mut live = 0;
        for i in 2..m.nodes.len() {
            if !reach[i] {
                continue;
            }
            let n = m.nodes[i];
            lv[n.level as usize].insert(i as u32);
            live += 1;
            if !n.lo.is_const() {
                rc[n.lo.0 as usize] += 1;
            }
            if !n.hi.is_const() {
                rc[n.hi.0 as usize] += 1;
            }
        }
        for r in roots {
            if !r.is_const() {
                rc[r.0 as usize] += 1;
            }
        }
        Sifter {
            m,
            rc,
            lv,
            live,
            swaps: 0,
            complete: true,
        }
    }

    fn inc(&mut self, x: BddRef) {
        if !x.is_const() {
            self.rc[x.0 as usize] += 1;
        }
    }

    fn dec(&mut self, x: BddRef) {
        let mut stack = vec![x];
        while let Some(x) = stack.pop() {
            if x.is_const() {
                continue;
            }
            let i = x.0 as usize;
            self.rc[i] -= 1;
            if self.rc[i] > 0 {
                continue;
            }
            let n = self.m.nodes[i];
            self.m.unique.remove(&(n.level, n.lo, n.hi));
            self.lv[n.level as usize].remove(&x.0);
            self.m.nodes[i].level = DEAD_LEVEL;
            self.m.free.push(x.0);
            self.live -= 1;
            stack.push(n.lo);
            stack.push(n.hi);
        }
    }

    /// Find-or-create `(level, lo, hi)` without a ceiling (a swap must not
    /// fail half-way).  The result is *borrowed*: a new node starts with
    /// reference count 0 and the caller [`Sifter::inc`]s it when it stores it.
    fn mk(&mut self, level: u32, lo: BddRef, hi: BddRef) -> BddRef {
        if lo == hi {
            return lo;
        }
        if let Some(&r) = self.m.unique.get(&(level, lo, hi)) {
            return r;
        }
        let r = self.m.alloc(Node { level, lo, hi });
        if self.rc.len() < self.m.nodes.len() {
            self.rc.resize(self.m.nodes.len(), 0);
        }
        self.rc[r.0 as usize] = 0;
        self.inc(lo);
        self.inc(hi);
        self.m.unique.insert((level, lo, hi), r);
        self.lv[level as usize].insert(r.0);
        self.live += 1;
        r
    }

    /// Exchange the variables at levels `i` and `i + 1`, rewriting nodes in
    /// place so every node index keeps denoting the same function.
    fn swap(&mut self, i: u32) {
        let up: Vec<u32> = self.lv[i as usize].iter().copied().collect();
        let down: Vec<u32> = self.lv[i as usize + 1].iter().copied().collect();
        // Cofactors of the upper nodes that mention the lower variable.
        let mut dependent: Vec<(u32, [BddRef; 4])> = Vec::new();
        let mut shifted: Vec<u32> = Vec::new();
        for &x in &up {
            let n = self.m.nodes[x as usize];
            let split = |c: BddRef, nodes: &[Node]| -> (BddRef, BddRef, bool) {
                if !c.is_const() && nodes[c.0 as usize].level == i + 1 {
                    let cn = nodes[c.0 as usize];
                    (cn.lo, cn.hi, true)
                } else {
                    (c, c, false)
                }
            };
            let (f00, f01, d0) = split(n.lo, &self.m.nodes);
            let (f10, f11, d1) = split(n.hi, &self.m.nodes);
            if d0 || d1 {
                dependent.push((x, [f00, f01, f10, f11]));
            } else {
                shifted.push(x);
            }
        }
        for &x in up.iter().chain(down.iter()) {
            let n = self.m.nodes[x as usize];
            self.m.unique.remove(&(n.level, n.lo, n.hi));
        }
        self.lv[i as usize].clear();
        self.lv[i as usize + 1].clear();
        // The lower variable's nodes move up a level.
        for &y in &down {
            self.m.nodes[y as usize].level = i;
            let n = self.m.nodes[y as usize];
            self.m.unique.insert((i, n.lo, n.hi), BddRef(y));
            self.lv[i as usize].insert(y);
        }
        // Upper nodes that do not mention it move down unchanged.
        for &x in &shifted {
            self.m.nodes[x as usize].level = i + 1;
            let n = self.m.nodes[x as usize];
            self.m.unique.insert((i + 1, n.lo, n.hi), BddRef(x));
            self.lv[i as usize + 1].insert(x);
        }
        // The rest become nodes of the lower variable over new upper nodes.
        for (x, [f00, f01, f10, f11]) in dependent {
            let old = self.m.nodes[x as usize];
            let nl = self.mk(i + 1, f00, f10);
            let nh = self.mk(i + 1, f01, f11);
            self.inc(nl);
            self.inc(nh);
            self.m.nodes[x as usize].lo = nl;
            self.m.nodes[x as usize].hi = nh;
            self.m.nodes[x as usize].level = i;
            self.m.unique.insert((i, nl, nh), BddRef(x));
            self.lv[i as usize].insert(x);
            self.dec(old.lo);
            self.dec(old.hi);
        }
        self.m.names.swap(i as usize, i as usize + 1);
        let (a, b) = (
            self.m.names[i as usize].clone(),
            self.m.names[i as usize + 1].clone(),
        );
        self.m.by_name.insert(a, i);
        self.m.by_name.insert(b, i + 1);
        self.swaps += 1;
    }

    fn step(&mut self, i: u32, opts: &ReorderOptions) -> Result<bool, BooleanError> {
        if self.swaps >= opts.max_swaps {
            self.complete = false;
            return Ok(false);
        }
        crate::budget::check().map_err(BooleanError::Budget)?;
        self.swap(i);
        Ok(true)
    }

    fn run(&mut self, opts: &ReorderOptions) -> Result<(), BooleanError> {
        let n = self.m.names.len() as u32;
        if n < 2 {
            return Ok(());
        }
        let mut order: Vec<(usize, String)> = (0..n as usize)
            .map(|l| (self.lv[l].len(), self.m.names[l].clone()))
            .collect();
        order.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for (_, name) in order {
            if !self.complete {
                break;
            }
            self.sift(&name, opts)?;
        }
        Ok(())
    }

    fn sift(&mut self, name: &str, opts: &ReorderOptions) -> Result<(), BooleanError> {
        let n = self.m.names.len() as u32;
        let mut pos = self.m.by_name[name];
        let mut best = (self.live, pos);
        let bound = |best: usize| (best as f64 * opts.max_growth).ceil() as usize;
        // Nearer end first.
        let down_first = pos >= n / 2;
        for pass in 0..2 {
            let down = (pass == 0) == down_first;
            if down {
                while pos + 1 < n {
                    if !self.step(pos, opts)? {
                        break;
                    }
                    pos += 1;
                    if self.live < best.0 {
                        best = (self.live, pos);
                    }
                    if self.live > bound(best.0) {
                        break;
                    }
                }
            } else {
                while pos > 0 {
                    if !self.step(pos - 1, opts)? {
                        break;
                    }
                    pos -= 1;
                    if self.live < best.0 {
                        best = (self.live, pos);
                    }
                    if self.live > bound(best.0) {
                        break;
                    }
                }
            }
        }
        // Settle at the best position seen (at most 2n swaps, outside the
        // swap limit so the variable is never stranded mid-excursion).
        while pos < best.1 {
            self.swap(pos);
            pos += 1;
        }
        while pos > best.1 {
            self.swap(pos - 1);
            pos -= 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_and_counts() {
        let mut m = BddManager::new();
        let a = m.var("a").unwrap();
        let b = m.var("b").unwrap();
        let c = m.var("c").unwrap();
        // (a ∧ b) ∨ (a ∧ ¬b) == a
        let ab = m.and(a, b).unwrap();
        let nb = m.not(b).unwrap();
        let anb = m.and(a, nb).unwrap();
        let f = m.or(ab, anb).unwrap();
        assert_eq!(f, a);
        // majority(a,b,c) has 4 models of 8
        let bc = m.and(b, c).unwrap();
        let ac = m.and(a, c).unwrap();
        let t = m.or(ab, bc).unwrap();
        let maj = m.or(t, ac).unwrap();
        assert_eq!(m.sat_count(maj, 3), 4);
        assert_eq!(m.sat_count(maj, 5), 16);
        assert_eq!(m.sat_count(BddRef::TRUE, 3), 8);
        // ∃a. maj = b ∨ c ; ∀a. maj = b ∧ c
        let ex = m.quantify(maj, &[0], false).unwrap();
        let b_or_c = m.or(b, c).unwrap();
        assert_eq!(ex, b_or_c);
        let fa = m.quantify(maj, &[0], true).unwrap();
        assert_eq!(fa, bc);
        // restrict
        let r = m.restrict(maj, 1, true).unwrap();
        let a_or_c = m.or(a, c).unwrap();
        assert_eq!(r, a_or_c);
        // xor self-inverse
        let x = m.xor(maj, a).unwrap();
        let back = m.xor(x, a).unwrap();
        assert_eq!(back, maj);
    }

    /// Truth table of `f` indexed by an assignment to the variables *by name*
    /// (`names[k]` is bit `k`), independent of the manager's current order.
    fn table_by_name(m: &BddManager, f: BddRef, names: &[String]) -> Vec<bool> {
        (0..1usize << names.len())
            .map(|a| {
                let asg: Vec<bool> = m
                    .var_names()
                    .iter()
                    .map(|v| {
                        let k = names.iter().position(|n| n == v).unwrap();
                        (a >> k) & 1 == 1
                    })
                    .collect();
                m.eval(f, &asg)
            })
            .collect()
    }

    fn inner_product(m: &mut BddManager, n: usize, interleaved: bool) -> BddRef {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        if interleaved {
            for i in 0..n {
                xs.push(m.var(&format!("x{i}")).unwrap());
                ys.push(m.var(&format!("y{i}")).unwrap());
            }
        } else {
            for i in 0..n {
                xs.push(m.var(&format!("x{i}")).unwrap());
            }
            for i in 0..n {
                ys.push(m.var(&format!("y{i}")).unwrap());
            }
        }
        let mut f = BddRef::FALSE;
        for i in 0..n {
            let t = m.and(xs[i], ys[i]).unwrap();
            f = m.or(f, t).unwrap();
        }
        f
    }

    #[test]
    fn sifting_repairs_a_bad_order() {
        let n = 8;
        let mut m = BddManager::new();
        let f = inner_product(&mut m, n, false);
        let names: Vec<String> = m.var_names().to_vec();
        let before = m.node_count(f);
        let count = m.sat_count(f, 2 * n as u32);
        let mut good = BddManager::new();
        let g = inner_product(&mut good, n, true);
        let optimum = good.node_count(g);
        assert_eq!(optimum, 2 * n);
        assert!(before > 20 * optimum, "bad order has {before} nodes");
        let table = table_by_name(&m, f, &names);
        let stats = m.reorder(&[f], &ReorderOptions::default()).unwrap();
        let after = m.node_count(f);
        assert!(stats.complete);
        assert_eq!(stats.nodes_after, after);
        assert!(after <= 2 * optimum, "{after} nodes vs optimum {optimum}");
        assert_eq!(m.sat_count(f, 2 * n as u32), count);
        assert_eq!(table_by_name(&m, f, &names), table);
        // Functions built after reordering canonicalise against the old ones.
        let x0 = m.var("x0").unwrap();
        let y0 = m.var("y0").unwrap();
        let t = m.and(x0, y0).unwrap();
        let g2 = m.or(f, t).unwrap();
        assert_eq!(g2, f);
    }

    #[test]
    fn reorder_preserves_random_functions() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..30 {
            let nv = 4 + (round % 5);
            let mut m = BddManager::new();
            let names: Vec<String> = (0..nv).map(|i| format!("v{i}")).collect();
            let vars: Vec<BddRef> = names.iter().map(|n| m.var(n).unwrap()).collect();
            let mut pool = vars.clone();
            for _ in 0..25 {
                let a = pool[(rnd() % pool.len() as u64) as usize];
                let b = pool[(rnd() % pool.len() as u64) as usize];
                let c = pool[(rnd() % pool.len() as u64) as usize];
                let r = match rnd() % 4 {
                    0 => m.and(a, b),
                    1 => m.or(a, b),
                    2 => m.xor(a, b),
                    _ => m.ite(a, b, c),
                }
                .unwrap();
                pool.push(r);
            }
            // Keep three functions, drop the rest (they become garbage).
            let keep: Vec<BddRef> = pool.iter().rev().take(3).copied().collect();
            let tables: Vec<Vec<bool>> = keep
                .iter()
                .map(|&f| table_by_name(&m, f, &names))
                .collect();
            let counts: Vec<_> = keep.iter().map(|&f| m.sat_count(f, nv as u32)).collect();
            let st = m.reorder(&keep, &ReorderOptions::default()).unwrap();
            assert!(st.nodes_after <= st.nodes_before, "{st:?}");
            for (k, &f) in keep.iter().enumerate() {
                assert_eq!(table_by_name(&m, f, &names), tables[k]);
                assert_eq!(m.sat_count(f, nv as u32), counts[k]);
            }
            // Operations after reordering still canonicalise.
            let nf = m.not(keep[0]).unwrap();
            let nnf = m.not(nf).unwrap();
            assert_eq!(nnf, keep[0]);
            let x = m.xor(keep[1], keep[2]).unwrap();
            let back = m.xor(x, keep[2]).unwrap();
            assert_eq!(back, keep[1]);
            // A second pass never grows the diagram.
            let again = m.reorder(&keep, &ReorderOptions::default()).unwrap();
            assert!(again.nodes_after <= again.nodes_before);
        }
    }

    #[test]
    fn auto_reorder_and_swap_limit() {
        let mut m = BddManager::new();
        m.set_auto_reorder(Some(100));
        let f = inner_product(&mut m, 8, false);
        m.pin(f);
        let st = m.maybe_reorder(&[]).unwrap().expect("past the threshold");
        assert!(st.nodes_after < st.nodes_before);
        // Back under the (raised) trigger: no second pass.
        assert!(m.maybe_reorder(&[]).unwrap().is_none());
        m.set_auto_reorder(None);
        assert!(m.maybe_reorder(&[]).unwrap().is_none());
        // A swap limit stops the pass early and says so.
        let mut m2 = BddManager::new();
        let g = inner_product(&mut m2, 8, false);
        let st = m2
            .reorder(
                &[g],
                &ReorderOptions {
                    max_swaps: 5,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(!st.complete);
        assert!(m2
            .reorder(
                &[g],
                &ReorderOptions {
                    max_growth: 0.5,
                    ..Default::default()
                }
            )
            .is_err());
    }

    #[test]
    fn multiplier_bits_shrink_under_sifting() {
        // All product bits of a 4x4 multiplier, operands declared apart.
        let mut m = BddManager::new();
        let n = 4;
        let a: Vec<BddRef> = (0..n).map(|i| m.var(&format!("a{i}")).unwrap()).collect();
        let b: Vec<BddRef> = (0..n).map(|i| m.var(&format!("b{i}")).unwrap()).collect();
        let mut acc = vec![BddRef::FALSE; 2 * n];
        for j in 0..n {
            let mut carry = BddRef::FALSE;
            for i in 0..n {
                let pp = m.and(a[i], b[j]).unwrap();
                let s0 = acc[i + j];
                let x = m.xor(s0, pp).unwrap();
                let sum = m.xor(x, carry).unwrap();
                let c1 = m.and(s0, pp).unwrap();
                let c2 = m.and(x, carry).unwrap();
                carry = m.or(c1, c2).unwrap();
                acc[i + j] = sum;
            }
            acc[n + j] = m.xor(acc[n + j], carry).unwrap();
        }
        let outs: Vec<BddRef> = acc;
        let names: Vec<String> = m.var_names().to_vec();
        let tables: Vec<Vec<bool>> = outs
            .iter()
            .map(|&f| table_by_name(&m, f, &names))
            .collect();
        let st = m.reorder(&outs, &ReorderOptions::default()).unwrap();
        assert!(st.nodes_after < st.nodes_before, "{st:?}");
        for (k, &f) in outs.iter().enumerate() {
            assert_eq!(table_by_name(&m, f, &names), tables[k]);
        }
    }

    #[test]
    fn node_ceiling_is_a_refusal() {
        let mut m = BddManager::with_max_nodes(8);
        let vs: Vec<_> = (0..6).map(|i| m.var(&format!("x{i}"))).collect();
        let err = vs.into_iter().collect::<Result<Vec<_>, _>>().map(|v| {
            let mut acc = BddRef::FALSE;
            for x in v {
                acc = m.xor(acc, x)?;
            }
            Ok::<_, BooleanError>(acc)
        });
        assert!(matches!(
            err,
            Err(BooleanError::TooLarge { .. }) | Ok(Err(BooleanError::TooLarge { .. }))
        ));
    }
}
