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
use std::collections::HashMap;

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
        }
    }

    /// Node ceiling.
    pub fn max_nodes(&self) -> usize {
        self.max_nodes
    }

    /// Total nodes allocated in the manager (including the two terminals and
    /// nodes no longer reachable from any live function).
    pub fn allocated_nodes(&self) -> usize {
        self.nodes.len()
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
        if self.nodes.len() >= self.max_nodes {
            return Err(BooleanError::TooLarge {
                what: "BDD nodes",
                limit: self.max_nodes as u64,
            });
        }
        if self.nodes.len() % 1024 == 0 {
            crate::budget::check().map_err(BooleanError::Budget)?;
        }
        let r = BddRef(self.nodes.len() as u32);
        self.nodes.push(Node { level, lo, hi });
        self.unique.insert((level, lo, hi), r);
        Ok(r)
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

    /// Drop the operation cache (the node table is kept).
    pub fn clear_cache(&mut self) {
        self.ite_cache.clear();
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
