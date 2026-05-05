use crate::kernel::{
    domain::Domain,
    expr::{BigFloat, BigInt, BigRat, ExprData, ExprId},
};
use std::fmt;

/// Canonical ∞ symbol name for [`ExprPool::pos_infinity`] / limits (V2-16).
pub const POS_INFINITY_SYMBOL: &str = "\u{221e}";

// ---------------------------------------------------------------------------
// Phase 30 — Sharded ExprPool for parallel workloads.
//
// Strategy:
//   * The `nodes` array (ExprId → ExprData) is append-only after creation, so
//     reads only need a shared reference.  We protect it with an `RwLock` so
//     many parallel readers can proceed without blocking each other.
//   * The `index` (ExprData → ExprId) requires exclusive access during
//     insertion.  Under `--features parallel` we replace `HashMap` behind a
//     `Mutex` with `DashMap`, which shards the map internally so concurrent
//     inserts on *different* keys don't contend.
//   * The `nodes` Vec is still behind a `Mutex` because we need an atomic
//     read-len + push.  This is a single word compare on the fast path
//     (cache-line friendly).  A future upgrade can switch to a `boxcar::Vec`
//     (lock-free append-only Vec) if profiling shows this is the bottleneck.
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
use dashmap::DashMap;

#[cfg(not(feature = "parallel"))]
use std::collections::HashMap;

#[cfg(not(feature = "parallel"))]
use std::sync::Mutex;
use std::sync::RwLock;

// ---------------------------------------------------------------------------
// PoolState — two variants depending on build features
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
struct PoolIndex(DashMap<ExprData, ExprId>);

#[cfg(not(feature = "parallel"))]
struct PoolIndex(HashMap<ExprData, ExprId>);

#[cfg(feature = "parallel")]
impl PoolIndex {
    fn new() -> Self {
        PoolIndex(DashMap::new())
    }
    fn get(&self, data: &ExprData) -> Option<ExprId> {
        self.0.get(data).map(|v| *v)
    }
    fn insert(&self, data: ExprData, id: ExprId) {
        self.0.insert(data, id);
    }
}

#[cfg(not(feature = "parallel"))]
impl PoolIndex {
    fn new() -> Self {
        PoolIndex(HashMap::new())
    }
    fn get(&self, data: &ExprData) -> Option<ExprId> {
        self.0.get(data).copied()
    }
    fn insert(&mut self, data: ExprData, id: ExprId) {
        self.0.insert(data, id);
    }
}

/// Owns all expression nodes. Every [`ExprId`] is valid only within its pool.
///
/// `ExprPool` is `Send + Sync`.
///
/// Under `--features parallel` the index uses `DashMap` to reduce contention
/// on concurrent inserts; the nodes `Vec` uses a `Mutex` (cheap: only locked
/// during the push, which is rare once the pool is warm).
pub struct ExprPool {
    /// Append-only; indexed by `ExprId.0`.  `RwLock` allows parallel reads.
    nodes: RwLock<Vec<ExprData>>,
    /// Deduplication index: ExprData → ExprId.
    #[cfg(feature = "parallel")]
    index: PoolIndex,
    #[cfg(not(feature = "parallel"))]
    index: Mutex<PoolIndex>,
}

unsafe impl Send for ExprPool {}
unsafe impl Sync for ExprPool {}

impl ExprPool {
    pub fn new() -> Self {
        ExprPool {
            nodes: RwLock::new(Vec::new()),
            #[cfg(feature = "parallel")]
            index: PoolIndex::new(),
            #[cfg(not(feature = "parallel"))]
            index: Mutex::new(PoolIndex::new()),
        }
    }

    /// Intern `data`, returning a shared [`ExprId`]. Identical structures
    /// always return the same id; structural equality ⟺ id equality.
    pub fn intern(&self, data: ExprData) -> ExprId {
        // --- fast path: already interned ---
        #[cfg(feature = "parallel")]
        {
            if let Some(id) = self.index.get(&data) {
                return id;
            }
            // Slow path: insert new node atomically.
            // We use a double-checked pattern: lock nodes, re-check index
            // (another thread may have inserted between our check and lock).
            let mut nodes = self.nodes.write().expect("ExprPool nodes RwLock poisoned");
            if let Some(id) = self.index.get(&data) {
                return id;
            }
            let id = ExprId(nodes.len() as u32);
            self.index.insert(data.clone(), id);
            nodes.push(data);
            id
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut idx = self.index.lock().expect("ExprPool index Mutex poisoned");
            if let Some(id) = idx.get(&data) {
                return id;
            }
            let mut nodes = self.nodes.write().expect("ExprPool nodes RwLock poisoned");
            let id = ExprId(nodes.len() as u32);
            idx.insert(data.clone(), id);
            nodes.push(data);
            id
        }
    }

    /// Borrow a node by id and apply `f` without cloning.
    pub fn with<R, F: FnOnce(&ExprData) -> R>(&self, id: ExprId, f: F) -> R {
        let nodes = self.nodes.read().expect("ExprPool nodes RwLock poisoned");
        f(&nodes[id.0 as usize])
    }

    /// Clone and return the `ExprData` for `id`.
    pub fn get(&self, id: ExprId) -> ExprData {
        self.with(id, |d| d.clone())
    }

    /// Number of distinct expressions interned so far.
    pub fn len(&self) -> usize {
        self.nodes
            .read()
            .expect("ExprPool nodes RwLock poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // -----------------------------------------------------------------------
    // Atom constructors
    // -----------------------------------------------------------------------

    /// Free symbol; multiplication treats it as commuting with every other factor (default).
    pub fn symbol(&self, name: impl Into<String>, domain: Domain) -> ExprId {
        self.symbol_commutative(name, domain, true)
    }

    /// Free symbol with explicit commutative flag (V3-2). `commutative: false` is for
    /// matrix or operator generators where `A*B` and `B*A` must remain distinct.
    pub fn symbol_commutative(
        &self,
        name: impl Into<String>,
        domain: Domain,
        commutative: bool,
    ) -> ExprId {
        self.intern(ExprData::Symbol {
            name: name.into(),
            domain,
            commutative,
        })
    }

    pub fn integer(&self, n: impl Into<rug::Integer>) -> ExprId {
        self.intern(ExprData::Integer(BigInt(n.into())))
    }

    pub fn rational(
        &self,
        numer: impl Into<rug::Integer>,
        denom: impl Into<rug::Integer>,
    ) -> ExprId {
        let r = rug::Rational::from((numer.into(), denom.into()));
        self.intern(ExprData::Rational(BigRat(r)))
    }

    pub fn float(&self, value: f64, prec: u32) -> ExprId {
        let f = rug::Float::with_val(prec, value);
        self.intern(ExprData::Float(BigFloat { inner: f, prec }))
    }

    // -----------------------------------------------------------------------
    // Compound constructors
    // -----------------------------------------------------------------------

    pub fn add(&self, mut args: Vec<ExprId>) -> ExprId {
        // Sort children at construction time so that commutativity holds
        // structurally: `a + b` and `b + a` intern to the same ExprId.
        // The sort key is the raw ExprId (opaque u32), which gives a stable,
        // deterministic canonical order.
        args.sort_unstable();
        self.intern(ExprData::Add(args))
    }

    pub fn mul(&self, mut args: Vec<ExprId>) -> ExprId {
        // Canonical sort only when every subtree is multiplicatively commutative (V3-2).
        let sort_ok = args
            .iter()
            .all(|&a| crate::kernel::expr_props::mult_tree_is_commutative(self, a));
        if sort_ok {
            args.sort_unstable();
        }
        self.intern(ExprData::Mul(args))
    }

    pub fn pow(&self, base: ExprId, exp: ExprId) -> ExprId {
        self.intern(ExprData::Pow { base, exp })
    }

    pub fn func(&self, name: impl Into<String>, args: Vec<ExprId>) -> ExprId {
        self.intern(ExprData::Func {
            name: name.into(),
            args,
        })
    }

    // -----------------------------------------------------------------------
    // PA-9 — Piecewise / Predicate constructors
    // -----------------------------------------------------------------------

    /// Build a `Piecewise` expression.
    ///
    /// Branches are `(cond, value)` pairs where `cond` must be a
    /// `Predicate` node.  The `default` value is used when no condition
    /// matches.
    pub fn piecewise(&self, branches: Vec<(ExprId, ExprId)>, default: ExprId) -> ExprId {
        self.intern(ExprData::Piecewise { branches, default })
    }

    /// Build a `Predicate` node (symbolic boolean condition).
    pub fn predicate(&self, kind: crate::kernel::expr::PredicateKind, args: Vec<ExprId>) -> ExprId {
        self.intern(ExprData::Predicate { kind, args })
    }

    // Convenience constructors for common predicates.
    pub fn pred_lt(&self, a: ExprId, b: ExprId) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::Lt, vec![a, b])
    }
    pub fn pred_le(&self, a: ExprId, b: ExprId) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::Le, vec![a, b])
    }
    pub fn pred_gt(&self, a: ExprId, b: ExprId) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::Gt, vec![a, b])
    }
    pub fn pred_ge(&self, a: ExprId, b: ExprId) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::Ge, vec![a, b])
    }
    pub fn pred_eq(&self, a: ExprId, b: ExprId) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::Eq, vec![a, b])
    }
    pub fn pred_ne(&self, a: ExprId, b: ExprId) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::Ne, vec![a, b])
    }
    pub fn pred_and(&self, args: Vec<ExprId>) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::And, args)
    }
    pub fn pred_or(&self, args: Vec<ExprId>) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::Or, args)
    }
    pub fn pred_not(&self, a: ExprId) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::Not, vec![a])
    }
    pub fn pred_true(&self) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::True, vec![])
    }
    pub fn pred_false(&self) -> ExprId {
        self.predicate(crate::kernel::expr::PredicateKind::False, vec![])
    }

    // V3-3 — first-order quantifiers (first-class `Formula` / FOFormula).
    /// `∀ var . body`
    pub fn forall(&self, var: ExprId, body: ExprId) -> ExprId {
        self.intern(ExprData::Forall { var, body })
    }

    /// `∃ var . body`
    pub fn exists(&self, var: ExprId, body: ExprId) -> ExprId {
        self.intern(ExprData::Exists { var, body })
    }

    /// `O(arg)` — symbolic big-O bound used in truncated series (V2-15).
    pub fn big_o(&self, arg: ExprId) -> ExprId {
        self.intern(ExprData::BigO(arg))
    }

    /// Canonical `+∞` symbol for limits at infinity (V2-16).
    pub fn pos_infinity(&self) -> ExprId {
        self.symbol(POS_INFINITY_SYMBOL, Domain::Positive)
    }

    // -----------------------------------------------------------------------
    // Display helper
    // -----------------------------------------------------------------------

    pub fn display(&self, id: ExprId) -> ExprDisplay<'_> {
        ExprDisplay { id, pool: self }
    }
}

impl Default for ExprPool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Display — pool-aware recursive formatter
// ---------------------------------------------------------------------------

/// Wraps an `(ExprId, &ExprPool)` pair so it can implement [`fmt::Display`].
pub struct ExprDisplay<'a> {
    pub id: ExprId,
    pub pool: &'a ExprPool,
}

impl fmt::Display for ExprDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Clone to release the lock before recursing; recursive display would
        // deadlock if the lock were held across child format calls.
        let data = self.pool.get(self.id);
        fmt_data(&data, self.pool, f)
    }
}

impl fmt::Debug for ExprDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

fn fmt_data(data: &ExprData, pool: &ExprPool, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match data {
        ExprData::Symbol { name, .. } => write!(f, "{}", name),
        ExprData::Integer(n) => write!(f, "{}", n),
        ExprData::Rational(r) => write!(f, "{}", r),
        ExprData::Float(fl) => write!(f, "{}", fl),
        ExprData::Add(args) => {
            write!(f, "(")?;
            for (i, &arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, " + ")?;
                }
                write!(f, "{}", pool.display(arg))?;
            }
            write!(f, ")")
        }
        ExprData::Mul(args) => {
            write!(f, "(")?;
            for (i, &arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, " * ")?;
                }
                write!(f, "{}", pool.display(arg))?;
            }
            write!(f, ")")
        }
        ExprData::Pow { base, exp } => {
            write!(f, "{}^{}", pool.display(*base), pool.display(*exp))
        }
        ExprData::Func { name, args } => {
            write!(f, "{}(", name)?;
            for (i, &arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", pool.display(arg))?;
            }
            write!(f, ")")
        }
        ExprData::Piecewise { branches, default } => {
            write!(f, "Piecewise(")?;
            for (i, (cond, val)) in branches.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "({}, {})", pool.display(*cond), pool.display(*val))?;
            }
            write!(f, "; default={})", pool.display(*default))
        }
        ExprData::Predicate { kind, args } => match kind {
            crate::kernel::expr::PredicateKind::True => write!(f, "True"),
            crate::kernel::expr::PredicateKind::False => write!(f, "False"),
            crate::kernel::expr::PredicateKind::Not => {
                write!(f, "¬({})", pool.display(args[0]))
            }
            crate::kernel::expr::PredicateKind::And | crate::kernel::expr::PredicateKind::Or => {
                write!(f, "(")?;
                for (i, &arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, " {} ", kind)?;
                    }
                    write!(f, "{}", pool.display(arg))?;
                }
                write!(f, ")")
            }
            _ => {
                write!(
                    f,
                    "({} {} {})",
                    pool.display(args[0]),
                    kind,
                    pool.display(args[1])
                )
            }
        },
        ExprData::Forall { var, body } => {
            write!(f, "∀ {} . {}", pool.display(*var), pool.display(*body))
        }
        ExprData::Exists { var, body } => {
            write!(f, "∃ {} . {}", pool.display(*var), pool.display(*body))
        }
        ExprData::BigO(arg) => {
            write!(f, "O({})", pool.display(*arg))
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::domain::Domain;

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn noncommutative_mul_orders_distinct() {
        let p = pool();
        let a = p.symbol_commutative("A", Domain::Real, false);
        let b = p.symbol_commutative("B", Domain::Real, false);
        assert_ne!(
            p.mul(vec![a, b]),
            p.mul(vec![b, a]),
            "A*B and B*A must not hash-cons together for NC symbols"
        );
    }

    #[test]
    fn symbol_commutative_is_structural() {
        let p = pool();
        let xc = p.symbol_commutative("x", Domain::Real, true);
        let xnc = p.symbol_commutative("x", Domain::Real, false);
        assert_ne!(xc, xnc);
    }

    // --- construction and equality ---

    #[test]
    fn symbol_interning() {
        let p = pool();
        let x1 = p.symbol("x", Domain::Real);
        let x2 = p.symbol("x", Domain::Real);
        assert_eq!(x1, x2, "same symbol must return same ExprId");
    }

    #[test]
    fn domain_is_structural() {
        let p = pool();
        let xr = p.symbol("x", Domain::Real);
        let xc = p.symbol("x", Domain::Complex);
        assert_ne!(xr, xc, "same name but different domain must be distinct");
    }

    #[test]
    fn integer_interning() {
        let p = pool();
        let a = p.integer(42_i32);
        let b = p.integer(42_i32);
        let c = p.integer(99_i32);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn rational_canonical() {
        let p = pool();
        // 2/4 reduces to 1/2
        let r1 = p.rational(2_i32, 4_i32);
        let r2 = p.rational(1_i32, 2_i32);
        assert_eq!(r1, r2, "rationals must be reduced to canonical form");
    }

    #[test]
    fn float_precision_is_structural() {
        let p = pool();
        let f53 = p.float(1.0, 53);
        let f64_ = p.float(1.0, 64);
        assert_ne!(
            f53, f64_,
            "same value but different precision is a different expr"
        );
    }

    // --- compound expressions and subexpression sharing ---

    #[test]
    fn subexpression_sharing() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let two = p.integer(2_i32);

        // Build x^2 twice; both must return the same ExprId.
        let xsq1 = p.pow(x, two);
        let xsq2 = p.pow(x, two);
        assert_eq!(xsq1, xsq2);

        // Pool should have exactly 3 nodes: x, 2, x^2.
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn add_interning() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let s1 = p.add(vec![x, y]);
        let s2 = p.add(vec![x, y]);
        assert_eq!(s1, s2);
    }

    #[test]
    fn arg_order_is_canonical() {
        // PA-3: Add/Mul children are sorted at construction time so that
        // commutativity holds structurally — a+b and b+a intern to the same ExprId.
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let s1 = p.add(vec![x, y]);
        let s2 = p.add(vec![y, x]);
        assert_eq!(s1, s2, "a+b and b+a must be the same expression after PA-3");
        let m1 = p.mul(vec![x, y]);
        let m2 = p.mul(vec![y, x]);
        assert_eq!(m1, m2, "a*b and b*a must be the same expression after PA-3");
    }

    #[test]
    fn func_interning() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let s1 = p.func("sin", vec![x]);
        let s2 = p.func("sin", vec![x]);
        let c1 = p.func("cos", vec![x]);
        assert_eq!(s1, s2);
        assert_ne!(s1, c1);
    }

    // --- display ---

    #[test]
    fn display_symbol() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        assert_eq!(p.display(x).to_string(), "x");
    }

    #[test]
    fn display_integer() {
        let p = pool();
        let n = p.integer(42_i32);
        assert_eq!(p.display(n).to_string(), "42");
    }

    #[test]
    fn display_pow() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let two = p.integer(2_i32);
        let xsq = p.pow(x, two);
        assert_eq!(p.display(xsq).to_string(), "x^2");
    }

    #[test]
    fn display_add() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let s = p.add(vec![x, y]);
        assert_eq!(p.display(s).to_string(), "(x + y)");
    }

    #[test]
    fn display_func() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let s = p.func("sin", vec![x]);
        assert_eq!(p.display(s).to_string(), "sin(x)");
    }

    #[test]
    fn display_nested() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let two = p.integer(2_i32);
        let xsq = p.pow(x, two);
        let one = p.integer(1_i32);
        let expr = p.add(vec![xsq, one]);
        assert_eq!(p.display(expr).to_string(), "(x^2 + 1)");
    }

    // --- send + sync: compile-time check ---

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn pool_is_send_sync() {
        assert_send_sync::<ExprPool>();
    }
}
