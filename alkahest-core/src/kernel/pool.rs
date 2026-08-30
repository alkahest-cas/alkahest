use crate::kernel::{
    domain::Domain,
    expr::{BigFloat, BigInt, BigRat, ExprData, ExprId},
};
use std::fmt;

/// Canonical ∞ symbol name for [`ExprPool::pos_infinity`] / limits (V2-16).
pub const POS_INFINITY_SYMBOL: &str = "\u{221e}";

// ---------------------------------------------------------------------------
// Lock-free arena for ExprPool nodes.
//
// Strategy:
//   * The `nodes` array (ExprId → ExprData) is a `boxcar::Vec` — a
//     lock-free, append-only, reference-stable segmented array.  Reads
//     (`with`, `get`, `len`) acquire no lock at all; they index directly
//     into the array via a single atomic load.
//   * The `index` (ExprData → ExprId) still requires coordination during
//     insertion to preserve hash-cons uniqueness:
//     - Under `--features parallel` we use `DashMap::entry` which holds a
//       per-shard write-lock only for the duration of the insert.  The
//       closure passed to `or_insert_with` calls `boxcar::push` (lock-free)
//       while the shard lock is held, so no two threads can insert the same
//       key.
//     - Without `parallel` the `Mutex<HashMap>` serialises all inserts as
//       before; the boxcar push happens while the Mutex is held.
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
use dashmap::DashMap;

#[cfg(not(feature = "parallel"))]
use std::collections::HashMap;

#[cfg(not(feature = "parallel"))]
use std::sync::Mutex;

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
    /// Atomically return the existing id for `key`, or call `f` to produce one
    /// and insert it.  The DashMap shard write-lock is held for the duration of
    /// `f`, guaranteeing at most one call to `f` per unique key.
    fn or_insert_with(&self, key: ExprData, f: impl FnOnce() -> ExprId) -> ExprId {
        *self.0.entry(key).or_insert_with(f)
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
/// Read operations (`with`, `get`, `len`) are fully lock-free — they index
/// into a `boxcar::Vec` via a single atomic load with no lock acquisition.
/// Write operations (`intern`) use a per-shard lock (parallel mode) or a
/// `Mutex` (non-parallel mode) only during new-node insertion.
/// A node plus the properties that are cheaper to record once than to recompute.
struct Node {
    data: ExprData,
    /// Whether every generator in this subtree commutes under multiplication.
    ///
    /// This is a bottom-up property, and hash-consing guarantees a node's
    /// children are interned before the node itself, so it is computed once
    /// here from the children's cached flags — O(arity) — instead of by
    /// walking the whole subtree on every query.
    mult_commutative: bool,
    /// Length of the longest root-to-leaf path in this subtree; a leaf is 1.
    ///
    /// Computed exactly like `mult_commutative` — once, at intern time, from
    /// the children's cached values — so [`ExprPool::depth`] is a single array
    /// read.  Recomputing it on demand is not an option: the pool is a DAG, so
    /// an unmemoised depth walk is exponential in the sharing, and a memoised
    /// one allocates a map per query.  Saturating, so a pathological expression
    /// pins at `u32::MAX` instead of wrapping to a small value.
    ///
    /// This is what lets every recursive consumer refuse a too-deep expression
    /// in O(1) rather than discovering the problem by overflowing the stack.
    depth: u32,
}

pub struct ExprPool {
    /// Lock-free, append-only, reference-stable node array.
    nodes: boxcar::Vec<Node>,
    /// Deduplication index: ExprData → ExprId.
    #[cfg(feature = "parallel")]
    index: PoolIndex,
    #[cfg(not(feature = "parallel"))]
    index: Mutex<PoolIndex>,
}

// `ExprPool` is `Send + Sync` *by inference*, not by assertion.
//
// This used to be `unsafe impl Send for ExprPool {}` / `unsafe impl Sync`, and
// nothing about the type ever needed it: every field is already `Send + Sync`
// (`boxcar::Vec<Node>` where `Node: Send + Sync`, `DashMap` under `parallel`,
// `Mutex<PoolIndex>` without it).  An unconditional `unsafe impl` on a type that
// derives the traits anyway is strictly worse than nothing, because it also
// *silences the check for the future*: add an `Rc`, a `Cell`, or a raw pointer
// to `ExprPool`, `Node` or `ExprData` and the compiler would have gone on
// certifying the pool as shareable across rayon workers and across
// `Python::allow_threads` — the exact boundary the pool is handed over most
// often.  The static assertion below re-arms that check: it costs nothing at
// run time and fails the build the moment a non-thread-safe field appears.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ExprPool>();
    assert_send_sync::<Node>();
    assert_send_sync::<ExprData>();
};

/// Largest number of children a flat `Add`/`Mul` may be spliced up to.
///
/// Bounds the self-combination blow-up that flat n-ary form introduces — see
/// the comment in `flatten_assoc`. Set from measurement: the largest honest
/// arity across this crate's test suite is 50 001, and the runaway shows up as
/// powers of two from 32 768 upward, so no threshold separates them by size
/// alone. This sits 2.6x above the honest maximum and still bounds the runaway,
/// which converges because a declined splice leaves the node nested.
pub(crate) const MAX_FLAT_ARITY: usize = 131_072;

impl ExprPool {
    pub fn new() -> Self {
        ExprPool {
            nodes: boxcar::Vec::new(),
            #[cfg(feature = "parallel")]
            index: PoolIndex::new(),
            #[cfg(not(feature = "parallel"))]
            index: Mutex::new(PoolIndex::new()),
        }
    }

    /// Intern `data`, returning a shared [`ExprId`]. Identical structures
    /// always return the same id; structural equality ⟺ id equality.
    pub fn intern(&self, data: ExprData) -> ExprId {
        #[cfg(feature = "parallel")]
        {
            // Fast path: lock-free DashMap read.
            if let Some(id) = self.index.get(&data) {
                return id;
            }
            // Slow path: DashMap shard write-lock ensures at most one push
            // per unique key.  `boxcar::push` is lock-free so it can be
            // called safely while the shard lock is held.
            self.index.or_insert_with(data.clone(), || {
                let node = self.make_node(data);
                ExprId(self.nodes.push(node) as u32)
            })
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut idx = self.index.lock().expect("ExprPool index Mutex poisoned");
            if let Some(id) = idx.get(&data) {
                return id;
            }
            let node = self.make_node(data.clone());
            let id = ExprId(self.nodes.push(node) as u32);
            idx.insert(data, id);
            id
        }
    }

    /// Wrap `data` with its cached properties.  Children are already interned,
    /// so their flags are just array reads.
    fn make_node(&self, data: ExprData) -> Node {
        let mult_commutative = self.compute_mult_commutative(&data);
        let depth = self.compute_depth(&data);
        Node {
            data,
            mult_commutative,
            depth,
        }
    }

    /// One level of the depth recurrence: `1 + max(child depths)`, reading each
    /// child's cached depth rather than descending into it.
    fn compute_depth(&self, data: &ExprData) -> u32 {
        let child = |c: ExprId| self.depth(c);
        let deepest = match data {
            ExprData::Symbol { .. }
            | ExprData::Integer(_)
            | ExprData::Rational(_)
            | ExprData::Float(_) => 0,
            ExprData::Add(args) | ExprData::Mul(args) => {
                args.iter().copied().map(child).max().unwrap_or(0)
            }
            ExprData::Pow { base, exp } => child(*base).max(child(*exp)),
            ExprData::Func { args, .. } => args.iter().copied().map(child).max().unwrap_or(0),
            ExprData::Piecewise { branches, default } => branches
                .iter()
                .map(|&(c, v)| child(c).max(child(v)))
                .max()
                .unwrap_or(0)
                .max(child(*default)),
            ExprData::Predicate { args, .. } => args.iter().copied().map(child).max().unwrap_or(0),
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                child(*var).max(child(*body))
            }
            ExprData::BigO(inner) => child(*inner),
            ExprData::RootSum { poly, body, .. } => child(*poly).max(child(*body)),
        };
        deepest.saturating_add(1)
    }

    /// One level of the `mult_tree_is_commutative` recurrence, reading each
    /// child's cached flag rather than descending into it.
    fn compute_mult_commutative(&self, data: &ExprData) -> bool {
        let child = |c: ExprId| self.is_mult_commutative(c);
        match data {
            ExprData::Symbol { commutative, .. } => *commutative,
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => true,
            ExprData::Add(args) | ExprData::Mul(args) => args.iter().copied().all(child),
            ExprData::Pow { base, exp } => child(*base) && child(*exp),
            ExprData::Func { args, .. } => args.iter().copied().all(child),
            ExprData::Piecewise { branches, default } => {
                branches.iter().all(|&(c, v)| child(c) && child(v)) && child(*default)
            }
            ExprData::Predicate { args, .. } => args.iter().copied().all(child),
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                child(*var) && child(*body)
            }
            ExprData::BigO(inner) => child(*inner),
            ExprData::RootSum { poly, body, .. } => child(*poly) && child(*body),
        }
    }

    /// Whether every generator in the subtree rooted at `id` commutes under
    /// multiplication.  O(1): the flag was computed when `id` was interned.
    pub fn is_mult_commutative(&self, id: ExprId) -> bool {
        self.node(id).mult_commutative
    }

    /// Length of the longest root-to-leaf path in the subtree rooted at `id`.
    ///
    /// A leaf (symbol or number) has depth 1.  O(1): the value was computed
    /// when `id` was interned.  Saturates at [`u32::MAX`].
    ///
    /// This is the quantity the expression-depth ceiling is applied to: the
    /// PyO3 entry points compare it against `MAX_EXPR_DEPTH` to decline a tree
    /// too deep to recurse over — see
    /// [`crate::kernel::depth::check_expr_depth`].
    pub fn depth(&self, id: ExprId) -> u32 {
        self.node(id).depth
    }

    fn node(&self, id: ExprId) -> &Node {
        self.nodes
            .get(id.0 as usize)
            .expect("ExprPool: ExprId out of range")
    }

    /// Borrow a node by id and apply `f` without cloning.  Lock-free.
    pub fn with<R, F: FnOnce(&ExprData) -> R>(&self, id: ExprId, f: F) -> R {
        f(&self.node(id).data)
    }

    /// Clone and return the `ExprData` for `id`.
    pub fn get(&self, id: ExprId) -> ExprData {
        self.with(id, |d| d.clone())
    }

    /// Number of distinct expressions interned so far.  Lock-free.
    pub fn len(&self) -> usize {
        self.nodes.count()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // -----------------------------------------------------------------------
    // Atom constructors
    // -----------------------------------------------------------------------

    /// Free symbol; multiplication treats it as commuting with every other factor (default).
    pub fn symbol(&self, name: impl Into<String>, domain: Domain) -> ExprId {
        self.symbol_commutative(name, domain, true)
    }

    /// Canonical name of the kernel-blessed imaginary unit `i = √(−1)`.
    ///
    /// Reserved: do not create an unrelated free symbol with this name and
    /// `Domain::Complex` — the simplifier applies the algebraic power rules
    /// `i² = −1`, `i³ = −i`, `i⁴ = 1`, … to any symbol matching this name and
    /// domain (see [`ExprPool::is_imaginary_unit`]).
    pub const IMAGINARY_UNIT_NAME: &'static str = "I";

    /// The first-class imaginary unit `i = √(−1)`.
    ///
    /// Represented as the interned, kernel-blessed commuting symbol
    /// [`IMAGINARY_UNIT_NAME`](Self::IMAGINARY_UNIT_NAME) with
    /// [`Domain::Complex`]. This is the *canonical* representation: the
    /// simplifier knows the algebraic identities `i² = −1`, `i³ = −i`,
    /// `i⁴ = 1`, and more generally `i^(4k+r) → i^r` for literal integer
    /// exponents (no branch-cut identities — `√(−1) → i`, `log`/`exp` of
    /// complex arguments etc. are *not* added).
    ///
    /// Differentiation treats it as a constant (`d/dx i = 0`, like `π`/`e`)
    /// and numeric evaluation declines (it has no `f64` value), matching the
    /// behaviour of other non-real atoms.
    pub fn imaginary_unit(&self) -> ExprId {
        self.symbol(Self::IMAGINARY_UNIT_NAME, Domain::Complex)
    }

    /// Returns `true` iff `id` is the canonical imaginary unit produced by
    /// [`ExprPool::imaginary_unit`] (an interned `Domain::Complex` symbol named
    /// [`IMAGINARY_UNIT_NAME`](Self::IMAGINARY_UNIT_NAME)).
    pub fn is_imaginary_unit(&self, id: ExprId) -> bool {
        self.with(id, |d| {
            matches!(
                d,
                ExprData::Symbol { name, domain, .. }
                    if name == Self::IMAGINARY_UNIT_NAME && *domain == Domain::Complex
            )
        })
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

    /// Splice same-operator children into `args`, in argument order.
    ///
    /// `mul` passes `want_mul = true` and splices nested `Mul`s; `add` passes
    /// `false` and splices nested `Add`s.  This is **associativity and nothing
    /// else** — no reordering beyond the canonical sort the caller applies
    /// afterwards, no constant folding, no identity elimination.  Splicing an
    /// empty `Mul`/`Add` child away is likewise value-preserving, since the
    /// empty product is 1 and the empty sum is 0.
    ///
    /// Every node reachable through [`ExprPool::add`] / [`ExprPool::mul`] is
    /// already flat, so in practice one level is all there is; the loop is
    /// nevertheless a full fixpoint because `intern` is public and
    /// [`crate::kernel::pool_persist`] restores whatever shape a file on disk
    /// holds, including nested nodes written by an older build.  The worklist
    /// is an explicit `Vec`, not recursion, so no nesting depth can overflow
    /// the native stack here.
    fn flatten_assoc(&self, args: Vec<ExprId>, want_mul: bool) -> Vec<ExprId> {
        /// The children to splice in for `data`, or `None` to keep it whole.
        fn splices(data: &ExprData, want_mul: bool) -> Option<&Vec<ExprId>> {
            match data {
                ExprData::Mul(children) if want_mul => Some(children),
                ExprData::Add(children) if !want_mul => Some(children),
                _ => None,
            }
        }

        // Hot path: nothing nested, so hand the caller its own vector back
        // without allocating a second one.
        if !args
            .iter()
            .any(|&a| splices(&self.node(a).data, want_mul).is_some())
        {
            return args;
        }

        // Arity ceiling.  Flat n-ary form removes the sharing that binary
        // nesting gave for free: `e = pool.mul([e, e])` in a loop used to build
        // `n` nodes with both children shared, and now *doubles* the child
        // count, so `n` rounds cost `2^n` children.  Twenty rounds is ~2M
        // children, twenty-five exhausts memory, and a real test in this repo
        // hung the suite at forty.
        //
        // Declining to splice — rather than refusing — keeps `mul`/`add` total
        // and infallible, which matters because they are two of the most-called
        // constructors in the crate and have no `Result` today.  Above the cap
        // the caller simply gets a nested node back, which is what it would
        // have got before flattening existed; `simplify` still flattens, so the
        // canonical form is unchanged for anything that goes through it.  The
        // doubling loop then converges: once a splice is declined the node stays
        // nested, so the next round starts from two children again.
        //
        // MAX_FLAT_ARITY is set from measurement, not taste.  Instrumenting
        // `cargo test -p alkahest-cas --lib` put the largest *honest* arity at
        // 50 001 (a test building one long sum a term at a time); the blow-up
        // showed up as the powers of two 32 768 … 2 097 152.  The two ranges
        // overlap, so no threshold separates them by size alone — this one sits
        // 2.6x above the honest maximum and still bounds the runaway.
        // The splice is a *fixpoint* — a spliced-in child may itself be an
        // `Add`/`Mul` and get expanded in turn — so the ceiling has to bound the
        // final width, not the first level. Checking one level ahead is not
        // enough and is actively misleading: after a decline the node is a
        // 2-child nest, so a one-level count reads as 4 while the worklist goes
        // on to expand the grandchildren to millions.
        //
        // Counting as we expand is exact and costs nothing in the common case,
        // where the loop finishes long before the ceiling is in sight.
        let mut out = Vec::with_capacity(args.len() + 4);
        let mut stack: Vec<ExprId> = args.iter().rev().copied().collect();
        while let Some(id) = stack.pop() {
            match splices(&self.node(id).data, want_mul) {
                Some(children) => {
                    if out.len() + stack.len() + children.len() > MAX_FLAT_ARITY {
                        // Abandon the splice and hand back the caller's own
                        // vector untouched. `mul`/`add` stay total: the caller
                        // gets a nested node, exactly what it would have got
                        // before flattening existed, and `simplify` still
                        // flattens later for anything that goes through it.
                        return args;
                    }
                    stack.extend(children.iter().rev().copied());
                }
                None => out.push(id),
            }
        }
        out
    }

    pub fn add(&self, args: Vec<ExprId>) -> ExprId {
        // Associativity holds structurally: `(a + b) + c` and `a + (b + c)`
        // both intern as the flat `Add([a, b, c])`.
        let mut args = self.flatten_assoc(args, false);
        // Sort children at construction time so that commutativity holds
        // structurally: `a + b` and `b + a` intern to the same ExprId.
        // The sort key is the raw ExprId (opaque u32), which gives a stable,
        // deterministic canonical order.
        args.sort_unstable();
        self.intern(ExprData::Add(args))
    }

    pub fn mul(&self, args: Vec<ExprId>) -> ExprId {
        // Associativity holds structurally, exactly as for `add`.  Splicing
        // preserves argument order, so it is sound for the non-commutative
        // generators of V3-2 as well as the commutative case.
        let mut args = self.flatten_assoc(args, true);
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

    /// `Σ_{c : poly(c)=0} body[var := c]` — a sum over the roots of `poly`.
    pub fn root_sum(&self, poly: ExprId, var: ExprId, body: ExprId) -> ExprId {
        self.intern(ExprData::RootSum { poly, var, body })
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
        let data = self.pool.get(self.id);
        fmt_data(&data, self.pool, f)
    }
}

impl fmt::Debug for ExprDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

/// Format a power base or exponent, parenthesizing compound subexpressions.
///
/// `Add`/`Mul` already render with outer parentheses, so wrapping again would
/// produce `((z + -2))^-1`. Only wrap forms that do not already self-group.
fn fmt_pow_atom(id: ExprId, pool: &ExprPool) -> String {
    let s = pool.display(id).to_string();
    let needs_parens = match pool.get(id) {
        ExprData::Symbol { .. } | ExprData::Integer(_) | ExprData::Float(_) => false,
        ExprData::Func { .. } => false,
        // Already printed as `(…)` by fmt_data.
        ExprData::Add(_) | ExprData::Mul(_) => false,
        ExprData::Rational(_)
        | ExprData::Pow { .. }
        | ExprData::Piecewise { .. }
        | ExprData::Predicate { .. }
        | ExprData::Forall { .. }
        | ExprData::Exists { .. }
        | ExprData::BigO(_)
        | ExprData::RootSum { .. } => true,
    };
    if needs_parens {
        format!("({s})")
    } else {
        s
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
            // Parenthesize compound bases/exponents so `x^(1/2)^3` cannot be
            // misread as `x^1 / 2^3`. Prefer `(x^(1/2))^3`.
            let base_s = fmt_pow_atom(*base, pool);
            let exp_s = fmt_pow_atom(*exp, pool);
            write!(f, "{base_s}^{exp_s}")
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
        ExprData::RootSum { poly, var, body } => {
            write!(
                f,
                "RootSum({}, {} . {})",
                pool.display(*poly),
                pool.display(*var),
                pool.display(*body)
            )
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

    // --- associativity: Add/Mul are flat at construction ---

    /// Read a node's `Mul`/`Add` children, or panic if it is neither.
    fn nary_args(p: &ExprPool, id: ExprId) -> Vec<ExprId> {
        p.with(id, |d| match d {
            ExprData::Add(a) | ExprData::Mul(a) => a.clone(),
            other => panic!("expected an Add or Mul, got {other:?}"),
        })
    }

    #[test]
    fn mul_splices_nested_children() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let z = p.symbol("z", Domain::Real);

        let flat = p.mul(vec![x, y, z]);
        let left = p.mul(vec![p.mul(vec![x, y]), z]); // (x·y)·z
        let right = p.mul(vec![x, p.mul(vec![y, z])]); // x·(y·z)

        assert_eq!(left, flat, "(x*y)*z must intern as the flat x*y*z");
        assert_eq!(right, flat, "x*(y*z) must intern as the flat x*y*z");
        assert_eq!(nary_args(&p, flat).len(), 3);
    }

    #[test]
    fn add_splices_nested_children() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let z = p.symbol("z", Domain::Real);

        let flat = p.add(vec![x, y, z]);
        assert_eq!(p.add(vec![p.add(vec![x, y]), z]), flat);
        assert_eq!(p.add(vec![x, p.add(vec![y, z])]), flat);
        assert_eq!(nary_args(&p, flat).len(), 3);
    }

    /// Splicing is per-operator: an `Add` inside a `Mul` (and vice versa) is a
    /// different operator and must be left alone, or the value would change.
    #[test]
    fn splicing_does_not_cross_operators() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let z = p.symbol("z", Domain::Real);

        let sum = p.add(vec![x, y]);
        let prod = p.mul(vec![sum, z]); // (x + y)·z  — must stay a 2-factor Mul
        assert_eq!(nary_args(&p, prod).len(), 2);
        assert_ne!(prod, p.mul(vec![x, y, z]));

        let prod2 = p.mul(vec![x, y]);
        let sum2 = p.add(vec![prod2, z]); // x·y + z
        assert_eq!(nary_args(&p, sum2).len(), 2);
        assert_ne!(sum2, p.add(vec![x, y, z]));
    }

    /// Splicing preserves argument order, so it is sound for the V3-2
    /// non-commutative generators, which are never sorted.
    #[test]
    fn splicing_preserves_order_for_noncommutative_generators() {
        let p = pool();
        let a = p.symbol_commutative("A", Domain::Real, false);
        let b = p.symbol_commutative("B", Domain::Real, false);
        let c = p.symbol_commutative("C", Domain::Real, false);

        let flat = p.mul(vec![a, b, c]);
        assert_eq!(p.mul(vec![p.mul(vec![a, b]), c]), flat);
        assert_eq!(p.mul(vec![a, p.mul(vec![b, c])]), flat);
        assert_eq!(p.display(flat).to_string(), "(A * B * C)");

        // Associativity only: A·B·C and B·A·C are still distinct expressions.
        assert_ne!(flat, p.mul(vec![b, a, c]));
    }

    /// Flattening is a fixpoint, and it runs on an explicit worklist rather
    /// than the native stack.  `intern` is public and `pool_persist` restores
    /// whatever a file holds, so a genuinely nested chain can still reach the
    /// constructors; a 50 000-deep one must splice without overflowing.
    #[test]
    fn splicing_a_deeply_nested_chain_does_not_recurse() {
        const N: i64 = 50_000;
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let mut acc = x;
        for i in 0..N {
            let k = p.integer(i);
            // Deliberately bypass `add` so the chain really is nested.
            acc = p.intern(ExprData::Add(vec![acc, k]));
        }
        assert_eq!(p.depth(acc), N as u32 + 1);

        let flat = p.add(vec![acc]);
        assert_eq!(p.depth(flat), 2);
        assert_eq!(nary_args(&p, flat).len(), N as usize + 1);
    }

    /// The empty product is 1 and the empty sum is 0, so splicing an empty
    /// child away is value-preserving too.
    #[test]
    fn splicing_drops_empty_same_operator_children() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let empty_mul = p.mul(vec![]);
        assert_eq!(p.mul(vec![x, empty_mul]), p.mul(vec![x]));
        let empty_add = p.add(vec![]);
        assert_eq!(p.add(vec![x, empty_add]), p.add(vec![x]));
    }

    /// Flattening only ever *increases* sharing: the three spellings of a
    /// three-factor product are now one node, not three.
    #[test]
    fn flattening_improves_hash_consing() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let z = p.symbol("z", Domain::Real);
        let before = p.len();
        p.mul(vec![p.mul(vec![x, y]), z]);
        p.mul(vec![x, p.mul(vec![y, z])]);
        p.mul(vec![x, y, z]);
        // Two intermediate pairs (x*y, y*z) plus the single shared flat node.
        assert_eq!(p.len() - before, 3);
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

#[cfg(test)]
mod flat_arity_cap_tests {
    use super::*;
    use crate::kernel::Domain;

    /// The blow-up this cap exists for: flat n-ary form removes the sharing
    /// binary nesting gave for free, so `e = e * e` *doubles* the child count.
    /// Forty rounds is 2^40 children — a real test in this repo hung the whole
    /// lib suite on exactly this shape. It must terminate, quickly.
    #[test]
    fn self_combination_terminates_instead_of_doubling_forever() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // 22 rounds, not 40: doubling from 2 children crosses the cap at round
        // 17, so this exercises the decline *and* the reset that follows it.
        // Rounds beyond that only re-run the same two phases, and each round
        // near the cap sorts and interns ~131k children — in a debug build that
        // is minutes of nothing new.
        let mut e = pool.mul(vec![x, pool.integer(2_i32)]);
        for _ in 0..22 {
            e = pool.mul(vec![e, e]);
        }
        // Reaching here at all is the assertion. Past the cap the splice is
        // declined and the node stays nested, so the loop converges rather
        // than growing: the result is small, not astronomically wide.
        let width = match &pool.get(e) {
            ExprData::Mul(children) => children.len(),
            _ => 1,
        };
        assert!(
            width <= MAX_FLAT_ARITY,
            "a declined splice must leave a bounded node, got {width} children"
        );
    }

    /// The cap must not fire on honest work. The measured maximum across this
    /// crate's suite is 50 001 children; a sum well past that still flattens.
    #[test]
    fn a_large_honest_sum_still_flattens() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let _ = x;
        let terms: Vec<_> = (0..60_000).map(|i| pool.integer(i)).collect();
        let sum = pool.add(terms);
        let ExprData::Add(children) = &pool.get(sum) else {
            panic!("expected a flat Add");
        };
        assert!(
            children.len() >= 59_000,
            "a 60k-term sum must stay flat, got {} children",
            children.len()
        );
    }

    /// Ordinary nesting is unaffected — the cap is not a behaviour change for
    /// anything of a realistic size.
    #[test]
    fn ordinary_nesting_still_splices() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let inner = pool.mul(vec![x, y]);
        assert_eq!(pool.mul(vec![inner, z]), pool.mul(vec![x, y, z]));
    }
}
