use crate::kernel::{
    domain::Domain,
    expr::{BigFloat, BigInt, BigRat, ExprData, ExprId},
};
use std::fmt;

/// Canonical ∞ symbol name for [`ExprPool::pos_infinity`] / limits (V2-16).
pub const POS_INFINITY_SYMBOL: &str = "\u{221e}";

// ---------------------------------------------------------------------------
// Lock-free arena for ExprPool nodes, and a single-storage index over it.
//
// Strategy:
//   * The `nodes` array (ExprId → Node) is a `boxcar::Vec` — a lock-free,
//     append-only, reference-stable segmented array.  Reads (`with`, `get`,
//     `len`) acquire no lock at all; they index directly into the array via a
//     single atomic load.  This is the *only* place an `ExprData` is stored.
//   * The `index` (ExprData → ExprId) is a set of 8-byte `Slot`s — an id and
//     32 bits of the node's hash — in `hashbrown::HashTable`s.  It stores no
//     `ExprData` of its own: a probe compares the candidate against the node
//     the slot points at.  (The index used to be a map *keyed by* `ExprData`,
//     so every node lived twice — once in the array and once, deep-cloned, as
//     a key — and every miss paid for the clone.)
//   * Hash-cons uniqueness under concurrency: the index is split into shards,
//     each behind an `RwLock`, chosen by the top bits of the hash.  A hit takes
//     the shard's read lock.  A miss takes its write lock, *re-probes under it*,
//     and only then pushes the node (lock-free) and records its slot, still
//     under the lock — so two threads interning the same value serialise on the
//     same shard, the second finds the first's slot, and a node is created at
//     most once per distinct value.
//   * Without `parallel` there is one shard; the code path is the same.
//   * Each `ExprData` is hashed exactly once per `intern`, with a per-pool
//     random seed (foldhash).  The table grows by re-bucketing the stored
//     32-bit hashes, so a resize never touches — let alone re-hashes — a node.
// ---------------------------------------------------------------------------

use hashbrown::HashTable;
use std::hash::BuildHasher;
use std::sync::{PoisonError, RwLock};

/// One index entry: the id of an interned node and the low 32 bits of its hash.
///
/// The hash bits are a fast reject (most probes that land on the wrong slot
/// never touch the node) and are what the table re-buckets on growth.  They
/// never decide equality: a hash collision falls through to comparing the
/// `ExprData` itself, so two distinct nodes can never be merged.
#[derive(Clone, Copy)]
struct Slot {
    id: u32,
    hash: u32,
}

/// The 64-bit table hash for a slot's 32 stored bits.  Duplicating them into
/// the high half gives hashbrown's control byte (the top 7 bits) and its bucket
/// index (the low bits) independent-looking inputs.
#[inline]
fn slot_hash(h32: u32) -> u64 {
    let h = u64::from(h32);
    h | (h << 32)
}

/// A shard, padded to its own cache lines so concurrent interns on neighbouring
/// shards do not contend on the lock word.
#[repr(align(128))]
struct Shard(RwLock<HashTable<Slot>>);

/// Number of index shards: one without `parallel`; with it, four per available
/// core rounded up to a power of two (the same rule `DashMap` uses), capped so
/// an idle pool stays small.  Computed once per process — reading the core
/// count can mean parsing cgroup files.
fn shard_count() -> usize {
    #[cfg(feature = "parallel")]
    {
        static SHARDS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *SHARDS.get_or_init(|| {
            let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
            (cores * 4).next_power_of_two().clamp(4, 64)
        })
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

struct PoolIndex {
    shards: Box<[Shard]>,
    /// `64 - log2(shards.len())`; the shard is `hash >> shard_shift` (the top
    /// bits), disjoint from the low 32 bits a `Slot` stores.
    shard_shift: u32,
    seed: foldhash::fast::RandomState,
}

impl PoolIndex {
    fn new() -> Self {
        let n = shard_count();
        debug_assert!(n.is_power_of_two());
        PoolIndex {
            shards: (0..n)
                .map(|_| Shard(RwLock::new(HashTable::new())))
                .collect(),
            shard_shift: 64 - n.trailing_zeros(),
            seed: foldhash::fast::RandomState::default(),
        }
    }

    /// Return the id for `data`, calling `make` to create the node if absent.
    fn get_or_insert_with(
        &self,
        nodes: &boxcar::Vec<Node>,
        data: ExprData,
        make: impl FnOnce(ExprData) -> ExprId,
    ) -> ExprId {
        let hash = self.seed.hash_one(&data);
        self.get_or_insert_hashed(nodes, hash, data, make)
    }

    /// [`Self::get_or_insert_with`] with the hash supplied, so tests can force
    /// collisions.
    ///
    /// Hit: one shard read-lock.  Miss: the shard write-lock, a re-probe under
    /// it, and `make` runs while the lock is held — so `make` runs at most once
    /// per distinct value, however many threads race to intern it.
    fn get_or_insert_hashed(
        &self,
        nodes: &boxcar::Vec<Node>,
        hash: u64,
        data: ExprData,
        make: impl FnOnce(ExprData) -> ExprId,
    ) -> ExprId {
        let h32 = hash as u32;
        let th = slot_hash(h32);
        // `checked_shr`: with a single shard the shift is 64.
        let shard = hash.checked_shr(self.shard_shift).unwrap_or(0) as usize;
        let lock = &self.shards[shard].0;
        let eq = |s: &Slot| s.hash == h32 && nodes[s.id as usize].data == data;

        // A panic in `make` (say, a child id from another pool) poisons the
        // lock, but it fires before the slot is recorded, so the table is still
        // consistent and later interns may carry on.
        if let Some(s) = lock
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .find(th, eq)
        {
            return ExprId(s.id);
        }
        let mut table = lock.write().unwrap_or_else(PoisonError::into_inner);
        if let Some(s) = table.find(th, eq) {
            return ExprId(s.id);
        }
        let id = make(data);
        table.insert_unique(
            th,
            Slot {
                id: id.0,
                hash: h32,
            },
            |s| slot_hash(s.hash),
        );
        id
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
    /// Whether some symbol in this subtree has [`Domain::Positive`] or
    /// [`Domain::NonZero`] — the domains
    /// `simplify::assumptions::collect_static_domain_facts` turns into
    /// rewrite facts.  Every `simplify` ends with that collection, and almost
    /// no expression has such a symbol, so the walk can answer from the root
    /// in O(1) and prune every subtree without one.  Computed like the flags
    /// above; the children counted are every child that walk descends into.
    static_domain_fact: bool,
    /// Whether some node in this subtree is `∞` or a non-finite `Float`
    /// (see [`crate::kernel::expr_props::is_non_finite_atom`]).  Computed like
    /// the flags above, so the simplifier can ask "may this term be infinite
    /// or NaN?" in O(1) before cancelling it.
    non_finite: bool,
}

pub struct ExprPool {
    /// Lock-free, append-only, reference-stable node array.
    nodes: boxcar::Vec<Node>,
    /// Deduplication index: ExprData → ExprId.  Holds ids, not data; see
    /// [`PoolIndex`].
    index: PoolIndex,
}

// `ExprPool` is `Send + Sync` *by inference*, not by assertion.
//
// This used to be `unsafe impl Send for ExprPool {}` / `unsafe impl Sync`, and
// nothing about the type ever needed it: every field is already `Send + Sync`
// (`boxcar::Vec<Node>` where `Node: Send + Sync`, and `PoolIndex`'s
// `RwLock`-guarded shards).  An unconditional `unsafe impl` on a type that
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
            index: PoolIndex::new(),
        }
    }

    /// Intern `data`, returning a shared [`ExprId`]. Identical structures
    /// always return the same id; structural equality ⟺ id equality.
    ///
    /// `data` is first brought to its canonical spelling, so that two
    /// spellings of one value cannot hold two ids:
    ///
    /// * a `Rational` whose denominator is `1` is the `Integer` numerator
    ///   (`rational(4, 2)` *is* `integer(2)`);
    /// * an `Add` or `Mul` of one argument is that argument, the empty `Add`
    ///   is `0` and the empty `Mul` is `1`;
    /// * a `Float` zero or `NaN` is stored with a positive sign.  `BigFloat`'s
    ///   `Eq` already identifies `-0.0` with `0.0` (and every `NaN` with every
    ///   other), so without this the node a pool kept would depend on which
    ///   spelling happened to be interned first.
    ///
    /// The result may therefore be a node of a different kind than `data`.
    /// Every constructor, and the pool-file loader, goes through here.
    pub fn intern(&self, data: ExprData) -> ExprId {
        let data = match self.canonical_form(data) {
            Ok(d) => d,
            Err(id) => return id,
        };
        // `boxcar::push` is lock-free, so it is safe to call while the index
        // shard's write lock is held.  `data` is moved into the node, not
        // cloned: the index keeps only the id.
        let make = |d: ExprData| {
            let node = self.make_node(d);
            ExprId(self.nodes.push(node) as u32)
        };
        self.index.get_or_insert_with(&self.nodes, data, make)
    }

    /// The canonical spelling of `data` (see [`ExprPool::intern`]), or
    /// `Err(id)` when it is already an interned node.
    fn canonical_form(&self, data: ExprData) -> Result<ExprData, ExprId> {
        match data {
            ExprData::Rational(r) if *r.0.denom() == 1 => {
                Ok(ExprData::Integer(BigInt(r.0.into_numer_denom().0)))
            }
            ExprData::Float(mut f)
                if (f.inner.is_zero() || f.inner.is_nan()) && f.inner.is_sign_negative() =>
            {
                f.inner.abs_mut();
                Ok(ExprData::Float(f))
            }
            ExprData::Add(args) if args.len() <= 1 => match args.first() {
                Some(&only) => Err(only),
                None => Err(self.integer(0_i32)),
            },
            ExprData::Mul(args) if args.len() <= 1 => match args.first() {
                Some(&only) => Err(only),
                None => Err(self.integer(1_i32)),
            },
            other => Ok(other),
        }
    }

    /// Wrap `data` with its cached properties.  Children are already interned,
    /// so their flags are just array reads.
    fn make_node(&self, data: ExprData) -> Node {
        let mult_commutative = self.compute_mult_commutative(&data);
        let depth = self.compute_depth(&data);
        let static_domain_fact = self.compute_static_domain_fact(&data);
        let non_finite = self.compute_non_finite(&data);
        Node {
            data,
            mult_commutative,
            depth,
            static_domain_fact,
            non_finite,
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
    /// One level of the static-domain-fact recurrence, from the children's
    /// cached flags.  Covers *every* child, piecewise conditions and binder
    /// variables included, because the fact walk descends into all of them.
    fn compute_static_domain_fact(&self, data: &ExprData) -> bool {
        let child = |c: ExprId| self.node(c).static_domain_fact;
        match data {
            ExprData::Symbol { domain, .. } => {
                matches!(domain, Domain::Positive | Domain::NonZero)
            }
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => false,
            ExprData::Add(args)
            | ExprData::Mul(args)
            | ExprData::Func { args, .. }
            | ExprData::Predicate { args, .. } => args.iter().copied().any(child),
            ExprData::Pow { base, exp } => child(*base) || child(*exp),
            ExprData::Piecewise { branches, default } => {
                branches.iter().any(|&(c, v)| child(c) || child(v)) || child(*default)
            }
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                child(*var) || child(*body)
            }
            ExprData::BigO(inner) => child(*inner),
            ExprData::RootSum { poly, var, body } => child(*poly) || child(*var) || child(*body),
        }
    }

    /// One level of the `non_finite` recurrence (every child, bound variables
    /// included — a flag that over-reports only makes a rule decline).
    fn compute_non_finite(&self, data: &ExprData) -> bool {
        let child = |c: ExprId| self.node(c).non_finite;
        match data {
            ExprData::Symbol { name, .. } => name == POS_INFINITY_SYMBOL,
            ExprData::Float(f) => !f.inner.is_finite(),
            ExprData::Integer(_) | ExprData::Rational(_) => false,
            ExprData::Add(args)
            | ExprData::Mul(args)
            | ExprData::Func { args, .. }
            | ExprData::Predicate { args, .. } => args.iter().copied().any(child),
            ExprData::Pow { base, exp } => child(*base) || child(*exp),
            ExprData::Piecewise { branches, default } => {
                branches.iter().any(|&(c, v)| child(c) || child(v)) || child(*default)
            }
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                child(*var) || child(*body)
            }
            ExprData::BigO(inner) => child(*inner),
            ExprData::RootSum { poly, var, body } => child(*poly) || child(*var) || child(*body),
        }
    }

    /// Whether some node of the subtree rooted at `id` is `∞` or a
    /// non-finite `Float`.  O(1): computed when `id` was interned.
    pub(crate) fn has_non_finite(&self, id: ExprId) -> bool {
        self.node(id).non_finite
    }

    /// Whether the subtree rooted at `id` contains a symbol whose domain is
    /// `Positive` or `NonZero`.  O(1): computed when `id` was interned.  When
    /// it is `false`, static domain-fact collection has nothing to find there.
    pub(crate) fn has_static_domain_fact(&self, id: ExprId) -> bool {
        self.node(id).static_domain_fact
    }

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

    /// `name(args…)`, unchecked.
    ///
    /// The arity of a built-in name is *not* validated here — this is the
    /// infallible constructor the library itself uses, always at the right
    /// arity.  Input from outside (a user, a parser, a file) should go through
    /// [`ExprPool::try_func`], which refuses `sin()` or `EllipticPi(x)`.
    pub fn func(&self, name: impl Into<String>, args: Vec<ExprId>) -> ExprId {
        self.intern(ExprData::Func {
            name: name.into(),
            args,
        })
    }

    /// `name(args…)`, refusing a built-in function name at the wrong arity
    /// (see [`crate::kernel::func_arity::known_func_arity`]) with
    /// [`FuncArityError`](crate::kernel::FuncArityError) (`E-POOL-002`).
    /// Names outside the built-in table (user functions) take any arity.
    pub fn try_func(
        &self,
        name: impl Into<String>,
        args: Vec<ExprId>,
    ) -> Result<ExprId, crate::kernel::FuncArityError> {
        let name = name.into();
        crate::kernel::FuncArityError::check(&name, args.len())?;
        Ok(self.func(name, args))
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

/// Format a power *base*.
///
/// Everything [`fmt_pow_atom`] wraps needs wrapping here too, plus any literal
/// that renders with a leading `-`: unary minus binds looser than `^` in this
/// crate's own parser (`BP_UNARY` < `BP_POW` in `parse.rs`), in Python and in
/// sympy, so `-1^n` re-reads as `-(1^n)`.  Only `(-1)^n` round-trips.
///
/// The exponent side deliberately keeps the bare form (`x^-1`): `^` is
/// right-associative and unary minus binds looser than it, so a `-` there is
/// already unambiguous.
fn fmt_pow_base(id: ExprId, pool: &ExprPool) -> String {
    let s = fmt_pow_atom(id, pool);
    if s.starts_with('-') {
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
            let base_s = fmt_pow_base(*base, pool);
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
            crate::kernel::expr::PredicateKind::Not if args.len() == 1 => {
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
            _ if args.len() == 2 => {
                write!(
                    f,
                    "({} {} {})",
                    pool.display(args[0]),
                    kind,
                    pool.display(args[1])
                )
            }
            // A comparison or `Not` with the wrong operand count — only the
            // raw `predicate` constructor can build one.  Print it rather
            // than index past the end.
            _ => {
                write!(f, "{kind}(")?;
                for (i, &arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", pool.display(arg))?;
                }
                write!(f, ")")
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

    /// One value, one id: a `Rational` with denominator 1 *is* the integer.
    #[test]
    fn rational_with_unit_denominator_is_the_integer() {
        let p = pool();
        assert_eq!(p.rational(4_i32, 2_i32), p.integer(2_i32));
        assert_eq!(p.rational(-6_i32, 3_i32), p.integer(-2_i32));
        assert_eq!(p.rational(0_i32, 5_i32), p.integer(0_i32));
        assert!(matches!(
            p.get(p.rational(4_i32, 2_i32)),
            ExprData::Integer(_)
        ));
        // Also through the raw `intern`, which the pool-file loader uses.
        let raw = p.intern(ExprData::Rational(BigRat(rug::Rational::from(7))));
        assert_eq!(raw, p.integer(7_i32));
    }

    /// `Add`/`Mul` of one argument is that argument; of none, the identity.
    #[test]
    fn unary_and_empty_sums_and_products_collapse() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        assert_eq!(p.add(vec![x]), x);
        assert_eq!(p.mul(vec![x]), x);
        assert_eq!(p.add(vec![]), p.integer(0_i32));
        assert_eq!(p.mul(vec![]), p.integer(1_i32));
        assert_eq!(p.add(vec![p.add(vec![x])]), x);
        assert_eq!(p.intern(ExprData::Mul(vec![x])), x);
        assert_eq!(p.intern(ExprData::Add(vec![])), p.integer(0_i32));
        // A non-commutative generator alone is still just itself.
        let a = p.symbol_commutative("A", Domain::Real, false);
        assert_eq!(p.mul(vec![a]), a);
        assert_eq!(p.display(p.add(vec![])).to_string(), "0");
        assert_eq!(p.display(p.mul(vec![])).to_string(), "1");
    }

    /// `-0.0` and `0.0` are one node (as `BigFloat`'s `Eq` already says), and
    /// the node kept is `+0.0` whichever spelling is interned first.
    #[test]
    fn signed_zero_floats_are_one_node_stored_positive() {
        let p = pool();
        let neg = p.float(-0.0, 53);
        let pos = p.float(0.0, 53);
        assert_eq!(neg, pos);
        match p.get(neg) {
            ExprData::Float(f) => assert!(f.inner.is_zero() && !f.inner.is_sign_negative()),
            other => panic!("{other:?}"),
        }
        assert_eq!(p.display(neg).to_string(), "0.0");
        let n1 = p.float(f64::NAN, 53);
        let n2 = p.float(-f64::NAN, 53);
        assert_eq!(n1, n2);
    }

    /// A `Float` never prints as something that reads back as an `Integer`.
    #[test]
    fn float_display_is_never_integer_shaped() {
        let p = pool();
        for (v, prec) in [
            (0.0, 53),
            (1.0, 53),
            (2.0_f64.powi(60), 53),
            (3.0, 64),
            (0.0, 200),
        ] {
            let s = p.display(p.float(v, prec)).to_string();
            assert!(s.contains(['.', 'e']), "{v} at {prec} printed as {s}");
            match p.get(crate::parse::parse(&s, &p, &mut Default::default()).unwrap()) {
                ExprData::Float(_) => {}
                other => panic!("{s} re-parsed as {other:?}"),
            }
        }
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

    /// The empty product is 1 and the empty sum is 0 — and since `intern`
    /// canonicalises them to exactly those literals, no empty same-operator
    /// child is left to splice: `x · Mul([])` is the ordinary `x · 1`, which
    /// `simplify` (not construction) reduces.
    #[test]
    fn empty_same_operator_children_are_the_identity_literals() {
        let p = pool();
        let x = p.symbol("x", Domain::Real);
        let empty_mul = p.mul(vec![]);
        assert_eq!(empty_mul, p.integer(1_i32));
        assert_eq!(p.mul(vec![x, empty_mul]), p.mul(vec![x, p.integer(1_i32)]));
        let empty_add = p.add(vec![]);
        assert_eq!(empty_add, p.integer(0_i32));
        assert_eq!(p.add(vec![x, empty_add]), p.add(vec![x, p.integer(0_i32)]));
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

    /// A negative power base must be parenthesised: `-1^n` re-reads as
    /// `-(1^n)` in this crate's own parser, in Python and in sympy.
    #[test]
    fn display_negative_pow_base_is_parenthesised() {
        let p = pool();
        let n = p.symbol("n", Domain::Real);
        let m1 = p.integer(-1_i32);
        assert_eq!(p.display(p.pow(m1, n)).to_string(), "(-1)^n");
        let m2 = p.integer(-2_i32);
        assert_eq!(p.display(p.pow(m2, n)).to_string(), "(-2)^n");
        let half = p.rational(-1, 2);
        assert_eq!(p.display(p.pow(half, n)).to_string(), "(-1/2)^n");
        // …including under a negative exponent, whose bare `-` is unambiguous.
        let m3 = p.integer(-3_i32);
        assert_eq!(p.display(p.pow(m2, m3)).to_string(), "(-2)^-3");
        // …and inside a product, the `b(n) = -16 * (-2)^n` boundary shape.
        // `mul` orders its arguments canonically, hence the factor order here.
        let m16 = p.integer(-16_i32);
        let prod = p.mul(vec![m16, p.pow(m2, n)]);
        assert_eq!(p.display(prod).to_string(), "((-2)^n * -16)");
    }

    /// A non-negative atom keeps the bare form — no gratuitous parentheses.
    #[test]
    fn display_positive_pow_base_is_bare() {
        let p = pool();
        let n = p.symbol("n", Domain::Real);
        let two = p.integer(2_i32);
        assert_eq!(p.display(p.pow(two, n)).to_string(), "2^n");
        let x = p.symbol("x", Domain::Real);
        assert_eq!(p.display(p.pow(x, n)).to_string(), "x^n");
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

#[cfg(test)]
mod intern_index_tests {
    use super::*;
    use crate::kernel::Domain;
    use std::sync::Barrier;

    /// Every thread interns the same values, in a different order, all
    /// starting at once.  Hash-consing must still give one id per value.
    #[test]
    fn concurrent_intern_of_the_same_values_agrees() {
        const THREADS: usize = 8;
        const N: i64 = 2_000;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let before = pool.len();
        let barrier = Barrier::new(THREADS);
        let per_thread: Vec<Vec<ExprId>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let (pool, barrier) = (&pool, &barrier);
                    s.spawn(move || {
                        barrier.wait();
                        // Visit 0..N in a thread-specific order (stride coprime
                        // to N), but record ids by value so they line up.
                        let stride = [1, 3, 7, 9, 11, 13, 17, 19][t];
                        let mut ids = vec![ExprId(u32::MAX); 4 * N as usize];
                        for j in 0..N {
                            let k = (j * stride) % N;
                            let i = k as usize;
                            ids[4 * i] = pool.integer(k - N / 2);
                            ids[4 * i + 1] = pool.rational(k, 7);
                            ids[4 * i + 2] = pool.float(k as f64 * 0.25, 53);
                            ids[4 * i + 3] = pool.add(vec![x, pool.integer(k + N)]);
                        }
                        ids
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for ids in &per_thread[1..] {
            assert_eq!(ids, &per_thread[0], "threads disagree on an id");
        }
        // Distinct values created: N integers in [-N/2, N/2), N rationals
        // k/7 (the multiples of 7 are integers already present, except those
        // outside [-N/2, N/2)), N floats, N integers k+N, N sums.
        let serial = ExprPool::new();
        let sx = serial.symbol("x", Domain::Real);
        let serial_before = serial.len();
        for k in 0..N {
            serial.integer(k - N / 2);
            serial.rational(k, 7);
            serial.float(k as f64 * 0.25, 53);
            serial.add(vec![sx, serial.integer(k + N)]);
        }
        assert_eq!(
            pool.len() - before,
            serial.len() - serial_before,
            "a racing intern created a duplicate node"
        );
        // And every id resolves back to the data that produced it.
        for k in 0..N {
            let i = 4 * k as usize;
            assert_eq!(
                pool.get(per_thread[0][i]),
                ExprData::Integer(crate::kernel::expr::BigInt(rug::Integer::from(k - N / 2)))
            );
        }
    }

    /// Intern through the index with a caller-chosen hash, exactly as
    /// `ExprPool::intern` does with the real one.
    fn intern_with_hash(pool: &ExprPool, hash: u64, data: ExprData) -> ExprId {
        pool.index
            .get_or_insert_hashed(&pool.nodes, hash, data, |d| {
                ExprId(pool.nodes.push(pool.make_node(d)) as u32)
            })
    }

    fn int(n: i64) -> ExprData {
        ExprData::Integer(crate::kernel::expr::BigInt(rug::Integer::from(n)))
    }

    /// A full 64-bit hash collision must not merge distinct nodes: the index
    /// stores only ids and hash bits, and decides equality by comparing the
    /// candidate against the stored node.
    #[test]
    fn forced_hash_collision_keeps_nodes_distinct() {
        let pool = ExprPool::new();
        // Every value gets the same hash, so they share a shard and a probe
        // sequence; 200 of them also force the colliding table to grow.
        let ids: Vec<ExprId> = (0..200)
            .map(|k| intern_with_hash(&pool, 7, int(k)))
            .collect();
        let distinct: std::collections::HashSet<ExprId> = ids.iter().copied().collect();
        assert_eq!(distinct.len(), 200, "a hash collision merged two nodes");
        for (k, &id) in ids.iter().enumerate() {
            assert_eq!(pool.get(id), int(k as i64));
            // A repeat finds the original, not a fresh copy.
            assert_eq!(intern_with_hash(&pool, 7, int(k as i64)), id);
        }
        // Same low 32 bits (the stored `Slot::hash`), different shard bits.
        let a = intern_with_hash(&pool, 0x0000_0000_dead_beef, int(-1));
        let b = intern_with_hash(&pool, 0xffff_ffff_dead_beef, int(-2));
        assert_ne!(a, b);
        assert_eq!(pool.len(), 202);
    }

    /// The pool never holds two nodes with equal data, and every node's data
    /// interns back to its own id.
    fn assert_hash_consed(pool: &ExprPool) {
        let mut seen = std::collections::HashMap::new();
        for i in 0..pool.len() {
            let id = ExprId(i as u32);
            let data = pool.get(id);
            assert_eq!(
                pool.intern(data.clone()),
                id,
                "node {i} re-interned elsewhere"
            );
            if let Some(prev) = seen.insert(data, id) {
                panic!("nodes {prev:?} and {id:?} hold equal data");
            }
        }
    }

    /// Many threads build the same *compound* structures concurrently — nested
    /// sums and products whose children are themselves being interned by the
    /// other threads at the same moment — while the index grows under them.
    /// Every thread must see the same ids, and no value may be stored twice.
    #[test]
    fn concurrent_compound_interning_stays_hash_consed() {
        const THREADS: usize = 8;
        const N: i64 = 1_500;
        let pool = ExprPool::new();
        let barrier = Barrier::new(THREADS);
        let per_thread: Vec<Vec<ExprId>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let (pool, barrier) = (&pool, &barrier);
                    s.spawn(move || {
                        let x = pool.symbol("x", Domain::Real);
                        let y = pool.symbol("y", Domain::Real);
                        barrier.wait();
                        let mut out = vec![ExprId(u32::MAX); N as usize];
                        // Half the threads walk forwards, half backwards, so
                        // the same node is often being created by two threads
                        // from opposite ends of the range.
                        let order: Vec<i64> = if t % 2 == 0 {
                            (0..N).collect()
                        } else {
                            (0..N).rev().collect()
                        };
                        for k in order {
                            let c = pool.integer(k);
                            let p = pool.pow(x, c);
                            let m = pool.mul(vec![c, p, y]);
                            let f = pool.func("sin", vec![m]);
                            out[k as usize] = pool.add(vec![f, p, pool.rational(k, 3)]);
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for ids in &per_thread[1..] {
            assert_eq!(ids, &per_thread[0], "threads disagree on an id");
        }
        assert_hash_consed(&pool);
    }

    /// Readers hitting existing entries while writers grow the same shards:
    /// every id handed out before or during the growth stays valid and stable.
    #[test]
    fn hits_during_concurrent_growth_are_stable() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let base: Vec<ExprId> = (0..500i64)
            .map(|k| pool.add(vec![x, pool.integer(k)]))
            .collect();
        std::thread::scope(|s| {
            for w in 0..4i64 {
                let pool = &pool;
                s.spawn(move || {
                    for k in 0..20_000i64 {
                        pool.integer(1_000_000 + w * 20_000 + k);
                    }
                });
            }
            for _ in 0..4 {
                let (pool, base) = (&pool, &base);
                s.spawn(move || {
                    for round in 0..40 {
                        for (k, &id) in base.iter().enumerate() {
                            let again = pool.add(vec![x, pool.integer(k as i64)]);
                            assert_eq!(again, id, "round {round}: id moved");
                        }
                    }
                });
            }
        });
        assert_eq!(pool.len(), 1 + 2 * 500 + 80_000);
        assert_hash_consed(&pool);
    }

    /// A panic while creating a node (here: a child id from another pool)
    /// must not wedge the index — it fires under the shard's write lock and
    /// poisons it, but before the slot is recorded, so later interns into the
    /// same shard carry on.
    #[test]
    fn a_panicking_intern_does_not_wedge_the_pool() {
        const H: u64 = 0x1234_5678_9abc_def0;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let bogus = ExprData::Pow {
            base: x,
            exp: ExprId(1_000_000),
        };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            intern_with_hash(&pool, H, bogus);
        }));
        assert!(r.is_err());
        let before = pool.len();
        // Same hash, so the same (now poisoned) shard.
        let a = intern_with_hash(&pool, H, int(5));
        assert_eq!(intern_with_hash(&pool, H, int(5)), a);
        assert_eq!(pool.get(a), int(5));
        assert_eq!(pool.len(), before + 1);
    }

    /// Growing the index past many resizes keeps every id stable.
    #[test]
    fn ids_survive_index_growth() {
        let pool = ExprPool::new();
        let ids: Vec<ExprId> = (0..50_000i64).map(|k| pool.integer(k)).collect();
        for (k, id) in ids.iter().enumerate() {
            assert_eq!(pool.integer(k as i64), *id);
        }
        assert_eq!(pool.len(), 50_000);
    }
}

#[cfg(test)]
mod intern_timing {
    //! `#[ignore]`d timing harness for number-atom hashing and interning.
    //! Run with
    //! `cargo test --release -p alkahest-core --lib intern_timing -- --ignored --nocapture`.
    use super::*;
    use crate::kernel::expr::{BigFloat, BigInt};
    use rug::ops::Pow;
    use std::hash::BuildHasher;
    use std::time::Instant;

    fn per_op(label: &str, iters: u32, mut f: impl FnMut()) {
        for _ in 0..(iters / 10).max(1) {
            f();
        }
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let ns = t.elapsed().as_nanos() as f64 / f64::from(iters);
        println!("{label:<40} {ns:>12.1} ns/op");
    }

    #[test]
    #[ignore]
    fn number_atom_hash_and_intern_timings() {
        let rs = std::collections::hash_map::RandomState::new();
        let small = BigInt(rug::Integer::from(123_456_789));
        let big = BigInt(rug::Integer::from(3).pow(2000));
        let huge = rug::Integer::from(1) << 100_000u32;
        let flt = BigFloat {
            inner: rug::Float::with_val(53, 1.1),
            prec: 53,
        };
        per_op("hash BigInt small", 1_000_000, || {
            std::hint::black_box(rs.hash_one(std::hint::black_box(&small)));
        });
        per_op("hash BigInt 3^2000", 100_000, || {
            std::hint::black_box(rs.hash_one(std::hint::black_box(&big)));
        });
        per_op("hash BigFloat 53-bit", 1_000_000, || {
            std::hint::black_box(rs.hash_one(std::hint::black_box(&flt)));
        });

        let p = ExprPool::new();
        for k in 0..1000i64 {
            p.integer(k);
        }
        let mut k = 0i64;
        per_op("pool.integer(k) hit", 1_000_000, || {
            k = (k + 1) % 1000;
            std::hint::black_box(p.integer(k));
        });
        per_op("pool.rational(1,3) hit", 1_000_000, || {
            std::hint::black_box(p.rational(1, 3));
        });
        per_op("pool.float(1.1,53) hit", 1_000_000, || {
            std::hint::black_box(p.float(1.1, 53));
        });
        p.integer(huge.clone());
        per_op("pool.integer(2^100000) hit", 10_000, || {
            std::hint::black_box(p.integer(huge.clone()));
        });

        let fresh = ExprPool::new();
        let mut n = 0i64;
        per_op("pool.integer miss (fresh values)", 1_000_000, || {
            n += 1;
            std::hint::black_box(fresh.integer(n));
        });
        let x = fresh.symbol("x", crate::kernel::Domain::Real);
        let mut m = 0i64;
        per_op("pool.add([x, k]) miss", 500_000, || {
            m += 1;
            let c = fresh.integer(m);
            std::hint::black_box(fresh.add(vec![x, c]));
        });
    }
}
