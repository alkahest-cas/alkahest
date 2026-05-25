//! Content-addressed cache for JIT-compiled functions.
//!
//! [`CompileCache`] maps `(ExprId, Vec<ExprId>)` → `Arc<CompiledFn>`.
//! Because [`ExprPool`](crate::kernel::ExprPool) already hash-conses
//! expressions, `ExprId` is a stable content key: the same expression tree
//! always produces the same `ExprId`.  This means **the cache key _is_ the
//! content hash** — no separate hashing of the expression tree is required.
//!
//! # Key design
//!
//! The cache key is `(ExprId, Vec<ExprId>)` — the root expression plus the
//! ordered list of input variables.  Two compilations of the same expression
//! with different variable orderings produce separate entries (and separate
//! compiled functions with different argument positions).
//!
//! # Lifetime of compiled code
//!
//! Each cached value is an `Arc<CompiledFn>`.  The compiled code stays alive
//! as long as any live `Arc` references it — clearing or dropping the cache
//! does not invalidate `Arc`s already returned to callers.
//!
//! # Thread safety
//!
//! `CompileCache` itself requires `&mut self` for writes and is therefore
//! single-owner.  Wrap in `Mutex<CompileCache>` or `RwLock<CompileCache>` for
//! shared multi-threaded access.  `Arc<CompiledFn>` is `Send + Sync` so
//! compiled functions can be freely shared across threads after retrieval.
//!
//! # Example
//!
//! ```
//! use alkahest_cas::kernel::{Domain, ExprPool};
//! use alkahest_cas::jit::CompileCache;
//! use std::sync::Arc;
//!
//! let pool = ExprPool::new();
//! let x = pool.symbol("x", Domain::Real);
//! let expr = pool.pow(x, pool.integer(2_i32));
//!
//! let mut cache = CompileCache::new();
//!
//! // First call compiles
//! let f1 = cache.compile(expr, &[x], &pool).unwrap();
//! // Second call is a cache hit — same Arc, no recompilation
//! let f2 = cache.compile(expr, &[x], &pool).unwrap();
//!
//! assert!(Arc::ptr_eq(&f1, &f2));
//! assert!((f1.call(&[3.0]) - 9.0).abs() < 1e-10);
//! ```

use super::{compile, CompiledFn, JitError};
use crate::kernel::{ExprId, ExprPool};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// `(expression root, ordered input variables)`.
///
/// The `Vec<ExprId>` captures variable order — two compilations of the same
/// expression with different orderings produce separate entries.
type CacheKey = (ExprId, Vec<ExprId>);

// ---------------------------------------------------------------------------
// CompileCache
// ---------------------------------------------------------------------------

/// Content-addressed cache of JIT-compiled functions.
///
/// See the [module documentation](self) for full details.
pub struct CompileCache {
    store: HashMap<CacheKey, Arc<CompiledFn>>,
    /// Total number of compilations (cache misses + initial compiles).
    compiles: u64,
    /// Total number of cache hits.
    hits: u64,
}

impl CompileCache {
    /// Create a new, empty cache.
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
            compiles: 0,
            hits: 0,
        }
    }

    /// Compile `expr` with the given `inputs`, returning a cached `Arc<CompiledFn>`.
    ///
    /// # Cache behaviour
    ///
    /// - **Miss**: the expression is compiled via [`jit::compile`](super::compile)
    ///   and the result is stored.  Subsequent calls with the same `(expr,
    ///   inputs)` pair return the cached value immediately.
    /// - **Hit**: returns `Arc::clone` of the cached value — O(1), no
    ///   recompilation.
    ///
    /// # Errors
    ///
    /// Returns `Err(JitError)` only on a cache miss where compilation fails.
    /// Cache hits never fail.
    pub fn compile(
        &mut self,
        expr: ExprId,
        inputs: &[ExprId],
        pool: &ExprPool,
    ) -> Result<Arc<CompiledFn>, JitError> {
        let key: CacheKey = (expr, inputs.to_vec());
        if let Some(cached) = self.store.get(&key) {
            self.hits += 1;
            return Ok(Arc::clone(cached));
        }
        self.compiles += 1;
        let compiled = Arc::new(compile(expr, inputs, pool)?);
        self.store.insert(key, Arc::clone(&compiled));
        Ok(compiled)
    }

    /// Number of `(expr, inputs)` pairs currently cached.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Returns `true` if a compiled function for `(expr, inputs)` is cached.
    pub fn contains(&self, expr: ExprId, inputs: &[ExprId]) -> bool {
        self.store.contains_key(&(expr, inputs.to_vec()))
    }

    /// Total number of compilations performed (cache misses that succeeded).
    pub fn compile_count(&self) -> u64 {
        self.compiles
    }

    /// Total number of cache hits.
    pub fn hit_count(&self) -> u64 {
        self.hits
    }

    /// Cache hit rate in `[0.0, 1.0]`; `0.0` when no lookups have been made.
    pub fn hit_rate(&self) -> f64 {
        let total = self.compiles + self.hits;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Evict all cached functions, freeing compiled code unless other `Arc`s
    /// keep them alive.
    pub fn clear(&mut self) {
        self.store.clear();
        // Keep statistics — they describe the lifetime of the cache, not just
        // the current contents.
    }

    /// Evict a single entry.  Returns the cached function if it was present.
    pub fn evict(&mut self, expr: ExprId, inputs: &[ExprId]) -> Option<Arc<CompiledFn>> {
        self.store.remove(&(expr, inputs.to_vec()))
    }
}

impl Default for CompileCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn cache_miss_then_hit() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(2_i32));

        let mut cache = CompileCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.compile_count(), 0);
        assert_eq!(cache.hit_count(), 0);

        let f1 = cache.compile(expr, &[x], &pool).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.compile_count(), 1);
        assert_eq!(cache.hit_count(), 0);

        let f2 = cache.compile(expr, &[x], &pool).unwrap();
        assert_eq!(cache.len(), 1); // still one entry
        assert_eq!(cache.compile_count(), 1); // no new compile
        assert_eq!(cache.hit_count(), 1);

        // Same Arc — identical pointer
        assert!(Arc::ptr_eq(&f1, &f2));
    }

    #[test]
    fn cache_correct_result() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(2_i32));

        let mut cache = CompileCache::new();
        let f = cache.compile(expr, &[x], &pool).unwrap();
        assert!((f.call(&[3.0]) - 9.0).abs() < 1e-10);
        assert!((f.call(&[5.0]) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn different_var_order_different_entry() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![x, y]);

        let mut cache = CompileCache::new();
        let f_xy = cache.compile(expr, &[x, y], &pool).unwrap();
        let f_yx = cache.compile(expr, &[y, x], &pool).unwrap();

        // Different orderings → different cache entries
        assert_eq!(cache.len(), 2);
        assert!(!Arc::ptr_eq(&f_xy, &f_yx));

        // f_xy: inputs[0]=x, inputs[1]=y; call(1.0, 2.0) → x=1, y=2 → 3
        assert!((f_xy.call(&[1.0, 2.0]) - 3.0).abs() < 1e-10);
        // f_yx: inputs[0]=y, inputs[1]=x; call(1.0, 2.0) → y=1, x=2 → 3
        assert!((f_yx.call(&[1.0, 2.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn different_exprs_different_entries() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sq = pool.pow(x, pool.integer(2_i32));
        let cube = pool.pow(x, pool.integer(3_i32));

        let mut cache = CompileCache::new();
        let f_sq = cache.compile(sq, &[x], &pool).unwrap();
        let f_cu = cache.compile(cube, &[x], &pool).unwrap();

        assert_eq!(cache.len(), 2);
        assert!(!Arc::ptr_eq(&f_sq, &f_cu));
        assert!((f_sq.call(&[3.0]) - 9.0).abs() < 1e-10);
        assert!((f_cu.call(&[3.0]) - 27.0).abs() < 1e-10);
    }

    #[test]
    fn arc_survives_cache_clear() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(2_i32));

        let mut cache = CompileCache::new();
        let f = cache.compile(expr, &[x], &pool).unwrap();

        cache.clear();
        assert!(cache.is_empty());

        // f still valid — Arc keeps it alive
        assert!((f.call(&[4.0]) - 16.0).abs() < 1e-10);
    }

    #[test]
    fn evict_removes_single_entry() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sq = pool.pow(x, pool.integer(2_i32));
        let cube = pool.pow(x, pool.integer(3_i32));

        let mut cache = CompileCache::new();
        cache.compile(sq, &[x], &pool).unwrap();
        cache.compile(cube, &[x], &pool).unwrap();
        assert_eq!(cache.len(), 2);

        let evicted = cache.evict(sq, &[x]);
        assert!(evicted.is_some());
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains(sq, &[x]));
        assert!(cache.contains(cube, &[x]));
    }

    #[test]
    fn contains_checks_key() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![x, y]);

        let mut cache = CompileCache::new();
        assert!(!cache.contains(expr, &[x, y]));
        cache.compile(expr, &[x, y], &pool).unwrap();
        assert!(cache.contains(expr, &[x, y]));
        assert!(!cache.contains(expr, &[y, x])); // different order
    }

    #[test]
    fn hit_rate_is_correct() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.pow(x, pool.integer(2_i32));

        let mut cache = CompileCache::new();
        assert_eq!(cache.hit_rate(), 0.0);

        cache.compile(expr, &[x], &pool).unwrap(); // miss
        assert_eq!(cache.hit_rate(), 0.0); // 0/1

        cache.compile(expr, &[x], &pool).unwrap(); // hit
        cache.compile(expr, &[x], &pool).unwrap(); // hit

        // 2 hits / 3 total = 2/3
        let rate = cache.hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-10);
    }
}
