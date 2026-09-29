//! Hash maps keyed by [`ExprId`] for the simplifier's per-pass side tables.
//!
//! An `ExprId` is a dense `u32` index handed out by the pool, so it needs no
//! DoS-resistant hashing: SipHash (the `std` default) spends most of a memo
//! probe mixing four bytes that are already unique.  [`IdHasher`] is the
//! multiplicative (Fibonacci) hash `rustc` uses for the same kind of key —
//! one multiply per probe.
//!
//! These maps are what replaced the `pool.len()`-sized arrays the level
//! scheduler and the shape probe used to allocate per pass: an array indexed
//! by `ExprId` is sized by the *pool*, not the expression, so simplifying a
//! 7-node expression in a pool of 8 million nodes allocated and zeroed two
//! 8-million-entry tables per pass.  A map is sized by what the traversal
//! actually reaches.

use crate::kernel::ExprId;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// Multiplicative hasher for small integer keys.  Not DoS-resistant; only for
/// keys the process generated itself, such as [`ExprId`].
#[derive(Default, Clone, Copy)]
pub(crate) struct IdHasher(u64);

/// 2^64 / φ, the Fibonacci-hashing multiplier.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

impl Hasher for IdHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(SEED);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub(crate) type IdBuildHasher = BuildHasherDefault<IdHasher>;

/// `HashMap<ExprId, V>` with [`IdHasher`].
pub(crate) type IdMap<V> = HashMap<ExprId, V, IdBuildHasher>;

/// `HashSet<ExprId>` with [`IdHasher`].
pub(crate) type IdSet = HashSet<ExprId, IdBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn behaves_like_a_map() {
        let pool = ExprPool::new();
        let ids: Vec<ExprId> = (0..1000).map(|i| pool.integer(i)).collect();
        let mut m: IdMap<usize> = IdMap::default();
        for (i, &id) in ids.iter().enumerate() {
            m.insert(id, i);
        }
        let x = pool.symbol("x", Domain::Real);
        assert_eq!(m.len(), 1000);
        assert!(!m.contains_key(&x));
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(m[&id], i);
        }
    }
}
