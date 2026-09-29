//! A `HashMap` keyed by [`ExprId`] with a hasher sized for it.
//!
//! The evaluators keep a per-call memo from node to value, and on small
//! expressions hashing that memo with SipHash (the `std` default) costs more
//! than the arithmetic it saves. An `ExprId` is a dense pool index: nothing an
//! adversary chooses, so HashDoS resistance buys nothing here, and a single
//! multiplicative mix spreads it across the table.

use crate::kernel::ExprId;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// `HashMap<ExprId, V>` with [`IdHasher`].
pub(crate) type IdMap<V> = HashMap<ExprId, V, BuildHasherDefault<IdHasher>>;

/// Fibonacci-hashing mixer for the `u32` inside an [`ExprId`].
#[derive(Default, Clone, Copy)]
pub(crate) struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        // hashbrown takes its bucket index from the low bits and its control
        // tag from the top 7, so the mix must reach both ends of the word.
        let h = (self.0 ^ u64::from(n)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = h ^ (h >> 32);
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Not reached for `ExprId` (it hashes as one `u32`); kept correct for
        // any other key shape.
        for &b in bytes {
            self.write_u32(u32::from(b));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn id_map_round_trips_many_ids() {
        let pool = ExprPool::new();
        let ids: Vec<ExprId> = (0..2000)
            .map(|i| pool.symbol(format!("v{i}"), Domain::Real))
            .collect();
        let mut m: IdMap<usize> = IdMap::default();
        for (i, &id) in ids.iter().enumerate() {
            m.insert(id, i);
        }
        assert_eq!(m.len(), ids.len());
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(m[id], i);
        }
    }
}
