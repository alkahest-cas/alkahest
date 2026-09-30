pub mod depth;
pub mod display;
pub mod domain;
pub mod eval_const;
mod eval_proptests;
pub mod expr;
pub mod expr_props;
pub mod func_arity;
pub mod pool;
pub mod pool_persist;
mod printer_roundtrip;
mod proptests;
pub mod subs;

pub use depth::{check_expr_depth, check_expr_depths, DepthLimitError, MAX_EXPR_DEPTH};
pub use display::{render_latex, render_unicode};
pub use domain::Domain;
pub use eval_const::{
    integer_is_exact_f64, integer_to_f64, pow_f64, pow_f64_integer_exponent, rational_to_f64,
    try_expr_f64, try_predicate_bool, try_predicate_bool_from_expr,
};
pub use expr::{BigFloat, BigInt, BigRat, ExprData, ExprId};
pub use expr_props::{
    contains_non_finite_atom, expr_contains_noncommutative_symbol, is_non_finite_atom,
    mult_tree_is_commutative,
};
pub use func_arity::{func_arity_ok, known_func_arity, FuncArityError};
pub use pool::{ExprDisplay, ExprPool};
#[allow(deprecated)]
pub use pool_persist::PoolPersistError;
pub use pool_persist::{load_from, open_persistent, save_to, IoError};
pub use subs::{fold_predicates, subs};

/// Hasher for the crate's internal tables keyed by [`ExprId`] (memo maps,
/// visited sets).  `std`'s default SipHash is built to resist HashDoS, which
/// costs several times more per probe than these tables' keys warrant: an
/// `ExprId` is a pool-assigned index, not attacker-chosen data.  Maps keyed by
/// strings or other caller-supplied data should keep the default hasher.
pub(crate) type IdBuildHasher = foldhash::fast::FixedState;

/// `HashMap<ExprId, V>` with the fast [`IdBuildHasher`].  Construct with
/// `IdMap::default()` (`HashMap::new` exists only for the default hasher).
pub(crate) type IdMap<V> = std::collections::HashMap<ExprId, V, IdBuildHasher>;

/// `HashSet<ExprId>` with the fast [`IdBuildHasher`].
pub(crate) type IdSet = std::collections::HashSet<ExprId, IdBuildHasher>;

/// A "visited" set of [`ExprId`]s that stays off the heap while small.
///
/// The polynomial builders record every compound node they convert, so that
/// a node reached a second time (a shared DAG node) can be memoised.  On the
/// small polynomials that dominate real use, allocating and growing a hash
/// set for that — three allocations and two rehashes for a quartic — cost
/// more than the conversion's own arithmetic.  The first [`Self::INLINE`] ids
/// live in an inline array searched linearly; beyond that they move to an
/// [`IdSet`].
pub(crate) struct IdSeen {
    inline: [ExprId; IdSeen::INLINE],
    len: usize,
    spill: IdSet,
}

impl Default for IdSeen {
    fn default() -> Self {
        IdSeen {
            inline: [ExprId(0); IdSeen::INLINE],
            len: 0,
            spill: IdSet::default(),
        }
    }
}

impl IdSeen {
    const INLINE: usize = 16;

    /// Adds `id`; returns `true` if it was not already present (the
    /// [`std::collections::HashSet::insert`] contract).
    pub(crate) fn insert(&mut self, id: ExprId) -> bool {
        if self.len < Self::INLINE {
            if self.inline[..self.len].contains(&id) {
                return false;
            }
            self.inline[self.len] = id;
            self.len += 1;
            return true;
        }
        if self.spill.is_empty() {
            if self.inline.contains(&id) {
                return false;
            }
            self.spill.reserve(2 * Self::INLINE);
            self.spill.extend(self.inline);
        }
        self.spill.insert(id)
    }
}

#[cfg(test)]
mod id_seen_tests {
    use super::*;

    /// `IdSeen::insert` answers exactly as `HashSet::insert` does, below,
    /// at and past the inline capacity, with repeats before and after the
    /// spill.
    #[test]
    fn id_seen_matches_a_hash_set_across_the_spill() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        for round in 0..200u32 {
            let universe = 1 + round % 60;
            let mut seen = IdSeen::default();
            let mut reference = std::collections::HashSet::new();
            for _ in 0..(3 * universe) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let id = ExprId((state % u64::from(universe)) as u32);
                assert_eq!(seen.insert(id), reference.insert(id), "round {round}");
            }
        }
    }
}
