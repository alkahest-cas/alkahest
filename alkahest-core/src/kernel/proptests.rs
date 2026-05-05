//! Property-based tests for the expression kernel.
//!
//! Invariants verified:
//!   1. Same data → same ExprId (intern is idempotent)
//!   2. Different data → different ExprId
//!   3. get(intern(data)) == data (round-trip)

#[cfg(test)]
mod tests {
    use crate::kernel::{domain::Domain, expr::ExprData, pool::ExprPool};
    use proptest::prelude::*;

    // -----------------------------------------------------------------------
    // Strategies
    // -----------------------------------------------------------------------

    fn domain_strategy() -> impl Strategy<Value = Domain> {
        prop_oneof![
            Just(Domain::Real),
            Just(Domain::Complex),
            Just(Domain::Integer),
            Just(Domain::Positive),
            Just(Domain::NonNegative),
            Just(Domain::NonZero),
        ]
    }

    fn name_strategy() -> impl Strategy<Value = String> {
        // Short alpha identifiers to keep expression trees readable in failures.
        "[a-z]{1,4}".prop_map(|s| s)
    }

    /// Generates a flat ExprData (atoms only) to avoid needing a pool in the
    /// strategy itself (compound nodes require valid ExprIds).
    fn atom_data_strategy() -> impl Strategy<Value = ExprData> {
        prop_oneof![
            (name_strategy(), domain_strategy()).prop_map(|(n, d)| ExprData::Symbol {
                name: n,
                domain: d,
                commutative: true,
            }),
            (i64::MIN..=i64::MAX).prop_map(|n| ExprData::Integer(crate::kernel::expr::BigInt(
                rug::Integer::from(n)
            ))),
            // Rationals: denom must be non-zero; use 1..1000 to avoid division
            (i32::MIN..=i32::MAX, 1_i32..=10000_i32).prop_map(|(n, d)| {
                ExprData::Rational(crate::kernel::expr::BigRat(rug::Rational::from((
                    rug::Integer::from(n),
                    rug::Integer::from(d),
                ))))
            }),
        ]
    }

    // -----------------------------------------------------------------------
    // Properties
    // -----------------------------------------------------------------------

    proptest! {
        /// Same data interned twice yields the same ExprId.
        #[test]
        fn same_data_same_id(data in atom_data_strategy()) {
            let p = ExprPool::new();
            let id1 = p.intern(data.clone());
            let id2 = p.intern(data);
            prop_assert_eq!(id1, id2);
        }

        /// After interning, the pool size grows by 1 for a fresh entry.
        #[test]
        fn fresh_intern_grows_pool(data in atom_data_strategy()) {
            let p = ExprPool::new();
            let before = p.len();
            p.intern(data.clone());
            let after_first = p.len();
            p.intern(data);
            let after_second = p.len();
            // first intern must add exactly one node
            prop_assert_eq!(after_first, before + 1);
            // second intern of the same data must not add a node
            prop_assert_eq!(after_second, after_first);
        }

        /// Round-trip: get(intern(data)) == data.
        #[test]
        fn round_trip(data in atom_data_strategy()) {
            let p = ExprPool::new();
            let id = p.intern(data.clone());
            prop_assert_eq!(p.get(id), data);
        }

        /// Two distinct symbols (different names or domains) produce distinct ids.
        #[test]
        fn distinct_symbols_distinct_ids(
            name1 in name_strategy(),
            name2 in name_strategy(),
            d1 in domain_strategy(),
            d2 in domain_strategy(),
        ) {
            // Only assert when the two symbols are structurally different.
            if name1 == name2 && d1 == d2 {
                return Ok(());
            }
            let p = ExprPool::new();
            let id1 = p.symbol(&name1, d1);
            let id2 = p.symbol(&name2, d2);
            if name1 != name2 || d1 != d2 {
                prop_assert_ne!(id1, id2);
            }
        }

        /// ExprId comparison mirrors structural equality for atom symbols.
        #[test]
        fn id_equality_mirrors_structural_equality(
            n in name_strategy(),
            d in domain_strategy(),
        ) {
            let p = ExprPool::new();
            let id_a = p.symbol(&n, d);
            let id_b = p.symbol(&n, d);
            // Same structure → same id → equal
            prop_assert_eq!(id_a, id_b);
        }
    }
}
