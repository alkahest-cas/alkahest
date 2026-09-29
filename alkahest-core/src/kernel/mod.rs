pub mod depth;
pub mod display;
pub mod domain;
pub mod eval_const;
mod eval_proptests;
pub mod expr;
pub mod expr_props;
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
pub use expr_props::{expr_contains_noncommutative_symbol, mult_tree_is_commutative};
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
