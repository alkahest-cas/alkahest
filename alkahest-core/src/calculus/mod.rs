//! Symbolic calculus utilities — truncated expansions, limits, …

pub mod limits;
pub mod series;

pub use limits::{limit, LimitDirection, LimitError};
pub use series::{series, Series, SeriesError};
