//! Symbolic calculus utilities — truncated expansions, limits, …

pub mod gruntz;
pub mod limits;
pub mod multilimit;
pub mod series;

pub use limits::{limit, LimitDirection, LimitError};
pub use multilimit::{multilimit, MultiLimit, PathWitness};
pub use series::{series, Series, SeriesError};
