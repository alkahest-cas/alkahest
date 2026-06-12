//! Symbolic calculus utilities — truncated expansions, limits, …

pub mod asymptotic;
pub mod fps;
pub mod gruntz;
pub mod limits;
pub mod multilimit;
pub mod series;

pub use asymptotic::{asymptotic_expand, AsymptoticError, AsymptoticExpansion, AsymptoticTerm};
pub use fps::{Fps, FpsError};
pub use limits::{limit, LimitDirection, LimitError};
pub use multilimit::{multilimit, MultiLimit, PathWitness};
pub use series::{series, Series, SeriesError};
