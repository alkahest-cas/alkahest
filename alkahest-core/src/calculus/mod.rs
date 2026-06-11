//! Symbolic calculus utilities — truncated expansions, limits, …

pub mod fps;
pub mod gruntz;
pub mod limits;
pub mod series;

pub use fps::{Fps, FpsError};
pub use limits::{limit, LimitDirection, LimitError};
pub use series::{series, Series, SeriesError};
