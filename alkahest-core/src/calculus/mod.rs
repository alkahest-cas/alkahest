//! Symbolic calculus utilities — truncated expansions, limits, …

pub mod asymptotic;
pub mod asymptotic_common;
pub mod euler_maclaurin;
pub mod fps;
pub mod gruntz;
pub mod limits;
pub mod multilimit;
pub mod puiseux;
pub mod series;
pub mod singularity;

pub use asymptotic::{asymptotic_expand, AsymptoticError, AsymptoticExpansion, AsymptoticTerm};
pub use fps::{Fps, FpsError};
pub use limits::{limit, LimitDirection, LimitError};
pub use multilimit::{multilimit, MultiLimit, PathWitness};
pub use puiseux::{puiseux_series, PuiseuxError, PuiseuxExpansion};
pub use series::{series, take_series_refusal, Series, SeriesError, SeriesRefusal};
