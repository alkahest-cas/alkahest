pub mod diff_impl;
pub mod forward;
pub mod reverse;

#[cfg(test)]
mod proptests;

pub use diff_impl::{diff, DiffError};
#[allow(deprecated)]
pub use forward::{diff_forward, DualValue, ForwardDiffError};
pub use reverse::{grad, try_grad};
