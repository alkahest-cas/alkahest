pub mod diff_impl;
pub mod forward;
pub mod reverse;
pub mod undefined;

#[cfg(test)]
mod proptests;

pub use diff_impl::{diff, DiffError};
#[allow(deprecated)]
pub use forward::{diff_forward, DualValue, ForwardDiffError};
pub use reverse::{grad, try_grad};
pub use undefined::{
    euler_lagrange, functions_to_symbols, is_undefined_function_name, replace_function,
    FunctionSymbol, UndefinedFunctionError,
};
