pub mod algebraic;
mod beta_family;
pub(crate) mod branch_check;
pub mod by_parts;
mod classics;
mod continuity;
mod derivative;
mod elliptic_exact;
mod elliptic_trig;
pub mod engine;
mod exp_log_normal;
mod exp_subst;
mod feynman;
pub mod gate;
pub mod norman;
mod piecewise;
mod prenorm;
mod rational_atan;
mod real_log;
pub(crate) mod residue_theorem;
pub mod risch;
pub mod special;
mod special_subst;
mod trig_angle;

pub use engine::{
    integrate, integrate_classified, integrate_definite, verify_antiderivative_exact,
    verify_antiderivative_status, verify_antiderivative_status_parametric,
    AntiderivativeVerification, IntegrationError,
};
pub use special::{basis_functions_used, IntegrationAnswer, SPECIAL_BASIS};
