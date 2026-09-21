//! Function fields of algebraic curves: divisors, the divisor class group and
//! Riemann–Roch.
//!
//! This module turns the algebraic-curve machinery that
//! [`crate::integrate::algebraic`] built for Risch integration into a
//! first-class API.  It **reuses** that machinery rather than reimplementing
//! it: Cantor arithmetic on Mumford representations, and the
//! reduction-modulo-good-primes torsion test, are the same code the symbolic
//! integrator relies on.
//!
//! # Scope — read this before using anything here
//!
//! The implemented class is the **imaginary hyperelliptic** function field
//!
//! ```text
//!     K = ℚ(x)[y] / (y² − a(x)),     a squarefree,  deg a = 2g + 1  odd,
//! ```
//!
//! together with **ℚ-rational places only**.  That boundary is not an
//! accident: the Mumford/Cantor representation reused from
//! [`crate::integrate::algebraic::jacobian_torsion`] and
//! [`crate::integrate::algebraic::coates`] is built on a *single rational place
//! at infinity*, and those modules are themselves explicitly scoped to `n = 2`
//! with `a` of odd degree.  Nothing here claims more.
//!
//! ## What is supported
//!
//! | Operation | Model | Places |
//! |---|---|---|
//! | [`FunctionField::genus`] | any accepted model, odd **or even** degree | — |
//! | [`Divisor`] arithmetic, [`FunctionFieldElement::divisor`] | imaginary (odd) only | rational only |
//! | [`DivisorClass`] (add, negate, reduce, compare, order) | imaginary (odd) only | rational only |
//! | [`riemann_roch`] — `dim L(D)` and a basis | imaginary (odd) only | rational only |
//! | [`FunctionField::canonical_divisor`] | imaginary (odd) only | — |
//!
//! Construction accepts `c₂·y² + c₁(x)·y + c₀(x)` with `c₂` a non-zero rational
//! constant, and normalises it to `y² = a(x)` with `a` squarefree by completing
//! the square and peeling square factors.  Both steps are ℚ(x)-isomorphisms;
//! [`FunctionField::normalisation`] records the coordinate change, because
//! **places are always expressed in the normalised model**.
//!
//! ## What is refused, and why
//!
//! Every boundary below is a typed `E-FFLD-*` refusal, never a guess.
//!
//! * **`deg_y f > 2`** (`E-FFLD-001`).  Trigonal and higher plane models, and
//!   superelliptic `yⁿ = a(x)` with `n > 2`.  There is no Mumford
//!   representation for them here; `integral_basis`/`vanhoeij` can produce an
//!   integral basis for such curves, but the class-group and Riemann–Roch
//!   algorithms in this module are `n = 2` constructions.
//! * **Even-degree ("real") models** `deg a = 2g+2` (`E-FFLD-002`).  These have
//!   **two** places above `x = ∞`, and the Jacobian arithmetic here measures
//!   every class against one rational base point.  The *genus* is still
//!   reported — it does not depend on the model — so `y² = x⁶ + x + 1` will
//!   tell you `g = 2` and then decline to build divisors.
//! * **Places of degree ≥ 2** (`E-FFLD-003`).  A conjugate pair `(α, ±√c)` with
//!   `c` a non-square, or any place over an irrational `α`, has no
//!   representation in [`Place`].  Dropping it would change the degree of the
//!   divisor silently, so operations that meet one refuse.  This is the most
//!   frequently hit boundary in practice: `div(y)` on `y² = x⁵ + 1` refuses,
//!   because `x⁵ + 1` has only one rational root.  The refusal is
//!   *conservative* — the rational-root search is capped, so it never proves
//!   irreducibility.
//! * **Non-torsion classes** (`E-FFLD-007`).  A **verdict**, not a refusal: no
//!   multiple of the divisor is principal.  Keep it apart from
//!   `E-FFLD-006`, which says the order could not be decided.
//!
//! ## Self-checks
//!
//! Two results are computed and then **withheld** if they fail their own check
//! (`E-FFLD-011`):
//!
//! * `div(u)` cross-checks the multiplicity at infinity obtained from degree
//!   balance against the one read off `deg p` and `deg q`.
//! * `riemann_roch` checks its dimension against Riemann's inequality
//!   `dim L(D) ≥ deg D + 1 − g`, and — above the canonical degree — against the
//!   Riemann–Roch equality `dim L(D) = deg D + 1 − g`.
//!
//! # Example
//!
//! ```
//! use alkahest_cas::experimental::{FunctionField, Divisor, Place, riemann_roch};
//! use rug::Integer;
//!
//! // y² = x³ − x, an elliptic curve.
//! let e = FunctionField::hyperelliptic_from_i64(&[0, -1, 0, 1]).unwrap();
//! assert_eq!(e.genus(), 1);
//!
//! // deg K = 2g − 2 = 0, and dim L(K) = g = 1.
//! let k = e.canonical_divisor().unwrap();
//! assert_eq!(k.degree(), Integer::from(0));
//! assert_eq!(riemann_roch(&k).unwrap().dimension(), 1);
//!
//! // dim L(3·∞) = 3 + 1 − 1 = 3, spanned by 1, x, y.
//! let d = Divisor::from_terms(e, [(Place::infinity(), Integer::from(3))]).unwrap();
//! assert_eq!(riemann_roch(&d).unwrap().dimension(), 3);
//! ```

mod classgroup;
mod divisor;
mod element;
mod error;
mod field;
#[cfg(test)]
mod proptests;
mod riemann_roch;
mod util;

pub use classgroup::DivisorClass;
pub use divisor::{Divisor, Place};
pub use element::FunctionFieldElement;
pub use error::FunctionFieldError;
pub use field::{FunctionField, Normalisation};
pub use riemann_roch::{riemann_roch, RiemannRochSpace};
