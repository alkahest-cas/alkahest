//! The registry entries for the orthogonal-polynomial families.
//!
//! One [`OrthoPrimitive`] per [`OrthoFamily`]; argument layout
//! `[n, params…, x]`.
//!
//! # Capability bundle
//!
//! * `diff_forward` — `∂/∂x` by the identities below (DLMF 18.9.15–18.9.20,
//!   14.10.5).  Each holds for a non-integer degree as well, through the
//!   hypergeometric representations, so a symbolic `n` is fine.  There is no
//!   closed form for the derivative in the degree or in a family parameter:
//!   when one of those depends on the differentiation variable the rule
//!   declines and `diff` refuses with `E-DIFF-001` rather than dropping the
//!   term.
//!
//!   | family | `d/dx` |
//!   |---|---|
//!   | `P_n` | `C_{n−1}^{(3/2)}(x)` |
//!   | `P_n^m` | `((n+m) P_{n−1}^m − n x P_n^m) / (1 − x²)` |
//!   | `H_n` | `2n H_{n−1}` |
//!   | `He_n` | `n He_{n−1}` |
//!   | `L_n^{(α)}` | `−L_{n−1}^{(α+1)}` |
//!   | `T_n` | `n U_{n−1}` |
//!   | `U_n` | `2 C_{n−1}^{(2)}` |
//!   | `C_n^{(a)}` | `2a C_{n−1}^{(a+1)}` |
//!   | `P_n^{(a,b)}` | `(n+a+b+1)/2 · P_{n−1}^{(a+1,b+1)}` |
//!
//! * `diff_reverse` — deliberately absent: one cotangent per argument would
//!   have to include the degree's, which does not exist.  Reverse mode still
//!   works through the generic fallback, which asks `diff_forward` for the
//!   partials of the arguments that actually depend on a variable.
//! * `numeric_f64` / `numeric_ball` — see [`super::numeric`].
//! * `simplify` — exact values at rational points for an integer degree, the
//!   degree-independent special values `P_ν(1) = 1`, `T_ν(1) = 1`,
//!   `U_ν(1) = ν + 1`, `L_ν(0) = 1`, and parity for an integer degree
//!   (`P_n(−x) = (−1)ⁿ P_n(x)`, …; Jacobi swaps `a ↔ b`).
//! * No Lean theorem, no LLVM lowering, no Taylor model.

use crate::ball::ArbBall;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::primitive::Primitive;

use super::{int_literal, mk_add, mk_mul, rational_literal, OrthoFamily};

/// Largest integer degree `simplify` expands at a rational point.  Past it
/// the node is left for numeric evaluation.
const MAX_SIMPLIFY_DEGREE: i64 = 200;

/// A registry primitive for one orthogonal-polynomial family.
pub struct OrthoPrimitive(pub OrthoFamily);

fn minus_one(pool: &ExprPool, n: ExprId) -> ExprId {
    match int_literal(pool, n) {
        Some(k) => pool.integer(k - 1u32),
        None => pool.add(vec![n, pool.integer(-1)]),
    }
}

fn plus_rational(pool: &ExprPool, a: ExprId, num: i64, den: i64) -> ExprId {
    match rational_literal(pool, a) {
        Some(q) => super::rat_node(pool, &(q + rug::Rational::from((num, den)))),
        None => pool.add(vec![a, pool.rational(num, den)]),
    }
}

/// `∂/∂x family(n, params…, x)`, with `n` and `params` constant.
pub(crate) fn local_derivative(
    family: OrthoFamily,
    n: ExprId,
    params: &[ExprId],
    x: ExprId,
    pool: &ExprPool,
) -> ExprId {
    if int_literal(pool, n).is_some_and(|k| k == 0) {
        return pool.integer(0);
    }
    let nm1 = minus_one(pool, n);
    let f = |fam: OrthoFamily, args: Vec<ExprId>| pool.func(fam.name(), args);
    match family {
        OrthoFamily::Legendre => f(OrthoFamily::Gegenbauer, vec![nm1, pool.rational(3, 2), x]),
        OrthoFamily::AssocLegendre => {
            let m = params[0];
            let n_plus_m = mk_add(pool, vec![n, m]);
            let p_nm1 = f(OrthoFamily::AssocLegendre, vec![nm1, m, x]);
            let p_n = f(OrthoFamily::AssocLegendre, vec![n, m, x]);
            let num = mk_add(
                pool,
                vec![
                    mk_mul(pool, vec![n_plus_m, p_nm1]),
                    mk_mul(pool, vec![pool.integer(-1), n, x, p_n]),
                ],
            );
            let one_minus_x2 = mk_add(
                pool,
                vec![
                    pool.integer(1),
                    mk_mul(pool, vec![pool.integer(-1), pool.pow(x, pool.integer(2))]),
                ],
            );
            mk_mul(pool, vec![num, pool.pow(one_minus_x2, pool.integer(-1))])
        }
        OrthoFamily::HermiteH => mk_mul(
            pool,
            vec![pool.integer(2), n, f(OrthoFamily::HermiteH, vec![nm1, x])],
        ),
        OrthoFamily::HermiteHe => mk_mul(pool, vec![n, f(OrthoFamily::HermiteHe, vec![nm1, x])]),
        OrthoFamily::Laguerre => mk_mul(
            pool,
            vec![
                pool.integer(-1),
                f(OrthoFamily::AssocLaguerre, vec![nm1, pool.integer(1), x]),
            ],
        ),
        OrthoFamily::AssocLaguerre => mk_mul(
            pool,
            vec![
                pool.integer(-1),
                f(
                    OrthoFamily::AssocLaguerre,
                    vec![nm1, plus_rational(pool, params[0], 1, 1), x],
                ),
            ],
        ),
        OrthoFamily::ChebyshevT => mk_mul(pool, vec![n, f(OrthoFamily::ChebyshevU, vec![nm1, x])]),
        OrthoFamily::ChebyshevU => mk_mul(
            pool,
            vec![
                pool.integer(2),
                f(OrthoFamily::Gegenbauer, vec![nm1, pool.integer(2), x]),
            ],
        ),
        OrthoFamily::Gegenbauer => {
            let a = params[0];
            mk_mul(
                pool,
                vec![
                    pool.integer(2),
                    a,
                    f(
                        OrthoFamily::Gegenbauer,
                        vec![nm1, plus_rational(pool, a, 1, 1), x],
                    ),
                ],
            )
        }
        OrthoFamily::Jacobi => {
            let (a, b) = (params[0], params[1]);
            let s = mk_add(pool, vec![n, a, b, pool.integer(1)]);
            mk_mul(
                pool,
                vec![
                    pool.rational(1, 2),
                    s,
                    f(
                        OrthoFamily::Jacobi,
                        vec![
                            nm1,
                            plus_rational(pool, a, 1, 1),
                            plus_rational(pool, b, 1, 1),
                            x,
                        ],
                    ),
                ],
            )
        }
    }
}

fn is_literal_zero(pool: &ExprPool, id: ExprId) -> bool {
    pool.with(id, |d| match d {
        ExprData::Integer(n) => n.0 == 0,
        ExprData::Rational(q) => q.0 == 0,
        _ => false,
    })
}

/// `y` when `x` is syntactically `−y` (a product with a literal `−1`
/// factor).
fn negated(pool: &ExprPool, x: ExprId) -> Option<ExprId> {
    let args = pool.with(x, |d| match d {
        ExprData::Mul(a) => Some(a.clone()),
        _ => None,
    })?;
    let minus = pool.integer(-1);
    let pos = args.iter().position(|&a| a == minus)?;
    let rest: Vec<ExprId> = args
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != pos)
        .map(|(_, &a)| a)
        .collect();
    Some(mk_mul(pool, rest))
}

impl OrthoPrimitive {
    fn split<'a>(&self, args: &'a [ExprId]) -> Option<(ExprId, &'a [ExprId], ExprId)> {
        if args.len() != self.0.arity() {
            return None;
        }
        let (n, rest) = args.split_first()?;
        let (x, params) = rest.split_last()?;
        Some((*n, params, *x))
    }
}

impl Primitive for OrthoPrimitive {
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        let a: Vec<String> = args.iter().map(|&a| pool.display(a).to_string()).collect();
        format!("{}({})", self.0.name(), a.join(", "))
    }

    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        let (n, params, x) = self.split(args)?;
        let fam = self.0;
        let n_int = int_literal(pool, n);

        // Exact value at a rational point (any parameters).
        if rational_literal(pool, x).is_some() {
            if let Some(k) = &n_int {
                if k.clone().abs() <= MAX_SIMPLIFY_DEGREE {
                    if let Ok(Some(v)) = super::explicit(fam, n, params, x, pool) {
                        return Some(v);
                    }
                }
            }
        }
        // Degree 0.
        if n_int.as_ref().is_some_and(|k| *k == 0) {
            return match fam {
                OrthoFamily::AssocLegendre => {
                    let m = int_literal(pool, params[0])?;
                    Some(pool.integer(if m == 0 { 1 } else { 0 }))
                }
                _ => Some(pool.integer(1)),
            };
        }
        // Degree-independent special values.
        let one = pool.integer(1);
        if x == one {
            match fam {
                OrthoFamily::Legendre | OrthoFamily::ChebyshevT => return Some(one),
                OrthoFamily::ChebyshevU => return Some(mk_add(pool, vec![n, one])),
                _ => {}
            }
        }
        if fam == OrthoFamily::Laguerre && is_literal_zero(pool, x) {
            return Some(one);
        }
        // Parity for an integer degree.
        let k = n_int?;
        let y = negated(pool, x)?;
        let odd = |e: &rug::Integer| e.is_odd();
        let (sign_odd, node) = match fam {
            OrthoFamily::Legendre
            | OrthoFamily::HermiteH
            | OrthoFamily::HermiteHe
            | OrthoFamily::ChebyshevT
            | OrthoFamily::ChebyshevU => (odd(&k), pool.func(fam.name(), vec![n, y])),
            OrthoFamily::Gegenbauer => (odd(&k), pool.func(fam.name(), vec![n, params[0], y])),
            OrthoFamily::AssocLegendre => {
                let m = int_literal(pool, params[0])?;
                (
                    odd(&rug::Integer::from(&k + &m)),
                    pool.func(fam.name(), vec![n, params[0], y]),
                )
            }
            OrthoFamily::Jacobi => (
                odd(&k),
                pool.func(fam.name(), vec![n, params[1], params[0], y]),
            ),
            OrthoFamily::Laguerre | OrthoFamily::AssocLaguerre => return None,
        };
        Some(if sign_odd {
            mk_mul(pool, vec![pool.integer(-1), node])
        } else {
            node
        })
    }

    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        let (n, params, x) = self.split(args)?;
        if crate::kernel::subs::mentions_var(n, wrt, pool)
            || params
                .iter()
                .any(|&p| crate::kernel::subs::mentions_var(p, wrt, pool))
        {
            // ∂/∂n and ∂/∂(parameter) have no closed form; decline rather
            // than drop the term.
            return None;
        }
        let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
        if is_literal_zero(pool, dx) {
            return Some(pool.integer(0));
        }
        let local = local_derivative(self.0, n, params, x, pool);
        Some(mk_mul(pool, vec![local, dx]))
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        super::numeric::eval_f64(self.0, args)
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        super::numeric::eval_ball(self.0, args)
    }
}
