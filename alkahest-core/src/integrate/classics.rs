//! Classical definite integrals whose antiderivatives are not elementary:
//! Catalan's constant and its relatives.
//!
//! `∫₀¹ atan(x)/x dx = G` is the defining integral of Catalan's constant: the
//! antiderivative is the inverse tangent integral `Ti₂(x) = Im Li₂(ix)`, which
//! is not elementary, so the FTC route refuses it (`E-INT-001`, "irreducible
//! product"). The same holds for the other integrals below — each is a
//! classical evaluation, not a computation this engine can perform, and none
//! of them comes out of `F(b) − F(a)`.
//!
//! # What is matched
//!
//! An entry is an integrand on an exact interval. The integrand is compared as
//! a multiset of `var`-dependent factors with integer exponents (so
//! `atan(x)/(x·(1+x²))`, `atan(x)·x⁻¹·(1+x²)⁻¹` and `(1/x)·atan(x)/(x²+1)` are
//! the same entry); `var`-free factors are a constant multiple carried into
//! the value. The interval must match exactly, or exactly reversed (which
//! negates). A sum is integrated term by term when at least one term is an
//! entry and every other term integrates on its own — linearity for
//! convergent pieces.
//!
//! Every value is checked against mpmath to 30 digits in the tests
//! (`tests/test_classic_refusals.py`), so an entry cannot drift.

use std::collections::BTreeMap;

use rug::Rational;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::subs::mentions_var;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::orthopoly::rational_literal;
use crate::simplify::engine::simplify;

use super::engine::IntegrationError;

fn simp(pool: &ExprPool, e: ExprId) -> ExprId {
    simplify(e, pool).value
}

/// `e` as `(var-free factors, var-dependent base → integer exponent)`.
type Factored = (Vec<ExprId>, BTreeMap<ExprId, Rational>);

fn decompose(e: ExprId, mult: &Rational, var: ExprId, pool: &ExprPool, out: &mut Factored) {
    if !mentions_var(e, var, pool) {
        out.0.push(if *mult == 1 {
            e
        } else {
            pool.pow(e, crate::orthopoly::rat_node(pool, mult))
        });
        return;
    }
    match pool.get(e) {
        ExprData::Mul(xs) => {
            for x in xs {
                decompose(x, mult, var, pool, out);
            }
        }
        ExprData::Pow { base, exp } => match rational_literal(pool, exp) {
            // Only integer exponents distribute over a product.
            Some(q) if *q.denom() == 1 => {
                decompose(base, &(mult.clone() * q), var, pool, out);
            }
            _ => {
                *out.1.entry(e).or_insert_with(|| Rational::from(0)) += mult;
            }
        },
        _ => {
            *out.1.entry(e).or_insert_with(|| Rational::from(0)) += mult;
        }
    }
}

fn factored(e: ExprId, var: ExprId, pool: &ExprPool) -> Factored {
    let mut out = (Vec::new(), BTreeMap::new());
    decompose(e, &Rational::from(1), var, pool, &mut out);
    out.1.retain(|_, q| *q != 0);
    out
}

struct Entry {
    integrand: ExprId,
    lower: ExprId,
    upper: ExprId,
    value: ExprId,
    name: &'static str,
}

/// The table, built in `pool` over `var` (all expressions simplified so they
/// are in the canonical form an input reaches).
fn table(var: ExprId, pool: &ExprPool) -> Vec<Entry> {
    let x = var;
    let int = |n: i64| pool.integer(n);
    let rat = |a: i64, b: i64| pool.rational(a, b);
    let pi = crate::eval::symbols::pi_symbol(pool);
    let g = crate::specfun::catalan(pool);
    let f = |name: &str, a: ExprId| pool.func(name, vec![a]);
    let mul = |v: Vec<ExprId>| pool.mul(v);
    let add = |v: Vec<ExprId>| pool.add(v);
    let pw = |b: ExprId, e: i64| pool.pow(b, int(e));
    let ln2 = f("log", int(2));
    let one_p_x2 = add(vec![int(1), pw(x, 2)]);
    let zero = int(0);
    let one = int(1);
    let inf = pool.pos_infinity();
    let pi_4 = mul(vec![rat(1, 4), pi]);
    let pi_2 = mul(vec![rat(1, 2), pi]);
    let raw = vec![
        // The defining integral of G (Ti₂(1)).
        (
            "catalan_atan_over_x",
            mul(vec![f("atan", x), pw(x, -1)]),
            zero,
            one,
            g,
        ),
        // And its x → 1/x image on [1, ∞).
        (
            "catalan_log_over_one_plus_x2",
            mul(vec![f("log", x), pw(one_p_x2, -1)]),
            zero,
            one,
            mul(vec![int(-1), g]),
        ),
        (
            "catalan_log_over_one_plus_x2",
            mul(vec![f("log", x), pw(one_p_x2, -1)]),
            one,
            inf,
            g,
        ),
        (
            "log_over_one_plus_x2_half_line",
            mul(vec![f("log", x), pw(one_p_x2, -1)]),
            zero,
            inf,
            zero,
        ),
        // x = tan t: ∫₀^{π/2} t·cot t dt = (π/2) ln 2.
        (
            "atan_over_x_one_plus_x2",
            mul(vec![f("atan", x), pw(x, -1), pw(one_p_x2, -1)]),
            zero,
            inf,
            mul(vec![pi_2, ln2]),
        ),
        // (π/2) ln 2 − G on [0, 1]; π ln 2 on [0, ∞).
        (
            "log_one_plus_x2_over_one_plus_x2",
            mul(vec![f("log", one_p_x2), pw(one_p_x2, -1)]),
            zero,
            one,
            add(vec![mul(vec![pi_2, ln2]), mul(vec![int(-1), g])]),
        ),
        (
            "log_one_plus_x2_over_one_plus_x2",
            mul(vec![f("log", one_p_x2), pw(one_p_x2, -1)]),
            zero,
            inf,
            mul(vec![pi, ln2]),
        ),
        // Serret's integral: (π/8) ln 2.
        (
            "serret",
            mul(vec![f("log", add(vec![one, x])), pw(one_p_x2, -1)]),
            zero,
            one,
            mul(vec![rat(1, 8), pi, ln2]),
        ),
        (
            "serret_atan_form",
            mul(vec![f("atan", x), pw(add(vec![one, x]), -1)]),
            zero,
            one,
            mul(vec![rat(1, 8), pi, ln2]),
        ),
        (
            "serret_trig_form",
            f("log", add(vec![one, f("tan", x)])),
            zero,
            pi_4,
            mul(vec![rat(1, 8), pi, ln2]),
        ),
        // ∫₀^{π/4} ln tan x = −G.
        (
            "catalan_log_tan",
            f("log", f("tan", x)),
            zero,
            pi_4,
            mul(vec![int(-1), g]),
        ),
        // ∫₀^{π/2} x / sin x = 2G.
        (
            "catalan_x_over_sin",
            mul(vec![x, pw(f("sin", x), -1)]),
            zero,
            pi_2,
            mul(vec![int(2), g]),
        ),
        // Euler's log-sine integral.
        (
            "euler_log_sin",
            f("log", f("sin", x)),
            zero,
            pi_2,
            mul(vec![rat(-1, 2), pi, ln2]),
        ),
        (
            "euler_log_cos",
            f("log", f("cos", x)),
            zero,
            pi_2,
            mul(vec![rat(-1, 2), pi, ln2]),
        ),
        // On [0, π/4]: −(π/4) ln 2 ∓ G/2.
        (
            "log_sin_quarter",
            f("log", f("sin", x)),
            zero,
            pi_4,
            add(vec![
                mul(vec![rat(-1, 4), pi, ln2]),
                mul(vec![rat(-1, 2), g]),
            ]),
        ),
        (
            "log_cos_quarter",
            f("log", f("cos", x)),
            zero,
            pi_4,
            add(vec![
                mul(vec![rat(-1, 4), pi, ln2]),
                mul(vec![rat(1, 2), g]),
            ]),
        ),
        // ∫₀^∞ x / cosh x = 2G.
        (
            "catalan_x_over_cosh",
            mul(vec![x, pw(f("cosh", x), -1)]),
            zero,
            inf,
            mul(vec![int(2), g]),
        ),
        // Differentiate under the integral in a: ∫₀¹ atan(x)/(x√(1−x²)) = (π/2) ln(1 + √2).
        (
            "atan_over_x_sqrt_one_minus_x2",
            mul(vec![
                f("atan", x),
                pw(x, -1),
                pw(f("sqrt", add(vec![one, mul(vec![int(-1), pw(x, 2)])])), -1),
            ]),
            zero,
            one,
            mul(vec![pi_2, f("log", add(vec![one, f("sqrt", int(2))]))]),
        ),
    ];
    raw.into_iter()
        .map(|(name, i, a, b, v)| Entry {
            integrand: simp(pool, i),
            lower: simp(pool, a),
            upper: simp(pool, b),
            value: simp(pool, v),
            name,
        })
        .collect()
}

/// `c` with `expr = c · entry` and `c` free of `var`, or `None`.
fn constant_multiple(
    expr: &Factored,
    entry: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let (entry_consts, entry_dep) = factored(entry, var, pool);
    if expr.1 != entry_dep {
        return None;
    }
    let mut c = expr.0.clone();
    for k in entry_consts {
        c.push(pool.pow(k, pool.integer(-1)));
    }
    Some(match c.len() {
        0 => pool.integer(1),
        1 => c[0],
        _ => pool.mul(c),
    })
}

/// The value of one table integral, or `None`.
fn lookup(
    expr: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, &'static str)> {
    let e = simp(pool, expr);
    let fx = factored(e, var, pool);
    if fx.1.is_empty() {
        return None;
    }
    for entry in table(var, pool) {
        let sign = if lower == entry.lower && upper == entry.upper {
            1
        } else if lower == entry.upper && upper == entry.lower {
            -1
        } else {
            continue;
        };
        if let Some(c) = constant_multiple(&fx, entry.integrand, var, pool) {
            let v = simp(pool, pool.mul(vec![pool.integer(sign), c, entry.value]));
            return Some((v, entry.name));
        }
    }
    None
}

/// The table route of [`super::engine::integrate_definite`]: a table integral,
/// a constant multiple of one, or a sum with at least one table term whose
/// other terms integrate on their own. `None` when no table entry is involved.
pub(super) fn try_definite(
    expr: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    if let Some((v, name)) = lookup(expr, var, lower, upper, pool) {
        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple(name, expr, v));
        return Some(Ok(DerivedExpr::with_log(v, log)));
    }
    let terms = match pool.get(simp(pool, expr)) {
        ExprData::Add(ts) => ts,
        _ => return None,
    };
    if !terms
        .iter()
        .any(|&t| lookup(t, var, lower, upper, pool).is_some())
    {
        return None;
    }
    let mut values = Vec::with_capacity(terms.len());
    let mut log = DerivationLog::new();
    for t in terms {
        match super::engine::integrate_definite(t, var, lower, upper, pool) {
            Ok(d) => {
                log = log.merge(d.log);
                values.push(d.value);
            }
            Err(e) => return Some(Err(e)),
        }
    }
    let v = simp(pool, pool.add(values));
    log.push(RewriteStep::simple("definite_linearity", expr, v));
    Some(Ok(DerivedExpr::with_log(v, log)))
}
