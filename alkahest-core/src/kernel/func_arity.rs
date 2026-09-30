//! Argument counts of the built-in functions.
//!
//! `ExprData::Func { name, args }` is variadic in the kernel: a user function
//! `f(x, y, z)` is as legal as `sin(x)`.  The *built-in* names, though, mean
//! one thing each, and nearly every subsystem that recognises one indexes its
//! arguments directly (`args[0]` for `sin`, `args[2]` for `EllipticPi`).  A
//! `sin()` or an `EllipticPi(x)` node is therefore not an expression at all —
//! it is a panic waiting for the first printer, differentiator or integrator
//! to look at it.
//!
//! [`known_func_arity`] is the single table of those counts.  The checked
//! constructor [`ExprPool::try_func`](crate::kernel::ExprPool::try_func), the
//! Python `pool.func` and both parsers refuse a built-in name at the wrong
//! arity with [`FuncArityError`] (`E-POOL-002`); the renderers and the
//! primitive registry consult the same table so that a node which reached the
//! pool some other way (the unchecked [`ExprPool::func`](crate::kernel::ExprPool::func), `intern`, a pool
//! file) degrades to a generic rendering or a declined operation rather than
//! a panic.

use std::fmt;

use crate::errors::AlkahestError;

/// `(min, max)` argument count for a built-in function name, or `None` for a
/// name the kernel attaches no meaning to (user functions such as `f`, and
/// the handful of internal markers).  `max == usize::MAX` means "no upper
/// bound".
pub fn known_func_arity(name: &str) -> Option<(usize, usize)> {
    let one = Some((1, 1));
    match name {
        // Elementary.
        "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "asin" | "acos" | "atan" | "asinh"
        | "acosh" | "atanh" | "exp" | "log" | "ln" | "sqrt" => one,
        // Piecewise-elementary and complex parts.
        "abs" | "sign" | "floor" | "ceil" | "round" | "heaviside" | "diracdelta" | "conjugate"
        | "re" | "im" | "arg" => one,
        // Special functions.
        "erf" | "erfc" | "gamma" | "digamma" | "trigamma" | "lambert_w" | "bessel_j0"
        | "bessel_j1" | "EllipticK" | "Ei" | "li" | "Si" | "Ci" | "Shi" | "Chi" | "fresnels"
        | "fresnelc" | "dilog" => one,
        // Complete (1 argument) or incomplete (2 arguments) second kind.
        "EllipticE" => Some((1, 2)),
        "EllipticF" | "atan2" => Some((2, 2)),
        "EllipticPi" => Some((3, 3)),
        "min" | "max" => Some((2, usize::MAX)),
        _ => None,
    }
}

/// Whether `name(args…)` with `n` arguments has the arity its name requires.
/// Always `true` for a name outside [`known_func_arity`]'s table.
pub fn func_arity_ok(name: &str, n: usize) -> bool {
    match known_func_arity(name) {
        Some((lo, hi)) => (lo..=hi).contains(&n),
        None => true,
    }
}

/// A built-in function applied to the wrong number of arguments.
///
/// Code `E-POOL-002`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuncArityError {
    name: String,
    got: usize,
    min: usize,
    max: usize,
}

impl FuncArityError {
    /// Check `name` against [`known_func_arity`]; `Err` on a mismatch.
    pub fn check(name: &str, got: usize) -> Result<(), FuncArityError> {
        match known_func_arity(name) {
            Some((min, max)) if !(min..=max).contains(&got) => Err(FuncArityError {
                name: name.to_string(),
                got,
                min,
                max,
            }),
            _ => Ok(()),
        }
    }

    /// The function name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// How many arguments were supplied.
    pub fn got(&self) -> usize {
        self.got
    }

    /// The accepted range, `(min, max)`; `max == usize::MAX` means unbounded.
    pub fn expected(&self) -> (usize, usize) {
        (self.min, self.max)
    }

    /// `"1"`, `"1 or 2"`, `"at least 2"` — the accepted counts in words.
    pub fn expected_text(&self) -> String {
        if self.max == usize::MAX {
            format!("at least {}", self.min)
        } else if self.min == self.max {
            format!("{}", self.min)
        } else if self.max == self.min + 1 {
            format!("{} or {}", self.min, self.max)
        } else {
            format!("{} to {}", self.min, self.max)
        }
    }
}

impl fmt::Display for FuncArityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plural = if self.min == 1 && self.max == 1 {
            "argument"
        } else {
            "arguments"
        };
        write!(
            f,
            "{} takes {} {plural}, got {}",
            self.name,
            self.expected_text(),
            self.got
        )
    }
}

impl std::error::Error for FuncArityError {}

impl AlkahestError for FuncArityError {
    fn code(&self) -> &'static str {
        "E-POOL-002"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "a built-in function name has a fixed number of arguments; \
             use a different name for a user-defined function of another arity",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_covers_the_registered_primitives() {
        let reg = crate::primitive::PrimitiveRegistry::dispatch_registry();
        for row in reg.coverage_report().rows {
            assert!(
                known_func_arity(&row.name).is_some(),
                "registered primitive `{}` has no arity entry",
                row.name
            );
        }
    }

    #[test]
    fn check_accepts_and_refuses() {
        assert!(FuncArityError::check("sin", 1).is_ok());
        assert!(FuncArityError::check("f", 0).is_ok());
        assert!(FuncArityError::check("f", 7).is_ok());
        assert!(FuncArityError::check("EllipticE", 2).is_ok());
        assert!(FuncArityError::check("min", 5).is_ok());
        let e = FuncArityError::check("sqrt", 0).unwrap_err();
        assert_eq!(e.code(), "E-POOL-002");
        assert_eq!(e.to_string(), "sqrt takes 1 argument, got 0");
        let e = FuncArityError::check("EllipticE", 3).unwrap_err();
        assert_eq!(e.to_string(), "EllipticE takes 1 or 2 arguments, got 3");
        let e = FuncArityError::check("max", 1).unwrap_err();
        assert_eq!(e.to_string(), "max takes at least 2 arguments, got 1");
    }
}

/// Every public operation must survive a wrong-arity built-in node without
/// panicking, however it got into the pool: `ExprPool::func` and `intern` are
/// unchecked, and a pool file is whatever is on disk.
#[cfg(test)]
mod wrong_arity_sweep {
    use super::known_func_arity;
    use crate::deriv::log::DerivedExpr;
    use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
    use std::collections::HashMap;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    const NAMES: &[&str] = &[
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "asin",
        "acos",
        "atan",
        "asinh",
        "acosh",
        "atanh",
        "exp",
        "log",
        "ln",
        "sqrt",
        "abs",
        "sign",
        "floor",
        "ceil",
        "round",
        "heaviside",
        "diracdelta",
        "conjugate",
        "re",
        "im",
        "arg",
        "erf",
        "erfc",
        "gamma",
        "digamma",
        "trigamma",
        "lambert_w",
        "bessel_j0",
        "bessel_j1",
        "EllipticK",
        "EllipticE",
        "EllipticF",
        "EllipticPi",
        "Ei",
        "li",
        "Si",
        "Ci",
        "Shi",
        "Chi",
        "fresnels",
        "fresnelc",
        "dilog",
        "atan2",
        "min",
        "max",
    ];

    type Op<'a> = (&'static str, Box<dyn Fn() + 'a>);

    fn ops(pool: &ExprPool, e: ExprId, x: ExprId) -> Vec<Op<'_>> {
        let zero = pool.integer(0_i32);
        vec![
            ("str", Box::new(move || drop(pool.display(e).to_string()))),
            (
                "latex",
                Box::new(move || drop(crate::kernel::render_latex(e, pool))),
            ),
            (
                "unicode",
                Box::new(move || drop(crate::kernel::render_unicode(e, pool))),
            ),
            (
                "diff",
                Box::new(move || drop(crate::diff::diff(e, x, pool))),
            ),
            (
                "grad",
                Box::new(move || drop(crate::diff::grad(e, &[x], pool))),
            ),
            (
                "simplify",
                Box::new(move || drop(crate::simplify::simplify(e, pool))),
            ),
            (
                "simplify_egraph",
                Box::new(move || drop(crate::simplify::simplify_egraph(e, pool))),
            ),
            (
                "integrate",
                Box::new(move || drop(crate::integrate::integrate(e, x, pool))),
            ),
            (
                "series",
                Box::new(move || drop(crate::calculus::series(e, x, zero, 4, pool))),
            ),
            (
                "limit",
                Box::new(move || {
                    drop(crate::calculus::limit(
                        e,
                        x,
                        zero,
                        crate::calculus::LimitDirection::Plus,
                        pool,
                    ))
                }),
            ),
            (
                "to_lean",
                Box::new(move || {
                    drop(crate::lean::emit_lean_expr(&DerivedExpr::new(e), pool));
                    if let Ok(d) = crate::diff::diff(e, x, pool) {
                        drop(crate::lean::emit_lean_expr(&d, pool));
                    }
                }),
            ),
            (
                "eval",
                Box::new(move || {
                    let env: HashMap<ExprId, f64> = [(x, 0.5)].into_iter().collect();
                    drop(crate::eval::eval_f64(e, pool, &env));
                    let _ = crate::jit::eval_interp(e, &env, pool);
                }),
            ),
        ]
    }

    #[test]
    fn no_operation_panics_on_a_wrong_arity_builtin() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut failures = Vec::new();
        for &name in NAMES {
            let (lo, hi) = known_func_arity(name).expect(name);
            for n in 0..=4usize {
                if (lo..=hi).contains(&n) {
                    continue;
                }
                let e = pool.intern(ExprData::Func {
                    name: name.to_string(),
                    args: vec![x; n],
                });
                // Bare, and as a subterm of an ordinary expression.
                let wrapped = pool.add(vec![pool.mul(vec![pool.integer(2_i32), e]), x]);
                for target in [e, wrapped] {
                    for (op, run) in ops(&pool, target, x) {
                        if catch_unwind(AssertUnwindSafe(run)).is_err() {
                            failures.push(format!("{op}({name}/{n})"));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "panicked: {}", failures.join(", "));
    }
}
