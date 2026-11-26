//! Reflective *kernel* certificates for indefinite integrals.
//!
//! Instead of mapping each rule to a Lean tactic (the [`super::hasderiv`]
//! emitter), the certificate is *data* decided by a verified checker in the
//! repository's Lean library `Alkahest.Kernel` (`lean/Alkahest/Kernel/`):
//!
//! ```text
//! import Alkahest.Kernel
//! open Alkahest.Kernel in
//! theorem alkahest_integral (x : ℝ) (h1 : x ≠ 0) :
//!     HasDerivAt (fun (x : ℝ) => Real.log x) (x⁻¹) x :=
//!   integral_sound (H := [.ne (.var 0)]) (F := .app .log (.var 0)) (f := .inv (.var 0))
//!     (w := ⟨[atoms], [relations], [cofactors]⟩) (by decide +kernel) [] x ⟨h1, trivial⟩
//! ```
//!
//! `integral_sound` is proved once in Lean: if `checkIntegral H F f w`
//! evaluates to `true`, then `F' = f` wherever `H` holds. The checker computes
//! the symbolic derivative `D 0 F` itself, decides the differentiability side
//! conditions from `H`, and checks `D 0 F - f = Σ cᵢ·(Lᵢ - Rᵢ)` as a polynomial
//! identity in atoms, where each `Lᵢ = Rᵢ` is a relation proved in the library.
//!
//! This module mirrors the Lean checker *exactly* (same `D`, same normal form,
//! same sign rules) so it can compute the atom table, relations and cofactors
//! the checker will accept, and it re-verifies the identity before emitting.
//! Anything it cannot encode is a withheld reason; the caller then falls back
//! to the tactic emitter. Nothing here runs unless a certificate is requested.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::rc::Rc;

use rug::{Integer, Rational};

use super::hasderiv::{
    exp_of, reduce_power, respell_rational_powers, single_var, special_function_reason, substitute,
    IntegrationCertWithheld, Mono, Poly, MAX_ROUNDS, WITHHELD_CONSTANT_SIGN, WITHHELD_EMPTY_DOMAIN,
    WITHHELD_IDENTITY, WITHHELD_ROOTSUM, WITHHELD_TOO_LARGE, WITHHELD_UNSUPPORTED_NODE,
};
use crate::kernel::{ExprData, ExprId, ExprPool};

/// How the kernel certificate's checker is run by Lean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelCheckMode {
    /// `by decide +kernel`: the Lean kernel evaluates the checker. The trusted
    /// base is Lean's kernel plus the `Alkahest.Kernel` soundness proofs.
    Kernel,
    /// `by native_decide`: the checker is compiled and run natively. Faster on
    /// huge certificates, but adds the compiler and `Lean.ofReduceBool` to the
    /// trusted base. Opt-in only.
    Native,
}

/// Kernel certificates larger than this are withheld (the tactic path may
/// still certify them).
const MAX_KERNEL_SRC_BYTES: usize = 2_000_000;

// ===========================================================================
// The AST (mirror of `Alkahest.Kernel.AExpr`)
// ===========================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
enum KFn {
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Sqrt,
    Atan,
    Asin,
    Acos,
    Sinh,
    Cosh,
    Tanh,
    Arsinh,
    Logabs,
}

impl KFn {
    fn from_name(name: &str) -> Option<KFn> {
        Some(match name {
            "exp" => KFn::Exp,
            "log" => KFn::Log,
            "sin" => KFn::Sin,
            "cos" => KFn::Cos,
            "tan" => KFn::Tan,
            "sqrt" => KFn::Sqrt,
            "atan" => KFn::Atan,
            "asin" => KFn::Asin,
            "acos" => KFn::Acos,
            "sinh" => KFn::Sinh,
            "cosh" => KFn::Cosh,
            "tanh" => KFn::Tanh,
            "asinh" => KFn::Arsinh,
            _ => return None,
        })
    }
    fn ctor(self) -> &'static str {
        match self {
            KFn::Exp => "exp",
            KFn::Log => "log",
            KFn::Sin => "sin",
            KFn::Cos => "cos",
            KFn::Tan => "tan",
            KFn::Sqrt => "sqrt",
            KFn::Atan => "atan",
            KFn::Asin => "asin",
            KFn::Acos => "acos",
            KFn::Sinh => "sinh",
            KFn::Cosh => "cosh",
            KFn::Tanh => "tanh",
            KFn::Arsinh => "arsinh",
            KFn::Logabs => "logabs",
        }
    }
    fn lean(self) -> &'static str {
        match self {
            KFn::Exp => "Real.exp",
            KFn::Log | KFn::Logabs => "Real.log",
            KFn::Sin => "Real.sin",
            KFn::Cos => "Real.cos",
            KFn::Tan => "Real.tan",
            KFn::Sqrt => "Real.sqrt",
            KFn::Atan => "Real.arctan",
            KFn::Asin => "Real.arcsin",
            KFn::Acos => "Real.arccos",
            KFn::Sinh => "Real.sinh",
            KFn::Cosh => "Real.cosh",
            KFn::Tanh => "Real.tanh",
            KFn::Arsinh => "Real.arsinh",
        }
    }
    /// `canonFn`: `log |a|` and `log a` are one atom.
    fn canon(self) -> KFn {
        if self == KFn::Logabs {
            KFn::Log
        } else {
            self
        }
    }
    fn eval(self, t: f64) -> f64 {
        match self {
            KFn::Exp => t.exp(),
            KFn::Log => {
                if t == 0.0 {
                    0.0
                } else {
                    t.abs().ln()
                }
            }
            KFn::Logabs => {
                if t == 0.0 {
                    0.0
                } else {
                    t.abs().ln()
                }
            }
            KFn::Sin => t.sin(),
            KFn::Cos => t.cos(),
            KFn::Tan => t.tan(),
            KFn::Sqrt => {
                if t <= 0.0 {
                    0.0
                } else {
                    t.sqrt()
                }
            }
            KFn::Atan => t.atan(),
            KFn::Asin => t.clamp(-1.0, 1.0).asin(),
            KFn::Acos => t.clamp(-1.0, 1.0).acos(),
            KFn::Sinh => t.sinh(),
            KFn::Cosh => t.cosh(),
            KFn::Tanh => t.tanh(),
            KFn::Arsinh => t.asinh(),
        }
    }
}

type R = Rc<AE>;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum AE {
    Var(u32),
    Zero,
    One,
    Lit(Integer),
    Pi,
    Add(R, R),
    Sub(R, R),
    Mul(R, R),
    Div(R, R),
    Neg(R),
    Inv(R),
    Pow(R, u32),
    App(KFn, R),
}

fn rc(e: AE) -> R {
    Rc::new(e)
}

/// `AExpr.ofNat`.
fn of_nat(n: &Integer) -> AE {
    if *n == 0 {
        AE::Zero
    } else if *n == 1 {
        AE::One
    } else {
        AE::Lit(n.clone())
    }
}

impl AE {
    fn has_var(&self) -> bool {
        match self {
            AE::Var(_) => true,
            AE::Zero | AE::One | AE::Lit(_) | AE::Pi => false,
            AE::Add(a, b) | AE::Sub(a, b) | AE::Mul(a, b) | AE::Div(a, b) => {
                a.has_var() || b.has_var()
            }
            AE::Neg(a) | AE::Inv(a) | AE::Pow(a, _) | AE::App(_, a) => a.has_var(),
        }
    }

    /// Constructor syntax (`.add (.var 0) .one`).
    fn data(&self, out: &mut String) {
        match self {
            AE::Var(i) => {
                let _ = write!(out, "(.var (nat_lit {i}))");
            }
            AE::Zero => out.push_str(".zero"),
            AE::One => out.push_str(".one"),
            AE::Lit(n) => {
                let _ = write!(out, "(.lit (nat_lit {n}))");
            }
            AE::Pi => out.push_str(".pi"),
            AE::Add(a, b) | AE::Sub(a, b) | AE::Mul(a, b) | AE::Div(a, b) => {
                let c = match self {
                    AE::Add(..) => "add",
                    AE::Sub(..) => "sub",
                    AE::Mul(..) => "mul",
                    _ => "div",
                };
                let _ = write!(out, "(.{c} ");
                a.data(out);
                out.push(' ');
                b.data(out);
                out.push(')');
            }
            AE::Neg(a) | AE::Inv(a) => {
                out.push_str(if matches!(self, AE::Neg(_)) {
                    "(.neg "
                } else {
                    "(.inv "
                });
                a.data(out);
                out.push(')');
            }
            AE::Pow(a, n) => {
                out.push_str("(.pow ");
                a.data(out);
                let _ = write!(out, " (nat_lit {n}))");
            }
            AE::App(f, a) => {
                let _ = write!(out, "(.app .{} ", f.ctor());
                a.data(out);
                out.push(')');
            }
        }
    }

    fn data_string(&self) -> String {
        let mut s = String::new();
        self.data(&mut s);
        s
    }

    /// The ordinary Lean spelling; `AExpr.eval` unfolds to exactly this term.
    fn lean(&self, names: &[String], out: &mut String) {
        match self {
            AE::Var(i) => {
                let _ = write!(out, "({} : ℝ)", names[*i as usize]);
            }
            AE::Zero => out.push_str("(0 : ℝ)"),
            AE::One => out.push_str("(1 : ℝ)"),
            AE::Lit(n) => {
                let _ = write!(out, "({n} : ℝ)");
            }
            AE::Pi => out.push_str("Real.pi"),
            AE::Add(a, b) | AE::Sub(a, b) | AE::Mul(a, b) | AE::Div(a, b) => {
                let op = match self {
                    AE::Add(..) => " + ",
                    AE::Sub(..) => " - ",
                    AE::Mul(..) => " * ",
                    _ => " / ",
                };
                out.push('(');
                a.lean(names, out);
                out.push_str(op);
                b.lean(names, out);
                out.push(')');
            }
            AE::Neg(a) => {
                out.push_str("(-");
                a.lean(names, out);
                out.push(')');
            }
            AE::Inv(a) => {
                out.push('(');
                a.lean(names, out);
                out.push_str(")⁻¹");
            }
            AE::Pow(a, n) => {
                out.push('(');
                a.lean(names, out);
                let _ = write!(out, " ^ ({n} : ℕ))");
            }
            AE::App(KFn::Logabs, a) => {
                out.push_str("Real.log |");
                a.lean(names, out);
                out.push('|');
            }
            AE::App(f, a) => {
                out.push_str(f.lean());
                out.push_str(" (");
                a.lean(names, out);
                out.push(')');
            }
        }
    }

    fn lean_string(&self, names: &[String]) -> String {
        let mut s = String::new();
        self.lean(names, &mut s);
        s
    }

    fn eval(&self, env: &[f64]) -> f64 {
        match self {
            AE::Var(i) => env.get(*i as usize).copied().unwrap_or(0.0),
            AE::Zero => 0.0,
            AE::One => 1.0,
            AE::Lit(n) => n.to_f64(),
            AE::Pi => std::f64::consts::PI,
            AE::Add(a, b) => a.eval(env) + b.eval(env),
            AE::Sub(a, b) => a.eval(env) - b.eval(env),
            AE::Mul(a, b) => a.eval(env) * b.eval(env),
            AE::Div(a, b) => {
                let d = b.eval(env);
                if d == 0.0 {
                    0.0
                } else {
                    a.eval(env) / d
                }
            }
            AE::Neg(a) => -a.eval(env),
            AE::Inv(a) => {
                let d = a.eval(env);
                if d == 0.0 {
                    0.0
                } else {
                    1.0 / d
                }
            }
            AE::Pow(a, n) => a.eval(env).powi(*n as i32),
            AE::App(f, a) => f.eval(a.eval(env)),
        }
    }
}

// ===========================================================================
// D, dconds (mirror of `Alkahest.Kernel.Deriv`)
// ===========================================================================

fn dfn(f: KFn, a: &R) -> AE {
    let one = || rc(AE::One);
    let sq = |e: R| rc(AE::Pow(e, 2));
    match f {
        KFn::Exp => AE::App(KFn::Exp, a.clone()),
        KFn::Log | KFn::Logabs => AE::Inv(a.clone()),
        KFn::Sin => AE::App(KFn::Cos, a.clone()),
        KFn::Cos => AE::Neg(rc(AE::App(KFn::Sin, a.clone()))),
        KFn::Tan => AE::Inv(sq(rc(AE::App(KFn::Cos, a.clone())))),
        KFn::Sqrt => AE::Inv(rc(AE::Mul(
            rc(AE::Lit(Integer::from(2))),
            rc(AE::App(KFn::Sqrt, a.clone())),
        ))),
        KFn::Atan => AE::Inv(rc(AE::Add(one(), sq(a.clone())))),
        KFn::Asin => AE::Inv(rc(AE::App(KFn::Sqrt, rc(AE::Sub(one(), sq(a.clone())))))),
        KFn::Acos => AE::Neg(rc(AE::Inv(rc(AE::App(
            KFn::Sqrt,
            rc(AE::Sub(one(), sq(a.clone()))),
        ))))),
        KFn::Sinh => AE::App(KFn::Cosh, a.clone()),
        KFn::Cosh => AE::App(KFn::Sinh, a.clone()),
        KFn::Tanh => AE::Inv(sq(rc(AE::App(KFn::Cosh, a.clone())))),
        KFn::Arsinh => AE::Inv(rc(AE::App(KFn::Sqrt, rc(AE::Add(one(), sq(a.clone())))))),
    }
}

/// `D i e`.
fn deriv(e: &R, i: u32) -> R {
    rc(match &**e {
        AE::Var(j) => {
            if *j == i {
                AE::One
            } else {
                AE::Zero
            }
        }
        AE::Zero | AE::One | AE::Lit(_) | AE::Pi => AE::Zero,
        AE::Add(a, b) => AE::Add(deriv(a, i), deriv(b, i)),
        AE::Sub(a, b) => AE::Sub(deriv(a, i), deriv(b, i)),
        AE::Mul(a, b) => AE::Add(
            rc(AE::Mul(deriv(a, i), b.clone())),
            rc(AE::Mul(a.clone(), deriv(b, i))),
        ),
        AE::Div(a, b) => AE::Div(
            rc(AE::Sub(
                rc(AE::Mul(deriv(a, i), b.clone())),
                rc(AE::Mul(a.clone(), deriv(b, i))),
            )),
            rc(AE::Pow(b.clone(), 2)),
        ),
        AE::Neg(a) => AE::Neg(deriv(a, i)),
        AE::Inv(a) => AE::Div(rc(AE::Neg(deriv(a, i))), rc(AE::Pow(a.clone(), 2))),
        AE::Pow(a, n) => AE::Mul(
            rc(AE::Mul(
                rc(of_nat(&Integer::from(*n))),
                rc(AE::Pow(a.clone(), n.saturating_sub(1))),
            )),
            deriv(a, i),
        ),
        AE::App(f, a) => AE::Mul(rc(dfn(*f, a)), deriv(a, i)),
    })
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Cond {
    Ne(R),
    Pos(R),
    Nonneg(R),
    Lt(R, R),
}

impl Cond {
    fn data(&self, out: &mut String) {
        match self {
            Cond::Ne(e) | Cond::Pos(e) | Cond::Nonneg(e) => {
                out.push_str(match self {
                    Cond::Ne(_) => "(.ne ",
                    Cond::Pos(_) => "(.pos ",
                    _ => "(.nonneg ",
                });
                e.data(out);
                out.push(')');
            }
            Cond::Lt(a, b) => {
                out.push_str("(.lt ");
                a.data(out);
                out.push(' ');
                b.data(out);
                out.push(')');
            }
        }
    }
    /// Spelling defeq to `Cond.holds`.
    fn lean(&self, names: &[String]) -> String {
        match self {
            Cond::Ne(e) => format!("{} ≠ 0", e.lean_string(names)),
            Cond::Pos(e) => format!("0 < {}", e.lean_string(names)),
            Cond::Nonneg(e) => format!("0 ≤ {}", e.lean_string(names)),
            Cond::Lt(a, b) => {
                // `-1 < u` / `u < 1` (the `arcsin` domain): the other side fixes
                // the type, and `-1` / `1` elaborate to `.neg .one` / `.one`.
                let side = |e: &R| match &**e {
                    AE::One => "1".to_string(),
                    AE::Neg(o) if **o == AE::One => "-1".to_string(),
                    _ => e.lean_string(names),
                };
                format!("{} < {}", side(a), side(b))
            }
        }
    }
    fn has_var(&self) -> bool {
        match self {
            Cond::Ne(e) | Cond::Pos(e) | Cond::Nonneg(e) => e.has_var(),
            Cond::Lt(a, b) => a.has_var() || b.has_var(),
        }
    }
    fn holds(&self, env: &[f64]) -> bool {
        match self {
            Cond::Ne(e) => {
                let v = e.eval(env);
                v.is_finite() && v.abs() > 1e-12
            }
            Cond::Pos(e) => {
                let v = e.eval(env);
                v.is_finite() && v > 1e-12
            }
            Cond::Nonneg(e) => {
                let v = e.eval(env);
                v.is_finite() && v >= 0.0
            }
            Cond::Lt(a, b) => {
                let (u, v) = (a.eval(env), b.eval(env));
                u.is_finite() && v.is_finite() && u + 1e-12 < v
            }
        }
    }
}

fn fconds(f: KFn, a: &R) -> Vec<Cond> {
    let m1 = || rc(AE::Neg(rc(AE::One)));
    match f {
        KFn::Log | KFn::Logabs => vec![Cond::Ne(a.clone())],
        KFn::Tan => vec![Cond::Ne(rc(AE::App(KFn::Cos, a.clone())))],
        KFn::Sqrt => vec![Cond::Pos(a.clone())],
        KFn::Asin | KFn::Acos => vec![Cond::Lt(m1(), a.clone()), Cond::Lt(a.clone(), rc(AE::One))],
        _ => vec![],
    }
}

fn dconds(e: &R, out: &mut Vec<Cond>) {
    match &**e {
        AE::Add(a, b) | AE::Sub(a, b) | AE::Mul(a, b) => {
            dconds(a, out);
            dconds(b, out);
        }
        AE::Div(a, b) => {
            dconds(a, out);
            dconds(b, out);
            out.push(Cond::Ne(b.clone()));
        }
        AE::Neg(a) | AE::Pow(a, _) => dconds(a, out),
        AE::Inv(a) => {
            dconds(a, out);
            out.push(Cond::Ne(a.clone()));
        }
        AE::App(f, a) => {
            dconds(a, out);
            out.extend(fconds(*f, a));
        }
        _ => {}
    }
}

// ===========================================================================
// Sign checker (mirror of `Alkahest.Kernel.Sign`)
// ===========================================================================

#[derive(Clone, Copy)]
struct Sign {
    pos: bool,
    nonneg: bool,
    ne: bool,
}

fn close(h: &[Cond], e: &R, s: Sign) -> Sign {
    let p = s.pos || h.contains(&Cond::Pos(e.clone()));
    Sign {
        pos: p,
        nonneg: p || s.nonneg || h.contains(&Cond::Nonneg(e.clone())),
        ne: p || s.ne || h.contains(&Cond::Ne(e.clone())),
    }
}

fn sign_of(h: &[Cond], e: &R) -> Sign {
    let mk = |pos, nonneg, ne| Sign { pos, nonneg, ne };
    match &**e {
        AE::Var(_) => close(h, e, mk(false, false, false)),
        AE::Zero => mk(false, true, false),
        AE::One | AE::Pi => mk(true, true, true),
        AE::Lit(n) => mk(*n > 0, true, *n > 0),
        AE::Add(a, b) => {
            let (sa, sb) = (sign_of(h, a), sign_of(h, b));
            close(
                h,
                e,
                mk(
                    (sa.pos && sb.nonneg) || (sa.nonneg && sb.pos),
                    sa.nonneg && sb.nonneg,
                    false,
                ),
            )
        }
        AE::Sub(a, b) => {
            let l = h.contains(&Cond::Lt(b.clone(), a.clone()));
            close(h, e, mk(l, l, l))
        }
        AE::Mul(a, b) | AE::Div(a, b) => {
            let (sa, sb) = (sign_of(h, a), sign_of(h, b));
            close(
                h,
                e,
                mk(sa.pos && sb.pos, sa.nonneg && sb.nonneg, sa.ne && sb.ne),
            )
        }
        AE::Neg(a) => {
            let sa = sign_of(h, a);
            close(h, e, mk(false, false, sa.ne))
        }
        AE::Inv(a) => {
            let sa = sign_of(h, a);
            close(h, e, mk(sa.pos, sa.nonneg, sa.ne))
        }
        AE::Pow(a, n) => {
            let sa = sign_of(h, a);
            let ev = n % 2 == 0;
            close(
                h,
                e,
                mk(
                    *n == 0 || sa.pos || (ev && sa.ne),
                    *n == 0 || ev || sa.nonneg,
                    *n == 0 || sa.ne,
                ),
            )
        }
        AE::App(f, a) => {
            let sa = sign_of(h, a);
            match f {
                KFn::Exp | KFn::Cosh => mk(true, true, true),
                KFn::Sqrt => close(h, e, mk(sa.pos, true, sa.pos)),
                _ => close(h, e, mk(false, false, false)),
            }
        }
    }
}

fn implied(h: &[Cond], c: &Cond) -> bool {
    match c {
        Cond::Ne(e) => sign_of(h, e).ne,
        Cond::Pos(e) => sign_of(h, e).pos,
        Cond::Nonneg(e) => sign_of(h, e).nonneg,
        Cond::Lt(a, b) => h.contains(c) || sign_of(h, &rc(AE::Sub(b.clone(), a.clone()))).pos,
    }
}

// ===========================================================================
// Relations (mirror of `Alkahest.Kernel.Rel`)
// ===========================================================================

#[derive(Clone, PartialEq, Eq, Debug)]
enum Rel {
    Inv(R),
    Sqrt(R),
    Pyth(R),
    Hpyth(R),
    Tan(R),
    Tanh(R),
    SinAdd(R, R),
    CosAdd(R, R),
    SinNeg(R),
    CosNeg(R),
    ExpAdd(R, R),
    ExpNeg(R),
    LogExp(R),
}

impl Rel {
    fn sides(&self) -> (AE, AE) {
        let app = |f: KFn, a: &R| rc(AE::App(f, a.clone()));
        let sq = |e: R| rc(AE::Pow(e, 2));
        let add = |a: &R, b: &R| rc(AE::Add(a.clone(), b.clone()));
        match self {
            Rel::Inv(u) => (AE::Mul(u.clone(), rc(AE::Inv(u.clone()))), AE::One),
            Rel::Sqrt(u) => (AE::Pow(app(KFn::Sqrt, u), 2), (**u).clone()),
            Rel::Pyth(a) => (AE::Add(sq(app(KFn::Sin, a)), sq(app(KFn::Cos, a))), AE::One),
            Rel::Hpyth(a) => (
                AE::Sub(sq(app(KFn::Cosh, a)), sq(app(KFn::Sinh, a))),
                AE::One,
            ),
            Rel::Tan(a) => (
                AE::App(KFn::Tan, a.clone()),
                AE::Div(app(KFn::Sin, a), app(KFn::Cos, a)),
            ),
            Rel::Tanh(a) => (
                AE::App(KFn::Tanh, a.clone()),
                AE::Div(app(KFn::Sinh, a), app(KFn::Cosh, a)),
            ),
            Rel::SinAdd(a, b) => (
                AE::App(KFn::Sin, add(a, b)),
                AE::Add(
                    rc(AE::Mul(app(KFn::Sin, a), app(KFn::Cos, b))),
                    rc(AE::Mul(app(KFn::Cos, a), app(KFn::Sin, b))),
                ),
            ),
            Rel::CosAdd(a, b) => (
                AE::App(KFn::Cos, add(a, b)),
                AE::Sub(
                    rc(AE::Mul(app(KFn::Cos, a), app(KFn::Cos, b))),
                    rc(AE::Mul(app(KFn::Sin, a), app(KFn::Sin, b))),
                ),
            ),
            Rel::SinNeg(a) => (
                AE::App(KFn::Sin, rc(AE::Neg(a.clone()))),
                AE::Neg(app(KFn::Sin, a)),
            ),
            Rel::CosNeg(a) => (
                AE::App(KFn::Cos, rc(AE::Neg(a.clone()))),
                AE::App(KFn::Cos, a.clone()),
            ),
            Rel::ExpAdd(a, b) => (
                AE::App(KFn::Exp, add(a, b)),
                AE::Mul(app(KFn::Exp, a), app(KFn::Exp, b)),
            ),
            Rel::ExpNeg(a) => (
                AE::App(KFn::Exp, rc(AE::Neg(a.clone()))),
                AE::Inv(app(KFn::Exp, a)),
            ),
            Rel::LogExp(a) => (AE::App(KFn::Log, app(KFn::Exp, a)), (**a).clone()),
        }
    }
    fn conds(&self) -> Vec<Cond> {
        match self {
            Rel::Inv(u) => vec![Cond::Ne(u.clone())],
            Rel::Sqrt(u) => vec![Cond::Nonneg(u.clone())],
            _ => vec![],
        }
    }
    fn data(&self, out: &mut String) {
        let (name, args): (&str, Vec<&R>) = match self {
            Rel::Inv(u) => ("inv", vec![u]),
            Rel::Sqrt(u) => ("sqrt", vec![u]),
            Rel::Pyth(a) => ("pyth", vec![a]),
            Rel::Hpyth(a) => ("hpyth", vec![a]),
            Rel::Tan(a) => ("tan", vec![a]),
            Rel::Tanh(a) => ("tanh", vec![a]),
            Rel::SinAdd(a, b) => ("sinAdd", vec![a, b]),
            Rel::CosAdd(a, b) => ("cosAdd", vec![a, b]),
            Rel::SinNeg(a) => ("sinNeg", vec![a]),
            Rel::CosNeg(a) => ("cosNeg", vec![a]),
            Rel::ExpAdd(a, b) => ("expAdd", vec![a, b]),
            Rel::ExpNeg(a) => ("expNeg", vec![a]),
            Rel::LogExp(a) => ("logExp", vec![a]),
        };
        let _ = write!(out, "(.{name}");
        for a in args {
            out.push(' ');
            a.data(out);
        }
        out.push(')');
    }
}

/// How the cofactor solver uses a relation's generator.
#[derive(Clone, Debug)]
enum Kind {
    /// `v - e`.
    Subst { v: u32, e: Poly },
    /// `v ^ d - tail`.
    Power { v: u32, d: u32, tail: Poly },
    /// `u · w - 1`.
    Inverse { w: u32, u: Poly },
}

// ===========================================================================
// Atom table and normal forms (mirror of `Alkahest.Kernel.Norm`)
// ===========================================================================

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Head {
    Var(u32),
    Pi,
    Fn(KFn),
    Inv,
}

struct Entry {
    head: Head,
    /// The argument as written in the certificate (its normal form is `poly`).
    arg: R,
    poly: Poly,
    depth: u32,
}

#[derive(Default)]
struct Table {
    atoms: Vec<Entry>,
    index: HashMap<(Head, Poly), u32>,
}

impl Table {
    fn depth_of(&self, p: &Poly) -> u32 {
        p.vars()
            .iter()
            .map(|&v| self.atoms[v as usize].depth)
            .max()
            .unwrap_or(0)
    }

    fn atom(&mut self, head: Head, poly: Poly, arg: R) -> u32 {
        let poly = match head {
            Head::Var(_) | Head::Pi => Poly::zero(),
            _ => poly,
        };
        if let Some(&i) = self.index.get(&(head.clone(), poly.clone())) {
            return i;
        }
        let depth = match head {
            Head::Var(_) | Head::Pi => 0,
            _ => 1 + self.depth_of(&poly),
        };
        let i = self.atoms.len() as u32;
        self.index.insert((head.clone(), poly.clone()), i);
        self.atoms.push(Entry {
            head,
            arg,
            poly,
            depth,
        });
        i
    }

    /// An expression whose normal form is atom `i`.
    fn atom_expr(&self, i: u32) -> R {
        let e = &self.atoms[i as usize];
        match &e.head {
            Head::Var(j) => rc(AE::Var(*j)),
            Head::Pi => rc(AE::Pi),
            Head::Fn(f) => rc(AE::App(*f, e.arg.clone())),
            Head::Inv => rc(AE::Inv(e.arg.clone())),
        }
    }

    /// An expression whose normal form is `p`.
    fn poly_expr(&self, p: &Poly) -> R {
        let mut acc: Option<R> = None;
        for (m, c) in &p.0 {
            let mut t: Option<R> = None;
            for &(v, e) in m {
                let a = self.atom_expr(v);
                let f = if e == 1 { a } else { rc(AE::Pow(a, e)) };
                t = Some(match t {
                    None => f,
                    Some(t) => rc(AE::Mul(t, f)),
                });
            }
            let mag = Rational::from(c.abs_ref());
            let coeff = if *mag.denom() == 1 {
                rc(of_nat(mag.numer()))
            } else {
                rc(AE::Div(rc(of_nat(mag.numer())), rc(of_nat(mag.denom()))))
            };
            let term = match t {
                None => coeff,
                Some(t) if mag == 1 => t,
                Some(t) => rc(AE::Mul(coeff, t)),
            };
            let neg = *c < 0;
            acc = Some(match (acc, neg) {
                (None, false) => term,
                (None, true) => rc(AE::Neg(term)),
                (Some(a), false) => rc(AE::Add(a, term)),
                (Some(a), true) => rc(AE::Sub(a, term)),
            });
        }
        acc.unwrap_or_else(|| rc(AE::Zero))
    }

    fn inv_atom(&mut self, i: u32) -> Poly {
        if self.atoms[i as usize].head == Head::Inv {
            return self.atoms[i as usize].poly.clone();
        }
        let arg = self.atom_expr(i);
        Poly::var(self.atom(Head::Inv, Poly::var(i), arg))
    }

    fn norm_inv(&mut self, p: Poly, src: &R) -> Result<Poly, IntegrationCertWithheld> {
        if p.is_zero() {
            return Ok(Poly::zero());
        }
        if p.len() == 1 {
            let (m, c) =
                p.0.iter()
                    .next()
                    .map(|(m, c)| (m.clone(), c.clone()))
                    .unwrap();
            let mut r = Poly::one();
            // `invMono` multiplies from the last factor inwards; the product is
            // the same polynomial in any order.
            for &(i, e) in &m {
                let w = self.inv_atom(i);
                r = r.mul(&w.pow(e)?)?;
            }
            return Ok(r.scale(&Rational::from(c.recip_ref())));
        }
        Ok(Poly::var(self.atom(Head::Inv, p, src.clone())))
    }

    fn norm(&mut self, e: &R) -> Result<Poly, IntegrationCertWithheld> {
        Ok(match &**e {
            AE::Var(i) => Poly::var(self.atom(Head::Var(*i), Poly::zero(), rc(AE::Zero))),
            AE::Zero => Poly::zero(),
            AE::One => Poly::one(),
            AE::Lit(n) => Poly::konst(Rational::from(n)),
            AE::Pi => Poly::var(self.atom(Head::Pi, Poly::zero(), rc(AE::Zero))),
            AE::Add(a, b) => {
                let p = self.norm(a)?;
                p.add(&self.norm(b)?)
            }
            AE::Sub(a, b) => {
                let p = self.norm(a)?;
                p.sub(&self.norm(b)?)
            }
            AE::Mul(a, b) => {
                let p = self.norm(a)?;
                p.mul(&self.norm(b)?)?
            }
            AE::Div(a, b) => {
                let p = self.norm(a)?;
                let q = self.norm(b)?;
                let r = self.norm_inv(q, b)?;
                p.mul(&r)?
            }
            AE::Neg(a) => self.norm(a)?.neg(),
            AE::Inv(a) => {
                let p = self.norm(a)?;
                self.norm_inv(p, a)?
            }
            AE::Pow(a, n) => {
                if *n > 64 {
                    return Err(WITHHELD_TOO_LARGE);
                }
                self.norm(a)?.pow(*n)?
            }
            AE::App(f, a) => {
                let p = self.norm(a)?;
                Poly::var(self.atom(Head::Fn(f.canon()), p, a.clone()))
            }
        })
    }
}

// ===========================================================================
// Expr → AExpr
// ===========================================================================

struct Conv<'a> {
    pool: &'a ExprPool,
    var: ExprId,
    names: Vec<String>,
    params: HashMap<ExprId, u32>,
}

fn int_of(e: ExprId, pool: &ExprPool) -> Option<Integer> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(n.0.clone()),
        _ => None,
    })
}

fn num_of(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(Rational::from(&n.0)),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    })
}

fn rat_ae(q: &Rational) -> R {
    let mag = Rational::from(q.abs_ref());
    let m = if *mag.denom() == 1 {
        rc(of_nat(mag.numer()))
    } else {
        rc(AE::Div(rc(of_nat(mag.numer())), rc(of_nat(mag.denom()))))
    };
    if *q < 0 {
        rc(AE::Neg(m))
    } else {
        m
    }
}

impl<'a> Conv<'a> {
    fn conv(&mut self, e: ExprId) -> Result<R, IntegrationCertWithheld> {
        enum Shape {
            Num(Rational),
            Sym(String),
            Add(Vec<ExprId>),
            Mul(Vec<ExprId>),
            Pow(ExprId, ExprId),
            Func(String, Vec<ExprId>),
            RootSum,
            Other,
        }
        let shape = self.pool.with(e, |d| match d {
            ExprData::Integer(n) => Shape::Num(Rational::from(&n.0)),
            ExprData::Rational(r) => Shape::Num(r.0.clone()),
            ExprData::Symbol { name, .. } => Shape::Sym(name.clone()),
            ExprData::Add(xs) => Shape::Add(xs.clone()),
            ExprData::Mul(xs) => Shape::Mul(xs.clone()),
            ExprData::Pow { base, exp } => Shape::Pow(*base, *exp),
            ExprData::Func { name, args } => Shape::Func(name.clone(), args.clone()),
            ExprData::RootSum { .. } => Shape::RootSum,
            _ => Shape::Other,
        });
        match shape {
            Shape::Num(q) => {
                if q.numer().significant_bits() > 4096 || q.denom().significant_bits() > 4096 {
                    return Err(WITHHELD_TOO_LARGE);
                }
                Ok(rat_ae(&q))
            }
            Shape::Sym(name) => {
                if e == self.var {
                    return Ok(rc(AE::Var(0)));
                }
                if name == "pi" {
                    return Ok(rc(AE::Pi));
                }
                if let Some(&i) = self.params.get(&e) {
                    return Ok(rc(AE::Var(i)));
                }
                if !super::is_lean_binder_name(&name)
                    || name == self.names[0]
                    || self.names.contains(&name)
                    || is_reserved(&name)
                {
                    return Err(WITHHELD_UNSUPPORTED_NODE);
                }
                let i = self.names.len() as u32;
                self.names.push(name);
                self.params.insert(e, i);
                Ok(rc(AE::Var(i)))
            }
            Shape::Add(xs) => {
                let mut acc: Option<R> = None;
                for x in xs {
                    let (neg, t) = self.signed_term(x)?;
                    acc = Some(match (acc, neg) {
                        (None, false) => t,
                        (None, true) => rc(AE::Neg(t)),
                        (Some(a), false) => rc(AE::Add(a, t)),
                        (Some(a), true) => rc(AE::Sub(a, t)),
                    });
                }
                acc.ok_or(WITHHELD_UNSUPPORTED_NODE)
            }
            Shape::Mul(xs) => {
                let (neg, t) = self.mul_term(&xs)?;
                Ok(if neg { rc(AE::Neg(t)) } else { t })
            }
            Shape::Pow(b, x) => {
                let n = int_of(x, self.pool).ok_or(WITHHELD_UNSUPPORTED_NODE)?;
                let n = n.to_i64().ok_or(WITHHELD_TOO_LARGE)?;
                if n.unsigned_abs() > 64 {
                    return Err(WITHHELD_TOO_LARGE);
                }
                let bb = self.conv(b)?;
                Ok(match n {
                    1 => bb,
                    -1 => rc(AE::Inv(bb)),
                    n if n >= 0 => rc(AE::Pow(bb, n as u32)),
                    n => rc(AE::Inv(rc(AE::Pow(bb, (-n) as u32)))),
                })
            }
            Shape::Func(name, args) => {
                if args.len() != 1 {
                    return Err(special_function_reason(&name));
                }
                if name == "log" {
                    let inner_abs = self.pool.with(args[0], |d| match d {
                        ExprData::Func { name, args } if name == "abs" && args.len() == 1 => {
                            Some(args[0])
                        }
                        _ => None,
                    });
                    if let Some(a) = inner_abs {
                        let a = self.conv(a)?;
                        return Ok(rc(AE::App(KFn::Logabs, a)));
                    }
                }
                let f = KFn::from_name(&name).ok_or_else(|| special_function_reason(&name))?;
                let a = self.conv(args[0])?;
                Ok(rc(AE::App(f, a)))
            }
            Shape::RootSum => Err(WITHHELD_ROOTSUM),
            Shape::Other => Err(WITHHELD_UNSUPPORTED_NODE),
        }
    }

    /// A summand as `(negated?, |term|)`, so sums print as `a - b`.
    fn signed_term(&mut self, x: ExprId) -> Result<(bool, R), IntegrationCertWithheld> {
        if let Some(q) = num_of(x, self.pool) {
            return Ok((q < 0, rat_ae(&Rational::from(q.abs_ref()))));
        }
        let xs = self.pool.with(x, |d| match d {
            ExprData::Mul(xs) => Some(xs.clone()),
            _ => None,
        });
        match xs {
            Some(xs) => self.mul_term(&xs),
            None => Ok((false, self.conv(x)?)),
        }
    }

    /// A product as `(negated?, |c| · num / den)`.
    fn mul_term(&mut self, xs: &[ExprId]) -> Result<(bool, R), IntegrationCertWithheld> {
        let mut coeff = Rational::from(1);
        let mut num: Vec<R> = Vec::new();
        let mut den: Vec<R> = Vec::new();
        for &x in xs {
            if let Some(q) = num_of(x, self.pool) {
                coeff *= q;
                continue;
            }
            let neg_pow = self.pool.with(x, |d| match d {
                ExprData::Pow { base, exp } => int_of(*exp, self.pool)
                    .filter(|n| *n < 0)
                    .map(|n| (*base, n)),
                _ => None,
            });
            match neg_pow {
                Some((b, n)) => {
                    let n = n.to_i64().ok_or(WITHHELD_TOO_LARGE)?;
                    if n.unsigned_abs() > 64 {
                        return Err(WITHHELD_TOO_LARGE);
                    }
                    let bb = self.conv(b)?;
                    den.push(if n == -1 {
                        bb
                    } else {
                        rc(AE::Pow(bb, (-n) as u32))
                    });
                }
                None => num.push(self.conv(x)?),
            }
        }
        let neg = coeff < 0;
        let mag = Rational::from(coeff.abs_ref());
        if mag.numer().significant_bits() > 4096 || mag.denom().significant_bits() > 4096 {
            return Err(WITHHELD_TOO_LARGE);
        }
        let prod = |v: Vec<R>| -> Option<R> { v.into_iter().reduce(|a, b| rc(AE::Mul(a, b))) };
        // numeric numerator in front, numeric denominator joins the symbolic one
        let mut top: Vec<R> = Vec::new();
        if *mag.numer() != 1 || num.is_empty() && den.is_empty() {
            top.push(rc(of_nat(mag.numer())));
        }
        top.extend(num);
        let mut bottom: Vec<R> = Vec::new();
        if *mag.denom() != 1 {
            bottom.push(rc(of_nat(mag.denom())));
        }
        bottom.extend(den);
        let t = match (prod(top), prod(bottom)) {
            (Some(t), None) => t,
            (Some(t), Some(b)) => rc(AE::Div(t, b)),
            (None, Some(b)) => rc(AE::Inv(b)),
            (None, None) => rc(AE::One),
        };
        Ok((neg, t))
    }
}

fn is_reserved(name: &str) -> bool {
    let numbered = |p: char| {
        name.starts_with(p) && name.len() > 1 && name[1..].chars().all(|c| c.is_ascii_digit())
    };
    numbered('h')
        || matches!(
            name,
            "D" | "H" | "F" | "f" | "w" | "ps" | "x" | "EulerGamma" | "trivial" | "Real"
        )
}

// ===========================================================================
// Relations and cofactors
// ===========================================================================

struct Solver {
    tbl: Table,
    rels: Vec<(Rel, Kind, Poly)>,
}

impl Solver {
    /// Register a relation (computing its generator through the table).
    fn push(&mut self, r: Rel) -> Result<(), IntegrationCertWithheld> {
        if self.rels.iter().any(|(q, _, _)| *q == r) {
            return Ok(());
        }
        let (l, rr) = r.sides();
        let lp = self.tbl.norm(&rc(l))?;
        let rp = self.tbl.norm(&rc(rr))?;
        let g = lp.sub(&rp);
        if g.is_zero() {
            return Ok(());
        }
        let kind = match &r {
            Rel::Inv(u) => {
                // `u · u⁻¹ - 1`: the inverse atom is the factor that is an inverse.
                let up = self.tbl.norm(u)?;
                let w = self.tbl.norm_inv(up.clone(), u)?;
                match single_var(&w) {
                    Some(w) if self.tbl.atoms[w as usize].head == Head::Inv => {
                        Kind::Inverse { w, u: up }
                    }
                    _ => return Ok(()),
                }
            }
            Rel::Sqrt(_) | Rel::Pyth(_) => {
                let (s, tail) = power_split(&lp, &rp)?;
                Kind::Power { v: s, d: 2, tail }
            }
            Rel::Hpyth(a) => {
                let c = self.tbl.norm(&rc(AE::App(KFn::Cosh, a.clone())))?;
                let c = single_var(&c).ok_or(WITHHELD_IDENTITY)?;
                // g = c² - s² - 1
                let tail = Poly::var(c).pow(2)?.sub(&g);
                Kind::Power { v: c, d: 2, tail }
            }
            _ => {
                let v = single_var(&lp).ok_or(WITHHELD_IDENTITY)?;
                if rp.has_var(v) {
                    return Ok(());
                }
                Kind::Subst { v, e: rp }
            }
        };
        self.rels.push((r, kind, g));
        Ok(())
    }

    /// Relations for every atom, until no new atoms appear.
    fn build(&mut self) -> Result<(), IntegrationCertWithheld> {
        let mut done = 0usize;
        let mut families_at = usize::MAX;
        loop {
            while done < self.tbl.atoms.len() {
                let i = done as u32;
                done += 1;
                let (head, arg, poly) = {
                    let e = &self.tbl.atoms[i as usize];
                    (e.head.clone(), e.arg.clone(), e.poly.clone())
                };
                match head {
                    Head::Inv => {
                        // `u · u⁻¹ = 1`, `u` the inverted expression.
                        let u = if poly.len() == 1 && single_var(&poly).is_some() {
                            self.tbl.atom_expr(single_var(&poly).unwrap())
                        } else {
                            arg.clone()
                        };
                        self.push(Rel::Inv(u))?;
                    }
                    Head::Fn(KFn::Sqrt) => self.push(Rel::Sqrt(arg.clone()))?,
                    Head::Fn(KFn::Tan) => self.push(Rel::Tan(arg.clone()))?,
                    Head::Fn(KFn::Tanh) => self.push(Rel::Tanh(arg.clone()))?,
                    Head::Fn(KFn::Log) => {
                        if let Some(v) = single_var(&poly) {
                            let inner = &self.tbl.atoms[v as usize];
                            if inner.head == Head::Fn(KFn::Exp) && poly == Poly::var(v) {
                                let a = inner.arg.clone();
                                self.push(Rel::LogExp(a))?;
                            }
                        }
                    }
                    _ => {}
                }
            }
            if families_at == self.tbl.atoms.len() {
                return Ok(());
            }
            self.families()?;
            families_at = self.tbl.atoms.len();
            if self.tbl.atoms.len() > 4000 || self.rels.len() > 4000 {
                return Err(WITHHELD_TOO_LARGE);
            }
        }
    }

    fn fun_atoms(&self, fs: &[KFn]) -> Vec<(u32, KFn, Poly)> {
        self.tbl
            .atoms
            .iter()
            .enumerate()
            .filter_map(|(i, a)| match a.head {
                Head::Fn(f) if fs.contains(&f) => Some((i as u32, f, a.poly.clone())),
                _ => None,
            })
            .collect()
    }

    /// Trig / exp / hyperbolic families: arguments that are integer multiples
    /// of a common base are related through addition formulas.
    fn families(&mut self) -> Result<(), IntegrationCertWithheld> {
        let trig = self.fun_atoms(&[KFn::Sin, KFn::Cos, KFn::Tan]);
        for (base, ks) in multiple_classes(&trig)? {
            let b = self.tbl.poly_expr(&base);
            let at = |k: i64, s: &Solver| s.tbl.poly_expr(&base.scale(&Rational::from(k)));
            self.push(Rel::Pyth(b.clone()))?;
            let kmax = ks.iter().map(|k| k.unsigned_abs()).max().unwrap_or(1) as i64;
            for k in 2..=kmax {
                let prev = at(k - 1, self);
                self.push(Rel::SinAdd(prev.clone(), b.clone()))?;
                self.push(Rel::CosAdd(prev, b.clone()))?;
            }
            for &k in &ks {
                if k < 0 {
                    let m = at(-k, self);
                    self.push(Rel::SinNeg(m.clone()))?;
                    self.push(Rel::CosNeg(m))?;
                }
            }
        }
        let exps = self.fun_atoms(&[KFn::Exp]);
        for (base, ks) in multiple_classes(&exps)? {
            if ks.iter().all(|&k| k == 1) {
                continue;
            }
            let b = self.tbl.poly_expr(&base);
            let kmax = ks.iter().map(|k| k.unsigned_abs()).max().unwrap_or(1) as i64;
            for k in 2..=kmax {
                let prev = self.tbl.poly_expr(&base.scale(&Rational::from(k - 1)));
                self.push(Rel::ExpAdd(prev, b.clone()))?;
            }
            for &k in &ks {
                if k < 0 {
                    let m = self.tbl.poly_expr(&base.scale(&Rational::from(-k)));
                    self.push(Rel::ExpNeg(m))?;
                }
            }
        }
        let hyp = self.fun_atoms(&[KFn::Sinh, KFn::Cosh]);
        let mut seen: Vec<Poly> = Vec::new();
        for (_, _, arg) in hyp {
            if seen.contains(&arg) {
                continue;
            }
            seen.push(arg.clone());
            let a = self.tbl.poly_expr(&arg);
            self.push(Rel::Hpyth(a))?;
        }
        Ok(())
    }

    /// Cofactors with `p = Σ cᵢ·gᵢ` (the reduction of [`super::hasderiv`]).
    fn certify_zero(&self, p: Poly) -> Result<Vec<Poly>, IntegrationCertWithheld> {
        let depth = |v: u32| self.tbl.atoms[v as usize].depth;
        let mut cof = vec![Poly::zero(); self.rels.len()];
        let mut cur = p;
        let mut mult = Poly::one();
        let inv_rels: Vec<(usize, u32, Poly)> = self
            .rels
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match &r.1 {
                Kind::Inverse { w, u } => Some((i, *w, u.clone())),
                _ => None,
            })
            .collect();
        let mut powers: Vec<(usize, u32)> = self
            .rels
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r.1 {
                Kind::Power { v, .. } => Some((i, depth(v))),
                _ => None,
            })
            .collect();
        powers.sort_by_key(|a| std::cmp::Reverse(a.1));
        for _round in 0..MAX_ROUNDS {
            if cur.is_zero() {
                return Ok(cof);
            }
            let mut progressed = false;
            for (i, r) in self.rels.iter().enumerate() {
                if let Kind::Subst { v, e } = &r.1 {
                    if cur.has_var(*v) {
                        let (pe, delta) = substitute(&cur, *v, e)?;
                        cof[i] = cof[i].add(&mult.mul(&delta)?);
                        cur = pe;
                        progressed = true;
                    }
                }
            }
            if progressed {
                continue;
            }
            for (ri, w, u) in &inv_rels {
                let Some(a) = single_var(u) else { continue };
                if *u != Poly::var(a) || !cur.has_var(*w) || !cur.has_var(a) {
                    continue;
                }
                if !cur.0.keys().any(|m| exp_of(m, a) > 0 && exp_of(m, *w) > 0) {
                    continue;
                }
                let aw = Poly::var(a).mul(&Poly::var(*w))?;
                let mut next = Poly::zero();
                let mut q = Poly::zero();
                for (m, c) in &cur.0 {
                    let t = exp_of(m, a).min(exp_of(m, *w));
                    if t == 0 {
                        next.add_term(m.clone(), c.clone());
                        continue;
                    }
                    let mut m2: Mono = Vec::new();
                    for &(v, e) in m {
                        let e2 = if v == a || v == *w { e - t } else { e };
                        if e2 > 0 {
                            m2.push((v, e2));
                        }
                    }
                    next.add_term(m2.clone(), c.clone());
                    let mut s_geo = Poly::zero();
                    let mut tp = Poly::one();
                    for _ in 0..t {
                        s_geo = s_geo.add(&tp);
                        tp = tp.mul(&aw)?;
                    }
                    let mut mono = Poly::zero();
                    mono.add_term(m2, c.clone());
                    q = q.add(&mono.mul(&s_geo)?);
                }
                cof[*ri] = cof[*ri].add(&mult.mul(&q)?);
                cur = next;
                progressed = true;
            }
            if progressed {
                continue;
            }
            let vars = cur.vars();
            if let Some((ri, w, u)) = inv_rels
                .iter()
                .filter(|(_, w, _)| vars.contains(w))
                .max_by_key(|(_, w, _)| depth(*w))
            {
                let cs = cur.split(*w);
                let kmax = (cs.len() - 1) as u32;
                let t = u.mul(&Poly::var(*w))?;
                let mut next = Poly::zero();
                let mut a = Poly::zero();
                for (k, c) in cs.iter().enumerate() {
                    if c.is_zero() {
                        continue;
                    }
                    let k = k as u32;
                    next = next.add(&c.mul(&u.pow(kmax - k)?)?);
                    let mut s_geo = Poly::zero();
                    let mut tp = Poly::one();
                    for _ in 0..(kmax - k) {
                        s_geo = s_geo.add(&tp);
                        tp = tp.mul(&t)?;
                    }
                    if !s_geo.is_zero() {
                        a = a.add(&c.mul(&Poly::var(*w).pow(k)?)?.mul(&s_geo)?);
                    }
                }
                cof[*ri] = cof[*ri].sub(&mult.mul(&a)?);
                cur = next;
                mult = mult.mul(&Poly::var(*w).pow(kmax)?)?;
                continue;
            }
            let mut reduced = false;
            for &(i, _) in &powers {
                if let Kind::Power { v, d, tail } = &self.rels[i].1 {
                    if cur.deg_in(*v) >= *d {
                        let (r, q) = reduce_power(&cur, *v, *d, tail)?;
                        cof[i] = cof[i].add(&mult.mul(&q)?);
                        cur = r;
                        reduced = true;
                    }
                }
            }
            if !reduced {
                return Err(WITHHELD_IDENTITY);
            }
        }
        Err(WITHHELD_TOO_LARGE)
    }
}

/// `lhs = s² + rest`: `(s, rhs - rest)` for `s² - tail`.
fn power_split(lp: &Poly, rp: &Poly) -> Result<(u32, Poly), IntegrationCertWithheld> {
    // the squared atom: the unique variable with a degree-2 pure power term
    let mut cand = None;
    for m in lp.0.keys() {
        if m.len() == 1 && m[0].1 == 2 {
            cand = Some(m[0].0);
            break;
        }
    }
    let s = cand.ok_or(WITHHELD_IDENTITY)?;
    let sq: Mono = vec![(s, 2)];
    let mut rest = lp.clone();
    rest.0.remove(&sq);
    if lp.0.get(&sq) != Some(&Rational::from(1)) {
        return Err(WITHHELD_IDENTITY);
    }
    Ok((s, rp.sub(&rest)))
}

/// Group `(atom, f, arg)` by arguments that are rational multiples of a common
/// base; returns `(base, [k])` with integer `k` (the argument is `k · base`).
fn multiple_classes(
    items: &[(u32, KFn, Poly)],
) -> Result<Vec<(Poly, Vec<i64>)>, IntegrationCertWithheld> {
    let mut classes: Vec<(Poly, Vec<Rational>)> = Vec::new();
    for (_, _, arg) in items {
        if arg.is_zero() {
            continue;
        }
        let mut placed = false;
        for (rep, members) in classes.iter_mut() {
            if let Some(q) = arg.ratio_to(rep) {
                members.push(q);
                placed = true;
                break;
            }
        }
        if !placed {
            classes.push((arg.clone(), vec![Rational::from(1)]));
        }
    }
    let mut out = Vec::new();
    for (rep, members) in classes {
        let mut num_g = Integer::new();
        let mut den_l = Integer::from(1);
        for q in &members {
            num_g = num_g.gcd(q.numer());
            den_l = den_l.lcm(q.denom());
        }
        let g = Rational::from((num_g, den_l));
        let mut ks = Vec::new();
        for q in members {
            let k = Rational::from(&q / &g);
            if *k.denom() != 1 {
                return Err(WITHHELD_IDENTITY);
            }
            let k = k.numer().to_i64().ok_or(WITHHELD_TOO_LARGE)?;
            if k.unsigned_abs() > 24 {
                return Err(WITHHELD_TOO_LARGE);
            }
            if !ks.contains(&k) {
                ks.push(k);
            }
        }
        out.push((rep.scale(&g), ks));
    }
    Ok(out)
}

/// `Mono.cmp`: lexicographic order on dense exponent vectors (a monomial order).
fn mono_cmp(a: &Mono, b: &Mono) -> std::cmp::Ordering {
    use std::cmp::Ordering::{Equal, Greater, Less};
    for k in 0.. {
        match (a.get(k), b.get(k)) {
            (None, None) => return Equal,
            (None, Some(_)) => return Less,
            (Some(_), None) => return Greater,
            (Some(&(i, e)), Some(&(j, f))) => {
                if i != j {
                    return if i < j { Greater } else { Less };
                }
                if e != f {
                    return e.cmp(&f);
                }
            }
        }
    }
    Equal
}

fn cof_data(p: &Poly, out: &mut String) {
    // In the kernel's own (descending) order, so `Poly.ofTerms` inserts each
    // term with one comparison instead of walking the list.
    let mut terms: Vec<(&Mono, &Rational)> = p.0.iter().collect();
    terms.sort_by(|x, y| mono_cmp(y.0, x.0));
    out.push('[');
    for (j, (m, c)) in terms.into_iter().enumerate() {
        if j > 0 {
            out.push_str(", ");
        }
        out.push_str("([");
        for (k, (v, e)) in m.iter().enumerate() {
            if k > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "(nat_lit {v}, nat_lit {e})");
        }
        // Raw literals: no `OfNat` instance search per number (the elaborator's
        // cost on large witnesses), and the kernel works on them directly.
        let n = c.numer();
        let num = if *n >= 0 {
            format!("Int.ofNat (nat_lit {n})")
        } else {
            format!("Int.negSucc (nat_lit {})", Integer::from(-n) - 1u32)
        };
        let _ = write!(out, "], {num}, nat_lit {})", c.denom());
    }
    out.push(']');
}

// ===========================================================================
// Entry points
// ===========================================================================

/// Lean header for kernel certificates.
pub(crate) fn kernel_header() -> &'static str {
    "import Alkahest.Kernel\n\n"
}

/// A checked identity `lhs = rhs`: the atom table and the relations used,
/// each with its cofactor.
struct Proof {
    tbl: Table,
    used: Vec<(Rel, Poly)>,
}

impl Proof {
    fn conds(&self) -> Vec<Cond> {
        self.used.iter().flat_map(|(r, _)| r.conds()).collect()
    }

    /// `⟨atoms, relations, cofactors⟩` (a `Witness`).
    fn data(&self) -> String {
        let mut atoms = String::from("[");
        for (i, a) in self.tbl.atoms.iter().enumerate() {
            if i > 0 {
                atoms.push_str(",\n      ");
            }
            let head = match &a.head {
                Head::Var(j) => format!("(.var (nat_lit {j}))"),
                Head::Pi => ".pi".to_string(),
                Head::Fn(f) => format!("(.fn .{})", f.ctor()),
                Head::Inv => ".inv".to_string(),
            };
            let _ = write!(atoms, "⟨{head}, {}⟩", a.arg.data_string());
        }
        atoms.push(']');
        let mut rels = String::from("[");
        let mut cofs = String::from("[");
        for (i, (r, c)) in self.used.iter().enumerate() {
            if i > 0 {
                rels.push_str(",\n      ");
                cofs.push_str(",\n      ");
            }
            r.data(&mut rels);
            cof_data(c, &mut cofs);
        }
        rels.push(']');
        cofs.push(']');
        format!("⟨{atoms},\n      {rels},\n      {cofs}⟩")
    }
}

/// Find relations and cofactors with `lhs - rhs = Σ cᵢ·(Lᵢ - Rᵢ)` in the
/// kernel's normal form, and re-check the identity exactly.
fn prove_identity(lhs: &R, rhs: &R) -> Result<Proof, IntegrationCertWithheld> {
    let mut sv = Solver {
        tbl: Table::default(),
        rels: Vec::new(),
    };
    let lp = sv.tbl.norm(lhs)?;
    let rp = sv.tbl.norm(rhs)?;
    let resid = lp.sub(&rp);
    let mut used: Vec<(Rel, Poly)> = Vec::new();
    if !resid.is_zero() {
        sv.build()?;
        let cof = sv.certify_zero(resid.clone())?;
        let mut combo = Poly::zero();
        for ((r, _, g), c) in sv.rels.iter().zip(cof) {
            if c.is_zero() {
                continue;
            }
            combo = combo.add(&c.mul(g)?);
            used.push((r.clone(), c));
        }
        if combo != resid {
            return Err(WITHHELD_IDENTITY);
        }
    }
    Ok(Proof { tbl: sv.tbl, used })
}

/// Hypotheses: every needed condition the sign checker cannot derive, after a
/// non-vacuity check (some sample point satisfies all of them).
fn hypotheses(needed: &[Cond], nvars: usize) -> Result<Vec<Cond>, IntegrationCertWithheld> {
    let mut hyps: Vec<Cond> = Vec::new();
    for c in needed {
        if implied(&hyps, c) {
            continue;
        }
        if !c.has_var() {
            return Err(WITHHELD_CONSTANT_SIGN);
        }
        hyps.push(c.clone());
    }
    if !hyps.is_empty() {
        const XS: [f64; 16] = [
            0.37, 0.61, 0.83, 1.37, 2.3, -0.42, 0.13, 3.7, -1.7, 5.3, 0.05, 0.95, -3.1, 9.1, -0.77,
            1.9,
        ];
        let found = XS.iter().enumerate().any(|(i, &xv)| {
            let mut env = vec![xv];
            for j in 1..nvars {
                env.push(0.7 + 0.37 * ((i + j) % 5) as f64);
            }
            hyps.iter().all(|c| c.holds(&env))
        });
        if !found {
            return Err(WITHHELD_EMPTY_DOMAIN);
        }
    }
    Ok(hyps)
}

/// The pieces shared by every kernel certificate: binders, `H` data, the
/// `HoldsAll` proof, the parameter list, the closing tactic and its trust note.
struct Frame {
    binders: String,
    hdata: String,
    hproof: String,
    params: String,
    decide: &'static str,
    trust: &'static str,
}

fn frame(names: &[String], hyps: &[Cond], mode: KernelCheckMode) -> Frame {
    let x = &names[0];
    let mut binders = format!("({x} : ℝ)");
    for p in &names[1..] {
        let _ = write!(binders, " ({p} : ℝ)");
    }
    for (i, h) in hyps.iter().enumerate() {
        let _ = write!(binders, " (h{} : {})", i + 1, h.lean(names));
    }
    let mut hdata = String::from("[");
    for (i, h) in hyps.iter().enumerate() {
        if i > 0 {
            hdata.push_str(", ");
        }
        h.data(&mut hdata);
    }
    hdata.push(']');
    let hproof = if hyps.is_empty() {
        "trivial".to_string()
    } else {
        let mut s = String::from("⟨");
        for i in 0..hyps.len() {
            let _ = write!(s, "h{}, ", i + 1);
        }
        s.push_str("trivial⟩");
        s
    };
    let (decide, trust) = match mode {
        KernelCheckMode::Kernel => (
            "by decide +kernel",
            "the Lean kernel evaluates the checker (`decide +kernel`)",
        ),
        KernelCheckMode::Native => (
            "by native_decide",
            "the checker is compiled and run natively (`native_decide`: trusts the compiler)",
        ),
    };
    Frame {
        binders,
        hdata,
        hproof,
        params: format!("[{}]", names[1..].join(", ")),
        decide,
        trust,
    }
}

fn finish(src: String) -> Result<String, IntegrationCertWithheld> {
    if src.len() > MAX_KERNEL_SRC_BYTES {
        return Err(WITHHELD_TOO_LARGE);
    }
    if super::contains_admission(&src) {
        return Err(WITHHELD_UNSUPPORTED_NODE);
    }
    Ok(src)
}

fn conv_for(var: ExprId, pool: &ExprPool) -> Result<Conv<'_>, IntegrationCertWithheld> {
    let var_name = pool
        .with(var, |d| match d {
            ExprData::Symbol { name, .. } => Some(name.clone()),
            _ => None,
        })
        .ok_or(WITHHELD_UNSUPPORTED_NODE)?;
    if !super::is_lean_binder_name(&var_name) || is_reserved(&var_name) && var_name != "x" {
        return Err(WITHHELD_UNSUPPORTED_NODE);
    }
    Ok(Conv {
        pool,
        var,
        names: vec![var_name],
        params: HashMap::new(),
    })
}

/// Emit a reflective-kernel certificate for `∫ integrand d(var) = antiderivative`.
///
/// The certificate imports the repository's Lean library `Alkahest.Kernel`
/// (`lean/`), states `HasDerivAt (fun x => F) f x` on an explicit domain, and
/// proves it by `integral_sound _ (by decide +kernel)` — or `by native_decide`
/// for [`KernelCheckMode::Native`]. Returns the `.lean` source, or the reason
/// the kernel path cannot encode this answer (the caller then falls back to
/// [`super::emit_integration_hasderiv_cert`]). Never contains `sorry`.
pub fn emit_integration_kernel_cert(
    antiderivative: ExprId,
    integrand: ExprId,
    var: ExprId,
    pool: &ExprPool,
    mode: KernelCheckMode,
) -> Result<String, IntegrationCertWithheld> {
    let mut cv = conv_for(var, pool)?;
    let mut memo = HashMap::new();
    let big = respell_rational_powers(antiderivative, pool, &mut memo)?;
    let small = respell_rational_powers(integrand, pool, &mut memo)?;
    let f_big = cv.conv(big)?;
    let f_small = cv.conv(small)?;
    let names = cv.names.clone();

    // The checker's derivative, its domain, and the identity `D F = f`.
    let proof = prove_identity(&deriv(&f_big, 0), &f_small)?;
    let mut needed = Vec::new();
    dconds(&f_big, &mut needed);
    needed.extend(proof.conds());
    let hyps = hypotheses(&needed, names.len())?;
    let fr = frame(&names, &hyps, mode);
    let x = &names[0];
    finish(format!(
        "{}-- ∫ {} d{x} = {}\n\
         -- Certified by the Alkahest reflective kernel: `integral_sound` (proved in\n\
         -- `Alkahest.Kernel.Check`) turns `checkIntegral H F f w = true` into the\n\
         -- statement; {}.\n\
         open Alkahest.Kernel in\n\
         theorem alkahest_integral {} :\n    \
         HasDerivAt (fun ({x} : ℝ) => {}) ({}) {x} :=\n  \
         integral_sound (H := {})\n    \
         (F := {})\n    \
         (f := {})\n    \
         (w := {})\n    \
         ({}) {} {x} {}\n",
        kernel_header(),
        pool.display(integrand),
        pool.display(antiderivative),
        fr.trust,
        fr.binders,
        f_big.lean_string(&names),
        f_small.lean_string(&names),
        fr.hdata,
        f_big.data_string(),
        f_small.data_string(),
        proof.data(),
        fr.decide,
        fr.params,
        fr.hproof,
    ))
}

/// Emit a reflective-kernel certificate that `y = solution` solves the ODE
/// `equation = 0` in `x`, where `equation` is written with the symbols `y`
/// and `derivs = [y', y'', …]` (the `dsolve` input convention).
///
/// The theorem states, on an explicit domain, that `S₀ = solution`,
/// `S₁ = S₀'`, …, `Sₙ = Sₙ₋₁'` (each as `HasDerivAt`, with `Sₖ` Alkahest's
/// derivative) and that the equation with `Sₖ` substituted for `y⁽ᵏ⁾` is `0`.
/// It is proved by `ode_sound _ (by decide +kernel)`. Free constants of the
/// solution (`C1`, …) become universally quantified parameters.
pub fn emit_ode_solution_kernel_cert(
    equation: ExprId,
    x: ExprId,
    y: ExprId,
    derivs: &[ExprId],
    solution: ExprId,
    pool: &ExprPool,
    mode: KernelCheckMode,
) -> Result<String, IntegrationCertWithheld> {
    if derivs.is_empty() || derivs.len() > 8 {
        return Err(WITHHELD_UNSUPPORTED_NODE);
    }
    let mut cv = conv_for(x, pool)?;
    // S₀ … Sₙ (Alkahest's own derivatives).
    let mut ss = vec![solution];
    for _ in 0..derivs.len() {
        let prev = *ss.last().unwrap();
        let d = crate::diff::diff(prev, x, pool).map_err(|_| WITHHELD_UNSUPPORTED_NODE)?;
        ss.push(d.value);
    }
    let mut map = HashMap::new();
    map.insert(y, ss[0]);
    for (k, &d) in derivs.iter().enumerate() {
        map.insert(d, ss[k + 1]);
    }
    let e_sub = crate::kernel::subs(equation, &map, pool);
    let mut memo = HashMap::new();
    let mut s_ae = Vec::new();
    for &s in &ss {
        let s = respell_rational_powers(s, pool, &mut memo)?;
        s_ae.push(cv.conv(s)?);
    }
    let e_resp = respell_rational_powers(e_sub, pool, &mut memo)?;
    let e_ae = cv.conv(e_resp)?;
    let names = cv.names.clone();

    let mut needed = Vec::new();
    let mut links = Vec::new();
    for k in 0..derivs.len() {
        let p = prove_identity(&deriv(&s_ae[k], 0), &s_ae[k + 1])?;
        dconds(&s_ae[k], &mut needed);
        needed.extend(p.conds());
        links.push(p);
    }
    let pe = prove_identity(&e_ae, &rc(AE::Zero))?;
    needed.extend(pe.conds());
    let hyps = hypotheses(&needed, names.len())?;
    let fr = frame(&names, &hyps, mode);
    let xn = &names[0];
    let chain: Vec<String> = (0..derivs.len())
        .map(|k| {
            format!(
                "HasDerivAt (fun ({xn} : ℝ) => {}) ({}) {xn}",
                s_ae[k].lean_string(&names),
                s_ae[k + 1].lean_string(&names)
            )
        })
        .collect();
    let ss_data: Vec<String> = s_ae.iter().map(|s| s.data_string()).collect();
    let ws: Vec<String> = links.iter().map(|p| p.data()).collect();
    finish(format!(
        "{}-- ODE: {} = 0 in {xn}, solution y = {}\n\
         -- Certified by the Alkahest reflective kernel: `ode_sound` (proved in\n\
         -- `Alkahest.Kernel.Check`) turns `checkOde H Ss ws E w = true` into the\n\
         -- statement; {}.\n\
         open Alkahest.Kernel in\n\
         theorem alkahest_ode_solution {} :\n    \
         ({}) ∧ {} = 0 :=\n  \
         ode_sound (H := {})\n    \
         (Ss := [{}])\n    \
         (ws := [{}])\n    \
         (E := {})\n    \
         (w := {})\n    \
         ({}) {} {xn} {}\n",
        kernel_header(),
        pool.display(equation),
        pool.display(solution),
        fr.trust,
        fr.binders,
        chain.join(" ∧\n      "),
        e_ae.lean_string(&names),
        fr.hdata,
        ss_data.join(",\n      "),
        ws.join(",\n      "),
        e_ae.data_string(),
        pe.data(),
        fr.decide,
        fr.params,
        fr.hproof,
    ))
}

/// `AExpr.bind`: substitute `σ[j]` (or `0` past the end) for `var j`.
fn bind(e: &R, sigma: &[R]) -> R {
    match &**e {
        AE::Var(j) => sigma
            .get(*j as usize)
            .cloned()
            .unwrap_or_else(|| rc(AE::Zero)),
        AE::Zero | AE::One | AE::Lit(_) | AE::Pi => e.clone(),
        AE::Add(a, b) => rc(AE::Add(bind(a, sigma), bind(b, sigma))),
        AE::Sub(a, b) => rc(AE::Sub(bind(a, sigma), bind(b, sigma))),
        AE::Mul(a, b) => rc(AE::Mul(bind(a, sigma), bind(b, sigma))),
        AE::Div(a, b) => rc(AE::Div(bind(a, sigma), bind(b, sigma))),
        AE::Neg(a) => rc(AE::Neg(bind(a, sigma))),
        AE::Inv(a) => rc(AE::Inv(bind(a, sigma))),
        AE::Pow(a, n) => rc(AE::Pow(bind(a, sigma), *n)),
        AE::App(f, a) => rc(AE::App(*f, bind(a, sigma))),
    }
}

fn max_var(e: &R) -> Option<u32> {
    match &**e {
        AE::Var(j) => Some(*j),
        AE::Zero | AE::One | AE::Lit(_) | AE::Pi => None,
        AE::Add(a, b) | AE::Sub(a, b) | AE::Mul(a, b) | AE::Div(a, b) => {
            match (max_var(a), max_var(b)) {
                (Some(x), Some(y)) => Some(x.max(y)),
                (x, None) => x,
                (None, y) => y,
            }
        }
        AE::Neg(a) | AE::Inv(a) | AE::Pow(a, _) | AE::App(_, a) => max_var(a),
    }
}

/// Emit a reflective-kernel certificate that the PDE operator `operator`
/// vanishes on the jet of `solution` at every point of `ℝ^d`.
///
/// `operator` is written in the coordinates `coords` (`d` of them) and the
/// jet symbols `u`, `lap` (the Laplacian over all of `coords`) and `grad`
/// (one per coordinate, or empty when the operator reads no gradient). The
/// theorem is stated with Mathlib's `deriv` along the coordinate lines of a
/// definition `fexU` of the solution, and proved by
/// `pde_sound _ (by decide +kernel)` (`Alkahest.Kernel.Pde`): the kernel
/// computes the jet with its own derivative, substitutes it and checks the
/// residual is zero. Differentiability conditions must hold everywhere
/// without hypotheses; otherwise the certificate is withheld.
#[allow(clippy::too_many_arguments)]
pub fn emit_pde_kernel_cert(
    operator: ExprId,
    solution: ExprId,
    coords: &[ExprId],
    u: ExprId,
    lap: ExprId,
    grad: &[ExprId],
    pool: &ExprPool,
    mode: KernelCheckMode,
) -> Result<String, IntegrationCertWithheld> {
    let d = coords.len();
    if d == 0 || d > 200 || !(grad.is_empty() || grad.len() == d) {
        return Err(WITHHELD_UNSUPPORTED_NODE);
    }
    let mut cv = conv_for(coords[0], pool)?;
    // Coordinates are var 0 … d-1, then u = d, Δu = d+1, ∂ᵢu = d+2+i.
    for (k, &c) in coords.iter().enumerate().skip(1) {
        if *cv.conv(c)? != AE::Var(k as u32) {
            return Err(WITHHELD_UNSUPPORTED_NODE);
        }
    }
    let mut jets = vec![u, lap];
    jets.extend_from_slice(grad);
    for (k, &j) in jets.iter().enumerate() {
        if *cv.conv(j)? != AE::Var((d + k) as u32) {
            return Err(WITHHELD_UNSUPPORTED_NODE);
        }
    }
    let mut memo = HashMap::new();
    let sol = respell_rational_powers(solution, pool, &mut memo)?;
    let op = respell_rational_powers(operator, pool, &mut memo)?;
    let u_ae = cv.conv(sol)?;
    if cv.names.len() != d + jets.len() || max_var(&u_ae).is_some_and(|m| m as usize >= d) {
        // a free constant or a jet symbol in the solution
        return Err(WITHHELD_UNSUPPORTED_NODE);
    }
    let op_ae = cv.conv(op)?;
    if cv.names.len() != d + jets.len() {
        return Err(WITHHELD_UNSUPPORTED_NODE);
    }
    // The jet, exactly as `lapE` / `gradE` build it.
    let mut lap_e = rc(AE::Zero);
    let mut grads = Vec::with_capacity(d);
    for i in 0..d as u32 {
        let g = deriv(&u_ae, i);
        lap_e = rc(AE::Add(lap_e, deriv(&g, i)));
        grads.push(g);
    }
    let mut sigma: Vec<R> = (0..d as u32).map(|i| rc(AE::Var(i))).collect();
    sigma.push(u_ae.clone());
    sigma.push(lap_e);
    sigma.extend(grads.iter().cloned());
    let resid = bind(&op_ae, &sigma);
    // Conditions: every one must follow from no hypotheses.
    let mut needed = Vec::new();
    dconds(&u_ae, &mut needed);
    for g in &grads {
        dconds(g, &mut needed);
    }
    let proof = prove_identity(&resid, &rc(AE::Zero))?;
    needed.extend(proof.conds());
    if needed.iter().any(|c| !implied(&[], c)) {
        return Err(WITHHELD_CONSTANT_SIGN);
    }
    // The statement: `fexU` is the solution; the jet in Mathlib's `deriv`.
    let coord_names: Vec<String> = cv.names[..d].to_vec();
    let binders = coord_names.join(" ");
    let call = |slot: Option<(usize, &str)>| -> String {
        let args: Vec<String> = coord_names
            .iter()
            .enumerate()
            .map(|(k, n)| match slot {
                Some((i, y)) if i == k => y.to_string(),
                _ => n.clone(),
            })
            .collect();
        format!("fexU {}", args.join(" "))
    };
    let mut lap_s = String::from("(0 : ℝ)");
    let mut grad_s = Vec::with_capacity(d);
    for (i, n) in coord_names.iter().enumerate() {
        lap_s = format!(
            "({lap_s} + deriv (deriv (fun (y : ℝ) => {})) {n})",
            call(Some((i, "y")))
        );
        grad_s.push(format!(
            "deriv (fun (y : ℝ) => {}) {n}",
            call(Some((i, "y")))
        ));
    }
    let mut names: Vec<String> = coord_names.clone();
    names.push(call(None));
    names.push(lap_s);
    names.extend(grad_s);
    if grad.is_empty() {
        names.truncate(d + 2);
    }
    let fr = frame(&coord_names, &[], mode);
    let stmt = op_ae.lean_string(&names);
    finish(format!(
        "{}-- PDE: {} = 0 for u = {} at every point of ℝ^{d}\n\
         -- Certified by the Alkahest reflective kernel: `pde_sound` (proved in\n\
         -- `Alkahest.Kernel.Pde`) turns `checkPde u op n w = true` into the\n\
         -- statement; {}.\n\
         noncomputable def fexU ({binders} : ℝ) : ℝ :=\n  {}\n\n\
         open Alkahest.Kernel in\n\
         theorem fex_pde ({binders} : ℝ) :\n    \
         {stmt} = 0 :=\n  \
         pde_sound (u := {})\n    \
         (op := {})\n    \
         (n := {d})\n    \
         (w := {})\n    \
         ({}) [{}] rfl\n",
        kernel_header(),
        pool.display(operator),
        pool.display(solution),
        fr.trust,
        u_ae.lean_string(&names),
        u_ae.data_string(),
        op_ae.data_string(),
        proof.data(),
        fr.decide,
        coord_names.join(", "),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn emit(f: impl Fn(&ExprPool, ExprId) -> (ExprId, ExprId)) -> Result<String, &'static str> {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let (big_f, small_f) = f(&pool, x);
        emit_integration_kernel_cert(big_f, small_f, x, &pool, KernelCheckMode::Kernel)
    }

    #[test]
    fn polynomial_times_trig_needs_no_relation() {
        let src = emit(|p, x| {
            let two = p.integer(2);
            let m1 = p.integer(-1);
            let x2 = p.pow(x, two);
            let s = p.func("sin", vec![x]);
            let c = p.func("cos", vec![x]);
            let big = p.add(vec![
                p.mul(vec![x2, m1, c]),
                p.mul(vec![x, two, s]),
                p.mul(vec![two, c]),
            ]);
            (big, p.mul(vec![x2, s]))
        })
        .expect("certificate");
        assert!(src.contains("import Alkahest.Kernel"), "{src}");
        assert!(src.contains("integral_sound"), "{src}");
        assert!(src.contains("by decide +kernel"), "{src}");
        assert!(src.contains("(H := [])"), "{src}");
        assert!(!src.contains("sorry"), "{src}");
    }

    fn pde_syms(p: &ExprPool, d: usize) -> (Vec<ExprId>, ExprId, ExprId, Vec<ExprId>) {
        let xs: Vec<ExprId> = (0..d)
            .map(|i| p.symbol(format!("x{i}"), Domain::Real))
            .collect();
        let u = p.symbol("u", Domain::Real);
        let lap = p.symbol("lap_u", Domain::Real);
        let g: Vec<ExprId> = (0..d)
            .map(|i| p.symbol(format!("u_x{i}"), Domain::Real))
            .collect();
        (xs, u, lap, g)
    }

    #[test]
    fn pde_certificate_for_an_eigenfunction() {
        // −Δu + |x|² u − d u = 0 for u = exp(−|x|²/2), d = 3.
        let p = ExprPool::new();
        let (xs, u, lap, _g) = pde_syms(&p, 3);
        let r2 = p.add(xs.iter().map(|&x| p.pow(x, p.integer(2))).collect());
        let sol = p.func("exp", vec![p.mul(vec![p.rational(-1, 2), r2])]);
        let op = p.add(vec![
            p.mul(vec![p.integer(-1), lap]),
            p.mul(vec![r2, u]),
            p.mul(vec![p.integer(-3), u]),
        ]);
        let src = emit_pde_kernel_cert(op, sol, &xs, u, lap, &[], &p, KernelCheckMode::Kernel)
            .expect("certificate");
        assert!(src.contains("pde_sound"), "{src}");
        assert!(src.contains("by decide +kernel"), "{src}");
        assert!(
            src.contains("noncomputable def fexU (x0 x1 x2 : ℝ)"),
            "{src}"
        );
        assert!(
            src.contains("deriv (deriv (fun (y : ℝ) => fexU y x1 x2)) x0"),
            "{src}"
        );
        assert!(!src.contains("sorry"), "{src}");
        // A wrong eigenvalue is not certified.
        let bad = p.add(vec![
            p.mul(vec![p.integer(-1), lap]),
            p.mul(vec![r2, u]),
            p.mul(vec![p.integer(-2), u]),
        ]);
        assert!(
            emit_pde_kernel_cert(bad, sol, &xs, u, lap, &[], &p, KernelCheckMode::Kernel).is_err()
        );
    }

    #[test]
    fn pde_certificate_uses_exp_relations_and_refuses_domains() {
        // Cubic Schrödinger with u = exp(S/2)/3, S = cos x0 + cos x1, and
        // V = −exp(S)/9 + (sin² x0 + sin² x1)/4 − S/2 (the FEX benchmark at
        // d = 2): needs exp(S) = exp(S/2)².
        let p = ExprPool::new();
        let (xs, u, lap, g) = pde_syms(&p, 2);
        let s_ = p.add(xs.iter().map(|&x| p.func("cos", vec![x])).collect());
        let half = p.rational(1, 2);
        let sol = p.mul(vec![
            p.rational(1, 3),
            p.func("exp", vec![p.mul(vec![half, s_])]),
        ]);
        let sin2 = p.add(
            xs.iter()
                .map(|&x| p.pow(p.func("sin", vec![x]), p.integer(2)))
                .collect(),
        );
        let v = p.add(vec![
            p.mul(vec![p.rational(-1, 9), p.func("exp", vec![s_])]),
            p.mul(vec![p.rational(1, 4), sin2]),
            p.mul(vec![p.rational(-1, 2), s_]),
        ]);
        let op = p.add(vec![
            p.mul(vec![p.integer(-1), lap]),
            p.pow(u, p.integer(3)),
            p.mul(vec![v, u]),
        ]);
        let src = emit_pde_kernel_cert(op, sol, &xs, u, lap, &[], &p, KernelCheckMode::Kernel)
            .expect("certificate");
        assert!(src.contains("expAdd"), "{src}");
        // log x0 is not differentiable everywhere: withheld, not assumed.
        let sol2 = p.func("log", vec![xs[0]]);
        let op2 = p.add(vec![p.mul(vec![xs[0], g[0]]), p.integer(-1)]);
        assert_eq!(
            emit_pde_kernel_cert(op2, sol2, &xs, u, lap, &g, &p, KernelCheckMode::Kernel)
                .unwrap_err(),
            WITHHELD_CONSTANT_SIGN
        );
    }

    #[test]
    fn log_states_its_domain() {
        let src = emit(|p, x| (p.func("log", vec![x]), p.pow(x, p.integer(-1)))).expect("cert");
        assert!(src.contains("(h1 : (x : ℝ) ≠ 0)"), "{src}");
        assert!(src.contains("(H := [(.ne (.var (nat_lit 0)))])"), "{src}");
    }

    #[test]
    fn double_angle_uses_a_relation() {
        let src = emit(|p, x| {
            let c2 = p.func("cos", vec![p.mul(vec![x, p.integer(2)])]);
            (
                p.mul(vec![c2, p.rational(-1, 4)]),
                p.mul(vec![p.func("sin", vec![x]), p.func("cos", vec![x])]),
            )
        })
        .expect("cert");
        assert!(src.contains(".sinAdd") || src.contains(".cosAdd"), "{src}");
    }

    #[test]
    fn sqrt_answer_derives_positivity() {
        // ∫ x·√(x²+1) = (x²+1)^(3/2) / 3: x² + 1 > 0 is derived, not assumed.
        let src = emit(|p, x| {
            let v = p.add(vec![p.pow(x, p.integer(2)), p.integer(1)]);
            let big = p.mul(vec![p.rational(1, 3), p.pow(v, p.rational(3, 2))]);
            (big, p.mul(vec![x, p.func("sqrt", vec![v])]))
        })
        .expect("cert");
        assert!(src.contains("(H := [])"), "{src}");
        assert!(src.contains(".sqrt"), "{src}");
    }

    #[test]
    fn wrong_antiderivative_is_withheld() {
        let r = emit(|p, x| (p.func("sin", vec![x]), p.func("sin", vec![x])));
        assert_eq!(r, Err(WITHHELD_IDENTITY));
    }

    #[test]
    fn native_mode_is_marked() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let src = emit_integration_kernel_cert(
            pool.func("exp", vec![x]),
            pool.func("exp", vec![x]),
            x,
            &pool,
            KernelCheckMode::Native,
        )
        .expect("cert");
        assert!(src.contains("by native_decide"), "{src}");
    }

    #[test]
    fn special_functions_fall_back() {
        let r = emit(|p, x| (p.func("Ei", vec![x]), p.func("exp", vec![x])));
        assert_eq!(r, Err(super::super::hasderiv::WITHHELD_SPECIAL_FUNCTION));
    }

    #[test]
    fn ode_solution_certifies() {
        // y'' + y = 0, y = C1 sin x + C2 cos x
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let yp = pool.symbol("yp", Domain::Real);
        let ypp = pool.symbol("ypp", Domain::Real);
        let c1 = pool.symbol("C1", Domain::Real);
        let c2 = pool.symbol("C2", Domain::Real);
        let sol = pool.add(vec![
            pool.mul(vec![c1, pool.func("sin", vec![x])]),
            pool.mul(vec![c2, pool.func("cos", vec![x])]),
        ]);
        let eq = pool.add(vec![ypp, y]);
        let src = emit_ode_solution_kernel_cert(
            eq,
            x,
            y,
            &[yp, ypp],
            sol,
            &pool,
            KernelCheckMode::Kernel,
        )
        .expect("cert");
        assert!(src.contains("ode_sound"), "{src}");
        assert!(src.contains("(C1 : ℝ) (C2 : ℝ)"), "{src}");
        // a non-solution is withheld
        let bad = pool.mul(vec![c1, pool.func("exp", vec![x])]);
        let r = emit_ode_solution_kernel_cert(
            eq,
            x,
            y,
            &[yp, ypp],
            bad,
            &pool,
            KernelCheckMode::Kernel,
        );
        assert_eq!(r, Err(WITHHELD_IDENTITY));
    }

    #[test]
    fn mirror_derivative_matches_lean_rules() {
        // D (x^3) = 3 * x^2 * 1, as the Lean `D` builds it.
        let e = rc(AE::Pow(rc(AE::Var(0)), 3));
        let d = deriv(&e, 0);
        assert_eq!(
            d.data_string(),
            "(.mul (.mul (.lit (nat_lit 3)) (.pow (.var (nat_lit 0)) (nat_lit 2))) .one)"
        );
        let mut t = Table::default();
        let p = t.norm(&d).unwrap();
        let mut want = Poly::zero();
        want.add_term(vec![(0, 2)], Rational::from(3));
        assert_eq!(p, want);
    }
}
