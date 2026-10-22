//! Complex-arithmetic fallback for real compiled evaluation.
//!
//! A closed form can be real-valued while some of its intermediates are not:
//! Cardano's formula for a cubic with three real roots (casus irreducibilis),
//! `cos(t·√ω²)` at a point where `ω² < 0` (it is `cosh(t·√−ω²)` there), or a
//! ZOH matrix built from those eigenvalues. Real `f64` arithmetic takes
//! `√(negative)` to NaN, and before this module `compile_expr` handed that NaN
//! back as the value — silently, at points where the true value is a finite
//! real number.
//!
//! [`ComplexFallback`] is the expression flattened into a straight-line
//! program over [`ComplexF64`] (the principal branches of
//! [`crate::eval::eval_complex_f64`]). A compiled function runs its real code
//! first; only a point whose real result is NaN is re-evaluated here. Along
//! with each value the program carries a **running bound on its rounding
//! error** (first-order, with generous constants), so the decision at the end
//! is made against the arithmetic that was actually done:
//!
//! * the imaginary part is within a small multiple of its error bound — it is
//!   indistinguishable from zero at double precision, and the real part is the
//!   value (`Some(re)`);
//! * the imaginary part is clearly larger — the value at that point is not a
//!   real number (`log(−1)`, `√−2`, `asin 2`), and NaN stays;
//! * the bound is not small compared with the values the evaluation went
//!   through (an argument whose error disk straddles a branch cut, a
//!   logarithm of something indistinguishable from zero) — no claim either
//!   way, and NaN stays.
//!
//! NaN therefore means "not a real number at this point, or not decidable in
//! double precision", never "real, but the real arithmetic could not reach
//! it". A node the complex evaluator has no kernel for (Piecewise, special
//! functions, `RootSum`, …) means no fallback is built, and the real code's
//! answer stands as before.

use crate::eval::ComplexF64;
use crate::kernel::{integer_to_f64, rational_to_f64, ExprData, ExprId, ExprPool};
use std::collections::HashMap;

/// Rounding unit used in the bounds (`2u`, deliberately generous).
const U: f64 = f64::EPSILON;
/// The imaginary part is accepted as zero when `|im| ≤ ACCEPT · err`.
const ACCEPT: f64 = 16.0;
/// The error bound must be below `SCALE_GUARD · (largest magnitude seen)`.
const SCALE_GUARD: f64 = 1.0 / ((1u64 << 26) as f64);

#[derive(Clone, Copy, Debug)]
enum Un {
    Sin,
    Cos,
    Tan,
    Exp,
    Sinh,
    Cosh,
    Abs,
    Log,
    Sqrt,
    Acos,
    Asin,
    Atan,
    Re,
    Im,
    Conj,
}

impl Un {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "sin" => Un::Sin,
            "cos" => Un::Cos,
            "tan" => Un::Tan,
            "exp" => Un::Exp,
            "sinh" => Un::Sinh,
            "cosh" => Un::Cosh,
            "abs" => Un::Abs,
            "log" => Un::Log,
            "sqrt" => Un::Sqrt,
            "acos" => Un::Acos,
            "asin" => Un::Asin,
            "atan" => Un::Atan,
            "re" => Un::Re,
            "im" => Un::Im,
            "conjugate" => Un::Conj,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
enum Op {
    Const(Val),
    Input(usize),
    Add(Box<[u32]>),
    Mul(Box<[u32]>),
    PowInt(u32, i64),
    Pow(u32, u32),
    Unary(Un, u32),
}

/// A value, an absolute bound on its distance from the exact value of the
/// node at the (exact) input point, and whether the exact value is known to be
/// real (imaginary part exactly zero). The last decides how a branch cut is
/// treated: an exactly-real negative number has the principal square root
/// `+i√|x|`, while a value with a rounding-level imaginary part next to the
/// negative axis could sit on either side of the cut.
#[derive(Clone, Copy, Debug)]
struct Val {
    z: ComplexF64,
    err: f64,
    real: bool,
}

impl Val {
    fn new(z: ComplexF64, err: f64, real: bool) -> Self {
        // An exactly-real value carries `+0` as its imaginary part, so that a
        // later branch cut reads it from above — the principal value.
        let z = if real { ComplexF64::new(z.re, 0.0) } else { z };
        Val { z, err, real }
    }
    fn abs(&self) -> f64 {
        self.z.re.hypot(self.z.im)
    }
}

fn cabs(z: ComplexF64) -> f64 {
    z.re.hypot(z.im)
}

/// A real-valued expression flattened for complex evaluation at points where
/// its real evaluation gave NaN. See the module docs.
#[derive(Clone, Debug)]
pub(crate) struct ComplexFallback {
    ops: Vec<Op>,
    n_inputs: usize,
}

impl ComplexFallback {
    /// Flatten `expr` over `inputs`, or `None` when its real NaN cannot hide
    /// a real value (see [`Self::can_return_to_the_reals`]) or it contains a
    /// node with no complex kernel here.
    pub(crate) fn build(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> Option<Self> {
        let (ops, _) = flatten(&[expr], inputs, pool)?;
        let fb = ComplexFallback {
            ops,
            n_inputs: inputs.len(),
        };
        fb.can_return_to_the_reals().then_some(fb)
    }
}

/// Flatten the DAG under `roots` into one operation list (shared nodes once,
/// children before parents); also returns each root's operation index.
/// `None` when a node has no complex kernel here.
fn flatten(roots: &[ExprId], inputs: &[ExprId], pool: &ExprPool) -> Option<(Vec<Op>, Vec<u32>)> {
    let mut index: HashMap<ExprId, u32> = HashMap::new();
    let mut ops: Vec<Op> = Vec::new();
    for &root in roots {
        // Iterative post-order: (node, children pushed?)
        let mut stack: Vec<(ExprId, bool)> = vec![(root, false)];
        while let Some((id, expanded)) = stack.pop() {
            if index.contains_key(&id) {
                continue;
            }
            let data = pool.get(id);
            if !expanded {
                stack.push((id, true));
                match &data {
                    ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
                        for &a in args.iter().rev() {
                            stack.push((a, false));
                        }
                    }
                    ExprData::Pow { base, exp } => {
                        stack.push((*exp, false));
                        stack.push((*base, false));
                    }
                    _ => {}
                }
                continue;
            }
            let ix = |e: &ExprId| -> Option<u32> { index.get(e).copied() };
            let op = match &data {
                ExprData::Integer(n) => {
                    let v = integer_to_f64(&n.0);
                    let exact = n.0.significant_bits() <= 53;
                    Op::Const(Val::new(
                        ComplexF64::new(v, 0.0),
                        if exact { 0.0 } else { U * v.abs() },
                        true,
                    ))
                }
                ExprData::Rational(r) => {
                    let v = rational_to_f64(&r.0);
                    Op::Const(Val::new(ComplexF64::new(v, 0.0), U * v.abs(), true))
                }
                ExprData::Float(f) => {
                    let v = f.inner.to_f64();
                    Op::Const(Val::new(ComplexF64::new(v, 0.0), U * v.abs(), true))
                }
                ExprData::Symbol { .. } => {
                    if let Some(k) = inputs.iter().position(|&i| i == id) {
                        Op::Input(k)
                    } else if pool.is_imaginary_unit(id) {
                        Op::Const(Val::new(ComplexF64::new(0.0, 1.0), 0.0, false))
                    } else if let Some(v) = crate::eval::symbols::constant_value_of(id, pool) {
                        Op::Const(Val::new(ComplexF64::new(v, 0.0), U * v.abs(), true))
                    } else {
                        return None;
                    }
                }
                ExprData::Add(args) => {
                    Op::Add(args.iter().map(ix).collect::<Option<Vec<_>>>()?.into())
                }
                ExprData::Mul(args) => {
                    Op::Mul(args.iter().map(ix).collect::<Option<Vec<_>>>()?.into())
                }
                ExprData::Pow { base, exp } => {
                    let b = ix(base)?;
                    let int_exp = match pool.get(*exp) {
                        ExprData::Integer(n) => Some(n.0.to_i64()?),
                        ExprData::Rational(r) if *r.0.denom() == 1 => Some(r.0.numer().to_i64()?),
                        _ => None,
                    };
                    match int_exp {
                        // (`−i64::MIN` does not exist; that power is left to the
                        // real code.)
                        Some(i64::MIN) => return None,
                        Some(n) => Op::PowInt(b, n),
                        None => Op::Pow(b, ix(exp)?),
                    }
                }
                ExprData::Func { name, args } if args.len() == 1 => {
                    Op::Unary(Un::from_name(name)?, ix(&args[0])?)
                }
                _ => return None,
            };
            index.insert(id, ops.len() as u32);
            ops.push(op);
        }
    }
    let roots = roots
        .iter()
        .map(|r| index.get(r).copied())
        .collect::<Option<Vec<_>>>()?;
    Some((ops, roots))
}

impl ComplexFallback {
    /// Can a non-real intermediate flow into an operation that may bring it
    /// back to the real axis? Only then can the real code's NaN hide a real
    /// value. A non-real value added to reals, multiplied by reals, inverted,
    /// or passed to `√`/`log`/`acos`/`asin`/`atan` stays non-real, so
    /// `√x + 1` or `log(x)` get no fallback — their NaN already means "not
    /// real". What does: two non-real terms of a sum or factors of a product
    /// (`A + B` in Cardano, `√x·√x`), an integer power `≥ 2` of one (`(√x)²`),
    /// a non-integer power of one, and `sin`/`cos`/`tan`/`exp`/`sinh`/`cosh`/
    /// `abs`/`re`/`im` of one (`cos(t·√ω²)`).
    ///
    /// Input-free subtrees are evaluated here, so `√2` is known real and
    /// `√−1` known not to be.
    fn can_return_to_the_reals(&self) -> bool {
        // Per op: its value when it does not depend on an input, and whether
        // it may be non-real.
        let mut konst: Vec<Option<Val>> = Vec::with_capacity(self.ops.len());
        let mut nonreal: Vec<bool> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let children: Vec<u32> = match op {
                Op::Const(_) | Op::Input(_) => Vec::new(),
                Op::Add(a) | Op::Mul(a) => a.to_vec(),
                Op::PowInt(b, _) => vec![*b],
                Op::Pow(b, e) => vec![*b, *e],
                Op::Unary(_, a) => vec![*a],
            };
            let is_const = !matches!(op, Op::Input(_))
                && children.iter().all(|&c| konst[c as usize].is_some());
            if is_const {
                let v = step(op, &konst, &[]);
                // An input-free value the evaluator cannot compute is treated
                // as possibly non-real (and then as an input).
                nonreal.push(v.map_or(true, |v| !v.real && v.z.im != 0.0));
                konst.push(v);
                continue;
            }
            konst.push(None);
            let nr = |c: &u32| nonreal[*c as usize];
            let count = children.iter().filter(|c| nr(c)).count();
            let (may_be_nonreal, returns) = match op {
                Op::Const(_) | Op::Input(_) => (false, false),
                Op::Add(_) | Op::Mul(_) => (count > 0, count >= 2),
                Op::PowInt(_, n) => (count > 0, count > 0 && n.unsigned_abs() >= 2),
                // `c^w` with a positive constant base and a real exponent
                // (`2^x`, `e^x`) stays real.
                Op::Pow(b, e)
                    if !nr(e) && konst[*b as usize].is_some_and(|v| v.real && v.z.re > 0.0) =>
                {
                    (false, false)
                }
                Op::Pow(..) => (true, count > 0),
                Op::Unary(f, _) => match f {
                    Un::Log | Un::Sqrt | Un::Acos | Un::Asin => (true, false),
                    Un::Atan | Un::Conj => (count > 0, false),
                    Un::Abs | Un::Re | Un::Im => (false, count > 0),
                    Un::Sin | Un::Cos | Un::Tan | Un::Exp | Un::Sinh | Un::Cosh => {
                        (count > 0, count > 0)
                    }
                },
            };
            if returns {
                return true;
            }
            nonreal.push(may_be_nonreal);
        }
        false
    }

    /// The real value at `inputs` when the complex evaluation shows the
    /// imaginary part is zero up to rounding; `None` otherwise (see the
    /// module docs).
    pub(crate) fn eval_real(&self, inputs: &[f64]) -> Option<f64> {
        debug_assert_eq!(inputs.len(), self.n_inputs);
        let mut vals: Vec<Val> = Vec::with_capacity(self.ops.len());
        let mut scale = 0.0f64;
        for op in &self.ops {
            let v = step(op, &vals, inputs)?;
            scale = scale.max(v.abs());
            vals.push(v);
        }
        let out = vals.last()?;
        if out.real {
            return Some(out.z.re);
        }
        if !(out.err.is_finite() && out.err <= SCALE_GUARD * scale) {
            return None;
        }
        (out.z.im.abs() <= ACCEPT * out.err).then_some(out.z.re)
    }
}

/// One operation over the values of earlier ones (`vals` may hold `None`
/// for an operation not computed, which then fails the step).
fn step<V: AsVal>(op: &Op, vals: &[V], inputs: &[f64]) -> Option<Val> {
    let at = |i: u32| vals[i as usize].as_val();
    let v = match op {
        Op::Const(v) => *v,
        Op::Input(k) => Val::new(ComplexF64::new(*inputs.get(*k)?, 0.0), 0.0, true),
        Op::Add(args) => {
            let mut z = ComplexF64::ZERO;
            let mut err = 0.0;
            let mut mag = 0.0;
            let mut real = true;
            for &a in args.iter() {
                let a = at(a)?;
                z = z.add(a.z);
                err += a.err;
                mag += a.abs();
                real &= a.real;
            }
            err += U * (args.len() as f64) * mag;
            Val::new(z, err, real)
        }
        Op::Mul(args) => {
            let mut acc = Val::new(ComplexF64::ONE, 0.0, true);
            for &a in args.iter() {
                acc = mul(acc, at(a)?);
            }
            acc
        }
        Op::PowInt(b, n) => pow_int(at(*b)?, *n)?,
        Op::Pow(b, e) => pow(at(*b)?, at(*e)?)?,
        Op::Unary(f, a) => unary(*f, at(*a)?)?,
    };
    if !(v.z.re.is_finite() && v.z.im.is_finite()) || v.err.is_nan() {
        return None;
    }
    Some(v)
}

trait AsVal {
    fn as_val(&self) -> Option<Val>;
}
impl AsVal for Val {
    fn as_val(&self) -> Option<Val> {
        Some(*self)
    }
}
impl AsVal for Option<Val> {
    fn as_val(&self) -> Option<Val> {
        *self
    }
}

fn mul(a: Val, b: Val) -> Val {
    let z = a.z.mul(b.z);
    let (ma, mb) = (a.abs(), b.abs());
    let err = ma * b.err + mb * a.err + a.err * b.err + 2.0 * U * ma * mb;
    Val::new(z, err, a.real && b.real)
}

/// `(1 + r)^n − 1` for `r ≥ 0`, `n ≥ 0`, without cancellation.
fn growth(r: f64, n: f64) -> f64 {
    (n * r.ln_1p()).exp_m1()
}

fn pow_int(a: Val, n: i64) -> Option<Val> {
    let z = a.z.powi(n).ok()?;
    let m = a.abs();
    let k = n.unsigned_abs() as f64;
    let bits = 64.0 - (n.unsigned_abs().leading_zeros() as f64);
    let rounding = U * 2.0 * (bits + 1.0) * cabs(z);
    let prop = if n == 0 || a.err == 0.0 {
        0.0
    } else if n > 0 {
        if m == 0.0 {
            a.err.powf(k)
        } else {
            m.powf(k) * growth(a.err / m, k)
        }
    } else if a.err >= m {
        f64::INFINITY
    } else {
        // |1/(a+δ)^k − 1/a^k| ≤ m^{-k}·((1 − e/m)^{-k} − 1)
        m.powf(-k) * (-k * (-a.err / m).ln_1p()).exp_m1()
    };
    Some(Val::new(z, prop + rounding, a.real))
}

/// Is the error disk of `a` touching the negative real axis (the cut of
/// `log`, `√` and non-integer powers) while `a` is not known to be real?
fn straddles_negative_axis(a: &Val) -> bool {
    !a.real && a.z.im.abs() <= a.err && a.z.re < a.err
}

fn log(a: Val) -> Option<Val> {
    let w = a.z.ln().ok()?;
    let m = a.abs();
    if a.err >= m || straddles_negative_axis(&a) || (a.real && a.z.re.abs() <= a.err) {
        return Some(Val::new(w, f64::INFINITY, false));
    }
    let err = a.err / (m - a.err) + U * (1.0 + cabs(w));
    Some(Val::new(w, err, a.real && a.z.re > 0.0))
}

fn exp(a: Val) -> Val {
    let w = a.z.exp();
    let mw = cabs(w);
    let err = mw * a.err.exp_m1() + U * mw * (2.0 + a.abs());
    Val::new(w, err, a.real)
}

fn pow(b: Val, e: Val) -> Option<Val> {
    // An integer-valued real exponent is a repeated product (as in the
    // complex evaluator).
    if e.real && e.err == 0.0 && e.z.re.fract() == 0.0 && e.z.re.abs() < 9.0e15 {
        return pow_int(b, e.z.re as i64);
    }
    let z = b.z.powc(e.z).ok()?;
    if b.z.re == 0.0 && b.z.im == 0.0 {
        // 0^w with real w > 0 (anything else was refused by `powc`).
        let err = if b.err == 0.0 {
            0.0
        } else if e.real && e.z.re > 0.0 {
            b.err.powf(e.z.re) * (1.0 + e.err)
        } else {
            f64::INFINITY
        };
        return Some(Val::new(z, err, e.real));
    }
    // z = exp(e · log b): compose the bounds; the value is `powc`'s own.
    let l = log(b)?;
    let m = mul(e, l);
    let x = exp(m);
    let real = b.real && e.real && b.z.re > 0.0 && b.err < b.z.re;
    Some(Val::new(z, x.err, real))
}

fn unary(f: Un, a: Val) -> Option<Val> {
    let m = a.abs();
    let z = a.z;
    Some(match f {
        Un::Exp => exp(a),
        Un::Log => log(a)?,
        Un::Sin | Un::Cos => {
            let w = if matches!(f, Un::Sin) {
                z.sin()
            } else {
                z.cos()
            };
            let ch = (z.im.abs() + a.err).cosh();
            let err = a.err * ch + U * ch * (2.0 + m);
            Val::new(w, err, a.real)
        }
        Un::Sinh | Un::Cosh => {
            let w = if matches!(f, Un::Sinh) {
                z.sinh()
            } else {
                z.cosh()
            };
            let ch = (z.re.abs() + a.err).cosh();
            let err = a.err * ch + U * ch * (2.0 + m);
            Val::new(w, err, a.real)
        }
        Un::Tan => {
            let w = z.tan().ok()?;
            let d = 1.0 + cabs(w).powi(2);
            let err = if 4.0 * a.err * d >= 1.0 {
                f64::INFINITY
            } else {
                2.0 * a.err * d + U * d * (2.0 + m)
            };
            Val::new(w, err, a.real)
        }
        Un::Abs => Val::new(ComplexF64::new(m, 0.0), a.err + U * m, true),
        Un::Re => Val::new(ComplexF64::new(z.re, 0.0), a.err, true),
        Un::Im => Val::new(ComplexF64::new(z.im, 0.0), a.err, true),
        Un::Conj => Val::new(ComplexF64::new(z.re, -z.im), a.err, a.real),
        Un::Sqrt => {
            let w = z.sqrt().ok()?;
            let mw = cabs(w);
            if a.err == 0.0 {
                Val::new(w, U * mw, a.real && z.re >= 0.0)
            } else if 2.0 * a.err >= m {
                // Any value of either branch over the error disk.
                Val::new(w, (m + a.err).sqrt() + mw, false)
            } else if straddles_negative_axis(&a) {
                Val::new(w, f64::INFINITY, false)
            } else {
                let err = a.err / (2.0 * (m - a.err).sqrt()) + U * mw;
                Val::new(w, err, a.real && z.re > a.err)
            }
        }
        Un::Acos | Un::Asin => {
            let w = if matches!(f, Un::Acos) {
                z.acos()
            } else {
                z.asin()
            }
            .ok()?;
            let mw = cabs(w);
            if a.real {
                let x = z.re.abs();
                if x + a.err < 1.0 {
                    let err = a.err / (1.0 - (x + a.err).powi(2)).sqrt() + U * (1.0 + mw);
                    Val::new(w, err, true)
                } else if x - a.err > 1.0 {
                    let err = a.err / ((x - a.err).powi(2) - 1.0).sqrt() + U * (1.0 + mw);
                    Val::new(w, err, false)
                } else {
                    // Next to ±1, where both functions have a square-root
                    // branch point.
                    Val::new(w, 4.0 * (a.err * (1.0 + x)).sqrt() + U * (1.0 + mw), false)
                }
            } else {
                let on_cut = z.im.abs() <= a.err && z.re.abs() + a.err > 1.0;
                let one_minus_z2 = ComplexF64::ONE.add(ComplexF64::new(-1.0, 0.0).mul(z.mul(z)));
                let dmin = cabs(one_minus_z2) - a.err * (2.0 * m + a.err);
                if on_cut || dmin <= 0.0 {
                    Val::new(w, f64::INFINITY, false)
                } else {
                    Val::new(w, a.err / dmin.sqrt() + U * (1.0 + mw), false)
                }
            }
        }
        Un::Atan => {
            let w = z.atan().ok()?;
            let mw = cabs(w);
            if a.real {
                let lo = (z.re.abs() - a.err).max(0.0);
                Val::new(w, a.err / (1.0 + lo * lo) + U * (1.0 + mw), true)
            } else {
                let on_cut = z.re.abs() <= a.err && z.im.abs() + a.err > 1.0;
                let one_plus_z2 = ComplexF64::ONE.add(z.mul(z));
                let dmin = cabs(one_plus_z2) - a.err * (2.0 * m + a.err);
                if on_cut || dmin <= 0.0 {
                    Val::new(w, f64::INFINITY, false)
                } else {
                    Val::new(w, a.err / dmin + U * (1.0 + mw), false)
                }
            }
        }
    })
}

// ---------------------------------------------------------------------------
// C emission — the same program and the same decision, for `emit_c_expr`
// ---------------------------------------------------------------------------

impl Un {
    fn c_name(self) -> &'static str {
        match self {
            Un::Sin => "sin",
            Un::Cos => "cos",
            Un::Tan => "tan",
            Un::Exp => "exp",
            Un::Sinh => "sinh",
            Un::Cosh => "cosh",
            Un::Abs => "abs",
            Un::Log => "log",
            Un::Sqrt => "sqrt",
            Un::Acos => "acos",
            Un::Asin => "asin",
            Un::Atan => "atan",
            Un::Re => "re",
            Un::Im => "im",
            Un::Conj => "conj",
        }
    }
}

fn c_lit(v: f64) -> String {
    if v.is_nan() {
        "NAN".into()
    } else if v.is_infinite() {
        if v > 0.0 { "INFINITY" } else { "(-INFINITY)" }.into()
    } else {
        format!("{v:?}")
    }
}

impl ComplexFallback {
    /// Does the program reference the imaginary unit? The real C emitter has
    /// no spelling for it, so such an output is emitted as its complex
    /// program alone.
    pub(crate) fn has_imaginary_constant(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op, Op::Const(v) if v.z.im != 0.0))
    }
}

/// The complex fallbacks of a C function's outputs.
pub(crate) struct CFallbacks {
    /// Per output: `None` when it needs no fallback, `Some(complex_only)`
    /// when it has one — `complex_only` when the output is written with the
    /// imaginary unit and must be the complex evaluation alone.
    pub(crate) outputs: Vec<Option<bool>>,
    /// [`C_PRELUDE`] and `static double {name}(int k, double var0, …)`,
    /// evaluating output `k`'s fallback (empty when no output has one).
    pub(crate) code: String,
}

/// C for the fallbacks of `exprs`, as one table over their shared DAG (a
/// node common to several outputs is one entry). Output `k` reads only its
/// own subtree (per-node success flags and subtree magnitudes), so its value
/// is exactly [`ComplexFallback::eval_real`]'s for `exprs[k]` alone — `NAN`
/// where that is `None`.
pub(crate) fn emit_c_fallbacks(
    exprs: &[ExprId],
    inputs: &[ExprId],
    var_names: &[&str],
    name: &str,
    pool: &ExprPool,
) -> CFallbacks {
    let mut outputs: Vec<Option<bool>> = exprs
        .iter()
        .map(|&e| ComplexFallback::build(e, inputs, pool).map(|f| f.has_imaginary_constant()))
        .collect();
    let needed: Vec<usize> = (0..exprs.len()).filter(|&k| outputs[k].is_some()).collect();
    let roots: Vec<ExprId> = needed.iter().map(|&k| exprs[k]).collect();
    let Some((ops, root_ix)) = (!roots.is_empty())
        .then(|| flatten(&roots, inputs, pool))
        .flatten()
    else {
        outputs.iter_mut().for_each(|o| *o = None);
        return CFallbacks {
            outputs,
            code: String::new(),
        };
    };
    // The program as data — `{code, first child, child count, integer}`
    // per operation, the children in `args`, constants as `{re, im, err,
    // real}` quadruples — run by `ak_run` from the prelude: a few dozen bytes
    // per operation rather than a line of straight-line C each.
    let mut table = Vec::with_capacity(ops.len());
    let mut args: Vec<u32> = Vec::new();
    let mut consts: Vec<String> = Vec::new();
    for op in &ops {
        let (code, first, count, n) = match op {
            Op::Const(v) => {
                let k = consts.len() / 4;
                consts.extend([
                    c_lit(v.z.re),
                    c_lit(v.z.im),
                    c_lit(v.err),
                    if v.real { "1.0" } else { "0.0" }.to_string(),
                ]);
                ("AK_OP_CONST".to_string(), k, 0, 0)
            }
            Op::Input(k) => ("AK_OP_INPUT".to_string(), *k, 0, 0),
            _ => {
                let children: Vec<u32> = match op {
                    Op::Add(a) | Op::Mul(a) => a.to_vec(),
                    Op::PowInt(b, _) => vec![*b],
                    Op::Pow(b, e) => vec![*b, *e],
                    Op::Unary(_, a) => vec![*a],
                    Op::Const(_) | Op::Input(_) => unreachable!(),
                };
                let first = args.len();
                args.extend(&children);
                let (code, n) = match op {
                    Op::Add(_) => ("AK_OP_ADD".to_string(), 0),
                    Op::Mul(_) => ("AK_OP_MUL".to_string(), 0),
                    Op::PowInt(_, n) => ("AK_OP_POWINT".to_string(), *n),
                    Op::Pow(..) => ("AK_OP_POW".to_string(), 0),
                    Op::Unary(f, _) => (format!("AK_OP_{}", f.c_name().to_uppercase()), 0),
                    Op::Const(_) | Op::Input(_) => unreachable!(),
                };
                (code, first, children.len(), n)
            }
        };
        table.push(format!("{{{code},{first},{count},{n}LL}}"));
    }
    let mut roots = vec!["-1".to_string(); exprs.len()];
    for (&k, &r) in needed.iter().zip(&root_ix) {
        roots[k] = r.to_string();
    }
    let or_zero = |v: Vec<String>| {
        if v.is_empty() {
            "0".to_string()
        } else {
            v.join(",")
        }
    };
    let params: String = var_names.iter().map(|n| format!(", double {n}")).collect();
    let n = ops.len();
    let mut s = String::from(C_PRELUDE);
    s.push_str(&format!(
        "static double {name}(int ak_out{params}) {{\n\
         \x20   static const ak_op ops[{n}] = {{{}}};\n\
         \x20   static const int args[] = {{{}}};\n\
         \x20   static const double consts[] = {{{}}};\n\
         \x20   static const int roots[{}] = {{{}}};\n\
         \x20   const double in[] = {{{}}};\n\
         \x20   ak_cv v[{n}];\n\
         \x20   int ok[{n}];\n\
         \x20   double sc[{n}];\n\
         \x20   if (ak_out < 0 || ak_out >= {} || roots[ak_out] < 0) return NAN;\n\
         \x20   return ak_run(ops, args, consts, in, roots[ak_out], v, ok, sc);\n\
         }}\n",
        table.join(","),
        or_zero(args.iter().map(u32::to_string).collect()),
        or_zero(consts),
        exprs.len(),
        roots.join(","),
        or_zero(var_names.iter().map(|v| v.to_string()).collect()),
        exprs.len(),
    ));
    CFallbacks { outputs, code: s }
}

/// The C runtime of [`emit_c_fallbacks`]: [`ComplexF64`]'s principal
/// branches and this module's error bounds, transliterated operation for
/// operation so the C and the JIT make the same decision.
pub(crate) const C_PRELUDE: &str = include_str!("complex_fallback.c");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn fb(expr: ExprId, inputs: &[ExprId], pool: &ExprPool) -> ComplexFallback {
        ComplexFallback::build(expr, inputs, pool).expect("fallback built")
    }

    #[test]
    fn not_built_for_expressions_that_stay_real() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.add(vec![
            pool.func("sin", vec![x]),
            pool.pow(x, pool.integer(3)),
        ]);
        assert!(ComplexFallback::build(e, &[x], &pool).is_none());
    }

    #[test]
    fn cos_of_sqrt_of_negative_is_cosh() {
        // cos(t·√w2) at w2 = −4: cos(2it) = cosh(2t).
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let w2 = pool.symbol("w2", Domain::Real);
        let e = pool.func(
            "cos",
            vec![pool.mul(vec![t, pool.pow(w2, pool.rational(1, 2))])],
        );
        let f = fb(e, &[t, w2], &pool);
        let v = f.eval_real(&[0.7, -4.0]).unwrap();
        assert!((v - (1.4f64).cosh()).abs() < 1e-14, "{v}");
        // sin(t√w2)/√w2 likewise: sinh(2t)/2.
        let s = pool.mul(vec![
            pool.func(
                "sin",
                vec![pool.mul(vec![t, pool.pow(w2, pool.rational(1, 2))])],
            ),
            pool.pow(w2, pool.rational(-1, 2)),
        ]);
        let v = fb(s, &[t, w2], &pool).eval_real(&[0.7, -4.0]).unwrap();
        assert!((v - (1.4f64).sinh() / 2.0).abs() < 1e-14, "{v}");
    }

    #[test]
    fn genuine_non_real_values_stay_undecided() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let sq = pool.func("sqrt", vec![x]);
        // No way back to the reals: no fallback at all.
        for e in [
            pool.func("log", vec![x]),
            sq,
            pool.add(vec![sq, pool.integer(1)]),
            pool.pow(sq, pool.integer(-1)),
            pool.func("asin", vec![x]),
            pool.pow(x, pool.rational(1, 3)),
            pool.func("log", vec![pool.func("sqrt", vec![x])]),
            pool.mul(vec![pool.imaginary_unit(), x]),
        ] {
            assert!(
                ComplexFallback::build(e, &[x], &pool).is_none(),
                "{}",
                pool.display(e)
            );
        }
        // A way back exists, but at these points the value is not real.
        for (e, at) in [
            (pool.func("exp", vec![sq]), -1.0),
            (pool.add(vec![pool.func("cos", vec![sq]), sq]), -1.0),
            (pool.add(vec![pool.func("cos", vec![sq]), sq]), -1e-300),
            (
                pool.mul(vec![pool.pow(sq, pool.integer(2)), pool.imaginary_unit()]),
                -1.0,
            ),
            (pool.func("cos", vec![pool.func("asin", vec![x])]), f64::NAN),
            (
                pool.add(vec![
                    pool.func("cos", vec![pool.pow(x, pool.rational(1, 3))]),
                    pool.pow(x, pool.rational(1, 3)),
                ]),
                -8.0,
            ),
        ] {
            let f = fb(e, &[x], &pool);
            assert_eq!(f.eval_real(&[at]), None, "{} at {at}", pool.display(e));
        }
        // cos(asin 2) = ±i√3 is not real either; cos(acos 2) = 2 is.
        let f = fb(
            pool.func("cos", vec![pool.func("asin", vec![x])]),
            &[x],
            &pool,
        );
        assert_eq!(f.eval_real(&[2.0]), None);
        let f = fb(
            pool.func("cos", vec![pool.func("acos", vec![x])]),
            &[x],
            &pool,
        );
        let v = f.eval_real(&[2.0]).unwrap();
        assert!((v - 2.0).abs() < 1e-14, "{v}");
    }

    #[test]
    fn i_times_i_is_minus_one() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let s = pool.func("sqrt", vec![x]);
        let e = pool.mul(vec![s, s]);
        // √x·√x at x = −3 is −3 on the principal branch.
        let v = fb(e, &[x], &pool).eval_real(&[-3.0]).unwrap();
        assert!((v + 3.0).abs() < 1e-14);
    }

    #[test]
    fn casus_irreducibilis_cardano() {
        // x³ − 3x + 1 = 0 has three real roots; Cardano with p = −3, q = 1:
        // u = (−q/2 + √(q²/4 + p³/27))^{1/3}, root = u − p/(3u) = u + 1/u.
        let pool = ExprPool::new();
        let q = pool.symbol("q", Domain::Real);
        let half = pool.rational(1, 2);
        let disc = pool.add(vec![
            pool.mul(vec![pool.pow(q, pool.integer(2)), pool.rational(1, 4)]),
            pool.integer(-1),
        ]);
        let u = pool.pow(
            pool.add(vec![
                pool.mul(vec![pool.integer(-1), half, q]),
                pool.pow(disc, half),
            ]),
            pool.rational(1, 3),
        );
        let root = pool.add(vec![u, pool.pow(u, pool.integer(-1))]);
        let v = fb(root, &[q], &pool).eval_real(&[1.0]).unwrap();
        // The principal cube root gives 2·cos(2π/9 ·…): check it is a root.
        assert!((v * v * v - 3.0 * v + 1.0).abs() < 1e-12, "{v}");
    }

    /// `e^{−βt/2}(cos(t√ω²/… ) + …)`-shaped closed form: every tier, scalar
    /// and batch, gives the real value where `ω² < 0` and NaN only where the
    /// expression is genuinely not real.
    #[test]
    fn every_tier_and_batch_path_uses_the_fallback() {
        use crate::jit::{compile_many_with, compile_with, CompileConfig, CompileTier};
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let w2 = pool.symbol("w2", Domain::Real);
        let root = pool.pow(w2, pool.rational(1, 2));
        let osc = pool.add(vec![
            pool.func("cos", vec![pool.mul(vec![t, root])]),
            pool.mul(vec![
                pool.func("sin", vec![pool.mul(vec![t, root])]),
                pool.pow(root, pool.integer(-1)),
            ]),
        ]);
        let not_real = pool.func("log", vec![w2]);
        let want = |tv: f64, w: f64| -> f64 {
            if w < 0.0 {
                let k = (-w).sqrt();
                (tv * k).cosh() + (tv * k).sinh() / k
            } else {
                let k = w.sqrt();
                (tv * k).cos() + (tv * k).sin() / k
            }
        };
        #[allow(unused_mut)] // no native tier in a default build
        let mut tiers = vec![CompileTier::Interpreter];
        #[cfg(feature = "cranelift")]
        tiers.push(CompileTier::Cranelift);
        #[cfg(feature = "jit")]
        tiers.push(CompileTier::Llvm);
        let ts: Vec<f64> = (0..40).map(|j| 0.1 * j as f64).collect();
        let ws: Vec<f64> = (0..40).map(|j| -3.0 + 0.17 * j as f64).collect();
        for tier in tiers {
            let config = CompileConfig {
                expected_evals: None,
                force_tier: Some(tier),
            };
            let f = compile_with(osc, &[t, w2], &pool, config).unwrap();
            for (&tv, &w) in ts.iter().zip(&ws) {
                let got = f.call(&[tv, w]);
                assert!((got - want(tv, w)).abs() < 1e-12, "{tier:?} {tv} {w} {got}");
            }
            let mut out = vec![0.0; ts.len()];
            f.call_batch(&[&ts, &ws], &mut out);
            let mut flat = ts.clone();
            flat.extend_from_slice(&ws);
            let mut bulk = vec![0.0; ts.len()];
            f.call_bulk(&flat, &mut bulk);
            for j in 0..ts.len() {
                assert!(
                    (out[j] - want(ts[j], ws[j])).abs() < 1e-12,
                    "{tier:?} batch {j}"
                );
                assert_eq!(out[j].to_bits(), bulk[j].to_bits());
            }
            let g = compile_with(not_real, &[w2], &pool, config).unwrap();
            assert!(g.call(&[-1.0]).is_nan(), "{tier:?}");
            assert_eq!(g.call(&[1.0]), 0.0);

            let v = compile_many_with(&[osc, not_real], &[t, w2], &pool, config).unwrap();
            let mut o = [0.0; 2];
            v.call(&[0.5, -2.0], &mut o);
            assert!((o[0] - want(0.5, -2.0)).abs() < 1e-12, "{tier:?} {o:?}");
            assert!(o[1].is_nan());
            let mut o0 = vec![0.0; ts.len()];
            let mut o1 = vec![0.0; ts.len()];
            v.call_batch(&[&ts, &ws], &mut [&mut o0[..], &mut o1[..]]);
            for j in 0..ts.len() {
                assert!((o0[j] - want(ts[j], ws[j])).abs() < 1e-12);
                assert_eq!(o1[j].is_nan(), ws[j] < 0.0);
            }
        }
    }

    #[test]
    fn eval_interp_real_reads_the_real_value() {
        use crate::jit::{eval_interp_checked, eval_interp_real, InterpEvalError};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("cos", vec![pool.func("sqrt", vec![x])]);
        let env = HashMap::from([(x, -4.0)]);
        let v = eval_interp_real(e, &env, &pool).unwrap();
        assert!((v - 2f64.cosh()).abs() < 1e-14);
        // The plain checked evaluator keeps real `f64` semantics.
        assert!(eval_interp_checked(e, &env, &pool).is_err());
        let l = pool.func("log", vec![x]);
        assert!(matches!(
            eval_interp_real(l, &env, &pool),
            Err(InterpEvalError::NonFinite)
        ));
    }

    /// The emitted C is a transliteration of [`ComplexFallback::eval_real`];
    /// it has to at least name only what [`C_PRELUDE`] defines.
    #[test]
    fn emitted_c_calls_only_prelude_helpers() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let all = [
            "sin",
            "cos",
            "tan",
            "exp",
            "sinh",
            "cosh",
            "abs",
            "log",
            "sqrt",
            "acos",
            "asin",
            "atan",
            "re",
            "im",
            "conjugate",
        ];
        let terms: Vec<ExprId> = all.iter().map(|f| pool.func(*f, vec![x])).collect();
        let e = pool.add(vec![
            pool.mul(terms.clone()),
            pool.pow(x, pool.rational(1, 3)),
            pool.pow(x, pool.integer(-3)),
            pool.mul(vec![pool.imaginary_unit(), x]),
        ]);
        let code = emit_c_fallbacks(&[e], &[x], &["x"], "g", &pool).code;
        assert!(
            code.contains("static double g(int ak_out, double x)"),
            "{code}"
        );
        // Every operation code in the table is one `ak_run` dispatches on.
        for f in all {
            let name = if f == "conjugate" { "conj" } else { f };
            let code_name = format!("AK_OP_{}", name.to_uppercase());
            assert!(code.contains(&format!("{{{code_name},")), "{code_name}");
            assert!(C_PRELUDE.contains(&code_name), "{code_name}");
            assert!(
                C_PRELUDE.contains(&format!("case {code_name}:")) || code_name == "AK_OP_CONJ",
                "{code_name}"
            );
        }
        for op in [
            "AK_OP_ADD",
            "AK_OP_MUL",
            "AK_OP_POW",
            "AK_OP_POWINT",
            "AK_OP_CONST",
        ] {
            assert!(code.contains(&format!("{{{op},")), "{op}");
        }
    }

    /// The emitted C, compiled and run, agrees with [`ComplexFallback::eval_real`]
    /// (skipped without a system C compiler).
    #[test]
    fn emitted_c_matches_eval_real() {
        let Some(cc) = ["/usr/bin/gcc", "/usr/bin/cc"]
            .into_iter()
            .find(|c| std::path::Path::new(c).exists())
        else {
            return;
        };
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let w = pool.symbol("w", Domain::Real);
        let r = pool.pow(w, pool.rational(1, 2));
        let exprs = [
            pool.add(vec![
                pool.func("cos", vec![pool.mul(vec![t, r])]),
                pool.mul(vec![
                    pool.func("sin", vec![pool.mul(vec![t, r])]),
                    pool.pow(r, pool.integer(-1)),
                ]),
            ]),
            pool.func("exp", vec![r]),
            pool.mul(vec![r, r, pool.func("acos", vec![t])]),
        ];
        let c = emit_c_fallbacks(&exprs, &[t, w], &["t", "w"], "g", &pool).code;
        let dir = std::env::temp_dir().join(format!("ak_cfb_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (src, bin) = (dir.join("g.c"), dir.join("g"));
        let pts = [
            (0.7, -4.0),
            (1.3, 2.0),
            (0.2, -0.5),
            (2.5, -1e-3),
            (0.5, 0.0),
        ];
        let mut main = String::from("#include <math.h>\n#include <stdio.h>\n");
        main.push_str(&c);
        main.push_str("int main(void) {\n");
        for (tv, wv) in pts {
            for k in 0..exprs.len() {
                main.push_str(&format!("    printf(\"%a\\n\", g({k}, {tv:?}, {wv:?}));\n"));
            }
        }
        main.push_str("    return 0;\n}\n");
        std::fs::write(&src, main).unwrap();
        let out = std::process::Command::new(cc)
            .args(["-std=c99", "-O2", "-Wall", "-Werror", "-o"])
            .arg(&bin)
            .arg(&src)
            .arg("-lm")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let run = std::process::Command::new(&bin).output().unwrap();
        let text = String::from_utf8(run.stdout).unwrap();
        let mut lines = text.lines();
        for (tv, wv) in pts {
            for &e in &exprs {
                let line = lines.next().unwrap();
                let got = if line.contains("nan") {
                    f64::NAN
                } else {
                    parse_hex_float(line)
                };
                let want = ComplexFallback::build(e, &[t, w], &pool)
                    .and_then(|f| f.eval_real(&[tv, wv]))
                    .unwrap_or(f64::NAN);
                assert!(
                    (got.is_nan() && want.is_nan())
                        || (got - want).abs() <= 1e-14 * (1.0 + want.abs()),
                    "{} at ({tv}, {wv}): C {got} vs Rust {want}",
                    pool.display(e)
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `%a` output (`0x1.8p+1`, `-0x0p+0`).
    fn parse_hex_float(s: &str) -> f64 {
        let s = s.trim();
        let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
        let s = s.trim_start_matches("0x");
        let (mant, exp) = s.split_once('p').unwrap();
        let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
        let mut v = i64::from_str_radix(int, 16).unwrap() as f64;
        let mut scale = 1.0 / 16.0;
        for ch in frac.chars() {
            v += ch.to_digit(16).unwrap() as f64 * scale;
            scale /= 16.0;
        }
        let v = v * 2f64.powi(exp.parse::<i32>().unwrap());
        if neg {
            -v
        } else {
            v
        }
    }
}
