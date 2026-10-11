//! C source emission with common-subexpression elimination.
//!
//! [`emit_expr_c`](super::emit_expr_c) and
//! [`emit_expr_c_vec`](super::emit_expr_c_vec) lower the expression DAG into a
//! small hash-consed IR, count how often every IR node is referenced across
//! **all** outputs, and give each non-leaf node referenced at least twice a
//! `const` temporary, emitted in dependency order.  Everything referenced once
//! is inlined into its single consumer.
//!
//! # Why an IR rather than the pool directly
//!
//! The pool already hash-conses, so `sin(q0)` is one node however often it
//! appears.  The emitter used to walk the pool and inline every node whose C
//! text *looked* short, which in practice meant every parenthesised sum and
//! product — so a node shared a thousand times was printed a thousand times.
//! An `n = 4` pendulum's equations of motion came out as 320 kB of C, and a
//! Black–Scholes price + gradient + Hessian as 664 `sqrt` and 131 `log` calls.
//!
//! The rewrites below also create sharing the pool cannot see:
//!
//! * `x^(-1/2)`, `x^(1/2)` and `x^(3/2)` all lower to one `sqrt(x)` node;
//! * `x^2`, `x^3` and `x^4` share one `x * x` node, and `x^-2` divides by it;
//! * `a · b^-1 · c^-2` becomes `a / (b · (c · c))`, one division;
//! * `-1 · t` becomes a negation, and a negated term of a sum a subtraction.
//!
//! None of them reorders a sum or a product of three or more operands (two
//! commute exactly), so the rounding of every `+` and
//! `·` the interpreter performs is the rounding the emitted C performs.  The
//! remaining differences are the ones listed on `Builder::lower_func` and the
//! division/strength-reduction rewrites above, each within an ulp or two.
//!
//! # Constants
//!
//! `π` (the symbol `pi`, unless the caller binds it as an input) and `e`
//! (`exp(1)`) are emitted as the nearest `double`, written with enough digits
//! to round-trip — `3.141592653589793`, `2.718281828459045` — rather than as
//! `M_PI`, which is not ISO C.  Rationals are emitted as the correctly rounded
//! double, exactly as the interpreter rounds them.

use super::{c_double_literal, c_integer_literal, EmitCError};
use crate::kernel::expr::PredicateKind;
use crate::kernel::{integer_is_exact_f64, rational_to_f64, ExprData, ExprId, ExprPool};
use rug::Rational;
use std::collections::HashMap;

/// One node of the codegen IR.  Children are indices into `Builder::nodes`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Node {
    /// The `i`-th function parameter.
    Var(usize),
    /// A literal, already rendered (`2.0`, `-0.4`, `INFINITY`, `1`).
    Lit(String),
    /// An n-ary sum; `true` marks a subtracted term.
    Sum(Vec<(bool, usize)>),
    /// An n-ary product.
    Prod(Vec<usize>),
    Div(usize, usize),
    Neg(usize),
    /// A call to a `<math.h>` function or an emitted helper.
    Call(&'static str, Vec<usize>),
    /// `c ? a : b`.
    Select(usize, usize, usize),
    /// A comparison (`<`, `<=`, …) — an `int` in C.
    Cmp(&'static str, usize, usize),
    All(Vec<usize>),
    Any(Vec<usize>),
    Not(usize),
}

impl Node {
    fn children(&self) -> Vec<usize> {
        match self {
            Node::Var(_) | Node::Lit(_) => Vec::new(),
            Node::Sum(ts) => ts.iter().map(|&(_, t)| t).collect(),
            Node::Prod(xs) | Node::Call(_, xs) | Node::All(xs) | Node::Any(xs) => xs.clone(),
            Node::Div(a, b) | Node::Cmp(_, a, b) => vec![*a, *b],
            Node::Neg(a) | Node::Not(a) => vec![*a],
            Node::Select(c, a, b) => vec![*c, *a, *b],
        }
    }

    fn is_leaf(&self) -> bool {
        matches!(self, Node::Var(_) | Node::Lit(_))
    }

    fn is_bool(&self) -> bool {
        matches!(
            self,
            Node::Cmp(..) | Node::All(_) | Node::Any(_) | Node::Not(_)
        )
    }
}

/// Self-contained C helpers the emitter can call.  Each is guarded by a macro
/// so several emitted functions can be pasted into one translation unit.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Helper {
    LambertW0,
}

impl Helper {
    fn source(self) -> &'static str {
        match self {
            Helper::LambertW0 => LAMBERT_W0_C,
        }
    }
}

/// Principal-branch Lambert `W₀(x)` for `x ≥ −1/e`, `NAN` below it.
///
/// Winitzki's starting value away from the branch point, the branch-point
/// series `−1 + p − p²/3 + 11p³/72` (`p = √(2(ex + 1))`) near it, then Halley
/// iterations until the step is below `4·10⁻¹⁶·|w|` (at most 12 — Halley
/// converges cubically, so 3–4 are typical), each step scaled by `e^{−w}` so
/// nothing overflows up to `DBL_MAX`.  Measured against mpmath (200 bits) on
/// 9 000 points of `[−1/e + 10⁻³, DBL_MAX]`: relative error ≤ 1·10⁻¹⁵.  Within
/// `10⁻³` of `−1/e` the problem itself is ill-conditioned (`W′ → ∞`) and the
/// error grows towards `~10⁻¹¹` at `x = −1/e + 10⁻¹²`; `W₀(−1/e) = −1` exactly.
const LAMBERT_W0_C: &str = "\
#ifndef ALKAHEST_LAMBERT_W0_DEFINED
#define ALKAHEST_LAMBERT_W0_DEFINED
/* Principal-branch Lambert W0(x), x >= -1/e; NAN below the branch point.
 * Relative error <= 1e-15 for x >= -1/e + 1e-3 (Halley iteration). */
static double alkahest_lambert_w0(double x) {
    const double em = -0.36787944117144233; /* -1/e */
    double w;
    int i;
    if (isnan(x) || x < em) return NAN;
    if (x == 0.0) return x;
    if (x == em) return -1.0;
    if (isinf(x)) return x;
    if (x < -0.32) {
        double p = sqrt(2.0 * (2.718281828459045 * x + 1.0));
        w = -1.0 + p * (1.0 + p * (-1.0 / 3.0 + p * (11.0 / 72.0)));
    } else {
        double l = log1p(x);
        w = l * (1.0 - log1p(l) / (2.0 + l));
    }
    for (i = 0; i < 12; ++i) {
        /* f = (w e^w - x) / e^w, scaled so nothing overflows near DBL_MAX. */
        double f = w - x * exp(-w);
        double wp1 = w + 1.0;
        double t;
        if (wp1 == 0.0) break;
        t = f / (wp1 - (w + 2.0) * f / (2.0 * wp1));
        w -= t;
        if (fabs(t) <= 4e-16 * fabs(w)) break;
    }
    return w;
}
#endif
";

struct Builder<'a> {
    pool: &'a ExprPool,
    vars: HashMap<ExprId, usize>,
    nodes: Vec<Node>,
    intern: HashMap<Node, usize>,
    memo: HashMap<ExprId, usize>,
    pred_memo: HashMap<ExprId, usize>,
    helpers: Vec<Helper>,
}

/// A rational exponent that the strength reduction handles without `pow`.
fn is_small_exponent(q: &Rational) -> bool {
    let d = q.denom();
    let n = q.numer();
    (*d == 1 && n.cmp_abs(&rug::Integer::from(4)).is_le())
        || (*d == 2 && n.cmp_abs(&rug::Integer::from(7)).is_le())
}

fn exact_exponent(e: &ExprData) -> Option<Rational> {
    match e {
        ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

impl<'a> Builder<'a> {
    fn new(pool: &'a ExprPool, vars: &[ExprId]) -> Self {
        let mut map = HashMap::new();
        for (i, &v) in vars.iter().enumerate() {
            // The first occurrence wins, as the old `HashMap::insert` order
            // made the *last* one win; a duplicated input is ambiguous either
            // way and both resolve to a parameter of the right value.
            map.entry(v).or_insert(i);
        }
        Builder {
            pool,
            vars: map,
            nodes: Vec::new(),
            intern: HashMap::new(),
            memo: HashMap::new(),
            pred_memo: HashMap::new(),
            helpers: Vec::new(),
        }
    }

    fn mk(&mut self, n: Node) -> usize {
        // A binary sum or product rounds the same in either order, so its
        // operands are put in a canonical order: `a·b` and `b·a` (which the
        // pool keeps apart when one side was built by a rewrite) share a node.
        // Three or more operands keep the interpreter's order, which fixes
        // the rounding.
        let n = match n {
            Node::Prod(mut v) if v.len() == 2 => {
                v.sort_unstable();
                Node::Prod(v)
            }
            Node::Sum(mut v) if v.len() == 2 => {
                v.sort_unstable();
                Node::Sum(v)
            }
            other => other,
        };
        if let Some(&i) = self.intern.get(&n) {
            return i;
        }
        let i = self.nodes.len();
        self.nodes.push(n.clone());
        self.intern.insert(n, i);
        i
    }

    fn lit(&mut self, s: String) -> usize {
        self.mk(Node::Lit(s))
    }

    fn one(&mut self) -> usize {
        self.lit("1.0".to_string())
    }

    fn neg(&mut self, a: usize) -> usize {
        if let Node::Neg(b) = self.nodes[a] {
            return b;
        }
        self.mk(Node::Neg(a))
    }

    fn prod(&mut self, mut xs: Vec<usize>) -> usize {
        match xs.len() {
            0 => self.one(),
            1 => xs.pop().unwrap(),
            _ => self.mk(Node::Prod(xs)),
        }
    }

    fn call(&mut self, f: &'static str, args: Vec<usize>) -> usize {
        self.mk(Node::Call(f, args))
    }

    /// `b^n` for `1 ≤ n ≤ 4` by repeated squaring, sharing `b·b`.
    fn powi(&mut self, b: usize, n: u32) -> usize {
        match n {
            1 => b,
            2 => self.mk(Node::Prod(vec![b, b])),
            3 => {
                let sq = self.powi(b, 2);
                self.mk(Node::Prod(vec![sq, b]))
            }
            4 => {
                let sq = self.powi(b, 2);
                self.mk(Node::Prod(vec![sq, sq]))
            }
            _ => unreachable!("powi is only called for 1..=4"),
        }
    }

    /// The value of a finite, non-negative literal node.
    fn literal_value(&self, i: usize) -> Option<f64> {
        match &self.nodes[i] {
            Node::Lit(l) => l.parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0),
            _ => None,
        }
    }

    /// `b^q` for an exact rational exponent.
    fn pow_exact(&mut self, b: usize, q: &Rational) -> usize {
        let n = q.numer();
        if *q.denom() == 1 {
            if n.is_zero() {
                return self.one();
            }
            if is_small_exponent(q) {
                let k = n.to_i32().unwrap();
                let p = self.powi(b, k.unsigned_abs());
                if k > 0 {
                    return p;
                }
                let one = self.one();
                return self.mk(Node::Div(one, p));
            }
            let e = self.lit(c_integer_literal(n));
            if integer_is_exact_f64(n) {
                return self.call("pow", vec![b, e]);
            }
            // `e` is the exponent *rounded to a double* — there is no other
            // kind of literal a C program has — and C's `pow` takes the sign
            // of a negative base from that rounded exponent's parity.
            // `10^30 + 1` rounds to the even `1e30`, so `pow(-1.0, 1e30)`
            // returned `+1` where the answer is `-1`.  Split the sign off and
            // let the double carry only the magnitude.
            let fb = self.call("fabs", vec![b]);
            let mag = self.call("pow", vec![fb, e]);
            if n.is_odd() {
                return self.call("copysign", vec![mag, b]);
            }
            return mag;
        }
        if *q.denom() == 2 && is_small_exponent(q) {
            let k = n.to_i32().unwrap();
            let s = match self.literal_value(b) {
                Some(v) => self.lit(c_double_literal(v.sqrt())),
                None => self.call("sqrt", vec![b]),
            };
            let m = (k.unsigned_abs() - 1) / 2;
            let p = if m == 0 {
                s
            } else {
                let bm = self.powi(b, m);
                self.mk(Node::Prod(vec![bm, s]))
            };
            if k > 0 {
                return p;
            }
            let one = self.one();
            return self.mk(Node::Div(one, p));
        }
        let e = self.lit(c_double_literal(rational_to_f64(q)));
        self.call("pow", vec![b, e])
    }

    /// `q^n` for an exact rational `q ≠ 0` and a small integer `n`, rounded
    /// once to the nearest double.  `x²/2` is often built as `x²·2⁻¹` by a
    /// caller and as `½·x²` by `diff`; folding `2⁻¹` makes the two the same
    /// node.  The interpreter evaluates `powf(2, −1)`, the same correctly
    /// rounded value.
    fn fold_numeric_pow(&self, base: ExprId, exp: ExprId) -> Option<f64> {
        let q = exact_exponent(&self.pool.get(base))?;
        let ExprData::Integer(n) = self.pool.get(exp) else {
            return None;
        };
        let n = n.0.to_i32()?;
        if q == 0 || n.unsigned_abs() > 64 {
            return None;
        }
        let bits = q.numer().significant_bits() + q.denom().significant_bits();
        if u64::from(bits) * u64::from(n.unsigned_abs()) > 8192 {
            return None;
        }
        let p = rug::ops::Pow::pow(q, n);
        Some(rational_to_f64(&p))
    }

    /// If `e` is a negative exact (integer or rational) literal, its magnitude
    /// as a literal node (`None` inside when the magnitude is exactly one).
    /// Floats are left as written, so a `-INFINITY` stays one literal.
    fn negative_literal(&mut self, e: ExprId) -> Option<Option<usize>> {
        match self.pool.get(e) {
            ExprData::Integer(n) if n.0 < 0 => {
                let m = rug::Integer::from(-&n.0);
                if m == 1 {
                    Some(None)
                } else {
                    Some(Some(self.lit(c_integer_literal(&m))))
                }
            }
            ExprData::Rational(r) if r.0 < 0 => {
                let m = Rational::from(-&r.0);
                Some(Some(self.lit(c_double_literal(rational_to_f64(&m)))))
            }
            _ => None,
        }
    }

    fn lower(&mut self, e: ExprId) -> Result<usize, EmitCError> {
        if let Some(&i) = self.vars.get(&e) {
            return Ok(self.mk(Node::Var(i)));
        }
        if let Some(&n) = self.memo.get(&e) {
            return Ok(n);
        }
        let n = self.lower_uncached(e)?;
        self.memo.insert(e, n);
        Ok(n)
    }

    fn lower_uncached(&mut self, e: ExprId) -> Result<usize, EmitCError> {
        Ok(match self.pool.get(e) {
            ExprData::Integer(n) => self.lit(c_integer_literal(&n.0)),
            ExprData::Rational(r) => self.lit(c_double_literal(rational_to_f64(&r.0))),
            ExprData::Float(f) => self.lit(c_double_literal(f.inner.to_f64())),
            ExprData::Symbol { name, .. } => {
                // `π` is an ordinary symbol in this crate; every evaluator
                // resolves it without a binding (see `eval::symbols`), and so
                // does the emitted C.  A caller that lists it as an input has
                // already been answered by the `vars` lookup above.
                if name == crate::eval::symbols::PI_NAME {
                    self.lit(c_double_literal(std::f64::consts::PI))
                } else {
                    return Err(EmitCError::MissingVariable(name));
                }
            }
            ExprData::Add(args) => {
                let mut terms = Vec::with_capacity(args.len());
                for a in args {
                    if let Some(mag) = self.negative_literal(a) {
                        let m = match mag {
                            Some(m) => m,
                            None => self.one(),
                        };
                        terms.push((true, m));
                        continue;
                    }
                    let t = self.lower(a)?;
                    match self.nodes[t] {
                        Node::Neg(u) => terms.push((true, u)),
                        _ => terms.push((false, t)),
                    }
                }
                self.mk(Node::Sum(terms))
            }
            ExprData::Mul(args) => {
                let mut negate = false;
                let mut num = Vec::with_capacity(args.len());
                let mut den = Vec::new();
                for a in args {
                    if let Some(mag) = self.negative_literal(a) {
                        negate = !negate;
                        if let Some(m) = mag {
                            num.push(m);
                        }
                        continue;
                    }
                    if let ExprData::Pow { base, exp } = self.pool.get(a) {
                        if let Some(q) = exact_exponent(&self.pool.get(exp)) {
                            if q < 0
                                && is_small_exponent(&q)
                                && self.fold_numeric_pow(base, exp).is_none()
                            {
                                let b = self.lower(base)?;
                                let pos = Rational::from(-&q);
                                den.push(self.pow_exact(b, &pos));
                                continue;
                            }
                        }
                    }
                    let f = self.lower(a)?;
                    match self.nodes[f] {
                        Node::Neg(u) => {
                            negate = !negate;
                            num.push(u);
                        }
                        _ => num.push(f),
                    }
                }
                let n = if den.is_empty() {
                    self.prod(num)
                } else {
                    let top = self.prod(num);
                    let bottom = self.prod(den);
                    self.mk(Node::Div(top, bottom))
                };
                if negate {
                    self.neg(n)
                } else {
                    n
                }
            }
            ExprData::Pow { base, exp } => {
                if let Some(v) = self.fold_numeric_pow(base, exp) {
                    return Ok(self.lit(c_double_literal(v)));
                }
                let b = self.lower(base)?;
                let ed = self.pool.get(exp);
                match exact_exponent(&ed) {
                    Some(q) => self.pow_exact(b, &q),
                    None => {
                        let x = self.lower(exp)?;
                        self.call("pow", vec![b, x])
                    }
                }
            }
            ExprData::Func { name, args } => self.lower_func(&name, &args)?,
            ExprData::Piecewise { branches, default } => {
                let mut acc = self.lower(default)?;
                for (cond, val) in branches.iter().rev() {
                    let c = self.lower_pred(*cond)?;
                    let v = self.lower(*val)?;
                    acc = self.mk(Node::Select(c, v, acc));
                }
                acc
            }
            ExprData::Predicate { .. } => {
                // A predicate in value position is its indicator, as in
                // `eval_interp`.
                let c = self.lower_pred(e)?;
                let one = self.one();
                let zero = self.lit("0.0".to_string());
                self.mk(Node::Select(c, one, zero))
            }
            other => return Err(EmitCError::UnsupportedNode(format!("{other:?}"))),
        })
    }

    /// Lower a function call.
    ///
    /// # Where the emitted C and the interpreter differ
    ///
    /// * **`gamma`.** Alkahest returns `+∞` at each pole of `Γ`, so `1/Γ(−n)`
    ///   is `0`; C99's `tgamma` returns `NaN` there — a weak refusal in place
    ///   of a correct value, never a wrong number.  Alkahest's `Γ` is also a
    ///   Lanczos approximation good to ~1e-15, where glibc's `tgamma` is
    ///   nearly correctly rounded.
    /// * **`lambert_w`.** The emitted helper is accurate to ~1e-15 relative;
    ///   the interpreter's `W₀` stops at a residual of `1e-15·max(|x|, 1)`, so
    ///   for `|x| ≪ 1` the C is the more accurate of the two.
    /// * **Integer powers** `x^2 … x^4` are repeated multiplication, which
    ///   rounds two or three times where `pow` rounds once (≤ 2 ulp).
    fn lower_func(&mut self, name: &str, args: &[ExprId]) -> Result<usize, EmitCError> {
        let unsupported = || EmitCError::UnsupportedFunction(name.to_string());
        // `e` itself, written as the nearest double.
        if name == "exp" && args.len() == 1 {
            if let ExprData::Integer(n) = self.pool.get(args[0]) {
                if n.0 == 1 {
                    return Ok(self.lit(c_double_literal(std::f64::consts::E)));
                }
            }
        }
        let c_name: Option<&'static str> = match (name, args.len()) {
            ("sin", 1) => Some("sin"),
            ("cos", 1) => Some("cos"),
            ("tan", 1) => Some("tan"),
            ("asin", 1) => Some("asin"),
            ("acos", 1) => Some("acos"),
            ("atan", 1) => Some("atan"),
            ("sinh", 1) => Some("sinh"),
            ("cosh", 1) => Some("cosh"),
            ("tanh", 1) => Some("tanh"),
            ("asinh", 1) => Some("asinh"),
            ("acosh", 1) => Some("acosh"),
            ("atanh", 1) => Some("atanh"),
            ("exp", 1) => Some("exp"),
            ("log", 1) => Some("log"),
            ("sqrt", 1) => Some("sqrt"),
            ("abs", 1) => Some("fabs"),
            ("erf", 1) => Some("erf"),
            ("erfc", 1) => Some("erfc"),
            ("gamma", 1) | ("tgamma", 1) => Some("tgamma"),
            ("floor", 1) => Some("floor"),
            ("ceil", 1) => Some("ceil"),
            // Rust's `f64::round` and C's `round` both round half away from 0.
            ("round", 1) => Some("round"),
            ("log10", 1) => Some("log10"),
            ("log2", 1) => Some("log2"),
            ("exp2", 1) => Some("exp2"),
            ("cbrt", 1) => Some("cbrt"),
            ("atan2", 2) => Some("atan2"),
            ("pow", 2) => Some("pow"),
            // `f64::min`/`max` and `fmin`/`fmax` both ignore a single NaN.
            ("min", 2) => Some("fmin"),
            ("max", 2) => Some("fmax"),
            ("lambert_w", 1) => Some("alkahest_lambert_w0"),
            ("sign", 1) | ("heaviside", 1) => None,
            _ => return Err(unsupported()),
        };
        let mut lowered = Vec::with_capacity(args.len());
        for &a in args {
            lowered.push(self.lower(a)?);
        }
        if let Some(c) = c_name {
            // `sqrt` of a literal (`sqrt(2)`, `sqrt(pi)`) is folded: IEEE
            // `sqrt` is correctly rounded, so Rust's and libm's agree exactly.
            if c == "sqrt" {
                if let Some(v) = self.literal_value(lowered[0]) {
                    return Ok(self.lit(c_double_literal(v.sqrt())));
                }
            }
            if c == "alkahest_lambert_w0" && !self.helpers.contains(&Helper::LambertW0) {
                self.helpers.push(Helper::LambertW0);
            }
            return Ok(self.call(c, lowered));
        }
        // sign / heaviside as ternaries; the value at 0 (and at NaN, which
        // fails both comparisons) matches the registry's `numeric_f64`.
        let a = lowered[0];
        let zero = self.lit("0.0".to_string());
        let one = self.one();
        let gt = self.mk(Node::Cmp(">", a, zero));
        let lt = self.mk(Node::Cmp("<", a, zero));
        Ok(if name == "sign" {
            let m1 = self.lit("-1.0".to_string());
            let inner = self.mk(Node::Select(lt, m1, zero));
            self.mk(Node::Select(gt, one, inner))
        } else {
            let half = self.lit("0.5".to_string());
            let inner = self.mk(Node::Select(lt, zero, half));
            self.mk(Node::Select(gt, one, inner))
        })
    }

    fn lower_pred(&mut self, p: ExprId) -> Result<usize, EmitCError> {
        if let Some(&n) = self.pred_memo.get(&p) {
            return Ok(n);
        }
        let ExprData::Predicate { kind, args } = self.pool.get(p) else {
            return Err(EmitCError::UnsupportedNode(
                "expected a Predicate node in condition position".to_string(),
            ));
        };
        let cmp = |k: &PredicateKind| -> Option<&'static str> {
            Some(match k {
                PredicateKind::Lt => "<",
                PredicateKind::Le => "<=",
                PredicateKind::Gt => ">",
                PredicateKind::Ge => ">=",
                PredicateKind::Eq => "==",
                PredicateKind::Ne => "!=",
                _ => return None,
            })
        };
        let n = match kind {
            PredicateKind::True => self.lit("1".to_string()),
            PredicateKind::False => self.lit("0".to_string()),
            PredicateKind::Not => {
                let a = self.lower_pred(args[0])?;
                self.mk(Node::Not(a))
            }
            PredicateKind::And | PredicateKind::Or => {
                let mut xs = Vec::with_capacity(args.len());
                for &a in &args {
                    xs.push(self.lower_pred(a)?);
                }
                if matches!(kind, PredicateKind::And) {
                    self.mk(Node::All(xs))
                } else {
                    self.mk(Node::Any(xs))
                }
            }
            ref k => {
                let op = cmp(k).expect("remaining predicate kinds are comparisons");
                let l = self.lower(args[0])?;
                let r = self.lower(args[1])?;
                self.mk(Node::Cmp(op, l, r))
            }
        };
        self.pred_memo.insert(p, n);
        Ok(n)
    }
}

/// Straight-line C for a set of outputs: the temporaries, then one C
/// expression per output, then the helper definitions the code calls.
pub(super) struct Lowered {
    pub stmts: Vec<String>,
    pub outputs: Vec<String>,
    pub prelude: String,
}

/// Wrap `s` in `(-…)`, guarding against a leading minus becoming `--`.
fn negated(s: &str) -> String {
    if s.starts_with('-') {
        format!("(-({s}))")
    } else {
        format!("(-{s})")
    }
}

struct Printer<'b> {
    nodes: &'b [Node],
    var_names: &'b [&'b str],
    uses: Vec<u32>,
    names: Vec<Option<String>>,
    stmts: Vec<String>,
    counter: usize,
}

impl Printer<'_> {
    fn render(&mut self, i: usize) -> String {
        if let Some(n) = &self.names[i] {
            return n.clone();
        }
        let nodes = self.nodes;
        let s = match &nodes[i] {
            Node::Var(k) => return self.var_names[*k].to_string(),
            Node::Lit(s) => return s.clone(),
            Node::Sum(ts) => {
                let mut out = String::from("(");
                for (j, &(neg, t)) in ts.iter().enumerate() {
                    let r = self.render(t);
                    match (j, neg) {
                        (0, false) => out.push_str(&r),
                        (0, true) => out.push_str(&negated(&r)),
                        (_, false) => {
                            out.push_str(" + ");
                            out.push_str(&r);
                        }
                        (_, true) => {
                            out.push_str(" - ");
                            out.push_str(&r);
                        }
                    }
                }
                out.push(')');
                out
            }
            Node::Prod(xs) => {
                let parts: Vec<String> = xs.iter().map(|&x| self.render(x)).collect();
                format!("({})", parts.join(" * "))
            }
            Node::Div(a, b) => {
                let (a, b) = (*a, *b);
                let ra = self.render(a);
                let rb = self.render(b);
                format!("({ra} / {rb})")
            }
            Node::Neg(a) => {
                let r = self.render(*a);
                negated(&r)
            }
            Node::Call(f, xs) => {
                let f = *f;
                let parts: Vec<String> = xs.iter().map(|&x| self.render(x)).collect();
                format!("{f}({})", parts.join(", "))
            }
            Node::Select(c, a, b) => {
                let (c, a, b) = (*c, *a, *b);
                let rc = self.render(c);
                let ra = self.render(a);
                let rb = self.render(b);
                format!("({rc} ? {ra} : {rb})")
            }
            Node::Cmp(op, a, b) => {
                let (op, a, b) = (*op, *a, *b);
                let ra = self.render(a);
                let rb = self.render(b);
                format!("({ra} {op} {rb})")
            }
            Node::All(xs) | Node::Any(xs) => {
                let sep = if matches!(nodes[i], Node::All(_)) {
                    " && "
                } else {
                    " || "
                };
                let parts: Vec<String> = xs.iter().map(|&x| self.render(x)).collect();
                format!("({})", parts.join(sep))
            }
            Node::Not(a) => {
                let r = self.render(*a);
                format!("(!{r})")
            }
        };
        if self.uses[i] >= 2 {
            let ty = if self.nodes[i].is_bool() {
                "int"
            } else {
                "double"
            };
            let name = format!("_t{}", self.counter);
            self.counter += 1;
            self.stmts.push(format!("    const {ty} {name} = {s};"));
            self.names[i] = Some(name.clone());
            return name;
        }
        s
    }
}

/// Lower `exprs` jointly: shared sub-expressions of different outputs become
/// one temporary.
pub(super) fn lower_outputs(
    exprs: &[ExprId],
    vars: &[ExprId],
    var_names: &[&str],
    pool: &ExprPool,
) -> Result<Lowered, EmitCError> {
    let mut b = Builder::new(pool, vars);
    let mut roots = Vec::with_capacity(exprs.len());
    for &e in exprs {
        roots.push(b.lower(e)?);
    }

    // Reference counts over the reachable IR.  Each reachable node is printed
    // exactly once (as a temporary or inline in its single consumer), so a
    // child is printed once per incoming edge unless it gets a temporary.
    let mut uses = vec![0u32; b.nodes.len()];
    let mut seen = vec![false; b.nodes.len()];
    let mut stack: Vec<usize> = Vec::new();
    for &r in &roots {
        uses[r] += 1;
        if !seen[r] {
            seen[r] = true;
            stack.push(r);
        }
    }
    while let Some(i) = stack.pop() {
        for c in b.nodes[i].children() {
            uses[c] += 1;
            if !seen[c] {
                seen[c] = true;
                stack.push(c);
            }
        }
    }
    for (i, n) in b.nodes.iter().enumerate() {
        if n.is_leaf() {
            uses[i] = 0;
        }
    }

    let mut pr = Printer {
        nodes: &b.nodes,
        var_names,
        uses,
        names: vec![None; b.nodes.len()],
        stmts: Vec::new(),
        counter: 0,
    };
    // An output referenced nowhere else is printed inline in its own slot.
    let mut outputs = Vec::with_capacity(roots.len());
    for &r in &roots {
        outputs.push(pr.render(r));
    }
    let mut helpers = b.helpers.clone();
    helpers.sort();
    let prelude: String = helpers.iter().map(|h| h.source()).collect();
    Ok(Lowered {
        stmts: pr.stmts,
        outputs,
        prelude,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{emit_expr_c, emit_expr_c_vec};
    use crate::errors::AlkahestError;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn shared_calls_are_emitted_once_across_outputs() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let s = p.func("sin", vec![x]);
        let c = p.func("cos", vec![y]);
        let outs = [p.mul(vec![s, c]), p.add(vec![s, c]), p.pow(s, p.integer(2))];
        let code = emit_expr_c_vec(&outs, &[x, y], &["x", "y"], "f", &p).unwrap();
        assert_eq!(code.matches("sin(").count(), 1, "{code}");
        assert_eq!(code.matches("cos(").count(), 1, "{code}");
        assert!(code.contains("const double _t"), "{code}");
    }

    #[test]
    fn small_powers_share_one_square_and_one_sqrt() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let pw = |n: i32| p.pow(x, p.integer(n));
        let half = |n: i32| p.pow(x, p.rational(n, 2));
        let outs = [
            p.add(vec![pw(2), pw(3), pw(4), pw(-2)]),
            p.add(vec![half(1), half(-1), half(3)]),
        ];
        let code = emit_expr_c_vec(&outs, &[x], &["x"], "f", &p).unwrap();
        assert!(!code.contains("pow("), "{code}");
        assert_eq!(code.matches("(x * x)").count(), 1, "{code}");
        assert_eq!(code.matches("sqrt(").count(), 1, "{code}");
    }

    #[test]
    fn division_and_negation_replace_reciprocal_and_minus_one() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        // x - 2*x/y
        let e = p.add(vec![
            x,
            p.mul(vec![p.integer(-2), x, p.pow(y, p.integer(-1))]),
        ]);
        let code = emit_expr_c(e, &[x, y], &["x", "y"], "f", &p).unwrap();
        assert!(code.contains(" - "), "{code}");
        assert!(code.contains("/ y"), "{code}");
        assert!(!code.contains("-2.0"), "{code}");
    }

    #[test]
    fn pi_and_e_are_round_trip_literals() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let pi = p.symbol("pi", Domain::Real);
        let e = p.add(vec![
            p.func("sin", vec![p.mul(vec![pi, x])]),
            p.func("exp", vec![p.integer(1)]),
            p.pow(p.integer(2), p.rational(1, 2)),
        ]);
        let code = emit_expr_c(e, &[x], &["x"], "f", &p).unwrap();
        assert!(code.contains("3.141592653589793"), "{code}");
        assert!(code.contains("2.718281828459045"), "{code}");
        assert!(code.contains("1.4142135623730951"), "{code}");
        assert!(!code.contains("exp(") && !code.contains("sqrt("), "{code}");
        // A caller that binds `pi` gets a parameter.
        let code = emit_expr_c(e, &[x, pi], &["x", "p"], "f", &p).unwrap();
        assert!(!code.contains("3.14159"), "{code}");
    }

    #[test]
    fn step_functions_lower_to_ternaries_and_lambert_w_to_a_helper() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let e = p.add(vec![
            p.func("heaviside", vec![x]),
            p.func("sign", vec![x]),
            p.func("lambert_w", vec![x]),
        ]);
        let code = emit_expr_c(e, &[x], &["x"], "f", &p).unwrap();
        assert!(code.contains("? 1.0 :"), "{code}");
        assert!(code.contains("0.5"), "{code}");
        assert!(
            code.starts_with("#ifndef ALKAHEST_LAMBERT_W0_DEFINED"),
            "{code}"
        );
        assert_eq!(code.matches("static double alkahest_lambert_w0").count(), 1);
    }

    #[test]
    fn refusals_carry_codegen_codes() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        let dd = p.func("diracdelta", vec![x]);
        let err = emit_expr_c(dd, &[x], &["x"], "f", &p).unwrap_err();
        assert_eq!(err.code(), "E-CODEGEN-001");
        let err = emit_expr_c(p.add(vec![x, y]), &[x], &["x"], "f", &p).unwrap_err();
        assert_eq!(err.code(), "E-CODEGEN-003");
        let err = emit_expr_c(p.big_o(x), &[x], &["x"], "f", &p).unwrap_err();
        assert_eq!(err.code(), "E-CODEGEN-002");
        assert!(err.remediation().is_some());
    }
}
