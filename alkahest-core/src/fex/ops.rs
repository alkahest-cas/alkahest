//! Operator tables 𝕌 (unary) and 𝔹 (binary) for FEX templates.
//!
//! Each unary operator knows its value and first three derivatives in closed
//! form ([`UnaryOp::eval4`]). The fused evaluator needs `f, f', f''` for the
//! forward Laplacian and `f'''` for the parameter Jacobian of the Laplacian
//! term, so all four are produced together.

use crate::kernel::{ExprId, ExprPool};

use super::error::FexError;

/// A unary operator in the FEX table 𝕌.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum UnaryOp {
    /// The constant `0`.
    Zero,
    /// The constant `1`.
    One,
    /// The identity `s`.
    Id,
    /// `s²`.
    Sq,
    /// `s³`.
    Cube,
    /// `s⁴`.
    Quartic,
    /// `exp(s)`.
    Exp,
    /// `sin(s)`.
    Sin,
    /// `cos(s)`.
    Cos,
    /// `log(s)`; a non-positive argument scores the candidate `+∞`.
    Log,
    /// `|s|`; its derivatives are taken as `sign(s)` and `0`.
    Abs,
    /// `tanh(s)` (in the committor FEX paper's table, Song–Cameron–Yang).
    Tanh,
}

/// A binary operator in the FEX table 𝔹.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum BinaryOp {
    /// `a + b`.
    Add,
    /// `a − b`.
    Sub,
    /// `a · b`.
    Mul,
    /// `a / b`; a zero denominator scores the candidate `+∞`.
    Div,
}

/// Every unary operator, in a fixed order (used to index per-operator caches).
pub const ALL_UNARY: [UnaryOp; 12] = [
    UnaryOp::Zero,
    UnaryOp::One,
    UnaryOp::Id,
    UnaryOp::Sq,
    UnaryOp::Cube,
    UnaryOp::Quartic,
    UnaryOp::Exp,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Log,
    UnaryOp::Abs,
    UnaryOp::Tanh,
];

/// The FEX paper's unary table, in the reference implementation's order:
/// `0, 1, id, (·)², (·)³, (·)⁴, exp, sin, cos`.
pub const PAPER_UNARY: [UnaryOp; 9] = [
    UnaryOp::Zero,
    UnaryOp::One,
    UnaryOp::Id,
    UnaryOp::Sq,
    UnaryOp::Cube,
    UnaryOp::Quartic,
    UnaryOp::Exp,
    UnaryOp::Sin,
    UnaryOp::Cos,
];

/// The FEX paper's binary table, in the reference implementation's order:
/// `+, ×, −`.
pub const PAPER_BINARY: [BinaryOp; 3] = [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Sub];

impl UnaryOp {
    /// Parse a table entry: `0`, `1`, `id`, `sq`, `cube`, `quartic`, `exp`,
    /// `sin`, `cos`, `log`, `abs`, `tanh` (plus a few aliases such as `x^2`).
    pub fn parse(name: &str) -> Result<UnaryOp, FexError> {
        let n = name.trim().to_ascii_lowercase();
        Ok(match n.as_str() {
            "0" | "zero" => UnaryOp::Zero,
            "1" | "one" => UnaryOp::One,
            "id" | "x" | "identity" => UnaryOp::Id,
            "sq" | "square" | "x^2" | "x**2" | "pow2" => UnaryOp::Sq,
            "cube" | "x^3" | "x**3" | "pow3" => UnaryOp::Cube,
            "quartic" | "x^4" | "x**4" | "pow4" => UnaryOp::Quartic,
            "exp" => UnaryOp::Exp,
            "sin" => UnaryOp::Sin,
            "cos" => UnaryOp::Cos,
            "log" | "ln" => UnaryOp::Log,
            "abs" => UnaryOp::Abs,
            "tanh" => UnaryOp::Tanh,
            _ => {
                return Err(FexError::UnknownOperator {
                    name: name.to_string(),
                    table: "unary",
                })
            }
        })
    }

    /// The canonical table name (`"sq"`, `"exp"`, …).
    pub fn name(self) -> &'static str {
        match self {
            UnaryOp::Zero => "0",
            UnaryOp::One => "1",
            UnaryOp::Id => "id",
            UnaryOp::Sq => "sq",
            UnaryOp::Cube => "cube",
            UnaryOp::Quartic => "quartic",
            UnaryOp::Exp => "exp",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Log => "log",
            UnaryOp::Abs => "abs",
            UnaryOp::Tanh => "tanh",
        }
    }

    /// Position in [`ALL_UNARY`].
    pub(crate) fn index(self) -> usize {
        ALL_UNARY
            .iter()
            .position(|&u| u == self)
            .expect("every UnaryOp is in ALL_UNARY")
    }

    /// True for the two constant operators `0` and `1`.
    pub fn is_constant(self) -> bool {
        matches!(self, UnaryOp::Zero | UnaryOp::One)
    }

    /// `[f(s), f'(s), f''(s), f'''(s)]`.
    #[inline]
    pub fn eval4(self, s: f64) -> [f64; 4] {
        match self {
            UnaryOp::Zero => [0.0, 0.0, 0.0, 0.0],
            UnaryOp::One => [1.0, 0.0, 0.0, 0.0],
            UnaryOp::Id => [s, 1.0, 0.0, 0.0],
            UnaryOp::Sq => [s * s, 2.0 * s, 2.0, 0.0],
            UnaryOp::Cube => [s * s * s, 3.0 * s * s, 6.0 * s, 6.0],
            UnaryOp::Quartic => {
                let s2 = s * s;
                [s2 * s2, 4.0 * s2 * s, 12.0 * s2, 24.0 * s]
            }
            UnaryOp::Exp => {
                let e = s.exp();
                [e, e, e, e]
            }
            UnaryOp::Sin => {
                let (sn, cs) = s.sin_cos();
                [sn, cs, -sn, -cs]
            }
            UnaryOp::Cos => {
                let (sn, cs) = s.sin_cos();
                [cs, -sn, -cs, sn]
            }
            UnaryOp::Log => {
                if s > 0.0 {
                    let r = 1.0 / s;
                    [s.ln(), r, -r * r, 2.0 * r * r * r]
                } else {
                    [f64::NAN; 4]
                }
            }
            UnaryOp::Abs => [s.abs(), if s >= 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0],
            UnaryOp::Tanh => {
                // t = tanh s: t' = 1 − t², t'' = −2t(1 − t²), t''' = (1 − t²)(6t² − 2).
                let t = s.tanh();
                let q = 1.0 - t * t;
                [t, q, -2.0 * t * q, q * (6.0 * t * t - 2.0)]
            }
        }
    }

    /// `op(arg)` as an expression in `pool`.
    pub fn apply_expr(self, arg: ExprId, pool: &ExprPool) -> ExprId {
        match self {
            UnaryOp::Zero => pool.integer(0),
            UnaryOp::One => pool.integer(1),
            UnaryOp::Id => arg,
            UnaryOp::Sq => pool.pow(arg, pool.integer(2)),
            UnaryOp::Cube => pool.pow(arg, pool.integer(3)),
            UnaryOp::Quartic => pool.pow(arg, pool.integer(4)),
            UnaryOp::Exp => pool.func("exp", vec![arg]),
            UnaryOp::Sin => pool.func("sin", vec![arg]),
            UnaryOp::Cos => pool.func("cos", vec![arg]),
            UnaryOp::Log => pool.func("log", vec![arg]),
            UnaryOp::Abs => pool.func("abs", vec![arg]),
            UnaryOp::Tanh => pool.func("tanh", vec![arg]),
        }
    }
}

impl BinaryOp {
    /// Parse a table entry: `+`, `-`, `*`, `/` (or `add`, `sub`, `mul`, `div`).
    pub fn parse(name: &str) -> Result<BinaryOp, FexError> {
        let n = name.trim().to_ascii_lowercase();
        Ok(match n.as_str() {
            "+" | "add" => BinaryOp::Add,
            "-" | "sub" => BinaryOp::Sub,
            "*" | "×" | "mul" => BinaryOp::Mul,
            "/" | "÷" | "div" => BinaryOp::Div,
            _ => {
                return Err(FexError::UnknownOperator {
                    name: name.to_string(),
                    table: "binary",
                })
            }
        })
    }

    /// The canonical table name (`"+"`, `"-"`, `"*"`, `"/"`).
    pub fn name(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
        }
    }

    /// `a op b` as an expression in `pool`.
    pub fn apply_expr(self, a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
        match self {
            BinaryOp::Add => pool.add(vec![a, b]),
            BinaryOp::Sub => pool.add(vec![a, pool.mul(vec![pool.integer(-1), b])]),
            BinaryOp::Mul => pool.mul(vec![a, b]),
            BinaryOp::Div => pool.mul(vec![a, pool.pow(b, pool.integer(-1))]),
        }
    }
}
