use crate::kernel::domain::Domain;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Opaque index into an [`crate::kernel::ExprPool`]. `Copy` — expressions are values, not owned objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExprId(pub(crate) u32);

// ---------------------------------------------------------------------------
// Atom wrappers — rug types lack Hash, so we provide it via string encoding.
// String encoding is O(n·log n) but correct and simple for v0.1.
// ---------------------------------------------------------------------------

/// Arbitrary-precision integer atom.
#[derive(Debug, Clone)]
pub struct BigInt(pub rug::Integer);

impl PartialEq for BigInt {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for BigInt {}

impl Hash for BigInt {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_string_radix(16).hash(state);
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Arbitrary-precision rational atom. Stored in canonical reduced form by rug.
#[derive(Debug, Clone)]
pub struct BigRat(pub rug::Rational);

impl PartialEq for BigRat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for BigRat {}

impl Hash for BigRat {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.numer().to_string_radix(16).hash(state);
        self.0.denom().to_string_radix(16).hash(state);
    }
}

impl fmt::Display for BigRat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self.0.denom() == 1 {
            write!(f, "{}", self.0.numer())
        } else {
            write!(f, "{}/{}", self.0.numer(), self.0.denom())
        }
    }
}

/// Arbitrary-precision floating-point atom. `prec` (precision in bits) is
/// part of structural identity: `Float(1.0, 53) != Float(1.0, 64)`.
#[derive(Debug, Clone)]
pub struct BigFloat {
    pub inner: rug::Float,
    pub prec: u32,
}

impl PartialEq for BigFloat {
    fn eq(&self, other: &Self) -> bool {
        if self.prec != other.prec {
            return false;
        }
        match (self.inner.is_nan(), other.inner.is_nan()) {
            (true, true) => true,
            (false, false) => self.inner == other.inner,
            _ => false,
        }
    }
}
impl Eq for BigFloat {}

impl Hash for BigFloat {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.prec.hash(state);
        if self.inner.is_nan() {
            "nan".hash(state);
        } else {
            // Exact hex representation; None → enough digits for exact round-trip.
            self.inner.to_string_radix(16, None).hash(state);
        }
    }
}

impl fmt::Display for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

// ---------------------------------------------------------------------------
// Predicate — symbolic boolean conditions for Piecewise expressions
// ---------------------------------------------------------------------------

/// Kind of a symbolic predicate.
///
/// Predicates are stored in the intern table as
/// `ExprData::Predicate { kind, args }`.  The `args` field holds the
/// operands as `ExprId` nodes.
///
/// | Kind | Arity | Meaning |
/// |------|-------|---------|
/// | `Lt` | 2     | `args[0]` < `args[1]` |
/// | `Le` | 2     | `args[0]` ≤ `args[1]` |
/// | `Gt` | 2     | `args[0]` > `args[1]` |
/// | `Ge` | 2     | `args[0]` ≥ `args[1]` |
/// | `Eq` | 2     | `args[0]` = `args[1]` (symbolic equality) |
/// | `Ne` | 2     | `args[0]` ≠ `args[1]` |
/// | `And`| n     | conjunction of n predicate ExprIds |
/// | `Or` | n     | disjunction of n predicate ExprIds |
/// | `Not`| 1     | negation |
/// | `True` | 0   | always-true |
/// | `False`| 0   | always-false |
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PredicateKind {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    Not,
    True,
    False,
}

impl fmt::Display for PredicateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PredicateKind::Lt => "<",
            PredicateKind::Le => "≤",
            PredicateKind::Gt => ">",
            PredicateKind::Ge => "≥",
            PredicateKind::Eq => "=",
            PredicateKind::Ne => "≠",
            PredicateKind::And => "∧",
            PredicateKind::Or => "∨",
            PredicateKind::Not => "¬",
            PredicateKind::True => "True",
            PredicateKind::False => "False",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Expression data — the structural content stored in the intern table.
// ---------------------------------------------------------------------------

/// Structural content of an expression node.
///
/// All compound nodes hold [`ExprId`] children, not owned sub-trees.
/// This keeps `ExprData` small and allows sharing via the intern table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExprData {
    // Atoms
    Symbol {
        name: String,
        domain: Domain,
        /// When `false`, this generator does not commute under multiplication; see V3-2.
        commutative: bool,
    },
    Integer(BigInt),
    Rational(BigRat),
    Float(BigFloat),
    // Compound (n-ary for Add/Mul; binary for Pow; variadic for Func)
    Add(Vec<ExprId>),
    Mul(Vec<ExprId>),
    Pow {
        base: ExprId,
        exp: ExprId,
    },
    Func {
        name: String,
        args: Vec<ExprId>,
    },
    // PA-9 — symbolic conditionals
    /// A piecewise expression: evaluates to `value_i` when `cond_i` holds,
    /// and to `default` when no condition matches.
    ///
    /// Conditions are `ExprData::Predicate` nodes stored in the pool.
    /// Branches are tried in order; the first matching condition wins.
    Piecewise {
        branches: Vec<(ExprId /* cond */, ExprId /* value */)>,
        default: ExprId,
    },
    /// A symbolic predicate (boolean condition over symbolic reals).
    Predicate {
        kind: PredicateKind,
        args: Vec<ExprId>,
    },
    /// Universal quantification (`∀ var . body`).  Used by first-order logic (V3-3).
    Forall {
        var: ExprId,
        body: ExprId,
    },
    /// Existential quantification (`∃ var . body`).
    Exists {
        var: ExprId,
        body: ExprId,
    },
    /// Landau big-O remainder: `O(arg)` as a symbolic order bound (V2-15 series API).
    BigO(ExprId),
}
