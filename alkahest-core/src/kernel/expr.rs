use crate::kernel::domain::Domain;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Opaque index into an [`crate::kernel::ExprPool`]. `Copy` — expressions are values, not owned objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExprId(pub(crate) u32);

// ---------------------------------------------------------------------------
// Atom wrappers.
//
// `Hash` for the number atoms is value-based and allocation-free: it reads the
// limbs directly.  It used to format the number as a hex string on every call,
// which made hashing a machine-size integer ~7x slower than the lookup around
// it — and the intern table hashes every atom on every lookup and rehashes
// every key when it grows.  Nothing persists these hashes (the pool file and
// the Merkle prototype use their own encodings), so changing them is free.
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
        // rug hashes the signed limb count and the limbs.  GMP keeps an
        // `Integer` normalised (no high zero limbs, zero has size 0), so equal
        // values always have identical (size, limbs) — consistent with `Eq`.
        self.0.hash(state);
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
        // rug keeps a `Rational` canonical (reduced, positive denominator), so
        // equal values have equal numerators and denominators.
        self.0.hash(state);
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
    /// Hashes exactly what [`PartialEq`] compares: the `prec` field and the
    /// *value* of `inner`.  In particular, matching `eq`:
    ///
    /// * every NaN hashes alike (`eq` treats any two NaNs as equal);
    /// * `+0` and `-0` hash alike (`rug`'s `==` says they are equal);
    /// * `inner`'s own MPFR precision is not hashed — `eq` compares values, so
    ///   `1.0` held at 53 and at 64 mantissa bits must collide.  MPFR stores
    ///   the significand left-aligned with the unused low bits zero, so equal
    ///   values differ only by extra all-zero low limbs, which are skipped.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.prec.hash(state);
        let f = &self.inner;
        if f.is_nan() {
            0u8.hash(state);
        } else if f.is_zero() {
            1u8.hash(state);
        } else if f.is_infinite() {
            2u8.hash(state);
            f.is_sign_negative().hash(state);
        } else {
            3u8.hash(state);
            f.is_sign_negative().hash(state);
            f.get_exp().hash(state);
            if let Some(sig) = f.get_significand() {
                let limbs = sig.as_limbs();
                let first = limbs.iter().position(|&l| l != 0).unwrap_or(limbs.len());
                limbs[first..].hash(state);
            }
        }
    }
}

impl fmt::Display for BigFloat {
    /// `rug`'s own rendering, made to read back as a float.
    ///
    /// `rug` prints a zero as `0`, which every parser in this crate reads as
    /// the *integer* `0` — a different node, and a different kind of number.
    /// A finite value whose rendering has neither a point nor an exponent is
    /// therefore given a `.0`.  The digit count is `rug`'s, which grows with
    /// `prec`, so the parsers can recover the precision from it (see
    /// [`crate::parse::float_literal_prec`]).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.inner.to_string();
        if self.inner.is_finite() && !s.contains(['.', 'e', 'E']) {
            write!(f, "{s}.0")
        } else {
            f.write_str(&s)
        }
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
    /// Sum over the roots of a polynomial: `Σ_{c : poly(c)=0} body[var := c]`.
    ///
    /// A binder (like [`ExprData::Exists`]): `var` is the bound root placeholder,
    /// `poly` is a univariate polynomial in `var`, and `body` is the summand
    /// (an expression in `var` and the free variables).  Used to represent the
    /// logarithmic part of a rational-function integral whose residues are
    /// algebraic numbers of degree ≥ 2 (Rothstein–Trager / Lazard–Rioboo–Trager).
    RootSum {
        poly: ExprId,
        var: ExprId,
        body: ExprId,
    },
}

#[cfg(test)]
mod number_hash_tests {
    //! `Hash` must agree with `Eq` for the number atoms: the intern table
    //! relies on it for hash-consing.
    use super::*;
    use proptest::prelude::*;
    use rug::{Assign, Float, Integer, Rational};
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;

    fn h<T: Hash>(rs: &RandomState, x: &T) -> u64 {
        rs.hash_one(x)
    }

    /// `a == b` must imply equal hashes (and `==` must be symmetric).
    fn assert_consistent<T: Hash + Eq + fmt::Debug>(a: &T, b: &T) {
        let rs = RandomState::new();
        if a == b {
            assert!(b == a, "Eq not symmetric: {a:?} vs {b:?}");
            assert_eq!(
                h(&rs, a),
                h(&rs, b),
                "equal values hash apart: {a:?} vs {b:?}"
            );
        }
    }

    fn float(inner_prec: u32, v: f64, prec: u32) -> BigFloat {
        BigFloat {
            inner: Float::with_val(inner_prec, v),
            prec,
        }
    }

    #[test]
    fn integer_hash_ignores_allocation() {
        // Same value, very different internal capacity / construction path.
        let a = BigInt(Integer::from(-42));
        let mut grown = Integer::with_capacity(10_000);
        grown.assign(Integer::from(1) << 5000u32);
        grown -= Integer::from(1) << 5000u32;
        grown -= 42;
        let b = BigInt(grown);
        assert_eq!(a, b);
        assert_consistent(&a, &b);
        let z1 = BigInt(Integer::new());
        let z2 = BigInt(Integer::from(7) - 7);
        assert_eq!(z1, z2);
        assert_consistent(&z1, &z2);
        // Sign is part of the value.
        let rs = RandomState::new();
        assert_ne!(
            h(&rs, &BigInt(Integer::from(5))),
            h(&rs, &BigInt(Integer::from(-5)))
        );
    }

    #[test]
    fn rational_hash_uses_canonical_form() {
        let a = BigRat(Rational::from((6, -4)));
        let b = BigRat(Rational::from((-3, 2)));
        assert_eq!(a, b);
        assert_consistent(&a, &b);
        let c = BigRat(Rational::from((10, 5)));
        let d = BigRat(Rational::from(2));
        assert_eq!(c, d);
        assert_consistent(&c, &d);
    }

    #[test]
    fn float_hash_matches_eq_on_special_values() {
        // ±0 are equal under `Eq`, so they must hash alike.
        let pz = float(53, 0.0, 53);
        let nz = float(53, -0.0, 53);
        assert!(nz.inner.is_sign_negative());
        assert_eq!(pz, nz);
        assert_consistent(&pz, &nz);
        // Every NaN equals every NaN (same `prec`), whatever its sign.
        let n1 = float(53, f64::NAN, 53);
        let n2 = BigFloat {
            inner: -Float::with_val(113, f64::NAN),
            prec: 53,
        };
        assert_eq!(n1, n2);
        assert_consistent(&n1, &n2);
        // ±∞ differ.
        assert_ne!(
            float(53, f64::INFINITY, 53),
            float(53, f64::NEG_INFINITY, 53)
        );
        // `prec` is part of identity.
        assert_ne!(float(53, 1.0, 53), float(53, 1.0, 64));
    }

    #[test]
    fn float_hash_ignores_inner_mpfr_precision() {
        // `Eq` compares values, not the MPFR precision `inner` happens to
        // carry, so the extra all-zero low limbs must not reach the hash.
        for v in [
            1.0,
            -1.5,
            0.1,
            1e300,
            -3.0e-300,
            f64::MIN_POSITIVE,
            f64::MAX,
        ] {
            for p in [53u32, 64, 65, 128, 129, 1000] {
                let a = float(53, v, 53);
                let b = float(p, v, 53);
                assert_eq!(a, b, "{v} at inner prec {p}");
                assert_consistent(&a, &b);
            }
        }
    }

    #[test]
    fn hashes_spread_over_distinct_values() {
        // Not a correctness requirement, but a degenerate hash would silently
        // turn the intern table into a linked list.
        let rs = RandomState::new();
        let mut seen = std::collections::HashSet::new();
        for k in -500i64..500 {
            seen.insert(h(&rs, &BigInt(Integer::from(k))));
            seen.insert(h(&rs, &BigRat(Rational::from((k, 7)))));
            seen.insert(h(&rs, &float(53, k as f64 * 0.37, 53)));
        }
        // 3000 values, minus the collisions under `Eq`: 0/7 == 0 is a
        // BigRat vs BigInt (different types, fine), -0.0 == 0.0 is one value.
        assert!(seen.len() >= 2990, "only {} distinct hashes", seen.len());
    }

    fn big_integer() -> impl Strategy<Value = Integer> {
        prop_oneof![
            any::<i64>().prop_map(Integer::from),
            any::<i128>().prop_map(Integer::from),
            (any::<bool>(), proptest::collection::vec(any::<u64>(), 0..6)).prop_map(
                |(neg, limbs)| {
                    let v = Integer::from_digits(&limbs, rug::integer::Order::Lsf);
                    if neg {
                        -v
                    } else {
                        v
                    }
                }
            ),
        ]
    }

    proptest! {
        #[test]
        fn prop_integer_eq_implies_hash(a in big_integer(), b in big_integer(), k in big_integer()) {
            // A second route to the same value, (a + k) - k, in an
            // over-allocated buffer.
            let mut round = Integer::with_capacity(4096);
            round.assign(&a + &k);
            round -= &k;
            prop_assert_eq!(&a, &round);
            assert_consistent(&BigInt(a.clone()), &BigInt(round));
            assert_consistent(&BigInt(a), &BigInt(b));
        }

        #[test]
        fn prop_rational_eq_implies_hash(
            n in big_integer(),
            d in big_integer(),
            k in big_integer(),
        ) {
            prop_assume!(d != 0 && k != 0);
            let r = BigRat(Rational::from((n.clone(), d.clone())));
            // Unreduced input with the same value: (n·k)/(d·k).
            let scaled = BigRat(Rational::from((n * &k, d * &k)));
            prop_assert_eq!(&r, &scaled);
            assert_consistent(&r, &scaled);
        }

        #[test]
        fn prop_float_eq_implies_hash(
            a in any::<f64>(),
            b in any::<f64>(),
            pa in 2u32..300,
            pb in 2u32..300,
            prec in prop_oneof![Just(53u32), Just(64u32), 2u32..200],
        ) {
            // Arbitrary pairs, including NaNs, ±0, ±∞ and subnormals.
            let x = BigFloat { inner: Float::with_val(pa, a), prec };
            let y = BigFloat { inner: Float::with_val(pb, b), prec };
            assert_consistent(&x, &y);
            // The same (rounded) value re-held at a wider MPFR precision.
            let wider = BigFloat {
                inner: Float::with_val(pa.max(pb) + 64, &x.inner),
                prec,
            };
            prop_assert_eq!(&x, &wider);
            assert_consistent(&x, &wider);
        }

        #[test]
        fn prop_distinct_floats_hash_apart(a in any::<f64>(), b in any::<f64>()) {
            let x = float(53, a, 53);
            let y = float(53, b, 53);
            let rs = RandomState::new();
            if x != y {
                prop_assert_ne!(h(&rs, &x), h(&rs, &y));
            }
        }
    }
}
