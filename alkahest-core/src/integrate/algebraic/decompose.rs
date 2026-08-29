//! Decompose a symbolic expression into `A(x) + B(x) * sqrt(P(x))`.
//!
//! Given the ExprId of the sqrt generator (either `sqrt(P)` or `P^(1/2)`)
//! and the expression to decompose, this module returns `(A, B)` such that
//! `expr == A + B * sqrt_id` where A and B are free of the sqrt generator.
//!
//! Arithmetic in the algebraic field K = Q(x)\[y\]/(y² - P):
//!   - Addition:       (a,b) + (c,d) = (a+c, b+d)
//!   - Multiplication: (a,b)·(c,d)   = (a·c + b·d·P, a·d + b·c)
//!   - Inversion:      (a,b)⁻¹       = (a/(a²−b²P), −b/(a²−b²P))
//!   - Integer power:  (a,b)^n via squaring

use super::poly_utils::{as_integer, is_free_of_subexpr};
use crate::kernel::{ExprData, ExprId, ExprPool};

// ---------------------------------------------------------------------------
// Field element: a + b*sqrt(P)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct FieldElem {
    pub a: ExprId, // rational part (free of sqrt)
    pub b: ExprId, // coefficient of sqrt (free of sqrt)
}

impl FieldElem {
    pub fn pure_rational(a: ExprId, pool: &ExprPool) -> Self {
        FieldElem {
            a,
            b: pool.integer(0_i32),
        }
    }
    pub fn pure_sqrt(b: ExprId, pool: &ExprPool) -> Self {
        FieldElem {
            a: pool.integer(0_i32),
            b,
        }
    }
    pub fn one(pool: &ExprPool) -> Self {
        FieldElem {
            a: pool.integer(1_i32),
            b: pool.integer(0_i32),
        }
    }
    pub fn zero(pool: &ExprPool) -> Self {
        FieldElem {
            a: pool.integer(0_i32),
            b: pool.integer(0_i32),
        }
    }

    pub fn add(self, other: FieldElem, pool: &ExprPool) -> FieldElem {
        let a = pool.add(vec![self.a, other.a]);
        let b = pool.add(vec![self.b, other.b]);
        FieldElem { a, b }
    }

    #[allow(dead_code)]
    pub fn neg(self, pool: &ExprPool) -> FieldElem {
        let neg1 = pool.integer(-1_i32);
        let a = pool.mul(vec![neg1, self.a]);
        let b = pool.mul(vec![neg1, self.b]);
        FieldElem { a, b }
    }

    /// Multiply two field elements: (a+b·y)·(c+d·y) = (a·c + b·d·P) + (a·d + b·c)·y
    pub fn mul(self, other: FieldElem, p: ExprId, pool: &ExprPool) -> FieldElem {
        // new_a = self.a * other.a + self.b * other.b * P
        let ac = pool.mul(vec![self.a, other.a]);
        let bd_p = pool.mul(vec![self.b, other.b, p]);
        let new_a = pool.add(vec![ac, bd_p]);
        // new_b = self.a * other.b + self.b * other.a
        let ad = pool.mul(vec![self.a, other.b]);
        let bc = pool.mul(vec![self.b, other.a]);
        let new_b = pool.add(vec![ad, bc]);
        FieldElem { a: new_a, b: new_b }
    }

    /// Invert: 1/(a+b·y) = (a−b·y) / (a²−b²·P) = conj/norm
    pub fn inv(self, p: ExprId, pool: &ExprPool) -> FieldElem {
        use super::poly_utils::is_zero_expr;
        // Special case: inv(0, b) = (0, (b·P)^{-1})
        // This avoids the messy (-b^2·P)^{-1} form that confuses later pattern matching.
        if is_zero_expr(self.a, pool) {
            let bp = pool.mul(vec![self.b, p]);
            let new_b = pool.pow(bp, pool.integer(-1_i32));
            return FieldElem {
                a: pool.integer(0_i32),
                b: new_b,
            };
        }
        // General case: norm = a^2 - b^2 * P
        let a2 = pool.pow(self.a, pool.integer(2_i32));
        let b2_p = pool.mul(vec![pool.pow(self.b, pool.integer(2_i32)), p]);
        let neg1 = pool.integer(-1_i32);
        let norm = pool.add(vec![a2, pool.mul(vec![neg1, b2_p])]);
        let norm_inv = pool.pow(norm, pool.integer(-1_i32));
        let new_a = pool.mul(vec![self.a, norm_inv]);
        let new_b = pool.mul(vec![neg1, self.b, norm_inv]);
        FieldElem { a: new_a, b: new_b }
    }

    /// Integer power (positive, negative, or zero)
    pub fn powi(self, n: i64, p: ExprId, pool: &ExprPool) -> FieldElem {
        if n == 0 {
            return FieldElem::one(pool);
        }
        if n < 0 {
            return self.inv(p, pool).powi(-n, p, pool);
        }
        if n == 1 {
            return self;
        }
        // Fast exponentiation by squaring
        let half = self.powi(n / 2, p, pool);
        let sq = half.mul(half, p, pool);
        if n % 2 == 0 {
            sq
        } else {
            sq.mul(self, p, pool)
        }
    }
}

// ---------------------------------------------------------------------------
// Main decomposition
// ---------------------------------------------------------------------------

/// Decompose `expr` into `(A, B)` such that `expr = A + B * sqrt_id`
/// where A and B are free of `sqrt_id`.
///
/// Returns `None` if the decomposition cannot be performed (e.g., sqrt appears
/// with a different argument than `p_expr`, or the expression is structurally
/// incompatible).
pub fn decompose_sqrt(
    expr: ExprId,
    sqrt_id: ExprId,
    p_expr: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let elem = decompose_elem(expr, sqrt_id, p_expr, pool)?;
    Some((elem.a, elem.b))
}

/// Recursive decomposition returning a `FieldElem`.
fn decompose_elem(
    expr: ExprId,
    sqrt_id: ExprId,
    p_expr: ExprId,
    pool: &ExprPool,
) -> Option<FieldElem> {
    // Base case: expr is the sqrt generator itself
    if expr == sqrt_id {
        return Some(FieldElem::pure_sqrt(pool.integer(1_i32), pool));
    }

    // Base case: expr is free of the sqrt generator
    if is_free_of_subexpr(expr, sqrt_id, pool) {
        return Some(FieldElem::pure_rational(expr, pool));
    }

    match pool.get(expr) {
        ExprData::Add(args) => {
            let mut acc = FieldElem::zero(pool);
            for a in &args {
                let elem = decompose_elem(*a, sqrt_id, p_expr, pool)?;
                acc = acc.add(elem, pool);
            }
            Some(acc)
        }

        ExprData::Mul(args) => {
            let mut acc = FieldElem::one(pool);
            for a in &args {
                let elem = decompose_elem(*a, sqrt_id, p_expr, pool)?;
                acc = acc.mul(elem, p_expr, pool);
            }
            Some(acc)
        }

        ExprData::Pow { base, exp } => {
            // Special case: sqrt_id^n or (p_expr)^(1/2) patterns
            if base == sqrt_id {
                // sqrt(P)^n
                let n = as_integer(exp, pool)?;
                // sqrt(P)^n = P^(n/2) for n even, P^((n-1)/2) * sqrt(P) for n odd
                if n == 0 {
                    return Some(FieldElem::one(pool));
                }
                if n > 0 {
                    let n_u = n as u32;
                    if n_u % 2 == 0 {
                        // P^(n/2) — fully rational
                        let p_pow = pool.pow(p_expr, pool.integer(n_u / 2));
                        return Some(FieldElem::pure_rational(p_pow, pool));
                    } else {
                        // P^((n-1)/2) * sqrt(P)
                        let p_pow = pool.pow(p_expr, pool.integer((n_u - 1) / 2));
                        return Some(FieldElem::pure_sqrt(p_pow, pool));
                    }
                } else {
                    // Negative power of sqrt(P): sqrt(P)^(-n) = 1/sqrt(P)^n
                    let base_elem = FieldElem::pure_sqrt(pool.integer(1_i32), pool);
                    return Some(base_elem.powi(n, p_expr, pool));
                }
            }

            // Fractional power: base^(p/q) where this is the sqrt generator
            // We handle Pow(p_expr, Rational(1/2)) → same as sqrt_id, already handled above
            // For Pow(base, Integer(n)) where base contains sqrt:
            if let Some(n) = as_integer(exp, pool) {
                let base_elem = decompose_elem(base, sqrt_id, p_expr, pool)?;
                return Some(base_elem.powi(n, p_expr, pool));
            }

            // Pow with non-integer exponent that isn't our sqrt generator: give up
            None
        }

        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            // This is a different sqrt — only allowed if it matches our generator
            if expr == sqrt_id {
                Some(FieldElem::pure_sqrt(pool.integer(1_i32), pool))
            } else {
                // Different algebraic generator — we don't support multiple generators
                None
            }
        }

        _ => {
            // Any other expression: if free of sqrt_id it's rational, else unsupported
            if is_free_of_subexpr(expr, sqrt_id, pool) {
                Some(FieldElem::pure_rational(expr, pool))
            } else {
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Normal form `A(x) + B(x)·√P` — rationalizing radical denominators
// ---------------------------------------------------------------------------
//
// # Why an Euler answer needs this
//
// The Euler substitution `t = x + √P` is a bijection onto a *rational*
// parameter, so what the `t`-integral returns is a rational function **of `t`** —
// and back-substituting leaves powers of `x + √P`, of both signs, rather than
// the radical.  `∫x/√(x²−1) dx` comes back as
//
// ```text
//   ½(x+√(x²−1)) − ½(x+√(x²−1))⁻¹      instead of      √(x²−1)
// ```
//
// As a final answer that is merely ugly.  As the `v` of an integration-by-parts
// step it is fatal — the next round must differentiate and re-integrate whatever
// shape it is handed, and rationalising the radical denominator is exactly what
// makes the difference (it is what unlocked Charlwood #35).
//
// # Why `FieldElem` + `simplify` is not enough
//
// `FieldElem::inv` *is* conjugate rationalization, but it builds the norm
// `a²−b²P` as an expression tree and leaves the cancelling to `simplify` — which
// does not expand it.  Normalizing `(x+√(x²−1))⁻¹` that way yields
// `x/(x²−(x²−1)) − √(x²−1)/(x²−(x²−1))`: rationalized in form, with the
// denominator `x²−(x²−1) = 1` still sitting there unevaluated.  Measured, that
// made every answer *longer*.
//
// So the arithmetic below runs on `QPoly` coefficient vectors, where
// `x²−(x²−1)` is the constant `1` by construction, with a `poly_gcd` reduction
// after every operation to keep the fraction in lowest terms.  Expressions are
// rebuilt only at the end.

use super::poly_utils::is_free_of;
use crate::integrate::risch::poly_rde::{
    degree, expr_to_qpoly, poly_add, poly_mul, poly_scale, qpoly_to_expr, trim,
};
use crate::integrate::risch::rational_rde::{expr_to_qrational, poly_div_exact, poly_gcd};

type QPoly = Vec<rug::Rational>;

/// `(a + b·y)/d` with `y² = P`, all of `a`, `b`, `d` in `ℚ[x]`.
#[derive(Clone)]
struct AlgFrac {
    a: QPoly,
    b: QPoly,
    d: QPoly,
}

/// Guard against a pathological exponent turning normalization into a blow-up.
const MAX_ALG_POW: i64 = 32;
/// Degree ceiling for an intermediate; beyond it we give up and keep the input.
const MAX_ALG_DEG: i64 = 64;

fn one_poly() -> QPoly {
    vec![rug::Rational::from(1)]
}

impl AlgFrac {
    fn rational(num: QPoly, den: QPoly) -> Self {
        AlgFrac {
            a: num,
            b: Vec::new(),
            d: den,
        }
        .reduced()
    }

    fn one() -> Self {
        AlgFrac {
            a: one_poly(),
            b: Vec::new(),
            d: one_poly(),
        }
    }

    fn zero() -> Self {
        AlgFrac {
            a: Vec::new(),
            b: Vec::new(),
            d: one_poly(),
        }
    }

    fn generator() -> Self {
        AlgFrac {
            a: Vec::new(),
            b: one_poly(),
            d: one_poly(),
        }
    }

    fn too_big(&self) -> bool {
        degree(&self.a).max(degree(&self.b)).max(degree(&self.d)) > MAX_ALG_DEG
    }

    /// Cancel `gcd(a, b, d)` and make `d` monic, so equal values get equal
    /// representations and `x²−(x²−1)` really is `1`.
    fn reduced(self) -> Self {
        let (a, b, d) = (trim(self.a), trim(self.b), trim(self.d));
        if d.is_empty() {
            return AlgFrac { a, b, d };
        }
        let g = poly_gcd(&poly_gcd(&a, &b), &d);
        let (a, b, d) = if degree(&g) >= 1 {
            (
                poly_div_exact(&a, &g),
                poly_div_exact(&b, &g),
                poly_div_exact(&d, &g),
            )
        } else {
            (a, b, d)
        };
        match d.last() {
            Some(lead) if *lead != 1 => {
                let inv = rug::Rational::from(1) / lead.clone();
                AlgFrac {
                    a: poly_scale(&a, &inv),
                    b: poly_scale(&b, &inv),
                    d: poly_scale(&d, &inv),
                }
            }
            _ => AlgFrac { a, b, d },
        }
    }

    fn add(self, o: AlgFrac) -> Self {
        AlgFrac {
            a: poly_add(&poly_mul(&self.a, &o.d), &poly_mul(&o.a, &self.d)),
            b: poly_add(&poly_mul(&self.b, &o.d), &poly_mul(&o.b, &self.d)),
            d: poly_mul(&self.d, &o.d),
        }
        .reduced()
    }

    /// `(a₁+b₁y)(a₂+b₂y) = (a₁a₂ + b₁b₂P) + (a₁b₂ + b₁a₂)y`.
    fn mul(self, o: AlgFrac, p: &QPoly) -> Self {
        AlgFrac {
            a: poly_add(
                &poly_mul(&self.a, &o.a),
                &poly_mul(&poly_mul(&self.b, &o.b), p),
            ),
            b: poly_add(&poly_mul(&self.a, &o.b), &poly_mul(&self.b, &o.a)),
            d: poly_mul(&self.d, &o.d),
        }
        .reduced()
    }

    /// `d/(a+by) = d·(a−by)/(a²−b²P)` — conjugate rationalization on
    /// coefficients, so the norm arrives already cancelled.
    fn inv(self, p: &QPoly) -> Option<Self> {
        let minus_one = rug::Rational::from(-1);
        let norm = poly_add(
            &poly_mul(&self.a, &self.a),
            &poly_scale(&poly_mul(&poly_mul(&self.b, &self.b), p), &minus_one),
        );
        if trim(norm.clone()).is_empty() {
            return None; // zero divisor in the field
        }
        Some(
            AlgFrac {
                a: poly_mul(&self.d, &self.a),
                b: poly_scale(&poly_mul(&self.d, &self.b), &minus_one),
                d: norm,
            }
            .reduced(),
        )
    }

    fn powi(self, n: i64, p: &QPoly) -> Option<Self> {
        if n.abs() > MAX_ALG_POW {
            return None;
        }
        if n < 0 {
            return self.inv(p)?.powi(-n, p);
        }
        let mut acc = AlgFrac::one();
        for _ in 0..n {
            acc = acc.mul(self.clone(), p);
            if acc.too_big() {
                return None;
            }
        }
        Some(acc)
    }

    fn to_expr(&self, var: ExprId, sqrt_id: ExprId, pool: &ExprPool) -> ExprId {
        let den = qpoly_to_expr(&self.d, var, pool);
        let inv_d = pool.pow(den, pool.integer(-1_i32));
        let a = pool.mul(vec![qpoly_to_expr(&self.a, var, pool), inv_d]);
        if trim(self.b.clone()).is_empty() {
            return a;
        }
        let b = pool.mul(vec![qpoly_to_expr(&self.b, var, pool), inv_d, sqrt_id]);
        pool.add(vec![a, b])
    }
}

/// Rewrite every algebraic subexpression of `expr` into the field normal form
/// `A(x) + B(x)·√P`, rationalizing radical denominators.
///
/// Applied to *maximal* subexpressions that are rational in `(x, √P)`.  A
/// transcendental head (`log`, `atan`, … — the logarithmic part of an answer) is
/// recursed *through*, so its argument is normalized and the head is untouched.
/// Anything the arithmetic cannot read, or that would blow past the degree and
/// exponent ceilings, is returned unchanged: this is total, never fails, and is
/// only ever a change of shape.  The caller still gates the result.
pub fn normalize_over_sqrt(expr: ExprId, radicand: ExprId, var: ExprId, pool: &ExprPool) -> ExprId {
    let Some(p) = expr_to_qpoly(radicand, var, pool).map(trim) else {
        return expr;
    };
    if degree(&p) < 1 {
        return expr;
    }
    let sqrt_id = pool.func("sqrt", vec![qpoly_to_expr(&p, var, pool)]);
    let canon = canonicalize_radicals(expr, &p, var, sqrt_id, pool);
    normalize_rec(canon, sqrt_id, &p, var, pool)
}

/// Point every spelling of the generator at one `ExprId`.
///
/// Two things make this necessary, and skipping it does not fail loudly — it
/// makes the normalization silently do nothing:
///
/// * `sqrt(u)` and `u^(1/2)` are used interchangeably (`simplify` produces
///   both), while the decomposition keys on a single node;
/// * the radicand handed in comes from the *integrand* and the one inside the
///   candidate has been through `simplify`, and those are different nodes for
///   the same polynomial — `parse("x^2-1")` builds `x² + (−1)·1`, which is not
///   the `ExprId` of `x² + (−1)`.  That mismatch is why `∫x/√(x²−1)` collapsed
///   to `√(x²−1)` from a builder-constructed tree and stayed in Euler form from
///   parser output: the same form-sensitivity class the Charlwood analysis
///   flags for `Mul` associativity, here on the generator identity.
///
/// So the test is *semantic*: any `sqrt(u)` or `u^(1/2)` whose `u` is the same
/// polynomial as `P` becomes `sqrt_id`.
fn canonicalize_radicals(
    expr: ExprId,
    p: &QPoly,
    var: ExprId,
    sqrt_id: ExprId,
    pool: &ExprPool,
) -> ExprId {
    if expr == sqrt_id {
        return expr;
    }
    let same_radicand = |u: ExprId| expr_to_qpoly(u, var, pool).is_some_and(|q| trim(q) == *p);
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "sqrt" && args.len() == 1 => {
            if same_radicand(args[0]) {
                sqrt_id
            } else {
                expr
            }
        }
        ExprData::Pow { base, exp } => {
            if let ExprData::Rational(r) = pool.get(exp) {
                if *r.0.numer() == 1 && *r.0.denom() == 2 && same_radicand(base) {
                    return sqrt_id;
                }
            }
            pool.pow(canonicalize_radicals(base, p, var, sqrt_id, pool), exp)
        }
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| canonicalize_radicals(a, p, var, sqrt_id, pool))
                .collect::<Vec<_>>(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| canonicalize_radicals(a, p, var, sqrt_id, pool))
                .collect::<Vec<_>>(),
        ),
        ExprData::Func { ref name, ref args } => pool.func(
            name,
            args.iter()
                .map(|&a| canonicalize_radicals(a, p, var, sqrt_id, pool))
                .collect::<Vec<_>>(),
        ),
        _ => expr,
    }
}

fn normalize_rec(expr: ExprId, sqrt_id: ExprId, p: &QPoly, var: ExprId, pool: &ExprPool) -> ExprId {
    if is_free_of_subexpr(expr, sqrt_id, pool) {
        return expr;
    }
    if let Some(f) = to_alg(expr, sqrt_id, p, var, pool) {
        return f.to_expr(var, sqrt_id, pool);
    }
    match pool.get(expr) {
        ExprData::Add(args) => pool.add(
            args.iter()
                .map(|&a| normalize_rec(a, sqrt_id, p, var, pool))
                .collect::<Vec<_>>(),
        ),
        ExprData::Mul(args) => pool.mul(
            args.iter()
                .map(|&a| normalize_rec(a, sqrt_id, p, var, pool))
                .collect::<Vec<_>>(),
        ),
        ExprData::Pow { base, exp } => pool.pow(normalize_rec(base, sqrt_id, p, var, pool), exp),
        ExprData::Func { ref name, ref args } => pool.func(
            name,
            args.iter()
                .map(|&a| normalize_rec(a, sqrt_id, p, var, pool))
                .collect::<Vec<_>>(),
        ),
        _ => expr,
    }
}

/// Read `expr` as an element of `ℚ(x)[y]/(y²−P)`, or `None` when it is not one
/// (a transcendental head, an irrational constant such as `√2`, a non-integer
/// exponent, an intermediate past the ceilings).
fn to_alg(
    expr: ExprId,
    sqrt_id: ExprId,
    p: &QPoly,
    var: ExprId,
    pool: &ExprPool,
) -> Option<AlgFrac> {
    if expr == sqrt_id {
        return Some(AlgFrac::generator());
    }
    if is_free_of_subexpr(expr, sqrt_id, pool) {
        let numeric = matches!(pool.get(expr), ExprData::Integer(_) | ExprData::Rational(_));
        if numeric || !is_free_of(expr, var, pool) {
            let (n, d) = expr_to_qrational(expr, var, pool)?;
            return Some(AlgFrac::rational(n, d));
        }
        // A `var`-free non-numeric factor (`√2`, `π`, …) has no `QPoly` form.
        return None;
    }
    let out = match pool.get(expr) {
        ExprData::Add(args) => {
            let mut acc = AlgFrac::zero();
            for a in &args {
                acc = acc.add(to_alg(*a, sqrt_id, p, var, pool)?);
                if acc.too_big() {
                    return None;
                }
            }
            acc
        }
        ExprData::Mul(args) => {
            let mut acc = AlgFrac::one();
            for a in &args {
                acc = acc.mul(to_alg(*a, sqrt_id, p, var, pool)?, p);
                if acc.too_big() {
                    return None;
                }
            }
            acc
        }
        ExprData::Pow { base, exp } => {
            let n = as_integer(exp, pool)?;
            to_alg(base, sqrt_id, p, var, pool)?.powi(n, p)?
        }
        _ => return None,
    };
    Some(out)
}
