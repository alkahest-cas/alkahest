//! Hamilton quaternions, and the rotation operator built on them.
//!
//! # Why this is a type and not a rewrite rule
//!
//! [`super::noncommutative`] handles Pauli and Clifford generators by matching
//! products of *non-commutative symbols* and firing a product table. That is
//! the right shape when the generators are opaque and a user writes
//! `σx·σy` by hand. It is the wrong shape here.
//!
//! A quaternion is used numerically as often as symbolically — an attitude
//! state is four floats — and every operation a GNC engineer wants
//! (`q₁q₂`, `q⁻¹`, `q v q⁻¹`, the rotation matrix) is a closed-form function of
//! the four components, not a normal form to be searched for. Expressing
//! `q v q⁻¹` as a rewrite problem over `1, i, j, k` means interning a product
//! of three general quaternions as an unexpanded tree and hoping the rules
//! reach a canonical form; expressing it as arithmetic on four `ExprId`s makes
//! it one deterministic evaluation that cannot half-finish. The
//! non-commutativity lives in [`Quaternion::mul`]'s formula, where it is
//! visible and testable, rather than in the order the simplifier happens to
//! visit a `Mul`.
//!
//! # Conventions
//!
//! * `q = w + xi + yj + zk` with `i² = j² = k² = ijk = −1`, hence
//!   `ij = k`, `ji = −k` ([`Quaternion::mul`] is asserted against both).
//! * [`Quaternion::rotate`] is the **active** rotation `v ↦ q v q⁻¹` in a
//!   right-handed frame: it moves the vector, it does not re-express it in a
//!   rotated frame. The inverse convention differs by a transpose, which is
//!   the single most common silent sign error in attitude code.
//! * Composition follows from that and is **not** symmetric:
//!   `(q₁q₂) v (q₁q₂)⁻¹ = q₁ (q₂ v q₂⁻¹) q₁⁻¹`, so `q₂` acts first and
//!   `R(q₁q₂) = R(q₁)·R(q₂)`. A test asserts exactly this, at random axes and
//!   angles, and asserts that the reversed product does *not* agree.
//! * `q` and `−q` are the same rotation. Nothing here canonicalises the sign.

use crate::jit::eval_interp_checked;
use crate::kernel::{ExprId, ExprPool};
use crate::matrix::zero_test::{zero_status, ZeroStatus};
use crate::matrix::{Matrix, MatrixError};
use crate::simplify::engine::simplify;
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a quaternion operation refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuaternionError {
    /// `|q|²` is zero, or could not be shown to be non-zero, and the operation
    /// divides by it.
    ///
    /// `q⁻¹ = q̄/|q|²`, so a zero norm has no inverse and no rotation operator.
    /// Over the reals only `q = 0` has `|q|² = 0`, but the components are
    /// arbitrary expressions here, so the question is a zero test and its
    /// undecided answer is a refusal.
    ZeroNorm {
        /// Rendered form of `|q|²`.
        norm_squared: String,
        /// `true` when `|q|²` was *proven* zero, `false` when undecided.
        proven_zero: bool,
    },
    /// The rotation has no axis: the vector part of `q` is zero, so `q` is a
    /// real scalar and `q v q⁻¹ = v` for every `v`.
    ///
    /// Every unit vector is then a valid axis, so there is no axis to return.
    /// Returning `(0,0,1)` — the conventional stand-in — is a stated answer to
    /// a question with no answer, which is the failure mode this crate refuses
    /// by policy.
    UndefinedAxis {
        /// Rendered form of the quaternion.
        quaternion: String,
    },
    /// A matrix handed to [`Quaternion::from_rotation_matrix`] is not a proper
    /// rotation, or its entries are not numbers.
    NotARotation {
        /// What failed: not 3×3, symbolic entries, `RᵀR ≠ I`, `det R = −1`, or
        /// the reconstruction check.
        reason: String,
    },
}

impl fmt::Display for QuaternionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuaternionError::ZeroNorm {
                norm_squared,
                proven_zero,
            } => {
                if *proven_zero {
                    write!(
                        f,
                        "quaternion has zero norm (|q|² = `{norm_squared}`); it has no inverse"
                    )
                } else {
                    write!(
                        f,
                        "|q|² = `{norm_squared}` could not be shown to be non-zero, and this operation divides by it"
                    )
                }
            }
            QuaternionError::UndefinedAxis { quaternion } => write!(
                f,
                "`{quaternion}` has zero vector part: it is the identity rotation, and every unit vector is an axis for it"
            ),
            QuaternionError::NotARotation { reason } => {
                write!(f, "not a proper rotation matrix: {reason}")
            }
        }
    }
}

impl std::error::Error for QuaternionError {}

impl crate::errors::AlkahestError for QuaternionError {
    fn code(&self) -> &'static str {
        match self {
            QuaternionError::ZeroNorm { .. } => "E-QUAT-001",
            QuaternionError::UndefinedAxis { .. } => "E-QUAT-002",
            QuaternionError::NotARotation { .. } => "E-QUAT-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            QuaternionError::ZeroNorm { .. } => Some(
                "use a non-zero quaternion; if the components are symbolic, substitute concrete values or declare the parameters so the norm can be decided",
            ),
            QuaternionError::UndefinedAxis { .. } => Some(
                "the rotation is the identity — report the angle as 0 and pick whatever axis your convention prefers, explicitly, at the call site",
            ),
            QuaternionError::NotARotation { .. } => Some(
                "pass a 3×3 matrix of numbers with RᵀR = I and det R = +1; a reflection (det = −1) is not a rotation and has no quaternion",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// The type
// ---------------------------------------------------------------------------

/// A quaternion `w + xi + yj + zk` over symbolic expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quaternion {
    w: ExprId,
    x: ExprId,
    y: ExprId,
    z: ExprId,
}

/// Numeric tolerance for the `RᵀR = I`, `det R = 1` and reconstruction checks
/// in [`Quaternion::from_rotation_matrix`].
///
/// Loose enough to accept a matrix assembled from `f64` trigonometry, tight
/// enough that a matrix that is not a rotation cannot slip through: the
/// nearest non-rotation to a rotation in these norms is a scaling or a shear,
/// both of which move the residual by many orders of magnitude more than this.
const ROTATION_TOL: f64 = 1e-9;

impl Quaternion {
    /// `w + xi + yj + zk` from its four components.
    pub fn new(w: ExprId, x: ExprId, y: ExprId, z: ExprId) -> Self {
        Quaternion { w, x, y, z }
    }

    /// The real quaternion `w`.
    pub fn scalar(w: ExprId, pool: &ExprPool) -> Self {
        let zero = pool.integer(0_i32);
        Quaternion {
            w,
            x: zero,
            y: zero,
            z: zero,
        }
    }

    /// The pure (vector) quaternion `0 + v₁i + v₂j + v₃k`.
    pub fn pure(v: &[ExprId; 3], pool: &ExprPool) -> Self {
        Quaternion {
            w: pool.integer(0_i32),
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }

    /// The multiplicative identity `1`.
    pub fn identity(pool: &ExprPool) -> Self {
        Quaternion::scalar(pool.integer(1_i32), pool)
    }

    /// The real part `w`.
    pub fn w(&self) -> ExprId {
        self.w
    }

    /// The `i` component.
    pub fn x(&self) -> ExprId {
        self.x
    }

    /// The `j` component.
    pub fn y(&self) -> ExprId {
        self.y
    }

    /// The `k` component.
    pub fn z(&self) -> ExprId {
        self.z
    }

    /// `[w, x, y, z]`, scalar first.
    pub fn components(&self) -> [ExprId; 4] {
        [self.w, self.x, self.y, self.z]
    }

    /// The vector part `[x, y, z]`.
    pub fn vector_part(&self) -> [ExprId; 3] {
        [self.x, self.y, self.z]
    }

    /// Simplify all four components.
    pub fn simplified(&self, pool: &ExprPool) -> Self {
        Quaternion {
            w: simplify(self.w, pool).value,
            x: simplify(self.x, pool).value,
            y: simplify(self.y, pool).value,
            z: simplify(self.z, pool).value,
        }
    }

    /// Componentwise sum.
    pub fn add(&self, other: &Quaternion, pool: &ExprPool) -> Self {
        Quaternion {
            w: simplify(pool.add(vec![self.w, other.w]), pool).value,
            x: simplify(pool.add(vec![self.x, other.x]), pool).value,
            y: simplify(pool.add(vec![self.y, other.y]), pool).value,
            z: simplify(pool.add(vec![self.z, other.z]), pool).value,
        }
    }

    /// Componentwise difference.
    pub fn sub(&self, other: &Quaternion, pool: &ExprPool) -> Self {
        let neg = pool.integer(-1_i32);
        let negated = other.scale(neg, pool);
        self.add(&negated, pool)
    }

    /// Multiply every component by the scalar `s`.
    pub fn scale(&self, s: ExprId, pool: &ExprPool) -> Self {
        Quaternion {
            w: simplify(pool.mul(vec![s, self.w]), pool).value,
            x: simplify(pool.mul(vec![s, self.x]), pool).value,
            y: simplify(pool.mul(vec![s, self.y]), pool).value,
            z: simplify(pool.mul(vec![s, self.z]), pool).value,
        }
    }

    /// The **Hamilton product** `self · other`. Non-commutative.
    ///
    /// ```text
    /// w = w₁w₂ − x₁x₂ − y₁y₂ − z₁z₂
    /// x = w₁x₂ + x₁w₂ + y₁z₂ − z₁y₂
    /// y = w₁y₂ − x₁z₂ + y₁w₂ + z₁x₂
    /// z = w₁z₂ + x₁y₂ − y₁x₂ + z₁w₂
    /// ```
    ///
    /// Equivalently `(s₁, v₁)(s₂, v₂) = (s₁s₂ − v₁·v₂, s₁v₂ + s₂v₁ + v₁×v₂)`;
    /// the cross product is what makes `ij = k` and `ji = −k`.
    pub fn mul(&self, other: &Quaternion, pool: &ExprPool) -> Self {
        let (a, b) = (self, other);
        let neg = pool.integer(-1_i32);
        let m = |p: ExprId, q: ExprId| pool.mul(vec![p, q]);
        let nm = |p: ExprId, q: ExprId| pool.mul(vec![neg, p, q]);

        let w = pool.add(vec![m(a.w, b.w), nm(a.x, b.x), nm(a.y, b.y), nm(a.z, b.z)]);
        let x = pool.add(vec![m(a.w, b.x), m(a.x, b.w), m(a.y, b.z), nm(a.z, b.y)]);
        let y = pool.add(vec![m(a.w, b.y), nm(a.x, b.z), m(a.y, b.w), m(a.z, b.x)]);
        let z = pool.add(vec![m(a.w, b.z), m(a.x, b.y), nm(a.y, b.x), m(a.z, b.w)]);

        Quaternion {
            w: simplify(w, pool).value,
            x: simplify(x, pool).value,
            y: simplify(y, pool).value,
            z: simplify(z, pool).value,
        }
    }

    /// `q̄ = w − xi − yj − zk`.
    pub fn conjugate(&self, pool: &ExprPool) -> Self {
        let neg = pool.integer(-1_i32);
        Quaternion {
            w: self.w,
            x: simplify(pool.mul(vec![neg, self.x]), pool).value,
            y: simplify(pool.mul(vec![neg, self.y]), pool).value,
            z: simplify(pool.mul(vec![neg, self.z]), pool).value,
        }
    }

    /// `|q|² = w² + x² + y² + z²`, simplified.
    pub fn norm_squared(&self, pool: &ExprPool) -> ExprId {
        let two = pool.integer(2_i32);
        let terms: Vec<ExprId> = self
            .components()
            .iter()
            .map(|&c| pool.pow(c, two))
            .collect();
        simplify(pool.add(terms), pool).value
    }

    /// `|q| = √(w² + x² + y² + z²)`, simplified.
    pub fn norm(&self, pool: &ExprPool) -> ExprId {
        let n2 = self.norm_squared(pool);
        simplify(pool.func("sqrt", vec![n2]), pool).value
    }

    /// `|q|²`, but only once it is known not to vanish.
    fn nonzero_norm_squared(&self, pool: &ExprPool) -> Result<ExprId, QuaternionError> {
        let n2 = self.norm_squared(pool);
        match zero_status(pool, n2) {
            ZeroStatus::NonZero => Ok(n2),
            status => Err(QuaternionError::ZeroNorm {
                norm_squared: pool.display(n2).to_string(),
                proven_zero: status == ZeroStatus::Zero,
            }),
        }
    }

    /// `q⁻¹ = q̄ / |q|²`.
    ///
    /// # Errors
    ///
    /// `E-QUAT-001` when `|q|²` is zero or its vanishing is undecided.
    pub fn inverse(&self, pool: &ExprPool) -> Result<Self, QuaternionError> {
        let n2 = self.nonzero_norm_squared(pool)?;
        let inv = pool.pow(n2, pool.integer(-1_i32));
        Ok(self.conjugate(pool).scale(inv, pool))
    }

    /// `q / |q|` — the unit quaternion with the same rotation.
    ///
    /// # Errors
    ///
    /// `E-QUAT-001` when `|q|²` is zero or its vanishing is undecided.
    pub fn normalize(&self, pool: &ExprPool) -> Result<Self, QuaternionError> {
        let n2 = self.nonzero_norm_squared(pool)?;
        let n = simplify(pool.func("sqrt", vec![n2]), pool).value;
        let inv = pool.pow(n, pool.integer(-1_i32));
        Ok(self.scale(inv, pool))
    }

    /// The **active** rotation `v ↦ q v q⁻¹`, returned as a 3-vector.
    ///
    /// `q` need not be a unit quaternion: the operator is invariant under
    /// `q ↦ λq`, because `λ` cancels against `λ⁻¹` in `q⁻¹`. The scalar part of
    /// `q v q⁻¹` is identically zero, so only the vector part is returned.
    ///
    /// # Errors
    ///
    /// `E-QUAT-001` when `|q|²` is zero or its vanishing is undecided.
    pub fn rotate(&self, v: &[ExprId; 3], pool: &ExprPool) -> Result<[ExprId; 3], QuaternionError> {
        let inv = self.inverse(pool)?;
        let p = Quaternion::pure(v, pool);
        let out = self.mul(&p, pool).mul(&inv, pool);
        Ok(out.vector_part())
    }

    /// The 3×3 rotation matrix `R` with `R v = q v q⁻¹`.
    ///
    /// ```text
    ///       1   ⎡ w²+x²−y²−z²   2(xy−wz)      2(xz+wy)    ⎤
    /// R = ───── ⎢ 2(xy+wz)      w²−x²+y²−z²   2(yz−wx)    ⎥
    ///     |q|²  ⎣ 2(xz−wy)      2(yz+wx)      w²−x²−y²+z² ⎦
    /// ```
    ///
    /// For a unit quaternion the `1/|q|²` is `1` and the diagonal reads
    /// `1 − 2(y²+z²)`, etc. The agreement `R v = q v q⁻¹` is asserted in the
    /// tests at random axes and angles, not assumed.
    ///
    /// # Errors
    ///
    /// `E-QUAT-001` when `|q|²` is zero or its vanishing is undecided.
    pub fn to_rotation_matrix(&self, pool: &ExprPool) -> Result<Matrix, QuaternionError> {
        let n2 = self.nonzero_norm_squared(pool)?;
        let inv = pool.pow(n2, pool.integer(-1_i32));
        let two = pool.integer(2_i32);
        let neg = pool.integer(-1_i32);
        let sq = |c: ExprId| pool.pow(c, two);
        let nsq = |c: ExprId| pool.mul(vec![neg, pool.pow(c, two)]);
        let p2 = |a: ExprId, b: ExprId| pool.mul(vec![two, a, b]);
        let m2 = |a: ExprId, b: ExprId| pool.mul(vec![neg, two, a, b]);
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);

        let raw = [
            [
                pool.add(vec![sq(w), sq(x), nsq(y), nsq(z)]),
                pool.add(vec![p2(x, y), m2(w, z)]),
                pool.add(vec![p2(x, z), p2(w, y)]),
            ],
            [
                pool.add(vec![p2(x, y), p2(w, z)]),
                pool.add(vec![sq(w), nsq(x), sq(y), nsq(z)]),
                pool.add(vec![p2(y, z), m2(w, x)]),
            ],
            [
                pool.add(vec![p2(x, z), m2(w, y)]),
                pool.add(vec![p2(y, z), p2(w, x)]),
                pool.add(vec![sq(w), nsq(x), nsq(y), sq(z)]),
            ],
        ];

        let rows: Vec<Vec<ExprId>> = raw
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&e| simplify(pool.mul(vec![inv, e]), pool).value)
                    .collect()
            })
            .collect();
        Matrix::new(rows).map_err(|e: MatrixError| QuaternionError::NotARotation {
            reason: e.to_string(),
        })
    }

    /// `q = cos(θ/2) + sin(θ/2)·û` for the rotation of `angle` about `axis`.
    ///
    /// `axis` is normalised internally, so it need not be a unit vector; the
    /// literal-`1` case (`(0,0,1)` and friends) skips the normalisation so the
    /// result stays readable. The returned quaternion is a unit quaternion and
    /// rotates by `angle` **about `axis`, right-handed**, in the sense of
    /// [`Quaternion::rotate`].
    ///
    /// # Errors
    ///
    /// `E-QUAT-001` when `|axis|²` is zero or its vanishing is undecided — the
    /// zero vector picks out no axis.
    pub fn from_axis_angle(
        axis: &[ExprId; 3],
        angle: ExprId,
        pool: &ExprPool,
    ) -> Result<Self, QuaternionError> {
        let n2 = crate::vector::norm_squared(axis, pool);
        let unit = match zero_status(pool, n2) {
            ZeroStatus::NonZero => {
                let one = pool.integer(1_i32);
                if n2 == one {
                    *axis
                } else {
                    let n = simplify(pool.func("sqrt", vec![n2]), pool).value;
                    let inv = pool.pow(n, pool.integer(-1_i32));
                    crate::vector::scale(inv, axis, pool)
                }
            }
            status => {
                return Err(QuaternionError::ZeroNorm {
                    norm_squared: pool.display(n2).to_string(),
                    proven_zero: status == ZeroStatus::Zero,
                })
            }
        };
        let half = pool.mul(vec![pool.rational(1, 2), angle]);
        let c = simplify(pool.func("cos", vec![half]), pool).value;
        let s = simplify(pool.func("sin", vec![half]), pool).value;
        Ok(Quaternion {
            w: c,
            x: simplify(pool.mul(vec![s, unit[0]]), pool).value,
            y: simplify(pool.mul(vec![s, unit[1]]), pool).value,
            z: simplify(pool.mul(vec![s, unit[2]]), pool).value,
        })
    }

    /// The axis–angle form: `(û, θ)` with `θ = 2·atan2(|v|, w)` and
    /// `û = v/|v|`, where `v` is the vector part.
    ///
    /// `atan2` rather than `acos(w/|q|)` because `acos` loses all precision as
    /// `θ → 0`, which is where attitude code spends most of its time, and
    /// because it does not need `q` normalised first.
    ///
    /// Since `|v| ≥ 0` the returned `θ` lies in `[0, 2π]`: a quaternion with
    /// `w < 0` reports an angle past `π` about the returned axis rather than
    /// the equivalent angle `2π − θ` about `−û`. Both describe the same
    /// rotation; nothing here canonicalises, because `q` and `−q` are the same
    /// rotation and choosing between them is the caller's convention.
    ///
    /// # Errors
    ///
    /// `E-QUAT-002` when the vector part is zero or its vanishing is
    /// undecided: `q` is then a real scalar, the rotation is the identity, and
    /// *every* unit vector is an axis for it. There is no axis to return.
    pub fn to_axis_angle(&self, pool: &ExprPool) -> Result<([ExprId; 3], ExprId), QuaternionError> {
        let v = self.vector_part();
        let n2 = crate::vector::norm_squared(&v, pool);
        if zero_status(pool, n2) != ZeroStatus::NonZero {
            return Err(QuaternionError::UndefinedAxis {
                quaternion: format!(
                    "{} + {}i + {}j + {}k",
                    pool.display(self.w),
                    pool.display(self.x),
                    pool.display(self.y),
                    pool.display(self.z)
                ),
            });
        }
        let n = simplify(pool.func("sqrt", vec![n2]), pool).value;
        let inv = pool.pow(n, pool.integer(-1_i32));
        let axis = crate::vector::scale(inv, &v, pool);
        let angle = simplify(
            pool.mul(vec![
                pool.integer(2_i32),
                pool.func("atan2", vec![n, self.w]),
            ]),
            pool,
        )
        .value;
        Ok((axis, angle))
    }

    /// Recover a quaternion from a **numeric** proper rotation matrix.
    ///
    /// Shepperd's method: the branch with the largest pivot is taken, so no
    /// square root of a near-zero quantity is ever divided by. The result is
    /// the unit quaternion `q` with `R(q) = R`, up to the unavoidable sign
    /// ambiguity (`q` and `−q` give the same `R`); the branch is chosen to
    /// make the pivot component positive.
    ///
    /// Three things are checked before anything is returned, and each is a
    /// refusal:
    ///
    /// 1. every entry is a number — the branch selection is a comparison, and
    ///    there is no comparison to make on symbols;
    /// 2. `RᵀR = I` and `det R = +1` — a reflection has no quaternion, and a
    ///    scaled or skewed matrix would otherwise yield a plausible unit
    ///    quaternion for a transform that is not a rotation;
    /// 3. `R(q)` is rebuilt and compared against the input entry by entry.
    ///
    /// # Errors
    ///
    /// `E-QUAT-003` for any of the three.
    pub fn from_rotation_matrix(m: &Matrix, pool: &ExprPool) -> Result<Self, QuaternionError> {
        if m.rows != 3 || m.cols != 3 {
            return Err(QuaternionError::NotARotation {
                reason: format!("expected a 3×3 matrix, got {}×{}", m.rows, m.cols),
            });
        }
        let mut r = [[0.0_f64; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let entry = m.get(i, j);
                match eval_interp_checked(entry, &std::collections::HashMap::new(), pool) {
                    Ok(v) if v.is_finite() => *cell = v,
                    _ => {
                        return Err(QuaternionError::NotARotation {
                            reason: format!(
                                "entry ({i},{j}) = `{}` is not a finite number; the branch \
                                 selection is a comparison between entries and there is none \
                                 to make on a symbol",
                                pool.display(entry)
                            ),
                        })
                    }
                }
            }
        }
        check_proper_rotation(&r)?;

        let q = shepperd(&r);
        // Rebuild and compare: the branch algebra is where this goes wrong.
        let rebuilt = rotation_matrix_f64(&q);
        let worst = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (rebuilt[i][j] - r[i][j]).abs())
            .fold(0.0_f64, f64::max);
        if worst > ROTATION_TOL {
            return Err(QuaternionError::NotARotation {
                reason: format!(
                    "the recovered quaternion does not reproduce the matrix \
                     (worst entry discrepancy {worst:.3e} > {ROTATION_TOL:.0e})"
                ),
            });
        }

        Ok(Quaternion {
            w: pool.float(q[0], 53),
            x: pool.float(q[1], 53),
            y: pool.float(q[2], 53),
            z: pool.float(q[3], 53),
        })
    }
}

fn check_proper_rotation(r: &[[f64; 3]; 3]) -> Result<(), QuaternionError> {
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let dot: f64 = (0..3).map(|k| r[k][i] * r[k][j]).sum();
            let target = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - target).abs());
        }
    }
    if worst > ROTATION_TOL {
        return Err(QuaternionError::NotARotation {
            reason: format!(
                "columns are not orthonormal: worst |RᵀR − I| entry is {worst:.3e} \
                 (tolerance {ROTATION_TOL:.0e})"
            ),
        });
    }
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    if (det - 1.0).abs() > ROTATION_TOL {
        return Err(QuaternionError::NotARotation {
            reason: format!(
                "det R = {det:.6}, not +1; an orthogonal matrix with det = −1 is a \
                 reflection and is not in the image of the quaternion rotation map"
            ),
        });
    }
    Ok(())
}

/// Shepperd's branch-selecting matrix-to-quaternion conversion. Returns
/// `[w, x, y, z]`.
fn shepperd(r: &[[f64; 3]; 3]) -> [f64; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    }
}

/// `R(q)` for a **unit** `q = [w, x, y, z]`, in `f64`. The `f64` mirror of
/// [`Quaternion::to_rotation_matrix`], used to re-check
/// [`Quaternion::from_rotation_matrix`]'s answer against its input.
fn rotation_matrix_f64(q: &[f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = *q;
    let n2 = w * w + x * x + y * y + z * z;
    let inv = 1.0 / n2;
    [
        [
            (w * w + x * x - y * y - z * z) * inv,
            2.0 * (x * y - w * z) * inv,
            2.0 * (x * z + w * y) * inv,
        ],
        [
            2.0 * (x * y + w * z) * inv,
            (w * w - x * x + y * y - z * z) * inv,
            2.0 * (y * z - w * x) * inv,
        ],
        [
            2.0 * (x * z - w * y) * inv,
            2.0 * (y * z + w * x) * inv,
            (w * w - x * x - y * y + z * z) * inv,
        ],
    ]
}
