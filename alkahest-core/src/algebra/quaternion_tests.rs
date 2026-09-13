//! Quaternion tests: the algebra, the rotation operator, and the order.
//!
//! The rotation tests do not check `q v q⁻¹` against another quaternion
//! routine — that would only prove this module is self-consistent. They check
//! it against **Rodrigues' rotation formula**
//! `v cos θ + (n̂ × v) sin θ + n̂ (n̂·v)(1 − cos θ)`, which is derived
//! independently of quaternions, and against the rotation matrix this module
//! builds by a separate formula.
//!
//! Composition order gets its own test that asserts both halves: that
//! `R(q₁q₂) = R(q₁)R(q₂)` and that `R(q₁q₂) ≠ R(q₂)R(q₁)` for the sample
//! rotations used. A convention error that reversed the order would satisfy
//! the first only if both sides were reversed together, which the Rodrigues
//! anchor rules out.

use super::quaternion::{Quaternion, QuaternionError};
use crate::errors::AlkahestError;
use crate::jit::eval_interp_checked;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::matrix::Matrix;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        lo + (hi - lo) * u
    }
}

fn num(e: ExprId, pool: &ExprPool) -> f64 {
    eval_interp_checked(e, &HashMap::new(), pool)
        .unwrap_or_else(|err| panic!("could not evaluate `{}`: {err:?}", pool.display(e)))
}

fn lit(v: f64, pool: &ExprPool) -> ExprId {
    pool.float(v, 53)
}

fn qnum(c: [f64; 4], pool: &ExprPool) -> Quaternion {
    Quaternion::new(
        lit(c[0], pool),
        lit(c[1], pool),
        lit(c[2], pool),
        lit(c[3], pool),
    )
}

fn vnum(v: [f64; 3], pool: &ExprPool) -> [ExprId; 3] {
    [lit(v[0], pool), lit(v[1], pool), lit(v[2], pool)]
}

fn qvals(q: &Quaternion, pool: &ExprPool) -> [f64; 4] {
    let c = q.components();
    [
        num(c[0], pool),
        num(c[1], pool),
        num(c[2], pool),
        num(c[3], pool),
    ]
}

fn vvals(v: &[ExprId; 3], pool: &ExprPool) -> [f64; 3] {
    [num(v[0], pool), num(v[1], pool), num(v[2], pool)]
}

fn mvals(m: &Matrix, pool: &ExprPool) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = num(m.get(i, j), pool);
        }
    }
    out
}

#[track_caller]
fn close(a: f64, b: f64, what: &str) {
    let scale = 1.0_f64.max(a.abs()).max(b.abs());
    assert!(
        (a - b).abs() <= 1e-9 * scale,
        "{what}: {a} vs {b} (gap {:.3e})",
        (a - b).abs()
    );
}

#[track_caller]
fn close_v(a: [f64; 3], b: [f64; 3], what: &str) {
    for i in 0..3 {
        close(a[i], b[i], &format!("{what}[{i}]"));
    }
}

/// Rodrigues' rotation formula — the independent oracle for `q v q⁻¹`.
fn rodrigues(axis: [f64; 3], angle: f64, v: [f64; 3]) -> [f64; 3] {
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let n = [axis[0] / len, axis[1] / len, axis[2] / len];
    let (c, s) = (angle.cos(), angle.sin());
    let cross = [
        n[1] * v[2] - n[2] * v[1],
        n[2] * v[0] - n[0] * v[2],
        n[0] * v[1] - n[1] * v[0],
    ];
    let dot = n[0] * v[0] + n[1] * v[1] + n[2] * v[2];
    [
        v[0] * c + cross[0] * s + n[0] * dot * (1.0 - c),
        v[1] * c + cross[1] * s + n[1] * dot * (1.0 - c),
        v[2] * c + cross[2] * s + n[2] * dot * (1.0 - c),
    ]
}

fn matvec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// A symbolic quaternion with four free real symbols.
fn symbolic(prefix: &str, pool: &ExprPool) -> Quaternion {
    Quaternion::new(
        pool.symbol(format!("{prefix}w"), Domain::Real),
        pool.symbol(format!("{prefix}x"), Domain::Real),
        pool.symbol(format!("{prefix}y"), Domain::Real),
        pool.symbol(format!("{prefix}z"), Domain::Real),
    )
}

fn bind(q: &Quaternion, vals: [f64; 4], env: &mut HashMap<ExprId, f64>) {
    for (id, v) in q.components().into_iter().zip(vals) {
        env.insert(id, v);
    }
}

// ---------------------------------------------------------------------------
// The algebra
// ---------------------------------------------------------------------------

#[test]
fn the_hamilton_product_table() {
    let p = ExprPool::new();
    let (o, l) = (p.integer(0_i32), p.integer(1_i32));
    let neg = p.integer(-1_i32);
    let one = Quaternion::new(l, o, o, o);
    let i = Quaternion::new(o, l, o, o);
    let j = Quaternion::new(o, o, l, o);
    let k = Quaternion::new(o, o, o, l);
    let minus_one = Quaternion::new(neg, o, o, o);
    let minus = |q: &Quaternion| q.scale(neg, &p);

    assert_eq!(i.mul(&j, &p), k, "ij = k");
    assert_eq!(j.mul(&i, &p), minus(&k), "ji = -k");
    assert_eq!(j.mul(&k, &p), i, "jk = i");
    assert_eq!(k.mul(&j, &p), minus(&i), "kj = -i");
    assert_eq!(k.mul(&i, &p), j, "ki = j");
    assert_eq!(i.mul(&k, &p), minus(&j), "ik = -j");
    assert_eq!(i.mul(&i, &p), minus_one, "i² = -1");
    assert_eq!(j.mul(&j, &p), minus_one, "j² = -1");
    assert_eq!(k.mul(&k, &p), minus_one, "k² = -1");
    assert_eq!(
        i.mul(&j, &p).mul(&k, &p),
        minus_one,
        "ijk = -1 (Hamilton's bridge relation)"
    );
    assert_eq!(one.mul(&i, &p), i, "1 is the identity");
    assert_eq!(i.mul(&one, &p), i, "1 is the identity on the right too");
}

#[test]
fn the_hamilton_product_does_not_commute() {
    let p = ExprPool::new();
    let a = symbolic("a", &p);
    let b = symbolic("b", &p);
    let ab = a.mul(&b, &p);
    let ba = b.mul(&a, &p);
    assert_ne!(
        ab, ba,
        "a general quaternion product must depend on the order of its factors"
    );
    // …and the difference is a real one, not a formatting artefact.
    let mut rng = Rng::new(0x1D1D_1D1D);
    let mut env = HashMap::new();
    bind(
        &a,
        [
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
        ],
        &mut env,
    );
    bind(
        &b,
        [
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
        ],
        &mut env,
    );
    let differs = (0..4).any(|i| {
        let l = eval_interp_checked(ab.components()[i], &env, &p).unwrap();
        let r = eval_interp_checked(ba.components()[i], &env, &p).unwrap();
        (l - r).abs() > 1e-9
    });
    assert!(differs, "ab and ba must differ numerically as well");
}

#[test]
fn the_product_is_associative() {
    let p = ExprPool::new();
    let mut rng = Rng::new(0x0A55_0C1A);
    for _ in 0..24 {
        let mk = |rng: &mut Rng| {
            qnum(
                [
                    rng.range(-2.0, 2.0),
                    rng.range(-2.0, 2.0),
                    rng.range(-2.0, 2.0),
                    rng.range(-2.0, 2.0),
                ],
                &p,
            )
        };
        let (a, b, c) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
        let left = a.mul(&b, &p).mul(&c, &p);
        let right = a.mul(&b.mul(&c, &p), &p);
        let (lv, rv) = (qvals(&left, &p), qvals(&right, &p));
        for i in 0..4 {
            close(lv[i], rv[i], &format!("(ab)c vs a(bc), component {i}"));
        }
    }
}

#[test]
fn the_norm_is_multiplicative() {
    // |q₁q₂| = |q₁||q₂| — Euler's four-square identity. Checked symbolically
    // first (the identity is polynomial, so the zero test can settle it) and
    // then numerically at random points.
    let p = ExprPool::new();
    let a = symbolic("a", &p);
    let b = symbolic("b", &p);
    let lhs = a.mul(&b, &p).norm_squared(&p);
    let rhs = p.mul(vec![a.norm_squared(&p), b.norm_squared(&p)]);
    let residual = p.add(vec![lhs, p.mul(vec![p.integer(-1_i32), rhs])]);
    assert_eq!(
        crate::matrix::zero_test::zero_status(&p, residual),
        crate::matrix::zero_test::ZeroStatus::Zero,
        "|q₁q₂|² − |q₁|²|q₂|² must be provably zero"
    );

    let mut rng = Rng::new(0x4A4A_4A4A);
    for _ in 0..24 {
        let mk = |rng: &mut Rng| {
            qnum(
                [
                    rng.range(-2.0, 2.0),
                    rng.range(-2.0, 2.0),
                    rng.range(-2.0, 2.0),
                    rng.range(-2.0, 2.0),
                ],
                &p,
            )
        };
        let (q1, q2) = (mk(&mut rng), mk(&mut rng));
        let prod = num(q1.mul(&q2, &p).norm(&p), &p);
        let separate = num(q1.norm(&p), &p) * num(q2.norm(&p), &p);
        close(prod, separate, "|q₁q₂| = |q₁||q₂|");
    }
}

#[test]
fn a_quaternion_times_its_inverse_is_one() {
    // For a *symbolic* `q` this is a rational-function identity: every
    // component of `q q⁻¹` carries a `(w²+x²+y²+z²)⁻¹` factor that the
    // simplifier does not cancel against the numerator, so the zero test
    // returns `Unknown` rather than `Zero`. It is checked numerically at random
    // points instead, on both sides, and exactly for a concrete quaternion.
    let p = ExprPool::new();
    let q = symbolic("q", &p);
    let inv = q
        .inverse(&p)
        .expect("a symbolic quaternion has a generic inverse");
    let one = [1.0, 0.0, 0.0, 0.0];
    let mut rng = Rng::new(0x1111_0001);
    for _ in 0..20 {
        let mut env = HashMap::new();
        bind(
            &q,
            [
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
            ],
            &mut env,
        );
        for (label, product) in [("q q⁻¹", q.mul(&inv, &p)), ("q⁻¹ q", inv.mul(&q, &p))] {
            for (i, &c) in product.components().iter().enumerate() {
                close(
                    eval_interp_checked(c, &env, &p).unwrap(),
                    one[i],
                    &format!("{label} component {i}"),
                );
            }
        }
    }

    // The concrete case closes exactly: 1 + i + j + k has |q|² = 4.
    let l = p.integer(1_i32);
    let qc = Quaternion::new(l, l, l, l);
    let invc = qc.inverse(&p).unwrap();
    assert_eq!(
        qc.mul(&invc, &p),
        Quaternion::identity(&p),
        "(1+i+j+k)(1+i+j+k)⁻¹ must be exactly 1"
    );
    assert_eq!(
        invc.mul(&qc, &p),
        Quaternion::identity(&p),
        "…from the left too"
    );
}

#[test]
fn conjugation_reverses_a_product() {
    // conj(q₁q₂) = conj(q₂)·conj(q₁) — the order matters here too.
    let p = ExprPool::new();
    let a = symbolic("a", &p);
    let b = symbolic("b", &p);
    let lhs = a.mul(&b, &p).conjugate(&p);
    let rhs = b.conjugate(&p).mul(&a.conjugate(&p), &p);
    assert_eq!(lhs, rhs, "conj(ab) must equal conj(b)conj(a)");
    let wrong = a.conjugate(&p).mul(&b.conjugate(&p), &p);
    assert_ne!(
        lhs, wrong,
        "conj(a)conj(b) is the *other* order and must not coincide with it"
    );
}

#[test]
fn the_zero_quaternion_has_no_inverse() {
    let p = ExprPool::new();
    let zero = Quaternion::scalar(p.integer(0_i32), &p);
    let err = zero.inverse(&p).expect_err("0 has no inverse");
    assert_eq!(err.code(), "E-QUAT-001");
    assert!(matches!(
        err,
        QuaternionError::ZeroNorm {
            proven_zero: true,
            ..
        }
    ));
    assert!(zero.normalize(&p).is_err());
    assert!(zero.to_rotation_matrix(&p).is_err());
    let v = [p.integer(1_i32), p.integer(0_i32), p.integer(0_i32)];
    assert!(zero.rotate(&v, &p).is_err());
}

// ---------------------------------------------------------------------------
// Rotation
// ---------------------------------------------------------------------------

/// Random `(axis, angle, vector)` triples for the rotation tests.
fn rotation_samples(n: usize, seed: u64) -> Vec<([f64; 3], f64, [f64; 3])> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let axis = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            // Keep the axis away from the origin so `axis/|axis|` is stable.
            let axis = if axis.iter().map(|a| a * a).sum::<f64>() < 0.04 {
                [0.31, -0.77, 0.55]
            } else {
                axis
            };
            let angle = rng.range(-3.0, 3.0);
            let v = [
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
                rng.range(-2.0, 2.0),
            ];
            (axis, angle, v)
        })
        .collect()
}

#[test]
fn the_rotation_operator_agrees_with_rodrigues() {
    let p = ExprPool::new();
    for (axis, angle, v) in rotation_samples(40, 0x0D01_0D01) {
        let q = Quaternion::from_axis_angle(&vnum(axis, &p), lit(angle, &p), &p).unwrap();
        let got = vvals(&q.rotate(&vnum(v, &p), &p).unwrap(), &p);
        close_v(got, rodrigues(axis, angle, v), "q v q⁻¹ vs Rodrigues");
    }
}

#[test]
fn the_rotation_matrix_agrees_with_the_rotation_operator() {
    let p = ExprPool::new();
    for (axis, angle, v) in rotation_samples(40, 0x51D2) {
        let q = Quaternion::from_axis_angle(&vnum(axis, &p), lit(angle, &p), &p).unwrap();
        let operator = vvals(&q.rotate(&vnum(v, &p), &p).unwrap(), &p);
        let matrix = matvec(mvals(&q.to_rotation_matrix(&p).unwrap(), &p), v);
        close_v(matrix, operator, "R(q)·v vs q v q⁻¹");
        close_v(matrix, rodrigues(axis, angle, v), "R(q)·v vs Rodrigues");
    }
}

#[test]
fn the_rotation_matrix_of_a_non_unit_quaternion_is_still_a_rotation() {
    // `q v q⁻¹` is invariant under `q ↦ λq`; so must `to_rotation_matrix` be,
    // which is what the `1/|q|²` factor is for.
    let p = ExprPool::new();
    let mut rng = Rng::new(0x5CA1_AB1E);
    for (axis, angle, v) in rotation_samples(20, 0x5CA2) {
        let lambda = rng.range(0.3, 4.0);
        let unit = Quaternion::from_axis_angle(&vnum(axis, &p), lit(angle, &p), &p).unwrap();
        let scaled = unit.scale(lit(lambda, &p), &p);
        close_v(
            matvec(mvals(&scaled.to_rotation_matrix(&p).unwrap(), &p), v),
            rodrigues(axis, angle, v),
            "R(λq)·v",
        );
        close_v(
            vvals(&scaled.rotate(&vnum(v, &p), &p).unwrap(), &p),
            rodrigues(axis, angle, v),
            "(λq) v (λq)⁻¹",
        );
    }
}

#[test]
fn rotation_by_a_unit_quaternion_preserves_length() {
    let p = ExprPool::new();
    for (axis, angle, v) in rotation_samples(30, 0x1E47) {
        let q = Quaternion::from_axis_angle(&vnum(axis, &p), lit(angle, &p), &p).unwrap();
        let out = vvals(&q.rotate(&vnum(v, &p), &p).unwrap(), &p);
        let before = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let after = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt();
        close(after, before, "|q v q⁻¹| = |v|");
    }
}

#[test]
fn a_rotation_about_z_is_right_handed() {
    // The concrete orientation check: rotating x̂ by θ about ẑ must land on
    // (cos θ, sin θ, 0), not (cos θ, −sin θ, 0). One sign, and every attitude
    // derived from it, hangs on this.
    let p = ExprPool::new();
    let z = [p.integer(0_i32), p.integer(0_i32), p.integer(1_i32)];
    let xhat = [p.integer(1_i32), p.integer(0_i32), p.integer(0_i32)];
    for theta in [0.3_f64, 1.1, 2.4, -0.9] {
        let q = Quaternion::from_axis_angle(&z, lit(theta, &p), &p).unwrap();
        let out = vvals(&q.rotate(&xhat, &p).unwrap(), &p);
        close_v(out, [theta.cos(), theta.sin(), 0.0], "R_z(θ)·x̂");
    }
}

#[test]
fn composition_multiplies_the_matrices_in_the_same_order() {
    // The classic place to be silently wrong.
    //
    // `(q₁q₂) v (q₁q₂)⁻¹ = q₁ (q₂ v q₂⁻¹) q₁⁻¹`, so `q₂` acts first and
    // `R(q₁q₂) = R(q₁)·R(q₂)`. Both halves are asserted: that the forward
    // order agrees *and* that the reversed order does not, so a test that
    // reversed both conventions at once could not pass.
    let p = ExprPool::new();
    let mut rng = Rng::new(0x0DE4_0DE4);
    let mut reversed_disagreed = 0;
    let samples = 25;
    for _ in 0..samples {
        let mk = |rng: &mut Rng| {
            let axis = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            let axis = if axis.iter().map(|a| a * a).sum::<f64>() < 0.04 {
                [0.2, 0.9, -0.4]
            } else {
                axis
            };
            (axis, rng.range(-3.0, 3.0))
        };
        let (axis1, ang1) = mk(&mut rng);
        let (axis2, ang2) = mk(&mut rng);
        let v = [
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
            rng.range(-2.0, 2.0),
        ];
        let q1 = Quaternion::from_axis_angle(&vnum(axis1, &p), lit(ang1, &p), &p).unwrap();
        let q2 = Quaternion::from_axis_angle(&vnum(axis2, &p), lit(ang2, &p), &p).unwrap();

        let r1 = mvals(&q1.to_rotation_matrix(&p).unwrap(), &p);
        let r2 = mvals(&q2.to_rotation_matrix(&p).unwrap(), &p);
        let composed = mvals(&q1.mul(&q2, &p).to_rotation_matrix(&p).unwrap(), &p);

        // R(q₁q₂)·v = R(q₁)·(R(q₂)·v): q₂ acts first.
        close_v(
            matvec(composed, v),
            matvec(r1, matvec(r2, v)),
            "R(q₁q₂)·v vs R(q₁)(R(q₂)v)",
        );
        // …and the same statement through the operator, with no matrix in it.
        close_v(
            vvals(&q1.mul(&q2, &p).rotate(&vnum(v, &p), &p).unwrap(), &p),
            vvals(
                &q1.rotate(&q2.rotate(&vnum(v, &p), &p).unwrap(), &p)
                    .unwrap(),
                &p,
            ),
            "(q₁q₂) v (q₁q₂)⁻¹ vs q₁(q₂ v q₂⁻¹)q₁⁻¹",
        );

        // The reversed order is a genuinely different rotation for
        // non-parallel axes. Counted rather than asserted per sample, because
        // two rotations about (nearly) the same axis do commute.
        let other = matvec(r2, matvec(r1, v));
        if (0..3).any(|i| (other[i] - matvec(composed, v)[i]).abs() > 1e-6) {
            reversed_disagreed += 1;
        }
    }
    assert!(
        reversed_disagreed >= samples * 3 / 4,
        "R(q₂)R(q₁) agreed with R(q₁q₂) in {} of {samples} samples — composition is \
         being applied symmetrically, which means the order convention is not being \
         tested at all",
        samples - reversed_disagreed
    );
}

#[test]
fn q_and_minus_q_are_the_same_rotation() {
    let p = ExprPool::new();
    let neg = p.integer(-1_i32);
    for (axis, angle, v) in rotation_samples(15, 0x5164) {
        let q = Quaternion::from_axis_angle(&vnum(axis, &p), lit(angle, &p), &p).unwrap();
        let minus_q = q.scale(neg, &p);
        close_v(
            vvals(&minus_q.rotate(&vnum(v, &p), &p).unwrap(), &p),
            vvals(&q.rotate(&vnum(v, &p), &p).unwrap(), &p),
            "(-q) v (-q)⁻¹ vs q v q⁻¹",
        );
    }
}

// ---------------------------------------------------------------------------
// Axis–angle
// ---------------------------------------------------------------------------

#[test]
fn axis_angle_round_trips() {
    let p = ExprPool::new();
    let mut rng = Rng::new(0xA713_A713);
    for _ in 0..30 {
        let axis = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let len = axis.iter().map(|a| a * a).sum::<f64>().sqrt();
        if len < 0.2 {
            continue;
        }
        let unit = [axis[0] / len, axis[1] / len, axis[2] / len];
        // θ ∈ (0, π) is where (axis, angle) is unique.
        let angle = rng.range(0.05, std::f64::consts::PI - 0.05);
        let q = Quaternion::from_axis_angle(&vnum(axis, &p), lit(angle, &p), &p).unwrap();
        let (got_axis, got_angle) = q.to_axis_angle(&p).unwrap();
        close(num(got_angle, &p), angle, "recovered angle");
        close_v(vvals(&got_axis, &p), unit, "recovered axis");
    }
}

#[test]
fn the_identity_rotation_has_no_axis() {
    // The refusal. `q = 1` rotates nothing, so *every* unit vector is an axis
    // for it and there is no axis to return. Returning the conventional
    // (0, 0, 1) would be a stated answer to a question with no answer.
    let p = ExprPool::new();
    let err = Quaternion::identity(&p)
        .to_axis_angle(&p)
        .expect_err("the identity quaternion has no axis");
    assert_eq!(err.code(), "E-QUAT-002");
    assert!(matches!(err, QuaternionError::UndefinedAxis { .. }));

    // …and so does −1, the other representative of the identity rotation
    // (a rotation by 2π).
    let minus_one = Quaternion::scalar(p.integer(-1_i32), &p);
    assert_eq!(
        minus_one.to_axis_angle(&p).unwrap_err().code(),
        "E-QUAT-002"
    );

    // A real scalar multiple of 1 is the same story.
    let three = Quaternion::scalar(p.integer(3_i32), &p);
    assert_eq!(three.to_axis_angle(&p).unwrap_err().code(), "E-QUAT-002");
}

#[test]
fn a_zero_axis_picks_out_no_rotation() {
    let p = ExprPool::new();
    let zero = [p.integer(0_i32); 3];
    let err = Quaternion::from_axis_angle(&zero, p.rational(1, 2), &p)
        .expect_err("the zero vector is not an axis");
    assert_eq!(err.code(), "E-QUAT-001");
}

// ---------------------------------------------------------------------------
// Matrix → quaternion
// ---------------------------------------------------------------------------

#[test]
fn from_rotation_matrix_round_trips_through_every_shepperd_branch() {
    // The four branches are selected by which of `tr`, `R₀₀`, `R₁₁`, `R₂₂` is
    // largest; a rotation by an angle near π about each axis in turn exercises
    // all four. Random samples on top of that.
    let p = ExprPool::new();
    let mut cases: Vec<([f64; 3], f64)> = vec![
        ([0.0, 0.0, 1.0], 0.2),                         // trace branch
        ([1.0, 0.0, 0.0], std::f64::consts::PI - 0.02), // R₀₀ branch
        ([0.0, 1.0, 0.0], std::f64::consts::PI - 0.02), // R₁₁ branch
        ([0.0, 0.0, 1.0], std::f64::consts::PI - 0.02), // R₂₂ branch
    ];
    cases.extend(
        rotation_samples(25, 0x5E79_5E79)
            .into_iter()
            .map(|(a, t, _)| (a, t)),
    );

    for (axis, angle) in cases {
        let q = Quaternion::from_axis_angle(&vnum(axis, &p), lit(angle, &p), &p).unwrap();
        let r = q.to_rotation_matrix(&p).unwrap();
        let recovered = Quaternion::from_rotation_matrix(&r, &p).unwrap();

        // The quaternion is recovered up to sign …
        let (a, b) = (qvals(&q, &p), qvals(&recovered, &p));
        let same = (0..4).all(|i| (a[i] - b[i]).abs() < 1e-8);
        let negated = (0..4).all(|i| (a[i] + b[i]).abs() < 1e-8);
        assert!(
            same || negated,
            "recovered quaternion {b:?} is neither q {a:?} nor −q"
        );
        // … and the matrix is reproduced exactly.
        let r2 = mvals(&recovered.to_rotation_matrix(&p).unwrap(), &p);
        let r1 = mvals(&r, &p);
        for i in 0..3 {
            close_v(r2[i], r1[i], &format!("R(q(R)) row {i}"));
        }
    }
}

#[test]
fn from_rotation_matrix_refuses_what_is_not_a_rotation() {
    let p = ExprPool::new();
    let f = |v: f64| lit(v, &p);

    // Not 3×3.
    let small = Matrix::new(vec![vec![f(1.0), f(0.0)], vec![f(0.0), f(1.0)]]).unwrap();
    assert_eq!(
        Quaternion::from_rotation_matrix(&small, &p)
            .unwrap_err()
            .code(),
        "E-QUAT-003"
    );

    // Symbolic entries: the branch selection is a comparison and there is none
    // to make. Note the matrix *is* a rotation for every real θ — refusing is
    // a capability limit stated honestly, not a claim that it is not one.
    let th = p.symbol("theta", Domain::Real);
    let (c, s) = (p.func("cos", vec![th]), p.func("sin", vec![th]));
    let neg_s = p.mul(vec![p.integer(-1_i32), s]);
    let (z, o) = (p.integer(0_i32), p.integer(1_i32));
    let sym = Matrix::new(vec![vec![c, neg_s, z], vec![s, c, z], vec![z, z, o]]).unwrap();
    let err = Quaternion::from_rotation_matrix(&sym, &p).unwrap_err();
    assert_eq!(err.code(), "E-QUAT-003");
    assert!(
        err.to_string().contains("not a finite number"),
        "the message must say *why* it refused: {err}"
    );

    // A reflection: orthogonal, det = −1. No quaternion represents it.
    let reflection = Matrix::new(vec![
        vec![f(1.0), f(0.0), f(0.0)],
        vec![f(0.0), f(1.0), f(0.0)],
        vec![f(0.0), f(0.0), f(-1.0)],
    ])
    .unwrap();
    let err = Quaternion::from_rotation_matrix(&reflection, &p).unwrap_err();
    assert_eq!(err.code(), "E-QUAT-003");
    assert!(
        err.to_string().contains("det"),
        "must name the determinant: {err}"
    );

    // A scaled rotation: `2·I` is not orthogonal. Shepperd would happily return
    // a unit quaternion for it.
    let scaled = Matrix::new(vec![
        vec![f(2.0), f(0.0), f(0.0)],
        vec![f(0.0), f(2.0), f(0.0)],
        vec![f(0.0), f(0.0), f(2.0)],
    ])
    .unwrap();
    assert_eq!(
        Quaternion::from_rotation_matrix(&scaled, &p)
            .unwrap_err()
            .code(),
        "E-QUAT-003"
    );

    // A shear that is close to a rotation but is not one.
    let shear = Matrix::new(vec![
        vec![f(1.0), f(1e-3), f(0.0)],
        vec![f(0.0), f(1.0), f(0.0)],
        vec![f(0.0), f(0.0), f(1.0)],
    ])
    .unwrap();
    assert_eq!(
        Quaternion::from_rotation_matrix(&shear, &p)
            .unwrap_err()
            .code(),
        "E-QUAT-003"
    );
}

#[test]
fn the_identity_matrix_maps_to_the_identity_quaternion() {
    let p = ExprPool::new();
    let identity = Matrix::identity(3, &p);
    let q = Quaternion::from_rotation_matrix(&identity, &p).unwrap();
    let v = qvals(&q, &p);
    close(v[0].abs(), 1.0, "|w| of the identity rotation");
    for c in &v[1..] {
        close(*c, 0.0, "vector part of the identity rotation");
    }
}
