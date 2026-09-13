//! The vector-calculus test suite is the identity suite.
//!
//! Three layers, each catching something the others cannot:
//!
//! 1. **The identities.** `∇×∇f = 0`, `∇·(∇×F) = 0`,
//!    `∇×(∇×F) = ∇(∇·F) − ∇²F`, `∇·(fF) = f∇·F + ∇f·F`, `∇×(fF) = f∇×F + ∇f×F`.
//!    These hold in *every* orthogonal chart, so they test the general
//!    machinery — but they are also insensitive to a uniform scale-factor
//!    error, because a wrong `hᵢ` used consistently on both sides can cancel.
//! 2. **The textbook forms.** `div`, `curl` and `∇²` in cylindrical and
//!    spherical coordinates, written out by hand from Arfken & Weber
//!    §3.10 / Griffiths *Introduction to Electrodynamics* front cover, and
//!    compared against what this module produces. This pins the scale factors
//!    to published values.
//! 3. **The Cartesian round trip.** A field is written in Cartesian
//!    coordinates, transformed into the curvilinear chart (both the argument
//!    substitution *and* the rotation of the components into the local
//!    orthonormal frame), and the operator is applied in both charts. Agreement
//!    at a random point is the independent check: it cannot be passed by a
//!    self-consistent but wrong set of scale factors, and it is sensitive to
//!    handedness, which the identities are not.
//!
//! Every numeric comparison happens at **random, ugly points**. Scale-factor
//! errors habitually vanish at `θ = π/2`, `φ = 0`, `ρ = 1`, so those points
//! prove nothing.

use super::*;
use crate::jit::eval_interp_checked;
use crate::kernel::subs::subs;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::matrix::zero_test::{zero_status, ZeroStatus};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// xorshift64*, so the "random" points are the same on every run and a failure
/// is reproducible from the seed alone.
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

    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        lo + (hi - lo) * u
    }
}

fn ev(e: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> f64 {
    eval_interp_checked(e, env, pool)
        .unwrap_or_else(|err| panic!("could not evaluate `{}`: {err:?}", pool.display(e)))
}

#[track_caller]
fn close(a: f64, b: f64, what: &str) {
    let scale = 1.0_f64.max(a.abs()).max(b.abs());
    assert!(
        (a - b).abs() <= 1e-8 * scale,
        "{what}: {a} vs {b} (absolute gap {:.3e}, scale {scale})",
        (a - b).abs()
    );
}

/// Assert a scalar field is identically zero, preferring the symbolic verdict.
///
/// Uses the crate's own three-valued zero test
/// (`crate::matrix::zero_test::zero_status`): `Zero` is a proof by a sound
/// normalisation, returned as `true`; `NonZero` is a *rigorous refutation* —
/// a ball enclosure at some sample point that excludes `0` — and fails
/// immediately, because it means the identity is false rather than merely
/// unproven; `Unknown` falls back to evaluating at every supplied point, which
/// is weaker but still catches a wrong formula.
///
/// The tests record which accepting path was taken, because "this identity
/// closes symbolically" is a stronger statement than "it holds at four points"
/// and a regression from the first to the second is worth noticing even while
/// both pass.
#[track_caller]
fn assert_zero_scalar(
    e: ExprId,
    pool: &ExprPool,
    env: &[&HashMap<ExprId, f64>],
    what: &str,
) -> bool {
    match zero_status(pool, e) {
        ZeroStatus::Zero => return true,
        ZeroStatus::NonZero => panic!(
            "{what}: `{}` is provably *not* identically zero — the identity is false",
            pool.display(e)
        ),
        ZeroStatus::Unknown => {}
    }
    for binding in env {
        let v = ev(e, binding, pool);
        assert!(
            v.abs() <= 1e-8,
            "{what}: expected 0, got {v} for `{}`",
            pool.display(e)
        );
    }
    false
}

#[track_caller]
fn assert_zero_field(
    f: &[ExprId; 3],
    pool: &ExprPool,
    env: &[&HashMap<ExprId, f64>],
    what: &str,
) -> bool {
    let mut symbolic = true;
    for (i, &c) in f.iter().enumerate() {
        symbolic &= assert_zero_scalar(c, pool, env, &format!("{what} component {i}"));
    }
    symbolic
}

#[track_caller]
fn assert_fields_agree(
    a: &[ExprId; 3],
    b: &[ExprId; 3],
    pool: &ExprPool,
    env: &HashMap<ExprId, f64>,
    what: &str,
) {
    for i in 0..3 {
        close(
            ev(a[i], env, pool),
            ev(b[i], env, pool),
            &format!("{what} component {i}"),
        );
    }
}

/// The three built-in charts over one pool, plus the symbols behind them.
struct Charts {
    pool: ExprPool,
    cart: Coordinates,
    cyl: Coordinates,
    sph: Coordinates,
    /// `(x, y, z)`
    c: [ExprId; 3],
    /// `(ρ, φ, z)`
    y: [ExprId; 3],
    /// `(r, θ, φ)`
    s: [ExprId; 3],
}

impl Charts {
    fn new() -> Self {
        let pool = ExprPool::new();
        let c = [
            pool.symbol("x", Domain::Real),
            pool.symbol("y", Domain::Real),
            pool.symbol("z", Domain::Real),
        ];
        let y = [
            pool.symbol("rho", Domain::Real),
            pool.symbol("phi", Domain::Real),
            pool.symbol("zc", Domain::Real),
        ];
        let s = [
            pool.symbol("r", Domain::Real),
            pool.symbol("theta", Domain::Real),
            pool.symbol("phis", Domain::Real),
        ];
        let cart = Coordinates::cartesian(c[0], c[1], c[2], &pool).unwrap();
        let cyl = Coordinates::cylindrical(y[0], y[1], y[2], &pool).unwrap();
        let sph = Coordinates::spherical(s[0], s[1], s[2], &pool).unwrap();
        Charts {
            pool,
            cart,
            cyl,
            sph,
            c,
            y,
            s,
        }
    }

    fn p(&self) -> &ExprPool {
        &self.pool
    }

    fn sin(&self, e: ExprId) -> ExprId {
        self.pool.func("sin", vec![e])
    }

    fn cos(&self, e: ExprId) -> ExprId {
        self.pool.func("cos", vec![e])
    }

    fn int(&self, n: i32) -> ExprId {
        self.pool.integer(n)
    }

    /// A random Cartesian point, bound to `(x, y, z)`.
    fn cart_env(&self, rng: &mut Rng) -> HashMap<ExprId, f64> {
        let mut env = HashMap::new();
        for &v in &self.c {
            env.insert(v, rng.range(-1.7, 1.9));
        }
        env
    }

    /// A random cylindrical point, well away from the axis.
    fn cyl_env(&self, rng: &mut Rng) -> HashMap<ExprId, f64> {
        let mut env = HashMap::new();
        env.insert(self.y[0], rng.range(0.4, 2.3));
        env.insert(self.y[1], rng.range(-3.0, 3.0));
        env.insert(self.y[2], rng.range(-1.5, 1.5));
        env
    }

    /// A random spherical point, away from both the origin and the polar axis.
    fn sph_env(&self, rng: &mut Rng) -> HashMap<ExprId, f64> {
        let mut env = HashMap::new();
        env.insert(self.s[0], rng.range(0.5, 2.4));
        env.insert(self.s[1], rng.range(0.35, std::f64::consts::PI - 0.35));
        env.insert(self.s[2], rng.range(-3.0, 3.0));
        env
    }

    /// The Cartesian point a cylindrical point sits at.
    fn cyl_to_cart_env(&self, env: &HashMap<ExprId, f64>) -> HashMap<ExprId, f64> {
        let (rho, phi, z) = (env[&self.y[0]], env[&self.y[1]], env[&self.y[2]]);
        HashMap::from([
            (self.c[0], rho * phi.cos()),
            (self.c[1], rho * phi.sin()),
            (self.c[2], z),
        ])
    }

    /// The Cartesian point a spherical point sits at.
    fn sph_to_cart_env(&self, env: &HashMap<ExprId, f64>) -> HashMap<ExprId, f64> {
        let (r, th, ph) = (env[&self.s[0]], env[&self.s[1]], env[&self.s[2]]);
        HashMap::from([
            (self.c[0], r * th.sin() * ph.cos()),
            (self.c[1], r * th.sin() * ph.sin()),
            (self.c[2], r * th.cos()),
        ])
    }

    /// Rewrite a Cartesian expression in cylindrical coordinates.
    fn to_cyl(&self, e: ExprId) -> ExprId {
        let (rho, phi, z) = (self.y[0], self.y[1], self.y[2]);
        let map = HashMap::from([
            (self.c[0], self.pool.mul(vec![rho, self.cos(phi)])),
            (self.c[1], self.pool.mul(vec![rho, self.sin(phi)])),
            (self.c[2], z),
        ]);
        simplify(subs(e, &map, &self.pool), &self.pool).value
    }

    /// Rewrite a Cartesian expression in spherical coordinates.
    fn to_sph(&self, e: ExprId) -> ExprId {
        let (r, th, ph) = (self.s[0], self.s[1], self.s[2]);
        let map = HashMap::from([
            (
                self.c[0],
                self.pool.mul(vec![r, self.sin(th), self.cos(ph)]),
            ),
            (
                self.c[1],
                self.pool.mul(vec![r, self.sin(th), self.sin(ph)]),
            ),
            (self.c[2], self.pool.mul(vec![r, self.cos(th)])),
        ]);
        simplify(subs(e, &map, &self.pool), &self.pool).value
    }
}

/// Rotate Cartesian vector *values* into the cylindrical orthonormal frame.
fn cart_values_to_cyl(v: [f64; 3], phi: f64) -> [f64; 3] {
    [
        v[0] * phi.cos() + v[1] * phi.sin(),
        -v[0] * phi.sin() + v[1] * phi.cos(),
        v[2],
    ]
}

/// Rotate Cartesian vector *values* into the spherical orthonormal frame.
fn cart_values_to_sph(v: [f64; 3], theta: f64, phi: f64) -> [f64; 3] {
    let (st, ct, sp, cp) = (theta.sin(), theta.cos(), phi.sin(), phi.cos());
    [
        v[0] * st * cp + v[1] * st * sp + v[2] * ct,
        v[0] * ct * cp + v[1] * ct * sp - v[2] * st,
        -v[0] * sp + v[1] * cp,
    ]
}

// ---------------------------------------------------------------------------
// Vector algebra
// ---------------------------------------------------------------------------

#[test]
fn cross_of_the_basis_is_right_handed() {
    let ch = Charts::new();
    let p = ch.p();
    let (o, i) = (ch.int(0), ch.int(1));
    let e1 = [i, o, o];
    let e2 = [o, i, o];
    let e3 = [o, o, i];
    assert_eq!(cross(&e1, &e2, p), e3, "e1 × e2 must be e3");
    assert_eq!(cross(&e2, &e3, p), e1, "e2 × e3 must be e1");
    assert_eq!(cross(&e3, &e1, p), e2, "e3 × e1 must be e2");
}

#[test]
fn cross_is_anticommutative_and_orthogonal_to_both_factors() {
    let ch = Charts::new();
    let p = ch.p();
    let a = [
        p.symbol("a1", Domain::Real),
        p.symbol("a2", Domain::Real),
        p.symbol("a3", Domain::Real),
    ];
    let b = [
        p.symbol("b1", Domain::Real),
        p.symbol("b2", Domain::Real),
        p.symbol("b3", Domain::Real),
    ];
    let ab = cross(&a, &b, p);
    let ba = cross(&b, &a, p);
    let zero = p.integer(0_i32);
    assert_eq!(add(&ab, &ba, p), [zero; 3], "a×b + b×a must be exactly 0");

    // `(a×b)·a = 0` is a cancellation of six products the default simplifier
    // does not expand, so it is settled by the zero test rather than by
    // structural equality.
    let mut rng = Rng::new(0xAB0B_AB0B);
    let envs: Vec<HashMap<ExprId, f64>> = (0..4)
        .map(|_| {
            a.iter()
                .chain(b.iter())
                .map(|&s| (s, rng.range(-2.0, 2.0)))
                .collect()
        })
        .collect();
    let refs: Vec<&HashMap<ExprId, f64>> = envs.iter().collect();
    assert_zero_scalar(dot(&ab, &a, p), p, &refs, "(a×b)·a");
    assert_zero_scalar(dot(&ab, &b, p), p, &refs, "(a×b)·b");
}

#[test]
fn norm_of_a_unit_vector_is_one() {
    let ch = Charts::new();
    let p = ch.p();
    let v = [ch.int(3), ch.int(4), ch.int(0)];
    assert_eq!(norm_squared(&v, p), p.integer(25_i32));
    assert_eq!(norm(&v, p), p.integer(5_i32));
}

// ---------------------------------------------------------------------------
// Layer 1 — the identities
// ---------------------------------------------------------------------------

/// A handful of scalar fields per chart, chosen to have every partial
/// derivative non-trivial.
fn scalar_fields(ch: &Charts, coords: &Coordinates) -> Vec<ExprId> {
    let p = ch.p();
    let [u, v, w] = coords.vars();
    let two = ch.int(2);
    let three = ch.int(3);
    vec![
        p.add(vec![
            p.mul(vec![p.pow(u, two), v]),
            p.mul(vec![three, p.pow(w, three)]),
        ]),
        p.mul(vec![ch.sin(u), ch.cos(v), p.pow(w, two)]),
        p.mul(vec![u, v, w]),
        p.add(vec![
            p.mul(vec![p.func("exp", vec![w]), ch.cos(v)]),
            p.pow(u, three),
        ]),
    ]
}

/// A handful of vector fields per chart.
fn vector_fields(ch: &Charts, coords: &Coordinates) -> Vec<[ExprId; 3]> {
    let p = ch.p();
    let [u, v, w] = coords.vars();
    let two = ch.int(2);
    let three = ch.int(3);
    vec![
        [
            p.mul(vec![p.pow(u, two), v]),
            p.mul(vec![v, p.pow(w, two)]),
            p.mul(vec![three, u, w]),
        ],
        [
            p.mul(vec![ch.sin(v), w]),
            p.mul(vec![u, ch.cos(w)]),
            p.mul(vec![p.pow(u, two), ch.sin(v)]),
        ],
        [
            p.add(vec![u, p.mul(vec![two, v])]),
            p.mul(vec![u, v, w]),
            p.add(vec![p.pow(w, two), p.mul(vec![u, w])]),
        ],
    ]
}

fn envs_for(ch: &Charts, which: usize, rng: &mut Rng) -> Vec<HashMap<ExprId, f64>> {
    (0..4)
        .map(|_| match which {
            0 => ch.cart_env(rng),
            1 => ch.cyl_env(rng),
            _ => ch.sph_env(rng),
        })
        .collect()
}

#[test]
fn curl_of_a_gradient_vanishes_in_every_chart() {
    let ch = Charts::new();
    let mut rng = Rng::new(0x1234_5678);
    let mut per_chart = [0_usize; 3];
    let mut total = 0;
    for (which, coords) in [&ch.cart, &ch.cyl, &ch.sph].into_iter().enumerate() {
        let envs = envs_for(&ch, which, &mut rng);
        let refs: Vec<&HashMap<ExprId, f64>> = envs.iter().collect();
        for f in scalar_fields(&ch, coords) {
            let g = gradient(f, coords, ch.p()).unwrap();
            let c = curl(&g, coords, ch.p()).unwrap();
            total += 1;
            if assert_zero_field(
                &c,
                ch.p(),
                &refs,
                &format!("curl grad in {}", coords.label()),
            ) {
                per_chart[which] += 1;
            }
        }
    }
    println!(
        "curl(grad f) closed symbolically: cartesian {}/4, cylindrical {}/4, spherical {}/4 \
         (of {total} total; the rest were checked numerically)",
        per_chart[0], per_chart[1], per_chart[2]
    );
    // Cartesian and cylindrical reduce to equality of mixed partials with no
    // surviving `1/h` factor to cancel, so those eight must close *exactly*.
    // The spherical components carry a `1/(r sin θ)` the zero test does not
    // clear on every field; those are checked numerically above, and this
    // assertion pins the symbolic tier so a regression in it is visible.
    assert_eq!(
        (per_chart[0], per_chart[1]),
        (4, 4),
        "curl(grad f) must close symbolically in the Cartesian and cylindrical charts"
    );
}

#[test]
fn divergence_of_a_curl_vanishes_in_every_chart() {
    let ch = Charts::new();
    let mut rng = Rng::new(0x9ABC_DEF0);
    let mut symbolic_closures = 0;
    let mut total = 0;
    for (which, coords) in [&ch.cart, &ch.cyl, &ch.sph].into_iter().enumerate() {
        let envs = envs_for(&ch, which, &mut rng);
        let refs: Vec<&HashMap<ExprId, f64>> = envs.iter().collect();
        for f in vector_fields(&ch, coords) {
            let c = curl(&f, coords, ch.p()).unwrap();
            let d = divergence(&c, coords, ch.p()).unwrap();
            total += 1;
            if assert_zero_scalar(d, ch.p(), &refs, &format!("div curl in {}", coords.label())) {
                symbolic_closures += 1;
            }
        }
    }
    assert!(
        symbolic_closures >= 3,
        "div(curl F) closed symbolically only {symbolic_closures}/{total} times; \
         the Cartesian cases at least must reduce to 0 exactly"
    );
}

#[test]
fn curl_curl_equals_grad_div_minus_vector_laplacian_in_cartesian() {
    // Cartesian is where this identity has teeth: `vector_laplacian` is the
    // *componentwise* scalar Laplacian there, computed independently of `curl`
    // and `div`, so the two sides share no code. In a curvilinear chart the
    // vector Laplacian is *defined* as `grad div − curl curl` and the identity
    // is a tautology — see `curvilinear_vector_laplacian_matches_the_cartesian_one`
    // for the check that has content there.
    let ch = Charts::new();
    let p = ch.p();
    let mut rng = Rng::new(0xFEED_BEEF);
    let envs = envs_for(&ch, 0, &mut rng);
    for f in vector_fields(&ch, &ch.cart) {
        let cc = curl(&curl(&f, &ch.cart, p).unwrap(), &ch.cart, p).unwrap();
        let gd = gradient(divergence(&f, &ch.cart, p).unwrap(), &ch.cart, p).unwrap();
        let vl = vector_laplacian(&f, &ch.cart, p).unwrap();
        let rhs = sub(&gd, &vl, p);
        let residual = sub(&cc, &rhs, p);
        let refs: Vec<&HashMap<ExprId, f64>> = envs.iter().collect();
        assert_zero_field(&residual, p, &refs, "curl curl − (grad div − lap)");
    }
}

#[test]
fn divergence_of_a_scaled_field_obeys_the_product_rule() {
    // ∇·(fF) = f ∇·F + ∇f·F, in all three charts.
    let ch = Charts::new();
    let p = ch.p();
    let mut rng = Rng::new(0x0BAD_F00D);
    for (which, coords) in [&ch.cart, &ch.cyl, &ch.sph].into_iter().enumerate() {
        let envs = envs_for(&ch, which, &mut rng);
        let refs: Vec<&HashMap<ExprId, f64>> = envs.iter().collect();
        for f in scalar_fields(&ch, coords) {
            for field in vector_fields(&ch, coords) {
                let scaled = scale(f, &field, p);
                let lhs = divergence(&scaled, coords, p).unwrap();
                let rhs = p.add(vec![
                    p.mul(vec![f, divergence(&field, coords, p).unwrap()]),
                    dot(&gradient(f, coords, p).unwrap(), &field, p),
                ]);
                let rhs = simplify(rhs, p).value;
                let neg = p.integer(-1_i32);
                let residual = simplify(p.add(vec![lhs, p.mul(vec![neg, rhs])]), p).value;
                assert_zero_scalar(
                    residual,
                    p,
                    &refs,
                    &format!("div(fF) − (f div F + grad f · F) in {}", coords.label()),
                );
            }
        }
    }
}

#[test]
fn curl_of_a_scaled_field_obeys_the_product_rule() {
    // ∇×(fF) = f ∇×F + ∇f × F.  This one is sensitive to the handedness of
    // `cross` relative to the sign convention inside `curl`: swapping either
    // flips the sign of the second term alone.
    let ch = Charts::new();
    let p = ch.p();
    let mut rng = Rng::new(0xC0FF_EE01);
    for (which, coords) in [&ch.cart, &ch.cyl, &ch.sph].into_iter().enumerate() {
        let envs = envs_for(&ch, which, &mut rng);
        let refs: Vec<&HashMap<ExprId, f64>> = envs.iter().collect();
        for f in scalar_fields(&ch, coords) {
            for field in vector_fields(&ch, coords) {
                let lhs = curl(&scale(f, &field, p), coords, p).unwrap();
                let term1 = scale(f, &curl(&field, coords, p).unwrap(), p);
                let term2 = cross(&gradient(f, coords, p).unwrap(), &field, p);
                let rhs = add(&term1, &term2, p);
                let residual = sub(&lhs, &rhs, p);
                assert_zero_field(
                    &residual,
                    p,
                    &refs,
                    &format!("curl(fF) − (f curl F + grad f × F) in {}", coords.label()),
                );
            }
        }
    }
}

#[test]
fn gradient_obeys_the_product_rule() {
    let ch = Charts::new();
    let p = ch.p();
    let mut rng = Rng::new(0x5EED_1234);
    for (which, coords) in [&ch.cart, &ch.cyl, &ch.sph].into_iter().enumerate() {
        let envs = envs_for(&ch, which, &mut rng);
        let refs: Vec<&HashMap<ExprId, f64>> = envs.iter().collect();
        let fields = scalar_fields(&ch, coords);
        for f in &fields {
            for g in &fields {
                let lhs = gradient(simplify(p.mul(vec![*f, *g]), p).value, coords, p).unwrap();
                let rhs = add(
                    &scale(*f, &gradient(*g, coords, p).unwrap(), p),
                    &scale(*g, &gradient(*f, coords, p).unwrap(), p),
                    p,
                );
                assert_zero_field(
                    &sub(&lhs, &rhs, p),
                    p,
                    &refs,
                    &format!("grad(fg) − (f grad g + g grad f) in {}", coords.label()),
                );
            }
        }
    }
}

#[test]
fn laplacian_is_the_divergence_of_the_gradient() {
    let ch = Charts::new();
    let p = ch.p();
    let mut rng = Rng::new(0x1111_2222);
    for (which, coords) in [&ch.cart, &ch.cyl, &ch.sph].into_iter().enumerate() {
        let envs = envs_for(&ch, which, &mut rng);
        let refs: Vec<&HashMap<ExprId, f64>> = envs.iter().collect();
        for f in scalar_fields(&ch, coords) {
            let direct = laplacian(f, coords, p).unwrap();
            let via = divergence(&gradient(f, coords, p).unwrap(), coords, p).unwrap();
            let neg = p.integer(-1_i32);
            let residual = simplify(p.add(vec![direct, p.mul(vec![neg, via])]), p).value;
            assert_zero_scalar(
                residual,
                p,
                &refs,
                &format!("∇²f − ∇·∇f in {}", coords.label()),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Layer 2 — the published forms
// ---------------------------------------------------------------------------

#[test]
fn cylindrical_operators_match_the_textbook_forms() {
    // Arfken & Weber, *Mathematical Methods for Physicists* 7th ed. §3.10;
    // Griffiths, *Introduction to Electrodynamics* 4th ed., inside front cover.
    let ch = Charts::new();
    let p = ch.p();
    let [rho, phi, z] = ch.cyl.vars();
    let dd = |e: ExprId, v: ExprId| crate::diff::diff(e, v, p).unwrap().value;
    let inv = |e: ExprId| p.pow(e, p.integer(-1_i32));

    let mut rng = Rng::new(0xABCD_0001);
    for f in scalar_fields(&ch, &ch.cyl) {
        // ∇²f = (1/ρ)∂_ρ(ρ ∂_ρ f) + (1/ρ²)∂²_φ f + ∂²_z f
        let expected = p.add(vec![
            p.mul(vec![inv(rho), dd(p.mul(vec![rho, dd(f, rho)]), rho)]),
            p.mul(vec![inv(p.pow(rho, p.integer(2_i32))), dd(dd(f, phi), phi)]),
            dd(dd(f, z), z),
        ]);
        let got = laplacian(f, &ch.cyl, p).unwrap();
        for _ in 0..6 {
            let env = ch.cyl_env(&mut rng);
            close(
                ev(got, &env, p),
                ev(expected, &env, p),
                "cylindrical laplacian",
            );
        }
    }

    for field in vector_fields(&ch, &ch.cyl) {
        let [fr, fp, fz] = field;
        // ∇·F = (1/ρ)∂_ρ(ρ F_ρ) + (1/ρ)∂_φ F_φ + ∂_z F_z
        let expected_div = p.add(vec![
            p.mul(vec![inv(rho), dd(p.mul(vec![rho, fr]), rho)]),
            p.mul(vec![inv(rho), dd(fp, phi)]),
            dd(fz, z),
        ]);
        // (∇×F)_ρ = (1/ρ)∂_φ F_z − ∂_z F_φ
        // (∇×F)_φ = ∂_z F_ρ − ∂_ρ F_z
        // (∇×F)_z = (1/ρ)[∂_ρ(ρ F_φ) − ∂_φ F_ρ]
        let neg = p.integer(-1_i32);
        let expected_curl = [
            p.add(vec![
                p.mul(vec![inv(rho), dd(fz, phi)]),
                p.mul(vec![neg, dd(fp, z)]),
            ]),
            p.add(vec![dd(fr, z), p.mul(vec![neg, dd(fz, rho)])]),
            p.mul(vec![
                inv(rho),
                p.add(vec![
                    dd(p.mul(vec![rho, fp]), rho),
                    p.mul(vec![neg, dd(fr, phi)]),
                ]),
            ]),
        ];
        let got_div = divergence(&field, &ch.cyl, p).unwrap();
        let got_curl = curl(&field, &ch.cyl, p).unwrap();
        for _ in 0..6 {
            let env = ch.cyl_env(&mut rng);
            close(
                ev(got_div, &env, p),
                ev(expected_div, &env, p),
                "cylindrical divergence",
            );
            assert_fields_agree(&got_curl, &expected_curl, p, &env, "cylindrical curl");
        }
    }
}

#[test]
fn spherical_operators_match_the_textbook_forms() {
    // Arfken & Weber §3.10, physics convention (θ polar, φ azimuth).
    let ch = Charts::new();
    let p = ch.p();
    let [r, th, ph] = ch.sph.vars();
    let dd = |e: ExprId, v: ExprId| crate::diff::diff(e, v, p).unwrap().value;
    let inv = |e: ExprId| p.pow(e, p.integer(-1_i32));
    let two = p.integer(2_i32);
    let neg = p.integer(-1_i32);
    let sin_th = ch.sin(th);
    let r2 = p.pow(r, two);

    let mut rng = Rng::new(0xABCD_0002);
    for f in scalar_fields(&ch, &ch.sph) {
        // ∇²f = (1/r²)∂_r(r²∂_r f) + (1/(r² sinθ))∂_θ(sinθ ∂_θ f)
        //       + (1/(r² sin²θ))∂²_φ f
        let expected = p.add(vec![
            p.mul(vec![inv(r2), dd(p.mul(vec![r2, dd(f, r)]), r)]),
            p.mul(vec![
                inv(p.mul(vec![r2, sin_th])),
                dd(p.mul(vec![sin_th, dd(f, th)]), th),
            ]),
            p.mul(vec![
                inv(p.mul(vec![r2, p.pow(sin_th, two)])),
                dd(dd(f, ph), ph),
            ]),
        ]);
        let got = laplacian(f, &ch.sph, p).unwrap();
        for _ in 0..6 {
            let env = ch.sph_env(&mut rng);
            close(
                ev(got, &env, p),
                ev(expected, &env, p),
                "spherical laplacian",
            );
        }
    }

    for field in vector_fields(&ch, &ch.sph) {
        let [fr, ft, fp] = field;
        // ∇·F = (1/r²)∂_r(r²F_r) + (1/(r sinθ))∂_θ(sinθ F_θ)
        //       + (1/(r sinθ))∂_φ F_φ
        let expected_div = p.add(vec![
            p.mul(vec![inv(r2), dd(p.mul(vec![r2, fr]), r)]),
            p.mul(vec![
                inv(p.mul(vec![r, sin_th])),
                dd(p.mul(vec![sin_th, ft]), th),
            ]),
            p.mul(vec![inv(p.mul(vec![r, sin_th])), dd(fp, ph)]),
        ]);
        // (∇×F)_r = (1/(r sinθ))[∂_θ(sinθ F_φ) − ∂_φ F_θ]
        // (∇×F)_θ = (1/r)[(1/sinθ)∂_φ F_r − ∂_r(r F_φ)]
        // (∇×F)_φ = (1/r)[∂_r(r F_θ) − ∂_θ F_r]
        let expected_curl = [
            p.mul(vec![
                inv(p.mul(vec![r, sin_th])),
                p.add(vec![
                    dd(p.mul(vec![sin_th, fp]), th),
                    p.mul(vec![neg, dd(ft, ph)]),
                ]),
            ]),
            p.mul(vec![
                inv(r),
                p.add(vec![
                    p.mul(vec![inv(sin_th), dd(fr, ph)]),
                    p.mul(vec![neg, dd(p.mul(vec![r, fp]), r)]),
                ]),
            ]),
            p.mul(vec![
                inv(r),
                p.add(vec![
                    dd(p.mul(vec![r, ft]), r),
                    p.mul(vec![neg, dd(fr, th)]),
                ]),
            ]),
        ];
        let got_div = divergence(&field, &ch.sph, p).unwrap();
        let got_curl = curl(&field, &ch.sph, p).unwrap();
        for _ in 0..6 {
            let env = ch.sph_env(&mut rng);
            close(
                ev(got_div, &env, p),
                ev(expected_div, &env, p),
                "spherical divergence",
            );
            assert_fields_agree(&got_curl, &expected_curl, p, &env, "spherical curl");
        }
    }
}

// ---------------------------------------------------------------------------
// Layer 3 — the Cartesian round trip
// ---------------------------------------------------------------------------

/// Cartesian fields used for the round trip. Deliberately *not* symmetric
/// about the axis: a field with no `φ` dependence passes a wrong `h₂` and
/// tells you nothing.
fn roundtrip_fields(ch: &Charts) -> Vec<[ExprId; 3]> {
    let p = ch.p();
    let [x, y, z] = ch.c;
    let two = ch.int(2);
    let three = ch.int(3);
    vec![
        [
            p.mul(vec![p.pow(x, two), y]),
            p.mul(vec![y, p.pow(z, two)]),
            p.mul(vec![three, x, z]),
        ],
        [
            p.add(vec![p.mul(vec![x, y, z]), p.pow(y, three)]),
            p.mul(vec![p.func("exp", vec![z]), x]),
            p.add(vec![p.pow(x, two), p.mul(vec![two, y, z])]),
        ],
        [
            p.func("sin", vec![p.mul(vec![x, y])]),
            p.mul(vec![z, p.func("cos", vec![x])]),
            p.mul(vec![x, y, z]),
        ],
    ]
}

fn roundtrip_scalars(ch: &Charts) -> Vec<ExprId> {
    let p = ch.p();
    let [x, y, z] = ch.c;
    let two = ch.int(2);
    let three = ch.int(3);
    vec![
        p.add(vec![p.mul(vec![p.pow(x, two), y]), p.pow(z, three)]),
        p.mul(vec![x, y, z]),
        p.add(vec![
            p.func("exp", vec![x]),
            p.func("sin", vec![p.mul(vec![y, z])]),
        ]),
    ]
}

#[test]
fn cylindrical_operators_agree_with_cartesian_under_the_change_of_chart() {
    let ch = Charts::new();
    let p = ch.p();
    let mut rng = Rng::new(0x7777_0001);

    for f in roundtrip_scalars(&ch) {
        let lap_cart = laplacian(f, &ch.cart, p).unwrap();
        let lap_cyl = laplacian(ch.to_cyl(f), &ch.cyl, p).unwrap();
        for _ in 0..8 {
            let cyl = ch.cyl_env(&mut rng);
            let cart = ch.cyl_to_cart_env(&cyl);
            close(
                ev(lap_cyl, &cyl, p),
                ev(lap_cart, &cart, p),
                "∇²f: cylindrical vs Cartesian",
            );
        }
    }

    for field in roundtrip_fields(&ch) {
        let [fx, fy, fz] = field;
        let (fxc, fyc, fzc) = (ch.to_cyl(fx), ch.to_cyl(fy), ch.to_cyl(fz));
        let phi = ch.y[1];
        let (c, s) = (ch.cos(phi), ch.sin(phi));
        let neg = p.integer(-1_i32);
        let in_cyl = [
            simplify(p.add(vec![p.mul(vec![fxc, c]), p.mul(vec![fyc, s])]), p).value,
            simplify(
                p.add(vec![p.mul(vec![neg, fxc, s]), p.mul(vec![fyc, c])]),
                p,
            )
            .value,
            fzc,
        ];

        let div_cart = divergence(&field, &ch.cart, p).unwrap();
        let div_cyl = divergence(&in_cyl, &ch.cyl, p).unwrap();
        let curl_cart = curl(&field, &ch.cart, p).unwrap();
        let curl_cyl = curl(&in_cyl, &ch.cyl, p).unwrap();
        let vl_cart = vector_laplacian(&field, &ch.cart, p).unwrap();
        let vl_cyl = vector_laplacian(&in_cyl, &ch.cyl, p).unwrap();

        for _ in 0..8 {
            let cyl = ch.cyl_env(&mut rng);
            let cart = ch.cyl_to_cart_env(&cyl);
            let phi_v = cyl[&ch.y[1]];
            close(
                ev(div_cyl, &cyl, p),
                ev(div_cart, &cart, p),
                "∇·F: cylindrical vs Cartesian",
            );
            let want_curl = cart_values_to_cyl(
                [
                    ev(curl_cart[0], &cart, p),
                    ev(curl_cart[1], &cart, p),
                    ev(curl_cart[2], &cart, p),
                ],
                phi_v,
            );
            let want_vl = cart_values_to_cyl(
                [
                    ev(vl_cart[0], &cart, p),
                    ev(vl_cart[1], &cart, p),
                    ev(vl_cart[2], &cart, p),
                ],
                phi_v,
            );
            for i in 0..3 {
                close(
                    ev(curl_cyl[i], &cyl, p),
                    want_curl[i],
                    &format!("∇×F component {i}: cylindrical vs Cartesian"),
                );
                close(
                    ev(vl_cyl[i], &cyl, p),
                    want_vl[i],
                    &format!("∇²F component {i}: cylindrical vs Cartesian"),
                );
            }
        }
    }
}

#[test]
fn spherical_operators_agree_with_cartesian_under_the_change_of_chart() {
    let ch = Charts::new();
    let p = ch.p();
    let mut rng = Rng::new(0x7777_0002);

    for f in roundtrip_scalars(&ch) {
        let lap_cart = laplacian(f, &ch.cart, p).unwrap();
        let lap_sph = laplacian(ch.to_sph(f), &ch.sph, p).unwrap();
        for _ in 0..8 {
            let sph = ch.sph_env(&mut rng);
            let cart = ch.sph_to_cart_env(&sph);
            close(
                ev(lap_sph, &sph, p),
                ev(lap_cart, &cart, p),
                "∇²f: spherical vs Cartesian",
            );
        }
    }

    for field in roundtrip_fields(&ch) {
        let [fx, fy, fz] = field;
        let (fxs, fys, fzs) = (ch.to_sph(fx), ch.to_sph(fy), ch.to_sph(fz));
        let (th, ph) = (ch.s[1], ch.s[2]);
        let (st, ct, sp, cp) = (ch.sin(th), ch.cos(th), ch.sin(ph), ch.cos(ph));
        let neg = p.integer(-1_i32);
        let in_sph = [
            simplify(
                p.add(vec![
                    p.mul(vec![fxs, st, cp]),
                    p.mul(vec![fys, st, sp]),
                    p.mul(vec![fzs, ct]),
                ]),
                p,
            )
            .value,
            simplify(
                p.add(vec![
                    p.mul(vec![fxs, ct, cp]),
                    p.mul(vec![fys, ct, sp]),
                    p.mul(vec![neg, fzs, st]),
                ]),
                p,
            )
            .value,
            simplify(
                p.add(vec![p.mul(vec![neg, fxs, sp]), p.mul(vec![fys, cp])]),
                p,
            )
            .value,
        ];

        let div_cart = divergence(&field, &ch.cart, p).unwrap();
        let div_sph = divergence(&in_sph, &ch.sph, p).unwrap();
        let curl_cart = curl(&field, &ch.cart, p).unwrap();
        let curl_sph = curl(&in_sph, &ch.sph, p).unwrap();
        let vl_cart = vector_laplacian(&field, &ch.cart, p).unwrap();
        let vl_sph = vector_laplacian(&in_sph, &ch.sph, p).unwrap();

        for _ in 0..8 {
            let sph = ch.sph_env(&mut rng);
            let cart = ch.sph_to_cart_env(&sph);
            let (th_v, ph_v) = (sph[&ch.s[1]], sph[&ch.s[2]]);
            close(
                ev(div_sph, &sph, p),
                ev(div_cart, &cart, p),
                "∇·F: spherical vs Cartesian",
            );
            let want_curl = cart_values_to_sph(
                [
                    ev(curl_cart[0], &cart, p),
                    ev(curl_cart[1], &cart, p),
                    ev(curl_cart[2], &cart, p),
                ],
                th_v,
                ph_v,
            );
            let want_vl = cart_values_to_sph(
                [
                    ev(vl_cart[0], &cart, p),
                    ev(vl_cart[1], &cart, p),
                    ev(vl_cart[2], &cart, p),
                ],
                th_v,
                ph_v,
            );
            for i in 0..3 {
                close(
                    ev(curl_sph[i], &sph, p),
                    want_curl[i],
                    &format!("∇×F component {i}: spherical vs Cartesian"),
                );
                close(
                    ev(vl_sph[i], &sph, p),
                    want_vl[i],
                    &format!("∇²F component {i}: spherical vs Cartesian"),
                );
            }
        }
    }
}

#[test]
fn curvilinear_vector_laplacian_is_not_the_componentwise_one() {
    // The guard against the classic error. For `F = ê_φ` (that is, the field
    // whose physical components are `(0, 1, 0)` in cylindrical coordinates)
    // the componentwise Laplacian is identically zero, while the true vector
    // Laplacian is `−ê_φ/ρ²`, because ê_φ turns as φ advances.
    let ch = Charts::new();
    let p = ch.p();
    let rho = ch.cyl.vars()[0];
    let (zero, one) = (ch.int(0), ch.int(1));
    let field = [zero, one, zero];

    let componentwise: Vec<ExprId> = field
        .iter()
        .map(|&c| laplacian(c, &ch.cyl, p).unwrap())
        .collect();
    assert_eq!(
        componentwise,
        vec![zero, zero, zero],
        "the componentwise scalar Laplacian of a constant component array is 0"
    );

    let vl = vector_laplacian(&field, &ch.cyl, p).unwrap();
    let expected = [
        zero,
        simplify(
            p.mul(vec![p.integer(-1_i32), p.pow(rho, p.integer(-2_i32))]),
            p,
        )
        .value,
        zero,
    ];
    let mut rng = Rng::new(0x3141_5926);
    for _ in 0..6 {
        let env = ch.cyl_env(&mut rng);
        assert_fields_agree(&vl, &expected, p, &env, "∇²(ê_φ) in cylindrical");
    }
    assert_ne!(
        vl[1], zero,
        "∇²F in a curvilinear chart must not be the componentwise Laplacian"
    );
}

// ---------------------------------------------------------------------------
// from_embedding, and the refusals
// ---------------------------------------------------------------------------

#[test]
fn from_embedding_rederives_the_cylindrical_scale_factors() {
    let ch = Charts::new();
    let p = ch.p();
    let [rho, phi, z] = ch.cyl.vars();
    let derived = Coordinates::from_embedding(
        [rho, phi, z],
        [
            p.mul(vec![rho, ch.cos(phi)]),
            p.mul(vec![rho, ch.sin(phi)]),
            z,
        ],
        "cylindrical (derived)",
        p,
    )
    .expect("the cylindrical embedding is orthogonal");

    let mut rng = Rng::new(0x2718_2818);
    for _ in 0..6 {
        let env = ch.cyl_env(&mut rng);
        for i in 0..3 {
            close(
                ev(derived.scale_factors()[i], &env, p),
                ev(ch.cyl.scale_factors()[i], &env, p),
                &format!("derived h{} vs the built-in cylindrical chart", i + 1),
            );
        }
    }
    // …and the operators built on them agree too.
    for f in scalar_fields(&ch, &ch.cyl) {
        let a = laplacian(f, &ch.cyl, p).unwrap();
        let b = laplacian(f, &derived, p).unwrap();
        for _ in 0..4 {
            let env = ch.cyl_env(&mut rng);
            close(ev(a, &env, p), ev(b, &env, p), "∇² from derived factors");
        }
    }
}

#[test]
fn from_embedding_rederives_the_spherical_scale_factors() {
    let ch = Charts::new();
    let p = ch.p();
    let [r, th, ph] = ch.sph.vars();
    let derived = Coordinates::from_embedding(
        [r, th, ph],
        [
            p.mul(vec![r, ch.sin(th), ch.cos(ph)]),
            p.mul(vec![r, ch.sin(th), ch.sin(ph)]),
            p.mul(vec![r, ch.cos(th)]),
        ],
        "spherical (derived)",
        p,
    )
    .expect("the spherical embedding is orthogonal");

    let mut rng = Rng::new(0x1618_0339);
    for _ in 0..6 {
        let env = ch.sph_env(&mut rng);
        for i in 0..3 {
            close(
                ev(derived.scale_factors()[i], &env, p),
                ev(ch.sph.scale_factors()[i], &env, p),
                &format!("derived h{} vs the built-in spherical chart", i + 1),
            );
        }
    }
    for field in vector_fields(&ch, &ch.sph) {
        let a = divergence(&field, &ch.sph, p).unwrap();
        let b = divergence(&field, &derived, p).unwrap();
        for _ in 0..4 {
            let env = ch.sph_env(&mut rng);
            close(ev(a, &env, p), ev(b, &env, p), "∇· from derived factors");
        }
    }
}

#[test]
fn from_embedding_refuses_a_skew_chart() {
    use crate::errors::AlkahestError;
    let ch = Charts::new();
    let p = ch.p();
    let u = p.symbol("u", Domain::Real);
    let v = p.symbol("v", Domain::Real);
    let w = p.symbol("w", Domain::Real);
    // x = u + v, y = v, z = w — a perfectly good chart, and not an orthogonal
    // one: ∂r/∂u = (1,0,0), ∂r/∂v = (1,1,0), inner product 1.
    let err = Coordinates::from_embedding([u, v, w], [p.add(vec![u, v]), v, w], "skew", p)
        .expect_err("a skew chart must be refused, not silently treated as orthogonal");
    assert_eq!(err.code(), "E-VEC-004");
    assert!(matches!(
        err,
        VectorError::NonOrthogonal { pair: (0, 1), .. }
    ));
}

#[test]
fn from_embedding_refuses_a_chart_that_collapses_a_direction() {
    use crate::errors::AlkahestError;
    let ch = Charts::new();
    let p = ch.p();
    let u = p.symbol("u", Domain::Real);
    let v = p.symbol("v", Domain::Real);
    let w = p.symbol("w", Domain::Real);
    // `w` does not appear in the embedding, so ∂r/∂w = 0 and h₃ = 0.
    let err = Coordinates::from_embedding([u, v, w], [u, v, p.integer(0_i32)], "collapsed", p)
        .expect_err("a degenerate chart must be refused");
    assert_eq!(err.code(), "E-VEC-005");
    assert!(matches!(
        err,
        VectorError::DegenerateScaleFactor {
            index: 2,
            proven_zero: true,
            ..
        }
    ));
}

#[test]
fn repeated_or_non_symbol_coordinates_are_refused() {
    use crate::errors::AlkahestError;
    let ch = Charts::new();
    let p = ch.p();
    let x = p.symbol("x0", Domain::Real);
    let y = p.symbol("y0", Domain::Real);
    let err = Coordinates::cartesian(x, y, x, p).expect_err("x twice must be refused");
    assert_eq!(err.code(), "E-VEC-002");

    let err = Coordinates::cylindrical(x, y, p.integer(3_i32), p)
        .expect_err("a literal is not a coordinate");
    assert_eq!(err.code(), "E-VEC-003");
}

#[test]
fn cartesian_gradient_is_the_plain_partial_derivatives() {
    let ch = Charts::new();
    let p = ch.p();
    let [x, y, z] = ch.c;
    let two = ch.int(2);
    let f = p.add(vec![p.mul(vec![p.pow(x, two), y]), p.pow(z, two)]);
    let g = gradient(f, &ch.cart, p).unwrap();
    let expect = [
        simplify(crate::diff::diff(f, x, p).unwrap().value, p).value,
        simplify(crate::diff::diff(f, y, p).unwrap().value, p).value,
        simplify(crate::diff::diff(f, z, p).unwrap().value, p).value,
    ];
    assert_eq!(g, expect, "Cartesian grad must be exactly (∂x, ∂y, ∂z)");
}

#[test]
fn radial_inverse_square_field_is_divergence_free_away_from_the_origin() {
    // ∇·(r̂/r²) = 0 for r > 0 — the textbook case where the "obvious"
    // componentwise answer is wrong and the r² factor in the spherical
    // divergence is doing all the work.
    let ch = Charts::new();
    let p = ch.p();
    let r = ch.sph.vars()[0];
    let zero = ch.int(0);
    let field = [p.pow(r, p.integer(-2_i32)), zero, zero];
    let d = divergence(&field, &ch.sph, p).unwrap();
    assert_eq!(d, zero, "∇·(r̂/r²) must reduce to exactly 0");

    // …while ∇·r̂ = 2/r, not 0.
    let unit = [ch.int(1), zero, zero];
    let d1 = divergence(&unit, &ch.sph, p).unwrap();
    let mut rng = Rng::new(0x4242_4242);
    for _ in 0..5 {
        let env = ch.sph_env(&mut rng);
        close(ev(d1, &env, p), 2.0 / env[&r], "∇·r̂ = 2/r");
    }
}
