//! Tests for the FEX engine: derivative correctness against symbolic `diff`,
//! interpreter-vs-instantiate agreement, canonicalisation invariants,
//! overflow guards, fitting, and error codes.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::sync::Arc;

use super::*;
use crate::diff::diff;
use crate::errors::AlkahestError;
use crate::jit::eval_interp;
use crate::kernel::{Domain, ExprId, ExprPool};

/// xorshift64* — deterministic, dependency-free.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unif(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn coords(pool: &ExprPool, d: usize) -> Vec<ExprId> {
    (0..d)
        .map(|i| pool.symbol(format!("x{i}"), Domain::Real))
        .collect()
}

fn ev_at(expr: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> f64 {
    eval_interp(expr, env, pool).expect("evaluable")
}

fn close(a: f64, b: f64, rtol: f64) -> bool {
    (a - b).abs() <= rtol * (1.0 + a.abs().max(b.abs()))
}

fn poisson_problem(d: usize) -> Problem {
    Problem::new(
        d,
        PdeOperator::poisson(),
        ScalarField::Const(-(d as f64)),
        ScalarField::Func(Arc::new(|x: &[f64]| {
            0.5 * x.iter().map(|v| v * v).sum::<f64>()
        })),
    )
    .unwrap()
}

fn grid(rng: &mut Rng, n: usize, d: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n * d).map(|_| rng.unif(lo, hi)).collect()
}

// ---------------------------------------------------------------------------
// Template IR
// ---------------------------------------------------------------------------

#[test]
fn template_slots_follow_reference_inorder() {
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, 4).unwrap();
    assert_eq!(t.shape(), "U(B(L,L))");
    let kinds: Vec<SlotKind> = t.slots().iter().map(|s| s.kind).collect();
    assert_eq!(
        kinds,
        vec![
            SlotKind::Leaf,
            SlotKind::Binary,
            SlotKind::Leaf,
            SlotKind::Unary
        ]
    );
    assert_eq!(t.n_params(), 5 + 5 + 2);
    assert_eq!(t.slot_arities(), vec![9, 3, 9, 9]);
    assert_eq!(t.n_skeletons(), 2187);
    let t3 = Template::new("depth3", &PAPER_UNARY, &PAPER_BINARY, 2).unwrap();
    assert_eq!(t3.n_slots(), 21);
}

#[test]
fn instantiate_matches_paper_formula() {
    // depth2_sub with e = [sq, +, 0, id]: id(L[sq] + L[0]) = α(Σwx² + b + b') + β
    let pool = ExprPool::new();
    let t = Template::new("U(B(L,L))", &PAPER_UNARY, &PAPER_BINARY, 2).unwrap();
    let xs = coords(&pool, 2);
    let e = [3, 0, 0, 2];
    let th = [0.5, 0.5, 0.0, 9.0, 9.0, 0.0, 1.0, 0.0];
    let u = t.instantiate(&e, &th, &xs, &pool).unwrap();
    let mut env = HashMap::new();
    env.insert(xs[0], 0.3);
    env.insert(xs[1], -0.7);
    assert!(close(ev_at(u, &env, &pool), 0.5 * (0.09 + 0.49), 1e-14));
    assert_eq!(t.describe(&e).unwrap(), "id((L[sq] + L[0]))");
}

// ---------------------------------------------------------------------------
// Derivative correctness: fused jets and Jacobians vs symbolic diff
// ---------------------------------------------------------------------------

/// Unary ops that are smooth everywhere on the sample domain.
const SMOOTH: [UnaryOp; 9] = PAPER_UNARY;

fn check_candidate_derivatives(
    t: &Template,
    e: &[usize],
    th: &[f64],
    x: &[f64],
    time: Option<usize>,
) {
    let d = t.dim();
    let pool = ExprPool::new();
    let xs = coords(&pool, d);
    let psyms: Vec<ExprId> = (0..t.n_params())
        .map(|k| pool.symbol(format!("th{k}"), Domain::Real))
        .collect();
    let usym = t.instantiate_with(e, &psyms, &xs, &pool).unwrap();
    let mut env = HashMap::new();
    for (s, v) in xs.iter().zip(x) {
        env.insert(*s, *v);
    }
    for (s, v) in psyms.iter().zip(th) {
        env.insert(*s, *v);
    }
    let mut prob = Problem::new(
        d,
        PdeOperator::poisson(),
        ScalarField::Zero,
        ScalarField::Zero,
    )
    .unwrap();
    if let Some(tt) = time {
        prob = prob.with_time(tt).unwrap();
    }
    let ev = Evaluator::new(t, &prob, x, &[]).unwrap();
    let jets = ev.jets_at(e, th, x).unwrap();
    let jj = ev.jet_jacobian_at(e, th, x).unwrap();
    let np = t.n_params();
    // u
    let u_ref = ev_at(usym, &env, &pool);
    assert!(close(jets.u[0], u_ref, 1e-10), "u {} vs {u_ref}", jets.u[0]);
    // ∇u, Δu
    let mut lap_expr_terms = Vec::new();
    let mut grads = Vec::new();
    for i in 0..d {
        let gi = diff(usym, xs[i], &pool).unwrap().value;
        grads.push(gi);
        let g_ref = ev_at(gi, &env, &pool);
        assert!(
            close(jets.grad[i], g_ref, 1e-9),
            "grad[{i}] {} vs {g_ref}",
            jets.grad[i]
        );
        if Some(i) != time {
            lap_expr_terms.push(diff(gi, xs[i], &pool).unwrap().value);
        }
    }
    let lap = pool.add(lap_expr_terms);
    let lap_ref = ev_at(lap, &env, &pool);
    assert!(
        close(jets.lap[0], lap_ref, 1e-9),
        "lap {} vs {lap_ref}",
        jets.lap[0]
    );
    // Jacobian rows: u, Δu, ∂ᵢu  w.r.t. every θ_k.
    let rows: Vec<ExprId> = std::iter::once(usym)
        .chain(std::iter::once(lap))
        .chain(grads.iter().copied())
        .collect();
    for (ri, &row) in rows.iter().enumerate() {
        for k in 0..np {
            let dk = diff(row, psyms[k], &pool).unwrap().value;
            let want = ev_at(dk, &env, &pool);
            let got = jj[ri * np + k];
            assert!(
                close(got, want, 1e-8),
                "J[{ri}][{k}] {got} vs {want} for {}",
                t.describe(e).unwrap()
            );
        }
    }
}

#[test]
fn fused_derivatives_match_symbolic_diff_random() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for shape in ["depth2_sub", "depth1", "depth2", "B(U(L),U(L))"] {
        let t = Template::new(shape, &SMOOTH, &PAPER_BINARY, 3).unwrap();
        for _ in 0..6 {
            let e: Vec<usize> = t.slot_arities().iter().map(|&k| rng.below(k)).collect();
            let th: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.8, 0.8)).collect();
            let x: Vec<f64> = (0..3).map(|_| rng.unif(-0.9, 0.9)).collect();
            check_candidate_derivatives(&t, &e, &th, &x, None);
        }
    }
}

#[test]
fn fused_derivatives_with_time_and_division_and_log() {
    let u = [
        UnaryOp::Id,
        UnaryOp::Sq,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sin,
    ];
    let b = [BinaryOp::Div, BinaryOp::Mul, BinaryOp::Sub];
    let t = Template::new("U(B(L,L))", &u, &b, 3).unwrap();
    // log( (L[sq]) / (L[exp]) ) with positive weights: argument > 0.
    let e = [1, 0, 2, 3];
    let th = [0.7, 0.4, 0.9, 0.3, 0.5, 0.2, 0.6, 0.8, 1.3, -0.2];
    let x = [0.3, -0.6, 0.45];
    check_candidate_derivatives(&t, &e, &th, &x, None);
    check_candidate_derivatives(&t, &e, &th, &x, Some(0));
    // sin(L[id] * L[sq]) with time coordinate 2
    let e2 = [0, 1, 1, 4];
    check_candidate_derivatives(&t, &e2, &th, &x, Some(2));
}

#[test]
fn residual_jacobian_matches_finite_differences() {
    let mut rng = Rng(42);
    let d = 4;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let prob = poisson_problem(d);
    let xi = grid(&mut rng, 30, d, -1.0, 1.0);
    let xb = grid(&mut rng, 10, d, -1.0, 1.0);
    let ev = Evaluator::new(&t, &prob, &xi, &xb).unwrap();
    let e = [7, 1, 3, 8]; // cos(L[sin] * L[sq])
    let th: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.5, 0.5)).collect();
    let m = ev.n_residuals();
    let np = t.n_params();
    let mut r = vec![0.0; m];
    let mut j = vec![0.0; m * np];
    assert!(ev
        .residual_and_jacobian(&e, &th, &mut r, Some(&mut j))
        .unwrap());
    let h = 1e-6;
    for k in 0..np {
        let mut tp = th.clone();
        let mut tm = th.clone();
        tp[k] += h;
        tm[k] -= h;
        let mut rp = vec![0.0; m];
        let mut rm = vec![0.0; m];
        ev.residual_and_jacobian(&e, &tp, &mut rp, None).unwrap();
        ev.residual_and_jacobian(&e, &tm, &mut rm, None).unwrap();
        for row in 0..m {
            let fd = (rp[row] - rm[row]) / (2.0 * h);
            assert!(
                (fd - j[row * np + k]).abs() < 1e-6 * (1.0 + fd.abs()),
                "row {row} param {k}: fd {fd} vs J {}",
                j[row * np + k]
            );
        }
    }
    // ‖r‖² is the loss, and the gradient is 2Jᵀr.
    let loss = ev.loss(&e, &th).unwrap();
    assert!(close(loss, r.iter().map(|v| v * v).sum(), 1e-12));
    let mut g = vec![0.0; np];
    ev.loss_and_grad(&e, &th, &mut g).unwrap();
    for k in 0..np {
        let want: f64 = (0..m).map(|row| 2.0 * r[row] * j[row * np + k]).sum();
        assert!(close(g[k], want, 1e-10));
    }
}

#[test]
fn batch_agrees_with_single_candidate_calls() {
    let mut rng = Rng(7);
    let d = 3;
    let t = Template::new("depth2", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let prob = poisson_problem(d);
    let xi = grid(&mut rng, 300, d, -1.0, 1.0);
    let xb = grid(&mut rng, 50, d, -1.0, 1.0);
    let ev = Evaluator::new(&t, &prob, &xi, &xb).unwrap();
    let b = 5;
    let ns = t.n_slots();
    let np = t.n_params();
    let m = ev.n_residuals();
    let es: Vec<usize> = (0..b)
        .flat_map(|_| {
            t.slot_arities()
                .into_iter()
                .map(|k| rng.below(k))
                .collect::<Vec<_>>()
        })
        .collect();
    let ths: Vec<f64> = (0..b * np).map(|_| rng.unif(-0.3, 0.3)).collect();
    let mut r = vec![0.0; b * m];
    let mut j = vec![0.0; b * m * np];
    let ok = ev
        .residual_and_jacobian_batch(&es, &ths, &mut r, Some(&mut j))
        .unwrap();
    let losses = ev.loss_batch(&es, &ths).unwrap();
    for c in 0..b {
        let e = &es[c * ns..(c + 1) * ns];
        let th = &ths[c * np..(c + 1) * np];
        let mut r1 = vec![0.0; m];
        let mut j1 = vec![0.0; m * np];
        let ok1 = ev
            .residual_and_jacobian(e, th, &mut r1, Some(&mut j1))
            .unwrap();
        assert_eq!(ok[c], ok1);
        assert_eq!(&r[c * m..(c + 1) * m], &r1[..]);
        assert_eq!(&j[c * m * np..(c + 1) * m * np], &j1[..]);
        let l1 = ev.loss(e, th).unwrap();
        assert!(close(losses[c], l1, 1e-12) || (losses[c].is_infinite() && l1.is_infinite()));
    }
}

// ---------------------------------------------------------------------------
// Interpreter vs instantiate, exhaustively over a small template
// ---------------------------------------------------------------------------

#[test]
fn interpreter_agrees_with_instantiate_on_every_skeleton() {
    let d = 2;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let pool = ExprPool::new();
    let xs = coords(&pool, d);
    let prob = poisson_problem(d);
    let mut rng = Rng(99);
    let pts = grid(&mut rng, 3, d, -1.0, 1.0);
    let ev = Evaluator::new(&t, &prob, &pts, &[]).unwrap();
    let th: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.6, 0.6)).collect();
    let ar = t.slot_arities();
    let mut count = 0;
    for a in 0..ar[0] {
        for b in 0..ar[1] {
            for c in 0..ar[2] {
                for u in 0..ar[3] {
                    let e = [a, b, c, u];
                    let expr = t.instantiate(&e, &th, &xs, &pool).unwrap();
                    let jets = ev.jets_at(&e, &th, &pts).unwrap();
                    for p in 0..3 {
                        let mut env = HashMap::new();
                        env.insert(xs[0], pts[p * d]);
                        env.insert(xs[1], pts[p * d + 1]);
                        let want = ev_at(expr, &env, &pool);
                        assert!(
                            close(jets.u[p], want, 1e-12),
                            "{e:?}: {} vs {want}",
                            jets.u[p]
                        );
                    }
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 2187);
}

// ---------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------

#[test]
fn overflow_and_division_by_zero_score_infinity_not_nan() {
    let u = [UnaryOp::Zero, UnaryOp::Id, UnaryOp::Exp, UnaryOp::Log];
    let b = [BinaryOp::Add, BinaryOp::Div];
    let d = 2;
    let t = Template::new("U(B(L,L))", &u, &b, d).unwrap();
    let prob = poisson_problem(d);
    let mut rng = Rng(3);
    let xi = grid(&mut rng, 20, d, -1.0, 1.0);
    let xb = grid(&mut rng, 5, d, -1.0, 1.0);
    let ev = Evaluator::new(&t, &prob, &xi, &xb).unwrap();
    let np = t.n_params();
    let m = ev.n_residuals();
    let cases: Vec<([usize; 4], Vec<f64>)> = vec![
        // exp(exp(huge)) overflow
        (
            [2, 0, 0, 2],
            vec![500.0, 500.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
        // L[0] / L[0] with zero biases: 0/0
        ([0, 1, 0, 1], vec![0.0; 8]),
        // log of a negative number
        ([0, 0, 0, 3], vec![0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0]),
    ];
    for (e, th) in cases {
        assert_eq!(th.len(), np);
        let mut r = vec![0.0; m];
        let mut j = vec![1.0; m * np];
        let ok = ev
            .residual_and_jacobian(&e, &th, &mut r, Some(&mut j))
            .unwrap();
        assert!(!ok, "{e:?} should be invalid");
        assert!(r.iter().all(|v| *v == f64::INFINITY));
        assert!(j.iter().all(|v| *v == 0.0));
        let l = ev.loss(&e, &th).unwrap();
        assert_eq!(l, f64::INFINITY);
        let mut g = vec![1.0; np];
        assert_eq!(ev.loss_and_grad(&e, &th, &mut g).unwrap(), f64::INFINITY);
        assert!(g.iter().all(|v| *v == 0.0));
        let fr = fit(&ev, &e, &th, &FitOptions::default()).unwrap();
        assert!(!fr.loss.is_nan());
    }
}

// ---------------------------------------------------------------------------
// Canonicalisation
// ---------------------------------------------------------------------------

fn key(t: &Template, e: &[usize]) -> String {
    canonical_key(t, e).unwrap()
}

#[test]
fn canonical_key_rewrites() {
    // PAPER_UNARY: 0 1 id sq cube quartic exp sin cos; PAPER_BINARY: + * -
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, 3).unwrap();
    // commutativity of + and *
    assert_eq!(key(&t, &[3, 0, 7, 6]), key(&t, &[7, 0, 3, 6]));
    assert_eq!(key(&t, &[3, 1, 7, 6]), key(&t, &[7, 1, 3, 6]));
    // − over a scale-closed operand is +
    assert_eq!(key(&t, &[3, 2, 7, 6]), key(&t, &[3, 0, 7, 6]));
    // id(L[sq] + L[0]) = L[sq] — id chain and 0-padding
    assert_eq!(key(&t, &[3, 0, 0, 2]), "L[sq]");
    assert_eq!(key(&t, &[0, 0, 3, 2]), "L[sq]");
    assert_eq!(key(&t, &[3, 2, 1, 2]), "L[sq]");
    // 0 and 1 are both free constants; anything of a constant is constant
    assert_eq!(key(&t, &[0, 1, 1, 6]), "c");
    assert_eq!(key(&t, &[0, 0, 0, 7]), "c");
    assert_eq!(key(&t, &[5, 0, 5, 0]), "c");
    // cos(T) = sin(T) for a shift-closed T
    assert_eq!(key(&t, &[2, 0, 0, 8]), key(&t, &[2, 0, 0, 7]));
    // L[φ] + L[φ] = L[φ]
    assert_eq!(key(&t, &[7, 0, 7, 2]), "L[sin]");
    // c · L[φ] = L[φ]
    assert_eq!(key(&t, &[1, 1, 4, 2]), "L[cube]");
    // genuinely different skeletons stay apart
    assert_ne!(key(&t, &[3, 1, 7, 2]), key(&t, &[3, 0, 7, 2]));
    assert_ne!(key(&t, &[3, 0, 7, 3]), key(&t, &[3, 0, 7, 2]));
    // determinism / idempotence
    assert_eq!(key(&t, &[6, 1, 2, 7]), key(&t, &[6, 1, 2, 7]));
}

#[test]
fn canonical_classes_of_the_paper_template() {
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, 3).unwrap();
    let reps = canonical_skeletons(&t, 1 << 20).unwrap();
    // The 2187 opcode vectors collapse to far fewer families.
    assert!(reps.len() < 700, "{} classes", reps.len());
    let keys: std::collections::HashSet<&str> = reps.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys.len(), reps.len());
    assert!(keys.contains("L[sq]"));
    assert!(keys.contains("sin((L[id] + L[cube]))"));
}

/// Order-0 regression problem `u ≈ target(x)`.
fn regression(d: usize, target: impl Fn(&[f64]) -> f64 + Send + Sync + 'static) -> Problem {
    Problem::new(
        d,
        PdeOperator::Linear(LinearOperator {
            lap: 0.0,
            grad: vec![],
            u: 1.0,
            cubic: 0.0,
            potential: false,
        }),
        ScalarField::Func(Arc::new(target)),
        ScalarField::Zero,
    )
    .unwrap()
}

#[test]
fn equal_keys_mean_equal_families_numerically() {
    // For each pair of skeletons with the same key, a random instance of the
    // first is fitted (to ~0) by the second.
    let d = 2;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let pairs: [([usize; 4], [usize; 4]); 5] = [
        ([3, 0, 7, 6], [7, 0, 3, 6]),
        ([3, 2, 7, 6], [3, 0, 7, 6]),
        ([3, 0, 0, 2], [3, 2, 1, 2]),
        ([2, 0, 0, 8], [2, 0, 0, 7]),
        ([7, 0, 7, 2], [7, 2, 7, 2]),
    ];
    let mut rng = Rng(11);
    let pts = grid(&mut rng, 60, d, -1.0, 1.0);
    for (a, b) in pairs {
        assert_eq!(key(&t, &a), key(&t, &b));
        let th: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.7, 0.7)).collect();
        let pool = ExprPool::new();
        let xs = coords(&pool, d);
        let ua = t.instantiate(&a, &th, &xs, &pool).unwrap();
        let fa = crate::jit::compile(ua, &xs, &pool).unwrap();
        let fa = Arc::new(fa);
        let prob = regression(d, move |x| fa.call(x));
        let ev = Evaluator::new(&t, &prob, &pts, &[]).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..8 {
            let th0: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-1.0, 1.0)).collect();
            let r = fit(
                &ev,
                &b,
                &th0,
                &FitOptions {
                    method: FitMethod::Lm { max_iter: 200 },
                    tol: 1e-26,
                    ..Default::default()
                },
            )
            .unwrap();
            best = best.min(r.loss);
            if best < 1e-18 {
                break;
            }
        }
        assert!(best < 1e-14, "{a:?} ~ {b:?}: best loss {best}");
    }
}

#[test]
fn redundant_params_keep_the_family() {
    let d = 2;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    // id(L[sq] + L[0]): root α, β and one of the constants are redundant.
    let e = [3, 0, 0, 2];
    let fixed = redundant_params(&t, &e).unwrap();
    assert_eq!(fixed[6], Some(1.0));
    assert_eq!(fixed[7], Some(0.0));
    assert_eq!(fixed[3], Some(0.0)); // L[0]'s weights
    assert_eq!(fixed[4], Some(0.0));
    // The pinned family still fits ½Σx² exactly.
    let prob = poisson_problem(d);
    let mut rng = Rng(5);
    let xi = grid(&mut rng, 80, d, -1.0, 1.0);
    let xb = grid(&mut rng, 40, d, -1.0, 1.0);
    let ev = Evaluator::new(&t, &prob, &xi, &xb).unwrap();
    let mut th: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.3, 0.3)).collect();
    for (k, f) in fixed.iter().enumerate() {
        if let Some(v) = f {
            th[k] = *v;
        }
    }
    let r = fit(&ev, &e, &th, &FitOptions::default()).unwrap();
    assert!(r.loss < 1e-20, "loss {}", r.loss);
    for (k, f) in fixed.iter().enumerate() {
        if let Some(v) = f {
            assert_eq!(r.theta[k], *v, "pinned parameter {k} moved");
        }
    }
}

#[test]
fn dedup_cache_and_same_function() {
    let d = 2;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let mut cache = DedupCache::new();
    let k1 = key(&t, &[3, 0, 7, 6]);
    assert!(cache.insert(k1.clone(), &[3, 0, 7, 6], &[0.0; 12], 1.0));
    let k2 = key(&t, &[7, 0, 3, 6]);
    assert!(cache.lookup(&k2).is_some());
    assert!(!cache.insert(k2.clone(), &[7, 0, 3, 6], &[0.0; 12], 2.0));
    assert!(cache.insert(k2, &[7, 0, 3, 6], &[0.0; 12], 0.5));
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.best(1)[0].1.loss, 0.5);

    let pool = ExprPool::new();
    let xs = coords(&pool, d);
    // id(L[sq] + L[0]) with ½ weights vs id(L[0] + L[sq]) with the same: equal.
    let th1 = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let th2 = [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0];
    let same = same_function(
        &t,
        &[3, 0, 0, 2],
        &th1,
        &[0, 0, 3, 2],
        &th2,
        &xs,
        &pool,
        1000,
        1e-9,
    )
    .unwrap();
    assert_eq!(same, Some(true));
    // ... with a slightly different weight (0.5000000001 snaps to 1/2).
    let th3 = [0.0, 0.0, 0.0, 0.500_000_000_1, 0.5, 0.0, 1.0, 0.0];
    let same = same_function(
        &t,
        &[3, 0, 0, 2],
        &th1,
        &[0, 0, 3, 2],
        &th3,
        &xs,
        &pool,
        1000,
        1e-9,
    )
    .unwrap();
    assert_eq!(same, Some(true));
    let th4 = [0.0, 0.0, 0.0, 0.25, 0.5, 0.0, 1.0, 0.0];
    let same = same_function(
        &t,
        &[3, 0, 0, 2],
        &th1,
        &[0, 0, 3, 2],
        &th4,
        &xs,
        &pool,
        1000,
        1e-9,
    )
    .unwrap();
    assert_ne!(same, Some(true));
}

// ---------------------------------------------------------------------------
// Problems and fitting
// ---------------------------------------------------------------------------

#[test]
fn lm_recovers_poisson_solution_to_float64_precision() {
    let d = 10;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let prob = poisson_problem(d);
    let mut rng = Rng(2024);
    let xi = grid(&mut rng, 400, d, -1.0, 1.0);
    let xb: Vec<f64> = {
        let mut v = grid(&mut rng, 200, d, -1.0, 1.0);
        for p in 0..200 {
            let i = rng.below(d);
            v[p * d + i] = if p % 2 == 0 { -1.0 } else { 1.0 };
        }
        v
    };
    let ev = Evaluator::new(&t, &prob, &xi, &xb).unwrap();
    let e = [3, 0, 0, 2]; // id(L[sq] + L[0])
    let th0: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.1, 0.1)).collect();
    let r = fit(&ev, &e, &th0, &FitOptions::default()).unwrap();
    assert!(r.loss < 1e-24, "loss {}", r.loss);
    // relative L2 error against ½Σx² on fresh points
    let test = grid(&mut rng, 2000, d, -1.0, 1.0);
    let j = ev.jets_at(&e, &r.theta, &test).unwrap();
    let (mut num, mut den) = (0.0, 0.0);
    for p in 0..2000 {
        let exact = 0.5 * test[p * d..(p + 1) * d].iter().map(|v| v * v).sum::<f64>();
        num += (j.u[p] - exact).powi(2);
        den += exact * exact;
    }
    assert!((num / den).sqrt() < 1e-11, "rel L2 {}", (num / den).sqrt());
}

#[test]
fn lm_recovers_conservation_law_solution() {
    // (d−1)π/4 · u_t − Σ_{i≥1} u_{x_i} = 0, u(0,x) = sin(π/4 Σ x_i); x_0 = t.
    let d = 5;
    let c = std::f64::consts::FRAC_PI_4;
    let mut coef = vec![-1.0; d];
    coef[0] = (d as f64 - 1.0) * c;
    let prob = Problem::new(
        d,
        PdeOperator::transport(coef),
        ScalarField::Zero,
        ScalarField::Func(Arc::new(move |x: &[f64]| {
            (c * x[1..].iter().sum::<f64>()).sin()
        })),
    )
    .unwrap()
    .with_time(0)
    .unwrap();
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let mut rng = Rng(77);
    let xi = grid(&mut rng, 400, d, 0.0, 1.0);
    let mut xb = grid(&mut rng, 200, d, 0.0, 1.0);
    for p in 0..200 {
        xb[p * d] = 0.0;
    }
    let ev = Evaluator::new(&t, &prob, &xi, &xb).unwrap();
    let e = [2, 0, 0, 7]; // sin(L[id] + L[0])
    let mut best = f64::INFINITY;
    for _ in 0..10 {
        let th0: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.5, 0.5)).collect();
        let r = fit(&ev, &e, &th0, &FitOptions::default()).unwrap();
        best = best.min(r.loss);
        if best < 1e-24 {
            break;
        }
    }
    assert!(best < 1e-22, "best loss {best}");
}

#[test]
fn adam_and_bfgs_reduce_the_loss() {
    let d = 3;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let prob = poisson_problem(d);
    let mut rng = Rng(8);
    let xi = grid(&mut rng, 200, d, -1.0, 1.0);
    let xb = grid(&mut rng, 100, d, -1.0, 1.0);
    let ev = Evaluator::new(&t, &prob, &xi, &xb).unwrap();
    let e = [3, 0, 0, 2];
    let th0: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.1, 0.1)).collect();
    let l0 = ev.loss(&e, &th0).unwrap();
    for method in [
        FitMethod::Adam {
            iters: 200,
            lr: 1e-2,
        },
        FitMethod::Bfgs { iters: 50 },
        FitMethod::AdamBfgs {
            adam: 20,
            lr: 1e-3,
            bfgs: 20,
        },
    ] {
        let r = fit(
            &ev,
            &e,
            &th0,
            &FitOptions {
                method,
                tol: 1e-30,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.loss < l0 * 0.5, "{method:?}: {} vs {l0}", r.loss);
    }
    let r = fit(
        &ev,
        &e,
        &th0,
        &FitOptions {
            method: FitMethod::Bfgs { iters: 300 },
            tol: 1e-30,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(r.loss < 1e-12, "bfgs {}", r.loss);
}

fn fd_grad(ev: &Evaluator, e: &[usize], th: &[f64]) -> Vec<f64> {
    let h = 1e-6;
    (0..th.len())
        .map(|k| {
            let mut a = th.to_vec();
            let mut b = th.to_vec();
            a[k] += h;
            b[k] -= h;
            (ev.loss(e, &a).unwrap() - ev.loss(e, &b).unwrap()) / (2.0 * h)
        })
        .collect()
}

#[test]
fn rayleigh_and_normalized_gradients_match_finite_differences() {
    let d = 3;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let mut rng = Rng(19);
    let xi = grid(&mut rng, 150, d, -1.0, 1.0);
    let xb = grid(&mut rng, 40, d, -1.0, 1.0);
    let v = ScalarField::Func(Arc::new(|x: &[f64]| x.iter().map(|a| a.cos()).sum::<f64>()));
    let ray = Problem::new(
        d,
        PdeOperator::poisson(),
        ScalarField::Zero,
        ScalarField::Zero,
    )
    .unwrap()
    .with_potential(v.clone())
    .with_loss(LossKind::Rayleigh);
    let nls = Problem::new(
        d,
        PdeOperator::cubic_schrodinger(),
        ScalarField::Zero,
        ScalarField::Zero,
    )
    .unwrap()
    .with_potential(v)
    .with_loss(LossKind::Normalized { power: 3.0 })
    .with_bc_weight(0.0);
    let e = [8, 0, 3, 6]; // exp(L[cos] + L[sq])
    let th: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.4, 0.4)).collect();
    for prob in [ray, nls] {
        let ev = Evaluator::new(&t, &prob, &xi, &xb).unwrap();
        let mut g = vec![0.0; t.n_params()];
        let l = ev.loss_and_grad(&e, &th, &mut g).unwrap();
        assert!(l.is_finite());
        let fd = fd_grad(&ev, &e, &th);
        for k in 0..g.len() {
            assert!(
                (g[k] - fd[k]).abs() < 1e-5 * (1.0 + fd[k].abs()),
                "{:?} k={k}: {} vs {}",
                prob.loss(),
                g[k],
                fd[k]
            );
        }
        let mut r = vec![0.0; ev.n_residuals()];
        let err = ev.residual_and_jacobian(&e, &th, &mut r, None).unwrap_err();
        assert_eq!(err.code(), "E-FEX-008");
        let err = fit(&ev, &e, &th, &FitOptions::default()).unwrap_err();
        assert_eq!(err.code(), "E-FEX-008");
    }
}

#[test]
fn expression_operator_matches_builtin() {
    let d = 3;
    let pool = ExprPool::new();
    let js = JetSymbols::new(&pool, d, false);
    // −Δu is recognised as the linear fast path.
    let neg_lap = pool.mul(vec![pool.integer(-1), js.lap]);
    let op = PdeOperator::from_expr(neg_lap, &js, None, &pool).unwrap();
    assert!(matches!(op, PdeOperator::Linear(ref l) if l.lap == -1.0 && l.u == 0.0));
    // −Δu + u³ + V(x)·u with V = Σ cos xᵢ, as an expression, vs the builtin.
    let vexpr = pool.add(
        js.coords
            .iter()
            .map(|&x| pool.func("cos", vec![x]))
            .collect(),
    );
    let f = pool.add(vec![
        neg_lap,
        pool.pow(js.u, pool.integer(3)),
        pool.mul(vec![vexpr, js.u]),
    ]);
    let op_expr = PdeOperator::from_expr(f, &js, None, &pool).unwrap();
    assert!(matches!(op_expr, PdeOperator::Compiled(_)));
    assert_eq!(op_expr.order(), 2);
    let p_expr = Problem::new(d, op_expr, ScalarField::Zero, ScalarField::Zero).unwrap();
    let p_builtin = Problem::new(
        d,
        PdeOperator::cubic_schrodinger(),
        ScalarField::Zero,
        ScalarField::Zero,
    )
    .unwrap()
    .with_potential(ScalarField::from_expr(vexpr, &js.coords, &pool).unwrap());
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let mut rng = Rng(31);
    let xi = grid(&mut rng, 64, d, -1.0, 1.0);
    let xb = grid(&mut rng, 16, d, -1.0, 1.0);
    let e1 = Evaluator::new(&t, &p_expr, &xi, &xb).unwrap();
    let e2 = Evaluator::new(&t, &p_builtin, &xi, &xb).unwrap();
    let e = [7, 1, 3, 6];
    let th: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.4, 0.4)).collect();
    let m = e1.n_residuals();
    let np = t.n_params();
    let (mut r1, mut r2) = (vec![0.0; m], vec![0.0; m]);
    let (mut j1, mut j2) = (vec![0.0; m * np], vec![0.0; m * np]);
    e1.residual_and_jacobian(&e, &th, &mut r1, Some(&mut j1))
        .unwrap();
    e2.residual_and_jacobian(&e, &th, &mut r2, Some(&mut j2))
        .unwrap();
    for i in 0..m {
        assert!(close(r1[i], r2[i], 1e-12));
    }
    for i in 0..m * np {
        assert!(close(j1[i], j2[i], 1e-11));
    }
    // u_t needs a time coordinate.
    let js_t = JetSymbols::new(&pool, d, true);
    let bad = PdeOperator::from_expr(js_t.ut.unwrap(), &js_t, None, &pool).unwrap_err();
    assert_eq!(bad.code(), "E-FEX-007");
    let ok = PdeOperator::from_expr(js_t.ut.unwrap(), &js_t, Some(0), &pool).unwrap();
    assert!(matches!(ok, PdeOperator::Linear(ref l) if l.grad[0] == 1.0));
    // A stray symbol is refused.
    let y = pool.symbol("y", Domain::Real);
    let bad = PdeOperator::from_expr(pool.mul(vec![y, js.u]), &js, None, &pool).unwrap_err();
    assert_eq!(bad.code(), "E-FEX-006");
}

#[test]
fn fit_batch_runs_every_candidate() {
    let d = 3;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let prob = poisson_problem(d);
    let mut rng = Rng(12);
    let xi = grid(&mut rng, 100, d, -1.0, 1.0);
    let xb = grid(&mut rng, 50, d, -1.0, 1.0);
    let ev = Evaluator::new(&t, &prob, &xi, &xb).unwrap();
    let es = [3, 0, 0, 2, 7, 1, 8, 6, 2, 0, 0, 3];
    let th0: Vec<f64> = (0..3 * t.n_params()).map(|_| rng.unif(-0.1, 0.1)).collect();
    let rs = fit_batch(&ev, &es, &th0, &FitOptions::default()).unwrap();
    assert_eq!(rs.len(), 3);
    assert!(rs[0].loss < 1e-20);
    assert!(rs[0].score() > rs[1].score());
}

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

#[test]
fn finalize_error_codes_match_the_registry() {
    // E-FEX-009 … 011 are raised from Python (alkahest.fex.finalize /
    // recovery_metric); the Rust variants carry the code and remediation the
    // Python side uses, and the registry agrees with them.
    for (e, code) in [
        (FexError::Finalize { detail: "x".into() }, "E-FEX-009"),
        (FexError::Family { detail: "x".into() }, "E-FEX-010"),
        (FexError::Recovery { detail: "x".into() }, "E-FEX-011"),
    ] {
        assert_eq!(e.code(), code);
        let spec = crate::errors::codes::REGISTRY
            .iter()
            .find(|s| s.code == code)
            .expect("registered");
        assert_eq!(spec.remediation, e.remediation());
        assert!(!e.to_string().is_empty());
    }
}

#[test]
fn error_codes() {
    let e = Template::new("U(B(L,L)", &PAPER_UNARY, &PAPER_BINARY, 2).unwrap_err();
    assert_eq!(e.code(), "E-FEX-001");
    assert!(e.remediation().is_some());
    assert_eq!(
        Template::new("Q", &PAPER_UNARY, &PAPER_BINARY, 2)
            .unwrap_err()
            .code(),
        "E-FEX-001"
    );
    assert_eq!(
        Template::new("B(L,L)", &PAPER_UNARY, &[], 2)
            .unwrap_err()
            .code(),
        "E-FEX-001"
    );
    assert_eq!(
        Template::from_names("L", &["sinh"], &[], 2)
            .unwrap_err()
            .code(),
        "E-FEX-002"
    );
    assert_eq!(
        Template::from_names("B(L,L)", &["id"], &["^"], 2)
            .unwrap_err()
            .code(),
        "E-FEX-002"
    );
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, 2).unwrap();
    assert_eq!(t.check_opcodes(&[0, 0, 0]).unwrap_err().code(), "E-FEX-003");
    assert_eq!(
        t.check_opcodes(&[0, 3, 0, 0]).unwrap_err().code(),
        "E-FEX-003"
    );
    let pool = ExprPool::new();
    let xs = coords(&pool, 2);
    assert_eq!(
        t.instantiate(&[0, 0, 0, 0], &[0.0; 3], &xs, &pool)
            .unwrap_err()
            .code(),
        "E-FEX-004"
    );
    assert_eq!(
        t.instantiate(&[0, 0, 0, 0], &[0.0; 8], &xs[..1], &pool)
            .unwrap_err()
            .code(),
        "E-FEX-007"
    );
    let prob = poisson_problem(2);
    assert_eq!(
        Evaluator::new(&t, &prob, &[0.0; 3], &[])
            .err()
            .unwrap()
            .code(),
        "E-FEX-005"
    );
    assert_eq!(
        Evaluator::new(&t, &prob, &[], &[]).err().unwrap().code(),
        "E-FEX-005"
    );
    let p3 = poisson_problem(3);
    assert_eq!(
        Evaluator::new(&t, &p3, &[0.0; 6], &[])
            .err()
            .unwrap()
            .code(),
        "E-FEX-007"
    );
    assert_eq!(
        poisson_problem(2).with_time(2).unwrap_err().code(),
        "E-FEX-007"
    );
    // every FexError code is registered
    for code in [
        "E-FEX-001",
        "E-FEX-002",
        "E-FEX-003",
        "E-FEX-004",
        "E-FEX-005",
        "E-FEX-006",
        "E-FEX-007",
        "E-FEX-008",
        "E-FEX-009",
        "E-FEX-010",
        "E-FEX-011",
    ] {
        assert!(
            crate::errors::codes::REGISTRY
                .iter()
                .any(|s| s.code == code),
            "{code}"
        );
    }
}

#[test]
fn tanh_derivatives_match_symbolic_diff() {
    // tanh in leaves and inner slots: the fused jets and their Jacobian rows
    // against symbolic diff of the instantiated candidate.
    let u = [UnaryOp::Id, UnaryOp::Tanh, UnaryOp::Sq, UnaryOp::Sin];
    let t = Template::new("U(B(L,L))", &u, &PAPER_BINARY, 3).unwrap();
    let mut rng = Rng(0x7a11);
    for e in [[1, 0, 0, 1], [1, 1, 2, 1], [0, 2, 1, 1], [1, 0, 3, 1]] {
        let th: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.9, 0.9)).collect();
        let x: Vec<f64> = (0..3).map(|_| rng.unif(-0.9, 0.9)).collect();
        check_candidate_derivatives(&t, &e, &th, &x, None);
    }
    // eval4 against finite differences of tanh itself.
    for s in [-2.5, -0.3, 0.0, 0.7, 3.1] {
        let [f, f1, f2, f3] = UnaryOp::Tanh.eval4(s);
        let h = 1e-5;
        let [fp, f1p, f2p, _] = UnaryOp::Tanh.eval4(s + h);
        let [fm, f1m, f2m, _] = UnaryOp::Tanh.eval4(s - h);
        assert!(close(f, f64::tanh(s), 1e-15));
        assert!(close(f1, (fp - fm) / (2.0 * h), 1e-8));
        assert!(close(f2, (f1p - f1m) / (2.0 * h), 1e-8));
        assert!(close(f3, (f2p - f2m) / (2.0 * h), 1e-8));
    }
    assert_eq!(UnaryOp::parse("tanh").unwrap(), UnaryOp::Tanh);
    assert_eq!(UnaryOp::Tanh.name(), "tanh");
}

#[test]
fn centered_least_squares_removes_the_interior_mean() {
    // −Δu = c with c unknown: every u = a·Σx² + b solves it for c = −2ad, so
    // the centred loss is zero there while the plain one is not; and the
    // centred Jacobian matches finite differences of the centred residual.
    let d = 3;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let mut rng = Rng(77);
    let xi = grid(&mut rng, 60, d, -1.0, 1.0);
    let mk = |loss: LossKind| {
        Problem::new(
            d,
            PdeOperator::poisson(),
            ScalarField::Zero,
            ScalarField::Zero,
        )
        .unwrap()
        .with_loss(loss)
        .with_bc_weight(0.0)
    };
    let centred = mk(LossKind::CenteredLeastSquares);
    assert!(centred.loss().is_least_squares());
    let plain = mk(LossKind::LeastSquares);
    let e = [3, 0, 0, 2]; // id(L[sq] + L[0])
    let mut th = vec![0.0; t.n_params()];
    for w in &mut th[0..d] {
        *w = 0.37;
    }
    th[d] = 0.2;
    th[t.n_params() - 2] = 1.0;
    let evc = Evaluator::new(&t, &centred, &xi, &[]).unwrap();
    let evp = Evaluator::new(&t, &plain, &xi, &[]).unwrap();
    assert!(evc.loss(&e, &th).unwrap() < 1e-28);
    assert!(evp.loss(&e, &th).unwrap() > 1.0);
    // A generic candidate: Jacobian rows against finite differences, mean
    // zero, and the gradient is 2Jᵀr.
    let e2 = [7, 1, 3, 8];
    let th2: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.5, 0.5)).collect();
    let m = evc.n_residuals();
    let np = t.n_params();
    let mut r = vec![0.0; m];
    let mut j = vec![0.0; m * np];
    assert!(evc
        .residual_and_jacobian(&e2, &th2, &mut r, Some(&mut j))
        .unwrap());
    assert!(r.iter().sum::<f64>().abs() < 1e-12);
    let h = 1e-6;
    for k in 0..np {
        let (mut a, mut b) = (th2.clone(), th2.clone());
        a[k] += h;
        b[k] -= h;
        let (mut ra, mut rb) = (vec![0.0; m], vec![0.0; m]);
        evc.residual_and_jacobian(&e2, &a, &mut ra, None).unwrap();
        evc.residual_and_jacobian(&e2, &b, &mut rb, None).unwrap();
        for row in 0..m {
            let fd = (ra[row] - rb[row]) / (2.0 * h);
            assert!((fd - j[row * np + k]).abs() < 1e-6 * (1.0 + fd.abs()));
        }
    }
    let mut g = vec![0.0; np];
    let l = evc.loss_and_grad(&e2, &th2, &mut g).unwrap();
    assert!(close(l, r.iter().map(|v| v * v).sum(), 1e-12));
    for k in 0..np {
        let want: f64 = (0..m).map(|row| 2.0 * r[row] * j[row * np + k]).sum();
        assert!(close(g[k], want, 1e-10));
    }
    // Levenberg–Marquardt applies and fits the family.
    let th0: Vec<f64> = (0..np).map(|_| rng.unif(-0.3, 0.3)).collect();
    let res = fit(&evc, &e, &th0, &FitOptions::default()).unwrap();
    assert!(res.loss < 1e-20, "centred LM loss {}", res.loss);
}

#[test]
fn search_layer_error_codes_match_the_registry() {
    // E-FEX-012 … 014 are raised from Python (alkahest.fex.transform_problem /
    // reduce_problem, search / solve options, TorchEvaluator); the Rust
    // variants carry the code and remediation the Python side uses.
    for (e, code) in [
        (FexError::Transform { detail: "x".into() }, "E-FEX-012"),
        (FexError::SearchOptions { detail: "x".into() }, "E-FEX-013"),
        (FexError::Backend { detail: "x".into() }, "E-FEX-014"),
    ] {
        assert_eq!(e.code(), code);
        let spec = crate::errors::codes::REGISTRY
            .iter()
            .find(|s| s.code == code)
            .expect("registered");
        assert_eq!(spec.remediation, e.remediation());
        assert!(!e.to_string().is_empty());
    }
}

/// Residual rows and Jacobian against central finite differences.
fn check_jacobian_fd(ev: &Evaluator, e: &[usize], th: &[f64], tol: f64) {
    let m = ev.n_residuals();
    let np = th.len();
    let mut r = vec![0.0; m];
    let mut j = vec![0.0; m * np];
    assert!(ev
        .residual_and_jacobian(e, th, &mut r, Some(&mut j))
        .unwrap());
    let h = 1e-6;
    for k in 0..np {
        let mut tp = th.to_vec();
        let mut tm = th.to_vec();
        tp[k] += h;
        tm[k] -= h;
        let mut rp = vec![0.0; m];
        let mut rm = vec![0.0; m];
        ev.residual_and_jacobian(e, &tp, &mut rp, None).unwrap();
        ev.residual_and_jacobian(e, &tm, &mut rm, None).unwrap();
        for row in 0..m {
            let fd = (rp[row] - rm[row]) / (2.0 * h);
            assert!(
                (fd - j[row * np + k]).abs() < tol * (1.0 + fd.abs()),
                "row {row} param {k}: fd {fd} vs J {}",
                j[row * np + k]
            );
        }
    }
    let loss = ev.loss(e, th).unwrap();
    assert!(close(loss, r.iter().map(|v| v * v).sum(), 1e-12));
    let mut g = vec![0.0; np];
    ev.loss_and_grad(e, th, &mut g).unwrap();
    for k in 0..np {
        let want: f64 = (0..m).map(|row| 2.0 * r[row] * j[row * np + k]).sum();
        assert!(close(g[k], want, 1e-10), "grad {k}: {} vs {want}", g[k]);
    }
}

#[test]
fn nonlocal_terms_match_direct_evaluation_and_their_jacobian() {
    // residual = −Δu(x) + 0.7·u(x + m) + 0.2·Δu(x + m) − f, against u and Δu
    // of the same candidate evaluated directly at the shifted points.
    let mut rng = Rng(7);
    let d = 3;
    let n = 20;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let shift = vec![0.3, -0.2, 0.1];
    let mk = || {
        Problem::new(
            d,
            PdeOperator::poisson(),
            ScalarField::Const(0.5),
            ScalarField::Zero,
        )
        .unwrap()
    };
    let prob = mk()
        .with_nonlocal(vec![NonlocalTerm::shifted(shift.clone(), 0.7, 0.2)])
        .unwrap();
    let local = mk();
    let xi = grid(&mut rng, n, d, -1.0, 1.0);
    let xb = grid(&mut rng, 6, d, -1.0, 1.0);
    let ev = Evaluator::new(&t, &prob, &xi, &xb).unwrap();
    let ev0 = Evaluator::new(&t, &local, &xi, &xb).unwrap();
    let ys: Vec<f64> = (0..n * d).map(|k| xi[k] + shift[k % d]).collect();
    for e in [[7usize, 1, 3, 8], [6, 0, 6, 3], [3, 0, 7, 2]] {
        let th: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.5, 0.5)).collect();
        let m = ev.n_residuals();
        let mut r = vec![0.0; m];
        let mut r0 = vec![0.0; m];
        ev.residual_and_jacobian(&e, &th, &mut r, None).unwrap();
        ev0.residual_and_jacobian(&e, &th, &mut r0, None).unwrap();
        let jets = ev.jets_at(&e, &th, &ys).unwrap();
        let w = 1.0 / (n as f64).sqrt();
        for p in 0..n {
            let want = r0[p] + w * (0.7 * jets.u[p] + 0.2 * jets.lap[p]);
            assert!(close(r[p], want, 1e-12), "row {p}: {} vs {want}", r[p]);
        }
        for p in n..m {
            assert_eq!(r[p], r0[p], "boundary rows are local");
        }
        check_jacobian_fd(&ev, &e, &th, 1e-6);
    }
}

#[test]
fn second_order_form_is_the_trace_against_the_hessian() {
    // Tr(A ∇²u) with a banded A, against finite differences of u.
    let mut rng = Rng(11);
    let d = 3;
    let n = 12;
    let t = Template::new("depth2_sub", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let form = SecondOrderForm {
        diag: vec![0.5, 1.0, 1.5],
        off: vec![(0, 1, 0.25), (1, 2, -0.4)],
    };
    let prob = Problem::new(
        d,
        PdeOperator::Linear(LinearOperator {
            lap: 1.0,
            grad: vec![],
            u: 0.0,
            cubic: 0.0,
            potential: false,
        }),
        ScalarField::Zero,
        ScalarField::Zero,
    )
    .unwrap()
    .with_second_order(form.clone())
    .unwrap();
    let xi = grid(&mut rng, n, d, -1.0, 1.0);
    let ev = Evaluator::new(&t, &prob, &xi, &[]).unwrap();
    let e = [7usize, 1, 3, 8];
    let th: Vec<f64> = (0..t.n_params()).map(|_| rng.unif(-0.5, 0.5)).collect();
    let mut r = vec![0.0; n];
    ev.residual_and_jacobian(&e, &th, &mut r, None).unwrap();
    let a_at = |i: usize, j: usize| -> f64 {
        if i == j {
            return form.diag[i];
        }
        form.off
            .iter()
            .find(|&&(a, b, _)| (a, b) == (i.min(j), i.max(j)))
            .map(|&(_, _, c)| c)
            .unwrap_or(0.0)
    };
    let h = 1e-4;
    let u = |x: &[f64]| ev.jets_at(&e, &th, x).unwrap().u[0];
    for p in 0..n {
        let x = &xi[p * d..(p + 1) * d];
        let mut tr = 0.0;
        for i in 0..d {
            for j in 0..d {
                let a = a_at(i, j);
                if a == 0.0 {
                    continue;
                }
                let at = |si: f64, sj: f64| {
                    let mut y = x.to_vec();
                    y[i] += si * h;
                    y[j] += sj * h;
                    u(&y)
                };
                let hij =
                    (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0)) / (4.0 * h * h);
                tr += a * hij;
            }
        }
        let got = r[p] * (n as f64).sqrt();
        assert!(
            (got - tr).abs() < 1e-5 * (1.0 + tr.abs()),
            "point {p}: {got} vs {tr}"
        );
    }
    check_jacobian_fd(&ev, &e, &th, 1e-6);
}

#[test]
fn nonlocal_problems_are_checked() {
    let d = 2;
    let base = || {
        Problem::new(
            d,
            PdeOperator::poisson(),
            ScalarField::Zero,
            ScalarField::Zero,
        )
        .unwrap()
    };
    let bad = base().with_nonlocal(vec![NonlocalTerm::shifted(vec![0.1], 1.0, 0.0)]);
    assert_eq!(bad.unwrap_err().code(), "E-FEX-015");
    let bad = base().with_second_order(SecondOrderForm {
        diag: vec![1.0, 1.0],
        off: vec![(1, 0, 0.5)],
    });
    assert_eq!(bad.unwrap_err().code(), "E-FEX-015");
    let t = Template::new("depth1", &PAPER_UNARY, &PAPER_BINARY, d).unwrap();
    let p = base()
        .with_loss(LossKind::Rayleigh)
        .with_nonlocal(vec![NonlocalTerm::shifted(vec![0.1, 0.1], 1.0, 0.0)])
        .unwrap();
    let err = Evaluator::new(&t, &p, &[0.1, 0.2], &[])
        .err()
        .expect("refused");
    assert_eq!(err.code(), "E-FEX-015");
    let spec = crate::errors::codes::REGISTRY
        .iter()
        .find(|s| s.code == "E-FEX-015")
        .expect("registered");
    assert_eq!(spec.remediation, err.remediation());
}
