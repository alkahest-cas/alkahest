use super::*;
use crate::jit::compile_many;
use crate::kernel::{Domain, ExprPool};

/// 1-D heat equation `u_t = u_xx` on `[0, 1]` with `n` interior points and
/// Dirichlet zeros at both ends (boundary nodes kept as constant padding),
/// centred second differences.
fn heat_program(n: usize, with_jac: bool) -> MolProgram {
    let pool = ExprPool::new();
    let (um, u0, up) = (
        pool.symbol("um", Domain::Real),
        pool.symbol("u0", Domain::Real),
        pool.symbol("up", Domain::Real),
    );
    let h = 1.0 / (n as f64 + 1.0);
    let c = pool.float(1.0 / (h * h), 53);
    let two = pool.integer(-2_i32);
    let lap = pool.mul(vec![c, pool.add(vec![um, up, pool.mul(vec![two, u0])])]);
    let k = compile_many(&[lap], &[um, u0, up], &pool).unwrap();
    let jk = compile_many(
        &[c, pool.mul(vec![c, pool.integer(-2_i32)]), c],
        &[um, u0, up],
        &pool,
    )
    .unwrap();
    let field = |o: isize| MolInput::Field {
        field: 0,
        offset: vec![o],
    };
    let op = MolKernelOp {
        kernel: Arc::new(k),
        inputs: vec![field(-1), field(0), field(1)],
        target: MolTarget::Rhs(0),
        region: vec![(1, n + 1)],
        jac_kernel: with_jac.then(|| Arc::new(jk)),
        jac_inputs: if with_jac { vec![0, 1, 2] } else { vec![] },
    };
    let states = (1..=n).map(|i| (0, i)).collect();
    MolProgram::new(
        vec![n + 2],
        1,
        vec![],
        states,
        vec![false; n],
        vec![MolOp::Kernel(op)],
    )
    .unwrap()
}

#[test]
fn heat_rhs_matches_second_difference() {
    let n = 9;
    let p = heat_program(n, true);
    let h = 0.1;
    let y: Vec<f64> = (1..=n).map(|i| ((i as f64) * h).powi(2)).collect();
    let mut out = vec![0.0; n];
    p.rhs(0.0, &y, &mut out).unwrap();
    // u = x^2 has u_xx = 2 in the interior; the last point sees u(1) = 0.
    for v in &out[..n - 1] {
        assert!((v - 2.0).abs() < 1e-9, "{v}");
    }
}

#[test]
fn jacobian_is_tridiagonal_and_exact() {
    let n = 6;
    let p = heat_program(n, true);
    let y = vec![0.3; n];
    let j = p.jacobian(0.0, &y).unwrap();
    assert_eq!(j.indptr.len(), n + 1);
    assert_eq!(j.indices.len(), 3 * n - 2);
    // Compare with finite differences of the rhs (linear: exact).
    let mut f0 = vec![0.0; n];
    p.rhs(0.0, &y, &mut f0).unwrap();
    for col in 0..n {
        let mut y1 = y.clone();
        y1[col] += 1.0;
        let mut f1 = vec![0.0; n];
        p.rhs(0.0, &y1, &mut f1).unwrap();
        for row in 0..n {
            let fd = f1[row] - f0[row];
            let seg = &j.indices[j.indptr[row]..j.indptr[row + 1]];
            let v = seg
                .iter()
                .position(|&c| c == col)
                .map_or(0.0, |k| j.data[j.indptr[row] + k]);
            assert!(
                (fd - v).abs() < 1e-6 * (1.0 + v.abs()),
                "({row},{col}) {fd} vs {v}"
            );
        }
    }
}

#[test]
fn rejects_uncovered_states() {
    let pool = ExprPool::new();
    let u = pool.symbol("u", Domain::Real);
    let k = compile_many(&[u], &[u], &pool).unwrap();
    let op = MolKernelOp {
        kernel: Arc::new(k),
        inputs: vec![MolInput::Field {
            field: 0,
            offset: vec![0],
        }],
        target: MolTarget::Rhs(0),
        region: vec![(0, 2)],
        jac_kernel: None,
        jac_inputs: vec![],
    };
    let r = MolProgram::new(
        vec![3],
        1,
        vec![],
        vec![(0, 0), (0, 1), (0, 2)],
        vec![false; 3],
        vec![MolOp::Kernel(op)],
    );
    assert!(matches!(r, Err(MolError::InvalidProgram(_))));
}

#[test]
fn banded_lu_solves_periodic_tridiagonal() {
    // Cyclic tridiagonal: RCM must keep the band narrow.
    let n = 50;
    let mut indptr = vec![0];
    let mut indices = vec![];
    let mut data = vec![];
    for r in 0..n {
        let mut row = vec![((r + n - 1) % n, -1.0), (r, 4.0), ((r + 1) % n, -1.5)];
        row.sort_by_key(|e| e.0);
        for (c, v) in row {
            indices.push(c);
            data.push(v);
        }
        indptr.push(indices.len());
    }
    let a = CsrMatrix {
        n_rows: n,
        n_cols: n,
        indptr,
        indices,
        data,
    };
    let perm = rcm_order(&a);
    let (kl, ku) = BandedLu::bandwidths(&a, &perm);
    assert!(kl <= 3 && ku <= 3, "bandwidths {kl}, {ku}");
    let lu = BandedLu::factor(&a, &perm, kl, ku).unwrap();
    let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let mut b = vec![0.0; n];
    a.matvec(&x, &mut b);
    lu.solve(&mut b);
    for (u, v) in b.iter().zip(&x) {
        assert!((u - v).abs() < 1e-12);
    }
}

fn exact_heat(n: usize, t: f64) -> Vec<f64> {
    let h = 1.0 / (n as f64 + 1.0);
    // Semi-discrete exact: sin(pi x) decays with the discrete eigenvalue.
    let lam = -4.0 / (h * h) * (std::f64::consts::PI * h / 2.0).sin().powi(2);
    (1..=n)
        .map(|i| (std::f64::consts::PI * i as f64 * h).sin() * (lam * t).exp())
        .collect()
}

#[test]
fn integrators_reproduce_the_semi_discrete_solution() {
    let n = 30;
    let p = heat_program(n, true);
    let y0 = exact_heat(n, 0.0);
    let t_eval = [0.05, 0.1];
    let ex = exact_heat(n, 0.1);
    for method in [MolMethod::Rk45, MolMethod::Bdf] {
        let opts = MolIntegrateOptions {
            method,
            rtol: 1e-8,
            atol: 1e-10,
            ..Default::default()
        };
        let sol = integrate(&p, 0.0, &y0, &t_eval, &opts).unwrap();
        assert_eq!(sol.t, t_eval.to_vec());
        let y = &sol.y[n..2 * n];
        let err = y
            .iter()
            .zip(&ex)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-6, "{method:?}: error {err}");
        if method == MolMethod::Bdf {
            // Stiff: far fewer steps than the explicit method needs.
            assert!(sol.stats.steps < 400, "{:?}", sol.stats);
        }
    }
    let opts = MolIntegrateOptions {
        method: MolMethod::Rk4,
        h0: Some(1e-4),
        ..Default::default()
    };
    let sol = integrate(&p, 0.0, &y0, &[0.1], &opts).unwrap();
    let err = sol
        .y
        .iter()
        .zip(&ex)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-8, "rk4 error {err}");
}

#[test]
fn bdf_integrates_an_index_one_dae() {
    // y' = -y + z, 0 = z - 2y  (z algebraic) => y' = y.  Build it as a
    // two-field program on a 1-point grid.
    let pool = ExprPool::new();
    let (y, z) = (
        pool.symbol("y", Domain::Real),
        pool.symbol("z", Domain::Real),
    );
    let fy = pool.add(vec![pool.mul(vec![pool.integer(-1_i32), y]), z]);
    let fz = pool.add(vec![z, pool.mul(vec![pool.integer(-2_i32), y])]);
    let mk = |e, jac: Vec<ExprId>| MolKernelOp {
        kernel: Arc::new(compile_many(&[e], &[y, z], &pool).unwrap()),
        inputs: vec![
            MolInput::Field {
                field: 0,
                offset: vec![0],
            },
            MolInput::Field {
                field: 1,
                offset: vec![0],
            },
        ],
        target: MolTarget::Rhs(0),
        region: vec![(0, 1)],
        jac_kernel: Some(Arc::new(compile_many(&jac, &[y, z], &pool).unwrap())),
        jac_inputs: vec![0, 1],
    };
    let ky = mk(fy, vec![pool.integer(-1_i32), pool.integer(1_i32)]);
    let mut kz = mk(fz, vec![pool.integer(-2_i32), pool.integer(1_i32)]);
    kz.target = MolTarget::Rhs(1);
    let p = MolProgram::new(
        vec![1],
        2,
        vec![],
        vec![(0, 0), (1, 0)],
        vec![false, true],
        vec![MolOp::Kernel(ky), MolOp::Kernel(kz)],
    )
    .unwrap();
    let opts = MolIntegrateOptions {
        method: MolMethod::Bdf,
        rtol: 1e-9,
        atol: 1e-12,
        ..Default::default()
    };
    // z0 inconsistent on purpose: the integrator fixes it.
    let sol = integrate(&p, 0.0, &[1.0, 0.0], &[1.0], &opts).unwrap();
    assert!((sol.y[0] - 1f64.exp()).abs() < 1e-6, "{}", sol.y[0]);
    assert!((sol.y[1] - 2.0 * 1f64.exp()).abs() < 1e-5);
    let r = integrate(
        &p,
        0.0,
        &[1.0, 0.0],
        &[1.0],
        &MolIntegrateOptions::default(),
    );
    assert!(matches!(r, Err(MolError::BadArgument(_))));
}

use crate::kernel::ExprId;

/// `u_t = Δu + r u (1 - u)` on `[0, 1]²` with `m × m` interior points,
/// Dirichlet zeros (constant padding), the 5-point Laplacian.
fn heat2d_program(m: usize, r: f64) -> MolProgram {
    let pool = ExprPool::new();
    let names = ["c", "w", "e", "s", "n"];
    let s: Vec<ExprId> = names
        .iter()
        .map(|nm| pool.symbol(*nm, Domain::Real))
        .collect();
    let h = 1.0 / (m as f64 + 1.0);
    let c = pool.float(1.0 / (h * h), 53);
    let lap = pool.mul(vec![
        c,
        pool.add(vec![
            s[1],
            s[2],
            s[3],
            s[4],
            pool.mul(vec![pool.integer(-4_i32), s[0]]),
        ]),
    ]);
    let rr = pool.float(r, 53);
    let one = pool.integer(1_i32);
    let reac = pool.mul(vec![
        rr,
        s[0],
        pool.add(vec![one, pool.mul(vec![pool.integer(-1_i32), s[0]])]),
    ]);
    let f = pool.add(vec![lap, reac]);
    // ∂f/∂c = -4/h² + r (1 - 2c); the neighbours: 1/h².
    let dfc = pool.add(vec![
        pool.mul(vec![c, pool.integer(-4_i32)]),
        pool.mul(vec![
            rr,
            pool.add(vec![one, pool.mul(vec![pool.integer(-2_i32), s[0]])]),
        ]),
    ]);
    let k = compile_many(&[f], &s, &pool).unwrap();
    let jk = compile_many(&[dfc, c, c, c, c], &s, &pool).unwrap();
    let offs: [[isize; 2]; 5] = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]];
    let inputs = offs
        .iter()
        .map(|o| MolInput::Field {
            field: 0,
            offset: o.to_vec(),
        })
        .collect();
    let op = MolKernelOp {
        kernel: Arc::new(k),
        inputs,
        target: MolTarget::Rhs(0),
        region: vec![(1, m + 1), (1, m + 1)],
        jac_kernel: Some(Arc::new(jk)),
        jac_inputs: vec![0, 1, 2, 3, 4],
    };
    let w = m + 2;
    let states = (1..=m)
        .flat_map(|i| (1..=m).map(move |j| (0, i * w + j)))
        .collect();
    MolProgram::new(
        vec![w, w],
        1,
        vec![],
        states,
        vec![false; m * m],
        vec![MolOp::Kernel(op)],
    )
    .unwrap()
}

fn sine2d(m: usize) -> Vec<f64> {
    let h = 1.0 / (m as f64 + 1.0);
    let pi = std::f64::consts::PI;
    (1..=m)
        .flat_map(|i| (1..=m).map(move |j| (pi * i as f64 * h).sin() * (pi * j as f64 * h).sin()))
        .collect()
}

fn bdf_with(
    p: &MolProgram,
    y0: &[f64],
    t: f64,
    lin: MolLinearSolver,
) -> (MolSolution, MolLinearStats) {
    let opts = MolIntegrateOptions {
        method: MolMethod::Bdf,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };
    let so = MolSolverOptions {
        linear_solver: lin,
        ..Default::default()
    };
    integrate_with(p, 0.0, y0, &[t], &opts, &so).unwrap()
}

#[test]
fn krylov_bdf_matches_the_direct_solver_in_2d() {
    let m = 24;
    let p = heat2d_program(m, 3.0);
    let y0 = sine2d(m);
    let (d, ld) = bdf_with(&p, &y0, 0.05, MolLinearSolver::Direct);
    let (k, lk) = bdf_with(&p, &y0, 0.05, MolLinearSolver::Krylov);
    assert_eq!(ld.solver, Some(MolLinearSolver::Direct));
    assert_eq!(lk.solver, Some(MolLinearSolver::Krylov));
    assert_eq!(ld.krylov_iters, 0);
    assert!(
        lk.krylov_iters > 0 && lk.preconditioner_setups > 0,
        "{lk:?}"
    );
    assert_eq!(lk.direct_fallbacks, 0, "{lk:?}");
    let err =
        d.y.iter()
            .zip(&k.y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
    // Both solve the same nonlinear systems to the Newton tolerance.
    assert!(err < 1e-6, "direct vs Krylov differ by {err}");
    assert!(k.y.iter().any(|v| v.abs() > 0.1));
}

#[test]
fn auto_picks_the_direct_solver_for_small_bands_and_krylov_for_wide_ones() {
    let p = heat2d_program(10, 0.0);
    let (_, l) = bdf_with(&p, &sine2d(10), 0.01, MolLinearSolver::Auto);
    assert_eq!(l.solver, Some(MolLinearSolver::Direct));
    // Lower the thresholds: the same problem goes to GMRES.
    let opts = MolIntegrateOptions {
        method: MolMethod::Bdf,
        ..Default::default()
    };
    let so = MolSolverOptions {
        auto_min_states: 50,
        auto_min_bandwidth: 5,
        ..Default::default()
    };
    let (_, l) = integrate_with(&p, 0.0, &sine2d(10), &[0.01], &opts, &so).unwrap();
    assert_eq!(l.solver, Some(MolLinearSolver::Krylov));
    // A band too large for `max_band_bytes`: Auto switches to GMRES
    // instead of refusing; Direct refuses.
    let small = MolIntegrateOptions {
        method: MolMethod::Bdf,
        max_band_bytes: 1024,
        ..Default::default()
    };
    let (_, l) = integrate_with(
        &p,
        0.0,
        &sine2d(10),
        &[0.01],
        &small,
        &MolSolverOptions::default(),
    )
    .unwrap();
    assert_eq!(l.solver, Some(MolLinearSolver::Krylov));
    let so = MolSolverOptions {
        linear_solver: MolLinearSolver::Direct,
        ..Default::default()
    };
    let r = integrate_with(&p, 0.0, &sine2d(10), &[0.01], &small, &so);
    assert!(matches!(r, Err(MolError::Integration(_))));
}

#[test]
fn krylov_bdf_integrates_an_index_one_dae() {
    // The DAE of `bdf_integrates_an_index_one_dae`, with GMRES + ILU(0).
    let pool = ExprPool::new();
    let (y, z) = (
        pool.symbol("y", Domain::Real),
        pool.symbol("z", Domain::Real),
    );
    let fy = pool.add(vec![pool.mul(vec![pool.integer(-1_i32), y]), z]);
    let fz = pool.add(vec![z, pool.mul(vec![pool.integer(-2_i32), y])]);
    let mk = |e, jac: Vec<ExprId>, tgt| MolKernelOp {
        kernel: Arc::new(compile_many(&[e], &[y, z], &pool).unwrap()),
        inputs: vec![
            MolInput::Field {
                field: 0,
                offset: vec![0],
            },
            MolInput::Field {
                field: 1,
                offset: vec![0],
            },
        ],
        target: tgt,
        region: vec![(0, 1)],
        jac_kernel: Some(Arc::new(compile_many(&jac, &[y, z], &pool).unwrap())),
        jac_inputs: vec![0, 1],
    };
    let ky = mk(
        fy,
        vec![pool.integer(-1_i32), pool.integer(1_i32)],
        MolTarget::Rhs(0),
    );
    let kz = mk(
        fz,
        vec![pool.integer(-2_i32), pool.integer(1_i32)],
        MolTarget::Rhs(1),
    );
    let p = MolProgram::new(
        vec![1],
        2,
        vec![],
        vec![(0, 0), (1, 0)],
        vec![false, true],
        vec![MolOp::Kernel(ky), MolOp::Kernel(kz)],
    )
    .unwrap();
    let opts = MolIntegrateOptions {
        method: MolMethod::Bdf,
        rtol: 1e-9,
        atol: 1e-12,
        ..Default::default()
    };
    let so = MolSolverOptions {
        linear_solver: MolLinearSolver::Krylov,
        ..Default::default()
    };
    let (sol, l) = integrate_with(&p, 0.0, &[1.0, 0.0], &[1.0], &opts, &so).unwrap();
    assert_eq!(l.solver, Some(MolLinearSolver::Krylov));
    assert!((sol.y[0] - 1f64.exp()).abs() < 1e-6, "{}", sol.y[0]);
    assert!((sol.y[1] - 2.0 * 1f64.exp()).abs() < 1e-5);
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn rhs_and_jacobian_are_bitwise_identical_across_thread_counts() {
    // Long enough rows that the kernels and the Jacobian assembly split
    // into many parallel chunks.
    let n = 40_000;
    let p = heat_program(n, true);
    let y: Vec<f64> = (0..n).map(|i| ((i * 37 % 101) as f64).sin()).collect();
    let run = |threads: usize| {
        with_threads(threads, || {
            let mut out = vec![0.0; n];
            p.rhs(0.3, &y, &mut out).unwrap();
            let j = p.jacobian(0.3, &y).unwrap();
            (out, j)
        })
    };
    let (r1, j1) = run(1);
    for threads in [2, 3, 8] {
        let (r, j) = run(threads);
        assert_eq!(bits(&r), bits(&r1), "rhs differs with {threads} threads");
        assert_eq!(j.indptr, j1.indptr);
        assert_eq!(j.indices, j1.indices);
        assert_eq!(bits(&j.data), bits(&j1.data), "{threads} threads");
    }
    // The 2-D program too (the span path).
    let m = 140;
    let p2 = heat2d_program(m, 2.0);
    let y2 = sine2d(m);
    let run2 = |threads: usize| {
        with_threads(threads, || {
            let mut out = vec![0.0; m * m];
            p2.rhs(0.0, &y2, &mut out).unwrap();
            (out, p2.jacobian(0.0, &y2).unwrap())
        })
    };
    let (a1, b1) = run2(1);
    let (a4, b4) = run2(4);
    assert_eq!(bits(&a1), bits(&a4));
    assert_eq!(bits(&b1.data), bits(&b4.data));
}

#[test]
fn krylov_integration_is_bitwise_identical_across_thread_counts() {
    // 130² states: the kernels and the Jacobian assembly run chunked (the
    // GMRES reductions go parallel only on larger systems; see krylov.rs).
    let m = 130;
    let p = heat2d_program(m, 1.0);
    let y0 = sine2d(m);
    let opts = MolIntegrateOptions {
        method: MolMethod::Bdf,
        rtol: 1e-6,
        atol: 1e-9,
        ..Default::default()
    };
    let so = MolSolverOptions {
        linear_solver: MolLinearSolver::Krylov,
        ..Default::default()
    };
    let run = |threads: usize| {
        with_threads(threads, || {
            integrate_with(&p, 0.0, &y0, &[2e-3], &opts, &so).unwrap()
        })
    };
    let (s1, l1) = run(1);
    let (s4, l4) = run(4);
    assert_eq!(bits(&s1.y), bits(&s4.y));
    assert_eq!(s1.stats, s4.stats);
    assert_eq!(l1, l4);
    assert!(l1.krylov_iters > 0);
}

/// `y' = 1`, `0 = s (z - y)`: with `s = 1e-12` the algebraic row's residual
/// is tiny however wrong `z` is, so a residual-based GMRES test would accept
/// a zero correction and leave `z` at its predictor. The left-preconditioned
/// test bounds the error instead: both solvers end at `z = y`.
fn scaled_dae(s: f64) -> MolProgram {
    let pool = ExprPool::new();
    let (y, z) = (
        pool.symbol("y", Domain::Real),
        pool.symbol("z", Domain::Real),
    );
    let sc = pool.float(s, 53);
    let fz = pool.mul(vec![
        sc,
        pool.add(vec![z, pool.mul(vec![pool.integer(-1_i32), y])]),
    ]);
    let inputs = vec![
        MolInput::Field {
            field: 0,
            offset: vec![0],
        },
        MolInput::Field {
            field: 1,
            offset: vec![0],
        },
    ];
    let ky = MolKernelOp {
        kernel: Arc::new(compile_many(&[pool.integer(1_i32)], &[y, z], &pool).unwrap()),
        inputs: inputs.clone(),
        target: MolTarget::Rhs(0),
        region: vec![(0, 1)],
        jac_kernel: None,
        jac_inputs: vec![],
    };
    let kz = MolKernelOp {
        kernel: Arc::new(compile_many(&[fz], &[y, z], &pool).unwrap()),
        inputs,
        target: MolTarget::Rhs(1),
        region: vec![(0, 1)],
        jac_kernel: Some(Arc::new(
            compile_many(
                &[pool.mul(vec![pool.integer(-1_i32), sc]), sc],
                &[y, z],
                &pool,
            )
            .unwrap(),
        )),
        jac_inputs: vec![0, 1],
    };
    MolProgram::new(
        vec![1],
        2,
        vec![],
        vec![(0, 0), (1, 0)],
        vec![false, true],
        vec![MolOp::Kernel(ky), MolOp::Kernel(kz)],
    )
    .unwrap()
}

#[test]
fn krylov_enforces_badly_scaled_algebraic_rows() {
    for s in [1.0, 1e-12] {
        let p = scaled_dae(s);
        let opts = MolIntegrateOptions {
            method: MolMethod::Bdf,
            h0: Some(0.1),
            ..Default::default()
        };
        for lin in [MolLinearSolver::Direct, MolLinearSolver::Krylov] {
            let so = MolSolverOptions {
                linear_solver: lin,
                ..Default::default()
            };
            let (sol, _) = integrate_with(&p, 0.0, &[1.0, 1.0], &[1.0], &opts, &so).unwrap();
            assert!(
                (sol.y[0] - 2.0).abs() < 1e-8,
                "{s} {lin:?}: y = {}",
                sol.y[0]
            );
            assert!(
                (sol.y[1] - sol.y[0]).abs() < 1e-6,
                "{s} {lin:?}: z = {} but y = {}",
                sol.y[1],
                sol.y[0]
            );
        }
    }
}

#[test]
fn invalid_solver_options_are_refused() {
    let p = heat_program(8, true);
    let y0 = exact_heat(8, 0.0);
    let opts = MolIntegrateOptions {
        method: MolMethod::Bdf,
        ..Default::default()
    };
    for so in [
        MolSolverOptions {
            krylov_tol_factor: f64::INFINITY,
            ..Default::default()
        },
        MolSolverOptions {
            krylov_tol_factor: 0.0,
            ..Default::default()
        },
        MolSolverOptions {
            gmres_restart: 0,
            ..Default::default()
        },
        MolSolverOptions {
            gmres_max_iters: 0,
            ..Default::default()
        },
    ] {
        let r = integrate_with(&p, 0.0, &y0, &[0.1], &opts, &so);
        assert!(matches!(r, Err(MolError::BadArgument(_))), "{so:?}");
    }
}
