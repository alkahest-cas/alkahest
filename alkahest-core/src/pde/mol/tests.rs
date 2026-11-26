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
