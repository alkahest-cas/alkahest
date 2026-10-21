use super::*;
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprPool};

fn sym(p: &ExprPool, s: &str) -> ExprId {
    p.symbol(s, Domain::Real)
}

fn int(p: &ExprPool, n: i64) -> ExprId {
    p.integer(n)
}

fn pw(p: &ExprPool, b: ExprId, k: i64) -> ExprId {
    p.pow(b, p.integer(k))
}

fn sub(p: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
    p.add(vec![a, p.mul(vec![p.integer(-1), b])])
}

fn div(p: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
    p.mul(vec![a, pw(p, b, -1)])
}

/// `e` is identically zero (exact canonical form).
fn zero(p: &ExprPool, e: ExprId, coords: &[ExprId]) -> bool {
    let c = canonical_form(&[e], coords, p).unwrap()[0];
    matches!(p.get(c), ExprData::Integer(ref z) if z.0 == 0)
}

fn same(p: &ExprPool, a: ExprId, b: ExprId, coords: &[ExprId]) -> bool {
    zero(p, sub(p, a, b), coords)
}

fn diag(p: &ExprPool, d: &[ExprId]) -> Vec<Vec<ExprId>> {
    let n = d.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { d[i] } else { int(p, 0) })
                .collect()
        })
        .collect()
}

struct Schw {
    p: ExprPool,
    x: Vec<ExprId>,
    m: ExprId,
}

fn schwarzschild() -> (Schw, Metric) {
    let p = ExprPool::new();
    let x: Vec<ExprId> = ["t", "r", "theta", "phi"]
        .iter()
        .map(|s| sym(&p, s))
        .collect();
    let m = sym(&p, "M");
    let (r, th) = (x[1], x[2]);
    // f = 1 − 2M/r
    let f = sub(&p, int(&p, 1), div(&p, p.mul(vec![int(&p, 2), m]), r));
    let sin2 = pw(&p, p.func("sin", vec![th]), 2);
    let g = diag(
        &p,
        &[
            p.mul(vec![int(&p, -1), f]),
            pw(&p, f, -1),
            pw(&p, r, 2),
            p.mul(vec![pw(&p, r, 2), sin2]),
        ],
    );
    let metric = Metric::new(&g, &x, &p).unwrap();
    (Schw { p, x, m }, metric)
}

#[test]
fn canonical_form_proves_trig_identities() {
    let p = ExprPool::new();
    let th = sym(&p, "theta");
    let x = sym(&p, "x");
    let s = p.func("sin", vec![th]);
    let c = p.func("cos", vec![th]);
    // sin² + cos² − 1
    let e = p.add(vec![pw(&p, s, 2), pw(&p, c, 2), int(&p, -1)]);
    assert!(zero(&p, e, &[th]));
    // sin(2θ) − 2 sin θ cos θ
    let s2 = p.func("sin", vec![p.mul(vec![int(&p, 2), th])]);
    let e = sub(&p, s2, p.mul(vec![int(&p, 2), s, c]));
    assert!(zero(&p, e, &[th]));
    // tan θ cos θ − sin θ
    let e = sub(&p, p.mul(vec![p.func("tan", vec![th]), c]), s);
    assert!(zero(&p, e, &[th]));
    // cosh² − sinh² − 1 and exp(x)·exp(−x) − 1
    let e = p.add(vec![
        pw(&p, p.func("cosh", vec![x]), 2),
        p.mul(vec![int(&p, -1), pw(&p, p.func("sinh", vec![x]), 2)]),
        int(&p, -1),
    ]);
    assert!(zero(&p, e, &[x]));
    let e = sub(
        &p,
        p.mul(vec![
            p.func("exp", vec![x]),
            p.func("exp", vec![p.mul(vec![int(&p, -1), x])]),
        ]),
        int(&p, 1),
    );
    assert!(zero(&p, e, &[x]));
    // (√x)² − x and 1/(1 + √x) − (1 − √x)/(1 − x)
    let w = p.pow(x, p.rational(1, 2));
    assert!(zero(&p, sub(&p, pw(&p, w, 2), x), &[x]));
    let lhs = pw(&p, p.add(vec![int(&p, 1), w]), -1);
    let rhs = div(&p, sub(&p, int(&p, 1), w), sub(&p, int(&p, 1), x));
    assert!(same(&p, lhs, rhs, &[x]));
    // a genuinely non-zero expression is not reported as zero
    assert!(!zero(&p, sub(&p, pw(&p, s, 2), pw(&p, c, 2)), &[th]));
}

#[test]
fn canonical_form_folds_sine_powers_back() {
    let p = ExprPool::new();
    let th = sym(&p, "theta");
    // cos θ / sin θ is printed as such, not as cos·sin/(1 − cos²)
    let e = div(&p, p.func("cos", vec![th]), p.func("sin", vec![th]));
    let c = canonical_form(&[e], &[th], &p).unwrap()[0];
    let shown = p.display(c).to_string();
    assert!(shown.contains("sin") && shown.contains("cos"), "{shown}");
    assert!(!shown.contains('+'), "{shown}");
}

#[test]
fn schwarzschild_christoffel_ricci_kretschmann() {
    let (s, mut m) = schwarzschild();
    let p = &s.p;
    let (r, th) = (s.x[1], s.x[2]);
    let gam = m.christoffel(p).unwrap();
    // Γ^r_tt = M (r − 2M) / r³
    let two_m = p.mul(vec![int(p, 2), s.m]);
    let exp_rtt = div(p, p.mul(vec![s.m, sub(p, r, two_m)]), pw(p, r, 3));
    assert!(same(p, gam[1][0][0], exp_rtt, &s.x));
    // Γ^t_tr = M / (r (r − 2M))
    let exp_ttr = div(p, s.m, p.mul(vec![r, sub(p, r, two_m)]));
    assert!(same(p, gam[0][0][1], exp_ttr, &s.x));
    assert!(same(p, gam[0][1][0], exp_ttr, &s.x));
    // Γ^θ_φφ = −sin θ cos θ,  Γ^φ_θφ = cot θ
    let exp_thpp = p.mul(vec![
        int(p, -1),
        p.func("sin", vec![th]),
        p.func("cos", vec![th]),
    ]);
    assert!(same(p, gam[2][3][3], exp_thpp, &s.x));
    let cot = div(p, p.func("cos", vec![th]), p.func("sin", vec![th]));
    assert!(same(p, gam[3][2][3], cot, &s.x));
    // Γ^r_φφ = −(r − 2M) sin² θ
    let exp_rpp = p.mul(vec![
        int(p, -1),
        sub(p, r, two_m),
        pw(p, p.func("sin", vec![th]), 2),
    ]);
    assert!(same(p, gam[1][3][3], exp_rpp, &s.x));
    // Vacuum: R_ab = 0, R = 0, G_ab = 0.
    assert!(m.is_ricci_flat().unwrap());
    for row in m.ricci(p).unwrap() {
        for e in row {
            assert!(matches!(p.get(e), ExprData::Integer(ref z) if z.0 == 0));
        }
    }
    assert!(zero(p, m.ricci_scalar(p).unwrap(), &s.x));
    // K = 48 M² / r⁶
    let k = m.kretschmann(p).unwrap();
    let exp_k = div(p, p.mul(vec![int(p, 48), pw(p, s.m, 2)]), pw(p, r, 6));
    assert!(same(p, k, exp_k, &s.x));
    // R^t_{rtr} = 2M / (r²(r − 2M))  (the one radial tidal component)
    let riem = m.riemann(p).unwrap();
    let exp_r = div(
        p,
        p.mul(vec![int(p, 2), s.m]),
        p.mul(vec![pw(p, r, 2), sub(p, r, two_m)]),
    );
    assert!(same(p, riem[0][1][0][1], exp_r, &s.x));
    // Ricci-flat ⇒ Weyl = Riemann.
    let w = m.weyl(p).unwrap();
    let rl = m.riemann_lower(p).unwrap();
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    assert!(same(p, w[a][b][c][d], rl[a][b][c][d], &s.x));
                }
            }
        }
    }
    let names: Vec<&str> = m.timings().iter().map(|t| t.0).collect();
    assert!(names.contains(&"riemann"));
}

#[test]
fn round_sphere_and_de_sitter_have_constant_curvature() {
    let p = ExprPool::new();
    let th = sym(&p, "theta");
    let ph = sym(&p, "phi");
    let rho = sym(&p, "rho");
    let r2 = pw(&p, rho, 2);
    let sin2 = pw(&p, p.func("sin", vec![th]), 2);
    let g = diag(&p, &[r2, p.mul(vec![r2, sin2])]);
    let mut m = Metric::new(&g, &[th, ph], &p).unwrap();
    let sc = m.ricci_scalar(&p).unwrap();
    assert!(same(&p, sc, div(&p, int(&p, 2), r2), &[th, ph]));
    assert_eq!(m.weyl(&p).unwrap_err().code(), "E-TENSOR-008");

    // de Sitter, static patch: f = 1 − r²/ℓ²; R_ab = (3/ℓ²) g_ab, R = 12/ℓ².
    let x: Vec<ExprId> = ["t", "r", "theta", "phi"]
        .iter()
        .map(|s| sym(&p, s))
        .collect();
    let l = sym(&p, "l");
    let r = x[1];
    let f = sub(&p, int(&p, 1), div(&p, pw(&p, r, 2), pw(&p, l, 2)));
    let sin2 = pw(&p, p.func("sin", vec![x[2]]), 2);
    let g = diag(
        &p,
        &[
            p.mul(vec![int(&p, -1), f]),
            pw(&p, f, -1),
            pw(&p, r, 2),
            p.mul(vec![pw(&p, r, 2), sin2]),
        ],
    );
    let mut m = Metric::new(&g, &x, &p).unwrap();
    let sc = m.ricci_scalar(&p).unwrap();
    assert!(same(&p, sc, div(&p, int(&p, 12), pw(&p, l, 2)), &x));
    let ric = m.ricci(&p).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let e = div(&p, p.mul(vec![int(&p, 3), g[i][j]]), pw(&p, l, 2));
            assert!(same(&p, ric[i][j], e, &x));
        }
    }
    // Maximally symmetric: the Weyl tensor vanishes.
    for a in m
        .weyl(&p)
        .unwrap()
        .into_iter()
        .flatten()
        .flatten()
        .flatten()
    {
        assert!(zero(&p, a, &x));
    }
}

#[test]
fn flrw_with_undefined_scale_factor() {
    // ds² = −dt² + a(t)² [dr²/(1 − k r²) + r² dΩ²]
    let p = ExprPool::new();
    let x: Vec<ExprId> = ["t", "r", "theta", "phi"]
        .iter()
        .map(|s| sym(&p, s))
        .collect();
    let (t, r) = (x[0], x[1]);
    let k = sym(&p, "k");
    let a = p.func("a", vec![t]);
    let a2 = pw(&p, a, 2);
    let one_m_kr2 = sub(&p, int(&p, 1), p.mul(vec![k, pw(&p, r, 2)]));
    let sin2 = pw(&p, p.func("sin", vec![x[2]]), 2);
    let g = diag(
        &p,
        &[
            int(&p, -1),
            div(&p, a2, one_m_kr2),
            p.mul(vec![a2, pw(&p, r, 2)]),
            p.mul(vec![a2, pw(&p, r, 2), sin2]),
        ],
    );
    let mut m = Metric::new(&g, &x, &p).unwrap();
    let ad = crate::diff::diff(a, t, &p).unwrap().value;
    let add = crate::diff::diff(ad, t, &p).unwrap().value;
    // R = 6 (a ä + ȧ² + k) / a²
    let exp_r = div(
        &p,
        p.mul(vec![
            int(&p, 6),
            p.add(vec![p.mul(vec![a, add]), pw(&p, ad, 2), k]),
        ]),
        a2,
    );
    let sc = m.ricci_scalar(&p).unwrap();
    assert!(same(&p, sc, exp_r, &x));
    // Friedmann: G_tt = 3 (ȧ² + k)/a²;  G_rr = −(2 a ä + ȧ² + k)/(1 − k r²)
    let ein = m.einstein(&p).unwrap();
    let exp_tt = div(
        &p,
        p.mul(vec![int(&p, 3), p.add(vec![pw(&p, ad, 2), k])]),
        a2,
    );
    assert!(same(&p, ein[0][0], exp_tt, &x));
    let exp_rr = div(
        &p,
        p.mul(vec![
            int(&p, -1),
            p.add(vec![p.mul(vec![int(&p, 2), a, add]), pw(&p, ad, 2), k]),
        ]),
        one_m_kr2,
    );
    assert!(same(&p, ein[1][1], exp_rr, &x));
    // Spatially homogeneous and isotropic: no off-diagonal Einstein components.
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                assert!(zero(&p, ein[i][j], &x));
            }
        }
    }
}

#[test]
fn kerr_is_ricci_flat() {
    // Boyer–Lindquist, Σ = r² + a² cos²θ, Δ = r² − 2Mr + a².
    let p = ExprPool::new();
    let x: Vec<ExprId> = ["t", "r", "theta", "phi"]
        .iter()
        .map(|s| sym(&p, s))
        .collect();
    let (r, th) = (x[1], x[2]);
    let m = sym(&p, "M");
    let a = sym(&p, "a");
    let cos2 = pw(&p, p.func("cos", vec![th]), 2);
    let sin2 = pw(&p, p.func("sin", vec![th]), 2);
    let sigma = p.add(vec![pw(&p, r, 2), p.mul(vec![pw(&p, a, 2), cos2])]);
    let delta = p.add(vec![
        pw(&p, r, 2),
        p.mul(vec![int(&p, -2), m, r]),
        pw(&p, a, 2),
    ]);
    let z = int(&p, 0);
    let two_mr = p.mul(vec![int(&p, 2), m, r]);
    let gtt = p.mul(vec![
        int(&p, -1),
        sub(&p, int(&p, 1), div(&p, two_mr, sigma)),
    ]);
    let gtp = p.mul(vec![
        int(&p, -1),
        div(&p, p.mul(vec![two_mr, a, sin2]), sigma),
    ]);
    let gpp = p.mul(vec![
        p.add(vec![
            pw(&p, r, 2),
            pw(&p, a, 2),
            div(&p, p.mul(vec![two_mr, pw(&p, a, 2), sin2]), sigma),
        ]),
        sin2,
    ]);
    let g = vec![
        vec![gtt, z, z, gtp],
        vec![z, div(&p, sigma, delta), z, z],
        vec![z, z, sigma, z],
        vec![gtp, z, z, gpp],
    ];
    let mut metric = Metric::new(&g, &x, &p).unwrap();
    assert!(metric.is_ricci_flat().unwrap());
}

#[test]
fn refusals_are_coded() {
    let p = ExprPool::new();
    let x = sym(&p, "x");
    let y = sym(&p, "y");
    // float entry
    let g = diag(&p, &[p.float(1.5, 53), int(&p, 1)]);
    assert_eq!(
        Metric::new(&g, &[x, y], &p).unwrap_err().code(),
        "E-TENSOR-004"
    );
    // not symmetric
    let g = vec![vec![int(&p, 1), x], vec![y, int(&p, 1)]];
    assert_eq!(
        Metric::new(&g, &[x, y], &p).unwrap_err().code(),
        "E-TENSOR-002"
    );
    // degenerate
    let g = vec![vec![x, x], vec![x, x]];
    assert_eq!(
        Metric::new(&g, &[x, y], &p).unwrap_err().code(),
        "E-TENSOR-003"
    );
    // shape and repeated coordinates
    let g = diag(&p, &[int(&p, 1), int(&p, 1)]);
    assert_eq!(
        Metric::new(&g, &[x], &p).unwrap_err().code(),
        "E-TENSOR-001"
    );
    assert_eq!(
        Metric::new(&g, &[x, x], &p).unwrap_err().code(),
        "E-TENSOR-001"
    );
    // budget trips surface the budget's own code
    let (_, mut m) = schwarzschild();
    let _guard = crate::budget::enter(crate::budget::Budget {
        max_steps: Some(3),
        ..crate::budget::Budget::new()
    });
    let err = m.kretschmann(&ExprPool::new()).unwrap_err();
    assert!(err.is_budget(), "{err}");
    assert!(err.code().starts_with("E-BUDGET"));
}

#[test]
fn undefined_function_metric_and_geodesics() {
    // 2-D: ds² = e^{2Φ(r)} dt²... use g = diag(−e^{2Φ(r)}, 1): Γ^r_tt = Φ' e^{2Φ}.
    let p = ExprPool::new();
    let t = sym(&p, "t");
    let r = sym(&p, "r");
    let phi = p.func("Phi", vec![r]);
    let e2 = p.func("exp", vec![p.mul(vec![int(&p, 2), phi])]);
    let g = diag(&p, &[p.mul(vec![int(&p, -1), e2]), int(&p, 1)]);
    let mut m = Metric::new(&g, &[t, r], &p).unwrap();
    let gam = m.christoffel(&p).unwrap();
    let dphi = crate::diff::diff(phi, r, &p).unwrap().value;
    assert!(same(&p, gam[1][0][0], p.mul(vec![dphi, e2]), &[t, r]));
    let vt = sym(&p, "vt");
    let vr = sym(&p, "vr");
    let acc = m.geodesic_accelerations(&[vt, vr], &p).unwrap();
    // ẗ = −2 Φ' ṫ ṙ
    let exp_t = p.mul(vec![int(&p, -2), dphi, vt, vr]);
    assert!(same(&p, acc[0], exp_t, &[t, r]));
}
