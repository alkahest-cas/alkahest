//! An in-tree profiler for the Risch–Norman pipeline.
//!
//! `cargo test -p alkahest-cas --release norman::bench -- --ignored --nocapture`
//!
//! There is no `perf` on the development machines this module was written on,
//! so attribution comes from [`super::profile`] instead.  The corpus is the
//! same 103 integrands as `temp-alkahest/testing/autogen/risch_norman_three_way.py`
//! so that the Rust-side attribution and the Python-side wall clock are talking
//! about the same inputs.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::profile::{self, Phase, N_PHASES};
use super::{integrate_parallel_risch, ParallelRischOutcome};
use crate::kernel::{Domain, ExprPool};

/// The 40-case probe followed by the 63-case textbook set.
pub(super) const CORPUS: &[(&str, &str)] = &[
    ("rat_cubic", "1/(x^3+x+1)"),
    ("rat_quintic", "1/(x^5-x-1)"),
    ("rat_quartic", "1/(x^4+1)"),
    ("rat_improper", "x^3/(x^2-1)"),
    ("rat_arctan", "1/(x^2+1)"),
    ("xlogx", "x*log(x)"),
    ("log3", "log(x)^3"),
    ("sqrtx_logx", "sqrt(x)*log(x)"),
    ("expsin", "exp(x)*sin(x)"),
    ("gauss_odd", "exp(-x^2)*x^3"),
    ("atan", "atan(x)"),
    ("sin5", "sin(x)^5"),
    ("x2sqrt1mx2", "x^2*sqrt(1-x^2)"),
    ("nested_exp", "exp(x)^2*exp(exp(x))"),
    ("ell_sqrt1mx4", "sqrt(1-x^4)"),
    ("ell_inv_sqrt1mx4", "1/sqrt(1-x^4)"),
    ("ell_sqrtx3p1", "sqrt(x^3+1)"),
    ("ne_gauss", "exp(x^2)"),
    ("ne_sinc", "sin(x)/x"),
    ("ne_x_expexp", "x*exp(exp(x))"),
    ("ne_exp_log", "exp(x)*log(x)"),
    ("ne_exp_over_sqrt", "exp(x)/sqrt(x^2+1)"),
    ("fail_logistic", "exp(x)/(exp(x)+1)"),
    ("fail_logistic2", "1/(1+exp(-x))"),
    ("fail_sqrt_tan", "sqrt(tan(x))"),
    ("form_a_div", "1/(x*log(x))"),
    ("form_a_mulpow", "x^(-1)*log(x)^(-1)"),
    ("form_a_powmul", "(x*log(x))^(-1)"),
    ("form_b_div", "exp(x)/(exp(x)+1)"),
    ("form_b_mulpow", "exp(x)*(1+exp(x))^(-1)"),
    ("form_b_powmul", "(exp(-x)*(exp(x)+1))^(-1)"),
    ("weak_dilog", "log(x)/(1+x)"),
    ("weak_polylog", "x/(exp(x)+1)"),
    ("weak_loglog", "log(log(x))"),
    ("weierstrass", "1/(1+sin(x))"),
    ("half_angle2", "1/(2+cos(x))"),
    ("expx_over_x", "exp(x)/x"),
    ("erf_like", "exp(-x^2)"),
    ("fresnel_like", "sin(x^2)"),
    ("ibp_x2expx", "x^2*exp(x)"),
    ("tb_poly", "x^5-3*x^2+7"),
    ("tb_poly_neg", "x^(-3)+x^(-1)"),
    ("tb_rat_simple", "1/(x-2)"),
    ("tb_rat_sq", "1/(x+1)^2"),
    ("tb_rat_cube", "1/(x-1)^3"),
    ("tb_rat_partial", "1/((x-1)*(x+2))"),
    ("tb_rat_partial3", "1/(x*(x-1)*(x+1))"),
    ("tb_rat_improper2", "(x^2+1)/(x-1)"),
    ("tb_rat_quad_num", "(2*x+3)/(x^2+3*x+5)"),
    ("tb_rat_arctan_shift", "1/(x^2+4)"),
    ("tb_rat_mixed", "(x^3+1)/(x^2*(x+1))"),
    ("tb_exp", "exp(x)"),
    ("tb_exp2x", "exp(2*x)"),
    ("tb_exp_neg", "exp(-x)"),
    ("tb_x_exp", "x*exp(x)"),
    ("tb_x3_exp", "x^3*exp(x)"),
    ("tb_x_exp2x", "x*exp(2*x)"),
    ("tb_exp_over_exp1_sq", "exp(x)/(exp(x)+1)^2"),
    ("tb_exp2_over_exp1", "exp(2*x)/(exp(x)+1)"),
    ("tb_exp3_over_exp1", "exp(3*x)/(exp(x)+1)"),
    ("tb_one_over_exp1", "1/(exp(x)+1)"),
    ("tb_one_over_exp_m1", "1/(exp(x)-1)"),
    ("tb_sech_like", "1/(exp(x)+exp(-x))"),
    ("tb_tanh_like", "(exp(x)-exp(-x))/(exp(x)+exp(-x))"),
    ("tb_exp_nested", "exp(exp(x))*exp(x)"),
    ("tb_exp_gauss_x", "x*exp(-x^2)"),
    ("tb_exp_gauss_x5", "x^5*exp(-x^2)"),
    ("tb_exp_of_x2_times_x", "x*exp(x^2)"),
    ("tb_exp_ratio", "exp(x)/(exp(2*x)+2*exp(x)+1)"),
    ("tb_exp_poly_ratio", "(x*exp(x)+exp(x))/x"),
    ("tb_log", "log(x)"),
    ("tb_log2", "log(x)^2"),
    ("tb_x2_log", "x^2*log(x)"),
    ("tb_log_over_x", "log(x)/x"),
    ("tb_log2_over_x", "log(x)^2/x"),
    ("tb_one_over_xlog2", "1/(x*log(x)^2)"),
    ("tb_log_of_x2", "log(x^2)"),
    ("tb_log_xp1", "log(x+1)"),
    ("tb_log_over_x2", "log(x)/x^2"),
    ("tb_loglog_over_x", "log(log(x))/x"),
    ("tb_one_over_xloglog", "1/(x*log(x)*log(log(x)))"),
    ("tb_log_deriv_rat", "(2*x)/(x^2+1)"),
    ("tb_log_shifted", "1/(2*x+3)"),
    ("tb_exp_log_mixed", "exp(x)*log(exp(x)+1)"),
    ("tb_log_exp1", "exp(x)/(exp(x)+1)+1/x"),
    ("tb_mixed_ratio", "(exp(x)+x)/(x*exp(x))"),
    ("tb_x_over_logx", "x/log(x)"),
    ("tb_exp_over_logx", "exp(x)/log(x)"),
    ("tb_sin", "sin(x)"),
    ("tb_cos2", "cos(x)^2"),
    ("tb_tan", "tan(x)"),
    ("tb_sec2", "1/cos(x)^2"),
    ("tb_sqrt", "sqrt(x)"),
    ("tb_x_sqrt1px2", "x/sqrt(1+x^2)"),
    ("tb_asin_like", "1/sqrt(1-x^2)"),
    ("tb_x_cbrt", "x^(1/3)"),
    ("tb_atan_num", "x/(x^2+1)"),
    ("tb_expsin2", "exp(x)*cos(x)"),
    ("tb_ne_gauss", "exp(-x^2)"),
    ("tb_ne_li", "1/log(x)"),
    ("tb_ne_ei", "exp(x)/x"),
    ("tb_ne_x2_over_exp1", "x^2/(exp(x)+1)"),
    ("tb_ne_logx_over_x1", "log(x)/(x+1)"),
];

/// Deliberately oversized integrands, chosen to push the ansatz towards the
/// `MAX_UNKNOWNS` / `MAX_EQUATIONS` caps.
///
/// The corpus above never builds a system larger than 41 × 30, so it cannot
/// answer the question "does the linear solve dominate at scale?".  These do.
const STRESS: &[(&str, &str)] = &[
    ("s_rat_d10", "1/(x^10+1)"),
    ("s_rat_d20", "1/(x^20+1)"),
    ("s_rat_d40", "1/(x^40+1)"),
    ("s_rat_d60", "1/(x^60+1)"),
    ("s_rat_d80", "1/(x^80+1)"),
    ("s_rat_num_d40", "(x^30+7*x^11+3)/(x^40+x+1)"),
    ("s_rat_sq_d20", "1/(x^20+1)^2"),
    ("s_poly_d100", "x^100+x^37+1"),
    ("s_exp_x20", "x^20*exp(x)"),
    ("s_exp_x40", "x^40*exp(x)"),
    ("s_exp_rat", "x^10/((exp(x)+1)*(x^5+1))"),
    ("s_exp_pow6", "exp(x)/(exp(x)+1)^6"),
    ("s_log_x20", "x^20*log(x)"),
    ("s_log_pow5", "log(x)^5"),
    ("s_log_rat", "x^6*log(x)/(x^8+1)"),
    ("s_mixed", "x^4*exp(x)*log(x)/(x^3+1)"),
    ("s_mixed2", "(x^6+exp(x))/(x^4*(exp(x)+1))"),
    ("s_expexp", "x^3*exp(x)*exp(exp(x))"),
];

/// One measured case.
struct Row {
    label: &'static str,
    solved: bool,
    best: Duration,
    shape: profile::Shape,
    phases: [Duration; N_PHASES],
}

/// Run the corpus, min-of-`reps`, and print an attribution table.
#[test]
#[ignore = "profiling harness; run explicitly with --ignored --nocapture"]
fn profile_corpus() {
    run_corpus(CORPUS, 7);
}

/// Where the time goes on systems that actually approach the caps.
#[test]
#[ignore = "profiling harness; run explicitly with --ignored --nocapture"]
fn profile_stress() {
    run_corpus(STRESS, 3);
}

fn run_corpus(corpus: &[(&'static str, &'static str)], reps: usize) {
    profile::enable();
    let mut rows: Vec<Row> = Vec::new();

    for (label, src) in corpus {
        // A fresh pool per case so hash-consing from earlier cases cannot make
        // a later one look cheaper than it is.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_owned(), x)]);
        let Ok(e) = crate::parse::parse(src, &pool, &mut syms) else {
            continue;
        };
        let mut best = Duration::MAX;
        let mut best_phases = [Duration::ZERO; N_PHASES];
        let mut solved = false;
        let mut shape = profile::Shape::default();
        for _ in 0..reps {
            profile::reset();
            let t0 = Instant::now();
            let out = integrate_parallel_risch(e, x, &pool);
            let dt = t0.elapsed();
            solved = matches!(out, ParallelRischOutcome::Solved { .. });
            if dt < best {
                best = dt;
                best_phases = profile::totals();
                shape = profile::shape();
            }
        }
        rows.push(Row {
            label,
            solved,
            best,
            shape,
            phases: best_phases,
        });
    }

    rows.sort_by_key(|r| std::cmp::Reverse(r.best));
    println!("\n=== per-case (min of {reps}), slowest first ===");
    println!(
        "{:<26} {:>4} {:>9} {:>6} {:>6}  phases (µs)",
        "case", "ok", "total µs", "eqs", "unk"
    );
    for r in rows.iter().take(30) {
        let mut ph = String::new();
        for (i, p) in Phase::ALL.iter().enumerate() {
            let us = r.phases[i].as_secs_f64() * 1e6;
            if us >= 1.0 {
                ph.push_str(&format!("{}={:.0} ", p.label(), us));
            }
        }
        println!(
            "{:<26} {:>4} {:>9.0} {:>6} {:>6}  {}",
            r.label,
            if r.solved { "yes" } else { "-" },
            r.best.as_secs_f64() * 1e6,
            r.shape.equations,
            r.shape.unknowns,
            ph
        );
    }

    let solved: Vec<&Row> = rows.iter().filter(|r| r.solved).collect();
    let mut times: Vec<f64> = solved.iter().map(|r| r.best.as_secs_f64() * 1e3).collect();
    times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let median = if times.is_empty() {
        0.0
    } else {
        times[times.len() / 2]
    };
    let worst = times.last().copied().unwrap_or(0.0);
    println!(
        "\nsolved {}/{}  median {:.3} ms  worst {:.3} ms",
        solved.len(),
        rows.len(),
        median,
        worst
    );

    println!("\n=== phase totals over the whole corpus ===");
    let mut totals = [Duration::ZERO; N_PHASES];
    for r in &rows {
        for (t, p) in totals.iter_mut().zip(r.phases.iter()) {
            *t += *p;
        }
    }
    let grand: f64 = totals.iter().map(|d| d.as_secs_f64()).sum();
    for (i, p) in Phase::ALL.iter().enumerate() {
        let s = totals[i].as_secs_f64();
        println!(
            "  {:<14} {:>9.2} ms  {:>5.1}%",
            p.label(),
            s * 1e3,
            if grand > 0.0 { 100.0 * s / grand } else { 0.0 }
        );
    }
    println!("  {:<14} {:>9.2} ms", "TOTAL", grand * 1e3);

    println!("\n=== phase totals over the SOLVED cases only ===");
    let mut totals = [Duration::ZERO; N_PHASES];
    for r in &rows {
        if !r.solved {
            continue;
        }
        for (t, p) in totals.iter_mut().zip(r.phases.iter()) {
            *t += *p;
        }
    }
    let grand: f64 = totals.iter().map(|d| d.as_secs_f64()).sum();
    for (i, p) in Phase::ALL.iter().enumerate() {
        let s = totals[i].as_secs_f64();
        println!(
            "  {:<14} {:>9.2} ms  {:>5.1}%",
            p.label(),
            s * 1e3,
            if grand > 0.0 { 100.0 * s / grand } else { 0.0 }
        );
    }
    println!("  {:<14} {:>9.2} ms", "TOTAL", grand * 1e3);

    // Evidence for the caps.  Every ceiling in this module is documented
    // against the largest value actually observed, not against a guess; this
    // table is where those numbers come from.
    println!("\n=== observed maxima (what the caps have to clear) ===");
    let max = |f: fn(&Row) -> usize| -> (usize, &'static str) {
        rows.iter()
            .map(|r| (f(r), r.label))
            .max_by_key(|(v, _)| *v)
            .unwrap_or((0, "-"))
    };
    for (name, cap, got) in [
        ("unknowns", 240usize, max(|r| r.shape.unknowns)),
        ("equations", 6000, max(|r| r.shape.equations)),
        ("denom terms", 2000, max(|r| r.shape.denom_terms)),
        ("generators", 7, max(|r| r.shape.generators)),
        ("atoms", 16, max(|r| r.shape.atoms)),
        ("exponent", 64, max(|r| r.shape.max_pow as usize)),
        ("nesting depth", 256, max(|r| r.shape.depth as usize)),
        ("solver cells", 50_000, max(|r| r.shape.peak_cells)),
    ] {
        let headroom = if got.0 == 0 {
            f64::INFINITY
        } else {
            cap as f64 / got.0 as f64
        };
        println!(
            "  {:<13} max {:>6} ({:<22}) cap {:>6}  headroom {:.0}x",
            name, got.0, got.1, cap, headroom
        );
    }
}
