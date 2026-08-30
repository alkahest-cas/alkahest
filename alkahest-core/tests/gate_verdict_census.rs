//! Census of verification-gate verdicts over an integration corpus.
//!
//! Not an assertion suite: it prints the verdict distribution so a change to
//! `simplify` can be scored on how often the gate reaches its strongest
//! verdict. Run with `--nocapture` to see the table.
//!
//! The single assertion is a ratchet: the gate must never *refute* an answer
//! the engine itself produced.

use alkahest_cas::integrate::gate::{
    verify, Domain as GDomain, EnclosurePolicy, GateOptions, Target, Verdict,
};
use alkahest_cas::kernel::{Domain, ExprPool};
use alkahest_cas::{integrate, parse, verify_antiderivative_exact};

/// Integrands spanning the families the engine advertises.
const CORPUS: &[&str] = &[
    // polynomial / power
    "x",
    "x^2",
    "x^3 + 2*x + 1",
    "1/x^2",
    "x^(1/2)",
    "1/x",
    "3*x^4 - x^2/2",
    "(2*x + 1)^3",
    "x/2 - x/3",
    // rational
    "1/(x + 1)",
    "1/(x^2 + 1)",
    "1/(x^2 - 1)",
    "x/(x^2 + 1)",
    "(x + 1)/(x^2 + 2*x + 5)",
    "1/(x^2 + 4)",
    "1/((x + 1)*(x + 2))",
    "x^2/(x^2 + 1)",
    "1/(x^3 + x)",
    // exponential / log
    "exp(x)",
    "exp(2*x)",
    "x*exp(x)",
    "exp(x)*sin(x)",
    "log(x)",
    "x*log(x)",
    "exp(x^2)*x",
    "1/(x*log(x))",
    "exp(-x)",
    // trig
    "sin(x)",
    "cos(x)",
    "sin(x)^2",
    "cos(x)^2",
    "sin(x)^3",
    "sin(x)*cos(x)",
    "tan(x)",
    "1/cos(x)^2",
    "sin(2*x)*cos(3*x)",
    "sin(x)^2*cos(x)^2",
    "x*sin(x)",
    "x^2*cos(x)",
    "sin(x)/(1 + cos(x))",
    // inverse trig / algebraic
    "1/(1 - x^2)^(1/2)",
    "1/(1 + x^2)",
    "x/(1 + x^2)^(1/2)",
    "(1 - x^2)^(1/2)",
    "1/(x^2 + 1)^(1/2)",
    "1/(x*(x^2 - 1)^(1/2))",
    // hyperbolic
    "sinh(x)",
    "cosh(x)",
    "tanh(x)",
    "sinh(x)*cosh(x)",
    // by-parts / mixed
    "x*exp(2*x)",
    "x^2*exp(x)",
    "log(x)/x",
    "atan(x)",
    "asin(x)",
    "x*atan(x)",
    "exp(x)*cos(x)",
    "x^3*log(x)",
    // rational-coefficient shapes (the reported gap)
    "sin(x)*3/4",
    "x/2 + x/3",
    "cos(x)/8 - cos(x)/8 + 1",
    "(3*x^2)/4 + x/2",
    // sqrt shapes
    "(x^2)^(1/2)",
    "x*(x^2 + 1)^(1/2)",
    "1/(x + 1)^(1/2)",
];

fn label(v: &Verdict) -> &'static str {
    match v {
        Verdict::Proven => "Proven",
        Verdict::EnclosureVerified { .. } => "EnclosureVerified",
        Verdict::SampledOnly { .. } => "SampledOnly",
        Verdict::Failed { .. } => "Failed",
        Verdict::Declined { .. } => "Declined",
    }
}

#[test]
fn gate_verdict_census() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let mut syms = std::collections::HashMap::from([("x".to_owned(), x)]);
    let samples: Vec<f64> = vec![0.317, 0.613, 1.117, 1.733, 2.219, 2.917, 3.413];
    let opts = GateOptions {
        enclosure: EnclosurePolicy::Skip,
        ..GateOptions::default()
    };

    let mut counts = std::collections::BTreeMap::<&'static str, usize>::new();
    let mut no_integral = 0usize;
    let mut rows: Vec<String> = Vec::new();
    // Cases where the gate's `Proven` depends on its *own* term-wise negation
    // helper rather than on `simplify`.  `verify_antiderivative_exact` builds
    // the residual the naive way, as `pool.mul([-1, f])`, so an entry here
    // means `simplify` still cannot push a negation through a sum and the
    // helper in `integrate/gate.rs` is still load-bearing.
    let mut needs_local_negate: Vec<&str> = Vec::new();

    for src in CORPUS {
        let f = match parse(src, &pool, &mut syms) {
            Ok(e) => e,
            Err(e) => {
                rows.push(format!("{src}\tPARSE_ERROR\t{e:?}"));
                no_integral += 1;
                continue;
            }
        };
        let cand = match integrate(f, x, &pool) {
            Ok(c) => c.value,
            Err(_) => {
                rows.push(format!("{src}\tNoIntegral"));
                no_integral += 1;
                continue;
            }
        };
        let domain = GDomain::from_samples(samples.clone());
        let v = verify(cand, &Target::symbolic(f), x, &domain, &opts, &pool);
        *counts.entry(label(&v)).or_insert(0) += 1;
        rows.push(format!("{src}\t{}", label(&v)));
        assert!(
            !matches!(v, Verdict::Failed { .. }),
            "gate refuted the engine's own answer for {src}"
        );
        if v == Verdict::Proven && !verify_antiderivative_exact(cand, f, x, &pool) {
            needs_local_negate.push(src);
        }
    }

    println!("--- gate verdict census ({} integrands) ---", CORPUS.len());
    for r in &rows {
        println!("{r}");
    }
    println!("--- totals ---");
    for (k, n) in &counts {
        println!("{k}\t{n}");
    }
    println!("NoIntegral\t{no_integral}");
    println!("needs_local_negate\t{needs_local_negate:?}");
    assert!(
        needs_local_negate.is_empty(),
        "`simplify` failed to close a residual the gate's own `negate` helper \
         closes, for: {needs_local_negate:?} — the helper is still load-bearing"
    );
}
