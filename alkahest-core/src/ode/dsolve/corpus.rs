//! ODE corpus harness (test-only).
//!
//! `cargo test -p alkahest-cas ode::dsolve::corpus -- --ignored --nocapture`
//! prints one `STATUS | class | name | method` line per corpus entry.  Used to
//! measure how `dsolve` coverage moves when the integration engine changes; the
//! same file compiles against older revisions, so the two runs are comparable.

use super::*;
use crate::parse::parse;
use std::collections::HashMap;

/// One corpus entry: `(class, name, order, equation)` where the equation is
/// written as `expr = 0` in `x`, `y`, and `yp`/`ypp`/`yppp`/`ypppp`.
type Entry = (&'static str, &'static str, usize, &'static str);

#[rustfmt::skip]
pub(crate) const CORPUS: &[Entry] = &[
    // ---- separable ---------------------------------------------------------
    ("separable", "y'=y",                  1, "yp - y"),
    ("separable", "logistic",              1, "yp - y*(1 - y)"),
    ("separable", "y'=x*y",                1, "yp - x*y"),
    ("separable", "y'=x^2*y^3",            1, "yp - x^2*y^3"),
    ("separable", "y'=exp(x)*y",           1, "yp - exp(x)*y"),
    ("separable", "y'=1+y^2",              1, "yp - 1 - y^2"),
    ("separable", "y'=y/x",                1, "yp - y/x"),
    ("separable", "y'=x*exp(-x^2)*y",      1, "yp - x*exp(-x^2)*y"),
    ("separable", "y'=log(x)*y",           1, "yp - log(x)*y"),
    ("separable", "y'=y*sin(x)^2",         1, "yp - y*sin(x)^2"),
    ("separable", "y'=y/(x*log(x))",       1, "yp - y/(x*log(x))"),
    ("separable", "y'=y*x/(1+x^2)",        1, "yp - y*x/(1 + x^2)"),

    // ---- linear first order ------------------------------------------------
    ("linear1", "y'-3y=x",                 1, "yp - 3*y - x"),
    ("linear1", "y'+y=exp(x)",             1, "yp + y - exp(x)"),
    ("linear1", "y'+y/x=1",                1, "yp + y/x - 1"),
    ("linear1", "y'+2xy=x",                1, "yp + 2*x*y - x"),
    ("linear1", "y'+3y=x",                 1, "yp + 3*y - x"),
    ("linear1", "y'+y*tan(x)=sin(x)",      1, "yp + y*tan(x) - sin(x)"),
    ("linear1", "y'-y/x=x*log(x)",         1, "yp - y/x - x*log(x)"),
    ("linear1", "y'+y/x=log(x)/x",         1, "yp + y/x - log(x)/x"),
    ("linear1", "y'+y=1/x",                1, "yp + y - 1/x"),
    ("linear1", "y'+y*x/(1+x^2)=x",        1, "yp + y*x/(1 + x^2) - x"),
    ("linear1", "y'+y/(x*log(x))=1",       1, "yp + y/(x*log(x)) - 1"),
    ("linear1", "y'-2xy=exp(x^2)*sin(x)",  1, "yp - 2*x*y - exp(x^2)*sin(x)"),
    ("linear1", "y'+y=x*exp(-x)",          1, "yp + y - x*exp(-x)"),
    ("linear1", "y'+y/x=cos(x)/x",         1, "yp + y/x - cos(x)/x"),

    // ---- Bernoulli ---------------------------------------------------------
    ("bernoulli", "y'+y=y^2",              1, "yp + y - y^2"),
    ("bernoulli", "y'+y/x=y^2",            1, "yp + y/x - y^2"),
    ("bernoulli", "y'-y=x*y^2",            1, "yp - y - x*y^2"),
    ("bernoulli", "y'+y=x^2*y^3",          1, "yp + y - x^2*y^3"),
    ("bernoulli", "y'+y/x=x*y^2",          1, "yp + y/x - x*y^2"),
    ("bernoulli", "y'+y*tan(x)=y^3",       1, "yp + y*tan(x) - y^3"),

    // ---- exact -------------------------------------------------------------
    ("exact", "(2x+y)+(x+2y)y'",           1, "(2*x + y) + (x + 2*y)*yp"),
    ("exact", "sin(xy) potential",         1, "y*cos(x*y) + x*cos(x*y)*yp"),
    ("exact", "(2x*log(x)+y)+(x+2y)y'",    1, "(2*x*log(x) + y) + (x + 2*y)*yp"),
    ("exact", "(1/x+y)+(x+1/y)y'",         1, "(1/x + y) + (x + 1/y)*yp"),
    ("exact", "exp(x)sin(y)",              1, "exp(x)*sin(y) + exp(x)*cos(y)*yp"),
    ("exact", "(y/(1+x^2))+atan(x)y'",     1, "y/(1 + x^2) + atan(x)*yp"),

    // ---- homogeneous -------------------------------------------------------
    ("homog", "y'=1+y/x",                  1, "yp - 1 - y/x"),
    ("homog", "y'=(x^2+y^2)/(x*y)",        1, "yp - (x^2 + y^2)/(x*y)"),
    ("homog", "y'=(x+y)/(x-y)",            1, "yp - (x + y)/(x - y)"),
    ("homog", "y'=y/x+tan(y/x)",           1, "yp - y/x - tan(y/x)"),
    ("homog", "y'=(y^2)/(x^2)",            1, "yp - y^2/x^2"),
    ("homog", "y'=(x+3y)/(x-y)",           1, "yp - (x + 3*y)/(x - y)"),

    // ---- Clairaut ----------------------------------------------------------
    ("clairaut", "y=x y'+(y')^2",          1, "y - x*yp - yp^2"),
    ("clairaut", "y=x y'+exp(y')",         1, "y - x*yp - exp(yp)"),
    ("clairaut", "y=x y'-log(y')",         1, "y - x*yp + log(yp)"),
    ("clairaut", "y=x y'+1/y'",            1, "y - x*yp - 1/yp"),

    // ---- Riccati -----------------------------------------------------------
    ("riccati", "y'=(y-x)^2+1",            1, "yp - (y - x)^2 - 1"),
    ("riccati", "y'=y^2-2xy+x^2+1",        1, "yp - y^2 + 2*x*y - x^2 - 1"),
    ("riccati", "y'=y^2-2xy+x^2+x",        1, "yp - y^2 + 2*x*y - x^2 - x"),

    // ---- second order, constant coefficients, homogeneous ------------------
    ("cc2-hom", "y''-y=0",                 2, "ypp - y"),
    ("cc2-hom", "y''+y=0",                 2, "ypp + y"),
    ("cc2-hom", "y''-3y'+2y=0",            2, "ypp - 3*yp + 2*y"),
    ("cc2-hom", "y''+2y'+y=0",             2, "ypp + 2*yp + y"),
    ("cc2-hom", "y''+2y'+5y=0",            2, "ypp + 2*yp + 5*y"),
    ("cc2-hom", "4y''-4y'+y=0",            2, "4*ypp - 4*yp + y"),

    // ---- second order, constant coefficients, forced -----------------------
    ("cc2-poly-exp", "y''+y=x^2",          2, "ypp + y - x^2"),
    ("cc2-poly-exp", "y''-y'-2y=exp(3x)",  2, "ypp - yp - 2*y - exp(3*x)"),
    ("cc2-poly-exp", "y''+y=x*exp(x)",     2, "ypp + y - x*exp(x)"),
    ("cc2-poly-exp", "y''+y'-6y=x*e^{2x}", 2, "ypp + yp - 6*y - x*exp(2*x)"),
    ("cc2-poly-exp", "y''+y=cos(x) (res)", 2, "ypp + y - cos(x)"),
    ("cc2-poly-exp", "y''-2y'+y=e^x (res)",2, "ypp - 2*yp + y - exp(x)"),
    ("cc2-poly-exp", "y''+4y=sin(2x)(res)",2, "ypp + 4*y - sin(2*x)"),

    ("cc2-vop", "y''+y=sec(x)",            2, "ypp + y - 1/cos(x)"),
    ("cc2-vop", "y''+y=tan(x)",            2, "ypp + y - tan(x)"),
    ("cc2-vop", "y''-y=1/x",               2, "ypp - y - 1/x"),
    ("cc2-vop", "y''-2y'+y=e^x/x",         2, "ypp - 2*yp + y - exp(x)/x"),
    ("cc2-vop", "y''+y=log(x)",            2, "ypp + y - log(x)"),
    ("cc2-vop", "y''-y=exp(x)/(1+exp(x))", 2, "ypp - y - exp(x)/(1 + exp(x))"),
    ("cc2-vop", "y''+y=1/x^2",             2, "ypp + y - 1/x^2"),
    ("cc2-vop", "y''-3y'+2y=1/(1+e^-x)",   2, "ypp - 3*yp + 2*y - 1/(1 + exp(-x))"),
    ("cc2-vop", "y''+y=x/(1+x^2)",         2, "ypp + y - x/(1 + x^2)"),
    ("cc2-vop", "y''-y=1/(1+exp(x))",      2, "ypp - y - 1/(1 + exp(x))"),
    ("cc2-vop", "y''+y=csc(x)",            2, "ypp + y - 1/sin(x)"),
    ("cc2-vop", "y''-y=exp(x)/x",          2, "ypp - y - exp(x)/x"),
    ("cc2-vop", "y''+y=sec(x)*tan(x)",     2, "ypp + y - tan(x)/cos(x)"),
    ("cc2-vop", "y''-4y=1/x",              2, "ypp - 4*y - 1/x"),
    ("cc2-vop", "y''+y=1/(1+sin(x))",      2, "ypp + y - 1/(1 + sin(x))"),
    ("cc2-vop", "y''-y=x/(x^2-1)",         2, "ypp - y - x/(x^2 - 1)"),

    // ---- Euler–Cauchy ------------------------------------------------------
    ("euler", "x^2y''+xy'-y=0",            2, "x^2*ypp + x*yp - y"),
    ("euler", "x^2y''-3xy'+4y=0",          2, "x^2*ypp - 3*x*yp + 4*y"),
    ("euler", "x^2y''+xy'+y=0",            2, "x^2*ypp + x*yp + y"),
    ("euler-nh", "x^2y''+xy'-y=x^2",       2, "x^2*ypp + x*yp - y - x^2"),
    ("euler-nh", "x^2y''-2xy'+2y=x^3",     2, "x^2*ypp - 2*x*yp + 2*y - x^3"),
    ("euler-nh", "x^2y''+xy'-y=log(x)",    2, "x^2*ypp + x*yp - y - log(x)"),
    ("euler-nh", "x^2y''-xy'+y=x",         2, "x^2*ypp - x*yp + y - x"),

    // ---- variable coefficient (reduction of order territory) ---------------
    ("varcoef", "y''-(2/x)y'+(2/x^2)y=0",  2, "ypp - (2/x)*yp + (2/x^2)*y"),
    ("varcoef", "(1-x^2)y''-2xy'+2y=0",    2, "(1 - x^2)*ypp - 2*x*yp + 2*y"),
    ("varcoef", "x y''-(x+1)y'+y=0",       2, "x*ypp - (x + 1)*yp + y"),
    ("varcoef", "y''-y'/x=x",              2, "ypp - yp/x - x"),
    ("varcoef", "x^2y''+xy'-y=x",          2, "x^2*ypp + x*yp - y - x"),

    // ---- higher order ------------------------------------------------------
    ("higher", "y'''-y'=0",                3, "yppp - yp"),
    ("higher", "y'''-6y''+11y'-6y=0",      3, "yppp - 6*ypp + 11*yp - 6*y"),
    ("higher", "y''''-y=0",                4, "ypppp - y"),
    ("higher", "y'''-3y''+3y'-y=0",        3, "yppp - 3*ypp + 3*yp - y"),
    ("higher", "y'''-y'=exp(x)",           3, "yppp - yp - exp(x)"),
    ("higher", "y'''-y'=1/x",              3, "yppp - yp - 1/x"),
    ("higher", "y'''+y'=sec(x)",            3, "yppp + yp - 1/cos(x)"),
    ("higher", "y'''+y'=tan(x)",            3, "yppp + yp - tan(x)"),
    ("higher", "y'''-y'=x^2",               3, "yppp - yp - x^2"),
];

/// Build the `OdeInput` for a corpus entry.  Derivative symbols are named
/// `y'`, `y''`, … (as `OdeInput` does) but bound in the parser under the ASCII
/// aliases `yp`, `ypp`, ….
fn build(entry: &Entry, pool: &ExprPool) -> Option<OdeInput> {
    let (_, _, order, src) = *entry;
    build_ode(order, src, pool)
}

/// Parse `src` (an expression in `x`, `y`, `yp`, `ypp`, …, read as `= 0`) into
/// an [`OdeInput`] of the given order.  Shared with the unit tests, which would
/// otherwise spell every equation out in `pool.add`/`pool.mul` calls.
pub(crate) fn build_ode(order: usize, src: &str, pool: &ExprPool) -> Option<OdeInput> {
    let x = pool.symbol("x", Domain::Real);
    let y = pool.symbol("y", Domain::Real);
    let (input, derivs) = OdeInput::higher_order(x, y, order, pool);
    let mut syms: HashMap<String, ExprId> = HashMap::new();
    syms.insert("x".to_owned(), x);
    syms.insert("y".to_owned(), y);
    for (k, &d) in derivs.iter().enumerate() {
        syms.insert(format!("y{}", "p".repeat(k + 1)), d);
    }
    let eq = parse(src, pool, &mut syms).ok()?;
    Some(input.with_equation(eq))
}

/// One-word outcome plus the method label for a corpus entry.
fn outcome(entry: &Entry) -> (String, String) {
    let pool = ExprPool::new();
    let Some(input) = build(entry, &pool) else {
        return ("PARSE_ERR".to_owned(), String::new());
    };
    match dsolve(&input, &pool) {
        Ok(res) => match res.solutions.first() {
            Some(s) => {
                // Independently re-verify (never trust the internal gate alone).
                let verdict = match residual_is_zero(&input, s.y_of_x, &s.constants, &pool) {
                    Ok(()) => "SOLVED",
                    Err(_) => "UNVERIFIED",
                };
                (verdict.to_owned(), s.method.to_owned())
            }
            None => ("EMPTY".to_owned(), String::new()),
        },
        Err(DsolveError::VerificationFailed(_)) => ("VERIFY_FAIL".to_owned(), String::new()),
        Err(_) => ("DECLINED".to_owned(), String::new()),
    }
}

/// Split the corpus's declines into the two reasons they can have.
///
/// `DECLINED` conflates them: a class may never have matched (nothing to
/// verify), or a class may have produced a candidate that the substitution gate
/// then refused.  Only the second kind is a *gate* problem — an answer the
/// solver had and threw away — so the ratio says whether work belongs in
/// `verify.rs` or in the solving classes and the integration engine.
#[test]
#[ignore = "measurement harness; run explicitly with --nocapture"]
fn decline_split_report() {
    let (mut no_candidate, mut gate_refused) = (0usize, 0usize);
    for e in CORPUS {
        let pool = ExprPool::new();
        let Some(input) = build(e, &pool) else {
            continue;
        };
        let _ = super::verify::take_gate_tally();
        let solved = dsolve(&input, &pool).is_ok();
        let (offered, refused) = super::verify::take_gate_tally();
        if solved {
            continue;
        }
        // `offered == refused` and `offered > 0` means every candidate any class
        // produced was rejected by the gate.
        if refused > 0 {
            gate_refused += 1;
            println!(
                "GATE_REFUSED\t{}\t{}\t{offered} offered, {refused} refused",
                e.0, e.1
            );
        } else {
            no_candidate += 1;
            println!("NO_CANDIDATE\t{}\t{}", e.0, e.1);
        }
    }
    println!("SPLIT\tno_candidate={no_candidate}\tgate_refused={gate_refused}");
}

#[test]
#[ignore = "measurement harness; run explicitly with --nocapture"]
fn corpus_report() {
    let mut solved = 0usize;
    for e in CORPUS {
        eprintln!("=== {}\t{}", e.0, e.1);
        let (status, method) = outcome(e);
        eprintln!("--- {status}\t{}\t{}\t{method}", e.0, e.1);
        if status == "SOLVED" {
            solved += 1;
        }
        println!("{status}\t{}\t{}\t{method}", e.0, e.1);
    }
    println!("TOTAL\t{solved}/{}", CORPUS.len());
}

/// The integrals `dsolve` needs and `integrate` declines, with the pairs that
/// differ only by spelling stated side by side.  Reported, not asserted — this
/// is a probe for the integration engine, and it must not fail the suite when
/// that engine improves.
#[test]
#[ignore = "integrator feedback probe; run explicitly with --nocapture"]
fn integrator_gap_probe() {
    let pool = ExprPool::new();
    let x = pool.symbol("x", Domain::Real);
    let mut syms: HashMap<String, ExprId> = HashMap::new();
    syms.insert("x".to_owned(), x);
    for src in [
        // Needs Ei — five corpus ODEs turn on this one integrand shape.
        "exp(x)/x",
        "exp(-1*x)/x",
        "exp(2*x)/x",
        // Needs Si/Ci.
        "sin(x)*log(x)",
        "sin(x)/x^2",
        // Elementary (u = e^x), currently declined.
        "exp(-2*x)/(1 + exp(-1*x))",
        "exp(2*x)/(1 + exp(x))",
        // Form sensitivity: these pairs are the same function.
        "sin(x)*tan(x)",
        "sin(x)*tan(x)/(cos(x)^2 + sin(x)^2)",
        "exp(x)^2/(1 + exp(x))",
    ] {
        let e = parse(src, &pool, &mut syms).expect("probe source parses");
        let verdict = match crate::integrate::engine::integrate(e, x, &pool) {
            Ok(d) => format!("CLOSES  {}", pool.display(d.value)),
            Err(err) => format!("DECLINES  {err}"),
        };
        println!("∫ {src} dx\t{verdict}");
    }
}
