use super::*;
use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprPool};
use std::collections::HashMap;

fn u(s: &str) -> Unit {
    parse_unit(s).unwrap_or_else(|e| panic!("{s}: {e}"))
}

fn q(n: i64, d: i64) -> Rational {
    Rational::from((n, d))
}

fn factor(from: &str, to: &str) -> Scale {
    u(from).conversion_factor(&u(to)).unwrap()
}

#[test]
fn inch_is_exactly_127_over_5000_metre() {
    let f = factor("inch", "m");
    assert_eq!(f.ratio, q(127, 5000));
    assert_eq!(f.pi_power, 0);
    assert_eq!(factor("ft", "inch").ratio, 12);
    assert_eq!(factor("mi", "ft").ratio, 5280);
}

#[test]
fn conversion_table() {
    // (from, to, exact ratio) — hand-checked against the SI brochure / NIST SP 811.
    let cases: &[(&str, &str, Rational)] = &[
        ("mg", "g", q(1, 1000)),
        ("µg", "mg", q(1, 1000)),
        ("ug", "mg", q(1, 1000)),
        ("mcg", "ug", q(1, 1)),
        ("mL", "L", q(1, 1000)),
        ("L", "m^3", q(1, 1000)),
        ("L/h", "mL/min", q(1000, 60)),
        ("mg*h/L", "ug*min/mL", q(60, 1)),
        ("mg/kg", "ug/g", q(1, 1)),
        ("M", "mol/L", q(1, 1)),
        ("mM", "mol/m^3", q(1, 1)),
        ("h", "s", q(3600, 1)),
        ("day", "h", q(24, 1)),
        ("kWh", "J", q(3_600_000, 1)),
        ("kcal", "J", q(4184, 1)),
        ("eV", "J", table::dec("1.602176634e-19")),
        ("bar", "Pa", q(100_000, 1)),
        ("atm", "Pa", q(101_325, 1)),
        ("atm", "Torr", q(760, 1)),
        ("lbf", "N", table::dec("4.4482216152605")),
        (
            "psi",
            "Pa",
            table::dec("4.4482216152605") / table::dec("0.00064516"),
        ),
        ("mph", "m/s", q(1397, 3125)),
        ("gal", "L", table::dec("3.785411784")),
        ("hp", "W", table::dec("745.69987158227022")),
        ("kg m s^-2", "N", q(1, 1)),
        ("N*m", "J", q(1, 1)),
        ("J/s", "W", q(1, 1)),
        ("V*A", "W", q(1, 1)),
        ("ohm", "V/A", q(1, 1)),
        ("kΩ", "Ω", q(1000, 1)),
        ("Hz", "1/s", q(1, 1)),
        ("degR", "K", q(5, 9)),
        ("delta_degF", "delta_degC", q(5, 9)),
        ("%", "1", q(1, 100)),
    ];
    for (a, b, r) in cases {
        let f = factor(a, b);
        assert_eq!(&f.ratio, r, "{a} -> {b}");
        assert_eq!(f.pi_power, 0, "{a} -> {b}");
    }
}

#[test]
fn degree_is_pi_over_180_radian() {
    let f = factor("deg", "rad");
    assert_eq!(f.ratio, q(1, 180));
    assert_eq!(f.pi_power, 1);
    assert!(u("rad").is_pure_number());
    assert!(!u("deg").is_pure_number());
}

#[test]
fn parse_forms() {
    assert_eq!(u("m/s^2"), u("m*s**-2"));
    assert_eq!(u("m/s^2"), u("m s⁻²"));
    assert_eq!(u("kg*m^2/(s^2*A)"), u("V*s"));
    assert_eq!(u("m^(1/2)").dimension().exponent(0), &q(1, 2));
    assert_eq!(u("mg*h/L").name(), "mg*h/L");
    assert_eq!(u("1/h").name(), "1/h");
    assert_eq!(u("kg m s^-2").name(), "kg*m/s^2");
    assert_eq!(u("1000*m"), u("km"));
    assert_eq!(u("1"), Unit::dimensionless());
    for bad in [
        "", "m2", "foo", "m^", "m^1/2", "(m", "m)", "3m", "0*m", "m**",
    ] {
        let e = parse_unit(bad).unwrap_err();
        assert_eq!(e.code(), "E-UNIT-002", "{bad}: {e}");
    }
}

#[test]
fn semantic_equality_ignores_names() {
    assert_eq!(u("N"), u("kg*m/s^2"));
    assert_eq!(u("J").to_base().name(), "kg*m^2/s^2");
    assert_ne!(u("mg"), u("g"));
}

#[test]
fn affine_temperature_only_converts() {
    let p = ExprPool::new();
    let c = u("degC");
    assert!(c.is_affine());
    for bad in ["degC*m", "1/degC", "degC^2", "°F/s"] {
        assert_eq!(parse_unit(bad).unwrap_err().code(), "E-UNIT-003", "{bad}");
    }
    assert_eq!(c.mul(&u("1")).unwrap_err().code(), "E-UNIT-003");
    // 100 °C = 212 °F = 373.15 K, exactly.
    let x = Quantity::new(p.integer(100), c.clone());
    let f = x.convert(&u("degF"), &p).unwrap();
    assert_eq!(crate::simplify::simplify(f.value, &p).value, p.integer(212));
    let k = x.convert(&u("K"), &p).unwrap();
    assert_eq!(
        crate::simplify::simplify(k.value, &p).value,
        p.rational(7463, 20)
    );
    // -40 °F = -40 °C
    let m40 = Quantity::new(p.integer(-40), u("degF"));
    let back = m40.convert(&c, &p).unwrap();
    assert_eq!(
        crate::simplify::simplify(back.value, &p).value,
        p.integer(-40)
    );
    // sums of affine quantities are refused
    let y = Quantity::new(p.integer(1), u("degC"));
    assert_eq!(x.add(&y, &p).unwrap_err().code(), "E-UNIT-003");
}

#[test]
fn incompatible_add_names_both_dimensions() {
    let p = ExprPool::new();
    let a = Quantity::new(p.integer(1), u("mg"));
    let b = Quantity::new(p.integer(1), u("L"));
    let e = a.add(&b, &p).unwrap_err();
    assert_eq!(e.code(), "E-UNIT-001");
    let msg = e.to_string();
    assert!(msg.contains("mass") && msg.contains("length^3"), "{msg}");
    assert_eq!(
        u("m").conversion_factor(&u("s")).unwrap_err().code(),
        "E-UNIT-001"
    );
}

#[test]
fn quantity_add_converts_exactly() {
    let p = ExprPool::new();
    let a = Quantity::new(p.integer(1), u("m"));
    let b = Quantity::new(p.integer(1), u("inch"));
    let s = a.add(&b, &p).unwrap();
    assert_eq!(s.unit.name(), "m");
    assert_eq!(
        crate::simplify::simplify(s.value, &p).value,
        p.rational(5127, 5000)
    );
}

#[test]
fn rational_powers_need_exact_roots() {
    assert_eq!(u("cm^2").pow(&q(1, 2)).unwrap(), u("cm"));
    assert_eq!(u("km").pow(&q(1, 2)).unwrap_err().code(), "E-UNIT-007");
    assert_eq!(u("m").pow(&q(1, 2)).unwrap().name(), "m^(1/2)");
}

#[test]
fn simplify_merges_same_dimension_factors() {
    let (f, s) = u("mg/kg").mul(&u("g")).unwrap().simplify();
    assert_eq!(s.name(), "mg");
    assert_eq!(f.ratio, q(1, 1000));
    let (f, s) = u("km/m").simplify();
    assert_eq!(s.name(), "1");
    assert_eq!(f.ratio, 1000);
    let (f, s) = u("L/h*min").simplify();
    assert_eq!(s.name(), "L");
    assert_eq!(f.ratio, q(1, 60));
    let (f, s) = u("Hz*s").simplify();
    assert!(s.is_pure_number());
    assert!(f.is_one());
}

#[test]
fn mechanics_force_and_energy() {
    let p = ExprPool::new();
    let m = Quantity::new(p.integer(2), u("kg"));
    let a = Quantity::new(p.integer(3), u("m/s^2"));
    let f = m.mul(&a, &p).unwrap();
    assert_eq!(f.unit, u("N"));
    let d = Quantity::new(p.integer(5), u("m"));
    let w = f.mul(&d, &p).unwrap();
    assert_eq!(w.unit, u("J"));
    let v = Quantity::new(p.integer(4), u("m/s"));
    let half = Quantity::new(p.rational(1, 2), Unit::dimensionless());
    let ke = half
        .mul(&m, &p)
        .unwrap()
        .mul(&v.pow(&q(2, 1), &p).unwrap(), &p)
        .unwrap();
    assert_eq!(ke.unit, u("J"));
    let total = w.add(&ke, &p).unwrap();
    assert_eq!(
        crate::simplify::simplify(total.value, &p).value,
        p.integer(46)
    );
}

fn bateman(p: &ExprPool) -> (ExprId, HashMap<ExprId, Unit>) {
    let s = |n: &str| p.symbol(n, Domain::Real);
    let (f_, d, ka, ke, v, t) = (s("F"), s("D"), s("ka"), s("ke"), s("V"), s("t"));
    let neg = |x| p.mul(vec![p.integer(-1), x]);
    let diff = p.add(vec![ka, neg(ke)]);
    let pre = p.mul(vec![f_, d, ka, p.pow(p.mul(vec![v, diff]), p.integer(-1))]);
    let e1 = p.func("exp", vec![neg(p.mul(vec![ke, t]))]);
    let e2 = p.func("exp", vec![neg(p.mul(vec![ka, t]))]);
    let c = p.mul(vec![pre, p.add(vec![e1, neg(e2)])]);
    let mut m = HashMap::new();
    m.insert(f_, u("1"));
    m.insert(d, u("mg"));
    m.insert(ka, u("1/h"));
    m.insert(ke, u("1/h"));
    m.insert(v, u("L"));
    m.insert(t, u("h"));
    (c, m)
}

#[test]
fn bateman_has_concentration_units() {
    let p = ExprPool::new();
    let (c, m) = bateman(&p);
    let unit = dimension_of(c, &m, &p).unwrap();
    assert_eq!(unit, u("mg/L"));
    assert_eq!(unit.dimension().to_string(), "mass/length^3");
}

#[test]
fn bateman_with_minutes_is_refused() {
    let p = ExprPool::new();
    let (c, mut m) = bateman(&p);
    m.insert(p.symbol("t", Domain::Real), u("min"));
    let e = dimension_of(c, &m, &p).unwrap_err();
    assert_eq!(e.code(), "E-UNIT-004", "{e}");
}

#[test]
fn dimension_of_refusals() {
    let p = ExprPool::new();
    let x = p.symbol("x", Domain::Real);
    let t = p.symbol("t", Domain::Real);
    let y = p.symbol("y", Domain::Real);
    let mut m = HashMap::new();
    m.insert(x, u("m"));
    m.insert(t, u("s"));
    // x + t: E-UNIT-001
    let e = dimension_of(p.add(vec![x, t]), &m, &p).unwrap_err();
    assert_eq!(e.code(), "E-UNIT-001");
    // sin(x): E-UNIT-004
    let e = dimension_of(p.func("sin", vec![x]), &m, &p).unwrap_err();
    assert_eq!(e.code(), "E-UNIT-004");
    // y unknown: E-UNIT-005
    let e = dimension_of(p.mul(vec![x, y]), &m, &p).unwrap_err();
    assert_eq!(e.code(), "E-UNIT-005");
    // x^y with x dimensional: E-UNIT-004
    m.insert(y, u("1"));
    let e = dimension_of(p.pow(x, y), &m, &p).unwrap_err();
    assert_eq!(e.code(), "E-UNIT-004");
    // x + y·km: same dimension, different units: E-UNIT-006
    let k = p.symbol("k", Domain::Real);
    m.insert(k, u("km"));
    let e = dimension_of(p.add(vec![x, k]), &m, &p).unwrap_err();
    assert_eq!(e.code(), "E-UNIT-006");
    // sqrt(x^2 + x^2) and x/t*t are fine; 0 + x is fine.
    let r = p.func(
        "sqrt",
        vec![p.add(vec![p.pow(x, p.integer(2)), p.pow(x, p.integer(2))])],
    );
    assert_eq!(dimension_of(r, &m, &p).unwrap(), u("m"));
    assert_eq!(
        dimension_of(p.add(vec![p.integer(0), x]), &m, &p).unwrap(),
        u("m")
    );
    // sin(theta) with theta in deg needs a conversion to rad first.
    let th = p.symbol("theta", Domain::Real);
    m.insert(th, u("deg"));
    let e = dimension_of(p.func("sin", vec![th]), &m, &p).unwrap_err();
    assert_eq!(e.code(), "E-UNIT-004");
    m.insert(th, u("rad"));
    assert!(dimension_of(p.func("sin", vec![th]), &m, &p)
        .unwrap()
        .is_pure_number());
    // pi is a pure number
    let pi = p.symbol("pi", Domain::Real);
    assert_eq!(dimension_of(p.mul(vec![pi, x]), &m, &p).unwrap(), u("m"));
}

#[test]
fn registry_has_every_code() {
    let codes: Vec<&str> = crate::errors::codes::REGISTRY
        .iter()
        .map(|s| s.code)
        .collect();
    for c in 1..=7 {
        assert!(codes.contains(&format!("E-UNIT-00{c}").as_str()));
    }
}

// ---------------------------------------------------------------------------
// F18 follow-ups
// ---------------------------------------------------------------------------

#[test]
fn unify_converts_hours_plus_minutes() {
    let p = ExprPool::new();
    let a = p.symbol("a", Domain::Real);
    let b = p.symbol("b", Domain::Real);
    let mut m = HashMap::new();
    m.insert(a, u("h"));
    m.insert(b, u("min"));
    let sum = p.add(vec![a, b]);
    // still refused by dimension_of
    assert_eq!(dimension_of(sum, &m, &p).unwrap_err().code(), "E-UNIT-006");
    // finest: min, a scaled by 60
    let (e, unit) = unify_units(sum, &m, &p, UnifyPolicy::Finest).unwrap();
    assert_eq!(unit, u("min"));
    assert_eq!(e, p.add(vec![p.mul(vec![p.integer(60), a]), b]));
    // coarsest: h, b scaled by 1/60
    let (e, unit) = unify_units(sum, &m, &p, UnifyPolicy::Coarsest).unwrap();
    assert_eq!(unit, u("h"));
    assert_eq!(e, p.add(vec![a, p.mul(vec![p.rational(1, 60), b])]));
    // already-uniform sums come back untouched
    m.insert(b, u("h"));
    let (e2, unit) = unify_units(sum, &m, &p, UnifyPolicy::Finest).unwrap();
    assert_eq!((e2, unit), (sum, u("h")));
}

#[test]
fn unify_nested_and_refusals() {
    let p = ExprPool::new();
    let a = p.symbol("a", Domain::Real);
    let b = p.symbol("b", Domain::Real);
    let x = p.symbol("x", Domain::Real);
    let c = p.symbol("c", Domain::Real);
    let mut m = HashMap::new();
    m.insert(a, u("h"));
    m.insert(b, u("min"));
    m.insert(x, u("m"));
    m.insert(c, u("degC"));
    // x * (a + b): the inner sum is converted, the product has unit m*min
    let e = p.mul(vec![x, p.add(vec![a, b])]);
    let (_, unit) = unify_units(e, &m, &p, UnifyPolicy::Finest).unwrap();
    assert_eq!(unit, u("m*min"));
    // different dimensions stay E-UNIT-001
    let bad = p.add(vec![a, x]);
    assert_eq!(
        unify_units(bad, &m, &p, UnifyPolicy::Finest)
            .unwrap_err()
            .code(),
        "E-UNIT-001"
    );
    // km^(1/2) stays E-UNIT-007
    assert_eq!(u("km").pow(&q(1, 2)).unwrap_err().code(), "E-UNIT-007");
    // an affine unit in a mixed sum is E-UNIT-003
    let k = p.symbol("k", Domain::Real);
    m.insert(k, u("degF"));
    let aff = p.add(vec![c, k]);
    assert_eq!(
        unify_units(aff, &m, &p, UnifyPolicy::Finest)
            .unwrap_err()
            .code(),
        "E-UNIT-003"
    );
    // max(a, b) converts
    let mx = p.func("max", vec![a, b]);
    let (e, unit) = unify_units(mx, &m, &p, UnifyPolicy::Finest).unwrap();
    assert_eq!(unit, u("min"));
    assert_eq!(e, p.func("max", vec![p.mul(vec![p.integer(60), a]), b]));
}

#[test]
fn derived_unit_recognition() {
    let n = |s: &str| u(s).simplify_derived(None).1.name();
    assert_eq!(n("kg*m/s^2"), "N");
    assert_eq!(n("kg*m^2/s^2"), "J");
    assert_eq!(n("kg*m^2/s^3"), "W");
    assert_eq!(n("kg/(m*s^2)"), "Pa");
    assert_eq!(n("A*s"), "C");
    assert_eq!(n("kg*m^2/(A*s^3)"), "V");
    assert_eq!(n("kg*m^2/(A^2*s^3)"), "ohm");
    assert_eq!(n("A^2*s^4/(kg*m^2)"), "F");
    assert_eq!(n("kg*m^2/(A^2*s^2)"), "H");
    assert_eq!(n("kg/(A*s^2)"), "T");
    assert_eq!(n("kg*m^2/(A*s^2)"), "Wb");
    assert_eq!(n("A^2*s^3/(kg*m^2)"), "S");
    assert_eq!(n("1/s"), "Hz");
    assert_eq!(n("N*m"), "J");
    // prefixed, exact in steps of 1000
    assert_eq!(n("g*m/s^2"), "mN");
    assert_eq!(n("kg*km/s^2"), "kN");
    // not exact / not applicable: left alone
    assert_eq!(n("kg*cm/s^2"), "kg*cm/s^2");
    assert_eq!(n("rad/s"), "rad/s");
    assert_eq!(n("m/s"), "m/s");
    assert_eq!(n("N"), "N");
    assert_eq!(n("kPa"), "kPa");
    // opt-out / allow-list
    assert_eq!(u("kg*m/s^2").to_derived(Some(&["J"])), None);
    assert!(u("kg*m/s^2").to_derived(Some(&["N", "J"])).is_some());
}

#[test]
fn dimension_of_through_derivative_nodes() {
    use crate::diff::undefined::derivative_of;
    let p = ExprPool::new();
    let t = p.symbol("t", Domain::Real);
    let x = p.symbol("x", Domain::Real);
    let mass = p.symbol("mass", Domain::Real);
    let f = p.func("f", vec![t]);
    let g = p.func("g", vec![x, t]);
    let mut m = HashMap::new();
    m.insert(t, u("s"));
    m.insert(x, u("m"));
    m.insert(f, u("m"));
    m.insert(g, u("K"));
    m.insert(mass, u("kg"));
    let d1 = derivative_of(f, &[1], &p).unwrap();
    assert_eq!(dimension_of(d1, &m, &p).unwrap(), u("m/s"));
    let d2 = derivative_of(f, &[2], &p).unwrap();
    assert_eq!(dimension_of(d2, &m, &p).unwrap(), u("m/s^2"));
    let gxt = derivative_of(g, &[1, 1], &p).unwrap();
    assert_eq!(dimension_of(gxt, &m, &p).unwrap(), u("K/(m*s)"));
    let gxx = derivative_of(g, &[2, 0], &p).unwrap();
    assert_eq!(dimension_of(gxx, &m, &p).unwrap(), u("K/m^2"));
    // m * d^2f/dt^2 is a force; adding f is not
    let force = p.mul(vec![mass, d2]);
    assert_eq!(dimension_of(force, &m, &p).unwrap(), u("N"));
    let bad = p.add(vec![d1, f]);
    assert_eq!(dimension_of(bad, &m, &p).unwrap_err().code(), "E-UNIT-001");
    // minutes in the independent variable change the scale
    m.insert(t, u("min"));
    assert_eq!(dimension_of(d1, &m, &p).unwrap(), u("m/min"));
    // f with no unit: E-UNIT-005
    m.remove(&f);
    assert_eq!(dimension_of(d1, &m, &p).unwrap_err().code(), "E-UNIT-005");
    // Integral(f, t) and the definite form
    m.insert(f, u("m"));
    m.insert(t, u("s"));
    let i = p.func("Integral", vec![f, t]);
    assert_eq!(dimension_of(i, &m, &p).unwrap(), u("m*s"));
    let tt = p.symbol("T", Domain::Real);
    m.insert(tt, u("s"));
    let i = p.func("Integral", vec![f, t, p.integer(0), tt]);
    assert_eq!(dimension_of(i, &m, &p).unwrap(), u("m*s"));
}

#[test]
fn log_scale_units_are_refused_with_a_pointer() {
    for name in ["dB", "Np", "dBm"] {
        let e = parse_unit(name).unwrap_err();
        assert_eq!(e.code(), "E-UNIT-002");
        assert!(e.to_string().contains("logarithmic"), "{e}");
    }
}
