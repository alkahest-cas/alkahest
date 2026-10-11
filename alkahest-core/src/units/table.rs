//! The unit table: SI base and derived units, SI prefixes, and the common
//! pharmacometric / engineering units. Every scale is an exact rational (times
//! a power of π for angles); decimal definitions are exact by convention
//! (`1 inch = 0.0254 m`, `1 lb = 0.45359237 kg`, `1 eV = 1.602176634e-19 J`,
//! `g_n = 9.80665 m/s²`, `1 cal = 4.184 J` thermochemical).

use super::{Dimension, Scale, Unit};
use rug::Rational;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Dimension exponents in the order length, mass, time, current, temperature,
/// amount, luminous intensity.
type Dim = [i32; 7];

const NONE: Dim = [0, 0, 0, 0, 0, 0, 0];
const LEN: Dim = [1, 0, 0, 0, 0, 0, 0];
const MASS: Dim = [0, 1, 0, 0, 0, 0, 0];
const TIME: Dim = [0, 0, 1, 0, 0, 0, 0];
const CURRENT: Dim = [0, 0, 0, 1, 0, 0, 0];
const TEMP: Dim = [0, 0, 0, 0, 1, 0, 0];
const AMOUNT: Dim = [0, 0, 0, 0, 0, 1, 0];
const LUM: Dim = [0, 0, 0, 0, 0, 0, 1];
const FREQ: Dim = [0, 0, -1, 0, 0, 0, 0];
const FORCE: Dim = [1, 1, -2, 0, 0, 0, 0];
const PRESSURE: Dim = [-1, 1, -2, 0, 0, 0, 0];
const ENERGY: Dim = [2, 1, -2, 0, 0, 0, 0];
const POWER: Dim = [2, 1, -3, 0, 0, 0, 0];
const CHARGE: Dim = [0, 0, 1, 1, 0, 0, 0];
const VOLTAGE: Dim = [2, 1, -3, -1, 0, 0, 0];
const RESISTANCE: Dim = [2, 1, -3, -2, 0, 0, 0];
const CAPACITANCE: Dim = [-2, -1, 4, 2, 0, 0, 0];
const INDUCTANCE: Dim = [2, 1, -2, -2, 0, 0, 0];
const FLUX_DENSITY: Dim = [0, 1, -2, -1, 0, 0, 0];
const FLUX: Dim = [2, 1, -2, -1, 0, 0, 0];
const CONDUCTANCE: Dim = [-2, -1, 3, 2, 0, 0, 0];
const VOLUME: Dim = [3, 0, 0, 0, 0, 0, 0];
const MOLARITY: Dim = [-3, 0, 0, 0, 0, 1, 0];
const SPEED: Dim = [1, 0, -1, 0, 0, 0, 0];

/// One table entry.
#[derive(Clone)]
pub(super) struct UnitDef {
    pub ratio: Rational,
    pub pi_power: i32,
    pub dim: Dim,
    pub offset: Option<Rational>,
    /// Accepts SI prefixes (`mg`, `kPa`, `µL`, `kWh`).
    pub prefixable: bool,
}

/// An exact rational from a decimal literal such as `"1.602176634e-19"`.
pub(super) fn dec(s: &str) -> Rational {
    try_dec(s).expect("unit table decimal literal")
}

/// [`dec`] for untrusted input: `None` when `s` is not a plain decimal.
pub(super) fn try_dec(s: &str) -> Option<Rational> {
    let (mant, exp) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], s[i + 1..].parse::<i32>().ok()?),
        None => (s, 0),
    };
    if exp.unsigned_abs() > 4096 {
        return None;
    }
    let (int_part, frac_part) = match mant.find('.') {
        Some(i) => (&mant[..i], &mant[i + 1..]),
        None => (mant, ""),
    };
    if !int_part
        .chars()
        .chain(frac_part.chars())
        .all(|c| c.is_ascii_digit())
        || (int_part.is_empty() && frac_part.is_empty())
    {
        return None;
    }
    let digits = format!("{int_part}{frac_part}");
    let n: rug::Integer = digits.parse().ok()?;
    let e10 = exp - frac_part.len() as i32;
    let p10 = |e: i32| rug::Integer::from(rug::Integer::u_pow_u(10, e.unsigned_abs()));
    Some(if e10 >= 0 {
        Rational::from(n * p10(e10))
    } else {
        Rational::from((n, p10(e10)))
    })
}

fn r(n: i64, d: i64) -> Rational {
    Rational::from((n, d))
}

/// SI prefixes, longest first so `da` wins over `d`.
pub(super) const PREFIXES: &[(&str, i32)] = &[
    ("da", 1),
    ("Y", 24),
    ("Z", 21),
    ("E", 18),
    ("P", 15),
    ("T", 12),
    ("G", 9),
    ("M", 6),
    ("k", 3),
    ("h", 2),
    ("d", -1),
    ("c", -2),
    ("m", -3),
    ("µ", -6), // U+00B5 MICRO SIGN
    ("μ", -6), // U+03BC GREEK SMALL LETTER MU
    ("u", -6),
    ("n", -9),
    ("p", -12),
    ("f", -15),
    ("a", -18),
    ("z", -21),
    ("y", -24),
];

fn table() -> &'static HashMap<&'static str, UnitDef> {
    static T: OnceLock<HashMap<&'static str, UnitDef>> = OnceLock::new();
    T.get_or_init(build)
}

fn build() -> HashMap<&'static str, UnitDef> {
    let mut t: HashMap<&'static str, UnitDef> = HashMap::new();
    let mut put = |names: &[&'static str], ratio: Rational, pi: i32, dim: Dim, prefixable: bool| {
        for n in names {
            t.insert(
                n,
                UnitDef {
                    ratio: ratio.clone(),
                    pi_power: pi,
                    dim,
                    offset: None,
                    prefixable,
                },
            );
        }
    };
    let one = || Rational::from(1);

    // SI base units (the kilogram is k + g).
    put(&["m"], one(), 0, LEN, true);
    put(&["g"], r(1, 1000), 0, MASS, true);
    put(&["s"], one(), 0, TIME, true);
    put(&["A"], one(), 0, CURRENT, true);
    put(&["K"], one(), 0, TEMP, true);
    put(&["mol"], one(), 0, AMOUNT, true);
    put(&["cd"], one(), 0, LUM, true);

    // SI derived units with special names.
    put(&["Hz"], one(), 0, FREQ, true);
    put(&["N"], one(), 0, FORCE, true);
    put(&["Pa"], one(), 0, PRESSURE, true);
    put(&["J"], one(), 0, ENERGY, true);
    put(&["W"], one(), 0, POWER, true);
    put(&["C"], one(), 0, CHARGE, true);
    put(&["V"], one(), 0, VOLTAGE, true);
    put(&["ohm", "Ω", "Ω"], one(), 0, RESISTANCE, true); // U+03A9, U+2126
    put(&["F"], one(), 0, CAPACITANCE, true);
    put(&["H"], one(), 0, INDUCTANCE, true);
    put(&["T"], one(), 0, FLUX_DENSITY, true);
    put(&["Wb"], one(), 0, FLUX, true);
    put(&["S"], one(), 0, CONDUCTANCE, true);
    put(&["rad", "sr"], one(), 0, NONE, false);

    // Accepted non-SI units.
    put(&["L", "l"], r(1, 1000), 0, VOLUME, true);
    put(&["M"], Rational::from(1000), 0, MOLARITY, true); // mol/L
    put(&["min"], Rational::from(60), 0, TIME, false);
    put(&["h", "hr"], Rational::from(3600), 0, TIME, false);
    put(&["day", "d"], Rational::from(86400), 0, TIME, false);
    put(&["week"], Rational::from(604800), 0, TIME, false);
    put(&["deg", "°"], r(1, 180), 1, NONE, false);
    put(&["eV"], dec("1.602176634e-19"), 0, ENERGY, true);
    put(&["cal"], dec("4.184"), 0, ENERGY, true);
    put(&["Wh"], Rational::from(3600), 0, ENERGY, true);
    put(&["bar"], Rational::from(100000), 0, PRESSURE, true);
    put(&["angstrom", "Å"], dec("1e-10"), 0, LEN, false);
    put(&["percent", "%"], r(1, 100), 0, NONE, false);
    put(&["ppm"], dec("1e-6"), 0, NONE, false);
    put(&["mcg"], dec("1e-9"), 0, MASS, false); // microgram, pharmacy spelling
    put(&["cc"], dec("1e-6"), 0, VOLUME, false); // cm^3

    // Imperial / US customary and engineering staples (exact by definition).
    let inch = dec("0.0254");
    let lb = dec("0.45359237");
    let gn = dec("9.80665");
    let lbf = Rational::from(&lb * &gn);
    put(&["inch", "in"], inch.clone(), 0, LEN, false);
    put(&["ft"], dec("0.3048"), 0, LEN, false);
    put(&["yd"], dec("0.9144"), 0, LEN, false);
    put(&["mi", "mile"], dec("1609.344"), 0, LEN, false);
    put(&["lb"], lb.clone(), 0, MASS, false);
    put(&["oz"], Rational::from(&lb / 16), 0, MASS, false);
    put(&["lbf"], lbf.clone(), 0, FORCE, false);
    put(
        &["psi"],
        Rational::from(&lbf / Rational::from(&inch * &inch)),
        0,
        PRESSURE,
        false,
    );
    put(&["atm"], Rational::from(101325), 0, PRESSURE, false);
    put(&["Torr", "torr"], r(101325, 760), 0, PRESSURE, false);
    put(&["mmHg"], dec("133.322387415"), 0, PRESSURE, false);
    put(&["mph"], r(1609344, 3600000), 0, SPEED, false);
    put(
        &["gal"],
        Rational::from(Rational::from(&inch * &inch) * &inch) * 231,
        0,
        VOLUME,
        false,
    );
    // Mechanical horsepower: 550 ft·lbf/s.
    put(
        &["hp"],
        Rational::from(&lbf * dec("0.3048")) * 550,
        0,
        POWER,
        false,
    );
    put(&["g0", "gn"], gn, 0, [1, 0, -2, 0, 0, 0, 0], false);

    // Temperature: Rankine and the two difference units are multiplicative.
    put(&["degR", "°R"], r(5, 9), 0, TEMP, false);
    put(&["delta_degC"], one(), 0, TEMP, false);
    put(&["delta_degF"], r(5, 9), 0, TEMP, false);
    // °C and °F are affine: K = x + 273.15, K = (x + 459.67)·5/9.
    for (names, ratio, offset) in [
        (&["degC", "°C", "celsius"][..], one(), dec("273.15")),
        (
            &["degF", "°F", "fahrenheit"][..],
            r(5, 9),
            Rational::from(dec("459.67") * r(5, 9)),
        ),
    ] {
        for n in names {
            t.insert(
                n,
                UnitDef {
                    ratio: ratio.clone(),
                    pi_power: 0,
                    dim: TEMP,
                    offset: Some(offset.clone()),
                    prefixable: false,
                },
            );
        }
    }
    t
}

fn def_to_unit(name: &str, d: &UnitDef, prefix_exp: i32) -> Unit {
    let mut ratio = d.ratio.clone();
    if prefix_exp != 0 {
        ratio *= dec(&format!("1e{prefix_exp}"));
    }
    Unit::atom(
        name,
        Scale {
            ratio,
            pi_power: Rational::from(d.pi_power),
        },
        Dimension::from_ints(d.dim),
        d.offset.clone(),
    )
}

/// Look up a single unit name: an exact table entry first, then an SI prefix
/// followed by a prefixable unit (`mg`, `kWh`, `µL`, `mM`).
pub(super) fn lookup(name: &str) -> Option<Unit> {
    let t = table();
    if let Some(d) = t.get(name) {
        return Some(def_to_unit(name, d, 0));
    }
    for (p, e) in PREFIXES {
        if let Some(rest) = name.strip_prefix(p) {
            if let Some(d) = t.get(rest) {
                if d.prefixable {
                    return Some(def_to_unit(name, d, *e));
                }
            }
        }
    }
    None
}

/// Every unit name in the table (without prefixes), sorted.
pub fn known_units() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = table().keys().copied().collect();
    v.sort_unstable();
    v
}
